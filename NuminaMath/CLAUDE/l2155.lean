import Mathlib

namespace NUMINAMATH_CALUDE_length_AX_in_tangent_circles_configuration_l2155_215534

/-- Two circles with radii r₁ and r₂ that are externally tangent -/
structure ExternallyTangentCircles (r₁ r₂ : ℝ) :=
  (center_distance : ℝ)
  (tangent_point : ℝ × ℝ)
  (external_tangent_length : ℝ)
  (h_center_distance : center_distance = r₁ + r₂)

/-- The configuration of two externally tangent circles with their common tangents -/
structure TangentCirclesConfiguration (r₁ r₂ : ℝ) extends ExternallyTangentCircles r₁ r₂ :=
  (common_external_tangent_point_A : ℝ × ℝ)
  (common_external_tangent_point_B : ℝ × ℝ)
  (common_internal_tangent_intersection : ℝ × ℝ)

/-- The theorem stating the length of AX in the given configuration -/
theorem length_AX_in_tangent_circles_configuration 
  (config : TangentCirclesConfiguration 20 13) : 
  ∃ (AX : ℝ), AX = 2 * Real.sqrt 65 :=
sorry

end NUMINAMATH_CALUDE_length_AX_in_tangent_circles_configuration_l2155_215534


namespace NUMINAMATH_CALUDE_sqrt_sum_fractions_l2155_215529

theorem sqrt_sum_fractions : Real.sqrt (1/25 + 1/36) = Real.sqrt 61 / 30 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_fractions_l2155_215529


namespace NUMINAMATH_CALUDE_log_relation_l2155_215515

theorem log_relation (a b : ℝ) (ha : a = Real.log 225 / Real.log 8) (hb : b = Real.log 15 / Real.log 2) : 
  a = (2 * b) / 3 := by
  sorry

end NUMINAMATH_CALUDE_log_relation_l2155_215515


namespace NUMINAMATH_CALUDE_square_number_difference_l2155_215535

theorem square_number_difference (n k l : ℕ) :
  (∃ x : ℕ, x^2 < n ∧ n < (x+1)^2) →  -- n is between consecutive squares
  (∃ x : ℕ, n - k = x^2) →            -- n - k is a square number
  (∃ x : ℕ, n + l = x^2) →            -- n + l is a square number
  (∃ x : ℕ, n - k - l = x^2) :=        -- n - k - l is a square number
by sorry

end NUMINAMATH_CALUDE_square_number_difference_l2155_215535


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_is_two_l2155_215521

theorem sum_of_x_and_y_is_two (x y : ℝ) (h : x^2 + y^2 = 10*x - 6*y - 34) : x + y = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_is_two_l2155_215521


namespace NUMINAMATH_CALUDE_robs_baseball_cards_l2155_215552

theorem robs_baseball_cards 
  (rob_doubles : ℕ) 
  (rob_total : ℕ) 
  (jess_doubles : ℕ) 
  (h1 : rob_doubles = rob_total / 3)
  (h2 : jess_doubles = 5 * rob_doubles)
  (h3 : jess_doubles = 40) : 
  rob_total = 24 := by
sorry

end NUMINAMATH_CALUDE_robs_baseball_cards_l2155_215552


namespace NUMINAMATH_CALUDE_opposite_of_negative_2023_l2155_215558

theorem opposite_of_negative_2023 :
  (∀ x : ℤ, x + (-2023) = 0 → x = 2023) :=
by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_2023_l2155_215558


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2155_215511

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) : ℕ+ → ℤ := fun n => a₁ + d * (n - 1)

theorem arithmetic_sequence_properties
  (a : ℕ+ → ℤ)
  (h₁ : a 1 = -60)
  (h₂ : a 17 = -12) :
  let d := (a 17 - a 1) / 16
  ∃ (S T : ℕ+ → ℤ),
    (∀ n : ℕ+, a n = arithmetic_sequence (-60) d n) ∧
    (∀ n : ℕ+, n < 22 → a n ≤ 0) ∧
    (a 22 > 0) ∧
    (∀ n : ℕ+, S n = n * (a 1 + a n) / 2) ∧
    (S 20 = S 21) ∧
    (S 20 = -630) ∧
    (∀ n : ℕ+, n ≤ 21 → T n = n * (123 - 3 * n) / 2) ∧
    (∀ n : ℕ+, n ≥ 22 → T n = (3 * n^2 - 123 * n + 2520) / 2) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2155_215511


namespace NUMINAMATH_CALUDE_qualification_rate_example_l2155_215524

/-- Calculates the qualification rate given the total number of boxes and the number of qualified boxes -/
def qualification_rate (total : ℕ) (qualified : ℕ) : ℚ :=
  (qualified : ℚ) / (total : ℚ) * 100

/-- Theorem stating that given 50 total boxes and 38 qualified boxes, the qualification rate is 76% -/
theorem qualification_rate_example : qualification_rate 50 38 = 76 := by
  sorry

end NUMINAMATH_CALUDE_qualification_rate_example_l2155_215524


namespace NUMINAMATH_CALUDE_min_chord_length_proof_l2155_215570

/-- The circle equation x^2 + y^2 - 6x = 0 -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 6*x = 0

/-- The point through which the chord passes -/
def point : ℝ × ℝ := (1, 2)

/-- The minimum length of the chord intercepted by the circle passing through the point -/
def min_chord_length : ℝ := 2

theorem min_chord_length_proof :
  ∀ (x y : ℝ), circle_equation x y →
  min_chord_length = (2 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_min_chord_length_proof_l2155_215570


namespace NUMINAMATH_CALUDE_rowing_problem_l2155_215527

/-- A rowing problem in a river with current and headwind -/
theorem rowing_problem (downstream_speed current_speed headwind_reduction : ℝ) 
  (h1 : downstream_speed = 22)
  (h2 : current_speed = 4.5)
  (h3 : headwind_reduction = 1.5) :
  let still_water_speed := downstream_speed - current_speed
  still_water_speed - current_speed - headwind_reduction = 11.5 := by
  sorry

end NUMINAMATH_CALUDE_rowing_problem_l2155_215527


namespace NUMINAMATH_CALUDE_reciprocal_complement_sum_square_l2155_215510

theorem reciprocal_complement_sum_square (p q r : ℝ) (h : p * q * r = 1) :
  (1 / (1 - p))^2 + (1 / (1 - q))^2 + (1 / (1 - r))^2 ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_complement_sum_square_l2155_215510


namespace NUMINAMATH_CALUDE_sum_of_xyz_l2155_215555

theorem sum_of_xyz (p q : ℝ) (x y z : ℤ) : 
  p^2 = 25/50 →
  q^2 = (3 + Real.sqrt 7)^2 / 14 →
  p < 0 →
  q > 0 →
  (p + q)^3 = (x : ℝ) * Real.sqrt (y : ℝ) / (z : ℝ) →
  x + y + z = 177230 := by
sorry

end NUMINAMATH_CALUDE_sum_of_xyz_l2155_215555


namespace NUMINAMATH_CALUDE_cosine_inequality_solution_l2155_215561

theorem cosine_inequality_solution (x : ℝ) : 
  0 ≤ x ∧ x ≤ 2 * π →
  (2 * Real.cos x ≤ |Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x))| ∧
   |Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x))| ≤ Real.sqrt 2) ↔
  (π / 4 ≤ x ∧ x ≤ 7 * π / 4) :=
by sorry

end NUMINAMATH_CALUDE_cosine_inequality_solution_l2155_215561


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_2sqrt6_l2155_215537

theorem sqrt_sum_equals_2sqrt6 : 
  Real.sqrt (9 - 6 * Real.sqrt 2) + Real.sqrt (9 + 6 * Real.sqrt 2) = 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_2sqrt6_l2155_215537


namespace NUMINAMATH_CALUDE_total_spending_correct_l2155_215563

-- Define the stores and their purchases
structure Store :=
  (items : List (String × Float))
  (discount : Float)
  (accessoryDeal : Option (Float × Float))
  (freeItem : Option Float)
  (shippingFee : Bool)

def stores : List Store := [
  ⟨[("shoes", 200)], 0.3, none, none, false⟩,
  ⟨[("shirts", 160), ("pants", 150)], 0.2, none, none, false⟩,
  ⟨[("jacket", 250), ("tie", 40), ("hat", 60)], 0, some (0.5, 0.5), none, false⟩,
  ⟨[("watch", 120), ("wallet", 49)], 0, none, some 49, true⟩,
  ⟨[("belt", 35), ("scarf", 45)], 0, none, none, true⟩
]

-- Define the overall discount and tax rates
def rewardsDiscount : Float := 0.05
def salesTax : Float := 0.08

-- Define the gift card amount
def giftCardAmount : Float := 50

-- Define the shipping fee
def shippingFee : Float := 5

-- Function to calculate the total spending
noncomputable def calculateTotalSpending (stores : List Store) (rewardsDiscount : Float) (salesTax : Float) (giftCardAmount : Float) (shippingFee : Float) : Float :=
  sorry

-- Theorem to prove
theorem total_spending_correct :
  calculateTotalSpending stores rewardsDiscount salesTax giftCardAmount shippingFee = 854.29 :=
sorry

end NUMINAMATH_CALUDE_total_spending_correct_l2155_215563


namespace NUMINAMATH_CALUDE_max_visible_sum_is_164_l2155_215523

/-- Represents a cube with six faces --/
structure Cube :=
  (faces : Fin 6 → ℕ)

/-- The set of numbers used to form each cube --/
def cube_numbers : Finset ℕ := {1, 2, 4, 8, 16, 32}

/-- A cube is valid if it uses exactly the numbers in cube_numbers --/
def valid_cube (c : Cube) : Prop :=
  (Finset.image c.faces (Finset.univ : Finset (Fin 6))) = cube_numbers

/-- The sum of visible faces when a cube is stacked --/
def visible_sum (c : Cube) (top : Bool) : ℕ :=
  if top then
    c.faces 0 + c.faces 1 + c.faces 2 + c.faces 3 + c.faces 4
  else
    c.faces 1 + c.faces 2 + c.faces 3 + c.faces 4

/-- The theorem to be proved --/
theorem max_visible_sum_is_164 :
  ∃ (c1 c2 c3 : Cube),
    valid_cube c1 ∧ valid_cube c2 ∧ valid_cube c3 ∧
    visible_sum c1 false + visible_sum c2 false + visible_sum c3 true = 164 ∧
    ∀ (d1 d2 d3 : Cube),
      valid_cube d1 → valid_cube d2 → valid_cube d3 →
      visible_sum d1 false + visible_sum d2 false + visible_sum d3 true ≤ 164 := by
  sorry

end NUMINAMATH_CALUDE_max_visible_sum_is_164_l2155_215523


namespace NUMINAMATH_CALUDE_projection_a_on_b_l2155_215594

def a : ℝ × ℝ := (2, 3)
def b : ℝ × ℝ := (3, -4)

theorem projection_a_on_b : 
  (a.1 * b.1 + a.2 * b.2) / Real.sqrt (b.1^2 + b.2^2) = -6/5 := by sorry

end NUMINAMATH_CALUDE_projection_a_on_b_l2155_215594


namespace NUMINAMATH_CALUDE_parabola_focus_coordinates_l2155_215547

/-- Given a parabola with equation y^2 = -4x, its focus has coordinates (-1, 0) -/
theorem parabola_focus_coordinates :
  let parabola := {(x, y) : ℝ × ℝ | y^2 = -4*x}
  ∃ (f : ℝ × ℝ), f ∈ parabola ∧ f = (-1, 0) ∧ ∀ (p : ℝ × ℝ), p ∈ parabola → ‖p - f‖ = ‖p - (p.1, 0)‖ := by
  sorry


end NUMINAMATH_CALUDE_parabola_focus_coordinates_l2155_215547


namespace NUMINAMATH_CALUDE_tan_B_in_triangle_l2155_215576

theorem tan_B_in_triangle (A B C : ℝ) (cosC : ℝ) (AC BC : ℝ) 
  (h1 : cosC = 2/3)
  (h2 : AC = 4)
  (h3 : BC = 3)
  (h4 : A + B + C = Real.pi) -- sum of angles in a triangle
  (h5 : 0 < AC ∧ 0 < BC) -- positive side lengths
  : Real.tan B = 4 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_tan_B_in_triangle_l2155_215576


namespace NUMINAMATH_CALUDE_remainder_of_2_pow_33_mod_9_l2155_215573

theorem remainder_of_2_pow_33_mod_9 : 2^33 % 9 = 8 := by sorry

end NUMINAMATH_CALUDE_remainder_of_2_pow_33_mod_9_l2155_215573


namespace NUMINAMATH_CALUDE_cubic_function_property_l2155_215512

/-- A cubic function g(x) = Ax³ + Bx² - Cx + D -/
def g (A B C D : ℝ) (x : ℝ) : ℝ := A * x^3 + B * x^2 - C * x + D

theorem cubic_function_property (A B C D : ℝ) :
  g A B C D 2 = 5 ∧ g A B C D (-1) = -8 ∧ g A B C D 0 = 2 →
  -12*A + 6*B - 3*C + D = 27.5 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_property_l2155_215512


namespace NUMINAMATH_CALUDE_manuscript_cost_example_l2155_215540

/-- Calculates the total cost of typing and revising a manuscript --/
def manuscript_cost (total_pages : ℕ) (pages_revised_once : ℕ) (pages_revised_twice : ℕ) 
  (initial_cost_per_page : ℕ) (revision_cost_per_page : ℕ) : ℕ :=
  let pages_not_revised := total_pages - pages_revised_once - pages_revised_twice
  let initial_typing_cost := total_pages * initial_cost_per_page
  let first_revision_cost := pages_revised_once * revision_cost_per_page
  let second_revision_cost := pages_revised_twice * revision_cost_per_page * 2
  initial_typing_cost + first_revision_cost + second_revision_cost

theorem manuscript_cost_example : 
  manuscript_cost 100 20 30 10 5 = 1400 := by
  sorry

end NUMINAMATH_CALUDE_manuscript_cost_example_l2155_215540


namespace NUMINAMATH_CALUDE_total_gross_profit_calculation_l2155_215598

/-- Represents the sales prices and costs for an item over three months -/
structure ItemData :=
  (sales_prices : Fin 3 → ℕ)
  (costs : Fin 3 → ℕ)
  (gross_profit_percentage : ℕ)

/-- Calculates the gross profit for an item in a given month -/
def gross_profit (item : ItemData) (month : Fin 3) : ℕ :=
  item.sales_prices month - item.costs month

/-- Calculates the total gross profit for an item over three months -/
def total_gross_profit (item : ItemData) : ℕ :=
  (gross_profit item 0) + (gross_profit item 1) + (gross_profit item 2)

/-- The main theorem to prove -/
theorem total_gross_profit_calculation 
  (item_a item_b item_c item_d : ItemData)
  (ha : item_a.sales_prices = ![44, 47, 50])
  (hac : item_a.costs = ![20, 22, 25])
  (hap : item_a.gross_profit_percentage = 120)
  (hb : item_b.sales_prices = ![60, 63, 65])
  (hbc : item_b.costs = ![30, 33, 35])
  (hbp : item_b.gross_profit_percentage = 150)
  (hc : item_c.sales_prices = ![80, 83, 85])
  (hcc : item_c.costs = ![40, 42, 45])
  (hcp : item_c.gross_profit_percentage = 100)
  (hd : item_d.sales_prices = ![100, 103, 105])
  (hdc : item_d.costs = ![50, 52, 55])
  (hdp : item_d.gross_profit_percentage = 130) :
  total_gross_profit item_a + total_gross_profit item_b + 
  total_gross_profit item_c + total_gross_profit item_d = 436 := by
  sorry

end NUMINAMATH_CALUDE_total_gross_profit_calculation_l2155_215598


namespace NUMINAMATH_CALUDE_total_puff_pastries_made_l2155_215544

/-- Theorem: Calculating total puff pastries made by volunteers -/
theorem total_puff_pastries_made
  (num_volunteers : ℕ)
  (trays_per_batch : ℕ)
  (pastries_per_tray : ℕ)
  (h1 : num_volunteers = 1000)
  (h2 : trays_per_batch = 8)
  (h3 : pastries_per_tray = 25) :
  num_volunteers * trays_per_batch * pastries_per_tray = 200000 :=
by sorry

end NUMINAMATH_CALUDE_total_puff_pastries_made_l2155_215544


namespace NUMINAMATH_CALUDE_line_slope_intercept_product_l2155_215516

/-- Given a line passing through points (-2, 3) and (3, -2), 
    the product of the square of its slope and its y-intercept equals 1 -/
theorem line_slope_intercept_product : 
  ∀ (m b : ℝ), 
    (∀ x y : ℝ, y = m * x + b) →  -- Line equation
    (3 = m * (-2) + b) →          -- Point (-2, 3) satisfies the equation
    (-2 = m * 3 + b) →            -- Point (3, -2) satisfies the equation
    m^2 * b = 1 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_intercept_product_l2155_215516


namespace NUMINAMATH_CALUDE_distance_from_origin_l2155_215588

theorem distance_from_origin (z : ℂ) (h : (3 - 4*Complex.I)*z = Complex.abs (4 + 3*Complex.I)) : 
  Complex.abs z = 1 := by sorry

end NUMINAMATH_CALUDE_distance_from_origin_l2155_215588


namespace NUMINAMATH_CALUDE_peach_difference_l2155_215556

theorem peach_difference (jill steven jake : ℕ) : 
  jill = 12 →
  steven = jill + 15 →
  jake = steven - 16 →
  jill - jake = 1 := by
sorry

end NUMINAMATH_CALUDE_peach_difference_l2155_215556


namespace NUMINAMATH_CALUDE_jessica_red_marbles_l2155_215567

theorem jessica_red_marbles (sandy_marbles : ℕ) (sandy_multiple : ℕ) :
  sandy_marbles = 144 →
  sandy_multiple = 4 →
  (sandy_marbles / sandy_multiple) / 12 = 3 := by
  sorry

end NUMINAMATH_CALUDE_jessica_red_marbles_l2155_215567


namespace NUMINAMATH_CALUDE_f_properties_l2155_215522

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (-2^x + a) / (2^x + 1)

theorem f_properties (a : ℝ) :
  (∀ x, f a x = -f a (-x)) →
  (a = 1 ∧
   (∀ x y, x < y → f a y < f a x) ∧
   (∀ k, (∀ t, 1 ≤ t ∧ t ≤ 3 → f a (t^2 - 2*t) + f a (2*t^2 - k) < 0) → k < -1/3)) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2155_215522


namespace NUMINAMATH_CALUDE_inequality_proof_l2155_215519

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a / (1 + b + c) + b / (1 + c + a) + c / (1 + a + b) ≥ 
       a * b / (1 + a + b) + b * c / (1 + b + c) + c * a / (1 + c + a)) :
  (a^2 + b^2 + c^2) / (a * b + b * c + c * a) + a + b + c + 2 ≥ 
  2 * (Real.sqrt (a * b) + Real.sqrt (b * c) + Real.sqrt (c * a)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2155_215519


namespace NUMINAMATH_CALUDE_max_gold_coins_theorem_l2155_215503

/-- Represents a toy that can be created -/
structure Toy where
  planks : ℕ
  value : ℕ

/-- Calculates the maximum gold coins that can be earned given a number of planks and a list of toys -/
def maxGoldCoins (totalPlanks : ℕ) (toys : List Toy) : ℕ :=
  sorry

/-- The theorem stating the maximum gold coins that can be earned -/
theorem max_gold_coins_theorem :
  let windmill : Toy := ⟨5, 6⟩
  let steamboat : Toy := ⟨7, 8⟩
  let airplane : Toy := ⟨14, 19⟩
  let toys : List Toy := [windmill, steamboat, airplane]
  maxGoldCoins 130 toys = 172 := by
  sorry

end NUMINAMATH_CALUDE_max_gold_coins_theorem_l2155_215503


namespace NUMINAMATH_CALUDE_calculation_proof_l2155_215554

theorem calculation_proof : 
  |(-7)| / ((2 / 3) - (1 / 5)) - (1 / 2) * ((-4)^2) = 7 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2155_215554


namespace NUMINAMATH_CALUDE_inequality_solution_max_value_condition_l2155_215582

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + x - a

-- Part 1
theorem inequality_solution (x : ℝ) :
  f 2 x > 1 ↔ x < -3/2 ∨ x > 1 := by sorry

-- Part 2
theorem max_value_condition (a : ℝ) :
  (∃ x, f a x = 17/8 ∧ ∀ y, f a y ≤ 17/8) →
  (a = -2 ∨ a = -1/8) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_max_value_condition_l2155_215582


namespace NUMINAMATH_CALUDE_solution_property_l2155_215506

theorem solution_property (m n : ℝ) (hm : m ≠ 0) (h : m^2 + n*m - m = 0) : m + n = 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_property_l2155_215506


namespace NUMINAMATH_CALUDE_sqrt_seven_plus_one_bounds_l2155_215590

theorem sqrt_seven_plus_one_bounds :
  3 < Real.sqrt 7 + 1 ∧ Real.sqrt 7 + 1 < 4 :=
by
  sorry

end NUMINAMATH_CALUDE_sqrt_seven_plus_one_bounds_l2155_215590


namespace NUMINAMATH_CALUDE_equation_solution_l2155_215586

theorem equation_solution :
  ∃! x : ℝ, x ≠ 3 ∧ x + 60 / (x - 3) = -13 :=
by
  -- The unique solution is x = -7
  use -7
  constructor
  · -- Prove that x = -7 satisfies the equation
    constructor
    · -- Prove -7 ≠ 3
      linarith
    · -- Prove -7 + 60 / (-7 - 3) = -13
      ring
  · -- Prove uniqueness
    intro y hy
    -- Assume y satisfies the equation
    have h1 : y ≠ 3 := hy.1
    have h2 : y + 60 / (y - 3) = -13 := hy.2
    -- Derive that y must equal -7
    sorry


end NUMINAMATH_CALUDE_equation_solution_l2155_215586


namespace NUMINAMATH_CALUDE_felipe_construction_time_l2155_215543

theorem felipe_construction_time :
  ∀ (felipe_time emilio_time : ℝ) (felipe_break emilio_break : ℝ),
    felipe_time + emilio_time = 7.5 * 12 →
    felipe_time = emilio_time / 2 →
    felipe_break = 6 →
    emilio_break = 2 * felipe_break →
    felipe_time + felipe_break = 36 :=
by
  sorry

end NUMINAMATH_CALUDE_felipe_construction_time_l2155_215543


namespace NUMINAMATH_CALUDE_triangle_expression_bounds_l2155_215513

theorem triangle_expression_bounds (A B C : Real) (a b c : Real) :
  (0 < A) → (A < π) →
  (0 < B) → (B < π) →
  (0 < C) → (C < π) →
  (A + B + C = π) →
  (c * Real.sin A = -a * Real.cos C) →
  ∃ (x : Real), (1 < x) ∧ 
  (x < (Real.sqrt 6 + Real.sqrt 2) / 2) ∧
  (x = Real.sqrt 3 * Real.sin A - Real.cos (B + 3 * π / 4)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_expression_bounds_l2155_215513


namespace NUMINAMATH_CALUDE_sum_of_digits_of_greatest_prime_factor_l2155_215562

def number : Nat := 15999

-- Define a function to get the greatest prime factor
def greatest_prime_factor (n : Nat) : Nat :=
  sorry

-- Define a function to sum the digits of a number
def sum_of_digits (n : Nat) : Nat :=
  sorry

-- Theorem statement
theorem sum_of_digits_of_greatest_prime_factor :
  sum_of_digits (greatest_prime_factor number) = 17 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_greatest_prime_factor_l2155_215562


namespace NUMINAMATH_CALUDE_car_speed_proof_l2155_215525

/-- Proves that a car's speed is 48 km/h if it takes 15 seconds longer to travel 1 km compared to 60 km/h -/
theorem car_speed_proof (v : ℝ) : v > 0 → (1 / v - 1 / 60) * 3600 = 15 ↔ v = 48 :=
by sorry

end NUMINAMATH_CALUDE_car_speed_proof_l2155_215525


namespace NUMINAMATH_CALUDE_rectangle_area_rectangle_area_is_120_l2155_215502

theorem rectangle_area (square_area : ℝ) (rectangle_breadth : ℝ) : ℝ :=
  let square_side := Real.sqrt square_area
  let circle_radius := square_side
  let rectangle_length := (2 / 5) * circle_radius
  let rectangle_area := rectangle_length * rectangle_breadth
  rectangle_area

theorem rectangle_area_is_120 :
  rectangle_area 900 10 = 120 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_rectangle_area_is_120_l2155_215502


namespace NUMINAMATH_CALUDE_vector_sum_zero_parallel_necessary_not_sufficient_l2155_215583

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

def parallel (a b : V) : Prop := ∃ (k : ℝ), b = k • a

theorem vector_sum_zero_parallel_necessary_not_sufficient :
  (∀ (a b : V), a ≠ 0 ∧ b ≠ 0 → (a + b = 0 → parallel a b)) ∧
  (∃ (a b : V), a ≠ 0 ∧ b ≠ 0 ∧ parallel a b ∧ a + b ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_vector_sum_zero_parallel_necessary_not_sufficient_l2155_215583


namespace NUMINAMATH_CALUDE_man_downstream_speed_l2155_215542

/-- Calculates the downstream speed given upstream and still water speeds -/
def downstream_speed (upstream : ℝ) (still_water : ℝ) : ℝ :=
  2 * still_water - upstream

theorem man_downstream_speed :
  let upstream_speed : ℝ := 12
  let still_water_speed : ℝ := 7
  downstream_speed upstream_speed still_water_speed = 14 := by
  sorry

end NUMINAMATH_CALUDE_man_downstream_speed_l2155_215542


namespace NUMINAMATH_CALUDE_table_height_is_33_l2155_215587

/-- Represents a block of wood -/
structure Block where
  length : ℝ
  width : ℝ

/-- Represents a table -/
structure Table where
  height : ℝ

/-- Represents the configuration of blocks on the table -/
inductive Configuration
| A
| B

/-- Calculates the total visible length for a given configuration -/
def totalVisibleLength (block : Block) (table : Table) (config : Configuration) : ℝ :=
  match config with
  | Configuration.A => block.length + table.height - block.width
  | Configuration.B => block.width + table.height - block.length

/-- Theorem stating that under the given conditions, the table's height is 33 inches -/
theorem table_height_is_33 (block : Block) (table : Table) :
  totalVisibleLength block table Configuration.A = 36 →
  totalVisibleLength block table Configuration.B = 30 →
  table.height = 33 := by
  sorry

#check table_height_is_33

end NUMINAMATH_CALUDE_table_height_is_33_l2155_215587


namespace NUMINAMATH_CALUDE_interval_of_decrease_l2155_215549

def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 2

theorem interval_of_decrease (a : ℝ) :
  (∀ x : ℝ, x ≤ 4 → (∀ y : ℝ, y < x → f a y > f a x)) →
  a = -3 := by
sorry

end NUMINAMATH_CALUDE_interval_of_decrease_l2155_215549


namespace NUMINAMATH_CALUDE_not_divisible_by_49_l2155_215531

theorem not_divisible_by_49 (n : ℤ) : ¬ ∃ k : ℤ, n^2 + 3*n + 4 = 49*k := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_49_l2155_215531


namespace NUMINAMATH_CALUDE_intercept_sum_modulo_13_l2155_215596

theorem intercept_sum_modulo_13 : ∃ (x₀ y₀ : ℕ), 
  x₀ < 13 ∧ y₀ < 13 ∧ 
  (4 * x₀ ≡ 1 [MOD 13]) ∧ 
  (3 * y₀ ≡ 12 [MOD 13]) ∧ 
  x₀ + y₀ = 14 := by
  sorry

end NUMINAMATH_CALUDE_intercept_sum_modulo_13_l2155_215596


namespace NUMINAMATH_CALUDE_t_value_l2155_215507

theorem t_value (p j t : ℝ) 
  (hj_p : j = p * (1 - 0.25))
  (hj_t : j = t * (1 - 0.20))
  (ht_p : t = p * (1 - t / 100)) :
  t = 6.25 := by sorry

end NUMINAMATH_CALUDE_t_value_l2155_215507


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2155_215553

theorem complex_equation_solution (a b : ℝ) : 
  (a : ℂ) + 3 * I = (b + I) * I → a = -1 ∧ b = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2155_215553


namespace NUMINAMATH_CALUDE_cubic_function_properties_monotonicity_interval_l2155_215532

/-- A cubic function f(x) = ax^3 + bx^2 passing through (1,4) with slope 9 at x = 1 -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2

/-- The derivative of f -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x

theorem cubic_function_properties (a b : ℝ) :
  f a b 1 = 4 ∧ f' a b 1 = 9 → a = 1 ∧ b = 3 :=
sorry

theorem monotonicity_interval (a b m : ℝ) :
  (a = 1 ∧ b = 3) →
  (∀ x ∈ Set.Icc m (m + 1), f' a b x ≥ 0) ↔ (m ≥ 0 ∨ m ≤ -3) :=
sorry

end NUMINAMATH_CALUDE_cubic_function_properties_monotonicity_interval_l2155_215532


namespace NUMINAMATH_CALUDE_x_coordinate_range_l2155_215585

-- Define the line L
def L (x y : ℝ) : Prop := x + y - 9 = 0

-- Define the circle M
def M (x y : ℝ) : Prop := 2*x^2 + 2*y^2 - 8*x - 8*y - 1 = 0

-- Define the triangle ABC
structure Triangle (A B C : ℝ × ℝ) : Prop where
  A_on_L : L A.1 A.2
  B_on_M : M B.1 B.2
  C_on_M : M C.1 C.2
  angle_BAC : Real.cos (45 * π / 180) = (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) /
    (Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) * Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2))

-- Define that AB passes through the center of M
def AB_through_center (A B : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 2 = A.1 + t * (B.1 - A.1) ∧ 2 = A.2 + t * (B.2 - A.2)

-- Main theorem
theorem x_coordinate_range (A B C : ℝ × ℝ) 
  (h_triangle : Triangle A B C) (h_AB : AB_through_center A B) : 
  3 ≤ A.1 ∧ A.1 ≤ 6 := by
  sorry

end NUMINAMATH_CALUDE_x_coordinate_range_l2155_215585


namespace NUMINAMATH_CALUDE_smallest_possible_b_l2155_215530

theorem smallest_possible_b (a b c : ℝ) : 
  1 < a → a < b → c = 2 → 
  (¬ (c + a > b ∧ c + b > a ∧ a + b > c)) →
  (¬ ((1/b) + (1/a) > c ∧ (1/b) + c > (1/a) ∧ (1/a) + c > (1/b))) →
  b ≥ 2 ∧ ∀ x, (x > 1 ∧ x < b → x ≥ a) → b = 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_possible_b_l2155_215530


namespace NUMINAMATH_CALUDE_sum_of_digits_l2155_215541

def is_single_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def number_representation (A B : ℕ) : ℕ := A * 100000 + 44610 + B

theorem sum_of_digits (A B : ℕ) : 
  is_single_digit A → 
  is_single_digit B → 
  (number_representation A B) % 72 = 0 → 
  A + B = 12 := by
sorry

end NUMINAMATH_CALUDE_sum_of_digits_l2155_215541


namespace NUMINAMATH_CALUDE_water_problem_solution_l2155_215564

def water_problem (total_water : ℕ) (original_serving : ℕ) (serving_reduction : ℕ) : ℕ :=
  let original_servings := total_water / original_serving
  let new_servings := original_servings - serving_reduction
  total_water / new_servings

theorem water_problem_solution :
  water_problem 64 8 4 = 16 := by
  sorry

end NUMINAMATH_CALUDE_water_problem_solution_l2155_215564


namespace NUMINAMATH_CALUDE_train_passing_bridge_time_l2155_215517

/-- Calculates the time for a train to pass a bridge -/
theorem train_passing_bridge_time (train_length : Real) (bridge_length : Real) (train_speed_kmh : Real) :
  let total_distance : Real := train_length + bridge_length
  let train_speed_ms : Real := train_speed_kmh * 1000 / 3600
  let time : Real := total_distance / train_speed_ms
  train_length = 200 ∧ bridge_length = 180 ∧ train_speed_kmh = 65 →
  ∃ ε > 0, |time - 21.04| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_train_passing_bridge_time_l2155_215517


namespace NUMINAMATH_CALUDE_three_digit_same_divisible_by_37_l2155_215514

theorem three_digit_same_divisible_by_37 (a : ℕ) (h : a ≤ 9) :
  ∃ k : ℕ, 111 * a = 37 * k := by
  sorry

end NUMINAMATH_CALUDE_three_digit_same_divisible_by_37_l2155_215514


namespace NUMINAMATH_CALUDE_x_sixth_geq_2a_minus_1_l2155_215578

theorem x_sixth_geq_2a_minus_1 (x a : ℝ) (h : x^5 - x^3 + x = a) : x^6 ≥ 2*a - 1 := by
  sorry

end NUMINAMATH_CALUDE_x_sixth_geq_2a_minus_1_l2155_215578


namespace NUMINAMATH_CALUDE_ellipse_line_slope_l2155_215574

/-- The slope of a line passing through the right focus of an ellipse -/
theorem ellipse_line_slope (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  let e := Real.sqrt 3 / 2
  let ellipse := fun (x y : ℝ) => x^2 / a^2 + y^2 / b^2 = 1
  let F := (Real.sqrt (a^2 - b^2), 0)
  ∀ (A B : ℝ × ℝ),
    ellipse A.1 A.2 ∧ 
    ellipse B.1 B.2 ∧ 
    A.2 > 0 ∧ 
    B.2 < 0 ∧
    (A.1 - F.1, A.2 - F.2) = 3 • (F.1 - B.1, F.2 - B.2) →
    (A.2 - B.2) / (A.1 - B.1) = -Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_line_slope_l2155_215574


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l2155_215505

/-- Given an ellipse and a hyperbola with shared foci, prove that the eccentricity of the ellipse is 1/2 -/
theorem ellipse_eccentricity (a b m n c : ℝ) : 
  a > 0 → b > 0 → m > 0 → n > 0 → a > b →
  (∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1) →  -- Ellipse equation
  (∀ x y : ℝ, x^2/m^2 - y^2/n^2 = 1) →  -- Hyperbola equation
  c^2 = a^2 - b^2 →                     -- Shared foci condition for ellipse
  c^2 = m^2 + n^2 →                     -- Shared foci condition for hyperbola
  c^2 = a * m →                         -- c is geometric mean of a and m
  n^2 = m^2 + c^2/2 →                   -- n^2 is arithmetic mean of 2m^2 and c^2
  c/a = 1/2 := by
sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l2155_215505


namespace NUMINAMATH_CALUDE_factorial_inequality_l2155_215538

theorem factorial_inequality (n p : ℕ) (h : 2 * p ≤ n) :
  (n - p).factorial / p.factorial ≤ ((n + 1) / 2 : ℚ) ^ (n - 2 * p) ∧
  ((n - p).factorial / p.factorial = ((n + 1) / 2 : ℚ) ^ (n - 2 * p) ↔ n = 2 * p ∨ n = 2 * p + 1) :=
by sorry

end NUMINAMATH_CALUDE_factorial_inequality_l2155_215538


namespace NUMINAMATH_CALUDE_flower_shop_expenses_flower_shop_weekly_expenses_l2155_215592

/-- Weekly expenses for running a flower shop -/
theorem flower_shop_expenses (rent : ℝ) (utility_rate : ℝ) (hours_per_day : ℕ) 
  (days_per_week : ℕ) (employees_per_shift : ℕ) (hourly_wage : ℝ) : ℝ :=
  let utilities := rent * utility_rate
  let employee_hours := hours_per_day * days_per_week * employees_per_shift
  let employee_wages := employee_hours * hourly_wage
  rent + utilities + employee_wages

/-- Proof of the flower shop's weekly expenses -/
theorem flower_shop_weekly_expenses :
  flower_shop_expenses 1200 0.2 16 5 2 12.5 = 3440 := by
  sorry

end NUMINAMATH_CALUDE_flower_shop_expenses_flower_shop_weekly_expenses_l2155_215592


namespace NUMINAMATH_CALUDE_ratio_of_distances_l2155_215520

/-- Given four points P, Q, R, and S on a line (in that order), with distances PQ = 3, QR = 7, and PS = 22,
    prove that the ratio of PR to QS is 10/19. -/
theorem ratio_of_distances (P Q R S : ℝ) (h_order : P < Q ∧ Q < R ∧ R < S) 
  (h_PQ : Q - P = 3) (h_QR : R - Q = 7) (h_PS : S - P = 22) : 
  (R - P) / (S - Q) = 10 / 19 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_distances_l2155_215520


namespace NUMINAMATH_CALUDE_intersection_in_first_quadrant_l2155_215518

/-- The range of k for which the intersection of two lines lies in the first quadrant -/
theorem intersection_in_first_quadrant (k : ℝ) : 
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ y = k * x - 1 ∧ x + y - 1 = 0) ↔ k > 1 :=
by sorry

end NUMINAMATH_CALUDE_intersection_in_first_quadrant_l2155_215518


namespace NUMINAMATH_CALUDE_common_number_in_overlapping_lists_l2155_215572

theorem common_number_in_overlapping_lists (l : List ℚ) : 
  l.length = 7 ∧ 
  (l.take 4).sum / 4 = 7 ∧ 
  (l.drop 3).sum / 4 = 11 ∧ 
  l.sum / 7 = 66 / 7 → 
  ∃ x ∈ l.take 4 ∩ l.drop 3, x = 6 := by
sorry

end NUMINAMATH_CALUDE_common_number_in_overlapping_lists_l2155_215572


namespace NUMINAMATH_CALUDE_area_ratio_of_squares_l2155_215569

theorem area_ratio_of_squares (side_A side_B : ℝ) (h1 : side_A = 36) (h2 : side_B = 42) :
  (side_A ^ 2) / (side_B ^ 2) = 36 / 49 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_of_squares_l2155_215569


namespace NUMINAMATH_CALUDE_not_perfect_squares_l2155_215584

theorem not_perfect_squares : ∃ (n : ℕ → ℕ), 
  (n 1 = 2048) ∧ 
  (n 2 = 2049) ∧ 
  (n 3 = 2050) ∧ 
  (n 4 = 2051) ∧ 
  (n 5 = 2052) ∧ 
  (∃ (a : ℕ), 1^(n 1) = a^2) ∧ 
  (¬∃ (b : ℕ), 2^(n 2) = b^2) ∧ 
  (∃ (c : ℕ), 3^(n 3) = c^2) ∧ 
  (¬∃ (d : ℕ), 4^(n 4) = d^2) ∧ 
  (∃ (e : ℕ), 5^(n 5) = e^2) :=
by sorry


end NUMINAMATH_CALUDE_not_perfect_squares_l2155_215584


namespace NUMINAMATH_CALUDE_man_mass_on_boat_l2155_215528

/-- The mass of a man who causes a boat to sink by a certain amount. -/
def mass_of_man (boat_length boat_breadth boat_sink_height water_density : ℝ) : ℝ :=
  boat_length * boat_breadth * boat_sink_height * water_density

/-- Theorem stating the mass of the man in the given problem. -/
theorem man_mass_on_boat : 
  let boat_length : ℝ := 8
  let boat_breadth : ℝ := 3
  let boat_sink_height : ℝ := 0.01  -- 1 cm in meters
  let water_density : ℝ := 1000     -- kg/m³
  mass_of_man boat_length boat_breadth boat_sink_height water_density = 240 := by
  sorry


end NUMINAMATH_CALUDE_man_mass_on_boat_l2155_215528


namespace NUMINAMATH_CALUDE_max_area_rectangle_with_perimeter_60_l2155_215546

/-- The maximum area of a rectangle with perimeter 60 is 225 -/
theorem max_area_rectangle_with_perimeter_60 :
  ∀ a b : ℝ,
  a > 0 → b > 0 →
  2 * a + 2 * b = 60 →
  a * b ≤ 225 :=
by sorry

end NUMINAMATH_CALUDE_max_area_rectangle_with_perimeter_60_l2155_215546


namespace NUMINAMATH_CALUDE_R_sufficient_not_necessary_for_Q_l2155_215501

-- Define the propositions
variable (R P Q S : Prop)

-- Define the given conditions
axiom S_necessary_for_R : R → S
axiom S_sufficient_for_P : S → P
axiom Q_necessary_for_P : P → Q
axiom Q_sufficient_for_S : Q → S

-- Theorem to prove
theorem R_sufficient_not_necessary_for_Q :
  (R → Q) ∧ ¬(Q → R) :=
sorry

end NUMINAMATH_CALUDE_R_sufficient_not_necessary_for_Q_l2155_215501


namespace NUMINAMATH_CALUDE_range_of_f_l2155_215533

-- Define the function
def f (x : ℝ) : ℝ := x^2 + 2*x + 3

-- State the theorem
theorem range_of_f :
  {y | ∃ x ≥ 0, f x = y} = {y | y ≥ 3} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l2155_215533


namespace NUMINAMATH_CALUDE_arithmetic_progression_relatively_prime_l2155_215508

theorem arithmetic_progression_relatively_prime :
  ∃ (a : ℕ → ℕ) (d : ℕ),
    (∀ n, 1 ≤ n → n ≤ 100 → a n > 0) ∧
    (∀ n m, 1 ≤ n → n < m → m ≤ 100 → a m > a n) ∧
    (∀ n, 1 < n → n ≤ 100 → a n - a (n-1) = d) ∧
    (∀ n m, 1 ≤ n → n < m → m ≤ 100 → Nat.gcd (a n) (a m) = 1) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_relatively_prime_l2155_215508


namespace NUMINAMATH_CALUDE_pizza_eaters_fraction_l2155_215560

theorem pizza_eaters_fraction (total_people : ℕ) (total_pizza : ℕ) (pieces_per_person : ℕ) (remaining_pizza : ℕ)
  (h1 : total_people = 15)
  (h2 : total_pizza = 50)
  (h3 : pieces_per_person = 4)
  (h4 : remaining_pizza = 14) :
  (total_pizza - remaining_pizza) / (pieces_per_person * total_people) = 3 / 5 := by
sorry

end NUMINAMATH_CALUDE_pizza_eaters_fraction_l2155_215560


namespace NUMINAMATH_CALUDE_csc_135_deg_l2155_215589

-- Define the cosecant function
noncomputable def csc (θ : Real) : Real := 1 / Real.sin θ

-- State the theorem
theorem csc_135_deg : csc (135 * π / 180) = Real.sqrt 2 := by
  -- Define the given conditions
  have sin_135 : Real.sin (135 * π / 180) = 1 / Real.sqrt 2 := by sorry
  have cos_135 : Real.cos (135 * π / 180) = -(1 / Real.sqrt 2) := by sorry

  -- Prove the theorem
  sorry

end NUMINAMATH_CALUDE_csc_135_deg_l2155_215589


namespace NUMINAMATH_CALUDE_lcm_count_l2155_215504

theorem lcm_count : 
  ∃! (n : ℕ), n > 0 ∧ 
  (∃ (S : Finset ℕ), S.card = n ∧ 
    (∀ k ∈ S, k > 0 ∧ Nat.lcm (6^9) (Nat.lcm (9^9) k) = 18^18) ∧
    (∀ k ∉ S, k > 0 → Nat.lcm (6^9) (Nat.lcm (9^9) k) ≠ 18^18)) := by
  sorry

end NUMINAMATH_CALUDE_lcm_count_l2155_215504


namespace NUMINAMATH_CALUDE_scientific_notation_of_nine_billion_l2155_215539

theorem scientific_notation_of_nine_billion :
  9000000000 = 9 * (10 : ℝ)^9 :=
by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_nine_billion_l2155_215539


namespace NUMINAMATH_CALUDE_inequality_proof_l2155_215557

theorem inequality_proof (x y z : ℝ) 
  (hx : x ≠ 1) (hy : y ≠ 1) (hz : z ≠ 1) (hxyz : x * y * z = 1) : 
  (x^2 / (x - 1)^2) + (y^2 / (y - 1)^2) + (z^2 / (z - 1)^2) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2155_215557


namespace NUMINAMATH_CALUDE_carson_giant_slide_rides_l2155_215566

/-- Represents the number of times Carson can ride the giant slide at the carnival -/
def giant_slide_rides (total_time minutes_per_hour roller_coaster_wait tilt_a_whirl_wait giant_slide_wait : ℕ)
  (roller_coaster_rides tilt_a_whirl_rides : ℕ) : ℕ :=
  let remaining_time := total_time * minutes_per_hour -
    (roller_coaster_wait * roller_coaster_rides + tilt_a_whirl_wait * tilt_a_whirl_rides)
  remaining_time / giant_slide_wait

/-- Theorem stating the number of times Carson can ride the giant slide -/
theorem carson_giant_slide_rides :
  giant_slide_rides 4 60 30 60 15 4 1 = 4 :=
sorry

end NUMINAMATH_CALUDE_carson_giant_slide_rides_l2155_215566


namespace NUMINAMATH_CALUDE_selection_ways_equal_210_l2155_215500

/-- The number of ways to select at least one boy from a group of 6 boys and G girls is 210 if and only if G = 1 -/
theorem selection_ways_equal_210 (G : ℕ) : (63 * 2^G = 210) ↔ G = 1 := by sorry

end NUMINAMATH_CALUDE_selection_ways_equal_210_l2155_215500


namespace NUMINAMATH_CALUDE_bus_ride_net_change_l2155_215559

/-- Represents the number of children on a bus and the changes at each stop -/
structure BusRide where
  initial : Int
  first_stop_off : Int
  first_stop_on : Int
  second_stop_off : Int
  final : Int

/-- Calculates the difference between total children who got off and got on -/
def net_change (ride : BusRide) : Int :=
  ride.first_stop_off + ride.second_stop_off - 
  (ride.first_stop_on + (ride.final - (ride.initial - ride.first_stop_off + ride.first_stop_on - ride.second_stop_off)))

/-- Theorem stating the net change in children for the given bus ride -/
theorem bus_ride_net_change :
  let ride : BusRide := {
    initial := 36,
    first_stop_off := 45,
    first_stop_on := 25,
    second_stop_off := 68,
    final := 12
  }
  net_change ride = 24 := by sorry

end NUMINAMATH_CALUDE_bus_ride_net_change_l2155_215559


namespace NUMINAMATH_CALUDE_sailboat_canvas_area_l2155_215593

/-- The total area of canvas needed for a model sailboat with three sails -/
theorem sailboat_canvas_area
  (rect_length : ℝ)
  (rect_width : ℝ)
  (tri1_base : ℝ)
  (tri1_height : ℝ)
  (tri2_base : ℝ)
  (tri2_height : ℝ)
  (h_rect_length : rect_length = 5)
  (h_rect_width : rect_width = 8)
  (h_tri1_base : tri1_base = 3)
  (h_tri1_height : tri1_height = 4)
  (h_tri2_base : tri2_base = 4)
  (h_tri2_height : tri2_height = 6) :
  rect_length * rect_width +
  (tri1_base * tri1_height) / 2 +
  (tri2_base * tri2_height) / 2 = 58 := by
sorry


end NUMINAMATH_CALUDE_sailboat_canvas_area_l2155_215593


namespace NUMINAMATH_CALUDE_smallest_marble_count_l2155_215551

/-- Represents the number of marbles of each color in the urn -/
structure MarbleCount where
  red : ℕ
  white : ℕ
  blue : ℕ
  green : ℕ
  yellow : ℕ

/-- Calculates the total number of marbles in the urn -/
def totalMarbles (m : MarbleCount) : ℕ :=
  m.red + m.white + m.blue + m.green + m.yellow

/-- Calculates the probability of drawing 5 red marbles -/
def probFiveRed (m : MarbleCount) : ℚ :=
  (m.red.choose 5 : ℚ) / (totalMarbles m).choose 5

/-- Calculates the probability of drawing 1 white and 4 red marbles -/
def probOneWhiteFourRed (m : MarbleCount) : ℚ :=
  ((m.white.choose 1 * m.red.choose 4) : ℚ) / (totalMarbles m).choose 5

/-- Calculates the probability of drawing 1 white, 1 blue, and 3 red marbles -/
def probOneWhiteOneBlueTwoRed (m : MarbleCount) : ℚ :=
  ((m.white.choose 1 * m.blue.choose 1 * m.red.choose 3) : ℚ) / (totalMarbles m).choose 5

/-- Calculates the probability of drawing 1 white, 1 blue, 1 green, and 2 red marbles -/
def probOneWhiteOneBlueOneGreenTwoRed (m : MarbleCount) : ℚ :=
  ((m.white.choose 1 * m.blue.choose 1 * m.green.choose 1 * m.red.choose 2) : ℚ) / (totalMarbles m).choose 5

/-- Calculates the probability of drawing one marble of each color -/
def probOneEachColor (m : MarbleCount) : ℚ :=
  ((m.white.choose 1 * m.blue.choose 1 * m.green.choose 1 * m.yellow.choose 1 * m.red.choose 1) : ℚ) / (totalMarbles m).choose 5

/-- Checks if all probabilities are equal -/
def allProbabilitiesEqual (m : MarbleCount) : Prop :=
  probFiveRed m = probOneWhiteFourRed m ∧
  probFiveRed m = probOneWhiteOneBlueTwoRed m ∧
  probFiveRed m = probOneWhiteOneBlueOneGreenTwoRed m ∧
  probFiveRed m = probOneEachColor m

/-- The main theorem stating that 33 is the smallest number of marbles satisfying the conditions -/
theorem smallest_marble_count : 
  ∃ (m : MarbleCount), totalMarbles m = 33 ∧ allProbabilitiesEqual m ∧
  ∀ (m' : MarbleCount), totalMarbles m' < 33 → ¬allProbabilitiesEqual m' :=
sorry

end NUMINAMATH_CALUDE_smallest_marble_count_l2155_215551


namespace NUMINAMATH_CALUDE_range_of_a_l2155_215595

theorem range_of_a (a : ℝ) : (∃ x : ℝ, |x - a| + |x| ≤ 4) → -4 ≤ a ∧ a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2155_215595


namespace NUMINAMATH_CALUDE_equation_properties_l2155_215575

variable (a : ℝ)
variable (z : ℂ)

def has_real_solution (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 - (a + Complex.I)*x - (Complex.I + 2) = 0

def has_imaginary_solution (a : ℝ) : Prop :=
  ∃ y : ℝ, y ≠ 0 ∧ (Complex.I*y)^2 - (a + Complex.I)*(Complex.I*y) - (Complex.I + 2) = 0

theorem equation_properties :
  (has_real_solution a ↔ a = 1) ∧
  ¬(has_imaginary_solution a) := by sorry

end NUMINAMATH_CALUDE_equation_properties_l2155_215575


namespace NUMINAMATH_CALUDE_sum_of_squares_l2155_215571

theorem sum_of_squares (a b : ℝ) : (a^2 + b^2) * (a^2 + b^2 + 4) = 12 → a^2 + b^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l2155_215571


namespace NUMINAMATH_CALUDE_circle_parabola_tangent_radius_l2155_215577

-- Define the parabola Γ
def Γ : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2^2 = 4 * p.1}

-- Define the circle Ω with center (1, r) and radius r
def Ω (r : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - r)^2 = r^2}

-- State the theorem
theorem circle_parabola_tangent_radius :
  ∃! r : ℝ, r > 0 ∧
  (∃! p : ℝ × ℝ, p ∈ Γ ∩ Ω r) ∧
  (1, 0) ∈ Ω r ∧
  (∀ ε > 0, ∃ q : ℝ × ℝ, q.2 = -ε ∧ q ∉ Ω r) ∧
  r = 4 * Real.sqrt 3 / 9 := by
sorry

end NUMINAMATH_CALUDE_circle_parabola_tangent_radius_l2155_215577


namespace NUMINAMATH_CALUDE_saree_price_calculation_l2155_215581

theorem saree_price_calculation (P : ℝ) : 
  P * (1 - 0.20) * (1 - 0.05) = 152 → P = 200 := by
  sorry

end NUMINAMATH_CALUDE_saree_price_calculation_l2155_215581


namespace NUMINAMATH_CALUDE_max_log_product_l2155_215550

theorem max_log_product (x y : ℝ) (hx : x > 1) (hy : y > 1) (h_sum : Real.log x / Real.log 10 + Real.log y / Real.log 10 = 4) :
  ∃ (max_val : ℝ), ∀ (a b : ℝ), a > 1 → b > 1 → Real.log a / Real.log 10 + Real.log b / Real.log 10 = 4 →
    (Real.log x / Real.log 10) * (Real.log y / Real.log 10) ≥ (Real.log a / Real.log 10) * (Real.log b / Real.log 10) ∧
    max_val = (Real.log x / Real.log 10) * (Real.log y / Real.log 10) ∧
    max_val = 4 :=
by sorry

end NUMINAMATH_CALUDE_max_log_product_l2155_215550


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2155_215580

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_property
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_condition : a 1 * (a 8)^3 * a 15 = 243) :
  (a 9)^3 / a 11 = 9 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l2155_215580


namespace NUMINAMATH_CALUDE_divisibility_problem_l2155_215545

theorem divisibility_problem (a b c d m : ℤ) 
  (h_m_pos : m > 0)
  (h_ac : m ∣ a * c)
  (h_bd : m ∣ b * d)
  (h_sum : m ∣ b * c + a * d) :
  (m ∣ b * c) ∧ (m ∣ a * d) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_problem_l2155_215545


namespace NUMINAMATH_CALUDE_math_competition_probabilities_l2155_215526

theorem math_competition_probabilities :
  let total_students : ℕ := 6
  let boys : ℕ := 3
  let girls : ℕ := 3
  let selected : ℕ := 2

  let prob_exactly_one_boy : ℚ := 3/5
  let prob_at_least_one_boy : ℚ := 4/5
  let prob_at_most_one_boy : ℚ := 4/5

  (total_students = boys + girls) →
  (prob_exactly_one_boy = 0.6) ∧
  (prob_at_least_one_boy = 0.8) ∧
  (prob_at_most_one_boy = 0.8) :=
by
  sorry

end NUMINAMATH_CALUDE_math_competition_probabilities_l2155_215526


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l2155_215591

-- Problem 1
theorem problem_1 : (π - 1)^0 - Real.sqrt 8 + |(- 2) * Real.sqrt 2| = 1 := by
  sorry

-- Problem 2
theorem problem_2 : ∀ x : ℝ, 3 * x - 2 > x + 4 ↔ x > 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l2155_215591


namespace NUMINAMATH_CALUDE_scooter_only_owners_l2155_215599

theorem scooter_only_owners (total : ℕ) (scooter : ℕ) (bike : ℕ) 
  (h1 : total = 450) 
  (h2 : scooter = 380) 
  (h3 : bike = 120) : 
  scooter - (scooter + bike - total) = 330 := by
  sorry

end NUMINAMATH_CALUDE_scooter_only_owners_l2155_215599


namespace NUMINAMATH_CALUDE_product_selection_and_testing_l2155_215536

def total_products : ℕ := 10
def defective_products : ℕ := 3

theorem product_selection_and_testing :
  (let selections_with_defective := Nat.choose total_products 3 - Nat.choose (total_products - defective_products) 3
   selections_with_defective = 85) ∧
  (let testing_methods := defective_products * Nat.choose (total_products - defective_products) 2
   testing_methods = 1512) := by sorry

end NUMINAMATH_CALUDE_product_selection_and_testing_l2155_215536


namespace NUMINAMATH_CALUDE_solve_system_l2155_215579

theorem solve_system (p q : ℚ) (eq1 : 5 * p + 3 * q = 7) (eq2 : 2 * p + 5 * q = 8) : p = 11 / 19 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l2155_215579


namespace NUMINAMATH_CALUDE_sum_of_coefficients_cubic_factorization_l2155_215509

theorem sum_of_coefficients_cubic_factorization :
  ∃ (p q r s t : ℤ), 
    (∀ y, 512 * y^3 + 27 = (p * y + q) * (r * y^2 + s * y + t)) ∧
    p + q + r + s + t = 60 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_cubic_factorization_l2155_215509


namespace NUMINAMATH_CALUDE_divisibility_property_l2155_215548

theorem divisibility_property (y : ℕ) (h : y ≠ 0) :
  (y - 1) ∣ (y^(y^2) - 2*y^(y+1) + 1) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_property_l2155_215548


namespace NUMINAMATH_CALUDE_polygon_sides_l2155_215565

theorem polygon_sides (n : ℕ) (angle_sum : ℝ) : 
  (n ≥ 3) →  -- Ensure it's a polygon
  (angle_sum = 2790) →  -- Given sum of angles except one
  (∃ x : ℝ, x > 0 ∧ x < 180 ∧ 180 * (n - 2) = angle_sum + x) →  -- Existence of the missing angle
  (n = 18) :=
by sorry

end NUMINAMATH_CALUDE_polygon_sides_l2155_215565


namespace NUMINAMATH_CALUDE_vector_expression_simplification_l2155_215568

variable {V : Type*} [AddCommGroup V]

theorem vector_expression_simplification
  (CE AC DE AD : V) :
  CE + AC - DE - AD = (0 : V) := by
  sorry

end NUMINAMATH_CALUDE_vector_expression_simplification_l2155_215568


namespace NUMINAMATH_CALUDE_f_at_negative_one_l2155_215597

def f (x : ℝ) : ℝ := 5 * (2 * x^3 - 3 * x^2 + 4 * x - 1)

theorem f_at_negative_one : f (-1) = -50 := by
  sorry

end NUMINAMATH_CALUDE_f_at_negative_one_l2155_215597
