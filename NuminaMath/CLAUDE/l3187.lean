import Mathlib

namespace same_color_probability_is_121_450_l3187_318727

/-- Represents a 30-sided die with colored sides -/
structure ColoredDie :=
  (maroon : ℕ)
  (teal : ℕ)
  (cyan : ℕ)
  (sparkly : ℕ)
  (total_sides : ℕ)
  (side_sum : maroon + teal + cyan + sparkly = total_sides)

/-- The probability of rolling the same color or element on two identical colored dice -/
def same_color_probability (d : ColoredDie) : ℚ :=
  (d.maroon^2 + d.teal^2 + d.cyan^2 + d.sparkly^2) / d.total_sides^2

/-- The specific die described in the problem -/
def problem_die : ColoredDie :=
  { maroon := 6
  , teal := 9
  , cyan := 10
  , sparkly := 5
  , total_sides := 30
  , side_sum := by rfl }

theorem same_color_probability_is_121_450 :
  same_color_probability problem_die = 121 / 450 := by
  sorry

end same_color_probability_is_121_450_l3187_318727


namespace semicircle_radius_l3187_318752

/-- The radius of a semi-circle given its perimeter -/
theorem semicircle_radius (perimeter : ℝ) (h : perimeter = 108) :
  ∃ r : ℝ, r = perimeter / (Real.pi + 2) := by
  sorry

end semicircle_radius_l3187_318752


namespace max_value_product_l3187_318771

theorem max_value_product (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 8) :
  (∀ a b : ℝ, a > 0 → b > 0 → a + b = 8 → (1 + a) * (1 + b) ≤ (1 + x) * (1 + y)) ∧
  (1 + x) * (1 + y) = 25 :=
sorry

end max_value_product_l3187_318771


namespace tickets_left_l3187_318725

def tickets_bought : ℕ := 13
def ticket_cost : ℕ := 9
def spent_on_ferris_wheel : ℕ := 81

theorem tickets_left : tickets_bought - (spent_on_ferris_wheel / ticket_cost) = 4 := by
  sorry

end tickets_left_l3187_318725


namespace inequality_proof_l3187_318772

theorem inequality_proof :
  (∀ x y : ℝ, x^2 + y^2 + 1 > x * (y + 1)) ∧
  (∀ k : ℝ, (∀ x y : ℝ, x^2 + y^2 + 1 ≥ k * x * (y + 1)) → k ≤ Real.sqrt 2) ∧
  (∀ k : ℝ, (∀ m n : ℤ, (m : ℝ)^2 + (n : ℝ)^2 + 1 ≥ k * (m : ℝ) * ((n : ℝ) + 1)) → k ≤ 3/2) :=
by sorry

end inequality_proof_l3187_318772


namespace sheep_count_l3187_318736

/-- The number of sheep in the meadow -/
def num_sheep : ℕ := 36

/-- The number of cows in the meadow -/
def num_cows : ℕ := 12

/-- The number of ears per cow -/
def ears_per_cow : ℕ := 2

/-- The number of legs per cow -/
def legs_per_cow : ℕ := 4

/-- Theorem stating that the number of sheep is 36 given the conditions -/
theorem sheep_count :
  num_sheep > num_cows * ears_per_cow ∧
  num_sheep < num_cows * legs_per_cow ∧
  num_sheep % 12 = 0 →
  num_sheep = 36 :=
by sorry

end sheep_count_l3187_318736


namespace curve_has_axis_of_symmetry_l3187_318764

/-- The equation of the curve -/
def curve_equation (x y : ℝ) : Prop :=
  x^2 - x*y + y^2 + x - y - 1 = 0

/-- The proposed axis of symmetry -/
def axis_of_symmetry (x y : ℝ) : Prop :=
  x + y = 0

/-- Theorem stating that the curve has the given axis of symmetry -/
theorem curve_has_axis_of_symmetry :
  ∀ (x y : ℝ), curve_equation x y ↔ curve_equation (-y) (-x) :=
sorry

end curve_has_axis_of_symmetry_l3187_318764


namespace hypotenuse_length_l3187_318753

/-- A right triangle with given perimeter and difference between median and altitude. -/
structure RightTriangle where
  /-- Side length BC -/
  a : ℝ
  /-- Side length AC -/
  b : ℝ
  /-- Hypotenuse length AB -/
  c : ℝ
  /-- Perimeter of the triangle -/
  perimeter_eq : a + b + c = 72
  /-- Pythagorean theorem -/
  pythagoras : a^2 + b^2 = c^2
  /-- Difference between median and altitude -/
  median_altitude_diff : c / 2 - (a * b) / c = 7

/-- The hypotenuse of a right triangle with the given properties is 32 cm. -/
theorem hypotenuse_length (t : RightTriangle) : t.c = 32 := by
  sorry

end hypotenuse_length_l3187_318753


namespace number_reduced_then_increased_l3187_318789

theorem number_reduced_then_increased : ∃ x : ℝ, (20 * (x / 5) = 40) ∧ (x = 10) := by
  sorry

end number_reduced_then_increased_l3187_318789


namespace negation_equivalence_l3187_318701

theorem negation_equivalence :
  (¬ ∃ x₀ : ℝ, x₀ > 1 ∧ x₀^2 - x₀ + 2016 > 0) ↔
  (¬ ∃ x : ℝ, x > 1 ∧ x^2 - x + 2016 ≤ 0) := by
sorry

end negation_equivalence_l3187_318701


namespace product_inequality_l3187_318705

theorem product_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  (a - 1 + 1 / b) * (b - 1 + 1 / c) * (c - 1 + 1 / a) ≤ 1 := by
  sorry

end product_inequality_l3187_318705


namespace prop_p_and_q_false_l3187_318738

theorem prop_p_and_q_false : 
  (¬(∀ a b : ℝ, a > b → a^2 > b^2)) ∧ 
  (¬(∃ x : ℝ, x^2 + 2 > 3*x)) := by
  sorry

end prop_p_and_q_false_l3187_318738


namespace geometric_progression_equality_l3187_318760

/-- Given a geometric progression a, b, c, d, prove that 
    (a^2 + b^2 + c^2)(b^2 + c^2 + d^2) = (ab + bc + cd)^2 -/
theorem geometric_progression_equality 
  (a b c d : ℝ) (h : ∃ (q : ℝ), b = a * q ∧ c = b * q ∧ d = c * q) : 
  (a^2 + b^2 + c^2) * (b^2 + c^2 + d^2) = (a*b + b*c + c*d)^2 := by
  sorry

end geometric_progression_equality_l3187_318760


namespace coat_price_calculation_shopper_pays_112_75_l3187_318733

/-- Calculate the final price of a coat after discounts and tax -/
theorem coat_price_calculation (original_price : ℝ) (initial_discount_percent : ℝ) 
  (additional_discount : ℝ) (sales_tax_percent : ℝ) : ℝ :=
  let price_after_initial_discount := original_price * (1 - initial_discount_percent / 100)
  let price_after_additional_discount := price_after_initial_discount - additional_discount
  let final_price := price_after_additional_discount * (1 + sales_tax_percent / 100)
  final_price

/-- Proof that the shopper pays $112.75 for the coat -/
theorem shopper_pays_112_75 :
  coat_price_calculation 150 25 10 10 = 112.75 := by
  sorry

end coat_price_calculation_shopper_pays_112_75_l3187_318733


namespace wire_length_for_cube_l3187_318784

-- Define the length of one edge of the cube
def edge_length : ℝ := 13

-- Define the number of edges in a cube
def cube_edges : ℕ := 12

-- Theorem stating the total wire length needed for the cube
theorem wire_length_for_cube : edge_length * cube_edges = 156 := by
  sorry

end wire_length_for_cube_l3187_318784


namespace propositions_correctness_l3187_318795

theorem propositions_correctness : 
  (∃ x : ℝ, x^2 ≥ x) ∧ 
  (4 ≥ 3) ∧ 
  ¬(∀ x : ℝ, x^2 ≥ x) ∧
  ¬(∀ x : ℝ, x^2 ≠ 1 ↔ (x ≠ 1 ∨ x ≠ -1)) := by
  sorry

end propositions_correctness_l3187_318795


namespace total_leaves_count_l3187_318747

/-- The number of pots of basil -/
def basil_pots : ℕ := 3

/-- The number of pots of rosemary -/
def rosemary_pots : ℕ := 9

/-- The number of pots of thyme -/
def thyme_pots : ℕ := 6

/-- The number of leaves per basil plant -/
def basil_leaves : ℕ := 4

/-- The number of leaves per rosemary plant -/
def rosemary_leaves : ℕ := 18

/-- The number of leaves per thyme plant -/
def thyme_leaves : ℕ := 30

/-- The total number of leaves from all plants -/
def total_leaves : ℕ := basil_pots * basil_leaves + rosemary_pots * rosemary_leaves + thyme_pots * thyme_leaves

theorem total_leaves_count : total_leaves = 354 := by
  sorry

end total_leaves_count_l3187_318747


namespace intersection_implies_determinant_one_l3187_318712

/-- Given three lines that intersect at one point, prove that the determinant is 1 -/
theorem intersection_implies_determinant_one 
  (a : ℝ) 
  (h1 : ∃ (x y : ℝ), ax + y + 3 = 0 ∧ x + y + 2 = 0 ∧ 2*x - y + 1 = 0) :
  Matrix.det ![![a, 1], ![1, 1]] = 1 := by
sorry

end intersection_implies_determinant_one_l3187_318712


namespace pond_length_l3187_318706

/-- Given a rectangular field with length 24 meters and width 12 meters, 
    containing a square pond whose area is 1/8 of the field's area,
    prove that the length of the pond is 6 meters. -/
theorem pond_length (field_length : ℝ) (field_width : ℝ) (pond_length : ℝ) : 
  field_length = 24 →
  field_width = 12 →
  field_length = 2 * field_width →
  pond_length^2 = (field_length * field_width) / 8 →
  pond_length = 6 :=
by sorry

end pond_length_l3187_318706


namespace stone_volume_l3187_318711

/-- The volume of a stone submerged in a cuboid-shaped container -/
theorem stone_volume (width length initial_height final_height : ℝ) 
  (hw : width = 15) 
  (hl : length = 20) 
  (hi : initial_height = 10) 
  (hf : final_height = 15) : 
  (final_height - initial_height) * width * length = 1500 := by
  sorry

end stone_volume_l3187_318711


namespace parabola_directrix_l3187_318758

-- Define a parabola with equation y^2 = 8x
def parabola (x y : ℝ) : Prop := y^2 = 8 * x

-- Define the directrix of a parabola
def directrix (x : ℝ) : Prop := x = -2

-- Theorem statement
theorem parabola_directrix :
  ∀ (x y : ℝ), parabola x y → directrix x :=
by sorry

end parabola_directrix_l3187_318758


namespace jungkook_item_sum_l3187_318730

theorem jungkook_item_sum : ∀ (a b : ℕ),
  a = 585 →
  a = b + 249 →
  a + b = 921 :=
by
  sorry

end jungkook_item_sum_l3187_318730


namespace circle_intersection_distance_l3187_318716

-- Define the circle M
def CircleM (x y : ℝ) : Prop :=
  x^2 + y^2 - 8*x + 6*y = 0

-- Define the points on the circle
def PointO : ℝ × ℝ := (0, 0)
def PointA : ℝ × ℝ := (1, 1)
def PointB : ℝ × ℝ := (4, 2)

-- Define the intersection points
def PointS : ℝ × ℝ := (8, 0)
def PointT : ℝ × ℝ := (0, -6)

-- Theorem statement
theorem circle_intersection_distance :
  CircleM PointO.1 PointO.2 ∧
  CircleM PointA.1 PointA.2 ∧
  CircleM PointB.1 PointB.2 ∧
  CircleM PointS.1 PointS.2 ∧
  CircleM PointT.1 PointT.2 ∧
  PointS.1 > 0 ∧
  PointT.2 < 0 →
  Real.sqrt ((PointS.1 - PointT.1)^2 + (PointS.2 - PointT.2)^2) = 10 :=
by sorry

end circle_intersection_distance_l3187_318716


namespace eliot_account_balance_l3187_318791

theorem eliot_account_balance (al eliot : ℝ) 
  (h1 : al > eliot)
  (h2 : al - eliot = (1 / 12) * (al + eliot))
  (h3 : 1.1 * al = 1.2 * eliot + 21) :
  eliot = 210 := by
sorry

end eliot_account_balance_l3187_318791


namespace non_equilateral_combinations_count_l3187_318749

/-- The number of dots evenly spaced on the circumference of a circle -/
def num_dots : ℕ := 6

/-- A function that calculates the number of combinations that do not form an equilateral triangle -/
def non_equilateral_combinations (n : ℕ) : ℕ :=
  if n = 1 then num_dots
  else if n = 2 then num_dots.choose 2
  else if n = 3 then num_dots.choose 3 - 2
  else 0

/-- The total number of combinations that do not form an equilateral triangle -/
def total_combinations : ℕ :=
  (non_equilateral_combinations 1) + (non_equilateral_combinations 2) + (non_equilateral_combinations 3)

theorem non_equilateral_combinations_count :
  total_combinations = 18 :=
by sorry

end non_equilateral_combinations_count_l3187_318749


namespace inequality_condition_l3187_318785

theorem inequality_condition (b : ℝ) (h : b > 0) :
  (∃ x : ℝ, |x - 5| + |x - 2| < b) ↔ b > 3 := by
  sorry

end inequality_condition_l3187_318785


namespace square_sum_equals_three_times_product_l3187_318741

theorem square_sum_equals_three_times_product
  (x y : ℝ) 
  (h1 : 1/x + 1/y = 5) 
  (h2 : x + y = 5*x*y) : 
  x^2 + y^2 = 3*x*y :=
by
  sorry

end square_sum_equals_three_times_product_l3187_318741


namespace sum_divides_8n_count_l3187_318778

theorem sum_divides_8n_count : 
  (∃ (S : Finset ℕ), S.card = 4 ∧ 
    (∀ n : ℕ, n > 0 → (n ∈ S ↔ (8 * n) % ((n * (n + 1)) / 2) = 0))) := by
  sorry

end sum_divides_8n_count_l3187_318778


namespace single_layer_cake_cost_l3187_318717

/-- The cost of a single layer cake slice -/
def single_layer_cost : ℝ := 4

/-- The cost of a double layer cake slice -/
def double_layer_cost : ℝ := 7

/-- The number of single layer cake slices bought -/
def single_layer_count : ℕ := 7

/-- The number of double layer cake slices bought -/
def double_layer_count : ℕ := 5

/-- The total amount spent -/
def total_spent : ℝ := 63

theorem single_layer_cake_cost :
  single_layer_cost * single_layer_count + double_layer_cost * double_layer_count = total_spent :=
by sorry

end single_layer_cake_cost_l3187_318717


namespace prob_at_least_one_of_three_l3187_318721

/-- The probability that at least one of three events occurs, given their individual probabilities -/
theorem prob_at_least_one_of_three (pA pB pC : ℝ) 
  (hA : 0 ≤ pA ∧ pA ≤ 1) 
  (hB : 0 ≤ pB ∧ pB ≤ 1) 
  (hC : 0 ≤ pC ∧ pC ≤ 1) 
  (h_pA : pA = 0.8) 
  (h_pB : pB = 0.6) 
  (h_pC : pC = 0.5) : 
  1 - (1 - pA) * (1 - pB) * (1 - pC) = 0.96 := by
sorry

end prob_at_least_one_of_three_l3187_318721


namespace yellow_ball_count_l3187_318703

theorem yellow_ball_count (red_count : ℕ) (total_count : ℕ) 
  (h1 : red_count = 10)
  (h2 : (red_count : ℚ) / total_count = 1 / 3) :
  total_count - red_count = 20 := by
  sorry

end yellow_ball_count_l3187_318703


namespace one_positive_integer_satisfies_condition_l3187_318765

theorem one_positive_integer_satisfies_condition : 
  ∃! (n : ℕ), n > 0 ∧ 21 - 3 * n > 15 :=
sorry

end one_positive_integer_satisfies_condition_l3187_318765


namespace combined_train_length_l3187_318794

/-- Calculates the combined length of two trains given their speeds and passing times. -/
theorem combined_train_length
  (speed_A speed_B speed_bike : ℝ)
  (time_A time_B : ℝ)
  (h1 : speed_A = 120)
  (h2 : speed_B = 100)
  (h3 : speed_bike = 64)
  (h4 : time_A = 75)
  (h5 : time_B = 90)
  (h6 : speed_A > speed_bike)
  (h7 : speed_B > speed_bike)
  (h8 : (speed_A - speed_bike) * time_A / 3600 + (speed_B - speed_bike) * time_B / 3600 = 2.067) :
  (speed_A - speed_bike) * time_A * 1000 / 3600 + (speed_B - speed_bike) * time_B * 1000 / 3600 = 2067 := by
  sorry

#check combined_train_length

end combined_train_length_l3187_318794


namespace roots_inequality_l3187_318740

theorem roots_inequality (m : ℝ) (x₁ x₂ : ℝ) (hm : m < -2) 
  (hx : x₁ < x₂) (hf₁ : Real.log x₁ - x₁ = m) (hf₂ : Real.log x₂ - x₂ = m) :
  x₁ * x₂^2 < 2 := by
  sorry

end roots_inequality_l3187_318740


namespace compound_molecular_weight_l3187_318782

/-- Calculates the molecular weight of a compound given its composition and atomic weights -/
def molecularWeight (fe_count o_count ca_count c_count : ℕ) 
                    (fe_weight o_weight ca_weight c_weight : ℝ) : ℝ :=
  fe_count * fe_weight + o_count * o_weight + ca_count * ca_weight + c_count * c_weight

/-- Theorem stating that the molecular weight of the given compound is 223.787 amu -/
theorem compound_molecular_weight :
  let fe_count : ℕ := 2
  let o_count : ℕ := 3
  let ca_count : ℕ := 1
  let c_count : ℕ := 2
  let fe_weight : ℝ := 55.845
  let o_weight : ℝ := 15.999
  let ca_weight : ℝ := 40.078
  let c_weight : ℝ := 12.011
  molecularWeight fe_count o_count ca_count c_count fe_weight o_weight ca_weight c_weight = 223.787 := by
  sorry

end compound_molecular_weight_l3187_318782


namespace complex_number_properties_l3187_318776

def complex_number (m : ℝ) : ℂ := (m^2 - 5*m + 6 : ℝ) + (m^2 - 3*m : ℝ) * Complex.I

theorem complex_number_properties :
  (∀ m : ℝ, (complex_number m).im = 0 ↔ m = 0 ∨ m = 3) ∧
  (∀ m : ℝ, (complex_number m).re = 0 ↔ m = 2) := by
  sorry

end complex_number_properties_l3187_318776


namespace hongfu_supermarket_salt_purchase_l3187_318761

/-- The number of bags of salt initially purchased by Hongfu Supermarket -/
def initial_salt : ℕ := 1200

/-- The fraction of salt sold in the first month -/
def first_month_sold : ℚ := 2/5

/-- The number of bags of salt sold in the second month -/
def second_month_sold : ℕ := 420

/-- The ratio of sold salt to remaining salt after the second month -/
def sold_to_remaining_ratio : ℚ := 3

theorem hongfu_supermarket_salt_purchase :
  initial_salt = 1200 ∧
  (initial_salt : ℚ) * first_month_sold + second_month_sold =
    sold_to_remaining_ratio * (initial_salt - (initial_salt : ℚ) * first_month_sold - second_month_sold) :=
by sorry

end hongfu_supermarket_salt_purchase_l3187_318761


namespace janelles_blue_marbles_gift_l3187_318704

/-- Calculates the number of blue marbles Janelle gave to her friend --/
def blue_marbles_given (initial_green : ℕ) (blue_bags : ℕ) (marbles_per_bag : ℕ) 
  (green_given : ℕ) (total_remaining : ℕ) : ℕ :=
  let total_blue := blue_bags * marbles_per_bag
  let total_before_gift := initial_green + total_blue
  let total_given := total_before_gift - total_remaining
  total_given - green_given

/-- Proves that Janelle gave 8 blue marbles to her friend --/
theorem janelles_blue_marbles_gift : 
  blue_marbles_given 26 6 10 6 72 = 8 := by
  sorry

end janelles_blue_marbles_gift_l3187_318704


namespace equation_solution_l3187_318783

theorem equation_solution (x : ℝ) (h : x ≥ -1) :
  Real.sqrt (x + 1) - 1 = x / (Real.sqrt (x + 1) + 1) := by
  sorry

#check equation_solution

end equation_solution_l3187_318783


namespace rectangle_perimeter_width_ratio_l3187_318731

/-- Given a rectangle with area 150 square centimeters and length 15 centimeters,
    prove that the ratio of its perimeter to its width is 5:1 -/
theorem rectangle_perimeter_width_ratio 
  (area : ℝ) (length : ℝ) (width : ℝ) (perimeter : ℝ) :
  area = 150 →
  length = 15 →
  area = length * width →
  perimeter = 2 * (length + width) →
  perimeter / width = 5 := by
sorry

end rectangle_perimeter_width_ratio_l3187_318731


namespace dice_probability_l3187_318708

theorem dice_probability (p_neither : ℚ) (h : p_neither = 4/9) : 
  1 - p_neither = 5/9 := by
  sorry

end dice_probability_l3187_318708


namespace locus_of_centers_l3187_318779

-- Define the circles C₁ and C₂
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 4
def C₂ (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 9

-- Define external tangency to C₁
def externally_tangent_C₁ (a b r : ℝ) : Prop := a^2 + b^2 = (r + 2)^2

-- Define internal tangency to C₂
def internally_tangent_C₂ (a b r : ℝ) : Prop := (a - 1)^2 + b^2 = (3 - r)^2

theorem locus_of_centers (a b : ℝ) : 
  (∃ r : ℝ, externally_tangent_C₁ a b r ∧ internally_tangent_C₂ a b r) → 
  4 * a^2 + 4 * b^2 - 25 = 0 := by
sorry

end locus_of_centers_l3187_318779


namespace babysitting_earnings_proof_l3187_318799

/-- Calculates earnings from babysitting given net profit, lemonade stand revenue, and operating cost -/
def earnings_from_babysitting (net_profit : ℕ) (lemonade_revenue : ℕ) (operating_cost : ℕ) : ℕ :=
  (net_profit + operating_cost) - lemonade_revenue

/-- Proves that earnings from babysitting equal $31 given the specific values -/
theorem babysitting_earnings_proof :
  earnings_from_babysitting 44 47 34 = 31 := by
sorry

end babysitting_earnings_proof_l3187_318799


namespace student_A_wrong_l3187_318754

-- Define the circle
def circle_center : ℝ × ℝ := (2, -3)
def circle_radius : ℝ := 5

-- Define the points
def point_D : ℝ × ℝ := (5, 1)
def point_A : ℝ × ℝ := (-2, -1)

-- Function to check if a point is on the circle
def is_on_circle (p : ℝ × ℝ) : Prop :=
  (p.1 - circle_center.1)^2 + (p.2 - circle_center.2)^2 = circle_radius^2

-- Theorem statement
theorem student_A_wrong :
  is_on_circle point_D ∧ ¬is_on_circle point_A :=
sorry

end student_A_wrong_l3187_318754


namespace mike_initial_cards_l3187_318709

theorem mike_initial_cards (sold : ℕ) (current : ℕ) (h1 : sold = 13) (h2 : current = 74) :
  current + sold = 87 := by
  sorry

end mike_initial_cards_l3187_318709


namespace circle_condition_l3187_318796

theorem circle_condition (m : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 - x + y + m = 0 ∧ 
   ∃ (h k r : ℝ), r > 0 ∧ ∀ (x y : ℝ), (x - h)^2 + (y - k)^2 = r^2 ↔ x^2 + y^2 - x + y + m = 0) 
  → m < (1/2 : ℝ) :=
by sorry

end circle_condition_l3187_318796


namespace angle_properties_l3187_318775

/-- Given an angle α whose terminal side passes through the point (sin(5π/6), cos(5π/6)),
    prove that α is in the fourth quadrant and the smallest positive angle with the same
    terminal side as α is 5π/3 -/
theorem angle_properties (α : Real) :
  (∃ r : Real, r > 0 ∧ r * Real.cos α = Real.cos (5 * Real.pi / 6) ∧
                    r * Real.sin α = Real.sin (5 * Real.pi / 6)) →
  (Real.cos α > 0 ∧ Real.sin α < 0) ∧
  (∀ β : Real, β > 0 ∧ Real.cos β = Real.cos α ∧ Real.sin β = Real.sin α → β ≥ 5 * Real.pi / 3) ∧
  (Real.cos (5 * Real.pi / 3) = Real.cos α ∧ Real.sin (5 * Real.pi / 3) = Real.sin α) :=
by sorry

end angle_properties_l3187_318775


namespace smallest_number_quotient_remainder_difference_l3187_318798

theorem smallest_number_quotient_remainder_difference : ∃ (n : ℕ), 
  (n > 0) ∧ 
  (n % 5 = 0) ∧
  (n / 5 > n % 34) ∧
  (∀ m : ℕ, m > 0 → m % 5 = 0 → m / 5 > m % 34 → m ≥ n) ∧
  (n / 5 - n % 34 = 8) := by
sorry

end smallest_number_quotient_remainder_difference_l3187_318798


namespace externally_tangent_circles_l3187_318719

theorem externally_tangent_circles (m : ℝ) : 
  let C₁ := {(x, y) : ℝ × ℝ | (x - m)^2 + (y + 2)^2 = 9}
  let C₂ := {(x, y) : ℝ × ℝ | (x + 1)^2 + (y - m)^2 = 4}
  (∃ (p : ℝ × ℝ), p ∈ C₁ ∧ p ∈ C₂ ∧ 
    (∀ (q : ℝ × ℝ), q ∈ C₁ ∧ q ∈ C₂ → q = p) ∧
    (∀ (r : ℝ × ℝ), r ∈ C₁ → ∃ (s : ℝ × ℝ), s ∈ C₂ ∧ s ≠ r)) →
  m = 2 ∨ m = -5 :=
by sorry

end externally_tangent_circles_l3187_318719


namespace set_cardinality_lower_bound_l3187_318797

variable (m : ℕ) (A : Finset ℤ) (B : Fin m → Finset ℤ)

theorem set_cardinality_lower_bound
  (h_m : m ≥ 2)
  (h_subset : ∀ i : Fin m, B i ⊆ A)
  (h_sum : ∀ i : Fin m, (B i).sum id = m ^ (i : ℕ).succ) :
  A.card ≥ m / 2 :=
sorry

end set_cardinality_lower_bound_l3187_318797


namespace initial_goldfish_count_l3187_318732

theorem initial_goldfish_count (died : ℕ) (remaining : ℕ) (h1 : died = 32) (h2 : remaining = 57) :
  died + remaining = 89 := by
  sorry

end initial_goldfish_count_l3187_318732


namespace zongzi_problem_l3187_318780

-- Define the types of zongzi gift boxes
inductive ZongziType
| RedDate
| EggYolk

-- Define the price and quantity of a zongzi gift box
structure ZongziBox where
  type : ZongziType
  price : ℕ
  quantity : ℕ

-- Define the problem parameters
def total_boxes : ℕ := 8
def max_cost : ℕ := 300
def total_recipients : ℕ := 65

-- Define the conditions of the problem
axiom price_relation : ∀ (rd : ZongziBox) (ey : ZongziBox),
  rd.type = ZongziType.RedDate → ey.type = ZongziType.EggYolk →
  3 * rd.price = 4 * ey.price

axiom combined_cost : ∀ (rd : ZongziBox) (ey : ZongziBox),
  rd.type = ZongziType.RedDate → ey.type = ZongziType.EggYolk →
  rd.price + 2 * ey.price = 100

axiom red_date_quantity : ∀ (rd : ZongziBox),
  rd.type = ZongziType.RedDate → rd.quantity = 10

axiom egg_yolk_quantity : ∀ (ey : ZongziBox),
  ey.type = ZongziType.EggYolk → ey.quantity = 6

-- Define the theorem to be proved
theorem zongzi_problem :
  ∃ (rd : ZongziBox) (ey : ZongziBox) (rd_count ey_count : ℕ),
    rd.type = ZongziType.RedDate ∧
    ey.type = ZongziType.EggYolk ∧
    rd.price = 40 ∧
    ey.price = 30 ∧
    rd_count = 5 ∧
    ey_count = 3 ∧
    rd_count + ey_count = total_boxes ∧
    rd_count * rd.price + ey_count * ey.price < max_cost ∧
    rd_count * rd.quantity + ey_count * ey.quantity ≥ total_recipients :=
  sorry


end zongzi_problem_l3187_318780


namespace product_odd_implies_sum_even_l3187_318751

theorem product_odd_implies_sum_even (a b : ℤ) : 
  Odd (a * b) → Even (a + b) := by
  sorry

end product_odd_implies_sum_even_l3187_318751


namespace speak_both_languages_l3187_318720

theorem speak_both_languages (total : ℕ) (latin : ℕ) (french : ℕ) (neither : ℕ) 
  (h_total : total = 25)
  (h_latin : latin = 13)
  (h_french : french = 15)
  (h_neither : neither = 6) :
  latin + french - (total - neither) = 9 := by
  sorry

end speak_both_languages_l3187_318720


namespace product_of_powers_l3187_318734

theorem product_of_powers (x y : ℝ) : -x^2 * y^3 * (2 * x * y^2) = -2 * x^3 * y^5 := by
  sorry

end product_of_powers_l3187_318734


namespace marathon_day3_miles_l3187_318769

/-- Represents the marathon runner's training schedule over 3 days -/
structure MarathonTraining where
  total_miles : ℝ
  day1_percent : ℝ
  day2_percent : ℝ

/-- Calculates the miles run on day 3 given the training schedule -/
def miles_on_day3 (mt : MarathonTraining) : ℝ :=
  mt.total_miles - (mt.total_miles * mt.day1_percent) - ((mt.total_miles - (mt.total_miles * mt.day1_percent)) * mt.day2_percent)

/-- Theorem stating that given the specific training schedule, the miles run on day 3 is 28 -/
theorem marathon_day3_miles :
  let mt : MarathonTraining := ⟨70, 0.2, 0.5⟩
  miles_on_day3 mt = 28 := by
  sorry

end marathon_day3_miles_l3187_318769


namespace max_receptivity_and_duration_receptivity_comparison_insufficient_duration_l3187_318773

-- Define the piecewise function f(x)
noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 10 then -0.1 * x^2 + 2.6 * x + 44
  else if 10 < x ∧ x ≤ 15 then 60
  else if 15 < x ∧ x ≤ 25 then -3 * x + 105
  else if 25 < x ∧ x ≤ 40 then 30
  else 0

-- Theorem 1: Maximum receptivity and duration
theorem max_receptivity_and_duration :
  (∀ x, 0 < x → x ≤ 40 → f x ≤ 60) ∧
  (∀ x, 10 ≤ x → x ≤ 15 → f x = 60) :=
sorry

-- Theorem 2: Receptivity comparison
theorem receptivity_comparison :
  f 5 > f 20 ∧ f 20 > f 35 :=
sorry

-- Theorem 3: Insufficient duration for required receptivity
theorem insufficient_duration :
  ¬ ∃ a : ℝ, 0 < a ∧ a + 12 ≤ 40 ∧ ∀ x, a ≤ x → x ≤ a + 12 → f x ≥ 56 :=
sorry

end max_receptivity_and_duration_receptivity_comparison_insufficient_duration_l3187_318773


namespace plate_cutting_theorem_l3187_318715

def can_measure (weights : List ℕ) (target : ℕ) : Prop :=
  ∃ (pos neg : List ℕ), pos.sum - neg.sum = target ∧ pos.toFinset ∪ neg.toFinset ⊆ weights.toFinset

theorem plate_cutting_theorem :
  let weights := [1, 3, 7]
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 11 → can_measure weights n :=
by sorry

end plate_cutting_theorem_l3187_318715


namespace triangle_value_l3187_318748

theorem triangle_value (p : ℤ) (h1 : ∃ triangle : ℤ, triangle + p = 67) 
  (h2 : ∃ triangle : ℤ, 3 * (triangle + p) - p = 185) : 
  ∃ triangle : ℤ, triangle = 51 := by
sorry

end triangle_value_l3187_318748


namespace black_burger_cost_l3187_318792

theorem black_burger_cost (salmon_cost chicken_cost total_bill : ℝ) 
  (h1 : salmon_cost = 40)
  (h2 : chicken_cost = 25)
  (h3 : total_bill = 92) : 
  ∃ (burger_cost : ℝ), 
    burger_cost = 15 ∧ 
    total_bill = (salmon_cost + burger_cost + chicken_cost) * 1.15 := by
  sorry

end black_burger_cost_l3187_318792


namespace unique_c_value_l3187_318787

theorem unique_c_value (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_order : a < b ∧ b < c) (h_sum : a + b + c = 11)
  (h_frac : (1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c = 1) : c = 6 := by
  sorry

end unique_c_value_l3187_318787


namespace base_conversion_2450_l3187_318714

/-- Converts a base-10 number to its base-8 representation -/
def toBase8 (n : ℕ) : ℕ := sorry

/-- Converts a base-8 number to its base-10 representation -/
def fromBase8 (n : ℕ) : ℕ := sorry

theorem base_conversion_2450 :
  toBase8 2450 = 4622 ∧ fromBase8 4622 = 2450 := by sorry

end base_conversion_2450_l3187_318714


namespace angle_ADE_measure_l3187_318781

/-- Triangle ABC -/
structure Triangle :=
  (A B C : ℝ)
  (sum_angles : A + B + C = 180)

/-- Pentagon ABCDE -/
structure Pentagon :=
  (A B C D E : ℝ)
  (sum_angles : A + B + C + D + E = 540)

/-- Circle circumscribed around a pentagon -/
structure CircumscribedCircle (p : Pentagon) := 
  (is_circumscribed : Bool)

/-- Pentagon with sides tangent to a circle -/
structure TangentPentagon (p : Pentagon) (c : CircumscribedCircle p) :=
  (is_tangent : Bool)

/-- Theorem: In a pentagon ABCDE constructed as described, the measure of angle ADE is 108° -/
theorem angle_ADE_measure 
  (t : Triangle)
  (p : Pentagon)
  (c : CircumscribedCircle p)
  (tp : TangentPentagon p c)
  (h1 : t.A = 60)
  (h2 : t.B = 50)
  (h3 : t.C = 70)
  (h4 : p.D ∈ Set.Ioo 0 (t.A + t.B))  -- D is on side AB
  (h5 : p.E ∈ Set.Ioo (t.A + t.B) (t.A + t.B + t.C))  -- E is on side BC
  : p.D = 108 :=
sorry

end angle_ADE_measure_l3187_318781


namespace gcd_from_lcm_and_ratio_l3187_318700

theorem gcd_from_lcm_and_ratio (A B : ℕ+) : 
  Nat.lcm A B = 180 → 
  (A : ℚ) / B = 2 / 5 → 
  Nat.gcd A B = 18 := by
sorry

end gcd_from_lcm_and_ratio_l3187_318700


namespace cosine_inequality_l3187_318728

theorem cosine_inequality (x y : Real) :
  x ∈ Set.Icc 0 (Real.pi / 2) →
  y ∈ Set.Icc 0 (Real.pi / 2) →
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), Real.cos (x + y) ≥ Real.cos x * Real.cos y) ↔
  y = 0 := by
sorry

end cosine_inequality_l3187_318728


namespace screen_paper_difference_l3187_318767

/-- The perimeter of a square-shaped piece of paper is shorter than the height of a computer screen. 
    The height of the screen is 100 cm, and the side of the square paper is 20 cm. 
    This theorem proves that the difference between the screen height and the paper perimeter is 20 cm. -/
theorem screen_paper_difference (screen_height paper_side : ℝ) 
  (h1 : screen_height = 100)
  (h2 : paper_side = 20)
  (h3 : 4 * paper_side < screen_height) : 
  screen_height - 4 * paper_side = 20 := by
  sorry

end screen_paper_difference_l3187_318767


namespace paintbrush_cost_l3187_318742

/-- The cost of each paintbrush given Marc's purchases -/
theorem paintbrush_cost (model_cars : ℕ) (car_cost : ℕ) (paint_bottles : ℕ) (paint_cost : ℕ) 
  (paintbrushes : ℕ) (total_spent : ℕ) : 
  model_cars = 5 → 
  car_cost = 20 → 
  paint_bottles = 5 → 
  paint_cost = 10 → 
  paintbrushes = 5 → 
  total_spent = 160 → 
  (total_spent - (model_cars * car_cost + paint_bottles * paint_cost)) / paintbrushes = 2 := by
  sorry

#check paintbrush_cost

end paintbrush_cost_l3187_318742


namespace field_B_most_stable_l3187_318770

-- Define the variances for each field
def variance_A : ℝ := 3.6
def variance_B : ℝ := 2.89
def variance_C : ℝ := 13.4
def variance_D : ℝ := 20.14

-- Define a function to compare two variances
def is_more_stable (v1 v2 : ℝ) : Prop := v1 < v2

-- Theorem stating that Field B has the lowest variance
theorem field_B_most_stable :
  is_more_stable variance_B variance_A ∧
  is_more_stable variance_B variance_C ∧
  is_more_stable variance_B variance_D :=
by sorry

end field_B_most_stable_l3187_318770


namespace root_product_reciprocal_sum_l3187_318726

theorem root_product_reciprocal_sum (p q : ℝ) 
  (h1 : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x^2 + p*x + q = (x - x1) * (x - x2))
  (h2 : ∃ x3 x4 : ℝ, x3 ≠ x4 ∧ x^2 + q*x + p = (x - x3) * (x - x4))
  (h3 : ∀ x1 x2 x3 x4 : ℝ, 
    (x^2 + p*x + q = (x - x1) * (x - x2)) → 
    (x^2 + q*x + p = (x - x3) * (x - x4)) → 
    x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4) :
  ∃ x1 x2 x3 x4 : ℝ, 
    (x^2 + p*x + q = (x - x1) * (x - x2)) ∧
    (x^2 + q*x + p = (x - x3) * (x - x4)) ∧
    1 / (x1 * x3) + 1 / (x1 * x4) + 1 / (x2 * x3) + 1 / (x2 * x4) = 1 :=
by sorry

end root_product_reciprocal_sum_l3187_318726


namespace rectangle_properties_l3187_318756

/-- A rectangle with one side of length 8 and another of length x -/
structure Rectangle where
  x : ℝ
  h_positive : x > 0

/-- The perimeter of the rectangle -/
def perimeter (rect : Rectangle) : ℝ := 2 * (8 + rect.x)

/-- The area of the rectangle -/
def area (rect : Rectangle) : ℝ := 8 * rect.x

theorem rectangle_properties (rect : Rectangle) :
  (perimeter rect = 16 + 2 * rect.x) ∧
  (area rect = 8 * rect.x) ∧
  (area rect = 80 → perimeter rect = 36) := by
  sorry


end rectangle_properties_l3187_318756


namespace choose_president_vice_president_l3187_318745

/-- The number of boys in the club -/
def num_boys : ℕ := 12

/-- The number of girls in the club -/
def num_girls : ℕ := 12

/-- The total number of members in the club -/
def total_members : ℕ := num_boys + num_girls

/-- The number of ways to choose a president and vice-president of opposite genders -/
def ways_to_choose : ℕ := num_boys * num_girls * 2

theorem choose_president_vice_president :
  ways_to_choose = 288 :=
by sorry

end choose_president_vice_president_l3187_318745


namespace derivative_of_x4_minus_7_l3187_318707

theorem derivative_of_x4_minus_7 (x : ℝ) :
  deriv (fun x => x^4 - 7) x = 4 * x^3 - 7 := by
  sorry

end derivative_of_x4_minus_7_l3187_318707


namespace ryan_marble_distribution_l3187_318768

theorem ryan_marble_distribution (total_marbles : ℕ) (marbles_per_friend : ℕ) (num_friends : ℕ) :
  total_marbles = 72 →
  marbles_per_friend = 8 →
  total_marbles = marbles_per_friend * num_friends →
  num_friends = 9 := by
sorry

end ryan_marble_distribution_l3187_318768


namespace average_weight_b_c_l3187_318743

/-- Given the weights of three people a, b, and c, prove that the average weight of b and c is 42 kg -/
theorem average_weight_b_c (a b c : ℝ) : 
  (a + b + c) / 3 = 43 →  -- The average weight of a, b, and c is 43 kg
  (a + b) / 2 = 48 →      -- The average weight of a and b is 48 kg
  b = 51 →                -- The weight of b is 51 kg
  (b + c) / 2 = 42 :=     -- The average weight of b and c is 42 kg
by
  sorry

end average_weight_b_c_l3187_318743


namespace odd_perfect_square_l3187_318723

theorem odd_perfect_square (n : ℕ+) 
  (h : (Finset.sum (Nat.divisors n.val) id) = 2 * n.val + 1) : 
  ∃ (k : ℕ), n.val = 2 * k + 1 ∧ ∃ (m : ℕ), n.val = m ^ 2 :=
sorry

end odd_perfect_square_l3187_318723


namespace class_average_theorem_l3187_318755

theorem class_average_theorem (total_students : ℕ) 
                               (excluded_students : ℕ) 
                               (excluded_average : ℝ) 
                               (remaining_average : ℝ) : 
  total_students = 56 →
  excluded_students = 8 →
  excluded_average = 20 →
  remaining_average = 90 →
  (total_students * (total_students * remaining_average - excluded_students * remaining_average + excluded_students * excluded_average)) / 
  (total_students * total_students) = 80 := by
sorry

end class_average_theorem_l3187_318755


namespace book_ratio_l3187_318757

def book_tournament (candice amanda kara patricia taylor : ℕ) : Prop :=
  candice = 3 * amanda ∧
  kara = amanda / 2 ∧
  patricia = 7 * kara ∧
  taylor = (candice + amanda + kara + patricia) / 4 ∧
  candice = 18

theorem book_ratio (candice amanda kara patricia taylor : ℕ) :
  book_tournament candice amanda kara patricia taylor →
  taylor * 5 = candice + amanda + kara + patricia + taylor :=
by sorry

end book_ratio_l3187_318757


namespace b_fraction_of_a_and_c_l3187_318718

def total_amount : ℕ := 1800

def a_share : ℕ := 600

theorem b_fraction_of_a_and_c (b_share c_share : ℕ) 
  (h1 : a_share = (2 : ℕ) * (b_share + c_share) / 5)
  (h2 : total_amount = a_share + b_share + c_share) :
  b_share * 6 = a_share + c_share :=
by sorry

end b_fraction_of_a_and_c_l3187_318718


namespace cube_surface_area_increase_l3187_318746

theorem cube_surface_area_increase :
  ∀ s : ℝ, s > 0 →
  let original_surface_area := 6 * s^2
  let new_edge_length := 1.4 * s
  let new_surface_area := 6 * new_edge_length^2
  (new_surface_area - original_surface_area) / original_surface_area = 0.96 :=
by
  sorry

#check cube_surface_area_increase

end cube_surface_area_increase_l3187_318746


namespace kite_diagonal_length_l3187_318710

/-- A rectangle ABCD with a kite WXYZ inscribed -/
structure RectangleWithKite where
  /-- Length of side AB -/
  ab : ℝ
  /-- Length of side BC -/
  bc : ℝ
  /-- Distance from A to W on AB -/
  aw : ℝ
  /-- Distance from C to Y on CD -/
  cy : ℝ
  /-- AB = CD = 5 -/
  h_ab : ab = 5
  /-- BC = AD = 10 -/
  h_bc : bc = 10
  /-- WX = WZ = √13 -/
  h_wx : aw ^ 2 + cy ^ 2 = 13
  /-- XY = ZY -/
  h_xy_zy : (bc - aw) ^ 2 + cy ^ 2 = (ab - cy) ^ 2 + aw ^ 2

/-- The length of XY in the kite WXYZ is √65 -/
theorem kite_diagonal_length (r : RectangleWithKite) : 
  (r.bc - r.aw) ^ 2 + r.cy ^ 2 = 65 := by
  sorry


end kite_diagonal_length_l3187_318710


namespace C_always_answers_yes_l3187_318759

-- Define the type of islander
inductive IslanderType
  | Knight
  | Liar

-- Define the islanders
def A : IslanderType := sorry
def B : IslanderType := sorry
def C : IslanderType := sorry

-- Define A's statement
def A_statement : Prop := (B = C)

-- Define the question asked to C
def question_to_C : Prop := (A = B)

-- Define C's answer
def C_answer : Prop := 
  match C with
  | IslanderType.Knight => question_to_C
  | IslanderType.Liar => ¬question_to_C

-- Theorem: C will always answer "Yes"
theorem C_always_answers_yes :
  ∀ (A B C : IslanderType),
  (A_statement ↔ (B = C)) →
  C_answer = true :=
sorry

end C_always_answers_yes_l3187_318759


namespace tetrahedron_formation_condition_l3187_318722

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a square with side length 2 -/
structure Square where
  A : Point2D
  B : Point2D
  C : Point2D
  D : Point2D

/-- Condition for forming a tetrahedron by folding triangles in a square -/
def canFormTetrahedron (s : Square) (x : ℝ) : Prop :=
  let E : Point2D := { x := (s.A.x + s.B.x) / 2, y := (s.A.y + s.B.y) / 2 }
  let F : Point2D := { x := s.B.x + x, y := s.B.y }
  let EA' := 1
  let EF := Real.sqrt (1 + x^2)
  let FA' := 2 - x
  EA' + EF > FA' ∧ EF + FA' > EA' ∧ FA' + EA' > EF

theorem tetrahedron_formation_condition (s : Square) :
  (∀ x, canFormTetrahedron s x ↔ 0 < x ∧ x < 4/3) :=
by sorry

end tetrahedron_formation_condition_l3187_318722


namespace chord_length_l3187_318737

/-- The length of the chord formed by the intersection of a circle and a line --/
theorem chord_length (x y : ℝ) : 
  let circle := (fun (x y : ℝ) ↦ (x - 1)^2 + y^2 = 4)
  let line := (fun (x y : ℝ) ↦ x + y + 1 = 0)
  let chord_length := 
    Real.sqrt (8 - 2 * ((1 * 1 + 1 * 0 + 1) / Real.sqrt (1^2 + 1^2))^2)
  (∃ (a b : ℝ × ℝ), circle a.1 a.2 ∧ circle b.1 b.2 ∧ 
                     line a.1 a.2 ∧ line b.1 b.2 ∧ 
                     a ≠ b) →
  chord_length = 2 * Real.sqrt 2 := by
sorry


end chord_length_l3187_318737


namespace square_diagonal_length_l3187_318766

/-- The length of the diagonal of a square with side length 50√2 cm is 100 cm. -/
theorem square_diagonal_length :
  let side_length : ℝ := 50 * Real.sqrt 2
  let diagonal_length : ℝ := 100
  diagonal_length = Real.sqrt (2 * side_length ^ 2) :=
by sorry

end square_diagonal_length_l3187_318766


namespace complex_product_pure_imaginary_l3187_318790

-- Define the imaginary unit
noncomputable def i : ℂ := Complex.I

-- Define the property of being a pure imaginary number
def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

-- Theorem statement
theorem complex_product_pure_imaginary (a : ℝ) :
  is_pure_imaginary ((1 + i) * (1 + a * i)) → a = 1 := by
  sorry

end complex_product_pure_imaginary_l3187_318790


namespace data_analytics_course_hours_l3187_318788

/-- Represents the total hours spent on a data analytics course -/
def course_total_hours (weeks : ℕ) (weekly_class_hours : ℕ) (weekly_homework_hours : ℕ) 
  (lab_sessions : ℕ) (lab_session_hours : ℕ) (project_hours : List ℕ) : ℕ :=
  weeks * (weekly_class_hours + weekly_homework_hours) + 
  lab_sessions * lab_session_hours + 
  project_hours.sum

/-- Theorem stating the total hours spent on the specific data analytics course -/
theorem data_analytics_course_hours : 
  course_total_hours 24 10 4 8 6 [10, 14, 18] = 426 := by
  sorry

end data_analytics_course_hours_l3187_318788


namespace min_area_triangle_abc_l3187_318793

/-- Triangle ABC with A at origin, B at (48,18), and C with integer coordinates has minimum area 3 -/
theorem min_area_triangle_abc : 
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (48, 18)
  ∃ (min_area : ℝ), min_area = 3 ∧ 
    ∀ (C : ℤ × ℤ), 
      let area := (1/2) * |A.1 * (B.2 - C.2) + B.1 * C.2 + C.1 * A.2 - (B.2 * C.1 + A.1 * B.2 + C.2 * A.1)|
      area ≥ min_area :=
by
  sorry


end min_area_triangle_abc_l3187_318793


namespace meal_price_calculation_l3187_318739

/-- Calculates the total price of a meal including tip -/
theorem meal_price_calculation (appetizer_cost entree_cost dessert_cost : ℚ)
  (num_entrees : ℕ) (tip_percentage : ℚ) :
  appetizer_cost = 9 ∧ 
  entree_cost = 20 ∧ 
  num_entrees = 2 ∧
  dessert_cost = 11 ∧
  tip_percentage = 30 / 100 →
  appetizer_cost + num_entrees * entree_cost + dessert_cost + 
  (appetizer_cost + num_entrees * entree_cost + dessert_cost) * tip_percentage = 78 := by
  sorry

end meal_price_calculation_l3187_318739


namespace constant_value_l3187_318724

-- Define the function f
def f (x : ℝ) : ℝ := x + 4

-- Define the equation
def equation (x : ℝ) (c : ℝ) : Prop :=
  (3 * f (x - 2)) / f 0 + c = f (2 * x + 1)

-- Theorem statement
theorem constant_value :
  ∃ (c : ℝ), equation 0.4 c ∧ ∀ (x : ℝ), equation x c → x = 0.4 :=
by sorry

end constant_value_l3187_318724


namespace late_time_calculation_l3187_318777

/-- Calculates the total late time for five students given the lateness of one student and the additional lateness of the other four. -/
def totalLateTime (firstStudentLateness : ℕ) (additionalLateness : ℕ) : ℕ :=
  firstStudentLateness + 4 * (firstStudentLateness + additionalLateness)

/-- Theorem stating that for the given scenario, the total late time is 140 minutes. -/
theorem late_time_calculation :
  totalLateTime 20 10 = 140 := by
  sorry

end late_time_calculation_l3187_318777


namespace angle_D_value_l3187_318735

-- Define the angles as real numbers (in degrees)
variable (A B C D : ℝ)

-- State the theorem
theorem angle_D_value 
  (h1 : A + B = 180)
  (h2 : C = 2 * D)
  (h3 : A = 100)
  (h4 : B + C + D = 180) :
  D = 100 / 3 := by
  sorry

end angle_D_value_l3187_318735


namespace percentage_difference_l3187_318786

theorem percentage_difference (a b : ℝ) 
  (ha : 3 = 0.15 * a) 
  (hb : 3 = 0.25 * b) : 
  a - b = 8 := by
  sorry

end percentage_difference_l3187_318786


namespace karen_ham_sandwich_days_l3187_318763

/-- The number of school days in a week -/
def school_days : ℕ := 5

/-- The number of days Karen packs peanut butter sandwiches -/
def peanut_butter_days : ℕ := 2

/-- The number of days Karen packs cake -/
def cake_days : ℕ := 1

/-- The probability of packing a ham sandwich and cake on the same day -/
def prob_ham_and_cake : ℚ := 12 / 100

/-- The number of days Karen packs ham sandwiches -/
def ham_days : ℕ := school_days - peanut_butter_days

theorem karen_ham_sandwich_days :
  ham_days = 3 ∧
  (ham_days : ℚ) / school_days * (cake_days : ℚ) / school_days = prob_ham_and_cake :=
sorry

end karen_ham_sandwich_days_l3187_318763


namespace polynomial_coefficient_theorem_l3187_318702

theorem polynomial_coefficient_theorem (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, a₀ + a₁ * (2 * x - 1) + a₂ * (2 * x - 1)^2 + a₃ * (2 * x - 1)^3 + 
             a₄ * (2 * x - 1)^4 + a₅ * (2 * x - 1)^5 = x^5) →
  a₂ = 5/16 := by
sorry

end polynomial_coefficient_theorem_l3187_318702


namespace mc_question_time_l3187_318744

-- Define the total number of questions
def total_questions : ℕ := 60

-- Define the number of multiple-choice questions
def mc_questions : ℕ := 30

-- Define the number of fill-in-the-blank questions
def fib_questions : ℕ := 30

-- Define the time to learn each fill-in-the-blank question (in minutes)
def fib_time : ℕ := 25

-- Define the total study time (in minutes)
def total_study_time : ℕ := 20 * 60

-- Define the function to calculate the time for multiple-choice questions
def mc_time (x : ℕ) : ℕ := x * mc_questions

-- Define the function to calculate the time for fill-in-the-blank questions
def fib_total_time : ℕ := fib_questions * fib_time

-- Theorem: The time to learn each multiple-choice question is 15 minutes
theorem mc_question_time : 
  ∃ (x : ℕ), mc_time x + fib_total_time = total_study_time ∧ x = 15 :=
by sorry

end mc_question_time_l3187_318744


namespace compound_proposition_truth_l3187_318729

theorem compound_proposition_truth (p q : Prop) 
  (h : ¬(¬p ∨ ¬q)) : (p ∧ q) ∧ (p ∨ q) := by
  sorry

end compound_proposition_truth_l3187_318729


namespace min_value_reciprocal_product_l3187_318750

theorem min_value_reciprocal_product (a b : ℝ) : 
  a > 0 → b > 0 → 2*a + b = 4 → (∀ x y : ℝ, x > 0 → y > 0 → 2*x + y = 4 → 1/(a*b) ≤ 1/(x*y)) ∧ (∃ a₀ b₀ : ℝ, a₀ > 0 ∧ b₀ > 0 ∧ 2*a₀ + b₀ = 4 ∧ 1/(a₀*b₀) = 1/2) :=
by sorry

end min_value_reciprocal_product_l3187_318750


namespace m_eq_one_sufficient_not_necessary_l3187_318762

/-- A function f(x) = ax^2 is a power function if a = 1 -/
def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, a = 1 ∧ ∀ x, f x = a * x^2

/-- The function f(x) = (m^2 - 4m + 4)x^2 -/
def f (m : ℝ) : ℝ → ℝ := λ x ↦ (m^2 - 4*m + 4) * x^2

/-- Theorem: m = 1 is sufficient but not necessary for f to be a power function -/
theorem m_eq_one_sufficient_not_necessary :
  (∃ m : ℝ, m = 1 → is_power_function (f m)) ∧
  ¬(∀ m : ℝ, is_power_function (f m) → m = 1) :=
by sorry

end m_eq_one_sufficient_not_necessary_l3187_318762


namespace circle_radius_is_two_l3187_318713

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 8*x + y^2 - 4*y + 16 = 0

-- Define the center of the circle
def circle_center : ℝ × ℝ := (4, 2)

-- Define the radius of the circle
def circle_radius : ℝ := 2

-- Theorem statement
theorem circle_radius_is_two :
  ∀ x y : ℝ, circle_equation x y →
  (x - circle_center.1)^2 + (y - circle_center.2)^2 = circle_radius^2 :=
by sorry

end circle_radius_is_two_l3187_318713


namespace infinite_perfect_square_phi_and_d_l3187_318774

/-- Euler's totient function -/
def phi (n : ℕ+) : ℕ := sorry

/-- Number of positive divisors function -/
def d (n : ℕ+) : ℕ := sorry

/-- A natural number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

/-- The set of positive integers n for which both φ(n) and d(n) are perfect squares -/
def S : Set ℕ+ := {n : ℕ+ | is_perfect_square (phi n) ∧ is_perfect_square (d n)}

theorem infinite_perfect_square_phi_and_d : Set.Infinite S := by sorry

end infinite_perfect_square_phi_and_d_l3187_318774
