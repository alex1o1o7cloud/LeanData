import Mathlib

namespace NUMINAMATH_CALUDE_arc_length_sixty_degrees_l3897_389796

theorem arc_length_sixty_degrees (r : ℝ) (h : r = 1) :
  let angle : ℝ := π / 3
  let arc_length : ℝ := r * angle
  arc_length = π / 3 := by sorry

end NUMINAMATH_CALUDE_arc_length_sixty_degrees_l3897_389796


namespace NUMINAMATH_CALUDE_boxes_with_neither_l3897_389756

theorem boxes_with_neither (total : ℕ) (with_stickers : ℕ) (with_cards : ℕ) (with_both : ℕ) :
  total = 15 →
  with_stickers = 8 →
  with_cards = 5 →
  with_both = 3 →
  total - (with_stickers + with_cards - with_both) = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_boxes_with_neither_l3897_389756


namespace NUMINAMATH_CALUDE_x_value_equality_l3897_389737

theorem x_value_equality : (2023^2 - 2023 - 10000) / 2023 = (2022 * 2023 - 10000) / 2023 := by
  sorry

end NUMINAMATH_CALUDE_x_value_equality_l3897_389737


namespace NUMINAMATH_CALUDE_prob_white_balls_same_color_l3897_389739

/-- The number of white balls in the box -/
def white_balls : ℕ := 6

/-- The number of black balls in the box -/
def black_balls : ℕ := 5

/-- The number of balls drawn -/
def balls_drawn : ℕ := 3

/-- The probability that the drawn balls are white, given they are the same color -/
def prob_white_given_same_color : ℚ := 2/3

theorem prob_white_balls_same_color :
  let total_same_color := Nat.choose white_balls balls_drawn + Nat.choose black_balls balls_drawn
  let prob := (Nat.choose white_balls balls_drawn : ℚ) / total_same_color
  prob = prob_white_given_same_color := by sorry

end NUMINAMATH_CALUDE_prob_white_balls_same_color_l3897_389739


namespace NUMINAMATH_CALUDE_no_positive_integer_solutions_l3897_389735

theorem no_positive_integer_solutions : 
  ¬∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ (x^2 * y^2)^2 - 14 * (x^2 * y^2) + 49 = 0 :=
by sorry

end NUMINAMATH_CALUDE_no_positive_integer_solutions_l3897_389735


namespace NUMINAMATH_CALUDE_min_value_of_sum_l3897_389701

theorem min_value_of_sum (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 1/a + 4/b = 1) :
  ∃ (min : ℝ), min = 9 ∧ ∀ (x y : ℝ), x > 0 → y > 0 → 1/x + 4/y = 1 → x + y ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_sum_l3897_389701


namespace NUMINAMATH_CALUDE_max_value_of_f_l3897_389716

noncomputable def f (x : ℝ) : ℝ := Real.sin x - Real.sqrt 3 * Real.cos x

theorem max_value_of_f :
  ∃ (M : ℝ), (∀ (x : ℝ), f x ≤ M) ∧ (∃ (x : ℝ), f x = M) ∧ M = 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_l3897_389716


namespace NUMINAMATH_CALUDE_sum_of_fifth_powers_l3897_389741

theorem sum_of_fifth_powers (ζ₁ ζ₂ ζ₃ : ℂ) 
  (sum_1 : ζ₁ + ζ₂ + ζ₃ = 2)
  (sum_2 : ζ₁^2 + ζ₂^2 + ζ₃^2 = 6)
  (sum_3 : ζ₁^3 + ζ₂^3 + ζ₃^3 = 8) :
  ζ₁^5 + ζ₂^5 + ζ₃^5 = 20 := by
sorry

end NUMINAMATH_CALUDE_sum_of_fifth_powers_l3897_389741


namespace NUMINAMATH_CALUDE_laura_minimum_score_l3897_389788

def minimum_score (score1 score2 score3 : ℝ) (required_average : ℝ) : ℝ :=
  4 * required_average - (score1 + score2 + score3)

theorem laura_minimum_score :
  minimum_score 80 78 76 85 = 106 := by sorry

end NUMINAMATH_CALUDE_laura_minimum_score_l3897_389788


namespace NUMINAMATH_CALUDE_closest_to_fraction_l3897_389726

def options : List ℝ := [50, 500, 1500, 1600, 2000]

theorem closest_to_fraction (options : List ℝ) :
  let fraction : ℝ := 351 / 0.22
  let differences := options.map (λ x => |x - fraction|)
  let min_diff := differences.minimum?
  let closest := options.find? (λ x => |x - fraction| = min_diff.get!)
  closest = some 1600 := by
  sorry

end NUMINAMATH_CALUDE_closest_to_fraction_l3897_389726


namespace NUMINAMATH_CALUDE_multiply_three_point_six_by_half_l3897_389786

theorem multiply_three_point_six_by_half : 3.6 * 0.5 = 1.8 := by
  sorry

end NUMINAMATH_CALUDE_multiply_three_point_six_by_half_l3897_389786


namespace NUMINAMATH_CALUDE_derivative_x_squared_sin_x_l3897_389727

theorem derivative_x_squared_sin_x :
  ∀ x : ℝ, deriv (λ x => x^2 * Real.sin x) x = 2 * x * Real.sin x + x^2 * Real.cos x :=
by sorry

end NUMINAMATH_CALUDE_derivative_x_squared_sin_x_l3897_389727


namespace NUMINAMATH_CALUDE_conference_handshakes_l3897_389733

/-- Represents a group of people at a conference -/
structure ConferenceGroup where
  total : ℕ
  group_a : ℕ
  group_b : ℕ
  h_total : total = group_a + group_b

/-- Calculates the number of handshakes in a conference group -/
def count_handshakes (g : ConferenceGroup) : ℕ :=
  (g.group_b * (g.group_b - 1)) / 2

/-- Theorem stating the number of handshakes in the specific conference scenario -/
theorem conference_handshakes :
  ∀ g : ConferenceGroup,
    g.total = 30 →
    g.group_a = 25 →
    g.group_b = 5 →
    count_handshakes g = 10 := by
  sorry


end NUMINAMATH_CALUDE_conference_handshakes_l3897_389733


namespace NUMINAMATH_CALUDE_apple_difference_l3897_389753

theorem apple_difference (ben_apples phillip_apples tom_apples : ℕ) : 
  ben_apples > phillip_apples →
  tom_apples = (3 * ben_apples) / 8 →
  phillip_apples = 40 →
  tom_apples = 18 →
  ben_apples - phillip_apples = 8 := by
sorry

end NUMINAMATH_CALUDE_apple_difference_l3897_389753


namespace NUMINAMATH_CALUDE_shape_area_l3897_389791

-- Define the shape
structure Shape where
  sides_equal : Bool
  right_angles : Bool
  num_squares : Nat
  small_square_side : Real

-- Define the theorem
theorem shape_area (s : Shape) 
  (h1 : s.sides_equal = true) 
  (h2 : s.right_angles = true) 
  (h3 : s.num_squares = 8) 
  (h4 : s.small_square_side = 2) : 
  s.num_squares * (s.small_square_side * s.small_square_side) = 32 := by
  sorry

end NUMINAMATH_CALUDE_shape_area_l3897_389791


namespace NUMINAMATH_CALUDE_lcm_150_414_l3897_389769

theorem lcm_150_414 : Nat.lcm 150 414 = 10350 := by
  sorry

end NUMINAMATH_CALUDE_lcm_150_414_l3897_389769


namespace NUMINAMATH_CALUDE_quadrilateral_area_is_2007_l3897_389795

/-- The area of a quadrilateral with vertices at (1, 3), (1, 1), (3, 1), and (2007, 2008) -/
def quadrilateral_area : ℝ :=
  let A : ℝ × ℝ := (1, 3)
  let B : ℝ × ℝ := (1, 1)
  let C : ℝ × ℝ := (3, 1)
  let D : ℝ × ℝ := (2007, 2008)
  -- Area calculation goes here
  0  -- Placeholder, replace with actual calculation

/-- Theorem stating that the area of the quadrilateral is 2007 square units -/
theorem quadrilateral_area_is_2007 : quadrilateral_area = 2007 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_is_2007_l3897_389795


namespace NUMINAMATH_CALUDE_problem_statement_l3897_389792

theorem problem_statement (x y : ℕ) (hx : x = 4) (hy : y = 3) : 5 * x + 2 * y * 3 = 38 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3897_389792


namespace NUMINAMATH_CALUDE_triangle_rectangle_ratio_l3897_389729

theorem triangle_rectangle_ratio : 
  ∀ (triangle_leg : ℝ) (rect_short_side : ℝ),
  triangle_leg > 0 ∧ rect_short_side > 0 →
  2 * triangle_leg + Real.sqrt 2 * triangle_leg = 48 →
  2 * (rect_short_side + 2 * rect_short_side) = 48 →
  (Real.sqrt 2 * triangle_leg) / rect_short_side = 3 * (2 * Real.sqrt 2 - 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_rectangle_ratio_l3897_389729


namespace NUMINAMATH_CALUDE_strawberry_picking_problem_l3897_389745

/-- Calculates the number of pounds of strawberries picked given the problem conditions -/
def strawberries_picked (entrance_fee : ℚ) (price_per_pound : ℚ) (num_people : ℕ) (total_paid : ℚ) : ℚ :=
  (total_paid + num_people * entrance_fee) / price_per_pound

/-- Theorem stating that under the given conditions, 7 pounds of strawberries were picked -/
theorem strawberry_picking_problem :
  let entrance_fee : ℚ := 4
  let price_per_pound : ℚ := 20
  let num_people : ℕ := 3
  let total_paid : ℚ := 128
  strawberries_picked entrance_fee price_per_pound num_people total_paid = 7 := by
  sorry


end NUMINAMATH_CALUDE_strawberry_picking_problem_l3897_389745


namespace NUMINAMATH_CALUDE_ox_and_sheep_cost_l3897_389768

theorem ox_and_sheep_cost (ox sheep : ℚ) 
  (h1 : 5 * ox + 2 * sheep = 10) 
  (h2 : 2 * ox + 8 * sheep = 8) : 
  ox = 16/9 ∧ sheep = 5/9 := by
  sorry

end NUMINAMATH_CALUDE_ox_and_sheep_cost_l3897_389768


namespace NUMINAMATH_CALUDE_abcdef_hex_to_binary_bits_l3897_389711

theorem abcdef_hex_to_binary_bits : ∃ (n : ℕ), n = 24 ∧ 
  (2^(n-1) : ℕ) ≤ (0xABCDEF : ℕ) ∧ (0xABCDEF : ℕ) < 2^n :=
by sorry

end NUMINAMATH_CALUDE_abcdef_hex_to_binary_bits_l3897_389711


namespace NUMINAMATH_CALUDE_original_rectangle_area_l3897_389755

/-- Given a rectangle whose dimensions are doubled to form a new rectangle with an area of 32 square meters, 
    prove that the area of the original rectangle is 8 square meters. -/
theorem original_rectangle_area (original_width original_height : ℝ) : 
  original_width > 0 → 
  original_height > 0 → 
  (2 * original_width) * (2 * original_height) = 32 → 
  original_width * original_height = 8 := by
sorry

end NUMINAMATH_CALUDE_original_rectangle_area_l3897_389755


namespace NUMINAMATH_CALUDE_book_sales_properties_l3897_389715

/-- Represents the daily sales and profit functions for a book selling business -/
structure BookSales where
  cost : ℝ              -- Cost price per book
  min_price : ℝ         -- Minimum selling price
  max_profit_rate : ℝ   -- Maximum profit rate
  base_sales : ℝ        -- Base sales at minimum price
  sales_decrease : ℝ    -- Sales decrease per unit price increase

variable (bs : BookSales)

/-- Daily sales as a function of price -/
def daily_sales (x : ℝ) : ℝ := bs.base_sales - bs.sales_decrease * (x - bs.min_price)

/-- Daily profit as a function of price -/
def daily_profit (x : ℝ) : ℝ := (x - bs.cost) * (daily_sales bs x)

/-- Theorem stating the properties of the book selling business -/
theorem book_sales_properties (bs : BookSales) 
  (h_cost : bs.cost = 40)
  (h_min_price : bs.min_price = 45)
  (h_max_profit_rate : bs.max_profit_rate = 0.5)
  (h_base_sales : bs.base_sales = 310)
  (h_sales_decrease : bs.sales_decrease = 10) :
  -- 1. Daily sales function
  (∀ x, daily_sales bs x = -10 * x + 760) ∧
  -- 2. Selling price range
  (∀ x, bs.min_price ≤ x ∧ x ≤ bs.cost * (1 + bs.max_profit_rate)) ∧
  -- 3. Profit-maximizing price
  (∃ x_max, ∀ x, daily_profit bs x ≤ daily_profit bs x_max ∧ x_max = 58) ∧
  -- 4. Maximum daily profit
  (∃ max_profit, max_profit = daily_profit bs 58 ∧ max_profit = 3240) ∧
  -- 5. Price for $2600 profit
  (∃ x_2600, daily_profit bs x_2600 = 2600 ∧ x_2600 = 50) := by
  sorry

end NUMINAMATH_CALUDE_book_sales_properties_l3897_389715


namespace NUMINAMATH_CALUDE_age_calculation_l3897_389782

/-- Given Luke's current age and Mr. Bernard's future age relative to Luke's current age, 
    calculate 10 years less than their average age. -/
theorem age_calculation (luke_age : ℕ) (bernard_future_age_multiplier : ℕ) : 
  luke_age = 20 →
  bernard_future_age_multiplier = 3 →
  10 + (luke_age + (bernard_future_age_multiplier * luke_age - 8)) / 2 - 10 = 26 := by
sorry

end NUMINAMATH_CALUDE_age_calculation_l3897_389782


namespace NUMINAMATH_CALUDE_min_colors_for_pyramid_game_l3897_389720

/-- Represents a pyramid with a regular polygon base -/
structure Pyramid :=
  (base_vertices : ℕ)

/-- The total number of edges in a pyramid -/
def total_edges (p : Pyramid) : ℕ := 2 * p.base_vertices

/-- The maximum degree of any vertex in the pyramid -/
def max_vertex_degree (p : Pyramid) : ℕ := p.base_vertices

/-- The minimal number of colors needed for the coloring game on a pyramid -/
def min_colors_needed (p : Pyramid) : ℕ := p.base_vertices

theorem min_colors_for_pyramid_game (p : Pyramid) (h : p.base_vertices = 2016) :
  min_colors_needed p = 2016 :=
sorry

end NUMINAMATH_CALUDE_min_colors_for_pyramid_game_l3897_389720


namespace NUMINAMATH_CALUDE_cardboard_box_square_cutout_l3897_389717

theorem cardboard_box_square_cutout (length width area : ℝ) 
  (h1 : length = 80)
  (h2 : width = 60)
  (h3 : area = 1500) :
  ∃ (x : ℝ), x > 0 ∧ x < 30 ∧ (length - 2*x) * (width - 2*x) = area ∧ x = 15 :=
sorry

end NUMINAMATH_CALUDE_cardboard_box_square_cutout_l3897_389717


namespace NUMINAMATH_CALUDE_sum_a_d_equals_ten_l3897_389759

theorem sum_a_d_equals_ten (a b c d : ℝ) 
  (h1 : a + b = 16) 
  (h2 : b + c = 9) 
  (h3 : c + d = 3) : 
  a + d = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_a_d_equals_ten_l3897_389759


namespace NUMINAMATH_CALUDE_little_john_money_distribution_l3897_389721

theorem little_john_money_distribution (initial_amount spent_on_sweets final_amount : ℚ) 
  (num_friends : ℕ) (h1 : initial_amount = 20.10) (h2 : spent_on_sweets = 1.05) 
  (h3 : final_amount = 17.05) (h4 : num_friends = 2) : 
  (initial_amount - final_amount - spent_on_sweets) / num_friends = 1 := by
  sorry

end NUMINAMATH_CALUDE_little_john_money_distribution_l3897_389721


namespace NUMINAMATH_CALUDE_equal_charge_at_20_minutes_l3897_389706

/-- United Telephone's base rate -/
def united_base : ℝ := 11

/-- United Telephone's per-minute rate -/
def united_per_minute : ℝ := 0.25

/-- Atlantic Call's base rate -/
def atlantic_base : ℝ := 12

/-- Atlantic Call's per-minute rate -/
def atlantic_per_minute : ℝ := 0.20

/-- The number of minutes at which both companies charge the same amount -/
def equal_charge_minutes : ℝ := 20

theorem equal_charge_at_20_minutes :
  united_base + united_per_minute * equal_charge_minutes =
  atlantic_base + atlantic_per_minute * equal_charge_minutes :=
sorry

end NUMINAMATH_CALUDE_equal_charge_at_20_minutes_l3897_389706


namespace NUMINAMATH_CALUDE_max_value_abcd_l3897_389758

theorem max_value_abcd (a b c d : ℝ) 
  (nonneg_a : 0 ≤ a) (nonneg_b : 0 ≤ b) (nonneg_c : 0 ≤ c) (nonneg_d : 0 ≤ d)
  (sum_squares : a^2 + b^2 + c^2 + d^2 = 1) :
  2 * a * b * Real.sqrt 2 + 2 * b * c + 2 * c * d ≤ 1 ∧ 
  ∃ a' b' c' d', 0 ≤ a' ∧ 0 ≤ b' ∧ 0 ≤ c' ∧ 0 ≤ d' ∧ 
    a'^2 + b'^2 + c'^2 + d'^2 = 1 ∧
    2 * a' * b' * Real.sqrt 2 + 2 * b' * c' + 2 * c' * d' = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_abcd_l3897_389758


namespace NUMINAMATH_CALUDE_system_solution_l3897_389794

theorem system_solution (x y : ℝ) 
  (h1 : Real.log (x + y) - Real.log 5 = Real.log x + Real.log y - Real.log 6)
  (h2 : Real.log x / (Real.log (y + 6) - (Real.log y + Real.log 6)) = -1)
  (hx : x > 0)
  (hy : y > 0)
  (hny : y ≠ 6/5)
  (hyb : y > -6) :
  x = 2 ∧ y = 3 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l3897_389794


namespace NUMINAMATH_CALUDE_mean_of_readings_l3897_389708

def readings : List ℝ := [2, 2.1, 2, 2.2]

theorem mean_of_readings (x : ℝ) (mean : ℝ) : 
  readings.length = 4 →
  mean = (readings.sum + x) / 5 := by
  sorry

end NUMINAMATH_CALUDE_mean_of_readings_l3897_389708


namespace NUMINAMATH_CALUDE_counterexample_necessity_l3897_389774

-- Define the concept of a mathematical statement
def MathStatement : Type := String

-- Define the concept of a proof method
inductive ProofMethod
| Direct : ProofMethod
| Counterexample : ProofMethod
| Other : ProofMethod

-- Define a property of mathematical statements
def CanBeProvedDirectly (s : MathStatement) : Prop := sorry

-- Define the theorem to be proved
theorem counterexample_necessity (s : MathStatement) :
  ¬(∀ s, ¬(CanBeProvedDirectly s) → (∀ m : ProofMethod, m = ProofMethod.Counterexample)) :=
sorry

end NUMINAMATH_CALUDE_counterexample_necessity_l3897_389774


namespace NUMINAMATH_CALUDE_equation_solutions_l3897_389747

theorem equation_solutions :
  (∀ x, x^2 - 8*x - 1 = 0 ↔ x = 4 + Real.sqrt 17 ∨ x = 4 - Real.sqrt 17) ∧
  (∀ x, x*(2*x - 5) = 4*x - 10 ↔ x = 5/2 ∨ x = 2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3897_389747


namespace NUMINAMATH_CALUDE_expression_simplification_l3897_389744

theorem expression_simplification (a b : ℝ) (h1 : a = 1) (h2 : b = -2) :
  ((a - 2*b)^2 - (a - 2*b)*(a + 2*b) - 4*b) / (-2*b) = 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3897_389744


namespace NUMINAMATH_CALUDE_divisibility_pairs_l3897_389770

theorem divisibility_pairs : 
  {p : ℕ × ℕ | (p.1 + 1) % p.2 = 0 ∧ (p.2^2 - p.2 + 1) % p.1 = 0} = 
  {(1, 1), (1, 2), (3, 2)} := by
sorry

end NUMINAMATH_CALUDE_divisibility_pairs_l3897_389770


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_ratio_l3897_389771

/-- Given an arithmetic sequence {a_n} where a_n ≠ 0 for all n,
    if a_1, a_3, and a_4 form a geometric sequence,
    then the common ratio of this geometric sequence is either 1 or 1/2. -/
theorem arithmetic_geometric_sequence_ratio
  (a : ℕ → ℝ)  -- The arithmetic sequence
  (h_arith : ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m)  -- Arithmetic sequence condition
  (h_nonzero : ∀ n : ℕ, a n ≠ 0)  -- Non-zero condition
  (h_geom : ∃ q : ℝ, a 3 = a 1 * q ∧ a 4 = a 3 * q)  -- Geometric sequence condition
  : ∃ q : ℝ, (q = 1 ∨ q = 1/2) ∧ a 3 = a 1 * q ∧ a 4 = a 3 * q :=
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_ratio_l3897_389771


namespace NUMINAMATH_CALUDE_restaurant_group_size_l3897_389773

theorem restaurant_group_size :
  ∀ (adult_meal_cost : ℕ) (kids_in_group : ℕ) (total_cost : ℕ),
    adult_meal_cost = 8 →
    kids_in_group = 2 →
    total_cost = 72 →
    ∃ (adults_in_group : ℕ),
      adults_in_group * adult_meal_cost = total_cost ∧
      adults_in_group + kids_in_group = 11 :=
by sorry

end NUMINAMATH_CALUDE_restaurant_group_size_l3897_389773


namespace NUMINAMATH_CALUDE_max_sum_digits_divisible_by_13_l3897_389743

theorem max_sum_digits_divisible_by_13 :
  ∀ A B C : ℕ,
  A < 10 → B < 10 → C < 10 →
  (2000 + 100 * A + 10 * B + C) % 13 = 0 →
  A + B + C ≤ 26 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_digits_divisible_by_13_l3897_389743


namespace NUMINAMATH_CALUDE_intersection_equal_A_l3897_389703

-- Define the universe
def U : Set Char := {'a', 'b', 'c', 'd'}

-- Define sets A and B
def A : Set Char := {'a', 'c'}
def B : Set Char := {'b'}

-- Define set C as the complement of A ∪ B in U
def C : Set Char := U \ (A ∪ B)

-- Theorem statement
theorem intersection_equal_A : A ∩ (C ∪ B) = A := by sorry

end NUMINAMATH_CALUDE_intersection_equal_A_l3897_389703


namespace NUMINAMATH_CALUDE_sue_nuts_count_l3897_389728

theorem sue_nuts_count (bill_nuts harry_nuts sue_nuts : ℕ) : 
  bill_nuts = 6 * harry_nuts →
  harry_nuts = 2 * sue_nuts →
  bill_nuts + harry_nuts = 672 →
  sue_nuts = 48 := by
sorry

end NUMINAMATH_CALUDE_sue_nuts_count_l3897_389728


namespace NUMINAMATH_CALUDE_stock_price_after_two_years_l3897_389730

/-- The final stock price after a 150% increase followed by a 30% decrease, given an initial price of $120 -/
theorem stock_price_after_two_years (initial_price : ℝ) (first_year_increase : ℝ) (second_year_decrease : ℝ) :
  initial_price = 120 →
  first_year_increase = 150 / 100 →
  second_year_decrease = 30 / 100 →
  initial_price * (1 + first_year_increase) * (1 - second_year_decrease) = 210 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_after_two_years_l3897_389730


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l3897_389779

/-- The equation of a line perpendicular to x-2y=3 and passing through (1,2) is y=-2x+4 -/
theorem perpendicular_line_equation :
  ∀ (x y : ℝ),
  (∃ (m b : ℝ), y = m*x + b ∧ 
                 (1, 2) ∈ {(x, y) | y = m*x + b} ∧
                 m * (1/2) = -1) →
  y = -2*x + 4 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l3897_389779


namespace NUMINAMATH_CALUDE_lily_family_vacation_suitcases_l3897_389766

/-- The number of suitcases Lily's family brings on vacation -/
def family_suitcases (num_siblings : ℕ) (suitcases_per_sibling : ℕ) (parent_suitcases : ℕ) : ℕ :=
  num_siblings * suitcases_per_sibling + parent_suitcases

/-- Theorem stating the total number of suitcases Lily's family brings on vacation -/
theorem lily_family_vacation_suitcases :
  family_suitcases 4 2 6 = 14 := by
  sorry

#eval family_suitcases 4 2 6

end NUMINAMATH_CALUDE_lily_family_vacation_suitcases_l3897_389766


namespace NUMINAMATH_CALUDE_fraction_addition_l3897_389707

theorem fraction_addition (x P Q : ℚ) : 
  (8 * x^2 - 9 * x + 20) / (4 * x^3 - 5 * x^2 - 26 * x + 24) = 
  P / (2 * x^2 - 5 * x + 3) + Q / (2 * x - 3) →
  P = 4/9 ∧ Q = 68/9 := by
sorry

end NUMINAMATH_CALUDE_fraction_addition_l3897_389707


namespace NUMINAMATH_CALUDE_simple_interest_time_period_l3897_389713

theorem simple_interest_time_period 
  (P : ℝ) -- Principal sum
  (R : ℝ) -- Rate of interest per annum
  (T : ℝ) -- Time period in years
  (h1 : R = 4) -- Given rate of interest is 4%
  (h2 : P / 5 = (P * R * T) / 100) -- Simple interest is one-fifth of principal and follows the formula
  : T = 5 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_time_period_l3897_389713


namespace NUMINAMATH_CALUDE_zilla_savings_proof_l3897_389709

def monthly_savings (total_earnings rent other_expenses : ℝ) : ℝ :=
  total_earnings - rent - other_expenses

theorem zilla_savings_proof 
  (total_earnings : ℝ)
  (rent_percentage : ℝ)
  (rent : ℝ)
  (h1 : rent_percentage = 0.07)
  (h2 : rent = 133)
  (h3 : rent = total_earnings * rent_percentage)
  (h4 : let other_expenses := total_earnings / 2;
        monthly_savings total_earnings rent other_expenses = 817) : 
  ∃ (savings : ℝ), savings = 817 ∧ savings = monthly_savings total_earnings rent (total_earnings / 2) :=
sorry

end NUMINAMATH_CALUDE_zilla_savings_proof_l3897_389709


namespace NUMINAMATH_CALUDE_cookie_radius_l3897_389780

theorem cookie_radius (x y : ℝ) :
  (x^2 + y^2 - 8 = 2*x + 4*y) →
  ∃ (h k r : ℝ), (x - h)^2 + (y - k)^2 = r^2 ∧ r = Real.sqrt 13 :=
by sorry

end NUMINAMATH_CALUDE_cookie_radius_l3897_389780


namespace NUMINAMATH_CALUDE_problem_2022_l3897_389772

theorem problem_2022 (a : ℝ) :
  (|2022 - a| + Real.sqrt (a - 2023) = a) → (a - 2022^2 = 2023) := by
  sorry

end NUMINAMATH_CALUDE_problem_2022_l3897_389772


namespace NUMINAMATH_CALUDE_not_necessarily_right_triangle_l3897_389793

/-- A triangle with angles A, B, and C. -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  angle_sum : A + B + C = 180
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C

/-- A right triangle is a triangle with one 90-degree angle. -/
def RightTriangle (t : Triangle) : Prop :=
  t.A = 90 ∨ t.B = 90 ∨ t.C = 90

/-- The condition that angles A and B are equal and twice angle C. -/
def AngleCondition (t : Triangle) : Prop :=
  t.A = t.B ∧ t.A = 2 * t.C

theorem not_necessarily_right_triangle :
  ∃ t : Triangle, AngleCondition t ∧ ¬RightTriangle t := by
  sorry

end NUMINAMATH_CALUDE_not_necessarily_right_triangle_l3897_389793


namespace NUMINAMATH_CALUDE_probability_same_color_value_l3897_389700

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : Nat)
  (black_cards : Nat)
  (red_cards : Nat)
  (h_total : total_cards = 52)
  (h_black : black_cards = 26)
  (h_red : red_cards = 26)
  (h_sum : black_cards + red_cards = total_cards)

/-- The probability of drawing four cards of the same color from a standard deck -/
def probability_same_color (d : Deck) : Rat :=
  2 * (d.black_cards.choose 4) / d.total_cards.choose 4

/-- Theorem stating the probability of drawing four cards of the same color -/
theorem probability_same_color_value (d : Deck) :
  probability_same_color d = 276 / 2499 := by
  sorry

end NUMINAMATH_CALUDE_probability_same_color_value_l3897_389700


namespace NUMINAMATH_CALUDE_pencils_given_equals_nine_l3897_389722

/-- The number of pencils in one stroke -/
def pencils_per_stroke : ℕ := 12

/-- The number of strokes Namjoon had -/
def namjoon_strokes : ℕ := 2

/-- The number of pencils Namjoon had left after giving some to Yoongi -/
def pencils_left : ℕ := 15

/-- The number of pencils Namjoon gave to Yoongi -/
def pencils_given_to_yoongi : ℕ := namjoon_strokes * pencils_per_stroke - pencils_left

theorem pencils_given_equals_nine : pencils_given_to_yoongi = 9 := by
  sorry

end NUMINAMATH_CALUDE_pencils_given_equals_nine_l3897_389722


namespace NUMINAMATH_CALUDE_absolute_difference_of_integers_l3897_389798

theorem absolute_difference_of_integers (x y : ℤ) 
  (h1 : x ≠ y)
  (h2 : (x + y) / 2 = 15)
  (h3 : Real.sqrt (x * y) + 6 = 15) : 
  |x - y| = 24 := by
  sorry

end NUMINAMATH_CALUDE_absolute_difference_of_integers_l3897_389798


namespace NUMINAMATH_CALUDE_digging_time_for_second_hole_l3897_389767

/-- Proves that given the conditions of the digging problem, the time required to dig the second hole is 6 hours -/
theorem digging_time_for_second_hole 
  (workers_first : ℕ) 
  (hours_first : ℕ) 
  (depth_first : ℕ) 
  (extra_workers : ℕ) 
  (depth_second : ℕ) 
  (h : workers_first = 45)
  (i : hours_first = 8)
  (j : depth_first = 30)
  (k : extra_workers = 65)
  (l : depth_second = 55) :
  (workers_first + extra_workers) * (660 / (workers_first + extra_workers) : ℚ) * depth_second = 
  workers_first * hours_first * depth_second := by
sorry

#eval (45 + 65) * (660 / (45 + 65) : ℚ)

end NUMINAMATH_CALUDE_digging_time_for_second_hole_l3897_389767


namespace NUMINAMATH_CALUDE_root_condition_for_k_l3897_389763

/-- The function f(x) = kx - 3 -/
def f (k : ℝ) (x : ℝ) : ℝ := k * x - 3

/-- A function has a root in an interval if its values at the endpoints have different signs -/
def has_root_in_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  f a * f b ≤ 0

theorem root_condition_for_k (k : ℝ) :
  (k ≥ 3 → has_root_in_interval (f k) (-1) 1) ∧
  (∃ k', k' < 3 ∧ has_root_in_interval (f k') (-1) 1) :=
sorry

end NUMINAMATH_CALUDE_root_condition_for_k_l3897_389763


namespace NUMINAMATH_CALUDE_min_value_theorem_l3897_389765

theorem min_value_theorem (a b c d : ℝ) 
  (hb : b ≠ 0) 
  (hd : d ≠ -1) 
  (h1 : (a^2 - Real.log a) / b = (c - 1) / (d + 1))
  (h2 : (a^2 - Real.log a) / b = 1) :
  ∃ (min : ℝ), min = 2 ∧ ∀ (x y : ℝ), (x^2 - Real.log x) / y = (c - 1) / (d + 1) → 
    (x - c)^2 + (y - d)^2 ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3897_389765


namespace NUMINAMATH_CALUDE_line_segment_both_symmetric_l3897_389789

-- Define the shapes
inductive Shape
  | EquilateralTriangle
  | IsoscelesTriangle
  | Parallelogram
  | LineSegment

-- Define symmetry properties
def isCentrallySymmetric (s : Shape) : Prop :=
  match s with
  | Shape.EquilateralTriangle => False
  | Shape.IsoscelesTriangle => False
  | Shape.Parallelogram => True
  | Shape.LineSegment => True

def isAxiallySymmetric (s : Shape) : Prop :=
  match s with
  | Shape.EquilateralTriangle => True
  | Shape.IsoscelesTriangle => True
  | Shape.Parallelogram => False
  | Shape.LineSegment => True

-- Theorem statement
theorem line_segment_both_symmetric :
  ∀ s : Shape, (isCentrallySymmetric s ∧ isAxiallySymmetric s) ↔ s = Shape.LineSegment :=
by sorry

end NUMINAMATH_CALUDE_line_segment_both_symmetric_l3897_389789


namespace NUMINAMATH_CALUDE_trigonometric_inequality_l3897_389750

theorem trigonometric_inequality (x : ℝ) (h : x ∈ Set.Ioo 0 (3 * π / 8)) :
  (1 / Real.sin (x / 3)) + (1 / Real.sin (8 * x / 3)) > 
  Real.sin (3 * x / 2) / (Real.sin (x / 2) * Real.sin (2 * x)) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_inequality_l3897_389750


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l3897_389731

theorem complex_fraction_equality : Complex.I * 5 / (1 - Complex.I) = -5/2 + Complex.I * (5/2) := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l3897_389731


namespace NUMINAMATH_CALUDE_rain_both_days_l3897_389749

-- Define the probabilities
def prob_rain_monday : ℝ := 0.62
def prob_rain_tuesday : ℝ := 0.54
def prob_no_rain : ℝ := 0.28

-- Theorem statement
theorem rain_both_days :
  let prob_rain_both := prob_rain_monday + prob_rain_tuesday - (1 - prob_no_rain)
  prob_rain_both = 0.44 := by sorry

end NUMINAMATH_CALUDE_rain_both_days_l3897_389749


namespace NUMINAMATH_CALUDE_spinner_points_north_l3897_389736

/-- Represents the four cardinal directions -/
inductive Direction
  | North
  | East
  | South
  | West

/-- Represents a rotation of the spinner -/
def rotate (initial : Direction) (revolutions : ℚ) : Direction :=
  sorry

/-- Theorem stating that after the described rotations, the spinner points north -/
theorem spinner_points_north :
  let initial_direction := Direction.North
  let clockwise_rotation := 7/2
  let counterclockwise_rotation := 5/2
  rotate (rotate initial_direction clockwise_rotation) (-counterclockwise_rotation) = Direction.North :=
by sorry

end NUMINAMATH_CALUDE_spinner_points_north_l3897_389736


namespace NUMINAMATH_CALUDE_altitude_intersection_property_l3897_389705

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a triangle -/
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

/-- Checks if a triangle is acute -/
def isAcute (t : Triangle) : Prop := sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Checks if a line is perpendicular to another line -/
def isPerpendicular (p1 p2 p3 p4 : Point) : Prop := sorry

/-- Finds the intersection point of two lines -/
def lineIntersection (p1 p2 p3 p4 : Point) : Point := sorry

/-- Theorem: In an acute triangle ABC with altitudes AP and BQ intersecting at H,
    if HP = 7 and HQ = 3, then (BP)(PC) - (AQ)(QC) = 40 -/
theorem altitude_intersection_property (t : Triangle) (P Q H : Point) :
  isAcute t →
  isPerpendicular t.A P t.B t.C →
  isPerpendicular t.B Q t.A t.C →
  H = lineIntersection t.A P t.B Q →
  distance H P = 7 →
  distance H Q = 3 →
  distance t.B P * distance P t.C - distance t.A Q * distance Q t.C = 40 := by
  sorry

end NUMINAMATH_CALUDE_altitude_intersection_property_l3897_389705


namespace NUMINAMATH_CALUDE_project_solution_l3897_389754

/-- Represents the time (in days) required for a person to complete the project alone. -/
structure ProjectTime where
  personA : ℝ
  personB : ℝ
  personC : ℝ

/-- Defines the conditions of the engineering project. -/
def ProjectConditions (t : ProjectTime) : Prop :=
  -- Person B works alone for 4 days
  4 / t.personB +
  -- Persons A and C work together for 6 days
  6 * (1 / t.personA + 1 / t.personC) +
  -- Person A completes the remaining work in 9 days
  9 / t.personA = 1 ∧
  -- Work completed by Person B is 1/3 of the work completed by Person A
  t.personB = 3 * t.personA ∧
  -- Work completed by Person C is 2 times the work completed by Person B
  t.personC = t.personB / 2

/-- Theorem stating the solution to the engineering project problem. -/
theorem project_solution :
  ∃ t : ProjectTime, ProjectConditions t ∧ t.personA = 30 ∧ t.personB = 24 ∧ t.personC = 18 :=
by sorry

end NUMINAMATH_CALUDE_project_solution_l3897_389754


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3897_389777

def A : Set ℝ := {x | x < 2}
def B : Set ℝ := {x | 1 < x ∧ x < 7}

theorem union_of_A_and_B : A ∪ B = {x | x < 7} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3897_389777


namespace NUMINAMATH_CALUDE_star_equation_solution_l3897_389785

def star (a b : ℕ) : ℕ := a^b + a*b

theorem star_equation_solution :
  ∀ a b : ℕ, 
  a ≥ 2 → b ≥ 2 → 
  star a b = 24 → 
  a + b = 6 := by sorry

end NUMINAMATH_CALUDE_star_equation_solution_l3897_389785


namespace NUMINAMATH_CALUDE_max_reflections_is_largest_l3897_389799

/-- Represents the angle between lines AD and CD in degrees -/
def angle_CDA : ℝ := 5

/-- Represents the maximum allowed path length -/
def max_path_length : ℝ := 100

/-- Calculates the total angle after n reflections -/
def total_angle (n : ℕ) : ℝ := n * angle_CDA

/-- Represents the condition that the total angle must not exceed 90 degrees -/
def angle_condition (n : ℕ) : Prop := total_angle n ≤ 90

/-- Represents an approximation of the path length after n reflections -/
def approx_path_length (n : ℕ) : ℝ := 2 * n * 5

/-- Represents the condition that the path length must not exceed the maximum allowed length -/
def path_length_condition (n : ℕ) : Prop := approx_path_length n ≤ max_path_length

/-- Represents the maximum number of reflections that satisfies all conditions -/
def max_reflections : ℕ := 10

/-- Theorem stating that max_reflections is the largest value that satisfies all conditions -/
theorem max_reflections_is_largest :
  (angle_condition max_reflections) ∧
  (path_length_condition max_reflections) ∧
  (∀ m : ℕ, m > max_reflections → ¬(angle_condition m ∧ path_length_condition m)) :=
sorry

end NUMINAMATH_CALUDE_max_reflections_is_largest_l3897_389799


namespace NUMINAMATH_CALUDE_equation_solutions_l3897_389724

theorem equation_solutions :
  (∃ (s1 s2 : Set ℝ),
    (s1 = {x : ℝ | (x - 1)^2 - 25 = 0} ∧ s1 = {6, -4}) ∧
    (s2 = {x : ℝ | 3*x*(x - 2) = x - 2} ∧ s2 = {2, 1/3})) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3897_389724


namespace NUMINAMATH_CALUDE_hamburger_count_l3897_389757

theorem hamburger_count (served left_over : ℕ) (h1 : served = 3) (h2 : left_over = 6) :
  served + left_over = 9 := by
  sorry

end NUMINAMATH_CALUDE_hamburger_count_l3897_389757


namespace NUMINAMATH_CALUDE_gcd_12347_30841_l3897_389710

theorem gcd_12347_30841 : Nat.gcd 12347 30841 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_12347_30841_l3897_389710


namespace NUMINAMATH_CALUDE_central_cell_value_l3897_389723

def table_sum (a : ℝ) : ℝ :=
  a + 4*a + 16*a + 3*a + 12*a + 48*a + 9*a + 36*a + 144*a

theorem central_cell_value (a : ℝ) (h : table_sum a = 546) : 12 * a = 24 := by
  sorry

end NUMINAMATH_CALUDE_central_cell_value_l3897_389723


namespace NUMINAMATH_CALUDE_perimeter_of_square_d_l3897_389746

/-- Given a square C with perimeter 32 cm and a square D with area equal to one-third the area of square C, 
    the perimeter of square D is (32√3)/3 cm. -/
theorem perimeter_of_square_d (C D : Real) : 
  (C = 32) →  -- perimeter of square C is 32 cm
  (D^2 = (C/4)^2 / 3) →  -- area of square D is one-third the area of square C
  (4 * D = 32 * Real.sqrt 3 / 3) := by  -- perimeter of square D is (32√3)/3 cm
sorry

end NUMINAMATH_CALUDE_perimeter_of_square_d_l3897_389746


namespace NUMINAMATH_CALUDE_min_value_theorem_l3897_389764

/-- Given positive real numbers a and b, and a function f with minimum value 4 -/
def f (x a b : ℝ) : ℝ := |x + a| + |x - b|

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (hf : ∀ x, f x a b ≥ 4) (hf_min : ∃ x, f x a b = 4) :
  (a + b = 4) ∧ (∀ a b, a > 0 → b > 0 → a + b = 4 → (1/4) * a^2 + (1/4) * b^2 ≥ 3/16) ∧
  (∃ a b, a > 0 ∧ b > 0 ∧ a + b = 4 ∧ (1/4) * a^2 + (1/4) * b^2 = 3/16) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3897_389764


namespace NUMINAMATH_CALUDE_equation_implies_a_equals_four_l3897_389751

theorem equation_implies_a_equals_four (a : ℝ) :
  (∀ x : ℝ, x^2 + a*x + 4 = (x + 2)^2) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_implies_a_equals_four_l3897_389751


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3897_389718

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem statement -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 3)^2 - 3*(a 3) + 1 = 0 →
  (a 7)^2 - 3*(a 7) + 1 = 0 →
  a 4 + a 6 = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3897_389718


namespace NUMINAMATH_CALUDE_cubic_root_sum_l3897_389702

theorem cubic_root_sum (a b c : ℝ) : 
  (40 * a^3 - 60 * a^2 + 26 * a - 1 = 0) →
  (40 * b^3 - 60 * b^2 + 26 * b - 1 = 0) →
  (40 * c^3 - 60 * c^2 + 26 * c - 1 = 0) →
  (0 < a) ∧ (a < 1) →
  (0 < b) ∧ (b < 1) →
  (0 < c) ∧ (c < 1) →
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c) = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l3897_389702


namespace NUMINAMATH_CALUDE_percentage_kindergarten_combined_l3897_389734

/-- Percentage of Kindergarten students in combined schools -/
theorem percentage_kindergarten_combined (pinegrove_total : ℕ) (maplewood_total : ℕ)
  (pinegrove_k_percent : ℚ) (maplewood_k_percent : ℚ)
  (h1 : pinegrove_total = 150)
  (h2 : maplewood_total = 250)
  (h3 : pinegrove_k_percent = 18/100)
  (h4 : maplewood_k_percent = 14/100) :
  (pinegrove_k_percent * pinegrove_total + maplewood_k_percent * maplewood_total) /
  (pinegrove_total + maplewood_total) = 155/1000 := by
  sorry

#check percentage_kindergarten_combined

end NUMINAMATH_CALUDE_percentage_kindergarten_combined_l3897_389734


namespace NUMINAMATH_CALUDE_student_comprehensive_score_l3897_389738

/-- Calculates the comprehensive score of a student in a competition --/
def comprehensiveScore (theoreticalWeight : ℝ) (innovativeWeight : ℝ) (presentationWeight : ℝ)
                       (theoreticalScore : ℝ) (innovativeScore : ℝ) (presentationScore : ℝ) : ℝ :=
  theoreticalWeight * theoreticalScore + innovativeWeight * innovativeScore + presentationWeight * presentationScore

/-- Theorem stating that the student's comprehensive score is 89.5 --/
theorem student_comprehensive_score :
  let theoreticalWeight : ℝ := 0.20
  let innovativeWeight : ℝ := 0.50
  let presentationWeight : ℝ := 0.30
  let theoreticalScore : ℝ := 80
  let innovativeScore : ℝ := 90
  let presentationScore : ℝ := 95
  comprehensiveScore theoreticalWeight innovativeWeight presentationWeight
                     theoreticalScore innovativeScore presentationScore = 89.5 := by
  sorry

end NUMINAMATH_CALUDE_student_comprehensive_score_l3897_389738


namespace NUMINAMATH_CALUDE_greatest_n_for_perfect_square_T_l3897_389752

/-- The greatest power of 4 that divides an even positive integer -/
def h (x : ℕ+) : ℕ :=
  sorry

/-- Sum of h(4k) from k = 1 to 2^(n-1) -/
def T (n : ℕ+) : ℕ :=
  sorry

/-- Predicate for perfect squares -/
def is_perfect_square (m : ℕ) : Prop :=
  ∃ k : ℕ, m = k^2

theorem greatest_n_for_perfect_square_T :
  ∀ n : ℕ+, n < 500 → is_perfect_square (T n) → n ≤ 143 ∧
  is_perfect_square (T 143) :=
sorry

end NUMINAMATH_CALUDE_greatest_n_for_perfect_square_T_l3897_389752


namespace NUMINAMATH_CALUDE_mark_parking_tickets_l3897_389775

theorem mark_parking_tickets :
  ∀ (mark_speeding mark_parking sarah_speeding sarah_parking : ℕ),
  mark_speeding + mark_parking + sarah_speeding + sarah_parking = 24 →
  mark_parking = 2 * sarah_parking →
  mark_speeding = sarah_speeding →
  sarah_speeding = 6 →
  mark_parking = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_mark_parking_tickets_l3897_389775


namespace NUMINAMATH_CALUDE_sum_is_composite_l3897_389776

theorem sum_is_composite (a b : ℕ) (h : 34 * a = 43 * b) : 
  ∃ (x y : ℕ), x > 1 ∧ y > 1 ∧ a + b = x * y :=
by sorry

end NUMINAMATH_CALUDE_sum_is_composite_l3897_389776


namespace NUMINAMATH_CALUDE_quadratic_root_property_l3897_389714

theorem quadratic_root_property : ∀ a b : ℝ, 
  (a^2 - 3*a + 1 = 0) → (b^2 - 3*b + 1 = 0) → (a + b - a*b = 2) := by sorry

end NUMINAMATH_CALUDE_quadratic_root_property_l3897_389714


namespace NUMINAMATH_CALUDE_point_movement_theorem_l3897_389790

/-- Represents the final position of a point on a number line after a series of movements -/
def final_position (initial : Int) (right_move : Int) (left_move : Int) : Int :=
  initial + right_move - left_move

/-- Theorem stating that given the specific movements in the problem, 
    the final position is -5 -/
theorem point_movement_theorem :
  final_position (-3) 5 7 = -5 := by
  sorry

end NUMINAMATH_CALUDE_point_movement_theorem_l3897_389790


namespace NUMINAMATH_CALUDE_chocolate_count_l3897_389732

theorem chocolate_count : ∀ x : ℚ,
  let day1_remaining := (3 / 5 : ℚ) * x - 3
  let day2_remaining := (3 / 4 : ℚ) * day1_remaining - 5
  day2_remaining = 10 → x = 105 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_count_l3897_389732


namespace NUMINAMATH_CALUDE_wildlife_population_estimate_l3897_389725

theorem wildlife_population_estimate 
  (tagged_released : ℕ) 
  (later_captured : ℕ) 
  (tagged_in_sample : ℕ) 
  (h1 : tagged_released = 1200)
  (h2 : later_captured = 1000)
  (h3 : tagged_in_sample = 100) :
  (tagged_released * later_captured) / tagged_in_sample = 12000 :=
by sorry

end NUMINAMATH_CALUDE_wildlife_population_estimate_l3897_389725


namespace NUMINAMATH_CALUDE_solve_equation_l3897_389778

theorem solve_equation : ∃ x : ℚ, 25 - (3 * 5) = (2 * x) + 1 ∧ x = 9/2 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3897_389778


namespace NUMINAMATH_CALUDE_ellipse_equation_l3897_389719

/-- Given an ellipse centered at the origin with foci on the x-axis,
    focal length 4, and eccentricity √2/2, its equation is x²/8 + y²/4 = 1 -/
theorem ellipse_equation (x y : ℝ) :
  let focal_length : ℝ := 4
  let eccentricity : ℝ := Real.sqrt 2 / 2
  x^2 / 8 + y^2 / 4 = 1 := by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l3897_389719


namespace NUMINAMATH_CALUDE_brians_books_l3897_389704

theorem brians_books (x : ℕ) : 
  x + 2 * 15 + (x + 2 * 15) / 2 = 75 → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_brians_books_l3897_389704


namespace NUMINAMATH_CALUDE_arithmetic_sequence_x_value_l3897_389740

/-- An arithmetic sequence with first four terms 1, x, a, and 2x has x = 2 -/
theorem arithmetic_sequence_x_value (x a : ℝ) : 
  (∃ d : ℝ, x = 1 + d ∧ a = x + d ∧ 2*x = a + d) → x = 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_x_value_l3897_389740


namespace NUMINAMATH_CALUDE_max_group_size_problem_l3897_389762

/-- The maximum number of people in a group for two classes with given total students and leftovers -/
def max_group_size (class1_total : ℕ) (class2_total : ℕ) (class1_leftover : ℕ) (class2_leftover : ℕ) : ℕ :=
  Nat.gcd (class1_total - class1_leftover) (class2_total - class2_leftover)

/-- Theorem stating that the maximum group size for the given problem is 16 -/
theorem max_group_size_problem : max_group_size 69 86 5 6 = 16 := by
  sorry

end NUMINAMATH_CALUDE_max_group_size_problem_l3897_389762


namespace NUMINAMATH_CALUDE_f_range_at_1_3_l3897_389781

def f (a b x y : ℝ) : ℝ := a * (x^3 + 3*x) + b * (y^2 + 2*y + 1)

theorem f_range_at_1_3 (a b : ℝ) (h1 : 1 ≤ f a b 1 2) (h2 : f a b 1 2 ≤ 2) 
  (h3 : 2 ≤ f a b 3 4) (h4 : f a b 3 4 ≤ 5) : 
  (3/2 : ℝ) ≤ f a b 1 3 ∧ f a b 1 3 ≤ 4 := by
sorry

end NUMINAMATH_CALUDE_f_range_at_1_3_l3897_389781


namespace NUMINAMATH_CALUDE_glove_pair_probability_l3897_389748

def num_black_pairs : ℕ := 6
def num_beige_pairs : ℕ := 4

def total_gloves : ℕ := 2 * (num_black_pairs + num_beige_pairs)

def prob_black_pair : ℚ := (num_black_pairs * 2 / total_gloves) * ((num_black_pairs * 2 - 1) / (total_gloves - 1))
def prob_beige_pair : ℚ := (num_beige_pairs * 2 / total_gloves) * ((num_beige_pairs * 2 - 1) / (total_gloves - 1))

theorem glove_pair_probability :
  prob_black_pair + prob_beige_pair = 47 / 95 := by
  sorry

end NUMINAMATH_CALUDE_glove_pair_probability_l3897_389748


namespace NUMINAMATH_CALUDE_parking_garage_spaces_l3897_389760

theorem parking_garage_spaces (level1 level2 level3 level4 : ℕ) : 
  level1 = 90 →
  level3 = level2 + 12 →
  level4 = level3 - 9 →
  level1 + level2 + level3 + level4 = 399 →
  level2 = level1 + 8 :=
by
  sorry

end NUMINAMATH_CALUDE_parking_garage_spaces_l3897_389760


namespace NUMINAMATH_CALUDE_trajectory_of_M_l3897_389761

/-- Given points A(-1,0) and B(1,0), and a point M(x,y), if the ratio of the slope of AM
    to the slope of BM is 3, then x = -2 -/
theorem trajectory_of_M (x y : ℝ) (hx : x ≠ 1 ∧ x ≠ -1) (hy : y ≠ 0) :
  (y / (x + 1)) / (y / (x - 1)) = 3 → x = -2 := by
  sorry

#check trajectory_of_M

end NUMINAMATH_CALUDE_trajectory_of_M_l3897_389761


namespace NUMINAMATH_CALUDE_line_perp_parallel_implies_planes_perp_l3897_389742

/-- A line in 3D space -/
structure Line3D where
  -- Add necessary fields

/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields

/-- Perpendicularity between a line and a plane -/
def Line3D.perp (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Parallelism between a line and a plane -/
def Line3D.parallel (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Perpendicularity between two planes -/
def Plane3D.perp (p1 p2 : Plane3D) : Prop :=
  sorry

/-- Theorem: If a line is perpendicular to one plane and parallel to another, 
    then the two planes are perpendicular to each other -/
theorem line_perp_parallel_implies_planes_perp 
  (l : Line3D) (α β : Plane3D) (h1 : l.perp α) (h2 : l.parallel β) : 
  α.perp β :=
sorry

end NUMINAMATH_CALUDE_line_perp_parallel_implies_planes_perp_l3897_389742


namespace NUMINAMATH_CALUDE_books_sold_l3897_389784

/-- Given Kaleb's initial and final book counts, along with the number of new books bought,
    prove the number of books he sold. -/
theorem books_sold (initial : ℕ) (new_bought : ℕ) (final : ℕ) 
    (h1 : initial = 34) 
    (h2 : new_bought = 7) 
    (h3 : final = 24) : 
  initial - final + new_bought = 17 := by
  sorry

end NUMINAMATH_CALUDE_books_sold_l3897_389784


namespace NUMINAMATH_CALUDE_root_form_sum_l3897_389712

/-- The cubic polynomial 2x^3 + 3x^2 - 5x - 2 = 0 has a real root of the form (∛p + ∛q + 2)/r 
    where p, q, and r are positive integers. -/
def has_root_of_form (p q r : ℕ+) : Prop :=
  ∃ x : ℝ, 2 * x^3 + 3 * x^2 - 5 * x - 2 = 0 ∧
           x = (Real.rpow p (1/3 : ℝ) + Real.rpow q (1/3 : ℝ) + 2) / r

/-- If the cubic polynomial has a root of the specified form, then p + q + r = 10. -/
theorem root_form_sum (p q r : ℕ+) : has_root_of_form p q r → p + q + r = 10 := by
  sorry

end NUMINAMATH_CALUDE_root_form_sum_l3897_389712


namespace NUMINAMATH_CALUDE_company_manager_fraction_l3897_389787

/-- Given a company with female managers, total female employees, and the condition that
    the fraction of managers is the same for all employees and male employees,
    prove that the fraction of employees who are managers is 0.4 -/
theorem company_manager_fraction (total_female_employees : ℕ) (female_managers : ℕ)
    (h1 : female_managers = 200)
    (h2 : total_female_employees = 500)
    (h3 : ∃ (f : ℚ), f * (total_female_employees : ℚ) = (female_managers : ℚ) ∧
                     f * ((total_female_employees : ℚ) - (female_managers : ℚ)) = 
                     (female_managers : ℚ) * ((total_female_employees : ℚ) / (female_managers : ℚ) - 1)) :
  ∃ (f : ℚ), f = 0.4 ∧ 
    f * (total_female_employees : ℚ) = (female_managers : ℚ) ∧
    f * ((total_female_employees : ℚ) - (female_managers : ℚ)) = 
    (female_managers : ℚ) * ((total_female_employees : ℚ) / (female_managers : ℚ) - 1) := by
  sorry


end NUMINAMATH_CALUDE_company_manager_fraction_l3897_389787


namespace NUMINAMATH_CALUDE_optimal_rental_plan_l3897_389797

/-- Represents the rental plan for cars -/
structure RentalPlan where
  typeA : ℕ
  typeB : ℕ

/-- Checks if a rental plan is valid according to the given conditions -/
def isValidPlan (plan : RentalPlan) : Prop :=
  plan.typeA + plan.typeB = 8 ∧
  35 * plan.typeA + 30 * plan.typeB ≥ 255 ∧
  400 * plan.typeA + 320 * plan.typeB ≤ 3000

/-- Calculates the total cost of a rental plan -/
def totalCost (plan : RentalPlan) : ℕ :=
  400 * plan.typeA + 320 * plan.typeB

/-- The optimal rental plan -/
def optimalPlan : RentalPlan :=
  { typeA := 3, typeB := 5 }

theorem optimal_rental_plan :
  isValidPlan optimalPlan ∧
  totalCost optimalPlan = 2800 ∧
  ∀ plan, isValidPlan plan → totalCost plan ≥ totalCost optimalPlan :=
by sorry

end NUMINAMATH_CALUDE_optimal_rental_plan_l3897_389797


namespace NUMINAMATH_CALUDE_absolute_value_ratio_l3897_389783

theorem absolute_value_ratio (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a^2 + b^2 = 10*a*b) :
  |((a + b) / (a - b))| = Real.sqrt 6 / 2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_ratio_l3897_389783
