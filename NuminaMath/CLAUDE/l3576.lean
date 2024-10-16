import Mathlib

namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3576_357678

/-- Given an arithmetic sequence {aₙ} with sum Sₙ of the first n terms,
    if S₆ > S₇ > S₅, then the common difference d < 0 and |a₆| > |a₇| -/
theorem arithmetic_sequence_properties
  (a : ℕ → ℝ)  -- The arithmetic sequence
  (S : ℕ → ℝ)  -- The sum function
  (d : ℝ)      -- The common difference
  (h1 : ∀ n, S n = n * (a 1 + a n) / 2)  -- Definition of Sₙ
  (h2 : ∀ n, a (n + 1) = a n + d)        -- Definition of arithmetic sequence
  (h3 : S 6 > S 7)                       -- Given condition
  (h4 : S 7 > S 5)                       -- Given condition
  : d < 0 ∧ |a 6| > |a 7| := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3576_357678


namespace NUMINAMATH_CALUDE_direct_proportion_through_point_one_two_l3576_357608

/-- The equation of a direct proportion function passing through (1, 2) -/
theorem direct_proportion_through_point_one_two :
  ∀ (k : ℝ), (∃ f : ℝ → ℝ, (∀ x, f x = k * x) ∧ f 1 = 2) → 
  (∀ x, k * x = 2 * x) :=
sorry

end NUMINAMATH_CALUDE_direct_proportion_through_point_one_two_l3576_357608


namespace NUMINAMATH_CALUDE_least_4_light_four_digit_l3576_357624

def is_4_light (n : ℕ) : Prop := n % 9 < 4

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

theorem least_4_light_four_digit : 
  (∀ n : ℕ, is_four_digit n → is_4_light n → 1000 ≤ n) ∧ is_four_digit 1000 ∧ is_4_light 1000 :=
sorry

end NUMINAMATH_CALUDE_least_4_light_four_digit_l3576_357624


namespace NUMINAMATH_CALUDE_translation_result_l3576_357676

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translates a point horizontally -/
def translateHorizontal (p : Point) (dx : ℝ) : Point :=
  ⟨p.x + dx, p.y⟩

/-- Translates a point vertically -/
def translateVertical (p : Point) (dy : ℝ) : Point :=
  ⟨p.x, p.y + dy⟩

/-- The initial point M -/
def M : Point := ⟨2, 5⟩

/-- The resulting point after translations -/
def resultingPoint : Point :=
  translateVertical (translateHorizontal M (-2)) (-3)

theorem translation_result :
  resultingPoint = ⟨0, 2⟩ := by sorry

end NUMINAMATH_CALUDE_translation_result_l3576_357676


namespace NUMINAMATH_CALUDE_initial_cats_in_shelter_l3576_357647

theorem initial_cats_in_shelter (initial_cats : ℕ) : 
  (initial_cats / 3 : ℚ) = (initial_cats / 3 : ℕ) →
  (4 * initial_cats / 3 + 8 * initial_cats / 3 : ℚ) = 60 →
  initial_cats = 15 := by
sorry

end NUMINAMATH_CALUDE_initial_cats_in_shelter_l3576_357647


namespace NUMINAMATH_CALUDE_average_of_first_20_multiples_of_17_l3576_357689

theorem average_of_first_20_multiples_of_17 : 
  let n : ℕ := 20
  let first_multiple : ℕ := 17
  let sum_of_multiples : ℕ := n * (first_multiple + n * first_multiple) / 2
  (sum_of_multiples : ℚ) / n = 178.5 := by sorry

end NUMINAMATH_CALUDE_average_of_first_20_multiples_of_17_l3576_357689


namespace NUMINAMATH_CALUDE_trigonometric_expression_l3576_357672

theorem trigonometric_expression (α : Real) 
  (h : Real.sin (π/4 + α) = 1/2) : 
  (Real.sin (5*π/4 + α) / Real.cos (9*π/4 + α)) * Real.cos (7*π/4 - α) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_l3576_357672


namespace NUMINAMATH_CALUDE_min_side_length_l3576_357631

/-- Given two triangles PQR and SQR sharing side QR, prove that the minimum possible
    integral length of QR is 15 cm, given the lengths of other sides. -/
theorem min_side_length (PQ PR SR QS : ℝ) (hPQ : PQ = 7) (hPR : PR = 15) (hSR : SR = 10) (hQS : QS = 25) :
  ∃ (QR : ℕ), QR ≥ 15 ∧ ∀ (n : ℕ), n ≥ 15 → (n : ℝ) > PR - PQ ∧ (n : ℝ) > QS - SR :=
by sorry

end NUMINAMATH_CALUDE_min_side_length_l3576_357631


namespace NUMINAMATH_CALUDE_john_driving_equation_l3576_357613

def speed_before_lunch : ℝ := 60
def speed_after_lunch : ℝ := 90
def total_distance : ℝ := 300
def total_time : ℝ := 4
def lunch_break : ℝ := 0.5

theorem john_driving_equation (t : ℝ) : 
  speed_before_lunch * t + speed_after_lunch * (total_time - lunch_break - t) = total_distance :=
sorry

end NUMINAMATH_CALUDE_john_driving_equation_l3576_357613


namespace NUMINAMATH_CALUDE_kyle_spent_one_third_l3576_357688

def dave_money : ℕ := 46
def kyle_initial_money : ℕ := 3 * dave_money - 12
def kyle_remaining_money : ℕ := 84

theorem kyle_spent_one_third : 
  (kyle_initial_money - kyle_remaining_money) / kyle_initial_money = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_kyle_spent_one_third_l3576_357688


namespace NUMINAMATH_CALUDE_organism_extinction_probability_l3576_357626

theorem organism_extinction_probability 
  (p q : ℝ) 
  (h1 : p = 0.6) 
  (h2 : q = 0.4) 
  (h3 : p + q = 1) : 
  q / p = 2 / 3 :=
by sorry

end NUMINAMATH_CALUDE_organism_extinction_probability_l3576_357626


namespace NUMINAMATH_CALUDE_perimeter_after_adding_tiles_l3576_357622

/-- Represents a rectangular figure composed of square tiles -/
structure TiledRectangle where
  length : ℕ
  width : ℕ
  extra_tiles : ℕ

/-- Calculates the perimeter of a TiledRectangle -/
def perimeter (rect : TiledRectangle) : ℕ :=
  2 * (rect.length + rect.width)

/-- The initial rectangular figure -/
def initial_rectangle : TiledRectangle :=
  { length := 5, width := 2, extra_tiles := 1 }

theorem perimeter_after_adding_tiles :
  ∃ (final_rect : TiledRectangle),
    perimeter initial_rectangle = 16 ∧
    final_rect.length + final_rect.width = initial_rectangle.length + initial_rectangle.width + 2 ∧
    final_rect.extra_tiles = initial_rectangle.extra_tiles + 2 ∧
    perimeter final_rect = 18 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_after_adding_tiles_l3576_357622


namespace NUMINAMATH_CALUDE_trigonometric_identities_l3576_357691

theorem trigonometric_identities (θ : ℝ) :
  (2 * Real.cos ((3 / 2) * Real.pi + θ) + Real.cos (Real.pi + θ)) /
  (3 * Real.sin (Real.pi - θ) + 2 * Real.sin ((5 / 2) * Real.pi + θ)) = 1 / 5 →
  (Real.tan θ = 3 / 13 ∧
   Real.sin θ ^ 2 + 3 * Real.sin θ * Real.cos θ = 20160 / 28561) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l3576_357691


namespace NUMINAMATH_CALUDE_original_number_is_45_l3576_357660

theorem original_number_is_45 (x : ℝ) : x - 30 = x / 3 → x = 45 := by sorry

end NUMINAMATH_CALUDE_original_number_is_45_l3576_357660


namespace NUMINAMATH_CALUDE_isosceles_triangle_side_lengths_l3576_357658

/-- Represents a triangle with side lengths a, b, and c. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  positive_a : 0 < a
  positive_b : 0 < b
  positive_c : 0 < c
  triangle_inequality_ab : a + b > c
  triangle_inequality_bc : b + c > a
  triangle_inequality_ca : c + a > b

/-- Represents an isosceles triangle. -/
structure IsoscelesTriangle extends Triangle where
  isosceles : (a = b) ∨ (b = c) ∨ (c = a)

/-- 
Given an isosceles triangle with perimeter 17 and one side length 4,
prove that the other two sides must both be 6.5.
-/
theorem isosceles_triangle_side_lengths :
  ∀ t : IsoscelesTriangle,
    t.a + t.b + t.c = 17 →
    (t.a = 4 ∨ t.b = 4 ∨ t.c = 4) →
    ((t.a = 6.5 ∧ t.b = 6.5 ∧ t.c = 4) ∨
     (t.a = 6.5 ∧ t.b = 4 ∧ t.c = 6.5) ∨
     (t.a = 4 ∧ t.b = 6.5 ∧ t.c = 6.5)) := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_side_lengths_l3576_357658


namespace NUMINAMATH_CALUDE_line_point_k_value_l3576_357640

/-- A line contains the points (7, 10), (-3, k), and (-11, 5). The value of k is 65/9. -/
theorem line_point_k_value :
  ∀ (k : ℚ),
  (∃ (m b : ℚ),
    (7 : ℚ) * m + b = 10 ∧
    (-3 : ℚ) * m + b = k ∧
    (-11 : ℚ) * m + b = 5) →
  k = 65 / 9 := by
sorry

end NUMINAMATH_CALUDE_line_point_k_value_l3576_357640


namespace NUMINAMATH_CALUDE_binomial_coefficient_two_l3576_357620

theorem binomial_coefficient_two (n : ℕ+) : Nat.choose n 2 = n * (n - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_two_l3576_357620


namespace NUMINAMATH_CALUDE_number_operation_result_l3576_357632

theorem number_operation_result (x : ℝ) : (x - 5) / 7 = 7 → (x - 24) / 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_number_operation_result_l3576_357632


namespace NUMINAMATH_CALUDE_equal_costs_at_twenty_l3576_357617

/-- Represents the cost function for company A -/
def cost_A (x : ℝ) : ℝ := 450 * x + 1000

/-- Represents the cost function for company B -/
def cost_B (x : ℝ) : ℝ := 500 * x

/-- Theorem stating that the costs are equal when 20 desks are purchased -/
theorem equal_costs_at_twenty :
  ∃ (x : ℝ), x = 20 ∧ cost_A x = cost_B x :=
sorry

end NUMINAMATH_CALUDE_equal_costs_at_twenty_l3576_357617


namespace NUMINAMATH_CALUDE_zoo_badge_problem_l3576_357685

/-- Represents the commemorative badges sold by the zoo -/
inductive Badge
| A
| B

/-- Represents the cost and selling prices of badges -/
structure BadgePrices where
  cost_A : ℝ
  cost_B : ℝ
  sell_A : ℝ
  sell_B : ℝ

/-- Represents the purchasing plan for badges -/
structure PurchasePlan where
  num_A : ℕ
  num_B : ℕ

/-- Calculates the total cost of a purchasing plan -/
def total_cost (prices : BadgePrices) (plan : PurchasePlan) : ℝ :=
  prices.cost_A * plan.num_A + prices.cost_B * plan.num_B

/-- Calculates the total profit of a purchasing plan -/
def total_profit (prices : BadgePrices) (plan : PurchasePlan) : ℝ :=
  (prices.sell_A - prices.cost_A) * plan.num_A + (prices.sell_B - prices.cost_B) * plan.num_B

/-- Theorem representing the zoo's badge problem -/
theorem zoo_badge_problem (prices : BadgePrices) 
  (h1 : prices.cost_A = prices.cost_B + 4)
  (h2 : 6 * prices.cost_A = 10 * prices.cost_B)
  (h3 : prices.sell_A = 13)
  (h4 : prices.sell_B = 8)
  : 
  prices.cost_A = 10 ∧ 
  prices.cost_B = 6 ∧
  ∃ (optimal_plan : PurchasePlan),
    optimal_plan.num_A + optimal_plan.num_B = 400 ∧
    total_cost prices optimal_plan ≤ 2800 ∧
    total_profit prices optimal_plan = 900 ∧
    ∀ (plan : PurchasePlan),
      plan.num_A + plan.num_B = 400 →
      total_cost prices plan ≤ 2800 →
      total_profit prices plan ≤ total_profit prices optimal_plan :=
by sorry


end NUMINAMATH_CALUDE_zoo_badge_problem_l3576_357685


namespace NUMINAMATH_CALUDE_cartoon_length_missy_cartoon_length_l3576_357699

/-- The length of a cartoon given specific TV watching conditions -/
theorem cartoon_length (reality_shows : ℕ) (reality_show_length : ℕ) (total_time : ℕ) : ℕ :=
  let cartoon_length := total_time - reality_shows * reality_show_length
  by
    sorry

/-- The length of Missy's cartoon is 10 minutes -/
theorem missy_cartoon_length : cartoon_length 5 28 150 = 10 := by
  sorry

end NUMINAMATH_CALUDE_cartoon_length_missy_cartoon_length_l3576_357699


namespace NUMINAMATH_CALUDE_product_of_fractions_l3576_357662

theorem product_of_fractions :
  (8 / 4) * (10 / 5) * (21 / 14) * (16 / 8) * (45 / 15) * (30 / 10) * (49 / 35) * (32 / 16) = 302.4 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l3576_357662


namespace NUMINAMATH_CALUDE_fraction_exponent_product_l3576_357666

theorem fraction_exponent_product : (5 / 6 : ℚ)^2 * (2 / 3 : ℚ)^3 = 50 / 243 := by sorry

end NUMINAMATH_CALUDE_fraction_exponent_product_l3576_357666


namespace NUMINAMATH_CALUDE_no_equal_product_l3576_357614

theorem no_equal_product (x y : ℕ) : x * (x + 1) ≠ 4 * y * (y + 1) := by
  sorry

end NUMINAMATH_CALUDE_no_equal_product_l3576_357614


namespace NUMINAMATH_CALUDE_twelfth_term_value_l3576_357667

/-- The nth term of a geometric sequence -/
def geometric_term (a₁ : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a₁ * r^(n - 1)

/-- The 12th term of the specific geometric sequence -/
def twelfth_term : ℚ :=
  geometric_term 5 (2/5) 12

theorem twelfth_term_value : twelfth_term = 10240/48828125 := by
  sorry

end NUMINAMATH_CALUDE_twelfth_term_value_l3576_357667


namespace NUMINAMATH_CALUDE_pole_shortening_l3576_357644

/-- Given a pole of length 20 meters that is shortened by 30%, prove that its new length is 14 meters. -/
theorem pole_shortening (original_length : ℝ) (shortening_percentage : ℝ) (new_length : ℝ) :
  original_length = 20 →
  shortening_percentage = 30 →
  new_length = original_length * (1 - shortening_percentage / 100) →
  new_length = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_pole_shortening_l3576_357644


namespace NUMINAMATH_CALUDE_neither_question_correct_percentage_l3576_357696

theorem neither_question_correct_percentage
  (p_first : ℝ)
  (p_second : ℝ)
  (p_both : ℝ)
  (h1 : p_first = 0.63)
  (h2 : p_second = 0.50)
  (h3 : p_both = 0.33)
  : 1 - (p_first + p_second - p_both) = 0.20 := by
  sorry

end NUMINAMATH_CALUDE_neither_question_correct_percentage_l3576_357696


namespace NUMINAMATH_CALUDE_sqrt_11_plus_1_bounds_l3576_357636

-- Define the theorem
theorem sqrt_11_plus_1_bounds : 4 < Real.sqrt 11 + 1 ∧ Real.sqrt 11 + 1 < 5 := by
  sorry

#check sqrt_11_plus_1_bounds

end NUMINAMATH_CALUDE_sqrt_11_plus_1_bounds_l3576_357636


namespace NUMINAMATH_CALUDE_count_valid_lists_l3576_357655

/-- A structure representing a list of five integers with the given properties -/
structure IntegerList :=
  (a b : ℕ+)
  (h1 : a < b)
  (h2 : 2 * a.val + 3 * b.val = 124)

/-- The number of valid integer lists -/
def validListCount : ℕ := sorry

/-- Theorem stating that there are exactly 8 valid integer lists -/
theorem count_valid_lists : validListCount = 8 := by sorry

end NUMINAMATH_CALUDE_count_valid_lists_l3576_357655


namespace NUMINAMATH_CALUDE_chess_and_go_problem_l3576_357611

theorem chess_and_go_problem (chess_price go_price : ℝ) 
  (h1 : 6 * chess_price + 5 * go_price = 190)
  (h2 : 8 * chess_price + 10 * go_price = 320)
  (budget : ℝ) (total_sets : ℕ)
  (h3 : budget ≤ 1800)
  (h4 : total_sets = 100) :
  chess_price = 15 ∧ 
  go_price = 20 ∧ 
  ∃ (min_chess : ℕ), min_chess ≥ 40 ∧ 
    chess_price * min_chess + go_price * (total_sets - min_chess) ≤ budget :=
by sorry

end NUMINAMATH_CALUDE_chess_and_go_problem_l3576_357611


namespace NUMINAMATH_CALUDE_probability_three_even_dice_l3576_357612

def num_dice : ℕ := 5
def faces_per_die : ℕ := 20
def target_even : ℕ := 3

theorem probability_three_even_dice :
  let p_even : ℚ := 1 / 2  -- Probability of rolling an even number on a single die
  let p_arrangement : ℚ := p_even ^ target_even * (1 - p_even) ^ (num_dice - target_even)
  let num_arrangements : ℕ := Nat.choose num_dice target_even
  num_arrangements * p_arrangement = 5 / 16 := by
sorry

end NUMINAMATH_CALUDE_probability_three_even_dice_l3576_357612


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l3576_357656

/-- Given a quadratic equation 5 * x^2 + 14 * x + 5 = 0 with two reciprocal roots,
    the coefficient of the squared term is 5. -/
theorem quadratic_coefficient (x : ℝ) :
  (5 * x^2 + 14 * x + 5 = 0) →
  (∃ (r₁ r₂ : ℝ), r₁ ≠ 0 ∧ r₂ ≠ 0 ∧ r₁ * r₂ = 1 ∧ 
    (x = r₁ ∨ x = r₂) ∧ 5 * r₁^2 + 14 * r₁ + 5 = 0 ∧ 5 * r₂^2 + 14 * r₂ + 5 = 0) →
  5 = 5 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l3576_357656


namespace NUMINAMATH_CALUDE_max_value_constraint_l3576_357629

theorem max_value_constraint (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a^2 + b^2 + c^2 = 3) :
  4 * a * b * Real.sqrt 3 + 12 * b * c ≤ Real.sqrt 39 :=
by sorry

end NUMINAMATH_CALUDE_max_value_constraint_l3576_357629


namespace NUMINAMATH_CALUDE_twelveSidedFigureArea_l3576_357627

/-- A point in 2D space --/
structure Point where
  x : ℝ
  y : ℝ

/-- A polygon defined by a list of vertices --/
structure Polygon where
  vertices : List Point

/-- The area of a polygon --/
noncomputable def area (p : Polygon) : ℝ := sorry

/-- Our specific 12-sided figure --/
def twelveSidedFigure : Polygon := {
  vertices := [
    { x := 2, y := 1 }, { x := 3, y := 2 }, { x := 3, y := 3 }, { x := 5, y := 3 },
    { x := 6, y := 4 }, { x := 5, y := 5 }, { x := 4, y := 5 }, { x := 3, y := 6 },
    { x := 2, y := 5 }, { x := 2, y := 4 }, { x := 1, y := 3 }, { x := 2, y := 2 }
  ]
}

theorem twelveSidedFigureArea : area twelveSidedFigure = 12 := by sorry

end NUMINAMATH_CALUDE_twelveSidedFigureArea_l3576_357627


namespace NUMINAMATH_CALUDE_marcus_pebble_count_l3576_357609

/-- Given an initial number of pebbles, calculate the number of pebbles
    after skipping half and receiving more. -/
def final_pebble_count (initial : ℕ) (received : ℕ) : ℕ :=
  initial / 2 + received

/-- Theorem stating that given 18 initial pebbles and 30 received pebbles,
    the final count is 39. -/
theorem marcus_pebble_count :
  final_pebble_count 18 30 = 39 := by
  sorry

end NUMINAMATH_CALUDE_marcus_pebble_count_l3576_357609


namespace NUMINAMATH_CALUDE_inequality_preservation_l3576_357659

theorem inequality_preservation (a b : ℝ) (h : a < b) : 2 - a > 2 - b := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l3576_357659


namespace NUMINAMATH_CALUDE_exists_small_angle_between_diagonals_l3576_357680

/-- A convex dodecagon is a convex polygon with 12 sides -/
structure ConvexDodecagon where
  -- We don't need to specify the exact structure, just that it exists
  is_convex : Bool
  num_sides : Nat
  num_sides_eq : num_sides = 12

/-- A diagonal of a polygon is a line segment that connects two non-adjacent vertices -/
structure Diagonal (P : ConvexDodecagon) where
  -- Again, we don't need to specify the exact structure

/-- The angle between two diagonals -/
def angle_between_diagonals (P : ConvexDodecagon) (d1 d2 : Diagonal P) : ℝ :=
  sorry -- We don't need to implement this, just declare it

theorem exists_small_angle_between_diagonals (P : ConvexDodecagon) :
  ∃ (d1 d2 : Diagonal P), angle_between_diagonals P d1 d2 ≤ 3 := by
  sorry


end NUMINAMATH_CALUDE_exists_small_angle_between_diagonals_l3576_357680


namespace NUMINAMATH_CALUDE_max_even_distribution_l3576_357605

/-- Represents the pencil distribution problem --/
def PencilDistribution (initial_pencils : ℕ) (initial_containers : ℕ) (first_addition : ℕ) (second_addition : ℕ) (final_containers : ℕ) : Prop :=
  let total_pencils : ℕ := initial_pencils + first_addition + second_addition
  let even_distribution : ℕ := total_pencils / final_containers
  even_distribution * final_containers ≤ total_pencils ∧
  (even_distribution + 1) * final_containers > total_pencils

/-- Theorem stating the maximum even distribution of pencils --/
theorem max_even_distribution :
  PencilDistribution 150 5 30 47 6 →
  ∃ (n : ℕ), n = 37 ∧ PencilDistribution 150 5 30 47 6 := by
  sorry

#check max_even_distribution

end NUMINAMATH_CALUDE_max_even_distribution_l3576_357605


namespace NUMINAMATH_CALUDE_circle_radius_circle_radius_proof_l3576_357628

/-- The radius of a circle with center (2, -3) passing through (5, -7) is 5 -/
theorem circle_radius : ℝ → Prop :=
  fun r : ℝ =>
    let center : ℝ × ℝ := (2, -3)
    let point : ℝ × ℝ := (5, -7)
    (center.1 - point.1)^2 + (center.2 - point.2)^2 = r^2 → r = 5

/-- Proof of the theorem -/
theorem circle_radius_proof : circle_radius 5 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_circle_radius_proof_l3576_357628


namespace NUMINAMATH_CALUDE_tangent_perpendicular_to_line_l3576_357673

open Real

theorem tangent_perpendicular_to_line (a : ℝ) : 
  let f (x : ℝ) := (2 - cos x) / sin x
  let x₀ : ℝ := π / 2
  let y₀ : ℝ := f x₀
  let m : ℝ := (deriv f) x₀
  (y₀ = 2) → (m * (-1/a) = -1) → a = 1 := by
sorry

end NUMINAMATH_CALUDE_tangent_perpendicular_to_line_l3576_357673


namespace NUMINAMATH_CALUDE_tesseract_triangles_l3576_357661

/-- The number of vertices in a tesseract -/
def tesseract_vertices : ℕ := 16

/-- The number of vertices required to form a triangle -/
def triangle_vertices : ℕ := 3

/-- The number of distinct triangles in a tesseract -/
def distinct_triangles : ℕ := Nat.choose tesseract_vertices triangle_vertices

theorem tesseract_triangles : distinct_triangles = 560 := by
  sorry

end NUMINAMATH_CALUDE_tesseract_triangles_l3576_357661


namespace NUMINAMATH_CALUDE_final_liquid_X_percentage_l3576_357674

/-- Composition of a solution -/
structure Solution :=
  (x : ℝ) -- Percentage of liquid X
  (water : ℝ) -- Percentage of water
  (z : ℝ) -- Percentage of liquid Z

/-- Given conditions -/
def solution_Y : Solution := ⟨20, 55, 25⟩
def initial_weight : ℝ := 12
def evaporated_water : ℝ := 4
def added_Y_weight : ℝ := 3
def solution_B : Solution := ⟨35, 15, 50⟩
def added_B_weight : ℝ := 2
def evaporation_factor : ℝ := 0.75
def solution_D : Solution := ⟨15, 60, 25⟩
def added_D_weight : ℝ := 6

/-- The theorem to prove -/
theorem final_liquid_X_percentage :
  let initial_X := solution_Y.x * initial_weight / 100
  let initial_Z := solution_Y.z * initial_weight / 100
  let remaining_water := solution_Y.water * initial_weight / 100 - evaporated_water
  let added_Y_X := solution_Y.x * added_Y_weight / 100
  let added_Y_water := solution_Y.water * added_Y_weight / 100
  let added_Y_Z := solution_Y.z * added_Y_weight / 100
  let added_B_X := solution_B.x * added_B_weight / 100
  let added_B_water := solution_B.water * added_B_weight / 100
  let added_B_Z := solution_B.z * added_B_weight / 100
  let total_before_evap := initial_X + initial_Z + remaining_water + added_Y_X + added_Y_water + added_Y_Z + added_B_X + added_B_water + added_B_Z
  let total_after_evap := total_before_evap * evaporation_factor
  let evaporated_water_2 := (1 - evaporation_factor) * (remaining_water + added_Y_water + added_B_water)
  let remaining_water_2 := remaining_water + added_Y_water + added_B_water - evaporated_water_2
  let added_D_X := solution_D.x * added_D_weight / 100
  let added_D_water := solution_D.water * added_D_weight / 100
  let added_D_Z := solution_D.z * added_D_weight / 100
  let final_X := initial_X + added_Y_X + added_B_X + added_D_X
  let final_water := remaining_water_2 + added_D_water
  let final_Z := initial_Z + added_Y_Z + added_B_Z + added_D_Z
  let final_total := final_X + final_water + final_Z
  final_X / final_total * 100 = 25.75 := by sorry

end NUMINAMATH_CALUDE_final_liquid_X_percentage_l3576_357674


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l3576_357695

theorem polynomial_division_remainder : ∃ q : Polynomial ℚ, 
  (2 * X^4 + 10 * X^3 - 45 * X^2 - 52 * X + 63) = 
  (X^2 + 6 * X - 7) * q + (48 * X - 70) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l3576_357695


namespace NUMINAMATH_CALUDE_warehouse_problem_l3576_357610

/-- Represents the time (in hours) it takes a team to move all goods in a warehouse -/
structure TeamSpeed :=
  (hours : ℝ)
  (positive : hours > 0)

/-- Represents the division of Team C's time between helping Team A and Team B -/
structure TeamCHelp :=
  (helpA : ℝ)
  (helpB : ℝ)
  (positive_helpA : helpA > 0)
  (positive_helpB : helpB > 0)

/-- The main theorem stating the solution to the warehouse problem -/
theorem warehouse_problem 
  (speedA : TeamSpeed) 
  (speedB : TeamSpeed) 
  (speedC : TeamSpeed)
  (h_speedA : speedA.hours = 6)
  (h_speedB : speedB.hours = 7)
  (h_speedC : speedC.hours = 14) :
  ∃ (help : TeamCHelp),
    help.helpA = 7/4 ∧ 
    help.helpB = 7/2 ∧
    help.helpA + help.helpB = speedA.hours * speedB.hours / (speedA.hours + speedB.hours) ∧
    1 / speedA.hours + 1 / speedC.hours * help.helpA = 1 ∧
    1 / speedB.hours + 1 / speedC.hours * help.helpB = 1 :=
sorry

end NUMINAMATH_CALUDE_warehouse_problem_l3576_357610


namespace NUMINAMATH_CALUDE_arithmetic_equality_l3576_357648

theorem arithmetic_equality : 54 + 98 / 14 + 23 * 17 - 200 - 312 / 6 = 200 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equality_l3576_357648


namespace NUMINAMATH_CALUDE_cinema_seating_arrangement_l3576_357653

def number_of_arrangements (n : ℕ) (must_together : ℕ) (must_not_together : ℕ) : ℕ :=
  (must_together.factorial * (n - must_together + 1).factorial) -
  (must_together.factorial * must_not_together.factorial * (n - must_together - must_not_together + 2).factorial)

theorem cinema_seating_arrangement :
  number_of_arrangements 6 2 2 = 144 := by
  sorry

end NUMINAMATH_CALUDE_cinema_seating_arrangement_l3576_357653


namespace NUMINAMATH_CALUDE_one_third_square_coloring_l3576_357684

theorem one_third_square_coloring (n : ℕ) (k : ℕ) : n = 18 ∧ k = 6 → Nat.choose n k = 18564 := by
  sorry

end NUMINAMATH_CALUDE_one_third_square_coloring_l3576_357684


namespace NUMINAMATH_CALUDE_min_value_zero_l3576_357606

/-- The quadratic form as a function of x, y, and k -/
def f (x y k : ℝ) : ℝ :=
  9 * x^2 - 12 * k * x * y + (4 * k^2 + 3) * y^2 - 6 * x - 3 * y + 9

/-- The theorem stating that 3/2 is the value of k that makes the minimum of f zero -/
theorem min_value_zero (k : ℝ) : 
  (∀ x y : ℝ, f x y k ≥ 0) ∧ (∃ x y : ℝ, f x y k = 0) ↔ k = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_zero_l3576_357606


namespace NUMINAMATH_CALUDE_rectangle_circle_area_ratio_l3576_357630

theorem rectangle_circle_area_ratio :
  ∀ (w r : ℝ),
  w > 0 → r > 0 →
  6 * w = 2 * π * r →
  (2 * w * w) / (π * r^2) = 2 * π / 9 := by
sorry

end NUMINAMATH_CALUDE_rectangle_circle_area_ratio_l3576_357630


namespace NUMINAMATH_CALUDE_salary_percentage_difference_l3576_357693

/-- Given two employees M and N with a total salary and N's individual salary,
    calculate the percentage difference between M's and N's salaries. -/
theorem salary_percentage_difference
  (total_salary : ℚ)
  (n_salary : ℚ)
  (h1 : total_salary = 616)
  (h2 : n_salary = 280) :
  (total_salary - n_salary) / n_salary * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_salary_percentage_difference_l3576_357693


namespace NUMINAMATH_CALUDE_dalton_needs_four_dollars_l3576_357686

/-- The amount of additional money Dalton needs to buy all items -/
def additional_money_needed (jump_rope_cost board_game_cost ball_cost saved_allowance uncle_gift : ℕ) : ℕ :=
  let total_cost := jump_rope_cost + board_game_cost + ball_cost
  let available_money := saved_allowance + uncle_gift
  if total_cost > available_money then
    total_cost - available_money
  else
    0

/-- Theorem stating that Dalton needs $4 more to buy all items -/
theorem dalton_needs_four_dollars : 
  additional_money_needed 7 12 4 6 13 = 4 := by
  sorry

end NUMINAMATH_CALUDE_dalton_needs_four_dollars_l3576_357686


namespace NUMINAMATH_CALUDE_gideon_marbles_fraction_l3576_357645

/-- The fraction of marbles Gideon gave to his sister -/
def fraction_given : ℚ := 3/4

theorem gideon_marbles_fraction :
  ∀ (f : ℚ),
  (100 : ℚ) = 100 →  -- Gideon has 100 marbles
  (45 : ℚ) = 45 →    -- Gideon is currently 45 years old
  2 * ((1 - f) * 100) = (45 + 5 : ℚ) →  -- After giving fraction f and doubling, he gets his age in 5 years
  f = fraction_given :=
by sorry

end NUMINAMATH_CALUDE_gideon_marbles_fraction_l3576_357645


namespace NUMINAMATH_CALUDE_smaller_fraction_problem_l3576_357652

theorem smaller_fraction_problem (x y : ℚ) 
  (sum_cond : x + y = 7/8)
  (prod_cond : x * y = 1/12) :
  min x y = (7 - Real.sqrt 17) / 16 := by
  sorry

end NUMINAMATH_CALUDE_smaller_fraction_problem_l3576_357652


namespace NUMINAMATH_CALUDE_family_ages_solution_l3576_357698

/-- Represents the ages of a father and his two children -/
structure FamilyAges where
  father : ℕ
  son : ℕ
  daughter : ℕ

/-- The conditions of the problem -/
def satisfiesConditions (ages : FamilyAges) : Prop :=
  ages.father + ages.son + ages.daughter = 110 ∧
  ages.son = ages.daughter ∧
  3 * ages.father = 186

/-- The theorem to be proved -/
theorem family_ages_solution :
  ∃ (ages : FamilyAges), satisfiesConditions ages ∧ 
    ages.father = 62 ∧ ages.son = 24 ∧ ages.daughter = 24 := by
  sorry

end NUMINAMATH_CALUDE_family_ages_solution_l3576_357698


namespace NUMINAMATH_CALUDE_triangle_side_length_l3576_357607

theorem triangle_side_length (AB BC AC : ℝ) : 
  AB = 6 → BC = 4 → 2 < AC ∧ AC < 10 → AC = 5 → True :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3576_357607


namespace NUMINAMATH_CALUDE_clouddale_total_rainfall_l3576_357600

/-- Calculates the total annual rainfall given the average monthly rainfall -/
def annual_rainfall (average_monthly : ℝ) : ℝ := average_monthly * 12

/-- Represents the rainfall data for Clouddale -/
structure ClouddaleRainfall where
  avg_2003 : ℝ  -- Average monthly rainfall in 2003
  increase_rate : ℝ  -- Percentage increase in 2004

/-- Theorem stating the total rainfall for both years in Clouddale -/
theorem clouddale_total_rainfall (data : ClouddaleRainfall) 
  (h1 : data.avg_2003 = 45)
  (h2 : data.increase_rate = 0.05) : 
  (annual_rainfall data.avg_2003 = 540) ∧ 
  (annual_rainfall (data.avg_2003 * (1 + data.increase_rate)) = 567) := by
  sorry

#eval annual_rainfall 45
#eval annual_rainfall (45 * 1.05)

end NUMINAMATH_CALUDE_clouddale_total_rainfall_l3576_357600


namespace NUMINAMATH_CALUDE_city_population_problem_l3576_357649

/-- Given three cities with the following conditions:
    - Richmond has 1000 more people than Victoria
    - Victoria has 4 times as many people as another city
    - Richmond has 3000 people
    Prove that the other city has 500 people. -/
theorem city_population_problem (richmond victoria other : ℕ) : 
  richmond = victoria + 1000 →
  victoria = 4 * other →
  richmond = 3000 →
  other = 500 := by
  sorry

end NUMINAMATH_CALUDE_city_population_problem_l3576_357649


namespace NUMINAMATH_CALUDE_power_product_equals_one_l3576_357601

theorem power_product_equals_one : (0.25 ^ 2023) * (4 ^ 2023) = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equals_one_l3576_357601


namespace NUMINAMATH_CALUDE_triangle_area_implies_sin_A_l3576_357679

-- Define a triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (h_positive : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_inequality : a < b + c ∧ b < a + c ∧ c < a + b)

-- Define the area of the triangle
def area (t : Triangle) : ℝ := t.a^2 - (t.b - t.c)^2

-- State the theorem
theorem triangle_area_implies_sin_A (t : Triangle) (h_area : area t = t.a^2 - (t.b - t.c)^2) :
  let A := Real.arccos ((t.b^2 + t.c^2 - t.a^2) / (2 * t.b * t.c))
  Real.sin A = 8 / 17 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_implies_sin_A_l3576_357679


namespace NUMINAMATH_CALUDE_paper_folding_thickness_l3576_357618

def initial_thickness : ℝ := 0.1
def num_folds : ℕ := 4

theorem paper_folding_thickness :
  initial_thickness * (2 ^ num_folds) = 1.6 := by
  sorry

end NUMINAMATH_CALUDE_paper_folding_thickness_l3576_357618


namespace NUMINAMATH_CALUDE_no_solution_equation1_solutions_equation2_l3576_357619

-- Define the equations
def equation1 (x : ℝ) : Prop := 1 + (3 * x) / (x - 2) = 6 / (x - 2)
def equation2 (x : ℝ) : Prop := x^2 + x - 6 = 0

-- Theorem for the first equation
theorem no_solution_equation1 : ¬ ∃ x : ℝ, equation1 x := by sorry

-- Theorem for the second equation
theorem solutions_equation2 : 
  (equation2 (-3) ∧ equation2 2) ∧ 
  (∀ x : ℝ, equation2 x → (x = -3 ∨ x = 2)) := by sorry

end NUMINAMATH_CALUDE_no_solution_equation1_solutions_equation2_l3576_357619


namespace NUMINAMATH_CALUDE_polygon_sides_from_angle_sum_l3576_357642

theorem polygon_sides_from_angle_sum (sum_of_angles : ℝ) :
  sum_of_angles = 1080 → ∃ n : ℕ, n = 8 ∧ sum_of_angles = 180 * (n - 2) := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_from_angle_sum_l3576_357642


namespace NUMINAMATH_CALUDE_abc_sum_equals_888_l3576_357641

/-- Given ABC + ABC + ABC = 888, where A, B, and C are all different single digit numbers, prove A = 2 -/
theorem abc_sum_equals_888 (A B C : ℕ) : 
  (100 * A + 10 * B + C) * 3 = 888 →
  A < 10 ∧ B < 10 ∧ C < 10 →
  A ≠ B ∧ B ≠ C ∧ A ≠ C →
  A = 2 := by
sorry

end NUMINAMATH_CALUDE_abc_sum_equals_888_l3576_357641


namespace NUMINAMATH_CALUDE_smallest_five_digit_multiple_l3576_357615

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

def is_divisible_by (n m : ℕ) : Prop := m ∣ n

theorem smallest_five_digit_multiple : 
  (∀ k : ℕ, is_five_digit k ∧ 
            is_divisible_by k 2 ∧ 
            is_divisible_by k 3 ∧ 
            is_divisible_by k 5 ∧ 
            is_divisible_by k 7 ∧ 
            is_divisible_by k 11 
            → k ≥ 11550) ∧ 
  is_five_digit 11550 ∧ 
  is_divisible_by 11550 2 ∧ 
  is_divisible_by 11550 3 ∧ 
  is_divisible_by 11550 5 ∧ 
  is_divisible_by 11550 7 ∧ 
  is_divisible_by 11550 11 :=
by sorry

end NUMINAMATH_CALUDE_smallest_five_digit_multiple_l3576_357615


namespace NUMINAMATH_CALUDE_homothety_transforms_circles_l3576_357603

/-- Two circles in a plane -/
structure TangentCircles where
  S₁ : Set (ℝ × ℝ)
  S₂ : Set (ℝ × ℝ)
  O₁ : ℝ × ℝ
  O₂ : ℝ × ℝ
  r : ℝ
  R : ℝ
  K : ℝ × ℝ
  h_circle₁ : ∀ p ∈ S₁, dist p O₁ = r
  h_circle₂ : ∀ p ∈ S₂, dist p O₂ = R
  h_tangent : K ∈ S₁ ∧ K ∈ S₂
  h_external : dist O₁ O₂ = r + R

/-- Homothety transformation -/
def homothety (center : ℝ × ℝ) (k : ℝ) (p : ℝ × ℝ) : ℝ × ℝ :=
  (center.1 + k * (p.1 - center.1), center.2 + k * (p.2 - center.2))

/-- Main theorem: Homothety transforms one circle into another -/
theorem homothety_transforms_circles (tc : TangentCircles) :
  ∃ h : Set (ℝ × ℝ) → Set (ℝ × ℝ),
    h tc.S₁ = tc.S₂ ∧
    ∀ p ∈ tc.S₁, h {p} = {homothety tc.K (tc.R / tc.r) p} :=
  sorry

end NUMINAMATH_CALUDE_homothety_transforms_circles_l3576_357603


namespace NUMINAMATH_CALUDE_tens_digit_of_9_pow_2023_l3576_357616

theorem tens_digit_of_9_pow_2023 : ∃ n : ℕ, n ≥ 10 ∧ n < 100 ∧ 9^2023 % 100 = n ∧ (n / 10) % 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_9_pow_2023_l3576_357616


namespace NUMINAMATH_CALUDE_no_odd_4digit_div5_no05_l3576_357634

theorem no_odd_4digit_div5_no05 : 
  ¬ ∃ n : ℕ, 
    1000 ≤ n ∧ n < 10000 ∧  -- 4-digit
    n % 2 = 1 ∧             -- odd
    n % 5 = 0 ∧             -- divisible by 5
    (∀ d : ℕ, d < 4 → (n / 10^d) % 10 ≠ 0 ∧ (n / 10^d) % 10 ≠ 5) -- no 0 or 5 digits
    := by sorry

end NUMINAMATH_CALUDE_no_odd_4digit_div5_no05_l3576_357634


namespace NUMINAMATH_CALUDE_cylinder_radius_problem_l3576_357637

theorem cylinder_radius_problem (h : ℝ) (r : ℝ) :
  h = 2 →
  (π * (r + 5)^2 * h - π * r^2 * h = π * r^2 * (h + 4) - π * r^2 * h) →
  r = (5 + 5 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_radius_problem_l3576_357637


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3576_357651

def A : Set Int := {-1, 0, 1, 2, 3}
def B : Set Int := {-3, -1, 1, 3, 5}

theorem intersection_of_A_and_B : A ∩ B = {-1, 1, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3576_357651


namespace NUMINAMATH_CALUDE_exam_max_marks_l3576_357633

theorem exam_max_marks (victor_score : ℝ) (victor_percentage : ℝ) (max_marks : ℝ) : 
  victor_score = 184 → 
  victor_percentage = 0.92 → 
  victor_score = victor_percentage * max_marks → 
  max_marks = 200 := by
sorry

end NUMINAMATH_CALUDE_exam_max_marks_l3576_357633


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_l3576_357687

theorem line_tangent_to_circle (k : ℝ) : 
  (∀ x y : ℝ, y = k * (x + Real.sqrt 3) → x^2 + (y - 1)^2 = 1 → 
    ∀ x' y' : ℝ, y' = k * (x' + Real.sqrt 3) → x'^2 + (y' - 1)^2 ≥ 1) →
  k = Real.sqrt 3 ∨ k = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_l3576_357687


namespace NUMINAMATH_CALUDE_min_value_of_function_l3576_357639

theorem min_value_of_function :
  let f : ℝ → ℝ := λ x => 5/4 - Real.sin x^2 - 3 * Real.cos x
  ∃ x₀ : ℝ, ∀ x : ℝ, f x ≥ f x₀ ∧ f x₀ = -7/4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_function_l3576_357639


namespace NUMINAMATH_CALUDE_gcd_count_for_product_360_l3576_357669

theorem gcd_count_for_product_360 (a b : ℕ+) : 
  (Nat.gcd a b * Nat.lcm a b = 360) → 
  (∃! (s : Finset ℕ), ∀ x, x ∈ s ↔ ∃ (a' b' : ℕ+), 
    Nat.gcd a' b' = x ∧ Nat.gcd a' b' * Nat.lcm a' b' = 360 ∧ s.card = 12) := by
  sorry

end NUMINAMATH_CALUDE_gcd_count_for_product_360_l3576_357669


namespace NUMINAMATH_CALUDE_simplify_expression_l3576_357650

theorem simplify_expression (a : ℝ) : (1 + a) * (1 - a) + a * (a - 2) = 1 - 2 * a := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3576_357650


namespace NUMINAMATH_CALUDE_scientific_notation_equals_original_l3576_357635

/-- Scientific notation representation of 470,000,000 -/
def scientific_notation : ℝ := 4.7 * (10 ^ 8)

/-- The original number -/
def original_number : ℕ := 470000000

theorem scientific_notation_equals_original : 
  (scientific_notation : ℝ) = original_number := by sorry

end NUMINAMATH_CALUDE_scientific_notation_equals_original_l3576_357635


namespace NUMINAMATH_CALUDE_orange_groups_count_l3576_357643

/-- The number of groups of oranges in Philip's collection -/
def orange_groups (total_oranges : ℕ) (oranges_per_group : ℕ) : ℕ :=
  total_oranges / oranges_per_group

/-- Theorem stating that the number of orange groups is 16 -/
theorem orange_groups_count :
  orange_groups 384 24 = 16 := by
  sorry

end NUMINAMATH_CALUDE_orange_groups_count_l3576_357643


namespace NUMINAMATH_CALUDE_triangle_abc_theorem_l3576_357657

theorem triangle_abc_theorem (a b c : ℝ) (A B C : ℝ) :
  -- Given conditions
  (0 < A ∧ A < π) →
  (0 < B ∧ B < π) →
  (0 < C ∧ C < π) →
  (A + B + C = π) →
  (b * Real.cos A + a * Real.cos B = -2 * c * Real.cos C) →
  (a + b = 6) →
  (1/2 * a * b * Real.sin C = 2 * Real.sqrt 3) →
  -- Conclusions
  (C = 2 * π / 3 ∧ c = 2 * Real.sqrt 7) :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_theorem_l3576_357657


namespace NUMINAMATH_CALUDE_max_correct_is_38_l3576_357638

/-- Represents the scoring system and result of a multiple-choice test -/
structure TestScoring where
  total_questions : ℕ
  correct_points : ℤ
  blank_points : ℤ
  incorrect_points : ℤ
  total_score : ℤ

/-- Calculates the maximum number of correct answers possible given a TestScoring -/
def max_correct_answers (ts : TestScoring) : ℕ :=
  sorry

/-- Theorem stating that for the given test conditions, the maximum number of correct answers is 38 -/
theorem max_correct_is_38 : 
  let ts : TestScoring := {
    total_questions := 60,
    correct_points := 5,
    blank_points := 0,
    incorrect_points := -2,
    total_score := 150
  }
  max_correct_answers ts = 38 := by
  sorry

end NUMINAMATH_CALUDE_max_correct_is_38_l3576_357638


namespace NUMINAMATH_CALUDE_a_equals_3y_l3576_357623

theorem a_equals_3y (a b y : ℝ) 
  (h1 : a ≠ b) 
  (h2 : a^3 - b^3 = 27 * y^3) 
  (h3 : a - b = 3 * y) : 
  a = 3 * y := by
sorry

end NUMINAMATH_CALUDE_a_equals_3y_l3576_357623


namespace NUMINAMATH_CALUDE_fifth_grade_class_size_is_correct_l3576_357677

/-- Represents the number of students in each fifth grade class -/
def fifth_grade_class_size : ℕ := 27

/-- Represents the total number of third grade classes -/
def third_grade_classes : ℕ := 5

/-- Represents the number of students in each third grade class -/
def third_grade_class_size : ℕ := 30

/-- Represents the total number of fourth grade classes -/
def fourth_grade_classes : ℕ := 4

/-- Represents the number of students in each fourth grade class -/
def fourth_grade_class_size : ℕ := 28

/-- Represents the total number of fifth grade classes -/
def fifth_grade_classes : ℕ := 4

/-- Represents the cost of a hamburger in cents -/
def hamburger_cost : ℕ := 210

/-- Represents the cost of carrots in cents -/
def carrots_cost : ℕ := 50

/-- Represents the cost of a cookie in cents -/
def cookie_cost : ℕ := 20

/-- Represents the total cost of all students' lunches in cents -/
def total_lunch_cost : ℕ := 103600

theorem fifth_grade_class_size_is_correct : 
  fifth_grade_class_size * fifth_grade_classes * (hamburger_cost + carrots_cost + cookie_cost) + 
  third_grade_classes * third_grade_class_size * (hamburger_cost + carrots_cost + cookie_cost) + 
  fourth_grade_classes * fourth_grade_class_size * (hamburger_cost + carrots_cost + cookie_cost) = 
  total_lunch_cost :=
by sorry

end NUMINAMATH_CALUDE_fifth_grade_class_size_is_correct_l3576_357677


namespace NUMINAMATH_CALUDE_total_toys_l3576_357668

theorem total_toys (bill_toys : ℕ) (hash_toys : ℕ) : 
  bill_toys = 60 → 
  hash_toys = (bill_toys / 2) + 9 → 
  bill_toys + hash_toys = 99 :=
by sorry

end NUMINAMATH_CALUDE_total_toys_l3576_357668


namespace NUMINAMATH_CALUDE_cubic_bijective_l3576_357654

/-- The cubic function from reals to reals -/
def f (x : ℝ) : ℝ := x^3

/-- Theorem stating that the cubic function is bijective -/
theorem cubic_bijective : Function.Bijective f := by sorry

end NUMINAMATH_CALUDE_cubic_bijective_l3576_357654


namespace NUMINAMATH_CALUDE_least_four_digit_multiple_of_seven_l3576_357625

theorem least_four_digit_multiple_of_seven : ∃ n : ℕ,
  n = 1001 ∧
  7 ∣ n ∧
  1000 ≤ n ∧
  n < 10000 ∧
  ∀ m : ℕ, 7 ∣ m → 1000 ≤ m → m < 10000 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_least_four_digit_multiple_of_seven_l3576_357625


namespace NUMINAMATH_CALUDE_father_seven_times_son_age_l3576_357690

/-- 
Given a father who is currently 38 years old and a son who is currently 14 years old,
this theorem proves that 10 years ago, the father was seven times as old as his son.
-/
theorem father_seven_times_son_age (father_age : Nat) (son_age : Nat) (years_ago : Nat) : 
  father_age = 38 → son_age = 14 → 
  (father_age - years_ago) = 7 * (son_age - years_ago) → 
  years_ago = 10 := by
sorry

end NUMINAMATH_CALUDE_father_seven_times_son_age_l3576_357690


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_problem_l3576_357602

theorem arithmetic_geometric_mean_problem (x y : ℝ) 
  (h_arithmetic : (x + y) / 2 = 20)
  (h_geometric : Real.sqrt (x * y) = Real.sqrt 110) :
  x^2 + y^2 = 1380 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_problem_l3576_357602


namespace NUMINAMATH_CALUDE_sophomore_sample_count_l3576_357692

/-- Represents a stratified sampling scenario in a high school. -/
structure HighSchoolSampling where
  total_students : ℕ
  sophomore_count : ℕ
  sample_size : ℕ

/-- Calculates the number of sophomores in a stratified sample. -/
def sophomores_in_sample (h : HighSchoolSampling) : ℕ :=
  (h.sophomore_count * h.sample_size) / h.total_students

/-- Theorem: The number of sophomores in the sample is 93 given the specific scenario. -/
theorem sophomore_sample_count (h : HighSchoolSampling) 
  (h_total : h.total_students = 2800)
  (h_sophomores : h.sophomore_count = 930)
  (h_sample : h.sample_size = 280) : 
  sophomores_in_sample h = 93 := by
sorry

end NUMINAMATH_CALUDE_sophomore_sample_count_l3576_357692


namespace NUMINAMATH_CALUDE_homework_reduction_equation_l3576_357683

theorem homework_reduction_equation 
  (initial_duration : ℝ) 
  (final_duration : ℝ) 
  (x : ℝ) 
  (h1 : initial_duration = 90) 
  (h2 : final_duration = 60) 
  (h3 : 0 ≤ x ∧ x < 1) : 
  initial_duration * (1 - x)^2 = final_duration := by
sorry

end NUMINAMATH_CALUDE_homework_reduction_equation_l3576_357683


namespace NUMINAMATH_CALUDE_mike_five_dollar_bills_l3576_357646

theorem mike_five_dollar_bills (total_amount : ℕ) (bill_denomination : ℕ) 
  (h1 : total_amount = 45)
  (h2 : bill_denomination = 5) :
  total_amount / bill_denomination = 9 := by
sorry

end NUMINAMATH_CALUDE_mike_five_dollar_bills_l3576_357646


namespace NUMINAMATH_CALUDE_river_problem_solution_l3576_357671

/-- Represents the problem of a boat traveling along a river -/
structure RiverProblem where
  total_distance : ℝ
  total_time : ℝ
  upstream_distance : ℝ
  downstream_distance : ℝ
  hTotalDistance : total_distance = 10
  hTotalTime : total_time = 5
  hEqualTime : upstream_distance / downstream_distance = 2 / 3

/-- Solution to the river problem -/
structure RiverSolution where
  current_speed : ℝ
  upstream_time : ℝ
  downstream_time : ℝ

/-- Theorem stating the solution to the river problem -/
theorem river_problem_solution (p : RiverProblem) : 
  ∃ (s : RiverSolution), 
    s.current_speed = 5 / 12 ∧ 
    s.upstream_time = 3 ∧ 
    s.downstream_time = 2 := by
  sorry

end NUMINAMATH_CALUDE_river_problem_solution_l3576_357671


namespace NUMINAMATH_CALUDE_brand_preference_ratio_l3576_357694

def total_respondents : ℕ := 180
def brand_x_preference : ℕ := 150

theorem brand_preference_ratio :
  let brand_y_preference := total_respondents - brand_x_preference
  (brand_x_preference : ℚ) / brand_y_preference = 5 / 1 := by
  sorry

end NUMINAMATH_CALUDE_brand_preference_ratio_l3576_357694


namespace NUMINAMATH_CALUDE_two_digit_number_problem_l3576_357675

theorem two_digit_number_problem (a b : ℕ) : 
  b = 2 * a → 
  (10 * a + b) - (10 * b + a) = 36 → 
  (a + b) - (b - a) = 8 := by
sorry

end NUMINAMATH_CALUDE_two_digit_number_problem_l3576_357675


namespace NUMINAMATH_CALUDE_third_shape_symmetric_l3576_357664

-- Define a type for F-like shapes
inductive FLikeShape
| first
| second
| third
| fourth
| fifth

-- Define a function to check if a shape has reflection symmetry
def has_reflection_symmetry (shape : FLikeShape) : Prop :=
  match shape with
  | FLikeShape.third => True
  | _ => False

-- Theorem statement
theorem third_shape_symmetric :
  ∃ (shape : FLikeShape), has_reflection_symmetry shape ∧ shape = FLikeShape.third :=
by
  sorry

#check third_shape_symmetric

end NUMINAMATH_CALUDE_third_shape_symmetric_l3576_357664


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3576_357665

/-- Given a geometric sequence {a_n} with common ratio q and S_n as the sum of its first n terms,
    if 3S_3 = a_4 - 2 and 3S_2 = a_3 - 2, then q = 4 -/
theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ)
  (q : ℝ)
  (h1 : ∀ n, S n = (a 1) * (1 - q^n) / (1 - q))
  (h2 : ∀ n, a (n+1) = a n * q)
  (h3 : 3 * S 3 = a 4 - 2)
  (h4 : 3 * S 2 = a 3 - 2) :
  q = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3576_357665


namespace NUMINAMATH_CALUDE_fraction_equality_l3576_357604

theorem fraction_equality (a b : ℝ) (h1 : a ≠ b) (h2 : b ≠ 0) 
  (h3 : a/b + (a+5*b)/(b+5*a) = 2) : a/b = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3576_357604


namespace NUMINAMATH_CALUDE_wyatts_money_l3576_357682

theorem wyatts_money (bread_quantity : ℕ) (juice_quantity : ℕ) 
  (bread_price : ℕ) (juice_price : ℕ) (money_left : ℕ) :
  bread_quantity = 5 →
  juice_quantity = 4 →
  bread_price = 5 →
  juice_price = 2 →
  money_left = 41 →
  bread_quantity * bread_price + juice_quantity * juice_price + money_left = 74 :=
by sorry

end NUMINAMATH_CALUDE_wyatts_money_l3576_357682


namespace NUMINAMATH_CALUDE_simplify_fraction_l3576_357697

theorem simplify_fraction (a : ℝ) (h : a ≠ 1) : 
  (a^2 / (a - 1)) - ((1 - 2*a) / (1 - a)) = a - 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3576_357697


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3576_357670

theorem complex_equation_solution (a b : ℝ) (i : ℂ) 
  (h1 : i * i = -1) 
  (h2 : (a - 2 * i) * i = b - i) : 
  a + b * i = -1 + 2 * i :=
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3576_357670


namespace NUMINAMATH_CALUDE_more_girls_than_boys_l3576_357681

theorem more_girls_than_boys (total_students : ℕ) (boys : ℕ) 
  (h1 : total_students = 466) 
  (h2 : boys = 127) 
  (h3 : boys < total_students - boys) : 
  (total_students - boys) - boys = 212 := by
  sorry

end NUMINAMATH_CALUDE_more_girls_than_boys_l3576_357681


namespace NUMINAMATH_CALUDE_remainder_twelve_thousand_one_hundred_eleven_div_three_l3576_357621

theorem remainder_twelve_thousand_one_hundred_eleven_div_three : 
  12111 % 3 = 0 := by
sorry

end NUMINAMATH_CALUDE_remainder_twelve_thousand_one_hundred_eleven_div_three_l3576_357621


namespace NUMINAMATH_CALUDE_expression_evaluation_l3576_357663

/-- The sum of the first n positive integers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The product of powers of x from 1 to n -/
def prod_powers (x : ℕ) (n : ℕ) : ℕ := x ^ sum_first_n n

/-- The product of powers of x for multiples of 3 up to 3n -/
def prod_powers_mult3 (x : ℕ) (n : ℕ) : ℕ := x ^ (3 * sum_first_n n)

theorem expression_evaluation (x : ℕ) (hx : x = 3) :
  prod_powers x 20 / prod_powers_mult3 x 10 = x ^ 45 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3576_357663
