import Mathlib

namespace NUMINAMATH_CALUDE_subset_P_l3197_319725

def P : Set ℝ := {x | x > -1}

theorem subset_P : {0} ⊆ P := by sorry

end NUMINAMATH_CALUDE_subset_P_l3197_319725


namespace NUMINAMATH_CALUDE_variance_scaling_l3197_319758

-- Define a function to calculate the variance of a list of numbers
noncomputable def variance (data : List ℝ) : ℝ := sorry

-- Define our theorem
theorem variance_scaling (data : List ℝ) :
  variance data = 4 → variance (List.map (· * 2) data) = 16 := by
  sorry

end NUMINAMATH_CALUDE_variance_scaling_l3197_319758


namespace NUMINAMATH_CALUDE_multiplication_subtraction_equality_l3197_319797

theorem multiplication_subtraction_equality : 75 * 3030 - 35 * 3030 = 121200 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_subtraction_equality_l3197_319797


namespace NUMINAMATH_CALUDE_distinct_elements_in_union_of_progressions_l3197_319767

def arithmetic_progression (a₀ : ℕ) (d : ℕ) (n : ℕ) : Finset ℕ :=
  Finset.image (λ k => a₀ + k * d) (Finset.range n)

theorem distinct_elements_in_union_of_progressions :
  let progression1 := arithmetic_progression 2 3 2023
  let progression2 := arithmetic_progression 10 7 2023
  (progression1 ∪ progression2).card = 3756 := by
  sorry

end NUMINAMATH_CALUDE_distinct_elements_in_union_of_progressions_l3197_319767


namespace NUMINAMATH_CALUDE_simplify_algebraic_expression_l3197_319714

theorem simplify_algebraic_expression (a b : ℝ) : 5*a*b - 7*a*b + 3*a*b = a*b := by
  sorry

end NUMINAMATH_CALUDE_simplify_algebraic_expression_l3197_319714


namespace NUMINAMATH_CALUDE_complex_reciprocal_modulus_l3197_319766

theorem complex_reciprocal_modulus (m : ℤ) (z : ℂ) : 
  z = m - 3 + (m - 1) * I ∧ 
  (z.re < 0 ∧ z.im > 0) →
  Complex.abs (z⁻¹) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_reciprocal_modulus_l3197_319766


namespace NUMINAMATH_CALUDE_max_product_of_digits_l3197_319763

def is_digit (n : ℕ) : Prop := n ≤ 9

theorem max_product_of_digits (E F G H : ℕ) 
  (hE : is_digit E) (hF : is_digit F) (hG : is_digit G) (hH : is_digit H)
  (distinct : E ≠ F ∧ E ≠ G ∧ E ≠ H ∧ F ≠ G ∧ F ≠ H ∧ G ≠ H)
  (h_int : ∃ (k : ℕ), E * F = k * (G - H))
  (h_max : ∀ (E' F' G' H' : ℕ), 
    is_digit E' → is_digit F' → is_digit G' → is_digit H' →
    E' ≠ F' ∧ E' ≠ G' ∧ E' ≠ H' ∧ F' ≠ G' ∧ F' ≠ H' ∧ G' ≠ H' →
    (∃ (k' : ℕ), E' * F' = k' * (G' - H')) →
    E * F ≥ E' * F') :
  E * F = 72 :=
sorry

end NUMINAMATH_CALUDE_max_product_of_digits_l3197_319763


namespace NUMINAMATH_CALUDE_no_real_arithmetic_progression_l3197_319749

theorem no_real_arithmetic_progression : ¬ ∃ (a b : ℝ), 
  (b - a = a - 12) ∧ (ab - b = b - a) := by
  sorry

end NUMINAMATH_CALUDE_no_real_arithmetic_progression_l3197_319749


namespace NUMINAMATH_CALUDE_mitzel_spending_l3197_319751

/-- Proves that Mitzel spent $14, given the conditions of the problem -/
theorem mitzel_spending (allowance : ℝ) (spent_percentage : ℝ) (remaining : ℝ) : 
  spent_percentage = 0.35 →
  remaining = 26 →
  (1 - spent_percentage) * allowance = remaining →
  spent_percentage * allowance = 14 := by
  sorry

end NUMINAMATH_CALUDE_mitzel_spending_l3197_319751


namespace NUMINAMATH_CALUDE_rectangle_b_product_l3197_319779

theorem rectangle_b_product : ∀ b₁ b₂ : ℝ,
  (∃ (rect : Set (ℝ × ℝ)), 
    rect = {(x, y) | 3 ≤ y ∧ y ≤ 8 ∧ ((x = 2 ∧ b₁ ≤ x) ∨ (x = b₁ ∧ x ≤ 2)) ∧
            ((x = 2 ∧ x ≤ b₂) ∨ (x = b₂ ∧ 2 ≤ x))} ∧
    (∀ (p q : ℝ × ℝ), p ∈ rect ∧ q ∈ rect → 
      (p.1 = q.1 ∨ p.2 = q.2) ∧ 
      (p.1 ≠ q.1 ∨ p.2 ≠ q.2))) →
  b₁ * b₂ = -21 :=
by sorry


end NUMINAMATH_CALUDE_rectangle_b_product_l3197_319779


namespace NUMINAMATH_CALUDE_area_of_triangle_fpg_l3197_319786

/-- Given a trapezoid EFGH with bases EF and GH, and point P at the intersection of diagonals,
    this theorem states that the area of triangle FPG is 28.125 square units. -/
theorem area_of_triangle_fpg (EF GH : ℝ) (area_EFGH : ℝ) :
  EF = 15 →
  GH = 25 →
  area_EFGH = 200 →
  ∃ (area_FPG : ℝ), area_FPG = 28.125 := by
  sorry

end NUMINAMATH_CALUDE_area_of_triangle_fpg_l3197_319786


namespace NUMINAMATH_CALUDE_line_parameterization_l3197_319701

/-- Given a line y = 2x - 10 parameterized by (x,y) = (g(t), 10t - 4), prove that g(t) = 5t + 3 -/
theorem line_parameterization (g : ℝ → ℝ) : 
  (∀ t, 2 * g t - 10 = 10 * t - 4) → 
  (∀ t, g t = 5 * t + 3) := by
sorry

end NUMINAMATH_CALUDE_line_parameterization_l3197_319701


namespace NUMINAMATH_CALUDE_kaylee_biscuit_sales_l3197_319759

/-- The number of boxes Kaylee needs to sell -/
def total_boxes : ℕ := 33

/-- The number of lemon biscuit boxes sold -/
def lemon_boxes : ℕ := 12

/-- The number of chocolate biscuit boxes sold -/
def chocolate_boxes : ℕ := 5

/-- The number of oatmeal biscuit boxes sold -/
def oatmeal_boxes : ℕ := 4

/-- The number of additional boxes Kaylee needs to sell -/
def additional_boxes : ℕ := total_boxes - (lemon_boxes + chocolate_boxes + oatmeal_boxes)

theorem kaylee_biscuit_sales : additional_boxes = 12 := by
  sorry

end NUMINAMATH_CALUDE_kaylee_biscuit_sales_l3197_319759


namespace NUMINAMATH_CALUDE_sum_and_difference_l3197_319706

theorem sum_and_difference : 2345 + 3452 + 4523 + 5234 - 1234 = 14320 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_difference_l3197_319706


namespace NUMINAMATH_CALUDE_rectangular_prism_sum_l3197_319762

/-- A rectangular prism is a three-dimensional shape with 6 faces, 12 edges, and 8 vertices. -/
structure RectangularPrism where
  faces : Nat
  edges : Nat
  vertices : Nat
  faces_eq : faces = 6
  edges_eq : edges = 12
  vertices_eq : vertices = 8

/-- The sum of faces, edges, and vertices of a rectangular prism is 26. -/
theorem rectangular_prism_sum (rp : RectangularPrism) : 
  rp.faces + rp.edges + rp.vertices = 26 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_sum_l3197_319762


namespace NUMINAMATH_CALUDE_tom_five_times_tim_l3197_319765

/-- Represents the typing speeds of Tim and Tom -/
structure TypingSpeed where
  tim : ℝ
  tom : ℝ

/-- The combined typing speed of Tim and Tom -/
def combinedSpeed (s : TypingSpeed) : ℝ := s.tim + s.tom

/-- Tom's increased typing speed (30% faster) -/
def tomIncreasedSpeed (s : TypingSpeed) : ℝ := s.tom * 1.3

/-- The combined speed when Tom types 30% faster -/
def combinedIncreasedSpeed (s : TypingSpeed) : ℝ := s.tim + tomIncreasedSpeed s

/-- The theorem stating that Tom's normal typing speed is 5 times Tim's -/
theorem tom_five_times_tim (s : TypingSpeed) 
  (h1 : combinedSpeed s = 12)
  (h2 : combinedIncreasedSpeed s = 15) : 
  s.tom = 5 * s.tim := by
  sorry

#check tom_five_times_tim

end NUMINAMATH_CALUDE_tom_five_times_tim_l3197_319765


namespace NUMINAMATH_CALUDE_sum_smallest_largest_consecutive_odds_l3197_319773

/-- Given an even number of consecutive odd integers with arithmetic mean y + 1,
    the sum of the smallest and largest integers is 2y. -/
theorem sum_smallest_largest_consecutive_odds (y : ℝ) (n : ℕ) (h : n > 0) :
  let a := y - 2 * n + 2
  let sequence := fun i => a + 2 * i
  let mean := (sequence 0 + sequence (2 * n - 1)) / 2
  (mean = y + 1) → (sequence 0 + sequence (2 * n - 1) = 2 * y) :=
by sorry

end NUMINAMATH_CALUDE_sum_smallest_largest_consecutive_odds_l3197_319773


namespace NUMINAMATH_CALUDE_multiplication_equality_l3197_319752

theorem multiplication_equality : 62519 * 9999 = 625127481 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_equality_l3197_319752


namespace NUMINAMATH_CALUDE_distance_traveled_l3197_319731

-- Define the speed in miles per hour
def speed : ℝ := 16

-- Define the time in hours
def time : ℝ := 5

-- Theorem to prove the distance traveled
theorem distance_traveled : speed * time = 80 := by
  sorry

end NUMINAMATH_CALUDE_distance_traveled_l3197_319731


namespace NUMINAMATH_CALUDE_product_of_integers_with_given_lcm_and_gcd_l3197_319747

theorem product_of_integers_with_given_lcm_and_gcd :
  ∀ a b : ℕ+, 
  (Nat.lcm a b = 60) → 
  (Nat.gcd a b = 12) → 
  (a * b = 720) :=
by
  sorry

end NUMINAMATH_CALUDE_product_of_integers_with_given_lcm_and_gcd_l3197_319747


namespace NUMINAMATH_CALUDE_he_more_apples_l3197_319756

/-- The number of apples Adam and Jackie have together -/
def total_adam_jackie : ℕ := 12

/-- The number of apples Adam has more than Jackie -/
def adam_more_than_jackie : ℕ := 8

/-- The number of apples He has -/
def he_apples : ℕ := 21

/-- The number of apples Jackie has -/
def jackie_apples : ℕ := 2

/-- The number of apples Adam has -/
def adam_apples : ℕ := jackie_apples + adam_more_than_jackie

theorem he_more_apples : he_apples - total_adam_jackie = 9 := by
  sorry

end NUMINAMATH_CALUDE_he_more_apples_l3197_319756


namespace NUMINAMATH_CALUDE_gcd_10011_15015_l3197_319783

theorem gcd_10011_15015 : Nat.gcd 10011 15015 = 1001 := by
  sorry

end NUMINAMATH_CALUDE_gcd_10011_15015_l3197_319783


namespace NUMINAMATH_CALUDE_g_of_8_eq_neg_46_l3197_319719

/-- A function g : ℝ → ℝ satisfying the given functional equation for all real x and y -/
def g_equation (g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, g x + g (3*x + y) + 7*x*y = g (4*x - 2*y) + 3*x^2 + 2

/-- Theorem stating that if g satisfies the functional equation, then g(8) = -46 -/
theorem g_of_8_eq_neg_46 (g : ℝ → ℝ) (h : g_equation g) : g 8 = -46 := by
  sorry

end NUMINAMATH_CALUDE_g_of_8_eq_neg_46_l3197_319719


namespace NUMINAMATH_CALUDE_unique_solution_l3197_319728

/-- A function y is a solution to the differential equation y' - y = cos x - sin x
    and is bounded as x approaches positive infinity -/
def IsSolution (y : ℝ → ℝ) : Prop :=
  (∀ x, (deriv y x) - y x = Real.cos x - Real.sin x) ∧
  (∃ M, ∀ x, x ≥ 0 → |y x| ≤ M)

/-- The unique solution to the differential equation y' - y = cos x - sin x
    that is bounded as x approaches positive infinity is y = - cos x -/
theorem unique_solution :
  ∃! y, IsSolution y ∧ (∀ x, y x = - Real.cos x) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l3197_319728


namespace NUMINAMATH_CALUDE_optimal_purchase_plan_l3197_319780

/-- Represents the purchase and selling prices of keychains --/
structure KeychainPrices where
  purchase_a : ℕ
  purchase_b : ℕ
  selling_a : ℕ
  selling_b : ℕ

/-- Represents the purchase plan for keychains --/
structure PurchasePlan where
  quantity_a : ℕ
  quantity_b : ℕ

/-- Calculates the total purchase cost for a given plan --/
def total_purchase_cost (prices : KeychainPrices) (plan : PurchasePlan) : ℕ :=
  prices.purchase_a * plan.quantity_a + prices.purchase_b * plan.quantity_b

/-- Calculates the total profit for a given plan --/
def total_profit (prices : KeychainPrices) (plan : PurchasePlan) : ℕ :=
  (prices.selling_a - prices.purchase_a) * plan.quantity_a +
  (prices.selling_b - prices.purchase_b) * plan.quantity_b

/-- Theorem: The optimal purchase plan maximizes profit --/
theorem optimal_purchase_plan (prices : KeychainPrices)
  (h_prices : prices.purchase_a = 30 ∧ prices.purchase_b = 25 ∧
              prices.selling_a = 45 ∧ prices.selling_b = 37) :
  ∃ (plan : PurchasePlan),
    plan.quantity_a + plan.quantity_b = 80 ∧
    total_purchase_cost prices plan ≤ 2200 ∧
    total_profit prices plan = 1080 ∧
    ∀ (other_plan : PurchasePlan),
      other_plan.quantity_a + other_plan.quantity_b = 80 →
      total_purchase_cost prices other_plan ≤ 2200 →
      total_profit prices other_plan ≤ total_profit prices plan :=
sorry

end NUMINAMATH_CALUDE_optimal_purchase_plan_l3197_319780


namespace NUMINAMATH_CALUDE_next_multiple_year_l3197_319771

theorem next_multiple_year : ∀ n : ℕ, 
  n > 2016 ∧ 
  n % 6 = 0 ∧ 
  n % 8 = 0 ∧ 
  n % 9 = 0 → 
  n ≥ 2088 :=
by
  sorry

end NUMINAMATH_CALUDE_next_multiple_year_l3197_319771


namespace NUMINAMATH_CALUDE_public_area_diameter_l3197_319764

/-- Represents the diameter of the outer boundary of a circular public area -/
def outer_boundary_diameter (play_area_diameter : ℝ) (garden_width : ℝ) (track_width : ℝ) : ℝ :=
  play_area_diameter + 2 * (garden_width + track_width)

/-- Theorem stating the diameter of the outer boundary of the running track -/
theorem public_area_diameter : 
  outer_boundary_diameter 14 6 4 = 34 := by
  sorry

end NUMINAMATH_CALUDE_public_area_diameter_l3197_319764


namespace NUMINAMATH_CALUDE_wall_bricks_count_l3197_319774

/-- Represents the number of bricks in the wall -/
def total_bricks : ℕ := 192

/-- Represents Beth's individual rate in bricks per hour -/
def beth_rate : ℚ := total_bricks / 8

/-- Represents Ben's individual rate in bricks per hour -/
def ben_rate : ℚ := total_bricks / 12

/-- Represents the reduction in combined output due to chatting, in bricks per hour -/
def chat_reduction : ℕ := 8

/-- Represents the time taken to complete the wall when working together, in hours -/
def time_together : ℕ := 6

theorem wall_bricks_count :
  (beth_rate + ben_rate - chat_reduction) * time_together = total_bricks := by
  sorry

#check wall_bricks_count

end NUMINAMATH_CALUDE_wall_bricks_count_l3197_319774


namespace NUMINAMATH_CALUDE_least_product_of_primes_above_50_l3197_319791

theorem least_product_of_primes_above_50 (p q : ℕ) : 
  p.Prime → q.Prime → p > 50 → q > 50 → p ≠ q → 
  ∀ r s : ℕ, r.Prime → s.Prime → r > 50 → s > 50 → r ≠ s → 
  p * q ≤ r * s :=
sorry

end NUMINAMATH_CALUDE_least_product_of_primes_above_50_l3197_319791


namespace NUMINAMATH_CALUDE_special_triangle_ratio_l3197_319729

/-- A scalene triangle with two medians equal to two altitudes -/
structure SpecialTriangle where
  -- The triangle is scalene
  is_scalene : Bool
  -- Two medians are equal to two altitudes
  two_medians_equal_altitudes : Bool

/-- The ratio of the third median to the third altitude -/
def third_median_altitude_ratio (t : SpecialTriangle) : ℚ :=
  7 / 2

/-- Theorem stating the ratio of the third median to the third altitude -/
theorem special_triangle_ratio (t : SpecialTriangle) 
  (h1 : t.is_scalene = true) 
  (h2 : t.two_medians_equal_altitudes = true) : 
  third_median_altitude_ratio t = 7 / 2 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_ratio_l3197_319729


namespace NUMINAMATH_CALUDE_solution_equation1_solution_equation2_l3197_319716

-- Define the first equation
def equation1 (x : ℝ) : Prop := 3 * x - 5 = 6 * x - 8

-- Define the second equation
def equation2 (x : ℝ) : Prop := (x + 1) / 2 - (2 * x - 1) / 3 = 1

-- Theorem for the first equation
theorem solution_equation1 : ∃ x : ℝ, equation1 x ∧ x = 1 := by sorry

-- Theorem for the second equation
theorem solution_equation2 : ∃ x : ℝ, equation2 x ∧ x = -1 := by sorry

end NUMINAMATH_CALUDE_solution_equation1_solution_equation2_l3197_319716


namespace NUMINAMATH_CALUDE_tournament_teams_count_l3197_319715

/-- Calculates the number of matches in a round-robin tournament for n teams -/
def matchesInGroup (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Represents a valid configuration of team groups -/
structure GroupConfig where
  g1 : ℕ
  g2 : ℕ
  g3 : ℕ
  g4 : ℕ
  h1 : g1 ≥ 2
  h2 : g2 ≥ 2
  h3 : g3 ≥ 2
  h4 : g4 ≥ 2
  h5 : matchesInGroup g1 + matchesInGroup g2 + matchesInGroup g3 + matchesInGroup g4 = 66

/-- The set of all possible total number of teams -/
def possibleTotalTeams : Set ℕ := {21, 22, 23, 24, 25}

theorem tournament_teams_count :
  ∀ (config : GroupConfig), (config.g1 + config.g2 + config.g3 + config.g4) ∈ possibleTotalTeams :=
by sorry

end NUMINAMATH_CALUDE_tournament_teams_count_l3197_319715


namespace NUMINAMATH_CALUDE_ant_path_theorem_l3197_319717

/-- Represents the three concentric square paths -/
structure SquarePaths where
  a : ℝ  -- Side length of the smallest square
  b : ℝ  -- Side length of the middle square
  c : ℝ  -- Side length of the largest square
  h1 : 0 < a
  h2 : a < b
  h3 : b < c

/-- Represents the positions of the three ants -/
structure AntPositions (p : SquarePaths) where
  mu : ℝ  -- Distance traveled by Mu
  ra : ℝ  -- Distance traveled by Ra
  vey : ℝ  -- Distance traveled by Vey
  h1 : mu = p.c  -- Mu reaches the lower-right corner of the largest square
  h2 : ra = p.c - 1  -- Ra's position on the right side of the middle square
  h3 : vey = 2 * (p.c - p.b + 1)  -- Vey's position on the right side of the smallest square

/-- The main theorem stating the conditions and the result -/
theorem ant_path_theorem (p : SquarePaths) (pos : AntPositions p) :
  (p.c - p.b = p.b - p.a) ∧ (p.b - p.a = 2) →
  p.a = 4 ∧ p.b = 6 ∧ p.c = 8 := by
  sorry

end NUMINAMATH_CALUDE_ant_path_theorem_l3197_319717


namespace NUMINAMATH_CALUDE_sara_purse_value_l3197_319712

/-- Represents the value of a coin in cents -/
def coin_value (coin : String) : ℕ :=
  match coin with
  | "penny" => 1
  | "nickel" => 5
  | "dime" => 10
  | "quarter" => 25
  | _ => 0

/-- Calculates the total value of coins in cents -/
def total_value (pennies nickels dimes quarters : ℕ) : ℕ :=
  pennies * coin_value "penny" +
  nickels * coin_value "nickel" +
  dimes * coin_value "dime" +
  quarters * coin_value "quarter"

/-- Converts a number of cents to a percentage of a dollar -/
def cents_to_percentage (cents : ℕ) : ℚ :=
  (cents : ℚ) / 100

theorem sara_purse_value :
  cents_to_percentage (total_value 3 2 1 2) = 73 / 100 := by
  sorry

end NUMINAMATH_CALUDE_sara_purse_value_l3197_319712


namespace NUMINAMATH_CALUDE_toms_initial_money_l3197_319785

theorem toms_initial_money (current_money : ℕ) (weekend_earnings : ℕ) (initial_money : ℕ) :
  current_money = 86 →
  weekend_earnings = 12 →
  current_money = initial_money + weekend_earnings →
  initial_money = 74 :=
by sorry

end NUMINAMATH_CALUDE_toms_initial_money_l3197_319785


namespace NUMINAMATH_CALUDE_building_height_l3197_319704

/-- The height of a building given shadow lengths -/
theorem building_height (shadow_building : ℝ) (height_post : ℝ) (shadow_post : ℝ)
  (h_shadow_building : shadow_building = 120)
  (h_height_post : height_post = 15)
  (h_shadow_post : shadow_post = 25) :
  (height_post / shadow_post) * shadow_building = 72 := by
  sorry

end NUMINAMATH_CALUDE_building_height_l3197_319704


namespace NUMINAMATH_CALUDE_largest_change_first_digit_l3197_319796

def original_number : ℚ := 0.05123

def change_digit (n : ℚ) (pos : ℕ) (new_digit : ℕ) : ℚ :=
  sorry

theorem largest_change_first_digit :
  ∀ pos : ℕ, pos > 0 → pos ≤ 5 →
    change_digit original_number 1 8 > change_digit original_number pos 8 :=
  sorry

end NUMINAMATH_CALUDE_largest_change_first_digit_l3197_319796


namespace NUMINAMATH_CALUDE_max_chord_line_l3197_319738

/-- A circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in the 2D plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The circle C: x^2 + y^2 + 4x + 3 = 0 -/
def C : Circle := { center := (-2, 0), radius := 1 }

/-- The point through which line l passes -/
def P : ℝ × ℝ := (2, 3)

/-- Function to check if a line passes through a point -/
def passes_through (l : Line) (p : ℝ × ℝ) : Prop :=
  l.a * p.1 + l.b * p.2 + l.c = 0

/-- Function to check if a line intersects a circle at two points -/
def intersects_circle (l : Line) (c : Circle) : Prop :=
  ∃ (p q : ℝ × ℝ), p ≠ q ∧ 
    passes_through l p ∧ passes_through l q ∧
    (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2 ∧
    (q.1 - c.center.1)^2 + (q.2 - c.center.2)^2 = c.radius^2

/-- Function to check if the chord formed by the intersection of a line and circle is maximized -/
def maximizes_chord (l : Line) (c : Circle) : Prop :=
  passes_through l c.center

/-- The theorem to be proved -/
theorem max_chord_line : 
  ∃ (l : Line), 
    passes_through l P ∧ 
    intersects_circle l C ∧ 
    maximizes_chord l C ∧ 
    l = { a := 3, b := -4, c := 6 } := by sorry

end NUMINAMATH_CALUDE_max_chord_line_l3197_319738


namespace NUMINAMATH_CALUDE_disjunction_true_l3197_319708

open Real

theorem disjunction_true : 
  (¬(∀ α : ℝ, sin (π - α) ≠ -sin α)) ∨ (∃ x : ℝ, x ≥ 0 ∧ sin x > x) := by
  sorry

end NUMINAMATH_CALUDE_disjunction_true_l3197_319708


namespace NUMINAMATH_CALUDE_square_park_area_l3197_319710

/-- The area of a square park with a side length of 30 meters is 900 square meters. -/
theorem square_park_area : 
  ∀ (park_side_length : ℝ), 
  park_side_length = 30 → 
  park_side_length * park_side_length = 900 :=
by
  sorry

end NUMINAMATH_CALUDE_square_park_area_l3197_319710


namespace NUMINAMATH_CALUDE_max_factorable_n_is_largest_l3197_319775

/-- A polynomial of the form 3x^2 + nx + 72 can be factored as (3x + A)(x + B) where A and B are integers -/
def is_factorable (n : ℤ) : Prop :=
  ∃ A B : ℤ, 3 * B + A = n ∧ A * B = 72

/-- The maximum value of n for which 3x^2 + nx + 72 can be factored as the product of two linear factors with integer coefficients -/
def max_factorable_n : ℤ := 217

/-- Theorem stating that max_factorable_n is the largest value of n for which the polynomial is factorable -/
theorem max_factorable_n_is_largest :
  is_factorable max_factorable_n ∧
  ∀ m : ℤ, m > max_factorable_n → ¬is_factorable m :=
by sorry

end NUMINAMATH_CALUDE_max_factorable_n_is_largest_l3197_319775


namespace NUMINAMATH_CALUDE_jellybean_problem_l3197_319772

/-- The number of jellybeans remaining after eating 25% --/
def eat_jellybeans (n : ℝ) : ℝ := 0.75 * n

/-- The number of jellybeans Jenny has initially --/
def initial_jellybeans : ℝ := 80

/-- The number of jellybeans added after the first day --/
def added_jellybeans : ℝ := 20

/-- The number of jellybeans remaining after three days --/
def remaining_jellybeans : ℝ := 
  eat_jellybeans (eat_jellybeans (eat_jellybeans initial_jellybeans + added_jellybeans))

theorem jellybean_problem : remaining_jellybeans = 45 := by sorry

end NUMINAMATH_CALUDE_jellybean_problem_l3197_319772


namespace NUMINAMATH_CALUDE_triangle_properties_l3197_319777

-- Define the triangle ABC
def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (3, 4)

-- Define the equations of median and altitude
def median_eq (x y : ℝ) : Prop := 2 * x + y - 7 = 0
def altitude_eq (x y : ℝ) : Prop := 2 * x - y - 2 = 0

-- Define vertex C
def C : ℝ × ℝ := (3, 1)

-- Define the area of the triangle
def triangle_area : ℝ := 3

-- Theorem statement
theorem triangle_properties :
  median_eq (C.1) (C.2) ∧ 
  altitude_eq (C.1) (C.2) →
  C = (3, 1) ∧ 
  triangle_area = 3 := by sorry

end NUMINAMATH_CALUDE_triangle_properties_l3197_319777


namespace NUMINAMATH_CALUDE_train_length_l3197_319792

/-- Proves that a train traveling at 40 km/hr crossing a pole in 9 seconds has a length of 100 meters. -/
theorem train_length (speed : ℝ) (time : ℝ) (length : ℝ) : 
  speed = 40 → -- speed in km/hr
  time = 9 → -- time in seconds
  length = speed * (1000 / 3600) * time → -- convert km/hr to m/s and multiply by time
  length = 100 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l3197_319792


namespace NUMINAMATH_CALUDE_sherries_banana_bread_l3197_319753

theorem sherries_banana_bread (recipe_loaves : ℕ) (recipe_bananas : ℕ) (total_bananas : ℕ) 
  (h1 : recipe_loaves = 3)
  (h2 : recipe_bananas = 1)
  (h3 : total_bananas = 33) :
  (total_bananas * recipe_loaves) / recipe_bananas = 99 :=
by sorry

end NUMINAMATH_CALUDE_sherries_banana_bread_l3197_319753


namespace NUMINAMATH_CALUDE_strikers_count_l3197_319746

/-- A soccer team composition -/
structure SoccerTeam where
  goalies : Nat
  defenders : Nat
  midfielders : Nat
  strikers : Nat

/-- The total number of players in a soccer team -/
def total_players (team : SoccerTeam) : Nat :=
  team.goalies + team.defenders + team.midfielders + team.strikers

/-- Theorem: Given the conditions, the number of strikers is 7 -/
theorem strikers_count (team : SoccerTeam)
  (h1 : team.goalies = 3)
  (h2 : team.defenders = 10)
  (h3 : team.midfielders = 2 * team.defenders)
  (h4 : total_players team = 40) :
  team.strikers = 7 := by
  sorry

end NUMINAMATH_CALUDE_strikers_count_l3197_319746


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3197_319721

/-- A geometric sequence {a_n} with common ratio q -/
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  geometric_sequence a q →
  (a 1 + a 2 = 30) →
  (a 3 + a 4 = 60) →
  (a 7 + a 8 = (a 1 + a 2) * q^6) :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3197_319721


namespace NUMINAMATH_CALUDE_trapezoid_length_in_divided_square_l3197_319733

/-- Given a square with side length 2 meters, divided into two congruent trapezoids and a quadrilateral,
    where the trapezoids have bases on two sides of the square and their other bases meet at the square's center,
    and all three shapes have equal areas, the length of the longer parallel side of each trapezoid is 5/3 meters. -/
theorem trapezoid_length_in_divided_square :
  let square_side : ℝ := 2
  let total_area : ℝ := square_side ^ 2
  let shape_area : ℝ := total_area / 3
  let shorter_base : ℝ := square_side / 2
  ∃ (longer_base : ℝ),
    longer_base = 5 / 3 ∧
    shape_area = (longer_base + shorter_base) * square_side / 4 :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_length_in_divided_square_l3197_319733


namespace NUMINAMATH_CALUDE_man_ownership_proof_l3197_319711

/-- The fraction of the business owned by the man -/
def man_ownership : ℚ := 2/3

/-- The value of the entire business in rupees -/
def business_value : ℕ := 60000

/-- The amount received from selling 3/4 of the man's shares in rupees -/
def sale_amount : ℕ := 30000

/-- The fraction of the man's shares that were sold -/
def sold_fraction : ℚ := 3/4

theorem man_ownership_proof :
  man_ownership * sold_fraction * business_value = sale_amount :=
sorry

end NUMINAMATH_CALUDE_man_ownership_proof_l3197_319711


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l3197_319799

theorem sum_of_reciprocals (a b : ℝ) 
  (ha : a^2 + 2*a = 2) 
  (hb : b^2 + 2*b = 2) : 
  (1/a + 1/b = 1) ∨ 
  (1/a + 1/b = Real.sqrt 3 + 1) ∨ 
  (1/a + 1/b = -Real.sqrt 3 + 1) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l3197_319799


namespace NUMINAMATH_CALUDE_cone_surface_area_minimization_l3197_319741

/-- For a cone with fixed volume, when the total surface area is minimized,
    there exists a relationship between the height and radius of the cone. -/
theorem cone_surface_area_minimization (V : ℝ) (V_pos : V > 0) :
  ∃ (R H : ℝ) (R_pos : R > 0) (H_pos : H > 0),
    (1/3 : ℝ) * Real.pi * R^2 * H = V ∧
    (∀ (r h : ℝ) (r_pos : r > 0) (h_pos : h > 0),
      (1/3 : ℝ) * Real.pi * r^2 * h = V →
      Real.pi * R^2 + Real.pi * R * Real.sqrt (R^2 + H^2) ≤
      Real.pi * r^2 + Real.pi * r * Real.sqrt (r^2 + h^2)) →
    ∃ (k : ℝ), H = k * R := by
  sorry

end NUMINAMATH_CALUDE_cone_surface_area_minimization_l3197_319741


namespace NUMINAMATH_CALUDE_shaded_area_of_square_grid_l3197_319727

/-- The area of a square composed of 25 congruent smaller squares, 
    where the diagonal of the larger square is 10 cm, is 50 square cm. -/
theorem shaded_area_of_square_grid (d : ℝ) (n : ℕ) : 
  d = 10 → n = 25 → (d^2 / 2) * (n / n^(1/2) : ℝ)^2 = 50 := by sorry

end NUMINAMATH_CALUDE_shaded_area_of_square_grid_l3197_319727


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_l3197_319768

theorem sum_of_x_and_y (x y : ℤ) (h1 : x - y = 36) (h2 : x = 28) : x + y = 20 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_l3197_319768


namespace NUMINAMATH_CALUDE_a_necessary_for_c_l3197_319754

theorem a_necessary_for_c (A B C : Prop) 
  (h1 : ¬A ↔ ¬B) (h2 : ¬B → ¬C) : C → A := by
  sorry

end NUMINAMATH_CALUDE_a_necessary_for_c_l3197_319754


namespace NUMINAMATH_CALUDE_smallest_multiple_l3197_319744

theorem smallest_multiple (n : ℕ) : n = 187 ↔ 
  n > 0 ∧ 
  17 ∣ n ∧ 
  n % 53 = 7 ∧ 
  ∀ m : ℕ, m > 0 → 17 ∣ m → m % 53 = 7 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_l3197_319744


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l3197_319734

theorem geometric_sequence_problem (a : ℕ → ℝ) (q : ℝ) :
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r) →  -- geometric sequence condition
  (abs q > 1) →
  (a 2 + a 7 = 2) →
  (a 4 * a 5 = -15) →
  (a 12 = -25/3) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l3197_319734


namespace NUMINAMATH_CALUDE_f_inequality_range_l3197_319770

noncomputable def f (x : ℝ) : ℝ :=
  if x < -1/2 then 2*x + 1
  else if x < 3/2 then 2 - (3-2*x)
  else 2*x + 1 + (2*x-3)

theorem f_inequality_range (a : ℝ) :
  (∃ x, f x < 1) →
  (∀ x, f x ≤ |a|) →
  |a| ≥ 4 := by sorry

end NUMINAMATH_CALUDE_f_inequality_range_l3197_319770


namespace NUMINAMATH_CALUDE_largest_common_term_l3197_319723

def arithmetic_sequence (a₀ d : ℤ) (n : ℕ) : ℤ := a₀ + d * (n - 1)

theorem largest_common_term (x : ℤ) :
  (∃ n m : ℕ, x = arithmetic_sequence 5 8 n ∧ x = arithmetic_sequence 3 9 m) ∧
  1 ≤ x ∧ x ≤ 150 ∧
  ∀ y : ℤ, (∃ n m : ℕ, y = arithmetic_sequence 5 8 n ∧ y = arithmetic_sequence 3 9 m) ∧
           1 ≤ y ∧ y ≤ 150 →
           y ≤ x →
  x = 93 := by sorry

end NUMINAMATH_CALUDE_largest_common_term_l3197_319723


namespace NUMINAMATH_CALUDE_amc10_paths_l3197_319720

/-- Represents the number of possible moves from each position -/
def num_moves : ℕ := 8

/-- Represents the length of the string "AMC10" -/
def word_length : ℕ := 5

/-- Calculates the number of paths to spell "AMC10" -/
def num_paths : ℕ := num_moves ^ (word_length - 1)

/-- Proves that the number of paths to spell "AMC10" is 4096 -/
theorem amc10_paths : num_paths = 4096 := by
  sorry

end NUMINAMATH_CALUDE_amc10_paths_l3197_319720


namespace NUMINAMATH_CALUDE_wheel_speed_l3197_319757

/-- The speed of the wheel in miles per hour -/
def r : ℝ := sorry

/-- The circumference of the wheel in feet -/
def circumference : ℝ := 11

/-- The time for one rotation in hours -/
def t : ℝ := sorry

/-- Conversion factor from feet to miles -/
def feet_per_mile : ℝ := 5280

/-- Conversion factor from hours to seconds -/
def seconds_per_hour : ℝ := 3600

/-- The relationship between speed, time, and distance -/
axiom speed_time_distance : r * t = circumference / feet_per_mile

/-- The relationship when time is decreased and speed is increased -/
axiom increased_speed_decreased_time : 
  (r + 5) * (t - 1 / (4 * seconds_per_hour)) = circumference / feet_per_mile

theorem wheel_speed : r = 10 := by sorry

end NUMINAMATH_CALUDE_wheel_speed_l3197_319757


namespace NUMINAMATH_CALUDE_sara_is_45_inches_tall_l3197_319745

-- Define the heights as natural numbers
def roy_height : ℕ := 36
def joe_height : ℕ := roy_height + 3
def sara_height : ℕ := joe_height + 6

-- Theorem statement
theorem sara_is_45_inches_tall : sara_height = 45 := by
  sorry

end NUMINAMATH_CALUDE_sara_is_45_inches_tall_l3197_319745


namespace NUMINAMATH_CALUDE_problem_solution_l3197_319722

def A (a : ℕ) : Set ℕ := {2, 5, a + 1}
def B (a : ℕ) : Set ℕ := {1, 3, a}
def U : Set ℕ := {x | x ≤ 6}

theorem problem_solution (a : ℕ) 
  (h1 : A a ∩ B a = {2, 3}) :
  (a = 2) ∧ 
  (A a ∪ B a = {1, 2, 3, 5}) ∧ 
  ((Uᶜ ∩ (A a)ᶜ) ∩ (Uᶜ ∩ (B a)ᶜ) = {0, 4, 6}) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3197_319722


namespace NUMINAMATH_CALUDE_regular_polygon_perimeter_l3197_319761

/-- A regular polygon with side length 7 units and exterior angle 90 degrees has a perimeter of 28 units. -/
theorem regular_polygon_perimeter (s : ℝ) (e : ℝ) : 
  s = 7 → e = 90 → (360 / e : ℝ) * s = 28 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_perimeter_l3197_319761


namespace NUMINAMATH_CALUDE_binary_multiplication_theorem_l3197_319743

/-- Converts a list of binary digits to a natural number -/
def binaryToNat (digits : List Bool) : Nat :=
  digits.foldl (fun acc d => 2 * acc + if d then 1 else 0) 0

/-- Converts a natural number to a list of binary digits -/
def natToBinary (n : Nat) : List Bool :=
  if n = 0 then [false] else
  let rec toBinary (m : Nat) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: toBinary (m / 2)
  toBinary n

theorem binary_multiplication_theorem :
  let a := [true, false, true, true]  -- 1101₂
  let b := [false, true, true]        -- 110₂
  let expected := [false, true, true, true, true, false, true]  -- 1011110₂
  binaryToNat a * binaryToNat b = binaryToNat expected := by
  sorry

end NUMINAMATH_CALUDE_binary_multiplication_theorem_l3197_319743


namespace NUMINAMATH_CALUDE_min_sum_of_distances_l3197_319760

/-- Triangle ABC with given side lengths -/
structure Triangle :=
  (A B C : ℝ × ℝ)
  (ab_eq_ac : dist A B = dist A C)
  (ab_eq_5 : dist A B = 5)
  (bc_eq_6 : dist B C = 6)

/-- Point on the sides of the triangle -/
def PointOnSides (t : Triangle) : Set (ℝ × ℝ) :=
  {P | ∃ (s : ℝ), 0 ≤ s ∧ s ≤ 1 ∧
    (P = (1 - s) • t.A + s • t.B ∨
     P = (1 - s) • t.B + s • t.C ∨
     P = (1 - s) • t.C + s • t.A)}

/-- Sum of distances from P to vertices -/
def SumOfDistances (t : Triangle) (P : ℝ × ℝ) : ℝ :=
  dist P t.A + dist P t.B + dist P t.C

/-- Theorem: Minimum sum of distances is 16 -/
theorem min_sum_of_distances (t : Triangle) :
  ∀ P ∈ PointOnSides t, SumOfDistances t P ≥ 16 :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_distances_l3197_319760


namespace NUMINAMATH_CALUDE_fourth_term_of_gp_l3197_319737

def geometric_progression (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * r ^ (n - 1)

theorem fourth_term_of_gp (x : ℝ) :
  let a₁ := x
  let a₂ := 3 * x + 3
  let a₃ := 5 * x + 5
  let r := a₂ / a₁
  (∀ n : ℕ, n ≥ 1 → geometric_progression a₁ r n = if n = 1 then a₁ else if n = 2 then a₂ else if n = 3 then a₃ else 0) →
  geometric_progression a₁ r 4 = -125 / 12 :=
by sorry

end NUMINAMATH_CALUDE_fourth_term_of_gp_l3197_319737


namespace NUMINAMATH_CALUDE_line_symmetry_l3197_319778

/-- Given a line l₁: y = 2x + 1 and a point p: (1, 1), 
    the line l₂: y = 2x - 3 is symmetric to l₁ about p -/
theorem line_symmetry (x y : ℝ) : 
  (y = 2*x + 1) → 
  (∃ (x' y' : ℝ), y' = 2*x' - 3 ∧ 
    ((x + x') / 2 = 1 ∧ (y + y') / 2 = 1)) :=
by sorry

end NUMINAMATH_CALUDE_line_symmetry_l3197_319778


namespace NUMINAMATH_CALUDE_max_value_of_trigonometric_expression_l3197_319784

theorem max_value_of_trigonometric_expression :
  ∀ α : Real, 0 ≤ α → α ≤ π / 2 →
  (1 / (Real.sin α ^ 4 + Real.cos α ^ 4) ≤ 2) ∧
  (∃ α₀, 0 ≤ α₀ ∧ α₀ ≤ π / 2 ∧ 1 / (Real.sin α₀ ^ 4 + Real.cos α₀ ^ 4) = 2) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_trigonometric_expression_l3197_319784


namespace NUMINAMATH_CALUDE_total_traffic_tickets_l3197_319748

/-- The total number of traffic tickets Mark and Sarah have -/
def total_tickets (mark_parking : ℕ) (sarah_parking : ℕ) (mark_speeding : ℕ) (sarah_speeding : ℕ) : ℕ :=
  mark_parking + sarah_parking + mark_speeding + sarah_speeding

/-- Theorem stating the total number of traffic tickets Mark and Sarah have -/
theorem total_traffic_tickets :
  ∀ (mark_parking sarah_parking mark_speeding sarah_speeding : ℕ),
  mark_parking = 2 * sarah_parking →
  mark_speeding = sarah_speeding →
  sarah_speeding = 6 →
  mark_parking = 8 →
  total_tickets mark_parking sarah_parking mark_speeding sarah_speeding = 24 := by
  sorry

end NUMINAMATH_CALUDE_total_traffic_tickets_l3197_319748


namespace NUMINAMATH_CALUDE_infinitely_many_pairs_satisfying_conditions_l3197_319730

theorem infinitely_many_pairs_satisfying_conditions :
  ∃ (a : ℕ → ℕ), (∀ n : ℕ,
    (Nat.gcd (a n) (a (n + 1)) = 1) ∧
    ((a n) ∣ ((a (n + 1))^2 - 5)) ∧
    ((a (n + 1)) ∣ ((a n)^2 - 5))) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_pairs_satisfying_conditions_l3197_319730


namespace NUMINAMATH_CALUDE_square_root_squared_sqrt_987654_squared_l3197_319781

theorem square_root_squared (n : ℝ) (hn : 0 ≤ n) : (Real.sqrt n) ^ 2 = n := by sorry

theorem sqrt_987654_squared : (Real.sqrt 987654) ^ 2 = 987654 := by sorry

end NUMINAMATH_CALUDE_square_root_squared_sqrt_987654_squared_l3197_319781


namespace NUMINAMATH_CALUDE_solution_condition_l3197_319787

theorem solution_condition (m : ℚ) : (∀ x, m * x = m ↔ x = 1) ↔ m ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_solution_condition_l3197_319787


namespace NUMINAMATH_CALUDE_minimum_point_of_translated_graph_l3197_319724

def f (x : ℝ) : ℝ := |x + 4| - 9

theorem minimum_point_of_translated_graph :
  ∃! (x y : ℝ), f x = y ∧ ∀ z : ℝ, f z ≥ y ∧ (x, y) = (-4, -9) := by sorry

end NUMINAMATH_CALUDE_minimum_point_of_translated_graph_l3197_319724


namespace NUMINAMATH_CALUDE_bus_ride_cost_l3197_319742

theorem bus_ride_cost (train_cost bus_cost : ℝ) : 
  train_cost = bus_cost + 6.85 →
  (train_cost * 0.85 + (bus_cost + 1.25)) = 10.50 →
  bus_cost = 1.85 := by
  sorry

end NUMINAMATH_CALUDE_bus_ride_cost_l3197_319742


namespace NUMINAMATH_CALUDE_team_selection_ways_l3197_319739

-- Define the number of teachers and students
def num_teachers : ℕ := 5
def num_students : ℕ := 10

-- Define the function to calculate the number of ways to select one person from a group
def select_one (n : ℕ) : ℕ := n

-- Define the function to calculate the total number of ways to form a team
def total_ways : ℕ := select_one num_teachers * select_one num_students

-- Theorem statement
theorem team_selection_ways : total_ways = 50 := by
  sorry

end NUMINAMATH_CALUDE_team_selection_ways_l3197_319739


namespace NUMINAMATH_CALUDE_f_2018_is_zero_l3197_319790

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the properties of f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def has_period_property (f : ℝ → ℝ) : Prop := ∀ x, f (x + 4) - f x = 2 * f 2

-- State the theorem
theorem f_2018_is_zero 
  (h_even : is_even f) 
  (h_period : has_period_property f) : 
  f 2018 = 0 := by sorry

end NUMINAMATH_CALUDE_f_2018_is_zero_l3197_319790


namespace NUMINAMATH_CALUDE_abcc_equals_1966_l3197_319709

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents a four-digit number ABCC -/
def ABCC (A B C : Digit) : ℕ := 1000 * A.val + 100 * B.val + 11 * C.val

/-- The equation given in the problem -/
def EquationHolds (A B C D E : Digit) : Prop :=
  ABCC A B C = (11 * D.val - E.val) * 100 + 11 * D.val * E.val

/-- All digits are distinct -/
def AllDistinct (A B C D E : Digit) : Prop :=
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧
  B ≠ C ∧ B ≠ D ∧ B ≠ E ∧
  C ≠ D ∧ C ≠ E ∧
  D ≠ E

theorem abcc_equals_1966 :
  ∃ (A B C D E : Digit), AllDistinct A B C D E ∧ EquationHolds A B C D E ∧ ABCC A B C = 1966 :=
sorry

end NUMINAMATH_CALUDE_abcc_equals_1966_l3197_319709


namespace NUMINAMATH_CALUDE_largest_initial_number_l3197_319789

theorem largest_initial_number :
  ∃ (a b c d e : ℕ), 
    189 + a + b + c + d + e = 200 ∧
    a ≥ 2 ∧ b ≥ 2 ∧ c ≥ 2 ∧ d ≥ 2 ∧ e ≥ 2 ∧
    ¬(189 ∣ a) ∧ ¬(189 ∣ b) ∧ ¬(189 ∣ c) ∧ ¬(189 ∣ d) ∧ ¬(189 ∣ e) ∧
    ∀ (n : ℕ), n > 189 → 
      ¬∃ (x y z w v : ℕ),
        n + x + y + z + w + v = 200 ∧
        x ≥ 2 ∧ y ≥ 2 ∧ z ≥ 2 ∧ w ≥ 2 ∧ v ≥ 2 ∧
        ¬(n ∣ x) ∧ ¬(n ∣ y) ∧ ¬(n ∣ z) ∧ ¬(n ∣ w) ∧ ¬(n ∣ v) :=
by sorry

end NUMINAMATH_CALUDE_largest_initial_number_l3197_319789


namespace NUMINAMATH_CALUDE_triangle_properties_l3197_319776

noncomputable section

-- Define the triangle ABC
variable (A B C : Real)
variable (a b c : Real)

-- Define the conditions
axiom triangle_angles : A + B + C = Real.pi
axiom cos_A : Real.cos A = 1/3
axiom side_a : a = Real.sqrt 3

-- Define the theorem
theorem triangle_properties :
  (Real.sin ((B + C) / 2))^2 + Real.cos (2 * A) = -1/9 ∧
  (∀ x y : Real, x * y ≤ 9/4 ∧ (x = b ∧ y = c → x * y = 9/4)) :=
sorry

end NUMINAMATH_CALUDE_triangle_properties_l3197_319776


namespace NUMINAMATH_CALUDE_jim_bike_shop_profit_l3197_319798

/-- Represents Jim's bike shop financials for a month -/
structure BikeShop where
  tire_repair_price : ℕ
  tire_repair_cost : ℕ
  tire_repairs_count : ℕ
  complex_repair_price : ℕ
  complex_repair_cost : ℕ
  complex_repairs_count : ℕ
  retail_profit : ℕ
  fixed_expenses : ℕ

/-- Calculates the total profit of the bike shop -/
def total_profit (shop : BikeShop) : ℤ :=
  (shop.tire_repair_price - shop.tire_repair_cost) * shop.tire_repairs_count +
  (shop.complex_repair_price - shop.complex_repair_cost) * shop.complex_repairs_count +
  shop.retail_profit - shop.fixed_expenses

/-- Theorem stating that Jim's bike shop profit is $3000 -/
theorem jim_bike_shop_profit :
  ∃ (shop : BikeShop),
    shop.tire_repair_price = 20 ∧
    shop.tire_repair_cost = 5 ∧
    shop.tire_repairs_count = 300 ∧
    shop.complex_repair_price = 300 ∧
    shop.complex_repair_cost = 50 ∧
    shop.complex_repairs_count = 2 ∧
    shop.retail_profit = 2000 ∧
    shop.fixed_expenses = 4000 ∧
    total_profit shop = 3000 := by
  sorry

end NUMINAMATH_CALUDE_jim_bike_shop_profit_l3197_319798


namespace NUMINAMATH_CALUDE_tan_equality_periodic_l3197_319702

theorem tan_equality_periodic (n : ℤ) : 
  -180 < n ∧ n < 180 ∧ Real.tan (n * π / 180) = Real.tan (1500 * π / 180) → n = 60 := by
  sorry

end NUMINAMATH_CALUDE_tan_equality_periodic_l3197_319702


namespace NUMINAMATH_CALUDE_units_digit_of_p_squared_plus_3_to_p_l3197_319705

theorem units_digit_of_p_squared_plus_3_to_p (p : ℕ) : 
  p = 2017^3 + 3^2017 → (p^2 + 3^p) % 10 = 5 := by
sorry

end NUMINAMATH_CALUDE_units_digit_of_p_squared_plus_3_to_p_l3197_319705


namespace NUMINAMATH_CALUDE_decorative_object_height_correct_l3197_319750

/-- Represents a circular fountain with water jets -/
structure Fountain where
  diameter : ℝ
  max_height : ℝ
  max_height_distance : ℝ
  decorative_object_height : ℝ

/-- Properties of the specific fountain described in the problem -/
def problem_fountain : Fountain where
  diameter := 20
  max_height := 8
  max_height_distance := 2
  decorative_object_height := 7.5

/-- Theorem stating that the decorative object height is correct for the given fountain parameters -/
theorem decorative_object_height_correct (f : Fountain) 
  (h1 : f.diameter = 20)
  (h2 : f.max_height = 8)
  (h3 : f.max_height_distance = 2) :
  f.decorative_object_height = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_decorative_object_height_correct_l3197_319750


namespace NUMINAMATH_CALUDE_min_value_expression_l3197_319735

theorem min_value_expression (x : ℝ) (hx : x > 0) :
  (4 + x) * (1 + x) / x ≥ 9 ∧ ∃ y > 0, (4 + y) * (1 + y) / y = 9 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l3197_319735


namespace NUMINAMATH_CALUDE_base_conversion_problem_l3197_319794

theorem base_conversion_problem : ∃! (b : ℕ), b > 1 ∧ b ^ 3 ≤ 216 ∧ 216 < b ^ 4 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_base_conversion_problem_l3197_319794


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_slopes_l3197_319726

/-- The slopes of the asymptotes for the hyperbola described by the equation x²/144 - y²/81 = 1 are ±3/4 -/
theorem hyperbola_asymptote_slopes (x y : ℝ) :
  x^2 / 144 - y^2 / 81 = 1 →
  ∃ (m : ℝ), m = 3/4 ∧ (∀ (x' y' : ℝ), x'^2 / 144 - y'^2 / 81 = 0 → y' = m * x' ∨ y' = -m * x') :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_slopes_l3197_319726


namespace NUMINAMATH_CALUDE_two_digit_number_property_l3197_319718

theorem two_digit_number_property : ∃! n : ℕ, 
  (n ≥ 10 ∧ n < 100) ∧ 
  (n / 10 = 2 * (n % 10)) ∧ 
  (∃ m : ℕ, n + (n / 10)^2 = m^2) ∧
  n = 21 := by
sorry

end NUMINAMATH_CALUDE_two_digit_number_property_l3197_319718


namespace NUMINAMATH_CALUDE_angle_A_value_triangle_area_l3197_319795

namespace TriangleABC

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The condition a * sin(B) = √3 * b * cos(A) -/
def condition1 (t : Triangle) : Prop :=
  t.a * Real.sin t.B = Real.sqrt 3 * t.b * Real.cos t.A

/-- The conditions a = 3 and b = 2c -/
def condition2 (t : Triangle) : Prop :=
  t.a = 3 ∧ t.b = 2 * t.c

/-- The theorem stating that if condition1 holds, then A = π/3 -/
theorem angle_A_value (t : Triangle) (h : condition1 t) : t.A = Real.pi / 3 := by
  sorry

/-- The theorem stating that if condition1 and condition2 hold, then the area of the triangle is (3√3)/2 -/
theorem triangle_area (t : Triangle) (h1 : condition1 t) (h2 : condition2 t) : 
  (1/2) * t.a * t.b * Real.sin t.C = (3 * Real.sqrt 3) / 2 := by
  sorry

end TriangleABC

end NUMINAMATH_CALUDE_angle_A_value_triangle_area_l3197_319795


namespace NUMINAMATH_CALUDE_mrsHiltFramePerimeter_l3197_319793

/-- An irregular octagon with specified side lengths -/
structure IrregularOctagon where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ
  side5 : ℝ
  side6 : ℝ
  side7 : ℝ
  side8 : ℝ

/-- Calculate the perimeter of an irregular octagon -/
def perimeter (o : IrregularOctagon) : ℝ :=
  o.side1 + o.side2 + o.side3 + o.side4 + o.side5 + o.side6 + o.side7 + o.side8

/-- Mrs. Hilt's irregular octagonal picture frame -/
def mrsHiltFrame : IrregularOctagon :=
  { side1 := 10
    side2 := 9
    side3 := 11
    side4 := 6
    side5 := 7
    side6 := 2
    side7 := 3
    side8 := 4 }

/-- Theorem: The perimeter of Mrs. Hilt's irregular octagonal picture frame is 52 inches -/
theorem mrsHiltFramePerimeter : perimeter mrsHiltFrame = 52 := by
  sorry

end NUMINAMATH_CALUDE_mrsHiltFramePerimeter_l3197_319793


namespace NUMINAMATH_CALUDE_gcd_867_2553_l3197_319703

theorem gcd_867_2553 : Nat.gcd 867 2553 = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_867_2553_l3197_319703


namespace NUMINAMATH_CALUDE_rent_and_earnings_increase_l3197_319707

theorem rent_and_earnings_increase (last_year_earnings : ℝ) (increase_percent : ℝ) : 
  (0.3 * (last_year_earnings * (1 + increase_percent / 100)) = 2.025 * (0.2 * last_year_earnings)) →
  increase_percent = 35 := by
  sorry

end NUMINAMATH_CALUDE_rent_and_earnings_increase_l3197_319707


namespace NUMINAMATH_CALUDE_journey_time_proof_l3197_319788

/-- Proves that the total journey time is 5 hours given the specified conditions -/
theorem journey_time_proof (total_distance : ℝ) (speed1 speed2 : ℝ) (time1 : ℝ) :
  total_distance = 240 ∧ 
  speed1 = 40 ∧ 
  speed2 = 60 ∧ 
  time1 = 3 →
  speed1 * time1 + (total_distance - speed1 * time1) / speed2 + time1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_journey_time_proof_l3197_319788


namespace NUMINAMATH_CALUDE_min_values_ab_and_a_plus_2b_l3197_319782

theorem min_values_ab_and_a_plus_2b (a b : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : 1 / a + 2 / b = 1) : 
  (∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 1 / x + 2 / y = 1 → x * y ≥ 8) ∧
  (∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 1 / x + 2 / y = 1 → x + 2 * y ≥ 9) :=
by sorry

#check min_values_ab_and_a_plus_2b

end NUMINAMATH_CALUDE_min_values_ab_and_a_plus_2b_l3197_319782


namespace NUMINAMATH_CALUDE_notch_volume_minimized_l3197_319736

/-- A cylindrical notch with angle θ between bounding planes -/
structure CylindricalNotch where
  θ : Real
  (θ_pos : θ > 0)
  (θ_lt_pi : θ < π)

/-- The volume of the notch given the angle φ between one bounding plane and the horizontal -/
noncomputable def notchVolume (n : CylindricalNotch) (φ : Real) : Real :=
  (2/3) * (Real.tan φ + Real.tan (n.θ - φ))

/-- Theorem: The volume of the notch is minimized when the bounding planes are at equal angles to the horizontal -/
theorem notch_volume_minimized (n : CylindricalNotch) :
  ∃ (φ_min : Real), φ_min = n.θ / 2 ∧
    ∀ (φ : Real), 0 < φ ∧ φ < n.θ → notchVolume n φ_min ≤ notchVolume n φ :=
sorry

end NUMINAMATH_CALUDE_notch_volume_minimized_l3197_319736


namespace NUMINAMATH_CALUDE_isosceles_triangle_condition_l3197_319769

theorem isosceles_triangle_condition (A B C : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C →  -- Ensure positive angles
  A + B + C = π →  -- Triangle angle sum
  2 * Real.cos B * Real.sin A = Real.sin C →  -- Given condition
  A = B  -- Conclusion: isosceles triangle
:= by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_condition_l3197_319769


namespace NUMINAMATH_CALUDE_sum_of_three_greater_than_five_l3197_319740

theorem sum_of_three_greater_than_five (a b c : ℕ) :
  a ∈ Finset.range 10 →
  b ∈ Finset.range 10 →
  c ∈ Finset.range 10 →
  a ≠ b →
  a ≠ c →
  b ≠ c →
  a + b + c > 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_greater_than_five_l3197_319740


namespace NUMINAMATH_CALUDE_wire_bending_l3197_319755

/-- Given a wire that can be bent into either a circle or a square, 
    if the area of the square is 7737.769850454057 cm², 
    then the radius of the circle is approximately 56 cm. -/
theorem wire_bending (square_area : ℝ) (circle_radius : ℝ) : 
  square_area = 7737.769850454057 → 
  (circle_radius ≥ 55.99 ∧ circle_radius ≤ 56.01) := by
  sorry

#check wire_bending

end NUMINAMATH_CALUDE_wire_bending_l3197_319755


namespace NUMINAMATH_CALUDE_min_perimeter_triangle_APF_l3197_319713

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 2

-- Define the right focus F
def F : ℝ × ℝ := (2, 0)

-- Define point A
def A : ℝ × ℝ := (-1, 1)

-- Define a point P on the left branch of the hyperbola
def P : ℝ × ℝ := sorry

-- Define the perimeter of triangle APF
def perimeter (P : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem min_perimeter_triangle_APF :
  ∀ P, hyperbola P.1 P.2 → P.1 < 0 →
  perimeter P ≥ 3 * Real.sqrt 2 + Real.sqrt 10 :=
sorry

end NUMINAMATH_CALUDE_min_perimeter_triangle_APF_l3197_319713


namespace NUMINAMATH_CALUDE_equation_solutions_l3197_319700

theorem equation_solutions :
  (∀ X : ℝ, X - 12 = 81 → X = 93) ∧
  (∀ X : ℝ, 5.1 + X = 10.5 → X = 5.4) ∧
  (∀ X : ℝ, 6 * X = 4.2 → X = 0.7) ∧
  (∀ X : ℝ, X / 0.4 = 12.5 → X = 5) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3197_319700


namespace NUMINAMATH_CALUDE_vectors_collinear_l3197_319732

def a : Fin 3 → ℝ := ![3, -1, 6]
def b : Fin 3 → ℝ := ![5, 7, 10]
def c₁ : Fin 3 → ℝ := λ i => 4 * a i - 2 * b i
def c₂ : Fin 3 → ℝ := λ i => b i - 2 * a i

theorem vectors_collinear : ∃ (k : ℝ), k ≠ 0 ∧ (∀ i : Fin 3, c₁ i = k * c₂ i) := by
  sorry

end NUMINAMATH_CALUDE_vectors_collinear_l3197_319732
