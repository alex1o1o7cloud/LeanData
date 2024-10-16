import Mathlib

namespace NUMINAMATH_CALUDE_mike_picked_52_peaches_l1501_150182

/-- The number of peaches Mike picked -/
def peaches_picked (initial final : ℕ) : ℕ := final - initial

/-- Proof that Mike picked 52 peaches -/
theorem mike_picked_52_peaches (initial final : ℕ) 
  (h1 : initial = 34) 
  (h2 : final = 86) : 
  peaches_picked initial final = 52 := by
  sorry

end NUMINAMATH_CALUDE_mike_picked_52_peaches_l1501_150182


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1501_150195

theorem quadratic_inequality (a b c d : ℝ) 
  (h1 : b > d) 
  (h2 : b > 0) 
  (h3 : d > 0) 
  (h4 : Real.sqrt (a^2 - 4*b) > Real.sqrt (c^2 - 4*d)) : 
  a^2 - c^2 > b - d := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1501_150195


namespace NUMINAMATH_CALUDE_factorization_x2_minus_4x_minus_12_minimum_value_4x2_plus_4x_minus_1_l1501_150190

-- Problem 1
theorem factorization_x2_minus_4x_minus_12 :
  ∀ x : ℝ, x^2 - 4*x - 12 = (x - 6) * (x + 2) := by sorry

-- Problem 2
theorem minimum_value_4x2_plus_4x_minus_1 :
  ∀ x : ℝ, 4*x^2 + 4*x - 1 ≥ -2 ∧
  ∃ x : ℝ, 4*x^2 + 4*x - 1 = -2 ∧ x = -1/2 := by sorry

end NUMINAMATH_CALUDE_factorization_x2_minus_4x_minus_12_minimum_value_4x2_plus_4x_minus_1_l1501_150190


namespace NUMINAMATH_CALUDE_seven_power_minus_three_times_two_power_l1501_150106

theorem seven_power_minus_three_times_two_power (x y : ℕ+) : 
  7^(x.val) - 3 * 2^(y.val) = 1 ↔ (x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 4) := by
sorry

end NUMINAMATH_CALUDE_seven_power_minus_three_times_two_power_l1501_150106


namespace NUMINAMATH_CALUDE_expression_equality_l1501_150168

theorem expression_equality : 3 * 2020 + 2 * 2020 - 4 * 2020 = 2020 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1501_150168


namespace NUMINAMATH_CALUDE_jose_investment_is_45000_l1501_150173

/-- Represents the investment and profit scenario of Tom and Jose's shop --/
structure ShopInvestment where
  tom_investment : ℕ
  jose_join_delay : ℕ
  total_profit : ℕ
  jose_profit : ℕ

/-- Calculates Jose's investment based on the given conditions --/
def calculate_jose_investment (shop : ShopInvestment) : ℕ :=
  let tom_investment_months := shop.tom_investment * 12
  let jose_investment_months := (12 - shop.jose_join_delay) * (shop.total_profit - shop.jose_profit) * 10 / shop.jose_profit
  jose_investment_months / (12 - shop.jose_join_delay)

/-- Theorem stating that Jose's investment is 45000 given the specified conditions --/
theorem jose_investment_is_45000 (shop : ShopInvestment)
  (h1 : shop.tom_investment = 30000)
  (h2 : shop.jose_join_delay = 2)
  (h3 : shop.total_profit = 36000)
  (h4 : shop.jose_profit = 20000) :
  calculate_jose_investment shop = 45000 := by
  sorry

#eval calculate_jose_investment ⟨30000, 2, 36000, 20000⟩

end NUMINAMATH_CALUDE_jose_investment_is_45000_l1501_150173


namespace NUMINAMATH_CALUDE_factor_of_polynomial_l1501_150180

theorem factor_of_polynomial (x : ℝ) : 
  ∃ (q : ℝ → ℝ), (x^4 - 6*x^2 + 9 : ℝ) = (x^2 - 3) * q x := by
  sorry

end NUMINAMATH_CALUDE_factor_of_polynomial_l1501_150180


namespace NUMINAMATH_CALUDE_minimum_value_of_expression_l1501_150126

theorem minimum_value_of_expression (x : ℝ) (h : x > 1) :
  x + 1 / (x - 1) ≥ 3 ∧ ∃ y > 1, y + 1 / (y - 1) = 3 :=
by sorry

end NUMINAMATH_CALUDE_minimum_value_of_expression_l1501_150126


namespace NUMINAMATH_CALUDE_red_jelly_beans_l1501_150117

/-- The number of red jelly beans in a bag, given the following conditions:
  1. It takes three bags of jelly beans to fill the fishbowl.
  2. Each bag has a similar distribution of colors.
  3. One bag contains: 13 black, 36 green, 28 purple, 32 yellow, and 18 white jelly beans.
  4. The total number of red and white jelly beans in the fishbowl is 126. -/
theorem red_jelly_beans (black green purple yellow white : ℕ)
  (h1 : black = 13)
  (h2 : green = 36)
  (h3 : purple = 28)
  (h4 : yellow = 32)
  (h5 : white = 18)
  (h6 : (red + white) * 3 = 126) :
  red = 24 :=
sorry

end NUMINAMATH_CALUDE_red_jelly_beans_l1501_150117


namespace NUMINAMATH_CALUDE_tan_beta_value_l1501_150186

theorem tan_beta_value (α β : Real) 
  (h1 : Real.tan α = 1/3) 
  (h2 : Real.tan (α + β) = 1/2) : 
  Real.tan β = 1/7 := by
sorry

end NUMINAMATH_CALUDE_tan_beta_value_l1501_150186


namespace NUMINAMATH_CALUDE_quadratic_root_sum_l1501_150179

theorem quadratic_root_sum (x₁ x₂ : ℝ) : 
  (2 * x₁^2 - 3 * x₁ - 5 = 0) → 
  (2 * x₂^2 - 3 * x₂ - 5 = 0) → 
  (x₁ ≠ x₂) →
  (x₁ + x₂ = 3/2) := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_sum_l1501_150179


namespace NUMINAMATH_CALUDE_hexagon_perimeter_l1501_150194

/-- The perimeter of a hexagon with side length 7 inches is 42 inches. -/
theorem hexagon_perimeter : 
  ∀ (hexagon_side_length : ℝ), 
  hexagon_side_length = 7 → 
  6 * hexagon_side_length = 42 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_perimeter_l1501_150194


namespace NUMINAMATH_CALUDE_birthday_celebration_attendance_l1501_150147

/-- The number of people who stayed at a birthday celebration --/
def people_stayed (total_guests : ℕ) (men : ℕ) (children_left : ℕ) : ℕ :=
  let women := total_guests / 2
  let children := total_guests - women - men
  let men_left := men / 3
  total_guests - men_left - children_left

/-- Theorem about the number of people who stayed at the birthday celebration --/
theorem birthday_celebration_attendance :
  people_stayed 60 15 5 = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_birthday_celebration_attendance_l1501_150147


namespace NUMINAMATH_CALUDE_unique_first_degree_polynomial_l1501_150148

/-- The polynomial p(x) = 2x + 1 -/
def p (x : ℝ) : ℝ := 2 * x + 1

/-- The polynomial q(x) = x -/
def q (x : ℝ) : ℝ := x

theorem unique_first_degree_polynomial :
  ∀ (x : ℝ), p (p (q x)) = q (p (p x)) ∧
  ∀ (r : ℝ → ℝ), (∃ (a b : ℝ), ∀ (x : ℝ), r x = a * x + b) →
  (∀ (x : ℝ), p (p (r x)) = r (p (p x))) →
  r = q :=
sorry

end NUMINAMATH_CALUDE_unique_first_degree_polynomial_l1501_150148


namespace NUMINAMATH_CALUDE_theta_range_l1501_150144

theorem theta_range (θ : Real) (h1 : θ ∈ Set.Icc 0 (2 * Real.pi))
  (h2 : Real.cos θ ^ 5 - Real.sin θ ^ 5 < 7 * (Real.sin θ ^ 3 - Real.cos θ ^ 3)) :
  θ ∈ Set.Ioo (Real.pi / 4) (5 * Real.pi / 4) := by
  sorry

end NUMINAMATH_CALUDE_theta_range_l1501_150144


namespace NUMINAMATH_CALUDE_sin_theta_minus_pi_l1501_150198

theorem sin_theta_minus_pi (θ : Real) :
  (∃ (x y : Real), x = 3 ∧ y = -4 ∧ x = Real.cos θ * Real.sqrt (x^2 + y^2) ∧ y = Real.sin θ * Real.sqrt (x^2 + y^2)) →
  Real.sin (θ - Real.pi) = 4/5 := by
sorry

end NUMINAMATH_CALUDE_sin_theta_minus_pi_l1501_150198


namespace NUMINAMATH_CALUDE_solve_equation_l1501_150105

theorem solve_equation (C D : ℚ) 
  (eq1 : 2 * C + 3 * D + 4 = 31)
  (eq2 : D = C + 2) :
  C = 21 / 5 ∧ D = 31 / 5 := by
sorry

end NUMINAMATH_CALUDE_solve_equation_l1501_150105


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l1501_150122

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 2

-- State the theorem
theorem circle_center_and_radius :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (∀ x y, circle_equation x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2) ∧
    center = (-1, 0) ∧
    radius = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l1501_150122


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l1501_150136

theorem parallel_vectors_x_value (x : ℝ) :
  let a : Fin 2 → ℝ := ![1, x]
  let b : Fin 2 → ℝ := ![2, 2 - x]
  (∃ (k : ℝ), a = k • b) → x = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l1501_150136


namespace NUMINAMATH_CALUDE_triangle_perimeter_l1501_150163

theorem triangle_perimeter (a b c : ℝ) (ha : a = 10) (hb : b = 7) (hc : c = 5)
  (h_triangle : a + b > c ∧ a + c > b ∧ b + c > a) :
  a + b + c = 22 := by
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l1501_150163


namespace NUMINAMATH_CALUDE_connie_marbles_left_l1501_150199

/-- Calculates the number of marbles Connie has left after giving some away. -/
def marblesLeft (initialMarbles giveAwayMarbles : ℕ) : ℕ :=
  initialMarbles - giveAwayMarbles

/-- Proves that Connie has 70 marbles left after giving 73 marbles to Juan,
    starting from 143 marbles. -/
theorem connie_marbles_left : marblesLeft 143 73 = 70 := by
  sorry

end NUMINAMATH_CALUDE_connie_marbles_left_l1501_150199


namespace NUMINAMATH_CALUDE_cone_surface_area_l1501_150164

/-- The surface area of a cone given its slant height and angle between slant height and axis -/
theorem cone_surface_area (slant_height : ℝ) (angle : ℝ) : 
  slant_height = 20 →
  angle = 30 * π / 180 →
  ∃ (surface_area : ℝ), surface_area = 300 * π := by
sorry

end NUMINAMATH_CALUDE_cone_surface_area_l1501_150164


namespace NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l1501_150187

-- Define propositions p and q
def p (x : ℝ) : Prop := |x| < 2
def q (x : ℝ) : Prop := x^2 - x - 2 < 0

-- Define the negations of p and q
def not_p (x : ℝ) : Prop := ¬(p x)
def not_q (x : ℝ) : Prop := ¬(q x)

-- Theorem stating that ¬p is a sufficient but not necessary condition for ¬q
theorem not_p_sufficient_not_necessary_for_not_q :
  (∀ x : ℝ, not_p x → not_q x) ∧ 
  (∃ x : ℝ, not_q x ∧ ¬(not_p x)) :=
sorry

end NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l1501_150187


namespace NUMINAMATH_CALUDE_prism_configuration_impossible_l1501_150177

/-- A rectangular prism in 3D space -/
structure RectangularPrism where
  x_min : ℝ
  x_max : ℝ
  y_min : ℝ
  y_max : ℝ
  z_min : ℝ
  z_max : ℝ
  h_x : x_min < x_max
  h_y : y_min < y_max
  h_z : z_min < z_max

/-- Two prisms intersect if their projections overlap on all axes -/
def intersects (p q : RectangularPrism) : Prop :=
  (p.x_min < q.x_max ∧ q.x_min < p.x_max) ∧
  (p.y_min < q.y_max ∧ q.y_min < p.y_max) ∧
  (p.z_min < q.z_max ∧ q.z_min < p.z_max)

/-- A configuration of 12 prisms satisfying the problem conditions -/
structure PrismConfiguration where
  prisms : Fin 12 → RectangularPrism
  h_intersects : ∀ i j : Fin 12, i ≠ j → 
    (i.val + 1) % 12 ≠ j.val ∧ (i.val + 11) % 12 ≠ j.val → 
    intersects (prisms i) (prisms j)
  h_non_intersects : ∀ i : Fin 12, 
    ¬intersects (prisms i) (prisms ⟨(i.val + 1) % 12, sorry⟩) ∧
    ¬intersects (prisms i) (prisms ⟨(i.val + 11) % 12, sorry⟩)

/-- The main theorem stating the impossibility of such a configuration -/
theorem prism_configuration_impossible : ¬∃ (config : PrismConfiguration), True :=
  sorry

end NUMINAMATH_CALUDE_prism_configuration_impossible_l1501_150177


namespace NUMINAMATH_CALUDE_gcd_problem_l1501_150118

theorem gcd_problem (X Y : ℕ) (h1 : Nat.lcm X Y = 180) (h2 : X * 5 = Y * 2) : 
  Nat.gcd X Y = 18 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l1501_150118


namespace NUMINAMATH_CALUDE_bake_sale_cookies_l1501_150183

/-- The number of cookies in each box -/
def cookies_per_box : ℕ := 48

/-- The number of boxes Abigail collected -/
def abigail_boxes : ℕ := 2

/-- The number of quarter boxes Grayson collected -/
def grayson_quarter_boxes : ℕ := 3

/-- The number of boxes Olivia collected -/
def olivia_boxes : ℕ := 3

/-- The total number of cookies collected -/
def total_cookies : ℕ := 276

theorem bake_sale_cookies : 
  cookies_per_box * abigail_boxes + 
  (cookies_per_box / 4) * grayson_quarter_boxes + 
  cookies_per_box * olivia_boxes = total_cookies := by
sorry

end NUMINAMATH_CALUDE_bake_sale_cookies_l1501_150183


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1501_150124

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  (a 4 + a 5 + a 6 + a 7 + a 8 = 150) →
  a 6 = 30 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1501_150124


namespace NUMINAMATH_CALUDE_logarithm_sum_simplification_l1501_150100

theorem logarithm_sum_simplification :
  1 / (Real.log 3 / Real.log 20 + 1) +
  1 / (Real.log 4 / Real.log 15 + 1) +
  1 / (Real.log 7 / Real.log 12 + 1) = 4 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_sum_simplification_l1501_150100


namespace NUMINAMATH_CALUDE_ribbon_fraction_per_box_l1501_150108

theorem ribbon_fraction_per_box 
  (total_fraction : ℚ) 
  (num_boxes : ℕ) 
  (h1 : total_fraction = 5/12) 
  (h2 : num_boxes = 5) : 
  total_fraction / num_boxes = 1/12 := by
sorry

end NUMINAMATH_CALUDE_ribbon_fraction_per_box_l1501_150108


namespace NUMINAMATH_CALUDE_optimal_pan_dimensions_l1501_150185

def is_valid_pan (m n : ℕ) : Prop :=
  (m - 2) * (n - 2) = 2 * m + 2 * n - 4

def perimeter (m n : ℕ) : ℕ := 2 * m + 2 * n

def area (m n : ℕ) : ℕ := m * n

theorem optimal_pan_dimensions :
  ∀ m n : ℕ, m > 2 ∧ n > 2 → is_valid_pan m n →
    (perimeter m n ≥ perimeter 6 8) ∧
    (perimeter m n = perimeter 6 8 → area m n ≤ area 6 8) ∧
    is_valid_pan 6 8 :=
by sorry

end NUMINAMATH_CALUDE_optimal_pan_dimensions_l1501_150185


namespace NUMINAMATH_CALUDE_two_identical_squares_exist_l1501_150133

-- Define the type for the table entries
inductive Entry
| Zero
| One

-- Define the 5x5 table
def Table := Fin 5 → Fin 5 → Entry

-- Define the property of having ones in top-left and bottom-right corners, and zeros in the other corners
def CornerCondition (t : Table) : Prop :=
  t 0 0 = Entry.One ∧
  t 4 4 = Entry.One ∧
  t 0 4 = Entry.Zero ∧
  t 4 0 = Entry.Zero

-- Define a 2x2 square in the table
def Square (t : Table) (i j : Fin 4) : Fin 2 → Fin 2 → Entry :=
  fun x y => t (i + x) (j + y)

-- Define when two squares are equal
def SquaresEqual (s1 s2 : Fin 2 → Fin 2 → Entry) : Prop :=
  ∀ (x y : Fin 2), s1 x y = s2 x y

-- The main theorem
theorem two_identical_squares_exist (t : Table) (h : CornerCondition t) :
  ∃ (i1 j1 i2 j2 : Fin 4), (i1, j1) ≠ (i2, j2) ∧
    SquaresEqual (Square t i1 j1) (Square t i2 j2) := by
  sorry

end NUMINAMATH_CALUDE_two_identical_squares_exist_l1501_150133


namespace NUMINAMATH_CALUDE_recruit_line_total_l1501_150129

/-- Represents the position of a person in the line of recruits -/
structure Position where
  front : Nat
  behind : Nat

/-- The line of recruits -/
structure RecruitLine where
  peter : Position
  nikolai : Position
  denis : Position
  total : Nat

/-- The conditions of the problem -/
def initial_conditions : RecruitLine := {
  peter := { front := 50, behind := 0 },
  nikolai := { front := 100, behind := 0 },
  denis := { front := 170, behind := 0 },
  total := 0
}

/-- The condition after turning around -/
def turn_around_condition (line : RecruitLine) : Prop :=
  (line.peter.behind = 50 ∧ line.nikolai.behind = 100 ∧ line.denis.behind = 170) ∧
  ((4 * line.peter.front = line.nikolai.front ∧ line.peter.behind = 4 * line.nikolai.behind) ∨
   (4 * line.nikolai.front = line.denis.front ∧ line.nikolai.behind = 4 * line.denis.behind) ∨
   (4 * line.peter.front = line.denis.front ∧ line.peter.behind = 4 * line.denis.behind))

/-- The theorem to prove -/
theorem recruit_line_total (line : RecruitLine) :
  turn_around_condition line →
  line.total = 211 :=
by sorry

end NUMINAMATH_CALUDE_recruit_line_total_l1501_150129


namespace NUMINAMATH_CALUDE_prime_power_difference_l1501_150159

theorem prime_power_difference (n : ℕ+) (p : ℕ) (k : ℕ) :
  (3 : ℕ) ^ n.val - (2 : ℕ) ^ n.val = p ^ k ∧ Nat.Prime p → Nat.Prime n.val :=
by sorry

end NUMINAMATH_CALUDE_prime_power_difference_l1501_150159


namespace NUMINAMATH_CALUDE_sugar_calculation_l1501_150145

theorem sugar_calculation (original_sugar : ℚ) (recipe_fraction : ℚ) : 
  original_sugar = 7 + 1/3 →
  recipe_fraction = 2/3 →
  recipe_fraction * original_sugar = 4 + 8/9 := by
sorry

end NUMINAMATH_CALUDE_sugar_calculation_l1501_150145


namespace NUMINAMATH_CALUDE_probability_three_same_group_l1501_150127

/-- The number of students in the school -/
def total_students : ℕ := 600

/-- The number of lunch groups -/
def num_groups : ℕ := 3

/-- Assumption that the groups are of equal size -/
axiom groups_equal_size : total_students % num_groups = 0

/-- The probability of a student being assigned to a specific group -/
def prob_one_group : ℚ := 1 / num_groups

/-- The probability of three specific students being assigned to the same lunch group -/
def prob_three_same_group : ℚ := prob_one_group * prob_one_group

theorem probability_three_same_group :
  prob_three_same_group = 1 / 9 :=
sorry

end NUMINAMATH_CALUDE_probability_three_same_group_l1501_150127


namespace NUMINAMATH_CALUDE_max_D_is_240_l1501_150146

/-- Represents a building block with three binary attributes -/
structure Block :=
  (shape : Bool)
  (color : Bool)
  (city : Bool)

/-- Calculates the number of ways to select n blocks from 8 blocks
    such that each subsequent block shares exactly two attributes
    with the previously selected block -/
def D (n : Nat) : Nat :=
  sorry

/-- The set of all possible blocks -/
def allBlocks : Finset Block :=
  sorry

theorem max_D_is_240 :
  2 ≤ 8 ∧ (∀ n : Nat, 2 ≤ n ∧ n ≤ 8 → D n ≤ 240) ∧ (∃ n : Nat, 2 ≤ n ∧ n ≤ 8 ∧ D n = 240) := by
  sorry

end NUMINAMATH_CALUDE_max_D_is_240_l1501_150146


namespace NUMINAMATH_CALUDE_sum_of_xy_l1501_150152

theorem sum_of_xy (x y : ℕ+) 
  (eq1 : 10 * x + y = 75)
  (eq2 : 10 * y + x = 57) : 
  x + y = 12 := by
sorry

end NUMINAMATH_CALUDE_sum_of_xy_l1501_150152


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_least_subtraction_1234567890_div_17_l1501_150176

theorem least_subtraction_for_divisibility (n : Nat) (d : Nat) (h : d > 0) :
  ∃ (k : Nat), k < d ∧ (n - k) % d = 0 ∧ ∀ (m : Nat), m < k → (n - m) % d ≠ 0 :=
by sorry

theorem least_subtraction_1234567890_div_17 :
  ∃ (k : Nat), k < 17 ∧ (1234567890 - k) % 17 = 0 ∧
  ∀ (m : Nat), m < k → (1234567890 - m) % 17 ≠ 0 ∧ k = 5 :=
by sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_least_subtraction_1234567890_div_17_l1501_150176


namespace NUMINAMATH_CALUDE_intersection_point_inequality_l1501_150150

theorem intersection_point_inequality (a b : ℝ) :
  (∃ x₀ : ℝ, x₀ > 0 ∧ Real.exp x₀ = a * Real.sin x₀ + b * Real.sqrt x₀) →
  a^2 + b^2 > Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_inequality_l1501_150150


namespace NUMINAMATH_CALUDE_presidency_meeting_combinations_l1501_150102

/-- The number of schools participating in the conference -/
def num_schools : ℕ := 4

/-- The number of members in each school -/
def members_per_school : ℕ := 5

/-- The number of representatives sent by the host school -/
def host_representatives : ℕ := 3

/-- The number of representatives sent by each non-host school -/
def non_host_representatives : ℕ := 1

/-- The total number of ways to choose representatives for the presidency meeting -/
def total_ways : ℕ := num_schools * (members_per_school.choose host_representatives) * 
  (members_per_school ^ (num_schools - 1))

theorem presidency_meeting_combinations : total_ways = 5000 := by
  sorry

end NUMINAMATH_CALUDE_presidency_meeting_combinations_l1501_150102


namespace NUMINAMATH_CALUDE_doughnut_cost_calculation_l1501_150103

theorem doughnut_cost_calculation (num_doughnuts : ℕ) (price_per_doughnut : ℚ) (profit : ℚ) :
  let total_revenue := num_doughnuts * price_per_doughnut
  let cost_of_ingredients := total_revenue - profit
  cost_of_ingredients = num_doughnuts * price_per_doughnut - profit :=
by sorry

-- Example usage with the given values
def dorothy_example : ℚ :=
  let num_doughnuts : ℕ := 25
  let price_per_doughnut : ℚ := 3
  let profit : ℚ := 22
  num_doughnuts * price_per_doughnut - profit

#eval dorothy_example -- This should evaluate to 53

end NUMINAMATH_CALUDE_doughnut_cost_calculation_l1501_150103


namespace NUMINAMATH_CALUDE_machinery_expenditure_l1501_150134

/-- Proves that the amount spent on machinery is $2000 --/
theorem machinery_expenditure (total : ℝ) (raw_materials : ℝ) (cash_percentage : ℝ) :
  total = 5555.56 →
  raw_materials = 3000 →
  cash_percentage = 0.1 →
  total = raw_materials + (total * cash_percentage) + 2000 := by
  sorry

end NUMINAMATH_CALUDE_machinery_expenditure_l1501_150134


namespace NUMINAMATH_CALUDE_time_interval_is_two_seconds_l1501_150178

/-- The time interval for birth and death rates in a city --/
def time_interval (birth_rate death_rate net_increase_per_day seconds_per_day : ℕ) : ℚ :=
  seconds_per_day / (net_increase_per_day / (birth_rate - death_rate))

/-- Theorem: The time interval for birth and death rates is 2 seconds --/
theorem time_interval_is_two_seconds :
  time_interval 4 2 86400 86400 = 2 := by
  sorry

#eval time_interval 4 2 86400 86400

end NUMINAMATH_CALUDE_time_interval_is_two_seconds_l1501_150178


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1501_150196

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | |x| > 1}
def B : Set ℝ := {x : ℝ | 0 < x ∧ x < 2}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1501_150196


namespace NUMINAMATH_CALUDE_regular_polygon_with_150_degree_angles_has_12_sides_l1501_150181

/-- A regular polygon with interior angles of 150 degrees has 12 sides. -/
theorem regular_polygon_with_150_degree_angles_has_12_sides :
  ∀ n : ℕ, 
    n > 2 →
    (∀ angle : ℝ, angle = 150) →
    (180 * (n - 2) : ℝ) = (n * 150 : ℝ) →
    n = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_with_150_degree_angles_has_12_sides_l1501_150181


namespace NUMINAMATH_CALUDE_expression_evaluation_l1501_150109

theorem expression_evaluation : -30 + 12 * (8 / 4)^2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1501_150109


namespace NUMINAMATH_CALUDE_math_competition_participation_l1501_150111

theorem math_competition_participation (total_students : ℕ) (non_participants : ℕ) 
  (h1 : total_students = 39) (h2 : non_participants = 26) :
  (total_students - non_participants : ℚ) / total_students = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_math_competition_participation_l1501_150111


namespace NUMINAMATH_CALUDE_tangent_sum_difference_l1501_150123

theorem tangent_sum_difference (α β : Real) 
  (h1 : Real.tan (α + β) = 2/5)
  (h2 : Real.tan (β - π/4) = 1/4) :
  Real.tan (α + π/4) = 3/22 := by
  sorry

end NUMINAMATH_CALUDE_tangent_sum_difference_l1501_150123


namespace NUMINAMATH_CALUDE_person_age_puzzle_l1501_150135

theorem person_age_puzzle : ∃ (age : ℕ), 
  (3 * (age + 3) - 3 * (age - 3) = age) ∧ (age = 18) := by
  sorry

end NUMINAMATH_CALUDE_person_age_puzzle_l1501_150135


namespace NUMINAMATH_CALUDE_car_distance_calculation_car_distance_is_432_l1501_150115

/-- Given a car's journey with known time and alternative speed, calculate the distance. -/
theorem car_distance_calculation (initial_time : ℝ) (new_speed : ℝ) (time_ratio : ℝ) : ℝ :=
  let new_time := initial_time * time_ratio
  let distance := new_speed * new_time
  distance

/-- Prove that the distance covered by the car is 432 km. -/
theorem car_distance_is_432 :
  car_distance_calculation 6 48 (3/2) = 432 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_calculation_car_distance_is_432_l1501_150115


namespace NUMINAMATH_CALUDE_product_closest_to_105_l1501_150130

def product : ℝ := 2.1 * (50.2 + 0.09)

def options : List ℝ := [100, 105, 106, 110]

theorem product_closest_to_105 : 
  ∀ x ∈ options, |product - 105| ≤ |product - x| := by
  sorry

end NUMINAMATH_CALUDE_product_closest_to_105_l1501_150130


namespace NUMINAMATH_CALUDE_odd_symmetric_function_sum_l1501_150175

/-- A function is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function is symmetric about x=2 if f(2+x) = f(2-x) for all x -/
def IsSymmetricAbout2 (f : ℝ → ℝ) : Prop :=
  ∀ x, f (2 + x) = f (2 - x)

theorem odd_symmetric_function_sum (f : ℝ → ℝ) 
    (h_odd : IsOdd f) 
    (h_sym : IsSymmetricAbout2 f) 
    (h_f2 : f 2 = 2018) : 
  f 2018 + f 2016 = 2018 := by
  sorry

end NUMINAMATH_CALUDE_odd_symmetric_function_sum_l1501_150175


namespace NUMINAMATH_CALUDE_caitlin_age_l1501_150174

/-- Prove that Caitlin's age is 29 years -/
theorem caitlin_age :
  let aunt_anna_age : ℕ := 54
  let brianna_age : ℕ := (2 * aunt_anna_age) / 3
  let caitlin_age : ℕ := brianna_age - 7
  caitlin_age = 29 := by
  sorry

end NUMINAMATH_CALUDE_caitlin_age_l1501_150174


namespace NUMINAMATH_CALUDE_intersection_M_N_l1501_150101

def M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}
def N : Set ℝ := {0, 1, 2}

theorem intersection_M_N : M ∩ N = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1501_150101


namespace NUMINAMATH_CALUDE_census_suitable_for_electricity_usage_l1501_150188

/-- Represents a survey method -/
inductive SurveyMethod
| Census
| Sampling

/-- Represents a survey population -/
structure Population where
  size : ℕ
  is_small : Bool
  is_manageable : Bool

/-- Represents a survey -/
structure Survey where
  population : Population
  method : SurveyMethod
  is_practical : Bool

/-- Theorem: A census method is most suitable for investigating the monthly average 
    electricity usage of 10 households in a residential building -/
theorem census_suitable_for_electricity_usage : 
  ∀ (p : Population) (s : Survey),
  p.size = 10 → 
  p.is_small = true → 
  p.is_manageable = true → 
  s.population = p → 
  s.is_practical = true → 
  s.method = SurveyMethod.Census :=
by sorry

end NUMINAMATH_CALUDE_census_suitable_for_electricity_usage_l1501_150188


namespace NUMINAMATH_CALUDE_leaps_per_meter_calculation_l1501_150184

/-- Represents the number of leaps in one meter given the relationships between strides, leaps, bounds, and meters. -/
def leaps_per_meter (x y z w u v : ℚ) : ℚ :=
  (u * w) / (v * z)

/-- Theorem stating that given the relationships between units, one meter equals (uw/vz) leaps. -/
theorem leaps_per_meter_calculation
  (x y z w u v : ℚ)
  (h1 : x * 1 = y)  -- x strides = y leaps
  (h2 : z * 1 = w)  -- z bounds = w leaps
  (h3 : u * 1 = v)  -- u bounds = v meters
  : leaps_per_meter x y z w u v = (u * w) / (v * z) := by
  sorry

#check leaps_per_meter_calculation

end NUMINAMATH_CALUDE_leaps_per_meter_calculation_l1501_150184


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l1501_150193

theorem quadratic_equal_roots (a : ℝ) :
  (∃ x : ℝ, x * (x + 1) + a * x = 0 ∧
   ∀ y : ℝ, y * (y + 1) + a * y = 0 → y = x) →
  a = -1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l1501_150193


namespace NUMINAMATH_CALUDE_power_product_equals_128_l1501_150114

theorem power_product_equals_128 (b : ℕ) (h : b = 2) : b^3 * b^4 = 128 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equals_128_l1501_150114


namespace NUMINAMATH_CALUDE_lines_per_page_l1501_150155

theorem lines_per_page (total_lines : ℕ) (num_pages : ℕ) (lines_per_page : ℕ) 
  (h1 : total_lines = 150)
  (h2 : num_pages = 5)
  (h3 : lines_per_page * num_pages = total_lines) :
  lines_per_page = 30 := by
  sorry

end NUMINAMATH_CALUDE_lines_per_page_l1501_150155


namespace NUMINAMATH_CALUDE_original_number_of_people_l1501_150143

theorem original_number_of_people (x : ℕ) : 
  (3 * x / 4 : ℚ) - (3 * x / 20 : ℚ) = 16 → x = 27 := by
  sorry

end NUMINAMATH_CALUDE_original_number_of_people_l1501_150143


namespace NUMINAMATH_CALUDE_power_of_625_four_fifths_l1501_150166

theorem power_of_625_four_fifths :
  (625 : ℝ) ^ (4/5 : ℝ) = 125 * (5 : ℝ) ^ (1/5 : ℝ) :=
by
  sorry

end NUMINAMATH_CALUDE_power_of_625_four_fifths_l1501_150166


namespace NUMINAMATH_CALUDE_triangle_equilateral_from_cos_product_l1501_150158

/-- A triangle is equilateral if all its angles are equal -/
def IsEquilateral (A B C : ℝ) : Prop := A = B ∧ B = C

/-- Given a triangle ABC, if cos(A-B)cos(B-C)cos(C-A)=1, then the triangle is equilateral -/
theorem triangle_equilateral_from_cos_product (A B C : ℝ) 
  (h : Real.cos (A - B) * Real.cos (B - C) * Real.cos (C - A) = 1) : 
  IsEquilateral A B C := by
  sorry


end NUMINAMATH_CALUDE_triangle_equilateral_from_cos_product_l1501_150158


namespace NUMINAMATH_CALUDE_sunscreen_price_proof_l1501_150119

/-- Calculates the discounted price of sunscreen for a year --/
def discounted_sunscreen_price (bottles_per_month : ℕ) (months_per_year : ℕ) 
  (price_per_bottle : ℚ) (discount_percentage : ℚ) : ℚ :=
  let total_bottles := bottles_per_month * months_per_year
  let total_price := total_bottles * price_per_bottle
  let discount_amount := total_price * (discount_percentage / 100)
  total_price - discount_amount

/-- Proves that the discounted price of sunscreen for a year is $252.00 --/
theorem sunscreen_price_proof :
  discounted_sunscreen_price 1 12 30 30 = 252 := by
  sorry

#eval discounted_sunscreen_price 1 12 30 30

end NUMINAMATH_CALUDE_sunscreen_price_proof_l1501_150119


namespace NUMINAMATH_CALUDE_bowling_ball_weight_l1501_150154

theorem bowling_ball_weight :
  ∀ (b c : ℝ),
  (5 * b = 3 * c) →
  (2 * c = 56) →
  (b = 16.8) :=
by
  sorry

end NUMINAMATH_CALUDE_bowling_ball_weight_l1501_150154


namespace NUMINAMATH_CALUDE_square_sum_equals_4014_l1501_150112

theorem square_sum_equals_4014 (a : ℝ) (h : (2006 - a) * (2004 - a) = 2005) :
  (2006 - a)^2 + (2004 - a)^2 = 4014 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equals_4014_l1501_150112


namespace NUMINAMATH_CALUDE_semicircle_bounded_rectangle_perimeter_l1501_150141

theorem semicircle_bounded_rectangle_perimeter :
  let rectangle_length : ℝ := 4 / π
  let rectangle_width : ℝ := 1 / π
  let long_side_arcs_perimeter : ℝ := 2 * π * rectangle_length / 2
  let short_side_arcs_perimeter : ℝ := π * rectangle_width
  long_side_arcs_perimeter + short_side_arcs_perimeter = 9 := by
  sorry

end NUMINAMATH_CALUDE_semicircle_bounded_rectangle_perimeter_l1501_150141


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l1501_150128

/-- An arithmetic sequence with the given properties has the general term a_n = n. -/
theorem arithmetic_sequence_property (a : ℕ → ℝ) (d : ℝ) : 
  d ≠ 0 ∧ 
  (∀ n, a (n + 1) = a n + d) ∧ 
  a 2 ^ 2 = a 1 * a 4 ∧ 
  a 5 + a 6 = 11 → 
  ∀ n, a n = n := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l1501_150128


namespace NUMINAMATH_CALUDE_coins_can_be_all_heads_l1501_150113

/-- Represents the state of a coin (heads or tails) -/
inductive CoinState
| Heads
| Tails

/-- Represents a sequence of 100 coins -/
def CoinSequence := Fin 100 → CoinState

/-- Represents an operation that flips 7 coins at equal intervals -/
structure FlipOperation where
  start : Fin 100  -- Starting position of the flip
  interval : Nat   -- Interval between flipped coins
  valid : start.val + 6 * interval < 100  -- Ensure operation is within bounds

/-- Applies a flip operation to a coin sequence -/
def applyFlip (seq : CoinSequence) (op : FlipOperation) : CoinSequence :=
  λ i => if ∃ k : Fin 7, i.val = op.start.val + k.val * op.interval
         then match seq i with
              | CoinState.Heads => CoinState.Tails
              | CoinState.Tails => CoinState.Heads
         else seq i

/-- Checks if all coins in the sequence are heads -/
def allHeads (seq : CoinSequence) : Prop :=
  ∀ i : Fin 100, seq i = CoinState.Heads

/-- The main theorem: it's possible to make all coins heads -/
theorem coins_can_be_all_heads :
  ∀ (initial : CoinSequence),
  ∃ (ops : List FlipOperation),
  allHeads (ops.foldl applyFlip initial) :=
sorry

end NUMINAMATH_CALUDE_coins_can_be_all_heads_l1501_150113


namespace NUMINAMATH_CALUDE_incorrect_permutations_hello_l1501_150149

def word := "hello"

theorem incorrect_permutations_hello :
  let total_letters := word.length
  let duplicate_letters := 2  -- number of 'l's
  let total_permutations := Nat.factorial total_letters
  let unique_permutations := total_permutations / (Nat.factorial duplicate_letters)
  unique_permutations - 1 = 59 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_permutations_hello_l1501_150149


namespace NUMINAMATH_CALUDE_base_prime_630_l1501_150138

/-- Base prime representation of a natural number -/
def BasePrime : ℕ → List ℕ := sorry

/-- Check if a list represents a valid base prime representation -/
def IsValidBasePrime : List ℕ → Prop := sorry

theorem base_prime_630 : 
  let bp := BasePrime 630
  IsValidBasePrime bp ∧ bp = [2, 1, 1, 0] := by sorry

end NUMINAMATH_CALUDE_base_prime_630_l1501_150138


namespace NUMINAMATH_CALUDE_pet_store_spiders_l1501_150197

theorem pet_store_spiders (initial_birds initial_puppies initial_cats : ℕ)
  (initial_spiders : ℕ) (sold_birds adopted_puppies loose_spiders : ℕ)
  (total_left : ℕ) :
  initial_birds = 12 →
  initial_puppies = 9 →
  initial_cats = 5 →
  sold_birds = initial_birds / 2 →
  adopted_puppies = 3 →
  loose_spiders = 7 →
  total_left = 25 →
  total_left = (initial_birds - sold_birds) + (initial_puppies - adopted_puppies) +
               initial_cats + (initial_spiders - loose_spiders) →
  initial_spiders = 15 :=
by sorry

end NUMINAMATH_CALUDE_pet_store_spiders_l1501_150197


namespace NUMINAMATH_CALUDE_tape_length_sum_l1501_150142

/-- Given three tapes A, B, and C with the following properties:
  * The length of tape A is 35 cm
  * The length of tape A is half the length of tape B
  * The length of tape C is 21 cm less than twice the length of tape A
  Prove that the sum of the lengths of tape B and tape C is 119 cm -/
theorem tape_length_sum (length_A length_B length_C : ℝ) : 
  length_A = 35 →
  length_A = length_B / 2 →
  length_C = 2 * length_A - 21 →
  length_B + length_C = 119 := by
  sorry

end NUMINAMATH_CALUDE_tape_length_sum_l1501_150142


namespace NUMINAMATH_CALUDE_bus_students_count_l1501_150140

/-- The number of students on the left side of the bus -/
def left_students : ℕ := 36

/-- The number of students on the right side of the bus -/
def right_students : ℕ := 27

/-- The total number of students on the bus -/
def total_students : ℕ := left_students + right_students

/-- Theorem: The total number of students on the bus is 63 -/
theorem bus_students_count : total_students = 63 := by
  sorry

end NUMINAMATH_CALUDE_bus_students_count_l1501_150140


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1501_150157

theorem inequality_solution_set : 
  {x : ℝ | (x - 1) * (x + 2) < 0} = Set.Ioo (-2 : ℝ) (1 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1501_150157


namespace NUMINAMATH_CALUDE_shara_monthly_return_l1501_150162

/-- Proves that the monthly return amount is $10 given the conditions of Shara's loan repayment --/
theorem shara_monthly_return (T M : ℝ) 
  (half_returned : T / 2 = 6 * M)
  (remaining_owed : T / 2 - 4 * M = 20) : M = 10 := by
  sorry

end NUMINAMATH_CALUDE_shara_monthly_return_l1501_150162


namespace NUMINAMATH_CALUDE_fib_100_mod_8_l1501_150131

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

-- Theorem statement
theorem fib_100_mod_8 : fib 100 % 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_fib_100_mod_8_l1501_150131


namespace NUMINAMATH_CALUDE_cara_charge_account_l1501_150192

/-- Represents the simple interest calculation for Cara's charge account --/
theorem cara_charge_account (initial_charge : ℝ) : 
  initial_charge * (1 + 0.05) = 56.7 → initial_charge = 54 := by
  sorry

end NUMINAMATH_CALUDE_cara_charge_account_l1501_150192


namespace NUMINAMATH_CALUDE_product_one_when_equal_absolute_log_l1501_150191

noncomputable def f (x : ℝ) : ℝ := |Real.log x|

theorem product_one_when_equal_absolute_log 
  (a b : ℝ) (h1 : a ≠ b) (h2 : a > 0) (h3 : b > 0) (h4 : f a = f b) : 
  a * b = 1 := by
  sorry

end NUMINAMATH_CALUDE_product_one_when_equal_absolute_log_l1501_150191


namespace NUMINAMATH_CALUDE_f_at_point_four_l1501_150156

def f (x : ℝ) : ℝ := 3 * x^6 + 4 * x^5 + 5 * x^4 + 6 * x^3 + 7 * x^2 + 8 * x + 1

theorem f_at_point_four : f 0.4 = 5.885248 := by
  sorry

end NUMINAMATH_CALUDE_f_at_point_four_l1501_150156


namespace NUMINAMATH_CALUDE_expression_simplification_l1501_150110

theorem expression_simplification (a : ℝ) (h1 : a ≠ 0) (h2 : a ≠ 1) (h3 : a ≠ -1) :
  (a - (2 * a - 1) / a) / ((a^2 - 1) / a) = (a - 1) / (a + 1) :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l1501_150110


namespace NUMINAMATH_CALUDE_trajectory_of_Q_l1501_150153

-- Define the line l
def line_l (x y : ℝ) : Prop := 2 * x + 4 * y + 3 = 0

-- Define point P on line l
def point_P (x y : ℝ) : Prop := line_l x y

-- Define the origin O
def origin : ℝ × ℝ := (0, 0)

-- Define the relation between O, Q, and P
def relation_OQP (qx qy px py : ℝ) : Prop :=
  2 * (qx - 0, qy - 0) = (px - qx, py - qy)

-- Theorem statement
theorem trajectory_of_Q (qx qy : ℝ) :
  (∃ px py, point_P px py ∧ relation_OQP qx qy px py) →
  2 * qx + 4 * qy + 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_trajectory_of_Q_l1501_150153


namespace NUMINAMATH_CALUDE_dihedral_angle_lower_bound_l1501_150151

/-- Given a regular n-sided polygon inscribed in an arbitrary great circle of a sphere,
    with tangent planes laid at each vertex, the dihedral angle φ of the resulting
    polyhedral angle satisfies φ ≥ π(1 - 2/n). -/
theorem dihedral_angle_lower_bound (n : ℕ) (φ : ℝ) 
  (h1 : n ≥ 3)  -- n is at least 3 for a polygon
  (h2 : φ > 0)  -- dihedral angle is positive
  (h3 : φ < π)  -- dihedral angle is less than π
  : φ ≥ π * (1 - 2 / n) :=
sorry

end NUMINAMATH_CALUDE_dihedral_angle_lower_bound_l1501_150151


namespace NUMINAMATH_CALUDE_parallel_segment_length_l1501_150171

theorem parallel_segment_length (base : ℝ) (a b c : ℝ) :
  base = 18 →
  a + b + c = 1 →
  a = (1/4 : ℝ) →
  b = (1/2 : ℝ) →
  c = (1/4 : ℝ) →
  ∃ (middle_segment : ℝ), middle_segment = 9 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_parallel_segment_length_l1501_150171


namespace NUMINAMATH_CALUDE_abs_neg_2023_l1501_150161

theorem abs_neg_2023 : |(-2023 : ℤ)| = 2023 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_2023_l1501_150161


namespace NUMINAMATH_CALUDE_hilt_bee_count_l1501_150104

/-- The number of bees Mrs. Hilt saw on the first day -/
def first_day_bees : ℕ := 144

/-- The multiplier for the number of bees on the second day -/
def day_two_multiplier : ℕ := 3

/-- The number of bees Mrs. Hilt saw on the second day -/
def second_day_bees : ℕ := first_day_bees * day_two_multiplier

/-- Theorem stating that Mrs. Hilt saw 432 bees on the second day -/
theorem hilt_bee_count : second_day_bees = 432 := by
  sorry

end NUMINAMATH_CALUDE_hilt_bee_count_l1501_150104


namespace NUMINAMATH_CALUDE_polynomial_expansion_l1501_150121

theorem polynomial_expansion (x : ℝ) :
  (3 * x^3 - 2 * x + 4) * (4 * x^2 - 3 * x + 5) =
  12 * x^5 - 9 * x^4 + 7 * x^3 + 10 * x^2 - 2 * x + 20 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l1501_150121


namespace NUMINAMATH_CALUDE_inequality_proof_l1501_150165

theorem inequality_proof (a b c d e f : ℝ) (h : b^2 ≤ a*c) :
  (a*f - c*d)^2 ≥ (a*e - b*d)*(b*f - c*e) := by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1501_150165


namespace NUMINAMATH_CALUDE_triangle_with_angle_ratio_not_necessarily_right_l1501_150116

/-- Triangle ABC with angles in the ratio 3:4:5 is not necessarily a right triangle -/
theorem triangle_with_angle_ratio_not_necessarily_right :
  ∀ (A B C : ℝ),
  (A + B + C = 180) →
  (A : ℝ) / 3 = (B : ℝ) / 4 →
  (B : ℝ) / 4 = (C : ℝ) / 5 →
  ¬ (A = 90 ∨ B = 90 ∨ C = 90) :=
by sorry

end NUMINAMATH_CALUDE_triangle_with_angle_ratio_not_necessarily_right_l1501_150116


namespace NUMINAMATH_CALUDE_fourth_term_is_five_l1501_150160

/-- An arithmetic sequence where the sum of the third and fifth terms is 10 -/
def ArithmeticSequence (a : ℝ) (d : ℝ) : Prop :=
  a + (a + 2*d) = 10

/-- The fourth term of the arithmetic sequence -/
def FourthTerm (a : ℝ) (d : ℝ) : ℝ := a + d

theorem fourth_term_is_five {a d : ℝ} (h : ArithmeticSequence a d) : FourthTerm a d = 5 := by
  sorry

end NUMINAMATH_CALUDE_fourth_term_is_five_l1501_150160


namespace NUMINAMATH_CALUDE_salaries_sum_l1501_150189

theorem salaries_sum (A_salary B_salary : ℝ) : 
  A_salary = 5250 →
  A_salary * 0.05 = B_salary * 0.15 →
  A_salary + B_salary = 7000 :=
by
  sorry

end NUMINAMATH_CALUDE_salaries_sum_l1501_150189


namespace NUMINAMATH_CALUDE_triangle_side_length_l1501_150170

theorem triangle_side_length (perimeter side2 side3 : ℝ) 
  (h_perimeter : perimeter = 160)
  (h_side2 : side2 = 50)
  (h_side3 : side3 = 70) :
  perimeter - side2 - side3 = 40 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1501_150170


namespace NUMINAMATH_CALUDE_average_equation_l1501_150139

theorem average_equation (a : ℝ) : 
  ((2 * a + 16) + (3 * a - 8)) / 2 = 74 → a = 28 := by sorry

end NUMINAMATH_CALUDE_average_equation_l1501_150139


namespace NUMINAMATH_CALUDE_carol_initial_cupcakes_l1501_150137

/-- Given that if Carol sold 9 cupcakes and made 28 more, she would have 49 cupcakes,
    prove that Carol initially made 30 cupcakes. -/
theorem carol_initial_cupcakes : 
  ∀ (initial : ℕ), 
  (initial - 9 + 28 = 49) → 
  initial = 30 := by
sorry

end NUMINAMATH_CALUDE_carol_initial_cupcakes_l1501_150137


namespace NUMINAMATH_CALUDE_power_function_value_l1501_150120

-- Define the power function type
def PowerFunction := ℝ → ℝ

-- Define the property of passing through the point (3, √3/3)
def PassesThroughPoint (f : PowerFunction) : Prop :=
  f 3 = Real.sqrt 3 / 3

-- State the theorem
theorem power_function_value (f : PowerFunction) 
  (h : PassesThroughPoint f) : f (1/4) = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_function_value_l1501_150120


namespace NUMINAMATH_CALUDE_sequence_properties_l1501_150132

/-- Sequence a_n with given properties -/
def sequence_a (n : ℕ) : ℝ :=
  sorry

/-- Sum of first n terms of sequence a_n -/
def S (n : ℕ) : ℝ :=
  sorry

/-- Sum of first n terms of sequence a_n / 2^n -/
def T (n : ℕ) : ℝ :=
  sorry

/-- Theorem stating the properties of the sequence and its sums -/
theorem sequence_properties :
  (∀ n : ℕ, n ≥ 1 → sequence_a n / S n = 2 / (n + 1)) ∧
  sequence_a 1 = 1 →
  (∀ n : ℕ, n ≥ 1 → sequence_a n = n) ∧
  (∀ n : ℕ, n ≥ 1 → T n = 2 - (n + 2) * (1/2)^n) :=
by
  sorry

end NUMINAMATH_CALUDE_sequence_properties_l1501_150132


namespace NUMINAMATH_CALUDE_workshop_salary_problem_l1501_150167

theorem workshop_salary_problem (total_workers : Nat) (avg_salary : ℝ) 
  (num_technicians : Nat) (avg_salary_technicians : ℝ) :
  total_workers = 28 →
  avg_salary = 8000 →
  num_technicians = 7 →
  avg_salary_technicians = 14000 →
  let remaining_workers := total_workers - num_technicians
  let total_salary := total_workers * avg_salary
  let total_salary_technicians := num_technicians * avg_salary_technicians
  let total_salary_remaining := total_salary - total_salary_technicians
  let avg_salary_remaining := total_salary_remaining / remaining_workers
  avg_salary_remaining = 6000 := by
sorry

end NUMINAMATH_CALUDE_workshop_salary_problem_l1501_150167


namespace NUMINAMATH_CALUDE_sticker_distribution_l1501_150125

/-- The number of ways to partition n identical objects into k non-empty parts -/
def partition_into_nonempty_parts (n k : ℕ) : ℕ := sorry

/-- Theorem: There are 7 ways to partition 10 identical objects into 5 non-empty parts -/
theorem sticker_distribution : partition_into_nonempty_parts 10 5 = 7 := by sorry

end NUMINAMATH_CALUDE_sticker_distribution_l1501_150125


namespace NUMINAMATH_CALUDE_combined_share_proof_l1501_150172

/-- Proves that given $12,000 to be distributed among 5 children in the ratio 2 : 4 : 3 : 1 : 5,
    the combined share of the children with ratios 1 and 5 is $4,800. -/
theorem combined_share_proof (total_money : ℕ) (num_children : ℕ) (ratio : List ℕ) :
  total_money = 12000 →
  num_children = 5 →
  ratio = [2, 4, 3, 1, 5] →
  (ratio.sum * 800 = total_money) →
  (List.get! ratio 3 * 800 + List.get! ratio 4 * 800 = 4800) :=
by sorry

end NUMINAMATH_CALUDE_combined_share_proof_l1501_150172


namespace NUMINAMATH_CALUDE_unique_prime_triple_l1501_150169

theorem unique_prime_triple : ∃! (p q r : ℕ), 
  Prime p ∧ Prime q ∧ Prime r ∧
  p > q ∧ q > r ∧
  Prime (p - q) ∧ Prime (p - r) ∧ Prime (q - r) ∧
  p = 7 ∧ q = 5 ∧ r = 2 := by
sorry

end NUMINAMATH_CALUDE_unique_prime_triple_l1501_150169


namespace NUMINAMATH_CALUDE_latoya_phone_card_initial_amount_l1501_150107

/-- The initial amount paid for a prepaid phone card -/
def initial_amount (call_cost_per_minute : ℚ) (call_duration : ℕ) (remaining_credit : ℚ) : ℚ :=
  call_cost_per_minute * call_duration + remaining_credit

/-- Theorem: The initial amount paid for Latoya's phone card is $30.00 -/
theorem latoya_phone_card_initial_amount :
  initial_amount (16 / 100) 22 26.48 = 30 :=
by sorry

end NUMINAMATH_CALUDE_latoya_phone_card_initial_amount_l1501_150107
