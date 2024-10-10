import Mathlib

namespace sum_of_areas_decomposition_l501_50145

/-- Represents a 1 by 1 by 1 cube -/
structure UnitCube where
  side : ℝ
  is_unit : side = 1

/-- Represents a triangle with vertices on the cube -/
structure CubeTriangle where
  vertices : Fin 3 → Fin 8

/-- The area of a triangle on the cube -/
noncomputable def triangle_area (t : CubeTriangle) : ℝ := sorry

/-- The sum of areas of all triangles on the cube -/
noncomputable def sum_of_triangle_areas (cube : UnitCube) : ℝ := sorry

/-- The theorem to be proved -/
theorem sum_of_areas_decomposition (cube : UnitCube) :
  ∃ (m n p : ℕ), sum_of_triangle_areas cube = m + Real.sqrt n + Real.sqrt p ∧ m + n + p = 348 := by
  sorry

end sum_of_areas_decomposition_l501_50145


namespace arithmetic_vector_sequence_sum_parallel_l501_50164

/-- An arithmetic vector sequence in 2D space -/
def ArithmeticVectorSequence (a : ℕ → ℝ × ℝ) : Prop :=
  ∃ d : ℝ × ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of the first n vectors in a sequence -/
def VectorSum (a : ℕ → ℝ × ℝ) (n : ℕ) : ℝ × ℝ :=
  (List.range n).map a |>.sum

/-- Two vectors are parallel -/
def Parallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v = k • w

theorem arithmetic_vector_sequence_sum_parallel
  (a : ℕ → ℝ × ℝ) (h : ArithmeticVectorSequence a) :
  Parallel (VectorSum a 21) (a 11) := by
  sorry

end arithmetic_vector_sequence_sum_parallel_l501_50164


namespace inverse_variation_problem_l501_50122

theorem inverse_variation_problem (y x : ℝ) (k : ℝ) (h1 : y * x^2 = k) 
  (h2 : 6 * 3^2 = k) (h3 : 2 * x^2 = k) : x = 3 * Real.sqrt 3 := by
  sorry

end inverse_variation_problem_l501_50122


namespace flora_initial_daily_milk_l501_50144

def total_milk : ℕ := 105
def weeks : ℕ := 3
def days_per_week : ℕ := 7
def brother_additional : ℕ := 2

theorem flora_initial_daily_milk :
  let total_days : ℕ := weeks * days_per_week
  let flora_initial_think : ℕ := total_milk / total_days
  flora_initial_think = 5 := by sorry

end flora_initial_daily_milk_l501_50144


namespace quadratic_equation_equivalence_l501_50108

theorem quadratic_equation_equivalence :
  ∀ x : ℝ, x^2 + 4*x - 1 = 0 ↔ (x + 2)^2 = 5 := by
sorry

end quadratic_equation_equivalence_l501_50108


namespace geometric_sequence_21st_term_l501_50141

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_21st_term
  (a : ℕ → ℝ)
  (h_geometric : geometric_sequence a)
  (h_first_term : a 1 = 3)
  (h_common_product : ∀ n : ℕ, a n * a (n + 1) = 15) :
  a 21 = 3 := by
sorry

end geometric_sequence_21st_term_l501_50141


namespace books_purchased_with_grant_l501_50186

/-- The number of books purchased by Silvergrove Public Library using a grant --/
theorem books_purchased_with_grant 
  (total_books : Nat) 
  (books_before_grant : Nat) 
  (h1 : total_books = 8582)
  (h2 : books_before_grant = 5935) :
  total_books - books_before_grant = 2647 := by
  sorry

end books_purchased_with_grant_l501_50186


namespace trapezium_area_l501_50179

-- Define the trapezium properties
def a : ℝ := 10 -- Length of one parallel side
def b : ℝ := 18 -- Length of the other parallel side
def h : ℝ := 15 -- Distance between parallel sides

-- Theorem statement
theorem trapezium_area : (1/2 : ℝ) * (a + b) * h = 210 := by
  sorry

end trapezium_area_l501_50179


namespace binomial_coefficient_equality_l501_50199

theorem binomial_coefficient_equality (m : ℕ) : 
  (Nat.choose 15 m = Nat.choose 15 (m - 3)) ↔ m = 9 := by sorry

end binomial_coefficient_equality_l501_50199


namespace unique_n_with_divisor_sum_property_l501_50152

def isDivisor (d n : ℕ) : Prop := n % d = 0

theorem unique_n_with_divisor_sum_property :
  ∃! n : ℕ+, 
    (∃ (d₁ d₂ d₃ d₄ : ℕ+),
      isDivisor d₁ n ∧ isDivisor d₂ n ∧ isDivisor d₃ n ∧ isDivisor d₄ n ∧
      d₁ < d₂ ∧ d₂ < d₃ ∧ d₃ < d₄ ∧
      d₁ = 1 ∧
      n = d₁^2 + d₂^2 + d₃^2 + d₄^2) ∧
    (∀ d : ℕ+, isDivisor d n → d = 1 ∨ d ≥ d₂) ∧
    n = 130 :=
by sorry

end unique_n_with_divisor_sum_property_l501_50152


namespace equation_solution_l501_50147

theorem equation_solution : ∃ x : ℝ, 4 * x - 7 = 5 ∧ x = 3 := by
  sorry

end equation_solution_l501_50147


namespace perimeter_after_increase_l501_50115

/-- Represents a triangle with side lengths a, b, and c. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_pos : 0 < a ∧ 0 < b ∧ 0 < c
  h_ineq : a < b + c ∧ b < a + c ∧ c < a + b

/-- The perimeter of a triangle. -/
def Triangle.perimeter (t : Triangle) : ℝ := t.a + t.b + t.c

/-- Given a triangle, returns a new triangle with two sides increased by 4 and one by 1. -/
def increaseSides (t : Triangle) : Triangle where
  a := t.a + 4
  b := t.b + 4
  c := t.c + 1
  h_pos := sorry
  h_ineq := sorry

theorem perimeter_after_increase (t : Triangle) 
    (h1 : t.a = 8)
    (h2 : t.b = 5)
    (h3 : t.c = 6) :
    (increaseSides t).perimeter = 28 := by
  sorry

end perimeter_after_increase_l501_50115


namespace factory_output_growth_rate_l501_50197

theorem factory_output_growth_rate (x : ℝ) : 
  (∀ y : ℝ, y > 0 → (1 + x)^2 * y = 1.2 * y) → 
  x < 0.1 := by
  sorry

end factory_output_growth_rate_l501_50197


namespace segment_length_theorem_solvability_condition_l501_50100

/-- Two mutually tangent circles with radii r₁ and r₂ -/
structure TangentCircles where
  r₁ : ℝ
  r₂ : ℝ
  r₁_pos : r₁ > 0
  r₂_pos : r₂ > 0

/-- A line intersecting two circles in four points, creating three equal segments -/
structure IntersectingLine (tc : TangentCircles) where
  d : ℝ
  d_pos : d > 0
  intersects_circles : True  -- This is a placeholder for the intersection property

/-- The main theorem relating the segment length to the radii -/
theorem segment_length_theorem (tc : TangentCircles) (l : IntersectingLine tc) :
    l.d^2 = (1/12) * (14*tc.r₁*tc.r₂ - tc.r₁^2 - tc.r₂^2) := by sorry

/-- The solvability condition for the problem -/
theorem solvability_condition (tc : TangentCircles) :
    (∃ l : IntersectingLine tc, True) ↔ 
    (7 - 4*Real.sqrt 3 ≤ tc.r₁ / tc.r₂ ∧ tc.r₁ / tc.r₂ ≤ 7 + 4*Real.sqrt 3) := by sorry

end segment_length_theorem_solvability_condition_l501_50100


namespace expression_value_l501_50163

theorem expression_value (x y : ℤ) (hx : x = -2) (hy : y = -4) :
  5 * (x - y)^2 - x * y = 28 := by
  sorry

end expression_value_l501_50163


namespace factorial_plus_24_equals_square_l501_50106

theorem factorial_plus_24_equals_square (n m : ℕ) : n.factorial + 24 = m ^ 2 ↔ (n = 1 ∧ m = 5) ∨ (n = 5 ∧ m = 12) := by
  sorry

end factorial_plus_24_equals_square_l501_50106


namespace graph_equation_is_intersecting_lines_l501_50124

theorem graph_equation_is_intersecting_lines :
  ∀ x y : ℝ, (x + y)^2 = x^2 + y^2 + 3*x*y ↔ x*y = 0 :=
by sorry

end graph_equation_is_intersecting_lines_l501_50124


namespace rachels_age_problem_l501_50113

/-- Rachel's age problem -/
theorem rachels_age_problem (rachel_age : ℕ) (grandfather_age : ℕ) (mother_age : ℕ) (father_age : ℕ) :
  rachel_age = 12 →
  grandfather_age = 7 * rachel_age →
  father_age = mother_age + 5 →
  father_age + (25 - rachel_age) = 60 →
  mother_age / grandfather_age = 1 / 2 := by
  sorry

end rachels_age_problem_l501_50113


namespace roots_transformation_l501_50156

theorem roots_transformation (r₁ r₂ r₃ : ℂ) : 
  (r₁^3 - 4*r₁^2 + 5*r₁ + 12 = 0) ∧ 
  (r₂^3 - 4*r₂^2 + 5*r₂ + 12 = 0) ∧ 
  (r₃^3 - 4*r₃^2 + 5*r₃ + 12 = 0) →
  ((3*r₁)^3 - 12*(3*r₁)^2 + 45*(3*r₁) + 324 = 0) ∧
  ((3*r₂)^3 - 12*(3*r₂)^2 + 45*(3*r₂) + 324 = 0) ∧
  ((3*r₃)^3 - 12*(3*r₃)^2 + 45*(3*r₃) + 324 = 0) :=
by sorry

end roots_transformation_l501_50156


namespace distance_between_points_l501_50188

theorem distance_between_points (A B : ℝ) : A = 3 ∧ B = -7 → |A - B| = 10 := by
  sorry

end distance_between_points_l501_50188


namespace solve_lemonade_problem_l501_50130

def lemonade_problem (price_per_cup : ℝ) (cups_sold : ℕ) (cost_lemons : ℝ) (cost_sugar : ℝ) (total_profit : ℝ) : Prop :=
  let total_revenue := price_per_cup * (cups_sold : ℝ)
  let known_expenses := cost_lemons + cost_sugar
  let cost_cups := total_revenue - known_expenses - total_profit
  cost_cups = 3

theorem solve_lemonade_problem :
  lemonade_problem 4 21 10 5 66 := by
  sorry

end solve_lemonade_problem_l501_50130


namespace perpendicular_to_two_planes_implies_parallel_l501_50166

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- Define the parallel relation between two planes
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_to_two_planes_implies_parallel 
  (α β : Plane) (a : Line) 
  (h_diff : α ≠ β) 
  (h_perp_α : perpendicular a α) 
  (h_perp_β : perpendicular a β) : 
  parallel α β :=
sorry

end perpendicular_to_two_planes_implies_parallel_l501_50166


namespace tan_pi_36_is_root_l501_50116

theorem tan_pi_36_is_root : 
  let f (x : ℝ) := x^3 - 3 * Real.tan (π/12) * x^2 - 3 * x + Real.tan (π/12)
  f (Real.tan (π/36)) = 0 := by sorry

end tan_pi_36_is_root_l501_50116


namespace perfect_square_binomial_l501_50132

/-- 
A quadratic expression x^2 - 20x + k is a perfect square binomial 
if and only if k = 100.
-/
theorem perfect_square_binomial (k : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, x^2 - 20*x + k = (x + b)^2) ↔ k = 100 := by
  sorry

end perfect_square_binomial_l501_50132


namespace sign_sum_theorem_l501_50154

theorem sign_sum_theorem (x y z w : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (hw : w ≠ 0) :
  let sign_sum := x / |x| + y / |y| + z / |z| + w / |w| + (x * y * z * w) / |x * y * z * w|
  sign_sum = 5 ∨ sign_sum = 1 ∨ sign_sum = -1 ∨ sign_sum = -5 := by
  sorry

end sign_sum_theorem_l501_50154


namespace max_ab_value_l501_50104

theorem max_ab_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (heq : a + 4*b + a*b = 3) :
  a * b ≤ 1 := by
sorry

end max_ab_value_l501_50104


namespace opposite_of_reciprocal_l501_50169

theorem opposite_of_reciprocal : -(1 / (-1/3)) = 3 := by sorry

end opposite_of_reciprocal_l501_50169


namespace rainwater_chickens_l501_50192

/-- Proves that Mr. Rainwater has 18 chickens given the conditions -/
theorem rainwater_chickens :
  ∀ (goats cows chickens : ℕ),
    cows = 9 →
    goats = 4 * cows →
    goats = 2 * chickens →
    chickens = 18 := by
  sorry

end rainwater_chickens_l501_50192


namespace impossibleToGather_l501_50191

/-- Represents the number of islands and ships -/
def n : ℕ := 1002

/-- Represents the position of a ship on the circular archipelago -/
def Position := Fin n

/-- Represents the fleet of ships -/
def Fleet := Multiset Position

/-- Represents a single day's movement of two ships -/
def Move := Position × Position × Position × Position

/-- Checks if all ships are gathered on a single island -/
def allGathered (fleet : Fleet) : Prop :=
  ∃ p : Position, fleet = Multiset.replicate n p

/-- Applies a move to the fleet -/
def applyMove (fleet : Fleet) (move : Move) : Fleet :=
  sorry

/-- The main theorem stating that it's impossible to gather all ships -/
theorem impossibleToGather (initialFleet : Fleet) :
  ¬∃ (moves : List Move), allGathered (moves.foldl applyMove initialFleet) :=
sorry

end impossibleToGather_l501_50191


namespace undefined_expression_l501_50182

theorem undefined_expression (x : ℝ) : 
  (x - 1) / (x^2 - 5*x + 6) = 0⁻¹ ↔ x = 2 ∨ x = 3 := by
  sorry

end undefined_expression_l501_50182


namespace max_area_isosceles_triangle_l501_50126

/-- A triangle with vertices A, B, and C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The angle at vertex A of a triangle -/
def angle_at_A (t : Triangle) : ℝ := sorry

/-- The semiperimeter of a triangle -/
def semiperimeter (t : Triangle) : ℝ := sorry

/-- The area of a triangle -/
def area (t : Triangle) : ℝ := sorry

/-- Predicate to check if a triangle is isosceles with base BC -/
def is_isosceles_BC (t : Triangle) : Prop := sorry

/-- Theorem: Among all triangles with fixed angle α at A and fixed semiperimeter p,
    the isosceles triangle with base BC has the largest area -/
theorem max_area_isosceles_triangle (α p : ℝ) :
  ∀ t : Triangle,
    angle_at_A t = α →
    semiperimeter t = p →
    ∀ t' : Triangle,
      angle_at_A t' = α →
      semiperimeter t' = p →
      is_isosceles_BC t' →
      area t ≤ area t' :=
sorry

end max_area_isosceles_triangle_l501_50126


namespace polynomial_divisibility_l501_50158

theorem polynomial_divisibility (a b c d : ℤ) :
  (∀ x : ℤ, ∃ k : ℤ, a * x^3 + b * x^2 + c * x + d = 5 * k) →
  (∃ ka kb kc kd : ℤ, a = 5 * ka ∧ b = 5 * kb ∧ c = 5 * kc ∧ d = 5 * kd) :=
by sorry

end polynomial_divisibility_l501_50158


namespace female_employees_at_least_60_l501_50129

/-- Represents the number of employees in different categories -/
structure EmployeeCount where
  total : Nat
  advancedDegree : Nat
  collegeDegreeOnly : Nat
  maleCollegeDegreeOnly : Nat

/-- Theorem stating that the number of female employees is at least 60 -/
theorem female_employees_at_least_60 (e : EmployeeCount)
  (h1 : e.total = 200)
  (h2 : e.advancedDegree = 100)
  (h3 : e.collegeDegreeOnly = 100)
  (h4 : e.maleCollegeDegreeOnly = 40) :
  ∃ (femaleCount : Nat), femaleCount ≥ 60 ∧ femaleCount ≤ e.total :=
by sorry

end female_employees_at_least_60_l501_50129


namespace second_question_correct_percentage_l501_50194

-- Define the percentages as real numbers between 0 and 100
def first_correct : ℝ := 80
def neither_correct : ℝ := 5
def both_correct : ℝ := 60

-- Define the function to calculate the percentage who answered the second question correctly
def second_correct : ℝ := 100 - neither_correct - first_correct + both_correct

-- Theorem statement
theorem second_question_correct_percentage :
  second_correct = 75 :=
sorry

end second_question_correct_percentage_l501_50194


namespace count_integers_with_same_remainder_l501_50118

theorem count_integers_with_same_remainder : ∃! (S : Finset ℕ),
  (∀ n ∈ S, 150 < n ∧ n < 250 ∧ ∃ r : ℕ, r ≤ 6 ∧ n % 7 = r ∧ n % 9 = r) ∧
  S.card = 7 := by sorry

end count_integers_with_same_remainder_l501_50118


namespace total_weight_equals_sum_l501_50150

/-- The weight of the blue ball in pounds -/
def blue_ball_weight : ℝ := 6

/-- The weight of the brown ball in pounds -/
def brown_ball_weight : ℝ := 3.12

/-- The total weight of both balls in pounds -/
def total_weight : ℝ := blue_ball_weight + brown_ball_weight

/-- Theorem: The total weight is equal to the sum of individual weights -/
theorem total_weight_equals_sum : total_weight = 9.12 := by
  sorry

end total_weight_equals_sum_l501_50150


namespace hotel_room_charge_comparison_l501_50131

theorem hotel_room_charge_comparison (P R G : ℝ) 
  (h1 : P = R - 0.2 * R) 
  (h2 : P = G - 0.1 * G) : 
  R = G * (1 + 0.125) := by
  sorry

end hotel_room_charge_comparison_l501_50131


namespace product_of_four_integers_l501_50138

theorem product_of_four_integers (A B C D : ℕ+) 
  (sum_eq : A + B + C + D = 100)
  (relation : A + 4 = B + 4 ∧ B + 4 = C + 4 ∧ C + 4 = D * 2) : 
  A * B * C * D = 351232 := by
  sorry

end product_of_four_integers_l501_50138


namespace min_value_theorem_l501_50137

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1 / (x + 3) + 1 / (y + 3) = 1 / 4) :
  x + 3 * y ≥ 4 + 8 * Real.sqrt 3 ∧ 
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 
    1 / (x₀ + 3) + 1 / (y₀ + 3) = 1 / 4 ∧
    x₀ + 3 * y₀ = 4 + 8 * Real.sqrt 3 := by
  sorry

end min_value_theorem_l501_50137


namespace legs_in_pool_l501_50136

/-- The number of people in Karen and Donald's family -/
def karen_donald_family : ℕ := 8

/-- The number of people in Tom and Eva's family -/
def tom_eva_family : ℕ := 6

/-- The total number of people in both families -/
def total_people : ℕ := karen_donald_family + tom_eva_family

/-- The number of people not in the pool -/
def people_not_in_pool : ℕ := 6

/-- The number of legs per person -/
def legs_per_person : ℕ := 2

theorem legs_in_pool : 
  (total_people - people_not_in_pool) * legs_per_person = 16 := by
  sorry

end legs_in_pool_l501_50136


namespace polynomial_value_given_condition_l501_50157

theorem polynomial_value_given_condition (x : ℝ) : 
  3 * x^3 - x = 1 → 9 * x^4 + 12 * x^3 - 3 * x^2 - 7 * x + 2001 = 2001 := by
  sorry

end polynomial_value_given_condition_l501_50157


namespace intersection_implies_sum_l501_50111

/-- Given two functions f and g that intersect at (2,5) and (8,3), prove that a + c = 10 -/
theorem intersection_implies_sum (a b c d : ℝ) : 
  (∀ x, -|x - a| + b = |x - c| + d → x = 2 ∨ x = 8) →
  -|2 - a| + b = 5 →
  -|8 - a| + b = 3 →
  |2 - c| + d = 5 →
  |8 - c| + d = 3 →
  a + c = 10 := by
sorry

end intersection_implies_sum_l501_50111


namespace line_vector_to_slope_intercept_l501_50128

/-- Given a line in vector form, prove it's equivalent to the slope-intercept form --/
theorem line_vector_to_slope_intercept :
  ∀ (x y : ℝ), 
  (-3 : ℝ) * (x - 3) + (-7 : ℝ) * (y - 14) = 0 ↔ 
  y = (-3/7 : ℝ) * x + 107/7 := by
  sorry

end line_vector_to_slope_intercept_l501_50128


namespace first_tree_groups_count_l501_50140

/-- Represents the number of years in one ring group -/
def years_per_group : ℕ := 6

/-- Represents the number of ring groups in the second tree -/
def second_tree_groups : ℕ := 40

/-- Represents the age difference between the first and second tree in years -/
def age_difference : ℕ := 180

/-- Calculates the number of ring groups in the first tree -/
def first_tree_groups : ℕ := 
  (second_tree_groups * years_per_group + age_difference) / years_per_group

theorem first_tree_groups_count : first_tree_groups = 70 := by
  sorry

end first_tree_groups_count_l501_50140


namespace sqrt_two_sum_l501_50127

theorem sqrt_two_sum : 2 * Real.sqrt 2 + 3 * Real.sqrt 2 = 5 * Real.sqrt 2 := by
  sorry

end sqrt_two_sum_l501_50127


namespace profit_at_8750_max_profit_price_l501_50195

-- Define constants
def cost_price : ℝ := 40
def initial_selling_price : ℝ := 50
def initial_monthly_sales : ℝ := 500
def price_increase_step : ℝ := 1
def sales_decrease_step : ℝ := 10

-- Define functions
def selling_price (x : ℝ) : ℝ := initial_selling_price + x
def monthly_sales (x : ℝ) : ℝ := initial_monthly_sales - sales_decrease_step * x
def monthly_profit (x : ℝ) : ℝ := (monthly_sales x) * (selling_price x - cost_price)

-- Theorem statements
theorem profit_at_8750 (x : ℝ) : 
  monthly_profit x = 8750 → (x = 25 ∨ x = 15) := by sorry

theorem max_profit_price : 
  ∃ x : ℝ, ∀ y : ℝ, monthly_profit x ≥ monthly_profit y ∧ selling_price x = 70 := by sorry

end profit_at_8750_max_profit_price_l501_50195


namespace flowerbed_fraction_is_five_thirty_sixths_l501_50171

/-- Represents the dimensions and properties of a rectangular yard with flower beds. -/
structure YardWithFlowerBeds where
  length : ℝ
  width : ℝ
  trapezoid_side1 : ℝ
  trapezoid_side2 : ℝ
  
/-- Calculates the fraction of the yard occupied by flower beds. -/
def flowerbed_fraction (yard : YardWithFlowerBeds) : ℚ :=
  sorry

/-- Theorem stating that the fraction of the yard occupied by flower beds is 5/36. -/
theorem flowerbed_fraction_is_five_thirty_sixths 
  (yard : YardWithFlowerBeds) 
  (h1 : yard.length = 30)
  (h2 : yard.width = 6)
  (h3 : yard.trapezoid_side1 = 20)
  (h4 : yard.trapezoid_side2 = 30) :
  flowerbed_fraction yard = 5 / 36 :=
sorry

end flowerbed_fraction_is_five_thirty_sixths_l501_50171


namespace percent_relationship_l501_50184

theorem percent_relationship (a b : ℝ) (h : a = 1.25 * b) : 4 * b = 3.2 * a := by
  sorry

end percent_relationship_l501_50184


namespace parallel_lines_in_special_triangle_l501_50177

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a line in 2D space -/
structure Line :=
  (a : Point)
  (b : Point)

/-- Checks if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop := sorry

/-- Constructs an equilateral triangle given three points -/
def equilateral_triangle (a b c : Point) : Prop := sorry

theorem parallel_lines_in_special_triangle 
  (A B C M K : Point) 
  (h1 : equilateral_triangle A B C)
  (h2 : M.x ≥ A.x ∧ M.x ≤ B.x ∧ M.y = A.y)  -- M is on side AB
  (h3 : equilateral_triangle M K C) :
  parallel (Line.mk A C) (Line.mk B K) := by
  sorry

end parallel_lines_in_special_triangle_l501_50177


namespace symmetric_through_swaps_l501_50168

/-- A binary digit (0 or 1) -/
inductive BinaryDigit : Type
| zero : BinaryDigit
| one : BinaryDigit

/-- A sequence of binary digits -/
def BinarySequence := List BinaryDigit

/-- Swap operation that exchanges two elements in a list at given indices -/
def swap (seq : BinarySequence) (i j : Nat) : BinarySequence :=
  sorry

/-- Check if a sequence is symmetric -/
def isSymmetric (seq : BinarySequence) : Prop :=
  sorry

/-- The main theorem stating that any binary sequence of length 1999 can be made symmetric through swaps -/
theorem symmetric_through_swaps (seq : BinarySequence) (h : seq.length = 1999) :
  ∃ (swapSequence : List (Nat × Nat)), 
    isSymmetric (swapSequence.foldl (λ s (i, j) => swap s i j) seq) :=
  sorry

end symmetric_through_swaps_l501_50168


namespace train_length_l501_50103

/-- Given a train that crosses three platforms with different lengths and times,
    this theorem proves that the length of the train is 30 meters. -/
theorem train_length (platform1_length platform2_length platform3_length : ℝ)
                     (platform1_time platform2_time platform3_time : ℝ)
                     (h1 : platform1_length = 180)
                     (h2 : platform2_length = 250)
                     (h3 : platform3_length = 320)
                     (h4 : platform1_time = 15)
                     (h5 : platform2_time = 20)
                     (h6 : platform3_time = 25) :
  ∃ (train_length : ℝ), 
    train_length = 30 ∧ 
    (train_length + platform1_length) / platform1_time = 
    (train_length + platform2_length) / platform2_time ∧
    (train_length + platform2_length) / platform2_time = 
    (train_length + platform3_length) / platform3_time :=
by sorry

end train_length_l501_50103


namespace parent_gift_cost_is_30_l501_50189

/-- The amount spent on each sibling's gift -/
def sibling_gift_cost : ℕ := 30

/-- The number of siblings -/
def num_siblings : ℕ := 3

/-- The total amount spent on all gifts -/
def total_spent : ℕ := 150

/-- The amount spent on each parent's gift -/
def parent_gift_cost : ℕ := (total_spent - sibling_gift_cost * num_siblings) / 2

/-- Theorem stating that the amount spent on each parent's gift is $30 -/
theorem parent_gift_cost_is_30 : parent_gift_cost = 30 := by
  sorry

end parent_gift_cost_is_30_l501_50189


namespace original_triangle_area_l501_50151

/-- Given a triangle whose dimensions are quadrupled to form a new triangle with an area of 256 square feet, 
    the area of the original triangle is 16 square feet. -/
theorem original_triangle_area (original : ℝ) (new : ℝ) : 
  new = 4 * original →  -- The dimensions are quadrupled
  new^2 = 256 →         -- The area of the new triangle is 256 square feet
  original^2 = 16 :=    -- The area of the original triangle is 16 square feet
by sorry

end original_triangle_area_l501_50151


namespace right_triangle_leg_divisible_by_three_l501_50117

theorem right_triangle_leg_divisible_by_three (a b c : ℕ) (h : a * a + b * b = c * c) :
  3 ∣ a ∨ 3 ∣ b :=
sorry

end right_triangle_leg_divisible_by_three_l501_50117


namespace inequality_proof_l501_50120

theorem inequality_proof (x : ℝ) : 
  x ∈ Set.Icc (1/4 : ℝ) 3 → x ≠ 2 → x ≠ 0 → (x - 1) / (x - 2) + (x + 3) / (3 * x) ≥ 4 := by
  sorry

end inequality_proof_l501_50120


namespace complex_number_magnitude_l501_50181

theorem complex_number_magnitude (a : ℝ) (z : ℂ) : 
  z = (1 + a * Complex.I) / Complex.I → 
  z.re = z.im →
  Complex.abs z = Real.sqrt 2 := by
sorry

end complex_number_magnitude_l501_50181


namespace M_mod_1000_l501_50175

/-- The number of 8-digit positive integers with strictly increasing digits -/
def M : ℕ := Nat.choose 9 8

/-- Theorem stating that M modulo 1000 equals 9 -/
theorem M_mod_1000 : M % 1000 = 9 := by
  sorry

end M_mod_1000_l501_50175


namespace smallest_sum_abc_l501_50178

theorem smallest_sum_abc (a b c : ℕ+) (h : (3 : ℕ) * a.val = (4 : ℕ) * b.val ∧ (4 : ℕ) * b.val = (7 : ℕ) * c.val) : 
  (a.val + b.val + c.val : ℕ) ≥ 61 ∧ ∃ (a' b' c' : ℕ+), (3 : ℕ) * a'.val = (4 : ℕ) * b'.val ∧ (4 : ℕ) * b'.val = (7 : ℕ) * c'.val ∧ a'.val + b'.val + c'.val = 61 :=
by
  sorry

#check smallest_sum_abc

end smallest_sum_abc_l501_50178


namespace min_value_theorem_l501_50119

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * y + 2 * x + 3 * y = 42) :
  x * y + 5 * x + 4 * y ≥ 55 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ * y₀ + 2 * x₀ + 3 * y₀ = 42 ∧ x₀ * y₀ + 5 * x₀ + 4 * y₀ = 55 :=
by sorry

end min_value_theorem_l501_50119


namespace grid_path_count_l501_50148

/-- The number of paths on a grid from (0,0) to (m,n) using only right and up moves -/
def gridPaths (m n : ℕ) : ℕ := Nat.choose (m + n) n

/-- The dimensions of our grid -/
def gridWidth : ℕ := 6
def gridHeight : ℕ := 5

/-- The total number of steps required -/
def totalSteps : ℕ := gridWidth + gridHeight

theorem grid_path_count :
  gridPaths gridWidth gridHeight = 462 := by
  sorry

#eval gridPaths gridWidth gridHeight

end grid_path_count_l501_50148


namespace continuous_injective_on_irrationals_implies_injective_monotonic_l501_50193

/-- A function is injective on irrational numbers -/
def InjectiveOnIrrationals (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, Irrational x → Irrational y → x ≠ y → f x ≠ f y

/-- A function is strictly monotonic -/
def StrictlyMonotonic (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y ∨ ∀ x y : ℝ, x < y → f x > f y

theorem continuous_injective_on_irrationals_implies_injective_monotonic
  (f : ℝ → ℝ) (hf_cont : Continuous f) (hf_inj_irr : InjectiveOnIrrationals f) :
  Function.Injective f ∧ StrictlyMonotonic f :=
sorry

end continuous_injective_on_irrationals_implies_injective_monotonic_l501_50193


namespace largest_k_ratio_l501_50133

theorem largest_k_ratio (a b c d : ℕ+) (h1 : a + b = c + d) (h2 : 2 * a * b = c * d) (h3 : a ≥ b) :
  (∀ k : ℝ, (a : ℝ) / (b : ℝ) ≥ k → k ≤ 3 + 2 * Real.sqrt 2) ∧
  (a : ℝ) / (b : ℝ) ≥ 3 + 2 * Real.sqrt 2 :=
sorry

end largest_k_ratio_l501_50133


namespace average_weight_proof_l501_50121

theorem average_weight_proof (total_boys : Nat) (group1_boys : Nat) (group2_boys : Nat)
  (group2_avg_weight : ℝ) (total_avg_weight : ℝ) (group1_avg_weight : ℝ) :
  total_boys = 30 →
  group1_boys = 22 →
  group2_boys = 8 →
  group2_avg_weight = 45.15 →
  total_avg_weight = 48.89 →
  group1_avg_weight = 50.25 →
  (group1_boys : ℝ) * group1_avg_weight + (group2_boys : ℝ) * group2_avg_weight =
    (total_boys : ℝ) * total_avg_weight :=
by sorry

end average_weight_proof_l501_50121


namespace megan_pop_albums_l501_50109

def country_albums : ℕ := 2
def songs_per_album : ℕ := 7
def total_songs : ℕ := 70

def pop_albums : ℕ := (total_songs - country_albums * songs_per_album) / songs_per_album

theorem megan_pop_albums : pop_albums = 8 := by
  sorry

end megan_pop_albums_l501_50109


namespace simplify_expression_l501_50107

theorem simplify_expression (y : ℝ) :
  3 * y - 7 * y^2 + 4 - (5 - 3 * y + 7 * y^2) = -14 * y^2 + 6 * y - 1 := by
  sorry

end simplify_expression_l501_50107


namespace smallest_solution_abs_quadratic_l501_50187

theorem smallest_solution_abs_quadratic (x : ℝ) :
  (|2 * x^2 + 3 * x - 1| = 33) →
  x ≥ ((-3 - Real.sqrt 281) / 4) ∧
  (|2 * (((-3 - Real.sqrt 281) / 4)^2) + 3 * ((-3 - Real.sqrt 281) / 4) - 1| = 33) :=
by sorry

end smallest_solution_abs_quadratic_l501_50187


namespace max_missed_problems_l501_50176

theorem max_missed_problems (total_problems : ℕ) (passing_percentage : ℚ) : 
  total_problems = 50 → 
  passing_percentage = 75/100 → 
  ∃ (max_missed : ℕ), max_missed = 12 ∧ 
    (∀ (missed : ℕ), missed ≤ max_missed → 
      (total_problems - missed) / total_problems ≥ passing_percentage) ∧
    (∀ (missed : ℕ), missed > max_missed → 
      (total_problems - missed) / total_problems < passing_percentage) :=
by sorry

end max_missed_problems_l501_50176


namespace incircle_tangent_smaller_triangle_perimeter_l501_50146

/-- Given a triangle with sides a, b, c and an inscribed incircle, 
    the perimeter of the smaller triangle formed by a tangent to the incircle 
    intersecting the two longer sides is equal to 2 * (semiperimeter - shortest_side) -/
theorem incircle_tangent_smaller_triangle_perimeter 
  (a b c : ℝ) 
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_sides : a = 6 ∧ b = 10 ∧ c = 12) : 
  let p := (a + b + c) / 2
  2 * (p - min a (min b c)) = 28 := by sorry

end incircle_tangent_smaller_triangle_perimeter_l501_50146


namespace base6_product_132_14_l501_50114

/-- Converts a base 6 number to base 10 --/
def base6ToBase10 (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

/-- Converts a base 10 number to base 6 --/
def base10ToBase6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc
      else aux (m / 6) ((m % 6) :: acc)
    aux n []

/-- Multiplies two base 6 numbers --/
def multiplyBase6 (a b : List Nat) : List Nat :=
  base10ToBase6 (base6ToBase10 a * base6ToBase10 b)

theorem base6_product_132_14 :
  multiplyBase6 [2, 3, 1] [4, 1] = [2, 3, 3, 2] := by sorry

end base6_product_132_14_l501_50114


namespace rectangle_diagonal_l501_50167

/-- The diagonal of a rectangle with width 16 and length 12 is 20. -/
theorem rectangle_diagonal : ∃ (d : ℝ), d = 20 ∧ d^2 = 16^2 + 12^2 := by
  sorry

end rectangle_diagonal_l501_50167


namespace simultaneous_equations_solution_l501_50174

theorem simultaneous_equations_solution (m : ℝ) : 
  ∃ (x y : ℝ), y = 3 * m * x + 2 ∧ y = (3 * m - 2) * x + 5 := by
  sorry

end simultaneous_equations_solution_l501_50174


namespace village_population_l501_50162

theorem village_population (population : ℕ) : 
  (96 : ℚ) / 100 * population = 23040 → population = 24000 := by
  sorry

end village_population_l501_50162


namespace average_age_campo_verde_l501_50159

/-- Proves that the average age of a population is 40 years, given the specified conditions -/
theorem average_age_campo_verde (H : ℕ) (h_positive : H > 0) : 
  let M := (3 / 2 : ℚ) * H
  let total_population := H + M
  let men_age_sum := 37 * H
  let women_age_sum := 42 * M
  let total_age_sum := men_age_sum + women_age_sum
  (total_age_sum / total_population : ℚ) = 40 := by
sorry


end average_age_campo_verde_l501_50159


namespace larans_weekly_profit_l501_50105

/-- Represents Laran's poster business --/
structure PosterBusiness where
  total_posters_per_day : ℕ
  large_posters_per_day : ℕ
  large_poster_price : ℕ
  large_poster_cost : ℕ
  small_poster_price : ℕ
  small_poster_cost : ℕ

/-- Calculates the weekly profit for the poster business --/
def weekly_profit (business : PosterBusiness) : ℕ :=
  let small_posters_per_day := business.total_posters_per_day - business.large_posters_per_day
  let large_poster_profit := business.large_poster_price - business.large_poster_cost
  let small_poster_profit := business.small_poster_price - business.small_poster_cost
  let daily_profit := business.large_posters_per_day * large_poster_profit + small_posters_per_day * small_poster_profit
  5 * daily_profit

/-- Laran's poster business setup --/
def larans_business : PosterBusiness :=
  { total_posters_per_day := 5
  , large_posters_per_day := 2
  , large_poster_price := 10
  , large_poster_cost := 5
  , small_poster_price := 6
  , small_poster_cost := 3 }

/-- Theorem stating that Laran's weekly profit is $95 --/
theorem larans_weekly_profit :
  weekly_profit larans_business = 95 := by
  sorry


end larans_weekly_profit_l501_50105


namespace unique_solutions_l501_50172

def is_solution (m n p : ℕ) : Prop :=
  p.Prime ∧ m > 0 ∧ n > 0 ∧ p^n + 3600 = m^2

theorem unique_solutions :
  ∀ m n p : ℕ,
    is_solution m n p ↔
      (m = 61 ∧ n = 2 ∧ p = 11) ∨
      (m = 65 ∧ n = 4 ∧ p = 5) ∨
      (m = 68 ∧ n = 10 ∧ p = 2) :=
by sorry

end unique_solutions_l501_50172


namespace multiples_of_2_are_even_is_universal_l501_50160

/-- A predicate representing a property of natural numbers -/
def P (n : ℕ) : Prop := Even n

/-- Definition of a universal proposition -/
def UniversalProposition (P : α → Prop) : Prop :=
  ∀ x, P x

/-- The statement "All multiples of 2 are even" -/
def AllMultiplesOf2AreEven : Prop :=
  ∀ n : ℕ, 2 ∣ n → Even n

/-- Theorem stating that "All multiples of 2 are even" is a universal proposition -/
theorem multiples_of_2_are_even_is_universal :
  UniversalProposition (λ n => 2 ∣ n → Even n) :=
sorry

end multiples_of_2_are_even_is_universal_l501_50160


namespace parallelogram_side_length_l501_50190

/-- 
Given a parallelogram with adjacent sides of lengths s and 2s forming a 60-degree angle,
if the area is 12√3 square units, then s = √6.
-/
theorem parallelogram_side_length (s : ℝ) : 
  s > 0 →  -- Assume s is positive
  (2 * s * s * Real.sqrt 3 = 12 * Real.sqrt 3) →  -- Area formula
  s = Real.sqrt 6 := by
  sorry

end parallelogram_side_length_l501_50190


namespace hexagon_triangle_perimeter_ratio_l501_50149

theorem hexagon_triangle_perimeter_ratio :
  ∀ (s_h s_t : ℝ),
  s_h > 0 → s_t > 0 →
  (s_t^2 * Real.sqrt 3) / 4 = 2 * ((3 * s_h^2 * Real.sqrt 3) / 2) →
  (3 * s_t) / (6 * s_h) = Real.sqrt 3 := by
sorry

end hexagon_triangle_perimeter_ratio_l501_50149


namespace arithmetic_progression_x_value_l501_50170

/-- An arithmetic progression with first three terms 2x - 3, 3x - 1, and 5x + 1 has x = 0 --/
theorem arithmetic_progression_x_value (x : ℝ) : 
  let a₁ : ℝ := 2*x - 3
  let a₂ : ℝ := 3*x - 1
  let a₃ : ℝ := 5*x + 1
  (a₂ - a₁ = a₃ - a₂) → x = 0 := by
  sorry

end arithmetic_progression_x_value_l501_50170


namespace problem_1_problem_2_l501_50101

-- Define the custom operation ⊛
def circledAst (a b : ℕ) : ℕ := sorry

-- Properties of ⊛
axiom circledAst_self (a : ℕ) : circledAst a a = a
axiom circledAst_zero (a : ℕ) : circledAst a 0 = 2 * a
axiom circledAst_add (a b c d : ℕ) : 
  (circledAst a b) + (circledAst c d) = circledAst (a + c) (b + d)

-- Theorems to prove
theorem problem_1 : circledAst (2 + 3) (0 + 3) = 7 := by sorry

theorem problem_2 : circledAst 1024 48 = 2000 := by sorry

end problem_1_problem_2_l501_50101


namespace price_of_33kg_apples_l501_50139

/-- The price of apples for a given weight, where the first 30 kg have a different price than additional kg. -/
def applePrice (l q : ℚ) (weight : ℚ) : ℚ :=
  if weight ≤ 30 then l * weight
  else l * 30 + q * (weight - 30)

/-- Theorem stating the price of 33 kg of apples -/
theorem price_of_33kg_apples (l q : ℚ) :
  (applePrice l q 15 = 150) →
  (applePrice l q 36 = 366) →
  (applePrice l q 33 = 333) := by
  sorry

end price_of_33kg_apples_l501_50139


namespace boat_speed_l501_50134

/-- The speed of a boat in still water, given its downstream and upstream speeds -/
theorem boat_speed (downstream upstream : ℝ) (h1 : downstream = 11) (h2 : upstream = 5) :
  ∃ (still_speed stream_speed : ℝ),
    still_speed + stream_speed = downstream ∧
    still_speed - stream_speed = upstream ∧
    still_speed = 8 := by
  sorry

end boat_speed_l501_50134


namespace quadratic_root_sum_l501_50142

theorem quadratic_root_sum (a b : ℤ) : 
  (∃ x : ℝ, x^2 + a*x + b = 0 ∧ x = Real.sqrt (7 - 4 * Real.sqrt 3)) →
  a + b = -3 :=
by sorry

end quadratic_root_sum_l501_50142


namespace x_squared_plus_nine_x_over_x_minus_three_squared_equals_90_l501_50123

theorem x_squared_plus_nine_x_over_x_minus_three_squared_equals_90 (x : ℝ) :
  x^2 + 9 * (x / (x - 3))^2 = 90 →
  ((x - 3)^2 * (x + 4)) / (3 * x - 4) = 36 / 11 ∨
  ((x - 3)^2 * (x + 4)) / (3 * x - 4) = 468 / 23 :=
by sorry

end x_squared_plus_nine_x_over_x_minus_three_squared_equals_90_l501_50123


namespace sum_reciprocals_l501_50143

/-- Given two positive integers m and n with sum 60, HCF 6, and LCM 210, prove that 1/m + 1/n = 1/21 -/
theorem sum_reciprocals (m n : ℕ+) 
  (h_sum : m + n = 60)
  (h_hcf : Nat.gcd m.val n.val = 6)
  (h_lcm : Nat.lcm m.val n.val = 210) : 
  1 / (m : ℚ) + 1 / (n : ℚ) = 1 / 21 := by
  sorry

end sum_reciprocals_l501_50143


namespace water_source_distance_l501_50173

-- Define the actual distance to the water source
def d : ℝ := sorry

-- Alice's statement is false
axiom alice_false : ¬(d ≥ 8)

-- Bob's statement is false
axiom bob_false : ¬(d ≤ 6)

-- Charlie's statement is false
axiom charlie_false : ¬(d = 7)

-- Theorem to prove
theorem water_source_distance :
  d ∈ Set.union (Set.Ioo 6 7) (Set.Ioo 7 8) :=
sorry

end water_source_distance_l501_50173


namespace existence_of_roots_part_a_non_existence_of_roots_part_b_l501_50180

-- Part a
theorem existence_of_roots_part_a : ∃ (a b : ℤ),
  (∀ x : ℝ, x^2 + a*x + b ≠ 0) ∧
  (∃ x : ℝ, ⌊x^2⌋ + a*x + b = 0) :=
sorry

-- Part b
theorem non_existence_of_roots_part_b : ¬∃ (a b : ℤ),
  (∀ x : ℝ, x^2 + 2*a*x + b ≠ 0) ∧
  (∃ x : ℝ, ⌊x^2⌋ + 2*a*x + b = 0) :=
sorry

end existence_of_roots_part_a_non_existence_of_roots_part_b_l501_50180


namespace polygon_angle_sum_l501_50135

theorem polygon_angle_sum (n : ℕ) : 
  (n ≥ 3) → 
  ((n - 2) * 180 + (180 - 180 / n)) = 2007 → 
  n = 13 := by
sorry

end polygon_angle_sum_l501_50135


namespace school_sections_l501_50102

theorem school_sections (boys girls : ℕ) (h1 : boys = 408) (h2 : girls = 288) : 
  (boys / (Nat.gcd boys girls)) + (girls / (Nat.gcd boys girls)) = 29 := by
  sorry

end school_sections_l501_50102


namespace summer_program_undergrads_l501_50153

theorem summer_program_undergrads (total_students : ℕ) 
  (coding_team_ugrad_percent : ℚ) (coding_team_grad_percent : ℚ) :
  total_students = 36 →
  coding_team_ugrad_percent = 1/5 →
  coding_team_grad_percent = 1/4 →
  ∃ (undergrads grads coding_team_size : ℕ),
    undergrads + grads = total_students ∧
    coding_team_size * 2 = coding_team_ugrad_percent * undergrads + coding_team_grad_percent * grads ∧
    undergrads = 20 := by
  sorry

end summer_program_undergrads_l501_50153


namespace unknown_number_proof_l501_50165

theorem unknown_number_proof (x : ℝ) : 1.75 * x = 63 → x = 36 := by
  sorry

end unknown_number_proof_l501_50165


namespace race_to_top_floor_l501_50110

/-- Represents the time taken by a person to reach the top floor of a building -/
def TimeTaken (stories : ℕ) (timePerStory : ℕ) (stopTime : ℕ) (stopsPerStory : ℕ) : ℕ :=
  stories * timePerStory + (stories - 2) * stopTime * stopsPerStory

/-- The maximum time taken between two people to reach the top floor -/
def MaxTimeTaken (time1 : ℕ) (time2 : ℕ) : ℕ :=
  max time1 time2

theorem race_to_top_floor :
  let stories := 20
  let lolaTimePerStory := 10
  let elevatorTimePerStory := 8
  let elevatorStopTime := 3
  let elevatorStopsPerStory := 1
  let lolaTime := TimeTaken stories lolaTimePerStory 0 0
  let taraTime := TimeTaken stories elevatorTimePerStory elevatorStopTime elevatorStopsPerStory
  MaxTimeTaken lolaTime taraTime = 214 :=
by sorry


end race_to_top_floor_l501_50110


namespace anika_pencils_excess_l501_50183

theorem anika_pencils_excess (reeta_pencils : ℕ) (total_pencils : ℕ) (anika_pencils : ℕ) : 
  reeta_pencils = 20 →
  anika_pencils + reeta_pencils = total_pencils →
  total_pencils = 64 →
  ∃ m : ℕ, anika_pencils = 2 * reeta_pencils + m →
  m = 4 := by
sorry

end anika_pencils_excess_l501_50183


namespace parallelogram_bisector_slope_l501_50161

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line passing through the origin -/
structure Line where
  slope : ℝ

/-- Checks if a line passes through a given point -/
def Line.passesThrough (l : Line) (p : Point) : Prop :=
  p.y = l.slope * p.x

/-- Checks if a line cuts a parallelogram into two congruent polygons -/
def cutsParallelogramCongruently (l : Line) (p1 p2 p3 p4 : Point) : Prop :=
  sorry -- Definition of this property

/-- The main theorem -/
theorem parallelogram_bisector_slope :
  ∀ (l : Line),
    let p1 : Point := ⟨12, 50⟩
    let p2 : Point := ⟨12, 120⟩
    let p3 : Point := ⟨30, 160⟩
    let p4 : Point := ⟨30, 90⟩
    l.passesThrough p1 ∧
    l.passesThrough p2 ∧
    l.passesThrough p3 ∧
    l.passesThrough p4 ∧
    cutsParallelogramCongruently l p1 p2 p3 p4 →
    l.slope = 5 :=
by
  sorry

#check parallelogram_bisector_slope

end parallelogram_bisector_slope_l501_50161


namespace sum_of_squares_l501_50125

theorem sum_of_squares (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h_sum : a + b + c = 0) (h_power : a^4 + b^4 + c^4 = a^6 + b^6 + c^6) :
  a^2 + b^2 + c^2 = 6/5 := by sorry

end sum_of_squares_l501_50125


namespace abs_4x_minus_6_not_positive_l501_50155

theorem abs_4x_minus_6_not_positive (x : ℚ) : 
  ¬(0 < |4 * x - 6|) ↔ x = 3/2 := by
  sorry

end abs_4x_minus_6_not_positive_l501_50155


namespace inequality_not_always_true_l501_50198

theorem inequality_not_always_true (x y w : ℝ) 
  (hx : x > 0) (hy : y > 0) (hxy : x^2 > y^2) (hw : w ≠ 0) :
  ¬ (∀ w, x^2 * w > y^2 * w) :=
by sorry

end inequality_not_always_true_l501_50198


namespace jake_initial_bitcoins_l501_50196

def initial_bitcoins : ℕ → Prop
| b => let after_first_donation := b - 20
       let after_giving_half := (after_first_donation) / 2
       let after_tripling := 3 * after_giving_half
       let final_amount := after_tripling - 10
       final_amount = 80

theorem jake_initial_bitcoins : initial_bitcoins 80 := by
  sorry

end jake_initial_bitcoins_l501_50196


namespace derivative_from_second_derivative_l501_50185

open Real

theorem derivative_from_second_derivative
  (f : ℝ → ℝ)
  (h : ∀ x, deriv^[2] f x = 3) :
  ∀ x, deriv f x = 3 :=
by
  sorry

end derivative_from_second_derivative_l501_50185


namespace distance_to_triangle_plane_l501_50112

-- Define the sphere and points
def Sphere : Type := ℝ × ℝ × ℝ
def Point : Type := ℝ × ℝ × ℝ

-- Define the center and radius of the sphere
def S : Sphere := sorry
def radius : ℝ := 25

-- Define the points on the sphere
def P : Point := sorry
def Q : Point := sorry
def R : Point := sorry

-- Define the distances between points
def PQ : ℝ := 15
def QR : ℝ := 20
def RP : ℝ := 25

-- Define the distance function
def distance (a b : Point) : ℝ := sorry

-- Define the function to calculate the distance from a point to a plane
def distToPlane (point : Point) (a b c : Point) : ℝ := sorry

-- Theorem statement
theorem distance_to_triangle_plane :
  distToPlane S P Q R = 25 * Real.sqrt 3 / 2 := by sorry

end distance_to_triangle_plane_l501_50112
