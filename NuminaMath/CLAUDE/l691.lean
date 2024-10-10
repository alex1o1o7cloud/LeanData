import Mathlib

namespace discount_problem_l691_69116

/-- The total cost after discount for a given number of toys, cost per toy, and discount percentage. -/
def totalCostAfterDiscount (numToys : ℕ) (costPerToy : ℚ) (discountPercentage : ℚ) : ℚ :=
  let totalCost := numToys * costPerToy
  let discountAmount := totalCost * (discountPercentage / 100)
  totalCost - discountAmount

/-- Theorem stating that the total cost after a 20% discount for 5 toys costing $3 each is $12. -/
theorem discount_problem : totalCostAfterDiscount 5 3 20 = 12 := by
  sorry

end discount_problem_l691_69116


namespace sum_of_g_42_and_neg_42_l691_69115

/-- Given a function g: ℝ → ℝ defined as g(x) = ax^8 + bx^6 - cx^4 + dx^2 + 5
    where a, b, c, d are real constants, if g(42) = 3,
    then g(42) + g(-42) = 6 -/
theorem sum_of_g_42_and_neg_42 (a b c d : ℝ) (g : ℝ → ℝ)
    (h1 : ∀ x, g x = a * x^8 + b * x^6 - c * x^4 + d * x^2 + 5)
    (h2 : g 42 = 3) :
  g 42 + g (-42) = 6 := by
  sorry

end sum_of_g_42_and_neg_42_l691_69115


namespace tylenol_interval_l691_69121

-- Define the problem parameters
def total_hours : ℝ := 12
def tablet_mg : ℝ := 500
def tablets_per_dose : ℝ := 2
def total_grams : ℝ := 3

-- Define the theorem
theorem tylenol_interval :
  let total_mg : ℝ := total_grams * 1000
  let total_tablets : ℝ := total_mg / tablet_mg
  let intervals : ℝ := total_tablets - 1
  total_hours / intervals = 2.4 := by sorry

end tylenol_interval_l691_69121


namespace log_problem_l691_69158

theorem log_problem (y : ℝ) (m : ℝ) 
  (h1 : Real.log 5 / Real.log 8 = y)
  (h2 : Real.log 125 / Real.log 2 = m * y) : 
  m = 9 := by
sorry

end log_problem_l691_69158


namespace min_value_theorem_l691_69180

theorem min_value_theorem (a b : ℝ) (ha : a > 0) 
  (h : ∀ x > 0, (a * x - 2) * (x^2 + b * x - 5) ≥ 0) :
  (∃ (b₀ : ℝ), b₀ + 4 / a = 2 * Real.sqrt 5) ∧ 
  (∀ (b₁ : ℝ), b₁ + 4 / a ≥ 2 * Real.sqrt 5) := by
sorry

end min_value_theorem_l691_69180


namespace average_hiring_per_week_l691_69177

def employee_hiring (week1 week2 week3 week4 : ℕ) : Prop :=
  (week1 = week2 + 200) ∧
  (week2 + 150 = week3) ∧
  (week4 = 2 * week3) ∧
  (week4 = 400)

theorem average_hiring_per_week 
  (week1 week2 week3 week4 : ℕ) 
  (h : employee_hiring week1 week2 week3 week4) : 
  (week1 + week2 + week3 + week4) / 4 = 225 := by
  sorry

end average_hiring_per_week_l691_69177


namespace inequality_preservation_l691_69144

theorem inequality_preservation (x y : ℝ) (h : x > y) : x - 3 > y - 3 := by
  sorry

end inequality_preservation_l691_69144


namespace power_of_product_cube_l691_69154

theorem power_of_product_cube (x : ℝ) : (2 * x^3)^2 = 4 * x^6 := by
  sorry

end power_of_product_cube_l691_69154


namespace rationalize_and_divide_l691_69143

theorem rationalize_and_divide : (8 / Real.sqrt 8) / 2 = Real.sqrt 2 := by
  sorry

end rationalize_and_divide_l691_69143


namespace thirty_factorial_trailing_zeros_l691_69117

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (Finset.range n).sum (fun k => (k + 1) / 5^(k + 1))

/-- 30! has 7 trailing zeros -/
theorem thirty_factorial_trailing_zeros :
  trailingZeros 30 = 7 := by
  sorry

end thirty_factorial_trailing_zeros_l691_69117


namespace probability_four_twos_value_l691_69156

def num_dice : ℕ := 8
def num_sides : ℕ := 8
def num_success : ℕ := 4

def probability_exactly_four_twos : ℚ :=
  (num_dice.choose num_success) * 
  ((1 : ℚ) / num_sides) ^ num_success * 
  ((num_sides - 1 : ℚ) / num_sides) ^ (num_dice - num_success)

theorem probability_four_twos_value : 
  probability_exactly_four_twos = 168070 / 16777216 := by sorry

end probability_four_twos_value_l691_69156


namespace friend_consumption_l691_69151

def total_people : ℕ := 8
def pizzas : ℕ := 5
def slices_per_pizza : ℕ := 8
def pasta_bowls : ℕ := 2
def garlic_breads : ℕ := 12

def ron_scott_pizza : ℕ := 10
def mark_pizza : ℕ := 2
def sam_pizza : ℕ := 4

def ron_scott_pasta_percent : ℚ := 40 / 100
def ron_scott_mark_garlic_percent : ℚ := 25 / 100

theorem friend_consumption :
  let remaining_friends := total_people - 4
  let remaining_pizza := pizzas * slices_per_pizza - (ron_scott_pizza + mark_pizza + sam_pizza)
  let remaining_pasta_percent := 1 - ron_scott_pasta_percent
  let remaining_garlic_percent := 1 - ron_scott_mark_garlic_percent
  (remaining_pizza / remaining_friends = 6) ∧
  (remaining_pasta_percent / (total_people - 2) = 10 / 100) ∧
  (remaining_garlic_percent * garlic_breads / (total_people - 3) = 1.8) := by
  sorry

end friend_consumption_l691_69151


namespace quadratic_root_relation_l691_69106

theorem quadratic_root_relation (p : ℝ) : 
  (∃ a : ℝ, a ≠ 0 ∧ (a^2 + p*a + 18 = 0) ∧ ((2*a)^2 + p*(2*a) + 18 = 0)) ↔ 
  (p = 9 ∨ p = -9) :=
sorry

end quadratic_root_relation_l691_69106


namespace total_weight_carrots_cucumbers_l691_69130

theorem total_weight_carrots_cucumbers : 
  ∀ (weight_carrots : ℝ) (weight_ratio : ℝ),
    weight_carrots = 250 →
    weight_ratio = 2.5 →
    weight_carrots + weight_ratio * weight_carrots = 875 :=
by
  sorry

end total_weight_carrots_cucumbers_l691_69130


namespace negation_of_forall_positive_l691_69112

theorem negation_of_forall_positive (S : Set ℚ) :
  (¬ ∀ x ∈ S, 2 * x + 1 > 0) ↔ (∃ x ∈ S, 2 * x + 1 ≤ 0) := by
  sorry

end negation_of_forall_positive_l691_69112


namespace product_of_binomials_l691_69198

theorem product_of_binomials (m : ℝ) : (-m + 2) * (-m - 2) = m^2 - 4 := by
  sorry

end product_of_binomials_l691_69198


namespace infinite_points_in_circle_l691_69162

open Set

-- Define the circle
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p | (p.1 - center.1)^2 + (p.2 - center.2)^2 < radius^2}

-- Define the condition for point P
def SatisfiesCondition (p : ℝ × ℝ) (a b : ℝ × ℝ) : Prop :=
  (p.1 - a.1)^2 + (p.2 - a.2)^2 + (p.1 - b.1)^2 + (p.2 - b.2)^2 ≤ 5

-- Theorem statement
theorem infinite_points_in_circle :
  let center := (0, 0)
  let radius := 2
  let a := (-2, 0)  -- One endpoint of the diameter
  let b := (2, 0)   -- Other endpoint of the diameter
  let valid_points := {p ∈ Circle center radius | SatisfiesCondition p a b}
  Infinite valid_points := by sorry

end infinite_points_in_circle_l691_69162


namespace triangle_construction_theorem_l691_69192

/-- A line in 2D space -/
structure Line where
  -- Define a line using two points
  p1 : ℝ × ℝ
  p2 : ℝ × ℝ

/-- A triangle in 2D space -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Check if a point lies on a line -/
def point_on_line (p : ℝ × ℝ) (l : Line) : Prop := sorry

/-- Check if a line is a perpendicular bisector of a triangle side -/
def is_perp_bisector (l : Line) (t : Triangle) : Prop := sorry

/-- The main theorem -/
theorem triangle_construction_theorem 
  (e f g : Line) -- Three given lines (perpendicular bisectors)
  (P : ℝ × ℝ)    -- Given point
  (h : point_on_line P e ∨ point_on_line P f ∨ point_on_line P g) -- P is on one of the lines
  : ∃ (t : Triangle), 
    (point_on_line P e ∧ is_perp_bisector e t) ∨ 
    (point_on_line P f ∧ is_perp_bisector f t) ∨ 
    (point_on_line P g ∧ is_perp_bisector g t) :=
sorry

end triangle_construction_theorem_l691_69192


namespace four_digit_divisible_by_five_l691_69147

theorem four_digit_divisible_by_five (n : ℕ) : 
  (5000 ≤ n ∧ n ≤ 5999) ∧ (n % 5 = 0) → 
  (Finset.filter (λ x : ℕ => (5000 ≤ x ∧ x ≤ 5999) ∧ (x % 5 = 0)) (Finset.range 10000)).card = 200 :=
by sorry

end four_digit_divisible_by_five_l691_69147


namespace abs_neg_three_halves_l691_69145

theorem abs_neg_three_halves : |(-3 : ℚ) / 2| = 3 / 2 := by
  sorry

end abs_neg_three_halves_l691_69145


namespace max_diagonal_sum_l691_69104

/-- A rhombus with side length 5 and diagonals d1 and d2 -/
structure Rhombus where
  side_length : ℝ
  d1 : ℝ
  d2 : ℝ
  side_is_5 : side_length = 5
  d1_le_6 : d1 ≤ 6
  d2_ge_6 : d2 ≥ 6

/-- The maximum sum of diagonals in a rhombus with given constraints is 14 -/
theorem max_diagonal_sum (r : Rhombus) : (r.d1 + r.d2 ≤ 14) ∧ (∃ (s : Rhombus), s.d1 + s.d2 = 14) :=
  sorry

end max_diagonal_sum_l691_69104


namespace saras_quarters_l691_69114

theorem saras_quarters (current_quarters borrowed_quarters : ℕ) 
  (h1 : current_quarters = 512)
  (h2 : borrowed_quarters = 271) :
  current_quarters + borrowed_quarters = 783 := by
  sorry

end saras_quarters_l691_69114


namespace bobs_spending_ratio_l691_69153

/-- Proves that given Bob's spending pattern, the ratio of Tuesday's spending to Monday's remaining amount is 1/5 -/
theorem bobs_spending_ratio : 
  ∀ (initial_amount : ℚ) (tuesday_spent : ℚ) (final_amount : ℚ),
  initial_amount = 80 →
  final_amount = 20 →
  tuesday_spent > 0 →
  tuesday_spent < 40 →
  20 = 40 - tuesday_spent - (3/8) * (40 - tuesday_spent) →
  tuesday_spent / 40 = 1/5 := by
sorry

end bobs_spending_ratio_l691_69153


namespace greatest_three_digit_multiple_of_17_l691_69142

theorem greatest_three_digit_multiple_of_17 : 
  ∀ n : ℕ, n ≤ 999 → n ≥ 100 → n % 17 = 0 → n ≤ 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l691_69142


namespace select_real_coins_l691_69168

/-- Represents the outcome of a weighing on a balance scale -/
inductive WeighingResult
| Equal : WeighingResult
| LeftHeavier : WeighingResult
| RightHeavier : WeighingResult

/-- Represents a group of coins -/
structure CoinGroup where
  total : Nat
  counterfeit : Nat

/-- Represents a weighing action on the balance scale -/
def weighing (left right : CoinGroup) : WeighingResult :=
  sorry

/-- Represents the process of selecting coins -/
def selectCoins (coins : CoinGroup) (weighings : Nat) : Option (Finset Nat) :=
  sorry

theorem select_real_coins 
  (total_coins : Nat)
  (counterfeit_coins : Nat)
  (max_weighings : Nat)
  (coins_to_select : Nat)
  (h1 : total_coins = 40)
  (h2 : counterfeit_coins = 3)
  (h3 : max_weighings = 3)
  (h4 : coins_to_select = 16)
  (h5 : counterfeit_coins < total_coins) :
  ∃ (selected : Finset Nat), 
    (selected.card = coins_to_select) ∧ 
    (∀ c ∈ selected, c ≤ total_coins - counterfeit_coins) ∧
    (selectCoins ⟨total_coins, counterfeit_coins⟩ max_weighings = some selected) :=
sorry

end select_real_coins_l691_69168


namespace length_of_projected_segment_l691_69164

/-- Given two points A and B on the y-axis, and their respective projections A' and B' on the line y = x,
    with AA' and BB' intersecting at point C, prove that the length of A'B' is 2.5√2. -/
theorem length_of_projected_segment (A B A' B' C : ℝ × ℝ) : 
  A = (0, 10) →
  B = (0, 15) →
  C = (3, 9) →
  (A'.1 = A'.2) →
  (B'.1 = B'.2) →
  (∃ t : ℝ, A + t • (C - A) = A') →
  (∃ s : ℝ, B + s • (C - B) = B') →
  ‖A' - B'‖ = 2.5 * Real.sqrt 2 :=
by sorry

end length_of_projected_segment_l691_69164


namespace angle_sum_proof_l691_69152

theorem angle_sum_proof (α β : Real) 
  (h1 : Real.tan α = 1/7) 
  (h2 : Real.tan β = 1/3) : 
  α + 2*β = π/4 := by sorry

end angle_sum_proof_l691_69152


namespace circle_line_intersection_l691_69187

/-- The value of 'a' for a circle with equation x^2 + y^2 - 2ax + 2y + 1 = 0,
    where the line y = -x + 1 passes through its center. -/
theorem circle_line_intersection (a : ℝ) : 
  (∃ x y : ℝ, x^2 + y^2 - 2*a*x + 2*y + 1 = 0 ∧ 
               y = -x + 1 ∧ 
               ∀ x' y' : ℝ, x'^2 + y'^2 - 2*a*x' + 2*y' + 1 = 0 → 
                 (x - x')^2 + (y - y')^2 ≤ (x' - a)^2 + (y' + 1)^2) → 
  a = 2 :=
by sorry

end circle_line_intersection_l691_69187


namespace brick_length_is_20_l691_69176

/-- The length of a brick in centimeters -/
def brick_length : ℝ := 20

/-- The width of a brick in centimeters -/
def brick_width : ℝ := 10

/-- The height of a brick in centimeters -/
def brick_height : ℝ := 7.5

/-- The length of the wall in meters -/
def wall_length : ℝ := 29

/-- The width of the wall in meters -/
def wall_width : ℝ := 2

/-- The height of the wall in meters -/
def wall_height : ℝ := 0.75

/-- The number of bricks in the wall -/
def number_of_bricks : ℕ := 29000

/-- Conversion factor from meters to centimeters -/
def m_to_cm : ℝ := 100

theorem brick_length_is_20 :
  brick_length = 20 :=
by
  have h1 : brick_length * brick_width * brick_height = 
    (wall_length * wall_width * wall_height * m_to_cm^3) / number_of_bricks :=
    sorry
  sorry

end brick_length_is_20_l691_69176


namespace function_composition_value_l691_69157

/-- Given a function g and a composition f[g(x)], prove that f(0) = 4/5 -/
theorem function_composition_value (g : ℝ → ℝ) (f : ℝ → ℝ) :
  (∀ x, g x = 1 - 3 * x) →
  (∀ x, f (g x) = (1 - x^2) / (1 + x^2)) →
  f 0 = 4/5 := by
  sorry

end function_composition_value_l691_69157


namespace tan_x_axis_intersection_l691_69172

theorem tan_x_axis_intersection :
  ∀ (x : ℝ), (∃ (n : ℤ), x = -π/8 + n*π/2) ↔ Real.tan (2*x + π/4) = 0 :=
by sorry

end tan_x_axis_intersection_l691_69172


namespace runner_problem_l691_69127

theorem runner_problem (v : ℝ) (h : v > 0) :
  (40 / v = 20 / v + 8) → (40 / (v / 2) = 16) := by
  sorry

end runner_problem_l691_69127


namespace cost_per_bag_is_seven_l691_69188

/-- Calculates the cost per bag given the number of bags, selling price, and desired profit --/
def cost_per_bag (num_bags : ℕ) (selling_price : ℚ) (desired_profit : ℚ) : ℚ :=
  (num_bags * selling_price - desired_profit) / num_bags

/-- Theorem: Given 100 bags sold at $10 each with a $300 profit, the cost per bag is $7 --/
theorem cost_per_bag_is_seven :
  cost_per_bag 100 10 300 = 7 := by
  sorry

end cost_per_bag_is_seven_l691_69188


namespace complex_division_l691_69138

theorem complex_division (i : ℂ) (h : i^2 = -1) : (1 + 2*i) / i = 2 - i := by
  sorry

end complex_division_l691_69138


namespace f_positive_on_interval_l691_69113

open Real

noncomputable def f (a x : ℝ) : ℝ := a * log x - x - a / x + 2 * a

theorem f_positive_on_interval (a : ℝ) :
  (∀ x ∈ Set.Icc 1 (exp 1), f a x > 0) ↔ a > (exp 2) / (3 * exp 1 - 1) := by
  sorry

end f_positive_on_interval_l691_69113


namespace square_side_length_l691_69133

theorem square_side_length (area : ℚ) (side : ℚ) (h1 : area = 9/16) (h2 : side * side = area) : side = 3/4 := by
  sorry

end square_side_length_l691_69133


namespace allocation_schemes_eq_540_l691_69129

/-- The number of ways to allocate teachers to schools -/
def allocation_schemes (math_teachers language_teachers schools : ℕ) : ℕ :=
  (math_teachers.factorial * language_teachers.factorial) / 
  ((math_teachers / schools).factorial ^ schools * 
   (language_teachers / schools).factorial ^ schools * schools.factorial)

/-- Theorem: The number of allocation schemes for the given problem is 540 -/
theorem allocation_schemes_eq_540 : 
  allocation_schemes 3 6 3 = 540 := by
  sorry

end allocation_schemes_eq_540_l691_69129


namespace existence_of_separated_points_l691_69184

/-- A type representing a segment in a plane -/
structure Segment where
  -- Add necessary fields

/-- A type representing a point in a plane -/
structure Point where
  -- Add necessary fields

/-- Checks if two segments are parallel -/
def are_parallel (s1 s2 : Segment) : Prop :=
  sorry

/-- Checks if two segments intersect -/
def intersect (s1 s2 : Segment) : Prop :=
  sorry

/-- Checks if a segment separates two points -/
def separates (s : Segment) (p1 p2 : Point) : Prop :=
  sorry

/-- Main theorem -/
theorem existence_of_separated_points (n : ℕ) (segments : Fin (n^2) → Segment)
  (h1 : ∀ i j, i ≠ j → ¬(are_parallel (segments i) (segments j)))
  (h2 : ∀ i j, i ≠ j → ¬(intersect (segments i) (segments j))) :
  ∃ (points : Fin n → Point),
    ∀ i j, i ≠ j → ∃ k, separates (segments k) (points i) (points j) :=
sorry

end existence_of_separated_points_l691_69184


namespace number_of_placements_is_36_l691_69181

/-- The number of ways to place 3 men and 4 women into groups -/
def number_of_placements : ℕ :=
  let num_men : ℕ := 3
  let num_women : ℕ := 4
  let num_groups_of_two : ℕ := 2
  let num_groups_of_three : ℕ := 1
  let ways_to_choose_man_for_three : ℕ := Nat.choose num_men 1
  let ways_to_choose_women_for_three : ℕ := Nat.choose num_women 2
  let ways_to_pair_remaining : ℕ := 2
  ways_to_choose_man_for_three * ways_to_choose_women_for_three * ways_to_pair_remaining

/-- Theorem stating that the number of placements is 36 -/
theorem number_of_placements_is_36 : number_of_placements = 36 := by
  sorry

end number_of_placements_is_36_l691_69181


namespace integral_sqrt_one_minus_x_squared_plus_x_squared_l691_69174

theorem integral_sqrt_one_minus_x_squared_plus_x_squared :
  ∫ x in (-1 : ℝ)..1, (Real.sqrt (1 - x^2) + x^2) = π / 2 + 2 / 3 := by
  sorry

end integral_sqrt_one_minus_x_squared_plus_x_squared_l691_69174


namespace binomial_variance_transform_l691_69120

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h_p : 0 ≤ p ∧ p ≤ 1

/-- Variance of a binomial random variable -/
def variance (X : BinomialRV) : ℝ := X.n * X.p * (1 - X.p)

/-- Variance of a linear transformation of a random variable -/
def varianceLinearTransform (X : BinomialRV) (a b : ℝ) : ℝ := a^2 * variance X

/-- The main theorem to prove -/
theorem binomial_variance_transform (ξ : BinomialRV) 
    (h_n : ξ.n = 100) (h_p : ξ.p = 0.3) : 
    varianceLinearTransform ξ 3 (-5) = 189 := by
  sorry

end binomial_variance_transform_l691_69120


namespace angle_trisection_l691_69191

theorem angle_trisection (α : ℝ) (h : α = 54) :
  ∃ β : ℝ, β * 3 = α ∧ β = 18 := by
  sorry

end angle_trisection_l691_69191


namespace tangent_inequality_l691_69124

theorem tangent_inequality (α β : Real) 
  (h1 : 0 < α) (h2 : α ≤ π/4) (h3 : 0 < β) (h4 : β ≤ π/4) : 
  Real.sqrt (Real.tan α * Real.tan β) ≤ Real.tan ((α + β)/2) ∧ 
  Real.tan ((α + β)/2) ≤ (Real.tan α + Real.tan β)/2 := by
  sorry

end tangent_inequality_l691_69124


namespace sum_210_72_in_base5_l691_69123

/-- Converts a decimal number to its base 5 representation -/
def toBase5 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of base 5 digits to a decimal number -/
def fromBase5 (digits : List ℕ) : ℕ :=
  sorry

theorem sum_210_72_in_base5 :
  toBase5 (210 + 72) = [2, 0, 6, 2] := by
  sorry

end sum_210_72_in_base5_l691_69123


namespace pole_reconfiguration_l691_69131

/-- Represents the configuration of electric poles on a road --/
structure RoadConfig where
  length : ℕ
  original_spacing : ℕ
  new_spacing : ℕ

/-- Calculates the number of holes needed for a given spacing --/
def holes_needed (config : RoadConfig) (spacing : ℕ) : ℕ :=
  config.length / spacing + 1

/-- Calculates the number of common holes between two spacings --/
def common_holes (config : RoadConfig) : ℕ :=
  config.length / (Nat.lcm config.original_spacing config.new_spacing) + 1

/-- The main theorem about the number of new holes and abandoned holes --/
theorem pole_reconfiguration (config : RoadConfig) 
  (h_length : config.length = 3000)
  (h_original : config.original_spacing = 50)
  (h_new : config.new_spacing = 60) :
  (holes_needed config config.new_spacing - common_holes config = 40) ∧
  (holes_needed config config.original_spacing - common_holes config = 50) := by
  sorry


end pole_reconfiguration_l691_69131


namespace shorts_price_is_6_l691_69103

/-- The price of a single jacket in dollars -/
def jacket_price : ℕ := 10

/-- The number of jackets bought -/
def num_jackets : ℕ := 3

/-- The price of a single pair of pants in dollars -/
def pants_price : ℕ := 12

/-- The number of pairs of pants bought -/
def num_pants : ℕ := 4

/-- The number of pairs of shorts bought -/
def num_shorts : ℕ := 2

/-- The total amount spent in dollars -/
def total_spent : ℕ := 90

theorem shorts_price_is_6 :
  ∃ (shorts_price : ℕ),
    shorts_price * num_shorts + 
    jacket_price * num_jackets + 
    pants_price * num_pants = total_spent ∧
    shorts_price = 6 :=
by sorry

end shorts_price_is_6_l691_69103


namespace total_selling_price_l691_69169

-- Define the cost and loss percentage for each item
def cost1 : ℕ := 750
def cost2 : ℕ := 1200
def cost3 : ℕ := 500
def loss_percent1 : ℚ := 10 / 100
def loss_percent2 : ℚ := 15 / 100
def loss_percent3 : ℚ := 5 / 100

-- Calculate the selling price of an item
def selling_price (cost : ℕ) (loss_percent : ℚ) : ℚ :=
  cost - (cost * loss_percent)

-- Define the theorem
theorem total_selling_price :
  selling_price cost1 loss_percent1 +
  selling_price cost2 loss_percent2 +
  selling_price cost3 loss_percent3 = 2170 := by
  sorry

end total_selling_price_l691_69169


namespace oak_trees_remaining_l691_69167

/-- The number of oak trees remaining in the park after cutting down damaged trees -/
def remaining_oak_trees (initial : ℕ) (cut_down : ℕ) : ℕ :=
  initial - cut_down

/-- Theorem stating that the number of remaining oak trees is 7 -/
theorem oak_trees_remaining :
  remaining_oak_trees 9 2 = 7 := by
  sorry

end oak_trees_remaining_l691_69167


namespace yellow_bows_count_l691_69196

theorem yellow_bows_count (total : ℕ) 
  (h_red : (total : ℚ) / 4 = total / 4)
  (h_blue : (total : ℚ) / 3 = total / 3)
  (h_green : (total : ℚ) / 6 = total / 6)
  (h_yellow : (total : ℚ) / 12 = total / 12)
  (h_white : (total : ℚ) - (total / 4 + total / 3 + total / 6 + total / 12) = 40) :
  (total : ℚ) / 12 = 20 := by
  sorry

end yellow_bows_count_l691_69196


namespace cyclic_quadrilateral_perpendicular_diagonals_l691_69199

/-- A convex quadrilateral with side lengths a, b, c, d in sequence, inscribed in a circle of radius R -/
structure CyclicQuadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  R : ℝ
  convex : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0
  cyclic : a^2 + b^2 + c^2 + d^2 = 8 * R^2

/-- The diagonals of a quadrilateral are perpendicular -/
def has_perpendicular_diagonals (q : CyclicQuadrilateral) : Prop :=
  ∃ (A B C D : ℝ × ℝ), 
    (A.1 - C.1) * (B.1 - D.1) + (A.2 - C.2) * (B.2 - D.2) = 0

/-- 
If a convex quadrilateral ABCD with side lengths a, b, c, d in sequence, 
inscribed in a circle with radius R, satisfies a^2 + b^2 + c^2 + d^2 = 8R^2, 
then the diagonals of the quadrilateral are perpendicular.
-/
theorem cyclic_quadrilateral_perpendicular_diagonals (q : CyclicQuadrilateral) :
  has_perpendicular_diagonals q :=
sorry

end cyclic_quadrilateral_perpendicular_diagonals_l691_69199


namespace pascal_interior_sum_l691_69155

/-- Represents the sum of interior numbers in a row of Pascal's Triangle -/
def interior_sum (n : ℕ) : ℕ := 2^(n-1) - 2

theorem pascal_interior_sum (h : interior_sum 6 = 30) : interior_sum 8 = 126 := by
  sorry

end pascal_interior_sum_l691_69155


namespace right_triangle_sets_l691_69111

def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

theorem right_triangle_sets :
  is_right_triangle 3 4 5 ∧
  ¬is_right_triangle 2 3 4 ∧
  ¬is_right_triangle 4 6 7 ∧
  ¬is_right_triangle 5 11 12 := by
  sorry

end right_triangle_sets_l691_69111


namespace algebraic_expression_value_l691_69194

theorem algebraic_expression_value (a b : ℝ) : 
  (2 * a * (-1)^3 - 3 * b * (-1) + 8 = 18) → 
  (9 * b - 6 * a + 2 = 32) := by
  sorry

end algebraic_expression_value_l691_69194


namespace intersection_empty_implies_a_nonnegative_l691_69163

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | x - a > 0}
def B : Set ℝ := {x | x ≤ 0}

-- State the theorem
theorem intersection_empty_implies_a_nonnegative (a : ℝ) :
  A a ∩ B = ∅ → a ≥ 0 := by
  sorry

end intersection_empty_implies_a_nonnegative_l691_69163


namespace largest_n_multiple_of_four_l691_69122

def expression (n : ℕ) : ℤ :=
  7 * (n - 3)^4 - n^2 + 12*n - 30

theorem largest_n_multiple_of_four :
  ∀ n : ℕ, n < 100000 →
    (4 ∣ expression n) →
    n ≤ 99999 ∧
    (4 ∣ expression 99999) ∧
    99999 < 100000 :=
by sorry

end largest_n_multiple_of_four_l691_69122


namespace sum_2012_terms_eq_4021_l691_69126

/-- A sequence where each term (after the second) is the sum of its previous and next terms -/
def SpecialSequence (a₀ a₁ : ℤ) : ℕ → ℤ
  | 0 => a₀
  | 1 => a₁
  | n + 2 => SpecialSequence a₀ a₁ (n + 1) + SpecialSequence a₀ a₁ n

/-- The sum of the first n terms of the special sequence -/
def SequenceSum (a₀ a₁ : ℤ) (n : ℕ) : ℤ :=
  (List.range n).map (SpecialSequence a₀ a₁) |>.sum

theorem sum_2012_terms_eq_4021 :
  SequenceSum 2010 2011 2012 = 4021 := by
  sorry

end sum_2012_terms_eq_4021_l691_69126


namespace odd_digits_base4_345_l691_69128

/-- Converts a natural number to its base-4 representation as a list of digits -/
def toBase4 (n : ℕ) : List ℕ :=
  sorry

/-- Counts the number of odd digits in a list of natural numbers -/
def countOddDigits (digits : List ℕ) : ℕ :=
  sorry

/-- The number of odd digits in the base-4 representation of 345 is 4 -/
theorem odd_digits_base4_345 : countOddDigits (toBase4 345) = 4 :=
  sorry

end odd_digits_base4_345_l691_69128


namespace number_difference_l691_69159

theorem number_difference (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a / b = 2 / 3) (h4 : a^3 + b^3 = 945) : b - a = 3 := by
  sorry

end number_difference_l691_69159


namespace value_of_c_l691_69170

theorem value_of_c (a b c : ℝ) : 
  8 = 0.04 * a → 
  4 = 0.08 * b → 
  c = b / a → 
  c = 0.25 := by
sorry

end value_of_c_l691_69170


namespace arithmetic_sequence_quadratic_root_l691_69149

theorem arithmetic_sequence_quadratic_root (x y z : ℝ) : 
  (∃ d : ℝ, y = x + d ∧ z = x + 2*d) →  -- arithmetic sequence
  x ≤ y ∧ y ≤ z ∧ z ≤ 10 →             -- ordering condition
  (∃! r : ℝ, z*r^2 + y*r + x = 0) →    -- quadratic has exactly one root
  (∃ r : ℝ, z*r^2 + y*r + x = 0 ∧ r = Real.sqrt 3) := by
sorry

end arithmetic_sequence_quadratic_root_l691_69149


namespace lines_always_parallel_l691_69135

/-- A linear function f(x) = kx + b -/
def f (k b x : ℝ) : ℝ := k * x + b

/-- Line l₁ represented by y = f(x) -/
def l₁ (k b : ℝ) : Set (ℝ × ℝ) := {(x, y) | y = f k b x}

/-- Line l₂ defined as y - y₀ = f(x) - f(x₀) -/
def l₂ (k b x₀ y₀ : ℝ) : Set (ℝ × ℝ) := {(x, y) | y - y₀ = f k b x - f k b x₀}

/-- Point P -/
def P (x₀ y₀ : ℝ) : ℝ × ℝ := (x₀, y₀)

theorem lines_always_parallel (k b x₀ y₀ : ℝ) 
  (h : P x₀ y₀ ∉ l₁ k b) : 
  ∃ (m : ℝ), ∀ (x y : ℝ), 
    ((x, y) ∈ l₁ k b ↔ y = k * x + m) ∧ 
    ((x, y) ∈ l₂ k b x₀ y₀ ↔ y = k * x + (y₀ - k * x₀)) :=
sorry

end lines_always_parallel_l691_69135


namespace solution_set_M_range_of_k_l691_69108

-- Define the function f(x)
def f (x : ℝ) : ℝ := |2*x - 4| - |x - 3|

-- Theorem for the solution set M
theorem solution_set_M : 
  {x : ℝ | f x ≤ 2} = Set.Icc (-1) 3 := by sorry

-- Theorem for the range of k
theorem range_of_k : 
  {k : ℝ | ∃ x, k^2 - 4*k - 3*f x = 0} = Set.Icc (-1) 3 := by sorry

end solution_set_M_range_of_k_l691_69108


namespace algebraic_simplification_l691_69150

theorem algebraic_simplification (a : ℝ) : a * (a + 2) - 2 * a = a^2 := by
  sorry

end algebraic_simplification_l691_69150


namespace division_result_l691_69118

theorem division_result : (32 / 8 : ℚ) = 4 := by
  sorry

end division_result_l691_69118


namespace simplify_expression_l691_69102

theorem simplify_expression (x y : ℝ) (h : y ≠ 0) :
  let P := x^2 + y^2
  let Q := x^2 - y^2
  ((P + 3*Q) / (P - Q)) - ((P - 3*Q) / (P + Q)) = (2*x^4 - y^4) / (x^2 * y^2) :=
by sorry

end simplify_expression_l691_69102


namespace arithmetic_calculation_l691_69136

theorem arithmetic_calculation : 10 - 9 + 8 * 7 + 6 - 5 * 4 / 2 + 3 - 1 = 55 := by
  sorry

end arithmetic_calculation_l691_69136


namespace books_sold_l691_69125

/-- Given that Tom initially had 5 books, bought 38 new books, and now has 39 books in total,
    prove that the number of books Tom sold is 4. -/
theorem books_sold (initial_books : ℕ) (new_books : ℕ) (total_books : ℕ) (sold_books : ℕ) : 
  initial_books = 5 → new_books = 38 → total_books = 39 → 
  initial_books - sold_books + new_books = total_books →
  sold_books = 4 := by
sorry

end books_sold_l691_69125


namespace kayak_trip_remaining_fraction_l691_69186

/-- Given a kayak trip with total distance and distance paddled before lunch,
    calculate the fraction of the trip remaining after lunch -/
theorem kayak_trip_remaining_fraction
  (total_distance : ℝ)
  (distance_before_lunch : ℝ)
  (h1 : total_distance = 36)
  (h2 : distance_before_lunch = 12)
  : (total_distance - distance_before_lunch) / total_distance = 2/3 := by
  sorry

end kayak_trip_remaining_fraction_l691_69186


namespace sum_of_squares_of_roots_l691_69171

theorem sum_of_squares_of_roots (x₁ x₂ : ℝ) : 
  (5 * x₁^2 + 20 * x₁ - 25 = 0) →
  (5 * x₂^2 + 20 * x₂ - 25 = 0) →
  (x₁ ≠ x₂) →
  x₁^2 + x₂^2 = 26 := by
sorry

end sum_of_squares_of_roots_l691_69171


namespace binomial_expansion_constant_term_l691_69183

theorem binomial_expansion_constant_term (a : ℝ) : 
  (∃ (f : ℝ → ℝ), (∀ x, x ≠ 0 → f x = (a * x^2 - 2 / Real.sqrt x)^5) ∧ 
   (∃ c, c = 160 ∧ (∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x| ∧ |x| < δ → |f x - c| < ε))) →
  a = 2 := by
sorry

end binomial_expansion_constant_term_l691_69183


namespace quadratic_other_intercept_l691_69166

/-- Given a quadratic function f(x) = ax² + bx + c with vertex (5, -3) and 
    one x-intercept at (1, 0), the x-coordinate of the other x-intercept is 9. -/
theorem quadratic_other_intercept 
  (a b c : ℝ) 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = a * x^2 + b * x + c) 
  (h2 : f 1 = 0) 
  (h3 : (5 : ℝ) = -b / (2 * a)) 
  (h4 : f 5 = -3) : 
  ∃ x, x ≠ 1 ∧ f x = 0 ∧ x = 9 := by
sorry

end quadratic_other_intercept_l691_69166


namespace hyperbola_properties_l691_69107

/-- Given a hyperbola with equation x²/4 - y²/12 = 1, prove its eccentricity and asymptotes. -/
theorem hyperbola_properties :
  let a := 2
  let b := 2 * Real.sqrt 3
  let c := 4
  let e := c / a
  let asymptote (x : ℝ) := Real.sqrt 3 * x
  (∀ x y : ℝ, x^2/4 - y^2/12 = 1 →
    (e = 2 ∧
    (∀ x : ℝ, y = asymptote x ∨ y = -asymptote x))) :=
by sorry

end hyperbola_properties_l691_69107


namespace calculation_proof_inequality_system_solution_l691_69141

-- Part 1
theorem calculation_proof : 
  |(-Real.sqrt 3)| + (3 - Real.pi)^(0 : ℝ) + (1/3)^(-2 : ℝ) = Real.sqrt 3 + 10 := by sorry

-- Part 2
theorem inequality_system_solution :
  {x : ℝ | 3*x + 1 > 2*(x - 1) ∧ x - 1 ≤ 3*x + 3} = {x : ℝ | x ≥ -2} := by sorry

end calculation_proof_inequality_system_solution_l691_69141


namespace triangle_inradius_l691_69137

/-- Given a triangle with perimeter 36 and area 45, prove that its inradius is 2.5 -/
theorem triangle_inradius (P : ℝ) (A : ℝ) (r : ℝ) : 
  P = 36 → A = 45 → A = r * (P / 2) → r = 2.5 := by
  sorry

end triangle_inradius_l691_69137


namespace rachel_math_problems_l691_69146

/-- The number of math problems Rachel solved in total -/
def total_problems (problems_per_minute : ℕ) (minutes_solved : ℕ) (problems_next_day : ℕ) : ℕ :=
  problems_per_minute * minutes_solved + problems_next_day

/-- Theorem stating that Rachel solved 151 math problems in total -/
theorem rachel_math_problems :
  total_problems 7 18 25 = 151 := by
  sorry

end rachel_math_problems_l691_69146


namespace consecutive_zeros_in_power_of_five_l691_69173

theorem consecutive_zeros_in_power_of_five : ∃ n : ℕ, n < 10^6 ∧ 5^n % 10^20 < 10^14 := by
  sorry

end consecutive_zeros_in_power_of_five_l691_69173


namespace trig_identities_l691_69189

theorem trig_identities (x : Real) 
  (h1 : 0 < x) (h2 : x < Real.pi) 
  (h3 : Real.sin x + Real.cos x = 7/13) : 
  (Real.sin x * Real.cos x = -60/169) ∧ 
  ((5 * Real.sin x + 4 * Real.cos x) / (15 * Real.sin x - 7 * Real.cos x) = 8/43) := by
  sorry

end trig_identities_l691_69189


namespace tangent_line_slope_is_one_l691_69160

/-- The slope of a line passing through (-1, 0) and tangent to y = e^x is 1 -/
theorem tangent_line_slope_is_one :
  ∀ (a : ℝ), 
    (∃ (k : ℝ), 
      (∀ x, k * (x + 1) = Real.exp x → x = a) ∧ 
      k * (a + 1) = Real.exp a ∧
      k = Real.exp a) →
    k = 1 :=
by
  sorry

end tangent_line_slope_is_one_l691_69160


namespace mary_sugar_calculation_l691_69139

/-- The amount of sugar Mary needs to add to her cake -/
def remaining_sugar (total_required : ℕ) (already_added : ℕ) : ℕ :=
  total_required - already_added

/-- Proof that Mary needs to add 11 more cups of sugar -/
theorem mary_sugar_calculation : remaining_sugar 13 2 = 11 := by
  sorry

end mary_sugar_calculation_l691_69139


namespace sum_of_ab_l691_69161

theorem sum_of_ab (a b : ℝ) (h : a^2 + b^2 + a^2*b^2 = 4*a*b - 1) :
  a + b = 2 ∨ a + b = -2 := by
  sorry

end sum_of_ab_l691_69161


namespace faraway_impossible_totals_l691_69148

/-- Represents the number of creatures in Faraway village -/
structure FarawayVillage where
  horses : ℕ
  goats : ℕ

/-- The total number of creatures in Faraway village -/
def total_creatures (v : FarawayVillage) : ℕ :=
  21 * v.horses + 6 * v.goats

/-- Theorem stating that 74 and 89 cannot be the total number of creatures -/
theorem faraway_impossible_totals :
  ¬ ∃ (v : FarawayVillage), total_creatures v = 74 ∨ total_creatures v = 89 := by
  sorry

end faraway_impossible_totals_l691_69148


namespace quadratic_function_property_l691_69193

def repeating_digits (k : ℕ) (p : ℕ) : ℚ := (k : ℚ) / 9 * (10^p - 1)

def f_k (k : ℕ) (x : ℚ) : ℚ := 9 / (k : ℚ) * x^2 + 2 * x

theorem quadratic_function_property (k : ℕ) (p : ℕ) 
  (h1 : 1 ≤ k) (h2 : k ≤ 9) (h3 : 0 < p) :
  f_k k (repeating_digits k p) = repeating_digits k (2 * p) :=
by sorry

end quadratic_function_property_l691_69193


namespace symmetric_origin_implies_sum_zero_l691_69175

-- Define a property for a function to be symmetric about the origin
def SymmetricAboutOrigin (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = -f (-x)

-- Theorem statement
theorem symmetric_origin_implies_sum_zero
  (f : ℝ → ℝ) (h : SymmetricAboutOrigin f) :
  ∀ x : ℝ, f x + f (-x) = 0 := by
  sorry

end symmetric_origin_implies_sum_zero_l691_69175


namespace square_root_five_plus_one_squared_minus_two_times_plus_seven_equals_eleven_l691_69178

theorem square_root_five_plus_one_squared_minus_two_times_plus_seven_equals_eleven :
  (Real.sqrt 5 + 1)^2 - 2 * (Real.sqrt 5 + 1) + 7 = 11 := by sorry

end square_root_five_plus_one_squared_minus_two_times_plus_seven_equals_eleven_l691_69178


namespace figure_x_value_l691_69109

/-- Given a figure composed of two squares, a right triangle, and a rectangle,
    where:
    - The right triangle has legs measuring 3x and 4x
    - One square has a side length of 4x
    - Another square has a side length of 6x
    - The rectangle has length 3x and width x
    - The total area of the figure is 1100 square inches
    Prove that the value of x is √(1100/61) -/
theorem figure_x_value :
  ∀ x : ℝ,
  (4*x)^2 + (6*x)^2 + (1/2 * 3*x * 4*x) + (3*x * x) = 1100 →
  x = Real.sqrt (1100 / 61) :=
by sorry

end figure_x_value_l691_69109


namespace negation_equivalence_l691_69195

theorem negation_equivalence : 
  (¬(∀ x : ℝ, (x = 0 ∨ x = 1) → x^2 - x = 0)) ↔ 
  (∀ x : ℝ, (x ≠ 0 ∧ x ≠ 1) → x^2 - x ≠ 0) :=
by sorry

end negation_equivalence_l691_69195


namespace right_triangle_with_35_hypotenuse_l691_69119

theorem right_triangle_with_35_hypotenuse (a b c : ℕ) : 
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  c = 35 →           -- Hypotenuse length
  b = a + 1 →        -- Consecutive integer legs
  a + b = 51         -- Sum of leg lengths
  := by sorry

end right_triangle_with_35_hypotenuse_l691_69119


namespace events_related_confidence_l691_69140

-- Define the confidence level
def confidence_level : ℝ := 0.95

-- Define the critical value for 95% confidence
def critical_value : ℝ := 3.841

-- Define the relationship between events A and B
def events_related (K : ℝ) : Prop := K^2 > critical_value

-- Theorem statement
theorem events_related_confidence (K : ℝ) :
  events_related K ↔ confidence_level = 0.95 :=
sorry

end events_related_confidence_l691_69140


namespace professional_ratio_l691_69179

/-- Represents a professional group with engineers, doctors, and lawyers. -/
structure ProfessionalGroup where
  numEngineers : ℕ
  numDoctors : ℕ
  numLawyers : ℕ

/-- The average age of the entire group -/
def groupAverageAge : ℝ := 45

/-- The average age of engineers -/
def engineerAverageAge : ℝ := 40

/-- The average age of doctors -/
def doctorAverageAge : ℝ := 50

/-- The average age of lawyers -/
def lawyerAverageAge : ℝ := 60

/-- Theorem stating the ratio of professionals in the group -/
theorem professional_ratio (group : ProfessionalGroup) :
  group.numEngineers * (doctorAverageAge - groupAverageAge) =
  group.numDoctors * (groupAverageAge - engineerAverageAge) ∧
  group.numEngineers * (lawyerAverageAge - groupAverageAge) =
  3 * group.numLawyers * (groupAverageAge - engineerAverageAge) :=
sorry

end professional_ratio_l691_69179


namespace complex_point_location_l691_69165

theorem complex_point_location (a b : ℝ) (i : ℂ) (h : i * i = -1) 
  (eq : a + i = (b + i) * (2 - i)) : 
  a > 0 ∧ b > 0 := by
  sorry

end complex_point_location_l691_69165


namespace emma_money_theorem_l691_69105

def emma_money_problem (initial_amount furniture_cost fraction_to_anna : ℚ) : Prop :=
  let remaining_after_furniture := initial_amount - furniture_cost
  let amount_to_anna := fraction_to_anna * remaining_after_furniture
  let final_amount := remaining_after_furniture - amount_to_anna
  final_amount = 400

theorem emma_money_theorem :
  emma_money_problem 2000 400 (3/4) := by
  sorry

end emma_money_theorem_l691_69105


namespace compute_alpha_l691_69182

variable (α β : ℂ)

theorem compute_alpha (h1 : (α + β).re > 0)
                       (h2 : (Complex.I * (α - 3 * β)).re > 0)
                       (h3 : β = 4 + 3 * Complex.I) :
  α = 12 - 3 * Complex.I := by
  sorry

end compute_alpha_l691_69182


namespace expression_evaluation_l691_69110

theorem expression_evaluation : 
  (4 * 6) / (12 * 15) * (5 * 12 * 15^2) / (2 * 6 * 5) = 2.5 := by
sorry

end expression_evaluation_l691_69110


namespace distinct_products_count_l691_69101

def S : Finset ℕ := {1, 3, 7, 9, 13}

def products : Finset ℕ :=
  (S.powerset.filter (λ s => s.card ≥ 2)).image (λ s => s.prod id)

theorem distinct_products_count : products.card = 11 := by
  sorry

end distinct_products_count_l691_69101


namespace als_original_portion_l691_69132

theorem als_original_portion (total_initial : ℕ) (total_final : ℕ) 
  (al_loss : ℕ) (al betty clare : ℕ) :
  total_initial = 1500 →
  total_final = 2250 →
  al_loss = 150 →
  al + betty + clare = total_initial →
  (al - al_loss) + 3 * betty + 3 * clare = total_final →
  al = 1050 :=
by sorry

end als_original_portion_l691_69132


namespace melanie_dimes_l691_69134

theorem melanie_dimes (initial : ℕ) (from_dad : ℕ) (total : ℕ) (from_mom : ℕ) : 
  initial = 7 → from_dad = 8 → total = 19 → from_mom = total - (initial + from_dad) → from_mom = 4 := by sorry

end melanie_dimes_l691_69134


namespace box_2_neg2_3_l691_69185

/-- Definition of the box operation for integers -/
def box (a b c : ℤ) : ℚ := a^b - b^c + c^a

/-- Theorem stating that the box operation applied to 2, -2, and 3 equals 69/4 -/
theorem box_2_neg2_3 : box 2 (-2) 3 = 69/4 := by sorry

end box_2_neg2_3_l691_69185


namespace grade_difference_l691_69100

theorem grade_difference (a b c : ℕ) : 
  a + b + c = 25 → 
  3 * a + 4 * b + 5 * c = 106 → 
  c - a = 6 := by
  sorry

end grade_difference_l691_69100


namespace car_trip_distance_l691_69190

theorem car_trip_distance (D : ℝ) 
  (h1 : D / 2 + D / 2 = D)  -- First stop at 1/2 of total distance
  (h2 : D / 2 - (D / 2) / 4 + (D / 2) / 4 = D / 2)  -- Second stop at 1/4 of remaining distance
  (h3 : D - D / 2 - (D / 2) / 4 = 105)  -- Remaining distance after second stop is 105 miles
  : D = 280 := by
sorry

end car_trip_distance_l691_69190


namespace pascal_burger_ratio_l691_69197

/-- The mass of fats in grams in a Pascal Burger -/
def mass_fats : ℕ := 32

/-- The mass of carbohydrates in grams in a Pascal Burger -/
def mass_carbs : ℕ := 48

/-- The ratio of fats to carbohydrates in a Pascal Burger -/
def fats_to_carbs_ratio : Rat := mass_fats / mass_carbs

theorem pascal_burger_ratio :
  fats_to_carbs_ratio = 2 / 3 := by sorry

end pascal_burger_ratio_l691_69197
