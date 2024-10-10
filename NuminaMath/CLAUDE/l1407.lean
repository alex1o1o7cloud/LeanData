import Mathlib

namespace certain_number_problem_l1407_140721

theorem certain_number_problem : ∃ x : ℤ, (5 + (x + 3) = 19) ∧ (x = 11) := by
  sorry

end certain_number_problem_l1407_140721


namespace banana_group_size_l1407_140756

def total_bananas : ℕ := 290
def banana_groups : ℕ := 2
def total_oranges : ℕ := 87
def orange_groups : ℕ := 93

theorem banana_group_size : total_bananas / banana_groups = 145 := by
  sorry

end banana_group_size_l1407_140756


namespace tan_theta_value_l1407_140763

theorem tan_theta_value (θ : Real) (h1 : 0 < θ) (h2 : θ < π/4) 
  (h3 : Real.tan θ + Real.tan (4*θ) = 0) : 
  Real.tan θ = Real.sqrt (5 - 2 * Real.sqrt 5) := by
  sorry

end tan_theta_value_l1407_140763


namespace range_of_m_l1407_140726

-- Define the propositions p and q
def p (x : ℝ) : Prop := x + 2 ≥ 0 ∧ x - 10 ≤ 0

def q (x m : ℝ) : Prop := 1 - m ≤ x ∧ x ≤ 1 + m ∧ m > 0

-- Define the negations of p and q
def not_p (x : ℝ) : Prop := ¬(p x)

def not_q (x m : ℝ) : Prop := ¬(q x m)

-- Define the necessary but not sufficient condition
def necessary_not_sufficient (m : ℝ) : Prop :=
  (∀ x, not_q x m → not_p x) ∧ 
  (∃ x, not_p x ∧ ¬(not_q x m))

-- State the theorem
theorem range_of_m :
  ∀ m : ℝ, necessary_not_sufficient m ↔ m ≥ 9 := by sorry

end range_of_m_l1407_140726


namespace max_value_of_expression_l1407_140751

theorem max_value_of_expression (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 20) :
  Real.sqrt (x + 64) + Real.sqrt (20 - x) + Real.sqrt (2 * x) ≤ Real.sqrt 285.72 :=
by sorry

end max_value_of_expression_l1407_140751


namespace minuend_value_l1407_140766

theorem minuend_value (M S D : ℤ) : 
  M + S + D = 2016 → M - S = D → M = 1008 := by
  sorry

end minuend_value_l1407_140766


namespace tangent_point_on_curve_l1407_140744

theorem tangent_point_on_curve (x y : ℝ) : 
  y = x^4 ∧ (4 : ℝ) * x^3 = 4 → x = 1 ∧ y = 1 :=
by sorry

end tangent_point_on_curve_l1407_140744


namespace dot_product_of_complex_vectors_l1407_140760

def complex_to_vector (z : ℂ) : ℝ × ℝ := (z.re, z.im)

theorem dot_product_of_complex_vectors :
  let Z₁ : ℂ := (1 - 2*I)*I
  let Z₂ : ℂ := (1 - 3*I) / (1 - I)
  let a : ℝ × ℝ := complex_to_vector Z₁
  let b : ℝ × ℝ := complex_to_vector Z₂
  (a.1 * b.1 + a.2 * b.2) = 3 := by sorry

end dot_product_of_complex_vectors_l1407_140760


namespace triangle_side_values_l1407_140774

def triangle_exists (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_side_values :
  ∀ y : ℕ+,
  (triangle_exists 8 11 (y.val ^ 2)) ↔ (y = 2 ∨ y = 3 ∨ y = 4) :=
by sorry

end triangle_side_values_l1407_140774


namespace smallest_three_digit_equal_sum_l1407_140716

/-- Sum of digits function -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- Proposition: 999 is the smallest three-digit number n such that
    Σ(n) = Σ(2n) = Σ(3n) = ... = Σ(n^2), where Σ(n) denotes the sum of the digits of n -/
theorem smallest_three_digit_equal_sum : 
  ∀ n : ℕ, 100 ≤ n → n < 999 → 
  (∃ k : ℕ, 2 ≤ k ∧ k ≤ n ∧ sumOfDigits n ≠ sumOfDigits (k * n)) ∨
  sumOfDigits n ≠ sumOfDigits (n * n) :=
by sorry

#check smallest_three_digit_equal_sum

end smallest_three_digit_equal_sum_l1407_140716


namespace p_plus_q_value_l1407_140709

theorem p_plus_q_value (p q : ℝ) 
  (hp : p^3 - 12*p^2 + 25*p - 75 = 0)
  (hq : 10*q^3 - 75*q^2 - 375*q + 3750 = 0) : 
  p + q = -5/2 := by
sorry

end p_plus_q_value_l1407_140709


namespace laptop_sticker_price_l1407_140723

/-- The sticker price of a laptop --/
def stickerPrice : ℝ := sorry

/-- The price at store A after discount and rebate --/
def priceA : ℝ := 0.82 * stickerPrice - 100

/-- The price at store B after discount --/
def priceB : ℝ := 0.75 * stickerPrice

/-- Theorem stating that the sticker price is $1300 given the conditions --/
theorem laptop_sticker_price : 
  priceB - priceA = 10 → stickerPrice = 1300 := by sorry

end laptop_sticker_price_l1407_140723


namespace largest_multiple_of_8_under_100_l1407_140747

theorem largest_multiple_of_8_under_100 : 
  ∃ n : ℕ, n * 8 = 96 ∧ 
  96 < 100 ∧ 
  ∀ m : ℕ, m * 8 < 100 → m * 8 ≤ 96 :=
by sorry

end largest_multiple_of_8_under_100_l1407_140747


namespace gcd_of_three_numbers_l1407_140719

theorem gcd_of_three_numbers : Nat.gcd 13847 (Nat.gcd 21353 34691) = 5 := by
  sorry

end gcd_of_three_numbers_l1407_140719


namespace cricket_run_rate_theorem_l1407_140727

/-- Represents a cricket game scenario -/
structure CricketGame where
  total_overs : ℕ
  first_part_overs : ℕ
  first_part_run_rate : ℚ
  target_runs : ℕ

/-- Calculates the required run rate for the remaining overs -/
def required_run_rate (game : CricketGame) : ℚ :=
  let remaining_overs := game.total_overs - game.first_part_overs
  let runs_scored := game.first_part_run_rate * game.first_part_overs
  let runs_needed := game.target_runs - runs_scored
  runs_needed / remaining_overs

/-- Theorem stating the required run rate for the given cricket game scenario -/
theorem cricket_run_rate_theorem (game : CricketGame)
  (h1 : game.total_overs = 50)
  (h2 : game.first_part_overs = 10)
  (h3 : game.first_part_run_rate = 3.8)
  (h4 : game.target_runs = 282) :
  required_run_rate game = 6.1 := by
  sorry

#eval required_run_rate {
  total_overs := 50,
  first_part_overs := 10,
  first_part_run_rate := 3.8,
  target_runs := 282
}

end cricket_run_rate_theorem_l1407_140727


namespace inventory_net_change_l1407_140735

/-- Represents the quantity of an ingredient on a given day -/
structure IngredientQuantity where
  day1 : Float
  day7 : Float

/-- Calculates the change in quantity for an ingredient -/
def calculateChange (q : IngredientQuantity) : Float :=
  q.day1 - q.day7

/-- Represents the inventory of all ingredients -/
structure Inventory where
  bakingPowder : IngredientQuantity
  flour : IngredientQuantity
  sugar : IngredientQuantity
  chocolateChips : IngredientQuantity

/-- Calculates the net change for all ingredients -/
def calculateNetChange (inv : Inventory) : Float :=
  calculateChange inv.bakingPowder +
  calculateChange inv.flour +
  calculateChange inv.sugar +
  calculateChange inv.chocolateChips

theorem inventory_net_change (inv : Inventory) 
  (h1 : inv.bakingPowder = { day1 := 4, day7 := 2.5 })
  (h2 : inv.flour = { day1 := 12, day7 := 7 })
  (h3 : inv.sugar = { day1 := 10, day7 := 6.5 })
  (h4 : inv.chocolateChips = { day1 := 6, day7 := 3.7 }) :
  calculateNetChange inv = 12.3 := by
  sorry

end inventory_net_change_l1407_140735


namespace composite_representation_l1407_140708

theorem composite_representation (n : ℕ) (h1 : n > 3) (h2 : ¬ Nat.Prime n) :
  ∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ n = a * b + b * c + c * a + 1 := by
  sorry

end composite_representation_l1407_140708


namespace inequality_proof_l1407_140776

theorem inequality_proof (x y z : ℝ) 
  (sum_zero : x + y + z = 0) 
  (abs_sum_le_one : |x| + |y| + |z| ≤ 1) : 
  x + y/3 + z/5 ≤ 2/5 := by
sorry

end inequality_proof_l1407_140776


namespace factor_tree_root_value_l1407_140722

/-- Represents a node in the factor tree -/
inductive FactorNode
  | Prime (n : Nat)
  | Composite (left right : FactorNode)

/-- Computes the value of a FactorNode -/
def nodeValue : FactorNode → Nat
  | FactorNode.Prime n => n
  | FactorNode.Composite left right => nodeValue left * nodeValue right

/-- The factor tree structure as given in the problem -/
def factorTree : FactorNode :=
  FactorNode.Composite
    (FactorNode.Composite
      (FactorNode.Prime 7)
      (FactorNode.Composite (FactorNode.Prime 7) (FactorNode.Prime 3)))
    (FactorNode.Composite
      (FactorNode.Prime 11)
      (FactorNode.Composite (FactorNode.Prime 11) (FactorNode.Prime 3)))

theorem factor_tree_root_value :
  nodeValue factorTree = 53361 := by
  sorry


end factor_tree_root_value_l1407_140722


namespace equation_solution_l1407_140771

theorem equation_solution (x y z : ℝ) :
  (x - y - 3)^2 + (y - z)^2 + (x - z)^2 = 3 →
  x = z + 1 ∧ y = z - 1 :=
by sorry

end equation_solution_l1407_140771


namespace dihedral_angle_ge_line_angle_l1407_140724

/-- A dihedral angle with its plane angle -/
structure DihedralAngle where
  φ : Real
  φ_nonneg : 0 ≤ φ
  φ_le_pi : φ ≤ π

/-- A line contained in one plane of a dihedral angle -/
structure ContainedLine (d : DihedralAngle) where
  θ : Real
  θ_nonneg : 0 ≤ θ
  θ_le_pi_div_2 : θ ≤ π / 2

/-- The plane angle of a dihedral angle is always greater than or equal to 
    the angle between any line in one of its planes and the other plane -/
theorem dihedral_angle_ge_line_angle (d : DihedralAngle) (l : ContainedLine d) : 
  d.φ ≥ l.θ := by
  sorry

end dihedral_angle_ge_line_angle_l1407_140724


namespace cube_edge_length_l1407_140737

theorem cube_edge_length (a : ℕ) (h1 : a > 0) :
  6 * a^2 = 3 * (12 * a) → a + 2 = 8 := by
  sorry

end cube_edge_length_l1407_140737


namespace boy_scouts_permission_slips_l1407_140764

theorem boy_scouts_permission_slips 
  (total_scouts : ℕ) 
  (total_with_slips : ℝ) 
  (total_boys : ℝ) 
  (girl_scouts_with_slips : ℝ) 
  (h1 : total_with_slips = 0.8 * total_scouts)
  (h2 : total_boys = 0.4 * total_scouts)
  (h3 : girl_scouts_with_slips = 0.8333 * (total_scouts - total_boys)) :
  (total_with_slips - girl_scouts_with_slips) / total_boys = 0.75 := by
sorry

end boy_scouts_permission_slips_l1407_140764


namespace expression_value_l1407_140770

theorem expression_value (x : ℝ) (h : x = 5) : 3 * x + 2 = 17 := by
  sorry

end expression_value_l1407_140770


namespace city_population_growth_l1407_140736

/-- Represents the birth rate and death rate in a city, and proves the birth rate given conditions --/
theorem city_population_growth (death_rate : ℕ) (net_increase : ℕ) (intervals_per_day : ℕ) 
  (h1 : death_rate = 3)
  (h2 : net_increase = 43200)
  (h3 : intervals_per_day = 43200) :
  ∃ (birth_rate : ℕ), 
    birth_rate = 4 ∧ 
    (birth_rate - death_rate) * intervals_per_day = net_increase :=
sorry

end city_population_growth_l1407_140736


namespace tangent_line_intersection_l1407_140706

/-- Given two circles in a 2D plane:
    Circle 1 with radius 3 and center (0, 0)
    Circle 2 with radius 5 and center (12, 0)
    The x-coordinate of the point where a line tangent to both circles
    intersects the x-axis (to the right of the origin) is 9/2. -/
theorem tangent_line_intersection (x : ℝ) : 
  (∃ (y : ℝ), (x^2 + y^2 = 3^2 ∧ ((x - 12)^2 + y^2 = 5^2))) → x = 9/2 :=
sorry

end tangent_line_intersection_l1407_140706


namespace simplify_expression_l1407_140787

theorem simplify_expression (x : ℝ) : 3*x + 5 - 2*x - 6 + 4*x + 7 - 5*x - 9 = -3 := by
  sorry

end simplify_expression_l1407_140787


namespace unique_positive_number_l1407_140780

theorem unique_positive_number : ∃! x : ℝ, x > 0 ∧ x + 17 = 60 * (1/x) := by
  sorry

end unique_positive_number_l1407_140780


namespace units_digit_sum_in_base_7_l1407_140779

/-- The base of the number system we're working in -/
def base : ℕ := 7

/-- Function to get the units digit of a number in the given base -/
def unitsDigit (n : ℕ) : ℕ := n % base

/-- First number in the sum -/
def num1 : ℕ := 52

/-- Second number in the sum -/
def num2 : ℕ := 62

/-- Theorem stating that the units digit of the sum of num1 and num2 in base 7 is 4 -/
theorem units_digit_sum_in_base_7 : 
  unitsDigit (num1 + num2) = 4 := by sorry

end units_digit_sum_in_base_7_l1407_140779


namespace return_time_is_11pm_l1407_140750

structure Journey where
  startTime : Nat
  totalDistance : Nat
  speedLevel : Nat
  speedUphill : Nat
  speedDownhill : Nat
  terrainDistribution : Nat

def calculateReturnTime (j : Journey) : Nat :=
  let oneWayTime := j.terrainDistribution / j.speedLevel +
                    j.terrainDistribution / j.speedUphill +
                    j.terrainDistribution / j.speedDownhill +
                    j.terrainDistribution / j.speedLevel
  let totalTime := 2 * oneWayTime
  j.startTime + totalTime

theorem return_time_is_11pm (j : Journey) 
  (h1 : j.startTime = 15) -- 3 pm in 24-hour format
  (h2 : j.totalDistance = 12)
  (h3 : j.speedLevel = 4)
  (h4 : j.speedUphill = 3)
  (h5 : j.speedDownhill = 6)
  (h6 : j.terrainDistribution = 4) -- Assumption of equal distribution
  : calculateReturnTime j = 23 := by -- 11 pm in 24-hour format
  sorry


end return_time_is_11pm_l1407_140750


namespace geometric_sequence_fifth_term_l1407_140700

/-- Given a geometric sequence a, prove that if a₃ = -9 and a₇ = -1, then a₅ = -3 -/
theorem geometric_sequence_fifth_term 
  (a : ℕ → ℝ) 
  (h_geom : ∀ n : ℕ, a (n + 1) = a n * (a 1 / a 0)) 
  (h_3 : a 3 = -9) 
  (h_7 : a 7 = -1) : 
  a 5 = -3 := by
sorry

end geometric_sequence_fifth_term_l1407_140700


namespace min_value_x_plus_4y_lower_bound_achievable_l1407_140703

theorem min_value_x_plus_4y (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1 / x + 1 / (2 * y) = 1) : 
  x + 4 * y ≥ 3 + 2 * Real.sqrt 2 := by
sorry

theorem lower_bound_achievable : 
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ 1 / x + 1 / (2 * y) = 1 ∧ 
  x + 4 * y = 3 + 2 * Real.sqrt 2 := by
sorry

end min_value_x_plus_4y_lower_bound_achievable_l1407_140703


namespace max_product_price_for_given_conditions_l1407_140702

/-- Represents a company's product line -/
structure ProductLine where
  numProducts : ℕ
  averagePrice : ℝ
  minPrice : ℝ
  numLowPriced : ℕ
  lowPriceThreshold : ℝ

/-- The greatest possible selling price of the most expensive product -/
def maxProductPrice (pl : ProductLine) : ℝ :=
  sorry

/-- Theorem stating the maximum product price for the given conditions -/
theorem max_product_price_for_given_conditions :
  let pl : ProductLine := {
    numProducts := 25,
    averagePrice := 1200,
    minPrice := 400,
    numLowPriced := 10,
    lowPriceThreshold := 1000
  }
  maxProductPrice pl = 12000 := by
  sorry

end max_product_price_for_given_conditions_l1407_140702


namespace order_of_abc_l1407_140707

theorem order_of_abc : 
  let a := Real.log 1.01
  let b := 2 / 201
  let c := Real.sqrt 1.02 - 1
  b < a ∧ a < c := by sorry

end order_of_abc_l1407_140707


namespace function_value_proof_l1407_140717

/-- Given a function f(x, z) = 2x^2 + y - z where f(2, 3) = 100, prove that f(5, 7) = 138 -/
theorem function_value_proof (y : ℝ) : 
  let f : ℝ → ℝ → ℝ := λ x z ↦ 2 * x^2 + y - z
  (f 2 3 = 100) → (f 5 7 = 138) := by
sorry

end function_value_proof_l1407_140717


namespace circle_center_and_radius_l1407_140739

/-- Given a circle with equation x² + y² - 4x + 2y + 2 = 0, 
    its center is at (2, -1) and its radius is √3. -/
theorem circle_center_and_radius :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (2, -1) ∧ 
    radius = Real.sqrt 3 ∧
    ∀ (x y : ℝ), x^2 + y^2 - 4*x + 2*y + 2 = 0 ↔ 
      (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
by sorry

end circle_center_and_radius_l1407_140739


namespace complex_equality_sum_l1407_140758

theorem complex_equality_sum (a b : ℝ) (h : a - 2 * Complex.I = b + a * Complex.I) : a + b = -4 := by
  sorry

end complex_equality_sum_l1407_140758


namespace binomial_product_factorial_l1407_140799

theorem binomial_product_factorial (n : ℕ) : 
  (Nat.choose (n + 2) n) * n.factorial = ((n + 2) * (n + 1) * n.factorial) / 2 := by
  sorry

end binomial_product_factorial_l1407_140799


namespace player_A_best_performance_l1407_140743

structure Player where
  name : String
  average_score : Float
  variance : Float

def players : List Player := [
  ⟨"A", 9.9, 4.2⟩,
  ⟨"B", 9.8, 5.2⟩,
  ⟨"C", 9.9, 5.2⟩,
  ⟨"D", 9.0, 4.2⟩
]

def has_best_performance (p : Player) (ps : List Player) : Prop :=
  ∀ q ∈ ps, p.average_score ≥ q.average_score ∧ 
    (p.average_score > q.average_score ∨ p.variance ≤ q.variance)

theorem player_A_best_performance :
  ∃ p ∈ players, p.name = "A" ∧ has_best_performance p players := by
  sorry

end player_A_best_performance_l1407_140743


namespace unique_g_function_l1407_140797

-- Define the properties of function g
def is_valid_g (g : ℝ → ℝ) : Prop :=
  (∀ x₁ x₂ : ℝ, g (x₁ + x₂) = g x₁ * g x₂) ∧
  (g 1 = 3) ∧
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → g x₁ < g x₂)

-- Theorem statement
theorem unique_g_function :
  ∃! g : ℝ → ℝ, is_valid_g g ∧ (∀ x : ℝ, g x = 3^x) :=
by sorry

end unique_g_function_l1407_140797


namespace circle_passes_through_intersections_l1407_140789

/-- Line l₁ -/
def l₁ (x y : ℝ) : Prop := x - 2*y = 0

/-- Line l₂ -/
def l₂ (x y : ℝ) : Prop := y + 1 = 0

/-- Line l₃ -/
def l₃ (x y : ℝ) : Prop := 2*x + y - 1 = 0

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 + x + 2*y - 1 = 0

/-- Theorem stating that the circle passes through the intersection points of the lines -/
theorem circle_passes_through_intersections :
  ∀ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ),
  (l₁ x₁ y₁ ∧ l₂ x₁ y₁) →
  (l₁ x₂ y₂ ∧ l₃ x₂ y₂) →
  (l₂ x₃ y₃ ∧ l₃ x₃ y₃) →
  circle_equation x₁ y₁ ∧ circle_equation x₂ y₂ ∧ circle_equation x₃ y₃ :=
by sorry

end circle_passes_through_intersections_l1407_140789


namespace invested_sum_is_700_l1407_140795

/-- Represents the simple interest calculation --/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Proves that the invested sum is $700 given the problem conditions --/
theorem invested_sum_is_700 
  (peter_amount : ℝ) 
  (david_amount : ℝ) 
  (peter_time : ℝ) 
  (david_time : ℝ) 
  (h1 : peter_amount = 815)
  (h2 : david_amount = 850)
  (h3 : peter_time = 3)
  (h4 : david_time = 4)
  : ∃ (principal rate : ℝ),
    simple_interest principal rate peter_time = peter_amount ∧
    simple_interest principal rate david_time = david_amount ∧
    principal = 700 := by
  sorry

end invested_sum_is_700_l1407_140795


namespace quadratic_function_values_l1407_140792

theorem quadratic_function_values (p q : ℝ) : ¬ (∀ x ∈ ({1, 2, 3} : Set ℝ), |x^2 + p*x + q| < (1/2 : ℝ)) := by
  sorry

end quadratic_function_values_l1407_140792


namespace sum_of_fractions_l1407_140752

theorem sum_of_fractions : 
  (2 / 10 : ℚ) + (4 / 10 : ℚ) + (6 / 10 : ℚ) + (8 / 10 : ℚ) + (10 / 10 : ℚ) + 
  (12 / 10 : ℚ) + (14 / 10 : ℚ) + (16 / 10 : ℚ) + (18 / 10 : ℚ) + (32 / 10 : ℚ) = 
  (122 / 10 : ℚ) := by
  sorry

end sum_of_fractions_l1407_140752


namespace student_weight_is_75_l1407_140786

/-- The student's weight in kilograms -/
def student_weight : ℝ := sorry

/-- The sister's weight in kilograms -/
def sister_weight : ℝ := sorry

/-- The total weight of the student and his sister is 110 kilograms -/
axiom total_weight : student_weight + sister_weight = 110

/-- If the student loses 5 kilograms, he will weigh twice as much as his sister -/
axiom weight_relation : student_weight - 5 = 2 * sister_weight

/-- The student's present weight is 75 kilograms -/
theorem student_weight_is_75 : student_weight = 75 := by sorry

end student_weight_is_75_l1407_140786


namespace josh_initial_money_l1407_140711

-- Define the given conditions
def spent_on_drink : ℝ := 1.75
def spent_additional : ℝ := 1.25
def money_left : ℝ := 6.00

-- Define the theorem
theorem josh_initial_money :
  ∃ (initial_money : ℝ),
    initial_money = spent_on_drink + spent_additional + money_left ∧
    initial_money = 9.00 := by
  sorry

end josh_initial_money_l1407_140711


namespace units_digit_of_power_difference_l1407_140759

theorem units_digit_of_power_difference : (5^35 - 6^21) % 10 = 9 := by
  sorry

end units_digit_of_power_difference_l1407_140759


namespace tangent_line_to_circle_l1407_140730

/-- A line tangent to a circle and passing through a point -/
theorem tangent_line_to_circle (x y : ℝ) : 
  -- The line equation
  (y - 4 = 3/4 * (x + 3)) →
  -- The line passes through (-3, 4)
  ((-3 : ℝ), (4 : ℝ)) ∈ {(x, y) | y - 4 = 3/4 * (x + 3)} →
  -- The line is tangent to the circle
  (∃! (p : ℝ × ℝ), p ∈ {(x, y) | x^2 + y^2 = 25} ∩ {(x, y) | y - 4 = 3/4 * (x + 3)}) :=
by sorry

end tangent_line_to_circle_l1407_140730


namespace aarti_work_theorem_l1407_140772

/-- Given that Aarti can complete a piece of work in a certain number of days,
    this function calculates how many days she needs to complete a multiple of that work. -/
def days_for_multiple_work (base_days : ℕ) (multiple : ℕ) : ℕ :=
  base_days * multiple

/-- Theorem stating that if Aarti can complete a piece of work in 5 days,
    then she will need 15 days to complete three times the work of the same type. -/
theorem aarti_work_theorem :
  days_for_multiple_work 5 3 = 15 := by sorry

end aarti_work_theorem_l1407_140772


namespace entire_square_shaded_l1407_140734

/-- The fraction of area shaded in the first step -/
def initial_shaded : ℚ := 5 / 9

/-- The fraction of area remaining unshaded after each step -/
def unshaded_fraction : ℚ := 4 / 9

/-- The sum of the infinite geometric series representing the total shaded area -/
def total_shaded_area : ℚ := initial_shaded / (1 - unshaded_fraction)

/-- Theorem stating that the entire square is shaded in the limit -/
theorem entire_square_shaded : total_shaded_area = 1 := by sorry

end entire_square_shaded_l1407_140734


namespace expression_equals_one_l1407_140796

theorem expression_equals_one (p q r : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (h_sum : p + q + r = 0) :
  (p^2 * q^2 / ((p^2 - q*r) * (q^2 - p*r))) +
  (p^2 * r^2 / ((p^2 - q*r) * (r^2 - p*q))) +
  (q^2 * r^2 / ((q^2 - p*r) * (r^2 - p*q))) = 1 := by
  sorry

end expression_equals_one_l1407_140796


namespace score_difference_is_1_25_l1407_140754

def score_distribution : List (Float × Float) :=
  [(0.20, 70), (0.20, 80), (0.25, 85), (0.25, 90), (0.10, 100)]

def median_score : Float := 85

def mean_score : Float :=
  score_distribution.foldl (λ acc (percent, score) => acc + percent * score) 0

theorem score_difference_is_1_25 :
  median_score - mean_score = 1.25 := by sorry

end score_difference_is_1_25_l1407_140754


namespace non_negative_inequality_l1407_140762

theorem non_negative_inequality (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) :
  a^4 + b^4 + c^4 - 2*(a^2*b^2 + a^2*c^2 + b^2*c^2) + a^2*b*c + b^2*a*c + c^2*a*b ≥ 0 := by
  sorry

end non_negative_inequality_l1407_140762


namespace bens_debtor_payment_l1407_140705

/-- Calculates the amount paid by Ben's debtor given his financial transactions -/
theorem bens_debtor_payment (initial_amount cheque_amount maintenance_cost final_amount : ℕ) : 
  initial_amount = 2000 ∧ 
  cheque_amount = 600 ∧ 
  maintenance_cost = 1200 ∧ 
  final_amount = 1000 → 
  final_amount = initial_amount - cheque_amount - maintenance_cost + 800 := by
  sorry

#check bens_debtor_payment

end bens_debtor_payment_l1407_140705


namespace flour_already_added_is_three_l1407_140740

/-- The number of cups of flour required by the recipe -/
def total_flour : ℕ := 9

/-- The number of cups of flour Mary still needs to add -/
def flour_to_add : ℕ := 6

/-- The number of cups of flour Mary has already put in -/
def flour_already_added : ℕ := total_flour - flour_to_add

theorem flour_already_added_is_three : flour_already_added = 3 := by
  sorry

end flour_already_added_is_three_l1407_140740


namespace walking_time_equals_time_saved_l1407_140725

/-- Represents the scenario of a man walking and his wife driving to meet him -/
structure CommuteScenario where
  usual_drive_time : ℝ
  actual_drive_time : ℝ
  time_saved : ℝ
  walking_time : ℝ

/-- Theorem stating that the walking time equals the time saved -/
theorem walking_time_equals_time_saved (scenario : CommuteScenario) 
  (h1 : scenario.usual_drive_time > 0)
  (h2 : scenario.actual_drive_time > 0)
  (h3 : scenario.time_saved > 0)
  (h4 : scenario.walking_time > 0)
  (h5 : scenario.usual_drive_time = scenario.actual_drive_time + scenario.time_saved)
  (h6 : scenario.walking_time = scenario.time_saved) : 
  scenario.walking_time = scenario.time_saved :=
by sorry

end walking_time_equals_time_saved_l1407_140725


namespace triangle_angle_measure_l1407_140733

theorem triangle_angle_measure (A B C : Real) : 
  A + B + C = 180 →
  B = A + 20 →
  C = 50 →
  B = 75 := by sorry

end triangle_angle_measure_l1407_140733


namespace min_container_cost_l1407_140784

/-- Represents the dimensions and cost of a rectangular container -/
structure Container where
  length : ℝ
  width : ℝ
  height : ℝ
  baseUnitCost : ℝ
  sideUnitCost : ℝ

/-- Calculates the total cost of the container -/
def totalCost (c : Container) : ℝ :=
  c.baseUnitCost * c.length * c.width + 
  c.sideUnitCost * 2 * (c.length + c.width) * c.height

/-- Theorem stating the minimum cost of the container -/
theorem min_container_cost :
  ∃ (c : Container),
    c.height = 1 ∧
    c.length * c.width * c.height = 4 ∧
    c.baseUnitCost = 20 ∧
    c.sideUnitCost = 10 ∧
    (∀ (d : Container),
      d.height = 1 →
      d.length * d.width * d.height = 4 →
      d.baseUnitCost = 20 →
      d.sideUnitCost = 10 →
      totalCost c ≤ totalCost d) ∧
    totalCost c = 160 := by
  sorry

end min_container_cost_l1407_140784


namespace every_positive_integer_appears_l1407_140782

/-- The smallest prime that doesn't divide k -/
def p (k : ℕ+) : ℕ := sorry

/-- The sequence a_n -/
def a : ℕ → ℕ+ → ℕ+
  | 0, a₀ => a₀
  | n + 1, a₀ => sorry

/-- Main theorem: every positive integer appears in the sequence -/
theorem every_positive_integer_appears (a₀ : ℕ+) :
  ∀ m : ℕ+, ∃ n : ℕ, a n a₀ = m := by sorry

end every_positive_integer_appears_l1407_140782


namespace multiply_specific_numbers_l1407_140791

theorem multiply_specific_numbers : 469160 * 9999 = 4691183840 := by
  sorry

end multiply_specific_numbers_l1407_140791


namespace even_function_implies_a_zero_l1407_140761

/-- A function f is even if f(x) = f(-x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- The function y = (x + 1)(x - a) -/
def f (a : ℝ) (x : ℝ) : ℝ :=
  (x + 1) * (x - a)

/-- If f(a) is an even function, then a = 0 -/
theorem even_function_implies_a_zero (a : ℝ) :
  IsEven (f a) → a = 0 := by
  sorry

end even_function_implies_a_zero_l1407_140761


namespace marbles_remaining_l1407_140746

theorem marbles_remaining (total : ℕ) (white : ℕ) (removed : ℕ) : 
  total = 50 → 
  white = 20 → 
  removed = 2 * (white - (total - white) / 2) → 
  total - removed = 40 := by
sorry

end marbles_remaining_l1407_140746


namespace greatest_prime_factor_of_210_l1407_140773

theorem greatest_prime_factor_of_210 : ∃ p : ℕ, Nat.Prime p ∧ p ∣ 210 ∧ ∀ q : ℕ, Nat.Prime q → q ∣ 210 → q ≤ p :=
  sorry

end greatest_prime_factor_of_210_l1407_140773


namespace largest_integer_with_remainder_l1407_140704

theorem largest_integer_with_remainder (n : ℕ) : 
  n < 61 ∧ n % 6 = 5 ∧ ∀ m : ℕ, m < 61 ∧ m % 6 = 5 → m ≤ n → n = 59 :=
by sorry

end largest_integer_with_remainder_l1407_140704


namespace arithmetic_mean_problem_l1407_140741

theorem arithmetic_mean_problem (x : ℚ) : 
  ((x + 6) + 18 + 3*x + 12 + (x + 9) + (3*x - 5)) / 6 = 19 → x = 37/4 := by
sorry

end arithmetic_mean_problem_l1407_140741


namespace inequality_always_true_l1407_140793

theorem inequality_always_true : ∀ x : ℝ, (x + 1) * (2 - x) < 4 := by
  sorry

end inequality_always_true_l1407_140793


namespace decreasing_function_k_bound_l1407_140798

/-- The function f(x) = kx³ + 3(k-1)x² - k² + 1 -/
def f (k : ℝ) (x : ℝ) : ℝ := k * x^3 + 3 * (k - 1) * x^2 - k^2 + 1

/-- The derivative of f(x) with respect to x -/
def f_deriv (k : ℝ) (x : ℝ) : ℝ := 3 * k * x^2 + 6 * (k - 1) * x

theorem decreasing_function_k_bound :
  ∀ k : ℝ, (∀ x ∈ Set.Ioo 0 4, f_deriv k x ≤ 0) → k ≤ 1/3 :=
by sorry

end decreasing_function_k_bound_l1407_140798


namespace total_cupcakes_calculation_l1407_140783

/-- The number of cupcakes ordered for each event -/
def cupcakes_per_event : ℝ := 96.0

/-- The number of different children's events -/
def number_of_events : ℝ := 8.0

/-- The total number of cupcakes needed -/
def total_cupcakes : ℝ := cupcakes_per_event * number_of_events

theorem total_cupcakes_calculation : total_cupcakes = 768.0 := by
  sorry

end total_cupcakes_calculation_l1407_140783


namespace fraction_multiplication_l1407_140781

theorem fraction_multiplication : (2 : ℚ) / 5 * (5 : ℚ) / 9 * (1 : ℚ) / 2 = (1 : ℚ) / 9 := by
  sorry

end fraction_multiplication_l1407_140781


namespace restaurant_bill_split_l1407_140794

theorem restaurant_bill_split (num_people : ℕ) (individual_payment : ℚ) (original_bill : ℚ) : 
  num_people = 8 →
  individual_payment = 314.15 →
  original_bill = num_people * individual_payment →
  original_bill = 2513.20 := by
sorry

end restaurant_bill_split_l1407_140794


namespace tulip_probability_l1407_140768

structure FlowerSet where
  roses : ℕ
  tulips : ℕ
  daisies : ℕ
  lilies : ℕ

def total_flowers (fs : FlowerSet) : ℕ :=
  fs.roses + fs.tulips + fs.daisies + fs.lilies

def probability_of_tulip (fs : FlowerSet) : ℚ :=
  fs.tulips / (total_flowers fs)

theorem tulip_probability (fs : FlowerSet) (h : fs = ⟨3, 2, 4, 6⟩) :
  probability_of_tulip fs = 2 / 15 := by
  sorry

end tulip_probability_l1407_140768


namespace bekah_reading_days_l1407_140720

/-- Given the total pages to read, pages already read, and pages to read per day,
    calculate the number of days left to finish reading. -/
def days_left_to_read (total_pages pages_read pages_per_day : ℕ) : ℕ :=
  (total_pages - pages_read) / pages_per_day

/-- Theorem: Given 408 total pages, 113 pages read, and 59 pages per day,
    the number of days left to finish reading is 5. -/
theorem bekah_reading_days : days_left_to_read 408 113 59 = 5 := by
  sorry

#eval days_left_to_read 408 113 59

end bekah_reading_days_l1407_140720


namespace greene_nursery_flower_count_l1407_140712

/-- The number of red roses at Greene Nursery -/
def red_roses : ℕ := 1491

/-- The number of yellow carnations at Greene Nursery -/
def yellow_carnations : ℕ := 3025

/-- The number of white roses at Greene Nursery -/
def white_roses : ℕ := 1768

/-- The number of purple tulips at Greene Nursery -/
def purple_tulips : ℕ := 2150

/-- The number of pink daisies at Greene Nursery -/
def pink_daisies : ℕ := 3500

/-- The number of blue irises at Greene Nursery -/
def blue_irises : ℕ := 2973

/-- The number of orange marigolds at Greene Nursery -/
def orange_marigolds : ℕ := 4234

/-- The total number of flowers at Greene Nursery -/
def total_flowers : ℕ := red_roses + yellow_carnations + white_roses + purple_tulips + 
                          pink_daisies + blue_irises + orange_marigolds

theorem greene_nursery_flower_count : total_flowers = 19141 := by
  sorry

end greene_nursery_flower_count_l1407_140712


namespace heroes_on_front_l1407_140718

theorem heroes_on_front (total : ℕ) (back : ℕ) (front : ℕ) : 
  total = 9 → back = 7 → total = front + back → front = 2 := by
  sorry

end heroes_on_front_l1407_140718


namespace triangle_third_side_l1407_140714

theorem triangle_third_side (a b : ℝ) (h₁ h₂ : ℝ) :
  a = 5 →
  b = 2 * Real.sqrt 6 →
  0 < h₁ →
  0 < h₂ →
  a * h₁ = b * h₂ →
  a + h₁ ≤ b + h₂ →
  ∃ c : ℝ, c * c = a * a + b * b ∧ c = 7 :=
by sorry

end triangle_third_side_l1407_140714


namespace problem_solution_l1407_140729

-- Define the solution set
def SolutionSet (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 4

-- Define the inequality
def Inequality (x m n : ℝ) : Prop := |x - m| ≤ n

theorem problem_solution :
  -- Conditions
  (∀ x, Inequality x m n ↔ SolutionSet x) →
  -- Part 1: Prove m = 2 and n = 2
  (m = 2 ∧ n = 2) ∧
  -- Part 2: Prove minimum value of a + b
  (∀ a b : ℝ, a > 0 → b > 0 → a + b = m/a + n/b → a + b ≥ 2 * Real.sqrt 2) ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a + b = m/a + n/b ∧ a + b = 2 * Real.sqrt 2) :=
by sorry

end problem_solution_l1407_140729


namespace sum_of_abs_coeffs_of_2x_minus_1_to_6th_l1407_140765

theorem sum_of_abs_coeffs_of_2x_minus_1_to_6th (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℤ) :
  (∀ x, (2*x - 1)^6 = a₆*x^6 + a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) →
  (|a₀| + |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| = 729) :=
by sorry

end sum_of_abs_coeffs_of_2x_minus_1_to_6th_l1407_140765


namespace muffin_count_l1407_140778

/-- Given a number of doughnuts and a ratio of doughnuts to muffins, 
    calculate the number of muffins -/
def calculate_muffins (num_doughnuts : ℕ) (doughnut_ratio : ℕ) (muffin_ratio : ℕ) : ℕ :=
  (num_doughnuts / doughnut_ratio) * muffin_ratio

/-- Theorem: Given 50 doughnuts and a ratio of 5 doughnuts to 1 muffin, 
    the number of muffins is 10 -/
theorem muffin_count : calculate_muffins 50 5 1 = 10 := by
  sorry

end muffin_count_l1407_140778


namespace sum_divisors_450_prime_factors_l1407_140749

/-- The sum of positive divisors of a natural number n -/
def sum_of_divisors (n : ℕ) : ℕ := sorry

/-- The number of distinct prime factors of a natural number n -/
def num_distinct_prime_factors (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of the positive divisors of 450 has exactly 3 distinct prime factors -/
theorem sum_divisors_450_prime_factors :
  num_distinct_prime_factors (sum_of_divisors 450) = 3 := by sorry

end sum_divisors_450_prime_factors_l1407_140749


namespace sum_of_largest_and_smallest_prime_factors_of_1365_l1407_140755

theorem sum_of_largest_and_smallest_prime_factors_of_1365 :
  ∃ (p q : ℕ), 
    Nat.Prime p ∧ 
    Nat.Prime q ∧ 
    p ∣ 1365 ∧ 
    q ∣ 1365 ∧ 
    (∀ r : ℕ, Nat.Prime r → r ∣ 1365 → p ≤ r ∧ r ≤ q) ∧ 
    p + q = 16 := by
  sorry

end sum_of_largest_and_smallest_prime_factors_of_1365_l1407_140755


namespace number_difference_l1407_140748

theorem number_difference (L S : ℕ) (hL : L = 1600) (hDiv : L = S * 16 + 15) : L - S = 1501 := by
  sorry

end number_difference_l1407_140748


namespace polynomial_factorization_l1407_140732

theorem polynomial_factorization (x y z : ℝ) :
  x^3 * (y^2 - z^2) + y^3 * (z^2 - x^2) + z^3 * (x^2 - y^2) =
  (x - y) * (y - z) * (z - x) * (x*y + x*z + y*z) := by sorry

end polynomial_factorization_l1407_140732


namespace length_of_BD_l1407_140788

-- Define the triangles and their properties
def right_triangle_ABC (b c : ℝ) : Prop :=
  ∃ (A B C : ℝ × ℝ), 
    (B.1 - A.1) * (C.2 - A.2) = (C.1 - A.1) * (B.2 - A.2) ∧ 
    (C.1 - B.1)^2 + (C.2 - B.2)^2 = b^2 ∧
    (A.1 - C.1)^2 + (A.2 - C.2)^2 = c^2

def right_triangle_ABD (b c : ℝ) : Prop :=
  ∃ (A B D : ℝ × ℝ), 
    (B.1 - A.1) * (D.2 - A.2) = (D.1 - A.1) * (B.2 - A.2) ∧
    (A.1 - D.1)^2 + (A.2 - D.2)^2 = 9 ∧
    (B.1 - A.1)^2 + (B.2 - A.2)^2 = b^2 + c^2

-- The theorem to prove
theorem length_of_BD (b c : ℝ) (h1 : b > 0) (h2 : c > 0) 
  (h3 : right_triangle_ABC b c) (h4 : right_triangle_ABD b c) :
  ∃ (B D : ℝ × ℝ), (B.1 - D.1)^2 + (B.2 - D.2)^2 = b^2 + c^2 - 9 := by
  sorry

end length_of_BD_l1407_140788


namespace inscribed_cylinder_radius_l1407_140790

/-- The radius of a cylinder inscribed in a cone with specific dimensions -/
theorem inscribed_cylinder_radius (cylinder_height : ℝ) (cylinder_radius : ℝ) 
  (cone_diameter : ℝ) (cone_altitude : ℝ) : 
  cylinder_height = 2 * cylinder_radius →
  cone_diameter = 8 →
  cone_altitude = 10 →
  cylinder_radius = 20 / 9 := by
  sorry

end inscribed_cylinder_radius_l1407_140790


namespace shaded_area_of_tiled_floor_l1407_140745

/-- The shaded area of a tiled floor with white quarter circles in each tile corner -/
theorem shaded_area_of_tiled_floor (floor_length floor_width tile_size radius : ℝ)
  (h_floor_length : floor_length = 12)
  (h_floor_width : floor_width = 16)
  (h_tile_size : tile_size = 2)
  (h_radius : radius = 1/2)
  (h_positive : floor_length > 0 ∧ floor_width > 0 ∧ tile_size > 0 ∧ radius > 0) :
  let num_tiles : ℝ := (floor_length * floor_width) / (tile_size * tile_size)
  let white_area_per_tile : ℝ := 4 * π * radius^2
  let shaded_area_per_tile : ℝ := tile_size * tile_size - white_area_per_tile
  num_tiles * shaded_area_per_tile = 192 - 48 * π :=
by sorry

end shaded_area_of_tiled_floor_l1407_140745


namespace total_nailcutter_sounds_l1407_140715

/-- The number of nails per customer -/
def nails_per_customer : ℕ := 20

/-- The number of customers -/
def number_of_customers : ℕ := 3

/-- The number of sounds produced per nail trimmed -/
def sounds_per_nail : ℕ := 1

/-- Theorem: The total number of nailcutter sounds produced for 3 customers is 60 -/
theorem total_nailcutter_sounds :
  nails_per_customer * number_of_customers * sounds_per_nail = 60 := by
  sorry

end total_nailcutter_sounds_l1407_140715


namespace solve_equation_l1407_140775

theorem solve_equation (x : ℝ) : 9 / (5 + 3 / x) = 1 → x = 3/4 := by
  sorry

end solve_equation_l1407_140775


namespace arithmetic_sequence_problem_l1407_140777

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a where a₅ = 3 and a₉ = 6,
    prove that a₁₃ = 9 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ)
    (h_arith : is_arithmetic_sequence a)
    (h_a5 : a 5 = 3)
    (h_a9 : a 9 = 6) :
  a 13 = 9 := by
  sorry

end arithmetic_sequence_problem_l1407_140777


namespace parabola_vertex_l1407_140713

/-- A parabola defined by the equation y^2 + 6y + 2x + 5 = 0 -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 + 6*p.2 + 2*p.1 + 5 = 0}

/-- The vertex of a parabola -/
def vertex (P : Set (ℝ × ℝ)) : ℝ × ℝ := sorry

/-- Theorem stating that the vertex of the given parabola is (2, -3) -/
theorem parabola_vertex : vertex Parabola = (2, -3) := by sorry

end parabola_vertex_l1407_140713


namespace carpet_length_independent_of_steps_carpet_sufficient_l1407_140738

/-- Represents a staircase with its properties --/
structure Staircase :=
  (steps : ℕ)
  (length : ℝ)
  (height : ℝ)

/-- Calculates the length of carpet required for a given staircase --/
def carpet_length (s : Staircase) : ℝ := s.length + s.height

/-- Theorem stating that carpet length depends only on staircase length and height --/
theorem carpet_length_independent_of_steps (s1 s2 : Staircase) :
  s1.length = s2.length → s1.height = s2.height →
  carpet_length s1 = carpet_length s2 := by
  sorry

/-- Specific instance for the problem --/
def staircase1 : Staircase := ⟨9, 2, 2⟩
def staircase2 : Staircase := ⟨10, 2, 2⟩

/-- Theorem stating that the carpet for staircase1 is enough for staircase2 --/
theorem carpet_sufficient : carpet_length staircase1 = carpet_length staircase2 := by
  sorry

end carpet_length_independent_of_steps_carpet_sufficient_l1407_140738


namespace inequality_solution_l1407_140731

/-- A quadratic function f(x) = ax^2 - bx + c where f(x) > 0 for x in (1, 3) -/
def f (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 - b * x + c

/-- The solution set of f(x) > 0 is (1, 3) -/
def f_positive_interval (a b c : ℝ) : Prop :=
  ∀ x, f a b c x > 0 ↔ 1 < x ∧ x < 3

theorem inequality_solution (a b c : ℝ) (h : f_positive_interval a b c) :
  ∀ t : ℝ, f a b c (|t| + 8) < f a b c (2 + t^2) ↔ -3 < t ∧ t < 3 ∧ t ≠ 0 :=
sorry

end inequality_solution_l1407_140731


namespace intersection_point_height_l1407_140742

theorem intersection_point_height (x : ℝ) : x ∈ (Set.Ioo 0 (π/2)) →
  6 * Real.cos x = 5 * Real.tan x →
  ∃ P₁ P₂ : ℝ × ℝ,
    P₁.1 = x ∧ P₁.2 = 0 ∧
    P₂.1 = x ∧ P₂.2 = (1/2) * Real.sin x ∧
    |P₂.2 - P₁.2| = 1/3 := by
  sorry

end intersection_point_height_l1407_140742


namespace sqrt_32_div_sqrt_8_eq_2_l1407_140753

theorem sqrt_32_div_sqrt_8_eq_2 : Real.sqrt 32 / Real.sqrt 8 = 2 := by
  sorry

end sqrt_32_div_sqrt_8_eq_2_l1407_140753


namespace function_always_positive_l1407_140757

-- Define a function f and its derivative f'
variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

-- State that f' is the derivative of f
variable (hf' : ∀ x, HasDerivAt f (f' x) x)

-- State the given condition
variable (h : ∀ x, 2 * f x + x * f' x > x^2)

-- Theorem to prove
theorem function_always_positive : ∀ x, f x > 0 := by sorry

end function_always_positive_l1407_140757


namespace hexagon_angle_sum_l1407_140728

/-- A hexagon is a polygon with six vertices and six edges. -/
structure Hexagon where
  vertices : Fin 6 → ℝ × ℝ

/-- The sum of interior angles of a hexagon in degrees. -/
def sum_of_angles (h : Hexagon) : ℝ := 
  sorry

/-- Theorem: In a hexagon where the sum of all interior angles is 90n degrees, n must equal 4. -/
theorem hexagon_angle_sum (h : Hexagon) (n : ℝ) 
  (h_sum : sum_of_angles h = 90 * n) : n = 4 := by
  sorry

end hexagon_angle_sum_l1407_140728


namespace point_coordinates_l1407_140701

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Determines if a point is in the second quadrant -/
def isSecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- The distance of a point to the x-axis -/
def distanceToXAxis (p : Point) : ℝ :=
  |p.y|

/-- The distance of a point to the y-axis -/
def distanceToYAxis (p : Point) : ℝ :=
  |p.x|

theorem point_coordinates :
  ∀ (p : Point),
    isSecondQuadrant p →
    distanceToXAxis p = 2 →
    distanceToYAxis p = 3 →
    p.x = -3 ∧ p.y = 2 := by
  sorry

end point_coordinates_l1407_140701


namespace divisors_of_pq_divisors_of_p2q_divisors_of_p2q2_divisors_of_pmqn_l1407_140769

-- Define p and q as distinct prime numbers
variable (p q : ℕ) [Fact (Nat.Prime p)] [Fact (Nat.Prime q)] (h : p ≠ q)

-- Define m and n as natural numbers
variable (m n : ℕ)

-- Function to count divisors
noncomputable def countDivisors (n : ℕ) : ℕ := (Nat.divisors n).card

-- Theorems to prove
theorem divisors_of_pq : countDivisors (p * q) = 4 := by sorry

theorem divisors_of_p2q : countDivisors (p^2 * q) = 6 := by sorry

theorem divisors_of_p2q2 : countDivisors (p^2 * q^2) = 9 := by sorry

theorem divisors_of_pmqn : countDivisors (p^m * q^n) = (m + 1) * (n + 1) := by sorry

end divisors_of_pq_divisors_of_p2q_divisors_of_p2q2_divisors_of_pmqn_l1407_140769


namespace s_iff_m_range_p_or_q_and_not_q_implies_m_range_l1407_140710

-- Define propositions p, q, and s
def p (m : ℝ) : Prop := ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (∀ (x y : ℝ), x^2 / (4 - m) + y^2 / m = 1 ↔ x^2 / a^2 + y^2 / b^2 = 1)

def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*m*x + 1 > 0

def s (m : ℝ) : Prop := ∃ x : ℝ, m*x^2 + 2*m*x + 2 = 0

-- Theorem 1
theorem s_iff_m_range (m : ℝ) : s m ↔ m < 0 ∨ m ≥ 2 := by sorry

-- Theorem 2
theorem p_or_q_and_not_q_implies_m_range (m : ℝ) : (p m ∨ q m) ∧ ¬(q m) → 1 ≤ m ∧ m < 2 := by sorry

end s_iff_m_range_p_or_q_and_not_q_implies_m_range_l1407_140710


namespace pi_irrational_among_given_numbers_l1407_140767

theorem pi_irrational_among_given_numbers :
  (∃ (a b : ℤ), (1 : ℝ) / 3 = a / b) ∧
  (∃ (c d : ℤ), (0.201 : ℝ) = c / d) ∧
  (∃ (e f : ℤ), Real.sqrt 9 = e / f) →
  ¬∃ (m n : ℤ), Real.pi = m / n :=
by sorry

end pi_irrational_among_given_numbers_l1407_140767


namespace cube_expansion_sum_l1407_140785

/-- Given that for any real number x, x^3 = a₀ + a₁(x-2) + a₂(x-2)² + a₃(x-2)³, 
    prove that a₁ + a₂ + a₃ = 19 -/
theorem cube_expansion_sum (a₀ a₁ a₂ a₃ : ℝ) 
    (h : ∀ x : ℝ, x^3 = a₀ + a₁*(x-2) + a₂*(x-2)^2 + a₃*(x-2)^3) : 
  a₁ + a₂ + a₃ = 19 := by
  sorry

end cube_expansion_sum_l1407_140785
