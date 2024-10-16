import Mathlib

namespace NUMINAMATH_CALUDE_f_minimum_and_g_inequality_l4001_400120

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x - 1

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := Real.exp x * (a * x + x * Real.cos x + 1)

theorem f_minimum_and_g_inequality :
  (∃! x : ℝ, ∀ y : ℝ, f y ≥ f x) ∧ f 0 = 0 ∧
  ∀ a : ℝ, a > -1 → ∀ x : ℝ, x > 0 ∧ x < 1 → g a x > 1 :=
sorry

end NUMINAMATH_CALUDE_f_minimum_and_g_inequality_l4001_400120


namespace NUMINAMATH_CALUDE_master_bedroom_size_l4001_400114

theorem master_bedroom_size (total_area : ℝ) (common_area : ℝ) (guest_ratio : ℝ) :
  total_area = 2300 →
  common_area = 1000 →
  guest_ratio = 1/4 →
  ∃ (master_size : ℝ),
    master_size = 1040 ∧
    total_area = common_area + master_size + guest_ratio * master_size :=
by
  sorry

end NUMINAMATH_CALUDE_master_bedroom_size_l4001_400114


namespace NUMINAMATH_CALUDE_hyperbola_locus_l4001_400161

/-- The locus of points P satisfying |PM| - |PN| = 4, where M(-3, 0) and N(3, 0) are fixed points -/
def rightBranchHyperbola : Set (ℝ × ℝ) :=
  {P | ‖P - (-3, 0)‖ - ‖P - (3, 0)‖ = 4 ∧ P.1 > 3}

/-- Theorem stating that the locus of points P satisfying |PM| - |PN| = 4 
    is the right branch of a hyperbola with foci M(-3, 0) and N(3, 0) -/
theorem hyperbola_locus :
  ∀ P : ℝ × ℝ, P ∈ rightBranchHyperbola ↔ 
    (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 
      (P.1 / a)^2 - (P.2 / b)^2 = 1 ∧
      a^2 - b^2 = 9 ∧
      P.1 > 3) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_locus_l4001_400161


namespace NUMINAMATH_CALUDE_at_least_one_inequality_holds_l4001_400100

theorem at_least_one_inequality_holds (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y > 2) :
  (1 + x) / y < 2 ∨ (1 + y) / x < 2 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_inequality_holds_l4001_400100


namespace NUMINAMATH_CALUDE_boys_in_class_l4001_400159

theorem boys_in_class (total : ℕ) (diff : ℕ) (boys : ℕ) : 
  total = 345 → diff = 69 → boys + (boys + diff) = total → boys = 138 := by
  sorry

end NUMINAMATH_CALUDE_boys_in_class_l4001_400159


namespace NUMINAMATH_CALUDE_problem_statement_l4001_400165

theorem problem_statement (x y : ℝ) (h1 : x + y = 5) (h2 : x * y = 3) :
  x + x^2 / y + y^2 / x + y = 95 / 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l4001_400165


namespace NUMINAMATH_CALUDE_tree_planting_problem_l4001_400124

theorem tree_planting_problem (n t : ℕ) 
  (h1 : 4 * n = t + 11) 
  (h2 : 2 * n = t - 13) : 
  n = 12 ∧ t = 37 := by
  sorry

end NUMINAMATH_CALUDE_tree_planting_problem_l4001_400124


namespace NUMINAMATH_CALUDE_greatest_common_piece_length_l4001_400198

theorem greatest_common_piece_length : Nat.gcd 42 (Nat.gcd 63 84) = 21 := by
  sorry

end NUMINAMATH_CALUDE_greatest_common_piece_length_l4001_400198


namespace NUMINAMATH_CALUDE_T_is_three_intersecting_lines_l4001_400102

-- Define the set T
def T : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               (5 = x + 3 ∧ 5 ≤ y - 3) ∨
               (5 = y - 3 ∧ 5 ≤ x + 3) ∨
               (x + 3 = y - 3 ∧ 5 ≤ x + 3)}

-- Define the three lines
def line1 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = 2 ∧ p.2 ≥ 8}
def line2 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = 8 ∧ p.1 ≥ 2}
def line3 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1 + 6 ∧ p.1 ≤ 2}

-- Define the intersection points
def point1 : ℝ × ℝ := (2, 8)
def point2 : ℝ × ℝ := (2, 8)
def point3 : ℝ × ℝ := (2, 8)

-- Theorem statement
theorem T_is_three_intersecting_lines :
  T = line1 ∪ line2 ∪ line3 ∧
  (∃ (p1 p2 p3 : ℝ × ℝ), p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧
    p1 ∈ line1 ∩ line2 ∧ p2 ∈ line2 ∩ line3 ∧ p3 ∈ line1 ∩ line3) :=
by sorry

end NUMINAMATH_CALUDE_T_is_three_intersecting_lines_l4001_400102


namespace NUMINAMATH_CALUDE_union_with_complement_l4001_400119

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 2, 3}
def B : Set Nat := {2, 4}

theorem union_with_complement : A ∪ (U \ B) = {1, 2, 3, 5} := by
  sorry

end NUMINAMATH_CALUDE_union_with_complement_l4001_400119


namespace NUMINAMATH_CALUDE_remainder_369975_div_6_l4001_400135

theorem remainder_369975_div_6 : 369975 % 6 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_369975_div_6_l4001_400135


namespace NUMINAMATH_CALUDE_product_mod_seventeen_is_zero_l4001_400176

theorem product_mod_seventeen_is_zero :
  (2001 * 2002 * 2003 * 2004 * 2005 * 2006 * 2007) % 17 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_seventeen_is_zero_l4001_400176


namespace NUMINAMATH_CALUDE_investment_calculation_l4001_400171

/-- Represents the total investment amount in dollars -/
def total_investment : ℝ := 22000

/-- Represents the amount invested at 8% interest rate in dollars -/
def investment_at_8_percent : ℝ := 17000

/-- Represents the total interest earned in dollars -/
def total_interest : ℝ := 1710

/-- Represents the interest rate for the 8% investment -/
def rate_8_percent : ℝ := 0.08

/-- Represents the interest rate for the 7% investment -/
def rate_7_percent : ℝ := 0.07

theorem investment_calculation :
  rate_8_percent * investment_at_8_percent +
  rate_7_percent * (total_investment - investment_at_8_percent) =
  total_interest :=
sorry

end NUMINAMATH_CALUDE_investment_calculation_l4001_400171


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l4001_400177

theorem partial_fraction_decomposition :
  ∀ (A B C : ℝ),
  (∀ x : ℝ, x ≠ 2 ∧ x ≠ 4 →
    5 * x / ((x - 4) * (x - 2)^2) = A / (x - 4) + B / (x - 2) + C / (x - 2)^2) ↔
  A = 5 ∧ B = -5 ∧ C = -5 :=
by sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l4001_400177


namespace NUMINAMATH_CALUDE_investment_interest_l4001_400196

/-- Calculate simple interest --/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem investment_interest (y : ℝ) : 
  simple_interest 3000 (y / 100) 2 = 60 * y := by
  sorry

end NUMINAMATH_CALUDE_investment_interest_l4001_400196


namespace NUMINAMATH_CALUDE_problem_solution_l4001_400187

theorem problem_solution (x : ℝ) (h1 : x ≠ 0) : Real.sqrt ((5 * x) / 7) = x → x = 5 / 7 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l4001_400187


namespace NUMINAMATH_CALUDE_regular_polygon_angle_relation_l4001_400104

theorem regular_polygon_angle_relation (m : ℕ) : m ≥ 3 →
  (120 : ℝ) = 4 * (360 / m) → m = 12 := by sorry

end NUMINAMATH_CALUDE_regular_polygon_angle_relation_l4001_400104


namespace NUMINAMATH_CALUDE_annual_savings_20_over_30_l4001_400166

/-- Represents the internet plans and their costs -/
structure InternetPlan where
  speed : ℕ  -- Speed in Mbps
  monthlyCost : ℕ  -- Monthly cost in dollars

/-- Calculates the annual cost of an internet plan -/
def annualCost (plan : InternetPlan) : ℕ :=
  plan.monthlyCost * 12

/-- Represents Marites' internet plans -/
def marites : {currentPlan : InternetPlan // currentPlan.speed = 10 ∧ currentPlan.monthlyCost = 20} :=
  ⟨⟨10, 20⟩, by simp⟩

/-- The 30 Mbps plan -/
def plan30 : InternetPlan :=
  ⟨30, 2 * marites.val.monthlyCost⟩

/-- The 20 Mbps plan -/
def plan20 : InternetPlan :=
  ⟨20, marites.val.monthlyCost + 10⟩

/-- Theorem: Annual savings when choosing 20 Mbps over 30 Mbps is $120 -/
theorem annual_savings_20_over_30 :
  annualCost plan30 - annualCost plan20 = 120 := by
  sorry

end NUMINAMATH_CALUDE_annual_savings_20_over_30_l4001_400166


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l4001_400168

/-- The eccentricity of a hyperbola with given properties -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let c := Real.sqrt (a^2 + b^2)
  let e := c / a
  let asymptote_slope := b / a
  (e = 2 * asymptote_slope) → e = 2 * Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l4001_400168


namespace NUMINAMATH_CALUDE_domain_all_reals_l4001_400172

theorem domain_all_reals (k : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, y = (3 * k * x^2 - 4 * x + 7) / (-7 * x^2 - 4 * x + k)) ↔ 
  k < -4/7 :=
sorry

end NUMINAMATH_CALUDE_domain_all_reals_l4001_400172


namespace NUMINAMATH_CALUDE_carol_peanuts_l4001_400184

/-- Given that Carol initially collects 2 peanuts and receives 5 more from her father,
    prove that Carol has a total of 7 peanuts. -/
theorem carol_peanuts (initial : ℕ) (received : ℕ) (total : ℕ) : 
  initial = 2 → received = 5 → total = initial + received → total = 7 := by
sorry

end NUMINAMATH_CALUDE_carol_peanuts_l4001_400184


namespace NUMINAMATH_CALUDE_circle_containing_three_points_l4001_400127

theorem circle_containing_three_points 
  (points : Finset (ℝ × ℝ)) 
  (h1 : points.card = 51) 
  (h2 : ∀ p ∈ points, 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1) :
  ∃ (center : ℝ × ℝ), ∃ (contained_points : Finset (ℝ × ℝ)),
    contained_points ⊆ points ∧
    contained_points.card ≥ 3 ∧
    ∀ p ∈ contained_points, Real.sqrt ((p.1 - center.1)^2 + (p.2 - center.2)^2) ≤ 1/7 :=
by
  sorry

end NUMINAMATH_CALUDE_circle_containing_three_points_l4001_400127


namespace NUMINAMATH_CALUDE_quadratic_real_roots_condition_l4001_400157

theorem quadratic_real_roots_condition (k : ℝ) : 
  (∃ x : ℝ, (k - 1) * x^2 + 2 * x + 1 = 0) ↔ (k ≤ 2 ∧ k ≠ 1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_condition_l4001_400157


namespace NUMINAMATH_CALUDE_investment_ratio_l4001_400111

/-- Given two investors P and Q who divide their profit in the ratio 2:3,
    where P invested 30000, prove that Q invested 45000. -/
theorem investment_ratio (p q : ℕ) (profit_ratio : ℚ) :
  profit_ratio = 2 / 3 →
  p = 30000 →
  q * profit_ratio = p * (1 - profit_ratio) →
  q = 45000 := by
sorry

end NUMINAMATH_CALUDE_investment_ratio_l4001_400111


namespace NUMINAMATH_CALUDE_teds_age_l4001_400192

theorem teds_age (s : ℝ) (t : ℝ) (a : ℝ) 
  (h1 : t = 2 * s + 17)
  (h2 : a = s / 2)
  (h3 : t + s + a = 72) : 
  ⌊t⌋ = 48 := by
sorry

end NUMINAMATH_CALUDE_teds_age_l4001_400192


namespace NUMINAMATH_CALUDE_sum_a_b_value_l4001_400143

theorem sum_a_b_value (a b : ℚ) (h1 : 2 * a + 5 * b = 43) (h2 : 8 * a + 2 * b = 50) :
  a + b = 34 / 3 := by
sorry

end NUMINAMATH_CALUDE_sum_a_b_value_l4001_400143


namespace NUMINAMATH_CALUDE_fractional_equation_integer_solution_l4001_400148

theorem fractional_equation_integer_solution (m : ℤ) : 
  (∃ x : ℤ, (m * x - 1) / (x - 2) + 1 / (2 - x) = 2 ∧ x ≠ 2) ↔ 
  (m = 4 ∨ m = 3 ∨ m = 0) :=
sorry

end NUMINAMATH_CALUDE_fractional_equation_integer_solution_l4001_400148


namespace NUMINAMATH_CALUDE_discount_store_purchase_solution_l4001_400158

/-- Represents the purchase scenario at the discount store -/
structure DiscountStorePurchase where
  totalItems : ℕ
  itemsAt9Yuan : ℕ
  totalCost : ℕ

/-- Theorem stating the number of items priced at 9 yuan -/
theorem discount_store_purchase_solution :
  ∀ (purchase : DiscountStorePurchase),
    purchase.totalItems % 2 = 0 ∧
    purchase.totalCost = 172 ∧
    purchase.totalCost = 8 * (purchase.totalItems - purchase.itemsAt9Yuan) + 9 * purchase.itemsAt9Yuan →
    purchase.itemsAt9Yuan = 12 := by
  sorry

end NUMINAMATH_CALUDE_discount_store_purchase_solution_l4001_400158


namespace NUMINAMATH_CALUDE_modular_equation_solution_l4001_400183

theorem modular_equation_solution : ∃ (n : ℤ), 0 ≤ n ∧ n < 144 ∧ (143 * n) % 144 = 105 % 144 ∧ n = 39 := by
  sorry

end NUMINAMATH_CALUDE_modular_equation_solution_l4001_400183


namespace NUMINAMATH_CALUDE_triangular_sum_perfect_squares_l4001_400193

def triangular_sum (K : ℕ) : ℕ := K * (K + 1) / 2

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem triangular_sum_perfect_squares :
  {K : ℕ | K > 0 ∧ K < 50 ∧ is_perfect_square (triangular_sum K)} = {1, 8, 49} := by
sorry

end NUMINAMATH_CALUDE_triangular_sum_perfect_squares_l4001_400193


namespace NUMINAMATH_CALUDE_speaking_orders_count_l4001_400149

/-- The number of students in the class --/
def totalStudents : ℕ := 7

/-- The number of students to be selected for speaking --/
def selectedSpeakers : ℕ := 4

/-- Function to calculate the number of speaking orders --/
def speakingOrders (total : ℕ) (selected : ℕ) : ℕ :=
  sorry

/-- Theorem stating the number of speaking orders under given conditions --/
theorem speaking_orders_count :
  speakingOrders totalStudents selectedSpeakers = 600 :=
sorry

end NUMINAMATH_CALUDE_speaking_orders_count_l4001_400149


namespace NUMINAMATH_CALUDE_inequality_system_solution_l4001_400164

theorem inequality_system_solution (a : ℝ) : 
  (∀ x : ℝ, (4 + x) / 3 > (x + 2) / 2 ∧ (x + a) / 2 < 0 ↔ x < 2) →
  a ≤ -2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l4001_400164


namespace NUMINAMATH_CALUDE_multiple_of_seven_problem_l4001_400126

theorem multiple_of_seven_problem (start : Nat) (count : Nat) (result : Nat) : 
  start = 21 → count = 47 → result = 329 → 
  ∃ (n : Nat), result = start + 7 * (count - 1) ∧ result % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_multiple_of_seven_problem_l4001_400126


namespace NUMINAMATH_CALUDE_linear_systems_solutions_l4001_400134

theorem linear_systems_solutions :
  -- System (1)
  let system1 : ℝ × ℝ → Prop := λ (x, y) ↦ 3 * x - 2 * y = 9 ∧ 2 * x + 3 * y = 19
  -- System (2)
  let system2 : ℝ × ℝ → Prop := λ (x, y) ↦ (2 * x + 1) / 5 - 1 = (y - 1) / 3 ∧ 2 * (y - x) - 3 * (1 - y) = 6
  -- Solutions
  let solution1 : ℝ × ℝ := (5, 3)
  let solution2 : ℝ × ℝ := (4, 17/5)
  -- Proof statements
  system1 solution1 ∧ system2 solution2 := by sorry

end NUMINAMATH_CALUDE_linear_systems_solutions_l4001_400134


namespace NUMINAMATH_CALUDE_sqrt_200_equals_10_l4001_400132

theorem sqrt_200_equals_10 : Real.sqrt 200 = 10 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_200_equals_10_l4001_400132


namespace NUMINAMATH_CALUDE_starting_lineup_count_l4001_400175

def total_members : ℕ := 12
def offensive_linemen : ℕ := 4
def quick_reflex_players : ℕ := 2

def starting_lineup_combinations : ℕ := offensive_linemen * quick_reflex_players * 1 * (total_members - 3)

theorem starting_lineup_count : starting_lineup_combinations = 72 := by
  sorry

end NUMINAMATH_CALUDE_starting_lineup_count_l4001_400175


namespace NUMINAMATH_CALUDE_san_francisco_super_bowl_probability_l4001_400150

theorem san_francisco_super_bowl_probability 
  (p_play : ℝ) 
  (p_not_play : ℝ) 
  (h1 : p_play = 9 * p_not_play) 
  (h2 : p_play + p_not_play = 1) : 
  p_play = 0.9 := by
sorry

end NUMINAMATH_CALUDE_san_francisco_super_bowl_probability_l4001_400150


namespace NUMINAMATH_CALUDE_cricketer_average_score_l4001_400181

theorem cricketer_average_score 
  (total_matches : ℕ) 
  (first_set_matches : ℕ) 
  (second_set_matches : ℕ) 
  (first_set_average : ℝ) 
  (second_set_average : ℝ) 
  (h1 : total_matches = first_set_matches + second_set_matches)
  (h2 : total_matches = 5)
  (h3 : first_set_matches = 2)
  (h4 : second_set_matches = 3)
  (h5 : first_set_average = 60)
  (h6 : second_set_average = 50) :
  (first_set_matches * first_set_average + second_set_matches * second_set_average) / total_matches = 54 := by
  sorry

end NUMINAMATH_CALUDE_cricketer_average_score_l4001_400181


namespace NUMINAMATH_CALUDE_pure_imaginary_solutions_l4001_400174

def polynomial (x : ℂ) : ℂ := x^4 - 4*x^3 + 10*x^2 - 64*x - 100

theorem pure_imaginary_solutions :
  ∀ x : ℂ, polynomial x = 0 ∧ ∃ k : ℝ, x = k * I ↔ x = 4 * I ∨ x = -4 * I :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_solutions_l4001_400174


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l4001_400136

theorem equal_roots_quadratic (k C : ℝ) : 
  (∀ x : ℝ, 2 * k * x^2 + 4 * k * x + C = 0) →
  (∃! r : ℝ, 2 * x^2 + 4 * x + C = 0) →
  C = 2 :=
by sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l4001_400136


namespace NUMINAMATH_CALUDE_pineapple_purchase_l4001_400105

/-- The number of pineapples bought by Steve and Georgia -/
def num_pineapples : ℕ := 12

/-- The cost of each pineapple in dollars -/
def cost_per_pineapple : ℚ := 5/4

/-- The shipping cost in dollars -/
def shipping_cost : ℚ := 21

/-- The total cost per pineapple (including shipping) in dollars -/
def total_cost_per_pineapple : ℚ := 3

theorem pineapple_purchase :
  (↑num_pineapples * cost_per_pineapple + shipping_cost) / ↑num_pineapples = total_cost_per_pineapple :=
sorry

end NUMINAMATH_CALUDE_pineapple_purchase_l4001_400105


namespace NUMINAMATH_CALUDE_circle_center_sum_l4001_400133

/-- For a circle with equation x^2 + y^2 = 6x + 8y - 15, if (h, k) is its center, then h + k = 7 -/
theorem circle_center_sum (h k : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 = 6*x + 8*y - 15 ↔ (x - h)^2 + (y - k)^2 = (h^2 + k^2 - (6*h + 8*k - 15))) →
  h + k = 7 := by
sorry

end NUMINAMATH_CALUDE_circle_center_sum_l4001_400133


namespace NUMINAMATH_CALUDE_min_value_of_f_inequality_theorem_l4001_400113

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |x + 2|

-- Theorem for the minimum value of f
theorem min_value_of_f : ∃ (p : ℝ), ∀ (x : ℝ), f x ≥ p ∧ ∃ (x₀ : ℝ), f x₀ = p :=
  sorry

-- Theorem for the inequality
theorem inequality_theorem (a b c : ℝ) (h : a^2 + 2*b^2 + 3*c^2 = 6) :
  |a + 2*b + 3*c| ≤ 6 :=
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_inequality_theorem_l4001_400113


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l4001_400178

/-- The line passing through a fixed point for all real values of a -/
def line (a x y : ℝ) : Prop := (a - 1) * x + a * y + 3 = 0

/-- The fixed point through which the line passes -/
def fixed_point : ℝ × ℝ := (3, -3)

/-- Theorem stating that the fixed point lies on the line for all real a -/
theorem fixed_point_on_line :
  ∀ a : ℝ, line a (fixed_point.1) (fixed_point.2) := by
sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l4001_400178


namespace NUMINAMATH_CALUDE_range_of_m_for_nonempty_solution_l4001_400199

theorem range_of_m_for_nonempty_solution (m : ℝ) : 
  (∃ x : ℝ, |x - 1| + |x + m| ≤ 4) → m ∈ Set.Icc (-5) 3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_for_nonempty_solution_l4001_400199


namespace NUMINAMATH_CALUDE_trivia_team_scoring_l4001_400145

/-- Trivia team scoring problem -/
theorem trivia_team_scoring
  (total_members : ℕ)
  (absent_members : ℕ)
  (total_points : ℕ)
  (h1 : total_members = 5)
  (h2 : absent_members = 2)
  (h3 : total_points = 18)
  : (total_points / (total_members - absent_members) = 6) :=
by
  sorry

#check trivia_team_scoring

end NUMINAMATH_CALUDE_trivia_team_scoring_l4001_400145


namespace NUMINAMATH_CALUDE_fair_distribution_result_l4001_400189

/-- Represents the fair distribution of talers in the bread-sharing scenario -/
def fair_distribution (loaves1 loaves2 : ℕ) (total_talers : ℕ) : ℕ × ℕ :=
  let total_loaves := loaves1 + loaves2
  let loaves_per_person := total_loaves / 3
  let talers_per_loaf := total_talers / loaves_per_person
  let remaining_loaves1 := loaves1 - loaves_per_person
  let remaining_loaves2 := loaves2 - loaves_per_person
  let talers1 := remaining_loaves1 * talers_per_loaf
  let talers2 := remaining_loaves2 * talers_per_loaf
  (talers1, talers2)

/-- The fair distribution of talers in the given scenario is (1, 7) -/
theorem fair_distribution_result :
  fair_distribution 3 5 8 = (1, 7) := by
  sorry

end NUMINAMATH_CALUDE_fair_distribution_result_l4001_400189


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l4001_400180

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Symmetry with respect to the y-axis -/
def symmetricYAxis (p q : Point) : Prop :=
  q.x = -p.x ∧ q.y = p.y

theorem symmetric_points_sum (a b : ℝ) :
  let p : Point := ⟨-2, 3⟩
  let q : Point := ⟨a, b⟩
  symmetricYAxis p q → a + b = 5 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l4001_400180


namespace NUMINAMATH_CALUDE_physics_marks_l4001_400129

theorem physics_marks (P C M : ℝ) 
  (avg_all : (P + C + M) / 3 = 60)
  (avg_pm : (P + M) / 2 = 90)
  (avg_pc : (P + C) / 2 = 70) :
  P = 140 := by
sorry

end NUMINAMATH_CALUDE_physics_marks_l4001_400129


namespace NUMINAMATH_CALUDE_field_division_l4001_400188

theorem field_division (total_area smaller_area : ℝ) (h1 : total_area = 500) (h2 : smaller_area = 225) :
  ∃ (larger_area difference_value : ℝ),
    larger_area + smaller_area = total_area ∧
    larger_area - smaller_area = difference_value / 5 ∧
    difference_value = 250 :=
by sorry

end NUMINAMATH_CALUDE_field_division_l4001_400188


namespace NUMINAMATH_CALUDE_two_digit_seven_times_sum_of_digits_l4001_400155

theorem two_digit_seven_times_sum_of_digits : 
  (∃! (s : Finset Nat), 
    (∀ n ∈ s, 10 ≤ n ∧ n < 100 ∧ n = 7 * (n / 10 + n % 10)) ∧ 
    Finset.card s = 4) := by
  sorry

end NUMINAMATH_CALUDE_two_digit_seven_times_sum_of_digits_l4001_400155


namespace NUMINAMATH_CALUDE_characterization_of_divisibility_implication_l4001_400107

theorem characterization_of_divisibility_implication (n : ℕ) (hn : n ≥ 1) :
  (∀ a b : ℕ, (11 ∣ a^n + b^n) → (11 ∣ a ∧ 11 ∣ b)) ↔ Even n :=
by sorry

end NUMINAMATH_CALUDE_characterization_of_divisibility_implication_l4001_400107


namespace NUMINAMATH_CALUDE_zero_not_in_2_16_l4001_400167

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the property of having only one zero
def has_unique_zero (f : ℝ → ℝ) : Prop :=
  ∃! x, f x = 0

-- Define the property of a zero being within an interval
def zero_in_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x, a < x ∧ x < b ∧ f x = 0

-- Theorem statement
theorem zero_not_in_2_16 (h1 : has_unique_zero f)
  (h2 : zero_in_interval f 0 16)
  (h3 : zero_in_interval f 0 8)
  (h4 : zero_in_interval f 0 4)
  (h5 : zero_in_interval f 0 2) :
  ¬∃ x, 2 < x ∧ x < 16 ∧ f x = 0 :=
by sorry

end NUMINAMATH_CALUDE_zero_not_in_2_16_l4001_400167


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l4001_400147

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x : ℝ, x^2 < 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l4001_400147


namespace NUMINAMATH_CALUDE_multiply_2a_3a_l4001_400195

theorem multiply_2a_3a (a : ℝ) : 2 * a * (3 * a) = 6 * a^2 := by sorry

end NUMINAMATH_CALUDE_multiply_2a_3a_l4001_400195


namespace NUMINAMATH_CALUDE_farmer_brown_additional_cost_farmer_brown_specific_additional_cost_l4001_400131

/-- The additional cost for Farmer Brown's new hay requirements -/
theorem farmer_brown_additional_cost 
  (original_bales : ℕ) 
  (multiplier : ℕ) 
  (original_cost_per_bale : ℕ) 
  (premium_cost_per_bale : ℕ) : ℕ :=
  let new_bales := original_bales * multiplier
  let original_total_cost := original_bales * original_cost_per_bale
  let new_total_cost := new_bales * premium_cost_per_bale
  new_total_cost - original_total_cost

/-- The additional cost for Farmer Brown's specific hay requirements is $3500 -/
theorem farmer_brown_specific_additional_cost :
  farmer_brown_additional_cost 20 5 25 40 = 3500 := by
  sorry

end NUMINAMATH_CALUDE_farmer_brown_additional_cost_farmer_brown_specific_additional_cost_l4001_400131


namespace NUMINAMATH_CALUDE_special_isosceles_triangle_angles_l4001_400154

/-- An isosceles triangle with a special angle bisector property -/
structure SpecialIsoscelesTriangle where
  -- The base angles of the isosceles triangle
  base_angle : ℝ
  -- The angle between the angle bisector from the vertex and the angle bisector to the lateral side
  bisector_angle : ℝ
  -- The condition that the bisector angle equals the vertex angle
  h_bisector_eq_vertex : bisector_angle = 180 - 2 * base_angle

/-- The possible angles of a special isosceles triangle -/
def special_triangle_angles (t : SpecialIsoscelesTriangle) : Prop :=
  (t.base_angle = 36 ∧ 180 - 2 * t.base_angle = 108) ∨
  (t.base_angle = 60 ∧ 180 - 2 * t.base_angle = 60)

/-- Theorem: The angles of a special isosceles triangle are either (36°, 36°, 108°) or (60°, 60°, 60°) -/
theorem special_isosceles_triangle_angles (t : SpecialIsoscelesTriangle) :
  special_triangle_angles t := by
  sorry

end NUMINAMATH_CALUDE_special_isosceles_triangle_angles_l4001_400154


namespace NUMINAMATH_CALUDE_six_balls_three_boxes_l4001_400141

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distributeBalls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 122 ways to distribute 6 distinguishable balls into 3 indistinguishable boxes -/
theorem six_balls_three_boxes : distributeBalls 6 3 = 122 := by sorry

end NUMINAMATH_CALUDE_six_balls_three_boxes_l4001_400141


namespace NUMINAMATH_CALUDE_buoy_radius_l4001_400140

/-- The radius of a spherical buoy given the dimensions of the hole it leaves in ice --/
theorem buoy_radius (hole_diameter : ℝ) (hole_depth : ℝ) (buoy_radius : ℝ) : 
  hole_diameter = 30 → hole_depth = 12 → buoy_radius = 15.375 := by
  sorry

#check buoy_radius

end NUMINAMATH_CALUDE_buoy_radius_l4001_400140


namespace NUMINAMATH_CALUDE_complex_subtraction_magnitude_l4001_400116

theorem complex_subtraction_magnitude : 
  Complex.abs ((3 - 10 * Complex.I) - (2 + 5 * Complex.I)) = Real.sqrt 26 := by
  sorry

end NUMINAMATH_CALUDE_complex_subtraction_magnitude_l4001_400116


namespace NUMINAMATH_CALUDE_count_non_consecutive_digits_999999_l4001_400138

/-- Counts integers from 0 to n without consecutive identical digits -/
def countNonConsecutiveDigits (n : ℕ) : ℕ :=
  sorry

/-- The sum of geometric series 9^1 + 9^2 + ... + 9^6 -/
def geometricSum : ℕ :=
  sorry

theorem count_non_consecutive_digits_999999 :
  countNonConsecutiveDigits 999999 = 597880 := by
  sorry

end NUMINAMATH_CALUDE_count_non_consecutive_digits_999999_l4001_400138


namespace NUMINAMATH_CALUDE_divisibility_by_thirteen_l4001_400179

theorem divisibility_by_thirteen (a b c : ℤ) (h : 13 ∣ (a + b + c)) :
  13 ∣ (a^2007 + b^2007 + c^2007 + 2 * 2007 * a * b * c) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_thirteen_l4001_400179


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l4001_400115

/-- The equation of the asymptote of a hyperbola -/
def asymptote_equation (a b : ℝ) : Set (ℝ × ℝ) :=
  {(x, y) | y = (b / a) * x ∨ y = -(b / a) * x}

/-- The focal length of a hyperbola -/
def focal_length (c : ℝ) : ℝ := 2 * c

theorem hyperbola_asymptote (b : ℝ) (h1 : b > 0) :
  let C : Set (ℝ × ℝ) := {(x, y) | x^2 / 16 - y^2 / b^2 = 1}
  let f : ℝ := focal_length 5
  asymptote_equation 4 3 = {(x, y) | y = (3 / 4) * x ∨ y = -(3 / 4) * x} :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l4001_400115


namespace NUMINAMATH_CALUDE_english_spanish_difference_l4001_400186

def hours_english : ℕ := 7
def hours_chinese : ℕ := 2
def hours_spanish : ℕ := 4

theorem english_spanish_difference : hours_english - hours_spanish = 3 := by
  sorry

end NUMINAMATH_CALUDE_english_spanish_difference_l4001_400186


namespace NUMINAMATH_CALUDE_function_passes_through_point_l4001_400125

theorem function_passes_through_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^(x - 1) + 2
  f 1 = 3 := by sorry

end NUMINAMATH_CALUDE_function_passes_through_point_l4001_400125


namespace NUMINAMATH_CALUDE_not_p_or_q_l4001_400194

-- Define proposition p
def p : Prop := ∀ x : ℝ, x^2 + x - 1 > 0

-- Define proposition q
def q : Prop := ∃ x : ℝ, (2 : ℝ)^x > (3 : ℝ)^x

-- Theorem to prove
theorem not_p_or_q : (¬p) ∨ q := by sorry

end NUMINAMATH_CALUDE_not_p_or_q_l4001_400194


namespace NUMINAMATH_CALUDE_vector_linear_combination_l4001_400112

/-- Given vectors a, b, and c in ℝ², prove that c = 2a - b -/
theorem vector_linear_combination (a b c : ℝ × ℝ) 
  (ha : a = (1, 2)) 
  (hb : b = (-2, 3)) 
  (hc : c = (4, 1)) : 
  c = 2 • a - b := by sorry

end NUMINAMATH_CALUDE_vector_linear_combination_l4001_400112


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l4001_400152

-- Define the sets M and N
def M : Set ℝ := {x | |x - 1| < 2}
def N : Set ℝ := {x | x * (x - 3) < 0}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {x | 0 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l4001_400152


namespace NUMINAMATH_CALUDE_f_max_min_sum_l4001_400101

noncomputable def f (x : ℝ) : ℝ :=
  ((Real.sqrt 1008 * x + Real.sqrt 1009)^2 + Real.sin (2018 * x)) / (2016 * x^2 + 2018)

def has_max_min (f : ℝ → ℝ) (M m : ℝ) : Prop :=
  (∀ x, f x ≤ M) ∧ (∃ x, f x = M) ∧ (∀ x, m ≤ f x) ∧ (∃ x, f x = m)

theorem f_max_min_sum :
  ∃ M m : ℝ, has_max_min f M m ∧ M + m = 1 :=
sorry

end NUMINAMATH_CALUDE_f_max_min_sum_l4001_400101


namespace NUMINAMATH_CALUDE_min_participants_in_tournament_l4001_400110

theorem min_participants_in_tournament : ∃ (n : ℕ) (k : ℕ),
  n = 11 ∧
  k < n / 2 ∧
  k > (45 * n) / 100 ∧
  ∀ (m : ℕ) (j : ℕ), m < n →
    (j < m / 2 ∧ j > (45 * m) / 100) → False :=
by sorry

end NUMINAMATH_CALUDE_min_participants_in_tournament_l4001_400110


namespace NUMINAMATH_CALUDE_total_cost_calculation_l4001_400191

def dog_cost : ℕ := 60
def cat_cost : ℕ := 40
def num_dogs : ℕ := 20
def num_cats : ℕ := 60

theorem total_cost_calculation : 
  dog_cost * num_dogs + cat_cost * num_cats = 3600 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_calculation_l4001_400191


namespace NUMINAMATH_CALUDE_complex_product_real_l4001_400153

theorem complex_product_real (b : ℝ) : 
  (Complex.I : ℂ) ^ 2 = -1 →
  (∃ (r : ℝ), (2 + Complex.I) * (b + Complex.I) = r) →
  b = -2 := by
sorry

end NUMINAMATH_CALUDE_complex_product_real_l4001_400153


namespace NUMINAMATH_CALUDE_abs_sum_zero_implies_sum_l4001_400142

theorem abs_sum_zero_implies_sum (x y : ℝ) :
  |x - 1| + |y + 3| = 0 → x + y = -2 := by
sorry

end NUMINAMATH_CALUDE_abs_sum_zero_implies_sum_l4001_400142


namespace NUMINAMATH_CALUDE_prime_divisibility_problem_l4001_400182

theorem prime_divisibility_problem (p : ℕ) (x : ℕ) (hp : Prime p) :
  (1 ≤ x ∧ x ≤ 2 * p) →
  (x^(p - 1) ∣ (p - 1)^x + 1) →
  ((p = 2 ∧ (x = 1 ∨ x = 2)) ∨ (p = 3 ∧ (x = 1 ∨ x = 3)) ∨ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_prime_divisibility_problem_l4001_400182


namespace NUMINAMATH_CALUDE_first_brother_is_treljalya_l4001_400103

structure Brother where
  name : String
  card_color : String
  tells_truth : Bool

def first_brother_statement_1 (b : Brother) : Prop :=
  b.name = "Treljalya"

def second_brother_statement (b : Brother) : Prop :=
  b.name = "Treljalya"

def first_brother_statement_2 (b : Brother) : Prop :=
  b.card_color = "orange"

def same_suit_rule (b1 b2 : Brother) : Prop :=
  b1.card_color = b2.card_color → b1.tells_truth ≠ b2.tells_truth

def different_suit_rule (b1 b2 : Brother) : Prop :=
  b1.card_color ≠ b2.card_color → b1.tells_truth = b2.tells_truth

theorem first_brother_is_treljalya (b1 b2 : Brother) :
  same_suit_rule b1 b2 →
  different_suit_rule b1 b2 →
  first_brother_statement_1 b1 →
  second_brother_statement b2 →
  first_brother_statement_2 b2 →
  b1.name = "Treljalya" :=
sorry

end NUMINAMATH_CALUDE_first_brother_is_treljalya_l4001_400103


namespace NUMINAMATH_CALUDE_container_volume_increase_l4001_400128

theorem container_volume_increase (original_volume : ℝ) (scale_factor : ℝ) : 
  original_volume = 5 → 
  scale_factor = 4 → 
  (scale_factor ^ 3) * original_volume = 320 := by
sorry

end NUMINAMATH_CALUDE_container_volume_increase_l4001_400128


namespace NUMINAMATH_CALUDE_quadratic_roots_contradiction_l4001_400146

theorem quadratic_roots_contradiction (a : ℝ) : 
  a ≥ 1 → ¬(∃ x y : ℝ, x ≠ y ∧ x^2 - 2*x + a = 0 ∧ y^2 - 2*y + a = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_quadratic_roots_contradiction_l4001_400146


namespace NUMINAMATH_CALUDE_valid_square_root_expression_l4001_400151

theorem valid_square_root_expression (a b : ℝ) : 
  (Real.sqrt (-a^2 * b^2) = -a * b) ↔ (a * b = 0) := by sorry

end NUMINAMATH_CALUDE_valid_square_root_expression_l4001_400151


namespace NUMINAMATH_CALUDE_binary_calculation_theorem_l4001_400109

/-- Represents a binary number as a list of bits (least significant bit first) -/
def BinaryNum := List Bool

/-- Converts a binary number to its decimal representation -/
def binary_to_decimal (b : BinaryNum) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

/-- Converts a decimal number to its binary representation -/
def decimal_to_binary (n : ℕ) : BinaryNum :=
  if n = 0 then [false] else
  let rec aux (m : ℕ) : BinaryNum :=
    if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
  aux n

/-- Multiplies two binary numbers -/
def binary_multiply (a b : BinaryNum) : BinaryNum :=
  decimal_to_binary (binary_to_decimal a * binary_to_decimal b)

/-- Divides a binary number by 2^n (equivalent to right shift by n) -/
def binary_divide_by_power_of_two (b : BinaryNum) (n : ℕ) : BinaryNum :=
  decimal_to_binary (binary_to_decimal b / 2^n)

theorem binary_calculation_theorem :
  let a : BinaryNum := [false, true, false, true, false, false, true, true]  -- 11001010₂
  let b : BinaryNum := [false, true, false, true, true]                      -- 11010₂
  let divisor : BinaryNum := [false, false, true]                            -- 100₂
  binary_divide_by_power_of_two (binary_multiply a b) 2 =
  [false, false, true, false, true, true, true, true, false, false]          -- 1001110100₂
  := by sorry

end NUMINAMATH_CALUDE_binary_calculation_theorem_l4001_400109


namespace NUMINAMATH_CALUDE_team_a_construction_team_b_construction_l4001_400173

-- Define the parameters
def total_length_1 : ℝ := 600
def initial_days : ℝ := 5
def additional_days : ℝ := 2
def daily_increase : ℝ := 20
def total_length_2 : ℝ := 1800
def team_b_initial : ℝ := 360
def team_b_increase : ℝ := 0.2

-- Define Team A's daily construction after increase
def team_a_daily (x : ℝ) : ℝ := x

-- Define Team B's daily construction after increase
def team_b_daily (m : ℝ) : ℝ := m * (1 + team_b_increase)

-- Theorem for Team A's daily construction
theorem team_a_construction :
  ∃ x : ℝ, initial_days * (team_a_daily x - daily_increase) + additional_days * team_a_daily x = total_length_1 ∧
  team_a_daily x = 100 := by sorry

-- Theorem for Team B's original daily construction
theorem team_b_construction :
  ∃ m : ℝ, team_b_initial / m + (total_length_2 / 2 - team_b_initial) / (team_b_daily m) = total_length_2 / 2 / 100 ∧
  m = 90 := by sorry

end NUMINAMATH_CALUDE_team_a_construction_team_b_construction_l4001_400173


namespace NUMINAMATH_CALUDE_f_monotonicity_and_inequality_l4001_400169

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x - 2 * Real.exp 1 * Real.log x

theorem f_monotonicity_and_inequality :
  (∀ x ∈ Set.Ioo 0 1, ∀ y ∈ Set.Ioo 0 1, x < y → f x > f y) ∧
  (∀ x ∈ Set.Ioi 1, ∀ y ∈ Set.Ioi 1, x < y → f x < f y) ∧
  (∀ b ≤ Real.exp 1, ∀ x > 0, f x ≥ b * (x^2 - 2*x + 2)) :=
by sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_inequality_l4001_400169


namespace NUMINAMATH_CALUDE_ending_number_proof_l4001_400162

theorem ending_number_proof (n : ℕ) (h1 : n > 45) (h2 : ∃ (evens : List ℕ), 
  evens.length = 30 ∧ 
  (∀ m ∈ evens, Even m ∧ m > 45 ∧ m ≤ n) ∧
  (∀ m, 45 < m ∧ m ≤ n ∧ Even m → m ∈ evens)) : 
  n = 104 := by
sorry

end NUMINAMATH_CALUDE_ending_number_proof_l4001_400162


namespace NUMINAMATH_CALUDE_simplify_expression_l4001_400137

theorem simplify_expression (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (4 * a^3 * b - 2 * a * b) / (2 * a * b) = 2 * a^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l4001_400137


namespace NUMINAMATH_CALUDE_probability_of_black_ball_l4001_400117

theorem probability_of_black_ball (p_red p_white p_black : ℝ) : 
  p_red = 0.43 → p_white = 0.27 → p_red + p_white + p_black = 1 → p_black = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_black_ball_l4001_400117


namespace NUMINAMATH_CALUDE_gabrielle_robins_count_l4001_400190

/-- The number of birds Gabrielle saw -/
def gabrielle_total : ℕ := sorry

/-- The number of robins Gabrielle saw -/
def gabrielle_robins : ℕ := sorry

/-- The number of cardinals Gabrielle saw -/
def gabrielle_cardinals : ℕ := 4

/-- The number of blue jays Gabrielle saw -/
def gabrielle_blue_jays : ℕ := 3

/-- The number of birds Chase saw -/
def chase_total : ℕ := 10

/-- The number of robins Chase saw -/
def chase_robins : ℕ := 2

/-- The number of blue jays Chase saw -/
def chase_blue_jays : ℕ := 3

/-- The number of cardinals Chase saw -/
def chase_cardinals : ℕ := 5

theorem gabrielle_robins_count :
  gabrielle_total = chase_total + chase_total / 5 ∧
  gabrielle_total = gabrielle_robins + gabrielle_cardinals + gabrielle_blue_jays ∧
  gabrielle_robins = 5 := by sorry

end NUMINAMATH_CALUDE_gabrielle_robins_count_l4001_400190


namespace NUMINAMATH_CALUDE_complex_modulus_equality_l4001_400156

theorem complex_modulus_equality (x : ℝ) (h : x > 0) :
  Complex.abs (10 + Complex.I * x) = 15 ↔ x = 5 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_equality_l4001_400156


namespace NUMINAMATH_CALUDE_convention_handshakes_l4001_400106

/-- The number of companies at the convention -/
def num_companies : ℕ := 3

/-- The number of representatives from each company -/
def reps_per_company : ℕ := 5

/-- The total number of people at the convention -/
def total_people : ℕ := num_companies * reps_per_company

/-- The number of people each person shakes hands with -/
def handshakes_per_person : ℕ := total_people - 1 - (reps_per_company - 1)

/-- The total number of handshakes at the convention -/
def total_handshakes : ℕ := (total_people * handshakes_per_person) / 2

theorem convention_handshakes :
  total_handshakes = 75 :=
sorry

end NUMINAMATH_CALUDE_convention_handshakes_l4001_400106


namespace NUMINAMATH_CALUDE_sequence_relation_l4001_400139

def a : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 4 * a (n + 1) - a n

def b : ℕ → ℤ
  | 0 => 1
  | 1 => 2
  | (n + 2) => 4 * b (n + 1) - b n

theorem sequence_relation (n : ℕ) : (b n)^2 = 3 * (a n)^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_sequence_relation_l4001_400139


namespace NUMINAMATH_CALUDE_kittens_sold_l4001_400170

theorem kittens_sold (initial_puppies initial_kittens puppies_sold remaining_pets : ℕ) : 
  initial_puppies = 7 →
  initial_kittens = 6 →
  puppies_sold = 2 →
  remaining_pets = 8 →
  initial_puppies + initial_kittens - puppies_sold - remaining_pets = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_kittens_sold_l4001_400170


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l4001_400108

theorem polynomial_division_remainder :
  ∃ q : Polynomial ℝ, 3 * X^5 + X^4 + 3 = (X - 2)^2 * q + (13 * X - 9) :=
sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l4001_400108


namespace NUMINAMATH_CALUDE_arithmetic_sequence_100th_term_l4001_400163

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, a (n + 1) = a n + 4

theorem arithmetic_sequence_100th_term (a : ℕ → ℕ) 
  (h : arithmetic_sequence a) : a 100 = 397 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_100th_term_l4001_400163


namespace NUMINAMATH_CALUDE_tan_alpha_two_l4001_400121

theorem tan_alpha_two (α : Real) (h : Real.tan α = 2) : 
  Real.tan (2 * α + π / 4) = 9 ∧ (6 * Real.sin α + Real.cos α) / (3 * Real.sin α - 2 * Real.cos α) = 13/4 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_two_l4001_400121


namespace NUMINAMATH_CALUDE_f_odd_and_decreasing_l4001_400123

-- Define the function f(x) = -x|x|
def f (x : ℝ) : ℝ := -x * abs x

-- Theorem statement
theorem f_odd_and_decreasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, x < y → f x > f y) :=
by sorry

end NUMINAMATH_CALUDE_f_odd_and_decreasing_l4001_400123


namespace NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l4001_400130

theorem polar_to_rectangular_conversion :
  let r : ℝ := 4
  let θ : ℝ := π / 3
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  (x, y) = (2, 2 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l4001_400130


namespace NUMINAMATH_CALUDE_f_properties_l4001_400197

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x)

theorem f_properties :
  (∀ x, f (-x) = -f x) ∧
  (∀ x y, x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l4001_400197


namespace NUMINAMATH_CALUDE_brick_width_calculation_l4001_400185

/-- Proves that given a courtyard of 25 meters by 16 meters, to be paved with 20,000 bricks of length 20 cm, the width of each brick must be 10 cm. -/
theorem brick_width_calculation (courtyard_length : ℝ) (courtyard_width : ℝ) 
  (brick_length : ℝ) (total_bricks : ℕ) :
  courtyard_length = 25 →
  courtyard_width = 16 →
  brick_length = 0.2 →
  total_bricks = 20000 →
  ∃ (brick_width : ℝ), 
    brick_width = 0.1 ∧ 
    (courtyard_length * courtyard_width * 10000) = (brick_length * brick_width * total_bricks) :=
by sorry

end NUMINAMATH_CALUDE_brick_width_calculation_l4001_400185


namespace NUMINAMATH_CALUDE_line_m_equation_l4001_400144

/-- Two distinct lines in the xy-plane -/
structure TwoLines where
  ℓ : Set (ℝ × ℝ)
  m : Set (ℝ × ℝ)
  distinct : ℓ ≠ m
  intersect_origin : (0, 0) ∈ ℓ ∩ m

/-- Equation of a line in the form ax + by = 0 -/
def LineEquation (a b : ℝ) : Set (ℝ × ℝ) :=
  {(x, y) | a * x + b * y = 0}

/-- Reflection of a point about a line -/
def reflect (p : ℝ × ℝ) (line : Set (ℝ × ℝ)) : ℝ × ℝ := sorry

theorem line_m_equation (lines : TwoLines) 
  (h_ℓ : lines.ℓ = LineEquation 2 1)
  (h_Q : reflect (3, -2) lines.ℓ = reflect (reflect (3, -2) lines.ℓ) lines.m)
  (h_Q'' : reflect (reflect (3, -2) lines.ℓ) lines.m = (-1, 5)) :
  lines.m = LineEquation 3 1 := by sorry

end NUMINAMATH_CALUDE_line_m_equation_l4001_400144


namespace NUMINAMATH_CALUDE_binomial_30_3_l4001_400122

theorem binomial_30_3 : Nat.choose 30 3 = 4060 := by
  sorry

end NUMINAMATH_CALUDE_binomial_30_3_l4001_400122


namespace NUMINAMATH_CALUDE_quadratic_symmetry_l4001_400118

/-- A quadratic function -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c

theorem quadratic_symmetry 
  (f : ℝ → ℝ) 
  (hf : QuadraticFunction f) 
  (h0 : f 0 = 3)
  (h1 : f 1 = 2)
  (h2 : f 2 = 3)
  (h3 : f 3 = 6)
  (h4 : f 4 = 11)
  (hm2 : f (-2) = 11) :
  f (-1) = 6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_symmetry_l4001_400118


namespace NUMINAMATH_CALUDE_g_is_even_l4001_400160

noncomputable def g (x : ℝ) : ℝ := Real.log (x^2 + Real.sqrt (1 + x^4))

theorem g_is_even : ∀ x : ℝ, g (-x) = g x := by sorry

end NUMINAMATH_CALUDE_g_is_even_l4001_400160
