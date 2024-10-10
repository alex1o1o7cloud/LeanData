import Mathlib

namespace circle_center_l1512_151273

/-- The equation of a circle in the form (x + h)² + (y + k)² = r², where (h, k) is the center. -/
def CircleEquation (h k r : ℝ) : ℝ → ℝ → Prop :=
  fun x y ↦ (x + h)^2 + (y + k)^2 = r^2

/-- The center of the circle (x + 2)² + y² = 5 is (-2, 0). -/
theorem circle_center :
  ∃ (h k : ℝ), CircleEquation h k (Real.sqrt 5) = CircleEquation 2 0 (Real.sqrt 5) ∧ h = -2 ∧ k = 0 := by
  sorry

end circle_center_l1512_151273


namespace min_sum_of_product_2310_l1512_151234

theorem min_sum_of_product_2310 (a b c : ℕ+) (h : a * b * c = 2310) :
  (∀ x y z : ℕ+, x * y * z = 2310 → a + b + c ≤ x + y + z) ∧ a + b + c = 52 := by
  sorry

end min_sum_of_product_2310_l1512_151234


namespace square_product_exists_l1512_151276

theorem square_product_exists (A : Finset ℕ+) (h1 : A.card = 2016) 
  (h2 : ∀ x ∈ A, ∀ p : ℕ, Nat.Prime p → p ∣ x.val → p < 30) : 
  ∃ a b c d : ℕ+, a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ d ∈ A ∧ 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  ∃ m : ℕ, (a.val * b.val * c.val * d.val : ℕ) = m ^ 2 := by
  sorry


end square_product_exists_l1512_151276


namespace probability_two_diamonds_one_ace_l1512_151233

/-- Represents a standard deck of 52 cards -/
def StandardDeck : ℕ := 52

/-- Number of diamonds in a standard deck -/
def DiamondCount : ℕ := 13

/-- Number of aces in a standard deck -/
def AceCount : ℕ := 4

/-- Probability of drawing two diamonds followed by an ace from a standard deck -/
def probabilityTwoDiamondsOneAce : ℚ :=
  (DiamondCount : ℚ) / StandardDeck *
  (DiamondCount - 1) / (StandardDeck - 1) *
  ((DiamondCount : ℚ) / StandardDeck * (AceCount - 1) / (StandardDeck - 2) +
   (StandardDeck - DiamondCount : ℚ) / StandardDeck * AceCount / (StandardDeck - 2))

theorem probability_two_diamonds_one_ace :
  probabilityTwoDiamondsOneAce = 29 / 11050 := by
  sorry

end probability_two_diamonds_one_ace_l1512_151233


namespace exponential_function_sum_of_extrema_l1512_151255

/-- Given an exponential function y = a^x, if the sum of its maximum and minimum values 
    on the interval [0,1] is 3, then a = 2 -/
theorem exponential_function_sum_of_extrema (a : ℝ) : 
  (a > 0) → 
  (∀ x ∈ Set.Icc 0 1, ∃ y, y = a^x) →
  (Real.exp a + 1 = 3 ∨ a + Real.exp a = 3) →
  a = 2 :=
by sorry

end exponential_function_sum_of_extrema_l1512_151255


namespace g_expression_l1512_151258

theorem g_expression (x : ℝ) (g : ℝ → ℝ) :
  (2 * x^5 + 4 * x^3 - 3 * x + g x = 7 * x^3 + 5 * x - 2) →
  (g x = -2 * x^5 + 3 * x^3 + 8 * x - 2) := by
  sorry

end g_expression_l1512_151258


namespace nickels_count_l1512_151289

/-- Represents the number of coins of each type found by Harriett --/
structure CoinCounts where
  quarters : Nat
  dimes : Nat
  nickels : Nat
  pennies : Nat

/-- Calculates the total value in cents for a given set of coin counts --/
def totalValue (coins : CoinCounts) : Nat :=
  coins.quarters * 25 + coins.dimes * 10 + coins.nickels * 5 + coins.pennies

/-- Theorem stating that given the number of quarters, dimes, pennies, and the total value,
    the number of nickels must be 3 to make the total $3.00 --/
theorem nickels_count (coins : CoinCounts) :
  coins.quarters = 10 ∧ coins.dimes = 3 ∧ coins.pennies = 5 ∧ totalValue coins = 300 →
  coins.nickels = 3 := by
  sorry


end nickels_count_l1512_151289


namespace max_planes_is_six_l1512_151282

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A plane in 3D space -/
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- A configuration of 6 points in 3D space -/
def Configuration := Fin 6 → Point3D

/-- Check if a point lies on a plane -/
def pointOnPlane (p : Point3D) (plane : Plane3D) : Prop :=
  plane.a * p.x + plane.b * p.y + plane.c * p.z + plane.d = 0

/-- Check if a plane contains at least 4 points from the configuration -/
def planeContainsAtLeast4Points (plane : Plane3D) (config : Configuration) : Prop :=
  ∃ (p1 p2 p3 p4 : Fin 6), p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 ∧
    pointOnPlane (config p1) plane ∧ pointOnPlane (config p2) plane ∧
    pointOnPlane (config p3) plane ∧ pointOnPlane (config p4) plane

/-- Check if no line passes through 4 points in the configuration -/
def noLinePasses4Points (config : Configuration) : Prop :=
  ∀ (p1 p2 p3 p4 : Fin 6), p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 →
    ¬∃ (a b c : ℝ), ∀ (p : Fin 6), p = p1 ∨ p = p2 ∨ p = p3 ∨ p = p4 →
      a * (config p).x + b * (config p).y + c = (config p).z

/-- The main theorem: The maximum number of planes satisfying the conditions is 6 -/
theorem max_planes_is_six (config : Configuration) 
    (h_no_line : noLinePasses4Points config) : 
    (∃ (planes : Fin 6 → Plane3D), ∀ (i : Fin 6), planeContainsAtLeast4Points (planes i) config) ∧
    (∀ (n : ℕ) (planes : Fin (n + 1) → Plane3D), 
      (∀ (i : Fin (n + 1)), planeContainsAtLeast4Points (planes i) config) → n ≤ 5) :=
  sorry


end max_planes_is_six_l1512_151282


namespace distribute_4_3_l1512_151295

/-- The number of ways to distribute n distinct objects into k distinct containers,
    with each container having at least one object. -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The theorem stating that there are 36 ways to distribute 4 distinct objects
    into 3 distinct containers, with each container having at least one object. -/
theorem distribute_4_3 : distribute 4 3 = 36 := by sorry

end distribute_4_3_l1512_151295


namespace completing_square_solution_l1512_151204

theorem completing_square_solution (x : ℝ) : 
  (x^2 + 4*x + 1 = 0) ↔ ((x + 2)^2 = 3) :=
by sorry

end completing_square_solution_l1512_151204


namespace no_solution_for_lcm_gcd_equation_l1512_151219

theorem no_solution_for_lcm_gcd_equation : 
  ¬ ∃ (n : ℕ), 
    (n > 0) ∧ 
    (Nat.lcm n 60 = Nat.gcd n 60 + 200) ∧ 
    (Nat.Prime n) ∧ 
    (60 % n = 0) := by
  sorry

end no_solution_for_lcm_gcd_equation_l1512_151219


namespace y_equals_five_l1512_151220

/-- Configuration of numbers in a triangular arrangement -/
structure NumberTriangle where
  y : ℝ
  z : ℝ
  second_row : ℝ
  third_row : ℝ
  h1 : second_row = y * 10
  h2 : third_row = second_row * z

/-- The value of y in the given configuration is 5 -/
theorem y_equals_five (t : NumberTriangle) (h3 : t.second_row = 50) (h4 : t.third_row = 300) : t.y = 5 := by
  sorry


end y_equals_five_l1512_151220


namespace second_calculator_price_l1512_151238

def total_calculators : ℕ := 85
def total_sales : ℚ := 3875
def first_calculator_count : ℕ := 35
def first_calculator_price : ℚ := 67

theorem second_calculator_price :
  let second_calculator_count := total_calculators - first_calculator_count
  let first_calculator_total := first_calculator_count * first_calculator_price
  let second_calculator_total := total_sales - first_calculator_total
  second_calculator_total / second_calculator_count = 30.6 := by
sorry

end second_calculator_price_l1512_151238


namespace james_seed_planting_l1512_151265

/-- Calculates the percentage of seeds planted -/
def percentage_planted (original_trees : ℕ) (plants_per_tree : ℕ) (seeds_per_plant : ℕ) (new_trees : ℕ) : ℚ :=
  (new_trees : ℚ) / ((original_trees * plants_per_tree * seeds_per_plant) : ℚ) * 100

/-- Proves that the percentage of seeds planted is 60% given the problem conditions -/
theorem james_seed_planting :
  let original_trees : ℕ := 2
  let plants_per_tree : ℕ := 20
  let seeds_per_plant : ℕ := 1
  let new_trees : ℕ := 24
  percentage_planted original_trees plants_per_tree seeds_per_plant new_trees = 60 := by
  sorry

end james_seed_planting_l1512_151265


namespace vector_magnitude_minimization_l1512_151247

/-- Given two unit vectors e₁ and e₂ with an angle of 60° between them,
    prove that |2e₁ + te₂| is minimized when t = -1 -/
theorem vector_magnitude_minimization (e₁ e₂ : ℝ × ℝ) :
  ‖e₁‖ = 1 →
  ‖e₂‖ = 1 →
  e₁ • e₂ = 1/2 →
  ∃ (t : ℝ), ∀ (s : ℝ), ‖2 • e₁ + t • e₂‖ ≤ ‖2 • e₁ + s • e₂‖ ∧ t = -1 := by
  sorry

end vector_magnitude_minimization_l1512_151247


namespace square_perimeter_l1512_151253

theorem square_perimeter (area : ℝ) (side : ℝ) (h1 : area = 625) (h2 : side * side = area) : 
  4 * side = 100 := by
  sorry

end square_perimeter_l1512_151253


namespace age_difference_theorem_l1512_151264

/-- Represents a two-digit age --/
structure TwoDigitAge where
  tens : Nat
  ones : Nat
  h1 : tens ≤ 9
  h2 : ones ≤ 9

def TwoDigitAge.toNat (age : TwoDigitAge) : Nat :=
  10 * age.tens + age.ones

theorem age_difference_theorem (anna ella : TwoDigitAge) 
  (h : anna.tens = ella.ones ∧ anna.ones = ella.tens) 
  (future_relation : (anna.toNat + 10) = 3 * (ella.toNat + 10)) :
  anna.toNat - ella.toNat = 54 := by
  sorry


end age_difference_theorem_l1512_151264


namespace linear_functions_inequality_l1512_151201

theorem linear_functions_inequality (k : ℝ) :
  (∀ x > -1, k * x - 2 < 2 * x + 3) →
  -3 ≤ k ∧ k ≤ 2 ∧ k ≠ 0 :=
by sorry

end linear_functions_inequality_l1512_151201


namespace turnip_potato_ratio_l1512_151235

/-- Given a ratio of potatoes to turnips and a new amount of potatoes, 
    calculate the amount of turnips that maintains the same ratio -/
def calculate_turnips (potato_ratio : ℚ) (turnip_ratio : ℚ) (new_potato : ℚ) : ℚ :=
  (new_potato * turnip_ratio) / potato_ratio

/-- Prove that given the initial ratio of 5 cups of potatoes to 2 cups of turnips,
    the amount of turnips that can be mixed with 20 cups of potatoes while 
    maintaining the same ratio is 8 cups -/
theorem turnip_potato_ratio : 
  let initial_potato : ℚ := 5
  let initial_turnip : ℚ := 2
  let new_potato : ℚ := 20
  calculate_turnips initial_potato initial_turnip new_potato = 8 := by
  sorry

end turnip_potato_ratio_l1512_151235


namespace championship_outcomes_8_3_l1512_151229

/-- The number of possible outcomes for championships -/
def championship_outcomes (num_students : ℕ) (num_championships : ℕ) : ℕ :=
  num_students ^ num_championships

/-- Theorem: The number of possible outcomes for 3 championships among 8 students is 512 -/
theorem championship_outcomes_8_3 :
  championship_outcomes 8 3 = 512 := by
  sorry

end championship_outcomes_8_3_l1512_151229


namespace upstream_distance_calculation_l1512_151246

/-- Represents the problem of calculating the upstream distance rowed by a man --/
theorem upstream_distance_calculation
  (downstream_distance : ℝ)
  (downstream_time : ℝ)
  (upstream_time : ℝ)
  (current_velocity : ℝ)
  (h1 : downstream_distance = 32)
  (h2 : downstream_time = 6)
  (h3 : upstream_time = 6)
  (h4 : current_velocity = 1.5)
  (h5 : downstream_time > 0)
  (h6 : upstream_time > 0)
  (h7 : current_velocity ≥ 0) :
  let still_water_speed := downstream_distance / downstream_time - current_velocity
  let upstream_distance := (still_water_speed - current_velocity) * upstream_time
  upstream_distance = 14 := by sorry


end upstream_distance_calculation_l1512_151246


namespace simplify_trig_expression_l1512_151286

theorem simplify_trig_expression (α : ℝ) :
  (2 * Real.sin (2 * α)) / (1 + Real.cos (2 * α)) = 2 * Real.tan α :=
by sorry

end simplify_trig_expression_l1512_151286


namespace ten_strikes_l1512_151268

/-- Represents the time it takes for a clock to strike a given number of times -/
def strike_time (strikes : ℕ) : ℝ :=
  sorry

/-- The clock takes 42 seconds to strike 7 times -/
axiom seven_strikes : strike_time 7 = 42

/-- Theorem: It takes 60 seconds for the clock to strike 10 times -/
theorem ten_strikes : strike_time 10 = 60 := by
  sorry

end ten_strikes_l1512_151268


namespace repeating_decimal_difference_l1512_151209

theorem repeating_decimal_difference : 
  (8 : ℚ) / 11 - 72 / 100 = 2 / 275 := by sorry

end repeating_decimal_difference_l1512_151209


namespace randy_lunch_cost_l1512_151218

theorem randy_lunch_cost (initial_amount : ℝ) (remaining_amount : ℝ) (lunch_cost : ℝ) :
  initial_amount = 30 →
  remaining_amount = 15 →
  remaining_amount = initial_amount - lunch_cost - (1/4) * (initial_amount - lunch_cost) →
  lunch_cost = 10 := by
  sorry

end randy_lunch_cost_l1512_151218


namespace selling_price_achieves_target_profit_selling_price_minimizes_inventory_l1512_151202

/-- Represents the selling price of a helmet -/
def selling_price : ℝ := 50

/-- Represents the cost price of a helmet -/
def cost_price : ℝ := 30

/-- Represents the initial selling price -/
def initial_price : ℝ := 40

/-- Represents the initial monthly sales volume -/
def initial_sales : ℝ := 600

/-- Represents the rate of decrease in sales volume per dollar increase in price -/
def sales_decrease_rate : ℝ := 10

/-- Represents the target monthly profit -/
def target_profit : ℝ := 10000

/-- Calculates the monthly sales volume based on the selling price -/
def monthly_sales (price : ℝ) : ℝ := initial_sales - sales_decrease_rate * (price - initial_price)

/-- Calculates the monthly profit based on the selling price -/
def monthly_profit (price : ℝ) : ℝ := (price - cost_price) * monthly_sales price

/-- Theorem stating that the selling price achieves the target monthly profit -/
theorem selling_price_achieves_target_profit : 
  monthly_profit selling_price = target_profit :=
sorry

/-- Theorem stating that the selling price minimizes inventory -/
theorem selling_price_minimizes_inventory :
  ∀ (price : ℝ), monthly_profit price = target_profit → price ≥ selling_price :=
sorry

end selling_price_achieves_target_profit_selling_price_minimizes_inventory_l1512_151202


namespace expression_simplification_l1512_151259

theorem expression_simplification (x y : ℤ) : 
  (x = 1) → (y = -2) → 
  2 * x^2 - (3 * (-5/3 * x^2 + 2/3 * x * y) - (x * y - 3 * x^2)) + 2 * x * y = 2 := by
  sorry

end expression_simplification_l1512_151259


namespace response_change_difference_l1512_151243

theorem response_change_difference (initial_yes initial_no final_yes final_no : ℚ) :
  initial_yes = 40/100 →
  initial_no = 60/100 →
  final_yes = 80/100 →
  final_no = 20/100 →
  initial_yes + initial_no = 1 →
  final_yes + final_no = 1 →
  ∃ (min_change max_change : ℚ),
    (∀ (change : ℚ), change ≥ min_change ∧ change ≤ max_change) ∧
    max_change - min_change = 20/100 :=
by sorry

end response_change_difference_l1512_151243


namespace polynomial_product_expansion_l1512_151239

theorem polynomial_product_expansion (x : ℝ) :
  (1 + x^2 + 2*x - x^4) * (3 - x^3 + 2*x^2 - 5*x) =
  x^7 - 2*x^6 + 4*x^5 - 3*x^4 - 2*x^3 - 4*x^2 + x + 3 := by
  sorry

end polynomial_product_expansion_l1512_151239


namespace cuboid_area_volume_l1512_151236

/-- Cuboid properties -/
def Cuboid (a b c : ℝ) : Prop :=
  c * Real.sqrt (a^2 + b^2) = 60 ∧
  a * Real.sqrt (b^2 + c^2) = 4 * Real.sqrt 153 ∧
  b * Real.sqrt (a^2 + c^2) = 12 * Real.sqrt 10

/-- Theorem: Surface area and volume of the cuboid -/
theorem cuboid_area_volume (a b c : ℝ) (h : Cuboid a b c) :
  2 * (a * b + b * c + a * c) = 192 ∧ a * b * c = 144 := by
  sorry

end cuboid_area_volume_l1512_151236


namespace cone_volume_l1512_151287

/-- Given a cone with slant height 15 cm and height 9 cm, its volume is 432π cubic centimeters. -/
theorem cone_volume (s h r : ℝ) (hs : s = 15) (hh : h = 9) 
  (hr : r^2 = s^2 - h^2) : (1/3 : ℝ) * π * r^2 * h = 432 * π := by
  sorry

end cone_volume_l1512_151287


namespace zongzi_production_theorem_l1512_151200

/-- The average daily production of zongzi for Team A -/
def team_a_production : ℝ := 200

/-- The average daily production of zongzi for Team B -/
def team_b_production : ℝ := 150

/-- Theorem stating that given the conditions, the average daily production
    of zongzi for Team A is 200 bags and for Team B is 150 bags -/
theorem zongzi_production_theorem :
  (team_a_production + team_b_production = 350) ∧
  (2 * team_a_production - team_b_production = 250) →
  team_a_production = 200 ∧ team_b_production = 150 := by
  sorry

end zongzi_production_theorem_l1512_151200


namespace complement_union_equals_set_l1512_151226

def U : Set Int := {-2, -1, 0, 1, 2, 3}
def A : Set Int := {-1, 0, 1}
def B : Set Int := {1, 2}

theorem complement_union_equals_set : 
  (U \ (A ∪ B)) = {-2, 3} := by sorry

end complement_union_equals_set_l1512_151226


namespace area_between_circles_l1512_151244

/-- The area between two externally tangent circles and their circumscribing circle -/
theorem area_between_circles (r₁ r₂ : ℝ) (h₁ : r₁ = 4) (h₂ : r₂ = 5) : 
  let R := (r₁ + r₂) / 2
  π * R^2 - π * r₁^2 - π * r₂^2 = 40 * π := by sorry

end area_between_circles_l1512_151244


namespace everton_college_calculator_cost_l1512_151294

/-- The total cost of calculators purchased by Everton college -/
def total_cost (scientific_count : ℕ) (graphing_count : ℕ) (scientific_price : ℕ) (graphing_price : ℕ) : ℕ :=
  scientific_count * scientific_price + graphing_count * graphing_price

/-- Theorem stating the total cost of calculators for Everton college -/
theorem everton_college_calculator_cost :
  total_cost 20 25 10 57 = 1625 := by
  sorry

end everton_college_calculator_cost_l1512_151294


namespace acute_angles_inequality_l1512_151217

theorem acute_angles_inequality (α β : Real) (h_α : 0 < α ∧ α < π / 2) (h_β : 0 < β ∧ β < π / 2) :
  Real.cos α * Real.sin (2 * α) * Real.sin (2 * β) ≤ 4 * Real.sqrt 3 / 9 := by
  sorry

end acute_angles_inequality_l1512_151217


namespace carols_allowance_l1512_151260

/-- Carol's allowance problem -/
theorem carols_allowance
  (fixed_allowance : ℚ)
  (extra_chore_pay : ℚ)
  (weeks : ℕ)
  (total_amount : ℚ)
  (avg_extra_chores : ℚ)
  (h1 : extra_chore_pay = 1.5)
  (h2 : weeks = 10)
  (h3 : total_amount = 425)
  (h4 : avg_extra_chores = 15) :
  fixed_allowance = 20 := by
  sorry

end carols_allowance_l1512_151260


namespace smallest_ending_nine_div_thirteen_l1512_151269

/-- A function that checks if a number ends with 9 -/
def endsWithNine (n : ℕ) : Prop := n % 10 = 9

/-- The theorem stating that 169 is the smallest positive integer ending in 9 and divisible by 13 -/
theorem smallest_ending_nine_div_thirteen :
  ∀ n : ℕ, n > 0 → endsWithNine n → n % 13 = 0 → n ≥ 169 :=
by sorry

end smallest_ending_nine_div_thirteen_l1512_151269


namespace book_gain_percent_l1512_151208

theorem book_gain_percent (marked_price : ℝ) (marked_price_pos : marked_price > 0) : 
  let cost_price := 0.64 * marked_price
  let selling_price := 0.88 * marked_price
  let profit := selling_price - cost_price
  let gain_percent := (profit / cost_price) * 100
  gain_percent = 37.5 := by sorry

end book_gain_percent_l1512_151208


namespace clubsuit_equation_solution_l1512_151254

/-- Definition of the clubsuit operation -/
def clubsuit (A B : ℝ) : ℝ := 4 * A + 3 * B + 7

/-- Theorem stating that A clubsuit 6 = 85 when A = 15 -/
theorem clubsuit_equation_solution :
  clubsuit 15 6 = 85 := by sorry

end clubsuit_equation_solution_l1512_151254


namespace triangle_inequality_theorem_set_A_forms_triangle_l1512_151222

def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_inequality_theorem (a b c : ℝ) :
  can_form_triangle a b c ↔ (a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) :=
sorry

theorem set_A_forms_triangle :
  can_form_triangle 8 6 5 :=
sorry

end triangle_inequality_theorem_set_A_forms_triangle_l1512_151222


namespace three_non_congruent_triangles_l1512_151290

/-- A triangle with integer side lengths -/
structure IntTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- Checks if two triangles are congruent -/
def are_congruent (t1 t2 : IntTriangle) : Prop :=
  (t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c) ∨
  (t1.a = t2.b ∧ t1.b = t2.c ∧ t1.c = t2.a) ∨
  (t1.a = t2.c ∧ t1.b = t2.a ∧ t1.c = t2.b)

/-- The set of all triangles with perimeter 11 -/
def triangles_with_perimeter_11 : Set IntTriangle :=
  {t : IntTriangle | t.a + t.b + t.c = 11}

/-- The theorem to be proved -/
theorem three_non_congruent_triangles : 
  ∃ (t1 t2 t3 : IntTriangle), 
    t1 ∈ triangles_with_perimeter_11 ∧
    t2 ∈ triangles_with_perimeter_11 ∧
    t3 ∈ triangles_with_perimeter_11 ∧
    ¬(are_congruent t1 t2) ∧
    ¬(are_congruent t1 t3) ∧
    ¬(are_congruent t2 t3) ∧
    ∀ (t : IntTriangle), t ∈ triangles_with_perimeter_11 → 
      (are_congruent t t1 ∨ are_congruent t t2 ∨ are_congruent t t3) :=
by
  sorry

end three_non_congruent_triangles_l1512_151290


namespace shaded_area_of_folded_rectangle_l1512_151275

/-- The area of the shaded region formed by folding a rectangular sheet along its diagonal -/
theorem shaded_area_of_folded_rectangle (length width : ℝ) (h_length : length = 12) (h_width : width = 18) :
  let rectangle_area := length * width
  let diagonal := Real.sqrt (length^2 + width^2)
  let triangle_area := (1 / 2) * diagonal * diagonal * (2 / 3)
  rectangle_area - triangle_area = 138 := by sorry

end shaded_area_of_folded_rectangle_l1512_151275


namespace sandy_carrots_l1512_151211

/-- Given that Sandy and Sam grew carrots together, with Sam growing 3 carrots
and a total of 9 carrots grown, prove that Sandy grew 6 carrots. -/
theorem sandy_carrots (total : ℕ) (sam : ℕ) (sandy : ℕ) 
  (h1 : total = 9)
  (h2 : sam = 3)
  (h3 : total = sam + sandy) :
  sandy = 6 := by
  sorry

end sandy_carrots_l1512_151211


namespace hyperbola_asymptotes_l1512_151279

/-- The hyperbola equation -/
def hyperbola_equation (x y a : ℝ) : Prop :=
  y^2 / (2 * a^2) - x^2 / a^2 = 1

/-- The asymptote equation -/
def asymptote_equation (x y : ℝ) : Prop :=
  y = Real.sqrt 2 * x ∨ y = -Real.sqrt 2 * x

/-- Theorem: The asymptotes of the given hyperbola are y = ±√2x -/
theorem hyperbola_asymptotes (a : ℝ) (h : a ≠ 0) :
  ∀ x y : ℝ, hyperbola_equation x y a → asymptote_equation x y :=
sorry

end hyperbola_asymptotes_l1512_151279


namespace imaginary_sum_l1512_151248

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_sum : i^55 + i^555 + i^5 = -i := by
  sorry

end imaginary_sum_l1512_151248


namespace max_value_of_s_l1512_151267

theorem max_value_of_s (p q r s : ℝ) 
  (sum_eq : p + q + r + s = 10) 
  (sum_products_eq : p*q + p*r + p*s + q*r + q*s + r*s = 20) : 
  s ≤ (5 + Real.sqrt 105) / 2 := by
  sorry

end max_value_of_s_l1512_151267


namespace evaluate_expression_l1512_151278

theorem evaluate_expression : 4 * (8 - 3) - 6 = 14 := by
  sorry

end evaluate_expression_l1512_151278


namespace inequality_proof_l1512_151293

theorem inequality_proof (a b c d x y : ℝ) 
  (ha : a > 1) (hb : b > 1) (hc : c > 1) (hd : d > 1)
  (h1 : a^x + b^y = (a^2 + b^2)^x)
  (h2 : c^x + d^y = 2^y * (c*d)^(y/2)) :
  x < y := by sorry

end inequality_proof_l1512_151293


namespace smallest_parabola_coefficient_l1512_151292

theorem smallest_parabola_coefficient (a b c : ℝ) : 
  (∃ (x y : ℝ), y = a * (x - 3/4)^2 - 25/16) →  -- vertex condition
  (∀ (x y : ℝ), y = a * x^2 + b * x + c) →      -- parabola equation
  a > 0 →                                       -- a is positive
  ∃ (n : ℚ), a + b + c = n →                    -- sum is rational
  ∀ (a' : ℝ), (∃ (b' c' : ℝ) (n' : ℚ), 
    (∀ (x y : ℝ), y = a' * x^2 + b' * x + c') ∧ 
    a' > 0 ∧ 
    a' + b' + c' = n') → 
  a ≤ a' →
  a = 41 := by
sorry

end smallest_parabola_coefficient_l1512_151292


namespace bridge_length_specific_bridge_length_l1512_151270

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time_s : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let total_distance := train_speed_ms * crossing_time_s
  total_distance - train_length

/-- Proof of the specific bridge length problem -/
theorem specific_bridge_length :
  bridge_length 150 45 30 = 225 := by
  sorry

end bridge_length_specific_bridge_length_l1512_151270


namespace sqrt_8_is_quadratic_radical_l1512_151242

-- Define what a quadratic radical is
def is_quadratic_radical (x : ℝ) : Prop :=
  ∃ (y : ℝ), y ≥ 0 ∧ x = Real.sqrt y

-- Theorem statement
theorem sqrt_8_is_quadratic_radical :
  is_quadratic_radical (Real.sqrt 8) ∧
  ¬(∀ x : ℝ, is_quadratic_radical (Real.sqrt x)) ∧
  ¬(∀ m n : ℝ, is_quadratic_radical (Real.sqrt (m + n))) :=
sorry

end sqrt_8_is_quadratic_radical_l1512_151242


namespace marys_thursday_payment_l1512_151249

theorem marys_thursday_payment 
  (credit_limit : ℕ) 
  (tuesday_payment : ℕ) 
  (remaining_balance : ℕ) 
  (h1 : credit_limit = 100)
  (h2 : tuesday_payment = 15)
  (h3 : remaining_balance = 62) :
  credit_limit - tuesday_payment - remaining_balance = 23 := by
sorry

end marys_thursday_payment_l1512_151249


namespace monotonic_increasing_range_l1512_151271

def f (a : ℝ) (x : ℝ) : ℝ := (a - 1) * x

theorem monotonic_increasing_range (a : ℝ) :
  Monotone (f a) → a > 2 := by
  sorry

end monotonic_increasing_range_l1512_151271


namespace probability_three_different_suits_l1512_151284

/-- Represents a standard deck of 52 cards -/
def StandardDeck : ℕ := 52

/-- Represents the number of suits in a standard deck -/
def NumberOfSuits : ℕ := 4

/-- Represents the number of cards in each suit -/
def CardsPerSuit : ℕ := 13

/-- The probability of picking three cards of different suits from a standard deck without replacement -/
def probabilityDifferentSuits : ℚ :=
  (CardsPerSuit * (StandardDeck - NumberOfSuits)) / 
  (StandardDeck * (StandardDeck - 1))

theorem probability_three_different_suits :
  probabilityDifferentSuits = 169 / 425 := by
  sorry

end probability_three_different_suits_l1512_151284


namespace division_equals_fraction_l1512_151299

theorem division_equals_fraction : 200 / (12 + 15 * 2 - 4)^2 = 50 / 361 := by
  sorry

end division_equals_fraction_l1512_151299


namespace roots_of_polynomial_l1512_151210

/-- The polynomial f(x) = x^3 - 3x^2 - x + 3 -/
def f (x : ℝ) : ℝ := x^3 - 3*x^2 - x + 3

theorem roots_of_polynomial :
  (∀ x : ℝ, f x = 0 ↔ x = 1 ∨ x = -1 ∨ x = 3) := by
  sorry

end roots_of_polynomial_l1512_151210


namespace percentage_relationship_z_less_than_y_l1512_151212

theorem percentage_relationship (w e y z : ℝ) 
  (hw : w = 0.6 * e)
  (he : e = 0.6 * y)
  (hz : z = w * 1.5000000000000002) :
  z = 0.54 * y :=
by sorry

-- The final result can be derived from this theorem
theorem z_less_than_y (w e y z : ℝ) 
  (hw : w = 0.6 * e)
  (he : e = 0.6 * y)
  (hz : z = w * 1.5000000000000002) :
  (y - z) / y = 0.46 :=
by sorry

end percentage_relationship_z_less_than_y_l1512_151212


namespace larger_integer_value_l1512_151288

theorem larger_integer_value (a b : ℕ+) 
  (h_quotient : (a : ℚ) / (b : ℚ) = 5 / 2)
  (h_product : (a : ℕ) * b = 160) : 
  max a b = 20 := by
sorry

end larger_integer_value_l1512_151288


namespace seventh_day_cans_l1512_151283

/-- A sequence where the first term is 4 and each subsequent term increases by 5 -/
def canSequence : ℕ → ℕ
  | 0 => 4
  | n + 1 => canSequence n + 5

/-- The 7th term of the sequence is 34 -/
theorem seventh_day_cans : canSequence 6 = 34 := by
  sorry

end seventh_day_cans_l1512_151283


namespace complex_exponential_sum_l1512_151206

theorem complex_exponential_sum (α β θ : ℝ) :
  Complex.exp (Complex.I * (α + θ)) + Complex.exp (Complex.I * (β + θ)) = (1/3 : ℂ) + (4/9 : ℂ) * Complex.I →
  Complex.exp (-Complex.I * (α + θ)) + Complex.exp (-Complex.I * (β + θ)) = (1/3 : ℂ) - (4/9 : ℂ) * Complex.I :=
by sorry

end complex_exponential_sum_l1512_151206


namespace round_balloons_count_l1512_151261

/-- The number of balloons in each bag of round balloons -/
def round_balloons_per_bag : ℕ := sorry

/-- The number of bags of round balloons -/
def round_balloon_bags : ℕ := 5

/-- The number of bags of long balloons -/
def long_balloon_bags : ℕ := 4

/-- The number of long balloons in each bag -/
def long_balloons_per_bag : ℕ := 30

/-- The number of round balloons that burst -/
def burst_balloons : ℕ := 5

/-- The total number of balloons left -/
def total_balloons_left : ℕ := 215

theorem round_balloons_count : round_balloons_per_bag = 20 := by
  sorry

end round_balloons_count_l1512_151261


namespace profit_growth_rate_l1512_151281

/-- The average monthly growth rate that achieves the target profit -/
def average_growth_rate : ℝ := 0.2

/-- The initial profit in June -/
def initial_profit : ℝ := 2500

/-- The target profit in August -/
def target_profit : ℝ := 3600

/-- The number of months between June and August -/
def months : ℕ := 2

theorem profit_growth_rate :
  initial_profit * (1 + average_growth_rate) ^ months = target_profit :=
sorry

end profit_growth_rate_l1512_151281


namespace fractions_product_one_l1512_151215

theorem fractions_product_one :
  ∃ (a b c : ℕ), 
    2 ≤ a ∧ a ≤ 2016 ∧
    2 ≤ b ∧ b ≤ 2016 ∧
    2 ≤ c ∧ c ≤ 2016 ∧
    (a : ℚ) / (2018 - a) * (b : ℚ) / (2018 - b) * (c : ℚ) / (2018 - c) = 1 :=
by sorry

end fractions_product_one_l1512_151215


namespace angle_sum_theorem_l1512_151225

theorem angle_sum_theorem (α β : Real) : 
  0 < α ∧ α < π/2 → -- α is acute
  0 < β ∧ β < π/2 → -- β is acute
  |Real.sin α - 1/2| + Real.sqrt ((Real.tan β - 1)^2) = 0 →
  α + β = π/2.4 := by
sorry

end angle_sum_theorem_l1512_151225


namespace negation_p_false_necessary_not_sufficient_l1512_151228

theorem negation_p_false_necessary_not_sufficient (p q : Prop) :
  (∃ (h : p ∧ q), ¬¬p) ∧ 
  (∃ (h : ¬¬p), ¬(p ∧ q)) := by
sorry

end negation_p_false_necessary_not_sufficient_l1512_151228


namespace cheryl_material_usage_l1512_151252

theorem cheryl_material_usage 
  (material1 : ℚ) 
  (material2 : ℚ) 
  (leftover : ℚ) 
  (h1 : material1 = 5 / 9) 
  (h2 : material2 = 1 / 3) 
  (h3 : leftover = 8 / 24) : 
  material1 + material2 - leftover = 5 / 9 := by
sorry

end cheryl_material_usage_l1512_151252


namespace maggies_earnings_proof_l1512_151296

/-- Calculates Maggie's earnings from selling magazine subscriptions -/
def maggies_earnings (price_per_subscription : ℕ) 
                     (parents_subscriptions : ℕ)
                     (grandfather_subscriptions : ℕ)
                     (neighbor1_subscriptions : ℕ) : ℕ :=
  let total_subscriptions := parents_subscriptions + 
                             grandfather_subscriptions + 
                             neighbor1_subscriptions + 
                             (2 * neighbor1_subscriptions)
  price_per_subscription * total_subscriptions

theorem maggies_earnings_proof : 
  maggies_earnings 5 4 1 2 = 55 := by
  sorry

#eval maggies_earnings 5 4 1 2

end maggies_earnings_proof_l1512_151296


namespace day_299_is_tuesday_l1512_151224

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Calculates the day of the week given a day number -/
def dayOfWeek (dayNumber : Nat) : DayOfWeek :=
  match dayNumber % 7 with
  | 0 => DayOfWeek.Sunday
  | 1 => DayOfWeek.Monday
  | 2 => DayOfWeek.Tuesday
  | 3 => DayOfWeek.Wednesday
  | 4 => DayOfWeek.Thursday
  | 5 => DayOfWeek.Friday
  | _ => DayOfWeek.Saturday

theorem day_299_is_tuesday (isLeapYear : Bool) :
  isLeapYear ∧ dayOfWeek 45 = DayOfWeek.Sunday →
  dayOfWeek 299 = DayOfWeek.Tuesday :=
by
  sorry

end day_299_is_tuesday_l1512_151224


namespace quality_difference_proof_l1512_151266

-- Define the data from the problem
def total_products : ℕ := 400
def machine_a_first_class : ℕ := 150
def machine_a_second_class : ℕ := 50
def machine_b_first_class : ℕ := 120
def machine_b_second_class : ℕ := 80

-- Define the K² formula
def k_squared (n a b c d : ℕ) : ℚ :=
  (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Define the theorem
theorem quality_difference_proof :
  (machine_a_first_class : ℚ) / (machine_a_first_class + machine_a_second_class) = 3/4 ∧
  (machine_b_first_class : ℚ) / (machine_b_first_class + machine_b_second_class) = 3/5 ∧
  k_squared total_products machine_a_first_class machine_a_second_class machine_b_first_class machine_b_second_class > 6635/1000 :=
by sorry

end quality_difference_proof_l1512_151266


namespace josh_lost_marbles_l1512_151285

/-- Represents the number of marbles Josh lost -/
def marbles_lost (initial current : ℕ) : ℕ := initial - current

/-- Theorem stating that Josh lost 5 marbles -/
theorem josh_lost_marbles : marbles_lost 9 4 = 5 := by sorry

end josh_lost_marbles_l1512_151285


namespace min_value_theorem_l1512_151203

theorem min_value_theorem (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  ∃ (m : ℝ), m = 4 ∧ (∀ x y, x > y ∧ y > 0 → x^2 + 1 / (y * (x - y)) ≥ m) ∧
  (∃ x y, x > y ∧ y > 0 ∧ x^2 + 1 / (y * (x - y)) = m) := by
  sorry

end min_value_theorem_l1512_151203


namespace least_addend_for_divisibility_problem_solution_l1512_151277

theorem least_addend_for_divisibility (n : Nat) (d : Nat) (h : d > 0) :
  ∃ (x : Nat), x < d ∧ (n + x) % d = 0 ∧ ∀ (y : Nat), y < x → (n + y) % d ≠ 0 :=
by sorry

theorem problem_solution :
  ∃ (x : Nat), x = 19 ∧ (1156 + x) % 25 = 0 ∧ ∀ (y : Nat), y < x → (1156 + y) % 25 ≠ 0 :=
by sorry

end least_addend_for_divisibility_problem_solution_l1512_151277


namespace line_point_k_value_l1512_151214

/-- A line contains the points (5,10), (-3,k), and (-11,5). This theorem proves that k = 7.5. -/
theorem line_point_k_value (k : ℝ) : 
  (∃ (m b : ℝ), 
    (10 = m * 5 + b) ∧ 
    (k = m * (-3) + b) ∧ 
    (5 = m * (-11) + b)) → 
  k = 7.5 := by
sorry

end line_point_k_value_l1512_151214


namespace sum_first_49_primes_l1512_151280

def first_n_primes (n : ℕ) : List ℕ := sorry

theorem sum_first_49_primes :
  (first_n_primes 49).sum = 10787 := by sorry

end sum_first_49_primes_l1512_151280


namespace triangular_seating_theorem_l1512_151216

/-- Represents a triangular seating arrangement in a cinema -/
structure TriangularSeating where
  /-- The number of the best seat (at the center of the height from the top vertex) -/
  best_seat : ℕ
  /-- The total number of seats in the arrangement -/
  total_seats : ℕ

/-- 
Theorem: In a triangular seating arrangement where the best seat 
(at the center of the height from the top vertex) is numbered 265, 
the total number of seats is 1035.
-/
theorem triangular_seating_theorem (ts : TriangularSeating) 
  (h : ts.best_seat = 265) : ts.total_seats = 1035 := by
  sorry

#check triangular_seating_theorem

end triangular_seating_theorem_l1512_151216


namespace sarah_and_bob_walking_l1512_151251

/-- Sarah's walking rate in miles per minute -/
def sarah_rate : ℚ := 1 / 18

/-- Time Sarah walks in minutes -/
def sarah_time : ℚ := 15

/-- Distance Sarah walks in miles -/
def sarah_distance : ℚ := sarah_rate * sarah_time

/-- Bob's walking rate in miles per minute -/
def bob_rate : ℚ := 2 * sarah_rate

/-- Time Bob takes to walk Sarah's distance in minutes -/
def bob_time : ℚ := sarah_distance / bob_rate

theorem sarah_and_bob_walking :
  sarah_distance = 5 / 6 ∧ bob_time = 15 / 2 := by
  sorry

end sarah_and_bob_walking_l1512_151251


namespace other_endpoint_coordinates_l1512_151272

/-- Given a line segment with midpoint (3, 7) and one endpoint at (0, 11),
    prove that the other endpoint is at (6, 3). -/
theorem other_endpoint_coordinates :
  ∀ (x y : ℝ),
  (3 = (0 + x) / 2) →
  (7 = (11 + y) / 2) →
  (x = 6 ∧ y = 3) := by
  sorry

end other_endpoint_coordinates_l1512_151272


namespace sticker_cost_theorem_l1512_151256

def total_sticker_cost (allowance : ℕ) (card_cost : ℕ) (stickers_per_person : ℕ) : ℕ :=
  2 * allowance - card_cost

theorem sticker_cost_theorem (allowance : ℕ) (card_cost : ℕ) (stickers_per_person : ℕ)
  (h1 : allowance = 9)
  (h2 : card_cost = 10)
  (h3 : stickers_per_person = 2) :
  total_sticker_cost allowance card_cost stickers_per_person = 8 := by
sorry

#eval total_sticker_cost 9 10 2

end sticker_cost_theorem_l1512_151256


namespace linear_system_solution_l1512_151291

theorem linear_system_solution (x y z : ℝ) 
  (eq1 : x + 2*y - z = 8) 
  (eq2 : 2*x - y + z = 18) : 
  8*x + y + z = 70 := by
sorry

end linear_system_solution_l1512_151291


namespace angle_value_proof_l1512_151263

theorem angle_value_proof (α : Real) (P : Real × Real) : 
  0 < α → α < Real.pi / 2 →
  P.1 = Real.sin (-50 * Real.pi / 180) →
  P.2 = Real.cos (130 * Real.pi / 180) →
  P ∈ {p : Real × Real | p.1 = Real.sin (5 * α) ∧ p.2 = Real.cos (5 * α)} →
  α = 44 * Real.pi / 180 := by
sorry

end angle_value_proof_l1512_151263


namespace probability_ratio_l1512_151230

/-- The number of slips in the hat -/
def total_slips : ℕ := 50

/-- The number of distinct numbers on the slips -/
def distinct_numbers : ℕ := 10

/-- The number of slips drawn -/
def drawn_slips : ℕ := 5

/-- The number of slips for each number -/
def slips_per_number : ℕ := 5

/-- The probability of drawing all five slips with the same number -/
def p : ℚ := (distinct_numbers : ℚ) / (Nat.choose total_slips drawn_slips : ℚ)

/-- The probability of drawing three slips with one number and two slips with another number -/
def q : ℚ := (Nat.choose distinct_numbers 2 * Nat.choose slips_per_number 3 * Nat.choose slips_per_number 2 : ℚ) / (Nat.choose total_slips drawn_slips : ℚ)

theorem probability_ratio :
  q / p = 450 := by sorry

end probability_ratio_l1512_151230


namespace ellipse_point_distance_l1512_151237

theorem ellipse_point_distance (P : ℝ × ℝ) :
  (P.1^2 / 6 + P.2^2 / 2 = 1) →
  (Real.sqrt ((P.1 + 2)^2 + P.2^2) + Real.sqrt ((P.1 - 2)^2 + P.2^2) +
   Real.sqrt (P.1^2 + (P.2 + 1)^2) + Real.sqrt (P.1^2 + (P.2 - 1)^2) = 4 * Real.sqrt 6) →
  (abs P.2 = Real.sqrt (6 / 13)) :=
by sorry

end ellipse_point_distance_l1512_151237


namespace unique_solution_l1512_151262

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) (α : ℝ) : Prop :=
  ∀ x y : ℝ, f (f (x + y) * f (x - y)) = x^2 + α * y * f y

/-- The main theorem stating the unique solution to the functional equation -/
theorem unique_solution :
  ∃! (α : ℝ) (f : ℝ → ℝ), SatisfiesEquation f α ∧ α = -1 ∧ ∀ x, f x = x :=
sorry

end unique_solution_l1512_151262


namespace car_traveler_speed_ratio_l1512_151250

/-- Represents the bridge in the problem -/
structure Bridge where
  length : ℝ
  mk_pos : length > 0

/-- Represents the traveler in the problem -/
structure Traveler where
  speed : ℝ
  mk_pos : speed > 0

/-- Represents the car in the problem -/
structure Car where
  speed : ℝ
  mk_pos : speed > 0

/-- The main theorem stating the ratio of car speed to traveler speed -/
theorem car_traveler_speed_ratio (b : Bridge) (t : Traveler) (c : Car) :
  (t.speed * (4 / 9) * b.length / t.speed = c.speed * (4 / 9) * b.length / c.speed) →
  (t.speed * (5 / 9) * b.length / t.speed = b.length / c.speed) →
  c.speed / t.speed = 9 := by
  sorry


end car_traveler_speed_ratio_l1512_151250


namespace cylinder_radius_problem_l1512_151274

theorem cylinder_radius_problem (r : ℝ) : 
  let h : ℝ := 3
  let volume_decrease_radius : ℝ := 3 * Real.pi * ((r - 4)^2 - r^2)
  let volume_decrease_height : ℝ := Real.pi * r^2 * (h - (h - 4))
  volume_decrease_radius = volume_decrease_height →
  (r = 6 + 2 * Real.sqrt 3 ∨ r = 6 - 2 * Real.sqrt 3) :=
by sorry

end cylinder_radius_problem_l1512_151274


namespace renovation_project_materials_l1512_151241

/-- The total number of truck-loads of material needed for a renovation project -/
theorem renovation_project_materials :
  let sand := 0.16666666666666666 * Real.pi
  let dirt := 0.3333333333333333 * Real.exp 1
  let cement := 0.16666666666666666 * Real.sqrt 2
  let gravel := 0.25 * Real.log 5
  abs ((sand + dirt + cement + gravel) - 1.8401374808985008) < 1e-10 := by
sorry

end renovation_project_materials_l1512_151241


namespace consecutive_integers_sum_l1512_151223

theorem consecutive_integers_sum (n : ℤ) : 
  n * (n + 1) = 20412 → n + (n + 1) = 287 := by
  sorry

end consecutive_integers_sum_l1512_151223


namespace digit_equality_proof_l1512_151240

theorem digit_equality_proof (n : ℕ) (a : ℕ) (k : ℕ) :
  (n ≥ 4) →
  (a ≤ 9) →
  (k ≥ 1) →
  (∃ n k, n * (n + 1) / 2 = (10^k - 1) * a / 9) ↔ (a = 5 ∨ a = 6) :=
by sorry

end digit_equality_proof_l1512_151240


namespace quadrilateral_diagonal_area_relation_l1512_151297

/-- Given a quadrilateral with area Q divided by its diagonals into 4 triangles with areas A, B, C, and D,
    prove that A * B * C * D = ((A+B)^2 * (B+C)^2 * (C+D)^2 * (D+A)^2) / Q^4 -/
theorem quadrilateral_diagonal_area_relation (Q A B C D : ℝ) 
    (hQ : Q > 0) 
    (hA : A > 0) (hB : B > 0) (hC : C > 0) (hD : D > 0)
    (hSum : A + B + C + D = Q) : 
  A * B * C * D = ((A+B)^2 * (B+C)^2 * (C+D)^2 * (D+A)^2) / Q^4 := by
  sorry

end quadrilateral_diagonal_area_relation_l1512_151297


namespace unique_solution_implies_any_real_l1512_151257

theorem unique_solution_implies_any_real (a : ℝ) : 
  (∃! x : ℝ, x^2 - 2*a*x + a^2 = 0) → ∀ b : ℝ, ∃ a : ℝ, a = b :=
by sorry

end unique_solution_implies_any_real_l1512_151257


namespace prime_quadratic_solution_l1512_151298

theorem prime_quadratic_solution : 
  {n : ℕ+ | Nat.Prime (n^4 - 27*n^2 + 121)} = {2, 5} := by sorry

end prime_quadratic_solution_l1512_151298


namespace carolyn_stitching_rate_l1512_151205

/-- Represents the number of stitches required for a flower -/
def flower_stitches : ℕ := 60

/-- Represents the number of stitches required for a unicorn -/
def unicorn_stitches : ℕ := 180

/-- Represents the number of stitches required for Godzilla -/
def godzilla_stitches : ℕ := 800

/-- Represents the number of unicorns in the embroidery -/
def num_unicorns : ℕ := 3

/-- Represents the number of flowers in the embroidery -/
def num_flowers : ℕ := 50

/-- Represents the total time Carolyn spends embroidering (in minutes) -/
def total_time : ℕ := 1085

/-- Calculates Carolyn's stitching rate -/
def stitching_rate : ℚ :=
  (godzilla_stitches + num_unicorns * unicorn_stitches + num_flowers * flower_stitches) / total_time

theorem carolyn_stitching_rate :
  stitching_rate = 4 := by sorry

end carolyn_stitching_rate_l1512_151205


namespace g_range_l1512_151207

noncomputable def g (x : ℝ) : ℝ :=
  (Real.arccos (x/3))^2 - 2*Real.pi * Real.arcsin (x/3) + (Real.arcsin (x/3))^2 + 
  (Real.pi^2/8) * (x^2 - 4*x + 12)

theorem g_range :
  ∀ x ∈ Set.Icc (-3 : ℝ) 3,
  g x ∈ Set.Icc (Real.pi^2/4 + 9*Real.pi^2/8) (Real.pi^2/4 + 33*Real.pi^2/8) :=
by sorry

end g_range_l1512_151207


namespace number_relationship_l1512_151232

theorem number_relationship (x : ℝ) : 
  (5 * x = 2 * x + 10) → (5 * x - 2 * x = 10) := by sorry

end number_relationship_l1512_151232


namespace beef_order_proof_l1512_151231

/-- Calculates the amount of beef ordered given the costs and total amount --/
def beef_ordered (beef_cost chicken_cost total_cost : ℚ) : ℚ :=
  total_cost / (beef_cost + 2 * chicken_cost)

/-- Proves that the amount of beef ordered is 1000 pounds given the problem conditions --/
theorem beef_order_proof :
  beef_ordered 8 3 14000 = 1000 := by
  sorry

end beef_order_proof_l1512_151231


namespace smallest_lcm_with_gcd_5_l1512_151213

theorem smallest_lcm_with_gcd_5 (k ℓ : ℕ) :
  k ≥ 1000 ∧ k < 10000 ∧ ℓ ≥ 1000 ∧ ℓ < 10000 ∧ Nat.gcd k ℓ = 5 →
  Nat.lcm k ℓ ≥ 201000 :=
by sorry

end smallest_lcm_with_gcd_5_l1512_151213


namespace remainder_problem_l1512_151245

theorem remainder_problem : ∃ x : ℕ, (71 * x) % 9 = 8 := by
  sorry

end remainder_problem_l1512_151245


namespace family_travel_distance_l1512_151227

/-- Proves that the total distance travelled is 448 km given the specified conditions --/
theorem family_travel_distance : 
  ∀ (total_distance : ℝ),
  (total_distance / (2 * 35) + total_distance / (2 * 40) = 12) →
  total_distance = 448 := by
sorry

end family_travel_distance_l1512_151227


namespace linear_control_periodic_bound_l1512_151221

/-- A function f: ℝ → ℝ is a linear control function if |f'(x)| ≤ 1 for all x ∈ ℝ -/
def LinearControlFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, DifferentiableAt ℝ f x ∧ |deriv f x| ≤ 1

/-- A function f: ℝ → ℝ is periodic with period T if f(x + T) = f(x) for all x ∈ ℝ -/
def Periodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  T > 0 ∧ ∀ x : ℝ, f (x + T) = f x

theorem linear_control_periodic_bound 
    (f : ℝ → ℝ) (T : ℝ) 
    (h1 : LinearControlFunction f)
    (h2 : StrictMono f)
    (h3 : Periodic f T) :
    ∀ x₁ x₂ : ℝ, |f x₁ - f x₂| ≤ T := by
  sorry


end linear_control_periodic_bound_l1512_151221
