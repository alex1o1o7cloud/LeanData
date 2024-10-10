import Mathlib

namespace triangle_problem_l591_59173

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  (a * Real.sin (2 * B) = Real.sqrt 3 * b * Real.sin A) →
  (Real.cos A = 1 / 3) →
  (B = π / 6) ∧
  (Real.sin C = (2 * Real.sqrt 6 + 1) / 6) :=
by sorry

end triangle_problem_l591_59173


namespace solution_system_equations_l591_59151

theorem solution_system_equations :
  ∃ (a b : ℝ), 
    (a * 2 + b * 1 = 7 ∧ a * 2 - b * 1 = 1) → 
    (a - b = -1) :=
by sorry

end solution_system_equations_l591_59151


namespace total_amount_theorem_l591_59103

/-- Calculate the selling price of an item given its purchase price and loss percentage -/
def sellingPrice (purchasePrice : ℚ) (lossPercentage : ℚ) : ℚ :=
  purchasePrice * (1 - lossPercentage / 100)

/-- Calculate the total amount received from selling three items -/
def totalAmountReceived (price1 price2 price3 : ℚ) (loss1 loss2 loss3 : ℚ) : ℚ :=
  sellingPrice price1 loss1 + sellingPrice price2 loss2 + sellingPrice price3 loss3

theorem total_amount_theorem (price1 price2 price3 loss1 loss2 loss3 : ℚ) :
  price1 = 600 ∧ price2 = 800 ∧ price3 = 1000 ∧
  loss1 = 20 ∧ loss2 = 25 ∧ loss3 = 30 →
  totalAmountReceived price1 price2 price3 loss1 loss2 loss3 = 1780 := by
  sorry

end total_amount_theorem_l591_59103


namespace equation_solution_l591_59165

theorem equation_solution : ∃! x : ℝ, 90 + 5 * 12 / (x / 3) = 91 ∧ x = 180 := by
  sorry

end equation_solution_l591_59165


namespace expression_evaluation_l591_59148

theorem expression_evaluation (a b c : ℝ) (ha : a = 13) (hb : b = 17) (hc : c = 19) :
  (b^2 * (1/c - 1/a) + c^2 * (1/a - 1/b) + a^2 * (1/b - 1/c)) /
  (b * (1/c - 1/a) + c * (1/a - 1/b) + a * (1/b - 1/c)) = a + b + c :=
by sorry

end expression_evaluation_l591_59148


namespace a_value_l591_59197

def A : Set ℝ := {0, 2}
def B (a : ℝ) : Set ℝ := {1, a^2}

theorem a_value (a : ℝ) : A ∪ B a = {0, 1, 2, 4} → a = 2 ∨ a = -2 := by
  sorry

end a_value_l591_59197


namespace infinitely_many_primes_with_special_property_l591_59192

theorem infinitely_many_primes_with_special_property :
  ∀ k : ℕ, ∃ (p n : ℕ), 
    p > k ∧ 
    Prime p ∧ 
    n > 0 ∧ 
    ¬(n ∣ (p - 1)) ∧ 
    (p ∣ (Nat.factorial n + 1)) :=
by sorry

end infinitely_many_primes_with_special_property_l591_59192


namespace tan_sqrt3_inequality_l591_59193

open Set Real

-- Define the set of x that satisfies the inequality
def S : Set ℝ := {x | tan x - Real.sqrt 3 ≤ 0}

-- Define the solution set
def T : Set ℝ := {x | ∃ k : ℤ, -π/2 + k*π < x ∧ x ≤ π/3 + k*π}

-- Theorem statement
theorem tan_sqrt3_inequality : S = T := by sorry

end tan_sqrt3_inequality_l591_59193


namespace winnieThePoohServings_l591_59130

/-- Represents the number of servings eaten by each character -/
structure Servings where
  cheburashka : ℕ
  winnieThePooh : ℕ
  carlson : ℕ

/-- The rate at which characters eat relative to each other -/
def eatingRate (s : Servings) : Prop :=
  5 * s.cheburashka = 2 * s.winnieThePooh ∧
  7 * s.winnieThePooh = 3 * s.carlson

/-- The total number of servings eaten by Cheburashka and Carlson -/
def totalServings (s : Servings) : Prop :=
  s.cheburashka + s.carlson = 82

/-- Theorem stating that Winnie-the-Pooh ate 30 servings -/
theorem winnieThePoohServings (s : Servings) 
  (h1 : eatingRate s) (h2 : totalServings s) : s.winnieThePooh = 30 := by
  sorry

end winnieThePoohServings_l591_59130


namespace cockatiel_eats_fifty_grams_weekly_l591_59190

/-- The amount of birdseed a cockatiel eats per week -/
def cockatiel_weekly_consumption (
  boxes_bought : ℕ
  ) (boxes_in_pantry : ℕ
  ) (parrot_weekly_consumption : ℕ
  ) (grams_per_box : ℕ
  ) (weeks_of_feeding : ℕ
  ) : ℕ :=
  let total_boxes := boxes_bought + boxes_in_pantry
  let total_grams := total_boxes * grams_per_box
  let parrot_total_consumption := parrot_weekly_consumption * weeks_of_feeding
  let cockatiel_total_consumption := total_grams - parrot_total_consumption
  cockatiel_total_consumption / weeks_of_feeding

/-- Theorem stating that given the conditions in the problem, 
    the cockatiel eats 50 grams of seeds each week -/
theorem cockatiel_eats_fifty_grams_weekly :
  cockatiel_weekly_consumption 3 5 100 225 12 = 50 := by
  sorry

end cockatiel_eats_fifty_grams_weekly_l591_59190


namespace perimeter_region_with_270_degree_arc_l591_59139

/-- The perimeter of a region formed by two radii and a 270° arc in a circle -/
theorem perimeter_region_with_270_degree_arc (r : ℝ) (h : r = 7) :
  2 * r + (3/4) * (2 * Real.pi * r) = 14 + (21 * Real.pi / 2) := by
  sorry

end perimeter_region_with_270_degree_arc_l591_59139


namespace trapezoid_properties_l591_59108

/-- Represents a trapezoid with side lengths and angles -/
structure Trapezoid where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  α : ℝ
  β : ℝ
  γ : ℝ
  δ : ℝ

/-- The main theorem about trapezoid properties -/
theorem trapezoid_properties (t : Trapezoid) :
  (t.a = t.d * Real.cos t.α + t.b * Real.cos t.β - t.c * Real.cos (t.β + t.γ)) ∧
  (t.a = t.d * Real.cos t.α + t.b * Real.cos t.β - t.c * Real.cos (t.α + t.δ)) ∧
  (t.a * Real.sin t.α = t.c * Real.sin t.δ + t.b * Real.sin (t.α + t.β)) ∧
  (t.a * Real.sin t.β = t.c * Real.sin t.γ + t.d * Real.sin (t.α + t.β)) := by
  sorry

end trapezoid_properties_l591_59108


namespace percent_relation_l591_59178

theorem percent_relation (x y z : ℝ) (h1 : x = 1.3 * y) (h2 : y = 0.6 * z) : 
  x = 0.78 * z := by
  sorry

end percent_relation_l591_59178


namespace roots_sum_square_l591_59142

theorem roots_sum_square (α β : ℝ) : 
  (α^2 - α - 2006 = 0) → 
  (β^2 - β - 2006 = 0) → 
  (α + β = 1) →
  α + β^2 = 2007 := by
sorry

end roots_sum_square_l591_59142


namespace fraction_simplification_l591_59182

theorem fraction_simplification : (10 : ℝ) / (10 * 11 - 10^2) = 1 := by sorry

end fraction_simplification_l591_59182


namespace square_fraction_count_l591_59181

theorem square_fraction_count : ∃! (n : ℤ), 0 < n ∧ n < 25 ∧ ∃ (k : ℤ), (n : ℚ) / (25 - n) = k^2 := by
  sorry

end square_fraction_count_l591_59181


namespace solve_for_y_l591_59162

theorem solve_for_y (x y : ℝ) (h1 : x = 100) (h2 : x^3*y - 3*x^2*y + 3*x*y = 3000000) : 
  y = 3000000 / 970299 := by sorry

end solve_for_y_l591_59162


namespace power_of_power_l591_59126

theorem power_of_power (a : ℝ) : (a^3)^2 = a^6 := by
  sorry

end power_of_power_l591_59126


namespace total_cakes_per_week_l591_59136

/-- Represents the quantities of cakes served during lunch on a weekday -/
structure LunchCakes :=
  (chocolate : ℕ)
  (vanilla : ℕ)
  (cheesecake : ℕ)

/-- Represents the quantities of cakes served during dinner on a weekday -/
structure DinnerCakes :=
  (chocolate : ℕ)
  (vanilla : ℕ)
  (cheesecake : ℕ)
  (carrot : ℕ)

/-- Calculates the total number of cakes served on a weekday -/
def weekdayTotal (lunch : LunchCakes) (dinner : DinnerCakes) : ℕ :=
  lunch.chocolate + lunch.vanilla + lunch.cheesecake +
  dinner.chocolate + dinner.vanilla + dinner.cheesecake + dinner.carrot

/-- Calculates the total number of cakes served on a weekend day -/
def weekendTotal (lunch : LunchCakes) (dinner : DinnerCakes) : ℕ :=
  2 * (lunch.chocolate + lunch.vanilla + lunch.cheesecake +
       dinner.chocolate + dinner.vanilla + dinner.cheesecake + dinner.carrot)

/-- Theorem: The total number of cakes served during an entire week is 522 -/
theorem total_cakes_per_week
  (lunch : LunchCakes)
  (dinner : DinnerCakes)
  (h1 : lunch.chocolate = 6)
  (h2 : lunch.vanilla = 8)
  (h3 : lunch.cheesecake = 10)
  (h4 : dinner.chocolate = 9)
  (h5 : dinner.vanilla = 7)
  (h6 : dinner.cheesecake = 5)
  (h7 : dinner.carrot = 13) :
  5 * weekdayTotal lunch dinner + 2 * weekendTotal lunch dinner = 522 := by
  sorry

end total_cakes_per_week_l591_59136


namespace dogGroupings_eq_2520_l591_59161

/-- The number of ways to divide 12 dogs into groups of 4, 6, and 2,
    with Rover in the 4-dog group and Spot in the 6-dog group. -/
def dogGroupings : ℕ :=
  (Nat.choose 10 3) * (Nat.choose 7 5)

/-- Theorem stating that the number of ways to divide the dogs is 2520. -/
theorem dogGroupings_eq_2520 : dogGroupings = 2520 := by
  sorry

end dogGroupings_eq_2520_l591_59161


namespace hash_four_two_l591_59172

-- Define the # operation
def hash (a b : ℝ) : ℝ := (a^2 + b^2) * (a - b)

-- Theorem statement
theorem hash_four_two : hash 4 2 = 40 := by
  sorry

end hash_four_two_l591_59172


namespace contradiction_assumption_l591_59159

theorem contradiction_assumption (a b c : ℝ) : 
  (¬(a > 0 ∧ b > 0 ∧ c > 0)) ↔ (¬(a > 0) ∨ ¬(b > 0) ∨ ¬(c > 0)) :=
by sorry

end contradiction_assumption_l591_59159


namespace lcm_18_36_l591_59102

theorem lcm_18_36 : Nat.lcm 18 36 = 36 := by
  sorry

end lcm_18_36_l591_59102


namespace surface_area_of_problem_structure_l591_59128

/-- Represents a structure made of unit cubes -/
structure CubeStructure where
  base : Nat × Nat × Nat  -- dimensions of the base cube
  stacked : Nat  -- number of cubes stacked on top
  total : Nat  -- total number of cubes

/-- Calculates the surface area of a cube structure -/
def surfaceArea (cs : CubeStructure) : Nat :=
  sorry

/-- The specific cube structure in the problem -/
def problemStructure : CubeStructure :=
  { base := (2, 2, 2),
    stacked := 4,
    total := 12 }

theorem surface_area_of_problem_structure :
  surfaceArea problemStructure = 32 :=
sorry

end surface_area_of_problem_structure_l591_59128


namespace angle_around_point_l591_59177

/-- 
Given three angles around a point in a plane, where one angle is 130°, 
and one of the other angles (y) is 30° more than the third angle (x), 
prove that x = 100° and y = 130°.
-/
theorem angle_around_point (x y : ℝ) : 
  x + y + 130 = 360 →   -- Sum of angles around a point is 360°
  y = x + 30 →          -- y is 30° more than x
  x = 100 ∧ y = 130 :=  -- Conclusion: x = 100° and y = 130°
by sorry

end angle_around_point_l591_59177


namespace remainder_divisibility_l591_59191

theorem remainder_divisibility (x : ℕ) (h : x > 0) :
  (200 % x = 2) → (398 % x = 2) := by
  sorry

end remainder_divisibility_l591_59191


namespace remainder_1493825_div_6_l591_59120

theorem remainder_1493825_div_6 : (1493825 % 6 = 5) := by
  sorry

end remainder_1493825_div_6_l591_59120


namespace spherical_segment_max_volume_l591_59163

/-- Given a spherical segment with surface area S, its maximum volume V is S √(S / (18π)) -/
theorem spherical_segment_max_volume (S : ℝ) (h : S > 0) :
  ∃ V : ℝ, V = S * Real.sqrt (S / (18 * Real.pi)) ∧
  ∀ (V' : ℝ), (∃ (R h : ℝ), R > 0 ∧ h > 0 ∧ h ≤ 2*R ∧ S = 2 * Real.pi * R * h ∧
                V' = Real.pi * h^2 * (3*R - h) / 3) →
  V' ≤ V :=
sorry

end spherical_segment_max_volume_l591_59163


namespace angle_x_is_72_degrees_l591_59153

-- Define a regular pentagon
structure RegularPentagon where
  -- All sides are equal (implied by regularity)
  -- All angles are equal (implied by regularity)

-- Define the enclosing structure
structure EnclosingStructure where
  pentagon : RegularPentagon
  -- Squares and triangles enclose the pentagon (implied by the structure)

-- Define the angle x formed by two squares and the pentagon
def angle_x (e : EnclosingStructure) : ℝ := sorry

-- Theorem statement
theorem angle_x_is_72_degrees (e : EnclosingStructure) : 
  angle_x e = 72 := by sorry

end angle_x_is_72_degrees_l591_59153


namespace car_speed_theorem_l591_59188

/-- Represents a car with specific driving characteristics -/
structure Car where
  cooldownTime : ℕ  -- Time required for cooldown in hours
  drivingCycleTime : ℕ  -- Time of continuous driving before cooldown in hours
  totalTime : ℕ  -- Total time of the journey in hours
  totalDistance : ℕ  -- Total distance covered in miles

/-- Calculates the speed of the car in miles per hour -/
def calculateSpeed (c : Car) : ℚ :=
  let totalCycles : ℕ := c.totalTime / (c.drivingCycleTime + c.cooldownTime)
  let remainingTime : ℕ := c.totalTime % (c.drivingCycleTime + c.cooldownTime)
  let actualDrivingTime : ℕ := (totalCycles * c.drivingCycleTime) + min remainingTime c.drivingCycleTime
  c.totalDistance / actualDrivingTime

theorem car_speed_theorem (c : Car) 
    (h1 : c.cooldownTime = 1)
    (h2 : c.drivingCycleTime = 5)
    (h3 : c.totalTime = 13)
    (h4 : c.totalDistance = 88) :
  calculateSpeed c = 8 := by
  sorry

end car_speed_theorem_l591_59188


namespace heloise_pets_l591_59166

theorem heloise_pets (total_pets : ℕ) (dogs_given : ℕ) : 
  total_pets = 189 →
  dogs_given = 10 →
  (∃ (dogs cats : ℕ), 
    dogs + cats = total_pets ∧ 
    dogs * 17 = cats * 10) →
  ∃ (remaining_dogs : ℕ), remaining_dogs = 60 :=
by sorry

end heloise_pets_l591_59166


namespace farm_animals_feet_count_l591_59155

theorem farm_animals_feet_count (total_heads : ℕ) (hen_count : ℕ) : 
  total_heads = 48 → hen_count = 28 → 
  (hen_count * 2 + (total_heads - hen_count) * 4 : ℕ) = 136 := by
  sorry

end farm_animals_feet_count_l591_59155


namespace largest_n_with_conditions_l591_59198

theorem largest_n_with_conditions : ∃ n : ℕ, n = 289 ∧ 
  (∃ m : ℤ, n^2 = (m+1)^3 - m^3) ∧
  (∃ k : ℕ, 2*n + 99 = k^2) ∧
  (∀ n' : ℕ, n' > n → 
    (¬∃ m : ℤ, n'^2 = (m+1)^3 - m^3) ∨
    (¬∃ k : ℕ, 2*n' + 99 = k^2)) :=
by sorry

end largest_n_with_conditions_l591_59198


namespace product_of_roots_zero_l591_59183

theorem product_of_roots_zero (a b c : ℝ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a^3 - 4*a = 0 →
  b^3 - 4*b = 0 →
  c^3 - 4*c = 0 →
  a * b * c = 0 := by
sorry

end product_of_roots_zero_l591_59183


namespace chord_length_implies_center_l591_59189

/-- Given a circle and a line cutting a chord from it, prove the possible values of the circle's center. -/
theorem chord_length_implies_center (a : ℝ) : 
  (∃ (x y : ℝ), (x - a)^2 + y^2 = 4 ∧ x - y - 2 = 0) → 
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    (x₁ - a)^2 + y₁^2 = 4 ∧ 
    (x₂ - a)^2 + y₂^2 = 4 ∧ 
    x₁ - y₁ - 2 = 0 ∧ 
    x₂ - y₂ - 2 = 0 ∧ 
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 8) → 
  a = 0 ∨ a = 4 := by
sorry

end chord_length_implies_center_l591_59189


namespace altitude_difference_example_l591_59152

/-- The difference between the highest and lowest altitudes among three given altitudes -/
def altitude_difference (a b c : Int) : Int :=
  max a (max b c) - min a (min b c)

/-- Theorem stating that the altitude difference for the given values is 77 meters -/
theorem altitude_difference_example : altitude_difference (-102) (-80) (-25) = 77 := by
  sorry

end altitude_difference_example_l591_59152


namespace distance_between_points_l591_59185

theorem distance_between_points (d : ℝ) : 
  (∃ (x : ℝ), d / 2 + x = d - 5) ∧ 
  (d / 2 + d / 2 - 45 / 8 = 45 / 8) → 
  d = 90 := by
sorry

end distance_between_points_l591_59185


namespace smallest_prime_divisor_of_sum_l591_59175

theorem smallest_prime_divisor_of_sum (n : ℕ) : 
  2 = Nat.minFac (3^13 + 9^11) := by sorry

end smallest_prime_divisor_of_sum_l591_59175


namespace pythagorean_triple_with_ratio_exists_l591_59160

theorem pythagorean_triple_with_ratio_exists (k : ℚ) (hk : k > 1) :
  ∃ (a b c : ℕ+), (a.val^2 + b.val^2 = c.val^2) ∧ ((a.val + c.val) / b.val : ℚ) = k := by
  sorry

end pythagorean_triple_with_ratio_exists_l591_59160


namespace last_digits_of_powers_l591_59127

theorem last_digits_of_powers : 
  (∃ n : ℕ, 1989^1989 ≡ 9 [MOD 10]) ∧
  (∃ n : ℕ, 1989^1992 ≡ 1 [MOD 10]) ∧
  (∃ n : ℕ, 1992^1989 ≡ 2 [MOD 10]) ∧
  (∃ n : ℕ, 1992^1992 ≡ 6 [MOD 10]) :=
by sorry

end last_digits_of_powers_l591_59127


namespace max_product_constraint_l591_59129

theorem max_product_constraint (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 4*y = 1) :
  x * y ≤ 1/16 := by
sorry

end max_product_constraint_l591_59129


namespace system_solution_l591_59146

theorem system_solution :
  ∃! (s : Set (ℝ × ℝ)), s = {(2, 4), (4, 2)} ∧
  ∀ (x y : ℝ), (x / y + y / x) * (x + y) = 15 ∧
                (x^2 / y^2 + y^2 / x^2) * (x^2 + y^2) = 85 →
                (x, y) ∈ s :=
by sorry

end system_solution_l591_59146


namespace integral_sqrt_one_minus_x_squared_l591_59167

theorem integral_sqrt_one_minus_x_squared : ∫ x in (-1)..(1), Real.sqrt (1 - x^2) = π / 2 := by
  sorry

end integral_sqrt_one_minus_x_squared_l591_59167


namespace max_sum_given_product_l591_59105

theorem max_sum_given_product (x y : ℝ) : 
  (2015 + x^2) * (2015 + y^2) = 2^22 → x + y ≤ 2 * Real.sqrt 33 :=
by sorry

end max_sum_given_product_l591_59105


namespace group_meal_cost_l591_59119

/-- Calculates the total cost for a group meal including tax and tip -/
def calculate_total_cost (vegetarian_price chicken_price steak_price kids_price : ℚ)
                         (tax_rate tip_rate : ℚ)
                         (vegetarian_count chicken_count steak_count kids_count : ℕ) : ℚ :=
  let subtotal := vegetarian_price * vegetarian_count +
                  chicken_price * chicken_count +
                  steak_price * steak_count +
                  kids_price * kids_count
  let tax := subtotal * tax_rate
  let tip := subtotal * tip_rate
  subtotal + tax + tip

/-- Theorem stating that the total cost for the given group is $90 -/
theorem group_meal_cost :
  calculate_total_cost 5 7 10 3 (1/10) (15/100) 3 4 2 3 = 90 := by
  sorry

end group_meal_cost_l591_59119


namespace probability_two_females_l591_59143

theorem probability_two_females (total : Nat) (females : Nat) (chosen : Nat) :
  total = 8 →
  females = 5 →
  chosen = 2 →
  (Nat.choose females chosen : ℚ) / (Nat.choose total chosen : ℚ) = 5 / 14 := by
  sorry

end probability_two_females_l591_59143


namespace bruce_initial_amount_l591_59170

def crayons_cost : ℕ := 5 * 5
def books_cost : ℕ := 10 * 5
def calculators_cost : ℕ := 3 * 5
def total_spent : ℕ := crayons_cost + books_cost + calculators_cost
def bags_cost : ℕ := 11 * 10
def initial_amount : ℕ := total_spent + bags_cost

theorem bruce_initial_amount : initial_amount = 200 := by
  sorry

end bruce_initial_amount_l591_59170


namespace raspberry_green_grape_difference_l591_59194

def fruit_salad (green_grapes raspberries red_grapes : ℕ) : Prop :=
  green_grapes + raspberries + red_grapes = 102 ∧
  red_grapes = 67 ∧
  red_grapes = 3 * green_grapes + 7 ∧
  raspberries < green_grapes

theorem raspberry_green_grape_difference 
  (green_grapes raspberries red_grapes : ℕ) :
  fruit_salad green_grapes raspberries red_grapes →
  green_grapes - raspberries = 5 := by
sorry

end raspberry_green_grape_difference_l591_59194


namespace sum_of_seven_terms_l591_59141

/-- An arithmetic sequence with a_4 = 7 -/
def arithmetic_seq (n : ℕ) : ℝ :=
  sorry

/-- Sum of first n terms of the arithmetic sequence -/
def S (n : ℕ) : ℝ :=
  sorry

theorem sum_of_seven_terms :
  arithmetic_seq 4 = 7 → S 7 = 49 :=
sorry

end sum_of_seven_terms_l591_59141


namespace triangle_max_area_triangle_area_eight_exists_l591_59132

/-- Triangle with sides a, b, c and area S -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  S : ℝ
  h1 : 4 * S = a^2 - (b - c)^2
  h2 : b + c = 8
  h3 : a > 0 ∧ b > 0 ∧ c > 0 -- Ensuring positive side lengths

/-- The maximum area of a triangle satisfying the given conditions is 8 -/
theorem triangle_max_area (t : Triangle) : t.S ≤ 8 := by
  sorry

/-- There exists a triangle satisfying the conditions with area equal to 8 -/
theorem triangle_area_eight_exists : ∃ t : Triangle, t.S = 8 := by
  sorry

end triangle_max_area_triangle_area_eight_exists_l591_59132


namespace complex_magnitude_problem_l591_59176

theorem complex_magnitude_problem (z : ℂ) (h : z * (1 + Complex.I) = 2 * Complex.I) :
  Complex.abs (z + 2 * Complex.I) = Real.sqrt 10 := by
  sorry

end complex_magnitude_problem_l591_59176


namespace equation_solution_l591_59140

theorem equation_solution : ∃ x : ℝ, 14*x + 15*x + 18*x + 11 = 152 ∧ x = 3 := by
  sorry

end equation_solution_l591_59140


namespace cone_height_from_lateral_surface_l591_59110

/-- Given a cone whose lateral surface is a semicircle with radius a,
    prove that the height of the cone is (√3/2)a. -/
theorem cone_height_from_lateral_surface (a : ℝ) (h : a > 0) :
  let l := a  -- slant height
  let r := a / 2  -- radius of the base
  let h := Real.sqrt ((l ^ 2) - (r ^ 2))  -- height of the cone
  h = (Real.sqrt 3 / 2) * a :=
by sorry

end cone_height_from_lateral_surface_l591_59110


namespace tangent_circles_triangle_area_l591_59144

/-- The area of the triangle formed by the points of tangency of three
    mutually externally tangent circles with radii 2, 3, and 4 -/
theorem tangent_circles_triangle_area :
  ∃ (A B C : ℝ × ℝ),
    let r₁ : ℝ := 2
    let r₂ : ℝ := 3
    let r₃ : ℝ := 4
    let O₁ : ℝ × ℝ := (0, 0)
    let O₂ : ℝ × ℝ := (r₁ + r₂, 0)
    let O₃ : ℝ × ℝ := (0, r₁ + r₃)
    -- A, B, C are points of tangency
    A.1^2 + A.2^2 = r₁^2 ∧
    (A.1 - (r₁ + r₂))^2 + A.2^2 = r₂^2 ∧
    B.1^2 + B.2^2 = r₁^2 ∧
    B.1^2 + (B.2 - (r₁ + r₃))^2 = r₃^2 ∧
    (C.1 - (r₁ + r₂))^2 + C.2^2 = r₂^2 ∧
    C.1^2 + (C.2 - (r₁ + r₃))^2 = r₃^2 →
    -- Area of triangle ABC
    abs (A.1 * B.2 + B.1 * C.2 + C.1 * A.2 - A.2 * B.1 - B.2 * C.1 - C.2 * A.1) / 2 = 25 / 14 := by
  sorry

end tangent_circles_triangle_area_l591_59144


namespace square_sum_given_diff_and_product_l591_59122

theorem square_sum_given_diff_and_product (x y : ℝ) 
  (h1 : x - y = 18) 
  (h2 : x * y = 9) : 
  x^2 + y^2 = 342 := by
sorry

end square_sum_given_diff_and_product_l591_59122


namespace factorial_sum_representations_l591_59157

/-- For any natural number n ≥ 4, there exist at least n! ways to write n! as a sum of elements
    from the set {1!, 2!, ..., (n-1)!}, where each element can be used multiple times. -/
theorem factorial_sum_representations (n : ℕ) (h : n ≥ 4) :
  ∃ (ways : ℕ), ways ≥ n! ∧
    ∀ (representation : List ℕ),
      (∀ k ∈ representation, k ∈ Finset.range n ∧ k > 0) →
      representation.sum = n! →
      (ways : ℕ) ≥ (representation.map Nat.factorial).sum :=
by sorry

end factorial_sum_representations_l591_59157


namespace machine_value_after_two_years_l591_59149

/-- The market value of a machine after two years, given its initial price and annual depreciation rate. -/
def market_value_after_two_years (initial_price : ℝ) (depreciation_rate : ℝ) : ℝ :=
  initial_price * (1 - depreciation_rate)^2

/-- Theorem stating that a machine purchased for $8000 with a 10% annual depreciation rate
    will have a market value of $6480 after two years. -/
theorem machine_value_after_two_years :
  market_value_after_two_years 8000 0.1 = 6480 := by
  sorry

end machine_value_after_two_years_l591_59149


namespace hyperbola_vertex_distance_l591_59118

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop :=
  4 * x^2 + 24 * x - 4 * y^2 + 16 * y + 44 = 0

/-- The distance between vertices of the hyperbola -/
def vertex_distance : ℝ := 2

/-- Theorem stating that the distance between vertices of the given hyperbola is 2 -/
theorem hyperbola_vertex_distance :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    hyperbola_equation x₁ y₁ ∧
    hyperbola_equation x₂ y₂ ∧
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧
    ∀ (x y : ℝ), hyperbola_equation x y → 
      (x - x₁)^2 + (y - y₁)^2 ≤ (x₁ - x₂)^2 + (y₁ - y₂)^2 ∧
      (x - x₂)^2 + (y - y₂)^2 ≤ (x₁ - x₂)^2 + (y₁ - y₂)^2 ∧
      (x₁ - x₂)^2 + (y₁ - y₂)^2 = vertex_distance^2 :=
by
  sorry

end hyperbola_vertex_distance_l591_59118


namespace larger_city_size_proof_l591_59137

/-- The number of cubic yards in the larger city -/
def larger_city_size : ℕ := 9000

/-- The population density in people per cubic yard -/
def population_density : ℕ := 80

/-- The size of the smaller city in cubic yards -/
def smaller_city_size : ℕ := 6400

/-- The population difference between the larger and smaller city -/
def population_difference : ℕ := 208000

theorem larger_city_size_proof :
  population_density * larger_city_size = 
  population_density * smaller_city_size + population_difference := by
  sorry

end larger_city_size_proof_l591_59137


namespace square_area_side_3_l591_59174

/-- The area of a square with side length 3 is 9 square units. -/
theorem square_area_side_3 : 
  ∀ (s : ℝ), s = 3 → s * s = 9 := by
  sorry

end square_area_side_3_l591_59174


namespace rectangle_arrangement_exists_l591_59195

theorem rectangle_arrangement_exists : ∃ (a b c d : ℕ+), 
  (a * b + c * d = 81) ∧ 
  ((2 * (a + b) = 4 * (c + d)) ∨ (4 * (a + b) = 2 * (c + d))) :=
sorry

end rectangle_arrangement_exists_l591_59195


namespace existence_of_integers_l591_59138

theorem existence_of_integers (a b : ℝ) (h : a ≠ b) : 
  ∃ (m n : ℤ), a * (m : ℝ) + b * (n : ℝ) < 0 ∧ b * (m : ℝ) + a * (n : ℝ) > 0 := by
sorry

end existence_of_integers_l591_59138


namespace regular_polygon_30_degree_central_angle_l591_59184

/-- A regular polygon with a central angle of 30° has 12 sides. -/
theorem regular_polygon_30_degree_central_angle (n : ℕ) : 
  (360 : ℝ) / n = 30 → n = 12 := by sorry

end regular_polygon_30_degree_central_angle_l591_59184


namespace min_odd_counties_big_island_l591_59117

/-- Represents a rectangular county with a diagonal road -/
structure County where
  has_diagonal_road : Bool

/-- Represents an island configuration -/
structure Island where
  counties : List County
  is_valid : Bool

/-- Checks if a given number of counties can form a valid Big Island configuration -/
def is_valid_big_island (n : Nat) : Prop :=
  ∃ (island : Island),
    island.counties.length = n ∧
    n % 2 = 1 ∧
    island.is_valid = true

/-- Theorem stating that 9 is the minimum odd number of counties for a valid Big Island -/
theorem min_odd_counties_big_island :
  (∀ k, k < 9 → k % 2 = 1 → ¬ is_valid_big_island k) ∧
  is_valid_big_island 9 := by
  sorry

end min_odd_counties_big_island_l591_59117


namespace missing_month_sale_correct_grocer_sale_problem_l591_59114

/-- Calculates the missing month's sale given sales data for 5 months and the average sale --/
def calculate_missing_month_sale (sale1 sale2 sale4 sale5 sale6 average_sale : ℕ) : ℕ :=
  6 * average_sale - (sale1 + sale2 + sale4 + sale5 + sale6)

/-- Proves that the calculated missing month's sale satisfies the average sale condition --/
theorem missing_month_sale_correct 
  (sale1 sale2 sale4 sale5 sale6 average_sale : ℕ) :
  let sale3 := calculate_missing_month_sale sale1 sale2 sale4 sale5 sale6 average_sale
  (sale1 + sale2 + sale3 + sale4 + sale5 + sale6) / 6 = average_sale := by
  sorry

/-- Applies the theorem to the specific problem values --/
theorem grocer_sale_problem :
  let sale1 : ℕ := 5921
  let sale2 : ℕ := 5468
  let sale4 : ℕ := 6088
  let sale5 : ℕ := 6433
  let sale6 : ℕ := 5922
  let average_sale : ℕ := 5900
  let sale3 := calculate_missing_month_sale sale1 sale2 sale4 sale5 sale6 average_sale
  sale3 = 5568 ∧ (sale1 + sale2 + sale3 + sale4 + sale5 + sale6) / 6 = average_sale := by
  sorry

end missing_month_sale_correct_grocer_sale_problem_l591_59114


namespace square_fits_in_unit_cube_l591_59112

theorem square_fits_in_unit_cube : ∃ (s : ℝ), s ≥ 1.05 ∧ 
  ∃ (x : ℝ), 0 < x ∧ x < 1 ∧ 
    s = min (Real.sqrt (2 * (1 - x)^2)) (Real.sqrt (1 + 2 * x^2)) :=
sorry

end square_fits_in_unit_cube_l591_59112


namespace corn_purchase_amount_l591_59154

/-- Represents the purchase of corn, beans, and rice -/
structure Purchase where
  corn : ℝ
  beans : ℝ
  rice : ℝ

/-- Checks if a purchase satisfies the given conditions -/
def isValidPurchase (p : Purchase) : Prop :=
  p.corn + p.beans + p.rice = 30 ∧
  1.1 * p.corn + 0.6 * p.beans + 0.9 * p.rice = 24 ∧
  p.rice = 0.5 * p.beans

theorem corn_purchase_amount :
  ∃ (p : Purchase), isValidPurchase p ∧ p.corn = 7.5 := by
  sorry

end corn_purchase_amount_l591_59154


namespace equation_solutions_l591_59196

theorem equation_solutions :
  (∃ x₁ x₂, 3 * (x₁ - 2)^2 = 27 ∧ 3 * (x₂ - 2)^2 = 27 ∧ x₁ = 5 ∧ x₂ = -1) ∧
  (∃ x, (x + 5)^3 + 27 = 0 ∧ x = -8) := by
  sorry

end equation_solutions_l591_59196


namespace right_triangle_hypotenuse_l591_59168

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 30 → b = 40 → c^2 = a^2 + b^2 → c = 50 := by sorry

end right_triangle_hypotenuse_l591_59168


namespace cake_division_l591_59100

theorem cake_division (n_cakes : ℕ) (n_girls : ℕ) (share : ℚ) :
  n_cakes = 11 →
  n_girls = 6 →
  share = 1 + 1/2 + 1/4 + 1/12 →
  ∃ (division : List (List ℚ)),
    (∀ piece ∈ division.join, piece ≠ 1/6) ∧
    (division.length = n_girls) ∧
    (∀ girl_share ∈ division, girl_share.sum = share) ∧
    (division.join.sum = n_cakes) :=
by sorry

end cake_division_l591_59100


namespace max_sum_abc_l591_59150

theorem max_sum_abc (a b c : ℕ) 
  (h1 : a < b) 
  (h2 : a + b = 719) 
  (h3 : c - a = 915) : 
  (∀ x y z : ℕ, x < y → x + y = 719 → z - x = 915 → x + y + z ≤ 1993) ∧ 
  (∃ x y z : ℕ, x < y ∧ x + y = 719 ∧ z - x = 915 ∧ x + y + z = 1993) :=
sorry

end max_sum_abc_l591_59150


namespace infinite_series_sum_equals_one_l591_59164

/-- The sum of the infinite series ∑(n=1 to ∞) (3n^2 - 2n + 1) / (n^4 - n^3 + n^2 - n + 1) is equal to 1. -/
theorem infinite_series_sum_equals_one :
  let a : ℕ → ℚ := λ n => (3*n^2 - 2*n + 1) / (n^4 - n^3 + n^2 - n + 1)
  ∑' n, a n = 1 := by
  sorry

end infinite_series_sum_equals_one_l591_59164


namespace sqrt_equation_solution_l591_59106

theorem sqrt_equation_solution :
  ∀ x : ℚ, (Real.sqrt (7 * x) / Real.sqrt (4 * (x + 2)) = 3) → x = -72 / 29 := by
  sorry

end sqrt_equation_solution_l591_59106


namespace count_missed_toddlers_l591_59133

/-- The number of toddlers Bill missed -/
def toddlers_missed (total_count : ℕ) (double_counted : ℕ) (actual_toddlers : ℕ) : ℕ :=
  actual_toddlers - (total_count - double_counted)

/-- Theorem stating that the number of toddlers Bill missed is equal to
    the actual number of toddlers minus the number he actually counted -/
theorem count_missed_toddlers 
  (total_count : ℕ) (double_counted : ℕ) (actual_toddlers : ℕ) :
  toddlers_missed total_count double_counted actual_toddlers = 
  actual_toddlers - (total_count - double_counted) :=
by
  sorry

end count_missed_toddlers_l591_59133


namespace collinear_probability_in_5x5_grid_l591_59111

-- Define the size of the grid
def gridSize : ℕ := 5

-- Define the number of dots to choose
def chosenDots : ℕ := 5

-- Define the number of collinear sets of 5 dots in a 5x5 grid
def collinearSets : ℕ := 12

-- Define the total number of ways to choose 5 dots out of 25
def totalCombinations : ℕ := Nat.choose (gridSize * gridSize) chosenDots

-- Theorem statement
theorem collinear_probability_in_5x5_grid :
  (collinearSets : ℚ) / totalCombinations = 2 / 8855 :=
sorry

end collinear_probability_in_5x5_grid_l591_59111


namespace intersection_and_union_when_m_is_3_intersection_equals_B_iff_m_leq_1_l591_59125

-- Define sets A and B
def A : Set ℝ := {x | -3 ≤ x ∧ x ≤ 2}
def B (m : ℝ) : Set ℝ := {x | 1 - m ≤ x ∧ x ≤ 3*m - 1}

-- Theorem for part (1)
theorem intersection_and_union_when_m_is_3 :
  (A ∩ B 3 = {x | -2 ≤ x ∧ x ≤ 2}) ∧
  (A ∪ B 3 = {x | -3 ≤ x ∧ x ≤ 8}) := by sorry

-- Theorem for part (2)
theorem intersection_equals_B_iff_m_leq_1 :
  ∀ m : ℝ, (A ∩ B m = B m) ↔ m ≤ 1 := by sorry

end intersection_and_union_when_m_is_3_intersection_equals_B_iff_m_leq_1_l591_59125


namespace expression_value_l591_59199

theorem expression_value (x y z : ℚ) 
  (h1 : 3 * x - 2 * y - 2 * z = 0)
  (h2 : x - 4 * y + 8 * z = 0)
  (h3 : z ≠ 0) :
  (3 * x^2 - 2 * x * y) / (y^2 + 4 * z^2) = 120 / 269 := by
  sorry

end expression_value_l591_59199


namespace distance_from_negative_one_l591_59124

theorem distance_from_negative_one : ∀ x : ℝ, |x - (-1)| = 6 ↔ x = 5 ∨ x = -7 := by
  sorry

end distance_from_negative_one_l591_59124


namespace average_of_specific_odds_l591_59187

def is_odd (n : ℕ) : Prop := n % 2 = 1

def is_in_range (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 10

def is_less_than_6 (n : ℕ) : Prop := n < 6

def meets_conditions (n : ℕ) : Prop :=
  is_odd n ∧ is_in_range n ∧ is_less_than_6 n

def numbers_meeting_conditions : List ℕ :=
  [1, 3, 5]

theorem average_of_specific_odds :
  (numbers_meeting_conditions.sum : ℚ) / numbers_meeting_conditions.length = 3 := by
  sorry

end average_of_specific_odds_l591_59187


namespace inequalities_proof_l591_59158

theorem inequalities_proof (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > 0) :
  (a * b > b * c) ∧ (2022^(a - c) + a > 2022^(b - c) + b) := by
  sorry

end inequalities_proof_l591_59158


namespace parabola_standard_equation_l591_59101

/-- A parabola with vertex at the origin passing through (-2, 4) -/
structure Parabola where
  /-- The equation of the parabola is either x^2 = ay or y^2 = bx for some a, b ∈ ℝ -/
  equation : (∃ a : ℝ, ∀ x y : ℝ, y = a * x^2) ∨ (∃ b : ℝ, ∀ x y : ℝ, x = b * y^2)
  /-- The parabola passes through the point (-2, 4) -/
  point : (∃ a : ℝ, 4 = a * (-2)^2) ∨ (∃ b : ℝ, -2 = b * 4^2)

/-- The standard equation of the parabola is either x^2 = y or y^2 = -8x -/
theorem parabola_standard_equation (p : Parabola) :
  (∀ x y : ℝ, y = x^2) ∨ (∀ x y : ℝ, x = -8 * y^2) :=
sorry

end parabola_standard_equation_l591_59101


namespace acid_mixture_theorem_l591_59115

/-- Represents an acid solution with a given volume and concentration. -/
structure AcidSolution where
  volume : ℝ
  concentration : ℝ

/-- Calculates the amount of pure acid in a solution. -/
def pureAcid (solution : AcidSolution) : ℝ :=
  solution.volume * solution.concentration

/-- Theorem: Mixing 4 liters of 60% acid solution with 16 liters of 75% acid solution
    results in a 72% acid solution with a total volume of 20 liters. -/
theorem acid_mixture_theorem : 
  let solution1 : AcidSolution := ⟨4, 0.6⟩
  let solution2 : AcidSolution := ⟨16, 0.75⟩
  let totalVolume := solution1.volume + solution2.volume
  let totalPureAcid := pureAcid solution1 + pureAcid solution2
  totalVolume = 20 ∧ 
  totalPureAcid / totalVolume = 0.72 := by
  sorry

end acid_mixture_theorem_l591_59115


namespace father_son_age_ratio_l591_59109

theorem father_son_age_ratio : 
  ∀ (father_age son_age : ℕ),
  father_age * son_age = 756 →
  (father_age + 6) / (son_age + 6) = 2 →
  father_age / son_age = 7 / 3 :=
by
  sorry

end father_son_age_ratio_l591_59109


namespace decreasing_linear_function_not_in_first_quadrant_l591_59113

/-- A linear function y = kx + b that decreases as x increases and has a negative y-intercept -/
structure DecreasingLinearFunction where
  k : ℝ
  b : ℝ
  k_neg : k < 0
  b_neg : b < 0

/-- The first quadrant of the Cartesian plane -/
def FirstQuadrant : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 > 0 ∧ p.2 > 0}

/-- The theorem stating that a decreasing linear function with negative y-intercept does not pass through the first quadrant -/
theorem decreasing_linear_function_not_in_first_quadrant (f : DecreasingLinearFunction) :
  ∀ x y, y = f.k * x + f.b → (x, y) ∉ FirstQuadrant := by
  sorry

end decreasing_linear_function_not_in_first_quadrant_l591_59113


namespace school_population_l591_59147

theorem school_population (total : ℚ) 
  (h1 : (2 : ℚ) / 3 * total = total - (1 : ℚ) / 3 * total) 
  (h2 : (1 : ℚ) / 10 * ((1 : ℚ) / 3 * total) = (1 : ℚ) / 3 * total - 90) 
  (h3 : (9 : ℚ) / 10 * ((1 : ℚ) / 3 * total) = 90) : 
  total = 300 := by sorry

end school_population_l591_59147


namespace ellipse_eccentricity_l591_59169

/-- Given an ellipse C and a line l, prove the eccentricity e --/
theorem ellipse_eccentricity (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  ∃ (e : ℝ), 
    (∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1 → 
      ∃ (A B M : ℝ × ℝ), 
        (A.2 = 0 ∧ B.1 = 0) ∧ 
        (M.2 = e * M.1 + a) ∧
        (A.1 = -a / e ∧ B.2 = a) ∧
        (M.1^2 / a^2 + M.2^2 / b^2 = 1) ∧
        ((M.1 - A.1)^2 + (M.2 - A.2)^2 = e^2 * ((B.1 - A.1)^2 + (B.2 - A.2)^2))) →
    e = (Real.sqrt 5 - 1) / 2 := by
  sorry

end ellipse_eccentricity_l591_59169


namespace bigger_part_of_60_l591_59156

theorem bigger_part_of_60 (x y : ℝ) (h1 : x + y = 60) (h2 : 10 * x + 22 * y = 780) 
  (h3 : x > 0) (h4 : y > 0) : max x y = 45 := by
  sorry

end bigger_part_of_60_l591_59156


namespace granger_cisco_spots_l591_59179

/-- The number of spots Rover has -/
def rover_spots : ℕ := 46

/-- The number of spots Cisco has -/
def cisco_spots : ℕ := rover_spots / 2 - 5

/-- The number of spots Granger has -/
def granger_spots : ℕ := 5 * cisco_spots

/-- The total number of spots Granger and Cisco have combined -/
def total_spots : ℕ := granger_spots + cisco_spots

theorem granger_cisco_spots : total_spots = 108 := by
  sorry

end granger_cisco_spots_l591_59179


namespace expression_simplification_l591_59121

theorem expression_simplification (a : ℚ) (h : a = -2) : 
  ((a + 7) / (a - 1) - 2 / (a + 1)) / ((a^2 + 3*a) / (a^2 - 1)) = -1/2 := by
  sorry

end expression_simplification_l591_59121


namespace janinas_pancakes_l591_59123

/-- Calculates the minimum number of pancakes Janina must sell to cover her expenses -/
theorem janinas_pancakes (rent : ℝ) (supplies : ℝ) (taxes_wages : ℝ) (price_per_pancake : ℝ) :
  rent = 75.50 →
  supplies = 28.40 →
  taxes_wages = 32.10 →
  price_per_pancake = 1.75 →
  ∃ n : ℕ, n ≥ 78 ∧ n * price_per_pancake ≥ rent + supplies + taxes_wages :=
by sorry

end janinas_pancakes_l591_59123


namespace pure_imaginary_fraction_l591_59145

theorem pure_imaginary_fraction (a : ℝ) : 
  (Complex.I * (1 - a) = -a - 1) → a = -1 := by
  sorry

end pure_imaginary_fraction_l591_59145


namespace arithmetic_geometric_mean_inequality_l591_59104

theorem arithmetic_geometric_mean_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + y + z) / 3 ≥ (x * y * z) ^ (1/3) := by
  sorry

end arithmetic_geometric_mean_inequality_l591_59104


namespace right_triangle_min_perimeter_l591_59107

theorem right_triangle_min_perimeter (a b c : ℝ) (h_area : a * b / 2 = 1) (h_right : a^2 + b^2 = c^2) :
  a + b + c ≥ 2 * Real.sqrt 2 + 2 :=
sorry

end right_triangle_min_perimeter_l591_59107


namespace egyptian_fraction_decomposition_l591_59171

theorem egyptian_fraction_decomposition (n : ℕ) (h : n ≥ 2) :
  (2 : ℚ) / (2 * n + 1) = 1 / (n + 1) + 1 / ((n + 1) * (2 * n + 1)) := by
  sorry

end egyptian_fraction_decomposition_l591_59171


namespace largest_fraction_l591_59116

theorem largest_fraction (a b c d e : ℚ) : 
  a = 2/5 → b = 3/6 → c = 5/10 → d = 7/15 → e = 8/20 →
  (b ≥ a ∧ b ≥ c ∧ b ≥ d ∧ b ≥ e) ∧
  (c ≥ a ∧ c ≥ b ∧ c ≥ d ∧ c ≥ e) ∧
  b = c := by
  sorry

end largest_fraction_l591_59116


namespace tangent_circle_radius_double_inscribed_l591_59135

/-- Given a right triangle ABC with legs a and b, hypotenuse c, inscribed circle radius r,
    circumscribed circle radius R, and a circle with radius ρ touching both legs and the
    circumscribed circle, prove that ρ = 2r. -/
theorem tangent_circle_radius_double_inscribed (a b c r R ρ : ℝ) :
  a > 0 → b > 0 → c > 0 → r > 0 → R > 0 → ρ > 0 →
  c^2 = a^2 + b^2 →  -- Pythagorean theorem for right triangle
  R = c / 2 →  -- Radius of circumscribed circle is half the hypotenuse
  r = (a + b - c) / 2 →  -- Formula for inscribed circle radius
  ρ^2 - (a + b - c) * ρ = 0 →  -- Equation derived from tangency conditions
  ρ = 2 * r := by
sorry

end tangent_circle_radius_double_inscribed_l591_59135


namespace orange_balls_count_l591_59134

theorem orange_balls_count (total : Nat) (red : Nat) (blue : Nat) (pink : Nat) (orange : Nat) : 
  total = 50 →
  red = 20 →
  blue = 10 →
  total = red + blue + pink + orange →
  pink = 3 * orange →
  orange = 5 := by
sorry

end orange_balls_count_l591_59134


namespace inequality_and_equality_l591_59131

theorem inequality_and_equality (a b : ℝ) (h1 : b ≠ -1) (h2 : b ≠ 0) :
  (b < -1 ∨ b > 0 → (1 + a)^2 / (1 + b) ≤ 1 + a^2 / b) ∧
  ((1 + a)^2 / (1 + b) = 1 + a^2 / b ↔ a = b) := by
  sorry

end inequality_and_equality_l591_59131


namespace train_cars_estimate_l591_59186

/-- The number of cars Trey counted -/
def cars_counted : ℕ := 8

/-- The time (in seconds) Trey spent counting -/
def counting_time : ℕ := 15

/-- The total time (in seconds) the train took to pass -/
def total_time : ℕ := 210

/-- The estimated number of cars in the train -/
def estimated_cars : ℕ := 112

/-- Theorem stating that the estimated number of cars is approximately correct -/
theorem train_cars_estimate :
  abs ((cars_counted : ℚ) / counting_time * total_time - estimated_cars) < 1 := by
  sorry


end train_cars_estimate_l591_59186


namespace range_of_b_over_a_l591_59180

theorem range_of_b_over_a (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h1 : 5 - 3 * a ≤ b) (h2 : b ≤ 4 - a) (h3 : Real.log b ≥ a) :
  ∃ (x : ℝ), x = b / a ∧ e ≤ x ∧ x ≤ 7 :=
sorry

end range_of_b_over_a_l591_59180
