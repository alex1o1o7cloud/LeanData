import Mathlib

namespace square_area_proof_l3811_381180

theorem square_area_proof (x : ℝ) :
  (6 * x - 27 = 30 - 2 * x) →
  (6 * x - 27) ^ 2 = 248.0625 := by
  sorry

end square_area_proof_l3811_381180


namespace min_value_f_when_m_1_existence_of_m_l3811_381160

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.log x + m / (2 * x)

def g (m : ℝ) (x : ℝ) : ℝ := x - 2 * m

theorem min_value_f_when_m_1 :
  ∃ x₀ > 0, ∀ x > 0, f 1 x₀ ≤ f 1 x ∧ f 1 x₀ = 1 - Real.log 2 := by sorry

theorem existence_of_m :
  ∃ m ∈ Set.Ioo (4/5 : ℝ) 1, ∀ x ∈ Set.Icc (Real.exp (-1)) 1,
    f m x > g m x + 1 := by sorry

end min_value_f_when_m_1_existence_of_m_l3811_381160


namespace seats_filled_percentage_l3811_381127

/-- Given a hall with 700 seats where 175 are vacant, prove that 75% of the seats are filled. -/
theorem seats_filled_percentage (total_seats : ℕ) (vacant_seats : ℕ) 
  (h1 : total_seats = 700) 
  (h2 : vacant_seats = 175) : 
  (((total_seats - vacant_seats : ℚ) / total_seats) * 100 : ℚ) = 75 := by
  sorry

#check seats_filled_percentage

end seats_filled_percentage_l3811_381127


namespace set_B_is_empty_l3811_381125

theorem set_B_is_empty : {x : ℝ | x^2 + 1 = 0} = ∅ := by sorry

end set_B_is_empty_l3811_381125


namespace games_won_is_fifteen_l3811_381103

/-- Represents the number of baseball games played by Dan's high school team. -/
def total_games : ℕ := 18

/-- Represents the number of games lost by Dan's high school team. -/
def games_lost : ℕ := 3

/-- Theorem stating that the number of games won is 15. -/
theorem games_won_is_fifteen : total_games - games_lost = 15 := by
  sorry

end games_won_is_fifteen_l3811_381103


namespace proposition_b_proposition_c_proposition_d_l3811_381107

-- Proposition B
theorem proposition_b (m n : ℝ) (h1 : m > n) (h2 : n > 0) :
  (m + 1) / (n + 1) < m / n := by sorry

-- Proposition C
theorem proposition_c (a b c : ℝ) (h1 : c > a) (h2 : a > b) (h3 : b > 0) :
  a / (c - a) > b / (c - b) := by sorry

-- Proposition D
theorem proposition_d (a b : ℝ) (h1 : a ≥ b) (h2 : b > -1) :
  a / (a + 1) ≥ b / (b + 1) := by sorry

end proposition_b_proposition_c_proposition_d_l3811_381107


namespace expansion_coefficient_l3811_381109

/-- Given that in the expansion of (ax+1)(x+1/x)^6, the coefficient of x^3 is 30, prove that a = 2 -/
theorem expansion_coefficient (a : ℝ) : 
  (∃ k : ℕ, (Nat.choose 6 k) * a = 30 ∧ 6 - 2*k + 1 = 3) → a = 2 := by
  sorry

end expansion_coefficient_l3811_381109


namespace frog_grasshopper_jump_difference_l3811_381137

theorem frog_grasshopper_jump_difference :
  let grasshopper_jump : ℕ := 25
  let frog_jump : ℕ := 40
  frog_jump - grasshopper_jump = 15 := by
  sorry

end frog_grasshopper_jump_difference_l3811_381137


namespace balance_scale_l3811_381149

/-- The weight of the book that balances the scale -/
def book_weight : ℝ := 1.1

/-- The weight of the first item on the scale -/
def weight1 : ℝ := 0.5

/-- The weight of each of the two identical items on the scale -/
def weight2 : ℝ := 0.3

/-- The number of identical items with weight2 -/
def count2 : ℕ := 2

theorem balance_scale :
  book_weight = weight1 + count2 * weight2 := by sorry

end balance_scale_l3811_381149


namespace inequality_system_solution_l3811_381197

-- Define the inequality system
def inequality_system (x : ℝ) : Prop := x + 1 > 0 ∧ x > -3

-- Define the solution set
def solution_set : Set ℝ := {x | x > -1}

-- Theorem statement
theorem inequality_system_solution :
  {x : ℝ | inequality_system x} = solution_set :=
by sorry

end inequality_system_solution_l3811_381197


namespace minimum_value_implies_a_l3811_381140

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x + 3

theorem minimum_value_implies_a (a : ℝ) :
  (∀ x ∈ Set.Icc 1 2, f a x ≥ 4) ∧
  (∃ x ∈ Set.Icc 1 2, f a x = 4) →
  a = Real.exp 1 - 1 := by
  sorry

end minimum_value_implies_a_l3811_381140


namespace intersection_complement_equality_l3811_381143

open Set

def U : Set ℕ := {0, 1, 2, 3, 4, 5}
def M : Set ℕ := {0, 3, 5}
def N : Set ℕ := {1, 4, 5}

theorem intersection_complement_equality : M ∩ (U \ N) = {0, 3} := by
  sorry

end intersection_complement_equality_l3811_381143


namespace rectangular_prism_diagonal_l3811_381183

theorem rectangular_prism_diagonal (l w h : ℝ) (hl : l = 15) (hw : w = 25) (hh : h = 12) :
  Real.sqrt (l^2 + w^2 + h^2) = Real.sqrt 994 := by
  sorry

end rectangular_prism_diagonal_l3811_381183


namespace isosceles_triangle_l3811_381172

theorem isosceles_triangle (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  a + b = Real.tan (C / 2) * (a * Real.tan A + b * Real.tan B) →
  A = B := by
  sorry

end isosceles_triangle_l3811_381172


namespace range_of_m_l3811_381105

def p (m : ℝ) : Prop := ∃ x₀ : ℝ, x₀^2 + m*x₀ + 2*m - 3 < 0

def q (m : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 - m ≤ 0

theorem range_of_m (m : ℝ) :
  (p m ∨ q m) ∧ ¬(p m ∧ q m) → m < 2 ∨ (4 ≤ m ∧ m ≤ 6) :=
sorry

end range_of_m_l3811_381105


namespace inequalities_proof_l3811_381132

theorem inequalities_proof (a b : ℝ) (h : 1/a > 1/b ∧ 1/b > 0) : 
  a^3 < b^3 ∧ Real.sqrt b - Real.sqrt a < Real.sqrt (b - a) := by
  sorry

end inequalities_proof_l3811_381132


namespace greatest_common_divisor_546_180_under_70_l3811_381115

def is_greatest_common_divisor (n : ℕ) : Prop :=
  n ∣ 546 ∧ n < 70 ∧ n ∣ 180 ∧
  ∀ m : ℕ, m ∣ 546 → m < 70 → m ∣ 180 → m ≤ n

theorem greatest_common_divisor_546_180_under_70 :
  is_greatest_common_divisor 6 := by sorry

end greatest_common_divisor_546_180_under_70_l3811_381115


namespace euler_family_mean_age_l3811_381139

def euler_family_ages : List ℝ := [12, 12, 12, 12, 9, 9, 15, 17]

theorem euler_family_mean_age : 
  (euler_family_ages.sum / euler_family_ages.length : ℝ) = 12.25 := by
  sorry

end euler_family_mean_age_l3811_381139


namespace circle_center_proof_l3811_381141

/-- A line in 2D space represented by the equation ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y = l.c

/-- Check if a point is equidistant from two parallel lines -/
def equidistantFromParallelLines (p : Point) (l1 l2 : Line) : Prop :=
  abs (l1.a * p.x + l1.b * p.y - l1.c) = abs (l2.a * p.x + l2.b * p.y - l2.c)

theorem circle_center_proof (l1 l2 l3 : Line) (p : Point) :
  l1.a = 3 ∧ l1.b = -4 ∧ l1.c = 12 ∧
  l2.a = 3 ∧ l2.b = -4 ∧ l2.c = -24 ∧
  l3.a = 1 ∧ l3.b = -2 ∧ l3.c = 0 ∧
  p.x = -6 ∧ p.y = -3 →
  pointOnLine p l3 ∧ equidistantFromParallelLines p l1 l2 :=
by sorry

end circle_center_proof_l3811_381141


namespace last_four_average_l3811_381186

theorem last_four_average (list : List ℝ) : 
  list.length = 7 →
  list.sum / 7 = 62 →
  (list.take 3).sum / 3 = 58 →
  (list.drop 3).sum / 4 = 65 := by
sorry

end last_four_average_l3811_381186


namespace f_properties_l3811_381189

noncomputable section

/-- The function f(x) = (ax+b)e^x -/
def f (a b x : ℝ) : ℝ := (a * x + b) * Real.exp x

/-- The condition that f has an extremum at x = -1 -/
def has_extremum_at_neg_one (a b : ℝ) : Prop :=
  ∃ ε > 0, ∀ x ∈ Set.Ioo (-1 - ε) (-1 + ε), f a b (-1) ≥ f a b x

/-- The condition that f(x) ≥ x^2 + 2x - 1 for x ≥ -1 -/
def satisfies_inequality (a b : ℝ) : Prop :=
  ∀ x ≥ -1, f a b x ≥ x^2 + 2*x - 1

/-- The main theorem -/
theorem f_properties (a b : ℝ) 
  (h1 : has_extremum_at_neg_one a b)
  (h2 : satisfies_inequality a b) :
  b = 0 ∧ 2 / Real.exp 1 ≤ a ∧ a ≤ 2 * Real.exp 1 :=
sorry

end

end f_properties_l3811_381189


namespace unique_square_divisible_by_five_l3811_381150

theorem unique_square_divisible_by_five : ∃! y : ℕ, 
  (∃ n : ℕ, y = n^2) ∧ 
  (∃ k : ℕ, y = 5 * k) ∧ 
  50 < y ∧ y < 120 :=
by
  -- The proof would go here
  sorry

end unique_square_divisible_by_five_l3811_381150


namespace specific_sculpture_surface_area_l3811_381163

/-- Represents a cube sculpture with three layers -/
structure CubeSculpture where
  bottomLayerCount : ℕ
  middleLayerCount : ℕ
  topLayerCount : ℕ
  cubeEdgeLength : ℝ

/-- Calculates the exposed surface area of a cube sculpture -/
def exposedSurfaceArea (sculpture : CubeSculpture) : ℝ :=
  sorry

/-- The theorem stating that the specific sculpture has 55 square meters of exposed surface area -/
theorem specific_sculpture_surface_area :
  let sculpture : CubeSculpture := {
    bottomLayerCount := 9
    middleLayerCount := 8
    topLayerCount := 3
    cubeEdgeLength := 1
  }
  exposedSurfaceArea sculpture = 55 := by
  sorry

end specific_sculpture_surface_area_l3811_381163


namespace product_digit_sum_l3811_381182

/-- Represents a 101-digit number with alternating digits --/
def AlternatingDigitNumber (a b : ℕ) : ℕ := sorry

/-- The first 101-digit number: 1010101...010101 --/
def num1 : ℕ := AlternatingDigitNumber 1 0

/-- The second 101-digit number: 7070707...070707 --/
def num2 : ℕ := AlternatingDigitNumber 7 0

/-- Returns the hundreds digit of a natural number --/
def hundredsDigit (n : ℕ) : ℕ := sorry

/-- Returns the units digit of a natural number --/
def unitsDigit (n : ℕ) : ℕ := sorry

theorem product_digit_sum :
  hundredsDigit (num1 * num2) + unitsDigit (num1 * num2) = 10 := by sorry

end product_digit_sum_l3811_381182


namespace special_triangle_longest_altitudes_sum_l3811_381176

/-- A triangle with sides 8, 15, and 17 -/
structure SpecialTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a = 8
  hb : b = 15
  hc : c = 17

/-- The sum of the lengths of the two longest altitudes in the special triangle -/
def longestAltitudesSum (t : SpecialTriangle) : ℝ := 23

/-- Theorem stating that the sum of the lengths of the two longest altitudes
    in the special triangle is 23 -/
theorem special_triangle_longest_altitudes_sum (t : SpecialTriangle) :
  longestAltitudesSum t = 23 := by sorry

end special_triangle_longest_altitudes_sum_l3811_381176


namespace square_of_binomial_constant_l3811_381192

theorem square_of_binomial_constant (p : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, 16 * x^2 + 40 * x + p = (a * x + b)^2) → p = 25 := by
  sorry

end square_of_binomial_constant_l3811_381192


namespace polynomial_simplification_l3811_381195

theorem polynomial_simplification (x : ℝ) : 
  3 - 5*x - 7*x^2 + 9 + 11*x - 13*x^2 - 15 + 17*x + 19*x^2 + 4*x^3 - 6*x^3 + 8*x^3 = 
  -3 + 23*x - x^2 + 6*x^3 := by
  sorry

end polynomial_simplification_l3811_381195


namespace fox_initial_money_l3811_381136

/-- The number of times Fox crosses the bridge -/
def crossings : ℕ := 3

/-- The toll Fox pays after each crossing -/
def toll : ℕ := 40

/-- Function to calculate Fox's money after n crossings -/
def foxMoney (initial : ℕ) (n : ℕ) : ℤ :=
  (2^n : ℤ) * initial - (2^n - 1) * toll

theorem fox_initial_money :
  ∃ x : ℕ, foxMoney x crossings = 0 ∧ x = 35 := by
  sorry

end fox_initial_money_l3811_381136


namespace bizarre_coin_expected_value_l3811_381177

/-- A bizarre weighted coin with three possible outcomes -/
inductive CoinOutcome
| Heads
| Tails
| Edge

/-- The probability of each outcome for the bizarre weighted coin -/
def probability (outcome : CoinOutcome) : ℚ :=
  match outcome with
  | CoinOutcome.Heads => 1/4
  | CoinOutcome.Tails => 1/2
  | CoinOutcome.Edge => 1/4

/-- The payoff for each outcome of the bizarre weighted coin -/
def payoff (outcome : CoinOutcome) : ℤ :=
  match outcome with
  | CoinOutcome.Heads => 1
  | CoinOutcome.Tails => 3
  | CoinOutcome.Edge => -8

/-- The expected value of flipping the bizarre weighted coin -/
def expected_value : ℚ :=
  (probability CoinOutcome.Heads * payoff CoinOutcome.Heads) +
  (probability CoinOutcome.Tails * payoff CoinOutcome.Tails) +
  (probability CoinOutcome.Edge * payoff CoinOutcome.Edge)

/-- Theorem stating that the expected value of flipping the bizarre weighted coin is -1/4 -/
theorem bizarre_coin_expected_value :
  expected_value = -1/4 := by
  sorry

end bizarre_coin_expected_value_l3811_381177


namespace weight_of_ten_moles_l3811_381151

/-- Represents an iron oxide compound with the number of iron and oxygen atoms -/
structure IronOxide where
  iron_atoms : ℕ
  oxygen_atoms : ℕ

/-- Calculates the molar mass of an iron oxide compound -/
def molar_mass (compound : IronOxide) : ℝ :=
  55.85 * compound.iron_atoms + 16.00 * compound.oxygen_atoms

/-- Calculates the weight of a given number of moles of an iron oxide compound -/
def weight (moles : ℝ) (compound : IronOxide) : ℝ :=
  moles * molar_mass compound

/-- Theorem: The weight of 10 moles of an iron oxide compound is 10 times its molar mass -/
theorem weight_of_ten_moles (compound : IronOxide) :
  weight 10 compound = 10 * molar_mass compound := by
  sorry

#check weight_of_ten_moles

end weight_of_ten_moles_l3811_381151


namespace prank_combinations_l3811_381106

/-- The number of choices for each day of the prank --/
def prank_choices : List Nat := [1, 2, 6, 3, 1]

/-- The total number of combinations for the prank --/
def total_combinations : Nat := prank_choices.prod

/-- Theorem stating that the total number of combinations is 36 --/
theorem prank_combinations :
  total_combinations = 36 := by sorry

end prank_combinations_l3811_381106


namespace quadratic_inequality_range_l3811_381104

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 2 * a * x + 1 > 0) → 0 ≤ a ∧ a < 1 := by
  sorry

end quadratic_inequality_range_l3811_381104


namespace recipe_ratio_change_l3811_381155

/-- Represents the ratio of ingredients in a recipe --/
structure RecipeRatio :=
  (flour : ℚ)
  (water : ℚ)
  (sugar : ℚ)

/-- The original recipe ratio --/
def original_ratio : RecipeRatio :=
  { flour := 7, water := 2, sugar := 1 }

/-- The new recipe ratio --/
def new_ratio : RecipeRatio :=
  { flour := 7, water := 1, sugar := 2 }

/-- The amount of water in the new recipe --/
def new_water_amount : ℚ := 2

/-- The amount of sugar in the new recipe --/
def new_sugar_amount : ℚ := 4

theorem recipe_ratio_change :
  (new_ratio.water - original_ratio.water) = -1 :=
sorry

end recipe_ratio_change_l3811_381155


namespace combined_selling_price_is_3620_l3811_381167

def article1_cost : ℝ := 1200
def article2_cost : ℝ := 800
def article3_cost : ℝ := 600

def article1_profit_rate : ℝ := 0.4
def article2_profit_rate : ℝ := 0.3
def article3_profit_rate : ℝ := 0.5

def selling_price (cost : ℝ) (profit_rate : ℝ) : ℝ :=
  cost * (1 + profit_rate)

def combined_selling_price : ℝ :=
  selling_price article1_cost article1_profit_rate +
  selling_price article2_cost article2_profit_rate +
  selling_price article3_cost article3_profit_rate

theorem combined_selling_price_is_3620 :
  combined_selling_price = 3620 := by
  sorry

end combined_selling_price_is_3620_l3811_381167


namespace jordan_novels_count_l3811_381166

theorem jordan_novels_count :
  ∀ (j a : ℕ),
  a = j / 10 →
  j = a + 108 →
  j = 120 :=
by
  sorry

end jordan_novels_count_l3811_381166


namespace cristinas_pace_l3811_381145

/-- Prove Cristina's pace in a race with given conditions -/
theorem cristinas_pace (race_distance : ℝ) (head_start : ℝ) (nickys_pace : ℝ) (catch_up_time : ℝ)
  (h1 : race_distance = 500)
  (h2 : head_start = 12)
  (h3 : nickys_pace = 3)
  (h4 : catch_up_time = 30) :
  let cristinas_distance := nickys_pace * (head_start + catch_up_time)
  cristinas_distance / catch_up_time = 5.4 := by
  sorry

#check cristinas_pace

end cristinas_pace_l3811_381145


namespace equation_solution_l3811_381169

theorem equation_solution : ∃ x : ℚ, (2 / 7) * (1 / 8) * x = 14 ∧ x = 392 := by
  sorry

end equation_solution_l3811_381169


namespace rabbit_carrots_l3811_381168

theorem rabbit_carrots (rabbit_carrots_per_hole fox_carrots_per_hole : ℕ)
  (hole_difference : ℕ) :
  rabbit_carrots_per_hole = 5 →
  fox_carrots_per_hole = 7 →
  hole_difference = 6 →
  ∃ (rabbit_holes fox_holes : ℕ),
    rabbit_holes = fox_holes + hole_difference ∧
    rabbit_carrots_per_hole * rabbit_holes = fox_carrots_per_hole * fox_holes ∧
    rabbit_carrots_per_hole * rabbit_holes = 105 :=
by sorry

end rabbit_carrots_l3811_381168


namespace diamond_commutative_l3811_381110

-- Define the set T of all non-zero integers
def T : Set Int := {x : Int | x ≠ 0}

-- Define the binary operation ◇
def diamond (a b : T) : Int := 3 * a * b + a + b

-- Theorem statement
theorem diamond_commutative : ∀ (a b : T), diamond a b = diamond b a := by
  sorry

end diamond_commutative_l3811_381110


namespace ambulance_ride_cost_dakota_ambulance_cost_l3811_381134

/-- Calculates the cost of the ambulance ride given hospital expenses and total bill -/
theorem ambulance_ride_cost 
  (days_in_hospital : ℕ) 
  (bed_cost_per_day : ℕ) 
  (specialist_cost_per_hour : ℕ) 
  (specialist_time_minutes : ℕ) 
  (num_specialists : ℕ) 
  (total_bill : ℕ) : ℕ :=
  let bed_cost := days_in_hospital * bed_cost_per_day
  let specialist_time_hours := specialist_time_minutes / 60
  let specialist_cost := num_specialists * (specialist_cost_per_hour * specialist_time_hours)
  let hospital_cost := bed_cost + specialist_cost
  total_bill - hospital_cost

/-- The cost of Dakota's ambulance ride -/
theorem dakota_ambulance_cost : 
  ambulance_ride_cost 3 900 250 15 2 4625 = 1675 := by
  sorry


end ambulance_ride_cost_dakota_ambulance_cost_l3811_381134


namespace man_downstream_speed_l3811_381187

/-- Calculates the downstream speed of a man given his upstream and still water speeds -/
def downstream_speed (upstream_speed still_water_speed : ℝ) : ℝ :=
  2 * still_water_speed - upstream_speed

/-- Theorem: Given a man's upstream speed of 25 kmph and still water speed of 45 kmph, 
    his downstream speed is 65 kmph -/
theorem man_downstream_speed :
  downstream_speed 25 45 = 65 := by
  sorry

end man_downstream_speed_l3811_381187


namespace digit_property_l3811_381178

theorem digit_property (z : Nat) :
  (z < 10) →
  (∀ k : Nat, k ≥ 1 → ∃ n : Nat, n ≥ 1 ∧ n^9 % (10^k) = z^k % (10^k)) ↔
  z = 1 ∨ z = 3 ∨ z = 7 ∨ z = 9 :=
by sorry

end digit_property_l3811_381178


namespace unique_solution_for_odd_prime_l3811_381144

theorem unique_solution_for_odd_prime (p : ℕ) (h_prime : Nat.Prime p) (h_odd : Odd p) :
  ∃! (x y : ℕ), x > 0 ∧ y > 0 ∧ x^2 + p*x = y^2 :=
by sorry

end unique_solution_for_odd_prime_l3811_381144


namespace terrell_hike_distance_l3811_381116

/-- Proves that given Terrell hiked 8.2 miles on Saturday and 9.8 miles in total,
    the distance he hiked on Sunday is 1.6 miles. -/
theorem terrell_hike_distance (saturday_distance : Real) (total_distance : Real)
    (h1 : saturday_distance = 8.2)
    (h2 : total_distance = 9.8) :
    total_distance - saturday_distance = 1.6 := by
  sorry

end terrell_hike_distance_l3811_381116


namespace incircle_touch_point_distance_special_triangle_incircle_touch_point_distance_l3811_381133

/-- Given a triangle with sides a, b, c, and an incircle that touches side c at point P,
    the distance from one endpoint of side c to P is (a + b + c) / 2 - b -/
theorem incircle_touch_point_distance (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c)
  (h₄ : a + b > c) (h₅ : b + c > a) (h₆ : c + a > b) :
  let s := (a + b + c) / 2
  (s - b) = ((a + b + c) / 2) - b :=
by sorry

/-- In a triangle with sides 4, 5, and 6, the distance from one vertex to the point 
    where the incircle touches the opposite side is 2.5 -/
theorem special_triangle_incircle_touch_point_distance :
  let a := 4
  let b := 5
  let c := 6
  let s := (a + b + c) / 2
  (s - b) = 2.5 :=
by sorry

end incircle_touch_point_distance_special_triangle_incircle_touch_point_distance_l3811_381133


namespace parabola_vertex_l3811_381101

/-- Given a quadratic function f(x) = -x^2 + cx + d where c and d are real numbers,
    and the solution to f(x) ≤ 0 is (-∞, -4] ∪ [6, ∞),
    prove that the vertex of the parabola is (1, 25). -/
theorem parabola_vertex (c d : ℝ) 
  (h : ∀ x, -x^2 + c*x + d ≤ 0 ↔ x ∈ Set.Iic (-4) ∪ Set.Ici 6) : 
  let f := fun x => -x^2 + c*x + d
  (1, f 1) = (1, 25) := by sorry

end parabola_vertex_l3811_381101


namespace multiply_y_value_l3811_381111

theorem multiply_y_value (x y : ℝ) (h1 : ∃ (n : ℝ), 5 * x = n * y) 
  (h2 : x * y ≠ 0) (h3 : (1/5 * x) / (1/6 * y) = 0.7200000000000001) : 
  ∃ (n : ℝ), 5 * x = n * y ∧ n = 18 := by
  sorry

end multiply_y_value_l3811_381111


namespace binomial_coefficient_problem_l3811_381194

/-- Given that the coefficient of x^3y^3 in the expansion of (x+ay)^6 is -160, prove that a = -2 -/
theorem binomial_coefficient_problem (a : ℝ) : 
  (Nat.choose 6 3 : ℝ) * a^3 = -160 → a = -2 := by
  sorry

end binomial_coefficient_problem_l3811_381194


namespace cheap_coat_duration_proof_l3811_381100

/-- The duration of the less expensive coat -/
def cheap_coat_duration : ℕ := 5

/-- The cost of the expensive coat -/
def expensive_coat_cost : ℕ := 300

/-- The duration of the expensive coat -/
def expensive_coat_duration : ℕ := 15

/-- The cost of the less expensive coat -/
def cheap_coat_cost : ℕ := 120

/-- The total time period considered -/
def total_time : ℕ := 30

/-- The amount saved by buying the expensive coat over the total time period -/
def amount_saved : ℕ := 120

theorem cheap_coat_duration_proof :
  cheap_coat_duration * cheap_coat_cost * (total_time / cheap_coat_duration) =
  expensive_coat_cost * (total_time / expensive_coat_duration) + amount_saved :=
by sorry

end cheap_coat_duration_proof_l3811_381100


namespace solution_set_f_intersection_condition_l3811_381159

-- Define the function f(x)
def f (m : ℝ) (x : ℝ) : ℝ := m - |x - 1| - |x + 1|

-- Define the function g(x)
def g (x : ℝ) : ℝ := x^2 + 2*x + 3

-- Theorem 1: Solution set of f(x) > 2 when m = 5
theorem solution_set_f (x : ℝ) : f 5 x > 2 ↔ -3/2 < x ∧ x < 3/2 := by sorry

-- Theorem 2: Condition for f(x) and g(x) to always intersect
theorem intersection_condition (m : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, f m y = g y) ↔ m ≥ 4 := by sorry

end solution_set_f_intersection_condition_l3811_381159


namespace abs_gt_two_necessary_not_sufficient_for_x_lt_neg_two_l3811_381175

theorem abs_gt_two_necessary_not_sufficient_for_x_lt_neg_two :
  (∀ x : ℝ, x < -2 → |x| > 2) ∧
  ¬(∀ x : ℝ, |x| > 2 → x < -2) :=
by sorry

end abs_gt_two_necessary_not_sufficient_for_x_lt_neg_two_l3811_381175


namespace four_student_committees_from_six_l3811_381131

theorem four_student_committees_from_six (n k : ℕ) : n = 6 ∧ k = 4 → Nat.choose n k = 15 := by
  sorry

end four_student_committees_from_six_l3811_381131


namespace first_digit_is_two_l3811_381113

/-- Represents a 3-digit number -/
structure ThreeDigitNumber where
  value : ℕ
  isThreeDigit : 100 ≤ value ∧ value < 1000

/-- Checks if a number is divisible by another number -/
def isDivisibleBy (n m : ℕ) : Prop := ∃ k : ℕ, n = m * k

/-- The main theorem -/
theorem first_digit_is_two
  (n : ThreeDigitNumber)
  (h1 : ∃ d : ℕ, d * 2 = n.value)
  (h2 : isDivisibleBy n.value 6)
  (h3 : ∃ d : ℕ, d * 2 = n.value ∧ d = 2)
  : n.value / 100 = 2 := by
  sorry

#check first_digit_is_two

end first_digit_is_two_l3811_381113


namespace fraction_equality_l3811_381153

theorem fraction_equality (a b : ℝ) (x : ℝ) (h1 : x = a / b) (h2 : a ≠ b) (h3 : b ≠ 0) :
  (a + 2*b) / (a - 2*b) = (x + 2) / (x - 2) := by
  sorry

end fraction_equality_l3811_381153


namespace geometric_progression_ratio_l3811_381108

/-- Given an infinitely decreasing geometric progression with sum S and terms a₁, a₂, a₃, ...,
    prove that S / (S - a₁) = a₁ / a₂ -/
theorem geometric_progression_ratio (S a₁ a₂ : ℝ) (a : ℕ → ℝ) :
  (∀ n, a n = a₁ * (a₂ / a₁) ^ (n - 1)) →  -- Geometric progression definition
  (a₂ / a₁ < 1) →                          -- Decreasing condition
  (S = ∑' n, a n) →                        -- S is the sum of the progression
  S / (S - a₁) = a₁ / a₂ := by
sorry

end geometric_progression_ratio_l3811_381108


namespace x_plus_y_value_l3811_381147

theorem x_plus_y_value (x y : ℤ) (h1 : x - y = 200) (h2 : y = 245) : x + y = 690 := by
  sorry

end x_plus_y_value_l3811_381147


namespace probability_one_from_each_name_l3811_381196

def total_cards : ℕ := 12
def alice_letters : ℕ := 5
def bob_letters : ℕ := 7

theorem probability_one_from_each_name :
  let prob_alice_then_bob := (alice_letters : ℚ) / total_cards * bob_letters / (total_cards - 1)
  let prob_bob_then_alice := (bob_letters : ℚ) / total_cards * alice_letters / (total_cards - 1)
  prob_alice_then_bob + prob_bob_then_alice = 35 / 66 := by
  sorry

end probability_one_from_each_name_l3811_381196


namespace quadratic_inequality_solution_l3811_381128

/-- Given a quadratic inequality x^2 + ax + b < 0 with solution set (-1, 4), prove that ab = 12 -/
theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x : ℝ, x^2 + a*x + b < 0 ↔ -1 < x ∧ x < 4) → a * b = 12 := by
  sorry

end quadratic_inequality_solution_l3811_381128


namespace total_weight_of_clothes_l3811_381148

/-- The total weight of clothes collected is 8.58 kg, given that male student's clothes weigh 2.6 kg and female student's clothes weigh 5.98 kg. -/
theorem total_weight_of_clothes (male_clothes : ℝ) (female_clothes : ℝ)
  (h1 : male_clothes = 2.6)
  (h2 : female_clothes = 5.98) :
  male_clothes + female_clothes = 8.58 := by
  sorry

end total_weight_of_clothes_l3811_381148


namespace base7_divisibility_l3811_381120

/-- Converts a base 7 number to decimal --/
def base7ToDecimal (a b c d : ℕ) : ℕ := a * 7^3 + b * 7^2 + c * 7 + d

/-- Checks if a number is divisible by 29 --/
def isDivisibleBy29 (n : ℕ) : Prop := ∃ k : ℕ, n = 29 * k

theorem base7_divisibility :
  ∃! y : ℕ, y ≤ 6 ∧ isDivisibleBy29 (base7ToDecimal 2 y 6 3) :=
sorry

end base7_divisibility_l3811_381120


namespace arithmetic_geometric_sequence_problem_l3811_381119

-- Define arithmetic sequence a_n
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Define geometric sequence b_n
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, b (n + 1) = r * b n

theorem arithmetic_geometric_sequence_problem
  (a : ℕ → ℝ) (b : ℕ → ℝ) (d : ℝ) :
  arithmetic_sequence a d →
  d ≠ 0 →
  2 * a 4 - (a 7)^2 + 2 * a 10 = 0 →
  geometric_sequence b →
  b 7 = a 7 →
  b 5 * b 9 = 16 :=
by sorry

end arithmetic_geometric_sequence_problem_l3811_381119


namespace telescope_visual_range_l3811_381135

/-- Given a telescope that increases visual range by 87.5% to 150 kilometers,
    prove that the initial visual range was 80 kilometers. -/
theorem telescope_visual_range (V : ℝ) : V + 0.875 * V = 150 → V = 80 := by
  sorry

end telescope_visual_range_l3811_381135


namespace luke_used_eight_stickers_l3811_381124

/-- The number of stickers Luke used to decorate the greeting card -/
def stickers_used_for_card (initial_stickers bought_stickers birthday_stickers : ℕ)
  (given_to_sister remaining_stickers : ℕ) : ℕ :=
  initial_stickers + bought_stickers + birthday_stickers - given_to_sister - remaining_stickers

/-- Theorem stating that Luke used 8 stickers to decorate the greeting card -/
theorem luke_used_eight_stickers :
  stickers_used_for_card 20 12 20 5 39 = 8 := by
  sorry

end luke_used_eight_stickers_l3811_381124


namespace cost_price_from_profit_loss_equality_l3811_381154

/-- The cost price of an article, given specific profit and loss conditions -/
theorem cost_price_from_profit_loss_equality (selling_price_profit selling_price_loss : ℕ) 
  (h : selling_price_profit = 66 ∧ selling_price_loss = 52) :
  ∃ cost_price : ℕ, 
    (selling_price_profit - cost_price = cost_price - selling_price_loss) ∧ 
    cost_price = 59 := by
  sorry

end cost_price_from_profit_loss_equality_l3811_381154


namespace investment_profit_ratio_l3811_381179

/-- Represents the profit ratio between two investors based on their capital and investment duration. -/
def profit_ratio (capital_a capital_b : ℕ) (duration_a duration_b : ℚ) : ℚ × ℚ :=
  let contribution_a := capital_a * duration_a
  let contribution_b := capital_b * duration_b
  (contribution_a, contribution_b)

/-- Theorem stating that given the specified investments and durations, the profit ratio is 2:1. -/
theorem investment_profit_ratio :
  let (ratio_a, ratio_b) := profit_ratio 27000 36000 12 (9/2)
  ratio_a / ratio_b = 2 := by
  sorry

end investment_profit_ratio_l3811_381179


namespace donation_distribution_l3811_381181

theorem donation_distribution (total : ℝ) (contingency : ℝ) : 
  total = 240 →
  contingency = 30 →
  (3 : ℝ) / 8 * total = total - (1 / 3 * total) - (1 / 4 * (total - 1 / 3 * total)) - contingency :=
by sorry

end donation_distribution_l3811_381181


namespace six_digit_pin_probability_six_digit_pin_probability_value_l3811_381152

/-- The probability of randomly selecting a 6-digit PIN with a non-zero first digit, 
    such that the first two digits are both 6 -/
theorem six_digit_pin_probability : ℝ :=
  let total_pins := 9 * 10^5  -- 9 choices for first digit, 10 choices each for other 5 digits
  let favorable_pins := 10^4  -- 4 digits can be any number from 0 to 9
  favorable_pins / total_pins

/-- The probability is equal to 1/90 -/
theorem six_digit_pin_probability_value : six_digit_pin_probability = 1 / 90 := by
  sorry

end six_digit_pin_probability_six_digit_pin_probability_value_l3811_381152


namespace tangency_points_coordinates_l3811_381174

/-- The coordinates of points of tangency to the discriminant parabola -/
theorem tangency_points_coordinates (p q : ℝ) :
  let parabola := {(x, y) : ℝ × ℝ | x^2 - 4*y = 0}
  let tangent_point := (p, q)
  ∃ (p₀ q₀ : ℝ), (p₀, q₀) ∈ parabola ∧
    (p₀ = p + Real.sqrt (p^2 - 4*q) ∨ p₀ = p - Real.sqrt (p^2 - 4*q)) ∧
    q₀ = (p^2 - 2*q + p * Real.sqrt (p^2 - 4*q)) / 2 ∨
    q₀ = (p^2 - 2*q - p * Real.sqrt (p^2 - 4*q)) / 2 :=
by sorry

end tangency_points_coordinates_l3811_381174


namespace divisibility_by_nine_l3811_381156

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem divisibility_by_nine (n : ℕ) : 
  (sum_of_digits n) % 9 = 0 → n % 9 = 0 := by sorry

end divisibility_by_nine_l3811_381156


namespace square_side_length_l3811_381170

theorem square_side_length (perimeter : ℝ) (area : ℝ) (h1 : perimeter = 44) (h2 : area = 121) :
  ∃ (side : ℝ), side * 4 = perimeter ∧ side * side = area ∧ side = 11 := by
  sorry

end square_side_length_l3811_381170


namespace complex_equation_sum_l3811_381198

theorem complex_equation_sum (a b : ℝ) :
  (a + 2 * Complex.I) * Complex.I = b + Complex.I →
  ∃ (result : ℝ), a + b = result :=
sorry

end complex_equation_sum_l3811_381198


namespace emily_beads_count_l3811_381161

theorem emily_beads_count (beads_per_necklace : ℕ) (necklaces_made : ℕ) 
  (h1 : beads_per_necklace = 5) 
  (h2 : necklaces_made = 4) : 
  beads_per_necklace * necklaces_made = 20 := by
  sorry

end emily_beads_count_l3811_381161


namespace ellipse_equation_l3811_381173

theorem ellipse_equation (A B C : ℝ × ℝ) (h1 : A = (-2, 0)) (h2 : B = (2, 0)) (h3 : C.1^2 + C.2^2 = 5) 
  (h4 : (C.1 - A.1) * (B.2 - A.2) = (C.2 - A.2) * (B.1 - A.1)) :
  ∃ (x y : ℝ), x^2/4 + 3*y^2/4 = 1 ∧ x^2 + y^2 = C.1^2 + C.2^2 := by
  sorry

#check ellipse_equation

end ellipse_equation_l3811_381173


namespace notebooks_last_fifty_days_l3811_381142

/-- The number of days notebooks last given the number of notebooks, pages per notebook, and pages used per day. -/
def notebook_days (num_notebooks : ℕ) (pages_per_notebook : ℕ) (pages_per_day : ℕ) : ℕ :=
  (num_notebooks * pages_per_notebook) / pages_per_day

/-- Theorem stating that 5 notebooks with 40 pages each, used at a rate of 4 pages per day, last for 50 days. -/
theorem notebooks_last_fifty_days :
  notebook_days 5 40 4 = 50 := by
  sorry

end notebooks_last_fifty_days_l3811_381142


namespace multiplication_addition_equality_l3811_381190

theorem multiplication_addition_equality : 26 * 33 + 67 * 26 = 2600 := by sorry

end multiplication_addition_equality_l3811_381190


namespace divisibility_of_integer_part_l3811_381118

theorem divisibility_of_integer_part (k : ℕ+) (n : ℕ) :
  let A : ℝ := k + 1/2 + Real.sqrt (k^2 + 1/4)
  (⌊A^n⌋ : ℤ) % k = 0 := by
  sorry

end divisibility_of_integer_part_l3811_381118


namespace equation_equivalence_l3811_381102

theorem equation_equivalence (x y : ℝ) : 
  (2 * x + y = 1) ↔ (y = 1 - 2 * x) := by sorry

end equation_equivalence_l3811_381102


namespace geometric_sequence_sum_inequality_l3811_381130

/-- A geometric sequence with positive terms and common ratio not equal to 1 -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  (∀ n, a n > 0) ∧ q > 0 ∧ q ≠ 1 ∧ ∀ n, a (n + 1) = q * a n

theorem geometric_sequence_sum_inequality
  (a : ℕ → ℝ) (q : ℝ) (h : GeometricSequence a q) :
  a 1 + a 8 > a 4 + a 5 :=
by sorry

end geometric_sequence_sum_inequality_l3811_381130


namespace algebraic_expression_value_l3811_381158

/-- Given an algebraic expression ax-2, if the value of the expression is 4 when x=2, then a=3 -/
theorem algebraic_expression_value (a : ℝ) : (a * 2 - 2 = 4) → a = 3 := by
  sorry

end algebraic_expression_value_l3811_381158


namespace uniform_price_is_250_l3811_381129

/-- Represents the agreement between an employer and a servant --/
structure Agreement where
  full_year_salary : ℕ
  uniform_included : Bool

/-- Represents the actual outcome of the servant's employment --/
structure Outcome where
  months_worked : ℕ
  salary_received : ℕ
  uniform_received : Bool

/-- Calculates the price of the uniform given the agreement and outcome --/
def uniform_price (agreement : Agreement) (outcome : Outcome) : ℕ :=
  agreement.full_year_salary - outcome.salary_received

/-- Theorem stating that under the given conditions, the uniform price is 250 --/
theorem uniform_price_is_250 (agreement : Agreement) (outcome : Outcome) :
  agreement.full_year_salary = 500 ∧
  agreement.uniform_included = true ∧
  outcome.months_worked = 9 ∧
  outcome.salary_received = 250 ∧
  outcome.uniform_received = true →
  uniform_price agreement outcome = 250 := by
  sorry

#eval uniform_price
  { full_year_salary := 500, uniform_included := true }
  { months_worked := 9, salary_received := 250, uniform_received := true }

end uniform_price_is_250_l3811_381129


namespace rectangular_prism_inequality_l3811_381126

theorem rectangular_prism_inequality (a b c l : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hl : l > 0)
  (h_diagonal : l^2 = a^2 + b^2 + c^2) : 
  (l^4 - a^4) * (l^4 - b^4) * (l^4 - c^4) ≥ 512 * a^4 * b^4 * c^4 := by
  sorry

end rectangular_prism_inequality_l3811_381126


namespace trigonometric_sum_equals_three_plus_sqrt_three_l3811_381199

theorem trigonometric_sum_equals_three_plus_sqrt_three :
  let sin_30 : ℝ := 1/2
  let cos_30 : ℝ := Real.sqrt 3 / 2
  let tan_30 : ℝ := sin_30 / cos_30
  3 * tan_30 + 6 * sin_30 = 3 + Real.sqrt 3 := by
sorry

end trigonometric_sum_equals_three_plus_sqrt_three_l3811_381199


namespace problem_solution_l3811_381112

noncomputable section

def f (x : ℝ) : ℝ := Real.log x

def g (m n x : ℝ) : ℝ := m * (x + n) / (x + 1)

def tangent_perpendicular (n : ℝ) : Prop :=
  let f' : ℝ → ℝ := λ x => 1 / x
  let g' : ℝ → ℝ := λ x => (1 - n) / ((x + 1) ^ 2)
  f' 1 * g' 1 = -1

def inequality_holds (m n : ℝ) : Prop :=
  ∀ x > 0, |f x| ≥ |g m n x|

theorem problem_solution :
  (∃ n : ℝ, tangent_perpendicular n ∧ n = 5) ∧
  (∃ n : ℝ, ∃ m : ℝ, m > 0 ∧ inequality_holds m n ∧ n = -1 ∧
    (∀ m' > 0, inequality_holds m' n → m' ≤ m) ∧ m = 2) := by sorry

end

end problem_solution_l3811_381112


namespace x_gt_2_necessary_not_sufficient_for_x_gt_3_l3811_381146

theorem x_gt_2_necessary_not_sufficient_for_x_gt_3 :
  (∀ x : ℝ, x > 3 → x > 2) ∧ 
  (∃ x : ℝ, x > 2 ∧ ¬(x > 3)) := by
  sorry

end x_gt_2_necessary_not_sufficient_for_x_gt_3_l3811_381146


namespace partial_fraction_decomposition_l3811_381138

theorem partial_fraction_decomposition (x : ℝ) (h1 : x ≠ 4) (h2 : x ≠ 7) :
  6 * x / ((x - 7) * (x - 4)^2) = 
    14/3 / (x - 7) + 26/33 / (x - 4) + (-8) / (x - 4)^2 := by
  sorry

end partial_fraction_decomposition_l3811_381138


namespace tangent_circles_distance_l3811_381162

/-- Given two circles of radius 5 that are externally tangent to each other
    and internally tangent to a circle of radius 13, the distance between
    their points of tangency with the larger circle is 2√39. -/
theorem tangent_circles_distance (r₁ r₂ R : ℝ) (h₁ : r₁ = 5) (h₂ : r₂ = 5) (h₃ : R = 13) :
  let d := 2 * (R - r₁)  -- distance between centers of small circles and large circle
  let s := r₁ + r₂       -- distance between centers of small circles
  2 * Real.sqrt ((d ^ 2) - (s / 2) ^ 2) = 2 * Real.sqrt 39 :=
by sorry

end tangent_circles_distance_l3811_381162


namespace candy_has_nine_pencils_l3811_381121

-- Define variables
def candy_pencils : ℕ := sorry
def caleb_pencils : ℕ := sorry
def calen_original_pencils : ℕ := sorry
def calen_final_pencils : ℕ := sorry

-- Define conditions
axiom caleb_pencils_def : caleb_pencils = 2 * candy_pencils - 3
axiom calen_original_pencils_def : calen_original_pencils = caleb_pencils + 5
axiom calen_final_pencils_def : calen_final_pencils = calen_original_pencils - 10
axiom calen_final_pencils_value : calen_final_pencils = 10

-- Theorem to prove
theorem candy_has_nine_pencils : candy_pencils = 9 := by sorry

end candy_has_nine_pencils_l3811_381121


namespace factor_x4_minus_81_l3811_381193

theorem factor_x4_minus_81 :
  ∀ x : ℝ, x^4 - 81 = (x - 3) * (x + 3) * (x^2 + 9) :=
by
  sorry

end factor_x4_minus_81_l3811_381193


namespace value_of_x_minus_y_l3811_381191

theorem value_of_x_minus_y (x y z : ℝ) 
  (eq1 : 3 * x - 5 * y = 5)
  (eq2 : x / (x + y) = 5 / 7)
  (eq3 : x + z * y = 10) :
  x - y = 3 := by
sorry

end value_of_x_minus_y_l3811_381191


namespace multiply_by_seven_l3811_381185

theorem multiply_by_seven (x : ℝ) (h : 8 * x = 64) : 7 * x = 56 := by
  sorry

end multiply_by_seven_l3811_381185


namespace bananas_cantaloupe_cost_l3811_381123

-- Define variables for the prices of each item
variable (a : ℚ) -- Price of a sack of apples
variable (b : ℚ) -- Price of a bunch of bananas
variable (c : ℚ) -- Price of a cantaloupe
variable (d : ℚ) -- Price of a carton of dates
variable (h : ℚ) -- Price of a jar of honey

-- Define the conditions
axiom total_cost : a + b + c + d + h = 30
axiom dates_cost : d = 4 * a
axiom cantaloupe_cost : c = 2 * a - b

-- Theorem to prove
theorem bananas_cantaloupe_cost : b + c = 50 / 7 := by
  sorry

end bananas_cantaloupe_cost_l3811_381123


namespace sqrt_sum_quotient_simplification_l3811_381188

theorem sqrt_sum_quotient_simplification :
  (Real.sqrt 27 + Real.sqrt 243) / Real.sqrt 75 = 12 / 5 := by
  sorry

end sqrt_sum_quotient_simplification_l3811_381188


namespace isosceles_triangle_base_length_l3811_381171

/-- An isosceles triangle with perimeter 20 and leg length 7 has a base length of 6 -/
theorem isosceles_triangle_base_length : ∀ (base leg : ℝ),
  leg = 7 → base + 2 * leg = 20 → base = 6 := by
  sorry

end isosceles_triangle_base_length_l3811_381171


namespace product_of_repeating_decimals_l3811_381164

-- Define the repeating decimals
def repeating_137 : ℚ := 137 / 999
def repeating_6 : ℚ := 2 / 3

-- Theorem statement
theorem product_of_repeating_decimals : 
  repeating_137 * repeating_6 = 274 / 2997 := by
  sorry

end product_of_repeating_decimals_l3811_381164


namespace sqrt_x_minus_5_real_l3811_381117

theorem sqrt_x_minus_5_real (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 5) ↔ x ≥ 5 := by sorry

end sqrt_x_minus_5_real_l3811_381117


namespace race_time_calculation_l3811_381184

/-- Represents a race between two runners A and B -/
structure Race where
  length : ℝ  -- Race length in meters
  lead_distance : ℝ  -- Distance by which A beats B
  lead_time : ℝ  -- Time by which A beats B
  a_time : ℝ  -- Time taken by A to complete the race

/-- Theorem stating that for the given race conditions, A's time is 5.25 seconds -/
theorem race_time_calculation (race : Race) 
  (h1 : race.length = 80)
  (h2 : race.lead_distance = 56)
  (h3 : race.lead_time = 7) :
  race.a_time = 5.25 := by
  sorry

#check race_time_calculation

end race_time_calculation_l3811_381184


namespace triangle_with_angle_ratio_1_2_3_is_right_angled_l3811_381165

/-- If the angles of a triangle are in the ratio 1:2:3, then the triangle is right-angled. -/
theorem triangle_with_angle_ratio_1_2_3_is_right_angled (A B C : ℝ) 
  (h_angle_sum : A + B + C = 180) 
  (h_angle_ratio : ∃ (x : ℝ), A = x ∧ B = 2*x ∧ C = 3*x) : 
  C = 90 := by
  sorry

end triangle_with_angle_ratio_1_2_3_is_right_angled_l3811_381165


namespace first_round_cookies_count_l3811_381122

/-- Represents the number of cookies sold in each round -/
structure CookieSales where
  first_round : ℕ
  second_round : ℕ

/-- Calculates the total number of cookies sold -/
def total_cookies (sales : CookieSales) : ℕ :=
  sales.first_round + sales.second_round

/-- Theorem: Given the total cookies sold and the number sold in the second round,
    we can determine the number sold in the first round -/
theorem first_round_cookies_count 
  (sales : CookieSales) 
  (h1 : sales.second_round = 27) 
  (h2 : total_cookies sales = 61) : 
  sales.first_round = 34 := by
  sorry

end first_round_cookies_count_l3811_381122


namespace ten_square_shape_perimeter_l3811_381157

/-- A shape made from unit squares joined edge to edge -/
structure UnitSquareShape where
  /-- The number of unit squares in the shape -/
  num_squares : ℕ
  /-- The perimeter of the shape in cm -/
  perimeter : ℕ

/-- Theorem: A shape made from 10 unit squares has a perimeter of 18 cm -/
theorem ten_square_shape_perimeter :
  ∀ (shape : UnitSquareShape),
    shape.num_squares = 10 →
    shape.perimeter = 18 :=
by
  sorry

end ten_square_shape_perimeter_l3811_381157


namespace smallest_number_with_given_remainders_l3811_381114

theorem smallest_number_with_given_remainders :
  ∃ b : ℕ, b ≥ 0 ∧
    b % 6 = 3 ∧
    b % 5 = 2 ∧
    b % 7 = 2 ∧
    (∀ c : ℕ, c ≥ 0 → c % 6 = 3 → c % 5 = 2 → c % 7 = 2 → b ≤ c) ∧
    b = 177 := by
  sorry

end smallest_number_with_given_remainders_l3811_381114
