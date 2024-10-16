import Mathlib

namespace NUMINAMATH_CALUDE_sine_cosine_sum_l2688_268877

theorem sine_cosine_sum (α : ℝ) (h : Real.sin (α - π/6) = 1/3) :
  Real.sin (2*α - π/6) + Real.cos (2*α) = 7/9 := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_sum_l2688_268877


namespace NUMINAMATH_CALUDE_arc_length_of_sector_l2688_268885

/-- Given a sector with a central angle of 60° and a radius of 6 cm, 
    the length of the arc is equal to 2π cm. -/
theorem arc_length_of_sector (α : Real) (r : Real) : 
  α = 60 * π / 180 → r = 6 → α * r = 2 * π := by sorry

end NUMINAMATH_CALUDE_arc_length_of_sector_l2688_268885


namespace NUMINAMATH_CALUDE_division_problem_l2688_268833

theorem division_problem (a b q : ℕ) (h1 : a - b = 1370) (h2 : a = 1626) (h3 : a = b * q + 15) : q = 6 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l2688_268833


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l2688_268845

theorem perfect_square_trinomial (x : ℝ) : x^2 - 6*x + 9 = (x - 3)^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l2688_268845


namespace NUMINAMATH_CALUDE_cousin_future_age_l2688_268887

/-- Given the ages of Nick and his relatives, prove the cousin's future age. -/
theorem cousin_future_age (nick_age : ℕ) (sister_age_diff : ℕ) (cousin_age_diff : ℕ) :
  nick_age = 13 →
  sister_age_diff = 6 →
  cousin_age_diff = 3 →
  let sister_age := nick_age + sister_age_diff
  let brother_age := (nick_age + sister_age) / 2
  let cousin_age := brother_age - cousin_age_diff
  cousin_age + (2 * brother_age - cousin_age) = 32 := by
  sorry

end NUMINAMATH_CALUDE_cousin_future_age_l2688_268887


namespace NUMINAMATH_CALUDE_hexagonal_prism_lateral_surface_area_l2688_268827

/-- The lateral surface area of a hexagonal prism -/
def lateral_surface_area (base_side_length : ℝ) (lateral_edge_length : ℝ) : ℝ :=
  6 * base_side_length * lateral_edge_length

/-- Theorem: The lateral surface area of a hexagonal prism with base side length 3 and lateral edge length 4 is 72 -/
theorem hexagonal_prism_lateral_surface_area :
  lateral_surface_area 3 4 = 72 := by
  sorry

end NUMINAMATH_CALUDE_hexagonal_prism_lateral_surface_area_l2688_268827


namespace NUMINAMATH_CALUDE_equality_of_solution_sets_implies_sum_l2688_268802

theorem equality_of_solution_sets_implies_sum (a b : ℝ) : 
  (∀ x : ℝ, |x - 2| > 1 ↔ x^2 + a*x + b > 0) → a + b = -1 := by
  sorry

end NUMINAMATH_CALUDE_equality_of_solution_sets_implies_sum_l2688_268802


namespace NUMINAMATH_CALUDE_unique_positive_zero_implies_negative_a_l2688_268814

/-- The function f(x) = ax³ + 3x² + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 3 * x^2 + 1

/-- The derivative of f(x) -/
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 6 * x

theorem unique_positive_zero_implies_negative_a :
  ∀ a : ℝ, (∃! x₀ : ℝ, f a x₀ = 0 ∧ x₀ > 0) → a < 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_positive_zero_implies_negative_a_l2688_268814


namespace NUMINAMATH_CALUDE_fib_150_mod_9_l2688_268880

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
| 0 => 1
| 1 => 1
| (n + 2) => fib n + fib (n + 1)

-- Define the property of Fibonacci sequence modulo 9 repeating every 24 terms
axiom fib_mod_9_period_24 : ∀ n : ℕ, fib n % 9 = fib (n % 24) % 9

-- Theorem: The remainder when the 150th Fibonacci number is divided by 9 is 8
theorem fib_150_mod_9 : fib 150 % 9 = 8 := by
  sorry

end NUMINAMATH_CALUDE_fib_150_mod_9_l2688_268880


namespace NUMINAMATH_CALUDE_percentage_spent_is_80_percent_l2688_268876

-- Define the costs and money amounts
def cheeseburger_cost : ℚ := 3
def milkshake_cost : ℚ := 5
def cheese_fries_cost : ℚ := 8
def jim_money : ℚ := 20
def cousin_money : ℚ := 10

-- Define the total cost of the meal
def total_cost : ℚ := 2 * cheeseburger_cost + 2 * milkshake_cost + cheese_fries_cost

-- Define the combined money
def combined_money : ℚ := jim_money + cousin_money

-- Theorem to prove
theorem percentage_spent_is_80_percent :
  (total_cost / combined_money) * 100 = 80 := by
  sorry

end NUMINAMATH_CALUDE_percentage_spent_is_80_percent_l2688_268876


namespace NUMINAMATH_CALUDE_first_year_payment_is_twenty_l2688_268864

/-- Calculates the first year payment given the total payment and yearly increases -/
def firstYearPayment (totalPayment : ℚ) (secondYearIncrease thirdYearIncrease fourthYearIncrease : ℚ) : ℚ :=
  (totalPayment - (secondYearIncrease + (secondYearIncrease + thirdYearIncrease) + 
   (secondYearIncrease + thirdYearIncrease + fourthYearIncrease))) / 4

/-- Theorem stating that the first year payment is 20.00 given the problem conditions -/
theorem first_year_payment_is_twenty :
  firstYearPayment 96 2 3 4 = 20 := by
  sorry

#eval firstYearPayment 96 2 3 4

end NUMINAMATH_CALUDE_first_year_payment_is_twenty_l2688_268864


namespace NUMINAMATH_CALUDE_polar_to_cartesian_line_l2688_268851

/-- The curve defined by the polar equation r = 1 / (sin θ + cos θ) is a straight line -/
theorem polar_to_cartesian_line : 
  ∀ (x y : ℝ), 
  (∃ (r θ : ℝ), r > 0 ∧ 
    x = r * Real.cos θ ∧ 
    y = r * Real.sin θ ∧ 
    r = 1 / (Real.sin θ + Real.cos θ)) 
  ↔ (x + y)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_line_l2688_268851


namespace NUMINAMATH_CALUDE_cubic_roots_relation_l2688_268888

theorem cubic_roots_relation (a b c r s t : ℝ) : 
  (∀ x, x^3 + 5*x^2 + 6*x - 13 = 0 ↔ x = a ∨ x = b ∨ x = c) →
  (∀ x, x^3 + r*x^2 + s*x + t = 0 ↔ x = a+1 ∨ x = b+1 ∨ x = c+1) →
  t = -15 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_relation_l2688_268888


namespace NUMINAMATH_CALUDE_carries_strawberry_harvest_l2688_268874

/-- Calculates the expected strawberry harvest from a rectangular garden. -/
def strawberry_harvest (length width planting_density yield_per_plant : ℕ) : ℕ :=
  length * width * planting_density * yield_per_plant

/-- Proves that Carrie's garden will yield 7200 strawberries. -/
theorem carries_strawberry_harvest :
  strawberry_harvest 10 12 5 12 = 7200 := by
  sorry

end NUMINAMATH_CALUDE_carries_strawberry_harvest_l2688_268874


namespace NUMINAMATH_CALUDE_football_cost_is_571_l2688_268809

/-- The cost of Alyssa's purchases -/
def total_cost : ℚ := 12.30

/-- The cost of the marbles Alyssa bought -/
def marbles_cost : ℚ := 6.59

/-- The cost of the football -/
def football_cost : ℚ := total_cost - marbles_cost

/-- Theorem stating that the football cost $5.71 -/
theorem football_cost_is_571 : football_cost = 5.71 := by
  sorry

end NUMINAMATH_CALUDE_football_cost_is_571_l2688_268809


namespace NUMINAMATH_CALUDE_zephyr_island_population_reaches_capacity_l2688_268831

/-- Represents the population growth on Zephyr Island -/
def zephyr_island_population (initial_year : ℕ) (initial_population : ℕ) (years_passed : ℕ) : ℕ :=
  initial_population * (4 ^ (years_passed / 20))

/-- Represents the maximum capacity of Zephyr Island -/
def zephyr_island_capacity (total_acres : ℕ) (acres_per_person : ℕ) : ℕ :=
  total_acres / acres_per_person

/-- Theorem stating that the population will reach or exceed the maximum capacity in 40 years -/
theorem zephyr_island_population_reaches_capacity :
  let initial_year := 2023
  let initial_population := 500
  let total_acres := 30000
  let acres_per_person := 2
  let years_to_capacity := 40
  zephyr_island_population initial_year initial_population years_to_capacity ≥ 
    zephyr_island_capacity total_acres acres_per_person ∧
  zephyr_island_population initial_year initial_population (years_to_capacity - 20) < 
    zephyr_island_capacity total_acres acres_per_person :=
by
  sorry


end NUMINAMATH_CALUDE_zephyr_island_population_reaches_capacity_l2688_268831


namespace NUMINAMATH_CALUDE_product_remainder_zero_l2688_268894

theorem product_remainder_zero : (2357 * 2369 * 2384 * 2391 * 3017 * 3079 * 3082) % 17 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_zero_l2688_268894


namespace NUMINAMATH_CALUDE_selection_ways_10_people_l2688_268824

/-- The number of ways to choose a president, vice-president, and 2-person committee from n people -/
def selection_ways (n : ℕ) : ℕ :=
  n * (n - 1) * (Nat.choose (n - 2) 2)

/-- Theorem stating that there are 2520 ways to make the selection from 10 people -/
theorem selection_ways_10_people :
  selection_ways 10 = 2520 := by
  sorry

end NUMINAMATH_CALUDE_selection_ways_10_people_l2688_268824


namespace NUMINAMATH_CALUDE_regular_square_pyramid_volume_l2688_268811

/-- The volume of a regular square pyramid with base edge length 2 and side edge length √6 is 8/3. -/
theorem regular_square_pyramid_volume : 
  ∀ (V : ℝ) (base_edge side_edge : ℝ),
  base_edge = 2 →
  side_edge = Real.sqrt 6 →
  V = (1 / 3) * base_edge^2 * Real.sqrt (side_edge^2 - (base_edge^2 / 2)) →
  V = 8 / 3 :=
by sorry

end NUMINAMATH_CALUDE_regular_square_pyramid_volume_l2688_268811


namespace NUMINAMATH_CALUDE_sphere_cylinder_volume_ratio_l2688_268803

theorem sphere_cylinder_volume_ratio :
  ∀ (r : ℝ), r > 0 →
  (4 / 3 * Real.pi * r^3) / (Real.pi * r^2 * (2 * r)) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sphere_cylinder_volume_ratio_l2688_268803


namespace NUMINAMATH_CALUDE_terminal_side_quadrant_l2688_268881

theorem terminal_side_quadrant (α : Real) :
  let P : ℝ × ℝ := (Real.sin 2, Real.cos 2)
  (∃ k : ℝ, k > 0 ∧ P = (k * Real.sin α, k * Real.cos α)) →
  Real.sin α > 0 ∧ Real.cos α < 0 :=
by
  sorry

end NUMINAMATH_CALUDE_terminal_side_quadrant_l2688_268881


namespace NUMINAMATH_CALUDE_T_equals_x_plus_one_to_fourth_l2688_268840

theorem T_equals_x_plus_one_to_fourth (x : ℝ) : 
  (x + 2)^4 - 4*(x + 2)^3 + 6*(x + 2)^2 - 4*(x + 2) + 1 = (x + 1)^4 := by
  sorry

end NUMINAMATH_CALUDE_T_equals_x_plus_one_to_fourth_l2688_268840


namespace NUMINAMATH_CALUDE_probability_of_three_pointing_l2688_268805

/-- The number of people in the room -/
def n : ℕ := 5

/-- The probability of one person pointing at two specific others -/
def p : ℚ := 1 / 6

/-- The number of ways to choose 2 people out of n -/
def choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The probability of a group of three all pointing at each other -/
def prob_three_pointing : ℚ := p^3

/-- The main theorem: probability of having a group of three all pointing at each other -/
theorem probability_of_three_pointing :
  (choose_two n : ℚ) * prob_three_pointing = 5 / 108 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_three_pointing_l2688_268805


namespace NUMINAMATH_CALUDE_final_flow_rate_l2688_268898

/-- Represents the flow rate of cleaner through a pipe at different time intervals --/
structure FlowRate :=
  (initial : ℝ)
  (after15min : ℝ)
  (final : ℝ)

/-- Theorem stating that given the initial conditions and total cleaner used, 
    the final flow rate must be 4 ounces per minute --/
theorem final_flow_rate 
  (flow : FlowRate)
  (total_time : ℝ)
  (total_cleaner : ℝ)
  (h1 : flow.initial = 2)
  (h2 : flow.after15min = 3)
  (h3 : total_time = 30)
  (h4 : total_cleaner = 80)
  : flow.final = 4 := by
  sorry

#check final_flow_rate

end NUMINAMATH_CALUDE_final_flow_rate_l2688_268898


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_2_range_of_a_for_solution_l2688_268882

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + 2 * |x - 1|

-- Theorem for the first part of the problem
theorem solution_set_when_a_is_2 :
  {x : ℝ | f 2 x > 5} = {x : ℝ | x < -1/3 ∨ x > 3} :=
sorry

-- Theorem for the second part of the problem
theorem range_of_a_for_solution :
  {a : ℝ | ∃ x, f a x - |x - 1| ≤ |a - 2|} = {a : ℝ | a ≤ 3/2} :=
sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_2_range_of_a_for_solution_l2688_268882


namespace NUMINAMATH_CALUDE_cards_drawn_l2688_268832

theorem cards_drawn (total_cards : ℕ) (face_cards : ℕ) (prob : ℚ) (n : ℕ) : 
  total_cards = 52 →
  face_cards = 12 →
  prob = 12 / 52 →
  (face_cards : ℚ) / n = prob →
  n = total_cards :=
by sorry

end NUMINAMATH_CALUDE_cards_drawn_l2688_268832


namespace NUMINAMATH_CALUDE_inconvenient_transportation_probability_l2688_268895

/-- The probability of selecting exactly 4 villages with inconvenient transportation
    out of 10 randomly selected villages from a group of 15 villages,
    where 7 have inconvenient transportation, is equal to 1/30. -/
theorem inconvenient_transportation_probability :
  let total_villages : ℕ := 15
  let inconvenient_villages : ℕ := 7
  let selected_villages : ℕ := 10
  let target_inconvenient : ℕ := 4
  
  Fintype.card {s : Finset (Fin total_villages) //
    s.card = selected_villages ∧
    (s.filter (λ i => i.val < inconvenient_villages)).card = target_inconvenient} /
  Fintype.card {s : Finset (Fin total_villages) // s.card = selected_villages} = 1 / 30 :=
by sorry

end NUMINAMATH_CALUDE_inconvenient_transportation_probability_l2688_268895


namespace NUMINAMATH_CALUDE_triangle_problem_l2688_268871

theorem triangle_problem (A B C : Real) (a b c : Real) :
  C = π / 3 →
  b = 8 →
  (1 / 2) * a * b * Real.sin C = 10 * Real.sqrt 3 →
  c = 7 ∧ Real.cos (B - C) = 13 / 14 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l2688_268871


namespace NUMINAMATH_CALUDE_unique_abc_sum_l2688_268818

theorem unique_abc_sum (x : ℝ) : 
  x = Real.sqrt ((Real.sqrt 37) / 2 + 3 / 2) →
  ∃! (a b c : ℕ+), 
    x^80 = 2*x^78 + 8*x^76 + 9*x^74 - x^40 + (a : ℝ)*x^36 + (b : ℝ)*x^34 + (c : ℝ)*x^30 ∧
    a + b + c = 151 := by
  sorry

end NUMINAMATH_CALUDE_unique_abc_sum_l2688_268818


namespace NUMINAMATH_CALUDE_triangle_third_side_l2688_268884

theorem triangle_third_side (a b c : ℝ) (angle : ℝ) : 
  a = 9 → b = 12 → angle = 150 * π / 180 → 
  c^2 = a^2 + b^2 - 2*a*b*(angle.cos) → 
  c = Real.sqrt (225 + 108 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_third_side_l2688_268884


namespace NUMINAMATH_CALUDE_shirt_price_proof_l2688_268860

theorem shirt_price_proof (shirt_price pants_price : ℝ) 
  (h1 : shirt_price ≠ pants_price)
  (h2 : 2 * shirt_price + 3 * pants_price = 120)
  (h3 : 3 * pants_price = 0.25 * 120) : 
  shirt_price = 45 := by
sorry

end NUMINAMATH_CALUDE_shirt_price_proof_l2688_268860


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l2688_268801

theorem greatest_three_digit_multiple_of_17 :
  ∀ n : ℕ, n ≤ 999 → n % 17 = 0 → n ≤ 986 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l2688_268801


namespace NUMINAMATH_CALUDE_ship_journey_theorem_l2688_268857

/-- A ship's journey over three days -/
structure ShipJourney where
  day1_distance : ℝ
  day2_multiplier : ℝ
  day3_additional : ℝ
  total_distance : ℝ

/-- The solution to the ship's journey problem -/
def ship_journey_solution (j : ShipJourney) : Prop :=
  j.day1_distance = 100 ∧
  j.day2_multiplier = 3 ∧
  j.total_distance = 810 ∧
  j.total_distance = j.day1_distance + (j.day2_multiplier * j.day1_distance) + 
                     (j.day2_multiplier * j.day1_distance + j.day3_additional) ∧
  j.day3_additional = 110

/-- Theorem stating the solution to the ship's journey problem -/
theorem ship_journey_theorem (j : ShipJourney) :
  ship_journey_solution j → j.day3_additional = 110 :=
by
  sorry


end NUMINAMATH_CALUDE_ship_journey_theorem_l2688_268857


namespace NUMINAMATH_CALUDE_only_C_not_like_terms_l2688_268846

-- Define what it means for two terms to be like terms
def are_like_terms (term1 term2 : ℕ → ℕ → ℝ) : Prop :=
  ∀ a b, ∃ k, term1 a b = k * term2 a b ∨ term2 a b = k * term1 a b

-- Define the terms from the problem
def term_A1 (_ _ : ℕ) : ℝ := -2
def term_A2 (_ _ : ℕ) : ℝ := 12

def term_B1 (a b : ℕ) : ℝ := -2 * a^2 * b
def term_B2 (a b : ℕ) : ℝ := a^2 * b

def term_C1 (m _ : ℕ) : ℝ := 2 * m
def term_C2 (_ n : ℕ) : ℝ := 2 * n

def term_D1 (x y : ℕ) : ℝ := -1 * x^2 * y^2
def term_D2 (x y : ℕ) : ℝ := 12 * x^2 * y^2

-- Theorem stating that only C is not like terms
theorem only_C_not_like_terms :
  are_like_terms term_A1 term_A2 ∧
  are_like_terms term_B1 term_B2 ∧
  ¬(are_like_terms term_C1 term_C2) ∧
  are_like_terms term_D1 term_D2 :=
sorry

end NUMINAMATH_CALUDE_only_C_not_like_terms_l2688_268846


namespace NUMINAMATH_CALUDE_dice_sum_probability_l2688_268807

/-- The number of faces on each die -/
def num_faces : ℕ := 6

/-- The number of dice rolled -/
def num_dice : ℕ := 5

/-- The target sum we're aiming for -/
def target_sum : ℕ := 15

/-- 
The number of ways to achieve the target sum when rolling the specified number of dice.
This is equivalent to the coefficient of x^target_sum in the expansion of (x + x^2 + ... + x^num_faces)^num_dice.
-/
def num_ways_to_achieve_sum : ℕ := 2002

theorem dice_sum_probability : 
  num_ways_to_achieve_sum = 2002 := by sorry

end NUMINAMATH_CALUDE_dice_sum_probability_l2688_268807


namespace NUMINAMATH_CALUDE_min_perimeter_is_18_l2688_268897

/-- Represents a triangle with side lengths a and b, where a = AB = BC and b = AC -/
structure IsoscelesTriangle where
  a : ℕ
  b : ℕ

/-- Represents the incircle and excircles of the triangle -/
structure TriangleCircles (t : IsoscelesTriangle) where
  inradius : ℝ
  exradius_A : ℝ
  exradius_B : ℝ
  exradius_C : ℝ

/-- Represents the smaller circle φ -/
structure SmallerCircle (t : IsoscelesTriangle) (c : TriangleCircles t) where
  radius : ℝ

/-- Checks if the given triangle satisfies all the tangency conditions -/
def satisfiesTangencyConditions (t : IsoscelesTriangle) (c : TriangleCircles t) (φ : SmallerCircle t c) : Prop :=
  c.exradius_A = c.inradius + c.exradius_A ∧
  c.exradius_B = c.inradius + c.exradius_B ∧
  c.exradius_C = c.inradius + c.exradius_C ∧
  φ.radius = c.inradius - c.exradius_A

/-- The main theorem stating the minimum perimeter -/
theorem min_perimeter_is_18 :
  ∃ (t : IsoscelesTriangle) (c : TriangleCircles t) (φ : SmallerCircle t c),
    satisfiesTangencyConditions t c φ ∧
    ∀ (t' : IsoscelesTriangle) (c' : TriangleCircles t') (φ' : SmallerCircle t' c'),
      satisfiesTangencyConditions t' c' φ' →
      2 * t.a + t.b ≤ 2 * t'.a + t'.b ∧
      2 * t.a + t.b = 18 :=
sorry

end NUMINAMATH_CALUDE_min_perimeter_is_18_l2688_268897


namespace NUMINAMATH_CALUDE_greenfield_basketball_association_l2688_268834

/-- The cost of a pair of socks in dollars -/
def sock_cost : ℕ := 6

/-- The cost of a T-shirt in dollars -/
def tshirt_cost : ℕ := sock_cost + 7

/-- The cost for one player's home game equipment in dollars -/
def home_cost : ℕ := 2 * sock_cost + tshirt_cost

/-- The cost for one player's away game equipment in dollars -/
def away_cost : ℕ := sock_cost + tshirt_cost

/-- The total cost for one player's equipment in dollars -/
def player_cost : ℕ := home_cost + away_cost

/-- The total cost for equipping all players in dollars -/
def total_cost : ℕ := 3100

/-- The number of players in the Association -/
def num_players : ℕ := 72

theorem greenfield_basketball_association :
  total_cost = num_players * player_cost := by
  sorry

end NUMINAMATH_CALUDE_greenfield_basketball_association_l2688_268834


namespace NUMINAMATH_CALUDE_marcus_scored_half_l2688_268850

/-- Calculates the percentage of team points scored by Marcus -/
def marcus_percentage (three_point_goals : ℕ) (two_point_goals : ℕ) (team_total : ℕ) : ℚ :=
  let marcus_points := 3 * three_point_goals + 2 * two_point_goals
  (marcus_points : ℚ) / team_total * 100

/-- Proves that Marcus scored 50% of the team's total points -/
theorem marcus_scored_half (three_point_goals two_point_goals team_total : ℕ) 
  (h1 : three_point_goals = 5)
  (h2 : two_point_goals = 10)
  (h3 : team_total = 70) :
  marcus_percentage three_point_goals two_point_goals team_total = 50 := by
sorry

#eval marcus_percentage 5 10 70

end NUMINAMATH_CALUDE_marcus_scored_half_l2688_268850


namespace NUMINAMATH_CALUDE_movie_channels_cost_12_l2688_268862

def basic_cable_cost : ℝ := 15
def total_cost : ℝ := 36

def movie_channel_cost : ℝ → Prop :=
  λ m => m > 0 ∧ 
         m + (m - 3) + basic_cable_cost = total_cost

theorem movie_channels_cost_12 : 
  movie_channel_cost 12 := by sorry

end NUMINAMATH_CALUDE_movie_channels_cost_12_l2688_268862


namespace NUMINAMATH_CALUDE_x_range_l2688_268870

-- Define the propositions p and q
def p (x : ℝ) : Prop := |x - 4| ≤ 6
def q (x : ℝ) : Prop := x^2 + 3*x ≥ 0

-- Define the theorem
theorem x_range :
  ∀ x : ℝ, (¬(p x ∧ q x) ∧ ¬(¬(p x))) → (-2 ≤ x ∧ x < 0) :=
by sorry

end NUMINAMATH_CALUDE_x_range_l2688_268870


namespace NUMINAMATH_CALUDE_intersection_line_of_circles_l2688_268829

-- Define the circles
def circle1 (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 4
def circle2 (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 9

-- Define the line
def line (x y : ℝ) : Prop := x + 2*y - 1 = 0

-- Theorem statement
theorem intersection_line_of_circles :
  ∀ x y : ℝ, circle1 x y ∧ circle2 x y → line x y :=
by sorry

end NUMINAMATH_CALUDE_intersection_line_of_circles_l2688_268829


namespace NUMINAMATH_CALUDE_dumbbell_collision_l2688_268826

/-- Represents a dumbbell with weightless rod and spheres at the ends -/
structure Dumbbell where
  length : ℝ  -- Half-length of the rod
  mass : ℝ    -- Mass of each sphere
  velocity : ℝ -- Initial velocity

/-- Represents the result of a collision between two dumbbells -/
inductive CollisionResult
  | Elastic : CollisionResult
  | Inelastic (ω : ℝ) : CollisionResult

/-- Simulates a collision between two identical dumbbells -/
def collide (d : Dumbbell) : CollisionResult → Prop
  | CollisionResult.Elastic => 
      d.velocity = d.velocity  -- Velocity remains unchanged after elastic collision
  | CollisionResult.Inelastic ω => 
      ω = d.velocity / (2 * d.length)  -- Angular velocity after inelastic collision

theorem dumbbell_collision (d : Dumbbell) :
  (collide d CollisionResult.Elastic) ∧ 
  (collide d (CollisionResult.Inelastic (d.velocity / (2 * d.length)))) :=
by sorry

end NUMINAMATH_CALUDE_dumbbell_collision_l2688_268826


namespace NUMINAMATH_CALUDE_special_polynomial_root_l2688_268879

/-- A fourth degree polynomial with specific root properties -/
structure SpecialPolynomial where
  P : ℝ → ℝ
  degree_four : ∃ (a b c d e : ℝ), ∀ x, P x = a*x^4 + b*x^3 + c*x^2 + d*x + e
  root_one : P 1 = 0
  root_three : P 3 = 0
  root_five : P 5 = 0
  derivative_root_seven : (deriv P) 7 = 0

/-- The remaining root of a SpecialPolynomial is 89/11 -/
theorem special_polynomial_root (p : SpecialPolynomial) : 
  ∃ (x : ℝ), x ≠ 1 ∧ x ≠ 3 ∧ x ≠ 5 ∧ p.P x = 0 ∧ x = 89/11 := by
  sorry

end NUMINAMATH_CALUDE_special_polynomial_root_l2688_268879


namespace NUMINAMATH_CALUDE_sum_of_integers_l2688_268820

theorem sum_of_integers (x y : ℕ+) 
  (h_diff : x - y = 18) 
  (h_prod : x * y = 72) : 
  x + y = 2 * Real.sqrt 153 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l2688_268820


namespace NUMINAMATH_CALUDE_airplane_seats_l2688_268842

theorem airplane_seats : ∀ s : ℕ,
  s ≥ 30 →
  (30 : ℝ) + 0.4 * s + (3/5) * s ≤ s →
  s = 150 :=
by
  sorry

end NUMINAMATH_CALUDE_airplane_seats_l2688_268842


namespace NUMINAMATH_CALUDE_smallest_square_side_exists_valid_division_5_l2688_268823

/-- Represents a square with integer side length -/
structure Square where
  side : ℕ

/-- Represents a division of a square into smaller squares -/
structure SquareDivision where
  original : Square
  parts : List Square
  sum_areas : (parts.map (λ s => s.side ^ 2)).sum = original.side ^ 2

/-- The property we want to prove -/
def is_valid_division (d : SquareDivision) : Prop :=
  d.parts.length = 15 ∧
  (d.parts.filter (λ s => s.side = 1)).length ≥ 12

/-- The main theorem -/
theorem smallest_square_side :
  ∀ d : SquareDivision, is_valid_division d → d.original.side ≥ 5 :=
by sorry

/-- The existence of a valid division with side 5 -/
theorem exists_valid_division_5 :
  ∃ d : SquareDivision, d.original.side = 5 ∧ is_valid_division d :=
by sorry

end NUMINAMATH_CALUDE_smallest_square_side_exists_valid_division_5_l2688_268823


namespace NUMINAMATH_CALUDE_club_diamond_heart_probability_l2688_268891

/-- Represents a standard deck of playing cards -/
structure Deck :=
  (total : Nat)
  (clubs : Nat)
  (diamonds : Nat)
  (hearts : Nat)

/-- The probability of drawing the sequence: club, diamond, heart -/
def sequence_probability (d : Deck) : ℚ :=
  (d.clubs : ℚ) / d.total *
  (d.diamonds : ℚ) / (d.total - 1) *
  (d.hearts : ℚ) / (d.total - 2)

theorem club_diamond_heart_probability :
  let standard_deck : Deck := ⟨52, 13, 13, 13⟩
  sequence_probability standard_deck = 2197 / 132600 := by
  sorry

end NUMINAMATH_CALUDE_club_diamond_heart_probability_l2688_268891


namespace NUMINAMATH_CALUDE_expression_value_l2688_268854

theorem expression_value (a b c d m : ℝ)
  (h1 : a * b = 1)  -- a and b are reciprocals
  (h2 : c + d = 0)  -- c and d are opposites
  (h3 : m = -1)     -- m equals -1
  : 2 * a * b - (c + d) + m^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2688_268854


namespace NUMINAMATH_CALUDE_carl_payment_percentage_l2688_268855

theorem carl_payment_percentage (property_damage medical_bills insurance_percentage carl_owes : ℚ)
  (h1 : property_damage = 40000)
  (h2 : medical_bills = 70000)
  (h3 : insurance_percentage = 80/100)
  (h4 : carl_owes = 22000) :
  carl_owes / (property_damage + medical_bills) = 20/100 := by
  sorry

end NUMINAMATH_CALUDE_carl_payment_percentage_l2688_268855


namespace NUMINAMATH_CALUDE_troy_computer_purchase_l2688_268883

/-- The amount of money Troy needs to buy a new computer -/
def additional_money_needed (new_computer_cost saved_amount old_computer_value : ℕ) : ℕ :=
  new_computer_cost - (saved_amount + old_computer_value)

/-- Theorem stating the amount Troy needs to buy the new computer -/
theorem troy_computer_purchase (new_computer_cost saved_amount old_computer_value : ℕ) 
  (h1 : new_computer_cost = 1200)
  (h2 : saved_amount = 450)
  (h3 : old_computer_value = 150) :
  additional_money_needed new_computer_cost saved_amount old_computer_value = 600 := by
  sorry

#eval additional_money_needed 1200 450 150

end NUMINAMATH_CALUDE_troy_computer_purchase_l2688_268883


namespace NUMINAMATH_CALUDE_lastTwoDigits_7_2012_l2688_268838

/-- The last two digits of 7^n, for any natural number n -/
def lastTwoDigits (n : ℕ) : ℕ :=
  (7^n) % 100

/-- The pattern of last two digits repeats every 4 exponents -/
axiom lastTwoDigitsPattern (k : ℕ) :
  (lastTwoDigits (4*k - 2) = 49) ∧
  (lastTwoDigits (4*k - 1) = 43) ∧
  (lastTwoDigits (4*k) = 1) ∧
  (lastTwoDigits (4*k + 1) = 7)

theorem lastTwoDigits_7_2012 :
  lastTwoDigits 2012 = 1 := by sorry

end NUMINAMATH_CALUDE_lastTwoDigits_7_2012_l2688_268838


namespace NUMINAMATH_CALUDE_simplify_expression_l2688_268849

theorem simplify_expression (x : ℝ) : 
  3*x + 6*x + 9*x + 12*x + 15*x + 18 + 24 = 45*x + 42 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2688_268849


namespace NUMINAMATH_CALUDE_work_for_series_springs_l2688_268843

/-- Work required to stretch a system of two springs in series -/
theorem work_for_series_springs (k₁ k₂ : ℝ) (x : ℝ) (h₁ : k₁ = 6000) (h₂ : k₂ = 12000) (h₃ : x = 0.1) :
  (1 / 2) * (1 / (1 / k₁ + 1 / k₂)) * x^2 = 20 := by
  sorry

#check work_for_series_springs

end NUMINAMATH_CALUDE_work_for_series_springs_l2688_268843


namespace NUMINAMATH_CALUDE_exists_unreachable_all_plus_configuration_l2688_268815

/-- Represents the sign in a cell: + or - -/
inductive Sign
| Plus
| Minus

/-- Represents an 8x8 grid of signs -/
def Grid := Fin 8 → Fin 8 → Sign

/-- Represents the allowed operations: flipping signs in 3x3 or 4x4 squares -/
def flip_square (g : Grid) (top_left : Fin 8 × Fin 8) (size : Fin 2) : Grid :=
  sorry

/-- Counts the number of minus signs in specific columns of the grid -/
def count_minus_outside_columns_3_6 (g : Grid) : Nat :=
  sorry

/-- Theorem stating that there exists a grid configuration that cannot be transformed to all plus signs -/
theorem exists_unreachable_all_plus_configuration :
  ∃ (initial : Grid), ¬∃ (final : Grid),
    (∀ i j, final i j = Sign.Plus) ∧
    (∃ (ops : List ((Fin 8 × Fin 8) × Fin 2)),
      final = ops.foldl (λ g (tl, s) => flip_square g tl s) initial) :=
  sorry

end NUMINAMATH_CALUDE_exists_unreachable_all_plus_configuration_l2688_268815


namespace NUMINAMATH_CALUDE_liberty_middle_school_math_competition_l2688_268872

theorem liberty_middle_school_math_competition (sixth_graders seventh_graders : ℕ) : 
  (3 * sixth_graders = 7 * seventh_graders) →
  (sixth_graders + seventh_graders = 140) →
  sixth_graders = 61 := by
  sorry

end NUMINAMATH_CALUDE_liberty_middle_school_math_competition_l2688_268872


namespace NUMINAMATH_CALUDE_shaded_area_sum_l2688_268866

/-- Represents the shaded area in each step of the square division pattern -/
def shadedAreaSeries : ℕ → ℚ
  | 0 => 1/4
  | n + 1 => (shadedAreaSeries n) / 16

/-- The sum of the infinite geometric series representing the total shaded area -/
def totalShadedArea : ℚ := 4/15

theorem shaded_area_sum :
  (∑' n, shadedAreaSeries n) = totalShadedArea := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_sum_l2688_268866


namespace NUMINAMATH_CALUDE_intersection_M_P_l2688_268861

-- Define the sets M and P
def M (a : ℝ) : Set ℝ := {x | x > a ∧ a^2 - 12*a + 20 < 0}
def P : Set ℝ := {x | x ≤ 10}

-- Theorem statement
theorem intersection_M_P (a : ℝ) : M a ∩ P = {x | a < x ∧ x ≤ 10} :=
by sorry

end NUMINAMATH_CALUDE_intersection_M_P_l2688_268861


namespace NUMINAMATH_CALUDE_expression_evaluation_l2688_268875

theorem expression_evaluation (a b : ℤ) (h1 : a = 4) (h2 : b = -1) :
  -2*a - b^2 + 2*a*b = -17 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2688_268875


namespace NUMINAMATH_CALUDE_sequence_properties_l2688_268817

def a (n : ℕ+) : ℚ := (3 * n - 2) / (3 * n + 1)

theorem sequence_properties :
  (a 10 = 28 / 31) ∧
  (a 3 = 7 / 10) ∧
  (∀ n : ℕ+, 0 < a n ∧ a n < 1) := by
  sorry

end NUMINAMATH_CALUDE_sequence_properties_l2688_268817


namespace NUMINAMATH_CALUDE_clock_angle_at_7_clock_angle_at_7_is_150_l2688_268841

/-- The smaller angle formed by the hands of a clock at 7 o'clock is 150 degrees. -/
theorem clock_angle_at_7 : ℝ :=
  let total_hours : ℕ := 12
  let total_degrees : ℝ := 360
  let hours_at_7 : ℕ := 7
  let angle_per_hour : ℝ := total_degrees / total_hours
  let hour_hand_angle : ℝ := angle_per_hour * hours_at_7
  let smaller_angle : ℝ := total_degrees - hour_hand_angle
  smaller_angle

/-- The theorem states that the smaller angle formed by the hands of a clock at 7 o'clock is 150 degrees. -/
theorem clock_angle_at_7_is_150 : clock_angle_at_7 = 150 := by
  sorry

end NUMINAMATH_CALUDE_clock_angle_at_7_clock_angle_at_7_is_150_l2688_268841


namespace NUMINAMATH_CALUDE_smallest_base_for_200_proof_l2688_268825

/-- The smallest base in which 200 (base 10) has exactly 6 digits -/
def smallest_base_for_200 : ℕ := 2

theorem smallest_base_for_200_proof :
  smallest_base_for_200 = 2 ∧
  2^7 ≤ 200 ∧
  200 < 2^8 ∧
  ∀ b : ℕ, 1 < b → b < 2 →
    (b^5 > 200 ∨ b^6 ≤ 200) :=
by sorry

end NUMINAMATH_CALUDE_smallest_base_for_200_proof_l2688_268825


namespace NUMINAMATH_CALUDE_cuboids_painted_count_l2688_268839

/-- The number of outer faces on a single cuboid -/
def faces_per_cuboid : ℕ := 6

/-- The total number of painted faces -/
def total_painted_faces : ℕ := 60

/-- The number of cuboids painted -/
def num_cuboids : ℕ := total_painted_faces / faces_per_cuboid

theorem cuboids_painted_count : num_cuboids = 10 := by
  sorry

end NUMINAMATH_CALUDE_cuboids_painted_count_l2688_268839


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l2688_268836

theorem quadratic_roots_sum (m n : ℝ) : 
  (m^2 + 2*m - 7 = 0) → (n^2 + 2*n - 7 = 0) → m^2 + 3*m + n = 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l2688_268836


namespace NUMINAMATH_CALUDE_not_p_and_q_l2688_268806

-- Define proposition p
def p : Prop := ∀ a b : ℝ, a > b → a > b^2

-- Define proposition q
def q : Prop := (∀ x : ℝ, x^2 + 2*x - 3 ≤ 0 → x ≤ 1) ∧ 
                (∃ x : ℝ, x ≤ 1 ∧ x^2 + 2*x - 3 > 0)

-- Theorem to prove
theorem not_p_and_q : ¬p ∧ q := by
  sorry

end NUMINAMATH_CALUDE_not_p_and_q_l2688_268806


namespace NUMINAMATH_CALUDE_cubic_function_properties_l2688_268856

/-- A cubic function with a local maximum at x = -1 and a local minimum at x = 3 -/
def f (a b c : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

/-- The derivative of f -/
def f_deriv (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem cubic_function_properties :
  ∃ (a b c : ℝ),
    (f_deriv a b (-1) = 0) ∧
    (f_deriv a b 3 = 0) ∧
    (f a b c (-1) = 7) ∧
    (a = -3) ∧
    (b = -9) ∧
    (c = 2) ∧
    (f a b c 3 = -25) := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_properties_l2688_268856


namespace NUMINAMATH_CALUDE_max_distance_covered_l2688_268844

/-- The maximum distance a person can cover in 6 hours, 
    given that they travel at 5 km/hr for half the distance 
    and 4 km/hr for the other half. -/
theorem max_distance_covered (total_time : ℝ) (speed1 : ℝ) (speed2 : ℝ) : 
  total_time = 6 →
  speed1 = 5 →
  speed2 = 4 →
  (total_time * speed1 * speed2) / (speed1 + speed2) = 120 / 9 := by
  sorry

end NUMINAMATH_CALUDE_max_distance_covered_l2688_268844


namespace NUMINAMATH_CALUDE_expression_unbounded_l2688_268868

theorem expression_unbounded (M : ℝ) (hM : M > 0) :
  ∃ x y z : ℝ, -1 < x ∧ x < 1 ∧ -1 < y ∧ y < 1 ∧ -1 < z ∧ z < 1 ∧
    (1 / ((1 - x^2) * (1 - y^2) * (1 - z^2)) +
     1 / ((1 + x^2) * (1 + y^2) * (1 + z^2))) > M :=
by sorry

end NUMINAMATH_CALUDE_expression_unbounded_l2688_268868


namespace NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l2688_268812

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_positive : ∀ n, a n > 0)
  (h_geom : geometric_sequence a)
  (h_eq : a 4 = (a 2)^2)
  (h_sum : a 2 + a 4 = 5/16) :
  a 5 = 1/32 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l2688_268812


namespace NUMINAMATH_CALUDE_triangle_inequality_sum_l2688_268886

theorem triangle_inequality_sum (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) :
  c / (a + b) + a / (b + c) + b / (c + a) > 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_sum_l2688_268886


namespace NUMINAMATH_CALUDE_ratio_x_to_y_l2688_268821

theorem ratio_x_to_y (x y : ℝ) (h : 0.1 * x = 0.2 * y) : x / y = 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_x_to_y_l2688_268821


namespace NUMINAMATH_CALUDE_scooter_profit_theorem_l2688_268867

def scooter_profit_problem (cost_price : ℝ) : Prop :=
  let repair_cost : ℝ := 500
  let profit : ℝ := 1100
  let selling_price : ℝ := cost_price + profit
  (0.1 * cost_price = repair_cost) ∧
  ((profit / cost_price) * 100 = 22)

theorem scooter_profit_theorem :
  ∃ (cost_price : ℝ), scooter_profit_problem cost_price := by
  sorry

end NUMINAMATH_CALUDE_scooter_profit_theorem_l2688_268867


namespace NUMINAMATH_CALUDE_x_zero_value_l2688_268813

open Real

noncomputable def f (x : ℝ) : ℝ := x * (2016 + log x)

theorem x_zero_value (x₀ : ℝ) (h : x₀ > 0) :
  (deriv f x₀ = 2017) → x₀ = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_zero_value_l2688_268813


namespace NUMINAMATH_CALUDE_hilt_pencil_cost_l2688_268847

/-- The cost of a pencil given total money and number of pencils that can be bought --/
def pencil_cost (total_money : ℚ) (num_pencils : ℕ) : ℚ :=
  total_money / num_pencils

theorem hilt_pencil_cost :
  pencil_cost 50 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_hilt_pencil_cost_l2688_268847


namespace NUMINAMATH_CALUDE_binary_division_and_double_l2688_268889

def binary_number : ℕ := 3666 -- 111011010010₂ in decimal

theorem binary_division_and_double :
  (binary_number % 4) * 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_binary_division_and_double_l2688_268889


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2688_268873

theorem expression_simplification_and_evaluation (x : ℝ) 
  (h1 : x ≠ -2) (h2 : x ≠ 0) (h3 : x ≠ 2) :
  (x^2 / (x - 2) + 4 / (2 - x)) / ((x^2 + 4*x + 4) / x) = x / (x + 2) ∧
  (1 : ℝ) / (1 + 2) = (1 : ℝ) / 3 := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2688_268873


namespace NUMINAMATH_CALUDE_apples_given_to_neighbor_l2688_268804

def initial_apples : ℕ := 127
def remaining_apples : ℕ := 39

theorem apples_given_to_neighbor :
  initial_apples - remaining_apples = 88 :=
by sorry

end NUMINAMATH_CALUDE_apples_given_to_neighbor_l2688_268804


namespace NUMINAMATH_CALUDE_semicircle_segment_sum_l2688_268869

-- Define the semicircle and its properties
structure Semicircle where
  r : ℝ
  a : ℝ
  A : ℝ × ℝ
  B : ℝ × ℝ
  M : ℝ × ℝ
  N : ℝ × ℝ
  h_diameter : dist A B = 2 * r
  h_AT : a > 0 ∧ 2 * a < r / 2
  h_M_on_semicircle : dist M A * dist M B = r ^ 2
  h_N_on_semicircle : dist N A * dist N B = r ^ 2
  h_M_condition : dist M (0, -2 * a) / dist M A = 1
  h_N_condition : dist N (0, -2 * a) / dist N A = 1
  h_M_N_distinct : M ≠ N

-- State the theorem
theorem semicircle_segment_sum (s : Semicircle) : dist s.A s.M + dist s.A s.N = dist s.A s.B := by
  sorry

end NUMINAMATH_CALUDE_semicircle_segment_sum_l2688_268869


namespace NUMINAMATH_CALUDE_negation_equivalence_l2688_268830

-- Define the original proposition
def original_proposition : Prop := ∃ x : ℝ, Real.exp x - x - 2 ≤ 0

-- Define the negation of the proposition
def negation_proposition : Prop := ∀ x : ℝ, Real.exp x - x - 2 > 0

-- Theorem stating the equivalence between the negation of the original proposition
-- and the negation_proposition
theorem negation_equivalence : 
  (¬ original_proposition) ↔ negation_proposition :=
sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2688_268830


namespace NUMINAMATH_CALUDE_bowling_team_weight_l2688_268865

theorem bowling_team_weight (initial_players : ℕ) (initial_avg : ℝ) 
  (new_player1_weight : ℝ) (new_avg : ℝ) :
  initial_players = 7 →
  initial_avg = 103 →
  new_player1_weight = 110 →
  new_avg = 99 →
  ∃ (new_player2_weight : ℝ),
    (initial_players * initial_avg + new_player1_weight + new_player2_weight) / 
    (initial_players + 2) = new_avg ∧
    new_player2_weight = 60 := by
  sorry

end NUMINAMATH_CALUDE_bowling_team_weight_l2688_268865


namespace NUMINAMATH_CALUDE_inscribed_cube_side_length_l2688_268848

/-- A cone with a circular base of radius 1 and height 3 --/
structure Cone :=
  (base_radius : ℝ := 1)
  (height : ℝ := 3)

/-- A cube inscribed in a cone such that four vertices lie on the base and four on the sloping sides --/
structure InscribedCube :=
  (cone : Cone)
  (side_length : ℝ)
  (four_vertices_on_base : Prop)
  (four_vertices_on_slope : Prop)

/-- The side length of the inscribed cube is 3√2 / (3 + √2) --/
theorem inscribed_cube_side_length (cube : InscribedCube) :
  cube.side_length = 3 * Real.sqrt 2 / (3 + Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_inscribed_cube_side_length_l2688_268848


namespace NUMINAMATH_CALUDE_difference_of_squares_l2688_268853

theorem difference_of_squares (a b : ℝ) : (a + b) * (a - b) = a^2 - b^2 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l2688_268853


namespace NUMINAMATH_CALUDE_no_real_solutions_l2688_268896

theorem no_real_solutions :
  ¬∃ y : ℝ, (8 * y^2 + 47 * y + 5) / (4 * y + 15) = 4 * y + 2 :=
by sorry

end NUMINAMATH_CALUDE_no_real_solutions_l2688_268896


namespace NUMINAMATH_CALUDE_xyz_product_l2688_268859

theorem xyz_product (x y z : ℂ) 
  (eq1 : x * y + 3 * y = -9)
  (eq2 : y * z + 3 * z = -9)
  (eq3 : z * x + 3 * x = -9) :
  x * y * z = 27 := by sorry

end NUMINAMATH_CALUDE_xyz_product_l2688_268859


namespace NUMINAMATH_CALUDE_swimmer_speed_in_still_water_l2688_268892

/-- Represents the speed of a swimmer in still water and the speed of the stream. -/
structure SwimmerSpeed where
  swimmer : ℝ  -- Speed of the swimmer in still water
  stream : ℝ   -- Speed of the stream

/-- Calculates the effective speed of the swimmer. -/
def effectiveSpeed (s : SwimmerSpeed) (downstream : Bool) : ℝ :=
  if downstream then s.swimmer + s.stream else s.swimmer - s.stream

/-- Theorem: Given the conditions of the swimmer's journey, his speed in still water is 7 km/h. -/
theorem swimmer_speed_in_still_water 
  (s : SwimmerSpeed) 
  (h1 : effectiveSpeed s true * 5 = 45)  -- Downstream condition
  (h2 : effectiveSpeed s false * 5 = 25) -- Upstream condition
  : s.swimmer = 7 := by
  sorry


end NUMINAMATH_CALUDE_swimmer_speed_in_still_water_l2688_268892


namespace NUMINAMATH_CALUDE_P_sufficient_not_necessary_Q_l2688_268893

/-- Triangle inequality condition -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Proposition P: segments can form a triangle -/
def P (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

/-- Proposition Q: sum of squares inequality -/
def Q (a b c : ℝ) : Prop :=
  a^2 + b^2 + c^2 < 2*(a*b + b*c + c*a)

/-- P is sufficient but not necessary for Q -/
theorem P_sufficient_not_necessary_Q :
  (∀ a b c : ℝ, P a b c → Q a b c) ∧
  (∃ a b c : ℝ, Q a b c ∧ ¬P a b c) := by
  sorry

end NUMINAMATH_CALUDE_P_sufficient_not_necessary_Q_l2688_268893


namespace NUMINAMATH_CALUDE_almeriense_polynomial_characterization_l2688_268800

/-- A polynomial is almeriense if it has the form x³ + ax² + bx + a
    and its three roots are positive real numbers in arithmetic progression. -/
def IsAlmeriense (p : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ),
    (∀ x, p x = x^3 + a*x^2 + b*x + a) ∧
    (∃ (r₁ r₂ r₃ : ℝ), r₁ > 0 ∧ r₂ > 0 ∧ r₃ > 0 ∧
      r₂ - r₁ = r₃ - r₂ ∧
      (∀ x, p x = (x - r₁) * (x - r₂) * (x - r₃)))

theorem almeriense_polynomial_characterization :
  ∀ p : ℝ → ℝ,
    IsAlmeriense p →
    p (7/4) = 0 →
    ((∀ x, p x = x^3 - (21/4)*x^2 + (73/8)*x - 21/4) ∨
     (∀ x, p x = x^3 - (291/56)*x^2 + (14113/1568)*x - 291/56)) :=
by sorry

end NUMINAMATH_CALUDE_almeriense_polynomial_characterization_l2688_268800


namespace NUMINAMATH_CALUDE_no_real_solutions_l2688_268819

theorem no_real_solutions : ¬∃ (x y : ℝ), 9^(x^2 + y) + 9^(x + y^2) = (1:ℝ)/3 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l2688_268819


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2688_268816

theorem inequality_solution_set (x y : ℝ) : 
  (∀ y > 0, (4 * (x^2 * y^2 + 4 * x * y^2 + 4 * x^2 * y + 16 * y^2 + 12 * x^2 * y)) / (x + y) > 3 * x^2 * y) ↔ 
  x > 0 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2688_268816


namespace NUMINAMATH_CALUDE_virginia_eggs_problem_l2688_268852

theorem virginia_eggs_problem (initial_eggs : ℕ) (amy_takes : ℕ) (john_takes : ℕ) (laura_takes : ℕ) 
  (h1 : initial_eggs = 372)
  (h2 : amy_takes = 15)
  (h3 : john_takes = 27)
  (h4 : laura_takes = 63) :
  initial_eggs - amy_takes - john_takes - laura_takes = 267 := by
  sorry

end NUMINAMATH_CALUDE_virginia_eggs_problem_l2688_268852


namespace NUMINAMATH_CALUDE_alice_bob_games_l2688_268858

/-- The number of players in the league -/
def total_players : ℕ := 12

/-- The number of players in each game -/
def players_per_game : ℕ := 6

/-- The number of games two specific players play together -/
def games_together : ℕ := 210

/-- The total number of possible game combinations -/
def total_combinations : ℕ := Nat.choose total_players players_per_game

theorem alice_bob_games :
  games_together = Nat.choose (total_players - 2) (players_per_game - 2) :=
by sorry

#check alice_bob_games

end NUMINAMATH_CALUDE_alice_bob_games_l2688_268858


namespace NUMINAMATH_CALUDE_magic_square_y_value_l2688_268808

def MagicSquare (a b c d e f g h i : ℤ) : Prop :=
  a + b + c = d + e + f ∧
  a + b + c = g + h + i ∧
  a + b + c = a + d + g ∧
  a + b + c = b + e + h ∧
  a + b + c = c + f + i ∧
  a + b + c = a + e + i ∧
  a + b + c = c + e + g

theorem magic_square_y_value :
  ∀ (y a b c d e : ℤ),
  MagicSquare y 23 101 4 a b c d e →
  y = -38 := by sorry

end NUMINAMATH_CALUDE_magic_square_y_value_l2688_268808


namespace NUMINAMATH_CALUDE_cosine_equation_solvability_l2688_268899

theorem cosine_equation_solvability (m : ℝ) :
  (∀ x : ℝ, ∃ y : ℝ, 4 * Real.cos y - Real.cos y ^ 2 + m - 3 = 0) ↔ 0 ≤ m ∧ m ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_cosine_equation_solvability_l2688_268899


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2688_268878

def A : Set ℝ := {x | x^2 - x ≤ 0}
def B : Set ℝ := {x | 2*x - 1 > 0}

theorem intersection_of_A_and_B : A ∩ B = Set.Ioo (1/2) 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2688_268878


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2688_268810

theorem inequality_solution_set (x : ℝ) : 4 * x - 2 ≤ 3 * (x - 1) ↔ x ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2688_268810


namespace NUMINAMATH_CALUDE_min_vertical_distance_l2688_268828

/-- The minimum vertical distance between y = |x-1| and y = -x^2 - 5x - 6 is 4 -/
theorem min_vertical_distance : ∃ (d : ℝ), d = 4 ∧ 
  ∀ (x : ℝ), d ≤ |x - 1| - (-x^2 - 5*x - 6) :=
by sorry

end NUMINAMATH_CALUDE_min_vertical_distance_l2688_268828


namespace NUMINAMATH_CALUDE_leah_savings_days_l2688_268890

/-- Proves that Leah saved for 20 days given the conditions of the problem -/
theorem leah_savings_days : ℕ :=
  let josiah_daily_savings : ℚ := 25 / 100
  let josiah_days : ℕ := 24
  let leah_daily_savings : ℚ := 1 / 2
  let megan_days : ℕ := 12
  let total_savings : ℚ := 28
  let leah_days : ℕ := 20

  have josiah_total : ℚ := josiah_daily_savings * josiah_days
  have megan_total : ℚ := 2 * leah_daily_savings * megan_days
  have leah_total : ℚ := leah_daily_savings * leah_days

  have savings_equation : josiah_total + leah_total + megan_total = total_savings := by sorry

  leah_days


end NUMINAMATH_CALUDE_leah_savings_days_l2688_268890


namespace NUMINAMATH_CALUDE_equation_system_solution_l2688_268822

/-- Represents a solution to the equation system -/
structure Solution :=
  (x : ℚ)
  (y : ℚ)

/-- Represents the equation system 2ax + y = 5 and 2x - by = 13 -/
def EquationSystem (a b : ℚ) (sol : Solution) : Prop :=
  2 * a * sol.x + sol.y = 5 ∧ 2 * sol.x - b * sol.y = 13

/-- Theorem stating the conditions and the correct solution -/
theorem equation_system_solution :
  let personA : Solution := ⟨7/2, -2⟩
  let personB : Solution := ⟨3, -7⟩
  let correctSol : Solution := ⟨2, -3⟩
  ∀ a b : ℚ,
    (EquationSystem 1 b personA) →  -- Person A misread a as 1
    (EquationSystem a 1 personB) →  -- Person B misread b as 1
    (a = 2 ∧ b = 3) ∧               -- Correct values of a and b
    (EquationSystem a b correctSol) -- Correct solution
  := by sorry

end NUMINAMATH_CALUDE_equation_system_solution_l2688_268822


namespace NUMINAMATH_CALUDE_aaron_position_2015_l2688_268835

/-- Represents a point on a 2D plane -/
structure Point where
  x : Int
  y : Int

/-- Represents a direction -/
inductive Direction
  | North
  | East
  | South
  | West

/-- Defines Aaron's walking pattern -/
def walk (n : Nat) : Point :=
  sorry

/-- The theorem to be proved -/
theorem aaron_position_2015 : walk 2015 = Point.mk 22 13 := by
  sorry

end NUMINAMATH_CALUDE_aaron_position_2015_l2688_268835


namespace NUMINAMATH_CALUDE_highway_mileage_calculation_l2688_268837

/-- Calculates the highway mileage of a car given total distance, city distance, city mileage, and total gas used. -/
theorem highway_mileage_calculation 
  (total_highway_distance : ℝ) 
  (total_city_distance : ℝ) 
  (city_mileage : ℝ) 
  (total_gas_used : ℝ) 
  (h1 : total_highway_distance = 210)
  (h2 : total_city_distance = 54)
  (h3 : city_mileage = 18)
  (h4 : total_gas_used = 9) :
  (total_highway_distance / (total_gas_used - total_city_distance / city_mileage)) = 35 := by
sorry

end NUMINAMATH_CALUDE_highway_mileage_calculation_l2688_268837


namespace NUMINAMATH_CALUDE_xiaoxiao_reading_plan_l2688_268863

/-- Given a book with a total number of pages, pages already read, and days to finish,
    calculate the average number of pages to read per day. -/
def averagePagesPerDay (totalPages pagesRead daysToFinish : ℕ) : ℚ :=
  (totalPages - pagesRead : ℚ) / daysToFinish

/-- Theorem stating that for a book with 160 pages, 60 pages read, and 5 days to finish,
    the average number of pages to read per day is 20. -/
theorem xiaoxiao_reading_plan :
  averagePagesPerDay 160 60 5 = 20 := by
  sorry

end NUMINAMATH_CALUDE_xiaoxiao_reading_plan_l2688_268863
