import Mathlib

namespace tangent_line_equation_minimum_value_l1643_164322

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a / x + Real.log x - 1

-- Define the domain of f
def domain (x : ℝ) : Prop := x > 0

-- State the theorem for the tangent line equation
theorem tangent_line_equation (a : ℝ) (h : a = 1) :
  ∃ (m b : ℝ), ∀ x y, domain x → (y = f a x) →
  (x = 2 → y = f a 2 → x - 4*y + 4*Real.log 2 - 4 = 0) :=
sorry

-- Define the interval (0, e]
def interval (x : ℝ) : Prop := 0 < x ∧ x ≤ Real.exp 1

-- State the theorem for the minimum value
theorem minimum_value (a : ℝ) :
  (a ≤ 0 → ¬∃ m, ∀ x, interval x → f a x ≥ m) ∧
  (0 < a → a < Real.exp 1 → ∃ m, m = Real.log a ∧ ∀ x, interval x → f a x ≥ m) ∧
  (a ≥ Real.exp 1 → ∃ m, m = a / Real.exp 1 ∧ ∀ x, interval x → f a x ≥ m) :=
sorry

end tangent_line_equation_minimum_value_l1643_164322


namespace problem_1_problem_2_problem_3_problem_4_l1643_164323

-- Problem 1
theorem problem_1 : Real.sqrt 27 - (1/3) * Real.sqrt 18 - Real.sqrt 12 = Real.sqrt 3 - Real.sqrt 2 := by sorry

-- Problem 2
theorem problem_2 : Real.sqrt 48 + Real.sqrt 30 - Real.sqrt (1/2) * Real.sqrt 12 + Real.sqrt 24 = 4 * Real.sqrt 3 + Real.sqrt 30 + Real.sqrt 6 := by sorry

-- Problem 3
theorem problem_3 : (2 - Real.sqrt 5) * (2 + Real.sqrt 5) - (2 - Real.sqrt 2)^2 = 4 * Real.sqrt 2 - 7 := by sorry

-- Problem 4
theorem problem_4 : (27 : Real)^(1/3) - (Real.sqrt 2 * Real.sqrt 6) / Real.sqrt 3 = 1 := by sorry

end problem_1_problem_2_problem_3_problem_4_l1643_164323


namespace division_problem_l1643_164386

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) :
  dividend = 122 →
  divisor = 20 →
  remainder = 2 →
  dividend = divisor * quotient + remainder →
  quotient = 6 := by
sorry

end division_problem_l1643_164386


namespace special_circle_equation_l1643_164358

/-- A circle passing through two points with a specific sum of intercepts -/
structure SpecialCircle where
  -- The circle passes through (4,2) and (-2,-6)
  passes_through_1 : x^2 + y^2 + D*x + E*y + F = 0 → 4^2 + 2^2 + 4*D + 2*E + F = 0
  passes_through_2 : x^2 + y^2 + D*x + E*y + F = 0 → (-2)^2 + (-6)^2 + (-2)*D + (-6)*E + F = 0
  -- Sum of intercepts is -2
  sum_of_intercepts : D + E = 2

/-- The standard equation of the special circle -/
def standard_equation (c : SpecialCircle) : Prop :=
  ∃ (x y : ℝ), (x - 1)^2 + (y + 2)^2 = 25

/-- Theorem stating that the given circle has the specified standard equation -/
theorem special_circle_equation (c : SpecialCircle) : standard_equation c :=
  sorry

end special_circle_equation_l1643_164358


namespace raisin_cost_fraction_l1643_164350

/-- Given a mixture of raisins and nuts, where the cost of nuts is twice that of raisins,
    prove that the cost of raisins is 3/11 of the total cost of the mixture. -/
theorem raisin_cost_fraction (raisin_cost : ℚ) : 
  let raisin_pounds : ℚ := 3
  let nut_pounds : ℚ := 4
  let nut_cost : ℚ := 2 * raisin_cost
  let total_raisin_cost : ℚ := raisin_pounds * raisin_cost
  let total_nut_cost : ℚ := nut_pounds * nut_cost
  let total_cost : ℚ := total_raisin_cost + total_nut_cost
  total_raisin_cost / total_cost = 3 / 11 := by
sorry

end raisin_cost_fraction_l1643_164350


namespace f_zero_implies_a_bound_l1643_164336

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2 * Real.exp 1 * x - (Real.log x) / x + a

theorem f_zero_implies_a_bound (a : ℝ) :
  (∃ x > 0, f x a = 0) →
  a ≤ Real.exp 2 + 1 / Real.exp 1 :=
by sorry

end f_zero_implies_a_bound_l1643_164336


namespace divisible_by_77_l1643_164387

theorem divisible_by_77 (n : ℕ) (h : ∀ k : ℕ, 2 ≤ k → k ≤ 76 → k ∣ n) : 77 ∣ n := by
  sorry

end divisible_by_77_l1643_164387


namespace greatest_common_factor_40_120_100_l1643_164356

theorem greatest_common_factor_40_120_100 : Nat.gcd 40 (Nat.gcd 120 100) = 20 := by
  sorry

end greatest_common_factor_40_120_100_l1643_164356


namespace euler_family_mean_age_l1643_164319

/-- The Euler family children's ages after one year -/
def euler_family_ages : List ℕ := [9, 9, 9, 9, 11, 13, 13]

/-- The number of children in the Euler family -/
def num_children : ℕ := 7

/-- The sum of the Euler family children's ages after one year -/
def sum_ages : ℕ := euler_family_ages.sum

theorem euler_family_mean_age :
  (sum_ages : ℚ) / num_children = 73 / 7 := by sorry

end euler_family_mean_age_l1643_164319


namespace quadratic_inequality_l1643_164327

/-- A quadratic function with specific properties -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_inequality (a b c : ℝ) (h_a : a > 0)
  (h_solution : ∀ x, QuadraticFunction a b c x > 0 ↔ x < -2 ∨ x > 4) :
  QuadraticFunction a b c 2 < QuadraticFunction a b c (-1) ∧
  QuadraticFunction a b c (-1) < QuadraticFunction a b c 5 := by
  sorry

end quadratic_inequality_l1643_164327


namespace remainder_a52_mod_52_l1643_164330

def concatenate_integers (n : ℕ) : ℕ :=
  -- Definition of a_n as the concatenation of integers from 1 to n
  sorry

theorem remainder_a52_mod_52 : concatenate_integers 52 % 52 = 28 := by
  sorry

end remainder_a52_mod_52_l1643_164330


namespace largest_divisor_of_consecutive_even_numbers_l1643_164391

theorem largest_divisor_of_consecutive_even_numbers : ∃ (m : ℕ), 
  (∀ (n : ℕ), (2*n) * (2*n + 2) * (2*n + 4) % m = 0) ∧ 
  (∀ (k : ℕ), k > m → ∃ (n : ℕ), (2*n) * (2*n + 2) * (2*n + 4) % k ≠ 0) :=
by
  -- The proof goes here
  sorry

end largest_divisor_of_consecutive_even_numbers_l1643_164391


namespace diophantine_equations_solutions_l1643_164364

-- Define the set of solutions for the first equation
def S₁ : Set (ℤ × ℤ) := {(x, y) | ∃ k : ℤ, x = 3 * k + 1 ∧ y = -2 * k + 1}

-- Define the set of solutions for the second equation
def S₂ : Set (ℤ × ℤ) := {(x, y) | ∃ k : ℤ, x = 5 * k ∧ y = 2 - 2 * k}

theorem diophantine_equations_solutions :
  (∀ (x y : ℤ), (2 * x + 3 * y = 5) ↔ (x, y) ∈ S₁) ∧
  (∀ (x y : ℤ), (2 * x + 5 * y = 10) ↔ (x, y) ∈ S₂) ∧
  (¬ ∃ (x y : ℤ), 3 * x + 9 * y = 2018) := by
  sorry

#check diophantine_equations_solutions

end diophantine_equations_solutions_l1643_164364


namespace tan_equation_solution_set_l1643_164347

theorem tan_equation_solution_set :
  {x : Real | Real.tan x = 2} = {x : Real | ∃ k : ℤ, x = k * Real.pi + Real.arctan 2} := by sorry

end tan_equation_solution_set_l1643_164347


namespace sum_of_three_greater_than_five_l1643_164368

theorem sum_of_three_greater_than_five (a b c : ℕ) :
  a ∈ Finset.range 10 →
  b ∈ Finset.range 10 →
  c ∈ Finset.range 10 →
  a ≠ b →
  a ≠ c →
  b ≠ c →
  a + b + c > 5 := by
  sorry

end sum_of_three_greater_than_five_l1643_164368


namespace sufficient_not_necessary_l1643_164338

theorem sufficient_not_necessary (x y : ℝ) :
  (∀ x y : ℝ, x > 1 ∧ y > 1 → x + y > 2) ∧
  (∃ x y : ℝ, x + y > 2 ∧ ¬(x > 1 ∧ y > 1)) :=
by sorry

end sufficient_not_necessary_l1643_164338


namespace coin_flip_probability_l1643_164397

/-- The probability of getting exactly k successes in n independent trials -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

/-- The number of coin flips -/
def n : ℕ := 3

/-- The number of times we want the coin to land tails up -/
def k : ℕ := 2

/-- The probability of the coin landing tails up on a single flip -/
def p : ℝ := 0.5

theorem coin_flip_probability : binomial_probability n k p = 0.375 := by
  sorry

end coin_flip_probability_l1643_164397


namespace rectangular_hall_dimension_difference_l1643_164301

/-- Proves that for a rectangular hall with width half the length and area 128 sq. m,
    the difference between length and width is 8 meters. -/
theorem rectangular_hall_dimension_difference :
  ∀ (length width : ℝ),
    width = length / 2 →
    length * width = 128 →
    length - width = 8 :=
by
  sorry

end rectangular_hall_dimension_difference_l1643_164301


namespace original_prices_calculation_l1643_164325

/-- Given the price increases and final prices of three items, prove their original prices. -/
theorem original_prices_calculation (computer_increase : ℝ) (tv_increase : ℝ) (fridge_increase : ℝ)
  (computer_final : ℝ) (tv_final : ℝ) (fridge_final : ℝ)
  (h1 : computer_increase = 0.30)
  (h2 : tv_increase = 0.20)
  (h3 : fridge_increase = 0.15)
  (h4 : computer_final = 377)
  (h5 : tv_final = 720)
  (h6 : fridge_final = 1150) :
  ∃ (computer_original tv_original fridge_original : ℝ),
    computer_original = 290 ∧
    tv_original = 600 ∧
    fridge_original = 1000 ∧
    computer_final = computer_original * (1 + computer_increase) ∧
    tv_final = tv_original * (1 + tv_increase) ∧
    fridge_final = fridge_original * (1 + fridge_increase) :=
by sorry

end original_prices_calculation_l1643_164325


namespace shirt_cost_l1643_164353

theorem shirt_cost (total_cost shirt_cost coat_cost : ℝ) : 
  total_cost = 600 →
  shirt_cost = (1/3) * coat_cost →
  shirt_cost + coat_cost = total_cost →
  shirt_cost = 150 := by
sorry

end shirt_cost_l1643_164353


namespace apple_cost_graph_properties_l1643_164337

def apple_cost (n : ℕ) : ℚ :=
  if n ≤ 10 then 20 * n else 18 * n

theorem apple_cost_graph_properties :
  ∃ (f : ℕ → ℚ),
    (∀ n : ℕ, 1 ≤ n ∧ n ≤ 20 → f n = apple_cost n) ∧
    (∀ n : ℕ, 1 ≤ n ∧ n < 10 → f (n + 1) - f n = 20) ∧
    (∀ n : ℕ, 10 < n ∧ n < 20 → f (n + 1) - f n = 18) ∧
    (f 11 - f 10 ≠ 20 ∧ f 11 - f 10 ≠ 18) :=
by sorry

end apple_cost_graph_properties_l1643_164337


namespace contrapositive_equivalence_l1643_164309

theorem contrapositive_equivalence (x : ℝ) :
  (x^2 < 1 → -1 < x ∧ x < 1) ↔ (x ≤ -1 ∨ x ≥ 1 → x^2 ≥ 1) :=
by sorry

end contrapositive_equivalence_l1643_164309


namespace tamara_height_l1643_164362

/-- Given that Tamara's height is 3 times Kim's height minus 4 inches,
    and their combined height is 92 inches, prove that Tamara is 68 inches tall. -/
theorem tamara_height (kim : ℝ) (tamara : ℝ) : 
  tamara = 3 * kim - 4 → 
  tamara + kim = 92 → 
  tamara = 68 := by
sorry

end tamara_height_l1643_164362


namespace triangle_area_from_perimeter_and_inradius_l1643_164354

/-- Given a triangle with perimeter 48 and inradius 2.5, prove its area is 60 -/
theorem triangle_area_from_perimeter_and_inradius :
  ∀ (T : Set ℝ) (perimeter inradius area : ℝ),
  (perimeter = 48) →
  (inradius = 2.5) →
  (area = inradius * (perimeter / 2)) →
  area = 60 := by
sorry

end triangle_area_from_perimeter_and_inradius_l1643_164354


namespace complex_power_difference_l1643_164302

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_power_difference :
  (1 + 2*i)^8 - (1 - 2*i)^8 = 672*i :=
by
  -- The proof goes here
  sorry

end complex_power_difference_l1643_164302


namespace product_of_integers_with_given_lcm_and_gcd_l1643_164396

theorem product_of_integers_with_given_lcm_and_gcd :
  ∀ a b : ℕ+, 
  (Nat.lcm a b = 60) → 
  (Nat.gcd a b = 12) → 
  (a * b = 720) :=
by
  sorry

end product_of_integers_with_given_lcm_and_gcd_l1643_164396


namespace simplify_expression_l1643_164344

theorem simplify_expression (x y z : ℝ) (hx : x ≠ 0) (hz : z ≠ 0) (hxy : x^2 ≠ y^2) :
  (x^2 - y^2)⁻¹ * (x⁻¹ - z⁻¹) = (z - x) * x⁻¹ * z⁻¹ * (x^2 - y^2)⁻¹ :=
by sorry

end simplify_expression_l1643_164344


namespace min_value_sum_reciprocals_l1643_164375

theorem min_value_sum_reciprocals (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a^2 + 4*b^2 + 9*c^2 = 4*b + 12*c - 2) :
  (1/a + 2/b + 3/c) ≥ 6 :=
sorry

end min_value_sum_reciprocals_l1643_164375


namespace sum_of_coefficients_l1643_164329

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x, (2 - x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + a₂ + a₃ + a₄ + a₅ = -31 := by
sorry

end sum_of_coefficients_l1643_164329


namespace max_value_at_point_one_two_l1643_164383

/-- The feasible region defined by the given constraints -/
def FeasibleRegion (x y : ℝ) : Prop :=
  x + 2*y ≤ 5 ∧ 2*x + y ≤ 4 ∧ x ≥ 0 ∧ y ≥ 0

/-- The objective function to be maximized -/
def ObjectiveFunction (x y : ℝ) : ℝ := 3*x + 4*y

/-- Theorem stating that the maximum value of the objective function
    in the feasible region is 11, achieved at (1, 2) -/
theorem max_value_at_point_one_two :
  ∃ (max : ℝ), max = 11 ∧
  ∃ (x₀ y₀ : ℝ), x₀ = 1 ∧ y₀ = 2 ∧
  FeasibleRegion x₀ y₀ ∧
  ObjectiveFunction x₀ y₀ = max ∧
  ∀ (x y : ℝ), FeasibleRegion x y → ObjectiveFunction x y ≤ max :=
by
  sorry

end max_value_at_point_one_two_l1643_164383


namespace intersection_of_A_and_B_l1643_164326

def set_A : Set ℝ := {x | Real.cos x = 0}
def set_B : Set ℝ := {x | x^2 - 5*x ≤ 0}

theorem intersection_of_A_and_B :
  set_A ∩ set_B = {Real.pi/2, 3*Real.pi/2} := by sorry

end intersection_of_A_and_B_l1643_164326


namespace donuts_distribution_l1643_164390

/-- Calculates the number of donuts each student who likes donuts receives -/
def donuts_per_student (total_donuts : ℕ) (total_students : ℕ) (donut_liking_ratio : ℚ) : ℚ :=
  total_donuts / (total_students * donut_liking_ratio)

/-- Proves that given 4 dozen donuts distributed among 80% of 30 students, 
    each student who likes donuts receives 2 donuts -/
theorem donuts_distribution : 
  donuts_per_student (4 * 12) 30 (4/5) = 2 := by
  sorry

end donuts_distribution_l1643_164390


namespace inequality_proof_l1643_164339

theorem inequality_proof (a b c d : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) 
  (h_prod : a * b * c * d = 1) 
  (h_ineq : a + b + c + d > a/b + b/c + c/d + d/a) : 
  a + b + c + d < b/a + c/b + d/c + a/d := by
sorry

end inequality_proof_l1643_164339


namespace square_side_length_l1643_164385

/-- The area enclosed between the circumferences of four circles described about the corners of a square -/
def enclosed_area : ℝ := 42.06195997410015

/-- Theorem: Given four equal circles described about the four corners of a square, 
    each touching two others, with the area enclosed between the circumferences 
    of the circles being 42.06195997410015 cm², the length of a side of the square is 14 cm. -/
theorem square_side_length (r : ℝ) (h1 : r > 0) 
  (h2 : 4 * r^2 - Real.pi * r^2 = enclosed_area) : 
  2 * r = 14 := by
  sorry

end square_side_length_l1643_164385


namespace buy_three_items_count_l1643_164313

/-- Represents the inventory of a store selling computer peripherals -/
structure StoreInventory where
  headphones : Nat
  mice : Nat
  keyboards : Nat
  keyboard_mouse_sets : Nat
  headphone_mouse_sets : Nat

/-- Calculates the number of ways to buy a headphone, a keyboard, and a mouse -/
def ways_to_buy_three (inventory : StoreInventory) : Nat :=
  inventory.keyboard_mouse_sets * inventory.headphones +
  inventory.headphone_mouse_sets * inventory.keyboards +
  inventory.headphones * inventory.mice * inventory.keyboards

/-- The theorem stating that there are 646 ways to buy three items -/
theorem buy_three_items_count (inventory : StoreInventory) 
  (h1 : inventory.headphones = 9)
  (h2 : inventory.mice = 13)
  (h3 : inventory.keyboards = 5)
  (h4 : inventory.keyboard_mouse_sets = 4)
  (h5 : inventory.headphone_mouse_sets = 5) :
  ways_to_buy_three inventory = 646 := by
  sorry

#eval ways_to_buy_three { headphones := 9, mice := 13, keyboards := 5, keyboard_mouse_sets := 4, headphone_mouse_sets := 5 }

end buy_three_items_count_l1643_164313


namespace sara_is_45_inches_tall_l1643_164371

-- Define the heights as natural numbers
def roy_height : ℕ := 36
def joe_height : ℕ := roy_height + 3
def sara_height : ℕ := joe_height + 6

-- Theorem statement
theorem sara_is_45_inches_tall : sara_height = 45 := by
  sorry

end sara_is_45_inches_tall_l1643_164371


namespace arithmetic_arrangement_l1643_164399

theorem arithmetic_arrangement :
  (1 / 8 * 1 / 9 * 1 / 28 = 1 / 2016) ∧
  ((1 / 8 - 1 / 9) * 1 / 28 = 1 / 2016) := by sorry

end arithmetic_arrangement_l1643_164399


namespace number_equation_solution_l1643_164384

theorem number_equation_solution : 
  ∃ x : ℝ, (3 * x = 2 * x - 7) ∧ (x = -7) := by sorry

end number_equation_solution_l1643_164384


namespace simplify_algebraic_expression_l1643_164381

theorem simplify_algebraic_expression (a b : ℝ) : 5*a*b - 7*a*b + 3*a*b = a*b := by
  sorry

end simplify_algebraic_expression_l1643_164381


namespace sequence_properties_l1643_164321

def S (n : ℕ) : ℝ := 3 * n^2 - 2 * n

def a : ℕ → ℝ := λ n => 6 * n - 5

theorem sequence_properties :
  (∀ n, S n = 3 * n^2 - 2 * n) →
  (∀ n, a n = 6 * n - 5) ∧
  (a 1 = 1) ∧
  (∀ n, n ≥ 2 → a n - a (n-1) = 6) :=
sorry

end sequence_properties_l1643_164321


namespace strikers_count_l1643_164395

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

end strikers_count_l1643_164395


namespace total_protein_consumed_l1643_164307

-- Define the protein content for each food item
def collagen_protein_per_2_scoops : ℚ := 18
def protein_powder_per_scoop : ℚ := 21
def steak_protein : ℚ := 56
def greek_yogurt_protein : ℚ := 15
def almond_protein_per_quarter_cup : ℚ := 6

-- Define the consumption quantities
def collagen_scoops : ℚ := 1
def protein_powder_scoops : ℚ := 2
def steak_quantity : ℚ := 1
def greek_yogurt_servings : ℚ := 1
def almond_cups : ℚ := 1/2

-- Theorem statement
theorem total_protein_consumed :
  (collagen_scoops / 2 * collagen_protein_per_2_scoops) +
  (protein_powder_scoops * protein_powder_per_scoop) +
  (steak_quantity * steak_protein) +
  (greek_yogurt_servings * greek_yogurt_protein) +
  (almond_cups / (1/4) * almond_protein_per_quarter_cup) = 134 := by
  sorry

end total_protein_consumed_l1643_164307


namespace circle_diameter_from_area_l1643_164370

theorem circle_diameter_from_area (A : ℝ) (d : ℝ) :
  A = 225 * Real.pi → d = 30 → A = Real.pi * (d / 2)^2 :=
by sorry

end circle_diameter_from_area_l1643_164370


namespace no_natural_solution_l1643_164359

theorem no_natural_solution : ¬∃ (x y : ℕ), 2 * x + 3 * y = 6 := by sorry

end no_natural_solution_l1643_164359


namespace smallest_yellow_marbles_l1643_164340

theorem smallest_yellow_marbles (n : ℕ) (h1 : n % 3 = 0) 
  (h2 : n ≥ 30) : ∃ (blue red green yellow : ℕ),
  blue = n / 3 ∧ 
  red = n / 3 ∧ 
  green = 10 ∧ 
  yellow = n - (blue + red + green) ∧ 
  yellow ≥ 0 ∧ 
  ∀ m : ℕ, m < n → ¬(∃ (b r g y : ℕ), 
    b = m / 3 ∧ 
    r = m / 3 ∧ 
    g = 10 ∧ 
    y = m - (b + r + g) ∧ 
    y ≥ 0 ∧ 
    m % 3 = 0) :=
by
  sorry

end smallest_yellow_marbles_l1643_164340


namespace orange_profit_problem_l1643_164377

/-- The number of oranges needed to make a profit of 200 cents -/
def oranges_needed (buy_price : ℚ) (sell_price : ℚ) (profit_goal : ℚ) : ℕ :=
  (profit_goal / (sell_price - buy_price)).ceil.toNat

/-- The problem statement -/
theorem orange_profit_problem :
  let buy_price : ℚ := 14 / 4
  let sell_price : ℚ := 25 / 6
  let profit_goal : ℚ := 200
  oranges_needed buy_price sell_price profit_goal = 300 := by
sorry

end orange_profit_problem_l1643_164377


namespace committee_rearrangements_count_l1643_164303

/-- The number of distinguishable rearrangements of the letters in "COMMITTEE" with all vowels at the beginning of the sequence -/
def committee_rearrangements : ℕ := sorry

/-- The number of vowels in "COMMITTEE" -/
def num_vowels : ℕ := 4

/-- The number of consonants in "COMMITTEE" -/
def num_consonants : ℕ := 5

/-- The number of repeated vowels (E) in "COMMITTEE" -/
def num_repeated_vowels : ℕ := 2

/-- The number of repeated consonants (M and T) in "COMMITTEE" -/
def num_repeated_consonants : ℕ := 2

theorem committee_rearrangements_count :
  committee_rearrangements = (Nat.factorial num_vowels / Nat.factorial num_repeated_vowels) *
                             (Nat.factorial num_consonants / (Nat.factorial num_repeated_consonants * Nat.factorial num_repeated_consonants)) :=
by sorry

end committee_rearrangements_count_l1643_164303


namespace price_reduction_theorem_l1643_164360

def original_price_A : ℝ := 500
def original_price_B : ℝ := 600
def original_price_C : ℝ := 700

def first_discount_rate : ℝ := 0.15
def second_discount_rate : ℝ := 0.10
def sales_tax_rate : ℝ := 0.07
def flat_discount_B : ℝ := 200

def total_original_price : ℝ := original_price_A + original_price_B + original_price_C

noncomputable def final_price_A : ℝ := 
  (original_price_A * (1 - first_discount_rate) * (1 - second_discount_rate)) * (1 + sales_tax_rate)

noncomputable def final_price_B : ℝ := 
  (original_price_B * (1 - first_discount_rate) * (1 - second_discount_rate)) - flat_discount_B

noncomputable def final_price_C : ℝ := 
  (original_price_C * (1 - second_discount_rate)) * (1 + sales_tax_rate)

noncomputable def total_final_price : ℝ := final_price_A + final_price_B + final_price_C

noncomputable def percentage_reduction : ℝ := 
  (total_original_price - total_final_price) / total_original_price * 100

theorem price_reduction_theorem : 
  25.42 ≤ percentage_reduction ∧ percentage_reduction < 25.43 :=
sorry

end price_reduction_theorem_l1643_164360


namespace ammonium_nitrate_formation_l1643_164300

-- Define the chemical species
def Ammonia : Type := Unit
def NitricAcid : Type := Unit
def AmmoniumNitrate : Type := Unit

-- Define the reaction
def reaction (nh3 : ℕ) (hno3 : ℕ) : ℕ :=
  min nh3 hno3

-- State the theorem
theorem ammonium_nitrate_formation 
  (nh3 : ℕ) -- Some moles of Ammonia
  (hno3 : ℕ) -- Moles of Nitric acid
  (h1 : hno3 = 3) -- 3 moles of Nitric acid are used
  (h2 : reaction nh3 hno3 = 3) -- Total moles of Ammonium nitrate formed are 3
  : reaction nh3 hno3 = 3 :=
by
  sorry

end ammonium_nitrate_formation_l1643_164300


namespace complex_parts_of_z_l1643_164320

def z : ℂ := 3 * Complex.I * (Complex.I + 1)

theorem complex_parts_of_z :
  Complex.re z = -3 ∧ Complex.im z = 3 := by sorry

end complex_parts_of_z_l1643_164320


namespace count_perfect_squares_l1643_164306

/-- The number of positive perfect square factors of (2^12)(3^15)(5^18)(7^8) -/
def num_perfect_square_factors : ℕ := 2800

/-- The product in question -/
def product : ℕ := 2^12 * 3^15 * 5^18 * 7^8

/-- A function that counts the number of positive perfect square factors of a natural number -/
def count_perfect_square_factors (n : ℕ) : ℕ := sorry

theorem count_perfect_squares :
  count_perfect_square_factors product = num_perfect_square_factors := by sorry

end count_perfect_squares_l1643_164306


namespace rationalize_denominator_l1643_164367

theorem rationalize_denominator :
  (1 : ℝ) / (Real.rpow 3 (1/3) + Real.rpow 27 (1/3)) = Real.rpow 9 (1/3) / 12 := by
  sorry

end rationalize_denominator_l1643_164367


namespace fraction_evaluation_l1643_164310

theorem fraction_evaluation : (2 + 3 * 6) / (23 + 6) = 20 / 29 := by
  sorry

end fraction_evaluation_l1643_164310


namespace bobby_candy_l1643_164343

def candy_problem (initial : ℕ) (final : ℕ) (second_round : ℕ) : Prop :=
  ∃ (first_round : ℕ), 
    initial - (first_round + second_round) = final ∧
    first_round + second_round < initial

theorem bobby_candy : candy_problem 21 7 9 → ∃ (x : ℕ), x = 5 := by
  sorry

end bobby_candy_l1643_164343


namespace horner_method_v3_l1643_164318

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 2x^5 + 3x^3 - 2x^2 + x - 1 -/
def f (x : ℝ) : ℝ := 2*x^5 + 3*x^3 - 2*x^2 + x - 1

/-- Coefficients of f(x) in descending order of degree -/
def f_coeffs : List ℝ := [2, 0, 3, -2, 1, -1]

theorem horner_method_v3 :
  let v₃ := horner (f_coeffs.take 4) 2
  v₃ = 20 := by sorry

end horner_method_v3_l1643_164318


namespace perfect_square_trinomial_condition_l1643_164363

/-- A trinomial ax^2 + bx + c is a perfect square if there exist real numbers p and q
    such that ax^2 + bx + c = (px + q)^2 for all x. -/
def is_perfect_square_trinomial (a b c : ℝ) : Prop :=
  ∃ p q : ℝ, ∀ x, a * x^2 + b * x + c = (p * x + q)^2

/-- If 4x^2 + 2kx + 25 is a perfect square trinomial, then k = ±10. -/
theorem perfect_square_trinomial_condition (k : ℝ) :
  is_perfect_square_trinomial 4 (2*k) 25 → k = 10 ∨ k = -10 := by
  sorry


end perfect_square_trinomial_condition_l1643_164363


namespace suits_sold_is_two_l1643_164311

/-- The number of suits sold given the commission rate, shirt sales, loafer sales, and total commission earned. -/
def suits_sold (commission_rate : ℚ) (num_shirts : ℕ) (shirt_price : ℚ) (num_loafers : ℕ) (loafer_price : ℚ) (suit_price : ℚ) (total_commission : ℚ) : ℕ :=
  sorry

/-- Theorem stating that the number of suits sold is 2 under the given conditions. -/
theorem suits_sold_is_two :
  suits_sold (15 / 100) 6 50 2 150 700 300 = 2 := by
  sorry

end suits_sold_is_two_l1643_164311


namespace inequality_implication_l1643_164398

theorem inequality_implication (p q r : ℝ) 
  (hr : r > 0) (hpq : p * q ≠ 0) (hineq : p * r < q * r) : 
  1 < q / p :=
sorry

end inequality_implication_l1643_164398


namespace line_parameterization_l1643_164352

/-- Given a line y = 2x - 10 parameterized by (x,y) = (g(t), 10t - 4), prove that g(t) = 5t + 3 -/
theorem line_parameterization (g : ℝ → ℝ) : 
  (∀ t, 2 * g t - 10 = 10 * t - 4) → 
  (∀ t, g t = 5 * t + 3) := by
sorry

end line_parameterization_l1643_164352


namespace parabola_property_l1643_164342

/-- A parabola with equation y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The y-coordinate of a point on the parabola given its x-coordinate -/
def Parabola.y_at (p : Parabola) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

theorem parabola_property (p : Parabola) :
  p.y_at (-3) = 4 →  -- vertex at (-3, 4)
  p.y_at (-2) = 7 →  -- passes through (-2, 7)
  3 * p.a + 2 * p.b + p.c = 76 := by
  sorry

end parabola_property_l1643_164342


namespace isosceles_triangle_relationship_l1643_164349

/-- An isosceles triangle with perimeter 10 cm -/
structure IsoscelesTriangle where
  /-- Length of each equal side in cm -/
  x : ℝ
  /-- Length of the base in cm -/
  y : ℝ
  /-- The triangle is isosceles -/
  isIsosceles : true
  /-- The perimeter is 10 cm -/
  perimeterIs10 : x + x + y = 10

/-- The relationship between y and x, and the range of x for the isosceles triangle -/
theorem isosceles_triangle_relationship (t : IsoscelesTriangle) :
  t.y = 10 - 2 * t.x ∧ 5/2 < t.x ∧ t.x < 5 := by
  sorry

end isosceles_triangle_relationship_l1643_164349


namespace eight_solutions_l1643_164328

/-- The function f(x) = x^2 - 2 -/
def f (x : ℝ) : ℝ := x^2 - 2

/-- The theorem stating that f(f(f(x))) = x has exactly eight distinct real solutions -/
theorem eight_solutions :
  ∃! (s : Finset ℝ), s.card = 8 ∧ (∀ x ∈ s, f (f (f x)) = x) ∧
    (∀ y : ℝ, f (f (f y)) = y → y ∈ s) :=
sorry

end eight_solutions_l1643_164328


namespace a_7_equals_neg_3_l1643_164331

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The theorem stating that a₇ = -3 in the given geometric sequence -/
theorem a_7_equals_neg_3 (a : ℕ → ℝ) :
  geometric_sequence a →
  a 5 ^ 2 + 2016 * a 5 + 9 = 0 →
  a 9 ^ 2 + 2016 * a 9 + 9 = 0 →
  a 7 = -3 := by
  sorry

end a_7_equals_neg_3_l1643_164331


namespace solution_set_characterization_l1643_164392

-- Define the properties of function f
def IsOddFunction (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def IsDecreasingFunction (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f x > f y

-- Define the solution set
def SolutionSet (f : ℝ → ℝ) : Set ℝ := {a | f (a^2) + f (2*a) > 0}

-- State the theorem
theorem solution_set_characterization 
  (f : ℝ → ℝ) 
  (h_odd : IsOddFunction f) 
  (h_decreasing : IsDecreasingFunction f) : 
  SolutionSet f = Set.Ioo (-2) 0 := by sorry

end solution_set_characterization_l1643_164392


namespace binomial_identity_solutions_l1643_164379

theorem binomial_identity_solutions (n : ℕ) :
  ∀ x y : ℝ, (x + y)^n = x^n + y^n ↔
    (n = 1 ∧ True) ∨
    (∃ k : ℕ, n = 2 * k ∧ (x = 0 ∨ y = 0)) ∨
    (∃ k : ℕ, n = 2 * k + 1 ∧ (x = 0 ∨ y = 0 ∨ x = -y)) :=
by sorry

end binomial_identity_solutions_l1643_164379


namespace amc10_paths_l1643_164394

/-- Represents the number of possible moves from each position -/
def num_moves : ℕ := 8

/-- Represents the length of the string "AMC10" -/
def word_length : ℕ := 5

/-- Calculates the number of paths to spell "AMC10" -/
def num_paths : ℕ := num_moves ^ (word_length - 1)

/-- Proves that the number of paths to spell "AMC10" is 4096 -/
theorem amc10_paths : num_paths = 4096 := by
  sorry

end amc10_paths_l1643_164394


namespace rectangles_on_4x4_grid_l1643_164380

/-- The number of lines in a 4x4 grid (both horizontal and vertical) -/
def gridLines : ℕ := 5

/-- The number of lines needed to form a rectangle (both horizontal and vertical) -/
def linesNeeded : ℕ := 2

/-- The number of ways to choose horizontal lines for a rectangle -/
def horizontalChoices : ℕ := Nat.choose gridLines linesNeeded

/-- The number of ways to choose vertical lines for a rectangle -/
def verticalChoices : ℕ := Nat.choose gridLines linesNeeded

/-- Theorem: The number of rectangles on a 4x4 grid is 100 -/
theorem rectangles_on_4x4_grid : horizontalChoices * verticalChoices = 100 := by
  sorry


end rectangles_on_4x4_grid_l1643_164380


namespace safe_count_theorem_l1643_164355

def is_p_safe (n p : ℕ) : Prop :=
  n % p > 2 ∧ n % p < p - 2

def count_safe (max : ℕ) : ℕ :=
  (max / (5 * 7 * 17)) * 48

theorem safe_count_theorem :
  count_safe 20000 = 1584 ∧
  ∀ n : ℕ, n ≤ 20000 →
    (is_p_safe n 5 ∧ is_p_safe n 7 ∧ is_p_safe n 17) ↔
    ∃ k : ℕ, k < 48 ∧ n ≡ k [MOD 595] :=
by sorry

end safe_count_theorem_l1643_164355


namespace cone_surface_area_minimization_l1643_164369

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

end cone_surface_area_minimization_l1643_164369


namespace equation_solutions_l1643_164351

theorem equation_solutions :
  (∀ X : ℝ, X - 12 = 81 → X = 93) ∧
  (∀ X : ℝ, 5.1 + X = 10.5 → X = 5.4) ∧
  (∀ X : ℝ, 6 * X = 4.2 → X = 0.7) ∧
  (∀ X : ℝ, X / 0.4 = 12.5 → X = 5) := by
  sorry

end equation_solutions_l1643_164351


namespace quadratic_equation_solution_difference_l1643_164361

theorem quadratic_equation_solution_difference : ∃ (x₁ x₂ : ℝ),
  (2 * x₁^2 - 6 * x₁ + 18 = 2 * x₁ + 82) ∧
  (2 * x₂^2 - 6 * x₂ + 18 = 2 * x₂ + 82) ∧
  x₁ ≠ x₂ ∧
  |x₁ - x₂| = 12 :=
by
  sorry

end quadratic_equation_solution_difference_l1643_164361


namespace x_squared_minus_y_squared_l1643_164315

theorem x_squared_minus_y_squared (x y : ℚ) 
  (h1 : x + y = 9/14) (h2 : x - y = 3/14) : x^2 - y^2 = 27/196 := by
  sorry

end x_squared_minus_y_squared_l1643_164315


namespace g_of_8_eq_neg_46_l1643_164393

/-- A function g : ℝ → ℝ satisfying the given functional equation for all real x and y -/
def g_equation (g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, g x + g (3*x + y) + 7*x*y = g (4*x - 2*y) + 3*x^2 + 2

/-- Theorem stating that if g satisfies the functional equation, then g(8) = -46 -/
theorem g_of_8_eq_neg_46 (g : ℝ → ℝ) (h : g_equation g) : g 8 = -46 := by
  sorry

end g_of_8_eq_neg_46_l1643_164393


namespace a_range_l1643_164334

-- Define the function f(x,a)
def f (x a : ℝ) : ℝ := a * x^3 - x^2 + 4*x + 3

-- State the theorem
theorem a_range :
  ∀ a : ℝ, (∀ x : ℝ, x ∈ Set.Icc (-2) 1 → f x a ≥ 0) → a ∈ Set.Icc (-6) (-2) :=
by sorry

end a_range_l1643_164334


namespace line_slope_l1643_164388

theorem line_slope (x y : ℝ) : 
  (3 * x - Real.sqrt 3 * y + 1 = 0) → 
  (∃ m : ℝ, y = m * x + (-1 / Real.sqrt 3) ∧ m = Real.sqrt 3) :=
sorry

end line_slope_l1643_164388


namespace tournament_teams_count_l1643_164382

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

end tournament_teams_count_l1643_164382


namespace train_speed_l1643_164341

/-- The speed of a train given its length and time to cross a pole -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 800) (h2 : time = 10) :
  length / time = 80 := by
  sorry

end train_speed_l1643_164341


namespace trig_identity_30_degrees_l1643_164332

theorem trig_identity_30_degrees :
  let tan30 : ℝ := 1 / Real.sqrt 3
  let sin30 : ℝ := 1 / 2
  (tan30^2 - sin30^2) / (tan30^2 * sin30^2) = 1 := by
sorry

end trig_identity_30_degrees_l1643_164332


namespace rope_ratio_proof_l1643_164373

theorem rope_ratio_proof (total_length shorter_length : ℕ) 
  (h1 : total_length = 40)
  (h2 : shorter_length = 16)
  (h3 : shorter_length < total_length) :
  (shorter_length : ℚ) / (total_length - shorter_length : ℚ) = 2 / 3 := by
sorry

end rope_ratio_proof_l1643_164373


namespace quadratic_root_zero_l1643_164346

theorem quadratic_root_zero (m : ℝ) : 
  (∃ x, (m - 1) * x^2 + 2 * x + m^2 - 1 = 0) ∧ 
  ((m - 1) * 0^2 + 2 * 0 + m^2 - 1 = 0) → 
  m = -1 := by
sorry

end quadratic_root_zero_l1643_164346


namespace number_equation_and_interval_l1643_164335

theorem number_equation_and_interval : ∃ (x : ℝ), 
  x = (1 / x) * x^2 + 3 ∧ x = 4 ∧ 3 < x ∧ x ≤ 6 := by
  sorry

end number_equation_and_interval_l1643_164335


namespace steve_bench_wood_length_l1643_164389

/-- Calculates the total length of wood needed for Steve's bench. -/
theorem steve_bench_wood_length : 
  let long_pieces : ℕ := 6
  let long_length : ℕ := 4
  let short_pieces : ℕ := 2
  let short_length : ℕ := 2
  long_pieces * long_length + short_pieces * short_length = 28 :=
by sorry

end steve_bench_wood_length_l1643_164389


namespace complex_fraction_evaluation_l1643_164376

theorem complex_fraction_evaluation :
  let i : ℂ := Complex.I
  (3 + i) / (1 + i) = 2 - i :=
by sorry

end complex_fraction_evaluation_l1643_164376


namespace smallest_three_digit_multiple_of_9_l1643_164333

theorem smallest_three_digit_multiple_of_9 :
  ∀ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ 9 ∣ n → n ≥ 108 :=
by sorry

end smallest_three_digit_multiple_of_9_l1643_164333


namespace prob_three_dice_l1643_164357

/-- The number of faces on a die -/
def num_faces : ℕ := 6

/-- The number of favorable outcomes on a single die (numbers greater than 2) -/
def favorable_outcomes : ℕ := 4

/-- The number of dice thrown simultaneously -/
def num_dice : ℕ := 3

/-- The probability of getting a number greater than 2 on a single die -/
def prob_single_die : ℚ := favorable_outcomes / num_faces

/-- The probability of getting a number greater than 2 on each of three dice -/
theorem prob_three_dice : (prob_single_die ^ num_dice : ℚ) = 8 / 27 := by
  sorry

end prob_three_dice_l1643_164357


namespace cosine_half_angle_in_interval_l1643_164305

theorem cosine_half_angle_in_interval (θ m : Real) 
  (h1 : 5/2 * Real.pi < θ) 
  (h2 : θ < 3 * Real.pi) 
  (h3 : |Real.cos θ| = m) : 
  Real.cos (θ/2) = -Real.sqrt ((1 - m)/2) := by
  sorry

end cosine_half_angle_in_interval_l1643_164305


namespace tan_theta_value_l1643_164366

/-- If the terminal side of angle θ passes through the point (-√3/2, 1/2), then tan θ = -√3/3 -/
theorem tan_theta_value (θ : Real) (h : ∃ (t : Real), t > 0 ∧ t * (-Real.sqrt 3 / 2) = Real.cos θ ∧ t * (1 / 2) = Real.sin θ) : 
  Real.tan θ = -Real.sqrt 3 / 3 := by
sorry

end tan_theta_value_l1643_164366


namespace largest_power_divisor_l1643_164314

theorem largest_power_divisor (m n : ℕ) (h1 : m = 1991^1992) (h2 : n = 1991^1990) :
  ∃ k : ℕ, k = 1991^1990 ∧ 
  k ∣ (1990*m + 1992*n) ∧ 
  ∀ l : ℕ, l > k → l = 1991^(1990 + (l.log 1991 - 1990)) → ¬(l ∣ (1990*m + 1992*n)) :=
by sorry

end largest_power_divisor_l1643_164314


namespace math_problem_solution_l1643_164365

theorem math_problem_solution :
  ∀ (S₁ S₂ S₃ S₁₂ S₁₃ S₂₃ S₁₂₃ : ℕ),
  S₁ + S₂ + S₃ + S₁₂ + S₁₃ + S₂₃ + S₁₂₃ = 100 →
  S₁ + S₁₂ + S₁₃ + S₁₂₃ = 60 →
  S₂ + S₁₂ + S₂₃ + S₁₂₃ = 60 →
  S₃ + S₁₃ + S₂₃ + S₁₂₃ = 60 →
  (S₁ + S₂ + S₃) - S₁₂₃ = 20 :=
by
  sorry

end math_problem_solution_l1643_164365


namespace propositions_correctness_l1643_164317

theorem propositions_correctness : 
  -- Proposition ②
  (∀ a b c : ℝ, a * c^2 > b * c^2 → a > b) ∧
  -- Proposition ③
  (∀ a b : ℝ, a > |b| → a > b) ∧
  -- Proposition ① (negation)
  (∃ a b : ℝ, a > b ∧ (1 / a ≥ 1 / b)) ∧
  -- Proposition ④ (negation)
  (∃ a b : ℝ, a > b ∧ a^2 ≤ b^2) :=
by sorry


end propositions_correctness_l1643_164317


namespace max_x2_plus_y2_l1643_164304

theorem max_x2_plus_y2 (x y : ℝ) (h1 : |x - y| ≤ 2) (h2 : |3*x + y| ≤ 6) : x^2 + y^2 ≤ 10 := by
  sorry

end max_x2_plus_y2_l1643_164304


namespace glass_bowls_problem_l1643_164378

/-- The number of glass bowls initially bought -/
def initial_bowls : ℕ := 2393

/-- The buying price per bowl in rupees -/
def buying_price : ℚ := 18

/-- The selling price per bowl in rupees -/
def selling_price : ℚ := 20

/-- The number of bowls sold -/
def bowls_sold : ℕ := 104

/-- The percentage gain -/
def percentage_gain : ℚ := 0.4830917874396135

theorem glass_bowls_problem :
  let total_cost : ℚ := initial_bowls * buying_price
  let revenue : ℚ := bowls_sold * selling_price
  let gain : ℚ := revenue - (bowls_sold * buying_price)
  percentage_gain = (gain / total_cost) * 100 :=
by sorry

end glass_bowls_problem_l1643_164378


namespace S_max_at_n_max_l1643_164324

/-- The sum of the first n terms of the sequence -/
def S (n : ℕ+) : ℤ := -n.val^2 + 8*n.val

/-- n_max is the value of n at which S_n attains its maximum value -/
def n_max : ℕ+ := 4

theorem S_max_at_n_max :
  ∀ n : ℕ+, S n ≤ S n_max :=
sorry

end S_max_at_n_max_l1643_164324


namespace max_x2_plus_y2_l1643_164312

theorem max_x2_plus_y2 (x y a : ℝ) (h1 : x + y = a + 1) (h2 : x * y = a^2 - 7*a + 16) :
  ∃ (max : ℝ), max = 32 ∧ ∀ (x' y' a' : ℝ), x' + y' = a' + 1 → x' * y' = a'^2 - 7*a' + 16 → x'^2 + y'^2 ≤ max :=
sorry

end max_x2_plus_y2_l1643_164312


namespace marys_number_l1643_164372

/-- Represents the scenario described in the problem -/
structure Scenario where
  j : Nat  -- John's number
  m : Nat  -- Mary's number
  sum : Nat := j + m
  product : Nat := j * m

/-- Predicate to check if a number has multiple factorizations -/
def hasMultipleFactorizations (n : Nat) : Prop :=
  ∃ a b c d : Nat, a * b = n ∧ c * d = n ∧ a ≠ c ∧ a ≠ d ∧ a > 1 ∧ b > 1 ∧ c > 1 ∧ d > 1

/-- The main theorem representing the problem -/
theorem marys_number (s : Scenario) : 
  s.product = 2002 ∧ 
  hasMultipleFactorizations 2002 ∧
  (∀ x : Nat, x * s.m = 2002 → hasMultipleFactorizations x) →
  s.m = 1001 := by
  sorry

#eval 1001 * 2  -- Should output 2002

end marys_number_l1643_164372


namespace least_positive_integer_for_multiple_of_five_l1643_164374

theorem least_positive_integer_for_multiple_of_five : 
  ∃ (n : ℕ), n > 0 ∧ (525 + n) % 5 = 0 ∧ ∀ (m : ℕ), m > 0 ∧ (525 + m) % 5 = 0 → n ≤ m :=
by sorry

end least_positive_integer_for_multiple_of_five_l1643_164374


namespace tea_containers_needed_l1643_164308

/-- The volume of tea in milliliters that each container can hold -/
def container_volume : ℕ := 500

/-- The minimum volume of tea in liters needed for the event -/
def required_volume : ℕ := 5

/-- Conversion factor from liters to milliliters -/
def liter_to_ml : ℕ := 1000

/-- The minimum number of containers needed to hold at least the required volume of tea -/
def min_containers : ℕ := 10

theorem tea_containers_needed :
  min_containers = 
    (required_volume * liter_to_ml + container_volume - 1) / container_volume :=
by sorry

end tea_containers_needed_l1643_164308


namespace gcd_420_882_l1643_164348

theorem gcd_420_882 : Nat.gcd 420 882 = 42 := by
  sorry

end gcd_420_882_l1643_164348


namespace triangle_inequality_sum_l1643_164316

theorem triangle_inequality_sum (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  a / (b + c) + b / (c + a) + c / (a + b) ≤ 2 := by
  sorry

end triangle_inequality_sum_l1643_164316


namespace probability_sum_six_l1643_164345

/-- The number of faces on each die -/
def numFaces : ℕ := 6

/-- The target sum we're looking for -/
def targetSum : ℕ := 6

/-- The set of possible outcomes when rolling two dice -/
def outcomes : Finset (ℕ × ℕ) :=
  Finset.product (Finset.range numFaces) (Finset.range numFaces)

/-- The set of favorable outcomes (sum equals targetSum) -/
def favorableOutcomes : Finset (ℕ × ℕ) :=
  outcomes.filter (fun p => p.1 + p.2 + 2 = targetSum)

/-- The probability of rolling a sum of 6 with two fair six-sided dice -/
theorem probability_sum_six :
  Nat.card favorableOutcomes / Nat.card outcomes = 5 / 36 := by
  sorry


end probability_sum_six_l1643_164345
