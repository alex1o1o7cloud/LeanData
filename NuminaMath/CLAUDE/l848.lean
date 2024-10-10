import Mathlib

namespace plant_prices_and_minimum_cost_l848_84882

/-- The price of a pot of green radish -/
def green_radish_price : ℝ := 4

/-- The price of a pot of spider plant -/
def spider_plant_price : ℝ := 12

/-- The total number of pots to be purchased -/
def total_pots : ℕ := 120

/-- The number of spider plant pots that minimizes the total cost -/
def optimal_spider_pots : ℕ := 80

/-- The number of green radish pots that minimizes the total cost -/
def optimal_green_radish_pots : ℕ := 40

theorem plant_prices_and_minimum_cost :
  (green_radish_price + spider_plant_price = 16) ∧
  (80 / green_radish_price = 2 * (120 / spider_plant_price)) ∧
  (optimal_spider_pots + optimal_green_radish_pots = total_pots) ∧
  (optimal_green_radish_pots ≤ optimal_spider_pots / 2) ∧
  (∀ a : ℕ, a + (total_pots - a) = total_pots →
    (total_pots - a) ≤ a / 2 →
    spider_plant_price * a + green_radish_price * (total_pots - a) ≥
    spider_plant_price * optimal_spider_pots + green_radish_price * optimal_green_radish_pots) :=
by sorry

end plant_prices_and_minimum_cost_l848_84882


namespace det_A_eq_cube_l848_84854

/-- The matrix A as defined in the problem -/
def A (x y z : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  !![1 + x^2 - y^2 - z^2, 2*(x*y + z), 2*(z*x - y);
    2*(x*y - z), 1 + y^2 - z^2 - x^2, 2*(y*z + x);
    2*(z*x + y), 2*(y*z - x), 1 + z^2 - x^2 - y^2]

/-- The theorem stating that the determinant of A is equal to (1 + x^2 + y^2 + z^2)^3 -/
theorem det_A_eq_cube (x y z : ℝ) : 
  Matrix.det (A x y z) = (1 + x^2 + y^2 + z^2)^3 := by
  sorry

end det_A_eq_cube_l848_84854


namespace greatest_power_of_seven_l848_84838

def r : ℕ := (List.range 50).foldl (· * ·) 1

theorem greatest_power_of_seven (k : ℕ) : k ≤ 8 ↔ (7^k : ℕ) ∣ r :=
sorry

end greatest_power_of_seven_l848_84838


namespace table_height_proof_l848_84846

/-- Given two configurations of a table and two identical wooden blocks,
    prove that the height of the table is 30 inches. -/
theorem table_height_proof (x y : ℝ) : 
  x + 30 - y = 32 ∧ y + 30 - x = 28 → 30 = 30 := by
  sorry

end table_height_proof_l848_84846


namespace expected_value_fair_12_sided_die_l848_84893

/-- A fair 12-sided die with faces numbered from 1 to 12 -/
def fair_12_sided_die : Finset ℕ := Finset.range 12

/-- The probability of each outcome for a fair 12-sided die -/
def prob_each_outcome : ℚ := 1 / 12

/-- The expected value of rolling a fair 12-sided die -/
def expected_value : ℚ := (fair_12_sided_die.sum (λ x => (x + 1) * prob_each_outcome))

/-- Theorem: The expected value of rolling a fair 12-sided die is 6.5 -/
theorem expected_value_fair_12_sided_die : expected_value = 13 / 2 := by
  sorry

end expected_value_fair_12_sided_die_l848_84893


namespace permutations_of_six_distinct_objects_l848_84814

theorem permutations_of_six_distinct_objects : Nat.factorial 6 = 720 := by
  sorry

end permutations_of_six_distinct_objects_l848_84814


namespace floor_tiling_floor_covered_l848_84858

/-- A square floor of size n × n can be completely covered by an equal number of 2 × 2 and 3 × 1 tiles if and only if n is a multiple of 7. -/
theorem floor_tiling (n : ℕ) : 
  (∃ (a : ℕ), n^2 = 7 * a) ↔ ∃ (k : ℕ), n = 7 * k :=
by sorry

/-- The number of tiles of each type needed to cover a square floor of size n × n, where n is a multiple of 7. -/
def num_tiles (n : ℕ) (h : ∃ (k : ℕ), n = 7 * k) : ℕ :=
  n^2 / 7

/-- Verification that the floor is completely covered using an equal number of 2 × 2 and 3 × 1 tiles. -/
theorem floor_covered (n : ℕ) (h : ∃ (k : ℕ), n = 7 * k) :
  let a := num_tiles n h
  4 * a + 3 * a = n^2 :=
by sorry

end floor_tiling_floor_covered_l848_84858


namespace inverse_variation_problem_l848_84888

theorem inverse_variation_problem (x y : ℝ) (k : ℝ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : x^2 * y = k) (h4 : 2^2 * 10 = k) (h5 : y = 4000) : x = 1/10 := by
  sorry

end inverse_variation_problem_l848_84888


namespace beadshop_profit_ratio_l848_84896

theorem beadshop_profit_ratio : 
  ∀ (total_profit monday_profit tuesday_profit wednesday_profit : ℝ),
    total_profit = 1200 →
    monday_profit = (1/3) * total_profit →
    wednesday_profit = 500 →
    tuesday_profit = total_profit - monday_profit - wednesday_profit →
    tuesday_profit / total_profit = 1/4 := by
  sorry

end beadshop_profit_ratio_l848_84896


namespace remaining_work_time_for_a_l848_84820

/-- The problem of calculating the remaining work time for person a -/
theorem remaining_work_time_for_a (a b c : ℝ) (h1 : a = 1 / 9) (h2 : b = 1 / 15) (h3 : c = 1 / 20) : 
  (1 - (10 * b + 5 * c)) / a = 3 / 4 := by
  sorry

end remaining_work_time_for_a_l848_84820


namespace circle_C_distance_range_l848_84886

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  (x - 4)^2 + (y - 2)^2 = 25

-- Define the points A and B
def point_A : ℝ × ℝ := (1, 1)
def point_B : ℝ × ℝ := (7, 4)

-- Define the function for |PA|^2 + |PB|^2
def sum_of_squared_distances (P : ℝ × ℝ) : ℝ :=
  (P.1 - point_A.1)^2 + (P.2 - point_A.2)^2 +
  (P.1 - point_B.1)^2 + (P.2 - point_B.2)^2

-- Theorem statement
theorem circle_C_distance_range :
  ∀ P : ℝ × ℝ, circle_C P.1 P.2 →
  103 ≤ sum_of_squared_distances P ∧ sum_of_squared_distances P ≤ 123 :=
sorry

end circle_C_distance_range_l848_84886


namespace weight_difference_l848_84816

/-- The weights of individuals A, B, C, D, and E --/
structure Weights where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ
  E : ℝ

/-- The conditions of the problem --/
def WeightConditions (w : Weights) : Prop :=
  (w.A + w.B + w.C) / 3 = 84 ∧
  (w.A + w.B + w.C + w.D) / 4 = 80 ∧
  (w.B + w.C + w.D + w.E) / 4 = 79 ∧
  w.A = 77 ∧
  w.E > w.D

/-- The theorem to prove --/
theorem weight_difference (w : Weights) (h : WeightConditions w) : w.E - w.D = 5 := by
  sorry

end weight_difference_l848_84816


namespace automotive_test_distance_l848_84892

theorem automotive_test_distance (d : ℝ) (h1 : d / 4 + d / 5 + d / 6 = 37) : 3 * d = 180 := by
  sorry

#check automotive_test_distance

end automotive_test_distance_l848_84892


namespace twin_brothers_age_l848_84899

theorem twin_brothers_age :
  ∀ x : ℕ,
  (x + 1) * (x + 1) = x * x + 17 →
  x = 8 :=
by
  sorry

end twin_brothers_age_l848_84899


namespace rabbit_travel_time_l848_84890

/-- Proves that a rabbit traveling at 5 miles per hour takes 24 minutes to cover 2 miles -/
theorem rabbit_travel_time :
  let speed : ℝ := 5  -- miles per hour
  let distance : ℝ := 2  -- miles
  let time_hours : ℝ := distance / speed
  let time_minutes : ℝ := time_hours * 60
  time_minutes = 24 := by sorry

end rabbit_travel_time_l848_84890


namespace lisa_investment_interest_l848_84810

/-- Calculates the interest earned on an investment with annual compounding -/
def interest_earned (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * ((1 + rate) ^ years - 1)

/-- The interest earned on Lisa's investment -/
theorem lisa_investment_interest :
  let principal : ℝ := 2000
  let rate : ℝ := 0.02
  let years : ℕ := 10
  ∃ ε > 0, |interest_earned principal rate years - 438| < ε :=
by sorry

end lisa_investment_interest_l848_84810


namespace gcd_problems_l848_84850

theorem gcd_problems : 
  (Nat.gcd 840 1785 = 105) ∧ (Nat.gcd 612 468 = 156) := by
  sorry

end gcd_problems_l848_84850


namespace polynomial_division_l848_84845

-- Define the polynomial
def p (a b c x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

-- Define the divisor polynomial
def q (x : ℝ) : ℝ := x^2 + 3*x - 4

-- State the theorem
theorem polynomial_division (a b c : ℝ) 
  (h : ∀ x, p a b c x = 0 → q x = 0) : 
  (4*a + c = 12) ∧ (2*a - 2*b - c = 14) := by
  sorry


end polynomial_division_l848_84845


namespace pizza_order_total_l848_84815

theorem pizza_order_total (m : ℕ) (total_pizzas : ℚ) : 
  m > 17 →
  (10 : ℚ) / m + 17 * ((10 : ℚ) / m) / 2 = total_pizzas →
  total_pizzas = 11 := by
  sorry

end pizza_order_total_l848_84815


namespace five_variable_inequality_l848_84834

theorem five_variable_inequality (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₃ > 0) (h₄ : x₄ > 0) (h₅ : x₅ > 0) : 
  (x₁ + x₂ + x₃ + x₄ + x₅)^2 ≥ 4*(x₁*x₂ + x₃*x₄ + x₅*x₁ + x₂*x₃ + x₄*x₅) := by
  sorry

end five_variable_inequality_l848_84834


namespace largest_multiple_less_than_negative_fifty_l848_84891

theorem largest_multiple_less_than_negative_fifty :
  ∀ n : ℤ, n * 12 < -50 → n * 12 ≤ -48 :=
by
  sorry

end largest_multiple_less_than_negative_fifty_l848_84891


namespace same_terminal_side_470_110_l848_84868

/-- Two angles have the same terminal side if their difference is a multiple of 360° --/
def same_terminal_side (α β : ℝ) : Prop := ∃ k : ℤ, α = β + k * 360

/-- The theorem states that 470° has the same terminal side as 110° --/
theorem same_terminal_side_470_110 : same_terminal_side 470 110 := by
  sorry

end same_terminal_side_470_110_l848_84868


namespace garden_length_l848_84842

/-- The length of a rectangular garden with perimeter 600 meters and breadth 200 meters is 100 meters. -/
theorem garden_length (perimeter : ℝ) (breadth : ℝ) (h1 : perimeter = 600) (h2 : breadth = 200) :
  2 * (breadth + (perimeter / 2 - breadth)) = perimeter ∧ perimeter / 2 - breadth = 100 := by
  sorry

end garden_length_l848_84842


namespace mike_books_before_sale_l848_84859

/-- The number of books Mike bought at the yard sale -/
def books_bought : ℕ := 21

/-- The total number of books Mike has now -/
def total_books_now : ℕ := 56

/-- The number of books Mike had before the yard sale -/
def books_before : ℕ := total_books_now - books_bought

theorem mike_books_before_sale : books_before = 35 := by
  sorry

end mike_books_before_sale_l848_84859


namespace modulus_of_z_l848_84828

theorem modulus_of_z (z : ℂ) : z = (2 * Complex.I) / (1 + Complex.I) → Complex.abs z = Real.sqrt 2 := by
  sorry

end modulus_of_z_l848_84828


namespace triangle_properties_l848_84804

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a line equation
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

def Triangle.hasAltitude (t : Triangle) (l : Line) : Prop :=
  l.a * t.A.1 + l.b * t.A.2 + l.c = 0

def Triangle.hasAngleBisector (t : Triangle) (l : Line) : Prop :=
  l.a * t.B.1 + l.b * t.B.2 + l.c = 0

theorem triangle_properties (t : Triangle) (altitude : Line) (bisector : Line) :
  t.A = (1, 1) →
  altitude = { a := 3, b := 1, c := -12 } →
  bisector = { a := 1, b := -2, c := 4 } →
  t.hasAltitude altitude →
  t.hasAngleBisector bisector →
  t.B = (-8, -2) ∧
  (∃ (l : Line), l = { a := 9, b := -13, c := 46 } ∧ 
    l.a * t.B.1 + l.b * t.B.2 + l.c = 0 ∧
    l.a * t.C.1 + l.b * t.C.2 + l.c = 0) :=
by sorry

end triangle_properties_l848_84804


namespace unique_zero_in_interval_l848_84876

def f (x : ℝ) := 2*x + x^3 - 2

theorem unique_zero_in_interval : ∃! x : ℝ, 0 < x ∧ x < 1 ∧ f x = 0 := by
  sorry

end unique_zero_in_interval_l848_84876


namespace chili_beans_cans_l848_84817

/-- Given a ratio of tomato soup cans to chili beans cans and the total number of cans,
    calculate the number of chili beans cans. -/
theorem chili_beans_cans (tomato_ratio chili_ratio total_cans : ℕ) :
  tomato_ratio ≠ 0 →
  chili_ratio = 2 * tomato_ratio →
  total_cans = tomato_ratio + chili_ratio →
  chili_ratio = 8 := by
  sorry

end chili_beans_cans_l848_84817


namespace square_sum_geq_product_sum_l848_84898

theorem square_sum_geq_product_sum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a^2 + b^2 + c^2 ≥ a*b + b*c + a*c :=
by sorry

end square_sum_geq_product_sum_l848_84898


namespace happy_number_transformation_l848_84841

def is_happy_number (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧
  (n / 100 + (n / 10 % 10) - n % 10 = 6)

def transform (m : ℕ) : ℕ :=
  let c := m % 10
  let a := m / 100
  let b := (m / 10) % 10
  2 * c * 100 + a * 10 + b

theorem happy_number_transformation :
  {m : ℕ | is_happy_number m ∧ is_happy_number (transform m)} = {532, 464} := by
  sorry

end happy_number_transformation_l848_84841


namespace opposite_of_three_l848_84827

theorem opposite_of_three : 
  ∃ x : ℝ, (3 + x = 0 ∧ x = -3) := by
  sorry

end opposite_of_three_l848_84827


namespace min_value_S_l848_84808

theorem min_value_S (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : x^2 + y^2 + z^2 = 1) :
  ∀ (x' y' z' : ℝ), x' > 0 → y' > 0 → z' > 0 → x'^2 + y'^2 + z'^2 = 1 →
    (1 + z) / (2 * x * y * z) ≤ (1 + z') / (2 * x' * y' * z') →
    (1 + z) / (2 * x * y * z) ≥ 4 :=
sorry

end min_value_S_l848_84808


namespace jellybeans_left_specific_l848_84807

/-- Calculates the number of jellybeans left in a jar after some children eat them. -/
def jellybeans_left (total : ℕ) (normal_class_size : ℕ) (absent : ℕ) (absent_eat : ℕ)
  (group1_size : ℕ) (group1_eat : ℕ) (group2_size : ℕ) (group2_eat : ℕ) : ℕ :=
  total - (group1_size * group1_eat + group2_size * group2_eat)

/-- Theorem stating the number of jellybeans left in the jar under specific conditions. -/
theorem jellybeans_left_specific : 
  jellybeans_left 250 24 2 7 12 5 10 4 = 150 := by
  sorry

end jellybeans_left_specific_l848_84807


namespace haley_marbles_count_l848_84862

def number_of_boys : ℕ := 2
def marbles_per_boy : ℕ := 10

theorem haley_marbles_count :
  number_of_boys * marbles_per_boy = 20 :=
by sorry

end haley_marbles_count_l848_84862


namespace cheryl_material_used_l848_84818

-- Define the amounts of materials
def material1 : ℚ := 2 / 9
def material2 : ℚ := 1 / 8
def leftover : ℚ := 4 / 18

-- Define the total amount bought
def total_bought : ℚ := material1 + material2

-- Define the theorem
theorem cheryl_material_used :
  total_bought - leftover = 1 / 8 := by
  sorry

end cheryl_material_used_l848_84818


namespace candy_bag_problem_l848_84830

theorem candy_bag_problem (n : ℕ) (r : ℕ) : 
  n > 0 →  -- Ensure the bag is not empty
  r > 0 →  -- Ensure there are red candies
  r ≤ n →  -- Ensure the number of red candies doesn't exceed the total
  (r : ℚ) / n = 5 / 6 →  -- Probability of choosing a red candy
  n = 6 :=
by sorry

end candy_bag_problem_l848_84830


namespace port_vessels_count_l848_84836

theorem port_vessels_count :
  let cruise_ships : ℕ := 4
  let cargo_ships : ℕ := 2 * cruise_ships
  let sailboats : ℕ := cargo_ships + 6
  let fishing_boats : ℕ := sailboats / 7
  let total_vessels : ℕ := cruise_ships + cargo_ships + sailboats + fishing_boats
  total_vessels = 28 :=
by sorry

end port_vessels_count_l848_84836


namespace distinct_prime_factors_sum_and_product_l848_84881

def number : Nat := 420

theorem distinct_prime_factors_sum_and_product :
  (Finset.sum (Nat.factors number).toFinset id = 17) ∧
  (Finset.prod (Nat.factors number).toFinset id = 210) := by
  sorry

end distinct_prime_factors_sum_and_product_l848_84881


namespace cube_volume_problem_l848_84824

theorem cube_volume_problem (a : ℝ) : 
  a > 0 → 
  (a + 2) * a * (a - 2) = a^3 - 24 → 
  a^3 = 216 := by
sorry

end cube_volume_problem_l848_84824


namespace f_min_at_4_l848_84839

/-- The quadratic function we're analyzing -/
def f (x : ℝ) : ℝ := x^2 - 8*x + 19

/-- Theorem stating that f attains its minimum at x = 4 -/
theorem f_min_at_4 : ∀ x : ℝ, f x ≥ f 4 := by sorry

end f_min_at_4_l848_84839


namespace four_Z_three_equals_negative_eleven_l848_84833

-- Define the Z operation
def Z (c d : ℤ) : ℤ := c^2 - 3*c*d + d^2

-- Theorem to prove
theorem four_Z_three_equals_negative_eleven : Z 4 3 = -11 := by
  sorry

end four_Z_three_equals_negative_eleven_l848_84833


namespace container_volume_increase_l848_84864

theorem container_volume_increase (original_volume : ℝ) :
  let new_volume := original_volume * 8
  2 * 2 * 2 * original_volume = new_volume :=
by sorry

end container_volume_increase_l848_84864


namespace sum_of_coefficients_l848_84805

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℝ) :
  (∀ x : ℝ, (1 + 2*x) * (1 - 2*x)^7 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8) →
  a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = 253 :=
by sorry

end sum_of_coefficients_l848_84805


namespace two_invariant_lines_l848_84806

/-- The transformation f: ℝ² → ℝ² defined by f(x,y) = (3y,2x) -/
def f (p : ℝ × ℝ) : ℝ × ℝ := (3 * p.2, 2 * p.1)

/-- A line in ℝ² represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- A line is invariant under f if for all points on the line, 
    their images under f also lie on the same line -/
def is_invariant (l : Line) : Prop :=
  ∀ x y : ℝ, y = l.slope * x + l.intercept → 
    (f (x, y)).2 = l.slope * (f (x, y)).1 + l.intercept

/-- There are exactly two distinct lines that are invariant under f -/
theorem two_invariant_lines : 
  ∃! (l1 l2 : Line), l1 ≠ l2 ∧ is_invariant l1 ∧ is_invariant l2 ∧
    (∀ l : Line, is_invariant l → l = l1 ∨ l = l2) :=
sorry

end two_invariant_lines_l848_84806


namespace number_of_proper_subsets_of_A_l848_84812

-- Define the universal set U
def U : Finset Nat := {0, 1, 2, 3}

-- Define set A based on its complement in U
def A : Finset Nat := U \ {2}

-- Theorem statement
theorem number_of_proper_subsets_of_A :
  (Finset.powerset A).card - 1 = 7 := by
  sorry

end number_of_proper_subsets_of_A_l848_84812


namespace tangent_line_to_circle_l848_84835

theorem tangent_line_to_circle (x y : ℝ) : 
  (∃ k : ℝ, (y = k * (x - Real.sqrt 2)) ∧ 
   ((k * x - y - k * Real.sqrt 2) ^ 2) / (k ^ 2 + 1) = 1) →
  (x - y - Real.sqrt 2 = 0 ∨ x + y - Real.sqrt 2 = 0) :=
by sorry

end tangent_line_to_circle_l848_84835


namespace carol_savings_per_week_l848_84856

/-- Proves that Carol saves $9 per week given the initial conditions and final equality of savings --/
theorem carol_savings_per_week (carol_initial : ℕ) (mike_initial : ℕ) (mike_savings : ℕ) (weeks : ℕ)
  (h1 : carol_initial = 60)
  (h2 : mike_initial = 90)
  (h3 : mike_savings = 3)
  (h4 : weeks = 5)
  (h5 : ∃ (carol_savings : ℕ), carol_initial + weeks * carol_savings = mike_initial + weeks * mike_savings) :
  ∃ (carol_savings : ℕ), carol_savings = 9 := by
  sorry

end carol_savings_per_week_l848_84856


namespace no_eulerian_path_four_odd_degree_l848_84865

/-- A simple graph represented by its vertex set and a function determining adjacency. -/
structure Graph (V : Type) where
  adj : V → V → Prop

/-- The degree of a vertex in a graph is the number of edges incident to it. -/
def degree (G : Graph V) (v : V) : ℕ := sorry

/-- A vertex has odd degree if its degree is odd. -/
def has_odd_degree (G : Graph V) (v : V) : Prop :=
  Odd (degree G v)

/-- An Eulerian path in a graph is a path that visits every edge exactly once. -/
def has_eulerian_path (G : Graph V) : Prop := sorry

/-- The main theorem: a graph with four vertices of odd degree does not have an Eulerian path. -/
theorem no_eulerian_path_four_odd_degree (V : Type) (G : Graph V) 
  (h : ∃ (a b c d : V), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
    has_odd_degree G a ∧ has_odd_degree G b ∧ has_odd_degree G c ∧ has_odd_degree G d) :
  ¬ has_eulerian_path G := by sorry

end no_eulerian_path_four_odd_degree_l848_84865


namespace hyperbola_properties_l848_84863

/-- Given a hyperbola with the equation (x²/a² - y²/b² = 1) where a > 0 and b > 0,
    if a perpendicular line from the right focus to an asymptote has length 2 and slope -1/2,
    then b = 2, the hyperbola equation is x² - y²/4 = 1, and the foot of the perpendicular
    is at (√5/5, 2√5/5). -/
theorem hyperbola_properties (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y c : ℝ),
    (x^2/a^2 - y^2/b^2 = 1) ∧  -- Equation of the hyperbola
    (c^2 = a^2 + b^2) ∧        -- Relation between c and a, b
    ((a^2/c - c)^2 + (a*b/c)^2 = 4) ∧  -- Length of perpendicular = 2
    (-1/2 = (a*b/c) / (a^2/c - c))) →  -- Slope of perpendicular = -1/2
  (b = 2 ∧ 
   (∀ x y, x^2 - y^2/4 = 1 ↔ x^2/a^2 - y^2/b^2 = 1) ∧
   (∃ x y, x = Real.sqrt 5 / 5 ∧ y = 2 * Real.sqrt 5 / 5 ∧
           b*x - a*y = 0 ∧ y = -a/b * (x - Real.sqrt (a^2 + b^2)))) :=
by sorry

end hyperbola_properties_l848_84863


namespace simplify_fraction_l848_84879

theorem simplify_fraction : (120 : ℚ) / 1800 = 1 / 15 := by sorry

end simplify_fraction_l848_84879


namespace product_smallest_prime_composite_l848_84823

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def isComposite (n : ℕ) : Prop := n > 1 ∧ ∃ m : ℕ, 1 < m ∧ m < n ∧ m ∣ n

def smallestPrime : ℕ := 2

def smallestComposite : ℕ := 4

theorem product_smallest_prime_composite :
  isPrime smallestPrime ∧
  isComposite smallestComposite ∧
  (∀ p : ℕ, isPrime p → p ≥ smallestPrime) ∧
  (∀ c : ℕ, isComposite c → c ≥ smallestComposite) →
  smallestPrime * smallestComposite = 8 :=
by sorry

end product_smallest_prime_composite_l848_84823


namespace no_function_satisfies_inequality_l848_84878

theorem no_function_satisfies_inequality :
  ∀ f : ℕ → ℕ, ∃ m n : ℕ, (m + f n)^2 < 3 * (f m)^2 + n^2 := by
  sorry

end no_function_satisfies_inequality_l848_84878


namespace stall_owner_earnings_l848_84871

/-- Represents the stall owner's game with ping-pong balls -/
structure BallGame where
  yellow_balls : ℕ := 3
  white_balls : ℕ := 2
  balls_drawn : ℕ := 3
  same_color_reward : ℕ := 5
  diff_color_cost : ℕ := 1
  daily_players : ℕ := 100
  days_in_month : ℕ := 30

/-- Calculates the expected monthly earnings of the stall owner -/
def expected_monthly_earnings (game : BallGame) : ℚ :=
  let total_balls := game.yellow_balls + game.white_balls
  let prob_same_color := (game.yellow_balls.choose game.balls_drawn) / (total_balls.choose game.balls_drawn)
  let daily_earnings := game.daily_players * (game.diff_color_cost * (1 - prob_same_color) - game.same_color_reward * prob_same_color)
  daily_earnings * game.days_in_month

/-- Theorem stating the expected monthly earnings of the stall owner -/
theorem stall_owner_earnings (game : BallGame) : 
  expected_monthly_earnings game = 1200 := by
  sorry

end stall_owner_earnings_l848_84871


namespace planes_parallel_if_perp_to_parallel_lines_l848_84813

-- Define the types for planes and lines
variable (α β : Plane) (l m : Line)

-- Define the relationships between planes and lines
def perpendicular (l : Line) (α : Plane) : Prop := sorry
def parallel_lines (l m : Line) : Prop := sorry
def parallel_planes (α β : Plane) : Prop := sorry

-- State the theorem
theorem planes_parallel_if_perp_to_parallel_lines 
  (h1 : perpendicular l α) 
  (h2 : perpendicular m β) 
  (h3 : parallel_lines l m) : 
  parallel_planes α β := by sorry

end planes_parallel_if_perp_to_parallel_lines_l848_84813


namespace union_equality_implies_a_geq_one_l848_84866

def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 1}
def B (a : ℝ) : Set ℝ := {x : ℝ | x ≤ a}

theorem union_equality_implies_a_geq_one (a : ℝ) :
  A ∪ B a = B a → a ≥ 1 := by
  sorry

end union_equality_implies_a_geq_one_l848_84866


namespace simplify_trig_expression_l848_84837

theorem simplify_trig_expression :
  (Real.sin (25 * π / 180) + Real.sin (35 * π / 180)) /
  (Real.cos (25 * π / 180) + Real.cos (35 * π / 180)) =
  Real.tan (30 * π / 180) := by sorry

end simplify_trig_expression_l848_84837


namespace tan_585_degrees_l848_84829

theorem tan_585_degrees : Real.tan (585 * π / 180) = 1 := by
  sorry

end tan_585_degrees_l848_84829


namespace quadrilateral_with_equal_opposite_sides_and_one_right_angle_not_necessarily_rectangle_l848_84861

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : Point)

-- Define properties of a quadrilateral
def has_opposite_sides_equal (q : Quadrilateral) : Prop := sorry
def has_one_right_angle (q : Quadrilateral) : Prop := sorry
def is_rectangle (q : Quadrilateral) : Prop := sorry

-- Theorem statement
theorem quadrilateral_with_equal_opposite_sides_and_one_right_angle_not_necessarily_rectangle :
  ∃ q : Quadrilateral, has_opposite_sides_equal q ∧ has_one_right_angle q ∧ ¬is_rectangle q := by
  sorry

end quadrilateral_with_equal_opposite_sides_and_one_right_angle_not_necessarily_rectangle_l848_84861


namespace light_bulb_ratio_l848_84832

theorem light_bulb_ratio (initial : ℕ) (used : ℕ) (left : ℕ) : 
  initial = 40 → used = 16 → left = 12 → 
  (initial - used - left) = left := by
  sorry

end light_bulb_ratio_l848_84832


namespace even_function_implies_a_zero_l848_84880

def f (a : ℝ) (x : ℝ) : ℝ := x^2 - abs (x + a)

theorem even_function_implies_a_zero (a : ℝ) :
  (∀ x : ℝ, f a x = f a (-x)) → a = 0 := by
  sorry

end even_function_implies_a_zero_l848_84880


namespace alpha_beta_sum_l848_84870

theorem alpha_beta_sum (α β : ℝ) :
  (∀ x : ℝ, (x - α) / (x + β) = (x^2 - 75*x + 1236) / (x^2 + 60*x - 3120)) →
  α + β = 139 := by
sorry

end alpha_beta_sum_l848_84870


namespace sufficient_not_necessary_l848_84857

theorem sufficient_not_necessary (x : ℝ) :
  ((x + 1) * (x - 2) > 0 → abs x ≥ 1) ∧
  ¬(abs x ≥ 1 → (x + 1) * (x - 2) > 0) :=
by sorry

end sufficient_not_necessary_l848_84857


namespace festival_expense_sharing_l848_84849

theorem festival_expense_sharing 
  (C D X : ℝ) 
  (h1 : C > D) 
  (h2 : C > 0) 
  (h3 : D > 0) 
  (h4 : X > 0) :
  let total_expense := C + D + X
  let alex_share := (2/3) * total_expense
  let morgan_share := (1/3) * total_expense
  let alex_paid := C + X/2
  let morgan_paid := D + X/2
  morgan_share - morgan_paid = (1/3)*C - (2/3)*D + X := by
sorry

end festival_expense_sharing_l848_84849


namespace negation_equivalence_l848_84848

open Real

theorem negation_equivalence : 
  (¬ ∃ x₀ : ℝ, (2 / x₀) + log x₀ ≤ 0) ↔ (∀ x : ℝ, (2 / x) + log x > 0) :=
by sorry

end negation_equivalence_l848_84848


namespace matrix_inverse_zero_l848_84826

def A : Matrix (Fin 2) (Fin 2) ℝ := !![4, -6; -2, 3]

theorem matrix_inverse_zero : 
  A⁻¹ = !![0, 0; 0, 0] := by sorry

end matrix_inverse_zero_l848_84826


namespace timmy_initial_money_l848_84884

/-- Represents the properties of oranges and Timmy's situation --/
structure OrangeProblem where
  calories_per_orange : ℕ
  cost_per_orange : ℚ
  calories_needed : ℕ
  money_left : ℚ

/-- Calculates Timmy's initial amount of money --/
def initial_money (p : OrangeProblem) : ℚ :=
  let oranges_needed := p.calories_needed / p.calories_per_orange
  let oranges_cost := oranges_needed * p.cost_per_orange
  oranges_cost + p.money_left

/-- Theorem stating that given the problem conditions, Timmy's initial money was $10.00 --/
theorem timmy_initial_money :
  let p : OrangeProblem := {
    calories_per_orange := 80,
    cost_per_orange := 6/5, -- $1.20 represented as a rational number
    calories_needed := 400,
    money_left := 4
  }
  initial_money p = 10 := by sorry

end timmy_initial_money_l848_84884


namespace circle_C_properties_l848_84889

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y + 1)^2 = 2

-- Define the lines
def line1 (x y : ℝ) : Prop := x - y = 0
def line2 (x y : ℝ) : Prop := x - y - 4 = 0
def line3 (x y : ℝ) : Prop := x + y = 0

-- Define tangency
def is_tangent (circle : (ℝ → ℝ → Prop)) (line : (ℝ → ℝ → Prop)) : Prop :=
  ∃ (x y : ℝ), circle x y ∧ line x y ∧
  ∀ (x' y' : ℝ), line x' y' → (x' - x)^2 + (y' - y)^2 ≥ 2

-- State the theorem
theorem circle_C_properties :
  is_tangent circle_C line1 ∧
  is_tangent circle_C line2 ∧
  ∃ (x y : ℝ), circle_C x y ∧ line3 x y :=
sorry

end circle_C_properties_l848_84889


namespace units_digit_of_7_pow_3_pow_5_l848_84821

def units_digit_pattern : ℕ → ℕ
| 0 => 7
| 1 => 9
| 2 => 3
| 3 => 1
| n + 4 => units_digit_pattern n

def power_mod (base exponent modulus : ℕ) : ℕ :=
  (base ^ exponent) % modulus

theorem units_digit_of_7_pow_3_pow_5 :
  units_digit_pattern (power_mod 3 5 4) = 3 := by sorry

end units_digit_of_7_pow_3_pow_5_l848_84821


namespace min_value_theorem_l848_84853

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  4 / x + 9 / y ≥ 25 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + y₀ = 1 ∧ 4 / x₀ + 9 / y₀ = 25 := by
  sorry

end min_value_theorem_l848_84853


namespace max_sum_with_constraints_l848_84874

theorem max_sum_with_constraints (a b : ℝ) 
  (h1 : 4 * a + 3 * b ≤ 10) 
  (h2 : 3 * a + 6 * b ≤ 12) : 
  a + b ≤ 14/5 ∧ ∃ (a' b' : ℝ), 4 * a' + 3 * b' = 10 ∧ 3 * a' + 6 * b' = 12 ∧ a' + b' = 14/5 :=
by
  sorry

end max_sum_with_constraints_l848_84874


namespace solve_exponential_equation_l848_84897

theorem solve_exponential_equation :
  ∃ x : ℝ, 5^(3*x) = Real.sqrt 125 ∧ x = (1/2 : ℝ) := by
  sorry

end solve_exponential_equation_l848_84897


namespace smallest_prime_twelve_less_than_square_l848_84819

theorem smallest_prime_twelve_less_than_square : ∃ n : ℕ, 
  (n > 0) ∧ 
  (Nat.Prime n) ∧ 
  (∃ m : ℕ, n = m^2 - 12) ∧
  (∀ k : ℕ, k > 0 → Nat.Prime k → (∃ l : ℕ, k = l^2 - 12) → k ≥ n) ∧
  n = 13 := by
sorry

end smallest_prime_twelve_less_than_square_l848_84819


namespace arithmetic_geometric_sequence_ratio_l848_84844

theorem arithmetic_geometric_sequence_ratio 
  (a : ℕ → ℝ) 
  (d : ℝ) 
  (h1 : d ≠ 0)
  (h2 : ∀ n, a (n + 1) = a n + d)  -- arithmetic sequence definition
  (h3 : (a 5)^2 = a 1 * a 17)  -- geometric sequence property
  : (a 5) / (a 1) = 3 :=
by sorry

end arithmetic_geometric_sequence_ratio_l848_84844


namespace triangle_altitude_on_rectangle_diagonal_l848_84894

/-- Given a rectangle with length l and width w, and a triangle constructed on its diagonal
    such that the area of the triangle equals the area of the rectangle,
    the altitude of the triangle drawn to the diagonal is 2lw / √(l^2 + w^2). -/
theorem triangle_altitude_on_rectangle_diagonal 
  (l w : ℝ) (hl : l > 0) (hw : w > 0) : 
  let diagonal := Real.sqrt (l^2 + w^2)
  let rectangle_area := l * w
  let triangle_area := (1 / 2) * diagonal * (2 * l * w / diagonal)
  triangle_area = rectangle_area →
  2 * l * w / diagonal = 2 * l * w / Real.sqrt (l^2 + w^2) :=
by sorry

end triangle_altitude_on_rectangle_diagonal_l848_84894


namespace birth_interval_proof_l848_84875

/-- Proves that the interval between births is 2 years given the conditions of the problem -/
theorem birth_interval_proof (num_children : ℕ) (youngest_age : ℕ) (total_age : ℕ) :
  num_children = 5 →
  youngest_age = 7 →
  total_age = 55 →
  (∃ interval : ℕ,
    total_age = youngest_age * num_children + interval * (num_children * (num_children - 1)) / 2 ∧
    interval = 2) := by
  sorry

end birth_interval_proof_l848_84875


namespace max_a_no_lattice_points_l848_84860

def is_lattice_point (x y : ℚ) : Prop := ∃ (n m : ℤ), x = n ∧ y = m

theorem max_a_no_lattice_points :
  ∃ (a : ℚ), a = 17/51 ∧
  (∀ (m x : ℚ), 1/3 < m → m < a → 0 < x → x ≤ 50 → 
    ¬ is_lattice_point x (m * x + 3)) ∧
  (∀ (a' : ℚ), a' > a → 
    ∃ (m x : ℚ), 1/3 < m → m < a' → 0 < x → x ≤ 50 → 
      is_lattice_point x (m * x + 3)) :=
sorry

end max_a_no_lattice_points_l848_84860


namespace melanie_book_count_l848_84840

/-- The total number of books Melanie has after buying more books is equal to the sum of her initial book count and the number of books she bought. -/
theorem melanie_book_count (initial_books new_books : ℝ) :
  let total_books := initial_books + new_books
  total_books = initial_books + new_books :=
by sorry

end melanie_book_count_l848_84840


namespace union_of_A_and_B_l848_84869

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -2 < x ∧ x ≤ 3}
def B : Set ℝ := {x : ℝ | 1 < x ∧ x < 5}

-- State the theorem
theorem union_of_A_and_B :
  A ∪ B = {x : ℝ | -2 < x ∧ x < 5} := by
  sorry

end union_of_A_and_B_l848_84869


namespace calvins_weight_loss_l848_84843

/-- Calvin's weight loss problem -/
theorem calvins_weight_loss
  (initial_weight : ℕ)
  (weight_loss_per_month : ℕ)
  (months : ℕ)
  (hw : initial_weight = 250)
  (hl : weight_loss_per_month = 8)
  (hm : months = 12) :
  initial_weight - (weight_loss_per_month * months) = 154 :=
by sorry

end calvins_weight_loss_l848_84843


namespace remainder_sum_l848_84809

theorem remainder_sum (x y : ℤ) 
  (hx : x % 80 = 75) 
  (hy : y % 120 = 117) : 
  (x + y) % 40 = 32 := by
sorry

end remainder_sum_l848_84809


namespace max_value_3m_4n_l848_84852

/-- The sum of the first m positive even numbers -/
def sumEven (m : ℕ) : ℕ := m * (m + 1)

/-- The sum of the first n positive odd numbers -/
def sumOdd (n : ℕ) : ℕ := n^2

/-- The constraint that the sum of m distinct positive even numbers 
    and n distinct positive odd numbers is 1987 -/
def constraint (m n : ℕ) : Prop := sumEven m + sumOdd n = 1987

/-- The theorem stating that the maximum value of 3m + 4n is 221 
    given the constraint -/
theorem max_value_3m_4n : 
  ∀ m n : ℕ, constraint m n → 3 * m + 4 * n ≤ 221 :=
sorry

end max_value_3m_4n_l848_84852


namespace simplify_square_roots_l848_84867

theorem simplify_square_roots : 
  (Real.sqrt 450 / Real.sqrt 200) - (Real.sqrt 98 / Real.sqrt 32) = -1/4 := by
  sorry

end simplify_square_roots_l848_84867


namespace sine_monotonicity_l848_84800

theorem sine_monotonicity (φ : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = Real.sin (2 * x + φ))
  (h2 : ∀ x, f x ≤ |f (π / 6)|) (h3 : f (π / 2) > f π) :
  ∀ k : ℤ, StrictMonoOn f (Set.Icc (k * π + π / 6) (k * π + 2 * π / 3)) :=
by sorry

end sine_monotonicity_l848_84800


namespace heptagon_angle_measure_l848_84895

-- Define the heptagon
structure Heptagon where
  G : ℝ
  E : ℝ
  O : ℝ
  M : ℝ
  T : ℝ
  R : ℝ
  Y : ℝ

-- Define the theorem
theorem heptagon_angle_measure (GEOMETRY : Heptagon) : 
  GEOMETRY.G = GEOMETRY.E ∧ 
  GEOMETRY.G = GEOMETRY.T ∧ 
  GEOMETRY.O + GEOMETRY.Y = 180 ∧
  GEOMETRY.M = GEOMETRY.R ∧
  GEOMETRY.M = 160 →
  GEOMETRY.G = 400 / 3 := by
sorry

end heptagon_angle_measure_l848_84895


namespace exists_nat_square_not_positive_exists_real_not_root_quadratic_always_positive_exists_prime_not_odd_l848_84802

-- 1. There exists a natural number whose square is not positive.
theorem exists_nat_square_not_positive : ∃ n : ℕ, ¬(n^2 > 0) := by sorry

-- 2. There exists a real number x that is not a root of the equation 5x-12=0.
theorem exists_real_not_root : ∃ x : ℝ, 5*x - 12 ≠ 0 := by sorry

-- 3. For all x ∈ ℝ, x^2 - 3x + 3 > 0.
theorem quadratic_always_positive : ∀ x : ℝ, x^2 - 3*x + 3 > 0 := by sorry

-- 4. There exists a prime number that is not odd.
theorem exists_prime_not_odd : ∃ p : ℕ, Nat.Prime p ∧ ¬Odd p := by sorry

end exists_nat_square_not_positive_exists_real_not_root_quadratic_always_positive_exists_prime_not_odd_l848_84802


namespace cycle_selling_price_l848_84811

theorem cycle_selling_price (cost_price : ℝ) (loss_percentage : ℝ) : 
  cost_price = 1200 → loss_percentage = 15 → 
  cost_price * (1 - loss_percentage / 100) = 1020 := by
  sorry

end cycle_selling_price_l848_84811


namespace matching_color_probability_l848_84872

def total_jellybeans_ava : ℕ := 4
def total_jellybeans_ben : ℕ := 8

def green_jellybeans_ava : ℕ := 2
def red_jellybeans_ava : ℕ := 2
def green_jellybeans_ben : ℕ := 2
def red_jellybeans_ben : ℕ := 3

theorem matching_color_probability :
  let p_green := (green_jellybeans_ava / total_jellybeans_ava) * (green_jellybeans_ben / total_jellybeans_ben)
  let p_red := (red_jellybeans_ava / total_jellybeans_ava) * (red_jellybeans_ben / total_jellybeans_ben)
  p_green + p_red = 5 / 16 := by
sorry

end matching_color_probability_l848_84872


namespace garden_area_l848_84885

theorem garden_area (total_posts : ℕ) (post_spacing : ℕ) (longer_side_ratio : ℕ) : 
  total_posts = 20 →
  post_spacing = 4 →
  longer_side_ratio = 2 →
  ∃ (short_side long_side : ℕ),
    short_side * long_side = 336 ∧
    short_side * longer_side_ratio = long_side ∧
    short_side * post_spacing = (short_side - 1) * post_spacing ∧
    long_side * post_spacing = (long_side - 1) * post_spacing ∧
    2 * (short_side + long_side) - 4 = total_posts :=
by sorry

end garden_area_l848_84885


namespace star_four_three_l848_84873

-- Define the star operation
def star (a b : ℝ) : ℝ := a^2 - a*b + b^2 + 2*a*b

-- State the theorem
theorem star_four_three : star 4 3 = 37 := by
  sorry

end star_four_three_l848_84873


namespace hannahs_work_hours_l848_84825

/-- Given Hannah's work conditions, prove the number of hours she worked -/
theorem hannahs_work_hours 
  (hourly_rate : ℕ) 
  (late_penalty : ℕ) 
  (times_late : ℕ) 
  (total_pay : ℕ) 
  (h1 : hourly_rate = 30)
  (h2 : late_penalty = 5)
  (h3 : times_late = 3)
  (h4 : total_pay = 525) :
  ∃ (hours_worked : ℕ), 
    hours_worked * hourly_rate - times_late * late_penalty = total_pay ∧ 
    hours_worked = 18 := by
  sorry

end hannahs_work_hours_l848_84825


namespace simplify_expression_l848_84877

theorem simplify_expression :
  ∃ (a b c : ℕ+),
    (((Real.sqrt 3 - 1) ^ (2 - Real.sqrt 5)) / ((Real.sqrt 3 + 1) ^ (2 + Real.sqrt 5)) =
     (1 - (1/2) * Real.sqrt 3) * (2 ^ (-Real.sqrt 5))) ∧
    (∀ (p : ℕ), Nat.Prime p → ¬(p^2 ∣ c.val)) ∧
    ((1 - (1/2) * Real.sqrt 3) * (2 ^ (-Real.sqrt 5)) = a - b * Real.sqrt c) :=
by sorry

end simplify_expression_l848_84877


namespace one_in_set_implies_x_one_or_neg_one_l848_84855

theorem one_in_set_implies_x_one_or_neg_one (x : ℝ) :
  (1 ∈ ({x, x^2} : Set ℝ)) → (x = 1 ∨ x = -1) := by
  sorry

end one_in_set_implies_x_one_or_neg_one_l848_84855


namespace sandwich_ratio_l848_84801

theorem sandwich_ratio : ∀ (first_day : ℕ), 
  first_day + (first_day - 2) + 2 = 12 →
  (first_day : ℚ) / 12 = 1 / 2 := by
sorry

end sandwich_ratio_l848_84801


namespace quartic_polynomial_value_l848_84887

def is_monic_quartic (q : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, ∀ x, q x = x^4 + a*x^3 + b*x^2 + c*x + d

theorem quartic_polynomial_value (q : ℝ → ℝ) :
  is_monic_quartic q →
  q 1 = 3 →
  q 2 = 6 →
  q 3 = 11 →
  q 4 = 18 →
  q 5 = 51 := by
  sorry

end quartic_polynomial_value_l848_84887


namespace triangle_median_sum_bounds_l848_84822

/-- For any triangle, the sum of its medians is greater than 3/4 of its perimeter
    but less than its perimeter. -/
theorem triangle_median_sum_bounds (a b c m_a m_b m_c : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_median_a : m_a^2 = (2*b^2 + 2*c^2 - a^2) / 4)
  (h_median_b : m_b^2 = (2*c^2 + 2*a^2 - b^2) / 4)
  (h_median_c : m_c^2 = (2*a^2 + 2*b^2 - c^2) / 4) :
  3/4 * (a + b + c) < m_a + m_b + m_c ∧ m_a + m_b + m_c < a + b + c := by
sorry

end triangle_median_sum_bounds_l848_84822


namespace derivative_at_three_l848_84883

theorem derivative_at_three : 
  let f (x : ℝ) := (x + 3) / (x^2 + 3)
  deriv f 3 = -1/6 := by sorry

end derivative_at_three_l848_84883


namespace parakeets_per_cage_l848_84847

theorem parakeets_per_cage (num_cages : ℕ) (parrots_per_cage : ℕ) (total_birds : ℕ) 
  (h1 : num_cages = 6)
  (h2 : parrots_per_cage = 2)
  (h3 : total_birds = 54) :
  (total_birds - num_cages * parrots_per_cage) / num_cages = 7 := by
  sorry

end parakeets_per_cage_l848_84847


namespace area_of_locus_enclosed_l848_84851

/-- The locus of the center of a circle touching y = -x and passing through (0, 1) -/
def locusOfCenter (x y : ℝ) : Prop :=
  x = y + Real.sqrt (4 * y - 2) ∨ x = y - Real.sqrt (4 * y - 2)

/-- The area enclosed by the locus and the line y = 1 -/
noncomputable def enclosedArea : ℝ :=
  ∫ y in (0)..(1), 2 * Real.sqrt (4 * y - 2)

theorem area_of_locus_enclosed : enclosedArea = 2 * Real.sqrt 2 / 3 := by
  sorry

end area_of_locus_enclosed_l848_84851


namespace min_xy_value_l848_84831

theorem min_xy_value (x y : ℕ+) (h : (1 : ℚ) / x + (1 : ℚ) / (3 * y) = (1 : ℚ) / 8) : 
  (x * y : ℕ) ≥ 96 := by
sorry

end min_xy_value_l848_84831


namespace M_inter_N_eq_l848_84803

def M : Set ℤ := {x | -3 < x ∧ x < 3}
def N : Set ℤ := {x | x < 1}

theorem M_inter_N_eq : M ∩ N = {-2, -1, 0} := by
  sorry

end M_inter_N_eq_l848_84803
