import Mathlib

namespace min_value_F_l1519_151952

/-- The function F as defined in the problem -/
def F (m n : ℝ) : ℝ := (m - n)^2 + (m^2 - n + 1)^2

/-- Theorem stating that the minimum value of F(m,n) is 9/32 -/
theorem min_value_F :
  ∀ m n : ℝ, F m n ≥ 9/32 := by
  sorry

end min_value_F_l1519_151952


namespace resort_cost_theorem_l1519_151973

def resort_problem (swimming_pool_cost : ℝ) : Prop :=
  let first_cabin_cost := swimming_pool_cost
  let second_cabin_cost := first_cabin_cost / 2
  let third_cabin_cost := second_cabin_cost / 3
  let land_cost := 4 * swimming_pool_cost
  swimming_pool_cost + first_cabin_cost + second_cabin_cost + third_cabin_cost + land_cost = 150000

theorem resort_cost_theorem :
  ∃ (swimming_pool_cost : ℝ), resort_problem swimming_pool_cost :=
sorry

end resort_cost_theorem_l1519_151973


namespace range_of_a_l1519_151905

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | 2 * a ≤ x ∧ x ≤ a^2 + 1}
def B (a : ℝ) : Set ℝ := {x | x^2 - 3*(a+1)*x + 2*(3*a+1) ≤ 0}

-- Define the condition that p is sufficient for q
def p_sufficient_for_q (a : ℝ) : Prop := A a ⊆ B a

-- State the theorem
theorem range_of_a : 
  ∀ a : ℝ, p_sufficient_for_q a ↔ (1 ≤ a ∧ a ≤ 3) ∨ a = -1 :=
sorry

end range_of_a_l1519_151905


namespace tomato_yield_per_plant_l1519_151995

theorem tomato_yield_per_plant 
  (rows : ℕ) 
  (plants_per_row : ℕ) 
  (total_yield : ℕ) 
  (h1 : rows = 30)
  (h2 : plants_per_row = 10)
  (h3 : total_yield = 6000) :
  total_yield / (rows * plants_per_row) = 20 := by
sorry

end tomato_yield_per_plant_l1519_151995


namespace inequalities_problem_l1519_151928

theorem inequalities_problem (a b c d : ℝ) 
  (ha : a > 0) 
  (hb1 : 0 > b) 
  (hb2 : b > -a) 
  (hc : c < d) 
  (hd : d < 0) : 
  (a / b + b / c < 0) ∧ 
  (a - c > b - d) ∧ 
  (a * (d - c) > b * (d - c)) := by
sorry

end inequalities_problem_l1519_151928


namespace quadratic_intercept_l1519_151970

/-- A quadratic function with vertex (5,10) and one x-intercept at (0,0) has its other x-intercept at x = 10 -/
theorem quadratic_intercept (a b c : ℝ) : 
  (∀ x, a * x^2 + b * x + c = 10 - a * (x - 5)^2) →  -- vertex form with vertex (5,10)
  (0^2 * a + 0 * b + c = 0) →                        -- (0,0) is an x-intercept
  (∃ x, x ≠ 0 ∧ x^2 * a + x * b + c = 0 ∧ x = 10) := by
sorry

end quadratic_intercept_l1519_151970


namespace quadratic_trinomial_theorem_l1519_151951

/-- A quadratic trinomial with real coefficients -/
structure QuadraticTrinomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Condition: replacing any coefficient with 1 results in a trinomial with exactly one root -/
def has_single_root_when_replaced (q : QuadraticTrinomial) : Prop :=
  (1^2 - 4*q.b*q.c = 0) ∧ (q.b^2 - 4*1*q.c = 0) ∧ (q.b^2 - 4*q.a*1 = 0)

/-- Theorem: If a quadratic trinomial satisfies the condition, then its coefficients are a = c = 1/2 and b = ±√2 -/
theorem quadratic_trinomial_theorem (q : QuadraticTrinomial) :
  has_single_root_when_replaced q →
  (q.a = 1/2 ∧ q.c = 1/2 ∧ (q.b = Real.sqrt 2 ∨ q.b = -Real.sqrt 2)) :=
by sorry

end quadratic_trinomial_theorem_l1519_151951


namespace modulus_complex_l1519_151983

theorem modulus_complex (α : Real) (h : π < α ∧ α < 2*π) :
  Complex.abs (1 + Complex.cos α + Complex.I * Complex.sin α) = 2 * Real.cos (α/2) := by
  sorry

end modulus_complex_l1519_151983


namespace equation_equivalence_l1519_151972

theorem equation_equivalence (x y : ℝ) 
  (hx1 : x ≠ 0) (hx2 : x ≠ 3) (hy1 : y ≠ 0) (hy2 : y ≠ 5) :
  (3 / x + 4 / y = 1 / 3) ↔ (9 * y / (y - 12) = x) :=
sorry

end equation_equivalence_l1519_151972


namespace negative_number_identification_l1519_151992

theorem negative_number_identification : 
  ((-1 : ℝ) < 0) ∧ (¬(0 < 0)) ∧ (¬(2 < 0)) ∧ (¬(Real.sqrt 2 < 0)) := by
  sorry

end negative_number_identification_l1519_151992


namespace sum_of_squared_coefficients_l1519_151912

/-- The polynomial resulting from simplifying 3(x^3 - x^2 + 4) - 5(x^4 - 2x^3 + x - 1) -/
def simplified_polynomial (x : ℝ) : ℝ :=
  -5 * x^4 + 13 * x^3 - 3 * x^2 - 5 * x + 17

/-- The coefficients of the simplified polynomial -/
def coefficients : List ℝ := [-5, 13, -3, -5, 17]

theorem sum_of_squared_coefficients :
  (coefficients.map (λ c => c^2)).sum = 517 := by sorry

end sum_of_squared_coefficients_l1519_151912


namespace cube_to_rectangular_solid_surface_area_ratio_l1519_151961

/-- The ratio of the surface area of a cube to the surface area of a rectangular solid
    with doubled length is 3/5. -/
theorem cube_to_rectangular_solid_surface_area_ratio :
  ∀ s : ℝ, s > 0 →
  (6 * s^2) / (2 * (2*s*s + 2*s*s + s*s)) = 3 / 5 := by
  sorry

end cube_to_rectangular_solid_surface_area_ratio_l1519_151961


namespace numerical_puzzle_solutions_l1519_151910

/-- A function that checks if a number is a two-digit number -/
def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

/-- A function that extracts the tens digit of a two-digit number -/
def tens_digit (n : ℕ) : ℕ :=
  n / 10

/-- A function that extracts the ones digit of a two-digit number -/
def ones_digit (n : ℕ) : ℕ :=
  n % 10

/-- The main theorem stating the solutions to the numerical puzzle -/
theorem numerical_puzzle_solutions :
  ∀ n : ℕ, is_two_digit n →
    (∃ b v : ℕ, 
      n = b^v ∧ 
      tens_digit n ≠ ones_digit n ∧
      b = ones_digit n) ↔ 
    (n = 32 ∨ n = 36 ∨ n = 64) :=
sorry

end numerical_puzzle_solutions_l1519_151910


namespace inequality_proof_l1519_151979

theorem inequality_proof (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : a - d > b - c := by
  sorry

end inequality_proof_l1519_151979


namespace kerosene_mixture_problem_l1519_151902

theorem kerosene_mixture_problem (first_liquid_percentage : ℝ) 
  (first_liquid_parts : ℝ) (second_liquid_parts : ℝ) (mixture_percentage : ℝ) :
  first_liquid_percentage = 25 →
  first_liquid_parts = 6 →
  second_liquid_parts = 4 →
  mixture_percentage = 27 →
  let total_parts := first_liquid_parts + second_liquid_parts
  let second_liquid_percentage := 
    (mixture_percentage * total_parts - first_liquid_percentage * first_liquid_parts) / second_liquid_parts
  second_liquid_percentage = 30 := by
  sorry

end kerosene_mixture_problem_l1519_151902


namespace renovation_project_equation_l1519_151965

/-- Represents the relationship between the number of workers hired in a renovation project --/
theorem renovation_project_equation (x y : ℕ) : 
  (∀ (carpenter_wage mason_wage labor_budget : ℕ), 
    carpenter_wage = 50 ∧ 
    mason_wage = 40 ∧ 
    labor_budget = 2000 → 
    50 * x + 40 * y ≤ 2000) ↔ 
  5 * x + 4 * y ≤ 200 :=
by sorry

end renovation_project_equation_l1519_151965


namespace largest_crate_dimension_l1519_151916

def crate_width : ℝ := 5
def crate_length : ℝ := 8
def pillar_radius : ℝ := 5

theorem largest_crate_dimension (height : ℝ) :
  height ≥ 2 * pillar_radius →
  crate_width ≥ 2 * pillar_radius →
  crate_length ≥ 2 * pillar_radius →
  (∃ (max_dim : ℝ), max_dim = max height (max crate_width crate_length) ∧ max_dim = 2 * pillar_radius) :=
by sorry

end largest_crate_dimension_l1519_151916


namespace zero_neither_positive_nor_negative_l1519_151966

theorem zero_neither_positive_nor_negative :
  ¬(0 > 0) ∧ ¬(0 < 0) :=
by sorry

end zero_neither_positive_nor_negative_l1519_151966


namespace exact_five_green_probability_l1519_151949

def total_marbles : ℕ := 12
def green_marbles : ℕ := 8
def purple_marbles : ℕ := 4
def total_draws : ℕ := 8
def green_draws : ℕ := 5

def prob_green : ℚ := green_marbles / total_marbles
def prob_purple : ℚ := purple_marbles / total_marbles

def binomial_coefficient (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

theorem exact_five_green_probability :
  (binomial_coefficient total_draws green_draws : ℚ) * 
  (prob_green ^ green_draws) * 
  (prob_purple ^ (total_draws - green_draws)) =
  56 * (2/3)^5 * (1/3)^3 := by sorry

end exact_five_green_probability_l1519_151949


namespace diff_same_digits_div_by_9_no_solution_to_puzzle_l1519_151934

-- Define a function to check if two numbers have the same digits
def haveSameDigits (a b : ℕ) : Prop := sorry

-- Define the property that the difference of numbers with the same digits is divisible by 9
theorem diff_same_digits_div_by_9 (a b : ℕ) (h : haveSameDigits a b) : 
  9 ∣ (a - b) := sorry

-- State the main theorem
theorem no_solution_to_puzzle : 
  ¬ ∃ (a b : ℕ), haveSameDigits a b ∧ a - b = 2018 * 2019 := by
  sorry

end diff_same_digits_div_by_9_no_solution_to_puzzle_l1519_151934


namespace reseating_women_circular_l1519_151927

-- Define the recurrence relation for reseating women
def R : ℕ → ℕ
  | 0 => 0  -- We define R(0) as 0 for completeness
  | 1 => 1
  | 2 => 2
  | (n + 3) => R (n + 2) + R (n + 1)

-- Theorem statement
theorem reseating_women_circular (n : ℕ) : R 15 = 987 := by
  sorry

-- You can also add additional lemmas to help prove the main theorem
lemma R_recurrence (n : ℕ) : n ≥ 3 → R n = R (n - 1) + R (n - 2) := by
  sorry

end reseating_women_circular_l1519_151927


namespace expression_value_l1519_151917

theorem expression_value (x y : ℤ) (hx : x = -2) (hy : y = 4) :
  5 * x - 2 * y + 7 = -11 := by
  sorry

end expression_value_l1519_151917


namespace octagon_area_ratio_octagon_area_ratio_proof_l1519_151986

/-- The ratio of the area of a regular octagon circumscribed about a circle
    to the area of a regular octagon inscribed in the same circle is 2. -/
theorem octagon_area_ratio : ℝ → ℝ → Prop :=
  fun (area_circumscribed area_inscribed : ℝ) =>
    area_circumscribed / area_inscribed = 2

/-- Given a circle with radius r, the area of its circumscribed regular octagon
    is twice the area of its inscribed regular octagon. -/
theorem octagon_area_ratio_proof (r : ℝ) (r_pos : r > 0) :
  ∃ (area_circumscribed area_inscribed : ℝ),
    area_circumscribed > 0 ∧
    area_inscribed > 0 ∧
    octagon_area_ratio area_circumscribed area_inscribed :=
by
  sorry

end octagon_area_ratio_octagon_area_ratio_proof_l1519_151986


namespace john_horses_count_l1519_151971

/-- Represents the number of horses John has -/
def num_horses : ℕ := 25

/-- Represents the number of feedings per day for each horse -/
def feedings_per_day : ℕ := 2

/-- Represents the amount of food in pounds per feeding -/
def food_per_feeding : ℕ := 20

/-- Represents the weight of a bag of food in pounds -/
def bag_weight : ℕ := 1000

/-- Represents the number of days -/
def num_days : ℕ := 60

/-- Represents the number of bags needed for the given number of days -/
def num_bags : ℕ := 60

theorem john_horses_count :
  num_horses * feedings_per_day * food_per_feeding * num_days = num_bags * bag_weight := by
  sorry


end john_horses_count_l1519_151971


namespace two_color_theorem_l1519_151987

/-- A type representing the two colors used for coloring regions -/
inductive Color
| Blue
| Red

/-- A type representing a circle in the plane -/
structure Circle where
  -- We don't need to define the internal structure of a circle for this problem

/-- A type representing a region in the plane -/
structure Region where
  -- We don't need to define the internal structure of a region for this problem

/-- A function type for coloring regions -/
def ColoringFunction := Region → Color

/-- Predicate to check if two regions are adjacent (separated by a circle arc) -/
def are_adjacent (r1 r2 : Region) : Prop := sorry

/-- Theorem stating the existence of a valid two-color coloring for n circles -/
theorem two_color_theorem (n : ℕ) (h : n ≥ 1) :
  ∃ (circles : Finset Circle) (regions : Finset Region) (coloring : ColoringFunction),
    (circles.card = n) ∧
    (∀ r1 r2 : Region, r1 ∈ regions → r2 ∈ regions → are_adjacent r1 r2 →
      coloring r1 ≠ coloring r2) :=
sorry

end two_color_theorem_l1519_151987


namespace simple_random_sampling_fairness_l1519_151991

/-- Represents the probability of being selected in a simple random sample -/
def SimpleSampleProb (n : ℕ) : ℚ := 1 / n

/-- Represents a group of students -/
structure StudentGroup where
  total : ℕ
  selected : ℕ
  toEliminate : ℕ

/-- Defines fairness based on equal probability of selection -/
def isFair (g : StudentGroup) : Prop :=
  ∀ (i j : ℕ), i ≤ g.selected → j ≤ g.selected →
    SimpleSampleProb g.selected = SimpleSampleProb g.selected

theorem simple_random_sampling_fairness 
  (students : StudentGroup) 
  (h1 : students.total = 102) 
  (h2 : students.selected = 20) 
  (h3 : students.toEliminate = 2) : 
  isFair students :=
sorry

end simple_random_sampling_fairness_l1519_151991


namespace triangle_angle_proof_l1519_151921

/-- Given a triangle with two known angles of 45° and 70°, prove that the third angle is 65° and the largest angle is 70°. -/
theorem triangle_angle_proof (a b c : ℝ) : 
  a = 45 → b = 70 → a + b + c = 180 → 
  c = 65 ∧ max a (max b c) = 70 := by
  sorry

end triangle_angle_proof_l1519_151921


namespace square_area_tripled_l1519_151976

theorem square_area_tripled (a : ℝ) (h : a > 0) :
  (a * Real.sqrt 3) ^ 2 = 3 * a ^ 2 := by
  sorry

end square_area_tripled_l1519_151976


namespace sweep_time_is_three_l1519_151946

/-- The time in minutes it takes to sweep one room -/
def sweep_time : ℝ := sorry

/-- The time in minutes it takes to wash one dish -/
def dish_time : ℝ := 2

/-- The time in minutes it takes to do one load of laundry -/
def laundry_time : ℝ := 9

/-- The number of rooms Anna sweeps -/
def anna_rooms : ℕ := 10

/-- The number of loads of laundry Billy does -/
def billy_laundry : ℕ := 2

/-- The number of dishes Billy washes -/
def billy_dishes : ℕ := 6

theorem sweep_time_is_three :
  sweep_time = 3 ∧
  anna_rooms * sweep_time = billy_laundry * laundry_time + billy_dishes * dish_time :=
by sorry

end sweep_time_is_three_l1519_151946


namespace f_greatest_lower_bound_l1519_151942

noncomputable def f (x : ℝ) : ℝ := (x - 3) * Real.exp x

theorem f_greatest_lower_bound :
  ∃ (k : ℝ), k = -Real.exp 2 ∧
  (∀ x > 2, f x > k) ∧
  (∀ ε > 0, ∃ x > 2, f x < k + ε) :=
sorry

end f_greatest_lower_bound_l1519_151942


namespace negation_equivalence_l1519_151989

theorem negation_equivalence (x y : ℝ) : 
  ¬(x^2 + y^2 = 0 → x = 0 ∧ y = 0) ↔ (x^2 + y^2 = 0 → x ≠ 0 ∨ y ≠ 0) := by
  sorry

end negation_equivalence_l1519_151989


namespace rod_length_l1519_151906

theorem rod_length (pieces : ℕ) (piece_length : ℝ) (h1 : pieces = 50) (h2 : piece_length = 0.85) :
  pieces * piece_length = 42.5 := by
  sorry

end rod_length_l1519_151906


namespace product_of_odds_over_sum_of_squares_l1519_151900

theorem product_of_odds_over_sum_of_squares : 
  (1 * 3 * 5 * 7) / (1^2 + 2^2 + 3^2 + 4^2) = 7 / 2 := by
  sorry

end product_of_odds_over_sum_of_squares_l1519_151900


namespace todd_sum_equals_l1519_151920

/-- Represents the counting game with Todd, Tadd, and Tucker -/
structure CountingGame where
  max_count : ℕ
  todd_turn_length : ℕ → ℕ
  todd_start_positions : ℕ → ℕ

/-- Calculates the sum of numbers Todd declares in the game -/
def todd_sum (game : CountingGame) : ℕ :=
  sorry

/-- The specific game instance described in the problem -/
def specific_game : CountingGame :=
  { max_count := 5000
  , todd_turn_length := λ n => n + 1
  , todd_start_positions := λ n => sorry }

/-- Theorem stating the sum of Todd's numbers equals a specific value -/
theorem todd_sum_equals (result : ℕ) : todd_sum specific_game = result :=
  sorry

end todd_sum_equals_l1519_151920


namespace chef_nut_purchase_l1519_151969

/-- The weight of almonds bought by the chef in kilograms -/
def almond_weight : ℝ := 0.14

/-- The weight of pecans bought by the chef in kilograms -/
def pecan_weight : ℝ := 0.38

/-- The total weight of nuts bought by the chef in kilograms -/
def total_weight : ℝ := almond_weight + pecan_weight

/-- Theorem stating that the total weight of nuts bought by the chef is 0.52 kilograms -/
theorem chef_nut_purchase : total_weight = 0.52 := by
  sorry

end chef_nut_purchase_l1519_151969


namespace complex_fraction_equality_l1519_151944

theorem complex_fraction_equality (a b : ℂ) 
  (h1 : (a + b) / (a - b) - (a - b) / (a + b) = 2)
  (h2 : a^2 + b^2 ≠ 0) :
  (a^4 + b^4) / (a^4 - b^4) - (a^4 - b^4) / (a^4 + b^4) = ((a^2 + b^2) - 3) / 3 :=
by sorry

end complex_fraction_equality_l1519_151944


namespace max_value_of_sum_products_l1519_151930

theorem max_value_of_sum_products (a b c d : ℝ) :
  a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 →
  a + b + c + d = 150 →
  a * b + b * c + c * d ≤ 5625 := by
sorry

end max_value_of_sum_products_l1519_151930


namespace geometric_sequence_first_term_l1519_151954

/-- Given a geometric sequence where the fourth term is 54 and the fifth term is 162,
    prove that the first term of the sequence is 2. -/
theorem geometric_sequence_first_term
  (a : ℝ)  -- First term of the sequence
  (r : ℝ)  -- Common ratio of the sequence
  (h1 : a * r^3 = 54)  -- Fourth term is 54
  (h2 : a * r^4 = 162)  -- Fifth term is 162
  : a = 2 := by
sorry

end geometric_sequence_first_term_l1519_151954


namespace pi_is_real_l1519_151937

-- Define π as a real number representing the ratio of a circle's circumference to its diameter
noncomputable def π : ℝ := Real.pi

-- Theorem stating that π is a real number
theorem pi_is_real : π ∈ Set.univ := by sorry

end pi_is_real_l1519_151937


namespace line_parallel_perpendicular_l1519_151941

-- Define the types for lines and planes
variable (L : Type*) [LinearOrderedField L]
variable (P : Type*) [LinearOrderedField P]

-- Define the parallel and perpendicular relations
variable (parallel : L → L → Prop)
variable (perp : L → P → Prop)

-- State the theorem
theorem line_parallel_perpendicular
  (m n : L) (α : P)
  (h1 : m ≠ n)
  (h2 : parallel m n)
  (h3 : perp m α) :
  perp n α :=
sorry

end line_parallel_perpendicular_l1519_151941


namespace socks_purchase_problem_l1519_151956

theorem socks_purchase_problem :
  ∃ (a b c : ℕ), 
    a + b + c = 15 ∧
    2 * a + 3 * b + 5 * c = 40 ∧
    a ≥ 1 ∧ b ≥ 1 ∧ c ≥ 1 ∧
    (a = 7 ∨ a = 9 ∨ a = 11) :=
by sorry

end socks_purchase_problem_l1519_151956


namespace smaller_pyramid_volume_l1519_151959

/-- The volume of a smaller pyramid cut from a larger right square pyramid -/
theorem smaller_pyramid_volume
  (base_edge : ℝ)
  (total_height : ℝ)
  (cut_height : ℝ)
  (h_base : base_edge = 12)
  (h_height : total_height = 18)
  (h_cut : cut_height = 6) :
  (1/3 : ℝ) * (cut_height / total_height)^2 * base_edge^2 * cut_height = 32 := by
sorry

end smaller_pyramid_volume_l1519_151959


namespace carrie_punch_ice_amount_l1519_151988

/-- Represents the ingredients and result of Carrie's punch recipe --/
structure PunchRecipe where
  mountain_dew_cans : Nat
  mountain_dew_oz_per_can : Nat
  fruit_juice_oz : Nat
  servings : Nat
  oz_per_serving : Nat

/-- Calculates the amount of ice added to the punch --/
def ice_added (recipe : PunchRecipe) : Nat :=
  recipe.servings * recipe.oz_per_serving - 
  (recipe.mountain_dew_cans * recipe.mountain_dew_oz_per_can + recipe.fruit_juice_oz)

/-- Theorem stating that Carrie added 28 oz of ice to her punch --/
theorem carrie_punch_ice_amount : 
  ice_added { mountain_dew_cans := 6
            , mountain_dew_oz_per_can := 12
            , fruit_juice_oz := 40
            , servings := 14
            , oz_per_serving := 10 } = 28 := by
  sorry

end carrie_punch_ice_amount_l1519_151988


namespace max_a_inequality_max_a_is_five_l1519_151938

theorem max_a_inequality (a : ℝ) : 
  (∀ x : ℝ, x^2 + |2*x - 6| ≥ a) → a ≤ 5 :=
by sorry

theorem max_a_is_five : 
  ∃ a : ℝ, (∀ x : ℝ, x^2 + |2*x - 6| ≥ a) ∧ a = 5 :=
by sorry

end max_a_inequality_max_a_is_five_l1519_151938


namespace cube_root_equation_solution_l1519_151903

theorem cube_root_equation_solution :
  ∃ y : ℝ, (30 * y + (30 * y + 24) ^ (1/3)) ^ (1/3) = 24 ∧ y = 460 :=
by
  sorry

end cube_root_equation_solution_l1519_151903


namespace total_albums_is_2835_l1519_151980

/-- The total number of albums owned by six people given certain relationships between their album counts. -/
def total_albums (adele_albums : ℕ) : ℕ :=
  let bridget_albums := adele_albums - 15
  let katrina_albums := 6 * bridget_albums
  let miriam_albums := 7 * katrina_albums
  let carlos_albums := 3 * miriam_albums
  let diane_albums := 2 * katrina_albums
  adele_albums + bridget_albums + katrina_albums + miriam_albums + carlos_albums + diane_albums

/-- Theorem stating that the total number of albums is 2835 given the conditions in the problem. -/
theorem total_albums_is_2835 : total_albums 30 = 2835 := by
  sorry

end total_albums_is_2835_l1519_151980


namespace shortest_player_height_l1519_151919

/-- Given the heights of four players, prove the height of the shortest player. -/
theorem shortest_player_height (T S P Q : ℝ)
  (h1 : T = 77.75)
  (h2 : T = S + 9.5)
  (h3 : P = S + 5)
  (h4 : Q = P - 3) :
  S = 68.25 := by
  sorry

end shortest_player_height_l1519_151919


namespace appropriate_word_count_l1519_151968

-- Define the presentation parameters
def min_duration : ℕ := 40
def max_duration : ℕ := 50
def speech_rate : ℕ := 160

-- Define the range of appropriate word counts
def min_words : ℕ := min_duration * speech_rate
def max_words : ℕ := max_duration * speech_rate

-- Theorem statement
theorem appropriate_word_count (word_count : ℕ) :
  (min_words ≤ word_count ∧ word_count ≤ max_words) ↔
  (word_count ≥ 6400 ∧ word_count ≤ 8000) :=
by sorry

end appropriate_word_count_l1519_151968


namespace odd_number_difference_difference_is_98_l1519_151990

theorem odd_number_difference : ℕ → Prop :=
  fun n => ∃ (a b : ℕ), 
    (a ≤ 100 ∧ b ≤ 100) ∧  -- Numbers are in the range 1 to 100
    (Odd a ∧ Odd b) ∧      -- Both numbers are odd
    (∀ k, k ≤ 100 → Odd k → a ≤ k ∧ k ≤ b) ∧  -- a is smallest, b is largest odd number
    b - a = n              -- Their difference is n

theorem difference_is_98 : odd_number_difference 98 := by
  sorry

end odd_number_difference_difference_is_98_l1519_151990


namespace sum_irrational_implies_component_irrational_l1519_151901

theorem sum_irrational_implies_component_irrational (a b c : ℝ) : 
  ¬ (∃ (q : ℚ), (a + b + c : ℝ) = q) → 
  ¬ (∃ (q₁ q₂ q₃ : ℚ), (a = q₁ ∧ b = q₂ ∧ c = q₃)) :=
by sorry

end sum_irrational_implies_component_irrational_l1519_151901


namespace gerbil_revenue_calculation_l1519_151958

/-- Calculates the total revenue from gerbil sales given the initial stock, percentage sold, original price, and discount rate. -/
def gerbil_revenue (initial_stock : ℕ) (percent_sold : ℚ) (original_price : ℚ) (discount_rate : ℚ) : ℚ :=
  let sold := ⌊initial_stock * percent_sold⌋
  let remaining := initial_stock - sold
  let discounted_price := original_price * (1 - discount_rate)
  sold * original_price + remaining * discounted_price

/-- Theorem stating that the total revenue from gerbil sales is $4696.80 given the specified conditions. -/
theorem gerbil_revenue_calculation :
  gerbil_revenue 450 (35/100) 12 (20/100) = 4696.80 := by
  sorry

end gerbil_revenue_calculation_l1519_151958


namespace price_change_theorem_l1519_151922

theorem price_change_theorem (initial_price : ℝ) 
  (jan_increase : ℝ) (feb_decrease : ℝ) (mar_increase : ℝ) (apr_decrease : ℝ) : 
  initial_price = 200 ∧ 
  jan_increase = 0.3 ∧ 
  feb_decrease = 0.1 ∧ 
  mar_increase = 0.2 ∧
  initial_price * (1 + jan_increase) * (1 - feb_decrease) * (1 + mar_increase) * (1 - apr_decrease) = initial_price →
  apr_decrease = 0.29 := by
  sorry

end price_change_theorem_l1519_151922


namespace chicken_wings_distribution_l1519_151996

theorem chicken_wings_distribution (num_friends : ℕ) (pre_cooked : ℕ) (additional_cooked : ℕ) :
  num_friends = 3 →
  pre_cooked = 8 →
  additional_cooked = 10 →
  (pre_cooked + additional_cooked) / num_friends = 6 := by
  sorry

end chicken_wings_distribution_l1519_151996


namespace parallel_vectors_fraction_l1519_151993

theorem parallel_vectors_fraction (x : ℝ) :
  let a : ℝ × ℝ := (Real.sin x, (3 : ℝ) / 2)
  let b : ℝ × ℝ := (Real.cos x, -1)
  (∃ (k : ℝ), a.1 = k * b.1 ∧ a.2 = k * b.2) →
  (2 * Real.sin x - Real.cos x) / (4 * Real.sin x + 3 * Real.cos x) = 4 / 3 :=
by sorry

end parallel_vectors_fraction_l1519_151993


namespace percentage_equation_l1519_151974

theorem percentage_equation (x : ℝ) : (0.3 / 100) * x = 0.15 → x = 50 := by
  sorry

end percentage_equation_l1519_151974


namespace A_equals_B_l1519_151955

def A : Set ℤ := {x | ∃ a b : ℤ, x = 12 * a + 8 * b}
def B : Set ℤ := {y | ∃ c d : ℤ, y = 20 * c + 16 * d}

theorem A_equals_B : A = B := by sorry

end A_equals_B_l1519_151955


namespace exam_comparison_l1519_151950

/-- Proves that Lyssa has 3 fewer correct answers than Precious in an exam with 75 items,
    where Lyssa answers 20% incorrectly and Precious makes 12 mistakes. -/
theorem exam_comparison (total_items : ℕ) (lyssa_incorrect_percent : ℚ) (precious_mistakes : ℕ)
  (h1 : total_items = 75)
  (h2 : lyssa_incorrect_percent = 1/5)
  (h3 : precious_mistakes = 12) :
  (total_items - (lyssa_incorrect_percent * total_items).floor) = 
  (total_items - precious_mistakes) - 3 :=
by sorry

end exam_comparison_l1519_151950


namespace total_accidents_in_four_minutes_l1519_151915

/-- Represents the number of seconds in 4 minutes -/
def total_seconds : ℕ := 4 * 60

/-- Represents the frequency of car collisions in seconds -/
def car_collision_frequency : ℕ := 3

/-- Represents the frequency of big crashes in seconds -/
def big_crash_frequency : ℕ := 7

/-- Represents the frequency of multi-vehicle pile-ups in seconds -/
def pile_up_frequency : ℕ := 15

/-- Represents the frequency of massive accidents in seconds -/
def massive_accident_frequency : ℕ := 25

/-- Calculates the number of accidents of a given type -/
def accidents_of_type (frequency : ℕ) : ℕ :=
  total_seconds / frequency

/-- Theorem stating the total number of accidents in 4 minutes -/
theorem total_accidents_in_four_minutes :
  accidents_of_type car_collision_frequency +
  accidents_of_type big_crash_frequency +
  accidents_of_type pile_up_frequency +
  accidents_of_type massive_accident_frequency = 139 := by
  sorry


end total_accidents_in_four_minutes_l1519_151915


namespace rooftop_steps_l1519_151924

/-- The total number of stair steps to reach the rooftop -/
def total_steps (climbed : ℕ) (remaining : ℕ) : ℕ := climbed + remaining

/-- Theorem stating that the total number of steps is 96 -/
theorem rooftop_steps : total_steps 74 22 = 96 := by
  sorry

end rooftop_steps_l1519_151924


namespace ballsInBoxes_correct_l1519_151981

/-- The number of ways to place four different balls into four numbered boxes with one empty box -/
def ballsInBoxes : ℕ :=
  -- Define the number of ways to place the balls
  -- We don't implement the actual calculation here
  144

/-- Theorem stating that the number of ways to place the balls is correct -/
theorem ballsInBoxes_correct : ballsInBoxes = 144 := by
  -- The proof would go here
  sorry

end ballsInBoxes_correct_l1519_151981


namespace parabola_vertex_distance_l1519_151977

/-- The parabola equation -/
def parabola (x c : ℝ) : ℝ := x^2 - 6*x + c - 2

/-- The vertex of the parabola -/
def vertex (c : ℝ) : ℝ × ℝ := (3, c - 11)

/-- The distance from the vertex to the x-axis -/
def distance_to_x_axis (c : ℝ) : ℝ := |c - 11|

theorem parabola_vertex_distance (c : ℝ) :
  distance_to_x_axis c = 3 → c = 8 ∨ c = 14 := by
  sorry

end parabola_vertex_distance_l1519_151977


namespace function_inequality_l1519_151923

theorem function_inequality (f : ℝ → ℝ) (h : Differentiable ℝ f) 
    (h1 : ∀ x, (x - 1) * (deriv f x) ≥ 0) : 
  f 0 + f 2 ≥ 2 * f 1 := by
  sorry

end function_inequality_l1519_151923


namespace arithmetic_sequence_problem_l1519_151904

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_a1 : a 1 = 2) 
  (h_sum : a 3 + a 5 = 10) : 
  a 7 = 8 := by
sorry

end arithmetic_sequence_problem_l1519_151904


namespace salary_percentage_decrease_l1519_151967

theorem salary_percentage_decrease 
  (x : ℝ) -- Original salary
  (h1 : x * 1.15 = 575) -- 15% increase condition
  (h2 : x * (1 - y / 100) = 560) -- y% decrease condition
  : y = 12 := by
  sorry

end salary_percentage_decrease_l1519_151967


namespace problem_solution_l1519_151947

theorem problem_solution (a : ℝ) (h1 : a > 0) : 
  (fun x => x^2 + 4) ((fun x => x^2 - 2) a) = 12 → 
  a = Real.sqrt (2 * (Real.sqrt 2 + 1)) := by
sorry

end problem_solution_l1519_151947


namespace marble_fraction_after_tripling_l1519_151997

theorem marble_fraction_after_tripling (total : ℚ) (h1 : total > 0) : 
  let blue := (4/7) * total
  let green := total - blue
  let new_green := 3 * green
  let new_total := blue + new_green
  new_green / new_total = 9/13 := by
sorry

end marble_fraction_after_tripling_l1519_151997


namespace debate_team_selection_l1519_151953

def total_students : ℕ := 9
def students_to_select : ℕ := 4
def specific_students : ℕ := 2

def select_with_condition (n k m : ℕ) : ℕ :=
  Nat.choose n k - Nat.choose (n - m) k

theorem debate_team_selection :
  select_with_condition total_students students_to_select specific_students = 91 := by
  sorry

end debate_team_selection_l1519_151953


namespace binary_1111_equals_15_l1519_151985

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binaryToDecimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of the number 15 -/
def binaryFifteen : List Bool := [true, true, true, true]

/-- Theorem stating that the binary representation "1111" is equal to 15 in decimal -/
theorem binary_1111_equals_15 : binaryToDecimal binaryFifteen = 15 := by
  sorry

end binary_1111_equals_15_l1519_151985


namespace max_nickels_l1519_151929

-- Define the coin types
inductive Coin
| Penny
| Nickel
| Dime
| Quarter

-- Define the wallet as a function from Coin to ℕ (number of each coin type)
def Wallet := Coin → ℕ

-- Define the value of each coin in cents
def coinValue : Coin → ℕ
| Coin.Penny => 1
| Coin.Nickel => 5
| Coin.Dime => 10
| Coin.Quarter => 25

-- Function to calculate the total value of coins in the wallet
def totalValue (w : Wallet) : ℕ :=
  (w Coin.Penny) * (coinValue Coin.Penny) +
  (w Coin.Nickel) * (coinValue Coin.Nickel) +
  (w Coin.Dime) * (coinValue Coin.Dime) +
  (w Coin.Quarter) * (coinValue Coin.Quarter)

-- Function to count the total number of coins in the wallet
def coinCount (w : Wallet) : ℕ :=
  (w Coin.Penny) + (w Coin.Nickel) + (w Coin.Dime) + (w Coin.Quarter)

-- Theorem statement
theorem max_nickels (w : Wallet) :
  (totalValue w = 15 * coinCount w) →
  (totalValue w + coinValue Coin.Dime = 16 * (coinCount w + 1)) →
  (w Coin.Nickel = 2) := by
  sorry


end max_nickels_l1519_151929


namespace sum_of_first_n_integers_second_difference_constant_sum_formula_l1519_151962

def f (n : ℕ) : ℕ := (List.range n).sum + n

theorem sum_of_first_n_integers (n : ℕ) : 
  f n = n * (n + 1) / 2 :=
by sorry

theorem second_difference_constant (n : ℕ) : 
  f (n + 2) - 2 * f (n + 1) + f n = 1 :=
by sorry

theorem sum_formula (n : ℕ) : 
  (List.range n).sum + n = n * (n + 1) / 2 :=
by
  have h1 := sum_of_first_n_integers n
  have h2 := second_difference_constant n
  sorry

end sum_of_first_n_integers_second_difference_constant_sum_formula_l1519_151962


namespace newspaper_sale_percentage_l1519_151925

/-- Represents the problem of calculating the percentage of newspapers John sells. -/
theorem newspaper_sale_percentage
  (total_newspapers : ℕ)
  (selling_price : ℚ)
  (discount_percentage : ℚ)
  (profit : ℚ)
  (h1 : total_newspapers = 500)
  (h2 : selling_price = 2)
  (h3 : discount_percentage = 75 / 100)
  (h4 : profit = 550)
  : (selling_price * (1 - discount_percentage) * total_newspapers + profit) / (selling_price * total_newspapers) = 4 / 5 :=
sorry

end newspaper_sale_percentage_l1519_151925


namespace linear_equation_solution_l1519_151931

theorem linear_equation_solution : 
  ∀ x : ℝ, (x + 1) / 3 = 0 ↔ x = -1 := by
  sorry

end linear_equation_solution_l1519_151931


namespace first_thrilling_thursday_l1519_151945

/-- Represents a date with a day and a month -/
structure Date where
  day : Nat
  month : Nat

/-- Represents a day of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Function to determine if a given date is a Thursday -/
def isThursday (d : Date) : Bool := sorry

/-- Function to determine if a given date is a Thrilling Thursday -/
def isThrillingThursday (d : Date) : Bool := sorry

/-- The number of days in November -/
def novemberDays : Nat := 30

/-- The start date of the school -/
def schoolStartDate : Date := ⟨2, 11⟩

/-- Theorem stating that the first Thrilling Thursday after school starts is November 30 -/
theorem first_thrilling_thursday :
  let firstThrillingThursday := Date.mk 30 11
  isThursday schoolStartDate ∧
  isThrillingThursday firstThrillingThursday ∧
  (∀ d : Date, schoolStartDate.day ≤ d.day ∧ d.day < firstThrillingThursday.day →
    ¬isThrillingThursday d) := by
  sorry

end first_thrilling_thursday_l1519_151945


namespace total_lines_eq_88_l1519_151913

/-- The number of sides in a triangle -/
def triangle_sides : ℕ := 3

/-- The number of sides in a square -/
def square_sides : ℕ := 4

/-- The number of sides in a pentagon -/
def pentagon_sides : ℕ := 5

/-- The number of triangles drawn -/
def num_triangles : ℕ := 12

/-- The number of squares drawn -/
def num_squares : ℕ := 8

/-- The number of pentagons drawn -/
def num_pentagons : ℕ := 4

/-- The total number of lines drawn -/
def total_lines : ℕ := num_triangles * triangle_sides + num_squares * square_sides + num_pentagons * pentagon_sides

theorem total_lines_eq_88 : total_lines = 88 := by
  sorry

end total_lines_eq_88_l1519_151913


namespace problem_solution_l1519_151939

def star (a b : ℕ) : ℕ := a^b + a*b

theorem problem_solution (a b : ℕ) (ha : a ≥ 2) (hb : b ≥ 2) (h : star a b = 40) : a + b = 7 := by
  sorry

end problem_solution_l1519_151939


namespace work_completion_equality_second_group_size_correct_l1519_151960

/-- The number of men in the first group -/
def first_group : ℕ := 12

/-- The number of days the first group takes to complete the work -/
def first_days : ℕ := 30

/-- The number of days the second group takes to complete the work -/
def second_days : ℕ := 36

/-- The number of men in the second group -/
def second_group : ℕ := 10

theorem work_completion_equality :
  first_group * first_days = second_group * second_days :=
by sorry

/-- Proves that the number of men in the second group is correct -/
theorem second_group_size_correct :
  second_group = (first_group * first_days) / second_days :=
by sorry

end work_completion_equality_second_group_size_correct_l1519_151960


namespace donna_bananas_l1519_151908

def total_bananas : ℕ := 200
def lydia_bananas : ℕ := 60

theorem donna_bananas : 
  ∀ (dawn_bananas : ℕ) (donna_bananas : ℕ),
  dawn_bananas = lydia_bananas + 40 →
  total_bananas = dawn_bananas + lydia_bananas + donna_bananas →
  donna_bananas = 40 := by
sorry

end donna_bananas_l1519_151908


namespace difference_of_greatest_values_l1519_151940

def is_valid_three_digit_number (x : ℕ) : Prop :=
  100 ≤ x ∧ x < 1000

def hundreds_digit (x : ℕ) : ℕ := (x / 100) % 10
def tens_digit (x : ℕ) : ℕ := (x / 10) % 10
def units_digit (x : ℕ) : ℕ := x % 10

def satisfies_conditions (x : ℕ) : Prop :=
  let a := hundreds_digit x
  let b := tens_digit x
  let c := units_digit x
  is_valid_three_digit_number x ∧ 2 * a = b ∧ b = 4 * c ∧ a > 0

theorem difference_of_greatest_values : 
  ∃ x₁ x₂ : ℕ, satisfies_conditions x₁ ∧ satisfies_conditions x₂ ∧
  (∀ x : ℕ, satisfies_conditions x → x ≤ x₁) ∧
  (∀ x : ℕ, satisfies_conditions x → x ≠ x₁ → x ≤ x₂) ∧
  x₁ - x₂ = 241 :=
sorry

end difference_of_greatest_values_l1519_151940


namespace girls_percentage_increase_l1519_151964

theorem girls_percentage_increase (initial_boys : ℕ) (final_total : ℕ) : 
  initial_boys = 15 →
  final_total = 51 →
  ∃ (initial_girls : ℕ),
    initial_girls = initial_boys + (initial_boys * 20 / 100) ∧
    final_total = initial_boys + 2 * initial_girls :=
by sorry

end girls_percentage_increase_l1519_151964


namespace min_max_inequality_l1519_151957

theorem min_max_inequality {a b x₁ x₂ x₃ x₄ : ℝ} 
  (ha : 0 < a) (hab : a < b) 
  (hx₁ : a ≤ x₁ ∧ x₁ ≤ b) (hx₂ : a ≤ x₂ ∧ x₂ ≤ b) 
  (hx₃ : a ≤ x₃ ∧ x₃ ≤ b) (hx₄ : a ≤ x₄ ∧ x₄ ≤ b) :
  1 ≤ (x₁^2/x₂ + x₂^2/x₃ + x₃^2/x₄ + x₄^2/x₁) / (x₁ + x₂ + x₃ + x₄) ∧
  (x₁^2/x₂ + x₂^2/x₃ + x₃^2/x₄ + x₄^2/x₁) / (x₁ + x₂ + x₃ + x₄) ≤ b/a + a/b - 1 :=
by sorry

end min_max_inequality_l1519_151957


namespace point_coordinates_wrt_origin_l1519_151978

/-- 
Given a point P with coordinates (-5, 3) in the Cartesian coordinate system,
prove that its coordinates with respect to the origin are (-5, 3).
-/
theorem point_coordinates_wrt_origin :
  let P : ℝ × ℝ := (-5, 3)
  P = (-5, 3) := by sorry

end point_coordinates_wrt_origin_l1519_151978


namespace complex_modulus_sqrt5_l1519_151932

theorem complex_modulus_sqrt5 (a b : ℝ) (i : ℂ) (h : i * i = -1) :
  (a - 2 * i) * i = b - i →
  Complex.abs (a + b * i) = Real.sqrt 5 := by
  sorry

end complex_modulus_sqrt5_l1519_151932


namespace symmetric_point_coordinates_l1519_151907

/-- A point in a 2D plane represented by its x and y coordinates. -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines symmetry with respect to the origin for two points. -/
def symmetricToOrigin (a b : Point) : Prop :=
  b.x = -a.x ∧ b.y = -a.y

/-- Theorem stating that if point A(5, -1) is symmetric to point B with respect to the origin,
    then the coordinates of point B are (-5, 1). -/
theorem symmetric_point_coordinates :
  let a : Point := ⟨5, -1⟩
  let b : Point := ⟨-5, 1⟩
  symmetricToOrigin a b :=
by
  sorry

end symmetric_point_coordinates_l1519_151907


namespace dice_roll_probability_l1519_151914

/-- The probability of rolling a number other than 1 on a standard die -/
def prob_not_one : ℚ := 5 / 6

/-- The number of dice rolled -/
def num_dice : ℕ := 4

/-- The probability that (a-1)(b-1)(c-1)(d-1) ≠ 0 when four standard dice are tossed -/
def prob_product_nonzero : ℚ := prob_not_one ^ num_dice

theorem dice_roll_probability :
  prob_product_nonzero = 625 / 1296 := by sorry

end dice_roll_probability_l1519_151914


namespace vacation_towel_problem_l1519_151943

theorem vacation_towel_problem (families : ℕ) (days : ℕ) (towels_per_person_per_day : ℕ) 
  (towels_per_load : ℕ) (total_loads : ℕ) :
  families = 3 →
  days = 7 →
  towels_per_person_per_day = 1 →
  towels_per_load = 14 →
  total_loads = 6 →
  (total_loads * towels_per_load) / (days * families) = 4 :=
by
  sorry


end vacation_towel_problem_l1519_151943


namespace floor_of_6_8_l1519_151982

theorem floor_of_6_8 : ⌊(6.8 : ℝ)⌋ = 6 := by sorry

end floor_of_6_8_l1519_151982


namespace recycling_program_earnings_l1519_151999

/-- Calculates the total money earned by Katrina and her friends in the recycling program -/
def total_money_earned (initial_signup_bonus : ℕ) (referral_bonus : ℕ) (friends_day1 : ℕ) (friends_week : ℕ) : ℕ :=
  let katrina_earnings := initial_signup_bonus + referral_bonus * (friends_day1 + friends_week)
  let friends_earnings := referral_bonus * (friends_day1 + friends_week)
  katrina_earnings + friends_earnings

/-- Proves that the total money earned by Katrina and her friends is $125.00 -/
theorem recycling_program_earnings :
  total_money_earned 5 5 5 7 = 125 :=
by
  sorry

end recycling_program_earnings_l1519_151999


namespace unique_four_digit_number_l1519_151963

/-- 
A four-digit number is a natural number between 1000 and 9999, inclusive.
-/
def FourDigitNumber (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

/-- 
Given a four-digit number, this function returns the three-digit number 
obtained by removing its leftmost digit.
-/
def RemoveLeftmostDigit (n : ℕ) : ℕ := n % 1000

/-- 
Theorem: 3500 is the only four-digit number N such that the three-digit number 
obtained by removing its leftmost digit is one-seventh of N.
-/
theorem unique_four_digit_number : 
  ∀ N : ℕ, FourDigitNumber N → 
    (RemoveLeftmostDigit N = N / 7 ↔ N = 3500) :=
by sorry

end unique_four_digit_number_l1519_151963


namespace range_of_a_l1519_151926

/-- Given that for any x ≥ 1, ln x - a(1 - 1/x) ≥ 0, prove that a ≤ 1 -/
theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ≥ 1 → Real.log x - a * (1 - 1/x) ≥ 0) → 
  a ≤ 1 :=
by sorry

end range_of_a_l1519_151926


namespace f_solutions_l1519_151994

def f (x : ℝ) : ℝ := x^2 + 2*x + 4

theorem f_solutions : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ = 12 ∧ f x₂ = 12 ∧ 
  (∀ x : ℝ, f x = 12 → x = x₁ ∨ x = x₂) :=
sorry

end f_solutions_l1519_151994


namespace geometric_sequence_logarithm_l1519_151933

/-- Given a geometric sequence {a_n} with common ratio -√2, 
    prove that ln(a_{2017})^2 - ln(a_{2016})^2 = ln(2) -/
theorem geometric_sequence_logarithm (a : ℕ → ℝ) 
  (h : ∀ n, a (n + 1) = a n * (-Real.sqrt 2)) :
  (Real.log (a 2017))^2 - (Real.log (a 2016))^2 = Real.log 2 := by
  sorry

end geometric_sequence_logarithm_l1519_151933


namespace f_increasing_and_odd_l1519_151935

def f (x : ℝ) : ℝ := x + x^3

theorem f_increasing_and_odd :
  (∀ x y, x < y → f x < f y) ∧ 
  (∀ x, f (-x) = -f x) :=
sorry

end f_increasing_and_odd_l1519_151935


namespace jackson_points_l1519_151909

theorem jackson_points (total_points : ℕ) (num_players : ℕ) (other_players : ℕ) (avg_points : ℕ) :
  total_points = 75 →
  num_players = 8 →
  other_players = 7 →
  avg_points = 6 →
  total_points - (other_players * avg_points) = 33 :=
by sorry

end jackson_points_l1519_151909


namespace charlies_coins_l1519_151918

theorem charlies_coins (total_coins : ℕ) (pennies nickels : ℕ) : 
  total_coins = 17 →
  pennies + nickels = total_coins →
  pennies = nickels + 2 →
  pennies * 1 + nickels * 5 = 44 :=
by sorry

end charlies_coins_l1519_151918


namespace norma_cards_l1519_151984

/-- Given that Norma has 88.0 cards initially and finds 70.0 more cards,
    prove that she will have 158.0 cards in total. -/
theorem norma_cards (initial_cards : Float) (found_cards : Float)
    (h1 : initial_cards = 88.0)
    (h2 : found_cards = 70.0) :
  initial_cards + found_cards = 158.0 := by
  sorry

end norma_cards_l1519_151984


namespace triangle_side_length_l1519_151998

theorem triangle_side_length (a b c : ℝ) (A : Real) :
  a = Real.sqrt 5 →
  b = Real.sqrt 15 →
  A = 30 * Real.pi / 180 →
  c = 2 * Real.sqrt 5 :=
by sorry

end triangle_side_length_l1519_151998


namespace function_properties_l1519_151936

noncomputable def f (x : ℝ) : ℝ := 2 * x / (x + 1)

theorem function_properties :
  (∀ x : ℝ, x ≠ -1 → f x = 2 * x / (x + 1)) ∧
  f 1 = 1 ∧
  f (-2) = 4 ∧
  (∃ c : ℝ, ∀ x : ℝ, x ≠ -1 → f x + f (c - x) = 4) ∧
  (∀ x m : ℝ, x ∈ Set.Icc 1 2 → 2 < m → m ≤ 4 → f x ≤ 2 * m / ((x + 1) * |x - m|)) :=
by sorry

end function_properties_l1519_151936


namespace sum_of_coordinates_on_inverse_graph_l1519_151911

-- Define the function f
def f : ℝ → ℝ := sorry

-- Theorem statement
theorem sum_of_coordinates_on_inverse_graph : 
  (f 2 = 6) → -- This condition is derived from (2,3) being on y=f(x)/2
  ∃ x y : ℝ, (y = 2 * (f⁻¹ x)) ∧ (x + y = 10) := by
  sorry

end sum_of_coordinates_on_inverse_graph_l1519_151911


namespace melissa_score_l1519_151975

/-- Calculates the total score for a player given points per game and number of games played -/
def totalScore (pointsPerGame : ℕ) (numGames : ℕ) : ℕ :=
  pointsPerGame * numGames

/-- Proves that a player scoring 7 points per game for 3 games has a total score of 21 points -/
theorem melissa_score : totalScore 7 3 = 21 := by
  sorry

end melissa_score_l1519_151975


namespace mMobile_first_two_lines_cost_l1519_151948

/-- The cost of a mobile phone plan for a family of 5 -/
structure MobilePlan where
  firstTwoLines : ℕ  -- Cost for first two lines
  additionalLine : ℕ  -- Cost for each additional line

/-- Calculate the total cost for 5 lines -/
def totalCost (plan : MobilePlan) : ℕ :=
  plan.firstTwoLines + 3 * plan.additionalLine

theorem mMobile_first_two_lines_cost : 
  ∃ (mMobile : MobilePlan),
    mMobile.additionalLine = 14 ∧
    ∃ (tMobile : MobilePlan),
      tMobile.firstTwoLines = 50 ∧
      tMobile.additionalLine = 16 ∧
      totalCost tMobile - totalCost mMobile = 11 ∧
      mMobile.firstTwoLines = 45 := by
  sorry

end mMobile_first_two_lines_cost_l1519_151948
