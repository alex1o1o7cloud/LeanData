import Mathlib

namespace problem_solution_l2413_241347

theorem problem_solution (a b c : ℝ) (h1 : |a| = 2) (h2 : a < 1) (h3 : b * c = 1) :
  a^3 + 3 - 4*b*c = -9 := by
sorry

end problem_solution_l2413_241347


namespace min_red_edges_six_red_edges_possible_l2413_241337

/-- Represents the color of an edge -/
inductive Color
| Red
| Green

/-- Represents a cube with colored edges -/
structure Cube :=
  (edges : Fin 12 → Color)

/-- Checks if a face has at least one red edge -/
def faceHasRedEdge (c : Cube) (face : Fin 6) : Prop := sorry

/-- The condition that every face of the cube has at least one red edge -/
def everyFaceHasRedEdge (c : Cube) : Prop :=
  ∀ face : Fin 6, faceHasRedEdge c face

/-- Counts the number of red edges in a cube -/
def countRedEdges (c : Cube) : Nat := sorry

/-- Theorem stating that the minimum number of red edges is 6 -/
theorem min_red_edges (c : Cube) (h : everyFaceHasRedEdge c) : 
  countRedEdges c ≥ 6 := sorry

/-- Theorem stating that 6 red edges is achievable -/
theorem six_red_edges_possible : 
  ∃ c : Cube, everyFaceHasRedEdge c ∧ countRedEdges c = 6 := sorry

end min_red_edges_six_red_edges_possible_l2413_241337


namespace cos_sum_of_complex_exponentials_l2413_241321

theorem cos_sum_of_complex_exponentials (γ δ : ℝ) :
  Complex.exp (γ * Complex.I) = (4 / 5 : ℂ) + (3 / 5 : ℂ) * Complex.I →
  Complex.exp (δ * Complex.I) = -(5 / 13 : ℂ) + (12 / 13 : ℂ) * Complex.I →
  Real.cos (γ + δ) = -(56 / 65) := by
sorry

end cos_sum_of_complex_exponentials_l2413_241321


namespace cube_sum_from_sum_and_square_sum_l2413_241341

theorem cube_sum_from_sum_and_square_sum (x y : ℝ) 
  (h1 : x + y = 5) 
  (h2 : x^2 + y^2 = 17) : 
  x^3 + y^3 = 65 := by sorry

end cube_sum_from_sum_and_square_sum_l2413_241341


namespace crackers_distribution_l2413_241359

theorem crackers_distribution (total_crackers : ℕ) (num_friends : ℕ) (crackers_per_friend : ℕ) :
  total_crackers = 81 →
  num_friends = 27 →
  total_crackers = num_friends * crackers_per_friend →
  crackers_per_friend = 3 :=
by
  sorry

end crackers_distribution_l2413_241359


namespace total_relaxing_is_66_l2413_241308

/-- Calculates the number of people remaining in a row after some leave --/
def remainingInRow (initial : ℕ) (leaving : ℕ) : ℕ :=
  if initial ≥ leaving then initial - leaving else 0

/-- Represents the beach scenario with 5 rows of people --/
structure BeachScenario where
  row1_initial : ℕ
  row1_leaving : ℕ
  row2_initial : ℕ
  row2_leaving : ℕ
  row3_initial : ℕ
  row3_leaving : ℕ
  row4_initial : ℕ
  row4_leaving : ℕ
  row5_initial : ℕ
  row5_leaving : ℕ

/-- Calculates the total number of people still relaxing on the beach --/
def totalRelaxing (scenario : BeachScenario) : ℕ :=
  remainingInRow scenario.row1_initial scenario.row1_leaving +
  remainingInRow scenario.row2_initial scenario.row2_leaving +
  remainingInRow scenario.row3_initial scenario.row3_leaving +
  remainingInRow scenario.row4_initial scenario.row4_leaving +
  remainingInRow scenario.row5_initial scenario.row5_leaving

/-- The given beach scenario --/
def givenScenario : BeachScenario :=
  { row1_initial := 24, row1_leaving := 7
  , row2_initial := 20, row2_leaving := 7
  , row3_initial := 18, row3_leaving := 2
  , row4_initial := 16, row4_leaving := 11
  , row5_initial := 30, row5_leaving := 15 }

/-- Theorem stating that the total number of people still relaxing is 66 --/
theorem total_relaxing_is_66 : totalRelaxing givenScenario = 66 := by
  sorry

end total_relaxing_is_66_l2413_241308


namespace circle_k_bound_l2413_241312

/-- A circle in the Cartesian plane --/
structure Circle where
  equation : ℝ → ℝ → ℝ → Prop

/-- The equation x^2 + y^2 - 2x + y + k = 0 represents a circle --/
def isCircle (k : ℝ) : Prop :=
  ∃ (c : Circle), ∀ (x y : ℝ), c.equation x y k ↔ x^2 + y^2 - 2*x + y + k = 0

/-- If x^2 + y^2 - 2x + y + k = 0 is the equation of a circle, then k < 5/4 --/
theorem circle_k_bound (k : ℝ) : isCircle k → k < 5/4 := by
  sorry

end circle_k_bound_l2413_241312


namespace inequality_proof_l2413_241310

theorem inequality_proof (a b c d : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_pos_d : d > 0)
  (h_product : a * b * c * d = 1) :
  a^2 + b^2 + c^2 + d^2 + a*b + a*c + a*d + b*c + b*d + c*d ≥ 10 := by
  sorry

end inequality_proof_l2413_241310


namespace arithmetic_sequence_with_special_terms_l2413_241393

def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_with_special_terms :
  ∃ (a : ℕ → ℤ) (d : ℤ),
    is_arithmetic_sequence a d ∧
    (∀ i j, i ≠ j → a i ≠ a j) ∧
    a 9 = (a 2) ^ 3 ∧
    (∃ n, a n = (a 2) ^ 2) ∧
    (∃ m, a m = (a 2) ^ 4) →
    a 1 = -24 ∧ a 2 = 6 :=
by sorry

end arithmetic_sequence_with_special_terms_l2413_241393


namespace basketball_team_composition_l2413_241334

-- Define the number of classes
def num_classes : ℕ := 8

-- Define the total number of players
def total_players : ℕ := 10

-- Define the function to calculate the number of composition methods
def composition_methods (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose (n + k - 1) k

-- Theorem statement
theorem basketball_team_composition :
  composition_methods (num_classes) (total_players - num_classes) = 36 := by
  sorry

end basketball_team_composition_l2413_241334


namespace tan_sum_special_l2413_241324

theorem tan_sum_special (θ : Real) (h : Real.tan θ = 2) : Real.tan (θ + π/4) = -3 := by
  sorry

end tan_sum_special_l2413_241324


namespace tram_speed_l2413_241336

/-- Given a tram passing an observer in 2 seconds and traversing a 96-meter tunnel in 10 seconds
    at a constant speed, the speed of the tram is 12 meters per second. -/
theorem tram_speed (passing_time : ℝ) (tunnel_length : ℝ) (tunnel_time : ℝ)
    (h1 : passing_time = 2)
    (h2 : tunnel_length = 96)
    (h3 : tunnel_time = 10) :
  ∃ (v : ℝ), v = 12 ∧ v * passing_time = v * 2 ∧ v * tunnel_time = v * 2 + tunnel_length :=
by
  sorry

#check tram_speed

end tram_speed_l2413_241336


namespace fencing_cost_per_meter_l2413_241369

/-- Proves that the fencing cost per meter is 60 cents for a rectangular park with given conditions -/
theorem fencing_cost_per_meter (length width : ℝ) (area perimeter total_cost : ℝ) : 
  length / width = 3 / 2 →
  area = 3750 →
  area = length * width →
  perimeter = 2 * (length + width) →
  total_cost = 150 →
  (total_cost / perimeter) * 100 = 60 :=
by sorry

end fencing_cost_per_meter_l2413_241369


namespace inequality_solution_l2413_241388

/-- Given an inequality ax^2 - 3x + 2 > 0 with solution set {x | x < 1 or x > b},
    where b > 1 and a > 0, prove that a = 1, b = 2, and the solution set for
    x^2 - 3x + 2 > 0 is {x | 1 < x < 2}. -/
theorem inequality_solution (a b : ℝ) 
    (h1 : ∀ x, a * x^2 - 3*x + 2 > 0 ↔ x < 1 ∨ x > b)
    (h2 : b > 1) 
    (h3 : a > 0) : 
    a = 1 ∧ b = 2 ∧ (∀ x, x^2 - 3*x + 2 > 0 ↔ 1 < x ∧ x < 2) := by
  sorry

end inequality_solution_l2413_241388


namespace optimal_discount_order_l2413_241391

def book_price : ℚ := 30
def flat_discount : ℚ := 5
def percentage_discount : ℚ := 0.25

def price_flat_then_percent : ℚ := (book_price - flat_discount) * (1 - percentage_discount)
def price_percent_then_flat : ℚ := book_price * (1 - percentage_discount) - flat_discount

theorem optimal_discount_order :
  price_percent_then_flat < price_flat_then_percent ∧
  price_flat_then_percent - price_percent_then_flat = 125 / 100 := by
  sorry

end optimal_discount_order_l2413_241391


namespace circle_area_through_two_points_l2413_241355

/-- The area of a circle with center P(2, -1) passing through Q(-4, 5) is 72π. -/
theorem circle_area_through_two_points :
  let P : ℝ × ℝ := (2, -1)
  let Q : ℝ × ℝ := (-4, 5)
  let r := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
  π * r^2 = 72 * π := by
  sorry


end circle_area_through_two_points_l2413_241355


namespace binomial_expansion_properties_l2413_241333

theorem binomial_expansion_properties (n : ℕ) :
  (∀ x : ℝ, x > 0 → 
    Nat.choose n 2 = Nat.choose n 6) →
  (n = 8 ∧ 
   ∀ k : ℕ, k ≤ n → (8 : ℝ) - (3 / 2 : ℝ) * k ≠ 0) :=
by sorry

end binomial_expansion_properties_l2413_241333


namespace rectangle_length_ratio_l2413_241389

theorem rectangle_length_ratio (L B : ℝ) (L' : ℝ) (h1 : L > 0) (h2 : B > 0) : 
  (L' * (3 * B) = (3/2) * (L * B)) → L' / L = 1/2 := by
  sorry

end rectangle_length_ratio_l2413_241389


namespace point_not_on_line_l2413_241370

theorem point_not_on_line (m b : ℝ) (h : m * b > 0) :
  ¬(0 = m * 1997 + b) :=
sorry

end point_not_on_line_l2413_241370


namespace completing_square_result_l2413_241340

theorem completing_square_result (x : ℝ) : x^2 + 4*x + 3 = 0 ↔ (x + 2)^2 = 1 := by
  sorry

end completing_square_result_l2413_241340


namespace unique_function_property_l2413_241362

def iterateFunc (f : ℕ → ℕ) : ℕ → ℕ → ℕ
| 0, x => x
| (n + 1), x => f (iterateFunc f n x)

theorem unique_function_property (f : ℕ → ℕ) 
  (h : ∀ x y : ℕ, 0 ≤ y + f x - iterateFunc f (f y) x ∧ y + f x - iterateFunc f (f y) x ≤ 1) :
  ∀ n : ℕ, f n = n + 1 :=
by sorry

end unique_function_property_l2413_241362


namespace square_garden_perimeter_l2413_241375

theorem square_garden_perimeter (q p : ℝ) (h1 : q > 0) (h2 : p > 0) :
  q = p + 21 → p = 28 := by
  sorry

end square_garden_perimeter_l2413_241375


namespace initial_angelfish_count_l2413_241322

/-- The number of fish initially in the tank -/
def initial_fish (angelfish : ℕ) : ℕ := 94 + angelfish + 89 + 58

/-- The number of fish sold -/
def sold_fish (angelfish : ℕ) : ℕ := 30 + 48 + 17 + 24

/-- The number of fish remaining after the sale -/
def remaining_fish (angelfish : ℕ) : ℕ := initial_fish angelfish - sold_fish angelfish

theorem initial_angelfish_count :
  ∃ (angelfish : ℕ), initial_fish angelfish > 0 ∧ remaining_fish angelfish = 198 ∧ angelfish = 76 := by
  sorry

end initial_angelfish_count_l2413_241322


namespace arithmetic_sequence_properties_l2413_241349

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_properties
  (a : ℕ → ℝ) (d : ℝ) (S : ℕ → ℝ)
  (h_arith : arithmetic_sequence a d)
  (h_sum : ∀ n : ℕ, S n = n * (2 * a 1 + (n - 1) * d) / 2)
  (h_positive : a 1 > 0)
  (h_condition : a 9 + a 10 = a 11) :
  (d < 0) ∧
  (∀ n : ℕ, n > 14 → S n ≤ 0) ∧
  (∃ n : ℕ, n = 14 ∧ S n > 0) :=
by sorry

end arithmetic_sequence_properties_l2413_241349


namespace polynomial_sum_of_coefficients_l2413_241319

theorem polynomial_sum_of_coefficients 
  (g : ℂ → ℂ) 
  (p q r s : ℝ) 
  (h1 : ∀ x, g x = x^4 + p*x^3 + q*x^2 + r*x + s)
  (h2 : g (3*I) = 0)
  (h3 : g (1 + 2*I) = 0) :
  p + q + r + s = 39 := by
sorry

end polynomial_sum_of_coefficients_l2413_241319


namespace point_coordinates_l2413_241354

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- The fourth quadrant of the Cartesian coordinate system -/
def fourth_quadrant (p : Point) : Prop := p.x > 0 ∧ p.y < 0

/-- The distance from a point to the x-axis -/
def distance_to_x_axis (p : Point) : ℝ := |p.y|

/-- The distance from a point to the y-axis -/
def distance_to_y_axis (p : Point) : ℝ := |p.x|

theorem point_coordinates :
  ∀ (A : Point),
    fourth_quadrant A →
    distance_to_x_axis A = 3 →
    distance_to_y_axis A = 6 →
    A.x = 6 ∧ A.y = -3 := by
  sorry

end point_coordinates_l2413_241354


namespace negation_equivalence_l2413_241353

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) ↔ (∀ x : ℝ, x^2 + 2*x + 2 > 0) := by
  sorry

end negation_equivalence_l2413_241353


namespace train_length_l2413_241307

/-- The length of a train given its speed, time to pass a platform, and platform length -/
theorem train_length (train_speed : ℝ) (pass_time : ℝ) (platform_length : ℝ) : 
  train_speed = 45 * (1000 / 3600) →
  pass_time = 40 →
  platform_length = 140 →
  train_speed * pass_time - platform_length = 360 := by
  sorry

#check train_length

end train_length_l2413_241307


namespace specific_shaded_square_ratio_l2413_241320

/-- A square divided into smaller squares with a shading pattern -/
structure ShadedSquare where
  /-- The number of equal triangles in the shaded area of each quarter -/
  shaded_triangles : ℕ
  /-- The number of equal triangles in the white area of each quarter -/
  white_triangles : ℕ

/-- The ratio of shaded area to white area in a ShadedSquare -/
def shaded_to_white_ratio (s : ShadedSquare) : ℚ :=
  s.shaded_triangles / s.white_triangles

/-- Theorem stating the ratio of shaded to white area for a specific configuration -/
theorem specific_shaded_square_ratio :
  ∃ (s : ShadedSquare), s.shaded_triangles = 5 ∧ s.white_triangles = 3 ∧ 
  shaded_to_white_ratio s = 5 / 3 := by
  sorry

end specific_shaded_square_ratio_l2413_241320


namespace root_sum_reciprocals_l2413_241326

theorem root_sum_reciprocals (a b c d : ℂ) : 
  (a^4 + 8*a^3 + 9*a^2 + 5*a + 4 = 0) →
  (b^4 + 8*b^3 + 9*b^2 + 5*b + 4 = 0) →
  (c^4 + 8*c^3 + 9*c^2 + 5*c + 4 = 0) →
  (d^4 + 8*d^3 + 9*d^2 + 5*d + 4 = 0) →
  (1/(a*b) + 1/(a*c) + 1/(a*d) + 1/(b*c) + 1/(b*d) + 1/(c*d) = 9/4) :=
by sorry

end root_sum_reciprocals_l2413_241326


namespace borrow_three_books_l2413_241318

/-- The number of ways to borrow at least one book out of three books -/
def borrow_methods (n : ℕ) : ℕ := 2^n - 1

/-- Theorem stating that the number of ways to borrow at least one book out of three books is 7 -/
theorem borrow_three_books : borrow_methods 3 = 7 := by
  sorry

end borrow_three_books_l2413_241318


namespace pyramid_tiers_count_l2413_241329

/-- Calculates the surface area of a pyramid with n tiers built from 1 cm³ cubes -/
def pyramidSurfaceArea (n : ℕ) : ℕ :=
  4 * n^2 + 2 * n

/-- A pyramid built from 1 cm³ cubes with a surface area of 2352 cm² has 24 tiers -/
theorem pyramid_tiers_count : ∃ n : ℕ, pyramidSurfaceArea n = 2352 ∧ n = 24 := by
  sorry

#eval pyramidSurfaceArea 24  -- Should output 2352

end pyramid_tiers_count_l2413_241329


namespace complex_magnitude_and_real_part_sum_of_combinations_l2413_241360

-- Problem 1
theorem complex_magnitude_and_real_part (z : ℂ) (ω : ℝ) 
  (h1 : ω = z + 1/z)
  (h2 : -1 < ω)
  (h3 : ω < 2) :
  Complex.abs z = 1 ∧ ∃ (a : ℝ), z.re = a ∧ -1/2 < a ∧ a < 1 :=
sorry

-- Problem 2
theorem sum_of_combinations : 
  (Nat.choose 5 4) + (Nat.choose 6 4) + (Nat.choose 7 4) + 
  (Nat.choose 8 4) + (Nat.choose 9 4) + (Nat.choose 10 4) = 461 :=
sorry

end complex_magnitude_and_real_part_sum_of_combinations_l2413_241360


namespace max_value_of_expression_l2413_241368

theorem max_value_of_expression :
  ∃ (a b c d : ℕ),
    ({a, b, c, d} : Finset ℕ) = {1, 2, 4, 5} →
    ∀ (w x y z : ℕ),
      ({w, x, y, z} : Finset ℕ) = {1, 2, 4, 5} →
      c * a^b - d ≤ 79 ∧
      (c * a^b - d = 79 → (a = 2 ∧ b = 4 ∧ c = 5 ∧ d = 1) ∨ (a = 4 ∧ b = 2 ∧ c = 5 ∧ d = 1)) :=
by sorry

#check max_value_of_expression

end max_value_of_expression_l2413_241368


namespace negation_of_universal_proposition_l2413_241384

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - x ≥ 0) ↔ (∃ x₀ : ℝ, x₀^2 - x₀ < 0) := by sorry

end negation_of_universal_proposition_l2413_241384


namespace function_value_at_two_l2413_241343

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x ≠ 0, f x - 3 * f (1 / x) = 3^x

theorem function_value_at_two
  (f : ℝ → ℝ) (h : FunctionalEquation f) :
  f 2 = -(9 + 3 * Real.sqrt 3) / 8 := by
  sorry

end function_value_at_two_l2413_241343


namespace problem_one_problem_two_problem_three_problem_four_l2413_241345

-- Problem 1
theorem problem_one : (-23) - (-58) + (-17) = 18 := by sorry

-- Problem 2
theorem problem_two : (-8) / (-1 - 1/9) * 0.125 = 9/10 := by sorry

-- Problem 3
theorem problem_three : (-1/3 - 1/4 + 1/15) * (-60) = 31 := by sorry

-- Problem 4
theorem problem_four : -1^2 * |(-1/4)| + (-1/2)^3 / (-1)^2023 = -1/8 := by sorry

end problem_one_problem_two_problem_three_problem_four_l2413_241345


namespace matrix_cube_property_l2413_241377

theorem matrix_cube_property (a b c : ℂ) : 
  let M : Matrix (Fin 3) (Fin 3) ℂ := !![a, b, c; b, c, a; c, a, b]
  (M^3 = 1) → (a*b*c = -1) → (a^3 + b^3 + c^3 = 4) := by
sorry

end matrix_cube_property_l2413_241377


namespace candy_bar_to_caramel_ratio_l2413_241315

/-- The price of caramel in dollars -/
def caramel_price : ℚ := 3

/-- The price of a candy bar as a multiple of the caramel price -/
def candy_bar_price (k : ℚ) : ℚ := k * caramel_price

/-- The price of cotton candy -/
def cotton_candy_price (k : ℚ) : ℚ := 2 * candy_bar_price k

/-- The total cost of 6 candy bars, 3 caramel, and 1 cotton candy -/
def total_cost (k : ℚ) : ℚ := 6 * candy_bar_price k + 3 * caramel_price + cotton_candy_price k

theorem candy_bar_to_caramel_ratio :
  ∃ k : ℚ, total_cost k = 57 ∧ candy_bar_price k / caramel_price = 2 := by
  sorry

end candy_bar_to_caramel_ratio_l2413_241315


namespace tan_two_implies_sin_2theta_over_cos_squared_minus_sin_squared_l2413_241311

theorem tan_two_implies_sin_2theta_over_cos_squared_minus_sin_squared (θ : Real) 
  (h : Real.tan θ = 2) : 
  (Real.sin (2 * θ)) / (Real.cos θ ^ 2 - Real.sin θ ^ 2) = -4/3 := by
  sorry

end tan_two_implies_sin_2theta_over_cos_squared_minus_sin_squared_l2413_241311


namespace key_chain_profit_percentage_l2413_241309

theorem key_chain_profit_percentage 
  (selling_price : ℝ)
  (old_cost new_cost : ℝ)
  (h1 : old_cost = 65)
  (h2 : new_cost = 50)
  (h3 : selling_price - new_cost = 0.5 * selling_price) :
  (selling_price - old_cost) / selling_price = 0.35 :=
by sorry

end key_chain_profit_percentage_l2413_241309


namespace function_decomposition_even_odd_l2413_241327

theorem function_decomposition_even_odd (f : ℝ → ℝ) :
  ∃! (f₀ f₁ : ℝ → ℝ),
    (∀ x, f x = f₀ x + f₁ x) ∧
    (∀ x, f₀ (-x) = f₀ x) ∧
    (∀ x, f₁ (-x) = -f₁ x) ∧
    (∀ x, f₀ x = (1/2) * (f x + f (-x))) ∧
    (∀ x, f₁ x = (1/2) * (f x - f (-x))) := by
  sorry

end function_decomposition_even_odd_l2413_241327


namespace students_with_B_grade_l2413_241304

def total_students : ℕ := 40

def prob_A (x : ℕ) : ℚ := (3 : ℚ) / 5 * x
def prob_B (x : ℕ) : ℚ := x
def prob_C (x : ℕ) : ℚ := (6 : ℚ) / 5 * x

theorem students_with_B_grade :
  ∃ x : ℕ, 
    x ≤ total_students ∧
    (prob_A x + prob_B x + prob_C x : ℚ) = total_students ∧
    x = 14 := by sorry

end students_with_B_grade_l2413_241304


namespace infinite_cube_square_triples_l2413_241305

theorem infinite_cube_square_triples :
  ∃ S : Set (ℤ × ℤ × ℤ), Set.Infinite S ∧ 
  ∀ (x y z : ℤ), (x, y, z) ∈ S → x^2 + y^2 + z^2 = x^3 + y^3 + z^3 :=
by
  sorry

end infinite_cube_square_triples_l2413_241305


namespace annual_reduction_equation_l2413_241358

/-- The total cost reduction percentage over two years -/
def total_reduction : ℝ := 0.36

/-- The average annual reduction percentage -/
def x : ℝ := sorry

/-- Theorem stating the relationship between the average annual reduction and total reduction -/
theorem annual_reduction_equation : (1 - x)^2 = 1 - total_reduction := by sorry

end annual_reduction_equation_l2413_241358


namespace goldfish_graph_is_finite_distinct_points_l2413_241364

/-- Represents the cost of purchasing goldfish -/
def goldfish_cost (n : ℕ) : ℚ :=
  if n ≥ 3 then 20 * n else 0

/-- The set of points representing goldfish purchases from 3 to 15 -/
def goldfish_graph : Set (ℕ × ℚ) :=
  {p | ∃ n : ℕ, 3 ≤ n ∧ n ≤ 15 ∧ p = (n, goldfish_cost n)}

theorem goldfish_graph_is_finite_distinct_points :
  Finite goldfish_graph ∧ ∀ p q : (ℕ × ℚ), p ∈ goldfish_graph → q ∈ goldfish_graph → p ≠ q → p.1 ≠ q.1 :=
sorry

end goldfish_graph_is_finite_distinct_points_l2413_241364


namespace repeating_decimal_726_eq_fraction_l2413_241325

/-- The definition of a repeating decimal with period 726 -/
def repeating_decimal_726 : ℚ :=
  726 / 999

/-- Theorem stating that 0.726726726... equals 242/333 -/
theorem repeating_decimal_726_eq_fraction : repeating_decimal_726 = 242 / 333 := by
  sorry

end repeating_decimal_726_eq_fraction_l2413_241325


namespace f_neither_odd_nor_even_l2413_241303

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2

-- Define the domain of f
def domain : Set ℝ := Set.Ioc (-5) 5

-- Define what it means for a function to be odd
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define what it means for a function to be even
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Theorem statement
theorem f_neither_odd_nor_even :
  ¬(is_odd f) ∧ ¬(is_even f) :=
sorry

end f_neither_odd_nor_even_l2413_241303


namespace consecutive_odd_numbers_sum_l2413_241392

theorem consecutive_odd_numbers_sum (a b c d : ℕ) : 
  (∃ x : ℕ, a = 2*x + 1 ∧ b = 2*x + 3 ∧ c = 2*x + 5 ∧ d = 2*x + 7) →  -- Consecutive odd numbers
  a + b + c + d = 112 →  -- Sum is 112
  b = 27  -- Second smallest is 27
:= by sorry

end consecutive_odd_numbers_sum_l2413_241392


namespace great_dane_weight_is_307_l2413_241351

/-- The weight of three dogs (chihuahua, pitbull, and great dane) -/
def total_weight : ℕ := 439

/-- The weight relationship between the pitbull and the chihuahua -/
def pitbull_weight (chihuahua_weight : ℕ) : ℕ := 3 * chihuahua_weight

/-- The weight relationship between the great dane and the pitbull -/
def great_dane_weight (pitbull_weight : ℕ) : ℕ := 3 * pitbull_weight + 10

/-- Theorem stating that the great dane weighs 307 pounds -/
theorem great_dane_weight_is_307 :
  ∃ (chihuahua_weight : ℕ),
    chihuahua_weight + pitbull_weight chihuahua_weight + great_dane_weight (pitbull_weight chihuahua_weight) = total_weight ∧
    great_dane_weight (pitbull_weight chihuahua_weight) = 307 :=
by
  sorry

end great_dane_weight_is_307_l2413_241351


namespace quadratic_inequality_problem_l2413_241363

-- Define the quadratic function
def f (a b x : ℝ) := a * x^2 - (a + 1) * x + b

-- Define the solution set condition
def solution_set (a b : ℝ) :=
  ∀ x, f a b x < 0 ↔ (x < -1/2 ∨ x > 1)

-- Define the inequality for part II
def g (x m : ℝ) := x^2 + (m - 4) * x + 3 - m

-- Main theorem
theorem quadratic_inequality_problem :
  ∃ a b : ℝ,
    solution_set a b ∧
    a = -2 ∧
    b = 1 ∧
    (∀ x, (∀ m ∈ Set.Icc 0 4, g x m ≥ 0) ↔ 
      (x ≤ -1 ∨ x = 1 ∨ x ≥ 3)) :=
by sorry

end quadratic_inequality_problem_l2413_241363


namespace sin_decreasing_omega_range_l2413_241356

theorem sin_decreasing_omega_range (ω : ℝ) (h_pos : ω > 0) :
  (∀ x ∈ Set.Icc (π / 4) (π / 2), 
    ∀ y ∈ Set.Icc (π / 4) (π / 2), 
    x < y → Real.sin (ω * x) > Real.sin (ω * y)) →
  ω ∈ Set.Icc 2 3 := by
  sorry

end sin_decreasing_omega_range_l2413_241356


namespace eunji_gymnastics_count_l2413_241397

/-- Represents the position of a student in a rectangular arrangement -/
structure StudentPosition where
  leftColumn : Nat
  rightColumn : Nat
  frontRow : Nat
  backRow : Nat

/-- Calculates the total number of students in a rectangular arrangement -/
def totalStudents (pos : StudentPosition) : Nat :=
  let totalColumns := pos.leftColumn + pos.rightColumn - 1
  let totalRows := pos.frontRow + pos.backRow - 1
  totalColumns * totalRows

/-- Theorem: Given Eunji's position, the total number of students is 441 -/
theorem eunji_gymnastics_count :
  let eunjiPosition : StudentPosition := {
    leftColumn := 8,
    rightColumn := 14,
    frontRow := 7,
    backRow := 15
  }
  totalStudents eunjiPosition = 441 := by
  sorry

end eunji_gymnastics_count_l2413_241397


namespace tan_negative_55_6_pi_l2413_241383

theorem tan_negative_55_6_pi : Real.tan (-55/6 * Real.pi) = -Real.sqrt 3 / 3 := by
  sorry

end tan_negative_55_6_pi_l2413_241383


namespace symmetric_point_xoy_plane_l2413_241366

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The xOy plane in 3D space -/
def xOyPlane : Set Point3D :=
  {p : Point3D | p.z = 0}

/-- Symmetry with respect to the xOy plane -/
def symmetricToXOyPlane (p q : Point3D) : Prop :=
  p.x = q.x ∧ p.y = q.y ∧ p.z = -q.z

theorem symmetric_point_xoy_plane :
  let m : Point3D := ⟨2, 5, 8⟩
  let n : Point3D := ⟨2, 5, -8⟩
  symmetricToXOyPlane m n := by
  sorry

end symmetric_point_xoy_plane_l2413_241366


namespace solution_of_equation_l2413_241382

theorem solution_of_equation (x : ℚ) : -2 * x + 11 = 0 ↔ x = 11 / 2 := by sorry

end solution_of_equation_l2413_241382


namespace smallest_prime_divisor_of_sum_l2413_241317

theorem smallest_prime_divisor_of_sum : 
  ∃ (p : ℕ), Prime p ∧ p ∣ (7^15 + 9^17) ∧ ∀ q, Prime q → q ∣ (7^15 + 9^17) → p ≤ q :=
by sorry

end smallest_prime_divisor_of_sum_l2413_241317


namespace coefficient_x5_eq_11_l2413_241328

/-- The coefficient of x^5 in the expansion of (x^2 + x - 1)^5 -/
def coefficient_x5 : ℤ :=
  (Nat.choose 5 0) * (Nat.choose 5 5) -
  (Nat.choose 5 1) * (Nat.choose 4 3) +
  (Nat.choose 5 2) * (Nat.choose 3 1)

/-- Theorem stating that the coefficient of x^5 in (x^2 + x - 1)^5 is 11 -/
theorem coefficient_x5_eq_11 : coefficient_x5 = 11 := by
  sorry

end coefficient_x5_eq_11_l2413_241328


namespace sets_intersection_union_l2413_241306

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - 2008*x - 2009 > 0}
def N (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b ≤ 0}

-- Define the open interval (2009, 2010]
def openInterval : Set ℝ := Set.Ioc 2009 2010

-- State the theorem
theorem sets_intersection_union (a b : ℝ) : 
  (M ∪ N a b = Set.univ) ∧ (M ∩ N a b = openInterval) → a = 2009 ∧ b = 2010 := by
  sorry

end sets_intersection_union_l2413_241306


namespace no_integer_fourth_root_l2413_241338

theorem no_integer_fourth_root : ¬∃ (n : ℕ), n > 0 ∧ 5^4 + 12^4 + 9^4 + 8^4 = n^4 := by
  sorry

end no_integer_fourth_root_l2413_241338


namespace particle_position_at_1989_l2413_241332

/-- Represents the position of a particle on a 2D plane -/
structure Position :=
  (x : ℝ)
  (y : ℝ)

/-- Defines the movement pattern of the particle -/
def move (t : ℕ) : Position :=
  sorry

/-- The theorem to be proved -/
theorem particle_position_at_1989 :
  move 1989 = Position.mk 0 0 := by
  sorry

end particle_position_at_1989_l2413_241332


namespace chess_tournament_participants_l2413_241350

theorem chess_tournament_participants (n m : ℕ) : 
  9 < n → n < 25 →  -- Total participants between 9 and 25
  (n - 2*m)^2 = n →  -- Derived equation from the condition about scoring half points against grandmasters
  (n = 16 ∧ (m = 6 ∨ m = 10)) := by
sorry

end chess_tournament_participants_l2413_241350


namespace tank_capacity_l2413_241385

theorem tank_capacity (C : ℚ) 
  (h1 : (3/4 : ℚ) * C + 4 = (7/8 : ℚ) * C) : C = 32 := by
  sorry

end tank_capacity_l2413_241385


namespace arrangements_count_l2413_241399

/-- The number of students in the row -/
def n : ℕ := 7

/-- The number of positions where A and B can be placed with one person in between -/
def positions : ℕ := 5

/-- The number of ways to arrange the remaining students -/
def remaining_arrangements : ℕ := Nat.factorial (n - 3)

/-- The number of ways A and B can switch places -/
def ab_switch : ℕ := 2

/-- The total number of arrangements for 7 students standing in a row,
    where there must be one person standing between students A and B -/
def total_arrangements : ℕ := positions * remaining_arrangements * ab_switch

theorem arrangements_count : total_arrangements = 1200 := by
  sorry

end arrangements_count_l2413_241399


namespace intersection_segment_equals_incircle_diameter_l2413_241344

/-- Right triangle with incircle and two circles on hypotenuse endpoints -/
structure RightTriangleWithCircles where
  -- Legs of the right triangle
  a : ℝ
  b : ℝ
  -- Hypotenuse of the right triangle
  c : ℝ
  -- Radius of the incircle
  r : ℝ
  -- The triangle is right-angled
  right_angle : a^2 + b^2 = c^2
  -- The incircle exists and touches all sides
  incircle : a + b - c = 2 * r
  -- All lengths are positive
  a_pos : a > 0
  b_pos : b > 0
  c_pos : c > 0
  r_pos : r > 0

/-- The length of the intersection segment equals the incircle diameter -/
theorem intersection_segment_equals_incircle_diameter 
  (t : RightTriangleWithCircles) : a + b - c = 2 * r :=
by sorry

end intersection_segment_equals_incircle_diameter_l2413_241344


namespace kittens_found_on_monday_l2413_241379

def solve_cat_problem (initial_cats : ℕ) (tuesday_cats : ℕ) (adoptions : ℕ) (cats_per_adoption : ℕ) (final_cats : ℕ) : ℕ :=
  initial_cats + tuesday_cats - (adoptions * cats_per_adoption) - final_cats

theorem kittens_found_on_monday :
  solve_cat_problem 20 1 3 2 17 = 2 := by
  sorry

end kittens_found_on_monday_l2413_241379


namespace boys_ratio_l2413_241331

theorem boys_ratio (total : ℕ) (boys : ℕ) (girls : ℕ) 
  (h1 : total = boys + girls) 
  (h2 : boys > 0 ∧ girls > 0) 
  (h3 : (boys : ℚ) / total = 3/5 * (girls : ℚ) / total) : 
  (boys : ℚ) / total = 3/8 := by
  sorry

end boys_ratio_l2413_241331


namespace ac_range_l2413_241348

-- Define the triangle
def Triangle (A B C : ℝ × ℝ) : Prop := sorry

-- Define the angle at a vertex
def Angle (A B C : ℝ × ℝ) : ℝ := sorry

-- Define the length of a side
def SideLength (A B : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem ac_range (A B C : ℝ × ℝ) : 
  Triangle A B C → 
  Angle A B C < π / 2 → 
  Angle B C A < π / 2 → 
  Angle C A B < π / 2 → 
  SideLength B C = 1 → 
  Angle B A C = 2 * Angle A B C → 
  Real.sqrt 2 < SideLength A C ∧ SideLength A C < Real.sqrt 3 := by
sorry

end ac_range_l2413_241348


namespace minutes_after_midnight_theorem_l2413_241387

/-- Represents a date and time -/
structure DateTime where
  year : Nat
  month : Nat
  day : Nat
  hour : Nat
  minute : Nat

/-- Adds minutes to a DateTime -/
def addMinutes (dt : DateTime) (minutes : Nat) : DateTime :=
  sorry

/-- The initial DateTime (midnight on February 1, 2022) -/
def initialDateTime : DateTime :=
  { year := 2022, month := 2, day := 1, hour := 0, minute := 0 }

/-- The final DateTime after adding 1553 minutes -/
def finalDateTime : DateTime :=
  addMinutes initialDateTime 1553

/-- Theorem stating that 1553 minutes after midnight on February 1, 2022 is February 2 at 1:53 AM -/
theorem minutes_after_midnight_theorem :
  finalDateTime = { year := 2022, month := 2, day := 2, hour := 1, minute := 53 } :=
  sorry

end minutes_after_midnight_theorem_l2413_241387


namespace range_of_K_l2413_241374

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x + 2| - 1
def g (x : ℝ) : ℝ := |3 - x| + 2

-- Define the theorem
theorem range_of_K (K : ℝ) : 
  (∀ x : ℝ, f x - g x ≤ K) → K ∈ Set.Ici 2 := by
  sorry

end range_of_K_l2413_241374


namespace expected_value_of_coins_l2413_241381

/-- The expected value of coins coming up heads when flipping four coins simultaneously -/
theorem expected_value_of_coins (nickel quarter half_dollar dollar : ℕ) 
  (h_nickel : nickel = 5)
  (h_quarter : quarter = 25)
  (h_half_dollar : half_dollar = 50)
  (h_dollar : dollar = 100)
  (p_heads : ℚ)
  (h_p_heads : p_heads = 1 / 2) : 
  p_heads * (nickel + quarter + half_dollar + dollar : ℚ) = 90 := by
sorry

end expected_value_of_coins_l2413_241381


namespace dissected_rectangle_perimeter_l2413_241330

/-- A rectangle dissected into nine non-overlapping squares -/
structure DissectedRectangle where
  width : ℕ+
  height : ℕ+
  squares : Fin 9 → ℕ+
  sum_squares : width * height = (squares 0).val + (squares 1).val + (squares 2).val + (squares 3).val + 
                                 (squares 4).val + (squares 5).val + (squares 6).val + (squares 7).val + 
                                 (squares 8).val

/-- The perimeter of a rectangle -/
def perimeter (rect : DissectedRectangle) : ℕ :=
  2 * (rect.width + rect.height)

/-- The theorem to be proved -/
theorem dissected_rectangle_perimeter (rect : DissectedRectangle) 
  (h_coprime : Nat.Coprime rect.width rect.height) : 
  perimeter rect = 260 := by
  sorry

end dissected_rectangle_perimeter_l2413_241330


namespace stations_visited_l2413_241373

theorem stations_visited (total_nails : ℕ) (nails_per_station : ℕ) (h1 : total_nails = 140) (h2 : nails_per_station = 7) :
  total_nails / nails_per_station = 20 := by
  sorry

end stations_visited_l2413_241373


namespace grant_total_earnings_l2413_241395

/-- The total amount Grant made from selling his baseball gear -/
def total_amount : ℝ :=
  let baseball_cards := 25
  let baseball_bat := 10
  let baseball_glove_original := 30
  let baseball_glove_discount := 0.20
  let baseball_glove := baseball_glove_original * (1 - baseball_glove_discount)
  let baseball_cleats := 10
  let num_cleats := 2
  baseball_cards + baseball_bat + baseball_glove + (baseball_cleats * num_cleats)

/-- Theorem stating that the total amount Grant made is $79 -/
theorem grant_total_earnings : total_amount = 79 := by
  sorry

end grant_total_earnings_l2413_241395


namespace birth_year_digit_sum_difference_l2413_241371

theorem birth_year_digit_sum_difference (m c d u : Nat) 
  (hm : m < 10) (hc : c < 10) (hd : d < 10) (hu : u < 10) :
  ∃ k : Int, (1000 * m + 100 * c + 10 * d + u) - (m + c + d + u) = 9 * k := by
  sorry

end birth_year_digit_sum_difference_l2413_241371


namespace computers_fixed_right_away_l2413_241376

theorem computers_fixed_right_away (total : ℕ) (unfixable_percent : ℚ) (spare_parts_percent : ℚ) :
  total = 20 →
  unfixable_percent = 20 / 100 →
  spare_parts_percent = 40 / 100 →
  (total : ℚ) * (1 - unfixable_percent - spare_parts_percent) = 8 := by
  sorry

end computers_fixed_right_away_l2413_241376


namespace even_function_inequality_l2413_241301

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the property of being an even function
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- State the theorem
theorem even_function_inequality (h1 : IsEven f) (h2 : f 2 < f 3) : f (-3) > f (-2) := by
  sorry

end even_function_inequality_l2413_241301


namespace committee_size_lower_bound_l2413_241394

/-- A structure representing a committee with its meeting details -/
structure Committee where
  total_meetings : ℕ
  members_per_meeting : ℕ
  total_members : ℕ

/-- The property that no two people have met more than once -/
def no_repeated_meetings (c : Committee) : Prop :=
  c.total_meetings * (c.members_per_meeting.choose 2) ≤ c.total_members.choose 2

/-- The theorem to be proved -/
theorem committee_size_lower_bound (c : Committee) 
  (h1 : c.total_meetings = 40)
  (h2 : c.members_per_meeting = 10)
  (h3 : no_repeated_meetings c) :
  c.total_members > 60 := by
  sorry

end committee_size_lower_bound_l2413_241394


namespace impossible_coverage_l2413_241398

/-- Represents a 5x5 board --/
def Board := Fin 5 → Fin 5 → Bool

/-- Represents a stromino (3x1 rectangle) --/
structure Stromino where
  start_row : Fin 5
  start_col : Fin 5
  is_horizontal : Bool

/-- Checks if a stromino is valid (fits within the board) --/
def is_valid_stromino (s : Stromino) : Bool :=
  if s.is_horizontal then
    s.start_col < 3
  else
    s.start_row < 3

/-- Counts how many strominos cover a given square --/
def count_coverage (board : Board) (strominos : List Stromino) (row col : Fin 5) : Nat :=
  sorry

/-- Checks if a given arrangement of strominos is valid --/
def is_valid_arrangement (strominos : List Stromino) : Bool :=
  sorry

/-- The main theorem stating that it's impossible to cover the board with 16 strominos --/
theorem impossible_coverage : ¬ ∃ (strominos : List Stromino),
  strominos.length = 16 ∧
  is_valid_arrangement strominos ∧
  ∀ (row col : Fin 5),
    let coverage := count_coverage (λ _ _ => true) strominos row col
    coverage = 1 ∨ coverage = 2 :=
  sorry

end impossible_coverage_l2413_241398


namespace cube_difference_l2413_241346

theorem cube_difference (a b : ℝ) (h1 : a - b = 7) (h2 : a^2 + b^2 = 53) :
  a^3 - b^3 = 385 := by
  sorry

end cube_difference_l2413_241346


namespace a_range_l2413_241367

def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 0 1, a ≥ Real.exp x

def q (a : ℝ) : Prop := ∃ x₀ : ℝ, x₀^2 + 4*x₀ + a = 0

theorem a_range (a : ℝ) (h : p a ∧ q a) : a ∈ Set.Icc (Real.exp 1) 4 := by
  sorry

end a_range_l2413_241367


namespace quebec_temperature_l2413_241339

-- Define the temperatures as integers (assuming we're working with whole numbers)
def temp_vancouver : ℤ := 22
def temp_calgary : ℤ := temp_vancouver - 19
def temp_quebec : ℤ := temp_calgary - 11

-- Theorem to prove
theorem quebec_temperature : temp_quebec = -8 := by
  sorry

end quebec_temperature_l2413_241339


namespace absolute_value_simplification_l2413_241357

theorem absolute_value_simplification (x : ℝ) (h : x < -1) :
  |x - 2 * Real.sqrt ((x + 1)^2)| = -3 * x - 2 := by
  sorry

end absolute_value_simplification_l2413_241357


namespace lieutenant_age_l2413_241372

theorem lieutenant_age : ∃ (n : ℕ), ∃ (x : ℕ), 
  -- Initial arrangement: n rows with n+5 soldiers each
  -- New arrangement: x rows (lieutenant's age) with n+9 soldiers each
  -- Total number of soldiers remains the same
  n * (n + 5) = x * (n + 9) ∧
  -- x represents a reasonable age for a lieutenant
  x > 18 ∧ x < 65 ∧
  -- The solution
  x = 24 := by
sorry

end lieutenant_age_l2413_241372


namespace circle_equation_l2413_241378

/-- The standard equation of a circle with center (-1, 2) passing through (2, -2) -/
theorem circle_equation : ∀ x y : ℝ, (x + 1)^2 + (y - 2)^2 = 25 ↔ 
  ((x + 1)^2 + (y - 2)^2 = ((2 + 1)^2 + (-2 - 2)^2) ∧ 
   (x, y) ≠ (-1, 2)) := by sorry

end circle_equation_l2413_241378


namespace book_pages_l2413_241323

/-- A book with a certain number of pages -/
structure Book where
  pages : ℕ

/-- Reading progress over four days -/
structure ReadingProgress where
  day1 : Rat
  day2 : Rat
  day3 : Rat
  day4 : ℕ

/-- Theorem stating the total number of pages in the book -/
theorem book_pages (b : Book) (rp : ReadingProgress) 
  (h1 : rp.day1 = 1/2)
  (h2 : rp.day2 = 1/4)
  (h3 : rp.day3 = 1/6)
  (h4 : rp.day4 = 20)
  (h5 : rp.day1 + rp.day2 + rp.day3 + (rp.day4 : Rat) / b.pages = 1) :
  b.pages = 240 := by
  sorry

end book_pages_l2413_241323


namespace largest_square_leftover_l2413_241361

def yarn_length : ℕ := 35

theorem largest_square_leftover (s : ℕ) : 
  (s * 4 ≤ yarn_length) ∧ 
  (∀ t : ℕ, t * 4 ≤ yarn_length → t ≤ s) →
  yarn_length - s * 4 = 3 :=
by sorry

end largest_square_leftover_l2413_241361


namespace parabola_properties_l2413_241300

-- Define the parabola function
def f (x : ℝ) : ℝ := 3 * x^2 + 6 * x + 4

-- Define the interval
def interval : Set ℝ := Set.Icc (-3) 2

-- Theorem statement
theorem parabola_properties :
  (∃ (x y : ℝ), x = -1 ∧ y = 1 ∧ ∀ (t : ℝ), f t ≥ f x) ∧ 
  (∀ (x₁ x₂ : ℝ), f x₁ = f x₂ → |x₁ + 1| = |x₂ + 1|) ∧
  (Set.Icc 1 28 = {y | ∃ x ∈ interval, f x = y}) :=
sorry

end parabola_properties_l2413_241300


namespace arithmetic_sequence_theorem_l2413_241352

-- Define the arithmetic sequence
def a (n : ℕ) : ℚ := 2 * n

-- Define the sequence b_n
def b (n : ℕ) : ℚ := 2^(n-1) * a n

-- Define the sum of the first n terms of b_n
def T (n : ℕ) : ℚ := (n - 1) * 2^(n + 1) + 2

-- State the theorem
theorem arithmetic_sequence_theorem (d : ℚ) (h_d : d ≠ 0) :
  (a 2 + 2 * a 4 = 20) ∧
  (∃ r : ℚ, a 3 = r * a 1 ∧ a 9 = r * a 3) →
  (∀ n : ℕ, n ≥ 1 → a n = 2 * n) ∧
  (∀ n : ℕ, T n = (n - 1) * 2^(n + 1) + 2) :=
by sorry


end arithmetic_sequence_theorem_l2413_241352


namespace nickel_chocolates_l2413_241313

theorem nickel_chocolates (robert : ℕ) (difference : ℕ) (nickel : ℕ) : 
  robert = 13 → 
  robert = nickel + difference → 
  difference = 9 → 
  nickel = 4 := by sorry

end nickel_chocolates_l2413_241313


namespace base6_divisibility_l2413_241396

/-- Converts a base 6 number to decimal --/
def base6ToDecimal (a b c d : ℕ) : ℕ :=
  a * 6^3 + b * 6^2 + c * 6 + d

/-- Checks if a number is divisible by 19 --/
def isDivisibleBy19 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 19 * k

theorem base6_divisibility :
  isDivisibleBy19 (base6ToDecimal 4 5 5 2) ∧
  ∀ x : ℕ, x < 5 → ¬isDivisibleBy19 (base6ToDecimal 4 5 x 2) :=
by sorry

end base6_divisibility_l2413_241396


namespace certain_number_operations_l2413_241386

theorem certain_number_operations (x : ℝ) : 
  ∃ (p q : ℕ), p < q ∧ ((x + 20) * 2 / 2 - 2 = x + 18) ∧ (x + 18 = (p : ℝ) / q * 88) := by
  sorry

end certain_number_operations_l2413_241386


namespace problem_statements_l2413_241316

theorem problem_statements :
  (∀ x : ℝ, x ≥ 0 → x + 1 + 1 / (x + 1) ≥ 2) ∧
  (∀ x : ℝ, x > 0 → (x + 1) / Real.sqrt x ≥ 2) ∧
  (∃ x : ℝ, x + 1 / x < 2) ∧
  (∀ x : ℝ, Real.sqrt (x^2 + 2) + 1 / Real.sqrt (x^2 + 2) > 2) :=
by
  sorry

end problem_statements_l2413_241316


namespace conditions_implications_l2413_241314

-- Define the conditions
def p (a : ℝ) : Prop := ∀ x > 0, Monotone (fun x => Real.log x / Real.log (a - 1))
def q (a : ℝ) : Prop := (2 - a) / (a - 3) > 0
def r (a : ℝ) : Prop := a < 3
def s (a : ℝ) : Prop := ∀ x, ∃ y, y = Real.log (x^2 - 2*x + a) / Real.log 10

-- State the theorem
theorem conditions_implications (a : ℝ) :
  (p a → a > 2) ∧
  (q a ↔ (2 < a ∧ a < 3)) ∧
  (s a → a > 1) ∧
  ((p a → q a) ∧ ¬(q a → p a)) ∧
  ((r a → q a) ∧ ¬(q a → r a)) :=
sorry

end conditions_implications_l2413_241314


namespace cylinder_volume_l2413_241335

/-- The volume of a cylinder with diameter 4 cm and height 5 cm is equal to π * 20 cm³ -/
theorem cylinder_volume (π : ℝ) (h : π = Real.pi) : 
  let d : ℝ := 4 -- diameter in cm
  let h : ℝ := 5 -- height in cm
  let r : ℝ := d / 2 -- radius in cm
  let v : ℝ := π * r^2 * h -- volume formula
  v = π * 20 := by sorry

end cylinder_volume_l2413_241335


namespace product_value_l2413_241380

def product_term (n : ℕ) : ℚ :=
  (n * (n + 2)) / ((n + 1) * (n + 1))

def product_sequence : ℕ → ℚ
  | 0 => 1
  | n + 1 => product_sequence n * product_term (n + 1)

theorem product_value : product_sequence 98 = 50 / 99 := by
  sorry

end product_value_l2413_241380


namespace ruby_initial_apples_l2413_241342

/-- The number of apples Ruby has initially -/
def initial_apples : ℕ := sorry

/-- The number of apples Emily takes away -/
def apples_taken : ℕ := 55

/-- The number of apples Ruby has left -/
def apples_left : ℕ := 8

/-- Theorem stating that Ruby's initial number of apples is 63 -/
theorem ruby_initial_apples : initial_apples = 63 := by sorry

end ruby_initial_apples_l2413_241342


namespace xy_value_l2413_241365

theorem xy_value (x y : ℝ) (h : Real.sqrt (x + 2) + (y - Real.sqrt 3) ^ 2 = 0) : 
  x * y = -2 * Real.sqrt 3 := by
sorry

end xy_value_l2413_241365


namespace triangle_perimeter_is_five_l2413_241390

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Theorem: If in a triangle ABC, b*cos(A) + a*cos(B) = c^2 and a = b = 2, 
    then the perimeter of the triangle is 5 -/
theorem triangle_perimeter_is_five (t : Triangle) 
  (h1 : t.b * Real.cos t.A + t.a * Real.cos t.B = t.c^2)
  (h2 : t.a = 2)
  (h3 : t.b = 2) : 
  t.a + t.b + t.c = 5 := by
  sorry


end triangle_perimeter_is_five_l2413_241390


namespace distinct_cube_constructions_proof_l2413_241302

/-- The number of distinct ways to construct a 2 × 2 × 2 cube 
    using 6 white unit cubes and 2 black unit cubes, 
    where constructions are considered the same if one can be rotated to match the other -/
def distinct_cube_constructions : ℕ := 3

/-- The total number of unit cubes used -/
def total_cubes : ℕ := 8

/-- The number of white unit cubes -/
def white_cubes : ℕ := 6

/-- The number of black unit cubes -/
def black_cubes : ℕ := 2

/-- The dimensions of the cube -/
def cube_dimensions : Fin 3 → ℕ := λ _ => 2

/-- The order of the rotational symmetry group of a cube -/
def cube_symmetry_order : ℕ := 24

theorem distinct_cube_constructions_proof :
  distinct_cube_constructions = 3 ∧
  total_cubes = white_cubes + black_cubes ∧
  (∀ i, cube_dimensions i = 2) ∧
  cube_symmetry_order = 24 := by
  sorry

end distinct_cube_constructions_proof_l2413_241302
