import Mathlib

namespace playground_ball_cost_l3269_326925

theorem playground_ball_cost (jump_rope_cost board_game_cost savings_from_allowance savings_from_uncle additional_needed : ℕ) :
  jump_rope_cost = 7 →
  board_game_cost = 12 →
  savings_from_allowance = 6 →
  savings_from_uncle = 13 →
  additional_needed = 4 →
  ∃ (playground_ball_cost : ℕ),
    playground_ball_cost = 4 ∧
    jump_rope_cost + board_game_cost + playground_ball_cost = savings_from_allowance + savings_from_uncle + additional_needed :=
by sorry

end playground_ball_cost_l3269_326925


namespace diploma_percentage_theorem_l3269_326935

/-- Represents the four income groups in country Z -/
inductive IncomeGroup
  | Low
  | LowerMiddle
  | UpperMiddle
  | High

/-- Returns the population percentage for a given income group -/
def population_percentage (group : IncomeGroup) : Real :=
  match group with
  | IncomeGroup.Low => 0.25
  | IncomeGroup.LowerMiddle => 0.35
  | IncomeGroup.UpperMiddle => 0.25
  | IncomeGroup.High => 0.15

/-- Returns the percentage of people with a university diploma for a given income group -/
def diploma_percentage (group : IncomeGroup) : Real :=
  match group with
  | IncomeGroup.Low => 0.05
  | IncomeGroup.LowerMiddle => 0.35
  | IncomeGroup.UpperMiddle => 0.60
  | IncomeGroup.High => 0.80

/-- Calculates the total percentage of the population with a university diploma -/
def total_diploma_percentage : Real :=
  (population_percentage IncomeGroup.Low * diploma_percentage IncomeGroup.Low) +
  (population_percentage IncomeGroup.LowerMiddle * diploma_percentage IncomeGroup.LowerMiddle) +
  (population_percentage IncomeGroup.UpperMiddle * diploma_percentage IncomeGroup.UpperMiddle) +
  (population_percentage IncomeGroup.High * diploma_percentage IncomeGroup.High)

theorem diploma_percentage_theorem :
  total_diploma_percentage = 0.405 := by
  sorry

end diploma_percentage_theorem_l3269_326935


namespace min_value_theorem_min_value_achieved_l3269_326927

theorem min_value_theorem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 2 * x + y = 1) :
  x^2 + (1/4) * y^2 ≥ 1/8 := by
sorry

theorem min_value_achieved (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 2 * x + y = 1) :
  (x^2 + (1/4) * y^2 = 1/8) ↔ (x = 1/4 ∧ y = 1/2) := by
sorry

end min_value_theorem_min_value_achieved_l3269_326927


namespace eighteen_games_equation_l3269_326921

/-- The number of games in a competition where each pair of teams plays once. -/
def numGames (x : ℕ) : ℕ := x * (x - 1) / 2

/-- Theorem stating that for x teams, 18 total games is equivalent to the equation 1/2 * x * (x-1) = 18 -/
theorem eighteen_games_equation (x : ℕ) :
  numGames x = 18 ↔ (x * (x - 1)) / 2 = 18 := by sorry

end eighteen_games_equation_l3269_326921


namespace isosceles_triangle_base_length_l3269_326990

/-- An isosceles triangle with integer side lengths and perimeter 10 has a base length of 2 or 4 -/
theorem isosceles_triangle_base_length : 
  ∀ x y : ℕ, 
  x > 0 → y > 0 →
  x + x + y = 10 → 
  y = 2 ∨ y = 4 :=
by sorry

end isosceles_triangle_base_length_l3269_326990


namespace diary_ratio_proof_l3269_326975

def diary_problem (initial_diaries : ℕ) (current_diaries : ℕ) : Prop :=
  let bought_diaries := 2 * initial_diaries
  let total_after_buying := initial_diaries + bought_diaries
  let lost_diaries := total_after_buying - current_diaries
  (lost_diaries : ℚ) / total_after_buying = 1 / 4

theorem diary_ratio_proof :
  diary_problem 8 18 := by
  sorry

end diary_ratio_proof_l3269_326975


namespace sin_plus_two_cos_equals_neg_two_fifths_l3269_326979

/-- Given a point P(-3,4) on the terminal side of angle θ, prove that sin θ + 2cos θ = -2/5 -/
theorem sin_plus_two_cos_equals_neg_two_fifths (θ : ℝ) (P : ℝ × ℝ) :
  P = (-3, 4) →
  (∃ r : ℝ, r > 0 ∧ r * (Real.cos θ) = -3 ∧ r * (Real.sin θ) = 4) →
  Real.sin θ + 2 * Real.cos θ = -2/5 := by
  sorry

end sin_plus_two_cos_equals_neg_two_fifths_l3269_326979


namespace initial_investment_l3269_326939

/-- Proves that the initial investment is 8000 given the specified conditions -/
theorem initial_investment (x : ℝ) : 
  (0.05 * x + 0.08 * 4000 = 0.06 * (x + 4000)) → x = 8000 :=
by
  sorry

end initial_investment_l3269_326939


namespace gcd_of_powers_of_47_plus_one_l3269_326970

theorem gcd_of_powers_of_47_plus_one (h : Prime 47) :
  Nat.gcd (47^5 + 1) (47^5 + 47^3 + 1) = 1 := by
  sorry

end gcd_of_powers_of_47_plus_one_l3269_326970


namespace range_of_x_l3269_326937

theorem range_of_x (x : ℝ) : 
  (¬ (x ∈ Set.Icc 2 5 ∨ x < 1 ∨ x > 4)) → 
  (x ∈ Set.Ioo 1 2) :=
sorry

end range_of_x_l3269_326937


namespace max_rectangles_correct_l3269_326922

/-- The maximum number of 1 × (n + 1) rectangles that can be cut from a 2n × 2n square -/
def max_rectangles (n : ℕ) : ℕ :=
  if n ≥ 4 then 4 * (n - 1)
  else if n = 1 then 2
  else if n = 2 then 5
  else 8

theorem max_rectangles_correct (n : ℕ) :
  max_rectangles n = 
    if n ≥ 4 then 4 * (n - 1)
    else if n = 1 then 2
    else if n = 2 then 5
    else 8 :=
by sorry

end max_rectangles_correct_l3269_326922


namespace tshirts_sold_equals_45_l3269_326948

/-- The number of t-shirts sold by the Razorback t-shirt Shop last week -/
def num_tshirts_sold : ℕ := 45

/-- The price of each t-shirt in dollars -/
def price_per_tshirt : ℕ := 16

/-- The total amount of money made in dollars -/
def total_money_made : ℕ := 720

/-- Theorem: The number of t-shirts sold is equal to 45 -/
theorem tshirts_sold_equals_45 :
  num_tshirts_sold = total_money_made / price_per_tshirt :=
by sorry

end tshirts_sold_equals_45_l3269_326948


namespace square_area_difference_l3269_326964

theorem square_area_difference (x : ℝ) : 
  (x + 2)^2 - x^2 = 32 → x + 2 = 9 :=
by sorry

end square_area_difference_l3269_326964


namespace bank_interest_rate_determination_l3269_326914

/-- Proves that given two equal deposits with the same interest rate but different time periods, 
    if the difference in interest is known, then the interest rate can be determined. -/
theorem bank_interest_rate_determination 
  (principal : ℝ) 
  (time1 time2 : ℝ) 
  (interest_difference : ℝ) : 
  principal = 640 →
  time1 = 3.5 →
  time2 = 5 →
  interest_difference = 144 →
  ∃ (rate : ℝ), 
    (principal * time2 * rate / 100 - principal * time1 * rate / 100 = interest_difference) ∧
    rate = 15 := by
  sorry

end bank_interest_rate_determination_l3269_326914


namespace sum_of_polynomials_l3269_326902

-- Define the polynomials
def f (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def g (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def h (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2

-- Theorem statement
theorem sum_of_polynomials (x : ℝ) : f x + g x + h x = -4 * x^2 + 12 * x - 12 := by
  sorry

end sum_of_polynomials_l3269_326902


namespace quadratic_coefficient_l3269_326956

/-- A quadratic function with vertex (2, -1) passing through (-1, -16) has a = -5/3 -/
theorem quadratic_coefficient (a b c : ℝ) : 
  (∀ x, (a * x^2 + b * x + c) = a * (x - 2)^2 - 1) →  -- vertex form
  (a * (-1)^2 + b * (-1) + c = -16) →                 -- passes through (-1, -16)
  a = -5/3 := by
sorry

end quadratic_coefficient_l3269_326956


namespace red_peaches_per_basket_l3269_326918

/-- Given 6 baskets of peaches with a total of 96 red peaches,
    prove that each basket contains 16 red peaches. -/
theorem red_peaches_per_basket :
  let total_baskets : ℕ := 6
  let total_red_peaches : ℕ := 96
  let green_peaches_per_basket : ℕ := 18
  (total_red_peaches / total_baskets : ℚ) = 16 := by
  sorry

end red_peaches_per_basket_l3269_326918


namespace inflection_points_collinear_l3269_326936

/-- The function f(x) = 9x^5 - 30x^3 + 19x -/
def f (x : ℝ) : ℝ := 9*x^5 - 30*x^3 + 19*x

/-- The inflection points of f(x) -/
def inflection_points : List (ℝ × ℝ) := [(-1, 2), (0, 0), (1, -2)]

/-- Theorem: The inflection points of f(x) are collinear -/
theorem inflection_points_collinear : 
  let points := inflection_points
  ∃ (m c : ℝ), ∀ (x y : ℝ), (x, y) ∈ points → y = m * x + c :=
by sorry

end inflection_points_collinear_l3269_326936


namespace quadratic_form_identity_l3269_326908

theorem quadratic_form_identity 
  (a b c d e f x y z : ℝ) 
  (h : a * x^2 + b * y^2 + c * z^2 + 2 * d * y * z + 2 * e * z * x + 2 * f * x * y = 0) :
  (d * y * z + e * z * x + f * x * y)^2 - b * c * y^2 * z^2 - c * a * z^2 * x^2 - a * b * x^2 * y^2 = 
  (1/4) * (x * Real.sqrt a + y * Real.sqrt b + z * Real.sqrt c) *
          (x * Real.sqrt a - y * Real.sqrt b + z * Real.sqrt c) *
          (x * Real.sqrt a + y * Real.sqrt b - z * Real.sqrt c) *
          (x * Real.sqrt a - y * Real.sqrt b - z * Real.sqrt c) := by
  sorry

end quadratic_form_identity_l3269_326908


namespace solution_set_f_leq_6_range_of_a_for_f_plus_g_geq_3_l3269_326924

-- Define the functions f and g
def f (a x : ℝ) : ℝ := |2*x - a| + a
def g (x : ℝ) : ℝ := |2*x - 1|

-- Theorem for the first part of the problem
theorem solution_set_f_leq_6 :
  {x : ℝ | f 2 x ≤ 6} = {x : ℝ | -1 ≤ x ∧ x ≤ 3} := by sorry

-- Theorem for the second part of the problem
theorem range_of_a_for_f_plus_g_geq_3 :
  {a : ℝ | ∀ x, f a x + g x ≥ 3} = {a : ℝ | a ≥ 2} := by sorry

end solution_set_f_leq_6_range_of_a_for_f_plus_g_geq_3_l3269_326924


namespace area_enclosed_by_curve_and_x_axis_l3269_326903

-- Define the curve function
def f (x : ℝ) : ℝ := 3 - 3 * x^2

-- Theorem statement
theorem area_enclosed_by_curve_and_x_axis : 
  ∫ x in (-1)..1, f x = 4 := by
  sorry

end area_enclosed_by_curve_and_x_axis_l3269_326903


namespace swimming_pool_length_l3269_326916

/-- Given a rectangular swimming pool with width 22 feet, surrounded by a deck of uniform width 3 feet,
    prove that if the total area of the pool and deck is 728 square feet, then the length of the pool is 20 feet. -/
theorem swimming_pool_length (pool_width deck_width total_area : ℝ) : 
  pool_width = 22 →
  deck_width = 3 →
  (pool_width + 2 * deck_width) * (pool_width + 2 * deck_width) = total_area →
  total_area = 728 →
  ∃ pool_length : ℝ, pool_length = 20 ∧ (pool_length + 2 * deck_width) * (pool_width + 2 * deck_width) = total_area :=
by sorry

end swimming_pool_length_l3269_326916


namespace cross_product_result_l3269_326940

def a : ℝ × ℝ × ℝ := (4, 2, -1)
def b : ℝ × ℝ × ℝ := (3, -5, 6)

def cross_product (v w : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (v.2.1 * w.2.2 - v.2.2 * w.2.1,
   v.2.2 * w.1 - v.1 * w.2.2,
   v.1 * w.2.1 - v.2.1 * w.1)

theorem cross_product_result :
  cross_product a b = (7, -27, -26) := by
  sorry

end cross_product_result_l3269_326940


namespace rectangular_prism_sum_l3269_326965

/-- A rectangular prism is a three-dimensional shape with 6 rectangular faces. -/
structure RectangularPrism where
  /-- The number of faces of a rectangular prism -/
  faces : ℕ
  /-- The number of edges of a rectangular prism -/
  edges : ℕ
  /-- The number of vertices of a rectangular prism -/
  vertices : ℕ
  /-- A rectangular prism has 6 faces -/
  face_count : faces = 6
  /-- A rectangular prism has 12 edges -/
  edge_count : edges = 12
  /-- A rectangular prism has 8 vertices -/
  vertex_count : vertices = 8

/-- The sum of faces, edges, and vertices of a rectangular prism is 26 -/
theorem rectangular_prism_sum (rp : RectangularPrism) : 
  rp.faces + rp.edges + rp.vertices = 26 := by
  sorry

end rectangular_prism_sum_l3269_326965


namespace glove_pair_probability_l3269_326985

def num_black_pairs : ℕ := 6
def num_beige_pairs : ℕ := 4

def total_gloves : ℕ := 2 * (num_black_pairs + num_beige_pairs)

def prob_black_pair : ℚ := (num_black_pairs * 2 / total_gloves) * ((num_black_pairs * 2 - 1) / (total_gloves - 1))
def prob_beige_pair : ℚ := (num_beige_pairs * 2 / total_gloves) * ((num_beige_pairs * 2 - 1) / (total_gloves - 1))

theorem glove_pair_probability :
  prob_black_pair + prob_beige_pair = 47 / 95 := by
  sorry

end glove_pair_probability_l3269_326985


namespace clay_capacity_scaling_l3269_326972

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  depth : ℝ
  width : ℝ
  length : ℝ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℝ := d.depth * d.width * d.length

/-- Theorem: Given a box with dimensions 3x4x6 cm holding 60g of clay,
    a box with dimensions 9x16x6 cm will hold 720g of clay -/
theorem clay_capacity_scaling (clayMass₁ : ℝ) :
  let box₁ : BoxDimensions := ⟨3, 4, 6⟩
  let box₂ : BoxDimensions := ⟨9, 16, 6⟩
  clayMass₁ = 60 →
  (boxVolume box₂ / boxVolume box₁) * clayMass₁ = 720 := by
  sorry

end clay_capacity_scaling_l3269_326972


namespace range_of_f_l3269_326906

-- Define the function
def f (x : ℝ) : ℝ := |x + 1| - |x - 2|

-- State the theorem
theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ -3 ≤ y ∧ y ≤ 3 := by sorry

end range_of_f_l3269_326906


namespace tower_arrangements_eq_4200_l3269_326919

/-- The number of ways to arrange 9 cubes out of 10 cubes (3 red, 3 blue, 4 green) -/
def tower_arrangements : ℕ := 
  Nat.choose 10 9 * (Nat.factorial 9 / (Nat.factorial 3 * Nat.factorial 3 * Nat.factorial 3))

/-- Theorem stating that the number of tower arrangements is 4200 -/
theorem tower_arrangements_eq_4200 : tower_arrangements = 4200 := by
  sorry

end tower_arrangements_eq_4200_l3269_326919


namespace matrix_not_invertible_sum_fractions_l3269_326996

theorem matrix_not_invertible_sum_fractions (a b c : ℝ) :
  let M := !![a, b, c; b, c, a; c, a, b]
  ¬(IsUnit (Matrix.det M)) →
  (a / (b + c) + b / (a + c) + c / (a + b) = -3) ∨
  (a / (b + c) + b / (a + c) + c / (a + b) = 3/2) :=
by sorry

end matrix_not_invertible_sum_fractions_l3269_326996


namespace max_gcd_consecutive_terms_l3269_326988

def a (n : ℕ) : ℕ := n.factorial + n^2

theorem max_gcd_consecutive_terms : 
  ∃ (k : ℕ), (∀ (n : ℕ), Nat.gcd (a n) (a (n + 1)) ≤ k) ∧ 
             (∃ (m : ℕ), Nat.gcd (a m) (a (m + 1)) = k) ∧ 
             k = 2 := by
  sorry

end max_gcd_consecutive_terms_l3269_326988


namespace simplify_sqrt_difference_l3269_326969

theorem simplify_sqrt_difference : Real.sqrt 18 - Real.sqrt 8 = Real.sqrt 2 := by
  sorry

end simplify_sqrt_difference_l3269_326969


namespace probability_defective_from_A_l3269_326995

-- Define the probabilities
def prob_factory_A : ℝ := 0.45
def prob_factory_B : ℝ := 0.55
def defect_rate_A : ℝ := 0.06
def defect_rate_B : ℝ := 0.05

-- Theorem statement
theorem probability_defective_from_A : 
  let prob_defective := prob_factory_A * defect_rate_A + prob_factory_B * defect_rate_B
  prob_factory_A * defect_rate_A / prob_defective = 54 / 109 := by
sorry

end probability_defective_from_A_l3269_326995


namespace poverty_decline_rate_l3269_326920

/-- The annual average decline rate of impoverished people -/
def annual_decline_rate : ℝ := 0.5

/-- The initial number of impoverished people in 2018 -/
def initial_population : ℕ := 40000

/-- The number of impoverished people in 2020 -/
def final_population : ℕ := 10000

/-- The time period in years -/
def time_period : ℕ := 2

theorem poverty_decline_rate :
  (↑initial_population * (1 - annual_decline_rate) ^ time_period = ↑final_population) ∧
  (0 < annual_decline_rate) ∧
  (annual_decline_rate < 1) := by
  sorry

end poverty_decline_rate_l3269_326920


namespace rectangle_area_l3269_326981

/-- A rectangle with specific properties -/
structure Rectangle where
  width : ℝ
  length : ℝ
  length_exceed_twice_width : length = 2 * width + 25
  perimeter_650 : 2 * (length + width) = 650

/-- The area of a rectangle with the given properties is 22500 -/
theorem rectangle_area (r : Rectangle) : r.length * r.width = 22500 := by
  sorry

end rectangle_area_l3269_326981


namespace will_baseball_card_pages_l3269_326951

/-- Calculates the number of pages needed to arrange baseball cards. -/
def pages_needed (cards_per_page : ℕ) (cards_2020 : ℕ) (cards_2015_2019 : ℕ) (duplicates : ℕ) : ℕ :=
  let unique_2020 := cards_2020
  let unique_2015_2019 := cards_2015_2019 - duplicates
  let pages_2020 := (unique_2020 + cards_per_page - 1) / cards_per_page
  let pages_2015_2019 := (unique_2015_2019 + cards_per_page - 1) / cards_per_page
  pages_2020 + pages_2015_2019

/-- Theorem stating the number of pages needed for Will's baseball card arrangement. -/
theorem will_baseball_card_pages :
  pages_needed 3 8 10 2 = 6 := by
  sorry

end will_baseball_card_pages_l3269_326951


namespace product_of_numbers_l3269_326909

theorem product_of_numbers (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : y / x = 15) (h4 : x + y = 400) : x * y = 9375 := by
  sorry

end product_of_numbers_l3269_326909


namespace grandma_crane_folding_l3269_326991

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : Nat) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + m
  { hours := totalMinutes / 60, minutes := totalMinutes % 60 }

theorem grandma_crane_folding :
  let foldTime : Nat := 3  -- Time to fold one crane
  let restTime : Nat := 1  -- Rest time after folding each crane
  let startTime : Time := { hours := 14, minutes := 30 }  -- 2:30 PM
  let numCranes : Nat := 5
  
  let totalFoldTime := foldTime * numCranes
  let totalRestTime := restTime * (numCranes - 1)
  let totalTime := totalFoldTime + totalRestTime
  
  addMinutes startTime totalTime = { hours := 14, minutes := 49 }  -- 2:49 PM
  := by sorry

end grandma_crane_folding_l3269_326991


namespace sin_ratio_comparison_l3269_326968

theorem sin_ratio_comparison :
  (Real.sin (3 * Real.pi / 180)) / (Real.sin (4 * Real.pi / 180)) >
  (Real.sin (1 * Real.pi / 180)) / (Real.sin (2 * Real.pi / 180)) :=
by sorry

end sin_ratio_comparison_l3269_326968


namespace intersection_point_property_l3269_326944

theorem intersection_point_property (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  b = -2 / a ∧ b = a + 3 → 1 / a - 1 / b = -3 / 2 := by
  sorry

end intersection_point_property_l3269_326944


namespace exists_fib_with_three_trailing_zeros_l3269_326942

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib n + fib (n + 1)

-- State the theorem
theorem exists_fib_with_three_trailing_zeros :
  ∃ n : ℕ, fib n % 1000 = 0 ∧ fib (n + 1) % 1000 = 0 ∧ fib (n + 2) % 1000 = 0 := by
  sorry


end exists_fib_with_three_trailing_zeros_l3269_326942


namespace binary_11001_is_25_l3269_326917

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_11001_is_25 :
  binary_to_decimal [true, false, false, true, true] = 25 := by
  sorry

end binary_11001_is_25_l3269_326917


namespace simplify_expressions_l3269_326967

theorem simplify_expressions :
  (3 * Real.sqrt 20 - Real.sqrt 45 + Real.sqrt (1/5) = 16 * Real.sqrt 5 / 5) ∧
  ((Real.sqrt 6 - 2 * Real.sqrt 3)^2 - (2 * Real.sqrt 5 + Real.sqrt 2) * (2 * Real.sqrt 5 - Real.sqrt 2) = -12 * Real.sqrt 2) := by
  sorry

end simplify_expressions_l3269_326967


namespace percent_of_x_l3269_326931

theorem percent_of_x (x y z : ℝ) 
  (h1 : 0.6 * (x - y) = 0.3 * (x + y + z)) 
  (h2 : 0.4 * (y - z) = 0.2 * (y + x - z)) : 
  y - z = x := by sorry

end percent_of_x_l3269_326931


namespace simplify_expression_a_l3269_326980

theorem simplify_expression_a (x a b : ℝ) :
  (3 * x^2 * (a^2 + b^2) - 3 * a^2 * b^2 + 3 * (x^2 + (a + b) * x + a * b) * (x * (x - a) - b * (x - a))) / x^2 = 3 * x^2 := by
  sorry

end simplify_expression_a_l3269_326980


namespace min_value_theorem_l3269_326974

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 1/y = 1) :
  1/(x-1) + 3/(y-1) ≥ 2 * Real.sqrt 3 :=
by sorry

end min_value_theorem_l3269_326974


namespace arithmetic_mean_of_fractions_l3269_326910

theorem arithmetic_mean_of_fractions : 
  (3 / 7 + 5 / 8) / 2 = 59 / 112 := by sorry

end arithmetic_mean_of_fractions_l3269_326910


namespace salary_percentage_increase_l3269_326997

theorem salary_percentage_increase 
  (original : ℝ) 
  (decrease_percent : ℝ) 
  (increase_percent : ℝ) 
  (overall_decrease_percent : ℝ) 
  (h1 : decrease_percent = 50) 
  (h2 : overall_decrease_percent = 35) 
  (h3 : original * (1 - decrease_percent / 100) * (1 + increase_percent / 100) = 
        original * (1 - overall_decrease_percent / 100)) : 
  increase_percent = 30 := by
sorry

end salary_percentage_increase_l3269_326997


namespace pascal_row_10_sum_l3269_326977

/-- The sum of the numbers in a row of Pascal's Triangle -/
def pascal_row_sum (n : ℕ) : ℕ := 2^n

/-- Theorem: The sum of the numbers in Row 10 of Pascal's Triangle is 1024 -/
theorem pascal_row_10_sum : pascal_row_sum 10 = 1024 := by
  sorry

end pascal_row_10_sum_l3269_326977


namespace intersection_of_M_and_N_l3269_326982

-- Define the sets M and N
def M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}
def N : Set ℝ := {x | x < 1}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = {x : ℝ | -2 ≤ x ∧ x < 1} := by sorry

end intersection_of_M_and_N_l3269_326982


namespace tan_product_lower_bound_l3269_326954

theorem tan_product_lower_bound (α β γ : Real) 
  (h_acute_α : 0 < α ∧ α < Real.pi / 2)
  (h_acute_β : 0 < β ∧ β < Real.pi / 2)
  (h_acute_γ : 0 < γ ∧ γ < Real.pi / 2)
  (h_cos_sum : Real.cos α ^ 2 + Real.cos β ^ 2 + Real.cos γ ^ 2 = 1) :
  Real.tan α * Real.tan β * Real.tan γ ≥ 2 * Real.sqrt 2 := by
  sorry

end tan_product_lower_bound_l3269_326954


namespace ellipse_k_range_l3269_326993

/-- The equation of an ellipse with parameter k -/
def ellipse_equation (x y k : ℝ) : Prop :=
  x^2 / (5 - k) + y^2 / (k - 3) = 1

/-- Conditions for the equation to represent an ellipse -/
def is_ellipse (k : ℝ) : Prop :=
  5 - k > 0 ∧ k - 3 > 0 ∧ 5 - k ≠ k - 3

/-- The range of k for which the equation represents an ellipse -/
theorem ellipse_k_range :
  ∀ k : ℝ, is_ellipse k ↔ (3 < k ∧ k < 5 ∧ k ≠ 4) :=
by sorry

end ellipse_k_range_l3269_326993


namespace blue_sock_pairs_l3269_326911

theorem blue_sock_pairs (n : ℕ) (k : ℕ) : n = 4 ∧ k = 2 → Nat.choose n k = 6 := by
  sorry

end blue_sock_pairs_l3269_326911


namespace complex_equation_solution_l3269_326992

theorem complex_equation_solution (z : ℂ) : (Complex.I * z = 4 + 3 * Complex.I) → z = 3 - 4 * Complex.I := by
  sorry

end complex_equation_solution_l3269_326992


namespace bag_contents_theorem_l3269_326915

/-- Represents the contents of a bag of colored balls. -/
structure BagContents where
  red : ℕ
  yellow : ℕ
  green : ℕ

/-- Calculates the probability of selecting two red balls. -/
def probTwoRed (bag : BagContents) : ℚ :=
  (bag.red.choose 2 : ℚ) / ((bag.red + bag.yellow + bag.green).choose 2 : ℚ)

/-- Calculates the probability of selecting one red and one yellow ball. -/
def probRedYellow (bag : BagContents) : ℚ :=
  ((bag.red.choose 1 * bag.yellow.choose 1) : ℚ) / ((bag.red + bag.yellow + bag.green).choose 2 : ℚ)

/-- Calculates the expected value of the number of red balls selected. -/
def expectedRedBalls (bag : BagContents) : ℚ :=
  0 * ((bag.yellow + bag.green).choose 2 : ℚ) / ((bag.red + bag.yellow + bag.green).choose 2 : ℚ) +
  1 * ((bag.red.choose 1 * (bag.yellow + bag.green).choose 1) : ℚ) / ((bag.red + bag.yellow + bag.green).choose 2 : ℚ) +
  2 * (bag.red.choose 2 : ℚ) / ((bag.red + bag.yellow + bag.green).choose 2 : ℚ)

/-- The main theorem stating the properties of the bag contents and expected value. -/
theorem bag_contents_theorem (bag : BagContents) :
  bag.red = 4 ∧ 
  probTwoRed bag = 1/6 ∧ 
  probRedYellow bag = 1/3 → 
  bag.yellow - bag.green = 1 ∧ 
  expectedRedBalls bag = 8/9 := by
  sorry


end bag_contents_theorem_l3269_326915


namespace intersection_of_A_and_B_l3269_326949

-- Define the sets A and B
def A : Set ℝ := {x | -2 < x ∧ x < 1}
def B : Set ℝ := {x | x * (x - 3) > 0}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | -2 < x ∧ x < 0} := by sorry

end intersection_of_A_and_B_l3269_326949


namespace geometric_sequence_sum_l3269_326933

/-- A geometric sequence with positive terms. -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25 →
  a 3 + a 5 = 5 := by
  sorry

end geometric_sequence_sum_l3269_326933


namespace ellipse_theorem_l3269_326923

/-- Represents an ellipse in standard form -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line passing through two points -/
structure Line where
  p1 : Point
  p2 : Point

/-- Represents a circle defined by its center and radius -/
structure Circle where
  center : Point
  radius : ℝ

/-- Main theorem about the ellipse and the maximum product -/
theorem ellipse_theorem (C : Ellipse) (P Q : Point) (l : Line) (F : Point) (circle : Circle) :
  (P.x = 1 ∧ P.y = Real.sqrt 2 / 2) →
  (Q.x = -Real.sqrt 2 ∧ Q.y = 0) →
  (C.a^2 * P.y^2 + C.b^2 * P.x^2 = C.a^2 * C.b^2) →
  (C.a^2 * Q.y^2 + C.b^2 * Q.x^2 = C.a^2 * C.b^2) →
  (∃ (A B E : Point), A ≠ B ∧ E ≠ F ∧
    (C.a^2 * A.y^2 + C.b^2 * A.x^2 = C.a^2 * C.b^2) ∧
    (C.a^2 * B.y^2 + C.b^2 * B.x^2 = C.a^2 * C.b^2) ∧
    (l.p1 = F ∧ l.p2 = A) ∧
    ((E.x - circle.center.x)^2 + (E.y - circle.center.y)^2 = circle.radius^2)) →
  (C.a = Real.sqrt 2 ∧ C.b = 1) ∧
  (∀ (A B E : Point), 
    (C.a^2 * A.y^2 + C.b^2 * A.x^2 = C.a^2 * C.b^2) →
    (C.a^2 * B.y^2 + C.b^2 * B.x^2 = C.a^2 * C.b^2) →
    ((E.x - circle.center.x)^2 + (E.y - circle.center.y)^2 = circle.radius^2) →
    Real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2) * Real.sqrt ((F.x - E.x)^2 + (F.y - E.y)^2) ≤ 1) :=
by sorry

end ellipse_theorem_l3269_326923


namespace cube_edge_sum_length_l3269_326952

/-- The sum of the lengths of all edges of a cube with edge length 15 cm is 180 cm. -/
theorem cube_edge_sum_length (edge_length : ℝ) (num_edges : ℕ) : 
  edge_length = 15 → num_edges = 12 → edge_length * num_edges = 180 := by
  sorry

end cube_edge_sum_length_l3269_326952


namespace division_result_l3269_326926

theorem division_result : (0.0204 : ℝ) / 17 = 0.0012 := by
  sorry

end division_result_l3269_326926


namespace system_nonzero_solution_iff_condition_l3269_326978

/-- The system of equations has a non-zero solution iff 2abc + ab + bc + ca - 1 = 0 -/
theorem system_nonzero_solution_iff_condition (a b c : ℝ) :
  (∃ x y z : ℝ, (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0) ∧
    x = b * y + c * z ∧
    y = c * z + a * x ∧
    z = a * x + b * y) ↔
  2 * a * b * c + a * b + b * c + c * a - 1 = 0 := by
sorry

end system_nonzero_solution_iff_condition_l3269_326978


namespace max_non_managers_l3269_326958

/-- Represents the number of managers in a department -/
def managers : ℕ := 11

/-- Represents the ratio of managers to non-managers -/
def ratio : ℚ := 7 / 37

/-- Theorem stating the maximum number of non-managers in a department -/
theorem max_non_managers :
  ∀ n : ℕ, (managers : ℚ) / n > ratio → n ≤ 58 :=
sorry

end max_non_managers_l3269_326958


namespace smallest_n_value_l3269_326928

theorem smallest_n_value (N : ℕ) (c₁ c₂ c₃ c₄ : ℕ) : 
  (c₁ ≤ N) ∧ (c₂ ≤ N) ∧ (c₃ ≤ N) ∧ (c₄ ≤ N) ∧ 
  (c₁ = 4 * c₂ - 3) ∧
  (N + c₂ = 4 * c₄) ∧
  (2 * N + c₃ = 4 * c₃ - 1) ∧
  (3 * N + c₄ = 4 * c₁ - 3) →
  N = 1 ∧ c₁ = 1 ∧ c₂ = 1 ∧ c₃ = 1 ∧ c₄ = 1 :=
by sorry

end smallest_n_value_l3269_326928


namespace intersection_equality_condition_l3269_326963

theorem intersection_equality_condition (M N P : Set α) :
  (∀ (M N P : Set α), M = N → M ∩ P = N ∩ P) ∧
  (∃ (M N P : Set α), M ∩ P = N ∩ P ∧ M ≠ N) :=
sorry

end intersection_equality_condition_l3269_326963


namespace hyperbola_foci_distance_l3269_326989

/-- The distance between the foci of a hyperbola with equation xy = 4 is 8 -/
theorem hyperbola_foci_distance :
  ∃ (t : ℝ), t > 0 ∧
  (∀ (x y : ℝ), x * y = 4 →
    ∃ (d : ℝ), d > 0 ∧
    ∀ (P : ℝ × ℝ), P.1 * P.2 = 4 →
      Real.sqrt ((P.1 + t)^2 + (P.2 + t)^2) - Real.sqrt ((P.1 - t)^2 + (P.2 - t)^2) = d) →
  Real.sqrt ((t + t)^2 + (t + t)^2) = 8 :=
by sorry


end hyperbola_foci_distance_l3269_326989


namespace song_storage_size_l3269_326930

-- Define the given values
def total_storage : ℕ := 16  -- in GB
def used_storage : ℕ := 4    -- in GB
def num_songs : ℕ := 400
def mb_per_gb : ℕ := 1000

-- Define the theorem
theorem song_storage_size :
  let available_storage : ℕ := total_storage - used_storage
  let available_storage_mb : ℕ := available_storage * mb_per_gb
  available_storage_mb / num_songs = 30 := by sorry

end song_storage_size_l3269_326930


namespace max_x_on_circle_l3269_326994

/-- The maximum x-coordinate of a point on the circle (x-10)^2 + (y-30)^2 = 100 is 20. -/
theorem max_x_on_circle : 
  ∀ x y : ℝ, (x - 10)^2 + (y - 30)^2 = 100 → x ≤ 20 :=
by sorry

end max_x_on_circle_l3269_326994


namespace sufficient_condition_l3269_326976

theorem sufficient_condition (x y : ℝ) : x^2 + y^2 < 4 → x*y + 4 > 2*x + 2*y := by
  sorry

end sufficient_condition_l3269_326976


namespace jogging_distance_l3269_326947

/-- Calculates the total distance jogged over a period of days given a constant speed and daily jogging time. -/
def total_distance_jogged (speed : ℝ) (hours_per_day : ℝ) (days : ℕ) : ℝ :=
  speed * hours_per_day * days

/-- Proves that jogging at 5 miles per hour for 2 hours a day for 5 days results in a total distance of 50 miles. -/
theorem jogging_distance : total_distance_jogged 5 2 5 = 50 := by
  sorry

end jogging_distance_l3269_326947


namespace rhombus_area_l3269_326998

/-- The area of a rhombus with side length √145 and diagonals differing by 10 units is 100 square units. -/
theorem rhombus_area (side_length : ℝ) (diagonal_difference : ℝ) (area : ℝ) :
  side_length = Real.sqrt 145 →
  diagonal_difference = 10 →
  area = 100 →
  ∃ (d1 d2 : ℝ), d1 > 0 ∧ d2 > 0 ∧ 
    d2 - d1 = diagonal_difference ∧
    d1 * d2 / 2 = area ∧
    d1^2 / 4 + d2^2 / 4 = side_length^2 :=
by sorry

end rhombus_area_l3269_326998


namespace find_number_B_l3269_326983

/-- Given that A = 5 and A = 2.8B - 0.6, prove that B = 2 -/
theorem find_number_B (A B : ℝ) (h1 : A = 5) (h2 : A = 2.8 * B - 0.6) : B = 2 := by
  sorry

end find_number_B_l3269_326983


namespace equation_solutions_l3269_326984

theorem equation_solutions :
  (∀ x, x^2 - 8*x - 1 = 0 ↔ x = 4 + Real.sqrt 17 ∨ x = 4 - Real.sqrt 17) ∧
  (∀ x, x*(2*x - 5) = 4*x - 10 ↔ x = 5/2 ∨ x = 2) := by
  sorry

end equation_solutions_l3269_326984


namespace divisibility_by_1001_l3269_326971

theorem divisibility_by_1001 (n : ℤ) : n ≡ 300^3000 [ZMOD 1001] → n ≡ 1 [ZMOD 1001] := by
  sorry

end divisibility_by_1001_l3269_326971


namespace coin_division_problem_l3269_326953

theorem coin_division_problem :
  ∃ n : ℕ, 
    n > 0 ∧
    n % 8 = 5 ∧
    n % 7 = 4 ∧
    (∀ m : ℕ, m > 0 → m % 8 = 5 → m % 7 = 4 → n ≤ m) ∧
    n % 9 = 8 :=
by
  -- Proof goes here
  sorry

end coin_division_problem_l3269_326953


namespace hyperbola_focus_l3269_326904

/-- Given a hyperbola with equation x^2 - ky^2 = 1 and one focus at (3,0), prove that k = 1/8 -/
theorem hyperbola_focus (k : ℝ) : 
  (∀ x y : ℝ, x^2 - k*y^2 = 1 → (∃ c : ℝ, c^2 = 9 ∧ c^2 = 1 + 1/k)) → 
  k = 1/8 := by sorry

end hyperbola_focus_l3269_326904


namespace exponential_inequality_l3269_326938

theorem exponential_inequality (x y a b : ℝ) 
  (h1 : x > y) (h2 : y > 1) 
  (h3 : 0 < a) (h4 : a < b) (h5 : b < 1) : 
  a^x < b^y := by
  sorry

end exponential_inequality_l3269_326938


namespace zero_point_in_interval_l3269_326955

noncomputable def f (x : ℝ) : ℝ := Real.log x - (1/2)^(x-2)

theorem zero_point_in_interval :
  ∃ x₀ ∈ Set.Ioo 2 3, f x₀ = 0 :=
by
  sorry

end zero_point_in_interval_l3269_326955


namespace negation_equivalence_l3269_326934

theorem negation_equivalence (p q : Prop) : 
  let m := p ∧ q
  (¬p ∨ ¬q) ↔ ¬m := by sorry

end negation_equivalence_l3269_326934


namespace company_french_speakers_l3269_326961

theorem company_french_speakers 
  (total_employees : ℝ) 
  (total_employees_positive : 0 < total_employees) :
  let men_percentage : ℝ := 65 / 100
  let women_percentage : ℝ := 1 - men_percentage
  let men_french_speakers_percentage : ℝ := 60 / 100
  let women_non_french_speakers_percentage : ℝ := 97.14285714285714 / 100
  let men_count : ℝ := men_percentage * total_employees
  let women_count : ℝ := women_percentage * total_employees
  let men_french_speakers : ℝ := men_french_speakers_percentage * men_count
  let women_french_speakers : ℝ := (1 - women_non_french_speakers_percentage) * women_count
  let total_french_speakers : ℝ := men_french_speakers + women_french_speakers
  let french_speakers_percentage : ℝ := total_french_speakers / total_employees * 100
  french_speakers_percentage = 40 := by
sorry


end company_french_speakers_l3269_326961


namespace percentage_problem_l3269_326932

theorem percentage_problem : ∃ p : ℝ, (p / 100) * 16 = 0.04 ∧ p = 0.25 := by
  sorry

end percentage_problem_l3269_326932


namespace complex_absolute_value_squared_l3269_326929

theorem complex_absolute_value_squared : 
  (Complex.abs (-3 - (8/5)*Complex.I))^2 = 289/25 := by
  sorry

end complex_absolute_value_squared_l3269_326929


namespace manny_marbles_l3269_326912

/-- Given a total of 120 marbles distributed in the ratio 4:5:6,
    prove that the person with the middle ratio (5) receives 40 marbles. -/
theorem manny_marbles (total : ℕ) (ratio_sum : ℕ) (manny_ratio : ℕ) :
  total = 120 →
  ratio_sum = 4 + 5 + 6 →
  manny_ratio = 5 →
  manny_ratio * (total / ratio_sum) = 40 :=
by sorry

end manny_marbles_l3269_326912


namespace course_selection_theorem_l3269_326905

def type_a_courses : ℕ := 3
def type_b_courses : ℕ := 4
def total_courses_to_choose : ℕ := 3

def ways_to_choose (n k : ℕ) : ℕ := Nat.choose n k

theorem course_selection_theorem :
  (ways_to_choose type_a_courses 2 * ways_to_choose type_b_courses 1) +
  (ways_to_choose type_a_courses 1 * ways_to_choose type_b_courses 2) = 30 := by
  sorry

end course_selection_theorem_l3269_326905


namespace race_car_cost_l3269_326907

theorem race_car_cost (mater_cost sally_cost race_car_cost : ℝ) : 
  mater_cost = 0.1 * race_car_cost →
  sally_cost = 3 * mater_cost →
  sally_cost = 42000 →
  race_car_cost = 140000 := by
sorry

end race_car_cost_l3269_326907


namespace markup_rate_proof_l3269_326959

theorem markup_rate_proof (S : ℝ) (h_positive : S > 0) : 
  let profit_rate : ℝ := 0.20
  let expense_rate : ℝ := 0.10
  let C : ℝ := S * (1 - profit_rate - expense_rate)
  ((S - C) / C) * 100 = 42.857 := by
sorry

end markup_rate_proof_l3269_326959


namespace base_6_number_identification_l3269_326900

def is_base_6_digit (d : ℕ) : Prop := d < 6

def is_base_6_number (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → is_base_6_digit d

theorem base_6_number_identification :
  ¬ is_base_6_number 66 ∧
  ¬ is_base_6_number 207 ∧
  ¬ is_base_6_number 652 ∧
  is_base_6_number 3142 :=
sorry

end base_6_number_identification_l3269_326900


namespace lydia_porch_flowers_l3269_326901

/-- The number of flowers on Lydia's porch --/
def flowers_on_porch (total_plants : ℕ) (flowering_percent : ℚ) 
  (seven_flower_percent : ℚ) (seven_flower_plants : ℕ) (four_flower_plants : ℕ) : ℕ :=
  seven_flower_plants * 7 + four_flower_plants * 4

/-- Theorem stating the number of flowers on Lydia's porch --/
theorem lydia_porch_flowers :
  flowers_on_porch 120 (35/100) (60/100) 8 6 = 80 := by
  sorry

end lydia_porch_flowers_l3269_326901


namespace true_propositions_l3269_326962

-- Define the four propositions
def proposition1 : Prop := sorry
def proposition2 : Prop := sorry
def proposition3 : Prop := sorry
def proposition4 : Prop := sorry

-- Theorem stating which propositions are true
theorem true_propositions : 
  (¬ proposition1) ∧ proposition2 ∧ proposition3 ∧ (¬ proposition4) := by
  sorry

end true_propositions_l3269_326962


namespace rain_both_days_l3269_326986

-- Define the probabilities
def prob_rain_monday : ℝ := 0.62
def prob_rain_tuesday : ℝ := 0.54
def prob_no_rain : ℝ := 0.28

-- Theorem statement
theorem rain_both_days :
  let prob_rain_both := prob_rain_monday + prob_rain_tuesday - (1 - prob_no_rain)
  prob_rain_both = 0.44 := by sorry

end rain_both_days_l3269_326986


namespace difference_of_squares_103_97_l3269_326966

theorem difference_of_squares_103_97 : 
  |((103 : ℚ) / 2)^2 - ((97 : ℚ) / 2)^2| = 300 := by
  sorry

end difference_of_squares_103_97_l3269_326966


namespace otts_money_fraction_l3269_326941

theorem otts_money_fraction (moe loki nick ott : ℚ) : 
  moe > 0 → loki > 0 → nick > 0 → ott = 0 →
  ∃ (x : ℚ), x > 0 ∧ 
    x = moe / 3 ∧ 
    x = loki / 5 ∧ 
    x = nick / 4 →
  (3 * x) / (moe + loki + nick + 3 * x) = 1 / 4 := by
  sorry

end otts_money_fraction_l3269_326941


namespace womens_tennis_handshakes_l3269_326960

theorem womens_tennis_handshakes (n : ℕ) (k : ℕ) (h1 : n = 4) (h2 : k = 2) : 
  (n * k * (n * k - k)) / 2 = 24 := by
  sorry

end womens_tennis_handshakes_l3269_326960


namespace science_fiction_section_pages_l3269_326913

/-- The number of books in the science fiction section -/
def num_books : ℕ := 8

/-- The number of pages in each book -/
def pages_per_book : ℕ := 478

/-- The total number of pages in the science fiction section -/
def total_pages : ℕ := num_books * pages_per_book

theorem science_fiction_section_pages :
  total_pages = 3824 := by sorry

end science_fiction_section_pages_l3269_326913


namespace g_of_5_equals_15_l3269_326945

-- Define the function g
def g (x : ℝ) : ℝ := x^2 - 2*x

-- State the theorem
theorem g_of_5_equals_15 : g 5 = 15 := by
  sorry

end g_of_5_equals_15_l3269_326945


namespace problem_solution_l3269_326950

theorem problem_solution : (1 / ((-5^4)^2)) * (-5)^9 = -5 := by
  sorry

end problem_solution_l3269_326950


namespace total_spent_is_thirteen_l3269_326946

-- Define the cost of items
def candy_bar_cost : ℕ := 7
def chocolate_cost : ℕ := 6

-- Define the total spent
def total_spent : ℕ := candy_bar_cost + chocolate_cost

-- Theorem to prove
theorem total_spent_is_thirteen : total_spent = 13 := by
  sorry

end total_spent_is_thirteen_l3269_326946


namespace population_difference_l3269_326973

/-- Given that the sum of populations of City A and City B exceeds the sum of populations
    of City B and City C by 5000, prove that the population of City A exceeds
    the population of City C by 5000. -/
theorem population_difference (A B C : ℕ) 
  (h : A + B = B + C + 5000) : A - C = 5000 := by
  sorry

end population_difference_l3269_326973


namespace trigonometric_sum_trigonometric_fraction_l3269_326957

-- Part 1
theorem trigonometric_sum : 
  Real.cos (9 * Real.pi / 4) + Real.tan (-Real.pi / 4) + Real.sin (21 * Real.pi) = Real.sqrt 2 / 2 - 1 :=
by sorry

-- Part 2
theorem trigonometric_fraction (θ : Real) (h : Real.sin θ = 2 * Real.cos θ) : 
  (Real.sin θ ^ 2 + 2 * Real.sin θ * Real.cos θ) / (2 * Real.sin θ ^ 2 - Real.cos θ ^ 2) = 8 / 7 :=
by sorry

end trigonometric_sum_trigonometric_fraction_l3269_326957


namespace margarets_mean_score_l3269_326987

def scores : List ℝ := [82, 85, 89, 91, 95, 97]

theorem margarets_mean_score (cyprians_mean : ℝ) (h1 : cyprians_mean = 88) :
  let total_sum := scores.sum
  let cyprians_sum := 3 * cyprians_mean
  let margarets_sum := total_sum - cyprians_sum
  margarets_sum / 3 = 91 + 2/3 := by sorry

end margarets_mean_score_l3269_326987


namespace jade_ball_problem_l3269_326943

/-- Represents the state of boxes as a list of natural numbers (0-6) -/
def BoxState := List Nat

/-- Converts a natural number to its base-7 representation -/
def toBase7 (n : Nat) : BoxState :=
  sorry

/-- Counts the number of carries (resets) needed to increment from 1 to n in base 7 -/
def countCarries (n : Nat) : Nat :=
  sorry

/-- Sums the digits in a BoxState -/
def sumDigits (state : BoxState) : Nat :=
  sorry

theorem jade_ball_problem (n : Nat) : 
  n = 1876 → 
  sumDigits (toBase7 n) = 10 ∧ 
  countCarries n = 3 := by
  sorry

end jade_ball_problem_l3269_326943


namespace expected_heads_is_60_l3269_326999

/-- The number of coins --/
def num_coins : ℕ := 64

/-- The maximum number of tosses for each coin --/
def max_tosses : ℕ := 4

/-- The probability of getting heads on a single toss --/
def p_heads : ℚ := 1/2

/-- The probability of getting heads after up to four tosses --/
def p_heads_four_tosses : ℚ := 
  p_heads + (1 - p_heads) * p_heads + (1 - p_heads)^2 * p_heads + (1 - p_heads)^3 * p_heads

/-- The expected number of coins showing heads after up to four tosses --/
def expected_heads : ℚ := num_coins * p_heads_four_tosses

theorem expected_heads_is_60 : expected_heads = 60 := by sorry

end expected_heads_is_60_l3269_326999
