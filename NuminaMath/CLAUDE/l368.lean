import Mathlib

namespace circle_diameter_from_inscribed_triangles_l368_36808

theorem circle_diameter_from_inscribed_triangles
  (triangle_a_side1 triangle_a_side2 triangle_a_hypotenuse : ℝ)
  (triangle_b_side1 triangle_b_side2 triangle_b_hypotenuse : ℝ)
  (h1 : triangle_a_side1 = 7)
  (h2 : triangle_a_side2 = 24)
  (h3 : triangle_a_hypotenuse = 39)
  (h4 : triangle_b_side1 = 15)
  (h5 : triangle_b_side2 = 36)
  (h6 : triangle_b_hypotenuse = 39)
  (h7 : triangle_a_side1^2 + triangle_a_side2^2 = triangle_a_hypotenuse^2)
  (h8 : triangle_b_side1^2 + triangle_b_side2^2 = triangle_b_hypotenuse^2)
  (h9 : triangle_a_hypotenuse = triangle_b_hypotenuse) :
  39 = triangle_a_hypotenuse ∧ 39 = triangle_b_hypotenuse := by
  sorry

#check circle_diameter_from_inscribed_triangles

end circle_diameter_from_inscribed_triangles_l368_36808


namespace f_maximum_l368_36866

/-- The quadratic function f(x) = -2x^2 + 8x - 6 -/
def f (x : ℝ) : ℝ := -2 * x^2 + 8 * x - 6

/-- The point where the maximum occurs -/
def x_max : ℝ := 2

theorem f_maximum :
  ∀ x : ℝ, f x ≤ f x_max :=
by sorry

end f_maximum_l368_36866


namespace sequences_count_l368_36885

/-- The number of students in the class -/
def num_students : ℕ := 15

/-- The number of class meetings per week -/
def meetings_per_week : ℕ := 3

/-- The number of different sequences of selecting 3 students from a group of 15 students,
    where each student can be selected at most once -/
def num_sequences : ℕ :=
  num_students * (num_students - 1) * (num_students - 2)

theorem sequences_count :
  num_sequences = 2730 := by
  sorry

end sequences_count_l368_36885


namespace last_remaining_200_l368_36884

/-- The last remaining number after the marking process -/
def lastRemainingNumber (n : ℕ) : ℕ :=
  if n ≤ 1 then n else 2 * lastRemainingNumber ((n + 1) / 2)

/-- The theorem stating that for 200 numbers, the last remaining is 128 -/
theorem last_remaining_200 :
  lastRemainingNumber 200 = 128 := by
  sorry

end last_remaining_200_l368_36884


namespace special_polygon_properties_l368_36892

/-- A polygon where the sum of interior angles is 1/4 more than the sum of exterior angles -/
structure SpecialPolygon where
  n : ℕ  -- number of sides
  h : (n - 2) * 180 = 360 + (1/4) * 360

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem special_polygon_properties (p : SpecialPolygon) :
  p.n = 12 ∧ num_diagonals p.n = 54 := by
  sorry

#check special_polygon_properties

end special_polygon_properties_l368_36892


namespace total_egg_weight_in_pounds_l368_36843

-- Define the weight of a single egg in pounds
def egg_weight : ℚ := 1 / 16

-- Define the number of dozens of eggs needed
def dozens_needed : ℕ := 8

-- Define the number of eggs in a dozen
def eggs_per_dozen : ℕ := 12

-- Theorem to prove
theorem total_egg_weight_in_pounds : 
  (dozens_needed * eggs_per_dozen : ℚ) * egg_weight = 6 := by
  sorry

end total_egg_weight_in_pounds_l368_36843


namespace fraction_chain_l368_36812

theorem fraction_chain (a b c d : ℝ) 
  (h1 : a / b = 5)
  (h2 : b / c = 1 / 4)
  (h3 : c / d = 7) :
  d / a = 4 / 35 := by sorry

end fraction_chain_l368_36812


namespace count_even_numbers_between_300_and_600_l368_36874

theorem count_even_numbers_between_300_and_600 :
  (Finset.filter (fun n => n % 2 = 0 ∧ 300 < n ∧ n < 600) (Finset.range 600)).card = 149 := by
  sorry

end count_even_numbers_between_300_and_600_l368_36874


namespace min_value_theorem_l368_36898

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1 / (x + 3) + 1 / (y + 3) = 1 / 4) :
  x + 3 * y ≥ 18 + 21 * Real.sqrt 3 ∧ 
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 
    1 / (x₀ + 3) + 1 / (y₀ + 3) = 1 / 4 ∧
    x₀ + 3 * y₀ = 18 + 21 * Real.sqrt 3 :=
by sorry

end min_value_theorem_l368_36898


namespace park_area_ratio_l368_36830

theorem park_area_ratio (s : ℝ) (h : s > 0) : 
  (s^2) / ((3*s)^2) = 1/9 := by
  sorry

end park_area_ratio_l368_36830


namespace candy_bar_cost_l368_36869

theorem candy_bar_cost (selling_price : ℕ) (bought : ℕ) (sold : ℕ) (profit : ℕ) : 
  selling_price = 100 ∧ bought = 50 ∧ sold = 48 ∧ profit = 800 →
  (selling_price * sold - profit) / bought = 80 := by
sorry

end candy_bar_cost_l368_36869


namespace parking_solution_is_correct_l368_36836

/-- Represents the parking lot problem. -/
structure ParkingLot where
  total_cars : ℕ
  total_fee : ℕ
  medium_fee : ℕ
  small_fee : ℕ

/-- Represents the solution to the parking lot problem. -/
structure ParkingSolution where
  medium_cars : ℕ
  small_cars : ℕ

/-- Checks if a given solution satisfies the parking lot conditions. -/
def is_valid_solution (p : ParkingLot) (s : ParkingSolution) : Prop :=
  s.medium_cars + s.small_cars = p.total_cars ∧
  s.medium_cars * p.medium_fee + s.small_cars * p.small_fee = p.total_fee

/-- The parking lot problem instance. -/
def parking_problem : ParkingLot :=
  { total_cars := 30
  , total_fee := 324
  , medium_fee := 15
  , small_fee := 8 }

/-- The proposed solution to the parking lot problem. -/
def parking_solution : ParkingSolution :=
  { medium_cars := 12
  , small_cars := 18 }

/-- Theorem stating that the proposed solution is correct for the given problem. -/
theorem parking_solution_is_correct :
  is_valid_solution parking_problem parking_solution := by
  sorry

end parking_solution_is_correct_l368_36836


namespace coin_identification_l368_36865

/-- Represents the type of a coin -/
inductive CoinType
| Genuine
| Counterfeit

/-- Represents the result of weighing two groups of coins -/
inductive WeighResult
| Even
| Odd

/-- Function to determine the coin type based on the weighing result -/
def determineCoinType (result : WeighResult) : CoinType :=
  match result with
  | WeighResult.Even => CoinType.Genuine
  | WeighResult.Odd => CoinType.Counterfeit

theorem coin_identification
  (total_coins : Nat)
  (counterfeit_coins : Nat)
  (weight_difference : Nat)
  (h1 : total_coins = 101)
  (h2 : counterfeit_coins = 50)
  (h3 : weight_difference = 1)
  : ∀ (specified_coin : CoinType) (weigh_result : WeighResult),
    determineCoinType weigh_result = specified_coin :=
  sorry

end coin_identification_l368_36865


namespace scientific_notation_34_million_l368_36809

theorem scientific_notation_34_million :
  (34 : ℝ) * 1000000 = 3.4 * (10 : ℝ) ^ 7 := by
  sorry

end scientific_notation_34_million_l368_36809


namespace simplify_expressions_l368_36896

theorem simplify_expressions 
  (x y a b : ℝ) : 
  (-3*x + 2*y - 5*x - 7*y = -8*x - 5*y) ∧ 
  (5*(3*a^2*b - a*b^2) - 4*(-a*b^2 + 3*a^2*b) = 3*a^2*b - a*b^2) := by
  sorry

end simplify_expressions_l368_36896


namespace small_apple_cost_is_correct_l368_36842

/-- The cost of a small apple -/
def small_apple_cost : ℚ := 1.5

/-- The cost of a medium apple -/
def medium_apple_cost : ℚ := 2

/-- The cost of a big apple -/
def big_apple_cost : ℚ := 3

/-- The number of small and medium apples bought -/
def small_medium_apples : ℕ := 6

/-- The number of big apples bought -/
def big_apples : ℕ := 8

/-- The total cost of all apples bought -/
def total_cost : ℚ := 45

/-- Theorem stating that the cost of each small apple is $1.50 -/
theorem small_apple_cost_is_correct : 
  small_apple_cost * small_medium_apples + 
  medium_apple_cost * small_medium_apples + 
  big_apple_cost * big_apples = total_cost := by
sorry

end small_apple_cost_is_correct_l368_36842


namespace sea_glass_collection_l368_36899

theorem sea_glass_collection (blanche_green : ℕ) (rose_red rose_blue : ℕ) (dorothy_total : ℕ)
  (h1 : blanche_green = 12)
  (h2 : rose_red = 9)
  (h3 : rose_blue = 11)
  (h4 : dorothy_total = 57) :
  ∃ (blanche_red : ℕ),
    dorothy_total = 2 * (blanche_red + rose_red) + 3 * rose_blue ∧
    blanche_red = 3 :=
by sorry

end sea_glass_collection_l368_36899


namespace rectangle_ratio_is_two_l368_36837

-- Define the side length of the inner square
def inner_square_side : ℝ := 1

-- Define the shorter side of the rectangle
def rectangle_short_side : ℝ := inner_square_side

-- Define the longer side of the rectangle
def rectangle_long_side : ℝ := 2 * inner_square_side

-- Define the side length of the outer square
def outer_square_side : ℝ := inner_square_side + 2 * rectangle_short_side

-- State the theorem
theorem rectangle_ratio_is_two :
  (outer_square_side ^ 2 = 9 * inner_square_side ^ 2) →
  (rectangle_long_side / rectangle_short_side = 2) :=
by sorry

end rectangle_ratio_is_two_l368_36837


namespace sqrt_of_four_l368_36800

theorem sqrt_of_four : Real.sqrt 4 = 2 := by
  sorry

end sqrt_of_four_l368_36800


namespace power_of_two_starts_with_1968_l368_36804

-- Define the conditions
def m : ℕ := 3^2
def k : ℕ := 2^3

-- Define a function to check if a number starts with 1968
def starts_with_1968 (x : ℕ) : Prop :=
  ∃ y : ℕ, 1968 * 10^y ≤ x ∧ x < 1969 * 10^y

-- State the theorem
theorem power_of_two_starts_with_1968 :
  ∃ n : ℕ, n > 2^k ∧ starts_with_1968 (2^n) ∧
  ∀ m : ℕ, m < n → ¬starts_with_1968 (2^m) :=
sorry

end power_of_two_starts_with_1968_l368_36804


namespace horner_method_example_l368_36815

def f (x : ℝ) : ℝ := 6*x^5 + 5*x^4 - 4*x^3 + 3*x^2 - 2*x + 1

theorem horner_method_example : f 2 = 249 := by
  sorry

end horner_method_example_l368_36815


namespace max_homework_time_l368_36814

/-- Represents the time spent on each subject in minutes -/
structure HomeworkTime where
  biology : ℕ
  history : ℕ
  geography : ℕ

/-- Calculates the total time spent on homework given the conditions -/
def total_homework_time (t : HomeworkTime) : ℕ :=
  t.biology + t.history + t.geography

/-- Theorem stating that Max's total homework time is 180 minutes -/
theorem max_homework_time :
  ∀ t : HomeworkTime,
  t.biology = 20 ∧
  t.history = 2 * t.biology ∧
  t.geography = 3 * t.history →
  total_homework_time t = 180 :=
by
  sorry

#check max_homework_time

end max_homework_time_l368_36814


namespace right_triangle_integer_sides_l368_36848

theorem right_triangle_integer_sides (a b c : ℕ) : 
  a^2 + b^2 = c^2 →  -- Pythagorean theorem (right triangle condition)
  Nat.gcd a (Nat.gcd b c) = 1 →  -- GCD of sides is 1
  ∃ m n : ℕ, 
    (a = 2*m*n ∧ b = m^2 - n^2 ∧ c = m^2 + n^2) ∨ 
    (b = 2*m*n ∧ a = m^2 - n^2 ∧ c = m^2 + n^2) :=
by sorry

end right_triangle_integer_sides_l368_36848


namespace december_sales_fraction_l368_36862

theorem december_sales_fraction (average_sales : ℝ) (h : average_sales > 0) :
  let january_to_november_sales := 11 * average_sales
  let december_sales := 5 * average_sales
  let total_annual_sales := january_to_november_sales + december_sales
  december_sales / total_annual_sales = 5 / 16 := by
sorry

end december_sales_fraction_l368_36862


namespace circles_intersection_range_l368_36845

-- Define the circles
def C₁ (t x y : ℝ) : Prop := x^2 + y^2 - 2*t*x + t^2 - 4 = 0
def C₂ (t x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*t*y + 4*t^2 - 8 = 0

-- Define the intersection condition
def intersect (t : ℝ) : Prop := ∃ x y : ℝ, C₁ t x y ∧ C₂ t x y

-- State the theorem
theorem circles_intersection_range :
  ∀ t : ℝ, intersect t ↔ ((-12/5 < t ∧ t < -2/5) ∨ (0 < t ∧ t < 2)) :=
by sorry

end circles_intersection_range_l368_36845


namespace range_of_c_l368_36826

theorem range_of_c (a b c : ℝ) 
  (ha : 6 < a ∧ a < 10) 
  (hb : a / 2 ≤ b ∧ b ≤ 2 * a) 
  (hc : c = a + b) : 
  9 < c ∧ c < 30 := by
  sorry

end range_of_c_l368_36826


namespace log_equation_implies_ratio_l368_36840

theorem log_equation_implies_ratio (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : Real.log (x - y) + Real.log (x + 2*y) = Real.log 2 + Real.log x + Real.log y) : 
  x / y = 2 := by
  sorry

end log_equation_implies_ratio_l368_36840


namespace parallel_condition_l368_36850

/-- Two lines are parallel if and only if their slopes are equal and they have different y-intercepts -/
def are_parallel (m : ℝ) : Prop :=
  ((-1 : ℝ) / (1 + m) = -m / 2) ∧ 
  ((2 - m) / (1 + m) ≠ -4)

/-- Line l₁: x + (1+m)y = 2-m -/
def line_l1 (m : ℝ) (x y : ℝ) : Prop :=
  x + (1 + m) * y = 2 - m

/-- Line l₂: 2mx + 4y = -16 -/
def line_l2 (m : ℝ) (x y : ℝ) : Prop :=
  2 * m * x + 4 * y = -16

theorem parallel_condition :
  ∀ m : ℝ, (m = 1) ↔ are_parallel m :=
sorry

end parallel_condition_l368_36850


namespace problem_statement_l368_36886

theorem problem_statement (x y z : ℝ) 
  (h1 : x ≠ y)
  (h2 : x^2 * (y + z) = 2019)
  (h3 : y^2 * (z + x) = 2019) :
  z^2 * (x + y) - x * y * z = 4038 := by
sorry

end problem_statement_l368_36886


namespace geometric_series_sum_l368_36813

theorem geometric_series_sum : 
  let s := ∑' k, (3^k : ℝ) / (9^k - 1)
  s = 1/2 := by
sorry

end geometric_series_sum_l368_36813


namespace sin_alpha_fourth_quadrant_l368_36803

theorem sin_alpha_fourth_quadrant (α : Real) : 
  (π/2 < α ∧ α < 2*π) →  -- α is in the fourth quadrant
  (Real.tan (π - α) = 5/12) → 
  (Real.sin α = -5/13) := by
sorry

end sin_alpha_fourth_quadrant_l368_36803


namespace angle_c_measure_l368_36828

theorem angle_c_measure (A B C : ℝ) : 
  A = 86 →
  B = 3 * C + 22 →
  A + B + C = 180 →
  C = 18 := by
sorry

end angle_c_measure_l368_36828


namespace rectangle_point_s_l368_36810

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Definition of a rectangle formed by four points -/
def isRectangle (p q r s : Point2D) : Prop :=
  (p.x = q.x ∧ r.x = s.x ∧ p.y = s.y ∧ q.y = r.y) ∨
  (p.x = s.x ∧ q.x = r.x ∧ p.y = q.y ∧ r.y = s.y)

/-- The theorem stating that given P, Q, and R, the point S forms a rectangle -/
theorem rectangle_point_s (p q r : Point2D)
  (h_p : p = ⟨3, -2⟩)
  (h_q : q = ⟨3, 1⟩)
  (h_r : r = ⟨7, 1⟩) :
  ∃ s : Point2D, s = ⟨7, -2⟩ ∧ isRectangle p q r s :=
sorry

end rectangle_point_s_l368_36810


namespace odd_square_plus_even_product_is_odd_l368_36864

theorem odd_square_plus_even_product_is_odd (k m : ℤ) : 
  let o : ℤ := 2 * k + 3
  let n : ℤ := 2 * m
  Odd (o^2 + n * o) := by
sorry

end odd_square_plus_even_product_is_odd_l368_36864


namespace uniform_random_transform_l368_36897

/-- A uniform random number on an interval -/
structure UniformRandom (a b : ℝ) where
  value : ℝ
  in_range : a ≤ value ∧ value ≤ b

/-- Theorem: If b₁ is a uniform random number on [0,1] and b = (b₁ - 0.5) * 6,
    then b is a uniform random number on [-3,3] -/
theorem uniform_random_transform (b₁ : UniformRandom 0 1) :
  let b := (b₁.value - 0.5) * 6
  ∃ (b' : UniformRandom (-3) 3), b'.value = b := by
  sorry

end uniform_random_transform_l368_36897


namespace probability_club_then_heart_l368_36876

-- Define the total number of cards in a standard deck
def total_cards : ℕ := 52

-- Define the number of clubs in a standard deck
def num_clubs : ℕ := 13

-- Define the number of hearts in a standard deck
def num_hearts : ℕ := 13

-- Theorem statement
theorem probability_club_then_heart :
  (num_clubs : ℚ) / total_cards * num_hearts / (total_cards - 1) = 13 / 204 := by
  sorry

end probability_club_then_heart_l368_36876


namespace polynomial_roots_and_constant_term_l368_36807

def polynomial (a b c d : ℤ) (x : ℝ) : ℝ := x^4 + a*x^3 + b*x^2 + c*x + d

theorem polynomial_roots_and_constant_term 
  (a b c d : ℤ) 
  (h1 : ∀ x : ℝ, polynomial a b c d x = 0 → (∃ n : ℕ, x = -↑n))
  (h2 : a + b + c + d = 2009) :
  d = 528 := by
  sorry

end polynomial_roots_and_constant_term_l368_36807


namespace condition_relationship_l368_36878

theorem condition_relationship :
  (∀ x : ℝ, (0 < x ∧ x < 5) → (|x - 2| < 3)) ∧
  (∃ x : ℝ, (|x - 2| < 3) ∧ ¬(0 < x ∧ x < 5)) :=
by sorry

end condition_relationship_l368_36878


namespace other_diagonal_length_l368_36822

/-- A rhombus is a quadrilateral with four equal sides -/
structure Rhombus where
  diag1 : ℝ
  diag2 : ℝ
  area : ℝ

/-- The area of a rhombus is half the product of its diagonals -/
axiom rhombus_area (r : Rhombus) : r.area = (r.diag1 * r.diag2) / 2

theorem other_diagonal_length (r : Rhombus) 
  (h1 : r.diag1 = 14)
  (h2 : r.area = 140) : 
  r.diag2 = 20 := by
sorry

end other_diagonal_length_l368_36822


namespace fgh_supermarkets_count_l368_36855

/-- The number of FGH supermarkets in the US -/
def us_supermarkets : ℕ := 42

/-- The number of FGH supermarkets in Canada -/
def canada_supermarkets : ℕ := us_supermarkets - 14

/-- The total number of FGH supermarkets -/
def total_supermarkets : ℕ := 70

theorem fgh_supermarkets_count :
  (us_supermarkets + canada_supermarkets = total_supermarkets) ∧
  (us_supermarkets = canada_supermarkets + 14) :=
by sorry

end fgh_supermarkets_count_l368_36855


namespace b_inequalities_l368_36859

theorem b_inequalities (a : ℝ) (h : a ∈ Set.Icc 0 1) :
  let b := a^3 + 1 / (1 + a)
  (b ≥ 1 - a + a^2) ∧ (3/4 < b ∧ b ≤ 3/2) := by
  sorry

end b_inequalities_l368_36859


namespace output_for_15_l368_36824

def function_machine (input : ℤ) : ℤ :=
  let step1 := input * 3
  if step1 > 25 then step1 - 7 else step1 + 10

theorem output_for_15 : function_machine 15 = 38 := by
  sorry

end output_for_15_l368_36824


namespace smallest_positive_integer_with_remainders_l368_36888

theorem smallest_positive_integer_with_remainders : ∃ x : ℕ, 
  (x > 0) ∧ 
  (x % 3 = 2) ∧ 
  (x % 4 = 3) ∧ 
  (x % 5 = 4) ∧ 
  (∀ y : ℕ, y > 0 → (y % 3 = 2) → (y % 4 = 3) → (y % 5 = 4) → y ≥ x) ∧
  x = 59 := by
sorry

end smallest_positive_integer_with_remainders_l368_36888


namespace product_equals_three_l368_36858

/-- The repeating decimal 0.333... --/
def repeating_third : ℚ := 1 / 3

/-- The product of 0.333... and 9 --/
def product : ℚ := repeating_third * 9

theorem product_equals_three : product = 3 := by sorry

end product_equals_three_l368_36858


namespace zoo_count_l368_36853

theorem zoo_count (zebras camels monkeys giraffes : ℕ) : 
  camels = zebras / 2 →
  monkeys = 4 * camels →
  giraffes = 2 →
  monkeys = giraffes + 22 →
  zebras = 12 :=
by
  sorry

end zoo_count_l368_36853


namespace quadratic_real_roots_condition_l368_36891

/-- 
Given a quadratic equation (m-1)x^2 + 3x - 1 = 0,
prove that for the equation to have real roots,
m must satisfy: m ≥ -5/4 and m ≠ 1
-/
theorem quadratic_real_roots_condition (m : ℝ) : 
  (∃ x : ℝ, (m - 1) * x^2 + 3 * x - 1 = 0) ↔ 
  (m ≥ -5/4 ∧ m ≠ 1) :=
sorry

end quadratic_real_roots_condition_l368_36891


namespace fish_corn_equivalence_l368_36839

theorem fish_corn_equivalence :
  ∀ (fish honey corn : ℚ),
  (5 * fish = 3 * honey) →
  (honey = 6 * corn) →
  (fish = 3.6 * corn) :=
by sorry

end fish_corn_equivalence_l368_36839


namespace composition_inverse_implies_value_l368_36849

-- Define the functions
def f (a b x : ℝ) : ℝ := a * x + b
def g (x : ℝ) : ℝ := -4 * x + 6
def h (a b x : ℝ) : ℝ := f a b (g x)

-- State the theorem
theorem composition_inverse_implies_value (a b : ℝ) :
  (∀ x, h a b x = x - 9) →
  2 * a - 3 * b = 22 := by
  sorry

end composition_inverse_implies_value_l368_36849


namespace circle_intersection_theorem_l368_36882

-- Define the circle equation
def circle_eq (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y + m = 0

-- Define the line equation
def line_eq (x y : ℝ) : Prop :=
  x + 2*y - 4 = 0

-- Define perpendicularity of two points from origin
def perp_from_origin (x1 y1 x2 y2 : ℝ) : Prop :=
  x1 * x2 + y1 * y2 = 0

theorem circle_intersection_theorem :
  ∀ m : ℝ,
  (∃ x y : ℝ, circle_eq x y m) ↔ m < 5 ∧
  (∃ x1 y1 x2 y2 : ℝ,
    circle_eq x1 y1 m ∧ circle_eq x2 y2 m ∧
    line_eq x1 y1 ∧ line_eq x2 y2 ∧
    perp_from_origin x1 y1 x2 y2 → m = 8/5) ∧
  (∃ x1 y1 x2 y2 : ℝ,
    circle_eq x1 y1 (8/5) ∧ circle_eq x2 y2 (8/5) ∧
    line_eq x1 y1 ∧ line_eq x2 y2 ∧
    perp_from_origin x1 y1 x2 y2 →
    ∀ x y : ℝ, x^2 + y^2 - (8/5)*x - (16/5)*y = 0 ↔
    ∃ t : ℝ, x = x1 + t*(x2 - x1) ∧ y = y1 + t*(y2 - y1) ∧ 0 ≤ t ∧ t ≤ 1) :=
by sorry

end circle_intersection_theorem_l368_36882


namespace length_of_CE_l368_36890

/-- Given a plot ABCD with specific measurements, prove the length of CE -/
theorem length_of_CE (AF ED AE : ℝ) (area_ABCD : ℝ) :
  AF = 30 ∧ ED = 50 ∧ AE = 120 ∧ area_ABCD = 7200 →
  ∃ CE : ℝ, CE = 138 ∧
    area_ABCD = (1/2 * AE * ED) + (1/2 * (AF + CE) * ED) := by
  sorry

end length_of_CE_l368_36890


namespace max_value_of_function_l368_36854

theorem max_value_of_function (a : ℝ) (ha : 0 < a ∧ a < 1) :
  ∀ x y : ℝ, |x| + |y| ≤ 1 → (∀ x' y' : ℝ, |x'| + |y'| ≤ 1 → a * x + y ≤ a * x' + y') →
  a * x + y = 1 :=
by sorry

end max_value_of_function_l368_36854


namespace gear_teeth_problem_l368_36893

theorem gear_teeth_problem (x y z : ℕ) (h1 : x > y) (h2 : y > z) (h3 : x + y + z = 60) (h4 : 4 * x - 20 = 5 * y) (h5 : 5 * y = 10 * z) : x = 30 ∧ y = 20 ∧ z = 10 := by
  sorry

end gear_teeth_problem_l368_36893


namespace line_segment_translation_l368_36872

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a translation vector -/
structure Translation where
  dx : ℝ
  dy : ℝ

/-- Applies a translation to a point -/
def translatePoint (p : Point) (t : Translation) : Point :=
  { x := p.x + t.dx, y := p.y + t.dy }

theorem line_segment_translation (A B : Point) (A_new : Point) :
  A = { x := 1, y := 2 } →
  B = { x := 7, y := 5 } →
  A_new = { x := -6, y := -3 } →
  let t : Translation := { dx := A_new.x - A.x, dy := A_new.y - A.y }
  translatePoint B t = { x := 0, y := 0 } := by sorry

end line_segment_translation_l368_36872


namespace airport_exchange_calculation_l368_36857

/-- Calculates the amount of dollars received when exchanging euros at an airport with a reduced exchange rate. -/
theorem airport_exchange_calculation (euros : ℝ) (normal_rate : ℝ) (airport_rate_fraction : ℝ) : 
  euros / normal_rate * airport_rate_fraction = 10 :=
by
  -- Assuming euros = 70, normal_rate = 5, and airport_rate_fraction = 5/7
  sorry

#check airport_exchange_calculation

end airport_exchange_calculation_l368_36857


namespace percent_of_whole_six_point_two_percent_of_thousand_l368_36883

theorem percent_of_whole (part whole : ℝ) (h : whole ≠ 0) :
  (part / whole) * 100 = part * 100 / whole := by sorry

theorem six_point_two_percent_of_thousand :
  (6.2 / 1000) * 100 = 0.62 := by sorry

end percent_of_whole_six_point_two_percent_of_thousand_l368_36883


namespace closest_perfect_square_to_314_l368_36835

/-- The perfect-square integer closest to 314 is 324. -/
theorem closest_perfect_square_to_314 : 
  ∀ n : ℕ, n ≠ 324 → n * n ≠ 0 → |314 - (324 : ℤ)| ≤ |314 - (n * n : ℤ)| := by
  sorry

end closest_perfect_square_to_314_l368_36835


namespace insect_count_l368_36887

/-- Given a number of leaves, ladybugs per leaf, and ants per leaf, 
    calculate the total number of ladybugs and ants combined. -/
def total_insects (leaves : ℕ) (ladybugs_per_leaf : ℕ) (ants_per_leaf : ℕ) : ℕ :=
  leaves * ladybugs_per_leaf + leaves * ants_per_leaf

/-- Theorem stating that given 84 leaves, 139 ladybugs per leaf, and 97 ants per leaf,
    the total number of ladybugs and ants combined is 19,824. -/
theorem insect_count : total_insects 84 139 97 = 19824 := by
  sorry

end insect_count_l368_36887


namespace quadratic_coefficients_l368_36821

def f (x : ℝ) : ℝ := x^2 - 4*x + 5

theorem quadratic_coefficients :
  ∃ (a b c : ℝ), (∀ x, f x = a*x^2 + b*x + c) ∧ a = 1 ∧ b = -4 ∧ c = 5 := by
  sorry

end quadratic_coefficients_l368_36821


namespace grazing_area_expansion_l368_36820

/-- Given a circular grazing area with an initial radius of 9 meters,
    if the area is increased by 1408 square meters,
    the new radius will be 23 meters. -/
theorem grazing_area_expansion (π : ℝ) (h : π > 0) :
  let r₁ : ℝ := 9
  let additional_area : ℝ := 1408
  let r₂ : ℝ := Real.sqrt (r₁^2 + additional_area / π)
  r₂ = 23 := by sorry

end grazing_area_expansion_l368_36820


namespace total_earnings_proof_l368_36847

def lauryn_earnings : ℝ := 2000
def aurelia_percentage : ℝ := 0.7

theorem total_earnings_proof :
  let aurelia_earnings := lauryn_earnings * aurelia_percentage
  lauryn_earnings + aurelia_earnings = 3400 :=
by sorry

end total_earnings_proof_l368_36847


namespace decimal_6_to_binary_l368_36823

def decimal_to_binary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 2) ((m % 2) :: acc)
    aux n []

theorem decimal_6_to_binary :
  decimal_to_binary 6 = [1, 1, 0] := by
  sorry

end decimal_6_to_binary_l368_36823


namespace equidistant_point_distance_l368_36852

-- Define the equilateral triangle DEF
def triangle_DEF : Set (ℝ × ℝ × ℝ) := sorry

-- Define the side length of triangle DEF
def side_length : ℝ := 300

-- Define points X and Y
def X : ℝ × ℝ × ℝ := sorry
def Y : ℝ × ℝ × ℝ := sorry

-- Define the property that X and Y are equidistant from vertices of DEF
def equidistant_X (X : ℝ × ℝ × ℝ) : Prop := sorry
def equidistant_Y (Y : ℝ × ℝ × ℝ) : Prop := sorry

-- Define the 90° dihedral angle between planes XDE and YDE
def dihedral_angle_90 (X Y : ℝ × ℝ × ℝ) : Prop := sorry

-- Define point R
def R : ℝ × ℝ × ℝ := sorry

-- Define the distance r
def r : ℝ := sorry

-- Define the property that R is equidistant from D, E, F, X, and Y
def equidistant_R (R : ℝ × ℝ × ℝ) : Prop := sorry

theorem equidistant_point_distance :
  equidistant_X X →
  equidistant_Y Y →
  dihedral_angle_90 X Y →
  equidistant_R R →
  r = 50 * Real.sqrt 6 := by sorry

end equidistant_point_distance_l368_36852


namespace siblings_weekly_water_consumption_l368_36846

/-- The amount of water consumed by three siblings in a week -/
def water_consumption (theo_daily : ℕ) (mason_daily : ℕ) (roxy_daily : ℕ) (days_in_week : ℕ) : ℕ :=
  (theo_daily + mason_daily + roxy_daily) * days_in_week

/-- Theorem stating that the siblings drink 168 cups of water in a week -/
theorem siblings_weekly_water_consumption :
  water_consumption 8 7 9 7 = 168 := by
  sorry

end siblings_weekly_water_consumption_l368_36846


namespace ball_returns_after_12_throws_l368_36841

/-- Represents the number of girls in the circle -/
def n : ℕ := 15

/-- Represents the number of girls skipped in each throw -/
def skip : ℕ := 4

/-- The function that determines the next girl to receive the ball -/
def next (x : ℕ) : ℕ := (x + skip + 1) % n + 1

/-- Represents the sequence of girls receiving the ball -/
def ball_sequence (k : ℕ) : ℕ := 
  Nat.iterate next 1 k

theorem ball_returns_after_12_throws : 
  ball_sequence 12 = 1 := by sorry

end ball_returns_after_12_throws_l368_36841


namespace soft_drink_storage_l368_36816

theorem soft_drink_storage (small_initial big_initial : ℕ) 
  (big_sold_percent : ℚ) (total_remaining : ℕ) :
  small_initial = 6000 →
  big_initial = 14000 →
  big_sold_percent = 23 / 100 →
  total_remaining = 15580 →
  ∃ (small_sold_percent : ℚ),
    small_sold_percent = 37 / 100 ∧
    (small_initial : ℚ) * (1 - small_sold_percent) + 
    (big_initial : ℚ) * (1 - big_sold_percent) = total_remaining := by
  sorry

end soft_drink_storage_l368_36816


namespace intersection_point_l368_36833

/-- The first curve equation -/
def curve1 (x : ℝ) : ℝ := x^3 + 3*x^2 + 4*x - 5

/-- The second curve equation -/
def curve2 (x : ℝ) : ℝ := 2*x^2 + 11

/-- Theorem stating that (2, 19) is the only intersection point of the two curves -/
theorem intersection_point : 
  (∃! p : ℝ × ℝ, curve1 p.1 = curve2 p.1 ∧ p.2 = curve1 p.1) ∧ 
  (∀ p : ℝ × ℝ, curve1 p.1 = curve2 p.1 → p = (2, 19)) := by
  sorry

end intersection_point_l368_36833


namespace smallest_non_triangle_forming_subtraction_l368_36829

theorem smallest_non_triangle_forming_subtraction : ∃ (x : ℕ), 
  (x > 0) ∧ 
  (∀ (y : ℕ), y < x → (7 - y) + (24 - y) > (26 - y)) ∧
  ((7 - x) + (24 - x) ≤ (26 - x)) ∧
  x = 5 := by
sorry

end smallest_non_triangle_forming_subtraction_l368_36829


namespace max_soap_boxes_in_carton_l368_36871

/-- Represents the dimensions of a box in inches -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℕ :=
  d.length * d.width * d.height

/-- The dimensions of the carton -/
def cartonDimensions : BoxDimensions :=
  { length := 25, width := 48, height := 60 }

/-- The dimensions of a soap box -/
def soapBoxDimensions : BoxDimensions :=
  { length := 8, width := 6, height := 5 }

/-- Theorem stating the maximum number of soap boxes that can fit in the carton -/
theorem max_soap_boxes_in_carton :
  (boxVolume cartonDimensions) / (boxVolume soapBoxDimensions) = 300 := by
  sorry

end max_soap_boxes_in_carton_l368_36871


namespace sin_2alpha_value_l368_36844

theorem sin_2alpha_value (α : Real) (h : Real.sin α - Real.cos α = 4/3) : 
  Real.sin (2 * α) = -7/9 := by
  sorry

end sin_2alpha_value_l368_36844


namespace train_circuit_time_l368_36870

/-- Represents the time in seconds -/
def seconds_per_circuit : ℕ := 71

/-- Represents the number of circuits -/
def num_circuits : ℕ := 6

/-- Converts seconds to minutes and remaining seconds -/
def seconds_to_minutes_and_seconds (total_seconds : ℕ) : ℕ × ℕ :=
  (total_seconds / 60, total_seconds % 60)

theorem train_circuit_time : 
  seconds_to_minutes_and_seconds (num_circuits * seconds_per_circuit) = (7, 6) := by
  sorry

end train_circuit_time_l368_36870


namespace inequality_relation_l368_36879

theorem inequality_relation (a b : ℝ) : 
  ¬(∀ a b : ℝ, a > b → 1/a < 1/b) ∧ ¬(∀ a b : ℝ, 1/a < 1/b → a > b) := by
  sorry

end inequality_relation_l368_36879


namespace maggies_work_hours_l368_36851

/-- Maggie's work hours problem -/
theorem maggies_work_hours 
  (office_rate : ℝ) 
  (tractor_rate : ℝ) 
  (total_income : ℝ) 
  (h1 : office_rate = 10)
  (h2 : tractor_rate = 12)
  (h3 : total_income = 416) : 
  ∃ (tractor_hours : ℝ),
    tractor_hours = 13 ∧ 
    office_rate * (2 * tractor_hours) + tractor_rate * tractor_hours = total_income :=
by sorry

end maggies_work_hours_l368_36851


namespace min_droppers_required_l368_36819

theorem min_droppers_required (container_volume : ℕ) (dropper_volume : ℕ) : container_volume = 265 → dropper_volume = 19 → (14 : ℕ) = (container_volume + dropper_volume - 1) / dropper_volume := by
  sorry

end min_droppers_required_l368_36819


namespace multiple_with_binary_digits_l368_36868

theorem multiple_with_binary_digits (n : ℤ) : ∃ k : ℤ,
  (∃ m : ℤ, k = n * m) ∧ 
  (∃ d : ℕ, d ≤ n ∧ k < 10^d) ∧
  (∀ i : ℕ, i < n → (k / 10^i) % 10 = 0 ∨ (k / 10^i) % 10 = 1) :=
sorry

end multiple_with_binary_digits_l368_36868


namespace smallest_n_divisible_by_57_l368_36805

theorem smallest_n_divisible_by_57 :
  ∃ (n : ℕ), n > 0 ∧ 57 ∣ (7^n + 2*n) ∧ ∀ (m : ℕ), m > 0 ∧ 57 ∣ (7^m + 2*m) → n ≤ m :=
by
  use 25
  sorry

end smallest_n_divisible_by_57_l368_36805


namespace decimal_arithmetic_l368_36806

theorem decimal_arithmetic : 
  (∃ x : ℝ, x = 3.92 + 0.4 ∧ x = 3.96) ∧
  (∃ y : ℝ, y = 4.93 - 1.5 ∧ y = 3.43) := by
  sorry

end decimal_arithmetic_l368_36806


namespace rearranged_number_bounds_l368_36802

/-- Given a natural number B, returns the number A obtained by moving the last digit of B to the first position --/
def rearrange_digits (B : ℕ) : ℕ :=
  let b := B % 10
  10^8 * b + (B - b) / 10

/-- Checks if two natural numbers are coprime --/
def are_coprime (a b : ℕ) : Prop :=
  Nat.gcd a b = 1

/-- Theorem stating the largest and smallest possible values of A given the conditions on B --/
theorem rearranged_number_bounds :
  ∀ B : ℕ,
  B > 222222222 →
  are_coprime B 18 →
  ∃ A : ℕ,
  A = rearrange_digits B ∧
  A ≤ 999999998 ∧
  A ≥ 122222224 ∧
  (∀ A' : ℕ, A' = rearrange_digits B → A' ≤ 999999998 ∧ A' ≥ 122222224) :=
sorry

end rearranged_number_bounds_l368_36802


namespace geometric_sequence_11th_term_l368_36801

/-- Represents a geometric sequence -/
structure GeometricSequence where
  -- The sequence function
  a : ℕ → ℝ
  -- The common ratio
  r : ℝ
  -- The geometric sequence property
  geom_prop : ∀ n : ℕ, a (n + 1) = a n * r

/-- Theorem: In a geometric sequence where the 5th term is -2 and the 8th term is -54, the 11th term is -1458 -/
theorem geometric_sequence_11th_term
  (seq : GeometricSequence)
  (h5 : seq.a 5 = -2)
  (h8 : seq.a 8 = -54) :
  seq.a 11 = -1458 := by
  sorry

end geometric_sequence_11th_term_l368_36801


namespace line_parabola_circle_intersection_l368_36860

/-- A line intersecting a parabola and a circle with specific conditions -/
theorem line_parabola_circle_intersection
  (k m : ℝ)
  (l : Set (ℝ × ℝ))
  (A B C D : ℝ × ℝ)
  (h_line : l = {(x, y) | y = k * x + m})
  (h_parabola : A ∈ l ∧ B ∈ l ∧ A.1^2 = 2 * A.2 ∧ B.1^2 = 2 * B.2)
  (h_midpoint : (A.1 + B.1) / 2 = 1)
  (h_circle : C ∈ l ∧ D ∈ l ∧ C.1^2 + C.2^2 = 12 ∧ D.1^2 + D.2^2 = 12)
  (h_equal_length : (A.1 - B.1)^2 + (A.2 - B.2)^2 = (C.1 - D.1)^2 + (C.2 - D.2)^2) :
  k = 1 ∧ m = 2 := by sorry

end line_parabola_circle_intersection_l368_36860


namespace defeat_points_zero_l368_36895

/-- Represents the point system and match results for a football competition. -/
structure FootballCompetition where
  victoryPoints : ℕ := 3
  drawPoints : ℕ := 1
  defeatPoints : ℕ
  totalMatches : ℕ := 20
  pointsAfter5Games : ℕ := 14
  minVictoriesRemaining : ℕ := 6
  finalPointTarget : ℕ := 40

/-- Theorem stating that the points for a defeat must be zero under the given conditions. -/
theorem defeat_points_zero (fc : FootballCompetition) : fc.defeatPoints = 0 := by
  sorry

#check defeat_points_zero

end defeat_points_zero_l368_36895


namespace smallest_multiple_of_9_and_6_l368_36831

def is_multiple (a b : ℕ) : Prop := ∃ k : ℕ, b = a * k

theorem smallest_multiple_of_9_and_6 : 
  (∀ n : ℕ, n > 0 ∧ is_multiple 9 n ∧ is_multiple 6 n → n ≥ 18) ∧
  (18 > 0 ∧ is_multiple 9 18 ∧ is_multiple 6 18) :=
sorry

end smallest_multiple_of_9_and_6_l368_36831


namespace vector_angle_theorem_l368_36818

-- Define a type for 3D vectors
def Vector3D := ℝ × ℝ × ℝ

-- Define a function to calculate the angle between two vectors
noncomputable def angle (v1 v2 : Vector3D) : ℝ := sorry

-- Define a predicate for non-zero vectors
def nonzero (v : Vector3D) : Prop := v ≠ (0, 0, 0)

theorem vector_angle_theorem (vectors : Fin 30 → Vector3D) 
  (h : ∀ i, nonzero (vectors i)) : 
  ∃ i j, i ≠ j ∧ angle (vectors i) (vectors j) < Real.pi / 4 := by sorry

end vector_angle_theorem_l368_36818


namespace square_greater_than_abs_l368_36856

theorem square_greater_than_abs (a b : ℝ) : a > |b| → a^2 > b^2 := by
  sorry

end square_greater_than_abs_l368_36856


namespace calculate_overall_profit_l368_36889

/-- Calculate the overall profit from selling two items with given purchase prices and profit/loss percentages -/
theorem calculate_overall_profit
  (grinder_price mobile_price : ℕ)
  (grinder_loss_percent mobile_profit_percent : ℚ)
  (h1 : grinder_price = 15000)
  (h2 : mobile_price = 10000)
  (h3 : grinder_loss_percent = 4 / 100)
  (h4 : mobile_profit_percent = 10 / 100)
  : ↑grinder_price * (1 - grinder_loss_percent) + 
    ↑mobile_price * (1 + mobile_profit_percent) - 
    ↑(grinder_price + mobile_price) = 400 := by
  sorry


end calculate_overall_profit_l368_36889


namespace peaches_before_picking_l368_36827

-- Define the variables
def peaches_picked : ℕ := 52
def total_peaches_now : ℕ := 86

-- Define the theorem
theorem peaches_before_picking (peaches_picked total_peaches_now : ℕ) :
  peaches_picked = 52 →
  total_peaches_now = 86 →
  total_peaches_now - peaches_picked = 34 := by
sorry

end peaches_before_picking_l368_36827


namespace f_value_at_2_l368_36825

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := x^5 + a*x^3 + b*x - 8

-- State the theorem
theorem f_value_at_2 (a b : ℝ) (h : f a b (-2) = 10) : f a b 2 = -26 := by
  sorry

end f_value_at_2_l368_36825


namespace girls_average_height_l368_36881

/-- Calculates the average height of female students in a class -/
def average_height_girls (total_students : ℕ) (boys : ℕ) (avg_height_all : ℚ) (avg_height_boys : ℚ) : ℚ :=
  let girls := total_students - boys
  let total_height := (total_students : ℚ) * avg_height_all
  let boys_height := (boys : ℚ) * avg_height_boys
  let girls_height := total_height - boys_height
  girls_height / (girls : ℚ)

/-- Theorem stating the average height of girls in the class -/
theorem girls_average_height :
  average_height_girls 30 18 140 144 = 134 := by
  sorry

end girls_average_height_l368_36881


namespace circle_area_above_line_l368_36863

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 - 8*x + y^2 - 16*y + 48 = 0

-- Define the line equation
def line_equation (y : ℝ) : Prop := y = 4

-- Theorem statement
theorem circle_area_above_line :
  ∃ (A : ℝ), 
    (∀ x y : ℝ, circle_equation x y → 
      (y > 4 → (x, y) ∈ {p : ℝ × ℝ | p.1^2 + p.2^2 ≤ A})) ∧
    A = 24 * Real.pi :=
sorry

end circle_area_above_line_l368_36863


namespace complex_fraction_simplification_l368_36880

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  ((4 - 6*i) / (4 + 6*i) + (4 + 6*i) / (4 - 6*i)) = (-10 : ℚ) / 13 := by
  sorry

end complex_fraction_simplification_l368_36880


namespace game_theorem_l368_36873

/-- Represents the game "What? Where? When?" with given conditions -/
structure Game where
  envelopes : ℕ := 13
  win_points : ℕ := 6
  win_prob : ℚ := 1/2

/-- Expected number of points for a single game -/
def expected_points (g : Game) : ℚ := sorry

/-- Expected number of points over 100 games -/
def expected_points_100 (g : Game) : ℚ := 100 * expected_points g

/-- Probability of an envelope being chosen in a game -/
def envelope_prob (g : Game) : ℚ := sorry

theorem game_theorem (g : Game) :
  expected_points_100 g = 465 ∧ envelope_prob g = 12/13 := by sorry

end game_theorem_l368_36873


namespace mn_length_is_8_l368_36875

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between two points on a line parallel to the x-axis -/
def distanceOnXParallelLine (p1 p2 : Point) : ℝ :=
  |p1.x - p2.x|

theorem mn_length_is_8 (x : ℝ) :
  let m : Point := ⟨x + 5, x - 4⟩
  let n : Point := ⟨-1, -2⟩
  m.y = n.y → distanceOnXParallelLine m n = 8 := by
  sorry

end mn_length_is_8_l368_36875


namespace consecutive_integers_product_plus_one_is_square_l368_36894

theorem consecutive_integers_product_plus_one_is_square (n : ℤ) : 
  n * (n + 1) * (n + 2) * (n + 3) + 1 = (n * (n + 3) + 1)^2 := by
  sorry

end consecutive_integers_product_plus_one_is_square_l368_36894


namespace remaining_money_calculation_l368_36838

/-- Calculates the remaining money after expenses given a salary and expense ratios -/
def remaining_money (salary : ℚ) (food_ratio : ℚ) (rent_ratio : ℚ) (clothes_ratio : ℚ) : ℚ :=
  salary * (1 - (food_ratio + rent_ratio + clothes_ratio))

/-- Theorem stating that given the specific salary and expense ratios, the remaining money is 17000 -/
theorem remaining_money_calculation :
  remaining_money 170000 (1/5) (1/10) (3/5) = 17000 := by
  sorry

end remaining_money_calculation_l368_36838


namespace riverbend_prep_distance_l368_36867

/-- Represents a relay race team -/
structure RelayTeam where
  name : String
  members : Nat
  raceLength : Nat

/-- Calculates the total distance covered by a relay team -/
def totalDistance (team : RelayTeam) : Nat :=
  team.members * team.raceLength

/-- Theorem stating that the total distance covered by Riverbend Prep is 1500 meters -/
theorem riverbend_prep_distance :
  let riverbendPrep : RelayTeam := ⟨"Riverbend Prep", 6, 250⟩
  totalDistance riverbendPrep = 1500 := by sorry

end riverbend_prep_distance_l368_36867


namespace intersection_when_a_is_half_intersection_empty_iff_l368_36817

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < 2*a + 1}
def B : Set ℝ := {x | 0 < x ∧ x < 1}

-- Theorem 1: When a = 1/2, A ∩ B = {x | 0 < x < 1}
theorem intersection_when_a_is_half : 
  A (1/2) ∩ B = {x : ℝ | 0 < x ∧ x < 1} := by sorry

-- Theorem 2: A ∩ B = ∅ if and only if a ≤ -1/2 or a ≥ 2
theorem intersection_empty_iff : 
  ∀ a : ℝ, A a ∩ B = ∅ ↔ a ≤ -1/2 ∨ a ≥ 2 := by sorry

end intersection_when_a_is_half_intersection_empty_iff_l368_36817


namespace inequality_proof_l368_36877

theorem inequality_proof (x y : ℝ) (h : 3 * x^2 + 2 * y^2 ≤ 6) : 2 * x + y ≤ Real.sqrt 11 := by
  sorry

end inequality_proof_l368_36877


namespace special_numbers_l368_36832

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def digit_sum (n : ℕ) : ℕ :=
  (n / 1000) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

def satisfies_conditions (n : ℕ) : Prop :=
  is_four_digit n ∧ n % 11 = 0 ∧ digit_sum n = 11

theorem special_numbers :
  {n : ℕ | satisfies_conditions n} =
  {2090, 3080, 4070, 5060, 6050, 7040, 8030, 9020} := by sorry

end special_numbers_l368_36832


namespace asymptotes_of_specific_hyperbola_l368_36811

/-- Represents a hyperbola in the xy-plane -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos : a > 0 ∧ b > 0

/-- The equation of a hyperbola in standard form -/
def hyperbola_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / h.a^2 - y^2 / h.b^2 = 1

/-- The equation of the asymptotes of a hyperbola -/
def asymptote_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  y = h.b / h.a * x ∨ y = -h.b / h.a * x

/-- The specific hyperbola we're interested in -/
def specific_hyperbola : Hyperbola where
  a := 1
  b := 2
  h_pos := by simp

theorem asymptotes_of_specific_hyperbola :
  ∀ x y : ℝ, asymptote_equation specific_hyperbola x y ↔ (y = 2*x ∨ y = -2*x) :=
sorry

end asymptotes_of_specific_hyperbola_l368_36811


namespace equation_solution_l368_36861

theorem equation_solution :
  ∃ (x : ℝ), x > 0 ∧ 4 * Real.sqrt (9 + x) + 4 * Real.sqrt (9 - x) = 10 * Real.sqrt 3 ∧
  x = Real.sqrt 80.859375 := by
  sorry

end equation_solution_l368_36861


namespace sin_6theta_l368_36834

theorem sin_6theta (θ : ℝ) :
  Complex.exp (θ * Complex.I) = (3 + Complex.I * Real.sqrt 8) / 4 →
  Real.sin (6 * θ) = -855 * Real.sqrt 2 / 1024 := by
  sorry

end sin_6theta_l368_36834
