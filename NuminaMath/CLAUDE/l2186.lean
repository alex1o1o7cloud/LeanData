import Mathlib

namespace diagonals_29_sided_polygon_l2186_218651

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A convex polygon with 29 sides has 377 diagonals -/
theorem diagonals_29_sided_polygon : num_diagonals 29 = 377 := by
  sorry

end diagonals_29_sided_polygon_l2186_218651


namespace fifth_month_sale_l2186_218611

def sales_first_four : List ℕ := [6235, 6927, 6855, 7230]
def required_sixth : ℕ := 5191
def desired_average : ℕ := 6500
def num_months : ℕ := 6

theorem fifth_month_sale :
  let total_required : ℕ := desired_average * num_months
  let sum_known : ℕ := (sales_first_four.sum + required_sixth)
  let fifth_month : ℕ := total_required - sum_known
  fifth_month = 6562 := by sorry

end fifth_month_sale_l2186_218611


namespace winnie_the_pooh_honey_l2186_218608

def honey_pot (initial_weight : ℝ) (empty_pot_weight : ℝ) : Prop :=
  ∃ (w1 w2 w3 w4 : ℝ),
    w1 = initial_weight / 2 ∧
    w2 = w1 / 2 ∧
    w3 = w2 / 2 ∧
    w4 = w3 / 2 ∧
    w4 = empty_pot_weight

theorem winnie_the_pooh_honey (empty_pot_weight : ℝ) 
  (h1 : empty_pot_weight = 200) : 
  ∃ (initial_weight : ℝ), 
    honey_pot initial_weight empty_pot_weight ∧ 
    initial_weight - empty_pot_weight = 3000 := by
  sorry

end winnie_the_pooh_honey_l2186_218608


namespace three_tangent_circles_range_l2186_218658

/-- Two circles with exactly three common tangents -/
structure ThreeTangentCircles where
  a : ℝ
  b : ℝ
  c1 : (x : ℝ) → (y : ℝ) → Prop := λ x y ↦ (x - a)^2 + y^2 = 1
  c2 : (x : ℝ) → (y : ℝ) → Prop := λ x y ↦ x^2 + y^2 - 2*b*y + b^2 - 4 = 0
  three_tangents : ∃! (p : ℝ × ℝ), c1 p.1 p.2 ∧ c2 p.1 p.2

/-- The range of a² + b² - 6a - 8b for circles with three common tangents -/
theorem three_tangent_circles_range (circles : ThreeTangentCircles) :
  -21 ≤ circles.a^2 + circles.b^2 - 6*circles.a - 8*circles.b ∧
  circles.a^2 + circles.b^2 - 6*circles.a - 8*circles.b ≤ 39 := by
  sorry

end three_tangent_circles_range_l2186_218658


namespace prime_square_plus_eight_l2186_218631

theorem prime_square_plus_eight (p : ℕ) : 
  Nat.Prime p → (Nat.Prime (p^2 + 8) ↔ p = 3) := by sorry

end prime_square_plus_eight_l2186_218631


namespace angle_complementary_to_complement_l2186_218687

theorem angle_complementary_to_complement (α : ℝ) : 
  (90 - α) + (180 - α) = 180 → α = 45 := by
  sorry

end angle_complementary_to_complement_l2186_218687


namespace deli_sandwich_count_l2186_218661

-- Define the types of sandwich components
structure SandwichComponents where
  breads : Nat
  meats : Nat
  cheeses : Nat

-- Define the forbidden combinations
structure ForbiddenCombinations where
  ham_cheddar : Nat
  white_chicken : Nat
  turkey_swiss : Nat

-- Define the function to calculate the number of possible sandwiches
def calculate_sandwiches (components : SandwichComponents) (forbidden : ForbiddenCombinations) : Nat :=
  components.breads * components.meats * components.cheeses - 
  (forbidden.ham_cheddar + forbidden.white_chicken + forbidden.turkey_swiss)

-- Theorem statement
theorem deli_sandwich_count :
  let components := SandwichComponents.mk 5 7 6
  let forbidden := ForbiddenCombinations.mk 5 6 5
  calculate_sandwiches components forbidden = 194 := by
  sorry


end deli_sandwich_count_l2186_218661


namespace number_equation_solution_l2186_218691

theorem number_equation_solution : 
  ∃ (N : ℝ), (16/100) * (40/100) * N = 5 * (8/100) * N ∧ N = 6.25 := by
  sorry

end number_equation_solution_l2186_218691


namespace triangle_formation_l2186_218649

/-- Checks if three line segments can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem triangle_formation :
  can_form_triangle 8 6 3 ∧
  ¬can_form_triangle 2 2 4 ∧
  ¬can_form_triangle 2 6 3 ∧
  ¬can_form_triangle 11 4 6 :=
by sorry

end triangle_formation_l2186_218649


namespace min_value_cyclic_fraction_min_value_cyclic_fraction_achievable_l2186_218620

theorem min_value_cyclic_fraction (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) : 
  a / b + b / c + c / d + d / a ≥ 4 :=
by sorry

theorem min_value_cyclic_fraction_achievable :
  ∃ (a b c d : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
  a / b + b / c + c / d + d / a = 4 :=
by sorry

end min_value_cyclic_fraction_min_value_cyclic_fraction_achievable_l2186_218620


namespace solution_set_f_leq_2abs_condition_on_abc_l2186_218696

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + |x - 2|

-- Theorem 1: Solution set of f(x) ≤ 2|x|
theorem solution_set_f_leq_2abs (x : ℝ) :
  x ∈ {y : ℝ | f y ≤ 2 * |y|} ↔ x ∈ Set.Icc 1 2 :=
sorry

-- Theorem 2: Condition on a, b, c
theorem condition_on_abc (a b c : ℝ) :
  (∀ x : ℝ, f x ≥ a^2 + 4*b^2 + 5*c^2 - 1/4) → a*c + 4*b*c ≤ 1 :=
sorry

end solution_set_f_leq_2abs_condition_on_abc_l2186_218696


namespace line_through_origin_and_intersection_l2186_218648

/-- The equation of the line passing through the origin and the intersection of two given lines -/
theorem line_through_origin_and_intersection (x y : ℝ) : 
  (x - 2*y + 2 = 0) →  -- Line 1 equation
  (2*x - y - 2 = 0) →  -- Line 2 equation
  (∃ t : ℝ, x = t ∧ y = t) -- Equation of the line y = x in parametric form
  := by sorry

end line_through_origin_and_intersection_l2186_218648


namespace complex_equation_solution_l2186_218652

theorem complex_equation_solution (z : ℂ) : (Complex.I * z = 1 - Complex.I) → z = -1 - Complex.I := by
  sorry

end complex_equation_solution_l2186_218652


namespace polynomial_coefficient_B_l2186_218693

theorem polynomial_coefficient_B (E F G : ℤ) :
  ∃ (r₁ r₂ r₃ r₄ r₅ r₆ : ℕ+),
    (∀ z : ℂ, z^6 - 15*z^5 + E*z^4 + (-287)*z^3 + F*z^2 + G*z + 64 = 
      (z - r₁) * (z - r₂) * (z - r₃) * (z - r₄) * (z - r₅) * (z - r₆)) ∧
    (r₁ + r₂ + r₃ + r₄ + r₅ + r₆ = 15) :=
  sorry

end polynomial_coefficient_B_l2186_218693


namespace intersection_distance_squared_is_zero_l2186_218679

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The square of the distance between two points in 2D space -/
def distanceSquared (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

/-- Determines if a point lies on a circle -/
def isOnCircle (p : ℝ × ℝ) (c : Circle) : Prop :=
  distanceSquared p c.center = c.radius^2

/-- The main theorem: The square of the distance between intersection points of two specific circles is 0 -/
theorem intersection_distance_squared_is_zero (c1 c2 : Circle)
    (h1 : c1 = { center := (3, -2), radius := 5 })
    (h2 : c2 = { center := (3, 6), radius := 3 }) :
    ∀ p1 p2 : ℝ × ℝ, isOnCircle p1 c1 ∧ isOnCircle p1 c2 ∧ isOnCircle p2 c1 ∧ isOnCircle p2 c2 →
    distanceSquared p1 p2 = 0 := by
  sorry


end intersection_distance_squared_is_zero_l2186_218679


namespace burning_candle_variables_l2186_218695

/-- Represents a burning candle -/
structure BurningCandle where
  a : ℝ  -- Original length in centimeters
  t : ℝ  -- Burning time in minutes
  y : ℝ  -- Remaining length in centimeters

/-- Predicate to check if a quantity is variable in the context of a burning candle -/
def isVariable (candle : BurningCandle) (quantity : ℝ) : Prop :=
  ∃ (candle' : BurningCandle), candle.a = candle'.a ∧ quantity ≠ candle'.t

theorem burning_candle_variables (candle : BurningCandle) :
  (isVariable candle candle.t ∧ isVariable candle candle.y) ∧
  ¬(isVariable candle candle.a) := by
  sorry

#check burning_candle_variables

end burning_candle_variables_l2186_218695


namespace mark_milk_purchase_l2186_218684

def problem (soup_price : ℕ) (soup_quantity : ℕ) (bread_price : ℕ) (bread_quantity : ℕ) 
             (cereal_price : ℕ) (cereal_quantity : ℕ) (milk_price : ℕ) (bill_value : ℕ) 
             (bill_quantity : ℕ) : ℕ :=
  let total_paid := bill_value * bill_quantity
  let other_items_cost := soup_price * soup_quantity + bread_price * bread_quantity + cereal_price * cereal_quantity
  let milk_total_cost := total_paid - other_items_cost
  milk_total_cost / milk_price

theorem mark_milk_purchase :
  problem 2 6 5 2 3 2 4 10 4 = 3 := by
  sorry

end mark_milk_purchase_l2186_218684


namespace max_threshold_price_l2186_218653

/-- Represents a company with a product line -/
structure Company where
  num_products : ℕ
  avg_price : ℝ
  min_price : ℝ
  max_price : ℝ
  num_below_threshold : ℕ

/-- The threshold price for a given company -/
def threshold_price (c : Company) : ℝ := sorry

theorem max_threshold_price (c : Company) :
  c.num_products = 25 →
  c.avg_price = 1200 →
  c.min_price = 400 →
  c.max_price = 13200 →
  c.num_below_threshold = 12 →
  threshold_price c ≤ 700 ∧
  ∀ t, t > 700 → ¬(threshold_price c = t) := by
  sorry

#check max_threshold_price

end max_threshold_price_l2186_218653


namespace problem_solution_l2186_218689

theorem problem_solution (a b c d m n : ℕ+) 
  (h1 : a^2 + b^2 + c^2 + d^2 = 1989)
  (h2 : a + b + c + d = m^2)
  (h3 : max a (max b (max c d)) = n^2) :
  m = 9 ∧ n = 6 := by
  sorry

end problem_solution_l2186_218689


namespace paige_albums_l2186_218601

def number_of_albums (total_pictures : ℕ) (first_album_pictures : ℕ) (pictures_per_album : ℕ) : ℕ :=
  (total_pictures - first_album_pictures) / pictures_per_album

theorem paige_albums : number_of_albums 35 14 7 = 3 := by
  sorry

end paige_albums_l2186_218601


namespace roses_unchanged_l2186_218610

def initial_roses : ℕ := 12
def initial_orchids : ℕ := 2
def final_orchids : ℕ := 21
def cut_orchids : ℕ := 19

theorem roses_unchanged (h : final_orchids - cut_orchids = initial_orchids) :
  initial_roses = initial_roses := by sorry

end roses_unchanged_l2186_218610


namespace simplify_expression_l2186_218665

theorem simplify_expression (x : ℝ) : 8*x + 15 - 3*x + 5 * 7 = 5*x + 50 := by
  sorry

end simplify_expression_l2186_218665


namespace max_sum_cubes_l2186_218604

theorem max_sum_cubes (x y z w : ℝ) (h : x^2 + y^2 + z^2 + w^2 = 16) :
  ∃ (M : ℝ), (∀ a b c d : ℝ, a^2 + b^2 + c^2 + d^2 = 16 → a^3 + b^3 + c^3 + d^3 ≤ M) ∧
             (∃ p q r s : ℝ, p^2 + q^2 + r^2 + s^2 = 16 ∧ p^3 + q^3 + r^3 + s^3 = M) ∧
             M = 64 :=
by sorry

end max_sum_cubes_l2186_218604


namespace deepak_current_age_l2186_218654

/-- Proves Deepak's current age given the ratio of ages and Arun's future age -/
theorem deepak_current_age 
  (arun_age : ℕ) 
  (deepak_age : ℕ) 
  (h1 : arun_age + 5 = 25) 
  (h2 : arun_age * 3 = deepak_age * 2) : 
  deepak_age = 30 := by
sorry

end deepak_current_age_l2186_218654


namespace theresa_extra_games_video_game_comparison_l2186_218634

-- Define the number of video games for each person
def tory_games : ℕ := 6
def theresa_games : ℕ := 11

-- Define the relationship between Julia's and Tory's games
def julia_games : ℕ := tory_games / 3

-- Define the relationship between Theresa's and Julia's games
def theresa_more_than_thrice_julia : Prop :=
  theresa_games > 3 * julia_games

-- Theorem to prove
theorem theresa_extra_games :
  theresa_games - 3 * julia_games = 5 :=
by
  sorry

-- Main theorem that encapsulates the problem
theorem video_game_comparison
  (h1 : theresa_more_than_thrice_julia)
  (h2 : julia_games = tory_games / 3)
  (h3 : tory_games = 6)
  (h4 : theresa_games = 11) :
  theresa_games - 3 * julia_games = 5 :=
by
  sorry

end theresa_extra_games_video_game_comparison_l2186_218634


namespace rates_sum_of_squares_l2186_218683

theorem rates_sum_of_squares : ∃ (b j s h : ℕ),
  (3 * b + 4 * j + 2 * s + 3 * h = 120) ∧
  (2 * b + 3 * j + 4 * s + 3 * h = 150) ∧
  (b^2 + j^2 + s^2 + h^2 = 1850) :=
by sorry

end rates_sum_of_squares_l2186_218683


namespace sin_cos_difference_simplification_l2186_218636

theorem sin_cos_difference_simplification :
  Real.sin (72 * π / 180) * Real.cos (12 * π / 180) - 
  Real.cos (72 * π / 180) * Real.sin (12 * π / 180) = 
  Real.sqrt 3 / 2 := by sorry

end sin_cos_difference_simplification_l2186_218636


namespace prob_TT_after_second_H_l2186_218609

/-- A fair coin flip sequence that stops when two consecutive flips are the same -/
inductive CoinFlipSequence
  | HH
  | TT
  | HTH : CoinFlipSequence → CoinFlipSequence
  | HTT : CoinFlipSequence

/-- The probability of a coin flip sequence -/
def prob : CoinFlipSequence → ℚ
  | CoinFlipSequence.HH => 1/4
  | CoinFlipSequence.TT => 1/4
  | CoinFlipSequence.HTH s => (1/8) * prob s
  | CoinFlipSequence.HTT => 1/8

/-- The probability of getting two tails in a row but seeing a second head before seeing a second tail -/
def probTTAfterSecondH : ℚ := prob CoinFlipSequence.HTT

theorem prob_TT_after_second_H : probTTAfterSecondH = 1/24 := by
  sorry

end prob_TT_after_second_H_l2186_218609


namespace total_food_is_338_l2186_218663

/-- The maximum amount of food (in pounds) consumed by an individual guest -/
def max_food_per_guest : ℝ := 2

/-- The minimum number of guests that attended the banquet -/
def min_guests : ℕ := 169

/-- The total amount of food consumed by all guests (in pounds) -/
def total_food_consumed : ℝ := min_guests * max_food_per_guest

/-- Theorem: The total amount of food consumed is 338 pounds -/
theorem total_food_is_338 : total_food_consumed = 338 := by
  sorry

end total_food_is_338_l2186_218663


namespace rectangle_area_increase_l2186_218656

theorem rectangle_area_increase (initial_length initial_width : ℝ) 
  (h_positive_length : initial_length > 0)
  (h_positive_width : initial_width > 0) :
  let increase_factor := 1.44
  let side_increase_factor := Real.sqrt increase_factor
  side_increase_factor = 1.2 := by sorry

end rectangle_area_increase_l2186_218656


namespace largest_square_tile_l2186_218686

theorem largest_square_tile (board_width board_length : ℕ) 
  (hw : board_width = 17) (hl : board_length = 23) :
  Nat.gcd board_width board_length = 1 := by
  sorry

end largest_square_tile_l2186_218686


namespace empty_solution_set_range_l2186_218662

theorem empty_solution_set_range (a : ℝ) : 
  (∀ x : ℝ, (a^2 - 4) * x^2 + (a + 2) * x - 1 < 0) ↔ (-2 < a ∧ a ≤ 6/5) :=
sorry

end empty_solution_set_range_l2186_218662


namespace exp_greater_than_power_over_factorial_l2186_218628

theorem exp_greater_than_power_over_factorial
  (x : ℝ) (n : ℕ) (h1 : x > 1) (h2 : n > 0) :
  Real.exp (x - 1) > x ^ n / n.factorial :=
sorry

end exp_greater_than_power_over_factorial_l2186_218628


namespace parabola_point_coordinates_l2186_218606

theorem parabola_point_coordinates :
  ∀ (x y : ℝ),
  (y = 2 * x^2) →                          -- M is on the parabola y = 2x^2
  (x > 0 ∧ y > 0) →                        -- M is in the first quadrant
  ((x - 0)^2 + (y - 1/8)^2 = (1/4)^2) →    -- Distance from M to focus is 1/4
  (x = Real.sqrt 2 / 8 ∧ y = 1/16) := by
sorry

end parabola_point_coordinates_l2186_218606


namespace quadrilateral_area_l2186_218619

/-- Represents a triangle partitioned into three triangles and a quadrilateral -/
structure PartitionedTriangle where
  /-- Area of the first triangle -/
  area1 : ℝ
  /-- Area of the second triangle -/
  area2 : ℝ
  /-- Area of the third triangle -/
  area3 : ℝ
  /-- Area of the quadrilateral -/
  areaQuad : ℝ

/-- The theorem stating that if the areas of the three triangles are 3, 7, and 7,
    then the area of the quadrilateral is 18 -/
theorem quadrilateral_area (t : PartitionedTriangle) 
    (h1 : t.area1 = 3) 
    (h2 : t.area2 = 7) 
    (h3 : t.area3 = 7) : 
    t.areaQuad = 18 := by
  sorry


end quadrilateral_area_l2186_218619


namespace negation_equivalence_l2186_218680

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x^2 + 1 > 2*x) ↔ (∃ x : ℝ, x^2 + 1 ≤ 2*x) := by
  sorry

end negation_equivalence_l2186_218680


namespace simplified_fraction_double_l2186_218600

theorem simplified_fraction_double (b : ℝ) :
  b = 5 → 2 * ((15 * b^4) / (75 * b^3)) = 2 := by
  sorry

end simplified_fraction_double_l2186_218600


namespace plane_through_skew_perp_existence_l2186_218639

-- Define the concept of skew lines
def are_skew (a b : Line3D) : Prop := sorry

-- Define the concept of perpendicular lines
def are_perpendicular (l1 l2 : Line3D) : Prop := sorry

-- Define a plane passing through a line and perpendicular to another line
def plane_through_perp_to (a b : Line3D) : Set Point3D := sorry

theorem plane_through_skew_perp_existence (a b : Line3D) 
  (h_skew : are_skew a b) : 
  (∃! p : Set Point3D, p = plane_through_perp_to a b) ↔ are_perpendicular a b :=
sorry

end plane_through_skew_perp_existence_l2186_218639


namespace no_seven_edge_polyhedron_l2186_218650

/-- A polyhedron in three-dimensional space. -/
structure Polyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  euler : vertices - edges + faces = 2
  min_degree : edges * 2 ≥ vertices * 3

/-- Theorem stating that no polyhedron can have exactly seven edges. -/
theorem no_seven_edge_polyhedron :
  ¬∃ (p : Polyhedron), p.edges = 7 := by
  sorry

end no_seven_edge_polyhedron_l2186_218650


namespace cans_distribution_l2186_218670

theorem cans_distribution (father_weight son_weight : ℕ) 
  (h1 : father_weight = 6500)
  (h2 : son_weight = 2600) : 
  ∃ (can_weight : ℕ), 
    300 ≤ can_weight ∧ 
    can_weight ≤ 400 ∧
    father_weight % can_weight = 0 ∧
    son_weight % can_weight = 0 ∧
    father_weight / can_weight = 20 ∧
    son_weight / can_weight = 8 := by
  sorry

end cans_distribution_l2186_218670


namespace bianca_coloring_books_l2186_218688

/-- The number of coloring books Bianca initially had -/
def initial_books : ℕ := 45

/-- The number of books Bianca gave away -/
def books_given_away : ℕ := 6

/-- The number of books Bianca bought -/
def books_bought : ℕ := 20

/-- The total number of books Bianca has after the transactions -/
def final_books : ℕ := 59

/-- Theorem stating that the initial number of books is correct -/
theorem bianca_coloring_books : 
  initial_books - books_given_away + books_bought = final_books := by
  sorry

#check bianca_coloring_books

end bianca_coloring_books_l2186_218688


namespace negation_of_proposition_l2186_218632

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, x > 0 → x^2 + x ≥ 0) ↔ (∃ x₀ : ℝ, x₀ > 0 ∧ x₀^2 + x₀ < 0) :=
by sorry

end negation_of_proposition_l2186_218632


namespace twentieth_group_number_l2186_218669

/-- Represents the total number of students -/
def total_students : ℕ := 400

/-- Represents the number of groups -/
def num_groups : ℕ := 20

/-- Represents the first group's drawn number -/
def first_group_number : ℕ := 11

/-- Calculates the drawn number for a given group -/
def drawn_number (group : ℕ) : ℕ :=
  first_group_number + (group - 1) * num_groups

/-- Theorem stating that the 20th group's drawn number is 391 -/
theorem twentieth_group_number :
  drawn_number num_groups = 391 := by
  sorry

end twentieth_group_number_l2186_218669


namespace largest_even_digit_number_with_four_proof_largest_even_digit_number_with_four_l2186_218692

def is_even_digit (d : ℕ) : Prop := d % 2 = 0 ∧ d < 10

def all_even_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → is_even_digit d

def contains_digit (n d : ℕ) : Prop := d ∈ n.digits 10

theorem largest_even_digit_number_with_four (n : ℕ) : Prop :=
  n = 5408 ∧
  all_even_digits n ∧
  contains_digit n 4 ∧
  n < 6000 ∧
  n % 8 = 0 ∧
  ∀ m : ℕ, m ≠ n →
    (all_even_digits m ∧ contains_digit m 4 ∧ m < 6000 ∧ m % 8 = 0) →
    m < n

theorem proof_largest_even_digit_number_with_four :
  ∃ n : ℕ, largest_even_digit_number_with_four n :=
sorry

end largest_even_digit_number_with_four_proof_largest_even_digit_number_with_four_l2186_218692


namespace carlton_outfits_l2186_218627

/-- The number of outfits Carlton has -/
def number_of_outfits (button_up_shirts : ℕ) : ℕ :=
  (2 * button_up_shirts) * button_up_shirts

/-- Theorem stating that Carlton has 18 outfits -/
theorem carlton_outfits : number_of_outfits 3 = 18 := by
  sorry

end carlton_outfits_l2186_218627


namespace circle_center_coordinate_difference_l2186_218616

/-- Given two points that are the endpoints of a circle's diameter,
    calculate the difference between the x and y coordinates of the center. -/
theorem circle_center_coordinate_difference
  (p1 : ℝ × ℝ) (p2 : ℝ × ℝ)
  (h1 : p1 = (10, -6))
  (h2 : p2 = (-2, 2))
  : (p1.1 + p2.1) / 2 - (p1.2 + p2.2) / 2 = 6 := by
  sorry

end circle_center_coordinate_difference_l2186_218616


namespace one_minus_repeating_third_eq_two_thirds_l2186_218630

/-- The value of the repeating decimal 0.333... --/
def repeating_third : ℚ := 1 / 3

/-- Theorem stating that 1 minus the repeating decimal 0.333... equals 2/3 --/
theorem one_minus_repeating_third_eq_two_thirds :
  1 - repeating_third = 2 / 3 := by sorry

end one_minus_repeating_third_eq_two_thirds_l2186_218630


namespace product_evaluation_l2186_218668

theorem product_evaluation (n : ℕ) (h : n = 3) : 
  (n - 2) * (n - 1) * n * (n + 1) * (n + 2) * (n + 3) * (n + 4) = 5040 := by
  sorry

end product_evaluation_l2186_218668


namespace range_of_a_l2186_218698

def A : Set ℝ := {x | 2 < x ∧ x < 8}
def B (a : ℝ) : Set ℝ := {x | a < x ∧ x < 2*a - 2}

theorem range_of_a (a : ℝ) : B a ⊆ A → a ≤ 5 := by sorry

end range_of_a_l2186_218698


namespace sum_of_real_solutions_l2186_218699

theorem sum_of_real_solutions (b : ℝ) (h : b > 2) :
  ∃ y : ℝ, y ≥ 0 ∧ Real.sqrt (b - Real.sqrt (b + y)) = y ∧
  y = (Real.sqrt (4 * b - 3) - 1) / 2 :=
by sorry

end sum_of_real_solutions_l2186_218699


namespace square_difference_theorem_l2186_218643

theorem square_difference_theorem :
  (41 : ℕ)^2 = 40^2 + 81 ∧ 39^2 = 40^2 - 79 := by
  sorry

end square_difference_theorem_l2186_218643


namespace horner_method_v2_l2186_218642

def f (x : ℝ) : ℝ := x^6 - 8*x^5 + 60*x^4 + 16*x^3 + 96*x^2 + 240*x + 64

def horner_v2 (a : ℝ) : ℝ :=
  let v0 := 1
  let v1 := v0 * a - 8
  v1 * a + 60

theorem horner_method_v2 :
  horner_v2 2 = 48 :=
by sorry

end horner_method_v2_l2186_218642


namespace smallest_divisible_by_15_16_18_l2186_218666

theorem smallest_divisible_by_15_16_18 : ∃ (n : ℕ), n > 0 ∧ 15 ∣ n ∧ 16 ∣ n ∧ 18 ∣ n ∧ ∀ (m : ℕ), m > 0 → 15 ∣ m → 16 ∣ m → 18 ∣ m → n ≤ m :=
by sorry

end smallest_divisible_by_15_16_18_l2186_218666


namespace sector_area_l2186_218625

/-- The area of a circular sector with central angle 60° and radius 10 cm is 50π/3 cm² -/
theorem sector_area (θ : Real) (r : Real) (h1 : θ = 60) (h2 : r = 10) :
  (θ / 360) * π * r^2 = 50 * π / 3 := by
  sorry

end sector_area_l2186_218625


namespace fraction_equals_zero_l2186_218638

theorem fraction_equals_zero (x : ℝ) (h : x ≠ 0) :
  (x - 5) / (6 * x) = 0 ↔ x = 5 := by
  sorry

end fraction_equals_zero_l2186_218638


namespace radical_calculations_l2186_218613

theorem radical_calculations :
  (∃ x y : ℝ, x^2 = 3 ∧ y^2 = 2 ∧
    (Real.sqrt 48 + Real.sqrt 8 - Real.sqrt 18 - Real.sqrt 12 = 2*x - y)) ∧
  (∃ a b c : ℝ, a^2 = 2 ∧ b^2 = 3 ∧ c^2 = 6 ∧
    (2*(a + b) - (b - a)^2 = 2*a + 2*b + 2*c - 5)) :=
by sorry

end radical_calculations_l2186_218613


namespace company_fund_problem_l2186_218622

theorem company_fund_problem (n : ℕ) : 
  (∀ (initial_fund : ℕ),
    initial_fund = 60 * n - 10 ∧ 
    initial_fund = 50 * n + 110) →
  60 * n - 10 = 710 := by
  sorry

end company_fund_problem_l2186_218622


namespace newspapers_julie_can_print_l2186_218644

-- Define the given conditions
def boxes : ℕ := 2
def packages_per_box : ℕ := 5
def sheets_per_package : ℕ := 250
def sheets_per_newspaper : ℕ := 25

-- Define the theorem
theorem newspapers_julie_can_print :
  (boxes * packages_per_box * sheets_per_package) / sheets_per_newspaper = 100 := by
  sorry

end newspapers_julie_can_print_l2186_218644


namespace reciprocal_sum_equality_implies_zero_product_l2186_218633

theorem reciprocal_sum_equality_implies_zero_product
  (a b c : ℝ) (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) 
  (eq : 1/a + 1/b + 1/c = 1/(a+b+c)) :
  (a+b)*(b+c)*(a+c) = 0 := by
sorry

end reciprocal_sum_equality_implies_zero_product_l2186_218633


namespace at_least_one_black_probability_l2186_218690

def total_balls : ℕ := 4
def white_balls : ℕ := 2
def black_balls : ℕ := 2
def drawn_balls : ℕ := 2

def probability_at_least_one_black : ℚ := 5 / 6

theorem at_least_one_black_probability :
  probability_at_least_one_black = 
    (Nat.choose total_balls drawn_balls - Nat.choose white_balls drawn_balls) / 
    Nat.choose total_balls drawn_balls :=
by sorry

end at_least_one_black_probability_l2186_218690


namespace fathers_age_l2186_218675

theorem fathers_age (S F : ℕ) 
  (h1 : 2 * S + F = 70) 
  (h2 : S + 2 * F = 95) : 
  F = 40 := by
sorry

end fathers_age_l2186_218675


namespace polynomial_factorization_l2186_218678

theorem polynomial_factorization (x y : ℝ) :
  x^4 + 4*y^4 = (x^2 - 2*x*y + 2*y^2) * (x^2 + 2*x*y + 2*y^2) := by
  sorry

end polynomial_factorization_l2186_218678


namespace part_one_part_two_part_two_converse_l2186_218681

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | 3 - a ≤ x ∧ x ≤ 2 + a}
def B : Set ℝ := {x | x < 1 ∨ x > 6}

-- Part 1
theorem part_one : A 3 ∩ (Set.univ \ B) = {x | 1 ≤ x ∧ x ≤ 5} := by sorry

-- Part 2
theorem part_two (a : ℝ) (h1 : a > 0) (h2 : A a ∩ B = ∅) :
  a ∈ {x | 0 < x ∧ x ≤ 2} := by sorry

theorem part_two_converse (a : ℝ) (h : a ∈ {x | 0 < x ∧ x ≤ 2}) :
  a > 0 ∧ A a ∩ B = ∅ := by sorry

end part_one_part_two_part_two_converse_l2186_218681


namespace sum_of_three_numbers_l2186_218635

theorem sum_of_three_numbers (x y z : ℝ) 
  (sum_xy : x + y = 31)
  (sum_yz : y + z = 41)
  (sum_zx : z + x = 55) :
  x + y + z = 63.5 := by
  sorry

end sum_of_three_numbers_l2186_218635


namespace number_calculation_l2186_218607

theorem number_calculation (N : ℝ) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N = 10 → (40/100 : ℝ) * N = 120 := by
  sorry

end number_calculation_l2186_218607


namespace cuboid_edge_length_l2186_218694

/-- Given a cuboid with two edges of 6 cm and a volume of 180 cm³, 
    the length of the third edge is 5 cm. -/
theorem cuboid_edge_length (edge1 edge3 volume : ℝ) : 
  edge1 = 6 → edge3 = 6 → volume = 180 → 
  ∃ edge2 : ℝ, edge1 * edge2 * edge3 = volume ∧ edge2 = 5 := by
  sorry

end cuboid_edge_length_l2186_218694


namespace inequality_proof_l2186_218674

theorem inequality_proof (a b : ℝ) (h1 : a < b) (h2 : b < 0) :
  1 / (a - b) < 1 / a :=
by sorry

end inequality_proof_l2186_218674


namespace savings_proof_l2186_218605

/-- Calculates savings given income and expenditure ratio -/
def calculate_savings (income : ℕ) (income_ratio expenditure_ratio : ℕ) : ℕ :=
  income - (income * expenditure_ratio) / income_ratio

/-- Proves that given the specified conditions, the savings are 3400 -/
theorem savings_proof (income : ℕ) (income_ratio expenditure_ratio : ℕ) 
  (h1 : income = 17000)
  (h2 : income_ratio = 5)
  (h3 : expenditure_ratio = 4) :
  calculate_savings income income_ratio expenditure_ratio = 3400 := by
  sorry

#eval calculate_savings 17000 5 4

end savings_proof_l2186_218605


namespace correct_bill_writing_l2186_218624

/-- Represents the monthly electricity bill in yuan -/
def monthly_bill : ℚ := 71.08

/-- The correct way to write the monthly electricity bill -/
def correct_writing : String := "71.08"

/-- Theorem stating that the correct way to write the monthly electricity bill is "71.08" -/
theorem correct_bill_writing : 
  toString monthly_bill = correct_writing := by sorry

end correct_bill_writing_l2186_218624


namespace units_digit_of_3968_pow_805_l2186_218660

theorem units_digit_of_3968_pow_805 : (3968^805) % 10 = 8 := by
  sorry

end units_digit_of_3968_pow_805_l2186_218660


namespace minute_hand_distance_l2186_218612

theorem minute_hand_distance (hand_length : ℝ) (time : ℝ) : 
  hand_length = 8 → time = 45 → 
  2 * π * hand_length * (time / 60) = 12 * π := by
sorry

end minute_hand_distance_l2186_218612


namespace perpendicular_line_x_intercept_l2186_218685

/-- The slope of the original line -/
def m₁ : ℚ := 4 / 3

/-- The slope of the perpendicular line -/
def m₂ : ℚ := -3 / 4

/-- The area of the triangle formed by the line and the coordinate axes -/
def A : ℚ := 6

/-- The x-intercept of the perpendicular line -/
def x_intercept : Set ℚ := {4, -4}

theorem perpendicular_line_x_intercept :
  ∀ (C : ℚ), (3 * C / 4) * (C / 3) / 2 = A → C ∈ x_intercept :=
sorry

end perpendicular_line_x_intercept_l2186_218685


namespace juan_reads_9000_pages_l2186_218664

/-- Calculates the total pages Juan can read from three books given their page counts, reading rates, and lunch time constraints. -/
def total_pages_read (book1_pages book2_pages book3_pages : ℕ) 
                     (book1_rate book2_rate book3_rate : ℕ) 
                     (lunch_time : ℕ) : ℕ :=
  let book1_read_time := book1_pages / book1_rate
  let book2_read_time := book2_pages / book2_rate
  let book3_read_time := book3_pages / book3_rate
  let book1_lunch_time := book1_read_time / 2
  let book2_lunch_time := book2_read_time / 2
  let book3_lunch_time := book3_read_time / 2
  let total_lunch_time := book1_lunch_time + book2_lunch_time + book3_lunch_time
  let remaining_time1 := book1_lunch_time - lunch_time
  let remaining_time2 := book2_lunch_time
  let remaining_time3 := book3_lunch_time
  remaining_time1 * book1_rate + remaining_time2 * book2_rate + remaining_time3 * book3_rate

/-- Theorem stating that given the specific conditions in the problem, Juan can read 9000 pages. -/
theorem juan_reads_9000_pages : 
  total_pages_read 4000 6000 8000 60 40 30 4 = 9000 := by
  sorry

end juan_reads_9000_pages_l2186_218664


namespace solution_is_three_fourths_l2186_218615

/-- The sum of the series given the value of x -/
def seriesSum (x : ℝ) : ℝ := 1 + 4*x + 8*x^2 + 12*x^3 + 16*x^4 + 20*x^5 + 24*x^6 + 28*x^7 + 32*x^8 + 36*x^9 + 40*x^10

/-- The theorem stating that 3/4 is the solution to the equation -/
theorem solution_is_three_fourths :
  ∃ (x : ℝ), x = 3/4 ∧ seriesSum x = 76 ∧ abs x < 1 := by
  sorry

end solution_is_three_fourths_l2186_218615


namespace equation_solution_l2186_218672

theorem equation_solution : 
  {x : ℝ | (x^3 - x^2)/(x^2 + 2*x + 1) + x = -2} = {-1/2, 2} := by sorry

end equation_solution_l2186_218672


namespace system_solution_l2186_218623

theorem system_solution (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x^y = z) (eq2 : y^z = x) (eq3 : z^x = y) :
  x = 1 ∧ y = 1 ∧ z = 1 := by
  sorry

end system_solution_l2186_218623


namespace ellipse_with_y_axis_focus_l2186_218603

/-- Given that θ is an interior angle of a triangle ABC and sin θ + cos θ = 3/4,
    prove that x^2 * sin θ - y^2 * cos θ = 1 represents an ellipse with focus on the y-axis -/
theorem ellipse_with_y_axis_focus (θ : Real) (x y : Real) 
  (h1 : 0 < θ ∧ θ < π) -- θ is an interior angle of a triangle
  (h2 : Real.sin θ + Real.cos θ = 3/4) -- given condition
  (h3 : x^2 * Real.sin θ - y^2 * Real.cos θ = 1) -- equation of the curve
  : ∃ (a b : Real), 
    0 < b ∧ b < a ∧ 
    (x^2 / a^2) + (y^2 / b^2) = 1 ∧ 
    (a^2 - b^2) / a^2 > 0 :=
sorry

end ellipse_with_y_axis_focus_l2186_218603


namespace salary_problem_l2186_218614

theorem salary_problem (total_salary : ℝ) (a_spend_rate : ℝ) (b_spend_rate : ℝ) 
  (h1 : total_salary = 14000)
  (h2 : a_spend_rate = 0.8)
  (h3 : b_spend_rate = 0.85)
  (h4 : (1 - a_spend_rate) * (total_salary - b_salary) = (1 - b_spend_rate) * b_salary) :
  b_salary = 8000 :=
by sorry

end salary_problem_l2186_218614


namespace sandys_shopping_money_l2186_218645

/-- Sandy's shopping problem -/
theorem sandys_shopping_money (initial_amount : ℝ) (spent_percentage : ℝ) (amount_left : ℝ) :
  initial_amount = 200 →
  spent_percentage = 30 →
  amount_left = initial_amount - (spent_percentage / 100 * initial_amount) →
  amount_left = 140 :=
by sorry

end sandys_shopping_money_l2186_218645


namespace students_in_all_three_activities_l2186_218682

/-- Represents the number of students in each activity and their intersections -/
structure ActivityCounts where
  total : ℕ
  meditation : ℕ
  chess : ℕ
  sculpture : ℕ
  exactlyTwo : ℕ
  allThree : ℕ

/-- The conditions of the problem -/
def problemConditions : ActivityCounts where
  total := 25
  meditation := 15
  chess := 18
  sculpture := 11
  exactlyTwo := 6
  allThree := 0  -- This is what we need to prove

theorem students_in_all_three_activities :
  ∃ (c : ActivityCounts), c.total = 25 ∧
    c.meditation = 15 ∧
    c.chess = 18 ∧
    c.sculpture = 11 ∧
    c.exactlyTwo = 6 ∧
    c.allThree = 7 ∧
    c.total = (c.meditation + c.chess + c.sculpture - 2 * c.exactlyTwo - 3 * c.allThree) :=
  sorry


end students_in_all_three_activities_l2186_218682


namespace expression_evaluation_l2186_218671

theorem expression_evaluation : 
  (2019^3 - 3 * 2019^2 * 2020 + 3 * 2019 * 2020^2 - 2020^3 + 6) / (2019 * 2020) = 5 / (2019 * 2020) := by
  sorry

end expression_evaluation_l2186_218671


namespace range_of_a_plus_3b_l2186_218677

theorem range_of_a_plus_3b (a b : ℝ) 
  (h1 : -1 ≤ a + b ∧ a + b ≤ 1) 
  (h2 : 1 ≤ a - 2*b ∧ a - 2*b ≤ 3) : 
  -11/3 ≤ a + 3*b ∧ a + 3*b ≤ 1 := by
  sorry

end range_of_a_plus_3b_l2186_218677


namespace vector_addition_rule_l2186_218626

variable {V : Type*} [AddCommGroup V]

theorem vector_addition_rule (A B C : V) : 
  (C - A) + (B - C) = B - A :=
sorry

end vector_addition_rule_l2186_218626


namespace product_sequence_sum_l2186_218646

theorem product_sequence_sum (a b : ℕ) (h1 : a / 3 = 18) (h2 : b = a - 1) : a + b = 107 := by
  sorry

end product_sequence_sum_l2186_218646


namespace polynomial_division_l2186_218667

theorem polynomial_division (x : ℝ) (h : x ≠ 0) :
  (6 * x^4 - 4 * x^3 + 2 * x^2) / (2 * x^2) = 3 * x^2 - 2 * x + 1 := by
  sorry

end polynomial_division_l2186_218667


namespace arithmetic_geometric_mean_inequality_l2186_218657

theorem arithmetic_geometric_mean_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + b) / 2 ≥ (2 * a * b) / (a + b) := by
  sorry

end arithmetic_geometric_mean_inequality_l2186_218657


namespace factorization_of_2x_squared_minus_8_l2186_218647

theorem factorization_of_2x_squared_minus_8 (x : ℝ) : 2 * x^2 - 8 = 2 * (x - 2) * (x + 2) := by
  sorry

end factorization_of_2x_squared_minus_8_l2186_218647


namespace negation_existential_real_l2186_218618

theorem negation_existential_real (f : ℝ → ℝ) :
  (¬ ∃ x : ℝ, f x < 0) ↔ (∀ x : ℝ, f x ≥ 0) := by
  sorry

end negation_existential_real_l2186_218618


namespace max_value_cyclic_sum_equality_condition_l2186_218629

theorem max_value_cyclic_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq_3 : a + b + c = 3) : 
  (a / (a^3 + b^2 + c)) + (b / (b^3 + c^2 + a)) + (c / (c^3 + a^2 + b)) ≤ 1 :=
by sorry

theorem equality_condition (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq_3 : a + b + c = 3) : 
  (a / (a^3 + b^2 + c)) + (b / (b^3 + c^2 + a)) + (c / (c^3 + a^2 + b)) = 1 ↔ 
  a = 1 ∧ b = 1 ∧ c = 1 :=
by sorry

end max_value_cyclic_sum_equality_condition_l2186_218629


namespace least_subtraction_for_divisibility_problem_solution_l2186_218640

theorem least_subtraction_for_divisibility (n : ℕ) (p : ℕ) (h_prime : Nat.Prime p) :
  ∃ (x : ℕ), x < p ∧ (n - x) % p = 0 ∧ ∀ (y : ℕ), y < x → (n - y) % p ≠ 0 :=
by
  sorry

theorem problem_solution :
  let n := 724946
  let p := 37
  (∃ (x : ℕ), x < p ∧ (n - x) % p = 0 ∧ ∀ (y : ℕ), y < x → (n - y) % p ≠ 0) ∧
  (17 < p ∧ (n - 17) % p = 0 ∧ ∀ (y : ℕ), y < 17 → (n - y) % p ≠ 0) :=
by
  sorry

end least_subtraction_for_divisibility_problem_solution_l2186_218640


namespace block_depth_l2186_218602

theorem block_depth (cube_volume : ℕ) (length width : ℕ) (fewer_cubes : ℕ) (d : ℕ) : 
  cube_volume = 5 →
  length = 7 →
  width = 7 →
  fewer_cubes = 194 →
  length * width * d * cube_volume - fewer_cubes * cube_volume = length * width * (d - 1) * cube_volume →
  d = 5 :=
by sorry

end block_depth_l2186_218602


namespace august_calculator_problem_l2186_218621

theorem august_calculator_problem (a b c : ℕ) : 
  a = 600 →
  b = 2 * a →
  c = a + b - 400 →
  a + b + c = 3200 :=
by sorry

end august_calculator_problem_l2186_218621


namespace piggy_bank_coins_l2186_218676

theorem piggy_bank_coins (sequence : Fin 6 → ℕ) 
  (h1 : sequence 0 = 72)
  (h2 : sequence 1 = 81)
  (h3 : sequence 2 = 90)
  (h5 : sequence 4 = 108)
  (h6 : sequence 5 = 117)
  (h_arithmetic : ∀ i : Fin 5, sequence (i + 1) - sequence i = sequence 1 - sequence 0) :
  sequence 3 = 99 := by
  sorry

end piggy_bank_coins_l2186_218676


namespace sum_of_series_equals_one_l2186_218637

/-- The sum of the infinite series ∑(n=1 to ∞) (4n-3)/(3^n) is equal to 1 -/
theorem sum_of_series_equals_one :
  (∑' n : ℕ, (4 * n - 3 : ℝ) / (3 : ℝ) ^ n) = 1 := by
sorry

end sum_of_series_equals_one_l2186_218637


namespace closest_integer_to_cube_root_l2186_218655

theorem closest_integer_to_cube_root (x : ℝ) : 
  x = (7^3 + 9^3 + 10^3 : ℝ)^(1/3) → 
  ∃ (n : ℤ), n = 13 ∧ ∀ (m : ℤ), |x - n| ≤ |x - m| :=
by sorry

end closest_integer_to_cube_root_l2186_218655


namespace reflection_and_shift_theorem_l2186_218641

-- Define the transformation properties
def is_reflection_and_shift (f h : ℝ → ℝ) : Prop :=
  ∀ x, h x = f (3 - x)

-- State the theorem
theorem reflection_and_shift_theorem (f h : ℝ → ℝ) 
  (h_def : is_reflection_and_shift f h) : 
  ∀ x, h x = f (3 - x) := by
  sorry

end reflection_and_shift_theorem_l2186_218641


namespace range_of_a_l2186_218617

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → a < -x^2 + 2*x) → 
  (∀ y : ℝ, y < 0 → ∃ x : ℝ, 0 ≤ x ∧ x ≤ 2 ∧ y < -x^2 + 2*x) ∧ 
  (∀ z : ℝ, z ≥ 0 → ∃ w : ℝ, 0 ≤ w ∧ w ≤ 2 ∧ z ≥ -w^2 + 2*w) :=
by sorry

end range_of_a_l2186_218617


namespace lee_initial_savings_l2186_218659

/-- Calculates Lee's initial savings before selling action figures -/
def initial_savings (sneaker_cost : ℕ) (action_figures_sold : ℕ) (price_per_figure : ℕ) (money_left : ℕ) : ℕ :=
  sneaker_cost + money_left - (action_figures_sold * price_per_figure)

theorem lee_initial_savings :
  initial_savings 90 10 10 25 = 15 := by
  sorry

end lee_initial_savings_l2186_218659


namespace triangle_line_equations_l2186_218673

/-- Triangle ABC with given properties -/
structure Triangle where
  A : ℝ × ℝ
  CM : ℝ → ℝ → Prop
  BH : ℝ → ℝ → Prop

/-- The given triangle satisfies the problem conditions -/
def given_triangle : Triangle where
  A := (5, 1)
  CM := fun x y ↦ 2 * x - y - 5 = 0
  BH := fun x y ↦ x - 2 * y - 5 = 0

/-- Line equation represented as ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Theorem stating the equations of line BC and its symmetric line -/
theorem triangle_line_equations (t : Triangle) 
  (h : t = given_triangle) : 
  ∃ (BC symmetric_BC : LineEquation),
    (BC.a = 6 ∧ BC.b = -5 ∧ BC.c = -9) ∧
    (symmetric_BC.a = 38 ∧ symmetric_BC.b = -9 ∧ symmetric_BC.c = -125) := by
  sorry

end triangle_line_equations_l2186_218673


namespace max_sum_given_constraints_l2186_218697

theorem max_sum_given_constraints (x y : ℝ) 
  (h1 : x^2 + y^2 = 130) 
  (h2 : x * y = 36) : 
  x + y ≤ Real.sqrt 202 ∧ ∃ (a b : ℝ), a^2 + b^2 = 130 ∧ a * b = 36 ∧ a + b = Real.sqrt 202 := by
  sorry

end max_sum_given_constraints_l2186_218697
