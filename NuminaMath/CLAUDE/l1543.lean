import Mathlib

namespace simplify_expression_l1543_154399

theorem simplify_expression (x : ℝ) : (3*x - 6)*(x + 9) - (x + 6)*(3*x - 2) = 5*x - 42 := by
  sorry

end simplify_expression_l1543_154399


namespace two_digit_cube_diff_reverse_l1543_154317

/-- A function that reverses a two-digit number -/
def reverse (n : ℕ) : ℕ :=
  10 * (n % 10) + (n / 10)

/-- A predicate that checks if a number is a positive perfect cube -/
def is_positive_perfect_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, k > 0 ∧ k^3 = n

/-- The main theorem -/
theorem two_digit_cube_diff_reverse :
  ∀ M : ℕ,
    10 ≤ M ∧ M < 100 ∧  -- M is a two-digit number
    (M % 10 ≠ 0) ∧      -- M's unit digit is non-zero
    is_positive_perfect_cube (M - reverse M) →
    M = 81 ∨ M = 92 :=
by sorry

end two_digit_cube_diff_reverse_l1543_154317


namespace square_side_length_l1543_154360

theorem square_side_length (s : ℝ) (h : s > 0) :
  s^2 = 6 * (4 * s) → s = 24 := by
  sorry

end square_side_length_l1543_154360


namespace count_multiples_of_seven_perfect_squares_l1543_154316

theorem count_multiples_of_seven_perfect_squares : 
  let lower_bound := 10^6
  let upper_bound := 10^9
  (Finset.range (Nat.floor (Real.sqrt (upper_bound / 49)) + 1) \ 
   Finset.range (Nat.floor (Real.sqrt (lower_bound / 49)))).card = 4376 := by
  sorry

end count_multiples_of_seven_perfect_squares_l1543_154316


namespace special_cylinder_lateral_area_l1543_154385

/-- A cylinder with base area S and lateral surface that unfolds into a square -/
structure SpecialCylinder where
  S : ℝ
  baseArea : S > 0
  lateralSurfaceIsSquare : True

/-- The lateral surface area of a SpecialCylinder is 4πS -/
theorem special_cylinder_lateral_area (c : SpecialCylinder) :
  ∃ (lateralArea : ℝ), lateralArea = 4 * Real.pi * c.S := by
  sorry

end special_cylinder_lateral_area_l1543_154385


namespace tank_filling_time_tank_filling_time_proof_l1543_154339

theorem tank_filling_time : ℝ → Prop :=
  fun T : ℝ =>
    let fill_rate_A : ℝ := 1 / 60
    let fill_rate_B : ℝ := 1 / 40
    let first_half : ℝ := T / 2 * fill_rate_B
    let second_half : ℝ := T / 2 * (fill_rate_A + fill_rate_B)
    (first_half + second_half = 1) → (T = 48)

-- The proof goes here
theorem tank_filling_time_proof : tank_filling_time 48 := by
  sorry

end tank_filling_time_tank_filling_time_proof_l1543_154339


namespace total_revenue_is_3610_l1543_154362

/-- Represents the quantity and price information for a fruit --/
structure Fruit where
  quantity : ℕ
  originalPrice : ℚ
  priceChange : ℚ

/-- Calculates the total revenue for a single fruit type --/
def calculateFruitRevenue (fruit : Fruit) : ℚ :=
  fruit.quantity * (fruit.originalPrice * (1 + fruit.priceChange))

/-- Theorem stating that the total revenue from all fruits is $3610 --/
theorem total_revenue_is_3610 
  (lemons : Fruit)
  (grapes : Fruit)
  (oranges : Fruit)
  (apples : Fruit)
  (kiwis : Fruit)
  (pineapples : Fruit)
  (h1 : lemons = { quantity := 80, originalPrice := 8, priceChange := 0.5 })
  (h2 : grapes = { quantity := 140, originalPrice := 7, priceChange := 0.25 })
  (h3 : oranges = { quantity := 60, originalPrice := 5, priceChange := 0.1 })
  (h4 : apples = { quantity := 100, originalPrice := 4, priceChange := 0.2 })
  (h5 : kiwis = { quantity := 50, originalPrice := 6, priceChange := -0.15 })
  (h6 : pineapples = { quantity := 30, originalPrice := 12, priceChange := 0 }) :
  calculateFruitRevenue lemons + calculateFruitRevenue grapes + 
  calculateFruitRevenue oranges + calculateFruitRevenue apples + 
  calculateFruitRevenue kiwis + calculateFruitRevenue pineapples = 3610 := by
  sorry

end total_revenue_is_3610_l1543_154362


namespace ceiling_distance_l1543_154329

/-- A point in a right-angled corner formed by two walls and a ceiling -/
structure CornerPoint where
  x : ℝ  -- distance from one wall
  y : ℝ  -- distance from the other wall
  z : ℝ  -- distance from the ceiling
  corner_distance : ℝ  -- distance from the corner point

/-- The theorem stating the distance from the ceiling for a specific point -/
theorem ceiling_distance (p : CornerPoint) 
  (h1 : p.x = 3)  -- 3 meters from one wall
  (h2 : p.y = 7)  -- 7 meters from the other wall
  (h3 : p.corner_distance = 10)  -- 10 meters from the corner point
  : p.z = Real.sqrt 42 := by
  sorry

end ceiling_distance_l1543_154329


namespace cyclic_sum_inequality_l1543_154392

theorem cyclic_sum_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ((b + c) * (a^4 - b^2 * c^2)) / (a*b + 2*b*c + c*a) +
  ((c + a) * (b^4 - c^2 * a^2)) / (b*c + 2*c*a + a*b) +
  ((a + b) * (c^4 - a^2 * b^2)) / (c*a + 2*a*b + b*c) ≥ 0 := by
  sorry

end cyclic_sum_inequality_l1543_154392


namespace polynomial_equality_l1543_154395

theorem polynomial_equality (x : ℝ) : ∃ (t a b : ℝ),
  (3 * x^2 - 4 * x + 5) * (5 * x^2 + t * x + 12) = 15 * x^4 - 47 * x^3 + a * x^2 + b * x + 60 ∧
  t = -9 ∧ a = -53 ∧ b = -156 := by
  sorry

end polynomial_equality_l1543_154395


namespace gamma_donuts_l1543_154349

/-- Proves that Gamma received 8 donuts given the conditions of the problem -/
theorem gamma_donuts : 
  ∀ (gamma_donuts : ℕ),
  (40 : ℕ) = 8 + 3 * gamma_donuts + gamma_donuts →
  gamma_donuts = 8 := by
sorry

end gamma_donuts_l1543_154349


namespace function_property_l1543_154314

theorem function_property (f : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, f (x + 19) ≤ f x + 19) 
  (h2 : ∀ x : ℝ, f (x + 94) ≥ f x + 94) : 
  ∀ x : ℝ, f (x + 1) = f x + 1 := by
  sorry

end function_property_l1543_154314


namespace divisible_by_two_l1543_154390

theorem divisible_by_two (a b : ℕ) : 
  (2 ∣ (a * b)) → (2 ∣ a) ∨ (2 ∣ b) := by
  sorry

end divisible_by_two_l1543_154390


namespace gcd_459_357_l1543_154374

theorem gcd_459_357 : Nat.gcd 459 357 = 51 := by
  sorry

end gcd_459_357_l1543_154374


namespace factorization_equality_l1543_154394

theorem factorization_equality (m n : ℝ) : 2 * m^2 * n - 8 * m * n + 8 * n = 2 * n * (m - 2)^2 := by
  sorry

end factorization_equality_l1543_154394


namespace solve_for_y_l1543_154364

theorem solve_for_y (x y : ℤ) (h1 : x^2 - x + 6 = y + 2) (h2 : x = -8) : y = 76 := by
  sorry

end solve_for_y_l1543_154364


namespace num_children_picked_apples_l1543_154378

/-- The number of baskets -/
def num_baskets : ℕ := 11

/-- The sum of apples picked by each child from all baskets -/
def apples_per_child : ℕ := (num_baskets * (num_baskets + 1)) / 2

/-- The total number of apples picked by all children -/
def total_apples_picked : ℕ := 660

/-- Theorem stating that the number of children who picked apples is 10 -/
theorem num_children_picked_apples : 
  total_apples_picked / apples_per_child = 10 := by
  sorry

end num_children_picked_apples_l1543_154378


namespace rectangle_square_ratio_l1543_154346

/-- Represents the configuration of rectangles around a square -/
structure RectangleSquareConfig where
  s : ℝ  -- Side length of the inner square
  x : ℝ  -- Shorter side of the rectangle
  y : ℝ  -- Longer side of the rectangle

/-- Theorem: If four congruent rectangles are placed around a central square such that 
    the area of the outer square is 9 times the area of the inner square, 
    then the ratio of the longer side to the shorter side of each rectangle is 2. -/
theorem rectangle_square_ratio 
  (config : RectangleSquareConfig) 
  (h1 : config.s > 0)  -- Inner square has positive side length
  (h2 : config.x > 0)  -- Rectangle has positive width
  (h3 : config.y > 0)  -- Rectangle has positive height
  (h4 : config.s + 2 * config.x = 3 * config.s)  -- Outer square side length relation
  (h5 : config.y + config.x = 3 * config.s)  -- Outer square side length relation (alternative)
  : config.y / config.x = 2 := by
  sorry

end rectangle_square_ratio_l1543_154346


namespace park_visitors_total_l1543_154363

theorem park_visitors_total (saturday_visitors : ℕ) (sunday_extra : ℕ) : 
  saturday_visitors = 200 → sunday_extra = 40 → 
  saturday_visitors + (saturday_visitors + sunday_extra) = 440 := by
sorry

end park_visitors_total_l1543_154363


namespace spanning_rectangles_odd_l1543_154351

/-- Represents a 2 × 1 rectangle used to cover the cube surface -/
structure Rectangle :=
  (spans_two_faces : Bool)

/-- Represents the surface of a 9 × 9 × 9 cube -/
structure CubeSurface :=
  (side_length : Nat)
  (covering : List Rectangle)

/-- Axiom: The cube is 9 × 9 × 9 -/
axiom cube_size : ∀ (c : CubeSurface), c.side_length = 9

/-- Axiom: The surface is completely covered without gaps or overlaps -/
axiom complete_coverage : ∀ (c : CubeSurface), c.covering.length * 2 = 6 * c.side_length^2

/-- Main theorem: The number of rectangles spanning two faces is odd -/
theorem spanning_rectangles_odd (c : CubeSurface) : 
  Odd (c.covering.filter Rectangle.spans_two_faces).length :=
sorry

end spanning_rectangles_odd_l1543_154351


namespace remainder_845307_div_6_l1543_154344

theorem remainder_845307_div_6 : Nat.mod 845307 6 = 3 := by
  sorry

end remainder_845307_div_6_l1543_154344


namespace chords_intersection_concyclic_l1543_154366

-- Define the ellipse
def ellipse (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define a point on the ellipse
structure PointOnEllipse (a b : ℝ) where
  x : ℝ
  y : ℝ
  on_ellipse : ellipse a b x y

-- Define the theorem
theorem chords_intersection_concyclic 
  (a b : ℝ) 
  (A B C D S : PointOnEllipse a b) 
  (h1 : S.x ≠ A.x ∨ S.y ≠ A.y) 
  (h2 : S.x ≠ B.x ∨ S.y ≠ B.y)
  (h3 : S.x ≠ C.x ∨ S.y ≠ C.y)
  (h4 : S.x ≠ D.x ∨ S.y ≠ D.y)
  (h5 : (A.y - S.y) * (C.x - S.x) = (A.x - S.x) * (C.y - S.y)) -- AB and CD intersect at S
  (h6 : (B.y - S.y) * (D.x - S.x) = (B.x - S.x) * (D.y - S.y)) -- AB and CD intersect at S
  (h7 : (A.y - S.y) * (C.x - S.x) = (C.y - S.y) * (D.x - S.x)) -- ∠ASC = ∠BSD
  : ∃ (center : ℝ × ℝ) (radius : ℝ),
    (A.x - center.1)^2 + (A.y - center.2)^2 = radius^2 ∧
    (B.x - center.1)^2 + (B.y - center.2)^2 = radius^2 ∧
    (C.x - center.1)^2 + (C.y - center.2)^2 = radius^2 ∧
    (D.x - center.1)^2 + (D.y - center.2)^2 = radius^2 := by
  sorry

end chords_intersection_concyclic_l1543_154366


namespace congruence_system_solutions_l1543_154381

theorem congruence_system_solutions (a b c : ℤ) : 
  ∃ (s : Finset ℤ), 
    (∀ x ∈ s, x ≥ 0 ∧ x < 2000 ∧ 
      x % 14 = a % 14 ∧ 
      x % 15 = b % 15 ∧ 
      x % 16 = c % 16) ∧
    s.card = 3 :=
by sorry

end congruence_system_solutions_l1543_154381


namespace no_good_filling_for_1399_l1543_154301

theorem no_good_filling_for_1399 :
  ¬ ∃ (f : Fin 1399 → Fin 2798), 
    (∀ i : Fin 1399, f i ≠ f (i + 1)) ∧ 
    (∀ i j : Fin 1399, i ≠ j → f i ≠ f j) ∧
    (∀ i : Fin 1399, (f i.succ - f i) % 2798 = i.val + 1) :=
by
  sorry

#check no_good_filling_for_1399

end no_good_filling_for_1399_l1543_154301


namespace m_range_l1543_154322

-- Define propositions p and q
def p (m : ℝ) : Prop := ∀ x, x^2 - 2*m*x + 7*m - 10 ≠ 0

def q (m : ℝ) : Prop := ∀ x > 0, x^2 - m*x + 4 ≥ 0

-- State the theorem
theorem m_range (m : ℝ) 
  (h1 : p m ∨ q m) 
  (h2 : p m ∧ q m) : 
  m ∈ Set.Ioo 2 4 ∪ {4} :=
sorry

end m_range_l1543_154322


namespace steve_total_cost_theorem_l1543_154379

def steve_total_cost (mike_dvd_price : ℝ) (steve_extra_dvd_price : ℝ) 
  (steve_extra_dvd_count : ℕ) (shipping_rate : ℝ) (tax_rate : ℝ) : ℝ :=
  let steve_favorite_dvd_price := 2 * mike_dvd_price
  let steve_extra_dvds_cost := steve_extra_dvd_count * steve_extra_dvd_price
  let total_dvds_cost := steve_favorite_dvd_price + steve_extra_dvds_cost
  let shipping_cost := shipping_rate * total_dvds_cost
  let subtotal := total_dvds_cost + shipping_cost
  let tax := tax_rate * subtotal
  subtotal + tax

theorem steve_total_cost_theorem :
  steve_total_cost 5 7 2 0.8 0.1 = 47.52 := by
  sorry

end steve_total_cost_theorem_l1543_154379


namespace line_tangent_to_circle_l1543_154340

theorem line_tangent_to_circle (t θ : ℝ) (α : ℝ) : 
  (∃ t, ∀ θ, (t * Real.cos α - (4 + 2 * Real.cos θ))^2 + (t * Real.sin α - 2 * Real.sin θ)^2 = 4) →
  α = π / 6 ∨ α = 5 * π / 6 := by
  sorry

end line_tangent_to_circle_l1543_154340


namespace chess_match_results_l1543_154330

/-- Chess match between players A and B -/
structure ChessMatch where
  prob_draw : ℝ
  prob_a_win : ℝ
  prob_b_win : ℝ

/-- Conditions for the chess match -/
def match_conditions : ChessMatch where
  prob_draw := 0.5
  prob_a_win := 0.3
  prob_b_win := 0.2

/-- Expected number of games in the match -/
def expected_games (m : ChessMatch) : ℝ := sorry

/-- Probability that player B wins the match -/
def prob_b_wins (m : ChessMatch) : ℝ := sorry

/-- Theorem stating the expected number of games and probability of B winning -/
theorem chess_match_results (m : ChessMatch) 
  (h1 : m = match_conditions) : 
  expected_games m = 3.175 ∧ prob_b_wins m = 0.315 := by sorry

end chess_match_results_l1543_154330


namespace lisa_spoon_count_l1543_154309

/-- The number of spoons Lisa has after replacing her old cutlery -/
def total_spoons (num_children : ℕ) (spoons_per_child : ℕ) (decorative_spoons : ℕ) 
  (large_spoons : ℕ) (teaspoons : ℕ) : ℕ :=
  num_children * spoons_per_child + decorative_spoons + large_spoons + teaspoons

/-- Proof that Lisa has 39 spoons in total -/
theorem lisa_spoon_count : 
  total_spoons 4 3 2 10 15 = 39 := by
  sorry

end lisa_spoon_count_l1543_154309


namespace or_equivalence_l1543_154352

-- Define the propositions
variable (p : Prop)  -- Athlete A's trial jump exceeded 2 meters
variable (q : Prop)  -- Athlete B's trial jump exceeded 2 meters

-- Define the statement "At least one of Athlete A or B exceeded 2 meters in their trial jump"
def atLeastOneExceeded (p q : Prop) : Prop :=
  p ∨ q

-- Theorem stating the equivalence
theorem or_equivalence :
  (p ∨ q) ↔ atLeastOneExceeded p q :=
sorry

end or_equivalence_l1543_154352


namespace parabola_point_relationship_l1543_154384

/-- A quadratic function with a symmetry axis at x = -1 -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  symmetry_axis : a ≠ 0 ∧ -|b| / (2 * a) = -1

/-- Three points on the parabola -/
structure ParabolaPoints where
  y₁ : ℝ
  y₂ : ℝ
  y₃ : ℝ
  on_parabola : ∀ (f : QuadraticFunction),
    f.a * (-14/3)^2 + |f.b| * (-14/3) + f.c = y₁ ∧
    f.a * (5/2)^2 + |f.b| * (5/2) + f.c = y₂ ∧
    f.a * 3^2 + |f.b| * 3 + f.c = y₃

/-- Theorem stating the relationship between y₁, y₂, and y₃ -/
theorem parabola_point_relationship (f : QuadraticFunction) (p : ParabolaPoints) :
  p.y₂ < p.y₁ ∧ p.y₁ < p.y₃ := by
  sorry

end parabola_point_relationship_l1543_154384


namespace four_dice_probability_l1543_154353

/-- The probability of a single standard six-sided die showing a specific number -/
def single_die_prob : ℚ := 1 / 6

/-- The number of dice being tossed simultaneously -/
def num_dice : ℕ := 4

/-- The probability of all dice showing the same specific number -/
def all_dice_prob : ℚ := (single_die_prob ^ num_dice)

/-- Theorem stating that the probability of all four standard six-sided dice 
    showing the number 3 when tossed simultaneously is 1/1296 -/
theorem four_dice_probability : all_dice_prob = 1 / 1296 := by
  sorry

end four_dice_probability_l1543_154353


namespace f_range_theorem_l1543_154333

-- Define the function f(x)
def f (x : ℝ) : ℝ := (x^2 - 1) * (x^2 - 12*x + 35)

-- State the theorem
theorem f_range_theorem :
  (∀ x : ℝ, f (6 - x) = f x) →  -- Symmetry condition
  (∃ y : ℝ, ∀ x : ℝ, f x ≥ y) ∧ -- Lower bound exists
  (∀ y : ℝ, y ≥ -36 → ∃ x : ℝ, f x = y) -- All values ≥ -36 are in the range
  :=
by sorry

end f_range_theorem_l1543_154333


namespace rectangle_sections_3x5_l1543_154304

/-- The number of rectangular sections (including squares) in a grid --/
def rectangleCount (width height : ℕ) : ℕ :=
  let squareCount := (width * (width + 1) * height * (height + 1)) / 4
  let rectangleCount := (width * (width + 1) * height * (height + 1)) / 4 - (width * height)
  squareCount + rectangleCount

/-- Theorem stating that the number of rectangular sections in a 3x5 grid is 72 --/
theorem rectangle_sections_3x5 :
  rectangleCount 3 5 = 72 := by
  sorry

end rectangle_sections_3x5_l1543_154304


namespace sheela_deposit_l1543_154347

/-- Sheela's monthly income in Rupees -/
def monthly_income : ℕ := 25000

/-- The percentage of monthly income deposited -/
def deposit_percentage : ℚ := 20 / 100

/-- Calculate the deposit amount based on monthly income and deposit percentage -/
def deposit_amount (income : ℕ) (percentage : ℚ) : ℚ :=
  percentage * income

/-- Theorem stating that Sheela's deposit amount is 5000 Rupees -/
theorem sheela_deposit :
  deposit_amount monthly_income deposit_percentage = 5000 := by
  sorry

end sheela_deposit_l1543_154347


namespace intersection_of_M_and_N_l1543_154375

def M : Set ℝ := {-1, 0, 1, 2}
def N : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 2}

theorem intersection_of_M_and_N : M ∩ N = {-1, 0, 1} := by sorry

end intersection_of_M_and_N_l1543_154375


namespace quiz_probability_l1543_154305

/-- The probability of answering a multiple-choice question with 5 options correctly -/
def prob_multiple_choice : ℚ := 1 / 5

/-- The probability of answering a true/false question correctly -/
def prob_true_false : ℚ := 1 / 2

/-- The number of true/false questions in the quiz -/
def num_true_false : ℕ := 4

/-- The probability of answering all questions in the quiz correctly -/
def prob_all_correct : ℚ := prob_multiple_choice * prob_true_false ^ num_true_false

theorem quiz_probability :
  prob_all_correct = 1 / 80 := by sorry

end quiz_probability_l1543_154305


namespace expression_equals_zero_l1543_154337

theorem expression_equals_zero (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x = y + 1) :
  (x + 1/x) * (y - 1/y) = 0 := by
  sorry

end expression_equals_zero_l1543_154337


namespace inequality_chain_l1543_154354

theorem inequality_chain (a b : ℝ) (ha : a < 0) (hb : -1 < b ∧ b < 0) :
  a * b > a * b^2 ∧ a * b^2 > a := by sorry

end inequality_chain_l1543_154354


namespace points_not_on_any_circle_l1543_154369

-- Define the circle equation
def circle_equation (x y α β : ℝ) : Prop :=
  α * ((x - 2)^2 + y^2 - 1) + β * ((x + 2)^2 + y^2 - 1) = 0

-- Define the set of points not on any circle
def points_not_on_circles : Set (ℝ × ℝ) :=
  {p | p.1 = 0 ∨ (p.1 = Real.sqrt 3 ∧ p.2 = 0) ∨ (p.1 = -Real.sqrt 3 ∧ p.2 = 0)}

-- Theorem statement
theorem points_not_on_any_circle :
  ∀ (p : ℝ × ℝ), p ∈ points_not_on_circles →
  ∀ (α β : ℝ), ¬(circle_equation p.1 p.2 α β) :=
by sorry

end points_not_on_any_circle_l1543_154369


namespace smallest_number_l1543_154300

theorem smallest_number (a b c : ℝ) : 
  c = 2 * a →
  b = 4 * a →
  (a + b + c) / 3 = 77 →
  a = 33 ∧ a ≤ b ∧ a ≤ c :=
by sorry

end smallest_number_l1543_154300


namespace parallel_implies_a_values_l_passes_through_point_l1543_154313

-- Define the lines l and n
def l (a x y : ℝ) : Prop := (a + 2) * x + a * y - 2 = 0
def n (a x y : ℝ) : Prop := (a - 2) * x + 3 * y - 6 = 0

-- Define parallel lines
def parallel (f g : ℝ → ℝ → ℝ → Prop) : Prop :=
  ∀ a, (∃ k ≠ 0, ∀ x y, f a x y ↔ g a (k * x) (k * y))

-- Theorem 1: If l is parallel to n, then a = 6 or a = -1
theorem parallel_implies_a_values :
  parallel l n → ∀ a, (a = 6 ∨ a = -1) :=
sorry

-- Theorem 2: Line l always passes through the point (1, -1)
theorem l_passes_through_point :
  ∀ a, l a 1 (-1) :=
sorry

end parallel_implies_a_values_l_passes_through_point_l1543_154313


namespace orange_apple_cost_l1543_154396

theorem orange_apple_cost : ∃ (x y : ℚ),
  (7 * x + 5 * y = 13) ∧
  (3 * x + 4 * y = 8) →
  (37 * x + 45 * y = 93) := by
  sorry

end orange_apple_cost_l1543_154396


namespace angle_relation_in_triangle_l1543_154367

theorem angle_relation_in_triangle (A B C : ℝ) (h_triangle : A + B + C = π) 
  (h_positive : 0 < A ∧ 0 < B ∧ 0 < C) (h_sin : Real.sin A > Real.sin B) : A > B := by
  sorry

end angle_relation_in_triangle_l1543_154367


namespace pokemon_cards_total_l1543_154380

/-- The number of people with Pokemon cards -/
def num_people : ℕ := 4

/-- The number of Pokemon cards each person has -/
def cards_per_person : ℕ := 14

/-- The total number of Pokemon cards -/
def total_cards : ℕ := num_people * cards_per_person

theorem pokemon_cards_total : total_cards = 56 := by
  sorry

end pokemon_cards_total_l1543_154380


namespace cost_per_square_meter_l1543_154383

def initial_land : ℝ := 300
def final_land : ℝ := 900
def total_cost : ℝ := 12000

theorem cost_per_square_meter :
  (total_cost / (final_land - initial_land)) = 20 := by sorry

end cost_per_square_meter_l1543_154383


namespace budget_calculation_l1543_154342

/-- The original budget in Euros -/
def original_budget : ℝ := sorry

/-- The amount left after spending -/
def amount_left : ℝ := 13500

/-- The fraction of budget spent on clothes -/
def clothes_fraction : ℝ := 0.25

/-- The discount on clothes -/
def clothes_discount : ℝ := 0.1

/-- The fraction of budget spent on groceries -/
def groceries_fraction : ℝ := 0.15

/-- The sales tax on groceries -/
def groceries_tax : ℝ := 0.05

/-- The fraction of budget spent on electronics -/
def electronics_fraction : ℝ := 0.1

/-- The exchange rate for electronics (EUR to USD) -/
def exchange_rate : ℝ := 1.2

/-- The fraction of budget spent on dining -/
def dining_fraction : ℝ := 0.05

/-- The service charge on dining -/
def dining_service_charge : ℝ := 0.12

theorem budget_calculation :
  amount_left = original_budget * (1 - (
    clothes_fraction * (1 - clothes_discount) +
    groceries_fraction * (1 + groceries_tax) +
    electronics_fraction * exchange_rate +
    dining_fraction * (1 + dining_service_charge)
  )) := by sorry

end budget_calculation_l1543_154342


namespace rectangle_length_fraction_of_circle_radius_l1543_154373

theorem rectangle_length_fraction_of_circle_radius 
  (square_area : ℝ) 
  (rectangle_area : ℝ) 
  (rectangle_breadth : ℝ) 
  (h1 : square_area = 1225) 
  (h2 : rectangle_area = 140) 
  (h3 : rectangle_breadth = 10) : 
  (rectangle_area / rectangle_breadth) / Real.sqrt square_area = 2 / 5 := by
  sorry

end rectangle_length_fraction_of_circle_radius_l1543_154373


namespace g_16_48_l1543_154306

/-- A function on ordered pairs of positive integers satisfying specific properties -/
def g : ℕ+ → ℕ+ → ℕ+ :=
  sorry

/-- The first property: g(x,x) = 2x -/
axiom g_diag (x : ℕ+) : g x x = 2 * x

/-- The second property: g(x,y) = g(y,x) -/
axiom g_comm (x y : ℕ+) : g x y = g y x

/-- The third property: (x + y) g(x,y) = x g(x, x + y) -/
axiom g_prop (x y : ℕ+) : (x + y) * g x y = x * g x (x + y)

/-- The main theorem: g(16, 48) = 96 -/
theorem g_16_48 : g 16 48 = 96 :=
  sorry

end g_16_48_l1543_154306


namespace unique_number_power_ten_sum_l1543_154371

theorem unique_number_power_ten_sum : ∃! (N : ℕ), 
  N > 0 ∧ 
  (∃ (k : ℕ), N + (Nat.factors N).foldl Nat.lcm 1 = 10^k) ∧
  N = 75 := by
  sorry

end unique_number_power_ten_sum_l1543_154371


namespace chess_tournament_games_l1543_154341

/-- Calculate the number of games in a chess tournament --/
def tournament_games (n : ℕ) : ℕ := n * (n - 1)

/-- The number of players in the tournament --/
def num_players : ℕ := 7

/-- Theorem: In a chess tournament with 7 players, where each player plays twice
    with every other player, the total number of games played is 84. --/
theorem chess_tournament_games :
  2 * tournament_games num_players = 84 := by
  sorry


end chess_tournament_games_l1543_154341


namespace greatest_integer_with_gcf_five_l1543_154303

theorem greatest_integer_with_gcf_five : ∃ n : ℕ, n < 200 ∧ Nat.gcd n 30 = 5 ∧ ∀ m : ℕ, m < 200 → Nat.gcd m 30 = 5 → m ≤ n := by
  sorry

end greatest_integer_with_gcf_five_l1543_154303


namespace circle_assignment_exists_l1543_154356

structure Circle where
  value : ℕ

structure Graph where
  A : Circle
  B : Circle
  C : Circle
  D : Circle

def connected (x y : Circle) (g : Graph) : Prop :=
  (x = g.A ∧ y = g.B) ∨ (x = g.B ∧ y = g.A) ∨
  (x = g.A ∧ y = g.D) ∨ (x = g.D ∧ y = g.A) ∨
  (x = g.B ∧ y = g.C) ∨ (x = g.C ∧ y = g.B)

def ratio (x y : Circle) : ℚ :=
  (x.value : ℚ) / (y.value : ℚ)

theorem circle_assignment_exists : ∃ g : Graph,
  (∀ x y : Circle, connected x y g → (ratio x y = 3 ∨ ratio x y = 9)) ∧
  (∀ x y : Circle, ¬connected x y g → (ratio x y ≠ 3 ∧ ratio x y ≠ 9)) :=
sorry

end circle_assignment_exists_l1543_154356


namespace parabola_equation_theorem_l1543_154312

/-- A parabola with vertex at the origin, x-axis as the axis of symmetry, 
    and passing through the point (4, -2) -/
structure Parabola where
  -- The parabola passes through (4, -2)
  passes_through : (4 : ℝ)^2 + (-2 : ℝ)^2 ≠ 0

/-- The equation of the parabola is either y^2 = x or x^2 = -8y -/
def parabola_equation (p : Parabola) : Prop :=
  (∀ x y : ℝ, y^2 = x) ∨ (∀ x y : ℝ, x^2 = -8*y)

/-- Theorem stating that the parabola satisfies one of the two equations -/
theorem parabola_equation_theorem (p : Parabola) : parabola_equation p := by
  sorry

end parabola_equation_theorem_l1543_154312


namespace ratio_limit_is_one_l1543_154335

/-- The ratio of the largest element (2^20) to the sum of other elements in the set {1, 2, 2^2, ..., 2^20} -/
def ratio (n : ℕ) : ℚ :=
  2^n / (2^n - 1)

/-- The limit of the ratio as n approaches infinity is 1 -/
theorem ratio_limit_is_one : 
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |ratio n - 1| < ε :=
sorry

end ratio_limit_is_one_l1543_154335


namespace intersection_perpendicular_tangents_l1543_154386

open Real

theorem intersection_perpendicular_tangents (a : ℝ) :
  ∃ x ∈ Set.Ioo 0 (π / 2),
    2 * sin x = a * cos x ∧
    (2 * cos x) * (-a * sin x) = -1 →
  a = 2 * sqrt 3 / 3 := by
sorry

end intersection_perpendicular_tangents_l1543_154386


namespace collinear_points_k_value_l1543_154348

/-- Three points are collinear if they lie on the same straight line -/
def collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  (p2.2 - p1.2) * (p3.1 - p2.1) = (p3.2 - p2.2) * (p2.1 - p1.1)

/-- The theorem stating that if (-2, -4), (5, k), and (15, 1) are collinear, then k = -33/17 -/
theorem collinear_points_k_value :
  ∀ k : ℝ, collinear (-2, -4) (5, k) (15, 1) → k = -33/17 := by
  sorry

end collinear_points_k_value_l1543_154348


namespace suma_work_time_l1543_154331

/-- Proves the time taken by Suma to complete the work alone -/
theorem suma_work_time (renu_time suma_renu_time : ℝ) 
  (h1 : renu_time = 8)
  (h2 : suma_renu_time = 3)
  (h3 : renu_time > 0)
  (h4 : suma_renu_time > 0) :
  ∃ (suma_time : ℝ), 
    suma_time > 0 ∧ 
    1 / renu_time + 1 / suma_time = 1 / suma_renu_time ∧ 
    suma_time = 24 / 5 := by
  sorry

end suma_work_time_l1543_154331


namespace rectangles_on_clock_face_l1543_154310

/-- The number of equally spaced points on a circle -/
def n : ℕ := 12

/-- A function that calculates the number of rectangles that can be formed
    by selecting 4 vertices from n equally spaced points on a circle -/
def count_rectangles (n : ℕ) : ℕ := sorry

/-- Theorem stating that the number of rectangles formed is 15 when n = 12 -/
theorem rectangles_on_clock_face : count_rectangles n = 15 := by sorry

end rectangles_on_clock_face_l1543_154310


namespace cyclic_sum_inequality_l1543_154307

theorem cyclic_sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a * (a^2 + b*c))/(b + c) + (b * (b^2 + a*c))/(a + c) + (c * (c^2 + a*b))/(a + b) ≥
  a*b + b*c + c*a := by
  sorry

end cyclic_sum_inequality_l1543_154307


namespace power_two_mod_nine_l1543_154334

theorem power_two_mod_nine : 2 ^ 46655 % 9 = 1 := by
  sorry

end power_two_mod_nine_l1543_154334


namespace sum_of_coefficients_l1543_154325

/-- Given the equation (1+x)+(1+x)^2+...+(1+x)^5 = a₀+a₁(1-x)+a₂(1-x)^2+...+a₅(1-x)^5,
    prove that a₁+a₂+a₃+a₄+a₅ = -57 -/
theorem sum_of_coefficients (x : ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) : 
  ((1+x) + (1+x)^2 + (1+x)^3 + (1+x)^4 + (1+x)^5 = 
   a₀ + a₁*(1-x) + a₂*(1-x)^2 + a₃*(1-x)^3 + a₄*(1-x)^4 + a₅*(1-x)^5) → 
  (a₁ + a₂ + a₃ + a₄ + a₅ = -57) := by
  sorry

end sum_of_coefficients_l1543_154325


namespace imaginary_part_of_z_l1543_154393

theorem imaginary_part_of_z (z : ℂ) (h : (1 - Complex.I) * z = 2 * Complex.I) :
  z.im = 1 := by
  sorry

end imaginary_part_of_z_l1543_154393


namespace work_increase_percentage_l1543_154302

/-- Proves that when 1/7 of the members are absent in an office, 
    the percentage increase in work for each remaining person is 100/6. -/
theorem work_increase_percentage (p : ℝ) (p_pos : p > 0) : 
  let absent_fraction : ℝ := 1/7
  let remaining_fraction : ℝ := 1 - absent_fraction
  let work_increase_ratio : ℝ := 1 / remaining_fraction
  let percentage_increase : ℝ := (work_increase_ratio - 1) * 100
  percentage_increase = 100/6 := by
sorry

#eval (100 : ℚ) / 6  -- To show the approximate decimal value

end work_increase_percentage_l1543_154302


namespace lindas_savings_l1543_154326

theorem lindas_savings (savings : ℝ) : 
  (3 / 5 : ℝ) * savings + 400 = savings → savings = 1000 := by
sorry

end lindas_savings_l1543_154326


namespace sam_drew_age_problem_l1543_154336

/-- The combined age of Sam and Drew given Sam's age and the relation between their ages -/
def combinedAge (samAge : ℕ) (drewAge : ℕ) : ℕ := samAge + drewAge

theorem sam_drew_age_problem :
  let samAge : ℕ := 18
  let drewAge : ℕ := 2 * samAge
  combinedAge samAge drewAge = 54 := by
  sorry

end sam_drew_age_problem_l1543_154336


namespace maintenance_check_increase_l1543_154320

theorem maintenance_check_increase (original_time : ℝ) (percentage_increase : ℝ) 
  (h1 : original_time = 45)
  (h2 : percentage_increase = 33.33333333333333) : 
  original_time * (1 + percentage_increase / 100) = 60 := by
sorry

end maintenance_check_increase_l1543_154320


namespace tangent_slope_at_1_l1543_154370

/-- The function f(x) = (x-2)(x^2+c) has an extremum at x=2 -/
def has_extremum_at_2 (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∃ ε > 0, ∀ x ∈ Set.Ioo (2 - ε) (2 + ε), f x ≤ f 2 ∨ f x ≥ f 2

/-- The main theorem -/
theorem tangent_slope_at_1 (c : ℝ) :
  let f : ℝ → ℝ := fun x ↦ (x - 2) * (x^2 + c)
  has_extremum_at_2 f c → (deriv f) 1 = -5 := by
  sorry

end tangent_slope_at_1_l1543_154370


namespace order_of_abc_l1543_154387

-- Define the constants
noncomputable def a : ℝ := Real.log 2 / Real.log 10
noncomputable def b : ℝ := Real.cos 2
noncomputable def c : ℝ := 2 ^ (1 / 5)

-- State the theorem
theorem order_of_abc : b < a ∧ a < c := by sorry

end order_of_abc_l1543_154387


namespace checkerboard_corner_sum_l1543_154345

/-- The size of the checkerboard -/
def boardSize : Nat := 9

/-- The total number of squares on the board -/
def totalSquares : Nat := boardSize * boardSize

/-- The number in the top-left corner -/
def topLeft : Nat := 1

/-- The number in the top-right corner -/
def topRight : Nat := boardSize

/-- The number in the bottom-left corner -/
def bottomLeft : Nat := totalSquares - boardSize + 1

/-- The number in the bottom-right corner -/
def bottomRight : Nat := totalSquares

/-- The sum of the numbers in the four corners of the checkerboard -/
def cornerSum : Nat := topLeft + topRight + bottomLeft + bottomRight

theorem checkerboard_corner_sum : cornerSum = 164 := by
  sorry

end checkerboard_corner_sum_l1543_154345


namespace total_pieces_four_row_triangle_l1543_154398

/-- Calculates the sum of the first n multiples of 3 -/
def sum_multiples_of_three (n : ℕ) : ℕ := 
  3 * n * (n + 1) / 2

/-- Calculates the sum of the first n even numbers -/
def sum_even_numbers (n : ℕ) : ℕ := 
  n * (n + 1)

/-- Represents the number of rows in the triangle configuration -/
def num_rows : ℕ := 4

/-- Theorem: The total number of pieces in a four-row triangle configuration is 60 -/
theorem total_pieces_four_row_triangle : 
  sum_multiples_of_three num_rows + sum_even_numbers (num_rows + 1) = 60 := by
  sorry

#eval sum_multiples_of_three num_rows + sum_even_numbers (num_rows + 1)

end total_pieces_four_row_triangle_l1543_154398


namespace bag_cost_is_eight_l1543_154321

/-- Represents the coffee consumption and cost for Maddie's mom --/
structure CoffeeConsumption where
  cups_per_day : ℕ
  ounces_per_cup : ℚ
  ounces_per_bag : ℚ
  milk_gallons_per_week : ℚ
  milk_cost_per_gallon : ℚ
  total_weekly_cost : ℚ

/-- Calculates the cost of a bag of coffee based on the given consumption data --/
def bag_cost (c : CoffeeConsumption) : ℚ :=
  let ounces_per_week := c.cups_per_day * c.ounces_per_cup * 7
  let bags_per_week := ounces_per_week / c.ounces_per_bag
  let milk_cost_per_week := c.milk_gallons_per_week * c.milk_cost_per_gallon
  let coffee_cost_per_week := c.total_weekly_cost - milk_cost_per_week
  coffee_cost_per_week / bags_per_week

/-- Theorem stating that the cost of a bag of coffee is $8 --/
theorem bag_cost_is_eight (c : CoffeeConsumption) 
  (h1 : c.cups_per_day = 2)
  (h2 : c.ounces_per_cup = 3/2)
  (h3 : c.ounces_per_bag = 21/2)
  (h4 : c.milk_gallons_per_week = 1/2)
  (h5 : c.milk_cost_per_gallon = 4)
  (h6 : c.total_weekly_cost = 18) :
  bag_cost c = 8 := by
  sorry

#eval bag_cost {
  cups_per_day := 2,
  ounces_per_cup := 3/2,
  ounces_per_bag := 21/2,
  milk_gallons_per_week := 1/2,
  milk_cost_per_gallon := 4,
  total_weekly_cost := 18
}

end bag_cost_is_eight_l1543_154321


namespace negation_equivalence_l1543_154308

theorem negation_equivalence :
  (¬ ∃ x : ℝ, Real.exp x < x) ↔ (∀ x : ℝ, Real.exp x ≥ x) := by
  sorry

end negation_equivalence_l1543_154308


namespace subset_condition_nonempty_intersection_l1543_154388

-- Define the sets A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}
def B (a : ℝ) : Set ℝ := {x | x > a}

-- Theorem 1: A ⊆ B iff a < -1
theorem subset_condition (a : ℝ) : A ⊆ B a ↔ a < -1 := by sorry

-- Theorem 2: A ∩ B ≠ ∅ iff a < 3
theorem nonempty_intersection (a : ℝ) : (A ∩ B a).Nonempty ↔ a < 3 := by sorry

end subset_condition_nonempty_intersection_l1543_154388


namespace total_stickers_is_36_l1543_154332

/-- The number of stickers Elizabeth uses on her water bottles -/
def total_stickers : ℕ :=
  let initial_bottles : ℕ := 20
  let lost_school : ℕ := 5
  let found_park : ℕ := 3
  let stolen_dance : ℕ := 4
  let misplaced_library : ℕ := 2
  let acquired_friend : ℕ := 6
  let stickers_school : ℕ := 4
  let stickers_dance : ℕ := 3
  let stickers_library : ℕ := 2

  let school_stickers := lost_school * stickers_school
  let dance_stickers := stolen_dance * stickers_dance
  let library_stickers := misplaced_library * stickers_library

  school_stickers + dance_stickers + library_stickers

theorem total_stickers_is_36 : total_stickers = 36 := by
  sorry

end total_stickers_is_36_l1543_154332


namespace quadratic_max_value_l1543_154350

theorem quadratic_max_value :
  let f : ℝ → ℝ := fun x ↦ -3 * x^2 + 6 * x + 4
  ∃ m : ℝ, (∀ x : ℝ, f x ≤ m) ∧ (∃ x : ℝ, f x = m) ∧ m = 7 := by
  sorry

end quadratic_max_value_l1543_154350


namespace altitude_intersection_angle_l1543_154315

-- Define the triangle ABC
structure Triangle :=
  (A B C : Point)

-- Define the altitude intersection point H
def H (t : Triangle) : Point := sorry

-- Define the angles of the triangle
def angle_BAC (t : Triangle) : ℝ := sorry
def angle_ABC (t : Triangle) : ℝ := sorry

-- Define the angle AHB
def angle_AHB (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem altitude_intersection_angle (t : Triangle) 
  (h1 : angle_BAC t = 40)
  (h2 : angle_ABC t = 65) :
  angle_AHB t = 105 := by sorry

end altitude_intersection_angle_l1543_154315


namespace derivative_f_l1543_154323

noncomputable def f (x : ℝ) : ℝ :=
  (4^x * (Real.log 4 * Real.sin (4*x) - 4 * Real.cos (4*x))) / (16 + Real.log 4^2)

theorem derivative_f (x : ℝ) :
  deriv f x = 4^x * Real.sin (4*x) :=
sorry

end derivative_f_l1543_154323


namespace total_time_calculation_l1543_154377

-- Define the constants
def performance_time : ℕ := 6
def practice_ratio : ℕ := 3
def tantrum_ratio : ℕ := 5

-- Define the theorem
theorem total_time_calculation :
  performance_time * (1 + practice_ratio + tantrum_ratio) = 54 :=
by sorry

end total_time_calculation_l1543_154377


namespace smallest_odd_four_digit_number_l1543_154324

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

def swap_digits (n : ℕ) : ℕ :=
  let a := n / 1000
  let b := (n / 100) % 10
  let c := (n / 10) % 10
  let d := n % 10
  1000 * c + 100 * b + 10 * a + d

theorem smallest_odd_four_digit_number (n : ℕ) : 
  is_four_digit n ∧ 
  n % 2 = 1 ∧
  swap_digits n - n = 5940 ∧
  n % 9 = 8 ∧
  (∀ m : ℕ, is_four_digit m ∧ m % 2 = 1 ∧ swap_digits m - m = 5940 ∧ m % 9 = 8 → n ≤ m) →
  n = 1979 :=
sorry

end smallest_odd_four_digit_number_l1543_154324


namespace absolute_value_inequality_l1543_154359

theorem absolute_value_inequality (x : ℝ) :
  (3 ≤ |x + 2| ∧ |x + 2| ≤ 7) ↔ ((1 ≤ x ∧ x ≤ 5) ∨ (-9 ≤ x ∧ x ≤ -5)) :=
by sorry

end absolute_value_inequality_l1543_154359


namespace jerry_added_ten_books_l1543_154343

/-- The number of books Jerry added to his shelf -/
def books_added (initial_books final_books : ℕ) : ℕ :=
  final_books - initial_books

/-- Theorem stating that Jerry added 10 books to his shelf -/
theorem jerry_added_ten_books (initial_books final_books : ℕ) 
  (h1 : initial_books = 9)
  (h2 : final_books = 19) : 
  books_added initial_books final_books = 10 := by
  sorry

end jerry_added_ten_books_l1543_154343


namespace history_class_grades_l1543_154361

theorem history_class_grades (total_students : ℕ) 
  (prob_A : ℚ) (prob_B : ℚ) (prob_C : ℚ) :
  total_students = 31 →
  prob_A = 0.7 * prob_B →
  prob_C = 1.4 * prob_B →
  prob_A + prob_B + prob_C = 1 →
  (total_students : ℚ) * prob_B = 10 := by
  sorry

end history_class_grades_l1543_154361


namespace equation_transformation_l1543_154397

theorem equation_transformation (x : ℝ) :
  (x + 2) / 4 = (2 * x + 3) / 6 →
  12 * ((x + 2) / 4) = 12 * ((2 * x + 3) / 6) →
  3 * (x + 2) = 2 * (2 * x + 3) :=
by
  sorry

end equation_transformation_l1543_154397


namespace equal_squares_from_sum_product_l1543_154328

theorem equal_squares_from_sum_product (a b c d : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (h : a * (b + c + d) = b * (a + c + d) ∧ 
       b * (a + c + d) = c * (a + b + d) ∧ 
       c * (a + b + d) = d * (a + b + c)) : 
  a^2 = b^2 ∧ b^2 = c^2 ∧ c^2 = d^2 := by
sorry

end equal_squares_from_sum_product_l1543_154328


namespace g_sum_equals_one_l1543_154365

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the conditions
axiom func_property : ∀ x y : ℝ, f (x - y) = f x * g y - g x * f y
axiom f_nonzero : ∀ x : ℝ, f x ≠ 0
axiom f_equal : f 1 = f 2

-- State the theorem
theorem g_sum_equals_one : g (-1) + g 1 = 1 := by sorry

end g_sum_equals_one_l1543_154365


namespace f_is_odd_iff_l1543_154358

/-- A function f is odd if f(-x) = -f(x) for all x in the domain -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

/-- The function f(x) = x|x + a| + b -/
def f (a b : ℝ) : ℝ → ℝ := fun x ↦ x * |x + a| + b

/-- Theorem: f is an odd function if and only if a = 0 and b = 0 -/
theorem f_is_odd_iff (a b : ℝ) :
  IsOdd (f a b) ↔ a = 0 ∧ b = 0 := by
  sorry

end f_is_odd_iff_l1543_154358


namespace exists_same_answer_question_l1543_154355

/-- Represents a person who either always tells the truth or always lies -/
inductive Person
| TruthTeller
| Liar

/-- Represents a question that can be asked to a person -/
def Question := Type

/-- Represents an answer to a question -/
def Answer := Type

/-- The response function that determines how a person answers a question -/
def respond (p : Person) (q : Question) : Answer :=
  sorry

/-- Theorem stating that there exists a question that elicits the same answer from both a truth-teller and a liar -/
theorem exists_same_answer_question :
  ∃ (q : Question), ∀ (p1 p2 : Person), p1 ≠ p2 → respond p1 q = respond p2 q :=
sorry

end exists_same_answer_question_l1543_154355


namespace function_properties_l1543_154311

/-- Given a function f(x) = a*sin(x) + b*cos(x) where ab ≠ 0, a and b are real numbers,
    and for all x in ℝ, f(x) ≥ f(5π/6), then:
    1. f(π/3) = 0
    2. The line passing through (a, b) intersects the graph of f(x) -/
theorem function_properties (a b : ℝ) (h1 : a * b ≠ 0) :
  let f := fun (x : ℝ) ↦ a * Real.sin x + b * Real.cos x
  (∀ x : ℝ, f x ≥ f (5 * Real.pi / 6)) →
  (f (Real.pi / 3) = 0) ∧
  (∃ x : ℝ, f x = a * x + b) := by
  sorry

end function_properties_l1543_154311


namespace greatest_integer_difference_l1543_154319

theorem greatest_integer_difference (x y : ℝ) (hx : 4 < x ∧ x < 6) (hy : 6 < y ∧ y < 10) :
  (∀ (a b : ℝ), 4 < a ∧ a < 6 → 6 < b ∧ b < 10 → ⌊b - a⌋ ≤ 4) ∧ 
  (∃ (a b : ℝ), 4 < a ∧ a < 6 ∧ 6 < b ∧ b < 10 ∧ ⌊b - a⌋ = 4) :=
by sorry

end greatest_integer_difference_l1543_154319


namespace system_three_solutions_l1543_154338

/-- The system of equations has exactly three solutions if and only if a = 49 or a = 40 - 4√51 -/
theorem system_three_solutions (a : ℝ) : 
  (∃! x y z : ℝ × ℝ, 
    ((abs (y.2 - 10) + abs (x.1 + 3) - 2) * (x.1^2 + x.2^2 - 6) = 0 ∧
     (x.1 + 3)^2 + (x.2 - 5)^2 = a) ∧
    ((abs (y.2 - 10) + abs (y.1 + 3) - 2) * (y.1^2 + y.2^2 - 6) = 0 ∧
     (y.1 + 3)^2 + (y.2 - 5)^2 = a) ∧
    ((abs (z.2 - 10) + abs (z.1 + 3) - 2) * (z.1^2 + z.2^2 - 6) = 0 ∧
     (z.1 + 3)^2 + (z.2 - 5)^2 = a) ∧
    x ≠ y ∧ y ≠ z ∧ x ≠ z) ↔ 
  (a = 49 ∨ a = 40 - 4 * Real.sqrt 51) :=
by sorry

end system_three_solutions_l1543_154338


namespace probability_of_two_tails_in_three_flips_l1543_154357

def probability_of_k_successes (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k : ℝ) * p^k * (1 - p)^(n - k)

theorem probability_of_two_tails_in_three_flips :
  probability_of_k_successes 3 2 (1/2) = 0.375 := by
sorry

end probability_of_two_tails_in_three_flips_l1543_154357


namespace triangle_properties_l1543_154368

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Theorem about a specific triangle satisfying certain conditions -/
theorem triangle_properties (t : Triangle) 
  (h1 : (2 * t.a + t.c) * Real.cos t.B + t.b * Real.cos t.C = 0)
  (h2 : t.b = Real.sqrt 13)
  (h3 : t.a + t.c = 4) :
  t.B = 2 * Real.pi / 3 ∧ 
  (1/2) * t.a * t.c * Real.sin t.B = 3 * Real.sqrt 3 / 4 := by
  sorry

end triangle_properties_l1543_154368


namespace find_M_l1543_154372

theorem find_M : ∃ M : ℕ+, (18^2 * 45^2 : ℕ) = 30^2 * M^2 ∧ M = 81 := by
  sorry

end find_M_l1543_154372


namespace largest_lcm_with_18_l1543_154327

theorem largest_lcm_with_18 :
  (List.map (lcm 18) [3, 6, 9, 12, 15, 18]).maximum? = some 90 := by
  sorry

end largest_lcm_with_18_l1543_154327


namespace semicircle_perimeter_l1543_154382

/-- The perimeter of a semicircle with radius 6.7 cm is equal to π * 6.7 + 13.4 cm. -/
theorem semicircle_perimeter : 
  let r : ℝ := 6.7
  ∀ π : ℝ, π * r + 2 * r = π * r + 13.4 := by
  sorry

end semicircle_perimeter_l1543_154382


namespace complex_arithmetic_equality_l1543_154389

theorem complex_arithmetic_equality : (80 + (5 * 12) / (180 / 3)) ^ 2 * (7 - (3^2)) = -13122 := by
  sorry

end complex_arithmetic_equality_l1543_154389


namespace consistent_number_theorem_l1543_154318

def is_consistent_number (m : ℕ) : Prop :=
  ∃ (a b c d : ℕ), m = 1000 * a + 100 * b + 10 * c + d ∧
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧
    a + b = c + d

def F (m : ℕ) : ℚ :=
  let m' := (m / 10) % 10 * 1000 + m % 10 * 100 + m / 1000 * 10 + (m / 100) % 10
  (m + m') / 101

def G (N : ℕ) : ℕ := N / 10 + N % 10

theorem consistent_number_theorem :
  ∀ (m : ℕ), is_consistent_number m →
    let a := m / 1000
    let b := (m / 100) % 10
    let c := (m / 10) % 10
    let d := m % 10
    let N := 10 * a + 2 * b
    a ≤ 8 →
    d = 1 →
    Even (G N) →
    ∃ (k : ℤ), F m - G N - 4 * a = k^2 + 3 →
    (k = 6 ∨ k = -6) ∧ m = 2231 := by
  sorry

end consistent_number_theorem_l1543_154318


namespace repair_cost_calculation_l1543_154376

/-- Represents the repair cost calculation for Ramu's car sale --/
theorem repair_cost_calculation (initial_cost selling_price profit_percent : ℝ) (R : ℝ) : 
  initial_cost = 34000 →
  selling_price = 65000 →
  profit_percent = 41.30434782608695 →
  profit_percent = ((selling_price - (initial_cost + R)) / (initial_cost + R)) * 100 :=
by sorry

end repair_cost_calculation_l1543_154376


namespace real_part_of_z_l1543_154391

theorem real_part_of_z (z : ℂ) (h : (1 + Complex.I) * z = 2) : z.re = 1 := by
  sorry

end real_part_of_z_l1543_154391
