import Mathlib

namespace NUMINAMATH_CALUDE_simple_interest_problem_l3713_371342

/-- Given a sum put at simple interest for 3 years, if increasing the interest
    rate by 1% results in Rs. 69 more interest, then the sum is Rs. 2300. -/
theorem simple_interest_problem (P : ℝ) (R : ℝ) : 
  (P * (R + 1) * 3 / 100 - P * R * 3 / 100 = 69) → P = 2300 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l3713_371342


namespace NUMINAMATH_CALUDE_quadratic_inequality_l3713_371320

theorem quadratic_inequality (y : ℝ) : y^2 + 7*y < 12 ↔ -9 < y ∧ y < 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l3713_371320


namespace NUMINAMATH_CALUDE_smallest_number_l3713_371378

theorem smallest_number (S : Set ℤ) (hS : S = {0, 1, -5, -1}) :
  ∃ m ∈ S, ∀ x ∈ S, m ≤ x ∧ m = -5 := by sorry

end NUMINAMATH_CALUDE_smallest_number_l3713_371378


namespace NUMINAMATH_CALUDE_newspaper_profit_is_550_l3713_371357

/-- Calculate the profit from selling newspapers given the following conditions:
  * Total number of newspapers bought
  * Selling price per newspaper
  * Percentage of newspapers sold
  * Percentage discount on buying price compared to selling price
-/
def calculate_profit (total_newspapers : ℕ) (selling_price : ℚ) (sold_percentage : ℚ) (discount_percentage : ℚ) : ℚ :=
  let buying_price := selling_price * (1 - discount_percentage)
  let total_cost := buying_price * total_newspapers
  let newspapers_sold := (sold_percentage * total_newspapers).floor
  let revenue := selling_price * newspapers_sold
  revenue - total_cost

/-- Theorem stating that under the given conditions, the profit is $550 -/
theorem newspaper_profit_is_550 :
  calculate_profit 500 2 0.8 0.75 = 550 := by
  sorry

end NUMINAMATH_CALUDE_newspaper_profit_is_550_l3713_371357


namespace NUMINAMATH_CALUDE_flowers_died_in_danes_garden_l3713_371394

/-- The number of flowers that died in Dane's daughters' garden -/
def flowers_died (initial_flowers : ℕ) (new_flowers : ℕ) (baskets : ℕ) (flowers_per_basket : ℕ) : ℕ :=
  initial_flowers + new_flowers - (baskets * flowers_per_basket)

/-- Theorem stating the number of flowers that died in the specific scenario -/
theorem flowers_died_in_danes_garden : flowers_died 10 20 5 4 = 10 := by
  sorry

end NUMINAMATH_CALUDE_flowers_died_in_danes_garden_l3713_371394


namespace NUMINAMATH_CALUDE_yacht_capacity_problem_l3713_371316

theorem yacht_capacity_problem (large_capacity small_capacity : ℕ) : 
  (2 * large_capacity + 3 * small_capacity = 57) → 
  (3 * large_capacity + 2 * small_capacity = 68) → 
  (3 * large_capacity + 6 * small_capacity = 96) :=
by sorry

end NUMINAMATH_CALUDE_yacht_capacity_problem_l3713_371316


namespace NUMINAMATH_CALUDE_paint_cans_used_paint_cans_theorem_l3713_371361

-- Define the initial number of rooms that could be painted
def initial_rooms : ℕ := 50

-- Define the number of cans lost
def lost_cans : ℕ := 5

-- Define the number of rooms that can be painted after losing cans
def remaining_rooms : ℕ := 38

-- Define the number of small rooms to be painted
def small_rooms : ℕ := 35

-- Define the number of large rooms to be painted
def large_rooms : ℕ := 5

-- Define the paint requirement ratio of large rooms to small rooms
def large_room_ratio : ℕ := 2

-- Theorem to prove
theorem paint_cans_used : ℕ := by
  -- The proof goes here
  sorry

-- Goal: prove that paint_cans_used = 19
theorem paint_cans_theorem : paint_cans_used = 19 := by
  sorry

end NUMINAMATH_CALUDE_paint_cans_used_paint_cans_theorem_l3713_371361


namespace NUMINAMATH_CALUDE_expand_product_l3713_371369

theorem expand_product (x : ℝ) : (x + 3) * (x^2 + 4*x + 5) = x^3 + 7*x^2 + 17*x + 15 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l3713_371369


namespace NUMINAMATH_CALUDE_cosine_of_angle_l3713_371341

/-- Given two vectors a and b in ℝ², prove that the cosine of the angle between them is -63/65,
    when a + b = (2, -8) and a - b = (-8, 16). -/
theorem cosine_of_angle (a b : ℝ × ℝ) 
    (sum_eq : a + b = (2, -8)) 
    (diff_eq : a - b = (-8, 16)) : 
  (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)) = -63/65 := by
  sorry

end NUMINAMATH_CALUDE_cosine_of_angle_l3713_371341


namespace NUMINAMATH_CALUDE_specific_trapezoid_diagonal_l3713_371349

/-- An isosceles trapezoid with given dimensions -/
structure IsoscelesTrapezoid where
  bottom_base : ℝ
  top_base : ℝ
  side : ℝ

/-- The diagonal of an isosceles trapezoid -/
def diagonal (t : IsoscelesTrapezoid) : ℝ :=
  sorry

/-- Theorem: The diagonal of the specific isosceles trapezoid is 12√3 -/
theorem specific_trapezoid_diagonal :
  let t : IsoscelesTrapezoid := {
    bottom_base := 24,
    top_base := 12,
    side := 12
  }
  diagonal t = 12 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_specific_trapezoid_diagonal_l3713_371349


namespace NUMINAMATH_CALUDE_gcd_of_three_numbers_l3713_371321

theorem gcd_of_three_numbers : Nat.gcd 8247 (Nat.gcd 13619 29826) = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_three_numbers_l3713_371321


namespace NUMINAMATH_CALUDE_function_intersects_x_axis_l3713_371339

theorem function_intersects_x_axis (a : ℝ) : 
  (∀ m : ℝ, ∃ x : ℝ, m * x^2 + x - m - a = 0) → 
  a ∈ Set.Icc (-1 : ℝ) 1 :=
by sorry

end NUMINAMATH_CALUDE_function_intersects_x_axis_l3713_371339


namespace NUMINAMATH_CALUDE_tommy_has_100_nickels_l3713_371333

/-- Represents Tommy's coin collection --/
structure CoinCollection where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ
  quarters : ℕ
  half_dollars : ℕ
  dollar_coins : ℕ

/-- Calculates the total value of the coin collection in cents --/
def total_value (c : CoinCollection) : ℕ :=
  c.pennies + 5 * c.nickels + 10 * c.dimes + 25 * c.quarters + 50 * c.half_dollars + 100 * c.dollar_coins

/-- Tommy's coin collection satisfies the given conditions --/
def tommy_collection (c : CoinCollection) : Prop :=
  c.dimes = c.pennies + 10 ∧
  c.nickels = 2 * c.dimes ∧
  c.quarters = 4 ∧
  c.pennies = 10 * c.quarters ∧
  c.half_dollars = c.quarters + 5 ∧
  c.dollar_coins = 3 * c.half_dollars ∧
  total_value c = 2000

theorem tommy_has_100_nickels :
  ∀ c : CoinCollection, tommy_collection c → c.nickels = 100 := by
  sorry

end NUMINAMATH_CALUDE_tommy_has_100_nickels_l3713_371333


namespace NUMINAMATH_CALUDE_congruent_figures_alignment_l3713_371398

/-- A plane figure represented as a set of points in ℝ² -/
def PlaneFigure : Type := Set (ℝ × ℝ)

/-- Congruence relation between two plane figures -/
def Congruent (F G : PlaneFigure) : Prop := sorry

/-- Parallel translation of a plane figure -/
def ParallelTranslation (v : ℝ × ℝ) (F : PlaneFigure) : PlaneFigure := sorry

/-- Rotation of a plane figure around a point -/
def Rotation (center : ℝ × ℝ) (angle : ℝ) (F : PlaneFigure) : PlaneFigure := sorry

theorem congruent_figures_alignment (F G : PlaneFigure) (h : Congruent F G) :
  (∃ v : ℝ × ℝ, ParallelTranslation v F = G) ∨
  (∃ center : ℝ × ℝ, ∃ angle : ℝ, Rotation center angle F = G) :=
by sorry

end NUMINAMATH_CALUDE_congruent_figures_alignment_l3713_371398


namespace NUMINAMATH_CALUDE_garden_topsoil_cost_is_112_l3713_371390

/-- The cost of topsoil for a rectangular garden -/
def garden_topsoil_cost (length width depth price_per_cubic_foot : ℝ) : ℝ :=
  length * width * depth * price_per_cubic_foot

/-- Theorem: The cost of topsoil for the given garden is $112 -/
theorem garden_topsoil_cost_is_112 :
  garden_topsoil_cost 8 4 0.5 7 = 112 := by
  sorry

end NUMINAMATH_CALUDE_garden_topsoil_cost_is_112_l3713_371390


namespace NUMINAMATH_CALUDE_inscribed_square_area_l3713_371338

theorem inscribed_square_area (R : ℝ) (h : R > 0) :
  (R^2 * (π - 2) / 4 = 2*π - 4) →
  (2 * R)^2 = 16 := by
sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l3713_371338


namespace NUMINAMATH_CALUDE_not_perfect_square_l3713_371313

theorem not_perfect_square (a b : ℕ+) (h : (a.val^2 - b.val^2) % 4 ≠ 0) :
  ¬ ∃ (k : ℕ), (a.val + 3*b.val) * (5*a.val + 7*b.val) = k^2 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_l3713_371313


namespace NUMINAMATH_CALUDE_no_real_roots_l3713_371392

theorem no_real_roots : ∀ x : ℝ, 2 * x^2 - 4 * x + 3 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l3713_371392


namespace NUMINAMATH_CALUDE_sum_of_two_numbers_l3713_371307

theorem sum_of_two_numbers (s l : ℕ) : s = 9 → l = 4 * s → s + l = 45 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_numbers_l3713_371307


namespace NUMINAMATH_CALUDE_equilateral_triangle_to_trapezoid_l3713_371379

/-- Represents a paper shape that can be folded -/
structure PaperShape where
  vertices : ℕ
  layers : ℕ

/-- Represents the process of folding a paper shape -/
def fold (initial : PaperShape) (final : PaperShape) : Prop :=
  ∃ (steps : ℕ), steps > 0 ∧ final.layers ≥ initial.layers

/-- An equilateral triangle -/
def equilateralTriangle : PaperShape :=
  { vertices := 3, layers := 1 }

/-- A trapezoid -/
def trapezoid : PaperShape :=
  { vertices := 4, layers := 3 }

theorem equilateral_triangle_to_trapezoid :
  fold equilateralTriangle trapezoid :=
sorry

#check equilateral_triangle_to_trapezoid

end NUMINAMATH_CALUDE_equilateral_triangle_to_trapezoid_l3713_371379


namespace NUMINAMATH_CALUDE_dribbles_proof_l3713_371315

/-- Calculates the total number of dribbles in a given time period with decreasing dribble rate --/
def totalDribbles (initialDribbles : ℕ) (initialTime : ℕ) (secondIntervalDribbles : ℕ) (secondIntervalTime : ℕ) (totalTime : ℕ) (decreaseRate : ℕ) : ℕ :=
  let remainingTime := totalTime - initialTime - secondIntervalTime
  let fullIntervals := remainingTime / secondIntervalTime
  let lastIntervalTime := remainingTime % secondIntervalTime
  
  let initialPeriodDribbles := initialDribbles
  let secondPeriodDribbles := secondIntervalDribbles
  
  let remainingFullIntervalsDribbles := 
    (List.range fullIntervals).foldl (fun acc i => 
      acc + (secondIntervalDribbles - i * decreaseRate)) 0
  
  let lastIntervalDribbles := 
    (secondIntervalDribbles - fullIntervals * decreaseRate) * lastIntervalTime / secondIntervalTime
  
  initialPeriodDribbles + secondPeriodDribbles + remainingFullIntervalsDribbles + lastIntervalDribbles

/-- The total number of dribbles in 27 seconds is 83 --/
theorem dribbles_proof : 
  totalDribbles 13 3 18 5 27 3 = 83 := by
  sorry

end NUMINAMATH_CALUDE_dribbles_proof_l3713_371315


namespace NUMINAMATH_CALUDE_power_equality_l3713_371336

/-- Given n ∈ ℕ, x = (1 + 1/n)^n, and y = (1 + 1/n)^(n+1), prove that x^y = y^x -/
theorem power_equality (n : ℕ) (x y : ℝ) 
  (hx : x = (1 + 1/n)^n) 
  (hy : y = (1 + 1/n)^(n+1)) : 
  x^y = y^x := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l3713_371336


namespace NUMINAMATH_CALUDE_problem_statement_l3713_371318

theorem problem_statement (x y : ℝ) 
  (h1 : 3 * x + 2 * y = 11) 
  (h2 : y = 1) : 
  5 * x + 3 = 18 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3713_371318


namespace NUMINAMATH_CALUDE_y_divisibility_l3713_371386

def y : ℕ := 128 + 192 + 256 + 320 + 576 + 704 + 6464 + 8192

theorem y_divisibility :
  (∃ k : ℕ, y = 8 * k) ∧
  (∃ k : ℕ, y = 16 * k) ∧
  (∃ k : ℕ, y = 32 * k) ∧
  (∃ k : ℕ, y = 64 * k) :=
sorry

end NUMINAMATH_CALUDE_y_divisibility_l3713_371386


namespace NUMINAMATH_CALUDE_lemonade_distribution_theorem_l3713_371359

/-- Represents the distribution of lemonade cups --/
structure LemonadeDistribution where
  total : ℕ
  sold_to_kids : ℕ

/-- Checks if the lemonade distribution is valid --/
def is_valid_distribution (d : LemonadeDistribution) : Prop :=
  d.total = 56 ∧
  d.sold_to_kids + d.sold_to_kids / 2 + 1 = d.total / 2

/-- Theorem stating that the valid distribution has 18 cups sold to kids --/
theorem lemonade_distribution_theorem (d : LemonadeDistribution) :
  is_valid_distribution d → d.sold_to_kids = 18 := by
  sorry

#check lemonade_distribution_theorem

end NUMINAMATH_CALUDE_lemonade_distribution_theorem_l3713_371359


namespace NUMINAMATH_CALUDE_soccer_league_total_games_l3713_371347

theorem soccer_league_total_games (n : ℕ) (m : ℕ) (p : ℕ) :
  n = 20 →  -- number of teams
  m = 3 →   -- number of mid-season tournament matches per team
  p = 8 →   -- number of teams in playoffs
  (n * (n - 1) / 2) + (n * m / 2) + (p - 1) * 2 = 454 := by
  sorry

end NUMINAMATH_CALUDE_soccer_league_total_games_l3713_371347


namespace NUMINAMATH_CALUDE_f_increasing_condition_and_range_f_range_on_interval_l3713_371310

/-- The function f(x) = x^2 - 4x -/
def f (x : ℝ) : ℝ := x^2 - 4*x

theorem f_increasing_condition_and_range (a : ℝ) :
  (∀ x ≥ 2*a - 1, MonotoneOn f (Set.Ici (2*a - 1))) → a ≥ 3/2 :=
sorry

theorem f_range_on_interval :
  Set.image f (Set.Icc 1 7) = Set.Icc (-4) 21 :=
sorry

end NUMINAMATH_CALUDE_f_increasing_condition_and_range_f_range_on_interval_l3713_371310


namespace NUMINAMATH_CALUDE_whole_number_between_l3713_371385

theorem whole_number_between : ∀ M : ℤ, (5.5 < (M : ℚ) / 4 ∧ (M : ℚ) / 4 < 6) → M = 23 := by
  sorry

end NUMINAMATH_CALUDE_whole_number_between_l3713_371385


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3713_371397

/-- An arithmetic sequence with a positive common difference -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  h_arithmetic : ∀ n, a (n + 1) = a n + d
  h_positive : d > 0

/-- The theorem statement -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence) 
  (h_sum : seq.a 1 + seq.a 2 + seq.a 3 = 15)
  (h_product : seq.a 1 * seq.a 2 * seq.a 3 = 45) :
  seq.a 2009 + seq.a 2010 + seq.a 2011 = 24111 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3713_371397


namespace NUMINAMATH_CALUDE_cone_cube_distance_l3713_371380

/-- The distance between the vertex of a cone and the closest vertex of a cube placed inside it. -/
theorem cone_cube_distance (cube_edge : ℝ) (cone_diameter cone_height : ℝ) 
  (h_cube_edge : cube_edge = 3)
  (h_cone_diameter : cone_diameter = 8)
  (h_cone_height : cone_height = 24)
  (h_diagonal_coincide : ∃ (diagonal : ℝ), diagonal = cube_edge * Real.sqrt 3 ∧ 
    diagonal = cone_height * (cone_diameter / 8)) :
  ∃ (distance : ℝ), distance = 6 * Real.sqrt 6 - Real.sqrt 3 := by
sorry


end NUMINAMATH_CALUDE_cone_cube_distance_l3713_371380


namespace NUMINAMATH_CALUDE_problem_statement_l3713_371395

open Real

theorem problem_statement (x₀ : ℝ) : 
  let f := fun (x : ℝ) ↦ x * (2016 + log x)
  let f' := deriv f
  f' x₀ = 2017 → x₀ = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3713_371395


namespace NUMINAMATH_CALUDE_right_triangle_shorter_leg_l3713_371351

theorem right_triangle_shorter_leg (a b c : ℕ) : 
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  c = 65 →           -- Hypotenuse length
  a ≤ b →            -- a is the shorter leg
  a = 25             -- Shorter leg length
  := by sorry

end NUMINAMATH_CALUDE_right_triangle_shorter_leg_l3713_371351


namespace NUMINAMATH_CALUDE_smallest_prime_factor_of_2023_l3713_371353

theorem smallest_prime_factor_of_2023 : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 2023 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 2023 → p ≤ q :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_of_2023_l3713_371353


namespace NUMINAMATH_CALUDE_fermat_class_size_l3713_371358

theorem fermat_class_size : ∀ (total : ℕ),
  (total > 0) →
  (0.2 * (total : ℝ) = (total * 20 / 100 : ℕ)) →
  (0.35 * (total : ℝ) = (total * 35 / 100 : ℕ)) →
  (total - (total * 20 / 100) - (total * 35 / 100) = 9) →
  total = 20 := by
sorry

end NUMINAMATH_CALUDE_fermat_class_size_l3713_371358


namespace NUMINAMATH_CALUDE_max_product_ab_l3713_371389

theorem max_product_ab (a b : ℝ) : (∀ x : ℝ, Real.exp x ≥ a * x + b) → a * b ≤ Real.exp 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_max_product_ab_l3713_371389


namespace NUMINAMATH_CALUDE_poster_enlargement_l3713_371362

/-- Calculates the height of an enlarged poster while maintaining proportions -/
def enlargedPosterHeight (originalWidth originalHeight newWidth : ℚ) : ℚ :=
  (newWidth / originalWidth) * originalHeight

/-- Theorem: Given a poster with original dimensions of 3 inches wide and 2 inches tall,
    when enlarged proportionally to a width of 12 inches, the new height will be 8 inches -/
theorem poster_enlargement :
  let originalWidth : ℚ := 3
  let originalHeight : ℚ := 2
  let newWidth : ℚ := 12
  enlargedPosterHeight originalWidth originalHeight newWidth = 8 := by
  sorry

end NUMINAMATH_CALUDE_poster_enlargement_l3713_371362


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l3713_371304

theorem complex_fraction_equality : Complex.I * Complex.I = -1 → (7 - Complex.I) / (3 + Complex.I) = 2 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l3713_371304


namespace NUMINAMATH_CALUDE_max_gcd_lcm_value_l3713_371326

theorem max_gcd_lcm_value (a b c : ℕ) 
  (h : Nat.gcd (Nat.lcm a b) c * Nat.lcm (Nat.gcd a b) c = 200) : 
  Nat.gcd (Nat.lcm a b) c ≤ 10 ∧ 
  ∃ (a' b' c' : ℕ), Nat.gcd (Nat.lcm a' b') c' = 10 ∧ 
    Nat.gcd (Nat.lcm a' b') c' * Nat.lcm (Nat.gcd a' b') c' = 200 := by
  sorry

end NUMINAMATH_CALUDE_max_gcd_lcm_value_l3713_371326


namespace NUMINAMATH_CALUDE_impossible_equal_distribution_l3713_371345

/-- Represents the state of coins on the hexagon vertices -/
def HexagonState := Fin 6 → ℕ

/-- The initial state of the hexagon -/
def initial_state : HexagonState := fun i => if i = 0 then 1 else 0

/-- Represents a valid move in the game -/
def valid_move (s1 s2 : HexagonState) : Prop :=
  ∃ (i j : Fin 6) (n : ℕ), 
    (j = i + 1 ∨ j = i - 1 ∨ (i = 5 ∧ j = 0) ∨ (i = 0 ∧ j = 5)) ∧
    s2 i + 6 * n = s1 i ∧
    s2 j = s1 j + 6 * n ∧
    ∀ k, k ≠ i ∧ k ≠ j → s2 k = s1 k

/-- A sequence of valid moves -/
def valid_sequence (s : ℕ → HexagonState) : Prop :=
  s 0 = initial_state ∧ ∀ n, valid_move (s n) (s (n + 1))

/-- The theorem to be proved -/
theorem impossible_equal_distribution :
  ¬∃ (s : ℕ → HexagonState) (n : ℕ), 
    valid_sequence s ∧ 
    (∀ (i j : Fin 6), s n i = s n j) :=
sorry

end NUMINAMATH_CALUDE_impossible_equal_distribution_l3713_371345


namespace NUMINAMATH_CALUDE_line_proof_circle_proof_l3713_371368

-- Define the line
def line_equation (x y : ℝ) : Prop := y = x + 2

-- Define the circle
def circle_equation (x y : ℝ) : Prop := (x + 2)^2 + (y - 3)^2 = 4

-- Theorem for the line
theorem line_proof (x y : ℝ) (h1 : line_equation x y) :
  y = x + 2 := by sorry

-- Theorem for the circle
theorem circle_proof (x y : ℝ) (h2 : circle_equation x y) :
  (x + 2)^2 + (y - 3)^2 = 4 := by sorry

end NUMINAMATH_CALUDE_line_proof_circle_proof_l3713_371368


namespace NUMINAMATH_CALUDE_gcd_condition_iff_prime_representation_l3713_371311

theorem gcd_condition_iff_prime_representation (x y : ℕ) : 
  (∀ n : ℕ, Nat.gcd (n * (Nat.factorial x - x * y - x - y + 2) + 2) 
                    (n * (Nat.factorial x - x * y - x - y + 3) + 3) > 1) ↔
  (∃ q : ℕ, Prime q ∧ q > 3 ∧ x = q - 1 ∧ y = (Nat.factorial (q - 1) - (q - 1)) / q) :=
by sorry

end NUMINAMATH_CALUDE_gcd_condition_iff_prime_representation_l3713_371311


namespace NUMINAMATH_CALUDE_tan_beta_rationality_l3713_371396

theorem tan_beta_rationality (p q : ℤ) (hq : q ≠ 0) (α β : ℝ) 
  (h_tan_α : Real.tan α = p / q)
  (h_tan_2β : Real.tan (2 * β) = Real.tan (3 * α)) :
  (∃ (r s : ℤ) (hs : s ≠ 0), Real.tan β = r / s) ↔ 
  ∃ (n : ℤ), p^2 + q^2 = n^2 :=
sorry

end NUMINAMATH_CALUDE_tan_beta_rationality_l3713_371396


namespace NUMINAMATH_CALUDE_star_divided_by_square_equals_sixteen_l3713_371322

-- Define the symbols as real numbers
variable (triangle circle square star : ℝ)

-- State the conditions
axiom triangle_plus_triangle : triangle + triangle = star
axiom circle_equals_square_plus_square : circle = square + square
axiom triangle_equals_four_circles : triangle = circle + circle + circle + circle

-- State the theorem to be proved
theorem star_divided_by_square_equals_sixteen :
  star / square = 16 := by sorry

end NUMINAMATH_CALUDE_star_divided_by_square_equals_sixteen_l3713_371322


namespace NUMINAMATH_CALUDE_tan_960_degrees_l3713_371356

theorem tan_960_degrees : Real.tan (960 * Real.pi / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_960_degrees_l3713_371356


namespace NUMINAMATH_CALUDE_saleems_baskets_l3713_371327

theorem saleems_baskets (initial_avg_cost : ℝ) (additional_basket_cost : ℝ) (new_avg_cost : ℝ) :
  initial_avg_cost = 4 →
  additional_basket_cost = 8 →
  new_avg_cost = 4.8 →
  ∃ x : ℕ, x > 0 ∧ 
    (x * initial_avg_cost + additional_basket_cost) / (x + 1 : ℝ) = new_avg_cost ∧
    x = 4 :=
by sorry

end NUMINAMATH_CALUDE_saleems_baskets_l3713_371327


namespace NUMINAMATH_CALUDE_calendar_reuse_calendar_2032_reuse_l3713_371346

def is_leap_year (year : ℕ) : Prop :=
  (year % 4 = 0 ∧ year % 100 ≠ 0) ∨ year % 400 = 0

def calendar_repeat_cycle : ℕ := 28

theorem calendar_reuse (start_year : ℕ) (reuse_year : ℕ) : Prop :=
  start_year = 2032 →
  is_leap_year start_year →
  reuse_year > start_year →
  is_leap_year reuse_year →
  reuse_year - start_year = calendar_repeat_cycle * ((reuse_year - start_year) / calendar_repeat_cycle) →
  reuse_year = 2060

theorem calendar_2032_reuse :
  ∃ (reuse_year : ℕ), calendar_reuse 2032 reuse_year :=
sorry

end NUMINAMATH_CALUDE_calendar_reuse_calendar_2032_reuse_l3713_371346


namespace NUMINAMATH_CALUDE_granger_son_age_l3713_371387

/-- Mr. Granger's age in relation to his son's age -/
def granger_age (son_age : ℕ) : ℕ := 2 * son_age + 10

/-- Mr. Granger's age last year in relation to his son's age last year -/
def granger_age_last_year (son_age : ℕ) : ℕ := 3 * (son_age - 1) - 4

/-- Theorem stating that Mr. Granger's son is 16 years old -/
theorem granger_son_age : 
  ∃ (son_age : ℕ), son_age > 0 ∧ 
  granger_age son_age - 1 = granger_age_last_year son_age ∧ 
  son_age = 16 := by
  sorry

end NUMINAMATH_CALUDE_granger_son_age_l3713_371387


namespace NUMINAMATH_CALUDE_required_speed_increase_l3713_371388

/-- Represents the travel scenario for Ivan's commute --/
structure TravelScenario where
  usual_travel_time : ℝ
  usual_speed : ℝ
  late_start : ℝ
  speed_increase : ℝ
  time_saved : ℝ

/-- The theorem stating the required speed increase to arrive on time --/
theorem required_speed_increase (scenario : TravelScenario) 
  (h1 : scenario.late_start = 40)
  (h2 : scenario.speed_increase = 0.6)
  (h3 : scenario.time_saved = 65)
  (h4 : scenario.usual_travel_time / (1 + scenario.speed_increase) = 
        scenario.usual_travel_time - scenario.time_saved) :
  (scenario.usual_travel_time / (scenario.usual_travel_time - scenario.late_start)) - 1 = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_required_speed_increase_l3713_371388


namespace NUMINAMATH_CALUDE_wallet_and_purse_cost_l3713_371360

/-- The combined cost of a wallet and purse, where the wallet costs $22 and the purse costs $3 less than four times the cost of the wallet, is $107. -/
theorem wallet_and_purse_cost : 
  let wallet_cost : ℕ := 22
  let purse_cost : ℕ := 4 * wallet_cost - 3
  wallet_cost + purse_cost = 107 := by sorry

end NUMINAMATH_CALUDE_wallet_and_purse_cost_l3713_371360


namespace NUMINAMATH_CALUDE_solve_chocolate_problem_l3713_371376

def chocolate_problem (price_per_bar : ℕ) (total_bars : ℕ) (revenue : ℕ) : Prop :=
  let sold_bars : ℕ := revenue / price_per_bar
  let unsold_bars : ℕ := total_bars - sold_bars
  unsold_bars = 4

theorem solve_chocolate_problem :
  chocolate_problem 3 7 9 := by
  sorry

end NUMINAMATH_CALUDE_solve_chocolate_problem_l3713_371376


namespace NUMINAMATH_CALUDE_parallel_vectors_magnitude_l3713_371364

/-- Given vectors a and b, where a is parallel to b, prove that the magnitude of b is 3√5 -/
theorem parallel_vectors_magnitude (a b : ℝ × ℝ) (x : ℝ) :
  a = (2, -1) →
  b = (x, 3) →
  ∃ (k : ℝ), a = k • b →
  ‖b‖ = 3 * Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_parallel_vectors_magnitude_l3713_371364


namespace NUMINAMATH_CALUDE_club_membership_l3713_371381

theorem club_membership (total_members men : ℕ) (h1 : total_members = 52) (h2 : men = 37) :
  total_members - men = 15 := by
  sorry

end NUMINAMATH_CALUDE_club_membership_l3713_371381


namespace NUMINAMATH_CALUDE_chads_rope_length_l3713_371372

/-- Given that the ratio of Joey's rope length to Chad's rope length is 8:3,
    and Joey's rope is 56 cm long, prove that Chad's rope length is 21 cm. -/
theorem chads_rope_length (joey_length : ℝ) (chad_length : ℝ) 
    (h1 : joey_length = 56)
    (h2 : joey_length / chad_length = 8 / 3) : 
  chad_length = 21 := by
  sorry

end NUMINAMATH_CALUDE_chads_rope_length_l3713_371372


namespace NUMINAMATH_CALUDE_exists_monotone_increasing_symmetric_about_point_exists_three_roots_l3713_371354

-- Define the function f
def f (b c x : ℝ) : ℝ := |x| * x + b * x + c

-- Statement 1
theorem exists_monotone_increasing :
  ∃ b : ℝ, b > 0 ∧ ∀ x y : ℝ, x < y → f b 0 x < f b 0 y :=
sorry

-- Statement 2
theorem symmetric_about_point :
  ∀ b c : ℝ, ∀ x : ℝ, f b c x = f b c (-x) :=
sorry

-- Statement 3
theorem exists_three_roots :
  ∃ b c : ℝ, ∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    f b c x = 0 ∧ f b c y = 0 ∧ f b c z = 0 :=
sorry

end NUMINAMATH_CALUDE_exists_monotone_increasing_symmetric_about_point_exists_three_roots_l3713_371354


namespace NUMINAMATH_CALUDE_drummer_trombone_difference_l3713_371344

/-- Represents the number of players for each instrument in the school band --/
structure BandComposition where
  flute : Nat
  trumpet : Nat
  trombone : Nat
  clarinet : Nat
  frenchHorn : Nat
  drummer : Nat

/-- Theorem stating the difference between drummers and trombone players --/
theorem drummer_trombone_difference (band : BandComposition) : 
  band.flute = 5 →
  band.trumpet = 3 * band.flute →
  band.trombone = band.trumpet - 8 →
  band.clarinet = 2 * band.flute →
  band.frenchHorn = band.trombone + 3 →
  band.flute + band.trumpet + band.trombone + band.clarinet + band.frenchHorn + band.drummer = 65 →
  band.drummer > band.trombone →
  band.drummer - band.trombone = 11 := by
sorry


end NUMINAMATH_CALUDE_drummer_trombone_difference_l3713_371344


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_perpendicular_line_proof_l3713_371348

/-- A line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are perpendicular -/
def Line.perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

theorem perpendicular_line_through_point 
  (given_line : Line) 
  (point : Point) 
  (result_line : Line) : Prop :=
  given_line.a = 3 ∧ 
  given_line.b = -4 ∧ 
  given_line.c = 6 ∧
  point.x = 4 ∧ 
  point.y = -1 ∧
  result_line.a = 4 ∧ 
  result_line.b = 3 ∧ 
  result_line.c = -13 ∧
  point.liesOn result_line ∧
  Line.perpendicular given_line result_line

-- The proof of the theorem
theorem perpendicular_line_proof : 
  ∃ (given_line result_line : Line) (point : Point),
  perpendicular_line_through_point given_line point result_line := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_perpendicular_line_proof_l3713_371348


namespace NUMINAMATH_CALUDE_log_equality_l3713_371371

theorem log_equality (y : ℝ) (m : ℝ) : 
  (Real.log 5 / Real.log 8 = y) → 
  (Real.log 125 / Real.log 2 = m * y) → 
  m = 9 := by
sorry

end NUMINAMATH_CALUDE_log_equality_l3713_371371


namespace NUMINAMATH_CALUDE_inequality_proof_l3713_371312

theorem inequality_proof (a b c : ℝ) (ha : a ≥ c) (hb : b ≥ c) (hc : c > 0) :
  Real.sqrt (c * (a - c)) + Real.sqrt (c * (b - c)) ≤ Real.sqrt (a * b) ∧
  (Real.sqrt (c * (a - c)) + Real.sqrt (c * (b - c)) = Real.sqrt (a * b) ↔ a * b = c * (a + b)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3713_371312


namespace NUMINAMATH_CALUDE_translation_preserves_shape_and_size_l3713_371300

-- Define a geometric figure
structure GeometricFigure where
  -- We don't need to specify the exact properties of a geometric figure for this statement
  dummy : Unit

-- Define a translation
def Translation := ℝ → ℝ → ℝ → ℝ

-- Define the concept of shape preservation
def PreservesShape (f : Translation) (fig : GeometricFigure) : Prop :=
  -- The exact definition is not provided in the problem, so we leave it abstract
  sorry

-- Define the concept of size preservation
def PreservesSize (f : Translation) (fig : GeometricFigure) : Prop :=
  -- The exact definition is not provided in the problem, so we leave it abstract
  sorry

-- Theorem statement
theorem translation_preserves_shape_and_size (f : Translation) (fig : GeometricFigure) :
  PreservesShape f fig ∧ PreservesSize f fig :=
by
  sorry

end NUMINAMATH_CALUDE_translation_preserves_shape_and_size_l3713_371300


namespace NUMINAMATH_CALUDE_subset_relation_l3713_371399

theorem subset_relation (P Q : Set ℝ) : Q ⊆ P :=
  by
  -- Define sets P and Q
  have h_P : P = {x : ℝ | x < 4} := by sorry
  have h_Q : Q = {x : ℝ | x^2 < 4} := by sorry

  -- Prove that Q is a subset of P
  sorry

end NUMINAMATH_CALUDE_subset_relation_l3713_371399


namespace NUMINAMATH_CALUDE_tangent_line_inclination_range_l3713_371375

open Real

theorem tangent_line_inclination_range (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 4) :
  let f := fun x => Real.log x + x / b
  let α := Real.arctan (((1 / a) + (1 / b)) : ℝ)
  π / 4 ≤ α ∧ α < π / 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_inclination_range_l3713_371375


namespace NUMINAMATH_CALUDE_systematic_sampling_theorem_l3713_371337

/-- Represents a systematic sampling of students. -/
structure SystematicSampling where
  total_students : ℕ
  num_groups : ℕ
  students_per_group : ℕ
  interval : ℕ

/-- Creates a systematic sampling for the given problem. -/
def create_sampling : SystematicSampling :=
  { total_students := 50
  , num_groups := 10
  , students_per_group := 5
  , interval := 10
  }

/-- Calculates the number drawn from a specific group given the number drawn from another group. -/
def calculate_number (s : SystematicSampling) (known_group : ℕ) (known_number : ℕ) (target_group : ℕ) : ℕ :=
  known_number + (target_group - known_group) * s.interval

/-- Theorem stating that if the number drawn from the third group is 13, 
    then the number drawn from the seventh group is 53. -/
theorem systematic_sampling_theorem (s : SystematicSampling) 
  (h1 : s = create_sampling) 
  (h2 : calculate_number s 3 13 7 = 53) : 
  calculate_number s 3 13 7 = 53 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_theorem_l3713_371337


namespace NUMINAMATH_CALUDE_positive_expressions_l3713_371331

theorem positive_expressions (a b c : ℝ) 
  (ha : 0 < a ∧ a < 2) 
  (hb : -2 < b ∧ b < 0) 
  (hc : 0 < c ∧ c < 3) : 
  0 < b + b^2 ∧ 0 < b + 3*b^2 := by
  sorry

end NUMINAMATH_CALUDE_positive_expressions_l3713_371331


namespace NUMINAMATH_CALUDE_absolute_value_nonnegative_l3713_371317

theorem absolute_value_nonnegative (a : ℝ) : |a| ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_nonnegative_l3713_371317


namespace NUMINAMATH_CALUDE_complement_intersection_equality_l3713_371324

-- Define the sets P and Q
def P : Set ℝ := {x : ℝ | x ≥ 2}
def Q : Set ℝ := {x : ℝ | 1 < x ∧ x ≤ 2}

-- Define the open interval (1, 2)
def open_interval : Set ℝ := {x : ℝ | 1 < x ∧ x < 2}

-- State the theorem
theorem complement_intersection_equality :
  (Set.univ \ P) ∩ Q = open_interval :=
sorry

end NUMINAMATH_CALUDE_complement_intersection_equality_l3713_371324


namespace NUMINAMATH_CALUDE_simplify_expression_l3713_371350

theorem simplify_expression (x y : ℝ) : 3*x + 5*x + 7*x + 2*y = 15*x + 2*y := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3713_371350


namespace NUMINAMATH_CALUDE_unknown_number_proof_l3713_371366

theorem unknown_number_proof (n : ℕ) : (n ^ 1) * 6 ^ 4 / 432 = 36 → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_proof_l3713_371366


namespace NUMINAMATH_CALUDE_melissa_driving_hours_l3713_371377

/-- Calculates the total hours Melissa spends driving in a year -/
def total_driving_hours (trips_per_month : ℕ) (hours_per_trip : ℕ) (months_per_year : ℕ) : ℕ :=
  trips_per_month * hours_per_trip * months_per_year

/-- Proves that Melissa spends 72 hours driving in a year -/
theorem melissa_driving_hours :
  total_driving_hours 2 3 12 = 72 := by
  sorry

end NUMINAMATH_CALUDE_melissa_driving_hours_l3713_371377


namespace NUMINAMATH_CALUDE_y_exceeds_x_by_25_percent_l3713_371330

theorem y_exceeds_x_by_25_percent (x y : ℝ) (h : x = 0.8 * y) : 
  (y - x) / x = 0.25 := by
sorry

end NUMINAMATH_CALUDE_y_exceeds_x_by_25_percent_l3713_371330


namespace NUMINAMATH_CALUDE_exponent_calculation_l3713_371365

theorem exponent_calculation : (-1 : ℝ)^53 + 3^(2^3 + 5^2 - 7^2) = -1 + (1 : ℝ) / 3^16 := by
  sorry

end NUMINAMATH_CALUDE_exponent_calculation_l3713_371365


namespace NUMINAMATH_CALUDE_binomial_expansion_arithmetic_sequence_max_coefficient_terms_sqrt_inequality_l3713_371328

-- Part 1
theorem binomial_expansion_arithmetic_sequence (n : ℕ) :
  (∃ r : ℚ, r ≠ 0 ∧ 
    n.choose 0 + (1/4) * n.choose 2 = 2 * (1/2) * n.choose 1) →
  n = 8 :=
sorry

-- Part 2
theorem max_coefficient_terms (n : ℕ) (x : ℝ) :
  n = 8 →
  ∃ c : ℝ, c > 0 ∧
    (∀ k : ℕ, k ≤ n → 
      c * x^(5 : ℝ) ≥ (1/(2^k : ℝ)) * n.choose k * x^((n - k : ℝ)/2)) ∧
    (∀ k : ℕ, k ≤ n → 
      c * x^(7/2 : ℝ) ≥ (1/(2^k : ℝ)) * n.choose k * x^((n - k : ℝ)/2)) :=
sorry

-- Part 3
theorem sqrt_inequality (a : ℝ) :
  a > 1 →
  Real.sqrt (a + 1) - Real.sqrt a < Real.sqrt a - Real.sqrt (a - 1) :=
sorry

end NUMINAMATH_CALUDE_binomial_expansion_arithmetic_sequence_max_coefficient_terms_sqrt_inequality_l3713_371328


namespace NUMINAMATH_CALUDE_largest_prime_sum_2500_l3713_371323

def isPrime (n : ℕ) : Prop := sorry

def sumOfPrimesUpTo (n : ℕ) : ℕ := sorry

theorem largest_prime_sum_2500 :
  ∀ p : ℕ, isPrime p →
    (p ≤ 151 → sumOfPrimesUpTo p ≤ 2500) ∧
    (p > 151 → sumOfPrimesUpTo p > 2500) :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_sum_2500_l3713_371323


namespace NUMINAMATH_CALUDE_weed_eating_money_l3713_371325

def mowing_money : ℕ := 14
def weeks_lasted : ℕ := 8
def weekly_spending : ℕ := 5

def total_money : ℕ := weeks_lasted * weekly_spending

theorem weed_eating_money :
  total_money - mowing_money = 26 := by sorry

end NUMINAMATH_CALUDE_weed_eating_money_l3713_371325


namespace NUMINAMATH_CALUDE_sequence_difference_l3713_371363

theorem sequence_difference (a : ℕ → ℝ) 
  (h : ∀ n : ℕ, a n + a (n + 1) = 4 * n + 3) : 
  a 10 - a 8 = 4 := by
sorry

end NUMINAMATH_CALUDE_sequence_difference_l3713_371363


namespace NUMINAMATH_CALUDE_complex_modulus_power_eight_l3713_371308

theorem complex_modulus_power_eight : Complex.abs ((1 - Complex.I) ^ 8) = 16 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_power_eight_l3713_371308


namespace NUMINAMATH_CALUDE_equal_numbers_product_l3713_371335

theorem equal_numbers_product (a b c d e : ℝ) : 
  (a + b + c + d + e) / 5 = 20 →
  a = 12 →
  b = 22 →
  c = 16 →
  d = e →
  d * e = 625 := by
sorry

end NUMINAMATH_CALUDE_equal_numbers_product_l3713_371335


namespace NUMINAMATH_CALUDE_distance_in_A_l3713_371393

/-- A positive three-digit integer -/
def ThreeDigitInt := { n : ℕ // 100 ≤ n ∧ n ≤ 999 }

/-- Sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Set of positive three-digit integers with digit sum 16 -/
def A : Set ThreeDigitInt := { a | digit_sum a.val = 16 }

/-- The greatest distance between two numbers in set A -/
def max_distance : ℕ := sorry

/-- The smallest distance between two numbers in set A -/
def min_distance : ℕ := sorry

/-- Theorem stating the greatest and smallest distances in set A -/
theorem distance_in_A : max_distance = 801 ∧ min_distance = 9 := by sorry

end NUMINAMATH_CALUDE_distance_in_A_l3713_371393


namespace NUMINAMATH_CALUDE_beavers_still_working_l3713_371306

theorem beavers_still_working (total : ℕ) (wood : ℕ) (dam : ℕ) (lodge : ℕ)
  (wood_break : ℕ) (dam_break : ℕ) (lodge_break : ℕ)
  (h1 : total = 12)
  (h2 : wood = 5)
  (h3 : dam = 4)
  (h4 : lodge = 3)
  (h5 : wood_break = 3)
  (h6 : dam_break = 2)
  (h7 : lodge_break = 1) :
  (wood - wood_break) + (dam - dam_break) + (lodge - lodge_break) = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_beavers_still_working_l3713_371306


namespace NUMINAMATH_CALUDE_grain_mass_calculation_l3713_371329

theorem grain_mass_calculation (given_mass : ℝ) (given_fraction : ℝ) (total_mass : ℝ) : 
  given_mass = 0.5 → given_fraction = 0.2 → given_mass = given_fraction * total_mass → total_mass = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_grain_mass_calculation_l3713_371329


namespace NUMINAMATH_CALUDE_angle_line_plane_range_l3713_371332

/-- The angle between a line and a plane is the acute angle between the line and its projection onto the plane. -/
def angle_line_plane (line : Line) (plane : Plane) : ℝ := sorry

theorem angle_line_plane_range (line : Line) (plane : Plane) :
  let θ := angle_line_plane line plane
  0 ≤ θ ∧ θ ≤ 90 :=
sorry

end NUMINAMATH_CALUDE_angle_line_plane_range_l3713_371332


namespace NUMINAMATH_CALUDE_lines_intersection_l3713_371314

-- Define the two lines
def line1 (t : ℝ) : ℝ × ℝ := (3 * t, 2 + 4 * t)
def line2 (u : ℝ) : ℝ × ℝ := (1 + u, 1 - u)

-- State the theorem
theorem lines_intersection :
  ∃! p : ℝ × ℝ, (∃ t : ℝ, line1 t = p) ∧ (∃ u : ℝ, line2 u = p) ∧ p = (0, 2) := by
  sorry

end NUMINAMATH_CALUDE_lines_intersection_l3713_371314


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_l3713_371319

/-- The line 3x - 4y - 5 = 0 is tangent to the circle (x - 1)^2 + (y + 3)^2 = 4 -/
theorem line_tangent_to_circle :
  ∃ (x y : ℝ),
    (3 * x - 4 * y - 5 = 0) ∧
    ((x - 1)^2 + (y + 3)^2 = 4) ∧
    (∀ (x' y' : ℝ), (3 * x' - 4 * y' - 5 = 0) → ((x' - 1)^2 + (y' + 3)^2 ≥ 4)) :=
sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_l3713_371319


namespace NUMINAMATH_CALUDE_fraction_comparison_l3713_371352

theorem fraction_comparison : (2 : ℝ) / 3 > (5 - Real.sqrt 11) / 3 := by sorry

end NUMINAMATH_CALUDE_fraction_comparison_l3713_371352


namespace NUMINAMATH_CALUDE_triangle_area_l3713_371384

/-- Given a triangle with side lengths 9, 12, and 15 units, its area is 54 square units. -/
theorem triangle_area : ∀ (a b c : ℝ), 
  a = 9 ∧ b = 12 ∧ c = 15 →
  (∃ (A : ℝ), A = (1/2) * a * b ∧ A = 54) :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l3713_371384


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l3713_371382

theorem pure_imaginary_condition (z : ℂ) (a b : ℝ) : 
  z = Complex.mk a b → z.re = 0 → a = 0 ∧ b ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l3713_371382


namespace NUMINAMATH_CALUDE_min_product_of_three_l3713_371370

def S : Finset Int := {-9, -5, -3, 0, 4, 6, 8}

theorem min_product_of_three (a b c : Int) (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) 
  (hdiff : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  ∃ (x y z : Int), x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
  x * y * z ≤ a * b * c ∧ x * y * z = -432 :=
sorry

end NUMINAMATH_CALUDE_min_product_of_three_l3713_371370


namespace NUMINAMATH_CALUDE_factorization_of_2m_squared_minus_18_l3713_371303

theorem factorization_of_2m_squared_minus_18 (m : ℝ) : 2 * m^2 - 18 = 2 * (m + 3) * (m - 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_2m_squared_minus_18_l3713_371303


namespace NUMINAMATH_CALUDE_exercise_book_cost_l3713_371383

/-- Proves that the total cost of buying 'a' exercise books at 0.8 yuan each is 0.8a yuan -/
theorem exercise_book_cost (a : ℝ) : 
  let cost_per_book : ℝ := 0.8
  let num_books : ℝ := a
  let total_cost : ℝ := cost_per_book * num_books
  total_cost = 0.8 * a := by sorry

end NUMINAMATH_CALUDE_exercise_book_cost_l3713_371383


namespace NUMINAMATH_CALUDE_distance_difference_l3713_371334

-- Define the travel parameters
def grayson_speed1 : ℝ := 25
def grayson_time1 : ℝ := 1
def grayson_speed2 : ℝ := 20
def grayson_time2 : ℝ := 0.5
def rudy_speed : ℝ := 10
def rudy_time : ℝ := 3

-- Calculate distances
def grayson_distance1 : ℝ := grayson_speed1 * grayson_time1
def grayson_distance2 : ℝ := grayson_speed2 * grayson_time2
def grayson_total_distance : ℝ := grayson_distance1 + grayson_distance2
def rudy_distance : ℝ := rudy_speed * rudy_time

-- Theorem to prove
theorem distance_difference :
  grayson_total_distance - rudy_distance = 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_difference_l3713_371334


namespace NUMINAMATH_CALUDE_complex_root_equation_l3713_371301

/-- Given a quadratic equation with complex coefficients and a real parameter,
    prove that if it has a real root, then the complex number formed by the
    parameter and the root has a specific value. -/
theorem complex_root_equation (a : ℝ) (b : ℝ) :
  (∃ x : ℝ, x^2 + (4 + Complex.I) * x + (4 : ℂ) + a * Complex.I = 0) →
  (b^2 + (4 + Complex.I) * b + (4 : ℂ) + a * Complex.I = 0) →
  (a + b * Complex.I = 2 - 2 * Complex.I) := by
  sorry

end NUMINAMATH_CALUDE_complex_root_equation_l3713_371301


namespace NUMINAMATH_CALUDE_valid_liar_counts_l3713_371373

/-- Represents the number of people in the room -/
def total_people : ℕ := 30

/-- Represents the possible numbers of liars in the room -/
def possible_liar_counts : List ℕ := [2, 3, 5, 6, 10, 15, 30]

/-- Predicate to check if a number is a valid liar count -/
def is_valid_liar_count (x : ℕ) : Prop :=
  x > 1 ∧ (total_people % x = 0) ∧
  ∃ (n : ℕ), (n + 1) * x = total_people

/-- Theorem stating that the possible_liar_counts are the only valid liar counts -/
theorem valid_liar_counts :
  ∀ (x : ℕ), is_valid_liar_count x ↔ x ∈ possible_liar_counts :=
by sorry

end NUMINAMATH_CALUDE_valid_liar_counts_l3713_371373


namespace NUMINAMATH_CALUDE_intersection_is_origin_l3713_371374

/-- The line l in Cartesian coordinates -/
def line_l (x y : ℝ) : Prop := y = Real.sqrt 3 * x

/-- The curve C in Cartesian coordinates -/
def curve_C (x y : ℝ) : Prop := y = (1/2) * x^2

/-- The intersection point of line l and curve C -/
def intersection_point : ℝ × ℝ := (0, 0)

/-- Theorem stating that the intersection point is (0, 0) -/
theorem intersection_is_origin :
  line_l (intersection_point.1) (intersection_point.2) ∧
  curve_C (intersection_point.1) (intersection_point.2) :=
sorry

end NUMINAMATH_CALUDE_intersection_is_origin_l3713_371374


namespace NUMINAMATH_CALUDE_dining_bill_calculation_l3713_371343

/-- Proves that the original bill before tip was $139 given the problem conditions -/
theorem dining_bill_calculation (people : ℕ) (tip_percentage : ℚ) (individual_payment : ℚ) :
  people = 5 ∧ 
  tip_percentage = 1/10 ∧ 
  individual_payment = 3058/100 →
  ∃ (original_bill : ℚ),
    original_bill * (1 + tip_percentage) = people * individual_payment ∧
    original_bill = 139 :=
by sorry

end NUMINAMATH_CALUDE_dining_bill_calculation_l3713_371343


namespace NUMINAMATH_CALUDE_park_trees_count_l3713_371302

theorem park_trees_count : ∃! n : ℕ, 
  80 < n ∧ n < 150 ∧ 
  n % 4 = 2 ∧ 
  n % 5 = 3 ∧ 
  n % 6 = 4 ∧ 
  n = 98 := by sorry

end NUMINAMATH_CALUDE_park_trees_count_l3713_371302


namespace NUMINAMATH_CALUDE_line_t_value_l3713_371367

/-- A line passing through points (2, 8), (6, 20), (10, 32), and (35, t) -/
structure Line where
  -- Define the slope of the line
  slope : ℝ
  -- Define the y-intercept of the line
  y_intercept : ℝ
  -- Ensure the line passes through (2, 8)
  point1_condition : 8 = slope * 2 + y_intercept
  -- Ensure the line passes through (6, 20)
  point2_condition : 20 = slope * 6 + y_intercept
  -- Ensure the line passes through (10, 32)
  point3_condition : 32 = slope * 10 + y_intercept

/-- The t-value for the point (35, t) on the line -/
def t_value (l : Line) : ℝ := l.slope * 35 + l.y_intercept

theorem line_t_value : ∀ l : Line, t_value l = 107 := by sorry

end NUMINAMATH_CALUDE_line_t_value_l3713_371367


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l3713_371340

theorem interest_rate_calculation (principal_B principal_C time_B time_C total_interest : ℕ) 
  (h1 : principal_B = 5000)
  (h2 : principal_C = 3000)
  (h3 : time_B = 2)
  (h4 : time_C = 4)
  (h5 : total_interest = 2640) :
  let rate := total_interest * 100 / (principal_B * time_B + principal_C * time_C)
  rate = 12 := by sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l3713_371340


namespace NUMINAMATH_CALUDE_two_solutions_for_equation_l3713_371309

theorem two_solutions_for_equation : 
  ∃! (n : ℕ), n = (Finset.filter 
    (fun p : ℕ × ℕ => 
      let a := p.1
      let b := p.2
      a > 0 ∧ b > 0 ∧ (a + b + 3)^2 = 4*(a^2 + b^2))
    (Finset.product (Finset.range 1000) (Finset.range 1000))).card 
  ∧ n = 2 := by sorry

end NUMINAMATH_CALUDE_two_solutions_for_equation_l3713_371309


namespace NUMINAMATH_CALUDE_total_numbers_l3713_371305

theorem total_numbers (average : ℝ) (first_six_average : ℝ) (last_eight_average : ℝ) (eighth_number : ℝ)
  (h1 : average = 60)
  (h2 : first_six_average = 57)
  (h3 : last_eight_average = 61)
  (h4 : eighth_number = 50) :
  ∃ n : ℕ, n = 13 ∧ 
    average * n = first_six_average * 6 + last_eight_average * 8 - eighth_number :=
by
  sorry


end NUMINAMATH_CALUDE_total_numbers_l3713_371305


namespace NUMINAMATH_CALUDE_quadratic_always_positive_l3713_371355

theorem quadratic_always_positive : ∀ x : ℝ, x^2 + 2*x + 2 > 0 := by sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_l3713_371355


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l3713_371391

-- Define the quadratic function
def f (a x : ℝ) : ℝ := a * x^2 - 2*x + 1

-- Define the property of having a non-empty solution set
def has_nonempty_solution_set (a : ℝ) : Prop :=
  ∃ x, f a x < 0

-- State the theorem
theorem necessary_but_not_sufficient_condition :
  (∃ a, has_nonempty_solution_set a ∧ a ≥ 2) ∧
  (∃ a, ¬has_nonempty_solution_set a ∧ a < 2) :=
sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l3713_371391
