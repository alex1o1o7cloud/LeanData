import Mathlib

namespace four_distinct_solutions_range_l743_74302

-- Define the equation
def f (x m : ℝ) : ℝ := x^2 - 4 * |x| + 5 - m

-- State the theorem
theorem four_distinct_solutions_range (m : ℝ) :
  (∃ a b c d : ℝ, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    f a m = 0 ∧ f b m = 0 ∧ f c m = 0 ∧ f d m = 0) →
  m ∈ Set.Ioo 1 5 :=
by sorry

end four_distinct_solutions_range_l743_74302


namespace anthony_friend_house_distance_l743_74386

/-- Given the distances between various locations, prove the distance to Anthony's friend's house -/
theorem anthony_friend_house_distance 
  (distance_to_work : ℝ) 
  (distance_to_gym : ℝ) 
  (distance_to_grocery : ℝ) 
  (distance_to_friend : ℝ) : 
  distance_to_work = 10 ∧ 
  distance_to_gym = (distance_to_work / 2) + 2 ∧
  distance_to_grocery = 4 ∧
  distance_to_grocery = 2 * distance_to_gym ∧
  distance_to_friend = 3 * (distance_to_gym + distance_to_grocery) →
  distance_to_friend = 63 := by
  sorry


end anthony_friend_house_distance_l743_74386


namespace pen_pricing_gain_percentage_l743_74333

theorem pen_pricing_gain_percentage :
  ∀ (C S : ℝ),
  C > 0 →
  20 * C = 12 * S →
  (S - C) / C * 100 = 200 / 3 :=
by
  sorry

end pen_pricing_gain_percentage_l743_74333


namespace largest_even_from_powerful_digits_l743_74397

/-- A natural number is powerful if n + (n+1) + (n+2) has no carrying over --/
def isPowerful (n : ℕ) : Prop :=
  ∀ d : ℕ, d > 0 → (n / 10^d % 10 + (n+1) / 10^d % 10 + (n+2) / 10^d % 10) < 10

/-- The set of powerful numbers less than 1000 --/
def powerfulSet : Set ℕ := {n | n < 1000 ∧ isPowerful n}

/-- The set of digits from powerful numbers less than 1000 --/
def powerfulDigits : Set ℕ := {d | ∃ n ∈ powerfulSet, ∃ k, n / 10^k % 10 = d}

/-- An even number formed by non-repeating digits from powerfulDigits --/
def validNumber (n : ℕ) : Prop :=
  n % 2 = 0 ∧ 
  (∀ d, d ∈ powerfulDigits → (∃! k, n / 10^k % 10 = d)) ∧
  (∀ k, n / 10^k % 10 ∈ powerfulDigits)

theorem largest_even_from_powerful_digits :
  ∃ n, validNumber n ∧ ∀ m, validNumber m → m ≤ n ∧ n = 43210 :=
sorry

end largest_even_from_powerful_digits_l743_74397


namespace shopping_spree_theorem_l743_74363

def shopping_spree (initial_amount : ℝ) (book_price : ℝ) (num_books : ℕ) 
  (game_price : ℝ) (water_bottle_price : ℝ) (snack_price : ℝ) (num_snacks : ℕ)
  (bundle_price : ℝ) (book_discount : ℝ) (tax_rate : ℝ) : ℝ :=
  let book_total := book_price * num_books
  let discounted_book_total := book_total * (1 - book_discount)
  let subtotal := discounted_book_total + game_price + water_bottle_price + 
                  (snack_price * num_snacks) + bundle_price
  let total_with_tax := subtotal * (1 + tax_rate)
  initial_amount - total_with_tax

theorem shopping_spree_theorem :
  shopping_spree 200 12 5 45 10 3 3 20 0.1 0.12 = 45.44 := by sorry

end shopping_spree_theorem_l743_74363


namespace expression_value_l743_74311

def numerator : ℤ := 20 - 19 + 18 - 17 + 16 - 15 + 14 - 13 + 12 - 11 + 10 - 9 + 8 - 7 + 6 - 5 + 4 - 3 + 2 - 1

def denominator : ℤ := 1 - 2 + 3 - 4 + 5 - 6 + 7 - 8 + 9 - 10 + 11 - 12 + 13 - 14 + 15 - 16 + 17 - 18 + 19 - 20

theorem expression_value : (numerator : ℚ) / denominator = -1 := by
  sorry

end expression_value_l743_74311


namespace larry_stickers_l743_74385

theorem larry_stickers (initial : ℕ) (lost : ℕ) (gained : ℕ) 
  (h1 : initial = 193) 
  (h2 : lost = 6) 
  (h3 : gained = 12) : 
  initial - lost + gained = 199 := by
  sorry

end larry_stickers_l743_74385


namespace min_sum_of_product_2550_l743_74375

theorem min_sum_of_product_2550 (a b c : ℕ+) (h : a * b * c = 2550) :
  ∃ (x y z : ℕ+), x * y * z = 2550 ∧ x + y + z ≤ a + b + c ∧ x + y + z = 48 :=
by sorry

end min_sum_of_product_2550_l743_74375


namespace min_tiles_needed_is_260_l743_74347

/-- Represents the dimensions of a rectangle in inches -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangle given its dimensions -/
def area (d : Dimensions) : ℕ := d.length * d.width

/-- Converts feet to inches -/
def feetToInches (feet : ℕ) : ℕ := feet * 12

/-- The dimensions of the tile -/
def tileDimensions : Dimensions := ⟨2, 5⟩

/-- The dimensions of the floor in feet -/
def floorDimensionsFeet : Dimensions := ⟨3, 6⟩

/-- The dimensions of the floor in inches -/
def floorDimensionsInches : Dimensions :=
  ⟨feetToInches floorDimensionsFeet.length, feetToInches floorDimensionsFeet.width⟩

/-- Calculates the minimum number of tiles needed to cover the floor -/
def minTilesNeeded : ℕ :=
  (area floorDimensionsInches + area tileDimensions - 1) / area tileDimensions

theorem min_tiles_needed_is_260 : minTilesNeeded = 260 := by
  sorry

end min_tiles_needed_is_260_l743_74347


namespace arithmetic_calculation_l743_74360

theorem arithmetic_calculation : 1984 + 180 / 60 - 284 = 1703 := by
  sorry

end arithmetic_calculation_l743_74360


namespace optimal_choice_is_104_l743_74370

/-- Counts the number of distinct rectangles with integer sides for a given perimeter --/
def countRectangles (perimeter : ℕ) : ℕ :=
  if perimeter % 2 = 0 then
    (perimeter / 4 : ℕ)
  else
    0

/-- Checks if a number is a valid choice in the game --/
def isValidChoice (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 105

/-- Theorem stating that 104 is the optimal choice for Grisha --/
theorem optimal_choice_is_104 :
  ∀ n, isValidChoice n → countRectangles 104 ≥ countRectangles n :=
by sorry

end optimal_choice_is_104_l743_74370


namespace no_function_satisfies_conditions_l743_74378

theorem no_function_satisfies_conditions : ¬∃ f : ℝ → ℝ, 
  (∀ y : ℝ, ∃ x : ℝ, f x = y) ∧ 
  (∀ x : ℝ, f (f x) = (x - 1) * f x + 2) := by
  sorry

end no_function_satisfies_conditions_l743_74378


namespace incorrect_height_calculation_l743_74309

theorem incorrect_height_calculation (n : ℕ) (initial_avg actual_avg actual_height : ℝ) :
  n = 20 ∧
  initial_avg = 175 ∧
  actual_avg = 173 ∧
  actual_height = 111 →
  ∃ incorrect_height : ℝ,
    incorrect_height = n * initial_avg - (n - 1) * actual_avg - actual_height :=
by
  sorry

end incorrect_height_calculation_l743_74309


namespace raw_materials_cost_l743_74307

def total_amount : ℝ := 93750
def machinery_cost : ℝ := 40000
def cash_percentage : ℝ := 0.20

theorem raw_materials_cost (raw_materials : ℝ) : raw_materials = 35000 :=
  by
    have cash : ℝ := total_amount * cash_percentage
    have total_equation : raw_materials + machinery_cost + cash = total_amount := by sorry
    sorry

end raw_materials_cost_l743_74307


namespace average_weight_a_b_l743_74327

/-- Given the weights of three people a, b, and c, prove that the average weight of a and b is 40 kg -/
theorem average_weight_a_b (a b c : ℝ) : 
  (a + b + c) / 3 = 45 ∧ 
  (b + c) / 2 = 43 ∧ 
  b = 31 → 
  (a + b) / 2 = 40 := by
sorry

end average_weight_a_b_l743_74327


namespace black_ball_from_red_bag_impossible_l743_74314

/-- Represents the color of a ball -/
inductive BallColor
  | Red
  | Black

/-- Represents the contents of a bag -/
structure Bag where
  balls : List BallColor

/-- Defines an impossible event -/
def impossibleEvent (p : ℝ) : Prop := p = 0

/-- Theorem: Drawing a black ball from a bag with only red balls is an impossible event -/
theorem black_ball_from_red_bag_impossible (bag : Bag) 
    (h : ∀ b ∈ bag.balls, b = BallColor.Red) : 
  impossibleEvent (Nat.card {i | bag.balls.get? i = some BallColor.Black} / bag.balls.length) := by
  sorry

end black_ball_from_red_bag_impossible_l743_74314


namespace rational_equation_solution_l743_74349

theorem rational_equation_solution (x : ℝ) : 
  -5 < x ∧ x < 3 → ((x^2 - 4*x + 5) / (2*x - 2) = 2 ↔ x = 4 - Real.sqrt 7) :=
by sorry

end rational_equation_solution_l743_74349


namespace chocolate_division_l743_74376

theorem chocolate_division (total_chocolate : ℚ) (num_piles : ℕ) (piles_to_shaina : ℕ) :
  total_chocolate = 64/7 →
  num_piles = 6 →
  piles_to_shaina = 2 →
  piles_to_shaina * (total_chocolate / num_piles) = 64/21 :=
by sorry

end chocolate_division_l743_74376


namespace all_triangles_present_l743_74306

/-- Represents a permissible triangle with angles (i/p)180°, (j/p)180°, (k/p)180° --/
structure PermissibleTriangle (p : ℕ) where
  i : ℕ
  j : ℕ
  k : ℕ
  sum_eq_p : i + j + k = p

/-- The set of all permissible triangles for a given prime p --/
def AllPermissibleTriangles (p : ℕ) : Set (PermissibleTriangle p) :=
  {t : PermissibleTriangle p | True}

/-- The set of triangles obtained after the division process stops --/
def FinalTriangleSet (p : ℕ) : Set (PermissibleTriangle p) :=
  sorry

/-- Theorem stating that the final set of triangles includes all permissible triangles --/
theorem all_triangles_present (p : ℕ) (h : Prime p) :
  FinalTriangleSet p = AllPermissibleTriangles p :=
sorry

end all_triangles_present_l743_74306


namespace f_at_5_l743_74317

/-- A function satisfying the given functional equation -/
def f : ℝ → ℝ :=
  sorry

/-- The functional equation that f satisfies for all x -/
axiom f_eq (x : ℝ) : 3 * f x + f (2 - x) = 4 * x^2 + 1

/-- The theorem to be proved -/
theorem f_at_5 : f 5 = 133 / 4 := by
  sorry

end f_at_5_l743_74317


namespace cage_cost_calculation_l743_74336

/-- The cost of the cage given the payment and change -/
def cage_cost (payment : ℚ) (change : ℚ) : ℚ :=
  payment - change

theorem cage_cost_calculation (payment : ℚ) (change : ℚ) 
  (h1 : payment = 20) 
  (h2 : change = 0.26) : 
  cage_cost payment change = 19.74 := by
  sorry

end cage_cost_calculation_l743_74336


namespace parallel_vectors_m_value_l743_74342

/-- Two vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_m_value :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (-1, m)
  parallel a b → m = -2 := by
  sorry

end parallel_vectors_m_value_l743_74342


namespace equation_solution_l743_74367

theorem equation_solution (a x : ℚ) : 
  (2 * (x - 2 * (x - a / 4)) = 3 * x) ∧ 
  ((x + a) / 9 - (1 - 3 * x) / 12 = 1) →
  a = 65 / 11 ∧ x = 13 / 11 := by
sorry

end equation_solution_l743_74367


namespace square_difference_169_168_l743_74343

theorem square_difference_169_168 : (169 : ℕ)^2 - (168 : ℕ)^2 = 337 := by
  sorry

end square_difference_169_168_l743_74343


namespace investment_proof_l743_74388

/-- Compound interest function -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

theorem investment_proof : 
  let principal : ℝ := 1000
  let rate : ℝ := 0.06
  let time : ℕ := 8
  let final_balance : ℝ := 1593.85
  compound_interest principal rate time = final_balance := by
sorry

end investment_proof_l743_74388


namespace cubic_roots_reciprocal_sum_squares_l743_74345

theorem cubic_roots_reciprocal_sum_squares (a b c d r s t : ℝ) : 
  a ≠ 0 → d ≠ 0 → 
  (∀ x, a * x^3 + b * x^2 + c * x + d = 0 ↔ x = r ∨ x = s ∨ x = t) →
  1/r^2 + 1/s^2 + 1/t^2 = (b^2 - 2*a*c) / d^2 := by
  sorry

end cubic_roots_reciprocal_sum_squares_l743_74345


namespace parabola_sum_l743_74338

/-- A parabola with coefficients p, q, and r. -/
structure Parabola where
  p : ℚ
  q : ℚ
  r : ℚ

/-- The y-coordinate of a point on the parabola given its x-coordinate. -/
def Parabola.y_coord (para : Parabola) (x : ℚ) : ℚ :=
  para.p * x^2 + para.q * x + para.r

theorem parabola_sum (para : Parabola) 
    (vertex : para.y_coord 3 = -2)
    (point : para.y_coord 6 = 5) :
    para.p + para.q + para.r = 4/3 := by
  sorry

end parabola_sum_l743_74338


namespace exists_vertex_reach_all_l743_74313

/-- A directed graph where every pair of vertices is connected by a directed edge. -/
structure CompleteDigraph (V : Type) where
  edge : V → V → Prop
  complete : ∀ (u v : V), u ≠ v → edge u v ∨ edge v u

/-- A path of length at most 2 exists between two vertices. -/
def PathLengthAtMostTwo {V : Type} (G : CompleteDigraph V) (u v : V) : Prop :=
  G.edge u v ∨ ∃ w : V, G.edge u w ∧ G.edge w v

/-- There exists a vertex from which every other vertex can be reached by a path of length at most 2. -/
theorem exists_vertex_reach_all {V : Type} (G : CompleteDigraph V) [Finite V] [Nonempty V] :
  ∃ u : V, ∀ v : V, u ≠ v → PathLengthAtMostTwo G u v := by sorry

end exists_vertex_reach_all_l743_74313


namespace chocolate_bar_cost_proof_l743_74383

/-- The original cost of one chocolate bar before discount -/
def chocolate_bar_cost : ℝ := 4.82

theorem chocolate_bar_cost_proof (
  gummy_bear_cost : ℝ)
  (chocolate_chip_cost : ℝ)
  (total_cost : ℝ)
  (gummy_bear_discount : ℝ)
  (chocolate_chip_discount : ℝ)
  (chocolate_bar_discount : ℝ)
  (h1 : gummy_bear_cost = 2)
  (h2 : chocolate_chip_cost = 5)
  (h3 : total_cost = 150)
  (h4 : gummy_bear_discount = 0.05)
  (h5 : chocolate_chip_discount = 0.10)
  (h6 : chocolate_bar_discount = 0.15)
  : chocolate_bar_cost = 4.82 := by
  sorry

#check chocolate_bar_cost_proof

end chocolate_bar_cost_proof_l743_74383


namespace cubic_root_cubes_l743_74364

/-- Given a cubic equation x^3 + ax^2 + bx + c = 0 with roots α, β, and γ,
    the cubic equation with roots α^3, β^3, and γ^3 is
    x^3 + (a^3 - 3ab + 3c)x^2 + (b^3 + 3c^2 - 3abc)x + c^3 -/
theorem cubic_root_cubes (a b c : ℝ) (α β γ : ℝ) :
  (∀ x : ℝ, x^3 + a*x^2 + b*x + c = 0 ↔ x = α ∨ x = β ∨ x = γ) →
  (∀ x : ℝ, x^3 + (a^3 - 3*a*b + 3*c)*x^2 + (b^3 + 3*c^2 - 3*a*b*c)*x + c^3 = 0
           ↔ x = α^3 ∨ x = β^3 ∨ x = γ^3) :=
by sorry

end cubic_root_cubes_l743_74364


namespace tangent_line_theorem_l743_74341

noncomputable def curve (x : ℝ) : ℝ := 2 * x^2 - x^3

def point_P : ℝ × ℝ := (0, -4)

def is_tangent_point (a : ℝ) : Prop :=
  ∃ (m : ℝ), curve a = 2 * a^2 - a^3 ∧
             m * a + (2 * a^2 - a^3) = -4 ∧
             m = 4 * a - 3 * a^2

theorem tangent_line_theorem :
  ∃ (a : ℝ), is_tangent_point a ∧ a = -1 ∧
  ∃ (m : ℝ), m = -7 ∧ 
  (∀ (x y : ℝ), y = m * x - 4 ↔ 7 * x + y + 4 = 0) :=
sorry

end tangent_line_theorem_l743_74341


namespace incircle_radius_given_tangent_circles_l743_74359

-- Define the triangle and circles
structure Triangle :=
  (A B C : Point)

structure Circle :=
  (center : Point)
  (radius : ℝ)

-- Define the property of being tangent
def is_tangent (c1 c2 : Circle) : Prop := sorry

-- Define the property of being inside a triangle
def is_inside (c : Circle) (t : Triangle) : Prop := sorry

-- Define the incircle of a triangle
def incircle (t : Triangle) : Circle := sorry

-- Main theorem
theorem incircle_radius_given_tangent_circles 
  (t : Triangle) (k : Circle) (k1 k2 k3 : Circle) :
  k = incircle t →
  is_inside k1 t ∧ is_inside k2 t ∧ is_inside k3 t →
  is_tangent k k1 ∧ is_tangent k k2 ∧ is_tangent k k3 →
  k1.radius = 1 ∧ k2.radius = 4 ∧ k3.radius = 9 →
  k.radius = 11 := by
  sorry

end incircle_radius_given_tangent_circles_l743_74359


namespace eggs_donated_to_charity_l743_74318

/-- Represents the number of eggs in a dozen --/
def dozen : ℕ := 12

/-- Represents the number of days Mortdecai collects eggs in a week --/
def collection_days : ℕ := 2

/-- Represents the number of dozen eggs Mortdecai collects per collection day --/
def eggs_collected_per_day : ℕ := 8

/-- Represents the number of dozen eggs Mortdecai delivers to the market --/
def eggs_to_market : ℕ := 3

/-- Represents the number of dozen eggs Mortdecai delivers to the mall --/
def eggs_to_mall : ℕ := 5

/-- Represents the number of dozen eggs Mortdecai uses for pie --/
def eggs_for_pie : ℕ := 4

/-- Theorem stating the number of eggs Mortdecai donates to charity --/
theorem eggs_donated_to_charity : 
  (collection_days * eggs_collected_per_day - (eggs_to_market + eggs_to_mall + eggs_for_pie)) * dozen = 48 := by
  sorry

end eggs_donated_to_charity_l743_74318


namespace no_positive_a_satisfies_inequality_l743_74300

theorem no_positive_a_satisfies_inequality :
  ∀ a : ℝ, a > 0 → ∃ x : ℝ, |Real.cos x| + |Real.cos (a * x)| ≤ Real.sin x + Real.sin (a * x) :=
by sorry

end no_positive_a_satisfies_inequality_l743_74300


namespace min_moves_to_no_moves_l743_74366

/-- Represents a chessboard configuration -/
structure ChessBoard (n : ℕ) where
  pieces : Fin n → Fin n → Bool

/-- A move on the chessboard -/
inductive Move (n : ℕ)
  | jump : Fin n → Fin n → Fin n → Fin n → Move n

/-- Predicate to check if a move is valid -/
def is_valid_move (n : ℕ) (board : ChessBoard n) (move : Move n) : Prop :=
  match move with
  | Move.jump from_x from_y to_x to_y =>
    -- Implement the logic for a valid move
    sorry

/-- Predicate to check if no further moves are possible -/
def no_moves_possible (n : ℕ) (board : ChessBoard n) : Prop :=
  ∀ (move : Move n), ¬(is_valid_move n board move)

/-- The main theorem -/
theorem min_moves_to_no_moves (n : ℕ) :
  ∀ (move_sequence : List (Move n)),
    (∃ (final_board : ChessBoard n),
      no_moves_possible n final_board ∧
      -- final_board is the result of applying move_sequence to the initial board
      sorry) →
    move_sequence.length ≥ ⌈(n^2 : ℚ) / 3⌉ :=
  sorry

end min_moves_to_no_moves_l743_74366


namespace pascal_triangle_15th_row_5th_number_l743_74325

theorem pascal_triangle_15th_row_5th_number : Nat.choose 15 4 = 1365 := by sorry

end pascal_triangle_15th_row_5th_number_l743_74325


namespace sum_of_roots_l743_74391

theorem sum_of_roots (p q r s : ℝ) : 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s →
  (∀ x : ℝ, x^2 - 12*p*x - 8*q = 0 ↔ x = r ∨ x = s) →
  (∀ x : ℝ, x^2 - 12*r*x - 8*s = 0 ↔ x = p ∨ x = q) →
  p + q + r + s = 1248 := by
sorry

end sum_of_roots_l743_74391


namespace polygon_interior_angle_sum_l743_74321

theorem polygon_interior_angle_sum (n : ℕ) (h : n * 40 = 360) : 
  (n - 2) * 180 = 1260 := by
  sorry

end polygon_interior_angle_sum_l743_74321


namespace find_divisor_l743_74323

theorem find_divisor (divisor : ℕ) : 
  (144 / divisor = 13) ∧ (144 % divisor = 1) → divisor = 11 := by
sorry

end find_divisor_l743_74323


namespace sin_cos_roots_quadratic_l743_74358

theorem sin_cos_roots_quadratic (θ : Real) (a : Real) : 
  (4 * Real.sin θ ^ 2 + 2 * a * Real.sin θ + a = 0) ∧ 
  (4 * Real.cos θ ^ 2 + 2 * a * Real.cos θ + a = 0) →
  a = 1 - Real.sqrt 5 := by
sorry

end sin_cos_roots_quadratic_l743_74358


namespace ludek_unique_stamps_l743_74351

theorem ludek_unique_stamps 
  (karel_mirek : ℕ) 
  (karel_ludek : ℕ) 
  (mirek_ludek : ℕ) 
  (karel_mirek_shared : ℕ) 
  (karel_ludek_shared : ℕ) 
  (mirek_ludek_shared : ℕ) 
  (h1 : karel_mirek = 101) 
  (h2 : karel_ludek = 115) 
  (h3 : mirek_ludek = 110) 
  (h4 : karel_mirek_shared = 5) 
  (h5 : karel_ludek_shared = 12) 
  (h6 : mirek_ludek_shared = 7) : 
  ∃ (ludek_total : ℕ), 
    ludek_total - karel_ludek_shared - mirek_ludek_shared = 43 :=
by sorry

end ludek_unique_stamps_l743_74351


namespace kristen_turtles_l743_74332

theorem kristen_turtles (trey kris kristen : ℕ) : 
  trey = 7 * kris →
  kris = kristen / 4 →
  trey = kristen + 9 →
  kristen = 12 := by
sorry

end kristen_turtles_l743_74332


namespace acute_angles_equal_positive_angles_less_than_90_l743_74353

-- Define the sets A and D
def A : Set ℝ := {θ | 0 < θ ∧ θ < Real.pi / 2}
def D : Set ℝ := {θ | 0 < θ ∧ θ < Real.pi / 2}

-- Theorem statement
theorem acute_angles_equal_positive_angles_less_than_90 : A = D := by
  sorry

end acute_angles_equal_positive_angles_less_than_90_l743_74353


namespace three_digit_sum_not_2021_l743_74329

theorem three_digit_sum_not_2021 (a b c : ℕ) : 
  a ≠ 0 → b ≠ 0 → c ≠ 0 → 
  a ≠ b → b ≠ c → a ≠ c → 
  a < 10 → b < 10 → c < 10 → 
  222 * (a + b + c) ≠ 2021 := by
  sorry

end three_digit_sum_not_2021_l743_74329


namespace paint_for_large_cube_l743_74328

-- Define the surface area of a cube
def surface_area (edge : ℝ) : ℝ := 6 * edge ^ 2

-- Define the paint required for a cube with edge 2 cm
def paint_for_2cm : ℝ := 1

-- Define the edge length of the larger cube
def large_cube_edge : ℝ := 6

-- Theorem to prove
theorem paint_for_large_cube : 
  (surface_area large_cube_edge / surface_area 2) * paint_for_2cm = 9 := by
  sorry

end paint_for_large_cube_l743_74328


namespace max_min_s_values_l743_74389

theorem max_min_s_values (x y : ℝ) (h : 4 * x^2 - 5 * x * y + 4 * y^2 = 5) :
  let s := x^2 + y^2
  (∀ a b : ℝ, 4 * a^2 - 5 * a * b + 4 * b^2 = 5 → a^2 + b^2 ≤ 10/3) ∧
  (∃ c d : ℝ, 4 * c^2 - 5 * c * d + 4 * d^2 = 5 ∧ c^2 + d^2 = 10/3) ∧
  (∀ a b : ℝ, 4 * a^2 - 5 * a * b + 4 * b^2 = 5 → a^2 + b^2 ≥ 10/13) ∧
  (∃ e f : ℝ, 4 * e^2 - 5 * e * f + 4 * f^2 = 5 ∧ e^2 + f^2 = 10/13) :=
by sorry

end max_min_s_values_l743_74389


namespace sports_club_problem_l743_74322

theorem sports_club_problem (total_members badminton_players tennis_players both : ℕ) 
  (h1 : total_members = 30)
  (h2 : badminton_players = 17)
  (h3 : tennis_players = 17)
  (h4 : both = 6) :
  total_members - (badminton_players + tennis_players - both) = 2 := by
sorry

end sports_club_problem_l743_74322


namespace hyperbola_minimum_value_l743_74301

theorem hyperbola_minimum_value (a b : ℝ) (ha : a ≥ 1) (hb : b ≥ 1) 
  (h_eccentricity : (a^2 + b^2) / a^2 = 4) :
  (∀ x y, x^2 / a^2 - y^2 / b^2 = 1) → 
  (∀ a' b', a' ≥ 1 → b' ≥ 1 → (a'^2 + b'^2) / a'^2 = 4 → 
    (b^2 + 1) / (Real.sqrt 3 * a) ≤ (b'^2 + 1) / (Real.sqrt 3 * a')) :=
by sorry

end hyperbola_minimum_value_l743_74301


namespace suitable_pairs_solution_l743_74344

def suitable_pair (a b : ℕ+) : Prop := (a + b) ∣ (a * b)

def pairs : List (ℕ+ × ℕ+) := [
  (3, 6), (4, 12), (5, 20), (6, 30), (7, 42), (8, 56),
  (9, 72), (10, 90), (11, 110), (12, 132), (13, 156), (14, 168)
]

theorem suitable_pairs_solution :
  (∀ (p : ℕ+ × ℕ+), p ∈ pairs → suitable_pair p.1 p.2) ∧
  (pairs.length = 12) ∧
  (∀ (n : ℕ+), (n ∈ pairs.map Prod.fst ∨ n ∈ pairs.map Prod.snd) →
    (pairs.map Prod.fst ++ pairs.map Prod.snd).count n = 1) ∧
  (∀ (p : ℕ+ × ℕ+), p ∉ pairs → p.1 ≤ 168 ∧ p.2 ≤ 168 → ¬suitable_pair p.1 p.2) :=
by sorry

end suitable_pairs_solution_l743_74344


namespace triangle_parallel_ratio_bounds_l743_74305

/-- Given a triangle ABC with sides a, b, c and an interior point O, 
    the ratios formed by lines through O parallel to the sides satisfy
    the given inequalities. -/
theorem triangle_parallel_ratio_bounds 
  (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (a' b' c' : ℝ)
  (ha' : a' > 0) (hb' : b' > 0) (hc' : c' > 0)
  (h_sum : a' / a + b' / b + c' / c = 3) :
  (max (a' / a) (max (b' / b) (c' / c)) ≥ 2 / 3) ∧ 
  (min (a' / a) (min (b' / b) (c' / c)) ≤ 2 / 3) := by
  sorry


end triangle_parallel_ratio_bounds_l743_74305


namespace apple_eating_duration_l743_74393

/-- The number of apples Eva needs to buy -/
def total_apples : ℕ := 14

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The number of weeks Eva should eat an apple every day -/
def weeks_to_eat_apples : ℚ := total_apples / days_per_week

theorem apple_eating_duration : weeks_to_eat_apples = 2 := by
  sorry

end apple_eating_duration_l743_74393


namespace rectangle_diagonal_l743_74304

/-- A rectangle with perimeter 72 meters and length-to-width ratio of 5:2 has a diagonal of 194/7 meters. -/
theorem rectangle_diagonal (length width : ℝ) : 
  2 * (length + width) = 72 →
  length / width = 5 / 2 →
  Real.sqrt (length^2 + width^2) = 194 / 7 := by
sorry

end rectangle_diagonal_l743_74304


namespace square_of_product_pow_two_l743_74319

theorem square_of_product_pow_two (a b : ℝ) : (a^2 * b)^2 = a^4 * b^2 := by
  sorry

end square_of_product_pow_two_l743_74319


namespace sqrt_three_minus_fraction_bound_l743_74348

theorem sqrt_three_minus_fraction_bound (n m : ℕ) (h : Real.sqrt 3 - (m : ℝ) / n > 0) :
  Real.sqrt 3 - (m : ℝ) / n > 1 / (2 * (m : ℝ) * n) :=
by sorry

end sqrt_three_minus_fraction_bound_l743_74348


namespace walk_time_to_school_l743_74316

/-- Represents Maria's travel to school -/
structure SchoolTravel where
  walkSpeed : ℝ
  skateSpeed : ℝ
  distance : ℝ

/-- The conditions of Maria's travel -/
def travelConditions (t : SchoolTravel) : Prop :=
  t.distance = 25 * t.walkSpeed + 13 * t.skateSpeed ∧
  t.distance = 11 * t.walkSpeed + 20 * t.skateSpeed

/-- The theorem to prove -/
theorem walk_time_to_school (t : SchoolTravel) 
  (h : travelConditions t) : t.distance / t.walkSpeed = 51 := by
  sorry

end walk_time_to_school_l743_74316


namespace extra_tip_amount_l743_74334

/-- The amount of a bill in dollars -/
def bill_amount : ℚ := 26

/-- The percentage of a bad tip -/
def bad_tip_percentage : ℚ := 5 / 100

/-- The percentage of a good tip -/
def good_tip_percentage : ℚ := 20 / 100

/-- Converts dollars to cents -/
def dollars_to_cents (dollars : ℚ) : ℚ := dollars * 100

/-- Calculates the tip amount in cents -/
def tip_amount (bill : ℚ) (percentage : ℚ) : ℚ :=
  dollars_to_cents (bill * percentage)

theorem extra_tip_amount :
  tip_amount bill_amount good_tip_percentage - tip_amount bill_amount bad_tip_percentage = 390 := by
  sorry

end extra_tip_amount_l743_74334


namespace ac_length_l743_74368

/-- Given a line segment AB of length 4 with a point C on it, prove that if AC is the mean
    proportional between AB and BC, then the length of AC is 2√5 - 2. -/
theorem ac_length (AB : ℝ) (C : ℝ) (hAB : AB = 4) (hC : 0 ≤ C ∧ C ≤ AB) 
  (hMean : C^2 = AB * (AB - C)) : C = 2 * Real.sqrt 5 - 2 := by
  sorry

end ac_length_l743_74368


namespace pierre_cake_consumption_l743_74377

theorem pierre_cake_consumption (total_weight : ℝ) (parts : ℕ) 
  (h1 : total_weight = 546)
  (h2 : parts = 12)
  (h3 : parts > 0) :
  let nathalie_portion := total_weight / parts
  let pierre_portion := 2.5 * nathalie_portion
  pierre_portion = 113.75 := by sorry

end pierre_cake_consumption_l743_74377


namespace last_digit_theorem_l743_74337

-- Define the property for the last digit of powers
def last_digit_property (a n k : ℕ) : Prop :=
  a^(4*n + k) % 10 = a^k % 10

-- Define the sum of specific powers
def sum_of_powers : ℕ :=
  (2^1997 + 3^1997 + 7^1997 + 9^1997) % 10

-- Theorem statement
theorem last_digit_theorem :
  (∀ (a n k : ℕ), last_digit_property a n k) ∧
  sum_of_powers = 1 := by
sorry

end last_digit_theorem_l743_74337


namespace missing_fraction_proof_l743_74387

theorem missing_fraction_proof (x : ℚ) : 
  1/2 + (-5/6) + 1/5 + 1/4 + (-9/20) + (-5/6) + x = 5/6 → x = 2 := by
  sorry

end missing_fraction_proof_l743_74387


namespace tangent_and_perpendicular_l743_74352

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + 3*x^2 - 1

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 + 6*x

-- Define the given line
def given_line (x y : ℝ) : Prop := 2*x - 6*y + 1 = 0

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := 3*x + y + 2 = 0

theorem tangent_and_perpendicular :
  ∃ (x₀ y₀ : ℝ),
    -- The point (x₀, y₀) is on the curve f
    f x₀ = y₀ ∧
    -- The tangent line passes through (x₀, y₀)
    tangent_line x₀ y₀ ∧
    -- The slope of the tangent line at (x₀, y₀) is f'(x₀)
    (3 : ℝ) = -f' x₀ ∧
    -- The given line and tangent line are perpendicular
    (2 : ℝ) * (3 : ℝ) = -(6 : ℝ) * (1 : ℝ) :=
by sorry

end tangent_and_perpendicular_l743_74352


namespace M_equals_N_l743_74357

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | ∃ y : ℝ, y = Real.sqrt x}
def N : Set ℝ := {y : ℝ | ∃ x : ℝ, y = x^2}

-- Theorem statement
theorem M_equals_N : M = N := by
  sorry

end M_equals_N_l743_74357


namespace sufficient_but_not_necessary_l743_74335

theorem sufficient_but_not_necessary (a b : ℝ) :
  (∀ a b, (a + b)/2 < Real.sqrt (a * b) → |a + b| = |a| + |b|) ∧
  (∃ a b, |a + b| = |a| + |b| ∧ (a + b)/2 ≥ Real.sqrt (a * b)) :=
sorry

end sufficient_but_not_necessary_l743_74335


namespace sqrt_x_minus_one_real_l743_74303

theorem sqrt_x_minus_one_real (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 1) ↔ x ≥ 1 := by sorry

end sqrt_x_minus_one_real_l743_74303


namespace pokemon_card_ratio_l743_74330

theorem pokemon_card_ratio (mark_cards lloyd_cards michael_cards : ℕ) : 
  mark_cards = lloyd_cards →
  mark_cards = michael_cards - 10 →
  michael_cards = 100 →
  mark_cards + lloyd_cards + michael_cards + 80 = 300 →
  mark_cards = lloyd_cards :=
by
  sorry

end pokemon_card_ratio_l743_74330


namespace hyperbola_eccentricity_condition_l743_74355

/-- Given a hyperbola mx^2 + y^2 = 1 where m < -1, if its eccentricity is exactly
    the geometric mean of the lengths of the real and imaginary axes, then m = -7 - 4√3 -/
theorem hyperbola_eccentricity_condition (m : ℝ) : 
  m < -1 →
  (∀ x y : ℝ, m * x^2 + y^2 = 1) →
  (∃ e a b : ℝ, e^2 = 4 * a * b ∧ a = 1 ∧ b^2 = -1/m) →
  m = -7 - 4 * Real.sqrt 3 :=
by sorry

end hyperbola_eccentricity_condition_l743_74355


namespace ada_original_seat_l743_74399

/-- Represents the seats in the theater --/
inductive Seat
| one
| two
| three
| four
| five
| six

/-- Represents the friends --/
inductive Friend
| ada
| bea
| ceci
| dee
| edie
| fi

/-- Represents the direction of movement --/
inductive Direction
| left
| right

/-- Defines a movement of a friend --/
structure Movement where
  friend : Friend
  distance : Nat
  direction : Direction

/-- Defines the seating arrangement --/
def SeatingArrangement := Friend → Seat

/-- Defines the set of movements --/
def Movements := List Movement

/-- Function to apply a movement to a seating arrangement --/
def applyMovement (arrangement : SeatingArrangement) (move : Movement) : SeatingArrangement :=
  sorry

/-- Function to apply all movements to a seating arrangement --/
def applyMovements (arrangement : SeatingArrangement) (moves : Movements) : SeatingArrangement :=
  sorry

/-- Theorem stating Ada's original seat --/
theorem ada_original_seat 
  (initial_arrangement : SeatingArrangement)
  (moves : Movements)
  (final_arrangement : SeatingArrangement) :
  (moves = [
    ⟨Friend.bea, 3, Direction.right⟩,
    ⟨Friend.ceci, 1, Direction.left⟩,
    ⟨Friend.dee, 1, Direction.right⟩,
    ⟨Friend.edie, 1, Direction.left⟩
  ]) →
  (final_arrangement = applyMovements initial_arrangement moves) →
  (final_arrangement Friend.ada = Seat.one ∨ final_arrangement Friend.ada = Seat.six) →
  (initial_arrangement Friend.ada = Seat.three) :=
sorry

end ada_original_seat_l743_74399


namespace complex_unit_vector_l743_74394

theorem complex_unit_vector (z : ℂ) (h : z = 3 + 4*I) : z / Complex.abs z = 3/5 + 4/5*I := by
  sorry

end complex_unit_vector_l743_74394


namespace china_mobile_charges_l743_74346

/-- Represents a mobile plan with a base fee and an excess charge per minute -/
structure MobilePlan where
  base_fee : ℝ
  excess_charge : ℝ

/-- Calculates the total call charges for a given mobile plan and excess minutes -/
def total_charges (plan : MobilePlan) (excess_minutes : ℝ) : ℝ :=
  plan.base_fee + plan.excess_charge * excess_minutes

/-- Theorem stating the relationship between total charges and excess minutes for the specific plan -/
theorem china_mobile_charges (x : ℝ) :
  let plan := MobilePlan.mk 39 0.19
  total_charges plan x = 0.19 * x + 39 := by
  sorry


end china_mobile_charges_l743_74346


namespace october_order_theorem_l743_74396

/-- Represents the order quantities for a specific month -/
structure MonthOrder where
  clawHammers : ℕ
  ballPeenHammers : ℕ
  sledgehammers : ℕ

/-- Calculates the next month's order based on the pattern -/
def nextMonthOrder (current : MonthOrder) : MonthOrder := sorry

/-- Calculates the total number of hammers in an order -/
def totalHammers (order : MonthOrder) : ℕ :=
  order.clawHammers + order.ballPeenHammers + order.sledgehammers

/-- Applies the seasonal increase to the total order -/
def applySeasonalIncrease (total : ℕ) (increase : Rat) : ℕ := sorry

/-- The order data for June, July, August, and September -/
def juneOrder : MonthOrder := ⟨3, 2, 1⟩
def julyOrder : MonthOrder := ⟨4, 3, 2⟩
def augustOrder : MonthOrder := ⟨6, 7, 3⟩
def septemberOrder : MonthOrder := ⟨9, 11, 4⟩

/-- The seasonal increase percentage -/
def seasonalIncrease : Rat := 7 / 100

theorem october_order_theorem :
  let octoberOrder := nextMonthOrder septemberOrder
  let totalBeforeIncrease := totalHammers octoberOrder
  let finalTotal := applySeasonalIncrease totalBeforeIncrease seasonalIncrease
  finalTotal = 32 := by sorry

end october_order_theorem_l743_74396


namespace girls_to_boys_ratio_l743_74381

theorem girls_to_boys_ratio (total : ℕ) (difference : ℕ) (girls boys : ℕ) : 
  total = 24 → 
  difference = 6 → 
  girls + boys = total → 
  girls = boys + difference → 
  (girls : ℚ) / (boys : ℚ) = 5 / 3 := by
  sorry

end girls_to_boys_ratio_l743_74381


namespace vertical_angles_equal_l743_74312

-- Define a point in a 2D plane
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line in a 2D plane using two points
structure Line2D where
  p1 : Point2D
  p2 : Point2D

-- Define an angle
structure Angle where
  vertex : Point2D
  ray1 : Point2D
  ray2 : Point2D

-- Define the intersection of two lines
def intersection (l1 l2 : Line2D) : Point2D :=
  sorry

-- Define vertical angles
def verticalAngles (l1 l2 : Line2D) : (Angle × Angle) :=
  sorry

-- Theorem: Vertical angles are equal
theorem vertical_angles_equal (l1 l2 : Line2D) :
  let (a1, a2) := verticalAngles l1 l2
  a1 = a2 :=
sorry

end vertical_angles_equal_l743_74312


namespace sum_longest_altitudes_eq_21_l743_74371

/-- A right triangle with sides 9, 12, and 15 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_angle : a^2 + b^2 = c^2
  side_a : a = 9
  side_b : b = 12
  side_c : c = 15

/-- The sum of the lengths of the two longest altitudes in the right triangle -/
def sum_longest_altitudes (t : RightTriangle) : ℝ :=
  t.a + t.b

theorem sum_longest_altitudes_eq_21 (t : RightTriangle) :
  sum_longest_altitudes t = 21 := by
  sorry

end sum_longest_altitudes_eq_21_l743_74371


namespace rearrange_segments_sum_l743_74373

theorem rearrange_segments_sum (a b : ℕ) : ∃ (f g : Fin 1961 → Fin 1961),
  ∀ (i : Fin 1961), ∃ (k : ℕ),
    (a + (f i : ℕ)) + (b + (g i : ℕ)) = k + i ∧ 
    k + 1961 > k + i ∧
    k + i ≥ k :=
sorry

end rearrange_segments_sum_l743_74373


namespace wire_necklace_length_l743_74374

def wire_problem (num_spools : ℕ) (spool_length : ℕ) (total_necklaces : ℕ) : ℕ :=
  (num_spools * spool_length) / total_necklaces

theorem wire_necklace_length :
  wire_problem 3 20 15 = 4 :=
by sorry

end wire_necklace_length_l743_74374


namespace inequalities_proof_l743_74356

theorem inequalities_proof (a b : ℝ) (h1 : b > a) (h2 : a * b > 0) :
  (1 / a > 1 / b) ∧ (a + b < 2 * b) := by
  sorry

end inequalities_proof_l743_74356


namespace other_factor_of_60n_l743_74395

theorem other_factor_of_60n (x : ℕ+) (h : ∀ n : ℕ, n ≥ 8 → (∃ k m : ℕ, 60 * n = x * k ∧ 60 * n = 8 * m)) :
  x ≥ 60 := by
  sorry

end other_factor_of_60n_l743_74395


namespace polynomial_remainder_l743_74380

theorem polynomial_remainder (x : ℝ) : 
  let p : ℝ → ℝ := λ x => x^5 - 2*x^3 + 4*x + 5
  p 2 = 29 := by sorry

end polynomial_remainder_l743_74380


namespace win_sector_area_l743_74392

theorem win_sector_area (r : ℝ) (p : ℝ) (A_win : ℝ) :
  r = 8 →
  p = 1 / 4 →
  A_win = p * π * r^2 →
  A_win = 16 * π :=
by sorry

end win_sector_area_l743_74392


namespace divide_multiply_result_l743_74365

theorem divide_multiply_result (x : ℝ) (h : x = 4.5) : (x / 6) * 12 = 9 := by
  sorry

end divide_multiply_result_l743_74365


namespace roofing_cost_calculation_l743_74361

theorem roofing_cost_calculation (total_needed : ℕ) (cost_per_foot : ℕ) (free_roofing : ℕ) : 
  total_needed = 300 → 
  cost_per_foot = 8 → 
  free_roofing = 250 → 
  (total_needed - free_roofing) * cost_per_foot = 400 := by
  sorry

end roofing_cost_calculation_l743_74361


namespace total_cans_eq_319_l743_74384

/-- The number of cans collected by five people given certain relationships between their collections. -/
def total_cans (solomon : ℕ) : ℕ :=
  let juwan := solomon / 3
  let levi := juwan / 2
  let gaby := (5 * solomon) / 2
  let michelle := gaby / 3
  solomon + juwan + levi + gaby + michelle

/-- Theorem stating that when Solomon collects 66 cans, the total number of cans collected by all five people is 319. -/
theorem total_cans_eq_319 : total_cans 66 = 319 := by
  sorry

end total_cans_eq_319_l743_74384


namespace cow_milk_production_l743_74379

/-- Given two groups of cows with different efficiencies, calculate the milk production of the second group based on the first group's rate. -/
theorem cow_milk_production
  (a b c d e f g : ℝ)
  (h₁ : a > 0)
  (h₂ : c > 0)
  (h₃ : f > 0) :
  let rate := b / (a * c * f)
  let second_group_production := d * rate * g * e
  second_group_production = b * d * e * g / (a * c * f) :=
by sorry

end cow_milk_production_l743_74379


namespace resultant_profit_is_four_percent_l743_74362

/-- Calculates the resultant profit percentage when an item is sold twice -/
def resultantProfitPercentage (firstProfit : Real) (secondLoss : Real) : Real :=
  let firstSalePrice := 1 + firstProfit
  let secondSalePrice := firstSalePrice * (1 - secondLoss)
  (secondSalePrice - 1) * 100

/-- Theorem: The resultant profit percentage when an item is sold with 30% profit
    and then resold with 20% loss is 4% -/
theorem resultant_profit_is_four_percent :
  resultantProfitPercentage 0.3 0.2 = 4 := by sorry

end resultant_profit_is_four_percent_l743_74362


namespace debby_candy_count_debby_candy_count_proof_l743_74350

theorem debby_candy_count : ℕ → Prop :=
  fun d : ℕ =>
    (∃ (sister_candy : ℕ) (eaten_candy : ℕ) (remaining_candy : ℕ),
      sister_candy = 42 ∧
      eaten_candy = 35 ∧
      remaining_candy = 39 ∧
      d + sister_candy - eaten_candy = remaining_candy) →
    d = 32

-- Proof
theorem debby_candy_count_proof : debby_candy_count 32 := by
  sorry

end debby_candy_count_debby_candy_count_proof_l743_74350


namespace fruit_to_grain_value_fruit_worth_in_grains_l743_74308

-- Define the exchange rates
def fruit_to_vegetable : ℚ := 3 / 4
def vegetable_to_grain : ℚ := 5

-- Theorem statement
theorem fruit_to_grain_value :
  fruit_to_vegetable * vegetable_to_grain = 15 / 4 :=
by sorry

-- Corollary to express the result as a mixed number
theorem fruit_worth_in_grains :
  fruit_to_vegetable * vegetable_to_grain = 3 + 3 / 4 :=
by sorry

end fruit_to_grain_value_fruit_worth_in_grains_l743_74308


namespace soccer_team_games_l743_74340

/-- Calculates the total number of games played by a soccer team given their win:loss:tie ratio and the number of games lost. -/
def total_games (win_ratio : ℕ) (loss_ratio : ℕ) (tie_ratio : ℕ) (games_lost : ℕ) : ℕ :=
  let games_per_part := games_lost / loss_ratio
  let total_parts := win_ratio + loss_ratio + tie_ratio
  total_parts * games_per_part

/-- Theorem stating that for a soccer team with a win:loss:tie ratio of 4:3:1 and 9 losses, the total number of games played is 24. -/
theorem soccer_team_games : 
  total_games 4 3 1 9 = 24 := by
  sorry

end soccer_team_games_l743_74340


namespace infinite_solutions_condition_l743_74369

theorem infinite_solutions_condition (b : ℝ) : 
  (∀ x : ℝ, 4 * (3 * x - b) = 3 * (4 * x + 16)) ↔ b = -12 := by
  sorry

end infinite_solutions_condition_l743_74369


namespace vector_angle_problem_l743_74324

/-- The angle between two 2D vectors -/
def angle_between (v w : ℝ × ℝ) : ℝ := sorry

/-- Converts degrees to radians -/
def deg_to_rad (deg : ℝ) : ℝ := sorry

theorem vector_angle_problem (a b : ℝ × ℝ) 
  (sum_eq : a.1 + b.1 = 2 ∧ a.2 + b.2 = -1)
  (a_eq : a = (1, 2)) :
  angle_between a b = deg_to_rad 135 := by sorry

end vector_angle_problem_l743_74324


namespace three_intersection_range_l743_74398

def f (x : ℝ) := x^3 - 3*x

theorem three_intersection_range :
  ∃ (a_min a_max : ℝ), a_min < a_max ∧
  (∀ a : ℝ, (∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
                               f x₁ = a ∧ f x₂ = a ∧ f x₃ = a) ↔
              a_min < a ∧ a < a_max) ∧
  a_min = -2 ∧ a_max = 2 :=
sorry

end three_intersection_range_l743_74398


namespace simplify_negative_a_minus_a_l743_74382

theorem simplify_negative_a_minus_a (a : ℝ) : -a - a = -2 * a := by
  sorry

end simplify_negative_a_minus_a_l743_74382


namespace marble_jar_count_l743_74315

theorem marble_jar_count :
  ∀ (total blue red green yellow : ℕ),
    2 * blue = total →
    4 * red = total →
    green = 27 →
    yellow = 14 →
    blue + red + green + yellow = total →
    total = 164 := by
  sorry

end marble_jar_count_l743_74315


namespace smallest_nine_digit_divisible_by_11_l743_74310

theorem smallest_nine_digit_divisible_by_11 : ℕ :=
  let n := 100000010
  have h1 : n ≥ 100000000 ∧ n < 1000000000 := by sorry
  have h2 : n % 11 = 0 := by sorry
  have h3 : ∀ m : ℕ, m ≥ 100000000 ∧ m < n → m % 11 ≠ 0 := by sorry
  n

end smallest_nine_digit_divisible_by_11_l743_74310


namespace negation_equivalence_l743_74354

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) ↔ (∀ x : ℝ, x^2 + 2*x + 2 > 0) :=
by sorry

end negation_equivalence_l743_74354


namespace complex_number_in_first_quadrant_l743_74331

/-- The complex number z = (3+i)/(1-i) is located in the first quadrant of the complex plane. -/
theorem complex_number_in_first_quadrant : 
  let z : ℂ := (3 + Complex.I) / (1 - Complex.I)
  (z.re > 0) ∧ (z.im > 0) :=
by
  sorry

end complex_number_in_first_quadrant_l743_74331


namespace f_seven_equals_negative_two_l743_74390

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem f_seven_equals_negative_two
  (f : ℝ → ℝ)
  (h_odd : is_odd f)
  (h_periodic : ∀ x, f (x + 4) = f x)
  (h_f_one : f 1 = 2) :
  f 7 = -2 := by
sorry

end f_seven_equals_negative_two_l743_74390


namespace average_equation_solution_l743_74326

theorem average_equation_solution (x : ℝ) : 
  (1/3 : ℝ) * ((2*x + 12) + (12*x + 4) + (4*x + 14)) = 8*x - 14 → x = 12 := by
sorry

end average_equation_solution_l743_74326


namespace subset_A_l743_74372

def A : Set ℝ := {x | x > -1}

theorem subset_A : {0} ⊆ A := by sorry

end subset_A_l743_74372


namespace omega_sum_l743_74320

theorem omega_sum (ω : ℂ) (h1 : ω^5 = 1) (h2 : ω ≠ 1) :
  ω^10 + ω^15 + ω^20 + ω^25 + ω^30 + ω^35 + ω^40 + ω^45 + ω^50 = 8 := by
  sorry

end omega_sum_l743_74320


namespace max_value_theorem_l743_74339

theorem max_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 1) 
  (h_equal_roots : a^2 = 4*(b-1)) : 
  (∃ (x : ℝ), (3*a + 2*b) / (a + b) ≤ x) ∧ 
  (∀ (y : ℝ), (3*a + 2*b) / (a + b) ≤ y → y ≥ 5/2) :=
sorry

end max_value_theorem_l743_74339
