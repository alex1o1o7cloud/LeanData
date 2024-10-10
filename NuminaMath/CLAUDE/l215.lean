import Mathlib

namespace on_time_probability_difference_l215_21507

theorem on_time_probability_difference 
  (p_plane : ℝ) 
  (p_train : ℝ) 
  (p_plane_on_time : ℝ) 
  (p_train_on_time : ℝ)
  (h_p_plane : p_plane = 0.7)
  (h_p_train : p_train = 0.3)
  (h_p_plane_on_time : p_plane_on_time = 0.8)
  (h_p_train_on_time : p_train_on_time = 0.9)
  (h_sum_prob : p_plane + p_train = 1) :
  let p_on_time := p_plane * p_plane_on_time + p_train * p_train_on_time
  let p_plane_given_on_time := (p_plane * p_plane_on_time) / p_on_time
  let p_train_given_on_time := (p_train * p_train_on_time) / p_on_time
  p_plane_given_on_time - p_train_given_on_time = 29 / 83 := by
sorry

end on_time_probability_difference_l215_21507


namespace solution_exists_l215_21545

theorem solution_exists (x : ℝ) (h1 : x > 0) (h2 : x * 3^x = 3^18) :
  ∃ k : ℕ, k = 15 ∧ k < x ∧ x < k + 1 := by
sorry

end solution_exists_l215_21545


namespace cosine_equality_l215_21598

theorem cosine_equality (n : ℤ) : 0 ≤ n ∧ n ≤ 180 → n = 38 → Real.cos (n * π / 180) = Real.cos (758 * π / 180) := by
  sorry

end cosine_equality_l215_21598


namespace bucket_pouring_l215_21558

theorem bucket_pouring (capacity_a capacity_b : ℚ) : 
  capacity_b = (1 / 2) * capacity_a →
  let initial_sand_a := (1 / 4) * capacity_a
  let initial_sand_b := (3 / 8) * capacity_b
  let final_sand_a := initial_sand_a + initial_sand_b
  final_sand_a = (7 / 16) * capacity_a :=
by sorry

end bucket_pouring_l215_21558


namespace leahs_coins_value_l215_21569

/-- Represents the number of coins of each type --/
structure CoinCount where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ

/-- Calculates the total value of coins in cents --/
def totalValue (coins : CoinCount) : ℕ :=
  coins.pennies + 5 * coins.nickels + 10 * coins.dimes

/-- Theorem stating that Leah's coins are worth 88 cents --/
theorem leahs_coins_value :
  ∃ (coins : CoinCount),
    coins.pennies + coins.nickels + coins.dimes = 20 ∧
    coins.pennies = coins.nickels ∧
    coins.pennies = coins.dimes + 4 ∧
    totalValue coins = 88 := by
  sorry

#check leahs_coins_value

end leahs_coins_value_l215_21569


namespace complement_of_A_l215_21542

def U : Set Nat := {1, 2, 3}
def A : Set Nat := {1, 3}

theorem complement_of_A : (U \ A) = {2} := by sorry

end complement_of_A_l215_21542


namespace consecutive_integers_average_l215_21544

theorem consecutive_integers_average (a : ℤ) (b : ℚ) : 
  (a > 0) →
  (b = (a + (a + 1) + (a + 2) + (a + 3) + (a + 4)) / 5) →
  ((b + (b + 10) + (b + 20)) / 3 = a + 12) :=
by sorry

end consecutive_integers_average_l215_21544


namespace pagoda_lanterns_sum_l215_21537

/-- Represents a pagoda with lanterns -/
structure Pagoda where
  layers : ℕ
  top_lanterns : ℕ
  total_lanterns : ℕ

/-- Calculates the number of lanterns on the bottom layer of the pagoda -/
def bottom_lanterns (p : Pagoda) : ℕ := p.top_lanterns * 2^(p.layers - 1)

/-- Calculates the sum of lanterns on all layers of the pagoda -/
def sum_lanterns (p : Pagoda) : ℕ := p.top_lanterns * (2^p.layers - 1)

/-- Theorem: For a 7-layer pagoda with lanterns doubling from top to bottom and
    a total of 381 lanterns, the sum of lanterns on the top and bottom layers is 195 -/
theorem pagoda_lanterns_sum :
  ∀ (p : Pagoda), p.layers = 7 → p.total_lanterns = 381 → sum_lanterns p = p.total_lanterns →
  p.top_lanterns + bottom_lanterns p = 195 :=
sorry

end pagoda_lanterns_sum_l215_21537


namespace pizza_toppings_l215_21523

theorem pizza_toppings (total_slices : ℕ) (pepperoni_slices : ℕ) (mushroom_slices : ℕ) :
  total_slices = 12 →
  pepperoni_slices = 6 →
  mushroom_slices = 10 →
  pepperoni_slices + mushroom_slices ≥ total_slices →
  ∃ (both_toppings : ℕ),
    both_toppings = pepperoni_slices + mushroom_slices - total_slices ∧
    both_toppings = 4 :=
by
  sorry

#check pizza_toppings

end pizza_toppings_l215_21523


namespace ricky_roses_l215_21555

def initial_roses : ℕ → ℕ → ℕ → ℕ → Prop
  | total, stolen, people, each =>
    total = stolen + people * each

theorem ricky_roses : initial_roses 40 4 9 4 := by
  sorry

end ricky_roses_l215_21555


namespace president_savings_l215_21518

theorem president_savings (total_funds : ℝ) (friends_percentage : ℝ) (family_percentage : ℝ) : 
  total_funds = 10000 →
  friends_percentage = 40 →
  family_percentage = 30 →
  let friends_contribution := (friends_percentage / 100) * total_funds
  let remaining_after_friends := total_funds - friends_contribution
  let family_contribution := (family_percentage / 100) * remaining_after_friends
  let president_savings := remaining_after_friends - family_contribution
  president_savings = 4200 := by
sorry

end president_savings_l215_21518


namespace unique_parallel_line_in_plane_l215_21568

/-- A structure representing a 3D space with lines and planes -/
structure Space3D where
  Point : Type
  Line : Type
  Plane : Type
  parallel_line_plane : Line → Plane → Prop
  line_in_plane : Line → Plane → Prop
  parallel_lines : Line → Line → Prop

/-- The theorem statement -/
theorem unique_parallel_line_in_plane 
  (S : Space3D) (l : S.Line) (α : S.Plane) : 
  (¬ S.parallel_line_plane l α) → 
  (¬ S.line_in_plane l α) → 
  ∃! m : S.Line, S.line_in_plane m α ∧ S.parallel_lines m l :=
sorry

end unique_parallel_line_in_plane_l215_21568


namespace coefficient_x4_sum_binomials_l215_21526

theorem coefficient_x4_sum_binomials : 
  (Finset.sum (Finset.range 3) (fun i => Nat.choose (i + 5) 4)) = 55 := by sorry

end coefficient_x4_sum_binomials_l215_21526


namespace infinite_solutions_condition_l215_21548

theorem infinite_solutions_condition (b : ℝ) : 
  (∀ x : ℝ, 4 * (3 * x - b) = 3 * (4 * x + 16)) ↔ b = -12 := by
  sorry

end infinite_solutions_condition_l215_21548


namespace regular_polygon_sides_l215_21581

theorem regular_polygon_sides (interior_angle : ℝ) : 
  interior_angle = 140 → (360 / (180 - interior_angle) : ℝ) = 9 := by
  sorry

end regular_polygon_sides_l215_21581


namespace function_inequality_l215_21532

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, (x - 1) * deriv f x ≥ 0) : f 0 + f 2 ≥ 2 * f 1 := by
  sorry

end function_inequality_l215_21532


namespace total_daisies_l215_21596

/-- Calculates the total number of daisies in Jack's flower crowns --/
theorem total_daisies (white pink red : ℕ) : 
  white = 6 ∧ 
  pink = 9 * white ∧ 
  red = 4 * pink - 3 → 
  white + pink + red = 273 := by
sorry


end total_daisies_l215_21596


namespace circle_tangency_l215_21536

/-- Two circles are tangent internally if the distance between their centers
    is equal to the absolute difference of their radii -/
def are_tangent_internally (c1_center c2_center : ℝ × ℝ) (r1 r2 : ℝ) : Prop :=
  (c1_center.1 - c2_center.1)^2 + (c1_center.2 - c2_center.2)^2 = (r1 - r2)^2

/-- The statement of the problem -/
theorem circle_tangency (m : ℝ) : 
  are_tangent_internally (m, -2) (-1, m) 3 2 ↔ m = -2 ∨ m = -1 := by
  sorry

end circle_tangency_l215_21536


namespace emily_trivia_score_l215_21500

/-- Emily's trivia game score calculation -/
theorem emily_trivia_score (first_round : ℤ) (last_round : ℤ) (final_score : ℤ) 
  (h1 : first_round = 16)
  (h2 : last_round = -48)
  (h3 : final_score = 1) :
  ∃ second_round : ℤ, first_round + second_round + last_round = final_score ∧ second_round = 33 := by
  sorry

end emily_trivia_score_l215_21500


namespace evaluate_expression_l215_21514

theorem evaluate_expression : 3000 * (3000^1500 + 3000^1500) = 2 * 3000^1501 := by
  sorry

end evaluate_expression_l215_21514


namespace factors_of_M_l215_21582

/-- The number of natural-number factors of M, where M = 2^4 · 3^3 · 7^1 -/
def num_factors (M : ℕ) : ℕ :=
  (5 : ℕ) * (4 : ℕ) * (2 : ℕ)

/-- M is defined as 2^4 · 3^3 · 7^1 -/
def M : ℕ := 2^4 * 3^3 * 7^1

theorem factors_of_M :
  num_factors M = 40 :=
by sorry

end factors_of_M_l215_21582


namespace function_composition_l215_21533

-- Define the function f
def f : ℝ → ℝ := fun x => 2 * x + 7

-- State the theorem
theorem function_composition (x : ℝ) : 
  (fun x => f (x - 1)) = (fun x => 2 * x + 5) → f (x^2) = 2 * x^2 + 7 := by
  sorry

end function_composition_l215_21533


namespace tetrahedron_division_ratio_l215_21550

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a tetrahedron -/
structure Tetrahedron where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Represents a plane -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Calculates the volume of a tetrahedron -/
def tetrahedronVolume (t : Tetrahedron) : ℝ := sorry

/-- Calculates the volume of a part of a tetrahedron cut by a plane -/
def partialTetrahedronVolume (t : Tetrahedron) (p : Plane) : ℝ := sorry

/-- Checks if a point lies on a line segment between two other points -/
def isOnLineSegment (p : Point3D) (a : Point3D) (b : Point3D) : Prop := sorry

/-- Checks if a point lies on the extension of a line segment beyond a point -/
def isOnLineExtension (p : Point3D) (a : Point3D) (b : Point3D) : Prop := sorry

/-- Theorem: The plane divides the tetrahedron in the ratio 2:33 -/
theorem tetrahedron_division_ratio (ABCD : Tetrahedron) (K M N : Point3D) (p : Plane) : 
  isOnLineSegment K ABCD.A ABCD.D ∧ 
  isOnLineExtension N ABCD.A ABCD.B ∧ 
  isOnLineExtension M ABCD.A ABCD.C ∧ 
  (ABCD.A.x - K.x) / (K.x - ABCD.D.x) = 3 ∧
  (N.x - ABCD.B.x) = (ABCD.B.x - ABCD.A.x) ∧
  (M.x - ABCD.C.x) / (ABCD.C.x - ABCD.A.x) = 1/3 ∧
  (p.a * K.x + p.b * K.y + p.c * K.z + p.d = 0) ∧
  (p.a * M.x + p.b * M.y + p.c * M.z + p.d = 0) ∧
  (p.a * N.x + p.b * N.y + p.c * N.z + p.d = 0) →
  (partialTetrahedronVolume ABCD p) / (tetrahedronVolume ABCD) = 2/35 := by
sorry

end tetrahedron_division_ratio_l215_21550


namespace integral_equation_solution_l215_21553

theorem integral_equation_solution (k : ℝ) : 
  (∫ x in (0:ℝ)..1, x - k) = (3/2 : ℝ) → k = -1 := by
  sorry

end integral_equation_solution_l215_21553


namespace hyperbola_equation_l215_21563

/-- Proves that given a hyperbola with specific conditions, its equation is x²/4 - y²/12 = 1 -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), x^2/a^2 - y^2/b^2 = 1) ∧  -- hyperbola equation
  (∃ (x y : ℝ), y = Real.sqrt 3 * x) ∧    -- asymptote condition
  (∃ (x y : ℝ), y^2 = 16*x ∧ x^2/a^2 + y^2/b^2 = 1) -- focus on directrix condition
  →
  a^2 = 4 ∧ b^2 = 12 :=
by sorry

end hyperbola_equation_l215_21563


namespace no_three_squares_l215_21584

theorem no_three_squares (x : ℤ) : ¬(∃ (a b c : ℤ), (2*x - 1 = a^2) ∧ (5*x - 1 = b^2) ∧ (13*x - 1 = c^2)) := by
  sorry

end no_three_squares_l215_21584


namespace cube_difference_l215_21540

theorem cube_difference (a b : ℝ) (h1 : a - b = 7) (h2 : a^2 + b^2 = 59) : 
  a^3 - b^3 = 448 := by
sorry

end cube_difference_l215_21540


namespace arithmetic_mean_problem_l215_21521

theorem arithmetic_mean_problem (x : ℝ) : 
  ((x + 10) + (3*x - 5) + 2*x + 18 + (2*x + 6)) / 5 = 30 → x = 15.125 := by
  sorry

end arithmetic_mean_problem_l215_21521


namespace collinear_vectors_l215_21531

/-- Given vectors a, b, and c in ℝ², prove that if a - 2b is collinear with c, then k = 1 -/
theorem collinear_vectors (a b c : ℝ × ℝ) (h : a = (Real.sqrt 3, 1)) (h' : b = (0, -1)) 
    (h'' : c = (k, Real.sqrt 3)) (h''' : ∃ t : ℝ, a - 2 • b = t • c) : k = 1 := by
  sorry

end collinear_vectors_l215_21531


namespace arithmetic_series_sum_is_1620_l215_21519

def arithmeticSeriesSum (a₁ : ℚ) (aₙ : ℚ) (d : ℚ) : ℚ :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

theorem arithmetic_series_sum_is_1620 :
  arithmeticSeriesSum 10 30 (1/4) = 1620 := by
  sorry

end arithmetic_series_sum_is_1620_l215_21519


namespace chess_tournament_games_l215_21585

/-- The number of players in the chess tournament -/
def num_players : ℕ := 7

/-- The total number of games played in the tournament -/
def total_games : ℕ := 42

/-- The number of times each player plays against each opponent -/
def games_per_pair : ℕ := 2

theorem chess_tournament_games :
  (num_players * (num_players - 1) * games_per_pair) / 2 = total_games :=
sorry

end chess_tournament_games_l215_21585


namespace luke_coin_count_l215_21502

/-- Represents the number of coins in each pile of quarters --/
def quarter_piles : List Nat := [4, 4, 6, 6, 6, 8]

/-- Represents the number of coins in each pile of dimes --/
def dime_piles : List Nat := [3, 5, 2, 2]

/-- Represents the number of coins in each pile of nickels --/
def nickel_piles : List Nat := [5, 5, 5, 7, 7, 10]

/-- Represents the number of coins in each pile of pennies --/
def penny_piles : List Nat := [12, 8, 20]

/-- Represents the number of coins in each pile of half dollars --/
def half_dollar_piles : List Nat := [2, 4]

/-- The total number of coins Luke has --/
def total_coins : Nat := quarter_piles.sum + dime_piles.sum + nickel_piles.sum + 
                         penny_piles.sum + half_dollar_piles.sum

theorem luke_coin_count : total_coins = 131 := by
  sorry

end luke_coin_count_l215_21502


namespace range_f_a2_values_of_a_min_3_l215_21511

-- Define the function f(x) with parameter a
def f (a : ℝ) (x : ℝ) : ℝ := 4 * x^2 - 4 * a * x + (a^2 - 2 * a + 2)

-- Part 1: Range of f(x) when a = 2 in [1, 2]
theorem range_f_a2 :
  ∀ y ∈ Set.Icc (-2) 2, ∃ x ∈ Set.Icc 1 2, f 2 x = y :=
sorry

-- Part 2: Values of a when minimum of f(x) in [0, 2] is 3
theorem values_of_a_min_3 :
  (∀ x ∈ Set.Icc 0 2, f a x ≥ 3) ∧ (∃ x ∈ Set.Icc 0 2, f a x = 3) →
  a = 1 - Real.sqrt 2 ∨ a = 5 + Real.sqrt 10 :=
sorry

end range_f_a2_values_of_a_min_3_l215_21511


namespace fixed_point_of_parabola_l215_21541

theorem fixed_point_of_parabola (s : ℝ) :
  let f : ℝ → ℝ := λ x ↦ 4 * x^2 + s * x - 3 * s
  f 3 = 36 := by sorry

end fixed_point_of_parabola_l215_21541


namespace box_side_face_area_l215_21561

theorem box_side_face_area (L W H : ℝ) 
  (h1 : W * H = (1/2) * (L * W))
  (h2 : L * W = 1.5 * (H * L))
  (h3 : L * W * H = 648) :
  H * L = 72 := by
  sorry

end box_side_face_area_l215_21561


namespace min_value_of_expression_l215_21547

theorem min_value_of_expression (x y : ℝ) (h : 2 * x - y = 4) :
  ∃ (m : ℝ), m = 8 ∧ ∀ (a b : ℝ), 2 * a - b = 4 → 4^a + (1/2)^b ≥ m :=
sorry

end min_value_of_expression_l215_21547


namespace remainder_theorem_l215_21501

theorem remainder_theorem : (1225^3 * 1227^4 * 1229^5) % 36 = 9 := by
  sorry

end remainder_theorem_l215_21501


namespace hyperbola_asymptotes_l215_21578

def hyperbola_equation (x y k : ℝ) : Prop :=
  x^2 / (k - 2016) + y^2 / (k - 2018) = 1

def asymptote_equation (x y : ℝ) : Prop :=
  x + y = 0 ∨ x - y = 0

theorem hyperbola_asymptotes (k : ℤ) :
  (∃ x y : ℝ, hyperbola_equation x y (k : ℝ)) →
  (∀ x y : ℝ, hyperbola_equation x y (k : ℝ) → asymptote_equation x y) :=
sorry

end hyperbola_asymptotes_l215_21578


namespace jezebel_flower_cost_l215_21508

/-- Calculates the total cost of flowers with discount and tax --/
def total_cost (red_roses white_lilies sunflowers blue_orchids : ℕ)
  (rose_price lily_price sunflower_price orchid_price : ℚ)
  (discount_rate tax_rate : ℚ) : ℚ :=
  let subtotal := red_roses * rose_price + white_lilies * lily_price +
                  sunflowers * sunflower_price + blue_orchids * orchid_price
  let discount := discount_rate * (red_roses * rose_price + white_lilies * lily_price)
  let after_discount := subtotal - discount
  let tax := tax_rate * after_discount
  after_discount + tax

/-- Theorem stating the total cost for Jezebel's flower purchase --/
theorem jezebel_flower_cost :
  total_cost 24 14 8 10 1.5 2.75 3 4.25 0.1 0.07 = 142.9 := by
  sorry

end jezebel_flower_cost_l215_21508


namespace odd_function_property_l215_21505

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_property (f : ℝ → ℝ) 
  (h_odd : is_odd_function f)
  (h_prop : ∀ x, f (-x + 1) = f (x + 1))
  (h_val : f (-1) = 1) :
  f 2017 = -1 := by
sorry

end odd_function_property_l215_21505


namespace circle_equation_and_intersection_range_l215_21595

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 2*Real.sqrt 3*y = 0

-- Define the line l
def line_l (t x y : ℝ) : Prop :=
  x = -1 - (Real.sqrt 3 / 2) * t ∧ y = Real.sqrt 3 + (1 / 2) * t

-- Define the intersection point P
def intersection_point (x y : ℝ) : Prop :=
  ∃ t : ℝ, line_l t x y ∧ circle_C x y

theorem circle_equation_and_intersection_range :
  (∀ ρ θ : ℝ, ρ = 4 * Real.sin (θ - Real.pi / 6) → 
    ∃ x y : ℝ, ρ * Real.cos θ = x ∧ ρ * Real.sin θ = y ∧ circle_C x y) ∧
  (∀ x y : ℝ, intersection_point x y → 
    -2 ≤ Real.sqrt 3 * x + y ∧ Real.sqrt 3 * x + y ≤ 2) := by sorry

end circle_equation_and_intersection_range_l215_21595


namespace shelf_filling_l215_21530

theorem shelf_filling (P Q T N K : ℕ) (hP : P > 0) (hQ : Q > 0) (hT : T > 0) (hN : N > 0) (hK : K > 0)
  (hUnique : P ≠ Q ∧ P ≠ T ∧ P ≠ N ∧ P ≠ K ∧ Q ≠ T ∧ Q ≠ N ∧ Q ≠ K ∧ T ≠ N ∧ T ≠ K ∧ N ≠ K)
  (hThicker : ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ y > x ∧ P * x + Q * y = T * x + N * y ∧ K * x = P * x + Q * y) :
  K = (P * K - T * Q) / (N - Q) :=
by sorry

end shelf_filling_l215_21530


namespace set_problem_l215_21587

def U : Set ℕ := {x | x ≤ 10}

theorem set_problem (A B : Set ℕ) 
  (h1 : A ⊆ U)
  (h2 : B ⊆ U)
  (h3 : A ∩ B = {4,5,6})
  (h4 : (U \ B) ∩ A = {2,3})
  (h5 : (U \ A) ∩ (U \ B) = {7,8}) :
  A = {2,3,4,5,6} ∧ B = {4,5,6,9,10} := by
  sorry

end set_problem_l215_21587


namespace at_least_one_meets_standard_l215_21559

theorem at_least_one_meets_standard (pA pB pC : ℝ) 
  (hA : pA = 0.8) (hB : pB = 0.6) (hC : pC = 0.5) :
  1 - (1 - pA) * (1 - pB) * (1 - pC) = 0.96 := by
  sorry

end at_least_one_meets_standard_l215_21559


namespace parabola_sum_a_c_l215_21504

/-- A parabola that intersects the x-axis at x = -1 -/
structure Parabola where
  a : ℝ
  c : ℝ
  intersect_at_neg_one : a * (-1)^2 + (-1) + c = 0

/-- The sum of a and c for a parabola intersecting the x-axis at x = -1 is 1 -/
theorem parabola_sum_a_c (p : Parabola) : p.a + p.c = 1 := by
  sorry

end parabola_sum_a_c_l215_21504


namespace cone_surface_area_minimization_l215_21552

/-- 
Given a right circular cone with fixed volume V, base radius R, and height H,
prove that H/R = 3 when the total surface area is minimized.
-/
theorem cone_surface_area_minimization (V : ℝ) (V_pos : V > 0) :
  ∃ (R H : ℝ), R > 0 ∧ H > 0 ∧
  (∀ (r h : ℝ), r > 0 → h > 0 → (1/3) * Real.pi * r^2 * h = V →
    R^2 * (Real.pi * R + Real.pi * Real.sqrt (R^2 + H^2)) ≤ 
    r^2 * (Real.pi * r + Real.pi * Real.sqrt (r^2 + h^2))) ∧
  H / R = 3 := by
  sorry


end cone_surface_area_minimization_l215_21552


namespace coin_flip_probability_difference_l215_21575

-- Define a fair coin
def fair_coin_prob : ℚ := 1/2

-- Define the number of flips
def num_flips : ℕ := 4

-- Define the probability of exactly 3 heads in 4 flips
def prob_3_heads : ℚ := Nat.choose num_flips 3 * fair_coin_prob^3 * (1 - fair_coin_prob)^(num_flips - 3)

-- Define the probability of 4 heads in 4 flips
def prob_4_heads : ℚ := fair_coin_prob^num_flips

-- Theorem statement
theorem coin_flip_probability_difference : 
  |prob_3_heads - prob_4_heads| = 7/16 := by sorry

end coin_flip_probability_difference_l215_21575


namespace prime_divisibility_pairs_l215_21570

theorem prime_divisibility_pairs (n p : ℕ) : 
  p.Prime → 
  n ≤ 2 * p → 
  (p - 1)^n + 1 ∣ n^(p - 1) → 
  ((n = 1 ∧ p.Prime) ∨ (n = 2 ∧ p = 2) ∨ (n = 3 ∧ p = 3)) := by
  sorry

end prime_divisibility_pairs_l215_21570


namespace line_segment_does_not_intersect_staircase_l215_21539

/-- Represents a step in the staircase -/
structure Step where
  width : Nat
  height : Nat

/-- Represents the staircase -/
def Staircase : List Step := List.range 2019 |>.map (fun i => { width := i + 1, height := 1 })

/-- The line segment from (0,0) to (2019,2019) -/
def LineSegment : Set (ℝ × ℝ) :=
  {p | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ p = (2019 * t, 2019 * t)}

/-- Checks if a point is on a step -/
def onStep (p : ℝ × ℝ) (s : Step) : Prop :=
  (s.width - 1 : ℝ) ≤ p.1 ∧ p.1 < s.width ∧
  (s.height - 1 : ℝ) ≤ p.2 ∧ p.2 < s.height

theorem line_segment_does_not_intersect_staircase :
  ∀ p ∈ LineSegment, ∀ s ∈ Staircase, ¬ onStep p s := by
  sorry

end line_segment_does_not_intersect_staircase_l215_21539


namespace additional_stickers_needed_l215_21554

def current_stickers : ℕ := 35
def row_size : ℕ := 8

theorem additional_stickers_needed :
  let next_multiple := (current_stickers + row_size - 1) / row_size * row_size
  next_multiple - current_stickers = 5 := by sorry

end additional_stickers_needed_l215_21554


namespace sum_of_divisors_360_l215_21579

-- Define the sum of positive divisors function
def sumOfDivisors (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem sum_of_divisors_360 (i j : ℕ) :
  sumOfDivisors (2^i * 3^j) = 360 → i = 3 ∧ j = 3 := by
  sorry

end sum_of_divisors_360_l215_21579


namespace kris_bullying_instances_l215_21594

/-- The number of days Kris is suspended for each bullying instance -/
def suspension_days_per_instance : ℕ := 3

/-- The total number of fingers and toes a typical person has -/
def typical_person_digits : ℕ := 20

/-- The total number of days Kris has been suspended -/
def total_suspension_days : ℕ := 3 * typical_person_digits

/-- The number of bullying instances Kris is responsible for -/
def bullying_instances : ℕ := total_suspension_days / suspension_days_per_instance

theorem kris_bullying_instances : bullying_instances = 20 := by
  sorry

end kris_bullying_instances_l215_21594


namespace cherry_pies_count_l215_21506

/-- Given a total number of pies and a ratio for distribution among three types,
    calculate the number of pies of the third type. -/
def calculate_third_type_pies (total : ℕ) (ratio1 ratio2 ratio3 : ℕ) : ℕ :=
  let ratio_sum := ratio1 + ratio2 + ratio3
  let pies_per_part := total / ratio_sum
  ratio3 * pies_per_part

/-- Theorem stating that given 40 pies distributed in the ratio 2:5:3,
    the number of cherry pies (third type) is 12. -/
theorem cherry_pies_count :
  calculate_third_type_pies 40 2 5 3 = 12 := by
  sorry

end cherry_pies_count_l215_21506


namespace complex_equation_sum_l215_21556

theorem complex_equation_sum (x y : ℝ) :
  (x + 2 * Complex.I = y - 1 + y * Complex.I) → x + y = 3 := by
  sorry

end complex_equation_sum_l215_21556


namespace lopez_seating_theorem_l215_21583

/-- Represents the number of family members -/
def family_members : ℕ := 5

/-- Represents the number of front seats in the car -/
def front_seats : ℕ := 2

/-- Represents the number of back seats in the car -/
def back_seats : ℕ := 3

/-- Represents the number of possible drivers (Mr. or Mrs. Lopez) -/
def possible_drivers : ℕ := 2

/-- Calculates the number of possible seating arrangements for the Lopez family -/
def seating_arrangements : ℕ :=
  possible_drivers * (family_members - 1) * Nat.factorial (family_members - 2)

theorem lopez_seating_theorem :
  seating_arrangements = 48 :=
sorry

end lopez_seating_theorem_l215_21583


namespace special_function_a_range_l215_21517

/-- A function satisfying the given properties -/
structure SpecialFunction where
  f : ℝ → ℝ
  even : ∀ x, f (-x) = f x
  increasing_nonneg : ∀ x₁ x₂, 0 ≤ x₁ ∧ 0 ≤ x₂ ∧ x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) > 0

/-- The theorem statement -/
theorem special_function_a_range (f : SpecialFunction) :
  {a : ℝ | ∀ x ∈ Set.Icc (1/2 : ℝ) 1, f.f (a * x + 1) ≤ f.f (x - 2)} = Set.Icc (-2 : ℝ) 0 := by
  sorry

end special_function_a_range_l215_21517


namespace industrial_lubricants_allocation_l215_21577

/-- Represents the budget allocation for Megatech Corporation --/
structure BudgetAllocation where
  microphotonics : ℝ
  home_electronics : ℝ
  food_additives : ℝ
  genetically_modified_microorganisms : ℝ
  basic_astrophysics_degrees : ℝ
  total_degrees : ℝ

/-- Theorem stating that the industrial lubricants allocation is 8% --/
theorem industrial_lubricants_allocation
  (budget : BudgetAllocation)
  (h1 : budget.microphotonics = 12)
  (h2 : budget.home_electronics = 24)
  (h3 : budget.food_additives = 15)
  (h4 : budget.genetically_modified_microorganisms = 29)
  (h5 : budget.basic_astrophysics_degrees = 43.2)
  (h6 : budget.total_degrees = 360) :
  100 - (budget.microphotonics + budget.home_electronics + budget.food_additives +
    budget.genetically_modified_microorganisms + budget.basic_astrophysics_degrees *
    100 / budget.total_degrees) = 8 := by
  sorry


end industrial_lubricants_allocation_l215_21577


namespace unique_three_digit_numbers_l215_21576

/-- The number of available digits -/
def n : ℕ := 5

/-- The number of digits to be used in each number -/
def r : ℕ := 3

/-- The number of unique three-digit numbers that can be formed without repetition -/
def uniqueNumbers : ℕ := n.choose r * r.factorial

theorem unique_three_digit_numbers :
  uniqueNumbers = 60 := by sorry

end unique_three_digit_numbers_l215_21576


namespace uncovered_fraction_of_plates_l215_21543

/-- The fraction of a circular plate with diameter 12 inches that is not covered
    by a smaller circular plate with diameter 10 inches placed on top of it is 11/36. -/
theorem uncovered_fraction_of_plates (small_diameter large_diameter : ℝ) 
  (h_small : small_diameter = 10)
  (h_large : large_diameter = 12) :
  (large_diameter^2 - small_diameter^2) / large_diameter^2 = 11 / 36 := by
  sorry

end uncovered_fraction_of_plates_l215_21543


namespace louisa_travel_problem_l215_21527

/-- Louisa's vacation travel problem -/
theorem louisa_travel_problem (speed : ℝ) (second_day_distance : ℝ) (time_difference : ℝ) :
  speed = 60 →
  second_day_distance = 420 →
  time_difference = 3 →
  ∃ (first_day_distance : ℝ),
    first_day_distance = speed * (second_day_distance / speed - time_difference) ∧
    first_day_distance = 240 :=
by sorry

end louisa_travel_problem_l215_21527


namespace product_expansion_sum_l215_21513

theorem product_expansion_sum (a b c d : ℤ) : 
  (∀ x, (5 * x^2 - 8 * x + 3) * (9 - 3 * x) = a * x^3 + b * x^2 + c * x + d) →
  10 * a + 5 * b + 2 * c + d = 60 := by
  sorry

end product_expansion_sum_l215_21513


namespace frog_final_position_l215_21538

def frog_jumps (n : ℕ) : ℕ := n * (n + 1) / 2

theorem frog_final_position :
  ∀ (total_positions : ℕ) (num_jumps : ℕ),
    total_positions = 6 →
    num_jumps = 20 →
    frog_jumps num_jumps % total_positions = 1 := by
  sorry

end frog_final_position_l215_21538


namespace intersection_length_l215_21590

/-- The length of the circular intersection between a sphere and a plane -/
theorem intersection_length (x y z : ℝ) : 
  x + y + z = 8 → 
  x * y + y * z + x * z = 14 → 
  (2 * Real.pi : ℝ) * (2 * Real.sqrt (11 / 3)) = 4 * Real.pi * Real.sqrt (11 / 3) := by
  sorry

#check intersection_length

end intersection_length_l215_21590


namespace ellen_smoothie_strawberries_l215_21586

/-- The amount of strawberries used in Ellen's smoothie recipe. -/
def strawberries : ℝ := 0.5 - (0.1 + 0.2)

/-- Theorem stating that Ellen used 0.2 cups of strawberries in her smoothie. -/
theorem ellen_smoothie_strawberries :
  strawberries = 0.2 := by sorry

end ellen_smoothie_strawberries_l215_21586


namespace not_all_mages_are_wizards_l215_21567

-- Define the universe of discourse
variable {U : Type}

-- Define predicates for being a mage, sorcerer, and wizard
variable (Mage Sorcerer Wizard : U → Prop)

-- State the theorem
theorem not_all_mages_are_wizards :
  (∃ x, Mage x ∧ ¬Sorcerer x) →
  (∀ x, Mage x ∧ Wizard x → Sorcerer x) →
  ∃ x, Mage x ∧ ¬Wizard x :=
by sorry

end not_all_mages_are_wizards_l215_21567


namespace total_students_correct_l215_21549

/-- The total number of students who appeared for the examination -/
def total_students : ℕ := 840

/-- The percentage of students who passed the examination -/
def pass_percentage : ℚ := 35 / 100

/-- The number of students who failed the examination -/
def failed_students : ℕ := 546

/-- Theorem stating that the total number of students is correct given the conditions -/
theorem total_students_correct : 
  (1 - pass_percentage) * total_students = failed_students := by sorry

end total_students_correct_l215_21549


namespace range_of_m_l215_21535

/-- Represents an ellipse with foci on the x-axis -/
def is_ellipse_x_axis (m : ℝ) : Prop :=
  ∃ x y : ℝ, x^2 / m + y^2 / (6 - m) = 1 ∧ m > 6 - m ∧ m > 0

/-- Represents a hyperbola with given eccentricity range -/
def is_hyperbola_with_eccentricity (m : ℝ) : Prop :=
  ∃ x y e : ℝ, y^2 / 5 - x^2 / m = 1 ∧ 
    e^2 = 1 + m / 5 ∧ 
    Real.sqrt 6 / 2 < e ∧ e < Real.sqrt 2 ∧
    m > 0

/-- The main theorem stating the range of m -/
theorem range_of_m (m : ℝ) : 
  (is_ellipse_x_axis m ∨ is_hyperbola_with_eccentricity m) ∧ 
  ¬(is_ellipse_x_axis m ∧ is_hyperbola_with_eccentricity m) →
  (5/2 < m ∧ m ≤ 3) ∨ (5 ≤ m ∧ m < 6) :=
sorry

end range_of_m_l215_21535


namespace equal_coverings_l215_21516

/-- Represents a 1993 x 1993 grid -/
def Grid := Fin 1993 × Fin 1993

/-- Represents a 1 x 2 rectangle -/
def Rectangle := Set (Fin 1993 × Fin 1993)

/-- Predicate to check if two squares are on the same edge of the grid -/
def on_same_edge (a b : Grid) : Prop :=
  (a.1 = b.1 ∧ (a.2 = 0 ∨ a.2 = 1992)) ∨
  (a.2 = b.2 ∧ (a.1 = 0 ∨ a.1 = 1992))

/-- Predicate to check if there's an odd number of squares between two squares -/
def odd_squares_between (a b : Grid) : Prop :=
  ∃ n : Nat, n % 2 = 1 ∧
  ((a.1 = b.1 ∧ abs (a.2 - b.2) = n + 1) ∨
   (a.2 = b.2 ∧ abs (a.1 - b.1) = n + 1))

/-- Type representing a covering of the grid with 1 x 2 rectangles -/
def Covering := Set Rectangle

/-- Predicate to check if a covering is valid (covers the entire grid except one square) -/
def valid_covering (c : Covering) (uncovered : Grid) : Prop := sorry

/-- The number of valid coverings that leave a given square uncovered -/
def num_coverings (uncovered : Grid) : Nat := sorry

theorem equal_coverings (A B : Grid)
  (h1 : on_same_edge A B)
  (h2 : odd_squares_between A B) :
  num_coverings A = num_coverings B := by sorry

end equal_coverings_l215_21516


namespace binary_1101001_plus_14_equals_119_l215_21512

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + (if b then 2^i else 0)) 0

/-- The binary representation of 1101001₂ -/
def binary_1101001 : List Bool := [true, false, false, true, false, true, true]

theorem binary_1101001_plus_14_equals_119 :
  binary_to_decimal binary_1101001 + 14 = 119 := by
  sorry

end binary_1101001_plus_14_equals_119_l215_21512


namespace cookie_price_is_three_l215_21591

/-- The price of each cookie in Zane's purchase --/
def cookie_price : ℚ := 3

/-- The total number of items (Oreos and cookies) --/
def total_items : ℕ := 65

/-- The ratio of Oreos to cookies --/
def oreo_cookie_ratio : ℚ := 4 / 9

/-- The price of each Oreo --/
def oreo_price : ℚ := 2

/-- The difference in total spent on cookies vs Oreos --/
def cookie_oreo_diff : ℚ := 95

theorem cookie_price_is_three :
  let num_cookies : ℚ := total_items / (1 + oreo_cookie_ratio)
  let num_oreos : ℚ := total_items - num_cookies
  let total_oreo_cost : ℚ := num_oreos * oreo_price
  let total_cookie_cost : ℚ := total_oreo_cost + cookie_oreo_diff
  cookie_price = total_cookie_cost / num_cookies :=
by sorry

end cookie_price_is_three_l215_21591


namespace square_area_calculation_l215_21592

theorem square_area_calculation (s r l : ℝ) : 
  l = (2/5) * r →
  r = s →
  l * 10 = 200 →
  s^2 = 2500 := by sorry

end square_area_calculation_l215_21592


namespace expression_simplification_l215_21571

/-- For x in the open interval (0, 1], the given expression simplifies to ∛((1-x)/(3x)) -/
theorem expression_simplification (x : ℝ) (h : 0 < x ∧ x ≤ 1) :
  1.37 * Real.rpow ((2 * x^2) / (9 + 18*x + 9*x^2)) (1/3) *
  Real.sqrt (((1 + x) * Real.rpow (1 - x) (1/3)) / x) *
  Real.rpow ((3 * Real.sqrt (1 - x^2)) / (2 * x * Real.sqrt x)) (1/3) =
  Real.rpow ((1 - x) / (3 * x)) (1/3) := by
sorry

end expression_simplification_l215_21571


namespace union_of_M_and_N_l215_21529

def M : Set ℕ := {0, 1, 2}

def N : Set ℕ := {y | ∃ x ∈ M, y = x^2}

theorem union_of_M_and_N : M ∪ N = {0, 1, 2, 4} := by sorry

end union_of_M_and_N_l215_21529


namespace sqrt_sum_comparison_l215_21562

theorem sqrt_sum_comparison : Real.sqrt 3 + Real.sqrt 7 < 2 * Real.sqrt 5 := by
  sorry

end sqrt_sum_comparison_l215_21562


namespace complex_product_QED_l215_21522

theorem complex_product_QED (Q E D : ℂ) : 
  Q = 7 + 3 * Complex.I → 
  E = 2 * Complex.I → 
  D = 7 - 3 * Complex.I → 
  Q * E * D = 116 * Complex.I :=
by
  sorry

end complex_product_QED_l215_21522


namespace aquarium_visitors_l215_21599

theorem aquarium_visitors (total : ℕ) (ill_percentage : ℚ) : 
  total = 500 → ill_percentage = 40 / 100 → 
  (total : ℚ) * (1 - ill_percentage) = 300 := by
  sorry

end aquarium_visitors_l215_21599


namespace remaining_staff_count_l215_21525

/-- Calculates the remaining staff in a cafe after some leave --/
theorem remaining_staff_count 
  (initial_chefs initial_waiters initial_busboys initial_hostesses : ℕ)
  (leaving_chefs leaving_waiters leaving_busboys leaving_hostesses : ℕ)
  (h1 : initial_chefs = 16)
  (h2 : initial_waiters = 16)
  (h3 : initial_busboys = 10)
  (h4 : initial_hostesses = 5)
  (h5 : leaving_chefs = 6)
  (h6 : leaving_waiters = 3)
  (h7 : leaving_busboys = 4)
  (h8 : leaving_hostesses = 2) :
  (initial_chefs - leaving_chefs) + (initial_waiters - leaving_waiters) + 
  (initial_busboys - leaving_busboys) + (initial_hostesses - leaving_hostesses) = 32 := by
  sorry

end remaining_staff_count_l215_21525


namespace abs_difference_sqrt_square_l215_21560

theorem abs_difference_sqrt_square (x α : ℝ) (h : x < α) :
  |x - Real.sqrt ((x - α)^2)| = α - 2*x := by
  sorry

end abs_difference_sqrt_square_l215_21560


namespace parallel_lines_planes_l215_21589

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation for lines and planes
variable (parallel : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_plane : Plane → Plane → Prop)

-- Define the "not contained in" relation for a line and a plane
variable (not_contained_in : Line → Plane → Prop)

-- State the theorem
theorem parallel_lines_planes 
  (l m : Line) 
  (α β : Plane) 
  (h_distinct_lines : l ≠ m)
  (h_distinct_planes : α ≠ β)
  (h_alpha_beta_parallel : parallel_plane α β)
  (h_l_alpha_parallel : parallel_line_plane l α)
  (h_l_m_parallel : parallel l m)
  (h_m_not_in_beta : not_contained_in m β) :
  parallel_line_plane m β :=
sorry

end parallel_lines_planes_l215_21589


namespace side_face_area_l215_21564

/-- A rectangular box with specific properties -/
structure Box where
  length : ℕ
  width : ℕ
  height : ℕ
  front_face_half_top : width * height = (length * width) / 2
  top_face_one_half_side : length * width = (3 * length * height) / 2
  volume : length * width * height = 5184
  perimeter_ratio : 2 * (length + height) = (12 * (length + width)) / 10

/-- The area of the side face of a box with the given properties is 384 square units -/
theorem side_face_area (b : Box) : b.length * b.height = 384 := by
  sorry

end side_face_area_l215_21564


namespace not_prime_sum_l215_21574

theorem not_prime_sum (a b c d : ℕ+) 
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) 
  (h_int : ∃ (n : ℤ), (a / (a + b) : ℚ) + (b / (b + c) : ℚ) + (c / (c + d) : ℚ) + (d / (d + a) : ℚ) = n) : 
  ¬ Nat.Prime (a + b + c + d) := by
sorry

end not_prime_sum_l215_21574


namespace regular_polygon_interior_angles_divisible_by_nine_l215_21597

theorem regular_polygon_interior_angles_divisible_by_nine :
  (∃ (S : Finset ℕ), S.card = 5 ∧
    (∀ n ∈ S, 3 ≤ n ∧ n ≤ 15 ∧ (180 - 360 / n) % 9 = 0) ∧
    (∀ n, 3 ≤ n → n ≤ 15 → (180 - 360 / n) % 9 = 0 → n ∈ S)) :=
by sorry

end regular_polygon_interior_angles_divisible_by_nine_l215_21597


namespace function_value_at_two_l215_21593

theorem function_value_at_two (f : ℝ → ℝ) (h : ∀ x, f (x + 1) = x^2 - 2*x) : f 2 = -1 := by
  sorry

end function_value_at_two_l215_21593


namespace bernardo_winning_number_l215_21566

theorem bernardo_winning_number : ∃ N : ℕ, 
  (N ≤ 1999) ∧ 
  (8 * N + 600 < 2000) ∧ 
  (8 * N + 700 ≥ 2000) ∧ 
  (∀ M : ℕ, M < N → 
    (M ≤ 1999 → 8 * M + 700 < 2000) ∨ 
    (8 * M + 600 ≥ 2000)) := by
  sorry

#eval Nat.find bernardo_winning_number

end bernardo_winning_number_l215_21566


namespace minimum_bailing_rate_l215_21557

/-- Proves that the minimum bailing rate is 13 gallons per minute -/
theorem minimum_bailing_rate
  (distance_to_shore : ℝ)
  (water_intake_rate : ℝ)
  (boat_capacity : ℝ)
  (rowing_speed : ℝ)
  (h1 : distance_to_shore = 3)
  (h2 : water_intake_rate = 15)
  (h3 : boat_capacity = 60)
  (h4 : rowing_speed = 6)
  : ∃ (bailing_rate : ℝ), 
    bailing_rate = 13 ∧ 
    bailing_rate * (distance_to_shore / rowing_speed * 60) ≥ 
    water_intake_rate * (distance_to_shore / rowing_speed * 60) - boat_capacity ∧
    ∀ (r : ℝ), r < bailing_rate → 
      r * (distance_to_shore / rowing_speed * 60) < 
      water_intake_rate * (distance_to_shore / rowing_speed * 60) - boat_capacity :=
by sorry


end minimum_bailing_rate_l215_21557


namespace planes_lines_parallelism_l215_21588

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the necessary relations
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (line_parallel : Line → Line → Prop)

-- State the theorem
theorem planes_lines_parallelism 
  (α β : Plane) (m n : Line) 
  (h_not_coincident : α ≠ β)
  (h_different_lines : m ≠ n)
  (h_parallel_planes : parallel α β)
  (h_n_perp_α : perpendicular n α)
  (h_m_perp_β : perpendicular m β) :
  line_parallel m n :=
sorry

end planes_lines_parallelism_l215_21588


namespace range_of_f_l215_21509

-- Define the function f
def f (x : ℝ) : ℝ := -x^2 + 4*x

-- Define the domain of x
def domain : Set ℝ := Set.Icc 0 2

-- Theorem statement
theorem range_of_f :
  Set.range (fun x => f x) = Set.Icc 0 4 := by sorry

end range_of_f_l215_21509


namespace trigonometric_expression_equality_l215_21528

theorem trigonometric_expression_equality : 
  (Real.sin (330 * π / 180) * Real.tan (-13 * π / 3)) / 
  (Real.cos (-19 * π / 6) * Real.cos (690 * π / 180)) = 
  -2 * Real.sqrt 3 / 3 := by sorry

end trigonometric_expression_equality_l215_21528


namespace alex_grocery_delivery_l215_21503

theorem alex_grocery_delivery (saved : ℝ) (car_cost : ℝ) (trip_charge : ℝ) (grocery_fee_percent : ℝ) (num_trips : ℕ) 
  (h1 : saved = 14500)
  (h2 : car_cost = 14600)
  (h3 : trip_charge = 1.5)
  (h4 : grocery_fee_percent = 0.05)
  (h5 : num_trips = 40) :
  ∃ (grocery_value : ℝ), 
    grocery_value * grocery_fee_percent = car_cost - saved - (trip_charge * num_trips) ∧ 
    grocery_value = 800 := by
sorry

end alex_grocery_delivery_l215_21503


namespace sign_of_b_is_negative_l215_21551

/-- Given that exactly two of a+b, a-b, ab, a/b are positive and the other two are negative, prove that b < 0 -/
theorem sign_of_b_is_negative (a b : ℝ) 
  (h : (a + b > 0 ∧ a - b > 0 ∧ a * b < 0 ∧ a / b < 0) ∨
       (a + b > 0 ∧ a - b < 0 ∧ a * b > 0 ∧ a / b < 0) ∨
       (a + b > 0 ∧ a - b < 0 ∧ a * b < 0 ∧ a / b > 0) ∨
       (a + b < 0 ∧ a - b > 0 ∧ a * b > 0 ∧ a / b < 0) ∨
       (a + b < 0 ∧ a - b > 0 ∧ a * b < 0 ∧ a / b > 0) ∨
       (a + b < 0 ∧ a - b < 0 ∧ a * b > 0 ∧ a / b > 0))
  (h_nonzero : b ≠ 0) : b < 0 := by
  sorry


end sign_of_b_is_negative_l215_21551


namespace inequality_solution_set_l215_21546

def inequality (x : ℝ) := x^2 - 3*x - 10 > 0

theorem inequality_solution_set :
  {x : ℝ | inequality x} = {x : ℝ | x > 5 ∨ x < -2} :=
by
  sorry

end inequality_solution_set_l215_21546


namespace opposite_of_one_fourth_l215_21572

theorem opposite_of_one_fourth : -(1 / 4 : ℚ) = -1 / 4 := by
  sorry

end opposite_of_one_fourth_l215_21572


namespace system_solution_unique_l215_21524

theorem system_solution_unique : ∃! (x y : ℝ), (3 * x - 4 * y = 12) ∧ (9 * x + 6 * y = -18) := by
  sorry

end system_solution_unique_l215_21524


namespace prob_red_tile_value_l215_21515

/-- The number of integers from 1 to 100 that are congruent to 3 mod 7 -/
def red_tiles : ℕ := (Finset.filter (fun n => n % 7 = 3) (Finset.range 100)).card

/-- The total number of tiles -/
def total_tiles : ℕ := 100

/-- The probability of selecting a red tile -/
def prob_red_tile : ℚ := red_tiles / total_tiles

theorem prob_red_tile_value :
  prob_red_tile = 7 / 50 := by sorry

end prob_red_tile_value_l215_21515


namespace clock_angle_theorem_l215_21520

/-- The angle (in degrees) the minute hand moves per minute -/
def minute_hand_speed : ℝ := 6

/-- The angle (in degrees) the hour hand moves per minute -/
def hour_hand_speed : ℝ := 0.5

/-- The current time in minutes past 3:00 -/
def t : ℝ := 23

/-- The position of the minute hand 8 minutes from now -/
def minute_hand_pos : ℝ := minute_hand_speed * (t + 8)

/-- The position of the hour hand 4 minutes ago -/
def hour_hand_pos : ℝ := 90 + hour_hand_speed * (t - 4)

/-- The theorem stating that the time is approximately 23 minutes past 3:00 -/
theorem clock_angle_theorem : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ 
  (|minute_hand_pos - hour_hand_pos| = 90 ∨ 
   |minute_hand_pos - hour_hand_pos| = 270) ∧
  t ≥ 0 ∧ t < 60 ∧ 
  |t - 23| < ε :=
sorry

end clock_angle_theorem_l215_21520


namespace f_increasing_l215_21534

def f (x : ℝ) := 2 * x

theorem f_increasing : ∀ x y : ℝ, x < y → f x < f y := by
  sorry

end f_increasing_l215_21534


namespace one_non_prime_expression_l215_21565

def expressions : List (ℕ → ℕ) := [
  (λ n => n^2 + (n+1)^2),
  (λ n => (n+1)^2 + (n+2)^2),
  (λ n => (n+2)^2 + (n+3)^2),
  (λ n => (n+3)^2 + (n+4)^2),
  (λ n => (n+4)^2 + (n+5)^2)
]

theorem one_non_prime_expression :
  (expressions.filter (λ f => ¬ Nat.Prime (f 1))).length = 1 := by
  sorry

end one_non_prime_expression_l215_21565


namespace vector_parallelism_l215_21510

theorem vector_parallelism (x : ℝ) : 
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (x, 1)
  (∃ (k : ℝ), k ≠ 0 ∧ (a.1 + 2 * b.1, a.2 + 2 * b.2) = k • (2 * a.1 - b.1, 2 * a.2 - b.2)) →
  x = (1 / 2 : ℝ) := by
sorry

end vector_parallelism_l215_21510


namespace yoongi_hoseok_age_sum_l215_21573

/-- Given the ages of Yoongi's aunt, the age difference between Yoongi and his aunt,
    and the age difference between Yoongi and Hoseok, prove that the sum of
    Yoongi and Hoseok's ages is 26 years. -/
theorem yoongi_hoseok_age_sum :
  ∀ (aunt_age : ℕ) (yoongi_aunt_diff : ℕ) (yoongi_hoseok_diff : ℕ),
  aunt_age = 38 →
  yoongi_aunt_diff = 23 →
  yoongi_hoseok_diff = 4 →
  (aunt_age - yoongi_aunt_diff) + (aunt_age - yoongi_aunt_diff - yoongi_hoseok_diff) = 26 :=
by sorry

end yoongi_hoseok_age_sum_l215_21573


namespace max_amount_received_back_l215_21580

/-- Represents the casino chip denominations -/
inductive ChipDenomination
  | twenty
  | hundred

/-- Calculates the value of a chip -/
def chipValue : ChipDenomination → ℕ
  | ChipDenomination.twenty => 20
  | ChipDenomination.hundred => 100

/-- Represents the number of chips lost for each denomination -/
structure ChipsLost where
  twenty : ℕ
  hundred : ℕ

/-- Calculates the total value of chips lost -/
def totalLost (chips : ChipsLost) : ℕ :=
  chips.twenty * chipValue ChipDenomination.twenty +
  chips.hundred * chipValue ChipDenomination.hundred

/-- Represents the casino scenario -/
structure CasinoScenario where
  totalBought : ℕ
  chipsLost : ChipsLost

/-- Calculates the amount received back -/
def amountReceivedBack (scenario : CasinoScenario) : ℕ :=
  scenario.totalBought - totalLost scenario.chipsLost

/-- The main theorem to prove -/
theorem max_amount_received_back :
  ∀ (scenario : CasinoScenario),
    scenario.totalBought = 3000 ∧
    scenario.chipsLost.twenty + scenario.chipsLost.hundred = 13 ∧
    (scenario.chipsLost.twenty = scenario.chipsLost.hundred + 3 ∨
     scenario.chipsLost.twenty = scenario.chipsLost.hundred - 3) →
    amountReceivedBack scenario ≤ 2340 :=
by
  sorry

#check max_amount_received_back

end max_amount_received_back_l215_21580
