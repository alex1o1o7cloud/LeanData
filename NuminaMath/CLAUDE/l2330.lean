import Mathlib

namespace parabola_equation_l2330_233011

/-- Given a parabola and a circle, prove the equation of the parabola -/
theorem parabola_equation (p : ℝ) (hp : p ≠ 0) :
  (∀ x y, x^2 = 2*p*y) →  -- Parabola equation
  (∀ x y, (x - 2)^2 + (y - 1)^2 = 1) →  -- Circle equation
  (∃ y, ∀ x, (x - 2)^2 + (y - 1)^2 = 1 ∧ y = -p/2) →  -- Axis of parabola is tangent to circle
  (∀ x y, x^2 = -8*y) :=  -- Conclusion: equation of the parabola
by sorry

end parabola_equation_l2330_233011


namespace parallel_segments_theorem_l2330_233066

/-- Represents a line segment with a length -/
structure Segment where
  length : ℝ

/-- Represents three parallel line segments intersecting another line segment -/
structure ParallelSegments where
  ab : Segment
  ef : Segment
  cd : Segment
  bc : Segment
  ab_parallel_ef : Bool
  ef_parallel_cd : Bool

/-- Given three parallel line segments intersecting another line segment,
    with specific lengths, the middle segment's length is 16 -/
theorem parallel_segments_theorem (p : ParallelSegments)
  (h1 : p.ab_parallel_ef = true)
  (h2 : p.ef_parallel_cd = true)
  (h3 : p.ab.length = 20)
  (h4 : p.cd.length = 80)
  (h5 : p.bc.length = 100) :
  p.ef.length = 16 := by
  sorry

end parallel_segments_theorem_l2330_233066


namespace no_distributive_laws_hold_l2330_233075

-- Define the # operation
def hash (a b : ℝ) : ℝ := a * b + 1

-- Theorem stating that none of the laws hold
theorem no_distributive_laws_hold :
  ¬(∀ (x y z : ℝ), hash x (y + z) = hash x y + hash x z) ∧
  ¬(∀ (x y z : ℝ), x + hash y z = hash (x + y) (x + z)) ∧
  ¬(∀ (x y z : ℝ), hash x (hash y z) = hash (hash x y) (hash x z)) := by
  sorry

end no_distributive_laws_hold_l2330_233075


namespace exists_valid_arrangement_l2330_233035

/-- A type representing a circular arrangement of 9 digits -/
def CircularArrangement := Fin 9 → Fin 9

/-- Checks if a number is composite -/
def is_composite (n : ℕ) : Prop := ∃ m, 1 < m ∧ m < n ∧ n % m = 0

/-- Checks if two adjacent digits in the arrangement form a composite number -/
def adjacent_composite (arr : CircularArrangement) (i : Fin 9) : Prop :=
  let n := (arr i).val * 10 + (arr ((i.val + 1) % 9)).val
  is_composite n

/-- The main theorem stating the existence of a valid arrangement -/
theorem exists_valid_arrangement : ∃ (arr : CircularArrangement), 
  (∀ i : Fin 9, 1 ≤ (arr i).val ∧ (arr i).val ≤ 9) ∧ 
  (∀ i j : Fin 9, i ≠ j → arr i ≠ arr j) ∧
  (∀ i : Fin 9, adjacent_composite arr i) :=
sorry

end exists_valid_arrangement_l2330_233035


namespace soccer_stars_draw_points_l2330_233059

/-- Represents a soccer team's season statistics -/
structure SoccerTeamStats where
  total_games : ℕ
  games_won : ℕ
  games_lost : ℕ
  points_per_win : ℕ
  total_points : ℕ

/-- Calculates the points earned for a draw given a team's season statistics -/
def points_per_draw (stats : SoccerTeamStats) : ℕ :=
  let games_drawn := stats.total_games - stats.games_won - stats.games_lost
  let points_from_wins := stats.games_won * stats.points_per_win
  let points_from_draws := stats.total_points - points_from_wins
  points_from_draws / games_drawn

/-- Theorem stating that Team Soccer Stars earns 1 point for each draw -/
theorem soccer_stars_draw_points :
  let stats : SoccerTeamStats := {
    total_games := 20,
    games_won := 14,
    games_lost := 2,
    points_per_win := 3,
    total_points := 46
  }
  points_per_draw stats = 1 := by sorry

end soccer_stars_draw_points_l2330_233059


namespace dilation_matrix_determinant_l2330_233054

theorem dilation_matrix_determinant :
  ∀ (E : Matrix (Fin 3) (Fin 3) ℝ),
  (∀ i j : Fin 3, i ≠ j → E i j = 0) →
  E 0 0 = 3 →
  E 1 1 = 5 →
  E 2 2 = 7 →
  Matrix.det E = 105 := by
sorry

end dilation_matrix_determinant_l2330_233054


namespace intersection_distance_l2330_233047

/-- The distance between intersection points of two curves with a ray in polar coordinates --/
theorem intersection_distance (θ : Real) : 
  let ρ₁ : Real := Real.sqrt (2 / (Real.cos θ ^ 2 - Real.sin θ ^ 2))
  let ρ₂ : Real := 4 * Real.cos θ
  θ = π / 6 → abs (ρ₁ - ρ₂) = 2 * Real.sqrt 3 - 2 := by
  sorry

end intersection_distance_l2330_233047


namespace average_of_set_l2330_233061

theorem average_of_set (S : Finset ℕ) (n : ℕ) (h_nonempty : S.Nonempty) :
  (∃ (max min : ℕ),
    max ∈ S ∧ min ∈ S ∧
    (∀ x ∈ S, x ≤ max) ∧
    (∀ x ∈ S, min ≤ x) ∧
    (S.sum id - max) / (S.card - 1) = 32 ∧
    (S.sum id - max - min) / (S.card - 2) = 35 ∧
    (S.sum id - min) / (S.card - 1) = 40 ∧
    max = min + 72) →
  S.sum id / S.card = 368 / 10 := by
sorry

end average_of_set_l2330_233061


namespace ellipse_eccentricity_square_trisection_l2330_233050

/-- Given an ellipse where the two trisection points on the minor axis and its two foci form a square,
    prove that its eccentricity is √10/10 -/
theorem ellipse_eccentricity_square_trisection (a b c : ℝ) :
  b = 3 * c →                    -- Condition: trisection points and foci form a square
  a ^ 2 = b ^ 2 + c ^ 2 →        -- Definition: relationship between semi-major axis, semi-minor axis, and focal distance
  c / a = (Real.sqrt 10) / 10 := by
sorry

end ellipse_eccentricity_square_trisection_l2330_233050


namespace shaded_area_between_circles_l2330_233026

theorem shaded_area_between_circles (r₁ r₂ : ℝ) : 
  r₁ = Real.sqrt 2 → r₂ = 2 * r₁ → π * r₂^2 - π * r₁^2 = 6 * π := by
  sorry

end shaded_area_between_circles_l2330_233026


namespace pencil_pen_problem_l2330_233082

theorem pencil_pen_problem (S : Finset Nat) (A B : Finset Nat) :
  S.card = 400 →
  A ⊆ S →
  B ⊆ S →
  A.card = 375 →
  B.card = 80 →
  S = A ∪ B →
  (A \ B).card = 320 := by
  sorry

end pencil_pen_problem_l2330_233082


namespace sum_of_roots_quadratic_l2330_233084

theorem sum_of_roots_quadratic (x : ℝ) :
  x^2 - 6*x + 8 = 0 → ∃ r₁ r₂ : ℝ, r₁ + r₂ = 6 ∧ x = r₁ ∨ x = r₂ :=
by
  sorry

end sum_of_roots_quadratic_l2330_233084


namespace portfolio_calculations_l2330_233074

/-- Represents a stock with its yield and quote -/
structure Stock where
  yield : ℝ
  quote : ℝ

/-- Calculates the weighted average yield of a portfolio -/
def weightedAverageYield (stocks : List Stock) (proportions : List ℝ) : ℝ :=
  sorry

/-- Calculates the overall quote of a portfolio -/
def overallQuote (stocks : List Stock) (proportions : List ℝ) (totalInvestment : ℝ) : ℝ :=
  sorry

/-- Theorem stating that weighted average yield and overall quote can be calculated -/
theorem portfolio_calculations 
  (stocks : List Stock) 
  (proportions : List ℝ) 
  (totalInvestment : ℝ) 
  (h1 : stocks.length = 3)
  (h2 : proportions.length = 3)
  (h3 : proportions.sum = 1)
  (h4 : totalInvestment > 0) :
  ∃ (avgYield overallQ : ℝ), 
    avgYield = weightedAverageYield stocks proportions ∧ 
    overallQ = overallQuote stocks proportions totalInvestment :=
  sorry

end portfolio_calculations_l2330_233074


namespace root_between_a_and_b_l2330_233028

theorem root_between_a_and_b (p q a b : ℝ) 
  (ha : a^2 + p*a + q = 0)
  (hb : b^2 - p*b - q = 0)
  (hq : q ≠ 0) :
  ∃ c ∈ Set.Ioo a b, c^2 + 2*p*c + 2*q = 0 := by
sorry

end root_between_a_and_b_l2330_233028


namespace racket_purchase_cost_l2330_233053

/-- The cost of two rackets with discounts -/
def total_cost (original_price : ℝ) : ℝ :=
  let first_racket_cost := original_price * (1 - 0.2)
  let second_racket_cost := original_price * 0.5
  first_racket_cost + second_racket_cost

/-- Theorem stating the total cost of two rackets -/
theorem racket_purchase_cost :
  total_cost 60 = 78 := by sorry

end racket_purchase_cost_l2330_233053


namespace inequality_equivalence_l2330_233088

theorem inequality_equivalence (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (3 : ℝ) ^ (1 / (x + 1/x)) > (3 : ℝ) ^ (1 / (y + 1/y)) ↔ 
  (x > 0 ∧ y < 0) ∨ 
  (x > y ∧ y > 0 ∧ x * y > 1) ∨ 
  (x < y ∧ y < 0 ∧ 0 < x * y ∧ x * y < 1) :=
by sorry

end inequality_equivalence_l2330_233088


namespace solution_x_chemical_b_percentage_l2330_233006

/-- Represents the composition of a chemical solution -/
structure Solution where
  a : ℝ  -- Percentage of chemical a
  b : ℝ  -- Percentage of chemical b

/-- Represents a mixture of two solutions -/
structure Mixture where
  x : Solution  -- First solution
  y : Solution  -- Second solution
  x_ratio : ℝ   -- Ratio of solution x in the mixture

/-- Given conditions of the problem -/
def problem_conditions : Prop :=
  ∃ (x y : Solution) (mix : Mixture),
    x.a = 0.40 ∧
    y.a = 0.50 ∧
    y.b = 0.50 ∧
    x.a + x.b = 1 ∧
    y.a + y.b = 1 ∧
    mix.x = x ∧
    mix.y = y ∧
    mix.x_ratio = 0.30 ∧
    mix.x_ratio * x.a + (1 - mix.x_ratio) * y.a = 0.47

/-- Theorem statement -/
theorem solution_x_chemical_b_percentage :
  problem_conditions →
  ∃ (x : Solution), x.b = 0.60 := by
  sorry

end solution_x_chemical_b_percentage_l2330_233006


namespace min_class_size_class_size_32_achievable_l2330_233063

/-- Represents the number of people in each group of the class --/
structure ClassGroups where
  first : ℕ
  second : ℕ
  third : ℕ

/-- The conditions of the tree-planting problem --/
def TreePlantingConditions (g : ClassGroups) : Prop :=
  g.second = (g.first + g.third) / 3 ∧
  4 * g.second = 5 * g.first + 3 * g.third - 72

/-- The theorem stating the minimum number of people in the class --/
theorem min_class_size (g : ClassGroups) 
  (h : TreePlantingConditions g) : 
  g.first + g.second + g.third ≥ 32 := by
  sorry

/-- The theorem stating that 32 is achievable --/
theorem class_size_32_achievable : 
  ∃ g : ClassGroups, TreePlantingConditions g ∧ g.first + g.second + g.third = 32 := by
  sorry

end min_class_size_class_size_32_achievable_l2330_233063


namespace min_value_theorem_l2330_233089

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_bisect : ∀ (x y : ℝ), 2*x + y - 2 = 0 → x^2 + y^2 - 2*a*x - 4*b*y + 1 = 0 → 
    ∃ (x' y' : ℝ), x'^2 + y'^2 - 2*a*x' - 4*b*y' + 1 = 0 ∧ 2*x' + y' - 2 ≠ 0) : 
  (∀ (a' b' : ℝ), a' > 0 → b' > 0 → 2/a' + 1/(2*b') ≥ 9/2) ∧ 
  (∃ (a' b' : ℝ), a' > 0 ∧ b' > 0 ∧ 2/a' + 1/(2*b') = 9/2) :=
by sorry

end min_value_theorem_l2330_233089


namespace digit_2500_is_3_l2330_233032

/-- Represents the decimal number obtained by writing integers from 999 down to 1 in reverse order -/
def reverse_decimal : ℚ :=
  sorry

/-- Returns the nth digit after the decimal point in the given rational number -/
def nth_digit (q : ℚ) (n : ℕ) : ℕ :=
  sorry

theorem digit_2500_is_3 : nth_digit reverse_decimal 2500 = 3 := by
  sorry

end digit_2500_is_3_l2330_233032


namespace inequality_equivalence_inequality_positive_reals_l2330_233067

-- Problem 1
theorem inequality_equivalence (x : ℝ) : (x + 2) / (2 - 3 * x) > 1 ↔ 0 < x ∧ x < 2 / 3 := by sorry

-- Problem 2
theorem inequality_positive_reals (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^2 + b^2 + c^2 ≥ a*b + b*c + a*c := by sorry

end inequality_equivalence_inequality_positive_reals_l2330_233067


namespace intersection_equals_interval_l2330_233068

-- Define the set M
def M : Set ℝ := {x | (x + 1) * (x - 2) ≤ 0}

-- Define the set N
def N : Set ℝ := {x | x > 1}

-- Theorem statement
theorem intersection_equals_interval : {x : ℝ | 1 < x ∧ x ≤ 2} = M ∩ N := by sorry

end intersection_equals_interval_l2330_233068


namespace stratified_sampling_best_l2330_233076

structure Population where
  total : ℕ
  group1 : ℕ
  group2 : ℕ
  sample_size : ℕ

def is_equal_proportion (p : Population) : Prop :=
  p.group1 = p.group2 ∧ p.total = p.group1 + p.group2

def maintains_proportion (p : Population) (method : String) : Prop :=
  method = "stratified sampling"

theorem stratified_sampling_best (p : Population) 
  (h1 : is_equal_proportion p) 
  (h2 : p.sample_size < p.total) :
  ∃ (method : String), maintains_proportion p method :=
sorry

end stratified_sampling_best_l2330_233076


namespace correct_minus_incorrect_l2330_233043

/-- Calculates the result following the order of operations -/
def J : ℤ := 12 - (3 * 4)

/-- Calculates the result ignoring parentheses and going from left to right -/
def A : ℤ := (12 - 3) * 4

/-- The difference between the correct calculation and the incorrect one -/
theorem correct_minus_incorrect : J - A = -36 := by sorry

end correct_minus_incorrect_l2330_233043


namespace divisor_count_squared_lt_4n_l2330_233013

def divisor_count (n : ℕ+) : ℕ := (Nat.divisors n.val).card

theorem divisor_count_squared_lt_4n (n : ℕ+) : (divisor_count n)^2 < 4 * n.val := by
  sorry

end divisor_count_squared_lt_4n_l2330_233013


namespace impossibleAllGood_l2330_233015

/-- A mushroom is either good or bad -/
inductive MushroomType
  | Good
  | Bad

/-- Definition of a mushroom -/
structure Mushroom where
  wormCount : ℕ
  type : MushroomType

/-- A basket of mushrooms -/
structure Basket where
  mushrooms : List Mushroom

/-- Function to determine if a mushroom is good -/
def isGoodMushroom (m : Mushroom) : Prop :=
  m.wormCount < 10

/-- Initial basket setup -/
def initialBasket : Basket :=
  { mushrooms := List.append
      (List.replicate 100 { wormCount := 10, type := MushroomType.Bad })
      (List.replicate 11 { wormCount := 0, type := MushroomType.Good }) }

/-- Theorem: It's impossible for all mushrooms to become good after redistribution -/
theorem impossibleAllGood (b : Basket) : ¬ ∀ m ∈ b.mushrooms, isGoodMushroom m := by
  sorry

#check impossibleAllGood initialBasket

end impossibleAllGood_l2330_233015


namespace no_2008_special_progressions_l2330_233096

theorem no_2008_special_progressions : ¬ ∃ (progressions : Fin 2008 → Set ℕ),
  -- Each set in progressions is an infinite arithmetic progression
  (∀ i, ∃ (a d : ℕ), d > 0 ∧ progressions i = {n : ℕ | ∃ k, n = a + k * d}) ∧
  -- There are finitely many positive integers not in any progression
  (∃ S : Finset ℕ, ∀ n, n ∉ S → ∃ i, n ∈ progressions i) ∧
  -- No two progressions intersect
  (∀ i j, i ≠ j → progressions i ∩ progressions j = ∅) ∧
  -- Each progression contains a prime number bigger than 2008
  (∀ i, ∃ p ∈ progressions i, p > 2008 ∧ Nat.Prime p) :=
by
  sorry

end no_2008_special_progressions_l2330_233096


namespace age_squares_sum_l2330_233051

theorem age_squares_sum (d t h : ℕ) : 
  t = 2 * d ∧ 
  h^2 + 4 * d = 5 * t ∧ 
  3 * h^2 = 7 * d^2 + 2 * t^2 →
  d^2 + h^2 + t^2 = 11 :=
by
  sorry

end age_squares_sum_l2330_233051


namespace arithmetic_to_geometric_l2330_233025

/-- An arithmetic sequence with given first two terms -/
def arithmetic_sequence (a₁ a₂ : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1) * (a₂ - a₁)

/-- Check if three numbers form a geometric sequence -/
def is_geometric_sequence (x y z : ℝ) : Prop :=
  y ^ 2 = x * z

theorem arithmetic_to_geometric :
  ∃ x : ℝ, is_geometric_sequence (x - 8) (x + (arithmetic_sequence (-8) (-6) 4))
                                 (x + (arithmetic_sequence (-8) (-6) 5)) ∧
            x = -1 := by
  sorry

end arithmetic_to_geometric_l2330_233025


namespace min_value_equiv_k_l2330_233085

/-- The polynomial function f(x, y, k) -/
def f (x y k : ℝ) : ℝ := 9*x^2 - 12*k*x*y + (2*k^2 + 3)*y^2 - 6*x - 9*y + 12

/-- The theorem stating the equivalence between the minimum value of f being 0 and k = √(3)/4 -/
theorem min_value_equiv_k (k : ℝ) : 
  (∀ x y : ℝ, f x y k ≥ 0) ∧ (∃ x y : ℝ, f x y k = 0) ↔ k = Real.sqrt 3 / 4 :=
sorry

end min_value_equiv_k_l2330_233085


namespace solve_video_game_problem_l2330_233003

def video_game_problem (total_games : ℕ) (potential_earnings : ℕ) (price_per_game : ℕ) : Prop :=
  let working_games := potential_earnings / price_per_game
  let non_working_games := total_games - working_games
  non_working_games = 8

theorem solve_video_game_problem :
  video_game_problem 16 56 7 :=
sorry

end solve_video_game_problem_l2330_233003


namespace new_cylinder_volume_l2330_233079

/-- Theorem: New volume of a cylinder after tripling radius and doubling height -/
theorem new_cylinder_volume (r h : ℝ) (h1 : r > 0) (h2 : h > 0) (h3 : π * r^2 * h = 15) :
  π * (3*r)^2 * (2*h) = 270 := by
  sorry

end new_cylinder_volume_l2330_233079


namespace tan_sum_reciprocal_l2330_233073

theorem tan_sum_reciprocal (x y : ℝ) 
  (h1 : (Real.sin x / Real.cos y) + (Real.sin y / Real.cos x) = 2)
  (h2 : (Real.cos x / Real.sin y) + (Real.cos y / Real.sin x) = 3) :
  (Real.tan x / Real.tan y) + (Real.tan y / Real.tan x) = 1 := by
  sorry

end tan_sum_reciprocal_l2330_233073


namespace remaining_liquid_weight_l2330_233030

/-- Proves that the weight of the remaining liquid after evaporation is 6 kg --/
theorem remaining_liquid_weight (initial_weight : ℝ) (evaporated_water : ℝ) (added_solution : ℝ) 
  (initial_x_percent : ℝ) (final_x_percent : ℝ) :
  initial_weight = 8 →
  evaporated_water = 2 →
  added_solution = 2 →
  initial_x_percent = 0.2 →
  final_x_percent = 0.25 →
  ∃ (remaining_weight : ℝ),
    remaining_weight = initial_weight - evaporated_water ∧
    (remaining_weight + added_solution) * final_x_percent = 
      initial_weight * initial_x_percent + added_solution * initial_x_percent ∧
    remaining_weight = 6 :=
by sorry

end remaining_liquid_weight_l2330_233030


namespace large_cylinder_height_l2330_233002

-- Define constants
def small_cylinder_diameter : ℝ := 3
def small_cylinder_height : ℝ := 6
def large_cylinder_diameter : ℝ := 20
def small_cylinders_to_fill : ℝ := 74.07407407407408

-- Define the theorem
theorem large_cylinder_height :
  let small_cylinder_volume := π * (small_cylinder_diameter / 2)^2 * small_cylinder_height
  let large_cylinder_radius := large_cylinder_diameter / 2
  let large_cylinder_volume := small_cylinders_to_fill * small_cylinder_volume
  large_cylinder_volume = π * large_cylinder_radius^2 * 10 := by
  sorry

end large_cylinder_height_l2330_233002


namespace unknown_rope_length_l2330_233070

/-- Calculates the length of an unknown rope given other rope lengths and conditions --/
theorem unknown_rope_length
  (known_ropes : List ℝ)
  (knot_loss : ℝ)
  (final_length : ℝ)
  (h1 : known_ropes = [8, 20, 2, 2, 2])
  (h2 : knot_loss = 1.2)
  (h3 : final_length = 35) :
  ∃ x : ℝ, x = 5.8 ∧ 
    final_length + (known_ropes.length * knot_loss) = 
    (known_ropes.sum + x) := by
  sorry


end unknown_rope_length_l2330_233070


namespace equation_solutions_l2330_233098

theorem equation_solutions (p : ℕ) (h_prime : Nat.Prime p) :
  ∃! (solutions : Finset (ℕ × ℕ)), 
    solutions.card = 3 ∧
    ∀ (x y : ℕ), (x, y) ∈ solutions ↔ 
      (x > 0 ∧ y > 0 ∧ (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / p) :=
by sorry

end equation_solutions_l2330_233098


namespace point_on_line_l2330_233049

/-- Given a point P(x, b) on the line x + y = 30, if the slope of OP is 4 (where O is the origin), then b = 24. -/
theorem point_on_line (x b : ℝ) : 
  x + b = 30 →  -- P(x, b) is on the line x + y = 30
  (b / x = 4) →  -- The slope of OP is 4
  b = 24 := by
sorry

end point_on_line_l2330_233049


namespace jeremy_age_l2330_233038

/-- Given the ages of Amy, Jeremy, and Chris, prove Jeremy's age --/
theorem jeremy_age (amy jeremy chris : ℕ) 
  (h1 : amy + jeremy + chris = 132)  -- Combined age
  (h2 : amy = jeremy / 3)            -- Amy's age relation to Jeremy
  (h3 : chris = 2 * amy)             -- Chris's age relation to Amy
  : jeremy = 66 := by
  sorry

end jeremy_age_l2330_233038


namespace power_product_equality_l2330_233042

theorem power_product_equality : 0.25^2015 * 4^2016 = 4 := by
  sorry

end power_product_equality_l2330_233042


namespace joe_hvac_zones_l2330_233062

def hvac_system (total_cost : ℕ) (vents_per_zone : ℕ) (cost_per_vent : ℕ) : ℕ :=
  (total_cost / cost_per_vent) / vents_per_zone

theorem joe_hvac_zones :
  hvac_system 20000 5 2000 = 2 := by
  sorry

end joe_hvac_zones_l2330_233062


namespace log_inequality_l2330_233034

theorem log_inequality (x : ℝ) (h : x > 0) : Real.log (1 + x) < x := by
  sorry

end log_inequality_l2330_233034


namespace no_prime_pair_divisibility_l2330_233052

theorem no_prime_pair_divisibility : ¬∃ (p q : ℕ), Prime p ∧ Prime q ∧ (p * q ∣ (2^p - 1) * (2^q - 1)) := by
  sorry

end no_prime_pair_divisibility_l2330_233052


namespace store_discount_calculation_l2330_233017

theorem store_discount_calculation (initial_discount : ℝ) (additional_discount : ℝ) (claimed_discount : ℝ) :
  initial_discount = 0.30 →
  additional_discount = 0.15 →
  claimed_discount = 0.45 →
  let remaining_after_initial := 1 - initial_discount
  let remaining_after_both := remaining_after_initial * (1 - additional_discount)
  let actual_discount := 1 - remaining_after_both
  (actual_discount = 0.405 ∧ claimed_discount - actual_discount = 0.045) := by
  sorry

#check store_discount_calculation

end store_discount_calculation_l2330_233017


namespace arithmetic_sequence_third_term_l2330_233005

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The third term of an arithmetic sequence {aₙ} where a₁ + a₅ = 6 equals 3 -/
theorem arithmetic_sequence_third_term
  (a : ℕ → ℝ)
  (h_arithmetic : ArithmeticSequence a)
  (h_sum : a 1 + a 5 = 6) :
  a 3 = 3 := by
  sorry

end arithmetic_sequence_third_term_l2330_233005


namespace jason_remaining_cards_l2330_233033

/-- The number of Pokemon cards Jason started with -/
def initial_cards : ℕ := 13

/-- The number of Pokemon cards Jason gave away -/
def cards_given_away : ℕ := 9

/-- The number of Pokemon cards Jason has now -/
def remaining_cards : ℕ := initial_cards - cards_given_away

theorem jason_remaining_cards : remaining_cards = 4 := by
  sorry

end jason_remaining_cards_l2330_233033


namespace tokens_theorem_l2330_233037

/-- The number of tokens Elsa has -/
def elsa_tokens : ℕ := 60

/-- The number of tokens Angus has -/
def x : ℕ := elsa_tokens - (elsa_tokens / 4)

/-- The number of tokens Bella has -/
def y : ℕ := elsa_tokens + (x^2 - 10)

theorem tokens_theorem : x = 45 ∧ y = 2075 := by
  sorry

end tokens_theorem_l2330_233037


namespace salt_dilution_l2330_233094

theorem salt_dilution (initial_seawater : ℝ) (initial_salt_percentage : ℝ) 
  (final_salt_percentage : ℝ) (added_freshwater : ℝ) :
  initial_seawater = 40 →
  initial_salt_percentage = 0.05 →
  final_salt_percentage = 0.02 →
  added_freshwater = 60 →
  (initial_seawater * initial_salt_percentage) / (initial_seawater + added_freshwater) = final_salt_percentage :=
by
  sorry

#check salt_dilution

end salt_dilution_l2330_233094


namespace min_cubes_for_box_l2330_233080

-- Define the box dimensions
def box_length : ℝ := 10
def box_width : ℝ := 18
def box_height : ℝ := 4

-- Define the volume of a single cube
def cube_volume : ℝ := 12

-- Theorem statement
theorem min_cubes_for_box :
  ⌈(box_length * box_width * box_height) / cube_volume⌉ = 60 := by
  sorry

end min_cubes_for_box_l2330_233080


namespace compute_M_v_minus_2w_l2330_233014

variable (M : Matrix (Fin 2) (Fin 2) ℝ)
variable (v w : Fin 2 → ℝ)

axiom Mv : M.mulVec v = ![4, 2]
axiom Mw : M.mulVec w = ![5, 1]

theorem compute_M_v_minus_2w :
  M.mulVec (v - 2 • w) = ![-6, 0] := by sorry

end compute_M_v_minus_2w_l2330_233014


namespace five_classes_in_school_l2330_233045

/-- Represents the number of students in each class -/
def class_sizes (n : ℕ) : ℕ → ℕ
  | 0 => 25
  | i + 1 => class_sizes n i - 2

/-- The total number of students in the school -/
def total_students (n : ℕ) : ℕ :=
  (List.range n).map (class_sizes n) |>.sum

/-- The theorem stating that there are 5 classes in the school -/
theorem five_classes_in_school :
  ∃ n : ℕ, n > 0 ∧ total_students n = 105 ∧ n = 5 :=
sorry

end five_classes_in_school_l2330_233045


namespace fraction_irreducible_l2330_233091

theorem fraction_irreducible (n : ℕ) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := by
  sorry

end fraction_irreducible_l2330_233091


namespace A_union_B_eq_l2330_233008

def A : Set ℝ := {x | x^2 - x - 2 < 0}

def B : Set ℝ := {x | x^2 - 3*x < 0}

theorem A_union_B_eq : A ∪ B = {x : ℝ | -1 < x ∧ x < 3} := by sorry

end A_union_B_eq_l2330_233008


namespace min_swaps_for_geese_order_l2330_233090

def initial_order : List ℕ := [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
def final_order : List ℕ := List.range 20 |>.map (· + 1)

def count_inversions (l : List ℕ) : ℕ :=
  l.foldr (fun x acc => acc + (l.filter (· < x) |>.filter (fun y => l.indexOf y > l.indexOf x) |>.length)) 0

def min_swaps_to_sort (l : List ℕ) : ℕ := count_inversions l

theorem min_swaps_for_geese_order :
  min_swaps_to_sort initial_order = 55 :=
sorry

end min_swaps_for_geese_order_l2330_233090


namespace triangle_problem_l2330_233044

noncomputable section

/-- Given a triangle ABC with the following properties:
  BC = √5
  AC = 3
  sin C = 2 * sin A
  Prove that:
  1. AB = 2√5
  2. sin(2A - π/4) = √2/10
-/
theorem triangle_problem (A B C : ℝ) (h1 : Real.sqrt 5 = BC)
  (h2 : 3 = AC) (h3 : Real.sin C = 2 * Real.sin A) :
  AB = 2 * Real.sqrt 5 ∧ Real.sin (2 * A - π / 4) = Real.sqrt 2 / 10 :=
by sorry

end

end triangle_problem_l2330_233044


namespace odometer_puzzle_l2330_233097

theorem odometer_puzzle (a b c d : ℕ) (h1 : a ≥ 1) (h2 : a + b + c + d = 10)
  (h3 : ∃ (x : ℕ), 1000 * (d - a) + 100 * (c - b) + 10 * (b - c) + (a - d) = 65 * x) :
  a^2 + b^2 + c^2 + d^2 = 42 := by
sorry

end odometer_puzzle_l2330_233097


namespace four_player_tournament_games_l2330_233021

/-- The number of games in a round-robin tournament -/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a tournament with 4 players, where each player plays against every
    other player exactly once, the total number of games played is 6. -/
theorem four_player_tournament_games :
  num_games 4 = 6 := by
  sorry

end four_player_tournament_games_l2330_233021


namespace red_blocks_count_l2330_233023

theorem red_blocks_count (red : ℕ) (yellow : ℕ) (blue : ℕ) 
  (h1 : yellow = red + 7)
  (h2 : blue = red + 14)
  (h3 : red + yellow + blue = 75) :
  red = 18 := by
  sorry

end red_blocks_count_l2330_233023


namespace max_candies_eaten_l2330_233083

theorem max_candies_eaten (n : Nat) (h : n = 46) : 
  (n * (n - 1)) / 2 = 1035 := by
  sorry

#check max_candies_eaten

end max_candies_eaten_l2330_233083


namespace find_M_and_N_l2330_233069

theorem find_M_and_N :
  ∀ M N : ℕ,
  0 < M ∧ M < 10 ∧ 0 < N ∧ N < 10 →
  8 * 10^7 + M * 10^6 + 420852 * 9 = N * 10^7 + 9889788 * 11 →
  M = 5 ∧ N = 6 := by
sorry

end find_M_and_N_l2330_233069


namespace y_in_terms_of_x_l2330_233004

theorem y_in_terms_of_x (p : ℝ) (x y : ℝ) 
  (hx : x = 1 + 3^p) (hy : y = 1 + 3^(-p)) : 
  y = x / (x - 1) := by
  sorry

end y_in_terms_of_x_l2330_233004


namespace count_distinct_arrangements_l2330_233064

/-- A regular five-pointed star with 10 positions for placing objects -/
structure StarArrangement where
  positions : Fin 10 → Fin 10

/-- The group of symmetries of a regular five-pointed star -/
def starSymmetryGroup : Fintype G := sorry

/-- The number of distinct arrangements of 10 different objects on a regular five-pointed star,
    considering rotations and reflections as equivalent -/
def distinctArrangements : ℕ := sorry

/-- Theorem stating the number of distinct arrangements -/
theorem count_distinct_arrangements :
  distinctArrangements = Nat.factorial 10 / 10 := by sorry

end count_distinct_arrangements_l2330_233064


namespace sin_585_degrees_l2330_233071

theorem sin_585_degrees :
  let π : ℝ := Real.pi
  let deg_to_rad (x : ℝ) : ℝ := x * π / 180
  ∀ (sin : ℝ → ℝ),
    (∀ x, sin (x + 2 * π) = sin x) →  -- Periodicity of sine
    (∀ x, sin (x + π) = -sin x) →     -- Sine of sum property
    sin (deg_to_rad 45) = Real.sqrt 2 / 2 →  -- Value of sin 45°
    sin (deg_to_rad 585) = -Real.sqrt 2 / 2 := by
sorry

end sin_585_degrees_l2330_233071


namespace lcm_180_616_l2330_233077

theorem lcm_180_616 : Nat.lcm 180 616 = 27720 := by
  sorry

end lcm_180_616_l2330_233077


namespace smallest_n_congruence_l2330_233086

theorem smallest_n_congruence : 
  ∃ (n : ℕ), n > 0 ∧ 1145 * n ≡ 1717 * n [ZMOD 36] ∧ 
  (∀ (m : ℕ), m > 0 ∧ m < n → ¬(1145 * m ≡ 1717 * m [ZMOD 36])) ∧ 
  n = 9 := by
  sorry

end smallest_n_congruence_l2330_233086


namespace alternating_sum_coefficients_l2330_233024

theorem alternating_sum_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (2*x + 1)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₀ - a₁ + a₂ - a₃ + a₄ - a₅ = -1 := by
sorry

end alternating_sum_coefficients_l2330_233024


namespace range_of_m_l2330_233018

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 4*x - 6

-- Define the theorem
theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Icc 0 m, f x ∈ Set.Icc (-10) (-6)) ∧
  (∃ x ∈ Set.Icc 0 m, f x = -10) ∧
  (∃ x ∈ Set.Icc 0 m, f x = -6) →
  m ∈ Set.Icc 2 4 :=
by sorry

end range_of_m_l2330_233018


namespace haley_josh_necklace_difference_haley_josh_necklace_difference_proof_l2330_233022

/-- Given the number of necklaces for Haley, Jason, and Josh, prove that Haley has 15 more necklaces than Josh. -/
theorem haley_josh_necklace_difference : ℕ → ℕ → ℕ → Prop :=
  fun haley jason josh =>
    (haley = jason + 5) →
    (josh = jason / 2) →
    (haley = 25) →
    (haley - josh = 15)

/-- Proof of the theorem -/
theorem haley_josh_necklace_difference_proof :
  ∀ haley jason josh, haley_josh_necklace_difference haley jason josh :=
by
  sorry

#check haley_josh_necklace_difference
#check haley_josh_necklace_difference_proof

end haley_josh_necklace_difference_haley_josh_necklace_difference_proof_l2330_233022


namespace negation_of_inequality_proposition_l2330_233087

theorem negation_of_inequality_proposition :
  (¬ ∀ a b : ℝ, a^2 + b^2 ≥ 2*a*b) ↔ (∃ a b : ℝ, a^2 + b^2 < 2*a*b) := by
  sorry

end negation_of_inequality_proposition_l2330_233087


namespace anhui_imports_exports_2012_l2330_233099

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem anhui_imports_exports_2012 :
  toScientificNotation (39.33 * 10^9) = ScientificNotation.mk 3.933 10 sorry := by
  sorry

end anhui_imports_exports_2012_l2330_233099


namespace best_overall_value_l2330_233036

structure Box where
  brand : String
  size : Nat
  price : Rat
  quality : Rat

def pricePerOunce (b : Box) : Rat :=
  b.price / b.size

def overallValue (b : Box) : Rat :=
  b.quality / (pricePerOunce b)

theorem best_overall_value (box1 box2 box3 box4 : Box) 
  (h1 : box1 = { brand := "A", size := 30, price := 480/100, quality := 9/2 })
  (h2 : box2 = { brand := "A", size := 20, price := 340/100, quality := 9/2 })
  (h3 : box3 = { brand := "B", size := 15, price := 200/100, quality := 39/10 })
  (h4 : box4 = { brand := "B", size := 25, price := 325/100, quality := 39/10 }) :
  overallValue box1 ≥ overallValue box2 ∧ 
  overallValue box1 ≥ overallValue box3 ∧ 
  overallValue box1 ≥ overallValue box4 := by
  sorry

#check best_overall_value

end best_overall_value_l2330_233036


namespace days_to_fill_tank_l2330_233060

def tank_capacity : ℝ := 350000 -- in milliliters
def min_daily_collection : ℝ := 1200 -- in milliliters
def max_daily_collection : ℝ := 2100 -- in milliliters

theorem days_to_fill_tank : 
  ∃ (days : ℕ), days = 213 ∧ 
  (tank_capacity / min_daily_collection ≤ days) ∧
  (tank_capacity / max_daily_collection ≤ days) ∧
  (∀ d : ℕ, d < days → d * max_daily_collection < tank_capacity) :=
sorry

end days_to_fill_tank_l2330_233060


namespace ajay_ride_distance_l2330_233040

/-- Given Ajay's speed and travel time, calculate the distance he rides -/
theorem ajay_ride_distance (speed : ℝ) (time : ℝ) (distance : ℝ) : 
  speed = 50 → time = 30 → distance = speed * time → distance = 1500 :=
by
  sorry

#check ajay_ride_distance

end ajay_ride_distance_l2330_233040


namespace sector_properties_l2330_233058

/-- Proves that a circular sector with perimeter 4 and area 1 has radius 1 and central angle 2 -/
theorem sector_properties :
  ∀ r θ : ℝ,
  r > 0 →
  θ > 0 →
  2 * r + θ * r = 4 →
  1 / 2 * θ * r^2 = 1 →
  r = 1 ∧ θ = 2 := by
sorry

end sector_properties_l2330_233058


namespace room_dimension_increase_l2330_233031

/-- 
  Given a rectangular room where increasing both length and breadth by y feet
  increases the perimeter by 16 feet, prove that y equals 4 feet.
-/
theorem room_dimension_increase (L B : ℝ) (y : ℝ) 
  (h : 2 * ((L + y) + (B + y)) = 2 * (L + B) + 16) : y = 4 :=
by sorry

end room_dimension_increase_l2330_233031


namespace solve_star_equation_l2330_233019

/-- Custom binary operation -/
def star (a b : ℚ) : ℚ := a * b + 3 * b - 2 * a

/-- Theorem stating the solution to the equation -/
theorem solve_star_equation : ∃ x : ℚ, star 3 x = 23 ∧ x = 29 / 6 := by
  sorry

end solve_star_equation_l2330_233019


namespace linear_function_condition_l2330_233048

/-- A linear function with respect to x of the form y = (m-2)x + 2 -/
def linearFunction (m : ℝ) (x : ℝ) : ℝ := (m - 2) * x + 2

/-- The condition for the function to be linear with respect to x -/
def isLinear (m : ℝ) : Prop := m ≠ 2

theorem linear_function_condition (m : ℝ) :
  (∀ x, ∃ y, y = linearFunction m x) ↔ isLinear m :=
sorry

end linear_function_condition_l2330_233048


namespace solution_set_intersection_range_l2330_233056

-- Define the function f(x)
def f (m : ℝ) (x : ℝ) : ℝ := m - |x - 1| - 2*|x + 1|

-- Part I
theorem solution_set (x : ℝ) : 
  x ∈ Set.Ioo (-4/3 : ℝ) 1 ↔ f 5 x > 2 := by sorry

-- Part II
theorem intersection_range (m : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, y = x^2 + 2*x + 3 ∧ y = f m x) ↔ m ≥ 4 := by sorry

end solution_set_intersection_range_l2330_233056


namespace triangle_properties_l2330_233092

/-- Given a triangle ABC with acute angles A and B, prove the following:
    1. If ∠C = π/3 and c = 2, then 2 + 2√3 < perimeter ≤ 6
    2. If sin²A + sin²B > sin²C, then sin²A + sin²B > 1 -/
theorem triangle_properties (A B C : Real) (a b c : Real) :
  0 < A ∧ A < π/2 →
  0 < B ∧ B < π/2 →
  A + B + C = π →
  a > 0 ∧ b > 0 ∧ c > 0 →
  (C = π/3 ∧ c = 2 → 2 + 2 * Real.sqrt 3 < a + b + c ∧ a + b + c ≤ 6) ∧
  (Real.sin A ^ 2 + Real.sin B ^ 2 > Real.sin C ^ 2 → Real.sin A ^ 2 + Real.sin B ^ 2 > 1) := by
  sorry


end triangle_properties_l2330_233092


namespace muffin_ratio_l2330_233020

/-- The number of muffins Sasha made -/
def sasha_muffins : ℕ := 30

/-- The price of each muffin in dollars -/
def muffin_price : ℕ := 4

/-- The total amount raised in dollars -/
def total_raised : ℕ := 900

/-- The number of muffins Melissa made -/
def melissa_muffins : ℕ := 120

/-- The number of muffins Tiffany made -/
def tiffany_muffins : ℕ := (sasha_muffins + melissa_muffins) / 2

/-- The total number of muffins made -/
def total_muffins : ℕ := sasha_muffins + melissa_muffins + tiffany_muffins

theorem muffin_ratio : 
  (total_muffins * muffin_price = total_raised) → 
  (melissa_muffins : ℚ) / sasha_muffins = 4 := by
sorry

end muffin_ratio_l2330_233020


namespace slope_angle_of_line_l2330_233046

theorem slope_angle_of_line (x y : ℝ) :
  y = -Real.sqrt 3 * x + 1 → Real.arctan (-Real.sqrt 3) * (180 / Real.pi) = 120 := by
  sorry

end slope_angle_of_line_l2330_233046


namespace transformed_ellipse_equation_l2330_233078

/-- The equation of the curve obtained by transforming points on the ellipse x²/4 + y² = 1
    by keeping the x-coordinate unchanged and doubling the y-coordinate -/
theorem transformed_ellipse_equation :
  ∀ (x y : ℝ), x^2 / 4 + y^2 = 1 → (x^2 + (2*y)^2 = 4) :=
by sorry

end transformed_ellipse_equation_l2330_233078


namespace sin_minus_pi_half_times_tan_pi_minus_l2330_233009

open Real

theorem sin_minus_pi_half_times_tan_pi_minus (α : ℝ) : 
  sin (α - π / 2) * tan (π - α) = sin α := by
  sorry

end sin_minus_pi_half_times_tan_pi_minus_l2330_233009


namespace lcm_gcd_product_10_15_l2330_233029

theorem lcm_gcd_product_10_15 : Nat.lcm 10 15 * Nat.gcd 10 15 = 150 := by
  sorry

end lcm_gcd_product_10_15_l2330_233029


namespace lawn_mowing_time_mowing_time_approx_2_3_l2330_233065

/-- Represents the lawn mowing problem -/
theorem lawn_mowing_time (lawn_length lawn_width : ℝ) 
                         (swath_width overlap : ℝ) 
                         (mowing_speed : ℝ) : ℝ :=
  let effective_swath := swath_width - overlap
  let num_strips := lawn_width / effective_swath
  let total_distance := num_strips * lawn_length
  let time_taken := total_distance / mowing_speed
  time_taken

/-- Proves that the time taken to mow the lawn is approximately 2.3 hours -/
theorem mowing_time_approx_2_3 :
  ∃ ε > 0, |lawn_mowing_time 120 180 (30/12) (2/12) 4000 - 2.3| < ε :=
sorry

end lawn_mowing_time_mowing_time_approx_2_3_l2330_233065


namespace complex_equation_solution_l2330_233007

theorem complex_equation_solution (a : ℝ) (z : ℂ) 
  (h1 : a ≥ 0) 
  (h2 : z * Complex.abs z + a * z + Complex.I = 0) : 
  z = Complex.I * ((a - Real.sqrt (a^2 + 4)) / 2) := by
  sorry

end complex_equation_solution_l2330_233007


namespace restaurant_tip_percentage_l2330_233093

/-- Calculates the tip percentage given the total bill and tip amount -/
def tip_percentage (total_bill : ℚ) (tip_amount : ℚ) : ℚ :=
  (tip_amount / total_bill) * 100

/-- Proves that for a $40 bill and $4 tip, the tip percentage is 10% -/
theorem restaurant_tip_percentage : tip_percentage 40 4 = 10 := by
  sorry

end restaurant_tip_percentage_l2330_233093


namespace initial_water_percentage_in_milk_initial_water_percentage_is_five_percent_l2330_233000

/-- Proves that the initial water percentage in milk is 5% given the specified conditions -/
theorem initial_water_percentage_in_milk : ℝ → Prop :=
  fun initial_percentage =>
    let initial_volume : ℝ := 10
    let pure_milk_added : ℝ := 15
    let final_percentage : ℝ := 2
    let final_volume : ℝ := 25
    let initial_water_volume := (initial_percentage / 100) * initial_volume
    let final_water_volume := (final_percentage / 100) * final_volume
    initial_water_volume = final_water_volume ∧ initial_percentage = 5

/-- The initial water percentage in milk is 5% -/
theorem initial_water_percentage_is_five_percent : 
  initial_water_percentage_in_milk 5 := by
  sorry

end initial_water_percentage_in_milk_initial_water_percentage_is_five_percent_l2330_233000


namespace slope_of_line_l2330_233057

theorem slope_of_line (x y : ℝ) :
  4 * x - 7 * y = 14 → (y - (-2)) / (x - 0) = 4 / 7 := by
  sorry

end slope_of_line_l2330_233057


namespace kylie_coins_left_l2330_233016

/-- The number of coins Kylie collected and gave away -/
structure CoinCollection where
  piggy_bank : ℕ
  from_brother : ℕ
  from_father : ℕ
  given_away : ℕ

/-- Calculate the number of coins Kylie has left -/
def coins_left (c : CoinCollection) : ℕ :=
  c.piggy_bank + c.from_brother + c.from_father - c.given_away

/-- Theorem stating that Kylie has 15 coins left -/
theorem kylie_coins_left :
  ∀ (c : CoinCollection),
  c.piggy_bank = 15 →
  c.from_brother = 13 →
  c.from_father = 8 →
  c.given_away = 21 →
  coins_left c = 15 :=
by
  sorry

end kylie_coins_left_l2330_233016


namespace fraction_irreducible_l2330_233072

theorem fraction_irreducible (n : ℕ) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := by
  sorry

end fraction_irreducible_l2330_233072


namespace min_a_for_g_zeros_l2330_233039

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x| + |x - 1|

-- Define the function g(x) in terms of f(x) and a
def g (a : ℝ) (x : ℝ) : ℝ := f x - a

-- Theorem statement
theorem min_a_for_g_zeros :
  ∃ (a : ℝ), (∃ (x : ℝ), g a x = 0) ∧
  (∀ (b : ℝ), b < a → ¬∃ (x : ℝ), g b x = 0) ∧
  a = 1 :=
sorry

end min_a_for_g_zeros_l2330_233039


namespace min_value_f_min_value_achieved_l2330_233055

def f (x : ℝ) : ℝ := 3 * x^2 + 2 * x + 1

theorem min_value_f :
  ∀ x : ℝ, x ≥ 0 → f x ≥ 1 := by
  sorry

theorem min_value_achieved :
  ∃ x : ℝ, x ≥ 0 ∧ f x = 1 := by
  sorry

end min_value_f_min_value_achieved_l2330_233055


namespace angle_ABD_measure_l2330_233041

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : Point)

-- Define the angles in the quadrilateral
def angle_ABC (q : Quadrilateral) : ℝ := 120
def angle_DAB (q : Quadrilateral) : ℝ := 30
def angle_ADB (q : Quadrilateral) : ℝ := 28

-- Define the theorem
theorem angle_ABD_measure (q : Quadrilateral) :
  angle_ABC q = 120 ∧ angle_DAB q = 30 ∧ angle_ADB q = 28 →
  ∃ (angle_ABD : ℝ), angle_ABD = 122 :=
sorry

end angle_ABD_measure_l2330_233041


namespace room_length_proof_l2330_233001

/-- Given a room with known width, total paving cost, and paving rate per square meter,
    prove that the length of the room is 5.5 meters. -/
theorem room_length_proof (width : ℝ) (total_cost : ℝ) (rate_per_sq_meter : ℝ) 
    (h1 : width = 3.75)
    (h2 : total_cost = 16500)
    (h3 : rate_per_sq_meter = 800) : 
  total_cost / rate_per_sq_meter / width = 5.5 := by
  sorry


end room_length_proof_l2330_233001


namespace quadratic_equation_roots_l2330_233095

theorem quadratic_equation_roots : ∃ x y : ℝ, x ≠ y ∧ 
  (x^2 + 2*x - 3 = 0) ∧ (y^2 + 2*y - 3 = 0) := by
  sorry

end quadratic_equation_roots_l2330_233095


namespace solution_set_implies_range_l2330_233027

/-- The solution set of the inequality ax^2 + ax - 4 < 0 is ℝ -/
def solution_set_is_reals (a : ℝ) : Prop :=
  ∀ x, a * x^2 + a * x - 4 < 0

/-- The range of a is (-16, 0] -/
def range_of_a : Set ℝ := Set.Ioc (-16) 0

theorem solution_set_implies_range :
  (∃ a, solution_set_is_reals a) → (∀ a, solution_set_is_reals a ↔ a ∈ range_of_a) :=
by sorry

end solution_set_implies_range_l2330_233027


namespace simplify_expression_l2330_233012

theorem simplify_expression (x : ℝ) : (2*x + 25) + (150*x + 35) + (50*x + 10) = 202*x + 70 := by
  sorry

end simplify_expression_l2330_233012


namespace boosters_club_average_sales_l2330_233010

/-- Calculates the average monthly sales for the Boosters Club --/
theorem boosters_club_average_sales
  (sales : List ℝ)
  (refund : ℝ)
  (h1 : sales = [90, 75, 55, 130, 110, 85])
  (h2 : refund = 25)
  (h3 : sales.length = 6) :
  (sales.sum - refund) / sales.length = 86.67 := by
  sorry

end boosters_club_average_sales_l2330_233010


namespace small_branches_count_l2330_233081

/-- Represents the structure of a plant with branches and small branches. -/
structure Plant where
  small_branches_per_branch : ℕ
  total_count : ℕ

/-- The plant satisfies the given conditions. -/
def valid_plant (p : Plant) : Prop :=
  p.total_count = 1 + p.small_branches_per_branch + p.small_branches_per_branch^2

/-- Theorem: Given the conditions, the number of small branches per branch is 9. -/
theorem small_branches_count (p : Plant) 
    (h : valid_plant p) 
    (h_total : p.total_count = 91) : 
  p.small_branches_per_branch = 9 := by
  sorry

#check small_branches_count

end small_branches_count_l2330_233081
