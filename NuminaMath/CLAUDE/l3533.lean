import Mathlib

namespace inscribed_cube_volume_is_three_root_six_over_four_l3533_353366

/-- A pyramid with a square base and equilateral triangle lateral faces -/
structure Pyramid :=
  (base_side : ℝ)
  (base_is_square : base_side > 0)
  (lateral_faces_are_equilateral : True)

/-- A cube inscribed in a pyramid -/
structure InscribedCube (p : Pyramid) :=
  (side_length : ℝ)
  (base_on_pyramid_base : True)
  (top_vertices_touch_midpoints : True)

/-- The volume of the inscribed cube -/
def inscribed_cube_volume (p : Pyramid) (c : InscribedCube p) : ℝ :=
  c.side_length ^ 3

/-- Main theorem: The volume of the inscribed cube is 3√6/4 -/
theorem inscribed_cube_volume_is_three_root_six_over_four 
  (p : Pyramid) 
  (h_base : p.base_side = 2)
  (c : InscribedCube p) : 
  inscribed_cube_volume p c = 3 * Real.sqrt 6 / 4 :=
sorry

end inscribed_cube_volume_is_three_root_six_over_four_l3533_353366


namespace evelyn_bottle_caps_l3533_353348

/-- The number of bottle caps Evelyn ends with after losing some -/
def bottle_caps_remaining (initial : Float) (lost : Float) : Float :=
  initial - lost

/-- Theorem: If Evelyn starts with 63.0 bottle caps and loses 18.0, she ends with 45.0 -/
theorem evelyn_bottle_caps : bottle_caps_remaining 63.0 18.0 = 45.0 := by
  sorry

end evelyn_bottle_caps_l3533_353348


namespace equation_has_one_solution_l3533_353372

theorem equation_has_one_solution :
  ∃! x : ℝ, x - 8 / (x - 2) = 4 - 8 / (x - 2) ∧ x ≠ 2 := by
  sorry

end equation_has_one_solution_l3533_353372


namespace infinite_rational_square_sum_169_l3533_353358

theorem infinite_rational_square_sum_169 : 
  ∀ n : ℕ, ∃ x y : ℚ, x^2 + y^2 = 169 ∧ 
  (∀ m : ℕ, m < n → ∃ x' y' : ℚ, x'^2 + y'^2 = 169 ∧ (x' ≠ x ∨ y' ≠ y)) :=
by sorry

end infinite_rational_square_sum_169_l3533_353358


namespace T_equals_eleven_l3533_353300

/-- Given a natural number S, we define F as the sum of powers of 2 from 0 to S -/
def F (S : ℕ) : ℝ := (2^(S+1) - 1)

/-- T is defined as the square root of the ratio of logarithms -/
noncomputable def T (S : ℕ) : ℝ := Real.sqrt (Real.log (1 + F S) / Real.log 2)

/-- The theorem states that for S = 120, T equals 11 -/
theorem T_equals_eleven : T 120 = 11 := by sorry

end T_equals_eleven_l3533_353300


namespace bezout_identity_solutions_l3533_353319

theorem bezout_identity_solutions (a b d u v : ℤ) 
  (h_gcd : d = Int.gcd a b) 
  (h_bezout : a * u + b * v = d) : 
  (∀ x y : ℤ, a * x + b * y = d ↔ ∃ k : ℤ, x = u + k * b ∧ y = v - k * a) ∧
  {p : ℤ × ℤ | a * p.1 + b * p.2 = d} = {p : ℤ × ℤ | ∃ k : ℤ, p = (u + k * b, v - k * a)} :=
by sorry

end bezout_identity_solutions_l3533_353319


namespace triangle_integer_points_l3533_353338

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Checks if a point has integer coordinates -/
def hasIntegerCoordinates (p : Point) : Prop :=
  ∃ (ix iy : ℤ), p.x = ↑ix ∧ p.y = ↑iy

/-- Checks if a point is inside or on the boundary of the triangle formed by three points -/
def isInsideOrOnBoundary (p A B C : Point) : Prop :=
  sorry -- Definition of this predicate

/-- Counts the number of points with integer coordinates inside or on the boundary of the triangle -/
def countIntegerPoints (A B C : Point) : ℕ :=
  sorry -- Definition of this function

/-- The main theorem -/
theorem triangle_integer_points (a : ℝ) :
  a > 0 →
  let A : Point := ⟨2 + a, 0⟩
  let B : Point := ⟨2 - a, 0⟩
  let C : Point := ⟨2, 1⟩
  (countIntegerPoints A B C = 4) ↔ (1 ≤ a ∧ a < 2) :=
sorry

end triangle_integer_points_l3533_353338


namespace divisibility_condition_l3533_353353

theorem divisibility_condition (n : ℤ) : 
  (n + 2) ∣ (n^2 + 3) ↔ n ∈ ({-9, -3, -1, 5} : Set ℤ) := by
  sorry

end divisibility_condition_l3533_353353


namespace find_x_l3533_353322

theorem find_x : ∃ x : ℝ, (3 * x + 5) / 5 = 13 ∧ x = 20 := by
  sorry

end find_x_l3533_353322


namespace local_max_value_l3533_353308

/-- The function f(x) = x³ - 12x --/
def f (x : ℝ) : ℝ := x^3 - 12*x

/-- m is the point of local maximum for f --/
def is_local_max (m : ℝ) : Prop :=
  ∃ δ > 0, ∀ x, |x - m| < δ → f x ≤ f m

theorem local_max_value :
  ∃ m : ℝ, is_local_max m ∧ m = -2 :=
sorry

end local_max_value_l3533_353308


namespace investment_amount_is_14400_l3533_353332

/-- Represents the investment scenario --/
structure Investment where
  face_value : ℕ
  premium_percentage : ℕ
  dividend_percentage : ℕ
  total_dividend : ℕ

/-- Calculates the amount invested given the investment parameters --/
def amount_invested (i : Investment) : ℕ :=
  let share_price := i.face_value + i.face_value * i.premium_percentage / 100
  let dividend_per_share := i.face_value * i.dividend_percentage / 100
  let num_shares := i.total_dividend / dividend_per_share
  num_shares * share_price

/-- Theorem stating that the amount invested is 14400 given the specific conditions --/
theorem investment_amount_is_14400 :
  ∀ i : Investment,
    i.face_value = 100 ∧
    i.premium_percentage = 20 ∧
    i.dividend_percentage = 5 ∧
    i.total_dividend = 600 →
    amount_invested i = 14400 :=
by
  sorry

end investment_amount_is_14400_l3533_353332


namespace hassan_apple_trees_count_l3533_353386

/-- The number of apple trees Hassan has -/
def hassan_apple_trees : ℕ := 1

/-- The number of orange trees Ahmed has -/
def ahmed_orange_trees : ℕ := 8

/-- The number of orange trees Hassan has -/
def hassan_orange_trees : ℕ := 2

/-- The number of apple trees Ahmed has -/
def ahmed_apple_trees : ℕ := 4 * hassan_apple_trees

/-- The total number of trees in Ahmed's orchard -/
def ahmed_total_trees : ℕ := ahmed_orange_trees + ahmed_apple_trees

/-- The total number of trees in Hassan's orchard -/
def hassan_total_trees : ℕ := hassan_orange_trees + hassan_apple_trees

theorem hassan_apple_trees_count :
  hassan_apple_trees = 1 ∧
  ahmed_orange_trees = 8 ∧
  hassan_orange_trees = 2 ∧
  ahmed_apple_trees = 4 * hassan_apple_trees ∧
  ahmed_total_trees = hassan_total_trees + 9 := by
  sorry

end hassan_apple_trees_count_l3533_353386


namespace initial_patio_rows_l3533_353356

/-- Represents a rectangular patio -/
structure Patio where
  rows : ℕ
  cols : ℕ

/-- Checks if a patio is valid according to the given conditions -/
def isValidPatio (p : Patio) : Prop :=
  p.rows * p.cols = 60 ∧
  2 * p.cols = (3 * p.rows) / 2 ∧
  (p.rows + 5) * (p.cols - 3) = 60

theorem initial_patio_rows : 
  ∃ (p : Patio), isValidPatio p ∧ p.rows = 10 := by
  sorry

end initial_patio_rows_l3533_353356


namespace candidate_X_loses_by_6_percent_l3533_353359

/-- Represents the political parties --/
inductive Party
  | Republican
  | Democrat
  | Independent

/-- Represents the candidates --/
inductive Candidate
  | X
  | Y

/-- The ratio of registered voters for each party --/
def partyRatio : Party → Nat
  | Party.Republican => 3
  | Party.Democrat => 2
  | Party.Independent => 5

/-- The percentage of voters from each party expected to vote for candidate X --/
def votePercentageForX : Party → Rat
  | Party.Republican => 70/100
  | Party.Democrat => 30/100
  | Party.Independent => 40/100

/-- The percentage of registered voters who will not vote --/
def nonVoterPercentage : Rat := 10/100

/-- Theorem stating that candidate X is expected to lose by approximately 6% --/
theorem candidate_X_loses_by_6_percent :
  ∃ (total_voters : Nat),
    total_voters > 0 →
    let votes_for_X := (partyRatio Party.Republican * (votePercentageForX Party.Republican : Rat) +
                        partyRatio Party.Democrat * (votePercentageForX Party.Democrat : Rat) +
                        partyRatio Party.Independent * (votePercentageForX Party.Independent : Rat)) *
                       (1 - nonVoterPercentage) * total_voters
    let votes_for_Y := (partyRatio Party.Republican * (1 - votePercentageForX Party.Republican : Rat) +
                        partyRatio Party.Democrat * (1 - votePercentageForX Party.Democrat : Rat) +
                        partyRatio Party.Independent * (1 - votePercentageForX Party.Independent : Rat)) *
                       (1 - nonVoterPercentage) * total_voters
    let total_votes := votes_for_X + votes_for_Y
    let percentage_difference := (votes_for_Y - votes_for_X) / total_votes * 100
    abs (percentage_difference - 6) < 1 := by
  sorry

end candidate_X_loses_by_6_percent_l3533_353359


namespace committee_selections_of_seven_l3533_353369

/-- The number of ways to select a chairperson and a deputy chairperson from a committee. -/
def committee_selections (n : ℕ) : ℕ := n * (n - 1)

/-- Theorem: The number of ways to select a chairperson and a deputy chairperson 
    from a committee of 7 members is 42. -/
theorem committee_selections_of_seven : committee_selections 7 = 42 := by
  sorry

end committee_selections_of_seven_l3533_353369


namespace john_needs_two_planks_l3533_353329

/-- The number of planks needed for a house wall --/
def planks_needed (total_nails : ℕ) (nails_per_plank : ℕ) : ℕ :=
  total_nails / nails_per_plank

/-- Theorem: John needs 2 planks for the house wall --/
theorem john_needs_two_planks :
  planks_needed 4 2 = 2 := by
  sorry

end john_needs_two_planks_l3533_353329


namespace horseback_riding_distance_l3533_353330

/-- Calculates the total distance traveled during a 3-day horseback riding trip -/
theorem horseback_riding_distance : 
  let day1_speed : ℝ := 5
  let day1_time : ℝ := 7
  let day2_speed1 : ℝ := 6
  let day2_time1 : ℝ := 6
  let day2_speed2 : ℝ := day2_speed1 / 2
  let day2_time2 : ℝ := 3
  let day3_speed : ℝ := 7
  let day3_time : ℝ := 5
  let total_distance : ℝ := 
    day1_speed * day1_time + 
    day2_speed1 * day2_time1 + day2_speed2 * day2_time2 + 
    day3_speed * day3_time
  total_distance = 115 := by
  sorry


end horseback_riding_distance_l3533_353330


namespace calculation_proof_l3533_353313

theorem calculation_proof : 
  100 - (25/8) / ((25/12) - (5/8)) * ((8/5) + (8/3)) = 636/7 := by
  sorry

end calculation_proof_l3533_353313


namespace intersection_M_N_l3533_353375

def M : Set ℤ := {m : ℤ | -3 < m ∧ m < 2}
def N : Set ℤ := {n : ℤ | -1 ≤ n ∧ n ≤ 3}

theorem intersection_M_N : M ∩ N = {-1, 0, 1} := by sorry

end intersection_M_N_l3533_353375


namespace set_intersection_problem_l3533_353302

theorem set_intersection_problem (p q : ℝ) : 
  let M := {x : ℝ | x^2 - 5*x ≤ 0}
  let N := {x : ℝ | p < x ∧ x < 6}
  ({x : ℝ | 2 < x ∧ x ≤ q} = M ∩ N) → p + q = 7 := by
sorry

end set_intersection_problem_l3533_353302


namespace fiftyFourthCardIsSpadeTwo_l3533_353396

/-- Represents a playing card suit -/
inductive Suit
| Spades
| Hearts

/-- Represents a playing card value -/
inductive Value
| Ace | Two | Three | Four | Five | Six | Seven | Eight | Nine | Ten | Jack | Queen | King

/-- Represents a playing card -/
structure Card where
  suit : Suit
  value : Value

/-- The sequence of cards in order -/
def cardSequence : List Card := sorry

/-- The length of one complete cycle in the sequence -/
def cycleLength : Nat := 26

/-- Function to get the nth card in the sequence -/
def getNthCard (n : Nat) : Card := sorry

theorem fiftyFourthCardIsSpadeTwo : 
  getNthCard 54 = Card.mk Suit.Spades Value.Two := by sorry

end fiftyFourthCardIsSpadeTwo_l3533_353396


namespace complex_equation_solution_l3533_353343

theorem complex_equation_solution (z : ℂ) (h : (1 + Complex.I) / z = 1 - Complex.I) : z = Complex.I := by
  sorry

end complex_equation_solution_l3533_353343


namespace probability_not_black_l3533_353383

theorem probability_not_black (white black red : ℕ) (h1 : white = 7) (h2 : black = 6) (h3 : red = 4) :
  (white + red : ℚ) / (white + black + red) = 11 / 17 := by
  sorry

end probability_not_black_l3533_353383


namespace simplify_expression_l3533_353384

theorem simplify_expression (x : ℝ) : 3*x^2 + 4 - 5*x^3 - x^3 + 3 - 3*x^2 = -6*x^3 + 7 := by
  sorry

end simplify_expression_l3533_353384


namespace tangent_circle_condition_l3533_353311

/-- The line 2x + y - 2 = 0 is tangent to the circle (x - 1)^2 + (y - a)^2 = 1 -/
def is_tangent (a : ℝ) : Prop :=
  ∃ (x y : ℝ), (2 * x + y - 2 = 0) ∧ ((x - 1)^2 + (y - a)^2 = 1)

/-- If the line is tangent to the circle, then a = ± √5 -/
theorem tangent_circle_condition (a : ℝ) :
  is_tangent a → (a = Real.sqrt 5 ∨ a = -Real.sqrt 5) := by sorry

end tangent_circle_condition_l3533_353311


namespace gcd_special_numbers_l3533_353395

theorem gcd_special_numbers : 
  let m : ℕ := 555555555
  let n : ℕ := 1111111111
  Nat.gcd m n = 1 := by sorry

end gcd_special_numbers_l3533_353395


namespace sequence_bound_l3533_353303

theorem sequence_bound (x : ℕ → ℝ) (c : ℝ) 
  (h1 : ∀ (i j : ℕ), i ≠ j → |x i - x j| ≥ 1 / (i + j))
  (h2 : ∀ (i : ℕ), 0 ≤ x i ∧ x i ≤ c) : 
  c ≥ 1 := by
sorry

end sequence_bound_l3533_353303


namespace arithmetic_sequence_sum_l3533_353389

theorem arithmetic_sequence_sum (a : ℕ → ℕ) : 
  (∀ n, a (n + 1) - a n = a 1 - a 0) →  -- arithmetic sequence condition
  a 0 = 3 →                            -- first term is 3
  a 1 = 9 →                            -- second term is 9
  a 6 = 33 →                           -- last (seventh) term is 33
  a 4 + a 5 = 60 :=                    -- sum of fifth and sixth terms is 60
by sorry

end arithmetic_sequence_sum_l3533_353389


namespace min_dot_product_vectors_l3533_353304

/-- The dot product of vectors (1, x) and (x, x+1) has a minimum value of -1 -/
theorem min_dot_product_vectors : 
  ∃ (min : ℝ), min = -1 ∧ 
  ∀ (x : ℝ), (1 * x + x * (x + 1)) ≥ min :=
by sorry

end min_dot_product_vectors_l3533_353304


namespace diana_statues_l3533_353328

/-- Given the amount of paint available and the amount required per statue, 
    calculate the number of statues that can be painted. -/
def statues_paintable (paint_available : ℚ) (paint_per_statue : ℚ) : ℚ :=
  paint_available / paint_per_statue

/-- Theorem: Diana can paint 2 statues with the remaining paint. -/
theorem diana_statues : 
  let paint_available : ℚ := 1/2
  let paint_per_statue : ℚ := 1/4
  statues_paintable paint_available paint_per_statue = 2 := by
  sorry

end diana_statues_l3533_353328


namespace warehouse_theorem_l3533_353323

def warehouse_problem (second_floor_space : ℝ) (boxes_space : ℝ) : Prop :=
  let first_floor_space := 2 * second_floor_space
  let total_space := first_floor_space + second_floor_space
  let available_space := total_space - boxes_space
  (boxes_space = 5000) ∧
  (boxes_space = second_floor_space / 4) ∧
  (available_space = 55000)

theorem warehouse_theorem :
  ∃ (second_floor_space : ℝ), warehouse_problem second_floor_space 5000 :=
sorry

end warehouse_theorem_l3533_353323


namespace physics_score_l3533_353352

/-- Represents the scores in physics, chemistry, and mathematics --/
structure Scores where
  physics : ℝ
  chemistry : ℝ
  mathematics : ℝ

/-- The average score of all three subjects is 65 --/
def average_all (s : Scores) : Prop :=
  (s.physics + s.chemistry + s.mathematics) / 3 = 65

/-- The average score of physics and mathematics is 90 --/
def average_physics_math (s : Scores) : Prop :=
  (s.physics + s.mathematics) / 2 = 90

/-- The average score of physics and chemistry is 70 --/
def average_physics_chem (s : Scores) : Prop :=
  (s.physics + s.chemistry) / 2 = 70

/-- Given the conditions, prove that the score in physics is 125 --/
theorem physics_score (s : Scores) 
  (h1 : average_all s) 
  (h2 : average_physics_math s) 
  (h3 : average_physics_chem s) : 
  s.physics = 125 := by
  sorry

end physics_score_l3533_353352


namespace tshirt_pricing_l3533_353398

def first_batch_cost : ℝ := 4000
def second_batch_cost : ℝ := 8800
def cost_difference : ℝ := 4
def discounted_quantity : ℕ := 40
def discount_rate : ℝ := 0.3
def min_profit_margin : ℝ := 0.8

def cost_price_first_batch : ℝ := 40
def cost_price_second_batch : ℝ := 44
def min_retail_price : ℝ := 80

theorem tshirt_pricing :
  let first_quantity := first_batch_cost / cost_price_first_batch
  let second_quantity := second_batch_cost / cost_price_second_batch
  let total_quantity := first_quantity + second_quantity
  (2 * first_quantity = second_quantity) ∧
  (cost_price_second_batch = cost_price_first_batch + cost_difference) ∧
  (min_retail_price * (total_quantity - discounted_quantity) +
   min_retail_price * (1 - discount_rate) * discounted_quantity ≥
   (first_batch_cost + second_batch_cost) * (1 + min_profit_margin)) :=
by sorry

end tshirt_pricing_l3533_353398


namespace simplify_square_roots_l3533_353394

theorem simplify_square_roots : 
  Real.sqrt 10 - Real.sqrt 40 + Real.sqrt 90 = 2 * Real.sqrt 10 := by
  sorry

end simplify_square_roots_l3533_353394


namespace prime_triplet_equation_l3533_353378

theorem prime_triplet_equation : 
  ∀ p q r : ℕ, 
    Prime p → Prime q → Prime r → 
    p^q + q^p = r → 
    ((p = 2 ∧ q = 3 ∧ r = 17) ∨ (p = 3 ∧ q = 2 ∧ r = 17)) := by
  sorry

end prime_triplet_equation_l3533_353378


namespace no_special_pentagon_l3533_353314

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a pentagon as a set of 5 points
def Pentagon : Type := { p : Finset Point3D // p.card = 5 }

-- Define a function to check if three points are colinear
def areColinear (p q r : Point3D) : Prop := sorry

-- Define a function to check if a point is in the interior of a triangle
def isInteriorPoint (p : Point3D) (t1 t2 t3 : Point3D) : Prop := sorry

-- Define a function to check if a line segment intersects a plane at an interior point of a triangle
def intersectsTriangleInterior (p1 p2 t1 t2 t3 : Point3D) : Prop := sorry

-- Main theorem
theorem no_special_pentagon : 
  ¬ ∃ (pent : Pentagon), 
    ∀ (v1 v2 v3 v4 v5 : Point3D),
      v1 ∈ pent.val → v2 ∈ pent.val → v3 ∈ pent.val → v4 ∈ pent.val → v5 ∈ pent.val →
      v1 ≠ v2 → v1 ≠ v3 → v1 ≠ v4 → v1 ≠ v5 → v2 ≠ v3 → v2 ≠ v4 → v2 ≠ v5 → v3 ≠ v4 → v3 ≠ v5 → v4 ≠ v5 →
      (intersectsTriangleInterior v1 v3 v2 v4 v5 ∧
       intersectsTriangleInterior v1 v4 v2 v3 v5 ∧
       intersectsTriangleInterior v2 v4 v1 v3 v5 ∧
       intersectsTriangleInterior v2 v5 v1 v3 v4 ∧
       intersectsTriangleInterior v3 v5 v1 v2 v4) :=
by sorry


end no_special_pentagon_l3533_353314


namespace cylinder_volume_equals_54_sqrt3_over_sqrt_pi_l3533_353333

/-- Given a cube with side length 3 and a cylinder with the same surface area as the cube,
    where the cylinder's height equals its diameter, prove that the volume of the cylinder
    is 54 * sqrt(3) / sqrt(π). -/
theorem cylinder_volume_equals_54_sqrt3_over_sqrt_pi
  (cube_side : ℝ)
  (cylinder_radius : ℝ)
  (cylinder_height : ℝ)
  (h1 : cube_side = 3)
  (h2 : 6 * cube_side^2 = 2 * π * cylinder_radius^2 + 2 * π * cylinder_radius * cylinder_height)
  (h3 : cylinder_height = 2 * cylinder_radius) :
  π * cylinder_radius^2 * cylinder_height = 54 * Real.sqrt 3 / Real.sqrt π :=
by sorry


end cylinder_volume_equals_54_sqrt3_over_sqrt_pi_l3533_353333


namespace f_plus_one_nonnegative_min_a_value_l3533_353388

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x * (Real.log x - 1)

-- Theorem 1: f(x) + 1 ≥ 0 for all x > 0
theorem f_plus_one_nonnegative : ∀ x > 0, f x + 1 ≥ 0 := by sorry

-- Theorem 2: The minimum value of a such that 4f'(x) ≤ a(x+1) - 8 for all x > 0 is 4
theorem min_a_value : 
  (∃ a : ℝ, ∀ x > 0, 4 * (Real.log x) ≤ a * (x + 1) - 8) ∧ 
  (∀ a < 4, ∃ x > 0, 4 * (Real.log x) > a * (x + 1) - 8) := by sorry

end f_plus_one_nonnegative_min_a_value_l3533_353388


namespace total_profit_is_6300_l3533_353341

/-- Represents the profit sharing scenario between Tom and Jose -/
structure ProfitSharing where
  tom_investment : ℕ
  tom_months : ℕ
  jose_investment : ℕ
  jose_months : ℕ
  jose_profit : ℕ

/-- Calculates the total profit based on the given profit sharing scenario -/
def calculate_total_profit (ps : ProfitSharing) : ℕ :=
  let tom_investment_months := ps.tom_investment * ps.tom_months
  let jose_investment_months := ps.jose_investment * ps.jose_months
  let ratio_denominator := tom_investment_months + jose_investment_months
  let tom_profit := (tom_investment_months * ps.jose_profit) / jose_investment_months
  tom_profit + ps.jose_profit

/-- Theorem stating that the total profit for the given scenario is 6300 -/
theorem total_profit_is_6300 (ps : ProfitSharing) 
  (h1 : ps.tom_investment = 3000) 
  (h2 : ps.tom_months = 12) 
  (h3 : ps.jose_investment = 4500) 
  (h4 : ps.jose_months = 10) 
  (h5 : ps.jose_profit = 3500) : 
  calculate_total_profit ps = 6300 := by
  sorry

end total_profit_is_6300_l3533_353341


namespace loss_percentage_calculation_l3533_353327

def cost_price : ℝ := 1500
def selling_price : ℝ := 1335

theorem loss_percentage_calculation :
  (cost_price - selling_price) / cost_price * 100 = 11 := by
sorry

end loss_percentage_calculation_l3533_353327


namespace ln_inequality_implies_inequality_l3533_353364

theorem ln_inequality_implies_inequality (a b : ℝ) : 
  Real.log a > Real.log b → a > b := by sorry

end ln_inequality_implies_inequality_l3533_353364


namespace min_x_plus_y_min_value_is_9_4_min_achieved_l3533_353373

-- Define the optimization problem
theorem min_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 4 / y + 1 / x = 4) :
  ∀ x' y' : ℝ, x' > 0 → y' > 0 → 4 / y' + 1 / x' = 4 → x + y ≤ x' + y' :=
by sorry

-- State the minimum value
theorem min_value_is_9_4 (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 4 / y + 1 / x = 4) :
  x + y ≥ 9 / 4 :=
by sorry

-- Prove the minimum is achieved
theorem min_achieved (ε : ℝ) (hε : ε > 0) :
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 4 / y + 1 / x = 4 ∧ x + y < 9 / 4 + ε :=
by sorry

end min_x_plus_y_min_value_is_9_4_min_achieved_l3533_353373


namespace max_product_sum_200_l3533_353370

theorem max_product_sum_200 : 
  ∀ x y : ℤ, x + y = 200 → x * y ≤ 10000 := by
  sorry

end max_product_sum_200_l3533_353370


namespace sam_filled_four_bags_saturday_l3533_353310

/-- The number of bags Sam filled on Saturday -/
def saturday_bags : ℕ := sorry

/-- The number of bags Sam filled on Sunday -/
def sunday_bags : ℕ := 3

/-- The number of cans in each bag -/
def cans_per_bag : ℕ := 6

/-- The total number of cans collected -/
def total_cans : ℕ := 42

/-- Theorem stating that Sam filled 4 bags on Saturday -/
theorem sam_filled_four_bags_saturday : saturday_bags = 4 := by
  sorry

end sam_filled_four_bags_saturday_l3533_353310


namespace x_value_l3533_353350

theorem x_value : ∃ x : ℚ, (10 * x = x + 20) ∧ (x = 20 / 9) := by sorry

end x_value_l3533_353350


namespace simplify_and_rationalize_l3533_353385

theorem simplify_and_rationalize (x : ℝ) :
  1 / (1 + 1 / (Real.sqrt 5 + 2)) = (Real.sqrt 5 + 1) / 4 := by
  sorry

end simplify_and_rationalize_l3533_353385


namespace new_person_weight_l3533_353363

/-- Given a group of 12 people where one person weighing 62 kg is replaced by a new person,
    causing the average weight to increase by 4.8 kg, prove that the new person weighs 119.6 kg. -/
theorem new_person_weight (initial_count : Nat) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 12 →
  weight_increase = 4.8 →
  replaced_weight = 62 →
  (initial_count : ℝ) * weight_increase + replaced_weight = 119.6 :=
by
  sorry

end new_person_weight_l3533_353363


namespace triangular_pyramid_distance_sum_l3533_353387

/-- A triangular pyramid with volume V, face areas (S₁, S₂, S₃, S₄), and distances (H₁, H₂, H₃, H₄) from any internal point Q to each face. -/
structure TriangularPyramid where
  V : ℝ
  S₁ : ℝ
  S₂ : ℝ
  S₃ : ℝ
  S₄ : ℝ
  H₁ : ℝ
  H₂ : ℝ
  H₃ : ℝ
  H₄ : ℝ
  K : ℝ
  volume_positive : V > 0
  areas_positive : S₁ > 0 ∧ S₂ > 0 ∧ S₃ > 0 ∧ S₄ > 0
  distances_positive : H₁ > 0 ∧ H₂ > 0 ∧ H₃ > 0 ∧ H₄ > 0
  K_positive : K > 0
  area_ratios : S₁ = K ∧ S₂ = 2*K ∧ S₃ = 3*K ∧ S₄ = 4*K

/-- The theorem stating the relationship between distances, volume, and K for a triangular pyramid. -/
theorem triangular_pyramid_distance_sum (p : TriangularPyramid) :
  p.H₁ + 2*p.H₂ + 3*p.H₃ + 4*p.H₄ = 3*p.V/p.K :=
by sorry

end triangular_pyramid_distance_sum_l3533_353387


namespace algae_cells_after_ten_days_l3533_353362

def algae_growth (initial_cells : ℕ) (split_factor : ℕ) (days : ℕ) : ℕ :=
  initial_cells * split_factor ^ days

theorem algae_cells_after_ten_days :
  algae_growth 1 3 10 = 59049 := by
  sorry

end algae_cells_after_ten_days_l3533_353362


namespace two_polygons_exist_l3533_353305

/-- Represents a polygon with a given number of sides. -/
structure Polygon where
  sides : ℕ

/-- Calculates the sum of interior angles of a polygon. -/
def sumInteriorAngles (p : Polygon) : ℕ :=
  (p.sides - 2) * 180

/-- Calculates the number of diagonals in a polygon. -/
def numDiagonals (p : Polygon) : ℕ :=
  p.sides * (p.sides - 3) / 2

/-- Theorem stating the existence of two polygons satisfying the given conditions. -/
theorem two_polygons_exist : ∃ (p1 p2 : Polygon),
  (sumInteriorAngles p1 + sumInteriorAngles p2 = 1260) ∧
  (numDiagonals p1 + numDiagonals p2 = 14) ∧
  ((p1.sides = 6 ∧ p2.sides = 5) ∨ (p1.sides = 5 ∧ p2.sides = 6)) := by
  sorry

end two_polygons_exist_l3533_353305


namespace frustum_volume_l3533_353301

/-- The volume of a frustum formed by cutting a square pyramid parallel to its base -/
theorem frustum_volume (base_edge : ℝ) (altitude : ℝ) (small_base_edge : ℝ) (small_altitude : ℝ) :
  base_edge = 16 →
  altitude = 10 →
  small_base_edge = 8 →
  small_altitude = 5 →
  let original_volume := (1 / 3) * base_edge^2 * altitude
  let small_volume := (1 / 3) * small_base_edge^2 * small_altitude
  original_volume - small_volume = 2240 / 3 :=
by sorry

end frustum_volume_l3533_353301


namespace final_expression_l3533_353379

theorem final_expression (y : ℝ) : 3 * (1/2 * (12*y + 3)) = 18*y + 4.5 := by
  sorry

end final_expression_l3533_353379


namespace generate_numbers_l3533_353351

/-- A type representing arithmetic expressions using five 3's -/
inductive Expr
  | three : Expr
  | add : Expr → Expr → Expr
  | sub : Expr → Expr → Expr
  | mul : Expr → Expr → Expr
  | div : Expr → Expr → Expr
  | pow : Expr → Expr → Expr

/-- Evaluate an expression to a rational number -/
def eval : Expr → ℚ
  | Expr.three => 3
  | Expr.add e1 e2 => eval e1 + eval e2
  | Expr.sub e1 e2 => eval e1 - eval e2
  | Expr.mul e1 e2 => eval e1 * eval e2
  | Expr.div e1 e2 => eval e1 / eval e2
  | Expr.pow e1 e2 => (eval e1) ^ (eval e2).num

/-- Count the number of 3's used in an expression -/
def count_threes : Expr → Nat
  | Expr.three => 1
  | Expr.add e1 e2 => count_threes e1 + count_threes e2
  | Expr.sub e1 e2 => count_threes e1 + count_threes e2
  | Expr.mul e1 e2 => count_threes e1 + count_threes e2
  | Expr.div e1 e2 => count_threes e1 + count_threes e2
  | Expr.pow e1 e2 => count_threes e1 + count_threes e2

/-- The main theorem stating that all integers from 1 to 39 can be generated -/
theorem generate_numbers :
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 39 →
  ∃ e : Expr, count_threes e = 5 ∧ eval e = n := by sorry

end generate_numbers_l3533_353351


namespace max_k_for_circle_intersection_l3533_353326

/-- The maximum value of k for which a circle with radius 1 centered on the line y = kx - 2
    has a common point with the circle x^2 + y^2 - 8x + 15 = 0 is 4/3 -/
theorem max_k_for_circle_intersection :
  let C : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 - 8*p.1 + 15 = 0}
  let line (k : ℝ) : Set (ℝ × ℝ) := {p | p.2 = k*p.1 - 2}
  let unit_circle_on_line (k : ℝ) : Set (Set (ℝ × ℝ)) :=
    {S | ∃ c ∈ line k, S = {p | (p.1 - c.1)^2 + (p.2 - c.2)^2 = 1}}
  ∀ k > 4/3, ∀ S ∈ unit_circle_on_line k, S ∩ C = ∅ ∧
  ∃ S ∈ unit_circle_on_line (4/3), S ∩ C ≠ ∅ :=
by sorry

end max_k_for_circle_intersection_l3533_353326


namespace compute_expression_l3533_353361

theorem compute_expression : 8 * (1 / 3)^3 - 1 = -19 / 27 := by
  sorry

end compute_expression_l3533_353361


namespace translator_selection_count_l3533_353321

/-- Represents the number of translators for each category -/
structure TranslatorCounts where
  total : Nat
  english : Nat
  japanese : Nat
  both : Nat

/-- Represents the required number of translators for each language -/
structure RequiredTranslators where
  english : Nat
  japanese : Nat

/-- Calculates the number of ways to select translators given the constraints -/
def countTranslatorSelections (counts : TranslatorCounts) (required : RequiredTranslators) : Nat :=
  sorry

/-- Theorem stating that there are 29 different ways to select the translators -/
theorem translator_selection_count :
  let counts : TranslatorCounts := ⟨8, 3, 3, 2⟩
  let required : RequiredTranslators := ⟨3, 2⟩
  countTranslatorSelections counts required = 29 :=
by sorry

end translator_selection_count_l3533_353321


namespace equation_solution_l3533_353357

theorem equation_solution : ∃ n : ℕ, 3^n * 9^n = 81^(n-12) ∧ n = 48 := by
  sorry

end equation_solution_l3533_353357


namespace solution_set_f_less_than_6_range_of_m_for_f_geq_m_squared_minus_3m_l3533_353377

-- Define the function f
def f (x : ℝ) : ℝ := |x + 2| + |2*x - 4|

-- Theorem for the solution set of f(x) < 6
theorem solution_set_f_less_than_6 :
  {x : ℝ | f x < 6} = {x : ℝ | 0 < x ∧ x < 8/3} := by sorry

-- Theorem for the range of m
theorem range_of_m_for_f_geq_m_squared_minus_3m :
  {m : ℝ | ∀ x, f x ≥ m^2 - 3*m} = {m : ℝ | -1 ≤ m ∧ m ≤ 4} := by sorry

end solution_set_f_less_than_6_range_of_m_for_f_geq_m_squared_minus_3m_l3533_353377


namespace arithmetic_seq_common_diff_l3533_353371

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ       -- Common difference
  S : ℕ → ℝ  -- Sum function
  seq_def : ∀ n, a (n + 1) = a n + d
  sum_def : ∀ n, S n = (n : ℝ) * (2 * a 1 + (n - 1) * d) / 2

/-- If 2S₃ = 3S₂ + 6 for an arithmetic sequence, then the common difference is 2 -/
theorem arithmetic_seq_common_diff (seq : ArithmeticSequence) 
    (h : 2 * seq.S 3 = 3 * seq.S 2 + 6) : seq.d = 2 := by
  sorry

end arithmetic_seq_common_diff_l3533_353371


namespace hyperbola_m_range_l3533_353306

-- Define the propositions p and q
def p (t : ℝ) : Prop := ∃ (x y : ℝ), x^2 / (t + 2) + y^2 / (t - 10) = 1

def q (t m : ℝ) : Prop := -m < t ∧ t < m + 1 ∧ m > 0

-- Define the sufficient but not necessary condition
def sufficient_not_necessary (p q : Prop) : Prop :=
  (q → p) ∧ ¬(p → q)

-- Theorem statement
theorem hyperbola_m_range :
  (∀ t, sufficient_not_necessary (p t) (∃ m, q t m)) →
  ∀ m, (m > 0 ∧ m ≤ 2) ↔ (∃ t, q t m ∧ p t) :=
sorry

end hyperbola_m_range_l3533_353306


namespace chord_length_perpendicular_chord_m_l3533_353392

-- Define the circle equation
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y + m = 0

-- Define the line equations
def line1_equation (x y : ℝ) : Prop :=
  x + y - 1 = 0

def line2_equation (x y : ℝ) : Prop :=
  x + 2*y - 4 = 0

-- Part 1: Chord length
theorem chord_length : 
  ∀ x y : ℝ, circle_equation x y 1 ∧ line1_equation x y → 
  ∃ chord_length : ℝ, chord_length = 2 * Real.sqrt 2 :=
sorry

-- Part 2: Value of m
theorem perpendicular_chord_m :
  ∃ m : ℝ, ∀ x1 y1 x2 y2 : ℝ,
    circle_equation x1 y1 m ∧ circle_equation x2 y2 m ∧
    line2_equation x1 y1 ∧ line2_equation x2 y2 ∧
    x1 * x2 + y1 * y2 = 0 →
    m = 8/5 :=
sorry

end chord_length_perpendicular_chord_m_l3533_353392


namespace hyperbola_standard_equation_l3533_353382

/-- The standard equation of a hyperbola given specific conditions -/
theorem hyperbola_standard_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1) →
  (1^2 + 2^2 = (2 * a^2 - 1)) →
  (b / a = 2) →
  (∃ (x y : ℝ), x^2 - y^2 / 4 = 1) :=
by sorry

end hyperbola_standard_equation_l3533_353382


namespace tangent_line_to_parabola_l3533_353390

/-- The value of d for which the line y = 3x + d is tangent to the parabola y^2 = 12x -/
theorem tangent_line_to_parabola : ∃ d : ℝ, 
  (∀ x y : ℝ, y = 3*x + d → y^2 = 12*x → 
    ∀ ε > 0, ∃ δ > 0, ∀ x' y' : ℝ, 
      (x' - x)^2 + (y' - y)^2 < δ^2 → 
      (y' - (3*x' + d))^2 > ε^2 * ((y')^2 - 12*x')) ∧
  d = 1 :=
sorry

end tangent_line_to_parabola_l3533_353390


namespace polynomial_symmetry_l3533_353342

/-- Given a polynomial function f(x) = ax^5 + bx^3 + cx + 7 where a, b, c are real constants,
    if f(-2011) = -17, then f(2011) = 31 -/
theorem polynomial_symmetry (a b c : ℝ) :
  let f := λ x : ℝ => a * x^5 + b * x^3 + c * x + 7
  (f (-2011) = -17) → (f 2011 = 31) := by
  sorry

end polynomial_symmetry_l3533_353342


namespace plane_intersection_properties_l3533_353344

-- Define the planes
variable (α β γ : Set (ℝ × ℝ × ℝ))

-- Define the perpendicularity and intersection relations
def perpendicular (p q : Set (ℝ × ℝ × ℝ)) : Prop := sorry
def intersects (p q : Set (ℝ × ℝ × ℝ)) : Prop := sorry
def intersects_not_perpendicularly (p q : Set (ℝ × ℝ × ℝ)) : Prop := sorry

-- Define what it means for a line to be in a plane
def line_in_plane (l : Set (ℝ × ℝ × ℝ)) (p : Set (ℝ × ℝ × ℝ)) : Prop := sorry

-- Define parallel and perpendicular for lines and planes
def line_parallel_to_plane (l : Set (ℝ × ℝ × ℝ)) (p : Set (ℝ × ℝ × ℝ)) : Prop := sorry
def line_perpendicular_to_plane (l : Set (ℝ × ℝ × ℝ)) (p : Set (ℝ × ℝ × ℝ)) : Prop := sorry

-- State the theorem
theorem plane_intersection_properties 
  (h1 : perpendicular β γ)
  (h2 : intersects_not_perpendicularly α γ) :
  (∃ (a : Set (ℝ × ℝ × ℝ)), line_in_plane a α ∧ line_parallel_to_plane a γ) ∧
  (∃ (c : Set (ℝ × ℝ × ℝ)), line_in_plane c γ ∧ line_perpendicular_to_plane c β) := by
  sorry

end plane_intersection_properties_l3533_353344


namespace expression_factorization_l3533_353368

theorem expression_factorization (y : ℝ) :
  (16 * y^6 + 36 * y^4 - 9) - (4 * y^6 - 9 * y^4 + 9) = 3 * (4 * y^6 + 15 * y^4 - 6) := by
  sorry

end expression_factorization_l3533_353368


namespace solution_set_and_range_of_a_l3533_353334

def f (a x : ℝ) : ℝ := |x - a| + x

theorem solution_set_and_range_of_a :
  (∀ x : ℝ, f 3 x ≥ x + 4 ↔ (x ≤ -1 ∨ x ≥ 7)) ∧
  (∀ a : ℝ, a > 0 →
    (∀ x : ℝ, x ∈ Set.Icc 1 3 → f a x ≥ x + 2 * a^2) ↔
    (-1 ≤ a ∧ a ≤ 1/2)) := by
  sorry

end solution_set_and_range_of_a_l3533_353334


namespace f_properties_l3533_353318

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (1/2) * m * (x - 1)^2 - 2*x + 3 + Real.log x

theorem f_properties (m : ℝ) (h : m ≥ 1) :
  (∃ a b, a > 0 ∧ b > 0 ∧ a < b ∧ ∀ x ∈ Set.Icc a b, (deriv (f m)) x ≤ 0) ∧
  (∃! m, ∀ x, x > 0 → (f m x = -x + 2 → x = 1)) ∧
  (∀ x, x > 0 → (f 1 x = -x + 2 → x = 1)) := by sorry

end f_properties_l3533_353318


namespace fib_last_four_zeros_exist_l3533_353309

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib n + fib (n + 1)

/-- Last four digits of a natural number -/
def lastFourDigits (n : ℕ) : ℕ :=
  n % 10000

/-- Theorem: There exists a term in the first 100,000,001 Fibonacci numbers whose last four digits are all zeros -/
theorem fib_last_four_zeros_exist : ∃ n : ℕ, n < 100000001 ∧ lastFourDigits (fib n) = 0 := by
  sorry


end fib_last_four_zeros_exist_l3533_353309


namespace inscribed_squares_area_ratio_l3533_353365

/-- The ratio of the area of a square inscribed in a quarter-circle to the area of a square inscribed in a full circle, both with radius r, is 1/4. -/
theorem inscribed_squares_area_ratio (r : ℝ) (hr : r > 0) :
  let s1 := r / Real.sqrt 2
  let s2 := r * Real.sqrt 2
  (s1 ^ 2) / (s2 ^ 2) = 1 / 4 := by
sorry

end inscribed_squares_area_ratio_l3533_353365


namespace point_P_properties_l3533_353345

def P (a : ℝ) : ℝ × ℝ := (-3*a - 4, 2 + a)

def Q : ℝ × ℝ := (5, 8)

theorem point_P_properties :
  (∀ a : ℝ, P a = (2, 0) → (P a).2 = 0) ∧
  (∀ a : ℝ, (P a).1 = Q.1 → P a = (5, -1)) := by
  sorry

end point_P_properties_l3533_353345


namespace pie_crust_flour_usage_l3533_353393

theorem pie_crust_flour_usage 
  (original_crusts : ℕ) 
  (original_flour_per_crust : ℚ) 
  (new_crusts : ℕ) :
  original_crusts = 30 →
  original_flour_per_crust = 1/6 →
  new_crusts = 25 →
  (original_crusts * original_flour_per_crust) / new_crusts = 1/5 :=
by sorry

end pie_crust_flour_usage_l3533_353393


namespace straight_angle_average_l3533_353331

theorem straight_angle_average (p q r s t : ℝ) : 
  p + q + r + s + t = 180 → (p + q + r + s + t) / 5 = 36 := by
  sorry

end straight_angle_average_l3533_353331


namespace tower_remainder_l3533_353399

/-- Represents the number of towers that can be built with cubes of sizes 1 to n -/
def T : ℕ → ℕ
| 0 => 1
| 1 => 1
| 2 => 6
| 3 => 18
| n+4 => 4 * T (n+3)

/-- The main theorem stating the result for 9 cubes -/
theorem tower_remainder : T 9 % 1000 = 296 := by
  sorry


end tower_remainder_l3533_353399


namespace certain_fraction_proof_l3533_353340

theorem certain_fraction_proof (x y : ℚ) :
  (x / y) / (3 / 7) = 0.46666666666666673 / (1 / 2) →
  x / y = 0.4 := by
sorry

end certain_fraction_proof_l3533_353340


namespace a_grade_implies_conditions_l3533_353325

-- Define the conditions for receiving an A
def receivesA (score : ℝ) (submittedAll : Bool) : Prop :=
  score ≥ 90 ∧ submittedAll

-- Define the theorem
theorem a_grade_implies_conditions 
  (score : ℝ) (submittedAll : Bool) :
  receivesA score submittedAll → 
  (score ≥ 90 ∧ submittedAll) :=
by
  sorry

-- The proof is omitted as per instructions

end a_grade_implies_conditions_l3533_353325


namespace expression_simplification_l3533_353307

theorem expression_simplification (x y : ℝ) 
  (hx : x = Real.sqrt 2) 
  (hy : y = 2 * Real.sqrt 2) : 
  (4 * y^2 - x^2) / (x^2 + 2*x*y + y^2) / ((x - 2*y) / (2*x^2 + 2*x*y)) = -10 * Real.sqrt 2 / 3 := by
  sorry

end expression_simplification_l3533_353307


namespace jackies_lotion_order_l3533_353380

/-- The number of lotion bottles Jackie ordered -/
def lotion_bottles : ℕ := 3

/-- The free shipping threshold in cents -/
def free_shipping_threshold : ℕ := 5000

/-- The total cost of shampoo and conditioner in cents -/
def shampoo_conditioner_cost : ℕ := 2000

/-- The cost of one bottle of lotion in cents -/
def lotion_cost : ℕ := 600

/-- The additional amount Jackie needs to spend to reach the free shipping threshold in cents -/
def additional_spend : ℕ := 1200

theorem jackies_lotion_order :
  lotion_bottles * lotion_cost = free_shipping_threshold - shampoo_conditioner_cost - additional_spend :=
by sorry

end jackies_lotion_order_l3533_353380


namespace correct_selling_price_l3533_353346

/-- The markup percentage applied to the cost price -/
def markup : ℚ := 25 / 100

/-- The cost price of the computer table in rupees -/
def cost_price : ℕ := 6672

/-- The selling price of the computer table in rupees -/
def selling_price : ℕ := 8340

/-- Theorem stating that the selling price is correct given the cost price and markup -/
theorem correct_selling_price : 
  (cost_price : ℚ) * (1 + markup) = selling_price := by sorry

end correct_selling_price_l3533_353346


namespace negative_square_of_two_l3533_353360

theorem negative_square_of_two : -2^2 = -4 := by
  sorry

end negative_square_of_two_l3533_353360


namespace sqrt_abs_sum_zero_implies_power_l3533_353337

theorem sqrt_abs_sum_zero_implies_power (x y : ℝ) :
  Real.sqrt (2 * x + 8) + |y - 3| = 0 → (x + y)^2021 = -1 := by
  sorry

end sqrt_abs_sum_zero_implies_power_l3533_353337


namespace candidate_selection_probability_l3533_353335

/-- Represents the probability distribution of Excel skills among job candidates -/
structure ExcelSkills where
  beginner : ℝ
  intermediate : ℝ
  advanced : ℝ
  none : ℝ
  sum_to_one : beginner + intermediate + advanced + none = 1

/-- Represents the probability distribution of shift preferences among job candidates -/
structure ShiftPreference where
  day : ℝ
  night : ℝ
  sum_to_one : day + night = 1

/-- Represents the probability distribution of weekend work preferences among job candidates -/
structure WeekendPreference where
  willing : ℝ
  not_willing : ℝ
  sum_to_one : willing + not_willing = 1

/-- Theorem stating the probability of selecting a candidate with specific characteristics -/
theorem candidate_selection_probability 
  (excel : ExcelSkills)
  (shift : ShiftPreference)
  (weekend : WeekendPreference)
  (h1 : excel.beginner = 0.35)
  (h2 : excel.intermediate = 0.25)
  (h3 : excel.advanced = 0.2)
  (h4 : excel.none = 0.2)
  (h5 : shift.day = 0.7)
  (h6 : shift.night = 0.3)
  (h7 : weekend.willing = 0.4)
  (h8 : weekend.not_willing = 0.6) :
  (excel.intermediate + excel.advanced) * shift.night * weekend.not_willing = 0.081 := by
  sorry

end candidate_selection_probability_l3533_353335


namespace product_of_two_numbers_l3533_353315

theorem product_of_two_numbers (x y : ℝ) 
  (h1 : x * y = 15 * (x - y)) 
  (h2 : x + y = 8 * (x - y)) : 
  x * y = 100 / 7 := by
sorry

end product_of_two_numbers_l3533_353315


namespace solution_set_of_equation_l3533_353317

theorem solution_set_of_equation (x : ℝ) : 
  (16 * Real.sin (π * x) * Real.cos (π * x) = 16 * x + 1 / x) ↔ (x = 1/4 ∨ x = -1/4) :=
sorry

end solution_set_of_equation_l3533_353317


namespace de_bruijn_semi_integer_l3533_353355

/-- A semi-integer rectangle is a rectangle where at least one vertex has integer coordinates -/
def SemiIntegerRectangle (d : ℕ) := (Fin d → ℝ) → Prop

/-- A box with dimensions B₁, ..., Bₗ -/
def Box (l : ℕ) (B : Fin l → ℝ) := Set (Fin l → ℝ)

/-- A block with dimensions b₁, ..., bₖ -/
def Block (k : ℕ) (b : Fin k → ℝ) := Set (Fin k → ℝ)

/-- A tiling of a box by blocks -/
def Tiling (l k : ℕ) (B : Fin l → ℝ) (b : Fin k → ℝ) := 
  Box l B → Set (Block k b)

theorem de_bruijn_semi_integer 
  (l k : ℕ) (B : Fin l → ℝ) (b : Fin k → ℝ) 
  (tiling : Tiling l k B b) 
  (semi_int : SemiIntegerRectangle k) :
  ∀ i, ∃ j, ∃ (n : ℕ), B j = n * b i :=
sorry

end de_bruijn_semi_integer_l3533_353355


namespace inscribed_circle_radius_l3533_353312

theorem inscribed_circle_radius (a b c : ℝ) (h1 : a = 8) (h2 : b = 15) (h3 : c = 17) :
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  area / s = 3 := by sorry

end inscribed_circle_radius_l3533_353312


namespace infinite_series_sum_l3533_353339

open Real
open BigOperators

theorem infinite_series_sum : 
  (∑' n : ℕ, (3 * n + 2) / (n * (n + 1) * (n + 3))) = 10/3 := by sorry

end infinite_series_sum_l3533_353339


namespace correct_matching_probability_l3533_353381

-- Define the number of students and pictures
def num_students : ℕ := 4

-- Define the function to calculate the factorial
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- Define the total number of possible arrangements
def total_arrangements : ℕ := factorial num_students

-- Define the number of correct arrangements
def correct_arrangements : ℕ := 1

-- State the theorem
theorem correct_matching_probability :
  (correct_arrangements : ℚ) / total_arrangements = 1 / 24 := by
  sorry

end correct_matching_probability_l3533_353381


namespace recipe_multiplier_is_six_l3533_353320

/-- Represents the ratio of butter to flour in a recipe -/
structure RecipeRatio where
  butter : ℚ
  flour : ℚ

/-- The original recipe ratio -/
def originalRatio : RecipeRatio := { butter := 2, flour := 5 }

/-- The amount of butter used in the new recipe -/
def newButterAmount : ℚ := 12

/-- Calculates how many times the original recipe is being made -/
def recipeMultiplier (original : RecipeRatio) (newButter : ℚ) : ℚ :=
  newButter / original.butter

theorem recipe_multiplier_is_six :
  recipeMultiplier originalRatio newButterAmount = 6 := by
  sorry

#eval recipeMultiplier originalRatio newButterAmount

end recipe_multiplier_is_six_l3533_353320


namespace quadratic_inequality_solution_set_l3533_353316

theorem quadratic_inequality_solution_set : 
  {x : ℝ | x^2 - 50*x + 500 ≤ 9} = {x : ℝ | 13.42 ≤ x ∧ x ≤ 36.58} := by
sorry

end quadratic_inequality_solution_set_l3533_353316


namespace friends_total_amount_l3533_353391

/-- The total amount of money received by three friends from selling video games -/
def total_amount (zachary_games : ℕ) (price_per_game : ℕ) (jason_percent : ℕ) (ryan_extra : ℕ) : ℕ :=
  let zachary_amount := zachary_games * price_per_game
  let jason_amount := zachary_amount + (jason_percent * zachary_amount) / 100
  let ryan_amount := jason_amount + ryan_extra
  zachary_amount + jason_amount + ryan_amount

/-- Theorem stating that the total amount received by the three friends is $770 -/
theorem friends_total_amount :
  total_amount 40 5 30 50 = 770 := by
  sorry

end friends_total_amount_l3533_353391


namespace problem_solution_l3533_353374

-- Define the region D
def D : Set (ℝ × ℝ) := {(x, y) | (x - 1)^2 + (y - 2)^2 ≤ 4}

-- Define proposition p
def p : Prop := ∀ (x y : ℝ), (x, y) ∈ D → 2*x + y ≤ 8

-- Define proposition q
def q : Prop := ∃ (x y : ℝ), (x, y) ∈ D ∧ 2*x + y ≤ -1

-- Theorem to prove
theorem problem_solution : (¬p ∨ q) ∧ (¬p ∧ ¬q) := by sorry

end problem_solution_l3533_353374


namespace function_characterization_l3533_353347

def f (x a b : ℝ) : ℝ := (x + a) * (b * x + 2 * a)

theorem function_characterization (a b : ℝ) :
  (∀ x, f x a b = f (-x) a b) →  -- f is even
  (∀ y, y ∈ Set.Iic 4 → ∃ x, f x a b = y) →  -- range is (-∞, 4]
  (∀ y, ∃ x, f x a b = y → y ≤ 4) →  -- range is (-∞, 4]
  (∀ x, f x a b = -2 * x^2 + 4) :=
by sorry

end function_characterization_l3533_353347


namespace marble_fraction_l3533_353349

theorem marble_fraction (x : ℚ) : 
  let initial_blue := (2 : ℚ) / 3 * x
  let initial_red := x - initial_blue
  let new_red := 2 * initial_red
  let new_blue := (3 : ℚ) / 2 * initial_blue
  let total_new := new_red + new_blue
  new_red / total_new = (2 : ℚ) / 5 := by sorry

end marble_fraction_l3533_353349


namespace hyperbola_asymptotes_l3533_353336

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 2 = 1

-- Define the asymptote equations
def asymptote1 (x y : ℝ) : Prop := y = (Real.sqrt 2 / 2) * x
def asymptote2 (x y : ℝ) : Prop := y = -(Real.sqrt 2 / 2) * x

-- Theorem statement
theorem hyperbola_asymptotes :
  ∀ (x y : ℝ), hyperbola x y →
  (asymptote1 x y ∨ asymptote2 x y) :=
sorry

end hyperbola_asymptotes_l3533_353336


namespace hamburger_count_l3533_353367

/-- The total number of hamburgers made for lunch -/
def total_hamburgers (initial : ℕ) (additional : ℕ) : ℕ :=
  initial + additional

/-- Theorem stating that the total number of hamburgers is the sum of initial and additional -/
theorem hamburger_count (initial : ℕ) (additional : ℕ) :
  total_hamburgers initial additional = initial + additional :=
by sorry

end hamburger_count_l3533_353367


namespace first_group_has_four_weavers_l3533_353324

/-- The number of mat-weavers in the first group -/
def first_group_weavers : ℕ := 4

/-- The number of mats woven by the first group -/
def first_group_mats : ℕ := 4

/-- The number of days taken by the first group -/
def first_group_days : ℕ := 4

/-- The number of mat-weavers in the second group -/
def second_group_weavers : ℕ := 8

/-- The number of mats woven by the second group -/
def second_group_mats : ℕ := 16

/-- The number of days taken by the second group -/
def second_group_days : ℕ := 8

/-- The rate of weaving is the same for both groups -/
axiom same_rate : (first_group_mats : ℚ) / first_group_weavers / first_group_days = 
                  (second_group_mats : ℚ) / second_group_weavers / second_group_days

theorem first_group_has_four_weavers : first_group_weavers = 4 := by
  sorry

end first_group_has_four_weavers_l3533_353324


namespace problem_1_problem_2_problem_3_l3533_353397

-- Problem 1
theorem problem_1 (x : ℝ) : x * x^3 + x^2 * x^2 = 2 * x^4 := by sorry

-- Problem 2
theorem problem_2 (p q : ℝ) : (-p*q)^3 = -p^3 * q^3 := by sorry

-- Problem 3
theorem problem_3 (a : ℝ) : a^3 * a^4 * a + (a^2)^4 - (-2*a^4)^2 = -2 * a^8 := by sorry

end problem_1_problem_2_problem_3_l3533_353397


namespace sqrt_of_repeating_ones_100_l3533_353354

theorem sqrt_of_repeating_ones_100 :
  let x := (10^100 - 1) / (9 * 10^100)
  0.10049987498 < Real.sqrt x ∧ Real.sqrt x < 0.10049987499 := by
  sorry

end sqrt_of_repeating_ones_100_l3533_353354


namespace call_center_team_ratio_l3533_353376

theorem call_center_team_ratio (a b : ℚ) : 
  (∀ (c : ℚ), a * (3/5 * c) / (b * c) = 3/11) →
  a / b = 5/11 := by
sorry

end call_center_team_ratio_l3533_353376
