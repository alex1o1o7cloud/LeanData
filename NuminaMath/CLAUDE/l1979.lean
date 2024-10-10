import Mathlib

namespace area_of_triangle_perimeter_of_triangle_l1979_197904

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfiesConditions (t : Triangle) : Prop :=
  t.b^2 + t.c^2 - t.a^2 = t.b * t.c ∧ t.b * t.c = 1

-- Define the additional condition for part II
def satisfiesAdditionalCondition (t : Triangle) : Prop :=
  4 * Real.cos t.B * Real.cos t.C - 1 = 0

-- Theorem for part I
theorem area_of_triangle (t : Triangle) (h : satisfiesConditions t) :
  (1/2) * t.b * t.c * Real.sin t.A = Real.sqrt 3 / 4 := by
  sorry

-- Theorem for part II
theorem perimeter_of_triangle (t : Triangle) 
  (h1 : satisfiesConditions t) (h2 : satisfiesAdditionalCondition t) :
  t.a + t.b + t.c = 3 := by
  sorry

end area_of_triangle_perimeter_of_triangle_l1979_197904


namespace distance_between_tangent_circles_l1979_197968

/-- The distance between the centers of two externally tangent circles -/
def distance_between_centers (r1 r2 : ℝ) : ℝ := r1 + r2

/-- Two circles are externally tangent -/
axiom externally_tangent (O O' : Set ℝ) : Prop

theorem distance_between_tangent_circles 
  (O O' : Set ℝ) (r1 r2 : ℝ) 
  (h1 : externally_tangent O O')
  (h2 : r1 = 8)
  (h3 : r2 = 3) : 
  distance_between_centers r1 r2 = 11 := by
  sorry

end distance_between_tangent_circles_l1979_197968


namespace min_distance_to_point_l1979_197942

theorem min_distance_to_point (x y : ℝ) (h : 6 * x + 8 * y - 1 = 0) :
  ∃ (min : ℝ), min = 7 / 10 ∧ ∀ (x' y' : ℝ), 6 * x' + 8 * y' - 1 = 0 →
    Real.sqrt (x' ^ 2 + y' ^ 2 - 2 * y' + 1) ≥ min :=
by sorry

end min_distance_to_point_l1979_197942


namespace randy_initial_amount_l1979_197945

/-- Proves that Randy's initial amount is $6166.67 given the problem conditions --/
theorem randy_initial_amount :
  ∀ (initial : ℝ),
  (3/4 : ℝ) * (initial + 2900) - 1300 = 5500 →
  initial = 6166.67 := by
sorry

end randy_initial_amount_l1979_197945


namespace floor_tile_equations_l1979_197926

/-- Represents the floor tile purchase scenario -/
structure FloorTilePurchase where
  x : ℕ  -- number of colored floor tiles
  y : ℕ  -- number of single-color floor tiles
  colored_cost : ℕ := 24  -- cost of colored tiles in yuan
  single_cost : ℕ := 12   -- cost of single-color tiles in yuan
  total_cost : ℕ := 2220  -- total cost in yuan

/-- The system of equations correctly represents the floor tile purchase scenario -/
theorem floor_tile_equations (purchase : FloorTilePurchase) : 
  (purchase.colored_cost * purchase.x + purchase.single_cost * purchase.y = purchase.total_cost) ∧
  (purchase.y = 2 * purchase.x - 15) := by
  sorry

end floor_tile_equations_l1979_197926


namespace toys_sold_proof_l1979_197977

/-- The number of toys sold by a man -/
def number_of_toys : ℕ := 18

/-- The selling price of the toys -/
def selling_price : ℕ := 23100

/-- The cost price of one toy -/
def cost_price : ℕ := 1100

/-- The gain from the sale -/
def gain : ℕ := 3 * cost_price

theorem toys_sold_proof :
  number_of_toys * cost_price + gain = selling_price :=
by sorry

end toys_sold_proof_l1979_197977


namespace complex_number_in_first_quadrant_l1979_197919

theorem complex_number_in_first_quadrant : 
  let z : ℂ := 1 / (2 - Complex.I)
  0 < z.re ∧ 0 < z.im :=
by sorry

end complex_number_in_first_quadrant_l1979_197919


namespace function_identification_l1979_197915

/-- A first-degree function -/
def first_degree_function (f : ℝ → ℝ) : Prop :=
  ∃ k m : ℝ, ∀ x, f x = k * x + m

/-- A second-degree function -/
def second_degree_function (g : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, g x = a * x^2 + b * x + c

/-- Function composition equality -/
def composition_equality (f g : ℝ → ℝ) : Prop :=
  ∀ x, f (g x) = g (f x)

/-- Tangent to x-axis -/
def tangent_to_x_axis (g : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, g x = 0 ∧ ∀ y : ℝ, y ≠ x → g y > 0

/-- Tangent to another function -/
def tangent_to_function (f g : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, f x = g x ∧ ∀ y : ℝ, y ≠ x → f y ≠ g y

theorem function_identification
  (f g : ℝ → ℝ)
  (h1 : first_degree_function f)
  (h2 : second_degree_function g)
  (h3 : composition_equality f g)
  (h4 : tangent_to_x_axis g)
  (h5 : tangent_to_function f g)
  (h6 : g 0 = 1/16) :
  (∀ x, f x = x) ∧ (∀ x, g x = x^2 + 1/2 * x + 1/16) := by
  sorry

end function_identification_l1979_197915


namespace trigonometric_simplification_l1979_197992

theorem trigonometric_simplification (α : ℝ) : 
  (Real.tan ((5 / 4) * Real.pi - 4 * α) * (Real.sin ((5 / 4) * Real.pi + 4 * α))^2) / 
  (1 - 2 * (Real.cos (4 * α))^2) = -1 / 2 := by
  sorry

end trigonometric_simplification_l1979_197992


namespace geometric_sequence_product_l1979_197996

theorem geometric_sequence_product (a : ℕ → ℝ) : 
  (∀ (n : ℕ), a (n + 1) / a n = a 2 / a 1) →  -- Geometric sequence condition
  a 1 = 1/2 →                                -- First term is 1/2
  a 5 = 8 →                                  -- Fifth term is 8
  a 2 * a 3 * a 4 = 8 :=                     -- Product of middle terms is 8
by
  sorry

end geometric_sequence_product_l1979_197996


namespace square_sum_geq_product_sum_l1979_197936

theorem square_sum_geq_product_sum (a b c : ℝ) : a^2 + b^2 + c^2 ≥ a*b + a*c + b*c := by
  sorry

end square_sum_geq_product_sum_l1979_197936


namespace some_club_members_not_committee_members_l1979_197993

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (ClubMember : U → Prop)
variable (CommitteeMember : U → Prop)
variable (Punctual : U → Prop)

-- State the theorem
theorem some_club_members_not_committee_members :
  (∃ x, ClubMember x ∧ ¬Punctual x) →
  (∀ x, CommitteeMember x → Punctual x) →
  ∃ x, ClubMember x ∧ ¬CommitteeMember x :=
by
  sorry


end some_club_members_not_committee_members_l1979_197993


namespace product_xy_equals_one_l1979_197985

theorem product_xy_equals_one (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : 1 / (1 + x + x^2) + 1 / (1 + y + y^2) + 1 / (1 + x + y) = 1) :
  x * y = 1 := by
  sorry

end product_xy_equals_one_l1979_197985


namespace square_partition_impossibility_l1979_197984

theorem square_partition_impossibility :
  ¬ ∃ (partition : List (ℕ × ℕ)),
    (∀ (rect : ℕ × ℕ), rect ∈ partition →
      (2 * (rect.1 + rect.2) = 18 ∨ 2 * (rect.1 + rect.2) = 22 ∨ 2 * (rect.1 + rect.2) = 26)) ∧
    (List.sum (partition.map (λ rect => rect.1 * rect.2)) = 35 * 35) :=
by
  sorry


end square_partition_impossibility_l1979_197984


namespace arithmetic_seq_sum_l1979_197973

-- Define an arithmetic sequence
def is_arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_seq_sum (a : ℕ → ℝ) :
  is_arithmetic_seq a →
  a 4 + a 6 + a 8 + a 10 + a 12 = 120 →
  a 8 = 24 := by
  sorry

end arithmetic_seq_sum_l1979_197973


namespace root_condition_implies_m_range_l1979_197934

theorem root_condition_implies_m_range (m : ℝ) :
  (∀ x : ℝ, (m / (2 * x - 4) = (1 - x) / (2 - x) - 2) → x > 0) →
  m < 6 ∧ m ≠ 2 :=
by sorry

end root_condition_implies_m_range_l1979_197934


namespace reflect_M_y_axis_l1979_197958

/-- Reflects a point across the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

/-- The original point M -/
def M : ℝ × ℝ := (3, 2)

/-- Theorem: Reflecting M(3,2) across the y-axis results in (-3,2) -/
theorem reflect_M_y_axis : reflect_y M = (-3, 2) := by
  sorry

end reflect_M_y_axis_l1979_197958


namespace opponents_team_points_l1979_197917

-- Define the points for each player
def max_points : ℕ := 5
def dulce_points : ℕ := 3

-- Define Val's points as twice the combined points of Max and Dulce
def val_points : ℕ := 2 * (max_points + dulce_points)

-- Define the total points of their team
def team_points : ℕ := max_points + dulce_points + val_points

-- Define the point difference between the teams
def point_difference : ℕ := 16

-- Theorem to prove
theorem opponents_team_points : 
  team_points + point_difference = 40 := by sorry

end opponents_team_points_l1979_197917


namespace or_sufficient_not_necessary_for_and_l1979_197914

theorem or_sufficient_not_necessary_for_and (p q : Prop) :
  (∃ (h : p ∨ q → p ∧ q), ¬(p ∧ q → p ∨ q)) := by sorry

end or_sufficient_not_necessary_for_and_l1979_197914


namespace range_of_m_l1979_197910

-- Define propositions p and q
def p (x : ℝ) : Prop := |1 - (x - 1) / 3| ≤ 2
def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

-- Define the set A (negation of q)
def A (m : ℝ) : Set ℝ := {x | x > 1 + m ∨ x < 1 - m}

-- Define the set B (negation of p)
def B : Set ℝ := {x | x > 10 ∨ x < -2}

-- Theorem statement
theorem range_of_m (m : ℝ) :
  (m > 0) →
  (∀ x, x ∈ A m → x ∈ B) →
  (∃ x, x ∈ B ∧ x ∉ A m) →
  m ≥ 9 :=
by sorry

end range_of_m_l1979_197910


namespace max_coverage_of_two_inch_card_l1979_197970

/-- A checkerboard square -/
structure CheckerboardSquare where
  size : Real
  (size_positive : size > 0)

/-- A square card -/
structure SquareCard where
  side_length : Real
  (side_length_positive : side_length > 0)

/-- Represents the coverage of a card on a checkerboard -/
def Coverage (card : SquareCard) (square : CheckerboardSquare) : Nat :=
  sorry

/-- Theorem: The maximum number of one-inch squares on a checkerboard 
    that can be covered by a 2-inch square card is 12 -/
theorem max_coverage_of_two_inch_card : 
  ∀ (board_square : CheckerboardSquare) (card : SquareCard),
    board_square.size = 1 → 
    card.side_length = 2 → 
    ∃ (n : Nat), Coverage card board_square = n ∧ 
      ∀ (m : Nat), Coverage card board_square ≤ m → n ≤ m ∧ n = 12 :=
sorry

end max_coverage_of_two_inch_card_l1979_197970


namespace infinite_sum_equals_ln2_squared_l1979_197971

/-- The infinite sum of the given series is equal to ln(2)² -/
theorem infinite_sum_equals_ln2_squared :
  ∑' k : ℕ, (3 * Real.log (4 * k + 2) / (4 * k + 2) -
             Real.log (4 * k + 3) / (4 * k + 3) -
             Real.log (4 * k + 4) / (4 * k + 4) -
             Real.log (4 * k + 5) / (4 * k + 5)) = (Real.log 2) ^ 2 := by
  sorry

end infinite_sum_equals_ln2_squared_l1979_197971


namespace angle_beta_proof_l1979_197948

theorem angle_beta_proof (α β : Real) (h1 : π / 2 < β) (h2 : β < π)
  (h3 : Real.tan (α + β) = 9 / 19) (h4 : Real.tan α = -4) :
  β = π - Real.arctan 5 := by
  sorry

end angle_beta_proof_l1979_197948


namespace boat_distance_calculation_l1979_197911

theorem boat_distance_calculation
  (boat_speed : ℝ)
  (stream_speed : ℝ)
  (total_time : ℝ)
  (h1 : boat_speed = 8)
  (h2 : stream_speed = 2)
  (h3 : total_time = 56)
  : ∃ (distance : ℝ),
    distance = 210 ∧
    total_time = distance / (boat_speed + stream_speed) + distance / (boat_speed - stream_speed) :=
by sorry

end boat_distance_calculation_l1979_197911


namespace stating_consecutive_sum_equals_odd_divisors_l1979_197967

/-- 
Given a positive integer n, count_consecutive_sum n returns the number of ways
n can be represented as a sum of one or more consecutive positive integers.
-/
def count_consecutive_sum (n : ℕ+) : ℕ := sorry

/-- 
Given a positive integer n, count_odd_divisors n returns the number of odd
divisors of n.
-/
def count_odd_divisors (n : ℕ+) : ℕ := sorry

/-- 
Theorem stating that for any positive integer n, the number of ways n can be
represented as a sum of one or more consecutive positive integers is equal to
the number of odd divisors of n.
-/
theorem consecutive_sum_equals_odd_divisors (n : ℕ+) :
  count_consecutive_sum n = count_odd_divisors n := by sorry

end stating_consecutive_sum_equals_odd_divisors_l1979_197967


namespace arithmetic_sequence_common_difference_l1979_197988

def isArithmeticSequence (a : ℕ → ℕ) (d : ℕ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def sumIsTerm (a : ℕ → ℕ) : Prop :=
  ∀ p q, ∃ k, a k = a p + a q

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℕ) (d : ℕ) 
  (h1 : isArithmeticSequence a d)
  (h2 : a 1 = 9)
  (h3 : sumIsTerm a) :
  d = 1 ∨ d = 3 ∨ d = 9 := by
  sorry

end arithmetic_sequence_common_difference_l1979_197988


namespace final_result_depends_on_blue_l1979_197950

/-- Represents the color of a sprite -/
inductive SpriteColor
| Red
| Blue

/-- Represents the state of the game -/
structure GameState where
  red : ℕ  -- number of red sprites
  blue : ℕ  -- number of blue sprites

/-- Represents the result of the game -/
def GameResult := SpriteColor

/-- The game rules for sprite collision -/
def collide (c1 c2 : SpriteColor) : SpriteColor :=
  match c1, c2 with
  | SpriteColor.Red, SpriteColor.Red => SpriteColor.Red
  | SpriteColor.Blue, SpriteColor.Blue => SpriteColor.Red
  | _, _ => SpriteColor.Blue

/-- The final result of the game -/
def finalResult (initial : GameState) : GameResult :=
  if initial.blue % 2 = 0 then SpriteColor.Red else SpriteColor.Blue

/-- The main theorem: the final result depends only on the initial number of blue sprites -/
theorem final_result_depends_on_blue (m n : ℕ) :
  finalResult { red := m, blue := n } = 
    if n % 2 = 0 then SpriteColor.Red else SpriteColor.Blue :=
by sorry

end final_result_depends_on_blue_l1979_197950


namespace norwich_carriages_l1979_197995

/-- The number of carriages in each town --/
structure Carriages where
  euston : ℕ
  norfolk : ℕ
  norwich : ℕ
  flying_scotsman : ℕ

/-- The conditions of the carriage problem --/
def carriage_problem (c : Carriages) : Prop :=
  c.euston = c.norfolk + 20 ∧
  c.flying_scotsman = c.norwich + 20 ∧
  c.euston = 130 ∧
  c.euston + c.norfolk + c.norwich + c.flying_scotsman = 460

/-- The theorem stating that Norwich had 100 carriages --/
theorem norwich_carriages :
  ∃ c : Carriages, carriage_problem c ∧ c.norwich = 100 := by
  sorry

end norwich_carriages_l1979_197995


namespace pentagon_area_l1979_197966

/-- Represents a pentagon formed by removing a triangular section from a rectangle --/
structure Pentagon where
  sides : Finset ℕ
  area : ℕ

/-- The theorem stating the area of the specific pentagon --/
theorem pentagon_area : ∃ (p : Pentagon), 
  p.sides = {17, 23, 26, 28, 34} ∧ p.area = 832 := by
  sorry

end pentagon_area_l1979_197966


namespace right_triangle_leg_divisible_by_three_l1979_197909

theorem right_triangle_leg_divisible_by_three 
  (a b c : ℕ) -- a, b are legs, c is hypotenuse
  (h_right : a^2 + b^2 = c^2) -- Pythagorean theorem
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) -- Positive sides
  : 3 ∣ a ∨ 3 ∣ b := by
sorry

end right_triangle_leg_divisible_by_three_l1979_197909


namespace quadratic_equation_solution_l1979_197979

theorem quadratic_equation_solution :
  let a : ℝ := 1
  let b : ℝ := 1
  let c : ℝ := -3
  let x₁ : ℝ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ : ℝ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  x₁^2 + x₁ - 3 = 0 ∧ x₂^2 + x₂ - 3 = 0 :=
by sorry

end quadratic_equation_solution_l1979_197979


namespace total_phones_sold_l1979_197940

/-- Calculates the total number of cell phones sold given the initial and final inventories, and the number of damaged/defective phones. -/
def cellPhonesSold (initialSamsung : ℕ) (finalSamsung : ℕ) (initialIPhone : ℕ) (finalIPhone : ℕ) (damagedSamsung : ℕ) (defectiveIPhone : ℕ) : ℕ :=
  (initialSamsung - damagedSamsung - finalSamsung) + (initialIPhone - defectiveIPhone - finalIPhone)

/-- Theorem stating that the total number of cell phones sold is 4 given the specific inventory and damage numbers. -/
theorem total_phones_sold :
  cellPhonesSold 14 10 8 5 2 1 = 4 := by
  sorry

end total_phones_sold_l1979_197940


namespace triangle_reciprocal_sum_l1979_197925

/-- For any triangle, the sum of reciprocals of altitudes equals the sum of reciprocals of exradii, which equals the reciprocal of the inradius. -/
theorem triangle_reciprocal_sum (a b c h_a h_b h_c r_a r_b r_c r A p : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ h_a > 0 ∧ h_b > 0 ∧ h_c > 0 ∧ r_a > 0 ∧ r_b > 0 ∧ r_c > 0 ∧ r > 0 ∧ A > 0 ∧ p > 0)
  (h_semiperimeter : p = (a + b + c) / 2)
  (h_area : A = p * r)
  (h_altitude_a : h_a = 2 * A / a)
  (h_altitude_b : h_b = 2 * A / b)
  (h_altitude_c : h_c = 2 * A / c)
  (h_exradius_a : r_a = A / (p - a))
  (h_exradius_b : r_b = A / (p - b))
  (h_exradius_c : r_c = A / (p - c)) :
  1 / h_a + 1 / h_b + 1 / h_c = 1 / r_a + 1 / r_b + 1 / r_c ∧
  1 / h_a + 1 / h_b + 1 / h_c = 1 / r := by sorry

end triangle_reciprocal_sum_l1979_197925


namespace circle_passes_through_points_circle_equation_l1979_197975

/-- A circle passing through three given points -/
def CircleThroughThreePoints (p1 p2 p3 : ℝ × ℝ) :=
  {(x, y) : ℝ × ℝ | x^2 + y^2 + D*x + E*y + F = 0}
  where
    D : ℝ := -8
    E : ℝ := 6
    F : ℝ := 0

/-- The circle passes through the given points -/
theorem circle_passes_through_points :
  let C := CircleThroughThreePoints (0, 0) (1, 1) (4, 2)
  (0, 0) ∈ C ∧ (1, 1) ∈ C ∧ (4, 2) ∈ C := by
  sorry

/-- The equation of the circle is x^2 + y^2 - 8x + 6y = 0 -/
theorem circle_equation (x y : ℝ) :
  let C := CircleThroughThreePoints (0, 0) (1, 1) (4, 2)
  (x, y) ∈ C ↔ x^2 + y^2 - 8*x + 6*y = 0 := by
  sorry

end circle_passes_through_points_circle_equation_l1979_197975


namespace remaining_hard_hats_l1979_197999

/-- Represents the number of hard hats in the truck -/
structure HardHats :=
  (pink : ℕ)
  (green : ℕ)
  (yellow : ℕ)

/-- Calculates the total number of hard hats -/
def totalHardHats (hats : HardHats) : ℕ :=
  hats.pink + hats.green + hats.yellow

/-- Represents the actions of Carl and John -/
def removeHardHats (initial : HardHats) : HardHats :=
  let afterCarl := HardHats.mk (initial.pink - 4) initial.green initial.yellow
  let johnPinkRemoval := 6
  HardHats.mk 
    (afterCarl.pink - johnPinkRemoval)
    (afterCarl.green - 2 * johnPinkRemoval)
    afterCarl.yellow

/-- The main theorem to prove -/
theorem remaining_hard_hats (initial : HardHats) 
  (h1 : initial.pink = 26) 
  (h2 : initial.green = 15) 
  (h3 : initial.yellow = 24) :
  totalHardHats (removeHardHats initial) = 43 := by
  sorry

end remaining_hard_hats_l1979_197999


namespace scientific_notation_equality_l1979_197998

theorem scientific_notation_equality : ∃ (a : ℝ) (n : ℤ), 
  0.00000043 = a * (10 : ℝ) ^ n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ a = 4.3 ∧ n = -7 := by
  sorry

end scientific_notation_equality_l1979_197998


namespace hyperbola_asymptote_slope_l1979_197930

/-- Given a hyperbola with equation (x-2)^2/144 - (y+3)^2/81 = 1, 
    the slope of its asymptotes is 3/4 -/
theorem hyperbola_asymptote_slope :
  ∀ (x y : ℝ), 
  ((x - 2)^2 / 144 - (y + 3)^2 / 81 = 1) →
  (∃ m : ℝ, m = 3/4 ∧ 
   (∀ ε > 0, ∃ x₀ y₀ : ℝ, 
    ((x₀ - 2)^2 / 144 - (y₀ + 3)^2 / 81 = 1) ∧
    abs (y₀ - (m * x₀ - 9/2)) < ε)) :=
by sorry

end hyperbola_asymptote_slope_l1979_197930


namespace sum_of_fifth_powers_l1979_197964

theorem sum_of_fifth_powers (a b u v : ℝ) 
  (h1 : a * u + b * v = 5)
  (h2 : a * u^2 + b * v^2 = 11)
  (h3 : a * u^3 + b * v^3 = 30)
  (h4 : a * u^4 + b * v^4 = 76) :
  a * u^5 + b * v^5 = 8264 / 319 := by
sorry

end sum_of_fifth_powers_l1979_197964


namespace function_equality_l1979_197997

theorem function_equality (k : ℝ) (x : ℝ) (h1 : k > 0) (h2 : x ≠ Real.sqrt k) :
  (x^2 - k) / (x - Real.sqrt k) = 3 * x → x = Real.sqrt k / 2 := by
  sorry

end function_equality_l1979_197997


namespace cassette_price_proof_l1979_197960

def total_money : ℕ := 37
def cd_price : ℕ := 14

theorem cassette_price_proof :
  ∃ (cassette_price : ℕ),
    2 * cd_price + cassette_price = total_money ∧
    cd_price + 2 * cassette_price = total_money - 5 ∧
    cassette_price = 9 := by
  sorry

end cassette_price_proof_l1979_197960


namespace loss_equivalent_pencils_proof_l1979_197902

/-- The number of pencils Patrick purchased -/
def total_pencils : ℕ := 60

/-- The ratio of cost to selling price for 60 pencils -/
def cost_to_sell_ratio : ℚ := 1.3333333333333333

/-- The number of pencils whose selling price equals the loss -/
def loss_equivalent_pencils : ℕ := 20

theorem loss_equivalent_pencils_proof :
  ∃ (selling_price : ℚ) (cost : ℚ),
    cost = cost_to_sell_ratio * selling_price ∧
    loss_equivalent_pencils * (selling_price / total_pencils) = cost - selling_price :=
by sorry

end loss_equivalent_pencils_proof_l1979_197902


namespace zach_ben_score_difference_l1979_197990

theorem zach_ben_score_difference :
  ∀ (zach_score ben_score : ℕ),
    zach_score = 42 →
    ben_score = 21 →
    zach_score - ben_score = 21 :=
by
  sorry

end zach_ben_score_difference_l1979_197990


namespace algebraic_expression_values_l1979_197931

theorem algebraic_expression_values (p q : ℝ) :
  (p * 1^3 + q * 1 + 1 = 2023) →
  (p * (-1)^3 + q * (-1) + 1 = -2021) := by
  sorry

end algebraic_expression_values_l1979_197931


namespace percentage_difference_l1979_197949

theorem percentage_difference (T : ℝ) (h1 : T > 0) : 
  let F := 0.70 * T
  let S := 0.90 * F
  (T - S) / T = 0.37 := by
sorry

end percentage_difference_l1979_197949


namespace fraction_equality_l1979_197986

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hd : y^2 - 1/x ≠ 0) :
  (x^2 - 1/y) / (y^2 - 1/x) = x/y := by
  sorry

end fraction_equality_l1979_197986


namespace division_inequality_quotient_invariance_l1979_197937

theorem division_inequality : 0.056 / 0.08 ≠ 0.56 / 0.08 := by
  -- The proof goes here
  sorry

-- Property of invariance of quotient
theorem quotient_invariance (a b c : ℝ) (hc : c ≠ 0) :
  a / b = (a * c) / (b * c) := by
  -- The proof goes here
  sorry

end division_inequality_quotient_invariance_l1979_197937


namespace sqrt_13_between_3_and_4_l1979_197943

theorem sqrt_13_between_3_and_4 (a : ℝ) (h : a = Real.sqrt 13) : 3 < a ∧ a < 4 := by
  sorry

end sqrt_13_between_3_and_4_l1979_197943


namespace center_square_side_length_l1979_197983

theorem center_square_side_length 
  (main_square_side : ℝ) 
  (l_shape_area_fraction : ℝ) 
  (num_l_shapes : ℕ) :
  main_square_side = 120 →
  l_shape_area_fraction = 1 / 5 →
  num_l_shapes = 4 →
  let total_area := main_square_side ^ 2
  let l_shapes_area := num_l_shapes * l_shape_area_fraction * total_area
  let center_square_area := total_area - l_shapes_area
  Real.sqrt center_square_area = 60 := by sorry

end center_square_side_length_l1979_197983


namespace intersection_of_M_and_N_l1979_197963

def M : Set ℝ := {x | Real.sqrt x < 4}
def N : Set ℝ := {x | 3 * x ≥ 1}

theorem intersection_of_M_and_N : M ∩ N = {x | 1/3 ≤ x ∧ x < 16} := by
  sorry

end intersection_of_M_and_N_l1979_197963


namespace thermal_underwear_sales_l1979_197982

def cost_price : ℕ := 50
def standard_price : ℕ := 70
def price_adjustments : List ℤ := [5, 2, 1, 0, -2]
def sets_sold : List ℕ := [7, 10, 15, 20, 23]

theorem thermal_underwear_sales :
  (List.sum (List.zipWith (· * ·) price_adjustments (List.map Int.ofNat sets_sold)) = 24) ∧
  ((standard_price - cost_price) * (List.sum sets_sold) + 24 = 1524) := by
  sorry

end thermal_underwear_sales_l1979_197982


namespace twice_x_minus_three_l1979_197959

theorem twice_x_minus_three (x : ℝ) : 2 * x - 3 = 2 * x - 3 := by
  sorry

end twice_x_minus_three_l1979_197959


namespace expression_evaluation_l1979_197927

theorem expression_evaluation :
  -2^3 + 36 / 3^2 * (-1/2 : ℝ) + |(-5 : ℝ)| = -5 := by sorry

end expression_evaluation_l1979_197927


namespace solution_value_l1979_197955

theorem solution_value (x y : ℚ) (h1 : 2 * x + y = 6) (h2 : x + 2 * y = 5) : 
  (x + y) / 3 = 11 / 9 := by
  sorry

end solution_value_l1979_197955


namespace line_segment_endpoint_l1979_197939

theorem line_segment_endpoint (x : ℝ) :
  (((x - 3)^2 + (4 + 2)^2).sqrt = 17) →
  (x < 0) →
  (x = 3 - Real.sqrt 253) := by
sorry

end line_segment_endpoint_l1979_197939


namespace solve_system_l1979_197944

theorem solve_system (x y : ℚ) (h1 : x/2 - 2*y = 2) (h2 : x/2 + 2*y = 12) : y = 5/2 := by
  sorry

end solve_system_l1979_197944


namespace tangent_line_properties_l1979_197976

noncomputable def f (x : ℝ) : ℝ := Real.exp x + 2

theorem tangent_line_properties :
  (∃ x : ℝ, (deriv f) x = 3) ∧
  (∃! t : ℝ, (f t - 2) / t = (deriv f) t) ∧
  (∃ t₁ t₂ : ℝ, t₁ ≠ t₂ ∧
    (f t₁ - 4) / (t₁ - 1) = (deriv f) t₁ ∧
    (f t₂ - 4) / (t₂ - 1) = (deriv f) t₂ ∧
    ∀ t : ℝ, t ≠ t₁ → t ≠ t₂ →
      (f t - 4) / (t - 1) ≠ (deriv f) t) :=
sorry

end tangent_line_properties_l1979_197976


namespace circle_M_properties_l1979_197923

-- Define the circle M
def circle_M : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 1)^2 = 4}

-- Define points A and B
def point_A : ℝ × ℝ := (1, -1)
def point_B : ℝ × ℝ := (-1, 1)

-- Define the line on which the center of M lies
def center_line (x y : ℝ) : Prop := x + y - 2 = 0

-- Theorem statement
theorem circle_M_properties :
  (point_A ∈ circle_M) ∧
  (point_B ∈ circle_M) ∧
  (∃ c : ℝ × ℝ, c ∈ circle_M ∧ center_line c.1 c.2) ∧
  (∀ p : ℝ × ℝ, p ∈ circle_M →
    (4 - Real.sqrt 7) / 3 ≤ (p.2 + 3) / (p.1 + 3) ∧
    (p.2 + 3) / (p.1 + 3) ≤ (4 + Real.sqrt 7) / 3) :=
by sorry

end circle_M_properties_l1979_197923


namespace vector_collinear_opposite_direction_l1979_197974

/-- Two vectors in ℝ² -/
def Vector2D : Type := ℝ × ℝ

/-- Check if two vectors are collinear -/
def collinear (v w : Vector2D) : Prop :=
  ∃ k : ℝ, v = (k * w.1, k * w.2)

/-- Check if two vectors have opposite directions -/
def opposite_directions (v w : Vector2D) : Prop :=
  ∃ k : ℝ, k < 0 ∧ v = (k * w.1, k * w.2)

/-- The main theorem -/
theorem vector_collinear_opposite_direction (m : ℝ) :
  let a : Vector2D := (m, 1)
  let b : Vector2D := (1, m)
  collinear a b → opposite_directions a b → m = -1 := by
  sorry

end vector_collinear_opposite_direction_l1979_197974


namespace set_equation_solution_l1979_197913

theorem set_equation_solution (p q : ℝ) : 
  let M := {x : ℝ | x^2 + p*x - 2 = 0}
  let N := {x : ℝ | x^2 - 2*x + q = 0}
  (M ∪ N = {-1, 0, 2}) → (p = -1 ∧ q = 0) := by
sorry

end set_equation_solution_l1979_197913


namespace boat_speed_in_still_water_l1979_197981

/-- Given a boat that travels 6 km/hr along a stream and 2 km/hr against the same stream,
    its speed in still water is 4 km/hr. -/
theorem boat_speed_in_still_water (boat_speed : ℝ) (stream_speed : ℝ) : 
  (boat_speed + stream_speed = 6) → 
  (boat_speed - stream_speed = 2) → 
  boat_speed = 4 := by
  sorry

end boat_speed_in_still_water_l1979_197981


namespace expression_bounds_l1979_197924

theorem expression_bounds (a b c d : Real) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) (hd : 0 ≤ d ∧ d ≤ 1) :
  2 * Real.sqrt 2 ≤ 
    Real.sqrt ((a^2)^2 + (b^2 - b^2)^2) + 
    Real.sqrt ((b^2)^2 + (c^2 - b^2)^2) + 
    Real.sqrt ((c^2)^2 + (d^2 - c^2)^2) + 
    Real.sqrt ((d^2)^2 + (a^2 - d^2)^2) ∧
  Real.sqrt ((a^2)^2 + (b^2 - b^2)^2) + 
    Real.sqrt ((b^2)^2 + (c^2 - b^2)^2) + 
    Real.sqrt ((c^2)^2 + (d^2 - c^2)^2) + 
    Real.sqrt ((d^2)^2 + (a^2 - d^2)^2) ≤ 4 :=
by sorry

end expression_bounds_l1979_197924


namespace factorial_square_root_theorem_l1979_197941

-- Define factorial function
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- State the theorem
theorem factorial_square_root_theorem :
  (((factorial 5 * factorial 4).sqrt) ^ 2 : ℕ) = 2880 := by
  sorry

end factorial_square_root_theorem_l1979_197941


namespace survey_is_sample_l1979_197952

/-- Represents the total number of students in the population -/
def population_size : ℕ := 32000

/-- Represents the number of students surveyed -/
def survey_size : ℕ := 1600

/-- Represents a student's weight -/
structure Weight where
  value : ℝ

/-- Represents the population of all students' weights -/
def population : Finset Weight := sorry

/-- Represents the surveyed students' weights -/
def survey : Finset Weight := sorry

/-- Theorem stating that the survey is a sample of the population -/
theorem survey_is_sample : survey ⊆ population ∧ survey.card = survey_size := by sorry

end survey_is_sample_l1979_197952


namespace jelly_bean_problem_l1979_197957

def jelly_bean_piles (initial_amount : ℕ) (amount_eaten : ℕ) (pile_weight : ℕ) : ℕ :=
  (initial_amount - amount_eaten) / pile_weight

theorem jelly_bean_problem :
  jelly_bean_piles 36 6 10 = 3 := by
  sorry

end jelly_bean_problem_l1979_197957


namespace neil_initial_games_l1979_197906

theorem neil_initial_games (henry_initial : ℕ) (henry_gave : ℕ) (henry_neil_ratio : ℕ) :
  henry_initial = 33 →
  henry_gave = 5 →
  henry_neil_ratio = 4 →
  henry_initial - henry_gave = henry_neil_ratio * (2 + henry_gave) :=
by
  sorry

end neil_initial_games_l1979_197906


namespace folded_quadrilateral_l1979_197978

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a quadrilateral defined by four points -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Checks if two points coincide after folding -/
def coincide (p1 p2 : Point) : Prop :=
  ∃ (m : Point), (m.x = (p1.x + p2.x) / 2 ∧ m.y = (p1.y + p2.y) / 2) ∧
  (p2.y - p1.y) * (m.x - p1.x) = (p1.x - p2.x) * (m.y - p1.y)

/-- Calculates the area of a quadrilateral -/
noncomputable def area (q : Quadrilateral) : ℝ :=
  sorry

/-- Main theorem -/
theorem folded_quadrilateral :
  ∀ (m n : ℝ),
  let q := Quadrilateral.mk
    (Point.mk 0 2)  -- A
    (Point.mk 4 0)  -- B
    (Point.mk 7 3)  -- C
    (Point.mk m n)  -- D
  coincide q.A q.B ∧ coincide q.C q.D →
  m = 3/5 ∧ n = 31/5 ∧ area q = 117/5 := by
  sorry

end folded_quadrilateral_l1979_197978


namespace chairs_for_play_l1979_197989

theorem chairs_for_play (rows : ℕ) (chairs_per_row : ℕ) 
  (h1 : rows = 27) (h2 : chairs_per_row = 16) : 
  rows * chairs_per_row = 432 := by
  sorry

end chairs_for_play_l1979_197989


namespace arthur_summer_reading_l1979_197903

theorem arthur_summer_reading (book1_pages book2_pages : ℕ) 
  (book1_read_percent : ℚ) (book2_read_fraction : ℚ) (pages_left : ℕ) : 
  book1_pages = 500 → 
  book2_pages = 1000 → 
  book1_read_percent = 80 / 100 → 
  book2_read_fraction = 1 / 5 → 
  pages_left = 200 → 
  (book1_pages * book1_read_percent).floor + 
  (book2_pages * book2_read_fraction).floor + 
  pages_left = 800 := by
  sorry

end arthur_summer_reading_l1979_197903


namespace joel_age_when_dad_twice_as_old_l1979_197965

/-- 
Given:
- Joel is currently 5 years old
- Joel's dad is currently 32 years old

Prove that Joel will be 27 years old when his dad is twice as old as him.
-/
theorem joel_age_when_dad_twice_as_old (joel_current_age : ℕ) (dad_current_age : ℕ) :
  joel_current_age = 5 →
  dad_current_age = 32 →
  ∃ (future_joel_age : ℕ), 
    future_joel_age + joel_current_age = dad_current_age ∧
    2 * future_joel_age = future_joel_age + dad_current_age ∧
    future_joel_age = 27 :=
by sorry

end joel_age_when_dad_twice_as_old_l1979_197965


namespace largest_power_dividing_factorial_squared_factorization_of_2025_l1979_197947

/-- The largest integer k such that 2025^k divides (2025!)^2 is 505. -/
theorem largest_power_dividing_factorial_squared : ∃ k : ℕ, k = 505 ∧ 
  (∀ m : ℕ, (2025 ^ m : ℕ) ∣ (Nat.factorial 2025)^2 → m ≤ k) ∧
  (2025 ^ k : ℕ) ∣ (Nat.factorial 2025)^2 := by
  sorry

/-- 2025 is equal to 3^4 * 5^2 -/
theorem factorization_of_2025 : 2025 = 3^4 * 5^2 := by
  sorry

end largest_power_dividing_factorial_squared_factorization_of_2025_l1979_197947


namespace science_club_neither_math_nor_physics_l1979_197935

theorem science_club_neither_math_nor_physics 
  (total : ℕ) 
  (math : ℕ) 
  (physics : ℕ) 
  (both : ℕ) 
  (h1 : total = 100) 
  (h2 : math = 65) 
  (h3 : physics = 43) 
  (h4 : both = 10) : 
  total - (math + physics - both) = 2 :=
by sorry

end science_club_neither_math_nor_physics_l1979_197935


namespace joy_pencil_count_l1979_197938

/-- The number of pencils Colleen has -/
def colleen_pencils : ℕ := 50

/-- The cost of each pencil in dollars -/
def pencil_cost : ℕ := 4

/-- The difference in dollars between what Colleen and Joy paid -/
def payment_difference : ℕ := 80

/-- The number of pencils Joy has -/
def joy_pencils : ℕ := 30

theorem joy_pencil_count :
  colleen_pencils * pencil_cost = joy_pencils * pencil_cost + payment_difference :=
sorry

end joy_pencil_count_l1979_197938


namespace no_solution_quadratic_inequality_l1979_197994

theorem no_solution_quadratic_inequality :
  ∀ x : ℝ, ¬(-x^2 + 2*x - 2 > 0) := by
sorry

end no_solution_quadratic_inequality_l1979_197994


namespace unique_solution_l1979_197916

/-- Define the function f(x, y) = (x + y)(x^2 + y^2) -/
def f (x y : ℝ) : ℝ := (x + y) * (x^2 + y^2)

/-- Theorem stating that the only solution to the system of equations is (0, 0, 0, 0) -/
theorem unique_solution (a b c d : ℝ) :
  f a b = f c d ∧ f a c = f b d ∧ f a d = f b c →
  a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0 := by
  sorry


end unique_solution_l1979_197916


namespace chocolate_count_l1979_197920

/-- The number of chocolates in each bag -/
def chocolates_per_bag : ℕ := 156

/-- The number of bags bought -/
def bags_bought : ℕ := 20

/-- The total number of chocolates -/
def total_chocolates : ℕ := chocolates_per_bag * bags_bought

theorem chocolate_count : total_chocolates = 3120 := by
  sorry

end chocolate_count_l1979_197920


namespace f_inverse_g_l1979_197962

noncomputable def f (x : ℝ) : ℝ := 3 - 7*x + x^2

noncomputable def g (x : ℝ) : ℝ := (7 + Real.sqrt (37 + 4*x)) / 2

theorem f_inverse_g : Function.LeftInverse g f ∧ Function.RightInverse g f := by
  sorry

end f_inverse_g_l1979_197962


namespace not_divisible_by_2020_l1979_197929

theorem not_divisible_by_2020 (k : ℕ) : ¬(2020 ∣ (k^3 - 3*k^2 + 2*k + 2)) := by
  sorry

end not_divisible_by_2020_l1979_197929


namespace jack_remaining_plates_l1979_197921

def initial_flower_plates : ℕ := 6
def initial_checked_plates : ℕ := 9
def initial_striped_plates : ℕ := 3
def smashed_flower_plates : ℕ := 2
def smashed_striped_plates : ℕ := 1

def remaining_plates : ℕ :=
  (initial_flower_plates - smashed_flower_plates) +
  initial_checked_plates +
  (initial_striped_plates - smashed_striped_plates) +
  (initial_checked_plates * initial_checked_plates)

theorem jack_remaining_plates :
  remaining_plates = 96 := by
  sorry

end jack_remaining_plates_l1979_197921


namespace total_peaches_is_273_l1979_197991

/-- The number of monkeys in the zoo --/
def num_monkeys : ℕ := 36

/-- The number of peaches each monkey receives in the first scenario --/
def peaches_per_monkey_scenario1 : ℕ := 6

/-- The number of peaches left over in the first scenario --/
def peaches_left_scenario1 : ℕ := 57

/-- The number of peaches each monkey should receive in the second scenario --/
def peaches_per_monkey_scenario2 : ℕ := 9

/-- The number of monkeys that get nothing in the second scenario --/
def monkeys_with_no_peaches : ℕ := 5

/-- The number of peaches the last monkey gets in the second scenario --/
def peaches_for_last_monkey : ℕ := 3

/-- The total number of peaches --/
def total_peaches : ℕ := num_monkeys * peaches_per_monkey_scenario1 + peaches_left_scenario1

theorem total_peaches_is_273 : total_peaches = 273 := by
  sorry

#eval total_peaches

end total_peaches_is_273_l1979_197991


namespace f_is_odd_sum_greater_than_two_l1979_197972

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (x^2 + 1) / x + x / (x^2 + 1)

-- Theorem 1: f is an odd function
theorem f_is_odd : ∀ x : ℝ, x ≠ 0 → f (-x) = -f x := by sorry

-- Theorem 2: If x₁ > 0, x₂ > 0, x₁ ≠ x₂, and f(x₁) = f(x₂), then x₁ + x₂ > 2
theorem sum_greater_than_two (x₁ x₂ : ℝ) (h1 : x₁ > 0) (h2 : x₂ > 0) (h3 : x₁ ≠ x₂) (h4 : f x₁ = f x₂) : x₁ + x₂ > 2 := by sorry

end f_is_odd_sum_greater_than_two_l1979_197972


namespace ratio_sum_theorem_l1979_197900

theorem ratio_sum_theorem (a b c : ℝ) 
  (h : ∃ k : ℝ, a = 2*k ∧ b = 3*k ∧ c = 5*k) : (a + b) / c = 1 := by
  sorry

end ratio_sum_theorem_l1979_197900


namespace parallel_lines_b_value_l1979_197928

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ : ℝ} : 
  (∃ b₁ b₂ : ℝ, ∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- The value of b for which the given lines are parallel -/
theorem parallel_lines_b_value (b : ℝ) : 
  (∃ c₁ c₂ : ℝ, ∀ x y : ℝ, 3 * y - 3 * b = 9 * x + c₁ ↔ y + 2 = (b + 9) * x + c₂) → 
  b = -6 := by
sorry


end parallel_lines_b_value_l1979_197928


namespace sum_of_roots_l1979_197961

theorem sum_of_roots (a b c d : ℝ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  (∀ x, x^2 - 12*a*x - 13*b = 0 ↔ x = c ∨ x = d) →
  (∀ x, x^2 - 12*c*x - 13*d = 0 ↔ x = a ∨ x = b) →
  a + b + c + d = 2028 :=
by sorry

end sum_of_roots_l1979_197961


namespace domain_of_function_1_l1979_197932

theorem domain_of_function_1 (x : ℝ) : 
  Set.univ = {x : ℝ | ∃ y : ℝ, y = (2 * x^2 - 1) / (x^2 + 3)} :=
sorry

end domain_of_function_1_l1979_197932


namespace triangle_circle_area_ratio_l1979_197922

-- Define a right-angled isosceles triangle
structure RightIsoscelesTriangle where
  leg : ℝ
  hypotenuse : ℝ
  hypotenuse_eq : hypotenuse = Real.sqrt 2 * leg

-- Define a circle
structure Circle where
  radius : ℝ

-- Define the theorem
theorem triangle_circle_area_ratio 
  (t : RightIsoscelesTriangle) 
  (c : Circle) 
  (perimeter_eq : 2 * t.leg + t.hypotenuse = 2 * Real.pi * c.radius) : 
  (t.leg^2 / 2) / (Real.pi * c.radius^2) = Real.pi * (3 - 2 * Real.sqrt 2) / 2 := by
  sorry


end triangle_circle_area_ratio_l1979_197922


namespace sum_product_ratio_theorem_l1979_197907

theorem sum_product_ratio_theorem (x y z : ℝ) (hxy : x ≠ y) (hyz : y ≠ z) (hxz : x ≠ z) (hsum : x + y + z = 12) :
  (x * y + y * z + z * x) / (x^2 + y^2 + z^2) = (144 - (x^2 + y^2 + z^2)) / (2 * (x^2 + y^2 + z^2)) := by
  sorry

end sum_product_ratio_theorem_l1979_197907


namespace factoring_expression_l1979_197954

theorem factoring_expression (x : ℝ) : 2 * x * (x + 3) + 4 * (x + 3) = 2 * (x + 2) * (x + 3) := by
  sorry

end factoring_expression_l1979_197954


namespace highest_red_probability_l1979_197969

/-- Represents the contents of a bag --/
structure Bag where
  red : ℕ
  white : ℕ

/-- The probability of drawing a red ball from a bag --/
def redProbability (bag : Bag) : ℚ :=
  bag.red / (bag.red + bag.white)

/-- The average probability of drawing a red ball from two bags --/
def averageProbability (bag1 bag2 : Bag) : ℚ :=
  (redProbability bag1 + redProbability bag2) / 2

/-- The theorem stating the highest probability of drawing a red ball --/
theorem highest_red_probability :
  ∃ (bag1 bag2 : Bag),
    bag1.red + bag2.red = 5 ∧
    bag1.white + bag2.white = 12 ∧
    bag1.red + bag1.white > 0 ∧
    bag2.red + bag2.white > 0 ∧
    averageProbability bag1 bag2 = 5/8 ∧
    ∀ (other1 other2 : Bag),
      other1.red + other2.red = 5 →
      other1.white + other2.white = 12 →
      other1.red + other1.white > 0 →
      other2.red + other2.white > 0 →
      averageProbability other1 other2 ≤ 5/8 :=
by sorry

end highest_red_probability_l1979_197969


namespace total_egg_rolls_l1979_197908

theorem total_egg_rolls (omar_rolls karen_rolls : ℕ) 
  (h1 : omar_rolls = 219) 
  (h2 : karen_rolls = 229) : 
  omar_rolls + karen_rolls = 448 := by
sorry

end total_egg_rolls_l1979_197908


namespace inequality_proof_l1979_197951

theorem inequality_proof (a b : ℝ) (h1 : b > a) (h2 : a > 0) : 2 * a + b / 2 ≥ 2 * Real.sqrt (a * b) := by
  sorry

end inequality_proof_l1979_197951


namespace expression_lower_bound_l1979_197905

theorem expression_lower_bound (n : ℤ) (L : ℤ) :
  (∃! (S : Finset ℤ), S.card = 25 ∧ ∀ m ∈ S, L < 4*m + 7 ∧ 4*m + 7 < 100) →
  L = 3 := by
  sorry

end expression_lower_bound_l1979_197905


namespace tan_double_angle_l1979_197987

theorem tan_double_angle (α : ℝ) (h : (1 + Real.cos (2 * α)) / Real.sin (2 * α) = 1/2) :
  Real.tan (2 * α) = -4/3 := by
  sorry

end tan_double_angle_l1979_197987


namespace rent_increase_for_tax_change_l1979_197956

/-- Proves that a 12.5% rent increase maintains the same net income when tax increases from 10% to 20% -/
theorem rent_increase_for_tax_change (a : ℝ) (h : a > 0) :
  let initial_net_income := a * (1 - 0.1)
  let new_rent := a * (1 + 0.125)
  let new_net_income := new_rent * (1 - 0.2)
  initial_net_income = new_net_income :=
by sorry

#check rent_increase_for_tax_change

end rent_increase_for_tax_change_l1979_197956


namespace solution_set_inequality_l1979_197901

theorem solution_set_inequality (x : ℝ) : 
  (1 / x < 1 / 3) ↔ (x < 0 ∨ x > 3) :=
by sorry

end solution_set_inequality_l1979_197901


namespace inscribed_rectangle_circle_circumference_l1979_197980

/-- Given a rectangle with sides 9 cm and 12 cm inscribed in a circle,
    prove that the circumference of the circle is 15π cm. -/
theorem inscribed_rectangle_circle_circumference :
  ∀ (circle : Real → Real → Prop) (rectangle : Real → Real → Prop),
    (∃ (x y : Real), rectangle x y ∧ x = 9 ∧ y = 12) →
    (∀ (x y : Real), rectangle x y → ∃ (center : Real × Real) (r : Real),
      circle = λ a b => (a - center.1)^2 + (b - center.2)^2 = r^2) →
    (∃ (circumference : Real), circumference = 15 * Real.pi) :=
by sorry

end inscribed_rectangle_circle_circumference_l1979_197980


namespace defective_shipped_percentage_l1979_197918

theorem defective_shipped_percentage 
  (total_units : ℝ) 
  (defective_rate : ℝ) 
  (shipped_rate : ℝ) 
  (h1 : defective_rate = 0.1) 
  (h2 : shipped_rate = 0.05) : 
  defective_rate * shipped_rate * 100 = 0.5 := by
sorry

end defective_shipped_percentage_l1979_197918


namespace complex_expression_equality_l1979_197953

theorem complex_expression_equality : (2 - Complex.I)^2 - (1 + 3 * Complex.I) = 2 - 7 * Complex.I := by
  sorry

end complex_expression_equality_l1979_197953


namespace hyperbola_standard_equation_ellipse_standard_equation_l1979_197912

-- Problem 1: Hyperbola
def hyperbola_equation (e : ℝ) (vertex_distance : ℝ) : Prop :=
  e = 5/3 ∧ vertex_distance = 6 →
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧
    (∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1 ↔ x^2/9 - y^2/16 = 1)

theorem hyperbola_standard_equation :
  hyperbola_equation (5/3) 6 :=
sorry

-- Problem 2: Ellipse
def ellipse_equation (major_minor_ratio : ℝ) (point : ℝ × ℝ) : Prop :=
  major_minor_ratio = 3 ∧ point = (3, 0) →
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧
    (∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 ↔ x^2/9 + y^2 = 1)) ∨
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧
    (∀ x y : ℝ, y^2/a^2 + x^2/b^2 = 1 ↔ y^2/81 + x^2/9 = 1))

theorem ellipse_standard_equation :
  ellipse_equation 3 (3, 0) :=
sorry

end hyperbola_standard_equation_ellipse_standard_equation_l1979_197912


namespace simplify_sqrt_expression_l1979_197933

theorem simplify_sqrt_expression : 
  Real.sqrt 10 - Real.sqrt 40 + Real.sqrt 90 = 2 * Real.sqrt 10 := by
  sorry

end simplify_sqrt_expression_l1979_197933


namespace age_ratio_proof_l1979_197946

/-- Represents the age of a person -/
structure Age :=
  (years : ℕ)

/-- Represents the ratio between two numbers -/
structure Ratio :=
  (numerator : ℕ)
  (denominator : ℕ)

/-- Given two people p and q, their ages 6 years ago, and their current total age,
    proves that the ratio of their current ages is 3:4 -/
theorem age_ratio_proof 
  (p q : Age) 
  (h1 : p.years + 6 = (q.years + 6) / 2)  -- 6 years ago, p was half of q in age
  (h2 : (p.years + 6) + (q.years + 6) = 21)  -- The total of their present ages is 21
  : Ratio.mk 3 4 = Ratio.mk (p.years + 6) (q.years + 6) :=
by
  sorry


end age_ratio_proof_l1979_197946
