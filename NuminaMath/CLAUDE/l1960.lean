import Mathlib

namespace NUMINAMATH_CALUDE_midpoint_sum_invariant_l1960_196008

/-- A polygon in the Cartesian plane -/
structure Polygon :=
  (vertices : List (ℝ × ℝ))

/-- Create a new polygon from the midpoints of the sides of a given polygon -/
def midpointPolygon (p : Polygon) : Polygon := sorry

/-- Sum of y-coordinates of a polygon's vertices -/
def sumYCoordinates (p : Polygon) : ℝ := sorry

theorem midpoint_sum_invariant (n : ℕ) (Q1 : Polygon) :
  n ≥ 3 →
  Q1.vertices.length = n →
  let Q2 := midpointPolygon Q1
  let Q3 := midpointPolygon Q2
  sumYCoordinates Q3 = sumYCoordinates Q1 := by sorry

end NUMINAMATH_CALUDE_midpoint_sum_invariant_l1960_196008


namespace NUMINAMATH_CALUDE_omega_range_l1960_196056

theorem omega_range (ω : ℝ) (a b : ℝ) (h_pos : ω > 0) 
  (h_ab : π ≤ a ∧ a < b ∧ b ≤ 2*π) 
  (h_sin : Real.sin (ω*a) + Real.sin (ω*b) = 2) : 
  (9/4 ≤ ω ∧ ω ≤ 5/2) ∨ (13/4 ≤ ω) :=
sorry

end NUMINAMATH_CALUDE_omega_range_l1960_196056


namespace NUMINAMATH_CALUDE_roots_on_circle_l1960_196058

theorem roots_on_circle : ∃ (r : ℝ), r = 2 / Real.sqrt 3 ∧
  ∀ (z : ℂ), (z + 2)^4 = 16 * z^4 →
  ∃ (c : ℂ), Complex.abs (z - c) = r :=
sorry

end NUMINAMATH_CALUDE_roots_on_circle_l1960_196058


namespace NUMINAMATH_CALUDE_scientific_notation_3900000000_l1960_196080

theorem scientific_notation_3900000000 :
  3900000000 = 3.9 * (10 ^ 9) := by sorry

end NUMINAMATH_CALUDE_scientific_notation_3900000000_l1960_196080


namespace NUMINAMATH_CALUDE_inequality_proof_l1960_196004

theorem inequality_proof (n : ℕ+) : (2*n+1)^(n : ℕ) ≥ (2*n)^(n : ℕ) + (2*n-1)^(n : ℕ) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1960_196004


namespace NUMINAMATH_CALUDE_exp_three_has_property_M_g_property_M_iff_l1960_196075

-- Define property M
def has_property_M (f : ℝ → ℝ) : Prop :=
  ∃ x₀ : ℝ, f (x₀ + 1) = f x₀ + f 1

-- Statement for f(x) = 3^x
theorem exp_three_has_property_M :
  ∃ x₀ : ℝ, (3 : ℝ)^(x₀ + 1) = (3 : ℝ)^x₀ + (3 : ℝ)^1 :=
sorry

-- Define g(x)
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := Real.log (a / (2 * x^2 + 1))

-- Statement for g(x)
theorem g_property_M_iff (a : ℝ) :
  (a > 0) →
  (has_property_M (g a) ↔ 6 - 3 * Real.sqrt 3 ≤ a ∧ a ≤ 6 + 3 * Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_exp_three_has_property_M_g_property_M_iff_l1960_196075


namespace NUMINAMATH_CALUDE_undeclared_major_fraction_l1960_196002

theorem undeclared_major_fraction (T : ℝ) (f : ℝ) : 
  T > 0 →
  (1/2 : ℝ) * T = T - (1/2 : ℝ) * T →
  (1/2 : ℝ) * T * (1 - (1/2 : ℝ) * (1 - f)) = (45/100 : ℝ) * T →
  f = 4/5 := by sorry

end NUMINAMATH_CALUDE_undeclared_major_fraction_l1960_196002


namespace NUMINAMATH_CALUDE_mrs_martin_coffee_cups_l1960_196014

-- Define the cost of a bagel
def bagel_cost : ℝ := 1.5

-- Define Mrs. Martin's purchase
def mrs_martin_total : ℝ := 12.75
def mrs_martin_bagels : ℕ := 2

-- Define Mr. Martin's purchase
def mr_martin_total : ℝ := 14.00
def mr_martin_coffee : ℕ := 2
def mr_martin_bagels : ℕ := 5

-- Theorem to prove
theorem mrs_martin_coffee_cups : ℕ := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_mrs_martin_coffee_cups_l1960_196014


namespace NUMINAMATH_CALUDE_equal_division_exists_l1960_196068

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A set of 2 million points in a plane -/
def PointSet : Type := Finset Point

/-- A line in a 2D plane, represented by its normal vector and a point on the line -/
structure Line where
  normal : Point
  point : Point

/-- Function to check if a point is on the left side of a line -/
def isLeftOfLine (p : Point) (l : Line) : Prop := sorry

/-- Function to count points on the left side of a line -/
def countLeftPoints (points : PointSet) (l : Line) : ℕ := sorry

/-- Theorem: There exists a line that divides 2 million points equally -/
theorem equal_division_exists (points : PointSet) (h : points.card = 2000000) : 
  ∃ l : Line, countLeftPoints points l = 1000000 := by sorry

end NUMINAMATH_CALUDE_equal_division_exists_l1960_196068


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_l1960_196017

theorem necessary_not_sufficient (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (∀ a b : ℝ, a ≠ 0 → b ≠ 0 → (a / b + b / a ≥ 2) → (a^2 + b^2 ≥ 2*a*b)) ∧
  (∃ a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ (a^2 + b^2 ≥ 2*a*b) ∧ (a / b + b / a < 2)) :=
by sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_l1960_196017


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1960_196022

theorem inequality_system_solution :
  {x : ℝ | 3*x - 1 ≥ x + 1 ∧ x + 4 > 4*x - 2} = {x : ℝ | 1 ≤ x ∧ x < 2} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1960_196022


namespace NUMINAMATH_CALUDE_parallel_tangent_length_l1960_196030

/-- Represents an isosceles triangle with an inscribed circle -/
structure IsoscelesTriangleWithInscribedCircle where
  base : ℝ
  height : ℝ
  inscribed_circle : Circle

/-- Represents a tangent line to the inscribed circle, parallel to the base -/
structure ParallelTangent where
  triangle : IsoscelesTriangleWithInscribedCircle
  length : ℝ

/-- The theorem statement -/
theorem parallel_tangent_length 
  (triangle : IsoscelesTriangleWithInscribedCircle) 
  (tangent : ParallelTangent) 
  (h1 : triangle.base = 12)
  (h2 : triangle.height = 8)
  (h3 : tangent.triangle = triangle) : 
  tangent.length = 3 := by sorry

end NUMINAMATH_CALUDE_parallel_tangent_length_l1960_196030


namespace NUMINAMATH_CALUDE_fraction_dislike_but_interested_l1960_196049

/-- Represents the student population at Novo Middle School -/
structure SchoolPopulation where
  total : ℕ
  artInterested : ℕ
  artUninterested : ℕ
  interestedLike : ℕ
  interestedDislike : ℕ
  uninterestedLike : ℕ
  uninterestedDislike : ℕ

/-- Theorem about the fraction of students who dislike art but are interested -/
theorem fraction_dislike_but_interested (pop : SchoolPopulation) : 
  pop.total = 200 ∧ 
  pop.artInterested = 150 ∧ 
  pop.artUninterested = 50 ∧
  pop.interestedLike = 105 ∧
  pop.interestedDislike = 45 ∧
  pop.uninterestedLike = 10 ∧
  pop.uninterestedDislike = 40 →
  (pop.interestedDislike : ℚ) / (pop.interestedDislike + pop.uninterestedDislike) = 9/17 := by
  sorry

#check fraction_dislike_but_interested

end NUMINAMATH_CALUDE_fraction_dislike_but_interested_l1960_196049


namespace NUMINAMATH_CALUDE_al_investment_l1960_196035

/-- Represents the investment scenario with four participants -/
structure Investment where
  al : ℝ
  betty : ℝ
  clare : ℝ
  dave : ℝ

/-- Defines the conditions of the investment problem -/
def valid_investment (i : Investment) : Prop :=
  i.al > 0 ∧ i.betty > 0 ∧ i.clare > 0 ∧ i.dave > 0 ∧  -- Each begins with a positive amount
  i.al ≠ i.betty ∧ i.al ≠ i.clare ∧ i.al ≠ i.dave ∧    -- Each begins with a different amount
  i.betty ≠ i.clare ∧ i.betty ≠ i.dave ∧ i.clare ≠ i.dave ∧
  i.al + i.betty + i.clare + i.dave = 2000 ∧           -- Total initial investment
  (i.al - 150) + (3 * i.betty) + (3 * i.clare) + (i.dave - 50) = 2500  -- Total after one year

/-- Theorem stating that under the given conditions, Al's original portion was 450 -/
theorem al_investment (i : Investment) (h : valid_investment i) : i.al = 450 := by
  sorry


end NUMINAMATH_CALUDE_al_investment_l1960_196035


namespace NUMINAMATH_CALUDE_root_equation_implication_l1960_196085

theorem root_equation_implication (m : ℝ) : 
  m^2 - m - 3 = 0 → m^2 - m - 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_root_equation_implication_l1960_196085


namespace NUMINAMATH_CALUDE_point_A_in_third_quadrant_l1960_196020

/-- A linear function y = -5ax + b with specific properties -/
structure LinearFunction where
  a : ℝ
  b : ℝ
  a_nonzero : a ≠ 0
  increasing : ∀ x₁ x₂, x₁ < x₂ → (-5 * a * x₁ + b) < (-5 * a * x₂ + b)
  ab_positive : a * b > 0

/-- The point A(a, b) -/
def point_A (f : LinearFunction) : ℝ × ℝ := (f.a, f.b)

/-- Third quadrant definition -/
def third_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 < 0

/-- Theorem stating that point A lies in the third quadrant -/
theorem point_A_in_third_quadrant (f : LinearFunction) :
  third_quadrant (point_A f) := by
  sorry


end NUMINAMATH_CALUDE_point_A_in_third_quadrant_l1960_196020


namespace NUMINAMATH_CALUDE_simplify_fraction_l1960_196001

theorem simplify_fraction (x y : ℚ) (hx : x = 2) (hy : y = 3) :
  8 * x * y^2 / (6 * x^2 * y) = 2 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1960_196001


namespace NUMINAMATH_CALUDE_hexagon_ratio_l1960_196094

/-- A hexagon with specific properties -/
structure Hexagon where
  area : ℝ
  below_rs_area : ℝ
  triangle_base : ℝ
  xr : ℝ
  rs : ℝ

/-- The theorem statement -/
theorem hexagon_ratio (h : Hexagon) (h_area : h.area = 13)
  (h_bisect : h.below_rs_area = h.area / 2)
  (h_below : h.below_rs_area = 2 + (h.triangle_base * (h.below_rs_area - 2) / h.triangle_base) / 2)
  (h_base : h.triangle_base = 4)
  (h_sum : h.xr + h.rs = h.triangle_base) :
  h.xr / h.rs = 1 := by sorry

end NUMINAMATH_CALUDE_hexagon_ratio_l1960_196094


namespace NUMINAMATH_CALUDE_max_winner_number_l1960_196045

/-- Represents a player in the tournament -/
structure Player where
  number : Nat
  deriving Repr

/-- Represents the tournament -/
def Tournament :=
  {players : Finset Player // players.card = 1024 ∧ ∀ p ∈ players, p.number ≤ 1024}

/-- Predicate for whether a player wins against another player -/
def wins (p1 p2 : Player) : Prop :=
  p1.number < p2.number ∧ p2.number - p1.number > 2

/-- The winner of the tournament -/
def tournamentWinner (t : Tournament) : Player :=
  sorry

/-- The theorem stating the maximum qualification number of the winner -/
theorem max_winner_number (t : Tournament) :
  (tournamentWinner t).number ≤ 20 :=
sorry

end NUMINAMATH_CALUDE_max_winner_number_l1960_196045


namespace NUMINAMATH_CALUDE_six_right_triangles_with_smallest_perimeter_l1960_196051

/-- A structure representing a triangle with integer sides -/
structure Triangle where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Check if a triangle is a right triangle -/
def is_right_triangle (t : Triangle) : Prop :=
  t.a ^ 2 + t.b ^ 2 = t.c ^ 2

/-- Calculate the perimeter of a triangle -/
def perimeter (t : Triangle) : ℕ :=
  t.a + t.b + t.c

/-- The set of six triangles with their side lengths -/
def six_triangles : List Triangle :=
  [⟨120, 288, 312⟩, ⟨144, 270, 306⟩, ⟨72, 320, 328⟩,
   ⟨45, 336, 339⟩, ⟨80, 315, 325⟩, ⟨180, 240, 300⟩]

/-- Theorem: There exist 6 rational right triangles with the same smallest possible perimeter of 720 -/
theorem six_right_triangles_with_smallest_perimeter :
  (∀ t ∈ six_triangles, is_right_triangle t) ∧
  (∀ t ∈ six_triangles, perimeter t = 720) ∧
  (∀ t : Triangle, is_right_triangle t → perimeter t < 720 → t ∉ six_triangles) :=
sorry

end NUMINAMATH_CALUDE_six_right_triangles_with_smallest_perimeter_l1960_196051


namespace NUMINAMATH_CALUDE_equation_solution_l1960_196036

theorem equation_solution : ∃! x : ℝ, (1 / (x + 10) + 1 / (x + 8) = 1 / (x + 11) + 1 / (x + 7)) ∧ x = -9 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1960_196036


namespace NUMINAMATH_CALUDE_gourmet_smores_cost_l1960_196091

/-- Represents the cost and pack information for an ingredient --/
structure IngredientInfo where
  single_cost : ℚ
  pack_size : ℕ
  pack_cost : ℚ

/-- Calculates the minimum cost to buy a certain quantity of an ingredient --/
def min_cost (info : IngredientInfo) (quantity : ℕ) : ℚ :=
  let packs_needed := (quantity + info.pack_size - 1) / info.pack_size
  packs_needed * info.pack_cost

/-- Calculates the total cost for all ingredients --/
def total_cost (people : ℕ) (smores_per_person : ℕ) : ℚ :=
  let graham_crackers := min_cost ⟨0.1, 20, 1.8⟩ (people * smores_per_person * 1)
  let marshmallows := min_cost ⟨0.15, 15, 2.0⟩ (people * smores_per_person * 1)
  let chocolate := min_cost ⟨0.25, 10, 2.0⟩ (people * smores_per_person * 1)
  let caramel := min_cost ⟨0.2, 25, 4.5⟩ (people * smores_per_person * 2)
  let toffee := min_cost ⟨0.05, 50, 2.0⟩ (people * smores_per_person * 4)
  graham_crackers + marshmallows + chocolate + caramel + toffee

theorem gourmet_smores_cost : total_cost 8 3 = 26.6 := by
  sorry

end NUMINAMATH_CALUDE_gourmet_smores_cost_l1960_196091


namespace NUMINAMATH_CALUDE_chad_bbq_ice_cost_l1960_196016

/-- The cost of ice for Chad's BBQ -/
def bbq_ice_cost (total_people : ℕ) (ice_per_person : ℕ) (package_size : ℕ) (cost_per_package : ℚ) : ℚ :=
  let total_ice := total_people * ice_per_person
  let packages_needed := (total_ice + package_size - 1) / package_size
  packages_needed * cost_per_package

/-- Theorem: The cost of ice for Chad's BBQ is $27 -/
theorem chad_bbq_ice_cost :
  bbq_ice_cost 20 3 10 (4.5 : ℚ) = 27 := by
  sorry

end NUMINAMATH_CALUDE_chad_bbq_ice_cost_l1960_196016


namespace NUMINAMATH_CALUDE_stating_shop_owner_cheat_percentage_l1960_196042

/-- Represents the percentage by which the shop owner cheats -/
def cheat_percentage : ℝ := 22.22222222222222

/-- Represents the profit percentage of the shop owner -/
def profit_percentage : ℝ := 22.22222222222222

/-- 
Theorem stating that if a shop owner cheats by the same percentage while buying and selling,
and their profit percentage is 22.22222222222222%, then the cheat percentage is also 22.22222222222222%.
-/
theorem shop_owner_cheat_percentage :
  cheat_percentage = profit_percentage :=
sorry

end NUMINAMATH_CALUDE_stating_shop_owner_cheat_percentage_l1960_196042


namespace NUMINAMATH_CALUDE_sum_of_series_l1960_196071

def series_sum (n : ℕ) : ℕ :=
  n * (n + 1) / 2 - 1

theorem sum_of_series :
  series_sum 15 - series_sum 1 = 91 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_series_l1960_196071


namespace NUMINAMATH_CALUDE_no_prime_factor_3j_plus_2_l1960_196043

/-- A number is a cube if it's the cube of some integer -/
def IsCube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

/-- The number of divisors of a natural number -/
def NumDivisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

/-- A number is the smallest with k divisors if it has k divisors and no smaller number has k divisors -/
def IsSmallestWithKDivisors (n k : ℕ) : Prop :=
  NumDivisors n = k ∧ ∀ m < n, NumDivisors m ≠ k

theorem no_prime_factor_3j_plus_2 (n k : ℕ) (h1 : IsSmallestWithKDivisors n k) (h2 : IsCube n) :
  ¬∃ (p : ℕ), Nat.Prime p ∧ (∃ j : ℕ, p = 3*j + 2) ∧ p ∣ k := by
  sorry

end NUMINAMATH_CALUDE_no_prime_factor_3j_plus_2_l1960_196043


namespace NUMINAMATH_CALUDE_largest_n_with_perfect_squares_l1960_196074

theorem largest_n_with_perfect_squares : ∃ (N : ℤ),
  (∃ (a : ℤ), N + 496 = a^2) ∧
  (∃ (b : ℤ), N + 224 = b^2) ∧
  (∀ (M : ℤ), M > N →
    ¬(∃ (x : ℤ), M + 496 = x^2) ∨
    ¬(∃ (y : ℤ), M + 224 = y^2)) ∧
  N = 4265 :=
sorry

end NUMINAMATH_CALUDE_largest_n_with_perfect_squares_l1960_196074


namespace NUMINAMATH_CALUDE_largest_angle_in_triangle_l1960_196052

theorem largest_angle_in_triangle (a b c : ℝ) (A : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  a + a * c + b * c = 2 * b →
  a - a * c + b * c = 2 * c →
  a = b + c + 2 * b * c * Real.cos A →
  A = 2 * π / 3 :=
by sorry

end NUMINAMATH_CALUDE_largest_angle_in_triangle_l1960_196052


namespace NUMINAMATH_CALUDE_six_is_simplified_quadratic_radical_l1960_196003

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def is_simplified_quadratic_radical (n : ℕ) : Prop :=
  n ≠ 0 ∧ ¬ is_perfect_square n ∧ ∀ m : ℕ, m > 1 → is_perfect_square m → ¬ (m ∣ n)

theorem six_is_simplified_quadratic_radical :
  is_simplified_quadratic_radical 6 :=
sorry

end NUMINAMATH_CALUDE_six_is_simplified_quadratic_radical_l1960_196003


namespace NUMINAMATH_CALUDE_first_player_wins_l1960_196072

/-- Represents a rectangular grid --/
structure Grid :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a player in the game --/
inductive Player
  | First
  | Second

/-- Represents the game state --/
structure GameState :=
  (grid : Grid)
  (currentPlayer : Player)
  (shadedCells : Set (ℕ × ℕ))

/-- Defines a valid move in the game --/
def ValidMove (state : GameState) (move : Set (ℕ × ℕ)) : Prop :=
  ∀ cell ∈ move,
    cell.1 ≤ state.grid.rows ∧
    cell.2 ≤ state.grid.cols ∧
    cell ∉ state.shadedCells

/-- Defines the winning condition --/
def IsWinningState (state : GameState) : Prop :=
  ∀ move : Set (ℕ × ℕ), ¬(ValidMove state move)

/-- Theorem: The first player has a winning strategy in a 19 × 94 grid game --/
theorem first_player_wins :
  ∃ (strategy : GameState → Set (ℕ × ℕ)),
    let initialState := GameState.mk (Grid.mk 19 94) Player.First ∅
    ∀ (game : ℕ → GameState),
      game 0 = initialState →
      (∀ n : ℕ,
        (game n).currentPlayer = Player.First →
        ValidMove (game n) (strategy (game n)) ∧
        (game (n + 1)).shadedCells = (game n).shadedCells ∪ (strategy (game n))) →
      (∀ n : ℕ,
        (game n).currentPlayer = Player.Second →
        ∃ move,
          ValidMove (game n) move ∧
          (game (n + 1)).shadedCells = (game n).shadedCells ∪ move) →
      ∃ m : ℕ, IsWinningState (game m) ∧ (game m).currentPlayer = Player.First :=
sorry

end NUMINAMATH_CALUDE_first_player_wins_l1960_196072


namespace NUMINAMATH_CALUDE_sugar_and_salt_pricing_l1960_196062

/-- Given the price of sugar and salt, prove the cost of a specific quantity -/
theorem sugar_and_salt_pricing
  (price_2kg_sugar_5kg_salt : ℝ)
  (price_1kg_sugar : ℝ)
  (h1 : price_2kg_sugar_5kg_salt = 5.50)
  (h2 : price_1kg_sugar = 1.50) :
  3 * price_1kg_sugar + (price_2kg_sugar_5kg_salt - 2 * price_1kg_sugar) / 5 = 5 :=
by sorry

end NUMINAMATH_CALUDE_sugar_and_salt_pricing_l1960_196062


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1960_196081

-- Define set A
def A : Set ℝ := {x | x^2 ≤ 4*x}

-- Define set B
def B : Set ℝ := {x | |x| ≥ 2}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 2 ≤ x ∧ x ≤ 4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1960_196081


namespace NUMINAMATH_CALUDE_smallest_n_with_odd_digits_l1960_196098

def all_digits_odd (n : ℕ) : Prop :=
  ∀ d, d ∈ (97 * n).digits 10 → d % 2 = 1

theorem smallest_n_with_odd_digits :
  ∀ n : ℕ, n > 1 →
    (all_digits_odd n → n ≥ 35) ∧
    (all_digits_odd 35) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_with_odd_digits_l1960_196098


namespace NUMINAMATH_CALUDE_martha_clothes_count_l1960_196093

/-- Calculates the total number of clothes Martha takes home from a shopping trip -/
def total_clothes (jackets_bought : ℕ) (tshirts_bought : ℕ) : ℕ :=
  let free_jackets := jackets_bought / 2
  let free_tshirts := tshirts_bought / 3
  jackets_bought + free_jackets + tshirts_bought + free_tshirts

/-- Proves that Martha takes home 18 clothes given the conditions of the problem -/
theorem martha_clothes_count :
  total_clothes 4 9 = 18 := by
  sorry

end NUMINAMATH_CALUDE_martha_clothes_count_l1960_196093


namespace NUMINAMATH_CALUDE_natural_number_power_equality_l1960_196054

theorem natural_number_power_equality (p q : ℕ) (h : p^p + q^q = p^q + q^p) : p = q := by
  sorry

end NUMINAMATH_CALUDE_natural_number_power_equality_l1960_196054


namespace NUMINAMATH_CALUDE_boys_in_class_l1960_196066

theorem boys_in_class (total : ℕ) (ratio_girls : ℕ) (ratio_boys : ℕ) (h1 : total = 56) (h2 : ratio_girls = 4) (h3 : ratio_boys = 3) : 
  (total * ratio_boys) / (ratio_girls + ratio_boys) = 24 := by
  sorry

end NUMINAMATH_CALUDE_boys_in_class_l1960_196066


namespace NUMINAMATH_CALUDE_rate_squares_sum_l1960_196031

theorem rate_squares_sum : ∃ (b j s : ℕ),
  3 * b + 2 * j + 4 * s = 70 ∧
  4 * b + 3 * j + 2 * s = 88 ∧
  b^2 + j^2 + s^2 = 405 := by
sorry

end NUMINAMATH_CALUDE_rate_squares_sum_l1960_196031


namespace NUMINAMATH_CALUDE_hotel_weekly_loss_l1960_196095

def weekly_profit_loss (operations_expenses taxes employee_salaries : ℚ) : ℚ :=
  let meetings_income := (5 / 8) * operations_expenses
  let events_income := (3 / 10) * operations_expenses
  let rooms_income := (11 / 20) * operations_expenses
  let total_income := meetings_income + events_income + rooms_income
  let total_expenses := operations_expenses + taxes + employee_salaries
  total_income - total_expenses

theorem hotel_weekly_loss :
  weekly_profit_loss 5000 1200 2500 = -1325 :=
by sorry

end NUMINAMATH_CALUDE_hotel_weekly_loss_l1960_196095


namespace NUMINAMATH_CALUDE_triangle_side_length_l1960_196033

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  A = π/3 →  -- 60 degrees in radians
  B = π/4 →  -- 45 degrees in radians
  b = 2 → 
  (a / Real.sin A = b / Real.sin B) →  -- Law of sines
  a = Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1960_196033


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_two_l1960_196087

theorem reciprocal_of_negative_two :
  (1 : ℚ) / (-2 : ℚ) = -1/2 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_two_l1960_196087


namespace NUMINAMATH_CALUDE_polynomial_multiplication_l1960_196083

theorem polynomial_multiplication (t : ℝ) :
  (3 * t^3 - 2 * t^2 + 4 * t - 1) * (2 * t^2 - 5 * t + 3) =
  6 * t^5 - 19 * t^4 + 27 * t^3 - 28 * t^2 + 17 * t - 3 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_multiplication_l1960_196083


namespace NUMINAMATH_CALUDE_compare_expressions_l1960_196026

theorem compare_expressions : -|(-5)| < -(-3) := by
  sorry

end NUMINAMATH_CALUDE_compare_expressions_l1960_196026


namespace NUMINAMATH_CALUDE_tan_eleven_pi_fourths_l1960_196077

theorem tan_eleven_pi_fourths : Real.tan (11 * π / 4) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_eleven_pi_fourths_l1960_196077


namespace NUMINAMATH_CALUDE_min_value_of_f_l1960_196007

def f (x : ℝ) : ℝ := 5 * x^2 - 30 * x + 2000

theorem min_value_of_f :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x ∧ f x_min = 1955 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1960_196007


namespace NUMINAMATH_CALUDE_chad_dog_food_packages_l1960_196076

/-- Given Chad's purchase of cat and dog food, prove he bought 2 packages of dog food -/
theorem chad_dog_food_packages : 
  ∀ (cat_packages dog_packages : ℕ),
  cat_packages = 6 →
  ∀ (cat_cans_per_package dog_cans_per_package : ℕ),
  cat_cans_per_package = 9 →
  dog_cans_per_package = 3 →
  cat_packages * cat_cans_per_package = dog_packages * dog_cans_per_package + 48 →
  dog_packages = 2 :=
by sorry

end NUMINAMATH_CALUDE_chad_dog_food_packages_l1960_196076


namespace NUMINAMATH_CALUDE_geometric_sequence_operations_l1960_196032

-- Define a geometric sequence
def IsGeometric (s : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, s (n + 1) = r * s n

-- Define the problem statement
theorem geometric_sequence_operations
  (a b : ℕ → ℝ)
  (ha : IsGeometric a)
  (hb : IsGeometric b)
  (hb_nonzero : ∀ n, b n ≠ 0) :
  IsGeometric (fun n ↦ a n * b n) ∧
  IsGeometric (fun n ↦ a n / b n) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_operations_l1960_196032


namespace NUMINAMATH_CALUDE_triangle_side_sum_range_l1960_196044

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ)
  (a b c : ℝ)
  (acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2)
  (law_of_sines : a / Real.sin A = b / Real.sin B)
  (law_of_cosines : Real.cos A = (b^2 + c^2 - a^2) / (2*b*c))

-- State the theorem
theorem triangle_side_sum_range (t : Triangle) 
  (h1 : Real.cos t.A / t.a + Real.cos t.C / t.c = Real.sin t.B * Real.sin t.C / (3 * Real.sin t.A))
  (h2 : Real.sqrt 3 * Real.sin t.C + Real.cos t.C = 2) :
  6 < t.a + t.b ∧ t.a + t.b ≤ 4 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_sum_range_l1960_196044


namespace NUMINAMATH_CALUDE_soldiers_food_calculation_l1960_196034

/-- Given the following conditions:
    1. Soldiers on the second side are given 2 pounds less food than the first side.
    2. The first side has 4000 soldiers.
    3. The second side has 500 soldiers fewer than the first side.
    4. The total amount of food both sides are eating altogether every day is 68000 pounds.

    Prove that the amount of food each soldier on the first side needs every day is 10 pounds. -/
theorem soldiers_food_calculation (food_first : ℝ) : 
  (4000 : ℝ) * food_first + (4000 - 500) * (food_first - 2) = 68000 → food_first = 10 := by
  sorry

end NUMINAMATH_CALUDE_soldiers_food_calculation_l1960_196034


namespace NUMINAMATH_CALUDE_smallest_five_digit_divisible_by_53_l1960_196023

theorem smallest_five_digit_divisible_by_53 : ∀ n : ℕ, 
  10000 ≤ n ∧ n < 100000 ∧ n % 53 = 0 → n ≥ 10017 :=
by sorry

end NUMINAMATH_CALUDE_smallest_five_digit_divisible_by_53_l1960_196023


namespace NUMINAMATH_CALUDE_function_decomposition_l1960_196096

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_even (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

theorem function_decomposition (f g : ℝ → ℝ) 
  (h_odd : is_odd f) (h_even : is_even g) 
  (h_sum : ∀ x, f x + g x = 2^x + 2*x) :
  (∀ x, g x = (2^x + 2^(-x)) / 2) ∧ 
  (∀ x, f x = 2^(x-1) + 2*x - 2^(-x-1)) :=
sorry

end NUMINAMATH_CALUDE_function_decomposition_l1960_196096


namespace NUMINAMATH_CALUDE_line_difference_l1960_196060

/-- Represents a character in the script --/
structure Character where
  lines : ℕ

/-- Represents the script with three characters --/
structure Script where
  char1 : Character
  char2 : Character
  char3 : Character

/-- Theorem: The difference in lines between the first and second character is 8 --/
theorem line_difference (s : Script) : 
  s.char3.lines = 2 →
  s.char2.lines = 3 * s.char3.lines + 6 →
  s.char1.lines = 20 →
  s.char1.lines - s.char2.lines = 8 := by
  sorry


end NUMINAMATH_CALUDE_line_difference_l1960_196060


namespace NUMINAMATH_CALUDE_bons_winning_probability_l1960_196067

/-- The probability of rolling a six on a six-sided die. -/
def probSix : ℚ := 1/6

/-- The probability of not rolling a six on a six-sided die. -/
def probNotSix : ℚ := 1 - probSix

/-- The probability that B. Bons wins the game. -/
noncomputable def probBonsWins : ℚ :=
  (probNotSix * probSix) / (1 - probNotSix * probNotSix)

theorem bons_winning_probability :
  probBonsWins = 5/11 := by sorry

end NUMINAMATH_CALUDE_bons_winning_probability_l1960_196067


namespace NUMINAMATH_CALUDE_trigonometric_identities_l1960_196000

theorem trigonometric_identities :
  (∃ x : ℝ, x = 75 * π / 180 ∧ (Real.cos x)^2 = (2 - Real.sqrt 3) / 4) ∧
  (∃ y z : ℝ, y = π / 180 ∧ z = 44 * π / 180 ∧
    Real.tan y + Real.tan z + Real.tan y * Real.tan z = 1) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l1960_196000


namespace NUMINAMATH_CALUDE_correct_additional_oil_l1960_196012

/-- The amount of oil needed per cylinder in ounces -/
def oil_per_cylinder : ℕ := 8

/-- The number of cylinders in George's car -/
def num_cylinders : ℕ := 6

/-- The amount of oil already added to the engine in ounces -/
def oil_already_added : ℕ := 16

/-- The additional amount of oil needed in ounces -/
def additional_oil_needed : ℕ := oil_per_cylinder * num_cylinders - oil_already_added

theorem correct_additional_oil : additional_oil_needed = 32 := by
  sorry

end NUMINAMATH_CALUDE_correct_additional_oil_l1960_196012


namespace NUMINAMATH_CALUDE_statement_a_statement_b_statement_c_statement_d_all_statements_correct_l1960_196099

-- Statement A
theorem statement_a (a b : ℝ) : a^2 = b^2 → |a| = |b| := by
  sorry

-- Statement B
theorem statement_b (a b : ℝ) : a + b = 0 → a^3 + b^3 = 0 := by
  sorry

-- Statement C
theorem statement_c (a b : ℝ) : a < b ∧ a ≠ 0 ∧ b ≠ 0 → 1/a > 1/b := by
  sorry

-- Statement D
theorem statement_d (a : ℝ) : -1 < a ∧ a < 0 → a^3 < a^5 := by
  sorry

-- All statements are correct
theorem all_statements_correct : 
  (∀ a b : ℝ, a^2 = b^2 → |a| = |b|) ∧
  (∀ a b : ℝ, a + b = 0 → a^3 + b^3 = 0) ∧
  (∀ a b : ℝ, a < b ∧ a ≠ 0 ∧ b ≠ 0 → 1/a > 1/b) ∧
  (∀ a : ℝ, -1 < a ∧ a < 0 → a^3 < a^5) := by
  sorry

end NUMINAMATH_CALUDE_statement_a_statement_b_statement_c_statement_d_all_statements_correct_l1960_196099


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_longest_side_l1960_196019

-- Define the triangle
structure Triangle :=
  (X Y Z : ℝ × ℝ)

-- Define properties of the triangle
def isIsoscelesRight (t : Triangle) : Prop :=
  -- We don't implement the full definition, just declare it as a property
  sorry

def longestSide (t : Triangle) (side : ℝ) : Prop :=
  -- We don't implement the full definition, just declare it as a property
  sorry

def triangleArea (t : Triangle) : ℝ :=
  -- We don't implement the full calculation, just declare it as a function
  sorry

-- Main theorem
theorem isosceles_right_triangle_longest_side 
  (t : Triangle) 
  (h1 : isIsoscelesRight t) 
  (h2 : longestSide t (dist t.X t.Y)) 
  (h3 : triangleArea t = 49) : 
  dist t.X t.Y = 14 :=
sorry

-- Note: dist is a function that calculates the distance between two points

end NUMINAMATH_CALUDE_isosceles_right_triangle_longest_side_l1960_196019


namespace NUMINAMATH_CALUDE_middle_number_not_unique_l1960_196057

/-- Represents a configuration of three cards with positive integers. -/
structure CardConfiguration where
  left : Nat
  middle : Nat
  right : Nat
  sum_is_15 : left + middle + right = 15
  increasing : left < middle ∧ middle < right

/-- Predicate to check if Alan can determine the other two numbers. -/
def alan_cant_determine (config : CardConfiguration) : Prop :=
  ∃ (other_config : CardConfiguration), other_config.left = config.left ∧ other_config ≠ config

/-- Predicate to check if Carlos can determine the other two numbers. -/
def carlos_cant_determine (config : CardConfiguration) : Prop :=
  ∃ (other_config : CardConfiguration), other_config.right = config.right ∧ other_config ≠ config

/-- Predicate to check if Brenda can determine the other two numbers. -/
def brenda_cant_determine (config : CardConfiguration) : Prop :=
  ∃ (other_config : CardConfiguration), other_config.middle = config.middle ∧ other_config ≠ config

/-- The main theorem stating that the middle number cannot be uniquely determined. -/
theorem middle_number_not_unique : ∃ (config1 config2 : CardConfiguration),
  config1.middle ≠ config2.middle ∧
  alan_cant_determine config1 ∧
  alan_cant_determine config2 ∧
  carlos_cant_determine config1 ∧
  carlos_cant_determine config2 ∧
  brenda_cant_determine config1 ∧
  brenda_cant_determine config2 :=
sorry

end NUMINAMATH_CALUDE_middle_number_not_unique_l1960_196057


namespace NUMINAMATH_CALUDE_inequality_and_ln2_bounds_l1960_196021

theorem inequality_and_ln2_bounds (x a : ℝ) (h1 : 0 < x) (h2 : x < a) :
  (2 * x / a < ∫ t in (a - x)..(a + x), 1 / t) ∧
  (∫ t in (a - x)..(a + x), 1 / t < x * (1 / (a + x) + 1 / (a - x))) ∧
  (0.68 < Real.log 2) ∧ (Real.log 2 < 0.71) := by
  sorry

end NUMINAMATH_CALUDE_inequality_and_ln2_bounds_l1960_196021


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l1960_196005

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_a2 : a 2 = 3)
  (h_sum : a 3 + a 4 = 9) :
  a 1 * a 6 = 14 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l1960_196005


namespace NUMINAMATH_CALUDE_football_goals_theorem_l1960_196029

theorem football_goals_theorem (goals_fifth_match : ℕ) (average_increase : ℚ) : 
  goals_fifth_match = 3 → average_increase = 0.2 → 
  ∃ (total_goals : ℕ), total_goals = 11 ∧ 
  (total_goals : ℚ) / 5 = (total_goals - goals_fifth_match : ℚ) / 4 + average_increase :=
by sorry

end NUMINAMATH_CALUDE_football_goals_theorem_l1960_196029


namespace NUMINAMATH_CALUDE_female_teachers_count_l1960_196059

/-- The number of teachers in the group -/
def total_teachers : ℕ := 5

/-- The probability of selecting a female teacher -/
def prob_female : ℚ := 7/10

/-- Calculates the number of combinations of k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- Calculates the probability of selecting two male teachers given x female teachers -/
def prob_two_male (x : ℕ) : ℚ :=
  1 - (choose (total_teachers - x) 2 : ℚ) / (choose total_teachers 2 : ℚ)

theorem female_teachers_count :
  ∃ x : ℕ, x ≤ total_teachers ∧ prob_two_male x = 1 - prob_female :=
sorry

end NUMINAMATH_CALUDE_female_teachers_count_l1960_196059


namespace NUMINAMATH_CALUDE_equilateral_triangle_expression_bound_l1960_196050

theorem equilateral_triangle_expression_bound (a : ℝ) (h : a > 0) : (3 * a^2) / (3 * a) > 0 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_expression_bound_l1960_196050


namespace NUMINAMATH_CALUDE_star_computation_l1960_196006

-- Define the * operation
def star (a b : ℚ) : ℚ := (a^2 - b^2) / (1 - a * b)

-- Theorem statement
theorem star_computation : star 1 (star 2 (star 3 4)) = -18 := by
  sorry

end NUMINAMATH_CALUDE_star_computation_l1960_196006


namespace NUMINAMATH_CALUDE_even_odd_sum_difference_prove_even_odd_sum_difference_l1960_196047

theorem even_odd_sum_difference : ℕ → Prop :=
  fun n =>
    let even_sum := (n + 1) * (2 + 2 * n)
    let odd_sum := n * (1 + 2 * n - 1)
    even_sum - odd_sum = 6017

theorem prove_even_odd_sum_difference :
  even_odd_sum_difference 2003 := by sorry

end NUMINAMATH_CALUDE_even_odd_sum_difference_prove_even_odd_sum_difference_l1960_196047


namespace NUMINAMATH_CALUDE_friends_video_count_l1960_196053

/-- The number of videos watched by three friends. -/
def total_videos (kelsey ekon uma : ℕ) : ℕ := kelsey + ekon + uma

/-- Theorem stating the total number of videos watched by the three friends. -/
theorem friends_video_count :
  ∀ (kelsey ekon uma : ℕ),
  kelsey = 160 →
  kelsey = ekon + 43 →
  uma = ekon + 17 →
  total_videos kelsey ekon uma = 411 :=
by
  sorry

end NUMINAMATH_CALUDE_friends_video_count_l1960_196053


namespace NUMINAMATH_CALUDE_shopkeeper_profit_l1960_196010

/-- Proves that if a shopkeeper sells an article with a 5% discount and earns a 23.5% profit,
    then selling the same article without a discount would result in a 30% profit. -/
theorem shopkeeper_profit (cost_price : ℝ) (cost_price_pos : cost_price > 0) :
  let discount_rate := 0.05
  let profit_rate_with_discount := 0.235
  let selling_price_with_discount := cost_price * (1 + profit_rate_with_discount)
  let marked_price := selling_price_with_discount / (1 - discount_rate)
  let profit_without_discount := marked_price - cost_price
  profit_without_discount / cost_price = 0.3 := by
sorry

end NUMINAMATH_CALUDE_shopkeeper_profit_l1960_196010


namespace NUMINAMATH_CALUDE_necessary_condition_for_inequality_l1960_196090

theorem necessary_condition_for_inequality (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc 2 3 → x^2 - a ≤ 0) → a ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_necessary_condition_for_inequality_l1960_196090


namespace NUMINAMATH_CALUDE_sister_brother_product_is_twelve_l1960_196038

/-- Represents a family with siblings -/
structure Family :=
  (num_sisters : ℕ)
  (num_brothers : ℕ)

/-- Calculates the product of sisters and brothers for a sister in the family -/
def sister_brother_product (f : Family) : ℕ :=
  (f.num_sisters - 1) * f.num_brothers

/-- Theorem stating that for a family where one sibling has 4 sisters and 4 brothers,
    the product of sisters and brothers for any sister is 12 -/
theorem sister_brother_product_is_twelve (f : Family) 
  (h : f.num_sisters = 4 ∧ f.num_brothers = 4) : 
  sister_brother_product f = 12 := by
  sorry

#eval sister_brother_product ⟨4, 4⟩

end NUMINAMATH_CALUDE_sister_brother_product_is_twelve_l1960_196038


namespace NUMINAMATH_CALUDE_money_left_calculation_l1960_196079

def initial_amount : ℝ := 200

def notebook_cost : ℝ := 4
def book_cost : ℝ := 12
def pen_cost : ℝ := 2
def sticker_pack_cost : ℝ := 6
def shoes_cost : ℝ := 40
def tshirt_cost : ℝ := 18

def notebooks_bought : ℕ := 7
def books_bought : ℕ := 2
def pens_bought : ℕ := 5
def sticker_packs_bought : ℕ := 3

def sales_tax_rate : ℝ := 0.05

def lunch_cost : ℝ := 15
def tip_amount : ℝ := 3
def transportation_cost : ℝ := 8
def charity_amount : ℝ := 10

def total_mall_purchase_cost : ℝ :=
  notebook_cost * notebooks_bought +
  book_cost * books_bought +
  pen_cost * pens_bought +
  sticker_pack_cost * sticker_packs_bought +
  shoes_cost +
  tshirt_cost

def total_mall_cost_with_tax : ℝ :=
  total_mall_purchase_cost * (1 + sales_tax_rate)

def total_expenses : ℝ :=
  total_mall_cost_with_tax +
  lunch_cost +
  tip_amount +
  transportation_cost +
  charity_amount

theorem money_left_calculation :
  initial_amount - total_expenses = 19.10 := by
  sorry

end NUMINAMATH_CALUDE_money_left_calculation_l1960_196079


namespace NUMINAMATH_CALUDE_largest_perfect_square_factor_of_4410_l1960_196055

def largest_perfect_square_factor (n : ℕ) : ℕ := sorry

theorem largest_perfect_square_factor_of_4410 :
  largest_perfect_square_factor 4410 = 441 := by sorry

end NUMINAMATH_CALUDE_largest_perfect_square_factor_of_4410_l1960_196055


namespace NUMINAMATH_CALUDE_midpoint_coordinate_sum_l1960_196028

theorem midpoint_coordinate_sum (a b c d e f : ℝ) 
  (h1 : a + b + c = 15) 
  (h2 : d + e + f = 9) : 
  (a + b) / 2 + (b + c) / 2 + (c + a) / 2 = 15 ∧ 
  (d + e) / 2 + (e + f) / 2 + (f + d) / 2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_sum_l1960_196028


namespace NUMINAMATH_CALUDE_sequence_value_l1960_196009

/-- Given a sequence {aₙ} satisfying aₙ₊₁ = 1 / (1 - aₙ) for all n ≥ 1,
    and a₂ = 2, prove that a₁ = 1/2 -/
theorem sequence_value (a : ℕ → ℚ)
  (h₁ : ∀ n : ℕ, n ≥ 1 → a (n + 1) = 1 / (1 - a n))
  (h₂ : a 2 = 2) :
  a 1 = 1/2 := by
sorry

end NUMINAMATH_CALUDE_sequence_value_l1960_196009


namespace NUMINAMATH_CALUDE_investment_growth_rate_l1960_196025

def annual_growth_rate (growth_rate : ℝ) (compounding_periods : ℕ) : ℝ :=
  ((growth_rate ^ (1 / compounding_periods)) ^ compounding_periods - 1) * 100

theorem investment_growth_rate 
  (P : ℝ) 
  (t : ℕ) 
  (h1 : P > 0) 
  (h2 : 1 ≤ t ∧ t ≤ 5) : 
  annual_growth_rate 1.20 2 = 20 := by
sorry

end NUMINAMATH_CALUDE_investment_growth_rate_l1960_196025


namespace NUMINAMATH_CALUDE_smallest_linear_combination_2023_54321_l1960_196070

theorem smallest_linear_combination_2023_54321 :
  ∃ (k : ℕ), k > 0 ∧ (∃ (m n : ℤ), k = 2023 * m + 54321 * n) ∧
  ∀ (j : ℕ), j > 0 → (∃ (x y : ℤ), j = 2023 * x + 54321 * y) → k ≤ j :=
by sorry

end NUMINAMATH_CALUDE_smallest_linear_combination_2023_54321_l1960_196070


namespace NUMINAMATH_CALUDE_average_and_difference_l1960_196041

theorem average_and_difference (x : ℝ) : 
  (40 + x + 15) / 3 = 35 → |x - 40| = 10 := by
  sorry

end NUMINAMATH_CALUDE_average_and_difference_l1960_196041


namespace NUMINAMATH_CALUDE_derivative_at_zero_l1960_196078

/-- Given a function f where f(x) = x^2 + 2f'(1), prove that f'(0) = 0 -/
theorem derivative_at_zero (f : ℝ → ℝ) (h : ∀ x, f x = x^2 + 2 * (deriv f 1)) :
  deriv f 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_derivative_at_zero_l1960_196078


namespace NUMINAMATH_CALUDE_goose_eggs_count_l1960_196011

theorem goose_eggs_count (
  total_eggs : ℕ
) (
  hatched_ratio : Real
) (
  first_month_survival_ratio : Real
) (
  first_year_death_ratio : Real
) (
  first_year_survivors : ℕ
) (
  h1 : hatched_ratio = 1 / 4
) (
  h2 : first_month_survival_ratio = 4 / 5
) (
  h3 : first_year_death_ratio = 2 / 5
) (
  h4 : first_year_survivors = 120
) (
  h5 : (hatched_ratio * first_month_survival_ratio * (1 - first_year_death_ratio) * total_eggs : Real) = first_year_survivors
) : total_eggs = 800 := by
  sorry

end NUMINAMATH_CALUDE_goose_eggs_count_l1960_196011


namespace NUMINAMATH_CALUDE_nickel_count_l1960_196015

def total_cents : ℕ := 400
def num_quarters : ℕ := 10
def num_dimes : ℕ := 12
def quarter_value : ℕ := 25
def dime_value : ℕ := 10
def nickel_value : ℕ := 5

theorem nickel_count : 
  (total_cents - (num_quarters * quarter_value + num_dimes * dime_value)) / nickel_value = 6 := by
  sorry

end NUMINAMATH_CALUDE_nickel_count_l1960_196015


namespace NUMINAMATH_CALUDE_simple_interest_problem_l1960_196040

def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * ((1 + rate) ^ time - 1)

def simple_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * rate * time

theorem simple_interest_problem (P : ℝ) : 
  simple_interest P 0.08 3 = 
  (1/2) * compound_interest 4000 0.1 2 → P = 1750 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l1960_196040


namespace NUMINAMATH_CALUDE_collinear_points_x_value_l1960_196037

/-- Given three points in a 2D plane, checks if they are collinear --/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (y₂ - y₁) * (x₃ - x₁) = (y₃ - y₁) * (x₂ - x₁)

/-- Theorem: If A(-1, -2), B(4, 8), and C(5, x) are collinear, then x = 10 --/
theorem collinear_points_x_value :
  collinear (-1) (-2) 4 8 5 x → x = 10 :=
by
  sorry

#check collinear_points_x_value

end NUMINAMATH_CALUDE_collinear_points_x_value_l1960_196037


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l1960_196065

theorem quadratic_roots_relation (b c : ℝ) : 
  (∀ x, x^2 + b*x + c = 0 ↔ ∃ y, 2*y^2 - 7*y + 6 = 0 ∧ x = y - 3) → 
  c = 3/2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l1960_196065


namespace NUMINAMATH_CALUDE_unbroken_seashells_l1960_196027

theorem unbroken_seashells (total : ℕ) (broken : ℕ) (h1 : total = 23) (h2 : broken = 11) :
  total - broken = 12 := by
  sorry

end NUMINAMATH_CALUDE_unbroken_seashells_l1960_196027


namespace NUMINAMATH_CALUDE_child_b_share_l1960_196089

def total_money : ℕ := 12000
def num_children : ℕ := 5
def ratio : List ℕ := [2, 3, 4, 5, 6]

theorem child_b_share :
  let total_parts := ratio.sum
  let part_value := total_money / total_parts
  let child_b_parts := ratio[1]
  child_b_parts * part_value = 1800 := by sorry

end NUMINAMATH_CALUDE_child_b_share_l1960_196089


namespace NUMINAMATH_CALUDE_no_ruler_for_quadratic_sum_l1960_196088

-- Define the type of monotonic functions on [0, 10]
def MonotonicOn (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x ∧ x ≤ y ∧ y ≤ 10 → f x ≤ f y

-- State the theorem
theorem no_ruler_for_quadratic_sum :
  ¬ ∃ (f g h : ℝ → ℝ),
    (MonotonicOn f) ∧ (MonotonicOn g) ∧ (MonotonicOn h) ∧
    (∀ x y, 0 ≤ x ∧ x ≤ 10 ∧ 0 ≤ y ∧ y ≤ 10 →
      f x + g y = h (x^2 + x*y + y^2)) :=
by sorry

end NUMINAMATH_CALUDE_no_ruler_for_quadratic_sum_l1960_196088


namespace NUMINAMATH_CALUDE_inequality_properties_l1960_196039

theorem inequality_properties (a b : ℝ) (h : a < b ∧ b < 0) :
  (1 / a > 1 / b) ∧ (a^2 > b^2) ∧ (a * b > b^2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_properties_l1960_196039


namespace NUMINAMATH_CALUDE_five_balls_three_boxes_l1960_196061

/-- The number of ways to place n distinguishable balls into k indistinguishable boxes -/
def placeBalls (n k : ℕ) : ℕ := sorry

/-- Theorem: There are 31 ways to place 5 distinguishable balls into 3 indistinguishable boxes -/
theorem five_balls_three_boxes : placeBalls 5 3 = 31 := by sorry

end NUMINAMATH_CALUDE_five_balls_three_boxes_l1960_196061


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1960_196084

theorem inequality_solution_set (x : ℝ) : -x^2 + 2*x + 3 ≥ 0 ↔ x ∈ Set.Icc (-1) 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1960_196084


namespace NUMINAMATH_CALUDE_farmers_wheat_harvest_l1960_196063

/-- The farmer's wheat harvest problem -/
theorem farmers_wheat_harvest 
  (estimated_harvest : ℕ) 
  (additional_harvest : ℕ) 
  (h1 : estimated_harvest = 48097)
  (h2 : additional_harvest = 684) :
  estimated_harvest + additional_harvest = 48781 :=
by sorry

end NUMINAMATH_CALUDE_farmers_wheat_harvest_l1960_196063


namespace NUMINAMATH_CALUDE_largest_five_digit_with_product_l1960_196046

def digit_product (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) * digit_product (n / 10)

theorem largest_five_digit_with_product : 
  ∀ n : ℕ, 
    10000 ≤ n ∧ n < 100000 ∧ 
    digit_product n = 9 * 8 * 7 * 6 * 5 → 
    n ≤ 98765 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_five_digit_with_product_l1960_196046


namespace NUMINAMATH_CALUDE_integer_average_l1960_196073

theorem integer_average (a b c d e : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 →
  max a (max b (max c (max d e))) - min a (min b (min c (min d e))) = 10 →
  max a (max b (max c (max d e))) ≤ 68 →
  (a + b + c + d + e) / 5 = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_integer_average_l1960_196073


namespace NUMINAMATH_CALUDE_rope_cutting_problem_l1960_196018

theorem rope_cutting_problem (rope1 rope2 rope3 rope4 : ℕ) 
  (h1 : rope1 = 30) (h2 : rope2 = 45) (h3 : rope3 = 60) (h4 : rope4 = 75) :
  Nat.gcd rope1 (Nat.gcd rope2 (Nat.gcd rope3 rope4)) = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_rope_cutting_problem_l1960_196018


namespace NUMINAMATH_CALUDE_sin_cos_pi_12_star_l1960_196024

-- Define the custom operation
def star (a b : ℝ) : ℝ := a^2 - a*b - b^2

-- State the theorem
theorem sin_cos_pi_12_star : 
  star (Real.sin (π/12)) (Real.cos (π/12)) = -(1 + 2*Real.sqrt 3) / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_pi_12_star_l1960_196024


namespace NUMINAMATH_CALUDE_bridge_length_proof_l1960_196048

/-- Given a train that crosses a bridge and passes a lamp post, prove the length of the bridge. -/
theorem bridge_length_proof (train_length : ℝ) (bridge_crossing_time : ℝ) (lamp_post_passing_time : ℝ)
  (h1 : train_length = 400)
  (h2 : bridge_crossing_time = 45)
  (h3 : lamp_post_passing_time = 15) :
  (bridge_crossing_time * train_length / lamp_post_passing_time) - train_length = 800 :=
by sorry

end NUMINAMATH_CALUDE_bridge_length_proof_l1960_196048


namespace NUMINAMATH_CALUDE_rotten_bananas_percentage_l1960_196064

theorem rotten_bananas_percentage (total_oranges total_bananas : ℕ)
  (rotten_oranges_percent good_fruits_percent : ℚ) :
  total_oranges = 600 →
  total_bananas = 400 →
  rotten_oranges_percent = 15 / 100 →
  good_fruits_percent = 898 / 1000 →
  (total_bananas - (good_fruits_percent * (total_oranges + total_bananas : ℚ) -
    ((1 - rotten_oranges_percent) * total_oranges))) / total_bananas = 3 / 100 := by
  sorry

end NUMINAMATH_CALUDE_rotten_bananas_percentage_l1960_196064


namespace NUMINAMATH_CALUDE_opposite_of_negative_two_thirds_l1960_196069

theorem opposite_of_negative_two_thirds :
  -(-(2/3)) = 2/3 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_two_thirds_l1960_196069


namespace NUMINAMATH_CALUDE_negative_difference_l1960_196097

theorem negative_difference (a b : ℝ) : -(a - b) = -a + b := by
  sorry

end NUMINAMATH_CALUDE_negative_difference_l1960_196097


namespace NUMINAMATH_CALUDE_remainder_7n_mod_4_l1960_196013

theorem remainder_7n_mod_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_7n_mod_4_l1960_196013


namespace NUMINAMATH_CALUDE_curve_C_equation_min_distance_QM_l1960_196082

-- Define points A and B
def A : ℝ × ℝ := (-1, 2)
def B : ℝ × ℝ := (0, 1)

-- Define the distance condition for point P
def distance_condition (P : ℝ × ℝ) : Prop :=
  (P.1 + 1)^2 + (P.2 - 2)^2 = 2 * (P.1^2 + (P.2 - 1)^2)

-- Define curve C
def C : Set (ℝ × ℝ) := {P | distance_condition P}

-- Define line l₁
def l₁ : Set (ℝ × ℝ) := {Q | 3 * Q.1 - 4 * Q.2 + 12 = 0}

-- Theorem for the equation of curve C
theorem curve_C_equation : C = {P : ℝ × ℝ | (P.1 - 1)^2 + P.2^2 = 4} := by sorry

-- Theorem for the minimum distance
theorem min_distance_QM : 
  ∀ Q ∈ l₁, ∃ M ∈ C, ∀ M' ∈ C, dist Q M ≤ dist Q M' ∧ dist Q M = Real.sqrt 5 := by sorry


end NUMINAMATH_CALUDE_curve_C_equation_min_distance_QM_l1960_196082


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1960_196086

theorem quadratic_inequality_solution_set (a x : ℝ) :
  let inequality := a * x^2 + (a - 1) * x - 1 < 0
  (a = 0 → (inequality ↔ x > -1)) ∧
  (a > 0 → (inequality ↔ -1 < x ∧ x < 1/a)) ∧
  (-1 < a ∧ a < 0 → (inequality ↔ x < 1/a ∨ x > -1)) ∧
  (a = -1 → (inequality ↔ x ≠ -1)) ∧
  (a < -1 → (inequality ↔ x < -1 ∨ x > 1/a)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1960_196086


namespace NUMINAMATH_CALUDE_probability_even_sum_l1960_196092

def card_set : Set Nat := {1, 2, 3, 4, 5, 6, 7, 8, 9}

def is_even_sum (pair : Nat × Nat) : Bool :=
  (pair.1 + pair.2) % 2 == 0

def total_combinations : Nat :=
  Nat.choose 9 2

def even_sum_combinations : Nat :=
  Nat.choose 4 2 + Nat.choose 5 2

theorem probability_even_sum :
  (even_sum_combinations : ℚ) / total_combinations = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_probability_even_sum_l1960_196092
