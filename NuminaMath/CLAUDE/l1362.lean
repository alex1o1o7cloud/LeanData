import Mathlib

namespace NUMINAMATH_CALUDE_blue_marble_ratio_l1362_136256

/-- Proves that the ratio of blue marbles to total marbles is 1:2 -/
theorem blue_marble_ratio (total : ℕ) (red : ℕ) (green : ℕ) (yellow : ℕ) (blue : ℕ) : 
  total = 164 → 
  red = total / 4 →
  green = 27 →
  yellow = 14 →
  blue = total - (red + green + yellow) →
  (blue : ℚ) / total = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_blue_marble_ratio_l1362_136256


namespace NUMINAMATH_CALUDE_point_P_properties_l1362_136243

def P (a : ℝ) : ℝ × ℝ := (-3*a - 4, 2 + a)

def Q : ℝ × ℝ := (5, 8)

theorem point_P_properties :
  (∀ a : ℝ, P a = (2, 0) → (P a).2 = 0) ∧
  (∀ a : ℝ, (P a).1 = Q.1 → P a = (5, -1)) := by
  sorry

end NUMINAMATH_CALUDE_point_P_properties_l1362_136243


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1362_136290

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  S : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  sum_formula : ∀ n, S n = n * (2 * a 1 + (n - 1) * d) / 2

/-- Theorem: If S_7/7 - S_4/4 = 3 for an arithmetic sequence, then its common difference is 2 -/
theorem arithmetic_sequence_common_difference
  (seq : ArithmeticSequence)
  (h : seq.S 7 / 7 - seq.S 4 / 4 = 3) :
  seq.d = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1362_136290


namespace NUMINAMATH_CALUDE_forest_gathering_handshakes_count_l1362_136211

/-- The number of handshakes at the Forest Gathering -/
def forest_gathering_handshakes : ℕ :=
  let total_gremlins : ℕ := 30
  let total_pixies : ℕ := 12
  let unfriendly_gremlins : ℕ := total_gremlins / 2
  let friendly_gremlins : ℕ := total_gremlins - unfriendly_gremlins
  
  -- Handshakes among friendly gremlins
  let friendly_gremlin_handshakes : ℕ := friendly_gremlins * (friendly_gremlins - 1) / 2
  
  -- Handshakes between friendly and unfriendly gremlins
  let mixed_gremlin_handshakes : ℕ := friendly_gremlins * unfriendly_gremlins
  
  -- Handshakes between all gremlins and pixies
  let gremlin_pixie_handshakes : ℕ := total_gremlins * total_pixies
  
  -- Total handshakes
  friendly_gremlin_handshakes + mixed_gremlin_handshakes + gremlin_pixie_handshakes

/-- Theorem stating that the number of handshakes at the Forest Gathering is 690 -/
theorem forest_gathering_handshakes_count : forest_gathering_handshakes = 690 := by
  sorry

end NUMINAMATH_CALUDE_forest_gathering_handshakes_count_l1362_136211


namespace NUMINAMATH_CALUDE_circle_symmetry_l1362_136204

/-- Given a circle with center (1,1) and radius √2, prove that if it's symmetric about the line y = kx + 3, then k = -2 -/
theorem circle_symmetry (k : ℝ) : 
  (∀ x y : ℝ, (x - 1)^2 + (y - 1)^2 = 2 → 
    ∃ x' y' : ℝ, (x' - 1)^2 + (y' - 1)^2 = 2 ∧ 
    ((x + x') / 2, (y + y') / 2) ∈ {(x, y) | y = k * x + 3}) →
  k = -2 :=
sorry

end NUMINAMATH_CALUDE_circle_symmetry_l1362_136204


namespace NUMINAMATH_CALUDE_molecular_weight_BaBr2_l1362_136273

/-- The atomic weight of barium in g/mol -/
def atomic_weight_Ba : ℝ := 137.33

/-- The atomic weight of bromine in g/mol -/
def atomic_weight_Br : ℝ := 79.90

/-- The number of bromine atoms in a barium bromide molecule -/
def Br_count : ℕ := 2

/-- The number of moles of barium bromide -/
def moles_BaBr2 : ℕ := 4

/-- Theorem: The molecular weight of 4 moles of Barium bromide (BaBr2) is 1188.52 grams -/
theorem molecular_weight_BaBr2 : 
  moles_BaBr2 * (atomic_weight_Ba + Br_count * atomic_weight_Br) = 1188.52 := by
  sorry

end NUMINAMATH_CALUDE_molecular_weight_BaBr2_l1362_136273


namespace NUMINAMATH_CALUDE_hexagon_count_l1362_136267

/-- Represents a regular hexagon divided into smaller equilateral triangles -/
structure DividedHexagon where
  side_length : ℕ
  num_small_triangles : ℕ
  small_triangle_side : ℕ

/-- Counts the number of regular hexagons that can be formed in a divided hexagon -/
def count_hexagons (h : DividedHexagon) : ℕ :=
  sorry

/-- Theorem stating the number of hexagons in the specific configuration -/
theorem hexagon_count (h : DividedHexagon) 
  (h_side : h.side_length = 3)
  (h_triangles : h.num_small_triangles = 54)
  (h_small_side : h.small_triangle_side = 1) :
  count_hexagons h = 36 :=
sorry

end NUMINAMATH_CALUDE_hexagon_count_l1362_136267


namespace NUMINAMATH_CALUDE_perimeter_of_C_l1362_136227

-- Define squares A, B, and C
def square_A : Real → Real := λ s ↦ 4 * s
def square_B : Real → Real := λ s ↦ 4 * s
def square_C : Real → Real := λ s ↦ 4 * s

-- Define the conditions
def perimeter_A : Real := 20
def perimeter_B : Real := 40

-- Define the relationship between side lengths
def side_C (side_A side_B : Real) : Real := 2 * (side_A + side_B)

-- Theorem to prove
theorem perimeter_of_C (side_A side_B : Real) 
  (h1 : square_A side_A = perimeter_A)
  (h2 : square_B side_B = perimeter_B)
  : square_C (side_C side_A side_B) = 120 := by
  sorry


end NUMINAMATH_CALUDE_perimeter_of_C_l1362_136227


namespace NUMINAMATH_CALUDE_pirate_coin_problem_l1362_136209

def coin_distribution (x : ℕ) : Prop :=
  let paul_coins := x
  let pete_coins := x * (x + 1) / 2
  pete_coins = 5 * paul_coins ∧ 
  paul_coins + pete_coins = 54

theorem pirate_coin_problem :
  ∃ x : ℕ, coin_distribution x :=
sorry

end NUMINAMATH_CALUDE_pirate_coin_problem_l1362_136209


namespace NUMINAMATH_CALUDE_a1_value_l1362_136278

noncomputable section

def f (x : ℝ) : ℝ :=
  if x ≥ 1 then x * Real.log x else Real.log x / x

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem a1_value (a : ℕ → ℝ) :
  is_geometric_sequence a →
  a 3 * a 4 * a 5 = 1 →
  f (a 1) + f (a 2) + f (a 3) + f (a 4) + f (a 5) + f (a 6) = 2 * a 1 →
  a 1 = Real.exp 2 := by sorry

end

end NUMINAMATH_CALUDE_a1_value_l1362_136278


namespace NUMINAMATH_CALUDE_complex_cube_equation_l1362_136299

def complex (x y : ℤ) := x + y * Complex.I

theorem complex_cube_equation (x y d : ℤ) (hx : x > 0) (hy : y > 0) :
  (complex x y)^3 = complex (-26) d → complex x y = complex 1 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_cube_equation_l1362_136299


namespace NUMINAMATH_CALUDE_plane_intersection_properties_l1362_136242

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

end NUMINAMATH_CALUDE_plane_intersection_properties_l1362_136242


namespace NUMINAMATH_CALUDE_smallest_k_for_64_power_gt_4_16_l1362_136270

theorem smallest_k_for_64_power_gt_4_16 : ∃ k : ℕ, k = 6 ∧ 64^k > 4^16 ∧ ∀ m : ℕ, m < k → 64^m ≤ 4^16 := by
  sorry

end NUMINAMATH_CALUDE_smallest_k_for_64_power_gt_4_16_l1362_136270


namespace NUMINAMATH_CALUDE_dartboard_angle_l1362_136276

theorem dartboard_angle (p : ℝ) (θ : ℝ) : 
  p = 1 / 8 → θ = p * 360 → θ = 45 :=
by sorry

end NUMINAMATH_CALUDE_dartboard_angle_l1362_136276


namespace NUMINAMATH_CALUDE_correct_selling_price_l1362_136236

/-- The markup percentage applied to the cost price -/
def markup : ℚ := 25 / 100

/-- The cost price of the computer table in rupees -/
def cost_price : ℕ := 6672

/-- The selling price of the computer table in rupees -/
def selling_price : ℕ := 8340

/-- Theorem stating that the selling price is correct given the cost price and markup -/
theorem correct_selling_price : 
  (cost_price : ℚ) * (1 + markup) = selling_price := by sorry

end NUMINAMATH_CALUDE_correct_selling_price_l1362_136236


namespace NUMINAMATH_CALUDE_seating_arrangements_count_l1362_136255

/-- The number of ways to seat n people in a row -/
def totalArrangements (n : ℕ) : ℕ := n.factorial

/-- The number of ways to seat n people in a row where m specific people sit together -/
def arrangementsWithGroupTogether (n m : ℕ) : ℕ := 
  (n - m + 1).factorial * m.factorial

/-- The number of ways to seat 10 people in a row where 4 specific people cannot sit in 4 consecutive seats -/
def seatingArrangements : ℕ := 
  totalArrangements 10 - arrangementsWithGroupTogether 10 4

theorem seating_arrangements_count : seatingArrangements = 3507840 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangements_count_l1362_136255


namespace NUMINAMATH_CALUDE_office_to_bedroom_ratio_l1362_136272

/-- Represents the energy consumption of lights in watts per hour -/
structure LightEnergy where
  bedroom : ℝ
  office : ℝ
  livingRoom : ℝ

/-- Calculates the total energy used over a given number of hours -/
def totalEnergyUsed (l : LightEnergy) (hours : ℝ) : ℝ :=
  (l.bedroom + l.office + l.livingRoom) * hours

/-- Theorem stating the ratio of office light energy to bedroom light energy -/
theorem office_to_bedroom_ratio (l : LightEnergy) :
  l.bedroom = 6 →
  l.livingRoom = 4 * l.bedroom →
  totalEnergyUsed l 2 = 96 →
  l.office / l.bedroom = 3 := by
sorry

end NUMINAMATH_CALUDE_office_to_bedroom_ratio_l1362_136272


namespace NUMINAMATH_CALUDE_power_72_in_terms_of_m_and_n_l1362_136238

theorem power_72_in_terms_of_m_and_n (a m n : ℝ) 
  (h1 : 2^a = m) (h2 : 3^a = n) : 72^a = m^3 * n^2 := by
  sorry

end NUMINAMATH_CALUDE_power_72_in_terms_of_m_and_n_l1362_136238


namespace NUMINAMATH_CALUDE_sphere_surface_area_given_cone_l1362_136202

/-- Given a cone and a sphere with equal volumes, where the radius of the base of the cone
    is twice the radius of the sphere, and the height of the cone is 1,
    prove that the surface area of the sphere is 4π. -/
theorem sphere_surface_area_given_cone (r : ℝ) :
  (4 / 3 * π * r^3 = 1 / 3 * π * (2*r)^2 * 1) →
  4 * π * r^2 = 4 * π := by
  sorry

#check sphere_surface_area_given_cone

end NUMINAMATH_CALUDE_sphere_surface_area_given_cone_l1362_136202


namespace NUMINAMATH_CALUDE_function_inequality_implies_a_bound_l1362_136215

theorem function_inequality_implies_a_bound (a : ℝ) : 
  (∀ x₁ ∈ Set.Icc 0 1, ∃ x₂ ∈ Set.Icc 1 2, 
    (x₁ - 1 / (x₁ + 1)) ≥ (x₂^2 - 2*a*x₂ + 4)) → 
  a ≥ 9/4 := by
sorry

end NUMINAMATH_CALUDE_function_inequality_implies_a_bound_l1362_136215


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_coinciding_foci_l1362_136263

/-- Given an ellipse and a hyperbola with coinciding foci, prove that b² of the ellipse equals 14.76 -/
theorem ellipse_hyperbola_coinciding_foci (b : ℝ) : 
  (∀ x y : ℝ, x^2/25 + y^2/b^2 = 1 → x^2/100 - y^2/64 = 1/16 → 
   ∃ c : ℝ, c^2 = 25 - b^2 ∧ c^2 = 10.25) → 
  b^2 = 14.76 := by
sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_coinciding_foci_l1362_136263


namespace NUMINAMATH_CALUDE_two_digit_number_problem_l1362_136265

/-- 
Given a two-digit number n = 10a + b, where a and b are single digits,
if 1000a + 100b = 37(100a + 10b + 1), then n = 27.
-/
theorem two_digit_number_problem (a b : ℕ) (h1 : a < 10) (h2 : b < 10) 
  (h3 : 1000 * a + 100 * b = 37 * (100 * a + 10 * b + 1)) :
  10 * a + b = 27 := by
  sorry

#check two_digit_number_problem

end NUMINAMATH_CALUDE_two_digit_number_problem_l1362_136265


namespace NUMINAMATH_CALUDE_sum_of_squared_distances_l1362_136249

/-- Two perpendicular lines in a Cartesian coordinate system -/
structure PerpendicularLines where
  a : ℝ
  l : Set (ℝ × ℝ) := {(x, y) | a * x + y - 1 = 0}
  m : Set (ℝ × ℝ) := {(x, y) | x - a * y + 3 = 0}
  P : ℝ × ℝ := (0, 1)
  Q : ℝ × ℝ := (-3, 0)
  M : ℝ × ℝ
  h_M_in_l : M ∈ l
  h_M_in_m : M ∈ m

/-- The sum of squared distances from M to P and Q is 10 -/
theorem sum_of_squared_distances (lines : PerpendicularLines) :
  (lines.M.1 - lines.P.1)^2 + (lines.M.2 - lines.P.2)^2 +
  (lines.M.1 - lines.Q.1)^2 + (lines.M.2 - lines.Q.2)^2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squared_distances_l1362_136249


namespace NUMINAMATH_CALUDE_pizza_toppings_combinations_l1362_136247

theorem pizza_toppings_combinations : Nat.choose 7 3 = 35 := by
  sorry

end NUMINAMATH_CALUDE_pizza_toppings_combinations_l1362_136247


namespace NUMINAMATH_CALUDE_pool_filling_time_l1362_136254

theorem pool_filling_time (a b c d : ℝ) 
  (h1 : a + b = 1/2)
  (h2 : b + c = 1/3)
  (h3 : c + d = 1/4) :
  a + d = 5/12 :=
sorry

end NUMINAMATH_CALUDE_pool_filling_time_l1362_136254


namespace NUMINAMATH_CALUDE_expression_evaluation_l1362_136286

theorem expression_evaluation : (-4)^6 / 4^4 + 2^5 * 5 - 7^2 = 127 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1362_136286


namespace NUMINAMATH_CALUDE_arithmetic_seq_common_diff_l1362_136239

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

end NUMINAMATH_CALUDE_arithmetic_seq_common_diff_l1362_136239


namespace NUMINAMATH_CALUDE_junior_toy_ratio_l1362_136226

theorem junior_toy_ratio :
  let num_rabbits : ℕ := 16
  let monday_toys : ℕ := 6
  let friday_toys : ℕ := 4 * monday_toys
  let wednesday_toys : ℕ := wednesday_toys -- Unknown variable
  let saturday_toys : ℕ := wednesday_toys / 2
  let toys_per_rabbit : ℕ := 3
  
  num_rabbits * toys_per_rabbit = monday_toys + wednesday_toys + friday_toys + saturday_toys →
  wednesday_toys = 2 * monday_toys :=
by sorry

end NUMINAMATH_CALUDE_junior_toy_ratio_l1362_136226


namespace NUMINAMATH_CALUDE_team_games_count_l1362_136218

/-- Proves that a team with the given win percentages played 175 games in total -/
theorem team_games_count (first_hundred_win_rate : Real) 
                          (remaining_win_rate : Real)
                          (total_win_rate : Real)
                          (h1 : first_hundred_win_rate = 0.85)
                          (h2 : remaining_win_rate = 0.5)
                          (h3 : total_win_rate = 0.7) : 
  ∃ (total_games : ℕ), total_games = 175 ∧ 
    (first_hundred_win_rate * 100 + remaining_win_rate * (total_games - 100)) / total_games = total_win_rate :=
by
  sorry


end NUMINAMATH_CALUDE_team_games_count_l1362_136218


namespace NUMINAMATH_CALUDE_square_root_of_product_plus_one_l1362_136225

theorem square_root_of_product_plus_one (a : ℕ) (h : a = 25) : 
  Real.sqrt (a * (a + 1) * (a + 2) * (a + 3) + 1) = a^2 + 3*a + 1 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_product_plus_one_l1362_136225


namespace NUMINAMATH_CALUDE_game_solvable_l1362_136208

/-- The game state, representing the positions of the red and blue beads -/
structure GameState where
  red : ℚ
  blue : ℚ

/-- The possible moves in the game -/
inductive Move
  | Red (k : ℤ)
  | Blue (k : ℤ)

/-- Apply a move to the game state -/
def applyMove (r : ℚ) (state : GameState) (move : Move) : GameState :=
  match move with
  | Move.Red k => 
      { red := state.blue + r^k * (state.red - state.blue),
        blue := state.blue }
  | Move.Blue k => 
      { red := state.red,
        blue := state.red + r^k * (state.blue - state.red) }

/-- A sequence of moves -/
def MoveSequence := List Move

/-- Apply a sequence of moves to the initial game state -/
def applyMoveSequence (r : ℚ) (moves : MoveSequence) : GameState :=
  moves.foldl (applyMove r) { red := 0, blue := 1 }

/-- The main theorem -/
theorem game_solvable (r : ℚ) : 
  (∃ (moves : MoveSequence), moves.length ≤ 2021 ∧ (applyMoveSequence r moves).red = 1) ↔ 
  (∃ (m : ℕ), m ≥ 1 ∧ m ≤ 1010 ∧ r = (m + 1) / m) :=
sorry

end NUMINAMATH_CALUDE_game_solvable_l1362_136208


namespace NUMINAMATH_CALUDE_jordan_rectangle_width_l1362_136274

/-- Given two rectangles with equal area, where one rectangle measures 8 inches by 15 inches
    and the other has a length of 4 inches, prove that the width of the second rectangle is 30 inches. -/
theorem jordan_rectangle_width (area carol_length carol_width jordan_length jordan_width : ℝ) :
  area = carol_length * carol_width →
  area = jordan_length * jordan_width →
  carol_length = 8 →
  carol_width = 15 →
  jordan_length = 4 →
  jordan_width = 30 := by
  sorry

end NUMINAMATH_CALUDE_jordan_rectangle_width_l1362_136274


namespace NUMINAMATH_CALUDE_cube_face_sum_l1362_136201

theorem cube_face_sum (a b c d e f : ℕ+) : 
  a * b * c + a * e * c + a * b * f + a * e * f + 
  d * b * c + d * e * c + d * b * f + d * e * f = 1001 →
  a + b + c + d + e + f = 31 := by
sorry

end NUMINAMATH_CALUDE_cube_face_sum_l1362_136201


namespace NUMINAMATH_CALUDE_marble_fraction_l1362_136244

theorem marble_fraction (x : ℚ) : 
  let initial_blue := (2 : ℚ) / 3 * x
  let initial_red := x - initial_blue
  let new_red := 2 * initial_red
  let new_blue := (3 : ℚ) / 2 * initial_blue
  let total_new := new_red + new_blue
  new_red / total_new = (2 : ℚ) / 5 := by sorry

end NUMINAMATH_CALUDE_marble_fraction_l1362_136244


namespace NUMINAMATH_CALUDE_shorts_cost_calculation_l1362_136277

def jacket_cost : ℝ := 14.82
def shirt_cost : ℝ := 12.51
def total_cost : ℝ := 42.33

theorem shorts_cost_calculation : 
  total_cost - jacket_cost - shirt_cost = 15 :=
by sorry

end NUMINAMATH_CALUDE_shorts_cost_calculation_l1362_136277


namespace NUMINAMATH_CALUDE_a_power_b_equals_sixteen_l1362_136214

theorem a_power_b_equals_sixteen (a b : ℝ) : (a - 4)^2 + |2 - b| = 0 → a^b = 16 := by
  sorry

end NUMINAMATH_CALUDE_a_power_b_equals_sixteen_l1362_136214


namespace NUMINAMATH_CALUDE_circle_symmetry_line_l1362_136216

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y + 1 = 0

-- Define the line equation
def line_eq (a x y : ℝ) : Prop := a*x + y + 1 = 0

-- Define symmetry condition
def is_symmetric (a : ℝ) : Prop :=
  ∃ (x y : ℝ), circle_eq x y ∧ line_eq a (-1) 2

-- Theorem statement
theorem circle_symmetry_line (a : ℝ) :
  is_symmetric a → a = 3 := by sorry

end NUMINAMATH_CALUDE_circle_symmetry_line_l1362_136216


namespace NUMINAMATH_CALUDE_bruce_calculators_l1362_136205

-- Define the given conditions
def total_money : ℕ := 200
def crayon_cost : ℕ := 5
def book_cost : ℕ := 5
def calculator_cost : ℕ := 5
def bag_cost : ℕ := 10
def crayon_packs : ℕ := 5
def books : ℕ := 10
def bags : ℕ := 11

-- Define the theorem
theorem bruce_calculators :
  let crayon_total := crayon_cost * crayon_packs
  let book_total := book_cost * books
  let remaining_after_books := total_money - (crayon_total + book_total)
  let bag_total := bag_cost * bags
  let remaining_for_calculators := remaining_after_books - bag_total
  remaining_for_calculators / calculator_cost = 3 := by
  sorry

end NUMINAMATH_CALUDE_bruce_calculators_l1362_136205


namespace NUMINAMATH_CALUDE_initial_patio_rows_l1362_136220

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

end NUMINAMATH_CALUDE_initial_patio_rows_l1362_136220


namespace NUMINAMATH_CALUDE_projection_theorem_l1362_136234

/-- Given two 2D vectors a and b, and a third vector c that satisfies a + c = 0,
    prove that the projection of c onto b is -√65/5 -/
theorem projection_theorem (a b c : ℝ × ℝ) : 
  a = (2, 3) → 
  b = (-4, 7) → 
  a + c = (0, 0) → 
  let proj_c_onto_b := (c.1 * b.1 + c.2 * b.2) / Real.sqrt (b.1^2 + b.2^2)
  proj_c_onto_b = -Real.sqrt 65 / 5 := by
  sorry

end NUMINAMATH_CALUDE_projection_theorem_l1362_136234


namespace NUMINAMATH_CALUDE_gcf_lcm_sum_8_12_l1362_136200

theorem gcf_lcm_sum_8_12 : Nat.gcd 8 12 + Nat.lcm 8 12 = 28 := by
  sorry

end NUMINAMATH_CALUDE_gcf_lcm_sum_8_12_l1362_136200


namespace NUMINAMATH_CALUDE_inverse_of_B_cubed_l1362_136253

open Matrix

/-- Given a 2x2 matrix B with its inverse, prove that the inverse of B^3 is equal to B^(-1) -/
theorem inverse_of_B_cubed (B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : B⁻¹ = ![![3, 4], ![-2, -3]]) : 
  (B^3)⁻¹ = ![![3, 4], ![-2, -3]] := by
  sorry

end NUMINAMATH_CALUDE_inverse_of_B_cubed_l1362_136253


namespace NUMINAMATH_CALUDE_benjie_current_age_l1362_136297

/-- Benjie's age in years -/
def benjie_age : ℕ := 6

/-- Margo's age in years -/
def margo_age : ℕ := 1

/-- The age difference between Benjie and Margo in years -/
def age_difference : ℕ := 5

/-- The number of years until Margo is 4 years old -/
def years_until_margo_4 : ℕ := 3

theorem benjie_current_age :
  (benjie_age = margo_age + age_difference) ∧
  (margo_age + years_until_margo_4 = 4) →
  benjie_age = 6 := by sorry

end NUMINAMATH_CALUDE_benjie_current_age_l1362_136297


namespace NUMINAMATH_CALUDE_cubic_root_equation_solution_l1362_136232

theorem cubic_root_equation_solution :
  ∀ x : ℝ, (((30 * x + (30 * x + 18) ^ (1/3)) ^ (1/3)) = 18) → x = 2907/15 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_equation_solution_l1362_136232


namespace NUMINAMATH_CALUDE_journey_speed_calculation_l1362_136261

/-- Represents the speed and distance of a journey between three towns -/
structure JourneyData where
  speed_qb : ℝ  -- Speed from Q to B
  speed_bc : ℝ  -- Speed from B to C
  dist_bc : ℝ   -- Distance from B to C
  avg_speed : ℝ -- Average speed of the whole journey

/-- Theorem stating the conditions and the result to be proved -/
theorem journey_speed_calculation (j : JourneyData) 
  (h1 : j.speed_qb = 60)
  (h2 : j.avg_speed = 36)
  (h3 : j.dist_bc > 0) :
  j.speed_qb = 60 ∧ 
  j.avg_speed = 36 ∧ 
  (3 * j.dist_bc) / (2 * j.dist_bc / j.speed_qb + j.dist_bc / j.speed_bc) = j.avg_speed →
  j.speed_bc = 20 := by
  sorry

end NUMINAMATH_CALUDE_journey_speed_calculation_l1362_136261


namespace NUMINAMATH_CALUDE_cube_opposite_face_l1362_136206

structure Cube where
  faces : Finset Char
  adjacent : Char → Char → Prop

def opposite (c : Cube) (f1 f2 : Char) : Prop :=
  f1 ∈ c.faces ∧ f2 ∈ c.faces ∧ ¬c.adjacent f1 f2 ∧
  ∀ f3 ∈ c.faces, f3 ≠ f1 ∧ f3 ≠ f2 → (c.adjacent f1 f3 ↔ ¬c.adjacent f2 f3)

theorem cube_opposite_face (c : Cube) :
  c.faces = {'x', 'A', 'B', 'C', 'D', 'E', 'F'} →
  c.adjacent 'x' 'A' →
  c.adjacent 'x' 'D' →
  c.adjacent 'x' 'F' →
  c.adjacent 'E' 'D' →
  ¬c.adjacent 'x' 'E' →
  opposite c 'x' 'B' := by
  sorry

end NUMINAMATH_CALUDE_cube_opposite_face_l1362_136206


namespace NUMINAMATH_CALUDE_shekar_science_score_l1362_136251

/-- Represents a student's scores across 5 subjects -/
structure StudentScores where
  mathematics : ℕ
  science : ℕ
  social_studies : ℕ
  english : ℕ
  biology : ℕ

/-- Calculates the average score -/
def average (s : StudentScores) : ℚ :=
  (s.mathematics + s.science + s.social_studies + s.english + s.biology) / 5

theorem shekar_science_score :
  ∀ (s : StudentScores),
    s.mathematics = 76 →
    s.social_studies = 82 →
    s.english = 67 →
    s.biology = 55 →
    average s = 69 →
    s.science = 65 := by
  sorry

#check shekar_science_score

end NUMINAMATH_CALUDE_shekar_science_score_l1362_136251


namespace NUMINAMATH_CALUDE_vector_magnitude_problem_l1362_136298

/-- Given two plane vectors satisfying certain conditions, prove that the magnitude of their linear combination is √13. -/
theorem vector_magnitude_problem (a b : ℝ × ℝ) : 
  ‖a‖ = Real.sqrt 2 →
  b = (1, 0) →
  a • (a - 2 • b) = 0 →
  ‖2 • a + b‖ = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_problem_l1362_136298


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1362_136258

/-- Given a hyperbola and a parabola with specific properties, prove the equation of the hyperbola -/
theorem hyperbola_equation (a b : ℝ) (P : ℝ × ℝ) :
  a > 0 → b > 0 →
  (∃ F : ℝ × ℝ, F = (2, 0) ∧ 
    (∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1 ↔ (x - F.1)^2/a^2 - (y - F.2)^2/b^2 = 1) ∧
    (P.2^2 = 8*P.1) ∧
    ((P.1 - F.1)^2 + (P.2 - F.2)^2 = 25)) →
  (∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1 ↔ x^2 - y^2/3 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1362_136258


namespace NUMINAMATH_CALUDE_perfect_cubes_between_powers_of_three_l1362_136231

theorem perfect_cubes_between_powers_of_three : 
  (Finset.filter (fun n : ℕ => 
    3^5 - 1 ≤ n^3 ∧ n^3 ≤ 3^15 + 1) 
    (Finset.range (Nat.floor (Real.rpow 3 5) + 1))).card = 18 := by
  sorry

end NUMINAMATH_CALUDE_perfect_cubes_between_powers_of_three_l1362_136231


namespace NUMINAMATH_CALUDE_sequence_formula_l1362_136245

/-- A sequence where S_n = n^2 * a_n for n ≥ 2, and a_1 = 1 -/
def sequence_a (n : ℕ) : ℚ := sorry

/-- Sum of the first n terms of the sequence -/
def S (n : ℕ) : ℚ := sorry

theorem sequence_formula :
  ∀ n : ℕ, n ≥ 1 →
  (∀ k : ℕ, k ≥ 2 → S k = k^2 * sequence_a k) →
  sequence_a 1 = 1 →
  sequence_a n = 1 / (n + 1) :=
sorry

end NUMINAMATH_CALUDE_sequence_formula_l1362_136245


namespace NUMINAMATH_CALUDE_stream_speed_l1362_136296

/-- Given a canoe that rows upstream at 8 km/hr and downstream at 12 km/hr,
    prove that the speed of the stream is 2 km/hr. -/
theorem stream_speed (upstream_speed downstream_speed : ℝ)
  (h_upstream : upstream_speed = 8)
  (h_downstream : downstream_speed = 12) :
  ∃ (canoe_speed stream_speed : ℝ),
    canoe_speed - stream_speed = upstream_speed ∧
    canoe_speed + stream_speed = downstream_speed ∧
    stream_speed = 2 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_l1362_136296


namespace NUMINAMATH_CALUDE_luncheon_tables_l1362_136262

theorem luncheon_tables (invited : ℕ) (no_show : ℕ) (per_table : ℕ) 
  (h1 : invited = 18) 
  (h2 : no_show = 12) 
  (h3 : per_table = 3) : 
  (invited - no_show) / per_table = 2 := by
  sorry

end NUMINAMATH_CALUDE_luncheon_tables_l1362_136262


namespace NUMINAMATH_CALUDE_algebraic_notation_correctness_l1362_136241

/-- Rules for algebraic notation --/
structure AlgebraicNotationRules where
  no_multiplication_sign : Bool
  number_before_variable : Bool
  proper_fraction : Bool
  correct_negative_placement : Bool

/-- Check if an expression follows algebraic notation rules --/
def follows_algebraic_notation (expr : String) (rules : AlgebraicNotationRules) : Bool :=
  rules.no_multiplication_sign ∧ 
  rules.number_before_variable ∧ 
  rules.proper_fraction ∧ 
  rules.correct_negative_placement

/-- Given expressions --/
def expr_A : String := "a×5"
def expr_B : String := "a7"
def expr_C : String := "3½x"
def expr_D : String := "-⅞x"

theorem algebraic_notation_correctness :
  follows_algebraic_notation expr_D 
    {no_multiplication_sign := true, 
     number_before_variable := true, 
     proper_fraction := true, 
     correct_negative_placement := true} ∧
  ¬follows_algebraic_notation expr_A 
    {no_multiplication_sign := false, 
     number_before_variable := false, 
     proper_fraction := true, 
     correct_negative_placement := true} ∧
  ¬follows_algebraic_notation expr_B
    {no_multiplication_sign := true, 
     number_before_variable := false, 
     proper_fraction := true, 
     correct_negative_placement := true} ∧
  ¬follows_algebraic_notation expr_C
    {no_multiplication_sign := true, 
     number_before_variable := true, 
     proper_fraction := false, 
     correct_negative_placement := true} :=
by sorry

end NUMINAMATH_CALUDE_algebraic_notation_correctness_l1362_136241


namespace NUMINAMATH_CALUDE_final_crayon_count_l1362_136295

/-- Represents the number of crayons in a drawer after a series of actions. -/
def crayons_in_drawer (initial : ℕ) (mary_takes : ℕ) (mark_takes : ℕ) (mary_returns : ℕ) (sarah_adds : ℕ) (john_takes : ℕ) : ℕ :=
  initial - mary_takes - mark_takes + mary_returns + sarah_adds - john_takes

/-- Theorem stating that given the initial number of crayons and the actions performed, 
    the final number of crayons in the drawer is 4. -/
theorem final_crayon_count :
  crayons_in_drawer 7 3 2 1 5 4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_final_crayon_count_l1362_136295


namespace NUMINAMATH_CALUDE_sum_of_edges_is_120_l1362_136284

/-- A rectangular solid with specific properties -/
structure RectangularSolid where
  -- The three dimensions of the solid
  a : ℝ
  b : ℝ
  c : ℝ
  -- Volume is 1000 cm³
  volume_eq : a * b * c = 1000
  -- Surface area is 600 cm²
  surface_area_eq : 2 * (a * b + b * c + a * c) = 600
  -- Dimensions are in geometric progression
  geometric_progression : ∃ (r : ℝ), b = a * r ∧ c = b * r

/-- The sum of all edge lengths of a rectangular solid -/
def sum_of_edges (solid : RectangularSolid) : ℝ :=
  4 * (solid.a + solid.b + solid.c)

/-- Theorem stating that the sum of all edge lengths is 120 cm -/
theorem sum_of_edges_is_120 (solid : RectangularSolid) :
  sum_of_edges solid = 120 := by
  sorry

#check sum_of_edges_is_120

end NUMINAMATH_CALUDE_sum_of_edges_is_120_l1362_136284


namespace NUMINAMATH_CALUDE_point_movement_to_x_axis_l1362_136230

/-- Given a point P with coordinates (m+2, 2m+4) that is moved 2 units up to point Q which lies on the x-axis, prove that the coordinates of Q are (-1, 0) -/
theorem point_movement_to_x_axis (m : ℝ) :
  let P : ℝ × ℝ := (m + 2, 2*m + 4)
  let Q : ℝ × ℝ := (P.1, P.2 + 2)
  Q.2 = 0 → Q = (-1, 0) := by sorry

end NUMINAMATH_CALUDE_point_movement_to_x_axis_l1362_136230


namespace NUMINAMATH_CALUDE_equation_solution_l1362_136221

theorem equation_solution : ∃ n : ℕ, 3^n * 9^n = 81^(n-12) ∧ n = 48 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1362_136221


namespace NUMINAMATH_CALUDE_blue_pens_count_l1362_136246

/-- Given a total number of pens and the difference between blue and red pens,
    calculate the number of blue pens. -/
def number_of_blue_pens (total : ℕ) (difference : ℕ) : ℕ :=
  (total + difference) / 2

theorem blue_pens_count :
  number_of_blue_pens 82 6 = 44 := by
  sorry

end NUMINAMATH_CALUDE_blue_pens_count_l1362_136246


namespace NUMINAMATH_CALUDE_smallest_of_four_consecutive_even_numbers_l1362_136282

theorem smallest_of_four_consecutive_even_numbers (x : ℤ) : 
  (∃ y z w : ℤ, y = x + 2 ∧ z = x + 4 ∧ w = x + 6 ∧ 
   x % 2 = 0 ∧ x + y + z + w = 140) → x = 32 := by
sorry

end NUMINAMATH_CALUDE_smallest_of_four_consecutive_even_numbers_l1362_136282


namespace NUMINAMATH_CALUDE_complex_magnitude_l1362_136229

theorem complex_magnitude (b : ℝ) : 
  let z : ℂ := (3 - b * Complex.I) / Complex.I
  (z.re = z.im) → Complex.abs z = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l1362_136229


namespace NUMINAMATH_CALUDE_f_positive_before_root_l1362_136280

noncomputable def f (x : ℝ) : ℝ := (1/3)^x - Real.log x / Real.log 2

theorem f_positive_before_root (x₀ a : ℝ) 
  (h_root : f x₀ = 0)
  (h_decreasing : ∀ x y, x < y → f x > f y)
  (h_a_pos : 0 < a)
  (h_a_lt_x₀ : a < x₀) : 
  f a > 0 := by
  sorry

end NUMINAMATH_CALUDE_f_positive_before_root_l1362_136280


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1362_136264

theorem quadratic_inequality_range (a : ℝ) :
  (∀ x : ℝ, a * x^2 + a * x - 1 ≤ 0) → -4 ≤ a ∧ a ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1362_136264


namespace NUMINAMATH_CALUDE_largest_four_digit_product_l1362_136266

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem largest_four_digit_product (m x y z : ℕ) : 
  m > 0 →
  m = x * y * (10 * x + y) * z →
  is_prime x →
  is_prime y →
  is_prime (10 * x + y) →
  is_prime z →
  x < 20 →
  y < 20 →
  z < 20 →
  x ≠ y →
  x ≠ 10 * x + y →
  y ≠ 10 * x + y →
  x ≠ z →
  y ≠ z →
  (10 * x + y) ≠ z →
  1000 ≤ m →
  m < 10000 →
  m ≤ 7478 :=
sorry

end NUMINAMATH_CALUDE_largest_four_digit_product_l1362_136266


namespace NUMINAMATH_CALUDE_square_sum_from_means_l1362_136275

theorem square_sum_from_means (x y : ℝ) 
  (h_arithmetic : (x + y) / 2 = 20) 
  (h_geometric : Real.sqrt (x * y) = Real.sqrt 100) : 
  x^2 + y^2 = 1400 := by
sorry

end NUMINAMATH_CALUDE_square_sum_from_means_l1362_136275


namespace NUMINAMATH_CALUDE_x_eq_one_sufficient_not_necessary_for_x_squared_eq_one_l1362_136212

theorem x_eq_one_sufficient_not_necessary_for_x_squared_eq_one :
  (∀ x : ℝ, x = 1 → x^2 = 1) ∧
  ¬(∀ x : ℝ, x^2 = 1 → x = 1) :=
by sorry

end NUMINAMATH_CALUDE_x_eq_one_sufficient_not_necessary_for_x_squared_eq_one_l1362_136212


namespace NUMINAMATH_CALUDE_chessboard_tromino_coverage_l1362_136285

/-- Represents a chessboard with alternating colors and black corners -/
structure Chessboard (n : ℕ) :=
  (is_odd : n % 2 = 1)
  (ge_seven : n ≥ 7)

/-- Calculates the number of black squares on the chessboard -/
def black_squares (board : Chessboard n) : ℕ :=
  (n^2 + 1) / 2

/-- Calculates the minimum number of trominos needed -/
def min_trominos (board : Chessboard n) : ℕ :=
  (n^2 + 1) / 6

theorem chessboard_tromino_coverage (n : ℕ) (board : Chessboard n) :
  (black_squares board) % 3 = 0 ∧
  min_trominos board = (n^2 + 1) / 6 :=
sorry

end NUMINAMATH_CALUDE_chessboard_tromino_coverage_l1362_136285


namespace NUMINAMATH_CALUDE_average_price_reduction_l1362_136289

theorem average_price_reduction (original_price final_price : ℝ) 
  (h1 : original_price = 60) 
  (h2 : final_price = 48.6) : 
  ∃ (x : ℝ), x = 0.1 ∧ original_price * (1 - x)^2 = final_price :=
sorry

end NUMINAMATH_CALUDE_average_price_reduction_l1362_136289


namespace NUMINAMATH_CALUDE_inequality_analysis_l1362_136210

theorem inequality_analysis (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hxy : x > y) (hz : z ≠ 0) :
  (∀ z, x + z > y + z) ∧
  (∀ z, x - z > y - z) ∧
  (∃ z, ¬(x * z > y * z)) ∧
  (∀ z, x / z^2 > y / z^2) ∧
  (∀ z, x * z^2 > y * z^2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_analysis_l1362_136210


namespace NUMINAMATH_CALUDE_equation_solution_l1362_136292

theorem equation_solution :
  ∃! (a b c d : ℚ), 
    a^2 + b^2 + c^2 + d^2 - a*b - b*c - c*d - d + 2/5 = 0 ∧
    a = 1/5 ∧ b = 2/5 ∧ c = 3/5 ∧ d = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1362_136292


namespace NUMINAMATH_CALUDE_green_balls_count_l1362_136291

theorem green_balls_count (total : ℕ) (white yellow red purple : ℕ) (prob_not_red_purple : ℚ) :
  total = 100 ∧
  white = 50 ∧
  yellow = 10 ∧
  red = 7 ∧
  purple = 3 ∧
  prob_not_red_purple = 9/10 →
  ∃ green : ℕ, green = 30 ∧ total = white + green + yellow + red + purple :=
by sorry

end NUMINAMATH_CALUDE_green_balls_count_l1362_136291


namespace NUMINAMATH_CALUDE_yellow_ball_probability_l1362_136213

/-- The probability of drawing a yellow ball from a bag with yellow, red, and white balls -/
theorem yellow_ball_probability (yellow red white : ℕ) : 
  yellow = 5 → red = 8 → white = 7 → 
  (yellow : ℚ) / (yellow + red + white) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_yellow_ball_probability_l1362_136213


namespace NUMINAMATH_CALUDE_gcd_10010_15015_l1362_136287

theorem gcd_10010_15015 : Nat.gcd 10010 15015 = 5005 := by
  sorry

end NUMINAMATH_CALUDE_gcd_10010_15015_l1362_136287


namespace NUMINAMATH_CALUDE_job_applicants_theorem_l1362_136268

theorem job_applicants_theorem (total : ℕ) (experienced : ℕ) (degreed : ℕ) (both : ℕ) :
  total = 30 →
  experienced = 10 →
  degreed = 18 →
  both = 9 →
  total - (experienced + degreed - both) = 11 :=
by sorry

end NUMINAMATH_CALUDE_job_applicants_theorem_l1362_136268


namespace NUMINAMATH_CALUDE_right_triangle_and_inverse_mod_l1362_136222

theorem right_triangle_and_inverse_mod : 
  (60^2 + 144^2 = 156^2) ∧ 
  (∃ n : ℕ, n < 3751 ∧ (300 * n) % 3751 = 1) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_and_inverse_mod_l1362_136222


namespace NUMINAMATH_CALUDE_train_length_l1362_136240

/-- Given a train traveling at 45 km/hr, crossing a bridge of 240.03 meters in 30 seconds,
    the length of the train is 134.97 meters. -/
theorem train_length (train_speed : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) :
  train_speed = 45 →
  bridge_length = 240.03 →
  crossing_time = 30 →
  (train_speed * 1000 / 3600 * crossing_time) - bridge_length = 134.97 := by
  sorry

#eval (45 * 1000 / 3600 * 30) - 240.03

end NUMINAMATH_CALUDE_train_length_l1362_136240


namespace NUMINAMATH_CALUDE_simplify_expression_l1362_136281

theorem simplify_expression (x : ℝ) : (x + 15) + (150 * x + 20) = 151 * x + 35 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1362_136281


namespace NUMINAMATH_CALUDE_other_number_proof_l1362_136294

theorem other_number_proof (a b : ℕ+) (h1 : Nat.gcd a b = 108) (h2 : Nat.lcm a b = 27720) (h3 : a = 216) : b = 64 := by
  sorry

end NUMINAMATH_CALUDE_other_number_proof_l1362_136294


namespace NUMINAMATH_CALUDE_ratio_problem_l1362_136269

theorem ratio_problem (a b x m : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  a / b = 4 / 5 → x = a * (1 + 1/4) → m = b * (1 - 4/5) → m / x = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l1362_136269


namespace NUMINAMATH_CALUDE_probability_sum_less_than_4_l1362_136257

/-- A square in the 2D plane --/
structure Square where
  bottomLeft : ℝ × ℝ
  topRight : ℝ × ℝ

/-- The probability that a point satisfies a condition within a given square --/
def probability (s : Square) (condition : ℝ × ℝ → Prop) : ℝ :=
  sorry

/-- The condition x + y < 4 --/
def sumLessThan4 (p : ℝ × ℝ) : Prop :=
  p.1 + p.2 < 4

theorem probability_sum_less_than_4 :
  let s : Square := { bottomLeft := (0, 0), topRight := (3, 3) }
  probability s sumLessThan4 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_probability_sum_less_than_4_l1362_136257


namespace NUMINAMATH_CALUDE_gcd_90_270_l1362_136259

theorem gcd_90_270 : Nat.gcd 90 270 = 90 := by
  sorry

end NUMINAMATH_CALUDE_gcd_90_270_l1362_136259


namespace NUMINAMATH_CALUDE_nested_radical_value_l1362_136248

theorem nested_radical_value : 
  ∃ x : ℝ, x = Real.sqrt (3 - x) → x = (-1 + Real.sqrt 13) / 2 := by
  sorry

end NUMINAMATH_CALUDE_nested_radical_value_l1362_136248


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l1362_136260

/-- Represents a repeating decimal with a single digit repeating. -/
def RepeatingDecimal (n : ℕ) : ℚ :=
  n / 9

/-- The sum of 0.666... + 0.222... - 0.444... equals 4/9 -/
theorem repeating_decimal_sum : 
  RepeatingDecimal 6 + RepeatingDecimal 2 - RepeatingDecimal 4 = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l1362_136260


namespace NUMINAMATH_CALUDE_card_value_proof_l1362_136233

/-- Given four cards W, X, Y, Z with certain conditions, prove Y is tagged with 300 --/
theorem card_value_proof (W X Y Z : ℕ) : 
  W = 200 →
  X = W / 2 →
  Z = 400 →
  W + X + Y + Z = 1000 →
  Y = 300 := by
  sorry

end NUMINAMATH_CALUDE_card_value_proof_l1362_136233


namespace NUMINAMATH_CALUDE_regression_coefficient_nonzero_l1362_136288

/-- Represents a regression line for two variables with a linear relationship -/
structure RegressionLine where
  a : ℝ
  b : ℝ

/-- Theorem: The regression coefficient b in a regression line y = a + bx 
    for two variables with a linear relationship cannot be equal to 0 -/
theorem regression_coefficient_nonzero (line : RegressionLine) : 
  line.b ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_regression_coefficient_nonzero_l1362_136288


namespace NUMINAMATH_CALUDE_negative_square_of_two_l1362_136235

theorem negative_square_of_two : -2^2 = -4 := by
  sorry

end NUMINAMATH_CALUDE_negative_square_of_two_l1362_136235


namespace NUMINAMATH_CALUDE_square_circle_relation_l1362_136217

theorem square_circle_relation (s r : ℝ) (h : s > 0) :
  4 * s = π * r^2 → r = 2 * Real.sqrt 2 / π := by
  sorry

end NUMINAMATH_CALUDE_square_circle_relation_l1362_136217


namespace NUMINAMATH_CALUDE_rectangle_area_with_hole_l1362_136203

theorem rectangle_area_with_hole (x : ℝ) : 
  (2*x + 8) * (x + 6) - (2*x - 2) * (x - 1) = 24*x + 46 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_with_hole_l1362_136203


namespace NUMINAMATH_CALUDE_radical_calculation_l1362_136224

theorem radical_calculation : 
  Real.sqrt (1 / 4) * Real.sqrt 16 - (Real.sqrt (1 / 9))⁻¹ - Real.sqrt 0 + Real.sqrt 45 / Real.sqrt 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_radical_calculation_l1362_136224


namespace NUMINAMATH_CALUDE_teacher_engineer_ratio_l1362_136279

theorem teacher_engineer_ratio 
  (t e : ℕ) -- t is the number of teachers, e is the number of engineers
  (h_group : t + e > 0) -- ensures the group is not empty
  (h_avg : (40 * t + 55 * e) / (t + e) = 45) -- average age of the entire group is 45
  : t = 2 * e := by
sorry

end NUMINAMATH_CALUDE_teacher_engineer_ratio_l1362_136279


namespace NUMINAMATH_CALUDE_de_bruijn_semi_integer_l1362_136219

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

end NUMINAMATH_CALUDE_de_bruijn_semi_integer_l1362_136219


namespace NUMINAMATH_CALUDE_sum_of_three_consecutive_odd_numbers_l1362_136252

theorem sum_of_three_consecutive_odd_numbers (n : ℕ) (h : n = 21) :
  n + (n + 2) + (n + 4) = 69 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_consecutive_odd_numbers_l1362_136252


namespace NUMINAMATH_CALUDE_pudding_weight_l1362_136237

theorem pudding_weight (w : ℝ) 
  (h1 : 9/11 * w + 4 = w - (w - (9/11 * w + 4)))
  (h2 : 9/11 * w + 52 = w + (w - (9/11 * w + 4))) :
  w = 154 := by sorry

end NUMINAMATH_CALUDE_pudding_weight_l1362_136237


namespace NUMINAMATH_CALUDE_square_roots_problem_l1362_136250

theorem square_roots_problem (x a : ℝ) : 
  x > 0 ∧ (a + 1) ^ 2 = x ∧ (a - 3) ^ 2 = x → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_roots_problem_l1362_136250


namespace NUMINAMATH_CALUDE_sum_of_five_consecutive_integers_l1362_136293

theorem sum_of_five_consecutive_integers (n : ℤ) : 
  (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) = 5 * n + 20 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_five_consecutive_integers_l1362_136293


namespace NUMINAMATH_CALUDE_leahs_coins_value_l1362_136223

theorem leahs_coins_value (d n : ℕ) : 
  d + n = 15 ∧ 
  d = 2 * (n + 3) → 
  10 * d + 5 * n = 135 :=
by sorry

end NUMINAMATH_CALUDE_leahs_coins_value_l1362_136223


namespace NUMINAMATH_CALUDE_sum_of_distinct_roots_is_zero_l1362_136271

theorem sum_of_distinct_roots_is_zero 
  (a b c x y : ℝ) 
  (ha : a^3 + a*x + y = 0)
  (hb : b^3 + b*x + y = 0)
  (hc : c^3 + c*x + y = 0)
  (hab : a ≠ b)
  (hbc : b ≠ c)
  (hac : a ≠ c) :
  a + b + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_distinct_roots_is_zero_l1362_136271


namespace NUMINAMATH_CALUDE_four_weighings_sufficient_three_weighings_insufficient_l1362_136283

/-- Represents the result of a weighing: lighter, equal, or heavier -/
inductive WeighingResult
  | Lighter
  | Equal
  | Heavier

/-- Represents a sequence of weighing results -/
def WeighingSequence := List WeighingResult

/-- The number of cans in the problem -/
def numCans : Nat := 80

/-- A function that simulates a weighing, returning a WeighingResult -/
def weighing (a b : Nat) : WeighingResult :=
  sorry

theorem four_weighings_sufficient :
  ∃ (f : Fin numCans → WeighingSequence),
    (∀ (i j : Fin numCans), i ≠ j → f i ≠ f j) ∧
    (∀ (s : WeighingSequence), s.length = 4) :=
  sorry

theorem three_weighings_insufficient :
  ¬∃ (f : Fin numCans → WeighingSequence),
    (∀ (i j : Fin numCans), i ≠ j → f i ≠ f j) ∧
    (∀ (s : WeighingSequence), s.length = 3) :=
  sorry

end NUMINAMATH_CALUDE_four_weighings_sufficient_three_weighings_insufficient_l1362_136283


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1362_136228

theorem quadratic_equation_roots (p q : ℤ) (h1 : p + q = 28) : 
  ∃ (x₁ x₂ : ℤ), x₁ > 0 ∧ x₂ > 0 ∧ x₁^2 + p*x₁ + q = 0 ∧ x₂^2 + p*x₂ + q = 0 ∧ 
  ((x₁ = 30 ∧ x₂ = 2) ∨ (x₁ = 2 ∧ x₂ = 30)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1362_136228


namespace NUMINAMATH_CALUDE_solution_set_equality_l1362_136207

theorem solution_set_equality (a : ℝ) : 
  (∀ x, (a - 1) * x < a + 5 ↔ 2 * x < 4) → a = 7 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_equality_l1362_136207
