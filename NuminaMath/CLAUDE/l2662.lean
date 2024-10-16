import Mathlib

namespace NUMINAMATH_CALUDE_plane_equation_proof_l2662_266202

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a vector in 3D space -/
structure Vector3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculates the vector between two points -/
def vectorBetweenPoints (p1 p2 : Point3D) : Vector3D :=
  { x := p2.x - p1.x
    y := p2.y - p1.y
    z := p2.z - p1.z }

/-- Checks if a vector is perpendicular to a plane -/
def isPerpendicularToPlane (v : Vector3D) (a b c : ℝ) : Prop :=
  a * v.x + b * v.y + c * v.z = 0

/-- Checks if a point lies on a plane -/
def isPointOnPlane (p : Point3D) (a b c d : ℝ) : Prop :=
  a * p.x + b * p.y + c * p.z + d = 0

theorem plane_equation_proof (A B C : Point3D) 
    (h1 : A.x = -4 ∧ A.y = -2 ∧ A.z = 5)
    (h2 : B.x = 3 ∧ B.y = -3 ∧ B.z = -7)
    (h3 : C.x = 9 ∧ C.y = 3 ∧ C.z = -7) :
    let BC := vectorBetweenPoints B C
    isPerpendicularToPlane BC 1 1 0 ∧ isPointOnPlane A 1 1 0 (-6) := by
  sorry

#check plane_equation_proof

end NUMINAMATH_CALUDE_plane_equation_proof_l2662_266202


namespace NUMINAMATH_CALUDE_difference_of_squares_l2662_266214

theorem difference_of_squares (x y : ℝ) : x^2 - 4*y^2 = (x - 2*y) * (x + 2*y) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l2662_266214


namespace NUMINAMATH_CALUDE_base_side_length_l2662_266257

/-- Represents a right pyramid with a square base -/
structure RightPyramid where
  base_side : ℝ
  slant_height : ℝ
  lateral_face_area : ℝ

/-- The lateral face area of a right pyramid is half the product of its base side and slant height -/
axiom lateral_face_area_formula (p : RightPyramid) : 
  p.lateral_face_area = (1/2) * p.base_side * p.slant_height

/-- 
Given a right pyramid with a square base, if its lateral face area is 120 square meters 
and its slant height is 24 meters, then the length of a side of its base is 10 meters.
-/
theorem base_side_length (p : RightPyramid) 
  (h1 : p.lateral_face_area = 120) 
  (h2 : p.slant_height = 24) : 
  p.base_side = 10 := by
sorry

end NUMINAMATH_CALUDE_base_side_length_l2662_266257


namespace NUMINAMATH_CALUDE_remaining_note_denomination_l2662_266212

theorem remaining_note_denomination 
  (total_amount : ℕ)
  (total_notes : ℕ)
  (fifty_notes : ℕ)
  (h1 : total_amount = 10350)
  (h2 : total_notes = 72)
  (h3 : fifty_notes = 57) :
  (total_amount - 50 * fifty_notes) / (total_notes - fifty_notes) = 500 := by
  sorry

end NUMINAMATH_CALUDE_remaining_note_denomination_l2662_266212


namespace NUMINAMATH_CALUDE_gcf_of_40_120_45_l2662_266231

theorem gcf_of_40_120_45 : Nat.gcd 40 (Nat.gcd 120 45) = 5 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_40_120_45_l2662_266231


namespace NUMINAMATH_CALUDE_halfway_point_fractions_l2662_266237

theorem halfway_point_fractions (a b : ℚ) (ha : a = 1/12) (hb : b = 13/12) :
  (a + b) / 2 = 7/12 := by
  sorry

end NUMINAMATH_CALUDE_halfway_point_fractions_l2662_266237


namespace NUMINAMATH_CALUDE_intersection_point_d_equals_two_l2662_266270

/-- A function f(x) = 4x + c where c is an integer -/
def f (c : ℤ) : ℝ → ℝ := λ x ↦ 4 * x + c

/-- The inverse of f -/
noncomputable def f_inv (c : ℤ) : ℝ → ℝ := λ x ↦ (x - c) / 4

theorem intersection_point_d_equals_two (c d : ℤ) :
  f c 2 = d ∧ f_inv c d = 2 → d = 2 := by sorry

end NUMINAMATH_CALUDE_intersection_point_d_equals_two_l2662_266270


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l2662_266248

theorem arithmetic_mean_problem (a : ℝ) : 
  (1 + a) / 2 = 2 → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l2662_266248


namespace NUMINAMATH_CALUDE_rectangular_field_perimeter_l2662_266275

/-- The perimeter of a rectangular field with length 7/5 of its width and width of 80 meters is 384 meters. -/
theorem rectangular_field_perimeter : 
  ∀ (length width : ℝ),
  length = (7/5) * width →
  width = 80 →
  2 * (length + width) = 384 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_perimeter_l2662_266275


namespace NUMINAMATH_CALUDE_largest_n_divisible_by_seven_l2662_266241

def expression (n : ℕ) : ℤ :=
  10 * (n - 3)^5 - 2 * n^2 + 20 * n - 36

theorem largest_n_divisible_by_seven :
  ∃ (n : ℕ), n < 50000 ∧
    7 ∣ expression n ∧
    ∀ (m : ℕ), m < 50000 → 7 ∣ expression m → m ≤ n :=
by
  use 49999
  sorry

end NUMINAMATH_CALUDE_largest_n_divisible_by_seven_l2662_266241


namespace NUMINAMATH_CALUDE_division_result_l2662_266234

theorem division_result : (24 : ℝ) / (52 - 40) = 2 := by
  sorry

end NUMINAMATH_CALUDE_division_result_l2662_266234


namespace NUMINAMATH_CALUDE_shaded_area_semicircles_pattern_l2662_266222

/-- The area of the shaded region in a 1-foot length of alternating semicircles pattern --/
theorem shaded_area_semicircles_pattern (foot_to_inch : ℝ) (diameter : ℝ) (π : ℝ) : 
  foot_to_inch = 12 →
  diameter = 2 →
  (foot_to_inch / diameter) * (π * (diameter / 2)^2) = 6 * π := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_semicircles_pattern_l2662_266222


namespace NUMINAMATH_CALUDE_girls_money_and_scarf_price_l2662_266245

-- Define variables
variable (x y s m v : ℝ)

-- Define the conditions
def conditions (x y s m v : ℝ) : Prop :=
  y + 40 < s ∧ s < y + 50 ∧
  x + 30 < s ∧ s ≤ x + 40 - m ∧ m < 10 ∧
  0.8 * s ≤ x + 20 ∧ 0.8 * s ≤ y + 30 ∧
  0.8 * s - 4 = y + 20 ∧
  y < 0.6 * s - 3 ∧ 0.6 * s - 3 < y + 10 ∧
  x - 10 < 0.6 * s - 3 ∧ 0.6 * s - 3 < x ∧
  x + y - 1.2 * s = v

-- Theorem statement
theorem girls_money_and_scarf_price (x y s m v : ℝ) 
  (h : conditions x y s m v) : 
  61 ≤ x ∧ x ≤ 69 ∧ 52 ≤ y ∧ y ≤ 60 ∧ 91 ≤ s ∧ s ≤ 106 := by
  sorry

end NUMINAMATH_CALUDE_girls_money_and_scarf_price_l2662_266245


namespace NUMINAMATH_CALUDE_x_squared_geq_one_necessary_not_sufficient_l2662_266280

theorem x_squared_geq_one_necessary_not_sufficient :
  (∀ x : ℝ, x > 1 → x^2 ≥ 1) ∧
  (∃ x : ℝ, x^2 ≥ 1 ∧ ¬(x > 1)) :=
by sorry

end NUMINAMATH_CALUDE_x_squared_geq_one_necessary_not_sufficient_l2662_266280


namespace NUMINAMATH_CALUDE_negation_equivalence_l2662_266299

theorem negation_equivalence : 
  (¬ ∃ x : ℝ, x^2 - 2*x + 1 < 0) ↔ (∀ x : ℝ, x^2 - 2*x + 1 ≥ 0) := by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2662_266299


namespace NUMINAMATH_CALUDE_percent_relation_l2662_266219

theorem percent_relation (P Q : ℝ) (h : (1/2) * P = (1/5) * Q) :
  P = (2/5) * Q := by
  sorry

end NUMINAMATH_CALUDE_percent_relation_l2662_266219


namespace NUMINAMATH_CALUDE_susan_game_remaining_spaces_l2662_266278

/-- A board game with a given number of spaces and a player's movements --/
structure BoardGame where
  total_spaces : ℕ
  first_move : ℕ
  second_move_forward : ℕ
  second_move_backward : ℕ
  third_move : ℕ

/-- Calculate the remaining spaces to reach the end of the game --/
def remaining_spaces (game : BoardGame) : ℕ :=
  game.total_spaces - (game.first_move + game.third_move + game.second_move_forward - game.second_move_backward)

/-- Theorem stating that for Susan's game, the remaining spaces is 37 --/
theorem susan_game_remaining_spaces :
  let game : BoardGame := {
    total_spaces := 48,
    first_move := 8,
    second_move_forward := 2,
    second_move_backward := 5,
    third_move := 6
  }
  remaining_spaces game = 37 := by sorry

end NUMINAMATH_CALUDE_susan_game_remaining_spaces_l2662_266278


namespace NUMINAMATH_CALUDE_candy_bars_purchased_l2662_266207

theorem candy_bars_purchased (initial_amount : ℕ) (candy_bar_cost : ℕ) (remaining_amount : ℕ) : 
  initial_amount = 20 →
  candy_bar_cost = 2 →
  remaining_amount = 12 →
  (initial_amount - remaining_amount) / candy_bar_cost = 4 := by
sorry

end NUMINAMATH_CALUDE_candy_bars_purchased_l2662_266207


namespace NUMINAMATH_CALUDE_certain_number_problem_l2662_266266

theorem certain_number_problem (x y : ℤ) : x = 15 ∧ 2 * x = (y - x) + 19 → y = 26 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l2662_266266


namespace NUMINAMATH_CALUDE_right_triangle_area_l2662_266295

theorem right_triangle_area (α : Real) (hypotenuse : Real) :
  α = 30 * π / 180 →
  hypotenuse = 20 →
  ∃ (area : Real), area = 50 * Real.sqrt 3 ∧
    area = (1 / 2) * (hypotenuse / 2) * (hypotenuse / 2 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2662_266295


namespace NUMINAMATH_CALUDE_range_of_x_l2662_266206

theorem range_of_x (a b c : ℝ) (h : a^2 + 2*b^2 + 3*c^2 = 6) :
  (∃ x : ℝ, ∀ y : ℝ, (∃ a' b' c' : ℝ, a'^2 + 2*b'^2 + 3*c'^2 = 6 ∧ a' + 2*b' + 3*c' > |y + 1|) ↔ -7 < y ∧ y < 5) :=
sorry

end NUMINAMATH_CALUDE_range_of_x_l2662_266206


namespace NUMINAMATH_CALUDE_cricket_team_average_age_l2662_266210

theorem cricket_team_average_age :
  ∀ (team_size : ℕ) (captain_age : ℕ) (wicket_keeper_age_diff : ℕ) (remaining_players_age_diff : ℕ),
    team_size = 11 →
    captain_age = 24 →
    wicket_keeper_age_diff = 3 →
    remaining_players_age_diff = 1 →
    ∃ (team_average_age : ℚ),
      team_average_age = 21 ∧
      (team_size : ℚ) * team_average_age = 
        captain_age + (captain_age + wicket_keeper_age_diff) + 
        ((team_size - 2) : ℚ) * (team_average_age - remaining_players_age_diff) :=
by
  sorry

end NUMINAMATH_CALUDE_cricket_team_average_age_l2662_266210


namespace NUMINAMATH_CALUDE_find_p_l2662_266274

theorem find_p (m : ℕ) (p : ℕ) :
  m = 34 →
  ((1 ^ (m + 1)) / (5 ^ (m + 1))) * ((1 ^ 18) / (4 ^ 18)) = 1 / (2 * (10 ^ p)) →
  p = 35 := by
  sorry

end NUMINAMATH_CALUDE_find_p_l2662_266274


namespace NUMINAMATH_CALUDE_lower_right_is_four_l2662_266290

def Grid := Fin 4 → Fin 4 → Fin 4

def valid_grid (g : Grid) : Prop :=
  (∀ i j k, i ≠ j → g i k ≠ g j k) ∧
  (∀ i j k, i ≠ j → g k i ≠ g k j)

def initial_conditions (g : Grid) : Prop :=
  g 0 1 = 1 ∧ g 0 3 = 2 ∧ g 1 2 = 3 ∧ g 2 0 = 3 ∧ g 3 1 = 0

theorem lower_right_is_four (g : Grid) 
  (h1 : valid_grid g) 
  (h2 : initial_conditions g) : 
  g 3 3 = 3 := by sorry

end NUMINAMATH_CALUDE_lower_right_is_four_l2662_266290


namespace NUMINAMATH_CALUDE_smallest_n_for_perfect_square_sum_l2662_266242

theorem smallest_n_for_perfect_square_sum (n : ℕ) : n = 7 ↔ 
  (∀ k ≥ n, ∀ x ∈ Finset.range k, ∃ y ∈ Finset.range k, y ≠ x ∧ ∃ m : ℕ, x + y = m^2) ∧
  (∀ n' < n, ∃ k ≥ n', ∃ x ∈ Finset.range k, ∀ y ∈ Finset.range k, y = x ∨ ∀ m : ℕ, x + y ≠ m^2) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_perfect_square_sum_l2662_266242


namespace NUMINAMATH_CALUDE_twelve_chairs_subsets_l2662_266297

/-- The number of chairs arranged in a circle -/
def n : ℕ := 12

/-- A function that calculates the number of subsets containing at least four adjacent chairs
    for n chairs arranged in a circle -/
def subsets_with_adjacent_chairs (n : ℕ) : ℕ := sorry

/-- Theorem stating that for 12 chairs arranged in a circle,
    the number of subsets containing at least four adjacent chairs is 1701 -/
theorem twelve_chairs_subsets :
  subsets_with_adjacent_chairs n = 1701 := by sorry

end NUMINAMATH_CALUDE_twelve_chairs_subsets_l2662_266297


namespace NUMINAMATH_CALUDE_chess_tournament_games_l2662_266200

theorem chess_tournament_games (n : ℕ) 
  (total_players : ℕ) (total_games : ℕ) :
  total_players = 6 →
  total_games = 30 →
  total_games = n * (total_players.choose 2) →
  n = 2 := by
sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l2662_266200


namespace NUMINAMATH_CALUDE_circle_extrema_l2662_266236

theorem circle_extrema (x y : ℝ) (h : (x - 3)^2 + (y - 3)^2 = 6) :
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    (x₁ - 3)^2 + (y₁ - 3)^2 = 6 ∧ 
    (x₂ - 3)^2 + (y₂ - 3)^2 = 6 ∧
    (∀ (x' y' : ℝ), (x' - 3)^2 + (y' - 3)^2 = 6 → y' / x' ≤ y₁ / x₁ ∧ y' / x' ≥ y₂ / x₂) ∧
    y₁ / x₁ = 3 + 2 * Real.sqrt 2 ∧
    y₂ / x₂ = 3 - 2 * Real.sqrt 2) ∧
  (∃ (x₃ y₃ x₄ y₄ : ℝ),
    (x₃ - 3)^2 + (y₃ - 3)^2 = 6 ∧
    (x₄ - 3)^2 + (y₄ - 3)^2 = 6 ∧
    (∀ (x' y' : ℝ), (x' - 3)^2 + (y' - 3)^2 = 6 → 
      Real.sqrt ((x' - 2)^2 + y'^2) ≤ Real.sqrt ((x₃ - 2)^2 + y₃^2) ∧
      Real.sqrt ((x' - 2)^2 + y'^2) ≥ Real.sqrt ((x₄ - 2)^2 + y₄^2)) ∧
    Real.sqrt ((x₃ - 2)^2 + y₃^2) = Real.sqrt 10 + Real.sqrt 6 ∧
    Real.sqrt ((x₄ - 2)^2 + y₄^2) = Real.sqrt 10 - Real.sqrt 6) := by
  sorry

end NUMINAMATH_CALUDE_circle_extrema_l2662_266236


namespace NUMINAMATH_CALUDE_min_balls_to_draw_correct_l2662_266269

/-- Represents the number of balls of each color in the container -/
structure BallContainer :=
  (red : ℕ)
  (green : ℕ)
  (yellow : ℕ)
  (blue : ℕ)
  (purple : ℕ)
  (orange : ℕ)

/-- The initial distribution of balls in the container -/
def initialContainer : BallContainer :=
  { red := 40
  , green := 25
  , yellow := 20
  , blue := 15
  , purple := 10
  , orange := 5 }

/-- The minimum number of balls of a single color we want to guarantee -/
def targetCount : ℕ := 18

/-- Function to calculate the minimum number of balls to draw -/
def minBallsToDraw (container : BallContainer) (target : ℕ) : ℕ :=
  sorry

theorem min_balls_to_draw_correct :
  minBallsToDraw initialContainer targetCount = 82 :=
sorry

end NUMINAMATH_CALUDE_min_balls_to_draw_correct_l2662_266269


namespace NUMINAMATH_CALUDE_trees_difference_l2662_266279

theorem trees_difference (initial_trees : ℕ) (dead_trees : ℕ) 
  (h1 : initial_trees = 14) (h2 : dead_trees = 9) : 
  dead_trees - (initial_trees - dead_trees) = 4 := by
  sorry

end NUMINAMATH_CALUDE_trees_difference_l2662_266279


namespace NUMINAMATH_CALUDE_ratio_equals_five_sixths_l2662_266282

theorem ratio_equals_five_sixths
  (a b c x y z : ℝ)
  (sum_squares_abc : a^2 + b^2 + c^2 = 25)
  (sum_squares_xyz : x^2 + y^2 + z^2 = 36)
  (dot_product : a*x + b*y + c*z = 30) :
  (a + b + c) / (x + y + z) = 5/6 := by
  sorry

#check ratio_equals_five_sixths

end NUMINAMATH_CALUDE_ratio_equals_five_sixths_l2662_266282


namespace NUMINAMATH_CALUDE_vector_parallel_tangent_l2662_266240

/-- Given points A, B, and C in a 2D Cartesian coordinate system,
    prove that vector AB equals (1, √3) and tan x equals √3 when AB is parallel to OC. -/
theorem vector_parallel_tangent (x : ℝ) : 
  let A : ℝ × ℝ := (-1, 0)
  let B : ℝ × ℝ := (0, Real.sqrt 3)
  let C : ℝ × ℝ := (Real.cos x, Real.sin x)
  let AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
  let OC : ℝ × ℝ := C
  AB.2 / AB.1 = OC.2 / OC.1 →
  AB = (1, Real.sqrt 3) ∧ Real.tan x = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_vector_parallel_tangent_l2662_266240


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l2662_266235

theorem cubic_equation_solution : 
  ∃ x : ℝ, x^3 + 2*(x+1)^3 + (x+2)^3 = (x+4)^3 ∧ x = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l2662_266235


namespace NUMINAMATH_CALUDE_two_digit_number_property_l2662_266289

theorem two_digit_number_property (n : ℕ) : 
  10 ≤ n ∧ n < 100 ∧ 
  (n / 10 + n % 10 = 3) →
  (n / 2 : ℚ) - (n / 4 : ℚ) = 5.25 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_property_l2662_266289


namespace NUMINAMATH_CALUDE_milk_price_problem_l2662_266243

theorem milk_price_problem (initial_cost initial_bottles subsequent_cost : ℝ) : 
  initial_cost = 108 →
  subsequent_cost = 90 →
  ∃ (price : ℝ), 
    initial_bottles * price = initial_cost ∧
    (initial_bottles + 1) * (price * 0.25) = subsequent_cost →
    price = 12 := by
  sorry

end NUMINAMATH_CALUDE_milk_price_problem_l2662_266243


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2662_266259

theorem sufficient_not_necessary_condition (a b c : ℝ) :
  (∀ a b c : ℝ, a * c^2 > b * c^2 → a > b) ∧
  (∃ a b c : ℝ, a > b ∧ a * c^2 ≤ b * c^2) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2662_266259


namespace NUMINAMATH_CALUDE_sum_of_coefficients_is_one_l2662_266217

theorem sum_of_coefficients_is_one : 
  let p (x : ℝ) := 3*(x^8 - 2*x^5 + 4*x^3 - 6) - 5*(2*x^4 + 3*x - 7) + 6*(x^6 - x^2 + 1)
  p 1 = 1 := by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_is_one_l2662_266217


namespace NUMINAMATH_CALUDE_tower_height_differences_l2662_266201

/-- Heights of towers in meters -/
def CN_Tower_height : ℝ := 553
def CN_Tower_Space_Needle_diff : ℝ := 369
def Eiffel_Tower_height : ℝ := 330
def Jeddah_Tower_predicted_height : ℝ := 1000

/-- Calculate the Space Needle height -/
def Space_Needle_height : ℝ := CN_Tower_height - CN_Tower_Space_Needle_diff

/-- Theorem stating the height differences -/
theorem tower_height_differences :
  (Eiffel_Tower_height - Space_Needle_height = 146) ∧
  (Jeddah_Tower_predicted_height - Eiffel_Tower_height = 670) :=
by sorry

end NUMINAMATH_CALUDE_tower_height_differences_l2662_266201


namespace NUMINAMATH_CALUDE_problem_statement_l2662_266298

theorem problem_statement (a b : ℝ) (h1 : a > 1) (h2 : a * b = a + b + 8) :
  (b > 1) ∧ 
  (∀ x y : ℝ, x > 1 ∧ x * y = x + y + 8 → a + b ≤ x + y) ∧ 
  (∀ x y : ℝ, x > 1 ∧ x * y = x + y + 8 → a * b ≤ x * y) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l2662_266298


namespace NUMINAMATH_CALUDE_factorization_equality_l2662_266205

theorem factorization_equality (a b : ℝ) : a * b^2 - 4 * a * b + 4 * a = a * (b - 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2662_266205


namespace NUMINAMATH_CALUDE_equation_solution_l2662_266228

theorem equation_solution :
  ∃ (x : ℚ), x ≠ -3 ∧ (x^2 + 3*x + 4) / (x + 3) = x + 6 ↔ x = -7/3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2662_266228


namespace NUMINAMATH_CALUDE_expression_equality_l2662_266224

theorem expression_equality : 
  Real.sqrt 8 - (2017 - Real.pi)^0 - 4^(-1 : ℤ) + (-1/2)^2 = 2 * Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l2662_266224


namespace NUMINAMATH_CALUDE_plane_equation_transformation_l2662_266260

theorem plane_equation_transformation (A B C D : ℝ) 
  (hA : A ≠ 0) (hB : B ≠ 0) (hC : C ≠ 0) (hD : D ≠ 0) :
  ∃ p q r : ℝ, 
    (∀ x y z : ℝ, A * x + B * y + C * z + D = 0 ↔ x / p + y / q + z / r = 1) ∧
    p = -D / A ∧ q = -D / B ∧ r = -D / C :=
by sorry

end NUMINAMATH_CALUDE_plane_equation_transformation_l2662_266260


namespace NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l2662_266256

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sixth_term
  (a : ℕ → ℝ)
  (h_geom : is_geometric_sequence a)
  (h_roots : a 2 ^ 2 - 13 * a 2 + 14 = 0 ∧ a 10 ^ 2 - 13 * a 10 + 14 = 0) :
  a 6 = Real.sqrt 14 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l2662_266256


namespace NUMINAMATH_CALUDE_inequality_holds_l2662_266229

theorem inequality_holds (x : ℝ) : 
  (4 * x^2) / (1 - Real.sqrt (1 + 2*x))^2 < 2*x + 9 ↔ 
  (x ≥ -1/2 ∧ x < 0) ∨ (x > 0 ∧ x < 45/8) := by sorry

end NUMINAMATH_CALUDE_inequality_holds_l2662_266229


namespace NUMINAMATH_CALUDE_speed_time_relationship_l2662_266253

theorem speed_time_relationship (t v : ℝ) : t = 5 * v^2 ∧ t = 20 → v = 2 := by
  sorry

end NUMINAMATH_CALUDE_speed_time_relationship_l2662_266253


namespace NUMINAMATH_CALUDE_binary_representation_properties_l2662_266264

def has_exactly_three_ones (n : ℕ) : Prop :=
  (n.digits 2).count 1 = 3

def is_multiple_of_617 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 617 * k

theorem binary_representation_properties (n : ℕ) 
  (h1 : is_multiple_of_617 n) 
  (h2 : has_exactly_three_ones n) : 
  ((n.digits 2).length ≥ 9) ∧ 
  ((n.digits 2).length = 10 → Even n) :=
sorry

end NUMINAMATH_CALUDE_binary_representation_properties_l2662_266264


namespace NUMINAMATH_CALUDE_tom_profit_l2662_266261

/-- Represents the types of properties Tom mows --/
inductive PropertyType
| Small
| Medium
| Large

/-- Calculates the total earnings from lawn mowing --/
def lawnMowingEarnings (smallCount medium_count largeCount : ℕ) : ℕ :=
  12 * smallCount + 15 * medium_count + 20 * largeCount

/-- Calculates the total earnings from side tasks --/
def sideTaskEarnings (taskCount : ℕ) : ℕ :=
  10 * taskCount

/-- Calculates the total expenses --/
def totalExpenses : ℕ := 20 + 10

/-- Calculates the total profit --/
def totalProfit (lawnEarnings sideEarnings : ℕ) : ℕ :=
  lawnEarnings + sideEarnings - totalExpenses

/-- Theorem stating Tom's profit for the given month --/
theorem tom_profit :
  totalProfit (lawnMowingEarnings 2 2 1) (sideTaskEarnings 5) = 94 := by
  sorry

end NUMINAMATH_CALUDE_tom_profit_l2662_266261


namespace NUMINAMATH_CALUDE_unique_quadratic_solution_l2662_266225

theorem unique_quadratic_solution (a : ℝ) (ha : a ≠ 0) :
  (∃! x, a * x^2 + 30 * x + 5 = 0) → 
  (∃ x, a * x^2 + 30 * x + 5 = 0 ∧ x = -1/3) :=
by sorry

end NUMINAMATH_CALUDE_unique_quadratic_solution_l2662_266225


namespace NUMINAMATH_CALUDE_mars_radius_scientific_notation_l2662_266250

theorem mars_radius_scientific_notation :
  3395000 = 3.395 * (10 ^ 6) := by sorry

end NUMINAMATH_CALUDE_mars_radius_scientific_notation_l2662_266250


namespace NUMINAMATH_CALUDE_base8_642_equals_base10_418_l2662_266267

/-- Converts a base-8 number to base-10 -/
def base8_to_base10 (x : ℕ) : ℕ :=
  let d₂ := x / 100
  let d₁ := (x / 10) % 10
  let d₀ := x % 10
  d₂ * 8^2 + d₁ * 8^1 + d₀ * 8^0

theorem base8_642_equals_base10_418 : base8_to_base10 642 = 418 := by
  sorry

end NUMINAMATH_CALUDE_base8_642_equals_base10_418_l2662_266267


namespace NUMINAMATH_CALUDE_min_perimeter_isosceles_triangle_l2662_266294

/-- Represents a triangle with integer side lengths where two sides are equal -/
structure IsoscelesTriangle where
  a : ℕ  -- length of BC
  b : ℕ  -- length of AB and AC
  ab_eq_ac : b = b  -- AB = AC

/-- Represents the geometric configuration described in the problem -/
structure GeometricConfiguration (t : IsoscelesTriangle) where
  ω_center_is_incenter : Bool
  excircle_bc_internal : Bool
  excircle_ab_external : Bool
  excircle_ac_not_tangent : Bool

/-- The theorem statement -/
theorem min_perimeter_isosceles_triangle 
  (t : IsoscelesTriangle) 
  (config : GeometricConfiguration t) : 
  2 * t.b + t.a ≥ 20 := by
  sorry

#check min_perimeter_isosceles_triangle

end NUMINAMATH_CALUDE_min_perimeter_isosceles_triangle_l2662_266294


namespace NUMINAMATH_CALUDE_inequality_holds_l2662_266283

open Real

/-- A function satisfying the given differential equation -/
def SatisfiesDiffEq (f : ℝ → ℝ) : Prop :=
  ∀ x > 0, x * (deriv^[2] f x) + 2 * f x = 1 / x^2

theorem inequality_holds (f : ℝ → ℝ) (hf : SatisfiesDiffEq f) :
  f 2 / 9 < f 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_l2662_266283


namespace NUMINAMATH_CALUDE_identity_implies_a_minus_b_equals_one_l2662_266247

theorem identity_implies_a_minus_b_equals_one :
  ∀ (a b : ℚ),
  (∀ (y : ℚ), y > 0 → a / (y - 3) + b / (y + 5) = (3 * y + 7) / ((y - 3) * (y + 5))) →
  a - b = 1 := by
sorry

end NUMINAMATH_CALUDE_identity_implies_a_minus_b_equals_one_l2662_266247


namespace NUMINAMATH_CALUDE_sum_of_y_coordinates_l2662_266262

theorem sum_of_y_coordinates : ∀ y₁ y₂ : ℝ,
  (4 - (-1))^2 + (y₁ - 3)^2 = 8^2 →
  (4 - (-1))^2 + (y₂ - 3)^2 = 8^2 →
  y₁ + y₂ = 6 := by sorry

end NUMINAMATH_CALUDE_sum_of_y_coordinates_l2662_266262


namespace NUMINAMATH_CALUDE_simplify_complex_expression_l2662_266227

theorem simplify_complex_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (((a^4 * b^2 * c)^(5/3) * (a^3 * b^2 * c)^2)^3)^(1/11) / a^(5/11) = a^3 * b^2 * c :=
by sorry

end NUMINAMATH_CALUDE_simplify_complex_expression_l2662_266227


namespace NUMINAMATH_CALUDE_digit_difference_quotient_l2662_266208

/-- Given that 524 in base 7 equals 3cd in base 10, where c and d are single digits,
    prove that (c - d) / 5 = -0.8 -/
theorem digit_difference_quotient (c d : ℕ) : 
  c < 10 → d < 10 → (5 * 7^2 + 2 * 7 + 4 : ℕ) = 300 + 10 * c + d → 
  (c - d : ℚ) / 5 = -4/5 := by sorry

end NUMINAMATH_CALUDE_digit_difference_quotient_l2662_266208


namespace NUMINAMATH_CALUDE_bill_apples_left_l2662_266221

/-- The number of apples Bill has left after distributing them -/
def apples_left (total : ℕ) (children : ℕ) (apples_per_child : ℕ) (pies : ℕ) (apples_per_pie : ℕ) : ℕ :=
  total - (children * apples_per_child + pies * apples_per_pie)

/-- Theorem: Bill has 24 apples left -/
theorem bill_apples_left :
  apples_left 50 2 3 2 10 = 24 := by
  sorry

end NUMINAMATH_CALUDE_bill_apples_left_l2662_266221


namespace NUMINAMATH_CALUDE_power_sum_equality_l2662_266286

theorem power_sum_equality (x y a b : ℝ) (h1 : x + y = a + b) (h2 : x^2 + y^2 = a^2 + b^2) :
  ∀ n : ℕ, x^n + y^n = a^n + b^n := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equality_l2662_266286


namespace NUMINAMATH_CALUDE_david_recreation_spending_l2662_266203

-- Define the wages from last week as a parameter
def last_week_wages : ℝ := sorry

-- Define the percentage spent on recreation last week
def last_week_recreation_percent : ℝ := 0.40

-- Define the wage reduction percentage
def wage_reduction_percent : ℝ := 0.05

-- Define the increase in recreation spending
def recreation_increase_percent : ℝ := 1.1875

-- Calculate this week's wages
def this_week_wages : ℝ := last_week_wages * (1 - wage_reduction_percent)

-- Calculate the amount spent on recreation last week
def last_week_recreation_amount : ℝ := last_week_wages * last_week_recreation_percent

-- Calculate the amount spent on recreation this week
def this_week_recreation_amount : ℝ := last_week_recreation_amount * recreation_increase_percent

-- Define the theorem
theorem david_recreation_spending :
  this_week_recreation_amount / this_week_wages = 0.5 := by sorry

end NUMINAMATH_CALUDE_david_recreation_spending_l2662_266203


namespace NUMINAMATH_CALUDE_cricket_run_rate_theorem_l2662_266255

/-- Represents a cricket game situation -/
structure CricketGame where
  totalOvers : ℕ
  firstPeriodOvers : ℕ
  firstPeriodRunRate : ℚ
  targetRuns : ℕ

/-- Calculates the required run rate for the remaining overs -/
def requiredRunRate (game : CricketGame) : ℚ :=
  let remainingOvers := game.totalOvers - game.firstPeriodOvers
  let runsScored := game.firstPeriodRunRate * game.firstPeriodOvers
  let runsNeeded := game.targetRuns - runsScored
  runsNeeded / remainingOvers

/-- Theorem stating the required run rate for the given game situation -/
theorem cricket_run_rate_theorem (game : CricketGame) 
  (h1 : game.totalOvers = 50)
  (h2 : game.firstPeriodOvers = 10)
  (h3 : game.firstPeriodRunRate = 21/5)  -- 4.2 as a fraction
  (h4 : game.targetRuns = 282) :
  requiredRunRate game = 6 := by
  sorry

end NUMINAMATH_CALUDE_cricket_run_rate_theorem_l2662_266255


namespace NUMINAMATH_CALUDE_geometric_sequence_proof_l2662_266273

theorem geometric_sequence_proof (a : ℕ → ℝ) :
  (∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)) →  -- geometric sequence condition
  a 1 + a 3 = 10 →                                  -- first given condition
  a 2 + a 4 = 5 →                                   -- second given condition
  ∀ n, a n = 2^(4 - n) :=                           -- conclusion to prove
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_proof_l2662_266273


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l2662_266254

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |x - 5| = 3 * x - 2 :=
by
  -- The unique solution is x = 7/4
  use 7/4
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l2662_266254


namespace NUMINAMATH_CALUDE_bowling_ball_weight_l2662_266287

/-- Given that 8 bowling balls weigh the same as 5 kayaks, and 4 kayaks weigh 120 pounds,
    prove that one bowling ball weighs 18.75 pounds. -/
theorem bowling_ball_weight :
  ∀ (bowl_weight kayak_weight : ℝ),
    8 * bowl_weight = 5 * kayak_weight →
    4 * kayak_weight = 120 →
    bowl_weight = 18.75 := by
  sorry

end NUMINAMATH_CALUDE_bowling_ball_weight_l2662_266287


namespace NUMINAMATH_CALUDE_equation_solution_l2662_266204

theorem equation_solution :
  let f (x : ℝ) := (x - 4)^4 + (x - 6)^4
  ∃ x₁ x₂ : ℝ, 
    (f x₁ = 240 ∧ f x₂ = 240) ∧
    x₁ = 5 + Real.sqrt (5 * Real.sqrt 2 - 3) ∧
    x₂ = 5 - Real.sqrt (5 * Real.sqrt 2 - 3) ∧
    ∀ x : ℝ, f x = 240 → (x = x₁ ∨ x = x₂) :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2662_266204


namespace NUMINAMATH_CALUDE_largest_number_l2662_266226

theorem largest_number (a b c d e : ℝ) : 
  a = 24680 + 1 / 1357 →
  b = 24680 - 1 / 1357 →
  c = 24680 * (1 / 1357) →
  d = 24680 / (1 / 1357) →
  e = 24680.1357 →
  d > a ∧ d > b ∧ d > c ∧ d > e := by
  sorry

end NUMINAMATH_CALUDE_largest_number_l2662_266226


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l2662_266251

-- Problem 1
theorem simplify_expression_1 (a : ℝ) : 
  a^2 - 3*a + 1 - a^2 + 6*a - 7 = 3*a - 6 := by sorry

-- Problem 2
theorem simplify_expression_2 (m n : ℝ) : 
  (3*m^2*n - 5*m*n) - 3*(4*m^2*n - 5*m*n) = -9*m^2*n + 10*m*n := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l2662_266251


namespace NUMINAMATH_CALUDE_pascal_ratio_row_l2662_266230

/-- Pascal's Triangle entry -/
def pascal (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

/-- Three consecutive entries in Pascal's Triangle are in ratio 3:4:5 -/
def ratio_condition (n : ℕ) (r : ℕ) : Prop :=
  ∃ (a b c : ℚ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧
    a * (pascal n (r+1)) = b * (pascal n r) ∧
    b * (pascal n (r+2)) = c * (pascal n (r+1)) ∧
    3 * b = 4 * a ∧ 4 * c = 5 * b

theorem pascal_ratio_row :
  ∃ (n : ℕ), n = 62 ∧ ∃ (r : ℕ), ratio_condition n r :=
sorry

end NUMINAMATH_CALUDE_pascal_ratio_row_l2662_266230


namespace NUMINAMATH_CALUDE_min_value_sum_squared_ratios_l2662_266246

theorem min_value_sum_squared_ratios (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 1) :
  (x^2 / y) + (y^2 / z) + (z^2 / x) ≥ 3 ∧
  ((x^2 / y) + (y^2 / z) + (z^2 / x) = 3 ↔ x = 1 ∧ y = 1 ∧ z = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_squared_ratios_l2662_266246


namespace NUMINAMATH_CALUDE_original_function_derivation_l2662_266218

/-- A linear function represented by its slope and y-intercept -/
structure LinearFunction where
  slope : ℝ
  intercept : ℝ

/-- Rotates a linear function 180° around the origin -/
def rotate180 (f : LinearFunction) : LinearFunction :=
  { slope := -f.slope, intercept := -f.intercept }

/-- Translates a linear function horizontally -/
def translateLeft (f : LinearFunction) (units : ℝ) : LinearFunction :=
  { slope := f.slope, intercept := f.intercept + f.slope * units }

/-- Checks if a linear function passes through two points -/
def passesThrough (f : LinearFunction) (x1 y1 x2 y2 : ℝ) : Prop :=
  f.slope * x1 + f.intercept = y1 ∧ f.slope * x2 + f.intercept = y2

theorem original_function_derivation (k b : ℝ) :
  let f := LinearFunction.mk k b
  let rotated := rotate180 f
  let translated := translateLeft rotated 2
  passesThrough translated (-4) 0 0 2 →
  k = 1/2 ∧ b = -1 := by sorry

end NUMINAMATH_CALUDE_original_function_derivation_l2662_266218


namespace NUMINAMATH_CALUDE_total_prime_factors_l2662_266265

-- Define the expression
def expression := (4 : ℕ) ^ 13 * 7 ^ 5 * 11 ^ 2

-- Define the prime factorization of 4
axiom four_eq_two_squared : (4 : ℕ) = 2 ^ 2

-- Define 7 and 11 as prime numbers
axiom seven_prime : Nat.Prime 7
axiom eleven_prime : Nat.Prime 11

-- Theorem statement
theorem total_prime_factors : 
  (Nat.factors expression).length = 33 :=
sorry

end NUMINAMATH_CALUDE_total_prime_factors_l2662_266265


namespace NUMINAMATH_CALUDE_route_number_theorem_l2662_266220

/-- Represents a digit on a seven-segment display -/
inductive Digit
| Zero | One | Two | Three | Four | Five | Six | Seven | Eight | Nine

/-- Represents a three-digit number -/
structure ThreeDigitNumber :=
  (hundreds : Digit)
  (tens : Digit)
  (units : Digit)

/-- Converts a ThreeDigitNumber to a natural number -/
def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  match n.hundreds, n.tens, n.units with
  | Digit.Three, Digit.Five, Digit.One => 351
  | Digit.Three, Digit.Five, Digit.Four => 354
  | Digit.Three, Digit.Five, Digit.Seven => 357
  | Digit.Three, Digit.Six, Digit.One => 361
  | Digit.Three, Digit.Six, Digit.Seven => 367
  | Digit.Three, Digit.Eight, Digit.One => 381
  | Digit.Three, Digit.Nine, Digit.One => 391
  | Digit.Three, Digit.Nine, Digit.Seven => 397
  | Digit.Eight, Digit.Five, Digit.One => 851
  | Digit.Nine, Digit.Five, Digit.One => 951
  | Digit.Nine, Digit.Five, Digit.Seven => 957
  | Digit.Nine, Digit.Six, Digit.One => 961
  | Digit.Nine, Digit.Nine, Digit.One => 991
  | _, _, _ => 0  -- Default case, should not occur in our problem

/-- The set of possible route numbers -/
def possibleRouteNumbers : Set Nat :=
  {351, 354, 357, 361, 367, 381, 391, 397, 851, 951, 957, 961, 991}

/-- The theorem stating that the displayed number 351 with two non-working segments
    can only result in the numbers in the possibleRouteNumbers set -/
theorem route_number_theorem (n : ThreeDigitNumber) 
    (h : n.toNat ∈ possibleRouteNumbers) : 
    ∃ (broken_segments : Nat), broken_segments ≤ 2 ∧ 
    n.toNat ∈ possibleRouteNumbers :=
  sorry

end NUMINAMATH_CALUDE_route_number_theorem_l2662_266220


namespace NUMINAMATH_CALUDE_crayons_left_correct_l2662_266291

/-- Represents the number of crayons and erasers Paul has -/
structure PaulsCrayonsAndErasers where
  initial_crayons : ℕ
  initial_erasers : ℕ
  remaining_difference : ℕ

/-- Calculates the number of crayons Paul has left -/
def crayons_left (p : PaulsCrayonsAndErasers) : ℕ :=
  p.initial_erasers + p.remaining_difference

theorem crayons_left_correct (p : PaulsCrayonsAndErasers) 
  (h : p.initial_crayons = 531 ∧ p.initial_erasers = 38 ∧ p.remaining_difference = 353) : 
  crayons_left p = 391 := by
  sorry

end NUMINAMATH_CALUDE_crayons_left_correct_l2662_266291


namespace NUMINAMATH_CALUDE_ac_squared_gt_bc_squared_sufficient_not_necessary_l2662_266272

theorem ac_squared_gt_bc_squared_sufficient_not_necessary
  (a b c : ℝ) (h : c ≠ 0) :
  (∀ a b, a * c^2 > b * c^2 → a > b) ∧
  ¬(∀ a b, a > b → a * c^2 > b * c^2) :=
sorry

end NUMINAMATH_CALUDE_ac_squared_gt_bc_squared_sufficient_not_necessary_l2662_266272


namespace NUMINAMATH_CALUDE_apple_picking_ratio_l2662_266223

theorem apple_picking_ratio : 
  ∀ (frank_apples susan_apples : ℕ) (x : ℚ),
    frank_apples = 36 →
    susan_apples = frank_apples * x →
    (susan_apples / 2 + frank_apples * 2 / 3 : ℚ) = 78 →
    x = 3 := by
  sorry

end NUMINAMATH_CALUDE_apple_picking_ratio_l2662_266223


namespace NUMINAMATH_CALUDE_number_subtraction_problem_l2662_266211

theorem number_subtraction_problem :
  ∀ x : ℤ, x - 2 = 6 → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_number_subtraction_problem_l2662_266211


namespace NUMINAMATH_CALUDE_cost_of_horse_cost_of_horse_proof_l2662_266268

/-- The cost of a horse given the conditions of Albert's purchase and sale -/
theorem cost_of_horse : ℝ :=
  let total_cost : ℝ := 13400
  let total_profit : ℝ := 1880
  let num_horses : ℕ := 4
  let num_cows : ℕ := 9
  let horse_profit_rate : ℝ := 0.1
  let cow_profit_rate : ℝ := 0.2

  2000

theorem cost_of_horse_proof (total_cost : ℝ) (total_profit : ℝ) 
  (num_horses num_cows : ℕ) (horse_profit_rate cow_profit_rate : ℝ) :
  total_cost = 13400 →
  total_profit = 1880 →
  num_horses = 4 →
  num_cows = 9 →
  horse_profit_rate = 0.1 →
  cow_profit_rate = 0.2 →
  ∃ (horse_cost cow_cost : ℝ),
    num_horses * horse_cost + num_cows * cow_cost = total_cost ∧
    num_horses * horse_cost * horse_profit_rate + num_cows * cow_cost * cow_profit_rate = total_profit ∧
    horse_cost = 2000 :=
by
  sorry

end NUMINAMATH_CALUDE_cost_of_horse_cost_of_horse_proof_l2662_266268


namespace NUMINAMATH_CALUDE_matrix_power_difference_l2662_266249

def B : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; 0, 2]

theorem matrix_power_difference : 
  B^10 - 3 * B^9 = !![0, 4; 0, -1] := by sorry

end NUMINAMATH_CALUDE_matrix_power_difference_l2662_266249


namespace NUMINAMATH_CALUDE_new_R_value_l2662_266233

/-- A function that calculates R given g and S -/
def R (g : ℝ) (S : ℝ) : ℝ := g * S - 7

/-- The theorem stating the new value of R after S increases by 50% -/
theorem new_R_value (g : ℝ) (S : ℝ) (h1 : S = 5) (h2 : R g S = 8) :
  R g (S * 1.5) = 15.5 := by
  sorry


end NUMINAMATH_CALUDE_new_R_value_l2662_266233


namespace NUMINAMATH_CALUDE_jet_flight_time_l2662_266285

/-- Given a jet flying with and against wind, calculate the time taken with tail wind -/
theorem jet_flight_time (distance : ℝ) (return_time : ℝ) (wind_speed : ℝ) 
  (h1 : distance = 2000)
  (h2 : return_time = 5)
  (h3 : wind_speed = 50) : 
  ∃ (jet_speed : ℝ) (tail_wind_time : ℝ),
    (jet_speed + wind_speed) * tail_wind_time = distance ∧
    (jet_speed - wind_speed) * return_time = distance ∧
    tail_wind_time = 4 := by
  sorry

end NUMINAMATH_CALUDE_jet_flight_time_l2662_266285


namespace NUMINAMATH_CALUDE_point_in_region_l2662_266284

def point : ℝ × ℝ := (0, -2)

theorem point_in_region (x y : ℝ) (h : (x, y) = point) : 
  x + y - 1 < 0 ∧ x - y + 1 > 0 := by
  sorry

end NUMINAMATH_CALUDE_point_in_region_l2662_266284


namespace NUMINAMATH_CALUDE_family_fruit_consumption_l2662_266258

/-- Represents the number of fruits in a box for each type of fruit -/
structure FruitBox where
  apples : ℕ := 14
  bananas : ℕ := 20
  oranges : ℕ := 12

/-- Represents the daily consumption of fruits for each family member -/
structure DailyConsumption where
  apples : ℕ := 2  -- Henry and his brother combined
  bananas : ℕ := 2 -- Henry's sister (on odd days)
  oranges : ℕ := 3 -- Father

/-- Represents the number of boxes for each type of fruit -/
structure FruitSupply where
  appleBoxes : ℕ := 3
  bananaBoxes : ℕ := 4
  orangeBoxes : ℕ := 5

/-- Calculates the maximum number of days the family can eat their preferred fruits together -/
def max_days_eating_fruits (box : FruitBox) (consumption : DailyConsumption) (supply : FruitSupply) : ℕ :=
  sorry

/-- Theorem stating that the family can eat their preferred fruits together for 20 days -/
theorem family_fruit_consumption 
  (box : FruitBox) 
  (consumption : DailyConsumption) 
  (supply : FruitSupply) 
  (orange_days : ℕ := 20) -- Oranges are only available for 20 days
  (h1 : box.apples = 14)
  (h2 : box.bananas = 20)
  (h3 : box.oranges = 12)
  (h4 : consumption.apples = 2)
  (h5 : consumption.bananas = 2)
  (h6 : consumption.oranges = 3)
  (h7 : supply.appleBoxes = 3)
  (h8 : supply.bananaBoxes = 4)
  (h9 : supply.orangeBoxes = 5) :
  max_days_eating_fruits box consumption supply = 20 :=
sorry

end NUMINAMATH_CALUDE_family_fruit_consumption_l2662_266258


namespace NUMINAMATH_CALUDE_shortest_distance_to_circle_l2662_266292

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 8*x + y^2 + 6*y + 9 = 0

/-- The shortest distance from the origin to the circle -/
def shortest_distance : ℝ := 1

/-- Theorem: The shortest distance from the origin to the circle defined by
    x^2 - 8x + y^2 + 6y + 9 = 0 is equal to 1 -/
theorem shortest_distance_to_circle :
  ∀ (x y : ℝ), circle_equation x y →
  ∃ (p : ℝ × ℝ), p ∈ {(x, y) | circle_equation x y} ∧
  ∀ (q : ℝ × ℝ), q ∈ {(x, y) | circle_equation x y} →
  Real.sqrt ((p.1 - 0)^2 + (p.2 - 0)^2) ≤ Real.sqrt ((q.1 - 0)^2 + (q.2 - 0)^2) ∧
  Real.sqrt ((p.1 - 0)^2 + (p.2 - 0)^2) = shortest_distance :=
by sorry

end NUMINAMATH_CALUDE_shortest_distance_to_circle_l2662_266292


namespace NUMINAMATH_CALUDE_intersection_equals_closed_open_interval_l2662_266215

-- Define sets A and B
def A : Set ℝ := {x : ℝ | 2 ≤ x ∧ x < 4}
def B : Set ℝ := {x : ℝ | x ≥ 3}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := A ∩ B

-- Theorem statement
theorem intersection_equals_closed_open_interval :
  A_intersect_B = {x : ℝ | 3 ≤ x ∧ x < 4} := by sorry

end NUMINAMATH_CALUDE_intersection_equals_closed_open_interval_l2662_266215


namespace NUMINAMATH_CALUDE_inequality_proof_l2662_266271

theorem inequality_proof (a b : ℝ) (h1 : a + b < 0) (h2 : b > 0) : a^2 > -a*b ∧ -a*b > b^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2662_266271


namespace NUMINAMATH_CALUDE_g_sum_lower_bound_l2662_266293

noncomputable def f (x : ℝ) : ℝ := Real.exp x - (1/2) * x^2

noncomputable def g (x : ℝ) : ℝ := f x + 3 * x + 1

theorem g_sum_lower_bound (x₁ x₂ : ℝ) (h : x₁ + x₂ ≥ 0) :
  g x₁ + g x₂ ≥ 4 := by sorry

end NUMINAMATH_CALUDE_g_sum_lower_bound_l2662_266293


namespace NUMINAMATH_CALUDE_white_balls_count_l2662_266252

theorem white_balls_count (total : ℕ) (p_yellow : ℚ) (h_total : total = 32) (h_p_yellow : p_yellow = 1/4) :
  total - (total * p_yellow).floor = 24 :=
sorry

end NUMINAMATH_CALUDE_white_balls_count_l2662_266252


namespace NUMINAMATH_CALUDE_dormitory_to_city_distance_l2662_266239

theorem dormitory_to_city_distance : 
  ∀ (total_distance : ℝ),
    (1/5 : ℝ) * total_distance + (2/3 : ℝ) * total_distance + 4 = total_distance →
    total_distance = 30 := by
  sorry

end NUMINAMATH_CALUDE_dormitory_to_city_distance_l2662_266239


namespace NUMINAMATH_CALUDE_velocity_zero_at_two_l2662_266216

-- Define the motion equation
def s (t : ℝ) : ℝ := -4 * t^3 + 48 * t

-- Define the velocity function (derivative of s)
def v (t : ℝ) : ℝ := -12 * t^2 + 48

-- Theorem stating that the positive time when velocity is zero is 2
theorem velocity_zero_at_two :
  ∃ (t : ℝ), t > 0 ∧ v t = 0 ∧ t = 2 := by
  sorry

end NUMINAMATH_CALUDE_velocity_zero_at_two_l2662_266216


namespace NUMINAMATH_CALUDE_suitcase_theorem_l2662_266276

/-- Represents the suitcase scenario at the airport -/
structure SuitcaseScenario where
  total_suitcases : ℕ
  business_suitcases : ℕ
  placement_interval : ℕ

/-- The probability of businesspeople waiting exactly 2 minutes for their last suitcase -/
def exact_wait_probability (s : SuitcaseScenario) : ℚ :=
  (Nat.choose 59 9 : ℚ) / (Nat.choose s.total_suitcases s.business_suitcases)

/-- The expected waiting time for businesspeople's last suitcase in seconds -/
def expected_wait_time (s : SuitcaseScenario) : ℚ :=
  4020 / 11

/-- Theorem stating the probability and expected waiting time for the suitcase scenario -/
theorem suitcase_theorem (s : SuitcaseScenario) 
  (h1 : s.total_suitcases = 200)
  (h2 : s.business_suitcases = 10)
  (h3 : s.placement_interval = 2) :
  exact_wait_probability s = (Nat.choose 59 9 : ℚ) / (Nat.choose 200 10) ∧
  expected_wait_time s = 4020 / 11 := by
  sorry

#eval exact_wait_probability ⟨200, 10, 2⟩
#eval expected_wait_time ⟨200, 10, 2⟩

end NUMINAMATH_CALUDE_suitcase_theorem_l2662_266276


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l2662_266281

-- Expression 1
theorem simplify_expression_1 (a b : ℝ) :
  -2 * a^2 * b - 3 * a * b^2 + 3 * a^2 * b - 4 * a * b^2 = a^2 * b - 7 * a * b^2 := by
  sorry

-- Expression 2
theorem simplify_expression_2 (x y z : ℝ) :
  2 * (x * y * z - 3 * x) + 5 * (2 * x - 3 * x * y * z) = 4 * x - 13 * x * y * z := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l2662_266281


namespace NUMINAMATH_CALUDE_pure_imaginary_value_l2662_266277

theorem pure_imaginary_value (a : ℝ) : 
  (Complex.mk (a^2 - 4*a + 3) (a - 1)).im ≠ 0 ∧ 
  (Complex.mk (a^2 - 4*a + 3) (a - 1)).re = 0 → 
  a = 3 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_value_l2662_266277


namespace NUMINAMATH_CALUDE_binomial_expansion_special_case_l2662_266288

theorem binomial_expansion_special_case : 7^4 + 4*(7^3) + 6*(7^2) + 4*7 + 1 = 8^4 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_special_case_l2662_266288


namespace NUMINAMATH_CALUDE_simplify_and_sum_exponents_l2662_266263

theorem simplify_and_sum_exponents 
  (a b c : ℝ) 
  (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) : 
  ∃ (k : ℝ), 
    (k > 0) ∧ 
    (k^3 = a^2 * b^4 * c^2) ∧ 
    ((72 * a^5 * b^7 * c^14)^(1/3) = 2 * 3^(2/3) * a * b * c^4 * k) ∧
    (1 + 1 + 4 = 6) := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_sum_exponents_l2662_266263


namespace NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_divisibility_of_four_consecutive_integers_optimality_of_twelve_l2662_266296

/-- The greatest whole number that must be a divisor of the product of any four consecutive positive integers is 12. -/
theorem greatest_divisor_four_consecutive_integers : ℕ :=
  let f : ℕ → ℕ := λ n => n * (n + 1) * (n + 2) * (n + 3)
  12

theorem divisibility_of_four_consecutive_integers (n : ℕ) :
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

theorem optimality_of_twelve (m : ℕ) :
  (∀ n : ℕ, m ∣ (n * (n + 1) * (n + 2) * (n + 3))) → m ≤ 12 :=
sorry

end NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_divisibility_of_four_consecutive_integers_optimality_of_twelve_l2662_266296


namespace NUMINAMATH_CALUDE_incorrect_observation_value_l2662_266238

/-- Given a set of observations with known properties, calculate the incorrect value. -/
theorem incorrect_observation_value 
  (n : ℕ) 
  (original_mean : ℝ) 
  (new_mean : ℝ) 
  (correct_value : ℝ) 
  (h1 : n = 50) 
  (h2 : original_mean = 36) 
  (h3 : new_mean = 36.5) 
  (h4 : correct_value = 34) : 
  ∃ (incorrect_value : ℝ), 
    incorrect_value = n * new_mean - (n - 1) * original_mean - correct_value + n * (new_mean - original_mean) :=
by
  sorry

#check incorrect_observation_value

end NUMINAMATH_CALUDE_incorrect_observation_value_l2662_266238


namespace NUMINAMATH_CALUDE_triangle_area_from_perimeter_and_inradius_l2662_266232

/-- Theorem: Area of a triangle with given perimeter and inradius -/
theorem triangle_area_from_perimeter_and_inradius
  (perimeter : ℝ) (inradius : ℝ) (h_perimeter : perimeter = 39)
  (h_inradius : inradius = 1.5) :
  perimeter * inradius / 4 = 29.25 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_from_perimeter_and_inradius_l2662_266232


namespace NUMINAMATH_CALUDE_bird_count_difference_l2662_266209

/-- Represents the count of birds on a single day -/
structure DailyCount where
  bluejays : ℕ
  cardinals : ℕ

/-- Calculates the difference between cardinals and blue jays for a single day -/
def dailyDifference (count : DailyCount) : ℤ :=
  count.cardinals - count.bluejays

/-- Theorem: The total difference between cardinals and blue jays over three days is 3 -/
theorem bird_count_difference (day1 day2 day3 : DailyCount)
  (h1 : day1 = { bluejays := 2, cardinals := 3 })
  (h2 : day2 = { bluejays := 3, cardinals := 3 })
  (h3 : day3 = { bluejays := 2, cardinals := 4 }) :
  dailyDifference day1 + dailyDifference day2 + dailyDifference day3 = 3 := by
  sorry

#eval dailyDifference { bluejays := 2, cardinals := 3 } +
      dailyDifference { bluejays := 3, cardinals := 3 } +
      dailyDifference { bluejays := 2, cardinals := 4 }

end NUMINAMATH_CALUDE_bird_count_difference_l2662_266209


namespace NUMINAMATH_CALUDE_quadratic_function_proof_l2662_266213

theorem quadratic_function_proof :
  let f : ℝ → ℝ := λ x => x^2 - 2*x - 3
  ∀ a b c : ℝ, a ≠ 0 →
  (∀ x : ℝ, f x = a * x^2 + b * x + c) →
  f (-2) = 5 ∧ f (-1) = 0 ∧ f 0 = -3 ∧ f 1 = -4 ∧ f 2 = -3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_proof_l2662_266213


namespace NUMINAMATH_CALUDE_probability_closer_to_center_l2662_266244

/-- The probability of a randomly chosen point within a circle of radius 5 being closer to the center than to the boundary, given an inner concentric circle of radius 2 -/
theorem probability_closer_to_center (outer_radius inner_radius : ℝ) : 
  outer_radius = 5 → 
  inner_radius = 2 → 
  (π * inner_radius^2) / (π * outer_radius^2) = 4 / 25 := by
  sorry

end NUMINAMATH_CALUDE_probability_closer_to_center_l2662_266244
