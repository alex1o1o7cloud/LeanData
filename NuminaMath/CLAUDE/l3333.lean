import Mathlib

namespace NUMINAMATH_CALUDE_no_distributive_laws_hold_l3333_333329

-- Define the * operation
def star (a b : ℝ) : ℝ := a + b + a * b

-- Theorem statement
theorem no_distributive_laws_hold :
  ∃ x y z : ℝ,
    (star x (y + z) ≠ star (star x y) (star x z)) ∧
    (x + star y z ≠ star (x + y) (x + z)) ∧
    (star x (star y z) ≠ star (star x y) (star x z)) :=
by
  sorry

end NUMINAMATH_CALUDE_no_distributive_laws_hold_l3333_333329


namespace NUMINAMATH_CALUDE_magic_king_episodes_l3333_333342

/-- Calculates the total number of episodes for a TV show with a given number of seasons and episodes per season for each half. -/
def totalEpisodes (totalSeasons : ℕ) (episodesFirstHalf : ℕ) (episodesSecondHalf : ℕ) : ℕ :=
  let halfSeasons := totalSeasons / 2
  halfSeasons * episodesFirstHalf + halfSeasons * episodesSecondHalf

/-- Proves that a show with 10 seasons, 20 episodes per season for the first half, and 25 episodes per season for the second half has 225 total episodes. -/
theorem magic_king_episodes : totalEpisodes 10 20 25 = 225 := by
  sorry

end NUMINAMATH_CALUDE_magic_king_episodes_l3333_333342


namespace NUMINAMATH_CALUDE_a_86_in_geometric_subsequence_l3333_333372

/-- Represents an arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence :=
  (a : ℕ → ℝ)
  (d : ℝ)
  (h_nonzero : d ≠ 0)
  (h_arithmetic : ∀ n : ℕ, a (n + 1) = a n + d)

/-- Represents a subsequence of an arithmetic sequence that forms a geometric sequence -/
structure GeometricSubsequence (as : ArithmeticSequence) :=
  (k : ℕ → ℕ)
  (h_geometric : ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, as.a (k (n + 1)) = r * as.a (k n))
  (h_k1 : k 1 = 1)
  (h_k2 : k 2 = 2)
  (h_k3 : k 3 = 6)

/-- The main theorem stating that a_86 is in the geometric subsequence -/
theorem a_86_in_geometric_subsequence (as : ArithmeticSequence) (gs : GeometricSubsequence as) :
  ∃ n : ℕ, gs.k n = 86 :=
sorry

end NUMINAMATH_CALUDE_a_86_in_geometric_subsequence_l3333_333372


namespace NUMINAMATH_CALUDE_triangle_horizontal_line_l3333_333323

/-- Given two intersecting lines and the area of the triangle they form with the x-axis,
    prove the equation of the horizontal line that completes this triangle. -/
theorem triangle_horizontal_line
  (line1 : ℝ → ℝ)
  (line2 : ℝ)
  (area : ℝ)
  (h1 : ∀ x, line1 x = x)
  (h2 : line2 = -9)
  (h3 : area = 40.5)
  : ∃ y : ℝ, y = 9 ∧ 
    (1/2 : ℝ) * |line2| * y = area ∧
    (line1 (-line2) = y) :=
by sorry

end NUMINAMATH_CALUDE_triangle_horizontal_line_l3333_333323


namespace NUMINAMATH_CALUDE_modified_cube_surface_area_l3333_333356

/-- Represents a modified cube with square holes cut through each face -/
structure ModifiedCube where
  edge_length : ℝ
  hole_side_length : ℝ

/-- Calculates the total surface area of a modified cube including inside surfaces -/
def total_surface_area (cube : ModifiedCube) : ℝ :=
  let original_surface_area := 6 * cube.edge_length^2
  let hole_area := 6 * cube.hole_side_length^2
  let new_exposed_area := 6 * 4 * cube.hole_side_length^2
  original_surface_area - hole_area + new_exposed_area

/-- Theorem stating that a cube with edge length 4 and hole side length 2 has a total surface area of 168 -/
theorem modified_cube_surface_area :
  let cube : ModifiedCube := { edge_length := 4, hole_side_length := 2 }
  total_surface_area cube = 168 := by
  sorry

end NUMINAMATH_CALUDE_modified_cube_surface_area_l3333_333356


namespace NUMINAMATH_CALUDE_problem_solution_l3333_333388

def U : Set ℕ := {2, 3, 4, 5, 6}

def A : Set ℕ := {x ∈ U | x^2 - 6*x + 8 = 0}

def B : Set ℕ := {2, 5, 6}

theorem problem_solution : (U \ A) ∪ B = {2, 3, 5, 6} := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3333_333388


namespace NUMINAMATH_CALUDE_second_player_wins_12_and_11_l3333_333335

/-- Represents the state of the daisy game -/
inductive DaisyState
  | petals (n : Nat)

/-- Represents a move in the daisy game -/
inductive DaisyMove
  | remove_one
  | remove_two

/-- Defines a valid move in the daisy game -/
def valid_move (state : DaisyState) (move : DaisyMove) : Prop :=
  match state, move with
  | DaisyState.petals n, DaisyMove.remove_one => n ≥ 1
  | DaisyState.petals n, DaisyMove.remove_two => n ≥ 2

/-- Applies a move to the current state -/
def apply_move (state : DaisyState) (move : DaisyMove) : DaisyState :=
  match state, move with
  | DaisyState.petals n, DaisyMove.remove_one => DaisyState.petals (n - 1)
  | DaisyState.petals n, DaisyMove.remove_two => DaisyState.petals (n - 2)

/-- Defines a winning strategy for the second player -/
def second_player_wins (initial_petals : Nat) : Prop :=
  ∀ (first_move : DaisyMove),
    valid_move (DaisyState.petals initial_petals) first_move →
    ∃ (strategy : DaisyState → DaisyMove),
      (∀ (state : DaisyState), valid_move state (strategy state)) ∧
      (∀ (game : Nat → DaisyState),
        game 0 = apply_move (DaisyState.petals initial_petals) first_move →
        (∀ n, game (n + 1) = apply_move (game n) (strategy (game n))) →
        ∃ k, ¬∃ move, valid_move (game k) move)

/-- The main theorem stating that the second player wins for both 12 and 11 initial petals -/
theorem second_player_wins_12_and_11 :
  second_player_wins 12 ∧ second_player_wins 11 := by sorry

end NUMINAMATH_CALUDE_second_player_wins_12_and_11_l3333_333335


namespace NUMINAMATH_CALUDE_probability_three_girls_l3333_333322

/-- The probability of choosing 3 girls from a club with 15 members (8 girls and 7 boys) is 8/65 -/
theorem probability_three_girls (total : ℕ) (girls : ℕ) (boys : ℕ) (h1 : total = 15) (h2 : girls = 8) (h3 : boys = 7) (h4 : total = girls + boys) :
  (Nat.choose girls 3 : ℚ) / (Nat.choose total 3) = 8 / 65 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_girls_l3333_333322


namespace NUMINAMATH_CALUDE_perpendicular_lines_from_perpendicular_planes_l3333_333383

-- Define the basic types
variable (Point : Type) (Line : Type) (Plane : Type)

-- Define the relations
variable (distinct : Line → Line → Prop)
variable (distinct_plane : Plane → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_plane : Plane → Plane → Prop)
variable (perpendicular_line : Line → Line → Prop)

-- Theorem statement
theorem perpendicular_lines_from_perpendicular_planes
  (m n : Line) (α β : Plane)
  (h1 : distinct m n)
  (h2 : distinct_plane α β)
  (h3 : perpendicular_plane α β)
  (h4 : perpendicular_line_plane m α)
  (h5 : perpendicular_line_plane n β) :
  perpendicular_line m n :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_from_perpendicular_planes_l3333_333383


namespace NUMINAMATH_CALUDE_square_perimeter_l3333_333349

theorem square_perimeter (area : ℝ) (side : ℝ) (perimeter : ℝ) :
  area = 675 →
  side * side = area →
  perimeter = 4 * side →
  1.5 * perimeter = 90 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l3333_333349


namespace NUMINAMATH_CALUDE_power_six_sum_l3333_333316

theorem power_six_sum (x : ℝ) (h : x + 1/x = 5) : x^6 + 1/x^6 = 12098 := by
  sorry

end NUMINAMATH_CALUDE_power_six_sum_l3333_333316


namespace NUMINAMATH_CALUDE_sports_club_membership_l3333_333304

theorem sports_club_membership (total : ℕ) (badminton : ℕ) (tennis : ℕ) (both : ℕ) :
  total = 28 →
  badminton = 17 →
  tennis = 19 →
  both = 10 →
  total - (badminton + tennis - both) = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_sports_club_membership_l3333_333304


namespace NUMINAMATH_CALUDE_probability_product_24_l3333_333358

def is_valid_die_roll (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 6

def product_equals_24 (a b c d : ℕ) : Prop :=
  is_valid_die_roll a ∧ is_valid_die_roll b ∧ is_valid_die_roll c ∧ is_valid_die_roll d ∧ a * b * c * d = 24

def count_valid_permutations : ℕ := 36

def total_outcomes : ℕ := 6^4

theorem probability_product_24 :
  (count_valid_permutations : ℚ) / total_outcomes = 1 / 36 :=
sorry

end NUMINAMATH_CALUDE_probability_product_24_l3333_333358


namespace NUMINAMATH_CALUDE_surfer_wave_height_l3333_333301

/-- Represents the height of the highest wave caught by a surfer. -/
def highest_wave (H : ℝ) : ℝ := 4 * H + 2

/-- Represents the height of the shortest wave caught by a surfer. -/
def shortest_wave (H : ℝ) : ℝ := H + 4

theorem surfer_wave_height (H : ℝ) 
  (h1 : shortest_wave H = 7 + 3) 
  (h2 : shortest_wave H = H + 4) : 
  highest_wave H = 26 := by
  sorry

end NUMINAMATH_CALUDE_surfer_wave_height_l3333_333301


namespace NUMINAMATH_CALUDE_min_perimeter_rectangle_l3333_333346

theorem min_perimeter_rectangle (area : Real) (perimeter : Real) : 
  area = 64 → perimeter ≥ 32 := by
  sorry

end NUMINAMATH_CALUDE_min_perimeter_rectangle_l3333_333346


namespace NUMINAMATH_CALUDE_fraction_equality_l3333_333361

theorem fraction_equality (a b c d e f : ℚ) 
  (h1 : a / b = 1 / 3) 
  (h2 : c / d = 1 / 3) 
  (h3 : e / f = 1 / 3) : 
  (3 * a - 2 * c + e) / (3 * b - 2 * d + f) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3333_333361


namespace NUMINAMATH_CALUDE_projection_property_l3333_333363

/-- A projection that takes (3, -3) to (75/26, -15/26) -/
def projection (v : ℝ × ℝ) : ℝ × ℝ :=
  sorry

theorem projection_property :
  projection (3, -3) = (75/26, -15/26) →
  projection ((5, 7) + (-3, -4)) = (35/26, -7/26) :=
by
  sorry

end NUMINAMATH_CALUDE_projection_property_l3333_333363


namespace NUMINAMATH_CALUDE_total_new_people_count_l3333_333376

/-- The number of people born in the country last year -/
def people_born : ℕ := 90171

/-- The number of people who immigrated to the country last year -/
def people_immigrated : ℕ := 16320

/-- The total number of new people in the country last year -/
def total_new_people : ℕ := people_born + people_immigrated

/-- Theorem stating that the total number of new people is 106491 -/
theorem total_new_people_count : total_new_people = 106491 := by
  sorry

end NUMINAMATH_CALUDE_total_new_people_count_l3333_333376


namespace NUMINAMATH_CALUDE_max_value_of_linear_combination_l3333_333373

theorem max_value_of_linear_combination (x y : ℝ) :
  x^2 + y^2 = 18*x + 8*y + 10 →
  4*x + 3*y ≤ 74 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_linear_combination_l3333_333373


namespace NUMINAMATH_CALUDE_fence_painting_rate_l3333_333364

theorem fence_painting_rate (num_fences : ℕ) (fence_length : ℕ) (total_earnings : ℚ) :
  num_fences = 50 →
  fence_length = 500 →
  total_earnings = 5000 →
  total_earnings / (num_fences * fence_length : ℚ) = 0.20 := by
  sorry

end NUMINAMATH_CALUDE_fence_painting_rate_l3333_333364


namespace NUMINAMATH_CALUDE_ant_population_growth_l3333_333359

/-- Represents the number of days passed -/
def days : ℕ := 4

/-- The growth factor for Species C per day -/
def growth_factor_C : ℕ := 2

/-- The growth factor for Species D per day -/
def growth_factor_D : ℕ := 4

/-- The total number of ants on Day 0 -/
def total_ants_day0 : ℕ := 35

/-- The total number of ants on Day 4 -/
def total_ants_day4 : ℕ := 3633

/-- The number of Species C ants on Day 0 -/
def species_C_day0 : ℕ := 22

/-- The number of Species D ants on Day 0 -/
def species_D_day0 : ℕ := 13

theorem ant_population_growth :
  species_C_day0 * growth_factor_C ^ days = 352 ∧
  species_C_day0 + species_D_day0 = total_ants_day0 ∧
  species_C_day0 * growth_factor_C ^ days + species_D_day0 * growth_factor_D ^ days = total_ants_day4 :=
by sorry

end NUMINAMATH_CALUDE_ant_population_growth_l3333_333359


namespace NUMINAMATH_CALUDE_circle_center_from_axis_intersections_l3333_333390

/-- Given a circle that intersects the x-axis at (a, 0) and (b, 0),
    and the y-axis at (0, c) and (0, d), its center is at ((a+b)/2, (c+d)/2) -/
theorem circle_center_from_axis_intersections 
  (a b c d : ℝ) : 
  ∃ (center : ℝ × ℝ),
    (∃ (circle : Set (ℝ × ℝ)), 
      (a, 0) ∈ circle ∧ 
      (b, 0) ∈ circle ∧ 
      (0, c) ∈ circle ∧ 
      (0, d) ∈ circle ∧
      center = ((a + b) / 2, (c + d) / 2) ∧
      ∀ p ∈ circle, (p.1 - center.1)^2 + (p.2 - center.2)^2 = 
        (a - center.1)^2 + (0 - center.2)^2) :=
by
  sorry

end NUMINAMATH_CALUDE_circle_center_from_axis_intersections_l3333_333390


namespace NUMINAMATH_CALUDE_cos_eighteen_degrees_l3333_333314

theorem cos_eighteen_degrees : Real.cos (18 * π / 180) = (5 + Real.sqrt 5) / 8 := by
  sorry

end NUMINAMATH_CALUDE_cos_eighteen_degrees_l3333_333314


namespace NUMINAMATH_CALUDE_four_people_seven_steps_l3333_333315

/-- The number of ways to arrange n people on m steps with at most k people per step -/
def arrangements (n m k : ℕ) : ℕ := sorry

/-- The number of ways 4 people can stand on 7 steps with at most 3 people per step -/
theorem four_people_seven_steps : arrangements 4 7 3 = 2394 := by sorry

end NUMINAMATH_CALUDE_four_people_seven_steps_l3333_333315


namespace NUMINAMATH_CALUDE_isosceles_triangle_angle_l3333_333370

/-- 
Given an isosceles triangle with inscribed circle radius r and circumscribed circle radius R,
if r/R = k, then the angle at the base of the triangle is arccos(k).
-/
theorem isosceles_triangle_angle (r R k : ℝ) (α : ℝ) : 
  r > 0 → R > 0 → k > 0 → k < 1 →  -- Ensure valid inputs
  (r / R = k) →                    -- Given ratio condition
  (α = Real.arccos k) →            -- Angle at the base
  True :=                          -- Placeholder for the theorem
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_angle_l3333_333370


namespace NUMINAMATH_CALUDE_three_digit_odd_count_l3333_333340

theorem three_digit_odd_count : 
  (Finset.filter 
    (fun n => n ≥ 100 ∧ n < 1000 ∧ n % 2 = 1) 
    (Finset.range 1000)).card = 450 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_odd_count_l3333_333340


namespace NUMINAMATH_CALUDE_total_missed_pitches_example_l3333_333399

/-- Represents a person's batting performance -/
structure BattingPerformance where
  tokens : Nat
  hits : Nat

/-- Calculates the total number of missed pitches for all players -/
def totalMissedPitches (pitchesPerToken : Nat) (performances : List BattingPerformance) : Nat :=
  performances.foldl (fun acc p => acc + p.tokens * pitchesPerToken - p.hits) 0

theorem total_missed_pitches_example :
  let pitchesPerToken := 15
  let macy := BattingPerformance.mk 11 50
  let piper := BattingPerformance.mk 17 55
  let quinn := BattingPerformance.mk 13 60
  let performances := [macy, piper, quinn]
  totalMissedPitches pitchesPerToken performances = 450 := by
  sorry

#eval totalMissedPitches 15 [BattingPerformance.mk 11 50, BattingPerformance.mk 17 55, BattingPerformance.mk 13 60]

end NUMINAMATH_CALUDE_total_missed_pitches_example_l3333_333399


namespace NUMINAMATH_CALUDE_largest_special_square_l3333_333392

def is_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def last_two_digits (n : ℕ) : ℕ := n % 100

def remove_last_two_digits (n : ℕ) : ℕ := (n - last_two_digits n) / 100

theorem largest_special_square : 
  ∀ n : ℕ, 
    (is_square n ∧ 
     n % 100 ≠ 0 ∧ 
     is_square (remove_last_two_digits n)) →
    n ≤ 1681 :=
sorry

end NUMINAMATH_CALUDE_largest_special_square_l3333_333392


namespace NUMINAMATH_CALUDE_factorization_x4_minus_y4_l3333_333369

theorem factorization_x4_minus_y4 (x y : ℝ) : x^4 - y^4 = (x^2 + y^2) * (x^2 - y^2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_x4_minus_y4_l3333_333369


namespace NUMINAMATH_CALUDE_grandma_last_birthday_age_l3333_333343

/-- Represents Grandma's age in various units -/
structure GrandmaAge where
  years : Nat
  months : Nat
  weeks : Nat
  days : Nat

/-- Calculates Grandma's age on her last birthday given her current age -/
def lastBirthdayAge (age : GrandmaAge) : Nat :=
  age.years + (age.months / 12) + 1

/-- Theorem stating that Grandma's age on her last birthday was 65 years -/
theorem grandma_last_birthday_age :
  let currentAge : GrandmaAge := { years := 60, months := 50, weeks := 40, days := 30 }
  lastBirthdayAge currentAge = 65 := by
  sorry

#eval lastBirthdayAge { years := 60, months := 50, weeks := 40, days := 30 }

end NUMINAMATH_CALUDE_grandma_last_birthday_age_l3333_333343


namespace NUMINAMATH_CALUDE_cookie_ratio_l3333_333386

theorem cookie_ratio (monday_cookies : ℕ) (total_cookies : ℕ) 
  (h1 : monday_cookies = 32)
  (h2 : total_cookies = 92)
  (h3 : ∃ f : ℚ, 
    monday_cookies + f * monday_cookies + (3 * f * monday_cookies - 4) = total_cookies) :
  ∃ f : ℚ, f = 1/2 ∧ 
    monday_cookies + f * monday_cookies + (3 * f * monday_cookies - 4) = total_cookies :=
by
  sorry

end NUMINAMATH_CALUDE_cookie_ratio_l3333_333386


namespace NUMINAMATH_CALUDE_amazon_pack_price_is_correct_l3333_333375

/-- The cost of a single lighter at the gas station in dollars -/
def gas_station_price : ℚ := 1.75

/-- The number of lighters Amanda wants to buy -/
def num_lighters : ℕ := 24

/-- The amount Amanda saves by buying online in dollars -/
def savings : ℚ := 32

/-- The cost of a pack of twelve lighters on Amazon in dollars -/
def amazon_pack_price : ℚ := 5

theorem amazon_pack_price_is_correct :
  amazon_pack_price = 5 ∧
  2 * amazon_pack_price = num_lighters * gas_station_price - savings :=
by sorry

end NUMINAMATH_CALUDE_amazon_pack_price_is_correct_l3333_333375


namespace NUMINAMATH_CALUDE_factors_of_96_with_square_sum_208_l3333_333348

theorem factors_of_96_with_square_sum_208 : 
  ∃ (a b : ℕ+), (a * b = 96) ∧ (a^2 + b^2 = 208) := by sorry

end NUMINAMATH_CALUDE_factors_of_96_with_square_sum_208_l3333_333348


namespace NUMINAMATH_CALUDE_chessboard_cannot_be_tiled_chessboard_with_corner_removed_cannot_be_tiled_l3333_333332

/-- Represents a chessboard -/
structure Chessboard where
  rows : Nat
  cols : Nat

/-- Represents a triomino -/
structure Triomino where
  length : Nat
  width : Nat

/-- Function to check if a chessboard can be tiled with triominoes -/
def canBeTiled (board : Chessboard) (triomino : Triomino) : Prop :=
  (board.rows * board.cols) % (triomino.length * triomino.width) = 0

/-- Function to check if a chessboard with one corner removed can be tiled with triominoes -/
def canBeTiledWithCornerRemoved (board : Chessboard) (triomino : Triomino) : Prop :=
  ∃ (colorA colorB colorC : Nat),
    colorA + colorB + colorC = board.rows * board.cols - 1 ∧
    colorA = colorB ∧ colorB = colorC

/-- Theorem: An 8x8 chessboard cannot be tiled with 3x1 triominoes -/
theorem chessboard_cannot_be_tiled :
  ¬ canBeTiled ⟨8, 8⟩ ⟨3, 1⟩ :=
sorry

/-- Theorem: An 8x8 chessboard with one corner removed cannot be tiled with 3x1 triominoes -/
theorem chessboard_with_corner_removed_cannot_be_tiled :
  ¬ canBeTiledWithCornerRemoved ⟨8, 8⟩ ⟨3, 1⟩ :=
sorry

end NUMINAMATH_CALUDE_chessboard_cannot_be_tiled_chessboard_with_corner_removed_cannot_be_tiled_l3333_333332


namespace NUMINAMATH_CALUDE_order_of_logarithmic_fractions_l3333_333326

theorem order_of_logarithmic_fractions :
  let a := (Real.log 2) / 2
  let b := (Real.log 3) / 3
  let c := 1 / Real.exp 1
  a < b ∧ b < c := by sorry

end NUMINAMATH_CALUDE_order_of_logarithmic_fractions_l3333_333326


namespace NUMINAMATH_CALUDE_cos_two_theta_value_l3333_333311

theorem cos_two_theta_value (θ : Real) 
  (h : Real.sin (θ / 2) - Real.cos (θ / 2) = Real.sqrt 6 / 3) : 
  Real.cos (2 * θ) = 7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_cos_two_theta_value_l3333_333311


namespace NUMINAMATH_CALUDE_koala_fiber_consumption_l3333_333355

/-- The amount of fiber a koala absorbs as a percentage of what it eats -/
def koala_absorption_rate : ℝ := 0.30

/-- The amount of fiber absorbed by the koala in one day (in ounces) -/
def fiber_absorbed : ℝ := 12

/-- Theorem: If a koala absorbs 30% of the fiber it eats and it absorbed 12 ounces of fiber in one day,
    then the total amount of fiber the koala ate that day was 40 ounces. -/
theorem koala_fiber_consumption :
  fiber_absorbed = koala_absorption_rate * 40 := by
  sorry

end NUMINAMATH_CALUDE_koala_fiber_consumption_l3333_333355


namespace NUMINAMATH_CALUDE_coefficient_x5_expansion_l3333_333394

/-- The coefficient of x^5 in the expansion of (2 + √x - x^2018/2017)^12 -/
def coefficient_x5 : ℕ :=
  -- Define the coefficient here
  264

/-- Theorem stating that the coefficient of x^5 in the expansion of (2 + √x - x^2018/2017)^12 is 264 -/
theorem coefficient_x5_expansion :
  coefficient_x5 = 264 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x5_expansion_l3333_333394


namespace NUMINAMATH_CALUDE_reflected_ray_is_correct_l3333_333338

/-- The line of reflection --/
def reflection_line : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 + p.2 = -1}

/-- Point P --/
def P : ℝ × ℝ := (1, 1)

/-- Point Q --/
def Q : ℝ × ℝ := (2, 3)

/-- The reflected ray --/
def reflected_ray : Set (ℝ × ℝ) := {p : ℝ × ℝ | 5 * p.1 - 4 * p.2 + 2 = 0}

/-- Theorem stating that the reflected ray is correct --/
theorem reflected_ray_is_correct : 
  ∃ (M : ℝ × ℝ), 
    (M ∈ reflected_ray) ∧ 
    (Q ∈ reflected_ray) ∧
    (∀ (X : ℝ × ℝ), X ∈ reflection_line → (X.1 - P.1) * (X.1 - M.1) + (X.2 - P.2) * (X.2 - M.2) = 0) :=
sorry

end NUMINAMATH_CALUDE_reflected_ray_is_correct_l3333_333338


namespace NUMINAMATH_CALUDE_profit_percentage_calculation_l3333_333330

theorem profit_percentage_calculation (selling_price : ℝ) (cost_price : ℝ) 
  (h : cost_price = 0.95 * selling_price) :
  (selling_price - cost_price) / cost_price * 100 = (1 / 0.95 - 1) * 100 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_calculation_l3333_333330


namespace NUMINAMATH_CALUDE_point_conversion_value_l3333_333317

/-- Calculates the value of each point conversion in James' football season. -/
theorem point_conversion_value
  (touchdowns_per_game : ℕ)
  (points_per_touchdown : ℕ)
  (num_games : ℕ)
  (num_conversions : ℕ)
  (old_record : ℕ)
  (points_above_record : ℕ)
  (h1 : touchdowns_per_game = 4)
  (h2 : points_per_touchdown = 6)
  (h3 : num_games = 15)
  (h4 : num_conversions = 6)
  (h5 : old_record = 300)
  (h6 : points_above_record = 72) :
  (old_record + points_above_record - touchdowns_per_game * points_per_touchdown * num_games) / num_conversions = 2 := by
  sorry

end NUMINAMATH_CALUDE_point_conversion_value_l3333_333317


namespace NUMINAMATH_CALUDE_chocolate_ticket_value_l3333_333380

/-- Represents the value of a chocolate box ticket in terms of the box's cost -/
def ticket_value : ℚ := 1 / 9

/-- Represents the number of tickets needed to get a free box -/
def tickets_for_free_box : ℕ := 10

/-- Theorem stating the value of a single ticket -/
theorem chocolate_ticket_value :
  ticket_value = 1 / (tickets_for_free_box - 1) :=
by sorry

end NUMINAMATH_CALUDE_chocolate_ticket_value_l3333_333380


namespace NUMINAMATH_CALUDE_choir_members_count_l3333_333345

theorem choir_members_count : ∃! n : ℕ, 
  200 ≤ n ∧ n ≤ 300 ∧ 
  (n + 4) % 10 = 0 ∧ 
  (n + 5) % 11 = 0 ∧ 
  n = 226 := by sorry

end NUMINAMATH_CALUDE_choir_members_count_l3333_333345


namespace NUMINAMATH_CALUDE_sod_area_second_section_l3333_333389

/-- Given the total area of sod needed and the area of the first section,
    prove that the area of the second section is 4800 square feet. -/
theorem sod_area_second_section
  (total_sod_squares : ℕ)
  (sod_square_size : ℕ)
  (first_section_length : ℕ)
  (first_section_width : ℕ)
  (h1 : total_sod_squares = 1500)
  (h2 : sod_square_size = 4)
  (h3 : first_section_length = 30)
  (h4 : first_section_width = 40) :
  total_sod_squares * sod_square_size - first_section_length * first_section_width = 4800 :=
by sorry

end NUMINAMATH_CALUDE_sod_area_second_section_l3333_333389


namespace NUMINAMATH_CALUDE_determinant_specific_matrix_l3333_333387

theorem determinant_specific_matrix :
  let matrix : Matrix (Fin 2) (Fin 2) ℝ := !![3, Real.sin (π / 6); 5, Real.cos (π / 3)]
  Matrix.det matrix = -1 := by
  sorry

end NUMINAMATH_CALUDE_determinant_specific_matrix_l3333_333387


namespace NUMINAMATH_CALUDE_horner_rule_operations_l3333_333397

/-- Horner's Rule evaluation steps for a polynomial -/
def hornerSteps (coeffs : List ℤ) (x : ℤ) : List ℤ :=
  match coeffs with
  | [] => []
  | a :: as => List.scanl (fun acc b => acc * x + b) a as

/-- Number of multiplications in Horner's Rule -/
def numMultiplications (coeffs : List ℤ) : ℕ :=
  match coeffs with
  | [] => 0
  | [_] => 0
  | _ :: _ => coeffs.length - 1

/-- Number of additions in Horner's Rule -/
def numAdditions (coeffs : List ℤ) : ℕ :=
  match coeffs with
  | [] => 0
  | [_] => 0
  | _ :: _ => coeffs.length - 1

/-- The polynomial f(x) = 5x^6 + 4x^5 + x^4 + 3x^3 - 81x^2 + 9x - 1 -/
def f : List ℤ := [5, 4, 1, 3, -81, 9, -1]

theorem horner_rule_operations :
  numMultiplications f = 6 ∧ numAdditions f = 6 :=
sorry

end NUMINAMATH_CALUDE_horner_rule_operations_l3333_333397


namespace NUMINAMATH_CALUDE_megan_initial_cupcakes_l3333_333365

/-- The number of cupcakes Todd ate -/
def todd_ate : ℕ := 43

/-- The number of packages Megan could make with the remaining cupcakes -/
def num_packages : ℕ := 4

/-- The number of cupcakes in each package -/
def cupcakes_per_package : ℕ := 7

/-- The initial number of cupcakes Megan baked -/
def initial_cupcakes : ℕ := todd_ate + num_packages * cupcakes_per_package

theorem megan_initial_cupcakes : initial_cupcakes = 71 := by
  sorry

end NUMINAMATH_CALUDE_megan_initial_cupcakes_l3333_333365


namespace NUMINAMATH_CALUDE_sum_of_min_max_x_l3333_333396

theorem sum_of_min_max_x (x y z : ℝ) (h1 : x + y + z = 5) (h2 : x^2 + y^2 + z^2 = 11) :
  ∃ (m M : ℝ), (∀ x' y' z' : ℝ, x' + y' + z' = 5 → x'^2 + y'^2 + z'^2 = 11 → m ≤ x' ∧ x' ≤ M) ∧
                m + M = 8/3 :=
sorry

end NUMINAMATH_CALUDE_sum_of_min_max_x_l3333_333396


namespace NUMINAMATH_CALUDE_amoeba_count_after_10_days_l3333_333379

/-- The number of amoebas in the puddle after a given number of days -/
def amoeba_count (days : ℕ) : ℕ :=
  3^days

/-- Theorem stating that after 10 days, there will be 59049 amoebas in the puddle -/
theorem amoeba_count_after_10_days : amoeba_count 10 = 59049 := by
  sorry

end NUMINAMATH_CALUDE_amoeba_count_after_10_days_l3333_333379


namespace NUMINAMATH_CALUDE_x_equals_3_sufficient_not_necessary_for_x_squared_9_l3333_333377

theorem x_equals_3_sufficient_not_necessary_for_x_squared_9 :
  (∀ x : ℝ, x = 3 → x^2 = 9) ∧
  (∃ x : ℝ, x^2 = 9 ∧ x ≠ 3) :=
by sorry

end NUMINAMATH_CALUDE_x_equals_3_sufficient_not_necessary_for_x_squared_9_l3333_333377


namespace NUMINAMATH_CALUDE_jordan_fish_count_l3333_333353

theorem jordan_fish_count (jordan perry : ℕ) : 
  perry = 2 * jordan → 
  (3 * (jordan + perry)) / 4 = 9 → 
  jordan = 4 := by
sorry

end NUMINAMATH_CALUDE_jordan_fish_count_l3333_333353


namespace NUMINAMATH_CALUDE_remainder_sum_l3333_333324

theorem remainder_sum (c d : ℤ) (hc : c % 100 = 86) (hd : d % 150 = 144) :
  (c + d) % 50 = 30 := by sorry

end NUMINAMATH_CALUDE_remainder_sum_l3333_333324


namespace NUMINAMATH_CALUDE_perimeter_after_adding_tiles_l3333_333350

/-- Represents a square tile with unit length sides -/
structure UnitTile where
  x : ℕ
  y : ℕ

/-- Represents a figure made of unit tiles -/
structure TileFigure where
  tiles : List UnitTile

/-- Calculates the perimeter of a figure made of unit tiles -/
def perimeter (figure : TileFigure) : ℕ :=
  sorry

/-- Checks if a tile is adjacent to any tile in the figure -/
def isAdjacent (tile : UnitTile) (figure : TileFigure) : Bool :=
  sorry

theorem perimeter_after_adding_tiles :
  ∃ (original : TileFigure) (new1 new2 : UnitTile),
    (original.tiles.length = 16) ∧
    (∀ t ∈ original.tiles, t.x < 4 ∧ t.y < 4) ∧
    (isAdjacent new1 original) ∧
    (isAdjacent new2 original) ∧
    (perimeter (TileFigure.mk (new1 :: new2 :: original.tiles)) = 18) :=
  sorry

end NUMINAMATH_CALUDE_perimeter_after_adding_tiles_l3333_333350


namespace NUMINAMATH_CALUDE_perpendicular_line_correct_l3333_333300

/-- A line in polar coordinates passing through (2, 0) and perpendicular to the polar axis --/
def perpendicular_line (ρ θ : ℝ) : Prop :=
  ρ * Real.cos θ = 2

theorem perpendicular_line_correct :
  ∀ ρ θ : ℝ, perpendicular_line ρ θ ↔ 
    (ρ * Real.cos θ = 2 ∧ ρ * Real.sin θ = 0) ∨
    (ρ = 2 ∧ θ = 0) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_correct_l3333_333300


namespace NUMINAMATH_CALUDE_train_length_l3333_333325

/-- The length of a train given its speed, the speed of a person running in the opposite direction, and the time it takes for the train to pass the person. -/
theorem train_length (train_speed : ℝ) (man_speed : ℝ) (passing_time : ℝ) :
  train_speed = 65 →
  man_speed = 7 →
  passing_time = 5.4995600351971845 →
  ∃ (train_length : ℝ), abs (train_length - 110) < 0.5 := by
  sorry


end NUMINAMATH_CALUDE_train_length_l3333_333325


namespace NUMINAMATH_CALUDE_lily_of_valley_price_increase_l3333_333398

/-- Calculates the percentage increase in selling price compared to buying price for Françoise's lily of the valley pots. -/
theorem lily_of_valley_price_increase 
  (buying_price : ℝ) 
  (num_pots : ℕ) 
  (amount_given_back : ℝ) 
  (h1 : buying_price = 12)
  (h2 : num_pots = 150)
  (h3 : amount_given_back = 450) :
  let total_cost := buying_price * num_pots
  let total_revenue := total_cost + amount_given_back
  let selling_price := total_revenue / num_pots
  (selling_price - buying_price) / buying_price * 100 = 25 := by
sorry


end NUMINAMATH_CALUDE_lily_of_valley_price_increase_l3333_333398


namespace NUMINAMATH_CALUDE_sequence_general_term_l3333_333331

theorem sequence_general_term (a : ℕ → ℝ) :
  (∀ n > 0, a (n - 1)^2 = a n^2 + 4) →
  a 1 = 1 →
  (∀ n > 0, a n > 0) →
  ∀ n > 0, a n = Real.sqrt (4 * n - 3) := by
  sorry

end NUMINAMATH_CALUDE_sequence_general_term_l3333_333331


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l3333_333374

-- Define the hyperbola
def hyperbola (x y a b : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Define the focal length and real axis relationship
def focal_length_relation (a : ℝ) : Prop := 2 * (2 * a) = 4 * a

-- Define the asymptote equation
def asymptote_equation (x y : ℝ) : Prop := y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x

-- Theorem statement
theorem hyperbola_asymptotes (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∀ x y, hyperbola x y a b) →
  focal_length_relation a →
  (∀ x y, asymptote_equation x y) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l3333_333374


namespace NUMINAMATH_CALUDE_yard_length_26_trees_l3333_333320

/-- The length of a yard with equally spaced trees -/
def yard_length (num_trees : ℕ) (distance_between : ℕ) : ℕ :=
  (num_trees - 1) * distance_between

/-- Theorem: The length of a yard with 26 trees planted at equal distances,
    with one tree at each end and 16 meters between consecutive trees, is 400 meters. -/
theorem yard_length_26_trees : yard_length 26 16 = 400 := by
  sorry

end NUMINAMATH_CALUDE_yard_length_26_trees_l3333_333320


namespace NUMINAMATH_CALUDE_unattainable_value_l3333_333385

theorem unattainable_value (x : ℝ) (hx : x ≠ -4/3) :
  ¬∃y, y = (2 - x) / (3 * x + 4) ∧ y = -1/3 := by
sorry

end NUMINAMATH_CALUDE_unattainable_value_l3333_333385


namespace NUMINAMATH_CALUDE_sum_of_cubes_l3333_333393

theorem sum_of_cubes (a b c d e : ℕ) : 
  a ∈ ({0, 1, 2} : Set ℕ) → 
  b ∈ ({0, 1, 2} : Set ℕ) → 
  c ∈ ({0, 1, 2} : Set ℕ) → 
  d ∈ ({0, 1, 2} : Set ℕ) → 
  e ∈ ({0, 1, 2} : Set ℕ) → 
  a + b + c + d + e = 6 → 
  a^2 + b^2 + c^2 + d^2 + e^2 = 10 → 
  a^3 + b^3 + c^3 + d^3 + e^3 = 18 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l3333_333393


namespace NUMINAMATH_CALUDE_f_neg_two_equals_six_l3333_333381

/-- The quadratic function f(x) -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * b * x + c

/-- The quadratic function g(x) -/
def g (a b c : ℝ) (x : ℝ) : ℝ := (a + 1) * x^2 + 2 * (b + 2) * x + (c + 4)

/-- The discriminant of a quadratic function ax^2 + bx + c -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem f_neg_two_equals_six (a b c : ℝ) :
  discriminant (a + 1) (b + 2) (c + 4) - discriminant a b c = 24 →
  f a b c (-2) = 6 := by
  sorry

#eval f 1 2 3 (-2)  -- Example usage

end NUMINAMATH_CALUDE_f_neg_two_equals_six_l3333_333381


namespace NUMINAMATH_CALUDE_min_product_of_three_l3333_333391

def S : Set Int := {-10, -7, -3, 0, 4, 6, 9}

theorem min_product_of_three (a b c : Int) (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  ∃ (x y z : Int), x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
  x * y * z = -540 ∧ 
  ∀ (p q r : Int), p ∈ S → q ∈ S → r ∈ S → p ≠ q → q ≠ r → p ≠ r → 
  p * q * r ≥ -540 :=
sorry

end NUMINAMATH_CALUDE_min_product_of_three_l3333_333391


namespace NUMINAMATH_CALUDE_greatest_x_value_l3333_333328

theorem greatest_x_value : 
  (∃ (x : ℤ), ∀ (y : ℤ), (2.13 * (10 : ℝ)^(y : ℝ) < 2100) → y ≤ x) ∧ 
  (2.13 * (10 : ℝ)^(2 : ℝ) < 2100) ∧ 
  (∀ (z : ℤ), z > 2 → 2.13 * (10 : ℝ)^(z : ℝ) ≥ 2100) :=
by sorry

end NUMINAMATH_CALUDE_greatest_x_value_l3333_333328


namespace NUMINAMATH_CALUDE_orange_boxes_l3333_333341

theorem orange_boxes (total_oranges : ℕ) (oranges_per_box : ℕ) (h1 : total_oranges = 35) (h2 : oranges_per_box = 5) :
  total_oranges / oranges_per_box = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_orange_boxes_l3333_333341


namespace NUMINAMATH_CALUDE_point_on_transformed_graph_l3333_333384

/-- Given a function g : ℝ → ℝ such that g(8) = 5, prove that (8/3, 14/9) is on the graph of 3y = g(3x)/3 + 3 and the sum of its coordinates is 38/9 -/
theorem point_on_transformed_graph (g : ℝ → ℝ) (h : g 8 = 5) :
  let f : ℝ → ℝ := λ x => (g (3 * x) / 3 + 3) / 3
  f (8/3) = 14/9 ∧ 8/3 + 14/9 = 38/9 := by
sorry

end NUMINAMATH_CALUDE_point_on_transformed_graph_l3333_333384


namespace NUMINAMATH_CALUDE_cubic_roots_sum_cubes_l3333_333307

theorem cubic_roots_sum_cubes (α β γ : ℂ) : 
  (8 * α^3 + 2012 * α + 2013 = 0) →
  (8 * β^3 + 2012 * β + 2013 = 0) →
  (8 * γ^3 + 2012 * γ + 2013 = 0) →
  (α + β)^3 + (β + γ)^3 + (γ + α)^3 = 6039 / 8 := by
  sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_cubes_l3333_333307


namespace NUMINAMATH_CALUDE_positive_integers_equality_l3333_333306

theorem positive_integers_equality (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  (4 * a * b - 1) ∣ ((4 * a^2 - 1)^2) → a = b := by
  sorry

end NUMINAMATH_CALUDE_positive_integers_equality_l3333_333306


namespace NUMINAMATH_CALUDE_rational_function_zeros_l3333_333352

theorem rational_function_zeros (x : ℝ) : 
  (x^2 - 5*x + 6) / (3*x - 1) = 0 ↔ x = 2 ∨ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_rational_function_zeros_l3333_333352


namespace NUMINAMATH_CALUDE_probability_all_green_apples_l3333_333333

def total_apples : ℕ := 10
def red_apples : ℕ := 6
def green_apples : ℕ := 4
def chosen_apples : ℕ := 3

theorem probability_all_green_apples :
  (Nat.choose green_apples chosen_apples : ℚ) / (Nat.choose total_apples chosen_apples) = 1 / 30 := by
  sorry

end NUMINAMATH_CALUDE_probability_all_green_apples_l3333_333333


namespace NUMINAMATH_CALUDE_curve_points_difference_l3333_333310

theorem curve_points_difference : 
  ∀ (a b : ℝ), a ≠ b → 
  (4 + a^2 = 8*a - 5) → 
  (4 + b^2 = 8*b - 5) → 
  |a - b| = 2 * Real.sqrt 7 := by
sorry

end NUMINAMATH_CALUDE_curve_points_difference_l3333_333310


namespace NUMINAMATH_CALUDE_puzzle_unique_solution_l3333_333321

/-- Represents a mapping from letters to digits -/
def LetterMapping := Char → Fin 10

/-- Checks if a mapping is valid (different letters map to different digits) -/
def is_valid_mapping (m : LetterMapping) : Prop :=
  ∀ c1 c2, c1 ≠ c2 → m c1 ≠ m c2

/-- Converts a word to a number using the given mapping -/
def word_to_number (word : List Char) (m : LetterMapping) : ℕ :=
  word.foldl (fun acc d => 10 * acc + (m d).val) 0

/-- The cryptarithmetic puzzle equation -/
def puzzle_equation (m : LetterMapping) : Prop :=
  let dodge := word_to_number ['D', 'O', 'D', 'G', 'E'] m
  let strike := word_to_number ['S', 'T', 'R', 'I', 'K', 'E'] m
  let fighting := word_to_number ['F', 'I', 'G', 'H', 'T', 'I', 'N', 'G'] m
  dodge + strike = fighting

/-- The main theorem stating that the puzzle has a unique solution -/
theorem puzzle_unique_solution :
  ∃! m : LetterMapping, is_valid_mapping m ∧ puzzle_equation m :=
sorry

end NUMINAMATH_CALUDE_puzzle_unique_solution_l3333_333321


namespace NUMINAMATH_CALUDE_tony_books_count_l3333_333313

/-- The number of books Tony read -/
def tony_books : ℕ := 23

/-- The number of books Dean read -/
def dean_books : ℕ := 12

/-- The number of books Breanna read -/
def breanna_books : ℕ := 17

/-- The number of books Tony and Dean both read -/
def tony_dean_overlap : ℕ := 3

/-- The number of books all three read -/
def all_overlap : ℕ := 1

/-- The total number of different books read by all three -/
def total_different_books : ℕ := 47

theorem tony_books_count :
  tony_books + dean_books - tony_dean_overlap + breanna_books - all_overlap = total_different_books :=
by sorry

end NUMINAMATH_CALUDE_tony_books_count_l3333_333313


namespace NUMINAMATH_CALUDE_businessmen_one_beverage_businessmen_one_beverage_proof_l3333_333357

/-- The number of businessmen who drank only one type of beverage at a conference -/
theorem businessmen_one_beverage (total : ℕ) (coffee tea juice : ℕ) 
  (coffee_tea coffee_juice tea_juice : ℕ) (all_three : ℕ) : ℕ :=
  let total_businessmen : ℕ := 35
  let coffee_drinkers : ℕ := 18
  let tea_drinkers : ℕ := 15
  let juice_drinkers : ℕ := 8
  let coffee_and_tea : ℕ := 6
  let tea_and_juice : ℕ := 4
  let coffee_and_juice : ℕ := 3
  let all_beverages : ℕ := 2
  21

#check businessmen_one_beverage

/-- Proof that 21 businessmen drank only one type of beverage -/
theorem businessmen_one_beverage_proof : 
  businessmen_one_beverage 35 18 15 8 6 3 4 2 = 21 := by
  sorry

end NUMINAMATH_CALUDE_businessmen_one_beverage_businessmen_one_beverage_proof_l3333_333357


namespace NUMINAMATH_CALUDE_tom_catches_jerry_after_80_min_l3333_333367

/-- Represents the shape of the track -/
inductive TrackShape
| FigureEight

/-- Represents a runner on the track -/
structure Runner where
  speed : ℝ
  position : ℝ

/-- Represents the state of the race at a given time -/
structure RaceState where
  tom : Runner
  jerry : Runner
  time : ℝ

/-- The initial state of the race -/
def initial_state : RaceState := sorry

/-- The state of the race after 20 minutes -/
def state_after_20_min : RaceState := sorry

/-- The state of the race after 35 minutes -/
def state_after_35_min : RaceState := sorry

/-- Predicate to check if a runner has completed a full lap -/
def has_completed_full_lap (runner : Runner) (time : ℝ) : Prop := sorry

/-- Predicate to check if Tom is directly above Jerry -/
def tom_above_jerry (state : RaceState) : Prop := sorry

/-- Predicate to check if Jerry is directly above Tom -/
def jerry_above_tom (state : RaceState) : Prop := sorry

/-- Predicate to check if Tom has returned to his starting point -/
def tom_at_start (state : RaceState) : Prop := sorry

/-- Predicate to check if Tom has caught up with Jerry -/
def tom_caught_jerry (state : RaceState) : Prop := sorry

theorem tom_catches_jerry_after_80_min 
  (track : TrackShape)
  (h1 : track = TrackShape.FigureEight)
  (h2 : jerry_above_tom initial_state)
  (h3 : tom_above_jerry state_after_20_min)
  (h4 : ¬ has_completed_full_lap initial_state.tom 20)
  (h5 : ¬ has_completed_full_lap initial_state.jerry 20)
  (h6 : tom_at_start state_after_35_min)
  : tom_caught_jerry { tom := initial_state.tom, jerry := initial_state.jerry, time := 80 } := by
  sorry

end NUMINAMATH_CALUDE_tom_catches_jerry_after_80_min_l3333_333367


namespace NUMINAMATH_CALUDE_tank_fill_time_l3333_333336

/-- Given three pipes with fill rates, calculates the time to fill a tank when all pipes are open -/
theorem tank_fill_time (p q r : ℝ) (hp : p = 1/3) (hq : q = 1/9) (hr : r = 1/18) :
  1 / (p + q + r) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tank_fill_time_l3333_333336


namespace NUMINAMATH_CALUDE_mixed_number_less_than_decimal_l3333_333334

theorem mixed_number_less_than_decimal : -1 - (3 / 5 : ℚ) < -1.5 := by sorry

end NUMINAMATH_CALUDE_mixed_number_less_than_decimal_l3333_333334


namespace NUMINAMATH_CALUDE_square_side_equations_l3333_333360

/-- Represents a line in 2D space --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  nonzero : a ≠ 0 ∨ b ≠ 0

/-- Represents a square in 2D space --/
structure Square where
  center : ℝ × ℝ
  side_length : ℝ
  parallel_line : Line

/-- Check if two lines are parallel --/
def are_parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Check if two lines are perpendicular --/
def are_perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- The main theorem --/
theorem square_side_equations (s : Square)
  (h1 : s.center = (-3, -4))
  (h2 : s.side_length = 2 * Real.sqrt 5)
  (h3 : s.parallel_line = ⟨2, 1, 3, Or.inl (by norm_num)⟩) :
  ∃ (l1 l2 l3 l4 : Line),
    (l1 = ⟨2, 1, 15, Or.inl (by norm_num)⟩) ∧
    (l2 = ⟨2, 1, 5, Or.inl (by norm_num)⟩) ∧
    (l3 = ⟨1, -2, 0, Or.inr (by norm_num)⟩) ∧
    (l4 = ⟨1, -2, -10, Or.inr (by norm_num)⟩) ∧
    are_parallel l1 s.parallel_line ∧
    are_parallel l2 s.parallel_line ∧
    are_perpendicular l1 l3 ∧
    are_perpendicular l1 l4 ∧
    are_perpendicular l2 l3 ∧
    are_perpendicular l2 l4 :=
  sorry

end NUMINAMATH_CALUDE_square_side_equations_l3333_333360


namespace NUMINAMATH_CALUDE_ring_endomorphism_properties_division_ring_commutativity_l3333_333308

structure RingWithEndomorphism (R : Type) [Ring R] :=
  (f : R → R)
  (f_surjective : Function.Surjective f)
  (f_hom : ∀ x y, f (x + y) = f x + f y)
  (f_hom_mul : ∀ x y, f (x * y) = f x * f y)
  (f_commutes : ∀ x, x * f x = f x * x)

theorem ring_endomorphism_properties {R : Type} [Ring R] (S : RingWithEndomorphism R) :
  (∀ x y : R, x * S.f y - S.f y * x = S.f x * y - y * S.f x) ∧
  (∀ x y : R, x * (x * y - y * x) = S.f x * (x * y - y * x)) :=
sorry

theorem division_ring_commutativity {R : Type} [DivisionRing R] (S : RingWithEndomorphism R) :
  (∃ x : R, S.f x ≠ x) → (∀ a b : R, a * b = b * a) :=
sorry

end NUMINAMATH_CALUDE_ring_endomorphism_properties_division_ring_commutativity_l3333_333308


namespace NUMINAMATH_CALUDE_no_isosceles_right_triangle_with_perimeter_60_l3333_333351

theorem no_isosceles_right_triangle_with_perimeter_60 :
  ¬ ∃ (a c : ℕ), 
    a > 0 ∧ 
    c > 0 ∧ 
    c * c = 2 * a * a ∧  -- Pythagorean theorem for isosceles right triangle
    2 * a + c = 60 :=    -- Perimeter condition
by sorry

end NUMINAMATH_CALUDE_no_isosceles_right_triangle_with_perimeter_60_l3333_333351


namespace NUMINAMATH_CALUDE_total_ways_eq_17922_l3333_333378

/-- Number of cookie flavors --/
def num_cookie_flavors : ℕ := 7

/-- Number of milk types --/
def num_milk_types : ℕ := 4

/-- Total number of products to purchase --/
def total_products : ℕ := 5

/-- Maximum number of same flavor Alpha can order --/
def alpha_max_same_flavor : ℕ := 2

/-- Function to calculate the number of ways Alpha can choose items --/
def alpha_choices (n : ℕ) : ℕ := sorry

/-- Function to calculate the number of ways Beta can choose cookies --/
def beta_choices (n : ℕ) : ℕ := sorry

/-- The total number of ways Alpha and Beta can purchase 5 products --/
def total_ways : ℕ := sorry

/-- Theorem stating the total number of ways is 17922 --/
theorem total_ways_eq_17922 : total_ways = 17922 := by sorry

end NUMINAMATH_CALUDE_total_ways_eq_17922_l3333_333378


namespace NUMINAMATH_CALUDE_solution_set_when_m_neg_one_range_of_m_l3333_333395

-- Define the function f
def f (x m : ℝ) : ℝ := |x + m| + |2*x - 1|

-- Part I
theorem solution_set_when_m_neg_one :
  {x : ℝ | f x (-1) ≤ 2} = {x : ℝ | 0 ≤ x ∧ x ≤ 4/3} := by sorry

-- Part II
def A (m : ℝ) : Set ℝ := {x : ℝ | f x m ≤ |2*x + 1|}

theorem range_of_m (h : Set.Icc (3/4 : ℝ) 2 ⊆ A m) :
  m ∈ Set.Icc (-11/4 : ℝ) 0 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_m_neg_one_range_of_m_l3333_333395


namespace NUMINAMATH_CALUDE_point_coordinates_l3333_333354

def fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

theorem point_coordinates :
  ∀ (x y : ℝ),
  fourth_quadrant x y →
  |y| = 12 →
  |x| = 4 →
  (x, y) = (4, -12) :=
by sorry

end NUMINAMATH_CALUDE_point_coordinates_l3333_333354


namespace NUMINAMATH_CALUDE_reciprocal_of_repeating_decimal_one_third_l3333_333327

theorem reciprocal_of_repeating_decimal_one_third (x : ℚ) : 
  (x = 1/3) → (1/x = 3) := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_repeating_decimal_one_third_l3333_333327


namespace NUMINAMATH_CALUDE_weight_difference_l3333_333339

def mildred_weight : ℕ := 59
def carol_weight : ℕ := 9

theorem weight_difference : mildred_weight - carol_weight = 50 := by
  sorry

end NUMINAMATH_CALUDE_weight_difference_l3333_333339


namespace NUMINAMATH_CALUDE_coin_identification_strategy_exists_l3333_333309

/-- Represents the result of a weighing -/
inductive WeighingResult
| Equal : WeighingResult
| LeftHeavier : WeighingResult
| RightHeavier : WeighingResult

/-- Represents a coin -/
structure Coin :=
(id : Nat)
(is_genuine : Bool)

/-- Represents the scale used for weighing -/
def Scale := List Coin → List Coin → WeighingResult

/-- Represents the strategy for identifying genuine coins -/
def IdentificationStrategy := Scale → List Coin → List Coin

theorem coin_identification_strategy_exists :
  ∀ (coins : List Coin),
    coins.length = 8 →
    (coins.filter (λ c => c.is_genuine)).length = 7 →
    ∃ (strategy : IdentificationStrategy),
      ∀ (scale : Scale),
        let identified := strategy scale coins
        identified.length ≥ 5 ∧
        ∀ c ∈ identified, c.is_genuine ∧
        ∀ c ∈ identified, c ∉ (coins.filter (λ c => ¬c.is_genuine)) :=
by sorry

end NUMINAMATH_CALUDE_coin_identification_strategy_exists_l3333_333309


namespace NUMINAMATH_CALUDE_quality_related_to_renovation_probability_two_qualified_l3333_333337

-- Define the data from the table
def qualified_before : ℕ := 60
def substandard_before : ℕ := 40
def qualified_after : ℕ := 80
def substandard_after : ℕ := 20
def total_sample : ℕ := 200

-- Define the K^2 statistic
def K_squared (a b c d : ℕ) : ℚ :=
  let n : ℕ := a + b + c + d
  (n : ℚ) * (a * d - b * c : ℚ)^2 / ((a + b : ℚ) * (c + d : ℚ) * (a + c : ℚ) * (b + d : ℚ))

-- Define the critical value for 99% certainty
def critical_value : ℚ := 6635 / 1000

-- Theorem for part 1
theorem quality_related_to_renovation :
  K_squared qualified_before substandard_before qualified_after substandard_after > critical_value := by
  sorry

-- Theorem for part 2
theorem probability_two_qualified :
  (Nat.choose 3 2 : ℚ) / (Nat.choose 5 2 : ℚ) = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_quality_related_to_renovation_probability_two_qualified_l3333_333337


namespace NUMINAMATH_CALUDE_sample_size_calculation_l3333_333318

/-- Given three districts with population ratios 2:3:5 and 100 people sampled from the largest district,
    the total sample size is 200. -/
theorem sample_size_calculation (ratio_a ratio_b ratio_c : ℕ) 
  (largest_district_sample : ℕ) :
  ratio_a = 2 → ratio_b = 3 → ratio_c = 5 → 
  largest_district_sample = 100 →
  (ratio_a + ratio_b + ratio_c : ℚ) * largest_district_sample / ratio_c = 200 :=
by sorry

end NUMINAMATH_CALUDE_sample_size_calculation_l3333_333318


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3333_333382

-- Define the expansion
def expansion (a : ℝ) (x : ℝ) : ℝ := (2 + a * x) * (1 + x)^5

-- Define the coefficient of x^2
def coeff_x2 (a : ℝ) : ℝ := 20 + 5 * a

-- Theorem statement
theorem sum_of_coefficients (a : ℝ) (h : coeff_x2 a = 15) : 
  ∃ (sum : ℝ), sum = expansion a 1 ∧ sum = 64 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3333_333382


namespace NUMINAMATH_CALUDE_chord_length_squared_l3333_333347

theorem chord_length_squared (r₁ r₂ R : ℝ) (h₁ : r₁ = 4) (h₂ : r₂ = 7) (h₃ : R = 10)
  (h₄ : r₁ > 0) (h₅ : r₂ > 0) (h₆ : R > 0) (h₇ : r₁ + r₂ < R) :
  let d := R - r₂
  ∃ x, x^2 = d^2 + (R - r₁)^2 ∧ 4 * x^2 = 364 :=
by sorry

end NUMINAMATH_CALUDE_chord_length_squared_l3333_333347


namespace NUMINAMATH_CALUDE_range_of_2x_plus_y_range_of_c_l3333_333368

-- Define the circle
def Circle (x y : ℝ) : Prop := x^2 + y^2 = 2*y

-- Statement for the range of 2x + y
theorem range_of_2x_plus_y (x y : ℝ) (h : Circle x y) :
  -1 - Real.sqrt 5 ≤ 2*x + y ∧ 2*x + y ≤ 1 + Real.sqrt 5 := by sorry

-- Statement for the range of c
theorem range_of_c (c : ℝ) (h : ∀ x y : ℝ, Circle x y → x + y + c > 0) :
  c ≥ -1 := by sorry

end NUMINAMATH_CALUDE_range_of_2x_plus_y_range_of_c_l3333_333368


namespace NUMINAMATH_CALUDE_ellipse_sum_bound_l3333_333312

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop :=
  x^2 / 144 + y^2 / 25 = 1

-- Theorem statement
theorem ellipse_sum_bound :
  ∀ x y : ℝ, is_on_ellipse x y → -13 ≤ x + y ∧ x + y ≤ 13 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_sum_bound_l3333_333312


namespace NUMINAMATH_CALUDE_count_special_numbers_eq_285_l3333_333303

def count_special_numbers : ℕ :=
  (Finset.range 9).sum (fun i =>
    (Finset.range (9 - i)).card * (Finset.range (9 - i)).card)

theorem count_special_numbers_eq_285 :
  count_special_numbers = 285 := by
  sorry

end NUMINAMATH_CALUDE_count_special_numbers_eq_285_l3333_333303


namespace NUMINAMATH_CALUDE_smallest_cube_ending_392_l3333_333344

theorem smallest_cube_ending_392 : 
  ∃ (n : ℕ), n > 0 ∧ n^3 % 1000 = 392 ∧ ∀ (m : ℕ), m > 0 ∧ m^3 % 1000 = 392 → n ≤ m :=
by
  use 22
  sorry

end NUMINAMATH_CALUDE_smallest_cube_ending_392_l3333_333344


namespace NUMINAMATH_CALUDE_hyperbola_sufficient_not_necessary_l3333_333366

/-- Hyperbola equation -/
def is_hyperbola (x y a b : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

/-- Asymptotes equation -/
def is_asymptote (x y a b : ℝ) : Prop :=
  y = b/a * x ∨ y = -b/a * x

/-- The hyperbola equation is a sufficient but not necessary condition for the asymptotes equation -/
theorem hyperbola_sufficient_not_necessary (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y, is_hyperbola x y a b → is_asymptote x y a b) ∧
  ¬(∀ x y, is_asymptote x y a b → is_hyperbola x y a b) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_sufficient_not_necessary_l3333_333366


namespace NUMINAMATH_CALUDE_units_digit_of_p_l3333_333362

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- Predicate for an integer having a positive units digit -/
def hasPositiveUnitsDigit (n : ℤ) : Prop := unitsDigit n.natAbs ≠ 0

theorem units_digit_of_p (p : ℤ) 
  (hp_pos : hasPositiveUnitsDigit p)
  (hp_cube_square : unitsDigit (p^3).natAbs = unitsDigit (p^2).natAbs)
  (hp_plus_two : unitsDigit ((p + 2).natAbs) = 8) :
  unitsDigit p.natAbs = 6 :=
sorry

end NUMINAMATH_CALUDE_units_digit_of_p_l3333_333362


namespace NUMINAMATH_CALUDE_min_sum_squares_l3333_333305

theorem min_sum_squares (x y : ℝ) (h : x * y - x - y = 1) :
  ∃ (min : ℝ), min = 6 - 4 * Real.sqrt 2 ∧ 
  ∀ (a b : ℝ), a * b - a - b = 1 → a^2 + b^2 ≥ min := by
  sorry

end NUMINAMATH_CALUDE_min_sum_squares_l3333_333305


namespace NUMINAMATH_CALUDE_distance_proof_l3333_333371

/- Define the speeds of A and B in meters per minute -/
def speed_A : ℝ := 60
def speed_B : ℝ := 80

/- Define the rest time of B in minutes -/
def rest_time : ℝ := 14

/- Define the distance between A and B -/
def distance_AB : ℝ := 1680

/- Theorem statement -/
theorem distance_proof :
  ∃ (t : ℝ), 
    t > 0 ∧
    speed_A * t + speed_B * t = distance_AB ∧
    (distance_AB / speed_A + distance_AB / speed_B + rest_time) / 2 = t :=
by sorry

#check distance_proof

end NUMINAMATH_CALUDE_distance_proof_l3333_333371


namespace NUMINAMATH_CALUDE_unknown_number_proof_l3333_333302

theorem unknown_number_proof (x : ℝ) : x + 5 * 12 / (180 / 3) = 66 → x = 65 := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_proof_l3333_333302


namespace NUMINAMATH_CALUDE_system_of_equations_l3333_333319

theorem system_of_equations (a b : ℝ) 
  (eq1 : 2 * a - b = 12) 
  (eq2 : a + 2 * b = 8) : 
  3 * a + b = 20 := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_l3333_333319
