import Mathlib

namespace NUMINAMATH_CALUDE_sector_max_area_l337_33701

/-- A sector with perimeter 20 has maximum area of 25 when radius is 5 and central angle is 2 radians -/
theorem sector_max_area (r θ : ℝ) (h_perimeter : 2 * r + r * θ = 20) :
  let S := r^2 * θ / 2
  S ≤ 25 ∧ (S = 25 ↔ r = 5 ∧ θ = 2) := by
  sorry


end NUMINAMATH_CALUDE_sector_max_area_l337_33701


namespace NUMINAMATH_CALUDE_segment_length_to_reflection_segment_length_F_to_F_l337_33761

/-- The length of a segment from a point to its reflection over the x-axis -/
theorem segment_length_to_reflection (x y : ℝ) : 
  let F : ℝ × ℝ := (x, y)
  let F' : ℝ × ℝ := (x, -y)
  Real.sqrt ((F'.1 - F.1)^2 + (F'.2 - F.2)^2) = 2 * abs y :=
by sorry

/-- The specific case for F(-4, 3) -/
theorem segment_length_F_to_F'_is_6 : 
  let F : ℝ × ℝ := (-4, 3)
  let F' : ℝ × ℝ := (-4, -3)
  Real.sqrt ((F'.1 - F.1)^2 + (F'.2 - F.2)^2) = 6 :=
by sorry

end NUMINAMATH_CALUDE_segment_length_to_reflection_segment_length_F_to_F_l337_33761


namespace NUMINAMATH_CALUDE_smallest_x_absolute_value_l337_33717

theorem smallest_x_absolute_value : ∃ x : ℝ, 
  (∀ y : ℝ, |5*y - 3| = 15 → x ≤ y) ∧ |5*x - 3| = 15 := by
  sorry

end NUMINAMATH_CALUDE_smallest_x_absolute_value_l337_33717


namespace NUMINAMATH_CALUDE_modular_congruence_l337_33739

theorem modular_congruence (n : ℕ) : 37^29 ≡ 7 [ZMOD 65] :=
by sorry

end NUMINAMATH_CALUDE_modular_congruence_l337_33739


namespace NUMINAMATH_CALUDE_not_adjacent_2010_2011_l337_33772

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def sorted_by_digit_sum_then_value (a b : ℕ) : Prop :=
  (sum_of_digits a < sum_of_digits b) ∨
  (sum_of_digits a = sum_of_digits b ∧ a < b)

theorem not_adjacent_2010_2011 (s : List ℕ) :
  s.length = 100 →
  (∃ k : ℕ, ∀ n ∈ s, k ≤ n ∧ n < k + 100) →
  s.Sorted sorted_by_digit_sum_then_value →
  ¬∃ i : ℕ, i < s.length - 1 ∧
    ((s.get ⟨i, by sorry⟩ = 2010 ∧ s.get ⟨i+1, by sorry⟩ = 2011) ∨
     (s.get ⟨i, by sorry⟩ = 2011 ∧ s.get ⟨i+1, by sorry⟩ = 2010)) :=
by sorry

end NUMINAMATH_CALUDE_not_adjacent_2010_2011_l337_33772


namespace NUMINAMATH_CALUDE_arrangement_two_rows_arrangement_person_not_at_ends_arrangement_girls_together_arrangement_boys_not_adjacent_l337_33774

-- 1
theorem arrangement_two_rows (n : ℕ) (m : ℕ) (h : n + m = 7) :
  (Nat.factorial 7) = 5040 :=
sorry

-- 2
theorem arrangement_person_not_at_ends (n : ℕ) (h : n = 7) :
  5 * (Nat.factorial 6) = 3600 :=
sorry

-- 3
theorem arrangement_girls_together (boys girls : ℕ) (h1 : boys = 3) (h2 : girls = 4) :
  (Nat.factorial 4) * (Nat.factorial 4) = 576 :=
sorry

-- 4
theorem arrangement_boys_not_adjacent (boys girls : ℕ) (h1 : boys = 3) (h2 : girls = 4) :
  (Nat.factorial 4) * (Nat.factorial 5 / Nat.factorial 2) = 1440 :=
sorry

end NUMINAMATH_CALUDE_arrangement_two_rows_arrangement_person_not_at_ends_arrangement_girls_together_arrangement_boys_not_adjacent_l337_33774


namespace NUMINAMATH_CALUDE_hyperbola_circle_intersection_l337_33722

/-- Given a hyperbola and a circle with specific properties, prove that m = 2 -/
theorem hyperbola_circle_intersection (a b m : ℝ) : 
  a > 0 → b > 0 → m > 0 →
  (∀ x y, x^2/a^2 - y^2/b^2 = 1) →
  (∃ c, c^2 = a^2 + b^2 ∧ c/a = Real.sqrt 2) →
  (∀ x y, (x - m)^2 + y^2 = 4) →
  (∃ x y, x = y ∧ (x - m)^2 + y^2 = 4 ∧ 2 * Real.sqrt (4 - (x - m)^2) = 2 * Real.sqrt 2) →
  m = 2 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_circle_intersection_l337_33722


namespace NUMINAMATH_CALUDE_dartboard_area_ratio_l337_33708

theorem dartboard_area_ratio :
  let outer_square_side : ℝ := 4
  let inner_square_side : ℝ := 2
  let triangle_leg : ℝ := 1 / Real.sqrt 2
  let s : ℝ := (1 / 2) * triangle_leg * triangle_leg
  let p : ℝ := (1 / 2) * (inner_square_side + outer_square_side) * (outer_square_side / 2 - triangle_leg)
  p / s = 12 := by sorry

end NUMINAMATH_CALUDE_dartboard_area_ratio_l337_33708


namespace NUMINAMATH_CALUDE_two_valid_arrangements_l337_33779

/-- Represents an arrangement of people in rows. -/
structure Arrangement where
  rows : ℕ
  front : ℕ

/-- Checks if an arrangement is valid according to the problem conditions. -/
def isValidArrangement (a : Arrangement) : Prop :=
  a.rows ≥ 3 ∧
  a.front * a.rows + a.rows * (a.rows - 1) / 2 = 100

/-- The main theorem stating that there are exactly two valid arrangements. -/
theorem two_valid_arrangements :
  ∃! (s : Finset Arrangement), (∀ a ∈ s, isValidArrangement a) ∧ s.card = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_valid_arrangements_l337_33779


namespace NUMINAMATH_CALUDE_tomato_picking_second_week_l337_33725

/-- Represents the number of tomatoes picked in each week -/
structure TomatoPicking where
  initial : ℕ
  first_week : ℕ
  second_week : ℕ
  third_week : ℕ
  remaining : ℕ

/-- Checks if the tomato picking satisfies the given conditions -/
def is_valid_picking (p : TomatoPicking) : Prop :=
  p.initial = 100 ∧
  p.first_week = p.initial / 4 ∧
  p.third_week = 2 * p.second_week ∧
  p.remaining = 15 ∧
  p.first_week + p.second_week + p.third_week + p.remaining = p.initial

theorem tomato_picking_second_week :
  ∀ p : TomatoPicking, is_valid_picking p → p.second_week = 20 := by
  sorry

end NUMINAMATH_CALUDE_tomato_picking_second_week_l337_33725


namespace NUMINAMATH_CALUDE_expand_product_l337_33713

theorem expand_product (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l337_33713


namespace NUMINAMATH_CALUDE_corner_removed_cube_edge_count_l337_33732

/-- Represents a cube with a given side length -/
structure Cube :=
  (sideLength : ℝ)

/-- Represents the solid formed by removing smaller cubes from the corners of a larger cube -/
structure CornerRemovedCube :=
  (originalCube : Cube)
  (removedCubeSize : ℝ)

/-- Calculates the number of edges in the solid formed by removing smaller cubes from the corners of a larger cube -/
def edgeCount (c : CornerRemovedCube) : ℕ :=
  12 * 2  -- Each original edge is split into two

/-- Theorem stating that removing cubes of side length 2 from each corner of a cube with side length 4 results in a solid with 24 edges -/
theorem corner_removed_cube_edge_count :
  let originalCube : Cube := ⟨4⟩
  let cornerRemovedCube : CornerRemovedCube := ⟨originalCube, 2⟩
  edgeCount cornerRemovedCube = 24 :=
by sorry

end NUMINAMATH_CALUDE_corner_removed_cube_edge_count_l337_33732


namespace NUMINAMATH_CALUDE_table_tennis_tournament_l337_33705

theorem table_tennis_tournament (n : ℕ) (x : ℕ) : 
  n > 3 → 
  Nat.choose (n - 3) 2 + 6 - x = 50 → 
  Nat.choose n 2 = 50 → 
  x = 1 := by
sorry

end NUMINAMATH_CALUDE_table_tennis_tournament_l337_33705


namespace NUMINAMATH_CALUDE_distance_walked_calculation_l337_33723

/-- Calculates the distance walked given the walking time and speed. -/
def distance_walked (time : ℝ) (speed : ℝ) : ℝ := time * speed

/-- Theorem: The distance walked is 499.98 meters given the specified conditions. -/
theorem distance_walked_calculation :
  let time : ℝ := 6
  let speed : ℝ := 83.33
  distance_walked time speed = 499.98 := by sorry

end NUMINAMATH_CALUDE_distance_walked_calculation_l337_33723


namespace NUMINAMATH_CALUDE_fraction_numerator_proof_l337_33709

theorem fraction_numerator_proof (x : ℚ) : 
  (x / (4 * x + 4) = 3 / 7) → x = -12 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_numerator_proof_l337_33709


namespace NUMINAMATH_CALUDE_complex_division_l337_33757

theorem complex_division : ((-2 : ℂ) - I) / I = -1 + 2*I := by sorry

end NUMINAMATH_CALUDE_complex_division_l337_33757


namespace NUMINAMATH_CALUDE_twenty_dollar_bills_l337_33718

theorem twenty_dollar_bills (total_amount : ℕ) (bill_denomination : ℕ) (h1 : total_amount = 280) (h2 : bill_denomination = 20) :
  total_amount / bill_denomination = 14 := by
sorry

end NUMINAMATH_CALUDE_twenty_dollar_bills_l337_33718


namespace NUMINAMATH_CALUDE_vector_decomposition_l337_33797

def x : Fin 3 → ℝ := ![6, -1, 7]
def p : Fin 3 → ℝ := ![1, -2, 0]
def q : Fin 3 → ℝ := ![-1, 1, 3]
def r : Fin 3 → ℝ := ![1, 0, 4]

theorem vector_decomposition :
  x = λ i => -p i - 3 * q i + 4 * r i :=
by sorry

end NUMINAMATH_CALUDE_vector_decomposition_l337_33797


namespace NUMINAMATH_CALUDE_painted_square_ratio_l337_33715

/-- Given a square with side length s and a brush of width w, 
    if the painted area along the midline and one diagonal is one-third of the square's area, 
    then the ratio s/w equals 2√2 + 1 -/
theorem painted_square_ratio (s w : ℝ) (h_positive_s : 0 < s) (h_positive_w : 0 < w) :
  s * w + 2 * (1/2 * ((s * Real.sqrt 2) / 2 - (w * Real.sqrt 2) / 2)^2) = s^2 / 3 →
  s / w = 2 * Real.sqrt 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_painted_square_ratio_l337_33715


namespace NUMINAMATH_CALUDE_stationery_store_pencils_l337_33743

theorem stationery_store_pencils (pens pencils markers : ℕ) : 
  pens * 6 = pencils * 5 →  -- ratio of pens to pencils is 5:6
  pens * 7 = markers * 5 →  -- ratio of pens to markers is 5:7
  pencils = pens + 4 →      -- 4 more pencils than pens
  markers = pens + 20 →     -- 20 more markers than pens
  pencils = 24 :=           -- prove that the number of pencils is 24
by sorry

end NUMINAMATH_CALUDE_stationery_store_pencils_l337_33743


namespace NUMINAMATH_CALUDE_rhombus_in_rectangle_perimeter_l337_33795

-- Define the points of the rectangle and rhombus
variable (I J K L E F G H : ℝ × ℝ)

-- Define the properties of the rectangle and rhombus
def is_rectangle (I J K L : ℝ × ℝ) : Prop := sorry
def is_rhombus (E F G H : ℝ × ℝ) : Prop := sorry
def inscribed (E F G H I J K L : ℝ × ℝ) : Prop := sorry
def interior_point (P Q R : ℝ × ℝ) : Prop := sorry

-- Define the distance function
def distance (P Q : ℝ × ℝ) : ℝ := sorry

-- Main theorem
theorem rhombus_in_rectangle_perimeter 
  (h_rectangle : is_rectangle I J K L)
  (h_rhombus : is_rhombus E F G H)
  (h_inscribed : inscribed E F G H I J K L)
  (h_E : interior_point E I J)
  (h_F : interior_point F J K)
  (h_G : interior_point G K L)
  (h_H : interior_point H L I)
  (h_IE : distance I E = 12)
  (h_EJ : distance E J = 25)
  (h_EG : distance E G = 35)
  (h_FH : distance F H = 42) :
  distance I J + distance J K + distance K L + distance L I = 110 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_in_rectangle_perimeter_l337_33795


namespace NUMINAMATH_CALUDE_chocolate_count_l337_33780

/-- The number of boxes of chocolates -/
def num_boxes : ℕ := 6

/-- The number of pieces in each box -/
def pieces_per_box : ℕ := 500

/-- The total number of chocolate pieces -/
def total_pieces : ℕ := num_boxes * pieces_per_box

theorem chocolate_count : total_pieces = 3000 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_count_l337_33780


namespace NUMINAMATH_CALUDE_make_up_average_is_95_percent_l337_33785

/-- Represents the average score of students who took the exam on the make-up date -/
def make_up_average (total_students : ℕ) (assigned_day_percent : ℚ) (assigned_day_average : ℚ) (overall_average : ℚ) : ℚ :=
  (overall_average * total_students - assigned_day_average * (assigned_day_percent * total_students)) / ((1 - assigned_day_percent) * total_students)

/-- Theorem stating the average score of students who took the exam on the make-up date -/
theorem make_up_average_is_95_percent :
  make_up_average 100 (70/100) (65/100) (74/100) = 95/100 := by
  sorry

end NUMINAMATH_CALUDE_make_up_average_is_95_percent_l337_33785


namespace NUMINAMATH_CALUDE_first_player_wins_l337_33759

/-- Represents the state of the game -/
structure GameState where
  player1Pos : Nat
  player2Pos : Nat

/-- Represents a valid move in the game -/
inductive Move where
  | one   : Move
  | two   : Move
  | three : Move
  | four  : Move

/-- The game board size -/
def boardSize : Nat := 101

/-- Checks if a move is valid given the current game state -/
def isValidMove (state : GameState) (player : Nat) (move : Move) : Bool :=
  match player, move with
  | 1, Move.one   => state.player1Pos + 1 < state.player2Pos
  | 1, Move.two   => state.player1Pos + 2 < state.player2Pos
  | 1, Move.three => state.player1Pos + 3 < state.player2Pos
  | 1, Move.four  => state.player1Pos + 4 < state.player2Pos
  | 2, Move.one   => state.player2Pos - 1 > state.player1Pos
  | 2, Move.two   => state.player2Pos - 2 > state.player1Pos
  | 2, Move.three => state.player2Pos - 3 > state.player1Pos
  | 2, Move.four  => state.player2Pos - 4 > state.player1Pos
  | _, _          => false

/-- Applies a move to the game state -/
def applyMove (state : GameState) (player : Nat) (move : Move) : GameState :=
  match player, move with
  | 1, Move.one   => { state with player1Pos := state.player1Pos + 1 }
  | 1, Move.two   => { state with player1Pos := state.player1Pos + 2 }
  | 1, Move.three => { state with player1Pos := state.player1Pos + 3 }
  | 1, Move.four  => { state with player1Pos := state.player1Pos + 4 }
  | 2, Move.one   => { state with player2Pos := state.player2Pos - 1 }
  | 2, Move.two   => { state with player2Pos := state.player2Pos - 2 }
  | 2, Move.three => { state with player2Pos := state.player2Pos - 3 }
  | 2, Move.four  => { state with player2Pos := state.player2Pos - 4 }
  | _, _          => state

/-- Checks if the game is over -/
def isGameOver (state : GameState) : Bool :=
  state.player1Pos = boardSize || state.player2Pos = 1

/-- Theorem: The first player has a winning strategy -/
theorem first_player_wins :
  ∃ (strategy : GameState → Move),
    ∀ (opponent_strategy : GameState → Move),
      let game_result := (sorry : GameState)  -- Simulate game play
      isGameOver game_result ∧ game_result.player1Pos = boardSize :=
sorry


end NUMINAMATH_CALUDE_first_player_wins_l337_33759


namespace NUMINAMATH_CALUDE_elephant_hole_theorem_l337_33766

/-- A paper represents a rectangular sheet with a given area -/
structure Paper where
  area : ℝ
  area_pos : area > 0

/-- A series of cuts can be represented as a function that transforms a paper -/
def Cut := Paper → Paper

/-- The theorem states that there exists a cut that can create a hole larger than the original paper -/
theorem elephant_hole_theorem (initial_paper : Paper) (k : ℝ) (h_k : k > 1) :
  ∃ (cut : Cut), (cut initial_paper).area > k * initial_paper.area := by
  sorry

end NUMINAMATH_CALUDE_elephant_hole_theorem_l337_33766


namespace NUMINAMATH_CALUDE_triangle_side_length_l337_33707

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) :
  a = 1 →
  b = Real.sqrt 3 →
  Real.sin B = Real.sin (2 * A) →
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  a * Real.sin C = b * Real.sin A →
  a * Real.sin C = c * Real.sin B →
  c = 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l337_33707


namespace NUMINAMATH_CALUDE_max_value_inequality_l337_33727

theorem max_value_inequality (a b c : ℝ) 
  (sum_eq : a + b + c = 3)
  (a_ge : a ≥ -1)
  (b_ge : b ≥ -2)
  (c_ge : c ≥ -4)
  (abc_nonneg : a * b * c ≥ 0) :
  Real.sqrt (4 * a + 4) + Real.sqrt (4 * b + 8) + Real.sqrt (4 * c + 16) ≤ 2 * Real.sqrt 30 :=
by sorry

end NUMINAMATH_CALUDE_max_value_inequality_l337_33727


namespace NUMINAMATH_CALUDE_two_numbers_squares_sum_cube_cubes_sum_square_l337_33787

theorem two_numbers_squares_sum_cube_cubes_sum_square :
  ∃ (a b : ℕ), a ≠ b ∧ a > 0 ∧ b > 0 ∧
  (∃ (c : ℕ), a^2 + b^2 = c^3) ∧
  (∃ (d : ℕ), a^3 + b^3 = d^2) := by
sorry

end NUMINAMATH_CALUDE_two_numbers_squares_sum_cube_cubes_sum_square_l337_33787


namespace NUMINAMATH_CALUDE_small_semicircle_radius_l337_33744

/-- Configuration of tangent shapes --/
structure TangentShapes where
  R : ℝ  -- Radius of large semicircle
  r : ℝ  -- Radius of circle
  x : ℝ  -- Radius of small semicircle

/-- Predicate for valid configuration --/
def is_valid_config (shapes : TangentShapes) : Prop :=
  shapes.R = 12 ∧ shapes.r = 6 ∧ shapes.x > 0

/-- Theorem stating the radius of the small semicircle --/
theorem small_semicircle_radius (shapes : TangentShapes) 
  (h : is_valid_config shapes) : shapes.x = 4 := by
  sorry

end NUMINAMATH_CALUDE_small_semicircle_radius_l337_33744


namespace NUMINAMATH_CALUDE_count_valid_triples_l337_33762

def valid_triple (x y z : ℕ+) : Prop :=
  Nat.lcm x.val y.val = 48 ∧
  Nat.lcm x.val z.val = 450 ∧
  Nat.lcm y.val z.val = 600

theorem count_valid_triples :
  ∃! (n : ℕ), ∃ (S : Finset (ℕ+ × ℕ+ × ℕ+)),
    S.card = n ∧
    (∀ (t : ℕ+ × ℕ+ × ℕ+), t ∈ S ↔ valid_triple t.1 t.2.1 t.2.2) ∧
    n = 5 :=
sorry

end NUMINAMATH_CALUDE_count_valid_triples_l337_33762


namespace NUMINAMATH_CALUDE_officers_count_l337_33784

/-- The number of ways to choose 4 distinct officers from a group of n people -/
def choose_officers (n : ℕ) : ℕ := n * (n - 1) * (n - 2) * (n - 3)

/-- The number of club members -/
def club_members : ℕ := 12

/-- Theorem stating that choosing 4 officers from 12 members results in 11880 possibilities -/
theorem officers_count : choose_officers club_members = 11880 := by
  sorry

end NUMINAMATH_CALUDE_officers_count_l337_33784


namespace NUMINAMATH_CALUDE_initial_girls_count_l337_33700

theorem initial_girls_count (b g : ℚ) : 
  (3 * (g - 20) = b) →
  (6 * (b - 60) = g - 20) →
  g = 700 / 17 := by
  sorry

end NUMINAMATH_CALUDE_initial_girls_count_l337_33700


namespace NUMINAMATH_CALUDE_inscribed_circle_segment_lengths_l337_33783

/-- Given a triangle with sides a, b, c and an inscribed circle, 
    the lengths of the segments into which the points of tangency divide the sides 
    are (a + b - c)/2, (a + c - b)/2, and (b + c - a)/2. -/
theorem inscribed_circle_segment_lengths 
  (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) : 
  ∃ (x y z : ℝ),
    x = (a + b - c) / 2 ∧
    y = (a + c - b) / 2 ∧
    z = (b + c - a) / 2 ∧
    x + y = a ∧
    x + z = b ∧
    y + z = c :=
by sorry


end NUMINAMATH_CALUDE_inscribed_circle_segment_lengths_l337_33783


namespace NUMINAMATH_CALUDE_line_x_axis_intersection_l337_33741

/-- The line equation -/
def line_equation (x y : ℝ) : Prop := 5 * y + 3 * x = 15

/-- A point is on the x-axis if its y-coordinate is 0 -/
def on_x_axis (x y : ℝ) : Prop := y = 0

/-- The intersection point of the line and the x-axis -/
def intersection_point : ℝ × ℝ := (5, 0)

/-- Theorem: The intersection point satisfies both the line equation and lies on the x-axis -/
theorem line_x_axis_intersection :
  let (x, y) := intersection_point
  line_equation x y ∧ on_x_axis x y :=
by sorry

end NUMINAMATH_CALUDE_line_x_axis_intersection_l337_33741


namespace NUMINAMATH_CALUDE_power_of_two_problem_l337_33733

theorem power_of_two_problem (a b : ℕ+) 
  (h1 : (2 ^ a.val) ^ b.val = 2 ^ 2) 
  (h2 : 2 ^ a.val * 2 ^ b.val = 8) : 
  2 ^ a.val = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_problem_l337_33733


namespace NUMINAMATH_CALUDE_equal_one_two_digit_prob_l337_33771

/-- A 20-sided die with numbers from 1 to 20 -/
def twentySidedDie : Finset ℕ := Finset.range 20

/-- The probability of rolling a one-digit number on a 20-sided die -/
def probOneDigit : ℚ := 9 / 20

/-- The probability of rolling a two-digit number on a 20-sided die -/
def probTwoDigit : ℚ := 11 / 20

/-- The number of dice rolled -/
def numDice : ℕ := 6

/-- The probability of rolling an equal number of one-digit and two-digit numbers on 6 20-sided dice -/
theorem equal_one_two_digit_prob : 
  (Nat.choose numDice (numDice / 2) : ℚ) * probOneDigit ^ (numDice / 2) * probTwoDigit ^ (numDice / 2) = 970701 / 3200000 := by
  sorry

end NUMINAMATH_CALUDE_equal_one_two_digit_prob_l337_33771


namespace NUMINAMATH_CALUDE_joan_has_eight_kittens_l337_33788

/-- The number of kittens Joan has at the end, given the initial conditions and actions. -/
def joans_final_kittens (joan_initial : ℕ) (neighbor_initial : ℕ) 
  (joan_gave_away : ℕ) (neighbor_gave_away : ℕ) (joan_wants_to_adopt : ℕ) : ℕ :=
  let joan_after_giving := joan_initial - joan_gave_away
  let neighbor_after_giving := neighbor_initial - neighbor_gave_away
  let joan_can_adopt := min joan_wants_to_adopt neighbor_after_giving
  joan_after_giving + joan_can_adopt

/-- Theorem stating that Joan ends up with 8 kittens given the specific conditions. -/
theorem joan_has_eight_kittens : 
  joans_final_kittens 8 6 2 4 3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_joan_has_eight_kittens_l337_33788


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l337_33710

theorem polynomial_evaluation :
  ∀ y : ℝ, y > 0 → y^2 - 3*y - 9 = 0 → y^3 - 3*y^2 - 9*y + 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l337_33710


namespace NUMINAMATH_CALUDE_max_ab_value_l337_33751

/-- Given a function f(x) = -a * ln(x) + (a+1)x - (1/2)x^2 where a > 0,
    if f(x) ≥ -(1/2)x^2 + ax + b holds for all x > 0,
    then the maximum value of ab is e/2 -/
theorem max_ab_value (a b : ℝ) (h_a : a > 0) :
  (∀ x > 0, -a * Real.log x + (a + 1) * x - (1/2) * x^2 ≥ -(1/2) * x^2 + a * x + b) →
  (∃ m : ℝ, m = Real.exp 1 / 2 ∧ a * b ≤ m ∧ ∀ c d : ℝ, c > 0 → (∀ x > 0, -c * Real.log x + (c + 1) * x - (1/2) * x^2 ≥ -(1/2) * x^2 + c * x + d) → c * d ≤ m) :=
by sorry

end NUMINAMATH_CALUDE_max_ab_value_l337_33751


namespace NUMINAMATH_CALUDE_sum_of_cubes_of_roots_l337_33737

theorem sum_of_cubes_of_roots (a b c : ℝ) : 
  (a^3 - 2*a^2 + 2*a - 3 = 0) →
  (b^3 - 2*b^2 + 2*b - 3 = 0) →
  (c^3 - 2*c^2 + 2*c - 3 = 0) →
  a^3 + b^3 + c^3 = 5 := by
sorry

end NUMINAMATH_CALUDE_sum_of_cubes_of_roots_l337_33737


namespace NUMINAMATH_CALUDE_total_legs_l337_33792

/-- The number of legs for a chicken -/
def chicken_legs : ℕ := 2

/-- The number of legs for a sheep -/
def sheep_legs : ℕ := 4

/-- The number of chickens Farmer Brown fed -/
def num_chickens : ℕ := 7

/-- The number of sheep Farmer Brown fed -/
def num_sheep : ℕ := 5

/-- Theorem stating the total number of legs among the animals Farmer Brown fed -/
theorem total_legs : num_chickens * chicken_legs + num_sheep * sheep_legs = 34 := by
  sorry

end NUMINAMATH_CALUDE_total_legs_l337_33792


namespace NUMINAMATH_CALUDE_algebraic_simplification_l337_33768

theorem algebraic_simplification (a b : ℝ) : 
  (a^3 * b^4)^2 / (a * b^2)^3 = a^3 * b^2 ∧ 
  (-a^2)^3 * a^2 + a^8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_simplification_l337_33768


namespace NUMINAMATH_CALUDE_piggy_bank_coins_l337_33740

/-- The number of coins remaining after each day of withdrawal --/
def coins_remaining (initial : ℕ) : Fin 9 → ℕ
| 0 => initial  -- Initial number of coins
| 1 => initial * 8 / 9  -- After day 1
| 2 => initial * 8 * 7 / (9 * 8)  -- After day 2
| 3 => initial * 8 * 7 * 6 / (9 * 8 * 7)  -- After day 3
| 4 => initial * 8 * 7 * 6 * 5 / (9 * 8 * 7 * 6)  -- After day 4
| 5 => initial * 8 * 7 * 6 * 5 * 4 / (9 * 8 * 7 * 6 * 5)  -- After day 5
| 6 => initial * 8 * 7 * 6 * 5 * 4 * 3 / (9 * 8 * 7 * 6 * 5 * 4)  -- After day 6
| 7 => initial * 8 * 7 * 6 * 5 * 4 * 3 * 2 / (9 * 8 * 7 * 6 * 5 * 4 * 3)  -- After day 7
| 8 => initial * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1 / (9 * 8 * 7 * 6 * 5 * 4 * 3 * 2)  -- After day 8

theorem piggy_bank_coins (initial : ℕ) : 
  (coins_remaining initial 8 = 5) → (initial = 45) := by
  sorry

#eval coins_remaining 45 8  -- Should output 5

end NUMINAMATH_CALUDE_piggy_bank_coins_l337_33740


namespace NUMINAMATH_CALUDE_complex_purely_imaginary_l337_33782

/-- If m^2(1+i) + (m+i)i^2 is purely imaginary and m is a real number, then m = 0 -/
theorem complex_purely_imaginary (m : ℝ) : 
  (Complex.I * (m^2 - 1) = m^2*(1 + Complex.I) + (m + Complex.I)*Complex.I^2) → m = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_purely_imaginary_l337_33782


namespace NUMINAMATH_CALUDE_perpendicular_bisector_b_l337_33764

/-- The value of b for which the line x + y = b is the perpendicular bisector
    of the line segment connecting (1,4) and (7,10) -/
theorem perpendicular_bisector_b : ∃ b : ℝ,
  (∀ x y : ℝ, x + y = b ↔ 
    ((x - 4)^2 + (y - 7)^2 = 9) ∧ 
    ((x - 1) * (7 - 1) + (y - 4) * (10 - 4) = 0)) ∧
  b = 11 := by
  sorry


end NUMINAMATH_CALUDE_perpendicular_bisector_b_l337_33764


namespace NUMINAMATH_CALUDE_coefficient_x7y2_is_20_l337_33728

/-- The coefficient of x^7y^2 in the expansion of (x-y)(x+y)^8 -/
def coefficient_x7y2 : ℕ :=
  (Nat.choose 8 2) - (Nat.choose 8 1)

/-- Theorem: The coefficient of x^7y^2 in the expansion of (x-y)(x+y)^8 is 20 -/
theorem coefficient_x7y2_is_20 : coefficient_x7y2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x7y2_is_20_l337_33728


namespace NUMINAMATH_CALUDE_line_always_intersects_ellipse_iff_m_in_range_l337_33747

-- Define the line equation
def line (k : ℝ) (x : ℝ) : ℝ := k * x + 1

-- Define the ellipse equation
def ellipse (m : ℝ) (x y : ℝ) : Prop := x^2 / 5 + y^2 / m = 1

-- Theorem statement
theorem line_always_intersects_ellipse_iff_m_in_range :
  ∀ m : ℝ, (∀ k : ℝ, ∃ x y : ℝ, line k x = y ∧ ellipse m x y) ↔ 
  (m ≥ 1 ∧ m ≠ 5) :=
sorry

end NUMINAMATH_CALUDE_line_always_intersects_ellipse_iff_m_in_range_l337_33747


namespace NUMINAMATH_CALUDE_polynomial_ascending_powers_l337_33799

theorem polynomial_ascending_powers (x : ℝ) :
  x^2 - 2 - 5*x^4 + 3*x^3 = -2 + x^2 + 3*x^3 - 5*x^4 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_ascending_powers_l337_33799


namespace NUMINAMATH_CALUDE_whales_next_year_l337_33716

/-- The number of whales last year -/
def whales_last_year : ℕ := 4000

/-- The number of whales this year -/
def whales_this_year : ℕ := 2 * whales_last_year

/-- The predicted increase in whales for next year -/
def predicted_increase : ℕ := 800

/-- The theorem stating the number of whales next year -/
theorem whales_next_year : whales_this_year + predicted_increase = 8800 := by
  sorry

end NUMINAMATH_CALUDE_whales_next_year_l337_33716


namespace NUMINAMATH_CALUDE_number_of_d_values_l337_33711

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def distinct (a b c d : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

def value (a b c d : ℕ) : ℕ := a * 1000 + b * 100 + b * 10 + c

def sum_equation (a b c d : ℕ) : Prop :=
  value a b b c + value b c d b = value d a c d

def one_carryover (a b c d : ℕ) : Prop :=
  (a + b) % 10 = d ∧ a + b ≥ 10

theorem number_of_d_values :
  ∃ (s : Finset ℕ),
    (∀ a b c d : ℕ,
      is_digit a → is_digit b → is_digit c → is_digit d →
      distinct a b c d →
      sum_equation a b c d →
      one_carryover a b c d →
      d ∈ s) ∧
    s.card = 5 := by sorry

end NUMINAMATH_CALUDE_number_of_d_values_l337_33711


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l337_33775

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (2*x - 1)^5 = a₀ + a₁*(x-1) + a₂*(x-1)^2 + a₃*(x-1)^3 + a₄*(x-1)^4 + a₅*(x-1)^5) →
  a₀ + a₁ + a₂ + a₃ + a₄ + a₅ = 243 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l337_33775


namespace NUMINAMATH_CALUDE_teresas_age_at_birth_l337_33769

/-- Proves Teresa's age when Michiko was born, given the current ages and Morio's age at Michiko's birth -/
theorem teresas_age_at_birth (teresa_current_age morio_current_age morio_age_at_birth : ℕ) 
  (h1 : teresa_current_age = 59)
  (h2 : morio_current_age = 71)
  (h3 : morio_age_at_birth = 38) :
  teresa_current_age - (morio_current_age - morio_age_at_birth) = 26 := by
  sorry

#check teresas_age_at_birth

end NUMINAMATH_CALUDE_teresas_age_at_birth_l337_33769


namespace NUMINAMATH_CALUDE_nell_cards_remaining_l337_33793

/-- Given that Nell had 242 cards initially and gave away 136 cards, prove that she has 106 cards left. -/
theorem nell_cards_remaining (initial_cards : ℕ) (cards_given_away : ℕ) (h1 : initial_cards = 242) (h2 : cards_given_away = 136) :
  initial_cards - cards_given_away = 106 := by
  sorry

end NUMINAMATH_CALUDE_nell_cards_remaining_l337_33793


namespace NUMINAMATH_CALUDE_hat_knitting_time_l337_33756

/-- Represents the time (in hours) to knit various items --/
structure KnittingTimes where
  hat : ℝ
  scarf : ℝ
  mitten : ℝ
  sock : ℝ
  sweater : ℝ

/-- Calculates the total time to knit one set of clothes --/
def timeForOneSet (t : KnittingTimes) : ℝ :=
  t.hat + t.scarf + 2 * t.mitten + 2 * t.sock + t.sweater

/-- The main theorem stating that the time to knit a hat is 2 hours --/
theorem hat_knitting_time (t : KnittingTimes) 
  (h_scarf : t.scarf = 3)
  (h_mitten : t.mitten = 1)
  (h_sock : t.sock = 1.5)
  (h_sweater : t.sweater = 6)
  (h_total_time : 3 * timeForOneSet t = 48) : 
  t.hat = 2 := by
  sorry

end NUMINAMATH_CALUDE_hat_knitting_time_l337_33756


namespace NUMINAMATH_CALUDE_apple_cost_calculation_l337_33796

def total_budget : ℝ := 60
def hummus_cost : ℝ := 5
def hummus_quantity : ℕ := 2
def chicken_cost : ℝ := 20
def bacon_cost : ℝ := 10
def vegetables_cost : ℝ := 10
def apple_quantity : ℕ := 5

theorem apple_cost_calculation :
  let other_items_cost := hummus_cost * hummus_quantity + chicken_cost + bacon_cost + vegetables_cost
  let remaining_budget := total_budget - other_items_cost
  remaining_budget / apple_quantity = 2 := by sorry

end NUMINAMATH_CALUDE_apple_cost_calculation_l337_33796


namespace NUMINAMATH_CALUDE_dave_winfield_home_runs_l337_33734

theorem dave_winfield_home_runs : ∃ (x : ℕ), 
  (755 = 2 * x - 175) ∧ x = 465 := by sorry

end NUMINAMATH_CALUDE_dave_winfield_home_runs_l337_33734


namespace NUMINAMATH_CALUDE_wickets_before_last_match_is_55_l337_33763

/-- Represents a bowler's statistics -/
structure BowlerStats where
  initialAverage : ℝ
  lastMatchWickets : ℕ
  lastMatchRuns : ℕ
  averageDecrease : ℝ

/-- Calculates the number of wickets taken before the last match -/
def wicketsBeforeLastMatch (stats : BowlerStats) : ℕ :=
  sorry

/-- Theorem stating that given the specific conditions, the bowler took 55 wickets before the last match -/
theorem wickets_before_last_match_is_55 (stats : BowlerStats)
  (h1 : stats.initialAverage = 12.4)
  (h2 : stats.lastMatchWickets = 4)
  (h3 : stats.lastMatchRuns = 26)
  (h4 : stats.averageDecrease = 0.4) :
  wicketsBeforeLastMatch stats = 55 :=
  sorry

end NUMINAMATH_CALUDE_wickets_before_last_match_is_55_l337_33763


namespace NUMINAMATH_CALUDE_rectangle_area_diagonal_l337_33758

theorem rectangle_area_diagonal (l w d : ℝ) (h_ratio : l / w = 5 / 4) (h_diagonal : l^2 + w^2 = d^2) :
  l * w = (20 / 41) * d^2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_diagonal_l337_33758


namespace NUMINAMATH_CALUDE_polynomial_triangle_inequality_l337_33755

/-- A polynomial with nonnegative coefficients -/
structure NonnegPolynomial (α : Type*) [OrderedSemiring α] where
  polynomial : Polynomial α
  nonneg_coeff : ∀ i, 0 ≤ polynomial.coeff i

/-- Definition of a triangle -/
def is_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a < b + c ∧ b < a + c ∧ c < a + b

theorem polynomial_triangle_inequality 
  {n : ℕ} (hn : n ≥ 2) (P : NonnegPolynomial ℝ) 
  (hP : P.polynomial.degree = n) 
  (a b c : ℝ) (h_triangle : is_triangle a b c) : 
  is_triangle (P.polynomial.eval a ^ (1/n : ℝ)) 
              (P.polynomial.eval b ^ (1/n : ℝ)) 
              (P.polynomial.eval c ^ (1/n : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_triangle_inequality_l337_33755


namespace NUMINAMATH_CALUDE_successful_table_filling_l337_33798

theorem successful_table_filling :
  ∃ (t : Fin 6 → Fin 3 → Bool),
    ∀ (r1 r2 : Fin 6) (c1 c2 : Fin 3),
      r1 ≠ r2 → c1 ≠ c2 →
        (t r1 c1 = t r1 c2 ∧ t r1 c1 = t r2 c1 ∧ t r1 c1 = t r2 c2) = False :=
by sorry

end NUMINAMATH_CALUDE_successful_table_filling_l337_33798


namespace NUMINAMATH_CALUDE_sequence_sum_1997_l337_33754

/-- The sum of the sequence up to n, where n is a multiple of 3 -/
def sequenceSum (n : ℕ) : ℤ :=
  if n % 3 = 0 then
    let groupCount := n / 3
    let negativeGroups := groupCount / 2
    let positiveGroups := (groupCount + 1) / 2
    negativeGroups * (-4) + positiveGroups * 3
  else
    0

theorem sequence_sum_1997 :
  sequenceSum 1995 + 1996 - 1997 = 1665 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_1997_l337_33754


namespace NUMINAMATH_CALUDE_D_72_l337_33790

/-- D(n) represents the number of ways to write n as a product of integers > 1, where order matters -/
def D (n : ℕ) : ℕ := sorry

/-- Theorem stating that D(72) = 48 -/
theorem D_72 : D 72 = 48 := by sorry

end NUMINAMATH_CALUDE_D_72_l337_33790


namespace NUMINAMATH_CALUDE_opposites_power_2004_l337_33704

theorem opposites_power_2004 (x y : ℝ) 
  (h : |x + 1| + |y + 2*x| = 0) : 
  (x + y)^2004 = 1 := by
  sorry

end NUMINAMATH_CALUDE_opposites_power_2004_l337_33704


namespace NUMINAMATH_CALUDE_event_A_subset_event_B_l337_33730

-- Define the sample space for tossing two coins
inductive CoinOutcome
  | HH -- Both heads
  | HT -- First head, second tail
  | TH -- First tail, second head
  | TT -- Both tails

-- Define the probability space
def coin_toss_space : Type := CoinOutcome

-- Define the events A and B
def event_A : Set coin_toss_space := {CoinOutcome.HH}
def event_B : Set coin_toss_space := {CoinOutcome.HH, CoinOutcome.TT}

-- State the theorem
theorem event_A_subset_event_B : event_A ⊆ event_B := by sorry

end NUMINAMATH_CALUDE_event_A_subset_event_B_l337_33730


namespace NUMINAMATH_CALUDE_num_mc_questions_is_two_l337_33791

/-- The number of true-false questions in the quiz -/
def num_tf : ℕ := 4

/-- The number of answer choices for each multiple-choice question -/
def num_mc_choices : ℕ := 4

/-- The total number of ways to write the answer key -/
def total_ways : ℕ := 224

/-- The number of ways to answer the true-false questions, excluding all-same answers -/
def tf_ways : ℕ := 2^num_tf - 2

/-- Theorem stating that the number of multiple-choice questions is 2 -/
theorem num_mc_questions_is_two :
  ∃ (n : ℕ), tf_ways * (num_mc_choices^n) = total_ways ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_num_mc_questions_is_two_l337_33791


namespace NUMINAMATH_CALUDE_inequality_problem_l337_33794

theorem inequality_problem :
  -- Part 1: Maximum value of m
  (∃ M : ℝ, (∀ m : ℝ, (∃ x : ℝ, |x - 2| - |x + 3| ≥ |m + 1|) → m ≤ M) ∧
    (∃ x : ℝ, |x - 2| - |x + 3| ≥ |M + 1|) ∧
    M = 4) ∧
  -- Part 2: Inequality for positive a, b, c
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a + 2*b + c = 4 →
    1 / (a + b) + 1 / (b + c) ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_problem_l337_33794


namespace NUMINAMATH_CALUDE_marble_distribution_l337_33712

theorem marble_distribution (n : ℕ) (hn : n = 720) :
  (Finset.filter (fun m => m > 1 ∧ m < n ∧ n % m = 0) (Finset.range (n + 1))).card = 28 := by
  sorry

end NUMINAMATH_CALUDE_marble_distribution_l337_33712


namespace NUMINAMATH_CALUDE_fish_problem_l337_33703

theorem fish_problem (trout_weight salmon_weight campers fish_per_camper bass_weight : ℕ) 
  (h1 : trout_weight = 8)
  (h2 : salmon_weight = 24)
  (h3 : campers = 22)
  (h4 : fish_per_camper = 2)
  (h5 : bass_weight = 2) :
  (campers * fish_per_camper - (trout_weight + salmon_weight)) / bass_weight = 6 := by
  sorry

end NUMINAMATH_CALUDE_fish_problem_l337_33703


namespace NUMINAMATH_CALUDE_pointDifference_l337_33742

/-- Represents a team's performance in a soccer tournament --/
structure TeamPerformance where
  wins : ℕ
  draws : ℕ

/-- Calculates the total points for a team based on their performance --/
def calculatePoints (team : TeamPerformance) : ℕ :=
  team.wins * 3 + team.draws * 1

/-- The scoring system and match results for Joe's team and the first-place team --/
def joesTeam : TeamPerformance := ⟨1, 3⟩
def firstPlaceTeam : TeamPerformance := ⟨2, 2⟩

/-- The theorem stating the difference in points between the first-place team and Joe's team --/
theorem pointDifference : calculatePoints firstPlaceTeam - calculatePoints joesTeam = 2 := by
  sorry

end NUMINAMATH_CALUDE_pointDifference_l337_33742


namespace NUMINAMATH_CALUDE_base_conversion_theorem_l337_33729

/-- Converts a number from base b to base 10 -/
def to_base_10 (digits : List ℕ) (b : ℕ) : ℕ :=
  digits.enum.foldl (λ sum (i, d) => sum + d * b^i) 0

theorem base_conversion_theorem :
  let base_5_123 := to_base_10 [3, 2, 1] 5
  let base_8_107 := to_base_10 [7, 0, 1] 8
  let base_9_4321 := to_base_10 [1, 2, 3, 4] 9
  (2468 / base_5_123) * base_8_107 + base_9_4321 = 7789 := by sorry

end NUMINAMATH_CALUDE_base_conversion_theorem_l337_33729


namespace NUMINAMATH_CALUDE_first_number_100th_group_l337_33767

/-- The sequence term at position n -/
def sequenceTerm (n : ℕ) : ℕ := 3^(n - 1)

/-- The sum of the first n natural numbers -/
def triangularNumber (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The position of the first number in the nth group -/
def firstNumberPosition (n : ℕ) : ℕ := triangularNumber (n - 1) + 1

/-- The first number in the nth group -/
def firstNumberInGroup (n : ℕ) : ℕ := sequenceTerm (firstNumberPosition n)

theorem first_number_100th_group :
  firstNumberInGroup 100 = 3^4950 := by sorry

end NUMINAMATH_CALUDE_first_number_100th_group_l337_33767


namespace NUMINAMATH_CALUDE_function_properties_l337_33750

-- Define the function f(x) = k - 1/x
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := k - 1/x

-- Theorem statement
theorem function_properties (k : ℝ) :
  -- 1. Domain of f
  (∀ x : ℝ, x ≠ 0 → f k x ∈ Set.univ) ∧
  -- 2. f is increasing on (0, +∞)
  (∀ x y : ℝ, 0 < x → x < y → f k x < f k y) ∧
  -- 3. If f is odd, then k = 0
  ((∀ x : ℝ, x ≠ 0 → f k (-x) = -(f k x)) → k = 0) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l337_33750


namespace NUMINAMATH_CALUDE_no_chess_tournament_with_804_games_l337_33781

theorem no_chess_tournament_with_804_games : ¬∃ (n : ℕ), n > 0 ∧ n * (n - 4) = 1608 := by
  sorry

end NUMINAMATH_CALUDE_no_chess_tournament_with_804_games_l337_33781


namespace NUMINAMATH_CALUDE_tangent_perpendicular_point_l337_33770

def f (x : ℝ) := x^4 - x

theorem tangent_perpendicular_point :
  ∃! p : ℝ × ℝ, 
    p.2 = f p.1 ∧ 
    (4 * p.1^3 - 1) * (-1/3) = -1 ∧
    p = (1, 0) := by
  sorry

end NUMINAMATH_CALUDE_tangent_perpendicular_point_l337_33770


namespace NUMINAMATH_CALUDE_mary_fruit_difference_l337_33720

/-- Proves that Mary has 33 fewer peaches than apples given the conditions about Jake, Steven, and Mary's fruits. -/
theorem mary_fruit_difference :
  ∀ (steven_apples steven_peaches jake_apples jake_peaches mary_apples mary_peaches : ℕ),
  steven_apples = 11 →
  steven_peaches = 18 →
  jake_peaches + 8 = steven_peaches →
  jake_apples = steven_apples + 10 →
  mary_apples = 2 * jake_apples →
  mary_peaches * 2 = steven_peaches →
  (mary_peaches : ℤ) - (mary_apples : ℤ) = -33 := by
sorry

end NUMINAMATH_CALUDE_mary_fruit_difference_l337_33720


namespace NUMINAMATH_CALUDE_cube_sum_over_product_l337_33789

theorem cube_sum_over_product (x y z : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h_sum : x + y + z = 20)
  (h_diff_sq : (x - y)^2 + (x - z)^2 + (y - z)^2 = x * y * z) :
  (x^3 + y^3 + z^3) / (x * y * z) = 13 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_over_product_l337_33789


namespace NUMINAMATH_CALUDE_sock_order_ratio_l337_33721

/-- Represents the number of pairs of socks -/
structure SockOrder where
  black : ℕ
  blue : ℕ

/-- Represents the price of socks -/
structure SockPrice where
  blue : ℝ

/-- Calculates the total cost of a sock order given the prices -/
def totalCost (order : SockOrder) (price : SockPrice) : ℝ :=
  order.black * (3 * price.blue) + order.blue * price.blue

theorem sock_order_ratio : ∀ (original : SockOrder) (price : SockPrice),
  original.black = 6 →
  totalCost { black := original.blue, blue := original.black } price = 1.6 * totalCost original price →
  (original.black : ℝ) / original.blue = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_sock_order_ratio_l337_33721


namespace NUMINAMATH_CALUDE_abs_neg_2023_l337_33778

theorem abs_neg_2023 : |(-2023 : ℤ)| = 2023 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_2023_l337_33778


namespace NUMINAMATH_CALUDE_sector_arc_length_l337_33753

/-- Given a circular sector with area 4 cm² and central angle 2 radians, 
    the length of its arc is 4 cm. -/
theorem sector_arc_length (area : ℝ) (central_angle : ℝ) (arc_length : ℝ) : 
  area = 4 → central_angle = 2 → arc_length = area / central_angle * 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_arc_length_l337_33753


namespace NUMINAMATH_CALUDE_P_is_circle_l337_33777

-- Define the set of points P(x, y) satisfying the given equation
def P : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 10 * Real.sqrt ((p.1 - 1)^2 + (p.2 - 2)^2) = |3 * p.1 - 4 * p.2|}

-- Theorem stating that the set P forms a circle
theorem P_is_circle : ∃ (c : ℝ × ℝ) (r : ℝ), P = {p : ℝ × ℝ | (p.1 - c.1)^2 + (p.2 - c.2)^2 = r^2} :=
sorry

end NUMINAMATH_CALUDE_P_is_circle_l337_33777


namespace NUMINAMATH_CALUDE_hill_climb_time_l337_33719

/-- Proves that the time taken to reach the top of the hill is 4 hours -/
theorem hill_climb_time (descent_time : ℝ) (avg_speed_total : ℝ) (avg_speed_climb : ℝ) :
  descent_time = 2 →
  avg_speed_total = 2 →
  avg_speed_climb = 1.5 →
  let ascent_time := 4
  let total_time := ascent_time + descent_time
  let total_distance := avg_speed_total * total_time
  let climb_distance := avg_speed_climb * ascent_time
  climb_distance * 2 = total_distance →
  ascent_time = 4 := by
  sorry

end NUMINAMATH_CALUDE_hill_climb_time_l337_33719


namespace NUMINAMATH_CALUDE_greatest_third_side_l337_33776

theorem greatest_third_side (a b : ℝ) (ha : a = 7) (hb : b = 10) :
  ∃ (c : ℕ), c = 16 ∧ 
  (∀ (x : ℕ), x > c → ¬(a + b > x ∧ b + x > a ∧ x + a > b)) :=
sorry

end NUMINAMATH_CALUDE_greatest_third_side_l337_33776


namespace NUMINAMATH_CALUDE_inverse_composition_l337_33735

-- Define the functions h and k
noncomputable def h : ℝ → ℝ := sorry
noncomputable def k : ℝ → ℝ := sorry

-- State the theorem
theorem inverse_composition (x : ℝ) : 
  (h⁻¹ ∘ k) x = 3 * x - 4 → k⁻¹ (h 8) = 8 := by sorry

end NUMINAMATH_CALUDE_inverse_composition_l337_33735


namespace NUMINAMATH_CALUDE_boot_price_calculation_l337_33746

theorem boot_price_calculation (discount_percent : ℝ) (discounted_price : ℝ) : 
  discount_percent = 20 → discounted_price = 72 → 
  discounted_price / (1 - discount_percent / 100) = 90 := by
sorry

end NUMINAMATH_CALUDE_boot_price_calculation_l337_33746


namespace NUMINAMATH_CALUDE_female_managers_count_l337_33738

/-- Represents a company with employees and managers -/
structure Company where
  total_employees : ℕ
  male_employees : ℕ
  female_employees : ℕ
  total_managers : ℕ
  male_managers : ℕ
  female_managers : ℕ

/-- Conditions for the company -/
def ValidCompany (c : Company) : Prop :=
  c.female_employees = 1000 ∧
  c.total_employees = c.male_employees + c.female_employees ∧
  c.total_managers = c.male_managers + c.female_managers ∧
  5 * c.total_managers = 2 * c.total_employees ∧
  5 * c.male_managers = 2 * c.male_employees

/-- Theorem stating that in a valid company, the number of female managers is 400 -/
theorem female_managers_count (c : Company) (h : ValidCompany c) :
  c.female_managers = 400 := by
  sorry


end NUMINAMATH_CALUDE_female_managers_count_l337_33738


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l337_33760

theorem sufficient_not_necessary (x : ℝ) : 
  (∀ x, x > 1 → x > 0) ∧ 
  (∃ x, x > 0 ∧ ¬(x > 1)) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l337_33760


namespace NUMINAMATH_CALUDE_diet_soda_bottles_l337_33706

/-- Given a grocery store inventory, calculate the number of diet soda bottles. -/
theorem diet_soda_bottles (total_bottles regular_soda_bottles : ℕ) 
  (h1 : total_bottles = 17)
  (h2 : regular_soda_bottles = 9) :
  total_bottles - regular_soda_bottles = 8 := by
  sorry

#check diet_soda_bottles

end NUMINAMATH_CALUDE_diet_soda_bottles_l337_33706


namespace NUMINAMATH_CALUDE_base_8_4531_equals_2393_l337_33749

def base_8_to_10 (a b c d : ℕ) : ℕ :=
  a * 8^3 + b * 8^2 + c * 8^1 + d * 8^0

theorem base_8_4531_equals_2393 :
  base_8_to_10 4 5 3 1 = 2393 := by
  sorry

end NUMINAMATH_CALUDE_base_8_4531_equals_2393_l337_33749


namespace NUMINAMATH_CALUDE_roots_of_varying_signs_l337_33765

theorem roots_of_varying_signs :
  (∃ x y : ℝ, x * y < 0 ∧ 4 * x^2 - 8 = 40 ∧ 4 * y^2 - 8 = 40) ∧
  (∃ x y : ℝ, x * y < 0 ∧ (3*x-2)^2 = (x+2)^2 ∧ (3*y-2)^2 = (y+2)^2) ∧
  (∃ x y : ℝ, x * y < 0 ∧ x^3 - 8*x^2 + 13*x + 10 = 0 ∧ y^3 - 8*y^2 + 13*y + 10 = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_roots_of_varying_signs_l337_33765


namespace NUMINAMATH_CALUDE_spherical_to_cartesian_l337_33731

/-- Conversion from spherical coordinates to Cartesian coordinates -/
theorem spherical_to_cartesian :
  let r : ℝ := 8
  let θ : ℝ := π / 3
  let φ : ℝ := π / 6
  let x : ℝ := r * Real.sin θ * Real.cos φ
  let y : ℝ := r * Real.sin θ * Real.sin φ
  let z : ℝ := r * Real.cos θ
  (x, y, z) = (6, 2 * Real.sqrt 3, 4) :=
by sorry

end NUMINAMATH_CALUDE_spherical_to_cartesian_l337_33731


namespace NUMINAMATH_CALUDE_train_distance_proof_l337_33702

-- Define the speeds of the trains
def speed_train1 : ℝ := 20
def speed_train2 : ℝ := 25

-- Define the difference in distance traveled
def distance_difference : ℝ := 55

-- Define the total distance between stations
def total_distance : ℝ := 495

-- Theorem statement
theorem train_distance_proof :
  ∃ (time : ℝ) (distance1 distance2 : ℝ),
    time > 0 ∧
    distance1 = speed_train1 * time ∧
    distance2 = speed_train2 * time ∧
    distance2 = distance1 + distance_difference ∧
    total_distance = distance1 + distance2 :=
by
  sorry

end NUMINAMATH_CALUDE_train_distance_proof_l337_33702


namespace NUMINAMATH_CALUDE_movie_production_cost_ratio_l337_33724

/-- Proves that the ratio of equipment rental cost to the combined cost of food and actors is 2:1 --/
theorem movie_production_cost_ratio :
  let actor_cost : ℕ := 1200
  let num_people : ℕ := 50
  let food_cost_per_person : ℕ := 3
  let total_food_cost : ℕ := num_people * food_cost_per_person
  let combined_cost : ℕ := actor_cost + total_food_cost
  let selling_price : ℕ := 10000
  let profit : ℕ := 5950
  let total_cost : ℕ := selling_price - profit
  let equipment_cost : ℕ := total_cost - combined_cost
  equipment_cost / combined_cost = 2 := by sorry

end NUMINAMATH_CALUDE_movie_production_cost_ratio_l337_33724


namespace NUMINAMATH_CALUDE_equilateral_triangle_side_length_l337_33745

/-- The side length of an equilateral triangle with inscribed circle and smaller touching circles -/
theorem equilateral_triangle_side_length (r : ℝ) (h : r > 0) : ∃ a : ℝ, 
  a > 0 ∧ 
  (∃ R : ℝ, R > 0 ∧ 
    -- R is the radius of the inscribed circle
    R = (a * Real.sqrt 3) / 6 ∧
    -- Relationship between R, r, and the altitude of the triangle
    R / r = (a * Real.sqrt 3 / 3) / (a * Real.sqrt 3 / 3 - R - r)) ∧
  a = 6 * r * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_side_length_l337_33745


namespace NUMINAMATH_CALUDE_sum_of_first_6n_equals_465_l337_33773

/-- The value of n that satisfies the given condition -/
def n : ℕ := 5

/-- The sum of the first k positive integers -/
def sum_first_k (k : ℕ) : ℕ := k * (k + 1) / 2

/-- The condition that the sum of the first 5n positive integers is 325 more than the sum of the first n positive integers -/
axiom condition : sum_first_k (5 * n) = sum_first_k n + 325

/-- The theorem to be proved -/
theorem sum_of_first_6n_equals_465 : sum_first_k (6 * n) = 465 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_first_6n_equals_465_l337_33773


namespace NUMINAMATH_CALUDE_quadratic_polynomial_condition_l337_33786

/-- A second degree polynomial -/
structure QuadraticPolynomial where
  u : ℝ
  v : ℝ
  w : ℝ

/-- Evaluation of a quadratic polynomial at a point x -/
def QuadraticPolynomial.eval (p : QuadraticPolynomial) (x : ℝ) : ℝ :=
  p.u * x^2 + p.v * x + p.w

theorem quadratic_polynomial_condition (p : QuadraticPolynomial) :
  (∀ a : ℝ, a ≥ 1 → p.eval (a^2 + a) ≥ a * p.eval (a + 1)) ↔
  (p.u > 0 ∧ p.w ≤ 4 * p.u) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_condition_l337_33786


namespace NUMINAMATH_CALUDE_increasing_implies_a_geq_neg_two_l337_33752

/-- A quadratic function f(x) = x^2 + 2(a-1)x - 3 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-1)*x - 3

/-- The property of f being increasing on [3, +∞) -/
def is_increasing_on_interval (a : ℝ) : Prop :=
  ∀ x y, x ≥ 3 → y ≥ 3 → x < y → f a x < f a y

/-- Theorem: If f is increasing on [3, +∞), then a ≥ -2 -/
theorem increasing_implies_a_geq_neg_two (a : ℝ) :
  is_increasing_on_interval a → a ≥ -2 :=
by sorry

end NUMINAMATH_CALUDE_increasing_implies_a_geq_neg_two_l337_33752


namespace NUMINAMATH_CALUDE_compare_expressions_l337_33748

theorem compare_expressions (x y : ℝ) (h1 : x < y) (h2 : y < 0) :
  (x^2 + y^2) * (x - y) > (x^2 - y^2) * (x + y) := by
  sorry

end NUMINAMATH_CALUDE_compare_expressions_l337_33748


namespace NUMINAMATH_CALUDE_abs_x_minus_3_minus_sqrt_x_minus_4_squared_l337_33736

theorem abs_x_minus_3_minus_sqrt_x_minus_4_squared (x : ℝ) (h : x < 3) :
  |x - 3 - Real.sqrt ((x - 4)^2)| = 7 - 2*x := by sorry

end NUMINAMATH_CALUDE_abs_x_minus_3_minus_sqrt_x_minus_4_squared_l337_33736


namespace NUMINAMATH_CALUDE_joan_picked_apples_l337_33714

/-- The number of apples Joan has now -/
def total_apples : ℕ := 70

/-- The number of apples Melanie gave to Joan -/
def melanie_apples : ℕ := 27

/-- The number of apples Joan picked from the orchard -/
def orchard_apples : ℕ := total_apples - melanie_apples

theorem joan_picked_apples : orchard_apples = 43 := by
  sorry

end NUMINAMATH_CALUDE_joan_picked_apples_l337_33714


namespace NUMINAMATH_CALUDE_sum_of_integers_l337_33726

theorem sum_of_integers (p q r s t : ℤ) : 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧
  q ≠ r ∧ q ≠ s ∧ q ≠ t ∧
  r ≠ s ∧ r ≠ t ∧
  s ≠ t ∧
  (7 - p) * (7 - q) * (7 - r) * (7 - s) * (7 - t) = -120 →
  p + q + r + s + t = 22 := by
sorry

end NUMINAMATH_CALUDE_sum_of_integers_l337_33726
