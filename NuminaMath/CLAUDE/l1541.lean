import Mathlib

namespace NUMINAMATH_CALUDE_gcf_36_60_90_l1541_154194

theorem gcf_36_60_90 : Nat.gcd 36 (Nat.gcd 60 90) = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcf_36_60_90_l1541_154194


namespace NUMINAMATH_CALUDE_dance_event_women_count_l1541_154106

/-- Proves that given the conditions of the dance event, the number of women who attended is 12 -/
theorem dance_event_women_count :
  ∀ (num_men : ℕ) (men_partners : ℕ) (women_partners : ℕ),
    num_men = 9 →
    men_partners = 4 →
    women_partners = 3 →
    ∃ (num_women : ℕ),
      num_women * women_partners = num_men * men_partners ∧
      num_women = 12 := by
  sorry

end NUMINAMATH_CALUDE_dance_event_women_count_l1541_154106


namespace NUMINAMATH_CALUDE_player_one_winning_strategy_l1541_154162

-- Define the chessboard
def Chessboard : Type := Fin 8 × Fin 8

-- Define the distance between two points on the chessboard
def distance (p1 p2 : Chessboard) : ℝ := sorry

-- Define a valid move
def validMove (prev curr next : Chessboard) : Prop :=
  distance curr next > distance prev curr

-- Define the game state
structure GameState :=
  (position : Chessboard)
  (lastMove : Option Chessboard)
  (playerTurn : Bool)  -- true for Player One, false for Player Two

-- Define the winning condition for Player One
def playerOneWins (game : GameState) : Prop :=
  ∀ move : Chessboard, ¬validMove (Option.getD game.lastMove game.position) game.position move

-- Theorem: Player One has a winning strategy
theorem player_one_winning_strategy :
  ∃ (strategy : GameState → Chessboard),
    ∀ (game : GameState),
      game.playerTurn → 
      validMove (Option.getD game.lastMove game.position) game.position (strategy game) ∧
      playerOneWins {
        position := strategy game,
        lastMove := some game.position,
        playerTurn := false
      } := sorry

end NUMINAMATH_CALUDE_player_one_winning_strategy_l1541_154162


namespace NUMINAMATH_CALUDE_number_problem_l1541_154160

theorem number_problem (N : ℝ) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N = 20 → (40/100 : ℝ) * N = 240 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1541_154160


namespace NUMINAMATH_CALUDE_expected_outcome_is_negative_two_thirds_l1541_154111

/-- Represents the sides of the die --/
inductive DieSide
| A
| B
| C

/-- The probability of rolling each side of the die --/
def probability (side : DieSide) : ℚ :=
  match side with
  | DieSide.A => 1/3
  | DieSide.B => 1/2
  | DieSide.C => 1/6

/-- The monetary outcome of rolling each side of the die --/
def monetaryOutcome (side : DieSide) : ℚ :=
  match side with
  | DieSide.A => 2
  | DieSide.B => -4
  | DieSide.C => 6

/-- The expected monetary outcome of rolling the die --/
def expectedOutcome : ℚ :=
  (probability DieSide.A * monetaryOutcome DieSide.A) +
  (probability DieSide.B * monetaryOutcome DieSide.B) +
  (probability DieSide.C * monetaryOutcome DieSide.C)

/-- Theorem stating that the expected monetary outcome is -2/3 --/
theorem expected_outcome_is_negative_two_thirds :
  expectedOutcome = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_expected_outcome_is_negative_two_thirds_l1541_154111


namespace NUMINAMATH_CALUDE_no_integer_cube_equal_3n2_plus_3n_plus_7_l1541_154178

theorem no_integer_cube_equal_3n2_plus_3n_plus_7 :
  ¬ ∃ (n m : ℤ), m^3 = 3*n^2 + 3*n + 7 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_cube_equal_3n2_plus_3n_plus_7_l1541_154178


namespace NUMINAMATH_CALUDE_f_bounds_f_inequality_solution_set_l1541_154117

def f (x : ℝ) : ℝ := |x - 2| - |x - 5|

theorem f_bounds : ∀ x : ℝ, -3 ≤ f x ∧ f x ≤ 3 := by sorry

theorem f_inequality_solution_set :
  {x : ℝ | f x ≥ x^2 - 8*x + 15} = {x : ℝ | 5 - Real.sqrt 3 ≤ x ∧ x ≤ 6} := by sorry

end NUMINAMATH_CALUDE_f_bounds_f_inequality_solution_set_l1541_154117


namespace NUMINAMATH_CALUDE_remainders_inequality_l1541_154139

theorem remainders_inequality (X Y M A B s t u : ℕ) : 
  X > Y →
  X % M = A →
  Y % M = B →
  X = Y + 8 →
  (X^2) % M = s →
  (Y^2) % M = t →
  ((A*B)^2) % M = u →
  (s ≠ t ∧ t ≠ u ∧ s ≠ u) :=
by sorry

end NUMINAMATH_CALUDE_remainders_inequality_l1541_154139


namespace NUMINAMATH_CALUDE_chocolate_probabilities_l1541_154195

theorem chocolate_probabilities (w1 n1 w2 n2 : ℕ) 
  (h1 : w1 ≤ n1) (h2 : w2 ≤ n2) (h3 : n1 > 0) (h4 : n2 > 0) :
  ∃ (w1' n1' w2' n2' : ℕ),
    w1' ≤ n1' ∧ w2' ≤ n2' ∧ n1' > 0 ∧ n2' > 0 ∧
    (w1' : ℚ) / n1' = (w1 + w2 : ℚ) / (n1 + n2) ∧
  ∃ (w1'' n1'' w2'' n2'' : ℕ),
    w1'' ≤ n1'' ∧ w2'' ≤ n2'' ∧ n1'' > 0 ∧ n2'' > 0 ∧
    ¬((w1'' : ℚ) / n1'' < (w1'' + w2'' : ℚ) / (n1'' + n2'') ∧
      (w1'' + w2'' : ℚ) / (n1'' + n2'') < (w2'' : ℚ) / n2'') :=
by sorry

end NUMINAMATH_CALUDE_chocolate_probabilities_l1541_154195


namespace NUMINAMATH_CALUDE_candidate_count_l1541_154149

theorem candidate_count (total : ℕ) : 
  (total * 6 / 100 : ℕ) + 84 = total * 7 / 100 → total = 8400 := by
  sorry

end NUMINAMATH_CALUDE_candidate_count_l1541_154149


namespace NUMINAMATH_CALUDE_scientific_notation_of_seven_nm_l1541_154114

-- Define the value of 7nm in meters
def seven_nm : ℝ := 0.000000007

-- Theorem to prove the scientific notation
theorem scientific_notation_of_seven_nm :
  ∃ (a : ℝ) (n : ℤ), seven_nm = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 7 ∧ n = -9 :=
by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_seven_nm_l1541_154114


namespace NUMINAMATH_CALUDE_hotel_air_conditioning_l1541_154174

theorem hotel_air_conditioning (total_rooms : ℚ) : 
  total_rooms > 0 →
  (3 / 4 : ℚ) * total_rooms + (1 / 4 : ℚ) * total_rooms = total_rooms →
  (3 / 5 : ℚ) * total_rooms = total_rooms * (3 / 5 : ℚ) →
  (2 / 3 : ℚ) * ((3 / 5 : ℚ) * total_rooms) = (2 / 5 : ℚ) * total_rooms →
  let rented_rooms := (3 / 4 : ℚ) * total_rooms
  let non_rented_rooms := total_rooms - rented_rooms
  let ac_rooms := (3 / 5 : ℚ) * total_rooms
  let rented_ac_rooms := (2 / 5 : ℚ) * total_rooms
  let non_rented_ac_rooms := ac_rooms - rented_ac_rooms
  (non_rented_ac_rooms / non_rented_rooms) * 100 = 80 := by
sorry


end NUMINAMATH_CALUDE_hotel_air_conditioning_l1541_154174


namespace NUMINAMATH_CALUDE_line_tangent_to_parabola_l1541_154125

/-- A line is tangent to a parabola if and only if the discriminant of the resulting quadratic equation is zero -/
theorem line_tangent_to_parabola (k : ℝ) :
  (∃ x y : ℝ, 3 * x + 5 * y + k = 0 ∧ y^2 = 24 * x ∧
    ∀ x' y' : ℝ, 3 * x' + 5 * y' + k = 0 ∧ y'^2 = 24 * x' → (x', y') = (x, y))
  ↔ k = 50 := by
  sorry

end NUMINAMATH_CALUDE_line_tangent_to_parabola_l1541_154125


namespace NUMINAMATH_CALUDE_union_condition_intersection_condition_l1541_154127

-- Define sets A and B
def A : Set ℝ := {x : ℝ | -2 < x ∧ x < 5}
def B (m : ℝ) : Set ℝ := {x : ℝ | m < x ∧ x < m + 3}

-- Theorem for part (I)
theorem union_condition (m : ℝ) :
  A ∪ B m = A ↔ m ∈ Set.Icc (-2) 2 := by sorry

-- Theorem for part (II)
theorem intersection_condition (m : ℝ) :
  (A ∩ B m).Nonempty ↔ m ∈ Set.Ioo (-5) 2 := by sorry

end NUMINAMATH_CALUDE_union_condition_intersection_condition_l1541_154127


namespace NUMINAMATH_CALUDE_negative_x_positive_l1541_154180

theorem negative_x_positive (x : ℝ) (h : x < 0) : -x > 0 := by
  sorry

end NUMINAMATH_CALUDE_negative_x_positive_l1541_154180


namespace NUMINAMATH_CALUDE_barbell_cost_increase_l1541_154155

theorem barbell_cost_increase (old_cost new_cost : ℝ) (h1 : old_cost = 250) (h2 : new_cost = 325) :
  (new_cost - old_cost) / old_cost * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_barbell_cost_increase_l1541_154155


namespace NUMINAMATH_CALUDE_max_planes_from_points_l1541_154107

/-- The number of points in space -/
def num_points : ℕ := 15

/-- A function that calculates the number of ways to choose k elements from n elements -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

/-- The theorem stating the maximum number of planes determined by the points -/
theorem max_planes_from_points :
  choose num_points 3 = 455 := by sorry

end NUMINAMATH_CALUDE_max_planes_from_points_l1541_154107


namespace NUMINAMATH_CALUDE_total_coins_is_twelve_l1541_154130

def coins_distribution (x : ℕ) : ℕ × ℕ := 
  (x * (x + 1) / 2, x / 2)

theorem total_coins_is_twelve :
  ∃ x : ℕ, 
    x > 0 ∧ 
    let (pete_coins, paul_coins) := coins_distribution x
    pete_coins = 5 * paul_coins ∧
    pete_coins + paul_coins = 12 := by
  sorry

end NUMINAMATH_CALUDE_total_coins_is_twelve_l1541_154130


namespace NUMINAMATH_CALUDE_bead_mixing_problem_l1541_154133

/-- Proves that the total number of boxes is 8 given the conditions of the bead mixing problem. -/
theorem bead_mixing_problem (red_cost yellow_cost mixed_cost : ℚ) 
  (boxes_per_color : ℕ) : 
  red_cost = 13/10 ∧ 
  yellow_cost = 2 ∧ 
  mixed_cost = 43/25 ∧ 
  boxes_per_color = 4 → 
  (red_cost * boxes_per_color + yellow_cost * boxes_per_color) / 
    (2 * boxes_per_color) = mixed_cost ∧
  2 * boxes_per_color = 8 := by
  sorry

end NUMINAMATH_CALUDE_bead_mixing_problem_l1541_154133


namespace NUMINAMATH_CALUDE_sarah_walked_4_6_miles_l1541_154109

/-- The distance Sarah walked in miles -/
def sarah_distance (blocks_west : ℕ) (blocks_south : ℕ) (miles_per_block : ℚ) : ℚ :=
  (blocks_west + blocks_south : ℚ) * miles_per_block

/-- Theorem stating that Sarah walked 4.6 miles -/
theorem sarah_walked_4_6_miles :
  sarah_distance 8 15 (1/5) = 46/10 := by
  sorry

end NUMINAMATH_CALUDE_sarah_walked_4_6_miles_l1541_154109


namespace NUMINAMATH_CALUDE_average_age_problem_l1541_154102

theorem average_age_problem (a b c : ℕ) : 
  (a + c) / 2 = 29 →
  b = 17 →
  (a + b + c) / 3 = 25 := by
sorry

end NUMINAMATH_CALUDE_average_age_problem_l1541_154102


namespace NUMINAMATH_CALUDE_video_game_time_increase_l1541_154157

/-- Calculates the percentage increase in video game time given the original rate,
    total reading time, and additional time after raise. -/
theorem video_game_time_increase
  (original_rate : ℕ)  -- Original minutes of video game time per hour of reading
  (reading_time : ℕ)   -- Total hours of reading
  (additional_time : ℕ) -- Additional minutes of video game time after raise
  (h1 : original_rate = 30)
  (h2 : reading_time = 12)
  (h3 : additional_time = 72) :
  (additional_time : ℚ) / (original_rate * reading_time) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_video_game_time_increase_l1541_154157


namespace NUMINAMATH_CALUDE_chess_players_never_lost_to_ai_l1541_154190

theorem chess_players_never_lost_to_ai (total_players : ℕ) (players_lost : ℕ) :
  total_players = 40 →
  players_lost = 30 →
  (total_players - players_lost : ℚ) / total_players = 1/4 := by
sorry

end NUMINAMATH_CALUDE_chess_players_never_lost_to_ai_l1541_154190


namespace NUMINAMATH_CALUDE_magnitude_of_complex_square_root_l1541_154171

theorem magnitude_of_complex_square_root (w : ℂ) (h : w^2 = 48 - 14*I) : 
  Complex.abs w = 5 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_magnitude_of_complex_square_root_l1541_154171


namespace NUMINAMATH_CALUDE_problem_solution_l1541_154124

theorem problem_solution (a m n : ℚ) : 
  (∀ x, (a * x - 3) * (2 * x + 1) - 4 * x^2 + m = (a - 6) * x) → 
  a * n + m * n = 1 → 
  2 * n^3 - 9 * n^2 + 8 * n = 157 / 125 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1541_154124


namespace NUMINAMATH_CALUDE_rectangle_on_circle_l1541_154150

theorem rectangle_on_circle (R : ℝ) (x y : ℝ) :
  x^2 + y^2 = R^2 →
  x * y = (12 * R / 35) * (x + y) →
  ((x = 3 * R / 5 ∧ y = 4 * R / 5) ∨ (x = 4 * R / 5 ∧ y = 3 * R / 5)) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_on_circle_l1541_154150


namespace NUMINAMATH_CALUDE_horner_method_v₂_l1541_154165

def f (x : ℝ) : ℝ := x^5 + x^4 + 2*x^3 + 3*x^2 + 4*x + 1

def horner_v₀ (x : ℝ) : ℝ := 1

def horner_v₁ (x : ℝ) : ℝ := horner_v₀ x * x + 4

def horner_v₂ (x : ℝ) : ℝ := horner_v₁ x * x + 3

theorem horner_method_v₂ : horner_v₂ 2 = 15 := by sorry

end NUMINAMATH_CALUDE_horner_method_v₂_l1541_154165


namespace NUMINAMATH_CALUDE_soldier_average_score_l1541_154131

theorem soldier_average_score : 
  let shots : List ℕ := List.replicate 6 10 ++ [9] ++ List.replicate 3 8
  (shots.sum : ℚ) / shots.length = 93/10 := by
  sorry

end NUMINAMATH_CALUDE_soldier_average_score_l1541_154131


namespace NUMINAMATH_CALUDE_cookie_count_theorem_l1541_154152

/-- Represents a pack of cookies with a specific number of cookies -/
structure CookiePack where
  cookies : ℕ

/-- Represents a person's purchase of cookie packs -/
structure Purchase where
  packA : ℕ
  packB : ℕ
  packC : ℕ
  packD : ℕ

def packA : CookiePack := ⟨15⟩
def packB : CookiePack := ⟨30⟩
def packC : CookiePack := ⟨45⟩
def packD : CookiePack := ⟨60⟩

def paulPurchase : Purchase := ⟨1, 2, 0, 0⟩
def paulaPurchase : Purchase := ⟨1, 0, 1, 0⟩

def totalCookies (p : Purchase) : ℕ :=
  p.packA * packA.cookies + p.packB * packB.cookies + p.packC * packC.cookies + p.packD * packD.cookies

theorem cookie_count_theorem :
  totalCookies paulPurchase + totalCookies paulaPurchase = 135 := by
  sorry


end NUMINAMATH_CALUDE_cookie_count_theorem_l1541_154152


namespace NUMINAMATH_CALUDE_max_sin_C_in_triangle_l1541_154126

theorem max_sin_C_in_triangle (A B C : Real) (h : ∀ A B C, (1 / Real.tan A) + (1 / Real.tan B) = 6 / Real.tan C) :
  ∃ (max_sin_C : Real), max_sin_C = Real.sqrt 15 / 4 ∧ ∀ (sin_C : Real), sin_C ≤ max_sin_C := by
  sorry

end NUMINAMATH_CALUDE_max_sin_C_in_triangle_l1541_154126


namespace NUMINAMATH_CALUDE_license_plate_palindrome_probability_l1541_154135

/-- The number of possible letters in the license plate. -/
def num_letters : ℕ := 26

/-- The number of possible digits in the license plate. -/
def num_digits : ℕ := 10

/-- The length of the letter sequence in the license plate. -/
def letter_seq_length : ℕ := 4

/-- The length of the digit sequence in the license plate. -/
def digit_seq_length : ℕ := 4

/-- The probability of a license plate containing at least one palindrome. -/
def palindrome_probability : ℚ := 775 / 67600

theorem license_plate_palindrome_probability :
  let letter_palindrome_prob := 1 / (num_letters ^ 2 : ℚ)
  let digit_palindrome_prob := 1 / (num_digits ^ 2 : ℚ)
  let total_prob := letter_palindrome_prob + digit_palindrome_prob - 
                    (letter_palindrome_prob * digit_palindrome_prob)
  total_prob = palindrome_probability := by sorry

end NUMINAMATH_CALUDE_license_plate_palindrome_probability_l1541_154135


namespace NUMINAMATH_CALUDE_division_remainder_l1541_154129

theorem division_remainder : ∃ q : ℤ, 3021 = 97 * q + 14 ∧ 0 ≤ 14 ∧ 14 < 97 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_l1541_154129


namespace NUMINAMATH_CALUDE_probability_of_white_ball_l1541_154185

def initial_white_balls : ℕ := 8
def initial_black_balls : ℕ := 10
def balls_removed : ℕ := 2

theorem probability_of_white_ball :
  let total_balls := initial_white_balls + initial_black_balls
  let remaining_balls := total_balls - balls_removed
  ∃ (p : ℚ), p = 37/98 ∧ 
    (∀ (w b : ℕ), w + b = remaining_balls → 
      (w : ℚ) / (w + b : ℚ) ≤ p) ∧
    (∃ (w b : ℕ), w + b = remaining_balls ∧ 
      (w : ℚ) / (w + b : ℚ) = p) :=
by sorry


end NUMINAMATH_CALUDE_probability_of_white_ball_l1541_154185


namespace NUMINAMATH_CALUDE_line_relation_in_plane_l1541_154158

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and subset relations
variable (parallel : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)

-- Define the positional relationships between lines
inductive LineRelation : Type
  | parallel : LineRelation
  | skew : LineRelation
  | intersecting : LineRelation

-- Define the theorem
theorem line_relation_in_plane (a b : Line) (α : Plane) 
  (h1 : parallel a α) (h2 : subset b α) :
  (∃ r : LineRelation, r = LineRelation.parallel ∨ r = LineRelation.skew) ∧
  ¬(∃ r : LineRelation, r = LineRelation.intersecting) :=
sorry

end NUMINAMATH_CALUDE_line_relation_in_plane_l1541_154158


namespace NUMINAMATH_CALUDE_tea_price_correct_l1541_154148

/-- Represents the prices and quantities of tea in two purchases -/
structure TeaPurchases where
  first_quantity_A : ℕ
  first_quantity_B : ℕ
  first_total_cost : ℕ
  second_quantity_A : ℕ
  second_quantity_B : ℕ
  second_total_cost : ℕ
  price_increase : ℚ

/-- The solution to the tea pricing problem -/
def tea_price_solution (tp : TeaPurchases) : ℚ × ℚ :=
  (100, 200)

/-- Theorem stating that the given solution is correct for the specified tea purchases -/
theorem tea_price_correct (tp : TeaPurchases) 
  (h1 : tp.first_quantity_A = 30)
  (h2 : tp.first_quantity_B = 20)
  (h3 : tp.first_total_cost = 7000)
  (h4 : tp.second_quantity_A = 20)
  (h5 : tp.second_quantity_B = 15)
  (h6 : tp.second_total_cost = 6000)
  (h7 : tp.price_increase = 1/5) : 
  let (price_A, price_B) := tea_price_solution tp
  (tp.first_quantity_A : ℚ) * price_A + (tp.first_quantity_B : ℚ) * price_B = tp.first_total_cost ∧
  (tp.second_quantity_A : ℚ) * price_A * (1 + tp.price_increase) + 
  (tp.second_quantity_B : ℚ) * price_B * (1 + tp.price_increase) = tp.second_total_cost :=
by
  sorry

#check tea_price_correct

end NUMINAMATH_CALUDE_tea_price_correct_l1541_154148


namespace NUMINAMATH_CALUDE_sixth_term_is_36_l1541_154192

/-- The sequence of squares of natural numbers from 1 to 7 -/
def square_sequence : Fin 7 → ℕ := fun n => (n + 1)^2

/-- The 6th term of the square sequence is 36 -/
theorem sixth_term_is_36 : square_sequence 5 = 36 := by
  sorry

end NUMINAMATH_CALUDE_sixth_term_is_36_l1541_154192


namespace NUMINAMATH_CALUDE_parabola_theorem_l1541_154103

/-- Parabola structure -/
structure Parabola where
  a : ℝ
  h : a ≠ 0

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line in 2D space -/
structure Line where
  m : ℝ
  b : ℝ

/-- Definition of the parabola E: x^2 = 4y -/
def parabolaE : Parabola := ⟨4, by norm_num⟩

/-- Focus of the parabola -/
def focusF : Point := ⟨0, 1⟩

/-- Origin point -/
def originO : Point := ⟨0, 0⟩

/-- Function to calculate the area of a triangle given three points -/
noncomputable def triangleArea (p1 p2 p3 : Point) : ℝ := sorry

/-- Function to check if two lines are parallel -/
def isParallel (l1 l2 : Line) : Prop := sorry

/-- Theorem statement -/
theorem parabola_theorem (l : Line) (A B : Point) 
  (h1 : A.x^2 = 4 * A.y) -- A is on the parabola
  (h2 : B.x^2 = 4 * B.y) -- B is on the parabola
  (h3 : focusF.y = l.m * focusF.x + l.b) -- l passes through F
  (h4 : A.y = l.m * A.x + l.b) -- A is on l
  (h5 : B.y = l.m * B.x + l.b) -- B is on l
  :
  (∃ (minArea : ℝ), minArea = 2 ∧ 
    ∀ (A' B' : Point), A'.x^2 = 4 * A'.y → B'.x^2 = 4 * B'.y → 
    A'.y = l.m * A'.x + l.b → B'.y = l.m * B'.x + l.b →
    triangleArea originO A' B' ≥ minArea) ∧ 
  (∃ (C : Point) (lAO lBC : Line), 
    C.y = -1 ∧ -- C is on the directrix
    A.y = lAO.m * A.x + lAO.b ∧ -- AO line
    originO.y = lAO.m * originO.x + lAO.b ∧
    C.y = lAO.m * C.x + lAO.b ∧
    B.x = C.x ∧ -- BC is vertical
    isParallel lBC ⟨0, 1⟩) -- BC is parallel to y-axis
  := by sorry

end NUMINAMATH_CALUDE_parabola_theorem_l1541_154103


namespace NUMINAMATH_CALUDE_card_ratio_proof_l1541_154110

theorem card_ratio_proof (total_cards baseball_cards : ℕ) 
  (h1 : total_cards = 125)
  (h2 : baseball_cards = 95) : 
  (baseball_cards : ℚ) / (total_cards - baseball_cards) = 19 / 6 := by
  sorry

end NUMINAMATH_CALUDE_card_ratio_proof_l1541_154110


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l1541_154101

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) :
  (∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence condition
  a 3 = 3 →  -- given condition
  a 2 ^ 2 = a 1 * a 4 →  -- geometric sequence condition
  a 5 = 5 ∨ a 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l1541_154101


namespace NUMINAMATH_CALUDE_greatest_four_digit_divisible_by_55_and_11_l1541_154120

def reverse_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.foldl (λ acc d => acc * 10 + d) 0

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

theorem greatest_four_digit_divisible_by_55_and_11 :
  ∃ (m : ℕ), is_four_digit m ∧
             m % 55 = 0 ∧
             (reverse_digits m) % 55 = 0 ∧
             m % 11 = 0 ∧
             (∀ (n : ℕ), is_four_digit n →
                         n % 55 = 0 →
                         (reverse_digits n) % 55 = 0 →
                         n % 11 = 0 →
                         n ≤ m) ∧
             m = 5445 :=
sorry

end NUMINAMATH_CALUDE_greatest_four_digit_divisible_by_55_and_11_l1541_154120


namespace NUMINAMATH_CALUDE_product_of_reals_l1541_154121

theorem product_of_reals (a b : ℝ) (sum_eq : a + b = 8) (cube_sum_eq : a^3 + b^3 = 152) : a * b = 15 := by
  sorry

end NUMINAMATH_CALUDE_product_of_reals_l1541_154121


namespace NUMINAMATH_CALUDE_quadratic_equation_condition_l1541_154173

/-- The equation (m-4)x^|m-2| + 2x - 5 = 0 is quadratic if and only if m = 0 -/
theorem quadratic_equation_condition (m : ℝ) : 
  (∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, (m - 4) * x^(|m - 2|) + 2*x - 5 = a*x^2 + b*x + c) ↔ m = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_condition_l1541_154173


namespace NUMINAMATH_CALUDE_family_chips_consumption_l1541_154141

/-- Calculates the number of chocolate chips each family member eats given the following conditions:
  - Each batch contains 12 cookies
  - The family has 4 total people
  - Kendra made three batches
  - Each cookie contains 2 chocolate chips
  - All family members get the same number of cookies
-/
def chips_per_person (cookies_per_batch : ℕ) (family_size : ℕ) (batches : ℕ) (chips_per_cookie : ℕ) : ℕ :=
  let total_cookies := cookies_per_batch * batches
  let cookies_per_person := total_cookies / family_size
  cookies_per_person * chips_per_cookie

/-- Proves that given the conditions in the problem, each family member eats 18 chocolate chips -/
theorem family_chips_consumption :
  chips_per_person 12 4 3 2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_family_chips_consumption_l1541_154141


namespace NUMINAMATH_CALUDE_conical_hopper_volume_l1541_154134

/-- The volume of a conical hopper with given dimensions -/
theorem conical_hopper_volume :
  let diameter : ℝ := 10
  let radius : ℝ := diameter / 2
  let height : ℝ := 0.6 * radius
  let volume : ℝ := (1 / 3) * Real.pi * radius^2 * height
  volume = 25 * Real.pi := by sorry

end NUMINAMATH_CALUDE_conical_hopper_volume_l1541_154134


namespace NUMINAMATH_CALUDE_cheetah_catch_fox_l1541_154137

/-- Represents the cheetah's speed in meters per second -/
def cheetah_speed : ℝ := 4

/-- Represents the fox's speed in meters per second -/
def fox_speed : ℝ := 3

/-- Represents the initial distance between the cheetah and the fox in meters -/
def initial_distance : ℝ := 30

/-- Theorem stating that the cheetah will catch the fox after running 120 meters -/
theorem cheetah_catch_fox : 
  cheetah_speed * (initial_distance / (cheetah_speed - fox_speed)) = 120 :=
sorry

end NUMINAMATH_CALUDE_cheetah_catch_fox_l1541_154137


namespace NUMINAMATH_CALUDE_cell_phone_customers_l1541_154112

theorem cell_phone_customers (total : ℕ) (us_customers : ℕ) 
  (h1 : total = 7422) 
  (h2 : us_customers = 723) : 
  total - us_customers = 6699 := by
  sorry

end NUMINAMATH_CALUDE_cell_phone_customers_l1541_154112


namespace NUMINAMATH_CALUDE_right_triangle_condition_l1541_154184

/-- If sin γ - cos α = cos β in a triangle, then the triangle is right-angled -/
theorem right_triangle_condition (α β γ : ℝ) (h_triangle : α + β + γ = Real.pi) 
  (h_condition : Real.sin γ - Real.cos α = Real.cos β) : 
  α = Real.pi / 2 ∨ β = Real.pi / 2 ∨ γ = Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_condition_l1541_154184


namespace NUMINAMATH_CALUDE_least_integral_b_value_l1541_154140

theorem least_integral_b_value : 
  (∃ b : ℤ, (∀ x y : ℝ, (x^2 + y^2)^2 ≤ b * (x^4 + y^4)) ∧ 
   (∀ b' : ℤ, b' < b → ∃ x y : ℝ, (x^2 + y^2)^2 > b' * (x^4 + y^4))) → 
  (∃ b : ℤ, b = 2 ∧ 
   (∀ x y : ℝ, (x^2 + y^2)^2 ≤ b * (x^4 + y^4)) ∧ 
   (∀ b' : ℤ, b' < b → ∃ x y : ℝ, (x^2 + y^2)^2 > b' * (x^4 + y^4))) :=
by sorry

end NUMINAMATH_CALUDE_least_integral_b_value_l1541_154140


namespace NUMINAMATH_CALUDE_sqrt_450_simplification_l1541_154142

theorem sqrt_450_simplification : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_450_simplification_l1541_154142


namespace NUMINAMATH_CALUDE_net_income_calculation_l1541_154153

def calculate_net_income (spring_lawn : ℝ) (spring_garden : ℝ) (summer_lawn : ℝ) (summer_garden : ℝ)
  (fall_lawn : ℝ) (fall_garden : ℝ) (winter_snow : ℝ)
  (spring_lawn_supplies : ℝ) (spring_garden_supplies : ℝ)
  (summer_lawn_supplies : ℝ) (summer_garden_supplies : ℝ)
  (fall_lawn_supplies : ℝ) (fall_garden_supplies : ℝ)
  (winter_snow_supplies : ℝ)
  (advertising_percent : ℝ) (maintenance_percent : ℝ) : ℝ :=
  let total_earnings := spring_lawn + spring_garden + summer_lawn + summer_garden +
                        fall_lawn + fall_garden + winter_snow
  let total_supplies := spring_lawn_supplies + spring_garden_supplies +
                        summer_lawn_supplies + summer_garden_supplies +
                        fall_lawn_supplies + fall_garden_supplies +
                        winter_snow_supplies
  let total_gardening := spring_garden + summer_garden + fall_garden
  let total_lawn_mowing := spring_lawn + summer_lawn + fall_lawn
  let advertising_expenses := advertising_percent * total_gardening
  let maintenance_expenses := maintenance_percent * total_lawn_mowing
  total_earnings - total_supplies - advertising_expenses - maintenance_expenses

theorem net_income_calculation :
  calculate_net_income 200 150 600 450 300 350 100
                       80 50 150 100 75 75 25
                       0.15 0.10 = 1342.50 := by
  sorry

end NUMINAMATH_CALUDE_net_income_calculation_l1541_154153


namespace NUMINAMATH_CALUDE_perimeter_of_right_triangle_with_circles_l1541_154143

/-- A right triangle with inscribed circles -/
structure RightTriangleWithCircles where
  -- The side lengths of the triangle
  a : ℝ
  b : ℝ
  c : ℝ
  -- The radius of the inscribed circles
  r : ℝ
  -- Conditions
  right_triangle : a^2 + b^2 = c^2
  isosceles : a = b
  circle_radius : r = 2
  -- Relationship between side lengths and circle radius
  side_circle_relation : a = 4 * r

/-- The perimeter of a right triangle with inscribed circles -/
def perimeter (t : RightTriangleWithCircles) : ℝ :=
  t.a + t.b + t.c

/-- Theorem: The perimeter of the specified right triangle with inscribed circles is 16 + 8√2 -/
theorem perimeter_of_right_triangle_with_circles (t : RightTriangleWithCircles) :
  perimeter t = 16 + 8 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_perimeter_of_right_triangle_with_circles_l1541_154143


namespace NUMINAMATH_CALUDE_work_together_time_l1541_154147

/-- 
Calculates the time taken to complete a job when two people work together, 
given their individual completion times.
-/
def time_together (time_david time_john : ℚ) : ℚ :=
  1 / (1 / time_david + 1 / time_john)

/-- 
Theorem: If David completes a job in 5 days and John completes the same job in 9 days,
then the time taken to complete the job when they work together is 45/14 days.
-/
theorem work_together_time : time_together 5 9 = 45 / 14 := by
  sorry

end NUMINAMATH_CALUDE_work_together_time_l1541_154147


namespace NUMINAMATH_CALUDE_water_consumption_l1541_154181

theorem water_consumption (yesterday_amount : ℝ) (percentage_decrease : ℝ) 
  (h1 : yesterday_amount = 48)
  (h2 : percentage_decrease = 4)
  (h3 : yesterday_amount = (100 - percentage_decrease) / 100 * two_days_ago_amount) :
  two_days_ago_amount = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_water_consumption_l1541_154181


namespace NUMINAMATH_CALUDE_cubic_function_extrema_difference_l1541_154159

/-- A cubic function with specific properties -/
def f (a b c : ℝ) (x : ℝ) : ℝ := x^3 + 3*a*x^2 + 3*b*x + c

/-- The derivative of f -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 6*a*x + 3*b

/-- The second derivative of f -/
def f'' (a : ℝ) (x : ℝ) : ℝ := 6*x + 6*a

theorem cubic_function_extrema_difference (a b c : ℝ) :
  (f' a b 2 = 0) →  -- Extremum at x = 2
  (f' a b 1 = -3) →  -- Tangent line at x = 1 has slope -3 (parallel to 6x + 2y + 5 = 0)
  (∃ (x_max x_min : ℝ), 
    (∀ x, f a b c x ≤ f a b c x_max) ∧ 
    (∀ x, f a b c x ≥ f a b c x_min) ∧
    (f a b c x_max - f a b c x_min = 4)) := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_extrema_difference_l1541_154159


namespace NUMINAMATH_CALUDE_obtuse_triangle_side_range_l1541_154169

theorem obtuse_triangle_side_range (a : ℝ) : 
  (∃ (θ : ℝ), 
    -- Triangle inequality
    a > 1 ∧ 
    -- Obtuse triangle condition
    π/2 < θ ∧ 
    -- Largest angle doesn't exceed 120°
    θ ≤ 2*π/3 ∧ 
    -- Cosine law for the largest angle
    Real.cos θ = (a^2 + (a+1)^2 - (a+2)^2) / (2*a*(a+1))) 
  ↔ 
  (3/2 ≤ a ∧ a < 3) :=
by sorry

end NUMINAMATH_CALUDE_obtuse_triangle_side_range_l1541_154169


namespace NUMINAMATH_CALUDE_A_intersect_B_empty_l1541_154105

def A : Set ℝ := {x | x^2 + 2*x - 3 < 0}
def B : Set ℝ := {-3, 1, 2}

theorem A_intersect_B_empty : A ∩ B = ∅ := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_empty_l1541_154105


namespace NUMINAMATH_CALUDE_speedster_convertibles_count_l1541_154175

/-- Represents the inventory of an automobile company -/
structure Inventory where
  total : ℕ
  speedsters : ℕ
  speedsterConvertibles : ℕ

/-- Conditions of the inventory -/
def inventoryConditions (i : Inventory) : Prop :=
  i.speedsters = (2 * i.total) / 3 ∧
  i.speedsterConvertibles = (4 * i.speedsters) / 5 ∧
  i.total - i.speedsters = 50

/-- Theorem stating that under the given conditions, there are 80 Speedster convertibles -/
theorem speedster_convertibles_count (i : Inventory) :
  inventoryConditions i → i.speedsterConvertibles = 80 := by
  sorry

end NUMINAMATH_CALUDE_speedster_convertibles_count_l1541_154175


namespace NUMINAMATH_CALUDE_linear_function_proof_l1541_154183

/-- A linear function passing through (-2, -1) and parallel to y = 2x - 3 -/
def f (x : ℝ) : ℝ := 2 * x + 3

theorem linear_function_proof :
  (∀ x y : ℝ, f y - f x = 2 * (y - x)) ∧  -- linearity and slope
  f (-2) = -1 ∧                           -- passes through (-2, -1)
  (∀ x : ℝ, f x - (2 * x - 3) = 3) :=     -- parallel to y = 2x - 3
by sorry

end NUMINAMATH_CALUDE_linear_function_proof_l1541_154183


namespace NUMINAMATH_CALUDE_stadium_seats_l1541_154113

/-- Represents the number of seats in the little league stadium -/
def total_seats : ℕ := sorry

/-- Represents the number of people who came to the game -/
def people_at_game : ℕ := 47

/-- Represents the number of people holding banners -/
def people_with_banners : ℕ := 38

/-- Represents the number of empty seats -/
def empty_seats : ℕ := 45

/-- Theorem stating that the total number of seats is equal to the sum of people at the game and empty seats -/
theorem stadium_seats : total_seats = people_at_game + empty_seats := by sorry

end NUMINAMATH_CALUDE_stadium_seats_l1541_154113


namespace NUMINAMATH_CALUDE_white_shirt_cost_is_25_l1541_154156

/-- Represents the t-shirt sale scenario -/
structure TShirtSale where
  totalShirts : ℕ
  saleTime : ℕ
  blackShirtCost : ℕ
  revenuePerMinute : ℕ

/-- Calculates the cost of white t-shirts given the sale conditions -/
def whiteShirtCost (sale : TShirtSale) : ℕ :=
  let totalRevenue := sale.revenuePerMinute * sale.saleTime
  let blackShirts := sale.totalShirts / 2
  let whiteShirts := sale.totalShirts / 2
  let blackRevenue := blackShirts * sale.blackShirtCost
  let whiteRevenue := totalRevenue - blackRevenue
  whiteRevenue / whiteShirts

/-- Theorem stating that the white t-shirt cost is $25 under the given conditions -/
theorem white_shirt_cost_is_25 (sale : TShirtSale) 
  (h1 : sale.totalShirts = 200)
  (h2 : sale.saleTime = 25)
  (h3 : sale.blackShirtCost = 30)
  (h4 : sale.revenuePerMinute = 220) :
  whiteShirtCost sale = 25 := by
  sorry

#eval whiteShirtCost { totalShirts := 200, saleTime := 25, blackShirtCost := 30, revenuePerMinute := 220 }

end NUMINAMATH_CALUDE_white_shirt_cost_is_25_l1541_154156


namespace NUMINAMATH_CALUDE_cubic_root_sum_l1541_154193

theorem cubic_root_sum (r s t : ℝ) : 
  r^3 - 15*r^2 + 13*r - 8 = 0 →
  s^3 - 15*s^2 + 13*s - 8 = 0 →
  t^3 - 15*t^2 + 13*t - 8 = 0 →
  r / (1/r + s*t) + s / (1/s + r*t) + t / (1/t + r*s) = 199/9 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l1541_154193


namespace NUMINAMATH_CALUDE_coin_division_problem_l1541_154145

theorem coin_division_problem : ∃ n : ℕ,
  (∀ m : ℕ, m > n → (m % 8 ≠ 6 ∨ m % 7 ≠ 5)) ∧
  n % 8 = 6 ∧
  n % 7 = 5 ∧
  n % 9 = 0 :=
sorry

end NUMINAMATH_CALUDE_coin_division_problem_l1541_154145


namespace NUMINAMATH_CALUDE_square_root_sum_l1541_154115

theorem square_root_sum (a b : ℕ+) : 
  (Real.sqrt (7 + a / b) = 7 * Real.sqrt (a / b)) → a + b = 55 := by
  sorry

end NUMINAMATH_CALUDE_square_root_sum_l1541_154115


namespace NUMINAMATH_CALUDE_chase_theorem_l1541_154176

/-- Represents the chase scenario between a greyhound and a rabbit. -/
structure ChaseScenario where
  n : ℕ  -- Initial lead of the rabbit in rabbit hops
  a : ℕ  -- Number of rabbit hops
  b : ℕ  -- Number of greyhound hops
  c : ℕ  -- Equivalent rabbit hops
  d : ℕ  -- Greyhound hops

/-- Calculates the number of hops the rabbit can make before being caught. -/
def rabbit_hops (scenario : ChaseScenario) : ℚ :=
  (scenario.a * scenario.d * scenario.n : ℚ) / (scenario.b * scenario.c - scenario.a * scenario.d)

/-- Calculates the number of hops the greyhound makes before catching the rabbit. -/
def greyhound_hops (scenario : ChaseScenario) : ℚ :=
  (scenario.b * scenario.d * scenario.n : ℚ) / (scenario.b * scenario.c - scenario.a * scenario.d)

/-- Theorem stating the correctness of the chase calculations. -/
theorem chase_theorem (scenario : ChaseScenario) 
  (h : scenario.b * scenario.c ≠ scenario.a * scenario.d) : 
  rabbit_hops scenario * (scenario.b * scenario.c : ℚ) / (scenario.a * scenario.d) = 
  greyhound_hops scenario * (scenario.c : ℚ) / scenario.d + scenario.n := by
  sorry

end NUMINAMATH_CALUDE_chase_theorem_l1541_154176


namespace NUMINAMATH_CALUDE_same_solution_equations_l1541_154186

theorem same_solution_equations (x b : ℝ) : 
  (2 * x + 7 = 3) ∧ (b * x - 10 = -2) → b = -4 := by
  sorry

end NUMINAMATH_CALUDE_same_solution_equations_l1541_154186


namespace NUMINAMATH_CALUDE_polynomial_root_properties_l1541_154198

def P (x p : ℂ) : ℂ := x^4 + 3*x^3 + 3*x + p

theorem polynomial_root_properties (p : ℝ) (x₁ : ℂ) 
  (h1 : Complex.abs x₁ = 1)
  (h2 : 2 * Complex.re x₁ = (Real.sqrt 17 - 3) / 2)
  (h3 : P x₁ p = 0) :
  p = -1 - 3 * x₁^3 - 3 * x₁ ∧
  x₁ = Complex.mk ((Real.sqrt 17 - 3) / 4) (Real.sqrt ((3 * Real.sqrt 17 - 5) / 8)) ∧
  ∀ n : ℕ+, x₁^(n : ℕ) ≠ 1 := by
sorry

end NUMINAMATH_CALUDE_polynomial_root_properties_l1541_154198


namespace NUMINAMATH_CALUDE_survey_result_l1541_154179

theorem survey_result (total_participants : ℕ) 
  (property_damage_believers : ℕ) 
  (electrical_fire_believers : ℕ) :
  (property_damage_believers : ℚ) / (total_participants : ℚ) = 784 / 1000 →
  (electrical_fire_believers : ℚ) / (property_damage_believers : ℚ) = 525 / 1000 →
  electrical_fire_believers = 31 →
  total_participants = 75 := by
sorry

end NUMINAMATH_CALUDE_survey_result_l1541_154179


namespace NUMINAMATH_CALUDE_smallest_arithmetic_mean_of_nine_consecutive_naturals_l1541_154136

theorem smallest_arithmetic_mean_of_nine_consecutive_naturals (n : ℕ) : 
  (∀ k : ℕ, k ∈ Finset.range 9 → (n + k) > 0) →
  (((List.range 9).map (λ k => n + k)).prod) % 1111 = 0 →
  (((List.range 9).map (λ k => n + k)).sum / 9 : ℚ) ≥ 97 :=
by sorry

end NUMINAMATH_CALUDE_smallest_arithmetic_mean_of_nine_consecutive_naturals_l1541_154136


namespace NUMINAMATH_CALUDE_limit_rational_power_to_one_l1541_154199

theorem limit_rational_power_to_one (a : ℝ) (h : a > 0) :
  ∀ (x : ℚ → ℝ), (∀ ε > 0, ∃ N, ∀ n ≥ N, |x n| < ε) →
    ∀ ε > 0, ∃ N, ∀ n ≥ N, |a^(x n) - 1| < ε :=
by sorry

end NUMINAMATH_CALUDE_limit_rational_power_to_one_l1541_154199


namespace NUMINAMATH_CALUDE_distinct_colorings_l1541_154132

/-- Represents the symmetry group of a regular decagon -/
def DecagonSymmetryGroup : Type := Unit

/-- The order of the decagon symmetry group -/
def decagon_symmetry_order : ℕ := 10

/-- The number of disks in the decagon -/
def total_disks : ℕ := 10

/-- The number of disks to be colored -/
def colored_disks : ℕ := 8

/-- The number of blue disks -/
def blue_disks : ℕ := 4

/-- The number of red disks -/
def red_disks : ℕ := 3

/-- The number of green disks -/
def green_disks : ℕ := 2

/-- The number of yellow disks -/
def yellow_disks : ℕ := 1

/-- The total number of colorings without considering symmetry -/
def total_colorings : ℕ := (total_disks.choose blue_disks) * 
                           ((total_disks - blue_disks).choose red_disks) * 
                           ((total_disks - blue_disks - red_disks).choose green_disks) * 
                           ((total_disks - blue_disks - red_disks - green_disks).choose yellow_disks)

/-- The number of distinct colorings considering symmetry -/
theorem distinct_colorings : 
  (total_colorings / decagon_symmetry_order : ℚ) = 1260 := by sorry

end NUMINAMATH_CALUDE_distinct_colorings_l1541_154132


namespace NUMINAMATH_CALUDE_tv_show_payment_l1541_154189

theorem tv_show_payment (main_characters minor_characters : ℕ)
  (major_pay_ratio : ℕ) (total_payment : ℕ) :
  main_characters = 5 →
  minor_characters = 4 →
  major_pay_ratio = 3 →
  total_payment = 285000 →
  ∃ (minor_pay : ℕ),
    minor_pay = 15000 ∧
    minor_pay * (minor_characters + main_characters * major_pay_ratio) = total_payment :=
by sorry

end NUMINAMATH_CALUDE_tv_show_payment_l1541_154189


namespace NUMINAMATH_CALUDE_clock_angle_120_elapsed_time_l1541_154187

/-- Represents the angle between clock hands at a given time --/
def clockAngle (hours minutes : ℝ) : ℝ :=
  (30 * hours + 0.5 * minutes) - (6 * minutes)

/-- Finds the time when the clock hands form a 120° angle after 6:00 PM --/
def findNextAngle120 : ℝ :=
  let f := fun t : ℝ => abs (clockAngle (6 + t / 60) (t % 60) - 120)
  sorry -- Minimize f(t) for 0 ≤ t < 60

theorem clock_angle_120_elapsed_time :
  ∃ t : ℝ, 0 < t ∧ t < 60 ∧ 
  abs (clockAngle 6 0 - 120) < 0.01 ∧
  abs (clockAngle (6 + t / 60) (t % 60) - 120) < 0.01 ∧
  abs (t - 43.64) < 0.01 :=
sorry

end NUMINAMATH_CALUDE_clock_angle_120_elapsed_time_l1541_154187


namespace NUMINAMATH_CALUDE_parallel_vectors_ratio_l1541_154168

theorem parallel_vectors_ratio (θ : ℝ) : 
  let a : ℝ × ℝ := (Real.cos θ, Real.sin θ)
  let b : ℝ × ℝ := (1, 3)
  (∃ (k : ℝ), a.1 = k * b.1 ∧ a.2 = k * b.2) →
  (Real.sin θ + Real.cos θ) / (Real.sin θ - Real.cos θ) = 2 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_ratio_l1541_154168


namespace NUMINAMATH_CALUDE_multiplication_fraction_result_l1541_154154

theorem multiplication_fraction_result : 12 * (1 / 17) * 34 = 24 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_fraction_result_l1541_154154


namespace NUMINAMATH_CALUDE_specific_field_area_l1541_154161

/-- Represents a rectangular field with partial fencing -/
structure PartiallyFencedField where
  length : ℝ
  width : ℝ
  uncovered_side : ℝ
  fencing : ℝ

/-- Calculates the area of a rectangular field -/
def field_area (f : PartiallyFencedField) : ℝ :=
  f.length * f.width

/-- Theorem: The area of a specific partially fenced field is 390 square feet -/
theorem specific_field_area :
  ∃ (f : PartiallyFencedField),
    f.uncovered_side = 20 ∧
    f.fencing = 59 ∧
    f.length = f.uncovered_side ∧
    2 * f.width + f.uncovered_side = f.fencing ∧
    field_area f = 390 := by
  sorry

end NUMINAMATH_CALUDE_specific_field_area_l1541_154161


namespace NUMINAMATH_CALUDE_sqrt_360000_equals_600_l1541_154167

theorem sqrt_360000_equals_600 : Real.sqrt 360000 = 600 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_360000_equals_600_l1541_154167


namespace NUMINAMATH_CALUDE_min_value_product_l1541_154146

theorem min_value_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 4) :
  (3 * a + b) * (2 * b + 3 * c) * (a * c + 4) ≥ 384 ∧
  ∃ (a' b' c' : ℝ), a' > 0 ∧ b' > 0 ∧ c' > 0 ∧ a' * b' * c' = 4 ∧
    (3 * a' + b') * (2 * b' + 3 * c') * (a' * c' + 4) = 384 :=
by sorry

end NUMINAMATH_CALUDE_min_value_product_l1541_154146


namespace NUMINAMATH_CALUDE_power_inequality_l1541_154172

theorem power_inequality (a b : ℝ) : a^6 + b^6 ≥ a^4*b^2 + a^2*b^4 := by
  sorry

end NUMINAMATH_CALUDE_power_inequality_l1541_154172


namespace NUMINAMATH_CALUDE_circle_diameter_from_area_l1541_154116

theorem circle_diameter_from_area (A : ℝ) (r : ℝ) (d : ℝ) : 
  A = 64 * Real.pi → A = Real.pi * r^2 → d = 2 * r → d = 16 := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_from_area_l1541_154116


namespace NUMINAMATH_CALUDE_unique_function_satisfying_condition_l1541_154138

theorem unique_function_satisfying_condition :
  ∀ f : ℕ → ℕ,
    (f 1 > 0) →
    (∀ m n : ℕ, f (m^2 + 3*n^2) = (f m)^2 + 3*(f n)^2) →
    (∀ n : ℕ, f n = n) := by
  sorry

end NUMINAMATH_CALUDE_unique_function_satisfying_condition_l1541_154138


namespace NUMINAMATH_CALUDE_bracket_calculation_l1541_154164

-- Define the single bracket operation
def single_bracket (a b c : ℚ) : ℚ := (a + b) / c

-- Define the double bracket operation
def double_bracket (a b c d e f : ℚ) : ℚ := single_bracket (a + b) (d + e) (c + f)

-- State the theorem
theorem bracket_calculation :
  let result := single_bracket
    (double_bracket 10 20 30 40 30 70)
    (double_bracket 8 4 12 18 9 27)
    1
  result = 0.04 + 4/39 := by sorry

end NUMINAMATH_CALUDE_bracket_calculation_l1541_154164


namespace NUMINAMATH_CALUDE_difference_is_198_l1541_154104

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  units : Nat
  hundreds_lt_10 : hundreds < 10
  tens_lt_10 : tens < 10
  units_lt_10 : units < 10
  hundreds_gt_0 : hundreds > 0

/-- Condition: hundreds digit is 2 more than units digit -/
def hundreds_2_more_than_units (n : ThreeDigitNumber) : Prop :=
  n.hundreds = n.units + 2

/-- The value of the three-digit number -/
def value (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.units

/-- The reversed number -/
def reversed (n : ThreeDigitNumber) : Nat :=
  100 * n.units + 10 * n.tens + n.hundreds

/-- The main theorem -/
theorem difference_is_198 (n : ThreeDigitNumber) 
  (h : hundreds_2_more_than_units n) : 
  value n - reversed n = 198 := by
  sorry


end NUMINAMATH_CALUDE_difference_is_198_l1541_154104


namespace NUMINAMATH_CALUDE_bucket_radius_l1541_154144

/-- Proves that a cylindrical bucket with height 36 cm, when emptied to form a conical heap
    of height 12 cm and base radius 63 cm, has a radius of 21 cm. -/
theorem bucket_radius (h_cylinder h_cone r_cone : ℝ) 
    (h_cylinder_val : h_cylinder = 36)
    (h_cone_val : h_cone = 12)
    (r_cone_val : r_cone = 63)
    (volume_eq : π * r_cylinder^2 * h_cylinder = (1/3) * π * r_cone^2 * h_cone) :
  r_cylinder = 21 :=
sorry

end NUMINAMATH_CALUDE_bucket_radius_l1541_154144


namespace NUMINAMATH_CALUDE_polly_tweets_l1541_154166

/-- Represents the tweet rate (tweets per minute) for different states of Polly --/
structure TweetRate where
  happy : ℕ
  hungry : ℕ
  mirror : ℕ

/-- Represents the duration (in minutes) Polly spends in each state --/
structure Duration where
  happy : ℕ
  hungry : ℕ
  mirror : ℕ

/-- Calculates the total number of tweets given tweet rates and durations --/
def totalTweets (rate : TweetRate) (duration : Duration) : ℕ :=
  rate.happy * duration.happy + rate.hungry * duration.hungry + rate.mirror * duration.mirror

/-- Theorem stating that given the specific conditions, Polly tweets 1340 times --/
theorem polly_tweets : 
  ∀ (rate : TweetRate) (duration : Duration),
  rate.happy = 18 ∧ rate.hungry = 4 ∧ rate.mirror = 45 ∧
  duration.happy = 20 ∧ duration.hungry = 20 ∧ duration.mirror = 20 →
  totalTweets rate duration = 1340 := by
  sorry


end NUMINAMATH_CALUDE_polly_tweets_l1541_154166


namespace NUMINAMATH_CALUDE_mathborough_rainfall_2004_l1541_154196

/-- The total rainfall in Mathborough for the year 2004 -/
def total_rainfall_2004 (avg_rainfall_2003 : ℕ) (rainfall_increase : ℕ) : ℕ := 
  let avg_rainfall_2004 := avg_rainfall_2003 + rainfall_increase
  let feb_rainfall := avg_rainfall_2004 * 29
  let other_months_rainfall := avg_rainfall_2004 * 30 * 11
  feb_rainfall + other_months_rainfall

/-- Theorem stating the total rainfall in Mathborough for 2004 -/
theorem mathborough_rainfall_2004 : 
  total_rainfall_2004 50 3 = 19027 := by
sorry

end NUMINAMATH_CALUDE_mathborough_rainfall_2004_l1541_154196


namespace NUMINAMATH_CALUDE_cubic_odd_and_increasing_l1541_154191

-- Define the function f(x) = x³
def f (x : ℝ) : ℝ := x^3

-- State the theorem
theorem cubic_odd_and_increasing :
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x y, 0 < x → x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_cubic_odd_and_increasing_l1541_154191


namespace NUMINAMATH_CALUDE_ocean_depth_for_specific_mountain_l1541_154151

/-- Represents a cone-shaped mountain partially submerged in water -/
structure SubmergedMountain where
  height : ℝ
  aboveWaterVolumeFraction : ℝ

/-- Calculates the depth of the ocean at the base of a submerged mountain -/
def oceanDepth (m : SubmergedMountain) : ℝ :=
  m.height * (1 - (m.aboveWaterVolumeFraction ^ (1/3)))

/-- The theorem stating the ocean depth for a specific mountain -/
theorem ocean_depth_for_specific_mountain : 
  let m : SubmergedMountain := { height := 12000, aboveWaterVolumeFraction := 1/5 }
  oceanDepth m = 864 := by
  sorry

end NUMINAMATH_CALUDE_ocean_depth_for_specific_mountain_l1541_154151


namespace NUMINAMATH_CALUDE_complex_absolute_value_product_l1541_154170

theorem complex_absolute_value_product : 
  ∃ (z w : ℂ), z = 3 * Real.sqrt 5 - 5 * I ∧ w = 2 * Real.sqrt 2 + 4 * I ∧ 
  Complex.abs (z * w) = 8 * Real.sqrt 105 := by
  sorry

end NUMINAMATH_CALUDE_complex_absolute_value_product_l1541_154170


namespace NUMINAMATH_CALUDE_correct_equation_only_E_is_true_l1541_154188

theorem correct_equation : 15618 = 1 + 5^6 - 1 * 8 := by
  sorry

-- The following definitions represent the conditions from the original problem
def equation_A : Prop := 15614 = 1 + 5^6 - 1 * 4
def equation_B : Prop := 15615 = 1 + 5^6 - 1 * 5
def equation_C : Prop := 15616 = 1 + 5^6 - 1 * 6
def equation_D : Prop := 15617 = 1 + 5^6 - 1 * 7
def equation_E : Prop := 15618 = 1 + 5^6 - 1 * 8

-- This theorem states that equation_E is the only true equation among the given options
theorem only_E_is_true : 
  ¬equation_A ∧ ¬equation_B ∧ ¬equation_C ∧ ¬equation_D ∧ equation_E := by
  sorry

end NUMINAMATH_CALUDE_correct_equation_only_E_is_true_l1541_154188


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_reciprocals_l1541_154182

theorem quadratic_roots_sum_reciprocals (a b : ℝ) 
  (ha : a^2 + a - 1 = 0) (hb : b^2 + b - 1 = 0) : 
  a/b + b/a = 2 ∨ a/b + b/a = -3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_reciprocals_l1541_154182


namespace NUMINAMATH_CALUDE_book_words_per_page_l1541_154108

theorem book_words_per_page :
  ∀ (words_per_page : ℕ),
    words_per_page ≤ 120 →
    (150 * words_per_page) % 221 = 210 →
    words_per_page = 48 := by
  sorry

end NUMINAMATH_CALUDE_book_words_per_page_l1541_154108


namespace NUMINAMATH_CALUDE_same_solution_equations_l1541_154119

theorem same_solution_equations (c : ℝ) : 
  (∃ x : ℝ, 3 * x + 8 = 5 ∧ c * x - 7 = 1) → c = -8 := by
  sorry

end NUMINAMATH_CALUDE_same_solution_equations_l1541_154119


namespace NUMINAMATH_CALUDE_dart_board_probability_l1541_154100

/-- The probability of a dart landing within the center square of a regular hexagon dart board -/
theorem dart_board_probability (x : ℝ) (x_pos : x > 0) : 
  let hexagon_area := 3 * Real.sqrt 3 * x^2 / 2
  let square_area := 3 * x^2 / 4
  square_area / hexagon_area = 1 / (2 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_dart_board_probability_l1541_154100


namespace NUMINAMATH_CALUDE_digit_sum_equation_l1541_154163

theorem digit_sum_equation (a : ℕ) : a * 1000 + a * 998 + a * 999 = 22997 → a = 7 := by
  sorry

end NUMINAMATH_CALUDE_digit_sum_equation_l1541_154163


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1541_154122

theorem complex_fraction_simplification :
  2017 * (2016 / 2017) / (2019 * (1 / 2016)) + 1 / 2017 = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1541_154122


namespace NUMINAMATH_CALUDE_race_length_proof_l1541_154128

/-- The length of a race where runner A beats runner B by 35 meters in 7 seconds,
    and A's time over the course is 33 seconds. -/
def race_length : ℝ := 910

theorem race_length_proof :
  let time_A : ℝ := 33
  let lead_distance : ℝ := 35
  let lead_time : ℝ := 7
  race_length = (lead_distance * time_A) / (lead_time / time_A) := by
  sorry

#check race_length_proof

end NUMINAMATH_CALUDE_race_length_proof_l1541_154128


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_quadratic_l1541_154177

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  (a * r₁^2 + b * r₁ + c = 0) ∧ (a * r₂^2 + b * r₂ + c = 0) →
  r₁ + r₂ = -b / a :=
by sorry

theorem sum_of_roots_specific_quadratic :
  let r₁ := (4 + Real.sqrt (16 - 12)) / 2
  let r₂ := (4 - Real.sqrt (16 - 12)) / 2
  (r₁^2 - 4*r₁ + 3 = 0) ∧ (r₂^2 - 4*r₂ + 3 = 0) →
  r₁ + r₂ = 4 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_quadratic_l1541_154177


namespace NUMINAMATH_CALUDE_triangle_side_formulas_l1541_154123

/-- Given a triangle ABC with sides a, b, c, altitude m from A, and midline k from A,
    where b + c = 2l, prove the expressions for sides a, b, and c. -/
theorem triangle_side_formulas (a b c l m k : ℝ) : 
  b + c = 2 * l →
  k^2 = (b^2 + c^2) / 4 + (a / 2)^2 →
  m = (b * c) / a →
  b = l + Real.sqrt ((k^2 - m^2) * (k^2 - l^2) / (k^2 - m^2 - l^2)) ∧
  c = l - Real.sqrt ((k^2 - m^2) * (k^2 - l^2) / (k^2 - m^2 - l^2)) ∧
  a = 2 * l * Real.sqrt ((k^2 - l^2) / (k^2 - m^2 - l^2)) := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_formulas_l1541_154123


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1541_154197

/-- 
Given an arithmetic sequence with n + 2 terms, where the first term is x and the last term is y,
prove that the common difference is (y - x) / (n + 1).
-/
theorem arithmetic_sequence_common_difference 
  (n : ℕ) (x y : ℝ) : 
  let d := (y - x) / (n + 1)
  ∀ (a : Fin (n + 2) → ℝ), 
    (a 0 = x) → 
    (a (Fin.last (n + 1)) = y) → 
    (∀ i : Fin (n + 1), a i.succ - a i = d) → 
    d = (y - x) / (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1541_154197


namespace NUMINAMATH_CALUDE_treys_chores_l1541_154118

theorem treys_chores (clean_house_tasks : ℕ) (shower_tasks : ℕ) (dinner_tasks : ℕ) 
  (total_time_hours : ℕ) (h1 : clean_house_tasks = 7) (h2 : shower_tasks = 1) 
  (h3 : dinner_tasks = 4) (h4 : total_time_hours = 2) : 
  (total_time_hours * 60) / (clean_house_tasks + shower_tasks + dinner_tasks) = 10 := by
  sorry

end NUMINAMATH_CALUDE_treys_chores_l1541_154118
