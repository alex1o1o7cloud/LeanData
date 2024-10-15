import Mathlib

namespace NUMINAMATH_CALUDE_same_theme_probability_l1049_104985

/-- The probability of two students choosing the same theme out of two options -/
theorem same_theme_probability (themes : Nat) (students : Nat) : 
  themes = 2 → students = 2 → (themes^students / 2) / themes^students = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_same_theme_probability_l1049_104985


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l1049_104971

theorem rectangle_dimensions (w l : ℕ) : 
  l = w + 5 →
  2 * l + 2 * w = 34 →
  w = 6 ∧ l = 11 := by
sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l1049_104971


namespace NUMINAMATH_CALUDE_first_term_of_geometric_sequence_l1049_104906

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (s : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, s (n + 1) = s n * r

/-- Given a geometric sequence where the 6th term is 32 and the 7th term is 64, the first term is 1. -/
theorem first_term_of_geometric_sequence
  (s : ℕ → ℝ)
  (h_geometric : IsGeometricSequence s)
  (h_6th : s 6 = 32)
  (h_7th : s 7 = 64) :
  s 1 = 1 := by
  sorry


end NUMINAMATH_CALUDE_first_term_of_geometric_sequence_l1049_104906


namespace NUMINAMATH_CALUDE_angle_between_asymptotes_l1049_104910

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 / 3 = 1

-- Define the asymptotes
def asymptote1 (x y : ℝ) : Prop := y = Real.sqrt 3 * x
def asymptote2 (x y : ℝ) : Prop := y = -Real.sqrt 3 * x

-- Theorem statement
theorem angle_between_asymptotes :
  ∃ (θ : ℝ), θ = 60 * π / 180 ∧
  (∀ (x y : ℝ), hyperbola x y → 
    (asymptote1 x y ∨ asymptote2 x y) →
    ∃ (x1 y1 x2 y2 : ℝ), 
      asymptote1 x1 y1 ∧ asymptote2 x2 y2 ∧
      Real.cos θ = (x1 * x2 + y1 * y2) / 
        (Real.sqrt (x1^2 + y1^2) * Real.sqrt (x2^2 + y2^2))) :=
by sorry

end NUMINAMATH_CALUDE_angle_between_asymptotes_l1049_104910


namespace NUMINAMATH_CALUDE_complex_modulus_range_l1049_104924

theorem complex_modulus_range (z k : ℂ) (h1 : Complex.abs z = Complex.abs (1 + k * z)) (h2 : Complex.abs k < 1) :
  1 / (Complex.abs k + 1) ≤ Complex.abs z ∧ Complex.abs z ≤ 1 / (1 - Complex.abs k) :=
by sorry

end NUMINAMATH_CALUDE_complex_modulus_range_l1049_104924


namespace NUMINAMATH_CALUDE_alberto_bjorn_distance_difference_l1049_104973

-- Define the speeds and time
def alberto_speed : ℝ := 12
def bjorn_speed : ℝ := 9
def time : ℝ := 6

-- Theorem statement
theorem alberto_bjorn_distance_difference :
  alberto_speed * time - bjorn_speed * time = 18 := by
  sorry

end NUMINAMATH_CALUDE_alberto_bjorn_distance_difference_l1049_104973


namespace NUMINAMATH_CALUDE_g_approaches_neg_inf_pos_g_approaches_neg_inf_neg_l1049_104963

/-- The function g(x) = -3x^4 + 50x^2 - 1 -/
def g (x : ℝ) : ℝ := -3 * x^4 + 50 * x^2 - 1

/-- Theorem stating that g(x) approaches negative infinity as x approaches positive infinity -/
theorem g_approaches_neg_inf_pos (ε : ℝ) : ∃ M : ℝ, ∀ x : ℝ, x > M → g x < ε :=
sorry

/-- Theorem stating that g(x) approaches negative infinity as x approaches negative infinity -/
theorem g_approaches_neg_inf_neg (ε : ℝ) : ∃ M : ℝ, ∀ x : ℝ, x < -M → g x < ε :=
sorry

end NUMINAMATH_CALUDE_g_approaches_neg_inf_pos_g_approaches_neg_inf_neg_l1049_104963


namespace NUMINAMATH_CALUDE_equation_solution_l1049_104955

theorem equation_solution :
  let x : ℚ := 32
  let n : ℚ := -5/6
  35 - (23 - (15 - x)) = 12 * n / (1 / 2) := by sorry

end NUMINAMATH_CALUDE_equation_solution_l1049_104955


namespace NUMINAMATH_CALUDE_unique_positive_solution_l1049_104905

-- Define the polynomial function
def f (x : ℝ) : ℝ := x^12 + 8*x^11 + 15*x^10 + 2023*x^9 - 1500*x^8

-- State the theorem
theorem unique_positive_solution :
  ∃! (x : ℝ), x > 0 ∧ f x = 0 :=
sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l1049_104905


namespace NUMINAMATH_CALUDE_garden_length_perimeter_ratio_l1049_104958

/-- Proves that for a rectangular garden with length 24 feet and width 18 feet, 
    the ratio of its length to its perimeter is 2:7. -/
theorem garden_length_perimeter_ratio :
  let length : ℕ := 24
  let width : ℕ := 18
  let perimeter : ℕ := 2 * (length + width)
  (length : ℚ) / perimeter = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_garden_length_perimeter_ratio_l1049_104958


namespace NUMINAMATH_CALUDE_queue_adjustment_ways_l1049_104914

theorem queue_adjustment_ways (n m k : ℕ) (hn : n = 10) (hm : m = 3) (hk : k = 2) :
  (Nat.choose (n - m) k) * (m + 1) * (m + 2) = 420 := by
  sorry

end NUMINAMATH_CALUDE_queue_adjustment_ways_l1049_104914


namespace NUMINAMATH_CALUDE_vector_addition_l1049_104999

theorem vector_addition (A B C : ℝ × ℝ) : 
  (B.1 - A.1, B.2 - A.2) = (0, 1) →
  (C.1 - B.1, C.2 - B.2) = (1, 0) →
  (C.1 - A.1, C.2 - A.2) = (1, 1) := by
sorry

end NUMINAMATH_CALUDE_vector_addition_l1049_104999


namespace NUMINAMATH_CALUDE_negation_of_proposition_l1049_104948

theorem negation_of_proposition :
  (∀ x : ℝ, 2^x - 2*x - 2 ≥ 0) ↔ ¬(∃ x : ℝ, 2^x - 2*x - 2 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l1049_104948


namespace NUMINAMATH_CALUDE_jesse_book_reading_l1049_104962

theorem jesse_book_reading (total_pages : ℕ) (pages_read : ℕ) (pages_left : ℕ) : 
  pages_left = 166 → 
  pages_read = total_pages / 3 → 
  pages_left = 2 * total_pages / 3 → 
  pages_read = 83 := by
sorry

end NUMINAMATH_CALUDE_jesse_book_reading_l1049_104962


namespace NUMINAMATH_CALUDE_crayon_cost_l1049_104994

theorem crayon_cost (total_students : ℕ) (buyers : ℕ) (crayons_per_student : ℕ) (crayon_cost : ℕ) :
  total_students = 50 →
  buyers > total_students / 2 →
  buyers * crayons_per_student * crayon_cost = 1998 →
  crayon_cost > crayons_per_student →
  crayon_cost = 37 :=
by sorry

end NUMINAMATH_CALUDE_crayon_cost_l1049_104994


namespace NUMINAMATH_CALUDE_difference_of_squares_consecutive_evens_l1049_104965

def consecutive_even_integers (a b c : ℤ) : Prop :=
  b = a + 2 ∧ c = b + 2

theorem difference_of_squares_consecutive_evens (a b c : ℤ) :
  consecutive_even_integers a b c →
  a + b + c = 1992 →
  c^2 - a^2 = 5312 :=
by sorry

end NUMINAMATH_CALUDE_difference_of_squares_consecutive_evens_l1049_104965


namespace NUMINAMATH_CALUDE_y_order_l1049_104953

/-- Quadratic function f(x) = -x² + 4x - 5 -/
def f (x : ℝ) : ℝ := -x^2 + 4*x - 5

/-- Given three points on the graph of f -/
def A : ℝ × ℝ := (-4, f (-4))
def B : ℝ × ℝ := (-3, f (-3))
def C : ℝ × ℝ := (1, f 1)

/-- y-coordinates of the points -/
def y₁ : ℝ := A.2
def y₂ : ℝ := B.2
def y₃ : ℝ := C.2

theorem y_order : y₁ < y₂ ∧ y₂ < y₃ := by
  sorry

end NUMINAMATH_CALUDE_y_order_l1049_104953


namespace NUMINAMATH_CALUDE_complement_of_union_l1049_104990

def U : Set ℕ := {0, 1, 2, 3, 4}
def M : Set ℕ := {0, 4}
def N : Set ℕ := {2, 4}

theorem complement_of_union (U M N : Set ℕ) (hU : U = {0, 1, 2, 3, 4}) (hM : M = {0, 4}) (hN : N = {2, 4}) :
  U \ (M ∪ N) = {1, 3} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_union_l1049_104990


namespace NUMINAMATH_CALUDE_unique_integer_solution_range_l1049_104925

open Real

theorem unique_integer_solution_range (a : ℝ) : 
  (∃! (x : ℤ), (log (20 - 5 * (x : ℝ)^2) > log (a - (x : ℝ)) + 1)) ↔ 
  (2 ≤ a ∧ a < 5/2) :=
by sorry

end NUMINAMATH_CALUDE_unique_integer_solution_range_l1049_104925


namespace NUMINAMATH_CALUDE_towel_rate_calculation_l1049_104923

/-- Given the prices and quantities of towels, calculates the unknown rate. -/
def unknown_towel_rate (qty1 qty2 qty3 : ℕ) (price1 price2 avg_price : ℚ) : ℚ :=
  ((qty1 + qty2 + qty3 : ℚ) * avg_price - qty1 * price1 - qty2 * price2) / qty3

/-- Theorem stating that under the given conditions, the unknown rate is 325. -/
theorem towel_rate_calculation :
  let qty1 := 3
  let qty2 := 5
  let qty3 := 2
  let price1 := 100
  let price2 := 150
  let avg_price := 170
  unknown_towel_rate qty1 qty2 qty3 price1 price2 avg_price = 325 := by
sorry

end NUMINAMATH_CALUDE_towel_rate_calculation_l1049_104923


namespace NUMINAMATH_CALUDE_negative_division_result_l1049_104998

theorem negative_division_result : (-150) / (-25) = 6 := by
  sorry

end NUMINAMATH_CALUDE_negative_division_result_l1049_104998


namespace NUMINAMATH_CALUDE_three_digit_segment_sum_l1049_104947

/-- Represents the number of horizontal and vertical segments for a digit --/
structure DigitSegments where
  horizontal : Nat
  vertical : Nat

/-- The set of all digits and their corresponding segment counts --/
def digit_segments : Fin 10 → DigitSegments := fun d =>
  match d with
  | 0 => ⟨2, 4⟩
  | 1 => ⟨0, 2⟩
  | 2 => ⟨2, 3⟩
  | 3 => ⟨3, 3⟩
  | 4 => ⟨1, 3⟩
  | 5 => ⟨2, 2⟩
  | 6 => ⟨1, 3⟩
  | 7 => ⟨1, 2⟩
  | 8 => ⟨3, 4⟩
  | 9 => ⟨2, 3⟩

theorem three_digit_segment_sum :
  ∃ (a b c : Fin 10),
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (digit_segments a).horizontal + (digit_segments b).horizontal + (digit_segments c).horizontal = 5 ∧
    (digit_segments a).vertical + (digit_segments b).vertical + (digit_segments c).vertical = 10 ∧
    a.val + b.val + c.val = 9 :=
by sorry


end NUMINAMATH_CALUDE_three_digit_segment_sum_l1049_104947


namespace NUMINAMATH_CALUDE_soccer_goals_average_l1049_104929

theorem soccer_goals_average : 
  let players_with_3_goals : ℕ := 2
  let players_with_4_goals : ℕ := 3
  let players_with_5_goals : ℕ := 1
  let players_with_6_goals : ℕ := 1
  let total_goals : ℕ := 3 * players_with_3_goals + 4 * players_with_4_goals + 
                          5 * players_with_5_goals + 6 * players_with_6_goals
  let total_players : ℕ := players_with_3_goals + players_with_4_goals + 
                           players_with_5_goals + players_with_6_goals
  (total_goals : ℚ) / total_players = 29 / 7 := by
  sorry

end NUMINAMATH_CALUDE_soccer_goals_average_l1049_104929


namespace NUMINAMATH_CALUDE_ring_worth_proof_l1049_104927

theorem ring_worth_proof (total_worth car_cost : ℕ) (h1 : total_worth = 14000) (h2 : car_cost = 2000) :
  ∃ (ring_cost : ℕ), 
    ring_cost + car_cost + 2 * ring_cost = total_worth ∧ 
    ring_cost = 4000 := by
  sorry

end NUMINAMATH_CALUDE_ring_worth_proof_l1049_104927


namespace NUMINAMATH_CALUDE_product_of_reciprocals_l1049_104903

theorem product_of_reciprocals (a b : ℝ) : 
  a = 1 / (2 - Real.sqrt 3) → 
  b = 1 / (2 + Real.sqrt 3) → 
  a * b = 1 := by
sorry

end NUMINAMATH_CALUDE_product_of_reciprocals_l1049_104903


namespace NUMINAMATH_CALUDE_smallest_gcd_bc_l1049_104946

theorem smallest_gcd_bc (a b c : ℕ+) (h1 : Nat.gcd a b = 240) (h2 : Nat.gcd a c = 1001) :
  ∃ (b' c' : ℕ+), Nat.gcd b'.val c'.val = 1 ∧ 
    ∀ (b'' c'' : ℕ+), Nat.gcd a b''.val = 240 → Nat.gcd a c''.val = 1001 → 
      Nat.gcd b''.val c''.val ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_smallest_gcd_bc_l1049_104946


namespace NUMINAMATH_CALUDE_complex_sum_theorem_l1049_104939

theorem complex_sum_theorem (a b c : ℂ) 
  (h1 : a^2 + a*b + b^2 = 1)
  (h2 : b^2 + b*c + c^2 = -1)
  (h3 : c^2 + c*a + a^2 = Complex.I) :
  a*b + b*c + c*a = Complex.I ∨ a*b + b*c + c*a = -Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_theorem_l1049_104939


namespace NUMINAMATH_CALUDE_expression_simplification_l1049_104920

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 2 - 3) :
  (3 * x) / (x^2 - 9) * (1 - 3 / x) - 2 / (x + 3) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1049_104920


namespace NUMINAMATH_CALUDE_correct_factorization_l1049_104918

theorem correct_factorization (x y : ℝ) : x^3 + 4*x^2*y + 4*x*y^2 = x * (x + 2*y)^2 := by
  sorry

end NUMINAMATH_CALUDE_correct_factorization_l1049_104918


namespace NUMINAMATH_CALUDE_common_intersection_theorem_l1049_104976

/-- The common intersection point of a family of lines -/
def common_intersection_point : ℝ × ℝ := (-1, 1)

/-- The equation of the family of lines -/
def line_equation (a b d x y : ℝ) : Prop :=
  (b + d) * x + b * y = b + 2 * d

theorem common_intersection_theorem :
  ∀ (a b d : ℝ), line_equation a b d (common_intersection_point.1) (common_intersection_point.2) ∧
  (∀ (x y : ℝ), (∀ a b d : ℝ, line_equation a b d x y) → (x, y) = common_intersection_point) :=
sorry

end NUMINAMATH_CALUDE_common_intersection_theorem_l1049_104976


namespace NUMINAMATH_CALUDE_nine_rings_puzzle_l1049_104921

def min_moves : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | 2 => 2
  | (n + 3) => min_moves (n + 2) + 2 * min_moves (n + 1) + 1

theorem nine_rings_puzzle : min_moves 7 = 85 := by
  sorry

end NUMINAMATH_CALUDE_nine_rings_puzzle_l1049_104921


namespace NUMINAMATH_CALUDE_definite_integral_equals_six_ln_five_l1049_104997

theorem definite_integral_equals_six_ln_five :
  ∫ x in (π / 4)..(Real.arccos (1 / Real.sqrt 26)),
    36 / ((6 - Real.tan x) * Real.sin (2 * x)) = 6 * Real.log 5 := by
  sorry

end NUMINAMATH_CALUDE_definite_integral_equals_six_ln_five_l1049_104997


namespace NUMINAMATH_CALUDE_guise_hot_dog_consumption_l1049_104943

/-- Proves that given the conditions of Guise's hot dog consumption, the daily increase was 2 hot dogs. -/
theorem guise_hot_dog_consumption (monday_consumption : ℕ) (total_by_wednesday : ℕ) (daily_increase : ℕ) : 
  monday_consumption = 10 →
  total_by_wednesday = 36 →
  total_by_wednesday = monday_consumption + (monday_consumption + daily_increase) + (monday_consumption + 2 * daily_increase) →
  daily_increase = 2 := by
  sorry

end NUMINAMATH_CALUDE_guise_hot_dog_consumption_l1049_104943


namespace NUMINAMATH_CALUDE_distance_proof_l1049_104984

/-- The distance between two locations A and B, where two buses meet under specific conditions --/
def distance_between_locations : ℝ :=
  let first_meeting_distance : ℝ := 85
  let second_meeting_distance : ℝ := 65
  3 * first_meeting_distance - second_meeting_distance

theorem distance_proof :
  let first_meeting_distance : ℝ := 85
  let second_meeting_distance : ℝ := 65
  let total_distance := distance_between_locations
  (∃ (speed_A speed_B : ℝ), speed_A > 0 ∧ speed_B > 0 ∧
    first_meeting_distance / speed_A = (total_distance - first_meeting_distance) / speed_B ∧
    (total_distance - first_meeting_distance + second_meeting_distance) / speed_A + 0.5 =
    (first_meeting_distance + (total_distance - second_meeting_distance)) / speed_B + 0.5) →
  total_distance = 190 := by
  sorry

end NUMINAMATH_CALUDE_distance_proof_l1049_104984


namespace NUMINAMATH_CALUDE_equivalent_division_l1049_104951

theorem equivalent_division (x : ℝ) : (x / (3/9)) * (2/15) = x / 2.5 := by
  sorry

end NUMINAMATH_CALUDE_equivalent_division_l1049_104951


namespace NUMINAMATH_CALUDE_negation_is_returning_transformation_l1049_104959

theorem negation_is_returning_transformation (a : ℝ) : -(-a) = a := by
  sorry

end NUMINAMATH_CALUDE_negation_is_returning_transformation_l1049_104959


namespace NUMINAMATH_CALUDE_vector_relations_l1049_104988

def a : ℝ × ℝ := (2, 3)
def b : ℝ × ℝ := (-3, 1)

theorem vector_relations (k : ℝ) :
  (((k * a.1 + b.1, k * a.2 + b.2) • (a.1 - 3 * b.1, a.2 - 3 * b.2) = 0) → k = 3/2) ∧
  ((∃ t : ℝ, (k * a.1 + b.1, k * a.2 + b.2) = t • (a.1 - 3 * b.1, a.2 - 3 * b.2)) → k = -1/3) :=
by sorry

end NUMINAMATH_CALUDE_vector_relations_l1049_104988


namespace NUMINAMATH_CALUDE_geometric_decreasing_condition_l1049_104931

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

def is_decreasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) ≤ a n

theorem geometric_decreasing_condition (a : ℕ → ℝ) 
  (h_geom : is_geometric_sequence a) (h_pos : a 1 > 0) :
  (is_decreasing_sequence a → a 1 > a 2) ∧
  ¬(a 1 > a 2 → is_decreasing_sequence a) :=
sorry

end NUMINAMATH_CALUDE_geometric_decreasing_condition_l1049_104931


namespace NUMINAMATH_CALUDE_alice_lost_second_game_l1049_104979

/-- Represents a participant in the arm-wrestling contest -/
inductive Participant
  | Alice
  | Belle
  | Cathy

/-- Represents the state of a participant in a game -/
inductive GameState
  | Playing
  | Resting

/-- Represents the result of a game for a participant -/
inductive GameResult
  | Win
  | Lose

/-- The total number of games played -/
def totalGames : Nat := 21

/-- The number of times each participant played -/
def timesPlayed (p : Participant) : Nat :=
  match p with
  | Participant.Alice => 10
  | Participant.Belle => 15
  | Participant.Cathy => 17

/-- The state of a participant in a specific game -/
def participantState (p : Participant) (gameNumber : Nat) : GameState := sorry

/-- The result of a game for a participant -/
def gameResult (p : Participant) (gameNumber : Nat) : Option GameResult := sorry

theorem alice_lost_second_game :
  gameResult Participant.Alice 2 = some GameResult.Lose := by sorry

end NUMINAMATH_CALUDE_alice_lost_second_game_l1049_104979


namespace NUMINAMATH_CALUDE_joan_initial_money_l1049_104900

/-- The amount of money Joan had initially -/
def initial_money : ℕ := 60

/-- The cost of one container of hummus -/
def hummus_cost : ℕ := 5

/-- The number of hummus containers Joan buys -/
def hummus_quantity : ℕ := 2

/-- The cost of chicken -/
def chicken_cost : ℕ := 20

/-- The cost of bacon -/
def bacon_cost : ℕ := 10

/-- The cost of vegetables -/
def vegetable_cost : ℕ := 10

/-- The cost of one apple -/
def apple_cost : ℕ := 2

/-- The number of apples Joan can buy with remaining money -/
def apple_quantity : ℕ := 5

theorem joan_initial_money :
  initial_money = 
    hummus_cost * hummus_quantity + 
    chicken_cost + 
    bacon_cost + 
    vegetable_cost + 
    apple_cost * apple_quantity := by
  sorry

end NUMINAMATH_CALUDE_joan_initial_money_l1049_104900


namespace NUMINAMATH_CALUDE_chessboard_border_covering_l1049_104975

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Number of ways to cover a 2xn rectangle with 1x2 dominos -/
def cover_2xn (n : ℕ) : ℕ := fib (n + 1)

/-- Number of ways to cover the 2-unit wide border of an 8x8 chessboard with 1x2 dominos -/
def cover_chessboard_border : ℕ :=
  let f9 := cover_2xn 8
  let f10 := cover_2xn 9
  let f11 := cover_2xn 10
  2 + 2 * f11^2 * f9^2 + 12 * f11 * f10^2 * f9 + 2 * f10^4

theorem chessboard_border_covering :
  cover_chessboard_border = 146458404 := by sorry

end NUMINAMATH_CALUDE_chessboard_border_covering_l1049_104975


namespace NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l1049_104952

theorem tan_alpha_plus_pi_fourth (α : Real) 
  (h1 : α > Real.pi / 2) 
  (h2 : α < Real.pi) 
  (h3 : Real.sin α = 5 / 13) : 
  Real.tan (α + Real.pi / 4) = 7 / 17 := by
sorry

end NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l1049_104952


namespace NUMINAMATH_CALUDE_f_monotone_decreasing_on_interval_l1049_104956

noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 - Real.log x

theorem f_monotone_decreasing_on_interval :
  ∀ x₁ x₂ : ℝ, 0 < x₁ → 0 < x₂ → x₁ ≤ x₂ → x₂ ≤ 1 → f x₁ ≥ f x₂ := by
  sorry

end NUMINAMATH_CALUDE_f_monotone_decreasing_on_interval_l1049_104956


namespace NUMINAMATH_CALUDE_product_mod_seven_l1049_104968

theorem product_mod_seven : (2031 * 2032 * 2033 * 2034) % 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_seven_l1049_104968


namespace NUMINAMATH_CALUDE_speaking_orders_eq_600_l1049_104919

-- Define the total number of people in the group
def total_people : ℕ := 7

-- Define the number of people to be selected
def selected_people : ℕ := 4

-- Function to calculate the number of speaking orders
def speaking_orders : ℕ :=
  -- Case 1: Only one of leader or deputy participates
  (2 * (total_people - 2).choose (selected_people - 1) * (selected_people).factorial) +
  -- Case 2: Both leader and deputy participate (not adjacent)
  ((total_people - 2).choose (selected_people - 2) * selected_people.factorial -
   (total_people - 2).choose (selected_people - 2) * 2 * (selected_people - 1).factorial)

-- Theorem statement
theorem speaking_orders_eq_600 :
  speaking_orders = 600 :=
sorry

end NUMINAMATH_CALUDE_speaking_orders_eq_600_l1049_104919


namespace NUMINAMATH_CALUDE_triangle_trigonometric_identities_l1049_104912

theorem triangle_trigonometric_identities (A B C : ℝ) (a b c : ℝ) :
  (A + B + C = π) →
  (a = 2 * Real.sin A) →
  (b = 2 * Real.sin B) →
  (c = 2 * Real.sin C) →
  (((a^2 * Real.sin (B - C)) / (Real.sin B * Real.sin C) +
    (b^2 * Real.sin (C - A)) / (Real.sin C * Real.sin A) +
    (c^2 * Real.sin (A - B)) / (Real.sin A * Real.sin B) = 0) ∧
   ((a^2 * Real.sin (B - C)) / (Real.sin B + Real.sin C) +
    (b^2 * Real.sin (C - A)) / (Real.sin C + Real.sin A) +
    (c^2 * Real.sin (A - B)) / (Real.sin A + Real.sin B) = 0)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_trigonometric_identities_l1049_104912


namespace NUMINAMATH_CALUDE_nonagon_diagonals_octagon_diagonals_decagon_diagonals_l1049_104949

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: The number of diagonals in a nonagon is 27 -/
theorem nonagon_diagonals : num_diagonals 9 = 27 := by sorry

/-- Theorem: The number of diagonals in an octagon is 20 -/
theorem octagon_diagonals : num_diagonals 8 = 20 := by sorry

/-- Theorem: The number of diagonals in a decagon is 35 -/
theorem decagon_diagonals : num_diagonals 10 = 35 := by sorry

end NUMINAMATH_CALUDE_nonagon_diagonals_octagon_diagonals_decagon_diagonals_l1049_104949


namespace NUMINAMATH_CALUDE_equation_solution_difference_l1049_104960

theorem equation_solution_difference : ∃ (r₁ r₂ : ℝ),
  r₁ ≠ r₂ ∧
  (r₁ + 5 ≠ 0 ∧ r₂ + 5 ≠ 0) ∧
  ((r₁^2 - 5*r₁ - 24) / (r₁ + 5) = 3*r₁ + 8) ∧
  ((r₂^2 - 5*r₂ - 24) / (r₂ + 5) = 3*r₂ + 8) ∧
  |r₁ - r₂| = 4 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_difference_l1049_104960


namespace NUMINAMATH_CALUDE_survivor_same_tribe_quit_probability_l1049_104935

/-- The probability of all three quitters being from the same tribe in a Survivor-like scenario -/
theorem survivor_same_tribe_quit_probability :
  let total_people : ℕ := 20
  let tribe_size : ℕ := 10
  let num_quitters : ℕ := 3
  let total_combinations := Nat.choose total_people num_quitters
  let same_tribe_combinations := 2 * Nat.choose tribe_size num_quitters
  (same_tribe_combinations : ℚ) / total_combinations = 4 / 19 := by
  sorry

end NUMINAMATH_CALUDE_survivor_same_tribe_quit_probability_l1049_104935


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_value_l1049_104907

/-- Given a line l passing through points (a-2, -1) and (-a-2, 1), 
    and perpendicular to the line 2x+3y+1=0, prove that a = -2/3 -/
theorem perpendicular_lines_a_value (a : ℝ) : 
  let l : Set (ℝ × ℝ) := {p | ∃ t : ℝ, p = (a - 2 + t*(-2*a), -1 + t*2)}
  let slope_l := (1 - (-1)) / ((-a - 2) - (a - 2))
  let slope_other := -2 / 3
  (∀ p ∈ l, 2 * p.1 + 3 * p.2 + 1 ≠ 0) → 
  (slope_l * slope_other = -1) →
  a = -2/3 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_a_value_l1049_104907


namespace NUMINAMATH_CALUDE_smallest_value_in_range_l1049_104961

theorem smallest_value_in_range (x : ℝ) (h1 : -1 < x) (h2 : x < 0) :
  (1 / x < x) ∧ (1 / x < x^2) ∧ (1 / x < 2*x) ∧ (1 / x < Real.sqrt (x^2)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_value_in_range_l1049_104961


namespace NUMINAMATH_CALUDE_arccos_cos_eq_x_div_3_l1049_104957

theorem arccos_cos_eq_x_div_3 (x : ℝ) :
  -Real.pi ≤ x ∧ x ≤ 2 * Real.pi →
  (Real.arccos (Real.cos x) = x / 3 ↔ x = 0 ∨ x = 3 * Real.pi / 2 ∨ x = -3 * Real.pi / 2) :=
by sorry

end NUMINAMATH_CALUDE_arccos_cos_eq_x_div_3_l1049_104957


namespace NUMINAMATH_CALUDE_max_sum_consecutive_integers_with_product_constraint_l1049_104978

theorem max_sum_consecutive_integers_with_product_constraint : 
  ∀ n : ℕ, n * (n + 1) < 500 → n + (n + 1) ≤ 43 :=
by
  sorry

end NUMINAMATH_CALUDE_max_sum_consecutive_integers_with_product_constraint_l1049_104978


namespace NUMINAMATH_CALUDE_faster_train_speed_l1049_104930

/-- Proves that the speed of the faster train is 180 km/h given the problem conditions --/
theorem faster_train_speed
  (train1_length : ℝ)
  (train2_length : ℝ)
  (initial_distance : ℝ)
  (slower_train_speed : ℝ)
  (time_to_meet : ℝ)
  (h1 : train1_length = 100)
  (h2 : train2_length = 200)
  (h3 : initial_distance = 450)
  (h4 : slower_train_speed = 90)
  (h5 : time_to_meet = 9.99920006399488)
  : ∃ (faster_train_speed : ℝ), faster_train_speed = 180 := by
  sorry

#check faster_train_speed

end NUMINAMATH_CALUDE_faster_train_speed_l1049_104930


namespace NUMINAMATH_CALUDE_parabola_point_to_directrix_distance_l1049_104967

/-- Proves that for a parabola y² = 2px containing the point (1, √5), 
    the distance from this point to the directrix is 9/4 -/
theorem parabola_point_to_directrix_distance :
  ∀ (p : ℝ), 
  (5 : ℝ) = 2 * p →  -- Condition from y² = 2px with (1, √5)
  (1 : ℝ) + p / 2 = 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_to_directrix_distance_l1049_104967


namespace NUMINAMATH_CALUDE_circle_center_coord_sum_l1049_104970

theorem circle_center_coord_sum (x y : ℝ) :
  x^2 + y^2 = 4*x - 6*y + 9 →
  ∃ (center_x center_y : ℝ), center_x + center_y = -1 ∧
    ∀ (point_x point_y : ℝ),
      (point_x - center_x)^2 + (point_y - center_y)^2 = (x - center_x)^2 + (y - center_y)^2 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_coord_sum_l1049_104970


namespace NUMINAMATH_CALUDE_four_digit_number_l1049_104942

/-- Represents a 6x6 grid of numbers -/
def Grid := Matrix (Fin 6) (Fin 6) Nat

/-- Check if a number is within the range 1 to 6 -/
def inRange (n : Nat) : Prop := 1 ≤ n ∧ n ≤ 6

/-- Check if a list of numbers contains no duplicates -/
def noDuplicates (l : List Nat) : Prop := l.Nodup

/-- Check if a number is prime -/
def isPrime (n : Nat) : Prop := Nat.Prime n

/-- Check if a number is composite -/
def isComposite (n : Nat) : Prop := ¬(isPrime n) ∧ n > 1

/-- Theorem: Under the given conditions, the four-digit number is 4123 -/
theorem four_digit_number (g : Grid) 
  (range_check : ∀ i j, inRange (g i j))
  (row_unique : ∀ i, noDuplicates (List.ofFn (λ j => g i j)))
  (col_unique : ∀ j, noDuplicates (List.ofFn (λ i => g i j)))
  (rect_unique : ∀ i j, noDuplicates [g i j, g i (j+1), g i (j+2), g (i+1) j, g (i+1) (j+1), g (i+1) (j+2)])
  (circle_sum : ∀ i j, isComposite (g i j + g (i+1) j) → ∀ k l, (k, l) ≠ (i, j) → isPrime (g k l + g (k+1) l))
  : ∃ i j k l, g i j = 4 ∧ g k j = 1 ∧ g k l = 2 ∧ g i l = 3 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_number_l1049_104942


namespace NUMINAMATH_CALUDE_solve_for_m_l1049_104950

theorem solve_for_m : ∃ m : ℝ, 
  (∀ x : ℝ, x > 2 ↔ x - 3*m + 1 > 0) → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_m_l1049_104950


namespace NUMINAMATH_CALUDE_roses_cut_equals_difference_l1049_104902

/-- The number of roses Jessica cut from her garden -/
def roses_cut : ℕ := sorry

/-- The initial number of roses in the vase -/
def initial_roses : ℕ := 7

/-- The final number of roses in the vase -/
def final_roses : ℕ := 23

/-- Theorem stating that the number of roses Jessica cut is equal to the difference between the final and initial number of roses in the vase -/
theorem roses_cut_equals_difference : roses_cut = final_roses - initial_roses := by sorry

end NUMINAMATH_CALUDE_roses_cut_equals_difference_l1049_104902


namespace NUMINAMATH_CALUDE_min_value_expression_l1049_104993

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (8 * z) / (3 * x + 2 * y) + (8 * x) / (2 * y + 3 * z) + y / (x + z) ≥ 4.5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1049_104993


namespace NUMINAMATH_CALUDE_oplus_composition_l1049_104913

/-- Definition of the ⊕ operation -/
def oplus (x y : ℝ) : ℝ := x^2 + y

/-- Theorem stating that h ⊕ (h ⊕ h) = 2h^2 + h -/
theorem oplus_composition (h : ℝ) : oplus h (oplus h h) = 2 * h^2 + h := by
  sorry

end NUMINAMATH_CALUDE_oplus_composition_l1049_104913


namespace NUMINAMATH_CALUDE_fraction_inequality_l1049_104933

theorem fraction_inequality (a b x : ℝ) (ha : a > 0) (hb : b > 0) (hx : x > 0) (hab : a > b) :
  b / a < (b + x) / (a + x) := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l1049_104933


namespace NUMINAMATH_CALUDE_brians_largest_integer_l1049_104982

theorem brians_largest_integer (x : ℤ) : 
  (∀ y : ℤ, 10 ≤ 8*y - 70 ∧ 8*y - 70 ≤ 99 → y ≤ x) ↔ x = 21 :=
by sorry

end NUMINAMATH_CALUDE_brians_largest_integer_l1049_104982


namespace NUMINAMATH_CALUDE_inequality_proof_l1049_104989

theorem inequality_proof (a b c : ℝ) (n : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hn : n ≥ 2) (habc : a * b * c = 1) :
  (a / (b + c)^(1/n : ℝ)) + (b / (c + a)^(1/n : ℝ)) + (c / (a + b)^(1/n : ℝ)) ≥ 3 / (2^(1/n : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1049_104989


namespace NUMINAMATH_CALUDE_metallic_sheet_length_is_48_l1049_104954

/-- Represents the dimensions and properties of a metallic sheet and the box made from it. -/
structure MetallicSheet where
  width : ℝ
  cutSize : ℝ
  boxVolume : ℝ

/-- Calculates the length of the original metallic sheet given its properties. -/
def calculateLength (sheet : MetallicSheet) : ℝ :=
  sorry

/-- Theorem stating that for a sheet with width 36m, cut size 8m, and resulting box volume 5120m³,
    the original length is 48m. -/
theorem metallic_sheet_length_is_48 :
  let sheet : MetallicSheet := ⟨36, 8, 5120⟩
  calculateLength sheet = 48 := by
  sorry

end NUMINAMATH_CALUDE_metallic_sheet_length_is_48_l1049_104954


namespace NUMINAMATH_CALUDE_comprehensive_formula_l1049_104917

theorem comprehensive_formula (h1 : 12 * 5 = 60) (h2 : 60 - 42 = 18) :
  12 * 5 - 42 = 18 := by
  sorry

end NUMINAMATH_CALUDE_comprehensive_formula_l1049_104917


namespace NUMINAMATH_CALUDE_overtime_hours_calculation_l1049_104928

-- Define the hourly rates and total hours
def ordinary_rate : ℚ := 60 / 100
def overtime_rate : ℚ := 90 / 100
def total_hours : ℕ := 50

-- Define the total pay in dollars
def total_pay : ℚ := 3240 / 100

-- Theorem statement
theorem overtime_hours_calculation :
  ∃ (ordinary_hours overtime_hours : ℕ),
    ordinary_hours + overtime_hours = total_hours ∧
    ordinary_rate * ordinary_hours + overtime_rate * overtime_hours = total_pay ∧
    overtime_hours = 8 := by
  sorry

end NUMINAMATH_CALUDE_overtime_hours_calculation_l1049_104928


namespace NUMINAMATH_CALUDE_students_per_class_is_twenty_l1049_104909

/-- Represents a school with teachers, a principal, classes, and students. -/
structure School where
  teachers : ℕ
  principal : ℕ
  classes : ℕ
  total_people : ℕ
  students_per_class : ℕ

/-- Theorem stating that in a school with given parameters, there are 20 students in each class. -/
theorem students_per_class_is_twenty (school : School)
  (h1 : school.teachers = 48)
  (h2 : school.principal = 1)
  (h3 : school.classes = 15)
  (h4 : school.total_people = 349)
  (h5 : school.total_people = school.teachers + school.principal + school.classes * school.students_per_class) :
  school.students_per_class = 20 := by
  sorry

end NUMINAMATH_CALUDE_students_per_class_is_twenty_l1049_104909


namespace NUMINAMATH_CALUDE_parabola_point_order_l1049_104932

/-- A parabola with equation y = -(x-2)^2 + k -/
structure Parabola where
  k : ℝ

/-- A point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Predicate to check if a point lies on the parabola -/
def lies_on (p : Point) (para : Parabola) : Prop :=
  p.y = -(p.x - 2)^2 + para.k

theorem parabola_point_order (para : Parabola) 
  (A B C : Point)
  (hA : A.x = -2) (hB : B.x = -1) (hC : C.x = 3)
  (liesA : lies_on A para) (liesB : lies_on B para) (liesC : lies_on C para) :
  A.y < B.y ∧ B.y < C.y := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_order_l1049_104932


namespace NUMINAMATH_CALUDE_fixed_point_exponential_function_l1049_104916

theorem fixed_point_exponential_function 
  (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x-1) + 1
  f 1 = 2 := by
sorry

end NUMINAMATH_CALUDE_fixed_point_exponential_function_l1049_104916


namespace NUMINAMATH_CALUDE_lunch_cost_theorem_l1049_104908

/-- Calculates the total cost of lunches for a field trip --/
def total_lunch_cost (total_people : ℕ) (extra_lunches : ℕ) (vegetarian : ℕ) (gluten_free : ℕ) 
  (nut_free : ℕ) (halal : ℕ) (veg_and_gf : ℕ) (regular_cost : ℕ) (special_cost : ℕ) 
  (veg_gf_cost : ℕ) : ℕ :=
  let total_lunches := total_people + extra_lunches
  let regular_lunches := total_lunches - (vegetarian + gluten_free + nut_free + halal - veg_and_gf)
  let regular_total := regular_lunches * regular_cost
  let vegetarian_total := (vegetarian - veg_and_gf) * special_cost
  let gluten_free_total := gluten_free * special_cost
  let nut_free_total := nut_free * special_cost
  let halal_total := halal * special_cost
  let veg_gf_total := veg_and_gf * veg_gf_cost
  regular_total + vegetarian_total + gluten_free_total + nut_free_total + halal_total + veg_gf_total

theorem lunch_cost_theorem :
  total_lunch_cost 41 3 10 5 3 4 2 7 8 9 = 346 := by
  sorry

#eval total_lunch_cost 41 3 10 5 3 4 2 7 8 9

end NUMINAMATH_CALUDE_lunch_cost_theorem_l1049_104908


namespace NUMINAMATH_CALUDE_wedge_volume_l1049_104940

/-- The volume of a wedge cut from a cylindrical log --/
theorem wedge_volume (d h r : ℝ) (h1 : d = 20) (h2 : h = d) (h3 : r = d / 2) : 
  ∃ (m : ℕ), (1 / 3) * π * r^2 * h = m * π ∧ m = 667 := by
  sorry

end NUMINAMATH_CALUDE_wedge_volume_l1049_104940


namespace NUMINAMATH_CALUDE_square_property_contradiction_l1049_104980

theorem square_property_contradiction (property : ℝ → ℝ) 
  (h_prop : ∀ x : ℝ, property x = (x^2) * property 1) : 
  ¬ (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ b = 5 * a ∧ property b = 5 * property a) :=
sorry

end NUMINAMATH_CALUDE_square_property_contradiction_l1049_104980


namespace NUMINAMATH_CALUDE_function_passes_through_point_l1049_104944

/-- The function f(x) = 3x + 1 passes through the point (2,7) -/
theorem function_passes_through_point :
  let f : ℝ → ℝ := λ x ↦ 3 * x + 1
  f 2 = 7 := by sorry

end NUMINAMATH_CALUDE_function_passes_through_point_l1049_104944


namespace NUMINAMATH_CALUDE_right_triangle_leg_length_l1049_104966

/-- Proves that in a right triangle with an area of 800 square feet and one leg of 40 feet, 
    the length of the other leg is also 40 feet. -/
theorem right_triangle_leg_length 
  (area : ℝ) 
  (base : ℝ) 
  (h : area = 800) 
  (b : base = 40) : 
  (2 * area) / base = 40 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_leg_length_l1049_104966


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l1049_104911

theorem quadratic_coefficient (a : ℚ) : 
  (∀ x, (x + 4)^2 * a = (x + 4)^2 * (-8/9)) → 
  a = -8/9 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l1049_104911


namespace NUMINAMATH_CALUDE_conference_handshakes_theorem_l1049_104934

/-- Represents the number of handshakes in a conference with specific group interactions -/
def conference_handshakes (total : ℕ) (group_a : ℕ) (group_b : ℕ) (group_c : ℕ) 
  (known_per_b : ℕ) : ℕ :=
  let handshakes_ab := group_b * (group_a - known_per_b)
  let handshakes_bc := group_b * group_c
  let handshakes_c := group_c * (group_c - 1) / 2
  let handshakes_ac := group_a * group_c
  handshakes_ab + handshakes_bc + handshakes_c + handshakes_ac

/-- Theorem stating that the number of handshakes in the given conference scenario is 535 -/
theorem conference_handshakes_theorem :
  conference_handshakes 50 30 15 5 10 = 535 := by
  sorry

end NUMINAMATH_CALUDE_conference_handshakes_theorem_l1049_104934


namespace NUMINAMATH_CALUDE_fraction_decimal_comparison_l1049_104995

theorem fraction_decimal_comparison : (1 : ℚ) / 4 - 0.250000025 = 1 / (4 * 10^7) := by sorry

end NUMINAMATH_CALUDE_fraction_decimal_comparison_l1049_104995


namespace NUMINAMATH_CALUDE_remainder_11_pow_2023_mod_33_l1049_104926

theorem remainder_11_pow_2023_mod_33 : 11^2023 % 33 = 11 := by
  sorry

end NUMINAMATH_CALUDE_remainder_11_pow_2023_mod_33_l1049_104926


namespace NUMINAMATH_CALUDE_nested_sqrt_value_l1049_104945

theorem nested_sqrt_value : ∃ x : ℝ, x > 0 ∧ x = Real.sqrt (2 - x) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_nested_sqrt_value_l1049_104945


namespace NUMINAMATH_CALUDE_division_remainder_problem_l1049_104991

theorem division_remainder_problem (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) :
  dividend = 172 →
  divisor = 17 →
  quotient = 10 →
  dividend = divisor * quotient + remainder →
  remainder = 2 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l1049_104991


namespace NUMINAMATH_CALUDE_ap_special_condition_l1049_104936

/-- An arithmetic progression is a sequence where the difference between
    successive terms is constant. -/
structure ArithmeticProgression where
  first : ℝ
  diff : ℝ

/-- The nth term of an arithmetic progression -/
def nthTerm (ap : ArithmeticProgression) (n : ℕ) : ℝ :=
  ap.first + (n - 1 : ℝ) * ap.diff

theorem ap_special_condition (ap : ArithmeticProgression) :
  nthTerm ap 4 + nthTerm ap 20 = nthTerm ap 8 + nthTerm ap 15 + nthTerm ap 12 →
  ap.first = 10 * ap.diff := by
  sorry

end NUMINAMATH_CALUDE_ap_special_condition_l1049_104936


namespace NUMINAMATH_CALUDE_coordinates_wrt_y_axis_l1049_104937

/-- Given a point A with coordinates (3,-1) in the standard coordinate system,
    its coordinates with respect to the y-axis are (-3,-1). -/
theorem coordinates_wrt_y_axis :
  let A : ℝ × ℝ := (3, -1)
  let A_y_axis : ℝ × ℝ := (-3, -1)
  A_y_axis = (- A.1, A.2) :=
by sorry

end NUMINAMATH_CALUDE_coordinates_wrt_y_axis_l1049_104937


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1049_104983

theorem complex_equation_solution :
  ∀ z : ℂ, z + 5 - 6*I = 3 + 4*I → z = -2 + 10*I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1049_104983


namespace NUMINAMATH_CALUDE_phi_value_l1049_104941

theorem phi_value (φ : Real) (h1 : 0 < φ ∧ φ < π / 2) 
  (h2 : Real.sqrt 3 * Real.sin (15 * π / 180) = Real.cos φ - Real.sin φ) : 
  φ = 30 * π / 180 := by
sorry

end NUMINAMATH_CALUDE_phi_value_l1049_104941


namespace NUMINAMATH_CALUDE_parallel_vectors_sum_l1049_104974

/-- Two vectors in R² are parallel if their components are proportional -/
def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem parallel_vectors_sum (x : ℝ) :
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (x, -2)
  parallel a b → a.1 + b.1 = -2 ∧ a.2 + b.2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_sum_l1049_104974


namespace NUMINAMATH_CALUDE_recruitment_probabilities_l1049_104969

/-- Represents the recruitment scenario -/
structure RecruitmentScenario where
  totalQuestions : Nat
  drawnQuestions : Nat
  knownQuestions : Nat
  minCorrect : Nat

/-- Calculates the probability of proceeding to the interview stage -/
def probabilityToInterview (scenario : RecruitmentScenario) : Rat :=
  sorry

/-- Represents the probability distribution of correctly answerable questions -/
structure ProbabilityDistribution where
  p0 : Rat
  p1 : Rat
  p2 : Rat
  p3 : Rat

/-- Calculates the probability distribution of correctly answerable questions -/
def probabilityDistribution (scenario : RecruitmentScenario) : ProbabilityDistribution :=
  sorry

theorem recruitment_probabilities 
  (scenario : RecruitmentScenario)
  (h1 : scenario.totalQuestions = 10)
  (h2 : scenario.drawnQuestions = 3)
  (h3 : scenario.knownQuestions = 6)
  (h4 : scenario.minCorrect = 2) :
  probabilityToInterview scenario = 2/3 ∧
  let dist := probabilityDistribution scenario
  dist.p0 = 1/30 ∧ dist.p1 = 3/10 ∧ dist.p2 = 1/2 ∧ dist.p3 = 1/6 :=
sorry

end NUMINAMATH_CALUDE_recruitment_probabilities_l1049_104969


namespace NUMINAMATH_CALUDE_triangle_inequality_sum_l1049_104981

/-- Given a triangle ABC with side lengths a, b, c, and a point P in the plane of the triangle
    with distances PA = p, PB = q, PC = r, prove that pq/ab + qr/bc + rp/ac ≥ 1 -/
theorem triangle_inequality_sum (a b c p q r : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ p > 0 ∧ q > 0 ∧ r > 0) :
  p * q / (a * b) + q * r / (b * c) + r * p / (a * c) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_sum_l1049_104981


namespace NUMINAMATH_CALUDE_garden_area_bounds_l1049_104922

/-- Represents a rectangular garden with given constraints -/
structure Garden where
  wall : ℝ
  fence : ℝ
  minParallelSide : ℝ

/-- The area of the garden as a function of the length perpendicular to the wall -/
def Garden.area (g : Garden) (x : ℝ) : ℝ :=
  x * (g.fence - 2 * x)

/-- Theorem stating the maximum and minimum areas of the garden -/
theorem garden_area_bounds (g : Garden) 
  (h_wall : g.wall = 12)
  (h_fence : g.fence = 40)
  (h_minSide : g.minParallelSide = 6) :
  (∃ x : ℝ, g.area x ≤ 168 ∧ 
   ∀ y : ℝ, g.minParallelSide ≤ g.fence - 2 * y → g.area y ≤ g.area x) ∧
  (∃ x : ℝ, g.area x ≥ 102 ∧ 
   ∀ y : ℝ, g.minParallelSide ≤ g.fence - 2 * y → g.area y ≥ g.area x) :=
sorry

end NUMINAMATH_CALUDE_garden_area_bounds_l1049_104922


namespace NUMINAMATH_CALUDE_right_triangle_from_parabolas_l1049_104964

theorem right_triangle_from_parabolas (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hac : a ≠ c)
  (h_intersect : ∃ x₀ : ℝ, x₀ ≠ 0 ∧ x₀^2 + 2*a*x₀ + b^2 = 0 ∧ x₀^2 + 2*c*x₀ - b^2 = 0) :
  a^2 = b^2 + c^2 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_from_parabolas_l1049_104964


namespace NUMINAMATH_CALUDE_track_length_l1049_104996

/-- The length of a circular track given specific running conditions -/
theorem track_length : 
  ∀ (L : ℝ),
  (∃ (d₁ d₂ : ℝ),
    d₁ = 100 ∧
    d₂ = 100 ∧
    d₁ + (L / 2 - d₁) = L / 2 ∧
    (L - d₁) + (L / 2 - d₁ + d₂) = L) →
  L = 200 :=
by sorry

end NUMINAMATH_CALUDE_track_length_l1049_104996


namespace NUMINAMATH_CALUDE_ratio_of_numbers_l1049_104904

theorem ratio_of_numbers (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b) 
  (h4 : a - b = 7 * ((a + b) / 2)) : a / b = 9 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_numbers_l1049_104904


namespace NUMINAMATH_CALUDE_number_equality_l1049_104901

theorem number_equality (x : ℝ) (h1 : x > 0) (h2 : (2/3) * x = (16/216) * (1/x)) : x = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_number_equality_l1049_104901


namespace NUMINAMATH_CALUDE_sarahs_lemonade_profit_l1049_104977

/-- Calculates the total profit for Sarah's lemonade stand --/
theorem sarahs_lemonade_profit
  (total_days : ℕ)
  (hot_days : ℕ)
  (cups_per_day : ℕ)
  (cost_per_cup : ℚ)
  (hot_day_price : ℚ)
  (hot_day_markup : ℚ)
  (h1 : total_days = 10)
  (h2 : hot_days = 3)
  (h3 : cups_per_day = 32)
  (h4 : cost_per_cup = 3/4)
  (h5 : hot_day_price = 1.6351744186046513)
  (h6 : hot_day_markup = 5/4) :
  ∃ (profit : ℚ), profit = 210.2265116279069 :=
by sorry

end NUMINAMATH_CALUDE_sarahs_lemonade_profit_l1049_104977


namespace NUMINAMATH_CALUDE_gold_coins_percentage_l1049_104915

/-- Represents the composition of objects in an urn --/
structure UrnComposition where
  beads_percent : ℝ
  marbles_percent : ℝ
  silver_coins_percent : ℝ
  gold_coins_percent : ℝ

/-- Theorem stating the percentage of gold coins in the urn --/
theorem gold_coins_percentage (u : UrnComposition) 
  (beads_cond : u.beads_percent = 0.3)
  (marbles_cond : u.marbles_percent = 0.1)
  (silver_coins_cond : u.silver_coins_percent = 0.45 * (1 - u.beads_percent - u.marbles_percent))
  (total_cond : u.beads_percent + u.marbles_percent + u.silver_coins_percent + u.gold_coins_percent = 1) :
  u.gold_coins_percent = 0.33 := by
  sorry

#check gold_coins_percentage

end NUMINAMATH_CALUDE_gold_coins_percentage_l1049_104915


namespace NUMINAMATH_CALUDE_ratio_to_nine_l1049_104987

/-- Given a ratio of 5:1 and a number 9, prove that the number x which satisfies this ratio is 45. -/
theorem ratio_to_nine : ∃ x : ℚ, (5 : ℚ) / 1 = x / 9 ∧ x = 45 := by
  sorry

end NUMINAMATH_CALUDE_ratio_to_nine_l1049_104987


namespace NUMINAMATH_CALUDE_total_kids_played_tag_l1049_104972

def monday : ℕ := 12
def tuesday : ℕ := 7
def wednesday : ℕ := 15
def thursday : ℕ := 10
def friday : ℕ := 18

theorem total_kids_played_tag : monday + tuesday + wednesday + thursday + friday = 62 := by
  sorry

end NUMINAMATH_CALUDE_total_kids_played_tag_l1049_104972


namespace NUMINAMATH_CALUDE_annas_money_l1049_104938

theorem annas_money (initial_amount : ℚ) : 
  (initial_amount * (1 - 3/8) * (1 - 1/5) = 36) → initial_amount = 72 := by
  sorry

end NUMINAMATH_CALUDE_annas_money_l1049_104938


namespace NUMINAMATH_CALUDE_optimal_purchase_l1049_104986

/-- Represents the cost and quantity of soccer balls and basketballs --/
structure BallPurchase where
  soccer_price : ℝ
  basketball_price : ℝ
  soccer_quantity : ℕ
  basketball_quantity : ℕ

/-- Defines the conditions of the ball purchase problem --/
def valid_purchase (p : BallPurchase) : Prop :=
  p.soccer_price + 3 * p.basketball_price = 275 ∧
  3 * p.soccer_price + 2 * p.basketball_price = 300 ∧
  p.soccer_quantity + p.basketball_quantity = 80 ∧
  p.soccer_quantity ≤ 3 * p.basketball_quantity

/-- Calculates the total cost of a ball purchase --/
def total_cost (p : BallPurchase) : ℝ :=
  p.soccer_price * p.soccer_quantity + p.basketball_price * p.basketball_quantity

/-- Theorem stating the most cost-effective purchase plan --/
theorem optimal_purchase :
  ∃ (p : BallPurchase),
    valid_purchase p ∧
    p.soccer_price = 50 ∧
    p.basketball_price = 75 ∧
    p.soccer_quantity = 60 ∧
    p.basketball_quantity = 20 ∧
    (∀ (q : BallPurchase), valid_purchase q → total_cost p ≤ total_cost q) :=
  sorry

end NUMINAMATH_CALUDE_optimal_purchase_l1049_104986


namespace NUMINAMATH_CALUDE_watch_ahead_by_16_minutes_l1049_104992

/-- Represents the time gain of a watch in minutes per hour -/
def time_gain : ℕ := 4

/-- Represents the start time in minutes after midnight -/
def start_time : ℕ := 10 * 60

/-- Represents the event time in minutes after midnight -/
def event_time : ℕ := 14 * 60

/-- Calculates the actual time passed given the time shown on the watch -/
def actual_time (watch_time : ℕ) : ℕ :=
  (watch_time * 60) / (60 + time_gain)

/-- Theorem stating that the watch shows 16 minutes ahead of the actual time -/
theorem watch_ahead_by_16_minutes :
  actual_time (event_time - start_time) = event_time - start_time - 16 := by
  sorry


end NUMINAMATH_CALUDE_watch_ahead_by_16_minutes_l1049_104992
