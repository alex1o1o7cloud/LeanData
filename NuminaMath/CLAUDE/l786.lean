import Mathlib

namespace NUMINAMATH_CALUDE_chess_tournament_participants_perfect_square_l786_78663

theorem chess_tournament_participants_perfect_square 
  (B : ℕ) -- number of boys
  (G : ℕ) -- number of girls
  (total_points : ℕ → ℕ → ℕ) -- function that calculates total points given boys and girls
  (h1 : ∀ x y, total_points x y = x * y) -- each participant plays once with every other
  (h2 : ∀ x y, 2 * (x * y) = x * (x - 1) + y * (y - 1)) -- half points from boys
  : ∃ k : ℕ, B + G = k^2 :=
sorry

end NUMINAMATH_CALUDE_chess_tournament_participants_perfect_square_l786_78663


namespace NUMINAMATH_CALUDE_smaller_pyramid_volume_l786_78678

/-- The volume of a smaller pyramid cut from a larger right square pyramid -/
theorem smaller_pyramid_volume
  (base_edge : ℝ)
  (total_height : ℝ)
  (cut_height : ℝ)
  (h_base : base_edge = 12)
  (h_height : total_height = 18)
  (h_cut : cut_height = 6) :
  (1/3 : ℝ) * (cut_height / total_height)^2 * base_edge^2 * cut_height = 32 := by
sorry

end NUMINAMATH_CALUDE_smaller_pyramid_volume_l786_78678


namespace NUMINAMATH_CALUDE_cube_to_rectangular_solid_surface_area_ratio_l786_78680

/-- The ratio of the surface area of a cube to the surface area of a rectangular solid
    with doubled length is 3/5. -/
theorem cube_to_rectangular_solid_surface_area_ratio :
  ∀ s : ℝ, s > 0 →
  (6 * s^2) / (2 * (2*s*s + 2*s*s + s*s)) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_cube_to_rectangular_solid_surface_area_ratio_l786_78680


namespace NUMINAMATH_CALUDE_hyogeun_weight_l786_78629

/-- Given the weights of three people satisfying certain conditions, 
    prove that one person's weight is as specified. -/
theorem hyogeun_weight (H S G : ℝ) : 
  H + S + G = 106.6 →
  G = S - 7.7 →
  S = H - 4.8 →
  H = 41.3 := by
sorry

end NUMINAMATH_CALUDE_hyogeun_weight_l786_78629


namespace NUMINAMATH_CALUDE_rabbit_speed_problem_l786_78664

theorem rabbit_speed_problem (rabbit_speed : ℕ) (x : ℕ) : 
  rabbit_speed = 45 →
  2 * (2 * rabbit_speed + x) = 188 →
  x = 4 := by
sorry

end NUMINAMATH_CALUDE_rabbit_speed_problem_l786_78664


namespace NUMINAMATH_CALUDE_inequality_proof_l786_78634

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_sum : a^3 + b^3 + c^3 = 3) : 
  (1 / (a^4 + 3)) + (1 / (b^4 + 3)) + (1 / (c^4 + 3)) ≥ 3/4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l786_78634


namespace NUMINAMATH_CALUDE_man_speed_against_current_and_headwind_l786_78624

/-- The speed of a man rowing in a river with current and headwind -/
def man_speed (downstream_speed current_speed headwind_reduction : ℝ) : ℝ :=
  downstream_speed - current_speed - current_speed - headwind_reduction

/-- Theorem stating the man's speed against current and headwind -/
theorem man_speed_against_current_and_headwind 
  (downstream_speed : ℝ) 
  (current_speed : ℝ) 
  (headwind_reduction : ℝ) 
  (h1 : downstream_speed = 22) 
  (h2 : current_speed = 4.5) 
  (h3 : headwind_reduction = 1.5) : 
  man_speed downstream_speed current_speed headwind_reduction = 11.5 := by
  sorry

#eval man_speed 22 4.5 1.5

end NUMINAMATH_CALUDE_man_speed_against_current_and_headwind_l786_78624


namespace NUMINAMATH_CALUDE_baseAngle_eq_pi_div_k_l786_78654

/-- An isosceles trapezoid inscribed around a circle -/
structure IsoscelesTrapezoidAroundCircle where
  /-- The ratio of the parallel sides -/
  k : ℝ
  /-- The angle at the base -/
  baseAngle : ℝ

/-- Theorem: The angle at the base of an isosceles trapezoid inscribed around a circle
    is equal to π/k, where k is the ratio of the parallel sides -/
theorem baseAngle_eq_pi_div_k (t : IsoscelesTrapezoidAroundCircle) :
  t.baseAngle = π / t.k :=
sorry

end NUMINAMATH_CALUDE_baseAngle_eq_pi_div_k_l786_78654


namespace NUMINAMATH_CALUDE_books_added_by_marta_l786_78653

theorem books_added_by_marta (initial_books final_books : ℕ) 
  (h1 : initial_books = 38)
  (h2 : final_books = 48)
  (h3 : final_books ≥ initial_books) :
  final_books - initial_books = 10 := by
  sorry

end NUMINAMATH_CALUDE_books_added_by_marta_l786_78653


namespace NUMINAMATH_CALUDE_linear_function_shift_l786_78671

/-- A linear function y = 2x + b shifted down by 2 units passing through (-1, 0) implies b = 4 -/
theorem linear_function_shift (b : ℝ) : 
  (∀ x y : ℝ, y = 2 * x + b - 2) →  -- shifted function
  (0 = 2 * (-1) + b - 2) →          -- passes through (-1, 0)
  b = 4 := by
sorry

end NUMINAMATH_CALUDE_linear_function_shift_l786_78671


namespace NUMINAMATH_CALUDE_sum_of_angles_l786_78655

-- Define a rectangle
structure Rectangle where
  angles : ℕ
  is_rectangle : angles = 4

-- Define a square
structure Square where
  angles : ℕ
  is_square : angles = 4

-- Theorem statement
theorem sum_of_angles (rect : Rectangle) (sq : Square) : rect.angles + sq.angles = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_angles_l786_78655


namespace NUMINAMATH_CALUDE_expression_evaluation_l786_78688

theorem expression_evaluation :
  let x : ℝ := 3
  let numerator := 4 + x^2 - x*(2+x) - 2^2
  let denominator := x^2 - 2*x + 3
  numerator / denominator = -1 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l786_78688


namespace NUMINAMATH_CALUDE_proportion_problem_l786_78601

theorem proportion_problem :
  ∃ (a b c d : ℝ),
    (a / b = c / d) ∧
    (a + d = 14) ∧
    (b + c = 11) ∧
    (a^2 + b^2 + c^2 + d^2 = 221) ∧
    (a = 12 ∧ b = 8 ∧ c = 3 ∧ d = 2) := by
  sorry

end NUMINAMATH_CALUDE_proportion_problem_l786_78601


namespace NUMINAMATH_CALUDE_parallel_vectors_k_value_l786_78650

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (c : ℝ), v.1 * w.2 = c * v.2 * w.1

theorem parallel_vectors_k_value :
  let a : ℝ × ℝ := (1, -2)
  let b : ℝ × ℝ := (3, 4)
  ∀ k : ℝ, are_parallel (a.1 - b.1, a.2 - b.2) (2 * a.1 + k * b.1, 2 * a.2 + k * b.2) →
    k = -2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_value_l786_78650


namespace NUMINAMATH_CALUDE_negative_200_means_send_out_l786_78603

/-- Represents a WeChat payment transaction -/
structure WeChatTransaction where
  amount : ℝ
  balance_before : ℝ
  balance_after : ℝ

/-- Axiom: Receiving money increases the balance -/
axiom receive_increases_balance {t : WeChatTransaction} (h : t.amount > 0) : 
  t.balance_after = t.balance_before + t.amount

/-- Axiom: Sending money decreases the balance -/
axiom send_decreases_balance {t : WeChatTransaction} (h : t.amount < 0) :
  t.balance_after = t.balance_before + t.amount

/-- The meaning of a -200 transaction in WeChat payments -/
theorem negative_200_means_send_out (t : WeChatTransaction) 
  (h1 : t.amount = -200)
  (h2 : t.balance_before = 867.35)
  (h3 : t.balance_after = 667.35) :
  "Sending out 200 yuan" = "The meaning of -200 in WeChat payments" := by
  sorry

end NUMINAMATH_CALUDE_negative_200_means_send_out_l786_78603


namespace NUMINAMATH_CALUDE_complex_product_theorem_l786_78626

theorem complex_product_theorem (z₁ z₂ : ℂ) 
  (h1 : Complex.abs z₁ = 2) 
  (h2 : Complex.abs z₂ = 3) 
  (h3 : 3 * z₁ - 2 * z₂ = (3/2 : ℂ) - Complex.I) : 
  z₁ * z₂ = -30/13 + 72/13 * Complex.I := by sorry

end NUMINAMATH_CALUDE_complex_product_theorem_l786_78626


namespace NUMINAMATH_CALUDE_sum_fifth_sixth_l786_78682

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q
  sum_first_two : a 1 + a 2 = 3
  sum_third_fourth : a 3 + a 4 = 12

/-- The sum of the fifth and sixth terms of the geometric sequence is 48 -/
theorem sum_fifth_sixth (seq : GeometricSequence) : seq.a 5 + seq.a 6 = 48 := by
  sorry

end NUMINAMATH_CALUDE_sum_fifth_sixth_l786_78682


namespace NUMINAMATH_CALUDE_fraction_change_l786_78675

theorem fraction_change (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (0.6 * x) / (0.4 * y) = 1.5 * (x / y) := by
sorry

end NUMINAMATH_CALUDE_fraction_change_l786_78675


namespace NUMINAMATH_CALUDE_fraction_equality_l786_78611

theorem fraction_equality (m n p q : ℚ) 
  (h1 : m / n = 18)
  (h2 : p / n = 2)
  (h3 : p / q = 1 / 12) :
  m / q = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l786_78611


namespace NUMINAMATH_CALUDE_number_puzzle_l786_78617

theorem number_puzzle (A B : ℝ) (h1 : A + B = 14.85) (h2 : B = 10 * A) : A = 1.35 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l786_78617


namespace NUMINAMATH_CALUDE_pages_already_read_l786_78609

/-- Theorem: Number of pages Rich has already read
Given a book with 372 pages, where Rich skipped 16 pages of maps and has 231 pages left to read,
prove that Rich has already read 125 pages. -/
theorem pages_already_read
  (total_pages : ℕ)
  (skipped_pages : ℕ)
  (pages_left : ℕ)
  (h1 : total_pages = 372)
  (h2 : skipped_pages = 16)
  (h3 : pages_left = 231) :
  total_pages - skipped_pages - pages_left = 125 := by
  sorry

end NUMINAMATH_CALUDE_pages_already_read_l786_78609


namespace NUMINAMATH_CALUDE_cannot_row_against_fast_stream_l786_78686

/-- A man rowing a boat in a stream -/
structure Rower where
  speedWithStream : ℝ
  speedInStillWater : ℝ

/-- Determine if a rower can go against the stream -/
def canRowAgainstStream (r : Rower) : Prop :=
  r.speedInStillWater > r.speedWithStream - r.speedInStillWater

/-- Theorem: A man cannot row against the stream if his speed in still water
    is less than the stream's speed -/
theorem cannot_row_against_fast_stream (r : Rower)
  (h1 : r.speedWithStream = 10)
  (h2 : r.speedInStillWater = 2) :
  ¬(canRowAgainstStream r) := by
  sorry

#check cannot_row_against_fast_stream

end NUMINAMATH_CALUDE_cannot_row_against_fast_stream_l786_78686


namespace NUMINAMATH_CALUDE_short_sleeve_students_l786_78642

/-- Proves the number of students wearing short sleeves in a class with given conditions -/
theorem short_sleeve_students (total : ℕ) (difference : ℕ) (short : ℕ) (long : ℕ) : 
  total = 36 →
  long - short = difference →
  difference = 24 →
  short + long = total →
  short = 6 := by sorry

end NUMINAMATH_CALUDE_short_sleeve_students_l786_78642


namespace NUMINAMATH_CALUDE_video_views_equation_l786_78643

/-- Represents the number of views on the first day -/
def initial_views : ℕ := 4400

/-- Represents the increase in views after 4 days -/
def increase_factor : ℕ := 10

/-- Represents the additional views after 2 more days -/
def additional_views : ℕ := 50000

/-- Represents the total views at the end -/
def total_views : ℕ := 94000

/-- Proves that the initial number of views satisfies the given equation -/
theorem video_views_equation : 
  increase_factor * initial_views + additional_views = total_views := by
  sorry

end NUMINAMATH_CALUDE_video_views_equation_l786_78643


namespace NUMINAMATH_CALUDE_allison_wins_prob_l786_78633

/-- Represents a 6-sided cube with specific face configurations -/
structure Cube where
  faces : Fin 6 → ℕ

/-- Allison's cube configuration -/
def allison_cube : Cube :=
  { faces := λ i => if i.val < 3 then 3 else 4 }

/-- Brian's cube configuration -/
def brian_cube : Cube :=
  { faces := λ i => i.val }

/-- Noah's cube configuration -/
def noah_cube : Cube :=
  { faces := λ i => if i.val < 3 then 2 else 6 }

/-- Probability of rolling a specific value on a cube -/
def prob_roll (c : Cube) (v : ℕ) : ℚ :=
  (Finset.filter (λ i => c.faces i = v) (Finset.univ : Finset (Fin 6))).card / 6

/-- Probability of rolling less than a value on a cube -/
def prob_roll_less (c : Cube) (v : ℕ) : ℚ :=
  (Finset.filter (λ i => c.faces i < v) (Finset.univ : Finset (Fin 6))).card / 6

/-- The main theorem stating the probability of Allison winning -/
theorem allison_wins_prob :
  (1 / 2) * (prob_roll_less brian_cube 3 * prob_roll_less noah_cube 3 +
             prob_roll_less brian_cube 4 * prob_roll_less noah_cube 4) = 7 / 24 := by
  sorry

#check allison_wins_prob

end NUMINAMATH_CALUDE_allison_wins_prob_l786_78633


namespace NUMINAMATH_CALUDE_shekar_average_marks_l786_78600

def shekar_scores : List ℕ := [76, 65, 82, 67, 55]

theorem shekar_average_marks :
  (shekar_scores.sum / shekar_scores.length : ℚ) = 69 := by
  sorry

end NUMINAMATH_CALUDE_shekar_average_marks_l786_78600


namespace NUMINAMATH_CALUDE_factorization_of_4x_squared_minus_16_l786_78635

theorem factorization_of_4x_squared_minus_16 (x : ℝ) : 4 * x^2 - 16 = 4 * (x + 2) * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_4x_squared_minus_16_l786_78635


namespace NUMINAMATH_CALUDE_average_weight_b_c_l786_78695

/-- Given the weights of three people a, b, and c, prove that the average weight of b and c is 47 kg. -/
theorem average_weight_b_c (a b c : ℝ) : 
  (a + b + c) / 3 = 45 →
  (a + b) / 2 = 40 →
  b = 39 →
  (b + c) / 2 = 47 := by
sorry

end NUMINAMATH_CALUDE_average_weight_b_c_l786_78695


namespace NUMINAMATH_CALUDE_initial_value_proof_l786_78627

theorem initial_value_proof (increase_rate : ℝ) (final_value : ℝ) (years : ℕ) : 
  increase_rate = 1/8 →
  years = 2 →
  final_value = 8100 →
  final_value = 6400 * (1 + increase_rate)^years →
  6400 = 6400 := by sorry

end NUMINAMATH_CALUDE_initial_value_proof_l786_78627


namespace NUMINAMATH_CALUDE_impossible_to_blacken_board_l786_78659

/-- Represents the state of a chessboard -/
def Chessboard := Fin 8 → Fin 8 → Bool

/-- A move is represented by its top-left corner and orientation -/
structure Move where
  row : Fin 8
  col : Fin 8
  horizontal : Bool

/-- Apply a move to a chessboard -/
def applyMove (board : Chessboard) (move : Move) : Chessboard :=
  sorry

/-- Count the number of black squares on the board -/
def countBlackSquares (board : Chessboard) : Nat :=
  sorry

/-- The initial all-white chessboard -/
def initialBoard : Chessboard :=
  fun _ _ => false

/-- The final all-black chessboard -/
def finalBoard : Chessboard :=
  fun _ _ => true

/-- Theorem: It's impossible to transform the initial board to the final board using only valid moves -/
theorem impossible_to_blacken_board :
  ¬∃ (moves : List Move), (moves.foldl applyMove initialBoard) = finalBoard :=
sorry

end NUMINAMATH_CALUDE_impossible_to_blacken_board_l786_78659


namespace NUMINAMATH_CALUDE_restaurant_profit_l786_78669

/-- The profit calculated with mistakes -/
def mistaken_profit : ℕ := 1320

/-- The difference in hundreds place due to the mistake -/
def hundreds_difference : ℕ := 8 - 3

/-- The difference in tens place due to the mistake -/
def tens_difference : ℕ := 8 - 5

/-- The actual profit of the restaurant -/
def actual_profit : ℕ := mistaken_profit - hundreds_difference * 100 + tens_difference * 10

theorem restaurant_profit : actual_profit = 850 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_profit_l786_78669


namespace NUMINAMATH_CALUDE_remaining_work_time_is_three_l786_78698

/-- The time taken by A to finish the remaining work after B has worked for 10 days -/
def remaining_work_time (a_time b_time b_worked_days : ℚ) : ℚ :=
  let b_work_rate := 1 / b_time
  let b_work_done := b_work_rate * b_worked_days
  let remaining_work := 1 - b_work_done
  let a_work_rate := 1 / a_time
  remaining_work / a_work_rate

/-- Theorem stating that A will take 3 days to finish the remaining work -/
theorem remaining_work_time_is_three :
  remaining_work_time 9 15 10 = 3 := by
  sorry

#eval remaining_work_time 9 15 10

end NUMINAMATH_CALUDE_remaining_work_time_is_three_l786_78698


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l786_78622

/-- Given two arithmetic sequences {a_n} and {b_n} with sums S_n and T_n respectively,
    if S_n / T_n = (2n - 3) / (n + 2) for all n, then a_5 / b_5 = 15 / 11 -/
theorem arithmetic_sequence_ratio (a b : ℕ → ℚ) (S T : ℕ → ℚ) 
    (h_arithmetic_a : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))
    (h_arithmetic_b : ∀ n, b (n + 1) - b n = b (n + 2) - b (n + 1))
    (h_sum_a : ∀ n, S n = (n : ℚ) * (a 1 + a n) / 2)
    (h_sum_b : ∀ n, T n = (n : ℚ) * (b 1 + b n) / 2)
    (h_ratio : ∀ n, S n / T n = (2 * n - 3) / (n + 2)) :
  a 5 / b 5 = 15 / 11 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l786_78622


namespace NUMINAMATH_CALUDE_jean_spots_ratio_l786_78630

/-- Jean the jaguar's spot distribution --/
def jean_spots (total_spots upper_torso_spots side_spots : ℕ) : Prop :=
  total_spots = upper_torso_spots + side_spots ∧
  upper_torso_spots = 30 ∧
  side_spots = 10 ∧
  2 * upper_torso_spots = total_spots

theorem jean_spots_ratio :
  ∀ total_spots upper_torso_spots side_spots,
  jean_spots total_spots upper_torso_spots side_spots →
  (total_spots / 2 : ℚ) / total_spots = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_jean_spots_ratio_l786_78630


namespace NUMINAMATH_CALUDE_product_sum_theorem_l786_78610

theorem product_sum_theorem (a b : ℕ) : 
  10 ≤ a ∧ a ≤ 99 ∧ 
  10 ≤ b ∧ b ≤ 99 ∧ 
  a * b = 1656 ∧ 
  (a % 10) * b < 1000 →
  a + b = 110 := by
sorry

end NUMINAMATH_CALUDE_product_sum_theorem_l786_78610


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_problem_l786_78677

/-- Three positive numbers form an arithmetic sequence -/
def is_arithmetic_sequence (a b c : ℝ) : Prop := b - a = c - b

/-- Three numbers form a geometric sequence -/
def is_geometric_sequence (a b c : ℝ) : Prop := b / a = c / b

theorem arithmetic_geometric_sequence_problem :
  ∀ a b c : ℝ,
  a > 0 ∧ b > 0 ∧ c > 0 →
  is_arithmetic_sequence a b c →
  a + b + c = 15 →
  is_geometric_sequence (a + 1) (b + 3) (c + 9) →
  a = 3 ∧ b = 5 ∧ c = 7 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_problem_l786_78677


namespace NUMINAMATH_CALUDE_expression_evaluation_l786_78661

theorem expression_evaluation :
  (3^1010 + 4^1012)^2 - (3^1010 - 4^1012)^2 = 10^2630 * 10^1012 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l786_78661


namespace NUMINAMATH_CALUDE_sum_fraction_problem_l786_78683

theorem sum_fraction_problem (a₁ a₂ a₃ a₄ a₅ : ℝ) 
  (h1 : a₁ / 2 + a₂ / 3 + a₃ / 4 + a₄ / 5 + a₅ / 6 = 1)
  (h2 : a₁ / 5 + a₂ / 6 + a₃ / 7 + a₄ / 8 + a₅ / 9 = 1 / 4)
  (h3 : a₁ / 10 + a₂ / 11 + a₃ / 12 + a₄ / 13 + a₅ / 14 = 1 / 9)
  (h4 : a₁ / 17 + a₂ / 18 + a₃ / 19 + a₄ / 20 + a₅ / 21 = 1 / 16)
  (h5 : a₁ / 26 + a₂ / 27 + a₃ / 28 + a₄ / 29 + a₅ / 30 = 1 / 25) :
  a₁ / 37 + a₂ / 38 + a₃ / 39 + a₄ / 40 + a₅ / 41 = 187465 / 6744582 := by
  sorry

end NUMINAMATH_CALUDE_sum_fraction_problem_l786_78683


namespace NUMINAMATH_CALUDE_fraction_numerator_l786_78692

theorem fraction_numerator (y : ℝ) (x : ℕ) (h1 : y > 0) 
  (h2 : (x / y) * y + (3 * y) / 10 = 0.35 * y) : x = 32 := by
  sorry

end NUMINAMATH_CALUDE_fraction_numerator_l786_78692


namespace NUMINAMATH_CALUDE_abc_inequality_l786_78656

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : 1 / (a^2 + 1) + 1 / (b^2 + 1) + 1 / (c^2 + 1) = 2) :
  a * b + b * c + c * a ≤ 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l786_78656


namespace NUMINAMATH_CALUDE_animals_equal_humps_l786_78604

/-- Represents the number of animals of each type in the herd -/
structure Herd where
  horses : ℕ
  oneHumpCamels : ℕ
  twoHumpCamels : ℕ

/-- Calculates the total number of humps in the herd -/
def totalHumps (h : Herd) : ℕ :=
  h.oneHumpCamels + 2 * h.twoHumpCamels

/-- Calculates the total number of animals in the herd -/
def totalAnimals (h : Herd) : ℕ :=
  h.horses + h.oneHumpCamels + h.twoHumpCamels

/-- Theorem stating that under the given conditions, the total number of animals equals the total number of humps -/
theorem animals_equal_humps (h : Herd) 
    (hump_count : totalHumps h = 200) 
    (equal_horses_twohumps : h.horses = h.twoHumpCamels) : 
  totalAnimals h = 200 := by
  sorry


end NUMINAMATH_CALUDE_animals_equal_humps_l786_78604


namespace NUMINAMATH_CALUDE_det_specific_matrix_l786_78605

theorem det_specific_matrix (x : ℝ) : 
  let A : Matrix (Fin 3) (Fin 3) ℝ := !![x + 2, x + 1, x; x, x + 2, x + 1; x + 1, x, x + 2]
  Matrix.det A = x^2 + 11*x + 9 := by
sorry

end NUMINAMATH_CALUDE_det_specific_matrix_l786_78605


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l786_78691

theorem cubic_equation_roots (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (h : a * b^2 + 1 = 0) :
  let f := fun x : ℝ => x / a + x^2 / b + x^3 / c - b * c
  (c > 0 → (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ f x₃ = 0)) ∧
  (c < 0 → (∃! x : ℝ, f x = 0)) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l786_78691


namespace NUMINAMATH_CALUDE_centrifuge_force_scientific_notation_l786_78674

theorem centrifuge_force_scientific_notation :
  17000 = 1.7 * (10 ^ 4) := by sorry

end NUMINAMATH_CALUDE_centrifuge_force_scientific_notation_l786_78674


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l786_78687

theorem point_in_fourth_quadrant (a : ℝ) :
  let A : ℝ × ℝ := (Real.sqrt a + 1, -3)
  A.1 > 0 ∧ A.2 < 0 :=
by sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l786_78687


namespace NUMINAMATH_CALUDE_total_grains_in_grey_parts_l786_78637

/-- Represents a circle with grains -/
structure GrainCircle where
  total : ℕ
  nonOverlapping : ℕ

/-- Calculates the number of grains in the overlapping part of a circle -/
def overlappingGrains (circle : GrainCircle) : ℕ :=
  circle.total - circle.nonOverlapping

/-- Represents two overlapping circles with grains -/
structure OverlappingCircles where
  circle1 : GrainCircle
  circle2 : GrainCircle

/-- Theorem: The total number of grains in both grey parts is 61 -/
theorem total_grains_in_grey_parts (circles : OverlappingCircles)
  (h1 : circles.circle1.total = 87)
  (h2 : circles.circle2.total = 110)
  (h3 : circles.circle1.nonOverlapping = 68)
  (h4 : circles.circle2.nonOverlapping = 68) :
  overlappingGrains circles.circle1 + overlappingGrains circles.circle2 = 61 := by
  sorry

end NUMINAMATH_CALUDE_total_grains_in_grey_parts_l786_78637


namespace NUMINAMATH_CALUDE_gifts_sent_calculation_l786_78625

/-- The number of gifts sent to the orphanage given the initial number of gifts and the number of gifts left -/
def gifts_sent_to_orphanage (initial_gifts : ℕ) (gifts_left : ℕ) : ℕ :=
  initial_gifts - gifts_left

/-- Theorem stating that for the given scenario, 66 gifts were sent to the orphanage -/
theorem gifts_sent_calculation :
  gifts_sent_to_orphanage 77 11 = 66 := by
  sorry

end NUMINAMATH_CALUDE_gifts_sent_calculation_l786_78625


namespace NUMINAMATH_CALUDE_barycentric_vector_relation_l786_78618

/-- For a triangle ABC and a point X with barycentric coordinates (α:β:γ) where α + β + γ = 1,
    the vector →XA is equal to β→BA + γ→CA. -/
theorem barycentric_vector_relation (A B C X : EuclideanSpace ℝ (Fin 3))
  (α β γ : ℝ) (h_barycentric : α + β + γ = 1)
  (h_X : X = α • A + β • B + γ • C) :
  X - A = β • (B - A) + γ • (C - A) := by
  sorry

end NUMINAMATH_CALUDE_barycentric_vector_relation_l786_78618


namespace NUMINAMATH_CALUDE_smallest_tetrahedron_volume_ellipsoid_l786_78612

/-- The smallest volume of a tetrahedron bounded by a tangent plane to an ellipsoid and coordinate planes -/
theorem smallest_tetrahedron_volume_ellipsoid (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∀ (V : ℝ), V ≥ (Real.sqrt 3 * a * b * c) / 2 → 
  ∃ (x y z : ℝ), x^2/a^2 + y^2/b^2 + z^2/c^2 = 1 ∧ 
    V = (1/6) * (a^2/x) * (b^2/y) * (c^2/z) :=
by sorry

end NUMINAMATH_CALUDE_smallest_tetrahedron_volume_ellipsoid_l786_78612


namespace NUMINAMATH_CALUDE_paul_filled_three_bags_sunday_l786_78607

/-- Calculates the number of bags filled on Sunday given the total cans collected,
    bags filled on Saturday, and cans per bag. -/
def bags_filled_sunday (total_cans : ℕ) (saturday_bags : ℕ) (cans_per_bag : ℕ) : ℕ :=
  (total_cans - saturday_bags * cans_per_bag) / cans_per_bag

/-- Proves that for the given problem, Paul filled 3 bags on Sunday. -/
theorem paul_filled_three_bags_sunday :
  bags_filled_sunday 72 6 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_paul_filled_three_bags_sunday_l786_78607


namespace NUMINAMATH_CALUDE_skew_lines_parallel_implication_l786_78636

/-- Two lines in 3D space -/
structure Line3D where
  -- This is a simplified representation of a line in 3D space
  -- In a real implementation, we might use vectors or points to define a line

/-- Predicate for two lines being skew -/
def are_skew (l1 l2 : Line3D) : Prop :=
  -- Two lines are skew if they are not parallel and do not intersect
  sorry

/-- Predicate for two lines being parallel -/
def are_parallel (l1 l2 : Line3D) : Prop :=
  -- Two lines are parallel if they have the same direction but do not intersect
  sorry

theorem skew_lines_parallel_implication (a b c : Line3D) 
  (h1 : are_skew a b) (h2 : are_parallel c a) : 
  ¬(are_parallel c b) := by
  sorry

end NUMINAMATH_CALUDE_skew_lines_parallel_implication_l786_78636


namespace NUMINAMATH_CALUDE_percentage_sum_theorem_l786_78697

theorem percentage_sum_theorem : (0.15 * 25) + (0.12 * 45) = 9.15 := by sorry

end NUMINAMATH_CALUDE_percentage_sum_theorem_l786_78697


namespace NUMINAMATH_CALUDE_solution_is_correct_l786_78646

/-- The equation to be solved -/
def equation (y : ℝ) : Prop :=
  1/6 + 6/y = 14/y - 1/14

/-- The theorem stating that y = 168/5 is the solution to the equation -/
theorem solution_is_correct : equation (168/5) := by
  sorry

end NUMINAMATH_CALUDE_solution_is_correct_l786_78646


namespace NUMINAMATH_CALUDE_commodity_tax_consumption_l786_78684

theorem commodity_tax_consumption (T C : ℝ) (h1 : T > 0) (h2 : C > 0) : 
  let new_tax := 0.75 * T
  let new_revenue := 0.825 * T * C
  let new_consumption := C * (1 + 10 / 100)
  new_tax * new_consumption = new_revenue := by sorry

end NUMINAMATH_CALUDE_commodity_tax_consumption_l786_78684


namespace NUMINAMATH_CALUDE_cone_volume_l786_78631

/-- The volume of a cone with slant height 15 cm and height 9 cm is 432π cubic centimeters. -/
theorem cone_volume (π : ℝ) : 
  let l : ℝ := 15  -- slant height
  let h : ℝ := 9   -- height
  let r : ℝ := Real.sqrt (l^2 - h^2)  -- radius of the base
  (1/3 : ℝ) * π * r^2 * h = 432 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_l786_78631


namespace NUMINAMATH_CALUDE_composite_polynomial_l786_78621

theorem composite_polynomial (n : ℕ) : 
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ n^3 + 3*n^2 + 6*n + 8 = a * b :=
sorry

end NUMINAMATH_CALUDE_composite_polynomial_l786_78621


namespace NUMINAMATH_CALUDE_washing_machine_cost_l786_78668

theorem washing_machine_cost (down_payment : ℝ) (down_payment_percentage : ℝ) (total_cost : ℝ) : 
  down_payment = 200 →
  down_payment_percentage = 25 →
  down_payment = (down_payment_percentage / 100) * total_cost →
  total_cost = 800 := by
sorry

end NUMINAMATH_CALUDE_washing_machine_cost_l786_78668


namespace NUMINAMATH_CALUDE_reciprocal_fraction_l786_78644

theorem reciprocal_fraction (x : ℝ) (y : ℝ) (h1 : x > 0) (h2 : x = 1) 
  (h3 : (2/3) * x = y * (1/x)) : y = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_fraction_l786_78644


namespace NUMINAMATH_CALUDE_f_neg_one_equals_two_l786_78685

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define g in terms of f
def g (x : ℝ) := f x + 4

-- State the theorem
theorem f_neg_one_equals_two
  (h_odd : ∀ x, f (-x) = -f x)  -- f is an odd function
  (h_g_one : g 1 = 2)           -- g(1) = 2
  : f (-1) = 2 := by
  sorry


end NUMINAMATH_CALUDE_f_neg_one_equals_two_l786_78685


namespace NUMINAMATH_CALUDE_toms_blue_marbles_l786_78660

theorem toms_blue_marbles (jason_blue : ℕ) (total_blue : ℕ) 
  (h1 : jason_blue = 44)
  (h2 : total_blue = 68) :
  total_blue - jason_blue = 24 := by
sorry

end NUMINAMATH_CALUDE_toms_blue_marbles_l786_78660


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l786_78606

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) : 
  (∀ x : ℝ, (2 - x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + a₂ + a₃ + a₄ + a₅ = -31 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l786_78606


namespace NUMINAMATH_CALUDE_exterior_angle_theorem_l786_78619

-- Define the triangle RWU
structure Triangle (R W U : Type) where
  angle_SWR : ℝ  -- Exterior angle
  angle_WRU : ℝ  -- Interior angle
  angle_WUR : ℝ  -- Interior angle (to be proved)
  straight_line : Prop  -- RTQU forms a straight line

-- State the theorem
theorem exterior_angle_theorem 
  (t : Triangle R W U) 
  (h1 : t.angle_SWR = 50)
  (h2 : t.angle_WRU = 30)
  (h3 : t.straight_line) : 
  t.angle_WUR = 20 := by
sorry

end NUMINAMATH_CALUDE_exterior_angle_theorem_l786_78619


namespace NUMINAMATH_CALUDE_work_completion_equality_second_group_size_correct_l786_78679

/-- The number of men in the first group -/
def first_group : ℕ := 12

/-- The number of days the first group takes to complete the work -/
def first_days : ℕ := 30

/-- The number of days the second group takes to complete the work -/
def second_days : ℕ := 36

/-- The number of men in the second group -/
def second_group : ℕ := 10

theorem work_completion_equality :
  first_group * first_days = second_group * second_days :=
by sorry

/-- Proves that the number of men in the second group is correct -/
theorem second_group_size_correct :
  second_group = (first_group * first_days) / second_days :=
by sorry

end NUMINAMATH_CALUDE_work_completion_equality_second_group_size_correct_l786_78679


namespace NUMINAMATH_CALUDE_function_decreasing_interval_l786_78665

def f (a b x : ℝ) : ℝ := x^3 - 3*a*x + b

theorem function_decreasing_interval
  (a b : ℝ)
  (h1 : a > 0)
  (h2 : ∃ x1, f a b x1 = 6 ∧ ∀ x, f a b x ≤ 6)
  (h3 : ∃ x2, f a b x2 = 2 ∧ ∀ x, f a b x ≥ 2) :
  ∀ x ∈ Set.Ioo (-1 : ℝ) 1, ∀ y ∈ Set.Ioo (-1 : ℝ) 1,
    x < y → f a b x > f a b y :=
by sorry

end NUMINAMATH_CALUDE_function_decreasing_interval_l786_78665


namespace NUMINAMATH_CALUDE_trigonometric_identity_l786_78670

theorem trigonometric_identity (α : ℝ) 
  (h : Real.cos (π / 6 - α) = Real.sqrt 3 / 3) : 
  Real.sin (α - π / 6) ^ 2 - Real.cos (5 * π / 6 + α) = (2 + Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l786_78670


namespace NUMINAMATH_CALUDE_amelias_dinner_leftover_l786_78658

/-- Calculates the amount of money Amelia has left after her dinner --/
def ameliasDinner (initialAmount : ℝ) (firstCourseCost : ℝ) (secondCourseExtra : ℝ) 
  (dessertPercent : ℝ) (drinkPercent : ℝ) (tipPercent : ℝ) : ℝ :=
  let secondCourseCost := firstCourseCost + secondCourseExtra
  let dessertCost := dessertPercent * secondCourseCost
  let firstThreeCoursesTotal := firstCourseCost + secondCourseCost + dessertCost
  let drinkCost := drinkPercent * firstThreeCoursesTotal
  let billBeforeTip := firstThreeCoursesTotal + drinkCost
  let tipAmount := tipPercent * billBeforeTip
  let totalBill := billBeforeTip + tipAmount
  initialAmount - totalBill

/-- Theorem stating that Amelia will have $4.80 left after her dinner --/
theorem amelias_dinner_leftover :
  ameliasDinner 60 15 5 0.25 0.20 0.15 = 4.80 := by
  sorry

#eval ameliasDinner 60 15 5 0.25 0.20 0.15

end NUMINAMATH_CALUDE_amelias_dinner_leftover_l786_78658


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l786_78696

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 4 * x * y) :
  1 / x + 1 / y = 4 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l786_78696


namespace NUMINAMATH_CALUDE_sqrt_x_minus_two_real_l786_78676

theorem sqrt_x_minus_two_real (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 2) ↔ x ≥ 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_two_real_l786_78676


namespace NUMINAMATH_CALUDE_no_snow_probability_l786_78615

theorem no_snow_probability (p1 p2 p3 p4 : ℚ) 
  (h1 : p1 = 2/3) (h2 : p2 = 3/4) (h3 : p3 = 5/6) (h4 : p4 = 1/2) :
  (1 - p1) * (1 - p2) * (1 - p3) * (1 - p4) = 1/144 := by
  sorry

end NUMINAMATH_CALUDE_no_snow_probability_l786_78615


namespace NUMINAMATH_CALUDE_binomial_30_3_l786_78639

theorem binomial_30_3 : Nat.choose 30 3 = 4060 := by
  sorry

end NUMINAMATH_CALUDE_binomial_30_3_l786_78639


namespace NUMINAMATH_CALUDE_quadratic_inequality_all_reals_l786_78620

theorem quadratic_inequality_all_reals (a b c : ℝ) :
  (∀ x, (a / 3) * x^2 + 2 * b * x - c < 0) ↔ (a > 0 ∧ 4 * b^2 - (4 / 3) * a * c < 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_all_reals_l786_78620


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l786_78640

theorem rhombus_perimeter (x₁ x₂ : ℝ) : 
  x₁^2 - 14*x₁ + 48 = 0 →
  x₂^2 - 14*x₂ + 48 = 0 →
  x₁ ≠ x₂ →
  let s := Real.sqrt ((x₁^2 + x₂^2) / 4)
  4 * s = 20 := by sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l786_78640


namespace NUMINAMATH_CALUDE_ferris_wheel_rides_l786_78689

theorem ferris_wheel_rides (rollercoaster_rides catapult_rides : ℕ) 
  (rollercoaster_cost catapult_cost ferris_wheel_cost total_tickets : ℕ) :
  rollercoaster_rides = 3 →
  catapult_rides = 2 →
  rollercoaster_cost = 4 →
  catapult_cost = 4 →
  ferris_wheel_cost = 1 →
  total_tickets = 21 →
  (total_tickets - (rollercoaster_rides * rollercoaster_cost + catapult_rides * catapult_cost)) / ferris_wheel_cost = 1 :=
by sorry

end NUMINAMATH_CALUDE_ferris_wheel_rides_l786_78689


namespace NUMINAMATH_CALUDE_inverse_inequality_l786_78673

theorem inverse_inequality (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 1 / a < 1 / b := by
  sorry

end NUMINAMATH_CALUDE_inverse_inequality_l786_78673


namespace NUMINAMATH_CALUDE_infinitely_many_primes_6n_plus_5_l786_78652

theorem infinitely_many_primes_6n_plus_5 :
  ∀ k : ℕ, ∃ p : ℕ, p > k ∧ Prime p ∧ ∃ n : ℕ, p = 6 * n + 5 := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_primes_6n_plus_5_l786_78652


namespace NUMINAMATH_CALUDE_janet_monday_wednesday_hours_l786_78641

/-- Janet's weekly gym schedule -/
structure GymSchedule where
  total_hours : ℝ
  monday_hours : ℝ
  tuesday_hours : ℝ
  wednesday_hours : ℝ
  friday_hours : ℝ

/-- Janet's gym schedule satisfies the given conditions -/
def janet_schedule (s : GymSchedule) : Prop :=
  s.total_hours = 5 ∧
  s.tuesday_hours = s.friday_hours ∧
  s.friday_hours = 1 ∧
  s.monday_hours = s.wednesday_hours

/-- Theorem: Janet spends 1.5 hours at the gym on Monday and Wednesday each -/
theorem janet_monday_wednesday_hours (s : GymSchedule) 
  (h : janet_schedule s) : s.monday_hours = 1.5 ∧ s.wednesday_hours = 1.5 := by
  sorry


end NUMINAMATH_CALUDE_janet_monday_wednesday_hours_l786_78641


namespace NUMINAMATH_CALUDE_tenPeopleCircularArrangements_l786_78693

/-- The number of unique circular arrangements of n people around a table,
    where rotations are considered the same. -/
def circularArrangements (n : ℕ) : ℕ := Nat.factorial (n - 1)

/-- Theorem stating that the number of unique circular arrangements
    of 10 people is equal to 9! -/
theorem tenPeopleCircularArrangements :
  circularArrangements 10 = Nat.factorial 9 := by
  sorry

end NUMINAMATH_CALUDE_tenPeopleCircularArrangements_l786_78693


namespace NUMINAMATH_CALUDE_divisor_sum_theorem_l786_78657

def sum_of_geometric_series (a r : ℕ) (n : ℕ) : ℕ := (a * (r^(n+1) - 1)) / (r - 1)

def sum_of_divisors (i j : ℕ) : ℕ := (sum_of_geometric_series 1 2 i) * (sum_of_geometric_series 1 3 j)

theorem divisor_sum_theorem (i j : ℕ) : sum_of_divisors i j = 360 → i + j = 5 := by
  sorry

end NUMINAMATH_CALUDE_divisor_sum_theorem_l786_78657


namespace NUMINAMATH_CALUDE_arcsin_equation_solution_l786_78608

theorem arcsin_equation_solution : 
  Real.arcsin (Real.sqrt (2/51)) + Real.arcsin (3 * Real.sqrt (2/51)) = π/4 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_equation_solution_l786_78608


namespace NUMINAMATH_CALUDE_max_power_of_two_product_l786_78628

open BigOperators

def is_permutation (a : Fin 17 → ℕ) : Prop :=
  ∀ i : Fin 17, ∃ j : Fin 17, a j = i.val + 1

theorem max_power_of_two_product (a : Fin 17 → ℕ) (n : ℕ) 
  (h_perm : is_permutation a) 
  (h_prod : ∏ i : Fin 17, (a i - a (i + 1)) = 2^n) : 
  n ≤ 40 ∧ ∃ a₀ : Fin 17 → ℕ, is_permutation a₀ ∧ ∏ i : Fin 17, (a₀ i - a₀ (i + 1)) = 2^40 :=
sorry

end NUMINAMATH_CALUDE_max_power_of_two_product_l786_78628


namespace NUMINAMATH_CALUDE_A_sufficient_for_B_l786_78690

def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x : ℝ | -3 ≤ x ∧ x ≤ 2}

theorem A_sufficient_for_B : ∀ x : ℝ, x ∈ A → x ∈ B := by
  sorry

end NUMINAMATH_CALUDE_A_sufficient_for_B_l786_78690


namespace NUMINAMATH_CALUDE_five_balls_three_boxes_l786_78662

/-- The number of ways to place distinguishable balls into indistinguishable boxes -/
def place_balls_in_boxes (n_balls : ℕ) (n_boxes : ℕ) : ℕ :=
  ((n_boxes ^ n_balls - n_boxes.choose 1 * (n_boxes - 1) ^ n_balls + n_boxes.choose 2) / n_boxes.factorial)

/-- Theorem: There are 25 ways to place 5 distinguishable balls into 3 indistinguishable boxes -/
theorem five_balls_three_boxes : place_balls_in_boxes 5 3 = 25 := by
  sorry

end NUMINAMATH_CALUDE_five_balls_three_boxes_l786_78662


namespace NUMINAMATH_CALUDE_complex_equation_solution_l786_78694

theorem complex_equation_solution (a : ℝ) : (a + Complex.I) * (1 - a * Complex.I) = 2 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l786_78694


namespace NUMINAMATH_CALUDE_inequality_solution_set_l786_78616

/-- A function satisfying the given conditions -/
def f_satisfies (f : ℝ → ℝ) : Prop :=
  f 0 = 2 ∧ 
  ∀ x₁ x₂, x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) > 1

/-- The solution set of the inequality -/
def solution_set (x : ℝ) : Prop :=
  Real.log 2 < x ∧ x < Real.log 3

/-- The main theorem -/
theorem inequality_solution_set (f : ℝ → ℝ) (hf : f_satisfies f) :
  ∀ x, f (Real.log (Real.exp x - 2)) < 2 + Real.log (Real.exp x - 2) ↔ solution_set x := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l786_78616


namespace NUMINAMATH_CALUDE_max_value_expression_l786_78632

theorem max_value_expression (x y : ℝ) : 
  (2 * x + 3 * y + 4) / Real.sqrt (x^2 + 4 * y^2 + 2) ≤ Real.sqrt 29 := by
  sorry

end NUMINAMATH_CALUDE_max_value_expression_l786_78632


namespace NUMINAMATH_CALUDE_perimeter_values_finite_l786_78623

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (AB BC CD AD : ℕ+)

-- Define the conditions for our specific quadrilateral
def ValidQuadrilateral (q : Quadrilateral) : Prop :=
  q.AB = 3 ∧ q.CD = 2 * q.AD

-- Define the perimeter
def Perimeter (q : Quadrilateral) : ℕ :=
  q.AB + q.BC + q.CD + q.AD

-- Define the right angle condition using Pythagorean theorem
def RightAngles (q : Quadrilateral) : Prop :=
  q.BC ^ 2 + (q.CD - q.AB) ^ 2 = q.AD ^ 2

-- Main theorem
theorem perimeter_values_finite :
  {p : ℕ | p < 3025 ∧ ∃ q : Quadrilateral, ValidQuadrilateral q ∧ RightAngles q ∧ Perimeter q = p}.Finite :=
sorry

end NUMINAMATH_CALUDE_perimeter_values_finite_l786_78623


namespace NUMINAMATH_CALUDE_technician_journey_percentage_l786_78613

theorem technician_journey_percentage (D : ℝ) (h : D > 0) : 
  let total_distance := 2 * D
  let completed_distance := 0.65 * total_distance
  let outbound_distance := D
  let return_distance_completed := completed_distance - outbound_distance
  (return_distance_completed / D) * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_technician_journey_percentage_l786_78613


namespace NUMINAMATH_CALUDE_negation_of_proposition_l786_78651

theorem negation_of_proposition (p : Prop) :
  (¬ (∀ x y : ℝ, x^2 + y^2 - 1 > 0)) ↔ (∃ x y : ℝ, x^2 + y^2 - 1 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l786_78651


namespace NUMINAMATH_CALUDE_payment_bill_value_l786_78666

/-- Proves the value of a single bill used for payment given the number of items,
    cost per item, number of change bills, and value of each change bill. -/
theorem payment_bill_value
  (num_games : ℕ)
  (cost_per_game : ℕ)
  (num_change_bills : ℕ)
  (change_bill_value : ℕ)
  (h1 : num_games = 6)
  (h2 : cost_per_game = 15)
  (h3 : num_change_bills = 2)
  (h4 : change_bill_value = 5) :
  num_games * cost_per_game + num_change_bills * change_bill_value = 100 := by
  sorry

#check payment_bill_value

end NUMINAMATH_CALUDE_payment_bill_value_l786_78666


namespace NUMINAMATH_CALUDE_fraction_sequence_2012th_term_l786_78645

/-- Represents the sequence of fractions as described in the problem -/
def fraction_sequence : ℕ → ℚ :=
  sorry

/-- The sum of the first n positive integers -/
def triangle_number (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem fraction_sequence_2012th_term :
  ∃ (n : ℕ), 
    triangle_number n ≤ 2012 ∧ 
    triangle_number (n + 1) > 2012 ∧
    63 * 64 / 2 = 2016 ∧
    fraction_sequence 2012 = 5 / 59 :=
by
  sorry

end NUMINAMATH_CALUDE_fraction_sequence_2012th_term_l786_78645


namespace NUMINAMATH_CALUDE_parallel_lines_imply_a_equals_3_l786_78667

-- Define the equations of the lines
def line1 (a x y : ℝ) : Prop := a * x + 3 * y - 1 = 0
def line2 (a x y : ℝ) : Prop := 2 * x + (a - 1) * y + 1 = 0

-- Define what it means for two lines to be parallel
def parallel (f g : ℝ → ℝ → Prop) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ ∀ (x y : ℝ), f x y ↔ g (k * x) (k * y)

-- State the theorem
theorem parallel_lines_imply_a_equals_3 (a : ℝ) :
  parallel (line1 a) (line2 a) → a = 3 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_imply_a_equals_3_l786_78667


namespace NUMINAMATH_CALUDE_toms_age_ratio_l786_78602

theorem toms_age_ratio (T N : ℚ) : 
  (T > 0) →  -- Tom's age is positive
  (N > 0) →  -- N is positive (number of years in the past)
  (T - N > 0) →  -- Tom's age N years ago was positive
  (T - 4*N > 0) →  -- Sum of children's ages N years ago was positive
  (T - N = 3 * (T - 4*N)) →  -- Condition about Tom's age N years ago
  T / N = 11 / 2 := by
  sorry

end NUMINAMATH_CALUDE_toms_age_ratio_l786_78602


namespace NUMINAMATH_CALUDE_expected_heads_alice_given_more_than_bob_l786_78614

/-- The number of coins each person flips -/
def n : ℕ := 20

/-- The expected number of heads Alice flipped given she flipped at least as many heads as Bob -/
noncomputable def expected_heads : ℝ :=
  n * (2^(2*n - 2) + Nat.choose (2*n - 1) (n - 1)) / (2^(2*n - 1) + Nat.choose (2*n - 1) (n - 1))

/-- Theorem stating the expected number of heads Alice flipped -/
theorem expected_heads_alice_given_more_than_bob :
  expected_heads = n * (2^(2*n - 2) + Nat.choose (2*n - 1) (n - 1)) / (2^(2*n - 1) + Nat.choose (2*n - 1) (n - 1)) :=
by sorry

end NUMINAMATH_CALUDE_expected_heads_alice_given_more_than_bob_l786_78614


namespace NUMINAMATH_CALUDE_afternoon_flier_fraction_l786_78648

theorem afternoon_flier_fraction (total_fliers : ℕ) (morning_fraction : ℚ) (left_for_next_day : ℕ) :
  total_fliers = 3000 →
  morning_fraction = 1 / 5 →
  left_for_next_day = 1800 →
  let morning_sent := total_fliers * morning_fraction
  let remaining_after_morning := total_fliers - morning_sent
  let afternoon_sent := remaining_after_morning - left_for_next_day
  afternoon_sent / remaining_after_morning = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_afternoon_flier_fraction_l786_78648


namespace NUMINAMATH_CALUDE_quadratic_even_function_coeff_l786_78672

/-- A quadratic function f(x) = ax^2 + (2a^2 - a)x + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + (2 * a^2 - a) * x + 1

/-- Definition of an even function -/
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem quadratic_even_function_coeff (a : ℝ) :
  is_even_function (f a) → a = 1/2 := by sorry

end NUMINAMATH_CALUDE_quadratic_even_function_coeff_l786_78672


namespace NUMINAMATH_CALUDE_max_green_socks_l786_78638

/-- Represents the count of socks in a basket -/
structure SockBasket where
  green : ℕ
  yellow : ℕ
  total_bound : green + yellow ≤ 2025

/-- The probability of selecting two green socks without replacement -/
def prob_two_green (b : SockBasket) : ℚ :=
  (b.green * (b.green - 1)) / ((b.green + b.yellow) * (b.green + b.yellow - 1))

/-- Theorem stating the maximum number of green socks possible -/
theorem max_green_socks (b : SockBasket) 
  (h : prob_two_green b = 1/3) : 
  b.green ≤ 990 ∧ ∃ b' : SockBasket, b'.green = 990 ∧ prob_two_green b' = 1/3 :=
sorry

end NUMINAMATH_CALUDE_max_green_socks_l786_78638


namespace NUMINAMATH_CALUDE_second_hole_depth_l786_78681

/-- Represents the depth of a hole dug by workers -/
def hole_depth (workers : ℕ) (hours : ℕ) (rate : ℚ) : ℚ :=
  (workers * hours : ℚ) * rate

theorem second_hole_depth :
  let initial_workers : ℕ := 45
  let initial_hours : ℕ := 8
  let initial_depth : ℚ := 30
  let extra_workers : ℕ := 65
  let second_hours : ℕ := 6
  
  let total_workers : ℕ := initial_workers + extra_workers
  let digging_rate : ℚ := initial_depth / (initial_workers * initial_hours)
  
  hole_depth total_workers second_hours digging_rate = 55 := by
  sorry


end NUMINAMATH_CALUDE_second_hole_depth_l786_78681


namespace NUMINAMATH_CALUDE_song_book_cost_l786_78649

def flute_cost : ℚ := 142.46
def tool_cost : ℚ := 8.89
def total_spent : ℚ := 158.35

theorem song_book_cost : 
  total_spent - (flute_cost + tool_cost) = 7 := by sorry

end NUMINAMATH_CALUDE_song_book_cost_l786_78649


namespace NUMINAMATH_CALUDE_potatoes_already_cooked_l786_78699

theorem potatoes_already_cooked 
  (total_potatoes : ℕ) 
  (cooking_time_per_potato : ℕ) 
  (remaining_cooking_time : ℕ) 
  (h1 : total_potatoes = 16)
  (h2 : cooking_time_per_potato = 5)
  (h3 : remaining_cooking_time = 45) :
  total_potatoes - (remaining_cooking_time / cooking_time_per_potato) = 7 :=
by sorry

end NUMINAMATH_CALUDE_potatoes_already_cooked_l786_78699


namespace NUMINAMATH_CALUDE_sin_equation_solutions_l786_78647

/-- The number of solutions to 2sin³x - 5sin²x + 2sinx = 0 in [0, 2π] is 5 -/
theorem sin_equation_solutions : 
  let f : ℝ → ℝ := λ x => 2 * Real.sin x ^ 3 - 5 * Real.sin x ^ 2 + 2 * Real.sin x
  ∃! (s : Finset ℝ), s.card = 5 ∧ 
    (∀ x ∈ s, 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ f x = 0) ∧
    (∀ x, 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ f x = 0 → x ∈ s) :=
by sorry

end NUMINAMATH_CALUDE_sin_equation_solutions_l786_78647
