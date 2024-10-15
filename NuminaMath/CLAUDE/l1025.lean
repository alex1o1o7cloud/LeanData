import Mathlib

namespace NUMINAMATH_CALUDE_tangent_equation_solution_l1025_102528

open Real

theorem tangent_equation_solution (x : ℝ) : 
  8.482 * (3 * tan x - tan x ^ 3) / (1 - tan x ^ 2) * (cos (3 * x) + cos x) = 2 * sin (5 * x) ↔ 
  (∃ k : ℤ, x = k * π) ∨ (∃ k : ℤ, x = π / 8 * (2 * k + 1)) :=
sorry

end NUMINAMATH_CALUDE_tangent_equation_solution_l1025_102528


namespace NUMINAMATH_CALUDE_speakers_cost_l1025_102536

def total_spent : ℚ := 387.85
def cd_player_cost : ℚ := 139.38
def new_tires_cost : ℚ := 112.46

theorem speakers_cost (total : ℚ) (cd : ℚ) (tires : ℚ) 
  (h1 : total = total_spent) 
  (h2 : cd = cd_player_cost) 
  (h3 : tires = new_tires_cost) : 
  total - (cd + tires) = 136.01 := by
  sorry

end NUMINAMATH_CALUDE_speakers_cost_l1025_102536


namespace NUMINAMATH_CALUDE_only_parallel_corresponding_angles_has_converse_l1025_102514

-- Define the basic geometric concepts
def Line : Type := sorry
def Angle : Type := sorry
def Triangle : Type := sorry

-- Define the geometric relations
def vertical_angles (a b : Angle) : Prop := sorry
def parallel_lines (l1 l2 : Line) : Prop := sorry
def corresponding_angles (a b : Angle) (l1 l2 : Line) : Prop := sorry
def congruent_triangles (t1 t2 : Triangle) : Prop := sorry
def right_angle (a : Angle) : Prop := sorry

-- Define the theorems
def vertical_angles_theorem (a b : Angle) : 
  vertical_angles a b → a = b := sorry

def parallel_corresponding_angles_theorem (l1 l2 : Line) (a b : Angle) :
  parallel_lines l1 l2 → corresponding_angles a b l1 l2 → a = b := sorry

def congruent_triangles_angles_theorem (t1 t2 : Triangle) (a1 a2 : Angle) :
  congruent_triangles t1 t2 → corresponding_angles a1 a2 t1 t2 → a1 = a2 := sorry

def right_angles_equal_theorem (a b : Angle) :
  right_angle a → right_angle b → a = b := sorry

-- The main theorem to prove
theorem only_parallel_corresponding_angles_has_converse :
  ∃ (l1 l2 : Line) (a b : Angle),
    (corresponding_angles a b l1 l2 ∧ a = b → parallel_lines l1 l2) ∧
    (¬∃ (a b : Angle), a = b → vertical_angles a b) ∧
    (¬∃ (t1 t2 : Triangle) (a1 a2 a3 b1 b2 b3 : Angle),
      a1 = b1 ∧ a2 = b2 ∧ a3 = b3 → congruent_triangles t1 t2) ∧
    (¬∃ (a b : Angle), a = b → right_angle a ∧ right_angle b) := by
  sorry

end NUMINAMATH_CALUDE_only_parallel_corresponding_angles_has_converse_l1025_102514


namespace NUMINAMATH_CALUDE_triangle_side_lengths_l1025_102571

-- Define the triangle ABC
theorem triangle_side_lengths (A B C : ℝ) (a b c : ℝ) (S : ℝ) :
  -- Conditions
  a = 1 →
  B = π / 4 → -- 45° in radians
  S = 2 →
  S = (1 / 2) * a * c * Real.sin B →
  a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos A →
  -- Conclusion
  c = 4 * Real.sqrt 2 ∧ b = 5 := by
sorry


end NUMINAMATH_CALUDE_triangle_side_lengths_l1025_102571


namespace NUMINAMATH_CALUDE_set_difference_equals_open_interval_l1025_102523

-- Define the sets M and N
def M : Set ℝ := {x | -1 < x ∧ x < 1}
def N : Set ℝ := {x | x ≠ 1 ∧ x / (x - 1) ≤ 0}

-- Define the open interval (-1, 0)
def open_interval : Set ℝ := {x | -1 < x ∧ x < 0}

-- Theorem statement
theorem set_difference_equals_open_interval : M \ N = open_interval := by
  sorry

end NUMINAMATH_CALUDE_set_difference_equals_open_interval_l1025_102523


namespace NUMINAMATH_CALUDE_runners_meeting_time_l1025_102507

/-- The time (in seconds) after which two runners meet at the starting point -/
def meetingTime (p_time q_time : ℕ) : ℕ :=
  Nat.lcm p_time q_time

/-- Theorem stating that two runners with given lap times meet after a specific time -/
theorem runners_meeting_time :
  meetingTime 252 198 = 2772 := by
  sorry

end NUMINAMATH_CALUDE_runners_meeting_time_l1025_102507


namespace NUMINAMATH_CALUDE_poles_not_moved_l1025_102546

theorem poles_not_moved (total_distance : ℕ) (original_spacing : ℕ) (new_spacing : ℕ) : 
  total_distance = 2340 ∧ 
  original_spacing = 45 ∧ 
  new_spacing = 60 → 
  (total_distance / (Nat.lcm original_spacing new_spacing)) - 1 = 12 := by
sorry

end NUMINAMATH_CALUDE_poles_not_moved_l1025_102546


namespace NUMINAMATH_CALUDE_calculate_expression_l1025_102573

theorem calculate_expression : 
  2 * Real.tan (60 * π / 180) - (-2023)^(0 : ℝ) + (1/2)^(-1 : ℝ) + |Real.sqrt 3 - 1| = 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l1025_102573


namespace NUMINAMATH_CALUDE_square_division_into_rectangles_l1025_102500

theorem square_division_into_rectangles :
  ∃ (s : ℝ), s > 0 ∧
  ∃ (a : ℝ), a > 0 ∧
  7 * (2 * a^2) ≤ s^2 ∧
  2 * a ≤ s :=
sorry

end NUMINAMATH_CALUDE_square_division_into_rectangles_l1025_102500


namespace NUMINAMATH_CALUDE_frank_pepe_height_difference_l1025_102591

-- Define the players
structure Player where
  name : String
  height : Float

-- Define the team
def team : List Player :=
  [
    { name := "Big Joe", height := 8 },
    { name := "Ben", height := 7 },
    { name := "Larry", height := 6 },
    { name := "Frank", height := 5.5 },
    { name := "Pepe", height := 4.5 }
  ]

-- Define the height difference function
def heightDifference (p1 p2 : Player) : Float :=
  p1.height - p2.height

-- Theorem statement
theorem frank_pepe_height_difference :
  let frank := team.find? (fun p => p.name = "Frank")
  let pepe := team.find? (fun p => p.name = "Pepe")
  ∀ (f p : Player), frank = some f → pepe = some p →
    heightDifference f p = 1 := by
  sorry

end NUMINAMATH_CALUDE_frank_pepe_height_difference_l1025_102591


namespace NUMINAMATH_CALUDE_sons_age_l1025_102572

theorem sons_age (father_age son_age : ℕ) 
  (h1 : 2 * son_age + father_age = 70)
  (h2 : 2 * father_age + son_age = 95)
  (h3 : father_age = 40) : son_age = 15 := by
  sorry

end NUMINAMATH_CALUDE_sons_age_l1025_102572


namespace NUMINAMATH_CALUDE_counterfeit_bag_identification_l1025_102550

/-- Represents a bag of coins -/
structure CoinBag where
  weight : ℕ  -- Weight of each coin in grams
  count : ℕ   -- Number of coins taken from the bag

/-- Creates a list of 10 coin bags with the specified counterfeit bag -/
def createBags (counterfeitBag : ℕ) : List CoinBag :=
  List.range 10 |>.map (fun i =>
    if i + 1 = counterfeitBag then
      { weight := 11, count := i + 1 }
    else
      { weight := 10, count := i + 1 })

/-- Calculates the total weight of coins from all bags -/
def totalWeight (bags : List CoinBag) : ℕ :=
  bags.foldl (fun acc bag => acc + bag.weight * bag.count) 0

/-- The main theorem to prove -/
theorem counterfeit_bag_identification
  (counterfeitBag : ℕ) (h1 : 1 ≤ counterfeitBag) (h2 : counterfeitBag ≤ 10) :
  totalWeight (createBags counterfeitBag) - 550 = counterfeitBag := by
  sorry

#check counterfeit_bag_identification

end NUMINAMATH_CALUDE_counterfeit_bag_identification_l1025_102550


namespace NUMINAMATH_CALUDE_stratified_sampling_example_l1025_102540

/-- Represents a population divided into two strata --/
structure Population :=
  (male_count : ℕ)
  (female_count : ℕ)

/-- Represents a sample taken from a population --/
structure Sample :=
  (male_count : ℕ)
  (female_count : ℕ)

/-- Defines a stratified sampling method --/
def is_stratified_sampling (pop : Population) (samp : Sample) : Prop :=
  (pop.male_count : ℚ) / (pop.male_count + pop.female_count) =
  (samp.male_count : ℚ) / (samp.male_count + samp.female_count)

/-- The theorem to be proved --/
theorem stratified_sampling_example :
  let pop := Population.mk 500 400
  let samp := Sample.mk 25 20
  is_stratified_sampling pop samp :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_example_l1025_102540


namespace NUMINAMATH_CALUDE_circle_ring_area_floor_l1025_102513

theorem circle_ring_area_floor :
  let r : ℝ := 30 / 3 -- radius of small circles
  let R : ℝ := 30 -- radius of large circle C
  let K : ℝ := 3 * Real.pi * r^2 -- area between large circle and six small circles
  ⌊K⌋ = 942 := by
  sorry

end NUMINAMATH_CALUDE_circle_ring_area_floor_l1025_102513


namespace NUMINAMATH_CALUDE_binomial_coefficient_modulo_prime_l1025_102530

theorem binomial_coefficient_modulo_prime (p : ℕ) (hp : Nat.Prime p) : 
  (Nat.choose (2 * p) p) ≡ 2 [MOD p] := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_modulo_prime_l1025_102530


namespace NUMINAMATH_CALUDE_power_three_sum_l1025_102588

theorem power_three_sum (m n : ℕ+) (x y : ℝ) 
  (hx : 3^(m.val) = x) 
  (hy : 9^(n.val) = y) : 
  3^(m.val + 2*n.val) = x * y := by
sorry

end NUMINAMATH_CALUDE_power_three_sum_l1025_102588


namespace NUMINAMATH_CALUDE_song_size_calculation_l1025_102568

/-- Given a total number of songs and total memory space occupied,
    calculate the size of each song. -/
def song_size (total_songs : ℕ) (total_memory : ℕ) : ℚ :=
  total_memory / total_songs

theorem song_size_calculation :
  let morning_songs : ℕ := 10
  let later_songs : ℕ := 15
  let night_songs : ℕ := 3
  let total_songs : ℕ := morning_songs + later_songs + night_songs
  let total_memory : ℕ := 140
  song_size total_songs total_memory = 5 := by
  sorry

end NUMINAMATH_CALUDE_song_size_calculation_l1025_102568


namespace NUMINAMATH_CALUDE_a_18_value_l1025_102585

def equal_sum_sequence (a : ℕ → ℝ) :=
  ∃ k : ℝ, ∀ n : ℕ, a n + a (n + 1) = k

theorem a_18_value (a : ℕ → ℝ) (h1 : equal_sum_sequence a) (h2 : a 1 = 2) (h3 : ∃ k : ℝ, k = 5 ∧ ∀ n : ℕ, a n + a (n + 1) = k) :
  a 18 = 3 := by
  sorry

end NUMINAMATH_CALUDE_a_18_value_l1025_102585


namespace NUMINAMATH_CALUDE_blue_to_red_ratio_l1025_102534

/-- Represents the number of socks of each color --/
structure SockCount where
  blue : ℕ
  black : ℕ
  red : ℕ
  white : ℕ

/-- The conditions of Joseph's sock collection --/
def josephsSocks : SockCount → Prop :=
  fun s => s.blue = s.black + 6 ∧
           s.red = s.white - 2 ∧
           s.red = 6 ∧
           s.blue + s.black + s.red + s.white = 28

/-- The theorem stating the ratio of blue to red socks --/
theorem blue_to_red_ratio (s : SockCount) (h : josephsSocks s) :
  s.blue / s.red = 7 / 3 := by
  sorry

end NUMINAMATH_CALUDE_blue_to_red_ratio_l1025_102534


namespace NUMINAMATH_CALUDE_fan_airflow_rate_l1025_102558

/-- Proves that the airflow rate of a fan is 10 liters per second, given the specified conditions. -/
theorem fan_airflow_rate : 
  ∀ (daily_operation_minutes : ℝ) (weekly_airflow_liters : ℝ),
    daily_operation_minutes = 10 →
    weekly_airflow_liters = 42000 →
    (weekly_airflow_liters / (daily_operation_minutes * 7 * 60)) = 10 := by
  sorry

end NUMINAMATH_CALUDE_fan_airflow_rate_l1025_102558


namespace NUMINAMATH_CALUDE_water_added_to_container_l1025_102589

/-- The amount of water added to a container -/
def water_added (capacity : ℝ) (initial_fraction : ℝ) (final_fraction : ℝ) : ℝ :=
  capacity * final_fraction - capacity * initial_fraction

/-- Theorem stating the amount of water added to the container -/
theorem water_added_to_container : 
  water_added 80 0.4 0.75 = 28 := by sorry

end NUMINAMATH_CALUDE_water_added_to_container_l1025_102589


namespace NUMINAMATH_CALUDE_election_winner_percentage_l1025_102512

theorem election_winner_percentage (total_votes winner_votes margin : ℕ) : 
  winner_votes = 992 →
  margin = 384 →
  total_votes = winner_votes + (winner_votes - margin) →
  (winner_votes : ℚ) / (total_votes : ℚ) = 62 / 100 :=
by
  sorry

end NUMINAMATH_CALUDE_election_winner_percentage_l1025_102512


namespace NUMINAMATH_CALUDE_count_integers_satisfying_equation_l1025_102505

-- Define the function g
def g (n : ℤ) : ℤ := ⌈(101 * n : ℚ) / 102⌉ - ⌊(102 * n : ℚ) / 103⌋

-- State the theorem
theorem count_integers_satisfying_equation : 
  (∃ (S : Finset ℤ), (∀ n ∈ S, g n = 1) ∧ (∀ n ∉ S, g n ≠ 1) ∧ Finset.card S = 10506) :=
sorry

end NUMINAMATH_CALUDE_count_integers_satisfying_equation_l1025_102505


namespace NUMINAMATH_CALUDE_mindy_tax_rate_is_25_percent_l1025_102552

/-- Calculates Mindy's tax rate given Mork's tax rate, their income ratio, and combined tax rate -/
def mindyTaxRate (morkTaxRate : ℚ) (incomeRatio : ℚ) (combinedTaxRate : ℚ) : ℚ :=
  (combinedTaxRate * (1 + incomeRatio) - morkTaxRate) / incomeRatio

/-- Proves that Mindy's tax rate is 25% given the specified conditions -/
theorem mindy_tax_rate_is_25_percent :
  mindyTaxRate (40 / 100) 4 (28 / 100) = 25 / 100 := by
  sorry

#eval mindyTaxRate (40 / 100) 4 (28 / 100)

end NUMINAMATH_CALUDE_mindy_tax_rate_is_25_percent_l1025_102552


namespace NUMINAMATH_CALUDE_ms_jones_class_size_l1025_102531

theorem ms_jones_class_size :
  ∀ (num_students : ℕ),
    (num_students : ℝ) * 0.3 * (1/3) * 10 = 50 →
    num_students = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_ms_jones_class_size_l1025_102531


namespace NUMINAMATH_CALUDE_green_squares_count_l1025_102556

/-- Represents a grid with colored squares -/
structure ColoredGrid where
  rows : Nat
  squares_per_row : Nat
  red_rows : Nat
  red_squares_per_row : Nat
  blue_rows : Nat

/-- Calculates the number of green squares in the grid -/
def green_squares (grid : ColoredGrid) : Nat :=
  grid.rows * grid.squares_per_row - 
  (grid.red_rows * grid.red_squares_per_row + grid.blue_rows * grid.squares_per_row)

/-- Theorem stating that the number of green squares in the given grid configuration is 66 -/
theorem green_squares_count (grid : ColoredGrid) 
  (h1 : grid.rows = 10)
  (h2 : grid.squares_per_row = 15)
  (h3 : grid.red_rows = 4)
  (h4 : grid.red_squares_per_row = 6)
  (h5 : grid.blue_rows = 4) :
  green_squares grid = 66 := by
  sorry

#eval green_squares { rows := 10, squares_per_row := 15, red_rows := 4, red_squares_per_row := 6, blue_rows := 4 }

end NUMINAMATH_CALUDE_green_squares_count_l1025_102556


namespace NUMINAMATH_CALUDE_f_negative_eight_equals_three_l1025_102524

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def has_period_property (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 4) = f x + 2 * f 2

theorem f_negative_eight_equals_three
  (f : ℝ → ℝ)
  (h_even : is_even_function f)
  (h_period : has_period_property f)
  (h_f_zero : f 0 = 3) :
  f (-8) = 3 := by
sorry

end NUMINAMATH_CALUDE_f_negative_eight_equals_three_l1025_102524


namespace NUMINAMATH_CALUDE_semicircle_area_shaded_area_proof_l1025_102548

/-- The area of semicircles lined up along a line -/
theorem semicircle_area (diameter : Real) (length : Real) : 
  diameter > 0 → length > 0 → 
  (length / diameter) * (π * diameter^2 / 8) = 3 * π * length / 2 := by
  sorry

/-- The specific case for the given problem -/
theorem shaded_area_proof :
  let diameter : Real := 4
  let length : Real := 24  -- 2 feet in inches
  (length / diameter) * (π * diameter^2 / 8) = 12 * π := by
  sorry

end NUMINAMATH_CALUDE_semicircle_area_shaded_area_proof_l1025_102548


namespace NUMINAMATH_CALUDE_max_correct_is_23_l1025_102566

/-- Represents a test score --/
structure TestScore where
  total_questions : ℕ
  correct_points : ℤ
  incorrect_points : ℤ
  total_score : ℤ

/-- Calculates the maximum number of correct answers for a given test score --/
def max_correct_answers (ts : TestScore) : ℕ :=
  sorry

/-- Theorem stating that for the given test conditions, the maximum number of correct answers is 23 --/
theorem max_correct_is_23 :
  let ts : TestScore := {
    total_questions := 30,
    correct_points := 4,
    incorrect_points := -1,
    total_score := 85
  }
  max_correct_answers ts = 23 := by
  sorry

end NUMINAMATH_CALUDE_max_correct_is_23_l1025_102566


namespace NUMINAMATH_CALUDE_grape_rate_calculation_l1025_102569

/-- The rate per kg for grapes -/
def grape_rate : ℝ := 68

/-- The amount of grapes purchased in kg -/
def grape_amount : ℝ := 7

/-- The rate per kg for mangoes -/
def mango_rate : ℝ := 48

/-- The amount of mangoes purchased in kg -/
def mango_amount : ℝ := 9

/-- The total amount paid -/
def total_paid : ℝ := 908

theorem grape_rate_calculation :
  grape_rate * grape_amount + mango_rate * mango_amount = total_paid :=
by sorry

end NUMINAMATH_CALUDE_grape_rate_calculation_l1025_102569


namespace NUMINAMATH_CALUDE_share_price_is_31_l1025_102539

/-- The price at which an investor bought shares, given the dividend rate,
    face value, and return on investment. -/
def share_purchase_price (dividend_rate : ℚ) (face_value : ℚ) (roi : ℚ) : ℚ :=
  (dividend_rate * face_value) / roi

/-- Theorem stating that under the given conditions, the share purchase price is 31. -/
theorem share_price_is_31 :
  let dividend_rate : ℚ := 155 / 1000
  let face_value : ℚ := 50
  let roi : ℚ := 1 / 4
  share_purchase_price dividend_rate face_value roi = 31 := by
  sorry

end NUMINAMATH_CALUDE_share_price_is_31_l1025_102539


namespace NUMINAMATH_CALUDE_order_of_surds_l1025_102582

theorem order_of_surds : 
  let a : ℝ := Real.sqrt 5 - Real.sqrt 3
  let b : ℝ := Real.sqrt 3 - 1
  let c : ℝ := Real.sqrt 7 - Real.sqrt 5
  b > a ∧ a > c := by sorry

end NUMINAMATH_CALUDE_order_of_surds_l1025_102582


namespace NUMINAMATH_CALUDE_no_integer_solution_l1025_102555

theorem no_integer_solution : ¬∃ (a b c d : ℤ), 
  (a * 19^3 + b * 19^2 + c * 19 + d = 1) ∧ 
  (a * 93^3 + b * 93^2 + c * 93 + d = 2) := by
sorry

end NUMINAMATH_CALUDE_no_integer_solution_l1025_102555


namespace NUMINAMATH_CALUDE_doubled_side_cube_weight_l1025_102547

/-- Represents the weight of a cube given its side length -/
def cube_weight (side_length : ℝ) : ℝ := sorry

theorem doubled_side_cube_weight (original_side : ℝ) :
  cube_weight original_side = 6 →
  cube_weight (2 * original_side) = 48 := by
  sorry

end NUMINAMATH_CALUDE_doubled_side_cube_weight_l1025_102547


namespace NUMINAMATH_CALUDE_rooks_knight_move_theorem_l1025_102596

/-- Represents a position on a chessboard -/
structure Position :=
  (row : Fin 8)
  (col : Fin 8)

/-- Represents a knight's move -/
structure KnightMove :=
  (drow : Int)
  (dcol : Int)

/-- Checks if a move is a valid knight's move -/
def isValidKnightMove (km : KnightMove) : Prop :=
  (km.drow.natAbs = 2 ∧ km.dcol.natAbs = 1) ∨ 
  (km.drow.natAbs = 1 ∧ km.dcol.natAbs = 2)

/-- Applies a knight's move to a position -/
def applyMove (p : Position) (km : KnightMove) : Position :=
  ⟨p.row + km.drow, p.col + km.dcol⟩

/-- Checks if two positions are non-attacking for rooks -/
def nonAttacking (p1 p2 : Position) : Prop :=
  p1.row ≠ p2.row ∧ p1.col ≠ p2.col

/-- The main theorem -/
theorem rooks_knight_move_theorem 
  (initial_positions : Fin 8 → Position)
  (h_initial_non_attacking : ∀ i j, i ≠ j → 
    nonAttacking (initial_positions i) (initial_positions j)) :
  ∃ (moves : Fin 8 → KnightMove),
    (∀ i, isValidKnightMove (moves i)) ∧
    (∀ i j, i ≠ j → 
      nonAttacking 
        (applyMove (initial_positions i) (moves i))
        (applyMove (initial_positions j) (moves j))) :=
  sorry


end NUMINAMATH_CALUDE_rooks_knight_move_theorem_l1025_102596


namespace NUMINAMATH_CALUDE_subtraction_of_decimals_l1025_102543

theorem subtraction_of_decimals : 25.019 - 3.2663 = 21.7527 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_decimals_l1025_102543


namespace NUMINAMATH_CALUDE_sqrt_equality_condition_l1025_102522

theorem sqrt_equality_condition (a b c : ℝ) :
  Real.sqrt (4 * a^2 + 9 * b^2) = 2 * a + 3 * b + c ↔
  12 * a * b + 4 * a * c + 6 * b * c + c^2 = 0 ∧ 2 * a + 3 * b + c ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equality_condition_l1025_102522


namespace NUMINAMATH_CALUDE_trapezoid_diagonal_segments_l1025_102554

/-- 
Theorem: In a trapezoid with bases a and c, and sides b and d, 
the segments AO and OC of diagonal AC divided by diagonal BD are:
AO = (c / (a + c)) * √(ac + (ad² - cb²) / (a - c))
OC = (a / (a + c)) * √(ac + (ad² - cb²) / (a - c))
-/
theorem trapezoid_diagonal_segments 
  (a c b d : ℝ) 
  (ha : a > 0) 
  (hc : c > 0) 
  (hb : b > 0) 
  (hd : d > 0) 
  (hac : a ≠ c) :
  ∃ (AO OC : ℝ),
    AO = (c / (a + c)) * Real.sqrt (a * c + (a * d^2 - c * b^2) / (a - c)) ∧
    OC = (a / (a + c)) * Real.sqrt (a * c + (a * d^2 - c * b^2) / (a - c)) :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_diagonal_segments_l1025_102554


namespace NUMINAMATH_CALUDE_unique_divisible_number_l1025_102590

def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

def is_divisible_by_4 (n : ℕ) : Prop := n % 4 = 0

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 1000) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

def last_two_digits (n : ℕ) : ℕ := n % 100

theorem unique_divisible_number :
  ∃! D : ℕ, D < 10 ∧ 
    is_divisible_by_3 (sum_of_digits (1000 + D * 10 + 4)) ∧ 
    is_divisible_by_4 (last_two_digits (1000 + D * 10 + 4)) :=
sorry

end NUMINAMATH_CALUDE_unique_divisible_number_l1025_102590


namespace NUMINAMATH_CALUDE_same_color_marble_probability_l1025_102570

theorem same_color_marble_probability : 
  let total_marbles : ℕ := 3 + 6 + 8
  let red_marbles : ℕ := 3
  let white_marbles : ℕ := 6
  let blue_marbles : ℕ := 8
  let drawn_marbles : ℕ := 4
  
  (Nat.choose white_marbles drawn_marbles + Nat.choose blue_marbles drawn_marbles : ℚ) /
  (Nat.choose total_marbles drawn_marbles : ℚ) = 17 / 476 :=
by sorry

end NUMINAMATH_CALUDE_same_color_marble_probability_l1025_102570


namespace NUMINAMATH_CALUDE_number_reciprocal_relation_l1025_102510

theorem number_reciprocal_relation (x y : ℝ) : 
  x > 0 → x = 3 → x + y = 60 * (1 / x) → y = 17 := by
  sorry

end NUMINAMATH_CALUDE_number_reciprocal_relation_l1025_102510


namespace NUMINAMATH_CALUDE_balls_in_boxes_l1025_102562

theorem balls_in_boxes (n : ℕ) (k : ℕ) (h1 : n = 6) (h2 : k = 2) :
  (k : ℕ) ^ n = 64 := by
  sorry

end NUMINAMATH_CALUDE_balls_in_boxes_l1025_102562


namespace NUMINAMATH_CALUDE_abs_2x_plus_1_gt_3_l1025_102533

theorem abs_2x_plus_1_gt_3 (x : ℝ) : |2*x + 1| > 3 ↔ x > 1 ∨ x < -2 := by
  sorry

end NUMINAMATH_CALUDE_abs_2x_plus_1_gt_3_l1025_102533


namespace NUMINAMATH_CALUDE_jerome_money_theorem_l1025_102551

/-- Calculates the amount of money Jerome has left after giving money to Meg and Bianca. -/
def jerome_money_left (initial_money : ℕ) (meg_amount : ℕ) (bianca_multiplier : ℕ) : ℕ :=
  initial_money - meg_amount - (meg_amount * bianca_multiplier)

/-- Proves that Jerome has $54 left after giving money to Meg and Bianca. -/
theorem jerome_money_theorem :
  let initial_money := 43 * 2
  let meg_amount := 8
  let bianca_multiplier := 3
  jerome_money_left initial_money meg_amount bianca_multiplier = 54 := by
  sorry

end NUMINAMATH_CALUDE_jerome_money_theorem_l1025_102551


namespace NUMINAMATH_CALUDE_marble_group_size_l1025_102529

theorem marble_group_size :
  ∀ (x : ℕ),
  (144 / x : ℚ) = (144 / (x + 2) : ℚ) + 1 →
  x = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_marble_group_size_l1025_102529


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l1025_102576

theorem quadratic_inequality_solution_sets
  (a b c : ℝ)
  (h : Set.Ioo (-2 : ℝ) 1 = {x : ℝ | a * x^2 + b * x + c > 0}) :
  {x : ℝ | c * x^2 - b * x + a < 0} = Set.Ioo (-1 : ℝ) (1/2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l1025_102576


namespace NUMINAMATH_CALUDE_cupboard_sale_percentage_below_cost_l1025_102584

def cost_price : ℕ := 3750
def additional_amount : ℕ := 1200
def profit_percentage : ℚ := 16 / 100

def selling_price_with_profit : ℚ := cost_price + profit_percentage * cost_price
def actual_selling_price : ℚ := selling_price_with_profit - additional_amount

theorem cupboard_sale_percentage_below_cost (cost_price : ℕ) (additional_amount : ℕ) 
  (profit_percentage : ℚ) (selling_price_with_profit : ℚ) (actual_selling_price : ℚ) :
  (cost_price - actual_selling_price) / cost_price = 16 / 100 :=
by sorry

end NUMINAMATH_CALUDE_cupboard_sale_percentage_below_cost_l1025_102584


namespace NUMINAMATH_CALUDE_total_prairie_area_l1025_102583

def prairie_size (dust_covered : ℕ) (untouched : ℕ) : ℕ :=
  dust_covered + untouched

theorem total_prairie_area : prairie_size 64535 522 = 65057 := by
  sorry

end NUMINAMATH_CALUDE_total_prairie_area_l1025_102583


namespace NUMINAMATH_CALUDE_sqrt_x_minus_9_meaningful_l1025_102595

theorem sqrt_x_minus_9_meaningful (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 9) ↔ x ≥ 9 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_9_meaningful_l1025_102595


namespace NUMINAMATH_CALUDE_infinitely_many_special_integers_l1025_102553

/-- A function that checks if a number is a perfect square -/
def isPerfectSquare (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

/-- A function that checks if a number is a perfect cube -/
def isPerfectCube (n : ℕ) : Prop := ∃ k : ℕ, n = k^3

/-- A function that checks if a number is a perfect fifth power -/
def isPerfectFifthPower (n : ℕ) : Prop := ∃ k : ℕ, n = k^5

/-- The main theorem stating that there are infinitely many integers satisfying the conditions -/
theorem infinitely_many_special_integers :
  ∃ f : ℕ → ℕ, Function.Injective f ∧
    ∀ k : ℕ, 
      isPerfectSquare (2 * f k) ∧ 
      isPerfectCube (3 * f k) ∧ 
      isPerfectFifthPower (5 * f k) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_special_integers_l1025_102553


namespace NUMINAMATH_CALUDE_larger_number_problem_l1025_102581

theorem larger_number_problem (x y : ℤ) 
  (sum_is_62 : x + y = 62) 
  (y_is_larger : y = x + 12) : 
  y = 37 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l1025_102581


namespace NUMINAMATH_CALUDE_expression_factorization_l1025_102597

theorem expression_factorization (b : ℝ) :
  (3 * b^4 + 66 * b^3 - 14) - (-4 * b^4 + 2 * b^3 - 14) = b^3 * (7 * b + 64) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l1025_102597


namespace NUMINAMATH_CALUDE_kathleens_allowance_increase_l1025_102549

theorem kathleens_allowance_increase (middle_school_allowance senior_year_allowance : ℚ) : 
  middle_school_allowance = 8 + 2 →
  senior_year_allowance = 2 * middle_school_allowance + 5 →
  (senior_year_allowance - middle_school_allowance) / middle_school_allowance * 100 = 150 := by
  sorry

end NUMINAMATH_CALUDE_kathleens_allowance_increase_l1025_102549


namespace NUMINAMATH_CALUDE_ralph_square_matchsticks_ralph_uses_eight_matchsticks_per_square_l1025_102517

theorem ralph_square_matchsticks (total_matchsticks : ℕ) (elvis_matchsticks_per_square : ℕ) 
  (elvis_squares : ℕ) (ralph_squares : ℕ) (matchsticks_left : ℕ) : ℕ :=
  let elvis_total_matchsticks := elvis_matchsticks_per_square * elvis_squares
  let total_used_matchsticks := total_matchsticks - matchsticks_left
  let ralph_total_matchsticks := total_used_matchsticks - elvis_total_matchsticks
  ralph_total_matchsticks / ralph_squares

theorem ralph_uses_eight_matchsticks_per_square :
  ralph_square_matchsticks 50 4 5 3 6 = 8 := by
  sorry

end NUMINAMATH_CALUDE_ralph_square_matchsticks_ralph_uses_eight_matchsticks_per_square_l1025_102517


namespace NUMINAMATH_CALUDE_rectangle_length_width_difference_l1025_102563

theorem rectangle_length_width_difference
  (perimeter : ℝ)
  (diagonal : ℝ)
  (h_perimeter : perimeter = 80)
  (h_diagonal : diagonal = 20 * Real.sqrt 2) :
  ∃ (length width : ℝ),
    length > 0 ∧ width > 0 ∧
    2 * (length + width) = perimeter ∧
    length^2 + width^2 = diagonal^2 ∧
    length - width = 0 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_width_difference_l1025_102563


namespace NUMINAMATH_CALUDE_algorithm_computes_gcd_l1025_102508

/-- The algorithm described in the problem -/
def algorithm (x y : ℕ) : ℕ :=
  let rec loop (m n : ℕ) : ℕ :=
    if m / n = m / n then n
    else loop n (m % n)
  loop (max x y) (min x y)

/-- Theorem stating that the algorithm computes the GCD -/
theorem algorithm_computes_gcd (x y : ℕ) :
  algorithm x y = Nat.gcd x y := by sorry

end NUMINAMATH_CALUDE_algorithm_computes_gcd_l1025_102508


namespace NUMINAMATH_CALUDE_equation_solution_l1025_102519

theorem equation_solution : 
  ∃! x : ℝ, x ≠ 1 ∧ x ≠ -6 ∧ (3*x + 6)/((x^2 + 5*x - 6)) = (3 - x)/(x - 1) ∧ x = -4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1025_102519


namespace NUMINAMATH_CALUDE_log_condition_equivalence_l1025_102527

theorem log_condition_equivalence (m n : ℝ) 
  (hm_pos : m > 0) (hm_neq_one : m ≠ 1) (hn_pos : n > 0) :
  (Real.log n / Real.log m < 0) ↔ ((m - 1) * (n - 1) < 0) := by
  sorry

end NUMINAMATH_CALUDE_log_condition_equivalence_l1025_102527


namespace NUMINAMATH_CALUDE_inheritance_problem_l1025_102560

theorem inheritance_problem (x y z w : ℚ) : 
  (y = 0.75 * x) →
  (z = 0.5 * x) →
  (w = 0.25 * x) →
  (y = 45) →
  (z = 2 * w) →
  (x + y + z + w = 150) :=
by sorry

end NUMINAMATH_CALUDE_inheritance_problem_l1025_102560


namespace NUMINAMATH_CALUDE_die_game_expected_value_l1025_102561

/-- A fair 8-sided die game where you win the rolled amount if it's a multiple of 3 -/
def die_game : ℝ := by sorry

/-- The expected value of the die game -/
theorem die_game_expected_value : die_game = 2.25 := by sorry

end NUMINAMATH_CALUDE_die_game_expected_value_l1025_102561


namespace NUMINAMATH_CALUDE_function_behavior_implies_a_range_l1025_102564

/-- The function f(x) defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2 + (a-1)*x + 1

/-- The derivative of f(x) with respect to x -/
def f_prime (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*a*x + (a-1)

theorem function_behavior_implies_a_range :
  ∀ a : ℝ,
  (∀ x ∈ Set.Ioo 1 4, (f_prime a x) < 0) →
  (∀ x ∈ Set.Ioi 6, (f_prime a x) > 0) →
  5 ≤ a ∧ a ≤ 7 :=
by sorry

end NUMINAMATH_CALUDE_function_behavior_implies_a_range_l1025_102564


namespace NUMINAMATH_CALUDE_half_area_triangle_l1025_102526

/-- A square in a 2D plane -/
structure Square where
  x : ℝ × ℝ
  y : ℝ × ℝ
  z : ℝ × ℝ
  w : ℝ × ℝ

/-- The area of a triangle given three points -/
def triangleArea (a b c : ℝ × ℝ) : ℝ := sorry

/-- The area of a square -/
def squareArea (s : Square) : ℝ := sorry

/-- Theorem: The coordinates (3, 3) for point T' result in the area of triangle YZT' 
    being half the area of square XYZW, given that XYZW is a square with X at (0,0) 
    and Z at (3,3) -/
theorem half_area_triangle (xyzw : Square) 
  (h1 : xyzw.x = (0, 0))
  (h2 : xyzw.z = (3, 3))
  (t' : ℝ × ℝ)
  (h3 : t' = (3, 3)) : 
  triangleArea xyzw.y xyzw.z t' = (1/2) * squareArea xyzw := by
  sorry

end NUMINAMATH_CALUDE_half_area_triangle_l1025_102526


namespace NUMINAMATH_CALUDE_tree_height_difference_l1025_102574

/-- The height of the birch tree in feet -/
def birch_height : ℚ := 49/4

/-- The height of the pine tree in feet -/
def pine_height : ℚ := 37/2

/-- The difference in height between the pine tree and the birch tree -/
def height_difference : ℚ := pine_height - birch_height

theorem tree_height_difference : height_difference = 25/4 := by sorry

end NUMINAMATH_CALUDE_tree_height_difference_l1025_102574


namespace NUMINAMATH_CALUDE_variable_value_proof_l1025_102541

theorem variable_value_proof (x a k some_variable : ℝ) :
  (3 * x + 2) * (2 * x - 7) = a * x^2 + k * x + some_variable →
  a - some_variable + k = 3 →
  some_variable = -14 := by
  sorry

end NUMINAMATH_CALUDE_variable_value_proof_l1025_102541


namespace NUMINAMATH_CALUDE_adult_males_in_town_l1025_102518

/-- Represents the population distribution in a small town -/
structure TownPopulation where
  total : ℕ
  ratio_children : ℕ
  ratio_adult_males : ℕ
  ratio_adult_females : ℕ

/-- Calculates the number of adult males in the town -/
def adult_males (town : TownPopulation) : ℕ :=
  let total_ratio := town.ratio_children + town.ratio_adult_males + town.ratio_adult_females
  (town.total / total_ratio) * town.ratio_adult_males

/-- Theorem stating the number of adult males in the specific town -/
theorem adult_males_in_town (town : TownPopulation) 
  (h1 : town.total = 480)
  (h2 : town.ratio_children = 1)
  (h3 : town.ratio_adult_males = 2)
  (h4 : town.ratio_adult_females = 2) :
  adult_males town = 192 := by
  sorry

end NUMINAMATH_CALUDE_adult_males_in_town_l1025_102518


namespace NUMINAMATH_CALUDE_wuyang_football_school_runners_l1025_102544

theorem wuyang_football_school_runners (x : ℕ) : 
  (x - 4) % 2 = 0 →
  (x - 5) % 3 = 0 →
  x % 5 = 0 →
  ∃ n : ℕ, x = n ^ 2 →
  250 - 10 ≤ x - 3 ∧ x - 3 ≤ 250 + 10 →
  x = 260 := by
sorry

end NUMINAMATH_CALUDE_wuyang_football_school_runners_l1025_102544


namespace NUMINAMATH_CALUDE_scale_division_l1025_102557

/-- Proves that dividing a scale of length 7 feet 12 inches into 4 equal parts results in parts that are 2 feet long each. -/
theorem scale_division (scale_length_feet : ℕ) (scale_length_inches : ℕ) (num_parts : ℕ) :
  scale_length_feet = 7 →
  scale_length_inches = 12 →
  num_parts = 4 →
  (scale_length_feet * 12 + scale_length_inches) / num_parts = 24 := by
  sorry

#check scale_division

end NUMINAMATH_CALUDE_scale_division_l1025_102557


namespace NUMINAMATH_CALUDE_cube_root_8000_l1025_102535

theorem cube_root_8000 :
  ∃ (c d : ℕ+), (c : ℝ) * (d : ℝ)^(1/3 : ℝ) = (8000 : ℝ)^(1/3 : ℝ) ∧
  c = 20 ∧ d = 1 ∧ c + d = 21 ∧
  ∀ (c' d' : ℕ+), (c' : ℝ) * (d' : ℝ)^(1/3 : ℝ) = (8000 : ℝ)^(1/3 : ℝ) → d ≤ d' :=
by sorry

end NUMINAMATH_CALUDE_cube_root_8000_l1025_102535


namespace NUMINAMATH_CALUDE_wire_ratio_proof_l1025_102525

theorem wire_ratio_proof (total_length shorter_length : ℕ) 
  (h1 : total_length = 140)
  (h2 : shorter_length = 40) :
  shorter_length * 5 = (total_length - shorter_length) * 2 := by
sorry

end NUMINAMATH_CALUDE_wire_ratio_proof_l1025_102525


namespace NUMINAMATH_CALUDE_tangent_line_and_inequality_l1025_102578

noncomputable section

-- Define the functions f and g
def f (x : ℝ) := x * Real.log x
def g (a : ℝ) (x : ℝ) := -x^2 + a*x - 3

-- State the theorem
theorem tangent_line_and_inequality (a : ℝ) :
  -- Part 1: Tangent line equation
  (∀ x : ℝ, HasDerivAt f (x - 1) 1) ∧
  -- Part 2: Inequality condition
  (∀ x : ℝ, x > 0 → 2 * f x ≥ g a x) ↔ a ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_and_inequality_l1025_102578


namespace NUMINAMATH_CALUDE_min_value_theorem_l1025_102587

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 1) :
  ∃ (min : ℝ), min = 6 ∧ ∀ (x y : ℝ), x > 0 → y > 0 → 1/x + 1/y = 1 → 1/(x-1) + 9/(y-1) ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1025_102587


namespace NUMINAMATH_CALUDE_function_properties_l1025_102511

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x ≤ f y

theorem function_properties
  (f : ℝ → ℝ)
  (h_even : is_even f)
  (h_neg : ∀ x, f (x + 1) = -f x)
  (h_incr : is_increasing_on f (-1) 0) :
  (∀ x, f (x + 2) = f x) ∧
  (∀ x, f (2 - x) = f x) ∧
  f 2 = f 0 :=
sorry

end NUMINAMATH_CALUDE_function_properties_l1025_102511


namespace NUMINAMATH_CALUDE_compound_carbon_atoms_l1025_102516

/-- Represents the number of Carbon atoms in a compound -/
def carbonAtoms (molecularWeight : ℕ) (hydrogenAtoms : ℕ) : ℕ :=
  (molecularWeight - hydrogenAtoms) / 12

/-- Proves that a compound with 6 Hydrogen atoms and a molecular weight of 78 amu contains 6 Carbon atoms -/
theorem compound_carbon_atoms :
  carbonAtoms 78 6 = 6 :=
by
  sorry

#eval carbonAtoms 78 6

end NUMINAMATH_CALUDE_compound_carbon_atoms_l1025_102516


namespace NUMINAMATH_CALUDE_quadratic_properties_l1025_102559

-- Define the quadratic function
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_properties (a b c m : ℝ) (h_a : a ≠ 0) :
  quadratic a b c (-2) = m ∧
  quadratic a b c (-1) = 1 ∧
  quadratic a b c 0 = -1 ∧
  quadratic a b c 1 = 1 ∧
  quadratic a b c 2 = 7 →
  (∀ x, quadratic a b c x = quadratic a b c (-x)) ∧  -- Symmetry axis at x = 0
  quadratic a b c 0 = -1 ∧                           -- Vertex at (0, -1)
  m = 7 ∧
  a > 0 := by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l1025_102559


namespace NUMINAMATH_CALUDE_no_solutions_in_interval_l1025_102579

theorem no_solutions_in_interval (a : ℤ) : 
  (∀ x ∈ Set.Icc (-1 : ℝ) 7, (x - 2 * (a : ℝ) + 1)^2 - 2*x + 4*(a : ℝ) - 10 ≠ 0) ↔ 
  (a ≤ -3 ∨ a ≥ 6) :=
sorry

end NUMINAMATH_CALUDE_no_solutions_in_interval_l1025_102579


namespace NUMINAMATH_CALUDE_decimal_division_multiplication_l1025_102506

theorem decimal_division_multiplication : (0.08 / 0.005) * 2 = 32 := by sorry

end NUMINAMATH_CALUDE_decimal_division_multiplication_l1025_102506


namespace NUMINAMATH_CALUDE_consecutive_divisible_numbers_l1025_102538

theorem consecutive_divisible_numbers :
  ∃ (n : ℕ),
    (5 ∣ n) ∧
    (4 ∣ n + 1) ∧
    (3 ∣ n + 2) ∧
    (∀ (m : ℕ), (5 ∣ m) ∧ (4 ∣ m + 1) ∧ (3 ∣ m + 2) → n ≤ m) ∧
    n = 55 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_divisible_numbers_l1025_102538


namespace NUMINAMATH_CALUDE_parabola_symmetry_l1025_102520

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line with inclination angle -/
structure Line where
  angle : ℝ

/-- Function to check if a point is on the parabola -/
def on_parabola (para : Parabola) (pt : Point) : Prop :=
  pt.y^2 = 2 * para.p * pt.x

/-- Function to check if a line passes through a point -/
def passes_through (l : Line) (pt : Point) : Prop :=
  true -- Simplified for this problem

/-- Function to check if two points are symmetric with respect to a line -/
def symmetric_wrt_line (p1 p2 : Point) (l : Line) : Prop :=
  true -- Simplified for this problem

/-- Main theorem -/
theorem parabola_symmetry (para : Parabola) (l : Line) (p q : Point) :
  l.angle = π / 6 →
  passes_through l (Point.mk (para.p / 2) 0) →
  on_parabola para p →
  q = Point.mk 5 0 →
  symmetric_wrt_line p q l →
  para.p = 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_symmetry_l1025_102520


namespace NUMINAMATH_CALUDE_grandfather_animals_l1025_102504

theorem grandfather_animals (h p k s : ℕ) : 
  h + p + k + s = 40 →
  h = 3 * k →
  s - 8 = h + p →
  40 - (1/4) * h + (3/4) * h = 46 →
  h = 12 ∧ p = 2 ∧ k = 4 ∧ s = 22 := by sorry

end NUMINAMATH_CALUDE_grandfather_animals_l1025_102504


namespace NUMINAMATH_CALUDE_matt_writing_difference_l1025_102567

/-- The number of words Matt can write per minute with his right hand -/
def right_hand_speed : ℕ := 10

/-- The number of words Matt can write per minute with his left hand -/
def left_hand_speed : ℕ := 7

/-- The duration of time in minutes -/
def duration : ℕ := 5

/-- The difference in words written between Matt's right and left hands over the given duration -/
def word_difference : ℕ := (right_hand_speed - left_hand_speed) * duration

theorem matt_writing_difference : word_difference = 15 := by
  sorry

end NUMINAMATH_CALUDE_matt_writing_difference_l1025_102567


namespace NUMINAMATH_CALUDE_infinite_hyperbolas_l1025_102575

/-- A hyperbola with asymptotes 2x ± 3y = 0 -/
def Hyperbola (k : ℝ) : Prop :=
  ∃ (x y : ℝ), 4 * x^2 - 9 * y^2 = k ∧ k ≠ 0

/-- The set of all hyperbolas with asymptotes 2x ± 3y = 0 -/
def HyperbolaSet : Set ℝ :=
  {k : ℝ | Hyperbola k}

/-- Theorem stating that there are infinitely many hyperbolas with asymptotes 2x ± 3y = 0 -/
theorem infinite_hyperbolas : Set.Infinite HyperbolaSet := by
  sorry

end NUMINAMATH_CALUDE_infinite_hyperbolas_l1025_102575


namespace NUMINAMATH_CALUDE_leroy_payment_l1025_102594

/-- The amount LeRoy must pay to equalize costs on a shared trip -/
theorem leroy_payment (A B C : ℝ) (h1 : A < B) (h2 : B < C) : 
  (B + C - 2 * A) / 3 = (A + B + C) / 3 - A := by
  sorry

#check leroy_payment

end NUMINAMATH_CALUDE_leroy_payment_l1025_102594


namespace NUMINAMATH_CALUDE_equation_solution_l1025_102598

theorem equation_solution : ∃ n : ℤ, 5^3 - 7 = 6^2 + n ∧ n = 82 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1025_102598


namespace NUMINAMATH_CALUDE_probability_zeros_not_adjacent_l1025_102592

-- Define the total number of elements
def total_elements : ℕ := 5

-- Define the number of ones
def num_ones : ℕ := 3

-- Define the number of zeros
def num_zeros : ℕ := 2

-- Define the total number of arrangements
def total_arrangements : ℕ := Nat.factorial total_elements

-- Define the number of arrangements where zeros are adjacent
def adjacent_zero_arrangements : ℕ := 2 * Nat.factorial (total_elements - 1)

-- Statement to prove
theorem probability_zeros_not_adjacent :
  (1 : ℚ) - (adjacent_zero_arrangements : ℚ) / total_arrangements = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_probability_zeros_not_adjacent_l1025_102592


namespace NUMINAMATH_CALUDE_infinitely_many_primes_dividing_special_form_l1025_102509

-- Define the set of primes we're interested in
def S : Set Nat :=
  {p | Nat.Prime p ∧ ∃ n : Nat, n > 0 ∧ p ∣ (2014^(2^n) + 2014)}

-- State the theorem
theorem infinitely_many_primes_dividing_special_form :
  Set.Infinite S :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_primes_dividing_special_form_l1025_102509


namespace NUMINAMATH_CALUDE_dan_picked_nine_apples_l1025_102599

/-- The number of apples Benny picked -/
def benny_apples : ℕ := 2

/-- The total number of apples picked by Benny and Dan -/
def total_apples : ℕ := 11

/-- The number of apples Dan picked -/
def dan_apples : ℕ := total_apples - benny_apples

theorem dan_picked_nine_apples : dan_apples = 9 := by
  sorry

end NUMINAMATH_CALUDE_dan_picked_nine_apples_l1025_102599


namespace NUMINAMATH_CALUDE_fruit_difference_l1025_102503

/-- Given the number of apples harvested and the ratio of peaches to apples,
    prove that the difference between the number of peaches and apples is 120. -/
theorem fruit_difference (apples : ℕ) (peach_ratio : ℕ) : apples = 60 → peach_ratio = 3 →
  peach_ratio * apples - apples = 120 := by
  sorry

end NUMINAMATH_CALUDE_fruit_difference_l1025_102503


namespace NUMINAMATH_CALUDE_min_marked_cells_13x13_board_l1025_102580

/-- Represents a rectangular board --/
structure Board :=
  (rows : Nat)
  (cols : Nat)

/-- Represents a rectangle that can be placed on the board --/
structure Rectangle :=
  (length : Nat)
  (width : Nat)

/-- Function to calculate the minimum number of cells to mark --/
def min_marked_cells (b : Board) (r : Rectangle) : Nat :=
  sorry

/-- Theorem stating the minimum number of cells to mark for the given problem --/
theorem min_marked_cells_13x13_board (b : Board) (r : Rectangle) :
  b.rows = 13 ∧ b.cols = 13 ∧ r.length = 6 ∧ r.width = 1 →
  min_marked_cells b r = 84 :=
sorry

end NUMINAMATH_CALUDE_min_marked_cells_13x13_board_l1025_102580


namespace NUMINAMATH_CALUDE_hot_dogs_remainder_l1025_102545

theorem hot_dogs_remainder : 25197631 % 17 = 10 := by
  sorry

end NUMINAMATH_CALUDE_hot_dogs_remainder_l1025_102545


namespace NUMINAMATH_CALUDE_shirts_not_washed_l1025_102586

theorem shirts_not_washed (short_sleeve : ℕ) (long_sleeve : ℕ) (washed : ℕ) 
  (h1 : short_sleeve = 9)
  (h2 : long_sleeve = 21)
  (h3 : washed = 29) :
  short_sleeve + long_sleeve - washed = 1 := by
  sorry

end NUMINAMATH_CALUDE_shirts_not_washed_l1025_102586


namespace NUMINAMATH_CALUDE_edward_money_problem_l1025_102537

theorem edward_money_problem (initial spent received final : ℤ) :
  spent = 17 →
  received = 10 →
  final = 7 →
  initial - spent + received = final →
  initial = 14 := by
sorry

end NUMINAMATH_CALUDE_edward_money_problem_l1025_102537


namespace NUMINAMATH_CALUDE_exists_monochromatic_equilateral_triangle_l1025_102565

-- Define a color type
inductive Color
| White
| Black

-- Define a point in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a coloring function
def coloring : Point → Color := sorry

-- Define an equilateral triangle
structure EquilateralTriangle where
  a : Point
  b : Point
  c : Point
  is_equilateral : sorry

-- Theorem statement
theorem exists_monochromatic_equilateral_triangle :
  ∃ (t : EquilateralTriangle), 
    coloring t.a = coloring t.b ∧ coloring t.b = coloring t.c :=
sorry

end NUMINAMATH_CALUDE_exists_monochromatic_equilateral_triangle_l1025_102565


namespace NUMINAMATH_CALUDE_max_value_expression_l1025_102593

def A : Set Int := {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5}

theorem max_value_expression (v w x y z : Int) 
  (hv : v ∈ A) (hw : w ∈ A) (hx : x ∈ A) (hy : y ∈ A) (hz : z ∈ A)
  (h_vw : v * w = x) (h_w : w ≠ 0) :
  (∀ v' w' x' y' z' : Int, 
    v' ∈ A → w' ∈ A → x' ∈ A → y' ∈ A → z' ∈ A → 
    v' * w' = x' → w' ≠ 0 →
    v * x - y * z ≥ v' * x' - y' * z') →
  v * x - y * z = 150 :=
sorry

end NUMINAMATH_CALUDE_max_value_expression_l1025_102593


namespace NUMINAMATH_CALUDE_problem_solution_l1025_102501

theorem problem_solution (x y : ℝ) (h1 : x^(2*y) = 81) (h2 : x = 9) : y = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1025_102501


namespace NUMINAMATH_CALUDE_modulus_of_z_l1025_102521

def z : ℂ := 3 + 4 * Complex.I

theorem modulus_of_z : Complex.abs z = 5 := by sorry

end NUMINAMATH_CALUDE_modulus_of_z_l1025_102521


namespace NUMINAMATH_CALUDE_correct_sampling_methods_l1025_102502

-- Define the community structure
structure Community where
  high_income : Nat
  middle_income : Nat
  low_income : Nat

-- Define the student group
structure StudentGroup where
  total : Nat

-- Define sampling methods
inductive SamplingMethod
  | Stratified
  | SimpleRandom
  | Systematic

-- Define the function to determine the correct sampling method for the community survey
def community_sampling_method (c : Community) (sample_size : Nat) : SamplingMethod :=
  sorry

-- Define the function to determine the correct sampling method for the student survey
def student_sampling_method (s : StudentGroup) (sample_size : Nat) : SamplingMethod :=
  sorry

-- Theorem stating the correct sampling methods for both surveys
theorem correct_sampling_methods 
  (community : Community)
  (students : StudentGroup) :
  community_sampling_method {high_income := 100, middle_income := 210, low_income := 90} 100 = SamplingMethod.Stratified ∧
  student_sampling_method {total := 10} 3 = SamplingMethod.SimpleRandom :=
sorry

end NUMINAMATH_CALUDE_correct_sampling_methods_l1025_102502


namespace NUMINAMATH_CALUDE_no_natural_solutions_l1025_102515

theorem no_natural_solutions (k x y z : ℕ) (h : k > 3) :
  x^2 + y^2 + z^2 ≠ k * x * y * z :=
sorry

end NUMINAMATH_CALUDE_no_natural_solutions_l1025_102515


namespace NUMINAMATH_CALUDE_brothers_age_difference_l1025_102532

/-- Represents a year in the format ABCD where A, B, C, D are digits --/
structure Year :=
  (value : ℕ)
  (in_19th_century : value ≥ 1800 ∧ value < 1900)

/-- Represents a year in the format ABCD where A, B, C, D are digits --/
structure Year' :=
  (value : ℕ)
  (in_20th_century : value ≥ 1900 ∧ value < 2000)

/-- Sum of digits of a number --/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

/-- Age of a person born in year y at the current year current_year --/
def age (y : ℕ) (current_year : ℕ) : ℕ := current_year - y

theorem brothers_age_difference 
  (peter_birth : Year) 
  (paul_birth : Year') 
  (current_year : ℕ) 
  (h1 : age peter_birth.value current_year = sum_of_digits peter_birth.value)
  (h2 : age paul_birth.value current_year = sum_of_digits paul_birth.value) :
  age peter_birth.value current_year - age paul_birth.value current_year = 9 :=
sorry

end NUMINAMATH_CALUDE_brothers_age_difference_l1025_102532


namespace NUMINAMATH_CALUDE_union_of_sets_l1025_102542

theorem union_of_sets (M N : Set ℕ) : 
  M = {0, 1, 3} → 
  N = {x | ∃ a ∈ M, x = 3 * a} → 
  M ∪ N = {0, 1, 3, 9} := by
sorry

end NUMINAMATH_CALUDE_union_of_sets_l1025_102542


namespace NUMINAMATH_CALUDE_largest_integer_less_than_M_div_100_l1025_102577

def factorial (n : ℕ) : ℕ := Nat.factorial n

def M : ℚ :=
  (1 / (factorial 3 * factorial 19) +
   1 / (factorial 4 * factorial 18) +
   1 / (factorial 5 * factorial 17) +
   1 / (factorial 6 * factorial 16) +
   1 / (factorial 7 * factorial 15) +
   1 / (factorial 8 * factorial 14) +
   1 / (factorial 9 * factorial 13) +
   1 / (factorial 10 * factorial 12)) * (factorial 1 * factorial 21)

theorem largest_integer_less_than_M_div_100 :
  Int.floor (M / 100) = 952 := by sorry

end NUMINAMATH_CALUDE_largest_integer_less_than_M_div_100_l1025_102577
