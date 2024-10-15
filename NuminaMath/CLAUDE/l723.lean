import Mathlib

namespace NUMINAMATH_CALUDE_range_of_y_over_x_l723_72304

def C (x y : ℝ) : Prop := x^2 + y^2 + 4*x + 3 = 0

theorem range_of_y_over_x : 
  ∀ x y : ℝ, C x y → ∃ t : ℝ, y / x = t ∧ -Real.sqrt 3 / 3 ≤ t ∧ t ≤ Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_y_over_x_l723_72304


namespace NUMINAMATH_CALUDE_second_number_possibilities_l723_72372

def is_valid_pair (x y : ℤ) : Prop :=
  (x = 14 ∨ y = 14) ∧ 2*x + 3*y = 94

theorem second_number_possibilities :
  ∃ (a b : ℤ), a ≠ b ∧ 
  (∀ x y, is_valid_pair x y → (x = 14 ∧ y = a) ∨ (y = 14 ∧ x = b)) :=
sorry

end NUMINAMATH_CALUDE_second_number_possibilities_l723_72372


namespace NUMINAMATH_CALUDE_lateral_surface_area_of_truncated_pyramid_l723_72399

/-- A regular truncated quadrilateral pyramid with an inscribed sphere -/
structure TruncatedPyramid where
  a : ℝ  -- height of the lateral face
  inscribed_sphere : Prop  -- property that a sphere can be inscribed

/-- The lateral surface area of a truncated quadrilateral pyramid -/
def lateral_surface_area (tp : TruncatedPyramid) : ℝ :=
  4 * tp.a^2

/-- Theorem: The lateral surface area of a regular truncated quadrilateral pyramid
    with an inscribed sphere is 4a^2, where a is the height of the lateral face -/
theorem lateral_surface_area_of_truncated_pyramid (tp : TruncatedPyramid) :
  tp.inscribed_sphere → lateral_surface_area tp = 4 * tp.a^2 := by
  sorry

end NUMINAMATH_CALUDE_lateral_surface_area_of_truncated_pyramid_l723_72399


namespace NUMINAMATH_CALUDE_ascending_order_fractions_l723_72358

theorem ascending_order_fractions (x y : ℝ) (hx : x < 0) (hy : 0 < y ∧ y < 1) :
  (1 / x) < (y / x) ∧ (y / x) < (y^2 / x) := by
  sorry

end NUMINAMATH_CALUDE_ascending_order_fractions_l723_72358


namespace NUMINAMATH_CALUDE_mack_journal_pages_l723_72315

/-- Calculates the number of pages written given time and rate -/
def pages_written (time minutes_per_page : ℕ) : ℕ :=
  time / minutes_per_page

/-- Represents Mack's journal writing over four days -/
structure JournalWriting where
  monday_time : ℕ
  monday_rate : ℕ
  tuesday_time : ℕ
  tuesday_rate : ℕ
  wednesday_pages : ℕ
  thursday_time1 : ℕ
  thursday_rate1 : ℕ
  thursday_time2 : ℕ
  thursday_rate2 : ℕ

/-- Calculates the total pages written over four days -/
def total_pages (j : JournalWriting) : ℕ :=
  pages_written j.monday_time j.monday_rate +
  pages_written j.tuesday_time j.tuesday_rate +
  j.wednesday_pages +
  pages_written j.thursday_time1 j.thursday_rate1 +
  pages_written j.thursday_time2 j.thursday_rate2

/-- Theorem stating the total pages written is 16 -/
theorem mack_journal_pages :
  ∀ j : JournalWriting,
    j.monday_time = 60 ∧
    j.monday_rate = 30 ∧
    j.tuesday_time = 45 ∧
    j.tuesday_rate = 15 ∧
    j.wednesday_pages = 5 ∧
    j.thursday_time1 = 30 ∧
    j.thursday_rate1 = 10 ∧
    j.thursday_time2 = 60 ∧
    j.thursday_rate2 = 20 →
    total_pages j = 16 := by
  sorry

end NUMINAMATH_CALUDE_mack_journal_pages_l723_72315


namespace NUMINAMATH_CALUDE_largest_percent_error_circle_area_l723_72316

/-- The largest possible percent error in the computed area of a circle, given a measurement error in its diameter --/
theorem largest_percent_error_circle_area (actual_diameter : ℝ) (max_error_percent : ℝ) :
  actual_diameter = 20 →
  max_error_percent = 20 →
  let max_measured_diameter := actual_diameter * (1 + max_error_percent / 100)
  let actual_area := Real.pi * (actual_diameter / 2) ^ 2
  let max_computed_area := Real.pi * (max_measured_diameter / 2) ^ 2
  let max_percent_error := (max_computed_area - actual_area) / actual_area * 100
  max_percent_error = 44 := by sorry

end NUMINAMATH_CALUDE_largest_percent_error_circle_area_l723_72316


namespace NUMINAMATH_CALUDE_absolute_value_quadratic_inequality_l723_72345

theorem absolute_value_quadratic_inequality (x : ℝ) :
  |3 * x^2 - 5 * x - 2| < 5 ↔ x > -1/3 ∧ x < 1/3 :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_quadratic_inequality_l723_72345


namespace NUMINAMATH_CALUDE_unique_solution_cubic_l723_72391

theorem unique_solution_cubic (c : ℝ) : c = 3/4 ↔ 
  ∃! (b : ℝ), b > 0 ∧ 
    ∃! (x : ℝ), x^3 + x^2 + (b^2 + 1/b^2) * x + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_cubic_l723_72391


namespace NUMINAMATH_CALUDE_sin_cos_product_positive_implies_quadrant_I_or_III_l723_72342

def is_in_quadrant_I_or_III (θ : Real) : Prop :=
  (0 < θ ∧ θ < Real.pi / 2) ∨ (Real.pi < θ ∧ θ < 3 * Real.pi / 2)

theorem sin_cos_product_positive_implies_quadrant_I_or_III (θ : Real) :
  Real.sin θ * Real.cos θ > 0 → is_in_quadrant_I_or_III θ :=
by
  sorry

end NUMINAMATH_CALUDE_sin_cos_product_positive_implies_quadrant_I_or_III_l723_72342


namespace NUMINAMATH_CALUDE_janes_homework_l723_72367

theorem janes_homework (x y z : ℝ) 
  (h1 : x - (y + z) = 15) 
  (h2 : x - y + z = 7) : 
  x - y = 11 := by
sorry

end NUMINAMATH_CALUDE_janes_homework_l723_72367


namespace NUMINAMATH_CALUDE_inequality_proof_l723_72322

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  (a + 1/b)^2 + (b + 1/c)^2 + (c + 1/a)^2 ≥ 3 * (a + b + c + 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l723_72322


namespace NUMINAMATH_CALUDE_point_p_properties_l723_72389

-- Define point P
def P (a : ℝ) : ℝ × ℝ := (2*a - 2, a + 5)

-- Define point Q
def Q : ℝ × ℝ := (4, 5)

theorem point_p_properties (a : ℝ) :
  -- Part 1
  (P a).2 = 0 → P a = (-12, 0) ∧
  -- Part 2
  (P a).1 = Q.1 → P a = (4, 8) ∧
  -- Part 3
  (P a).1 < 0 ∧ (P a).2 > 0 ∧ |(P a).1| = |(P a).2| → a^2022 + 2022 = 2023 :=
by sorry

end NUMINAMATH_CALUDE_point_p_properties_l723_72389


namespace NUMINAMATH_CALUDE_two_valid_solutions_l723_72323

def original_number : ℕ := 20192020

def is_valid (a b : ℕ) : Prop :=
  a < 10 ∧ b < 10 ∧ (a * 1000000000 + original_number * 10 + b) % 72 = 0

theorem two_valid_solutions :
  ∃! (s : Set (ℕ × ℕ)), s = {(2, 0), (3, 8)} ∧ 
    ∀ (a b : ℕ), (a, b) ∈ s ↔ is_valid a b :=
sorry

end NUMINAMATH_CALUDE_two_valid_solutions_l723_72323


namespace NUMINAMATH_CALUDE_circle_center_radius_sum_l723_72338

theorem circle_center_radius_sum (x y : ℝ) : 
  x^2 - 16*x + y^2 - 18*y = -81 → 
  ∃ (a b r : ℝ), (x - a)^2 + (y - b)^2 = r^2 ∧ a + b + r = 17 + Real.sqrt 145 := by
sorry

end NUMINAMATH_CALUDE_circle_center_radius_sum_l723_72338


namespace NUMINAMATH_CALUDE_correct_oranges_count_l723_72373

/-- Calculates the number of oranges needed to reach a desired total fruit count -/
def oranges_needed (total_desired : ℕ) (apples : ℕ) (bananas : ℕ) : ℕ :=
  total_desired - (apples + bananas)

theorem correct_oranges_count : oranges_needed 12 3 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_correct_oranges_count_l723_72373


namespace NUMINAMATH_CALUDE_product_of_special_numbers_l723_72363

theorem product_of_special_numbers (a b : ℝ) 
  (ha : a = Real.exp (2 - a)) 
  (hb : 1 + Real.log b = Real.exp (2 - (1 + Real.log b))) : 
  a * b = Real.exp 1 := by
sorry

end NUMINAMATH_CALUDE_product_of_special_numbers_l723_72363


namespace NUMINAMATH_CALUDE_distance_to_big_rock_l723_72374

/-- The distance to Big Rock given the rower's speed, river's speed, and round trip time -/
theorem distance_to_big_rock 
  (rower_speed : ℝ) 
  (river_speed : ℝ) 
  (round_trip_time : ℝ) 
  (h1 : rower_speed = 6)
  (h2 : river_speed = 2)
  (h3 : round_trip_time = 1) :
  (rower_speed + river_speed) * (rower_speed - river_speed) * round_trip_time / 
  (rower_speed + river_speed + rower_speed - river_speed) = 8/3 := by
sorry

end NUMINAMATH_CALUDE_distance_to_big_rock_l723_72374


namespace NUMINAMATH_CALUDE_speeding_ticket_problem_l723_72355

theorem speeding_ticket_problem (total_motorists : ℕ) 
  (h1 : total_motorists > 0) 
  (exceed_limit : ℕ) 
  (receive_tickets : ℕ) 
  (h2 : exceed_limit = total_motorists * 25 / 100) 
  (h3 : receive_tickets = total_motorists * 20 / 100) :
  (exceed_limit - receive_tickets) * 100 / exceed_limit = 20 := by
sorry

end NUMINAMATH_CALUDE_speeding_ticket_problem_l723_72355


namespace NUMINAMATH_CALUDE_square_ending_same_nonzero_digits_l723_72392

theorem square_ending_same_nonzero_digits (n : ℕ) :
  (∃ d : ℕ, d ≠ 0 ∧ d < 10 ∧ n^2 % 100 = d * 10 + d) →
  n^2 % 100 = 44 := by
sorry

end NUMINAMATH_CALUDE_square_ending_same_nonzero_digits_l723_72392


namespace NUMINAMATH_CALUDE_inequality_proof_l723_72378

theorem inequality_proof (a b c d e : ℝ) 
  (h1 : a ≤ b) (h2 : b ≤ c) (h3 : c ≤ d) (h4 : d ≤ e)
  (h5 : a + b + c + d + e = 1) :
  a * d + d * c + c * b + b * e + e * a ≤ 1/5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l723_72378


namespace NUMINAMATH_CALUDE_polygon_sides_difference_l723_72368

/-- Number of diagonals in a polygon with n sides -/
def diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- The problem statement -/
theorem polygon_sides_difference : ∃ (m : ℕ),
  m > 3 ∧
  diagonals m = 3 * diagonals (m - 3) ∧
  ∃ (x : ℕ),
    diagonals (m + x) = 7 * diagonals m ∧
    x = 12 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_difference_l723_72368


namespace NUMINAMATH_CALUDE_cube_side_length_l723_72330

theorem cube_side_length (volume : ℝ) (side : ℝ) : 
  volume = 729 → side^3 = volume → side = 9 := by
  sorry

end NUMINAMATH_CALUDE_cube_side_length_l723_72330


namespace NUMINAMATH_CALUDE_max_blocks_in_box_l723_72326

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents the dimensions of a block -/
structure BlockDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the maximum number of blocks that can fit in a box -/
def maxBlocksFit (box : BoxDimensions) (block : BlockDimensions) : ℕ :=
  (box.length / block.length) * (box.width / block.width) * (box.height / block.height)

theorem max_blocks_in_box :
  let box : BoxDimensions := ⟨5, 4, 3⟩
  let block : BlockDimensions := ⟨1, 2, 2⟩
  maxBlocksFit box block = 12 := by
  sorry


end NUMINAMATH_CALUDE_max_blocks_in_box_l723_72326


namespace NUMINAMATH_CALUDE_fifty_second_card_is_ace_l723_72381

-- Define the card ranks
inductive Rank
| King | Queen | Jack | Ten | Nine | Eight | Seven | Six | Five | Four | Three | Two | Ace

-- Define the reversed order of cards
def reversedOrder : List Rank := [
  Rank.King, Rank.Queen, Rank.Jack, Rank.Ten, Rank.Nine, Rank.Eight, Rank.Seven,
  Rank.Six, Rank.Five, Rank.Four, Rank.Three, Rank.Two, Rank.Ace
]

-- Define the number of cards in a cycle
def cardsPerCycle : Nat := 13

-- Define the position we're interested in
def targetPosition : Nat := 52

-- Theorem: The 52nd card in the reversed deck is an Ace
theorem fifty_second_card_is_ace :
  (targetPosition - 1) % cardsPerCycle = cardsPerCycle - 1 →
  reversedOrder[(targetPosition - 1) % cardsPerCycle] = Rank.Ace :=
by
  sorry

#check fifty_second_card_is_ace

end NUMINAMATH_CALUDE_fifty_second_card_is_ace_l723_72381


namespace NUMINAMATH_CALUDE_gcd_of_three_numbers_l723_72339

theorem gcd_of_three_numbers : Nat.gcd 4557 (Nat.gcd 1953 5115) = 93 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_three_numbers_l723_72339


namespace NUMINAMATH_CALUDE_smallest_k_for_fifteen_digit_period_l723_72393

/-- Represents a positive rational number with a decimal representation having a minimal period of 30 digits -/
def RationalWith30DigitPeriod : Type := { q : ℚ // q > 0 ∧ ∃ m : ℕ+, q = m / (10^30 - 1) }

/-- Given two positive rational numbers with 30-digit periods, returns true if their difference has a 15-digit period -/
def hasFifteenDigitPeriodDiff (a b : RationalWith30DigitPeriod) : Prop :=
  ∃ p : ℤ, (a.val - b.val : ℚ) = p / (10^15 - 1)

/-- Given two positive rational numbers with 30-digit periods and a natural number k,
    returns true if their sum with k times the second number has a 15-digit period -/
def hasFifteenDigitPeriodSum (a b : RationalWith30DigitPeriod) (k : ℕ) : Prop :=
  ∃ q : ℤ, (a.val + k * b.val : ℚ) = q / (10^15 - 1)

theorem smallest_k_for_fifteen_digit_period (a b : RationalWith30DigitPeriod)
  (h : hasFifteenDigitPeriodDiff a b) :
  (∀ k < 6, ¬hasFifteenDigitPeriodSum a b k) ∧ hasFifteenDigitPeriodSum a b 6 :=
sorry

end NUMINAMATH_CALUDE_smallest_k_for_fifteen_digit_period_l723_72393


namespace NUMINAMATH_CALUDE_problem_1_l723_72390

theorem problem_1 (m : ℝ) (h : -m^2 = m) : m^2 + m + 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l723_72390


namespace NUMINAMATH_CALUDE_secretary_typing_arrangements_l723_72352

def remaining_letters : Finset Nat := {1, 2, 3, 4, 6, 7, 8, 10}

def possible_arrangements (s : Finset Nat) : Nat :=
  Finset.card s + 2

theorem secretary_typing_arrangements :
  (Finset.powerset remaining_letters).sum (fun s => Nat.choose 8 (Finset.card s) * possible_arrangements s) = 1400 := by
  sorry

end NUMINAMATH_CALUDE_secretary_typing_arrangements_l723_72352


namespace NUMINAMATH_CALUDE_final_price_after_discounts_l723_72386

def original_price : ℝ := 200
def weekend_discount : ℝ := 0.4
def wednesday_discount : ℝ := 0.2

theorem final_price_after_discounts :
  (original_price * (1 - weekend_discount)) * (1 - wednesday_discount) = 96 := by
  sorry

end NUMINAMATH_CALUDE_final_price_after_discounts_l723_72386


namespace NUMINAMATH_CALUDE_log_sum_difference_equals_two_l723_72303

theorem log_sum_difference_equals_two :
  Real.log 50 / Real.log 10 + Real.log 20 / Real.log 10 - Real.log 10 / Real.log 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_difference_equals_two_l723_72303


namespace NUMINAMATH_CALUDE_min_value_sum_l723_72319

theorem min_value_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 2/b = 1) :
  a + 2*b ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_l723_72319


namespace NUMINAMATH_CALUDE_meeting_point_distance_from_top_l723_72306

-- Define the race parameters
def race_length : ℝ := 12
def uphill_length : ℝ := 6
def downhill_length : ℝ := 6

-- Define Jack's parameters
def jack_start_time : ℝ := 0
def jack_uphill_speed : ℝ := 12
def jack_downhill_speed : ℝ := 18

-- Define Jill's parameters
def jill_start_time : ℝ := 0.25  -- 15 minutes = 0.25 hours
def jill_uphill_speed : ℝ := 14
def jill_downhill_speed : ℝ := 19

-- Define the theorem
theorem meeting_point_distance_from_top : 
  ∃ (meeting_time : ℝ) (meeting_distance : ℝ),
    meeting_time > jack_start_time + (uphill_length / jack_uphill_speed) ∧
    meeting_time > jill_start_time ∧
    meeting_time < jill_start_time + (uphill_length / jill_uphill_speed) ∧
    meeting_distance = uphill_length - (meeting_time - jill_start_time) * jill_uphill_speed ∧
    meeting_distance = downhill_length - (meeting_time - (jack_start_time + uphill_length / jack_uphill_speed)) * jack_downhill_speed ∧
    meeting_distance = 699 / 64 := by
  sorry

end NUMINAMATH_CALUDE_meeting_point_distance_from_top_l723_72306


namespace NUMINAMATH_CALUDE_num_lineups_eq_1782_l723_72384

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose 5 starters from a team of 15 players,
    including 4 quadruplets, with at most one quadruplet in the starting lineup -/
def num_lineups : ℕ :=
  let total_players : ℕ := 15
  let num_quadruplets : ℕ := 4
  let non_quadruplet_players : ℕ := total_players - num_quadruplets
  let starters : ℕ := 5
  (choose non_quadruplet_players starters) +
  (num_quadruplets * choose non_quadruplet_players (starters - 1))

theorem num_lineups_eq_1782 : num_lineups = 1782 := by
  sorry

end NUMINAMATH_CALUDE_num_lineups_eq_1782_l723_72384


namespace NUMINAMATH_CALUDE_range_of_a_for_subset_l723_72340

-- Define the sets A and B
def A (a : ℝ) := {x : ℝ | 3 + a ≤ x ∧ x ≤ 4 + 3*a}
def B := {x : ℝ | (x + 4) / (5 - x) ≥ 0 ∧ x ≠ 5}

-- State the theorem
theorem range_of_a_for_subset : 
  {a : ℝ | ∀ x, x ∈ A a → x ∈ B} = {a : ℝ | -1/2 ≤ a ∧ a < 1/3} := by sorry

end NUMINAMATH_CALUDE_range_of_a_for_subset_l723_72340


namespace NUMINAMATH_CALUDE_power_tower_mod_500_l723_72382

theorem power_tower_mod_500 : 
  5^(5^(5^5)) ≡ 125 [ZMOD 500] := by sorry

end NUMINAMATH_CALUDE_power_tower_mod_500_l723_72382


namespace NUMINAMATH_CALUDE_absolute_value_simplification_l723_72396

theorem absolute_value_simplification : |(-6 - 4)| = 6 + 4 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_simplification_l723_72396


namespace NUMINAMATH_CALUDE_expression_evaluation_l723_72317

theorem expression_evaluation :
  let a : ℤ := -2
  let expr := 3 * a^2 + (a^2 + (5 * a^2 - 2 * a) - 3 * (a^2 - 3 * a))
  expr = 10 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l723_72317


namespace NUMINAMATH_CALUDE_subset_implies_m_value_l723_72361

def A (m : ℝ) : Set ℝ := {1, 3, 2*m+3}
def B (m : ℝ) : Set ℝ := {3, m^2}

theorem subset_implies_m_value (m : ℝ) : B m ⊆ A m → m = 1 ∨ m = 3 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_m_value_l723_72361


namespace NUMINAMATH_CALUDE_parallel_lines_m_values_l723_72309

-- Define the lines l₁ and l₂
def l₁ (m x y : ℝ) : Prop := 2 * x + (m + 1) * y + 4 = 0
def l₂ (m x y : ℝ) : Prop := m * x + 3 * y - 2 = 0

-- Define what it means for two lines to be parallel
def parallel (f g : ℝ → ℝ → Prop) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ ∀ (x y : ℝ), f x y ↔ g (k * x) (k * y)

-- State the theorem
theorem parallel_lines_m_values :
  ∀ (m : ℝ), parallel (l₁ m) (l₂ m) ↔ (m = -3 ∨ m = 2) :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_m_values_l723_72309


namespace NUMINAMATH_CALUDE_y_intercept_of_line_l723_72359

/-- The y-intercept of the line x/a² - y/b² = 1 is -b², where a and b are non-zero real numbers. -/
theorem y_intercept_of_line (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ∃ y : ℝ, (0 : ℝ) / a^2 - y / b^2 = 1 ∧ y = -b^2 := by sorry

end NUMINAMATH_CALUDE_y_intercept_of_line_l723_72359


namespace NUMINAMATH_CALUDE_unique_number_l723_72369

def is_valid_increase (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = 10 * a + b ∧ a < 10 ∧ b < 10 ∧
  ∃ (c d : ℕ), (c = 2 ∨ c = 4) ∧ (d = 2 ∨ d = 4) ∧
  4 * n = 10 * (a + c) + (b + d)

theorem unique_number : ∀ n : ℕ, is_valid_increase n ↔ n = 14 := by sorry

end NUMINAMATH_CALUDE_unique_number_l723_72369


namespace NUMINAMATH_CALUDE_hyperbola_other_asymptote_l723_72398

/-- Given a hyperbola with one asymptote y = -2x + 5 and foci with x-coordinate 2,
    prove that the equation of the other asymptote is y = 2x - 3 -/
theorem hyperbola_other_asymptote (x y : ℝ) :
  let asymptote1 : ℝ → ℝ := λ x => -2 * x + 5
  let foci_x : ℝ := 2
  let center_x : ℝ := foci_x
  let center_y : ℝ := asymptote1 center_x
  let asymptote2 : ℝ → ℝ := λ x => 2 * x - 3
  (∀ x, y = asymptote1 x) → (y = asymptote2 x) := by
sorry

end NUMINAMATH_CALUDE_hyperbola_other_asymptote_l723_72398


namespace NUMINAMATH_CALUDE_oranges_per_box_l723_72321

theorem oranges_per_box (total_oranges : ℕ) (num_boxes : ℚ) 
  (h1 : total_oranges = 72) 
  (h2 : num_boxes = 3) : 
  (total_oranges : ℚ) / num_boxes = 24 := by
sorry

end NUMINAMATH_CALUDE_oranges_per_box_l723_72321


namespace NUMINAMATH_CALUDE_complex_vector_magnitude_l723_72300

/-- Given two complex numbers, prove that the magnitude of their difference is √29 -/
theorem complex_vector_magnitude (z1 z2 : ℂ) : 
  z1 = 1 - 2*I ∧ z2 = -1 + 3*I → Complex.abs (z2 - z1) = Real.sqrt 29 := by
  sorry

end NUMINAMATH_CALUDE_complex_vector_magnitude_l723_72300


namespace NUMINAMATH_CALUDE_two_thousand_fourteenth_smallest_perimeter_l723_72314

/-- A right triangle with integer side lengths forming an arithmetic sequence -/
structure ArithmeticRightTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  a_lt_b : a < b
  b_lt_c : b < c
  is_arithmetic : b - a = c - b
  is_right : a^2 + b^2 = c^2

/-- The perimeter of an arithmetic right triangle -/
def perimeter (t : ArithmeticRightTriangle) : ℕ := t.a + t.b + t.c

/-- The theorem stating that the 2014th smallest perimeter of arithmetic right triangles is 24168 -/
theorem two_thousand_fourteenth_smallest_perimeter :
  (ArithmeticRightTriangle.mk 6042 8056 10070 (by sorry) (by sorry) (by sorry) (by sorry) |>
    perimeter) = 24168 := by sorry

end NUMINAMATH_CALUDE_two_thousand_fourteenth_smallest_perimeter_l723_72314


namespace NUMINAMATH_CALUDE_square_area_not_correlation_l723_72337

/-- A relationship between two variables -/
structure Relationship (α β : Type) where
  relate : α → β → Prop

/-- A correlation is a relationship that is not deterministic -/
def IsCorrelation {α β : Type} (r : Relationship α β) : Prop :=
  ∃ (x : α) (y₁ y₂ : β), y₁ ≠ y₂ ∧ r.relate x y₁ ∧ r.relate x y₂

/-- The relationship between a square's side length and its area -/
def SquareAreaRelationship : Relationship ℝ ℝ :=
  { relate := λ side area => area = side ^ 2 }

/-- Theorem: The relationship between a square's side length and its area is not a correlation -/
theorem square_area_not_correlation : ¬ IsCorrelation SquareAreaRelationship := by
  sorry

end NUMINAMATH_CALUDE_square_area_not_correlation_l723_72337


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l723_72347

theorem diophantine_equation_solutions :
  ∃ (S : Finset (ℕ × ℕ)), 
    (∀ (p : ℕ × ℕ), p ∈ S → p.1 > 0 ∧ p.2 > 0 ∧ p.1 ≤ 1980 ∧ 4 * p.1^3 - 3 * p.1 + 1 = 2 * p.2^2) ∧
    S.card ≥ 31 :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l723_72347


namespace NUMINAMATH_CALUDE_triangle_area_l723_72356

theorem triangle_area (a c : ℝ) (B : ℝ) (h1 : a = 3 * Real.sqrt 3) (h2 : c = 2) (h3 : B = π / 3) :
  (1 / 2) * a * c * Real.sin B = 9 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l723_72356


namespace NUMINAMATH_CALUDE_function_composition_equality_l723_72301

theorem function_composition_equality (b : ℝ) (h1 : b > 0) : 
  let g : ℝ → ℝ := λ x ↦ b * x^2 - Real.cos (π * x)
  g (g 1) = -Real.cos π → b = 1 := by
sorry

end NUMINAMATH_CALUDE_function_composition_equality_l723_72301


namespace NUMINAMATH_CALUDE_work_completion_time_l723_72354

/-- The number of days it takes for A and B together to complete the work -/
def total_days : ℕ := 24

/-- The speed ratio of A to B -/
def speed_ratio : ℕ := 3

/-- The number of days it takes for A alone to complete the work -/
def days_for_A : ℕ := 32

theorem work_completion_time :
  speed_ratio * total_days = (speed_ratio + 1) * days_for_A :=
sorry

end NUMINAMATH_CALUDE_work_completion_time_l723_72354


namespace NUMINAMATH_CALUDE_system_solution_l723_72387

theorem system_solution : 
  ∀ x y : ℝ, x > 0 → y > 0 →
  (3*y - Real.sqrt (y/x) - 6*Real.sqrt (x*y) + 2 = 0 ∧ 
   x^2 + 81*x^2*y^4 = 2*y^2) →
  ((x = 1/3 ∧ y = 1/3) ∨ 
   (x = Real.sqrt (Real.sqrt 31) / 12 ∧ y = Real.sqrt (Real.sqrt 31) / 3)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l723_72387


namespace NUMINAMATH_CALUDE_residue_mod_17_l723_72349

theorem residue_mod_17 : (243 * 15 - 22 * 8 + 5) % 17 = 5 := by
  sorry

end NUMINAMATH_CALUDE_residue_mod_17_l723_72349


namespace NUMINAMATH_CALUDE_total_cost_calculation_l723_72343

def coffee_maker_price : ℝ := 70
def blender_price : ℝ := 100
def coffee_maker_discount : ℝ := 0.20
def blender_discount : ℝ := 0.15
def sales_tax_rate : ℝ := 0.08
def extended_warranty_cost : ℝ := 25
def shipping_fee : ℝ := 12

def total_cost : ℝ :=
  let discounted_coffee_maker := coffee_maker_price * (1 - coffee_maker_discount)
  let discounted_blender := blender_price * (1 - blender_discount)
  let subtotal := 2 * discounted_coffee_maker + discounted_blender
  let sales_tax := subtotal * sales_tax_rate
  subtotal + sales_tax + extended_warranty_cost + shipping_fee

theorem total_cost_calculation :
  total_cost = 249.76 := by sorry

end NUMINAMATH_CALUDE_total_cost_calculation_l723_72343


namespace NUMINAMATH_CALUDE_extreme_value_a_1_monotonicity_a_leq_neg_1_monotonicity_a_gt_neg_1_l723_72366

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (x^2 + 1 + a) / x - a * Real.log x

-- Theorem for the extreme value when a = 1
theorem extreme_value_a_1 :
  ∃ (x_min : ℝ), x_min > 0 ∧ 
  (∀ x > 0, f 1 x_min ≤ f 1 x) ∧
  f 1 x_min = Real.sqrt 2 + 3/2 - (1/2) * Real.log 2 :=
sorry

-- Theorem for monotonicity when a ≤ -1
theorem monotonicity_a_leq_neg_1 (a : ℝ) (h : a ≤ -1) :
  ∀ x y, 0 < x → 0 < y → x < y → f a x < f a y :=
sorry

-- Theorem for monotonicity when a > -1
theorem monotonicity_a_gt_neg_1 (a : ℝ) (h : a > -1) :
  (∀ x y, 0 < x → x < y → y < 1 + a → f a x > f a y) ∧
  (∀ x y, 1 + a < x → x < y → f a x < f a y) :=
sorry

end NUMINAMATH_CALUDE_extreme_value_a_1_monotonicity_a_leq_neg_1_monotonicity_a_gt_neg_1_l723_72366


namespace NUMINAMATH_CALUDE_max_knights_is_eight_l723_72324

/-- Represents a person who can be either a knight or a liar -/
inductive Person
| knight
| liar

/-- The type of statements a person can make about their number -/
inductive Statement
| greater_than (n : ℕ)
| less_than (n : ℕ)

/-- A function that determines if a statement is true for a given number -/
def is_true_statement (s : Statement) (num : ℕ) : Prop :=
  match s with
  | Statement.greater_than n => num > n
  | Statement.less_than n => num < n

/-- A function that determines if a person's statements are consistent with their type -/
def consistent_statements (p : Person) (num : ℕ) (s1 s2 : Statement) : Prop :=
  match p with
  | Person.knight => is_true_statement s1 num ∧ is_true_statement s2 num
  | Person.liar => ¬(is_true_statement s1 num) ∧ ¬(is_true_statement s2 num)

theorem max_knights_is_eight :
  ∃ (people : Fin 10 → Person) (numbers : Fin 10 → ℕ) 
    (statements1 statements2 : Fin 10 → Statement),
    (∀ i : Fin 10, ∃ n : ℕ, statements1 i = Statement.greater_than n ∧ n = i.val + 1) ∧
    (∀ i : Fin 10, ∃ n : ℕ, statements2 i = Statement.less_than n ∧ n ≤ 10) ∧
    (∀ i : Fin 10, consistent_statements (people i) (numbers i) (statements1 i) (statements2 i)) ∧
    (∀ n : ℕ, n > 8 → ¬∃ (people : Fin n → Person) (numbers : Fin n → ℕ) 
      (statements1 statements2 : Fin n → Statement),
      (∀ i : Fin n, ∃ m : ℕ, statements1 i = Statement.greater_than m ∧ m = i.val + 1) ∧
      (∀ i : Fin n, ∃ m : ℕ, statements2 i = Statement.less_than m ∧ m ≤ n) ∧
      (∀ i : Fin n, consistent_statements (people i) (numbers i) (statements1 i) (statements2 i)) ∧
      (∀ i : Fin n, people i = Person.knight)) :=
by sorry

end NUMINAMATH_CALUDE_max_knights_is_eight_l723_72324


namespace NUMINAMATH_CALUDE_arcsin_sum_inequality_l723_72344

theorem arcsin_sum_inequality (x y : ℝ) : 
  Real.arcsin x + Real.arcsin y > π / 2 ↔ 
  x ∈ Set.Icc 0 1 ∧ y ∈ Set.Icc 0 1 ∧ x^2 + y^2 > 1 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_sum_inequality_l723_72344


namespace NUMINAMATH_CALUDE_quadratic_two_real_roots_l723_72327

theorem quadratic_two_real_roots (b c : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ ∀ z : ℝ, z^2 + b*z + c = 0 ↔ z = x ∨ z = y) ↔ b^2 - 4*c ≥ 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_two_real_roots_l723_72327


namespace NUMINAMATH_CALUDE_work_completion_time_l723_72397

/-- Given that two workers A and B together complete a work in a certain number of days,
    and one worker alone can complete the work in a different number of days,
    we can determine how long it takes for both workers together to complete the work. -/
theorem work_completion_time
  (days_together : ℝ)
  (days_a_alone : ℝ)
  (h1 : days_together > 0)
  (h2 : days_a_alone > 0)
  (h3 : days_together < days_a_alone)
  (h4 : (1 / days_together) = (1 / days_a_alone) + (1 / days_b_alone))
  (h5 : days_together = 6)
  (h6 : days_a_alone = 10) :
  days_together = 6 :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l723_72397


namespace NUMINAMATH_CALUDE_bargain_bin_books_l723_72302

/-- Calculate the number of books in a bargain bin after sales and additions. -/
def booksInBin (initial : ℕ) (sold : ℕ) (added : ℕ) : ℕ :=
  initial - sold + added

/-- Theorem stating that for the given values, the number of books in the bin is 11. -/
theorem bargain_bin_books : booksInBin 4 3 10 = 11 := by
  sorry

end NUMINAMATH_CALUDE_bargain_bin_books_l723_72302


namespace NUMINAMATH_CALUDE_probability_relates_to_uncertain_events_l723_72365

-- Define the basic types of events
inductive Event
  | Certain
  | Impossible
  | Random

-- Define a probability function
def probability (e : Event) : Real :=
  match e with
  | Event.Certain => 1
  | Event.Impossible => 0
  | Event.Random => sorry -- Assumes a value between 0 and 1

-- Define what it means for an event to be uncertain
def is_uncertain (e : Event) : Prop :=
  e = Event.Random

-- State the theorem
theorem probability_relates_to_uncertain_events :
  ∃ (e : Event), is_uncertain e ∧ 0 < probability e ∧ probability e < 1 :=
sorry

end NUMINAMATH_CALUDE_probability_relates_to_uncertain_events_l723_72365


namespace NUMINAMATH_CALUDE_driver_weekly_distance_l723_72360

/-- Represents the driving schedule for a city bus driver --/
structure DrivingSchedule where
  mwf_hours : ℝ
  mwf_speed : ℝ
  tue_hours : ℝ
  tue_speed : ℝ
  thu_hours : ℝ
  thu_speed : ℝ

/-- Calculates the total distance traveled by the driver in a week --/
def totalDistanceTraveled (schedule : DrivingSchedule) : ℝ :=
  3 * (schedule.mwf_hours * schedule.mwf_speed) +
  schedule.tue_hours * schedule.tue_speed +
  schedule.thu_hours * schedule.thu_speed

/-- Theorem stating that the driver travels 148 kilometers in a week --/
theorem driver_weekly_distance (schedule : DrivingSchedule)
  (h1 : schedule.mwf_hours = 3)
  (h2 : schedule.mwf_speed = 12)
  (h3 : schedule.tue_hours = 2.5)
  (h4 : schedule.tue_speed = 9)
  (h5 : schedule.thu_hours = 2.5)
  (h6 : schedule.thu_speed = 7) :
  totalDistanceTraveled schedule = 148 := by
  sorry

#eval totalDistanceTraveled {
  mwf_hours := 3,
  mwf_speed := 12,
  tue_hours := 2.5,
  tue_speed := 9,
  thu_hours := 2.5,
  thu_speed := 7
}

end NUMINAMATH_CALUDE_driver_weekly_distance_l723_72360


namespace NUMINAMATH_CALUDE_inscribed_cube_volume_l723_72385

/-- The volume of a cube inscribed in a sphere, which is itself inscribed in a larger cube. -/
theorem inscribed_cube_volume (outer_cube_edge : ℝ) (h : outer_cube_edge = 12) :
  let sphere_diameter := outer_cube_edge
  let sphere_radius := sphere_diameter / 2
  let inner_cube_diagonal := sphere_diameter
  let inner_cube_edge := inner_cube_diagonal / Real.sqrt 3
  let inner_cube_volume := inner_cube_edge ^ 3
  inner_cube_volume = 192 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_cube_volume_l723_72385


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l723_72307

/-- Given a principal amount and an interest rate, if the simple interest
    for 2 years is $400 and the compound interest for 2 years is $440,
    then the interest rate is 20%. -/
theorem interest_rate_calculation (P r : ℝ) :
  P * r * 2 = 400 →
  P * ((1 + r)^2 - 1) = 440 →
  r = 0.20 := by sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l723_72307


namespace NUMINAMATH_CALUDE_percentage_problem_l723_72383

theorem percentage_problem : 
  let product := 45 * 8
  let total := 900
  let percentage := (product / total) * 100
  percentage = 40 := by sorry

end NUMINAMATH_CALUDE_percentage_problem_l723_72383


namespace NUMINAMATH_CALUDE_average_words_in_crossword_puzzle_l723_72332

/-- The number of words needed to use up a pencil -/
def words_per_pencil : ℕ := 1050

/-- The number of days in two weeks -/
def days_in_two_weeks : ℕ := 14

/-- The number of puzzles completed in two weeks -/
def puzzles_in_two_weeks : ℕ := days_in_two_weeks

/-- The average number of words in each crossword puzzle -/
def average_words_per_puzzle : ℚ := words_per_pencil / puzzles_in_two_weeks

theorem average_words_in_crossword_puzzle :
  average_words_per_puzzle = 75 := by sorry

end NUMINAMATH_CALUDE_average_words_in_crossword_puzzle_l723_72332


namespace NUMINAMATH_CALUDE_speed_doubling_l723_72308

theorem speed_doubling (distance : ℝ) (original_time : ℝ) (new_time : ℝ) 
  (h1 : distance = 440)
  (h2 : original_time = 3)
  (h3 : new_time = original_time / 2)
  : (distance / new_time) = 2 * (distance / original_time) := by
  sorry

#check speed_doubling

end NUMINAMATH_CALUDE_speed_doubling_l723_72308


namespace NUMINAMATH_CALUDE_min_value_parallel_vectors_l723_72336

/-- Given vectors a and b, with m > 0, n > 0, and a parallel to b, 
    the minimum value of 1/m + 8/n is 9/2 -/
theorem min_value_parallel_vectors (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  let a : ℝ × ℝ := (m, 1)
  let b : ℝ × ℝ := (4 - n, 2)
  (∃ k : ℝ, a = k • b) →
  (∀ m' n' : ℝ, m' > 0 → n' > 0 → 1 / m' + 8 / n' ≥ 9 / 2) ∧
  (∃ m' n' : ℝ, m' > 0 ∧ n' > 0 ∧ 1 / m' + 8 / n' = 9 / 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_parallel_vectors_l723_72336


namespace NUMINAMATH_CALUDE_equal_sequences_l723_72357

theorem equal_sequences (n : ℕ) (a b : Fin n → ℕ) 
  (h_gcd : Nat.gcd n 6 = 1)
  (h_a_pos : ∀ i, a i > 0)
  (h_b_pos : ∀ i, b i > 0)
  (h_a_inc : ∀ i j, i < j → a i < a j)
  (h_b_inc : ∀ i j, i < j → b i < b j)
  (h_sum_eq : ∀ j k l, j < k → k < l → a j + a k + a l = b j + b k + b l) :
  ∀ i, a i = b i :=
sorry

end NUMINAMATH_CALUDE_equal_sequences_l723_72357


namespace NUMINAMATH_CALUDE_inequality_range_l723_72331

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, |x - 1| + |x + 2| > a^2 + a + 1) → 
  a > -2 ∧ a < 1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_range_l723_72331


namespace NUMINAMATH_CALUDE_tenth_stage_toothpicks_l723_72318

/-- The number of toothpicks in the nth stage of the sequence -/
def toothpicks (n : ℕ) : ℕ := 4 + 3 * (n - 1)

/-- The 10th stage of the sequence has 31 toothpicks -/
theorem tenth_stage_toothpicks : toothpicks 10 = 31 := by
  sorry

end NUMINAMATH_CALUDE_tenth_stage_toothpicks_l723_72318


namespace NUMINAMATH_CALUDE_transistor_count_2010_l723_72395

def initial_year : ℕ := 1985
def final_year : ℕ := 2010
def initial_transistors : ℕ := 500000
def doubling_period : ℕ := 2

def moores_law (t : ℕ) : ℕ := initial_transistors * 2^((t - initial_year) / doubling_period)

theorem transistor_count_2010 : moores_law final_year = 2048000000 := by
  sorry

end NUMINAMATH_CALUDE_transistor_count_2010_l723_72395


namespace NUMINAMATH_CALUDE_tangent_line_at_0_1_l723_72364

/-- A line that is tangent to the unit circle at (0, 1) has the equation y = 1 -/
theorem tangent_line_at_0_1 (l : Set (ℝ × ℝ)) :
  (∀ p ∈ l, p.1^2 + p.2^2 = 1 → p = (0, 1)) →  -- l is tangent to the circle
  (0, 1) ∈ l →                                 -- l passes through (0, 1)
  l = {p : ℝ × ℝ | p.2 = 1} :=                 -- l has the equation y = 1
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_0_1_l723_72364


namespace NUMINAMATH_CALUDE_complement_A_union_B_l723_72350

-- Define the sets A and B
def A : Set ℝ := {x | x < -1 ∨ (2 ≤ x ∧ x < 3)}
def B : Set ℝ := {x | -2 ≤ x ∧ x < 4}

-- State the theorem
theorem complement_A_union_B :
  (Set.univ \ A) ∪ B = {x : ℝ | x ≥ -2} := by sorry

end NUMINAMATH_CALUDE_complement_A_union_B_l723_72350


namespace NUMINAMATH_CALUDE_imaginary_sum_equals_4_minus_4i_l723_72313

theorem imaginary_sum_equals_4_minus_4i :
  let i : ℂ := Complex.I
  (i + 2 * i^2 + 3 * i^3 + 4 * i^4 + 5 * i^5 + 6 * i^6 + 7 * i^7 + 8 * i^8) = (4 : ℂ) - 4 * i :=
by sorry

end NUMINAMATH_CALUDE_imaginary_sum_equals_4_minus_4i_l723_72313


namespace NUMINAMATH_CALUDE_polynomial_remainder_l723_72335

theorem polynomial_remainder (x : ℝ) : 
  (8 * x^4 - 20 * x^3 + 28 * x^2 - 32 * x + 15) % (4 * x - 8) = 31 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l723_72335


namespace NUMINAMATH_CALUDE_real_imaginary_intersection_empty_l723_72380

-- Define the universal set C (complex numbers)
variable (C : Type)

-- Define R (real numbers) and I (pure imaginary numbers) as subsets of C
variable (R I : Set C)

-- Theorem statement
theorem real_imaginary_intersection_empty : R ∩ I = ∅ := by
  sorry

end NUMINAMATH_CALUDE_real_imaginary_intersection_empty_l723_72380


namespace NUMINAMATH_CALUDE_nine_digit_palindromes_l723_72379

/-- A function that returns the number of n-digit palindromic integers using only the digits 1, 2, and 3 -/
def count_palindromes (n : ℕ) : ℕ :=
  if n % 2 = 0 then 3^(n/2) else 3^((n+1)/2)

/-- The number of positive nine-digit palindromic integers using only the digits 1, 2, and 3 is 243 -/
theorem nine_digit_palindromes : count_palindromes 9 = 243 := by
  sorry

end NUMINAMATH_CALUDE_nine_digit_palindromes_l723_72379


namespace NUMINAMATH_CALUDE_pizza_pieces_l723_72312

theorem pizza_pieces (total_pizzas : ℕ) (total_cost : ℕ) (cost_per_piece : ℕ) : 
  total_pizzas = 4 → total_cost = 80 → cost_per_piece = 4 → 
  (total_cost / total_pizzas) / cost_per_piece = 5 := by
  sorry

end NUMINAMATH_CALUDE_pizza_pieces_l723_72312


namespace NUMINAMATH_CALUDE_circle_equation_l723_72375

/-- Given a circle with center (2, 1) and a line containing its common chord
    with the circle x^2 + y^2 - 3x = 0 passing through (5, -2),
    prove that the equation of the circle is (x-2)^2 + (y-1)^2 = 4 -/
theorem circle_equation (x y : ℝ) :
  let center := (2, 1)
  let known_circle := fun (x y : ℝ) => x^2 + y^2 - 3*x = 0
  let common_chord_point := (5, -2)
  let circle_eq := fun (x y : ℝ) => (x - 2)^2 + (y - 1)^2 = 4
  (∃ (line : ℝ → ℝ → Prop),
    (∀ x y, line x y ↔ known_circle x y ∨ circle_eq x y) ∧
    line common_chord_point.1 common_chord_point.2) →
  circle_eq x y :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l723_72375


namespace NUMINAMATH_CALUDE_same_color_plate_probability_l723_72376

/-- The probability of selecting 3 plates of the same color from a set of 6 red plates and 5 blue plates is 2/11. -/
theorem same_color_plate_probability :
  let total_plates : ℕ := 11
  let red_plates : ℕ := 6
  let blue_plates : ℕ := 5
  let selected_plates : ℕ := 3
  let total_combinations := Nat.choose total_plates selected_plates
  let red_combinations := Nat.choose red_plates selected_plates
  let blue_combinations := Nat.choose blue_plates selected_plates
  let same_color_combinations := red_combinations + blue_combinations
  (same_color_combinations : ℚ) / total_combinations = 2 / 11 := by
  sorry

end NUMINAMATH_CALUDE_same_color_plate_probability_l723_72376


namespace NUMINAMATH_CALUDE_f_of_tan_squared_l723_72325

noncomputable def f (x : ℝ) : ℝ := 1 / ((x / (x - 1)))

theorem f_of_tan_squared (t : ℝ) (h1 : 0 ≤ t) (h2 : t ≤ π/2) :
  f (Real.tan t ^ 2) = Real.tan t ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_f_of_tan_squared_l723_72325


namespace NUMINAMATH_CALUDE_cube_painting_probability_l723_72333

/-- The number of colors used to paint the cube -/
def num_colors : ℕ := 3

/-- The number of faces on a cube -/
def num_faces : ℕ := 6

/-- The probability of each color for a single face -/
def color_probability : ℚ := 1 / 3

/-- The total number of possible color arrangements for the cube -/
def total_arrangements : ℕ := num_colors ^ num_faces

/-- The number of favorable arrangements where the cube can be placed with four vertical faces of the same color -/
def favorable_arrangements : ℕ := 75

/-- The probability of painting the cube such that it can be placed with four vertical faces of the same color -/
def probability_four_same : ℚ := favorable_arrangements / total_arrangements

theorem cube_painting_probability :
  probability_four_same = 25 / 243 := by sorry

end NUMINAMATH_CALUDE_cube_painting_probability_l723_72333


namespace NUMINAMATH_CALUDE_condition_a_necessary_not_sufficient_l723_72348

-- Define Condition A
def condition_a (x y : ℝ) : Prop :=
  2 < x + y ∧ x + y < 4 ∧ 0 < x * y ∧ x * y < 3

-- Define Condition B
def condition_b (x y : ℝ) : Prop :=
  0 < x ∧ x < 1 ∧ 2 < y ∧ y < 3

-- Theorem stating that Condition A is necessary but not sufficient for Condition B
theorem condition_a_necessary_not_sufficient :
  (∀ x y : ℝ, condition_b x y → condition_a x y) ∧
  (∃ x y : ℝ, condition_a x y ∧ ¬condition_b x y) := by
  sorry

end NUMINAMATH_CALUDE_condition_a_necessary_not_sufficient_l723_72348


namespace NUMINAMATH_CALUDE_quadratic_inequality_l723_72362

theorem quadratic_inequality (x : ℝ) : x^2 - 3*x - 4 > 0 ↔ x < -1 ∨ x > 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l723_72362


namespace NUMINAMATH_CALUDE_jude_current_age_l723_72311

/-- Heath's age today -/
def heath_age_today : ℕ := 16

/-- The number of years in the future when the age comparison is made -/
def years_in_future : ℕ := 5

/-- Heath's age in the future -/
def heath_age_future : ℕ := heath_age_today + years_in_future

/-- Jude's age in the future -/
def jude_age_future : ℕ := heath_age_future / 3

/-- Jude's age today -/
def jude_age_today : ℕ := jude_age_future - years_in_future

theorem jude_current_age : jude_age_today = 2 := by
  sorry

end NUMINAMATH_CALUDE_jude_current_age_l723_72311


namespace NUMINAMATH_CALUDE_apple_savings_proof_l723_72328

/-- The price in dollars for a pack of apples at Store 1 -/
def store1_price : ℚ := 3

/-- The number of apples in a pack at Store 1 -/
def store1_apples : ℕ := 6

/-- The price in dollars for a pack of apples at Store 2 -/
def store2_price : ℚ := 4

/-- The number of apples in a pack at Store 2 -/
def store2_apples : ℕ := 10

/-- The savings in cents per apple when buying from Store 2 instead of Store 1 -/
def savings_per_apple : ℕ := 10

theorem apple_savings_proof :
  (store1_price / store1_apples - store2_price / store2_apples) * 100 = savings_per_apple := by
  sorry

end NUMINAMATH_CALUDE_apple_savings_proof_l723_72328


namespace NUMINAMATH_CALUDE_xavier_yvonne_not_zelda_prob_l723_72334

-- Define the probabilities of success for each person
def xavier_prob : ℚ := 1/4
def yvonne_prob : ℚ := 2/3
def zelda_prob : ℚ := 5/8

-- Define the probability of the desired outcome
def desired_outcome_prob : ℚ := xavier_prob * yvonne_prob * (1 - zelda_prob)

-- Theorem statement
theorem xavier_yvonne_not_zelda_prob : desired_outcome_prob = 1/16 := by
  sorry

end NUMINAMATH_CALUDE_xavier_yvonne_not_zelda_prob_l723_72334


namespace NUMINAMATH_CALUDE_farm_tree_count_l723_72370

/-- Represents the number of trees of each type that fell during the typhoon -/
structure FallenTrees where
  narra : ℕ
  mahogany : ℕ
  total : ℕ
  one_more_mahogany : mahogany = narra + 1
  sum_equals_total : narra + mahogany = total

/-- Calculates the final number of trees on the farm -/
def final_tree_count (initial_mahogany initial_narra total_fallen : ℕ) (fallen : FallenTrees) : ℕ :=
  let remaining := initial_mahogany + initial_narra - total_fallen
  let new_narra := 2 * fallen.narra
  let new_mahogany := 3 * fallen.mahogany
  remaining + new_narra + new_mahogany

/-- The theorem to be proved -/
theorem farm_tree_count :
  ∃ (fallen : FallenTrees),
    fallen.total = 5 ∧
    final_tree_count 50 30 5 fallen = 88 := by
  sorry

end NUMINAMATH_CALUDE_farm_tree_count_l723_72370


namespace NUMINAMATH_CALUDE_triangle_abc_degenerate_l723_72346

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A parabola defined by y^2 = 4x -/
def Parabola := {p : Point | p.y^2 = 4 * p.x}

/-- A horizontal line defined by y = 2 -/
def HorizontalLine := {p : Point | p.y = 2}

/-- Theorem: The intersection of the horizontal line y = 2 and the parabola y^2 = 4x 
    results in a point that coincides with A(1,2), making triangle ABC degenerate -/
theorem triangle_abc_degenerate (A : Point) (h1 : A.x = 1) (h2 : A.y = 2) :
  ∃ (B : Point), B ∈ Parabola ∧ B ∈ HorizontalLine ∧ B = A :=
sorry

end NUMINAMATH_CALUDE_triangle_abc_degenerate_l723_72346


namespace NUMINAMATH_CALUDE_train_length_l723_72305

theorem train_length (bridge_length : ℝ) (total_time : ℝ) (on_bridge_time : ℝ) :
  bridge_length = 600 →
  total_time = 30 →
  on_bridge_time = 20 →
  (bridge_length + (bridge_length * on_bridge_time / total_time)) / (total_time - on_bridge_time) = 120 :=
by sorry

end NUMINAMATH_CALUDE_train_length_l723_72305


namespace NUMINAMATH_CALUDE_intersection_A_B_when_a_neg_one_range_of_a_when_A_subset_B_l723_72320

/-- Definition of set A -/
def A (a : ℝ) : Set ℝ := {x : ℝ | 0 < 2*x + a ∧ 2*x + a ≤ 3}

/-- Definition of set B -/
def B : Set ℝ := {x : ℝ | -1/2 < x ∧ x < 2}

/-- Theorem for the intersection of A and B when a = -1 -/
theorem intersection_A_B_when_a_neg_one :
  A (-1) ∩ B = {x : ℝ | 1/2 < x ∧ x < 2} := by sorry

/-- Theorem for the range of a when A is a subset of B -/
theorem range_of_a_when_A_subset_B :
  ∀ a : ℝ, A a ⊆ B ↔ -1 < a ∧ a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_when_a_neg_one_range_of_a_when_A_subset_B_l723_72320


namespace NUMINAMATH_CALUDE_principal_calculation_l723_72371

/-- Proves that given the specified conditions, the principal is 6200 --/
theorem principal_calculation (rate : ℚ) (time : ℕ) (interest_difference : ℚ) :
  rate = 5 / 100 →
  time = 10 →
  interest_difference = 3100 →
  ∃ (principal : ℚ), principal * rate * time = principal - interest_difference ∧ principal = 6200 := by
  sorry

end NUMINAMATH_CALUDE_principal_calculation_l723_72371


namespace NUMINAMATH_CALUDE_max_adjusted_employees_range_of_a_l723_72351

/- Define the total number of employees -/
def total_employees : ℕ := 1000

/- Define the original average profit per employee (in yuan) -/
def original_profit : ℕ := 100000

/- Define the function for adjusted employees' average profit -/
def adjusted_profit (a x : ℝ) : ℝ := 10000 * (a - 0.008 * x)

/- Define the function for remaining employees' average profit -/
def remaining_profit (x : ℝ) : ℝ := original_profit * (1 + 0.004 * x)

/- Theorem for part I -/
theorem max_adjusted_employees :
  ∃ (max_x : ℕ), max_x = 750 ∧
  ∀ (x : ℕ), x > 0 → x ≤ max_x →
  (total_employees - x : ℝ) * remaining_profit x ≥ total_employees * original_profit ∧
  ¬∃ (y : ℕ), y > max_x ∧ y > 0 ∧
  (total_employees - y : ℝ) * remaining_profit y ≥ total_employees * original_profit :=
sorry

/- Theorem for part II -/
theorem range_of_a (x : ℝ) (hx : 0 < x ∧ x ≤ 750) :
  ∃ (lower upper : ℝ), lower = 0 ∧ upper = 7 ∧
  ∀ (a : ℝ), a > lower ∧ a ≤ upper →
  x * adjusted_profit a x ≤ (total_employees - x) * remaining_profit x ∧
  ¬∃ (b : ℝ), b > upper ∧
  x * adjusted_profit b x ≤ (total_employees - x) * remaining_profit x :=
sorry

end NUMINAMATH_CALUDE_max_adjusted_employees_range_of_a_l723_72351


namespace NUMINAMATH_CALUDE_right_triangle_pythagorean_representation_l723_72394

theorem right_triangle_pythagorean_representation
  (a b c : ℕ)
  (d : ℤ)
  (h_order : a < b ∧ b < c)
  (h_gcd : Nat.gcd (c - a) (c - b) = 1)
  (h_right_triangle : (a + d)^2 + (b + d)^2 = (c + d)^2) :
  ∃ l m : ℤ, (c : ℤ) + d = l^2 + m^2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_pythagorean_representation_l723_72394


namespace NUMINAMATH_CALUDE_hexalia_base_theorem_l723_72388

/-- Converts a number from base s to base 10 -/
def toBase10 (digits : List Nat) (s : Nat) : Nat :=
  digits.foldr (fun d acc => d + s * acc) 0

/-- The base s used in Hexalia -/
def s : Nat :=
  sorry

/-- The cost of the computer in base s -/
def cost : List Nat :=
  [5, 3, 0]

/-- The amount paid in base s -/
def paid : List Nat :=
  [1, 2, 0, 0]

/-- The change received in base s -/
def change : List Nat :=
  [4, 5, 5]

/-- Theorem stating that the base s satisfies the transaction equation -/
theorem hexalia_base_theorem :
  toBase10 cost s + toBase10 change s = toBase10 paid s ∧ s = 10 :=
sorry

end NUMINAMATH_CALUDE_hexalia_base_theorem_l723_72388


namespace NUMINAMATH_CALUDE_jimmy_stair_time_l723_72329

/-- The sum of an arithmetic sequence -/
def arithmeticSum (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

/-- Jimmy's stair climbing time -/
theorem jimmy_stair_time : arithmeticSum 20 7 7 = 287 := by
  sorry

end NUMINAMATH_CALUDE_jimmy_stair_time_l723_72329


namespace NUMINAMATH_CALUDE_prob_xi_equals_three_l723_72377

/-- A random variable following a binomial distribution B(6, 1/2) -/
def ξ : ℕ → ℝ := sorry

/-- The probability mass function for ξ -/
def P (k : ℕ) : ℝ := sorry

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- Theorem: The probability that ξ equals 3 is 5/16 -/
theorem prob_xi_equals_three : P 3 = 5 / 16 := by sorry

end NUMINAMATH_CALUDE_prob_xi_equals_three_l723_72377


namespace NUMINAMATH_CALUDE_smallest_n_satisfying_condition_l723_72353

def trailing_zeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

def satisfies_condition (n : ℕ) : Prop :=
  n > 6 ∧ trailing_zeros (3 * n) = 4 * trailing_zeros n

theorem smallest_n_satisfying_condition :
  ∃ (n : ℕ), satisfies_condition n ∧ ∀ m, satisfies_condition m → n ≤ m :=
sorry

end NUMINAMATH_CALUDE_smallest_n_satisfying_condition_l723_72353


namespace NUMINAMATH_CALUDE_fraction_enlargement_l723_72341

theorem fraction_enlargement (x y : ℝ) (h : x + y ≠ 0) :
  (2 * (3 * x) * (3 * y)) / ((3 * x) + (3 * y)) = 3 * ((2 * x * y) / (x + y)) :=
by sorry

end NUMINAMATH_CALUDE_fraction_enlargement_l723_72341


namespace NUMINAMATH_CALUDE_rice_trader_problem_l723_72310

/-- A rice trader problem -/
theorem rice_trader_problem (initial_stock restocked final_stock : ℕ) 
  (h1 : initial_stock = 55)
  (h2 : restocked = 132)
  (h3 : final_stock = 164) :
  ∃ (sold : ℕ), initial_stock - sold + restocked = final_stock ∧ sold = 23 := by
  sorry

end NUMINAMATH_CALUDE_rice_trader_problem_l723_72310
