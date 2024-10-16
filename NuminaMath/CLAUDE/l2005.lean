import Mathlib

namespace NUMINAMATH_CALUDE_cabbage_area_is_one_sq_foot_l2005_200575

/-- Represents the cabbage garden problem --/
structure CabbageGarden where
  area_this_year : ℕ
  area_last_year : ℕ
  cabbages_this_year : ℕ
  cabbages_last_year : ℕ

/-- The area per cabbage is 1 square foot --/
theorem cabbage_area_is_one_sq_foot (garden : CabbageGarden)
  (h1 : garden.area_this_year = garden.cabbages_this_year)
  (h2 : garden.area_last_year = garden.cabbages_last_year)
  (h3 : garden.cabbages_this_year = 4096)
  (h4 : garden.cabbages_last_year = 3969)
  (h5 : ∃ n : ℕ, garden.area_this_year = n * n)
  (h6 : ∃ m : ℕ, garden.area_last_year = m * m) :
  garden.area_this_year / garden.cabbages_this_year = 1 := by
  sorry

#check cabbage_area_is_one_sq_foot

end NUMINAMATH_CALUDE_cabbage_area_is_one_sq_foot_l2005_200575


namespace NUMINAMATH_CALUDE_unique_a_for_set_equality_l2005_200590

/-- Given sets A and B, prove that there is exactly one real number a that satisfies A ∪ B = A -/
theorem unique_a_for_set_equality :
  ∃! (a : ℝ), ({1, 3, a^2} ∪ {1, a+2} : Set ℝ) = {1, 3, a^2} ∧ a = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_a_for_set_equality_l2005_200590


namespace NUMINAMATH_CALUDE_trophy_cost_l2005_200510

def total_cost (a b : ℕ) : ℚ := (a * 1000 + 999 + b) / 10

theorem trophy_cost (a b : ℕ) (h1 : a < 10) (h2 : b < 10) 
  (h3 : (a * 1000 + 999 + b) % 8 = 0) 
  (h4 : (a + 9 + 9 + 9 + b) % 9 = 0) : 
  (total_cost a b) / 72 = 11.11 := by
  sorry

end NUMINAMATH_CALUDE_trophy_cost_l2005_200510


namespace NUMINAMATH_CALUDE_product_equality_l2005_200543

theorem product_equality : (6000 * 0) = (6 * 0) := by
  sorry

end NUMINAMATH_CALUDE_product_equality_l2005_200543


namespace NUMINAMATH_CALUDE_sqrt_sum_equality_l2005_200599

theorem sqrt_sum_equality : 
  Real.sqrt 2 + Real.sqrt (2 + 4) + Real.sqrt (2 + 4 + 6) + Real.sqrt (2 + 4 + 6 + 8) = 
  Real.sqrt 2 + Real.sqrt 6 + 2 * Real.sqrt 3 + 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equality_l2005_200599


namespace NUMINAMATH_CALUDE_expression_value_l2005_200548

theorem expression_value : 6 * (3/2 + 2/3) = 13 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2005_200548


namespace NUMINAMATH_CALUDE_total_owed_after_borrowing_l2005_200534

/-- The total amount owed when borrowing additional money -/
theorem total_owed_after_borrowing (initial_debt additional_borrowed : ℕ) :
  initial_debt = 20 →
  additional_borrowed = 8 →
  initial_debt + additional_borrowed = 28 := by
  sorry

end NUMINAMATH_CALUDE_total_owed_after_borrowing_l2005_200534


namespace NUMINAMATH_CALUDE_divisibility_problem_l2005_200544

theorem divisibility_problem (n a b c d : ℤ) 
  (hn : n > 0) 
  (h1 : n ∣ (a + b + c + d)) 
  (h2 : n ∣ (a^2 + b^2 + c^2 + d^2)) : 
  n ∣ (a^4 + b^4 + c^4 + d^4 + 4*a*b*c*d) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_problem_l2005_200544


namespace NUMINAMATH_CALUDE_ratio_of_trigonometric_equation_l2005_200516

theorem ratio_of_trigonometric_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : (a * Real.sin (π/5) + b * Real.cos (π/5)) / (a * Real.cos (π/5) - b * Real.sin (π/5)) = Real.tan (8*π/15)) :
  b / a = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_trigonometric_equation_l2005_200516


namespace NUMINAMATH_CALUDE_expansion_coefficient_sum_l2005_200560

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The coefficient of x^k in the expansion of (1-2x)^n -/
def coeff (n k : ℕ) : ℤ :=
  (-2)^k * binomial n k

theorem expansion_coefficient_sum (n : ℕ) 
  (h : coeff n 1 + coeff n 4 = 70) : 
  coeff n 5 = -32 := by sorry

end NUMINAMATH_CALUDE_expansion_coefficient_sum_l2005_200560


namespace NUMINAMATH_CALUDE_min_theta_value_l2005_200547

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x) + Real.cos (ω * x)

theorem min_theta_value (ω : ℝ) (h_ω_pos : ω > 0) :
  (∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f ω (x + p) + |f ω (x + p)| = f ω x + |f ω x| ∧
    ∀ (q : ℝ), q > 0 → (∀ (x : ℝ), f ω (x + q) + |f ω (x + q)| = f ω x + |f ω x|) → p ≤ q) →
  (∃ (θ : ℝ), θ > 0 ∧ ∀ (x : ℝ), f ω x ≥ f ω θ) →
  (∃ (θ_min : ℝ), θ_min > 0 ∧ 
    (∀ (x : ℝ), f ω x ≥ f ω θ_min) ∧
    (∀ (θ : ℝ), θ > 0 → (∀ (x : ℝ), f ω x ≥ f ω θ) → θ_min ≤ θ) ∧
    θ_min = 5 * Real.pi / 8) :=
sorry

end NUMINAMATH_CALUDE_min_theta_value_l2005_200547


namespace NUMINAMATH_CALUDE_sum_of_interior_angles_octagon_l2005_200529

theorem sum_of_interior_angles_octagon (a : ℝ) : a = 1080 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_interior_angles_octagon_l2005_200529


namespace NUMINAMATH_CALUDE_minimum_jumps_circle_l2005_200530

/-- Represents a jump on the circle of points -/
inductive Jump
| Two  : Jump  -- Jump of 2 points
| Three : Jump  -- Jump of 3 points

/-- Represents a sequence of jumps -/
def JumpSequence := List Jump

/-- Function to check if a sequence of jumps visits all points and returns to start -/
def validSequence (n : Nat) (seq : JumpSequence) : Prop :=
  -- Implementation details omitted
  sorry

theorem minimum_jumps_circle :
  ∀ (seq : JumpSequence),
    validSequence 2016 seq →
    seq.length ≥ 2017 :=
by sorry

end NUMINAMATH_CALUDE_minimum_jumps_circle_l2005_200530


namespace NUMINAMATH_CALUDE_solution_set_of_equation_l2005_200508

theorem solution_set_of_equation (x y : ℝ) : 
  (|x*y| + |x - y + 1| = 0) ↔ ((x = 0 ∧ y = 1) ∨ (x = -1 ∧ y = 0)) := by
  sorry

end NUMINAMATH_CALUDE_solution_set_of_equation_l2005_200508


namespace NUMINAMATH_CALUDE_rhombus_area_l2005_200574

/-- The area of a rhombus with side length 20 and one diagonal of length 16 is 64√21 -/
theorem rhombus_area (side : ℝ) (diagonal1 : ℝ) (diagonal2 : ℝ) : 
  side = 20 → diagonal1 = 16 → diagonal2 = 8 * Real.sqrt 21 →
  (1/2) * diagonal1 * diagonal2 = 64 * Real.sqrt 21 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_l2005_200574


namespace NUMINAMATH_CALUDE_three_digit_sum_product_l2005_200531

theorem three_digit_sum_product (x : ℕ) (h1 : 1 ≤ x) (h2 : x ≤ 9) :
  let y : ℕ := 9
  let z : ℕ := 9
  100 * x + 10 * y + z = x + y + z + x * y + y * z + z * x + x * y * z :=
by sorry

end NUMINAMATH_CALUDE_three_digit_sum_product_l2005_200531


namespace NUMINAMATH_CALUDE_digit_puzzle_solution_l2005_200513

theorem digit_puzzle_solution :
  ∃! (A B C D E F G H J : ℕ),
    (A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧ E < 10 ∧ F < 10 ∧ G < 10 ∧ H < 10 ∧ J < 10) ∧
    (A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧ A ≠ G ∧ A ≠ H ∧ A ≠ J ∧
     B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧ B ≠ G ∧ B ≠ H ∧ B ≠ J ∧
     C ≠ D ∧ C ≠ E ∧ C ≠ F ∧ C ≠ G ∧ C ≠ H ∧ C ≠ J ∧
     D ≠ E ∧ D ≠ F ∧ D ≠ G ∧ D ≠ H ∧ D ≠ J ∧
     E ≠ F ∧ E ≠ G ∧ E ≠ H ∧ E ≠ J ∧
     F ≠ G ∧ F ≠ H ∧ F ≠ J ∧
     G ≠ H ∧ G ≠ J ∧
     H ≠ J) ∧
    (100 * A + 10 * B + C + 100 * D + 10 * E + F + 10 * G + E = 100 * G + 10 * E + F) ∧
    (100 * G + 10 * E + F + 10 * D + E = 100 * H + 10 * F + J) ∧
    A = 2 ∧ B = 3 ∧ C = 0 ∧ D = 1 ∧ E = 7 ∧ F = 8 ∧ G = 4 ∧ H = 5 ∧ J = 6 :=
by sorry

end NUMINAMATH_CALUDE_digit_puzzle_solution_l2005_200513


namespace NUMINAMATH_CALUDE_safe_elixir_preparations_l2005_200518

/-- Represents the number of magical herbs available. -/
def num_herbs : ℕ := 4

/-- Represents the number of mystical crystals available. -/
def num_crystals : ℕ := 6

/-- Represents the number of forbidden herb-crystal combinations. -/
def num_forbidden : ℕ := 3

/-- Calculates the number of safe elixir preparations. -/
def safe_preparations : ℕ := num_herbs * num_crystals - num_forbidden

/-- Theorem stating that the number of safe elixir preparations is 21. -/
theorem safe_elixir_preparations :
  safe_preparations = 21 := by sorry

end NUMINAMATH_CALUDE_safe_elixir_preparations_l2005_200518


namespace NUMINAMATH_CALUDE_jack_and_jill_meeting_point_l2005_200562

/-- Represents the hill run by Jack and Jill -/
structure HillRun where
  length : ℝ
  jack_uphill_speed : ℝ
  jack_downhill_speed : ℝ
  jill_uphill_speed : ℝ
  jill_downhill_speed : ℝ
  jack_pause_time : ℝ
  jack_pause_location : ℝ

/-- Calculates the meeting point of Jack and Jill -/
def meeting_point (h : HillRun) : ℝ :=
  sorry

/-- The theorem to be proved -/
theorem jack_and_jill_meeting_point (h : HillRun) 
  (h_length : h.length = 6)
  (h_jack_up : h.jack_uphill_speed = 12)
  (h_jack_down : h.jack_downhill_speed = 18)
  (h_jill_up : h.jill_uphill_speed = 15)
  (h_jill_down : h.jill_downhill_speed = 21)
  (h_pause_time : h.jack_pause_time = 0.25)
  (h_pause_loc : h.jack_pause_location = 3) :
  meeting_point h = 63 / 22 := by
  sorry

end NUMINAMATH_CALUDE_jack_and_jill_meeting_point_l2005_200562


namespace NUMINAMATH_CALUDE_centers_on_line_l2005_200576

-- Define the family of circles
def circle_family (k : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*k*x + (4*k + 10)*y + 10*k + 20 = 0

-- Define the line equation
def center_line (x y : ℝ) : Prop :=
  2*x - y - 5 = 0

-- Theorem statement
theorem centers_on_line :
  ∀ k : ℝ, k ≠ -1 →
  ∃ x y : ℝ, circle_family k x y ∧ center_line x y :=
sorry

end NUMINAMATH_CALUDE_centers_on_line_l2005_200576


namespace NUMINAMATH_CALUDE_circular_arrangement_size_l2005_200577

/-- Represents a circular arrangement of students and a teacher. -/
structure CircularArrangement where
  total_positions : ℕ
  teacher_position : ℕ

/-- Defines the property of two positions being opposite in the circle. -/
def is_opposite (c : CircularArrangement) (pos1 pos2 : ℕ) : Prop :=
  (pos2 - pos1) % c.total_positions = c.total_positions / 2

/-- The main theorem stating the total number of positions in the arrangement. -/
theorem circular_arrangement_size :
  ∀ (c : CircularArrangement),
    (is_opposite c 6 16) →
    (c.teacher_position ≤ c.total_positions) →
    (c.total_positions = 23) :=
by sorry

end NUMINAMATH_CALUDE_circular_arrangement_size_l2005_200577


namespace NUMINAMATH_CALUDE_at_least_one_leq_neg_two_l2005_200503

theorem at_least_one_leq_neg_two (a b c : ℝ) 
  (ha : a < 0) (hb : b < 0) (hc : c < 0) : 
  (a + 1/b ≤ -2) ∨ (b + 1/c ≤ -2) ∨ (c + 1/a ≤ -2) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_leq_neg_two_l2005_200503


namespace NUMINAMATH_CALUDE_log_expression_arbitrarily_small_l2005_200522

theorem log_expression_arbitrarily_small :
  ∀ ε > 0, ∃ x > (2/3 : ℝ), Real.log (x^2 + 3) - 2 * Real.log x < ε :=
by sorry

end NUMINAMATH_CALUDE_log_expression_arbitrarily_small_l2005_200522


namespace NUMINAMATH_CALUDE_cheapest_candle_combination_l2005_200584

/-- Represents a candle with its burning time and cost -/
structure Candle where
  burn_time : ℕ
  cost : ℕ

/-- Finds the minimum cost to measure exactly one minute using given candles -/
def min_cost_to_measure_one_minute (candles : List Candle) : ℕ :=
  sorry

/-- The problem statement -/
theorem cheapest_candle_combination :
  let big_candle : Candle := { burn_time := 16, cost := 16 }
  let small_candle : Candle := { burn_time := 7, cost := 7 }
  let candles : List Candle := [big_candle, small_candle]
  min_cost_to_measure_one_minute candles = 97 :=
sorry

end NUMINAMATH_CALUDE_cheapest_candle_combination_l2005_200584


namespace NUMINAMATH_CALUDE_fifteenth_student_age_l2005_200586

theorem fifteenth_student_age
  (total_students : Nat)
  (average_age : ℕ)
  (group1_students : Nat)
  (group1_average : ℕ)
  (group2_students : Nat)
  (group2_average : ℕ)
  (h1 : total_students = 15)
  (h2 : average_age = 15)
  (h3 : group1_students = 6)
  (h4 : group1_average = 14)
  (h5 : group2_students = 8)
  (h6 : group2_average = 16)
  (h7 : group1_students + group2_students + 1 = total_students) :
  total_students * average_age - (group1_students * group1_average + group2_students * group2_average) = 13 :=
by sorry

end NUMINAMATH_CALUDE_fifteenth_student_age_l2005_200586


namespace NUMINAMATH_CALUDE_polar_to_cartesian_conversion_l2005_200553

theorem polar_to_cartesian_conversion :
  let r : ℝ := 2
  let θ : ℝ := 2 * Real.pi / 3
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  (x = -1) ∧ (y = Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_conversion_l2005_200553


namespace NUMINAMATH_CALUDE_even_function_property_l2005_200552

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Define the theorem
theorem even_function_property (f : ℝ → ℝ) 
  (h1 : EvenFunction f) 
  (h2 : ∀ x < 0, f x = x * (x + 1)) : 
  ∀ x > 0, f x = x * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_even_function_property_l2005_200552


namespace NUMINAMATH_CALUDE_craig_dave_bench_press_ratio_l2005_200528

/-- Proves that Craig's bench press is 20% of Dave's bench press -/
theorem craig_dave_bench_press_ratio :
  let dave_weight : ℝ := 175
  let dave_bench_press : ℝ := 3 * dave_weight
  let mark_bench_press : ℝ := 55
  let craig_bench_press : ℝ := mark_bench_press + 50
  (craig_bench_press / dave_bench_press) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_craig_dave_bench_press_ratio_l2005_200528


namespace NUMINAMATH_CALUDE_lcm_1640_1020_l2005_200541

theorem lcm_1640_1020 : Nat.lcm 1640 1020 = 83640 := by
  sorry

end NUMINAMATH_CALUDE_lcm_1640_1020_l2005_200541


namespace NUMINAMATH_CALUDE_dream_team_strategy_l2005_200592

/-- Represents the probabilities of correct answers for each team member and category -/
structure TeamProbabilities where
  a_category_a : ℝ
  a_category_b : ℝ
  b_category_a : ℝ
  b_category_b : ℝ

/-- Calculates the probability of entering the final round when answering a specific category first -/
def probability_enter_final (probs : TeamProbabilities) (start_with_a : Bool) : ℝ :=
  if start_with_a then
    let p3 := probs.a_category_a * probs.b_category_a * probs.a_category_b * (1 - probs.b_category_b) +
              probs.a_category_a * probs.b_category_a * (1 - probs.a_category_b) * probs.b_category_b
    let p4 := probs.a_category_a * probs.b_category_a * probs.a_category_b * probs.b_category_b
    p3 + p4
  else
    let p3 := probs.a_category_b * probs.b_category_b * probs.a_category_a * (1 - probs.b_category_a) +
              probs.a_category_b * probs.b_category_b * (1 - probs.a_category_a) * probs.b_category_a
    let p4 := probs.a_category_b * probs.b_category_b * probs.a_category_a * probs.b_category_a
    p3 + p4

/-- The main theorem to be proved -/
theorem dream_team_strategy (probs : TeamProbabilities)
  (h1 : probs.a_category_a = 0.7)
  (h2 : probs.a_category_b = 0.5)
  (h3 : probs.b_category_a = 0.4)
  (h4 : probs.b_category_b = 0.8) :
  probability_enter_final probs false > probability_enter_final probs true :=
by sorry

end NUMINAMATH_CALUDE_dream_team_strategy_l2005_200592


namespace NUMINAMATH_CALUDE_other_side_heads_probability_is_two_thirds_l2005_200556

/-- Represents the three types of coins -/
inductive Coin
  | Normal
  | TwoHeads
  | TwoTails

/-- Represents the possible outcomes of a coin toss -/
inductive CoinSide
  | Heads
  | Tails

/-- The probability of selecting each type of coin -/
def coinProbability (c : Coin) : ℚ :=
  match c with
  | Coin.Normal => 1/3
  | Coin.TwoHeads => 1/3
  | Coin.TwoTails => 1/3

/-- The probability of getting heads when tossing a specific coin -/
def headsUpProbability (c : Coin) : ℚ :=
  match c with
  | Coin.Normal => 1/2
  | Coin.TwoHeads => 1
  | Coin.TwoTails => 0

/-- The probability that the other side is heads given that heads is showing -/
def otherSideHeadsProbability : ℚ := by
  sorry

theorem other_side_heads_probability_is_two_thirds :
  otherSideHeadsProbability = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_other_side_heads_probability_is_two_thirds_l2005_200556


namespace NUMINAMATH_CALUDE_joses_swimming_pool_charge_l2005_200593

/-- Proves that the daily charge for kids in Jose's swimming pool is $3 -/
theorem joses_swimming_pool_charge (kid_charge : ℚ) (adult_charge : ℚ) 
  (h1 : adult_charge = 2 * kid_charge) 
  (h2 : 8 * kid_charge + 10 * adult_charge = 588 / 7) : 
  kid_charge = 3 := by
  sorry

end NUMINAMATH_CALUDE_joses_swimming_pool_charge_l2005_200593


namespace NUMINAMATH_CALUDE_abs_sum_minimum_l2005_200540

theorem abs_sum_minimum (x y : ℝ) : |x - 1| + |x| + |y - 1| + |y + 1| ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_minimum_l2005_200540


namespace NUMINAMATH_CALUDE_min_cuts_for_3inch_to_1inch_cube_l2005_200537

/-- Represents a three-dimensional cube -/
structure Cube where
  side_length : ℕ

/-- Represents a cut on a cube -/
inductive Cut
  | plane : Cut

/-- The minimum number of cuts required to divide a cube into smaller cubes -/
def min_cuts (original : Cube) (target : Cube) : ℕ := sorry

/-- The number of smaller cubes that can be created from a larger cube -/
def num_smaller_cubes (original : Cube) (target : Cube) : ℕ := 
  (original.side_length / target.side_length) ^ 3

theorem min_cuts_for_3inch_to_1inch_cube : 
  let original := Cube.mk 3
  let target := Cube.mk 1
  min_cuts original target = 6 ∧ 
  num_smaller_cubes original target = 27 := by sorry

end NUMINAMATH_CALUDE_min_cuts_for_3inch_to_1inch_cube_l2005_200537


namespace NUMINAMATH_CALUDE_max_log_sum_l2005_200524

theorem max_log_sum (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_eq : 2*x + y = 20) :
  ∃ (max_val : ℝ), max_val = 2 - Real.log 2 ∧ 
  ∀ (a b : ℝ), a > 0 → b > 0 → 2*a + b = 20 → Real.log a + Real.log b ≤ max_val :=
sorry

end NUMINAMATH_CALUDE_max_log_sum_l2005_200524


namespace NUMINAMATH_CALUDE_owen_profit_l2005_200589

/-- Calculates the profit from selling face masks given the following conditions:
  * Number of boxes bought
  * Cost per box
  * Number of masks per box
  * Number of boxes repacked
  * Number of large packets sold
  * Price of large packets
  * Number of masks in large packets
  * Price of small baggies
  * Number of masks in small baggies
-/
def calculate_profit (
  boxes_bought : ℕ
  ) (cost_per_box : ℚ
  ) (masks_per_box : ℕ
  ) (boxes_repacked : ℕ
  ) (large_packets_sold : ℕ
  ) (large_packet_price : ℚ
  ) (masks_per_large_packet : ℕ
  ) (small_baggie_price : ℚ
  ) (masks_per_small_baggie : ℕ
  ) : ℚ :=
  let total_cost := boxes_bought * cost_per_box
  let total_masks := boxes_bought * masks_per_box
  let repacked_masks := boxes_repacked * masks_per_box
  let large_packet_revenue := large_packets_sold * large_packet_price
  let remaining_masks := total_masks - (large_packets_sold * masks_per_large_packet)
  let small_baggies := remaining_masks / masks_per_small_baggie
  let small_baggie_revenue := small_baggies * small_baggie_price
  let total_revenue := large_packet_revenue + small_baggie_revenue
  total_revenue - total_cost

theorem owen_profit :
  calculate_profit 12 9 50 6 3 12 100 3 10 = 18 := by
  sorry

end NUMINAMATH_CALUDE_owen_profit_l2005_200589


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_55_l2005_200569

def is_divisible_by (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

theorem smallest_four_digit_divisible_by_55 :
  ∀ n : ℕ, is_four_digit n → is_divisible_by n 55 → n ≥ 1100 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_55_l2005_200569


namespace NUMINAMATH_CALUDE_perfect_square_between_prime_sums_l2005_200572

/-- Represents the nth prime number -/
def nthPrime (n : ℕ) : ℕ := sorry

/-- Represents the sum of the first n prime numbers -/
def sumFirstNPrimes (n : ℕ) : ℕ :=
  (List.range n).map nthPrime |>.sum

/-- Theorem: For any n, there exists a perfect square between the sum of the first n primes
    and the sum of the first n+1 primes -/
theorem perfect_square_between_prime_sums (n : ℕ) :
  ∃ m : ℕ, sumFirstNPrimes n ≤ m^2 ∧ m^2 ≤ sumFirstNPrimes (n+1) := by sorry

end NUMINAMATH_CALUDE_perfect_square_between_prime_sums_l2005_200572


namespace NUMINAMATH_CALUDE_sum_of_ab_l2005_200567

theorem sum_of_ab (a b : ℝ) (h1 : a * b = 5) (h2 : 1 / a^2 + 1 / b^2 = 0.6) : 
  a + b = 5 ∨ a + b = -5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ab_l2005_200567


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l2005_200511

theorem arithmetic_sequence_length : 
  ∀ (a₁ n d : ℕ) (aₙ : ℕ), 
    a₁ = 3 → 
    d = 3 → 
    aₙ = 144 → 
    aₙ = a₁ + (n - 1) * d → 
    n = 48 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l2005_200511


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l2005_200515

/-- Given a geometric sequence {aₙ} where the first three terms are x, 2x+2, and 3x+3 respectively,
    prove that the fourth term a₄ = -27/2. -/
theorem geometric_sequence_fourth_term (x : ℝ) (a : ℕ → ℝ) :
  a 1 = x ∧ a 2 = 2*x + 2 ∧ a 3 = 3*x + 3 ∧ 
  (∀ n : ℕ, n ≥ 1 → a (n + 1) / a n = a 2 / a 1) →
  a 4 = -27/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l2005_200515


namespace NUMINAMATH_CALUDE_parabola_vertex_l2005_200566

/-- The parabola is defined by the equation y = (x - 1)^2 - 2 -/
def parabola (x : ℝ) : ℝ := (x - 1)^2 - 2

/-- The vertex of a parabola is the point where it reaches its maximum or minimum -/
def is_vertex (x y : ℝ) : Prop :=
  ∀ t : ℝ, parabola t ≥ parabola x ∨ parabola t ≤ parabola x

theorem parabola_vertex :
  is_vertex 1 (-2) :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l2005_200566


namespace NUMINAMATH_CALUDE_students_taking_one_subject_l2005_200598

theorem students_taking_one_subject (both : ℕ) (algebra : ℕ) (geometry_only : ℕ)
  (h1 : both = 16)
  (h2 : algebra = 36)
  (h3 : geometry_only = 15) :
  algebra - both + geometry_only = 35 := by
  sorry

end NUMINAMATH_CALUDE_students_taking_one_subject_l2005_200598


namespace NUMINAMATH_CALUDE_arithmetic_progression_rth_term_l2005_200514

/-- Given an arithmetic progression where the sum of n terms is 2n + 3n^2 for every n,
    prove that the r-th term is 6r - 1. -/
theorem arithmetic_progression_rth_term (r : ℕ) :
  let S : ℕ → ℕ := λ n => 2*n + 3*n^2
  let a : ℕ → ℤ := λ k => S k - S (k-1)
  a r = 6*r - 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_rth_term_l2005_200514


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2005_200501

def A : Set ℝ := {-1, 0, 1}
def B : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 1}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2005_200501


namespace NUMINAMATH_CALUDE_root_product_sum_l2005_200519

theorem root_product_sum (p q r : ℂ) : 
  (6 * p^3 - 9 * p^2 + 16 * p - 12 = 0) →
  (6 * q^3 - 9 * q^2 + 16 * q - 12 = 0) →
  (6 * r^3 - 9 * r^2 + 16 * r - 12 = 0) →
  p * q + p * r + q * r = 8/3 := by
sorry

end NUMINAMATH_CALUDE_root_product_sum_l2005_200519


namespace NUMINAMATH_CALUDE_train_length_proof_l2005_200594

/-- Proves that the length of a train is equal to the total length of the train and bridge,
    given the train's speed, time to cross the bridge, and total length of train and bridge. -/
theorem train_length_proof (train_speed : ℝ) (crossing_time : ℝ) (total_length : ℝ) :
  train_speed = 45 * (1000 / 3600) →
  crossing_time = 30 →
  total_length = 245 →
  total_length = train_speed * crossing_time - total_length + total_length :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_proof_l2005_200594


namespace NUMINAMATH_CALUDE_rectangle_perimeter_theorem_l2005_200550

theorem rectangle_perimeter_theorem (a b : ℕ) : 
  a ≠ b →                 -- non-square condition
  a * b = 4 * (2 * a + 2 * b) →  -- area equals four times perimeter
  2 * (a + b) = 66 :=     -- perimeter is 66
by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_theorem_l2005_200550


namespace NUMINAMATH_CALUDE_eighteenth_replacement_is_march_l2005_200506

def months : List String := ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

def replacement_interval : Nat := 5

def first_replacement_month : String := "February"

def nth_replacement (n : Nat) : Nat :=
  replacement_interval * (n - 1)

theorem eighteenth_replacement_is_march :
  let months_after_february := nth_replacement 18
  let month_index := months_after_february % 12
  let replacement_month := months[(months.indexOf first_replacement_month + month_index) % 12]
  replacement_month = "March" := by
  sorry

end NUMINAMATH_CALUDE_eighteenth_replacement_is_march_l2005_200506


namespace NUMINAMATH_CALUDE_james_remaining_money_l2005_200579

/-- Calculates the remaining money for James after his purchases --/
def remaining_money (weekly_allowance : ℕ) (weeks_saved : ℕ) : ℕ :=
  let total_savings := weekly_allowance * weeks_saved
  let after_video_game := total_savings / 2
  let after_book := after_video_game - (after_video_game / 4)
  after_book

/-- Proves that James has $15 left after his purchases --/
theorem james_remaining_money :
  remaining_money 10 4 = 15 := by
  sorry

end NUMINAMATH_CALUDE_james_remaining_money_l2005_200579


namespace NUMINAMATH_CALUDE_group_selection_problem_l2005_200536

theorem group_selection_problem (n : ℕ) (k : ℕ) : n = 30 ∧ k = 3 → Nat.choose n k = 4060 := by
  sorry

end NUMINAMATH_CALUDE_group_selection_problem_l2005_200536


namespace NUMINAMATH_CALUDE_inverse_of_B_squared_l2005_200582

def B_inv : Matrix (Fin 2) (Fin 2) ℤ := !![1, 4; -2, -7]

theorem inverse_of_B_squared :
  let B_squared_inv : Matrix (Fin 2) (Fin 2) ℤ := !![(-7), (-24); 12, 41]
  (B_inv * B_inv) * (B_inv⁻¹ * B_inv⁻¹) = 1 ∧ (B_inv⁻¹ * B_inv⁻¹) * (B_inv * B_inv) = 1 := by
  sorry

end NUMINAMATH_CALUDE_inverse_of_B_squared_l2005_200582


namespace NUMINAMATH_CALUDE_cubic_factorization_l2005_200587

theorem cubic_factorization (x : ℝ) : x^3 + 3*x^2 - 4 = (x-1)*(x+2)^2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l2005_200587


namespace NUMINAMATH_CALUDE_cubic_polynomial_coefficient_l2005_200539

theorem cubic_polynomial_coefficient (a₀ a₁ a₂ a₃ : ℝ) :
  (∀ x : ℝ, x^3 = a₀ + a₁*(x-2) + a₂*(x-2)^2 + a₃*(x-2)^3) →
  a₂ = 6 := by
sorry

end NUMINAMATH_CALUDE_cubic_polynomial_coefficient_l2005_200539


namespace NUMINAMATH_CALUDE_frustum_properties_l2005_200563

/-- Frustum properties -/
structure Frustum where
  r₁ : ℝ  -- radius of top base
  r₂ : ℝ  -- radius of bottom base
  l : ℝ   -- slant height
  h : ℝ   -- height

/-- Theorem about a specific frustum -/
theorem frustum_properties (f : Frustum) (h_r₁ : f.r₁ = 2) (h_r₂ : f.r₂ = 6)
    (h_lateral_area : π * (f.r₁ + f.r₂) * f.l = π * f.r₁^2 + π * f.r₂^2) :
    f.l = 5 ∧ π * f.h * (f.r₁^2 + f.r₂^2 + f.r₁ * f.r₂) / 3 = 52 * π := by
  sorry


end NUMINAMATH_CALUDE_frustum_properties_l2005_200563


namespace NUMINAMATH_CALUDE_cake_slices_proof_l2005_200549

/-- The number of calories in each slice of cake -/
def calories_per_cake_slice : ℕ := 347

/-- The number of brownies in a pan -/
def brownies_per_pan : ℕ := 6

/-- The number of calories in each brownie -/
def calories_per_brownie : ℕ := 375

/-- The difference in calories between the cake and the pan of brownies -/
def calorie_difference : ℕ := 526

/-- The number of slices in the cake -/
def cake_slices : ℕ := 8

theorem cake_slices_proof :
  cake_slices * calories_per_cake_slice = 
  brownies_per_pan * calories_per_brownie + calorie_difference := by
  sorry

end NUMINAMATH_CALUDE_cake_slices_proof_l2005_200549


namespace NUMINAMATH_CALUDE_geometric_progression_values_l2005_200502

theorem geometric_progression_values (p : ℝ) : 
  (∃ r : ℝ, r ≠ 0 ∧ (9 * p + 10) * r = 3 * p ∧ (3 * p) * r = |p - 8|) ↔ 
  (p = -1 ∨ p = 40 / 9) := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_values_l2005_200502


namespace NUMINAMATH_CALUDE_prob_at_least_one_japanese_events_independent_iff_l2005_200532

-- Define the Little Green Lotus structure
structure LittleGreenLotus where
  isBoy : Bool
  speaksJapanese : Bool
  speaksKorean : Bool

-- Define the total number of Little Green Lotus
def totalLotus : ℕ := 36

-- Define the number of boys and girls
def numBoys : ℕ := 12
def numGirls : ℕ := 24

-- Define the number of boys and girls who can speak Japanese
def numBoysJapanese : ℕ := 8
def numGirlsJapanese : ℕ := 12

-- Define the number of boys and girls who can speak Korean as variables
variable (m n : ℕ)

-- Define the constraints on m
axiom m_bounds : 6 ≤ m ∧ m ≤ 8

-- Define the events A and B
def eventA (lotus : LittleGreenLotus) : Prop := lotus.isBoy
def eventB (lotus : LittleGreenLotus) : Prop := lotus.speaksKorean

-- Theorem 1: Probability of at least one of two randomly selected Little Green Lotus can speak Japanese
theorem prob_at_least_one_japanese :
  (totalLotus.choose 2 - (totalLotus - numBoysJapanese - numGirlsJapanese).choose 2) / totalLotus.choose 2 = 17 / 21 := by
  sorry

-- Theorem 2: Events A and B are independent if and only if n = 2m
theorem events_independent_iff (m n : ℕ) (h : 6 ≤ m ∧ m ≤ 8) :
  (numBoys * (m + n) = m * totalLotus) ↔ n = 2 * m := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_japanese_events_independent_iff_l2005_200532


namespace NUMINAMATH_CALUDE_base_representation_theorem_l2005_200573

theorem base_representation_theorem :
  (∃ b : ℕ, 1 < b ∧ b < 1993 ∧ 1994 = 2 * (1 + b)) ∧
  (∀ b : ℕ, 1 < b → b < 1992 → ¬∃ n : ℕ, n ≥ 2 ∧ 1993 * (b - 1) = b^n - 1) :=
by sorry

end NUMINAMATH_CALUDE_base_representation_theorem_l2005_200573


namespace NUMINAMATH_CALUDE_brother_age_l2005_200564

theorem brother_age (man_age brother_age : ℕ) : 
  man_age = brother_age + 12 →
  man_age + 2 = 2 * (brother_age + 2) →
  brother_age = 10 := by
sorry

end NUMINAMATH_CALUDE_brother_age_l2005_200564


namespace NUMINAMATH_CALUDE_combined_original_price_l2005_200565

/-- Given a pair of shoes with a 20% discount sold for $480 and a dress with a 30% discount sold for $350,
    prove that the combined original price of the shoes and dress is $1100. -/
theorem combined_original_price 
  (shoes_discount : Real) (dress_discount : Real)
  (shoes_discounted_price : Real) (dress_discounted_price : Real)
  (h1 : shoes_discount = 0.2)
  (h2 : dress_discount = 0.3)
  (h3 : shoes_discounted_price = 480)
  (h4 : dress_discounted_price = 350) :
  (shoes_discounted_price / (1 - shoes_discount)) + (dress_discounted_price / (1 - dress_discount)) = 1100 := by
  sorry

end NUMINAMATH_CALUDE_combined_original_price_l2005_200565


namespace NUMINAMATH_CALUDE_box_values_equality_l2005_200555

theorem box_values_equality : 40506000 = 4 * 10000000 + 5 * 100000 + 6 * 1000 := by
  sorry

end NUMINAMATH_CALUDE_box_values_equality_l2005_200555


namespace NUMINAMATH_CALUDE_ab_value_l2005_200523

theorem ab_value (a b : ℝ) : (a - b - 3) * (a - b + 3) = 40 → (a - b = 7 ∨ a - b = -7) := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l2005_200523


namespace NUMINAMATH_CALUDE_tv_sale_net_effect_l2005_200551

/-- Given a TV set with an original price P, this theorem proves the net effect on total sale value
    after applying discounts, considering sales increase and variable costs. -/
theorem tv_sale_net_effect (P : ℝ) (original_volume : ℝ) (h_pos : P > 0) (h_vol_pos : original_volume > 0) :
  let price_after_initial_reduction := P * (1 - 0.22)
  let bulk_discount := price_after_initial_reduction * 0.05
  let loyalty_discount := price_after_initial_reduction * 0.10
  let price_after_all_discounts := price_after_initial_reduction - bulk_discount - loyalty_discount
  let new_sales_volume := original_volume * 1.86
  let variable_cost_per_unit := price_after_all_discounts * 0.10
  let net_price_after_costs := price_after_all_discounts - variable_cost_per_unit
  let original_total_sale := P * original_volume
  let new_total_sale := net_price_after_costs * new_sales_volume
  let net_effect := new_total_sale - original_total_sale
  ∃ ε > 0, |net_effect / original_total_sale - 0.109862| < ε :=
by sorry

end NUMINAMATH_CALUDE_tv_sale_net_effect_l2005_200551


namespace NUMINAMATH_CALUDE_set_equality_implies_sum_l2005_200545

theorem set_equality_implies_sum (a b : ℝ) : 
  ({-1, a} : Set ℝ) = ({b, 1} : Set ℝ) → a + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_sum_l2005_200545


namespace NUMINAMATH_CALUDE_card_stack_problem_l2005_200585

theorem card_stack_problem (n : ℕ) : 
  let total_cards := 2 * n
  let pile_A := n
  let pile_B := n
  let card_80_position := 80
  (card_80_position ≤ pile_A) →
  (card_80_position % 2 = 1) →
  (∃ (new_position : ℕ), new_position = card_80_position ∧ 
    new_position = pile_B + (card_80_position + 1) / 2) →
  total_cards = 240 := by
sorry

end NUMINAMATH_CALUDE_card_stack_problem_l2005_200585


namespace NUMINAMATH_CALUDE_range_of_function_l2005_200583

theorem range_of_function (a b : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : 1 < b) 
  (h4 : b = (1 + Real.sqrt 5) / 2 * a) : 
  ∃ (x : ℝ), (9 - 9 * Real.sqrt 5) / 32 < a * (b - 3/2) ∧ 
             a * (b - 3/2) < (Real.sqrt 5 - 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_function_l2005_200583


namespace NUMINAMATH_CALUDE_remainder_problem_l2005_200526

theorem remainder_problem (s t u : ℕ) 
  (hs : s % 12 = 4)
  (ht : t % 12 = 5)
  (hu : u % 12 = 7)
  (hst : s > t)
  (htu : t > u) :
  ((s - t) + (t - u)) % 12 = 9 :=
by sorry

end NUMINAMATH_CALUDE_remainder_problem_l2005_200526


namespace NUMINAMATH_CALUDE_possible_zero_point_l2005_200591

theorem possible_zero_point (f : ℝ → ℝ) 
  (h1 : f 2015 < 0) 
  (h2 : f 2016 < 0) 
  (h3 : f 2017 > 0) : 
  ∃ x ∈ Set.Ioo 2015 2016, f x = 0 ∨ ∀ x ∈ Set.Ioo 2015 2016, f x ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_possible_zero_point_l2005_200591


namespace NUMINAMATH_CALUDE_max_profit_and_break_even_l2005_200505

/-- Revenue function (in ten thousand yuan) -/
def R (x : ℝ) : ℝ := 5 * x - x^2

/-- Cost function (in ten thousand yuan) -/
def C (x : ℝ) : ℝ := 0.5 + 0.25 * x

/-- Profit function (in ten thousand yuan) -/
def profit (x : ℝ) : ℝ := R x - C x

/-- Annual demand in hundreds of units -/
def annual_demand : ℝ := 5

theorem max_profit_and_break_even :
  ∃ (max_profit_units : ℝ) (break_even_lower break_even_upper : ℝ),
    (∀ x, 0 ≤ x → x ≤ annual_demand → profit x ≤ profit max_profit_units) ∧
    (max_profit_units = 4.75) ∧
    (break_even_lower = 0.1) ∧
    (break_even_upper = 48) ∧
    (∀ x, break_even_lower ≤ x → x ≤ break_even_upper → profit x ≥ 0) :=
  sorry

end NUMINAMATH_CALUDE_max_profit_and_break_even_l2005_200505


namespace NUMINAMATH_CALUDE_trig_equation_solution_l2005_200597

theorem trig_equation_solution (x : Real) : 
  (Real.sin (π/4 + 5*x) * Real.cos (π/4 + 2*x) = Real.sin (π/4 + x) * Real.sin (π/4 - 6*x)) ↔ 
  (∃ n : Int, x = n * π/4) :=
by sorry

end NUMINAMATH_CALUDE_trig_equation_solution_l2005_200597


namespace NUMINAMATH_CALUDE_complex_fraction_evaluation_l2005_200546

theorem complex_fraction_evaluation (a b : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : a^2 + a*b + b^2 = 0) : 
  (a^5 + b^5) / (a + b)^5 = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_evaluation_l2005_200546


namespace NUMINAMATH_CALUDE_fifth_term_of_sequence_l2005_200542

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem fifth_term_of_sequence (x y : ℝ) :
  ∀ a : ℕ → ℝ, arithmetic_sequence a →
  a 1 = x - y → a 2 = x → a 3 = x + y → a 4 = x + 2*y →
  a 5 = x + 3*y := by
sorry

end NUMINAMATH_CALUDE_fifth_term_of_sequence_l2005_200542


namespace NUMINAMATH_CALUDE_dot_product_range_l2005_200509

/-- Given a fixed point M(0, 4) and a point P(x, y) on the circle x^2 + y^2 = 4,
    the dot product of MP⃗ and OP⃗ is bounded between -4 and 12. -/
theorem dot_product_range (x y : ℝ) : 
  x^2 + y^2 = 4 → 
  -4 ≤ x * x + y * y - 4 * y ∧ x * x + y * y - 4 * y ≤ 12 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_range_l2005_200509


namespace NUMINAMATH_CALUDE_chord_length_l2005_200558

theorem chord_length (r : ℝ) (h : r = 15) : 
  ∃ (c : ℝ), c = 26 * Real.sqrt 3 ∧ 
  c = 2 * Real.sqrt (r^2 - (r/2)^2) := by
sorry

end NUMINAMATH_CALUDE_chord_length_l2005_200558


namespace NUMINAMATH_CALUDE_frog_jump_probability_l2005_200568

-- Define the probability function
noncomputable def Q (x y : ℝ) : ℝ := sorry

-- Define the boundary conditions
axiom vertical_boundary : ∀ y, 0 ≤ y ∧ y ≤ 6 → Q 0 y = 1 ∧ Q 6 y = 1
axiom horizontal_boundary : ∀ x, 0 ≤ x ∧ x ≤ 6 → Q x 0 = 0 ∧ Q x 6 = 0

-- Define the recursive relation
axiom recursive_relation : 
  Q 2 3 = (1/4) * Q 1 3 + (1/4) * Q 3 3 + (1/4) * Q 2 2 + (1/4) * Q 2 4

-- Theorem to prove
theorem frog_jump_probability : Q 2 3 = 5/8 := by sorry

end NUMINAMATH_CALUDE_frog_jump_probability_l2005_200568


namespace NUMINAMATH_CALUDE_min_value_of_3x_plus_4y_min_value_is_five_l2005_200596

theorem min_value_of_3x_plus_4y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 3 * y = 5 * x * y) :
  ∀ a b : ℝ, a > 0 → b > 0 → a + 3 * b = 5 * a * b → 3 * x + 4 * y ≤ 3 * a + 4 * b :=
by sorry

theorem min_value_is_five (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 3 * y = 5 * x * y) :
  3 * x + 4 * y ≥ 5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_3x_plus_4y_min_value_is_five_l2005_200596


namespace NUMINAMATH_CALUDE_chessboard_touching_squares_probability_l2005_200570

/-- Represents a square on the chessboard -/
structure Square where
  row : Fin 8
  col : Fin 8

/-- Checks if two squares are touching -/
def are_touching (s1 s2 : Square) : Prop :=
  (s1.row = s2.row ∧ s1.col.val + 1 = s2.col.val) ∨
  (s1.row = s2.row ∧ s1.col.val = s2.col.val + 1) ∨
  (s1.col = s2.col ∧ s1.row.val + 1 = s2.row.val) ∨
  (s1.col = s2.col ∧ s1.row.val = s2.row.val + 1) ∨
  (s1.row.val + 1 = s2.row.val ∧ s1.col.val + 1 = s2.col.val) ∨
  (s1.row.val + 1 = s2.row.val ∧ s1.col.val = s2.col.val + 1) ∨
  (s1.row.val = s2.row.val + 1 ∧ s1.col.val + 1 = s2.col.val) ∨
  (s1.row.val = s2.row.val + 1 ∧ s1.col.val = s2.col.val + 1)

/-- Checks if two squares are the same color -/
def same_color (s1 s2 : Square) : Prop :=
  (s1.row.val + s1.col.val) % 2 = (s2.row.val + s2.col.val) % 2

theorem chessboard_touching_squares_probability :
  ∀ (s1 s2 : Square), s1 ≠ s2 → are_touching s1 s2 → ¬(same_color s1 s2) :=
by sorry

end NUMINAMATH_CALUDE_chessboard_touching_squares_probability_l2005_200570


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_conditions_l2005_200554

/-- Represents the equation (x^2)/(2m) - (y^2)/(m-6) = 1 as an ellipse with foci on the y-axis -/
def proposition_p (m : ℝ) : Prop :=
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a > b ∧
  (∀ x y : ℝ, x^2 / (2*m) - y^2 / (m-6) = 1 ↔ x^2 / a^2 + y^2 / b^2 = 1)

/-- Represents the equation (x^2)/(m+1) + (y^2)/(m-1) = 1 as a hyperbola -/
def proposition_q (m : ℝ) : Prop :=
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧
  (∀ x y : ℝ, x^2 / (m+1) + y^2 / (m-1) = 1 ↔ x^2 / a^2 - y^2 / b^2 = 1)

/-- Theorem stating the conditions for proposition_p and proposition_q -/
theorem ellipse_hyperbola_conditions (m : ℝ) :
  (proposition_p m ↔ 0 < m ∧ m < 2) ∧
  (¬proposition_q m ↔ m ≤ -1 ∨ m ≥ 1) :=
sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_conditions_l2005_200554


namespace NUMINAMATH_CALUDE_rollercoaster_time_interval_l2005_200581

theorem rollercoaster_time_interval
  (total_students : ℕ)
  (total_time : ℕ)
  (group_size : ℕ)
  (h1 : total_students = 21)
  (h2 : total_time = 15)
  (h3 : group_size = 7)
  : (total_time / (total_students / group_size) : ℚ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_rollercoaster_time_interval_l2005_200581


namespace NUMINAMATH_CALUDE_polygon_contains_center_l2005_200538

/-- A convex polygon type -/
structure ConvexPolygon where
  area : ℝ
  isConvex : Bool

/-- A circle type -/
structure Circle where
  radius : ℝ

/-- Predicate to check if a polygon is inside a circle -/
def isInside (p : ConvexPolygon) (c : Circle) : Prop :=
  sorry

/-- Predicate to check if a polygon contains the center of a circle -/
def containsCenter (p : ConvexPolygon) (c : Circle) : Prop :=
  sorry

/-- Theorem statement -/
theorem polygon_contains_center (p : ConvexPolygon) (c : Circle) :
  p.area = 7 ∧ p.isConvex = true ∧ c.radius = 2 ∧ isInside p c → containsCenter p c :=
sorry

end NUMINAMATH_CALUDE_polygon_contains_center_l2005_200538


namespace NUMINAMATH_CALUDE_acme_vowel_soup_sequences_l2005_200517

def vowel_count : Nat := 5
def sequence_length : Nat := 5
def min_vowel_quantity : Nat := 3

theorem acme_vowel_soup_sequences :
  (vowel_count ^ sequence_length : Nat) = 3125 :=
by sorry

end NUMINAMATH_CALUDE_acme_vowel_soup_sequences_l2005_200517


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l2005_200527

def p (m : ℝ) (x : ℝ) : ℝ := 4 * x^3 - 16 * x^2 + m * x - 20

theorem polynomial_divisibility (m : ℝ) :
  (∃ q : ℝ → ℝ, ∀ x, p m x = (x - 4) * q x) →
  (m = 5 ∧ ¬∃ r : ℝ → ℝ, ∀ x, p 5 x = (x - 5) * r x) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l2005_200527


namespace NUMINAMATH_CALUDE_unique_g_two_l2005_200588

/-- A function satisfying the given functional equation -/
def FunctionalEquation (g : ℝ → ℝ) : Prop :=
  ∀ x y, x > 0 → y > 0 → g x * g y = g (x * y) + 2010 * (1 / x^2 + 1 / y^2 + 2009)

theorem unique_g_two (g : ℝ → ℝ) (h : FunctionalEquation g) :
    ∃! v, g 2 = v ∧ v = 8041 / 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_g_two_l2005_200588


namespace NUMINAMATH_CALUDE_sine_function_amplitude_l2005_200535

theorem sine_function_amplitude (a b : ℝ) (ha : a < 0) (hb : b > 0) :
  (∀ x, |a * Real.sin (b * x)| ≤ 3) ∧ (∃ x, |a * Real.sin (b * x)| = 3) → a = -3 := by
  sorry

end NUMINAMATH_CALUDE_sine_function_amplitude_l2005_200535


namespace NUMINAMATH_CALUDE_vasya_numbers_l2005_200561

theorem vasya_numbers : ∃ (x y : ℝ), x + y = x * y ∧ x + y = x / y ∧ x = (1 : ℝ) / 2 ∧ y = -1 := by
  sorry

end NUMINAMATH_CALUDE_vasya_numbers_l2005_200561


namespace NUMINAMATH_CALUDE_rectangle_selections_count_l2005_200580

/-- The number of ways to choose lines to form a rectangle with color constraints -/
def rectangle_selections (total_horizontal : ℕ) (red_horizontal : ℕ) 
                         (total_vertical : ℕ) (blue_vertical : ℕ) : ℕ :=
  let horizontal_selections := 
    (red_horizontal.choose 1 * (total_horizontal - red_horizontal).choose 1) +
    (red_horizontal.choose 2 * (total_horizontal - red_horizontal).choose 0)
  let vertical_selections := 
    (blue_vertical.choose 1 * (total_vertical - blue_vertical).choose 1) +
    (blue_vertical.choose 2 * (total_vertical - blue_vertical).choose 0)
  horizontal_selections * vertical_selections

/-- Theorem stating the number of ways to choose lines for the rectangle -/
theorem rectangle_selections_count :
  rectangle_selections 6 3 5 2 = 84 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_selections_count_l2005_200580


namespace NUMINAMATH_CALUDE_ipod_final_price_l2005_200520

/-- Calculates the final price of an item after two discounts and a compound sales tax. -/
def final_price (original_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) (tax_rate : ℝ) : ℝ :=
  let price_after_discount1 := original_price * (1 - discount1)
  let price_after_discount2 := price_after_discount1 * (1 - discount2)
  price_after_discount2 * (1 + tax_rate)

/-- Theorem stating that the final price of the iPod is approximately $77.08 -/
theorem ipod_final_price :
  ∃ ε > 0, |final_price 128 (7/20) 0.15 0.09 - 77.08| < ε :=
sorry

end NUMINAMATH_CALUDE_ipod_final_price_l2005_200520


namespace NUMINAMATH_CALUDE_degrees_to_radians_1920_l2005_200512

theorem degrees_to_radians_1920 : 
  (1920 : ℝ) * (π / 180) = (32 * π) / 3 := by sorry

end NUMINAMATH_CALUDE_degrees_to_radians_1920_l2005_200512


namespace NUMINAMATH_CALUDE_quadratic_vertex_form_l2005_200533

theorem quadratic_vertex_form (x : ℝ) : 
  ∃ (a h k : ℝ), 3 * x^2 + 9 * x + 20 = a * (x - h)^2 + k ∧ h = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_vertex_form_l2005_200533


namespace NUMINAMATH_CALUDE_current_speed_is_correct_l2005_200557

/-- Represents the speed of a swimmer in still water -/
def swimmer_speed : ℝ := 6.5

/-- Represents the speed of the current -/
def current_speed : ℝ := 4.5

/-- Represents the distance traveled downstream -/
def downstream_distance : ℝ := 55

/-- Represents the distance traveled upstream -/
def upstream_distance : ℝ := 10

/-- Represents the time taken for both downstream and upstream journeys -/
def travel_time : ℝ := 5

/-- Theorem stating that given the conditions, the speed of the current is 4.5 km/h -/
theorem current_speed_is_correct : 
  downstream_distance / travel_time = swimmer_speed + current_speed ∧
  upstream_distance / travel_time = swimmer_speed - current_speed →
  current_speed = 4.5 := by
  sorry

#check current_speed_is_correct

end NUMINAMATH_CALUDE_current_speed_is_correct_l2005_200557


namespace NUMINAMATH_CALUDE_sum_difference_even_odd_100_l2005_200500

/-- Sum of first n positive even integers -/
def sumEvenIntegers (n : ℕ) : ℕ := n * (n + 1)

/-- Sum of first n positive odd integers -/
def sumOddIntegers (n : ℕ) : ℕ := n^2

theorem sum_difference_even_odd_100 :
  sumEvenIntegers 100 - sumOddIntegers 100 = 100 := by
  sorry

end NUMINAMATH_CALUDE_sum_difference_even_odd_100_l2005_200500


namespace NUMINAMATH_CALUDE_smallest_linear_combination_divides_l2005_200578

theorem smallest_linear_combination_divides (a b x₀ y₀ : ℤ) 
  (h_not_zero : a ≠ 0 ∨ b ≠ 0)
  (h_smallest : ∀ x y : ℤ, a * x + b * y > 0 → a * x₀ + b * y₀ ≤ a * x + b * y) :
  ∀ x y : ℤ, ∃ k : ℤ, a * x + b * y = k * (a * x₀ + b * y₀) := by
sorry

end NUMINAMATH_CALUDE_smallest_linear_combination_divides_l2005_200578


namespace NUMINAMATH_CALUDE_ariel_fencing_years_l2005_200559

theorem ariel_fencing_years (birth_year : ℕ) (fencing_start : ℕ) (current_age : ℕ) : 
  birth_year = 1992 → fencing_start = 2006 → current_age = 30 → 
  fencing_start - birth_year - current_age = 16 := by
  sorry

end NUMINAMATH_CALUDE_ariel_fencing_years_l2005_200559


namespace NUMINAMATH_CALUDE_largest_fraction_l2005_200525

theorem largest_fraction : 
  let fractions := [1/5, 2/10, 7/15, 9/20, 3/6]
  ∀ x ∈ fractions, x ≤ (3:ℚ)/6 := by
sorry

end NUMINAMATH_CALUDE_largest_fraction_l2005_200525


namespace NUMINAMATH_CALUDE_fruit_pricing_problem_l2005_200595

theorem fruit_pricing_problem (x y : ℚ) : 
  x + y = 1000 →
  (11/9) * x + (4/7) * y = 999 →
  (9 * (11/9) = 11 ∧ 7 * (4/7) = 4) :=
by sorry

end NUMINAMATH_CALUDE_fruit_pricing_problem_l2005_200595


namespace NUMINAMATH_CALUDE_f_of_two_equals_one_l2005_200521

-- Define the function f
def f : ℝ → ℝ := fun x => (x - 1)^2

-- Theorem statement
theorem f_of_two_equals_one : f 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_f_of_two_equals_one_l2005_200521


namespace NUMINAMATH_CALUDE_decimal_93_to_binary_binary_to_decimal_93_l2005_200507

/-- Converts a natural number to its binary representation as a list of bits -/
def to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
  aux n

/-- Converts a list of bits to its decimal representation -/
def from_binary (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

theorem decimal_93_to_binary :
  to_binary 93 = [true, false, true, true, true, false, true] :=
sorry

theorem binary_to_decimal_93 :
  from_binary [true, false, true, true, true, false, true] = 93 :=
sorry

end NUMINAMATH_CALUDE_decimal_93_to_binary_binary_to_decimal_93_l2005_200507


namespace NUMINAMATH_CALUDE_complex_number_purely_imaginary_l2005_200571

theorem complex_number_purely_imaginary (a : ℝ) : 
  (a = -1) ↔ (∃ (t : ℝ), (1 + I) / (1 + a * I) = t * I) :=
sorry

end NUMINAMATH_CALUDE_complex_number_purely_imaginary_l2005_200571


namespace NUMINAMATH_CALUDE_investment_interest_proof_l2005_200504

/-- Calculates the interest earned on an investment with annual compounding -/
def interest_earned (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate) ^ years - principal

/-- Proves that the interest earned on a $500 investment at 2% annual rate for 3 years is approximately $30.60 -/
theorem investment_interest_proof :
  let principal := 500
  let rate := 0.02
  let years := 3
  abs (interest_earned principal rate years - 30.60) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_investment_interest_proof_l2005_200504
