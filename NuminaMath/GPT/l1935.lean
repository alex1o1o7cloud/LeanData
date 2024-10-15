import Mathlib

namespace NUMINAMATH_GPT_paper_cups_pallets_l1935_193562

theorem paper_cups_pallets (total_pallets : ℕ) (paper_towels_fraction tissues_fraction paper_plates_fraction : ℚ) :
  total_pallets = 20 → paper_towels_fraction = 1 / 2 → tissues_fraction = 1 / 4 → paper_plates_fraction = 1 / 5 →
  total_pallets - (total_pallets * paper_towels_fraction + total_pallets * tissues_fraction + total_pallets * paper_plates_fraction) = 1 :=
by sorry

end NUMINAMATH_GPT_paper_cups_pallets_l1935_193562


namespace NUMINAMATH_GPT_lisa_matching_pair_probability_l1935_193590

theorem lisa_matching_pair_probability :
  let total_socks := 22
  let gray_socks := 12
  let white_socks := 10
  let total_pairs := total_socks * (total_socks - 1) / 2
  let gray_pairs := gray_socks * (gray_socks - 1) / 2
  let white_pairs := white_socks * (white_socks - 1) / 2
  let matching_pairs := gray_pairs + white_pairs
  let probability := matching_pairs / total_pairs
  probability = (111 / 231) :=
by
  sorry

end NUMINAMATH_GPT_lisa_matching_pair_probability_l1935_193590


namespace NUMINAMATH_GPT_magnitude_of_complex_l1935_193505

def complex_number := Complex.mk 2 3 -- Define the complex number 2+3i

theorem magnitude_of_complex : Complex.abs complex_number = Real.sqrt 13 := by
  sorry

end NUMINAMATH_GPT_magnitude_of_complex_l1935_193505


namespace NUMINAMATH_GPT_average_interest_rate_l1935_193594

theorem average_interest_rate
  (x : ℝ)
  (h₀ : 0 ≤ x)
  (h₁ : x ≤ 5000)
  (h₂ : 0.05 * x = 0.03 * (5000 - x)) :
  (0.05 * x + 0.03 * (5000 - x)) / 5000 = 0.0375 :=
by
  sorry

end NUMINAMATH_GPT_average_interest_rate_l1935_193594


namespace NUMINAMATH_GPT_quadratic_eq_coeff_m_l1935_193574

theorem quadratic_eq_coeff_m (m : ℤ) : 
  (|m| = 2 ∧ m + 2 ≠ 0) → m = 2 := 
by
  intro h
  sorry

end NUMINAMATH_GPT_quadratic_eq_coeff_m_l1935_193574


namespace NUMINAMATH_GPT_min_value_geometric_sequence_l1935_193546

-- Definitions based on conditions
noncomputable def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) := ∀ n : ℕ, a (n + 1) = a n * q

-- We need to state the problem using the above definitions
theorem min_value_geometric_sequence 
  (a : ℕ → ℝ) (q : ℝ) (s t : ℕ) 
  (h_seq : is_geometric_sequence a q) 
  (h_q : q ≠ 1) 
  (h_st : a s * a t = (a 5) ^ 2) 
  (h_s_pos : s > 0) 
  (h_t_pos : t > 0) 
  : 4 / s + 1 / (4 * t) = 5 / 8 := sorry

end NUMINAMATH_GPT_min_value_geometric_sequence_l1935_193546


namespace NUMINAMATH_GPT_jonah_total_raisins_l1935_193527

-- Define the amounts of yellow and black raisins added
def yellow_raisins : ℝ := 0.3
def black_raisins : ℝ := 0.4

-- The main statement to be proved
theorem jonah_total_raisins : yellow_raisins + black_raisins = 0.7 :=
by 
  sorry

end NUMINAMATH_GPT_jonah_total_raisins_l1935_193527


namespace NUMINAMATH_GPT_probability_queen_then_diamond_l1935_193552

-- Define a standard deck of 52 cards
def deck := List.range 52

-- Define a function to check if a card is a Queen
def is_queen (card : ℕ) : Prop :=
card % 13 = 10

-- Define a function to check if a card is a Diamond. Here assuming index for diamond starts at 0 and ends at 12
def is_diamond (card : ℕ) : Prop :=
card / 13 = 0

-- The main theorem statement
theorem probability_queen_then_diamond : 
  let prob := 1 / 52 * 12 / 51 + 3 / 52 * 13 / 51
  prob = 52 / 221 :=
by
  sorry

end NUMINAMATH_GPT_probability_queen_then_diamond_l1935_193552


namespace NUMINAMATH_GPT_sixth_number_is_811_l1935_193588

noncomputable def sixth_number_in_21st_row : ℕ := 
  let n := 21 
  let k := 6
  let total_numbers_up_to_previous_row := n * n
  let position_in_row := total_numbers_up_to_previous_row + k
  2 * position_in_row - 1

theorem sixth_number_is_811 : sixth_number_in_21st_row = 811 := by
  sorry

end NUMINAMATH_GPT_sixth_number_is_811_l1935_193588


namespace NUMINAMATH_GPT_maintain_constant_chromosomes_l1935_193517

-- Definitions
def meiosis_reduces_chromosomes (original_chromosomes : ℕ) : ℕ := original_chromosomes / 2

def fertilization_restores_chromosomes (half_chromosomes : ℕ) : ℕ := half_chromosomes * 2

-- The proof problem
theorem maintain_constant_chromosomes (original_chromosomes : ℕ) (somatic_chromosomes : ℕ) :
  meiosis_reduces_chromosomes original_chromosomes = somatic_chromosomes / 2 ∧
  fertilization_restores_chromosomes (meiosis_reduces_chromosomes original_chromosomes) = somatic_chromosomes :=
sorry

end NUMINAMATH_GPT_maintain_constant_chromosomes_l1935_193517


namespace NUMINAMATH_GPT_trader_gain_l1935_193500

-- Conditions
def cost_price (pen : Type) : ℕ → ℝ := sorry -- Type to represent the cost price of a pen
def selling_price (pen : Type) : ℕ → ℝ := sorry -- Type to represent the selling price of a pen
def gain_percentage : ℝ := 0.40 -- 40% gain

-- Statement of the problem to prove
theorem trader_gain (C : ℝ) (N : ℕ) : 
  (100 : ℕ) * C * gain_percentage = N * C → 
  N = 40 :=
by
  sorry

end NUMINAMATH_GPT_trader_gain_l1935_193500


namespace NUMINAMATH_GPT_digit_a_solution_l1935_193534

theorem digit_a_solution :
  ∃ a : ℕ, a000 + a998 + a999 = 22997 → a = 7 :=
sorry

end NUMINAMATH_GPT_digit_a_solution_l1935_193534


namespace NUMINAMATH_GPT_total_cost_is_15_l1935_193579

def toast_cost : ℕ := 1
def egg_cost : ℕ := 3

def dale_toast : ℕ := 2
def dale_eggs : ℕ := 2

def andrew_toast : ℕ := 1
def andrew_eggs : ℕ := 2

def dale_breakfast_cost := dale_toast * toast_cost + dale_eggs * egg_cost
def andrew_breakfast_cost := andrew_toast * toast_cost + andrew_eggs * egg_cost

def total_breakfast_cost := dale_breakfast_cost + andrew_breakfast_cost

theorem total_cost_is_15 : total_breakfast_cost = 15 := by
  sorry

end NUMINAMATH_GPT_total_cost_is_15_l1935_193579


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1935_193553

theorem solution_set_of_inequality :
  { x : ℝ | x * (x - 1) ≤ 0 } = { x : ℝ | 0 ≤ x ∧ x ≤ 1 } :=
by
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1935_193553


namespace NUMINAMATH_GPT_loaves_of_bread_can_bake_l1935_193528

def total_flour_in_cupboard := 200
def total_flour_on_counter := 100
def total_flour_in_pantry := 100
def flour_per_loaf := 200

theorem loaves_of_bread_can_bake :
  (total_flour_in_cupboard + total_flour_on_counter + total_flour_in_pantry) / flour_per_loaf = 2 := by
  sorry

end NUMINAMATH_GPT_loaves_of_bread_can_bake_l1935_193528


namespace NUMINAMATH_GPT_number_of_red_yarns_l1935_193533

-- Definitions
def scarves_per_yarn : Nat := 3
def blue_yarns : Nat := 6
def yellow_yarns : Nat := 4
def total_scarves : Nat := 36

-- Theorem
theorem number_of_red_yarns (R : Nat) (H1 : scarves_per_yarn * blue_yarns + scarves_per_yarn * yellow_yarns + scarves_per_yarn * R = total_scarves) :
  R = 2 :=
by
  sorry

end NUMINAMATH_GPT_number_of_red_yarns_l1935_193533


namespace NUMINAMATH_GPT_veronica_flashlight_distance_l1935_193589

theorem veronica_flashlight_distance (V F Vel : ℕ) 
  (h1 : F = 3 * V)
  (h2 : Vel = 5 * F - 2000)
  (h3 : Vel = V + 12000) : 
  V = 1000 := 
by {
  sorry 
}

end NUMINAMATH_GPT_veronica_flashlight_distance_l1935_193589


namespace NUMINAMATH_GPT_exit_forest_strategy_l1935_193506

/-- A strategy ensuring the parachutist will exit the forest with a path length of less than 2.5l -/
theorem exit_forest_strategy (l : Real) : 
  ∃ (path_length : Real), path_length < 2.5 * l :=
by
  use 2.278 * l
  sorry

end NUMINAMATH_GPT_exit_forest_strategy_l1935_193506


namespace NUMINAMATH_GPT_minimal_perimeter_triangle_l1935_193508

noncomputable def cos_P : ℚ := 3 / 5
noncomputable def cos_Q : ℚ := 24 / 25
noncomputable def cos_R : ℚ := -1 / 5

theorem minimal_perimeter_triangle
  (P Q R : ℝ) (a b c : ℕ)
  (h0 : a^2 + b^2 + c^2 - 2 * a * b * cos_P - 2 * b * c * cos_Q - 2 * c * a * cos_R = 0)
  (h1 : cos_P^2 + (1 - cos_P^2) = 1)
  (h2 : cos_Q^2 + (1 - cos_Q^2) = 1)
  (h3 : cos_R^2 + (1 - cos_R^2) = 1) :
  a + b + c = 47 :=
sorry

end NUMINAMATH_GPT_minimal_perimeter_triangle_l1935_193508


namespace NUMINAMATH_GPT_expand_polynomial_l1935_193554

theorem expand_polynomial : 
  ∀ (x : ℝ), (5 * x - 3) * (2 * x^2 + 4 * x + 1) = 10 * x^3 + 14 * x^2 - 7 * x - 3 :=
by
  intro x
  sorry

end NUMINAMATH_GPT_expand_polynomial_l1935_193554


namespace NUMINAMATH_GPT_harry_terry_difference_l1935_193530

theorem harry_terry_difference :
  let H := 8 - (2 + 5)
  let T := 8 - 2 + 5
  H - T = -10 :=
by 
  sorry

end NUMINAMATH_GPT_harry_terry_difference_l1935_193530


namespace NUMINAMATH_GPT_cubic_yards_to_cubic_feet_l1935_193560

theorem cubic_yards_to_cubic_feet (yards_to_feet: 1 = 3): 6 * 27 = 162 := by
  -- We know from the setup that:
  -- 1 cubic yard = 27 cubic feet
  -- Hence,
  -- 6 cubic yards = 6 * 27 = 162 cubic feet
  sorry

end NUMINAMATH_GPT_cubic_yards_to_cubic_feet_l1935_193560


namespace NUMINAMATH_GPT_repeated_digit_in_mod_sequence_l1935_193522

theorem repeated_digit_in_mod_sequence : 
  ∃ (x y : ℕ), x ≠ y ∧ (2^1970 % 9 = 4) ∧ 
  (∀ n : ℕ, n < 10 → n = 2^1970 % 9 → n = x ∨ n = y) :=
sorry

end NUMINAMATH_GPT_repeated_digit_in_mod_sequence_l1935_193522


namespace NUMINAMATH_GPT_friend_reading_time_l1935_193513

def my_reading_time : ℕ := 120  -- It takes me 120 minutes to read the novella

def speed_ratio : ℕ := 3  -- My friend reads three times as fast as I do

theorem friend_reading_time : my_reading_time / speed_ratio = 40 := by
  -- Proof
  sorry

end NUMINAMATH_GPT_friend_reading_time_l1935_193513


namespace NUMINAMATH_GPT_radius_of_sphere_inscribed_in_box_l1935_193509

theorem radius_of_sphere_inscribed_in_box (a b c s : ℝ)
  (h1 : a + b + c = 42)
  (h2 : 2 * (a * b + b * c + c * a) = 576)
  (h3 : (2 * s)^2 = a^2 + b^2 + c^2) :
  s = 3 * Real.sqrt 33 :=
by sorry

end NUMINAMATH_GPT_radius_of_sphere_inscribed_in_box_l1935_193509


namespace NUMINAMATH_GPT_average_side_length_of_squares_l1935_193526

theorem average_side_length_of_squares (a1 a2 a3 a4 : ℕ) 
(h1 : a1 = 36) (h2 : a2 = 64) (h3 : a3 = 100) (h4 : a4 = 144) :
(Real.sqrt a1 + Real.sqrt a2 + Real.sqrt a3 + Real.sqrt a4) / 4 = 9 := 
by
  sorry

end NUMINAMATH_GPT_average_side_length_of_squares_l1935_193526


namespace NUMINAMATH_GPT_find_a_of_extremum_l1935_193545

theorem find_a_of_extremum (a b : ℝ) (f : ℝ → ℝ) (f' : ℝ → ℝ)
  (h1 : f x = x^3 + a*x^2 + b*x + a^2)
  (h2 : f' x = 3*x^2 + 2*a*x + b)
  (h3 : f' 1 = 0)
  (h4 : f 1 = 10) : a = 4 := by
  sorry

end NUMINAMATH_GPT_find_a_of_extremum_l1935_193545


namespace NUMINAMATH_GPT_infinite_series_sum_l1935_193501

theorem infinite_series_sum
  (a b : ℝ)
  (h1 : (∑' n : ℕ, a / (b ^ (n + 1))) = 4) :
  (∑' n : ℕ, a / ((a + b) ^ (n + 1))) = 4 / 5 := 
sorry

end NUMINAMATH_GPT_infinite_series_sum_l1935_193501


namespace NUMINAMATH_GPT_total_lives_l1935_193539

/-- Suppose there are initially 4 players, then 5 more players join. Each player has 3 lives.
    Prove that the total number of lives is equal to 27. -/
theorem total_lives (initial_players : ℕ) (additional_players : ℕ) (lives_per_player : ℕ) 
  (h_initial : initial_players = 4) (h_additional : additional_players = 5) (h_lives : lives_per_player = 3) : 
  initial_players + additional_players = 9 ∧ 
  (initial_players + additional_players) * lives_per_player = 27 :=
by
  sorry

end NUMINAMATH_GPT_total_lives_l1935_193539


namespace NUMINAMATH_GPT_part1_l1935_193537

theorem part1 (a b c : ℤ) (h : a + b + c = 0) : a^3 + a^2 * c - a * b * c + b^2 * c + b^3 = 0 := 
sorry

end NUMINAMATH_GPT_part1_l1935_193537


namespace NUMINAMATH_GPT_pencil_ratio_l1935_193543

theorem pencil_ratio (B G : ℕ) (h1 : ∀ (n : ℕ), n = 20) 
  (h2 : ∀ (n : ℕ), n = 40) 
  (h3 : ∀ (n : ℕ), n = 160) 
  (h4 : G = 20 + B)
  (h5 : B + 20 + G + 40 = 160) : 
  (B / 20) = 4 := 
  by sorry

end NUMINAMATH_GPT_pencil_ratio_l1935_193543


namespace NUMINAMATH_GPT_y_range_l1935_193532

theorem y_range (x y : ℝ) (h1 : 4 * x + y = 1) (h2 : -1 < x) (h3 : x ≤ 2) : -7 ≤ y ∧ y < -3 := 
by
  sorry

end NUMINAMATH_GPT_y_range_l1935_193532


namespace NUMINAMATH_GPT_rashmi_bus_stop_distance_l1935_193576

theorem rashmi_bus_stop_distance
  (T D : ℝ)
  (h1 : 5 * (T + 10/60) = D)
  (h2 : 6 * (T - 10/60) = D) :
  D = 5 :=
by
  sorry

end NUMINAMATH_GPT_rashmi_bus_stop_distance_l1935_193576


namespace NUMINAMATH_GPT_grains_in_batch_l1935_193592

-- Define the given constants from the problem
def total_rice_shi : ℕ := 1680
def sample_total_grains : ℕ := 250
def sample_containing_grains : ℕ := 25

-- Define the statement to be proven
theorem grains_in_batch : (total_rice_shi * (sample_containing_grains / sample_total_grains)) = 168 := by
  -- Proof steps will go here
  sorry

end NUMINAMATH_GPT_grains_in_batch_l1935_193592


namespace NUMINAMATH_GPT_longer_train_length_l1935_193587

def length_of_longer_train
  (speed_train1 : ℝ) (speed_train2 : ℝ)
  (length_shorter_train : ℝ) (time_to_clear : ℝ)
  (relative_speed : ℝ := (speed_train1 + speed_train2) * 1000 / 3600)
  (total_distance : ℝ := relative_speed * time_to_clear) : ℝ :=
  total_distance - length_shorter_train

theorem longer_train_length :
  length_of_longer_train 80 55 121 7.626056582140095 = 164.9771230827526 :=
by
  unfold length_of_longer_train
  norm_num
  sorry  -- This placeholder is used to avoid writing out the full proof.

end NUMINAMATH_GPT_longer_train_length_l1935_193587


namespace NUMINAMATH_GPT_lowest_temperature_l1935_193571

theorem lowest_temperature 
  (temps : Fin 5 → ℝ) 
  (avg_temp : (temps 0 + temps 1 + temps 2 + temps 3 + temps 4) / 5 = 60)
  (max_range : ∀ i j, temps i - temps j ≤ 75) : 
  ∃ L : ℝ, L = 0 ∧ ∃ i, temps i = L :=
by 
  sorry

end NUMINAMATH_GPT_lowest_temperature_l1935_193571


namespace NUMINAMATH_GPT_bogatyrs_truthful_count_l1935_193564

noncomputable def number_of_truthful_warriors (total_warriors: ℕ) (sword_yes: ℕ) (spear_yes: ℕ) (axe_yes: ℕ) (bow_yes: ℕ) : ℕ :=
  let total_yes := sword_yes + spear_yes + axe_yes + bow_yes
  let lying_warriors := (total_yes - total_warriors) / 2
  total_warriors - lying_warriors

theorem bogatyrs_truthful_count :
  number_of_truthful_warriors 33 13 15 20 27 = 12 := by
  sorry

end NUMINAMATH_GPT_bogatyrs_truthful_count_l1935_193564


namespace NUMINAMATH_GPT_calculate_expression_l1935_193507

theorem calculate_expression : (1000^2) / (252^2 - 248^2) = 500 := sorry

end NUMINAMATH_GPT_calculate_expression_l1935_193507


namespace NUMINAMATH_GPT_binom_60_3_l1935_193599

theorem binom_60_3 : Nat.choose 60 3 = 34220 := by
  sorry

end NUMINAMATH_GPT_binom_60_3_l1935_193599


namespace NUMINAMATH_GPT_calculate_expression_l1935_193593

theorem calculate_expression : 14 - (-12) + (-25) - 17 = -16 := by
  -- definitions from conditions are understood and used here implicitly
  sorry

end NUMINAMATH_GPT_calculate_expression_l1935_193593


namespace NUMINAMATH_GPT_taehyung_math_score_l1935_193569

theorem taehyung_math_score
  (avg_before : ℝ)
  (drop_in_avg : ℝ)
  (num_subjects_before : ℕ)
  (num_subjects_after : ℕ)
  (avg_after : ℝ)
  (total_before : ℝ)
  (total_after : ℝ)
  (math_score : ℝ) :
  avg_before = 95 →
  drop_in_avg = 3 →
  num_subjects_before = 3 →
  num_subjects_after = 4 →
  avg_after = avg_before - drop_in_avg →
  total_before = avg_before * num_subjects_before →
  total_after = avg_after * num_subjects_after →
  math_score = total_after - total_before →
  math_score = 83 :=
by
  intros
  sorry

end NUMINAMATH_GPT_taehyung_math_score_l1935_193569


namespace NUMINAMATH_GPT_part1_part2_l1935_193504

-- Proof for part 1
theorem part1 (x : ℤ) : (x - 1 ∣ x - 3 ↔ (x = -1 ∨ x = 0 ∨ x = 2 ∨ x = 3)) :=
by sorry

-- Proof for part 2
theorem part2 (x : ℤ) : (x + 2 ∣ x^2 + 3 ↔ (x = -9 ∨ x = -3 ∨ x = -1 ∨ x = 5)) :=
by sorry

end NUMINAMATH_GPT_part1_part2_l1935_193504


namespace NUMINAMATH_GPT_find_a7_l1935_193520

section GeometricSequence

variables {a : ℕ → ℝ} (q : ℝ) (a1 : ℝ)

-- a_n is defined as a geometric sequence
def geometric_sequence (a : ℕ → ℝ) (a1 q : ℝ) : Prop :=
∀ n, a n = a1 * q^(n-1)

-- Given conditions
axiom h1 : geometric_sequence a a1 q
axiom h2 : a 2 * a 4 * a 5 = a 3 * a 6
axiom h3 : a 9 * a 10 = -8

--The proof goal
theorem find_a7 : a 7 = -2 :=
sorry

end GeometricSequence

end NUMINAMATH_GPT_find_a7_l1935_193520


namespace NUMINAMATH_GPT_min_avg_score_less_than_record_l1935_193551

theorem min_avg_score_less_than_record
  (old_record_avg : ℝ := 287.5)
  (players : ℕ := 6)
  (rounds : ℕ := 12)
  (total_points_11_rounds : ℝ := 19350.5)
  (bonus_points_9_rounds : ℕ := 300) :
  ∀ final_round_avg : ℝ, (final_round_avg = (old_record_avg * players * rounds - total_points_11_rounds + bonus_points_9_rounds) / players) →
  old_record_avg - final_round_avg = 12.5833 :=
by {
  sorry
}

end NUMINAMATH_GPT_min_avg_score_less_than_record_l1935_193551


namespace NUMINAMATH_GPT_gel_pen_ratio_l1935_193583

-- Definitions corresponding to the conditions in the problem
variables (x y : ℕ) (b g : ℝ)

-- The total amount paid 
def total_amount := x * b + y * g

-- Condition given in the problem
def condition1 := (x + y) * g = 4 * total_amount x y b g
def condition2 := (x + y) * b = (1/2) * total_amount x y b g

-- The theorem to prove the ratio of the price of a gel pen to a ballpoint pen is 8
theorem gel_pen_ratio (x y : ℕ) (b g : ℝ) (h1 : condition1 x y b g) (h2 : condition2 x y b g) : 
  g = 8 * b := by
  sorry

end NUMINAMATH_GPT_gel_pen_ratio_l1935_193583


namespace NUMINAMATH_GPT_lydia_candy_problem_l1935_193577

theorem lydia_candy_problem :
  ∃ m: ℕ, (∀ k: ℕ, (k * 24 = Nat.lcm (Nat.lcm 16 18) 20) → k ≥ m) ∧ 24 * m = Nat.lcm (Nat.lcm 16 18) 20 ∧ m = 30 :=
by
  sorry

end NUMINAMATH_GPT_lydia_candy_problem_l1935_193577


namespace NUMINAMATH_GPT_problem_1_problem_2_l1935_193568

noncomputable def f (x a : ℝ) : ℝ := |2 * x - a| + |2 * x - 1|

-- Problem 1: When a = -1, prove the solution set for f(x) ≤ 2 is [-1/2, 1/2].
theorem problem_1 (x : ℝ) : (f x (-1) ≤ 2) ↔ (-1/2 ≤ x ∧ x ≤ 1/2) := 
sorry

-- Problem 2: If the solution set of f(x) ≤ |2x + 1| contains the interval [1/2, 1], find the range of a.
theorem problem_2 (a : ℝ) : (∀ x, (1/2 ≤ x ∧ x ≤ 1) → f x a ≤ |2 * x + 1|) ↔ (0 ≤ a ∧ a ≤ 3) :=
sorry

end NUMINAMATH_GPT_problem_1_problem_2_l1935_193568


namespace NUMINAMATH_GPT_susie_earnings_l1935_193536

-- Define the constants and conditions
def price_per_slice : ℕ := 3
def price_per_whole_pizza : ℕ := 15
def slices_sold : ℕ := 24
def whole_pizzas_sold : ℕ := 3

-- Calculate earnings from slices and whole pizzas
def earnings_from_slices : ℕ := slices_sold * price_per_slice
def earnings_from_whole_pizzas : ℕ := whole_pizzas_sold * price_per_whole_pizza
def total_earnings : ℕ := earnings_from_slices + earnings_from_whole_pizzas

-- Prove that the total earnings are $117
theorem susie_earnings : total_earnings = 117 := by
  sorry

end NUMINAMATH_GPT_susie_earnings_l1935_193536


namespace NUMINAMATH_GPT_common_solutions_for_y_l1935_193531

theorem common_solutions_for_y (x y : ℝ) :
  (x^2 + y^2 = 16) ∧ (x^2 - 3 * y = 12) ↔ (y = -4 ∨ y = 1) :=
by
  sorry

end NUMINAMATH_GPT_common_solutions_for_y_l1935_193531


namespace NUMINAMATH_GPT_leah_probability_of_seeing_change_l1935_193549

open Set

-- Define the length of each color interval
def green_duration := 45
def yellow_duration := 5
def red_duration := 35

-- Total cycle duration
def total_cycle_duration := green_duration + yellow_duration + red_duration

-- Leah's viewing intervals
def change_intervals : Set (ℕ × ℕ) :=
  {(40, 45), (45, 50), (80, 85)}

-- Probability calculation
def favorable_time := 15
def probability_of_change := (favorable_time : ℚ) / (total_cycle_duration : ℚ)

theorem leah_probability_of_seeing_change : probability_of_change = 3 / 17 :=
by
  -- We use sorry here as we are only required to state the theorem without proof.
  sorry

end NUMINAMATH_GPT_leah_probability_of_seeing_change_l1935_193549


namespace NUMINAMATH_GPT_zoe_remaining_pictures_l1935_193518

-- Definitions for the problem conditions
def monday_pictures := 24
def tuesday_pictures := 37
def wednesday_pictures := 50
def thursday_pictures := 33
def friday_pictures := 44

def rate_first := 4
def rate_second := 5
def rate_third := 6
def rate_fourth := 3
def rate_fifth := 7

def days_colored (start_day : ℕ) (end_day := 6) := end_day - start_day

def remaining_pictures (total_pictures : ℕ) (rate_per_day : ℕ) (days : ℕ) : ℕ :=
  total_pictures - (rate_per_day * days)

-- Main theorem statement
theorem zoe_remaining_pictures : 
  remaining_pictures monday_pictures rate_first (days_colored 1) +
  remaining_pictures tuesday_pictures rate_second (days_colored 2) +
  remaining_pictures wednesday_pictures rate_third (days_colored 3) +
  remaining_pictures thursday_pictures rate_fourth (days_colored 4) +
  remaining_pictures friday_pictures rate_fifth (days_colored 5) = 117 :=
  sorry

end NUMINAMATH_GPT_zoe_remaining_pictures_l1935_193518


namespace NUMINAMATH_GPT_max_area_rectangle_l1935_193566

theorem max_area_rectangle (x y : ℝ) (h : 2 * x + 2 * y = 40) : x * y ≤ 100 :=
by
  sorry

end NUMINAMATH_GPT_max_area_rectangle_l1935_193566


namespace NUMINAMATH_GPT_max_dominoes_in_grid_l1935_193595

-- Definitions representing the conditions
def total_squares (rows cols : ℕ) : ℕ := rows * cols
def domino_squares : ℕ := 3
def max_dominoes (total domino : ℕ) : ℕ := total / domino

-- Statement of the problem
theorem max_dominoes_in_grid : max_dominoes (total_squares 20 19) domino_squares = 126 :=
by
  -- placeholders for the actual proof
  sorry

end NUMINAMATH_GPT_max_dominoes_in_grid_l1935_193595


namespace NUMINAMATH_GPT_pizzeria_large_pizzas_sold_l1935_193541

theorem pizzeria_large_pizzas_sold (price_small price_large total_earnings num_small_pizzas num_large_pizzas : ℕ):
  price_small = 2 →
  price_large = 8 →
  total_earnings = 40 →
  num_small_pizzas = 8 →
  total_earnings = price_small * num_small_pizzas + price_large * num_large_pizzas →
  num_large_pizzas = 3 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_pizzeria_large_pizzas_sold_l1935_193541


namespace NUMINAMATH_GPT_minimum_value_of_f_range_of_t_l1935_193586

def f (x : ℝ) : ℝ := abs (x + 2) + abs (x - 4)

theorem minimum_value_of_f : ∀ x, f x ≥ 6 ∧ ∃ x0 : ℝ, f x0 = 6 := 
by sorry

theorem range_of_t (t : ℝ) : (t ≤ -2 ∨ t ≥ 3) ↔ ∃ x : ℝ, -3 ≤ x ∧ x ≤ 5 ∧ f x ≤ t^2 - t :=
by sorry

end NUMINAMATH_GPT_minimum_value_of_f_range_of_t_l1935_193586


namespace NUMINAMATH_GPT_find_function_expression_l1935_193503

noncomputable def f (a b x : ℝ) : ℝ := 2 ^ (a * x + b)

theorem find_function_expression
  (a b : ℝ)
  (h1 : f a b 1 = 2)
  (h2 : ∃ g : ℝ → ℝ, (∀ x y : ℝ, f (-a) (-b) x = y ↔ f a b y = x) ∧ g (f a b 1) = 1) :
  ∃ (a b : ℝ), f a b x = 2 ^ (-x + 2) :=
by
  sorry

end NUMINAMATH_GPT_find_function_expression_l1935_193503


namespace NUMINAMATH_GPT_surface_area_change_l1935_193542

noncomputable def original_surface_area (l w h : ℝ) : ℝ :=
  2 * (l * w + l * h + w * h)

noncomputable def new_surface_area (l w h c : ℝ) : ℝ :=
  original_surface_area l w h - 
  (3 * (c * c)) + 
  (2 * c * c)

theorem surface_area_change (l w h c : ℝ) (hl : l = 5) (hw : w = 4) (hh : h = 3) (hc : c = 2) :
  new_surface_area l w h c = original_surface_area l w h - 8 :=
by 
  sorry

end NUMINAMATH_GPT_surface_area_change_l1935_193542


namespace NUMINAMATH_GPT_line_equation_l1935_193563

theorem line_equation (x y : ℝ) : 
  ((y = 1 → x = 2) ∧ ((x,y) = (1,1) ∨ (x,y) = (3,5)))
  → (2 * x - y - 3 = 0) ∨ (x = 2) :=
sorry

end NUMINAMATH_GPT_line_equation_l1935_193563


namespace NUMINAMATH_GPT_avg_one_fourth_class_l1935_193585

variable (N : ℕ) -- Total number of students

-- Define the average grade for the entire class
def avg_entire_class : ℝ := 84

-- Define the average grade of three fourths of the class
def avg_three_fourths_class : ℝ := 80

-- Statement to prove
theorem avg_one_fourth_class (A : ℝ) (h1 : 1/4 * A + 3/4 * avg_three_fourths_class = avg_entire_class) : 
  A = 96 := 
sorry

end NUMINAMATH_GPT_avg_one_fourth_class_l1935_193585


namespace NUMINAMATH_GPT_flagpole_height_in_inches_l1935_193581

theorem flagpole_height_in_inches
  (height_lamppost shadow_lamppost : ℚ)
  (height_flagpole shadow_flagpole : ℚ)
  (h₁ : height_lamppost = 50)
  (h₂ : shadow_lamppost = 12)
  (h₃ : shadow_flagpole = 18 / 12) :
  height_flagpole * 12 = 75 :=
by
  -- Note: To keep the theorem concise, proof steps are omitted
  sorry

end NUMINAMATH_GPT_flagpole_height_in_inches_l1935_193581


namespace NUMINAMATH_GPT_minewaska_state_park_l1935_193572

variable (B H : Nat)

theorem minewaska_state_park (hikers_bike_riders_sum : H + B = 676) (hikers_more_than_bike_riders : H = B + 178) : H = 427 :=
sorry

end NUMINAMATH_GPT_minewaska_state_park_l1935_193572


namespace NUMINAMATH_GPT_polygon_interior_angle_l1935_193512

theorem polygon_interior_angle (n : ℕ) (h1 : ∀ (i : ℕ), i < n → (n - 2) * 180 / n = 140): n = 9 := 
sorry

end NUMINAMATH_GPT_polygon_interior_angle_l1935_193512


namespace NUMINAMATH_GPT_expected_profit_l1935_193591

namespace DailyLottery

/-- Definitions for the problem -/

def ticket_cost : ℝ := 2
def first_prize : ℝ := 100
def second_prize : ℝ := 10
def prob_first_prize : ℝ := 0.001
def prob_second_prize : ℝ := 0.1
def prob_no_prize : ℝ := 1 - prob_first_prize - prob_second_prize

/-- Expected profit calculation as a theorem -/

theorem expected_profit :
  (first_prize * prob_first_prize + second_prize * prob_second_prize + 0 * prob_no_prize) - ticket_cost = -0.9 :=
by
  sorry

end DailyLottery

end NUMINAMATH_GPT_expected_profit_l1935_193591


namespace NUMINAMATH_GPT_florida_north_dakota_license_plate_difference_l1935_193510

theorem florida_north_dakota_license_plate_difference :
  let florida_license_plates := 26^3 * 10^3
  let north_dakota_license_plates := 26^3 * 10^3
  florida_license_plates = north_dakota_license_plates :=
by
  let florida_license_plates := 26^3 * 10^3
  let north_dakota_license_plates := 26^3 * 10^3
  show florida_license_plates = north_dakota_license_plates
  sorry

end NUMINAMATH_GPT_florida_north_dakota_license_plate_difference_l1935_193510


namespace NUMINAMATH_GPT_polar_coordinates_of_point_l1935_193540

open Real

theorem polar_coordinates_of_point :
  ∃ r θ : ℝ, r = 4 ∧ θ = 5 * π / 3 ∧ r > 0 ∧ 0 ≤ θ ∧ θ < 2 * π ∧ 
           (∃ x y : ℝ, x = 2 ∧ y = -2 * sqrt 3 ∧ x = r * cos θ ∧ y = r * sin θ) :=
sorry

end NUMINAMATH_GPT_polar_coordinates_of_point_l1935_193540


namespace NUMINAMATH_GPT_upstream_distance_is_48_l1935_193584

variables (distance_downstream time_downstream time_upstream speed_stream : ℝ)
variables (speed_boat distance_upstream : ℝ)

-- Given conditions
axiom h1 : distance_downstream = 84
axiom h2 : time_downstream = 2
axiom h3 : time_upstream = 2
axiom h4 : speed_stream = 9

-- Define the effective speeds
def speed_downstream (speed_boat speed_stream : ℝ) := speed_boat + speed_stream
def speed_upstream (speed_boat speed_stream : ℝ) := speed_boat - speed_stream

-- Equations based on travel times and distances
axiom eq1 : distance_downstream = (speed_downstream speed_boat speed_stream) * time_downstream
axiom eq2 : distance_upstream = (speed_upstream speed_boat speed_stream) * time_upstream

-- Theorem to prove the distance rowed upstream is 48 km
theorem upstream_distance_is_48 :
  distance_upstream = 48 :=
by
  sorry

end NUMINAMATH_GPT_upstream_distance_is_48_l1935_193584


namespace NUMINAMATH_GPT_cost_of_black_and_white_drawing_l1935_193573

-- Given the cost of the color drawing is 1.5 times the cost of the black and white drawing
-- and John paid $240 for the color drawing, we need to prove the cost of the black and white drawing is $160.

theorem cost_of_black_and_white_drawing (C : ℝ) (h : 1.5 * C = 240) : C = 160 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_black_and_white_drawing_l1935_193573


namespace NUMINAMATH_GPT_number_of_roses_cut_l1935_193559

-- Let's define the initial and final conditions
def initial_roses : ℕ := 6
def final_roses : ℕ := 16

-- Define the number of roses Mary cut from her garden
def roses_cut := final_roses - initial_roses

-- Now, we state the theorem we aim to prove
theorem number_of_roses_cut : roses_cut = 10 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_number_of_roses_cut_l1935_193559


namespace NUMINAMATH_GPT_prob_3_tails_in_8_flips_l1935_193558

def unfair_coin_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k) * (p^k) * ((1 - p)^(n - k))

def probability_of_3_tails : ℚ :=
  unfair_coin_probability 8 3 (2/3)

theorem prob_3_tails_in_8_flips :
  probability_of_3_tails = 448 / 6561 :=
by
  sorry

end NUMINAMATH_GPT_prob_3_tails_in_8_flips_l1935_193558


namespace NUMINAMATH_GPT_sequence_formula_l1935_193515

theorem sequence_formula (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : ∀ n > 1, a n - a (n - 1) = 2^(n-1)) : a n = 2^n - 1 := 
sorry

end NUMINAMATH_GPT_sequence_formula_l1935_193515


namespace NUMINAMATH_GPT_range_of_m_l1935_193598

theorem range_of_m (a m x : ℝ) (p q : Prop) :
  (p ↔ ∃ (a : ℝ) (m : ℝ), ∀ (x : ℝ), 4 * x^2 - 2 * a * x + 2 * a + 5 = 0) →
  (q ↔ 1 - m ≤ x ∧ x ≤ 1 + m ∧ m > 0) →
  (¬ p → ¬ q) →
  (∀ a, -2 ≤ a ∧ a ≤ 10) →
  (1 - m ≤ -2) ∧ (1 + m ≥ 10) →
  m ≥ 9 :=
by
  intros hp hq npnq ha hm
  sorry  -- Proof omitted

end NUMINAMATH_GPT_range_of_m_l1935_193598


namespace NUMINAMATH_GPT_abs_diff_squares_l1935_193524

theorem abs_diff_squares (a b : ℝ) (ha : a = 105) (hb : b = 103) : |a^2 - b^2| = 416 :=
by
  sorry

end NUMINAMATH_GPT_abs_diff_squares_l1935_193524


namespace NUMINAMATH_GPT_noRepeatedDigitsFourDigit_noRepeatedDigitsFiveDigitDiv5_noRepeatedDigitsFourDigitGreaterThan1325_l1935_193565

-- Problem 1: Four-digit numbers with no repeated digits
theorem noRepeatedDigitsFourDigit :
  ∃ (n : ℕ), (n = 120) := sorry

-- Problem 2: Five-digit numbers with no repeated digits and divisible by 5
theorem noRepeatedDigitsFiveDigitDiv5 :
  ∃ (n : ℕ), (n = 216) := sorry

-- Problem 3: Four-digit numbers with no repeated digits and greater than 1325
theorem noRepeatedDigitsFourDigitGreaterThan1325 :
  ∃ (n : ℕ), (n = 181) := sorry

end NUMINAMATH_GPT_noRepeatedDigitsFourDigit_noRepeatedDigitsFiveDigitDiv5_noRepeatedDigitsFourDigitGreaterThan1325_l1935_193565


namespace NUMINAMATH_GPT_students_in_line_l1935_193556

theorem students_in_line (n : ℕ) (h : 1 ≤ n ∧ n ≤ 130) : 
  n = 3 ∨ n = 43 ∨ n = 129 :=
by
  sorry

end NUMINAMATH_GPT_students_in_line_l1935_193556


namespace NUMINAMATH_GPT_exists_monomial_l1935_193521

variables (x y : ℕ) -- Define x and y as natural numbers

theorem exists_monomial :
  ∃ (c : ℕ) (e_x e_y : ℕ), c = 3 ∧ e_x + e_y = 3 ∧ (c * x ^ e_x * y ^ e_y) = (3 * x ^ e_x * y ^ e_y) :=
by
  sorry

end NUMINAMATH_GPT_exists_monomial_l1935_193521


namespace NUMINAMATH_GPT_common_difference_range_l1935_193567

theorem common_difference_range (a : ℕ → ℝ) (d : ℝ) (h : a 3 = 2) (h_pos : ∀ n, a n > 0) (h_arith : ∀ n, a (n + 1) = a n + d) : 0 ≤ d ∧ d < 1 :=
by
  sorry

end NUMINAMATH_GPT_common_difference_range_l1935_193567


namespace NUMINAMATH_GPT_remainder_mod_17_zero_l1935_193523

theorem remainder_mod_17_zero :
  let x1 := 2002 + 3
  let x2 := 2003 + 3
  let x3 := 2004 + 3
  let x4 := 2005 + 3
  let x5 := 2006 + 3
  let x6 := 2007 + 3
  ( (x1 % 17) * (x2 % 17) * (x3 % 17) * (x4 % 17) * (x5 % 17) * (x6 % 17) ) % 17 = 0 :=
by
  let x1 := 2002 + 3
  let x2 := 2003 + 3
  let x3 := 2004 + 3
  let x4 := 2005 + 3
  let x5 := 2006 + 3
  let x6 := 2007 + 3
  sorry

end NUMINAMATH_GPT_remainder_mod_17_zero_l1935_193523


namespace NUMINAMATH_GPT_smallest_n_positive_odd_integer_l1935_193582

theorem smallest_n_positive_odd_integer (n : ℕ) (h1 : n % 2 = 1) (h2 : 3 ^ ((n + 1)^2 / 5) > 500) : n = 6 := sorry

end NUMINAMATH_GPT_smallest_n_positive_odd_integer_l1935_193582


namespace NUMINAMATH_GPT_parallel_lines_m_condition_l1935_193580

theorem parallel_lines_m_condition (m : ℝ) : 
  (∀ (x y : ℝ), (2 * x - m * y - 1 = 0) ↔ ((m - 1) * x - y + 1 = 0)) → m = 2 :=
by
  sorry

end NUMINAMATH_GPT_parallel_lines_m_condition_l1935_193580


namespace NUMINAMATH_GPT_function_property_l1935_193570

theorem function_property 
  (f : ℝ → ℝ) 
  (hf : ∀ x, x ≠ 0 → f (1 - 2 * x) = (1 - x^2) / (x^2)) 
  : 
  (f (1 / 2) = 15) ∧
  (∀ x, x ≠ 1 → f (x) = 4 / (x - 1)^2 - 1) ∧
  (∀ x, x ≠ 0 → x ≠ 1 → f (1 / x) = 4 * x^2 / (x - 1)^2 - 1) :=
by {
  sorry
}

end NUMINAMATH_GPT_function_property_l1935_193570


namespace NUMINAMATH_GPT_sequence_bound_l1935_193525

theorem sequence_bound (a b c : ℕ → ℝ) :
  (a 0 = 1) ∧ (b 0 = 0) ∧ (c 0 = 0) ∧
  (∀ n, n ≥ 1 → a n = a (n-1) + c (n-1) / n) ∧
  (∀ n, n ≥ 1 → b n = b (n-1) + a (n-1) / n) ∧
  (∀ n, n ≥ 1 → c n = c (n-1) + b (n-1) / n) →
  ∀ n, n ≥ 1 → |a n - (n + 1) / 3| < 2 / Real.sqrt (3 * n) :=
by sorry

end NUMINAMATH_GPT_sequence_bound_l1935_193525


namespace NUMINAMATH_GPT_probability_of_odd_numbers_l1935_193561

def binomial (n k : ℕ) : ℕ :=
  Nat.choose n k

def probability_of_5_odd_numbers_in_7_rolls : ℚ :=
  (binomial 7 5) * (1 / 2)^5 * (1 / 2)^(7-5)

theorem probability_of_odd_numbers :
  probability_of_5_odd_numbers_in_7_rolls = 21 / 128 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_odd_numbers_l1935_193561


namespace NUMINAMATH_GPT_ratio_of_c_and_d_l1935_193547

theorem ratio_of_c_and_d 
  (x y c d : ℝ)
  (h₁ : 4 * x - 2 * y = c)
  (h₂ : 6 * y - 12 * x = d) 
  (h₃ : d ≠ 0) : 
  c / d = -1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_c_and_d_l1935_193547


namespace NUMINAMATH_GPT_natasha_time_to_top_l1935_193519

theorem natasha_time_to_top (T : ℝ) 
  (descent_time : ℝ) 
  (whole_journey_avg_speed : ℝ) 
  (climbing_speed : ℝ) 
  (desc_time_condition : descent_time = 2) 
  (whole_journey_avg_speed_condition : whole_journey_avg_speed = 3.5) 
  (climbing_speed_condition : climbing_speed = 2.625) 
  (distance_to_top : ℝ := climbing_speed * T) 
  (avg_speed_condition : whole_journey_avg_speed = 2 * distance_to_top / (T + descent_time)) :
  T = 4 := by
  sorry

end NUMINAMATH_GPT_natasha_time_to_top_l1935_193519


namespace NUMINAMATH_GPT_hot_dogs_sold_next_innings_l1935_193535

-- Defining the conditions
variables (total_initial hot_dogs_sold_first_innings hot_dogs_left : ℕ)

-- Given conditions that need to hold true
axiom initial_count : total_initial = 91
axiom first_innings_sold : hot_dogs_sold_first_innings = 19
axiom remaining_hot_dogs : hot_dogs_left = 45

-- Prove the number of hot dogs sold during the next three innings is 27
theorem hot_dogs_sold_next_innings : total_initial - (hot_dogs_sold_first_innings + hot_dogs_left) = 27 :=
by
  sorry

end NUMINAMATH_GPT_hot_dogs_sold_next_innings_l1935_193535


namespace NUMINAMATH_GPT_train_second_speed_20_l1935_193502

variable (x v: ℕ)

theorem train_second_speed_20 
  (h1 : (x / 40) + (2 * x / v) = (6 * x / 48)) : 
  v = 20 := by 
  sorry

end NUMINAMATH_GPT_train_second_speed_20_l1935_193502


namespace NUMINAMATH_GPT_pos_real_unique_solution_l1935_193548

theorem pos_real_unique_solution (x : ℝ) (hx_pos : 0 < x) (h : (x - 3) / 8 = 5 / (x - 8)) : x = 16 :=
sorry

end NUMINAMATH_GPT_pos_real_unique_solution_l1935_193548


namespace NUMINAMATH_GPT_david_more_pushups_than_zachary_l1935_193511

-- Definitions based on conditions
def david_pushups : ℕ := 37
def zachary_pushups : ℕ := 7

-- Theorem statement proving the answer
theorem david_more_pushups_than_zachary : david_pushups - zachary_pushups = 30 := by
  sorry

end NUMINAMATH_GPT_david_more_pushups_than_zachary_l1935_193511


namespace NUMINAMATH_GPT_no_digit_c_make_2C4_multiple_of_5_l1935_193578

theorem no_digit_c_make_2C4_multiple_of_5 : ∀ C, ¬ (C ≥ 0 ∧ C ≤ 9 ∧ (20 * 10 + C * 10 + 4) % 5 = 0) :=
by intros C hC; sorry

end NUMINAMATH_GPT_no_digit_c_make_2C4_multiple_of_5_l1935_193578


namespace NUMINAMATH_GPT_total_digits_written_total_digit_1_appearances_digit_at_position_2016_l1935_193538

-- Problem 1
theorem total_digits_written : 
  let digits_1_to_9 := 9
  let digits_10_to_99 := 90 * 2
  let digits_100_to_999 := 900 * 3
  digits_1_to_9 + digits_10_to_99 + digits_100_to_999 = 2889 := 
by
  sorry

-- Problem 2
theorem total_digit_1_appearances : 
  let digit_1_as_1_digit := 1
  let digit_1_as_2_digits := 10 + 9
  let digit_1_as_3_digits := 100 + 9 * 10 + 9 * 10
  digit_1_as_1_digit + digit_1_as_2_digits + digit_1_as_3_digits = 300 := 
by
  sorry

-- Problem 3
theorem digit_at_position_2016 : 
  let position_1_to_99 := 9 + 90 * 2
  let remaining_positions := 2016 - position_1_to_99
  let three_digit_positions := remaining_positions / 3
  let specific_number := 100 + three_digit_positions - 1
  specific_number % 10 = 8 := 
by
  sorry

end NUMINAMATH_GPT_total_digits_written_total_digit_1_appearances_digit_at_position_2016_l1935_193538


namespace NUMINAMATH_GPT_box_height_l1935_193544

variables (length width : ℕ) (cube_volume cubes total_volume : ℕ)
variable (height : ℕ)

theorem box_height :
  length = 12 →
  width = 16 →
  cube_volume = 3 →
  cubes = 384 →
  total_volume = cubes * cube_volume →
  total_volume = length * width * height →
  height = 6 :=
by
  intros
  sorry

end NUMINAMATH_GPT_box_height_l1935_193544


namespace NUMINAMATH_GPT_solve_inequalities_l1935_193575

theorem solve_inequalities (x : ℝ) (h₁ : 5 * x - 8 > 12 - 2 * x) (h₂ : |x - 1| ≤ 3) : 
  (20 / 7) < x ∧ x ≤ 4 :=
by
  sorry

end NUMINAMATH_GPT_solve_inequalities_l1935_193575


namespace NUMINAMATH_GPT_calculate_remaining_area_l1935_193557

/-- In a rectangular plot of land ABCD, where AB = 20 meters and BC = 12 meters, 
    a triangular garden ABE is installed where AE = 15 meters and BE intersects AE at a perpendicular angle, 
    the area of the remaining part of the land which is not occupied by the garden is 150 square meters. -/
theorem calculate_remaining_area 
  (AB BC AE : ℝ) 
  (hAB : AB = 20) 
  (hBC : BC = 12) 
  (hAE : AE = 15)
  (h_perpendicular : true) : -- BE ⊥ AE implying right triangle ABE
  ∃ area_remaining : ℝ, area_remaining = 150 :=
by
  sorry

end NUMINAMATH_GPT_calculate_remaining_area_l1935_193557


namespace NUMINAMATH_GPT_triangle_inequality_l1935_193514

variable (R r e f : ℝ)

theorem triangle_inequality (h1 : ∃ (A B C : ℝ × ℝ), true)
                            (h2 : true) :
  R^2 - e^2 ≥ 4 * (r^2 - f^2) :=
by sorry

end NUMINAMATH_GPT_triangle_inequality_l1935_193514


namespace NUMINAMATH_GPT_positive_root_condition_negative_root_condition_zero_root_condition_l1935_193555

variable (a b c : ℝ)

-- Condition for a positive root
theorem positive_root_condition : 
  ((a > 0 ∧ b > c) ∨ (a < 0 ∧ b < c)) ↔ (∃ x : ℝ, x > 0 ∧ a * x = b - c) :=
sorry

-- Condition for a negative root
theorem negative_root_condition : 
  ((a > 0 ∧ b < c) ∨ (a < 0 ∧ b > c)) ↔ (∃ x : ℝ, x < 0 ∧ a * x = b - c) :=
sorry

-- Condition for a root equal to zero
theorem zero_root_condition : 
  (a ≠ 0 ∧ b = c) ↔ (∃ x : ℝ, x = 0 ∧ a * x = b - c) :=
sorry

end NUMINAMATH_GPT_positive_root_condition_negative_root_condition_zero_root_condition_l1935_193555


namespace NUMINAMATH_GPT_trajectory_of_midpoint_l1935_193516

theorem trajectory_of_midpoint (Q : ℝ × ℝ) (P : ℝ × ℝ) (N : ℝ × ℝ)
  (h1 : Q.1^2 - Q.2^2 = 1)
  (h2 : N = (2 * P.1 - Q.1, 2 * P.2 - Q.2))
  (h3 : N.1 + N.2 = 2)
  (h4 : (P.2 - Q.2) / (P.1 - Q.1) = 1) :
  2 * P.1^2 - 2 * P.2^2 - 2 * P.1 + 2 * P.2 - 1 = 0 :=
  sorry

end NUMINAMATH_GPT_trajectory_of_midpoint_l1935_193516


namespace NUMINAMATH_GPT_reema_loan_period_l1935_193529

theorem reema_loan_period (P SI : ℕ) (R : ℚ) (h1 : P = 1200) (h2 : SI = 432) (h3 : R = 6) : 
  ∃ T : ℕ, SI = (P * R * T) / 100 ∧ T = 6 :=
by
  sorry

end NUMINAMATH_GPT_reema_loan_period_l1935_193529


namespace NUMINAMATH_GPT_largest_odd_integer_satisfying_inequality_l1935_193596

theorem largest_odd_integer_satisfying_inequality : 
  ∃ (x : ℤ), (x % 2 = 1) ∧ (1 / 4 < x / 6) ∧ (x / 6 < 7 / 9) ∧ (∀ y : ℤ, (y % 2 = 1) ∧ (1 / 4 < y / 6) ∧ (y / 6 < 7 / 9) → y ≤ x) :=
sorry

end NUMINAMATH_GPT_largest_odd_integer_satisfying_inequality_l1935_193596


namespace NUMINAMATH_GPT_greendale_high_school_points_l1935_193550

-- Define the conditions
def roosevelt_points_first_game : ℕ := 30
def roosevelt_points_second_game : ℕ := roosevelt_points_first_game / 2
def roosevelt_points_third_game : ℕ := roosevelt_points_second_game * 3
def roosevelt_total_points : ℕ := roosevelt_points_first_game + roosevelt_points_second_game + roosevelt_points_third_game
def roosevelt_total_with_bonus : ℕ := roosevelt_total_points + 50
def difference : ℕ := 10

-- Define the assertion for Greendale's points
def greendale_points : ℕ := roosevelt_total_with_bonus - difference

-- The theorem to be proved
theorem greendale_high_school_points : greendale_points = 130 :=
by
  -- Proof should be here, but we add sorry to skip it as per the instructions
  sorry

end NUMINAMATH_GPT_greendale_high_school_points_l1935_193550


namespace NUMINAMATH_GPT_real_part_is_neg4_l1935_193597

def real_part_of_z (z : ℂ) : ℝ :=
  z.re

theorem real_part_is_neg4 (i : ℂ) (h : i^2 = -1) :
  real_part_of_z ((3 + 4 * i) * i) = -4 := by
  sorry

end NUMINAMATH_GPT_real_part_is_neg4_l1935_193597
