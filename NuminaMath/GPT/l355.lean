import Mathlib

namespace second_player_wins_l355_35519

-- Defining the chess board and initial positions of the rooks
inductive Square : Type
| a1 | a2 | a3 | a4 | a5 | a6 | a7 | a8
| b1 | b2 | b3 | b4 | b5 | b6 | b7 | b8
| c1 | c2 | c3 | c4 | c5 | c6 | c7 | c8
| d1 | d2 | d3 | d4 | d5 | d6 | d7 | d8
| e1 | e2 | e3 | e4 | e5 | e6 | e7 | e8
| f1 | f2 | f3 | f4 | f5 | f6 | f7 | f8
| g1 | g2 | g3 | g4 | g5 | g6 | g7 | g8
| h1 | h2 | h3 | h4 | h5 | h6 | h7 | h8
deriving DecidableEq

-- Define the initial positions of the rooks
def initial_white_rook_position : Square := Square.b2
def initial_black_rook_position : Square := Square.c4

-- Define the rules of movement: a rook can move horizontally or vertically unless blocked
def rook_can_move (start finish : Square) : Prop :=
  -- Only horizontal or vertical moves allowed
  sorry

-- Define conditions for a square being attacked by a rook at a given position
def is_attacked_by_rook (position target : Square) : Prop :=
  sorry

-- Define the condition for a player to be in a winning position if no moves are illegal
def player_can_win (white_position black_position : Square) : Prop :=
  sorry

-- The main theorem: Second player (black rook) can ensure a win
theorem second_player_wins : player_can_win initial_white_rook_position initial_black_rook_position :=
  sorry

end second_player_wins_l355_35519


namespace optimal_route_l355_35507

-- Define the probabilities of no traffic jam on each road segment.
def P_AC : ℚ := 9 / 10
def P_CD : ℚ := 14 / 15
def P_DB : ℚ := 5 / 6
def P_CF : ℚ := 9 / 10
def P_FB : ℚ := 15 / 16
def P_AE : ℚ := 9 / 10
def P_EF : ℚ := 9 / 10
def P_FB2 : ℚ := 19 / 20  -- Alias for repeated probability

-- Define the probability of encountering a traffic jam on a route
def prob_traffic_jam (p_no_jam : ℚ) : ℚ := 1 - p_no_jam

-- Define the probabilities of encountering a traffic jam along each route.
def P_ACDB_jam : ℚ := prob_traffic_jam (P_AC * P_CD * P_DB)
def P_ACFB_jam : ℚ := prob_traffic_jam (P_AC * P_CF * P_FB)
def P_AEFB_jam : ℚ := prob_traffic_jam (P_AE * P_EF * P_FB2)

-- State the theorem to prove the optimal route
theorem optimal_route : P_ACDB_jam < P_ACFB_jam ∧ P_ACDB_jam < P_AEFB_jam :=
by { sorry }

end optimal_route_l355_35507


namespace jerry_total_logs_l355_35554

def logs_from_trees (p m w : Nat) : Nat :=
  80 * p + 60 * m + 100 * w

theorem jerry_total_logs :
  logs_from_trees 8 3 4 = 1220 :=
by
  -- Proof here
  sorry

end jerry_total_logs_l355_35554


namespace friends_count_l355_35533

theorem friends_count (n : ℕ) (average_rent : ℝ) (new_average_rent : ℝ) (original_rent : ℝ) (increase_percent : ℝ)
  (H1 : average_rent = 800)
  (H2 : new_average_rent = 870)
  (H3 : original_rent = 1400)
  (H4 : increase_percent = 0.20) :
  n = 4 :=
by
  -- Define the initial total rent
  let initial_total_rent := n * average_rent
  -- Define the increased rent for one person
  let increased_rent := original_rent * (1 + increase_percent)
  -- Define the new total rent
  let new_total_rent := initial_total_rent - original_rent + increased_rent
  -- Set up the new average rent equation
  have rent_equation := new_total_rent = n * new_average_rent
  sorry

end friends_count_l355_35533


namespace ben_min_sales_l355_35516

theorem ben_min_sales 
    (old_salary : ℕ := 75000) 
    (new_base_salary : ℕ := 45000) 
    (commission_rate : ℚ := 0.15) 
    (sale_amount : ℕ := 750) : 
    ∃ (n : ℕ), n ≥ 267 ∧ (old_salary ≤ new_base_salary + n * ⌊commission_rate * sale_amount⌋) :=
by 
  sorry

end ben_min_sales_l355_35516


namespace chef_leftover_potatoes_l355_35501

-- Defining the conditions as variables
def fries_per_potato := 25
def total_potatoes := 15
def fries_needed := 200

-- Calculating the number of potatoes needed.
def potatoes_needed : ℕ :=
  fries_needed / fries_per_potato

-- Calculating the leftover potatoes.
def leftovers : ℕ :=
  total_potatoes - potatoes_needed

-- The theorem statement
theorem chef_leftover_potatoes :
  leftovers = 7 :=
by
  -- the actual proof is omitted.
  sorry

end chef_leftover_potatoes_l355_35501


namespace evaluate_expression_l355_35590

theorem evaluate_expression : 
  ∀ (x y : ℕ), x = 3 → y = 2 → (5 * x^(y + 1) + 6 * y^(x + 1) = 231) := by 
  intros x y hx hy
  rw [hx, hy]
  sorry

end evaluate_expression_l355_35590


namespace dog_weights_l355_35540

structure DogWeightProgression where
  initial: ℕ   -- initial weight in pounds
  week_9: ℕ    -- weight at 9 weeks in pounds
  month_3: ℕ  -- weight at 3 months in pounds
  month_5: ℕ  -- weight at 5 months in pounds
  year_1: ℕ   -- weight at 1 year in pounds

theorem dog_weights :
  ∃ (golden_retriever labrador poodle : DogWeightProgression),
  golden_retriever.initial = 6 ∧
  golden_retriever.week_9 = 12 ∧
  golden_retriever.month_3 = 24 ∧
  golden_retriever.month_5 = 48 ∧
  golden_retriever.year_1 = 78 ∧
  labrador.initial = 8 ∧
  labrador.week_9 = 24 ∧
  labrador.month_3 = 36 ∧
  labrador.month_5 = 72 ∧
  labrador.year_1 = 102 ∧
  poodle.initial = 4 ∧
  poodle.week_9 = 16 ∧
  poodle.month_3 = 32 ∧
  poodle.month_5 = 32 ∧
  poodle.year_1 = 52 :=
by 
  have golden_retriever : DogWeightProgression := { initial := 6, week_9 := 12, month_3 := 24, month_5 := 48, year_1 := 78 }
  have labrador : DogWeightProgression := { initial := 8, week_9 := 24, month_3 := 36, month_5 := 72, year_1 := 102 }
  have poodle : DogWeightProgression := { initial := 4, week_9 := 16, month_3 := 32, month_5 := 32, year_1 := 52 }
  use golden_retriever, labrador, poodle
  repeat { split };
  { sorry }

end dog_weights_l355_35540


namespace intersection_P_Q_l355_35551

-- Define set P
def P : Set ℝ := {1, 2, 3, 4}

-- Define set Q
def Q : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

-- Define the problem statement as a theorem
theorem intersection_P_Q : P ∩ Q = {1, 2} :=
by
  sorry

end intersection_P_Q_l355_35551


namespace least_rice_l355_35596

variable (o r : ℝ)

-- Conditions
def condition_1 : Prop := o ≥ 8 + r / 2
def condition_2 : Prop := o ≤ 3 * r

-- The main theorem we want to prove
theorem least_rice (h1 : condition_1 o r) (h2 : condition_2 o r) : r ≥ 4 :=
sorry

end least_rice_l355_35596


namespace alice_flips_heads_probability_l355_35535

def prob_heads (n k : ℕ) (p q : ℚ) : ℚ :=
  (Nat.choose n k) * (p ^ k) * (q ^ (n - k))

theorem alice_flips_heads_probability :
  prob_heads 8 3 (1/3 : ℚ) (2/3 : ℚ) = 1792 / 6561 :=
by
  sorry

end alice_flips_heads_probability_l355_35535


namespace necessary_but_not_sufficient_l355_35568

-- Define α as an interior angle of triangle ABC
def is_interior_angle_of_triangle (α : ℝ) : Prop :=
  0 < α ∧ α < 180

-- Define the sine condition
def sine_condition (α : ℝ) : Prop :=
  Real.sin α = Real.sqrt 2 / 2

-- Define the main theorem
theorem necessary_but_not_sufficient (α : ℝ) (h1 : is_interior_angle_of_triangle α) (h2 : sine_condition α) :
  (sine_condition α) ↔ (α = 45) ∨ (α = 135) := by
  sorry

end necessary_but_not_sufficient_l355_35568


namespace value_of_adams_collection_l355_35567

theorem value_of_adams_collection (num_coins : ℕ) (coins_value : ℕ) (total_value_4coins : ℕ) (h1 : num_coins = 20) (h2 : total_value_4coins = 16) (h3 : ∀ k, k = 4 → coins_value = total_value_4coins / k) : 
  num_coins * coins_value = 80 := 
by {
  sorry
}

end value_of_adams_collection_l355_35567


namespace value_of_f_log_20_l355_35555

variable (f : ℝ → ℝ)
variable (h₁ : ∀ x : ℝ, f (-x) = -f x)
variable (h₂ : ∀ x : ℝ, f (x - 2) = f (x + 2))
variable (h₃ : ∀ x : ℝ, x > -1 ∧ x < 0 → f x = 2^x + 1/5)

theorem value_of_f_log_20 : f (Real.log 20 / Real.log 2) = -1 := sorry

end value_of_f_log_20_l355_35555


namespace min_value_expression_l355_35592

theorem min_value_expression (x y : ℝ) (hx : 0 ≤ x ∧ x ≤ 3) (hy : 1 ≤ y ∧ y ≤ 4) : 
  ∃ z, z = (x + y) / x ∧ z = 4 / 3 := by
  sorry

end min_value_expression_l355_35592


namespace circle_area_from_intersection_l355_35537

-- Statement of the problem
theorem circle_area_from_intersection (r : ℝ) (A B : ℝ × ℝ)
  (h_circle : ∀ x y, (x + 2) ^ 2 + y ^ 2 = r ^ 2 ↔ (x, y) = A ∨ (x, y) = B)
  (h_parabola : ∀ x y, y ^ 2 = 20 * x ↔ (x, y) = A ∨ (x, y) = B)
  (h_axis_sym : A.1 = -5 ∧ B.1 = -5)
  (h_AB_dist : |A.2 - B.2| = 8) : π * r ^ 2 = 25 * π :=
by
  sorry

end circle_area_from_intersection_l355_35537


namespace need_to_work_24_hours_per_week_l355_35506

-- Definitions
def original_hours_per_week := 20
def total_weeks := 12
def target_income := 3000

def missed_weeks := 2
def remaining_weeks := total_weeks - missed_weeks

-- Calculation
def new_hours_per_week := (original_hours_per_week * total_weeks) / remaining_weeks

-- Statement of the theorem
theorem need_to_work_24_hours_per_week : new_hours_per_week = 24 := 
by 
  -- Adding sorry to skip the proof, focusing on the statement.
  sorry

end need_to_work_24_hours_per_week_l355_35506


namespace probability_X_Y_Z_problems_l355_35582

-- Define the success probabilities for Problem A
def P_X_A : ℚ := 1 / 5
def P_Y_A : ℚ := 1 / 2

-- Define the success probabilities for Problem B
def P_Y_B : ℚ := 3 / 5

-- Define the negation of success probabilities for Problem C
def P_Y_not_C : ℚ := 5 / 8
def P_X_not_C : ℚ := 3 / 4
def P_Z_not_C : ℚ := 7 / 16

-- State the final probability theorem
theorem probability_X_Y_Z_problems :
  P_X_A * P_Y_A * P_Y_B * P_Y_not_C * P_X_not_C * P_Z_not_C = 63 / 2048 := 
sorry

end probability_X_Y_Z_problems_l355_35582


namespace prob_diff_colors_correct_l355_35511

def total_chips := 6 + 5 + 4 + 3

def prob_diff_colors : ℚ :=
  (6 / total_chips * (12 / total_chips) +
  5 / total_chips * (13 / total_chips) +
  4 / total_chips * (14 / total_chips) +
  3 / total_chips * (15 / total_chips))

theorem prob_diff_colors_correct :
  prob_diff_colors = 119 / 162 := by
  sorry

end prob_diff_colors_correct_l355_35511


namespace find_b_num_days_worked_l355_35560

noncomputable def a_num_days_worked := 6
noncomputable def b_num_days_worked := 9  -- This is what we want to verify
noncomputable def c_num_days_worked := 4

noncomputable def c_daily_wage := 105
noncomputable def wage_ratio_a := 3
noncomputable def wage_ratio_b := 4
noncomputable def wage_ratio_c := 5

-- Helper to find daily wages for a and b given the ratio and c's wage
noncomputable def x := c_daily_wage / wage_ratio_c
noncomputable def a_daily_wage := wage_ratio_a * x
noncomputable def b_daily_wage := wage_ratio_b * x

-- Calculate total earnings
noncomputable def a_total_earning := a_num_days_worked * a_daily_wage
noncomputable def c_total_earning := c_num_days_worked * c_daily_wage
noncomputable def total_earning := 1554
noncomputable def b_total_earning := b_num_days_worked * b_daily_wage

theorem find_b_num_days_worked : total_earning = a_total_earning + b_total_earning + c_total_earning → b_num_days_worked = 9 := by
  sorry

end find_b_num_days_worked_l355_35560


namespace candy_probability_l355_35566

/-- 
A jar has 15 red candies, 15 blue candies, and 10 green candies. Terry picks three candies at random,
then Mary picks three of the remaining candies at random. Calculate the probability that they get 
the same color combination, irrespective of order, expressed as a fraction $m/n,$ where $m$ and $n$ 
are relatively prime positive integers. Find $m+n.$ -/
theorem candy_probability :
  let num_red := 15
  let num_blue := 15
  let num_green := 10
  let total_candies := num_red + num_blue + num_green
  let Terry_picks := 3
  let Mary_picks := 3
  let prob_equal_comb := (118545 : ℚ) / 2192991
  let m := 118545
  let n := 2192991
  m + n = 2310536 := sorry

end candy_probability_l355_35566


namespace find_m_l355_35528

theorem find_m (m : ℝ) (α : ℝ) (h_cos : Real.cos α = -3/5) (h_p : ((Real.cos α = m / (Real.sqrt (m^2 + 4^2)))) ∧ (Real.cos α < 0) ∧ (m < 0)) :

  m = -3 :=
by 
  sorry

end find_m_l355_35528


namespace relationship_among_m_n_k_l355_35530

theorem relationship_among_m_n_k :
  (¬ ∃ x : ℝ, |2 * x - 3| + m = 0) → 
  (∃! x: ℝ, |3 * x - 4| + n = 0) → 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ |4 * x₁ - 5| + k = 0 ∧ |4 * x₂ - 5| + k = 0) →
  (m > n ∧ n > k) :=
by
  intros h1 h2 h3
  -- Proof part will be added here
  sorry

end relationship_among_m_n_k_l355_35530


namespace row_sum_1005_equals_20092_l355_35565

theorem row_sum_1005_equals_20092 :
  let row := 1005
  let n := row
  let first_element := n
  let num_elements := 2 * n - 1
  let last_element := first_element + (num_elements - 1)
  let sum_row := num_elements * (first_element + last_element) / 2
  sum_row = 20092 :=
by
  sorry

end row_sum_1005_equals_20092_l355_35565


namespace cicely_100th_birthday_l355_35508

-- Definition of the conditions
def birth_year (birthday_year : ℕ) (birthday_age : ℕ) : ℕ :=
  birthday_year - birthday_age

def birthday (birth_year : ℕ) (age : ℕ) : ℕ :=
  birth_year + age

-- The problem restatement in Lean 4
theorem cicely_100th_birthday (birthday_year : ℕ) (birthday_age : ℕ) (expected_year : ℕ) :
  birthday_year = 1939 → birthday_age = 21 → expected_year = 2018 → birthday (birth_year birthday_year birthday_age) 100 = expected_year :=
by
  intros h1 h2 h3
  rw [birthday, birth_year]
  rw [h1, h2]
  sorry

end cicely_100th_birthday_l355_35508


namespace mod_congruent_integers_l355_35595

theorem mod_congruent_integers (n : ℕ) : (∃ k : ℕ, n = 13 * k + 7 ∧ 7 + 13 * k < 2000) ↔ 
  (7 + 13 * 153 < 2000 ∧ ∀ m : ℕ, (0 ≤ m ∧ m ≤ 153) → n = 13 * m + 7) := by
  sorry

end mod_congruent_integers_l355_35595


namespace numbers_combination_to_24_l355_35562

theorem numbers_combination_to_24 :
  (40 / 4) + 12 + 2 = 24 :=
by
  sorry

end numbers_combination_to_24_l355_35562


namespace boat_sinking_weight_range_l355_35500

theorem boat_sinking_weight_range
  (L_min L_max : ℝ)
  (B_min B_max : ℝ)
  (D_min D_max : ℝ)
  (sink_rate : ℝ)
  (down_min down_max : ℝ)
  (min_weight max_weight : ℝ)
  (condition1 : 3 ≤ L_min ∧ L_max ≤ 5)
  (condition2 : 2 ≤ B_min ∧ B_max ≤ 3)
  (condition3 : 1 ≤ D_min ∧ D_max ≤ 2)
  (condition4 : sink_rate = 0.01)
  (condition5 : 0.03 ≤ down_min ∧ down_max ≤ 0.06)
  (condition6 : ∀ D, D_min ≤ D ∧ D ≤ D_max → (D - down_max) ≥ 0.5)
  (condition7 : min_weight = down_min * (10 / 0.01))
  (condition8 : max_weight = down_max * (10 / 0.01)) :
  min_weight = 30 ∧ max_weight = 60 := 
sorry

end boat_sinking_weight_range_l355_35500


namespace greatest_integer_gcd_l355_35584

theorem greatest_integer_gcd (n : ℕ) (h₁ : n < 150) (h₂ : Nat.gcd n 30 = 5) : n ≤ 145 :=
by
  sorry

end greatest_integer_gcd_l355_35584


namespace maximum_possible_median_l355_35581

theorem maximum_possible_median
  (total_cans : ℕ)
  (total_customers : ℕ)
  (min_cans_per_customer : ℕ)
  (alt_min_cans_per_customer : ℕ)
  (exact_min_cans_count : ℕ)
  (atleast_min_cans_count : ℕ)
  (min_cans_customers : ℕ)
  (alt_min_cans_customer: ℕ): 
  (total_cans = 300) → 
  (total_customers = 120) →
  (min_cans_per_customer = 2) →
  (alt_min_cans_per_customer = 4) →
  (min_cans_customers = 59) →
  (alt_min_cans_customer = 61) →
  (min_cans_per_customer * min_cans_customers + alt_min_cans_per_customer * (total_customers - min_cans_customers) = total_cans) →
  max (min_cans_per_customer + 1) (alt_min_cans_per_customer - 1) = 3 :=
sorry

end maximum_possible_median_l355_35581


namespace total_sales_correct_l355_35553

def maries_newspapers : ℝ := 275.0
def maries_magazines : ℝ := 150.0
def total_sales := maries_newspapers + maries_magazines

theorem total_sales_correct :
  total_sales = 425.0 :=
by
  -- Proof omitted
  sorry

end total_sales_correct_l355_35553


namespace transformed_circle_eq_l355_35575

theorem transformed_circle_eq (x y : ℝ) (h : x^2 + y^2 = 1) : x^2 + 9 * (y / 3)^2 = 1 := by
  sorry

end transformed_circle_eq_l355_35575


namespace collections_in_bag_l355_35570

noncomputable def distinct_collections : ℕ :=
  let vowels := ['A', 'I', 'O']
  let consonants := ['M', 'H', 'C', 'N', 'T', 'T']
  let case1 := Nat.choose 3 2 * Nat.choose 6 3 -- when 0 or 1 T falls off
  let case2 := Nat.choose 3 2 * Nat.choose 5 1 -- when both T's fall off
  case1 + case2

theorem collections_in_bag : distinct_collections = 75 := 
  by
  -- proof goes here
  sorry

end collections_in_bag_l355_35570


namespace rotated_angle_540_deg_l355_35543

theorem rotated_angle_540_deg (θ : ℝ) (h : θ = 60) : 
  (θ - 540) % 360 % 180 = 60 :=
by
  sorry

end rotated_angle_540_deg_l355_35543


namespace paper_fold_length_l355_35525

theorem paper_fold_length (length_orig : ℝ) (h : length_orig = 12) : length_orig / 2 = 6 :=
by
  rw [h]
  norm_num

end paper_fold_length_l355_35525


namespace one_positive_zero_l355_35536

noncomputable def f (x a : ℝ) : ℝ := x * Real.exp x - a * x - 1

theorem one_positive_zero (a : ℝ) : ∃ x : ℝ, x > 0 ∧ f x a = 0 :=
sorry

end one_positive_zero_l355_35536


namespace circle_equation_l355_35556

theorem circle_equation (x y : ℝ) : (3 * x - 4 * y + 12 = 0) → (x^2 + 4 * x + y^2 - 3 * y = 0) :=
sorry

end circle_equation_l355_35556


namespace tiles_needed_l355_35514

-- Definitions for the problem
def width_wall : ℕ := 36
def length_wall : ℕ := 72
def width_tile : ℕ := 3
def length_tile : ℕ := 4

-- The area of the wall
def A_wall : ℕ := width_wall * length_wall

-- The area of one tile
def A_tile : ℕ := width_tile * length_tile

-- The number of tiles needed
def number_of_tiles : ℕ := A_wall / A_tile

-- Proof statement
theorem tiles_needed : number_of_tiles = 216 := by
  sorry

end tiles_needed_l355_35514


namespace parabola_perpendicular_bisector_intersects_x_axis_l355_35529

theorem parabola_perpendicular_bisector_intersects_x_axis
  (x1 y1 x2 y2 : ℝ) 
  (A_on_parabola : y1^2 = 2 * x1)
  (B_on_parabola : y2^2 = 2 * x2) 
  (k m : ℝ) 
  (AB_line : ∀ x y, y = k * x + m)
  (k_not_zero : k ≠ 0) 
  (k_m_condition : (1 / k^2) - (m / k) > 0) :
  ∃ x0 : ℝ, x0 = (1 / k^2) - (m / k) + 1 ∧ x0 > 1 :=
by
  sorry

end parabola_perpendicular_bisector_intersects_x_axis_l355_35529


namespace rational_reciprocal_pow_2014_l355_35558

theorem rational_reciprocal_pow_2014 (a : ℚ) (h : a = 1 / a) : a ^ 2014 = 1 := by
  sorry

end rational_reciprocal_pow_2014_l355_35558


namespace sara_remaining_red_balloons_l355_35559

-- Given conditions
def initial_red_balloons := 31
def red_balloons_given := 24

-- Statement to prove
theorem sara_remaining_red_balloons : (initial_red_balloons - red_balloons_given = 7) :=
by
  -- Proof can be skipped
  sorry

end sara_remaining_red_balloons_l355_35559


namespace smallest_integer_odd_sequence_l355_35542

/-- Given the median of a set of consecutive odd integers is 157 and the greatest integer in the set is 171,
    prove that the smallest integer in the set is 149. -/
theorem smallest_integer_odd_sequence (median greatest : ℤ) (h_median : median = 157) (h_greatest : greatest = 171) :
  ∃ smallest : ℤ, smallest = 149 :=
by
  sorry

end smallest_integer_odd_sequence_l355_35542


namespace will_total_clothes_l355_35517

theorem will_total_clothes (n1 n2 n3 : ℕ) (h1 : n1 = 32) (h2 : n2 = 9) (h3 : n3 = 3) : n1 + n2 * n3 = 59 := 
by
  sorry

end will_total_clothes_l355_35517


namespace extended_ohara_triple_example_l355_35546

theorem extended_ohara_triple_example : 
  (2 * Real.sqrt 49 + Real.sqrt 64 = 22) :=
by
  -- We are stating the conditions and required proof here.
  sorry

end extended_ohara_triple_example_l355_35546


namespace no_integer_root_l355_35588

theorem no_integer_root (q : ℤ) : ¬ ∃ x : ℤ, x^2 + 7 * x - 14 * (q^2 + 1) = 0 := sorry

end no_integer_root_l355_35588


namespace area_of_given_parallelogram_l355_35531

def parallelogram_base : ℝ := 24
def parallelogram_height : ℝ := 16
def parallelogram_area (b h : ℝ) : ℝ := b * h

theorem area_of_given_parallelogram : parallelogram_area parallelogram_base parallelogram_height = 384 := 
by sorry

end area_of_given_parallelogram_l355_35531


namespace functional_relationship_and_point_l355_35569

noncomputable def directly_proportional (y x : ℝ) : Prop := ∃ k : ℝ, k ≠ 0 ∧ y = k * x

theorem functional_relationship_and_point :
  (∀ x y, directly_proportional y x → y = 2 * x) ∧ 
  (∀ a : ℝ, (∃ (y : ℝ), y = 3 ∧ directly_proportional y a) → a = 3 / 2) :=
by
  sorry

end functional_relationship_and_point_l355_35569


namespace mean_of_y_and_18_is_neg1_l355_35585

theorem mean_of_y_and_18_is_neg1 (y : ℤ) : 
  ((4 + 6 + 10 + 14) / 4) = ((y + 18) / 2) → y = -1 := 
by 
  -- Placeholder for the proof
  sorry

end mean_of_y_and_18_is_neg1_l355_35585


namespace tetrahedron_cut_off_vertices_l355_35510

theorem tetrahedron_cut_off_vertices :
  ∀ (V E : ℕ) (cut_effect : ℕ → ℕ),
    -- Initial conditions
    V = 4 → E = 6 →
    -- Effect of each cut (cutting one vertex introduces 3 new edges)
    (∀ v, v ≤ V → cut_effect v = 3 * v) →
    -- Prove the number of edges in the new figure
    (E + cut_effect V) = 18 :=
by
  intros V E cut_effect hV hE hcut
  sorry

end tetrahedron_cut_off_vertices_l355_35510


namespace expression_simplification_l355_35586

theorem expression_simplification : (4^2 * 7 / (8 * 9^2) * (8 * 9 * 11^2) / (4 * 7 * 11)) = 44 / 9 :=
by
  sorry

end expression_simplification_l355_35586


namespace cows_grazed_by_C_l355_35589

-- Define the initial conditions as constants
def cows_grazed_A : ℕ := 24
def months_grazed_A : ℕ := 3
def cows_grazed_B : ℕ := 10
def months_grazed_B : ℕ := 5
def cows_grazed_D : ℕ := 21
def months_grazed_D : ℕ := 3
def share_rent_A : ℕ := 1440
def total_rent : ℕ := 6500

-- Define the cow-months calculation for A, B, D
def cow_months_A : ℕ := cows_grazed_A * months_grazed_A
def cow_months_B : ℕ := cows_grazed_B * months_grazed_B
def cow_months_D : ℕ := cows_grazed_D * months_grazed_D

-- Let x be the number of cows grazed by C
variable (x : ℕ)

-- Define the cow-months calculation for C
def cow_months_C : ℕ := x * 4

-- Define rent per cow-month
def rent_per_cow_month : ℕ := share_rent_A / cow_months_A

-- Proof problem statement
theorem cows_grazed_by_C : 
  (6500 = (cow_months_A + cow_months_B + cow_months_C x + cow_months_D) * rent_per_cow_month) →
  x = 35 := by
  sorry

end cows_grazed_by_C_l355_35589


namespace average_weight_of_Arun_l355_35538

def arun_opinion (w : ℝ) : Prop := 66 < w ∧ w < 72
def brother_opinion (w : ℝ) : Prop := 60 < w ∧ w < 70
def mother_opinion (w : ℝ) : Prop := w ≤ 69

theorem average_weight_of_Arun :
  (∀ w, arun_opinion w → brother_opinion w → mother_opinion w → 
    (w = 67 ∨ w = 68 ∨ w = 69)) →
  avg_weight = 68 :=
sorry

end average_weight_of_Arun_l355_35538


namespace log_product_eq_one_l355_35598

noncomputable def log_base (b x : ℝ) := Real.log x / Real.log b

theorem log_product_eq_one :
  log_base 5 2 * log_base 4 25 = 1 := 
by
  sorry

end log_product_eq_one_l355_35598


namespace complex_number_solution_l355_35503

theorem complex_number_solution (z i : ℂ) (h : z * (i - i^2) = 1 + i^3) (h1 : i^2 = -1) (h2 : i^3 = -i) (h3 : i^4 = 1) : 
  z = -i := 
by 
  sorry

end complex_number_solution_l355_35503


namespace total_lifespan_l355_35513

theorem total_lifespan (B H F : ℕ)
  (hB : B = 10)
  (hH : H = B - 6)
  (hF : F = 4 * H) :
  B + H + F = 30 := by
  sorry

end total_lifespan_l355_35513


namespace find_chemistry_marks_l355_35577

theorem find_chemistry_marks (marks_english marks_math marks_physics marks_biology : ℤ)
    (average_marks total_subjects : ℤ)
    (h1 : marks_english = 36)
    (h2 : marks_math = 35)
    (h3 : marks_physics = 42)
    (h4 : marks_biology = 55)
    (h5 : average_marks = 45)
    (h6 : total_subjects = 5) :
    (225 - (marks_english + marks_math + marks_physics + marks_biology)) = 57 :=
by
  sorry

end find_chemistry_marks_l355_35577


namespace initial_amount_l355_35572

theorem initial_amount 
  (spend1 spend2 left : ℝ)
  (hspend1 : spend1 = 1.75) 
  (hspend2 : spend2 = 1.25) 
  (hleft : left = 6.00) : 
  spend1 + spend2 + left = 9.00 := 
by
  -- Proof is omitted
  sorry

end initial_amount_l355_35572


namespace cricket_player_average_increase_l355_35578

theorem cricket_player_average_increase
  (average : ℕ) (n : ℕ) (next_innings_runs : ℕ) 
  (x : ℕ) 
  (h1 : average = 32)
  (h2 : n = 20)
  (h3 : next_innings_runs = 200)
  (total_runs := average * n)
  (new_total_runs := total_runs + next_innings_runs)
  (new_average := (average + x))
  (new_total := new_average * (n + 1)):
  new_total_runs = 840 →
  new_total = 840 →
  x = 8 :=
by
  sorry

end cricket_player_average_increase_l355_35578


namespace birds_joined_l355_35597

variable (initialBirds : ℕ) (totalBirds : ℕ)

theorem birds_joined (h1 : initialBirds = 2) (h2 : totalBirds = 6) : (totalBirds - initialBirds) = 4 :=
by
  sorry

end birds_joined_l355_35597


namespace shelves_needed_l355_35557

def total_books : ℕ := 46
def books_taken_by_librarian : ℕ := 10
def books_per_shelf : ℕ := 4
def remaining_books : ℕ := total_books - books_taken_by_librarian
def needed_shelves : ℕ := 9

theorem shelves_needed : remaining_books / books_per_shelf = needed_shelves := by
  sorry

end shelves_needed_l355_35557


namespace range_of_a_l355_35591

theorem range_of_a (a : ℝ) : (∃ x : ℝ, |x + 2| - |x - 3| ≤ a) → a ≥ -5 := by
  sorry

end range_of_a_l355_35591


namespace odd_positive_93rd_l355_35523

theorem odd_positive_93rd : 
  (2 * 93 - 1) = 185 := 
by sorry

end odd_positive_93rd_l355_35523


namespace three_solutions_no_solutions_2891_l355_35532

theorem three_solutions (n : ℤ) (hpos : n > 0) (hx : ∃ (x y : ℤ), x^3 - 3 * x * y^2 + y^3 = n) :
  ∃ (x1 y1 x2 y2 x3 y3 : ℤ), 
    x1^3 - 3 * x1 * y1^2 + y1^3 = n ∧ 
    x2^3 - 3 * x2 * y2^2 + y2^3 = n ∧ 
    x3^3 - 3 * x3 * y3^2 + y3^3 = n := 
sorry

theorem no_solutions_2891 : ¬ ∃ (x y : ℤ), x^3 - 3 * x * y^2 + y^3 = 2891 :=
sorry

end three_solutions_no_solutions_2891_l355_35532


namespace mrs_li_actual_birthdays_l355_35574
   
   def is_leap_year (year : ℕ) : Prop :=
     (year % 4 = 0 ∧ year % 100 ≠ 0) ∨ (year % 400 = 0)
   
   def num_leap_years (start end_ : ℕ) : ℕ :=
     (start / 4 - start / 100 + start / 400) -
     (end_ / 4 - end_ / 100 + end_ / 400)
   
   theorem mrs_li_actual_birthdays : num_leap_years 1944 2011 = 16 :=
   by
     -- Calculation logic for the proof
     sorry
   
end mrs_li_actual_birthdays_l355_35574


namespace match_piles_l355_35534

theorem match_piles (a b c : ℕ) (h : a + b + c = 96)
    (h1 : 2 * b = a + c) (h2 : 2 * c = b + a) (h3 : 2 * a = c + b) : 
    a = 44 ∧ b = 28 ∧ c = 24 :=
  sorry

end match_piles_l355_35534


namespace chinese_character_symmetry_l355_35599

-- Definitions of the characters and their symmetry properties
def is_symmetric (ch : String) : Prop :=
  ch = "喜"

-- Hypotheses (conditions)
def option_A := "喜"
def option_B := "欢"
def option_C := "数"
def option_D := "学"

-- Lean statement to prove the symmetry
theorem chinese_character_symmetry :
  is_symmetric option_A ∧ 
  ¬ is_symmetric option_B ∧ 
  ¬ is_symmetric option_C ∧ 
  ¬ is_symmetric option_D :=
by
  sorry

end chinese_character_symmetry_l355_35599


namespace temperature_representation_l355_35548

-- Defining the temperature representation problem
def posTemp := 10 -- $10^\circ \mathrm{C}$ above zero
def negTemp := -10 -- $10^\circ \mathrm{C}$ below zero
def aboveZero (temp : Int) : Prop := temp > 0
def belowZero (temp : Int) : Prop := temp < 0

-- The proof statement to be proved using the given conditions
theorem temperature_representation : 
  (aboveZero posTemp → posTemp = 10) ∧ (belowZero negTemp → negTemp = -10) := 
  by
    sorry -- Proof would go here

end temperature_representation_l355_35548


namespace geometric_sequence_inserted_product_l355_35576

theorem geometric_sequence_inserted_product :
  ∃ (a b c : ℝ), a * b * c = 216 ∧
    (∃ (q : ℝ), 
      a = (8/3) * q ∧ 
      b = a * q ∧ 
      c = b * q ∧ 
      (8/3) * q^4 = 27/2) :=
sorry

end geometric_sequence_inserted_product_l355_35576


namespace min_a_plus_b_l355_35524

theorem min_a_plus_b (a b : ℝ) (h : a^2 + 2 * b^2 = 6) : a + b ≥ -3 :=
sorry

end min_a_plus_b_l355_35524


namespace larger_solution_quadratic_l355_35502

theorem larger_solution_quadratic (x : ℝ) : x^2 - 13 * x + 42 = 0 → x = 7 ∨ x = 6 ∧ x > 6 :=
by
  sorry

end larger_solution_quadratic_l355_35502


namespace track_is_600_l355_35522

noncomputable def track_length (x : ℝ) : Prop :=
  ∃ (s_b s_s : ℝ), 
      s_b > 0 ∧ s_s > 0 ∧
      (∀ t, t > 0 → ((s_b * t = 120 ∧ s_s * t = x / 2 - 120) ∨ 
                     (s_s * (t + 180 / s_s) - s_s * t = x / 2 + 60 
                      ∧ s_b * (t + 180 / s_s) - s_b * t = x / 2 - 60)))

theorem track_is_600 : track_length 600 :=
sorry

end track_is_600_l355_35522


namespace divisor_correct_l355_35518

/--
Given that \(10^{23} - 7\) divided by \(d\) leaves a remainder 3, 
prove that \(d\) is equal to \(10^{23} - 10\).
-/
theorem divisor_correct :
  ∃ d : ℤ, (10^23 - 7) % d = 3 ∧ d = 10^23 - 10 :=
by
  sorry

end divisor_correct_l355_35518


namespace quadratic_solutions_l355_35564

theorem quadratic_solutions (x : ℝ) : x * (x - 1) = 1 - x ↔ x = 1 ∨ x = -1 :=
by
  sorry

end quadratic_solutions_l355_35564


namespace range_of_k_l355_35520

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x ≤ 1 ∨ x ≥ 3}
def B (k : ℝ) : Set ℝ := {x | k < x ∧ x < 2 * k + 1}
def A_complement : Set ℝ := {x | 1 < x ∧ x < 3}

theorem range_of_k (k : ℝ) : ((A_complement ∩ (B k)) = ∅) ↔ (k ∈ Set.Iic 0 ∪ Set.Ici 3) := sorry

end range_of_k_l355_35520


namespace slower_speed_l355_35541

theorem slower_speed (f e d : ℕ) (h1 : f = 14) (h2 : e = 20) (h3 : d = 50) (x : ℕ) : 
  (50 / x : ℚ) = (50 / 14 : ℚ) + (20 / 14 : ℚ) → x = 10 := by
  sorry

end slower_speed_l355_35541


namespace correct_option_l355_35521

theorem correct_option :
  (2 * Real.sqrt 5) + (3 * Real.sqrt 5) = 5 * Real.sqrt 5 :=
by sorry

end correct_option_l355_35521


namespace card_average_value_l355_35526

theorem card_average_value (n : ℕ) (h : (2 * n + 1) / 3 = 2023) : n = 3034 :=
sorry

end card_average_value_l355_35526


namespace remaining_gnomes_total_l355_35527

/--
The remaining number of gnomes in the three forests after the owner takes his specified percentages.
-/
theorem remaining_gnomes_total :
  let westerville_gnomes := 20
  let ravenswood_gnomes := 4 * westerville_gnomes
  let greenwood_grove_gnomes := ravenswood_gnomes + (25 * ravenswood_gnomes) / 100
  let remaining_ravenswood := ravenswood_gnomes - (40 * ravenswood_gnomes) / 100
  let remaining_westerville := westerville_gnomes - (30 * westerville_gnomes) / 100
  let remaining_greenwood_grove := greenwood_grove_gnomes - (50 * greenwood_grove_gnomes) / 100
  remaining_ravenswood + remaining_westerville + remaining_greenwood_grove = 112 := by
  sorry

end remaining_gnomes_total_l355_35527


namespace find_x_l355_35593

theorem find_x (x : ℝ) : 17 + x + 2 * x + 13 = 60 → x = 10 :=
by
  sorry

end find_x_l355_35593


namespace y_coordinate_of_intersection_l355_35594

def line_eq (x t : ℝ) : ℝ := -2 * x + t

def parabola_eq (x : ℝ) : ℝ := (x - 1) ^ 2 + 1

def intersection_condition (x y t : ℝ) : Prop :=
  y = line_eq x t ∧ y = parabola_eq x ∧ x ≥ 0 ∧ y ≥ 0

theorem y_coordinate_of_intersection (x y : ℝ) (t : ℝ) (h_t : t = 11)
  (h_intersection : intersection_condition x y t) :
  y = 5 := by
  sorry

end y_coordinate_of_intersection_l355_35594


namespace problem_l355_35561

variable (a b : ℝ)

theorem problem (h₁ : a + b = 2) (h₂ : a - b = 3) : a^2 - b^2 = 6 := by
  sorry

end problem_l355_35561


namespace problem_statement_l355_35545

noncomputable def f (x k : ℝ) : ℝ :=
  (1/5) * (x - k + 4500 / x)

noncomputable def fuel_consumption_100km (x k : ℝ) : ℝ :=
  100 / x * f x k

theorem problem_statement (x k : ℝ)
  (hx1 : 60 ≤ x) (hx2 : x ≤ 120)
  (hk1 : 60 ≤ k) (hk2 : k ≤ 100)
  (H : f 120 k = 11.5) :

  (∀ x, 60 ≤ x ∧ x ≤ 100 → f x k ≤ 9 ∧ 
  (if 75 ≤ k ∧ k ≤ 100 then fuel_consumption_100km (9000 / k) k = 20 - k^2 / 900
   else fuel_consumption_100km 120 k = 105 / 4 - k / 6)) :=
  sorry

end problem_statement_l355_35545


namespace maria_total_cost_l355_35583

variable (pencil_cost : ℕ)
variable (pen_cost : ℕ)

def total_cost (pencil_cost pen_cost : ℕ) : ℕ :=
  pencil_cost + pen_cost

theorem maria_total_cost : pencil_cost = 8 → pen_cost = pencil_cost / 2 → total_cost pencil_cost pen_cost = 12 := by
  sorry

end maria_total_cost_l355_35583


namespace playground_area_22500_l355_35504

noncomputable def rectangle_playground_area (w l : ℕ) : ℕ :=
  w * l

theorem playground_area_22500 (w l : ℕ) (h1 : l = 2 * w + 25) (h2 : 2 * l + 2 * w = 650) :
  rectangle_playground_area w l = 22500 := by
  sorry

end playground_area_22500_l355_35504


namespace probability_diff_colors_l355_35571

-- Definitions based on the conditions provided.
-- Total number of chips
def total_chips := 15

-- Individual probabilities of drawing each color first
def prob_green_first := 6 / total_chips
def prob_purple_first := 5 / total_chips
def prob_orange_first := 4 / total_chips

-- Probabilities of drawing a different color second
def prob_not_green := 9 / total_chips
def prob_not_purple := 10 / total_chips
def prob_not_orange := 11 / total_chips

-- Combined probabilities for each case
def prob_green_then_diff := prob_green_first * prob_not_green
def prob_purple_then_diff := prob_purple_first * prob_not_purple
def prob_orange_then_diff := prob_orange_first * prob_not_orange

-- Total probability of drawing two chips of different colors
def total_prob_diff_colors := prob_green_then_diff + prob_purple_then_diff + prob_orange_then_diff

-- Theorem statement to be proved
theorem probability_diff_colors : total_prob_diff_colors = 148 / 225 :=
by
  -- Proof would go here
  sorry

end probability_diff_colors_l355_35571


namespace roots_condition_l355_35512

theorem roots_condition (m : ℝ) (f : ℝ → ℝ) (x1 x2 : ℝ) (h_f : ∀ x, f x = x^2 + 2*(m - 1)*x - 5*m - 2) 
  (h_roots : ∃ x1 x2, x1 < 1 ∧ 1 < x2 ∧ f x1 = 0 ∧ f x2 = 0) : 
  m > 1 := 
by
  sorry

end roots_condition_l355_35512


namespace minimal_disks_needed_l355_35580

-- Define the capacity of one disk
def disk_capacity : ℝ := 2.0

-- Define the number of files and their sizes
def num_files_0_9 : ℕ := 5
def size_file_0_9 : ℝ := 0.9

def num_files_0_8 : ℕ := 15
def size_file_0_8 : ℝ := 0.8

def num_files_0_5 : ℕ := 20
def size_file_0_5 : ℝ := 0.5

-- Total number of files
def total_files : ℕ := num_files_0_9 + num_files_0_8 + num_files_0_5

-- Proof statement: the minimal number of disks needed to store all files given their sizes and the disk capacity
theorem minimal_disks_needed : 
  ∀ (d : ℕ), 
    d = 18 → 
    total_files = 40 → 
    disk_capacity = 2.0 → 
    ((num_files_0_9 * size_file_0_9 + num_files_0_8 * size_file_0_8 + num_files_0_5 * size_file_0_5) / disk_capacity) ≤ d
  :=
by
  sorry

end minimal_disks_needed_l355_35580


namespace find_square_divisible_by_four_l355_35547

/-- There exists an x such that x is a square number, x is divisible by four, 
and 39 < x < 80, and that x = 64 is such a number. --/
theorem find_square_divisible_by_four : ∃ (x : ℕ), (∃ (n : ℕ), x = n^2) ∧ (x % 4 = 0) ∧ (39 < x ∧ x < 80) ∧ x = 64 :=
  sorry

end find_square_divisible_by_four_l355_35547


namespace inequality_of_abc_l355_35544

theorem inequality_of_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (a + b + c) * (1/a + 1/b + 1/c) ≥ 9 :=
sorry

end inequality_of_abc_l355_35544


namespace vectors_collinear_has_solution_l355_35579

-- Define the vectors
def a (x : ℝ) : ℝ × ℝ := (x^2 - 1, 2 + x)
def b (x : ℝ) : ℝ × ℝ := (x, 1)

-- Collinearity condition (cross product must be zero) as a function
def collinear (x : ℝ) : Prop := (a x).1 * (b x).2 - (b x).1 * (a x).2 = 0

-- The proof statement
theorem vectors_collinear_has_solution (x : ℝ) (h : collinear x) : x = -1 / 2 :=
sorry

end vectors_collinear_has_solution_l355_35579


namespace dave_guitar_strings_l355_35515

theorem dave_guitar_strings (strings_per_night : ℕ) (shows_per_week : ℕ) (weeks : ℕ)
  (h1 : strings_per_night = 4)
  (h2 : shows_per_week = 6)
  (h3 : weeks = 24) : 
  strings_per_night * shows_per_week * weeks = 576 :=
by
  sorry

end dave_guitar_strings_l355_35515


namespace polygon_sides_from_interior_angles_l355_35549

theorem polygon_sides_from_interior_angles (S : ℕ) (h : S = 1260) : S = (9 - 2) * 180 :=
by
  sorry

end polygon_sides_from_interior_angles_l355_35549


namespace cos_B_eq_find_b_eq_l355_35550

variable (A B C a b c : ℝ)

-- Given conditions
axiom sin_A_plus_C_eq : Real.sin (A + C) = 8 * Real.sin (B / 2) ^ 2
axiom a_plus_c : a + c = 6
axiom area_of_triangle : 1 / 2 * a * c * Real.sin B = 2

-- Proving cos B
theorem cos_B_eq :
  Real.cos B = 15 / 17 :=
sorry

-- Proving b given the area and sides condition
theorem find_b_eq :
  Real.cos B = 15 / 17 → b = 2 :=
sorry

end cos_B_eq_find_b_eq_l355_35550


namespace ratio_kid_to_adult_ticket_l355_35563

theorem ratio_kid_to_adult_ticket (A : ℝ) : 
  (6 * 5 + 2 * A = 50) → (5 / A = 1 / 2) :=
by
  sorry

end ratio_kid_to_adult_ticket_l355_35563


namespace set_intersection_complement_l355_35539

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5, 6}

-- Define set A
def A : Set ℕ := {1, 2, 3}

-- Define set B
def B : Set ℕ := {3, 4, 5}

-- State the theorem
theorem set_intersection_complement :
  (U \ A) ∩ B = {4, 5} := by
  sorry

end set_intersection_complement_l355_35539


namespace Diego_more_than_half_Martha_l355_35552

theorem Diego_more_than_half_Martha (M D : ℕ) (H1 : M = 90)
  (H2 : D > M / 2)
  (H3 : M + D = 145):
  D - M / 2 = 10 :=
by
  sorry

end Diego_more_than_half_Martha_l355_35552


namespace correct_answer_l355_35587

-- Definition of the correctness condition
def indicates_number (phrase : String) : Prop :=
  (phrase = "Noun + Cardinal Number") ∨ (phrase = "the + Ordinal Number + Noun")

-- Example phrases to be evaluated
def class_first : String := "Class First"
def the_class_one : String := "the Class One"
def class_one : String := "Class One"
def first_class : String := "First Class"

-- The goal is to prove that "Class One" meets the condition
theorem correct_answer : indicates_number "Class One" :=
by {
  -- Insert detailed proof steps here, currently omitted
  sorry
}

end correct_answer_l355_35587


namespace remainder_expression_l355_35509

theorem remainder_expression (x y u v : ℕ) (h1 : x = u * y + v) (h2 : 0 ≤ v) (h3 : v < y) : 
  (x + 3 * u * y) % y = v := 
by
  sorry

end remainder_expression_l355_35509


namespace ages_of_residents_l355_35573

theorem ages_of_residents (a b c : ℕ)
  (h1 : a * b * c = 1296)
  (h2 : a + b + c = 91)
  (h3 : ∀ x y z : ℕ, x * y * z = 1296 → x + y + z = 91 → (x < 80 ∧ y < 80 ∧ z < 80) → (x = 1 ∧ y = 18 ∧ z = 72)) :
  (a = 1 ∧ b = 18 ∧ c = 72 ∨ a = 1 ∧ b = 72 ∧ c = 18 ∨ a = 18 ∧ b = 1 ∧ c = 72 ∨ a = 18 ∧ b = 72 ∧ c = 1 ∨ a = 72 ∧ b = 1 ∧ c = 18 ∨ a = 72 ∧ b = 18 ∧ c = 1) :=
by
  sorry

end ages_of_residents_l355_35573


namespace find_prime_solution_l355_35505

theorem find_prime_solution :
  ∀ p x y : ℕ, Prime p → x > 0 → y > 0 →
    (p ^ x = y ^ 3 + 1) ↔ 
    ((p = 2 ∧ x = 1 ∧ y = 1) ∨ (p = 3 ∧ x = 2 ∧ y = 2)) := 
by
  sorry

end find_prime_solution_l355_35505
