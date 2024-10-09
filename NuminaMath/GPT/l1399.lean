import Mathlib

namespace cannot_achieve_1_5_percent_salt_solution_l1399_139984

-- Define the initial concentrations and volumes
def initial_state (V1 V2 : ℝ) (C1 C2 : ℝ) : Prop :=
  V1 = 1 ∧ C1 = 0 ∧ V2 = 1 ∧ C2 = 0.02

-- Define the transfer and mixing operation
noncomputable def transfer_and_mix (V1_old V2_old C1_old C2_old : ℝ) (amount_to_transfer : ℝ)
  (new_V1 new_V2 new_C1 new_C2 : ℝ) : Prop :=
  amount_to_transfer ≤ V2_old ∧
  new_V1 = V1_old + amount_to_transfer ∧
  new_V2 = V2_old - amount_to_transfer ∧
  new_C1 = (V1_old * C1_old + amount_to_transfer * C2_old) / new_V1 ∧
  new_C2 = (V2_old * C2_old - amount_to_transfer * C2_old) / new_V2

-- Prove that it is impossible to achieve a 1.5% salt concentration in container 1
theorem cannot_achieve_1_5_percent_salt_solution :
  ∀ V1 V2 C1 C2, initial_state V1 V2 C1 C2 →
  ¬ ∃ V1' V2' C1' C2', transfer_and_mix V1 V2 C1 C2 0.5 V1' V2' C1' C2' ∧ C1' = 0.015 :=
by
  intros
  sorry

end cannot_achieve_1_5_percent_salt_solution_l1399_139984


namespace table_mat_length_l1399_139977

noncomputable def calculate_y (r : ℝ) (n : ℕ) (w : ℝ) : ℝ :=
  let θ := 2 * Real.pi / n
  let y_side := 2 * r * Real.sin (θ / 2)
  y_side

theorem table_mat_length :
  calculate_y 6 8 1 = 3 * Real.sqrt (2 - Real.sqrt 2) :=
by
  sorry

end table_mat_length_l1399_139977


namespace parallelogram_opposite_sides_equal_l1399_139954

-- Given definitions and properties of a parallelogram
structure Parallelogram (α : Type*) [Add α] [AddCommGroup α] [Module ℝ α] :=
(a b c d : α) 
(parallel_a : a + b = c + d)
(parallel_b : b + c = d + a)
(parallel_c : c + d = a + b)
(parallel_d : d + a = b + c)

open Parallelogram

-- Define problem statement to prove opposite sides are equal
theorem parallelogram_opposite_sides_equal {α : Type*} [Add α] [AddCommGroup α] [Module ℝ α] 
  (p : Parallelogram α) : 
  p.a = p.c ∧ p.b = p.d :=
sorry -- Proof goes here

end parallelogram_opposite_sides_equal_l1399_139954


namespace difference_of_squares_144_l1399_139902

theorem difference_of_squares_144 (n : ℕ) (h : 3 * n + 3 < 150) : (n + 2)^2 - n^2 = 144 :=
by
  -- Given the conditions, we need to show this holds.
  sorry

end difference_of_squares_144_l1399_139902


namespace weight_difference_l1399_139918

open Real

def yellow_weight : ℝ := 0.6
def green_weight : ℝ := 0.4
def red_weight : ℝ := 0.8
def blue_weight : ℝ := 0.5

def weights : List ℝ := [yellow_weight, green_weight, red_weight, blue_weight]

theorem weight_difference : (List.maximum weights).getD 0 - (List.minimum weights).getD 0 = 0.4 :=
by
  sorry

end weight_difference_l1399_139918


namespace other_girl_age_l1399_139975

theorem other_girl_age (x : ℕ) (h1 : 13 + x = 27) : x = 14 := by
  sorry

end other_girl_age_l1399_139975


namespace meaningful_fraction_l1399_139913

theorem meaningful_fraction (x : ℝ) : (∃ y, y = (1 / (x - 2))) ↔ x ≠ 2 :=
by
  sorry

end meaningful_fraction_l1399_139913


namespace steve_cookie_boxes_l1399_139927

theorem steve_cookie_boxes (total_spent milk_cost cereal_cost banana_cost apple_cost : ℝ)
  (num_cereals num_bananas num_apples : ℕ) (cookie_cost_multiplier : ℝ) (cookie_cost : ℝ)
  (cookie_boxes : ℕ) :
  total_spent = 25 ∧ milk_cost = 3 ∧ cereal_cost = 3.5 ∧ banana_cost = 0.25 ∧ apple_cost = 0.5 ∧
  cookie_cost_multiplier = 2 ∧ 
  num_cereals = 2 ∧ num_bananas = 4 ∧ num_apples = 4 ∧
  cookie_cost = cookie_cost_multiplier * milk_cost ∧
  total_spent = (milk_cost + num_cereals * cereal_cost + num_bananas * banana_cost + num_apples * apple_cost + cookie_boxes * cookie_cost)
  → cookie_boxes = 2 :=
sorry

end steve_cookie_boxes_l1399_139927


namespace pseudo_symmetry_abscissa_l1399_139972

noncomputable def f (x : ℝ) : ℝ := x^2 - 6*x + 4 * Real.log x

theorem pseudo_symmetry_abscissa :
  ∃ x0 : ℝ, x0 = Real.sqrt 2 ∧
    (∀ x : ℝ, x ≠ x0 → (f x - ((2*x0 + 4/x0 - 6)*(x - x0) + x0^2 - 6*x0 + 4*Real.log x0)) / (x - x0) > 0) :=
sorry

end pseudo_symmetry_abscissa_l1399_139972


namespace number_eq_180_l1399_139946

theorem number_eq_180 (x : ℝ) (h : 64 + 5 * 12 / (x / 3) = 65) : x = 180 :=
sorry

end number_eq_180_l1399_139946


namespace determine_g_l1399_139905

variable (g : ℕ → ℕ)

theorem determine_g (h : ∀ x, g (x + 1) = 2 * x + 3) : ∀ x, g x = 2 * x + 1 :=
by
  sorry

end determine_g_l1399_139905


namespace frac_series_simplification_l1399_139937

theorem frac_series_simplification :
  (1 - 2 + 4 - 8 + 16 - 32 + 64 - 128 : ℚ) / (1^2 + 2^2 - 4^2 + 8^2 + 16^2 - 32^2 + 64^2 - 128^2 : ℚ) = 1 / 113 := 
by
  sorry

end frac_series_simplification_l1399_139937


namespace total_whales_correct_l1399_139921

def first_trip_male_whales : ℕ := 28
def first_trip_female_whales : ℕ := 2 * first_trip_male_whales
def first_trip_total_whales : ℕ := first_trip_male_whales + first_trip_female_whales

def second_trip_baby_whales : ℕ := 8
def second_trip_parent_whales : ℕ := 2 * second_trip_baby_whales
def second_trip_total_whales : ℕ := second_trip_baby_whales + second_trip_parent_whales

def third_trip_male_whales : ℕ := first_trip_male_whales / 2
def third_trip_female_whales : ℕ := first_trip_female_whales
def third_trip_total_whales : ℕ := third_trip_male_whales + third_trip_female_whales

def total_whales_seen : ℕ :=
  first_trip_total_whales + second_trip_total_whales + third_trip_total_whales

theorem total_whales_correct : total_whales_seen = 178 := by
  sorry

end total_whales_correct_l1399_139921


namespace jacks_walking_rate_l1399_139979

variable (distance : ℝ) (hours : ℝ) (minutes : ℝ)

theorem jacks_walking_rate (h_distance : distance = 4) (h_hours : hours = 1) (h_minutes : minutes = 15) :
  distance / (hours + minutes / 60) = 3.2 :=
by
  sorry

end jacks_walking_rate_l1399_139979


namespace first_duck_fraction_l1399_139985

-- Definitions based on the conditions
variable (total_bread : ℕ) (left_bread : ℕ) (second_duck_bread : ℕ) (third_duck_bread : ℕ)

-- Given values
def given_values : Prop :=
  total_bread = 100 ∧ left_bread = 30 ∧ second_duck_bread = 13 ∧ third_duck_bread = 7

-- Proof statement
theorem first_duck_fraction (h : given_values total_bread left_bread second_duck_bread third_duck_bread) :
  (total_bread - left_bread) - (second_duck_bread + third_duck_bread) = 1/2 * total_bread := by 
  sorry

end first_duck_fraction_l1399_139985


namespace min_value_z_l1399_139965

theorem min_value_z (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x + y = 1) :
  (x + 1/x) * (y + 1/y) ≥ 25/4 := 
sorry

end min_value_z_l1399_139965


namespace edward_can_buy_candies_l1399_139960

theorem edward_can_buy_candies (whack_a_mole_tickets skee_ball_tickets candy_cost : ℕ)
  (h1 : whack_a_mole_tickets = 3) (h2 : skee_ball_tickets = 5) (h3 : candy_cost = 4) :
  (whack_a_mole_tickets + skee_ball_tickets) / candy_cost = 2 :=
by
  sorry

end edward_can_buy_candies_l1399_139960


namespace arithmetic_sequence_a_value_l1399_139923

theorem arithmetic_sequence_a_value :
  ∀ (a : ℤ), (-7) - a = a - 1 → a = -3 :=
by
  intro a
  intro h
  sorry

end arithmetic_sequence_a_value_l1399_139923


namespace tangent_line_eq_l1399_139945

theorem tangent_line_eq (x y: ℝ):
  (x^2 + y^2 = 4) → ((2, 3) = (x, y)) →
  (x = 2 ∨ 5 * x - 12 * y + 26 = 0) :=
by
  sorry

end tangent_line_eq_l1399_139945


namespace arithmetic_sequence_sum_range_l1399_139907

noncomputable def S_n (a d : ℝ) (n : ℕ) : ℝ := n * (2 * a + (n - 1) * d) / 2

theorem arithmetic_sequence_sum_range 
  (a d : ℝ)
  (h1 : 1 ≤ a + 3 * d) 
  (h2 : a + 3 * d ≤ 4)
  (h3 : 2 ≤ a + 4 * d)
  (h4 : a + 4 * d ≤ 3) 
  : 0 ≤ S_n a d 6 ∧ S_n a d 6 ≤ 30 := 
sorry

end arithmetic_sequence_sum_range_l1399_139907


namespace quadratic_roots_difference_l1399_139995

theorem quadratic_roots_difference (p q : ℝ) (hp : 0 < p) (hq : 0 < q) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 - x2 = 2 ∧ x1 * x2 = q ∧ x1 + x2 = -p) → p = 2 * Real.sqrt (q + 1) :=
by
  sorry

end quadratic_roots_difference_l1399_139995


namespace area_of_triangle_l1399_139989

theorem area_of_triangle:
  let line1 := λ x => 3 * x - 6
  let line2 := λ x => -2 * x + 18
  let y_axis: ℝ → ℝ := λ _ => 0
  let intersection := (4.8, line1 4.8)
  let y_intercept1 := (0, -6)
  let y_intercept2 := (0, 18)
  (1/2) * 24 * 4.8 = 57.6 := by
  sorry

end area_of_triangle_l1399_139989


namespace total_cost_of_ads_l1399_139990

-- Define the conditions
def cost_ad1 := 3500
def minutes_ad1 := 2
def cost_ad2 := 4500
def minutes_ad2 := 3
def cost_ad3 := 3000
def minutes_ad3 := 3
def cost_ad4 := 4000
def minutes_ad4 := 2
def cost_ad5 := 5500
def minutes_ad5 := 5

-- Define the function to calculate the total cost
def total_cost :=
  (cost_ad1 * minutes_ad1) +
  (cost_ad2 * minutes_ad2) +
  (cost_ad3 * minutes_ad3) +
  (cost_ad4 * minutes_ad4) +
  (cost_ad5 * minutes_ad5)

-- The statement to prove
theorem total_cost_of_ads : total_cost = 66000 := by
  sorry

end total_cost_of_ads_l1399_139990


namespace exists_six_distinct_naturals_l1399_139948

theorem exists_six_distinct_naturals :
  ∃ (a b c d e f : ℕ), 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ 
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ 
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ 
    d ≠ e ∧ d ≠ f ∧ 
    e ≠ f ∧ 
    a + b + c + d + e + f = 3528 ∧
    (1/a + 1/b + 1/c + 1/d + 1/e + 1/f : ℝ) = 3528 / 2012 :=
sorry

end exists_six_distinct_naturals_l1399_139948


namespace train_length_l1399_139981

def train_speed_kmph := 25 -- speed of train in km/h
def man_speed_kmph := 2 -- speed of man in km/h
def crossing_time_sec := 52 -- time to cross in seconds

noncomputable def length_of_train : ℝ :=
  let relative_speed_kmph := train_speed_kmph + man_speed_kmph -- relative speed in km/h
  let relative_speed_mps := relative_speed_kmph * (5 / 18) -- convert to m/s
  relative_speed_mps * crossing_time_sec -- length of train in meters

theorem train_length : length_of_train = 390 :=
  by sorry -- proof omitted

end train_length_l1399_139981


namespace starting_lineups_possible_l1399_139931

open Nat

theorem starting_lineups_possible (total_players : ℕ) (all_stars : ℕ) (lineup_size : ℕ) 
  (fixed_in_lineup : ℕ) (choose_size : ℕ) 
  (h_fixed : fixed_in_lineup = all_stars)
  (h_remaining : total_players - fixed_in_lineup = choose_size)
  (h_lineup : lineup_size = all_stars + choose_size) :
  (Nat.choose choose_size 3 = 220) :=
by
  sorry

end starting_lineups_possible_l1399_139931


namespace joe_egg_count_l1399_139973

theorem joe_egg_count : 
  let clubhouse : ℕ := 12
  let park : ℕ := 5
  let townhall : ℕ := 3
  clubhouse + park + townhall = 20 :=
by
  sorry

end joe_egg_count_l1399_139973


namespace range_of_k_l1399_139966

noncomputable def triangle_range (A B C : ℝ) (a b c k : ℝ) : Prop :=
  A + B + C = Real.pi ∧
  (B = Real.pi / 3) ∧       -- From arithmetic sequence and solving for B
  a^2 + c^2 = k * b^2 ∧
  (1 < k ∧ k <= 2)

theorem range_of_k (A B C a b c k : ℝ) :
  A + B + C = Real.pi →
  (B = Real.pi - (A + C)) →
  (B = Real.pi / 3) →
  a^2 + c^2 = k * b^2 →
  0 < A ∧ A < 2*Real.pi/3 →
  1 < k ∧ k <= 2 :=
by
  sorry

end range_of_k_l1399_139966


namespace find_beta_l1399_139988

theorem find_beta 
  (α β : ℝ)
  (h1 : Real.cos α = 1 / 7)
  (h2 : Real.cos (α + β) = -11 / 14)
  (h3 : 0 < α ∧ α < Real.pi / 2)
  (h4 : Real.pi / 2 < α + β ∧ α + β < Real.pi) :
  β = Real.pi / 3 := 
sorry

end find_beta_l1399_139988


namespace simplified_result_l1399_139933

theorem simplified_result (a b M : ℝ) (h1 : (2 * a) / (a ^ 2 - b ^ 2) - 1 / M = 1 / (a - b))
  (h2 : M - (a - b) = 2 * b) : (2 * a) / (a ^ 2 - b ^ 2) - 1 / (a - b) = 1 / (a + b) :=
by
  sorry

end simplified_result_l1399_139933


namespace num_valid_arrangements_without_A_at_start_and_B_at_end_l1399_139932

-- Define a predicate for person A being at the beginning
def A_at_beginning (arrangement : List ℕ) : Prop :=
  arrangement.head! = 1

-- Define a predicate for person B being at the end
def B_at_end (arrangement : List ℕ) : Prop :=
  arrangement.getLast! = 2

-- Main theorem stating the number of valid arrangements
theorem num_valid_arrangements_without_A_at_start_and_B_at_end : ∃ (count : ℕ), count = 78 :=
by
  have total_arrangements := Nat.factorial 5
  have A_at_start_arrangements := Nat.factorial 4
  have B_at_end_arrangements := Nat.factorial 4
  have both_A_and_B_arrangements := Nat.factorial 3
  let valid_arrangements := total_arrangements - 2 * A_at_start_arrangements + both_A_and_B_arrangements
  use valid_arrangements
  sorry

end num_valid_arrangements_without_A_at_start_and_B_at_end_l1399_139932


namespace probability_of_at_least_one_die_shows_2_is_correct_l1399_139942

-- Definitions for the conditions
def total_outcomes : ℕ := 64
def neither_die_shows_2_outcomes : ℕ := 49
def favorability (total : ℕ) (exclusion : ℕ) : ℕ := total - exclusion
def favorable_outcomes : ℕ := favorability total_outcomes neither_die_shows_2_outcomes
def probability (favorable : ℕ) (total : ℕ) : ℚ := favorable / total

-- Mathematically equivalent proof problem statement
theorem probability_of_at_least_one_die_shows_2_is_correct : 
  probability favorable_outcomes total_outcomes = 15 / 64 :=
sorry

end probability_of_at_least_one_die_shows_2_is_correct_l1399_139942


namespace number_of_subsets_l1399_139961

theorem number_of_subsets (x y : Type) :  ∃ s : Finset (Finset Type), s.card = 4 := 
sorry

end number_of_subsets_l1399_139961


namespace binom_expansion_l1399_139915

/-- Given the binomial expansion of (sqrt(x) + 3x)^n for n < 15, 
    with the binomial coefficients of the 9th, 10th, and 11th terms forming an arithmetic sequence,
    we conclude that n must be 14 and describe all the rational terms in the expansion.
-/
theorem binom_expansion (n : ℕ) (h : n < 15)
  (h_seq : Nat.choose n 8 + Nat.choose n 10 = 2 * Nat.choose n 9) :
  n = 14 ∧
  (∃ (t1 t2 t3 : ℕ), 
    (t1 = 1 ∧ (Nat.choose 14 0 : ℕ) * (x ^ 7 : ℤ) = x ^ 7) ∧
    (t2 = 164 ∧ (Nat.choose 14 6 : ℕ) * (x ^ 6 : ℤ) = 164 * x ^ 6) ∧
    (t3 = 91 ∧ (Nat.choose 14 12 : ℕ) * (x ^ 5 : ℤ) = 91 * x ^ 5)) := 
  sorry

end binom_expansion_l1399_139915


namespace ball_bounces_l1399_139952

theorem ball_bounces (k : ℕ) :
  1500 * (2 / 3 : ℝ)^k < 2 ↔ k ≥ 19 :=
sorry

end ball_bounces_l1399_139952


namespace find_number_of_pourings_l1399_139904

-- Define the sequence of remaining water after each pouring
def remaining_water (n : ℕ) : ℚ :=
  (2 : ℚ) / (n + 2)

-- The main theorem statement
theorem find_number_of_pourings :
  ∃ n : ℕ, remaining_water n = 1 / 8 :=
by
  sorry

end find_number_of_pourings_l1399_139904


namespace total_students_surveyed_l1399_139986

variable (T : ℕ)
variable (F : ℕ)

theorem total_students_surveyed :
  (F = 20 + 60) → (F = 40 * (T / 100)) → (T = 200) :=
by
  intros h1 h2
  sorry

end total_students_surveyed_l1399_139986


namespace relationship_abc_d_l1399_139969

theorem relationship_abc_d : 
  ∀ (a b c d : ℝ), 
  a < b → 
  d < c → 
  (c - a) * (c - b) < 0 → 
  (d - a) * (d - b) > 0 → 
  d < a ∧ a < c ∧ c < b :=
by
  intros a b c d a_lt_b d_lt_c h1 h2
  sorry

end relationship_abc_d_l1399_139969


namespace period_tan_half_l1399_139930

noncomputable def period_of_tan_half : Real :=
  2 * Real.pi

theorem period_tan_half (f : Real → Real) (h : ∀ x, f x = Real.tan (x / 2)) :
  ∀ x, f (x + period_of_tan_half) = f x := 
by 
  sorry

end period_tan_half_l1399_139930


namespace calculate_expression_l1399_139934

theorem calculate_expression : 
  (1 / 2) ^ (-2: ℤ) - 3 * Real.tan (Real.pi / 6) - abs (Real.sqrt 3 - 2) = 2 := 
by
  sorry

end calculate_expression_l1399_139934


namespace solve_exp_l1399_139976

theorem solve_exp (x : ℕ) : 8^x = 2^9 → x = 3 :=
by
  sorry

end solve_exp_l1399_139976


namespace coordinates_of_point_in_fourth_quadrant_l1399_139959

theorem coordinates_of_point_in_fourth_quadrant 
  (P : ℝ × ℝ)
  (h₁ : P.1 > 0) -- P is in the fourth quadrant, so x > 0
  (h₂ : P.2 < 0) -- P is in the fourth quadrant, so y < 0
  (dist_x_axis : P.2 = -5) -- Distance from P to x-axis is 5 (absolute value of y)
  (dist_y_axis : P.1 = 3)  -- Distance from P to y-axis is 3 (absolute value of x)
  : P = (3, -5) :=
sorry

end coordinates_of_point_in_fourth_quadrant_l1399_139959


namespace find_a9_l1399_139982

theorem find_a9 (a_1 a_2 : ℕ) (a : ℕ → ℕ) 
  (h1 : ∀ n ≥ 1, a (n + 2) = a (n + 1) + a n)
  (h2 : a 7 = 210)
  (h3 : a 1 = a_1)
  (h4 : a 2 = a_2) : 
  a 9 = 550 := by
  sorry

end find_a9_l1399_139982


namespace sum_of_digits_l1399_139992

theorem sum_of_digits (a b : ℕ) (h1 : 4 * 100 + a * 10 + 4 + 258 = 7 * 100 + b * 10 + 2) (h2 : (7 * 100 + b * 10 + 2) % 3 = 0) :
  a + b = 4 :=
sorry

end sum_of_digits_l1399_139992


namespace solution_set_l1399_139908

def f (x : ℝ) : ℝ := sorry

axiom ax1 : ∀ a b : ℝ, f (a + b) = f a + f b - 1
axiom ax2 : ∀ x : ℝ, x > 0 → f x > 1
axiom ax3 : f 4 = 5

theorem solution_set (x : ℝ) : f (3 * x^2 - x - 2) < 3 ↔ (-1 < x ∧ x < 4 / 3) :=
by
  sorry

end solution_set_l1399_139908


namespace toys_ratio_l1399_139999

-- Definitions of given conditions
variables (rabbits : ℕ) (toys_monday toys_wednesday toys_friday toys_saturday total_toys : ℕ)
variables (h_rabbits : rabbits = 16)
variables (h_toys_monday : toys_monday = 6)
variables (h_toys_friday : toys_friday = 4 * toys_monday)
variables (h_toys_saturday : toys_saturday = toys_wednesday / 2)
variables (h_total_toys : total_toys = rabbits * 3)

-- Define the Lean theorem to state the problem conditions and prove the ratio
theorem toys_ratio (h : toys_monday + toys_wednesday + toys_friday + toys_saturday = total_toys) :
  (if (2 * toys_wednesday = 12) then 2 else 1) = 2 :=
by 
  sorry

end toys_ratio_l1399_139999


namespace variance_scaled_l1399_139998

-- Let V represent the variance of the set of data
def original_variance : ℝ := 3
def scale_factor : ℝ := 3

-- Prove that the new variance is 27 
theorem variance_scaled (V : ℝ) (s : ℝ) (hV : V = 3) (hs : s = 3) : s^2 * V = 27 := by
  sorry

end variance_scaled_l1399_139998


namespace intersection_points_vary_with_a_l1399_139909

-- Define the lines
def line1 (x : ℝ) : ℝ := x + 1
def line2 (a x : ℝ) : ℝ := a * x + 1

-- Prove that the number of intersection points varies with a
theorem intersection_points_vary_with_a (a : ℝ) : 
  (∃ x : ℝ, line1 x = line2 a x) ↔ 
    (if a = 1 then true else true) :=
by 
  sorry

end intersection_points_vary_with_a_l1399_139909


namespace union_complement_eq_l1399_139983

open Set

variable (U A B : Set ℕ)

def complement (U A : Set ℕ) : Set ℕ := { x | x ∈ U ∧ x ∉ A }

theorem union_complement_eq (U A B : Set ℕ) (hU : U = {0, 1, 2, 3, 4}) (hA : A = {1, 2, 3}) (hB : B = {2, 4}) :
  (complement U A) ∪ B = {0, 2, 4} :=
by
  rw [hU, hA, hB]
  sorry

end union_complement_eq_l1399_139983


namespace andrey_stamps_count_l1399_139940

theorem andrey_stamps_count (x : ℕ) : 
  (x % 3 = 1) ∧ (x % 5 = 3) ∧ (x % 7 = 5) ∧ (150 < x ∧ x ≤ 300) → x = 208 := 
by 
  sorry

end andrey_stamps_count_l1399_139940


namespace albert_large_pizzas_l1399_139911

-- Define the conditions
def large_pizza_slices : ℕ := 16
def small_pizza_slices : ℕ := 8
def num_small_pizzas : ℕ := 2
def total_slices_eaten : ℕ := 48

-- Define the question and requirement to prove
def number_of_large_pizzas (L : ℕ) : Prop :=
  large_pizza_slices * L + small_pizza_slices * num_small_pizzas = total_slices_eaten

theorem albert_large_pizzas :
  number_of_large_pizzas 2 :=
by
  sorry

end albert_large_pizzas_l1399_139911


namespace complete_the_square_solution_l1399_139964

theorem complete_the_square_solution (x : ℝ) :
  (∃ x, x^2 + 2 * x - 1 = 0) → (x + 1)^2 = 2 :=
sorry

end complete_the_square_solution_l1399_139964


namespace train_speed_l1399_139912

/-- 
Given:
- Length of train L is 390 meters (0.39 km)
- Speed of man Vm is 2 km/h
- Time to cross man T is 52 seconds

Prove:
- The speed of the train Vt is 25 km/h
--/
theorem train_speed 
  (L : ℝ) (Vm : ℝ) (T : ℝ) (Vt : ℝ)
  (h1 : L = 0.39) 
  (h2 : Vm = 2) 
  (h3 : T = 52 / 3600) 
  (h4 : Vt + Vm = L / T) :
  Vt = 25 :=
by sorry

end train_speed_l1399_139912


namespace part1_part2_part3_l1399_139917

section Part1
variables {a b : ℝ}

theorem part1 (h1 : a + b = 3) (h2 : a * b = 2) : a^2 + b^2 = 5 := 
sorry
end Part1

section Part2
variables {a b c : ℝ}

theorem part2 (h1 : a + b + c = 6) (h2 : a * b + b * c + c * a = 11) : a^2 + b^2 + c^2 = 14 := 
sorry
end Part2

section Part3
variables {a b c : ℝ}

theorem part3 (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 = 6) : a^4 + b^4 + c^4 = 18 :=
sorry
end Part3

end part1_part2_part3_l1399_139917


namespace geometric_sequence_product_l1399_139938

theorem geometric_sequence_product
  (a : ℕ → ℝ) (b : ℕ → ℝ)
  (hA_seq : ∀ n, a n = a 1 + (n - 1) * (a 2 - a 1))
  (hA_not_zero : ∀ n, a n ≠ 0)
  (h_condition : a 4 - 2 * (a 7)^2 + 3 * a 8 = 0)
  (hB_seq : ∀ n, b n = b 1 * (b 2 / b 1)^(n - 1))
  (hB7 : b 7 = a 7) :
  b 3 * b 7 * b 11 = 8 := 
sorry

end geometric_sequence_product_l1399_139938


namespace total_songs_l1399_139910

open Nat

/-- Define the overall context and setup for the problem --/
def girls : List String := ["Mary", "Alina", "Tina", "Hanna"]

def hanna_songs : ℕ := 7
def mary_songs : ℕ := 4

def alina_songs (a : ℕ) : Prop := a > mary_songs ∧ a < hanna_songs
def tina_songs (t : ℕ) : Prop := t > mary_songs ∧ t < hanna_songs

theorem total_songs (a t : ℕ) (h_alina : alina_songs a) (h_tina : tina_songs t) : 
  (11 + a + t) % 3 = 0 → (7 + 4 + a + t) / 3 = 7 := by
  sorry

end total_songs_l1399_139910


namespace fraction_simplification_l1399_139991

theorem fraction_simplification : 1 + 1 / (1 - 1 / (2 + 1 / 3)) = 11 / 4 :=
by
  sorry

end fraction_simplification_l1399_139991


namespace mary_earns_per_home_l1399_139978

theorem mary_earns_per_home :
  let total_earned := 12696
  let homes_cleaned := 276.0
  total_earned / homes_cleaned = 46 :=
by
  sorry

end mary_earns_per_home_l1399_139978


namespace solve_for_x_l1399_139925

variable {x y : ℝ}

theorem solve_for_x (h1 : y = (x^2 - 9) / (x - 3)) (h2 : y = 3 * x) : x = 3 / 2 := by
  sorry

end solve_for_x_l1399_139925


namespace ensure_two_of_each_l1399_139950

theorem ensure_two_of_each {A B : ℕ} (hA : A = 10) (hB : B = 10) :
  ∃ n : ℕ, n = 12 ∧
  ∀ (extracted : ℕ → ℕ),
    (extracted 0 + extracted 1 = n) →
    (extracted 0 ≥ 2 ∧ extracted 1 ≥ 2) :=
by
  sorry

end ensure_two_of_each_l1399_139950


namespace second_player_always_wins_l1399_139993

open Nat

theorem second_player_always_wins (cards : Finset ℕ) (h_card_count : cards.card = 16) :
  ∃ strategy : ℕ → ℕ, ∀ total_score : ℕ,
  total_score ≤ 22 → (total_score + strategy total_score > 22 ∨ 
  (∃ next_score : ℕ, total_score + next_score ≤ 22 ∧ strategy (total_score + next_score) = 1)) :=
sorry

end second_player_always_wins_l1399_139993


namespace sin_inequality_in_triangle_l1399_139967

theorem sin_inequality_in_triangle (A B C : ℝ) (h_sum : A + B + C = Real.pi) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) :
  Real.sin A * Real.sin (A / 2) + Real.sin B * Real.sin (B / 2) + Real.sin C * Real.sin (C / 2) ≤ (4 * Real.sqrt 3) / 3 :=
by
  sorry

end sin_inequality_in_triangle_l1399_139967


namespace range_of_m_l1399_139920

noncomputable def f (x m : ℝ) : ℝ :=
if x < 0 then (x - m) ^ 2 - 2 else 2 * x ^ 3 - 3 * x ^ 2

theorem range_of_m (m : ℝ) : (∃ x : ℝ, f x m = -1) ↔ m ≥ 1 :=
by
  sorry

end range_of_m_l1399_139920


namespace int_power_sum_is_integer_l1399_139929

def is_integer (x : ℝ) : Prop := ∃ n : ℤ, x = n

theorem int_power_sum_is_integer {x : ℝ} (h : is_integer (x + 1/x)) (n : ℤ) : is_integer (x^n + 1/x^n) :=
by
  sorry

end int_power_sum_is_integer_l1399_139929


namespace solve_quadratics_l1399_139903

theorem solve_quadratics :
  (∃ x : ℝ, x^2 + 5 * x - 24 = 0) ∧ (∃ y, y^2 + 5 * y - 24 = 0) ∧
  (∃ z : ℝ, 3 * z^2 + 2 * z - 4 = 0) ∧ (∃ w, 3 * w^2 + 2 * w - 4 = 0) :=
by {
  sorry
}

end solve_quadratics_l1399_139903


namespace idempotent_elements_are_zero_l1399_139939

-- Definitions based on conditions specified in the problem
variables {R : Type*} [Ring R] [CharZero R]
variable {e f g : R}

def idempotent (x : R) : Prop := x * x = x

-- The theorem to be proved
theorem idempotent_elements_are_zero (h_e : idempotent e) (h_f : idempotent f) (h_g : idempotent g) (h_sum : e + f + g = 0) : 
  e = 0 ∧ f = 0 ∧ g = 0 := 
sorry

end idempotent_elements_are_zero_l1399_139939


namespace problem_l1399_139936

theorem problem (x : ℝ) (h : x + 2 / x = 4) : - (5 * x) / (x^2 + 2) = -5 / 4 := 
sorry

end problem_l1399_139936


namespace locus_of_p_ratio_distances_l1399_139970

theorem locus_of_p_ratio_distances :
  (∀ (P : ℝ × ℝ), (dist P (1, 0) = (1 / 3) * abs (P.1 - 9)) →
  (P.1^2 / 9 + P.2^2 / 8 = 1)) :=
by
  sorry

end locus_of_p_ratio_distances_l1399_139970


namespace greatest_area_difference_l1399_139924

theorem greatest_area_difference :
  ∃ (l1 w1 l2 w2 : ℕ), 2 * l1 + 2 * w1 = 200 ∧ 2 * l2 + 2 * w2 = 200 ∧
  (l1 * w1 - l2 * w2 = 2401) :=
by
  sorry

end greatest_area_difference_l1399_139924


namespace theta_in_third_quadrant_l1399_139980

theorem theta_in_third_quadrant (θ : ℝ) (h1 : Real.tan θ > 0) (h2 : Real.sin θ < 0) : 
  ∃ q : ℕ, q = 3 := 
sorry

end theta_in_third_quadrant_l1399_139980


namespace not_divisible_by_4_8_16_32_l1399_139914

def x := 80 + 112 + 144 + 176 + 304 + 368 + 3248 + 17

theorem not_divisible_by_4_8_16_32 : 
  ¬ (x % 4 = 0) ∧ ¬ (x % 8 = 0) ∧ ¬ (x % 16 = 0) ∧ ¬ (x % 32 = 0) := 
by 
  sorry

end not_divisible_by_4_8_16_32_l1399_139914


namespace remainder_when_3m_div_by_5_l1399_139926

variable (m k : ℤ)

theorem remainder_when_3m_div_by_5 (h : m % 5 = 2) : (3 * m) % 5 = 1 := by
  sorry

end remainder_when_3m_div_by_5_l1399_139926


namespace kenya_peanuts_correct_l1399_139901

def jose_peanuts : ℕ := 85
def kenya_more_peanuts : ℕ := 48

def kenya_peanuts : ℕ := jose_peanuts + kenya_more_peanuts

theorem kenya_peanuts_correct : kenya_peanuts = 133 := by
  sorry

end kenya_peanuts_correct_l1399_139901


namespace minimum_value_l1399_139935

theorem minimum_value (a : ℝ) (h₀ : 0 < a) (h₁ : a < 3) :
  ∃ a : ℝ, (0 < a ∧ a < 3) ∧ (1 / a + 4 / (8 - a) = 9 / 8) := by
sorry

end minimum_value_l1399_139935


namespace honda_cars_in_city_l1399_139941

variable (H N : ℕ)

theorem honda_cars_in_city (total_cars : ℕ)
                         (total_red_car_ratio : ℚ)
                         (honda_red_car_ratio : ℚ)
                         (non_honda_red_car_ratio : ℚ)
                         (total_red_cars : ℕ)
                         (h : total_cars = 9000)
                         (h1 : total_red_car_ratio = 0.6)
                         (h2 : honda_red_car_ratio = 0.9)
                         (h3 : non_honda_red_car_ratio = 0.225)
                         (h4 : total_red_cars = 5400)
                         (h5 : H + N = total_cars)
                         (h6 : honda_red_car_ratio * H + non_honda_red_car_ratio * N = total_red_cars) :
  H = 5000 := by
  -- Proof goes here
  sorry

end honda_cars_in_city_l1399_139941


namespace non_empty_subsets_count_l1399_139994

def odd_set : Finset ℕ := {1, 3, 5, 7, 9}
def even_set : Finset ℕ := {2, 4, 6, 8}

noncomputable def num_non_empty_subsets_odd : ℕ := 2 ^ odd_set.card - 1
noncomputable def num_non_empty_subsets_even : ℕ := 2 ^ even_set.card - 1

theorem non_empty_subsets_count :
  num_non_empty_subsets_odd + num_non_empty_subsets_even = 46 :=
by sorry

end non_empty_subsets_count_l1399_139994


namespace probability_correct_l1399_139997

-- Define the conditions of the problem
def total_white_balls : ℕ := 6
def total_black_balls : ℕ := 5
def total_balls : ℕ := total_white_balls + total_black_balls
def total_ways_draw_two_balls : ℕ := Nat.choose total_balls 2
def ways_choose_one_white_ball : ℕ := Nat.choose total_white_balls 1
def ways_choose_one_black_ball : ℕ := Nat.choose total_black_balls 1
def total_successful_outcomes : ℕ := ways_choose_one_white_ball * ways_choose_one_black_ball

-- Define the probability calculation
def probability_drawing_one_white_one_black : ℚ := total_successful_outcomes / total_ways_draw_two_balls

-- State the theorem
theorem probability_correct :
  probability_drawing_one_white_one_black = 6 / 11 :=
by
  sorry

end probability_correct_l1399_139997


namespace complex_multiplication_l1399_139947

theorem complex_multiplication (i : ℂ) (hi : i^2 = -1) : (1 + i) * (1 - i) = 1 := 
by
  sorry

end complex_multiplication_l1399_139947


namespace largest_hexagon_angle_l1399_139968

-- We define the conditions first
def angle_ratios (x : ℝ) := [3*x, 3*x, 3*x, 4*x, 5*x, 6*x]
def sum_of_angles (angles : List ℝ) := angles.sum = 720

-- Now we state our proof goal
theorem largest_hexagon_angle :
  ∀ (x : ℝ), sum_of_angles (angle_ratios x) → 6 * x = 180 :=
by
  intro x
  intro h
  sorry

end largest_hexagon_angle_l1399_139968


namespace birds_on_fence_l1399_139953

theorem birds_on_fence (B S : ℕ): 
  S = 3 →
  S + 6 = B + 5 →
  B = 4 :=
by
  intros h1 h2
  sorry

end birds_on_fence_l1399_139953


namespace c_completes_in_three_days_l1399_139951

variables (r_A r_B r_C : ℝ)
variables (h1 : r_A + r_B = 1/3)
variables (h2 : r_B + r_C = 1/3)
variables (h3 : r_A + r_C = 2/3)

theorem c_completes_in_three_days : 1 / r_C = 3 :=
by sorry

end c_completes_in_three_days_l1399_139951


namespace number_of_zeros_f_l1399_139996

noncomputable def f (x : ℝ) : ℝ := Real.log x - x^2 + 2 * x + 5

theorem number_of_zeros_f : 
  (∃ a b : ℝ, f a = 0 ∧ f b = 0 ∧ 0 < a ∧ 0 < b ∧ a ≠ b) ∧ ∀ c, f c = 0 → c = a ∨ c = b :=
by
  sorry

end number_of_zeros_f_l1399_139996


namespace union_of_sets_l1399_139949

-- Definition for set M
def M : Set ℝ := {x | x^2 - 4 * x + 3 < 0}

-- Definition for set N
def N : Set ℝ := {x | 2 * x + 1 < 5}

-- The theorem linking M and N
theorem union_of_sets : M ∪ N = {x | x < 3} :=
by
  -- Proof goes here
  sorry

end union_of_sets_l1399_139949


namespace range_of_constant_c_in_quadrant_I_l1399_139943

theorem range_of_constant_c_in_quadrant_I (c : ℝ) (x y : ℝ)
  (h1 : x - 2 * y = 4)
  (h2 : 2 * c * x + y = 5)
  (hx_pos : x > 0)
  (hy_pos : y > 0) : 
  -1 / 4 < c ∧ c < 5 / 8 := 
sorry

end range_of_constant_c_in_quadrant_I_l1399_139943


namespace selling_price_to_achieve_profit_l1399_139919

theorem selling_price_to_achieve_profit :
  ∃ (x : ℝ), let original_price := 210
              let purchase_price := 190
              let avg_sales_initial := 8
              let profit_goal := 280
              (210 - x = 200) ∧
              let profit_per_item := original_price - purchase_price - x
              let avg_sales_quantity := avg_sales_initial + 2 * x
              profit_per_item * avg_sales_quantity = profit_goal := by
  sorry

end selling_price_to_achieve_profit_l1399_139919


namespace time_for_b_and_d_together_l1399_139922

theorem time_for_b_and_d_together :
  let A_rate := 1 / 3
  let D_rate := 1 / 4
  (∃ B_rate C_rate : ℚ,
    B_rate + C_rate = 1 / 3 ∧
    A_rate + C_rate = 1 / 2 ∧
    1 / (B_rate + D_rate) = 2.4) :=
  
by
  let A_rate := 1 / 3
  let D_rate := 1 / 4
  use 1 / 6, 1 / 6
  sorry

end time_for_b_and_d_together_l1399_139922


namespace speed_of_man_l1399_139928

-- Define all given conditions and constants

def trainLength : ℝ := 110 -- in meters
def trainSpeed : ℝ := 40 -- in km/hr
def timeToPass : ℝ := 8.799296056315494 -- in seconds

-- We want to prove that the speed of the man is approximately 4.9968 km/hr
theorem speed_of_man :
  let trainSpeedMS := trainSpeed * (1000 / 3600)
  let relativeSpeed := trainLength / timeToPass
  let manSpeedMS := relativeSpeed - trainSpeedMS
  let manSpeedKMH := manSpeedMS * (3600 / 1000)
  abs (manSpeedKMH - 4.9968) < 0.01 := sorry

end speed_of_man_l1399_139928


namespace tom_split_number_of_apples_l1399_139906

theorem tom_split_number_of_apples
    (S : ℕ)
    (h1 : S = 8 * A)
    (h2 : A * 5 / 8 / 2 = 5) :
    A = 2 :=
by
  sorry

end tom_split_number_of_apples_l1399_139906


namespace inequality_multiplication_l1399_139956

theorem inequality_multiplication (a b : ℝ) (h : a > b) : 3 * a > 3 * b :=
sorry

end inequality_multiplication_l1399_139956


namespace factorization_example_l1399_139955

theorem factorization_example :
  (x : ℝ) → (x^2 + 6 * x + 9 = (x + 3)^2) :=
by
  sorry

end factorization_example_l1399_139955


namespace prove_real_roots_and_find_m_l1399_139916

-- Condition: The quadratic equation
def quadratic_eq (m x : ℝ) : Prop := x^2 - (m-1)*x + m-2 = 0

-- Condition: Discriminant
def discriminant (m : ℝ) : ℝ := (m-3)^2

-- Define the problem as a proposition
theorem prove_real_roots_and_find_m (m : ℝ) :
  (discriminant m ≥ 0) ∧ 
  (|3 - m| = 3 → (m = 0 ∨ m = 6)) :=
by
  sorry

end prove_real_roots_and_find_m_l1399_139916


namespace unique_quantities_not_determinable_l1399_139900

noncomputable def impossible_to_determine_unique_quantities 
(x y : ℝ) : Prop :=
  let acid1 := 54 * 0.35
  let acid2 := 48 * 0.25
  ∀ (final_acid : ℝ), ¬(0.35 * x + 0.25 * y = final_acid ∧ final_acid = 0.75 * (x + y))

theorem unique_quantities_not_determinable :
  impossible_to_determine_unique_quantities 54 48 :=
by
  sorry

end unique_quantities_not_determinable_l1399_139900


namespace annie_accident_chance_l1399_139971

def temperature_effect (temp: ℤ) : ℚ := ((32 - temp) / 3 * 5)

def road_condition_effect (condition: ℚ) : ℚ := condition

def wind_speed_effect (speed: ℤ) : ℚ := if (speed > 20) then ((speed - 20) / 10 * 3) else 0

def skid_chance (temp: ℤ) (condition: ℚ) (speed: ℤ) : ℚ :=
  temperature_effect temp + road_condition_effect condition + wind_speed_effect speed

def accident_chance (skid_chance: ℚ) (tire_effect: ℚ) : ℚ :=
  skid_chance * tire_effect

theorem annie_accident_chance :
  (temperature_effect 8 + road_condition_effect 15 + wind_speed_effect 35) * 0.75 = 43.5 :=
by sorry

end annie_accident_chance_l1399_139971


namespace count_perfect_fourth_powers_l1399_139974

theorem count_perfect_fourth_powers: 
  ∃ n_count: ℕ, n_count = 4 ∧ ∀ n: ℕ, (50 ≤ n^4 ∧ n^4 ≤ 2000) → (n = 3 ∨ n = 4 ∨ n = 5 ∨ n = 6) :=
by {
  sorry
}

end count_perfect_fourth_powers_l1399_139974


namespace problem_solution_l1399_139987

theorem problem_solution (x y z : ℝ) (h1 : x * y + y * z + z * x = 4) (h2 : x * y * z = 6) :
  (x * y - (3 / 2) * (x + y)) * (y * z - (3 / 2) * (y + z)) * (z * x - (3 / 2) * (z + x)) = 81 / 4 :=
by
  sorry

end problem_solution_l1399_139987


namespace min_value_expr_l1399_139963

theorem min_value_expr (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) : 
  ∃ x : ℝ, x = 6 ∧ x = (2 * a + b) / c + (2 * a + c) / b + (2 * b + c) / a :=
by
  sorry

end min_value_expr_l1399_139963


namespace find_pair_l1399_139957

noncomputable def x_n (n : ℕ) : ℝ := n / (n + 2016)

theorem find_pair :
  ∃ (m n : ℕ), x_n 2016 = (x_n m) * (x_n n) ∧ (m = 6048 ∧ n = 4032) :=
by {
  sorry
}

end find_pair_l1399_139957


namespace mean_score_is_74_l1399_139962

theorem mean_score_is_74 (σ q : ℝ)
  (h1 : 58 = q - 2 * σ)
  (h2 : 98 = q + 3 * σ) :
  q = 74 :=
by
  sorry

end mean_score_is_74_l1399_139962


namespace coeff_x5_in_expansion_l1399_139944

noncomputable def binomial_expansion_coeff (n k : ℕ) (x : ℝ) : ℝ :=
  Real.sqrt x ^ (n - k) * 2 ^ k * (Nat.choose n k)

theorem coeff_x5_in_expansion :
  (binomial_expansion_coeff 12 2 x) = 264 :=
by
  sorry

end coeff_x5_in_expansion_l1399_139944


namespace pratt_certificate_space_bound_l1399_139958

-- Define the Pratt certificate space function λ(p)
noncomputable def pratt_space (p : ℕ) : ℝ := sorry

-- Define the log_2 function (if not already available in Mathlib)
noncomputable def log2 (x : ℝ) : ℝ := sorry

-- Assuming that p is a prime number
variable {p : ℕ} (hp : Nat.Prime p)

-- The proof problem
theorem pratt_certificate_space_bound (hp : Nat.Prime p) :
  pratt_space p ≤ 6 * (log2 p) ^ 2 := 
sorry

end pratt_certificate_space_bound_l1399_139958
