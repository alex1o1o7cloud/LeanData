import Mathlib

namespace interval_of_decrease_for_f_l454_45456

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 2 * x - 3)

def decreasing_interval (s : Set ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ s → y ∈ s → x < y → f y < f x

theorem interval_of_decrease_for_f :
  decreasing_interval {x : ℝ | x < -1} f :=
by
  sorry

end interval_of_decrease_for_f_l454_45456


namespace sequence_a4_value_l454_45402

theorem sequence_a4_value : 
  ∀ (a : ℕ → ℕ), a 1 = 2 → (∀ n, n ≥ 2 → a n = a (n - 1) + n) → a 4 = 11 :=
by
  sorry

end sequence_a4_value_l454_45402


namespace min_points_tenth_game_l454_45484

-- Defining the scores for each segment of games
def first_five_games : List ℕ := [18, 15, 13, 17, 19]
def next_four_games : List ℕ := [14, 20, 12, 21]

-- Calculating the total score after 9 games
def total_score_after_nine_games : ℕ := first_five_games.sum + next_four_games.sum

-- Defining the required total points after 10 games for an average greater than 17
def required_total_points := 171

-- Proving the number of points needed in the 10th game
theorem min_points_tenth_game (s₁ s₂ : List ℕ) (h₁ : s₁ = first_five_games) (h₂ : s₂ = next_four_games) :
    s₁.sum + s₂.sum + x ≥ required_total_points → x ≥ 22 :=
  sorry

end min_points_tenth_game_l454_45484


namespace unique_necklace_arrangements_l454_45473

-- Definitions
def num_beads : Nat := 7

def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- The number of unique ways to arrange the beads on a necklace
-- considering rotations and reflections
theorem unique_necklace_arrangements : (factorial num_beads) / (num_beads * 2) = 360 := 
by
  sorry

end unique_necklace_arrangements_l454_45473


namespace find_f_minus_half_l454_45470

-- Definitions
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def function_definition (f : ℝ → ℝ) : Prop :=
  ∀ x, x > 0 → f x = 4^x

-- Theorem statement
theorem find_f_minus_half {f : ℝ → ℝ}
  (h_odd : is_odd_function f)
  (h_def : function_definition f) :
  f (-1/2) = -2 :=
by
  sorry

end find_f_minus_half_l454_45470


namespace range_of_a_l454_45488

-- Define the function f
def f (x a : ℝ) : ℝ := |x - 1| + |x - a|

-- State the problem
theorem range_of_a (a : ℝ) : (∀ x : ℝ, f x a ≥ 2) → (a ≤ -1 ∨ a ≥ 3) :=
by 
  sorry

end range_of_a_l454_45488


namespace prob_point_in_region_l454_45483

theorem prob_point_in_region :
  let rect_area := 18
  let intersect_area := 15 / 2
  let probability := intersect_area / rect_area
  probability = 5 / 12 :=
by
  sorry

end prob_point_in_region_l454_45483


namespace ratio_norm_lisa_l454_45435

-- Define the number of photos taken by each photographer.
variable (L M N : ℕ)

-- Given conditions
def norm_photos : Prop := N = 110
def photo_sum_condition : Prop := L + M = M + N - 60

-- Prove the ratio of Norm's photos to Lisa's photos.
theorem ratio_norm_lisa (h1 : norm_photos N) (h2 : photo_sum_condition L M N) : N / L = 11 / 5 := 
by
  sorry

end ratio_norm_lisa_l454_45435


namespace total_songs_performed_l454_45486

theorem total_songs_performed :
  ∃ N : ℕ, 
  (∃ e d o : ℕ, 
     (e > 3 ∧ e < 9) ∧ (d > 3 ∧ d < 9) ∧ (o > 3 ∧ o < 9)
      ∧ N = (9 + 3 + e + d + o) / 4) ∧ N = 6 :=
sorry

end total_songs_performed_l454_45486


namespace arithmetic_sequence_k_l454_45449

theorem arithmetic_sequence_k (d : ℤ) (h_d : d ≠ 0) (a : ℕ → ℤ) 
  (h_seq : ∀ n, a n = 0 + n * d) (h_k : a 21 = a 0 + a 1 + a 2 + a 3 + a 4 + a 5 + a 6):
  21 = 21 :=
by
  -- This would be the problem setup
  -- The proof would go here
  sorry

end arithmetic_sequence_k_l454_45449


namespace third_pipe_empty_time_l454_45430

theorem third_pipe_empty_time :
  let A_rate := 1/60
  let B_rate := 1/75
  let combined_rate := 1/50
  let third_pipe_rate := combined_rate - (A_rate + B_rate)
  let time_to_empty := 1 / third_pipe_rate
  time_to_empty = 100 :=
by
  sorry

end third_pipe_empty_time_l454_45430


namespace total_lawns_mowed_l454_45457

theorem total_lawns_mowed (earned_per_lawn forgotten_lawns total_earned : ℕ) 
    (h1 : earned_per_lawn = 9) 
    (h2 : forgotten_lawns = 8) 
    (h3 : total_earned = 54) : 
    ∃ (total_lawns : ℕ), total_lawns = 14 :=
by
    sorry

end total_lawns_mowed_l454_45457


namespace chord_length_l454_45433

theorem chord_length (t : ℝ) :
  (∃ x y, x = 1 + 2 * t ∧ y = 2 + t ∧ x ^ 2 + y ^ 2 = 9) →
  ((1.8 - (-3)) ^ 2 + (2.4 - 0) ^ 2 = (12 / 5 * Real.sqrt 5) ^ 2) :=
by
  sorry

end chord_length_l454_45433


namespace find_usual_time_l454_45489

variable (R T : ℝ)

theorem find_usual_time
  (h_condition :  R * T = (9 / 8) * R * (T - 4)) :
  T = 36 :=
by
  sorry

end find_usual_time_l454_45489


namespace pens_distributed_evenly_l454_45442

theorem pens_distributed_evenly (S : ℕ) (P : ℕ) (pencils : ℕ) 
  (hS : S = 10) (hpencils : pencils = 920) 
  (h_pencils_distributed : pencils % S = 0) 
  (h_pens_distributed : P % S = 0) : 
  ∃ k : ℕ, P = 10 * k :=
by 
  sorry

end pens_distributed_evenly_l454_45442


namespace students_in_grade6_l454_45406

noncomputable def num_students_total : ℕ := 100
noncomputable def num_students_grade4 : ℕ := 30
noncomputable def num_students_grade5 : ℕ := 35
noncomputable def num_students_grade6 : ℕ := num_students_total - (num_students_grade4 + num_students_grade5)

theorem students_in_grade6 : num_students_grade6 = 35 := by
  sorry

end students_in_grade6_l454_45406


namespace negation_of_existence_l454_45490

theorem negation_of_existence (m : ℤ) :
  (¬ ∃ x : ℤ, x^2 + 2*x + m ≤ 0) ↔ (∀ x : ℤ, x^2 + 2*x + m > 0) :=
by
  sorry

end negation_of_existence_l454_45490


namespace perpendicular_line_through_point_l454_45412

noncomputable def is_perpendicular (m₁ m₂ : ℝ) : Prop :=
  m₁ * m₂ = -1

theorem perpendicular_line_through_point
  (line : ℝ → ℝ)
  (P : ℝ × ℝ)
  (h_line_eq : ∀ x, line x = 3 * x + 8)
  (hP : P = (2,1)) :
  ∃ a b c : ℝ, a * (P.1) + b * (P.2) + c = 0 ∧ is_perpendicular 3 (-b / a) ∧ a * 1 + b * 3 + c = 0 :=
sorry

end perpendicular_line_through_point_l454_45412


namespace necessary_but_not_sufficient_for_gt_one_l454_45460

variable (x : ℝ)

theorem necessary_but_not_sufficient_for_gt_one (h : x^2 > 1) : ¬(x^2 > 1 ↔ x > 1) ∧ (x > 1 → x^2 > 1) :=
by
  sorry

end necessary_but_not_sufficient_for_gt_one_l454_45460


namespace total_distance_traveled_l454_45477

def trip_duration : ℕ := 8
def speed_first_half : ℕ := 70
def speed_second_half : ℕ := 85
def time_each_half : ℕ := trip_duration / 2

theorem total_distance_traveled :
  let distance_first_half := time_each_half * speed_first_half
  let distance_second_half := time_each_half * speed_second_half
  let total_distance := distance_first_half + distance_second_half
  total_distance = 620 := by
  sorry

end total_distance_traveled_l454_45477


namespace hyperbola_eccentricity_range_l454_45482

theorem hyperbola_eccentricity_range
  (a b t : ℝ)
  (h_a_pos : a > 0)
  (h_b_pos : b > 0)
  (h_condition : a > b) :
  ∃ e : ℝ, e = Real.sqrt (1 + (b / a)^2) ∧ 1 < e ∧ e < Real.sqrt 2 :=
by
  sorry

end hyperbola_eccentricity_range_l454_45482


namespace triangle_problem_l454_45410

theorem triangle_problem (n : ℕ) (h : 1 < n ∧ n < 4) : n = 2 ∨ n = 3 :=
by
  -- Valid realizability proof omitted
  sorry

end triangle_problem_l454_45410


namespace dividend_is_5336_l454_45475

theorem dividend_is_5336 (D Q R : ℕ) (h1 : D = 10 * Q) (h2 : D = 5 * R) (h3 : R = 46) :
  (D * Q + R) = 5336 :=
by {
  sorry
}

end dividend_is_5336_l454_45475


namespace real_number_set_condition_l454_45478

theorem real_number_set_condition (x : ℝ) :
  (x ≠ 1) ∧ (x^2 - x ≠ 1) ∧ (x^2 - x ≠ x) →
  x ≠ 0 ∧ x ≠ 1 ∧ x ≠ 2 ∧ x ≠ (1 + Real.sqrt 5) / 2 ∧ x ≠ (1 - Real.sqrt 5) / 2 := 
by
  sorry

end real_number_set_condition_l454_45478


namespace problem_1_problem_2_l454_45462

theorem problem_1 (α : ℝ) (hα : Real.tan α = 2) :
  Real.tan (α + Real.pi / 4) = -3 :=
by
  sorry

theorem problem_2 (α : ℝ) (hα : Real.tan α = 2) :
  (6 * Real.sin α + Real.cos α) / (3 * Real.sin α - 2 * Real.cos α) = 13 / 4 :=
by
  sorry

end problem_1_problem_2_l454_45462


namespace blue_pill_cost_l454_45414

theorem blue_pill_cost :
  ∃ y : ℝ, ∀ (red_pill_cost blue_pill_cost : ℝ),
    (blue_pill_cost = red_pill_cost + 2) ∧
    (21 * (blue_pill_cost + red_pill_cost) = 819) →
    blue_pill_cost = 20.5 :=
by sorry

end blue_pill_cost_l454_45414


namespace average_side_length_of_squares_l454_45431

noncomputable def side_length (area : ℝ) : ℝ :=
  Real.sqrt area

noncomputable def average_side_length (areas : List ℝ) : ℝ :=
  (areas.map side_length).sum / (areas.length : ℝ)

theorem average_side_length_of_squares :
  average_side_length [25, 64, 144] = 25 / 3 :=
by
  sorry

end average_side_length_of_squares_l454_45431


namespace part_one_part_two_l454_45441

-- Defining the function and its first derivative
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x
def f' (a b : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + b

-- Part (Ⅰ)
theorem part_one (a b : ℝ)
  (H1 : f' a b 3 = 24)
  (H2 : f' a b 1 = 0) :
  a = 1 ∧ b = -3 ∧ (∀ x, -1 ≤ x ∧ x ≤ 1 → f' 1 (-3) x ≤ 0) :=
sorry

-- Part (Ⅱ)
theorem part_two (b : ℝ)
  (H1 : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → 3 * x^2 + b ≤ 0) :
  b ≤ -3 :=
sorry

end part_one_part_two_l454_45441


namespace cone_volume_l454_45487

theorem cone_volume (R h : ℝ) (hR : 0 ≤ R) (hh : 0 ≤ h) : 
  (∫ x in (0 : ℝ)..h, π * (R / h * x)^2) = (1 / 3) * π * R^2 * h :=
by
  sorry

end cone_volume_l454_45487


namespace Faye_crayons_l454_45495

theorem Faye_crayons (rows crayons_per_row : ℕ) (h_rows : rows = 7) (h_crayons_per_row : crayons_per_row = 30) : rows * crayons_per_row = 210 :=
by
  sorry

end Faye_crayons_l454_45495


namespace bacteria_growth_returns_six_l454_45443

theorem bacteria_growth_returns_six (n : ℕ) (h : (4 * 2 ^ n > 200)) : n = 6 :=
sorry

end bacteria_growth_returns_six_l454_45443


namespace ratio_of_games_played_to_losses_l454_45422

-- Definitions based on the conditions
def total_games_played : ℕ := 10
def games_won : ℕ := 5
def games_lost : ℕ := total_games_played - games_won

-- The proof problem
theorem ratio_of_games_played_to_losses : (total_games_played / Nat.gcd total_games_played games_lost) = 2 ∧ (games_lost / Nat.gcd total_games_played games_lost) = 1 :=
by
  sorry

end ratio_of_games_played_to_losses_l454_45422


namespace correct_option_l454_45428

-- Definitions for universe set, and subsets A and B
def S : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {2, 5}

-- The proof goal
theorem correct_option : A ⊆ S \ B :=
by
  sorry

end correct_option_l454_45428


namespace pieces_length_l454_45438

theorem pieces_length (L M S : ℝ) (h1 : L + M + S = 180)
  (h2 : L = M + S + 30)
  (h3 : M = L / 2 - 10) :
  L = 105 ∧ M = 42.5 ∧ S = 32.5 :=
by
  sorry

end pieces_length_l454_45438


namespace values_of_x_for_g_l454_45498

def g (x : ℝ) : ℝ := x^2 - 5 * x

theorem values_of_x_for_g (x : ℝ) :
  g (g x) = g x ↔ x = 0 ∨ x = 5 ∨ x = 6 ∨ x = -1 := by
    sorry

end values_of_x_for_g_l454_45498


namespace little_johns_money_left_l454_45455

def J_initial : ℝ := 7.10
def S : ℝ := 1.05
def F : ℝ := 1.00

theorem little_johns_money_left :
  J_initial - (S + 2 * F) = 4.05 :=
by sorry

end little_johns_money_left_l454_45455


namespace greatest_multiple_l454_45411

theorem greatest_multiple (n : ℕ) (h1 : n < 1000) (h2 : n % 5 = 0) (h3 : n % 6 = 0) : n = 990 :=
sorry

end greatest_multiple_l454_45411


namespace find_value_l454_45468

theorem find_value (x y : ℚ) (hx : x = 5 / 7) (hy : y = 7 / 5) :
  (1 / 3 * x^8 * y^9 + 1 / 7) = 64 / 105 := by
  sorry

end find_value_l454_45468


namespace find_minimum_value_l454_45464

noncomputable def fixed_point_at_2_2 (a : ℝ) (ha_pos : a > 0) (ha_ne : a ≠ 1) : Prop :=
∀ (x : ℝ), a^(2-x) + 1 = 2 ↔ x = 2

noncomputable def point_on_line (m n : ℝ) (hmn_pos : m * n > 0) : Prop :=
2 * m + 2 * n = 1

theorem find_minimum_value (m n : ℝ) (hmn_pos : m * n > 0) :
  (fixed_point_at_2_2 a ha_pos ha_ne) → (point_on_line m n hmn_pos) → (1/m + 1/n ≥ 8) :=
sorry

end find_minimum_value_l454_45464


namespace two_squares_always_similar_l454_45400

-- Define geometric shapes and their properties
inductive Shape
| Rectangle : Shape
| Rhombus   : Shape
| Square    : Shape
| RightAngledTriangle : Shape

-- Define similarity condition
def similar (s1 s2 : Shape) : Prop :=
  match s1, s2 with
  | Shape.Square, Shape.Square => true
  | _, _ => false

-- Prove that two squares are always similar
theorem two_squares_always_similar : similar Shape.Square Shape.Square = true :=
by
  sorry

end two_squares_always_similar_l454_45400


namespace needed_correct_to_pass_l454_45426

def total_questions : Nat := 120
def genetics_questions : Nat := 20
def ecology_questions : Nat := 50
def evolution_questions : Nat := 50

def correct_genetics : Nat := (60 * genetics_questions) / 100
def correct_ecology : Nat := (50 * ecology_questions) / 100
def correct_evolution : Nat := (70 * evolution_questions) / 100
def total_correct : Nat := correct_genetics + correct_ecology + correct_evolution

def passing_rate : Nat := 65
def passing_score : Nat := (passing_rate * total_questions) / 100

theorem needed_correct_to_pass : (passing_score - total_correct) = 6 := 
by
  sorry

end needed_correct_to_pass_l454_45426


namespace find_z_l454_45448

-- Definitions of the conditions
def equation_1 (x y : ℝ) : Prop := x^2 - 3 * x + 6 = y - 10
def equation_2 (y z : ℝ) : Prop := y = 2 * z
def x_value (x : ℝ) : Prop := x = -5

-- Lean theorem statement
theorem find_z (x y z : ℝ) (h1 : equation_1 x y) (h2 : equation_2 y z) (h3 : x_value x) : z = 28 :=
sorry

end find_z_l454_45448


namespace line_increase_is_110_l454_45427

noncomputable def original_lines (increased_lines : ℕ) (percentage_increase : ℚ) : ℚ :=
  increased_lines / (1 + percentage_increase)

theorem line_increase_is_110
  (L' : ℕ)
  (percentage_increase : ℚ)
  (hL' : L' = 240)
  (hp : percentage_increase = 0.8461538461538461) :
  L' - original_lines L' percentage_increase = 110 :=
by
  sorry

end line_increase_is_110_l454_45427


namespace right_triangle_perimeter_l454_45420

-- Conditions
variable (a : ℝ) (b : ℝ) (c : ℝ)
variable (h_area : 1 / 2 * 15 * b = 150)
variable (h_pythagorean : a^2 + b^2 = c^2)
variable (h_a : a = 15)

-- The theorem to prove the perimeter is 60 units
theorem right_triangle_perimeter : a + b + c = 60 := by
  sorry

end right_triangle_perimeter_l454_45420


namespace population_difference_l454_45485

variable (A B C : ℝ)

-- Conditions
def population_condition (A B C : ℝ) : Prop := A + B = B + C + 5000

-- The proof statement
theorem population_difference (h : population_condition A B C) : A - C = 5000 :=
by sorry

end population_difference_l454_45485


namespace least_x_divisibility_l454_45439

theorem least_x_divisibility :
  ∃ x : ℕ, (x > 0) ∧ ((x^2 + 164) % 3 = 0) ∧ ((x^2 + 164) % 4 = 0) ∧ ((x^2 + 164) % 5 = 0) ∧
  ((x^2 + 164) % 6 = 0) ∧ ((x^2 + 164) % 7 = 0) ∧ ((x^2 + 164) % 8 = 0) ∧ 
  ((x^2 + 164) % 9 = 0) ∧ ((x^2 + 164) % 10 = 0) ∧ ((x^2 + 164) % 11 = 0) ∧ x = 166 → 
  3 = 3 :=
by
  sorry

end least_x_divisibility_l454_45439


namespace calculate_binom_l454_45424

theorem calculate_binom : 2 * Nat.choose 30 3 = 8120 := 
by 
  sorry

end calculate_binom_l454_45424


namespace valid_password_count_l454_45444

/-- 
The number of valid 4-digit ATM passwords at Fred's Bank, composed of digits from 0 to 9,
that do not start with the sequence "9,1,1" and do not end with the digit "5",
is 8991.
-/
theorem valid_password_count : 
  let total_passwords : ℕ := 10000
  let start_911 : ℕ := 10
  let end_5 : ℕ := 1000
  let start_911_end_5 : ℕ := 1
  total_passwords - (start_911 + end_5 - start_911_end_5) = 8991 :=
by
  let total_passwords : ℕ := 10000
  let start_911 : ℕ := 10
  let end_5 : ℕ := 1000
  let start_911_end_5 : ℕ := 1
  show total_passwords - (start_911 + end_5 - start_911_end_5) = 8991
  sorry

end valid_password_count_l454_45444


namespace max_cookie_price_l454_45408

theorem max_cookie_price :
  ∃ k p : ℕ, 
    (8 * k + 3 * p < 200) ∧ 
    (4 * k + 5 * p > 150) ∧
    (∀ k' p' : ℕ, (8 * k' + 3 * p' < 200) ∧ (4 * k' + 5 * p' > 150) → k' ≤ 19) :=
sorry

end max_cookie_price_l454_45408


namespace winning_configurations_for_blake_l454_45436

def isWinningConfigurationForBlake (config : List ℕ) := 
  let nimSum := config.foldl (xor) 0
  nimSum = 0

theorem winning_configurations_for_blake :
  (isWinningConfigurationForBlake [8, 2, 3]) ∧ 
  (isWinningConfigurationForBlake [9, 3, 3]) ∧ 
  (isWinningConfigurationForBlake [9, 5, 2]) :=
by {
  sorry
}

end winning_configurations_for_blake_l454_45436


namespace mustard_at_first_table_l454_45432

theorem mustard_at_first_table (M : ℝ) :
  (M + 0.25 + 0.38 = 0.88) → M = 0.25 :=
by
  intro h
  sorry

end mustard_at_first_table_l454_45432


namespace sum_or_difference_div_by_100_l454_45469

theorem sum_or_difference_div_by_100 (s : Finset ℤ) (h_card : s.card = 52) :
  ∃ (a b : ℤ), a ∈ s ∧ b ∈ s ∧ (a ≠ b) ∧ (100 ∣ (a + b) ∨ 100 ∣ (a - b)) :=
by
  sorry

end sum_or_difference_div_by_100_l454_45469


namespace hair_cut_off_length_l454_45446

def initial_hair_length : ℕ := 18
def hair_length_after_haircut : ℕ := 9

theorem hair_cut_off_length :
  initial_hair_length - hair_length_after_haircut = 9 :=
sorry

end hair_cut_off_length_l454_45446


namespace students_per_bus_l454_45491

/-- The number of students who can be accommodated in each bus -/
theorem students_per_bus (total_students : ℕ) (students_in_cars : ℕ) (num_buses : ℕ) 
(h1 : total_students = 375) (h2 : students_in_cars = 4) (h3 : num_buses = 7) : 
(total_students - students_in_cars) / num_buses = 53 :=
by
  sorry

end students_per_bus_l454_45491


namespace find_eccentricity_l454_45445

noncomputable def ellipse_gamma (a b : ℝ) (ha_gt : a > 0) (hb_gt : b > 0) (h : a > b) : Prop :=
∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1

def ellipse_focus (a b : ℝ) : Prop :=
∀ (x y : ℝ), x = 3 → y = 0

def vertex_A (b : ℝ) : Prop :=
∀ (x y : ℝ), x = 0 → y = b

def vertex_B (b : ℝ) : Prop :=
∀ (x y : ℝ), x = 0 → y = -b

def point_N : Prop :=
∀ (x y : ℝ), x = 12 → y = 0

theorem find_eccentricity : 
∀ (a b : ℝ) (ha_gt : a > 0) (hb_gt : b > 0) (h : a > b), 
  ellipse_gamma a b ha_gt hb_gt h → 
  ellipse_focus a b → 
  vertex_A b → 
  vertex_B b → 
  point_N → 
  ∃ e : ℝ, e = 1 / 2 := 
by 
  sorry

end find_eccentricity_l454_45445


namespace tea_in_box_l454_45437

theorem tea_in_box (tea_per_day ounces_per_week ounces_per_box : ℝ) 
    (H1 : tea_per_day = 1 / 5) 
    (H2 : ounces_per_week = tea_per_day * 7) 
    (H3 : ounces_per_box = ounces_per_week * 20) : 
    ounces_per_box = 28 := 
by
  sorry

end tea_in_box_l454_45437


namespace total_fish_in_pond_l454_45466

theorem total_fish_in_pond (N : ℕ) (h1 : 80 ≤ N) (h2 : 5 ≤ 150) (h_marked_dist : (5 : ℚ) / 150 = (80 : ℚ) / N) : N = 2400 := by
  sorry

end total_fish_in_pond_l454_45466


namespace number_of_balls_l454_45404

theorem number_of_balls (x : ℕ) (h : x - 20 = 30 - x) : x = 25 :=
sorry

end number_of_balls_l454_45404


namespace liked_product_B_l454_45492

-- Define the conditions as assumptions
variables (X : ℝ)

-- Assumptions
axiom liked_both : 23 = 23
axiom liked_neither : 23 = 23

-- The main theorem that needs to be proven
theorem liked_product_B (X : ℝ) : ∃ Y : ℝ, Y = 100 - X :=
by sorry

end liked_product_B_l454_45492


namespace both_reunions_l454_45463

theorem both_reunions (U O H B : ℕ) 
  (hU : U = 100) 
  (hO : O = 50) 
  (hH : H = 62) 
  (attend_one : U = O + H - B) :  
  B = 12 := 
by 
  sorry

end both_reunions_l454_45463


namespace folded_paper_area_ratio_l454_45493

theorem folded_paper_area_ratio (s : ℝ) (h : s > 0) :
  let A := s^2
  let rectangle_area := (s / 2) * s
  let triangle_area := (1 / 2) * (s / 2) * s
  let folded_area := 4 * rectangle_area - triangle_area
  (folded_area / A) = 7 / 4 :=
by
  let A := s^2
  let rectangle_area := (s / 2) * s
  let triangle_area := (1 / 2) * (s / 2) * s
  let folded_area := 4 * rectangle_area - triangle_area
  show (folded_area / A) = 7 / 4
  sorry

end folded_paper_area_ratio_l454_45493


namespace angle_between_diagonals_l454_45472

open Real

theorem angle_between_diagonals
  (a b c : ℝ) :
  ∃ θ : ℝ, θ = arccos (a^2 / sqrt ((a^2 + b^2) * (a^2 + c^2))) :=
by
  -- Placeholder for the proof
  sorry

end angle_between_diagonals_l454_45472


namespace gcd_47_power5_1_l454_45440
-- Import the necessary Lean library

-- Mathematically equivalent proof problem in Lean 4
theorem gcd_47_power5_1 (a b : ℕ) (h1 : a = 47^5 + 1) (h2 : b = 47^5 + 47^3 + 1) :
  Nat.gcd a b = 1 :=
by
  sorry

end gcd_47_power5_1_l454_45440


namespace smallest_integer_is_77_l454_45416

theorem smallest_integer_is_77 
  (A B C D E F G : ℤ)
  (h_uniq: A < B ∧ B < C ∧ C < D ∧ D < E ∧ E < F ∧ F < G)
  (h_sum: A + B + C + D + E + F + G = 840)
  (h_largest: G = 190)
  (h_two_smallest_sum: A + B = 156) : 
  A = 77 :=
sorry

end smallest_integer_is_77_l454_45416


namespace carousel_rotation_time_l454_45496

-- Definitions and Conditions
variables (a v U x : ℝ)

-- Conditions given in the problem
def condition1 : Prop := (U * a - v * a = 2 * Real.pi)
def condition2 : Prop := (v * a = U * (x - a / 2))

-- Statement to prove
theorem carousel_rotation_time :
  condition1 a v U ∧ condition2 a v U x → x = 2 * a / 3 :=
by
  intro h
  have c1 := h.1
  have c2 := h.2
  sorry

end carousel_rotation_time_l454_45496


namespace inverse_proportional_p_q_l454_45494

theorem inverse_proportional_p_q (k : ℚ)
  (h1 : ∀ p q : ℚ, p * q = k)
  (h2 : (30 : ℚ) * (4 : ℚ) = k) :
  p = 12 ↔ (10 : ℚ) * p = k :=
by
  sorry

end inverse_proportional_p_q_l454_45494


namespace cindy_correct_answer_l454_45418

theorem cindy_correct_answer (x : ℕ) (h : (x - 9) / 3 = 43) : (x - 3) / 9 = 15 :=
by
sorry

end cindy_correct_answer_l454_45418


namespace minimum_number_of_odd_integers_among_six_l454_45419

theorem minimum_number_of_odd_integers_among_six : 
  ∀ (x y a b m n : ℤ), 
    x + y = 28 →
    x + y + a + b = 45 →
    x + y + a + b + m + n = 63 →
    ∃ (odd_count : ℕ), odd_count = 1 :=
by sorry

end minimum_number_of_odd_integers_among_six_l454_45419


namespace least_width_l454_45423

theorem least_width (w : ℝ) (h_nonneg : w ≥ 0) (h_area : w * (w + 10) ≥ 150) : w = 10 :=
sorry

end least_width_l454_45423


namespace conic_sections_union_l454_45425

theorem conic_sections_union :
  ∀ (x y : ℝ), (y^4 - 4*x^4 = 2*y^2 - 1) ↔ 
               (y^2 - 2*x^2 = 1) ∨ (y^2 + 2*x^2 = 1) := 
by
  sorry

end conic_sections_union_l454_45425


namespace benny_turnips_l454_45459

-- Definitions and conditions
def melanie_turnips : ℕ := 139
def total_turnips : ℕ := 252

-- Question to prove
theorem benny_turnips : ∃ b : ℕ, b = total_turnips - melanie_turnips ∧ b = 113 :=
by {
    sorry
}

end benny_turnips_l454_45459


namespace geometric_sequence_max_product_l454_45479

theorem geometric_sequence_max_product
  (b : ℕ → ℝ) (q : ℝ) (b1 : ℝ)
  (h_b1_pos : b1 > 0)
  (h_q : 0 < q ∧ q < 1)
  (h_b : ∀ n, b (n + 1) = b n * q)
  (h_b7_gt_1 : b 7 > 1)
  (h_b8_lt_1 : b 8 < 1) :
  (∀ (n : ℕ), n = 7 → b 1 * b 2 * b 3 * b 4 * b 5 * b 6 * b 7 = b 1 * b 2 * b 3 * b 4 * b 5 * b 6 * b 7) :=
by {
  sorry
}

end geometric_sequence_max_product_l454_45479


namespace coordinates_on_y_axis_l454_45417

theorem coordinates_on_y_axis (m : ℝ) (h : m + 1 = 0) : (m + 1, m + 4) = (0, 3) :=
by
  sorry

end coordinates_on_y_axis_l454_45417


namespace incorrect_inequality_l454_45403

theorem incorrect_inequality (a b c : ℝ) (h : a > b) : ¬ (forall c, a * c > b * c) :=
by
  intro h'
  have h'' := h' c
  sorry

end incorrect_inequality_l454_45403


namespace political_exam_pass_l454_45415

-- Define the students' statements.
def A_statement (C_passed : Prop) : Prop := C_passed
def B_statement (B_passed : Prop) : Prop := ¬ B_passed
def C_statement (A_statement : Prop) : Prop := A_statement

-- Define the problem conditions.
def condition_1 (A_passed B_passed C_passed : Prop) : Prop := ¬A_passed ∨ ¬B_passed ∨ ¬C_passed
def condition_2 (A_passed B_passed C_passed : Prop) := A_statement C_passed
def condition_3 (A_passed B_passed C_passed : Prop) := B_statement B_passed
def condition_4 (A_passed B_passed C_passed : Prop) := C_statement (A_statement C_passed)
def condition_5 (A_statement_true B_statement_true C_statement_true : Prop) : Prop := 
  (¬A_statement_true ∧ B_statement_true ∧ C_statement_true) ∨
  (A_statement_true ∧ ¬B_statement_true ∧ C_statement_true) ∨
  (A_statement_true ∧ B_statement_true ∧ ¬C_statement_true)

-- Define the proof problem.
theorem political_exam_pass : 
  ∀ (A_passed B_passed C_passed : Prop),
  condition_1 A_passed B_passed C_passed →
  condition_2 A_passed B_passed C_passed →
  condition_3 A_passed B_passed C_passed →
  condition_4 A_passed B_passed C_passed →
  ∃ (A_statement_true B_statement_true C_statement_true : Prop), 
  condition_5 A_statement_true B_statement_true C_statement_true →
  ¬A_passed
:= by { sorry }

end political_exam_pass_l454_45415


namespace find_x_l454_45452

def set_of_numbers := [1, 2, 4, 5, 6, 9, 9, 10]

theorem find_x {x : ℝ} (h : (set_of_numbers.sum + x) / 9 = 7) : x = 17 :=
by
  sorry

end find_x_l454_45452


namespace trig_identity_simplified_l454_45453

open Real

theorem trig_identity_simplified :
  (sin (15 * π / 180) + cos (15 * π / 180)) * (sin (15 * π / 180) - cos (15 * π / 180)) = - (sqrt 3 / 2) :=
by
  sorry

end trig_identity_simplified_l454_45453


namespace ratio_wealth_citizen_XY_l454_45407

noncomputable def wealth_ratio_XY 
  (P W : ℝ) 
  (h1 : 0 < P) 
  (h2 : 0 < W) : ℝ :=
  let pop_X := 0.4 * P
  let wealth_X_before_tax := 0.5 * W
  let tax_X := 0.1 * wealth_X_before_tax
  let wealth_X_after_tax := wealth_X_before_tax - tax_X
  let wealth_per_citizen_X := wealth_X_after_tax / pop_X

  let pop_Y := 0.3 * P
  let wealth_Y := 0.6 * W
  let wealth_per_citizen_Y := wealth_Y / pop_Y

  wealth_per_citizen_X / wealth_per_citizen_Y

theorem ratio_wealth_citizen_XY 
  (P W : ℝ) 
  (h1 : 0 < P) 
  (h2 : 0 < W) : 
  wealth_ratio_XY P W h1 h2 = 9 / 16 := 
by
  sorry

end ratio_wealth_citizen_XY_l454_45407


namespace fraction_of_square_shaded_is_half_l454_45481

theorem fraction_of_square_shaded_is_half {s : ℝ} (h : s > 0) :
  let O := (0, 0)
  let P := (0, s)
  let Q := (s, s / 2)
  let area_square := s^2
  let area_triangle_OPQ := 1 / 2 * s^2 / 2
  let shaded_area := area_square - area_triangle_OPQ
  (shaded_area / area_square) = 1 / 2 :=
by
  sorry

end fraction_of_square_shaded_is_half_l454_45481


namespace g_of_5_l454_45401

noncomputable def g (x : ℝ) : ℝ := -2 / x

theorem g_of_5 (x : ℝ) : g (g (g (g (g x)))) = -2 / x :=
by
  sorry

end g_of_5_l454_45401


namespace no_super_squarish_numbers_l454_45497

def is_super_squarish (M : ℕ) : Prop :=
  let a := M / 100000 % 100
  let b := M / 1000 % 1000
  let c := M % 100
  (M ≥ 1000000 ∧ M < 10000000) ∧
  (M % 10 ≠ 0 ∧ (M / 10) % 10 ≠ 0 ∧ (M / 100) % 10 ≠ 0 ∧ (M / 1000) % 10 ≠ 0 ∧
    (M / 10000) % 10 ≠ 0 ∧ (M / 100000) % 10 ≠ 0 ∧ (M / 1000000) % 10 ≠ 0) ∧
  (∃ y : ℕ, y * y = M) ∧
  (∃ f g : ℕ, f * f = a ∧ 2 * f * g = b ∧ g * g = c) ∧
  (10 ≤ a ∧ a ≤ 99) ∧
  (100 ≤ b ∧ b ≤ 999) ∧
  (10 ≤ c ∧ c ≤ 99)

theorem no_super_squarish_numbers : ∀ M : ℕ, is_super_squarish M → false :=
sorry

end no_super_squarish_numbers_l454_45497


namespace scoops_for_mom_l454_45474

/-- 
  Each scoop of ice cream costs $2.
  Pierre gets 3 scoops.
  The total bill is $14.
  Prove that Pierre's mom gets 4 scoops.
-/
theorem scoops_for_mom
  (scoop_cost : ℕ)
  (pierre_scoops : ℕ)
  (total_bill : ℕ) :
  scoop_cost = 2 → pierre_scoops = 3 → total_bill = 14 → 
  (total_bill - pierre_scoops * scoop_cost) / scoop_cost = 4 := 
by
  intros h1 h2 h3
  sorry

end scoops_for_mom_l454_45474


namespace total_shoes_l454_45409

variable (a b c d : Nat)

theorem total_shoes (h1 : a = 7) (h2 : b = a + 2) (h3 : c = 0) (h4 : d = 2 * (a + b + c)) :
  a + b + c + d = 48 :=
sorry

end total_shoes_l454_45409


namespace no_real_a_b_l454_45451

noncomputable def SetA (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ n : ℤ, p.1 = n ∧ p.2 = n * a + b}

noncomputable def SetB : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ m : ℤ, p.1 = m ∧ p.2 = 3 * m^2 + 15}

noncomputable def SetC : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 ≤ 144}

theorem no_real_a_b :
  ¬ ∃ (a b : ℝ), (∃ p ∈ SetA a b, p ∈ SetB) ∧ (a, b) ∈ SetC :=
by
    sorry

end no_real_a_b_l454_45451


namespace sufficient_not_necessary_condition_l454_45429

noncomputable def sequence_increasing_condition (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, n > 0 → a (n + 1) > |a n|

noncomputable def is_increasing_sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, a n < a (n + 1)

theorem sufficient_not_necessary_condition (a : ℕ → ℝ) :
  sequence_increasing_condition a → is_increasing_sequence a ∧ ¬(∀ b : ℕ → ℝ, is_increasing_sequence b → sequence_increasing_condition b) :=
sorry

end sufficient_not_necessary_condition_l454_45429


namespace max_x_minus_y_l454_45405

theorem max_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) : (x - y) ≤ 1 + 3 * (Real.sqrt 2) :=
sorry

end max_x_minus_y_l454_45405


namespace number_of_diagonals_octagon_heptagon_diff_l454_45465

def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem number_of_diagonals_octagon_heptagon_diff :
  let A := number_of_diagonals 8
  let B := number_of_diagonals 7
  A - B = 6 :=
by
  sorry

end number_of_diagonals_octagon_heptagon_diff_l454_45465


namespace milk_production_l454_45413

variables (a b c d e : ℕ) (h1 : a > 0) (h2 : c > 0)

def summer_rate := b / (a * c) -- Rate in summer per cow per day
def winter_rate := 2 * summer_rate -- Rate in winter per cow per day

noncomputable def total_milk_produced := (d * summer_rate * e) + (d * winter_rate * e)

theorem milk_production (h : d > 0) : total_milk_produced a b c d e = 3 * b * d * e / (a * c) :=
by sorry

end milk_production_l454_45413


namespace stocking_stuffers_total_l454_45499

theorem stocking_stuffers_total 
  (candy_canes_per_child beanie_babies_per_child books_per_child : ℕ)
  (num_children : ℕ)
  (h1 : candy_canes_per_child = 4)
  (h2 : beanie_babies_per_child = 2)
  (h3 : books_per_child = 1)
  (h4 : num_children = 3) :
  candy_canes_per_child + beanie_babies_per_child + books_per_child * num_children = 21 :=
by
  sorry

end stocking_stuffers_total_l454_45499


namespace hadley_total_walking_distance_l454_45434

-- Definitions of the distances walked to each location
def distance_grocery_store : ℕ := 2
def distance_pet_store : ℕ := distance_grocery_store - 1
def distance_home : ℕ := 4 - 1

-- Total distance walked by Hadley
def total_distance : ℕ := distance_grocery_store + distance_pet_store + distance_home

-- Statement to be proved
theorem hadley_total_walking_distance : total_distance = 6 := by
  sorry

end hadley_total_walking_distance_l454_45434


namespace values_of_z_l454_45467

theorem values_of_z (z : ℤ) (hz : 0 < z) :
  (z^2 - 50 * z + 550 ≤ 10) ↔ (20 ≤ z ∧ z ≤ 30) := sorry

end values_of_z_l454_45467


namespace sum_of_coefficients_l454_45480

theorem sum_of_coefficients (x y : ℝ) : 
  (2 * x - 3 * y) ^ 9 = -1 :=
by
  sorry

end sum_of_coefficients_l454_45480


namespace three_sport_players_l454_45450

def total_members := 50
def B := 22
def T := 28
def Ba := 18
def BT := 10
def BBa := 8
def TBa := 12
def N := 4
def All := 8

theorem three_sport_players : B + T + Ba - (BT + BBa + TBa) + All = total_members - N :=
by
suffices h : 22 + 28 + 18 - (10 + 8 + 12) + 8 = 50 - 4
exact h
-- The detailed proof is left as an exercise
sorry

end three_sport_players_l454_45450


namespace sequence_formula_l454_45471

noncomputable def a (n : ℕ) : ℕ := n

theorem sequence_formula (n : ℕ) (h : 0 < n) (S_n : ℕ → ℕ) 
  (hSn : ∀ m : ℕ, S_n m = (1 / 2 : ℚ) * (a m)^2 + (1 / 2 : ℚ) * m) : a n = n :=
by
  sorry

end sequence_formula_l454_45471


namespace Mikail_money_left_after_purchase_l454_45454

def Mikail_age_tomorrow : ℕ := 9  -- Defining Mikail's age tomorrow as 9.

def gift_per_year : ℕ := 5  -- Defining the gift amount per year of age as $5.

def video_game_cost : ℕ := 80  -- Defining the cost of the video game as $80.

def calculate_gift (age : ℕ) : ℕ := age * gift_per_year  -- Function to calculate the gift money he receives based on his age.

-- The statement we need to prove:
theorem Mikail_money_left_after_purchase : 
    calculate_gift Mikail_age_tomorrow < video_game_cost → calculate_gift Mikail_age_tomorrow - video_game_cost = 0 :=
by
  sorry

end Mikail_money_left_after_purchase_l454_45454


namespace line_equation_parallel_l454_45458

theorem line_equation_parallel (x₁ y₁ m : ℝ) (h₁ : (x₁, y₁) = (1, -2)) (h₂ : m = 2) :
  ∃ a b c : ℝ, a * x₁ + b * y₁ + c = 0 ∧ a * 2 + b * 1 + c = 4 := by
sorry

end line_equation_parallel_l454_45458


namespace father_l454_45447

variable (R F M : ℕ)
variable (h1 : F = 4 * R)
variable (h2 : 4 * R + 8 = M * (R + 8))
variable (h3 : 4 * R + 16 = 2 * (R + 16))

theorem father's_age_ratio (hR : R = 8) : (F + 8) / (R + 8) = 5 / 2 := by
  sorry

end father_l454_45447


namespace polynomial_expansion_l454_45421

theorem polynomial_expansion :
  (7 * X^2 + 5 * X - 3) * (3 * X^3 + 2 * X^2 + 1) = 
  21 * X^5 + 29 * X^4 + X^3 + X^2 + 5 * X - 3 :=
sorry

end polynomial_expansion_l454_45421


namespace cumulative_profit_exceeds_technical_renovation_expressions_for_A_n_B_n_l454_45476

noncomputable def A_n (n : ℕ) : ℝ :=
  490 * n - 10 * n^2

noncomputable def B_n (n : ℕ) : ℝ :=
  500 * n + 400 - 500 / 2^(n-1)

theorem cumulative_profit_exceeds_technical_renovation :
  ∀ n : ℕ, n ≥ 4 → B_n n > A_n n :=
by
  sorry  -- Proof goes here

theorem expressions_for_A_n_B_n (n : ℕ) :
  A_n n = 490 * n - 10 * n^2 ∧
  B_n n = 500 * n + 400 - 500 / 2^(n-1) :=
by
  sorry  -- Proof goes here

end cumulative_profit_exceeds_technical_renovation_expressions_for_A_n_B_n_l454_45476


namespace determine_triangle_area_l454_45461

noncomputable def triangle_area_proof : Prop :=
  let height : ℝ := 2
  let angle_ratio : ℝ := 2 / 1
  let smaller_base_part : ℝ := 1
  let larger_base_part : ℝ := 7 / 3
  let base := smaller_base_part + larger_base_part
  let area := (1 / 2) * base * height
  area = 11 / 3

theorem determine_triangle_area : triangle_area_proof :=
by
  sorry

end determine_triangle_area_l454_45461
