import Mathlib

namespace NUMINAMATH_GPT_cassie_water_bottle_ounces_l1109_110998

-- Define the given quantities
def cups_per_day : ℕ := 12
def ounces_per_cup : ℕ := 8
def refills_per_day : ℕ := 6

-- Define the total ounces of water Cassie drinks per day
def total_ounces_per_day := cups_per_day * ounces_per_cup

-- Define the ounces her water bottle holds
def ounces_per_bottle := total_ounces_per_day / refills_per_day

-- Prove the statement
theorem cassie_water_bottle_ounces : 
  ounces_per_bottle = 16 := by 
  sorry

end NUMINAMATH_GPT_cassie_water_bottle_ounces_l1109_110998


namespace NUMINAMATH_GPT_find_number_l1109_110980

theorem find_number (N : ℝ) (h : 0.4 * (3 / 5) * N = 36) : N = 150 := 
sorry

end NUMINAMATH_GPT_find_number_l1109_110980


namespace NUMINAMATH_GPT_two_digit_square_difference_l1109_110985

-- Define the problem in Lean
theorem two_digit_square_difference :
  ∃ (X Y : ℕ), (10 ≤ X ∧ X ≤ 99) ∧ (10 ≤ Y ∧ Y ≤ 99) ∧ (X > Y) ∧
  (∃ (t : ℕ), (1 ≤ t ∧ t ≤ 9) ∧ (X^2 - Y^2 = 100 * t)) :=
sorry

end NUMINAMATH_GPT_two_digit_square_difference_l1109_110985


namespace NUMINAMATH_GPT_total_simple_interest_is_correct_l1109_110967

noncomputable def principal : ℝ := 15041.875
noncomputable def rate : ℝ := 8
noncomputable def time : ℝ := 5
noncomputable def simple_interest (P R T : ℝ) : ℝ := P * R * T / 100

theorem total_simple_interest_is_correct :
  simple_interest principal rate time = 6016.75 := 
sorry

end NUMINAMATH_GPT_total_simple_interest_is_correct_l1109_110967


namespace NUMINAMATH_GPT_pets_beds_calculation_l1109_110946

theorem pets_beds_calculation
  (initial_beds : ℕ)
  (additional_beds : ℕ)
  (total_pets : ℕ)
  (H1 : initial_beds = 12)
  (H2 : additional_beds = 8)
  (H3 : total_pets = 10) :
  (initial_beds + additional_beds) / total_pets = 2 := 
by 
  sorry

end NUMINAMATH_GPT_pets_beds_calculation_l1109_110946


namespace NUMINAMATH_GPT_teresa_marks_ratio_l1109_110950

theorem teresa_marks_ratio (science music social_studies total_marks physics_ratio : ℝ) 
  (h_science : science = 70)
  (h_music : music = 80)
  (h_social_studies : social_studies = 85)
  (h_total_marks : total_marks = 275)
  (h_physics : science + music + social_studies + physics_ratio * music = total_marks) :
  physics_ratio = 1 / 2 :=
by
  subst h_science
  subst h_music
  subst h_social_studies
  subst h_total_marks
  have : 70 + 80 + 85 + physics_ratio * 80 = 275 := h_physics
  linarith

end NUMINAMATH_GPT_teresa_marks_ratio_l1109_110950


namespace NUMINAMATH_GPT_slope_of_l_l1109_110963

noncomputable def C (θ : ℝ) : ℝ × ℝ := (2 * Real.cos θ, 4 * Real.sin θ)
noncomputable def l (t α : ℝ) : ℝ × ℝ := (1 + t * Real.cos α, 2 + t * Real.sin α)

theorem slope_of_l
  (α θ₁ θ₂ t₁ t₂ : ℝ)
  (h_midpoint : (C θ₁).fst + (C θ₂).fst = 1 + (t₁ + t₂) * Real.cos α ∧ 
                (C θ₁).snd + (C θ₂).snd = 2 + (t₁ + t₂) * Real.sin α) :
  Real.tan α = -2 :=
by
  sorry

end NUMINAMATH_GPT_slope_of_l_l1109_110963


namespace NUMINAMATH_GPT_largest_base4_is_largest_l1109_110956

theorem largest_base4_is_largest 
  (n1 : ℕ) (n2 : ℕ) (n3 : ℕ) (n4 : ℕ)
  (h1 : n1 = 31) (h2 : n2 = 52) (h3 : n3 = 54) (h4 : n4 = 46) :
  n3 = Nat.max (Nat.max n1 n2) (Nat.max n3 n4) :=
by
  sorry

end NUMINAMATH_GPT_largest_base4_is_largest_l1109_110956


namespace NUMINAMATH_GPT_number_of_black_balls_l1109_110971

variable (T : ℝ)
variable (red_balls : ℝ := 21)
variable (prop_red : ℝ := 0.42)
variable (prop_white : ℝ := 0.28)
variable (white_balls : ℝ := 0.28 * T)

noncomputable def total_balls : ℝ := red_balls / prop_red

theorem number_of_black_balls :
  T = total_balls → 
  ∃ black_balls : ℝ, black_balls = total_balls - red_balls - white_balls ∧ black_balls = 15 := 
by
  intro hT
  let black_balls := total_balls - red_balls - white_balls
  use black_balls
  simp [total_balls]
  sorry

end NUMINAMATH_GPT_number_of_black_balls_l1109_110971


namespace NUMINAMATH_GPT_quadratic_roots_l1109_110972

theorem quadratic_roots (m : ℝ) (h1 : m > 4) :
  (∃ x y : ℝ, x ≠ y ∧ (m-5) * x^2 - 2 * (m + 2) * x + m = 0 ∧ (m-5) * y^2 - 2 * (m + 2) * y + m = 0)
  ∨ (m = 5 ∧ ∃ x : ℝ, (m-5) * x^2 - 2 * (m + 2) * x + m = 0)
  ∨ (¬((∃ x y : ℝ, x ≠ y ∧ (m-5) * x^2 - 2 * (m + 2) * x + m = 0) ∨ (m = 5 ∧ ∃ x : ℝ, (m-5) * x^2 - 2 * (m + 2) * x + m = 0))) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_roots_l1109_110972


namespace NUMINAMATH_GPT_find_sum_of_digits_l1109_110979

theorem find_sum_of_digits (a b c d : ℕ) 
  (h1 : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h2 : a = 1)
  (h3 : 1000 * a + 100 * b + 10 * c + d - (100 * b + 10 * c + d) < 100)
  : a + b + c + d = 2 := 
sorry

end NUMINAMATH_GPT_find_sum_of_digits_l1109_110979


namespace NUMINAMATH_GPT_A_salary_is_3000_l1109_110937

theorem A_salary_is_3000 
    (x y : ℝ) 
    (h1 : x + y = 4000)
    (h2 : 0.05 * x = 0.15 * y) 
    : x = 3000 := by
  sorry

end NUMINAMATH_GPT_A_salary_is_3000_l1109_110937


namespace NUMINAMATH_GPT_unit_digit_power3_58_l1109_110999

theorem unit_digit_power3_58 : (3 ^ 58) % 10 = 9 := by
  -- proof steps will be provided here
  sorry

end NUMINAMATH_GPT_unit_digit_power3_58_l1109_110999


namespace NUMINAMATH_GPT_b_power_a_equals_nine_l1109_110959

theorem b_power_a_equals_nine (a b : ℝ) (h : |a - 2| + (b + 3)^2 = 0) : b^a = 9 := by
  sorry

end NUMINAMATH_GPT_b_power_a_equals_nine_l1109_110959


namespace NUMINAMATH_GPT_remainder_expression_l1109_110907

theorem remainder_expression (x y u v : ℕ) (h1 : x = u * y + v) (h2 : 0 ≤ v) (h3 : v < y) : 
  (x + 3 * u * y) % y = v := 
by
  sorry

end NUMINAMATH_GPT_remainder_expression_l1109_110907


namespace NUMINAMATH_GPT_sales_professionals_count_l1109_110992

theorem sales_professionals_count :
  (∀ (C : ℕ) (MC : ℕ) (M : ℕ), C = 500 → MC = 10 → M = 5 → C / M / MC = 10) :=
by
  intros C MC M hC hMC hM
  sorry

end NUMINAMATH_GPT_sales_professionals_count_l1109_110992


namespace NUMINAMATH_GPT_tiles_needed_l1109_110910

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

end NUMINAMATH_GPT_tiles_needed_l1109_110910


namespace NUMINAMATH_GPT_range_of_a_l1109_110911

open Set

def p (a : ℝ) := ∀ x : ℝ, x^2 + 2 * a * x + 4 > 0
def q (a : ℝ) := ∀ x : ℝ, x ∈ (Icc 1 2) → x^2 ≥ a

theorem range_of_a (a : ℝ) : 
  (p a ∨ q a) ∧ ¬(p a ∧ q a) ↔ a ∈ (Ioo 1 2 ∪ Iic (-2)) :=
by sorry

end NUMINAMATH_GPT_range_of_a_l1109_110911


namespace NUMINAMATH_GPT_percent_increase_sales_l1109_110984

-- Define constants for sales
def sales_last_year : ℕ := 320
def sales_this_year : ℕ := 480

-- Define the percent increase formula
def percent_increase (old_value new_value : ℕ) : ℚ :=
  ((new_value - old_value) / old_value) * 100

-- Prove the percent increase from last year to this year is 50%
theorem percent_increase_sales : percent_increase sales_last_year sales_this_year = 50 := by
  sorry

end NUMINAMATH_GPT_percent_increase_sales_l1109_110984


namespace NUMINAMATH_GPT_inequality_a5_b5_c5_l1109_110948

theorem inequality_a5_b5_c5 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a^5 + b^5 + c^5 ≥ a^3 * b * c + a * b^3 * c + a * b * c^3 :=
by
  sorry

end NUMINAMATH_GPT_inequality_a5_b5_c5_l1109_110948


namespace NUMINAMATH_GPT_contradiction_proof_l1109_110978

theorem contradiction_proof (x y : ℝ) (h1 : x + y < 2) (h2 : 1 < x) (h3 : 1 < y) : false := 
by 
  sorry

end NUMINAMATH_GPT_contradiction_proof_l1109_110978


namespace NUMINAMATH_GPT_foci_distance_of_hyperbola_l1109_110968

theorem foci_distance_of_hyperbola :
  let a_sq := 25
  let b_sq := 9
  let c := Real.sqrt (a_sq + b_sq)
  2 * c = 2 * Real.sqrt 34 :=
by
  let a_sq := 25
  let b_sq := 9
  let c := Real.sqrt (a_sq + b_sq)
  sorry

end NUMINAMATH_GPT_foci_distance_of_hyperbola_l1109_110968


namespace NUMINAMATH_GPT_correct_option_l1109_110943

theorem correct_option :
  (2 * Real.sqrt 5) + (3 * Real.sqrt 5) = 5 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_GPT_correct_option_l1109_110943


namespace NUMINAMATH_GPT_set_proof_l1109_110995

open Set

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {5, 6, 7}

theorem set_proof :
  (U \ A) ∩ (U \ B) = {4, 8} := by
  sorry

end NUMINAMATH_GPT_set_proof_l1109_110995


namespace NUMINAMATH_GPT_find_prime_solution_l1109_110913

theorem find_prime_solution :
  ∀ p x y : ℕ, Prime p → x > 0 → y > 0 →
    (p ^ x = y ^ 3 + 1) ↔ 
    ((p = 2 ∧ x = 1 ∧ y = 1) ∨ (p = 3 ∧ x = 2 ∧ y = 2)) := 
by
  sorry

end NUMINAMATH_GPT_find_prime_solution_l1109_110913


namespace NUMINAMATH_GPT_graph_of_x2_minus_y2_eq_0_is_two_intersecting_lines_l1109_110969

theorem graph_of_x2_minus_y2_eq_0_is_two_intersecting_lines :
  ∀ x y : ℝ, (x^2 - y^2 = 0) ↔ (y = x ∨ y = -x) := 
by
  sorry

end NUMINAMATH_GPT_graph_of_x2_minus_y2_eq_0_is_two_intersecting_lines_l1109_110969


namespace NUMINAMATH_GPT_large_planter_holds_seeds_l1109_110996

theorem large_planter_holds_seeds (total_seeds : ℕ) (small_planter_capacity : ℕ) (num_small_planters : ℕ) (num_large_planters : ℕ) 
  (h1 : total_seeds = 200)
  (h2 : small_planter_capacity = 4)
  (h3 : num_small_planters = 30)
  (h4 : num_large_planters = 4) : 
  (total_seeds - num_small_planters * small_planter_capacity) / num_large_planters = 20 := by
  sorry

end NUMINAMATH_GPT_large_planter_holds_seeds_l1109_110996


namespace NUMINAMATH_GPT_range_of_a1_l1109_110947

theorem range_of_a1 (a : ℕ → ℝ) (h : ∀ n, a (n + 1) = 1 / (2 - a n)) 
  (h_pos : ∀ n, a (n + 1) > a n) : a 1 < 1 := 
sorry

end NUMINAMATH_GPT_range_of_a1_l1109_110947


namespace NUMINAMATH_GPT_jane_emily_total_accessories_l1109_110922

def total_accessories : ℕ :=
  let jane_dresses := 4 * 10
  let emily_dresses := 3 * 8
  let jane_ribbons := 3 * jane_dresses
  let jane_buttons := 2 * jane_dresses
  let jane_lace_trims := 1 * jane_dresses
  let jane_beads := 4 * jane_dresses
  let emily_ribbons := 2 * emily_dresses
  let emily_buttons := 3 * emily_dresses
  let emily_lace_trims := 2 * emily_dresses
  let emily_beads := 5 * emily_dresses
  let emily_bows := 1 * emily_dresses
  jane_ribbons + jane_buttons + jane_lace_trims + jane_beads +
  emily_ribbons + emily_buttons + emily_lace_trims + emily_beads + emily_bows 

theorem jane_emily_total_accessories : total_accessories = 712 := 
by
  sorry

end NUMINAMATH_GPT_jane_emily_total_accessories_l1109_110922


namespace NUMINAMATH_GPT_find_x_l1109_110955

theorem find_x (a b x : ℕ) (h1 : a = 105) (h2 : b = 147) (h3 : a^3 = 21 * x * 15 * b) : x = 25 :=
by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_find_x_l1109_110955


namespace NUMINAMATH_GPT_hours_per_day_in_deliberation_l1109_110903

noncomputable def jury_selection_days : ℕ := 2
noncomputable def trial_days : ℕ := 4 * jury_selection_days
noncomputable def total_deliberation_hours : ℕ := 6 * 24
noncomputable def total_days_on_jury_duty : ℕ := 19

theorem hours_per_day_in_deliberation :
  (total_deliberation_hours / (total_days_on_jury_duty - (jury_selection_days + trial_days))) = 16 :=
by
  sorry

end NUMINAMATH_GPT_hours_per_day_in_deliberation_l1109_110903


namespace NUMINAMATH_GPT_sum_2001_and_1015_l1109_110932

theorem sum_2001_and_1015 :
  2001 + 1015 = 3016 :=
sorry

end NUMINAMATH_GPT_sum_2001_and_1015_l1109_110932


namespace NUMINAMATH_GPT_find_missing_edge_l1109_110970

-- Define the known parameters
def volume : ℕ := 80
def edge1 : ℕ := 2
def edge3 : ℕ := 8

-- Define the missing edge
def missing_edge : ℕ := 5

-- State the problem
theorem find_missing_edge (volume : ℕ) (edge1 : ℕ) (edge3 : ℕ) (missing_edge : ℕ) :
  volume = edge1 * missing_edge * edge3 →
  missing_edge = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_missing_edge_l1109_110970


namespace NUMINAMATH_GPT_average_monthly_bill_l1109_110966

-- Definitions based on conditions
def first_4_months_average := 30
def last_2_months_average := 24
def first_4_months_total := 4 * first_4_months_average
def last_2_months_total := 2 * last_2_months_average
def total_spent := first_4_months_total + last_2_months_total
def total_months := 6

-- The theorem statement
theorem average_monthly_bill : total_spent / total_months = 28 := by
  sorry

end NUMINAMATH_GPT_average_monthly_bill_l1109_110966


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l1109_110986

theorem quadratic_inequality_solution :
  {x : ℝ | x^2 - 3 * x - 18 < 0} = {x : ℝ | -3 < x ∧ x < 6} :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l1109_110986


namespace NUMINAMATH_GPT_total_books_count_l1109_110962

theorem total_books_count (total_cost : ℕ) (math_book_cost : ℕ) (history_book_cost : ℕ) 
    (math_books_count : ℕ) (history_books_count : ℕ) (total_books : ℕ) :
    total_cost = 390 ∧ math_book_cost = 4 ∧ history_book_cost = 5 ∧ 
    math_books_count = 10 ∧ total_books = math_books_count + history_books_count ∧ 
    total_cost = (math_book_cost * math_books_count) + (history_book_cost * history_books_count) →
    total_books = 80 := by
  sorry

end NUMINAMATH_GPT_total_books_count_l1109_110962


namespace NUMINAMATH_GPT_prob_diff_colors_correct_l1109_110915

def total_chips := 6 + 5 + 4 + 3

def prob_diff_colors : ℚ :=
  (6 / total_chips * (12 / total_chips) +
  5 / total_chips * (13 / total_chips) +
  4 / total_chips * (14 / total_chips) +
  3 / total_chips * (15 / total_chips))

theorem prob_diff_colors_correct :
  prob_diff_colors = 119 / 162 := by
  sorry

end NUMINAMATH_GPT_prob_diff_colors_correct_l1109_110915


namespace NUMINAMATH_GPT_electricity_bill_written_as_decimal_l1109_110983

-- Definitions as conditions
def number : ℝ := 71.08

-- Proof statement
theorem electricity_bill_written_as_decimal : number = 71.08 :=
by sorry

end NUMINAMATH_GPT_electricity_bill_written_as_decimal_l1109_110983


namespace NUMINAMATH_GPT_train_speed_l1109_110991

theorem train_speed :
  let train_length := 200 -- in meters
  let platform_length := 175.03 -- in meters
  let time_taken := 25 -- in seconds
  let total_distance := train_length + platform_length -- total distance in meters
  let speed_mps := total_distance / time_taken -- speed in meters per second
  let speed_kmph := speed_mps * 3.6 -- converting speed to kilometers per hour
  speed_kmph = 54.00432 := sorry

end NUMINAMATH_GPT_train_speed_l1109_110991


namespace NUMINAMATH_GPT_card_average_value_l1109_110919

theorem card_average_value (n : ℕ) (h : (2 * n + 1) / 3 = 2023) : n = 3034 :=
sorry

end NUMINAMATH_GPT_card_average_value_l1109_110919


namespace NUMINAMATH_GPT_countDivisorsOf72Pow8_l1109_110958

-- Definitions of conditions in Lean 4
def isPerfectSquare (a b : ℕ) : Prop := a % 2 = 0 ∧ b % 2 = 0
def isPerfectCube (a b : ℕ) : Prop := a % 3 = 0 ∧ b % 3 = 0
def isPerfectSixthPower (a b : ℕ) : Prop := a % 6 = 0 ∧ b % 6 = 0

def countPerfectSquares : ℕ := 13 * 9
def countPerfectCubes : ℕ := 9 * 6
def countPerfectSixthPowers : ℕ := 5 * 3

-- The proof problem to prove the number of such divisors is 156
theorem countDivisorsOf72Pow8:
  (countPerfectSquares + countPerfectCubes - countPerfectSixthPowers) = 156 :=
by
  sorry

end NUMINAMATH_GPT_countDivisorsOf72Pow8_l1109_110958


namespace NUMINAMATH_GPT_second_player_wins_l1109_110924

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

end NUMINAMATH_GPT_second_player_wins_l1109_110924


namespace NUMINAMATH_GPT_circle_radius_5_l1109_110965

-- The circle equation given
def circle_eq (x y : ℝ) (c : ℝ) : Prop :=
  x^2 + 4 * x + y^2 + 8 * y + c = 0

-- The radius condition given
def radius_condition : Prop :=
  5 = (25 : ℝ).sqrt

-- The final proof statement
theorem circle_radius_5 (c : ℝ) : 
  (∀ x y : ℝ, circle_eq x y c) → radius_condition → c = -5 := 
by
  sorry

end NUMINAMATH_GPT_circle_radius_5_l1109_110965


namespace NUMINAMATH_GPT_ben_min_sales_l1109_110926

theorem ben_min_sales 
    (old_salary : ℕ := 75000) 
    (new_base_salary : ℕ := 45000) 
    (commission_rate : ℚ := 0.15) 
    (sale_amount : ℕ := 750) : 
    ∃ (n : ℕ), n ≥ 267 ∧ (old_salary ≤ new_base_salary + n * ⌊commission_rate * sale_amount⌋) :=
by 
  sorry

end NUMINAMATH_GPT_ben_min_sales_l1109_110926


namespace NUMINAMATH_GPT_unfolded_side_view_of_cone_is_sector_l1109_110933

theorem unfolded_side_view_of_cone_is_sector 
  (shape : Type)
  (curved_side : shape)
  (straight_side1 : shape)
  (straight_side2 : shape) 
  (condition1 : ∃ (s : shape), s = curved_side) 
  (condition2 : ∃ (s1 s2 : shape), s1 = straight_side1 ∧ s2 = straight_side2)
  : shape = sector :=
sorry

end NUMINAMATH_GPT_unfolded_side_view_of_cone_is_sector_l1109_110933


namespace NUMINAMATH_GPT_factorize_expression_l1109_110939

theorem factorize_expression (a b : ℝ) : 2 * a ^ 2 - 8 * b ^ 2 = 2 * (a + 2 * b) * (a - 2 * b) :=
by
  sorry

end NUMINAMATH_GPT_factorize_expression_l1109_110939


namespace NUMINAMATH_GPT_remainder_when_divided_by_100_l1109_110973

-- Define the given m
def m : ℕ := 76^2006 - 76

-- State the theorem
theorem remainder_when_divided_by_100 : m % 100 = 0 :=
by
  sorry

end NUMINAMATH_GPT_remainder_when_divided_by_100_l1109_110973


namespace NUMINAMATH_GPT_larger_solution_quadratic_l1109_110917

theorem larger_solution_quadratic (x : ℝ) : x^2 - 13 * x + 42 = 0 → x = 7 ∨ x = 6 ∧ x > 6 :=
by
  sorry

end NUMINAMATH_GPT_larger_solution_quadratic_l1109_110917


namespace NUMINAMATH_GPT_average_speed_correct_l1109_110990

noncomputable def average_speed (d v_up v_down : ℝ) : ℝ :=
  let t_up := d / v_up
  let t_down := d / v_down
  let total_distance := 2 * d
  let total_time := t_up + t_down
  total_distance / total_time

theorem average_speed_correct :
  average_speed 0.2 24 36 = 28.8 := by {
  sorry
}

end NUMINAMATH_GPT_average_speed_correct_l1109_110990


namespace NUMINAMATH_GPT_paper_fold_length_l1109_110918

theorem paper_fold_length (length_orig : ℝ) (h : length_orig = 12) : length_orig / 2 = 6 :=
by
  rw [h]
  norm_num

end NUMINAMATH_GPT_paper_fold_length_l1109_110918


namespace NUMINAMATH_GPT_partitions_equiv_l1109_110925

-- Definition of partitions into distinct integers
def a (n : ℕ) : ℕ := sorry  -- Placeholder for the actual definition or count function

-- Definition of partitions into odd integers
def b (n : ℕ) : ℕ := sorry  -- Placeholder for the actual definition or count function

-- Theorem stating that the number of partitions into distinct integers equals the number of partitions into odd integers
theorem partitions_equiv (n : ℕ) : a n = b n :=
sorry

end NUMINAMATH_GPT_partitions_equiv_l1109_110925


namespace NUMINAMATH_GPT_gain_percent_is_approx_30_11_l1109_110949

-- Definitions for cost price (CP) and selling price (SP)
def CP : ℕ := 930
def SP : ℕ := 1210

-- Definition for gain percent
noncomputable def gain_percent : ℚ :=
  ((SP - CP : ℚ) / CP) * 100

-- Statement to prove the gain percent is approximately 30.11%
theorem gain_percent_is_approx_30_11 :
  abs (gain_percent - 30.11) < 0.01 := by
  sorry

end NUMINAMATH_GPT_gain_percent_is_approx_30_11_l1109_110949


namespace NUMINAMATH_GPT_percentage_problem_l1109_110940

variable (y x z : ℝ)

def A := y * x^2 + 3 * z - 6

theorem percentage_problem (h : A y x z > 0) :
  (2 * A y x z / 5) + (3 * A y x z / 10) = (70 / 100) * A y x z :=
by
  sorry

end NUMINAMATH_GPT_percentage_problem_l1109_110940


namespace NUMINAMATH_GPT_parabola_perpendicular_bisector_intersects_x_axis_l1109_110938

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

end NUMINAMATH_GPT_parabola_perpendicular_bisector_intersects_x_axis_l1109_110938


namespace NUMINAMATH_GPT_probability_at_least_one_hit_l1109_110941

-- Define probabilities of each shooter hitting the target
def P_A : ℚ := 1 / 2
def P_B : ℚ := 1 / 3
def P_C : ℚ := 1 / 4

-- Define the complementary probabilities (each shooter misses the target)
def P_A_miss : ℚ := 1 - P_A
def P_B_miss : ℚ := 1 - P_B
def P_C_miss : ℚ := 1 - P_C

-- Calculate the probability of all shooters missing the target
def P_all_miss : ℚ := P_A_miss * P_B_miss * P_C_miss

-- Calculate the probability of at least one shooter hitting the target
def P_at_least_one_hit : ℚ := 1 - P_all_miss

-- The theorem to be proved
theorem probability_at_least_one_hit : 
  P_at_least_one_hit = 3 / 4 := 
by sorry

end NUMINAMATH_GPT_probability_at_least_one_hit_l1109_110941


namespace NUMINAMATH_GPT_smallest_b_value_l1109_110957

noncomputable def smallest_b (a b : ℝ) : ℝ :=
if a > 2 ∧ 2 < a ∧ a < b 
   ∧ (2 + a ≤ b) 
   ∧ ((1 / a) + (1 / b) ≤ 1 / 2) 
then b else 0

theorem smallest_b_value : ∀ (a b : ℝ), 
  (2 < a) → (a < b) → (2 + a ≤ b) → 
  ((1 / a) + (1 / b) ≤ 1 / 2) → 
  b = 3 + Real.sqrt 5 := sorry

end NUMINAMATH_GPT_smallest_b_value_l1109_110957


namespace NUMINAMATH_GPT_margo_donation_l1109_110975

variable (M J : ℤ)

theorem margo_donation (h1: J = 4700) (h2: (|J - M| / 2) = 200) : M = 4300 :=
sorry

end NUMINAMATH_GPT_margo_donation_l1109_110975


namespace NUMINAMATH_GPT_fixed_point_at_5_75_l1109_110935

-- Defining the function
def quadratic_function (k : ℝ) (x : ℝ) : ℝ := 3 * x^2 + k * x - 5 * k

-- Stating the theorem that the graph passes through the fixed point (5, 75)
theorem fixed_point_at_5_75 (k : ℝ) : quadratic_function k 5 = 75 := by
  sorry

end NUMINAMATH_GPT_fixed_point_at_5_75_l1109_110935


namespace NUMINAMATH_GPT_minimize_feed_costs_l1109_110993

theorem minimize_feed_costs 
  (x y : ℝ)
  (h1: 5 * x + 3 * y ≥ 30)
  (h2: 2.5 * x + 3 * y ≥ 22.5)
  (h3: x ≥ 0)
  (h4: y ≥ 0)
  : (x = 3 ∧ y = 5) ∧ (x + y = 8) := 
sorry

end NUMINAMATH_GPT_minimize_feed_costs_l1109_110993


namespace NUMINAMATH_GPT_divisor_correct_l1109_110923

/--
Given that \(10^{23} - 7\) divided by \(d\) leaves a remainder 3, 
prove that \(d\) is equal to \(10^{23} - 10\).
-/
theorem divisor_correct :
  ∃ d : ℤ, (10^23 - 7) % d = 3 ∧ d = 10^23 - 10 :=
by
  sorry

end NUMINAMATH_GPT_divisor_correct_l1109_110923


namespace NUMINAMATH_GPT_minimize_cost_per_km_l1109_110920

section ship_cost_minimization

variables (u v k : ℝ) (fuel_cost other_cost total_cost_per_km: ℝ)

-- Condition 1: The fuel cost per unit time is directly proportional to the cube of its speed.
def fuel_cost_eq : Prop := u = k * v^3

-- Condition 2: When the speed of the ship is 10 km/h, the fuel cost is 35 yuan per hour.
def fuel_cost_at_10 : Prop := u = 35 ∧ v = 10

-- Condition 3: The other costs are 560 yuan per hour.
def other_cost_eq : Prop := other_cost = 560

-- Condition 4: The maximum speed of the ship is 25 km/h.
def max_speed : Prop := v ≤ 25

-- Prove that the speed of the ship that minimizes the cost per kilometer is 20 km/h.
theorem minimize_cost_per_km : 
  fuel_cost_eq u v k ∧ fuel_cost_at_10 u v ∧ other_cost_eq other_cost ∧ max_speed v → v = 20 :=
by
  sorry

end ship_cost_minimization

end NUMINAMATH_GPT_minimize_cost_per_km_l1109_110920


namespace NUMINAMATH_GPT_cicely_100th_birthday_l1109_110908

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

end NUMINAMATH_GPT_cicely_100th_birthday_l1109_110908


namespace NUMINAMATH_GPT_first_number_in_expression_l1109_110977

theorem first_number_in_expression (a b c d e : ℝ)
  (h_expr : (a * b * c) / d + e = 2229) :
  a = 26.3 :=
  sorry

end NUMINAMATH_GPT_first_number_in_expression_l1109_110977


namespace NUMINAMATH_GPT_quadratic_solution_l1109_110929

theorem quadratic_solution (x : ℝ) : (x^2 + 6 * x + 8 = -2 * (x + 4) * (x + 5)) ↔ (x = -8 ∨ x = -4) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_solution_l1109_110929


namespace NUMINAMATH_GPT_complex_ratio_proof_l1109_110960

noncomputable def complex_ratio (x y : ℂ) : ℂ :=
  ((x^6 + y^6) / (x^6 - y^6)) - ((x^6 - y^6) / (x^6 + y^6))

theorem complex_ratio_proof (x y : ℂ) (h : ((x - y) / (x + y)) - ((x + y) / (x - y)) = 2) :
  complex_ratio x y = L :=
  sorry

end NUMINAMATH_GPT_complex_ratio_proof_l1109_110960


namespace NUMINAMATH_GPT_remaining_gnomes_total_l1109_110936

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

end NUMINAMATH_GPT_remaining_gnomes_total_l1109_110936


namespace NUMINAMATH_GPT_firm_partners_initial_count_l1109_110912

theorem firm_partners_initial_count
  (x : ℕ)
  (h1 : 2*x/(63*x + 35) = 1/34)
  (h2 : 2*x/(20*x + 10) = 1/15) :
  2*x = 14 :=
by
  sorry

end NUMINAMATH_GPT_firm_partners_initial_count_l1109_110912


namespace NUMINAMATH_GPT_boat_sinking_weight_range_l1109_110900

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

end NUMINAMATH_GPT_boat_sinking_weight_range_l1109_110900


namespace NUMINAMATH_GPT_cost_of_new_game_l1109_110997

theorem cost_of_new_game (initial_money : ℕ) (money_left : ℕ) (toy_cost : ℕ) (toy_count : ℕ)
  (h_initial : initial_money = 68) (h_toy_cost : toy_cost = 7) (h_toy_count : toy_count = 3) 
  (h_money_left : money_left = toy_count * toy_cost) :
  initial_money - money_left = 47 :=
by {
  sorry
}

end NUMINAMATH_GPT_cost_of_new_game_l1109_110997


namespace NUMINAMATH_GPT_range_of_k_l1109_110942

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x ≤ 1 ∨ x ≥ 3}
def B (k : ℝ) : Set ℝ := {x | k < x ∧ x < 2 * k + 1}
def A_complement : Set ℝ := {x | 1 < x ∧ x < 3}

theorem range_of_k (k : ℝ) : ((A_complement ∩ (B k)) = ∅) ↔ (k ∈ Set.Iic 0 ∪ Set.Ici 3) := sorry

end NUMINAMATH_GPT_range_of_k_l1109_110942


namespace NUMINAMATH_GPT_dave_guitar_strings_l1109_110914

theorem dave_guitar_strings (strings_per_night : ℕ) (shows_per_week : ℕ) (weeks : ℕ)
  (h1 : strings_per_night = 4)
  (h2 : shows_per_week = 6)
  (h3 : weeks = 24) : 
  strings_per_night * shows_per_week * weeks = 576 :=
by
  sorry

end NUMINAMATH_GPT_dave_guitar_strings_l1109_110914


namespace NUMINAMATH_GPT_complex_number_solution_l1109_110904

theorem complex_number_solution (z i : ℂ) (h : z * (i - i^2) = 1 + i^3) (h1 : i^2 = -1) (h2 : i^3 = -i) (h3 : i^4 = 1) : 
  z = -i := 
by 
  sorry

end NUMINAMATH_GPT_complex_number_solution_l1109_110904


namespace NUMINAMATH_GPT_playground_area_22500_l1109_110905

noncomputable def rectangle_playground_area (w l : ℕ) : ℕ :=
  w * l

theorem playground_area_22500 (w l : ℕ) (h1 : l = 2 * w + 25) (h2 : 2 * l + 2 * w = 650) :
  rectangle_playground_area w l = 22500 := by
  sorry

end NUMINAMATH_GPT_playground_area_22500_l1109_110905


namespace NUMINAMATH_GPT_prime_implies_power_of_two_l1109_110906

-- Conditions:
def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ m : ℕ, m ∣ p → m = 1 ∨ m = p

-- Problem:
theorem prime_implies_power_of_two (n : ℕ) (h : is_prime (2^n + 1)) : ∃ k : ℕ, n = 2^k := sorry

end NUMINAMATH_GPT_prime_implies_power_of_two_l1109_110906


namespace NUMINAMATH_GPT_round_robin_games_l1109_110927

theorem round_robin_games (x : ℕ) (h : 45 = (1 / 2) * x * (x - 1)) : (1 / 2) * x * (x - 1) = 45 :=
sorry

end NUMINAMATH_GPT_round_robin_games_l1109_110927


namespace NUMINAMATH_GPT_modulus_complex_number_l1109_110989

theorem modulus_complex_number (i : ℂ) (h : i = Complex.I) : 
  Complex.abs (1 / (i - 1)) = Real.sqrt 2 / 2 :=
by
  sorry

end NUMINAMATH_GPT_modulus_complex_number_l1109_110989


namespace NUMINAMATH_GPT_inradius_of_triangle_l1109_110902

theorem inradius_of_triangle (A p r s : ℝ) (h1 : A = 3 * p) (h2 : A = r * s) (h3 : s = p / 2) :
  r = 6 :=
by
  sorry

end NUMINAMATH_GPT_inradius_of_triangle_l1109_110902


namespace NUMINAMATH_GPT_evaluate_expression_l1109_110945

theorem evaluate_expression :
  let a := 12
  let b := 14
  let c := 18
  (144 * ((1:ℝ)/b - (1:ℝ)/c) + 196 * ((1:ℝ)/c - (1:ℝ)/a) + 324 * ((1:ℝ)/a - (1:ℝ)/b)) /
  (a * ((1:ℝ)/b - (1:ℝ)/c) + b * ((1:ℝ)/c - (1:ℝ)/a) + c * ((1:ℝ)/a - (1:ℝ)/b)) = a + b + c := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1109_110945


namespace NUMINAMATH_GPT_solve_equation_l1109_110954

theorem solve_equation (x : ℝ) : 
  (9 - 3 * x) * (3 ^ x) - (x - 2) * (x ^ 2 - 5 * x + 6) = 0 ↔ x = 3 :=
by sorry

end NUMINAMATH_GPT_solve_equation_l1109_110954


namespace NUMINAMATH_GPT_symmetric_origin_coordinates_l1109_110930

def symmetric_coordinates (x y : ℚ) (x_line y_line : ℚ) : Prop :=
  x_line - 2 * y_line + 2 = 0 ∧ y_line = -2 * x_line ∧ x = -4/5 ∧ y = 8/5

theorem symmetric_origin_coordinates :
  ∃ (x_0 y_0 : ℚ), symmetric_coordinates x_0 y_0 (-4/5) (8/5) :=
by
  use -4/5, 8/5
  sorry

end NUMINAMATH_GPT_symmetric_origin_coordinates_l1109_110930


namespace NUMINAMATH_GPT_chef_leftover_potatoes_l1109_110916

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

end NUMINAMATH_GPT_chef_leftover_potatoes_l1109_110916


namespace NUMINAMATH_GPT_track_is_600_l1109_110944

noncomputable def track_length (x : ℝ) : Prop :=
  ∃ (s_b s_s : ℝ), 
      s_b > 0 ∧ s_s > 0 ∧
      (∀ t, t > 0 → ((s_b * t = 120 ∧ s_s * t = x / 2 - 120) ∨ 
                     (s_s * (t + 180 / s_s) - s_s * t = x / 2 + 60 
                      ∧ s_b * (t + 180 / s_s) - s_b * t = x / 2 - 60)))

theorem track_is_600 : track_length 600 :=
sorry

end NUMINAMATH_GPT_track_is_600_l1109_110944


namespace NUMINAMATH_GPT_total_pink_crayons_l1109_110981

-- Define the conditions
def Mara_crayons : ℕ := 40
def Mara_pink_percent : ℕ := 10
def Luna_crayons : ℕ := 50
def Luna_pink_percent : ℕ := 20

-- Define the proof problem statement
theorem total_pink_crayons : 
  (Mara_crayons * Mara_pink_percent / 100) + (Luna_crayons * Luna_pink_percent / 100) = 14 := 
by sorry

end NUMINAMATH_GPT_total_pink_crayons_l1109_110981


namespace NUMINAMATH_GPT_eqn_intersecting_straight_lines_l1109_110976

theorem eqn_intersecting_straight_lines (x y : ℝ) : 
  x^2 - y^2 = 0 → (y = x ∨ y = -x) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_eqn_intersecting_straight_lines_l1109_110976


namespace NUMINAMATH_GPT_tom_purchased_8_kg_of_apples_l1109_110951

noncomputable def number_of_apples_purchased (price_per_kg_apple : ℤ) (price_per_kg_mango : ℤ) (kg_mangoes : ℤ) (total_paid : ℤ) : ℤ :=
  let total_cost_mangoes := price_per_kg_mango * kg_mangoes
  total_paid - total_cost_mangoes / price_per_kg_apple

theorem tom_purchased_8_kg_of_apples : 
  number_of_apples_purchased 70 65 9 1145 = 8 := 
by {
  -- Expand the definitions and simplify
  sorry
}

end NUMINAMATH_GPT_tom_purchased_8_kg_of_apples_l1109_110951


namespace NUMINAMATH_GPT_school_minimum_payment_l1109_110953

noncomputable def individual_ticket_price : ℝ := 6
noncomputable def group_ticket_price : ℝ := 40
noncomputable def discount : ℝ := 0.9
noncomputable def students : ℕ := 1258

-- Define the minimum amount the school should pay
noncomputable def minimum_amount := 4536

theorem school_minimum_payment :
  (students / 10 : ℝ) * group_ticket_price * discount + 
  (students % 10) * individual_ticket_price * discount = minimum_amount := sorry

end NUMINAMATH_GPT_school_minimum_payment_l1109_110953


namespace NUMINAMATH_GPT_intersection_of_sets_l1109_110952

noncomputable def A : Set ℤ := {x | x^2 - 1 = 0}
def B : Set ℤ := {-1, 2, 5}

theorem intersection_of_sets : A ∩ B = {-1} :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_sets_l1109_110952


namespace NUMINAMATH_GPT_value_20_percent_greater_l1109_110928

theorem value_20_percent_greater (x : ℝ) : (x = 88 * 1.20) ↔ (x = 105.6) :=
by
  sorry

end NUMINAMATH_GPT_value_20_percent_greater_l1109_110928


namespace NUMINAMATH_GPT_fraction_stamp_collection_l1109_110988

theorem fraction_stamp_collection (sold_amount total_value : ℝ) (sold_for : sold_amount = 28) (total : total_value = 49) : sold_amount / total_value = 4 / 7 :=
by
  sorry

end NUMINAMATH_GPT_fraction_stamp_collection_l1109_110988


namespace NUMINAMATH_GPT_no_magpies_left_l1109_110987

theorem no_magpies_left (initial_magpies killed_magpies : ℕ) (fly_away : Prop):
  initial_magpies = 40 → killed_magpies = 6 → fly_away → ∀ M : ℕ, M = 0 :=
by
  intro h0 h1 h2
  sorry

end NUMINAMATH_GPT_no_magpies_left_l1109_110987


namespace NUMINAMATH_GPT_remainder_of_concatenated_number_l1109_110964

def concatenated_number : ℕ :=
  -- Definition of the concatenated number
  -- That is 123456789101112...4344
  -- For simplicity, we'll just assign it directly
  1234567891011121314151617181920212223242526272829303132333435363738394041424344

theorem remainder_of_concatenated_number :
  concatenated_number % 45 = 9 :=
sorry

end NUMINAMATH_GPT_remainder_of_concatenated_number_l1109_110964


namespace NUMINAMATH_GPT_reading_schedule_correct_l1109_110934

-- Defining the conditions
def total_words : ℕ := 34685
def words_day1 (x : ℕ) : ℕ := x
def words_day2 (x : ℕ) : ℕ := 2 * x
def words_day3 (x : ℕ) : ℕ := 4 * x

-- Defining the main statement of the problem
theorem reading_schedule_correct (x : ℕ) : 
  words_day1 x + words_day2 x + words_day3 x = total_words := 
sorry

end NUMINAMATH_GPT_reading_schedule_correct_l1109_110934


namespace NUMINAMATH_GPT_sum_of_reciprocals_of_numbers_l1109_110982

theorem sum_of_reciprocals_of_numbers (x y : ℕ) (h_sum : x + y = 45) (h_hcf : Nat.gcd x y = 3)
    (h_lcm : Nat.lcm x y = 100) : 1/x + 1/y = 3/20 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_reciprocals_of_numbers_l1109_110982


namespace NUMINAMATH_GPT_total_lifespan_l1109_110909

theorem total_lifespan (B H F : ℕ)
  (hB : B = 10)
  (hH : H = B - 6)
  (hF : F = 4 * H) :
  B + H + F = 30 := by
  sorry

end NUMINAMATH_GPT_total_lifespan_l1109_110909


namespace NUMINAMATH_GPT_equipment_total_cost_l1109_110961

def cost_jersey : ℝ := 25
def cost_shorts : ℝ := 15.20
def cost_socks : ℝ := 6.80
def cost_cleats : ℝ := 40
def cost_water_bottle : ℝ := 12
def cost_one_player := cost_jersey + cost_shorts + cost_socks + cost_cleats + cost_water_bottle
def num_players : ℕ := 25
def total_cost_for_team : ℝ := cost_one_player * num_players

theorem equipment_total_cost :
  total_cost_for_team = 2475 := by
  sorry

end NUMINAMATH_GPT_equipment_total_cost_l1109_110961


namespace NUMINAMATH_GPT_fundraiser_total_money_l1109_110974

def fundraiser_money : ℝ :=
  let brownies_students := 70
  let brownies_each := 20
  let brownies_price := 1.50
  let cookies_students := 40
  let cookies_each := 30
  let cookies_price := 2.25
  let donuts_students := 35
  let donuts_each := 18
  let donuts_price := 3.00
  let cupcakes_students := 25
  let cupcakes_each := 12
  let cupcakes_price := 2.50
  let total_brownies := brownies_students * brownies_each
  let total_cookies := cookies_students * cookies_each
  let total_donuts := donuts_students * donuts_each
  let total_cupcakes := cupcakes_students * cupcakes_each
  let money_brownies := total_brownies * brownies_price
  let money_cookies := total_cookies * cookies_price
  let money_donuts := total_donuts * donuts_price
  let money_cupcakes := total_cupcakes * cupcakes_price
  money_brownies + money_cookies + money_donuts + money_cupcakes

theorem fundraiser_total_money : fundraiser_money = 7440 := sorry

end NUMINAMATH_GPT_fundraiser_total_money_l1109_110974


namespace NUMINAMATH_GPT_david_english_marks_l1109_110994

theorem david_english_marks :
  let Mathematics := 45
  let Physics := 72
  let Chemistry := 77
  let Biology := 75
  let AverageMarks := 68.2
  let TotalSubjects := 5
  let TotalMarks := AverageMarks * TotalSubjects
  let MarksInEnglish := TotalMarks - (Mathematics + Physics + Chemistry + Biology)
  MarksInEnglish = 72 :=
by
  sorry

end NUMINAMATH_GPT_david_english_marks_l1109_110994


namespace NUMINAMATH_GPT_anthony_balloon_count_l1109_110921

variable (Tom Luke Anthony : ℕ)

theorem anthony_balloon_count
  (h1 : Tom = 3 * Luke)
  (h2 : Luke = Anthony / 4)
  (hTom : Tom = 33) :
  Anthony = 44 := by
    sorry

end NUMINAMATH_GPT_anthony_balloon_count_l1109_110921


namespace NUMINAMATH_GPT_ratio_a_to_c_l1109_110901

theorem ratio_a_to_c (a b c : ℕ) (h1 : a / b = 5 / 3) (h2 : b / c = 1 / 5) : a / c = 1 / 3 :=
sorry

end NUMINAMATH_GPT_ratio_a_to_c_l1109_110901


namespace NUMINAMATH_GPT_symmetric_conic_transform_l1109_110931

open Real

theorem symmetric_conic_transform (x y : ℝ) 
  (h1 : 2 * x^2 + 4 * x * y + 5 * y^2 - 22 = 0)
  (h2 : x - y + 1 = 0) : 
  5 * x^2 + 4 * x * y + 2 * y^2 + 6 * x - 19 = 0 := 
sorry

end NUMINAMATH_GPT_symmetric_conic_transform_l1109_110931
