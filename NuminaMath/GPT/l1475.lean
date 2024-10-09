import Mathlib

namespace bridge_length_l1475_147571

theorem bridge_length (train_length : ℕ) (crossing_time : ℕ) (train_speed_kmh : ℕ) :
  train_length = 500 → crossing_time = 45 → train_speed_kmh = 64 → 
  ∃ (bridge_length : ℝ), bridge_length = 300.1 :=
by
  intros h1 h2 h3
  have speed_mps := (train_speed_kmh * 1000) / 3600
  have total_distance := speed_mps * crossing_time
  have bridge_length_calculated := total_distance - train_length
  use bridge_length_calculated
  sorry

end bridge_length_l1475_147571


namespace minimum_apples_l1475_147523

theorem minimum_apples (n : ℕ) : 
  n % 4 = 1 ∧ n % 5 = 2 ∧ n % 9 = 7 → n = 97 := 
by 
  -- To be proved
  sorry

end minimum_apples_l1475_147523


namespace g_is_odd_l1475_147568

noncomputable def g (x : ℝ) : ℝ := (7^x - 1) / (7^x + 1)

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x :=
by
  intros x
  sorry

end g_is_odd_l1475_147568


namespace infinitely_many_n_prime_l1475_147509

theorem infinitely_many_n_prime (p : ℕ) [Fact (Nat.Prime p)] : ∃ᶠ n in at_top, p ∣ 2^n - n := 
sorry

end infinitely_many_n_prime_l1475_147509


namespace find_a_l1475_147597

-- Assuming the existence of functions and variables as per conditions
variable (f : ℝ → ℝ)
variable (a : ℝ)
variable (x : ℝ)

-- Defining the given conditions
axiom cond1 : ∀ x : ℝ, f (1/2 * x - 1) = 2 * x - 5
axiom cond2 : f a = 6

-- Now stating the proof goal
theorem find_a : a = 7 / 4 := by
  sorry

end find_a_l1475_147597


namespace smallest_sum_of_squares_l1475_147543

theorem smallest_sum_of_squares (a b : ℕ) (h : a - b = 221) : a + b = 229 :=
sorry

end smallest_sum_of_squares_l1475_147543


namespace find_phi_l1475_147572

open Real

theorem find_phi (φ : ℝ) (hφ : |φ| < π / 2)
  (h_symm : ∀ x, sin (2 * x + φ) = sin (2 * ((2 * π / 3 - x) / 2) + φ)) :
  φ = -π / 6 :=
by
  sorry

end find_phi_l1475_147572


namespace correct_answer_l1475_147510

def vector := (Int × Int)

-- Definitions of vectors given in conditions
def m : vector := (2, 1)
def n : vector := (0, -2)

def vec_add (v1 v2 : vector) : vector :=
  (v1.1 + v2.1, v1.2 + v2.2)

def vec_scalar_mult (c : Int) (v : vector) : vector :=
  (c * v.1, c * v.2)

def vec_dot (v1 v2 : vector) : Int :=
  v1.1 * v2.1 + v1.2 * v2.2

-- The condition vector combined
def combined_vector := vec_add m (vec_scalar_mult 2 n)

-- The problem is to prove this:
theorem correct_answer : vec_dot (3, 2) combined_vector = 0 :=
  sorry

end correct_answer_l1475_147510


namespace total_problems_l1475_147564

theorem total_problems (math_pages reading_pages problems_per_page : ℕ) :
  math_pages = 2 →
  reading_pages = 4 →
  problems_per_page = 5 →
  (math_pages + reading_pages) * problems_per_page = 30 :=
by
  sorry

end total_problems_l1475_147564


namespace find_a_if_f_is_even_l1475_147554

noncomputable def f (x a : ℝ) : ℝ := (x + a) * (x - 2)

theorem find_a_if_f_is_even (a : ℝ) (h : ∀ x : ℝ, f x a = f (-x) a) : a = 2 := by
  sorry

end find_a_if_f_is_even_l1475_147554


namespace solution_set_of_inequality_l1475_147548

noncomputable def f (x : ℝ) : ℝ := 2^x - 2^(-x)

theorem solution_set_of_inequality :
  {x : ℝ | f (2 * x + 1) + f (1) ≥ 0} = {x : ℝ | -1 ≤ x} :=
by
  sorry

end solution_set_of_inequality_l1475_147548


namespace one_fourth_of_2_pow_30_eq_2_pow_x_l1475_147526

theorem one_fourth_of_2_pow_30_eq_2_pow_x (x : ℕ) : (1 / 4 : ℝ) * (2:ℝ)^30 = (2:ℝ)^x → x = 28 := by
  sorry

end one_fourth_of_2_pow_30_eq_2_pow_x_l1475_147526


namespace solutions_exist_l1475_147570

theorem solutions_exist (k : ℤ) : ∃ x y : ℤ, (x = 3 * k + 2) ∧ (y = 7 * k + 4) ∧ (7 * x - 3 * y = 2) :=
by {
  -- Proof will be filled in here
  sorry
}

end solutions_exist_l1475_147570


namespace Sarah_shampoo_conditioner_usage_l1475_147556

theorem Sarah_shampoo_conditioner_usage (daily_shampoo : ℝ) (daily_conditioner : ℝ) (days_in_week : ℝ) (weeks : ℝ) (total_days : ℝ) (daily_total : ℝ) (total_usage : ℝ) :
  daily_shampoo = 1 → 
  daily_conditioner = daily_shampoo / 2 → 
  days_in_week = 7 → 
  weeks = 2 → 
  total_days = days_in_week * weeks → 
  daily_total = daily_shampoo + daily_conditioner → 
  total_usage = daily_total * total_days → 
  total_usage = 21 := by
  sorry

end Sarah_shampoo_conditioner_usage_l1475_147556


namespace paul_packed_total_toys_l1475_147537

def small_box_small_toys : ℕ := 8
def medium_box_medium_toys : ℕ := 12
def large_box_large_toys : ℕ := 7
def large_box_small_toys : ℕ := 3
def small_box_medium_toys : ℕ := 5

def small_box : ℕ := small_box_small_toys + small_box_medium_toys
def medium_box : ℕ := medium_box_medium_toys
def large_box : ℕ := large_box_large_toys + large_box_small_toys

def total_toys : ℕ := small_box + medium_box + large_box

theorem paul_packed_total_toys : total_toys = 35 :=
by sorry

end paul_packed_total_toys_l1475_147537


namespace Bo_knew_percentage_l1475_147518

-- Definitions from the conditions
def total_flashcards := 800
def words_per_day := 16
def days := 40
def total_words_to_learn := words_per_day * days
def known_words := total_flashcards - total_words_to_learn

-- Statement that we need to prove
theorem Bo_knew_percentage : (known_words.toFloat / total_flashcards.toFloat) * 100 = 20 :=
by
  sorry  -- Proof is omitted as per the instructions

end Bo_knew_percentage_l1475_147518


namespace diamond_2_3_l1475_147502

def diamond (a b : ℕ) : ℕ := a * b^2 - b + 1

theorem diamond_2_3 : diamond 2 3 = 16 :=
by
  -- Imported definition and theorem structure.
  sorry

end diamond_2_3_l1475_147502


namespace degenerate_ellipse_single_point_l1475_147515

theorem degenerate_ellipse_single_point (c : ℝ) :
  (∀ x y : ℝ, 3 * x^2 + y^2 + 6 * x - 12 * y + c = 0 → (x = -1 ∧ y = 6)) ↔ c = -39 :=
by
  sorry

end degenerate_ellipse_single_point_l1475_147515


namespace total_money_collected_l1475_147514

def number_of_people := 610
def price_adult := 2
def price_child := 1
def number_of_adults := 350

theorem total_money_collected :
  (number_of_people - number_of_adults) * price_child + number_of_adults * price_adult = 960 := by
  sorry

end total_money_collected_l1475_147514


namespace polygon_sides_from_diagonals_l1475_147585

theorem polygon_sides_from_diagonals (D : ℕ) (hD : D = 16) : 
  ∃ n : ℕ, 2 * D = n * (n - 3) ∧ n = 7 :=
by
  use 7
  simp [hD]
  norm_num
  sorry

end polygon_sides_from_diagonals_l1475_147585


namespace prove_trig_inequality_l1475_147586

noncomputable def trig_inequality : Prop :=
  (0 < 1 / 2) ∧ (1 / 2 < Real.pi / 6) ∧
  (∀ x y : ℝ, (0 < x) ∧ (x < y) ∧ (y < Real.pi / 6) → Real.sin x < Real.sin y) ∧
  (∀ x y : ℝ, (0 < x) ∧ (x < y) ∧ (y < Real.pi / 6) → Real.cos x > Real.cos y) →
  (Real.cos (1 / 2) > Real.tan (1 / 2) ∧ Real.tan (1 / 2) > Real.sin (1 / 2))

theorem prove_trig_inequality : trig_inequality :=
by
  sorry

end prove_trig_inequality_l1475_147586


namespace projection_ratio_zero_l1475_147590

variables (v w u p q : ℝ → ℝ) -- Assuming vectors are functions from ℝ to ℝ
variables (norm : (ℝ → ℝ) → ℝ) -- norm is a function from vectors to ℝ
variables (proj : (ℝ → ℝ) → (ℝ → ℝ) → (ℝ → ℝ)) -- proj is the projection function

-- Assume the conditions
axiom proj_p : p = proj v w
axiom proj_q : q = proj p u
axiom perp_uv : ∀ t, v t * u t = 0 -- u is perpendicular to v
axiom norm_ratio : norm p / norm v = 3 / 8

theorem projection_ratio_zero : norm q / norm v = 0 :=
by sorry

end projection_ratio_zero_l1475_147590


namespace ticket_sales_revenue_l1475_147574

theorem ticket_sales_revenue (total_tickets advance_tickets same_day_tickets price_advance price_same_day: ℕ) 
    (h1: total_tickets = 60) 
    (h2: price_advance = 20) 
    (h3: price_same_day = 30) 
    (h4: advance_tickets = 20) 
    (h5: same_day_tickets = total_tickets - advance_tickets):
    advance_tickets * price_advance + same_day_tickets * price_same_day = 1600 := 
by
  sorry

end ticket_sales_revenue_l1475_147574


namespace ball_max_height_l1475_147508

theorem ball_max_height : 
  (∃ t : ℝ, 
    ∀ u : ℝ, -16 * u ^ 2 + 80 * u + 35 ≤ -16 * t ^ 2 + 80 * t + 35 ∧ 
    -16 * t ^ 2 + 80 * t + 35 = 135) :=
sorry

end ball_max_height_l1475_147508


namespace sum_of_squares_of_roots_l1475_147562

theorem sum_of_squares_of_roots (s₁ s₂ : ℝ) (h1 : s₁ + s₂ = 9) (h2 : s₁ * s₂ = 14) :
  s₁^2 + s₂^2 = 53 :=
by
  sorry

end sum_of_squares_of_roots_l1475_147562


namespace umbrella_cost_l1475_147589

theorem umbrella_cost (number_of_umbrellas : Nat) (total_cost : Nat) (h1 : number_of_umbrellas = 3) (h2 : total_cost = 24) :
  (total_cost / number_of_umbrellas) = 8 :=
by
  -- The proof will go here
  sorry

end umbrella_cost_l1475_147589


namespace subcommittee_count_l1475_147512

-- Define the conditions: number of Republicans and Democrats in the Senate committee
def numRepublicans : ℕ := 10
def numDemocrats : ℕ := 8
def chooseRepublicans : ℕ := 4
def chooseDemocrats : ℕ := 3

-- Define the main proof problem based on the conditions and the correct answer
theorem subcommittee_count :
  (Nat.choose numRepublicans chooseRepublicans) * (Nat.choose numDemocrats chooseDemocrats) = 11760 := by
  sorry

end subcommittee_count_l1475_147512


namespace find_initial_number_l1475_147565

theorem find_initial_number (x : ℕ) (h : ∃ y : ℕ, x * y = 4 ∧ y = 2) : x = 2 :=
by
  sorry

end find_initial_number_l1475_147565


namespace kira_night_songs_l1475_147542

-- Definitions for the conditions
def morning_songs : ℕ := 10
def later_songs : ℕ := 15
def song_size_mb : ℕ := 5
def total_new_songs_memory_mb : ℕ := 140

-- Assert the number of songs Kira downloaded at night
theorem kira_night_songs : (total_new_songs_memory_mb - (morning_songs * song_size_mb + later_songs * song_size_mb)) / song_size_mb = 3 :=
by
  sorry

end kira_night_songs_l1475_147542


namespace total_hours_watched_l1475_147582

/-- Given a 100-hour long video, Lila watches it at twice the average speed, and Roger watches it at the average speed. Both watched six such videos. We aim to prove that the total number of hours watched by Lila and Roger together is 900 hours. -/
theorem total_hours_watched {video_length lila_speed_multiplier roger_speed_multiplier num_videos : ℕ} 
  (h1 : video_length = 100)
  (h2 : lila_speed_multiplier = 2) 
  (h3 : roger_speed_multiplier = 1)
  (h4 : num_videos = 6) :
  (num_videos * (video_length / lila_speed_multiplier) + num_videos * (video_length / roger_speed_multiplier)) = 900 := 
sorry

end total_hours_watched_l1475_147582


namespace claim1_claim2_l1475_147539

theorem claim1 (n : ℤ) (hs : ∃ l : List ℤ, l.length = n ∧ l.prod = n ∧ l.sum = 0) : 
  ∃ k : ℤ, n = 4 * k := 
sorry

theorem claim2 (n : ℕ) (h : n % 4 = 0) : 
  ∃ l : List ℤ, l.length = n ∧ l.prod = n ∧ l.sum = 0 := 
sorry

end claim1_claim2_l1475_147539


namespace school_dance_attendance_l1475_147595

theorem school_dance_attendance (P : ℝ)
  (h1 : 0.1 * P = (P - (0.9 * P)))
  (h2 : 0.9 * P = (2/3) * (0.9 * P) + (1/3) * (0.9 * P))
  (h3 : 30 = (1/3) * (0.9 * P)) :
  P = 100 :=
by
  sorry

end school_dance_attendance_l1475_147595


namespace pairball_playing_time_l1475_147553

-- Define the conditions of the problem
def num_children : ℕ := 7
def total_minutes : ℕ := 105
def total_child_minutes : ℕ := 2 * total_minutes

-- Define the theorem to prove
theorem pairball_playing_time : total_child_minutes / num_children = 30 :=
by sorry

end pairball_playing_time_l1475_147553


namespace distance_of_points_in_polar_coordinates_l1475_147519

theorem distance_of_points_in_polar_coordinates
  (A : Real × Real) (B : Real × Real) (θ1 θ2 : Real)
  (hA : A = (5, θ1)) (hB : B = (12, θ2))
  (hθ : θ1 - θ2 = Real.pi / 2) : 
  dist (5 * Real.cos θ1, 5 * Real.sin θ1) (12 * Real.cos θ2, 12 * Real.sin θ2) = 13 := 
by sorry

end distance_of_points_in_polar_coordinates_l1475_147519


namespace number_not_equal_54_l1475_147522

def initial_number : ℕ := 12
def target_number : ℕ := 54
def total_time : ℕ := 60

theorem number_not_equal_54 (n : ℕ) (time : ℕ) : (time = total_time) → (n = initial_number) → 
  (∀ t : ℕ, t ≤ time → (n = n * 2 ∨ n = n / 2 ∨ n = n * 3 ∨ n = n / 3)) → n ≠ target_number :=
by
  sorry

end number_not_equal_54_l1475_147522


namespace sqrt_expression_simplify_l1475_147521

theorem sqrt_expression_simplify : 
  2 * Real.sqrt 12 * (Real.sqrt 3 / 4) / (10 * Real.sqrt 2) = 3 * Real.sqrt 2 / 20 :=
by 
  sorry

end sqrt_expression_simplify_l1475_147521


namespace pyramid_base_sidelength_l1475_147530

theorem pyramid_base_sidelength (A : ℝ) (h : ℝ) (s : ℝ) 
  (hA : A = 120) (hh : h = 24) (area_eq : A = 1/2 * s * h) : s = 10 := by
  sorry

end pyramid_base_sidelength_l1475_147530


namespace height_of_first_podium_l1475_147584

noncomputable def height_of_podium_2_cm := 53.0
noncomputable def height_of_podium_2_mm := 7.0
noncomputable def height_on_podium_2_cm := 190.0
noncomputable def height_on_podium_1_cm := 232.0
noncomputable def height_on_podium_1_mm := 5.0

def expected_height_of_podium_1_cm := 96.2

theorem height_of_first_podium :
  let height_podium_2 := height_of_podium_2_cm + height_of_podium_2_mm / 10.0
  let height_podium_1 := height_on_podium_1_cm + height_on_podium_1_mm / 10.0
  let hyeonjoo_height := height_on_podium_2_cm - height_podium_2
  height_podium_1 - hyeonjoo_height = expected_height_of_podium_1_cm :=
by sorry

end height_of_first_podium_l1475_147584


namespace second_root_of_quadratic_l1475_147507

theorem second_root_of_quadratic (p q r : ℝ) (quad_eqn : ∀ x, 2 * p * (q - r) * x^2 + 3 * q * (r - p) * x + 4 * r * (p - q) = 0) (root : 2 * p * (q - r) * 2^2 + 3 * q * (r - p) * 2 + 4 * r * (p - q) = 0) :
    ∃ r₂ : ℝ, r₂ = (r * (p - q)) / (p * (q - r)) :=
sorry

end second_root_of_quadratic_l1475_147507


namespace cost_of_apple_l1475_147503

variable (A O : ℝ)

theorem cost_of_apple :
  (6 * A + 3 * O = 1.77) ∧ (2 * A + 5 * O = 1.27) → A = 0.21 :=
by
  intro h
  -- Proof goes here
  sorry

end cost_of_apple_l1475_147503


namespace units_digit_k_squared_plus_two_exp_k_eq_7_l1475_147577

/-- Define k as given in the problem -/
def k : ℕ := 2010^2 + 2^2010

/-- Final statement that needs to be proved -/
theorem units_digit_k_squared_plus_two_exp_k_eq_7 : (k^2 + 2^k) % 10 = 7 := 
by
  sorry

end units_digit_k_squared_plus_two_exp_k_eq_7_l1475_147577


namespace value_of_a_l1475_147599

theorem value_of_a (x : ℝ) (n : ℕ) (h : x > 0) (h_n : n > 0) :
  (∀ k : ℕ, 1 ≤ k → k ≤ n → x + k ≥ k + 1) → a = n^n :=
by
  sorry

end value_of_a_l1475_147599


namespace problem_statement_l1475_147500

noncomputable def expr (x y z : ℝ) : ℝ :=
  (x^2 * y^2) / ((x^2 - y*z) * (y^2 - x*z)) +
  (x^2 * z^2) / ((x^2 - y*z) * (z^2 - x*y)) +
  (y^2 * z^2) / ((y^2 - x*z) * (z^2 - x*y))

theorem problem_statement (x y z : ℝ) (h₁ : x ≠ 0) (h₂ : y ≠ 0) (h₃ : z ≠ 0) (h₄ : x + y + z = -1) :
  expr x y z = 1 := by
  sorry

end problem_statement_l1475_147500


namespace required_fencing_l1475_147587

-- Definitions from conditions
def length_uncovered : ℝ := 30
def area : ℝ := 720

-- Prove that the amount of fencing required is 78 feet
theorem required_fencing : 
  ∃ (W : ℝ), (area = length_uncovered * W) ∧ (2 * W + length_uncovered = 78) := 
sorry

end required_fencing_l1475_147587


namespace cylindrical_to_rectangular_l1475_147546

theorem cylindrical_to_rectangular (r θ z : ℝ) 
  (h₁ : r = 7) (h₂ : θ = 5 * Real.pi / 4) (h₃ : z = 6) : 
  (r * Real.cos θ, r * Real.sin θ, z) = 
  (-7 * Real.sqrt 2 / 2, -7 * Real.sqrt 2 / 2, 6) := 
by 
  sorry

end cylindrical_to_rectangular_l1475_147546


namespace find_f_neg_2010_6_l1475_147598

noncomputable def f : ℝ → ℝ := sorry

axiom f_add_one (x : ℝ) : f (x + 1) + f x = 3

axiom f_on_interval (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) : f x = 2 - x

theorem find_f_neg_2010_6 : f (-2010.6) = 1.4 := by {
  sorry
}

end find_f_neg_2010_6_l1475_147598


namespace part1_l1475_147576

variables (a c : ℝ × ℝ)
variables (a_parallel_c : ∃ k : ℝ, c = (k * a.1, k * a.2))
variables (a_value : a = (1,2))
variables (c_magnitude : (c.1 ^ 2 + c.2 ^ 2) = (3 * Real.sqrt 5) ^ 2)

theorem part1: c = (3, 6) ∨ c = (-3, -6) :=
by
  sorry

end part1_l1475_147576


namespace swimming_pool_area_l1475_147535

open Nat

-- Define the width (w) and length (l) with given conditions
def width (w : ℕ) : Prop :=
  exists (l : ℕ), l = 2 * w + 40 ∧ 2 * w + 2 * l = 800

-- Define the area of the swimming pool
def pool_area (w l : ℕ) : ℕ :=
  w * l

theorem swimming_pool_area : 
  ∃ (w l : ℕ), width w ∧ width l -> pool_area w l = 33600 :=
by
  sorry

end swimming_pool_area_l1475_147535


namespace cost_of_bench_l1475_147532

variables (cost_table cost_bench : ℕ)

theorem cost_of_bench :
  cost_table + cost_bench = 450 ∧ cost_table = 2 * cost_bench → cost_bench = 150 :=
by
  sorry

end cost_of_bench_l1475_147532


namespace average_marks_l1475_147533

theorem average_marks (M P C : ℕ) (h1 : M + P = 60) (h2 : C = P + 10) : (M + C) / 2 = 35 := 
by
  sorry

end average_marks_l1475_147533


namespace center_square_number_l1475_147552

def in_center_square (grid : Matrix (Fin 3) (Fin 3) ℕ) : ℕ := grid 1 1

theorem center_square_number
  (grid : Matrix (Fin 3) (Fin 3) ℕ)
  (consecutive_share_edge : ∀ (i j : Fin 3) (n : ℕ), 
                              (i < 2 ∨ j < 2) →
                              (∃ d, d ∈ [(-1,0), (1,0), (0,-1), (0,1)] ∧ 
                              grid (i + d.1) (j + d.2) = n + 1))
  (corner_sum_20 : grid 0 0 + grid 0 2 + grid 2 0 + grid 2 2 = 20)
  (diagonal_sum_15 : 
    (grid 0 0 + grid 1 1 + grid 2 2 = 15) 
    ∨ 
    (grid 0 2 + grid 1 1 + grid 2 0 = 15))
  : in_center_square grid = 5 := sorry

end center_square_number_l1475_147552


namespace find_F_l1475_147513

theorem find_F (C F : ℝ) (h1 : C = (4 / 7) * (F - 40)) (h2 : C = 35) : F = 101.25 :=
  sorry

end find_F_l1475_147513


namespace sum_of_numbers_in_ratio_with_lcm_l1475_147593

theorem sum_of_numbers_in_ratio_with_lcm
  (x : ℕ)
  (h1 : Nat.lcm (2 * x) (Nat.lcm (3 * x) (5 * x)) = 120) :
  (2 * x) + (3 * x) + (5 * x) = 40 := 
sorry

end sum_of_numbers_in_ratio_with_lcm_l1475_147593


namespace imo1983_q24_l1475_147528

theorem imo1983_q24 :
  ∃ (S : Finset ℕ), S.card = 1983 ∧ 
    (∀ x ∈ S, x > 0 ∧ x ≤ 10^5) ∧
    (∀ (x y z : ℕ), x ∈ S → y ∈ S → z ∈ S → x ≠ y → x ≠ z → y ≠ z → (x + z ≠ 2 * y)) :=
sorry

end imo1983_q24_l1475_147528


namespace exists_unique_n_digit_number_with_one_l1475_147504

def n_digit_number (n : ℕ) : Type := {l : List ℕ // l.length = n ∧ ∀ x ∈ l, x = 1 ∨ x = 2 ∨ x = 3}

theorem exists_unique_n_digit_number_with_one (n : ℕ) (hn : n > 0) :
  ∃ x : n_digit_number n, x.val.count 1 = 1 ∧ ∀ y : n_digit_number n, y ≠ x → x.val.append [1] ≠ y.val.append [1] :=
sorry

end exists_unique_n_digit_number_with_one_l1475_147504


namespace f_7_eq_minus_1_l1475_147511

-- Define the odd function f with the given properties
def is_odd_function (f : ℝ → ℝ) :=
  ∀ x, f (-x) = -f x

def period_2 (f : ℝ → ℝ) :=
  ∀ x, f (x + 2) = -f x

def f_restricted (f : ℝ → ℝ) :=
  ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 -> f x = x

-- The main statement: Under the given conditions, f(7) = -1
theorem f_7_eq_minus_1 (f : ℝ → ℝ)
  (H1 : is_odd_function f)
  (H2 : period_2 f)
  (H3 : f_restricted f) :
  f 7 = -1 :=
by
  sorry

end f_7_eq_minus_1_l1475_147511


namespace equation_b_not_symmetric_about_x_axis_l1475_147588

def equationA (x y : ℝ) : Prop := x^2 - x + y^2 = 1
def equationB (x y : ℝ) : Prop := x^2 * y + x * y^2 = 1
def equationC (x y : ℝ) : Prop := 2 * x^2 - y^2 = 1
def equationD (x y : ℝ) : Prop := x + y^2 = -1

def symmetric_about_x_axis (f : ℝ → ℝ → Prop) : Prop :=
  ∀ x y : ℝ, f x y ↔ f x (-y)

theorem equation_b_not_symmetric_about_x_axis : 
  ¬ symmetric_about_x_axis (equationB) :=
sorry

end equation_b_not_symmetric_about_x_axis_l1475_147588


namespace sum_of_coordinates_of_point_D_l1475_147527

theorem sum_of_coordinates_of_point_D
  (N : ℝ × ℝ := (6,2))
  (C : ℝ × ℝ := (10, -2))
  (h : ∃ D : ℝ × ℝ, (N = ((C.1 + D.1) / 2, (C.2 + D.2) / 2))) :
  ∃ (D : ℝ × ℝ), D.1 + D.2 = 8 := 
by
  obtain ⟨D, hD⟩ := h
  sorry

end sum_of_coordinates_of_point_D_l1475_147527


namespace possible_values_for_N_l1475_147517

theorem possible_values_for_N (N : ℕ) (h₁ : 8 < N) (h₂ : 1 ≤ N - 1) :
  22 < N ∧ N ≤ 25 ↔ (N = 23 ∨ N = 24 ∨ N = 25) :=
by 
  sorry

end possible_values_for_N_l1475_147517


namespace correct_addition_result_l1475_147524

-- Define the particular number x and state the condition.
variable (x : ℕ) (h₁ : x + 21 = 52)

-- Assert that the correct result when adding 40 to x is 71.
theorem correct_addition_result : x + 40 = 71 :=
by
  -- Proof would go here; represented as a placeholder for now.
  sorry

end correct_addition_result_l1475_147524


namespace intersection_polar_coords_l1475_147525

noncomputable def polar_coord_intersection (rho theta : ℝ) : Prop :=
  (rho * (Real.sqrt 3 * Real.cos theta - Real.sin theta) = 2) ∧ (rho = 4 * Real.sin theta)

theorem intersection_polar_coords :
  ∃ (rho theta : ℝ), polar_coord_intersection rho theta ∧ rho = 2 ∧ theta = (Real.pi / 6) := 
sorry

end intersection_polar_coords_l1475_147525


namespace find_coefficients_sum_l1475_147534

theorem find_coefficients_sum (a_0 a_1 a_2 a_3 : ℝ) (h : ∀ x : ℝ, x^3 = a_0 + a_1 * (x-2) + a_2 * (x-2)^2 + a_3 * (x-2)^3) :
  a_1 + a_2 + a_3 = 19 :=
by
  sorry

end find_coefficients_sum_l1475_147534


namespace combined_mass_of_individuals_l1475_147520

-- Define constants and assumptions
def boat_length : ℝ := 4 -- in meters
def boat_breadth : ℝ := 3 -- in meters
def sink_depth_first_person : ℝ := 0.01 -- in meters (1 cm)
def sink_depth_second_person : ℝ := 0.02 -- in meters (2 cm)
def density_water : ℝ := 1000 -- in kg/m³ (density of freshwater)

-- Define volumes displaced
def volume_displaced_first : ℝ := boat_length * boat_breadth * sink_depth_first_person
def volume_displaced_both : ℝ := boat_length * boat_breadth * (sink_depth_first_person + sink_depth_second_person)

-- Define weights (which are equal to the masses under the assumption of constant gravity)
def weight_first_person : ℝ := volume_displaced_first * density_water
def weight_both_persons : ℝ := volume_displaced_both * density_water

-- Statement to prove the combined weight
theorem combined_mass_of_individuals : weight_both_persons = 360 :=
by
  -- Skip the proof
  sorry

end combined_mass_of_individuals_l1475_147520


namespace range_of_a_l1475_147505

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 - a * x + 1 > 0) ↔ (-2 < a ∧ a < 2) :=
sorry

end range_of_a_l1475_147505


namespace shaded_area_l1475_147547

/--
Given a larger square containing a smaller square entirely within it,
where the side length of the smaller square is 5 units
and the side length of the larger square is 10 units,
prove that the area of the shaded region (the area of the larger square minus the area of the smaller square) is 75 square units.
-/
theorem shaded_area :
  let side_length_smaller := 5
  let side_length_larger := 10
  let area_larger := side_length_larger * side_length_larger
  let area_smaller := side_length_smaller * side_length_smaller
  area_larger - area_smaller = 75 := 
by
  let side_length_smaller := 5
  let side_length_larger := 10
  let area_larger := side_length_larger * side_length_larger
  let area_smaller := side_length_smaller * side_length_smaller
  sorry

end shaded_area_l1475_147547


namespace identity_proof_l1475_147538

theorem identity_proof (a b c : ℝ) (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) :
    (b - c) / ((a - b) * (a - c)) + (c - a) / ((b - c) * (b - a)) + (a - b) / ((c - a) * (c - b)) =
    2 / (a - b) + 2 / (b - c) + 2 / (c - a) :=
by
  sorry

end identity_proof_l1475_147538


namespace min_value_expression_l1475_147581

theorem min_value_expression :
  ∃ x : ℝ, (x+2) * (x+3) * (x+5) * (x+6) + 2024 = 2021.75 :=
sorry

end min_value_expression_l1475_147581


namespace garden_roller_area_l1475_147551

theorem garden_roller_area (D : ℝ) (A : ℝ) (π : ℝ) (L_new : ℝ) :
  D = 1.4 → A = 88 → π = 22/7 → L_new = 4 → A = 5 * (2 * π * (D / 2) * L_new) :=
by sorry

end garden_roller_area_l1475_147551


namespace cars_in_fourth_store_l1475_147501

theorem cars_in_fourth_store
  (mean : ℝ) 
  (a1 a2 a3 a5 : ℝ) 
  (num_stores : ℝ) 
  (mean_value : mean = 20.8) 
  (a1_value : a1 = 30) 
  (a2_value : a2 = 14) 
  (a3_value : a3 = 14) 
  (a5_value : a5 = 25) 
  (num_stores_value : num_stores = 5) :
  ∃ x : ℝ, (a1 + a2 + a3 + x + a5) / num_stores = mean ∧ x = 21 :=
by
  sorry

end cars_in_fourth_store_l1475_147501


namespace smallest_a_gcd_77_88_l1475_147569

theorem smallest_a_gcd_77_88 :
  ∃ (a : ℕ), a > 0 ∧ (∀ b, b > 0 → b < a → (gcd b 77 > 1 ∧ gcd b 88 > 1) → false) ∧ gcd a 77 > 1 ∧ gcd a 88 > 1 ∧ a = 11 :=
by
  sorry

end smallest_a_gcd_77_88_l1475_147569


namespace triangle_OMN_area_l1475_147544

noncomputable def rho (theta : ℝ) : ℝ := 4 * Real.cos theta + 2 * Real.sin theta

theorem triangle_OMN_area :
  let l1 (x y : ℝ) := y = (Real.sqrt 3 / 3) * x
  let l2 (x y : ℝ) := y = Real.sqrt 3 * x
  let C (x y : ℝ) := (x - 2)^2 + (y - 1)^2 = 5
  let OM := 2 * Real.sqrt 3 + 1
  let ON := 2 + Real.sqrt 3
  let angle_MON := Real.pi / 6
  let area_OMN := (1 / 2) * OM * ON * Real.sin angle_MON
  (4 * (Real.sqrt 3 + 2) + 5 * Real.sqrt 3 = 8 + 5 * Real.sqrt 3) → 
  area_OMN = (8 + 5 * Real.sqrt 3) / 4 :=
sorry

end triangle_OMN_area_l1475_147544


namespace costco_container_holds_one_gallon_l1475_147529

theorem costco_container_holds_one_gallon
  (costco_cost : ℕ := 8)
  (store_cost_per_bottle : ℕ := 3)
  (savings : ℕ := 16)
  (ounces_per_bottle : ℕ := 16)
  (ounces_per_gallon : ℕ := 128) :
  ∃ (gallons : ℕ), gallons = 1 :=
by
  sorry

end costco_container_holds_one_gallon_l1475_147529


namespace plane_equation_l1475_147557

theorem plane_equation :
  ∃ (A B C D : ℤ), A > 0 ∧ Int.gcd (Int.natAbs A) (Int.gcd (Int.natAbs B) (Int.gcd (Int.natAbs C) (Int.natAbs D))) = 1 ∧ 
  (∀ (x y z : ℤ), (x, y, z) = (0, 0, 0) ∨ (x, y, z) = (2, 0, -2) → A * x + B * y + C * z + D = 0) ∧ 
  ∀ (x y z : ℤ), (A = 1 ∧ B = -5 ∧ C = 1 ∧ D = 0) := sorry

end plane_equation_l1475_147557


namespace hall_length_l1475_147563

theorem hall_length (L B A : ℝ) (h1 : B = 2 / 3 * L) (h2 : A = 2400) (h3 : A = L * B) : L = 60 := by
  -- proof steps here
  sorry

end hall_length_l1475_147563


namespace sum_of_neg_ints_l1475_147575

theorem sum_of_neg_ints (xs : List Int) (h₁ : ∀ x ∈ xs, x < 0)
  (h₂ : ∀ x ∈ xs, 3 < |x| ∧ |x| < 6) : xs.sum = -9 :=
sorry

end sum_of_neg_ints_l1475_147575


namespace minimize_J_l1475_147559

noncomputable def H (p q : ℝ) : ℝ :=
  -3 * p * q + 4 * p * (1 - q) + 4 * (1 - p) * q - 5 * (1 - p) * (1 - q)

noncomputable def J (p : ℝ) : ℝ :=
  if p < 0 then 0 else if p > 1 then 1 else if (9 * p - 5 > 4 - 7 * p) then 9 * p - 5 else 4 - 7 * p

theorem minimize_J :
  ∃ (p : ℝ), 0 ≤ p ∧ p ≤ 1 ∧ J p = J (9 / 16) := by
  sorry

end minimize_J_l1475_147559


namespace linda_loan_interest_difference_l1475_147506

theorem linda_loan_interest_difference :
  let P : ℝ := 8000
  let r : ℝ := 0.10
  let t : ℕ := 3
  let n_monthly : ℕ := 12
  let n_annual : ℕ := 1
  let A_monthly : ℝ := P * (1 + r / (n_monthly : ℝ))^(n_monthly * t)
  let A_annual : ℝ := P * (1 + r)^t
  A_monthly - A_annual = 151.07 :=
by
  sorry

end linda_loan_interest_difference_l1475_147506


namespace exists_root_interval_l1475_147580

def f (x : ℝ) : ℝ := x^2 + 12 * x - 15

theorem exists_root_interval :
  (f 1.1 < 0) ∧ (f 1.2 > 0) → ∃ x : ℝ, 1.1 < x ∧ x < 1.2 ∧ f x = 0 := 
by
  intro h
  sorry

end exists_root_interval_l1475_147580


namespace f_always_positive_l1475_147592

noncomputable def f (x : ℝ) : ℝ := x^8 - x^5 + x^2 - x + 1

theorem f_always_positive : ∀ x : ℝ, 0 < f x := by
  sorry

end f_always_positive_l1475_147592


namespace price_of_second_set_of_knives_l1475_147536

def john_visits_houses_per_day : ℕ := 50
def percent_buying_per_day : ℝ := 0.20
def price_first_set : ℝ := 50
def weekly_sales : ℝ := 5000
def work_days_per_week : ℕ := 5

theorem price_of_second_set_of_knives
  (john_visits_houses_per_day : ℕ)
  (percent_buying_per_day : ℝ)
  (price_first_set : ℝ)
  (weekly_sales : ℝ)
  (work_days_per_week : ℕ) :
  0 < percent_buying_per_day ∧ percent_buying_per_day ≤ 1 ∧
  weekly_sales = 5000 ∧ 
  work_days_per_week = 5 ∧
  john_visits_houses_per_day = 50 ∧
  price_first_set = 50 → 
  (∃ price_second_set : ℝ, price_second_set = 150) :=
  sorry

end price_of_second_set_of_knives_l1475_147536


namespace binary_to_decimal_l1475_147516

/-- The binary number 1011 (base 2) equals 11 (base 10). -/
theorem binary_to_decimal : (1 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0) = 11 := by
  sorry

end binary_to_decimal_l1475_147516


namespace factor_expression_l1475_147596

variable (b : ℤ)

theorem factor_expression : 221 * b^2 + 17 * b = 17 * b * (13 * b + 1) :=
by
  sorry

end factor_expression_l1475_147596


namespace total_earnings_l1475_147566

theorem total_earnings (x y : ℕ) 
  (h1 : 2 * x * y = 250) : 
  58 * (x * y) = 7250 := 
by
  sorry

end total_earnings_l1475_147566


namespace arithmetic_operation_equals_l1475_147550

theorem arithmetic_operation_equals :
  12.1212 + 17.0005 - 9.1103 = 20.0114 := 
by 
  sorry

end arithmetic_operation_equals_l1475_147550


namespace calculate_expression_l1475_147561

theorem calculate_expression :
  (5 * 7 + 10 * 4 - 35 / 5 + 18 / 3 : ℝ) = 74 := by
  sorry

end calculate_expression_l1475_147561


namespace find_p_l1475_147560

variable (a b c p : ℚ)

theorem find_p (h1 : 5 / (a + b) = p / (a + c)) (h2 : p / (a + c) = 8 / (c - b)) : p = 13 := by
  sorry

end find_p_l1475_147560


namespace age_product_difference_is_nine_l1475_147558

namespace ArnoldDanny

def current_age := 4
def product_today (A : ℕ) := A * A
def product_next_year (A : ℕ) := (A + 1) * (A + 1)
def difference (A : ℕ) := product_next_year A - product_today A

theorem age_product_difference_is_nine :
  difference current_age = 9 :=
by
  sorry

end ArnoldDanny

end age_product_difference_is_nine_l1475_147558


namespace division_reciprocal_multiplication_l1475_147579

theorem division_reciprocal_multiplication : (4 / (8 / 13 : ℚ)) = (13 / 2 : ℚ) := 
by
  sorry

end division_reciprocal_multiplication_l1475_147579


namespace exists_root_in_interval_l1475_147549

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 5

theorem exists_root_in_interval : ∃ x, (2 < x ∧ x < 3) ∧ f x = 0 := 
by
  -- Assuming f(2) < 0 and f(3) > 0
  have h1 : f 2 < 0 := sorry
  have h2 : f 3 > 0 := sorry
  -- From the intermediate value theorem, there exists a c in (2, 3) such that f(c) = 0
  sorry

end exists_root_in_interval_l1475_147549


namespace treble_of_doubled_and_increased_l1475_147583

theorem treble_of_doubled_and_increased (initial_number : ℕ) (result : ℕ) : 
  initial_number = 15 → (initial_number * 2 + 5) * 3 = result → result = 105 := 
by 
  intros h1 h2
  rw [h1] at h2
  linarith

end treble_of_doubled_and_increased_l1475_147583


namespace josh_total_candies_l1475_147540

def josh_initial_candies (initial_candies given_siblings : ℕ) : Prop :=
  ∃ (remaining_1 best_friend josh_eats share_others : ℕ),
    (remaining_1 = initial_candies - given_siblings) ∧
    (best_friend = remaining_1 / 2) ∧
    (josh_eats = 16) ∧
    (share_others = 19) ∧
    (remaining_1 = 2 * (josh_eats + share_others))

theorem josh_total_candies : josh_initial_candies 100 30 :=
by
  sorry

end josh_total_candies_l1475_147540


namespace calculate_120_percent_l1475_147573

theorem calculate_120_percent (x : ℝ) (h : 0.20 * x = 100) : 1.20 * x = 600 :=
sorry

end calculate_120_percent_l1475_147573


namespace ratio_bisector_circumradius_l1475_147594

theorem ratio_bisector_circumradius (h_a h_b h_c : ℝ) (ha_val : h_a = 1/3) (hb_val : h_b = 1/4) (hc_val : h_c = 1/5) :
  ∃ (CD R : ℝ), CD / R = 24 * Real.sqrt 2 / 35 :=
by
  sorry

end ratio_bisector_circumradius_l1475_147594


namespace intersection_M_N_l1475_147567

def M : Set ℝ := { x | -1 < x ∧ x < 1 }
def N : Set ℝ := { x | x / (x - 1) ≤ 0 }

theorem intersection_M_N :
  M ∩ N = { x | 0 ≤ x ∧ x < 1 } :=
by
  sorry

end intersection_M_N_l1475_147567


namespace variance_of_scores_l1475_147555

-- Define the student's scores
def scores : List ℕ := [130, 125, 126, 126, 128]

-- Define a function to calculate the mean
def mean (l : List ℕ) : ℕ :=
  l.sum / l.length

-- Define a function to calculate the variance
def variance (l : List ℕ) : ℕ :=
  let avg := mean l
  (l.map (λ x => (x - avg) * (x - avg))).sum / l.length

-- The proof statement (no proof provided, use sorry)
theorem variance_of_scores : variance scores = 3 := by sorry

end variance_of_scores_l1475_147555


namespace max_value_of_k_l1475_147541

noncomputable def max_possible_k (x y : ℝ) (k : ℝ) : Prop :=
  0 < x ∧ 0 < y ∧ 0 < k ∧
  (3 = k^2 * (x^2 / y^2 + y^2 / x^2) + k * (x / y + y / x))

theorem max_value_of_k (x y : ℝ) (k : ℝ) :
  max_possible_k x y k → k ≤ (-1 + Real.sqrt 7) / 2 :=
sorry

end max_value_of_k_l1475_147541


namespace units_digit_product_l1475_147578

theorem units_digit_product :
  let nums : List Nat := [7, 17, 27, 37, 47, 57, 67, 77, 87, 97]
  let product := nums.prod
  (product % 10) = 9 :=
by
  sorry

end units_digit_product_l1475_147578


namespace rectangle_height_l1475_147591

-- Defining the conditions
def base : ℝ := 9
def area : ℝ := 33.3

-- Stating the proof problem
theorem rectangle_height : (area / base) = 3.7 :=
by
  sorry

end rectangle_height_l1475_147591


namespace find_bc_l1475_147531

noncomputable def setA : Set ℝ := {x | x^2 + x - 2 ≤ 0}
noncomputable def setB : Set ℝ := {x | 2 < x + 1 ∧ x + 1 ≤ 4}
noncomputable def setAB : Set ℝ := setA ∪ setB
noncomputable def setC (b c : ℝ) : Set ℝ := {x | x^2 + b * x + c > 0}

theorem find_bc (b c : ℝ) :
  (setAB ∩ setC b c = ∅) ∧ (setAB ∪ setC b c = Set.univ) →
  b = -1 ∧ c = -6 :=
by
  sorry

end find_bc_l1475_147531


namespace cross_covers_two_rectangles_l1475_147545

def Chessboard := Fin 8 × Fin 8

def is_cross (center : Chessboard) (point : Chessboard) : Prop :=
  (point.1 = center.1 ∧ (point.2 = center.2 - 1 ∨ point.2 = center.2 + 1)) ∨
  (point.2 = center.2 ∧ (point.1 = center.1 - 1 ∨ point.1 = center.1 + 1)) ∨
  (point = center)

def Rectangle_1x3 (rect : Fin 22) : Chessboard → Prop := sorry -- This represents Alina's rectangles
def Rectangle_1x2 (rect : Fin 22) : Chessboard → Prop := sorry -- This represents Polina's rectangles

theorem cross_covers_two_rectangles :
  ∃ center : Chessboard, 
    (∃ (rect_a : Fin 22), ∃ (rect_b : Fin 22), rect_a ≠ rect_b ∧ 
      (∀ p, is_cross center p → Rectangle_1x3 rect_a p) ∧ 
      (∀ p, is_cross center p → Rectangle_1x2 rect_b p)) ∨
    (∃ (rect_a : Fin 22), ∃ (rect_b : Fin 22), rect_a ≠ rect_b ∧ 
      (∀ p, is_cross center p → Rectangle_1x3 rect_a p) ∧ 
      (∀ p, is_cross center p → Rectangle_1x3 rect_b p)) ∨
    (∃ (rect_a : Fin 22), ∃ (rect_b : Fin 22), rect_a ≠ rect_b ∧ 
      (∀ p, is_cross center p → Rectangle_1x2 rect_a p) ∧ 
      (∀ p, is_cross center p → Rectangle_1x2 rect_b p)) :=
sorry

end cross_covers_two_rectangles_l1475_147545
