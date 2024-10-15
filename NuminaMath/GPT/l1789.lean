import Mathlib

namespace NUMINAMATH_GPT_smallest_number_divisible_by_9_and_remainder_1_mod_2_thru_6_l1789_178903

theorem smallest_number_divisible_by_9_and_remainder_1_mod_2_thru_6 : 
  ∃ n : ℕ, (∃ k : ℕ, n = 60 * k + 1) ∧ n % 9 = 0 ∧ ∀ m : ℕ, (∃ k' : ℕ, m = 60 * k' + 1) ∧ m % 9 = 0 → n ≤ m :=
by
  sorry

end NUMINAMATH_GPT_smallest_number_divisible_by_9_and_remainder_1_mod_2_thru_6_l1789_178903


namespace NUMINAMATH_GPT_regular_polygon_sides_l1789_178946

theorem regular_polygon_sides (exterior_angle : ℝ) (total_exterior_angle_sum : ℝ) (h1 : exterior_angle = 18) (h2 : total_exterior_angle_sum = 360) :
  let n := total_exterior_angle_sum / exterior_angle
  n = 20 :=
by
  sorry

end NUMINAMATH_GPT_regular_polygon_sides_l1789_178946


namespace NUMINAMATH_GPT_disk_diameter_solution_l1789_178906

noncomputable def disk_diameter_condition : Prop :=
∃ x : ℝ, 
  (4 * Real.sqrt 3 + 2 * Real.pi) * x^2 - 12 * x + Real.sqrt 3 = 0 ∧
  x < Real.sqrt 3 / 6 ∧ 
  2 * x = 0.36

theorem disk_diameter_solution : exists (x : ℝ), 
  disk_diameter_condition := 
sorry

end NUMINAMATH_GPT_disk_diameter_solution_l1789_178906


namespace NUMINAMATH_GPT_combined_age_of_sam_and_drew_l1789_178954

theorem combined_age_of_sam_and_drew
  (sam_age : ℕ)
  (drew_age : ℕ)
  (h1 : sam_age = 18)
  (h2 : sam_age = drew_age / 2):
  sam_age + drew_age = 54 := sorry

end NUMINAMATH_GPT_combined_age_of_sam_and_drew_l1789_178954


namespace NUMINAMATH_GPT_son_present_age_l1789_178919

variable (S M : ℕ)

-- Condition 1: M = S + 20
def man_age_relation (S M : ℕ) : Prop := M = S + 20

-- Condition 2: In two years, the man's age will be twice the age of his son
def age_relation_in_two_years (S M : ℕ) : Prop := M + 2 = 2*(S + 2)

theorem son_present_age : 
  ∀ (S M : ℕ), man_age_relation S M → age_relation_in_two_years S M → S = 18 :=
by
  intros S M h1 h2
  sorry

end NUMINAMATH_GPT_son_present_age_l1789_178919


namespace NUMINAMATH_GPT_Shawn_scored_6_points_l1789_178947

theorem Shawn_scored_6_points
  (points_per_basket : ℤ)
  (matthew_points : ℤ)
  (total_baskets : ℤ)
  (h1 : points_per_basket = 3)
  (h2 : matthew_points = 9)
  (h3 : total_baskets = 5)
  : (∃ shawn_points : ℤ, shawn_points = 6) :=
by
  sorry

end NUMINAMATH_GPT_Shawn_scored_6_points_l1789_178947


namespace NUMINAMATH_GPT_count_distinct_rat_k_l1789_178986

theorem count_distinct_rat_k : 
  (∃ N : ℕ, N = 108 ∧ ∀ k : ℚ, abs k < 300 → (∃ x : ℤ, 3 * x^2 + k * x + 20 = 0) →
  (∃! k, abs k < 300 ∧ (∃ x : ℤ, 3 * x^2 + k * x + 20 = 0))) :=
sorry

end NUMINAMATH_GPT_count_distinct_rat_k_l1789_178986


namespace NUMINAMATH_GPT_car_rental_cost_l1789_178922

def daily_rental_rate : ℝ := 29
def per_mile_charge : ℝ := 0.08
def rental_duration : ℕ := 1
def distance_driven : ℝ := 214.0

theorem car_rental_cost : 
  (daily_rental_rate * rental_duration + per_mile_charge * distance_driven) = 46.12 := 
by 
  sorry

end NUMINAMATH_GPT_car_rental_cost_l1789_178922


namespace NUMINAMATH_GPT_rectangle_perimeter_l1789_178909

theorem rectangle_perimeter (a b : ℕ) (h1 : a ≠ b) (h2 : a * b = 4 * (a + b)) : 2 * (a + b) = 36 := by
  sorry

end NUMINAMATH_GPT_rectangle_perimeter_l1789_178909


namespace NUMINAMATH_GPT_preferred_pets_combination_l1789_178965

-- Define the number of puppies, kittens, and hamsters
def num_puppies : ℕ := 20
def num_kittens : ℕ := 10
def num_hamsters : ℕ := 12

-- State the main theorem to prove, that the number of ways Alice, Bob, and Charlie 
-- can buy their preferred pets is 2400
theorem preferred_pets_combination : num_puppies * num_kittens * num_hamsters = 2400 :=
by
  sorry

end NUMINAMATH_GPT_preferred_pets_combination_l1789_178965


namespace NUMINAMATH_GPT_c_finish_work_in_6_days_l1789_178956

theorem c_finish_work_in_6_days (a b c : ℝ) (ha : a = 1/36) (hb : b = 1/18) (habc : a + b + c = 1/4) : c = 1/6 :=
by
  sorry

end NUMINAMATH_GPT_c_finish_work_in_6_days_l1789_178956


namespace NUMINAMATH_GPT_drank_bottles_of_juice_l1789_178929

theorem drank_bottles_of_juice
  (bottles_in_refrigerator : ℕ)
  (bottles_in_pantry : ℕ)
  (bottles_bought : ℕ)
  (bottles_left : ℕ)
  (initial_bottles := bottles_in_refrigerator + bottles_in_pantry)
  (total_bottles := initial_bottles + bottles_bought)
  (bottles_drank := total_bottles - bottles_left) :
  bottles_in_refrigerator = 4 ∧
  bottles_in_pantry = 4 ∧
  bottles_bought = 5 ∧
  bottles_left = 10 →
  bottles_drank = 3 :=
by sorry

end NUMINAMATH_GPT_drank_bottles_of_juice_l1789_178929


namespace NUMINAMATH_GPT_emma_garden_area_l1789_178992

-- Define the given conditions
def EmmaGarden (total_posts : ℕ) (posts_on_shorter_side : ℕ) (posts_on_longer_side : ℕ) (distance_between_posts : ℕ) : Prop :=
  total_posts = 24 ∧
  distance_between_posts = 6 ∧
  (posts_on_longer_side + 1) = 3 * (posts_on_shorter_side + 1) ∧
  2 * (posts_on_shorter_side + 1 + posts_on_longer_side + 1) = 24

-- The theorem to prove
theorem emma_garden_area : ∃ (length width : ℕ), EmmaGarden 24 2 8 6 ∧ (length = 6 * (2) ∧ width = 6 * (8 - 1)) ∧ (length * width = 576) :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_emma_garden_area_l1789_178992


namespace NUMINAMATH_GPT_largest_a1_l1789_178941

theorem largest_a1
  (a : ℕ+ → ℝ)
  (h_pos : ∀ n, 0 < a n)
  (h_eq : ∀ n, (2 * a (n + 1) - a n) * (a (n + 1) * a n - 1) = 0)
  (h_initial : a 1 = a 10) :
  ∃ (max_a1 : ℝ), max_a1 = 16 ∧ ∀ x, x = a 1 → x ≤ 16 :=
by
  sorry

end NUMINAMATH_GPT_largest_a1_l1789_178941


namespace NUMINAMATH_GPT_unit_digit_seven_consecutive_l1789_178974

theorem unit_digit_seven_consecutive (n : ℕ) : 
  (n * (n + 1) * (n + 2) * (n + 3) * (n + 4) * (n + 5) * (n + 6)) % 10 = 0 := 
by
  sorry

end NUMINAMATH_GPT_unit_digit_seven_consecutive_l1789_178974


namespace NUMINAMATH_GPT_factor_polynomial_l1789_178961

theorem factor_polynomial (x : ℤ) :
  36 * x ^ 6 - 189 * x ^ 12 + 81 * x ^ 9 = 9 * x ^ 6 * (4 + 9 * x ^ 3 - 21 * x ^ 6) := 
sorry

end NUMINAMATH_GPT_factor_polynomial_l1789_178961


namespace NUMINAMATH_GPT_vitya_catchup_time_l1789_178958

-- Define the conditions
def left_home_together (vitya_mom_start_same_time: Bool) :=
  vitya_mom_start_same_time = true

def same_speed (vitya_speed mom_speed : ℕ) :=
  vitya_speed = mom_speed

def initial_distance (time : ℕ) (speed : ℕ) :=
  2 * time * speed = 20 * speed

def increased_speed (vitya_speed mom_speed : ℕ) :=
  vitya_speed = 5 * mom_speed

def relative_speed (vitya_speed mom_speed : ℕ) :=
  vitya_speed - mom_speed = 4 * mom_speed

def catchup_time (distance relative_speed : ℕ) :=
  distance / relative_speed = 5

-- The main theorem stating the problem
theorem vitya_catchup_time (vitya_speed mom_speed : ℕ) (t : ℕ) (realization_time : ℕ) :
  left_home_together true →
  same_speed vitya_speed mom_speed →
  initial_distance realization_time mom_speed →
  increased_speed (5 * mom_speed) mom_speed →
  relative_speed (5 * mom_speed) mom_speed →
  catchup_time (20 * mom_speed) (4 * mom_speed) :=
by
  intros
  sorry

end NUMINAMATH_GPT_vitya_catchup_time_l1789_178958


namespace NUMINAMATH_GPT_chord_length_l1789_178917

theorem chord_length (a b : ℝ) (M : ℝ) (h : M * M = a * b) : ∃ AB : ℝ, AB = 2 * Real.sqrt (a * b) :=
by
  sorry

end NUMINAMATH_GPT_chord_length_l1789_178917


namespace NUMINAMATH_GPT_find_number_l1789_178951

def incorrect_multiplication (x : ℕ) : ℕ := 394 * x
def correct_multiplication (x : ℕ) : ℕ := 493 * x
def difference (x : ℕ) : ℕ := correct_multiplication x - incorrect_multiplication x
def expected_difference : ℕ := 78426

theorem find_number (x : ℕ) (h : difference x = expected_difference) : x = 792 := by
  sorry

end NUMINAMATH_GPT_find_number_l1789_178951


namespace NUMINAMATH_GPT_min_ab_sum_l1789_178915

theorem min_ab_sum (a b : ℤ) (h : a * b = 72) : a + b >= -17 :=
by
  sorry

end NUMINAMATH_GPT_min_ab_sum_l1789_178915


namespace NUMINAMATH_GPT_union_M_N_is_real_l1789_178908

def M : Set ℝ := {x | x^2 + x > 0}
def N : Set ℝ := {x | |x| > 2}

theorem union_M_N_is_real : M ∪ N = Set.univ := by
  sorry

end NUMINAMATH_GPT_union_M_N_is_real_l1789_178908


namespace NUMINAMATH_GPT_min_distance_squared_l1789_178901

theorem min_distance_squared (a b c d : ℝ) (e : ℝ) (h₀ : e = Real.exp 1) 
  (h₁ : (a - 2 * Real.exp a) / b = 1) (h₂ : (2 - c) / d = 1) :
  (a - c)^2 + (b - d)^2 = 8 := by
  sorry

end NUMINAMATH_GPT_min_distance_squared_l1789_178901


namespace NUMINAMATH_GPT_initial_dozens_of_doughnuts_l1789_178905

theorem initial_dozens_of_doughnuts (doughnuts_eaten doughnuts_left : ℕ)
  (h_eaten : doughnuts_eaten = 8)
  (h_left : doughnuts_left = 16) :
  (doughnuts_eaten + doughnuts_left) / 12 = 2 := by
  sorry

end NUMINAMATH_GPT_initial_dozens_of_doughnuts_l1789_178905


namespace NUMINAMATH_GPT_option_b_correct_l1789_178997

theorem option_b_correct (a b : ℝ) (h : a ≠ b) : (1 / (a - b) + 1 / (b - a) = 0) :=
by
  sorry

end NUMINAMATH_GPT_option_b_correct_l1789_178997


namespace NUMINAMATH_GPT_insufficient_data_l1789_178912

variable (M P O : ℝ)

theorem insufficient_data
  (h1 : M < P)
  (h2 : O > M) :
  ¬(P < O) ∧ ¬(O < P) ∧ ¬(P = O) := 
sorry

end NUMINAMATH_GPT_insufficient_data_l1789_178912


namespace NUMINAMATH_GPT_boat_distance_against_stream_l1789_178979

-- Define the speed of the boat in still water
def speed_boat_still : ℝ := 8

-- Define the distance covered by the boat along the stream in one hour
def distance_along_stream : ℝ := 11

-- Define the time duration for the journey
def time_duration : ℝ := 1

-- Define the speed of the stream
def speed_stream : ℝ := distance_along_stream - speed_boat_still

-- Define the speed of the boat against the stream
def speed_against_stream : ℝ := speed_boat_still - speed_stream

-- Define the distance covered by the boat against the stream in one hour
def distance_against_stream (t : ℝ) : ℝ := speed_against_stream * t

-- The main theorem: The boat travels 5 km against the stream in one hour
theorem boat_distance_against_stream : distance_against_stream time_duration = 5 := by
  sorry

end NUMINAMATH_GPT_boat_distance_against_stream_l1789_178979


namespace NUMINAMATH_GPT_polynomial_three_positive_roots_inequality_polynomial_three_positive_roots_equality_condition_l1789_178967

theorem polynomial_three_positive_roots_inequality
  (a b c : ℝ)
  (x1 x2 x3 : ℝ)
  (h1 : x1 > 0) (h2 : x2 > 0) (h3 : x3 > 0)
  (h_poly : ∀ x, x^3 + a * x^2 + b * x + c = 0) :
  2 * a^3 + 9 * c ≤ 7 * a * b :=
sorry

theorem polynomial_three_positive_roots_equality_condition
  (a b c : ℝ)
  (x1 x2 x3 : ℝ)
  (h1 : x1 > 0) (h2 : x2 > 0) (h3 : x3 > 0)
  (h_poly : ∀ x, x^3 + a * x^2 + b * x + c = 0) :
  (2 * a^3 + 9 * c = 7 * a * b) ↔ (x1 = x2 ∧ x2 = x3) :=
sorry

end NUMINAMATH_GPT_polynomial_three_positive_roots_inequality_polynomial_three_positive_roots_equality_condition_l1789_178967


namespace NUMINAMATH_GPT_solve_for_x_l1789_178980

theorem solve_for_x (x : ℝ) (h : (1/3) + (1/x) = 2/3) : x = 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1789_178980


namespace NUMINAMATH_GPT_fiona_initial_seat_l1789_178983

theorem fiona_initial_seat (greg hannah ian jane kayla lou : Fin 7)
  (greg_final : Fin 7 := greg + 3)
  (hannah_final : Fin 7 := hannah - 2)
  (ian_final : Fin 7 := jane)
  (jane_final : Fin 7 := ian)
  (kayla_final : Fin 7 := kayla + 1)
  (lou_final : Fin 7 := lou - 2)
  (fiona_final : Fin 7) :
  (fiona_final = 0 ∨ fiona_final = 6) →
  ∀ (fiona_initial : Fin 7), 
  (greg_final ≠ fiona_initial ∧ hannah_final ≠ fiona_initial ∧ ian_final ≠ fiona_initial ∧ 
   jane_final ≠ fiona_initial ∧ kayla_final ≠ fiona_initial ∧ lou_final ≠ fiona_initial) →
  fiona_initial = 0 :=
by
  sorry

end NUMINAMATH_GPT_fiona_initial_seat_l1789_178983


namespace NUMINAMATH_GPT_rays_form_straight_lines_l1789_178926

theorem rays_form_straight_lines
  (α β : ℝ)
  (h1 : 2 * α + 2 * β = 360) :
  α + β = 180 :=
by
  -- proof details are skipped using sorry
  sorry

end NUMINAMATH_GPT_rays_form_straight_lines_l1789_178926


namespace NUMINAMATH_GPT_line_not_in_first_quadrant_l1789_178934

theorem line_not_in_first_quadrant (m x : ℝ) (h : mx + 3 = 4) (hx : x = 1) : 
  ∀ x y : ℝ, y = (m - 2) * x - 3 → ¬(0 < x ∧ 0 < y) :=
by
  -- The actual proof would go here
  sorry

end NUMINAMATH_GPT_line_not_in_first_quadrant_l1789_178934


namespace NUMINAMATH_GPT_proof_problem_l1789_178970

variables (p q : Prop)

theorem proof_problem (h₁ : p) (h₂ : ¬ q) : ¬ p ∨ ¬ q :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l1789_178970


namespace NUMINAMATH_GPT_alice_wins_with_optimal_strategy_l1789_178990

theorem alice_wins_with_optimal_strategy :
  (∀ (N : ℕ) (X Y : ℕ), N = 270000 → N = X * Y → gcd X Y ≠ 1 → 
    (∃ (alice : ℕ → ℕ → Prop), ∀ N, ∃ (X Y : ℕ), alice N (X * Y) → gcd X Y ≠ 1) ∧
    (∀ (bob : ℕ → ℕ → ℕ → Prop), ∀ N X Y, bob N X Y → gcd X Y ≠ 1)) →
  (N : ℕ) → N = 270000 → gcd N 1 ≠ 1 :=
by
  sorry

end NUMINAMATH_GPT_alice_wins_with_optimal_strategy_l1789_178990


namespace NUMINAMATH_GPT_simultaneous_equations_solution_exists_l1789_178953

theorem simultaneous_equations_solution_exists (m : ℝ) : 
  (∃ (x y : ℝ), y = m * x + 6 ∧ y = (2 * m - 3) * x + 9) ↔ m ≠ 3 :=
by
  sorry

end NUMINAMATH_GPT_simultaneous_equations_solution_exists_l1789_178953


namespace NUMINAMATH_GPT_total_distance_biked_l1789_178977

theorem total_distance_biked :
  let monday_distance := 12
  let tuesday_distance := 2 * monday_distance - 3
  let wednesday_distance := 2 * 11
  let thursday_distance := wednesday_distance + 2
  let friday_distance := thursday_distance + 2
  let saturday_distance := friday_distance + 2
  let sunday_distance := 3 * 6
  monday_distance + tuesday_distance + wednesday_distance + thursday_distance + friday_distance + saturday_distance + sunday_distance = 151 := 
by
  sorry

end NUMINAMATH_GPT_total_distance_biked_l1789_178977


namespace NUMINAMATH_GPT_floor_equation_solution_l1789_178982

theorem floor_equation_solution (x : ℝ) :
  (⌊⌊3 * x⌋ + 1/3⌋ = ⌊x + 5⌋) ↔ (7/3 ≤ x ∧ x < 3) := 
sorry

end NUMINAMATH_GPT_floor_equation_solution_l1789_178982


namespace NUMINAMATH_GPT_complete_square_solution_l1789_178914

theorem complete_square_solution (x: ℝ) : (x^2 + 8 * x - 3 = 0) -> ((x + 4)^2 = 19) := 
by
  sorry

end NUMINAMATH_GPT_complete_square_solution_l1789_178914


namespace NUMINAMATH_GPT_line_equation_minimized_area_l1789_178996

theorem line_equation_minimized_area :
  ∀ (l_1 l_2 l_3 : ℝ × ℝ → Prop) (l : ℝ × ℝ → Prop),
    (∀ x y : ℝ, l_1 (x, y) ↔ 3 * x + 2 * y - 1 = 0) ∧
    (∀ x y : ℝ, l_2 (x, y) ↔ 5 * x + 2 * y + 1 = 0) ∧
    (∀ x y : ℝ, l_3 (x, y) ↔ 3 * x - 5 * y + 6 = 0) →
    (∃ c : ℝ, ∀ x y : ℝ, l (x, y) ↔ 3 * x - 5 * y + c = 0) →
    (∃ x y : ℝ, l_1 (x, y) ∧ l_2 (x, y) ∧ l (x, y)) →
    (∀ a : ℝ, ∀ x y : ℝ, l (x, y) ↔ x + y = a) →
    (∃ k : ℝ, k > 0 ∧ ∀ x y : ℝ, l (x, y) ↔ 2 * x - y + 4 = 0) → 
    sorry :=
sorry

end NUMINAMATH_GPT_line_equation_minimized_area_l1789_178996


namespace NUMINAMATH_GPT_gcd_of_polynomial_and_multiple_l1789_178944

-- Definitions based on given conditions
def multiple_of (a b : ℕ) : Prop := ∃ k, a = k * b

-- The main statement of the problem
theorem gcd_of_polynomial_and_multiple (y : ℕ) (h : multiple_of y 56790) :
  Nat.gcd ((3 * y + 2) * (5 * y + 3) * (11 * y + 7) * (y + 17)) y = 714 :=
sorry

end NUMINAMATH_GPT_gcd_of_polynomial_and_multiple_l1789_178944


namespace NUMINAMATH_GPT_forty_percent_of_n_l1789_178904

theorem forty_percent_of_n (N : ℝ) (h : (1/4) * (1/3) * (2/5) * N = 16) : 0.40 * N = 384 :=
by
  sorry

end NUMINAMATH_GPT_forty_percent_of_n_l1789_178904


namespace NUMINAMATH_GPT_stratified_sampling_young_employees_l1789_178935

variable (total_employees elderly_employees middle_aged_employees young_employees sample_size : ℕ)

-- Conditions
axiom total_employees_eq : total_employees = 750
axiom elderly_employees_eq : elderly_employees = 150
axiom middle_aged_employees_eq : middle_aged_employees = 250
axiom young_employees_eq : young_employees = 350
axiom sample_size_eq : sample_size = 15

-- The proof problem
theorem stratified_sampling_young_employees :
  young_employees / total_employees * sample_size = 7 :=
by
  sorry

end NUMINAMATH_GPT_stratified_sampling_young_employees_l1789_178935


namespace NUMINAMATH_GPT_string_cheese_packages_l1789_178960

theorem string_cheese_packages (days_per_week : ℕ) (weeks : ℕ) (oldest_daily : ℕ) (youngest_daily : ℕ) (pack_size : ℕ) 
    (H1 : days_per_week = 5)
    (H2 : weeks = 4)
    (H3 : oldest_daily = 2)
    (H4 : youngest_daily = 1)
    (H5 : pack_size = 30) 
  : (oldest_daily * days_per_week + youngest_daily * days_per_week) * weeks / pack_size = 2 :=
  sorry

end NUMINAMATH_GPT_string_cheese_packages_l1789_178960


namespace NUMINAMATH_GPT_B_work_days_l1789_178930

theorem B_work_days (a b : ℝ) (h1 : a + b = 1/4) (h2 : a = 1/14) : 1 / b = 5.6 :=
by
  sorry

end NUMINAMATH_GPT_B_work_days_l1789_178930


namespace NUMINAMATH_GPT_length_of_plot_correct_l1789_178918

noncomputable def length_of_plot (b : ℕ) : ℕ := b + 30

theorem length_of_plot_correct (b : ℕ) (cost_per_meter total_cost : ℝ) 
    (h1 : length_of_plot b = b + 30)
    (h2 : cost_per_meter = 26.50)
    (h3 : total_cost = 5300)
    (h4 : 2 * (b + (b + 30)) * cost_per_meter = total_cost) :
    length_of_plot 35 = 65 :=
by
  sorry

end NUMINAMATH_GPT_length_of_plot_correct_l1789_178918


namespace NUMINAMATH_GPT_nancy_initial_bottle_caps_l1789_178902

theorem nancy_initial_bottle_caps (found additional_bottle_caps: ℕ) (total_bottle_caps: ℕ) (h1: additional_bottle_caps = 88) (h2: total_bottle_caps = 179) : 
  (total_bottle_caps - additional_bottle_caps) = 91 :=
by
  sorry

end NUMINAMATH_GPT_nancy_initial_bottle_caps_l1789_178902


namespace NUMINAMATH_GPT_smallest_factor_of_32_not_8_l1789_178991

theorem smallest_factor_of_32_not_8 : ∃ n : ℕ, n = 16 ∧ (n ∣ 32) ∧ ¬(n ∣ 8) ∧ ∀ m : ℕ, (m ∣ 32) ∧ ¬(m ∣ 8) → n ≤ m :=
by
  sorry

end NUMINAMATH_GPT_smallest_factor_of_32_not_8_l1789_178991


namespace NUMINAMATH_GPT_corrected_mean_l1789_178952

theorem corrected_mean (n : ℕ) (mean incorrect_observation correct_observation : ℝ) (h_n : n = 50) (h_mean : mean = 32) (h_incorrect : incorrect_observation = 23) (h_correct : correct_observation = 48) : 
  (mean * n + (correct_observation - incorrect_observation)) / n = 32.5 := 
by 
  sorry

end NUMINAMATH_GPT_corrected_mean_l1789_178952


namespace NUMINAMATH_GPT_problem1_problem2_l1789_178931

-- Statement for problem 1
theorem problem1 : 
  (-2020 - 2 / 3) + (2019 + 3 / 4) + (-2018 - 5 / 6) + (2017 + 1 / 2) = -2 - 1 / 4 := 
sorry

-- Statement for problem 2
theorem problem2 : 
  (-1 - 1 / 2) + (-2000 - 5 / 6) + (4000 + 3 / 4) + (-1999 - 2 / 3) = -5 / 4 := 
sorry

end NUMINAMATH_GPT_problem1_problem2_l1789_178931


namespace NUMINAMATH_GPT_polynomial_pattern_1_polynomial_pattern_2_polynomial_calculation_polynomial_factorization_l1789_178907

theorem polynomial_pattern_1 (a b : ℝ) : (a + b) * (a ^ 2 - a * b + b ^ 2) = a ^ 3 + b ^ 3 :=
sorry

theorem polynomial_pattern_2 (a b : ℝ) : (a - b) * (a ^ 2 + a * b + b ^ 2) = a ^ 3 - b ^ 3 :=
sorry

theorem polynomial_calculation (a b : ℝ) : (a + 2 * b) * (a ^ 2 - 2 * a * b + 4 * b ^ 2) = a ^ 3 + 8 * b ^ 3 :=
sorry

theorem polynomial_factorization (a : ℝ) : a ^ 3 - 8 = (a - 2) * (a ^ 2 + 2 * a + 4) :=
sorry

end NUMINAMATH_GPT_polynomial_pattern_1_polynomial_pattern_2_polynomial_calculation_polynomial_factorization_l1789_178907


namespace NUMINAMATH_GPT_doughnuts_per_person_l1789_178989

theorem doughnuts_per_person :
  let samuel_doughnuts := 24
  let cathy_doughnuts := 36
  let total_doughnuts := samuel_doughnuts + cathy_doughnuts
  let total_people := 10
  total_doughnuts / total_people = 6 := 
by
  -- Definitions and conditions from the problem
  let samuel_doughnuts := 24
  let cathy_doughnuts := 36
  let total_doughnuts := samuel_doughnuts + cathy_doughnuts
  let total_people := 10
  -- Goal to prove
  show total_doughnuts / total_people = 6
  sorry

end NUMINAMATH_GPT_doughnuts_per_person_l1789_178989


namespace NUMINAMATH_GPT_exceeds_threshold_at_8_l1789_178950

def geometric_sum (a r n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1)

def exceeds_threshold (n : ℕ) : Prop :=
  geometric_sum 2 2 n ≥ 500

theorem exceeds_threshold_at_8 :
  ∀ n < 8, ¬exceeds_threshold n ∧ exceeds_threshold 8 :=
by
  sorry

end NUMINAMATH_GPT_exceeds_threshold_at_8_l1789_178950


namespace NUMINAMATH_GPT_problem_statement_l1789_178942

open Real

theorem problem_statement :
  (10 * 1.8 - 2 * 1.5) / 0.3 + 3^(2/3) - Real.log 4 = 50.6938 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1789_178942


namespace NUMINAMATH_GPT_vector_sum_l1789_178995

-- Define the vectors a and b
def a : ℝ × ℝ × ℝ := (1, 2, 3)
def b : ℝ × ℝ × ℝ := (-1, 0, 1)

-- Define the target vector c
def c : ℝ × ℝ × ℝ := (-1, 2, 5)

-- State the theorem to be proven
theorem vector_sum : a + (2:ℝ) • b = c :=
by 
  -- Not providing the proof, just adding a sorry
  sorry

end NUMINAMATH_GPT_vector_sum_l1789_178995


namespace NUMINAMATH_GPT_non_officers_count_l1789_178993

theorem non_officers_count (avg_salary_all : ℕ) (avg_salary_officers : ℕ) (avg_salary_non_officers : ℕ) (num_officers : ℕ) 
  (N : ℕ) 
  (h_avg_salary_all : avg_salary_all = 120) 
  (h_avg_salary_officers : avg_salary_officers = 430) 
  (h_avg_salary_non_officers : avg_salary_non_officers = 110) 
  (h_num_officers : num_officers = 15) 
  (h_eq : avg_salary_all * (num_officers + N) = avg_salary_officers * num_officers + avg_salary_non_officers * N) 
  : N = 465 :=
by
  -- Proof would be here
  sorry

end NUMINAMATH_GPT_non_officers_count_l1789_178993


namespace NUMINAMATH_GPT_find_number_of_cats_l1789_178924

theorem find_number_of_cats (dogs ferrets cats total_shoes shoes_per_animal : ℕ) 
  (h_dogs : dogs = 3)
  (h_ferrets : ferrets = 1)
  (h_total_shoes : total_shoes = 24)
  (h_shoes_per_animal : shoes_per_animal = 4) :
  cats = (total_shoes - (dogs + ferrets) * shoes_per_animal) / shoes_per_animal := by
  sorry

end NUMINAMATH_GPT_find_number_of_cats_l1789_178924


namespace NUMINAMATH_GPT_product_of_two_numbers_l1789_178973

theorem product_of_two_numbers (a b : ℝ) (h1 : a + b = 60) (h2 : a - b = 10) : a * b = 875 :=
sorry

end NUMINAMATH_GPT_product_of_two_numbers_l1789_178973


namespace NUMINAMATH_GPT_tens_digit_of_3_pow_2010_l1789_178969

theorem tens_digit_of_3_pow_2010 : (3^2010 / 10) % 10 = 4 := by
  sorry

end NUMINAMATH_GPT_tens_digit_of_3_pow_2010_l1789_178969


namespace NUMINAMATH_GPT_machineA_finishing_time_l1789_178923

theorem machineA_finishing_time
  (A : ℝ)
  (hA : 0 < A)
  (hB : 0 < 12)
  (hC : 0 < 6)
  (h_total_time : 0 < 2)
  (h_work_done_per_hour : (1 / A) + (1 / 12) + (1 / 6) = 1 / 2) :
  A = 4 := sorry

end NUMINAMATH_GPT_machineA_finishing_time_l1789_178923


namespace NUMINAMATH_GPT_largest_digit_divisible_by_6_l1789_178978

def is_even (n : ℕ) : Prop := n % 2 = 0

def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

theorem largest_digit_divisible_by_6 : ∃ (N : ℕ), 0 ≤ N ∧ N ≤ 9 ∧ is_even N ∧ is_divisible_by_3 (26 + N) ∧ 
  (∀ (N' : ℕ), 0 ≤ N' ∧ N' ≤ 9 ∧ is_even N' ∧ is_divisible_by_3 (26 + N') → N' ≤ N) :=
sorry

end NUMINAMATH_GPT_largest_digit_divisible_by_6_l1789_178978


namespace NUMINAMATH_GPT_number_of_cars_l1789_178984

theorem number_of_cars (x : ℕ) (h : 3 * (x - 2) = 2 * x + 9) : x = 15 :=
by {
  sorry
}

end NUMINAMATH_GPT_number_of_cars_l1789_178984


namespace NUMINAMATH_GPT_sum_infinite_geometric_l1789_178998

theorem sum_infinite_geometric (a r : ℝ) (ha : a = 2) (hr : r = 1/3) : 
  ∑' n : ℕ, a * r^n = 3 := by
  sorry

end NUMINAMATH_GPT_sum_infinite_geometric_l1789_178998


namespace NUMINAMATH_GPT_face_value_of_stock_l1789_178999

-- Define variables and constants
def quoted_price : ℝ := 200
def yield_quoted : ℝ := 0.10
def percentage_yield : ℝ := 0.20

-- Define the annual income from the quoted price and percentage yield
def annual_income_from_quoted_price : ℝ := yield_quoted * quoted_price
def annual_income_from_face_value (FV : ℝ) : ℝ := percentage_yield * FV

-- Problem statement to prove
theorem face_value_of_stock (FV : ℝ) :
  annual_income_from_face_value FV = annual_income_from_quoted_price →
  FV = 100 := 
by
  sorry

end NUMINAMATH_GPT_face_value_of_stock_l1789_178999


namespace NUMINAMATH_GPT_vertex_coordinates_quadratic_through_point_a_less_than_neg_2_fifth_l1789_178968

noncomputable def quadratic_function (a : ℝ) (x : ℝ) : ℝ := (x + 1) * (a * x + 2 * a + 2)

theorem vertex_coordinates (a : ℝ) (H : a = 1) : 
    (∃ v_x v_y : ℝ, quadratic_function a v_x = v_y ∧ v_x = -5 / 2 ∧ v_y = -9 / 4) := 
by {
    sorry
}

theorem quadratic_through_point : 
    (∃ a : ℝ, (quadratic_function a 0 = -2) ∧ (∀ x, quadratic_function a x = -2 * (x + 1)^2)) := 
by {
    sorry
}

theorem a_less_than_neg_2_fifth 
  (x1 x2 y1 y2 a : ℝ) (H1 : x1 + x2 = 2) (H2 : x1 < x2) (H3 : y1 > y2) 
  (Hfunc : ∀ x, quadratic_function (a * x + 2 * a + 2) (x + 1) = quadratic_function x y) :
    a < -2 / 5 := 
by {
    sorry
}

end NUMINAMATH_GPT_vertex_coordinates_quadratic_through_point_a_less_than_neg_2_fifth_l1789_178968


namespace NUMINAMATH_GPT_integer_to_sixth_power_l1789_178988

theorem integer_to_sixth_power (a b : ℕ) (h : 3^a * 3^b = 3^(a + b)) (ha : a = 12) (hb : b = 18) : 
  ∃ x : ℕ, x = 243 ∧ x^6 = 3^(a + b) :=
by
  sorry

end NUMINAMATH_GPT_integer_to_sixth_power_l1789_178988


namespace NUMINAMATH_GPT_find_height_of_pyramid_l1789_178940

noncomputable def volume (B h : ℝ) : ℝ := (1/3) * B * h
noncomputable def area_of_isosceles_right_triangle (leg : ℝ) : ℝ := (1/2) * leg * leg

theorem find_height_of_pyramid (leg : ℝ) (h : ℝ) (V : ℝ) (B : ℝ)
  (Hleg : leg = 3)
  (Hvol : V = 6)
  (Hbase : B = area_of_isosceles_right_triangle leg)
  (Hvol_eq : V = volume B h) :
  h = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_height_of_pyramid_l1789_178940


namespace NUMINAMATH_GPT_has_three_real_zeros_l1789_178939

noncomputable def f (x m : ℝ) : ℝ := 2 * x^3 - 6 * x + m

theorem has_three_real_zeros (m : ℝ) : 
    (∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ f x₁ m = 0 ∧ f x₂ m = 0 ∧ f x₃ m = 0) ↔ (-4 < m ∧ m < 4) :=
sorry

end NUMINAMATH_GPT_has_three_real_zeros_l1789_178939


namespace NUMINAMATH_GPT_pow_1986_mod_7_l1789_178921

theorem pow_1986_mod_7 : (5 ^ 1986) % 7 = 1 := by
  sorry

end NUMINAMATH_GPT_pow_1986_mod_7_l1789_178921


namespace NUMINAMATH_GPT_jongkook_points_l1789_178936

-- Define the conditions in the problem
def num_questions_solved_each : ℕ := 18
def shinhye_points : ℕ := 100
def jongkook_correct_6_points : ℕ := 8
def jongkook_correct_5_points : ℕ := 6
def points_per_question_6 : ℕ := 6
def points_per_question_5 : ℕ := 5
def jongkook_wrong_questions : ℕ := num_questions_solved_each - jongkook_correct_6_points - jongkook_correct_5_points

-- Calculate Jongkook's points from correct answers
def jongkook_points_from_6 : ℕ := jongkook_correct_6_points * points_per_question_6
def jongkook_points_from_5 : ℕ := jongkook_correct_5_points * points_per_question_5

-- Calculate total points
def jongkook_total_points : ℕ := jongkook_points_from_6 + jongkook_points_from_5

-- Prove that Jongkook's total points is 78
theorem jongkook_points : jongkook_total_points = 78 :=
by
  sorry

end NUMINAMATH_GPT_jongkook_points_l1789_178936


namespace NUMINAMATH_GPT_hyperbola_focal_length_l1789_178949

theorem hyperbola_focal_length (m : ℝ) : 
  (∀ x y : ℝ, (x^2 / m - y^2 / 4 = 1)) ∧ (∀ f : ℝ, f = 6) → m = 5 := 
  by 
    -- Using the condition that the focal length is 6
    sorry

end NUMINAMATH_GPT_hyperbola_focal_length_l1789_178949


namespace NUMINAMATH_GPT_simplify_and_evaluate_l1789_178972

theorem simplify_and_evaluate (a : ℝ) (h : a^2 + 2 * a - 1 = 0) :
  ((a - 2) / (a^2 + 2 * a) - (a - 1) / (a^2 + 4 * a + 4)) / ((a - 4) / (a + 2)) = 1 / 3 :=
by sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l1789_178972


namespace NUMINAMATH_GPT_parallel_lines_a_value_l1789_178927

theorem parallel_lines_a_value 
    (a : ℝ) 
    (l₁ : ∀ x y : ℝ, 2 * x + y - 1 = 0) 
    (l₂ : ∀ x y : ℝ, (a - 1) * x + 3 * y - 2 = 0) 
    (h_parallel : ∀ x y : ℝ, 2 / (a - 1) = 1 / 3) : 
    a = 7 := 
    sorry

end NUMINAMATH_GPT_parallel_lines_a_value_l1789_178927


namespace NUMINAMATH_GPT_test_score_based_on_preparation_l1789_178920

theorem test_score_based_on_preparation :
  (grade_varies_directly_with_effective_hours : Prop) →
  (effective_hour_constant : ℝ) →
  (actual_hours_first_test : ℕ) →
  (actual_hours_second_test : ℕ) →
  (score_first_test : ℕ) →
  effective_hour_constant = 0.8 →
  actual_hours_first_test = 5 →
  score_first_test = 80 →
  actual_hours_second_test = 6 →
  grade_varies_directly_with_effective_hours →
  ∃ score_second_test : ℕ, score_second_test = 96 := by
  sorry

end NUMINAMATH_GPT_test_score_based_on_preparation_l1789_178920


namespace NUMINAMATH_GPT_fill_cistern_time_l1789_178959

theorem fill_cistern_time (fill_ratio : ℚ) (time_for_fill_ratio : ℚ) :
  fill_ratio = 1/11 ∧ time_for_fill_ratio = 4 → (11 * time_for_fill_ratio) = 44 :=
by
  sorry

end NUMINAMATH_GPT_fill_cistern_time_l1789_178959


namespace NUMINAMATH_GPT_curve_is_line_l1789_178943

theorem curve_is_line (θ : ℝ) (hθ : θ = 5 * Real.pi / 6) : 
  ∃ a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ ∀ (r : ℝ), r = 0 ↔
  (∃ p : ℝ × ℝ, p.1 = r * Real.cos θ ∧ p.2 = r * Real.sin θ ∧
                p.1 * a + p.2 * b = 0) :=
sorry

end NUMINAMATH_GPT_curve_is_line_l1789_178943


namespace NUMINAMATH_GPT_no_integer_solutions_for_eq_l1789_178928

theorem no_integer_solutions_for_eq {x y : ℤ} : ¬ (∃ x y : ℤ, (x + 7) * (x + 6) = 8 * y + 3) := by
  sorry

end NUMINAMATH_GPT_no_integer_solutions_for_eq_l1789_178928


namespace NUMINAMATH_GPT_binomial_expansion_coefficient_l1789_178948

theorem binomial_expansion_coefficient (a : ℝ)
  (h : ∃ r, 9 - 3 * r = 6 ∧ (-a)^r * (Nat.choose 9 r) = 36) :
  a = -4 :=
  sorry

end NUMINAMATH_GPT_binomial_expansion_coefficient_l1789_178948


namespace NUMINAMATH_GPT_find_number_with_divisors_condition_l1789_178957

theorem find_number_with_divisors_condition :
  ∃ n : ℕ, (∃ d1 d2 d3 d4 : ℕ, 1 ≤ d1 ∧ d1 < d2 ∧ d2 < d3 ∧ d3 < d4 ∧ d4 * d4 ∣ n ∧
    d1 * d1 + d2 * d2 + d3 * d3 + d4 * d4 = n) ∧ n = 130 :=
by
  sorry

end NUMINAMATH_GPT_find_number_with_divisors_condition_l1789_178957


namespace NUMINAMATH_GPT_range_of_a_l1789_178975

open Real

theorem range_of_a (a : ℝ) :
  ((a = 0 ∨ (a > 0 ∧ a^2 - 4 * a < 0)) ∨ (a^2 - 2 * a - 3 < 0)) ∧
  ¬((a = 0 ∨ (a > 0 ∧ a^2 - 4 * a < 0)) ∧ (a^2 - 2 * a - 3 < 0)) ↔
  (-1 < a ∧ a < 0) ∨ (3 ≤ a ∧ a < 4) := 
sorry

end NUMINAMATH_GPT_range_of_a_l1789_178975


namespace NUMINAMATH_GPT_required_tents_l1789_178963

def numberOfPeopleInMattFamily : ℕ := 1 + 2
def numberOfPeopleInBrotherFamily : ℕ := 1 + 1 + 4
def numberOfPeopleInUncleJoeFamily : ℕ := 1 + 1 + 3
def totalNumberOfPeople : ℕ := numberOfPeopleInMattFamily + numberOfPeopleInBrotherFamily + numberOfPeopleInUncleJoeFamily
def numberOfPeopleSleepingInHouse : ℕ := 4
def numberOfPeopleSleepingInTents : ℕ := totalNumberOfPeople - numberOfPeopleSleepingInHouse
def peoplePerTent : ℕ := 2

def numberOfTentsNeeded : ℕ :=
  numberOfPeopleSleepingInTents / peoplePerTent

theorem required_tents : numberOfTentsNeeded = 5 := by
  sorry

end NUMINAMATH_GPT_required_tents_l1789_178963


namespace NUMINAMATH_GPT_length_width_ratio_l1789_178962

theorem length_width_ratio 
  (W : ℕ) (P : ℕ) (L : ℕ)
  (hW : W = 90) 
  (hP : P = 432) 
  (hP_eq : P = 2 * L + 2 * W) : 
  (L / W = 7 / 5) := 
  sorry

end NUMINAMATH_GPT_length_width_ratio_l1789_178962


namespace NUMINAMATH_GPT_Cade_remaining_marbles_l1789_178985

def initial_marbles := 87
def given_marbles := 8
def remaining_marbles := initial_marbles - given_marbles

theorem Cade_remaining_marbles : remaining_marbles = 79 := by
  sorry

end NUMINAMATH_GPT_Cade_remaining_marbles_l1789_178985


namespace NUMINAMATH_GPT_sum_of_powers_l1789_178910

theorem sum_of_powers (x : ℝ) (h1 : x^10 - 3*x + 2 = 0) (h2 : x ≠ 1) : 
  x^9 + x^8 + x^7 + x^6 + x^5 + x^4 + x^3 + x^2 + x + 1 = 3 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_powers_l1789_178910


namespace NUMINAMATH_GPT_angle_half_in_first_quadrant_l1789_178994

theorem angle_half_in_first_quadrant (α : ℝ) (hα : 90 < α ∧ α < 180) : 0 < α / 2 ∧ α / 2 < 90 := 
sorry

end NUMINAMATH_GPT_angle_half_in_first_quadrant_l1789_178994


namespace NUMINAMATH_GPT_find_ax5_by5_l1789_178987

variables (a b x y: ℝ)

theorem find_ax5_by5 (h1 : a * x + b * y = 5)
                      (h2 : a * x^2 + b * y^2 = 11)
                      (h3 : a * x^3 + b * y^3 = 24)
                      (h4 : a * x^4 + b * y^4 = 56) :
                      a * x^5 + b * y^5 = 180.36 :=
sorry

end NUMINAMATH_GPT_find_ax5_by5_l1789_178987


namespace NUMINAMATH_GPT_shem_earnings_l1789_178913

theorem shem_earnings (kem_hourly: ℝ) (ratio: ℝ) (workday_hours: ℝ) (shem_hourly: ℝ) (shem_daily: ℝ) :
  kem_hourly = 4 →
  ratio = 2.5 →
  shem_hourly = kem_hourly * ratio →
  workday_hours = 8 →
  shem_daily = shem_hourly * workday_hours →
  shem_daily = 80 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_shem_earnings_l1789_178913


namespace NUMINAMATH_GPT_solution_l1789_178933

theorem solution (x : ℝ) 
  (h1 : 1/x < 3)
  (h2 : 1/x > -4) 
  (h3 : x^2 - 3*x + 2 < 0) : 
  1 < x ∧ x < 2 :=
sorry

end NUMINAMATH_GPT_solution_l1789_178933


namespace NUMINAMATH_GPT_xiaohong_total_score_l1789_178932

theorem xiaohong_total_score :
  ∀ (midterm_score final_score : ℕ) (midterm_weight final_weight : ℝ),
    midterm_score = 80 →
    final_score = 90 →
    midterm_weight = 0.4 →
    final_weight = 0.6 →
    (midterm_score * midterm_weight + final_score * final_weight) = 86 :=
by
  intros midterm_score final_score midterm_weight final_weight
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_xiaohong_total_score_l1789_178932


namespace NUMINAMATH_GPT_second_sample_correct_l1789_178911

def total_samples : ℕ := 7341
def first_sample : ℕ := 4221
def second_sample : ℕ := total_samples - first_sample

theorem second_sample_correct : second_sample = 3120 :=
by
  sorry

end NUMINAMATH_GPT_second_sample_correct_l1789_178911


namespace NUMINAMATH_GPT_model_y_completion_time_l1789_178971

theorem model_y_completion_time
  (rate_model_x : ℕ → ℝ)
  (rate_model_y : ℕ → ℝ)
  (num_model_x : ℕ)
  (num_model_y : ℕ)
  (time_model_x : ℝ)
  (combined_rate : ℝ)
  (same_number : num_model_y = num_model_x)
  (task_completion_x : ∀ x, rate_model_x x = 1 / time_model_x)
  (total_model_x : num_model_x = 24)
  (task_completion_y : ∀ y, rate_model_y y = 1 / y)
  (one_minute_completion : num_model_x * rate_model_x 1 + num_model_y * rate_model_y 36 = combined_rate)
  : 36 = time_model_x * 2 :=
by
  sorry

end NUMINAMATH_GPT_model_y_completion_time_l1789_178971


namespace NUMINAMATH_GPT_sum_of_fractions_l1789_178981

theorem sum_of_fractions :
  (3 / 12 : Real) + (6 / 120) + (9 / 1200) = 0.3075 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_fractions_l1789_178981


namespace NUMINAMATH_GPT_boys_from_other_communities_l1789_178966

theorem boys_from_other_communities (total_boys : ℕ) (percent_muslims percent_hindus percent_sikhs : ℕ) 
    (h_total_boys : total_boys = 300)
    (h_percent_muslims : percent_muslims = 44)
    (h_percent_hindus : percent_hindus = 28)
    (h_percent_sikhs : percent_sikhs = 10) :
  ∃ (percent_others : ℕ), percent_others = 100 - (percent_muslims + percent_hindus + percent_sikhs) ∧ 
                             (percent_others * total_boys / 100) = 54 := 
by 
  sorry

end NUMINAMATH_GPT_boys_from_other_communities_l1789_178966


namespace NUMINAMATH_GPT_sandy_total_sums_attempted_l1789_178916

theorem sandy_total_sums_attempted (C I : ℕ) 
  (marks_per_correct_sum : ℕ := 3) 
  (marks_lost_per_incorrect_sum : ℕ := 2) 
  (total_marks : ℕ := 45) 
  (correct_sums : ℕ := 21) 
  (H : 3 * correct_sums - 2 * I = total_marks) 
  : C + I = 30 := 
by 
  sorry

end NUMINAMATH_GPT_sandy_total_sums_attempted_l1789_178916


namespace NUMINAMATH_GPT_max_value_q_l1789_178938

namespace proof

theorem max_value_q (A M C : ℕ) (h : A + M + C = 15) :
  A * M * C + A * M + M * C + C * A ≤ 200 :=
sorry

end proof

end NUMINAMATH_GPT_max_value_q_l1789_178938


namespace NUMINAMATH_GPT_minimum_value_l1789_178964

theorem minimum_value (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x - 2 * y + 3 = 0) : 
  ∃ z : ℝ, z = 3 ∧ (∀ z' : ℝ, (z' = y^2 / x) → z ≤ z') :=
sorry

end NUMINAMATH_GPT_minimum_value_l1789_178964


namespace NUMINAMATH_GPT_exists_four_digit_number_sum_digits_14_divisible_by_14_l1789_178955

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 1000) % 10 + (n / 100 % 10) % 10 + (n / 10 % 10) % 10 + (n % 10)

theorem exists_four_digit_number_sum_digits_14_divisible_by_14 :
  ∃ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ sum_of_digits n = 14 ∧ n % 14 = 0 :=
sorry

end NUMINAMATH_GPT_exists_four_digit_number_sum_digits_14_divisible_by_14_l1789_178955


namespace NUMINAMATH_GPT_monthly_salary_l1789_178937

variable {S : ℝ}

-- Conditions based on the problem description
def spends_on_food (S : ℝ) : ℝ := 0.40 * S
def spends_on_house_rent (S : ℝ) : ℝ := 0.20 * S
def spends_on_entertainment (S : ℝ) : ℝ := 0.10 * S
def spends_on_conveyance (S : ℝ) : ℝ := 0.10 * S
def savings (S : ℝ) : ℝ := 0.20 * S

-- Given savings
def savings_amount : ℝ := 2500

-- The proof statement for the monthly salary
theorem monthly_salary (h : savings S = savings_amount) : S = 12500 := by
  sorry

end NUMINAMATH_GPT_monthly_salary_l1789_178937


namespace NUMINAMATH_GPT_infinite_solutions_d_eq_5_l1789_178900

theorem infinite_solutions_d_eq_5 :
  ∃ (d : ℝ), d = 5 ∧ ∀ (y : ℝ), 3 * (5 + d * y) = 15 * y + 15 :=
by
  sorry

end NUMINAMATH_GPT_infinite_solutions_d_eq_5_l1789_178900


namespace NUMINAMATH_GPT_total_amount_is_correct_l1789_178945

variable (w x y z R : ℝ)
variable (hx : x = 0.345 * w)
variable (hy : y = 0.45625 * w)
variable (hz : z = 0.61875 * w)
variable (hy_value : y = 112.50)

theorem total_amount_is_correct :
  R = w + x + y + z → R = 596.8150684931507 := by
  sorry

end NUMINAMATH_GPT_total_amount_is_correct_l1789_178945


namespace NUMINAMATH_GPT_shortest_distance_from_parabola_to_line_l1789_178925

open Real

noncomputable def parabola_point (M : ℝ × ℝ) : Prop :=
  M.snd^2 = 6 * M.fst

noncomputable def distance_to_line (M : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  abs (a * M.fst + b * M.snd + c) / sqrt (a^2 + b^2)

theorem shortest_distance_from_parabola_to_line (M : ℝ × ℝ) (h : parabola_point M) :
  distance_to_line M 3 (-4) 12 = 3 :=
by
  sorry

end NUMINAMATH_GPT_shortest_distance_from_parabola_to_line_l1789_178925


namespace NUMINAMATH_GPT_thabo_books_l1789_178976

theorem thabo_books :
  ∃ (H P F : ℕ), 
    P = H + 20 ∧ 
    F = 2 * P ∧ 
    H + P + F = 200 ∧ 
    H = 35 :=
by
  sorry

end NUMINAMATH_GPT_thabo_books_l1789_178976
