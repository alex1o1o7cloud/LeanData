import Mathlib

namespace isosceles_triangle_count_l80_8083

theorem isosceles_triangle_count : 
  ∃ (count : ℕ), count = 6 ∧ 
  ∀ (a b c : ℕ), a + b + c = 25 → 
  (a = b ∨ a = c ∨ b = c) → 
  a ≠ b ∨ c ≠ b ∨ a ≠ c → 
  ∃ (x y z : ℕ), x = a ∧ y = b ∧ z = c := 
sorry

end isosceles_triangle_count_l80_8083


namespace probability_X_eq_Y_correct_l80_8011

noncomputable def probability_X_eq_Y : ℝ :=
  let lower_bound := -20 * Real.pi
  let upper_bound := 20 * Real.pi
  let total_pairs := (upper_bound - lower_bound) * (upper_bound - lower_bound)
  let matching_pairs := 81
  matching_pairs / total_pairs

theorem probability_X_eq_Y_correct :
  probability_X_eq_Y = 81 / 1681 :=
by
  unfold probability_X_eq_Y
  sorry

end probability_X_eq_Y_correct_l80_8011


namespace charge_move_increases_energy_l80_8075

noncomputable def energy_increase_when_charge_moved : ℝ :=
  let initial_energy := 15
  let energy_per_pair := initial_energy / 3
  let new_energy_AB := energy_per_pair
  let new_energy_AC := 2 * energy_per_pair
  let new_energy_BC := 2 * energy_per_pair
  let final_energy := new_energy_AB + new_energy_AC + new_energy_BC
  final_energy - initial_energy

theorem charge_move_increases_energy :
  energy_increase_when_charge_moved = 10 :=
by
  sorry

end charge_move_increases_energy_l80_8075


namespace set_intersection_example_l80_8003

theorem set_intersection_example (A : Set ℕ) (B : Set ℕ) (hA : A = {1, 3, 5}) (hB : B = {3, 4}) :
  A ∩ B = {3} :=
by
  sorry

end set_intersection_example_l80_8003


namespace apple_difference_l80_8014

def carla_apples : ℕ := 7
def tim_apples : ℕ := 1

theorem apple_difference : carla_apples - tim_apples = 6 := by
  sorry

end apple_difference_l80_8014


namespace instantaneous_velocity_at_3_l80_8086

-- Definitions based on the conditions.
def displacement (t : ℝ) := 2 * t ^ 3

-- The statement to prove.
theorem instantaneous_velocity_at_3 : (deriv displacement 3) = 54 := by
  sorry

end instantaneous_velocity_at_3_l80_8086


namespace fraction_value_l80_8046

theorem fraction_value (a b : ℝ) (h : 1 / a - 1 / b = 4) : 
    (a - 2 * a * b - b) / (2 * a + 7 * a * b - 2 * b) = 6 :=
by
  sorry

end fraction_value_l80_8046


namespace coordinates_of_P_l80_8029

theorem coordinates_of_P (m : ℝ) (P : ℝ × ℝ) :
  P = (2 * m, m + 8) ∧ 2 * m = 0 → P = (0, 8) := by
  intros hm
  sorry

end coordinates_of_P_l80_8029


namespace desired_markup_percentage_l80_8006

theorem desired_markup_percentage
  (initial_price : ℝ) (markup_rate : ℝ) (wholesale_price : ℝ) (additional_increase : ℝ) 
  (h1 : initial_price = wholesale_price * (1 + markup_rate)) 
  (h2 : initial_price = 34) 
  (h3 : markup_rate = 0.70) 
  (h4 : additional_increase = 6) 
  : ( (initial_price + additional_increase - wholesale_price) / wholesale_price * 100 ) = 100 := 
by
  sorry

end desired_markup_percentage_l80_8006


namespace black_pens_per_student_l80_8062

theorem black_pens_per_student (number_of_students : ℕ)
                               (red_pens_per_student : ℕ)
                               (taken_first_month : ℕ)
                               (taken_second_month : ℕ)
                               (pens_after_splitting : ℕ)
                               (initial_black_pens_per_student : ℕ) : 
  number_of_students = 3 → 
  red_pens_per_student = 62 → 
  taken_first_month = 37 → 
  taken_second_month = 41 → 
  pens_after_splitting = 79 → 
  initial_black_pens_per_student = 43 :=
by sorry

end black_pens_per_student_l80_8062


namespace length_of_each_piece_after_subdividing_l80_8007

theorem length_of_each_piece_after_subdividing (total_length : ℝ) (num_initial_cuts : ℝ) (num_pieces_given : ℝ) (num_subdivisions : ℝ) (final_length : ℝ) : 
  total_length = 200 → 
  num_initial_cuts = 4 → 
  num_pieces_given = 2 → 
  num_subdivisions = 2 → 
  final_length = (total_length / num_initial_cuts / num_subdivisions) → 
  final_length = 25 := 
by 
  intros h1 h2 h3 h4 h5 
  sorry

end length_of_each_piece_after_subdividing_l80_8007


namespace modulo_residue_l80_8091

theorem modulo_residue:
  (247 + 5 * 40 + 7 * 143 + 4 * (2^3 - 1)) % 13 = 7 :=
by
  sorry

end modulo_residue_l80_8091


namespace at_least_one_angle_not_less_than_sixty_l80_8077

theorem at_least_one_angle_not_less_than_sixty (A B C : ℝ)
  (hABC_sum : A + B + C = 180)
  (hA : A < 60)
  (hB : B < 60)
  (hC : C < 60) : false :=
by
  sorry

end at_least_one_angle_not_less_than_sixty_l80_8077


namespace frances_towels_weight_in_ounces_l80_8095

theorem frances_towels_weight_in_ounces (Mary_towels Frances_towels : ℕ) (Mary_weight Frances_weight : ℝ) (total_weight : ℝ) :
  Mary_towels = 24 ∧ Mary_towels = 4 * Frances_towels ∧ total_weight = Mary_weight + Frances_weight →
  Frances_weight * 16 = 240 :=
by
  sorry

end frances_towels_weight_in_ounces_l80_8095


namespace intersection_eq_l80_8039

-- Define the sets M and N using the given conditions
def M : Set ℝ := { x | x < 1 / 2 }
def N : Set ℝ := { x | x ≥ -4 }

-- The goal is to prove that the intersection of M and N is { x | -4 ≤ x < 1 / 2 }
theorem intersection_eq : M ∩ N = { x | -4 ≤ x ∧ x < (1 / 2) } :=
by
  sorry

end intersection_eq_l80_8039


namespace perpendicular_planes_l80_8058

variables (b c : Line) (α β : Plane)
axiom line_in_plane (b : Line) (α : Plane) : Prop -- b ⊆ α
axiom line_parallel_plane (c : Line) (α : Plane) : Prop -- c ∥ α
axiom lines_are_skew (b c : Line) : Prop -- b and c could be skew
axiom planes_are_perpendicular (α β : Plane) : Prop -- α ⊥ β
axiom line_perpendicular_plane (c : Line) (β : Plane) : Prop -- c ⊥ β

theorem perpendicular_planes (hcα : line_in_plane c α) (hcβ : line_perpendicular_plane c β) : planes_are_perpendicular α β := 
sorry

end perpendicular_planes_l80_8058


namespace quadratic_inequality_solution_range_l80_8027

theorem quadratic_inequality_solution_range (a : ℝ) :
  (∃ x : ℝ, x^2 + (a-1)*x + 1 < 0) → (a > 3 ∨ a < -1) :=
by
  sorry

end quadratic_inequality_solution_range_l80_8027


namespace find_m_if_parallel_l80_8012

theorem find_m_if_parallel 
  (m : ℚ) 
  (a : ℚ × ℚ := (-2, 3)) 
  (b : ℚ × ℚ := (1, m - 3/2)) 
  (h : ∃ k : ℚ, (a.1 = k * b.1) ∧ (a.2 = k * b.2)) : 
  m = 0 := 
  sorry

end find_m_if_parallel_l80_8012


namespace andrew_purchase_grapes_l80_8041

theorem andrew_purchase_grapes (G : ℕ) (h : 70 * G + 495 = 1055) : G = 8 :=
by
  sorry

end andrew_purchase_grapes_l80_8041


namespace smallest_abs_sum_of_products_l80_8040

noncomputable def g (x : ℝ) : ℝ := x^4 + 16 * x^3 + 69 * x^2 + 112 * x + 64

theorem smallest_abs_sum_of_products :
  (∀ w1 w2 w3 w4 : ℝ, g w1 = 0 ∧ g w2 = 0 ∧ g w3 = 0 ∧ g w4 = 0 → 
   |w1 * w2 + w3 * w4| ≥ 8) ∧ 
  (∃ w1 w2 w3 w4 : ℝ, g w1 = 0 ∧ g w2 = 0 ∧ g w3 = 0 ∧ g w4 = 0 ∧ 
   |w1 * w2 + w3 * w4| = 8) :=
sorry

end smallest_abs_sum_of_products_l80_8040


namespace prime_ge_7_p2_sub1_div_by_30_l80_8099

theorem prime_ge_7_p2_sub1_div_by_30 (p : ℕ) (hp : Nat.Prime p) (h7 : p ≥ 7) : 30 ∣ (p^2 - 1) :=
sorry

end prime_ge_7_p2_sub1_div_by_30_l80_8099


namespace factorize_expr_l80_8098

noncomputable def example_expr (x : ℝ) : ℝ := 2 * x^2 - 4 * x

theorem factorize_expr (x : ℝ) : example_expr x = 2 * x * (x - 2) := by
  sorry

end factorize_expr_l80_8098


namespace inverse_prop_l80_8034

theorem inverse_prop (a b : ℝ) : (a > b) → (|a| > |b|) :=
sorry

end inverse_prop_l80_8034


namespace area_difference_l80_8002

theorem area_difference (r1 r2 : ℝ) (h1 : r1 = 30) (h2 : r2 = 15 / 2) :
  π * r1^2 - π * r2^2 = 843.75 * π :=
by
  rw [h1, h2]
  sorry

end area_difference_l80_8002


namespace magnitude_z_l80_8054

open Complex

theorem magnitude_z
  (z w : ℂ)
  (h1 : abs (2 * z - w) = 25)
  (h2 : abs (z + 2 * w) = 5)
  (h3 : abs (z + w) = 2) : abs z = 9 := 
by 
  sorry

end magnitude_z_l80_8054


namespace average_rate_first_half_80_l80_8047

theorem average_rate_first_half_80
    (total_distance : ℝ)
    (average_rate_trip : ℝ)
    (distance_first_half : ℝ)
    (time_first_half : ℝ)
    (time_second_half : ℝ)
    (time_total : ℝ)
    (R : ℝ)
    (H1 : total_distance = 640)
    (H2 : average_rate_trip = 40)
    (H3 : distance_first_half = total_distance / 2)
    (H4 : time_first_half = distance_first_half / R)
    (H5 : time_second_half = 3 * time_first_half)
    (H6 : time_total = time_first_half + time_second_half)
    (H7 : average_rate_trip = total_distance / time_total) :
    R = 80 := 
by 
  -- Given conditions
  sorry

end average_rate_first_half_80_l80_8047


namespace necessary_but_not_sufficient_l80_8064

theorem necessary_but_not_sufficient
  (x y : ℝ) :
  (x^2 + y^2 ≤ 2*x → x^2 + y^2 ≤ 4) ∧ ¬ (x^2 + y^2 ≤ 4 → x^2 + y^2 ≤ 2*x) :=
by {
  sorry
}

end necessary_but_not_sufficient_l80_8064


namespace heating_time_l80_8089

def T_initial: ℝ := 20
def T_final: ℝ := 100
def rate: ℝ := 5

theorem heating_time : (T_final - T_initial) / rate = 16 := by
  sorry

end heating_time_l80_8089


namespace blue_lipstick_count_l80_8057

def total_students : Nat := 200

def colored_lipstick_students (total : Nat) : Nat :=
  total / 2

def red_lipstick_students (colored : Nat) : Nat :=
  colored / 4

def blue_lipstick_students (red : Nat) : Nat :=
  red / 5

theorem blue_lipstick_count :
  blue_lipstick_students (red_lipstick_students (colored_lipstick_students total_students)) = 5 := 
sorry

end blue_lipstick_count_l80_8057


namespace ceil_sqrt_180_eq_14_l80_8036

theorem ceil_sqrt_180_eq_14
  (h : 13 < Real.sqrt 180 ∧ Real.sqrt 180 < 14) :
  Int.ceil (Real.sqrt 180) = 14 :=
  sorry

end ceil_sqrt_180_eq_14_l80_8036


namespace stratified_sampling_l80_8087

theorem stratified_sampling
  (ratio_first : ℕ)
  (ratio_second : ℕ)
  (ratio_third : ℕ)
  (sample_size : ℕ)
  (h_ratio : ratio_first = 3 ∧ ratio_second = 4 ∧ ratio_third = 3)
  (h_sample_size : sample_size = 50) :
  (ratio_second * sample_size) / (ratio_first + ratio_second + ratio_third) = 20 :=
by
  sorry

end stratified_sampling_l80_8087


namespace part1_part2_part3_l80_8022

-- Define the necessary constants and functions as per conditions
variable (a : ℝ) (f : ℝ → ℝ)
variable (hpos : a > 0) (hfa : f a = 1)

-- Conditions based on the problem statement
variable (hodd : ∀ x, f (-x) = -f x)
variable (hfe : ∀ x1 x2, f (x1 - x2) = (f x1 * f x2 + 1) / (f x2 - f x1))

-- 1. Prove that f(2a) = 0
theorem part1  : f (2 * a) = 0 := sorry

-- 2. Prove that there exists a constant T > 0 such that f(x + T) = f(x)
theorem part2 : ∃ T > 0, ∀ x, f (x + 4 * a) = f x := sorry

-- 3. Prove f(x) is decreasing on (0, 4a) given x ∈ (0, 2a) implies f(x) > 0
theorem part3 (hx_correct : ∀ x, 0 < x ∧ x < 2 * a → 0 < f x) :
  ∀ x1 x2, 0 < x2 ∧ x2  < x1 ∧ x1 < 4 * a → f x2 > f x1 := sorry

end part1_part2_part3_l80_8022


namespace amount_received_by_sam_l80_8079

def P : ℝ := 15000
def r : ℝ := 0.10
def n : ℝ := 2
def t : ℝ := 1

noncomputable def compoundInterest (P r n t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem amount_received_by_sam : compoundInterest P r n t = 16537.50 := by
  sorry

end amount_received_by_sam_l80_8079


namespace probability_of_2_reds_before_3_greens_l80_8005

theorem probability_of_2_reds_before_3_greens :
  let total_chips := 7
  let red_chips := 4
  let green_chips := 3
  let total_arrangements := Nat.choose total_chips green_chips
  let favorable_arrangements := Nat.choose 5 2
  (favorable_arrangements / total_arrangements : ℚ) = (2 / 7 : ℚ) :=
by
  let total_chips := 7
  let red_chips := 4
  let green_chips := 3
  let total_arrangements := Nat.choose total_chips green_chips
  let favorable_arrangements := Nat.choose 5 2
  have fraction_computation :
    (favorable_arrangements : ℚ) / (total_arrangements : ℚ) = (2 / 7 : ℚ)
  {
    sorry
  }
  exact fraction_computation

end probability_of_2_reds_before_3_greens_l80_8005


namespace sum_of_integers_l80_8088

theorem sum_of_integers (a b c : ℤ) (h1 : a = (1 / 3) * (b + c)) (h2 : b = (1 / 5) * (a + c)) (h3 : c = 35) : a + b + c = 60 :=
by
  sorry

end sum_of_integers_l80_8088


namespace Rebecca_worked_56_l80_8052

-- Define the conditions
variables (x : ℕ)
def Toby_hours := 2 * x - 10
def Rebecca_hours := Toby_hours - 8
def Total_hours := x + Toby_hours + Rebecca_hours

-- Theorem stating that under the given conditions, Rebecca worked 56 hours
theorem Rebecca_worked_56 
  (h : Total_hours = 157) 
  (hx : x = 37) : Rebecca_hours = 56 :=
by sorry

end Rebecca_worked_56_l80_8052


namespace valid_grid_count_l80_8076

def is_adjacent (i j : ℕ) (n : ℕ) : Prop :=
  (i = j + 1 ∨ i + 1 = j ∨ (i = n - 1 ∧ j = 0) ∨ (i = 0 ∧ j = n - 1))

def valid_grid (grid : ℕ → ℕ → ℕ) : Prop :=
  ∀ i j, 0 ≤ i ∧ i < 4 ∧ 0 ≤ j ∧ j < 4 →
         (is_adjacent i (i+1) 4 → grid i (i+1) * grid i (i+1) = 0) ∧ 
         (is_adjacent j (j+1) 4 → grid (j+1) j * grid (j+1) j = 0)

theorem valid_grid_count : 
  ∃ s : ℕ, s = 1234 ∧
    (∃ grid : ℕ → ℕ → ℕ, valid_grid grid) :=
sorry

end valid_grid_count_l80_8076


namespace pete_should_leave_by_0730_l80_8018

def walking_time : ℕ := 10
def train_time : ℕ := 80
def latest_arrival_time : String := "0900"
def departure_time : String := "0730"

theorem pete_should_leave_by_0730 :
  (latest_arrival_time = "0900" → walking_time = 10 ∧ train_time = 80 → departure_time = "0730") := by
  sorry

end pete_should_leave_by_0730_l80_8018


namespace trisha_hourly_wage_l80_8073

theorem trisha_hourly_wage (annual_take_home_pay : ℝ) (percent_withheld : ℝ)
  (hours_per_week : ℝ) (weeks_per_year : ℝ) (hourly_wage : ℝ) :
  annual_take_home_pay = 24960 ∧ 
  percent_withheld = 0.20 ∧ 
  hours_per_week = 40 ∧ 
  weeks_per_year = 52 ∧ 
  hourly_wage = (annual_take_home_pay / (0.80 * (hours_per_week * weeks_per_year))) → 
  hourly_wage = 15 :=
by sorry

end trisha_hourly_wage_l80_8073


namespace initial_average_weight_l80_8063

theorem initial_average_weight 
    (W : ℝ)
    (a b c d e : ℝ)
    (h1 : (a + b + c) / 3 = W)
    (h2 : (a + b + c + d) / 4 = W)
    (h3 : (b + c + d + (d + 3)) / 4 = 68)
    (h4 : a = 81) :
    W = 70 := 
sorry

end initial_average_weight_l80_8063


namespace equal_striped_areas_l80_8030

theorem equal_striped_areas (A B C D : ℝ) (h_AD_DB : D = A + B) (h_CD2 : C^2 = A * B) :
  (π * C^2 / 4 = π * B^2 / 8 - π * A^2 / 8 - π * D^2 / 8) := 
sorry

end equal_striped_areas_l80_8030


namespace eval_x2_sub_y2_l80_8017

theorem eval_x2_sub_y2 (x y : ℝ) (h1 : x + y = 10) (h2 : 2 * x + y = 13) : x^2 - y^2 = -40 := by
  sorry

end eval_x2_sub_y2_l80_8017


namespace min_value_eq_216_l80_8008

noncomputable def min_value (a b c : ℝ) : ℝ :=
  (a^2 + 4*a + 1) * (b^2 + 4*b + 1) * (c^2 + 4*c + 1) / (a * b * c)

theorem min_value_eq_216 {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  min_value a b c = 216 :=
sorry

end min_value_eq_216_l80_8008


namespace value_of_n_l80_8085

theorem value_of_n : ∃ (n : ℕ), 6 * 8 * 3 * n = Nat.factorial 8 ∧ n = 280 :=
by
  use 280
  sorry

end value_of_n_l80_8085


namespace largest_initial_number_l80_8082

theorem largest_initial_number (a₁ a₂ a₃ a₄ a₅ : ℕ) (n : ℕ) (h1 : ¬ ∀ (k : ℕ), k ∣ n → k = 1) 
    (h2 : ¬ ∀ (k : ℕ), k ∣ (n + a₁) → k = 1) 
    (h3 : ¬ ∀ (k : ℕ), k ∣ (n + a₁ + a₂) → k = 1) 
    (h4 : ¬ ∀ (k : ℕ), k ∣ (n + a₁ + a₂ + a₃) → k = 1) 
    (h5 : ¬ ∀ (k : ℕ), k ∣ (n + a₁ + a₂ + a₃ + a₄) → k = 1)
    (h_sum : n + a₁ + a₂ + a₃ + a₄ + a₅ = 100) :
  n = 89 := 
sorry

end largest_initial_number_l80_8082


namespace max_garden_area_l80_8016

-- Definitions of conditions
def shorter_side (s : ℕ) := s
def longer_side (s : ℕ) := 2 * s
def total_perimeter (s : ℕ) := 2 * shorter_side s + 2 * longer_side s 
def garden_area (s : ℕ) := shorter_side s * longer_side s

-- Theorem with given conditions and conclusion to be proven
theorem max_garden_area (s : ℕ) (h_perimeter : total_perimeter s = 480) : garden_area s = 12800 :=
by
  sorry

end max_garden_area_l80_8016


namespace correct_simplification_l80_8037

theorem correct_simplification (m a b x y : ℝ) :
  ¬ (4 * m - m = 3) ∧
  ¬ (a^2 * b - a * b^2 = 0) ∧
  ¬ (2 * a^3 - 3 * a^3 = a^3) ∧
  (x * y - 2 * x * y = - x * y) :=
by {
  sorry
}

end correct_simplification_l80_8037


namespace find_other_number_l80_8065

theorem find_other_number (n : ℕ) (h_lcm : Nat.lcm 12 n = 60) (h_hcf : Nat.gcd 12 n = 3) : n = 15 := by
  sorry

end find_other_number_l80_8065


namespace prob_B_win_correct_l80_8078

-- Define the probabilities for player A winning and a draw
def prob_A_win : ℝ := 0.3
def prob_draw : ℝ := 0.4

-- Define the total probability of all outcomes
def total_prob : ℝ := 1

-- Define the probability of player B winning
def prob_B_win : ℝ := total_prob - prob_A_win - prob_draw

-- Proof problem: Prove that the probability of player B winning is 0.3
theorem prob_B_win_correct : prob_B_win = 0.3 :=
by
  -- The proof would go here, but we use sorry to skip it for now.
  sorry

end prob_B_win_correct_l80_8078


namespace problem1_l80_8068

theorem problem1 (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 15) : 
  (x * y = 5) ∧ ((x - y)^2 = 5) :=
by
  sorry

end problem1_l80_8068


namespace vertex_of_parabola_minimum_value_for_x_ge_2_l80_8059

theorem vertex_of_parabola :
  ∀ x y : ℝ, y = x^2 + 2*x - 3 → ∃ (vx vy : ℝ), (vx = -1) ∧ (vy = -4) :=
by
  sorry

theorem minimum_value_for_x_ge_2 :
  ∀ x : ℝ, x ≥ 2 → y = x^2 + 2*x - 3 → ∃ (min_val : ℝ), min_val = 5 :=
by
  sorry

end vertex_of_parabola_minimum_value_for_x_ge_2_l80_8059


namespace max_children_l80_8028

/-- Total quantities -/
def total_apples : ℕ := 55
def total_cookies : ℕ := 114
def total_chocolates : ℕ := 83

/-- Leftover quantities after distribution -/
def leftover_apples : ℕ := 3
def leftover_cookies : ℕ := 10
def leftover_chocolates : ℕ := 5

/-- Distributed quantities -/
def distributed_apples : ℕ := total_apples - leftover_apples
def distributed_cookies : ℕ := total_cookies - leftover_cookies
def distributed_chocolates : ℕ := total_chocolates - leftover_chocolates

/-- The theorem states the maximum number of children -/
theorem max_children : Nat.gcd (Nat.gcd distributed_apples distributed_cookies) distributed_chocolates = 26 :=
by
  sorry

end max_children_l80_8028


namespace overall_average_mark_l80_8092

theorem overall_average_mark :
  let n1 := 70
  let mean1 := 50
  let n2 := 35
  let mean2 := 60
  let n3 := 45
  let mean3 := 55
  let n4 := 42
  let mean4 := 45
  (n1 * mean1 + n2 * mean2 + n3 * mean3 + n4 * mean4 : ℝ) / (n1 + n2 + n3 + n4) = 51.89 := 
by {
  sorry
}

end overall_average_mark_l80_8092


namespace arithmetic_sequence_a1_a5_product_l80_8020

theorem arithmetic_sequence_a1_a5_product 
  (a : ℕ → ℚ) 
  (h_arithmetic : ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d)
  (h_a3 : a 3 = 3) 
  (h_cond : (1 / a 1) + (1 / a 5) = 6 / 5) : 
  a 1 * a 5 = 5 := 
by
  sorry

end arithmetic_sequence_a1_a5_product_l80_8020


namespace compute_f3_l80_8096

def f (n : ℕ) : ℕ :=
  if n < 5 then n^2 + 4*n + 3 else 2*n + 1

theorem compute_f3 : f (f (f 3)) = 99 :=
by
  sorry

end compute_f3_l80_8096


namespace sequence_property_l80_8010

def sequence_conditions (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧
  a 2 = 3 ∧
  ∀ n ≥ 3, S n + S (n - 2) = 2 * S (n - 1) + n

theorem sequence_property (a : ℕ → ℕ) (S : ℕ → ℕ) (h : sequence_conditions a S) : 
  ∀ n ≥ 3, a n = a (n - 1) + n :=
  sorry

end sequence_property_l80_8010


namespace combined_annual_income_l80_8053

-- Define the given conditions and verify the combined annual income
def A_ratio : ℤ := 5
def B_ratio : ℤ := 2
def C_ratio : ℤ := 3
def D_ratio : ℤ := 4

def C_income : ℤ := 15000
def B_income : ℤ := 16800
def A_income : ℤ := 25000
def D_income : ℤ := 21250

theorem combined_annual_income :
  (A_income + B_income + C_income + D_income) * 12 = 936600 :=
by
  sorry

end combined_annual_income_l80_8053


namespace BoatsRUs_total_canoes_l80_8042

def totalCanoesBuiltByJuly (a : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1)

theorem BoatsRUs_total_canoes :
  totalCanoesBuiltByJuly 5 3 7 = 5465 :=
by
  sorry

end BoatsRUs_total_canoes_l80_8042


namespace rhombus_area_and_perimeter_l80_8021

theorem rhombus_area_and_perimeter (d1 d2 : ℝ) (h1 : d1 = 18) (h2 : d2 = 26) :
  let area := (d1 * d2) / 2
  let s := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
  let perimeter := 4 * s
  area = 234 ∧ perimeter = 20 * Real.sqrt 10 := by
  sorry

end rhombus_area_and_perimeter_l80_8021


namespace number_of_ways_to_choose_bases_l80_8043

-- Definitions of the conditions
def num_students : Nat := 4
def num_bases : Nat := 3

-- The main statement that we need to prove
theorem number_of_ways_to_choose_bases : (num_bases ^ num_students) = 81 := by
  sorry

end number_of_ways_to_choose_bases_l80_8043


namespace age_of_older_friend_l80_8025

theorem age_of_older_friend (a b : ℕ) (h1 : a - b = 2) (h2 : a + b = 74) : a = 38 :=
by
  sorry

end age_of_older_friend_l80_8025


namespace train_speed_without_stoppages_l80_8093

theorem train_speed_without_stoppages 
  (distance_with_stoppages : ℝ)
  (avg_speed_with_stoppages : ℝ)
  (stoppage_time_per_hour : ℝ)
  (distance_without_stoppages : ℝ)
  (avg_speed_without_stoppages : ℝ) :
  avg_speed_with_stoppages = 200 → 
  stoppage_time_per_hour = 20 / 60 →
  distance_without_stoppages = distance_with_stoppages * avg_speed_without_stoppages →
  distance_with_stoppages = avg_speed_with_stoppages →
  avg_speed_without_stoppages == 300 := 
by
  intros
  sorry

end train_speed_without_stoppages_l80_8093


namespace sum_odd_numbers_to_2019_is_correct_l80_8070

-- Define the sequence sum
def sum_first_n_odd (n : ℕ) : ℕ := n * n

-- Define the specific problem
theorem sum_odd_numbers_to_2019_is_correct : sum_first_n_odd 1010 = 1020100 :=
by
  -- Sorry placeholder for the proof
  sorry

end sum_odd_numbers_to_2019_is_correct_l80_8070


namespace find_a_and_b_nth_equation_conjecture_l80_8071

theorem find_a_and_b {a b : ℤ} (h1 : 1^2 + 2^2 - 3^2 = 1 * a - b)
                                        (h2 : 2^2 + 3^2 - 4^2 = 2 * 0 - b)
                                        (h3 : 3^2 + 4^2 - 5^2 = 3 * 1 - b)
                                        (h4 : 4^2 + 5^2 - 6^2 = 4 * 2 - b):
    a = -1 ∧ b = 3 :=
    sorry

theorem nth_equation_conjecture (n : ℤ) :
  n^2 + (n+1)^2 - (n+2)^2 = n * (n-2) - 3 :=
  sorry

end find_a_and_b_nth_equation_conjecture_l80_8071


namespace form_of_reasoning_is_wrong_l80_8000

-- Let's define the conditions
def some_rat_nums_are_proper_fractions : Prop :=
  ∃ q : ℚ, (q.num : ℤ) ≠ q.den ∧ (q.den : ℤ) ≠ 1 ∧ q.den ≠ 0

def integers_are_rational_numbers : Prop :=
  ∀ n : ℤ, ∃ q : ℚ, q = n

-- The major premise of the syllogism
def major_premise := some_rat_nums_are_proper_fractions

-- The minor premise of the syllogism
def minor_premise := integers_are_rational_numbers

-- The conclusion of the syllogism
def conclusion := ∀ n : ℤ, ∃ q : ℚ, (q.num : ℤ) ≠ q.den ∧ (q.den : ℤ) ≠ 1 ∧ q.den ≠ 0

-- We need to prove that the form of reasoning is wrong
theorem form_of_reasoning_is_wrong (H1 : major_premise) (H2 : minor_premise) : ¬ conclusion :=
by
  sorry -- proof to be filled in

end form_of_reasoning_is_wrong_l80_8000


namespace least_subtraction_for_divisibility_l80_8045

theorem least_subtraction_for_divisibility (n : ℕ) (h : n = 964807) : ∃ k, k = 7 ∧ (n - k) % 8 = 0 :=
by 
  sorry

end least_subtraction_for_divisibility_l80_8045


namespace find_v_l80_8081

theorem find_v (v : ℝ) (h : (v - v / 3) - ((v - v / 3) / 3) = 4) : v = 9 := 
by 
  sorry

end find_v_l80_8081


namespace new_player_weight_l80_8013

theorem new_player_weight 
  (original_players : ℕ)
  (original_avg_weight : ℝ)
  (new_players : ℕ)
  (new_avg_weight : ℝ)
  (new_total_weight : ℝ) :
  original_players = 20 →
  original_avg_weight = 180 →
  new_players = 21 →
  new_avg_weight = 181.42857142857142 →
  new_total_weight = 3810 →
  (new_total_weight - original_players * original_avg_weight) = 210 :=
by
  intros
  sorry

end new_player_weight_l80_8013


namespace remainder_of_product_l80_8049

theorem remainder_of_product (a b n : ℕ) (ha : a % n = 7) (hb : b % n = 1) :
  ((a * b) % n) = 7 :=
by
  -- Definitions as per the conditions
  let a := 63
  let b := 65
  let n := 8
  /- Now prove the statement -/
  sorry

end remainder_of_product_l80_8049


namespace harry_weekly_earnings_l80_8060

def dogs_walked_MWF := 7
def dogs_walked_Tue := 12
def dogs_walked_Thu := 9
def pay_per_dog := 5

theorem harry_weekly_earnings : 
  dogs_walked_MWF * pay_per_dog * 3 + dogs_walked_Tue * pay_per_dog + dogs_walked_Thu * pay_per_dog = 210 :=
by
  sorry

end harry_weekly_earnings_l80_8060


namespace star_polygon_points_l80_8031

theorem star_polygon_points (n : ℕ) (A B : ℕ → ℝ) 
  (h_angles_congruent_A : ∀ i j, A i = A j)
  (h_angles_congruent_B : ∀ i j, B i = B j)
  (h_angle_relation : ∀ i, A i = B i - 15) :
  n = 24 :=
by
  sorry

end star_polygon_points_l80_8031


namespace olympic_rings_area_l80_8080

theorem olympic_rings_area (d R r: ℝ) 
  (hyp_d : d = 12 * Real.sqrt 2) 
  (hyp_R : R = 11) 
  (hyp_r : r = 9) 
  (overlap_area : ∀ (i j : ℕ), i ≠ j → 592 = 5 * π * (R ^ 2 - r ^ 2) - 8 * 4.54): 
  592.0 = 5 * π * (R ^ 2 - r ^ 2) - 8 * 4.54 := 
by sorry

end olympic_rings_area_l80_8080


namespace complement_intersection_l80_8044

-- Definitions of sets and complements
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 5}
def C_U_A : Set ℕ := {x | x ∈ U ∧ x ∉ A}
def C_U_B : Set ℕ := {x | x ∈ U ∧ x ∉ B}

-- The proof statement
theorem complement_intersection {U A B C_U_A C_U_B : Set ℕ} (h1 : U = {1, 2, 3, 4, 5}) (h2 : A = {1, 2, 3}) (h3 : B = {2, 5}) (h4 : C_U_A = {x | x ∈ U ∧ x ∉ A}) (h5 : C_U_B = {x | x ∈ U ∧ x ∉ B}) : 
  (C_U_A ∩ C_U_B) = {4} :=
by 
  sorry

end complement_intersection_l80_8044


namespace sum_of_possible_radii_l80_8019

theorem sum_of_possible_radii :
  ∃ r1 r2 : ℝ, 
    (∀ r, (r - 5)^2 + r^2 = (r + 2)^2 → r = r1 ∨ r = r2) ∧ 
    r1 + r2 = 14 :=
sorry

end sum_of_possible_radii_l80_8019


namespace second_acid_solution_percentage_l80_8084

-- Definitions of the problem conditions
def P : ℝ := 75
def V₁ : ℝ := 4
def C₁ : ℝ := 0.60
def V₂ : ℝ := 20
def C₂ : ℝ := 0.72

/-
Given that 4 liters of a 60% acid solution are mixed with a certain volume of another acid solution
to get 20 liters of 72% solution, prove that the percentage of the second acid solution must be 75%.
-/
theorem second_acid_solution_percentage
  (x : ℝ) -- volume of the second acid solution
  (P_percent : ℝ := P) -- percentage of the second acid solution
  (h1 : V₁ + x = V₂) -- condition on volume
  (h2 : C₁ * V₁ + (P_percent / 100) * x = C₂ * V₂) -- condition on acid content
  : P_percent = P := 
by
  -- Moving forward with proof the lean proof
  sorry

end second_acid_solution_percentage_l80_8084


namespace most_stable_performance_l80_8009

theorem most_stable_performance 
    (s_A s_B s_C s_D : ℝ)
    (hA : s_A = 1.5)
    (hB : s_B = 2.6)
    (hC : s_C = 1.7)
    (hD : s_D = 2.8)
    (mean_score : ∀ (x : ℝ), x = 88.5) :
    s_A < s_C ∧ s_C < s_B ∧ s_B < s_D := by
  sorry

end most_stable_performance_l80_8009


namespace tammy_average_speed_second_day_l80_8094

theorem tammy_average_speed_second_day :
  ∃ v t : ℝ, 
  t + (t - 2) + (t + 1) = 20 ∧
  v * t + (v + 0.5) * (t - 2) + (v - 0.5) * (t + 1) = 80 ∧
  (v + 0.5) = 4.575 :=
by 
  sorry

end tammy_average_speed_second_day_l80_8094


namespace smallest_five_digit_number_divisible_by_first_five_primes_l80_8050

theorem smallest_five_digit_number_divisible_by_first_five_primes : 
  ∃ n, (n >= 10000) ∧ (n < 100000) ∧ (∀ p ∈ [2, 3, 5, 7, 11], p ∣ n) ∧ n = 11550 :=
by
  sorry

end smallest_five_digit_number_divisible_by_first_five_primes_l80_8050


namespace bases_for_204_base_b_l80_8072

theorem bases_for_204_base_b (b : ℕ) : (∃ n : ℤ, 2 * b^2 + 4 = n^2) ↔ b = 4 ∨ b = 6 ∨ b = 8 ∨ b = 10 :=
by
  sorry

end bases_for_204_base_b_l80_8072


namespace gcd_of_polynomials_l80_8004

/-- Given that a is an odd multiple of 7877, the greatest common divisor of
       7a^2 + 54a + 117 and 3a + 10 is 1. -/
theorem gcd_of_polynomials (a : ℤ) (h1 : a % 2 = 1) (h2 : 7877 ∣ a) :
  Int.gcd (7 * a ^ 2 + 54 * a + 117) (3 * a + 10) = 1 :=
sorry

end gcd_of_polynomials_l80_8004


namespace find_sample_size_l80_8051

-- Define the frequencies
def frequencies (k : ℕ) : List ℕ := [2 * k, 3 * k, 4 * k, 6 * k, 4 * k, k]

-- Define the sum of the first three frequencies
def sum_first_three_frequencies (k : ℕ) : ℕ := 2 * k + 3 * k + 4 * k

-- Define the total number of data points
def total_data_points (k : ℕ) : ℕ := 2 * k + 3 * k + 4 * k + 6 * k + 4 * k + k

-- Define the main theorem
theorem find_sample_size (n k : ℕ) (h1 : sum_first_three_frequencies k = 27)
  (h2 : total_data_points k = n) : n = 60 := by
  sorry

end find_sample_size_l80_8051


namespace third_root_of_cubic_equation_l80_8023

-- Definitions
variable (a b : ℚ) -- We use rational numbers due to the fractions involved
def cubic_equation (x : ℚ) : ℚ := a * x^3 + (a + 3 * b) * x^2 + (2 * b - 4 * a) * x + (10 - a)

-- Conditions
axiom h1 : cubic_equation a b (-1) = 0
axiom h2 : cubic_equation a b 4 = 0

-- The theorem we aim to prove
theorem third_root_of_cubic_equation : ∃ (c : ℚ), c = -62 / 19 ∧ cubic_equation a b c = 0 :=
sorry

end third_root_of_cubic_equation_l80_8023


namespace silverware_probability_l80_8074

def numWaysTotal (totalPieces : ℕ) (choosePieces : ℕ) : ℕ :=
  Nat.choose totalPieces choosePieces

def numWaysForks (forks : ℕ) (chooseForks : ℕ) : ℕ :=
  Nat.choose forks chooseForks

def numWaysSpoons (spoons : ℕ) (chooseSpoons : ℕ) : ℕ :=
  Nat.choose spoons chooseSpoons

def numWaysKnives (knives : ℕ) (chooseKnives : ℕ) : ℕ :=
  Nat.choose knives chooseKnives

def favorableOutcomes (forks : ℕ) (spoons : ℕ) (knives : ℕ) : ℕ :=
  numWaysForks forks 2 * numWaysSpoons spoons 1 * numWaysKnives knives 1

def probability (totalWays : ℕ) (favorableWays : ℕ) : ℚ :=
  favorableWays / totalWays

theorem silverware_probability :
  probability (numWaysTotal 18 4) (favorableOutcomes 5 7 6) = 7 / 51 := by
  sorry

end silverware_probability_l80_8074


namespace jon_original_number_l80_8069

theorem jon_original_number :
  ∃ y : ℤ, (5 * (3 * y + 6) - 8 = 142) ∧ (y = 8) :=
sorry

end jon_original_number_l80_8069


namespace sales_volume_function_max_profit_min_boxes_for_2000_profit_l80_8090

-- Definitions and conditions
def cost_per_box : ℝ := 20
def min_selling_price : ℝ := 25
def init_boxes_sold : ℝ := 250
def price_increase_effect : ℝ := 10
def max_selling_price : ℝ := 38

-- Question 1: Find functional relationship between daily sales volume y and selling price per box x
theorem sales_volume_function (x : ℝ) (hx : x ≥ min_selling_price) :
  ∃ y, y = -10 * x + 500 := by
  sorry

-- Question 2: Find the price per box to maximize daily sales profit and the maximum profit
theorem max_profit (x : ℝ) (hx : x = 35) :
  ∃ P, P = -10 * (x-20) * (x) + 500 * (x-20) := by
  sorry

-- Question 3: Determine min boxes sold to make at least 2000 yuan given price does not exceed 38 yuan
theorem min_boxes_for_2000_profit (x : ℝ) (hx : min_selling_price ≤ x ∧ x ≤ max_selling_price ∧ 
                             -10 * (x-20) * (-10 * x + 500) ≥ 2000) :
  ∃ y, y = -10 * x + 500 ∧ y ≥ 120 := by
  sorry

end sales_volume_function_max_profit_min_boxes_for_2000_profit_l80_8090


namespace no_such_integers_l80_8061

def p (x : ℤ) : ℤ := x^2 + x - 70

theorem no_such_integers : ¬ (∃ m n : ℤ, 0 < m ∧ m < n ∧ n ∣ p m ∧ (n + 1) ∣ p (m + 1)) :=
by
  sorry

end no_such_integers_l80_8061


namespace only_solution_l80_8033

theorem only_solution (a b c : ℕ) (ha_pos : 0 < a) (hb_pos : 0 < b) (hc_pos : 0 < c)
    (h_le : a ≤ b ∧ b ≤ c) (h_gcd : Int.gcd (Int.gcd a b) c = 1) 
    (h_div_a2b : a^3 + b^3 + c^3 % (a^2 * b) = 0)
    (h_div_b2c : a^3 + b^3 + c^3 % (b^2 * c) = 0)
    (h_div_c2a : a^3 + b^3 + c^3 % (c^2 * a) = 0) : 
    a = 1 ∧ b = 1 ∧ c = 1 :=
  by
  sorry

end only_solution_l80_8033


namespace largest_n_multiple_3_l80_8038

theorem largest_n_multiple_3 (n : ℕ) (h1 : n < 100000) (h2 : (8 * (n + 2)^5 - n^2 + 14 * n - 30) % 3 = 0) : n = 99999 := 
sorry

end largest_n_multiple_3_l80_8038


namespace percent_of_z_l80_8024

variable (x y z : ℝ)

theorem percent_of_z :
  x = 1.20 * y →
  y = 0.40 * z →
  x = 0.48 * z :=
by
  intros h1 h2
  sorry

end percent_of_z_l80_8024


namespace nabla_example_l80_8048

def nabla (a b : ℕ) : ℕ := 2 + b ^ a

theorem nabla_example : nabla (nabla 1 2) 3 = 83 :=
  by
  sorry

end nabla_example_l80_8048


namespace min_value_a_b_l80_8097

variable (a b : ℝ)

theorem min_value_a_b (ha : a > 1) (hb : b > 1) (hab : a * b - (a + b) = 1) : 
  a + b ≥ 2 * (Real.sqrt 2 + 1) :=
sorry

end min_value_a_b_l80_8097


namespace pants_cost_correct_l80_8001

def shirt_cost : ℕ := 43
def tie_cost : ℕ := 15
def total_paid : ℕ := 200
def change_received : ℕ := 2

def total_spent : ℕ := total_paid - change_received
def combined_cost : ℕ := shirt_cost + tie_cost
def pants_cost : ℕ := total_spent - combined_cost

theorem pants_cost_correct : pants_cost = 140 :=
by
  -- We'll leave the proof as an exercise.
  sorry

end pants_cost_correct_l80_8001


namespace n_values_satisfy_condition_l80_8055

-- Define the exponential functions
def exp1 (n : ℤ) : ℚ := (-1/2) ^ n
def exp2 (n : ℤ) : ℚ := (-1/5) ^ n

-- Define the set of possible values for n
def valid_n : List ℤ := [-2, -1, 0, 1, 2, 3]

-- Define the condition for n to satisfy the inequality
def satisfies_condition (n : ℤ) : Prop := exp1 n > exp2 n

-- Prove that the only values of n that satisfy the condition are -1 and 2
theorem n_values_satisfy_condition :
  ∀ n ∈ valid_n, satisfies_condition n ↔ (n = -1 ∨ n = 2) :=
by
  intro n
  sorry

end n_values_satisfy_condition_l80_8055


namespace min_value_eq_144_l80_8015

noncomputable def min_value (x y z w : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0) (h_pos_w : w > 0) (h_sum : x + y + z + w = 1) : ℝ :=
  if x <= 0 ∨ y <= 0 ∨ z <= 0 ∨ w <= 0 then 0 else (x + y + z) / (x * y * z * w)

theorem min_value_eq_144 (x y z w : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0) (h_pos_w : w > 0) (h_sum : x + y + z + w = 1) :
  min_value x y z w h_pos_x h_pos_y h_pos_z h_pos_w h_sum = 144 :=
sorry

end min_value_eq_144_l80_8015


namespace solve_equation_1_solve_equation_2_l80_8032

theorem solve_equation_1 (x : ℝ) : (x + 2) ^ 2 = 3 * (x + 2) ↔ x = -2 ∨ x = 1 := by
  sorry

theorem solve_equation_2 (x : ℝ) : x ^ 2 - 8 * x + 3 = 0 ↔ x = 4 + Real.sqrt 13 ∨ x = 4 - Real.sqrt 13 := by
  sorry

end solve_equation_1_solve_equation_2_l80_8032


namespace cos_pi_over_3_plus_2alpha_l80_8035

variable (α : ℝ)

theorem cos_pi_over_3_plus_2alpha (h : Real.sin (π / 3 - α) = 1 / 3) :
  Real.cos (π / 3 + 2 * α) = 7 / 9 :=
  sorry

end cos_pi_over_3_plus_2alpha_l80_8035


namespace find_f_cos_10_l80_8026

noncomputable def f : ℝ → ℝ := sorry

axiom f_def (x : ℝ) : f (Real.sin x) = Real.cos (3 * x)

theorem find_f_cos_10 : f (Real.cos (10 * Real.pi / 180)) = -1/2 := by
  sorry

end find_f_cos_10_l80_8026


namespace problem_solution_l80_8066

noncomputable def expr := 
  (Real.tan (Real.pi / 15) - Real.sqrt 3) / ((4 * (Real.cos (Real.pi / 15))^2 - 2) * Real.sin (Real.pi / 15))

theorem problem_solution : expr = -4 :=
by
  sorry

end problem_solution_l80_8066


namespace partition_triangle_l80_8056

theorem partition_triangle (triangle : List ℕ) (h_triangle_sum : triangle.sum = 63) :
  ∃ (parts : List (List ℕ)), parts.length = 3 ∧ 
  (∀ part ∈ parts, part.sum = 21) ∧ 
  parts.bind id = triangle :=
by
  sorry

end partition_triangle_l80_8056


namespace cost_of_sculpture_cny_l80_8067

def exchange_rate_usd_to_nad := 8 -- 1 USD = 8 NAD
def exchange_rate_usd_to_cny := 5  -- 1 USD = 5 CNY
def cost_of_sculpture_nad := 160  -- Cost of sculpture in NAD

theorem cost_of_sculpture_cny : (cost_of_sculpture_nad / exchange_rate_usd_to_nad) * exchange_rate_usd_to_cny = 100 := by
  sorry

end cost_of_sculpture_cny_l80_8067
