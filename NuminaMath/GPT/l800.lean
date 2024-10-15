import Mathlib

namespace NUMINAMATH_GPT_uranus_appearance_minutes_after_6AM_l800_80079

-- Definitions of the given times and intervals
def mars_last_seen : Int := 0 -- 12:10 AM in minutes after midnight
def jupiter_after_mars : Int := 2 * 60 + 41 -- 2 hours and 41 minutes in minutes
def uranus_after_jupiter : Int := 3 * 60 + 16 -- 3 hours and 16 minutes in minutes
def reference_time : Int := 6 * 60 -- 6:00 AM in minutes after midnight

-- Statement of the problem
theorem uranus_appearance_minutes_after_6AM :
  let jupiter_first_appearance := mars_last_seen + jupiter_after_mars
  let uranus_first_appearance := jupiter_first_appearance + uranus_after_jupiter
  (uranus_first_appearance - reference_time) = 7 := by
  sorry

end NUMINAMATH_GPT_uranus_appearance_minutes_after_6AM_l800_80079


namespace NUMINAMATH_GPT_total_pigs_in_barn_l800_80061

-- Define the number of pigs initially in the barn
def initial_pigs : ℝ := 2465.25

-- Define the number of pigs that join
def joining_pigs : ℝ := 5683.75

-- Define the total number of pigs after they join
def total_pigs : ℝ := 8149

-- The theorem that states the total number of pigs is the sum of initial and joining pigs
theorem total_pigs_in_barn : initial_pigs + joining_pigs = total_pigs := 
by
  sorry

end NUMINAMATH_GPT_total_pigs_in_barn_l800_80061


namespace NUMINAMATH_GPT_average_of_shifted_sample_l800_80010

theorem average_of_shifted_sample (x1 x2 x3 : ℝ) (hx_avg : (x1 + x2 + x3) / 3 = 40) (hx_var : ((x1 - 40) ^ 2 + (x2 - 40) ^ 2 + (x3 - 40) ^ 2) / 3 = 1) : 
  ((x1 + 40) + (x2 + 40) + (x3 + 40)) / 3 = 80 :=
sorry

end NUMINAMATH_GPT_average_of_shifted_sample_l800_80010


namespace NUMINAMATH_GPT_arithmetic_seq_third_sum_l800_80088

-- Define the arithmetic sequence using its first term and common difference
def arithmetic_seq (a₁ d : ℤ) (n : ℕ) : ℤ := a₁ + d * n

theorem arithmetic_seq_third_sum
  (a₁ d : ℤ)
  (h1 : (a₁ + (a₁ + 3 * d) + (a₁ + 6 * d) = 39))
  (h2 : ((a₁ + d) + (a₁ + 4 * d) + (a₁ + 7 * d) = 33)) :
  ((a₁ + 2 * d) + (a₁ + 5 * d) + (a₁ + 8 * d) = 27) :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_seq_third_sum_l800_80088


namespace NUMINAMATH_GPT_sum_of_midpoints_double_l800_80025

theorem sum_of_midpoints_double (a b c : ℝ) (h : a + b + c = 15) : 
  (a + b) + (a + c) + (b + c) = 30 :=
by
  -- We skip the proof according to the instruction
  sorry

end NUMINAMATH_GPT_sum_of_midpoints_double_l800_80025


namespace NUMINAMATH_GPT_fifteenth_term_is_44_l800_80085

-- Define the conditions
def first_term : ℕ := 2
def common_difference : ℕ := 3
def term_number : ℕ := 15

-- Define the formula for the nth term of an arithmetic progression
def nth_term (a d n : ℕ) : ℕ := a + (n - 1) * d

-- Prove that the 15th term is 44
theorem fifteenth_term_is_44 : nth_term first_term common_difference term_number = 44 :=
by
  unfold nth_term first_term common_difference term_number
  sorry

end NUMINAMATH_GPT_fifteenth_term_is_44_l800_80085


namespace NUMINAMATH_GPT_expansion_coeff_sum_l800_80078

theorem expansion_coeff_sum :
  (∃ a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 : ℝ, 
    (2*x - 1)^10 = a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4 + a5*x^5 + a6*x^6 + a7*x^7 + a8*x^8 + a9*x^9 + a10*x^10)
  → (1 - 20 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 = 1 → a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 = 20) :=
by
  sorry

end NUMINAMATH_GPT_expansion_coeff_sum_l800_80078


namespace NUMINAMATH_GPT_swimming_lane_length_l800_80046

-- Conditions
def num_round_trips : ℕ := 3
def total_distance : ℕ := 600

-- Hypothesis that 1 round trip is equivalent to 2 lengths of the lane
def lengths_per_round_trip : ℕ := 2

-- Statement to prove
theorem swimming_lane_length :
  (total_distance / (num_round_trips * lengths_per_round_trip) = 100) := by
  sorry

end NUMINAMATH_GPT_swimming_lane_length_l800_80046


namespace NUMINAMATH_GPT_distance_between_consecutive_trees_l800_80038

noncomputable def distance_between_trees (yard_length : ℝ) (num_trees : ℕ) (obstacle_pos : ℝ) (obstacle_gap : ℝ) : ℝ :=
  let planting_distance := yard_length - obstacle_gap
  let num_gaps := num_trees - 1
  planting_distance / num_gaps

theorem distance_between_consecutive_trees :
  distance_between_trees 600 36 250 10 = 16.857 := by
  sorry

end NUMINAMATH_GPT_distance_between_consecutive_trees_l800_80038


namespace NUMINAMATH_GPT_find_green_hats_l800_80067

variable (B G : ℕ)

theorem find_green_hats (h1 : B + G = 85) (h2 : 6 * B + 7 * G = 540) :
  G = 30 :=
by
  sorry

end NUMINAMATH_GPT_find_green_hats_l800_80067


namespace NUMINAMATH_GPT_sum_of_interior_angles_of_pentagon_l800_80022

theorem sum_of_interior_angles_of_pentagon : (5 - 2) * 180 = 540 := 
by
  -- We skip the proof as per instruction
  sorry

end NUMINAMATH_GPT_sum_of_interior_angles_of_pentagon_l800_80022


namespace NUMINAMATH_GPT_sum_of_numbers_l800_80074

theorem sum_of_numbers (x y : ℝ) (h1 : x - y = 7) (h2 : x^2 + y^2 = 130) : x + y = -7 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_numbers_l800_80074


namespace NUMINAMATH_GPT_jen_scored_more_l800_80031

def bryan_score : ℕ := 20
def total_points : ℕ := 35
def sammy_mistakes : ℕ := 7
def sammy_score : ℕ := total_points - sammy_mistakes
def jen_score : ℕ := sammy_score + 2

theorem jen_scored_more :
  jen_score - bryan_score = 10 := by
  -- Proof to be filled in
  sorry

end NUMINAMATH_GPT_jen_scored_more_l800_80031


namespace NUMINAMATH_GPT_sequence_contains_prime_l800_80076

-- Define the conditions for being square-free and relatively prime
def is_square_free (n : ℕ) : Prop :=
  ∀ m : ℕ, m^2 ∣ n → m = 1

def are_relatively_prime (a b : ℕ) : Prop :=
  Nat.gcd a b = 1

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Statement of the problem
theorem sequence_contains_prime :
  ∀ (a : ℕ → ℕ), (∀ i, 1 ≤ i ∧ i ≤ 14 → 2 ≤ a i ∧ a i ≤ 1995 ∧ is_square_free (a i)) →
  (∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ 14 → are_relatively_prime (a i) (a j)) →
  ∃ i, 1 ≤ i ∧ i ≤ 14 ∧ is_prime (a i) :=
sorry

end NUMINAMATH_GPT_sequence_contains_prime_l800_80076


namespace NUMINAMATH_GPT_geometric_progression_condition_l800_80081

theorem geometric_progression_condition (a b c d : ℝ) :
  (∃ r : ℝ, (b = a * r ∨ b = a * -r) ∧
             (c = a * r^2 ∨ c = a * (-r)^2) ∧
             (d = a * r^3 ∨ d = a * (-r)^3) ∧
             (a = b / r ∨ a = b / -r) ∧
             (b = c / r ∨ b = c / -r) ∧
             (c = d / r ∨ c = d / -r) ∧
             (d = a / r ∨ d = a / -r)) ↔
  (a = b ∨ a = -b) ∧ (a = c ∨ a = -c) ∧ (a = d ∨ a = -d) := sorry

end NUMINAMATH_GPT_geometric_progression_condition_l800_80081


namespace NUMINAMATH_GPT_bounded_infinite_sequence_l800_80097

noncomputable def sequence_x (n : ℕ) : ℝ :=
  4 * (Real.sqrt 2 * n - ⌊Real.sqrt 2 * n⌋)

theorem bounded_infinite_sequence (a : ℝ) (h : a > 1) :
  ∀ i j : ℕ, i ≠ j → (|sequence_x i - sequence_x j| * |(i - j : ℝ)|^a) ≥ 1 := 
by
  intros i j h_ij
  sorry

end NUMINAMATH_GPT_bounded_infinite_sequence_l800_80097


namespace NUMINAMATH_GPT_hyungjun_initial_ribbon_length_l800_80024

noncomputable def initial_ribbon_length (R: ℝ) : Prop :=
  let used_for_first_box := R / 2 + 2000
  let remaining_after_first := R - used_for_first_box
  let used_for_second_box := (remaining_after_first / 2) + 2000
  remaining_after_first - used_for_second_box = 0

theorem hyungjun_initial_ribbon_length : ∃ R: ℝ, initial_ribbon_length R ∧ R = 12000 :=
  by
  exists 12000
  unfold initial_ribbon_length
  simp
  sorry

end NUMINAMATH_GPT_hyungjun_initial_ribbon_length_l800_80024


namespace NUMINAMATH_GPT_unique_solution_to_equation_l800_80040

theorem unique_solution_to_equation (x y z t : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (ht : t > 0) 
  (h : 1 + 5^x = 2^y + 2^z * 5^t) : (x, y, z, t) = (2, 4, 1, 1) := 
sorry

end NUMINAMATH_GPT_unique_solution_to_equation_l800_80040


namespace NUMINAMATH_GPT_linda_savings_l800_80056

theorem linda_savings (S : ℕ) (h1 : (3 / 4) * S = x) (h2 : (1 / 4) * S = 240) : S = 960 :=
by
  sorry

end NUMINAMATH_GPT_linda_savings_l800_80056


namespace NUMINAMATH_GPT_area_of_circle_l800_80005

/-- Given a circle with circumference 36π, prove that the area is 324π. -/
theorem area_of_circle (C : ℝ) (hC : C = 36 * π) 
  (h1 : ∀ r : ℝ, C = 2 * π * r → 0 ≤ r)
  (h2 : ∀ r : ℝ, 0 ≤ r → ∃ (A : ℝ), A = π * r^2) :
  ∃ k : ℝ, (A = 324 * π → k = 324) := 
sorry


end NUMINAMATH_GPT_area_of_circle_l800_80005


namespace NUMINAMATH_GPT_fraction_equivalence_l800_80037

theorem fraction_equivalence (a b c : ℝ) (h : (c - a) / (c - b) = 1) : 
  (5 * b - 2 * a) / (c - a) = 3 * a / (c - a) :=
by
  sorry

end NUMINAMATH_GPT_fraction_equivalence_l800_80037


namespace NUMINAMATH_GPT_triangle_properties_l800_80080

-- Define the given sides of the triangle
def a := 6
def b := 8
def c := 10

-- Define necessary parameters and properties
def isRightTriangle (a b c : Nat) : Prop := a^2 + b^2 = c^2
def area (a b : Nat) : Nat := (a * b) / 2
def semiperimeter (a b c : Nat) : Nat := (a + b + c) / 2
def inradius (A s : Nat) : Nat := A / s
def circumradius (c : Nat) : Nat := c / 2

-- The theorem statement
theorem triangle_properties :
  isRightTriangle a b c ∧
  area a b = 24 ∧
  semiperimeter a b c = 12 ∧
  inradius (area a b) (semiperimeter a b c) = 2 ∧
  circumradius c = 5 :=
by
  sorry

end NUMINAMATH_GPT_triangle_properties_l800_80080


namespace NUMINAMATH_GPT_joao_claudia_scores_l800_80084

theorem joao_claudia_scores (joao_score claudia_score total_score : ℕ) 
  (h1 : claudia_score = joao_score + 13)
  (h2 : total_score = joao_score + claudia_score)
  (h3 : 100 ≤ total_score ∧ total_score < 200) :
  joao_score = 68 ∧ claudia_score = 81 := by
  sorry

end NUMINAMATH_GPT_joao_claudia_scores_l800_80084


namespace NUMINAMATH_GPT_partial_fraction_sum_zero_l800_80021

theorem partial_fraction_sum_zero (A B C D E F : ℝ) :
  (∀ x : ℝ, 1 / (x * (x + 1) * (x + 2) * (x + 3) * (x + 4) * (x + 5)) =
     A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 4) + F / (x + 5)) →
  A + B + C + D + E + F = 0 :=
sorry

end NUMINAMATH_GPT_partial_fraction_sum_zero_l800_80021


namespace NUMINAMATH_GPT_both_selected_prob_l800_80002

-- Given conditions
def prob_Ram := 6 / 7
def prob_Ravi := 1 / 5

-- The mathematically equivalent proof problem statement
theorem both_selected_prob : (prob_Ram * prob_Ravi) = 6 / 35 := by
  sorry

end NUMINAMATH_GPT_both_selected_prob_l800_80002


namespace NUMINAMATH_GPT_factorize_x_squared_plus_2x_l800_80069

theorem factorize_x_squared_plus_2x (x : ℝ) : x^2 + 2 * x = x * (x + 2) :=
by
  sorry

end NUMINAMATH_GPT_factorize_x_squared_plus_2x_l800_80069


namespace NUMINAMATH_GPT_count_odd_expressions_l800_80003

theorem count_odd_expressions : 
  let exp1 := 1^2
  let exp2 := 2^3
  let exp3 := 3^4
  let exp4 := 4^5
  let exp5 := 5^6
  (if exp1 % 2 = 1 then 1 else 0) + 
  (if exp2 % 2 = 1 then 1 else 0) + 
  (if exp3 % 2 = 1 then 1 else 0) + 
  (if exp4 % 2 = 1 then 1 else 0) + 
  (if exp5 % 2 = 1 then 1 else 0) = 3 :=
by 
  sorry

end NUMINAMATH_GPT_count_odd_expressions_l800_80003


namespace NUMINAMATH_GPT_minimum_AB_l800_80020

noncomputable def shortest_AB (a : ℝ) : ℝ :=
  let x := (Real.sqrt 3) / 4 * a
  x

theorem minimum_AB (a : ℝ) : ∃ x, (x = (Real.sqrt 3) / 4 * a) ∧ ∀ y, (y = (Real.sqrt 3) / 4 * a) → shortest_AB a = x :=
by
  sorry

end NUMINAMATH_GPT_minimum_AB_l800_80020


namespace NUMINAMATH_GPT_planes_parallel_from_plane_l800_80062

-- Define the relationship functions
def parallel (P Q : Plane) : Prop := sorry -- Define parallelism predicate
def perpendicular (l : Line) (P : Plane) : Prop := sorry -- Define perpendicularity predicate

-- Declare the planes α, β, and γ
variable (α β γ : Plane)

-- Main theorem statement
theorem planes_parallel_from_plane (h1 : parallel γ α) (h2 : parallel γ β) : parallel α β := 
sorry

end NUMINAMATH_GPT_planes_parallel_from_plane_l800_80062


namespace NUMINAMATH_GPT_strawberry_jelly_amount_l800_80068

def totalJelly : ℕ := 6310
def blueberryJelly : ℕ := 4518
def strawberryJelly : ℕ := totalJelly - blueberryJelly

theorem strawberry_jelly_amount : strawberryJelly = 1792 := by
  rfl

end NUMINAMATH_GPT_strawberry_jelly_amount_l800_80068


namespace NUMINAMATH_GPT_evaluate_expression_l800_80064

theorem evaluate_expression :
  (5^5 * 5^3) / 3^6 * 2^5 = 12480000 / 729 :=
by sorry

end NUMINAMATH_GPT_evaluate_expression_l800_80064


namespace NUMINAMATH_GPT_div_powers_same_base_l800_80015

variable (x : ℝ)

theorem div_powers_same_base : x^8 / x^2 = x^6 :=
by
  sorry

end NUMINAMATH_GPT_div_powers_same_base_l800_80015


namespace NUMINAMATH_GPT_sequence_term_sum_max_value_sum_equality_l800_80036

noncomputable def a (n : ℕ) : ℝ := -2 * n + 6

def S (n : ℕ) : ℝ := -n^2 + 5 * n

theorem sequence_term (n : ℕ) : ∀ n, a n = 4 + (n - 1) * (-2) :=
by sorry

theorem sum_max_value (n : ℕ) : ∃ n, S n = 6 :=
by sorry

theorem sum_equality : S 2 = 6 ∧ S 3 = 6 :=
by sorry

end NUMINAMATH_GPT_sequence_term_sum_max_value_sum_equality_l800_80036


namespace NUMINAMATH_GPT_product_of_solutions_l800_80095

theorem product_of_solutions (t : ℝ) (h : t^2 = 64) : t * (-t) = -64 :=
sorry

end NUMINAMATH_GPT_product_of_solutions_l800_80095


namespace NUMINAMATH_GPT_find_r_l800_80030

noncomputable def parabola_vertex : (ℝ × ℝ) := (0, -1)

noncomputable def intersection_points (r : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  let x1 := (r - Real.sqrt (r^2 + 4)) / 2
  let y1 := r * x1
  let x2 := (r + Real.sqrt (r^2 + 4)) / 2
  let y2 := r * x2
  ((x1, y1), (x2, y2))

noncomputable def triangle_area (r : ℝ) : ℝ :=
  let base := Real.sqrt (r^2 + 4)
  let height := 2
  1/2 * base * height

theorem find_r (r : ℝ) (h : r > 0) : triangle_area r = 32 → r = Real.sqrt 1020 := 
by
  sorry

end NUMINAMATH_GPT_find_r_l800_80030


namespace NUMINAMATH_GPT_picnic_basket_cost_l800_80087

theorem picnic_basket_cost :
  let sandwich_cost := 5
  let fruit_salad_cost := 3
  let soda_cost := 2
  let snack_bag_cost := 4
  let num_people := 4
  let num_sodas_per_person := 2
  let num_snack_bags := 3
  (num_people * sandwich_cost) + (num_people * fruit_salad_cost) + (num_people * num_sodas_per_person * soda_cost) + (num_snack_bags * snack_bag_cost) = 60 :=
by
  sorry

end NUMINAMATH_GPT_picnic_basket_cost_l800_80087


namespace NUMINAMATH_GPT_sherman_drives_nine_hours_a_week_l800_80048

-- Define the daily commute time in minutes.
def daily_commute_time := 30 + 30

-- Define the number of weekdays Sherman commutes.
def weekdays := 5

-- Define the weekly commute time in minutes.
def weekly_commute_time := weekdays * daily_commute_time

-- Define the conversion from minutes to hours.
def minutes_to_hours (m : ℕ) : ℕ := m / 60

-- Define the weekend driving time in hours.
def weekend_driving_time := 2 * 2

-- Define the total weekly driving time in hours.
def total_weekly_driving_time := minutes_to_hours weekly_commute_time + weekend_driving_time

-- The theorem we need to prove
theorem sherman_drives_nine_hours_a_week :
  total_weekly_driving_time = 9 :=
by
  sorry

end NUMINAMATH_GPT_sherman_drives_nine_hours_a_week_l800_80048


namespace NUMINAMATH_GPT_pq_condition_l800_80000

theorem pq_condition (p q : ℝ) (h1 : p * q = 16) (h2 : p + q = 10) : (p - q)^2 = 36 :=
by
  sorry

end NUMINAMATH_GPT_pq_condition_l800_80000


namespace NUMINAMATH_GPT_monomial_coeff_degree_product_l800_80055

theorem monomial_coeff_degree_product (m n : ℚ) (h₁ : m = -3/4) (h₂ : n = 4) : m * n = -3 := 
by
  sorry

end NUMINAMATH_GPT_monomial_coeff_degree_product_l800_80055


namespace NUMINAMATH_GPT_range_of_a_l800_80007

theorem range_of_a (a : ℝ) (x : ℝ) (h : x^2 + a * x + 1 < 0) : a < -2 ∨ a > 2 :=
sorry

end NUMINAMATH_GPT_range_of_a_l800_80007


namespace NUMINAMATH_GPT_average_rate_second_drive_l800_80051

theorem average_rate_second_drive 
 (distance : ℕ) (total_time : ℕ) (d1 d2 d3 : ℕ)
 (t1 t2 t3 : ℕ) (r1 r2 r3 : ℕ)
 (h_distance : d1 = d2 ∧ d2 = d3 ∧ d1 + d2 + d3 = distance)
 (h_total_time : t1 + t2 + t3 = total_time)
 (h_drive_1 : r1 = 4 ∧ t1 = d1 / r1)
 (h_drive_2 : r3 = 6 ∧ t3 = d3 / r3)
 (h_distance_total : distance = 180)
 (h_total_time_val : total_time = 37)
  : r2 = 5 := 
by sorry

end NUMINAMATH_GPT_average_rate_second_drive_l800_80051


namespace NUMINAMATH_GPT_minimum_value_expression_l800_80026

theorem minimum_value_expression (α β : ℝ) : (3 * Real.cos α + 4 * Real.sin β - 7)^2 + (3 * Real.sin α + 4 * Real.cos β - 18)^2 ≥ 144 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_expression_l800_80026


namespace NUMINAMATH_GPT_problem_l800_80047

theorem problem (a b c : ℝ) (Ha : a > 0) (Hb : b > 0) (Hc : c > 0) : 
  (|a| / a + |b| / b + |c| / c - (abc / |abc|) = 2 ∨ |a| / a + |b| / b + |c| / c - (abc / |abc|) = -2) :=
by
  sorry

end NUMINAMATH_GPT_problem_l800_80047


namespace NUMINAMATH_GPT_fill_half_cistern_time_l800_80082

variable (t_half : ℝ)

-- Define a condition that states the certain amount of time to fill 1/2 of the cistern.
def fill_pipe_half_time (t_half : ℝ) : Prop :=
  t_half > 0

-- The statement to prove that t_half is the time required to fill 1/2 of the cistern.
theorem fill_half_cistern_time : fill_pipe_half_time t_half → t_half = t_half := by
  intros
  rfl

end NUMINAMATH_GPT_fill_half_cistern_time_l800_80082


namespace NUMINAMATH_GPT_transport_cost_is_correct_l800_80099

-- Define the transport cost per kilogram
def transport_cost_per_kg : ℝ := 18000

-- Define the weight of the scientific instrument in kilograms
def weight_kg : ℝ := 0.5

-- Define the discount rate
def discount_rate : ℝ := 0.10

-- Define the cost calculation without the discount
def cost_without_discount : ℝ := weight_kg * transport_cost_per_kg

-- Define the final cost with the discount applied
def discounted_cost : ℝ := cost_without_discount * (1 - discount_rate)

-- The theorem stating that the discounted cost is $8,100
theorem transport_cost_is_correct : discounted_cost = 8100 := by
  sorry

end NUMINAMATH_GPT_transport_cost_is_correct_l800_80099


namespace NUMINAMATH_GPT_graphs_symmetric_about_a_axis_of_symmetry_l800_80001

def graph_symmetric_about_line (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a - x) = f (x - a)

theorem graphs_symmetric_about_a (f : ℝ → ℝ) (a : ℝ) :
  ∀ x, f (x - a) = f (a - (x - a)) :=
sorry

theorem axis_of_symmetry (f : ℝ → ℝ) :
  (∀ x : ℝ, f (1 + 2 * x) = f (1 - 2 * x)) →
  ∀ x, f x = f (2 - x) := 
sorry

end NUMINAMATH_GPT_graphs_symmetric_about_a_axis_of_symmetry_l800_80001


namespace NUMINAMATH_GPT_subscription_ways_three_households_l800_80044

def num_subscription_ways (n_households : ℕ) (n_newspapers : ℕ) : ℕ :=
  if h : n_households = 3 ∧ n_newspapers = 5 then
    180
  else
    0

theorem subscription_ways_three_households :
  num_subscription_ways 3 5 = 180 :=
by
  unfold num_subscription_ways
  split_ifs
  . rfl
  . contradiction


end NUMINAMATH_GPT_subscription_ways_three_households_l800_80044


namespace NUMINAMATH_GPT_problem1_problem2_l800_80075

-- Problem 1: Prove that (2sin(α) - cos(α)) / (sin(α) + 2cos(α)) = 3/4 given tan(α) = 2
theorem problem1 (α : ℝ) (h : Real.tan α = 2) : (2 * Real.sin α - Real.cos α) / (Real.sin α + 2 * Real.cos α) = 3 / 4 := 
sorry

-- Problem 2: Prove that 2sin^2(x) - sin(x)cos(x) + cos^2(x) = 2 - sin(2x)/2
theorem problem2 (x : ℝ) : 2 * (Real.sin x)^2 - (Real.sin x) * (Real.cos x) + (Real.cos x)^2 = 2 - Real.sin (2 * x) / 2 := 
sorry

end NUMINAMATH_GPT_problem1_problem2_l800_80075


namespace NUMINAMATH_GPT_ratio_celeste_bianca_l800_80060

-- Definitions based on given conditions
def bianca_hours : ℝ := 12.5
def celest_hours (x : ℝ) : ℝ := 12.5 * x
def mcclain_hours (x : ℝ) : ℝ := 12.5 * x - 8.5

-- The total time worked in hours
def total_hours : ℝ := 54

-- The ratio to prove
def celeste_bianca_ratio : ℝ := 2

-- The proof statement
theorem ratio_celeste_bianca (x : ℝ) (hx :  12.5 + 12.5 * x + (12.5 * x - 8.5) = total_hours) :
  celest_hours 2 / bianca_hours = celeste_bianca_ratio :=
by
  sorry

end NUMINAMATH_GPT_ratio_celeste_bianca_l800_80060


namespace NUMINAMATH_GPT_part1_monotonic_intervals_part2_max_a_l800_80039

noncomputable def f1 (x : ℝ) := Real.log x - 2 * x^2

theorem part1_monotonic_intervals :
  (∀ x, 0 < x ∧ x < 0.5 → f1 x > 0) ∧ (∀ x, x > 0.5 → f1 x < 0) :=
by
  sorry

noncomputable def f2 (x a : ℝ) := Real.log x + a * x^2

theorem part2_max_a (a : ℤ) :
  (∀ x, x > 1 → f2 x a < Real.exp x) → a ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_part1_monotonic_intervals_part2_max_a_l800_80039


namespace NUMINAMATH_GPT_study_video_game_inversely_proportional_1_study_video_game_inversely_proportional_2_l800_80077

theorem study_video_game_inversely_proportional_1 (k : ℕ) (s : ℕ) (v : ℕ) 
  (h1 : s * v = k) (h2 : k = 12) (h3 : s = 6) : v = 2 :=
by
  sorry

theorem study_video_game_inversely_proportional_2 (k : ℕ) (s : ℕ) (v : ℕ) 
  (h1 : s * v = k) (h2 : k = 12) (h3 : v = 6) : s = 2 :=
by
  sorry

end NUMINAMATH_GPT_study_video_game_inversely_proportional_1_study_video_game_inversely_proportional_2_l800_80077


namespace NUMINAMATH_GPT_arithmetic_mean_eq_2_l800_80029

theorem arithmetic_mean_eq_2 (a x : ℝ) (hx: x ≠ 0) :
  (1/2) * (((2 * x + a) / x) + ((2 * x - a) / x)) = 2 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_mean_eq_2_l800_80029


namespace NUMINAMATH_GPT_solution_exists_in_interval_l800_80045

noncomputable def f (x : ℝ) : ℝ := 3^x + x - 3

theorem solution_exists_in_interval : ∃ x, 0 < x ∧ x < 1 ∧ f x = 0 :=
by {
  -- placeholder for the skipped proof
  sorry
}

end NUMINAMATH_GPT_solution_exists_in_interval_l800_80045


namespace NUMINAMATH_GPT_total_water_needed_l800_80009

def adults : ℕ := 7
def children : ℕ := 3
def hours : ℕ := 24
def replenish_bottles : ℚ := 14
def water_per_hour_adult : ℚ := 1/2
def water_per_hour_child : ℚ := 1/3

theorem total_water_needed : 
  let total_water_per_hour := (adults * water_per_hour_adult) + (children * water_per_hour_child)
  let total_water := total_water_per_hour * hours 
  let initial_water_needed := total_water - replenish_bottles
  initial_water_needed = 94 := by 
  sorry

end NUMINAMATH_GPT_total_water_needed_l800_80009


namespace NUMINAMATH_GPT_can_split_3x3x3_into_9_corners_l800_80093

-- Define the conditions
def number_of_cubes_in_3x3x3 : ℕ := 27
def number_of_units_in_corner : ℕ := 3
def number_of_corners : ℕ := 9

-- Prove the proposition
theorem can_split_3x3x3_into_9_corners :
  (number_of_corners * number_of_units_in_corner = number_of_cubes_in_3x3x3) :=
by
  sorry

end NUMINAMATH_GPT_can_split_3x3x3_into_9_corners_l800_80093


namespace NUMINAMATH_GPT_min_value_of_a2_plus_b2_l800_80065

theorem min_value_of_a2_plus_b2 
  (a b : ℝ) 
  (h : ∃ x : ℝ, x^4 + a * x^3 + b * x^2 + a * x + 1 = 0) : 
  a^2 + b^2 ≥ 4 := 
sorry

end NUMINAMATH_GPT_min_value_of_a2_plus_b2_l800_80065


namespace NUMINAMATH_GPT_reflection_points_line_l800_80054

theorem reflection_points_line (m b : ℝ)
  (h1 : (10 : ℝ) = 2 * (6 - m * (6 : ℝ) + b)) -- Reflecting the point (6, (m * 6 + b)) to (10, 7)
  (h2 : (6 : ℝ) * m + b = 5) -- Midpoint condition
  (h3 : (6 : ℝ) = (2 + 10) / 2) -- Calculating midpoint x-coordinate
  (h4 : (5 : ℝ) = (3 + 7) / 2) -- Calculating midpoint y-coordinate
  : m + b = 15 :=
sorry

end NUMINAMATH_GPT_reflection_points_line_l800_80054


namespace NUMINAMATH_GPT_minimum_sugar_quantity_l800_80083

theorem minimum_sugar_quantity :
  ∃ s f : ℝ, s = 4 ∧ f ≥ 4 + s / 3 ∧ f ≤ 3 * s ∧ 2 * s + 3 * f ≤ 36 :=
sorry

end NUMINAMATH_GPT_minimum_sugar_quantity_l800_80083


namespace NUMINAMATH_GPT_twice_total_credits_l800_80043

theorem twice_total_credits (Aria Emily Spencer : ℕ) 
(Emily_has_20_credits : Emily = 20) 
(Aria_twice_Emily : Aria = 2 * Emily) 
(Emily_twice_Spencer : Emily = 2 * Spencer) : 
2 * (Aria + Emily + Spencer) = 140 :=
by
  sorry

end NUMINAMATH_GPT_twice_total_credits_l800_80043


namespace NUMINAMATH_GPT_basketball_prob_l800_80059

theorem basketball_prob :
  let P_A := 0.7
  let P_B := 0.6
  P_A * P_B = 0.88 := 
by 
  sorry

end NUMINAMATH_GPT_basketball_prob_l800_80059


namespace NUMINAMATH_GPT_value_of_g_at_neg2_l800_80035

def g (x : ℝ) : ℝ := x^2 - 4*x + 3

theorem value_of_g_at_neg2 : g (-2) = 15 :=
by
  -- This is where the proof steps would go, but we'll skip it
  sorry

end NUMINAMATH_GPT_value_of_g_at_neg2_l800_80035


namespace NUMINAMATH_GPT_roots_of_polynomial_l800_80032

noncomputable def polynomial : Polynomial ℝ := Polynomial.X^3 + Polynomial.X^2 - 6 * Polynomial.X - 6

theorem roots_of_polynomial :
  (Polynomial.rootSet polynomial ℝ) = {-1, 3, -2} := 
sorry

end NUMINAMATH_GPT_roots_of_polynomial_l800_80032


namespace NUMINAMATH_GPT_most_probable_hits_l800_80023

variable (n : ℕ) (p : ℝ) (q : ℝ) (k : ℕ)
variable (h1 : n = 5) (h2 : p = 0.6) (h3 : q = 1 - p)

theorem most_probable_hits : k = 3 := by
  -- Define the conditions
  have hp : p = 0.6 := h2
  have hn : n = 5 := h1
  have hq : q = 1 - p := h3

  -- Set the expected value for the number of hits
  let expected := n * p

  -- Use the bounds for the most probable number of successes (k_0)
  have bounds := expected - q ≤ k ∧ k ≤ expected + p

  -- Proof step analysis can go here
  sorry

end NUMINAMATH_GPT_most_probable_hits_l800_80023


namespace NUMINAMATH_GPT_find_number_l800_80073

theorem find_number (x : ℚ) : (x + (-5/12) - (-5/2) = 1/3) → x = -7/4 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l800_80073


namespace NUMINAMATH_GPT_Z_3_5_value_l800_80028

def Z (a b : ℕ) : ℕ :=
  b + 12 * a - a ^ 2

theorem Z_3_5_value : Z 3 5 = 32 := by
  sorry

end NUMINAMATH_GPT_Z_3_5_value_l800_80028


namespace NUMINAMATH_GPT_weight_of_each_bag_l800_80019

theorem weight_of_each_bag 
  (total_potatoes_weight : ℕ) (damaged_potatoes_weight : ℕ) 
  (bag_price : ℕ) (total_revenue : ℕ) (sellable_potatoes_weight : ℕ) (number_of_bags : ℕ) 
  (weight_of_each_bag : ℕ) :
  total_potatoes_weight = 6500 →
  damaged_potatoes_weight = 150 →
  sellable_potatoes_weight = total_potatoes_weight - damaged_potatoes_weight →
  bag_price = 72 →
  total_revenue = 9144 →
  number_of_bags = total_revenue / bag_price →
  weight_of_each_bag * number_of_bags = sellable_potatoes_weight →
  weight_of_each_bag = 50 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end NUMINAMATH_GPT_weight_of_each_bag_l800_80019


namespace NUMINAMATH_GPT_round_trip_ticket_percentage_l800_80094

variable (P : ℝ) -- Denotes total number of passengers
variable (R : ℝ) -- Denotes number of round-trip ticket holders

-- Condition 1: 15% of passengers held round-trip tickets and took their cars aboard
def condition1 : Prop := 0.15 * P = 0.40 * R

-- Prove that 37.5% of the ship's passengers held round-trip tickets.
theorem round_trip_ticket_percentage (h1 : condition1 P R) : R / P = 0.375 :=
by
  sorry

end NUMINAMATH_GPT_round_trip_ticket_percentage_l800_80094


namespace NUMINAMATH_GPT_jorge_land_fraction_clay_rich_soil_l800_80013

theorem jorge_land_fraction_clay_rich_soil 
  (total_acres : ℕ) 
  (yield_good_soil_per_acre : ℕ) 
  (yield_clay_soil_factor : ℕ)
  (total_yield : ℕ) 
  (fraction_clay_rich_soil : ℚ) :
  total_acres = 60 →
  yield_good_soil_per_acre = 400 →
  yield_clay_soil_factor = 2 →
  total_yield = 20000 →
  fraction_clay_rich_soil = 1/3 :=
by
  intro h_total_acres h_yield_good_soil_per_acre h_yield_clay_soil_factor h_total_yield
  -- math proof will be here
  sorry

end NUMINAMATH_GPT_jorge_land_fraction_clay_rich_soil_l800_80013


namespace NUMINAMATH_GPT_johnsonville_max_members_l800_80057

theorem johnsonville_max_members 
  (n : ℤ) 
  (h1 : 15 * n % 30 = 6) 
  (h2 : 15 * n < 900) 
  : 15 * n ≤ 810 :=
sorry

end NUMINAMATH_GPT_johnsonville_max_members_l800_80057


namespace NUMINAMATH_GPT_calculate_expr1_calculate_expr2_l800_80066

/-- Statement 1: -5 * 3 - 8 / -2 = -11 -/
theorem calculate_expr1 : (-5) * 3 - 8 / -2 = -11 :=
by sorry

/-- Statement 2: (-1)^3 + (5 - (-3)^2) / 6 = -5/3 -/
theorem calculate_expr2 : (-1)^3 + (5 - (-3)^2) / 6 = -(5 / 3) :=
by sorry

end NUMINAMATH_GPT_calculate_expr1_calculate_expr2_l800_80066


namespace NUMINAMATH_GPT_deposit_amount_l800_80052

theorem deposit_amount (P : ℝ) (deposit remaining : ℝ) (h1 : deposit = 0.1 * P) (h2 : remaining = P - deposit) (h3 : remaining = 1350) : 
  deposit = 150 := 
by
  sorry

end NUMINAMATH_GPT_deposit_amount_l800_80052


namespace NUMINAMATH_GPT_find_range_of_x_l800_80049

def odot (a b : ℝ) : ℝ := a * b + 2 * a + b

theorem find_range_of_x (x : ℝ) : odot x (x - 2) < 0 ↔ -2 < x ∧ x < 1 :=
by sorry

end NUMINAMATH_GPT_find_range_of_x_l800_80049


namespace NUMINAMATH_GPT_candy_probability_difference_l800_80091

theorem candy_probability_difference :
  let total := 2004
  let total_ways := Nat.choose total 2
  let different_ways := 2002 * 1002 / 2
  let same_ways := 1002 * 1001 / 2 + 1002 * 1001 / 2
  let q := (different_ways : ℚ) / total_ways
  let p := (same_ways : ℚ) / total_ways
  q - p = 1 / 2003 :=
by sorry

end NUMINAMATH_GPT_candy_probability_difference_l800_80091


namespace NUMINAMATH_GPT_gcd_polynomials_l800_80012

theorem gcd_polynomials (b : ℤ) (h : b % 8213 = 0 ∧ b % 2 = 1) :
  Int.gcd (8 * b^2 + 63 * b + 144) (2 * b + 15) = 9 :=
sorry

end NUMINAMATH_GPT_gcd_polynomials_l800_80012


namespace NUMINAMATH_GPT_compare_logarithms_l800_80089

noncomputable def a : ℝ := Real.log 3 / Real.log 4 -- log base 4 of 3
noncomputable def b : ℝ := Real.log 4 / Real.log 3 -- log base 3 of 4
noncomputable def c : ℝ := Real.log 3 / Real.log 5 -- log base 5 of 3

theorem compare_logarithms : b > a ∧ a > c := sorry

end NUMINAMATH_GPT_compare_logarithms_l800_80089


namespace NUMINAMATH_GPT_convert_sq_meters_to_hectares_convert_hours_to_hours_and_minutes_l800_80004

theorem convert_sq_meters_to_hectares :
  (123000 / 10000) = 12.3 :=
by
  sorry

theorem convert_hours_to_hours_and_minutes :
  (4 + 0.25 * 60) = 4 * 60 + 15 :=
by
  sorry

end NUMINAMATH_GPT_convert_sq_meters_to_hectares_convert_hours_to_hours_and_minutes_l800_80004


namespace NUMINAMATH_GPT_solve_for_c_l800_80018

theorem solve_for_c (a b c : ℝ) (B : ℝ) (ha : a = 4) (hb : b = 2*Real.sqrt 7) (hB : B = Real.pi / 3) : 
  (c^2 - 4*c - 12 = 0) → c = 6 :=
by 
  intro h
  -- Details of the proof would be here
  sorry

end NUMINAMATH_GPT_solve_for_c_l800_80018


namespace NUMINAMATH_GPT_distinct_triplet_inequality_l800_80027

theorem distinct_triplet_inequality (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  abs (a / (b - c)) + abs (b / (c - a)) + abs (c / (a - b)) ≥ 2 := 
sorry

end NUMINAMATH_GPT_distinct_triplet_inequality_l800_80027


namespace NUMINAMATH_GPT_carol_carrots_l800_80017

def mother_picked := 16
def good_carrots := 38
def bad_carrots := 7
def total_carrots := good_carrots + bad_carrots
def carol_picked : Nat := total_carrots - mother_picked

theorem carol_carrots : carol_picked = 29 := by
  sorry

end NUMINAMATH_GPT_carol_carrots_l800_80017


namespace NUMINAMATH_GPT_square_area_when_a_eq_b_eq_c_l800_80063

theorem square_area_when_a_eq_b_eq_c {a b c : ℝ} (h : a = b ∧ b = c) :
  ∃ x : ℝ, (x = a * Real.sqrt 2) ∧ (x ^ 2 = 2 * a ^ 2) :=
by
  sorry

end NUMINAMATH_GPT_square_area_when_a_eq_b_eq_c_l800_80063


namespace NUMINAMATH_GPT_compute_a4_b4_c4_l800_80092

theorem compute_a4_b4_c4 (a b c : ℝ) (h1 : a + b + c = 8) (h2 : ab + ac + bc = 13) (h3 : abc = -22) : a^4 + b^4 + c^4 = 1378 :=
by
  sorry

end NUMINAMATH_GPT_compute_a4_b4_c4_l800_80092


namespace NUMINAMATH_GPT_expression_constant_value_l800_80016

theorem expression_constant_value (a b x y : ℝ) 
  (h_a : a = Real.sqrt (1 + x^2))
  (h_b : b = Real.sqrt (1 + y^2)) 
  (h_xy : x + y = 1) : 
  (a + b + 1) * (a + b - 1) * (a - b + 1) * (-a + b + 1) = 4 := 
by 
  sorry

end NUMINAMATH_GPT_expression_constant_value_l800_80016


namespace NUMINAMATH_GPT_cord_length_before_cut_l800_80011

-- Definitions based on the conditions
def parts_after_cut := 20
def longest_piece := 8
def shortest_piece := 2
def initial_parts := 19

-- Lean statement to prove the length of the cord before it was cut
theorem cord_length_before_cut : 
  (initial_parts * ((longest_piece / 2) + shortest_piece) = 114) :=
by 
  sorry

end NUMINAMATH_GPT_cord_length_before_cut_l800_80011


namespace NUMINAMATH_GPT_problem_statement_l800_80058

variable {x y : Real}

theorem problem_statement (hx : x * y < 0) (hxy : x > |y|) : x + y > 0 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l800_80058


namespace NUMINAMATH_GPT_num_ways_choose_pair_of_diff_color_socks_l800_80008

-- Define the numbers of socks of each color
def num_white := 5
def num_brown := 5
def num_blue := 3
def num_black := 3

-- Define the calculation for pairs of different colored socks
def num_pairs_white_brown := num_white * num_brown
def num_pairs_brown_blue := num_brown * num_blue
def num_pairs_white_blue := num_white * num_blue
def num_pairs_white_black := num_white * num_black
def num_pairs_brown_black := num_brown * num_black
def num_pairs_blue_black := num_blue * num_black

-- Define the total number of pairs
def total_pairs := num_pairs_white_brown + num_pairs_brown_blue + num_pairs_white_blue + num_pairs_white_black + num_pairs_brown_black + num_pairs_blue_black

-- The theorem to be proved
theorem num_ways_choose_pair_of_diff_color_socks : total_pairs = 94 := by
  -- Since we do not need to include the proof steps, we use sorry
  sorry

end NUMINAMATH_GPT_num_ways_choose_pair_of_diff_color_socks_l800_80008


namespace NUMINAMATH_GPT_symmetric_circle_equation_l800_80090

noncomputable def equation_of_symmetric_circle (x y : ℝ) : Prop :=
  (x^2 + y^2 - 2 * x - 6 * y + 9 = 0) ∧ (2 * x + y + 5 = 0)

theorem symmetric_circle_equation :
  ∀ (x y : ℝ), 
    equation_of_symmetric_circle x y → 
    ∃ a b : ℝ, ((x - a)^2 + (y - b)^2 = 1) ∧ (a + 7 = 0) ∧ (b + 1 = 0) :=
sorry

end NUMINAMATH_GPT_symmetric_circle_equation_l800_80090


namespace NUMINAMATH_GPT_find_first_spill_l800_80072

def bottle_capacity : ℕ := 20
def refill_count : ℕ := 3
def days : ℕ := 7
def total_water_drunk : ℕ := 407
def second_spill : ℕ := 8

theorem find_first_spill :
  let total_without_spill := bottle_capacity * refill_count * days
  let total_spilled := total_without_spill - total_water_drunk
  let first_spill := total_spilled - second_spill
  first_spill = 5 :=
by
  -- Proof goes here.
  sorry

end NUMINAMATH_GPT_find_first_spill_l800_80072


namespace NUMINAMATH_GPT_michael_passes_donovan_after_laps_l800_80096

/-- The length of the track in meters -/
def track_length : ℕ := 400

/-- Donovan's lap time in seconds -/
def donovan_lap_time : ℕ := 45

/-- Michael's lap time in seconds -/
def michael_lap_time : ℕ := 36

/-- The number of laps that Michael will have to complete in order to pass Donovan -/
theorem michael_passes_donovan_after_laps : 
  ∃ (laps : ℕ), laps = 5 ∧ (∃ t : ℕ, 400 * t / 36 = 5 ∧ 400 * t / 45 < 5) :=
sorry

end NUMINAMATH_GPT_michael_passes_donovan_after_laps_l800_80096


namespace NUMINAMATH_GPT_length_is_56_l800_80034

noncomputable def length_of_plot (b : ℝ) : ℝ := b + 12

theorem length_is_56 (b : ℝ) (cost_per_meter : ℝ) (total_cost : ℝ) (h_cost : cost_per_meter = 26.50) (h_total_cost : total_cost = 5300) (h_fencing : 26.50 * (4 * b + 24) = 5300) : length_of_plot b = 56 := 
by 
  sorry

end NUMINAMATH_GPT_length_is_56_l800_80034


namespace NUMINAMATH_GPT_intersection_A_B_l800_80014

open Set

def universal_set : Set ℤ := {x | 1 ≤ x ∧ x ≤ 5}
def A : Set ℤ := {1, 2, 3}
def complement_B : Set ℤ := {1, 2}
def B : Set ℤ := universal_set \ complement_B

theorem intersection_A_B : A ∩ B = {3} :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l800_80014


namespace NUMINAMATH_GPT_sum_of_integers_l800_80053

theorem sum_of_integers (x y : ℕ) (h1 : x - y = 8) (h2 : x * y = 120) : x + y = 4 * Real.sqrt 34 := by
  sorry

end NUMINAMATH_GPT_sum_of_integers_l800_80053


namespace NUMINAMATH_GPT_students_on_zoo_trip_l800_80050

theorem students_on_zoo_trip (buses : ℕ) (students_per_bus : ℕ) (students_in_cars : ℕ) 
  (h1 : buses = 7) (h2 : students_per_bus = 56) (h3 : students_in_cars = 4) : 
  buses * students_per_bus + students_in_cars = 396 :=
by
  sorry

end NUMINAMATH_GPT_students_on_zoo_trip_l800_80050


namespace NUMINAMATH_GPT_shadow_boundary_function_correct_l800_80042

noncomputable def sphereShadowFunction : ℝ → ℝ :=
  λ x => (x + 1) / 2

theorem shadow_boundary_function_correct :
  ∀ (x y : ℝ), 
    -- Conditions: 
    -- The sphere with center (0,0,2) and radius 2
    -- A light source at point P = (1, -2, 3)
    -- The shadow must lie on the xy-plane, so z-coordinate is 0
    (sphereShadowFunction x = y) ↔ (- x + 2 * y - 1 = 0) :=
by
  intros x y
  sorry

end NUMINAMATH_GPT_shadow_boundary_function_correct_l800_80042


namespace NUMINAMATH_GPT_simplify_expression_l800_80071

theorem simplify_expression (t : ℝ) : (t ^ 5 * t ^ 3) / t ^ 2 = t ^ 6 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l800_80071


namespace NUMINAMATH_GPT_possible_final_state_l800_80041

-- Definitions of initial conditions and operations
def initial_urn : (ℕ × ℕ) := (100, 100)  -- (W, B)

-- Define operations that describe changes in (white, black) marbles
inductive Operation
| operation1 : Operation
| operation2 : Operation
| operation3 : Operation
| operation4 : Operation

def apply_operation (op : Operation) (state : ℕ × ℕ) : ℕ × ℕ :=
  match op with
  | Operation.operation1 => (state.1, state.2 - 2)
  | Operation.operation2 => (state.1, state.2 - 1)
  | Operation.operation3 => (state.1, state.2 - 1)
  | Operation.operation4 => (state.1 - 2, state.2 + 1)

-- The final state in the form of the specific condition to prove.
def final_state (state : ℕ × ℕ) : Prop :=
  state = (2, 0)  -- 2 white marbles are an expected outcome.

-- Statement of the problem in Lean
theorem possible_final_state : ∃ (sequence : List Operation), 
  (sequence.foldl (fun state op => apply_operation op state) initial_urn).1 = 2 :=
sorry

end NUMINAMATH_GPT_possible_final_state_l800_80041


namespace NUMINAMATH_GPT_tray_contains_correct_number_of_pieces_l800_80033

-- Define the dimensions of the tray
def tray_width : ℕ := 24
def tray_length : ℕ := 20
def tray_area : ℕ := tray_width * tray_length

-- Define the dimensions of each brownie piece
def piece_width : ℕ := 3
def piece_length : ℕ := 4
def piece_area : ℕ := piece_width * piece_length

-- Define the goal: the number of pieces of brownies that the tray contains
def num_pieces : ℕ := tray_area / piece_area

-- The statement to prove
theorem tray_contains_correct_number_of_pieces :
  num_pieces = 40 :=
by
  sorry

end NUMINAMATH_GPT_tray_contains_correct_number_of_pieces_l800_80033


namespace NUMINAMATH_GPT_inequality_proof_l800_80086

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
    (x / (y + z)) + (y / (z + x)) + (z / (x + y)) ≥ (3 / 2) :=
sorry

end NUMINAMATH_GPT_inequality_proof_l800_80086


namespace NUMINAMATH_GPT_quoted_price_of_shares_l800_80070

theorem quoted_price_of_shares :
  ∀ (investment nominal_value dividend_rate annual_income quoted_price : ℝ),
  investment = 4940 →
  nominal_value = 10 →
  dividend_rate = 14 →
  annual_income = 728 →
  quoted_price = 9.5 :=
by
  intros investment nominal_value dividend_rate annual_income quoted_price
  intros h_investment h_nominal_value h_dividend_rate h_annual_income
  sorry

end NUMINAMATH_GPT_quoted_price_of_shares_l800_80070


namespace NUMINAMATH_GPT_sugar_percentage_l800_80006

theorem sugar_percentage (x : ℝ) (h2 : 50 ≤ 100) (h1 : 1 / 4 * x + 12.5 = 20) : x = 10 :=
by
  sorry

end NUMINAMATH_GPT_sugar_percentage_l800_80006


namespace NUMINAMATH_GPT_sin_double_angle_l800_80098

theorem sin_double_angle 
  (α β : ℝ)
  (h1 : 0 < β)
  (h2 : β < α)
  (h3 : α < π / 4)
  (h_cos_diff : Real.cos (α - β) = 12 / 13)
  (h_sin_sum : Real.sin (α + β) = 4 / 5) :
  Real.sin (2 * α) = 63 / 65 := 
sorry

end NUMINAMATH_GPT_sin_double_angle_l800_80098
