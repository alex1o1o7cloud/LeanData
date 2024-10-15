import Mathlib

namespace NUMINAMATH_GPT_solve_equation_real_l649_64950

theorem solve_equation_real (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 2) (h3 : x ≠ 4) :
  (x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 5) * (x - 2) * (x - 1) / ((x - 4) * (x - 2) * (x - 1)) = 1 ↔
  x = (9 + Real.sqrt 5) / 2 ∨ x = (9 - Real.sqrt 5) / 2 :=
by  
  sorry

end NUMINAMATH_GPT_solve_equation_real_l649_64950


namespace NUMINAMATH_GPT_greatest_possible_x_for_equation_l649_64952

theorem greatest_possible_x_for_equation :
  ∃ x, (x = (9 : ℝ) / 5) ∧ 
  ((5 * x - 20) / (4 * x - 5))^2 + ((5 * x - 20) / (4 * x - 5)) = 20 := by
  sorry

end NUMINAMATH_GPT_greatest_possible_x_for_equation_l649_64952


namespace NUMINAMATH_GPT_pentagon_diagonals_l649_64979

def number_of_sides_pentagon : ℕ := 5
def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem pentagon_diagonals : number_of_diagonals number_of_sides_pentagon = 5 := by
  sorry

end NUMINAMATH_GPT_pentagon_diagonals_l649_64979


namespace NUMINAMATH_GPT_parabola_line_intersection_l649_64928

/-- 
Given the parabola y^2 = -x and the line l: y = k(x + 1) intersect at points A and B,
(Ⅰ) Find the range of values for k;
(Ⅱ) Let O be the vertex of the parabola, prove that OA ⟂ OB.
-/
theorem parabola_line_intersection (k : ℝ) (A B : ℝ × ℝ)
  (hA : A.2 ^ 2 = -A.1) (hB : B.2 ^ 2 = -B.1)
  (hlineA : A.2 = k * (A.1 + 1)) (hlineB : B.2 = k * (B.1 + 1)) :
  (k ≠ 0) ∧ ((A.2 * B.2 = -1) → A.1 * B.1 * (A.2 * B.2) = -1) :=
by
  sorry

end NUMINAMATH_GPT_parabola_line_intersection_l649_64928


namespace NUMINAMATH_GPT_angle_ABC_measure_l649_64924

theorem angle_ABC_measure
  (angle_CBD : ℝ)
  (angle_sum_around_B : ℝ)
  (angle_ABD : ℝ)
  (h1 : angle_CBD = 90)
  (h2 : angle_sum_around_B = 200)
  (h3 : angle_ABD = 60) :
  ∃ angle_ABC : ℝ, angle_ABC = 50 :=
by
  sorry

end NUMINAMATH_GPT_angle_ABC_measure_l649_64924


namespace NUMINAMATH_GPT_mary_saw_total_snakes_l649_64961

theorem mary_saw_total_snakes :
  let breedingBalls := 3
  let snakesPerBall := 8
  let pairsOfSnakes := 6
  let snakesPerPair := 2
  let totalSnakes := breedingBalls * snakesPerBall + pairsOfSnakes * snakesPerPair
  totalSnakes = 36 :=
by
  /- Definitions -/ 
  let breedingBalls := 3
  let snakesPerBall := 8
  let pairsOfSnakes := 6
  let snakesPerPair := 2
  let totalSnakes := breedingBalls * snakesPerBall + pairsOfSnakes * snakesPerPair
  /- Main proof statement -/
  show totalSnakes = 36
  sorry

end NUMINAMATH_GPT_mary_saw_total_snakes_l649_64961


namespace NUMINAMATH_GPT_equal_vectors_implies_collinear_l649_64966

-- Definitions for vectors and their properties
variables {V : Type*} [AddCommGroup V] [Module ℝ V]

def collinear (u v : V) : Prop := ∃ (a : ℝ), v = a • u 

def equal_vectors (u v : V) : Prop := u = v

theorem equal_vectors_implies_collinear (u v : V)
  (h : equal_vectors u v) : collinear u v :=
by sorry

end NUMINAMATH_GPT_equal_vectors_implies_collinear_l649_64966


namespace NUMINAMATH_GPT_abs_z_bounds_l649_64991

open Complex

theorem abs_z_bounds (z : ℂ) (h : abs (z + 1/z) = 1) : 
  (Real.sqrt 5 - 1) / 2 ≤ abs z ∧ abs z ≤ (Real.sqrt 5 + 1) / 2 := 
sorry

end NUMINAMATH_GPT_abs_z_bounds_l649_64991


namespace NUMINAMATH_GPT_carol_blocks_l649_64997

theorem carol_blocks (initial_blocks lost_blocks final_blocks : ℕ) 
  (h_initial : initial_blocks = 42) 
  (h_lost : lost_blocks = 25) : 
  final_blocks = initial_blocks - lost_blocks → final_blocks = 17 := by
  sorry

end NUMINAMATH_GPT_carol_blocks_l649_64997


namespace NUMINAMATH_GPT_directrix_of_parabola_l649_64993

-- Define the equation of the parabola and what we need to prove
def parabola_equation (x : ℝ) : ℝ := 2 * x^2 + 6

-- Theorem stating the directrix of the given parabola
theorem directrix_of_parabola :
  ∀ x : ℝ, y = parabola_equation x → y = 47 / 8 := 
by
  sorry

end NUMINAMATH_GPT_directrix_of_parabola_l649_64993


namespace NUMINAMATH_GPT_distance_after_one_hour_l649_64933

-- Definitions representing the problem's conditions
def initial_distance : ℕ := 20
def speed_athos : ℕ := 4
def speed_aramis : ℕ := 5

-- The goal is to prove that the possible distances after one hour are among the specified values
theorem distance_after_one_hour :
  ∃ d : ℕ, d = 11 ∨ d = 29 ∨ d = 21 ∨ d = 19 :=
sorry -- proof not required as per the instructions

end NUMINAMATH_GPT_distance_after_one_hour_l649_64933


namespace NUMINAMATH_GPT_more_customers_after_lunch_rush_l649_64975

-- Definitions for conditions
def initial_customers : ℝ := 29.0
def added_customers : ℝ := 20.0
def total_customers : ℝ := 83.0

-- The number of additional customers that came in after the lunch rush
def additional_customers (initial additional total : ℝ) : ℝ :=
  total - (initial + additional)

-- Statement to prove
theorem more_customers_after_lunch_rush :
  additional_customers initial_customers added_customers total_customers = 34.0 :=
by
  sorry

end NUMINAMATH_GPT_more_customers_after_lunch_rush_l649_64975


namespace NUMINAMATH_GPT_limit_of_sequence_l649_64982

theorem limit_of_sequence (a_n : ℕ → ℝ) (a : ℝ) :
  (∀ n : ℕ, a_n n = (2 * (n ^ 3)) / ((n ^ 3) - 2)) →
  a = 2 →
  (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a_n n - a| < ε) :=
by
  intros h1 h2 ε hε
  sorry

end NUMINAMATH_GPT_limit_of_sequence_l649_64982


namespace NUMINAMATH_GPT_factor_sum_l649_64923

variable (x y : ℝ)

theorem factor_sum :
  let a := 1
  let b := -2
  let c := 1
  let d := 2
  let e := 4
  let f := 1
  let g := 2
  let h := 1
  let j := -2
  let k := 4
  (27 * x^9 - 512 * y^9) = ((a * x + b * y) * (c * x^3 + d * x * y^2 + e * y^3) * 
  (f * x + g * y) * (h * x^3 + j * x * y^2 + k * y^3)) → 
  (a + b + c + d + e + f + g + h + j + k = 12) :=
by
  sorry

end NUMINAMATH_GPT_factor_sum_l649_64923


namespace NUMINAMATH_GPT_geometric_sequence_b_value_l649_64967

theorem geometric_sequence_b_value (b : ℝ) 
  (h1 : ∃ r : ℝ, 30 * r = b ∧ b * r = 9 / 4)
  (h2 : b > 0) : b = 3 * Real.sqrt 30 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_b_value_l649_64967


namespace NUMINAMATH_GPT_number_of_rectangles_required_l649_64902

theorem number_of_rectangles_required
  (width : ℝ) (area : ℝ) (total_length : ℝ) (length : ℝ)
  (H1 : width = 42) (H2 : area = 1638) (H3 : total_length = 390) (H4 : length = area / width)
  : (total_length / length) = 10 := 
sorry

end NUMINAMATH_GPT_number_of_rectangles_required_l649_64902


namespace NUMINAMATH_GPT_find_A_l649_64920

variable {a b : ℝ}

theorem find_A (h : (5 * a + 3 * b)^2 = (5 * a - 3 * b)^2 + A) : A = 60 * a * b :=
sorry

end NUMINAMATH_GPT_find_A_l649_64920


namespace NUMINAMATH_GPT_power_function_k_values_l649_64949

theorem power_function_k_values (k : ℝ) :
  (∃ (a : ℝ), (k^2 - k - 5) = a ∧ (∀ x : ℝ, (k^2 - k - 5) * x^3 = a * x^3)) →
  (k = 3 ∨ k = -2) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_power_function_k_values_l649_64949


namespace NUMINAMATH_GPT_verify_quadratic_solution_l649_64913

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def quadratic_roots : Prop :=
  ∃ (p q : ℕ) (x1 x2 : ℤ), is_prime p ∧ is_prime q ∧ 
  (x1 + x2 = -(p : ℤ)) ∧ (x1 * x2 = (3 * q : ℤ)) ∧ x1 < 0 ∧ x2 < 0 ∧ 
  ((p = 7 ∧ q = 2) ∨ (p = 5 ∧ q = 2))

theorem verify_quadratic_solution : quadratic_roots :=
  by {
    sorry
  }

end NUMINAMATH_GPT_verify_quadratic_solution_l649_64913


namespace NUMINAMATH_GPT_probability_of_selecting_cooking_l649_64955

def total_courses : ℕ := 4
def favorable_outcomes : ℕ := 1

theorem probability_of_selecting_cooking : (favorable_outcomes : ℚ) / total_courses = 1 / 4 := 
by 
  sorry

end NUMINAMATH_GPT_probability_of_selecting_cooking_l649_64955


namespace NUMINAMATH_GPT_find_m_minus_n_l649_64921

noncomputable def m_abs := 4
noncomputable def n_abs := 6

theorem find_m_minus_n (m n : ℝ) (h1 : |m| = m_abs) (h2 : |n| = n_abs) (h3 : |m + n| = m + n) : m - n = -2 ∨ m - n = -10 :=
sorry

end NUMINAMATH_GPT_find_m_minus_n_l649_64921


namespace NUMINAMATH_GPT_fries_remaining_time_l649_64985

def recommendedTime : ℕ := 5 * 60
def timeInOven : ℕ := 45
def remainingTime : ℕ := recommendedTime - timeInOven

theorem fries_remaining_time : remainingTime = 255 :=
by
  sorry

end NUMINAMATH_GPT_fries_remaining_time_l649_64985


namespace NUMINAMATH_GPT_kayak_rental_cost_l649_64984

theorem kayak_rental_cost (F : ℝ) (C : ℝ) (h1 : ∀ t : ℝ, C = F + 5 * t)
  (h2 : C = 30) : C = 45 :=
sorry

end NUMINAMATH_GPT_kayak_rental_cost_l649_64984


namespace NUMINAMATH_GPT_race_length_l649_64934

theorem race_length (A_time : ℕ) (diff_distance diff_time : ℕ) (A_time_eq : A_time = 380)
  (diff_distance_eq : diff_distance = 50) (diff_time_eq : diff_time = 20) :
  let B_speed := diff_distance / diff_time
  let B_time := A_time + diff_time
  let race_length := B_speed * B_time
  race_length = 1000 := 
by
  sorry

end NUMINAMATH_GPT_race_length_l649_64934


namespace NUMINAMATH_GPT_distance_between_A_and_B_l649_64960

theorem distance_between_A_and_B 
  (d : ℝ)
  (h1 : ∀ (t : ℝ), (t = 2 * (t / 2)) → t = 200) 
  (h2 : ∀ (t : ℝ), 100 = d - (t / 2 + 50))
  (h3 : ∀ (t : ℝ), d = 2 * (d - 60)): 
  d = 300 :=
sorry

end NUMINAMATH_GPT_distance_between_A_and_B_l649_64960


namespace NUMINAMATH_GPT_min_x_plus_4y_min_value_l649_64973

noncomputable def min_x_plus_4y (x y: ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1/x + 1/(2 * y) = 1) : ℝ :=
  x + 4 * y

theorem min_x_plus_4y_min_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1/x + 1/(2 * y) = 1) :
  min_x_plus_4y x y hx hy h = 3 + 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_min_x_plus_4y_min_value_l649_64973


namespace NUMINAMATH_GPT_cos_of_sum_eq_one_l649_64906

theorem cos_of_sum_eq_one
  (x y : ℝ)
  (a : ℝ)
  (h1 : x ∈ Set.Icc (-Real.pi / 4) (Real.pi / 4))
  (h2 : y ∈ Set.Icc (-Real.pi / 4) (Real.pi / 4))
  (h3 : x^3 + Real.sin x - 2 * a = 0)
  (h4 : 4 * y^3 + Real.sin y * Real.cos y + a = 0) :
  Real.cos (x + 2 * y) = 1 := 
by
  sorry

end NUMINAMATH_GPT_cos_of_sum_eq_one_l649_64906


namespace NUMINAMATH_GPT_seventh_numbers_sum_l649_64916

def first_row_seq (n : ℕ) : ℕ := n^2 + n - 1

def second_row_seq (n : ℕ) : ℕ := n * (n + 1) / 2

theorem seventh_numbers_sum :
  first_row_seq 7 + second_row_seq 7 = 83 :=
by
  -- Skipping the proof
  sorry

end NUMINAMATH_GPT_seventh_numbers_sum_l649_64916


namespace NUMINAMATH_GPT_remainder_when_divided_by_x_plus_2_l649_64981

-- Define the polynomial q(x)
def q (M N D x : ℝ) : ℝ := M * x^4 + N * x^2 + D * x - 5

-- Define the given conditions
def cond1 (M N D : ℝ) : Prop := q M N D 2 = 15

-- The theorem statement we want to prove
theorem remainder_when_divided_by_x_plus_2 (M N D : ℝ) (h1 : cond1 M N D) : q M N D (-2) = 15 :=
sorry

end NUMINAMATH_GPT_remainder_when_divided_by_x_plus_2_l649_64981


namespace NUMINAMATH_GPT_length_of_bridge_l649_64947

theorem length_of_bridge
    (speed_kmh : Real)
    (time_minutes : Real)
    (speed_cond : speed_kmh = 5)
    (time_cond : time_minutes = 15) :
    let speed_mmin := speed_kmh * 1000 / 60
    let distance_m := speed_mmin * time_minutes
    distance_m = 1250 :=
by
    sorry

end NUMINAMATH_GPT_length_of_bridge_l649_64947


namespace NUMINAMATH_GPT_twenty_seven_cubes_volume_l649_64936

def volume_surface_relation (x V S : ℝ) : Prop :=
  V = x^3 ∧ S = 6 * x^2 ∧ V + S = (4 / 3) * (12 * x)

theorem twenty_seven_cubes_volume (x : ℝ) (hx : volume_surface_relation x (x^3) (6 * x^2)) : 
  27 * (x^3) = 216 :=
by
  sorry

end NUMINAMATH_GPT_twenty_seven_cubes_volume_l649_64936


namespace NUMINAMATH_GPT_find_m_l649_64965

open Set

def A : Set ℕ := {1, 3, 5}
def B (m : ℕ) : Set ℕ := {1, m}
def C (m : ℕ) : Set ℕ := {1, m}

theorem find_m (m : ℕ) (h : A ∩ B m = C m) : m = 3 ∨ m = 5 :=
sorry

end NUMINAMATH_GPT_find_m_l649_64965


namespace NUMINAMATH_GPT_contradiction_assumption_l649_64937

-- Proposition P: "Among a, b, c, d, at least one is negative"
def P (a b c d : ℝ) : Prop :=
  a < 0 ∨ b < 0 ∨ c < 0 ∨ d < 0

-- Correct assumption when using contradiction: all are non-negative
def notP (a b c d : ℝ) : Prop :=
  a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0

-- Proof problem statement: assuming notP leads to contradiction to prove P
theorem contradiction_assumption (a b c d : ℝ) (h : ¬ P a b c d) : notP a b c d :=
by
  sorry

end NUMINAMATH_GPT_contradiction_assumption_l649_64937


namespace NUMINAMATH_GPT_spaceship_travel_distance_l649_64918

-- Define each leg of the journey
def distance1 := 0.5
def distance2 := 0.1
def distance3 := 0.1

-- Define the total distance traveled
def total_distance := distance1 + distance2 + distance3

-- The statement to prove
theorem spaceship_travel_distance : total_distance = 0.7 := sorry

end NUMINAMATH_GPT_spaceship_travel_distance_l649_64918


namespace NUMINAMATH_GPT_ratio_of_number_halving_l649_64929

theorem ratio_of_number_halving (x y : ℕ) (h1 : y = x / 2) (h2 : y = 9) : x / y = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_number_halving_l649_64929


namespace NUMINAMATH_GPT_partitions_distinct_parts_eq_odd_parts_l649_64953

def num_partitions_into_distinct_parts (n : ℕ) : ℕ := sorry
def num_partitions_into_odd_parts (n : ℕ) : ℕ := sorry

theorem partitions_distinct_parts_eq_odd_parts (n : ℕ) :
  num_partitions_into_distinct_parts n = num_partitions_into_odd_parts n :=
  sorry

end NUMINAMATH_GPT_partitions_distinct_parts_eq_odd_parts_l649_64953


namespace NUMINAMATH_GPT_problem_I_problem_II_l649_64992

open Set

variable (a x : ℝ)

def p : Prop := ∀ x ∈ Icc (1 : ℝ) 2, x^2 - a ≥ 0
def q : Prop := ∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0

theorem problem_I (hp : p a) : a ≤ 1 :=
  sorry

theorem problem_II (hpq : ¬ (p a ∧ q a)) : a ∈ Ioo (-2 : ℝ) (1 : ℝ) ∪ Ioi 1 :=
  sorry

end NUMINAMATH_GPT_problem_I_problem_II_l649_64992


namespace NUMINAMATH_GPT_total_green_and_yellow_peaches_in_basket_l649_64968

def num_red_peaches := 5
def num_yellow_peaches := 14
def num_green_peaches := 6

theorem total_green_and_yellow_peaches_in_basket :
  num_yellow_peaches + num_green_peaches = 20 :=
by
  sorry

end NUMINAMATH_GPT_total_green_and_yellow_peaches_in_basket_l649_64968


namespace NUMINAMATH_GPT_price_reduction_equation_l649_64941

variable (x : ℝ)

theorem price_reduction_equation :
    (58 * (1 - x)^2 = 43) :=
sorry

end NUMINAMATH_GPT_price_reduction_equation_l649_64941


namespace NUMINAMATH_GPT_curve_not_parabola_l649_64935

theorem curve_not_parabola (k : ℝ) : ¬ (∃ (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ c = 1 ∧ a * x^2 + b * y = c) :=
sorry

end NUMINAMATH_GPT_curve_not_parabola_l649_64935


namespace NUMINAMATH_GPT_russom_greatest_number_of_envelopes_l649_64943

theorem russom_greatest_number_of_envelopes :
  ∃ n, n > 0 ∧ 18 % n = 0 ∧ 12 % n = 0 ∧ ∀ m, m > 0 ∧ 18 % m = 0 ∧ 12 % m = 0 → m ≤ n :=
sorry

end NUMINAMATH_GPT_russom_greatest_number_of_envelopes_l649_64943


namespace NUMINAMATH_GPT_contractor_engaged_days_l649_64925

/-
  Prove that the contractor was engaged for 30 days given that:
  1. The contractor receives Rs. 25 for each day he works.
  2. The contractor is fined Rs. 7.50 for each day he is absent.
  3. The contractor gets Rs. 425 in total.
  4. The contractor was absent for 10 days.
-/

theorem contractor_engaged_days
  (wage_per_day : ℕ) (fine_per_absent_day : ℕ) (total_amount_received : ℕ) (absent_days : ℕ) 
  (total_days_engaged : ℕ) :
  wage_per_day = 25 →
  fine_per_absent_day = 7500 / 1000 → -- Rs. 7.50 = 7500 / 1000 (to avoid float, using integers)
  total_amount_received = 425 →
  absent_days = 10 →
  total_days_engaged = (total_amount_received + absent_days * fine_per_absent_day) / wage_per_day + absent_days →
  total_days_engaged = 30 :=
by
  sorry

end NUMINAMATH_GPT_contractor_engaged_days_l649_64925


namespace NUMINAMATH_GPT_gerbils_left_l649_64957

theorem gerbils_left (initial count sold : ℕ) (h_initial : count = 85) (h_sold : sold = 69) : 
  count - sold = 16 := 
by 
  sorry

end NUMINAMATH_GPT_gerbils_left_l649_64957


namespace NUMINAMATH_GPT_sequence_statements_correct_l649_64958

theorem sequence_statements_correct (S : ℕ → ℝ) (a : ℕ → ℝ) (T : ℕ → ℝ) 
(h_S_nonzero : ∀ n, n > 0 → S n ≠ 0)
(h_S_T_relation : ∀ n, n > 0 → S n + T n = S n * T n) :
  (a 1 = 2) ∧ (∀ n, n > 0 → T n - T (n - 1) = 1) ∧ (∀ n, n > 0 → S n = (n + 1) / n) :=
by
  sorry

end NUMINAMATH_GPT_sequence_statements_correct_l649_64958


namespace NUMINAMATH_GPT_average_speed_is_correct_l649_64969

namespace CyclistTrip

-- Define the trip parameters
def distance_north := 10 -- kilometers
def speed_north := 15 -- kilometers per hour
def rest_time := 10 / 60 -- hours
def distance_south := 10 -- kilometers
def speed_south := 20 -- kilometers per hour

-- The total trip distance
def total_distance := distance_north + distance_south -- kilometers

-- Calculate the time for each segment
def time_north := distance_north / speed_north -- hours
def time_south := distance_south / speed_south -- hours

-- Total time for the trip
def total_time := time_north + rest_time + time_south -- hours

-- Calculate the average speed
def average_speed := total_distance / total_time -- kilometers per hour

theorem average_speed_is_correct : average_speed = 15 := by
  sorry

end CyclistTrip

end NUMINAMATH_GPT_average_speed_is_correct_l649_64969


namespace NUMINAMATH_GPT_part1_l649_64917

theorem part1 : 2 * Real.tan (60 * Real.pi / 180) * Real.cos (30 * Real.pi / 180) - (Real.sin (45 * Real.pi / 180)) ^ 2 = 5 / 2 := 
sorry

end NUMINAMATH_GPT_part1_l649_64917


namespace NUMINAMATH_GPT_probability_rain_weekend_l649_64931

theorem probability_rain_weekend :
  let p_rain_saturday := 0.30
  let p_rain_sunday := 0.60
  let p_rain_sunday_given_rain_saturday := 0.40
  let p_no_rain_saturday := 1 - p_rain_saturday
  let p_no_rain_sunday_given_no_rain_saturday := 1 - p_rain_sunday
  let p_no_rain_both_days := p_no_rain_saturday * p_no_rain_sunday_given_no_rain_saturday
  let p_rain_sunday_given_rain_saturday := 1 - p_rain_sunday_given_rain_saturday
  let p_no_rain_sunday_given_rain_saturday := p_rain_saturday * p_rain_sunday_given_rain_saturday
  let p_no_rain_all_scenarios := p_no_rain_both_days + p_no_rain_sunday_given_rain_saturday
  let p_rain_weekend := 1 - p_no_rain_all_scenarios
  p_rain_weekend = 0.54 :=
sorry

end NUMINAMATH_GPT_probability_rain_weekend_l649_64931


namespace NUMINAMATH_GPT_ratio_of_a_to_b_and_c_l649_64971

theorem ratio_of_a_to_b_and_c (A B C : ℝ) (h1 : A = 160) (h2 : A + B + C = 400) (h3 : B = (2/3) * (A + C)) :
  A / (B + C) = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_a_to_b_and_c_l649_64971


namespace NUMINAMATH_GPT_largest_possible_perimeter_l649_64976

theorem largest_possible_perimeter
  (a b c : ℕ)
  (h1 : a > 2 ∧ b > 2 ∧ c > 2)  -- sides are greater than 2
  (h2 : a = c ∨ b = c ∨ a = b)  -- at least two polygons are congruent
  (h3 : (a - 2) * (b - 2) = 8 ∨ (a - 2) * (c - 2) = 8 ∨ (b - 2) * (c - 2) = 8)  -- possible factorizations
  (h4 : (a - 2) + (b - 2) + (c - 2) = 12)  -- sum of interior angles at A is 360 degrees
  : 2 * a + 2 * b + 2 * c - 6 ≤ 21 :=
sorry

end NUMINAMATH_GPT_largest_possible_perimeter_l649_64976


namespace NUMINAMATH_GPT_num_arrangement_options_l649_64977

def competition_events := ["kicking shuttlecocks", "jumping rope", "tug-of-war", "pushing the train", "multi-person multi-foot"]

def is_valid_arrangement (arrangement : List String) : Prop :=
  arrangement.length = 5 ∧
  arrangement.getLast? = some "tug-of-war" ∧
  arrangement.get? 0 ≠ some "multi-person multi-foot"

noncomputable def count_valid_arrangements : ℕ :=
  let positions := ["kicking shuttlecocks", "jumping rope", "pushing the train"]
  3 * positions.permutations.length

theorem num_arrangement_options : count_valid_arrangements = 18 :=
by
  sorry

end NUMINAMATH_GPT_num_arrangement_options_l649_64977


namespace NUMINAMATH_GPT_feet_in_mile_l649_64922

theorem feet_in_mile (d t : ℝ) (speed_mph : ℝ) (speed_fps : ℝ) (miles_to_feet : ℝ) (hours_to_seconds : ℝ) :
  d = 200 → t = 4 → speed_mph = 34.09 → miles_to_feet = 5280 → hours_to_seconds = 3600 → 
  speed_fps = d / t → speed_fps = speed_mph * miles_to_feet / hours_to_seconds → 
  miles_to_feet = 5280 :=
by
  intros hd ht hspeed_mph hmiles_to_feet hhours_to_seconds hspeed_fps_eq hconversion
  -- You can add the proof steps here.
  sorry

end NUMINAMATH_GPT_feet_in_mile_l649_64922


namespace NUMINAMATH_GPT_mat_radius_increase_l649_64970

theorem mat_radius_increase (C1 C2 : ℝ) (h1 : C1 = 40) (h2 : C2 = 50) :
  let r1 := C1 / (2 * Real.pi)
  let r2 := C2 / (2 * Real.pi)
  (r2 - r1) = 5 / Real.pi := by
  sorry

end NUMINAMATH_GPT_mat_radius_increase_l649_64970


namespace NUMINAMATH_GPT_deductive_vs_inductive_l649_64942

def is_inductive_reasoning (stmt : String) : Prop :=
  match stmt with
  | "C" => True
  | _ => False

theorem deductive_vs_inductive (A B C D : String) 
  (hA : A = "All trigonometric functions are periodic functions, sin(x) is a trigonometric function, therefore sin(x) is a periodic function.")
  (hB : B = "All odd numbers cannot be divided by 2, 525 is an odd number, therefore 525 cannot be divided by 2.")
  (hC : C = "From 1=1^2, 1+3=2^2, 1+3+5=3^2, it follows that 1+3+…+(2n-1)=n^2 (n ∈ ℕ*)")
  (hD : D = "If two lines are parallel, the corresponding angles are equal. If ∠A and ∠B are corresponding angles of two parallel lines, then ∠A = ∠B.") :
  is_inductive_reasoning C :=
by
  sorry

end NUMINAMATH_GPT_deductive_vs_inductive_l649_64942


namespace NUMINAMATH_GPT_parabola_hyperbola_focus_l649_64915

theorem parabola_hyperbola_focus (p : ℝ) (h : p > 0) :
  (∀ x y : ℝ, (y ^ 2 = 2 * p * x) ∧ (x ^ 2 / 4 - y ^ 2 / 5 = 1) → p = 6) :=
by
  sorry

end NUMINAMATH_GPT_parabola_hyperbola_focus_l649_64915


namespace NUMINAMATH_GPT_eighth_binomial_term_l649_64980

theorem eighth_binomial_term :
  let n := 10
  let a := 2 * x
  let b := 1
  let k := 7
  (Nat.choose n k) * (a ^ k) * (b ^ (n - k)) = 960 * (x ^ 3) := by
  sorry

end NUMINAMATH_GPT_eighth_binomial_term_l649_64980


namespace NUMINAMATH_GPT_inequality_solution_set_l649_64944

theorem inequality_solution_set (a : ℝ) : (-16 < a ∧ a ≤ 0) ↔ (∀ x : ℝ, a * x^2 + a * x - 4 < 0) :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_set_l649_64944


namespace NUMINAMATH_GPT_abes_age_after_x_years_l649_64946

-- Given conditions
def A : ℕ := 28
def sum_condition (x : ℕ) : Prop := (A + (A - x) = 35)

-- Proof statement
theorem abes_age_after_x_years
  (x : ℕ)
  (h : sum_condition x) :
  (A + x = 49) :=
  sorry

end NUMINAMATH_GPT_abes_age_after_x_years_l649_64946


namespace NUMINAMATH_GPT_proof_age_gladys_l649_64994

-- Definitions of ages
def age_gladys : ℕ := 30
def age_lucas : ℕ := 5
def age_billy : ℕ := 10

-- Conditions
def condition1 : Prop := age_gladys = 2 * (age_billy + age_lucas)
def condition2 : Prop := age_gladys = 3 * age_billy
def condition3 : Prop := age_lucas + 3 = 8

-- Theorem to prove the correct age of Gladys
theorem proof_age_gladys (G L B : ℕ)
  (h1 : G = 2 * (B + L))
  (h2 : G = 3 * B)
  (h3 : L + 3 = 8) :
  G = 30 :=
sorry

end NUMINAMATH_GPT_proof_age_gladys_l649_64994


namespace NUMINAMATH_GPT_point_on_angle_bisector_l649_64940

theorem point_on_angle_bisector (a b : ℝ) (h : (a, b) = (b, a)) : a = b ∨ a = -b := 
by
  sorry

end NUMINAMATH_GPT_point_on_angle_bisector_l649_64940


namespace NUMINAMATH_GPT_solve_for_x_l649_64954

theorem solve_for_x (x : ℝ) (h : (4/7) * (2/5) * x = 8) : x = 35 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l649_64954


namespace NUMINAMATH_GPT_zarnin_staffing_l649_64956

theorem zarnin_staffing (n total unsuitable : ℕ) (unsuitable_factor : ℕ) (job_openings : ℕ)
  (h1 : total = 30) 
  (h2 : unsuitable_factor = 2 / 3) 
  (h3 : unsuitable = unsuitable_factor * total) 
  (h4 : n = total - unsuitable)
  (h5 : job_openings = 5) :
  (n - 0) * (n - 1) * (n - 2) * (n - 3) * (n - 4) = 30240 := by
    sorry

end NUMINAMATH_GPT_zarnin_staffing_l649_64956


namespace NUMINAMATH_GPT_remaining_amount_is_16_l649_64983

-- Define initial amount of money Sam has.
def initial_amount : ℕ := 79

-- Define cost per book.
def cost_per_book : ℕ := 7

-- Define the number of books.
def number_of_books : ℕ := 9

-- Define the total cost of books.
def total_cost : ℕ := cost_per_book * number_of_books

-- Define the remaining amount of money after buying the books.
def remaining_amount : ℕ := initial_amount - total_cost

-- Prove the remaining amount is 16 dollars.
theorem remaining_amount_is_16 : remaining_amount = 16 := by
  rfl

end NUMINAMATH_GPT_remaining_amount_is_16_l649_64983


namespace NUMINAMATH_GPT_number_of_insects_l649_64927

theorem number_of_insects (total_legs : ℕ) (legs_per_insect : ℕ) (h1 : total_legs = 48) (h2 : legs_per_insect = 6) : (total_legs / legs_per_insect) = 8 := by
  sorry

end NUMINAMATH_GPT_number_of_insects_l649_64927


namespace NUMINAMATH_GPT_length_of_train_l649_64999

theorem length_of_train (V L : ℝ) (h1 : L = V * 18) (h2 : L + 250 = V * 33) : L = 300 :=
by
  sorry

end NUMINAMATH_GPT_length_of_train_l649_64999


namespace NUMINAMATH_GPT_hemisphere_surface_area_l649_64900

theorem hemisphere_surface_area (r : ℝ) (h : r = 10) : 
  (4 * Real.pi * r^2) / 2 + (Real.pi * r^2) = 300 * Real.pi := by
  sorry

end NUMINAMATH_GPT_hemisphere_surface_area_l649_64900


namespace NUMINAMATH_GPT_ellipse_foci_distance_l649_64926

theorem ellipse_foci_distance 
  (a b : ℝ) 
  (h_a : a = 8) 
  (h_b : b = 3) : 
  2 * (Real.sqrt (a^2 - b^2)) = 2 * Real.sqrt 55 := 
by
  rw [h_a, h_b]
  sorry

end NUMINAMATH_GPT_ellipse_foci_distance_l649_64926


namespace NUMINAMATH_GPT_caitlin_bracelets_l649_64903

-- Define the conditions
def twice_as_many_small_beads (x y : Nat) : Prop :=
  y = 2 * x

def total_large_small_beads (total large small : Nat) : Prop :=
  total = large + small ∧ large = small

def bracelet_beads (large_beads_per_bracelet small_beads_per_bracelet large_per_bracelet : Nat) : Prop :=
  small_beads_per_bracelet = 2 * large_per_bracelet

def total_bracelets (total_large_beads large_per_bracelet bracelets : Nat) : Prop :=
  bracelets = total_large_beads / large_per_bracelet

-- The theorem to be proved
theorem caitlin_bracelets (total_beads large_per_bracelet small_per_bracelet : Nat) (bracelets : Nat) :
    total_beads = 528 ∧
    large_per_bracelet = 12 ∧
    twice_as_many_small_beads large_per_bracelet small_per_bracelet ∧
    total_large_small_beads total_beads 264 264 ∧
    bracelet_beads large_per_bracelet small_per_bracelet 12 ∧
    total_bracelets 264 12 bracelets
  → bracelets = 22 := by
  sorry

end NUMINAMATH_GPT_caitlin_bracelets_l649_64903


namespace NUMINAMATH_GPT_candy_days_l649_64996

theorem candy_days (neighbor_candy older_sister_candy candy_per_day : ℝ) 
  (h1 : neighbor_candy = 11.0) 
  (h2 : older_sister_candy = 5.0) 
  (h3 : candy_per_day = 8.0) : 
  ((neighbor_candy + older_sister_candy) / candy_per_day) = 2.0 := 
by 
  sorry

end NUMINAMATH_GPT_candy_days_l649_64996


namespace NUMINAMATH_GPT_rate_per_meter_eq_2_5_l649_64938

-- Definitions of the conditions
def diameter : ℝ := 14
def total_cost : ℝ := 109.96

-- The theorem to be proven
theorem rate_per_meter_eq_2_5 (π : ℝ) (hπ : π = 3.14159) : 
  diameter = 14 ∧ total_cost = 109.96 → (109.96 / (π * 14)) = 2.5 :=
by
  sorry

end NUMINAMATH_GPT_rate_per_meter_eq_2_5_l649_64938


namespace NUMINAMATH_GPT_box_volume_l649_64919

theorem box_volume (initial_length initial_width cut_length : ℕ)
  (length_condition : initial_length = 13) (width_condition : initial_width = 9)
  (cut_condition : cut_length = 2) : 
  (initial_length - 2 * cut_length) * (initial_width - 2 * cut_length) * cut_length = 90 := 
by
  sorry

end NUMINAMATH_GPT_box_volume_l649_64919


namespace NUMINAMATH_GPT_max_min_diff_half_dollars_l649_64907

-- Definitions based only on conditions
variables (a c d : ℕ)

-- Conditions:
def condition1 : Prop := a + c + d = 60
def condition2 : Prop := 5 * a + 25 * c + 50 * d = 1000

-- The mathematically equivalent proof statement
theorem max_min_diff_half_dollars : condition1 a c d → condition2 a c d → (∃ d_min d_max : ℕ, d_min = 0 ∧ d_max = 15 ∧ d_max - d_min = 15) :=
by
  intros
  sorry

end NUMINAMATH_GPT_max_min_diff_half_dollars_l649_64907


namespace NUMINAMATH_GPT_prime_square_mod_30_l649_64948

theorem prime_square_mod_30 (p : ℕ) (hp : Nat.Prime p) (hp_gt_5 : p > 5) : 
  p^2 % 30 = 1 ∨ p^2 % 30 = 19 := 
sorry

end NUMINAMATH_GPT_prime_square_mod_30_l649_64948


namespace NUMINAMATH_GPT_interest_earned_l649_64972

theorem interest_earned (P : ℝ) (r : ℝ) (n : ℕ) (A : ℝ) : 
  P = 2000 → r = 0.05 → n = 5 → 
  A = P * (1 + r)^n → 
  A - P = 552.56 :=
by
  intro hP hr hn hA
  rw [hP, hr, hn] at hA
  sorry

end NUMINAMATH_GPT_interest_earned_l649_64972


namespace NUMINAMATH_GPT_rational_with_smallest_absolute_value_is_zero_l649_64995

theorem rational_with_smallest_absolute_value_is_zero (r : ℚ) :
  (forall r : ℚ, |r| ≥ 0) →
  (forall r : ℚ, r ≠ 0 → |r| > 0) →
  |r| = 0 ↔ r = 0 := sorry

end NUMINAMATH_GPT_rational_with_smallest_absolute_value_is_zero_l649_64995


namespace NUMINAMATH_GPT_sum_a1_a11_l649_64932

theorem sum_a1_a11 
  (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 a_11 : ℤ) 
  (h1 : a_0 = -512) 
  (h2 : -2 = a_0 + a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 + a_9 + a_10 + a_11) 
  : a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 + a_9 + a_10 + a_11 = 510 :=
sorry

end NUMINAMATH_GPT_sum_a1_a11_l649_64932


namespace NUMINAMATH_GPT_students_remaining_l649_64989

theorem students_remaining (students_showed_up : ℕ) (students_checked_out : ℕ) (students_left : ℕ) :
  students_showed_up = 16 → students_checked_out = 7 → students_left = students_showed_up - students_checked_out → students_left = 9 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_students_remaining_l649_64989


namespace NUMINAMATH_GPT_scientific_notation_of_105000_l649_64914

theorem scientific_notation_of_105000 : (105000 : ℝ) = 1.05 * 10^5 := 
by {
  sorry
}

end NUMINAMATH_GPT_scientific_notation_of_105000_l649_64914


namespace NUMINAMATH_GPT_total_distance_is_20_l649_64959

noncomputable def total_distance_walked (x : ℝ) : ℝ :=
  let flat_distance := 4 * x
  let uphill_time := (2 / 3) * (5 - x)
  let uphill_distance := 3 * uphill_time
  let downhill_time := (1 / 3) * (5 - x)
  let downhill_distance := 6 * downhill_time
  flat_distance + uphill_distance + downhill_distance

theorem total_distance_is_20 :
  ∃ x : ℝ, x >= 0 ∧ x <= 5 ∧ total_distance_walked x = 20 :=
by
  -- The existence proof is omitted (hence the sorry)
  sorry

end NUMINAMATH_GPT_total_distance_is_20_l649_64959


namespace NUMINAMATH_GPT_int_pairs_satisfy_eq_l649_64962

theorem int_pairs_satisfy_eq (x y : ℤ) : (x^2 = y^2 + 2 * y + 13) ↔ ((x = 4 ∧ y = 1) ∨ (x = -4 ∧ y = -5)) :=
by 
  sorry

end NUMINAMATH_GPT_int_pairs_satisfy_eq_l649_64962


namespace NUMINAMATH_GPT_claire_balance_after_week_l649_64905

theorem claire_balance_after_week :
  ∀ (gift_card : ℝ) (latte_cost croissant_cost : ℝ) (days : ℕ) (cookie_cost : ℝ) (cookies : ℕ),
  gift_card = 100 ∧
  latte_cost = 3.75 ∧
  croissant_cost = 3.50 ∧
  days = 7 ∧
  cookie_cost = 1.25 ∧
  cookies = 5 →
  (gift_card - (days * (latte_cost + croissant_cost) + cookie_cost * cookies) = 43) :=
by
  -- Skipping proof details with sorry
  sorry

end NUMINAMATH_GPT_claire_balance_after_week_l649_64905


namespace NUMINAMATH_GPT_parabola_functions_eq_l649_64964

noncomputable def f (x : ℝ) (b : ℝ) (c : ℝ) : ℝ := x^2 + b * x + c
noncomputable def g (x : ℝ) (c : ℝ) (b : ℝ) : ℝ := x^2 + c * x + b

theorem parabola_functions_eq : ∀ (x₁ x₂ : ℝ), 
  (∃ t : ℝ, (f t b c = g t c b) ∧ (t = 1)) → 
    (f x₁ 2 (-3) = x₁^2 + 2 * x₁ - 3) ∧ (g x₂ (-3) 2 = x₂^2 - 3 * x₂ + 2) :=
sorry

end NUMINAMATH_GPT_parabola_functions_eq_l649_64964


namespace NUMINAMATH_GPT_train_passing_time_l649_64904

theorem train_passing_time (L : ℕ) (v_kmph : ℕ) (v_mps : ℕ) (time : ℕ)
  (h1 : L = 90)
  (h2 : v_kmph = 36)
  (h3 : v_mps = v_kmph * (1000 / 3600))
  (h4 : v_mps = 10)
  (h5 : time = L / v_mps) :
  time = 9 := by
  sorry

end NUMINAMATH_GPT_train_passing_time_l649_64904


namespace NUMINAMATH_GPT_molecular_weight_of_one_mole_l649_64951

theorem molecular_weight_of_one_mole 
  (total_weight : ℝ) (n_moles : ℝ) (mw_per_mole : ℝ)
  (h : total_weight = 792) (h2 : n_moles = 9) 
  (h3 : total_weight = n_moles * mw_per_mole) 
  : mw_per_mole = 88 :=
by
  sorry

end NUMINAMATH_GPT_molecular_weight_of_one_mole_l649_64951


namespace NUMINAMATH_GPT_actual_area_of_park_l649_64939

-- Definitions of given conditions
def map_scale : ℕ := 250 -- scale: 1 inch = 250 miles
def map_length : ℕ := 6 -- length on map in inches
def map_width : ℕ := 4 -- width on map in inches

-- Definition of actual lengths
def actual_length : ℕ := map_length * map_scale -- actual length in miles
def actual_width : ℕ := map_width * map_scale -- actual width in miles

-- Theorem to prove the actual area
theorem actual_area_of_park : actual_length * actual_width = 1500000 := by
  -- By the conditions provided, the actual length and width in miles can be calculated directly:
  -- actual_length = 6 * 250 = 1500
  -- actual_width = 4 * 250 = 1000
  -- actual_area = 1500 * 1000 = 1500000
  sorry

end NUMINAMATH_GPT_actual_area_of_park_l649_64939


namespace NUMINAMATH_GPT_gcd_2728_1575_l649_64930

theorem gcd_2728_1575 : Int.gcd 2728 1575 = 1 :=
by sorry

end NUMINAMATH_GPT_gcd_2728_1575_l649_64930


namespace NUMINAMATH_GPT_max_profit_correctness_l649_64988

noncomputable def daily_purchase_max_profit := 
  let purchase_price := 4.2
  let selling_price := 6
  let return_price := 1.2
  let days_sold_10kg := 10
  let days_sold_6kg := 20
  let days_in_month := 30
  let profit_function (x : ℝ) := 
    10 * x * (selling_price - purchase_price) + 
    days_sold_6kg * 6 * (selling_price - purchase_price) + 
    days_sold_6kg * (x - 6) * (return_price - purchase_price)
  (6, profit_function 6)

theorem max_profit_correctness : daily_purchase_max_profit = (6, 324) :=
  sorry

end NUMINAMATH_GPT_max_profit_correctness_l649_64988


namespace NUMINAMATH_GPT_tomatoes_picked_yesterday_l649_64910

/-
Given:
1. The farmer initially had 171 tomatoes.
2. The farmer picked some tomatoes yesterday (Y).
3. The farmer picked 30 tomatoes today.
4. The farmer will have 7 tomatoes left after today.

Prove:
The number of tomatoes the farmer picked yesterday (Y) is 134.
-/

theorem tomatoes_picked_yesterday (Y : ℕ) (h : 171 - Y - 30 = 7) : Y = 134 :=
sorry

end NUMINAMATH_GPT_tomatoes_picked_yesterday_l649_64910


namespace NUMINAMATH_GPT_fraction_to_terminating_decimal_l649_64912

theorem fraction_to_terminating_decimal :
  (45 : ℚ) / 64 = (703125 : ℚ) / 1000000 := by
  sorry

end NUMINAMATH_GPT_fraction_to_terminating_decimal_l649_64912


namespace NUMINAMATH_GPT_winding_clock_available_time_l649_64974

theorem winding_clock_available_time
    (minute_hand_restriction_interval: ℕ := 5) -- Each interval the minute hand restricts
    (hour_hand_restriction_interval: ℕ := 60) -- Each interval the hour hand restricts
    (intervals_per_12_hours: ℕ := 2) -- Number of restricted intervals in each 12-hour cycle
    (minutes_in_day: ℕ := 24 * 60) -- Total minutes in 24 hours
    : (minutes_in_day - ((minute_hand_restriction_interval * intervals_per_12_hours * 12) + 
                         (hour_hand_restriction_interval * intervals_per_12_hours * 2))) = 1080 :=
by
  -- Skipping the proof steps
  sorry

end NUMINAMATH_GPT_winding_clock_available_time_l649_64974


namespace NUMINAMATH_GPT_calculate_expression_l649_64963

-- Theorem statement for the provided problem
theorem calculate_expression :
  ((18 ^ 15 / 18 ^ 14)^3 * 8 ^ 3) / 4 ^ 5 = 2916 := by
  sorry

end NUMINAMATH_GPT_calculate_expression_l649_64963


namespace NUMINAMATH_GPT_smallest_positive_period_of_f_max_min_values_of_f_in_interval_l649_64901

noncomputable def vec_a (x : ℝ) : ℝ × ℝ := (Real.cos x, -1 / 2)
noncomputable def vec_b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin x, Real.cos (2 * x))
noncomputable def f (x : ℝ) : ℝ := (vec_a x).1 * (vec_b x).1 + (vec_a x).2 * (vec_b x).2

theorem smallest_positive_period_of_f :
  (∀ x : ℝ, f (x + Real.pi) = f x) ∧ (∀ T > 0, (∀ x : ℝ, f (x + T) = f x) → T ≥ Real.pi) :=
by sorry

theorem max_min_values_of_f_in_interval :
  ∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≤ 1 ∧ f x ≥ -1 / 2 :=
by sorry

end NUMINAMATH_GPT_smallest_positive_period_of_f_max_min_values_of_f_in_interval_l649_64901


namespace NUMINAMATH_GPT_other_diagonal_of_rhombus_l649_64986

noncomputable def calculate_other_diagonal (area d1 : ℝ) : ℝ :=
  (area * 2) / d1

theorem other_diagonal_of_rhombus {a1 a2 : ℝ} (area_eq : a1 = 21.46) (d1_eq : a2 = 7.4) : calculate_other_diagonal a1 a2 = 5.8 :=
by
  rw [area_eq, d1_eq]
  norm_num
  -- The next step would involve proving that (21.46 * 2) / 7.4 = 5.8 in a formal proof.
  sorry

end NUMINAMATH_GPT_other_diagonal_of_rhombus_l649_64986


namespace NUMINAMATH_GPT_solve_y_minus_x_l649_64908

theorem solve_y_minus_x (x y : ℝ) (h1 : x + y = 399) (h2 : x / y = 0.9) : y - x = 21 :=
sorry

end NUMINAMATH_GPT_solve_y_minus_x_l649_64908


namespace NUMINAMATH_GPT_purchase_gifts_and_have_money_left_l649_64987

/-
  We start with 5000 forints in our pocket to buy gifts, visiting three stores.
  In each store, we find a gift that we like and purchase it if we have enough money. 
  The prices in each store are independently 1000, 1500, or 2000 forints, each with a probability of 1/3. 
  What is the probability that we can purchase gifts from all three stores 
  and still have money left (i.e., the total expenditure is at most 4500 forints)?
-/

def giftProbability (totalForints : ℕ) (prices : List ℕ) : ℚ :=
  let outcomes := prices |>.product prices |>.product prices
  let favorable := outcomes.filter (λ ((p1, p2), p3) => p1 + p2 + p3 <= totalForints)
  favorable.length / outcomes.length

theorem purchase_gifts_and_have_money_left :
  giftProbability 4500 [1000, 1500, 2000] = 17 / 27 :=
sorry

end NUMINAMATH_GPT_purchase_gifts_and_have_money_left_l649_64987


namespace NUMINAMATH_GPT_inequality_solution_l649_64911

theorem inequality_solution (x : ℝ) (h : 1 - x > x - 1) : x < 1 :=
sorry

end NUMINAMATH_GPT_inequality_solution_l649_64911


namespace NUMINAMATH_GPT_range_of_m_l649_64978

-- Definitions of propositions
def is_circle (m : ℝ) : Prop :=
  ∃ x y : ℝ, (x - m)^2 + y^2 = 2 * m - m^2 ∧ 2 * m - m^2 > 0

def is_hyperbola_eccentricity_in_interval (m : ℝ) : Prop :=
  1 < Real.sqrt (1 + m / 5) ∧ Real.sqrt (1 + m / 5) < 2

-- Proving the main statement
theorem range_of_m (m : ℝ) (h1 : is_circle m ∨ is_hyperbola_eccentricity_in_interval m)
  (h2 : ¬ (is_circle m ∧ is_hyperbola_eccentricity_in_interval m)) : 2 ≤ m ∧ m < 15 :=
sorry

end NUMINAMATH_GPT_range_of_m_l649_64978


namespace NUMINAMATH_GPT_trig_inequality_l649_64909
open Real

theorem trig_inequality (α β γ x y z : ℝ) 
  (h1 : α + β + γ = π)
  (h2 : x + y + z = 0) :
  y * z * (sin α)^2 + z * x * (sin β)^2 + x * y * (sin γ)^2 ≤ 0 := 
sorry

end NUMINAMATH_GPT_trig_inequality_l649_64909


namespace NUMINAMATH_GPT_inequality_holds_l649_64990

theorem inequality_holds (a b : ℝ) (h : a < b) (h₀ : b < 0) : - (1 / a) < - (1 / b) :=
sorry

end NUMINAMATH_GPT_inequality_holds_l649_64990


namespace NUMINAMATH_GPT_sum_of_ages_l649_64998

-- Definition of the ages based on the intervals and the youngest child's age.
def youngest_age : ℕ := 6
def second_youngest_age : ℕ := youngest_age + 2
def middle_age : ℕ := youngest_age + 4
def second_oldest_age : ℕ := youngest_age + 6
def oldest_age : ℕ := youngest_age + 8

-- The theorem stating the total sum of the ages of the children, given the conditions.
theorem sum_of_ages :
  youngest_age + second_youngest_age + middle_age + second_oldest_age + oldest_age = 50 :=
by sorry

end NUMINAMATH_GPT_sum_of_ages_l649_64998


namespace NUMINAMATH_GPT_find_m_l649_64945

theorem find_m (m : ℕ) (h1 : 0 ≤ m ∧ m ≤ 9) (h2 : (8 + 4 + 5 + 9) - (6 + m + 3 + 7) % 11 = 0) : m = 9 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l649_64945
