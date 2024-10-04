import Mathlib

namespace complement_union_in_universe_l774_774376

variable (U : Set ℕ := {1, 2, 3, 4, 5})
variable (M : Set ℕ := {1, 3})
variable (N : Set ℕ := {1, 2})

theorem complement_union_in_universe :
  (U \ (M ∪ N)) = {4, 5} :=
by
  sorry

end complement_union_in_universe_l774_774376


namespace fold_triangle_eq_Length_l774_774068

noncomputable def fold_triangle_square (a b c : ℝ) (side : ℝ) (touch_distance : ℝ) : ℝ :=
let fold_pt := touch_distance in
let PA := side - fold_pt in
let PB := fold_pt in
let PQ := PA^2 - PA * fold_pt + fold_pt^2 in
PQ

theorem fold_triangle_eq_Length {a b c : ℝ} : 
  fold_triangle_square a b c 12 9 = 59319 / 1225 :=
by
  sorry

end fold_triangle_eq_Length_l774_774068


namespace gcd_is_12_l774_774181

noncomputable def gcd_problem (b : ℤ) : Prop :=
  b % 2027 = 0 → Int.gcd (b^2 + 7*b + 18) (b + 6) = 12

-- Now, let's state the theorem
theorem gcd_is_12 (b : ℤ) : gcd_problem b :=
  sorry

end gcd_is_12_l774_774181


namespace trinomial_has_two_roots_l774_774910

theorem trinomial_has_two_roots (a b c : ℝ) (h : b^2 - 4 * a * c > 0) :
  let Δ := (2 * (a + b))^2 - 4 * 3 * a * (b + c) in Δ > 0 :=
by
  sorry

end trinomial_has_two_roots_l774_774910


namespace eccentricity_of_C2_l774_774575

noncomputable def find_eccentricity : ℝ :=
  let a := 3    -- semi-major axis of ellipse C_2
  let b := sqrt 5  -- semi-minor axis of ellipse C_2
  let c := sqrt (a^2 - b^2)  -- distance from the center to a focus
  c / a  -- formula for eccentricity

theorem eccentricity_of_C2 :
  find_eccentricity = 2 / 3 :=
by
  sorry

end eccentricity_of_C2_l774_774575


namespace perpendicular_aj_jc_l774_774233

theorem perpendicular_aj_jc (A B C D J : Point) (h_triangle_ABC : IsTriangle A B C)
  (h_angle_A : angle A = 108) (h_AB_eq_AC : AB = AC) (h_extend_AC_to_D : Extend AC D)
  (h_AD_eq_BC : AD = BC) (h_J_midpoint_BD : Midpoint J B D) 
  : Perpendicular (Line.through A J) (Line.through J C) :=
sorry

end perpendicular_aj_jc_l774_774233


namespace width_of_rectangle_11_l774_774372

variable (L W : ℕ)

-- The conditions: 
-- 1. The perimeter is 48cm
-- 2. Width is 2 cm shorter than length
def is_rectangle (L W : ℕ) : Prop :=
  2 * L + 2 * W = 48 ∧ W = L - 2

-- The statement we need to prove
theorem width_of_rectangle_11 (L W : ℕ) (h : is_rectangle L W) : W = 11 :=
by
  sorry

end width_of_rectangle_11_l774_774372


namespace find_m_l774_774395

theorem find_m (m : ℕ) : 
  (∃ (k : ℕ), k = 12 / m ∧ 
  6 * (k - 2) ^ 2 = 12 * (k - 2)) → 
  m = 3 :=
by
  -- Given part of the proof problem
  intro h,
  cases' h with k hk,
  have eq1 : k = 12 / m := hk.left,
  have eq2 : 6 * (k - 2) ^ 2 = 12 * (k - 2) := hk.right,
  sorry

end find_m_l774_774395


namespace comparison_of_a_b_c_l774_774544

def a : ℝ := Real.log 2 / Real.log 3
def b : ℝ := 0.3 ^ 0.5
def c : ℝ := 0.5 ^ (-0.2)

theorem comparison_of_a_b_c : a < b ∧ b < c := by
  sorry

end comparison_of_a_b_c_l774_774544


namespace girl_speed_l774_774902

theorem girl_speed (distance time : ℝ) (h_distance : distance = 96) (h_time : time = 16) : distance / time = 6 :=
by
  sorry

end girl_speed_l774_774902


namespace largest_x_floor_condition_l774_774998

theorem largest_x_floor_condition :
  ∃ x : ℝ, (⌊x⌋ : ℝ) / x = 8 / 9 ∧
      (∀ y : ℝ, (⌊y⌋ : ℝ) / y = 8 / 9 → y ≤ x) →
  x = 63 / 8 :=
by
  sorry

end largest_x_floor_condition_l774_774998


namespace ordered_pairs_harmonic_mean_4_30_l774_774154

-- Definitions based on given conditions
def is_harmonic_mean (x y : ℕ) (h : Nat) : Prop := 2 * x * y = h * (x + y)

def num_ordered_pairs_with_harmonic_mean (n : ℕ) : ℕ :=
  (Nat.divisors (2 * n)).filter (λ d, d < n).length

theorem ordered_pairs_harmonic_mean_4_30 :
  num_ordered_pairs_with_harmonic_mean (2 ^ 59) = 59 :=
by
  -- Skipping the proof with sorry.
  sorry

end ordered_pairs_harmonic_mean_4_30_l774_774154


namespace pigeonhole_principle_computers_l774_774382

theorem pigeonhole_principle_computers (c : Fin 7 → ℕ) (h : (∑ i, c i) = 20) :
  ∃ i j : Fin 7, i ≠ j ∧ c i = c j :=
sorry

end pigeonhole_principle_computers_l774_774382


namespace proof_problem_l774_774777

def sum_even_ints (n : ℕ) : ℕ := n * (n + 1)
def sum_odd_ints (n : ℕ) : ℕ := n^2
def sum_specific_primes : ℕ := [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97].sum

theorem proof_problem : (sum_even_ints 100 - sum_odd_ints 100) + sum_specific_primes = 1063 :=
by
  sorry

end proof_problem_l774_774777


namespace number_of_terms_in_sequence_l774_774114

def is_arithmetic_sequence (a d : ℕ) (u : ℕ → ℕ) : Prop :=
∀ n : ℕ, u n = a + n * d

def nth_term (u : ℕ → ℕ) (n : ℕ) : ℕ :=
u n

theorem number_of_terms_in_sequence : 
  ∃ n : ℕ, nth_term (λ n, 2 + n * 3) n = 110 ∧ n + 1 = 37 :=
by 
  sorry

end number_of_terms_in_sequence_l774_774114


namespace third_pipe_empty_rate_time_l774_774427

-- Definitions based on given conditions
def pipeA_fill_rate : ℚ := 1 / 60
def pipeB_fill_rate : ℚ := 1 / 75
def combined_fill_rate : ℚ := 1 / 50

-- Define what we need to prove
theorem third_pipe_empty_rate_time :
  let pipeC_empty_rate := 3 / 300 in
  let third_pipe_empty_time := 100 in
  (pipeA_fill_rate + pipeB_fill_rate - pipeC_empty_rate = combined_fill_rate) ∧ 
  (1 / pipeC_empty_rate = third_pipe_empty_time) := 
by
  sorry

end third_pipe_empty_rate_time_l774_774427


namespace upper_limit_of_total_people_l774_774516

theorem upper_limit_of_total_people (T : ℕ) (h1 : (3 / 7 : ℚ) * T = 30) (h2 : (5 / 10 : ℚ) * T = T / 2) : T ≤ 70 :=
begin
  have hT : T = 30 * (7 / 3),
  { rw ← mul_right_inj' (by norm_num : (3 / 7 : ℚ) ≠ 0),
    norm_cast at h1,
    rwa [← eq_div_iff_mul_eq (by norm_num : (3 : ℚ) ≠ 0)] },
  norm_cast at hT,
  rw ← hT,
  norm_num,
end

end upper_limit_of_total_people_l774_774516


namespace trig_identity_example_l774_774026

theorem trig_identity_example :
  sin (20 * Real.pi / 180) * cos (10 * Real.pi / 180) + sin (10 * Real.pi / 180) * sin (70 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end trig_identity_example_l774_774026


namespace intersection_of_lines_l774_774278

open EuclideanGeometry

theorem intersection_of_lines
  (ABC : Triangle)
  (A B C : Point)
  (hABC : ABC = mkTriangle A B C)
  (h_acute : acute_triangle ABC)
  (h_distinct : A ≠ C ∧ C ≠ B)
  (M : Point)
  (hM : M = midpoint A B)
  (H : Point)
  (hH : H = orthocenter ABC)
  (D : Point)
  (hD : D = foot_of_altitude A C B)
  (E : Point)
  (hE : E = foot_of_altitude B C A)
  : ∃ (S : Point), collinear A B S ∧ collinear D E S ∧ perpendicular S M (line.mk M H) :=
by
  sorry

end intersection_of_lines_l774_774278


namespace number_of_perfect_square_divisors_of_450_l774_774640

theorem number_of_perfect_square_divisors_of_450 : 
    let p := 450;
    let factors := [(3, 2), (5, 2), (2, 1)];
    ∃ n, (n = 4 ∧ 
          ∀ (d : ℕ), d ∣ p → 
                     (∃ (a b c : ℕ), d = 2^a * 3^b * 5^c ∧ 
                              (a = 0) ∧ (b = 0 ∨ b = 2) ∧ (c = 0 ∨ c = 2) → 
                              a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 = 0)) :=
    sorry

end number_of_perfect_square_divisors_of_450_l774_774640


namespace probability_closer_to_6_than_2_l774_774072

noncomputable def midpoint (a b : ℝ) : ℝ := (a + b) / 2

theorem probability_closer_to_6_than_2 : 
  let total_length := 10
  let length_from_4_to_10 := 10 - 4
  total_length > 0 → 
  (length_from_4_to_10 : ℝ) / (total_length : ℝ) = 0.6 :=
by
  intro h_total_length_pos
  rw [length_from_4_to_10, total_length]
  norm_num
  sorry

end probability_closer_to_6_than_2_l774_774072


namespace nearest_integer_sum_fractions_l774_774416

/-- 
Given the fractions 2007 / 2999, 8001 / 5998, and 2001 / 3999,
we assert that the integer 3 is the nearest in value to their sum.
-/
theorem nearest_integer_sum_fractions : 
  let f1 := 2007 / 2999
  let f2 := 8001 / 5998
  let f3 := 2001 / 3999
  let sum := f1 + f2 + f3
  (abs (sum - 3) < abs (sum - 2)) ∧
  (abs (sum - 3) < abs (sum - 4)) ∧
  (abs (sum - 3) < abs (sum - 1)) ∧
  (abs (sum - 3) < abs (sum - 5)) :=
by
  let f1 := 2007 / 2999
  let f2 := 8001 / 5998
  let f3 := 2001 / 3999
  let sum := f1 + f2 + f3
  sorry

end nearest_integer_sum_fractions_l774_774416


namespace coeff_x3_in_expansion_l774_774344

theorem coeff_x3_in_expansion : 
  (coeff (x^3) in (1 - 2*x) * (1 - x)^5) = -30 :=
sorry

end coeff_x3_in_expansion_l774_774344


namespace collinear_Xa_Xb_Xc_l774_774876

theorem collinear_Xa_Xb_Xc
  (A B C : Point)
  (hABC : Triangle A B C)
  (circumcircle : Circle)
  (Hcircum : CircumCircle ABC circumcircle)
  (AA' : Line)
  (AA'_mid : Median A (B, C))
  (A'' : Point)
  (HAA'c : ExtendToCircumCircle A AA' circumcircle A'')
  (APa : Line)
  (HAPa : Diameter circumcircle APa)
  (A' : Point)
  (Hperp : Perpendicular A' APa (Tangent circumcircle A'') X_a)
  (Hb : DefineX_B Similar)
  (Hc : DefineX_C Similar) :
  Collinear X_a X_b X_c :=
sorry

end collinear_Xa_Xb_Xc_l774_774876


namespace perfect_square_divisors_count_450_l774_774648

theorem perfect_square_divisors_count_450 :
  let is_divisor (d n : ℕ) := n % d = 0
  let is_perfect_square (n : ℕ) := ∃ m : ℕ, m * m = n
  let prime_factorization_450 := [(2, 1), (3, 2), (5, 2)]
  ∃ (count : ℕ), count = 4 ∧ count = (Finset.filter 
    (λ d, is_perfect_square d ∧ is_divisor d 450)
    (Finset.powersetPrimeFactors prime_factorization_450)).card := sorry

end perfect_square_divisors_count_450_l774_774648


namespace odd_function_f_value_a_b_range_of_k_l774_774555

open Real

noncomputable def f (x : ℝ) : ℝ := (-2^x + 1) / (2^(x+1) + 2)

theorem odd_function_f: ∀ x: ℝ, f (-x) = - f x := 
by 
  sorry
  
theorem value_a_b (a b: ℝ) : 
  (f(0)=0 → b=1) ∧ (f (-1) = -f (1) → a = 2) :=
by 
  sorry

theorem range_of_k (k : ℝ) :
  (∀ t: ℝ, f(t^2 - 2 * t) + f(2 * t^2 - k) < 0) → k < -1/3 :=
by
  sorry

end odd_function_f_value_a_b_range_of_k_l774_774555


namespace perfect_square_factors_450_l774_774625

theorem perfect_square_factors_450 : 
  ∃ n, n = 4 ∧ (∀ k | 450, k ∣ 450 → is_square k) → (∃ m1 m2 m3 : ℕ, 450 = 2^m1 * 3^m2 * 5^m3 ) :=
by sorry

end perfect_square_factors_450_l774_774625


namespace range_of_a_l774_774584

noncomputable def f (x : ℝ) : ℝ := x^2 - (1 / 2) * Real.log x + (3 / 2)

theorem range_of_a (a : ℝ) :
  (∀ (x : ℝ), a - 1 < x ∧ x < a + 1 → 0 < x) ∧
  (¬ ∀ (x : ℝ), a - 1 < x ∧ x < a + 1 → monotone_on f (set.Ioo (a - 1) (a + 1))) →
  1 ≤ a ∧ a < (3 / 2) := 
sorry

end range_of_a_l774_774584


namespace ball_total_distance_traveled_l774_774080

theorem ball_total_distance_traveled :
  let initial_height := 150
  let rebound_ratio := 1 / 3
  let descent_1 := initial_height
  let ascent_1 := initial_height * rebound_ratio
  let descent_2 := ascent_1
  let ascent_2 := ascent_1 * rebound_ratio
  let descent_3 := ascent_2
  let ascent_3 := ascent_2 * rebound_ratio
  let descent_4 := ascent_3
  let ascent_4 := ascent_3 * rebound_ratio
  let descent_5 := ascent_4
  let total_descents := descent_1 + descent_2 + descent_3 + descent_4 + descent_5
  let total_ascents := ascent_1 + ascent_2 + ascent_3 + ascent_4
  let total_distance := total_descents + total_ascents
  total_distance ≈ 298.16 :=
by
  sorry

end ball_total_distance_traveled_l774_774080


namespace max_n_possible_l774_774840

theorem max_n_possible (k : ℕ) (h_k : k > 1) : ∃ n : ℕ, n = k - 1 :=
by
  sorry

end max_n_possible_l774_774840


namespace triangle_abf_area_l774_774859

theorem triangle_abf_area {A B C D E F : Point} 
  (square_ABCD : square ABCD) 
  (vertex_E : is_vertex E (equilateral_triangle ABE)) 
  (F_intersection : F = (diagonal BD).intersection (line_segment AE)) 
  (AB_length : dist A B = Real.sqrt (1 + Real.sqrt 3)) :
  area (triangle ABF) = Real.sqrt 3 / 2 :=
sorry

end triangle_abf_area_l774_774859


namespace michael_truck_meet_once_l774_774781

/-- Michael walks at 6 feet per second -/
def michael_speed := 6

/-- Trash pails are located every 300 feet along the path -/
def pail_distance := 300

/-- A garbage truck travels at 15 feet per second -/
def truck_speed := 15

/-- The garbage truck stops for 45 seconds at each pail -/
def truck_stop_time := 45

/-- Michael passes a pail just as the truck leaves the next pail -/
def initial_distance := 300

/-- Prove that Michael and the truck meet exactly 1 time -/
theorem michael_truck_meet_once :
  ∀ (meeting_times : ℕ), meeting_times = 1 := by
  sorry

end michael_truck_meet_once_l774_774781


namespace required_investment_amount_l774_774687

def comp_interest (FV : ℝ) (r : ℝ) (n : ℕ) : ℝ := FV / (1 + r) ^ n

def present_value := comp_interest 750000 0.07 15

theorem required_investment_amount :
  present_value ≈ 271882.35 := by sorry

end required_investment_amount_l774_774687


namespace domino_covering_count_l774_774712

theorem domino_covering_count : 
  let F : ℕ → ℕ := λ n,
    if n = 0 then 1
    else if n = 1 then 1
    else (F (n - 1) + F (n - 2)) in
  let count_configurations := 2 + 2 * (F 11)^2 * (F 9)^2 + 12 * (F 11) * (F 10)^2 * (F 9) + 2 * (F 10)^4 in
  count_configurations = 146458404 :=
by
  let F : ℕ → ℕ := λ n,
    if n = 0 then 1
    else if n = 1 then 1
    else (F (n - 1) + F (n - 2)) in
  let count_configurations := 2 + 2 * (F 11)^2 * (F 9)^2 + 12 * (F 11) * (F 10)^2 * (F 9) + 2 * (F 10)^4 in
  sorry

end domino_covering_count_l774_774712


namespace num_tables_l774_774439

theorem num_tables (T : ℕ) : 
  (6 * T = (17 / 3) * T) → 
  T = 6 :=
sorry

end num_tables_l774_774439


namespace remainder_mod_8_l774_774868

theorem remainder_mod_8 (x : ℤ) (h : x % 63 = 25) : x % 8 = 1 := 
sorry

end remainder_mod_8_l774_774868


namespace calc_sum_euler_form_l774_774485

theorem calc_sum_euler_form :
  12 * complex.exp (complex.I * (3 * real.pi / 13)) + 
  12 * complex.exp (complex.I * (7 * real.pi / 26)) = 
  12 * real.sqrt (2 + real.sqrt 2) * complex.exp (complex.I * (3.25 * real.pi / 13)) :=
sorry

end calc_sum_euler_form_l774_774485


namespace crazy_silly_school_books_l774_774005

theorem crazy_silly_school_books (n : ℕ) :
  (∃ books movies read_books watched_movies remaining_movies,
    books = n ∧ 
    movies = 17 ∧ 
    read_books = 19 ∧ 
    watched_movies = 7 ∧ 
    remaining_movies = 10 ∧ 
    watched_movies + remaining_movies = movies) →
  n = 19 :=
by
  intros h
  obtain ⟨books, movies, read_books, watched_movies, remaining_movies,
          h_books, h_movies, h_read_books, h_watched_movies, h_remaining_movies, h_total_movies⟩ := h
  rw [h_books, h_read_books]
  exact rfl

#eval crazy_silly_school_books 19 

end crazy_silly_school_books_l774_774005


namespace sum_of_digits_of_45_times_40_l774_774370

theorem sum_of_digits_of_45_times_40 : ∑ d in (digits 10 (45 * 40)).to_finset, d = 9 :=
sorry

end sum_of_digits_of_45_times_40_l774_774370


namespace possible_measure_of_angle_AOC_l774_774741

-- Given conditions
def angle_AOB : ℝ := 120
def OC_bisects_angle_AOB (x : ℝ) : Prop := x = 60
def OD_bisects_angle_AOB_and_OC_bisects_angle (x y : ℝ) : Prop :=
  (y = 60 ∧ (x = 30 ∨ x = 90))

-- Theorem statement
theorem possible_measure_of_angle_AOC (angle_AOC : ℝ) :
  (OC_bisects_angle_AOB angle_AOC ∨ 
  (OD_bisects_angle_AOB_and_OC_bisects_angle angle_AOC 60)) →
  (angle_AOC = 30 ∨ angle_AOC = 60 ∨ angle_AOC = 90) :=
by
  sorry

end possible_measure_of_angle_AOC_l774_774741


namespace find_radius_of_small_semicircle_l774_774734

noncomputable def radius_of_small_semicircle (R : ℝ) (r : ℝ) :=
  ∀ (x : ℝ),
    (12: ℝ = R) ∧ (6: ℝ = r) →
    (∃ (x: ℝ), R - x + r = sqrt((r + x)^2 - r^2)) →
    x = 4

theorem find_radius_of_small_semicircle : radius_of_small_semicircle 12 6 :=
begin
  unfold radius_of_small_semicircle,
  intro x,
  assume h1 h2,
  cases h2,
  sorry,
end

end find_radius_of_small_semicircle_l774_774734


namespace percent_employed_population_l774_774723

theorem percent_employed_population (P : ℝ) (E : ℝ) (h1 : 0.48 * P = P * 0.75 * E) : E = 0.64 :=
by
  have h2 : 0.48 = 0.75 * E, from (mul_right_inj' (ne_of_gt (by norm_num1 : 0 < P))).mp h1
  exact (eq_div_iff (by norm_num1 : 0.75 ≠ 0)).mpr h2

end percent_employed_population_l774_774723


namespace max_value_x_div_y_l774_774295

variables {x y a b : ℝ}

theorem max_value_x_div_y (h1 : x ≥ y) (h2 : y > 0) (h3 : 0 ≤ a) (h4 : a ≤ x) (h5 : 0 ≤ b) (h6 : b ≤ y) 
  (h7 : (x - a)^2 + (y - b)^2 = x^2 + b^2) (h8 : x^2 + b^2 = y^2 + a^2) :
  x / y ≤ (2 * Real.sqrt 3) / 3 :=
sorry

end max_value_x_div_y_l774_774295


namespace diagonals_bisect_each_other_l774_774245

theorem diagonals_bisect_each_other (ABCD: Parallelogram) : 
  ∀ P Q : Point, midpoint P Q = intersection (diagonal AC) (diagonal BD) := sorry

end diagonals_bisect_each_other_l774_774245


namespace fraction_scaled_l774_774692

theorem fraction_scaled (x y : ℝ) :
  ∃ (k : ℝ), (k = 3 * y) ∧ ((5 * x + 3 * y) / (x + 3 * y) = 5 * ((x + (3 * y)) / (x + (3 * y)))) := 
  sorry

end fraction_scaled_l774_774692


namespace sum_last_two_digits_is_correct_l774_774864

def fibs : List Nat := [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233]

def factorial_last_two_digits (n : Nat) : Nat :=
  (Nat.factorial n) % 100

def modified_fib_factorial_series : List Nat :=
  fibs.map (λ k => (factorial_last_two_digits k + 2) % 100)

def sum_last_two_digits : Nat :=
  (modified_fib_factorial_series.sum) % 100

theorem sum_last_two_digits_is_correct :
  sum_last_two_digits = 14 :=
sorry

end sum_last_two_digits_is_correct_l774_774864


namespace find_expression_l774_774493

variable {a b θ : ℝ}

theorem find_expression
  (h : (sin(θ)^6 / a^2) + (cos(θ)^6 / b^2) = 1 / (a^2 + b^2)) :
  (sin(θ)^12 / a^5) + (cos(θ)^12 / b^5) = 1 / a^5 + 1 / b^5 :=
sorry

end find_expression_l774_774493


namespace length_of_living_room_l774_774353

theorem length_of_living_room (width area : ℝ) (h_width : width = 14) (h_area : area = 215.6) :
  ∃ length : ℝ, length = 15.4 ∧ area = length * width :=
by
  sorry

end length_of_living_room_l774_774353


namespace trajectory_of_M_is_line_segment_l774_774246

-- Define the structure for a regular tetrahedron
structure Tetrahedron :=
(P A B C : Point)
(is_regular : RegularTetrahedron P A B C)

-- Define the point M
def M (ABC : Triangle) : Point := sorry

-- Define the distances from point M to each face forming an arithmetic sequence
def distances_form_arithmetic_sequence (P A B C M : Point) : Prop :=
    let d₁ := distance_from_point_to_face M (P, A, B)
    let d₂ := distance_from_point_to_face M (P, B, C)
    let d₃ := distance_from_point_to_face M (P, C, A)
    (d₂ - d₁) = (d₃ - d₂)

-- Define the main theorem stating the result
theorem trajectory_of_M_is_line_segment (P A B C : Point) (ABC : Triangle) (h1 : RegularTetrahedron P A B C)
    (M : Point) (h2 : is_in_triangle M ABC) (h3 : distances_form_arithmetic_sequence P A B C M) :
    ∃ l : Line, (trajectory_of M ABC) = l :=
sorry

end trajectory_of_M_is_line_segment_l774_774246


namespace common_ratio_geometric_series_l774_774369

theorem common_ratio_geometric_series (a r S : ℝ) (h₁ : S = a / (1 - r))
  (h₂ : r ≠ 1)
  (h₃ : r^4 * S = S / 81) :
  r = 1/3 :=
by 
  sorry

end common_ratio_geometric_series_l774_774369


namespace find_a_for_one_solution_l774_774996

noncomputable def unique_solution_interval : Set ℝ :=
  {-1 ≤ a ∧ a < -1/2} ∨ {-1/2 < a ∧ a < 1}

theorem find_a_for_one_solution (a : ℝ) :
  (∃! x : ℝ, a * |x - 1| + (x^2 - 7*x + 12) / (3 - x) = 0) ↔ (a ∈ unique_solution_interval) :=
sorry

end find_a_for_one_solution_l774_774996


namespace distance_A_to_original_position_l774_774939

theorem distance_A_to_original_position (s : ℝ) (A B C D : ℝ) :
  let area := 18 in
  let side := real.sqrt area in
  let diagonal_half := side * real.sqrt 2 / 2 in
  let black_visible_area := side * side / 2 in
  let white_visible_area := black_visible_area in
  black_visible_area = white_visible_area →
  s = side →
  A = diagonal_half →
  B = side / 2 →
  C = diagonal_half →
  D = side / 2 →
  A = D + 1 →
  B = A + 2 →
  (C = B + 3 ∧
  C - A = 3) :=
begin
  sorry
end

end distance_A_to_original_position_l774_774939


namespace problem1_problem2_l774_774889

open Nat

-- Define the main condition on n
def condition (n : ℕ) : Prop := n > 2

-- Define the property for a set of n consecutive numbers
def property (s : set ℕ) (n : ℕ) : Prop :=
  ∃ (k : ℕ), s = {k, k+1, ..., k+n-1} ∧ 
             (k+n-1) ∣ (List.lcm (list.filter (≠ (k+n-1)) [k, k+1, ..., k+n-1])) 

-- First statement: For n > 2, there exists a set with the property iff n ≥ 4.
theorem problem1 (n : ℕ) (hn : condition n) : ∃ s, property s n ↔ n ≥ 4 :=
sorry

-- Second statement: For n = 4, there is exactly one such set.
theorem problem2 : ∃! s, property s 4 :=
sorry

end problem1_problem2_l774_774889


namespace hexagonal_pyramid_distance_l774_774392

-- We define the variables mentioned in the conditions
variables (A B : ℝ) (ha hb : ℝ) (d : ℝ)

-- Define the conditions
def conditions : Prop :=
  A = 96 * Real.sqrt 3 ∧
  B = 216 * Real.sqrt 3 ∧
  d = 12 ∧
  ha > 0 ∧
  hb > 0 ∧
  ha ≥ hb

-- Define the mathematical goal:
def goal : Prop :=
  (B / A = 9 / 4) ∧
  Real.sqrt (B / A) = 3 / 2 ∧
  (ha - hb = d) ∧
  ha = 36

-- Define the theorem to prove the goal under given conditions
theorem hexagonal_pyramid_distance (A B : ℝ) (ha hb d : ℝ) 
  (h_cond : conditions A B ha hb d) : goal A B ha hb d :=
  by 
    cases h_cond with hA h_cond1
    cases h_cond1 with hB h_cond2
    cases h_cond2 with hd h_cond3
    cases h_cond3 with hha hhb h_ge
    split
    repeat {
        sorry
    }

end hexagonal_pyramid_distance_l774_774392


namespace sequence_equalities_condition_l774_774857

theorem sequence_equalities_condition (a b c : ℝ) (h1 : a = 0) (h2 : b ≠ 0) (h3 : c ≠ 1) :
  (a + abc / (a - bc + b)) / (b + abc / (a - ac + b)) =
  (a - ab / (a + 2b)) / (b - ab / (2a + b)) ∧
  (a - ab / (a + 2b)) / (b - ab / (2a + b)) =
  (2ab / (a - b) + a) / (2ab / (a - b) - b) ∧
  (2ab / (a - b) + a) / (2ab / (a - b) - b) = 
   a / b :=
by
  sorry

end sequence_equalities_condition_l774_774857


namespace trajectory_equation_minimum_distance_l774_774259

section part1

variable (t : ℝ)

def P (t : ℝ) : ℝ × ℝ := (t^2, 2 * t)

theorem trajectory_equation :
  ∀ t : ℝ, (P t).2 ^ 2 = 4 * (P t).1 := by
  sorry

end part1

section part2

variable (t : ℝ)

def line_eq (x y : ℝ) : Prop := x - y + 2 = 0

theorem minimum_distance :
  let P := (1, 2)
  line_eq P.1 P.2 ∧ ∀ (t : ℝ), |(t^2 - 2*t + 2)| / (Real.sqrt 2) ≥ Real.sqrt(2) / 2 := by
  sorry

end part2

end trajectory_equation_minimum_distance_l774_774259


namespace exists_a_for_solution_l774_774995

theorem exists_a_for_solution (b : ℝ) (h : b ∈ set.Ici (3/8) ∪ set.Iio 0) :
  ∃ a x y : ℝ, x = |y - b| + (3 / b) ∧ x^2 + y^2 + 32 = a * (2 * y - a) + 12 * x :=
sorry

end exists_a_for_solution_l774_774995


namespace new_volume_increased_dimensions_l774_774931

theorem new_volume_increased_dimensions (l w h : ℝ) 
  (h_vol : l * w * h = 5000) 
  (h_sa : l * w + w * h + h * l = 900) 
  (h_sum_edges : l + w + h = 60) : 
  (l + 2) * (w + 2) * (h + 2) = 7048 := 
by 
  sorry

end new_volume_increased_dimensions_l774_774931


namespace max_cables_installed_l774_774088

variable (employees : ℕ)
variable (brandA : ℕ)
variable (brandB : ℕ)
variable (specificBrandB : ℕ)
variable (maxConnections : ℕ)

axiom connectivity_constraints :
  employees = 50 ∧
  brandA = 15 ∧
  brandB = 35 ∧
  specificBrandB = 10 ∧
  maxConnections = 150

theorem max_cables_installed :
  employees = 50 →
  brandA = 15 →
  brandB = 35 →
  specificBrandB = 10 →
  ∃ cables : ℕ, cables = 150 :=
by
  intros h1 h2 h3 h4
  use 150
  exact connectivity_constraints.mpr ⟨h1, h2, h3, h4, rfl⟩

end max_cables_installed_l774_774088


namespace difference_in_surface_areas_l774_774457

-- Define the conditions: volumes and number of cubes
def V_large : ℕ := 343
def n : ℕ := 343
def V_small : ℕ := 1

-- Define the function to calculate the side length of a cube given its volume
def side_length (V : ℕ) : ℕ := V^(1/3 : ℕ)

-- Specify the side lengths of the larger and smaller cubes
def s_large : ℕ := side_length V_large
def s_small : ℕ := side_length V_small

-- Define the function to calculate the surface area of a cube given its side length
def surface_area (s : ℕ) : ℕ := 6 * s^2

-- Specify the surface areas of the larger cube and the total of the smaller cubes
def SA_large : ℕ := surface_area s_large
def SA_small_total : ℕ := n * surface_area s_small

-- State the theorem to prove
theorem difference_in_surface_areas : SA_small_total - SA_large = 1764 :=
by {
  -- Intentionally omit proof, as per instructions
  sorry
}

end difference_in_surface_areas_l774_774457


namespace area_of_triangle_PQR_l774_774852

theorem area_of_triangle_PQR : 
  let P := (0, 2)
  let Q := (3, 0)
  let R := (1, 6)
  let rectangle_width := 6
  let rectangle_height := 3
  triangle_area P Q R = 6 :=
begin
  sorry
end

end area_of_triangle_PQR_l774_774852


namespace time_against_current_l774_774070

-- Define the conditions:
def swimming_speed_still_water : ℝ := 6  -- Speed in still water (km/h)
def current_speed : ℝ := 2  -- Speed of the water current (km/h)
def time_with_current : ℝ := 3.5  -- Time taken to swim with the current (hours)

-- Define effective speeds:
def effective_speed_against_current (swimming_speed_still_water current_speed: ℝ) : ℝ :=
  swimming_speed_still_water - current_speed

def effective_speed_with_current (swimming_speed_still_water current_speed: ℝ) : ℝ :=
  swimming_speed_still_water + current_speed

-- Calculate the distance covered with the current:
def distance_with_current (time_with_current effective_speed_with_current: ℝ) : ℝ :=
  time_with_current * effective_speed_with_current

-- Define the proof goal:
theorem time_against_current (h1 : swimming_speed_still_water = 6) (h2 : current_speed = 2)
  (h3 : time_with_current = 3.5) :
  ∃ (t : ℝ), t = 7 := by
  sorry

end time_against_current_l774_774070


namespace hexagon_circle_tangent_radius_exists_l774_774556

-- Define the geometrical setup of the problem
def regular_hexagon (A B C D E F : Point) :=
  ∀ i ∈ {0, 1, 2, 3, 4, 5},  
  ∀ j ∈ {0, 1, 2, 3, 4, 5},
  (i = j + 1 ∨ i = j - 1 ∨ (i = 0 ∧ j = 5) ∨ (i = 5 ∧ j = 0)) → (distance (vertices i) (vertices j) = 10)

-- Define the tangent circle properties
def tangent_circle (A F C D : Point) (r : ℝ) :=
  passes_through A F r ∧ tangent_to C D r

-- Lean statement
theorem hexagon_circle_tangent_radius_exists :
  ∃ (r : ℝ), ∀ (A B C D E F G H O : Point),
  regular_hexagon A B C D E F ∧
  tangent_circle A F C D r →
  r = 20 := sorry

end hexagon_circle_tangent_radius_exists_l774_774556


namespace projection_ratio_parallelogram_l774_774772

-- Definitions of points and projections
variables (A B C D M E F : Type) [point : Euc3Space X]
variable [parallelogram : parallelogram A B C D]
variable [on_segment_AC : on_segment M A C]
variable [orthogonal_projection_E : orthogonal_projection E M A B]
variable [orthogonal_projection_F : orthogonal_projection F M C D]

-- Theorem statement
theorem projection_ratio_parallelogram :
  parallelogram A B C D →
  on_segment M A C →
  orthogonal_projection E M A B →
  orthogonal_projection F M C D →
  (ME / MF) = (AD / AB) :=
begin
  intros, 
  sorry
end

end projection_ratio_parallelogram_l774_774772


namespace perfect_square_divisors_of_450_l774_774614

theorem perfect_square_divisors_of_450 :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let e1 := 1
  let e2 := 2
  let e3 := 2
  ∃ (count : ℕ), count = 4 ∧
  (∀ (d : ℕ), d ∣ (p1^e1 * p2^e2 * p3^e3) →
    (∀ (pe : ℕ × ℕ), pe ∈ (Multiset.ofList [(p1, e1), (p2, e2), (p3, e3)]).powerset → 
      (∃ (ex1 ex2 ex3: ℕ), ex1 ≤ e1 ∧ ex2 ≤ e2 ∧ ex3 ≤ e3 ∧ ex1 % 2 = 0 ∧ ex2 % 2 = 0 ∧ ex3 % 2 = 0))) :=
sorry

end perfect_square_divisors_of_450_l774_774614


namespace range_of_m_l774_774848

theorem range_of_m (α : ℝ) (m : ℝ) :
  (cos α - sqrt 3 * sin α = (4 * m - 6) / (4 - m)) →
  -1 ≤ m ∧ m ≤ 7 / 3 :=
sorry

end range_of_m_l774_774848


namespace general_terms_a_b_sum_of_first_n_terms_l774_774560

variables (a : ℕ → ℕ) (S : ℕ → ℕ) (b : ℕ → ℕ)

axiom a_pos : ∀ n : ℕ, n ≥ 1 → a n > 0

axiom aS_relation : ∀ n : ℕ, n ≥ 1 → (a n + 1) ^ 2 = 4 * S n

axiom b_conditions : b 1 + b 3 = 30 ∧ b 4 + b 6 = 810

theorem general_terms_a_b :
  (∀ n : ℕ, n ≥ 1 → a n = 2 * n - 1) ∧ (∀ n : ℕ, b n = 3 ^ n) :=
sorry

theorem sum_of_first_n_terms (n : ℕ) :
  let T : ℕ → ℕ := λ n, (-1) ^ n * a n + b n in
  (n % 2 = 0 → T n = n + (3^(n + 1) - 3) / 2) ∧
  (n % 2 = 1 → T n = -n + (3^(n + 1) - 3) / 2) :=
sorry

end general_terms_a_b_sum_of_first_n_terms_l774_774560


namespace perfect_square_factors_450_l774_774628

theorem perfect_square_factors_450 : 
  ∃ n, n = 4 ∧ (∀ k | 450, k ∣ 450 → is_square k) → (∃ m1 m2 m3 : ℕ, 450 = 2^m1 * 3^m2 * 5^m3 ) :=
by sorry

end perfect_square_factors_450_l774_774628


namespace complex_quadrant_l774_774169

theorem complex_quadrant (z : ℂ) (h : z * (1 - 2 * complex.I) = complex.I) : 
  - (2 : ℝ) / 5 < 0 ∧ (1 : ℝ) / 5 > 0 :=
by {
  sorry
}

end complex_quadrant_l774_774169


namespace only_funcA_is_direct_proportion_l774_774411

def isDirectProportion (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x

def funcA (x : ℝ) : ℝ := (1/2) * x
def funcB (x : ℝ) : ℝ := 2 / x
def funcC (x : ℝ) : ℝ := x^2
def funcD (x : ℝ) : ℝ := 2 * x - 1

theorem only_funcA_is_direct_proportion : 
  isDirectProportion funcA ∧ 
  ¬ isDirectProportion funcB ∧ 
  ¬ isDirectProportion funcC ∧ 
  ¬ isDirectProportion funcD := 
by {
  sorry
}

end only_funcA_is_direct_proportion_l774_774411


namespace number_of_perfect_square_factors_l774_774659

def is_prime_factorization_valid : Bool :=
  let p1 := 2
  let p1_pow := 1
  let p2 := 3
  let p2_pow := 2
  let p3 := 5
  let p3_pow := 2
  (450 = (p1^p1_pow) * (p2^p2_pow) * (p3^p3_pow))
  
def is_perfect_square (n : Nat) : Bool :=
  match n with
  | 1 => true
  | _ => let second_digit := Math.sqrt n * Math.sqrt n;
         second_digit == n

theorem number_of_perfect_square_factors : (number_of_perfect_square_factors 450 = 4) :=
by sorry

end number_of_perfect_square_factors_l774_774659


namespace radius_of_small_semicircle_l774_774731

theorem radius_of_small_semicircle
  (radius_big_semicircle : ℝ)
  (radius_inner_circle : ℝ) 
  (pairwise_tangent : ∀ x : ℝ, x = radius_big_semicircle → x = 12 ∧ 
                                x = radius_inner_circle → x = 6 ∧ 
                                true) :
  ∃ (r : ℝ), r = 4 :=
by 
  sorry

end radius_of_small_semicircle_l774_774731


namespace sum_reciprocal_sin_to_cot_difference_l774_774318

theorem sum_reciprocal_sin_to_cot_difference (n : ℕ) (hn : n ≠ 0)
  (x : ℝ) (h : ∀ k : ℕ, k ≤ n → x ≠ (N : ℤ) * Real.pi / 2^k) :
  (∑ k in Finset.range n, 1 / Real.sin (2^k * x)) = Real.cot x - Real.cot (2^n * x) :=
sorry

end sum_reciprocal_sin_to_cot_difference_l774_774318


namespace reflection_line_l774_774819

theorem reflection_line (m b : ℝ) :
  let p1 := (2 : ℝ, -2 : ℝ)
  let p2 := (8 : ℝ, 4 : ℝ)
  (∃ m b : ℝ, ∀ (x y : ℝ), (2 * (mx + y - b) = x - 2) ∧ (2 * (my + x - b) = y + 2)) →
  m + b = 5 :=
sorry

end reflection_line_l774_774819


namespace boys_girls_relationship_l774_774247

-- Definitions of boys (b) and girls (g)
def first_five_girls_sum : ℕ := 7 + 9 + 11 + 13 + 15
def extra_girls_per_boy : ℕ := 17
def boys_up_to_five (i : ℕ) : ℕ := 7 + 2 * (i - 1)
def girls_for_boys (b : ℕ) : ℕ :=
  if b ≤ 5 then ∑ i in Finset.range (b + 1), boys_up_to_five i
  else first_five_girls_sum + extra_girls_per_boy * (b - 5)

theorem boys_girls_relationship (b g : ℕ) : 
  g = girls_for_boys b ↔ b = (g + 30) / 17 := 
by
  sorry

end boys_girls_relationship_l774_774247


namespace prove_a_l774_774582

noncomputable def f (x : ℝ) : ℝ

theorem prove_a (a : ℝ) (h : f a + f 1 = 0) : a = -3 := 
by sorry

end prove_a_l774_774582


namespace perfect_square_factors_count_450_l774_774669

theorem perfect_square_factors_count_450 : 
  (∃ n : ℕ, n = 4 ∧ (∀ d : ℕ, d ∣ 450 → ∃ k : ℕ, d = k * k)) := sorry

end perfect_square_factors_count_450_l774_774669


namespace number_of_perfect_square_factors_l774_774664

def is_prime_factorization_valid : Bool :=
  let p1 := 2
  let p1_pow := 1
  let p2 := 3
  let p2_pow := 2
  let p3 := 5
  let p3_pow := 2
  (450 = (p1^p1_pow) * (p2^p2_pow) * (p3^p3_pow))
  
def is_perfect_square (n : Nat) : Bool :=
  match n with
  | 1 => true
  | _ => let second_digit := Math.sqrt n * Math.sqrt n;
         second_digit == n

theorem number_of_perfect_square_factors : (number_of_perfect_square_factors 450 = 4) :=
by sorry

end number_of_perfect_square_factors_l774_774664


namespace max_cups_destroyed_l774_774379

-- Defining the initial condition
def initial_cups : ℕ := 10
def initial_pebbles : ℕ := 10
def A_strategy : (list ℕ) → (list ℕ) := sorry -- Placeholder for the strategy function
def B_action (cup_state : list ℕ) : list ℕ := sorry -- Placeholder for B's action

-- Defining the main theorem to be proved
theorem max_cups_destroyed (n : ℕ) : n ≤ initial_cups - 4 → ∃ (A_strategy : (list ℕ) → (list ℕ)) (B_action : (list ℕ) → (list ℕ)), 
  ∀ (initial_state : list ℕ) (rounds : ℕ),
    (length initial_state = initial_cups) ∧ (∀ k, k < initial_cups → nth initial_state k = initial_pebbles) →
    (∀ r (current_state : list ℕ), r < rounds → current_state = B_action (A_strategy current_state)) →
    (length (nth rounds initial_state) = initial_cups - n) :=
sorry -- Proof placeholder

end max_cups_destroyed_l774_774379


namespace intersecting_chords_l774_774239

theorem intersecting_chords 
  (O P A B C D : Type)
  (AP PB CP PD : ℝ)
  (AP_length : AP = 3)
  (PB_length : PB = 4)
  (CP_length : CP = 5)
  (right_angle : ∠ APB = 90) :
  CP * PD = 12 / 5 :=
by
  sorry

end intersecting_chords_l774_774239


namespace area_of_tangent_segments_l774_774168

theorem area_of_tangent_segments (r : ℝ) (segment_length : ℝ) (pi_val : ℝ) 
  (h1 : r = 3) 
  (h2 : segment_length = 4) 
  (h3 : pi_val = Real.pi) :
  let inner_radius := r,
      outer_radius := Real.sqrt (r^2 + (segment_length / 2)^2),
      area := pi_val * outer_radius^2 - pi_val * inner_radius^2 in
      area = 4 * pi_val := by
  sorry

end area_of_tangent_segments_l774_774168


namespace draw_points_value_l774_774238

theorem draw_points_value
  (D : ℕ) -- Let D be the number of points for a draw
  (victory_points : ℕ := 3) -- points for a victory
  (defeat_points : ℕ := 0) -- points for a defeat
  (total_matches : ℕ := 20) -- total matches
  (points_after_5_games : ℕ := 8) -- points scored in the first 5 games
  (minimum_wins_remaining : ℕ := 9) -- at least 9 matches should be won in the remaining matches
  (target_points : ℕ := 40) : -- target points by the end of the tournament
  D = 1 := 
by 
  sorry


end draw_points_value_l774_774238


namespace range_a_l774_774297

variable {a x : ℝ}

def p (a x : ℝ) := x^2 - 4 * a * x + 3 * a^2 < 0
def q (x : ℝ) := x^2 + 2 * x - 8 > 0

theorem range_a (a : ℝ) :
  (¬ ∀ x, p a x → q x) → ¬ q ∅ → (a ≤ -4 ∨ a ≥ 2 ∨ a = 0) :=
by
  sorry

end range_a_l774_774297


namespace consecutive_negative_product_sum_l774_774362

theorem consecutive_negative_product_sum (n : ℤ) (h : n * (n + 1) = 2850) : n + (n + 1) = -107 :=
sorry

end consecutive_negative_product_sum_l774_774362


namespace grid_at_3_1_is_B_l774_774959

-- Define the problem statement with conditions
def grid_condition (grid: ℕ × ℕ → char) : Prop :=
  (grid (1, 1) = 'A') ∧
  (grid (4, 1) = 'D') ∧
  (grid (5, 1) = 'E') ∧
  (∀ i, i ∈ {1, 2, 3, 4, 5} → ∀ j, j ∈ {1, 2, 3, 4, 5} → 
    {grid (i, j)} = {'A', 'B', 'C', 'D', 'E'}) ∧
  (∀ i, {grid (i, 1), grid (i, 2), grid (i, 3), grid (i, 4), grid (i, 5)} = {'A', 'B', 'C', 'D', 'E'}) ∧
  (∀ j, {grid (1, j), grid (2, j), grid (3, j), grid (4, j), grid (5, j)} = {'A', 'B', 'C', 'D', 'E'}) ∧
  ({grid (1, 1), grid (2, 2), grid (3, 3), grid (4, 4), grid (5, 5)} = {'A', 'B', 'C', 'D', 'E'}) ∧
  ({grid (1, 5), grid (2, 4), grid (3, 3), grid (4, 2), grid (5, 1)} = {'A', 'B', 'C', 'D', 'E'})

theorem grid_at_3_1_is_B (grid: ℕ × ℕ → char) (h: grid_condition grid) : grid (3, 1) = 'B' :=
  sorry

end grid_at_3_1_is_B_l774_774959


namespace geometric_sequence_angles_count_l774_774507

theorem geometric_sequence_angles_count :
  ∃ (θs : Finset ℝ), (∀ θ ∈ θs, 0 < θ ∧ θ < 2 * Real.pi ∧ θ % (Real.pi / 2) ≠ 0) ∧
                     (∀ θ ∈ θs, 
                        ∃ r : ℝ, r ≠ 1 ∧ r ≠ -1 ∧ 
                        ((sin θ = r * cos θ ∧ cos θ = r * tan θ) ∨ 
                         (sin θ = r * tan θ ∧ tan θ = r * cos θ))) ∧
                     θs.card = 4 :=
by
  sorry

end geometric_sequence_angles_count_l774_774507


namespace students_represent_x_percent_of_boys_l774_774895

def number_of_boys (total_population : ℕ) (boy_percentage : ℝ) : ℕ :=
  (boy_percentage * total_population).to_nat

def students_representing_x_percent (x : ℝ) (number_of_boys : ℕ) : ℝ :=
  (x / 100) * number_of_boys

theorem students_represent_x_percent_of_boys (x : ℝ) :
  students_representing_x_percent x (number_of_boys 150 0.4) = (x / 100) * 60 :=
by
  sorry

#eval students_represent_x_percent_of_boys 20 -- Example: for x = 20, it should evaluate to 12.

end students_represent_x_percent_of_boys_l774_774895


namespace min_value_of_A_ge_3_l774_774552

variable {x y z : ℝ} (hx : 0 < x ∧ x ≤ 1) (hy : 0 < y ∧ y ≤ 1) (hz : 0 < z ∧ z ≤ 1)

def A (x y z : ℝ) := (x + 2 * y) * Real.sqrt (x + y - x * y) + (y + 2 * z) * Real.sqrt (y + z - y * z) + (z + 2 * x) * Real.sqrt (z + x - z * x) / (x * y + y * z + z * x)

theorem min_value_of_A_ge_3 (hx : 0 < x ∧ x ≤ 1) (hy : 0 < y ∧ y ≤ 1) (hz : 0 < z ∧ z ≤ 1) :
  A x y z ≥ 3 := by
  sorry

end min_value_of_A_ge_3_l774_774552


namespace pieces_equality_l774_774124

-- Define the pieces of chocolate and their areas.
def piece1_area : ℝ := 6 -- Area of triangle EBC
def piece2_area : ℝ := 6 -- Area of triangle AEC
def piece3_area : ℝ := 6 -- Area of polygon AHGFD
def piece4_area : ℝ := 6 -- Area of polygon CFGH

-- State the problem: proving the equality of the areas.
theorem pieces_equality : piece1_area = piece2_area ∧ piece2_area = piece3_area ∧ piece3_area = piece4_area :=
by
  sorry

end pieces_equality_l774_774124


namespace vins_distance_to_school_l774_774015

-- Defining the distance to school and other necessary distances
variable (x : ℕ) -- Distance to school
constant distance_home : ℕ := 7 -- Distance home via a different route
constant total_distance_per_round_trip : ℕ := x + distance_home
constant total_round_trips : ℕ := 5
constant total_distance : ℕ := 65 -- Total distance ridden in the week

-- The theorem we wish to prove
theorem vins_distance_to_school : 5 * (x + distance_home) = total_distance → x = 6 :=
begin
  intro h,
  -- The proof is omitted
  sorry
end

end vins_distance_to_school_l774_774015


namespace total_flowers_l774_774381

def tulips : ℕ := 3
def carnations : ℕ := 4

theorem total_flowers : tulips + carnations = 7 := by
  sorry

end total_flowers_l774_774381


namespace perfect_square_factors_450_l774_774623

theorem perfect_square_factors_450 : 
  ∃ n, n = 4 ∧ (∀ k | 450, k ∣ 450 → is_square k) → (∃ m1 m2 m3 : ℕ, 450 = 2^m1 * 3^m2 * 5^m3 ) :=
by sorry

end perfect_square_factors_450_l774_774623


namespace number_of_perfect_square_factors_l774_774662

def is_prime_factorization_valid : Bool :=
  let p1 := 2
  let p1_pow := 1
  let p2 := 3
  let p2_pow := 2
  let p3 := 5
  let p3_pow := 2
  (450 = (p1^p1_pow) * (p2^p2_pow) * (p3^p3_pow))
  
def is_perfect_square (n : Nat) : Bool :=
  match n with
  | 1 => true
  | _ => let second_digit := Math.sqrt n * Math.sqrt n;
         second_digit == n

theorem number_of_perfect_square_factors : (number_of_perfect_square_factors 450 = 4) :=
by sorry

end number_of_perfect_square_factors_l774_774662


namespace sales_in_fourth_month_l774_774454

theorem sales_in_fourth_month 
  (s1 s2 s3 s5 s6 avg : ℝ) 
  (h_s1 : s1 = 8435) 
  (h_s2 : s2 = 8927) 
  (h_s3 : s3 = 8855) 
  (h_s5 : s5 = 8562) 
  (h_s6 : s6 = 6991)
  (h_avg : avg = 8500) :
  (6 * avg - (s1 + s2 + s3 + s5 + s6) = 9230) := 
by
  rw [h_avg, h_s1, h_s2, h_s3, h_s5, h_s6]
  simp
  rw [mul_comm 6 8500, ← add_assoc, mul_comm 8500 6]
  norm_num

end sales_in_fourth_month_l774_774454


namespace average_after_31st_inning_l774_774892

-- Define the conditions as Lean definitions
def initial_average (A : ℝ) := A

def total_runs_before_31st_inning (A : ℝ) := 30 * A

def score_in_31st_inning := 105

def new_average (A : ℝ) := A + 3

def total_runs_after_31st_inning (A : ℝ) := total_runs_before_31st_inning A + score_in_31st_inning

-- Define the statement to prove the batsman's average after the 31st inning is 15
theorem average_after_31st_inning (A : ℝ) : total_runs_after_31st_inning A = 31 * (new_average A) → new_average A = 15 := by
  sorry

end average_after_31st_inning_l774_774892


namespace find_ivans_number_l774_774159

theorem find_ivans_number :
  ∃ (a b c d e f g h i j k l : ℕ),
    10 ≤ a ∧ a < 100 ∧
    10 ≤ b ∧ b < 100 ∧
    10 ≤ c ∧ c < 100 ∧
    10 ≤ d ∧ d < 100 ∧
    1000 ≤ e ∧ e < 10000 ∧
    (a * 10^10 + b * 10^8 + c * 10^6 + d * 10^4 + e) = 132040530321 := sorry

end find_ivans_number_l774_774159


namespace Andrew_stamps_permits_today_l774_774480

def Andrew_stamps (total_workday_hours : ℝ) (appointment_hours : ℝ) 
    (meeting_hours : ℝ) (email_hours : ℝ) (stamping_rate : ℝ) : ℕ :=
  let remaining_hours := total_workday_hours - (appointment_hours + meeting_hours + email_hours)
  let permits := remaining_hours * stamping_rate
  permits.to_nat -- rounding down to the nearest integer

theorem Andrew_stamps_permits_today :
  Andrew_stamps 8 6 0.5 0.75 50 = 37 := by
  -- Proof logic to show that the calculated permits today is 37
  sorry

end Andrew_stamps_permits_today_l774_774480


namespace blue_tetrahedron_volume_l774_774057

theorem blue_tetrahedron_volume (s : ℝ) (H : s = 8) :
  let V := s^3 in
  let tetrahedron_volume := V / 3 in
  tetrahedron_volume = 512 / 3 :=
by
  sorry

end blue_tetrahedron_volume_l774_774057


namespace angle_A_is_36_degrees_l774_774724

-- Define the points and segments with necessary conditions
variables {A B C D : Type} [EuclideanGeometry A B C D] 
-- Define the equality conditions for segments
variable (h1 : AB = AC)
variable (h2 : D ∈ segment A C)
variable (h3 : isAngleBisector BD (angle B A C))
variable (h4 : BD = BC)

-- Define the main theorem
theorem angle_A_is_36_degrees :
  angle A = 36 :=
by
  sorry

end angle_A_is_36_degrees_l774_774724


namespace least_total_number_of_bananas_l774_774846

noncomputable def monkey_bananas (b1 b2 b3 : ℕ) (k : ℕ) :=
  (1 / 2) * b1 + (1 / 12) * b2 + (3 / 32) * b3 = 4 * k ∧
  (1 / 6) * b1 + (2 / 3) * b2 + (3 / 32) * b3 = 3 * k ∧
  (1 / 6) * b1 + (1 / 12) * b2 + (3 / 4) * b3 = 2 * k

theorem least_total_number_of_bananas : 
  ∃ b1 b2 b3 k : ℕ, monkey_bananas b1 b2 b3 k ∧ b1 + b2 + b3 = 148 :=
begin
  sorry
end

end least_total_number_of_bananas_l774_774846


namespace value_of_x_l774_774213

theorem value_of_x : (∃ x : ℝ, (1 / 8) * 2 ^ 36 = 8 ^ x) → x = 11 := by
  intro h
  rcases h with ⟨x, hx⟩
  have h1 : 1 / 8 = 2 ^ (-3) := by norm_num
  rw [h1, ←pow_add] at hx
  norm_num at hx
  have h2 : 8 = 2 ^ 3 := by norm_num
  rw [h2, pow_mul] at hx
  norm_num at hx
  exact hx.symm

end value_of_x_l774_774213


namespace find_unit_prices_l774_774086

-- Define the prices of brush and chess set
variables (x y : ℝ)

-- Condition 1: Buying 5 brushes and 12 chess sets costs 315 yuan
def condition1 : Prop := 5 * x + 12 * y = 315

-- Condition 2: Buying 8 brushes and 6 chess sets costs 240 yuan
def condition2 : Prop := 8 * x + 6 * y = 240

-- Prove that the unit price of each brush is 15 yuan and each chess set is 20 yuan
theorem find_unit_prices (hx : condition1 x y) (hy : condition2 x y) :
  x = 15 ∧ y = 20 := 
sorry

end find_unit_prices_l774_774086


namespace concyclic_points_l774_774433

noncomputable def orthocenter (A B C : Point) : Point := sorry
noncomputable def midpoint (P Q : Point) : Point := sorry
def is_cyclic (P Q R S : Point) : Prop := sorry

variables (A B C D E F H A1 B1 C1 A2 B2 C2 : Point)

axioms
  (h_acute_triangle : is_triangle A B C)
  (h_altitudes : are_altitudes A B C AD BE CF)
  (h_orthocenter : H = orthocenter A B C)
  (h_A1 : A1 ∈ ray AD ∧ distance A A1 = distance H D)
  (h_B1 : B1 ∈ ray BE ∧ distance B B1 = distance H E)
  (h_C1 : C1 ∈ ray CF ∧ distance C C1 = distance H F)
  (h_midpoints : A2 = midpoint A1 D ∧ B2 = midpoint B1 E ∧ C2 = midpoint C1 F)

theorem concyclic_points : is_cyclic H A2 B2 C2 := sorry

end concyclic_points_l774_774433


namespace range_of_a_l774_774567

theorem range_of_a
  (f g : ℝ → ℝ)
  (even_fff : ∀ x, f (-x) = f x)
  (odd_g : ∀ x, g (-x) = -g x)
  (eqn1 : ∀ x, f x + g x = 2^(x+1))
  (h : ℝ → ℝ := λ x, a * f(2 * x) + g x)
  (cond : ∀ x₁ x₂ ∈ Icc (0 : ℝ) 1, abs (h x₁ - h x₂) ≤ 25 / 8) :
  -2 ≤ a ∧ a ≤ 13 / 18 := sorry

end range_of_a_l774_774567


namespace complex_point_quadrant_l774_774257

/-- Determine the quadrant of the point corresponding to a given complex number -/
theorem complex_point_quadrant : (∃ (z : ℂ), z = -2 - 3 * complex.I ∧ z / complex.I = 2 * complex.I - 3 → (complex.re ((-2 - 3 * complex.I) / complex.I), complex.im ((-2 - 3 * complex.I) / complex.I)) ∈ ({ (x, y) | x < 0 ∧ y > 0 })) :=
begin
  sorry
end

end complex_point_quadrant_l774_774257


namespace log_x_64_eq_12_by_13_l774_774223

-- Declaration of conditions
variables (x : ℝ)

-- Given condition
noncomputable def condition := log 8 (5 * x) = 3

-- Target statement to prove
theorem log_x_64_eq_12_by_13 (h : condition x) : log x 64 = 12 / 13 := 
sorry

end log_x_64_eq_12_by_13_l774_774223


namespace min_value_of_f_l774_774521

noncomputable def f (x : ℝ) : ℝ := 2 * real.sqrt x + 1 / x + x^2

theorem min_value_of_f : ∃ x > 0, f x = 4 :=
by
  sorry

end min_value_of_f_l774_774521


namespace number_of_perfect_square_factors_l774_774665

def is_prime_factorization_valid : Bool :=
  let p1 := 2
  let p1_pow := 1
  let p2 := 3
  let p2_pow := 2
  let p3 := 5
  let p3_pow := 2
  (450 = (p1^p1_pow) * (p2^p2_pow) * (p3^p3_pow))
  
def is_perfect_square (n : Nat) : Bool :=
  match n with
  | 1 => true
  | _ => let second_digit := Math.sqrt n * Math.sqrt n;
         second_digit == n

theorem number_of_perfect_square_factors : (number_of_perfect_square_factors 450 = 4) :=
by sorry

end number_of_perfect_square_factors_l774_774665


namespace solve_inequality_l774_774568

def even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

def given_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, x >= 0 → f x = x^3 - 8

theorem solve_inequality (f : ℝ → ℝ) (h_even : even_function f) (h_given : given_function f) :
  {x | f (x - 2) > 0} = {x | x < 0 ∨ x > 4} :=
by
  sorry

end solve_inequality_l774_774568


namespace overall_percentage_change_is_113_point_4_l774_774081

-- Define the conditions
def total_customers_survey_1 := 100
def male_percentage_survey_1 := 60
def respondents_survey_1 := 10
def male_respondents_survey_1 := 5

def total_customers_survey_2 := 80
def male_percentage_survey_2 := 70
def respondents_survey_2 := 16
def male_respondents_survey_2 := 12

def total_customers_survey_3 := 70
def male_percentage_survey_3 := 40
def respondents_survey_3 := 21
def male_respondents_survey_3 := 13

def total_customers_survey_4 := 90
def male_percentage_survey_4 := 50
def respondents_survey_4 := 27
def male_respondents_survey_4 := 8

-- Define the calculated response rates
def original_male_response_rate := (male_respondents_survey_1.toFloat / (total_customers_survey_1 * male_percentage_survey_1 / 100).toFloat) * 100
def final_male_response_rate := (male_respondents_survey_4.toFloat / (total_customers_survey_4 * male_percentage_survey_4 / 100).toFloat) * 100

-- Calculate the percentage change in response rate
def percentage_change := ((final_male_response_rate - original_male_response_rate) / original_male_response_rate) * 100

-- The target theorem 
theorem overall_percentage_change_is_113_point_4 : percentage_change = 113.4 := sorry

end overall_percentage_change_is_113_point_4_l774_774081


namespace quadratic_trinomial_has_two_roots_l774_774924

theorem quadratic_trinomial_has_two_roots
  (a b c : ℝ) (h : b^2 - 4 * a * c > 0) : (2 * (a + b))^2 - 4 * 3 * a * (b + c) > 0 := by
  sorry

end quadratic_trinomial_has_two_roots_l774_774924


namespace convex_polyhedron_space_diagonals_l774_774896

theorem convex_polyhedron_space_diagonals
  (vertices : ℕ)
  (edges : ℕ)
  (faces : ℕ)
  (triangular_faces : ℕ)
  (hexagonal_faces : ℕ)
  (total_faces : faces = triangular_faces + hexagonal_faces)
  (vertices_eq : vertices = 30)
  (edges_eq : edges = 72)
  (triangular_faces_eq : triangular_faces = 32)
  (hexagonal_faces_eq : hexagonal_faces = 12)
  (faces_eq : faces = 44) :
  ((vertices * (vertices - 1)) / 2) - edges - 
  (triangular_faces * 0 + hexagonal_faces * ((6 * (6 - 3)) / 2)) = 255 := by
sorry

end convex_polyhedron_space_diagonals_l774_774896


namespace perfect_square_divisors_count_450_l774_774644

theorem perfect_square_divisors_count_450 :
  let is_divisor (d n : ℕ) := n % d = 0
  let is_perfect_square (n : ℕ) := ∃ m : ℕ, m * m = n
  let prime_factorization_450 := [(2, 1), (3, 2), (5, 2)]
  ∃ (count : ℕ), count = 4 ∧ count = (Finset.filter 
    (λ d, is_perfect_square d ∧ is_divisor d 450)
    (Finset.powersetPrimeFactors prime_factorization_450)).card := sorry

end perfect_square_divisors_count_450_l774_774644


namespace circle_area_l774_774021

theorem circle_area : (diameter : ℝ) (d : diameter = 6) : π * (diameter / 2) ^ 2 = 9 * π :=
by
  sorry

end circle_area_l774_774021


namespace average_weight_section_A_l774_774380

theorem average_weight_section_A (W_A : ℝ) :
  (let section_A_students := 60 in
   let section_B_students := 70 in
   let average_weight_B := 80 in
   let total_students := section_A_students + section_B_students in
   let average_weight_class := 70.77 in
   (60 * W_A + 70 * average_weight_B) / total_students = average_weight_class
  ) → W_A = 59.985 :=
by
  sorry

end average_weight_section_A_l774_774380


namespace new_volume_increased_dimensions_l774_774930

theorem new_volume_increased_dimensions (l w h : ℝ) 
  (h_vol : l * w * h = 5000) 
  (h_sa : l * w + w * h + h * l = 900) 
  (h_sum_edges : l + w + h = 60) : 
  (l + 2) * (w + 2) * (h + 2) = 7048 := 
by 
  sorry

end new_volume_increased_dimensions_l774_774930


namespace triangle_at_most_one_obtuse_l774_774030

theorem triangle_at_most_one_obtuse :
  ∀ {α β γ : ℝ} (h : α + β + γ = π) (hα : 0 < α ∧ α < π) (hβ : 0 < β ∧ β < π) (hγ : 0 < γ ∧ γ < π), 
  (α > π / 2 → β ≤ π / 2 ∧ γ ≤ π / 2) :=
by
  intros α β γ h hα hβ hγ hα_obtuse
  have hβ_obtuse : ¬(β > π / 2) := sorry
  have hγ_obtuse : ¬(γ > π / 2) := sorry
  exact ⟨hβ_obtuse, hγ_obtuse⟩

end triangle_at_most_one_obtuse_l774_774030


namespace bowling_average_change_l774_774906

theorem bowling_average_change (old_avg : ℝ) (wickets_last : ℕ) (runs_last : ℕ) (wickets_before : ℕ)
  (h_old_avg : old_avg = 12.4)
  (h_wickets_last : wickets_last = 8)
  (h_runs_last : runs_last = 26)
  (h_wickets_before : wickets_before = 175) :
  old_avg - ((old_avg * wickets_before + runs_last)/(wickets_before + wickets_last)) = 0.4 :=
by {
  sorry
}

end bowling_average_change_l774_774906


namespace aaron_total_earnings_l774_774470

-- Definitions of the conditions
def monday_hours := 1.5
def tuesday_hours := 65 / 60.0
def wednesday_hours := 3.0
def thursday_hours := 45 / 60.0
def normal_rate := 4
def double_rate := 2 * normal_rate

-- Correct earnings calculation
def monday_earnings := monday_hours * normal_rate
def tuesday_earnings := tuesday_hours * normal_rate
def wednesday_earnings := wednesday_hours * double_rate
def thursday_earnings := thursday_hours * normal_rate
def total_earnings := monday_earnings + tuesday_earnings + wednesday_earnings + thursday_earnings

-- The proof statement
theorem aaron_total_earnings : total_earnings = 37.33 := by
  rw [monday_earnings, tuesday_earnings, wednesday_earnings, thursday_earnings]
  -- Use the correct values directly
  simp [monday_hours, tuesday_hours, wednesday_hours, thursday_hours, normal_rate, double_rate]
  sorry

end aaron_total_earnings_l774_774470


namespace find_radius_of_small_semicircle_l774_774735

noncomputable def radius_of_small_semicircle (R : ℝ) (r : ℝ) :=
  ∀ (x : ℝ),
    (12: ℝ = R) ∧ (6: ℝ = r) →
    (∃ (x: ℝ), R - x + r = sqrt((r + x)^2 - r^2)) →
    x = 4

theorem find_radius_of_small_semicircle : radius_of_small_semicircle 12 6 :=
begin
  unfold radius_of_small_semicircle,
  intro x,
  assume h1 h2,
  cases h2,
  sorry,
end

end find_radius_of_small_semicircle_l774_774735


namespace perfect_square_divisors_of_450_l774_774620

theorem perfect_square_divisors_of_450 :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let e1 := 1
  let e2 := 2
  let e3 := 2
  ∃ (count : ℕ), count = 4 ∧
  (∀ (d : ℕ), d ∣ (p1^e1 * p2^e2 * p3^e3) →
    (∀ (pe : ℕ × ℕ), pe ∈ (Multiset.ofList [(p1, e1), (p2, e2), (p3, e3)]).powerset → 
      (∃ (ex1 ex2 ex3: ℕ), ex1 ≤ e1 ∧ ex2 ≤ e2 ∧ ex3 ≤ e3 ∧ ex1 % 2 = 0 ∧ ex2 % 2 = 0 ∧ ex3 % 2 = 0))) :=
sorry

end perfect_square_divisors_of_450_l774_774620


namespace number_of_perfect_square_divisors_of_450_l774_774634

theorem number_of_perfect_square_divisors_of_450 : 
    let p := 450;
    let factors := [(3, 2), (5, 2), (2, 1)];
    ∃ n, (n = 4 ∧ 
          ∀ (d : ℕ), d ∣ p → 
                     (∃ (a b c : ℕ), d = 2^a * 3^b * 5^c ∧ 
                              (a = 0) ∧ (b = 0 ∨ b = 2) ∧ (c = 0 ∨ c = 2) → 
                              a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 = 0)) :=
    sorry

end number_of_perfect_square_divisors_of_450_l774_774634


namespace reassignment_impossible_l774_774242

-- Define the problem conditions
def rows := 5
def columns := 7
def num_students := 34
def unoccupied_position := (2, 3) -- Zero-indexed to represent the center

-- Function to determine if a move is valid (adjacent)
def is_adjacent (pos1 pos2 : (Nat, Nat)) : Bool := 
  let (x1, y1) := pos1
  let (x2, y2) := pos2
  (abs (x1 - x2) = 1 ∧ y1 = y2) ∨ (abs (y1 - y2) = 1 ∧ x1 = x2)

-- Prove that the reassignment is impossible
theorem reassignment_impossible : 
  ∀ (new_positions : List (Nat × Nat)), 
    (∀ pos ∈ new_positions, is_adjacent pos unoccupied_position → False) → 
    new_positions.length = num_students → 
    False :=
by
  -- Proof omitted
  sorry

end reassignment_impossible_l774_774242


namespace susan_age_indeterminate_l774_774891

-- Definitions and conditions
def james_age_in_15_years : ℕ := 37
def current_james_age : ℕ := james_age_in_15_years - 15
def james_age_8_years_ago : ℕ := current_james_age - 8
def janet_age_8_years_ago : ℕ := james_age_8_years_ago / 2
def current_janet_age : ℕ := janet_age_8_years_ago + 8

-- Problem: Prove that without Janet's age when Susan was born, we cannot determine Susan's age in 5 years.
theorem susan_age_indeterminate (susan_current_age : ℕ) : 
  (∃ janet_age_when_susan_born : ℕ, susan_current_age = current_janet_age - janet_age_when_susan_born) → 
  ¬ (∃ susan_age_in_5_years : ℕ, susan_age_in_5_years = susan_current_age + 5) := 
by
  sorry

end susan_age_indeterminate_l774_774891


namespace length_NR_l774_774467

-- Given conditions
def lengthPQ : ℝ := 8
def midpointN (P Q : ℝ × ℝ) (N : ℝ × ℝ) : Prop := N = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)
def radius := lengthPQ / 2

-- Definitions derived from provided conditions
def midpointT : ℝ × ℝ := (lengthPQ / 2, 0)
def lengthRS := lengthPQ
def midpointU (R S : ℝ × ℝ) : ℝ × ℝ := (R.1, (R.2 + S.2) / 2)

-- The problem to prove
theorem length_NR {P Q R S N T U : ℝ × ℝ} (N_mid: midpointN P Q N)
  (lengthPQ_eq: dist P Q = lengthPQ)
  (R_eq: R = (0, lengthPQ))
  (S_eq: S = (lengthPQ, lengthPQ))
  (N_eq: N = (lengthPQ / 2, radius))
  (T_eq: T = midpointT)
  (U_eq: U = midpointU R S)
  : dist N R = 4 * sqrt 5 :=
by
  sorry

end length_NR_l774_774467


namespace unbounded_sequence_in_range_l774_774980

def f1 (n : ℕ) : ℕ :=
  if n = 1 then 1
  else let prime_factors := Nat.factors n in
       let grouped_factors := prime_factors.groupBy id in
       (grouped_factors.map (λ (l) => (l.head! + 1) ^ (l.length - 1))).foldl (*)

def fm : ℕ → ℕ → ℕ
| 1, n := f1 n
| (m + 1), n := f1 (fm m n)

def is_unbounded_sequence (n : ℕ) : Prop :=
  ∀ N : ℕ, ∃ m : ℕ, fm m n > N

def count_interesting_numbers (N : ℕ) : ℕ :=
  ((List.range (N + 1)).filter is_unbounded_sequence).length

theorem unbounded_sequence_in_range : count_interesting_numbers 100 = 4 := 
sorry

end unbounded_sequence_in_range_l774_774980


namespace distinct_complex_solutions_l774_774336

noncomputable def P : Real → Real := λ z, z^4 + 5
noncomputable def Q : Real → Real := λ z, z^3 + 2
noncomputable def S : Real → Real := λ z, z + 1
noncomputable def R : Real → Real := λ z, z^7 + z + 11

theorem distinct_complex_solutions :
  ∃ (N : ℕ), N = 2 ∧
  ∀ z, P(z) * Q(z) + S(z) = R(z) → 
    (z = 0 ∨ z = -5/2) :=
begin
  sorry
end

end distinct_complex_solutions_l774_774336


namespace put_letters_in_mailboxes_l774_774323

theorem put_letters_in_mailboxes :
  (3:ℕ)^4 = 81 :=
by
  sorry

end put_letters_in_mailboxes_l774_774323


namespace perfect_square_factors_450_l774_774652

def prime_factorization (n : ℕ) : Prop :=
  n = 2^1 * 3^2 * 5^2

theorem perfect_square_factors_450 : prime_factorization 450 → ∃ n : ℕ, n = 4 :=
by
  sorry

end perfect_square_factors_450_l774_774652


namespace tangent_min_value_l774_774255

-- Define the trigonometric functions and the problem setting
variable {A B C : ℝ}

-- Declare that the triangles under consideration are acute
hypothesis (hA : 0 < A ∧ A < π / 2)
hypothesis (hB : 0 < B ∧ B < π / 2)
hypothesis (hC : 0 < C ∧ C < π / 2)
hypothesis (hSum : A + B + C = π)

-- Given condition in the problem
hypothesis (hSin : Real.sin A = 2 * Real.sin B * Real.sin C)

-- The main theorem to prove
theorem tangent_min_value : 
  ∃ (A B C : ℝ), (A > 0 ∧ A < π / 2) ∧ (B > 0 ∧ B < π / 2) ∧ (C > 0 ∧ C < π / 2) ∧ A + B + C = π ∧ 
  Real.sin A = 2 * Real.sin B * Real.sin C ∧  
  ∀ (a b c: ℝ), (Real.tan a + 2 * Real.tan b * Real.tan c + Real.tan a * Real.tan b * Real.tan c ≥ 16) :=
sorry

end tangent_min_value_l774_774255


namespace youngest_brother_age_l774_774805

theorem youngest_brother_age (x : ℕ) (h : x + (x + 1) + (x + 2) = 96) : x = 31 :=
sorry

end youngest_brother_age_l774_774805


namespace probability_of_prime_spinner_l774_774466

def is_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7 ∨ n = 11 ∨ n = 13 ∨ n = 17 ∨ n = 19 ∨ n = 23 ∨ n = 29

def spinner_sections : List ℕ := [4, 6, 7, 1, 8, 9, 10, 3]

theorem probability_of_prime_spinner : 
  (spinner_sections.filter is_prime).length.toFloat / spinner_sections.length.toFloat = 1 / 4 := 
by 
  sorry

end probability_of_prime_spinner_l774_774466


namespace radius_of_smaller_semicircle_l774_774737

theorem radius_of_smaller_semicircle :
  ∃ x : ℝ, 0 < x ∧
    let AB := 6 in
    let AC := 12 - x in
    let BC := 6 + x in
    (AB = 6) ∧ 
    (AC = 12 - x) ∧ 
    (BC = 6 + x) ∧
    (AB^2 + AC^2 = BC^2) ∧
    x = 4 := 
by
  use 4
  split
  { exact zero_lt_four }
  split
  { reflexivity }
  split
  { reflexivity }
  split
  { reflexivity }
  { sorry }

end radius_of_smaller_semicircle_l774_774737


namespace steamboat_instructions_l774_774468

theorem steamboat_instructions (sailors: Fin 26 → ℤ) (instruction_sum: ℤ) :
  instruction_sum = 2017 ^ 2017 →
  (∀ i : Fin 25, sailors i.succ = sailors i + 2 ∨ sailors i.succ = sailors i - 2) →
  False :=
begin
  assume h1 h2,
  sorry
end

end steamboat_instructions_l774_774468


namespace scientists_count_l774_774438

theorem scientists_count (S W N N_p : ℕ) 
  (h1 : W = 31) 
  (h2 : W_p = 14)
  (h3 : N_p = 25 - W_p)
  (h4 : N_p = N + 3)
  (h5 : N_p = 11) 
  (h6 : N = 8)
  : S = W + N_p + N := by
  unfold W
  unfold N_p
  unfold N
  unfold W_p
  -- We would have a full proof here
  sorry

end scientists_count_l774_774438


namespace schedule_lecturers_l774_774064

-- Conditions
variables (Lecturers : Type) [Fintype Lecturers] [DecidableEq Lecturers]
variables (drBlair drAdams drChen : Lecturers)
variable (otherLecturers : Finset Lecturers)
variable (h_distinct : ∀ l ∈ insert drBlair (insert drAdams (insert drChen otherLecturers.toFinset)), l ≠ drAdams → l ≠ drChen → l ≠ drBlair)

-- Question and correct answer
theorem schedule_lecturers 
  (h_size : otherLecturers.card = 4) 
  (h_allLecturers : insert drBlair (insert drAdams (insert drChen otherLecturers.toFinset)) = Finset.univ) 
  (h_condition : ∀ (σ : Perm (Fin 7)), ∃ i j k, i < j ∧ j < k ∧ σ i = drAdams ∧ σ j = drChen ∧ σ k = drBlair): 
  Fintype.card (Perm (Fin 7)) = 5760 :=
sorry

end schedule_lecturers_l774_774064


namespace solve_oranges_problem_find_plans_and_max_profit_l774_774955

theorem solve_oranges_problem :
  ∃ (a b : ℕ), 15 * a + 20 * b = 430 ∧ 10 * a + 8 * b = 212 ∧ a = 10 ∧ b = 14 := by
    sorry

theorem find_plans_and_max_profit (a b : ℕ) (h₁ : 15 * a + 20 * b = 430) (h₂ : 10 * a + 8 * b = 212) (ha : a = 10) (hb : b = 14) :
  ∃ (x : ℕ), 58 ≤ x ∧ x ≤ 60 ∧ (10 * x + 14 * (100 - x) ≥ 1160) ∧ (10 * x + 14 * (100 - x) ≤ 1168) ∧ (1000 - 4 * x = 768) :=
    sorry

end solve_oranges_problem_find_plans_and_max_profit_l774_774955


namespace max_rel_prime_composite_l774_774843

def is_composite (n : ℕ) : Prop := ∃ m k, 1 < m ∧ 1 < k ∧ n = m * k
def are_rel_prime (a b : ℕ) : Prop := Nat.gcd a b = 1

theorem max_rel_prime_composite (A : Finset ℕ) (h1 : ∀ x ∈ A, 10 ≤ x ∧ x ≤ 99)
  (h2 : ∀ x ∈ A, is_composite x) (h3 : ∀ x y ∈ A, x ≠ y → are_rel_prime x y) : A.card ≤ 4 := 
sorry

end max_rel_prime_composite_l774_774843


namespace f_f_1_plus_i_eq_3_l774_774171

-- Define the function f as specified
def f (x : ℂ) : ℂ :=
  if x.im = 0 then 1 + x else ((1 - complex.i) / complex.abs complex.i) * x

-- State the theorem to be proved
theorem f_f_1_plus_i_eq_3 : f (f (1 + complex.i)) = 3 := by
  sorry

end f_f_1_plus_i_eq_3_l774_774171


namespace total_value_proof_l774_774444

def total_bills : ℕ := 126
def five_dollar_bills : ℕ := 84
def ten_dollar_bills : ℕ := total_bills - five_dollar_bills
def value_five_dollar_bills : ℕ := five_dollar_bills * 5
def value_ten_dollar_bills : ℕ := ten_dollar_bills * 10
def total_value : ℕ := value_five_dollar_bills + value_ten_dollar_bills

theorem total_value_proof : total_value = 840 := by
  unfold total_value value_five_dollar_bills value_ten_dollar_bills
  unfold five_dollar_bills ten_dollar_bills total_bills
  -- Calculation steps to show that value_five_dollar_bills + value_ten_dollar_bills = 840
  sorry

end total_value_proof_l774_774444


namespace find_special_5_digit_number_l774_774599

theorem find_special_5_digit_number :
  ∃! (A : ℤ), (10000 ≤ A ∧ A < 100000) ∧ (A^2 % 100000 = A) ∧ A = 90625 :=
sorry

end find_special_5_digit_number_l774_774599


namespace smaller_octagon_area_fraction_l774_774838

theorem smaller_octagon_area_fraction {A B C D E F G H : Point} (h_reg_octagon : regular_octagon A B C D E F G H) :
  let smaller_octagon := midpoints_octagon A B C D E F G H in
  area smaller_octagon = (1/4) * area (octagon A B C D E F G H) :=
by sorry

end smaller_octagon_area_fraction_l774_774838


namespace max_points_each_player_l774_774393

-- Definitions for the problem conditions
def first_player_cards : List Nat := List.range' 2 1000 (by norm_num) |>.map (λ i => 2 * i)
def second_player_cards : List Nat := List.range' 1 1001 (by norm_num) |>.map (λ i => 2 * i - 1)

-- The main theorem statement
theorem max_points_each_player :
  ∃ (fp_points sp_points : Nat),
    (fp_points = 499) ∧ (sp_points = 501) ∧
    ∀ (fp_strategy sp_strategy : List Nat → Nat),
      (∀ (fp_cards sp_cards : List Nat) (n_fp_cards n_sp_cards : Nat),
         fp_cards ⊆ first_player_cards →
         sp_cards ⊆ second_player_cards →
         fp_cards.length = n_fp_cards →
         sp_cards.length = n_sp_cards →
         n_fp_cards + n_sp_cards = 2000 →
         ∃ fp_points sp_points, fp_points + sp_points = 1000) :=
begin
  -- Initial variables
  existsi 499,
  existsi 501,

  -- Proof of guaranteed points
  split,
  sorry,
  split,
  sorry,
  intros fp_strategy sp_strategy fp_cards sp_cards n_fp_cards n_sp_cards,
  sorry
end

end max_points_each_player_l774_774393


namespace number_of_perfect_square_divisors_of_450_l774_774636

theorem number_of_perfect_square_divisors_of_450 : 
    let p := 450;
    let factors := [(3, 2), (5, 2), (2, 1)];
    ∃ n, (n = 4 ∧ 
          ∀ (d : ℕ), d ∣ p → 
                     (∃ (a b c : ℕ), d = 2^a * 3^b * 5^c ∧ 
                              (a = 0) ∧ (b = 0 ∨ b = 2) ∧ (c = 0 ∨ c = 2) → 
                              a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 = 0)) :=
    sorry

end number_of_perfect_square_divisors_of_450_l774_774636


namespace michael_num_dogs_l774_774780

variable (total_cost : ℕ)
variable (cost_per_animal : ℕ)
variable (num_cats : ℕ)
variable (num_dogs : ℕ)

-- Conditions
def michael_total_cost := total_cost = 65
def michael_num_cats := num_cats = 2
def michael_cost_per_animal := cost_per_animal = 13

-- Theorem to prove
theorem michael_num_dogs (h_total_cost : michael_total_cost total_cost)
                         (h_num_cats : michael_num_cats num_cats)
                         (h_cost_per_animal : michael_cost_per_animal cost_per_animal) :
  num_dogs = 3 :=
by
  sorry

end michael_num_dogs_l774_774780


namespace greatest_common_divisor_546_180_l774_774020

theorem greatest_common_divisor_546_180 : 
  ∃ d, d < 70 ∧ d > 0 ∧ d ∣ 546 ∧ d ∣ 180 ∧ ∀ x, x < 70 ∧ x > 0 ∧ x ∣ 546 ∧ x ∣ 180 → x ≤ d → x = 6 :=
by
  sorry

end greatest_common_divisor_546_180_l774_774020


namespace reciprocal_neg_three_half_l774_774831

theorem reciprocal_neg_three_half : (-3 - 1 / 2)⁻¹ = -2 / 7 :=
by
  sorry

end reciprocal_neg_three_half_l774_774831


namespace perfect_square_factors_450_l774_774658

def prime_factorization (n : ℕ) : Prop :=
  n = 2^1 * 3^2 * 5^2

theorem perfect_square_factors_450 : prime_factorization 450 → ∃ n : ℕ, n = 4 :=
by
  sorry

end perfect_square_factors_450_l774_774658


namespace part1_part2_l774_774205

namespace VectorProblems

open Real

-- Definition of the vectors
def m (a θ : ℝ) : ℝ × ℝ := (a - sin θ, -1 / 2)
def n (θ : ℝ) : ℝ × ℝ := (1 / 2, cos θ)

-- Statements for the two parts of the problem

-- Part 1: Perpendicular vectors and \(\sin 2\theta\)
theorem part1 (θ : ℝ) : m (sqrt 2 / 2) θ ⋅ n θ = 0 → sin (2 * θ) = -1 / 2 :=
by
  sorry

-- Part 2: Parallel vectors and \(\tan\theta\)
theorem part2 (θ : ℝ) : (∃ k : ℝ, m 0 θ = k • n θ) → tan θ = 2 + sqrt 3 ∨ tan θ = 2 - sqrt 3 :=
by
  sorry

end VectorProblems

end part1_part2_l774_774205


namespace area_of_triangle_formed_by_tangent_line_l774_774806

def curve (x : ℝ) : ℝ := Real.exp (x / 3)

def point_on_curve : ℝ × ℝ := (6, Real.exp 2)

noncomputable def tangent_line (x : ℝ) : ℝ :=
  (1 / 3) * Real.exp 2 * x - Real.exp 2

def x_intercept : ℝ := 3
def y_intercept : ℝ := -Real.exp 2

def triangle_area : ℝ := (1 / 2) * x_intercept * Real.abs y_intercept

theorem area_of_triangle_formed_by_tangent_line :
  triangle_area = (3 / 2) * Real.exp 2 := by
  sorry

end area_of_triangle_formed_by_tangent_line_l774_774806


namespace radius_of_small_semicircle_l774_774729

theorem radius_of_small_semicircle
  (radius_big_semicircle : ℝ)
  (radius_inner_circle : ℝ) 
  (pairwise_tangent : ∀ x : ℝ, x = radius_big_semicircle → x = 12 ∧ 
                                x = radius_inner_circle → x = 6 ∧ 
                                true) :
  ∃ (r : ℝ), r = 4 :=
by 
  sorry

end radius_of_small_semicircle_l774_774729


namespace wheel_rpm_l774_774364

noncomputable def radius_of_wheel := 70 -- in cm
noncomputable def speed_of_bus := 66 -- in km/h

noncomputable def revolutions_per_minute (r : ℝ) (v : ℝ) : ℝ :=
  let cm_per_km := 100000
  let min_per_hour := 60
  let speed_cm_per_min := (v * cm_per_km) / min_per_hour
  let circumference := 2 * Real.pi * r
  speed_cm_per_min / circumference

theorem wheel_rpm :
  revolutions_per_minute radius_of_wheel speed_of_bus ≈ 250.11 := by
  sorry

end wheel_rpm_l774_774364


namespace polynomial_no_value_1996_irrational_exponents_l774_774048

theorem polynomial_no_value_1996 (P : ℤ[X]) (a1 a2 a3 a4 a5 a6 a7 : ℤ) (h1 : P.eval a1 = -2) (h2 : P.eval a2 = -2) (h3 : P.eval a3 = -2) (h4 : P.eval a4 = -2) (h5 : P.eval a5 = -2) (h6 : P.eval a6 = -2) (h7 : P.eval a7 = -2) :
  ∀ x : ℤ, P.eval x ≠ 1996 :=
sorry

theorem irrational_exponents : ∃ (x y : ℝ), (¬ rational x) ∧ (¬ rational y) ∧ (rational (x^y)) :=
sorry

end polynomial_no_value_1996_irrational_exponents_l774_774048


namespace minimum_n_94_pos_int_sum_2016_partition_32_63_l774_774578

theorem minimum_n_94_pos_int_sum_2016_partition_32_63 :
  ∃ n : ℕ, (∃ x : fin n → ℕ, (finset.univ.sum x = 2016)
  ∧ (∃ (partition1 : fin n → fin 32 → ℕ) (partition2 : fin n → fin 63 → ℕ),
       (∀ i : fin 32, finset.univ.sum (λ j, partition1 j i) = 63) 
     ∧ (∀ j : fin 63, finset.univ.sum (λ i, partition2 i j) = 32))
  ∧ n = 94) :=
begin
  -- Proof omitted
  sorry
end

end minimum_n_94_pos_int_sum_2016_partition_32_63_l774_774578


namespace find_value_of_r_l774_774058

theorem find_value_of_r (a r : ℝ) (h1 : a / (1 - r) = 20) (h2 : a * r / (1 - r^2) = 8) : r = 2 / 3 :=
by
  sorry

end find_value_of_r_l774_774058


namespace element_exists_distinct_l774_774277

variable (A : Type) [Fintype A] (n : ℕ) (A_1 A_2 ... A_n : Finset A)
hypothesis (H1 : ∀ i j, i ≠ j → A_i ≠ A_j)

theorem element_exists_distinct :
  ∃ a ∈ A, ∀ i j, i ≠ j → (A_i \ {a}) ≠ (A_j \ {a}) := by
  sorry

end element_exists_distinct_l774_774277


namespace shepherd_A_sheep_count_eq_seven_l774_774842

noncomputable def number_of_sheep_B : ℕ := 5

theorem shepherd_A_sheep_count_eq_seven :
  (let A_sheep := number_of_sheep_B + 2 in A_sheep = 7) :=
by
  let A_sheep := number_of_sheep_B + 2
  exact rfl

end shepherd_A_sheep_count_eq_seven_l774_774842


namespace point_inside_circle_l774_774179

-- Definitions for the conditions of the problem
variables (a b : ℝ)
def is_root (a b : ℝ) (x : ℝ) : Prop := x^2 - x - real.sqrt 2 = 0
def distinct_roots (a b : ℝ) : Prop := a ≠ b ∧ is_root a b a ∧ is_root a b b

-- Definition of the circle
def on_circle (x y : ℝ) : Prop := x^2 + y^2 = 8

-- The Lean statement for the math proof problem
theorem point_inside_circle (a b : ℝ) (h : distinct_roots a b) : (a, b).dist (0, 0) < real.sqrt 8 := sorry

end point_inside_circle_l774_774179


namespace sum_T_eq_l774_774972

-- Define the summation in the problem
def sum_T : ℤ := ∑ k in Finset.range 25, (-1) ^ k * Nat.choose 49 (2 * k)

-- State the theorem to be proved
theorem sum_T_eq : sum_T = -2^24 := 
  sorry

end sum_T_eq_l774_774972


namespace three_pos_reals_inequality_l774_774770

open Real

theorem three_pos_reals_inequality 
  (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_eq : a + b + c = a^2 + b^2 + c^2) :
  ((a^2) / (a^2 + b * c) + (b^2) / (b^2 + c * a) + (c^2) / (c^2 + a * b)) ≥ (a + b + c) / 2 :=
by
  sorry

end three_pos_reals_inequality_l774_774770


namespace time_for_train_to_pass_jogger_l774_774904

-- Definitions of the conditions
def jogger_speed : Real := 9 -- in kmph
def train_speed : Real := 45 -- in kmph
def distance_ahead_of_jogger : Real := 240 -- in meters
def length_of_train : Real := 120 -- in meters

-- Conversion factor from kmph to m/s
def kmph_to_mps (speed : Real) : Real := speed * (1000 / 3600)

-- The main theorem we want to prove
theorem time_for_train_to_pass_jogger : 
  (distance_ahead_of_jogger + length_of_train) / (kmph_to_mps (train_speed - jogger_speed)) = 36 := 
by
  -- For now, we include a sorry to denote the proof body.
  sorry

end time_for_train_to_pass_jogger_l774_774904


namespace infinitely_many_pairs_l774_774039

theorem infinitely_many_pairs (n : ℕ) : 
  ∃∞ (k l : ℕ), k ≥ 2 ∧ l ≥ 2 ∧ k! * l! = n! := 
sorry

end infinitely_many_pairs_l774_774039


namespace perfect_square_factors_450_l774_774650

def prime_factorization (n : ℕ) : Prop :=
  n = 2^1 * 3^2 * 5^2

theorem perfect_square_factors_450 : prime_factorization 450 → ∃ n : ℕ, n = 4 :=
by
  sorry

end perfect_square_factors_450_l774_774650


namespace number_of_complementary_sets_l774_774515

structure Card where
  shape : Nat
  color : Nat
  shade : Nat
  size : Nat

def isComplementarySet (c1 c2 c3 : Card) : Prop :=
  (c1.shape = c2.shape ∧ c2.shape = c3.shape ∨ c1.shape ≠ c2.shape ∧ c2.shape ≠ c3.shape ∧ c1.shape ≠ c3.shape) ∧
  (c1.color = c2.color ∧ c2.color = c3.color ∨ c1.color ≠ c2.color ∧ c2.color ≠ c3.color ∧ c1.color ≠ c3.color) ∧
  (c1.shade = c2.shade ∧ c2.shade = c3.shade ∨ c1.shade ≠ c2.shade ∧ c2.shade ≠ c3.shade ∧ c1.shade ≠ c3.shade) ∧
  (c1.size = c2.size ∧ c2.size = c3.size ∨ c1.size ≠ c2.size ∧ c2.size ≠ c3.size ∧ c1.size ≠ c3.size)

def totalComplementarySets (deck : List Card) : Nat :=
  (deck.powerset.filter (fun s => s.length = 3 ∧ isComplementarySet s.head (s.tail).head (s.tail.tail).head)).length

theorem number_of_complementary_sets (deck : List Card) (h : deck.length = 81) : totalComplementarySets deck = 6483 :=
  sorry

end number_of_complementary_sets_l774_774515


namespace range_of_m_l774_774577

open Real

noncomputable def x (y : ℝ) : ℝ := 2 / (1 - 1 / y)

theorem range_of_m (y : ℝ) (m : ℝ) (h1 : y > 0) (h2 : 1 - 1 / y > 0) (h3 : -4 < m) (h4 : m < 2) : 
  x y + 2 * y > m^2 + 2 * m := 
by 
  have hx_pos : x y > 0 := sorry
  have hxy_eq : 2 / x y + 1 / y = 1 := sorry
  have hxy_ge : x y + 2 * y ≥ 8 := sorry
  have h_m_le : 8 > m^2 + 2 * m := sorry
  exact sorry

end range_of_m_l774_774577


namespace sum_middle_three_cards_l774_774973

-- Represent cards as sets with their corresponding numbers
def red_cards := {2, 4, 7, 8}
def blue_cards := {6, 8, 9}
def green_cards := {5, 10}

-- Define the positions of the cards in the arrangement
def card_arrangement : List ℕ := [6, 10, 8, 8, 7, 5] -- B1, G2, R4, B2, R7, G1

-- Define a predicate to check the condition of alternating colors
def alternates_colors (arr : List ℕ) : Prop :=
  ∀ (i : ℕ) (h : i < arr.length - 1), 
    let c1 := if i % 2 == 0 then "B" else "R" in
    let c2 := if (i + 1) % 2 == 0 then "B" else "R" in
    c1 ≠ c2

-- Define a predicate to check the arithmetic conditions between adjacent cards
def valid_arithmetic_conditions (arr : List ℕ) : Prop :=
  ∀ (i : ℕ) (h : i < arr.length - 1),
    let a := arr.get ⟨i, h⟩ in
    let b := arr.get ⟨i+1, sorry⟩ in -- i < arr.length - 1 ensures i+1 < arr.length
    (a ∣ b) ∨ (b ∣ a)

-- The final theorem to prove
theorem sum_middle_three_cards : 
  alternates_colors card_arrangement ∧ valid_arithmetic_conditions card_arrangement → 
  card_arrangement.get ⟨2, sorry⟩ + card_arrangement.get ⟨3, sorry⟩ + card_arrangement.get ⟨4, sorry⟩ = 23 :=
  sorry

end sum_middle_three_cards_l774_774973


namespace has_two_roots_l774_774916

-- Define the discriminant of the quadratic trinomial
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Main Lean statement
theorem has_two_roots
  (a b c : ℝ)
  (h : discriminant a b c > 0) :
  discriminant (3 * a) (2 * (a + b)) (b + c) > 0 := by
  sorry

end has_two_roots_l774_774916


namespace ordering_abc_l774_774546

section ordering_abc

def a : ℝ := Real.logb 3 (Real.sqrt 2)
def b : ℝ := 0.3^0.5
def c : ℝ := 0.5^(-0.2)

theorem ordering_abc : a < b ∧ b < c := by
  have h0 : a = Real.logb 3 (Real.sqrt 2) := rfl
  have h1 : b = 0.3^0.5 := rfl
  have h2 : c = 0.5^(-0.2) := rfl
  sorry

end ordering_abc

end ordering_abc_l774_774546


namespace theresa_gave_blocks_l774_774397

theorem theresa_gave_blocks (initial_blocks total_blocks : ℕ) (h_initial : initial_blocks = 4) (h_total : total_blocks = 83) : total_blocks - initial_blocks = 79 :=
by
  rw [h_initial, h_total]
  exact rfl

end theresa_gave_blocks_l774_774397


namespace perfect_square_factors_450_l774_774607

theorem perfect_square_factors_450 :
  ∃ (n : ℕ), n = 4 ∧ ∀ (d : ℕ), d ∣ 450 → ∃ (k : ℕ), k^2 = d ↔ d = 1 ∨ d = 9 ∨ d = 25 ∨ d = 225 :=
by
  sorry

end perfect_square_factors_450_l774_774607


namespace sum_abcd_not_prime_l774_774296

noncomputable def sum_not_prime (a b c d : ℕ) : Prop :=
  a*b = c*d → ¬ prime (a + b + c + d)

-- The theorem to prove
theorem sum_abcd_not_prime (a b c d : ℕ) (h_pos: 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) (h: a * b = c * d) : ¬ prime (a + b + c + d) :=
sorry

end sum_abcd_not_prime_l774_774296


namespace bricklayer_team_size_l774_774942

/-- Problem: Prove the number of bricklayers in the team -/
theorem bricklayer_team_size
  (x : ℕ)
  (h1 : 432 = (432 * (x - 4) / x) + 9 * (x - 4)) :
  x = 16 :=
sorry

end bricklayer_team_size_l774_774942


namespace log2_9_eq_kx_log4_3_l774_774281

theorem log2_9_eq_kx_log4_3 (x k : ℝ) (h1: log 4 3 = x) (h2: log 2 9 = k * x) : k = 4 := 
by {
  sorry
}

end log2_9_eq_kx_log4_3_l774_774281


namespace matrix_power_eigenvector_l774_774775

section MatrixEigen
variable (B : Matrix (Fin 2) (Fin 2) ℝ) (v : Fin 2 → ℝ)

theorem matrix_power_eigenvector (h : B.mulVec ![3, -1] = ![-12, 4]) :
  (B ^ 5).mulVec ![3, -1] = ![-3072, 1024] := 
  sorry
end MatrixEigen

end matrix_power_eigenvector_l774_774775


namespace avg_salary_all_is_8000_l774_774807

-- Definitions based on step a)
def avg_salary_technicians : ℝ := 12000
def avg_salary_non_technicians : ℝ := 6000
def total_workers : ℕ := 21
def num_technicians : ℕ := 7

def num_non_technicians : ℕ := total_workers - num_technicians
def total_salary_technicians : ℝ := num_technicians * avg_salary_technicians
def total_salary_non_technicians : ℝ := num_non_technicians * avg_salary_non_technicians
def total_salary_all_workers : ℝ := total_salary_technicians + total_salary_non_technicians

def avg_salary_all_workers : ℝ := total_salary_all_workers / total_workers

-- The theorem statement proving the average salary for all workers given the conditions
theorem avg_salary_all_is_8000 :
  avg_salary_all_workers = 8000 := 
sorry

end avg_salary_all_is_8000_l774_774807


namespace commodity_price_l774_774813

-- Define the variables for the prices of the commodities.
variable (x y : ℝ)

-- Conditions given in the problem.
def total_cost (x y : ℝ) : Prop := x + y = 827
def price_difference (x y : ℝ) : Prop := x = y + 127

-- The main statement to prove.
theorem commodity_price (x y : ℝ) (h1 : total_cost x y) (h2 : price_difference x y) : x = 477 :=
by
  sorry

end commodity_price_l774_774813


namespace max_area_triangle_hyperbola_l774_774202

noncomputable def area_triangle_max (b : ℝ) (hb : 0 < b ∧ b < 2) : ℝ :=
  let A := (-real.sqrt (4 - b^2), 0)
  let B := (real.sqrt (4 - b^2), 0)
  let C := (0, b)
  let area := (1 / 2) * ((B.1 - A.1) * (C.2 - A.2)) -- Heron's formula
  area

theorem max_area_triangle_hyperbola : ∀ b (hb : 0 < b ∧ b < 2),
  area_triangle_max b hb ≤ 2 :=
by {
  sorry
}

end max_area_triangle_hyperbola_l774_774202


namespace eval_sqrt_4_8_pow_12_l774_774127

theorem eval_sqrt_4_8_pow_12 : ((8 : ℝ)^(1 / 4))^12 = 512 :=
by
  -- This is where the proof steps would go 
  sorry

end eval_sqrt_4_8_pow_12_l774_774127


namespace find_angle_C_find_a_plus_b_l774_774725

-- Problem 1
theorem find_angle_C 
  (m : ℝ × ℝ) 
  (n : ℝ × ℝ) 
  (hm : m = (Real.cos (C / 2), Real.sin (C / 2))) 
  (hn : n = (Real.cos (C / 2), -Real.sin (C / 2))) 
  (angle_mn : ∠ m n = π / 3) : 
  C = π / 3 := 
sorry

-- Problem 2
theorem find_a_plus_b 
  (a b : ℝ) 
  (C : ℝ) 
  (c := 7 / 2) 
  (area := 3 * Real.sqrt 3 / 2) 
  (C_eq : C = π / 3) 
  (area_eq : 1 / 2 * a * b * Real.sin (π / 3) = 3 * Real.sqrt 3 / 2) : 
  a + b = 11 / 2 := 
sorry

end find_angle_C_find_a_plus_b_l774_774725


namespace vector_perpendicular_find_alpha_l774_774302

variable {α : ℝ} (hα : 0 ≤ α ∧ α < 360)
noncomputable def a : ℝ × ℝ := (Real.cos α, Real.sin α)
noncomputable def b : ℝ × ℝ := (-1 / 2, Real.sqrt 3 / 2)

theorem vector_perpendicular 
  (h1 : vectorAdd a b)
  (h2 : vectorSub a b)
  : dotProd (vectorAdd a b) (vectorSub a b) = 0 := sorry

theorem find_alpha 
  (h3 : vectorMag (scalarMul (sqrt 3) a) b = vectorMag a (scalarMul (sqrt 3) b)) 
  : α = 30 ∨ α = 210 := sorry

end vector_perpendicular_find_alpha_l774_774302


namespace number_of_paintings_l774_774440

def is_valid_painting (grid : Matrix (Fin 3) (Fin 3) Bool) : Prop :=
  ∀ i j, grid i j = true → 
    (∀ k, k.succ < 3 → grid k j = true → ¬ grid (k.succ) j = false) ∧
    (∀ l, l.succ < 3 → grid i l = true → ¬ grid i (l.succ) = false)

theorem number_of_paintings : 
  ∃ n, n = 50 ∧ 
       ∃ f : Finset (Matrix (Fin 3) (Fin 3) Bool), 
         (∀ grid ∈ f, is_valid_painting grid) ∧ 
         Finset.card f = n :=
sorry

end number_of_paintings_l774_774440


namespace factorize_expr1_factorize_expr2_l774_774990

-- Define the expressions
def expr1 (m x y : ℝ) : ℝ := 3 * m * x - 6 * m * y
def expr2 (x : ℝ) : ℝ := 1 - 25 * x^2

-- Define the factorized forms
def factorized_expr1 (m x y : ℝ) : ℝ := 3 * m * (x - 2 * y)
def factorized_expr2 (x : ℝ) : ℝ := (1 + 5 * x) * (1 - 5 * x)

-- Proof problems
theorem factorize_expr1 (m x y : ℝ) : expr1 m x y = factorized_expr1 m x y := sorry
theorem factorize_expr2 (x : ℝ) : expr2 x = factorized_expr2 x := sorry

end factorize_expr1_factorize_expr2_l774_774990


namespace china_junior_high_math_league_1989_problem_l774_774960

theorem china_junior_high_math_league_1989_problem 
  (A B C D E : Point) 
  (h_square : square A B C D)
  (h_DE_EC : E = midpoint D C)
  (h_Angle_CDE_60 : ∠ C D E = 60°)
  (angle1 angle2 angle3 angle4 : ℝ)
  (h_Angle1_4_ratio : angle1 / angle4 = 4 / 1)
  (h_Angle1_3_ratio : angle1 / angle3 = 1 / 1)
  (h_Angle_Sum_1_2_3_4_ratio : (angle1 + angle2) / (angle3 + angle4) = 5 / 3) : 
  (h_Angle1_4_ratio ∧ h_Angle_Sum_1_2_3_4_ratio ∧ ¬ h_Angle1_3_ratio) ∨
  (h_Angle1_4_ratio ∧ h_Angle1_3_ratio ∧ ¬ h_Angle_Sum_1_2_3_4_ratio) :=
sorry

end china_junior_high_math_league_1989_problem_l774_774960


namespace factorial_division_l774_774491

noncomputable def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

theorem factorial_division :
  (factorial 10) / ((factorial 7) * (factorial 3)) = 120 := by sorry

end factorial_division_l774_774491


namespace tangent_line_equation_l774_774120

theorem tangent_line_equation :
  let parabola := λ x : ℝ, x^2 + x + 1 in
  let point_on_parabola := (-1 : ℝ, 1 : ℝ) in
  let derivative := λ x : ℝ, 2 * x + 1 in
  point_on_parabola.2 = parabola point_on_parabola.1 →
  derivative (point_on_parabola.1) = -1 →
  ∃ (m b : ℝ), m * (point_on_parabola.1) + b = point_on_parabola.2 ∧ m = -1 ∧ b = 0 :=
begin
  sorry
end

end tangent_line_equation_l774_774120


namespace symmetric_point_on_bc_l774_774728

noncomputable def triangle_symmetry_condition
  (A B C B1 C1 I : Type*)
  [IsTriangle A B C] (angle_A : Angle A = 60)
  (BB1_bisects : AngleBisector BB1 A B)
  (CC1_bisects : AngleBisector CC1 A C)
  (B1C1_line : Line B1 C1)
  (Intersection_I : Intersection BB1 CC1 I) : Prop :=
  PointSymmetric A B1C1_line BC_line

theorem symmetric_point_on_bc
  (A B C B1 C1 I : Type*)
  [IsTriangle A B C] (angle_A : Angle A = 60)
  (BB1_bisects : AngleBisector BB1 A B)
  (CC1_bisects : AngleBisector CC1 A C)
  (B1C1_line : Line B1 C1)
  (Intersection_I : Intersection BB1 CC1 I) :
  triangle_symmetry_condition A B C B1 C1 I angle_A BB1_bisects CC1_bisects B1C1_line Intersection_I :=
sorry

end symmetric_point_on_bc_l774_774728


namespace floor_factorial_fraction_l774_774099

theorem floor_factorial_fraction : 
  (⌊(2009.factorial + 2006.factorial) / (2008.factorial + 2007.factorial)⌋ : ℤ) = 2008 := 
by 
  sorry

end floor_factorial_fraction_l774_774099


namespace sum_of_fractions_l774_774968

theorem sum_of_fractions :
  (1 / 10) + (2 / 10) + (3 / 10) + (4 / 10) + (5 / 10) + (6 / 10) + (7 / 10) + (8 / 10) + (9 / 10) + (10 / 10) + (11 / 10) = 6.6 :=
by {
  have hsum : (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11) = 66 := by sorry,
  have hdiv : 66 / 10 = 6.6 := by sorry,
  exact calc
    (1 / 10) + (2 / 10) + (3 / 10) + (4 / 10) + (5 / 10) + (6 / 10) + (7 / 10) + (8 / 10) + (9 / 10) + (10 / 10) + (11 / 10)
    = (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11) / 10 : by sorry
    ... = 66 / 10 : by rw [hsum]
    ... = 6.6 : by rw [hdiv]
}

end sum_of_fractions_l774_774968


namespace fraction_before_simplification_is_24_56_l774_774451

-- Definitions of conditions
def fraction_before_simplification_simplifies_to_3_7 (a b : ℕ) : Prop :=
  (a ≠ 0 ∧ b ≠ 0) ∧ Int.gcd a b = 1 ∧ (a = 3 * Int.gcd a b ∧ b = 7 * Int.gcd a b)

def sum_of_numerator_and_denominator_is_80 (a b : ℕ) : Prop :=
  a + b = 80

-- Theorem to prove
theorem fraction_before_simplification_is_24_56 (a b : ℕ) :
  fraction_before_simplification_simplifies_to_3_7 a b →
  sum_of_numerator_and_denominator_is_80 a b →
  (a, b) = (24, 56) :=
sorry

end fraction_before_simplification_is_24_56_l774_774451


namespace bill_property_taxes_l774_774963

theorem bill_property_taxes 
  (take_home_salary sales_taxes gross_salary : ℕ)
  (income_tax_rate : ℚ)
  (take_home_salary_eq : take_home_salary = 40000)
  (sales_taxes_eq : sales_taxes = 3000)
  (gross_salary_eq : gross_salary = 50000)
  (income_tax_rate_eq : income_tax_rate = 0.1) :
  let income_taxes := (income_tax_rate * gross_salary) 
  let property_taxes := gross_salary - (income_taxes + sales_taxes + take_home_salary)
  property_taxes = 2000 := by
  sorry

end bill_property_taxes_l774_774963


namespace wheel_of_fortune_prob_l774_774706

-- Define the values on the wheel
inductive WheelValue
| Bankrupt
| Value2000
| Value500
| Value6000
| Value700
| Value300

-- Define a function to calculate the total value of three spins
def spin_value (s1 s2 s3 : WheelValue) : ℕ :=
  match (s1, s2, s3) with
  | (WheelValue.Value2000, WheelValue.Value500, WheelValue.Value200)
  | (WheelValue.Value500, WheelValue.Value2000, WheelValue.Value200)
  | (WheelValue.Value200, WheelValue.Value500, WheelValue.Value2000)
  | (WheelValue.Value2000, WheelValue.Value200, WheelValue.Value500)
  | (WheelValue.Value500, WheelValue.Value500, WheelValue.Value1700)
  | (WheelValue.Value500, WheelValue.Value1700, WheelValue.Value500)
  | (WheelValue.Value1700, WheelValue.Value500, WheelValue.Value500) := 2700
  | _ := 0

-- Define a function to count the number of favorable outcomes
def favorable_outcomes : ℕ :=
  List.foldl (λ acc x => acc + if spin_value x.1 x.2 x.3 = 2700 then 1 else 0) 0 $
  List.product (List.product [WheelValue.Value2000, WheelValue.Value500, WheelValue.Value200] [WheelValue.Value2000, WheelValue.Value500, WheelValue.Value200]) 
               [WheelValue.Value2000, WheelValue.Value500, WheelValue.Value200]

-- Define the total number of possible outcomes
def total_outcomes : ℕ :=
  6 * 6 * 6

-- Define a function to calculate the probability
def probability : ℚ :=
  favorable_outcomes / total_outcomes

theorem wheel_of_fortune_prob : probability = 1 / 36 := by
  sorry -- Proof omitted

end wheel_of_fortune_prob_l774_774706


namespace quadratic_trinomial_has_two_roots_l774_774920

theorem quadratic_trinomial_has_two_roots
  (a b c : ℝ) (h : b^2 - 4 * a * c > 0) :
  4 * (a^2 - a * b + b^2 - 3 * a * c) > 0 :=
by
  sorry

end quadratic_trinomial_has_two_roots_l774_774920


namespace number_of_perfect_square_factors_l774_774660

def is_prime_factorization_valid : Bool :=
  let p1 := 2
  let p1_pow := 1
  let p2 := 3
  let p2_pow := 2
  let p3 := 5
  let p3_pow := 2
  (450 = (p1^p1_pow) * (p2^p2_pow) * (p3^p3_pow))
  
def is_perfect_square (n : Nat) : Bool :=
  match n with
  | 1 => true
  | _ => let second_digit := Math.sqrt n * Math.sqrt n;
         second_digit == n

theorem number_of_perfect_square_factors : (number_of_perfect_square_factors 450 = 4) :=
by sorry

end number_of_perfect_square_factors_l774_774660


namespace number_of_positive_divisors_of_M_l774_774290

-- Definitions based on given conditions
def M : ℕ := 49^6 + 6 * 49^5 + 15 * 49^4 + 20 * 49^3 + 15 * 49^2 + 6 * 49 + 1

-- Proof statement to verify the given number of positive divisors
theorem number_of_positive_divisors_of_M : Nat.factors_count M = 91 :=
by
  sorry

end number_of_positive_divisors_of_M_l774_774290


namespace value_a_2016_l774_774557

-- Definitions of the sequence and its properties
def seq_a : ℕ → ℤ
| 1 := -1
| n := sorry -- Placeholder as the full definition isn't provided

-- Conditions definitions
def condition_1 : seq_a 1 = -1 :=
by sorry

def condition_2 (n : ℕ) (h : n ≥ 2) : |seq_a n - seq_a (n-1)| = 2^(n-1) :=
by sorry

def subseq_decreasing (n : ℕ) : seq_a (2*n-1) > seq_a (2*n+1) :=
by sorry

def subseq_increasing (n : ℕ) : seq_a (2*n) < seq_a (2*n+2) :=
by sorry

-- The theorem to prove the specific value of aₙ at n = 2016
theorem value_a_2016 : seq_a 2016 = (2^2016 - 1) / 3 :=
by 
  -- Use the defined conditions
  sorry

end value_a_2016_l774_774557


namespace perfect_square_divisors_count_450_l774_774647

theorem perfect_square_divisors_count_450 :
  let is_divisor (d n : ℕ) := n % d = 0
  let is_perfect_square (n : ℕ) := ∃ m : ℕ, m * m = n
  let prime_factorization_450 := [(2, 1), (3, 2), (5, 2)]
  ∃ (count : ℕ), count = 4 ∧ count = (Finset.filter 
    (λ d, is_perfect_square d ∧ is_divisor d 450)
    (Finset.powersetPrimeFactors prime_factorization_450)).card := sorry

end perfect_square_divisors_count_450_l774_774647


namespace cos_angle_AOB_is_zero_l774_774346

theorem cos_angle_AOB_is_zero (A B C D O : Type)
  [MetricSpace A]
  [MetricSpace B]
  [MetricSpace C]
  [MetricSpace D]
  [MetricSpace O]
  (h1 : DiagonalsIntersectAt O A B C D)
  (h2 : Length AC = 15)
  (h3 : Length BC = 40)
  : cos (Angle A O B) = 0 := sorry

end cos_angle_AOB_is_zero_l774_774346


namespace sum_of_median_squares_l774_774025

def median_length_squared (a b c : ℝ) : ℝ :=
  (1/4) * (2 * b^2 + 2 * c^2 - a^2)

theorem sum_of_median_squares 
  (AB BC AC : ℝ)
  (h1 : AB = 15)
  (h2 : BC = 13)
  (h3 : AC = 14) :
  median_length_squared AB BC AC 
  + median_length_squared BC AC AB 
  + median_length_squared AC AB BC = 442.5 := 
by
  simp only [median_length_squared, h1, h2, h3]
  sorry

end sum_of_median_squares_l774_774025


namespace area_of_PSQR_is_60_l774_774258

noncomputable def area_of_trapezoid_PSQR (area_PQR : ℕ) (num_small_triangles : ℕ) (area_small_triangle : ℕ) : ℕ :=
let area_PSR := area_PQR / num_small_triangles in
area_PQR - area_PSR

theorem area_of_PSQR_is_60 :
  let area_PQR := 72 in
  let num_small_triangles := 6 in
  let area_small_triangle := 2 in
  area_of_trapezoid_PSQR area_PQR num_small_triangles area_small_triangle = 60 :=
by
  let area_PQR := 72
  let num_small_triangles := 6
  let area_small_triangle := 2
  have h1 : area_PQR / num_small_triangles = 12, from rfl
  have h2 : area_PQR - 12 = 60, from rfl
  exact h2
  sorry

end area_of_PSQR_is_60_l774_774258


namespace number_of_ordered_pairs_l774_774437

theorem number_of_ordered_pairs (S : Finset (Fin 6)) :
  S.card = 6 → 
  let valid_pairs := { (a, b, c, d) ∈ S.product (S.product (S.product S)) | (a * b + c * d) % 7 = 0 } in
  valid_pairs.card = 216 :=
by
  sorry

end number_of_ordered_pairs_l774_774437


namespace guinea_pigs_food_difference_l774_774799

theorem guinea_pigs_food_difference :
  ∀ (first second third total : ℕ),
  first = 2 →
  second = first * 2 →
  total = 13 →
  first + second + third = total →
  third - second = 3 :=
by 
  intros first second third total h1 h2 h3 h4
  sorry

end guinea_pigs_food_difference_l774_774799


namespace oblique_asymptote_l774_774018

noncomputable def f (x : ℝ) : ℝ := (2 * x ^ 3 + 2 * x ^ 2 + 7 * x + 10) / (2 * x + 3)

theorem oblique_asymptote :
  ∀ x, (f(x) - (x^2 - (1 / 2) * x)) → 0 (as x → ∞) :=
begin
  sorry
end

end oblique_asymptote_l774_774018


namespace alice_has_ball_after_three_turns_l774_774473

def alice_keeps_ball (prob_Alice_to_Bob: ℚ) (prob_Bob_to_Alice: ℚ): ℚ := 
  let prob_Alice_keeps := 1 - prob_Alice_to_Bob
  let prob_Bob_keeps := 1 - prob_Bob_to_Alice
  let path1 := prob_Alice_to_Bob * prob_Bob_to_Alice * prob_Alice_keeps
  let path2 := prob_Alice_keeps * prob_Alice_keeps * prob_Alice_keeps
  path1 + path2

theorem alice_has_ball_after_three_turns:
  alice_keeps_ball (1/2) (1/3) = 5/24 := 
by
  sorry

end alice_has_ball_after_three_turns_l774_774473


namespace midpoint_equidistant_l774_774243

open EuclideanGeometry

variables {A B C D K L M N O : Point}
variables (h_quadrilateral : ConvexQuadrilateral A B C D)
variables (h_angle_ABC : ∠ B A C = 90)
variables (h_angle_ADC : ∠ D A C = 90)
variables (h_K_on_AB : OnLine K (Line A B))
variables (h_L_on_BC : OnLine L (Line B C))
variables (h_M_on_CD : OnLine M (Line C D))
variables (h_N_on_DA : OnLine N (Line D A))
variables (h_rectangle : IsRectangle K L M N)
variables (h_mid_AC : O = Midpoint A C)

theorem midpoint_equidistant :
  DistLinePoint (Line K L) O = DistLinePoint (Line M N) O :=
sorry

end midpoint_equidistant_l774_774243


namespace triangle_geometry_theorem_l774_774009

noncomputable def triangle_geometry_problem : Prop :=
  ∃ (A B C D : Type) 
    (angle : A → A → A → ℝ) 
    (degree_measure : ℝ) 
    (isosceles_triangle : A → A → A → Prop) 
    (inside_triangle : A → A → A → A → Prop), 
    isosceles_triangle A B C ∧ isosceles_triangle A D C ∧
    AB = BC ∧ AD = DC ∧
    inside_triangle A B C D ∧
    angle A B C = 60 ∧
    angle A D C = 150 ∧
    degree_measure (angle B A D) = 45

-- We just provided the statement, the proof will be provided as sorry.
theorem triangle_geometry_theorem : triangle_geometry_problem := sorry

end triangle_geometry_theorem_l774_774009


namespace complex_sum_a_b_l774_774542

theorem complex_sum_a_b 
  (a b : ℝ) 
  (h : (a : ℂ) + (b : ℂ) * complex.I = (2 * complex.I) / (1 + complex.I)) : 
  a + b = 2 := 
sorry

end complex_sum_a_b_l774_774542


namespace solve_for_x_l774_774218

theorem solve_for_x (x : ℕ) (h : (1 / 8) * 2 ^ 36 = 8 ^ x) : x = 11 :=
by
sorry

end solve_for_x_l774_774218


namespace max_ab_l774_774885

theorem max_ab (a b : ℝ) (h : 4 * a + b = 1) (ha : a > 0) (hb : b > 0) : ab <= 1 / 16 :=
sorry

end max_ab_l774_774885


namespace radius_of_smaller_semicircle_l774_774738

theorem radius_of_smaller_semicircle :
  ∃ x : ℝ, 0 < x ∧
    let AB := 6 in
    let AC := 12 - x in
    let BC := 6 + x in
    (AB = 6) ∧ 
    (AC = 12 - x) ∧ 
    (BC = 6 + x) ∧
    (AB^2 + AC^2 = BC^2) ∧
    x = 4 := 
by
  use 4
  split
  { exact zero_lt_four }
  split
  { reflexivity }
  split
  { reflexivity }
  split
  { reflexivity }
  { sorry }

end radius_of_smaller_semicircle_l774_774738


namespace daniel_stickers_l774_774090

def stickers_data 
    (total_stickers : Nat)
    (fred_extra : Nat)
    (andrew_kept : Nat) : Prop :=
  total_stickers = 750 ∧ fred_extra = 120 ∧ andrew_kept = 130

theorem daniel_stickers (D : Nat) :
  stickers_data 750 120 130 → D + (D + 120) = 750 - 130 → D = 250 :=
by
  intros h_data h_eq
  sorry

end daniel_stickers_l774_774090


namespace haley_cider_pints_l774_774597

noncomputable def apples_per_farmhand_per_hour := 240
noncomputable def working_hours := 5
noncomputable def total_farmhands := 6

noncomputable def golden_delicious_per_pint := 20
noncomputable def pink_lady_per_pint := 40
noncomputable def golden_delicious_ratio := 1
noncomputable def pink_lady_ratio := 2

noncomputable def total_apples := total_farmhands * apples_per_farmhand_per_hour * working_hours
noncomputable def total_parts := golden_delicious_ratio + pink_lady_ratio

noncomputable def golden_delicious_apples := total_apples / total_parts
noncomputable def pink_lady_apples := golden_delicious_apples * pink_lady_ratio

noncomputable def pints_golden_delicious := golden_delicious_apples / golden_delicious_per_pint
noncomputable def pints_pink_lady := pink_lady_apples / pink_lady_per_pint

theorem haley_cider_pints : 
  total_apples = 7200 → 
  golden_delicious_apples = 2400 → 
  pink_lady_apples = 4800 → 
  pints_golden_delicious = 120 → 
  pints_pink_lady = 120 → 
  pints_golden_delicious = pints_pink_lady →
  pints_golden_delicious = 120 :=
by
  sorry

end haley_cider_pints_l774_774597


namespace red_higher_than_green_l774_774796

noncomputable def red_probability (k : ℕ) : ℝ :=
  if k % 2 = 0 then 3^(-k : ℤ) else 2^(-k : ℤ)

noncomputable def green_probability (k : ℕ) : ℝ :=
  if k % 2 = 0 then 2^(-k : ℤ) else 3^(-k : ℤ)

noncomputable def higher_number_probability : ℝ :=
  ∑ k in (finset.range ∞), ∑ j in (finset.Icc (k + 1) ∞), red_probability k * green_probability j

theorem red_higher_than_green (h : higher_number_probability = 0.2) : higher_number_probability = 0.2 :=
  sorry

end red_higher_than_green_l774_774796


namespace complex_expression_l774_774488

theorem complex_expression (i : ℂ) (h₁ : i^2 = -1) (h₂ : i^4 = 1) :
  (i + i^3)^100 + (i + i^2 + i^3 + i^4 + i^5)^120 = 1 := by
  sorry

end complex_expression_l774_774488


namespace machine_a_production_rate_l774_774878

def machine_q_time_to_produce_220_sprockets (h : ℝ) : Prop :=
  (220 : ℝ) / (h + 10) = 198 / h

def sprockets_per_hour_machine_q (h : ℝ) : ℝ :=
  220 / h

def sprockets_per_hour_machine_a (h : ℝ) : ℝ :=
  sprockets_per_hour_machine_q h * (100 / 110)

theorem machine_a_production_rate : ∀ (h : ℝ), 
  machine_q_time_to_produce_220_sprockets h → 
  sprockets_per_hour_machine_a h = 20 / 9 :=
by
  sorry

end machine_a_production_rate_l774_774878


namespace smallest_angle_convex_22gon_arithmetic_sequence_l774_774345

theorem smallest_angle_convex_22gon_arithmetic_sequence :
  ∃ (a : ℝ), a = 164 ∧ 
  (∀ (n : ℕ), n < 22 → 
    let angle := (163.636) + (2 * n) * d in
    angle < 180 ∧
    (∀ j < 21, angle j.succ > angle j)) :=
begin
  sorry
end

end smallest_angle_convex_22gon_arithmetic_sequence_l774_774345


namespace eval_sqrt4_8_pow12_l774_774131

-- Define the fourth root of 8
def fourthRootOfEight : ℝ := 8 ^ (1 / 4)

-- Define the original expression
def expr := (fourthRootOfEight) ^ 12

-- The theorem to prove
theorem eval_sqrt4_8_pow12: expr = 512 := by
  sorry

end eval_sqrt4_8_pow12_l774_774131


namespace percentage_of_seventh_graders_l774_774003

-- Define constants for the problem
def num_seventh_graders := 64
def percentage_sixth_graders := 0.38
def num_sixth_graders := 76

-- Define the condition that relates percentage and number of sixth graders to total students
def total_students := num_sixth_graders / percentage_sixth_graders

-- The final statement to prove
theorem percentage_of_seventh_graders : 
    (num_seventh_graders / total_students) * 100 = 32 := by
  sorry

end percentage_of_seventh_graders_l774_774003


namespace max_value_expression_l774_774766

noncomputable def max_expression (y : ℝ) : ℝ := (y^2 + 3 - real.sqrt (y^4 + 9)) / y

theorem max_value_expression :
  ∀ y : ℝ, y > 0 → max_expression y ≤ 6 / (2 * real.sqrt 3 + real.sqrt 6) :=
begin
  sorry
end

end max_value_expression_l774_774766


namespace find_ellipse_equation_find_line_slope_range_l774_774566

noncomputable section

open Real

def equation_of_ellipse (a b : ℝ) (p : ℝ × ℝ) (eccentricity : ℝ) := 
  (a > b ∧ b > 0) ∧ 
  p  = (sqrt 3, 1 / 2) ∧ 
  eccentricity = (sqrt 3 / 2) ∧
  3 / a^2 + 1 / (4 * b^2) = 1 ∧ 
  a^2 = b^2 + (a * eccentricity)^2

theorem find_ellipse_equation :
  ∃ (a b : ℝ), equation_of_ellipse a b (sqrt 3, 1 / 2) (sqrt 3 / 2) ∧
  (a = 2 ∧ b = 1) ∧ 
  (∀ x y, (x ^ 2) / 4 + (y ^ 2) = 1) :=
sorry

def line_slope_condition (O : ℝ × ℝ) (N : ℝ × ℝ) (k : ℝ) :=
  N = (0, sqrt 2) ∧
  ((k < - sqrt 6 / 2) ∨ (k > sqrt 6 / 2)) ∧ 
  (
    let x1 := (- 8 * sqrt 2 * k) / (1 + 4 * k ^ 2) in
    let x2 := _ in
    (1 + k^2) * 4 / (1 + 4 * k^2) + sqrt 2 * k * (- 8 * sqrt 2 * k / (1 + 4 * k^2)) + 2 < 0
  ) 

theorem find_line_slope_range :
  ∃ (k : ℝ), line_slope_condition (0, 0) (0, sqrt 2) k :=
sorry

end find_ellipse_equation_find_line_slope_range_l774_774566


namespace numPerfectSquareFactorsOf450_l774_774677

def isPerfectSquare (n : Nat) : Prop :=
  ∃ k : Nat, n = k * k

theorem numPerfectSquareFactorsOf450 : 
  ∃! n : Nat, 
    (∀ d : Nat, d ∣ 450 → isPerfectSquare d) → n = 4 := 
by
  sorry

end numPerfectSquareFactorsOf450_l774_774677


namespace total_bricks_fill_box_l774_774907

-- Define brick and box volumes based on conditions
def volume_brick1 := 2 * 5 * 8
def volume_brick2 := 2 * 3 * 7
def volume_box := 10 * 11 * 14

-- Define the main proof problem
theorem total_bricks_fill_box (x y : ℕ) (h1 : volume_brick1 * x + volume_brick2 * y = volume_box) :
  x + y = 24 :=
by
  -- Left as an exercise (proof steps are not included per instructions)
  sorry

end total_bricks_fill_box_l774_774907


namespace A_div_B_eq_26_l774_774501

noncomputable def A' : ℝ := ∑' (n : ℕ) in { n | n % 2 = 1 ∧ n % 5 ≠ 0 }, (if (n + 1) / 2 % 2 = 1 then (-1 : ℝ) else 1) * (1 / (n : ℝ)^2)
noncomputable def B' : ℝ := ∑' (k : ℕ) (hk : k > 0), (-1)^(k + 1) * (1 / ((5 * k : ℝ)^2))

theorem A_div_B_eq_26 : A' / B' = 26 := sorry

end A_div_B_eq_26_l774_774501


namespace second_discount_percentage_l774_774367

variable (initial_price first_discount final_price : ℝ) (D : ℝ)

-- Define the initial price, first discount rate, and final price after successive discounts
def initial_conditions :=
  initial_price = 400 ∧ first_discount = 0.1 ∧ final_price = 331.2

-- Define the price after first discount
def price_after_first_discount (initial_price first_discount : ℝ) : ℝ :=
  initial_price * (1 - first_discount)

-- Define the price after second discount
def price_after_second_discount (price_after_first_discount D : ℝ) : ℝ :=
  price_after_first_discount * (1 - D / 100)

-- Statement to prove
theorem second_discount_percentage
  (h : initial_conditions initial_price first_discount final_price) :
  price_after_second_discount (price_after_first_discount initial_price first_discount) D = final_price → D = 8 :=
by sorry

end second_discount_percentage_l774_774367


namespace person_speed_l774_774420

noncomputable def distance_meters : ℝ := 1080
noncomputable def time_minutes : ℝ := 14
noncomputable def distance_kilometers : ℝ := distance_meters / 1000
noncomputable def time_hours : ℝ := time_minutes / 60
noncomputable def speed_km_per_hour : ℝ := distance_kilometers / time_hours

theorem person_speed :
  abs (speed_km_per_hour - 4.63) < 0.01 :=
by
  -- conditions extracted
  let distance_in_km := distance_meters / 1000
  let time_in_hours := time_minutes / 60
  let speed := distance_in_km / time_in_hours
  -- We expect speed to be approximately 4.63
  sorry 

end person_speed_l774_774420


namespace relatively_prime_dates_in_july_l774_774936

-- Define the month of July with 31 days.
def july_days := {d : ℕ | 1 ≤ d ∧ d ≤ 31}

-- Define relatively prime dates in July.
def is_relatively_prime (d : ℕ) : Prop := Nat.gcd 7 d = 1

-- Count how many dates in July are relatively prime with the 7th month.
def count_relatively_prime_dates_in_july : ℕ :=
  Finset.card (Finset.filter is_relatively_prime (Finset.filter (λ d, d ∈ july_days) (Finset.range 32)))

-- Statement to prove
theorem relatively_prime_dates_in_july : count_relatively_prime_dates_in_july = 27 :=
  sorry

end relatively_prime_dates_in_july_l774_774936


namespace false_complementary_not_equal_l774_774033

theorem false_complementary_not_equal :
  (∀ (α β : ℝ), α + β = 90 → α ≠ β) → False :=
by 
  assume (h : ∀ (α β : ℝ), α + β = 90 → α ≠ β),
  have counter_example : 45 + 45 = 90 := by norm_num,
  have equal_angles : 45 = 45 := by norm_num,
  have contradiction := h 45 45 counter_example,
  contradiction sorry

end false_complementary_not_equal_l774_774033


namespace log_order_l774_774984

theorem log_order : log 0.76 < 0.76 ∧ 0.76 < 60.7 :=
by
  sorry

end log_order_l774_774984


namespace islanders_liars_l774_774091

inductive Person
| A
| B

open Person

def is_liar (p : Person) : Prop :=
  sorry -- placeholder for the actual definition

def makes_statement (p : Person) (statement : Prop) : Prop :=
  sorry -- placeholder for the actual definition

theorem islanders_liars :
  makes_statement A (is_liar A ∧ ¬ is_liar B) →
  is_liar A ∧ is_liar B :=
by
  sorry

end islanders_liars_l774_774091


namespace speed_of_second_cyclist_l774_774391

theorem speed_of_second_cyclist (v : ℝ) 
  (circumference : ℝ) 
  (time : ℝ) 
  (speed_first_cyclist : ℝ)
  (meet_time : ℝ)
  (circ_full: circumference = 300) 
  (time_full: time = 20)
  (speed_first: speed_first_cyclist = 7)
  (meet_full: meet_time = time):

  v = 8 := 
by
  sorry

end speed_of_second_cyclist_l774_774391


namespace train_length_and_speed_l774_774742

theorem train_length_and_speed (L_bridge : ℕ) (t_cross : ℕ) (t_on_bridge : ℕ) (L_train : ℕ) (v_train : ℕ)
  (h_bridge : L_bridge = 1000)
  (h_t_cross : t_cross = 60)
  (h_t_on_bridge : t_on_bridge = 40)
  (h_crossing_eq : (L_bridge + L_train) / t_cross = v_train)
  (h_on_bridge_eq : L_bridge / t_on_bridge = v_train) : 
  L_train = 200 ∧ v_train = 20 := 
  by
  sorry

end train_length_and_speed_l774_774742


namespace length_of_each_brick_l774_774056

theorem length_of_each_brick 
  (courtyard_length : ℝ) (courtyard_width : ℝ)
  (brick_width : ℝ) (total_bricks : ℕ)
  (h1 : courtyard_length = 25) (h2 : courtyard_width = 15)
  (h3 : brick_width = 10) (h4 : total_bricks = 18750) :
  (3750 * 100) / (total_bricks * brick_width) = 20 :=
begin
  sorry
end

end length_of_each_brick_l774_774056


namespace orthocenter_relation_l774_774121

variables {A B C A1 B1 C1 M : Type} 
variables [acute_angled_triangle : is_acute_angled_triangle A B C]
variables [altitudes_feet : feet_of_altitudes A1 B1 C1 A B C]
variables [orthocenter : is_orthocenter M A B C]
variables [circumradius : radius R B]
variables [inradius : radius r A1 B1 C1]

theorem orthocenter_relation
  (h1 : acute_angled_triangle A B C)
  (h2 : altitudes_feet A1 B1 C1 A B C)
  (h3 : orthocenter M A B C)
  (h4 : circumradius R B)
  (h5 : inradius r A1 B1 C1) :
  MA * MA₁ = 2 * R * r :=
begin
  sorry -- proof to be provided
end

end orthocenter_relation_l774_774121


namespace miles_driven_on_tuesday_l774_774084

-- Define the conditions given in the problem
theorem miles_driven_on_tuesday (T : ℕ) (h_avg : (12 + T + 21) / 3 = 17) :
  T = 18 :=
by
  -- We state what we want to prove, but we leave the proof with sorry
  sorry

end miles_driven_on_tuesday_l774_774084


namespace total_salaries_eq_3000_l774_774832

noncomputable def A_salary : ℝ := 2250
noncomputable def A_savings : ℝ := 0.05 * A_salary

variables (B_salary A_savings B_savings : ℝ)
axiom B_savings_eq_A_savings : A_savings = B_savings
axiom B_savings_def : B_savings = 0.15 * B_salary

theorem total_salaries_eq_3000 (h : A_savings = B_savings_def) : 
  A_salary + B_salary = 3000 :=
sorry

end total_salaries_eq_3000_l774_774832


namespace find_C_and_D_l774_774538

theorem find_C_and_D (C D : ℚ) (h1 : 5 * C + 3 * D - 4 = 47) (h2 : C = D + 2) : 
  C = 57 / 8 ∧ D = 41 / 8 :=
by 
  sorry

end find_C_and_D_l774_774538


namespace g_values_multiplication_l774_774292

noncomputable def g : ℝ → ℝ := sorry

axiom g_property : ∀ x y, g (g x - y) = g x + g (g y - g (-x)) + 2 * x

theorem g_values_multiplication :
  let m := (set_of (λ y, g 4 = y)).to_finset.card in
  let t := (set_of (λ y, g 4 = y)).to_finset.sum id in
  m * t = -8 :=
by
  sorry

end g_values_multiplication_l774_774292


namespace rate_of_interest_is_six_l774_774941

-- Define the given conditions
def principal : ℝ := 12500
def final_amount : ℝ := 15500
def time : ℕ := 4

-- Define the simple interest formula component
def simple_interest (P R T : ℝ) : ℝ := (P * R * T) / 100

-- Define the interest earned
def interest_earned : ℝ := final_amount - principal

-- The target theorem to prove the rate of interest is 6%
theorem rate_of_interest_is_six :
  ∃ (R : ℝ), simple_interest principal R time = interest_earned ∧ R = 6 :=
by
  use 6
  -- Here you would normally provide the proof, but it is skipped as per instructions
  sorry

end rate_of_interest_is_six_l774_774941


namespace leo_weight_l774_774289

variables (L K E : ℝ)

theorem leo_weight :
  L + K + E = 210 ∧
  L + 10 = 1.5 * K ∧
  L + 10 = 0.75 * E → 
  L = 63.33 :=
begin
  sorry
end

end leo_weight_l774_774289


namespace incorrect_value_at_x5_l774_774388

theorem incorrect_value_at_x5 
  (f : ℕ → ℕ) 
  (provided_values : List ℕ) 
  (h_f : ∀ x, f x = 2 * x ^ 2 + 3 * x + 5)
  (h_provided_values : provided_values = [10, 18, 29, 44, 63, 84, 111, 140]) : 
  ¬ (f 5 = provided_values.get! 4) := 
by
  sorry

end incorrect_value_at_x5_l774_774388


namespace symmetric_about_x_axis_l774_774714

noncomputable def P (a b : ℝ) : Prop := P = (a, 1)
noncomputable def Q (a b : ℝ) : Prop := Q = (2, b)

theorem symmetric_about_x_axis (a b : ℝ) (h1 : a = 2) (h2 : 1 = -b) : a + b = 1 :=
by {
  sorry
}

end symmetric_about_x_axis_l774_774714


namespace perfect_square_factors_450_l774_774624

theorem perfect_square_factors_450 : 
  ∃ n, n = 4 ∧ (∀ k | 450, k ∣ 450 → is_square k) → (∃ m1 m2 m3 : ℕ, 450 = 2^m1 * 3^m2 * 5^m3 ) :=
by sorry

end perfect_square_factors_450_l774_774624


namespace find_salary_l774_774875

theorem find_salary (S : ℤ) (food house_rent clothes left : ℤ) 
  (h_food : food = S / 5) 
  (h_house_rent : house_rent = S / 10) 
  (h_clothes : clothes = 3 * S / 5) 
  (h_left : left = 18000) 
  (h_spent : food + house_rent + clothes + left = S) : 
  S = 180000 :=
by {
  sorry
}

end find_salary_l774_774875


namespace min_people_required_l774_774107

-- Define the given conditions as Lean definitions and assumptions
variables (initial_days : ℕ) (days_passed : ℕ) (total_people : ℕ) (work_done : ℚ)
variables (remaining_days : ℕ) (remaining_work : ℚ)

-- Initialize the given conditions
def task_conditions : Prop :=
  initial_days = 40 ∧
  days_passed = 10 ∧
  total_people = 12 ∧
  work_done = 2 / 5 ∧
  remaining_days = initial_days - days_passed ∧
  remaining_work = 3 / 5

-- The theorem that needs to be proved.
theorem min_people_required (task_conditions) : ∃ n : ℕ, (n / 10 : ℚ) = 3 / 5 ∧ n = 6 := sorry

end min_people_required_l774_774107


namespace least_sum_of_exponents_l774_774221

theorem least_sum_of_exponents (n : ℕ) (h₁ : n = 800) (h₂ : ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ n = 2^a + 2^b + 2^c ∧ a + b + c = 22) :
  ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ n = 2^a + 2^b + 2^c ∧ a + b + c = 22 :=
begin
  sorry
end

end least_sum_of_exponents_l774_774221


namespace log_base_32_2_eq_one_fifth_l774_774988

theorem log_base_32_2_eq_one_fifth : logb 32 2 = 1 / 5 := 
by sorry

end log_base_32_2_eq_one_fifth_l774_774988


namespace simplify_expr_1_l774_774886

theorem simplify_expr_1 (a : ℝ) : (2 * a - 3) ^ 2 + (2 * a + 3) * (2 * a - 3) = 8 * a ^ 2 - 12 * a :=
by
  sorry

end simplify_expr_1_l774_774886


namespace correct_statement_about_surveys_l774_774413

-- Definitions for each statement
def statement_A : Prop :=
  ∀ survey : Type, (comprehensive survey → suitable survey)

def statement_B : Prop :=
  (∀ student : Type, (vision_test student → comprehensive survey))

def statement_C : Prop :=
  ∀ community household : Type, (sample household from community) → sample_size (sample household from community) = 1500

def statement_D : Prop :=
  ∀ school student basketball_team : Type, (height_sample basketball_team) → objective_estimate school student (height_sample basketball_team)

-- The theorem to prove
theorem correct_statement_about_surveys (A B C D : Prop) (hA : statement_A) (hB : statement_B) (hC : statement_C) (hD : statement_D) : B :=
sorry

end correct_statement_about_surveys_l774_774413


namespace initial_decaf_percentage_l774_774059

theorem initial_decaf_percentage 
  (initial_stock : ℝ) (additional_stock : ℝ) 
  (additional_stock_decaf_percent : ℝ) 
  (total_stock_decaf_percent : ℝ) : 
  initial_stock = 400 → 
  additional_stock = 100 →
  additional_stock_decaf_percent = 70 →
  total_stock_decaf_percent = 30 → 
  ∃ x : ℝ, x = 20 := 
by
  intros h_initial_stock h_additional_stock h_additional_stock_decaf_percent h_total_stock_decaf_percent
  use 20
  sorry

end initial_decaf_percentage_l774_774059


namespace johns_donation_is_correct_l774_774847

/-
Conditions:
1. Alice, Bob, and Carol donated different amounts.
2. The ratio of Alice's, Bob's, and Carol's donations is 3:2:5.
3. The sum of Alice's and Bob's donations is $120.
4. The average contribution increases by 50% and reaches $75 per person after John donates.

The statement to prove:
John's donation is $240.
-/

def donations_ratio : ℕ × ℕ × ℕ := (3, 2, 5)
def sum_Alice_Bob : ℕ := 120
def new_avg_after_john : ℕ := 75
def num_people_before_john : ℕ := 3
def avg_increase_factor : ℚ := 1.5

theorem johns_donation_is_correct (A B C J : ℕ) 
  (h1 : A * 2 = B * 3) 
  (h2 : B * 5 = C * 2) 
  (h3 : A + B = sum_Alice_Bob) 
  (h4 : (A + B + C) / num_people_before_john = 80) 
  (h5 : ((A + B + C + J) / (num_people_before_john + 1)) = new_avg_after_john) :
  J = 240 := 
sorry

end johns_donation_is_correct_l774_774847


namespace carmela_gives_each_l774_774096

noncomputable def money_needed_to_give_each (carmela : ℕ) (cousins : ℕ) (cousins_count : ℕ) : ℕ :=
  let total_cousins_money := cousins * cousins_count
  let total_money := carmela + total_cousins_money
  let people_count := 1 + cousins_count
  let equal_share := total_money / people_count
  let total_giveaway := carmela - equal_share
  total_giveaway / cousins_count

theorem carmela_gives_each (carmela : ℕ) (cousins : ℕ) (cousins_count : ℕ) (h_carmela : carmela = 7) (h_cousins : cousins = 2) (h_cousins_count : cousins_count = 4) :
  money_needed_to_give_each carmela cousins cousins_count = 1 :=
by
  rw [h_carmela, h_cousins, h_cousins_count]
  sorry

end carmela_gives_each_l774_774096


namespace has_two_roots_l774_774915

-- Define the discriminant of the quadratic trinomial
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Main Lean statement
theorem has_two_roots
  (a b c : ℝ)
  (h : discriminant a b c > 0) :
  discriminant (3 * a) (2 * (a + b)) (b + c) > 0 := by
  sorry

end has_two_roots_l774_774915


namespace prism_surface_area_l774_774828

theorem prism_surface_area (a b h : ℝ) (V : ℝ) 
  (ratio_areas : a^2 / b^2 = 1 / 16)
  (height_condition : h = 3 * a)
  (volume_given : V = 567) :
  let S := a^2 + b^2 + 4 * (a + b) * (h/2) in
  S ≈ (450 : ℝ) :=
by
  sorry

end prism_surface_area_l774_774828


namespace find_a5_l774_774719

variable {a_n : ℕ → ℝ}

def is_geometric_sequence (a_n : ℕ → ℝ) : Prop :=
∀ m n, ∃ r, r ≠ 0 ∧ a_n (m + n) = a_n m * r ^ n

noncomputable def a_3 := some (classical.some_spec (classical.some (exists_pair_of_roots_eq (by norm_num : polynomial ℝ 3 0 (- 11) 0 9 = 0) a_3 a_7))).fst
noncomputable def a_7 := some (classical.some_spec (classical.some (exists_pair_of_roots_eq (by norm_num : polynomial ℝ 3 0 (- 11) 0 9 = 0) a_3 a_7))).snd

theorem find_a5 (h_geo : is_geometric_sequence a_n)
  (h_roots : polynomial_roots (by norm_num : polynomial ℝ 3 0 (- 11) 0 9) [a_3, a_7]) :
  a_n 5 = real.sqrt 3 :=
sorry

end find_a5_l774_774719


namespace area_of_triangle_MOI_is_11_over_4_l774_774702

-- Define the given points
def P : ℝ × ℝ := (0, 0)
def Q : ℝ × ℝ := (8, 0)
def R : ℝ × ℝ := (0, 6)

-- Define the circumcenter O of triangle PQR
def O : ℝ × ℝ := (4, 3)

-- Define the incenter I of triangle PQR
def I : ℝ × ℝ := (24 / 11, 18 / 11)

-- Define the point M such that a circle centered at M is tangent to PR, QR, and the circumcircle of triangle PQR
def M : ℝ × ℝ := (5 / 2, 5 / 2)

-- Define the area of triangle MOI using the Shoelace Theorem
noncomputable def area_MOI : ℝ :=
  let (x1, y1) := M
  let (x2, y2) := O
  let (x3, y3) := I
  (1 / 2) * abs (x1 * y2 + x2 * y3 + x3 * y1 - y1 * x2 - y2 * x3 - y3 * x1)

-- The theorem we want to prove: area of triangle MOI equals 11 / 4
theorem area_of_triangle_MOI_is_11_over_4 : area_MOI = 11 / 4 := sorry

end area_of_triangle_MOI_is_11_over_4_l774_774702


namespace count_numbers_with_three_transitions_l774_774967

-- Define the function to count transitions in the binary representation of a number
def transition_count (n : ℕ) : ℕ := 
  let bits := Nat.digits 2 n
  ((bits.zip (bits.tail)).filter (λ pair, pair.1 ≠ pair.2)).length

-- Define the main property we're interested in
def has_three_transitions (n : ℕ) : Prop := transition_count n = 3

-- The main theorem statement
theorem count_numbers_with_three_transitions : 
  (Finset.filter has_three_transitions (Finset.range 51)).card = 9 :=
by
  sorry

end count_numbers_with_three_transitions_l774_774967


namespace perfect_square_factors_count_450_l774_774671

theorem perfect_square_factors_count_450 : 
  (∃ n : ℕ, n = 4 ∧ (∀ d : ℕ, d ∣ 450 → ∃ k : ℕ, d = k * k)) := sorry

end perfect_square_factors_count_450_l774_774671


namespace number_of_valid_menus_l774_774055

-- Define the conditions as functions or predicates:
def is_valid_menu (menu : List String) : Prop :=
  menu.length = 7 ∧
  ∀ i, i < 6 → menu.nth i ≠ menu.nth (i+1) ∧
  menu.nth 2 = some "pie" ∧
  menu.nth 5 = some "pudding"

-- Define the desserts options:
def desserts : List String := ["cake", "pie", "ice cream", "pudding"]

-- Define the problem statement:
theorem number_of_valid_menus : 
  {menu : List String // is_valid_menu menu}.card = 972 :=
by {
  sorry
}

end number_of_valid_menus_l774_774055


namespace rohan_house_rent_percentage_l774_774798

noncomputable def house_rent_percentage (food_percentage entertainment_percentage conveyance_percentage salary savings: ℝ) : ℝ :=
  100 - (food_percentage + entertainment_percentage + conveyance_percentage + (savings / salary * 100))

-- Conditions
def food_percentage : ℝ := 40
def entertainment_percentage : ℝ := 10
def conveyance_percentage : ℝ := 10
def salary : ℝ := 10000
def savings : ℝ := 2000

-- Theorem
theorem rohan_house_rent_percentage :
  house_rent_percentage food_percentage entertainment_percentage conveyance_percentage salary savings = 20 := 
sorry

end rohan_house_rent_percentage_l774_774798


namespace find_Q1_plus_Qm1_l774_774755

noncomputable def Q (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + l

variables (l a b c : ℝ)
variables (Q_is_cubic : ∀ x, Q x = a * x^3 + b * x^2 + c * x + l)
variables (Q0 : Q 0 = l) (Q2 : Q 2 = 3 * l) (Qm2 : Q (-2) = 5 * l)

theorem find_Q1_plus_Qm1 : Q 1 + Q (-1) = (7 / 2) * l :=
by
  sorry

end find_Q1_plus_Qm1_l774_774755


namespace find_x_eq_37_l774_774150

theorem find_x_eq_37 (x : ℝ) (h : sqrt (5 * x + 11) = 14) : x = 37 :=
sorry

end find_x_eq_37_l774_774150


namespace number_of_perfect_square_factors_l774_774661

def is_prime_factorization_valid : Bool :=
  let p1 := 2
  let p1_pow := 1
  let p2 := 3
  let p2_pow := 2
  let p3 := 5
  let p3_pow := 2
  (450 = (p1^p1_pow) * (p2^p2_pow) * (p3^p3_pow))
  
def is_perfect_square (n : Nat) : Bool :=
  match n with
  | 1 => true
  | _ => let second_digit := Math.sqrt n * Math.sqrt n;
         second_digit == n

theorem number_of_perfect_square_factors : (number_of_perfect_square_factors 450 = 4) :=
by sorry

end number_of_perfect_square_factors_l774_774661


namespace num_friends_l774_774310

-- Define the friends
def Mary : Prop := ∃ n : ℕ, n = 6
def Sam : Prop := ∃ n : ℕ, n = 6
def Keith : Prop := ∃ n : ℕ, n = 6
def Alyssa : Prop := ∃ n : ℕ, n = 6

-- Define the set of friends
def friends : set Prop := {Mary, Sam, Keith, Alyssa}

-- Statement to prove
theorem num_friends (h1 : Mary) (h2 : Sam) (h3 : Keith) (h4 : Alyssa) : 
  set.card friends = 4 :=
by sorry

end num_friends_l774_774310


namespace max_parts_after_one_more_line_l774_774069

theorem max_parts_after_one_more_line (M N : ℕ) :
  let initial_parts := M * N
  let max_parts := initial_parts + M + N - 1
  (parallelogram_parts_with_extra_line M N) = max_parts :=
by
  -- Define the concept of parts in a parallelogram with M and N divisions
  sorry

end max_parts_after_one_more_line_l774_774069


namespace minimize_sqrt_difference_l774_774294

noncomputable def x (p : ℕ) (hp : p.prime ∧ p % 2 = 1) : ℕ := (p - 1) / 2
noncomputable def y (p : ℕ) (hp : p.prime ∧ p % 2 = 1) : ℕ := (p + 1) / 2

theorem minimize_sqrt_difference (p : ℕ) (hp : p.prime ∧ p % 2 = 1) : 
  ∃ x y : ℕ, x = (p - 1) / 2 ∧ y = (p + 1) / 2 ∧ x ≤ y ∧ 
  (real.sqrt (2 * p) - real.sqrt x - real.sqrt y) ≥ 0 :=
by
  use x p hp, y p hp
  split 
  sorry
  sorry
  sorry

end minimize_sqrt_difference_l774_774294


namespace max_value_n_l774_774786

noncomputable def max_points_on_ellipse (a c d : ℝ) (P : ℕ → ℝ × ℝ) (F : ℝ × ℝ) : ℕ :=
  let min_distance := a - c
  let max_distance := a + c
  let common_difference := (max_distance - min_distance) / (199)
  if common_difference >= d then 201 else 0

theorem max_value_n : 
  ∀ (P : ℕ → (ℝ × ℝ)) (F : ℝ × ℝ),
  (∀ n, 0 < n → (P n).1 ^ 2 / 4 + (P n).2 ^ 2 / 3 = 1) →  -- Points lie on the ellipse
  (∃ a b : ℝ, 
    let c := sqrt (a^2 - b^2) in
    (∀ n, |dist (P n) F| forms an arithmetic sequence with a common difference not less than 1 / 100) →
    max_points_on_ellipse 2 1 (1 / 100) P F = 201) := 
sorry

end max_value_n_l774_774786


namespace neither_odd_nor_even_and_increasing_l774_774229

def power_function (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x, f(x) = x^α

open Real

theorem neither_odd_nor_even_and_increasing (f : ℝ → ℝ) (h : power_function f) 
  (h5 : f 5 = sqrt 5) : 
  (¬ (∀ x, f(-x) = f x) ∧ ¬ (∀ x, f(-x) = -f x)) ∧ (∀ x y, 0 < x ∧ x < y → f x < f y) :=
by
  sorry

end neither_odd_nor_even_and_increasing_l774_774229


namespace geometric_sequence_first_term_l774_774153

theorem geometric_sequence_first_term (a b c : ℕ) (r : ℕ) (h1 : r = 2) (h2 : b = a * r)
  (h3 : c = b * r) (h4 : 32 = c * r) (h5 : 64 = 32 * r) :
  a = 4 :=
by sorry

end geometric_sequence_first_term_l774_774153


namespace volume_of_wedge_is_correct_l774_774898

-- Defining the radius and height of the cylinder (cheese)
def radius : ℝ := 4
def height : ℝ := 10
def angle : ℝ := 60 / 360

-- Volume of the entire cylinder
def cylinder_volume : ℝ := Real.pi * radius^2 * height

-- Volume of the wedge (60° section)
def wedge_volume : ℝ := angle * cylinder_volume

-- Desired volume in cubic centimeters (approx)
def desired_volume : ℝ := 83.74

-- Statement of the problem
theorem volume_of_wedge_is_correct :
  (wedge_volume ≈ desired_volume) :=
by
  sorry

end volume_of_wedge_is_correct_l774_774898


namespace smallest_angle_is_180_over_7_deg_l774_774102

noncomputable def smallest_positive_angle (x : ℝ) : Prop :=
  tan (6 * x) = (sin x - cos x) / (sin x + cos x)

theorem smallest_angle_is_180_over_7_deg :
  ∃ x : ℝ, x > 0 ∧ smallest_positive_angle x ∧ x = 180 / 7 :=
by
  sorry

end smallest_angle_is_180_over_7_deg_l774_774102


namespace quadrilateral_with_irrational_triangle_area_l774_774118

variable {A B C D O : Type}
variable [Triangle A B O] [Triangle B C O] [Triangle C D O] [Triangle D A O]
variable (ABCD : Quadrilateral ℝ) (f : ∀ O ∈ interior ABCD, RationalArea A B O → RationalArea B C O → RationalArea C D O → RationalArea D A O → False)

theorem quadrilateral_with_irrational_triangle_area : 
  ∃ (ABCD : Quadrilateral ℝ), area ABCD = 1 ∧ 
  ∀ (O ∈ interior ABCD), (area (A B O) ∉ ℚ) ∨ (area (B C O) ∉ ℚ) ∨ (area (C D O) ∉ ℚ) ∨ (area (D A O) ∉ ℚ) :=
begin
  sorry
end

end quadrilateral_with_irrational_triangle_area_l774_774118


namespace red_blue_segment_intersections_l774_774784

theorem red_blue_segment_intersections 
  (n : ℕ) 
  (points : Finset (Point circle)) 
  (colored_points : ∀ p : Point circle, p ∈ points → (p.color = Color.Red ∨ p.color = Color.Blue)) 
  (pairs_red : Finset (Point circle × Point circle)) 
  (pairs_blue : Finset (Point circle × Point circle))
  (no_three_intersect : ∀ s1 s2 s3 : Segment, 
    s1.pair ∈ pairs_red ∪ pairs_blue → 
    s2.pair ∈ pairs_red ∪ pairs_blue → 
    s3.pair ∈ pairs_red ∪ pairs_blue → 
    ¬segments_intersect_at_single_point s1 s2 s3) :
  ∃ k ≥ n, occurs_intersection k :=
sorry


end red_blue_segment_intersections_l774_774784


namespace arithmetic_mean_solve_x_l774_774966

theorem arithmetic_mean_solve_x (x : ℚ) :
  (x + 10 + 20 + 3 * x + 15 + 3 * x + 6) / 5 = 30 → x = 99 / 7 :=
by 
sorry

end arithmetic_mean_solve_x_l774_774966


namespace ball_distribution_l774_774002

theorem ball_distribution :
  let white_combinations : ℕ := Nat.choose 5 2,
      red_combinations : ℕ := Nat.choose 6 2,
      yellow_combinations : ℕ := Nat.choose 7 2
  in white_combinations * red_combinations * yellow_combinations = 3150 := by
  sorry

end ball_distribution_l774_774002


namespace monotonically_decreasing_exists_odd_function_l774_774199

-- Problem 1: Monotonicity of the function
theorem monotonically_decreasing (m : ℝ) :
  ∀ x1 x2 : ℝ, x1 > x2 → ∀ x : ℝ, (f : ℝ → ℝ) :=
  let f := λ x, 2 / (3 ^ x + 1) + m
  ∀ x1 x2 : ℝ, x1 > x2 → f x1 < f x2 :=
sorry

-- Problem 2: Existence of m for odd function
theorem exists_odd_function :
  ∃ m : ℝ, ∀ x : ℝ, (f : ℝ → ℝ) :=
  let f := λ x, 2 / (3 ^ x + 1) + m
  ∃ m : ℝ, ∀ x : ℝ, f (-x) = -f x :=
  exists.intro (-1) sorry

end monotonically_decreasing_exists_odd_function_l774_774199


namespace tree_age_difference_l774_774481

theorem tree_age_difference
  (groups_rings : ℕ)
  (rings_per_group : ℕ)
  (first_tree_groups : ℕ)
  (second_tree_groups : ℕ)
  (rings_per_year : ℕ)
  (h_rg : rings_per_group = 6)
  (h_ftg : first_tree_groups = 70)
  (h_stg : second_tree_groups = 40)
  (h_rpy : rings_per_year = 1) :
  ((first_tree_groups * rings_per_group) - (second_tree_groups * rings_per_group)) = 180 := 
by
  sorry

end tree_age_difference_l774_774481


namespace find_a_and_root_l774_774194

def equation_has_double_root (a x : ℝ) : Prop :=
  a * x^2 + 4 * x - 1 = 0

theorem find_a_and_root (a x : ℝ)
  (h_eqn : equation_has_double_root a x)
  (h_discriminant : 16 + 4 * a = 0) :
  a = -4 ∧ x = 1 / 2 :=
sorry

end find_a_and_root_l774_774194


namespace number_of_solutions_l774_774756

def greatest_integer (x : ℝ) : ℤ :=
  ⌊x⌋

theorem number_of_solutions (n : ℤ) (hn : 1 ≤ n) :
  set.card {x : ℝ | (1 : ℝ) ≤ x ∧ x ≤ (n : ℝ) ∧ x^2 - (greatest_integer x)^2 = (x - greatest_integer x)^2 } = (n^2 - n + 1 : ℤ) :=
by
  sorry

end number_of_solutions_l774_774756


namespace matrix_power_example_l774_774758

theorem matrix_power_example (B : Matrix (Fin 2) (Fin 2) ℝ) (v : Fin 2 → ℝ)
  (h : B.mul_vec! v = ![9, -12]) :
  (B ^ 3).mul_vec! v = ![81, -108] :=
by
  have v_eq : v = ![3, -4] := by
    -- Establish v == [3, -4] from the provided condition
    sorry
  rw [v_eq] at h
  -- Now prove the main statement
  sorry

end matrix_power_example_l774_774758


namespace count_divisible_neither_5_nor_7_below_500_l774_774359

def count_divisible_by (n k : ℕ) : ℕ := (n - 1) / k

def count_divisible_by_5_or_7_below (n : ℕ) : ℕ :=
  let count_5 := count_divisible_by n 5
  let count_7 := count_divisible_by n 7
  let count_35 := count_divisible_by n 35
  count_5 + count_7 - count_35

def count_divisible_neither_5_nor_7_below (n : ℕ) : ℕ :=
  n - 1 - count_divisible_by_5_or_7_below n

theorem count_divisible_neither_5_nor_7_below_500 : count_divisible_neither_5_nor_7_below 500 = 343 :=
by
  sorry

end count_divisible_neither_5_nor_7_below_500_l774_774359


namespace find_ns_product_l774_774279

-- Definitions based on problem conditions
def S := {x : ℝ // x > 0}

def f (x : S) : ℝ := sorry

axiom functional_equation (x y : S) :
  f(x) * f(y) = f(x * y) + 2023 * (1 / x.val + 1 / y.val + 2022)

-- Main theorem statement to be proved
theorem find_ns_product (n s : ℝ) (h_n : n = 1) (h_s : s = (1 / 3) + 2022) :
  n * s = 6067 / 3 :=
by { rw [h_n, h_s], norm_num, }

end find_ns_product_l774_774279


namespace swimming_speed_in_still_water_l774_774071

theorem swimming_speed_in_still_water 
  (speed_of_water : ℝ) (distance : ℝ) (time : ℝ) (v : ℝ) 
  (h_water_speed : speed_of_water = 2) 
  (h_time_distance : time = 4 ∧ distance = 8) :
  v = 4 :=
by
  sorry

end swimming_speed_in_still_water_l774_774071


namespace perfect_square_factors_450_l774_774626

theorem perfect_square_factors_450 : 
  ∃ n, n = 4 ∧ (∀ k | 450, k ∣ 450 → is_square k) → (∃ m1 m2 m3 : ℕ, 450 = 2^m1 * 3^m2 * 5^m3 ) :=
by sorry

end perfect_square_factors_450_l774_774626


namespace hands_coincide_7_30_7_45_l774_774266

noncomputable def clock_hands: ℝ → ℝ × ℝ :=
  λ t, (30 * t / 60 + 0.5 * t, 6 * t)

noncomputable def clock_hands_overlap (t: ℝ): Prop :=
  (30 * (t / 60) + 0.5 * t = 6 * t)

def in_interval (a b t: ℝ): Prop :=
  a ≤ t ∧ t ≤ b

theorem hands_coincide_7_30_7_45 :
  ∃ t: ℝ, clock_hands_overlap (7 * 60 + t) ∧ in_interval 30 45 t :=
  by
  sorry

end hands_coincide_7_30_7_45_l774_774266


namespace overlap_area_l774_774103

def point := (ℝ × ℝ)

def rectangle_vertices : list point := [(0, 0), (3, 0), (3, 2), (0, 2)]
def triangle_vertices : list point := [(2, 0), (2, 2), (4, 2)]

-- Area function for a triangle given the vertices
noncomputable def triangle_area (A B C : point) : ℝ :=
  let (x1, y1) := A in
  let (x2, y2) := B in
  let (x3, y3) := C in
  0.5 * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

def overlap_vertices : list point := [(2, 0), (3, 0), (3, 2)]

theorem overlap_area : triangle_area (2, 0) (3, 0) (3, 2) = 1 :=
by {
  -- Simplifying the area calculation
  sorry
}

end overlap_area_l774_774103


namespace bob_weekly_increase_l774_774964

def raise_per_hour : ℝ := 0.50
def hours_per_week : ℝ := 40
def monthly_reduction : ℝ := 60
def weeks_per_month : ℝ := 4

theorem bob_weekly_increase : 
  let weekly_income_increase := raise_per_hour * hours_per_week,
      weekly_reduction := monthly_reduction / weeks_per_month,
      net_weekly_increase := weekly_income_increase - weekly_reduction in
  net_weekly_increase = 5 := by
  sorry -- Proof to be completed

end bob_weekly_increase_l774_774964


namespace fraction_subtraction_l774_774865

theorem fraction_subtraction :
  ∀ (a b : ℕ),
    a = 3 + 5 + 7 →
    b = 2 + 4 + 6 →
    ((a : ℚ) / b - (b : ℚ) / a) = 9 / 20 :=
by
  intros a b ha hb
  rw [ha, hb]
  norm_num

variable : ℕ
#eval 3+5+7
#eval 2+4+6

end fraction_subtraction_l774_774865


namespace men_count_in_first_group_is_20_l774_774051

noncomputable def men_needed_to_build_fountain (work1 : ℝ) (days1 : ℕ) (length1 : ℝ) (workers2 : ℕ) (days2 : ℕ) (length2 : ℝ) (work_per_man_per_day2 : ℝ) : ℕ :=
  let work_per_day2 := length2 / days2
  let work_per_man_per_day2 := work_per_day2 / workers2
  let total_work1 := length1 / days1
  Nat.floor (total_work1 / work_per_man_per_day2)

theorem men_count_in_first_group_is_20 :
  men_needed_to_build_fountain 56 6 56 35 3 49 (49 / (35 * 3)) = 20 :=
by
  sorry

end men_count_in_first_group_is_20_l774_774051


namespace commodity_price_l774_774812

-- Define the variables for the prices of the commodities.
variable (x y : ℝ)

-- Conditions given in the problem.
def total_cost (x y : ℝ) : Prop := x + y = 827
def price_difference (x y : ℝ) : Prop := x = y + 127

-- The main statement to prove.
theorem commodity_price (x y : ℝ) (h1 : total_cost x y) (h2 : price_difference x y) : x = 477 :=
by
  sorry

end commodity_price_l774_774812


namespace find_M_l774_774524

theorem find_M : ∃ (M : ℕ), 16^2 * 40^2 = 20^2 * M^2 ∧ M = 32 := by
  use 32
  have h1 : 16^2 * 40^2 = 2^(8 + 6) * 5^2 := by
    calc
      16^2 * 40^2 = (2^4)^2 * (2^3 * 5)^2 : by norm_num
              ... = 2^8 * (2^3 * 5)^2 : by rw [pow_mul] 
              ... = 2^8 * (2^3)^2 * (5)^2 : by rw [mul_pow]
              ... = 2^8 * 2^6 * 5^2 : by norm_num 
              ... = 2^(8 + 6) * 5^2 : by norm_num

  have h2 : 20^2 = (2^2 * 5)^2 := by norm_num
  have h3 : 20^2 = 2^4 * 5^2 := by rw [mul_pow, pow_mul]

  have h4 : 20^2 * 32^2 = (2^4 * 5^2) * 2^(10) := by
    calc
      20^2 * 32^2 = (2^4 * 5^2) * 2^(10) : by rw [←h3, ←h1]
              ... = (2^(4 + 10))* 5^2 : by rw [←pow_add]

  have h5 : 2^(14) * 5^2 = (2^(4 + 10)) * 5^2 := by norm_num
  exact ⟨32, ⟨by rw [←h4, ←h5], rfl⟩⟩

end find_M_l774_774524


namespace box_volume_increase_l774_774933

theorem box_volume_increase (l w h : ℝ) 
  (h1 : l * w * h = 5000)
  (h2 : l * w + w * h + h * l = 900)
  (h3 : l + w + h = 60) : 
  (l + 2) * (w + 2) * (h + 2) = 7048 := 
by 
  sorry

end box_volume_increase_l774_774933


namespace seating_arrangements_count_l774_774475

-- Define the problem's domain
inductive Person
| Alice | Bob | Carla | Derek | Eric

open Person

-- Function to check if two people can sit next to each other based on the given conditions
def validPair (p1 p2 : Person) : Bool :=
  match p1, p2 with
  | Alice, Bob => false
  | Bob, Alice => false
  | Carla, Alice => false
  | Carla, Derek => false
  | Alice, Carla => false
  | Derek, Carla => false
  | _, _ => true

-- Function to check if a seating arrangement is valid given the conditions
def validSeating (seating: List Person) : Bool :=
  List.allPairs seating (λ p1 p2 => validPair p1 p2)

-- Problem definition: Prove the number of valid seating arrangements is 20
theorem seating_arrangements_count :
  (List.permutations [Alice, Bob, Carla, Derek, Eric]).count (λ p => validSeating p) = 20 := 
sorry

end seating_arrangements_count_l774_774475


namespace numPerfectSquareFactorsOf450_l774_774685

def isPerfectSquare (n : Nat) : Prop :=
  ∃ k : Nat, n = k * k

theorem numPerfectSquareFactorsOf450 : 
  ∃! n : Nat, 
    (∀ d : Nat, d ∣ 450 → isPerfectSquare d) → n = 4 := 
by
  sorry

end numPerfectSquareFactorsOf450_l774_774685


namespace hunter_probability_same_l774_774060

noncomputable theory

def probability_correct_with_one_dog (p : ℝ) : ℝ :=
  p

def probability_correct_with_two_dogs (p : ℝ) : ℝ :=
  let both_correct := p * p
  let one_correct_one_incorrect := 2 * (p * (1 - p))
  both_correct * 1 + one_correct_one_incorrect * (1 / 2) + (1 - p) ^ 2 * 0

theorem hunter_probability_same (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) :
  probability_correct_with_one_dog p = probability_correct_with_two_dogs p :=
by
  -- The proof will go here
  sorry

end hunter_probability_same_l774_774060


namespace maximum_value_of_func_l774_774185

noncomputable def func (x : ℝ) : ℝ := 4 * x - 2 + 1 / (4 * x - 5)

theorem maximum_value_of_func (x : ℝ) (h : x < 5 / 4) : ∃ y, y = 1 ∧ ∀ z, z = func x → z ≤ y :=
sorry

end maximum_value_of_func_l774_774185


namespace average_y_min_value_y_max_total_score_first_5_games_l774_774241

-- Define notions of scoring and averaging
variable {x y : ℕ}
variable scores6 : ℕ := 22
variable scores7 : ℕ := 15
variable scores8 : ℕ := 12
variable scores9 : ℕ := 19

-- Lean's theorem prover to define each goal

theorem average_y (x : ℕ) :
  y = (5 * x + 68) / 9 :=
by
  unfold y
  sorry

theorem min_value_y :
  y ≥ 12 :=
by
  unfold y
  have hx : x ≥ 8 := sorry
  have h_ineq : y ≥ (5 * 8 + 68) / 9 := sorry
  exact h_ineq

theorem max_total_score_first_5_games (hx : x < 17) :
  5 * x - 1 ≤ 84 :=
by
  unfold y
  sorry

end average_y_min_value_y_max_total_score_first_5_games_l774_774241


namespace box_volume_increase_l774_774934

theorem box_volume_increase (l w h : ℝ) 
  (h1 : l * w * h = 5000)
  (h2 : l * w + w * h + h * l = 900)
  (h3 : l + w + h = 60) : 
  (l + 2) * (w + 2) * (h + 2) = 7048 := 
by 
  sorry

end box_volume_increase_l774_774934


namespace caller_wins_both_at_35_l774_774873

theorem caller_wins_both_at_35 (n : ℕ) :
  ∀ n, (n % 5 = 0 ∧ n % 7 = 0) ↔ n = 35 :=
by
  sorry

end caller_wins_both_at_35_l774_774873


namespace ellipse_equation_constant_dot_product_fixed_point_exists_l774_774565

-- Definitions based on conditions
def a := 2
def b := sqrt 2
def ellipse_eq (x y : ℝ) := (x^2 / a^2) + (y^2 / b^2) = 1
def C := (-2, 0)
def D := (2, 0)
def M (y₀ : ℝ) := (2, y₀)
def P (x₁ y₁ : ℝ) := (x₁, y₁)
def OP (x₁ y₁ : ℝ) := (x₁, y₁)
def OM (y₀ : ℝ) := (2, y₀)
def Q := (0, 0)

-- Proof statements
theorem ellipse_equation : ellipse_eq 4 2 := sorry

theorem constant_dot_product (y₀ : ℝ) (x₁ y₁ : ℝ) 
    (hP : ellipse_eq x₁ y₁) (hOP : OP x₁ y₁) (hOM : OM y₀) :
  OP x₁ y₁ • OM y₀ = 4 := sorry

theorem fixed_point_exists (y₀ : ℝ) (x₁ y₁ : ℝ)
    (hP : ellipse_eq x₁ y₁) (hM : M y₀) : 
  ∃ Qx Qy, Qx = 0 ∧ Qy = 0 ∧ circle_with_diameter (M y₀) (P x₁ y₁) Q :=
sorry

end ellipse_equation_constant_dot_product_fixed_point_exists_l774_774565


namespace find_g_values_l774_774588

variables (f g : ℝ → ℝ)

-- Conditions
axiom cond1 : ∀ x y, g (x - y) = g x * g y + f x * f y
axiom cond2 : f (-1) = -1
axiom cond3 : f 0 = 0
axiom cond4 : f 1 = 1

-- Goal
theorem find_g_values : g 0 = 1 ∧ g 1 = 0 ∧ g 2 = -1 :=
by
  sorry

end find_g_values_l774_774588


namespace dividend_percentage_calc_l774_774062

def price_per_share (face_value premium : ℝ) : ℝ :=
  face_value + premium * face_value

def number_of_shares (investment share_price : ℝ) : ℝ :=
  investment / share_price

def dividend_per_share (total_dividend num_shares : ℝ) : ℝ :=
  total_dividend / num_shares

def dividend_percentage (div_per_share face_value : ℝ) : ℝ :=
  (div_per_share / face_value) * 100

theorem dividend_percentage_calc :
  let face_value := 100.0
  let premium := 0.2
  let investment := 14400.0
  let total_dividend := 600.0
  let share_price := price_per_share face_value premium
  let num_shares := number_of_shares investment share_price
  let div_per_share := dividend_per_share total_dividend num_shares
  in dividend_percentage div_per_share face_value = 5 := by
  sorry

end dividend_percentage_calc_l774_774062


namespace probability_same_color_l774_774954

section
variable (black brown : ℕ)
variable (total_chairs : ℕ := black + brown)
variable (prob_same_color : ℚ := (black / (total_chairs : ℚ)) * ((black - 1) / (total_chairs - 1)) + 
                                  (brown / (total_chairs : ℚ)) * ((brown - 1) / (total_chairs - 1)))

theorem probability_same_color (h_black : black = 15) (h_brown : brown = 18) :
  prob_same_color black brown ≈ 0.489 := by
  simp [*] -- simplification can be done here; detailed proof steps are omitted
  sorry
end

end probability_same_color_l774_774954


namespace angle_XWY_l774_774727

/-- In triangle XYZ, altitudes XK and YL intersect at a point W.
    Given ∠XYZ = 52° and ∠XZY = 64°, we aim to prove that ∠XWY = 116°. -/
theorem angle_XWY (X Y Z W K L : Type) [In_triangle X Y Z]
  (altitude_XK : Altitude X Y Z K W)
  (altitude_YL : Altitude Y Z X L W)
  (angle_XYZ : ∠XYZ = 52)
  (angle_XZY : ∠XZY = 64) :
  ∠XWY = 116 :=
by
  sorry

end angle_XWY_l774_774727


namespace spending_less_l774_774908

-- Define the original costs in USD for each category.
def cost_A_usd : ℝ := 520
def cost_B_usd : ℝ := 860
def cost_C_usd : ℝ := 620

-- Define the budget cuts for each category.
def cut_A : ℝ := 0.25
def cut_B : ℝ := 0.35
def cut_C : ℝ := 0.30

-- Conversion rate from USD to EUR.
def conversion_rate : ℝ := 0.85

-- Sales tax rate.
def tax_rate : ℝ := 0.07

-- Calculate the reduced cost after budget cuts for each category.
def reduced_cost_A_usd := cost_A_usd * (1 - cut_A)
def reduced_cost_B_usd := cost_B_usd * (1 - cut_B)
def reduced_cost_C_usd := cost_C_usd * (1 - cut_C)

-- Convert costs from USD to EUR.
def reduced_cost_A_eur := reduced_cost_A_usd * conversion_rate
def reduced_cost_B_eur := reduced_cost_B_usd * conversion_rate
def reduced_cost_C_eur := reduced_cost_C_usd * conversion_rate

-- Calculate the total reduced cost in EUR before tax.
def total_reduced_cost_eur := reduced_cost_A_eur + reduced_cost_B_eur + reduced_cost_C_eur

-- Calculate the tax amount on the reduced cost.
def tax_reduced_cost := total_reduced_cost_eur * tax_rate

-- Total reduced cost in EUR after tax.
def total_reduced_cost_with_tax := total_reduced_cost_eur + tax_reduced_cost

-- Calculate the original costs in EUR without any cuts.
def original_cost_A_eur := cost_A_usd * conversion_rate
def original_cost_B_eur := cost_B_usd * conversion_rate
def original_cost_C_eur := cost_C_usd * conversion_rate

-- Calculate the total original cost in EUR before tax.
def total_original_cost_eur := original_cost_A_eur + original_cost_B_eur + original_cost_C_eur

-- Calculate the tax amount on the original cost.
def tax_original_cost := total_original_cost_eur * tax_rate

-- Total original cost in EUR after tax.
def total_original_cost_with_tax := total_original_cost_eur + tax_original_cost

-- Difference in spending.
def spending_difference := total_original_cost_with_tax - total_reduced_cost_with_tax

-- Prove the company must spend €561.1615 less.
theorem spending_less : spending_difference = 561.1615 := 
by 
  sorry

end spending_less_l774_774908


namespace circle_radius_increase_l774_774363

theorem circle_radius_increase (r r' : ℝ) (h : π * r'^2 = (25.44 / 100 + 1) * π * r^2) : 
  (r' - r) / r * 100 = 12 :=
by sorry

end circle_radius_increase_l774_774363


namespace log_base_half_iff_l774_774162

theorem log_base_half_iff (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (log (1/2) a > log (1/2) b) ↔ (a < b) :=
sorry

end log_base_half_iff_l774_774162


namespace percentage_of_125_equals_75_l774_774052

theorem percentage_of_125_equals_75 (p : ℝ) (h : p * 125 = 75) : p = 60 / 100 :=
by
  sorry

end percentage_of_125_equals_75_l774_774052


namespace train_speed_l774_774851

theorem train_speed (v : ℝ) : (∃ t : ℝ, 2 * v + t * v = 285 ∧ t = 285 / 38) → v = 30 :=
by
  sorry

end train_speed_l774_774851


namespace dice_no_1_or_6_probability_l774_774160

theorem dice_no_1_or_6_probability :
  let outcomes := {1, 2, 3, 4, 5, 6}
  let favourable_outcomes := {2, 3, 4, 5}
  let single_die_probability := (set.card favourable_outcomes) / (set.card outcomes)
  (single_die_probability)^4 = 16 / 81 :=
by 
  sorry

end dice_no_1_or_6_probability_l774_774160


namespace perfect_square_divisors_count_450_l774_774645

theorem perfect_square_divisors_count_450 :
  let is_divisor (d n : ℕ) := n % d = 0
  let is_perfect_square (n : ℕ) := ∃ m : ℕ, m * m = n
  let prime_factorization_450 := [(2, 1), (3, 2), (5, 2)]
  ∃ (count : ℕ), count = 4 ∧ count = (Finset.filter 
    (λ d, is_perfect_square d ∧ is_divisor d 450)
    (Finset.powersetPrimeFactors prime_factorization_450)).card := sorry

end perfect_square_divisors_count_450_l774_774645


namespace exponential_comparison_l774_774166

theorem exponential_comparison
  (a : ℕ := 3^55)
  (b : ℕ := 4^44)
  (c : ℕ := 5^33) :
  c < a ∧ a < b :=
by
  sorry

end exponential_comparison_l774_774166


namespace gcd_of_3150_and_9800_is_350_l774_774019

-- Definition of the two numbers
def num1 : ℕ := 3150
def num2 : ℕ := 9800

-- The greatest common factor of num1 and num2 is 350
theorem gcd_of_3150_and_9800_is_350 : Nat.gcd num1 num2 = 350 := by
  sorry

end gcd_of_3150_and_9800_is_350_l774_774019


namespace parallelogram_x_y_values_l774_774116

def parallelogram_solve (x y : ℝ) : Prop :=
  (4 * x + 1 = 11 ∧ 10 * y - 3 = 5) → x + y = 3.3

theorem parallelogram_x_y_values :
  parallelogram_solve 2.5 0.8 :=
by
  intros h,
  have hx : 4 * 2.5 + 1 = 11 := by {
    calc 4 * 2.5 + 1 = 10 + 1 : by sorry
               ... = 11      : by sorry
  },
  have hy: 10 * 0.8 - 3 = 5 := by {
    calc 10 * 0.8 - 3 = 8 - 3 : by sorry
                ... = 5       : by sorry
  },
  have h_sum: 2.5 + 0.8 = 3.3 := by {
    calc 2.5 + 0.8 = 3.3 : by sorry
  },
  exact h_sum

#exit

end parallelogram_x_y_values_l774_774116


namespace shortest_side_of_triangle_l774_774232

noncomputable def triangle_ABC :=
  {A B C D E : Type}
  [metric_space.{0} A]
  [metric_space.{0} B]
  [metric_space.{0} C]
  [metric_space.{0} D]
  [metric_space.{0} E]

def length (x y : Type) [metric_space.{0} x] [metric_space.{0} y] := sorry

def is_trisected (A B C D E : Type) := sorry

theorem shortest_side_of_triangle
  (A B C D E : Type)
  [metric_space.{0} A]
  [metric_space.{0} B]
  [metric_space.{0} C]
  [metric_space.{0} D]
  [metric_space.{0} E]
  (BD : ℝ) (DE : ℝ) (EC : ℝ)
  (H1 : BD = 4)
  (H2 : DE = 2)
  (H3 : EC = 9)
  (H4 : length A B = 2 * length A C)
  (trisect : is_trisected A B C D E)
  : length A C = b := 
sorry

end shortest_side_of_triangle_l774_774232


namespace integral_value_l774_774429

open Real

/-- Given integral problem: ∫ (x^2 / sqrt(9 - x^2)), from 0 to 3/2 -/
noncomputable def definite_integral : ℝ :=
  ∫ x in 0..(3 / 2), x^2 / sqrt(9 - x^2)

theorem integral_value :
  definite_integral = (3 * π / 4) - (9 * sqrt 3 / 8) :=
  by
    -- Skip the proof for now
    sorry

end integral_value_l774_774429


namespace interest_rate_payment_plan_l774_774315

theorem interest_rate_payment_plan
  (purchase_price : ℕ)
  (down_payment : ℕ)
  (monthly_payment : ℕ)
  (payment_months : ℕ)
  (total_paid : ℕ := down_payment + monthly_payment * payment_months)
  (extra_paid : ℕ := total_paid - purchase_price)
  (interest_rate : ℚ := (extra_paid * 100) / purchase_price) :
  purchase_price = 112 →
  down_payment = 12 →
  monthly_payment = 10 →
  payment_months = 12 →
  interest_rate.roundNearestTenth = 17.9 :=
by
  sorry

end interest_rate_payment_plan_l774_774315


namespace sum_of_coefficients_l774_774027

noncomputable def binomial (n k : ℕ) : ℕ :=
if h : k ≤ n then Nat.choose n k else 0

theorem sum_of_coefficients :
  let expr := 2 - (2 / x^2)
  let expansion := expr^8
  let k_values := {11, 12, 13}
  sum (k_values.map (λ k, binomial 8 k)) = 0 :=
by sorry

end sum_of_coefficients_l774_774027


namespace sum_of_distances_const_iff_sum_of_normals_zero_l774_774320

variables {k : ℕ}
variables {polygon : fin k → ℝ × ℝ}  -- Vertex points of the convex polygon
variables (n : fin k → ℝ × ℝ)  -- Unit outward normal vectors
variables (M : fin k → ℝ × ℝ)  -- Arbitrary points on the sides
variables (X : ℝ × ℝ)  -- Interior point

noncomputable def distance_to_side (i : fin k) : ℝ :=
(inner ⟨X - M i⟩ (n i))

theorem sum_of_distances_const_iff_sum_of_normals_zero :
  (∀ X Y : (ℝ × ℝ), (∑ i, distance_to_side X i) = (∑ i, distance_to_side Y i)) ↔
  (∑ i, n i) = (0, 0) :=
sorry

end sum_of_distances_const_iff_sum_of_normals_zero_l774_774320


namespace shaded_area_l774_774378

-- Define the conditions
def side_length_square : ℝ := 10
def radius_circle : ℝ := 3 * Real.sqrt 2

-- Define the given areas
def area_square := side_length_square ^ 2
def area_one_circle := Real.pi * radius_circle ^ 2
def area_four_circles := 4 * area_one_circle

-- The theorem to prove
theorem shaded_area : area_square - area_four_circles = 100 - 72 * Real.pi :=
by
  -- This would involve the steps stated
  sorry

end shaded_area_l774_774378


namespace increasing_interval_theorem_l774_774225

-- Define the functions f and g
def f (ω : ℝ) (x : ℝ) := 2 * Real.sin (ω * x + (Real.pi / 4))
def g (x : ℝ) := 2 * Real.cos (2 * x - (Real.pi / 4))

-- Define the condition for the axis of symmetry being the same
def symmetry_condition (ω : ℝ) : Prop :=
  (2 * Real.pi / ω) = (2 * Real.pi / 2)

-- Define the increasing interval of f on [0, π] given ω = 2 
def increasing_interval_f : set ℝ := set.Icc 0 (Real.pi / 8)

-- Main theorem statement
theorem increasing_interval_theorem :
  ∀ (ω : ℝ), (ω > 0) → (symmetry_condition ω) →
  (∀ x ∈ set.Icc 0 Real.pi, f 2 x ∈ increasing_interval_f) :=
  by
    intros ω ω_pos symm_cond
    sorry

end increasing_interval_theorem_l774_774225


namespace radius_of_smaller_semicircle_l774_774740

theorem radius_of_smaller_semicircle :
  ∃ x : ℝ, 0 < x ∧
    let AB := 6 in
    let AC := 12 - x in
    let BC := 6 + x in
    (AB = 6) ∧ 
    (AC = 12 - x) ∧ 
    (BC = 6 + x) ∧
    (AB^2 + AC^2 = BC^2) ∧
    x = 4 := 
by
  use 4
  split
  { exact zero_lt_four }
  split
  { reflexivity }
  split
  { reflexivity }
  split
  { reflexivity }
  { sorry }

end radius_of_smaller_semicircle_l774_774740


namespace total_students_l774_774252

theorem total_students (T : ℕ) (h : 0.65 * T = 351) : T = 540 :=
sorry

end total_students_l774_774252


namespace ball_hits_ground_time_l774_774817

-- Define the parameters for the height equation
def height (t : ℝ) : ℝ := -20 * t^2 + 32 * t + 60

-- The main theorem we want to prove
theorem ball_hits_ground_time :
  ∃ t : ℝ, t = (4 + Real.sqrt 91) / 5 ∧ height t = 0 :=
by
  sorry

end ball_hits_ground_time_l774_774817


namespace has_two_roots_l774_774914

-- Define the discriminant of the quadratic trinomial
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Main Lean statement
theorem has_two_roots
  (a b c : ℝ)
  (h : discriminant a b c > 0) :
  discriminant (3 * a) (2 * (a + b)) (b + c) > 0 := by
  sorry

end has_two_roots_l774_774914


namespace expression_positive_l774_774326

variable {a b c : ℝ}

theorem expression_positive (h₀ : 0 < a ∧ a < 2) (h₁ : -2 < b ∧ b < 0) : 0 < b + a^2 :=
by
  sorry

end expression_positive_l774_774326


namespace perfect_square_factors_count_450_l774_774670

theorem perfect_square_factors_count_450 : 
  (∃ n : ℕ, n = 4 ∧ (∀ d : ℕ, d ∣ 450 → ∃ k : ℕ, d = k * k)) := sorry

end perfect_square_factors_count_450_l774_774670


namespace find_k_no_xy_term_l774_774361

theorem find_k_no_xy_term (k : ℝ) :
  (¬ ∃ x y : ℝ, (-x^2 - 3 * k * x * y - 3 * y^2 + 9 * x * y - 8) = (- x^2 - 3 * y^2 - 8)) → k = 3 :=
by
  sorry

end find_k_no_xy_term_l774_774361


namespace area_BEIH_l774_774704

noncomputable def point (x y : ℝ) : ℝ × ℝ := (x, y)

noncomputable def area_quad (B E I H : ℝ × ℝ) : ℝ :=
  (1/2) * ((B.1 * E.2 + E.1 * I.2 + I.1 * H.2 + H.1 * B.2) - (B.2 * E.1 + E.2 * I.1 + I.2 * H.1 + H.2 * B.1))

theorem area_BEIH :
  let A : ℝ × ℝ := point 0 3
  let B : ℝ × ℝ := point 0 0
  let C : ℝ × ℝ := point 3 0
  let D : ℝ × ℝ := point 3 3
  let E : ℝ × ℝ := point 0 2
  let F : ℝ × ℝ := point 1 0
  let I : ℝ × ℝ := point (3/10) 2.1
  let H : ℝ × ℝ := point (3/4) (3/4)
  area_quad B E I H = 1.0125 :=
by
  sorry

end area_BEIH_l774_774704


namespace flowers_given_l774_774328

theorem flowers_given (initial total : ℕ) (h_initial : initial = 67) (h_total : total = 90) :
  ∃ x, x = total - initial ∧ x = 23 := by
  use 23
  split
  · simp [h_initial, h_total]
    sorry

end flowers_given_l774_774328


namespace _l774_774903

noncomputable def gear_speeds_relationship (x y z : ℕ) (ω₁ ω₂ ω₃ : ℝ) 
  (h1 : 2 * x * ω₁ = 3 * y * ω₂)
  (h2 : 3 * y * ω₂ = 4 * z * ω₃) : Prop :=
  ω₁ = (2 * z / x) * ω₃ ∧ ω₂ = (4 * z / (3 * y)) * ω₃

-- Example theorem statement
example (x y z : ℕ) (ω₁ ω₂ ω₃ : ℝ)
  (h1 : 2 * x * ω₁ = 3 * y * ω₂)
  (h2 : 3 * y * ω₂ = 4 * z * ω₃) : gear_speeds_relationship x y z ω₁ ω₂ ω₃ h1 h2 :=
by sorry

end _l774_774903


namespace tan_A_triangle_l774_774263

theorem tan_A_triangle (A B C : Type)
  [IsRightTriangle A B C]  -- condition 1: right triangle, angle B = 90°
  (BC : ℝ) (hBC : BC = 5)  -- condition 2: BC = 5
  (AB : ℝ) (hAB : AB = Real.sqrt 34)  -- condition 3: AB = sqrt(34)
  : Real.tan (angle A) = 5 / 3 := by
  sorry

end tan_A_triangle_l774_774263


namespace GouguPrinciple_l774_774339

-- Definitions according to conditions
def volumes_not_equal (A B : Type) : Prop := sorry -- p: volumes of A and B are not equal
def cross_sections_not_equal (A B : Type) : Prop := sorry -- q: cross-sectional areas of A and B are not always equal

-- The theorem to be proven
theorem GouguPrinciple (A B : Type) (h1 : volumes_not_equal A B) : cross_sections_not_equal A B :=
sorry

end GouguPrinciple_l774_774339


namespace ball_arrangements_correct_l774_774710

-- Define the number of balls of each color
def black_balls := 2
def white_balls := 3
def red_balls := 4

-- Define the condition: no black ball is next to a white ball
def no_black_next_to_white (arrangement : List Nat) : Prop :=
  ∀ i, (i < arrangement.length - 1) → 
       (arrangement[i] = 1 → arrangement[i+1] ≠ 2) ∧ 
       (arrangement[i] = 2 → arrangement[i+1] ≠ 1)

-- Given the number of balls, the total arrangements meeting the condition
def total_arrangements := 200

-- Statement to prove in Lean 4
theorem ball_arrangements_correct :
  ∃ arrangements : List (List Nat),
    (arrangements.length = total_arrangements) ∧
    (∀ arrangement ∈ arrangements,
      arrangement.count 1 = black_balls ∧
      arrangement.count 2 = white_balls ∧
      arrangement.count 3 = red_balls ∧
      no_black_next_to_white arrangement) :=
sorry

end ball_arrangements_correct_l774_774710


namespace largest_T_l774_774142

theorem largest_T (T : ℝ) (a b c d e : ℝ) 
  (h1: a ≥ 0) (h2: b ≥ 0) (h3: c ≥ 0) (h4: d ≥ 0) (h5: e ≥ 0)
  (h_sum : a + b = c + d + e)
  (h_T : T ≤ (Real.sqrt 30) / (30 + 12 * Real.sqrt 6)) : 
  Real.sqrt (a^2 + b^2 + c^2 + d^2 + e^2) ≥ T * (Real.sqrt a + Real.sqrt b + Real.sqrt c + Real.sqrt d + Real.sqrt e)^2 :=
sorry

end largest_T_l774_774142


namespace minimum_positive_period_of_f_l774_774204

noncomputable def f (x : ℝ) : ℝ :=
  let a := (Real.sin x, Real.cos x)
  let b := (Real.sin x, Real.sin x)
  a.1 * b.1 + a.2 * b.2

theorem minimum_positive_period_of_f :
  ∃ T > 0, ∀ x, f(x + T) = f(x) ∧ 
  ∀ T' > 0, (∀ x, f(x + T') = f(x)) → T' ≥ T := 
sorry

end minimum_positive_period_of_f_l774_774204


namespace max_diagonals_in_convex_polygon_l774_774449

theorem max_diagonals_in_convex_polygon (n : ℕ) (h : n = 2011) (convex : convex_polygon n) :
    max_diagonals n convex = 4016 :=
sorry

end max_diagonals_in_convex_polygon_l774_774449


namespace equal_areas_RPK_RQL_l774_774700

variables {A B C R P Q K L : Type}
variables [triangle A B C] [bisector B C A R] [circumcircle A B C R]
variables [perp_bisector B C P] [perp_bisector A C Q]
variables [midpoint K B C] [midpoint L A C]

theorem equal_areas_RPK_RQL :
  let area_RPK := triangle_area R P K in
  let area_RQL := triangle_area R Q L in
  area_RPK = area_RQL :=
sorry

end equal_areas_RPK_RQL_l774_774700


namespace swans_after_10_years_l774_774136

-- Defining the initial conditions
def initial_swans : ℕ := 15

-- Condition that the number of swans doubles every 2 years
def double_every_two_years (n t : ℕ) : ℕ := n * (2 ^ (t / 2))

-- Prove that after 10 years, the number of swans will be 480
theorem swans_after_10_years : double_every_two_years initial_swans 10 = 480 :=
by
  sorry

end swans_after_10_years_l774_774136


namespace volume_of_pyramid_l774_774045

theorem volume_of_pyramid (V_cube : ℝ) (h : ℝ) (A : ℝ) (V_pyramid : ℝ) : 
  V_cube = 27 → 
  h = 3 → 
  A = 4.5 → 
  V_pyramid = (1/3) * A * h → 
  V_pyramid = 4.5 := 
by 
  intros V_cube_eq h_eq A_eq V_pyramid_eq 
  sorry

end volume_of_pyramid_l774_774045


namespace hot_dog_cost_l774_774958

theorem hot_dog_cost : 
  ∃ h d : ℝ, (3 * h + 4 * d = 10) ∧ (2 * h + 3 * d = 7) ∧ (d = 1) := 
by 
  sorry

end hot_dog_cost_l774_774958


namespace eval_sqrt_4_8_pow_12_l774_774129

theorem eval_sqrt_4_8_pow_12 : ((8 : ℝ)^(1 / 4))^12 = 512 :=
by
  -- This is where the proof steps would go 
  sorry

end eval_sqrt_4_8_pow_12_l774_774129


namespace totalFriendsAreFour_l774_774308

-- Define the friends
def friends := ["Mary", "Sam", "Keith", "Alyssa"]

-- Define the number of friends
def numberOfFriends (f : List String) : ℕ := f.length

-- Claim that the number of friends is 4
theorem totalFriendsAreFour : numberOfFriends friends = 4 :=
by
  -- Skip proof
  sorry

end totalFriendsAreFour_l774_774308


namespace range_of_a_l774_774760

open Set

variable {x a : ℝ}

def p (x a : ℝ) := x^2 + 2 * a * x - 3 * a^2 < 0 ∧ a > 0
def q (x : ℝ) := x^2 + 2 * x - 8 < 0

theorem range_of_a (h : ∀ x, p x a → q x): 0 < a ∧ a ≤ 4 / 3 := 
  sorry

end range_of_a_l774_774760


namespace proof_exists_special_set_l774_774751

noncomputable def exists_special_set : Prop :=
  ∃ S : Set ℕ, S.card = 2012 ∧
    ∀ a b ∈ S, a ≠ b → Nat.gcd a b > 1 ∧
    ∀ a b c ∈ S, a ≠ b → b ≠ c → a ≠ c → Nat.gcd (Nat.gcd a b) c = 1

theorem proof_exists_special_set : exists_special_set := 
  sorry

end proof_exists_special_set_l774_774751


namespace triangle_sin_geometric_progression_l774_774726

theorem triangle_sin_geometric_progression
  (A B C : ℝ)
  (h_non_neg : 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π)
  (h_triangle : A + B + C = π)
  (h_geo_prog : ∃ r : ℝ, r > 0 ∧ (sin A = r * sin B) ∧ (sin C = r^2 * sin B)) :
  B ≤ π / 3 :=
sorry

end triangle_sin_geometric_progression_l774_774726


namespace polar_equation_l774_774827

theorem polar_equation (y ρ θ : ℝ) (x : ℝ) 
  (h1 : y = ρ * Real.sin θ) 
  (h2 : x = ρ * Real.cos θ) 
  (h3 : y^2 = 12 * x) : 
  ρ * (Real.sin θ)^2 = 12 * Real.cos θ := 
by
  sorry

end polar_equation_l774_774827


namespace cricket_bat_selling_price_l774_774897

theorem cricket_bat_selling_price
    (profit : ℝ)
    (profit_percentage : ℝ)
    (CP : ℝ)
    (SP : ℝ)
    (h_profit : profit = 255)
    (h_profit_percentage : profit_percentage = 42.857142857142854)
    (h_CP : CP = 255 * 100 / 42.857142857142854)
    (h_SP : SP = CP + profit) :
    SP = 850 :=
by
  skip -- This is where the proof would go
  sorry -- Placeholder for the required proof

end cricket_bat_selling_price_l774_774897


namespace shaded_area_of_rotated_square_is_four_thirds_l774_774940

noncomputable def common_shaded_area_of_rotated_square (β : ℝ) (h1 : 0 < β) (h2 : β < π / 2) (h_cos_beta : Real.cos β = 3 / 5) : ℝ :=
  let side_length := 2
  let area := side_length * side_length / 3 * 2
  area

theorem shaded_area_of_rotated_square_is_four_thirds
  (β : ℝ)
  (h1 : 0 < β)
  (h2 : β < π / 2)
  (h_cos_beta : Real.cos β = 3 / 5) :
  common_shaded_area_of_rotated_square β h1 h2 h_cos_beta = 4 / 3 :=
sorry

end shaded_area_of_rotated_square_is_four_thirds_l774_774940


namespace total_cost_is_correct_l774_774745

noncomputable def nights : ℕ := 3
noncomputable def cost_per_night : ℕ := 250
noncomputable def discount : ℕ := 100

theorem total_cost_is_correct :
  (nights * cost_per_night) - discount = 650 := by
sorry

end total_cost_is_correct_l774_774745


namespace toby_pulls_loaded_sled_distance_l774_774849

theorem toby_pulls_loaded_sled_distance :
  ∃ (x : ℝ), (x / 10 + 120 / 20 + 80 / 10 + 140 / 20 = 39) ∧ x = 180 :=
by
  exists 180
  split
  calc
    180 / 10 + 120 / 20 + 80 / 10 + 140 / 20 = 18 + 6 + 8 + 7 : by norm_num
    ... = 39 : by norm_num
  norm_num

end toby_pulls_loaded_sled_distance_l774_774849


namespace question1_question2_l774_774583

noncomputable def monotonically_increasing_range (f : ℝ → ℝ) : Prop :=
∀ x y : ℝ, x ≤ y → f x ≤ f y

noncomputable def fₐ (a : ℝ) (x : ℝ) : ℝ := a * x - Real.sin x

-- Question 1: If function f(x) is monotonically increasing on ℝ, determine the range of the real number a.
theorem question1 (a : ℝ) 
  (h : monotonically_increasing_range (fₐ a)) : 
  1 ≤ a := 
sorry

noncomputable def f_half (x : ℝ) : ℝ := 0.5 * x - Real.sin x

-- bounds for critical points
noncomputable def interval := Set.Icc 0 (Real.pi / 2)

-- Question 2: When a = 1/2, find the maximum and minimum value of the function f(x) in the interval [0, π/2].
theorem question2 : 
  let m := Real.pi / 6 - Real.sqrt 3 / 2 in 
  Let M := 0 in
  ∀ x ∈ interval, 
  f_half x ≥ m ∧ f_half x ≤ M := 
sorry

end question1_question2_l774_774583


namespace log_gt_one_over_x_plus_one_l774_774531

theorem log_gt_one_over_x_plus_one (x : ℝ) (h : x > 0) : log (1 + x) > 1 / (x + 1) :=
by sorry

end log_gt_one_over_x_plus_one_l774_774531


namespace wall_bricks_count_l774_774965

def alice_rate (y : ℕ) : ℕ := y / 8
def bob_rate (y : ℕ) : ℕ := y / 12
def combined_rate (y : ℕ) : ℕ := (5 * y) / 24 - 12
def effective_working_time : ℕ := 6

theorem wall_bricks_count :
  ∃ y : ℕ, (combined_rate y * effective_working_time = y) ∧ y = 288 :=
by
  sorry

end wall_bricks_count_l774_774965


namespace value_of_x_l774_774215

theorem value_of_x : (∃ x : ℝ, (1 / 8) * 2 ^ 36 = 8 ^ x) → x = 11 := by
  intro h
  rcases h with ⟨x, hx⟩
  have h1 : 1 / 8 = 2 ^ (-3) := by norm_num
  rw [h1, ←pow_add] at hx
  norm_num at hx
  have h2 : 8 = 2 ^ 3 := by norm_num
  rw [h2, pow_mul] at hx
  norm_num at hx
  exact hx.symm

end value_of_x_l774_774215


namespace sum_floor_log4_l774_774971

noncomputable def floor_log4_sum : ℕ :=
  (List.range 256).sum (λ N => Int.to_nat (Int.floor (Real.log N / Real.log 4)))

theorem sum_floor_log4 : floor_log4_sum = 1252 := by
  sorry

end sum_floor_log4_l774_774971


namespace unique_covering_configurations_l774_774123

def right_angled_triangle : Type := sorry
def face (c : Type) : Type := sorry
def cube : Type := sorry
def white : right_angled_triangle → Prop := sorry
def black : right_angled_triangle → Prop := sorry
def sum_of_angles_at_vertex := sorry
def configurations : Type := sorry
def rotations_of_cube : configurations → configurations → Prop := sorry

theorem unique_covering_configurations
  (c : cube)
  (each_face_divided : ∀ f : face c, ∃ t1 t2 : right_angled_triangle, white t1 ∧ black t2)
  (angles_meeting_at_vertex : ∀ v, sum_of_angles_at_vertex v = sorry) :
  ∃! cfgs : Array configurations, cfgs.size = 2 ∧ ∀ (cfg1 cfg2 : configurations), rotations_of_cube cfg1 cfg2 → cfg1 = cfg2 :=
sorry

end unique_covering_configurations_l774_774123


namespace extracellular_proof_l774_774410

-- Define the components
def component1 : Set String := {"Na＋", "antibodies", "plasma proteins"}
def component2 : Set String := {"Hemoglobin", "O2", "glucose"}
def component3 : Set String := {"glucose", "CO2", "insulin"}
def component4 : Set String := {"Hormones", "neurotransmitter vesicles", "amino acids"}

-- Define the properties of being a part of the extracellular fluid
def is_extracellular (x : Set String) : Prop :=
  x = component1 ∨ x = component3

-- State the theorem to prove
theorem extracellular_proof : is_extracellular component1 ∧ ¬is_extracellular component2 ∧ is_extracellular component3 ∧ ¬is_extracellular component4 :=
by
  sorry

end extracellular_proof_l774_774410


namespace chord_length_tangent_to_ln_curve_l774_774586

noncomputable def length_of_chord_intercepted 
  (f : ℝ → ℝ) 
  (l : ℝ → ℝ) 
  (C : ℝ → ℝ → ℝ) := 
  2 * Real.sqrt (4 - (Real.sqrt(2) / 2)^2)

theorem chord_length_tangent_to_ln_curve :
  let f : ℝ → ℝ := λ x, x * Real.log x,
      l : ℝ → ℝ := λ x, x - 1,
      C : ℝ × ℝ := (2, 0) in
  length_of_chord_intercepted f l (λ x y, (x - 2)^2 + y^2) = Real.sqrt 14 :=
by
  sorry

end chord_length_tangent_to_ln_curve_l774_774586


namespace find_BC_line_eq_l774_774581

def line1_altitude : Prop := ∃ x y : ℝ, 2*x - 3*y + 1 = 0
def line2_altitude : Prop := ∃ x y : ℝ, x + y = 0
def vertex_A : Prop := ∃ a1 a2 : ℝ, a1 = 1 ∧ a2 = 2
def side_BC_equation : Prop := ∃ b c d : ℝ, b = 2 ∧ c = 3 ∧ d = 7

theorem find_BC_line_eq (H1 : line1_altitude) (H2 : line2_altitude) (H3 : vertex_A) : side_BC_equation :=
sorry

end find_BC_line_eq_l774_774581


namespace ivy_collectors_dolls_l774_774509

theorem ivy_collectors_dolls (dina_dolls ivy_dolls : ℕ) (h1 : dina_dolls = 2 * ivy_dolls) (h2 : dina_dolls = 60) : (2 * ivy_dolls / 3) = 20 :=
by 
  -- Calculating ivy_dolls from dina_dolls
  have ivy_calc : ivy_dolls = dina_dolls / 2,
  calc
    ivy_dolls = dina_dolls / 2 : by rw [←h1]
             ... = 60 / 2 : by rw h2
             ... = 30 : by norm_num,

  -- Plugging ivy_dolls value back into the 2/3 formula
  calc
    (2 * ivy_dolls / 3) = (2 * 30 / 3) : by rw ivy_calc
                      ... = 20 : by norm_num

end ivy_collectors_dolls_l774_774509


namespace trains_crossing_time_correct_l774_774041

noncomputable def train_crossing_time 
  (length_first_train length_second_train : ℕ) 
  (speed_first_train speed_second_train : ℕ) 
  (opp_direction : Prop) : ℕ :=
if opp_direction then (
  let relative_speed := (speed_first_train + speed_second_train) * (5 / 18) in
  let combined_length := length_first_train + length_second_train in
  let time_in_seconds := combined_length / relative_speed in
  time_in_seconds.to_nat
)
else
  0

theorem trains_crossing_time_correct : 
  train_crossing_time 200 300 60 40 true = 18 :=
by 
  -- Actual proof omitted
  sorry

end trains_crossing_time_correct_l774_774041


namespace numPerfectSquareFactorsOf450_l774_774680

def isPerfectSquare (n : Nat) : Prop :=
  ∃ k : Nat, n = k * k

theorem numPerfectSquareFactorsOf450 : 
  ∃! n : Nat, 
    (∀ d : Nat, d ∣ 450 → isPerfectSquare d) → n = 4 := 
by
  sorry

end numPerfectSquareFactorsOf450_l774_774680


namespace jonathan_typing_time_l774_774271

theorem jonathan_typing_time 
(J : ℕ) 
(h_combined_rate : (1 / (J : ℝ)) + (1 / 30) + (1 / 24) = 1 / 10) : 
  J = 40 :=
by {
  sorry
}

end jonathan_typing_time_l774_774271


namespace douglas_total_vote_percent_l774_774251

-- Definitions based on the conditions
def countyX_votes (V : ℕ) : ℕ := 2 * V
def countyY_votes (V : ℕ) : ℕ := V
def douglasX_votes (V : ℕ) : ℕ := 72 * countyX_votes(V) / 100
def douglasY_votes (V : ℕ) : ℕ := 36 * countyY_votes(V) / 100
def total_votes (V : ℕ) : ℕ := countyX_votes(V) + countyY_votes(V)
def douglas_total_votes (V : ℕ) : ℕ := douglasX_votes(V) + douglasY_votes(V)

theorem douglas_total_vote_percent (V : ℕ) : 
  100 * douglas_total_votes(V) / total_votes(V) = 60 := 
by
  sorry

end douglas_total_vote_percent_l774_774251


namespace trees_died_in_typhoon_l774_774596

-- Define the total number of trees, survived trees, and died trees
def total_trees : ℕ := 14

def survived_trees (S : ℕ) : ℕ := S

def died_trees (S : ℕ) : ℕ := S + 4

-- The Lean statement that formalizes the proof problem
theorem trees_died_in_typhoon : ∃ S : ℕ, survived_trees S + died_trees S = total_trees ∧ died_trees S = 9 :=
by
  -- Provide a placeholder for the proof
  sorry

end trees_died_in_typhoon_l774_774596


namespace perfect_square_factors_450_l774_774609

theorem perfect_square_factors_450 :
  ∃ (n : ℕ), n = 4 ∧ ∀ (d : ℕ), d ∣ 450 → ∃ (k : ℕ), k^2 = d ↔ d = 1 ∨ d = 9 ∨ d = 25 ∨ d = 225 :=
by
  sorry

end perfect_square_factors_450_l774_774609


namespace find_angle_B_find_area_ABC_l774_774173

variables {A B C : Type} [EuclideanGeometry A B C]
variables (a b c : ℝ) (angleB : ℝ) (area : ℝ)

-- Conditions
axiom triangle_ABC : is_triangle A B C
axiom sides_ABC : triangle_sides A B C a b c
axiom vectors_parallel : (a-c, a-b) ∥ (a+b, c)

-- Given values
axiom a_value : a = 1
axiom b_value : b = sqrt 7

noncomputable def angle_B : ℝ := Real.cosB a b c
noncomputable def area_ABC : ℝ := Triangle.area a b c angle_B

-- The proof statements
theorem find_angle_B :
  angle_B = π / 3 :=
sorry

theorem find_area_ABC :
  area = 3 * sqrt 3 / 4 :=
sorry

end find_angle_B_find_area_ABC_l774_774173


namespace neg_ex_triangle_symm_l774_774357

theorem neg_ex_triangle_symm :
  ¬ ∃ (T : Type) [Triangle T], symmetric T ↔ ∀ (T : Type) [Triangle T], ¬ symmetric T :=
by
  sorry

end neg_ex_triangle_symm_l774_774357


namespace joe_paint_usage_l774_774270

noncomputable def paint_used_after_four_weeks : ℝ := 
  let total_paint := 480
  let first_week_paint := (1/5) * total_paint
  let second_week_paint := (1/6) * (total_paint - first_week_paint)
  let third_week_paint := (1/7) * (total_paint - first_week_paint - second_week_paint)
  let fourth_week_paint := (2/9) * (total_paint - first_week_paint - second_week_paint - third_week_paint)
  first_week_paint + second_week_paint + third_week_paint + fourth_week_paint

theorem joe_paint_usage :
  abs (paint_used_after_four_weeks - 266.66) < 0.01 :=
sorry

end joe_paint_usage_l774_774270


namespace cone_liquid_levels_ratio_l774_774856

theorem cone_liquid_levels_ratio 
  (h₁ h₂ : ℝ) (r₁ r₂ : ℝ) (v₁ v₂ : ℝ) (marble_volume : ℝ) 
  (initial_volume : v₁ = v₂) 
  (r₁ : r₁ = 4) 
  (r₂ : r₂ = 8) 
  (marble_volume = (4/3) * π * 2^3) 
  : ((rise_in_smaller_cone : ℝ) / (rise_in_larger_cone: ℝ)) = 4 :=
by
  sorry

end cone_liquid_levels_ratio_l774_774856


namespace find_f_at_3_l774_774495

noncomputable def f (x : ℝ) : ℝ :=
  ((x + 1) * (x^2 + 1) * (x^4 + 1) * (x^8 + 1) * (x^16 + 1) * (x^32 + 1) - 1) / (x^(2^6 - 1) - 1)

theorem find_f_at_3 : f 3 = 3 :=
by
  sorry

end find_f_at_3_l774_774495


namespace max_committees_l774_774053

-- Lean 4 statement for the given problem
theorem max_committees (club : Type) (member : club → Prop) (committee : club → Prop)
  (H1 : ∃ S : set club, S.card = 25 ∧ ∀ x ∈ S, member x) 
  (H2 : ∀ c : set club, c.card = 5 → (∀ x ∈ c, committee x))
  (H3 : ∀ c1 c2 : set club, c1 ≠ c2 → c1.card = 5 → c2.card = 5 → (c1 ∩ c2).card ≤ 1) :
  ∃ upper_bound : ℕ, upper_bound = 30 ∧ ∀ list_of_committees : list (set club),
  (∀ c ∈ list_of_committees, c.card = 5) → (∀ (c1 c2 ∈ list_of_committees), c1 ≠ c2 → (c1 ∩ c2).card ≤ 1) → list_of_committees.length ≤ upper_bound :=
by
  sorry

end max_committees_l774_774053


namespace area_region_correct_l774_774861

def area_of_region : ℝ := 
  let eq := ∀ x y : ℝ, x^2 + y^2 - 7 = 4y - 14x + 3
  in 63 * Real.pi

theorem area_region_correct :
  (∀ x y : ℝ, x^2 + y^2 - 7 = 4y - 14x + 3) → area_of_region = 63 * Real.pi :=
by
  assume h : ∀ x y : ℝ, x^2 + y^2 - 7 = 4y - 14x + 3
  sorry

end area_region_correct_l774_774861


namespace consumer_installment_credit_l774_774877

-- Define the conditions.
def auto_installment_credit_percentage := (35 / 100)  -- 35%
def finance_companies_credit := 40  -- $40 billion
def finance_companies_credit_fraction := 1 / 3  -- 1/3

-- The problem statement.
theorem consumer_installment_credit
  (C : ℝ)  -- the total consumer installment credit
  (A : ℝ)  -- the total automobile installment credit
  (H1 : A = auto_installment_credit_percentage * C)  -- 1. Automobile installment credit is 35% of C.
  (H2 : finance_companies_credit = finance_companies_credit_fraction * A)  -- 2. Automobile finance companies extended $40 billion, which is 1/3 of the auto installment credit.
  : C = 342.857 :=
sorry  -- Proof is omitted.

end consumer_installment_credit_l774_774877


namespace find_sandwich_cost_l774_774016

theorem find_sandwich_cost (S : ℝ) :
  3 * S + 2 * 4 = 26 → S = 6 :=
by
  intro h
  sorry

end find_sandwich_cost_l774_774016


namespace rationalize_denom_l774_774325

theorem rationalize_denom (A B C : ℤ) (hC_pos : C > 0) (B_not_div : ∀ p : ℕ, prime p → ¬ (p^3 ∣ B)) :
  (∃ A B C, 2*3^3 = A*B ∧ 21 = C ∧ A + B + C = 72) :=
by {
  use [2, 49, 21],
  split,
  { rw mul_assoc, rw pow_succ, rw pow_two },
    split,
  { linarith }, 
  { linarith } 
}

end rationalize_denom_l774_774325


namespace polynomial_divisibility_l774_774536

def P (a : ℤ) (x : ℤ) : ℤ := x^1000 + a*x^2 + 9

theorem polynomial_divisibility (a : ℤ) : (P a (-1) = 0) ↔ (a = -10) := by
  sorry

end polynomial_divisibility_l774_774536


namespace midpoint_on_curve_line_ab_eq_neg_x_minus_1_l774_774749

open Real

-- Define the curve
def curve (a x : ℝ) := x^3 - 3 * a * x^2

-- Define the midpoint function
def midpoint (A B : ℝ × ℝ) : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Part 1: Prove that the midpoint lies on the curve
theorem midpoint_on_curve (a m : ℝ) 
  (x₊ x₋ : ℝ) 
  (hx₊ : x₊ = a + sqrt (a^2 + m / 3))
  (hx₋ : x₋ = a - sqrt (a^2 + m / 3)) :
  let A := (x₊, curve a x₊),
      B := (x₋, curve a x₋),
      C := midpoint A B in
  C.2 = curve a C.1 := sorry

-- Part 2: Prove that a = 1 and m = 4 given the line equation
theorem line_ab_eq_neg_x_minus_1 (a m : ℝ) 
  (hx : m = -1)
  (hab : ∀ x, -x - 1 = -1 * x - 1)
  : a = 1 ∧ m = 4 := sorry

end midpoint_on_curve_line_ab_eq_neg_x_minus_1_l774_774749


namespace ratio_of_unit_prices_l774_774482

variables (v p : ℝ)

def brand_z_volume := 1.3 * v
def brand_z_price  := 0.85 * p

theorem ratio_of_unit_prices :
  (brand_z_price v p / brand_z_volume v) / (p / v) = 17 / 26 :=
sorry

end ratio_of_unit_prices_l774_774482


namespace countValidSequences_l774_774600

def isValidSequence (seq : List ℤ) : Prop :=
  list.length seq = 8 ∧
  list.head seq = some 1 ∧
  list.getLast seq = some 3 ∧
  ∀ i, (i < 7) → ((seq.get! (i + 1) = seq.get! i + 1) ∨ (seq.get! (i + 1) = seq.get! i + 2)) ∧
  (seq.get! i = 0 → seq.get! (i + 1) = 1)

theorem countValidSequences : 
  (∃ seq, isValidSequence seq) → (finset.filter isValidSequence (finset.univ : finset (list ℤ))).card = 21 :=
by
  sorry

end countValidSequences_l774_774600


namespace intersect_point_l774_774350

noncomputable def f (x : ℤ) (b : ℤ) : ℤ := 5 * x + b
noncomputable def f_inv (x : ℤ) (b : ℤ) : ℤ := (x - b) / 5

theorem intersect_point (a b : ℤ) (h_intersections : (f (-3) b = a ∧ f a b = -3)) : a = -3 :=
by
  sorry

end intersect_point_l774_774350


namespace pow_mod_cycle_remainder_5_pow_2023_l774_774402

theorem pow_mod_cycle (n : ℕ) : (5^n % 6 = if n % 2 = 1 then 5 else 1) := 
by sorry

theorem remainder_5_pow_2023 : 5^2023 % 6 = 5 :=
by
  have cycle_properties : ∀ n, 5^n % 6 = (if n % 2 = 1 then 5 else 1) := pow_mod_cycle
  calc
    5^2023 % 6 = if 2023 % 2 = 1 then 5 else 1 := cycle_properties 2023
             ... = 5                   := by norm_num

end pow_mod_cycle_remainder_5_pow_2023_l774_774402


namespace propositionA_propositionB_l774_774870

theorem propositionA (α : ℝ) (h₀ : 0 < α) (h₁ : α < π / 2) : sin α > 0 := sorry

theorem propositionB : ∃ α : ℝ, 0 < α ∧ α < π / 2 ∧ cos (2 * α) > 0 :=
  by {
    use π / 12,
    split, linarith,
    split, linarith,
    have h_cos : cos (2 * (π / 12)) = cos (π / 6),
    rw cos_two_mul,
    norm_num,
    exact sorry
  }

end propositionA_propositionB_l774_774870


namespace solve_inequality_l774_774332

noncomputable def lg (n : ℝ) : ℝ := Real.log10 n

theorem solve_inequality (x : ℝ) (h₀ : lg 20 > 1) (h₁ : x ∈ Set.Ioo 0 Real.pi) :
  (lg 20)^(2 * Real.cos x) > 1 ↔ x ∈ Set.Ioo 0 (Real.pi / 2) :=
by
  sorry

end solve_inequality_l774_774332


namespace square_number_increased_decreased_by_five_remains_square_l774_774991

theorem square_number_increased_decreased_by_five_remains_square :
  ∃ x : ℤ, ∃ u v : ℤ, x^2 + 5 = u^2 ∧ x^2 - 5 = v^2 := by
  sorry

end square_number_increased_decreased_by_five_remains_square_l774_774991


namespace smallest_integer_M_exists_l774_774147

theorem smallest_integer_M_exists :
  ∃ (M : ℕ), 
    (M > 0) ∧ 
    (∃ (x y z : ℕ), 
      (x = M ∨ x = M + 1 ∨ x = M + 2) ∧ 
      (y = M ∨ y = M + 1 ∨ y = M + 2) ∧ 
      (z = M ∨ z = M + 1 ∨ z = M + 2) ∧ 
      ((x = M ∨ x = M + 1 ∨ x = M + 2) ∧ x % 8 = 0) ∧ 
      ((y = M ∨ y = M + 1 ∨ y = M + 2) ∧ y % 9 = 0) ∧ 
      ((z = M ∨ z = M + 1 ∨ z = M + 2) ∧ z % 25 = 0) ) ∧ 
    M = 200 := 
by
  sorry

end smallest_integer_M_exists_l774_774147


namespace box_volume_increase_l774_774932

theorem box_volume_increase (l w h : ℝ) 
  (h1 : l * w * h = 5000)
  (h2 : l * w + w * h + h * l = 900)
  (h3 : l + w + h = 60) : 
  (l + 2) * (w + 2) * (h + 2) = 7048 := 
by 
  sorry

end box_volume_increase_l774_774932


namespace cards_choice_ways_l774_774686

theorem cards_choice_ways (S : List Char) (cards : Finset (Char × ℕ)) :
  (∀ c ∈ cards, c.1 ∈ S) ∧
  (∀ (c1 c2 : Char × ℕ), c1 ∈ cards → c2 ∈ cards → c1 ≠ c2 → c1.1 ≠ c2.1) ∧
  (∃ c ∈ cards, c.2 = 1 ∧ c.1 = 'H') →
  (∃ c ∈ cards, c.2 = 1) →
  ∃ (ways : ℕ), ways = 1014 := 
sorry

end cards_choice_ways_l774_774686


namespace find_angle_A_find_range_expression_l774_774264

-- Define the variables and conditions in a way consistent with Lean's syntax
variables {α β γ : Type}
variables (a b c : ℝ) (A B C : ℝ)

-- The mathematical conditions translated to Lean
def triangle_condition (a b c A B C : ℝ) : Prop := (b + c) / a = Real.cos B + Real.cos C

-- Statement for Proof 1: Prove that A = π/2 given the conditions
theorem find_angle_A (h : triangle_condition a b c A B C) : A = Real.pi / 2 :=
sorry

-- Statement for Proof 2: Prove the range of the given expression under the given conditions
theorem find_range_expression (h : triangle_condition a b c A B C) (hA : A = Real.pi / 2) :
  ∃ (l u : ℝ), l = Real.sqrt 3 + 2 ∧ u = Real.sqrt 3 + 3 ∧ (2 * Real.cos (B / 2) ^ 2 + 2 * Real.sqrt 3 * Real.cos (C / 2) ^ 2) ∈ Set.Ioc l u :=
sorry

end find_angle_A_find_range_expression_l774_774264


namespace complex_roots_real_power_six_count_l774_774000

theorem complex_roots_real_power_six_count :
  let solutions := {z : ℂ | z ^ 24 = 1}
  (real_solutions := {z : ℂ | z ∈ solutions ∧ z ^ 6 ∈ ℝ}) 
  in
  solutions.card = 24 ∧ real_solutions.card = 12 :=
by
  sorry

end complex_roots_real_power_six_count_l774_774000


namespace perfect_square_factors_count_450_l774_774676

theorem perfect_square_factors_count_450 : 
  (∃ n : ℕ, n = 4 ∧ (∀ d : ℕ, d ∣ 450 → ∃ k : ℕ, d = k * k)) := sorry

end perfect_square_factors_count_450_l774_774676


namespace arithmetic_sequence_general_formula_bn_sequence_sum_l774_774250

/-- 
  In an arithmetic sequence {a_n}, a_2 = 5 and a_6 = 21. 
  Prove the general formula for the nth term a_n and the sum of the first n terms S_n. 
-/
theorem arithmetic_sequence_general_formula (a : ℕ → ℤ) (S : ℕ → ℤ) (h1 : a 2 = 5) (h2 : a 6 = 21) : 
  (∀ n, a n = 4 * n - 3) ∧ (∀ n, S n = n * (2 * n - 1)) := 
sorry

/--
  Given b_n = 2 / (S_n + 5 * n), prove the sum of the first n terms T_n for the sequence {b_n}.
-/
theorem bn_sequence_sum (a : ℕ → ℤ) (S : ℕ → ℤ) (b : ℕ → ℚ) (T : ℕ → ℚ) 
  (h1 : a 2 = 5) (h2 : a 6 = 21) 
  (ha : ∀ n, a n = 4 * n - 3) (hS : ∀ n, S n = n * (2 * n - 1)) 
  (hb : ∀ n, b n = 2 / (S n + 5 * n)) : 
  (∀ n, T n = 3 / 4 - 1 / (2 * (n + 1)) - 1 / (2 * (n + 2))) :=
sorry

end arithmetic_sequence_general_formula_bn_sequence_sum_l774_774250


namespace number_of_perfect_square_divisors_of_450_l774_774632

theorem number_of_perfect_square_divisors_of_450 : 
    let p := 450;
    let factors := [(3, 2), (5, 2), (2, 1)];
    ∃ n, (n = 4 ∧ 
          ∀ (d : ℕ), d ∣ p → 
                     (∃ (a b c : ℕ), d = 2^a * 3^b * 5^c ∧ 
                              (a = 0) ∧ (b = 0 ∨ b = 2) ∧ (c = 0 ∨ c = 2) → 
                              a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 = 0)) :=
    sorry

end number_of_perfect_square_divisors_of_450_l774_774632


namespace perfect_square_divisors_of_450_l774_774622

theorem perfect_square_divisors_of_450 :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let e1 := 1
  let e2 := 2
  let e3 := 2
  ∃ (count : ℕ), count = 4 ∧
  (∀ (d : ℕ), d ∣ (p1^e1 * p2^e2 * p3^e3) →
    (∀ (pe : ℕ × ℕ), pe ∈ (Multiset.ofList [(p1, e1), (p2, e2), (p3, e3)]).powerset → 
      (∃ (ex1 ex2 ex3: ℕ), ex1 ≤ e1 ∧ ex2 ≤ e2 ∧ ex3 ≤ e3 ∧ ex1 % 2 = 0 ∧ ex2 % 2 = 0 ∧ ex3 % 2 = 0))) :=
sorry

end perfect_square_divisors_of_450_l774_774622


namespace calc_annual_interest_rate_l774_774386

variables (P A T : ℝ) (x : ℝ)
variables (hx : x / 100 ∈ ℝ)

theorem calc_annual_interest_rate (hP : P = 3000) 
                                   (hA: A = 3243)
                                   (hT: T = 3) 
                                   (hx : x / 100 = x * 0.01) : 
  P + (P * (x * 0.01) * T) = A :=
by
  sorry

end calc_annual_interest_rate_l774_774386


namespace subsets_with_mean_six_l774_774209

open Finset

def originalSet : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12}

def targetSumRemoved : ℕ := 19
def targetMeanRemaining : ℕ := 6

theorem subsets_with_mean_six :
  (originalSet.filter (λ s, s.card = 3 ∧ s.sum = targetSumRemoved)).card = 4 := sorry

end subsets_with_mean_six_l774_774209


namespace digit_sum_condition_l774_774425

theorem digit_sum_condition (d : ℕ) (h_d_range : d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) : 
  (156 + 12 * d) % 9 = 0 ↔ d = 5 ∨ d = 8 :=
begin
  sorry
end

end digit_sum_condition_l774_774425


namespace midpoint_locus_circle_l774_774010

open EuclideanGeometry

/- Definitions -/
variables (A P Q O O1 O2: Point)
variables (secant: Line)
variables (intersects1: intersects secant (Circle O1 (distance O1 A)))
variables (intersects2: intersects secant (Circle O2 (distance O2 A)))
variables (rotation: rotates secant around A)

/- Proof statement -/
theorem midpoint_locus_circle :
    locus (midpoint P Q) (rotation secant around A) = Circle A (2 * distance O A) :=
sorry

end midpoint_locus_circle_l774_774010


namespace donovan_lap_time_correct_l774_774987

-- Definitions of the conditions
def donovan_lap_time (D : ℝ) : Prop :=
  let michael_lap_time := 36
  let laps_to_pass := 5.000000000000002
  (D / michael_lap_time) = laps_to_pass

-- Statement of what we need to prove
theorem donovan_lap_time_correct : ∃ D : ℝ, donovan_lap_time D ∧ D = 180 :=
by 
  let D := 180.00000000000007
  have h : donovan_lap_time D := 
    by
      unfold donovan_lap_time
      assume michael_lap_time := 36
      assume laps_to_pass := 5.000000000000002
      calc
        (D / michael_lap_time) = laps_to_pass : by sorry
  exact ⟨D, h, rfl⟩

end donovan_lap_time_correct_l774_774987


namespace perfect_square_factors_450_l774_774606

theorem perfect_square_factors_450 :
  ∃ (n : ℕ), n = 4 ∧ ∀ (d : ℕ), d ∣ 450 → ∃ (k : ℕ), k^2 = d ↔ d = 1 ∨ d = 9 ∨ d = 25 ∨ d = 225 :=
by
  sorry

end perfect_square_factors_450_l774_774606


namespace student_teams_partitionable_l774_774890

theorem student_teams_partitionable 
  (total_students : ℕ) 
  (schools : ℕ) 
  (students_per_school : fin s → ℕ)
  (total_students_eq : total_students = 300)
  (total_schools_geq_four : schools ≥ 4) 
  : ∃ (teams : list (fin total_students → ℕ)), 
    ∀ team ∈ teams, (∃ s, ∀ i ∈ team, students_per_school i = s) ∨ (∀ i ∈ team, students_per_school i ≠ students_per_school (list.head team)) :=
by
  sorry

end student_teams_partitionable_l774_774890


namespace C_plus_D_l774_774844

theorem C_plus_D (D C : ℚ) (h : ∀ x : ℚ, x ≠ 3 ∧ x ≠ 5 → (D * x - 17) / ((x - 3) * (x - 5)) = C / (x - 3) + 2 / (x - 5)) :
  C + D = 32 / 5 :=
by
  sorry

end C_plus_D_l774_774844


namespace monomial_coefficient_degree_l774_774695

def monomial : ℝ → ℕ → ℕ → ℕ → ℤ × ℕ := λ coeff a_exp b_exp c_exp, (coeff, a_exp + b_exp + c_exp)

theorem monomial_coefficient_degree :
  monomial (-2) 1 2 4 = (-2, 7) :=
by
  sorry

end monomial_coefficient_degree_l774_774695


namespace comparison_of_a_b_c_l774_774543

def a : ℝ := Real.log 2 / Real.log 3
def b : ℝ := 0.3 ^ 0.5
def c : ℝ := 0.5 ^ (-0.2)

theorem comparison_of_a_b_c : a < b ∧ b < c := by
  sorry

end comparison_of_a_b_c_l774_774543


namespace probability_inequality_l774_774288

open Complex

noncomputable def roots_of_unity (n : ℕ) : list ℂ := 
  (list.range n).map (λ k, Complex.exp (2 * Real.pi * Complex.I * (k : ℂ) / n))

theorem probability_inequality (v w : ℂ) 
  (h_v : v ∈ roots_of_unity 1997) 
  (h_w : w ∈ roots_of_unity 1997) 
  (h_vw : v ≠ w) :
  (finset.filter (λ (w : ℂ), (sqrt (2 + sqrt 3) ≤ Complex.abs (v + w))) 
    (finset.univ.filter (λ w, w ∈ roots_of_unity 1997 ∧ w ≠ v))).card.to_rat / 
    ((roots_of_unity 1997).length - 1) = 83 / 499 :=
sorry

end probability_inequality_l774_774288


namespace swans_in_10_years_l774_774135

def doubling_time := 2
def initial_swans := 15
def periods := 10 / doubling_time

theorem swans_in_10_years : 
  (initial_swans * 2 ^ periods) = 480 := 
by
  sorry

end swans_in_10_years_l774_774135


namespace handshaking_problem_l774_774244

/-- In a group of ten people, each person shakes hands with exactly three other people 
    from the group. Let M be the number of ways this handshaking can occur, considering 
    two handshaking arrangements different if at least two people who shake hands under 
    one arrangement do not shake hands under the other arrangement. Determine the remainder 
    when M is divided by 1000. -/
theorem handshaking_problem :
  let num_people := 10
  let handshakes_per_person := 3
  let M := number_of_handshakes num_people handshakes_per_person
  M % 1000 = 288 :=
sorry

end handshaking_problem_l774_774244


namespace cougar_sleep_hours_l774_774450

-- Definitions
def total_sleep_hours (C Z : Nat) : Prop :=
  C + Z = 70

def zebra_cougar_difference (C Z : Nat) : Prop :=
  Z = C + 2

-- Theorem statement
theorem cougar_sleep_hours :
  ∃ C : Nat, ∃ Z : Nat, zebra_cougar_difference C Z ∧ total_sleep_hours C Z ∧ C = 34 :=
sorry

end cougar_sleep_hours_l774_774450


namespace find_a_l774_774219

variable (a : ℤ)
variable (x y : ℤ)

theorem find_a (h1 : (a - 6) * x - y ^ (a - 6) = 1)
               (h2 : a - 6 ≠ 0)
               (h3 : a - 6 = 1) :
  a = 7 := 
sorry

end find_a_l774_774219


namespace polar_to_cartesian_l774_774814

theorem polar_to_cartesian (ρ θ : ℝ) : (ρ * Real.cos θ = 0) → ρ = 0 ∨ θ = π/2 :=
by 
  sorry

end polar_to_cartesian_l774_774814


namespace concyclicity_of_C_X_Y_A_l774_774808

-- Definitions and conditions
variables {A B C D X Y A' : Point}
variables (AB BC CD DA BD : Line)

-- Conditions
axiom parallelogram_ABCD : Parallelogram A B C D
axiom angle_bisector_A_intersects_BC_at_X : AngleBisectorIntersects A B C X
axiom angle_bisector_A_intersects_CD_at_Y : AngleBisectorIntersects A D C Y
axiom A_prime_symmetric_A_wrt_BD : SymmetricPoint A BD A'

-- Theorem statement
theorem concyclicity_of_C_X_Y_A' :
  ConcyclicPoints C X Y A' :=
sorry

end concyclicity_of_C_X_Y_A_l774_774808


namespace find_numbers_l774_774815

theorem find_numbers (A B C : ℝ) 
  (h1 : A - B = 1860) 
  (h2 : 0.075 * A = 0.125 * B) 
  (h3 : 0.15 * B = 0.05 * C) : 
  A = 4650 ∧ B = 2790 ∧ C = 8370 := 
by
  sorry

end find_numbers_l774_774815


namespace sqrt_thirteen_l774_774211

theorem sqrt_thirteen (x : ℂ) (hx : 11 = x^6 + (1 / x)^6) : x^3 + (1 / x^3) = Complex.sqrt 13 :=
by
  sorry

end sqrt_thirteen_l774_774211


namespace shortest_distance_around_circle_l774_774254

-- Definitions
def point (x y : ℝ) : Prop := true

def circle (center : point) (radius : ℝ) : Prop := 
  ∃ x y : ℝ, (x - 6)^2 + (y - 8)^2 = radius^2

-- Points definitions
def p1 := point 0 0
def p2 := point 12 16
def c := circle (point 6 8) 5

-- Theorem
theorem shortest_distance_around_circle (p1 p2 : point) (c : circle) :
  ∃ d : ℝ, d = 10 * Real.sqrt 3 + 5 * Real.pi / 3 := sorry

end shortest_distance_around_circle_l774_774254


namespace equalize_money_l774_774098

theorem equalize_money (
  Carmela_money : ℕ, Cousin_money : ℕ, num_cousins : ℕ) :
  Carmela_money = 7 → Cousin_money = 2 → num_cousins = 4 →
  ∀ (x : ℕ), Carmela_money - num_cousins * x = Cousin_money + x ∧
  (Carmela_money - num_cousins * x = Cousin_money + x) →
  x = 1 :=
by
  intros hCarmela_money hCousin_money hnum_cousins hx hfinal_eq
  sorry

end equalize_money_l774_774098


namespace train_distance_proof_l774_774785

theorem train_distance_proof (c₁ c₂ c₃ : ℝ) : 
  (5 / c₁ + 5 / c₂ = 15) →
  (5 / c₂ + 5 / c₃ = 11) →
  ∀ (x : ℝ), (x / c₁ = 10 / c₂ + (10 + x) / c₃) →
  x = 27.5 := 
by
  sorry

end train_distance_proof_l774_774785


namespace mask_pack_duration_l774_774953

theorem mask_pack_duration:
  ∀ (masks : ℕ) (family_members : ℕ) (change_frequency : ℕ),
    masks = 100 →
    family_members = (1 + 2 + 2) →
    change_frequency = 4 →
    (masks / family_members) * change_frequency = 80 := 
by
  intros masks family_members change_frequency h_masks h_family_members h_change_frequency
  rw [h_masks, h_family_members, h_change_frequency]
  norm_num
  sorry

end mask_pack_duration_l774_774953


namespace semicircle_proof_l774_774464

noncomputable def floor_of_inscribed_circle_radius (R : ℝ) : ℕ :=
  let r := R * (3.sqrt - 1) / (2 + 3.sqrt)
  in r.to_nat

theorem semicircle_proof :
  floor_of_inscribed_circle_radius 2021 = 673 :=
begin
  sorry
end

end semicircle_proof_l774_774464


namespace largest_five_digit_divisible_by_3_and_4_l774_774399

theorem largest_five_digit_divisible_by_3_and_4 : ∃ n : ℕ, (10000 ≤ n ∧ n ≤ 99999) ∧ (n % 3 = 0) ∧ (n % 4 = 0) ∧ ∀ m : ℕ, (10000 ≤ m ∧ m ≤ 99999) ∧ (m % 3 = 0) ∧ (m % 4 = 0) → m ≤ n :=
  ∃ (n : ℕ), (10000 ≤ n ∧ n ≤ 99999) ∧ (n % 3 = 0) ∧ (n % 4 = 0) ∧ ∀ m : ℕ, (10000 ≤ m ∧ m ≤ 99999) ∧ (m % 3 = 0) ∧ (m % 4 = 0) → m ≤ n := 99996 :=
sorry

end largest_five_digit_divisible_by_3_and_4_l774_774399


namespace arithmetic_sequence_common_difference_l774_774563

theorem arithmetic_sequence_common_difference 
  (a : Nat → Int)
  (a1 : a 1 = 5)
  (a6_a8_sum : a 6 + a 8 = 58) :
  ∃ d, ∀ n, a n = 5 + (n - 1) * d ∧ d = 4 := 
by 
  sorry

end arithmetic_sequence_common_difference_l774_774563


namespace export_company_plans_l774_774510

theorem export_company_plans :
  let cities := {1, 2, 3, 4, 5}
  in let plans := 
      (Finset.card (Finset.powerset_len 4 cities)) + 
      (Finset.card (Finset.powerset_len 3 cities) * 
       (Finset.card (Finset.powerset_len 2 cities))) + 
      (Finset.card (Finset.powerset_len 2 cities))
  in plans = 45 :=
by
  let cities := {1, 2, 3, 4, 5}
  let plans := 
    (Finset.card (Finset.powerset_len 4 cities)) + 
    (Finset.card (Finset.powerset_len 3 cities) * 
     (Finset.card (Finset.powerset_len 2 cities))) + 
    (Finset.card (Finset.powerset_len 2 cities))
  have h1 : Finset.card (Finset.powerset_len 4 cities) = 5 := sorry
  have h2 : Finset.card (Finset.powerset_len 3 cities) = 5 := sorry
  have h3 : Finset.card (Finset.powerset_len 2 cities) = 10 := sorry
  have h4 : 5 * (5 * 6) = 30 := sorry
  have h5 : 5 + 30 + 10 = 45 := sorry
  exact h5

end export_company_plans_l774_774510


namespace part1_monotonicity_part2_range_of_a_l774_774195

noncomputable def f (x a : ℝ) := (x^2 + 2 * x + a) / x

theorem part1_monotonicity : ∀ x ∈ Icc (1:ℝ) (⊤), ∀ a, (a = 1/2) → (∃ f' f, (f = λ x, f x (1/2)) ∧ (f' = deriv f) ∧ ∀ x ∈ Icc (1:ℝ) (⊤), f' x > 0) := 
sorry

theorem part2_range_of_a : ∀ x ∈ Icc (1:ℝ) (⊤), (f x a > 0) → (a > -3) := 
sorry

end part1_monotonicity_part2_range_of_a_l774_774195


namespace number_of_testing_methods_l774_774845

theorem number_of_testing_methods (plans_stage1 plans_stage2 : ℕ) 
  (h1 : plans_stage1 = 3) 
  (h2 : plans_stage2 = 5) : 
  plans_stage1 * plans_stage2 = 15 :=
by
  rw [h1, h2]
  rfl

end number_of_testing_methods_l774_774845


namespace find_int_solutions_l774_774994

theorem find_int_solutions (x : ℤ) :
  (∃ p : ℤ, Prime p ∧ 2*x^2 - x - 36 = p^2) ↔ (x = 5 ∨ x = 13) := 
sorry

end find_int_solutions_l774_774994


namespace add_alcohol_solve_l774_774050

variable (x : ℝ)

def initial_solution_volume : ℝ := 6
def initial_alcohol_fraction : ℝ := 0.20
def desired_alcohol_fraction : ℝ := 0.50

def initial_alcohol_content : ℝ := initial_alcohol_fraction * initial_solution_volume
def total_solution_volume_after_addition : ℝ := initial_solution_volume + x
def total_alcohol_content_after_addition : ℝ := initial_alcohol_content + x

theorem add_alcohol_solve (x : ℝ) :
  (initial_alcohol_content + x) / (initial_solution_volume + x) = desired_alcohol_fraction →
  x = 3.6 :=
by
  sorry

end add_alcohol_solve_l774_774050


namespace length_segment_AB_slope_half_line_equation_midpoint_P_l774_774580

section ellipse_segment

variable (x y : ℝ)

-- Given conditions
def ellipse := (x ^ 2) / 36 + (y ^ 2) / 9 = 1
def pointP := (4, 2 : ℝ)

-- 1. Length of segment AB when the slope is 1/2
theorem length_segment_AB_slope_half :
  ∀ (x1 y1 x2 y2 : ℝ),
  (y1 - 2) / (x1 - 4) = 1 / 2 →
  ellipse x1 y1 →
  ellipse x2 y2 →
  ((x1 - x2) ^ 2 + (y1 - y2) ^ 2).sqrt = 3 * (10).sqrt :=
sorry

-- 2. Equation of line when point P is midpoint of AB
theorem line_equation_midpoint_P :
  ∀ (x1 y1 x2 y2 : ℝ),
  let A := (x1, y1),
  let B := (x2, y2),
  (x1 + x2) / 2 = 4 ∧ (y1 + y2) / 2 = 2 →
  ellipse x1 y1 →
  ellipse x2 y2 →
  ∀ (x y : ℝ),
  (y - 2 = -(1 / 2) * (x - 4)) →
  x + 2 * y - 8 = 0 :=
sorry

end ellipse_segment

end length_segment_AB_slope_half_line_equation_midpoint_P_l774_774580


namespace circle_division_l774_774109

-- Define the circle and its properties
structure Circle where
  center : Point
  radius : ℝ

-- Define the Point type
structure Point where
  x : ℝ
  y : ℝ

-- Define the existence of a division into 12 parts
theorem circle_division (C : Circle) (O : C.center) :
  ∃ (divs : ℕ), divs = 12 ∧ ∀ (part : ℕ), part < divs → ¬(C.center ∈ boundary part) := 
sorry

end circle_division_l774_774109


namespace mass_of_ln_x_arc_l774_774144

noncomputable def mass_of_arc (k : ℝ) : ℝ :=
  k / 3 * (10 * real.sqrt 10 - 2 * real.sqrt 2)

theorem mass_of_ln_x_arc : 
  ∀ (k : ℝ), mass_of_arc k = k / 3 * (10 * real.sqrt 10 - 2 * real.sqrt 2) := 
by
  intros k
  sorry

end mass_of_ln_x_arc_l774_774144


namespace max_notebooks_no_more_than_11_l774_774333

noncomputable def maxNotebooks (money : ℕ) (cost_single : ℕ) (cost_pack4 : ℕ) (cost_pack7 : ℕ) (max_pack7 : ℕ) : ℕ :=
if money >= cost_pack7 then
  if (money - cost_pack7) >= cost_pack4 then 7 + 4
  else if (money - cost_pack7) >= cost_single then 7 + 1
  else 7
else if money >= cost_pack4 then
  if (money - cost_pack4) >= cost_pack4 then 4 + 4
  else if (money - cost_pack4) >= cost_single then 4 + 1
  else 4
else
  money / cost_single

theorem max_notebooks_no_more_than_11 :
  maxNotebooks 15 2 6 9 1 = 11 :=
by
  sorry

end max_notebooks_no_more_than_11_l774_774333


namespace trihedral_angle_bisector_tetrahedron_bisector_intersection_l774_774423

-- Definitions for the problem:
def bisector_planes_intersection (P1 P2 P3 : Plane) : Line := sorry

def tetrahedron_intersection (T : Tetrahedron) : Point := sorry
 
-- Lean statements for the theorems:

-- Part (a): Prove that the three bisector planes of the dihedral angles of a trihedral angle intersect along one line.
theorem trihedral_angle_bisector (P1 P2 P3 : Plane) (A : Point)
  (h1 : A ∈ P1) (h2 : A ∈ P2) (h3 : A ∈ P3) : 
  ∃ L : Line, ∀ K : Point,
  K ∈ bisector_planes_intersection P1 P2 P3 ↔ 
  distance K P1 = distance K P2 ∧ 
  distance K P2 = distance K P3 := 
sorry

-- Part (b): Prove that the four bisectors of the trihedral angles of a tetrahedron intersect at one point, which is the center of the inscribed sphere.
theorem tetrahedron_bisector_intersection (T : Tetrahedron) : 
  ∃ O : Point, ∀ P : Plane, 
  (P ∈ faces T) → distance O P = distance_from_center_to_face T :=
sorry

end trihedral_angle_bisector_tetrahedron_bisector_intersection_l774_774423


namespace tangent_line_at_one_neg_two_l774_774183

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then (Real.log (-x) + 2 * x)
  else (Real.log x + 2 * x)

theorem tangent_line_at_one_neg_two :
  (∀ x : ℝ, f (-x) = f x) ∧
  (f 1 = -2) →
  ∃ (m b : ℝ), (∀ x : ℝ, (y - (-2) = f' 1 * (x - 1)) → y = m * x + b) ∧ (m = 3) ∧ (b = -5) :=
begin
  sorry
end

end tangent_line_at_one_neg_two_l774_774183


namespace even_digit_three_digit_number_count_l774_774601

theorem even_digit_three_digit_number_count : 
  (∀ n : ℕ, 100 ≤ n ∧ n ≤ 999 → 
  (∀ d ∈ [n / 100, (n % 100) / 10, n % 10], d ∈ {2, 4, 6, 8}) →
  ∃ count : ℕ, count = 64) :=
by
  sorry

end even_digit_three_digit_number_count_l774_774601


namespace arrangement_count_l774_774956

def students := Fin 5
def classes := Fin 4

theorem arrangement_count : 
  (∃ (f : students → classes), 
    (∀ c : classes, ∃ s : students, f s = c) ∧ 
    (∀ s : students, f s ≠ f 1 → s ≠ 1) ∧ 
    (f 1 ≠ 0)) 
  = 180 := 
sorry

end arrangement_count_l774_774956


namespace perfect_square_factors_450_l774_774613

theorem perfect_square_factors_450 :
  ∃ (n : ℕ), n = 4 ∧ ∀ (d : ℕ), d ∣ 450 → ∃ (k : ℕ), k^2 = d ↔ d = 1 ∨ d = 9 ∨ d = 25 ∨ d = 225 :=
by
  sorry

end perfect_square_factors_450_l774_774613


namespace inequality_solution_l774_774148

theorem inequality_solution (x : ℝ) : 2 ≥ 1 / (x - 1) ↔ x ∈ set.interval_oc (-∞ : ℝ) 1 ∪ set.interval_co (3/2 : ℝ) (+∞ : ℝ) :=
sorry

end inequality_solution_l774_774148


namespace number_of_perfect_square_divisors_of_450_l774_774635

theorem number_of_perfect_square_divisors_of_450 : 
    let p := 450;
    let factors := [(3, 2), (5, 2), (2, 1)];
    ∃ n, (n = 4 ∧ 
          ∀ (d : ℕ), d ∣ p → 
                     (∃ (a b c : ℕ), d = 2^a * 3^b * 5^c ∧ 
                              (a = 0) ∧ (b = 0 ∨ b = 2) ∧ (c = 0 ∨ c = 2) → 
                              a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 = 0)) :=
    sorry

end number_of_perfect_square_divisors_of_450_l774_774635


namespace domain_log_function_domain_l774_774347

theorem domain_log_function :
  (∀ x : ℝ, 5 - x > 0 ∧ x - 2 > 0 ∧ x - 2 ≠ 1 → 2 < x ∧ x < 5 ∧ x ≠ 3) :=
by
  intro x
  intro h
  have h₁ : 5 - x > 0 := h.left.left
  have h₂ : x - 2 > 0 := h.left.right
  have h₃ : x - 2 ≠ 1 := h.right
  exact sorry

theorem domain :
  setOf (λ x, 5 - x > 0 ∧ x - 2 > 0 ∧ x - 2 ≠ 1) = (set.Ioo 2 3) ∪ (set.Ioc 3 5) :=
by
  ext x
  split
  {
    intro h
    split
    {
      rw set.mem_union
      rw set.mem_Ioo at *
      cases (lt_trichotomy x 3) with hx hx
      {
        left,
        exact ⟨h.left.right, hx.left⟩
      }
      {
        cases hx with heq hx2
        {
          exfalso,
          exact h.right heq
        }
        {
          right,
          exact ⟨hx2.left, h.left.left⟩
        }
      }
    },
    {
      intro h
      cases h
      {
        split
        {
          exact h.left,
          split
          {
            exact h.right,
            intro heq,
            rw heq at h.left,                      
            exact ⟨h.left, h.left.trans h.right⟩
          }
        },
        split
        {
          exact h.left.left,
          split
          {
            exact h.left.right,
            exact λ heq, heq.not_gt h.left.left
          }
        }
      }
    }
  }

end domain_log_function_domain_l774_774347


namespace initial_games_l774_774748

theorem initial_games (X : ℕ) (h1 : X + 31 - 105 = 6) : X = 80 :=
by
  sorry

end initial_games_l774_774748


namespace csc_squared_product_product_m_n_sum_equals_181_l774_774035

theorem csc_squared_product :
  (∏ k in Finset.range 90, (Real.csc ((2 * k + 1) * Real.pi / 180)) ^ 2) = 2 ^ 179 := 
sorry

theorem product_m_n_sum_equals_181 :
  let m := 2
  let n := 179
  m + n = 181 :=
begin
  -- This proof only needs to confirm the final result involving m and n
  let m := 2,
  let n := 179,
  show m + n = 181,
  simp,
end

end csc_squared_product_product_m_n_sum_equals_181_l774_774035


namespace length_of_DQ_l774_774240

variables (M C D E Q : Point) (circle : Circle)
variables (y : ℝ) (CQ_length : ℝ)
variables (arc_CDE_midpoint : IsArcMidpoint M C D E)
variables (MQ_perpendicular_DE : PerpendicularSegment MQ DE Q)
variables (CD_length_EQ_y : Distance C D = y)
variables (CQ_length_EQ_y_plus_3 : Distance C Q = y + 3)

theorem length_of_DQ : Distance D Q = y + 3 :=
by 
  -- The actual proof will go here
  sorry

end length_of_DQ_l774_774240


namespace julieta_total_spent_l774_774272

def original_price_backpack : ℕ := 50
def original_price_ring_binder : ℕ := 20
def quantity_ring_binders : ℕ := 3
def price_increase_backpack : ℕ := 5
def price_decrease_ring_binder : ℕ := 2

def total_spent (original_price_backpack original_price_ring_binder quantity_ring_binders price_increase_backpack price_decrease_ring_binder : ℕ) : ℕ :=
  let new_price_backpack := original_price_backpack + price_increase_backpack
  let new_price_ring_binder := original_price_ring_binder - price_decrease_ring_binder
  new_price_backpack + (new_price_ring_binder * quantity_ring_binders)

theorem julieta_total_spent :
  total_spent original_price_backpack original_price_ring_binder quantity_ring_binders price_increase_backpack price_decrease_ring_binder = 109 :=
by 
  -- Proof steps are omitted intentionally
  sorry

end julieta_total_spent_l774_774272


namespace angle_BC_CD_l774_774341

theorem angle_BC_CD (α β : ℝ) (two_equal_rt_triangles : Prop)
  (common_hypotenuse : Prop) (angle_between_planes : ∠ = α)
  (angle_between_legs : ∠ = β) : 
  ∃ θ, θ = 2 * arcsin (sqrt (sin((α + β) / 2) * sin((α - β) / 2))) 
     ∧ ∠(BC, CD) = θ := 
sorry

end angle_BC_CD_l774_774341


namespace perfect_square_factors_450_l774_774608

theorem perfect_square_factors_450 :
  ∃ (n : ℕ), n = 4 ∧ ∀ (d : ℕ), d ∣ 450 → ∃ (k : ℕ), k^2 = d ↔ d = 1 ∨ d = 9 ∨ d = 25 ∨ d = 225 :=
by
  sorry

end perfect_square_factors_450_l774_774608


namespace slope_angle_45_degrees_l774_774436

-- Define the line as a linear equation
def line_eq (x y : ℝ) : Prop := x - y + 8 = 0

-- Define the slope angle α such that 0° ≤ α < 180°
-- and the tangent of the slope angle equals the slope of the line
def slope_angle (α : ℝ) : Prop :=
  ∀ x y : ℝ, line_eq x y → tan α = 1

-- Prove that the slope angle α is 45°
theorem slope_angle_45_degrees :
  ∃ α : ℝ, slope_angle α ∧ α = 45 :=
by
  sorry

end slope_angle_45_degrees_l774_774436


namespace numPerfectSquareFactorsOf450_l774_774681

def isPerfectSquare (n : Nat) : Prop :=
  ∃ k : Nat, n = k * k

theorem numPerfectSquareFactorsOf450 : 
  ∃! n : Nat, 
    (∀ d : Nat, d ∣ 450 → isPerfectSquare d) → n = 4 := 
by
  sorry

end numPerfectSquareFactorsOf450_l774_774681


namespace analytic_expression_monotonic_intervals_decreasing_monotonic_intervals_increasing_range_on_interval_l774_774170

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + (Real.pi / 6))

theorem analytic_expression :
  (∀ x : ℝ, 0 < 2 ∧ 0 < 2 ∧ 0 < (Real.pi / 6) ∧ (Real.pi / 6) < Real.pi / 2 ∧
            f(x) = 2 * Real.sin (2 * x + (Real.pi / 6))) :=
sorry

theorem monotonic_intervals_decreasing (k : ℤ) :
  ∀ x : ℝ, k * Real.pi + Real.pi / 6 ≤ x ∧ x ≤ k * Real.pi + (2 * Real.pi) / 3 → 
  ∀ y : ℝ, (k * Real.pi + Real.pi / 6 ≤ y) ∧ (y ≤ x) → f(y) ≥ f(x) :=
sorry

theorem monotonic_intervals_increasing (k : ℤ) :
  ∀ x : ℝ, k * Real.pi - Real.pi / 3 ≤ x ∧ x ≤ k * Real.pi + Real.pi / 6 → 
  ∀ y : ℝ, (k * Real.pi - Real.pi / 3 ≤ y) ∧ (y ≤ x) → f(y) ≤ f(x) :=
sorry

theorem range_on_interval :
  (∀ x : ℝ, (0 ≤ x ∧ x ≤ Real.pi / 12) → 1 ≤ f(x) ∧ f(x) ≤ Real.sqrt 3) :=
sorry

end analytic_expression_monotonic_intervals_decreasing_monotonic_intervals_increasing_range_on_interval_l774_774170


namespace remainder_of_6_power_700_mod_72_l774_774863

theorem remainder_of_6_power_700_mod_72 : (6^700) % 72 = 0 :=
by
  sorry

end remainder_of_6_power_700_mod_72_l774_774863


namespace sequence_term_l774_774579

theorem sequence_term (S : ℕ → ℕ) (a : ℕ → ℕ) : 
  (∀ n, S n = n^2 + 3n) → 
  (S 1 = a 1) × (∀ n ≥ 2, a n = S n - S (n - 1)) →
  (∀ n, a n = 2n + 2) := 
by
  sorry

end sequence_term_l774_774579


namespace sin_cos_problem_l774_774761

noncomputable def sin_cos_relation (x y : ℝ) : ℝ :=
  let hx := (Real.sin x / Real.sin y = 3)
  let hy := (Real.cos x / Real.cos y = 1 / 2)
  have h1 : Real.sin 2 * x / Real.sin 2 * y = 3 / 2 := sorry
  have h2 : Real.cos 2 * x / Real.cos 2 * y = -19 / 29 := sorry
  h1 + h2

theorem sin_cos_problem (x y : ℝ)
    (hx : Real.sin x / Real.sin y = 3) 
    (hy : Real.cos x / Real.cos y = 1 / 2) : 
    Real.sin_cos_relation x y = 49 / 58 :=
sorry

end sin_cos_problem_l774_774761


namespace find_p_l774_774028

def vec1 : ℝ × ℝ := (-3, 2)
def vec2 : ℝ × ℝ := (4, 5)
def r (t : ℝ) : ℝ × ℝ := (7 * t - 3, 3 * t + 2)
def p : ℝ × ℝ := (-69 / 58, 161 / 58)
def direction : ℝ × ℝ := (7, 3)

theorem find_p : ∃ t : ℝ, (r t).fst * direction.fst + (r t).snd * direction.snd = 0 ∧ 
                            r t = p := by
  sorry

end find_p_l774_774028


namespace num_friends_l774_774309

-- Define the friends
def Mary : Prop := ∃ n : ℕ, n = 6
def Sam : Prop := ∃ n : ℕ, n = 6
def Keith : Prop := ∃ n : ℕ, n = 6
def Alyssa : Prop := ∃ n : ℕ, n = 6

-- Define the set of friends
def friends : set Prop := {Mary, Sam, Keith, Alyssa}

-- Statement to prove
theorem num_friends (h1 : Mary) (h2 : Sam) (h3 : Keith) (h4 : Alyssa) : 
  set.card friends = 4 :=
by sorry

end num_friends_l774_774309


namespace isosceles_triangle_of_lines_l774_774105

def intersection (l1 l2 : (ℝ × ℝ → Prop)) : Prop :=
  ∃ x y, l1 (x, y) ∧ l2 (x, y)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

noncomputable def is_isosceles (p1 p2 p3 : ℝ × ℝ) : Prop :=
  distance p1 p2 = distance p1 p3 ∨ 
  distance p1 p2 = distance p2 p3 ∨ 
  distance p1 p3 = distance p2 p3

theorem isosceles_triangle_of_lines : 
  let l1 := λ p : ℝ × ℝ, p.2 = 2 * p.1 + 3
  let l2 := λ p : ℝ × ℝ, p.2 = -2 * p.1 + 3
  let l3 := λ p : ℝ × ℝ, p.2 = -3
  have h1 : intersection l1 l2,
  have h2 : intersection l1 l3,
  have h3 : intersection l2 l3,
  ∃ (p1 p2 p3 : ℝ × ℝ), l1 p1 ∧ l2 p2 ∧ l3 p3 ∧ 
    is_isosceles p1 p2 p3 :=
sorry

end isosceles_triangle_of_lines_l774_774105


namespace smallest_three_digit_multiple_of_6_5_8_9_eq_360_l774_774023

theorem smallest_three_digit_multiple_of_6_5_8_9_eq_360 :
  ∃ n : ℕ, n ≥ 100 ∧ n ≤ 999 ∧ (n % 6 = 0 ∧ n % 5 = 0 ∧ n % 8 = 0 ∧ n % 9 = 0) ∧ n = 360 := 
by
  sorry

end smallest_three_digit_multiple_of_6_5_8_9_eq_360_l774_774023


namespace sprint_time_l774_774306

def speed (Mark : Type) : ℝ := 6.0
def distance (Mark : Type) : ℝ := 144.0

theorem sprint_time (Mark : Type) : (distance Mark) / (speed Mark) = 24 := by
  sorry

end sprint_time_l774_774306


namespace g_of_square_sub_one_l774_774110

variable {R : Type*} [LinearOrderedField R]

def g (x : R) : R := 3

theorem g_of_square_sub_one (x : R) : g ((x - 1)^2) = 3 := 
by sorry

end g_of_square_sub_one_l774_774110


namespace carmela_gives_each_l774_774095

noncomputable def money_needed_to_give_each (carmela : ℕ) (cousins : ℕ) (cousins_count : ℕ) : ℕ :=
  let total_cousins_money := cousins * cousins_count
  let total_money := carmela + total_cousins_money
  let people_count := 1 + cousins_count
  let equal_share := total_money / people_count
  let total_giveaway := carmela - equal_share
  total_giveaway / cousins_count

theorem carmela_gives_each (carmela : ℕ) (cousins : ℕ) (cousins_count : ℕ) (h_carmela : carmela = 7) (h_cousins : cousins = 2) (h_cousins_count : cousins_count = 4) :
  money_needed_to_give_each carmela cousins cousins_count = 1 :=
by
  rw [h_carmela, h_cousins, h_cousins_count]
  sorry

end carmela_gives_each_l774_774095


namespace pine_saplings_in_sample_l774_774900

theorem pine_saplings_in_sample 
    (total_saplings : ℕ)
    (pine_saplings : ℕ)
    (sample_size : ℕ)
    (h1 : total_saplings = 30000)
    (h2 : pine_saplings = 4000)
    (h3 : sample_size = 150) :
    pine_saplings * sample_size / total_saplings = 20 :=
by {
  have proportion : ℚ := pine_saplings / total_saplings,
  calc  (pine_saplings * sample_size) / total_saplings
      = (4000 * 150) / 30000 : by simp [h1, h2, h3]
  ... = (2 * 150) / 15 : by norm_num
  ... = 20 : by norm_num,
}

end pine_saplings_in_sample_l774_774900


namespace region_to_the_upper_left_of_line_l774_774351

variable (x y : ℝ)

def line_eqn := 3 * x - 2 * y - 6 = 0

def region := 3 * x - 2 * y - 6 < 0

theorem region_to_the_upper_left_of_line :
  ∃ rect_upper_left, (rect_upper_left = region) := 
sorry

end region_to_the_upper_left_of_line_l774_774351


namespace Delta_zero_iff_c_eq_sqrt30_l774_774301

def Delta (a b c : ℝ) : ℝ := c^2 - 3 * a * b

theorem Delta_zero_iff_c_eq_sqrt30 (a b c : ℝ) (h₁ : a = 2) (h₂ : b = 5) (h₃ : Delta a b c = 0) : 
  c = √30 ∨ c = -√30 :=
by
  sorry

end Delta_zero_iff_c_eq_sqrt30_l774_774301


namespace perfect_square_factors_450_l774_774654

def prime_factorization (n : ℕ) : Prop :=
  n = 2^1 * 3^2 * 5^2

theorem perfect_square_factors_450 : prime_factorization 450 → ∃ n : ℕ, n = 4 :=
by
  sorry

end perfect_square_factors_450_l774_774654


namespace nails_to_buy_l774_774008

-- Define the initial number of nails Tom has
def initial_nails : ℝ := 247

-- Define the number of nails found in the toolshed
def toolshed_nails : ℝ := 144

-- Define the number of nails found in a drawer
def drawer_nails : ℝ := 0.5

-- Define the number of nails given by the neighbor
def neighbor_nails : ℝ := 58.75

-- Define the total number of nails needed for the project
def total_needed_nails : ℝ := 625.25

-- Define the total number of nails Tom already has
def total_existing_nails : ℝ := 
  initial_nails + toolshed_nails + drawer_nails + neighbor_nails

-- Prove that Tom needs to buy 175 more nails
theorem nails_to_buy :
  total_needed_nails - total_existing_nails = 175 := by
  sorry

end nails_to_buy_l774_774008


namespace initial_deposit_l774_774313

-- Definitions based on the problem conditions
def amount_after_one_year : ℝ := 121.00000000000001
def annual_interest_rate : ℝ := 0.20
def compounding_frequency : ℕ := 2
def time_period : ℝ := 1

-- Compound interest formula definition
def compound_interest (P r : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
  P * (1 + r / n)^ (n * t)

-- The problem proof statement
theorem initial_deposit :
  ∃ (P : ℝ), compound_interest P annual_interest_rate compounding_frequency time_period = amount_after_one_year ∧ P = 100 :=
sorry

end initial_deposit_l774_774313


namespace tenth_term_arith_seq_l774_774494

variable (a1 d : Int) -- Initial term and common difference
variable (n : Nat) -- nth term

-- Definition of the nth term in an arithmetic sequence
def arithmeticSeq (a1 d : Int) (n : Nat) : Int :=
  a1 + (n - 1) * d

-- Specific values for the problem
def a_10 : Int :=
  arithmeticSeq 10 (-3) 10

-- The theorem we want to prove
theorem tenth_term_arith_seq : a_10 = -17 := by
  sorry

end tenth_term_arith_seq_l774_774494


namespace fraction_equality_l774_774791

theorem fraction_equality 
  (a b c d : ℝ)
  (h1 : a + c = 2 * b)
  (h2 : 2 * b * d = c * (b + d))
  (hb : b ≠ 0)
  (hd : d ≠ 0) :
  a / b = c / d :=
sorry

end fraction_equality_l774_774791


namespace sum_first_2500_terms_l774_774465

noncomputable def sum_seq (b : ℕ → ℤ) (n : ℕ) : ℤ :=
  if h : n > 0 then (Finset.range n).sum (λ k, b (k + 1)) else 0

theorem sum_first_2500_terms (b : ℕ → ℤ)
  (h1 : ∀ n ≥ 3, b n = b (n - 1) - b (n - 2))
  (h2 : sum_seq b 1000 = 1230)
  (h3 : sum_seq b 1230 = 1000) :
  sum_seq b 2500 = 1230 := sorry

end sum_first_2500_terms_l774_774465


namespace weight_loss_percentage_l774_774419

theorem weight_loss_percentage (W : ℝ) (hW : W > 0) : 
  let new_weight := 0.89 * W
  let final_weight_with_clothes := new_weight * 1.02
  (W - final_weight_with_clothes) / W * 100 = 9.22 := by
  sorry

end weight_loss_percentage_l774_774419


namespace rickshaw_distance_l774_774065

theorem rickshaw_distance :
  ∃ (distance : ℝ), 
  (13.5 + (distance - 1) * (2.50 / (1 / 3))) = 103.5 ∧ distance = 13 :=
by
  sorry

end rickshaw_distance_l774_774065


namespace count_diff_of_squares_l774_774207

theorem count_diff_of_squares (n : ℕ) (h : 1 ≤ n ∧ n ≤ 1000) : 
  (∃ x y : ℕ, n = x^2 - y^2) ↔ n ∈ (range 1001).filter (λ z, z % 2 = 1 ∨ z % 4 = 0) :=
by {
  sorry
}

end count_diff_of_squares_l774_774207


namespace two_people_meet_l774_774456

-- Define the labyrinth parameters and geometric progression
def labyrinth (n : ℕ) (r : ℝ) := (fin n → ℝ)

-- Define the properties of the circles
def circles_in_progression (n : ℕ) (r : ℝ) (lab : labyrinth n r) : Prop :=
  ∀ i : fin n, lab i = r * (2 ^ (i : ℕ))

-- Define the movement properties of the two individuals
def same_speed_opposite_directions (n : ℕ) : Prop :=
  ∀ i j : fin n, i ≠ j → 
    (travels (person₁) i → travels (person₂) j) ∧ 
    (travels (person₁) j → travels (person₂) i)

-- Prove that they will meet
theorem two_people_meet (n : ℕ) (r : ℝ) (lab : labyrinth n r)
  (hp : circles_in_progression n r lab)
  (hm : same_speed_opposite_directions n) :
  ∃ t : ℝ, travels (person₁) C_{n-1} t = travels (person₂) C_{n-1} t := 
sorry

end two_people_meet_l774_774456


namespace option_C_is_always_odd_l774_774032

def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

theorem option_C_is_always_odd (k : ℤ) : is_odd (2007 + 2 * k ^ 2) :=
sorry

end option_C_is_always_odd_l774_774032


namespace set_D_forms_triangle_l774_774872

theorem set_D_forms_triangle (a b c : ℕ) (h1 : a = 4) (h2 : b = 5) (h3 : c = 6) : a + b > c ∧ a + c > b ∧ b + c > a := by
  rw [h1, h2, h3]
  show 4 + 5 > 6 ∧ 4 + 6 > 5 ∧ 5 + 6 > 4
  sorry

end set_D_forms_triangle_l774_774872


namespace intersection_lies_on_kiepert_hyperbola_l774_774787

noncomputable def triangle (α β γ : ℝ) : Type :=
{A B C : Type} -- Type definition for our triangle vertices.

noncomputable def isosceles_triangle (A B C : Type) (φ : ℝ) : Prop :=
-- Definition for an isosceles triangle with a base angle φ.

def lie_on_kiepert_hyperbola (P : Type) (A B C : Type) : Prop :=
-- Definition for a point lying on the Kiepert hyperbola of triangle ABC.

theorem intersection_lies_on_kiepert_hyperbola
  (ABC : triangle α β γ)
  (φ : ℝ)
  (hφ1 : 0 < φ ∧ φ < π / 2 ∨ -π / 2 < φ ∧ φ < 0)
  (AC1B_isosceles : isosceles_triangle ABC.A ABC.B ABC.C φ)
  (BA1C_isosceles : isosceles_triangle ABC.B ABC.C ABC.A φ)
  (AB1C_isosceles : isosceles_triangle ABC.A ABC.C ABC.B φ) :
  ∃ (P : α), 
  (lie_on_kiepert_hyperbola P ABC.A ABC.B ABC.C) ∧
  (lies_on_line P ABC.A A1) ∧
  (lies_on_line P ABC.B B1) ∧
  (lies_on_line P ABC.C C1) := 
sorry

end intersection_lies_on_kiepert_hyperbola_l774_774787


namespace games_against_other_division_l774_774442

theorem games_against_other_division
  (N M : ℕ) (h1 : N > 2 * M) (h2 : M > 5)
  (total_games : N * 4 + 5 * M = 82) :
  5 * M = 30 :=
by
  sorry

end games_against_other_division_l774_774442


namespace functional_equation_solution_l774_774503

theorem functional_equation_solution :
  ∀ (f : ℝ → ℝ), (∀ x y : ℝ, f (x^2 - y^2) = x * f x - y * f y) →
  ∃ c : ℝ, ∀ x : ℝ, f x = c * x :=
by
  sorry

end functional_equation_solution_l774_774503


namespace angle_AKD_l774_774316

/-- Given a square ABCD with points M and K on sides BC and CD respectively,
    such that ∠BAM = ∠CKM = 30°, prove that ∠AKD = 75°. -/
theorem angle_AKD (A B C D M K : Point) (h_square : is_square A B C D)
  (h_M : M ∈ segment B C) (h_K : K ∈ segment C D)
  (h_angle_BAM : angle A B M = 30) (h_angle_CKM : angle C K M = 30) :
  angle A K D = 75 :=
sorry

end angle_AKD_l774_774316


namespace class_average_weight_l774_774839

def average_weight_of_class (A B C D : ℕ) (weightA weightB weightC weightD : ℝ) : ℝ :=
  let totalWeight := A * weightA + B * weightB + C * weightC + D * weightD
  let totalStudents := A + B + C + D
  totalWeight / totalStudents

theorem class_average_weight :
  average_weight_of_class 30 20 15 25 40.5 35.7 38.3 37.2 = 38.15 :=
by
  sorry

end class_average_weight_l774_774839


namespace perfect_square_factors_count_450_l774_774672

theorem perfect_square_factors_count_450 : 
  (∃ n : ℕ, n = 4 ∧ (∀ d : ℕ, d ∣ 450 → ∃ k : ℕ, d = k * k)) := sorry

end perfect_square_factors_count_450_l774_774672


namespace Dan_room_cleaning_time_l774_774268

theorem Dan_room_cleaning_time (Clara_time : ℕ) (Dan_ratio : ℚ) (h1 : Clara_time = 40) (h2 : Dan_ratio = 3/8) : 
  Clara_time * Dan_ratio = 15 := 
by {
  -- Leveraging given conditions
  rw [h1, h2],
  -- Calculating Dan's cleaning time based on the provided formulas
  norm_num,
  sorry
}

end Dan_room_cleaning_time_l774_774268


namespace adults_at_show_l774_774471

theorem adults_at_show (A C : ℕ) (h1 : A = 2 * C) (h2 : 5.50 * A + 2.50 * C = 1026) :
  A = 152 :=
by
  sorry

end adults_at_show_l774_774471


namespace min_value_of_A_ge_3_l774_774553

variable {x y z : ℝ} (hx : 0 < x ∧ x ≤ 1) (hy : 0 < y ∧ y ≤ 1) (hz : 0 < z ∧ z ≤ 1)

def A (x y z : ℝ) := (x + 2 * y) * Real.sqrt (x + y - x * y) + (y + 2 * z) * Real.sqrt (y + z - y * z) + (z + 2 * x) * Real.sqrt (z + x - z * x) / (x * y + y * z + z * x)

theorem min_value_of_A_ge_3 (hx : 0 < x ∧ x ≤ 1) (hy : 0 < y ∧ y ≤ 1) (hz : 0 < z ∧ z ≤ 1) :
  A x y z ≥ 3 := by
  sorry

end min_value_of_A_ge_3_l774_774553


namespace perfect_square_factors_count_450_l774_774668

theorem perfect_square_factors_count_450 : 
  (∃ n : ℕ, n = 4 ∧ (∀ d : ℕ, d ∣ 450 → ∃ k : ℕ, d = k * k)) := sorry

end perfect_square_factors_count_450_l774_774668


namespace quadratic_has_equal_roots_l774_774228

theorem quadratic_has_equal_roots (b : ℝ) (h : ∃ x : ℝ, b*x^2 + 2*b*x + 4 = 0 ∧ b*x^2 + 2*b*x + 4 = 0) :
  b = 4 :=
sorry

end quadratic_has_equal_roots_l774_774228


namespace convert_to_spherical_l774_774978

noncomputable def rectangular_to_spherical (x y z : ℝ) : ℝ × ℝ × ℝ :=
  let ρ := Real.sqrt (x^2 + y^2 + z^2)
  let φ := Real.arccos (z / ρ)
  let θ := if x = 0 ∧ y > 0 then Real.pi / 2
           else if x = 0 ∧ y < 0 then 3 * Real.pi / 2
           else if x > 0 then Real.arctan (y / x)
           else if y >= 0 then Real.arctan (y / x) + Real.pi
           else Real.arctan (y / x) - Real.pi
  (ρ, θ, φ)

theorem convert_to_spherical :
  rectangular_to_spherical (3 * Real.sqrt 2) (-4) 5 =
  (Real.sqrt 59, 2 * Real.pi + Real.arctan ((-4) / (3 * Real.sqrt 2)), Real.arccos (5 / Real.sqrt 59)) :=
by
  sorry

end convert_to_spherical_l774_774978


namespace Z_integer_if_le_infinitely_many_a_l774_774533

-- Define Z(a, b)
def Z (a b : ℕ) : ℚ := ((3 * a).factorial * (4 * b).factorial) / ((a.factorial ^ 4) * (b.factorial ^ 3))

-- Part (a): Prove that Z(a, b) is an integer if a ≤ b
theorem Z_integer_if_le (a b : ℕ) (h : a ≤ b) : Z a b ∈ ℤ := sorry

-- Part (b): Prove that for each b there are infinitely many a such that Z(a, b) is not an integer
theorem infinitely_many_a (b : ℕ) : ∃ᶠ a in at_top, ¬(Z a b ∈ ℤ) := sorry

end Z_integer_if_le_infinitely_many_a_l774_774533


namespace number_of_perfect_square_factors_l774_774667

def is_prime_factorization_valid : Bool :=
  let p1 := 2
  let p1_pow := 1
  let p2 := 3
  let p2_pow := 2
  let p3 := 5
  let p3_pow := 2
  (450 = (p1^p1_pow) * (p2^p2_pow) * (p3^p3_pow))
  
def is_perfect_square (n : Nat) : Bool :=
  match n with
  | 1 => true
  | _ => let second_digit := Math.sqrt n * Math.sqrt n;
         second_digit == n

theorem number_of_perfect_square_factors : (number_of_perfect_square_factors 450 = 4) :=
by sorry

end number_of_perfect_square_factors_l774_774667


namespace volume_after_increase_l774_774926

variable (l w h : ℝ)

def original_volume (l w h : ℝ) : ℝ := l * w * h
def original_surface_area (l w h : ℝ) : ℝ := 2 * (l * w + w * h + h * l)
def original_edge_sum (l w h : ℝ) : ℝ := 4 * (l + w + h)
def increased_volume (l w h : ℝ) : ℝ := (l + 2) * (w + 2) * (h + 2)

theorem volume_after_increase :
  original_volume l w h = 5000 →
  original_surface_area l w h = 1800 →
  original_edge_sum l w h = 240 →
  increased_volume l w h = 7048 := by
  sorry

end volume_after_increase_l774_774926


namespace palindromic_sum_l774_774441

def is_palindrome (n : ℕ) : Prop :=
  let s := toString n
  s = s.reverse

theorem palindromic_sum (x : ℕ) (h1 : 100 ≤ x ∧ x ≤ 999) (h2 : is_palindrome x) (h3 : is_palindrome (x + 45)) :
  ((x / 100) + ((x % 100) / 10) + (x % 10) = 20) :=
sorry

end palindromic_sum_l774_774441


namespace swans_in_10_years_l774_774134

def doubling_time := 2
def initial_swans := 15
def periods := 10 / doubling_time

theorem swans_in_10_years : 
  (initial_swans * 2 ^ periods) = 480 := 
by
  sorry

end swans_in_10_years_l774_774134


namespace radius_of_circle_l774_774337

/-- Given the equation of a circle x^2 + y^2 - 8 = 2x + 4y,
    we need to prove that the radius of the circle is sqrt 13. -/
theorem radius_of_circle : 
    ∃ r : ℝ, (∀ x y : ℝ, x^2 + y^2 - 8 = 2*x + 4*y → r = Real.sqrt 13) :=
by
    sorry

end radius_of_circle_l774_774337


namespace find_side_c_l774_774701

-- Definitions of the given conditions
def triangle_sides (A B C a b c : ℝ) : Prop :=
  a^2 = b^2 + c^2 - 2 * b * c * Real.cos A ∧
  A = Real.pi / 6 ∧
  a = 1 ∧
  b = Real.sqrt 3

-- Main statement to prove
theorem find_side_c {A B C a b c : ℝ} (h : triangle_sides A B C a b c) :
  c = 2 ∨ c = 1 :=
by
  -- Matching the conditions
  obtain ⟨h1, hA, ha, hb⟩ := h,
  have h_eq := h1, -- Use the cosine rule
  have h_cosine := h_eq.trans 
    (by simp only [hA, ha, hb, Real.sqrt_eq_rpow, Real.rpow_two, Real.cos_pi_div_six, zero_sub, MulZeroClass.zero_mul, add_left_neg, add_eq_zero_iff_eq_neg, neg_mul_eq_neg_mul_symm]; ring),
  sorry

end find_side_c_l774_774701


namespace minimum_n_minus_m_abs_l774_774177

theorem minimum_n_minus_m_abs (f g : ℝ → ℝ)
  (hf : ∀ x, f x = Real.exp x + 2 * x)
  (hg : ∀ x, g x = 4 * x)
  (m n : ℝ)
  (h_cond : f m = g n) :
  |n - m| = (1 / 2) - (1 / 2) * Real.log 2 := 
sorry

end minimum_n_minus_m_abs_l774_774177


namespace count_sequences_l774_774432

def seq (a: ℕ → ℕ) := ∀ n > 2, a n = 3 * a (n - 1) - 2 * a (n - 2)

theorem count_sequences (a: ℕ → ℕ) (h_seq: seq a)
  (h1: ∀ n, a n > 0)
  : (∃ s : set (ℕ → ℕ), ∀ f ∈ s, seq f ∧ a 1 = f 1 ∧ a 2 = f 2 ∧ ((f 2010) ≤ 2 ^ 2012) ∧ (card s = 36 * 2 ^ 2009 + 36)) :=
sorry

end count_sequences_l774_774432


namespace perfect_square_factors_450_l774_774605

theorem perfect_square_factors_450 :
  ∃ (n : ℕ), n = 4 ∧ ∀ (d : ℕ), d ∣ 450 → ∃ (k : ℕ), k^2 = d ↔ d = 1 ∨ d = 9 ∨ d = 25 ∨ d = 225 :=
by
  sorry

end perfect_square_factors_450_l774_774605


namespace first_graders_wearing_yellow_l774_774340

theorem first_graders_wearing_yellow (cf1 := 5.80) (kf := 101) (cf2 := 5.00) (cs2 := 5.60) (kf2 := 107)
  (cg := 5.25) (kf3 := 108) (total := 2317) : 
  let cost_k := kf * cf1,
      cost_g := kf3 * cg,
      cost_s2 := kf2 * cs2,
      total_without_f := cost_k + cost_g + cost_s2,
      cost_f := total - total_without_f,
      f_count := cost_f / cf2 in
  f_count = 113 :=
by
  sorry

end first_graders_wearing_yellow_l774_774340


namespace find_x_l774_774541

def vec := (ℝ × ℝ)

def a : vec := (1, 1)
def b (x : ℝ) : vec := (3, x)

def add_vec (v1 v2 : vec) : vec := (v1.1 + v2.1, v1.2 + v2.2)
def dot_product (v1 v2 : vec) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

theorem find_x (x : ℝ) (h : dot_product a (add_vec a (b x)) = 0) : x = -5 :=
by
  -- Proof steps (irrelevant for now)
  sorry

end find_x_l774_774541


namespace x_needs_20_days_to_finish_work_alone_l774_774044

-- Definitions based on the conditions
def work_done_by_y_per_day := 1 / 16
def work_done_by_y_in_12_days := 12 * work_done_by_y_per_day
def remaining_work_for_x := 1 - work_done_by_y_in_12_days
def x_worked_days := 5
def work_done_by_x_in_5_days := remaining_work_for_x
def work_rate_x := work_done_by_x_in_5_days / x_worked_days

-- Main theorem stating that x needs 20 days to finish the work alone
theorem x_needs_20_days_to_finish_work_alone :
  1 / work_rate_x = 20 :=
by
  unfold work_done_by_y_per_day work_done_by_y_in_12_days remaining_work_for_x x_worked_days work_done_by_x_in_5_days work_rate_x
  sorry

end x_needs_20_days_to_finish_work_alone_l774_774044


namespace sum_of_leading_coefficients_is_zero_l774_774013

theorem sum_of_leading_coefficients_is_zero
  (p q a b : ℝ)
  (h_a_nonzero : a ≠ 0)
  (h_vertex1_on_parabola2 : b = pa^2)
  (h_vertex2_on_parabola1 : 0 = qa^2 + b) :
  p + q = 0 :=
begin
  -- proof goes here
  sorry,
end

end sum_of_leading_coefficients_is_zero_l774_774013


namespace totalFriendsAreFour_l774_774307

-- Define the friends
def friends := ["Mary", "Sam", "Keith", "Alyssa"]

-- Define the number of friends
def numberOfFriends (f : List String) : ℕ := f.length

-- Claim that the number of friends is 4
theorem totalFriendsAreFour : numberOfFriends friends = 4 :=
by
  -- Skip proof
  sorry

end totalFriendsAreFour_l774_774307


namespace problem1_problem2_l774_774094

-- Problem 1: Prove that the given expression simplifies to a^{1/6}
theorem problem1 (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a ^ (2/3) * b⁻¹) ^ (-1/2) * a ^ (1/2) * b ^ (1/3) / (a * b^5) ^ (1/6) = a ^ (1/6) :=
sorry

-- Problem 2: Prove that the given logarithmic expression simplifies to 1/2
theorem problem2 : 
  (1/2 * log (32 / 49)) - (4 / 3 * log (sqrt 8)) + (log (sqrt 245)) = 1 / 2 :=
sorry

end problem1_problem2_l774_774094


namespace trinomial_has_two_roots_l774_774909

theorem trinomial_has_two_roots (a b c : ℝ) (h : b^2 - 4 * a * c > 0) :
  let Δ := (2 * (a + b))^2 - 4 * 3 * a * (b + c) in Δ > 0 :=
by
  sorry

end trinomial_has_two_roots_l774_774909


namespace find_a_l774_774198

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x > 0 then Real.log x else a^x

theorem find_a (a : ℝ) (h_pos : 0 < a) (h_ne : a ≠ 1)
  (h : f a (Real.exp 2) = f a (-2)) : a = Real.sqrt 2 / 2 :=
sorry

end find_a_l774_774198


namespace value_of_linear_combination_l774_774431

theorem value_of_linear_combination :
  ∀ (x1 x2 x3 x4 x5 : ℝ),
    2*x1 + x2 + x3 + x4 + x5 = 6 →
    x1 + 2*x2 + x3 + x4 + x5 = 12 →
    x1 + x2 + 2*x3 + x4 + x5 = 24 →
    x1 + x2 + x3 + 2*x4 + x5 = 48 →
    x1 + x2 + x3 + x4 + 2*x5 = 96 →
    3*x4 + 2*x5 = 181 :=
by
  intros x1 x2 x3 x4 x5 h1 h2 h3 h4 h5
  sorry

end value_of_linear_combination_l774_774431


namespace arithmetic_mean_l774_774783

theorem arithmetic_mean (a b c : ℚ) (h1 : a = 8/11) (h2 : b = 9/11) (h3 : c = 7/11) : a = (b + c) / 2 := by
  rw [h1, h2, h3]
  have : 8/11 = (9/11 + 7/11) / 2 := by sorry
  exact this

end arithmetic_mean_l774_774783


namespace number_of_perfect_square_factors_l774_774663

def is_prime_factorization_valid : Bool :=
  let p1 := 2
  let p1_pow := 1
  let p2 := 3
  let p2_pow := 2
  let p3 := 5
  let p3_pow := 2
  (450 = (p1^p1_pow) * (p2^p2_pow) * (p3^p3_pow))
  
def is_perfect_square (n : Nat) : Bool :=
  match n with
  | 1 => true
  | _ => let second_digit := Math.sqrt n * Math.sqrt n;
         second_digit == n

theorem number_of_perfect_square_factors : (number_of_perfect_square_factors 450 = 4) :=
by sorry

end number_of_perfect_square_factors_l774_774663


namespace coat_shirt_ratio_l774_774312

variable (P S C k : ℕ)

axiom h1 : P + S = 100
axiom h2 : P + C = 244
axiom h3 : C = k * S
axiom h4 : C = 180

theorem coat_shirt_ratio (P S C k : ℕ) (h1 : P + S = 100) (h2 : P + C = 244) (h3 : C = k * S) (h4 : C = 180) :
  C / S = 5 :=
sorry

end coat_shirt_ratio_l774_774312


namespace garden_perimeter_is_64_l774_774880

-- Define the playground dimensions and its area 
def playground_length := 16
def playground_width := 12
def playground_area := playground_length * playground_width

-- Define the garden width and its area being the same as the playground's area
def garden_width := 8
def garden_area := playground_area

-- Calculate the garden's length
def garden_length := garden_area / garden_width

-- Calculate the perimeter of the garden
def garden_perimeter := 2 * (garden_length + garden_width)

theorem garden_perimeter_is_64 :
  garden_perimeter = 64 := 
sorry

end garden_perimeter_is_64_l774_774880


namespace walk_direction_east_l774_774698

theorem walk_direction_east (m : ℤ) (h : m = -2023) : m = -(-2023) :=
by
  sorry

end walk_direction_east_l774_774698


namespace polynomial_root_recip_squares_l774_774106

theorem polynomial_root_recip_squares (a b c : ℝ) 
  (h1 : a + b + c = 6) (h2 : a * b + b * c + c * a = 11) (h3 : a * b * c = 6):
  1 / a^2 + 1 / b^2 + 1 / c^2 = 49 / 36 :=
sorry

end polynomial_root_recip_squares_l774_774106


namespace don_paints_tiles_equation_l774_774119

variable (D : ℕ)
variable (Ken Laura Kim total_tiles : ℕ)

def tiles_per_minute (D Ken Laura Kim : ℕ) := D + Ken + Laura + Kim

axiom don_tiles : ∀ D, Ken = D + 2
axiom laura_tiles : ∀ D, Laura = 2 * (D + 2)
axiom kim_tiles : ∀ D, Kim = 2 * (D + 2) - 3
axiom total_painted_in_15_min : ∀ D, total_tiles = 375 
axiom paint_rate : ∀ D, total_tiles = 15 * 25

theorem don_paints_tiles_equation : 
  ∀ D, 6 * D + 7 = 25 := sorry

end don_paints_tiles_equation_l774_774119


namespace player_A_max_cheese_l774_774855

theorem player_A_max_cheese : 
  ∃ A B : ℕ, 
    A = 30 ∧ B = 20 ∧
    ∀ (split : ℕ → ℕ → Prop),
      (split 50 2 → split 2 5) → 
      A + B = 50 := 
begin
  sorry
end

end player_A_max_cheese_l774_774855


namespace calculate_taxi_fare_l774_774472

theorem calculate_taxi_fare :
  ∀ (f_80 f_120: ℝ), f_80 = 160 ∧ f_80 = 20 + (80 * (140/80)) →
                      f_120 = 20 + (120 * (140/80)) →
                      f_120 = 230 :=
by
  intro f_80 f_120
  rintro ⟨h80, h_proportional⟩ h_120
  sorry

end calculate_taxi_fare_l774_774472


namespace exactly_two_pass_probability_l774_774946

theorem exactly_two_pass_probability (PA PB PC : ℚ) (hPA : PA = 2 / 3) (hPB : PB = 3 / 4) (hPC : PC = 2 / 5) :
  ((PA * PB * (1 - PC)) + (PA * (1 - PB) * PC) + ((1 - PA) * PB * PC) = 7 / 15) := by
  sorry

end exactly_two_pass_probability_l774_774946


namespace number_of_perfect_square_factors_l774_774666

def is_prime_factorization_valid : Bool :=
  let p1 := 2
  let p1_pow := 1
  let p2 := 3
  let p2_pow := 2
  let p3 := 5
  let p3_pow := 2
  (450 = (p1^p1_pow) * (p2^p2_pow) * (p3^p3_pow))
  
def is_perfect_square (n : Nat) : Bool :=
  match n with
  | 1 => true
  | _ => let second_digit := Math.sqrt n * Math.sqrt n;
         second_digit == n

theorem number_of_perfect_square_factors : (number_of_perfect_square_factors 450 = 4) :=
by sorry

end number_of_perfect_square_factors_l774_774666


namespace quadratic_trinomial_has_two_roots_l774_774918

theorem quadratic_trinomial_has_two_roots
  (a b c : ℝ) (h : b^2 - 4 * a * c > 0) :
  4 * (a^2 - a * b + b^2 - 3 * a * c) > 0 :=
by
  sorry

end quadratic_trinomial_has_two_roots_l774_774918


namespace jimmy_income_l774_774327

variable (J : ℝ)

def rebecca_income : ℝ := 15000
def income_increase : ℝ := 3000
def rebecca_income_after_increase : ℝ := rebecca_income + income_increase
def combined_income : ℝ := 2 * rebecca_income_after_increase

theorem jimmy_income (h : rebecca_income_after_increase + J = combined_income) : 
  J = 18000 := by
  sorry

end jimmy_income_l774_774327


namespace parallelogram_area_correct_l774_774517

-- Define parameters for the base and height of the parallelogram
def base : ℝ := 48
def height : ℝ := 36

-- Define the function to calculate the area of a parallelogram
def areaOfParallelogram (base height : ℝ) : ℝ := base * height

-- Define the theorem statement
theorem parallelogram_area_correct : areaOfParallelogram base height = 1728 := by
  -- proof would go here
  sorry

end parallelogram_area_correct_l774_774517


namespace second_machine_time_equation_l774_774074

theorem second_machine_time_equation :
  (∃ x : ℝ, (1 / 15 + 1 / x = 1 / 5)) :=
by
  -- Define the rates for the first and second machine given the conditions
  let rate1 := 1000 / 15
  let rate2 := 1000
  -- Define their combined rate under the given conditions
  let combined_rate := 1000 / 5
  -- Formulate the equation to find the second machine's time.
  have eq : 1 / 15 + 1 /( rate2 / x) = 1 / 5 := sorry
  exact ⟨_, _⟩

end second_machine_time_equation_l774_774074


namespace complete_square_l774_774487

theorem complete_square (x : ℝ) : (x^2 - 2 * x - 2 = 0) → ((x - 1)^2 = 3) :=
by
  intro h
  sorry

end complete_square_l774_774487


namespace cost_of_fencing_per_meter_l774_774352

theorem cost_of_fencing_per_meter :
  ∀ (b : ℕ) (fence_cost total_cost : ℕ), 
    let l := b + 60 in
    l = 80 →
    total_cost = 5300 →
    2 * l + 2 * b = 200 →
    fence_cost * (2 * l + 2 * b) = total_cost →
    fence_cost = 26.5 :=
by {
  intro b,
  intro fence_cost,
  intro total_cost,
  intros,
  sorry
}

end cost_of_fencing_per_meter_l774_774352


namespace existence_of_five_regular_polyhedra_l774_774073

def regular_polyhedron (n m : ℕ) : Prop :=
  n ≥ 3 ∧ m ≥ 3 ∧ (2 / m + 2 / n > 1)

theorem existence_of_five_regular_polyhedra :
  ∃ (n m : ℕ), regular_polyhedron n m → 
    (n = 3 ∧ m = 3 ∨ 
     n = 4 ∧ m = 3 ∨ 
     n = 3 ∧ m = 4 ∨ 
     n = 5 ∧ m = 3 ∨ 
     n = 3 ∧ m = 5) :=
by
  sorry

end existence_of_five_regular_polyhedra_l774_774073


namespace log_base_0_7_monotonicity_g_function_monotonicity_l774_774434

-- Problem 1
theorem log_base_0_7_monotonicity (x : ℝ) (h : x^2 - 3 * x + 2 > 0) :
  (x > 2 → (∀ x y, x < y → log (0.7) (y^2 - 3 * y + 2) < log (0.7) (x^2 - 3 * x + 2))) ∧
  (x < 1 → (∀ x y, x < y → log (0.7) (x^2 - 3 * x + 2) < log (0.7) (y^2 - 3 * y + 2))) :=
sorry

-- Problem 2
theorem g_function_monotonicity (g : ℝ → ℝ) (x : ℝ)
  (h : g x = 8 + 2 * (2 - x^2) - (2 - x^2)^2) :
  ((∀ x, x < -1 → g x < g (x + 1)) ∧
   (∀ x, 0 < x ∧ x < 1 → g x < g (x + 0.5)) ∧
   (∀ x, 1 < x → g (x - 1) > g x) ∧
   (∀ x, -1 < x ∧ x < 0 → g (x - 0.5) > g x)) :=
sorry

end log_base_0_7_monotonicity_g_function_monotonicity_l774_774434


namespace area_of_triangle_l774_774234

theorem area_of_triangle 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (median_AM : ℝ)
  (h1 : 2 * b - (real.sqrt 3) * c = (real.sqrt 3) * a * (real.cos C))
  (h2 : B = real.pi / 6) 
  (h3 : median_AM = real.sqrt 7) 
  (h4 : ∀ (a b c : ℝ), a * sin(A) = b * sin(B) = c * sin(C))
  : (1 / 2) * b ^ 2 * (real.sin C) = real.sqrt 3 :=
sorry

end area_of_triangle_l774_774234


namespace quadratic_real_roots_count_l774_774492

theorem quadratic_real_roots_count : 
  let s := {1, 2, 3, 4, 5, 7, 8, 9} in
  let valid_pairs := { (b, c) : ℕ × ℕ | b ∈ s ∧ c ∈ s ∧ (b % 2 = 1 ∨ c % 2 = 1) ∧ b^2 ≥ 4 * c } in
  valid_pairs.card = 19 :=
by sorry

end quadratic_real_roots_count_l774_774492


namespace directed_segments_relation_l774_774387

variables (A M N K B C D : Point)
variable [is_parallelogram A M N K]

def line_through (A : Point) : Line := sorry
def intersects_at (L1 L2 : Line) (P : Point) : Prop := sorry
def directed_segment (P Q : Point) : Segment := sorry

theorem directed_segments_relation
  (h1: intersects_at (line_through A) (line_through M K) B)
  (h2: intersects_at (line_through A) (line_through K N) C)
  (h3: intersects_at (line_through A) (line_through M N) D)
  (seg_AB : Segment := directed_segment A B)
  (seg_AC : Segment := directed_segment A C)
  (seg_AD : Segment := directed_segment A D)
  : 1 / seg_AB = 1 / seg_AC + 1 / seg_AD := sorry

end directed_segments_relation_l774_774387


namespace angle_BAC_degree_l774_774394

/-- Two tangents to a circle are drawn from a point A. The points of contact B and C divide
    the circle into arcs with lengths in the ratio 3 : 5.
    We want to prove that the degree measure of ∠BAC is 67.5°.
-/
theorem angle_BAC_degree (O A B C : Point) (hTangent1 : Tangent A B) (hTangent2 : Tangent A C)
    (hArcRatio : arcLength B C = 3 / 8 * 360 ∧ arcLength C B' = 5 / 8 * 360) :
    measureAngle BAC = 67.5 := 
sorry

end angle_BAC_degree_l774_774394


namespace Xiaolong_dad_age_correct_l774_774036
noncomputable def Xiaolong_age (x : ℕ) : ℕ := x
noncomputable def mom_age (x : ℕ) : ℕ := 9 * x
noncomputable def dad_age (x : ℕ) : ℕ := 9 * x + 3
noncomputable def dad_age_next_year (x : ℕ) : ℕ := 9 * x + 4
noncomputable def Xiaolong_age_next_year (x : ℕ) : ℕ := x + 1
noncomputable def dad_age_predicated_next_year (x : ℕ) : ℕ := 8 * (x + 1)

theorem Xiaolong_dad_age_correct (x : ℕ) (h : 9 * x + 4 = 8 * (x + 1)) : dad_age x = 39 := by
  sorry

end Xiaolong_dad_age_correct_l774_774036


namespace has_two_roots_l774_774913

-- Define the discriminant of the quadratic trinomial
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Main Lean statement
theorem has_two_roots
  (a b c : ℝ)
  (h : discriminant a b c > 0) :
  discriminant (3 * a) (2 * (a + b)) (b + c) > 0 := by
  sorry

end has_two_roots_l774_774913


namespace number_of_incorrect_expressions_is_3_l774_774477

def expr1 : Prop := ¬ ({0} ∈ ({0, 1, 2} : set ℕ))
def expr2 : Prop := ∅ ⊆ ({0} : set ℕ)
def expr3 : Prop := ({0, 1, 2} : set ℕ) ⊆ ({1, 2, 0} : set ℕ)
def expr4 : Prop := ¬ (0 ∈ (∅ : set ℕ))
def expr5 : Prop := (∅ : set ℕ) ∩ ({0} : set ℕ) = ∅

theorem number_of_incorrect_expressions_is_3 :
  (¬ expr1) ∧ (expr2) ∧ (expr3) ∧ (¬ expr4) ∧ (¬ expr5) → 3 = 3 :=
by
  sorry

end number_of_incorrect_expressions_is_3_l774_774477


namespace perfect_square_factors_count_450_l774_774675

theorem perfect_square_factors_count_450 : 
  (∃ n : ℕ, n = 4 ∧ (∀ d : ℕ, d ∣ 450 → ∃ k : ℕ, d = k * k)) := sorry

end perfect_square_factors_count_450_l774_774675


namespace polynomial_roots_l774_774528

theorem polynomial_roots :
  (∃ x : ℝ, x = 2 ∨ x = 3 ∨ x = 1 / 2 ∨ x = 1 / 3) → 
  (6 * (x ^ 4) - 35 * (x ^ 3) + 62 * (x ^ 2) - 35 * x + 6 = 0) :=
by
  intros x h
  cases h
  -- 
  sorry

end polynomial_roots_l774_774528


namespace slices_with_both_toppings_l774_774049

-- Definitions and conditions directly from the problem statement
def total_slices : ℕ := 24
def pepperoni_slices : ℕ := 15
def mushroom_slices : ℕ := 14

-- Theorem proving the number of slices with both toppings
theorem slices_with_both_toppings :
  (∃ n : ℕ, n + (pepperoni_slices - n) + (mushroom_slices - n) = total_slices) → ∃ n : ℕ, n = 5 := 
by 
  sorry

end slices_with_both_toppings_l774_774049


namespace parallel_perpendicular_l774_774790

variable {P : Type} [Plane P] (a b c : Line P)

theorem parallel_perpendicular (h1 : a ⊥ c) (h2 : b ⊥ c) : a ∥ b :=
by sorry

end parallel_perpendicular_l774_774790


namespace determine_key_placement_l774_774841

structure Box (color : Type) :=
(sentence : string)

def red : Box := Box.mk "The key is not in here."
def yellow : Box := Box.mk "The key is in here."
def blue : Box := Box.mk "The key is not in the yellow box."

theorem determine_key_placement (red_sentence_true : ¬ (key_in red)) 
(yellow_sentence_true : key_in yellow) 
(blue_sentence_true : ¬ (key_in yellow))
(exactly_one_true : (red_sentence_true → ¬ yellow_sentence_true) ∧ 
(red_sentence_true → ¬ blue_sentence_true) ∧ 
(yellow_sentence_true → ¬ red_sentence_true) ∧ 
(yellow_sentence_true → ¬ blue_sentence_true) ∧ 
(blue_sentence_true → ¬ red_sentence_true) ∧ 
(blue_sentence_true → ¬ yellow_sentence_true)) :
  key_in red :=
sorry

end determine_key_placement_l774_774841


namespace frank_used_2_bags_l774_774539

theorem frank_used_2_bags (total_candy : ℕ) (candy_per_bag : ℕ) (h1 : total_candy = 16) (h2 : candy_per_bag = 8) : (total_candy / candy_per_bag) = 2 := 
by
  sorry

end frank_used_2_bags_l774_774539


namespace matrix_A3_zero_l774_774757

open Matrix

variables {R : Type*} [Field R]

def mat2 := Matrix (Fin 2) (Fin 2) R

theorem matrix_A3_zero (A : mat2) (h : A ^ 4 = 0) : A ^ 3 = 0 := by
  sorry

end matrix_A3_zero_l774_774757


namespace negation_of_existential_l774_774355

def symmetrical_figure (T : Type) : Prop := 
  sorry -- define what it means to be a symmetrical figure

def triangle (T : Type) : Prop := 
  sorry -- define what it means to be a triangle

theorem negation_of_existential (T : Type) [triangle T] :
  ¬(∃ t : T, symmetrical_figure t) ↔ ∀ t : T, ¬symmetrical_figure t :=
by
  sorry

end negation_of_existential_l774_774355


namespace largest_common_number_in_arithmetic_sequences_l774_774947

theorem largest_common_number_in_arithmetic_sequences (x : ℕ)
  (h1 : x ≡ 2 [MOD 8])
  (h2 : x ≡ 5 [MOD 9])
  (h3 : x < 200) : x = 194 :=
by sorry

end largest_common_number_in_arithmetic_sequences_l774_774947


namespace find_minimal_period_find_max_value_find_sin_2alpha_l774_774200

noncomputable def f (x : ℝ) : ℝ := sin x + sin (x + π / 2)

theorem find_minimal_period : ∃ T > 0, ∀ x : ℝ, f (x + T) = f x ∧ ∀ T' > 0, f (x + T') = f x → T' ≥ T :=
  sorry

theorem find_max_value : ∃ (max_ : ℝ) (xs : set ℝ), 
  (∀ x : ℝ, f x ≤ max_) ∧ 
  (∀ k : ℤ, (π / 4 + 2 * k * π) ∈ xs) ∧ 
  (∀ x ∈ xs, f x = max_) :=
  sorry

theorem find_sin_2alpha (α : ℝ) (h : f α = 3 / 4) : sin (2 * α) = -23 / 32 :=
  sorry

end find_minimal_period_find_max_value_find_sin_2alpha_l774_774200


namespace count_integers_reaching_1_l774_774753

def f (n : ℕ) : ℕ :=
  if n % 2 = 1 then n^3 + 1 else n / 2

def reaches_one (n : ℕ) : Prop :=
  ∃ k : ℕ, (Nat.iterate f k n = 1)

theorem count_integers_reaching_1 :
  (Finset.filter reaches_one (Finset.range 101)).card = 7 := 
sorry

end count_integers_reaching_1_l774_774753


namespace mul_104_96_l774_774975

theorem mul_104_96 : 104 * 96 = 9984 := by
  calc
    104 * 96 = (100 + 4) * (100 - 4) : by rw [← rfl, ← rfl]
    ... = 100^2 - 4^2 : by apply mul_self_sub_mul_self
    ... = 10000 - 16 : by rw [pow_two, pow_two]
    ... = 9984 : by norm_num


end mul_104_96_l774_774975


namespace find_y_l774_774151

theorem find_y {y : ℕ} (h : 16^(-3) = 4^(72/y) / (4^(42/y) * 16^(25/y))) : y = 10 / 3 :=
sorry

end find_y_l774_774151


namespace prime_root_ratio_l774_774014

theorem prime_root_ratio (a b : ℕ) (ha : a.prime) (hb : b.prime) (h_sum : a + b = 21) (h_root : ∃ t : ℤ, t = a * b ∧ (∀ x : ℤ, x^2 - 21 * x + t = 0)) : 
  (b / a + a / b : ℚ) = 365 / 38 :=
sorry

end prime_root_ratio_l774_774014


namespace quadratic_trinomial_has_two_roots_l774_774922

theorem quadratic_trinomial_has_two_roots
  (a b c : ℝ) (h : b^2 - 4 * a * c > 0) : (2 * (a + b))^2 - 4 * 3 * a * (b + c) > 0 := by
  sorry

end quadratic_trinomial_has_two_roots_l774_774922


namespace angle_and_ratio_problem_l774_774282

-- Define given conditions
variables {A B C a b c : ℝ}
variable [hApos : 0 < A]
variable [hBpos : 0 < B]
variable [hCpos : 0 < C]
variable [hsum : A + B + C = π]
variable [hacosC : b + a * Real.cos C = 0]
variable [hsinA : Real.sin A = 2 * Real.sin (A + C)]

-- Hypothesis that angles are opposite to sides
hypothesis (hangles :=
  ∃ (A B C a b c : ℝ),  
    0 < A ∧ 0 < B ∧ 0 < C ∧ 
    A + B + C = π ∧ 
    b + a * Real.cos C = 0 ∧ 
    Real.sin A = 2 * Real.sin (A + C))

-- Define the target proof problem
theorem angle_and_ratio_problem :
  (C = 2 * π / 3 ∧ (c / a) = Real.sqrt 2) → 
  (0 < C ∧ C < π) :=
by 
  intros hC hratio
  sorry

end angle_and_ratio_problem_l774_774282


namespace range_of_m_l774_774156

theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, 3 * sin x + 4 * cos x = 2 * m - 1) ↔ -2 ≤ m ∧ m ≤ 3 := 
sorry

end range_of_m_l774_774156


namespace remainder_of_5_pow_2023_mod_6_l774_774401

theorem remainder_of_5_pow_2023_mod_6 : 5^2023 % 6 = 5 := 
by sorry

end remainder_of_5_pow_2023_mod_6_l774_774401


namespace find_radius_of_small_semicircle_l774_774733

noncomputable def radius_of_small_semicircle (R : ℝ) (r : ℝ) :=
  ∀ (x : ℝ),
    (12: ℝ = R) ∧ (6: ℝ = r) →
    (∃ (x: ℝ), R - x + r = sqrt((r + x)^2 - r^2)) →
    x = 4

theorem find_radius_of_small_semicircle : radius_of_small_semicircle 12 6 :=
begin
  unfold radius_of_small_semicircle,
  intro x,
  assume h1 h2,
  cases h2,
  sorry,
end

end find_radius_of_small_semicircle_l774_774733


namespace erin_total_money_l774_774125

def quarters_per_machine := 80
def dimes_per_machine := 100
def value_per_quarter := 0.25
def value_per_dime := 0.10
def machines := 3

theorem erin_total_money :
  (quarters_per_machine * value_per_quarter + dimes_per_machine * value_per_dime) * machines = 90 :=
by
  sorry

end erin_total_money_l774_774125


namespace perfect_square_divisors_count_450_l774_774646

theorem perfect_square_divisors_count_450 :
  let is_divisor (d n : ℕ) := n % d = 0
  let is_perfect_square (n : ℕ) := ∃ m : ℕ, m * m = n
  let prime_factorization_450 := [(2, 1), (3, 2), (5, 2)]
  ∃ (count : ℕ), count = 4 ∧ count = (Finset.filter 
    (λ d, is_perfect_square d ∧ is_divisor d 450)
    (Finset.powersetPrimeFactors prime_factorization_450)).card := sorry

end perfect_square_divisors_count_450_l774_774646


namespace planes_divide_space_l774_774208

-- Define the types for planes and space
universe u
constant α : Type u
constant Plane : Type u
constant divide_parts : Plane → Plane → ℕ

-- Define the conditions: parallel and intersecting planes
axiom parallel (p1 p2 : Plane) : Prop
axiom intersect (p1 p2 : Plane) : Prop

-- Axiom for the number of parts divided by parallel planes
axiom parallel_parts (p1 p2 : Plane) : parallel p1 p2 → divide_parts p1 p2 = 3

-- Axiom for the number of parts divided by intersecting planes
axiom intersect_parts (p1 p2 : Plane) : intersect p1 p2 → divide_parts p1 p2 = 4

-- The theorem to prove
theorem planes_divide_space (p1 p2 : Plane) : (parallel p1 p2 ∧ divide_parts p1 p2 = 3) ∨ (intersect p1 p2 ∧ divide_parts p1 p2 = 4) := 
sorry

end planes_divide_space_l774_774208


namespace inscribed_parallelogram_is_rectangle_circumscribed_parallelogram_is_rhombus_both_inscribed_and_circumscribed_is_square_l774_774417

-- Problem 1: Inscribed parallelograms are rectangles
theorem inscribed_parallelogram_is_rectangle (P : Parallelogram) (h_suppl : ∀ α β, α = P.angleA → β = P.angleC → α + β = 180) :
  P.isInscribedInCircle ↔ P.isRectangle :=
by sorry

-- Problem 2: Circumscribed parallelograms are rhombuses
theorem circumscribed_parallelogram_is_rhombus (P : Parallelogram) (h_sum_lengths : ∀ a b, a = P.sideA → b = P.sideB → a + P.sideC = b + P.sideD) :
  P.isCircumscribedAroundCircle ↔ P.isRhombus :=
by sorry

-- Problem 3: Both inscribed and circumscribed parallelograms are squares
theorem both_inscribed_and_circumscribed_is_square (P : Parallelogram) :
  (P.isInscribedInCircle ∧ P.isCircumscribedAroundCircle) ↔ P.isSquare :=
by sorry

end inscribed_parallelogram_is_rectangle_circumscribed_parallelogram_is_rhombus_both_inscribed_and_circumscribed_is_square_l774_774417


namespace largest_constant_D_l774_774997

theorem largest_constant_D (D : ℝ) 
  (h : ∀ (x y : ℝ), x^2 + y^2 + 4 ≥ D * (x + y)) : 
  D ≤ 2 * Real.sqrt 2 :=
sorry

end largest_constant_D_l774_774997


namespace parabola_equation_standard_form_l774_774529

theorem parabola_equation_standard_form (p : ℝ) (x y : ℝ)
    (h₁ : y^2 = 2 * p * x)
    (h₂ : y = -4)
    (h₃ : x = -2) : y^2 = -8 * x := by
  sorry

end parabola_equation_standard_form_l774_774529


namespace part1_part2_l774_774155

-- Definitions based on the conditions provided
def f (n a : ℕ) : ℕ := sorry  -- This represents the number of coefficients divisible by 3 in (x+1)^a * (x+2)^(n-a).

def F (n : ℕ) : ℕ := 
  Nat.min (List.map (λ a, f n a) (List.range (n + 1)))

-- Part (1): Prove that there exist infinitely many positive integer n such that F(n) ≥ (n - 1) / 3.
theorem part1 : ∃ᶠ n in Nat.at_top, F n ≥ (n - 1) / 3 := sorry

-- Part (2): Prove that for any positive integer n, F(n) ≤ (n - 1) / 3.
theorem part2 : ∀ n:ℕ, 0 < n → F n ≤ (n - 1) / 3 := sorry

end part1_part2_l774_774155


namespace numPerfectSquareFactorsOf450_l774_774678

def isPerfectSquare (n : Nat) : Prop :=
  ∃ k : Nat, n = k * k

theorem numPerfectSquareFactorsOf450 : 
  ∃! n : Nat, 
    (∀ d : Nat, d ∣ 450 → isPerfectSquare d) → n = 4 := 
by
  sorry

end numPerfectSquareFactorsOf450_l774_774678


namespace a_n_formula_correct_sum_S_n_formula_correct_l774_774558

def a : ℕ → ℤ
| 0       := 1
| 1       := 4
| (n + 2) := 3 * a (n + 1) - 2 * a n

def general_formula (n : ℕ) : ℤ := 3 * 2^(n - 1) - 2

noncomputable def S (n : ℕ) : ℤ := ∑ k in finset.range (n + 1), a k

noncomputable def S_formula (n : ℕ) : ℤ := 3 * (2^(n + 1) - 1) - 2 * (n + 1)

theorem a_n_formula_correct (n : ℕ) : a n = general_formula n := sorry

theorem sum_S_n_formula_correct (n : ℕ) : S n = S_formula n := sorry

end a_n_formula_correct_sum_S_n_formula_correct_l774_774558


namespace product_divisible_by_3_l774_774802

noncomputable def probability_divisible_by_3 : ℚ :=
  let spinner_C := {1, 2, 3, 4, 5}
  let spinner_D := {1, 2, 3, 4}
  let prob_C_not_div_by_3 := (spinner_C \ {3}).card.to_rat / spinner_C.card
  let prob_D_not_div_by_3 := (spinner_D \ {3}).card.to_rat / spinner_D.card
  1 - prob_C_not_div_by_3 * prob_D_not_div_by_3

theorem product_divisible_by_3:
  probability_divisible_by_3 = 2 / 5 := by
  sorry

end product_divisible_by_3_l774_774802


namespace color_count_l774_774017

theorem color_count :
  ∃ c : Fin 10 → Fin 3, 
    (∀ i j : Fin 10, (i - j).odd → c i ≠ c j) ∧ 
    set.card {c : Fin 10 → Fin 3 // ∀ i j : Fin 10, (i - j).odd → c i ≠ c j} = 186 :=
by
  -- Existence part
  sorry

end color_count_l774_774017


namespace product_of_binom_l774_774101

open Nat

-- Definitions of binomial coefficients and their properties.

def binom (n k : ℕ) : ℕ :=
  (factorial n) / (factorial k * factorial (n - k))

theorem product_of_binom (h1 : binom 12 6 = 924) (h2 : binom 5 2 = 10) : 
  binom 12 6 * (binom 5 2)^2 = 92400 :=
by
  -- The proof is not required, just the statement.
  have h3 : (binom 5 2)^2 = 100 := by
  sorry
  calc
    binom 12 6 * (binom 5 2)^2 = 924 * 100 : by rw [h1, h3]
                              ... = 92400 : by norm_num

end product_of_binom_l774_774101


namespace perfect_square_factors_count_450_l774_774673

theorem perfect_square_factors_count_450 : 
  (∃ n : ℕ, n = 4 ∧ (∀ d : ℕ, d ∣ 450 → ∃ k : ℕ, d = k * k)) := sorry

end perfect_square_factors_count_450_l774_774673


namespace oldest_child_age_l774_774343

-- Define the conditions
def average_age (a b c : ℕ) : ℕ := (a + b + c) / 3
def age_youngest := 5
def age_middle := 8
def average := 8

-- define age of oldest and the theorem
def age_oldest := 24 - (age_youngest + age_middle)

theorem oldest_child_age :
  age_oldest = 11 :=
by
  -- Provide the conditions and the proof (extra where necessary)
  have h1 : average_age age_youngest age_middle age_oldest = average, 
    from rfl,
  sorry

end oldest_child_age_l774_774343


namespace determinant_of_given_matrix_l774_774093

-- Define the matrix
def matrix_2x2 := ![![2, 4], ![1, 3]]

-- Define the determinant function for a 2x2 matrix
def det_2x2 (M : Matrix (Fin 2) (Fin 2) ℝ) : ℝ :=
  M 0 0 * M 1 1 - M 0 1 * M 1 0

-- The proof problem: prove the determinant of the given matrix is equal to 2
theorem determinant_of_given_matrix : det_2x2 matrix_2x2 = 2 :=
  sorry

end determinant_of_given_matrix_l774_774093


namespace calc_100m_plus_n_l774_774276

noncomputable def f (x : ℝ) : ℝ := x^3 + 3 * x - 1

lemma sum_of_roots_eq_zero {a b c : ℝ} (h : f a = 0) (h1 : f b = 0) (h2 : f c = 0) : a + b + c = 0 :=
by sorry

lemma prod_of_roots_eq_three {a b c : ℝ} (h : f a = 0) (h1 : f b = 0) (h2 : f c = 0) : a * b + b * c + c * a = 3 :=
by sorry

lemma product_of_roots_eq_one {a b c : ℝ} (h : f a = 0) (h1 : f b = 0) (h2 : f c = 0) : a * b * c = 1 :=
by sorry

theorem calc_100m_plus_n :
  (∃ (a b c : ℝ) (m n : ℕ), f a = 0 ∧ f b = 0 ∧ f c = 0 ∧
    ∑ (x : ℝ) in {a, b, c}, 1 / (3 - x^3) = 39 / 89 ∧
    m = 39 ∧ n = 89 ∧ ∃ (g : ℕ), m.gcd n = g ∧ g = 1) →
  100 * 39 + 89 = 3989 :=
by sorry

end calc_100m_plus_n_l774_774276


namespace concave_convex_tendency_range_l774_774230

noncomputable def f'' (x : ℝ) (m : ℝ) : ℝ := m / x - 2 * Real.log x

def concave_convex_tendency (f'' : ℝ → ℝ → ℝ) : Prop :=
  ∃ x1 x2 > 0, x1 < x2 ∧ f'' x1 = 0 ∧ f'' x2 = 0

theorem concave_convex_tendency_range (m : ℝ) :
  concave_convex_tendency (f'' m) ↔ m ∈ Ioo (-2 / Real.exp 1) 0 :=
sorry

end concave_convex_tendency_range_l774_774230


namespace find_annual_interest_rate_l774_774518

theorem find_annual_interest_rate :
  ∃ r : ℝ, r = 0.04 ∧ ∃ P t n CI A : ℝ, 
    P = 10000 ∧
    t = 2 ∧
    n = 2 ∧
    CI = 824.32 ∧
    A = P + CI ∧
    A = P * (1 + r / n) ^ (n * t) :=
begin
  sorry
end

end find_annual_interest_rate_l774_774518


namespace range_of_a_l774_774867

noncomputable theory

-- Definitions of the conditions
def condition_1 (x : ℝ) : Prop := 0 < x ∧ x ≤ 1 / 2
def condition_2 (a x : ℝ) : Prop := 4 * sin (π / 3 * x) - log a x < 0

-- The theorem
theorem range_of_a (a : ℝ) : (∀ x : ℝ, condition_1 x → condition_2 a x) → (a > sqrt 2 / 2 ∧ a < 1) :=
by
  intros h
  sorry

end range_of_a_l774_774867


namespace minimum_n_l774_774720

noncomputable def a (n : ℕ) : ℕ := 2 ^ (n - 2)

noncomputable def b (n : ℕ) : ℕ := n - 6 + a n

noncomputable def S (n : ℕ) : ℕ := (n * (n - 11)) / 2 + (2 ^ n - 1) / 2

theorem minimum_n (n : ℕ) (hn : n ≥ 5) : S 5 > 0 := by
  sorry

end minimum_n_l774_774720


namespace probability_even_sum_l774_774011

-- Defining the set and necessary functions
def numbers : Finset ℕ := {1, 1, 2, 3, 3, 4, 5}

-- Function to check if a pair of numbers sum to an even number
def is_even_sum (a b : ℕ) : Prop :=
  (a + b) % 2 = 0

-- Pairs of distinct numbers from the set
def distinct_pairs := {p : ℕ × ℕ // p.fst ≠ p.snd ∧ p.fst ∈ numbers ∧ p.snd ∈ numbers}

-- Counting pairs with even sum
def even_sum_pairs : Finset (ℕ × ℕ) :=
  Finset.filter (λ p, is_even_sum p.fst p.snd) (numbers.product numbers)

-- The probability calculation
theorem probability_even_sum : 
  (even_sum_pairs.card : ℚ) / (distinct_pairs.card : ℚ) = 4 / 21 :=
by
  sorry

end probability_even_sum_l774_774011


namespace complete_the_square_correct_l774_774029

noncomputable def complete_the_square (x : ℝ) : Prop :=
  x^2 - 2 * x - 1 = 0 ↔ (x - 1)^2 = 2

theorem complete_the_square_correct : ∀ x : ℝ, complete_the_square x := by
  sorry

end complete_the_square_correct_l774_774029


namespace diesel_amount_is_four_l774_774448

noncomputable theory

-- Definitions for the conditions
def diesel (D : ℝ) := D
def petrol := 4
def water := 2.666666666666667

-- Ratio condition
def ratio_condition (D : ℝ) : Prop := (3 / 5) = D / (petrol + water)

-- Proven statement
theorem diesel_amount_is_four (D : ℝ) (h : ratio_condition D) : D = 4 :=
by {
  sorry
}

end diesel_amount_is_four_l774_774448


namespace swans_after_10_years_l774_774137

-- Defining the initial conditions
def initial_swans : ℕ := 15

-- Condition that the number of swans doubles every 2 years
def double_every_two_years (n t : ℕ) : ℕ := n * (2 ^ (t / 2))

-- Prove that after 10 years, the number of swans will be 480
theorem swans_after_10_years : double_every_two_years initial_swans 10 = 480 :=
by
  sorry

end swans_after_10_years_l774_774137


namespace horizontal_length_of_monitor_l774_774304

def monitor_diagonal := 32
def aspect_ratio_horizontal := 16
def aspect_ratio_height := 9

theorem horizontal_length_of_monitor :
  ∃ (horizontal_length : ℝ), horizontal_length = 512 / Real.sqrt 337 := by
  sorry

end horizontal_length_of_monitor_l774_774304


namespace perfect_square_factors_450_l774_774610

theorem perfect_square_factors_450 :
  ∃ (n : ℕ), n = 4 ∧ ∀ (d : ℕ), d ∣ 450 → ∃ (k : ℕ), k^2 = d ↔ d = 1 ∨ d = 9 ∨ d = 25 ∨ d = 225 :=
by
  sorry

end perfect_square_factors_450_l774_774610


namespace jonah_served_first_intermission_l774_774512

theorem jonah_served_first_intermission : 
  let pitchers_second := 0.4166666666666667
  let pitchers_third := 0.25
  let total_pitchers := 0.9166666666666666
  total_pitchers - (pitchers_second + pitchers_third) = 0.25 :=
by 
  let pitchers_second := 0.4166666666666667
  let pitchers_third := 0.25
  let total_pitchers := 0.9166666666666666
  calc
    total_pitchers - (pitchers_second + pitchers_third)
    = 0.9166666666666666 - (0.4166666666666667 + 0.25) : by sorry
    = 0.25 : by sorry

end jonah_served_first_intermission_l774_774512


namespace smallest_b_is_2_plus_sqrt_3_l774_774298

open Real

noncomputable def smallest_b (a b : ℝ) : ℝ :=
  if (2 < a ∧ a < b ∧ (¬(2 + a > b ∧ 2 + b > a ∧ a + b > 2)) ∧
    (¬(1 / b + 1 / a > 2 ∧ 1 / a + 2 > 1 / b ∧ 2 + 1 / b > 1 / a)))
  then b else 0

theorem smallest_b_is_2_plus_sqrt_3 (a b : ℝ) :
  2 < a ∧ a < b ∧ (¬(2 + a > b ∧ 2 + b > a ∧ a + b > 2)) ∧
    (¬(1 / b + 1 / a > 2 ∧ 1 / a + 2 > 1 / b ∧ 2 + 1 / b > 1 / a)) →
  b = 2 + sqrt 3 := sorry

end smallest_b_is_2_plus_sqrt_3_l774_774298


namespace eccentricity_range_l774_774176

open Real

theorem eccentricity_range (a b c : ℝ) (λ : ℝ) 
  (h1 : a > b) (h2 : b > 0) (h3 : 1/2 ≤ λ) (h4 : λ ≤ 2) 
  (P : ℝ × ℝ) (h5 : abs (P.1 + c) = λ * abs (P.2 - c))
  (h6 : angle (P.1 + c) (P.2 - c) = pi / 2) 
  (e : ℝ) (h7 : c^2 = a^2 - b^2) : 
  sqrt 2 / 2 ≤ e ∧ e ≤ sqrt 5 / 3 := 
sorry

end eccentricity_range_l774_774176


namespace diagonal_square_length_eq_30_l774_774860

theorem diagonal_square_length_eq_30 (A : ℝ) (hA : A = 450) :
  let s := Real.sqrt A in
  let d := s * Real.sqrt 2 in
  d = 30 :=
by
  sorry

end diagonal_square_length_eq_30_l774_774860


namespace numPerfectSquareFactorsOf450_l774_774682

def isPerfectSquare (n : Nat) : Prop :=
  ∃ k : Nat, n = k * k

theorem numPerfectSquareFactorsOf450 : 
  ∃! n : Nat, 
    (∀ d : Nat, d ∣ 450 → isPerfectSquare d) → n = 4 := 
by
  sorry

end numPerfectSquareFactorsOf450_l774_774682


namespace sum_n_k_l774_774203

variable (A : ℕ → Set ℕ) (n : ℕ → ℕ)

-- Condition: Definition of the set A_k
def A_k (k : ℕ) : Set ℕ :=
  {x | ∃ (a : Fin (k-1) → Fin 2),
        x = 2^(k-1) + (Fin (k-1)).Sum (λ i, a i * 2^(a i + i))}

-- Condition: n_k represents the sum of all elements in A_k
def n_k (k : ℕ) : ℕ :=
  (A_k k).Sum id

-- Theorem statement to prove the required sum
theorem sum_n_k : ∑ k in Finset.range 2015, n_k k = (2^4031 + 1) / 3 - 2^2015 := sorry

end sum_n_k_l774_774203


namespace smallest_delicious_l774_774329

def is_delicious (n : ℕ) : Prop :=
  n ≥ 2 ∧ ∃ (a : Fin n → ℕ), (∑ i, a i = 2014) ∧ (Function.Injective (λ i, a i % n))

theorem smallest_delicious : ∃ n, is_delicious n ∧ ∀ m, is_delicious m → n ≤ m ∧ n = 4 := 
by
  sorry

end smallest_delicious_l774_774329


namespace perfect_square_divisors_count_450_l774_774649

theorem perfect_square_divisors_count_450 :
  let is_divisor (d n : ℕ) := n % d = 0
  let is_perfect_square (n : ℕ) := ∃ m : ℕ, m * m = n
  let prime_factorization_450 := [(2, 1), (3, 2), (5, 2)]
  ∃ (count : ℕ), count = 4 ∧ count = (Finset.filter 
    (λ d, is_perfect_square d ∧ is_divisor d 450)
    (Finset.powersetPrimeFactors prime_factorization_450)).card := sorry

end perfect_square_divisors_count_450_l774_774649


namespace total_money_l774_774893

theorem total_money (a b c : ℕ) (h_ratio : (a / 2) / (b / 3) / (c / 4) = 1) (h_c : c = 306) : 
  a + b + c = 782 := 
by sorry

end total_money_l774_774893


namespace f_four_sum_f_seven_sum_f_n_l774_774982

def f (n : ℕ) : ℚ := 1 / (2 ^ n + 1 : ℚ) - 1 / (2 ^ (n + 1) + 1 : ℚ)

theorem f_four : f 4 = 16 / 561 :=
by sorry

theorem sum_f_seven : (∑ k in Finset.range 7, f (k + 1)) = 253 / 771 :=
by sorry

theorem sum_f_n (n : ℕ) : (∑ k in Finset.range n, f (k + 1)) = (2 ^ (n + 1) - 2) / (3 * (2 ^ (n + 1) + 1 : ℚ)) :=
by sorry

end f_four_sum_f_seven_sum_f_n_l774_774982


namespace common_divisors_45_48_l774_774602

theorem common_divisors_45_48 : 
  let divisors_45 := {1, -1, 3, -3, 5, -5, 9, -9, 15, -15, 45, -45}
  let divisors_48 := {1, -1, 2, -2, 3, -3, 4, -4, 6, -6, 8, -8, 12, -12, 16, -16, 24, -24, 48, -48}
  (divisors_45 ∩ divisors_48).card = 4 := 
by
  sorry

end common_divisors_45_48_l774_774602


namespace perimeter_triangle_MNC_l774_774789

/-- 
Points M and N are the midpoints of the sides AC and CB of the isosceles triangle ACB.
Point L lies on the median BM such that BL : BM = 4 : 9.
A circle with center at point L is tangent to the line MN and intersects the line AB at points Q and T.
QT = 2 and AB = 8.
Prove that the perimeter of the triangle MNC is 2(2 + sqrt(13)).
-/
theorem perimeter_triangle_MNC
  (A B C M N L Q T : Point)
  (ACB_isosceles : is_isosceles_triangle A C B)
  (M_midpoint : is_midpoint M A C)
  (N_midpoint : is_midpoint N C B)
  (L_on_median : lies_on_median L B M 4 9)
  (circle_at_L_tangent_to_MN : tangential_at L MN)
  (Q_T_intersects_AB : intersects_line Q T AB)
  (QT_eq_2 : distance Q T = 2)
  (AB_eq_8 : distance A B = 8):
  perimeter (triangle M N C) = 2 * (2 + sqrt 13) :=
by sorry

end perimeter_triangle_MNC_l774_774789


namespace grid_sum_l774_774398

theorem grid_sum (n : ℕ) (b : Fin n → Fin n) : 
  (∑ i : Fin n, (i : ℕ) * n + (b i : ℕ)) = (n ^ 3 + n) / 2 := 
  sorry

end grid_sum_l774_774398


namespace card_orders_ascending_l774_774938

theorem card_orders_ascending (cards : List ℕ) (h : cards.length = 17 ∧ cards.perm (List.range 1 18))
  (h_sorted : ∃ (card : ℕ) (pos : ℕ), card ∈ cards ∧ 0 ≤ pos ∧ pos ≤ 17 ∧ sorted (remove_nth card cards |> insert_nth pos card)) :
  ∃ orders : Finset (List ℕ), orders.card = 256 := by
  sorry

end card_orders_ascending_l774_774938


namespace integer_values_factorable_l774_774992

theorem integer_values_factorable : Set {n : ℤ | ∃ a b c d e : ℤ, 
  (c = -a) ∧ (e = (b - a + d)) ∧ (p(x) = (x^2 + ax + b) * (x^3 + cx^2 + dx + e)) ∧ integer_coefficient_polynomials (x^2 + ax + b) (x^3 + cx^2 + dx + e)}
    = Set [-2, -1, 10, 19, 34, 342] :=
    sorry

end integer_values_factorable_l774_774992


namespace selection_with_at_least_three_surgeons_arrangement_girls_together_arrangement_girls_not_adjacent_l774_774887

theorem selection_with_at_least_three_surgeons :
  (choose 5 3 * choose 4 2) + (choose 5 4 * choose 4 1) + (choose 5 5 * choose 4 0) = 81 :=
by
  sorry

theorem arrangement_girls_together :
  (fact 6 * fact 4) = 17280 :=
by
  sorry

theorem arrangement_girls_not_adjacent :
  (fact 5 * (choose 6 4) * fact 4) = 43200 :=
by
  sorry

end selection_with_at_least_three_surgeons_arrangement_girls_together_arrangement_girls_not_adjacent_l774_774887


namespace coordinate_sum_l774_774189

theorem coordinate_sum (f : ℝ → ℝ) (h₁ : f 6 = 10) : 
  let y := (5 * f (3 * 2) + 7) / 2 in
  2 + y = 30.5 :=
by
  sorry

end coordinate_sum_l774_774189


namespace number_of_boys_in_first_group_l774_774690

-- Define the daily work done by a man and a boy
variables {M B x : ℝ}

-- Condition: The daily work done by a man is twice that of a boy
def daily_work_man_eq_twice_boy : Prop := M = 2 * B

-- Condition: Expression for the first group (12 men and x boys in 5 days)
def work_done_first_group : ℝ := 5 * (12 * M + x * B)

-- Condition: Expression for the second group (13 men and 24 boys in 4 days)
def work_done_second_group : ℝ := 4 * (13 * M + 24 * B)

-- Given conditions, we need to show that the number of boys in the first group is 16
theorem number_of_boys_in_first_group
    (h1 : daily_work_man_eq_twice_boy)
    (h2 : work_done_first_group = work_done_second_group) :
  x = 16 := by
  sorry

end number_of_boys_in_first_group_l774_774690


namespace perpendicular_lines_l774_774149

-- Definitions based on the conditions
def direction_vector_line1 (b : ℝ) : ℝ × ℝ × ℝ :=
  (b, -3, 2)

def direction_vector_line2 : ℝ × ℝ × ℝ :=
  (2, 4, 3)

-- Mathematically equivalent proof problem
theorem perpendicular_lines (b : ℝ) :
  let v1 := direction_vector_line1 b
  let v2 := direction_vector_line2
  (v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3 = 0) → b = 3 :=
by
  intros
  sorry

end perpendicular_lines_l774_774149


namespace quadratic_trinomial_has_two_roots_l774_774919

theorem quadratic_trinomial_has_two_roots
  (a b c : ℝ) (h : b^2 - 4 * a * c > 0) :
  4 * (a^2 - a * b + b^2 - 3 * a * c) > 0 :=
by
  sorry

end quadratic_trinomial_has_two_roots_l774_774919


namespace equalize_money_l774_774097

theorem equalize_money (
  Carmela_money : ℕ, Cousin_money : ℕ, num_cousins : ℕ) :
  Carmela_money = 7 → Cousin_money = 2 → num_cousins = 4 →
  ∀ (x : ℕ), Carmela_money - num_cousins * x = Cousin_money + x ∧
  (Carmela_money - num_cousins * x = Cousin_money + x) →
  x = 1 :=
by
  intros hCarmela_money hCousin_money hnum_cousins hx hfinal_eq
  sorry

end equalize_money_l774_774097


namespace natasha_average_climbing_speed_l774_774879

theorem natasha_average_climbing_speed :
  ∀ (D : ℝ),
  (∀ (V_avg_total : ℝ), V_avg_total = 3.5 → 
  ∀ (t_climb t_descend : ℝ), t_climb = 4 → t_descend = 2 → 
  ∀ (V_avg_climb : ℝ), V_avg_climb = D / t_climb → 
  ∀ (D_total : ℝ), D_total = 2 * D → 
  ∀ (t_total : ℝ), t_total = t_climb + t_descend →
  ∀ (V_avg_journey : ℝ), V_avg_journey = D_total / t_total →
  V_avg_journey = V_avg_total 
  → V_avg_climb = 2.625) :=
begin
  -- proof goes here
  sorry
end

end natasha_average_climbing_speed_l774_774879


namespace mary_visited_two_shops_l774_774779

-- Define the costs of items
def cost_shirt : ℝ := 13.04
def cost_jacket : ℝ := 12.27
def total_cost : ℝ := 25.31

-- Define the number of shops visited
def number_of_shops : ℕ := 2

-- Proof that Mary visited 2 shops given the conditions
theorem mary_visited_two_shops (h_shirt : cost_shirt = 13.04) (h_jacket : cost_jacket = 12.27) (h_total : cost_shirt + cost_jacket = total_cost) : number_of_shops = 2 :=
by
  sorry

end mary_visited_two_shops_l774_774779


namespace product_of_solutions_l774_774525

theorem product_of_solutions : 
  ∀ y : ℝ, (|y| = 3 * (|y| - 2)) → ∃ a b : ℝ, (a = 3 ∧ b = -3) ∧ (a * b = -9) := 
by 
  sorry

end product_of_solutions_l774_774525


namespace chocolate_factory_production_l774_774809

theorem chocolate_factory_production
  (candies_per_hour : ℕ)
  (total_candies : ℕ)
  (days : ℕ)
  (total_hours : ℕ := total_candies / candies_per_hour)
  (hours_per_day : ℕ := total_hours / days)
  (h1 : candies_per_hour = 50)
  (h2 : total_candies = 4000)
  (h3 : days = 8) :
  hours_per_day = 10 := by
  sorry

end chocolate_factory_production_l774_774809


namespace negation_of_divisible_by_2_even_l774_774354

theorem negation_of_divisible_by_2_even :
  (¬ ∀ n : ℤ, (∃ k, n = 2 * k) → (∃ k, n = 2 * k ∧ n % 2 = 0)) ↔
  ∃ n : ℤ, (∃ k, n = 2 * k) ∧ ¬ (n % 2 = 0) :=
by
  sorry

end negation_of_divisible_by_2_even_l774_774354


namespace mask_pack_duration_l774_774952

theorem mask_pack_duration:
  ∀ (masks : ℕ) (family_members : ℕ) (change_frequency : ℕ),
    masks = 100 →
    family_members = (1 + 2 + 2) →
    change_frequency = 4 →
    (masks / family_members) * change_frequency = 80 := 
by
  intros masks family_members change_frequency h_masks h_family_members h_change_frequency
  rw [h_masks, h_family_members, h_change_frequency]
  norm_num
  sorry

end mask_pack_duration_l774_774952


namespace division_theorem_l774_774765

theorem division_theorem (y : ℕ) (h : y ≠ 0) : (y - 1) ∣ (y^(y^2) - 2 * y^(y + 1) + 1) := 
by
  sorry

end division_theorem_l774_774765


namespace neg_ex_triangle_symm_l774_774358

theorem neg_ex_triangle_symm :
  ¬ ∃ (T : Type) [Triangle T], symmetric T ↔ ∀ (T : Type) [Triangle T], ¬ symmetric T :=
by
  sorry

end neg_ex_triangle_symm_l774_774358


namespace perfect_square_divisors_of_450_l774_774619

theorem perfect_square_divisors_of_450 :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let e1 := 1
  let e2 := 2
  let e3 := 2
  ∃ (count : ℕ), count = 4 ∧
  (∀ (d : ℕ), d ∣ (p1^e1 * p2^e2 * p3^e3) →
    (∀ (pe : ℕ × ℕ), pe ∈ (Multiset.ofList [(p1, e1), (p2, e2), (p3, e3)]).powerset → 
      (∃ (ex1 ex2 ex3: ℕ), ex1 ≤ e1 ∧ ex2 ≤ e2 ∧ ex3 ≤ e3 ∧ ex1 % 2 = 0 ∧ ex2 % 2 = 0 ∧ ex3 % 2 = 0))) :=
sorry

end perfect_square_divisors_of_450_l774_774619


namespace paint_faces_red_not_sum_twelve_l774_774087

-- Conditions
def die_faces : Finset ℕ := {1, 2, 3, 4, 5, 6}

-- Define the main problem statement
theorem paint_faces_red_not_sum_twelve
    (f : Finset ℕ → Finset (Finset ℕ)) :
    (f die_faces).card = 20 → 
    (f die_faces).filter (λ s, Finset.card s = 3 ∧ s.sum ≠ 12).card = 17 :=
by
  sorry

end paint_faces_red_not_sum_twelve_l774_774087


namespace sum_distances_l774_774888

noncomputable def ellipse_polar_eqn (ρ θ : ℝ) : Prop :=
  ρ^2 = 12 / (3 * (Real.cos θ)^2 + 4 * (Real.sin θ)^2)

def parametric_eqn_line (x y t : ℝ) : Prop :=
  x = 2 + (Real.sqrt 2)/2 * t ∧ y = (Real.sqrt 2)/2 * t

def foci (F1 F2 : ℝ × ℝ) : Prop :=
  F1 = (-1, 0) ∧ F2 = (1, 0)

def distance (p : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  abs (a * p.1 + b * p.2 + c) / Real.sqrt (a^2 + b^2)

theorem sum_distances (d1 d2 : ℝ) :
  (∀ ρ θ x y t F1 F2,
    ellipse_polar_eqn ρ θ ∧
    parametric_eqn_line x y t ∧
    foci F1 F2 →
    d1 = distance F1 1 (-1) (-2) ∧
    d2 = distance F2 1 (-1) (-2) ∧
    d1 + d2 = 2 * Real.sqrt 2) := sorry

end sum_distances_l774_774888


namespace perfect_square_factors_450_l774_774627

theorem perfect_square_factors_450 : 
  ∃ n, n = 4 ∧ (∀ k | 450, k ∣ 450 → is_square k) → (∃ m1 m2 m3 : ℕ, 450 = 2^m1 * 3^m2 * 5^m3 ) :=
by sorry

end perfect_square_factors_450_l774_774627


namespace problem_I_problem_II_l774_774559

-- Define the sequence {a_n} satisfying the given condition
def seq_cond (a : ℕ → ℚ) : Prop :=
  ∀ n, n > 0 → (∑ i in Finset.range n + 1, (2 * (i + 1) - 1) * a (i + 1)) = n

-- Define the general formula for the sequence
def gen_formula (a : ℕ → ℚ) : Prop :=
  ∀ n, n > 0 → a n = (1 : ℚ) / (2 * n - 1)

-- Define the sum S_n of the first n terms of the sequence {a_n a_{n+1}}
def sum_sn (a : ℕ → ℚ) (S : ℕ → ℚ) : Prop :=
  ∀ n, n > 0 → S n = (∑ i in Finset.range n , a (i + 1) * a (i + 2))

-- Problem I: Prove the general formula for {a_n}
theorem problem_I (a : ℕ → ℚ) (h : seq_cond a) : gen_formula a :=
sorry

-- Problem II: Prove the sum S_n of the first n terms of the sequence {a_n * a_{n+1}}
theorem problem_II (a : ℕ → ℚ) (S : ℕ → ℚ) (h_gen : gen_formula a) : sum_sn_sum a S :=
sorry

end problem_I_problem_II_l774_774559


namespace cone_height_20_l774_774054

def is_isosceles_right_triangle (r : ℝ) (h : ℝ) : Prop :=
  h = r

def cone_volume (r : ℝ) (h : ℝ) : ℝ :=
  (1 / 3) * π * r^2 * h

theorem cone_height_20 (r h : ℝ)
    (h_rt_triangle : is_isosceles_right_triangle r h)
    (h_volume : cone_volume r h = 2592 * π) :
    h = 20 :=
by
  sorry

end cone_height_20_l774_774054


namespace limit_P_n_eq_1_div_e_sq_l774_774293

noncomputable def P_n (n : ℕ) : ℝ :=
  let prob_not_in_any := 2^(-n : ℝ)
  let prob_in_all := 2^(-n : ℝ)
  let prob_in_at_least_one_but_not_all := 1 - 2 * (2^(-n : ℝ))
  prob_in_at_least_one_but_not_all^(2^n)

theorem limit_P_n_eq_1_div_e_sq :
  ∀ (S : Finset ℕ), (∀ n > 0, S.card = 2^n) →
  let A := λ (n : ℕ), Finset.powerset S
  (∀ (A_i : ℕ → Finset ℕ), (∀ i, i < n → A_i i ∈ A n)) →
  ∀ n, P_n n = (1 - 2^(1 - n))^(2^n) →
  filter.tendsto (λ n, P_n n) filter.at_top (nhds (1 / ℘(2)))
 := 
sorry

end limit_P_n_eq_1_div_e_sq_l774_774293


namespace rachel_wrote_six_pages_l774_774324

theorem rachel_wrote_six_pages
  (write_rate : ℕ)
  (research_time : ℕ)
  (editing_time : ℕ)
  (total_time : ℕ)
  (total_time_in_minutes : ℕ := total_time * 60)
  (actual_time_writing : ℕ := total_time_in_minutes - (research_time + editing_time))
  (pages_written : ℕ := actual_time_writing / write_rate) :
  write_rate = 30 →
  research_time = 45 →
  editing_time = 75 →
  total_time = 5 →
  pages_written = 6 :=
by
  intros h1 h2 h3 h4
  subst h1
  subst h2
  subst h3
  subst h4
  have h5 : total_time_in_minutes = 300 := by sorry
  have h6 : actual_time_writing = 180 := by sorry
  have h7 : pages_written = 6 := by sorry
  exact h7

end rachel_wrote_six_pages_l774_774324


namespace cot_value_l774_774152

def a (n : ℕ) := 1 + n + n^2

lemma arccot_eq_arctan (n : ℕ) : Real.arccot (a n) = Real.arctan (n+1) - Real.arctan n := sorry

lemma theta_def : 
  let θ := Real.arccot 3 + Real.arccot 7 + Real.arccot 13 + Real.arccot 21 in
  θ = Real.arctan 5 - Real.arctan 1 := sorry

theorem cot_value :
  let θ := Real.arccot 3 + Real.arccot 7 + Real.arccot 13 + Real.arccot 21 in
  10 * Real.cot θ = 15 := sorry

end cot_value_l774_774152


namespace sequence_common_difference_l774_774591

theorem sequence_common_difference (a : ℕ → ℤ) (h : ∀ n, a n = 2 * n + 5) : 
  ∀ n, a (n + 1) - a n = 2 := 
by
  intro n
  calc
    a (n + 1) = 2 * (n + 1) + 5 := h (n + 1)
    ... = 2 * n + 2 + 5 := by ring
    ... = 2 * n + 7 := by ring
    ... - (a n) 
    ... = (2 * n + 7) - (2 * n + 5) := by rw h n
    ... = 2 := by ring
  sorry

end sequence_common_difference_l774_774591


namespace exists_non_union_sets_l774_774549

theorem exists_non_union_sets (n : ℕ) (h : n ≥ 5) :
  ∃ (r : ℕ) (A : finset (finset ℕ)), r = ⌊real.sqrt (2 * n)⌋ ∧ A.card = r ∧ ∀ (i j k : fin A.card), i ≠ j ∧ i ≠ k ∧ j ≠ k → ¬ (A.elems i = A.elems j ∪ A.elems k) :=
sorry

end exists_non_union_sets_l774_774549


namespace triangle_side_length_l774_774262

theorem triangle_side_length (BC : ℝ) (A : ℝ) (B : ℝ) (AB : ℝ) :
  BC = 2 → A = π / 3 → B = π / 4 → AB = (3 * Real.sqrt 2 + Real.sqrt 6) / 3 :=
by
  sorry

end triangle_side_length_l774_774262


namespace margin_in_terms_of_ratio_l774_774227

variable (S m : ℝ)

theorem margin_in_terms_of_ratio (h1 : M = (1/m) * S) (h2 : C = S - M) : M = (1/m) * S :=
sorry

end margin_in_terms_of_ratio_l774_774227


namespace length_of_CD_l774_774368

namespace TetrahedronProof

-- Define the edges of the tetrahedron
def edges : Set ℕ := {7, 13, 18, 27, 36, 41}

-- Define the edge between A and B
def AB : ℕ := 41

-- Define a predicate for possible tetrahedron configurations satisfying the triangle inequality
def satisfies_triangle_inequality (a b c : ℕ) : Prop := a + b > c ∧ a + c > b ∧ b + c > a 

-- State the theorem which needs proof
theorem length_of_CD (h : (subset (insert AB edges)) : edges) :
  ∃ CD, CD = 13 ∧
         (∀ (A B : ℕ), A ∈ edges ∧ B ∈ edges ∧ A ≠ B → satisfies_triangle_inequality AB A B) ∧
         (∀ (C D : ℕ), C ∈ edges ∧ D ∈ edges ∧ C ≠ D → satisfies_triangle_inequality CD C D)
:=
sorry -- Proof required

end TetrahedronProof

end length_of_CD_l774_774368


namespace equilateral_trapezoids_same_perimeter_l774_774390

theorem equilateral_trapezoids_same_perimeter (AB DE FG : ℝ) 
  (h1 : AB = 1) 
  (h2 : DE + FG = 21 / 13)
  (E G : Point ℝ) : DE + FG = 21 / 13 :=
sorry

end equilateral_trapezoids_same_perimeter_l774_774390


namespace rectangle_area_l774_774925

theorem rectangle_area
  (line : ∀ x, 6 = x * x + 4 * x + 3 → x = -2 + Real.sqrt 7 ∨ x = -2 - Real.sqrt 7)
  (shorter_side : ∃ l, l = 2 * Real.sqrt 7 ∧ ∃ s, s = l + 3) :
  ∃ a, a = 28 + 12 * Real.sqrt 7 :=
by
  sorry

end rectangle_area_l774_774925


namespace numbers_divisible_by_12_in_1_to_100_l774_774945

theorem numbers_divisible_by_12_in_1_to_100 :
  (∃ n : ℕ, n = finset.card (finset.filter (λ x, x % 12 = 0) (finset.range 101))) → n = 8 :=
sorry

end numbers_divisible_by_12_in_1_to_100_l774_774945


namespace marbles_leftover_l774_774797

theorem marbles_leftover (r p : ℕ) (hr : r % 8 = 5) (hp : p % 8 = 7) : (r + p) % 8 = 4 :=
by
  sorry

end marbles_leftover_l774_774797


namespace number_of_spacy_subsets_S8_l774_774112

def isSpacy (s : Set ℕ) : Prop :=
  ∀ ⦃a b c : ℕ⦄, a < b → b < c → a ∈ s → b ∈ s → c ∈ s → false

def S_n := { x : ℕ | 1 ≤ x ∧ x ≤ 8 }
def countSpacySubsets (n : ℕ) : ℕ :=
  if n = 1 then 2 else
  if n = 2 then 3 else
  if n = 3 then 4 else
  countSpacySubsets (n - 3) + countSpacySubsets (n - 1)

theorem number_of_spacy_subsets_S8 : countSpacySubsets 8 = 28 := sorry

end number_of_spacy_subsets_S8_l774_774112


namespace hank_mowing_income_l774_774206

-- Define constants related to the problem
def carwash_income := 100
def bake_sale_income := 80
def total_donations := 200

-- Define the percentage donations
def carwash_donation_percentage := 0.90
def bake_sale_donation_percentage := 0.75

-- Calculate the donations from the carwash and bake sale
def carwash_donations := carwash_donation_percentage * carwash_income
def bake_sale_donations := bake_sale_donation_percentage * bake_sale_income

-- Define total donations excluding the mowing lawns
def donations_excluding_mowing := carwash_donations + bake_sale_donations

-- Define the condition for money made from mowing lawns
def money_from_mowing := total_donations - donations_excluding_mowing

-- Prove that the money Hank made from mowing lawns is $50
theorem hank_mowing_income : money_from_mowing = 50 := 
by
  rw [money_from_mowing, donations_excluding_mowing, carwash_donations, bake_sale_donations]
  simp only [carwash_donation_percentage, bake_sale_donation_percentage, carwash_income, bake_sale_income, total_donations]
  norm_num
  sorry

end hank_mowing_income_l774_774206


namespace shoe_size_15_length_l774_774038

-- Definitions according to conditions
def length_of_smallest_shoe (L : ℝ) : Prop :=
  L + (17 - 8) * (1 / 5) = 1.4 * L

def length_of_shoe_in_size (L : ℝ) (s : ℕ) : ℝ :=
  L + (s - 8) * (1 / 5)

-- Theorem to be proved
theorem shoe_size_15_length (L : ℝ) (h : length_of_smallest_shoe L) :
  length_of_shoe_in_size L 15 = 5.9 :=
by sorry

end shoe_size_15_length_l774_774038


namespace perfect_square_factors_450_l774_774612

theorem perfect_square_factors_450 :
  ∃ (n : ℕ), n = 4 ∧ ∀ (d : ℕ), d ∣ 450 → ∃ (k : ℕ), k^2 = d ↔ d = 1 ∨ d = 9 ∨ d = 25 ∨ d = 225 :=
by
  sorry

end perfect_square_factors_450_l774_774612


namespace minimum_black_edges_l774_774122

-- Definitions representing the conditions
def Cube (faces edges : ℕ) := faces = 6 ∧ edges = 12
def colored (edges : ℕ) := edges = 12 ∧ ∀ (e : ℕ), e ≤ edges → e = 0 ∨ e = 1
def face_edges (face_edges : ℕ) := face_edges = 4 ∧ ∀ (faces : ℕ), faces ≤ 6 → faces * face_edges = 6 * 4 

-- The number of black edges on each face is exactly 2
def black_edges_per_face (black_edges_per_face : ℕ) := black_edges_per_face = 2 ∧ ∀ (faces : ℕ), faces ≤ 6 ∧ faces * black_edges_per_face = 6 * 2

-- Prove that the minimum number of black edges given the conditions above is 8
theorem minimum_black_edges (faces edges black_edges_per_face : ℕ) [Cube faces edges] [colored edges] [face_edges face_edges] [black_edges_per_face black_edges_per_face] : 
  8 ≤ faces * edges / face_edges := 
by
  sorry

end minimum_black_edges_l774_774122


namespace min_ratio_number_l774_774040

theorem min_ratio_number (H T U : ℕ) (h1 : H - T = 8 ∨ T - H = 8) (hH : 1 ≤ H ∧ H ≤ 9) (hT : 0 ≤ T ∧ T ≤ 9) (hU : 0 ≤ U ∧ U ≤ 9) :
  100 * H + 10 * T + U = 190 :=
by sorry

end min_ratio_number_l774_774040


namespace ab_cd_not_prime_l774_774768

theorem ab_cd_not_prime (a b c d : ℕ) (h1 : a > b) (h2 : b > c) (h3 : c > d) (h4 : d > 0)
                        (h5 : ac + bd = (b + d - a + c) * (b + d + a - c)) : ¬ Prime (ab + cd) := by
  sorry

end ab_cd_not_prime_l774_774768


namespace smallest_degree_of_polynomial_l774_774458

-- Definition of a polynomial with rational coefficients having specified roots
noncomputable def polynomial_with_roots : Polynomial ℚ :=
  Polynomial.prod (Finset.range 500).image (λ n, X - C (n + 1 + real.sqrt (n + 2)))

-- Main theorem stating the required degree of the polynomial
theorem smallest_degree_of_polynomial : 
  (polynomial_with_roots.degree.to_nat >= 979) :=
begin
  -- Proof steps would be provided here, but are omitted according to the instructions.
  sorry
end

end smallest_degree_of_polynomial_l774_774458


namespace total_worth_of_travelers_checks_l774_774943

theorem total_worth_of_travelers_checks (x y : ℕ) (h1 : x + y = 30) (h2 : 50 * (x - 18) + 100 * y = 900) : 
  50 * x + 100 * y = 1800 := 
by
  sorry

end total_worth_of_travelers_checks_l774_774943


namespace find_ellipse_equation_area_triangle_l774_774193

def ellipse_condition_1 (a b c : ℝ) : Prop := c / a = Real.sqrt 2 / 2
def ellipse_condition_2 (a b c : ℝ) : Prop := a^2 = b^2 + c^2
def ellipse_condition_3 (a b c : ℝ) : Prop := 1 / b^2 + 2 / a^2 = 1

noncomputable def point_A_on_ellipse (a b : ℝ) : Prop := (1 / a^2) + (sqrt 2 / b^2) = 1

def line_slope_condition (slope : ℝ) : Prop := slope = Real.sqrt 2

theorem find_ellipse_equation_area_triangle
  (a b c : ℝ) (slope : ℝ)
  (h1 : ellipse_condition_1 a b c)
  (h2 : ellipse_condition_2 a b c)
  (h3 : ellipse_condition_3 a b c)
  (h4 : point_A_on_ellipse a b)
  (h_slope : line_slope_condition slope) :
  (a = 2 ∧ b = Real.sqrt 2 ∧ c = Real.sqrt 2 ∧ ellipse_eq : Real := (fun x y => x^2 / 2 + y^2 / 4 = 1))
  ∧ (max_area : Real = Real.sqrt 2) :=
sorry

end find_ellipse_equation_area_triangle_l774_774193


namespace sum_of_three_numbers_is_zero_l774_774821

noncomputable def mean (a b c : ℝ) : ℝ := (a + b + c) / 3

theorem sum_of_three_numbers_is_zero
  (a b c : ℝ) 
  (h1 : a ≤ b ∧ b ≤ c) 
  (h2 : median := b)
  (h3 : mean a b c = a + 20) 
  (h4 : mean a b c = c - 10)
  (h5 : b = 10) :
  a + b + c = 0 :=
by 
  sorry

end sum_of_three_numbers_is_zero_l774_774821


namespace equal_segments_on_circle_l774_774430

theorem equal_segments_on_circle
  (A M B C D : Point)
  (k : Circle)
  (points_on_k : ∀ P ∈ {A, M, B, C, D}, P ∈ k)
  (order_on_k : ∃ P Q R S T ∈ k, (P = A ∧ Q = M ∧ R = B ∧ S = C ∧ T = D))
  (MA_eq_MB : segment_length M A = segment_length M B)
  (P : Point)
  (intersect_AC_MD : P ∈ line_through A C ∩ line_through M D)
  (Q : Point)
  (intersect_BD_MC : Q ∈ line_through B D ∩ line_through M C)
  (X Y : Point)
  (intersect_PQ_k : ∃ X Y ∈ k, X ≠ Y ∧ X ≠ P ∧ Y ≠ Q ∧ X Y ∈ line_through P Q) :
  segment_length M X = segment_length M Y := by
  sorry

end equal_segments_on_circle_l774_774430


namespace equal_segments_l774_774573

theorem equal_segments {O : Type*} [plane : EuclideanPlane O] 
  {M N R : O} (h_midpoint : Midpoint R M N)
  {A B C D : O} (h_chords : ChordsIntersect R A B C D)
  {Γ : ConicSection O} (h_conic_pass : PassesThrough Γ A ∧ PassesThrough Γ B ∧ PassesThrough Γ C ∧ PassesThrough Γ D)
  {P Q : O} (h_intersect : Intersects Γ P Q MN) :
  |PR| = |RQ| :=
sorry

end equal_segments_l774_774573


namespace directrix_of_parabola_l774_774519

theorem directrix_of_parabola :
  ∀ (x y : ℝ), y = (x^2 - 4 * x + 3) / 8 → y = -9 / 8 :=
by
  sorry

end directrix_of_parabola_l774_774519


namespace perfect_square_factors_450_l774_774631

theorem perfect_square_factors_450 : 
  ∃ n, n = 4 ∧ (∀ k | 450, k ∣ 450 → is_square k) → (∃ m1 m2 m3 : ℕ, 450 = 2^m1 * 3^m2 * 5^m3 ) :=
by sorry

end perfect_square_factors_450_l774_774631


namespace famous_quote_author_l774_774407

-- conditions
def statement_date := "July 20, 1969"
def mission := "Apollo 11"
def astronauts := ["Neil Armstrong", "Buzz Aldrin", "Michael Collins"]
def first_to_moon := "Neil Armstrong"

-- goal
theorem famous_quote_author : (statement_date = "July 20, 1969") ∧ (mission = "Apollo 11") ∧ (astronauts = ["Neil Armstrong", "Buzz Aldrin", "Michael Collins"]) ∧ (first_to_moon = "Neil Armstrong") → "Neil Armstrong" = "Neil Armstrong" :=
by 
  intros _; 
  exact rfl

end famous_quote_author_l774_774407


namespace relationship_among_a_b_c_l774_774547

noncomputable def a : ℝ := Real.log 3 / Real.log 0.5
noncomputable def b : ℝ := Real.sqrt 2
noncomputable def c : ℝ := 0.5 ^ 0.3

theorem relationship_among_a_b_c : b > c ∧ c > a :=
by
  sorry

end relationship_among_a_b_c_l774_774547


namespace min_three_beverages_overlap_l774_774782

variable (a b c d : ℝ)
variable (ha : a = 0.9)
variable (hb : b = 0.8)
variable (hc : c = 0.7)

theorem min_three_beverages_overlap : d = 0.7 :=
by
  sorry

end min_three_beverages_overlap_l774_774782


namespace circles_tangent_internally_l774_774830

def radius_O1 := 3 -- radius of circle O1 in cm
def radius_O2 := 5 -- radius of circle O2 in cm
def distance_centers := 2 -- distance between centers of circle O1 and O2 in cm

theorem circles_tangent_internally :
  abs (radius_O1 - radius_O2) = distance_centers →
  true := 
by
  sorry

end circles_tangent_internally_l774_774830


namespace quadratic_has_solutions_l774_774115

theorem quadratic_has_solutions (k : ℝ)
  (h1 : ∀ a b c : ℝ, k ≠ 0 → a * -2^2 + b * -2 + c = 0 → a * (5/2)^2 + b * (5/2) + c = 0)
  (h2 : ∀ a b c : ℝ, k ≠ 0 → b^2 - 4 * a * c > 0) :
  k = 2 :=
begin
  sorry
end

end quadratic_has_solutions_l774_774115


namespace point_equidistant_to_quadrilateral_vertices_l774_774459

theorem point_equidistant_to_quadrilateral_vertices (A B C D P : Point) (hconvex : ConvexQuadrilateral A B C D) 
(hdist : dist P A = dist P B ∧ dist P B = dist P C ∧ dist P C = dist P D) :
  (InsideQuadrilateral P A B C D ∨ OnEdgeQuadrilateral P A B C D ∨ OutsideQuadrilateral P A B C D) :=
  by
    sorry

end point_equidistant_to_quadrilateral_vertices_l774_774459


namespace max_value_expression_l774_774224

variables {V : Type} [inner_product_space ℝ V]

def unit_vector (v : V) : Prop := ⟪v, v⟫ = 1

theorem max_value_expression (a b c : V) 
  (ha : unit_vector a) 
  (hb : unit_vector b) 
  (hc : unit_vector c) 
  (orth_ab : ⟪a, b⟫ = 0) : 
  ∃ x : ℝ, x = 8 ∧ 
  (‖a - b‖^2 + ‖a - c‖^2 + ‖b - c‖^2 ≤ x) :=
sorry

end max_value_expression_l774_774224


namespace min_colors_needed_l774_774999

-- Define the problem in Lean
theorem min_colors_needed (n : ℕ) :
  ∃ colors : ℕ, colors = n ∧ (∀ (s t : ℕ), s ∣ ((n - 24)! ) ∧ t ∣ ((n - 24)!) ∧ s ≠ t → (∃ color_s color_t : ℕ, (color_s ≠ color_t ∧ color_s ∣ s ∧ color_t ∣ t))) ↔ n = 50 := 
sorry

end min_colors_needed_l774_774999


namespace above_198_eq_170_l774_774248

/--
In a triangular-shaped sequence of numbers, where the 
first row has 1 number, the second row has 3 numbers, the 
third row has 5 numbers, and so forth, prove that the 
number directly above 198 is 170.
-/
theorem above_198_eq_170 : ∃ x, x = 170 ∧ 
  (∃ n, ∃ m, n > 0 ∧ m = 198 ∧ 
    (∀ k, (∑ i in range (k + 1), (2 * i + 1)) = k^2) ∧ 
    (2 * n - 1 ≥ m) ∧ 
    let s := n - 1 in
    let first_num_in_prev_row := s^2 + 1 in
    first_num_in_prev_row + (∃ pos, pos > 0 ∧ pos = (m - (s^2 + 1) + 1)) = x) :=
sorry

end above_198_eq_170_l774_774248


namespace divides_expression_l774_774763

theorem divides_expression (y : ℕ) (h : y ≠ 0) : (y - 1) ∣ (y ^ (y ^ 2) - 2 * y ^ (y + 1) + 1) :=
sorry

end divides_expression_l774_774763


namespace solitaire_game_removal_condition_l774_774079

theorem solitaire_game_removal_condition (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  (∃ (remove_all_markers : bool), remove_all_markers = true) ↔ (m % 2 = 1 ∨ n % 2 = 1) := sorry

end solitaire_game_removal_condition_l774_774079


namespace line_circle_intersection_perp_l774_774576

theorem line_circle_intersection_perp {a : ℝ} 
    (h : ∀ x y : ℝ, x^2 + y^2 + 2*x - 4*y - 4 = 0 → x - y + a = 0) 
    (AC_perp_BC : ∀ A B C : ℝ × ℝ, A ≠ B ∧ AC ⊥ BC) :
    a = 0 ∨ a = 6 :=
sorry

end line_circle_intersection_perp_l774_774576


namespace inf_nats_not_sum_of_n_powers_l774_774771

open Nat

theorem inf_nats_not_sum_of_n_powers (n : ℕ) (n_gt_one : n > 1) : 
  ∃^∞ m : ℕ, ¬ ∃ (a : Fin n → ℕ), (∑ i, (a i) ^ n) = m :=
sorry

end inf_nats_not_sum_of_n_powers_l774_774771


namespace division_theorem_l774_774764

theorem division_theorem (y : ℕ) (h : y ≠ 0) : (y - 1) ∣ (y^(y^2) - 2 * y^(y + 1) + 1) := 
by
  sorry

end division_theorem_l774_774764


namespace find_y_l774_774231

theorem find_y (x y : ℝ) (h1 : x = 4 * y) (h2 : (1 / 2) * x = 1) : y = 1 / 2 :=
by
  sorry

end find_y_l774_774231


namespace integer_solutions_abs_lt_4_sqrt_2_l774_774603

theorem integer_solutions_abs_lt_4_sqrt_2 : 
  {x : ℤ | abs x < 4 * real.sqrt 2}.finite ∧ 
  {x : ℤ | abs x < 4 * real.sqrt 2}.to_finset.card = 11 :=
by sorry

end integer_solutions_abs_lt_4_sqrt_2_l774_774603


namespace twisty_number_count_l774_774066

-- Definition of twisty number form
def is_twisty (n : ℕ) : Prop :=
  let digits := (nat.digits 10 n) in
  n < 100000 ∧
  n ≥ 10000 ∧
  digits.head! = digits.get! 4 ∧
  digits.get! 1 ≠ digits.head! ∧
  (∀ i, i < 4 → i % 2 = 0 → digits.get! i = digits.head!) ∧
  (∀ i, i < 4 → i % 2 = 1 → digits.get! i = digits.get! 1)

-- Divisibility by 4 and 3
def divisible_by_12 (n : ℕ) : Prop :=
  n % 4 = 0 ∧ (3 * ((nat.digits 10 n).head!) + 2 * ((nat.digits 10 n).get! 1)) % 3 = 0

-- Combining twisty and divisibility conditions
def twisty_div_by_12 (n : ℕ) : Prop :=
  is_twisty n ∧ divisible_by_12 n

-- Counting the number of such five-digit twisty numbers
def count_twisty_div_by_12 : ℕ :=
  (finset.filter twisty_div_by_12 (finset.range 100000)).card

theorem twisty_number_count : count_twisty_div_by_12 = 1 := 
by {
  sorry -- Proof to be provided
}

end twisty_number_count_l774_774066


namespace solve_equation_l774_774331

theorem solve_equation (x : ℝ) (h : (x^2 - x + 2) / (x - 1) = x + 3) (h1 : x ≠ 1) : 
  x = 5 / 3 :=
sorry

end solve_equation_l774_774331


namespace increasing_interval_f_l774_774983

noncomputable def f (x : ℝ) : ℝ := 1 / (x * Real.log x)

theorem increasing_interval_f : ∀ x : ℝ, 0 < x ∧ x < 1 / Real.exp 1 → 0 < f x → f'_x > 0 := 
begin
  sorry
end

end increasing_interval_f_l774_774983


namespace perfect_square_factors_450_l774_774651

def prime_factorization (n : ℕ) : Prop :=
  n = 2^1 * 3^2 * 5^2

theorem perfect_square_factors_450 : prime_factorization 450 → ∃ n : ℕ, n = 4 :=
by
  sorry

end perfect_square_factors_450_l774_774651


namespace range_of_a_l774_774226

theorem range_of_a (a : ℝ) :
  (abs (15 - 3 * a) / 5 ≤ 3) → (0 ≤ a ∧ a ≤ 10) :=
by
  intro h
  sorry

end range_of_a_l774_774226


namespace geometric_sequence_first_term_l774_774836

theorem geometric_sequence_first_term (a r : ℚ) (third_term fourth_term : ℚ) 
  (h1 : third_term = a * r^2)
  (h2 : fourth_term = a * r^3)
  (h3 : third_term = 27)
  (h4 : fourth_term = 36) : 
  a = 243 / 16 :=
by
  sorry

end geometric_sequence_first_term_l774_774836


namespace ccl4_amount_l774_774496

def step1 (c2h6 cl2 : ℕ) : ℕ × ℕ := 
  if c2h6 ≤ cl2 then (c2h6, cl2 - c2h6) else (cl2, 0)

def step2 (ch3cl cl2 : ℕ) : ℕ × ℕ := 
  if ch3cl ≤ cl2 then (ch3cl, cl2 - ch3cl) else (cl2, 0)

def step3 (ch2cl2 cl2 : ℕ) : ℕ × ℕ := 
  if ch2cl2 ≤ cl2 then (ch2cl2, cl2 - ch2cl2) else (cl2, 0)

def step4 (chcl3 cl2 : ℕ) : ℕ × ℕ := 
  if chcl3 ≤ cl2 then (chcl3, cl2 - chcl3) else (cl2, 0)

theorem ccl4_amount (C2H6 Cl2 : ℕ) (h1 : C2H6 = 2) (h2 : Cl2 = 14) : 
  let (ch3cl, cl2_step1) := step1 C2H6 Cl2 in
  let (ch2cl2, cl2_step2) := step2 ch3cl cl2_step1 in
  let (chcl3, cl2_step3) := step3 ch2cl2 cl2_step2 in
  let (ccl4, cl2_step4) := step4 chcl3 cl2_step3 in
  ccl4 = 2 :=
by {
  sorry
}

end ccl4_amount_l774_774496


namespace find_abc_l774_774769

theorem find_abc
  {a b c : ℤ} (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : a + b + c = 30)
  (h2 : 1/a + 1/b + 1/c + 672/(a*b*c) = 1) :
  a * b * c = 2808 :=
sorry

end find_abc_l774_774769


namespace angle_of_inclination_l774_774505

theorem angle_of_inclination (x y : ℝ) : 
  (2*x + y - 1 = 0) → (∃ α : ℝ, tan α = -2 ∧ 0 ≤ α ∧ α ≤ π ∧ α = π - arctan 2) := 
by
  intro h
  use π - arctan 2
  sorry

end angle_of_inclination_l774_774505


namespace P_equals_one_l774_774884

-- Definition of the problem expression P
def P (α : ℝ) : ℝ := 
  (1 / 2) * ((1 - cos (2 * α)) / (cos α ^ (-2) - 1) + (1 + cos (2 * α)) / (sin α ^ (-2) - 1)) 
  + ((cos (2 * α)) / (sin (2 * α)) + cos 2 α + sin 2 α)

-- Statement that asserts P equals 1 for all real α
theorem P_equals_one (α : ℝ) : P α = 1 := by
  sorry

end P_equals_one_l774_774884


namespace symmetric_points_l774_774693

theorem symmetric_points (a b : ℤ) (h1 : (a, -2) = (1, -2)) (h2 : (-1, b) = (-1, -2)) :
  (a + b) ^ 2023 = -1 := by
  -- We know from the conditions:
  -- (a, -2) and (1, -2) implies a = 1
  -- (-1, b) and (-1, -2) implies b = -2
  -- Thus it follows that:
  sorry

end symmetric_points_l774_774693


namespace product_fraction_equals_1_div_501_l774_774484

theorem product_fraction_equals_1_div_501 :
  (∏ n in Finset.range 500, (2 * (n + 1)) / (2 * (n + 1) + 2)) = 1 / 501 :=
by
  sorry

end product_fraction_equals_1_div_501_l774_774484


namespace perfect_square_divisors_of_450_l774_774618

theorem perfect_square_divisors_of_450 :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let e1 := 1
  let e2 := 2
  let e3 := 2
  ∃ (count : ℕ), count = 4 ∧
  (∀ (d : ℕ), d ∣ (p1^e1 * p2^e2 * p3^e3) →
    (∀ (pe : ℕ × ℕ), pe ∈ (Multiset.ofList [(p1, e1), (p2, e2), (p3, e3)]).powerset → 
      (∃ (ex1 ex2 ex3: ℕ), ex1 ≤ e1 ∧ ex2 ≤ e2 ∧ ex3 ≤ e3 ∧ ex1 % 2 = 0 ∧ ex2 % 2 = 0 ∧ ex3 % 2 = 0))) :=
sorry

end perfect_square_divisors_of_450_l774_774618


namespace sum_of_odd_integers_in_range_l774_774024

theorem sum_of_odd_integers_in_range :
  let lower_bound := -15.7
  let upper_bound := 12.6
  ∑ i in finset.filter (λ x : ℤ, x % 2 ≠ 0) (finset.Icc ⌈lower_bound⌉ ⌊upper_bound⌋) = -28 :=
by
  sorry

end sum_of_odd_integers_in_range_l774_774024


namespace a_2010_l774_774502

noncomputable def f : ℝ → ℝ := sorry
def a (n : ℕ) := f n

axiom f_recurrence (x : ℝ) : f (x + 1) = f x + 1
axiom f_initial : f 1 = 2

theorem a_2010 : a 2010 = 2011 := sorry

end a_2010_l774_774502


namespace gain_of_B_approximation_l774_774222

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem gain_of_B_approximation :
  let P := 3200
  let r1 := 0.12
  let t1 := 3
  let r2 := 0.145
  let t2 := 5
  B_gain := compound_interest P r2 1 t2 - compound_interest P r1 1 t1
  abs (B_gain - 1940.57) < 0.01 := sorry

end gain_of_B_approximation_l774_774222


namespace find_a3_l774_774564

def sequence_sum (n : ℕ) : ℕ := n^2 + n

def a (n : ℕ) : ℕ := sequence_sum n - sequence_sum (n - 1)

theorem find_a3 : a 3 = 6 := by
  sorry

end find_a3_l774_774564


namespace number_of_towers_remainder_l774_774447

def edge_lengths := {k : ℕ | 1 ≤ k ∧ k ≤ 8}

def is_valid_tower (tower : List ℕ) : Prop :=
  ∀ i, i < tower.length - 1 → tower.nth i.succ ≤ (tower.nth i + 2)

def number_of_towers := 
  {towers : List (List ℕ) | 
     ∀ tower ∈ towers, 
     (∀ cube ∈ tower, cube ∈ edge_lengths) ∧ 
      is_valid_tower tower}

/-- Given the conditions on edge lengths and the validity of towers, 
    the remainder of the number of different towers (T) when divided 
    by 1000 is 458. -/
theorem number_of_towers_remainder : 
  let T := number_of_towers.size in
  T % 1000 = 458 := 
sorry

end number_of_towers_remainder_l774_774447


namespace incorrect_proof_statement_l774_774253

theorem incorrect_proof_statement (A B C : Prop) (H1 : AxiomsOrPostulatesDoNotRequireProof : Prop) (H2 : InMathematicalProofsStartWithUnprovenAssumptions : Prop)
    (H3 : ProofCanConcludeWithTrueStatementFromIncorrectAssumption : Prop) (H4 : OrderOfPropositionsInfluencesValidity : Prop)
    (H5 : ProofByContradiction : Prop) :
  (A → True) → (B → True) → (C → False) → (D → False) → (E → True) → OrderOfPropositionsInfluencesValidity = False :=
by
  sorry

end incorrect_proof_statement_l774_774253


namespace part1_part2_l774_774197

noncomputable def f (x : ℝ) (a : ℝ) := Real.exp (x - 1) - (4 * a - 3) / (6 * x)
noncomputable def g (x : ℝ) (a : ℝ) := (1 / 3) * a * x^2 + (1 / 2) * x - (a - 1)

theorem part1 (a : ℝ) :
  let f_prime (x : ℝ) := Real.exp (x - 1) + (4 * a - 3) / (6 * x^2)
  in f_prime 1 * (-1 / 2) = -1 -> a = 9 / 4 := 
by {
  intro h,
  have : Real.exp 0 + (4 * a - 3) / 6 = 1,
  simp [h], sorry
}

theorem part2 (a : ℝ) :
  (∀ x : ℝ, x ≥ 1 -> f x a ≥ g x a) -> a ≤ 1 := 
by {
  intro h,
  sorry
}

end part1_part2_l774_774197


namespace cos_sin_cos_min_value_cos_sin_cos_max_value_l774_774165

theorem cos_sin_cos_min_value (x y z : ℝ)
  (h1 : x ≥ y)
  (h2 : y ≥ z)
  (h3 : z ≥ π / 12)
  (h4 : x + y + z = π / 2) :
  cos x * sin y * cos z ≥ 1 / 8 := sorry

theorem cos_sin_cos_max_value (x y z : ℝ)
  (h1 : x ≥ y)
  (h2 : y ≥ z)
  (h3 : z ≥ π / 12)
  (h4 : x + y + z = π / 2) :
  cos x * sin y * cos z ≤ (2 + real.sqrt 3) / 8 := sorry

end cos_sin_cos_min_value_cos_sin_cos_max_value_l774_774165


namespace not_perfect_square_7_301_l774_774031

theorem not_perfect_square_7_301 :
  ¬ ∃ x : ℝ, x^2 = 7^301 := sorry

end not_perfect_square_7_301_l774_774031


namespace conjugate_z_in_second_quadrant_l774_774192

def z : ℂ := 2 / (complex.i - 1)

def conjugate_z : ℂ := complex.conj z

def is_in_second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

theorem conjugate_z_in_second_quadrant :
  is_in_second_quadrant conjugate_z :=
sorry

end conjugate_z_in_second_quadrant_l774_774192


namespace tangent_perpendicular_intervals_of_monotonicity_l774_774299

def f (x a : ℝ) : ℝ := (x - 1)^2 - a * Real.log x

theorem tangent_perpendicular (a : ℝ) : 
  (∀ x : ℝ, ∀ f' : ℝ, ∀ b : ℝ, ((x - 1)^2 - a * Real.log x = f x a) →
    (f' = (dx / dx) (f x a)) → 
    ((f' * (x + 2 * (f x 1) - 1) = -1) → 
    (a = 2))) := sorry

theorem intervals_of_monotonicity (a : ℝ) : 
  (f' := (dx / dx) (f x a)) → 
    (f' x >= 0) → 
    ∀ x : ℝ, ∀ b : ℝ,
      ((f' * b = 0) →
      (if a >= ½ then 
        (⟶ increasing_on (0, ∞))
      else if a <= 0 then 
        (⟶ decreasing_on (0, (1 + sqrt (1 - 2 * a)) / 2)
        ⟶ increasing_on ((1 + sqrt (1 - 2 * a)) / 2, ∞))
      else 
        (⟶ increasing_on (0, (1 - sqrt (1 - 2 * a)) / 2)
        ⟶ decreasing_on ((1 - sqrt (1 - 2 * a)) / 2, (1 + sqrt (1 - 2 * a)) / 2)
        ⟶ increasing_on ((1 + sqrt (1 - 2 * a)) / 2, ∞)) := sorry


end tangent_perpendicular_intervals_of_monotonicity_l774_774299


namespace find_pairs_l774_774993

theorem find_pairs (p q : ℤ) (a b : ℤ) :
  (p^2 - 4 * q = a^2) ∧ (q^2 - 4 * p = b^2) ↔ 
    (p = 4 ∧ q = 4) ∨ (p = 9 ∧ q = 8) ∨ (p = 8 ∧ q = 9) :=
by
  sorry

end find_pairs_l774_774993


namespace sequence_inequality_l774_774752

variable (a : ℕ → ℝ)

theorem sequence_inequality (h1 : ∀ n, 0 ≤ a n ∧ a n ≤ 1)
                           (h2 : ∀ n, a n - 2 * a (n + 1) + a (n + 2) ≥ 0) :
  ∀ n, 0 ≤ (n + 1 : ℝ) * (a n - a (n + 1)) ∧ (n + 1 : ℝ) * (a n - a (n + 1)) ≤ 2 :=
by
  sorry

end sequence_inequality_l774_774752


namespace mark_initial_kept_percentage_l774_774305

-- Defining the conditions
def initial_friends : Nat := 100
def remaining_friends : Nat := 70
def percentage_contacted (P : ℝ) := 100 - P
def percentage_responded : ℝ := 0.5

-- Theorem statement: Mark initially kept 40% of his friends
theorem mark_initial_kept_percentage (P : ℝ) : 
  (P / 100 * initial_friends) + (percentage_contacted P / 100 * initial_friends * percentage_responded) = remaining_friends → 
  P = 40 := by
  sorry

end mark_initial_kept_percentage_l774_774305


namespace prove_PropositionC_true_l774_774871

noncomputable def PropositionA (a b : ℝ) : Prop :=
ab > 0 → a > 0 ∧ b > 0

structure RightAngledTriangle :=
(angle1 angle2 : ℝ)
(hw : angle1 + angle2 = 90)

def PropositionB (T1 T2 : RightAngledTriangle) : Prop :=
T1.angle1 = T2.angle1 ∧ T1.angle2 = T2.angle2 → T1 = T2

structure Point :=
(x y : ℝ)

def PointEquidistantFromSides (A1 A2 : ℝ) (P : Point) :=
P.x = P.y

def PropositionC (A1 A2 : ℝ) (P : Point) : Prop :=
PointEquidistantFromSides A1 A2 P → P.x = P.y

structure Quadrilateral :=
(a1 a2 b1 b2 : ℝ)

def PropositionD (Q : Quadrilateral) : Prop :=
(a1 = a2 ∧ b1 = b2 ∧ a1 ∥ a2) → (Q)

theorem prove_PropositionC_true : PropositionC:=
by
  sorry

end prove_PropositionC_true_l774_774871


namespace factorial_division_l774_774489

noncomputable def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

theorem factorial_division :
  (factorial 10) / ((factorial 7) * (factorial 3)) = 120 := by sorry

end factorial_division_l774_774489


namespace color_opposite_orange_is_indigo_l774_774818

-- Define the colors
inductive Color
| O | B | Y | S | V | I

-- Define a structure representing a view of the cube
structure CubeView where
  top : Color
  front : Color
  right : Color

-- Given views
def view1 := CubeView.mk Color.B Color.Y Color.S
def view2 := CubeView.mk Color.B Color.V Color.S
def view3 := CubeView.mk Color.B Color.I Color.Y

-- The statement to be proved: the color opposite to orange (O) is indigo (I), given the views
theorem color_opposite_orange_is_indigo (v1 v2 v3 : CubeView) :
  v1 = view1 →
  v2 = view2 →
  v3 = view3 →
  ∃ opposite_color : Color, opposite_color = Color.I :=
  by
    sorry

end color_opposite_orange_is_indigo_l774_774818


namespace solve_for_n_l774_774800

theorem solve_for_n (n : ℕ) (h : 2^n * 8^n = 64^(n - 30)) : n = 90 :=
by {
  sorry
}

end solve_for_n_l774_774800


namespace mask_duration_l774_774951

theorem mask_duration (family_size : ℕ) (total_masks : ℕ) (replacement_interval : ℕ) 
  (h_family_size : family_size = 5) 
  (h_total_masks : total_masks = 100) 
  (h_replacement_interval : replacement_interval = 4) :
  total_masks / family_size * replacement_interval = 80 :=
by
  rw [h_family_size, h_total_masks, h_replacement_interval]
  norm_num
  sorry

end mask_duration_l774_774951


namespace sum_of_degrees_is_even_l774_774508

-- Definitions based on the given conditions
variable (V : Type) -- Type for vertices
variable [Fintype V] -- Assumption that the set of vertices is finite
variable (G : SimpleGraph V) -- The graph G
variable (h_odd_deg : ∀ v ∈ V, G.degree v % 2 = 1 → #Finset.filter (λ v, G.degree v % 2 = 1) (Fintype.elems V) = 12) -- Condition that there are 12 odd-degree vertices

-- The theorem to be stated, corresponding to our proof problem
theorem sum_of_degrees_is_even :
  ∃ E, G.sum_degree = 2 * E := 
sorry

end sum_of_degrees_is_even_l774_774508


namespace estelles_classmate_borrowed_sheets_l774_774126

def estelles_notebook (pages : List ℕ) (borrowed : List ℕ) : Prop :=
  let remaining := pages.filter (λ x, ¬borrowed.contains x)
  let mean_remaining := remaining.sum / remaining.length
  mean_remaining = 49

theorem estelles_classmate_borrowed_sheets :
  ∃ (c : ℕ), c = 10 ∧ 
  estelles_notebook (List.range 100) 
  ([x for x in List.range 100 if x < (50 - c)] ++ 
   [x for x in List.range 100 if x > 50 + c]) :=
sorry

end estelles_classmate_borrowed_sheets_l774_774126


namespace ratio_new_circumference_to_area_l774_774446

theorem ratio_new_circumference_to_area (r₀ : ℝ) (h₀ : r₀ > 0) :
  let new_radius := 1.5 * r₀
  let new_circumference := 2 * Real.pi * new_radius
  let new_area := Real.pi * new_radius^2
  new_circumference / new_area = 4 / (3 * r₀) :=
by
  let new_radius := 1.5 * r₀
  let new_circumference := 2 * Real.pi * new_radius
  let new_area := Real.pi * new_radius^2
  have h : new_circumference / new_area = 4 / (3 * r₀)
  sorry

end ratio_new_circumference_to_area_l774_774446


namespace sum_of_squares_of_roots_poly1_l774_774422

noncomputable def poly1 : Polynomial ℝ := 
  Polynomial.C 2 - Polynomial.mul (Polynomial.mul (Polynomial.X - Polynomial.C 3) (Polynomial.X - Polynomial.C 1)) (Polynomial.X + Polynomial.C 2)

theorem sum_of_squares_of_roots_poly1 : 
  ∑ i in ({0, 1, 2, 3} : Finset ℝ).filter (λ r, poly1.eval r = 0), r^2 = 79 := sorry

end sum_of_squares_of_roots_poly1_l774_774422


namespace find_natural_number_A_l774_774145

theorem find_natural_number_A :
  ∃ A : ℕ, (∀ (x y : ℕ), (0 ≤ x ∧ x ≤ 6 ∧ 0 ≤ y ∧ y ≤ 7 ∧ x + y ≥ A) → 
  ((x = 3 ∧ y = 7) ∨ (x = 4 ∧ (y = 6 ∨ y = 7)) ∨ 
  (x = 5 ∧ (y = 5 ∨ y = 6 ∨ y = 7)) ∨ 
  (x = 6 ∧ (y = 4 ∨ y = 5 ∨ y = 6 ∨ y = 7)) ∨ 
  (x + y < A ∧ (56 - (AI*I + A) / 2) = A))) := 
begin
  existsi 10,
  sorry
end

end find_natural_number_A_l774_774145


namespace maximum_X_no_three_align_l774_774703

open Set

theorem maximum_X_no_three_align (grid : Fin 5 × Fin 5 → Prop) :
  (∀ (row : Fin 5), ∃ (count : ℕ), count ≤ 2 ∧ (∃ positions : Finset (Fin 5 × Fin 5), 
  positions.card = count ∧ (∀ p ∈ positions, p.1 = row ∧ grid p))) →
  (∀ (col : Fin 5), ∃ (count : ℕ), count ≤ 2 ∧ (∃ positions : Finset (Fin 5 × Fin 5), 
  positions.card = count ∧ (∀ p ∈ positions, p.2 = col ∧ grid p))) →
  (∀ (d : Int), ∃ (count : ℕ), count ≤ 2 ∧ (∃ positions : Finset (Fin 5 × Fin 5), 
  positions.card = count ∧ (∀ p ∈ positions, (p.1 : Int) - (p.2 : Int) = d ∧ grid p))) →
  ∃ (count : ℕ), count = 12 ∧ (∃ positions : Finset (Fin 5 × Fin 5), 
  positions.card = count ∧ (∀ p1 p2 ∈ positions, p1 ≠ p2 → (p1.1 ≠ p2.1 ∧ p1.2 ≠ p2.2 ∧ 
  (p1.1 : Int) - (p1.2 : Int) ≠ (p2.1 : Int) - (p2.2 : Int) ∧ 
  (p1.1 : Int) + (p1.2 : Int) ≠ (p2.1 : Int) + (p2.2 : Int)))) :=
sorry

end maximum_X_no_three_align_l774_774703


namespace alternate_perpendicular_products_equal_l774_774319

theorem alternate_perpendicular_products_equal (A : Point) (O : Point) (r : ℝ) (n : ℕ) (hn : Even (2 * n)) :
  let P := λ k, (draw_perpendicular A (side_of_inscribed_polygon (2 * n) k O r))
 in (∏ k in (Finset.range n).filter (λ x, Even x), P (2 * k)) = 1 :=
sorry

end alternate_perpendicular_products_equal_l774_774319


namespace perfect_square_factors_450_l774_774630

theorem perfect_square_factors_450 : 
  ∃ n, n = 4 ∧ (∀ k | 450, k ∣ 450 → is_square k) → (∃ m1 m2 m3 : ℕ, 450 = 2^m1 * 3^m2 * 5^m3 ) :=
by sorry

end perfect_square_factors_450_l774_774630


namespace water_level_after_opening_valve_l774_774012

-- Define the initial conditions and final height to be proved
def initial_water_height_cm : ℝ := 40
def initial_oil_height_cm : ℝ := 40
def water_density : ℝ := 1000
def oil_density : ℝ := 700
def final_water_height_cm : ℝ := 34

-- The proof that the final height of water after equilibrium will be 34 cm
theorem water_level_after_opening_valve :
  ∀ (h_w h_o : ℝ),
  (water_density * h_w = oil_density * h_o) ∧ (h_w + h_o = initial_water_height_cm + initial_oil_height_cm) →
  h_w = final_water_height_cm :=
by
  -- Here goes the proof, skipped with sorry
  sorry

end water_level_after_opening_valve_l774_774012


namespace intersection_is_integer_for_m_l774_774717

noncomputable def intersects_at_integer_point (m : ℤ) : Prop :=
∃ x y : ℤ, y = x - 4 ∧ y = m * x + 2 * m

theorem intersection_is_integer_for_m :
  intersects_at_integer_point 8 :=
by
  -- The proof would go here
  sorry

end intersection_is_integer_for_m_l774_774717


namespace gain_percentage_is_15_l774_774061

-- Initial conditions
def CP_A : ℤ := 100
def CP_B : ℤ := 200
def CP_C : ℤ := 300
def SP_A : ℤ := 110
def SP_B : ℤ := 250
def SP_C : ℤ := 330

-- Definitions for total values
def Total_CP : ℤ := CP_A + CP_B + CP_C
def Total_SP : ℤ := SP_A + SP_B + SP_C
def Overall_gain : ℤ := Total_SP - Total_CP
def Gain_percentage : ℚ := (Overall_gain * 100) / Total_CP

-- Theorem to prove the overall gain percentage
theorem gain_percentage_is_15 :
  Gain_percentage = 15 := 
by
  -- Proof placeholder
  sorry

end gain_percentage_is_15_l774_774061


namespace function_satisfy_f1_function_satisfy_f2_l774_774869

noncomputable def f1 (x : ℝ) : ℝ := 2
noncomputable def f2 (x : ℝ) : ℝ := x

theorem function_satisfy_f1 : 
  ∀ x y : ℝ, x > 0 → y > 0 → f1 (x + y) + f1 x * f1 y = f1 (x * y) + f1 x + f1 y :=
by 
  intros x y hx hy
  unfold f1
  sorry

theorem function_satisfy_f2 :
  ∀ x y : ℝ, x > 0 → y > 0 → f2 (x + y) + f2 x * f2 y = f2 (x * y) + f2 x + f2 y :=
by 
  intros x y hx hy
  unfold f2
  sorry

end function_satisfy_f1_function_satisfy_f2_l774_774869


namespace maximum_value_l774_774820

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x) ^ 2 * (Real.tan x) + Real.cos (2 * x)

theorem maximum_value : ∃ x, f x = sqrt 2 :=
sorry

end maximum_value_l774_774820


namespace mask_duration_l774_774950

theorem mask_duration (family_size : ℕ) (total_masks : ℕ) (replacement_interval : ℕ) 
  (h_family_size : family_size = 5) 
  (h_total_masks : total_masks = 100) 
  (h_replacement_interval : replacement_interval = 4) :
  total_masks / family_size * replacement_interval = 80 :=
by
  rw [h_family_size, h_total_masks, h_replacement_interval]
  norm_num
  sorry

end mask_duration_l774_774950


namespace evaluate_expression_l774_774513

theorem evaluate_expression : (7 ^ 14 / 49 ^ 6) = 49 :=
by
  have h : 49 = 7 ^ 2 := rfl
  sorry

end evaluate_expression_l774_774513


namespace num_even_divisors_of_210_l774_774823

open Set

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k
def divisors (n : ℕ) : Set ℕ := {d | d ∣ n}
def even_divisors (n : ℕ) : Set ℕ := {d | d ∣ n ∧ is_even d}

theorem num_even_divisors_of_210 : 
  let n := 210 in 
  let prime_factors_210 := [2, 3, 5, 7] in
  ∀ n = 2 * 3 * 5 * 7,
  (even_divisors n).card = 8 := 
by
  intro n h
  sorry

end num_even_divisors_of_210_l774_774823


namespace inequality_proof_l774_774164

noncomputable def a : ℝ := Real.log 0.3 / Real.log 2
noncomputable def b : ℝ := 2 ^ 0.3
noncomputable def c : ℝ := 0.3 ^ 2

theorem inequality_proof : b > c ∧ c > a := by
  -- No proof required, hence sorry
  sorry

end inequality_proof_l774_774164


namespace scheduling_ways_l774_774210

theorem scheduling_ways (n m : ℕ) (courses periods : Finset ℕ) 
  (h_courses : courses.card = 4) (h_periods : periods.card = 7)
  (h_no_consecutive : ∀ (i j : ℕ) (ih : i ∈ courses) (jh : j ∈ courses), (abs (i - j) > 1)) : 
  ∃ ways : ℕ, ways = 360 :=
begin
  -- Given the conditions h_courses, h_periods, and h_no_consecutive, 
  -- prove that the number of valid ways to schedule the courses
  -- is 360.
  sorry
end

end scheduling_ways_l774_774210


namespace quadrilateral_side_length_l774_774883

theorem quadrilateral_side_length
  (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
  (AD BC AC BD CD : ℝ)
  (h1 : AD = BC)
  (h2 : ∠ A D C > ∠ B C D) :
  AC > BD :=
sorry

end quadrilateral_side_length_l774_774883


namespace derivative_of_cubic_root_l774_774047

theorem derivative_of_cubic_root (x_0 : ℝ) : (f : ℝ → ℝ) (h1 : f = λ x, x^3) (h2 : deriv f x_0 = 6) 
  : x_0 = real.sqrt 2 ∨ x_0 = -real.sqrt 2 := 
by 
  sorry

end derivative_of_cubic_root_l774_774047


namespace similar_not_congruent_impossible_equilateral_cuts_l774_774443

-- Problem Conditions
def triangular_prism (T : Type) := Π (cut1 cut2 : T), ¬(cut1 = cut2) 

-- Proof Problems
theorem similar_not_congruent (T : Type) [triangle T] 
  (prism : triangular_prism T) : 
  ∃ (cut1 cut2 : T), (triangle.similar cut1 cut2) ∧ ¬(triangle.congruent cut1 cut2) := 
sorry

theorem impossible_equilateral_cuts (T : Type) [triangle T] 
  (prism : triangular_prism T) : 
  ∃ (cut1 cut2 : T), (triangle.equilateral cut1 ∧ triangle.side_length cut1 = 1) → (triangle.equilateral cut2 ∧ triangle.side_length cut2 = 2) → false := 
sorry

end similar_not_congruent_impossible_equilateral_cuts_l774_774443


namespace abc_equality_l774_774334

noncomputable def abc_value (a b c : ℝ) : ℝ := (11 + Real.sqrt 117) / 2

theorem abc_equality (a b c : ℝ) (h1 : a + 1/b = 5) (h2 : b + 1/c = 2) (h3 : (c + 1/a)^2 = 4) :
  a * b * c = abc_value a b c := 
sorry

end abc_equality_l774_774334


namespace trigonometric_expression_l774_774540

theorem trigonometric_expression
  (α : ℝ)
  (h : (Real.tan α - 3) * (Real.sin α + Real.cos α + 3) = 0) :
  2 + (2 / 3) * Real.sin α ^ 2 + (1 / 4) * Real.cos α ^ 2 = 21 / 8 := 
by sorry

end trigonometric_expression_l774_774540


namespace total_cost_is_716_mom_has_enough_money_l774_774082

/-- Definition of the price of the table lamp -/
def table_lamp_price : ℕ := 86

/-- Definition of the price of the electric fan -/
def electric_fan_price : ℕ := 185

/-- Definition of the price of the bicycle -/
def bicycle_price : ℕ := 445

/-- The total cost of buying all three items -/
def total_cost : ℕ := table_lamp_price + electric_fan_price + bicycle_price

/-- Mom's money -/
def mom_money : ℕ := 300

/-- Problem 1: Prove that the total cost equals 716 -/
theorem total_cost_is_716 : total_cost = 716 := 
by 
  sorry

/-- Problem 2: Prove that Mom has enough money to buy a table lamp and an electric fan -/
theorem mom_has_enough_money : table_lamp_price + electric_fan_price ≤ mom_money :=
by 
  sorry

end total_cost_is_716_mom_has_enough_money_l774_774082


namespace workers_load_truck_together_l774_774469

theorem workers_load_truck_together (h₁ : (1 : ℚ) / 6) (h₂ : (1 : ℚ) / 8) : 
  1 / (h₁ + h₂) = 24 / 7 := 
by
  sorry

end workers_load_truck_together_l774_774469


namespace math_problem_l774_774212

theorem math_problem (a b : ℝ) (h1 : 4 + a = 5 - b) (h2 : 5 + b = 8 + a) : 4 - a = 3 :=
by
  sorry

end math_problem_l774_774212


namespace range_of_a_l774_774182

noncomputable def f (x : ℝ) : ℝ := if x > 0 then log2 (2^x / (2^x + 1)) else log2 ((2^x / (2^x + 1)))

def isEven (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f(x) = f(-x)

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) (is_even_f : isEven f) :
  (∀ x > 0, f x = log2 (2^x / (2^x + 1))) → 
  (∀ t ∈ set.Icc (1/2 : ℝ) 2, f (t + a) - f (t - 1) ≥ 0) → 
  (a ≥ 0 ∨ a ≤ -3 ∨ a = -1) :=
by
  sorry

end range_of_a_l774_774182


namespace find_function_l774_774141

theorem find_function (S : ℝ → ℝ) 
  (h1 : ∀ x, deriv S x = 2 / real.sqrt (5 - x))
  (h2 : S 1 = -1) : 
  ∀ x, S x = 7 - 4 * real.sqrt (5 - x) :=
by
  -- This is a placeholder for the proof
  sorry

end find_function_l774_774141


namespace area_triangle_ABC_2K_l774_774699

variables {A B C H M : Type} [AffineSpace ℝ A] [AffineSpace ℝ B] [AffineSpace ℝ C] [AffineSpace ℝ H]  [AffineSpace ℝ M]
variable {K : ℝ}

-- Define triangle ABC with right angle at C, altitude CH, and median CM
def triangle_ABC (A B C : AffineSpace ℝ) : Prop :=
  is_right_triangle A B C ∧
  is_altitude C H A B ∧
  is_median C M A B ∧
  bisects_right_angle A C B M ∧
  area (triangle C H M) = K

-- Define the proof statement
theorem area_triangle_ABC_2K (A B C : AffineSpace ℝ) (h : triangle_ABC A B C) : 
  area (triangle A B C) = 2 * K := sorry

end area_triangle_ABC_2K_l774_774699


namespace pair_count_eq_pair_count_3_l774_774300

open Set Nat

def pair_count (n : ℕ) : ℕ :=
  ∑ k in range (n-1), (2^(n-1) - 2^(k-1))

theorem pair_count_eq (n : ℕ) (hn : n ≥ 3) : 
  pair_count n = (n-2) * 2^(n-1) + 1 :=
sorry

-- Special case for n = 3
theorem pair_count_3 : pair_count 3 = 5 :=
sorry

end pair_count_eq_pair_count_3_l774_774300


namespace min_value_of_a_plus_2b_l774_774178

theorem min_value_of_a_plus_2b
  (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1 / a + 1 / b = 1) : 
  a + 2 * b ≥ 3 + 2 * Real.sqrt 2 :=
begin
  sorry
end

end min_value_of_a_plus_2b_l774_774178


namespace flower_town_no_possible_transactions_l774_774409

theorem flower_town_no_possible_transactions :
  ∀ (n : ℕ), n = 1990 →
  ∀ (c : ℕ), c = 10 →
  (∀ (total_give : ℕ), total_give = n * c →
   ∀ (transaction_coins : ℕ), transaction_coins = 2 →
   (total_give % transaction_coins = 0) ∧ (total_give ≠ 3 * k) ∀ k, k ∈ ℕ)
   → false :=
by
  intros n hn c hc total_give htotal_give transaction_coins htransaction_coins
  intro h
  have h1 : total_give % 2 = 0, from h.left
  have h2 : ∃ k, total_give = 3 * k, from h.right
  sorry

end flower_town_no_possible_transactions_l774_774409


namespace symmetric_points_sum_l774_774716

variable (a b : ℝ)

theorem symmetric_points_sum (h₁ : P = (a, 1)) (h₂ : Q = (2, b)) (symmetry_condition : symmetric_about_x_axis P Q) : a + b = 1 := 
by
  sorry

end symmetric_points_sum_l774_774716


namespace period_of_f_l774_774862

noncomputable def f (x : ℝ) : ℝ := (Real.tan (x/3)) + (Real.sin x)

theorem period_of_f : ∃ T > 0, ∀ x, f (x + T) = f x :=
  sorry

end period_of_f_l774_774862


namespace line_of_intersection_l774_774500

theorem line_of_intersection (x y z : ℝ) :
  (2 * x + 3 * y + 3 * z - 9 = 0) ∧ (4 * x + 2 * y + z - 8 = 0) →
  ((x / 4.5 + y / 3 + z / 3 = 1) ∧ (x / 2 + y / 4 + z / 8 = 1)) :=
by
  sorry

end line_of_intersection_l774_774500


namespace integral_one_over_x_plus_x_l774_774133

open Real

theorem integral_one_over_x_plus_x :
  ∫ x in 1..Real.exp 1, (1 / x + x) = 1/2 * Real.exp 2 + 1/2 :=
by sorry

end integral_one_over_x_plus_x_l774_774133


namespace bela_always_wins_l774_774092

theorem bela_always_wins (m : ℕ) (h : m > 10) : 
  ∃ strategy : (ℕ → ℝ) → Prop, 
  ∀ (turns : ℕ) (moves : ℕ → ℝ),
    let valid_move := λ n, ∀ k < n, abs (moves k - moves n) > 1
    turns > 0 → moves 0 = m ∧ (∀ n < turns, valid_move n) → strategy moves → 
    (∃ n < turns, ∀ k < n, abs (moves k - moves n) ≤ 1) ∨ turns > m := 
by
  sorry

end bela_always_wins_l774_774092


namespace knife_to_utensils_ratio_l774_774514

-- Conditions introduced as definitions
def hand_mitts_price : ℝ := 14.00
def apron_price : ℝ := 16.00
def utensils_price : ℝ := 10.00
def discount_rate : ℝ := 0.25
def total_spend_after_discount : ℝ := 135.00
def nieces : ℕ := 3

-- Additional calculated values based on conditions
def subtotal_before_discount : ℝ :=
  (hand_mitts_price * nieces) + (apron_price * nieces) + (utensils_price * nieces)

def total_before_discount : ℝ :=
  total_spend_after_discount / (1 - discount_rate)

def knives_total_cost_before_discount : ℝ :=
  total_before_discount - subtotal_before_discount

def knife_price : ℝ :=
  knives_total_cost_before_discount / nieces

-- Proof statement
theorem knife_to_utensils_ratio : knife_price / utensils_price = 2 := 
by
  -- sorry is used here to skip the actual proof
  sorry

end knife_to_utensils_ratio_l774_774514


namespace perfect_square_divisors_of_450_l774_774616

theorem perfect_square_divisors_of_450 :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let e1 := 1
  let e2 := 2
  let e3 := 2
  ∃ (count : ℕ), count = 4 ∧
  (∀ (d : ℕ), d ∣ (p1^e1 * p2^e2 * p3^e3) →
    (∀ (pe : ℕ × ℕ), pe ∈ (Multiset.ofList [(p1, e1), (p2, e2), (p3, e3)]).powerset → 
      (∃ (ex1 ex2 ex3: ℕ), ex1 ≤ e1 ∧ ex2 ≤ e2 ∧ ex3 ≤ e3 ∧ ex1 % 2 = 0 ∧ ex2 % 2 = 0 ∧ ex3 % 2 = 0))) :=
sorry

end perfect_square_divisors_of_450_l774_774616


namespace divides_expression_l774_774762

theorem divides_expression (y : ℕ) (h : y ≠ 0) : (y - 1) ∣ (y ^ (y ^ 2) - 2 * y ^ (y + 1) + 1) :=
sorry

end divides_expression_l774_774762


namespace q1_includes_specific_counterfeit_q2_excludes_specific_counterfeit_q3_exactly_two_counterfeits_q4_at_least_two_counterfeits_q5_at_most_two_counterfeits_l774_774894

-- Condition definitions
def total_goods := 35
def counterfeit_goods := 15
def genuine_goods := total_goods - counterfeit_goods
def select_goods := 3

-- Combinations
noncomputable def comb (n k : ℕ) : ℕ :=
  if k > n then 0 else (nat.factorial n) / ((nat.factorial k) * (nat.factorial (n - k)))

-- Question (1)
theorem q1_includes_specific_counterfeit (c_specific : ℕ) (h1: c_specific < counterfeit_goods) :
  (comb (total_goods - 1) (select_goods - 1)) = 561 :=
  sorry

-- Question (2)
theorem q2_excludes_specific_counterfeit (c_specific : ℕ) (h2: c_specific < counterfeit_goods) :
  (comb (total_goods - 1) select_goods) = 5984 :=
  sorry

-- Question (3)
theorem q3_exactly_two_counterfeits :
  (comb counterfeit_goods 2) * (comb genuine_goods 1) = 2100 :=
  sorry

-- Question (4)
theorem q4_at_least_two_counterfeits :
  ((comb counterfeit_goods 2) * (comb genuine_goods 1)) + (comb counterfeit_goods 3) = 2555 :=
  sorry

-- Question (5)
theorem q5_at_most_two_counterfeits :
  ((comb counterfeit_goods 2) * (comb genuine_goods 1)) +
  ((comb counterfeit_goods 1) * (comb genuine_goods 2)) +
  (comb genuine_goods 3) = 6090 :=
  sorry

end q1_includes_specific_counterfeit_q2_excludes_specific_counterfeit_q3_exactly_two_counterfeits_q4_at_least_two_counterfeits_q5_at_most_two_counterfeits_l774_774894


namespace max_value_iff_perpendicular_l774_774267

/- Point P lies inside angle A. A line through P intersects the sides of the angle at points B and C. -/
variables {A P B C D : Type*} [MetricSpace P]
variables {AP PB PC AD BD CD : ℝ}

/- Represent the question: When does (1/PB + 1/PC) achieve its maximum value? -/
def max_value_condition (P B C : P) : Prop := 
  AP ∠_between_ (B, C)

/- Given conditions: Inside the angle ∠A, point P and line intersects at points B and C. AD ⊥ BC, AD = AP -/
def is_perpendicular_to (AP AD : P) : Prop := 
  AD ⊥ BC

/- The proof statement to prove: (1/PB + 1/PC) achieves its maximum value if and only if AP ⊥ BC -/
theorem max_value_iff_perpendicular {P B C D : P} :
  max_value_condition P B C ↔ is_perpendicular_to AP AD :=
sorry

end max_value_iff_perpendicular_l774_774267


namespace plane_split_four_regions_l774_774497

theorem plane_split_four_regions :
  (∀ x y : ℝ, y = 3 * x ∨ x = 3 * y) → (exists regions : ℕ, regions = 4) :=
by
  sorry

end plane_split_four_regions_l774_774497


namespace square_roots_of_four_ninths_cube_root_of_neg_sixty_four_l774_774825

theorem square_roots_of_four_ninths : {x : ℚ | x ^ 2 = 4 / 9} = {2 / 3, -2 / 3} :=
by
  sorry

theorem cube_root_of_neg_sixty_four : {y : ℚ | y ^ 3 = -64} = {-4} :=
by
  sorry

end square_roots_of_four_ninths_cube_root_of_neg_sixty_four_l774_774825


namespace remainder_of_5_pow_2023_mod_6_l774_774400

theorem remainder_of_5_pow_2023_mod_6 : 5^2023 % 6 = 5 := 
by sorry

end remainder_of_5_pow_2023_mod_6_l774_774400


namespace angle_OBC_is_3pi_over_16_l774_774377

theorem angle_OBC_is_3pi_over_16
  (O A B C D E F : Point)
  (circle_center_O : circle O)
  (A_on_circle : A ∈ circle_center_O)
  (B_on_circle : B ∈ circle_center_O)
  (C_on_circle : C ∈ circle_center_O)
  (D_on_circle : D ∈ circle_center_O)
  (E_on_circle : E ∈ circle_center_O)
  (F_on_circle : F ∈ circle_center_O)
  (circle_intersects_AD_at_E : lies_on_circle (O, AD) E)
  (circle_intersects_AB_at_F : lies_on_circle (O, AB) F)
  (chord_BF_eq_FE : chord_length O B F = chord_length O F E)
  (chord_FE_eq_ED : chord_length O F E = chord_length O E D)
  (chord_ED_eq_BF : chord_length O E D = chord_length O B F)
  (chord_BC_eq_CD : chord_length O B C = chord_length O C D)
  (∠DAB_is_right_angle : angle D A B = real.pi / 2) :
  angle O B C = 3 * real.pi / 16 :=
by
  sorry

end angle_OBC_is_3pi_over_16_l774_774377


namespace bisection_and_tangent_properties_find_tangent_PCB_l774_774236

/-- Part (1): Given a triangle PBC with ∠PBC = 60°, a tangent to the circumcircle of PBC is drawn through P,
  meeting the extension of CB at A. Points D and E are on PA and the circumcircle respectively such that ∠DBE = 90°
  and PD = PE. If BF bisects ∠PBC, and AF, BP, and CD are concurrent, then prove this setup. -/
theorem bisection_and_tangent_properties
  {P B C A D E F : Point}
  (h1 : angle P B C = 60)
  (circumcircle : circle O P B C)
  (tangent_through_P : tangent P circumcircle meets CB at A)
  (on_PA : D lies_on PA)
  (on_circumcircle : E lies_on circumcircle)
  (right_angle_DBE : angle D B E = 90)
  (PD_eq_PE : PD = PE)
  (intersect_F : BE intersects PC at F)
  (concurrent_AF_BP_CD : concurrent AF BP CD) :
  bisects B F angle P B C :=
sorry

/-- Part (2): Given the same setup as above, find the value of tangent of angle PCB. -/
theorem find_tangent_PCB
  {P B C A D E F : Point}
  (h1 : angle P B C = 60)
  (circumcircle : circle O P B C)
  (tangent_through_P : tangent P circumcircle meets CB at A)
  (on_PA : D lies_on PA)
  (on_circumcircle : E lies_on circumcircle)
  (right_angle_DBE : angle D B E = 90)
  (PD_eq_PE : PD = PE)
  (intersect_F : BE intersects PC at F)
  (concurrent_AF_BP_CD : concurrent AF BP CD) :
  tan (angle P C B) = (6 + sqrt 3) / 11 :=
sorry

end bisection_and_tangent_properties_find_tangent_PCB_l774_774236


namespace correct_statistical_survey_statement_l774_774415

-- Define the conditions as Lean definitions
def condition_A : Prop := ¬ (∀ survey, comprehensive_survey survey)
def condition_B : Prop := ∀ students, comprehensive_survey (conduct_vision_tests students)
def condition_C : Prop := ∀ (households samples : ℕ), (samples ≠ households) → (sample_size households samples = samples)
def condition_D : Prop := ∀ students basketball_team, has_bias (sample_height basketball_team) students

-- Define the question
def question : Prop := ∀ statements : list Prop, (statements = [condition_A, condition_B, condition_C, condition_D]) →
  (correct_statement_in_statements statements = condition_B)

-- The proof statement
theorem correct_statistical_survey_statement : question :=
sorry

end correct_statistical_survey_statement_l774_774415


namespace ordering_abc_l774_774545

section ordering_abc

def a : ℝ := Real.logb 3 (Real.sqrt 2)
def b : ℝ := 0.3^0.5
def c : ℝ := 0.5^(-0.2)

theorem ordering_abc : a < b ∧ b < c := by
  have h0 : a = Real.logb 3 (Real.sqrt 2) := rfl
  have h1 : b = 0.3^0.5 := rfl
  have h2 : c = 0.5^(-0.2) := rfl
  sorry

end ordering_abc

end ordering_abc_l774_774545


namespace sum_c_2017_l774_774562

def a (n : ℕ) : ℕ := 3 * n + 1

def b (n : ℕ) : ℕ := 4^(n-1)

def c (n : ℕ) : ℕ := if n = 1 then 7 else 3 * 4^(n-1)

theorem sum_c_2017 : (Finset.range 2017).sum c = 4^2017 + 3 :=
by
  -- definitions and required assumptions
  sorry

end sum_c_2017_l774_774562


namespace candy_bar_cost_l774_774511

theorem candy_bar_cost :
  ∀ (members : ℕ) (avg_candy_bars : ℕ) (total_earnings : ℝ), 
  members = 20 →
  avg_candy_bars = 8 →
  total_earnings = 80 →
  total_earnings / (members * avg_candy_bars) = 0.50 :=
by
  intros members avg_candy_bars total_earnings h_mem h_avg h_earn
  sorry

end candy_bar_cost_l774_774511


namespace integral_sqrt_x_2_minus_x_l774_774837

theorem integral_sqrt_x_2_minus_x :
  ∫ x in 0..1, Real.sqrt (x * (2 - x)) = Real.pi / 4 :=
by
  sorry

end integral_sqrt_x_2_minus_x_l774_774837


namespace tangent_line_at_point_1_range_of_a_for_f_nonnegative_l774_774201

noncomputable def f (a x : ℝ) : ℝ := (2 * a * x - log x) * log x - 2 * a * x + 2

noncomputable def f_prime (a x : ℝ) : ℝ := 
  2 * (1 - (1 / x)) * log x  -- When a = 1
  + 2 * (a * x - 1) / x * log x  -- For general a

theorem tangent_line_at_point_1 (a : ℝ) : 
  a = 1 → (tangent_line (f a) 1 = λ x, 0) := sorry

theorem range_of_a_for_f_nonnegative (a : ℝ) : 
  (∀ x, 1 ≤ x → f a x ≥ 0) ↔ (1 / real.exp 2 ≤ a ∧ a ≤ 1) := sorry

end tangent_line_at_point_1_range_of_a_for_f_nonnegative_l774_774201


namespace volume_after_increase_l774_774928

variable (l w h : ℝ)

def original_volume (l w h : ℝ) : ℝ := l * w * h
def original_surface_area (l w h : ℝ) : ℝ := 2 * (l * w + w * h + h * l)
def original_edge_sum (l w h : ℝ) : ℝ := 4 * (l + w + h)
def increased_volume (l w h : ℝ) : ℝ := (l + 2) * (w + 2) * (h + 2)

theorem volume_after_increase :
  original_volume l w h = 5000 →
  original_surface_area l w h = 1800 →
  original_edge_sum l w h = 240 →
  increased_volume l w h = 7048 := by
  sorry

end volume_after_increase_l774_774928


namespace root_in_interval_l774_774506

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * Real.log x + x - (1 / x) - 2
noncomputable def f' (x : ℝ) : ℝ := (1 / (2 * x)) + 1 + (1 / (x ^ 2))

theorem root_in_interval : ∃ x ∈ Set.Ioo 2 Real.exp 1, f x = 0 :=
by {
  have mono : ∀ x > 0, f' x > 0,
  { intro x_pos, sorry },
  have f_at_2 : f 2 < 0, by { sorry },
  have f_at_e : f (Real.exp 1) > 0, by { sorry },
  rwa [← Set.mem_Ioo, ← exists_mem] at this,
  apply Intermediate_Value_Theorem,
  exact ⟨f_at_2, f_at_e, ⟨2_pos, Real.exp_pos⟩, mono⟩,
  apply_instance,
  use x,
  exact ⟨2_lt_e, e_lt_exp, h, hx⟩,
}

end root_in_interval_l774_774506


namespace tangent_and_cotangent_polynomials_l774_774689

theorem tangent_and_cotangent_polynomials (p r : ℝ) (α β : ℝ)
  (h_tan_roots : ∀ x : ℝ, x^2 - 2 * p * x + (p^2 - 1) = 0 ↔ x = (Real.tan α) ∨ x = (Real.tan β))
  (h_cot_roots : ∀ x : ℝ, x^2 - 2 * r * x + (r^2 - 1) = 0 ↔ x = (Real.cot α) ∨ x = (Real.cot β)) :
  r^2 = p^2 / (p^2 - 1) :=
 by sorry

end tangent_and_cotangent_polynomials_l774_774689


namespace perfect_square_divisors_count_450_l774_774643

theorem perfect_square_divisors_count_450 :
  let is_divisor (d n : ℕ) := n % d = 0
  let is_perfect_square (n : ℕ) := ∃ m : ℕ, m * m = n
  let prime_factorization_450 := [(2, 1), (3, 2), (5, 2)]
  ∃ (count : ℕ), count = 4 ∧ count = (Finset.filter 
    (λ d, is_perfect_square d ∧ is_divisor d 450)
    (Finset.powersetPrimeFactors prime_factorization_450)).card := sorry

end perfect_square_divisors_count_450_l774_774643


namespace find_prices_max_sets_of_go_compare_options_l774_774595

theorem find_prices (x y : ℕ) (h1 : 2 * x + 3 * y = 140) (h2 : 4 * x + y = 130) :
  x = 25 ∧ y = 30 :=
by sorry

theorem max_sets_of_go (m : ℕ) (h3 : 25 * (80 - m) + 30 * m ≤ 2250) :
  m ≤ 50 :=
by sorry

theorem compare_options (a : ℕ) :
  (a < 10 → 27 * a < 21 * a + 60) ∧ (a = 10 → 27 * a = 21 * a + 60) ∧ (a > 10 → 27 * a > 21 * a + 60) :=
by sorry

end find_prices_max_sets_of_go_compare_options_l774_774595


namespace probability_53_Sundays_in_leap_year_l774_774146

noncomputable def leapYear_days := 366
def weeks_in_leapYear := 52
def extra_days := leapYear_days - (weeks_in_leapYear * 7)
def combinations_of_extra_days := 7
def combinations_with_53_Sundays := 2

theorem probability_53_Sundays_in_leap_year :
  (combinations_with_53_Sundays / combinations_of_extra_days) = (2 / 7) :=
by
  sorry

end probability_53_Sundays_in_leap_year_l774_774146


namespace exact_value_expression_l774_774100

theorem exact_value_expression : |3 * Real.pi - |Real.pi - 10|| = 4 * Real.pi - 10 :=
sorry

end exact_value_expression_l774_774100


namespace radius_of_small_semicircle_l774_774732

theorem radius_of_small_semicircle
  (radius_big_semicircle : ℝ)
  (radius_inner_circle : ℝ) 
  (pairwise_tangent : ∀ x : ℝ, x = radius_big_semicircle → x = 12 ∧ 
                                x = radius_inner_circle → x = 6 ∧ 
                                true) :
  ∃ (r : ℝ), r = 4 :=
by 
  sorry

end radius_of_small_semicircle_l774_774732


namespace max_miles_for_15_dollars_l774_774373

noncomputable def fare (miles: ℝ) : ℝ :=
  if miles <= 0.75 then 3.50
  else 3.50 + 2.5 * (miles - 0.75)

def total_cost (miles: ℝ) : ℝ := fare miles + 3

theorem max_miles_for_15_dollars : 
  ∃ miles, 
    total_cost miles = 15 ∧ 
    miles = 4.15 :=
sorry

end max_miles_for_15_dollars_l774_774373


namespace perfect_square_factors_450_l774_774655

def prime_factorization (n : ℕ) : Prop :=
  n = 2^1 * 3^2 * 5^2

theorem perfect_square_factors_450 : prime_factorization 450 → ∃ n : ℕ, n = 4 :=
by
  sorry

end perfect_square_factors_450_l774_774655


namespace find_detergent_volume_l774_774366

variable (B D W : ℕ)
variable (B' D' W': ℕ)
variable (water_volume: unit)
variable (detergent_volume: unit)

def original_ratio (B D W : ℕ) : Prop := B = 2 * W / 100 ∧ D = 40 * W / 100

def altered_ratio (B' D' W' B D W : ℕ) : Prop :=
  B' = 3 * B ∧ D' = D / 2 ∧ W' = W ∧ W' = 300

theorem find_detergent_volume {B D W B' D' W'} (h₀ : original_ratio B D W) (h₁ : altered_ratio B' D' W' B D W) :
  D' = 120 :=
sorry

end find_detergent_volume_l774_774366


namespace number_of_girls_l774_774833

theorem number_of_girls (classes : ℕ) (students_per_class : ℕ) (boys : ℕ) (girls : ℕ) 
  (h1 : classes = 4) 
  (h2 : students_per_class = 25) 
  (h3 : boys = 56) 
  (h4 : girls = (classes * students_per_class) - boys) : 
  girls = 44 :=
by
  sorry

end number_of_girls_l774_774833


namespace minimum_P_l774_774532

-- Definitions for the problem conditions
def isOdd (n : ℤ) : Prop := n % 2 = 1

def closestInt (x : ℝ) : ℤ := round x

-- Probability P(k) where P(k) is the probability that the equation holds
noncomputable def P (k : ℤ) : ℝ :=
  -- This represents the probability calculation as per problem context
  sorry

-- The statement to prove the minimum value of P(k)
theorem minimum_P :
  ∀ (k : ℤ), (1 ≤ k ∧ k ≤ 99 ∧ isOdd k) → P k ≥ P 67 ∧ P 67 = 34/67 :=
by
  sorry

end minimum_P_l774_774532


namespace inequality_proof_l774_774589

open Nat

noncomputable def a_seq (n : ℕ) : ℕ :=
  if n = 0 then 0 else if n = 1 then 2 else 2 * 4^(n-1)

noncomputable def b_seq (n : ℕ) : ℕ :=
  if n = 0 then 0 else log 2 (a_seq n)

def S_sum (n : ℕ) : ℕ :=
  ∑ i in range (n+1), b_seq i

def c_seq (n : ℕ) : ℕ :=
  (b_seq n + 1) / 2

theorem inequality_proof (n : ℕ) : 
  (∑ i in range n, sqrt (c_seq i * c_seq (i + 1))) < (1 / 2) * S_sum (n + 1) := 
sorry

end inequality_proof_l774_774589


namespace find_real_parts_l774_774163

theorem find_real_parts (a b : ℝ) (i : ℂ) (hi : i*i = -1) 
(h : a + b*i = (1 - i) * i) : a = 1 ∧ b = -1 :=
sorry

end find_real_parts_l774_774163


namespace geometric_series_sum_l774_774969

theorem geometric_series_sum :
  let a := 3
  let r := 1 / 3
  (∑' (n : ℕ), a * r ^ n) = 9 / 2 :=
by
  -- Definitions
  let a : ℚ := 3
  let r : ℚ := 1 / 3
  -- We need to show the sum of the series is 9 / 2
  have h : (∑' (n : ℕ), a * r ^ n) = 9 / 2 := sorry
  exact h

end geometric_series_sum_l774_774969


namespace share_of_C_l774_774085

/-- Given the conditions:
  - Total investment is Rs. 120,000.
  - A's investment is Rs. 6,000 more than B's.
  - B's investment is Rs. 8,000 more than C's.
  - Profit distribution ratio among A, B, and C is 4:3:2.
  - Total profit is Rs. 50,000.
Prove that C's share of the profit is Rs. 11,111.11. -/
theorem share_of_C (total_investment : ℝ)
  (A_more_than_B : ℝ)
  (B_more_than_C : ℝ)
  (profit_distribution : ℝ)
  (total_profit : ℝ) :
  total_investment = 120000 →
  A_more_than_B = 6000 →
  B_more_than_C = 8000 →
  profit_distribution = 4 / 9 →
  total_profit = 50000 →
  ∃ (C_share : ℝ), C_share = 11111.11 :=
by
  sorry

end share_of_C_l774_774085


namespace exists_d_pos_l774_774274

/-- Assuming a sequence of positive real numbers (a_n) where 
    lim (a_n) as n -> ∞ is 0, and there exists a constant c > 0 
    such that |a_(n+1) - a_n| <= c * a_n^2 for all n >= 1, 
    show that there exists a constant d > 0 
    such that n * a_n >= d for all n >= 1. -/
theorem exists_d_pos (a : Nat → ℝ) (c : ℝ) (h1 : ∀ n, a n > 0)
  (h2 : Tendsto a atTop (𝓝 0))
  (h3 : ∀ n, |a (n + 1) - a n| ≤ c * (a n)^2) 
  (hc : 0 < c) : ∃ d > 0, ∀ n, n * a n ≥ d := 
sorry

end exists_d_pos_l774_774274


namespace two_values_for_g50_zero_l774_774285

-- Define g0 function
def g0 (x : ℝ) : ℝ :=
  if x < -150 then x + 300
  else if x < 150 then -x
  else x - 300

-- Define gn function recursively
def gn : ℕ → ℝ → ℝ
| 0, x => g0 x
| (n + 1), x => abs (gn n x) - 10

-- Main theorem statement
theorem two_values_for_g50_zero :
  (∃ x, gn 50 x = 0) ∧ ∀ x, gn 50 x = 0 → (x = -500 ∨ x = 800) :=
begin
  sorry
end

end two_values_for_g50_zero_l774_774285


namespace total_revenue_correct_l774_774822

noncomputable def revenue_calculation : ℕ :=
  let fair_tickets := 60
  let fair_price := 15
  let baseball_tickets := fair_tickets / 3
  let baseball_price := 10
  let play_tickets := 2 * fair_tickets
  let play_price := 12
  fair_tickets * fair_price
  + baseball_tickets * baseball_price
  + play_tickets * play_price

theorem total_revenue_correct : revenue_calculation = 2540 :=
  by
  sorry

end total_revenue_correct_l774_774822


namespace triangle_trig_identity_l774_774265

variables {A B C : ℝ} {a b c h : ℝ}

theorem triangle_trig_identity 
  (h_cond : c - a = h) : 
  sin ((C - A) / 2) + cos ((C + A) / 2) = 1 :=
sorry

end triangle_trig_identity_l774_774265


namespace double_summation_l774_774970

theorem double_summation :
  (∑ i in Finset.range 50, ∑ j in Finset.range 50, (2 * (i + 1) + 3 * (j + 1))) = 318750 := 
by
  sorry

end double_summation_l774_774970


namespace martha_black_butterflies_l774_774778

theorem martha_black_butterflies (total_butterflies blue_butterflies yellow_butterflies : ℕ)
  (h1 : total_butterflies = 11)
  (h2 : blue_butterflies = 4)
  (h3 : blue_butterflies = 2 * yellow_butterflies) :
  ∃ black_butterflies : ℕ, black_butterflies = total_butterflies - blue_butterflies - yellow_butterflies :=
sorry

end martha_black_butterflies_l774_774778


namespace trinomial_has_two_roots_l774_774911

theorem trinomial_has_two_roots (a b c : ℝ) (h : b^2 - 4 * a * c > 0) :
  let Δ := (2 * (a + b))^2 - 4 * 3 * a * (b + c) in Δ > 0 :=
by
  sorry

end trinomial_has_two_roots_l774_774911


namespace nested_log_eq_6322_l774_774986

noncomputable def nested_log : ℝ := 
  let f : ℝ → ℝ := λ x, real.logb 2 (64 + x) in
  real.lim (λ n, (f^[n] 0))

theorem nested_log_eq_6322 (ε : ℝ) (hε : 0 < ε) : abs (nested_log - 6.322) < ε :=
sorry

end nested_log_eq_6322_l774_774986


namespace find_sum_of_series_l774_774688

theorem find_sum_of_series (x : ℝ) (a a1 a2 : ℝ) :
  (∀ x : ℝ, (1 - 2 * x) ^ 2011 = a + a1 * x + a2 * x^2 + ∑ i in Finset.range 2011, (a2 * x^i)) →
  a = 1 →
  (a + a1 + a2 + ∑ i in Finset.range 2011, (a2)) = -1 →
  (a + a1) + (a + a2) + ∑ i in Finset.range 2011, (a + a2) = 2009 :=
by
  sorry

end find_sum_of_series_l774_774688


namespace solve_for_x_l774_774217

theorem solve_for_x (x : ℕ) (h : (1 / 8) * 2 ^ 36 = 8 ^ x) : x = 11 :=
by
sorry

end solve_for_x_l774_774217


namespace positive_integer_smaller_than_sqrt_two_l774_774418

theorem positive_integer_smaller_than_sqrt_two :
  ∃ n : ℕ, 0 < n ∧ n < real.sqrt 2 ∧ n = 1 :=
by
  sorry

end positive_integer_smaller_than_sqrt_two_l774_774418


namespace perfect_square_factors_count_450_l774_774674

theorem perfect_square_factors_count_450 : 
  (∃ n : ℕ, n = 4 ∧ (∀ d : ℕ, d ∣ 450 → ∃ k : ℕ, d = k * k)) := sorry

end perfect_square_factors_count_450_l774_774674


namespace negation_of_existential_l774_774356

def symmetrical_figure (T : Type) : Prop := 
  sorry -- define what it means to be a symmetrical figure

def triangle (T : Type) : Prop := 
  sorry -- define what it means to be a triangle

theorem negation_of_existential (T : Type) [triangle T] :
  ¬(∃ t : T, symmetrical_figure t) ↔ ∀ t : T, ¬symmetrical_figure t :=
by
  sorry

end negation_of_existential_l774_774356


namespace polynomial_root_range_l774_774139

theorem polynomial_root_range (x b : ℝ) (h : x^3 + b * x^2 - x + b = 0) : b ∈ set.Iic 0 := sorry

end polynomial_root_range_l774_774139


namespace equal_sum_sequence_S9_l774_774979

/-- Define an equal sum sequence with a common sum of 5 and initial term a₁ = 2. -/
def equal_sum_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n + a (n + 1) = 5

/-- The sum Sₙ of the first n terms of the sequence. -/
def sum_sequence (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, a i

/-- Given the conditions of the sequence, prove that S₉ = 22. -/
theorem equal_sum_sequence_S9 (a : ℕ → ℝ)
  (h_eq_sum_seq : equal_sum_sequence a)
  (h_a1 : a 1 = 2) :
  sum_sequence a 9 = 22 :=
sorry

end equal_sum_sequence_S9_l774_774979


namespace numPerfectSquareFactorsOf450_l774_774679

def isPerfectSquare (n : Nat) : Prop :=
  ∃ k : Nat, n = k * k

theorem numPerfectSquareFactorsOf450 : 
  ∃! n : Nat, 
    (∀ d : Nat, d ∣ 450 → isPerfectSquare d) → n = 4 := 
by
  sorry

end numPerfectSquareFactorsOf450_l774_774679


namespace power_inequality_l774_774572

theorem power_inequality (n : ℕ) (x : ℝ) (h1 : 0 < n) (h2 : x > -1) : (1 + x)^n ≥ 1 + n * x :=
sorry

end power_inequality_l774_774572


namespace perfect_square_divisors_count_450_l774_774641

theorem perfect_square_divisors_count_450 :
  let is_divisor (d n : ℕ) := n % d = 0
  let is_perfect_square (n : ℕ) := ∃ m : ℕ, m * m = n
  let prime_factorization_450 := [(2, 1), (3, 2), (5, 2)]
  ∃ (count : ℕ), count = 4 ∧ count = (Finset.filter 
    (λ d, is_perfect_square d ∧ is_divisor d 450)
    (Finset.powersetPrimeFactors prime_factorization_450)).card := sorry

end perfect_square_divisors_count_450_l774_774641


namespace number_of_valid_seatings_l774_774476

open List

def Person : Type := 
| Alice
| Bob
| Carla
| Derek
| Eric

def refuses_to_sit_next_to (p1 p2 : Person) : Prop :=
  (p1 = .Alice ∧ (p2 = .Bob ∨ p2 = .Carla)) ∨
  (p1 = .Carla ∧ (p2 = .Bob ∨ p2 = .Derek)) ∨
  (p1 = .Derek ∧ p2 = .Eric)

def valid_seating (seating : List Person) : Prop :=
  (seating.length = 5) ∧
  ∀ (i j : ℕ), i < seating.length → j < seating.length → |i - j| = 1 → ¬refuses_to_sit_next_to (seating.nthLe i (by sorry)) (seating.nthLe j (by sorry))

theorem number_of_valid_seatings : 
  (finset.univ.filter valid_seating).card = 12 := 
by
  sorry

end number_of_valid_seatings_l774_774476


namespace solve_equation_1_solve_equation_2_l774_774801

theorem solve_equation_1 :
  ∀ x : ℝ, x * (x + 1) = (x + 1) ↔ (x = -1 ∨ x = 1) :=
by
  intros x
  have h := eq_zero_or_eq_zero_of_mul_eq_zero (x + 1) ((x - 1))
  apply eq.symm (eq_add_zero x (-1)) at h.1
  apply eq_symm (eq_sub_zero x 1)) at h.2
  sorry

theorem solve_equation_2 :
  ∀ x : ℝ, 2 * x^2 - 3 * x - 1 = 0 ↔ (x = (3 + real.sqrt 17) / 4 ∨ x = (3 - real.sqrt 17) / 4) :=
by
  intros x
  let a := 2
  let b := -3
  let c := -1
  have D:= b^2 -4*a*c
  apply eq_if σ(( -b + real.sqrt D)) ((2*a) at x
  apply eq_if σ(( -b - real.sqrt D)) ((2*a) at x
  sorry

end solve_equation_1_solve_equation_2_l774_774801


namespace time_to_finish_by_p_l774_774428

theorem time_to_finish_by_p (P_rate Q_rate : ℝ) (worked_together_hours remaining_job_rate : ℝ) :
    P_rate = 1/3 ∧ Q_rate = 1/9 ∧ worked_together_hours = 2 ∧ remaining_job_rate = 1 - (worked_together_hours * (P_rate + Q_rate)) → 
    (remaining_job_rate / P_rate) * 60 = 20 := 
by
  sorry

end time_to_finish_by_p_l774_774428


namespace perfect_square_factors_450_l774_774657

def prime_factorization (n : ℕ) : Prop :=
  n = 2^1 * 3^2 * 5^2

theorem perfect_square_factors_450 : prime_factorization 450 → ∃ n : ℕ, n = 4 :=
by
  sorry

end perfect_square_factors_450_l774_774657


namespace construct_triangle_give_perimeter_height_angle_l774_774499

theorem construct_triangle_give_perimeter_height_angle
  (P h : ℝ) (α : ℝ)
  (hP : P > 0) (hh : h > 0) (hα : 0 < α ∧ α < π) :
  ∃ (A B C : ℝ × ℝ), 
    let AB := (A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2 in
    let hC := (A.1 - B.1) * (C.2 - (A.2 + B.2) / 2) in
    let angleA := ∠ABC in
    AB + ∥B - C∥ + ∥C - A∥ = P ∧
    hC = h ∧
    angleA = α :=
begin
  sorry
end

end construct_triangle_give_perimeter_height_angle_l774_774499


namespace initial_production_rate_l774_774948

variable (x : ℕ) (t : ℝ)

-- Conditions
def produces_initial (x : ℕ) (t : ℝ) : Prop := x * t = 60
def produces_subsequent : Prop := 60 * 1 = 60
def overall_average (t : ℝ) : Prop := 72 = 120 / (t + 1)

-- Goal: Prove the initial production rate
theorem initial_production_rate : 
  (∃ t : ℝ, produces_initial x t ∧ produces_subsequent ∧ overall_average t) → x = 90 := 
  by
    sorry

end initial_production_rate_l774_774948


namespace pow_mod_cycle_remainder_5_pow_2023_l774_774403

theorem pow_mod_cycle (n : ℕ) : (5^n % 6 = if n % 2 = 1 then 5 else 1) := 
by sorry

theorem remainder_5_pow_2023 : 5^2023 % 6 = 5 :=
by
  have cycle_properties : ∀ n, 5^n % 6 = (if n % 2 = 1 then 5 else 1) := pow_mod_cycle
  calc
    5^2023 % 6 = if 2023 % 2 = 1 then 5 else 1 := cycle_properties 2023
             ... = 5                   := by norm_num

end pow_mod_cycle_remainder_5_pow_2023_l774_774403


namespace abc_inequality_l774_774750

variable {a b c : ℝ}

theorem abc_inequality (h₀ : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h₁ : a + b + c = 6)
  (h₂ : a * b + b * c + c * a = 9) :
  0 < a * b * c ∧ a * b * c < 4 := by
  sorry

end abc_inequality_l774_774750


namespace find_ordered_pair_l774_774523

open Polynomial

theorem find_ordered_pair (a b : ℝ) :
  (∀ x : ℝ, (((x^3 + a * x^2 + 17 * x + 10 = 0) ∧ (x^3 + b * x^2 + 20 * x + 12 = 0)) → 
  (x = -6 ∧ y = -7))) :=
sorry

end find_ordered_pair_l774_774523


namespace mutual_acquaintance_exists_l774_774383

-- G represent the gathering graph
variable {G : Type*} [Graph G]
variable [DecidableRel (Graph.adj : G → G → Prop)]

noncomputable def at_least_k (n : ℕ) : Prop := 
  ∀ v : G, (Card (Graph.neighbors v) ≥ ⌊n / 2⌋)

noncomputable def condition_of_any (n : ℕ) : Prop := 
  ∀ S ⊆ vertex_set G, Card S = ⌊n / 2⌋ → 
    (∃ x ∈ S, ∃ y ∈ S, x ≠ y ∧ Graph.adj x y) ∨ 
    (∃ x ∈ (vertex_set G \ S), ∃ y ∈ (vertex_set G \ S), x ≠ y ∧ Graph.adj x y)

theorem mutual_acquaintance_exists (n : ℕ) (h1 : n ≥ 6) 
  (h2 : at_least_k n) (h3 : condition_of_any n) :
  ∃ (a b c : G), a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ Graph.adj a b ∧ Graph.adj b c ∧ Graph.adj a c :=
sorry

end mutual_acquaintance_exists_l774_774383


namespace correct_statistical_survey_statement_l774_774414

-- Define the conditions as Lean definitions
def condition_A : Prop := ¬ (∀ survey, comprehensive_survey survey)
def condition_B : Prop := ∀ students, comprehensive_survey (conduct_vision_tests students)
def condition_C : Prop := ∀ (households samples : ℕ), (samples ≠ households) → (sample_size households samples = samples)
def condition_D : Prop := ∀ students basketball_team, has_bias (sample_height basketball_team) students

-- Define the question
def question : Prop := ∀ statements : list Prop, (statements = [condition_A, condition_B, condition_C, condition_D]) →
  (correct_statement_in_statements statements = condition_B)

-- The proof statement
theorem correct_statistical_survey_statement : question :=
sorry

end correct_statistical_survey_statement_l774_774414


namespace limit_ln_pn_div_n_l774_774007

def balls_and_boxes_limit (n : ℕ) : ℝ :=
  let pn := (1 - 1 / (2 * n)) ^ n + n * (1 - 1 / (2 * n)) ^ (n - 1) * (1 / (2 * n)) in
  Real.log pn / n

theorem limit_ln_pn_div_n :
  ∀ n : ℕ, n > 0 → 
  ∃ lim : ℝ, 
    filter.eventually (λ n, balls_and_boxes_limit n = lim) filter.at_top :=
begin
  intros n hn,
  use 0,
  sorry
end

end limit_ln_pn_div_n_l774_774007


namespace number_of_perfect_square_divisors_of_450_l774_774638

theorem number_of_perfect_square_divisors_of_450 : 
    let p := 450;
    let factors := [(3, 2), (5, 2), (2, 1)];
    ∃ n, (n = 4 ∧ 
          ∀ (d : ℕ), d ∣ p → 
                     (∃ (a b c : ℕ), d = 2^a * 3^b * 5^c ∧ 
                              (a = 0) ∧ (b = 0 ∨ b = 2) ∧ (c = 0 ∨ c = 2) → 
                              a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 = 0)) :=
    sorry

end number_of_perfect_square_divisors_of_450_l774_774638


namespace solve_linear_equation_l774_774374

theorem solve_linear_equation : ∀ x : ℝ, 4 * (2 * x - 1) = 1 - 3 * (x + 2) → x = -1 / 11 :=
by
  intro x h
  -- Proof to be filled in
  sorry

end solve_linear_equation_l774_774374


namespace semicircle_perimeter_approx_l774_774076

noncomputable def approximate_perimeter_of_semicircle (r : ℝ) (pi_approx : ℝ) : ℝ :=
  pi_approx * r + 2 * r

theorem semicircle_perimeter_approx :
  approximate_perimeter_of_semicircle 11 3.14159 ≈ 56.56 :=
by
  sorry

end semicircle_perimeter_approx_l774_774076


namespace perfect_square_factors_450_l774_774611

theorem perfect_square_factors_450 :
  ∃ (n : ℕ), n = 4 ∧ ∀ (d : ℕ), d ∣ 450 → ∃ (k : ℕ), k^2 = d ↔ d = 1 ∨ d = 9 ∨ d = 25 ∨ d = 225 :=
by
  sorry

end perfect_square_factors_450_l774_774611


namespace midpoints_concyclic_l774_774275

-- Assuming basic definitions for points and cyclicity
structure Point :=
  (x : ℝ)
  (y : ℝ)

def is_midpoint (P A B : Point) : Prop :=
  P.x = (A.x + B.x) / 2 ∧ P.y = (A.y + B.y) / 2

def is_concyclic (P Q R S : Point) : Prop :=
  -- TODO: This definition needs a specific implementation, 
  -- but for now, we'll use it as a placeholder
  sorry

theorem midpoints_concyclic {A B C D : Point}
  (h₁ : A.x = 0 ∧ A.y = 0)
  (h₂ : B.x > 0 ∧ B.y = 0)
  (h₃ : ∠A B C = 90 ∧ ∠C D A = 90)
  (h₄ : is_concyclic A D (Point.mk ((B.x + A.x) / 2) ((B.y + A.y) / 2)) (Point.mk ((B.x + C.x) / 2) ((B.y + C.y) / 2))) :
  is_concyclic (Point.mk (A.x) (A.y + D.y / 2)) (Point.mk (C.x / 2) (C.y / 2)) B C :=
by
  sorry

end midpoints_concyclic_l774_774275


namespace number_of_real_roots_eq_3_eq_m_l774_774157

theorem number_of_real_roots_eq_3_eq_m {x m : ℝ} (h : ∀ x, x^2 - 2 * |x| + 2 = m) : m = 2 :=
sorry

end number_of_real_roots_eq_3_eq_m_l774_774157


namespace probability_left_shoe_l774_774022

theorem probability_left_shoe (total_pairs : ℕ) (h : total_pairs = 3) : 
  let left_shoes := 3 in
  let total_shoes := total_pairs * 2 in
  (left_shoes : ℚ) / total_shoes = 1 / 2 :=
by
  sorry

end probability_left_shoe_l774_774022


namespace prob_f_x0_leq_0_l774_774977

def f (x : ℝ) : ℝ := x^2 - 2 * x - 3

theorem prob_f_x0_leq_0 :
  ∀ (x0 : ℝ), x0 ∈ set.Icc (-4:ℝ) 4 → 
  (set.measure (λ x, f x ≤ 0) (set.Icc (-4:ℝ) 4)).toReal / 
  (set.measure (set.Icc (-4:ℝ) 4)).toReal = 1 / 2 :=
by
  sorry

end prob_f_x0_leq_0_l774_774977


namespace quotient_remainder_difference_l774_774067

theorem quotient_remainder_difference :
  ∀ (N Q Q' R : ℕ), 
    N = 75 →
    N = 5 * Q →
    N = 34 * Q' + R →
    Q > R →
    Q - R = 8 :=
by
  intros N Q Q' R hN hDiv5 hDiv34 hGt
  sorry

end quotient_remainder_difference_l774_774067


namespace exists_large_integers_satisfying_equation_l774_774794

theorem exists_large_integers_satisfying_equation :
  ∃ a b c d : ℤ, (|a| > 1000000) ∧ (|b| > 1000000) ∧ (|c| > 1000000) ∧ (|d| > 1000000) ∧
  (1 / a + 1 / b + 1 / c + 1 / d = 1 / (a * b * c * d)) :=
by
  let n := 1000001 -- n is any integer greater than 1,000,000.
  let a := -n
  let b := n + 1
  let c := n * (n + 1) + 1
  let d := n * (n + 1) * (n * (n + 1) + 1) + 1
  use [a, b, c, d]
  have h_abs_a : |a| > 1000000 := by sorry
  have h_abs_b : |b| > 1000000 := by sorry
  have h_abs_c : |c| > 1000000 := by sorry
  have h_abs_d : |d| > 1000000 := by sorry
  have h_eq : (1 / a + 1 / b + 1 / c + 1 / d = 1 / (a * b * c * d)) := by sorry
  exact ⟨h_abs_a, h_abs_b, h_abs_c, h_abs_d, h_eq⟩

end exists_large_integers_satisfying_equation_l774_774794


namespace rebus_solution_exists_l774_774406

theorem rebus_solution_exists:
    ∃ (A B C D E F G H I J : ℕ),
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧ A ≠ G ∧ A ≠ H ∧ A ≠ I ∧ A ≠ J ∧
    B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧ B ≠ G ∧ B ≠ H ∧ B ≠ I ∧ B ≠ J ∧
    C ≠ D ∧ C ≠ E ∧ C ≠ F ∧ C ≠ G ∧ C ≠ H ∧ C ≠ I ∧ C ≠ J ∧
    D ≠ E ∧ D ≠ F ∧ D ≠ G ∧ D ≠ H ∧ D ≠ I ∧ D ≠ J ∧
    E ≠ F ∧ E ≠ G ∧ E ≠ H ∧ E ≠ I ∧ E ≠ J ∧
    F ≠ G ∧ F ≠ H ∧ F ≠ I ∧ F ≠ J ∧
    G ≠ H ∧ G ≠ I ∧ G ≠ J ∧
    H ≠ I ∧ H ≠ J ∧
    I ≠ J ∧
    100 * A + 10 * B + C + 100 * D + 10 * E + F = 1000 * G + 100 * H + 10 * I + J
:=
    ⟨8, 7, 9, 4, 2, 6, 1, 3, 0, 5,
    by repeat {dec_trivial}⟩

end rebus_solution_exists_l774_774406


namespace tom_seashells_left_l774_774850

def initial_seashells : ℕ := 5
def given_away_seashells : ℕ := 2

theorem tom_seashells_left : (initial_seashells - given_away_seashells) = 3 :=
by
  sorry

end tom_seashells_left_l774_774850


namespace quadratic_trinomial_has_two_roots_l774_774921

theorem quadratic_trinomial_has_two_roots
  (a b c : ℝ) (h : b^2 - 4 * a * c > 0) : (2 * (a + b))^2 - 4 * 3 * a * (b + c) > 0 := by
  sorry

end quadratic_trinomial_has_two_roots_l774_774921


namespace sum_inequality_l774_774754

theorem sum_inequality (n : ℕ) (a : ℝ) (λ : ℝ) (x : ℕ → ℝ)
  (h1 : 0 ≤ λ) (h2 : λ ≤ 2)
  (h3 : ∀ i : ℕ, 1 ≤ i → i ≤ n → 0 < x i) 
  (h4 : ∀ i : ℕ, 1 ≤ i → i ≤ n → x i ≤ a) :
  ∑ i in Finset.range n, (1 + (i : ℝ) * λ) * x (i + 1) 
  ≥ (2 + (n - 1 : ℝ) * λ) / (2 * a * n : ℝ) * (∑ i in Finset.range n, x (i + 1)) ^ 2 :=
sorry

end sum_inequality_l774_774754


namespace least_of_10_consecutive_odd_integers_average_154_l774_774694

theorem least_of_10_consecutive_odd_integers_average_154 (x : ℤ)
  (h_avg : (x + (x + 2) + (x + 4) + (x + 6) + (x + 8) + (x + 10) + (x + 12) + (x + 14) + (x + 16) + (x + 18)) / 10 = 154) :
  x = 145 :=
by 
  sorry

end least_of_10_consecutive_odd_integers_average_154_l774_774694


namespace angle_ABE_eq_angle_CBD_l774_774788

theorem angle_ABE_eq_angle_CBD 
  (A B C D P Q E : Point)
  (hP : P ∈ Segment A D)
  (hQ : Q ∈ Segment D C)
  (hE : E ∈ (Line A Q) ∩ (Line C P))
  (hAngle : ∠ A B P = ∠ C B Q) :
  ∠ A B E = ∠ C B D :=
sorry

end angle_ABE_eq_angle_CBD_l774_774788


namespace three_times_two_to_the_n_minus_one_gt_n_squared_plus_three_l774_774335

theorem three_times_two_to_the_n_minus_one_gt_n_squared_plus_three (n : ℕ) (h : n ≥ 4) : 3 * 2^(n-1) > n^2 + 3 := by
  sorry

end three_times_two_to_the_n_minus_one_gt_n_squared_plus_three_l774_774335


namespace sum_invested_7000_l774_774445

-- Define the conditions
def interest_15 (P : ℝ) : ℝ := P * 0.15 * 2
def interest_12 (P : ℝ) : ℝ := P * 0.12 * 2

-- Main statement to prove
theorem sum_invested_7000 (P : ℝ) (h : interest_15 P - interest_12 P = 420) : P = 7000 := by
  sorry

end sum_invested_7000_l774_774445


namespace roots_quadratic_eq_identity1_roots_quadratic_eq_identity2_l774_774792

variables {α : Type*} [Field α] (a b c x1 x2 : α)

theorem roots_quadratic_eq_identity1 (h_eq_roots: ∀ x, a * x^2 + b * x + c = 0 → (x = x1 ∨ x = x2)) 
(h_root1: a * x1^2 + b * x1 + c = 0) (h_root2: a * x2^2 + b * x2 + c = 0) :
  x1^2 + x2^2 = (b^2 - 2 * a * c) / a^2 :=
sorry

theorem roots_quadratic_eq_identity2 (h_eq_roots: ∀ x, a * x^2 + b * x + c = 0 → (x = x1 ∨ x = x2)) 
(h_root1: a * x1^2 + b * x1 + c = 0) (h_root2: a * x2^2 + b * x2 + c = 0) :
  x1^3 + x2^3 = (3 * a * b * c - b^3) / a^3 :=
sorry

end roots_quadratic_eq_identity1_roots_quadratic_eq_identity2_l774_774792


namespace perfect_square_divisors_of_450_l774_774621

theorem perfect_square_divisors_of_450 :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let e1 := 1
  let e2 := 2
  let e3 := 2
  ∃ (count : ℕ), count = 4 ∧
  (∀ (d : ℕ), d ∣ (p1^e1 * p2^e2 * p3^e3) →
    (∀ (pe : ℕ × ℕ), pe ∈ (Multiset.ofList [(p1, e1), (p2, e2), (p3, e3)]).powerset → 
      (∃ (ex1 ex2 ex3: ℕ), ex1 ≤ e1 ∧ ex2 ≤ e2 ∧ ex3 ≤ e3 ∧ ex1 % 2 = 0 ∧ ex2 % 2 = 0 ∧ ex3 % 2 = 0))) :=
sorry

end perfect_square_divisors_of_450_l774_774621


namespace smallest_modulus_complex_l774_774767

theorem smallest_modulus_complex (z : ℂ) (h : |z - 10| + |z + 3 * complex.I| = 15) : |z| = 2 :=
sorry

end smallest_modulus_complex_l774_774767


namespace inequality_solution_l774_774140

noncomputable def quadratic_roots : Set ℝ :=
  let a := (8 / 5) * (1 + (1 / (10 * (2.sqrt)))) in
  let b := (8 / 5) * (1 - (1 / (10 * (2.sqrt)))) in
  {x : ℝ | x ∈ (set.Ioo a.real (-0)) ∪ set.Ioo (-0) b.real }

theorem inequality_solution (x : ℝ) (hx : x ≠ 0) :
  (∃ a b : ℝ, a = ((8 / 5) * (1 + (1 / (10 * (2.sqrt))))) ∧ b = ((8 / 5) * (1 - (1 / (10 * (2.sqrt))))) ∧
  ((1 / (x^2 + 2) + 1/2) > (5 / x + 21 / 10)) ↔ x ∈ (set.Ioo a (-0)) ∪ (set.Ioo (-0) b)) :=
begin
  sorry
end

end inequality_solution_l774_774140


namespace jordan_trip_shorter_percentage_l774_774462

theorem jordan_trip_shorter_percentage (w : ℝ) (h : w > 0) :
  let C := 6 * w in
  let d := w * Real.sqrt 5 in
  ((C - d) / C) * 100 ≈ 63 := by
  sorry

end jordan_trip_shorter_percentage_l774_774462


namespace prop_p_implies_m_exactly_one_prop_p_q_implies_m_l774_774571

variables {x m : ℝ}

-- Condition 1: Proposition p
def proposition_p := ∀ x, -3 < x ∧ x < 1 → x^2 + 4 * x + 9 - m > 0

-- Condition 2: Proposition q
def proposition_q := ∃ x, 0 < x ∧ x^2 - 2 * m * x + 1 < 0

-- Part 1: Prove range of m when proposition p is true
theorem prop_p_implies_m :
  proposition_p → m < 5 :=
sorry

-- Part 2: Prove range of m when exactly one of propositions p and q is true
theorem exactly_one_prop_p_q_implies_m :
  (proposition_p ∧ ¬ proposition_q ∨ ¬ proposition_p ∧ proposition_q) → m ∈ Set.Iic 1 ∪ Set.Ici 5 :=
sorry

end prop_p_implies_m_exactly_one_prop_p_q_implies_m_l774_774571


namespace factorial_division_l774_774490

noncomputable def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

theorem factorial_division :
  (factorial 10) / ((factorial 7) * (factorial 3)) = 120 := by sorry

end factorial_division_l774_774490


namespace find_radius_l774_774722

noncomputable def Point := (ℝ × ℝ)

def center : Point := (-2, -3)
def inside_point : Point := (-2, 3)
def outside_point : Point := (6, -3)
def on_circle_point : Point := (6, 3)
def tangent_point : Point := (0, -3)

def radius (r : ℤ) : Prop :=
  ∃ (A B C D E : Point),
  A = center ∧ 
  B = inside_point ∧ 
  C = outside_point ∧ 
  D = on_circle_point ∧ 
  E = tangent_point ∧
  dist A E = r ∧
  dist B A < r ∧
  dist C A > r ∧
  is_isosceles_triangle A C D ∧ 
  dist D A = dist D C ∧
  ((dist A C) * (dist D A) / 2 : ℤ) = triangle_area A C D

theorem find_radius : radius 2 :=
sorry

end find_radius_l774_774722


namespace mul_104_96_l774_774976

theorem mul_104_96 : 104 * 96 = 9984 := by
  calc
    104 * 96 = (100 + 4) * (100 - 4) : by rw [← rfl, ← rfl]
    ... = 100^2 - 4^2 : by apply mul_self_sub_mul_self
    ... = 10000 - 16 : by rw [pow_two, pow_two]
    ... = 9984 : by norm_num


end mul_104_96_l774_774976


namespace jackie_eligible_for_free_shipping_l774_774269

def shampoo_cost : ℝ := 2 * 12.50
def conditioner_cost : ℝ := 3 * 15.00
def face_cream_cost : ℝ := 20.00  -- Considering the buy-one-get-one-free deal

def subtotal : ℝ := shampoo_cost + conditioner_cost + face_cream_cost
def discount : ℝ := 0.10 * subtotal
def total_after_discount : ℝ := subtotal - discount

theorem jackie_eligible_for_free_shipping : total_after_discount >= 75 := by
  sorry

end jackie_eligible_for_free_shipping_l774_774269


namespace sum_of_coefficients_l774_774530

def polynomial : (ℤ → ℤ) := λ x, -3*(x^8 - 2*x^5 + 4*x^3 - 6) + 5*(x^4 + 3*x^2) - 2*(x^6 - 5)

theorem sum_of_coefficients :
  polynomial 1 = 37 := by
  sorry

end sum_of_coefficients_l774_774530


namespace find_b_and_compare_f_l774_774696

noncomputable def f (x : ℝ) (b : ℝ) (c : ℝ) := -x^2 + b * x + c

theorem find_b_and_compare_f :
  ∀ (c : ℝ), (∀ x : ℝ, f (2 + x) 4 c = f (2 - x) 4 c) →
  (∃ b : ℝ, b = 4) ∧ (∀ a : ℝ, f (5/4) 4 c ≥ f (-a^2 - a + 1) 4 c) 
  ∧ (∀ a : ℝ, a = -1/2 → f (5/4) 4 c = f (-a^2 - a + 1) 4 c) :=
begin
  sorry
end

end find_b_and_compare_f_l774_774696


namespace volume_of_solid_of_revolution_l774_774196

noncomputable def f : ℝ → ℝ
| x := if 0 ≤ x ∧ x ≤ 1 then 2*x else sqrt (-x^2 + 2*x + 3)

theorem volume_of_solid_of_revolution :
  ∀ V : ℝ, (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = 2 * x) ∧ 
           (∀ x : ℝ, 1 < x ∧ x ≤ 3 → f x = sqrt (-x^2 + 2*x + 3)) →
           V = (1/3 * π * 2^2 * 1 + 1/2 * 4/3 * π * 2^3) →
           V = (20 * π) / 3 := by
  sorry

end volume_of_solid_of_revolution_l774_774196


namespace perfect_square_divisors_of_450_l774_774615

theorem perfect_square_divisors_of_450 :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let e1 := 1
  let e2 := 2
  let e3 := 2
  ∃ (count : ℕ), count = 4 ∧
  (∀ (d : ℕ), d ∣ (p1^e1 * p2^e2 * p3^e3) →
    (∀ (pe : ℕ × ℕ), pe ∈ (Multiset.ofList [(p1, e1), (p2, e2), (p3, e3)]).powerset → 
      (∃ (ex1 ex2 ex3: ℕ), ex1 ≤ e1 ∧ ex2 ≤ e2 ∧ ex3 ≤ e3 ∧ ex1 % 2 = 0 ∧ ex2 % 2 = 0 ∧ ex3 % 2 = 0))) :=
sorry

end perfect_square_divisors_of_450_l774_774615


namespace hyperbola_focal_length_l774_774816

-- Definitions of conditions from the problem
variables (a b c : ℝ)
variables (ha : a > 0) (hb : b > 0)
variable (eccentricity : ℝ) (h_e : eccentricity = 2)
variable (distance_to_asymptote : ℝ) (h_d : distance_to_asymptote = sqrt 3)

-- The question translated into a proof
theorem hyperbola_focal_length :
  (∃ c : ℝ, eccentricity = c / a ∧ distance_to_asymptote = ((b * c) / (sqrt (a ^ 2 + b ^ 2)))) → 2 * c = 4 :=
by
  sorry

end hyperbola_focal_length_l774_774816


namespace percent_divisibles_l774_774866

def count_divisibles (n k : ℕ) : ℕ :=
  (n / k)

theorem percent_divisibles (n k total : ℕ) (h : n = 200) (hk : k = 21) 
  (htotal : total = count_divisibles n k) : 
  total * 100 / n = 4.5 := by
  sorry

end percent_divisibles_l774_774866


namespace radius_of_tangent_circle_correct_l774_774773

noncomputable def radius_of_tangent_circle (ABC : Triangle) (A_b B_a : Point) 
  (A_c C_a : Point) (B_c C_b : Point) : ℝ :=
  if h : ABC.is_equilateral ∧ ABC.side_length = 15 ∧
      (Triangle.mk' A_b A_c ABC.A).is_equilateral ∧ (Triangle.mk' A_b A_c ABC.A).side_length = 3 ∧
      (Triangle.mk' B_a B_c ABC.B).is_equilateral ∧ (Triangle.mk' B_a B_c ABC.B).side_length = 4 ∧
      (Triangle.mk' C_b C_a ABC.C).is_equilateral ∧ (Triangle.mk' C_b C_a ABC.C).side_length = 5
  then 3 * Real.sqrt 3
  else 0

theorem radius_of_tangent_circle_correct (ABC : Triangle) (A_b B_a : Point) 
  (A_c C_a : Point) (B_c C_b : Point) :
  ABC.is_equilateral ∧ ABC.side_length = 15 ∧
  (Triangle.mk' A_b A_c ABC.A).is_equilateral ∧ (Triangle.mk' A_b A_c ABC.A).side_length = 3 ∧
  (Triangle.mk' B_a B_c ABC.B).is_equilateral ∧ (Triangle.mk' B_a B_c ABC.B).side_length = 4 ∧
  (Triangle.mk' C_b C_a ABC.C).is_equilateral ∧ (Triangle.mk' C_b C_a ABC.C).side_length = 5 →
  radius_of_tangent_circle ABC A_b B_a A_c C_a B_c C_b = 3 * Real.sqrt 3 :=
begin
  -- Proof goes here
  sorry
end

end radius_of_tangent_circle_correct_l774_774773


namespace tangent_line_at_x0_l774_774537

-- Define the curve function
def curve (x : ℝ) : ℝ := 14 * real.sqrt x - 15 * real.cbrt x + 2

-- Define the tangent line function
def tangent_line (x : ℝ) : ℝ := 2 * x - 1

-- Define the point of tangency
def x0 : ℝ := 1

-- Define the point-slope form result at x = x0
def point (x : ℝ) : ℝ := curve x0

-- Prove the equation of the tangent line at x = x0
theorem tangent_line_at_x0 : (tangent_line x0) = curve x0 :=
by
  sorry

end tangent_line_at_x0_l774_774537


namespace concurrency_of_transversals_l774_774375

theorem concurrency_of_transversals 
  {A B C A1 B1 C1 A2 B2 C2: Type}
  [triangle : Triangle A B C]
  [concurrent : Concurrency (A1 B1 C1)]
  [circle : Circle Thru A1 B1 C1]
  (intersect_A2 : Intersect (circle) (BC) = A2)
  (intersect_B2 : Intersect (circle) (CA) = B2)
  (intersect_C2 : Intersect (circle) (AB) = C2) :
  Concurrency (A2 B2 C2) :=
by
  sorry

end concurrency_of_transversals_l774_774375


namespace concurrency_of_circumcircles_l774_774826

-- Definition and conditions
variables {A B C D E M N P Q : Type} [Quadrilateral A B C D]

-- Points P and Q lie on the diagonals AC and BD respectively
variables (hP : is_on_diagonal P A C)
variables (hQ : is_on_diagonal Q B D)

-- Line PQ meets the sides AD and BC at points M and N respectively
variables (hPQ_AD : intersects PQ AD M)
variables (hPQ_BC : intersects PQ BC N)

-- Condition: (AP / AC) + (BQ / BD) = 1
variable (h_ratio : (segment_ratio A P C) + (segment_ratio B Q D) = 1)

-- Theorem statement: The circumcircles of triangles AMP, BNQ, DMQ, and CNP are concurrent
theorem concurrency_of_circumcircles :
  concurent_circles (circumcircle_of_triangle A M P) 
                                (circumcircle_of_triangle B N Q) 
                                (circumcircle_of_triangle D M Q) 
                                (circumcircle_of_triangle C N P) := 
sorry

end concurrency_of_circumcircles_l774_774826


namespace initial_amount_l774_774424

theorem initial_amount (x : ℝ) (h : 0.015 * x = 750) : x = 50000 :=
by
  sorry

end initial_amount_l774_774424


namespace find_m_n_g_monotonic_intervals_l774_774587

-- Define the function f and its derivative
def f (x m n: ℝ) := x^3 + 3*m*x^2 + n*x
def f_prime (x m n: ℝ) := 3*x^2 + 6*m*x + n

-- Define the function g based on f
def g (x m n: ℝ) := f x m n - x^3 - 3*log x

-- Prove that m = 2/3 and n = 1 given the conditions
theorem find_m_n (h1 : f (-1) m n = 0) (h2 : f_prime (-1) m n = 0) : m = 2/3 ∧ n = 1 :=
by sorry

-- Prove the monotonic intervals of g
theorem g_monotonic_intervals (m n : ℝ) (hmn : m = 2/3 ∧ n = 1) :
  (∀ x, 0 < x ∧ x < 3/4 → g' x m n < 0) ∧ (∀ x, x > 3/4 → g' x m n > 0) :=
by sorry

end find_m_n_g_monotonic_intervals_l774_774587


namespace range_of_f_l774_774527

def f (x : ℝ) : ℝ := (x + 1) / (x^2 + 1)

theorem range_of_f :
  Set.Icc 0 0.6 = {y : ℝ | ∃ x : ℝ, y = f x} :=
sorry

end range_of_f_l774_774527


namespace coordinates_reflect_y_axis_l774_774718

def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

theorem coordinates_reflect_y_axis (p : ℝ × ℝ) (h : p = (5, 2)) : reflect_y_axis p = (-5, 2) :=
by
  rw [h]
  rfl

end coordinates_reflect_y_axis_l774_774718


namespace fewest_posts_l774_774935

def grazingAreaPosts (length width post_interval rock_wall_length : ℕ) : ℕ :=
  let side1 := width / post_interval + 1
  let side2 := length / post_interval
  side1 + 2 * side2

theorem fewest_posts (length width post_interval rock_wall_length posts : ℕ) :
  length = 70 ∧ width = 50 ∧ post_interval = 10 ∧ rock_wall_length = 150 ∧ posts = 18 →
  grazingAreaPosts length width post_interval rock_wall_length = posts := 
by
  intros h
  obtain ⟨hl, hw, hp, hr, ht⟩ := h
  simp [grazingAreaPosts, hl, hw, hp, hr]
  sorry

end fewest_posts_l774_774935


namespace max_min_area_difference_l774_774385

theorem max_min_area_difference (l w : ℝ) (hlw : l + w = 100) :
  let A := l * (100 - l) in
  let A_max := 2500 in
  let A_min := 99 in
  (A_max - A_min) = 2401 := 
by {
  sorry
}

end max_min_area_difference_l774_774385


namespace perfect_square_factors_450_l774_774629

theorem perfect_square_factors_450 : 
  ∃ n, n = 4 ∧ (∀ k | 450, k ∣ 450 → is_square k) → (∃ m1 m2 m3 : ℕ, 450 = 2^m1 * 3^m2 * 5^m3 ) :=
by sorry

end perfect_square_factors_450_l774_774629


namespace EM_median_length_of_triangle_Lean4EM_length_CM_l774_774461

variables (A B C D E M K : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
  [MetricSpace D] [MetricSpace E] [MetricSpace M] [MetricSpace K]
  (circumscribed : ∀ {a b c d : Type}, Set (MetricSpace (a + b + c + d))
  (inscribed_quad : ∀ {A B C D : Type}, circumscribed A B C D)
  (AD : ℝ) (AB : ℝ) 
  (α : ℝ)
  (circumscribed_perpendicular_intersect : a = ∈A ∩ B (MetricSpace C + D ))

theorem EM_median_length_of_triangle (h1: circumscribed)
  (h2: @circumscribed_perpendicular_intersect α A B C D E)
  (h3: (α: α α α) )
  (h4: ∈circumscribed (∀α , H ((A, B: point). E→ α) α α α) : MetricSpace GH,
  (perpendicular_E_AB : Set (MetricSpace LinearAB α α A)) α α |
  (→triangle O {AD = ZBot }{ AB : α { }}}isGivenCondition == ∈name.isGivenCondition )):

   1.  setor : circle (right_edge = ( MetricSpace isGivenCondition ) α α α α  α α α  (DE := sqrt { A }))) = generalized.acolal : concl
       Facts provided): conc ¯ α circle neighbor.isGivenCondition α)
       ∈var 
       let κοινός AD := 8 
       let j := ( (isMetricSpaceGeneralized) B (Platonic_fractal αconc_tβ ) ; letenoscaleFlag := 8 )

  |-  αth_to_abc_circle αconcis Given  αAB αα (- MetricSpace isGivenCondition tactics optimization = ; Placement αα;)
  
  ∑ A B C D α circle_whenc αperpendicular αcircumference;
entire.scaleFlag α α α α α α. generalized arithmetic E α α  α α α ) :

∀ Given ααintersection Median_of_set_perpendicular_P isGiven_ MetricSpace_generalized,
 EM_ → α Median α_linear_tri_equ_part_set αproof_linear_dimension is,  ": 

theorem Lean4EM_length_CM:
(noncomputable def  generalizedMedianProofPartEqn α → Median:= ) :
(@LinearProofPredicate topical.unique_alpha_triple_circle metricsoid _S αα ∀ (A ={subCircles_}.  MetricSpace.maximum α α αα αaatitcip Median α: A ; Median_generalize_medium_eq  :

noncomputable def EM :
=
1  MetricSpace αβ ααα TriangularAlpha αα ααα : ∀ generalized α α 
den_ad eq_sat Sorted_ACircle := TopologicalSetsoid
  { Metric generalized_A. SetConditional α} 
  → ( αβmedian_linearization Uniqueθ { Median).
    LinearMidSpace: ∀  Median proof α∃ ;
;
meta αα α
EM 

∃( 01234)
=      ∃GeneralMeasurement MetricAlphaGeneral cosine ααMetricSpace α A B C D M



prod (∃ EMtd model): circle_length_eq_segment 
(
∃intersect_angle_auto ∀ αα 
 perpendicular)

content!  Topological proofOf
lemma
 (
generalized_provided_combined ACircle 
circ_quad_triangle 

initialize_generalization Metric


median_ACircle :
EM_median_length_of_triangle_def := 
circle
ax.: {8 
  )  
 Dec_length_integral_alloc=" Metric Angle α) theorem solve_generalizedM_circ

 sorry qproof αProof  axiom_basic_stat M intersect αproving α median_A ∘   
α:
  α) 
Reflex @"η 
eq_proof αEM_: 
α    = 
  Metric  := LinearMed νprojection_cos_metric 
Metric_is αgeneralize combinator_ix_general_formal 


end EM_median_length_of_triangle_Lean4EM_length_CM_l774_774461


namespace cylinder_radius_l774_774905

theorem cylinder_radius (shadow_length_cylinder : ℝ) (height_flagpole : ℝ) (shadow_length_flagpole : ℝ) (parallel_sun_rays : Prop) (flagpole_line_segment : Prop) :
  shadow_length_cylinder = 12 → height_flagpole = 1.5 → shadow_length_flagpole = 3 → parallel_sun_rays → flagpole_line_segment → 
  ∃ r : ℝ, r = 6 :=
by
  intros h_cylinder_shadow h_flagpole_height h_flagpole_shadow h_sun_rays h_flagpole_segment
  have h_tan_theta : tan (θ : ℝ) = 0.5 := by sorry
  have h_tan_cylinder : tan θ = r / 12 := by sorry
  have h_radius : r = 6 := by sorry
  existsi 6
  exact h_radius

end cylinder_radius_l774_774905


namespace side_length_ratio_sum_l774_774365

theorem side_length_ratio_sum (a b c : ℕ) (h : (a + b + c = 11)) :
  let area_ratio := (18 : ℚ) / (98 : ℚ)
  let side_length_ratio := (a * real.sqrt b) / c
  sqrt(area_ratio) = side_length_ratio := 
sorry

end side_length_ratio_sum_l774_774365


namespace complex_div_identity_l774_774184

theorem complex_div_identity :
  (∃ i : ℂ, i^2 = -1) →
  (∃ i : ℂ, ∀ h: i^2 = -1, (i / (1 - i) = -1/2 + (1/2)*i)) :=
by
  intro h
  cases h with i hi
  use i
  intro hi
  sorry

end complex_div_identity_l774_774184


namespace correct_operation_l774_774944

variable (a b : ℝ)

theorem correct_operation : (-2 * a ^ 2) ^ 2 = 4 * a ^ 4 := by
  sorry

end correct_operation_l774_774944


namespace fencing_required_l774_774421

theorem fencing_required (L W A F : ℝ) (hL : L = 20) (hA : A = 390) (hArea : A = L * W) (hF : F = 2 * W + L) : F = 59 :=
by
  sorry

end fencing_required_l774_774421


namespace volume_after_increase_l774_774927

variable (l w h : ℝ)

def original_volume (l w h : ℝ) : ℝ := l * w * h
def original_surface_area (l w h : ℝ) : ℝ := 2 * (l * w + w * h + h * l)
def original_edge_sum (l w h : ℝ) : ℝ := 4 * (l + w + h)
def increased_volume (l w h : ℝ) : ℝ := (l + 2) * (w + 2) * (h + 2)

theorem volume_after_increase :
  original_volume l w h = 5000 →
  original_surface_area l w h = 1800 →
  original_edge_sum l w h = 240 →
  increased_volume l w h = 7048 := by
  sorry

end volume_after_increase_l774_774927


namespace june_time_to_bernard_l774_774747

noncomputable def june_to_julia_distance := 2.5 -- miles
noncomputable def june_to_julia_time := 10 -- minutes
noncomputable def headwind_reduction := 0.20 -- 20%

noncomputable def june_speed := june_to_julia_distance / june_to_julia_time -- miles per minute
noncomputable def distance_to_bernard := 4 -- miles
noncomputable def reduced_speed := june_speed * (1 - headwind_reduction) -- miles per minute

theorem june_time_to_bernard :
  (distance_to_bernard / reduced_speed = 20) :=
by
  sorry

end june_time_to_bernard_l774_774747


namespace perfect_square_divisors_count_450_l774_774642

theorem perfect_square_divisors_count_450 :
  let is_divisor (d n : ℕ) := n % d = 0
  let is_perfect_square (n : ℕ) := ∃ m : ℕ, m * m = n
  let prime_factorization_450 := [(2, 1), (3, 2), (5, 2)]
  ∃ (count : ℕ), count = 4 ∧ count = (Finset.filter 
    (λ d, is_perfect_square d ∧ is_divisor d 450)
    (Finset.powersetPrimeFactors prime_factorization_450)).card := sorry

end perfect_square_divisors_count_450_l774_774642


namespace transformable_to_nonnegative_sums_l774_774256

-- Define an m x n table as a matrix
def Matrix (m n : ℕ) := Fin m → Fin n → ℤ

-- Define the operation of changing the sign of all elements in a row
def change_sign_row (A : Matrix m n) (i : Fin m) : Matrix m n :=
  λ x y, if x = i then -A x y else A x y

-- Define the operation of changing the sign of all elements in a column
def change_sign_col (A : Matrix m n) (j : Fin n) : Matrix m n :=
  λ x y, if y = j then -A x y else A x y

-- Define the sum of all elements in a matrix
def matrix_sum (A : Matrix m n) : ℤ :=
  Finset.sum (Finset.univ.product Finset.univ) (λ ⟨i, j⟩, A i j)

-- Define the sum of all elements in a row
def row_sum (A : Matrix m n) (i : Fin m) : ℤ :=
  Finset.sum Finset.univ (λ j, A i j)

-- Define the sum of all elements in a column
def col_sum (A : Matrix m n) (j : Fin n) : ℤ :=
  Finset.sum Finset.univ (λ i, A i j)

-- The main theorem we need to prove
theorem transformable_to_nonnegative_sums (A : Matrix m n) :
  ∃ B : Matrix m n,  
  (∀ i : Fin m, 0 ≤ row_sum B i) ∧
  (∀ j : Fin n, 0 ≤ col_sum B j) :=
begin
  sorry
end

end transformable_to_nonnegative_sums_l774_774256


namespace geometric_sum_is_15_l774_774371

theorem geometric_sum_is_15 (S : ℕ → ℝ) (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, S n = a 1 * (1 - q ^ n) / (1 - q)) ∧
  (4 * a 1, 2 * a 2, a 3) is_arithmetic_sequence ∧
  a 1 = 1 →
  S 4 = 15 :=
by sorry

end geometric_sum_is_15_l774_774371


namespace radius_of_small_semicircle_l774_774730

theorem radius_of_small_semicircle
  (radius_big_semicircle : ℝ)
  (radius_inner_circle : ℝ) 
  (pairwise_tangent : ∀ x : ℝ, x = radius_big_semicircle → x = 12 ∧ 
                                x = radius_inner_circle → x = 6 ∧ 
                                true) :
  ∃ (r : ℝ), r = 4 :=
by 
  sorry

end radius_of_small_semicircle_l774_774730


namespace minimum_value_expression_l774_774551

theorem minimum_value_expression (x y z : ℝ) (hx : x ∈ set.Icc 0 1) (hy : y ∈ set.Icc 0 1) (hz : z ∈ set.Icc 0 1) :
  (∀ x y z, 0 < x ∧ x ≤ 1 → 0 < y ∧ y ≤ 1 → 0 < z ∧ z ≤ 1 → 
  A = (x + 2 * y) * real.sqrt (x + y - x * y) + (y + 2 * z) * real.sqrt (y + z - y * z) + (z + 2 * x) * real.sqrt (z + x - z * x) / (x * y + y * z + z * x)) ≥ 3 :=
sorry

end minimum_value_expression_l774_774551


namespace solve_inequality_l774_774985

theorem solve_inequality : {x : ℝ | 3 * x ^ 2 - 7 * x - 6 < 0} = {x : ℝ | -2 / 3 < x ∧ x < 3} :=
sorry

end solve_inequality_l774_774985


namespace vendelin_pastels_l774_774314

theorem vendelin_pastels (M V W : ℕ) (h1 : M = 5) (h2 : V < 5) (h3 : W = M + V) (h4 : M + V + W = 7 * V) : W = 7 := 
sorry

end vendelin_pastels_l774_774314


namespace parking_time_l774_774237

theorem parking_time (H_min_fee : 0.5) (H_additional_fee_half_hour : 0.5) (total_fee_paid : 5.5) :
  parking_time = 6 := by
  -- Given the conditions:
  -- 1. The minimum fee is 0.5 yuan.
  -- 2. An additional 0.5 yuan is charged for every extra 0.5 hour if parking exceeds 1 hour.
  -- 3. The car paid 5.5 yuan when leaving the parking lot.
  sorry

end parking_time_l774_774237


namespace corrected_mean_is_124_931_l774_774426

/-
Given:
- original_mean : Real = 125.6
- num_observations : Nat = 100
- incorrect_obs1 : Real = 95.3
- incorrect_obs2 : Real = -15.9
- correct_obs1 : Real = 48.2
- correct_obs2 : Real = -35.7

Prove:
- new_mean == 124.931
-/

noncomputable def original_mean : ℝ := 125.6
def num_observations : ℕ := 100
noncomputable def incorrect_obs1 : ℝ := 95.3
noncomputable def incorrect_obs2 : ℝ := -15.9
noncomputable def correct_obs1 : ℝ := 48.2
noncomputable def correct_obs2 : ℝ := -35.7

noncomputable def incorrect_total_sum : ℝ := original_mean * num_observations
noncomputable def sum_incorrect_obs : ℝ := incorrect_obs1 + incorrect_obs2
noncomputable def sum_correct_obs : ℝ := correct_obs1 + correct_obs2
noncomputable def corrected_total_sum : ℝ := incorrect_total_sum - sum_incorrect_obs + sum_correct_obs
noncomputable def new_mean : ℝ := corrected_total_sum / num_observations

theorem corrected_mean_is_124_931 : new_mean = 124.931 := sorry

end corrected_mean_is_124_931_l774_774426


namespace longest_pole_length_l774_774143

theorem longest_pole_length
  (l w h : ℝ)
  (hl : l = 24)
  (hw : w = 18)
  (hh : h = 16) :
  real.sqrt (l^2 + w^2 + h^2) = 34 :=
by 
  rw [hl, hw, hh]
  calc
    real.sqrt (24^2 + 18^2 + 16^2) = real.sqrt (576 + 324 + 256) : by congr; norm_num
                                    ... = real.sqrt 1156 : by norm_num
                                    ... = 34 : by norm_num

end longest_pole_length_l774_774143


namespace sum_of_distinct_integers_l774_774283

theorem sum_of_distinct_integers 
  (a b c d e : ℤ)
  (h1 : (7 - a) * (7 - b) * (7 - c) * (7 - d) * (7 - e) = 60)
  (h2 : (7 - a) ≠ (7 - b) ∧ (7 - a) ≠ (7 - c) ∧ (7 - a) ≠ (7 - d) ∧ (7 - a) ≠ (7 - e))
  (h3 : (7 - b) ≠ (7 - c) ∧ (7 - b) ≠ (7 - d) ∧ (7 - b) ≠ (7 - e))
  (h4 : (7 - c) ≠ (7 - d) ∧ (7 - c) ≠ (7 - e))
  (h5 : (7 - d) ≠ (7 - e)) : 
  a + b + c + d + e = 24 := 
sorry

end sum_of_distinct_integers_l774_774283


namespace smallest_y_for_divisibility_by_11_l774_774043

theorem smallest_y_for_divisibility_by_11 :
  ∃ y : ℕ, ((14 - y) % 11 = 0) ∧ y = 14 :=
by
  use 14
  split
  exact Nat.mod_eq_zero_of_dvd (dvd_refl _)
  rfl

end smallest_y_for_divisibility_by_11_l774_774043


namespace find_lambda_l774_774220

open Real

variables (a b c : ℝ^3) (λ : ℝ)

axiom (a_norm : ‖a‖ = λ)
axiom (b_norm : ‖b‖ = λ)
axiom (c_norm : ‖c‖ = λ)
axiom (a_dot_b : a ⬝ b = 0)
axiom (a_dot_c : a ⬝ c = 2)
axiom (b_dot_c : b ⬝ c = 1)

theorem find_lambda (λ : ℝ) (a b c : ℝ^3) : 
  (‖a‖ = λ) → (‖b‖ = λ) → (‖c‖ = λ) → 
  (a ⬝ b = 0) → (a ⬝ c = 2) → (b ⬝ c = 1) → 
  λ = (5 : ℝ) ^ (1/4:ℝ) := 
by
  intro a_norm b_norm c_norm a_dot_b a_dot_c b_dot_c
  sorry

end find_lambda_l774_774220


namespace tangent_bisector_circle_l774_774167

open EuclideanGeometry

noncomputable def circle : Type := sorry   -- placeholder for the correct definition of a circle
noncomputable def Point : Type := sorry    -- placeholder for the correct definition of a point

noncomputable def tangent_line (c : circle) (P : Point) : Point := sorry
noncomputable def opposite_point (c : circle) (B : Point) : Point := sorry
noncomputable def perpendicular_foot (A : Point) (B : Point) (D : Point) : Point := sorry
noncomputable def bisector (P : Point) (D : Point) (A : Point) (C : Point) : Prop := sorry

theorem tangent_bisector_circle (c : circle) (O : Point) (r : ℝ) (P : Point)
  (A B : Point) (D C : Point) :
  tangent_line c P = A ∧ tangent_line c P = B ∧
  (opposite_point c B = D) ∧ 
  (perpendicular_foot A B D = C) →
  bisector P D A C :=
by
  sorry

end tangent_bisector_circle_l774_774167


namespace conditional_probability_l774_774175

noncomputable def eventA (x : ℝ) : Prop := (0 < x) ∧ (x < 1/2)
noncomputable def eventB (x : ℝ) : Prop := (1/4 < x) ∧ (x < 1)

theorem conditional_probability :
  (∃ x : ℝ, 0 < x ∧ x < 1) →
  (let P : (Set ℝ) → ℝ := λ s, if (s = Set.univ) then 1 else (PseudoEMetricSpace.volume s) / (PseudoEMetricSpace.volume Set.univ) in
  P {x | (eventA x) ∧ (eventB x)} / P {x | eventA x}) = 1/2 :=
sorry

end conditional_probability_l774_774175


namespace eval_sqrt4_8_pow12_l774_774130

-- Define the fourth root of 8
def fourthRootOfEight : ℝ := 8 ^ (1 / 4)

-- Define the original expression
def expr := (fourthRootOfEight) ^ 12

-- The theorem to prove
theorem eval_sqrt4_8_pow12: expr = 512 := by
  sorry

end eval_sqrt4_8_pow12_l774_774130


namespace find_y_coord_l774_774881

structure Point3D :=
  (x : ℝ) (y : ℝ) (z : ℝ)

def distance (p1 p2 : Point3D) : ℝ :=
  real.sqrt ((p2.x - p1.x) ^ 2 + (p2.y - p1.y) ^ 2 + (p2.z - p1.z) ^ 2)

def A (y : ℝ) : Point3D := Point3D.mk 0 y 0
def B : Point3D := Point3D.mk 1 6 4
def C : Point3D := Point3D.mk 5 7 1

theorem find_y_coord : ∃ y : ℝ, distance (A y) B = distance (A y) C ∧ y = 11 :=
by
  sorry

end find_y_coord_l774_774881


namespace exists_zero_in_interval_l774_774342
open Set Real Function

noncomputable def f (x : ℝ) := Real.exp x * Real.log 2 + x^2 - 6*x - 1

theorem exists_zero_in_interval : 
  ∃ c ∈ Ioo 3 4, f c = 0 :=
begin
  -- The proof goes here
  sorry
end

end exists_zero_in_interval_l774_774342


namespace max_jars_same_coins_l774_774001

theorem max_jars_same_coins : ∃ (N : ℕ), N = 2014 ∧ 
  (∀ d : ℕ, ∀ jars : Fin 2017 → ℕ, 
    (∀ i : Fin 2008, ∃ k : Fin 2017, jars k = d → (∀ j : ℕ, j < 10 → k + j < 2017 → jars ((k + j) % 2017) = d)) → 
    N = 2014) :=
by 
  have h1 : ∀ N d, N = 2014 →
    ∃ (k l : Fin 2017), ∀ j : ℕ, j < 10 → k + j < 2017 → jars ((k + j) % 2017) = d,
  {
    -- Proof outline, not an actual proof:
    -- Assume N = 2014, and prove the conditions accordingly using finite induction and combinatorics
    sorry
  }
  existsi 2014
  simp
  exact h1

end max_jars_same_coins_l774_774001


namespace perfect_square_divisors_of_450_l774_774617

theorem perfect_square_divisors_of_450 :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let e1 := 1
  let e2 := 2
  let e3 := 2
  ∃ (count : ℕ), count = 4 ∧
  (∀ (d : ℕ), d ∣ (p1^e1 * p2^e2 * p3^e3) →
    (∀ (pe : ℕ × ℕ), pe ∈ (Multiset.ofList [(p1, e1), (p2, e2), (p3, e3)]).powerset → 
      (∃ (ex1 ex2 ex3: ℕ), ex1 ≤ e1 ∧ ex2 ≤ e2 ∧ ex3 ≤ e3 ∧ ex1 % 2 = 0 ∧ ex2 % 2 = 0 ∧ ex3 % 2 = 0))) :=
sorry

end perfect_square_divisors_of_450_l774_774617


namespace solve_for_x_l774_774216

theorem solve_for_x (x : ℕ) (h : (1 / 8) * 2 ^ 36 = 8 ^ x) : x = 11 :=
by
sorry

end solve_for_x_l774_774216


namespace minimum_elapsed_time_l774_774961

theorem minimum_elapsed_time : 
  let initial_time := 45  -- in minutes
  let final_time := 3 * 60 + 30  -- 3 hours 30 minutes in minutes
  let elapsed_time := final_time - initial_time
  elapsed_time = 2 * 60 + 45 :=
by
  sorry

end minimum_elapsed_time_l774_774961


namespace heather_lighter_than_combined_weights_l774_774598

noncomputable def heather_weight : ℝ := 87.5
noncomputable def emily_weight : ℝ := 45.3
noncomputable def elizabeth_weight : ℝ := 38.7
noncomputable def george_weight : ℝ := 56.9

theorem heather_lighter_than_combined_weights :
  heather_weight - (emily_weight + elizabeth_weight + george_weight) = -53.4 :=
by 
  sorry

end heather_lighter_than_combined_weights_l774_774598


namespace find_radius_of_small_semicircle_l774_774736

noncomputable def radius_of_small_semicircle (R : ℝ) (r : ℝ) :=
  ∀ (x : ℝ),
    (12: ℝ = R) ∧ (6: ℝ = r) →
    (∃ (x: ℝ), R - x + r = sqrt((r + x)^2 - r^2)) →
    x = 4

theorem find_radius_of_small_semicircle : radius_of_small_semicircle 12 6 :=
begin
  unfold radius_of_small_semicircle,
  intro x,
  assume h1 h2,
  cases h2,
  sorry,
end

end find_radius_of_small_semicircle_l774_774736


namespace trinomial_has_two_roots_l774_774912

theorem trinomial_has_two_roots (a b c : ℝ) (h : b^2 - 4 * a * c > 0) :
  let Δ := (2 * (a + b))^2 - 4 * 3 * a * (b + c) in Δ > 0 :=
by
  sorry

end trinomial_has_two_roots_l774_774912


namespace quadratic_trinomial_has_two_roots_l774_774917

theorem quadratic_trinomial_has_two_roots
  (a b c : ℝ) (h : b^2 - 4 * a * c > 0) :
  4 * (a^2 - a * b + b^2 - 3 * a * c) > 0 :=
by
  sorry

end quadratic_trinomial_has_two_roots_l774_774917


namespace percentage_left_on_account_of_fear_l774_774249

-- Definitions and conditions
def initial_population : ℕ := 4500
def percentage_died_by_bombardment : ℝ := 0.10
def reduced_population : ℕ := 3240

-- The percentage of the remaining people who left the village on account of fear
theorem percentage_left_on_account_of_fear :
  (initial_population - initial_population * percentage_died_by_bombardment.real_to_nat - reduced_population) / (initial_population - initial_population * percentage_died_by_bombardment.real_to_nat) * 100 = 20 :=
sorry

end percentage_left_on_account_of_fear_l774_774249


namespace purchase_price_of_first_commodity_l774_774810

-- Define the conditions
variable (price_first price_second : ℝ)
variable (h1 : price_first - price_second = 127)
variable (h2 : price_first + price_second = 827)

-- Prove the purchase price of the first commodity is $477
theorem purchase_price_of_first_commodity : price_first = 477 :=
by
  sorry

end purchase_price_of_first_commodity_l774_774810


namespace urn_gold_coins_percent_l774_774089

theorem urn_gold_coins_percent (perc_beads : ℝ) (perc_silver_coins : ℝ) (perc_gold_coins : ℝ) :
  perc_beads = 0.2 →
  perc_silver_coins = 0.4 →
  perc_gold_coins = 0.48 :=
by
  intros h1 h2
  sorry

end urn_gold_coins_percent_l774_774089


namespace eval_sqrt4_8_pow12_l774_774132

-- Define the fourth root of 8
def fourthRootOfEight : ℝ := 8 ^ (1 / 4)

-- Define the original expression
def expr := (fourthRootOfEight) ^ 12

-- The theorem to prove
theorem eval_sqrt4_8_pow12: expr = 512 := by
  sorry

end eval_sqrt4_8_pow12_l774_774132


namespace cylinder_cone_volume_ratio_l774_774874

theorem cylinder_cone_volume_ratio
  (h r : ℝ) (hcylinder : volume_cylinder = real.pi * r^2 * h)
  (hcone : volume_cone = (1/3) * real.pi * r^2 * h) :
  volume_cylinder / volume_cone = 3 :=
sorry

end cylinder_cone_volume_ratio_l774_774874


namespace find_constants_for_sine_cubed_identity_l774_774504

theorem find_constants_for_sine_cubed_identity :
  ∃ c d : ℝ, (∀ θ : ℝ, sin θ ^ 3 = c * sin (3 * θ) + d * sin θ) ∧ (c = -1/4) ∧ (d = 3/4) :=
begin
  -- This is the starting point provided by the user
  sorry
end

end find_constants_for_sine_cubed_identity_l774_774504


namespace number_of_perfect_square_divisors_of_450_l774_774637

theorem number_of_perfect_square_divisors_of_450 : 
    let p := 450;
    let factors := [(3, 2), (5, 2), (2, 1)];
    ∃ n, (n = 4 ∧ 
          ∀ (d : ℕ), d ∣ p → 
                     (∃ (a b c : ℕ), d = 2^a * 3^b * 5^c ∧ 
                              (a = 0) ∧ (b = 0 ∨ b = 2) ∧ (c = 0 ∨ c = 2) → 
                              a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 = 0)) :=
    sorry

end number_of_perfect_square_divisors_of_450_l774_774637


namespace find_a_l774_774554

open Complex

noncomputable def Z (a : ℂ) : ℂ := (a + I) / (1 + I)

theorem find_a (a : ℝ) (h : Im (Z a) = Z a) : a = -1 :=
  sorry

end find_a_l774_774554


namespace brocard_points_on_circle_l774_774774

-- Definitions of key points and properties
def circumcenter (ABC : Triangle) : Point := sorry
def lemoinePoint (ABC : Triangle) : Point := sorry
def brocardPoints (ABC : Triangle) : (Point × Point) := sorry
def brocardAngle (ABC : Triangle) : Angle := sorry
def circleDiameter (p1 p2 : Point) : Circle := sorry

-- The statement of the problem
theorem brocard_points_on_circle 
  (ABC : Triangle)
  (O : Point := circumcenter ABC)
  (K : Point := lemoinePoint ABC)
  (P Q : Point × Point := brocardPoints ABC)
  (φ : Angle := brocardAngle ABC) :
  (pointsOnCircle (P.1, P.2) (circleDiameter K O)) ∧
  (OP = OQ) ∧
  (anglePOQ = 2 * φ) :=
sorry

end brocard_points_on_circle_l774_774774


namespace mean_minus_median_eq_four_l774_774535

theorem mean_minus_median_eq_four (x : ℕ) (hx: 0 < x) :
  let mean := (x + (x + 2) + (x + 4) + (x + 7) + (x + 27)) / 5
  let median := (list.sort (≤) [x, x + 2, x + 4, x + 7, x + 27]).get ⟨2, sorry⟩
  mean - median = 4 := by
  sorry

end mean_minus_median_eq_four_l774_774535


namespace find_ratio_of_constants_l774_774498

theorem find_ratio_of_constants (x y c d : ℚ) (hx : x ≠ 0) (hy : y ≠ 0) (hd : d ≠ 0) 
  (h₁ : 8 * x - 6 * y = c) (h₂ : 12 * y - 18 * x = d) : c / d = -4 / 9 := 
sorry

end find_ratio_of_constants_l774_774498


namespace am_dn_xy_concurrent_l774_774291

noncomputable def conccurency_AM_DN_XY (A B C D X Y Z P M N : Point) (h1 : A ≠ B) (h2 : B ≠ C) 
(h3 : C ≠ D) (h4 : collinear_points [A, B, C, D]) (h5 : on_circle X (circle_AC A C)) 
(h6 : on_circle Y (circle_BD B D)) (h7 : intercepts_line Z (line_XY X Y) (line_BC B C)) 
(h8 : P ≠ Z) (h9 : on_line P (line_XY X Y)) (h10 : on_circle C (circle_AC A C)) 
(h11 : on_circle M (circle_AC A C)) (h12 : interception_points C M (line_CP C P)) 
(h13 : on_circle B (circle_BD B D)) (h14 : on_circle N (circle_BD B D)) 
(h15 : interception_points B N (line_BP B P)) (h16 : inter_line_points_XY_AM (line_XY X Y) (line_AM A M))
(h17 : inter_line_points_XY_DN (line_XY X Y) (line_DN D N)): Prop :=
collinear_points [A, M, (inter_line_points_XY_AM (line_XY X Y) (line_AM A M))] 
∧ collinear_points [D, N, (inter_line_points_XY_DN (line_XY X Y) (line_DN D N))] 
∧ concurrency (line_AM A M) (line_DN D N) (line_XY X Y)

-- A placeholder statement for the overall theorem
theorem am_dn_xy_concurrent (A B C D X Y Z P M N : Point) 
(h1 : A ≠ B) (h2 : B ≠ C) (h3 : C ≠ D) (h4 : collinear_points [A, B, C, D])
(h5 : circle_with_diameter_AC X C) (h6 : circle_with_diameter_BD Y D)(h7 : intercepts_Z_at_XY_BC Z X Y B C)
(h8 : P ≠ Z) (h9 : P_on_XY P X Y) 
(h10 : on_circle C (circle_AC A C)) (h11 : on_circle M (circle_AC A C))
(h12 : intercept_CP_at_CM C M P) (h13 : on_circle B (circle_BD B D))
(h14 : on_circle N (circle_BD B D)) (h15 : intercept_BP_at_BN B N P): 
inter_line_points_XY_AM (line_XY X Y) (line_AM A M) = inter_line_points_XY_DN (line_XY X Y) (line_DN D N) :=
sorry

end am_dn_xy_concurrent_l774_774291


namespace find_x_l774_774042

theorem find_x (x y : ℝ) (h1 : x + 2 * y = 10) (h2 : y = 4) : x = 2 :=
by
  sorry

end find_x_l774_774042


namespace foma_wait_time_probability_l774_774962

noncomputable def probability_no_more_than_four_minutes_wait (x y : ℝ) : ℝ :=
if h : 2 < x ∧ x < y ∧ y < 10 ∧ y - x ≤ 4 then
  (1 / 2)
else 0

theorem foma_wait_time_probability :
  ∀ (x y : ℝ), 2 < x → x < y → y < 10 → 
  (probability_no_more_than_four_minutes_wait x y) = 1 / 2 :=
sorry

end foma_wait_time_probability_l774_774962


namespace numPerfectSquareFactorsOf450_l774_774684

def isPerfectSquare (n : Nat) : Prop :=
  ∃ k : Nat, n = k * k

theorem numPerfectSquareFactorsOf450 : 
  ∃! n : Nat, 
    (∀ d : Nat, d ∣ 450 → isPerfectSquare d) → n = 4 := 
by
  sorry

end numPerfectSquareFactorsOf450_l774_774684


namespace problem1_problem2_l774_774486

-- Problem 1: Simplifying the expression 
theorem problem1 :
  (sqrt 6 * sqrt 3) / (sqrt 24) = sqrt 3 / 2 :=
sorry

-- Problem 2: Simplifying the more complex expression
theorem problem2 :
  ((-1/2) ^ (-2)) + (1/2 * sqrt 12) - (3/4 * sqrt 48) + (sqrt 3 - 1)^0 = 5 - 2 * sqrt 3 :=
sorry

end problem1_problem2_l774_774486


namespace fib_product_value_l774_774338

noncomputable def fib : ℕ → ℕ
| 0       := 0
| 1       := 1
| (n + 2) := fib (n + 1) + fib n

theorem fib_product_value : 
  (∏ i in Finset.range 2019 \ Finset.range 1, (1 - (fib (i + 1) ^ 2 / fib (i + 2) ^ 2))) = 1 / 2 :=
sorry

end fib_product_value_l774_774338


namespace max_product_is_863_l774_774858

noncomputable def max_product_odd_sum_digits : ℕ :=
  let digits := {2, 3, 4, 6, 7, 8, 9}
  let sum_of_digits n := (n / 100) + ((n % 100) / 10) + (n % 10)
  let is_odd n := (sum_of_digits n) % 2 = 1
  let options := {n | ∀ a ∈ digits, ∀ b ∈ (digits \ {a}),
                     ∀ c ∈ (digits \ {a, b}), n = 100*a + 10*b + c ∧ is_odd n}
  let abs_max := nat.max (λ n1 n2 : ℕ, ∃ m1 m2, m1 = 1000*((digits \ {n1}) \ {n2}) + 10*((digits \ {n1}) \ {n2, n2}) + (digits \ {n1, n2, n2}) * n1 * n2 ) 
  {n | n ∈ options ∧ 
       ∃ m l ∈ {p : ℕ | p ∈ digits ∧ p ≠ n}, m * l = abs_max}
  863

theorem max_product_is_863 : max_product_odd_sum_digits = 863 :=
  by
    sorry

end max_product_is_863_l774_774858


namespace unique_solution_for_y_l774_774981

def operation (x y : ℝ) : ℝ := 4 * x - 2 * y + x^2 * y

theorem unique_solution_for_y : ∃! (y : ℝ), operation 3 y = 20 :=
by {
  sorry
}

end unique_solution_for_y_l774_774981


namespace perfect_square_factors_450_l774_774653

def prime_factorization (n : ℕ) : Prop :=
  n = 2^1 * 3^2 * 5^2

theorem perfect_square_factors_450 : prime_factorization 450 → ∃ n : ℕ, n = 4 :=
by
  sorry

end perfect_square_factors_450_l774_774653


namespace alice_study_time_for_average_75_l774_774474

variable (study_time : ℕ → ℚ)
variable (score : ℕ → ℚ)

def inverse_relation := ∀ n, study_time n * score n = 120

theorem alice_study_time_for_average_75
  (inverse_relation : inverse_relation study_time score)
  (study_time_1 : study_time 1 = 2)
  (score_1 : score 1 = 60)
  : study_time 2 = 4/3 := by
  sorry

end alice_study_time_for_average_75_l774_774474


namespace markup_price_l774_774829

noncomputable def purchase_price : ℝ := 150
noncomputable def overhead_percentage : ℝ := 0.12
noncomputable def net_profit : ℝ := 35
noncomputable def sales_tax_percentage : ℝ := 0.05
noncomputable def discount_percentage : ℝ := 0.15

noncomputable def overhead_cost : ℝ := overhead_percentage * purchase_price
noncomputable def sales_tax : ℝ := sales_tax_percentage * purchase_price
noncomputable def total_amount_before_discount : ℝ := purchase_price + overhead_cost + net_profit + sales_tax
noncomputable def marked_price (total_amount_before_discount : ℝ) (discount_percentage : ℝ) : ℝ := total_amount_before_discount / (1 - discount_percentage)

theorem markup_price :
  marked_price total_amount_before_discount discount_percentage ≈ 247.65 := 
sorry

end markup_price_l774_774829


namespace side_length_of_square_l774_774037

theorem side_length_of_square (total_length : ℝ) (sides : ℕ) (h1 : total_length = 100) (h2 : sides = 4) :
  (total_length / (sides : ℝ) = 25) :=
by
  sorry

end side_length_of_square_l774_774037


namespace range_of_t_l774_774452

noncomputable def f (x : ℝ) : ℝ :=
if x ∈ [0, 1) then x^2 - x
else if x ∈ [1, 2] then (1 / 10) * (x - 2)
else 0  -- this handles all x outside [0, 2], will be used for f(x+2) = 2f(x)

axiom f_property : ∀ x : ℝ, f (x + 2) = 2 * f x

theorem range_of_t : ∀ t : ℝ, (∀ x : ℝ, 4 ≤ x ∧ x ≤ 6 → f x ≥ t^2 - 2 * t - 4) ↔ -1 ≤ t ∧ t ≤ 3 := sorry

end range_of_t_l774_774452


namespace carlo_practice_time_is_consistent_l774_774974

variable (monday tuesday wednesday thursday friday total_practice_time : ℕ)

theorem carlo_practice_time_is_consistent :
  monday = 2 * (wednesday - 10) ∧
  tuesday = wednesday - 10 ∧
  wednesday > thursday ∧
  thursday = 50 ∧
  total_practice_time = 300 ∧
  friday = 60 →
  (monday + tuesday + wednesday + thursday + friday = 300) →
  (wednesday - thursday = 5) :=
by
  assume h h_total
  sorry

end carlo_practice_time_is_consistent_l774_774974


namespace symmetric_points_sum_l774_774715

variable (a b : ℝ)

theorem symmetric_points_sum (h₁ : P = (a, 1)) (h₂ : Q = (2, b)) (symmetry_condition : symmetric_about_x_axis P Q) : a + b = 1 := 
by
  sorry

end symmetric_points_sum_l774_774715


namespace clarinet_hourly_rate_l774_774743

def weekly_hours_clarinet : ℕ := 3
def weekly_hours_piano : ℕ := 5
def hourly_rate_piano : ℕ := 28
def annual_additional_cost_piano : ℕ := 1040
def weeks_in_year : ℕ := 52

theorem clarinet_hourly_rate :
  let annual_hours_clarinet := weekly_hours_clarinet * weeks_in_year,
      annual_hours_piano := weekly_hours_piano * weeks_in_year,
      annual_cost_piano := annual_hours_piano * hourly_rate_piano,
      annual_cost_clarinet (C : ℕ) := annual_hours_clarinet * C in
  ∃ C : ℕ, annual_cost_piano = annual_cost_clarinet C + annual_additional_cost_piano ∧ C = 40 :=
by
  sorry

end clarinet_hourly_rate_l774_774743


namespace zero_point_of_log_a_x_plus_x_minus_m_interval_0_1_l774_774191

theorem zero_point_of_log_a_x_plus_x_minus_m_interval_0_1
  (a m : ℝ) (h₀ : 1 < a) (h₁ : ∃ x : ℝ, 0 < x ∧ x < 1 ∧ log a x + x - m = 0) :
  m < 1 :=
begin
  sorry
end

end zero_point_of_log_a_x_plus_x_minus_m_interval_0_1_l774_774191


namespace Tom_weekend_sleep_hours_l774_774389

theorem Tom_weekend_sleep_hours
  (weeknight_sleep : ℕ)
  (ideal_sleep_per_night : ℕ)
  (sleep_deficit : ℕ)
  (weekend_nights : ℕ) :
  weeknight_sleep = 5 →
  ideal_sleep_per_night = 8 →
  sleep_deficit = 19 →
  weekend_nights = 2 →
  let total_weeknight_sleep := weeknight_sleep * 5 in
  let ideal_total_sleep := ideal_sleep_per_night * 7 in
  let actual_total_sleep := ideal_total_sleep - sleep_deficit in
  let weekend_sleep := actual_total_sleep - total_weeknight_sleep in
  weekend_sleep / weekend_nights = 6 :=
by
  intros
  simp_all
  sorry

end Tom_weekend_sleep_hours_l774_774389


namespace least_marbles_l774_774405

theorem least_marbles (n : ℕ) (h1 : n % 7 = 3) (h2 : n % 4 = 2) (h3 : n % 6 = 1) : n = 402 :=
by
  sorry

end least_marbles_l774_774405


namespace manufacturing_section_degrees_l774_774804

variable (percentage_manufacturing : ℝ) (total_degrees : ℝ)

theorem manufacturing_section_degrees
  (h1 : percentage_manufacturing = 0.40)
  (h2 : total_degrees = 360) :
  percentage_manufacturing * total_degrees = 144 := 
by 
  sorry

end manufacturing_section_degrees_l774_774804


namespace perpendicular_vectors_l774_774396

def vector_a (t : ℝ) : ℝ × ℝ := (t, 1)
def vector_b : ℝ × ℝ := (2, 4)

theorem perpendicular_vectors (t : ℝ) : vector_a t.1 * vector_b.1 + vector_a t.2 * vector_b.2 = 0 -> t = -2 :=
  sorry

end perpendicular_vectors_l774_774396


namespace g_inv_correct_l774_774303

-- Definitions for the four functions and their invertibility
variables {X Y : Type} [Nonempty X] [Nonempty Y]

noncomputable def a : X → Y := sorry
noncomputable def b : Y → X := sorry
noncomputable def c : X → Y := sorry
noncomputable def d : Y → X := sorry

noncomputable def a_inv : Y → X := sorry
noncomputable def b_inv : X → Y := sorry
noncomputable def c_inv : Y → X := sorry
noncomputable def d_inv : X → Y := sorry

axiom a_inv_is_inv : ∀ x, a (a_inv x) = x ∧ a_inv (a x) = x
axiom b_inv_is_inv : ∀ x, b (b_inv x) = x ∧ b_inv (b x) = x
axiom c_inv_is_inv : ∀ x, c (c_inv x) = x ∧ c_inv (c x) = x
axiom d_inv_is_inv : ∀ x, d (d_inv x) = x ∧ d_inv (d x) = x

-- Function g and its inverse
noncomputable def g (x : X) : X := b (a (d (c x)))
noncomputable def g_inv (x : X) : X := c_inv (d_inv (a_inv (b_inv x)))

-- The theorem to be proved
theorem g_inv_correct (x : X) : g_inv (g x) = x :=
by {
  have h1 : ∀ x, g x = b (a (d (c x))) := sorry,
  have h2 : ∀ y, b_inv y = a (d (c y)) := sorry,
  have h3 : ∀ z, a_inv z = d (c z) := sorry,
  have h4 : ∀ w, d_inv w = c w := sorry,
  have h5 : ∀ v, c_inv v = v := sorry,
  sorry
}

end g_inv_correct_l774_774303


namespace min_distance_points_l774_774188

noncomputable def minDistPQ : ℝ :=
√2 * (1 - Real.log 2)

theorem min_distance_points :
  ∃ (a b : ℝ), P : (a, exp a / 2) ∧ Q : (b, Real.log (2 * b)) ∧ 
  (minDistPQ = sqrt ((a - b) ^ 2 + (exp a / 2 - Real.log (2 * b)) ^ 2)) :=
sorry

end min_distance_points_l774_774188


namespace max_min_3_5_max_min_neg1_3_l774_774548

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^2 - 4 * x + 3

-- Theorem for maximum and minimum values of f(x) on [3,5]
theorem max_min_3_5 :
  (∀ x ∈ set.Icc (3 : ℝ) 5, f x ≤ 8) ∧ (∃ x ∈ set.Icc (3 : ℝ) 5, f x = 8) ∧
  (∀ x ∈ set.Icc (3 : ℝ) 5, f x ≥ 0) ∧ (∃ x ∈ set.Icc (3 : ℝ) 5, f x = 0) :=
sorry

-- Theorem for maximum and minimum values of f(x) on [-1,3]
theorem max_min_neg1_3 :
  (∀ x ∈ set.Icc (-1 : ℝ) 3, f x ≤ 8) ∧ (∃ x ∈ set.Icc (-1 : ℝ) 3, f x = 8) ∧
  (∀ x ∈ set.Icc (-1 : ℝ) 3, f x ≥ -1) ∧ (∃ x ∈ set.Icc (-1 : ℝ) 3, f x = -1) :=
sorry

end max_min_3_5_max_min_neg1_3_l774_774548


namespace problem_proof_l774_774190

noncomputable def bin_coeff (n k : ℕ) : ℕ := Nat.choose n k

theorem problem_proof :
  (∃ n : ℕ, (∑ k in range (n + 1), (sqrt (x^2 / 3) + 3 * x^2) ^ k = 32)) ∧
  (n = 5 →
    (∃ t3 t4 : ℕ, t3 = bin_coeff 5 3 * (sqrt (x^2 / 3))^3 * (3*x^2)^2 ∧
     t4 = bin_coeff 5 4 * (sqrt (x^2 / 3))^2 * (3*x^2)^3 ∧
     t3 = 90 * x^6 ∧
     t4 = 270 * x ^ (22 / 3))) :=
by
  -- Insert the actual proof here
  sorry

end problem_proof_l774_774190


namespace average_GPA_of_class_l774_774349

theorem average_GPA_of_class (n : ℕ) (h1 : n > 0) 
  (GPA1 : ℝ := 60) (GPA2 : ℝ := 66) 
  (students_ratio1 : ℝ := 1 / 3) (students_ratio2 : ℝ := 2 / 3) :
  let total_students := (students_ratio1 * n + students_ratio2 * n)
  let total_GPA := (students_ratio1 * n * GPA1 + students_ratio2 * n * GPA2)
  let average_GPA := total_GPA / total_students
  average_GPA = 64 := by
    sorry

end average_GPA_of_class_l774_774349


namespace minimum_value_expression_l774_774550

theorem minimum_value_expression (x y z : ℝ) (hx : x ∈ set.Icc 0 1) (hy : y ∈ set.Icc 0 1) (hz : z ∈ set.Icc 0 1) :
  (∀ x y z, 0 < x ∧ x ≤ 1 → 0 < y ∧ y ≤ 1 → 0 < z ∧ z ≤ 1 → 
  A = (x + 2 * y) * real.sqrt (x + y - x * y) + (y + 2 * z) * real.sqrt (y + z - y * z) + (z + 2 * x) * real.sqrt (z + x - z * x) / (x * y + y * z + z * x)) ≥ 3 :=
sorry

end minimum_value_expression_l774_774550


namespace simplify_expression_l774_774330

noncomputable def y := 
  Real.cos (2 * Real.pi / 15) + 
  Real.cos (4 * Real.pi / 15) + 
  Real.cos (8 * Real.pi / 15) + 
  Real.cos (14 * Real.pi / 15)

theorem simplify_expression : 
  y = (-1 + Real.sqrt 61) / 4 := 
sorry

end simplify_expression_l774_774330


namespace ball_arrangements_l774_774709

theorem ball_arrangements (black white red : ℕ) (h_black : black = 2) (h_white : white = 3) (h_red : red = 4) :
  ∑ (arrangements : (Fin (black + white + red) → ℕ) (h_no_adj : ∀ i, arrangements (i - 1) = arrangements (i + 1) → arrangements i ≠ 1 ∨ arrangements i ≠ 2), 1) = 200 :=
sorry

end ball_arrangements_l774_774709


namespace product_of_solutions_l774_774526

theorem product_of_solutions : 
  ∀ y : ℝ, (|y| = 3 * (|y| - 2)) → ∃ a b : ℝ, (a = 3 ∧ b = -3) ∧ (a * b = -9) := 
by 
  sorry

end product_of_solutions_l774_774526


namespace number_of_perfect_square_divisors_of_450_l774_774639

theorem number_of_perfect_square_divisors_of_450 : 
    let p := 450;
    let factors := [(3, 2), (5, 2), (2, 1)];
    ∃ n, (n = 4 ∧ 
          ∀ (d : ℕ), d ∣ p → 
                     (∃ (a b c : ℕ), d = 2^a * 3^b * 5^c ∧ 
                              (a = 0) ∧ (b = 0 ∨ b = 2) ∧ (c = 0 ∨ c = 2) → 
                              a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 = 0)) :=
    sorry

end number_of_perfect_square_divisors_of_450_l774_774639


namespace equation_has_real_roots_l774_774321

theorem equation_has_real_roots (a b : ℝ) (h : ¬ (a = 0 ∧ b = 0)) :
  ∃ x : ℝ, x ≠ 1 ∧ (a^2 / x + b^2 / (x - 1) = 1) :=
by
  sorry

end equation_has_real_roots_l774_774321


namespace salmon_migration_multiple_l774_774384

theorem salmon_migration_multiple (initial_salmons current_salmons : ℕ) 
  (h_initial : initial_salmons = 500) (h_current : current_salmons = 5500) :
  (current_salmons - initial_salmons) / initial_salmons = 10 :=
by
  rw [h_initial, h_current]
  sorry

end salmon_migration_multiple_l774_774384


namespace total_fat_served_l774_774455

theorem total_fat_served :
  let herring_fat := 40
  let eel_fat := 20
  let pike_fat := eel_fat + 10
  let salmon_fat := 35
  let halibut_fat := 50
  let herrings_served := 40
  let eels_served := 30
  let pikes_served := 25
  let salmons_served := 20
  let halibuts_served := 15
  let total_fat := (herring_fat * herrings_served) +
                   (eel_fat * eels_served) +
                   (pike_fat * pikes_served) +
                   (salmon_fat * salmons_served) +
                   (halibut_fat * halibuts_served)
  in total_fat = 4400 := 
by
  -- Proof
  sorry

end total_fat_served_l774_774455


namespace value_of_x_l774_774214

theorem value_of_x : (∃ x : ℝ, (1 / 8) * 2 ^ 36 = 8 ^ x) → x = 11 := by
  intro h
  rcases h with ⟨x, hx⟩
  have h1 : 1 / 8 = 2 ^ (-3) := by norm_num
  rw [h1, ←pow_add] at hx
  norm_num at hx
  have h2 : 8 = 2 ^ 3 := by norm_num
  rw [h2, pow_mul] at hx
  norm_num at hx
  exact hx.symm

end value_of_x_l774_774214


namespace range_of_m_l774_774590

variables (m : ℝ)

def p : Prop := ∀ x : ℝ, 0 < x → (1/2 : ℝ)^x + m - 1 < 0
def q : Prop := ∃ x : ℝ, 0 < x ∧ m * x^2 + 4 * x - 1 = 0

theorem range_of_m (h : p m ∧ q m) : -4 ≤ m ∧ m ≤ 0 :=
sorry

end range_of_m_l774_774590


namespace zeros_of_f_l774_774824

noncomputable def f : ℝ → ℝ :=
λ x, if x ≤ 0 then x^2 + 2*x - 3 else -2 + Real.log x

theorem zeros_of_f :
  {x | f x = 0}.finite ∧ {x | f x = 0}.to_finset.card = 2 :=
by
  sorry

end zeros_of_f_l774_774824


namespace problem_sufficient_necessary_condition_l774_774691

open Set

variable {x : ℝ}

def P (x : ℝ) : Prop := abs (x - 2) < 3
def Q (x : ℝ) : Prop := x^2 - 8 * x + 15 < 0

theorem problem_sufficient_necessary_condition :
    (∀ x, Q x → P x) ∧ (∃ x, P x ∧ ¬ Q x) :=
by
  sorry

end problem_sufficient_necessary_condition_l774_774691


namespace probability_of_non_target_quadratic_equation_l774_774104

def total_pairs := 21 * 21

def valid_pairs : ℕ := 37

def probability_not_distinct_and_positive_sum : ℝ := (total_pairs - valid_pairs) / total_pairs.toReal

theorem probability_of_non_target_quadratic_equation :
  probability_not_distinct_and_positive_sum = 404 / 441 :=
by
  -- Initial definitions
  let total_pairs_eq : total_pairs = 441 := rfl
  let valid_pairs_eq : valid_pairs = 37 := rfl

  -- Calculation of the probability
  have prob_eq : probability_not_distinct_and_positive_sum = (441 - 37) / 441.toReal :=
    by rw [total_pairs_eq, valid_pairs_eq]

  -- Conclude the proof
  rw prob_eq
  norm_num  -- Simplifies the numerical fraction
  sorry     -- Placeholder for completing the detailed verification if needed

end probability_of_non_target_quadratic_equation_l774_774104


namespace arrange_numbers_l774_774957

theorem arrange_numbers (a b c : ℝ) (h1 : a = 2) (h2 : b = 0.3) (h3 : c = (3 : ℝ))
    (exp_positive : 2^0.3 > 0)
    (log1_negative : log 0.3 2 < 0)
    (log2_negative : log 0.3 3 < 0)
    (log_decreasing : ∀ x y : ℝ, 0 < x ∧ 0 < y ∧ x < 1 ∧ y < 1 ∧ x < y → log 0.3 x > log 0.3 y)
    (h3_gt_2 : 3 > 2) : 
    log 0.3 3 < log 0.3 2 ∧ log 0.3 2 < 2^0.3 :=
by
  sorry

end arrange_numbers_l774_774957


namespace smallest_third_value_geom_prog_l774_774077

theorem smallest_third_value_geom_prog :
  ∃ d : ℝ, 
    (7, 7 + d, 7 + 2 * d) ∧ (10 + d) * (10 + d) = 7 * (22 + 2 * d) ∧ 
    ∀ term1 term2 term3, term1 = 7 ∧ term2 = 10 + d ∧ term3 = 22 + 2 * d 
    → term3 ≥ -1 :=
by sorry

end smallest_third_value_geom_prog_l774_774077


namespace smallest_possible_QNNN_l774_774408

theorem smallest_possible_QNNN :
  ∃ (Q N : ℕ), (N = 1 ∨ N = 5 ∨ N = 6) ∧ (NN = 10 * N + N) ∧ (Q * 1000 + NN * 10 + N = NN * N) ∧ (Q * 1000 + NN * 10 + N) = 275 :=
sorry

end smallest_possible_QNNN_l774_774408


namespace slope_of_midpoints_l774_774404

-- Define the endpoints of the segments
def A := (1, 2 : ℝ)
def B := (3, 5 : ℝ)
def C := (4, 1 : ℝ)
def D := (7, 7 : ℝ)

-- Define the midpoint function for a segment
def midpoint (P Q : ℝ × ℝ) : ℝ × ℝ :=
  ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

-- Midpoints of the given segments
def mid_AB := midpoint A B
def mid_CD := midpoint C D

-- Define the slope function between two points
def slope (P Q : ℝ × ℝ) : ℝ :=
  (Q.2 - P.2) / (Q.1 - P.1)

-- The theorem stating the slope of the line containing the midpoints
theorem slope_of_midpoints :
  slope mid_AB mid_CD = 1 / 7 :=
by sorry

end slope_of_midpoints_l774_774404


namespace inequality_one_inequality_two_l774_774322

theorem inequality_one (a b : ℝ) : 
    a^2 + b^2 ≥ (a + b)^2 / 2 := 
by
    sorry

theorem inequality_two (a b : ℝ) : 
    a^2 + b^2 ≥ 2 * (a - b - 1) := 
by
    sorry

end inequality_one_inequality_two_l774_774322


namespace seventh_degree_solution_l774_774034

theorem seventh_degree_solution (a b x : ℝ) :
  (x^7 - 7 * a * x^5 + 14 * a^2 * x^3 - 7 * a^3 * x = b) ↔
  ∃ α β : ℝ, α + β = x ∧ α * β = a ∧ α^7 + β^7 = b :=
by
  sorry

end seventh_degree_solution_l774_774034


namespace geometric_sequence_common_ratio_l774_774901

-- Define a sequence as a list of real numbers
def seq : List ℚ := [8, -20, 50, -125]

-- Define the common ratio of a geometric sequence
def common_ratio (l : List ℚ) : ℚ := l.head! / l.tail!.head!

-- The theorem to prove the common ratio is -5/2
theorem geometric_sequence_common_ratio :
  common_ratio seq = -5 / 2 :=
by
  sorry

end geometric_sequence_common_ratio_l774_774901


namespace problem_l774_774260

def sequence (a : ℕ → ℤ) : Prop :=
  a 1 = 0 ∧ ∀ n, a (n + 2) + (-1)^(n) * a n = 2

def S_n (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  ∑ i in Finset.range n, a (i + 1)

theorem problem (a : ℕ → ℤ) (h : sequence a) :
  S_n a 2016 - S_n a 2013 = 2016 :=
sorry

end problem_l774_774260


namespace supplementary_angle_proof_l774_774574

noncomputable def complementary_angle (α : ℝ) : ℝ := 125 + 12 / 60

noncomputable def calculate_angle (c : ℝ) := 180 - c

noncomputable def supplementary_angle (α : ℝ) := 90 - α

theorem supplementary_angle_proof :
    let α := calculate_angle (complementary_angle α)
    supplementary_angle α = 35 + 12 / 60 := 
by
  sorry

end supplementary_angle_proof_l774_774574


namespace compute_star_of_negatives_l774_774360

noncomputable def operation (a b : ℝ) : ℝ := (a - b) / (1 - a * b)

theorem compute_star_of_negatives :
  operation (-1) (operation (-2) (operation (-3) ( ... (operation (-99) (-100)) ...))) = (-1) * x := by
  sorry

end compute_star_of_negatives_l774_774360


namespace max_trains_after_5_years_l774_774311

noncomputable def model_trains (years: ℕ) (birthday_trains: ℕ) (christmas_trains: ℕ) (parents_doubling: ℕ → ℕ) : ℕ :=
  let trains_per_year := birthday_trains + christmas_trains in
  let total_trains := trains_per_year * years in
  parents_doubling total_trains

theorem max_trains_after_5_years : 
  model_trains 5 1 2 (λ x, 2 * x) = 30 :=
by
  rw model_trains
  -- further steps and computation could be inserted here if required
  sorry

end max_trains_after_5_years_l774_774311


namespace angle_sum_equals_l774_774187

variables {α : Type*} 
variables {O O' O'' : Point α} {T A B C D E I : Point α}

-- Definitions of the required geometrical conditions
def is_internally_tangent (O O' T : Point α) : Prop := sorry
def is_on_chord (A B C D E : Point α) (O : Point α) : Prop := sorry
def is_tangent (D E : Point α) (O' : Point α) : Prop := sorry
def intersects_at (DE A O' I : Point α) : Prop := sorry

-- The main proof problem statement
theorem angle_sum_equals (h1 : is_internally_tangent O O' T)
                        (h2 : is_on_chord A B C D E O)
                        (h3 : is_tangent D E O')
                        (h4 : intersects_at DE A O' I):
  ∠ABI + ∠ACI = ∠BTI := sorry

end angle_sum_equals_l774_774187


namespace shadow_stretch_rate_is_5_feet_per_hour_l774_774138

-- Given conditions
def shadow_length_in_inches (hours_past_noon : ℕ) : ℕ := 360
def hours_past_noon : ℕ := 6

-- Convert inches to feet
def inches_to_feet (inches : ℕ) : ℕ := inches / 12

-- Calculate rate of increase of shadow length per hour
def rate_of_shadow_stretch_per_hour : ℕ := inches_to_feet (shadow_length_in_inches hours_past_noon) / hours_past_noon

theorem shadow_stretch_rate_is_5_feet_per_hour :
  rate_of_shadow_stretch_per_hour = 5 := by
  sorry

end shadow_stretch_rate_is_5_feet_per_hour_l774_774138


namespace ship_distance_change_l774_774937

theorem ship_distance_change (r : ℝ) (X A B C D : Point) :
  (dist X A = r) →
  (dist X B = r) →
  (dist X C = r * sqrt 2) →
  (dist X D = r * sqrt 2) →
  (length (segment B C) = r) →
  (length (segment C D) = 2 * r) →
  perpendicular (radius_at B) (segment B C) →
  parallel (tangent_at B) (segment C D) →
  dist X D = r * sqrt 2 :=
by
  sorry

end ship_distance_change_l774_774937


namespace find_k_l774_774949

-- Let's define the conditions need as per the problem statement.
open_locale real

-- Definitions for the context of the problem.
def is_equilateral_triangle (A B C : Type) (triangle : Triangle A B C) : Prop := 
  (triangle.angle A B C = π / 3) ∧ 
  (triangle.angle B C A = π / 3) ∧ 
  (triangle.angle C A B = π / 3)

-- Statement problem to be proven in Lean 4.
theorem find_k (A B C D : Type) (triangle : Triangle A B C) 
  (h_equilateral : is_equilateral_triangle A B C triangle) 
  (h_angle : triangle.angle B A C = 2 * triangle.angle D) 
  (k : ℝ) : 
  triangle.angle A B C = k * π → k = 1 / 3 := 
sorry

end find_k_l774_774949


namespace trees_planted_l774_774004

-- Definitions for the quantities of lindens (x) and birches (y)
variables (x y : ℕ)

-- Definitions matching the given problem conditions
def condition1 := x + y > 14
def condition2 := y + 18 > 2 * x
def condition3 := x > 2 * y

-- The theorem stating that if the conditions hold, then x = 11 and y = 5
theorem trees_planted (h1 : condition1 x y) (h2 : condition2 x y) (h3 : condition3 x y) : 
  x = 11 ∧ y = 5 := 
sorry

end trees_planted_l774_774004


namespace sum_of_reciprocal_squares_of_centroid_coords_eq_36_l774_774083

theorem sum_of_reciprocal_squares_of_centroid_coords_eq_36
  (α β γ : ℝ)
  (h_sum : α + β + γ = 10)
  (h_dist : 1 / (sqrt (1 / (α^2) + 1 / (β^2) + 1 / (γ^2))) = 2)
  (p q r : ℝ)
  (h_centroid : p = α / 3 ∧ q = β / 3 ∧ r = γ / 3) :
  1 / p^2 + 1 / q^2 + 1 / r^2 = 36 := by
  sorry

end sum_of_reciprocal_squares_of_centroid_coords_eq_36_l774_774083


namespace ratio_john_to_total_cost_l774_774746

noncomputable def cost_first_8_years := 8 * 10000
noncomputable def cost_next_10_years := 10 * 20000
noncomputable def university_tuition := 250000
noncomputable def cost_john_paid := 265000
noncomputable def total_cost := cost_first_8_years + cost_next_10_years + university_tuition

theorem ratio_john_to_total_cost : (cost_john_paid / total_cost : ℚ) = 1 / 2 := by
  sorry

end ratio_john_to_total_cost_l774_774746


namespace smallest_number_among_bases_l774_774108

def convert_to_decimal (base : ℕ) (digits : list ℕ) : ℕ :=
  digits.reverse.enum.map (λ ⟨i, d⟩, d * base ^ i).sum

theorem smallest_number_among_bases :
  let d20_7 := convert_to_decimal 7 [2, 0] in
  let d30_5 := convert_to_decimal 5 [3, 0] in
  let d23_6 := convert_to_decimal 6 [2, 3] in
  let d31_4 := convert_to_decimal 4 [3, 1] in
  d31_4 < d20_7 ∧ d31_4 < d30_5 ∧ d31_4 < d23_6 :=
by
  let d20_7 := convert_to_decimal 7 [2, 0]
  let d30_5 := convert_to_decimal 5 [3, 0]
  let d23_6 := convert_to_decimal 6 [2, 3]
  let d31_4 := convert_to_decimal 4 [3, 1]
  show d31_4 < d20_7 ∧ d31_4 < d30_5 ∧ d31_4 < d23_6
  sorry

end smallest_number_among_bases_l774_774108


namespace volume_of_sphere_is_pi_over_6_l774_774697

-- Define the surface area of the cube
def surface_area_cube : ℝ := 6

-- Define the edge length of the cube based on its surface area
def edge_length_cube (A : ℝ) : ℝ := real.sqrt (A / 6)

-- Define the radius of the sphere inscribed in the cube
def radius_sphere (a : ℝ) : ℝ := a / 2

-- Define the volume of the sphere
def volume_sphere (r : ℝ) : ℝ := (4 / 3) * real.pi * r^3

-- The proof statement that the volume of the sphere is π/6 given the conditions
theorem volume_of_sphere_is_pi_over_6 :
  volume_sphere (radius_sphere (edge_length_cube surface_area_cube)) = real.pi / 6 :=
by
  -- Proof steps would go here
  sorry

end volume_of_sphere_is_pi_over_6_l774_774697


namespace ball_arrangements_correct_l774_774711

-- Define the number of balls of each color
def black_balls := 2
def white_balls := 3
def red_balls := 4

-- Define the condition: no black ball is next to a white ball
def no_black_next_to_white (arrangement : List Nat) : Prop :=
  ∀ i, (i < arrangement.length - 1) → 
       (arrangement[i] = 1 → arrangement[i+1] ≠ 2) ∧ 
       (arrangement[i] = 2 → arrangement[i+1] ≠ 1)

-- Given the number of balls, the total arrangements meeting the condition
def total_arrangements := 200

-- Statement to prove in Lean 4
theorem ball_arrangements_correct :
  ∃ arrangements : List (List Nat),
    (arrangements.length = total_arrangements) ∧
    (∀ arrangement ∈ arrangements,
      arrangement.count 1 = black_balls ∧
      arrangement.count 2 = white_balls ∧
      arrangement.count 3 = red_balls ∧
      no_black_next_to_white arrangement) :=
sorry

end ball_arrangements_correct_l774_774711


namespace probability_xy_minus_x_minus_y_odd_l774_774854

theorem probability_xy_minus_x_minus_y_odd :
  let S := {x | x ∈ Finset.range 16 \ {0}} in
  (∀ x y : ℕ, x ∈ S ∧ y ∈ S ∧ x ≠ y → 
    (xy - x - y).odd) = (4 / 5) := sorry

end probability_xy_minus_x_minus_y_odd_l774_774854


namespace profit_or_loss_l774_774899

-- Given conditions
def original_price_A := 23.04 / 1.44
def original_price_B := 23.04 / 0.64
def selling_price_A := 23.04
def selling_price_B := 23.04

-- Mathematical statement to be proved
theorem profit_or_loss (x y : ℝ) (hx : x = original_price_A) (hy : y = original_price_B) :
  (x + y) - (selling_price_A + selling_price_B) = 5.92 :=
by
  sorry

end profit_or_loss_l774_774899


namespace total_time_climbing_stairs_l774_774744

theorem total_time_climbing_stairs :
  let a := 30
  let d := 7
  let n := 8
  let sum_arithmetic_series (a d n : ℕ) : ℕ := n * (2 * a + (n - 1) * d) / 2
  in sum_arithmetic_series a d n = 436 :=
by
  let a := 30
  let d := 7
  let n := 8
  let sum_arithmetic_series (a d n : ℕ) : ℕ := n * (2 * a + (n - 1) * d) / 2
  show sum_arithmetic_series a d n = 436
  sorry

end total_time_climbing_stairs_l774_774744


namespace min_value_f_l774_774570

noncomputable def f (a b : ℝ) : ℝ := (1 / a^5 + a^5 - 2) * (1 / b^5 + b^5 - 2)

theorem min_value_f :
  ∀ (a b : ℝ), a > 0 → b > 0 → a + b = 1 → f a b ≥ (31^4 / 32^2) :=
by
  intros
  sorry

end min_value_f_l774_774570


namespace general_term_l774_774592

def seq (a : ℕ → ℝ) : Prop :=
  (a 1 = 1 / 2) ∧
  (a 2 = 1 / 3 * (1 - a 1)) ∧
  (a 3 = 1 / 4 * (1 - a 1 - a 2)) ∧
  (∀ n : ℕ, n > 3 → a (n+1) = 1 / (n+2) * (1 - ∑ i in range (n+1), a i))

theorem general_term (a : ℕ → ℝ) (h : seq a) (n : ℕ) (hn : n > 0) : 
  a n = 1 / (n * (n + 1)) :=
sorry

end general_term_l774_774592


namespace decreasing_iff_a_in_range_l774_774561

variable {a : ℝ} (n : ℕ)
variable {a_pos : 0 < a}
variable {a_ne_one : a ≠ 1}

def a_n := (n : ℝ) * a^n

theorem decreasing_iff_a_in_range : (∀ n : ℕ, 0 < n → a_n n > a_n (n + 1)) ↔ (0 < a ∧ a < 1/2) := by
  sorry

end decreasing_iff_a_in_range_l774_774561


namespace arithmetic_sequence_a9_l774_774835

noncomputable def S (n : ℕ) (a₁ aₙ : ℝ) : ℝ := (n * (a₁ + aₙ)) / 2

theorem arithmetic_sequence_a9 (a₁ a₁₇ : ℝ) (h1 : S 17 a₁ a₁₇ = 102) : (a₁ + a₁₇) / 2 = 6 :=
by
  sorry

end arithmetic_sequence_a9_l774_774835


namespace new_volume_increased_dimensions_l774_774929

theorem new_volume_increased_dimensions (l w h : ℝ) 
  (h_vol : l * w * h = 5000) 
  (h_sa : l * w + w * h + h * l = 900) 
  (h_sum_edges : l + w + h = 60) : 
  (l + 2) * (w + 2) * (h + 2) = 7048 := 
by 
  sorry

end new_volume_increased_dimensions_l774_774929


namespace symmetric_about_x_axis_l774_774713

noncomputable def P (a b : ℝ) : Prop := P = (a, 1)
noncomputable def Q (a b : ℝ) : Prop := Q = (2, b)

theorem symmetric_about_x_axis (a b : ℝ) (h1 : a = 2) (h2 : 1 = -b) : a + b = 1 :=
by {
  sorry
}

end symmetric_about_x_axis_l774_774713


namespace sum_of_squares_eq_l774_774721

variable (a : ℕ → ℕ)
variable (n : ℕ)

-- Condition: For any natural number n, a₁ + 2a₂ + 2²a₃ + ... + 2ⁿ⁻¹aₙ = 2²ⁿ - 1
axiom sequence_condition 
  (h : ∀ n : ℕ, ∑ i in Finset.range n, 2^i * a (i + 1) = 2^(2 * n) - 1)

-- Theorem to prove: a₁² + a₂² + a₃² + ... + aₙ² = 3(4ⁿ - 1)
theorem sum_of_squares_eq (h : ∀ n : ℕ, ∑ i in Finset.range n, 2^i * a (i + 1) = 2^(2 * n) - 1) :
  ∑ i in Finset.range n, (a (i + 1))^2 = 3 * (4^n - 1) :=
by
  sorry

end sum_of_squares_eq_l774_774721


namespace section_perimeter_le_max_face_perimeter_l774_774793

structure Tetrahedron where
  a1 a2 a3 a4 a5 a6 : ℝ

def perimeter (x y z : ℝ) : ℝ :=
  x + y + z

noncomputable def max_perimeter (T : Tetrahedron) : ℝ :=
  max (perimeter T.a1 T.a2 T.a4)
    (max (perimeter T.a2 T.a3 T.a5)
      (max (perimeter T.a1 T.a3 T.a6)
        (perimeter T.a4 T.a5 T.a6)))

theorem section_perimeter_le_max_face_perimeter (T : Tetrahedron) (s : set (Tetrahedron → ℝ)) :
  ∃ P : ℝ, (∀ section ∈ s, P ≤ max_perimeter T) := by
  sorry

end section_perimeter_le_max_face_perimeter_l774_774793


namespace expr_evaluation_expr_neq_3_5_l774_774989

noncomputable def evaluate_expr : ℚ := 1 / (3 - (1 / (3 - (1 / (3 - (1 / 3))))))

theorem expr_evaluation : evaluate_expr = 8 / 21 := by
  sorry

theorem expr_neq_3_5 : evaluate_expr ≠ 3 / 5 := by
  have h : evaluate_expr = 8 / 21 := by sorry
  rw [h]
  norm_num
  intro h_eq
  have : (21 * 3) = (8 * 5) := by simp [h_eq]
  norm_num at this

end expr_evaluation_expr_neq_3_5_l774_774989


namespace num_ways_select_with_constraints_l774_774834

theorem num_ways_select_with_constraints (total_students : ℕ) (select_students : ℕ) (A_and_B : ℕ) (A_not_and_B : ℕ) :
  total_students = 10 → select_students = 6 → A_and_B = 70 → A_not_and_B = 140 → 
  (finset.card (finset.powerset_len select_students (finset.range total_students)) - A_and_B) = A_not_and_B :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end num_ways_select_with_constraints_l774_774834


namespace AD_bisects_OB_l774_774174

noncomputable theory

variables {O A B C D : Point}
variables {k : Circle} (h1 : k.tangent O A B) 
variables (h2 : Line.parallel (Line.mk A B) (Line.mk O B))
variables (h3 : Circle.intersect_at k (Line.mk O C) D)
variables {Bisects : Line.bisects (Line.mk A D) (Segment.mk O B)}

theorem AD_bisects_OB (O A B C D : Point) (k : Circle) 
  (h1 : k.tangent O A B)
  (h2 : Line.parallel (Line.mk A B) (Line.mk O B))
  (h3 : Circle.intersect_at k (Line.mk O C) D)
  : Line.bisects (Line.mk A D) (Segment.mk O B) :=
sorry

end AD_bisects_OB_l774_774174


namespace race_result_l774_774705

noncomputable def distance_beat (speed: ℝ) (time_difference: ℝ) : ℝ :=
  speed * time_difference

theorem race_result :
  let total_distance := 1000  -- meters
  let time_a := 210.22222222222223  -- seconds
  let time_diff := 12 -- seconds
  let speed_a := total_distance / time_a
  distance_beat speed_a time_diff ≈ 57.072 :=
by
  sorry

end race_result_l774_774705


namespace prism_visibility_percentage_l774_774463

theorem prism_visibility_percentage
  (base_edge : ℝ)
  (height : ℝ)
  (cell_side : ℝ)
  (wraps : ℕ)
  (lateral_surface_area : ℝ)
  (transparent_area : ℝ) :
  base_edge = 3.2 →
  height = 5 →
  cell_side = 1 →
  wraps = 2 →
  lateral_surface_area = base_edge * height * 3 →
  transparent_area = 13.8 →
  (transparent_area / lateral_surface_area) * 100 = 28.75 :=
by
  intros h_base_edge h_height h_cell_side h_wraps h_lateral_surface_area h_transparent_area
  sorry

end prism_visibility_percentage_l774_774463


namespace max_regions_with_three_triangles_l774_774273

theorem max_regions_with_three_triangles : 
  ∀ (triangles : List (List (ℝ × ℝ))), triangles.length = 3 →
  is_triangle (triangles.nth 0).getOrElse [] →
  is_triangle (triangles.nth 1).getOrElse [] →
  is_triangle (triangles.nth 2).getOrElse [] →
  ∃ (regions : ℕ), regions = 20 := 
by
  sorry

def is_triangle (points : List (ℝ × ℝ)) : Prop :=
  points.length = 3 ∧
  distinct points ∧
  collinear points = false

end max_regions_with_three_triangles_l774_774273


namespace angle_between_vectors_perpendicular_lambda_l774_774594

def vec2 := (ℝ × ℝ)

def a : vec2 := (1, 2)
def b : vec2 := (-3, 4)
def a_plus_b : vec2 := (1 - 3, 2 + 4)
def a_minus_b : vec2 := (1 + 3, 2 - 4)
def dot_product (u v : vec2) : ℝ := u.1 * v.1 + u.2 * v.2
def norm (u : vec2) : ℝ := Real.sqrt (u.1 ^ 2 + u.2 ^ 2)
def angle (u v : vec2) : ℝ := Real.arccos (dot_product u v / (norm u * norm v))

theorem angle_between_vectors :
  angle a_plus_b a_minus_b = (3 * Real.pi) / 4 :=
sorry

theorem perpendicular_lambda : ∀ (λ : ℝ),
  dot_product a (a.1 - 3 * λ, a.2 + 4 * λ) = 0 → λ = -1 :=
sorry

end angle_between_vectors_perpendicular_lambda_l774_774594


namespace train_crossing_time_l774_774453

-- Definitions of the conditions
def train_speed_km_per_hr : ℕ := 72
def platform_length_m : ℕ := 150
def train_length_m : ℕ := 370

-- The conclusion that we want to prove
theorem train_crossing_time : 
  let speed_m_per_s := train_speed_km_per_hr * 5 / 18,
      total_distance_m := train_length_m + platform_length_m,
      crossing_time_s := total_distance_m / speed_m_per_s
  in crossing_time_s = 26 := 
by
  intros,
  have h1 : speed_m_per_s = 20, 
  { sorry }, -- speed conversion proof temporarily omitted
  have h2 : total_distance_m = 520,
  { sorry }, -- distance calculation proof temporarily omitted
  have h3 : crossing_time_s = 520 / 20,
  { sorry }, -- time calculation proof temporarily omitted
  have h4 : 520 / 20 = 26,
  { sorry }, -- final calculation proof temporarily omitted
  exact eq.trans h3 h4

end train_crossing_time_l774_774453


namespace correct_statement_about_surveys_l774_774412

-- Definitions for each statement
def statement_A : Prop :=
  ∀ survey : Type, (comprehensive survey → suitable survey)

def statement_B : Prop :=
  (∀ student : Type, (vision_test student → comprehensive survey))

def statement_C : Prop :=
  ∀ community household : Type, (sample household from community) → sample_size (sample household from community) = 1500

def statement_D : Prop :=
  ∀ school student basketball_team : Type, (height_sample basketball_team) → objective_estimate school student (height_sample basketball_team)

-- The theorem to prove
theorem correct_statement_about_surveys (A B C D : Prop) (hA : statement_A) (hB : statement_B) (hC : statement_C) (hD : statement_D) : B :=
sorry

end correct_statement_about_surveys_l774_774412


namespace new_sequence_2003rd_term_l774_774111

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def without_perfect_squares (s : List ℕ) : List ℕ :=
  s.filter (λ n => ¬ is_perfect_square n)

def seq := List.range (2049)  -- Will contain [0, 1, 2, ..., 2048]
                               -- But we start from 0, need to adjust index

def new_sequence := without_perfect_squares (seq.tail)  -- Removing 0-index and squares

#eval new_sequence.nth 2002  -- Should be 2048 if new_sequence is correct

theorem new_sequence_2003rd_term : (new_sequence.nth 2002).get_or_else 0 = 2048 :=
by sorry

end new_sequence_2003rd_term_l774_774111


namespace length_CD_l774_774261

-- Define the conditions as a triangle ABC with given sides
variables {A B C D : Type}
variables {AB BC AC : ℝ}
variables (hAB : AB = 8) (hBC : BC = 15) (hAC : AC = 17)
variables (hBisector : true) -- \(CD\) is the angle bisector of \(\angle C\)

-- Define the theorem to prove the correct answer
theorem length_CD 
    (A B C D : Type) 
    (AB BC AC : ℝ) 
    [hAB : AB = 8] 
    [hBC : BC = 15] 
    [hAC : AC = 17]
    (hBisector : true) : 
    ∃ CD : ℝ, CD = real.sqrt 125950 / 23 :=
sorry

end length_CD_l774_774261


namespace no_hyperdeficient_integers_l774_774759

def sum_of_divisors (n : ℕ) : ℕ := (Nat.divisors n).sum

def is_hyperdeficient (n : ℕ) : Prop :=
  sum_of_divisors (sum_of_divisors n) = n + 4

theorem no_hyperdeficient_integers : ∀ n : ℕ, ¬ is_hyperdeficient n :=
by
  intros n
  sorry

end no_hyperdeficient_integers_l774_774759


namespace trajectory_of_M_l774_774593

-- Define the two circles C1 and C2
def C1 (x y : ℝ) : Prop := (x + 3)^2 + y^2 = 1
def C2 (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 9

-- Define the condition for the moving circle M being tangent to both circles
def isTangent (Mx My : ℝ) : Prop := 
  let distC1 := (Mx + 3)^2 + My^2
  let distC2 := (Mx - 3)^2 + My^2
  distC2 - distC1 = 4

-- The equation of the trajectory of M
theorem trajectory_of_M (Mx My : ℝ) (h : isTangent Mx My) : 
  Mx^2 - (My^2 / 8) = 1 ∧ Mx < 0 :=
sorry

end trajectory_of_M_l774_774593


namespace find_q_minus_p_l774_774287

theorem find_q_minus_p (p q : ℕ) (h1 : 0 < p) (h2 : 0 < q) 
  (h3 : 6 * q < 11 * p) (h4 : 9 * p < 5 * q) (h_min : ∀ r : ℕ, r > 0 → (6:ℚ)/11 < (p:ℚ)/r → (p:ℚ)/r < (5:ℚ)/9 → q ≤ r) :
  q - p = 9 :=
sorry

end find_q_minus_p_l774_774287


namespace integer_solutions_count_eq_3_l774_774604

theorem integer_solutions_count_eq_3 : 
  {x : ℤ | (x^2 - 3 * x + 2)^(x + 1) = 1}.finite.card = 3 := 
by
  sorry

end integer_solutions_count_eq_3_l774_774604


namespace chord_length_of_line_circle_l774_774520

-- Definitions for the conditions
def line_eq (x y : ℝ) : Prop := 3 * x - 4 * y + 3 = 0
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 1

-- The main theorem statement
theorem chord_length_of_line_circle :
  let r : ℝ := 1,
      d : ℝ := 3 / 5
  in 2 * real.sqrt (r^2 - d^2) = 8 / 5 := by
    sorry

end chord_length_of_line_circle_l774_774520


namespace max_sine_sum_l774_774235

theorem max_sine_sum (a b c A B C : ℝ) (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ A > 0 ∧ B > 0 ∧ C > 0)
  (h_sides_angles : c * sin A = sqrt 3 * a * cos C) : ∃ M : ℝ, M = sqrt 3 ∧ ∀ A B : ℝ, sin A + sin B ≤ M :=
sorry

end max_sine_sum_l774_774235


namespace coefficient_x3_in_expansion_l774_774113

theorem coefficient_x3_in_expansion : 
  (∃ (r : ℕ), 5 - r / 2 = 3 ∧ 2 * Nat.choose 5 r = 10) :=
by 
  sorry

end coefficient_x3_in_expansion_l774_774113


namespace eight_by_eight_grid_possible_l774_774882

def isValidGrid (grid : Fin 8 → Fin 8 → ℕ) : Prop :=
  (∀ col1 col2 : Fin 8, (∑ row: Fin 8, grid row col1) = (∑ row: Fin 8, grid row col2)) ∧
  (∀ row1 row2 : Fin 8, row1 ≠ row2 → (∑ col: Fin 8, grid row1 col) ≠ (∑ col: Fin 8, grid row2 col))

theorem eight_by_eight_grid_possible : 
  ∃ (grid : Fin 8 → Fin 8 → ℕ), isValidGrid grid :=
sorry

end eight_by_eight_grid_possible_l774_774882


namespace min_shaded_triangles_l774_774479

/-- 
Given an equilateral triangle with side length 8, divided into smaller equilateral triangles 
with side length 1. Prove that the minimum number of smaller triangles that need to be shaded such that 
every intersection point of the lines (including those on the edges) is a vertex of at least one shaded triangle is 15.
-/
theorem min_shaded_triangles (side_length_big : ℕ) (side_length_small : ℕ) 
  (h_big : side_length_big = 8) (h_small : side_length_small = 1) : 
  ∃ min_triangles : ℕ, min_triangles = 15 :=
by
  use 15
  sorry

end min_shaded_triangles_l774_774479


namespace sum_of_digits_of_n_l774_774286

theorem sum_of_digits_of_n :
  ∃ n : ℕ,
    n > 2000 ∧
    n + 135 % 75 = 15 ∧
    n + 75 % 135 = 45 ∧
    (n = 2025 ∧ (2 + 0 + 2 + 5 = 9)) :=
by
  sorry

end sum_of_digits_of_n_l774_774286


namespace quadratic_trinomial_has_two_roots_l774_774923

theorem quadratic_trinomial_has_two_roots
  (a b c : ℝ) (h : b^2 - 4 * a * c > 0) : (2 * (a + b))^2 - 4 * 3 * a * (b + c) > 0 := by
  sorry

end quadratic_trinomial_has_two_roots_l774_774923


namespace eval_sqrt_4_8_pow_12_l774_774128

theorem eval_sqrt_4_8_pow_12 : ((8 : ℝ)^(1 / 4))^12 = 512 :=
by
  -- This is where the proof steps would go 
  sorry

end eval_sqrt_4_8_pow_12_l774_774128


namespace closest_fraction_to_winning_ratio_l774_774707

theorem closest_fraction_to_winning_ratio :
  let winning_fraction := (23 : ℚ) / 150 in
  (|winning_fraction - 1/5| > |winning_fraction - 1/7| ∧
   |winning_fraction - 1/6| > |winning_fraction - 1/7| ∧
   |winning_fraction - 1/8| > |winning_fraction - 1/7| ∧
   |winning_fraction - 1/9| > |winning_fraction - 1/7|)
:=
by
  sorry

end closest_fraction_to_winning_ratio_l774_774707


namespace slope_range_of_line_l774_774172

/-- A mathematical proof problem to verify the range of the slope of a line
that passes through a given point (-1, -1) and intersects a circle. -/
theorem slope_range_of_line (
  k : ℝ
) : (∃ x y : ℝ, (y + 1 = k * (x + 1)) ∧ (x - 2) ^ 2 + y ^ 2 = 1) ↔ (0 < k ∧ k < 3 / 4) := 
by
  sorry  

end slope_range_of_line_l774_774172


namespace area_of_triangle_ABC1_is_5_area_of_triangle_ABC2_is_5_l774_774569

structure Point :=
  (x : ℝ)
  (y : ℝ)

noncomputable def area_of_triangle (A B C : Point) : ℝ :=
  1 / 2 * abs (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y))

def A : Point := ⟨3, 4⟩
def B : Point := ⟨6, 6⟩
def C1 : Point := ⟨0, 16 / 3⟩
def C2 : Point := ⟨0, -4 / 3⟩

theorem area_of_triangle_ABC1_is_5 : area_of_triangle A B C1 = 5 := by
  sorry

theorem area_of_triangle_ABC2_is_5 : area_of_triangle A B C2 = 5 := by
  sorry

end area_of_triangle_ABC1_is_5_area_of_triangle_ABC2_is_5_l774_774569


namespace find_coordinates_of_C_l774_774317

/-
Points A(13, 11) and B(5, -1) are vertices of ΔABC with AB = AC. 
The altitude from A meets the opposite side at D(2, 7).
What are the coordinates of point C?
-/
theorem find_coordinates_of_C :
  let A := (13, 11 : ℝ × ℝ)
  let B := (5, -1 : ℝ × ℝ)
  let D := (2, 7 : ℝ × ℝ) 
  ∃ C : ℝ × ℝ, 
    dist A B = dist A C ∧ 
    let M := ((B.1 + C.1)/2, (B.2 + C.2)/2 : ℝ × ℝ) in
    M = D ∧
    C = (-1, 15) :=
by
  sorry

end find_coordinates_of_C_l774_774317


namespace max_fraction_l774_774186

theorem max_fraction (x y : ℝ) (h1 : -6 ≤ x) (h2 : x ≤ -3) (h3 : 3 ≤ y) (h4 : y ≤ 5) :
  (∀ x y, -6 ≤ x → x ≤ -3 → 3 ≤ y → y ≤ 5 → (x - y) / y ≤ -2) :=
by
  sorry

end max_fraction_l774_774186


namespace workers_faster_when_together_l774_774006

-- Let T_A, T_B, T_C be the time taken by workers A, B, and C respectively to complete the work alone
variables (T_A T_B T_C : ℝ)

-- Define the condition given: T must be greater than 0 for all worker times
variable (h : T_A > 0 ∧ T_B > 0 ∧ T_C > 0)

-- Define the rate when taking shifts (condition sums up the inverse working times)
def rate_when_taking_shifts : ℝ := 0.5 * (1 / T_A + 1 / T_B) + 0.5 * (1 / T_B + 1 / T_C) + 0.5 * (1 / T_A + 1 / T_C)

-- Define the rate when working simultaneously
def rate_when_together : ℝ := 1 / (1 / T_A + 1 / T_B + 1 / T_C)

-- The mathematically equivalent statement
theorem workers_faster_when_together (h : T_A > 0 ∧ T_B > 0 ∧ T_C > 0) :
  rate_when_together T_A T_B T_C = 2.5 * rate_when_taking_shifts T_A T_B T_C :=
sorry

end workers_faster_when_together_l774_774006


namespace perfect_square_factors_450_l774_774656

def prime_factorization (n : ℕ) : Prop :=
  n = 2^1 * 3^2 * 5^2

theorem perfect_square_factors_450 : prime_factorization 450 → ∃ n : ℕ, n = 4 :=
by
  sorry

end perfect_square_factors_450_l774_774656


namespace purchase_price_of_first_commodity_l774_774811

-- Define the conditions
variable (price_first price_second : ℝ)
variable (h1 : price_first - price_second = 127)
variable (h2 : price_first + price_second = 827)

-- Prove the purchase price of the first commodity is $477
theorem purchase_price_of_first_commodity : price_first = 477 :=
by
  sorry

end purchase_price_of_first_commodity_l774_774811


namespace range_of_b_l774_774585

noncomputable def f (x : ℝ) : ℝ := ln x - (1 / 4) * x + (3 / (4 * x)) - 1

noncomputable def g (x b : ℝ) : ℝ := -x^2 + 2 * b * x - 4

theorem range_of_b (b : ℝ) : 
  (∀ x1 ∈ set.Ioo 0 2, ∀ x2 ∈ set.Icc 1 2, f x1 ≥ g x2 b) → b ≤ Real.sqrt 14 / 2 :=
sorry

end range_of_b_l774_774585


namespace positive_integer_iff_positive_x_l774_774158

theorem positive_integer_iff_positive_x (x : ℝ) (hx : x ≠ 0) : 
  (∃ k : ℤ, k > 0 ∧ (|x - 3 * |x|| / x = k)) ↔ (x > 0) := 
sorry

end positive_integer_iff_positive_x_l774_774158


namespace number_of_perfect_square_divisors_of_450_l774_774633

theorem number_of_perfect_square_divisors_of_450 : 
    let p := 450;
    let factors := [(3, 2), (5, 2), (2, 1)];
    ∃ n, (n = 4 ∧ 
          ∀ (d : ℕ), d ∣ p → 
                     (∃ (a b c : ℕ), d = 2^a * 3^b * 5^c ∧ 
                              (a = 0) ∧ (b = 0 ∨ b = 2) ∧ (c = 0 ∨ c = 2) → 
                              a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 = 0)) :=
    sorry

end number_of_perfect_square_divisors_of_450_l774_774633


namespace measure_of_one_exterior_angle_l774_774460

theorem measure_of_one_exterior_angle (n : ℕ) (h : n > 2) : 
  n > 2 → ∃ (angle : ℝ), angle = 360 / n :=
by 
  sorry

end measure_of_one_exterior_angle_l774_774460


namespace numPerfectSquareFactorsOf450_l774_774683

def isPerfectSquare (n : Nat) : Prop :=
  ∃ k : Nat, n = k * k

theorem numPerfectSquareFactorsOf450 : 
  ∃! n : Nat, 
    (∀ d : Nat, d ∣ 450 → isPerfectSquare d) → n = 4 := 
by
  sorry

end numPerfectSquareFactorsOf450_l774_774683


namespace regression_line_quadrants_l774_774078

def regression_line_passes_through (b a : ℝ) : Prop :=
  ∀ (x : ℝ), x ≠ 1.5 → (b > 0) → (((b * x + a) < 0) ∧ x < 1.5) ∨ ((b * x + a) > 0 ∧ x > 1.5)

theorem regression_line_quadrants
  (b a : ℝ)
  (hx_mean : 1.5)
  (hy_mean : -4.3)
  (h_slope : b > 0)
  (h_passes_through : regression_line_passes_through b a) :
    true := 
sorry

end regression_line_quadrants_l774_774078


namespace coefficient_of_x2_in_f_l774_774776

noncomputable def f (x : ℝ) : ℝ :=
  (1 / 2 * x - Real.sqrt 2) ^ (3 * ∫ (x : ℝ) in (-Real.pi / 2)..(Real.pi / 2), Real.cos x)

theorem coefficient_of_x2_in_f :
  (∃ (c : ℝ), c = 15 ∧
    (∃ (r : ℕ), r = 4 ∧
      Polynomial.coeff (Polynomial.expand ℝ (1 / 2 * X - (Polynomial.C (Real.sqrt 2))).toExpr (6 : ℕ)) 2 = c)) :=
by
  sorry

end coefficient_of_x2_in_f_l774_774776


namespace polynomial_remainder_division_l774_774161

theorem polynomial_remainder_division
  (a b : ℝ) (f : ℝ → ℝ) (c d : ℝ)
  (h1 : a ≠ b)
  (h2 : f a = c)
  (h3 : f b = d) :
  ∃ g : ℝ → ℝ, f = (λ x, (x - a) * (x - b) * g x + (c - d) / (a - b) * x + (a * d - b * c) / (a - b)) :=
by {
  sorry -- The proof would go here
}

end polynomial_remainder_division_l774_774161


namespace ball_arrangements_l774_774708

theorem ball_arrangements (black white red : ℕ) (h_black : black = 2) (h_white : white = 3) (h_red : red = 4) :
  ∑ (arrangements : (Fin (black + white + red) → ℕ) (h_no_adj : ∀ i, arrangements (i - 1) = arrangements (i + 1) → arrangements i ≠ 1 ∨ arrangements i ≠ 2), 1) = 200 :=
sorry

end ball_arrangements_l774_774708


namespace visits_vs_students_l774_774803

variable {p t : ℕ} (set_system : Finset (Finset ℕ))
variables (students : Finset ℕ) [decidable_eq ℕ]

-- Conditions
def valid_set_system :=
  ∀ (s : Finset ℕ), s ∈ set_system → (¬ ∀ x, x ∈ s) ∧
    (∀ (a b : ℕ), a ≠ b → ∃! (s : Finset ℕ), a ∈ s ∧ b ∈ s) ∧
    (∀ (a b : ℕ), (s1 ≠ s2 ∧ a ∈ s1 ∧ a ∈ s2 ∧ b ∈ s1 ∧ b ∈ s2) → false)

-- Goal
theorem visits_vs_students (h1 : valid_set_system set_system) (h2 : ∀ s ∈ set_system, ¬ (students ⊆ s)) 
(h3 : ∀ a b ∈ students, ∃ s ∈ set_system, a ∈ s ∧ b ∈ s) (h4 : ∀ s ∈ set_system, s.card > 1) : t ≥ p :=
sorry

end visits_vs_students_l774_774803


namespace digit_pairs_for_divisibility_by_36_l774_774348

theorem digit_pairs_for_divisibility_by_36 (A B : ℕ) :
  (0 ≤ A) ∧ (A ≤ 9) ∧ (0 ≤ B) ∧ (B ≤ 9) ∧
  (∃ k4 k9 : ℕ, (10 * 5 + B = 4 * k4) ∧ (20 + A + B = 9 * k9)) ↔ 
  ((A = 5 ∧ B = 2) ∨ (A = 1 ∧ B = 6)) :=
by sorry

end digit_pairs_for_divisibility_by_36_l774_774348


namespace difference_in_subset_l774_774117

theorem difference_in_subset (S : Finset ℕ) (hS : S = {1, 2, 3, 4, 5}) :
  ∀ A B : Finset ℕ, A ∪ B = S → A ∩ B = ∅ →
  ∃ a b c ∈ A, a - b = c ∨ ∃ a b c ∈ B, a - b = c := by
  sorry

end difference_in_subset_l774_774117


namespace average_trees_per_student_l774_774075

theorem average_trees_per_student :
  let num_students := 50
  let trees_per_student := [3, 4, 5, 6]
  let students := [20, 15, 10, 5]
  (students.zip trees_per_student).sum (λ ⟨n_s, t_s⟩, n_s * t_s) / num_students = 4 := 
by
  sorry

end average_trees_per_student_l774_774075


namespace probability_incorrect_pairs_l774_774853

theorem probability_incorrect_pairs 
  (k : ℕ) (h_k : k < 6)
  : let m := 7
    let n := 72
    m + n = 79 :=
by
  sorry

end probability_incorrect_pairs_l774_774853


namespace number_of_divisors_l774_774534

/-- 
If n is a positive integer such that 90 * n^2 has 90 positive integer divisors,
then 100 * n^4 has 245 positive integer divisors.
-/
theorem number_of_divisors (n : ℕ) (hn : 0 < n)
  (h : ∏ d in (finset.divisors (90 * n^2)), 1 = 90) :
  ∏ d in (finset.divisors (100 * n^4)), 1 = 245 := 
sorry

end number_of_divisors_l774_774534


namespace moles_of_CO2_formed_l774_774522

variables (HNO3 NaHCO3 NaNO3 CO2 H2O : Type) (one_mole : ℕ)

-- Define a balanced reaction
def balanced_reaction : Prop :=
  ∀ (hno3 na_hco3 : ℕ), (hno3 = 1 ∧ na_hco3 = 1) → 
  (hno3 + na_hco3 = (1 : ℕ) + (1 : ℕ))

theorem moles_of_CO2_formed
  (hno3_moles : ℕ) (na_hco3_moles : ℕ)
  (hno3_moles_eq : hno3_moles = 1) 
  (na_hco3_moles_eq : na_hco3_moles = 1) :
  (balanced_reaction one_mole hno3_moles) ∧ (balanced_reaction one_mole na_hco3_moles) → 
  one_mole = 1 :=
by
sory

end moles_of_CO2_formed_l774_774522


namespace area_of_triangle_ABC_is_25_l774_774046

open Real

variables {A B C O : Point}
variables (radius : ℝ) (bc : ℝ)
variables [triangle_data : TriangleData O A B C]
variables (is_diameter : bc = 2 * radius)
variables (radius_value : radius = 5)
variables (a_on_circle : PointOnCircle O A radius)
variables (ao_perpendicular_bc : Perpendicular AO BC)

theorem area_of_triangle_ABC_is_25
  (is_diameter : bc = 2 * radius)
  (radius_value : radius = 5)
  (a_on_circle : PointOnCircle O A radius)
  (ao_perpendicular_bc : Perpendicular AO BC) :
  area_of_triangle A B C = 25 :=
sorry

end area_of_triangle_ABC_is_25_l774_774046


namespace radius_of_smaller_semicircle_l774_774739

theorem radius_of_smaller_semicircle :
  ∃ x : ℝ, 0 < x ∧
    let AB := 6 in
    let AC := 12 - x in
    let BC := 6 + x in
    (AB = 6) ∧ 
    (AC = 12 - x) ∧ 
    (BC = 6 + x) ∧
    (AB^2 + AC^2 = BC^2) ∧
    x = 4 := 
by
  use 4
  split
  { exact zero_lt_four }
  split
  { reflexivity }
  split
  { reflexivity }
  split
  { reflexivity }
  { sorry }

end radius_of_smaller_semicircle_l774_774739


namespace prove_boat_rowing_problem_l774_774063

def boat_rowing_problem 
    (D_d := 78)  -- Distance downstream in km
    (T_d := 2)  -- Time downstream in hours
    (T_u := 2)  -- Time upstream in hours
    (V_s := 7)  -- Speed of the stream in km/h
: Prop :=
    let V_b := 32 in   -- Boat's speed in still water (derived)
    let D_u := 50 in   -- Distance upstream (what we aim to prove)
    D_d / T_d = 39 ∧         -- Downstream speed
    V_b = 39 - V_s ∧         -- Boat's speed in still water
    V_b - V_s = D_u / T_u  ∧ -- Upstream speed
    D_u = 50                     -- Distance rowed upstream

theorem prove_boat_rowing_problem : boat_rowing_problem := by
  sorry

end prove_boat_rowing_problem_l774_774063


namespace hyperbola_asymptote_l774_774435

theorem hyperbola_asymptote (m : ℝ) (h : m > 0) : 
  (∃ m : ℝ, m > 0 ∧ (\forall x y : ℝ, ((x + sqrt 3 * y = 0) ↔ (abs (y / x) = sqrt 3/m)))) → 
  m = sqrt 3 :=
sorry

end hyperbola_asymptote_l774_774435


namespace perception_permutations_l774_774483

theorem perception_permutations : 
  let total_letters := 10
  let e_count := 2
  let p_count := 2
  let i_count := 2
  let n_count := 2
  (Nat.factorial total_letters) / ((Nat.factorial e_count) * (Nat.factorial p_count) * (Nat.factorial i_count) * (Nat.factorial n_count)) = 226800 := by
begin
  sorry
end

end perception_permutations_l774_774483


namespace number_of_solids_with_two_identical_views_is_three_l774_774478

def solid_views (solid : String) : (String × String × String) :=
  match solid with
  | "Cube" => ("square", "square", "square")
  | "Cylinder" => ("rectangle", "rectangle", "circle")
  | "Cone" => ("triangle", "triangle", "circle")
  | "RegularQuadrangularPrism" => ("rectangle", "rectangle", "square")
  | "Sphere" => ("circle", "circle", "circle")
  | _ => ("", "", "")

def count_solids_with_two_identical_views : Nat :=
  ["Cube", "Cylinder", "Cone", "RegularQuadrangularPrism", "Sphere"].count (λ s, 
    let (v1, v2, v3) := solid_views s
    (v1 = v2 ∧ v2 ≠ v3) ∨ (v1 = v3 ∧ v1 ≠ v2) ∨ (v2 = v3 ∧ v1 ≠ v2))

theorem number_of_solids_with_two_identical_views_is_three :
  count_solids_with_two_identical_views = 3 :=
by
  sorry

end number_of_solids_with_two_identical_views_is_three_l774_774478


namespace ratio_of_sides_l774_774795

theorem ratio_of_sides 
  (a b c d : ℝ) 
  (h1 : (a * b) / (c * d) = 0.16) 
  (h2 : b / d = 2 / 5) : 
  a / c = 0.4 := 
by 
  sorry

end ratio_of_sides_l774_774795


namespace fraction_f_div_log10_2_eq_n_l774_774284

def pascals_triangle_sum (n : ℕ) : ℕ := 2^n

def f (n : ℕ) : ℝ := Real.log10 (pascals_triangle_sum n)

theorem fraction_f_div_log10_2_eq_n (n : ℕ) : (f n) / (Real.log10 2) = n :=
by
  sorry

end fraction_f_div_log10_2_eq_n_l774_774284


namespace area_transformation_l774_774280

theorem area_transformation (T : set (ℝ × ℝ)) (hT : measure_theory.measure (MeasureSpace.volume) T = 12) : 
  let M := matrix.of ![[3, 4], [8, -2]] in
  det M = -38 → measure_theory.measure (MeasureSpace.volume) (image (λ x : ℝ × ℝ, (M 0 0 * x.1 + M 0 1 * x.2, M 1 0 * x.1 + M 1 1 * x.2)) T) = 456 :=
by
  sorry

end area_transformation_l774_774280


namespace log_exp_relationship_l774_774180

theorem log_exp_relationship:
  let a := Real.log 0.9 / Real.log 0.8 
  let b := Real.log 0.9 / Real.log 1.1 
  let c := 1.1 ^ 0.9
  in b < a ∧ a < c :=
by
  let a := Real.log 0.9 / Real.log 0.8
  let b := Real.log 0.9 / Real.log 1.1
  let c := 1.1 ^ 0.9
  have h1 : 0 < a := by sorry
  have h2 : a < 1 := by sorry
  have h3 : b < 0 := by sorry
  have h4 : 1 < c := by sorry
  have h5 : b < a := by sorry
  have h6 : a < c := by sorry
  exact ⟨h5, h6⟩

end log_exp_relationship_l774_774180
