import Mathlib

namespace commute_times_abs_difference_l2020_202090

theorem commute_times_abs_difference (x y : ℝ)
  (h_avg : (x + y + 10 + 11 + 9) / 5 = 10)
  (h_var : ((x - 10)^2 + (y - 10)^2 + 0^2 + 1^2 + (-1)^2) / 5 = 2) 
  : |x - y| = 4 :=
sorry

end commute_times_abs_difference_l2020_202090


namespace compute_expression_l2020_202032

theorem compute_expression : (3 + 7)^2 + (3^2 + 7^2 + 5^2) = 183 := by
  sorry

end compute_expression_l2020_202032


namespace avg_move_to_california_l2020_202031

noncomputable def avg_people_per_hour (total_people : ℕ) (total_days : ℕ) : ℕ :=
  let total_hours := total_days * 24
  let avg_per_hour := total_people / total_hours
  let remainder := total_people % total_hours
  if remainder * 2 < total_hours then avg_per_hour else avg_per_hour + 1

theorem avg_move_to_california : avg_people_per_hour 3500 5 = 29 := by
  sorry

end avg_move_to_california_l2020_202031


namespace simplify_expression_l2020_202066

variable (x : ℝ)

def expr := (5*x^10 + 8*x^8 + 3*x^6) + (2*x^12 + 3*x^10 + x^8 + 4*x^6 + 2*x^2 + 7)

theorem simplify_expression : expr x = 2*x^12 + 8*x^10 + 9*x^8 + 7*x^6 + 2*x^2 + 7 :=
by
  sorry

end simplify_expression_l2020_202066


namespace ava_planted_more_trees_l2020_202042

theorem ava_planted_more_trees (L : ℕ) (h1 : 9 + L = 15) : 9 - L = 3 := 
by
  sorry

end ava_planted_more_trees_l2020_202042


namespace fifth_friend_payment_l2020_202059

def contributions (a b c d e : ℕ) : Prop :=
  a + b + c + d + e = 120 ∧
  a = (1 / 3 : ℕ) * (b + c + d + e) ∧
  b = (1 / 4 : ℕ) * (a + c + d + e) ∧
  c = (1 / 5 : ℕ) * (a + b + d + e)

theorem fifth_friend_payment (a b c d e : ℕ) (h : contributions a b c d e) : e = 13 :=
sorry

end fifth_friend_payment_l2020_202059


namespace distance_travelled_by_gavril_l2020_202040

noncomputable def smartphoneFullyDischargesInVideoWatching : ℝ := 3
noncomputable def smartphoneFullyDischargesInPlayingTetris : ℝ := 5
noncomputable def speedForHalfDistanceFirst : ℝ := 80
noncomputable def speedForHalfDistanceSecond : ℝ := 60
noncomputable def averageSpeed (distance speed time : ℝ) :=
  distance / time = speed

theorem distance_travelled_by_gavril : 
  ∃ S : ℝ, 
    (∃ t : ℝ, 
      (t / 2 / smartphoneFullyDischargesInVideoWatching + t / 2 / smartphoneFullyDischargesInPlayingTetris = 1) ∧ 
      (S / 2 / t / 2 = speedForHalfDistanceFirst) ∧
      (S / 2 / t / 2 = speedForHalfDistanceSecond)) ∧
     S = 257 := 
sorry

end distance_travelled_by_gavril_l2020_202040


namespace solution1_solution2_l2020_202048

open Real

noncomputable def problem1 (a b : ℝ) : Prop :=
a = 2 ∧ b = 2

noncomputable def problem2 (b : ℝ) : Prop :=
b = (2 * (sqrt 3 + sqrt 2)) / 3

theorem solution1 (a b : ℝ) (c : ℝ) (C : ℝ) (area : ℝ)
  (h1 : c = 2)
  (h2 : C = π / 3)
  (h3 : area = sqrt 3)
  (h4 : (1 / 2) * a * b * sin C = area) :
  problem1 a b :=
by sorry

theorem solution2 (a b : ℝ) (c : ℝ) (C : ℝ) (cosA : ℝ)
  (h1 : c = 2)
  (h2 : C = π / 3)
  (h3 : cosA = sqrt 3 / 3)
  (h4 : sin (arccos (sqrt 3 / 3)) = sqrt 6 / 3)
  (h5 : (a / (sqrt 6 / 3)) = (2 / (sqrt 3 / 2)))
  (h6 : ((b / ((3 + sqrt 6) / 6)) = (2 / (sqrt 3 / 2)))) :
  problem2 b :=
by sorry

end solution1_solution2_l2020_202048


namespace square_side_measurement_error_l2020_202013

theorem square_side_measurement_error {S S' : ℝ} (h1 : S' = S * Real.sqrt 1.0816) :
  ((S' - S) / S) * 100 = 4 := by
  sorry

end square_side_measurement_error_l2020_202013


namespace hyperbola_t_square_l2020_202049

theorem hyperbola_t_square (t : ℝ)
  (h1 : ∃ a : ℝ, ∀ (x y : ℝ), (y^2 / 4) - (5 * x^2 / 64) = 1 ↔ ((x, y) = (2, t) ∨ (x, y) = (4, -3) ∨ (x, y) = (0, -2))) :
  t^2 = 21 / 4 :=
by
  -- We need to prove t² = 21/4 given the conditions
  sorry

end hyperbola_t_square_l2020_202049


namespace mirror_side_length_l2020_202037

theorem mirror_side_length
  (width_wall : ℝ)
  (length_wall : ℝ)
  (area_wall : ℝ)
  (area_mirror : ℝ)
  (side_length_mirror : ℝ)
  (h1 : width_wall = 32)
  (h2 : length_wall = 20.25)
  (h3 : area_wall = width_wall * length_wall)
  (h4 : area_mirror = area_wall / 2)
  (h5 : side_length_mirror * side_length_mirror = area_mirror)
  : side_length_mirror = 18 := by
  sorry

end mirror_side_length_l2020_202037


namespace jaylen_bell_peppers_ratio_l2020_202022

theorem jaylen_bell_peppers_ratio :
  ∃ j_bell_p, ∃ k_bell_p, ∃ j_green_b, ∃ k_green_b, ∃ j_carrots, ∃ j_cucumbers, ∃ j_total_veg,
  j_carrots = 5 ∧
  j_cucumbers = 2 ∧
  k_bell_p = 2 ∧
  k_green_b = 20 ∧
  j_green_b = 20 / 2 - 3 ∧
  j_total_veg = 18 ∧
  j_carrots + j_cucumbers + j_green_b + j_bell_p = j_total_veg ∧
  j_bell_p / k_bell_p = 2 :=
sorry

end jaylen_bell_peppers_ratio_l2020_202022


namespace prove_solutions_l2020_202075

noncomputable def solution1 (x : ℝ) : Prop :=
  3 * x^2 + 6 = abs (-25 + x)

theorem prove_solutions :
  solution1 ( (-1 + Real.sqrt 229) / 6 ) ∧ solution1 ( (-1 - Real.sqrt 229) / 6 ) :=
by
  sorry

end prove_solutions_l2020_202075


namespace circles_intersect_l2020_202000

theorem circles_intersect (m c : ℝ) (h1 : (1:ℝ) = (5 + (-m))) (h2 : (3:ℝ) = (5 + (c - (-2)))) :
  m + c = 3 :=
sorry

end circles_intersect_l2020_202000


namespace eden_bears_count_l2020_202035

-- Define the main hypothesis
def initial_bears : Nat := 20
def favorite_bears : Nat := 8
def remaining_bears := initial_bears - favorite_bears

def number_of_sisters : Nat := 3
def bears_per_sister := remaining_bears / number_of_sisters

def eden_initial_bears : Nat := 10
def eden_final_bears := eden_initial_bears + bears_per_sister

theorem eden_bears_count : eden_final_bears = 14 :=
by
  unfold eden_final_bears eden_initial_bears bears_per_sister remaining_bears initial_bears favorite_bears
  norm_num
  sorry

end eden_bears_count_l2020_202035


namespace find_AC_l2020_202001

theorem find_AC (A B C : ℝ) (r1 r2 : ℝ) (AB : ℝ) (AC : ℝ) 
  (h_rad1 : r1 = 1) (h_rad2 : r2 = 3) (h_AB : AB = 2 * Real.sqrt 5) 
  (h_AC : AC = AB / 4) :
  AC = Real.sqrt 5 / 2 :=
by
  sorry

end find_AC_l2020_202001


namespace abc_inequality_l2020_202088

theorem abc_inequality (a b c : ℝ) : a^2 + b^2 + c^2 ≥ ab + ac + bc :=
by
  sorry

end abc_inequality_l2020_202088


namespace find_m_value_l2020_202044

noncomputable def hyperbola_m_value (m : ℝ) : Prop :=
  let a := 1
  let b := 2 * a
  m = -(1/4)

theorem find_m_value :
  (∀ x y : ℝ, x^2 + m * y^2 = 1 → b = 2 * a) → hyperbola_m_value m :=
by
  intro h
  sorry

end find_m_value_l2020_202044


namespace radius_of_large_circle_l2020_202073

/-- Five circles are described with the given properties. -/
def small_circle_radius : ℝ := 2

/-- The angle between any centers of the small circles is 72 degrees due to equal spacing. -/
def angle_between_centers : ℝ := 72

/-- The final theorem states that the radius of the larger circle is as follows. -/
theorem radius_of_large_circle (number_of_circles : ℕ)
        (radius_small : ℝ)
        (angle : ℝ)
        (internally_tangent : ∀ (i : ℕ), i < number_of_circles → Prop)
        (externally_tangent : ∀ (i j : ℕ), i ≠ j → i < number_of_circles → j < number_of_circles → Prop) :
  number_of_circles = 5 →
  radius_small = small_circle_radius →
  angle = angle_between_centers →
  (∃ R : ℝ, R = 4 * Real.sqrt 5 - 2) 
:= by
  -- mathematical proof goes here
  sorry

end radius_of_large_circle_l2020_202073


namespace pool_buckets_l2020_202067

theorem pool_buckets (buckets_george_per_round buckets_harry_per_round rounds : ℕ) 
  (h_george : buckets_george_per_round = 2) 
  (h_harry : buckets_harry_per_round = 3) 
  (h_rounds : rounds = 22) : 
  buckets_george_per_round + buckets_harry_per_round * rounds = 110 := 
by 
  sorry

end pool_buckets_l2020_202067


namespace max_books_per_student_l2020_202086

theorem max_books_per_student
  (total_students : ℕ)
  (students_0_books : ℕ)
  (students_1_book : ℕ)
  (students_2_books : ℕ)
  (students_at_least_3_books : ℕ)
  (avg_books_per_student : ℕ)
  (max_books_limit : ℕ)
  (total_books_available : ℕ) :
  total_students = 20 →
  students_0_books = 2 →
  students_1_book = 10 →
  students_2_books = 5 →
  students_at_least_3_books = total_students - students_0_books - students_1_book - students_2_books →
  avg_books_per_student = 2 →
  max_books_limit = 5 →
  total_books_available = 60 →
  avg_books_per_student * total_students = 40 →
  total_books_available = 60 →
  max_books_limit = 5 :=
by sorry

end max_books_per_student_l2020_202086


namespace total_gold_coins_l2020_202069

theorem total_gold_coins (n c : ℕ) 
  (h1 : n = 11 * (c - 3))
  (h2 : n = 7 * c + 5) : 
  n = 75 := 
by 
  sorry

end total_gold_coins_l2020_202069


namespace tetrahedron_planes_count_l2020_202025

def tetrahedron_planes : ℕ :=
  let vertices := 4
  let midpoints := 6
  -- The total number of planes calculated by considering different combinations
  4      -- planes formed by three vertices
  + 6    -- planes formed by two vertices and one midpoint
  + 12   -- planes formed by one vertex and two midpoints
  + 7    -- planes formed by three midpoints

theorem tetrahedron_planes_count :
  tetrahedron_planes = 29 :=
by
  sorry

end tetrahedron_planes_count_l2020_202025


namespace calculate_lassis_from_nine_mangoes_l2020_202082

variable (mangoes_lassis_ratio : ℕ → ℕ → Prop)
variable (cost_per_mango : ℕ)

def num_lassis (mangoes : ℕ) : ℕ :=
  5 * mangoes
  
theorem calculate_lassis_from_nine_mangoes
  (h1 : mangoes_lassis_ratio 15 3)
  (h2 : cost_per_mango = 2) :
  num_lassis 9 = 45 :=
by
  sorry

end calculate_lassis_from_nine_mangoes_l2020_202082


namespace shaded_area_l2020_202072

-- Let A be the length of the side of the smaller square
def A : ℝ := 4

-- Let B be the length of the side of the larger square
def B : ℝ := 12

-- The problem is to prove that the area of the shaded region is 10 square inches
theorem shaded_area (A B : ℝ) (hA : A = 4) (hB : B = 12) :
  (A * A) - (1/2 * (B / (B + A)) * A * B) = 10 := by
  sorry

end shaded_area_l2020_202072


namespace vikki_hourly_pay_rate_l2020_202056

-- Define the variables and conditions
def hours_worked : ℝ := 42
def tax_rate : ℝ := 0.20
def insurance_rate : ℝ := 0.05
def union_dues : ℝ := 5
def net_pay : ℝ := 310

-- Define Vikki's hourly pay rate (we will solve for this)
variable (hourly_pay : ℝ)

-- Define the gross earnings
def gross_earnings (hourly_pay : ℝ) : ℝ := hours_worked * hourly_pay

-- Define the total deductions
def total_deductions (hourly_pay : ℝ) : ℝ := (tax_rate * gross_earnings hourly_pay) + (insurance_rate * gross_earnings hourly_pay) + union_dues

-- Define the net pay
def calculate_net_pay (hourly_pay : ℝ) : ℝ := gross_earnings hourly_pay - total_deductions hourly_pay

-- Prove the solution
theorem vikki_hourly_pay_rate : calculate_net_pay hourly_pay = net_pay → hourly_pay = 10 := by
  sorry

end vikki_hourly_pay_rate_l2020_202056


namespace car_speed_624km_in_2_2_5_hours_l2020_202080

theorem car_speed_624km_in_2_2_5_hours : 
  ∀ (distance time_in_hours : ℝ), distance = 624 → time_in_hours = 2 + (2/5) → distance / time_in_hours = 260 :=
by
  intros distance time_in_hours h_dist h_time
  sorry

end car_speed_624km_in_2_2_5_hours_l2020_202080


namespace find_2nd_month_sales_l2020_202096

def sales_of_1st_month : ℝ := 2500
def sales_of_3rd_month : ℝ := 9855
def sales_of_4th_month : ℝ := 7230
def sales_of_5th_month : ℝ := 7000
def sales_of_6th_month : ℝ := 11915
def average_sales : ℝ := 7500
def months : ℕ := 6
def total_required_sales : ℝ := average_sales * months
def total_known_sales : ℝ := sales_of_1st_month + sales_of_3rd_month + sales_of_4th_month + sales_of_5th_month + sales_of_6th_month

theorem find_2nd_month_sales : 
  ∃ (sales_of_2nd_month : ℝ), total_required_sales = sales_of_1st_month + sales_of_2nd_month + sales_of_3rd_month + sales_of_4th_month + sales_of_5th_month + sales_of_6th_month ∧ sales_of_2nd_month = 10500 := by
  sorry

end find_2nd_month_sales_l2020_202096


namespace b_share_l2020_202092

theorem b_share (a b c : ℕ) (h1 : a + b + c = 120) (h2 : a = b + 20) (h3 : a = c - 20) : b = 20 :=
by
  sorry

end b_share_l2020_202092


namespace find_h_for_expression_l2020_202065

theorem find_h_for_expression (a k : ℝ) (h : ℝ) :
  (∃ a k : ℝ, ∀ x : ℝ, x^2 - 6*x + 1 = a*(x - h)^3 + k) ↔ h = 2 :=
by
  sorry

end find_h_for_expression_l2020_202065


namespace B_days_to_complete_work_l2020_202023

theorem B_days_to_complete_work (A_days : ℕ) (efficiency_less_percent : ℕ) 
  (hA : A_days = 12) (hB_efficiency : efficiency_less_percent = 20) :
  let A_work_rate := 1 / 12
  let B_work_rate := (1 - (20 / 100)) * A_work_rate
  let B_days := 1 / B_work_rate
  B_days = 15 :=
by
  sorry

end B_days_to_complete_work_l2020_202023


namespace probability_of_scoring_l2020_202030

theorem probability_of_scoring :
  ∀ (p : ℝ), (p + (1 / 3) * p = 1) → (p = 3 / 4) → (p * (1 - p) = 3 / 16) :=
by
  intros p h1 h2
  sorry

end probability_of_scoring_l2020_202030


namespace problem_solution_l2020_202061

theorem problem_solution (x : ℝ) (h : (18 / 100) * 42 = (27 / 100) * x) : x = 28 :=
sorry

end problem_solution_l2020_202061


namespace prime_ge_7_not_divisible_by_40_l2020_202018

theorem prime_ge_7_not_divisible_by_40 (p : ℕ) (hp_prime : Nat.Prime p) (hp_ge_7 : p ≥ 7) : ¬ (40 ∣ (p^3 - 1)) :=
sorry

end prime_ge_7_not_divisible_by_40_l2020_202018


namespace count_ordered_triples_l2020_202033

theorem count_ordered_triples (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : a^2 = b^2 + c^2) (h5 : b^2 = a^2 + c^2) (h6 : c^2 = a^2 + b^2) : 
  (a = b ∧ b = c ∧ a ≠ 0) ∨ (a = -b ∧ b = c ∧ a ≠ 0) ∨ (a = b ∧ b = -c ∧ a ≠ 0) ∨ (a = -b ∧ b = -c ∧ a ≠ 0) :=
sorry

end count_ordered_triples_l2020_202033


namespace color_nat_two_colors_no_sum_power_of_two_l2020_202011

theorem color_nat_two_colors_no_sum_power_of_two :
  ∃ (f : ℕ → ℕ), (∀ a b : ℕ, a ≠ b → f a = f b → ∃ c : ℕ, c > 0 ∧ c ≠ 1 ∧ c ≠ 2 ∧ (a + b ≠ 2 ^ c)) :=
sorry

end color_nat_two_colors_no_sum_power_of_two_l2020_202011


namespace cos_135_eq_neg_sqrt2_div_2_sin_135_eq_sqrt2_div_2_l2020_202054

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by sorry

theorem sin_135_eq_sqrt2_div_2 : Real.sin (135 * Real.pi / 180) = Real.sqrt 2 / 2 :=
by sorry

end cos_135_eq_neg_sqrt2_div_2_sin_135_eq_sqrt2_div_2_l2020_202054


namespace number_of_men_in_first_group_l2020_202081

-- Define the conditions
def condition1 (M : ℕ) : Prop := M * 80 = 20 * 40

-- State the main theorem to be proved
theorem number_of_men_in_first_group (M : ℕ) (h : condition1 M) : M = 10 := by
  sorry

end number_of_men_in_first_group_l2020_202081


namespace tank_capacity_l2020_202091

theorem tank_capacity (T : ℚ) (h1 : 0 ≤ T)
  (h2 : 9 + (3 / 4) * T = (9 / 10) * T) : T = 60 :=
sorry

end tank_capacity_l2020_202091


namespace final_value_of_A_l2020_202076

theorem final_value_of_A (A : ℤ) (h₁ : A = 15) (h₂ : A = -A + 5) : A = -10 := 
by 
  sorry

end final_value_of_A_l2020_202076


namespace arith_seq_sum_proof_l2020_202051

open Function

variable (a : ℕ → ℕ) -- Define the arithmetic sequence
variables (S : ℕ → ℕ) -- Define the sum function of the sequence

-- Conditions: S_8 = 9 and S_5 = 6
axiom S8 : S 8 = 9
axiom S5 : S 5 = 6

-- Mathematical equivalence
theorem arith_seq_sum_proof : S 13 = 13 :=
sorry

end arith_seq_sum_proof_l2020_202051


namespace sum_inequality_l2020_202024

open Real

theorem sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a + b + c ≤ (a * b / (a + b)) + (b * c / (b + c)) + (c * a / (c + a)) + 
             (1 / 2) * ((a * b / c) + (b * c / a) + (c * a / b)) :=
by
  sorry

end sum_inequality_l2020_202024


namespace range_of_a_l2020_202079

theorem range_of_a (x a : ℝ) (p : Prop) (q : Prop) (H₁ : p ↔ (x < -3 ∨ x > 1))
  (H₂ : q ↔ (x > a))
  (H₃ : ¬p → ¬q) (H₄ : ¬q → ¬p → false) : a ≥ 1 :=
sorry

end range_of_a_l2020_202079


namespace cricketer_average_score_l2020_202055

variable {A : ℤ} -- A represents the average score after 18 innings

theorem cricketer_average_score
  (h1 : (19 * (A + 4) = 18 * A + 98)) :
  A + 4 = 26 := by
  sorry

end cricketer_average_score_l2020_202055


namespace binom_coefficient_largest_l2020_202039

theorem binom_coefficient_largest (n : ℕ) (h : (n / 2) + 1 = 7) : n = 12 :=
by
  sorry

end binom_coefficient_largest_l2020_202039


namespace range_of_a_l2020_202045

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = Real.exp x) :
  (∀ x : ℝ, f x ≥ Real.exp x + a) ↔ a ≤ 0 :=
by
  sorry

end range_of_a_l2020_202045


namespace student_A_more_stable_l2020_202097

-- Given conditions
def average_score (n : ℕ) (score : ℕ) := score = 110
def variance_A := 3.6
def variance_B := 4.4

-- Prove that student A has more stable scores than student B
theorem student_A_more_stable : variance_A < variance_B :=
by
  -- Skipping the actual proof
  sorry

end student_A_more_stable_l2020_202097


namespace largest_integer_less_than_100_with_remainder_5_l2020_202029

theorem largest_integer_less_than_100_with_remainder_5 :
  ∃ n, (n < 100 ∧ n % 8 = 5) ∧ ∀ m, (m < 100 ∧ m % 8 = 5) → m ≤ n :=
sorry

end largest_integer_less_than_100_with_remainder_5_l2020_202029


namespace fraction_of_orange_juice_correct_l2020_202027

-- Define the capacities of the pitchers
def capacity := 800

-- Define the fractions of orange juice and apple juice in the first pitcher
def orangeJuiceFraction1 := 1 / 4
def appleJuiceFraction1 := 1 / 8

-- Define the fractions of orange juice and apple juice in the second pitcher
def orangeJuiceFraction2 := 1 / 5
def appleJuiceFraction2 := 1 / 10

-- Define the total volumes of the contents in each pitcher
def totalVolume := 2 * capacity -- total volume in the large container after pouring

-- Define the orange juice volumes in each pitcher
def orangeJuiceVolume1 := orangeJuiceFraction1 * capacity
def orangeJuiceVolume2 := orangeJuiceFraction2 * capacity

-- Calculate the total volume of orange juice in the large container
def totalOrangeJuiceVolume := orangeJuiceVolume1 + orangeJuiceVolume2

-- Define the fraction of orange juice in the large container
def orangeJuiceFraction := totalOrangeJuiceVolume / totalVolume

theorem fraction_of_orange_juice_correct :
  orangeJuiceFraction = 9 / 40 :=
by
  sorry

end fraction_of_orange_juice_correct_l2020_202027


namespace four_digit_number_8802_l2020_202041

theorem four_digit_number_8802 (x : ℕ) (a b c d : ℕ) (h1 : 1000 ≤ x ∧ x ≤ 9999)
  (h2 : x = 1000 * a + 100 * b + 10 * c + d)
  (h3 : a ≠ 0)  -- since a 4-digit number cannot start with 0
  (h4 : a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10) : 
  x + 8802 = 1099 + 8802 :=
by
  sorry

end four_digit_number_8802_l2020_202041


namespace weight_of_brand_b_l2020_202064

theorem weight_of_brand_b (w_a w_b : ℕ) (vol_a vol_b : ℕ) (total_volume total_weight : ℕ) 
  (h1 : w_a = 950) 
  (h2 : vol_a = 3) 
  (h3 : vol_b = 2) 
  (h4 : total_volume = 4) 
  (h5 : total_weight = 3640) 
  (h6 : vol_a + vol_b = total_volume) 
  (h7 : vol_a * w_a + vol_b * w_b = total_weight) : 
  w_b = 395 := 
by {
  sorry
}

end weight_of_brand_b_l2020_202064


namespace total_fencing_cost_l2020_202087

def side1 : ℕ := 34
def side2 : ℕ := 28
def side3 : ℕ := 45
def side4 : ℕ := 50
def side5 : ℕ := 55

def cost1_per_meter : ℕ := 2
def cost2_per_meter : ℕ := 2
def cost3_per_meter : ℕ := 3
def cost4_per_meter : ℕ := 3
def cost5_per_meter : ℕ := 4

def total_cost : ℕ :=
  side1 * cost1_per_meter +
  side2 * cost2_per_meter +
  side3 * cost3_per_meter +
  side4 * cost4_per_meter +
  side5 * cost5_per_meter

theorem total_fencing_cost : total_cost = 629 := by
  sorry

end total_fencing_cost_l2020_202087


namespace emily_dog_count_l2020_202015

theorem emily_dog_count (dogs : ℕ) 
  (food_per_day_per_dog : ℕ := 250) 
  (vacation_days : ℕ := 14)
  (total_food_kg : ℕ := 14)
  (kg_to_grams : ℕ := 1000) 
  (total_food_grams : ℕ := total_food_kg * kg_to_grams)
  (food_needed_per_dog : ℕ := food_per_day_per_dog * vacation_days) 
  (total_food_needed : ℕ := dogs * food_needed_per_dog) 
  (h : total_food_needed = total_food_grams) : 
  dogs = 4 := 
sorry

end emily_dog_count_l2020_202015


namespace stickers_given_to_sister_l2020_202043

variable (initial bought birthday used left given : ℕ)

theorem stickers_given_to_sister :
  (initial = 20) →
  (bought = 12) →
  (birthday = 20) →
  (used = 8) →
  (left = 39) →
  (given = (initial + bought + birthday - used - left)) →
  given = 5 := by
  intros
  sorry

end stickers_given_to_sister_l2020_202043


namespace find_third_side_of_triangle_l2020_202084

noncomputable def area_triangle_given_sides_angle {a b c : ℝ} (A : ℝ) : Prop :=
  A = 1/2 * a * b * Real.sin c

noncomputable def cosine_law_third_side {a b c : ℝ} (cosα : ℝ) : Prop :=
  c^2 = a^2 + b^2 - 2 * a * b * cosα

theorem find_third_side_of_triangle (a b : ℝ) (Area : ℝ) (h_a : a = 2 * Real.sqrt 2) (h_b : b = 3) (h_Area : Area = 3) :
  ∃ c : ℝ, (c = Real.sqrt 5 ∨ c = Real.sqrt 29) :=
by
  sorry

end find_third_side_of_triangle_l2020_202084


namespace thomas_probability_of_two_pairs_l2020_202008

def number_of_ways_to_choose_five_socks := Nat.choose 12 5
def number_of_ways_to_choose_two_pairs_of_colors := Nat.choose 4 2
def number_of_ways_to_choose_one_color_for_single_sock := Nat.choose 2 1
def number_of_ways_to_choose_two_socks_from_three := Nat.choose 3 2
def number_of_ways_to_choose_one_sock_from_three := Nat.choose 3 1

theorem thomas_probability_of_two_pairs : 
  number_of_ways_to_choose_five_socks = 792 →
  number_of_ways_to_choose_two_pairs_of_colors = 6 →
  number_of_ways_to_choose_one_color_for_single_sock = 2 →
  number_of_ways_to_choose_two_socks_from_three = 3 →
  number_of_ways_to_choose_one_sock_from_three = 3 →
  6 * 2 * 3 * 3 * 3 = 324 →
  (324 : ℚ) / 792 = 9 / 22 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end thomas_probability_of_two_pairs_l2020_202008


namespace rectangle_perimeter_l2020_202083

theorem rectangle_perimeter (l d : ℝ) (h_l : l = 8) (h_d : d = 17) :
  ∃ w : ℝ, (d^2 = l^2 + w^2) ∧ (2*l + 2*w = 46) :=
by
  sorry

end rectangle_perimeter_l2020_202083


namespace expected_value_eight_sided_die_win_l2020_202063

/-- The expected value of winning with a fair 8-sided die, where the win is \( n^3 \) dollars if \( n \) is rolled, is 162 dollars. -/
theorem expected_value_eight_sided_die_win :
  (1 / 8) * (1^3) + (1 / 8) * (2^3) + (1 / 8) * (3^3) + (1 / 8) * (4^3) +
  (1 / 8) * (5^3) + (1 / 8) * (6^3) + (1 / 8) * (7^3) + (1 / 8) * (8^3) = 162 := 
by
  -- Simplification and calculation here
  sorry

end expected_value_eight_sided_die_win_l2020_202063


namespace history_paper_pages_l2020_202050

theorem history_paper_pages (p d : ℕ) (h1 : p = 11) (h2 : d = 3) : p * d = 33 :=
by
  sorry

end history_paper_pages_l2020_202050


namespace top_z_teams_l2020_202078

theorem top_z_teams (n : ℕ) (h : (n * (n - 1)) / 2 = 45) : n = 10 := 
sorry

end top_z_teams_l2020_202078


namespace new_average_daily_production_l2020_202014

theorem new_average_daily_production 
  (n : ℕ) 
  (avg_past_n_days : ℕ) 
  (today_production : ℕ)
  (new_avg_production : ℕ)
  (hn : n = 5) 
  (havg : avg_past_n_days = 60) 
  (htoday : today_production = 90) 
  (hnew_avg : new_avg_production = 65)
  : (n + 1 = 6) ∧ ((n * 60 + today_production) = 390) ∧ (390 / 6 = 65) :=
by
  sorry

end new_average_daily_production_l2020_202014


namespace probability_of_non_defective_pens_l2020_202095

-- Define the number of total pens, defective pens, and pens to be selected
def total_pens : ℕ := 15
def defective_pens : ℕ := 5
def selected_pens : ℕ := 3

-- Define the number of non-defective pens
def non_defective_pens : ℕ := total_pens - defective_pens

-- Define the combination function
def combination (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Define the total ways to choose 3 pens from 15 pens
def total_ways : ℕ := combination total_pens selected_pens

-- Define the ways to choose 3 non-defective pens from the non-defective pens
def non_defective_ways : ℕ := combination non_defective_pens selected_pens

-- Define the probability
def probability : ℚ := non_defective_ways / total_ways

-- Statement we need to prove
theorem probability_of_non_defective_pens : probability = 120 / 455 := by
  -- Proof to be completed
  sorry

end probability_of_non_defective_pens_l2020_202095


namespace find_y_l2020_202005

theorem find_y (x y : ℤ) (h₁ : x ^ 2 + x + 4 = y - 4) (h₂ : x = 3) : y = 20 :=
by 
  sorry

end find_y_l2020_202005


namespace sequence_subsequence_l2020_202068

theorem sequence_subsequence :
  ∃ (a : Fin 101 → ℕ), 
  (∀ i, a i = i + 1) ∧ 
  ∃ (b : Fin 11 → ℕ), 
  (b 0 < b 1 ∧ b 1 < b 2 ∧ b 2 < b 3 ∧ b 3 < b 4 ∧ b 4 < b 5 ∧ 
  b 5 < b 6 ∧ b 6 < b 7 ∧ b 7 < b 8 ∧ b 8 < b 9 ∧ b 9 < b 10) ∨ 
  (b 0 > b 1 ∧ b 1 > b 2 ∧ b 2 > b 3 ∧ b 3 > b 4 ∧ b 4 > b 5 ∧ 
  b 5 > b 6 ∧ b 6 > b 7 ∧ b 7 > b 8 ∧ b 8 > b 9 ∧ b 9 > b 10) :=
by {
  sorry
}

end sequence_subsequence_l2020_202068


namespace same_terminal_side_eq_l2020_202053

theorem same_terminal_side_eq (α : ℝ) : 
    (∃ k : ℤ, α = 2 * k * Real.pi - Real.pi / 3) ↔ α = 5 * Real.pi / 3 :=
by sorry

end same_terminal_side_eq_l2020_202053


namespace digit_multiplication_sum_l2020_202077

-- Define the main problem statement in Lean 4
theorem digit_multiplication_sum (A B E F : ℕ) (h1 : 0 ≤ A ∧ A ≤ 9) 
                                            (h2 : 0 ≤ B ∧ B ≤ 9) 
                                            (h3 : 0 ≤ E ∧ E ≤ 9)
                                            (h4 : 0 ≤ F ∧ F ≤ 9)
                                            (h5 : A ≠ B) 
                                            (h6 : A ≠ E) 
                                            (h7 : A ≠ F)
                                            (h8 : B ≠ E)
                                            (h9 : B ≠ F)
                                            (h10 : E ≠ F)
                                            (h11 : (100 * A + 10 * B + E) * F = 1001 * E + 100 * A)
                                            : A + B = 5 :=
sorry

end digit_multiplication_sum_l2020_202077


namespace star_operation_l2020_202012

def new_op (a b : ℝ) : ℝ :=
  a^2 + b^2 - a * b

theorem star_operation (x y : ℝ) : 
  new_op (x + 2 * y) (y + 3 * x) = 7 * x^2 + 3 * y^2 + 3 * (x * y) :=
by
  sorry

end star_operation_l2020_202012


namespace only_A_can_form_triangle_l2020_202046

/--
Prove that from the given sets of lengths, only the set {5cm, 8cm, 12cm} can form a valid triangle.

Given:
- A: 5 cm, 8 cm, 12 cm
- B: 2 cm, 3 cm, 6 cm
- C: 3 cm, 3 cm, 6 cm
- D: 4 cm, 7 cm, 11 cm

We need to show that only Set A satisfies the triangle inequality theorem.
-/
theorem only_A_can_form_triangle :
  (∀ (a b c : ℕ), a = 5 ∧ b = 8 ∧ c = 12 → a + b > c ∧ a + c > b ∧ b + c > a) ∧
  (∀ (a b c : ℕ), a = 2 ∧ b = 3 ∧ c = 6 → ¬(a + b > c ∧ a + c > b ∧ b + c > a)) ∧
  (∀ (a b c : ℕ), a = 3 ∧ b = 3 ∧ c = 6 → ¬(a + b > c ∧ a + c > b ∧ b + c > a)) ∧
  (∀ (a b c : ℕ), a = 4 ∧ b = 7 ∧ c = 11 → ¬(a + b > c ∧ a + c > b ∧ b + c > a)) :=
by
  sorry -- Proof to be provided

end only_A_can_form_triangle_l2020_202046


namespace incorrect_description_is_A_l2020_202021

-- Definitions for the conditions
def description_A := "Increasing the concentration of reactants increases the percentage of activated molecules, accelerating the reaction rate."
def description_B := "Increasing the pressure of a gaseous reaction system increases the number of activated molecules per unit volume, accelerating the rate of the gas reaction."
def description_C := "Raising the temperature of the reaction increases the percentage of activated molecules, increases the probability of effective collisions, and increases the reaction rate."
def description_D := "Catalysts increase the reaction rate by changing the reaction path and lowering the activation energy required for the reaction."

-- Problem Statement
theorem incorrect_description_is_A :
  description_A ≠ correct :=
  sorry

end incorrect_description_is_A_l2020_202021


namespace debby_soda_bottles_l2020_202009

noncomputable def total_bottles (d t : ℕ) : ℕ := d * t

theorem debby_soda_bottles :
  ∀ (d t: ℕ), d = 9 → t = 40 → total_bottles d t = 360 :=
by
  intros d t h1 h2
  sorry

end debby_soda_bottles_l2020_202009


namespace cost_price_A_l2020_202016

-- Establishing the definitions based on the conditions from a)

def profit_A_to_B (CP_A : ℝ) : ℝ := 1.20 * CP_A
def profit_B_to_C (CP_B : ℝ) : ℝ := 1.25 * CP_B
def price_paid_by_C : ℝ := 222

-- Stating the theorem to be proven:
theorem cost_price_A (CP_A : ℝ) (H : profit_B_to_C (profit_A_to_B CP_A) = price_paid_by_C) : CP_A = 148 :=
by 
  sorry

end cost_price_A_l2020_202016


namespace triangle_inequality_shortest_side_l2020_202099

theorem triangle_inequality_shortest_side (a b c : ℝ) (h_triangle: a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a) 
  (h_inequality : a^2 + b^2 > 5 * c^2) : c ≤ a ∧ c ≤ b :=
sorry

end triangle_inequality_shortest_side_l2020_202099


namespace MountainRidgeAcademy_l2020_202026

theorem MountainRidgeAcademy (j s : ℕ) 
  (h1 : 3/4 * j = 1/2 * s) : s = 3/2 * j := 
by 
  sorry

end MountainRidgeAcademy_l2020_202026


namespace reflection_eq_l2020_202036

theorem reflection_eq (x y : ℝ) : 
    let line_eq (x y : ℝ) := 2 * x + 3 * y - 5 = 0 
    let reflection_eq (x y : ℝ) := 3 * x + 2 * y - 5 = 0 
    (∀ (x y : ℝ), line_eq x y ↔ reflection_eq y x) →
    reflection_eq x y :=
by
    sorry

end reflection_eq_l2020_202036


namespace chessboard_piece_arrangements_l2020_202052

-- Define the problem in Lean
theorem chessboard_piece_arrangements (black_pos white_pos : ℕ)
  (black_pos_neq_white_pos : black_pos ≠ white_pos)
  (valid_position : black_pos < 64 ∧ white_pos < 64) :
  ¬(∀ (move : ℕ → ℕ → Prop), (move black_pos white_pos) → ∃! (p : ℕ × ℕ), move (p.fst) (p.snd)) :=
by sorry

end chessboard_piece_arrangements_l2020_202052


namespace isosceles_triangle_median_length_l2020_202085

noncomputable def median_length (b h : ℝ) : ℝ :=
  let a := Real.sqrt ((b / 2) ^ 2 + h ^ 2)
  let m_a := Real.sqrt ((2 * a ^ 2 + 2 * b ^ 2 - a ^ 2) / 4)
  m_a

theorem isosceles_triangle_median_length :
  median_length 16 10 = Real.sqrt 146 :=
by
  sorry

end isosceles_triangle_median_length_l2020_202085


namespace area_of_trapezium_l2020_202070

variables (x : ℝ) (h : x > 0)

def shorter_base := 2 * x
def altitude := 2 * x
def longer_base := 6 * x

theorem area_of_trapezium (hx : x > 0) :
  (1 / 2) * (shorter_base x + longer_base x) * altitude x = 8 * x^2 := 
sorry

end area_of_trapezium_l2020_202070


namespace find_number_l2020_202010

theorem find_number:
  ∃ x : ℝ, x + 1.35 + 0.123 = 1.794 ∧ x = 0.321 :=
by
  sorry

end find_number_l2020_202010


namespace smallest_A_l2020_202057

theorem smallest_A (A B C D E : ℕ) 
  (hA_even : A % 2 = 0)
  (hB_even : B % 2 = 0)
  (hC_even : C % 2 = 0)
  (hD_even : D % 2 = 0)
  (hE_even : E % 2 = 0)
  (hA_three_digit : 100 ≤ A ∧ A < 1000)
  (hB_three_digit : 100 ≤ B ∧ B < 1000)
  (hC_three_digit : 100 ≤ C ∧ C < 1000)
  (hD_three_digit : 100 ≤ D ∧ D < 1000)
  (hE_three_digit : 100 ≤ E ∧ E < 1000)
  (h_sorted : A < B ∧ B < C ∧ C < D ∧ D < E)
  (h_sum : A + B + C + D + E = 4306) :
  A = 326 :=
sorry

end smallest_A_l2020_202057


namespace quadratic_square_binomial_l2020_202004

theorem quadratic_square_binomial (d : ℝ) : (∃ b : ℝ, (x : ℝ) -> (x + b)^2 = x^2 + 110 * x + d) ↔ d = 3025 :=
by
  sorry

end quadratic_square_binomial_l2020_202004


namespace ratio_of_volumes_l2020_202038

def cone_radius_X := 10
def cone_height_X := 15
def cone_radius_Y := 15
def cone_height_Y := 10

noncomputable def volume_cone (r h : ℝ) := (1 / 3) * Real.pi * r^2 * h

noncomputable def volume_X := volume_cone cone_radius_X cone_height_X
noncomputable def volume_Y := volume_cone cone_radius_Y cone_height_Y

theorem ratio_of_volumes : volume_X / volume_Y = 2 / 3 := sorry

end ratio_of_volumes_l2020_202038


namespace algae_coverage_double_l2020_202094

theorem algae_coverage_double (algae_cov : ℕ → ℝ) (h1 : ∀ n : ℕ, algae_cov (n + 2) = 2 * algae_cov n)
  (h2 : algae_cov 24 = 1) : algae_cov 18 = 0.125 :=
by
  sorry

end algae_coverage_double_l2020_202094


namespace tan_alpha_implies_fraction_l2020_202003

theorem tan_alpha_implies_fraction (α : ℝ) (h : Real.tan α = -3/2) : 
  (Real.sin α + 2 * Real.cos α) / (Real.cos α - Real.sin α) = 1 / 5 := 
sorry

end tan_alpha_implies_fraction_l2020_202003


namespace isosceles_triangle_perimeter_l2020_202019

theorem isosceles_triangle_perimeter :
  (∃ x y : ℝ, x^2 - 6*x + 8 = 0 ∧ y^2 - 6*y + 8 = 0 ∧ (x = 2 ∧ y = 4) ∧ 2 + 4 + 4 = 10) :=
by
  sorry

end isosceles_triangle_perimeter_l2020_202019


namespace part1_monotonicity_part2_minimum_range_l2020_202028

noncomputable def f (k x : ℝ) : ℝ := (k + x) / (x - 1) * Real.log x

theorem part1_monotonicity (x : ℝ) (h : x ≠ 1) :
    k = 0 → f k x = (x / (x - 1)) * Real.log x ∧ 
    (0 < x ∧ x < 1 ∨ 1 < x) → Monotone (f k) :=
sorry

theorem part2_minimum_range (k : ℝ) :
    (∃ x ∈ Set.Ioi 1, IsLocalMin (f k) x) ↔ k ∈ Set.Ioi 1 :=
sorry

end part1_monotonicity_part2_minimum_range_l2020_202028


namespace cos_double_angle_l2020_202093

theorem cos_double_angle (α : ℝ) (h : Real.sin (Real.pi + α) = 1 / 3) : Real.cos (2 * α) = 7 / 9 := 
by 
  sorry

end cos_double_angle_l2020_202093


namespace arithmetic_sequence_n_equals_8_l2020_202034

theorem arithmetic_sequence_n_equals_8
  (a : ℕ → ℝ) 
  (h_arith : ∀ n m, a (n + 1) - a n = a (m + 1) - a m) 
  (h2 : a 2 + a 5 = 18)
  (h3 : a 3 * a 4 = 32)
  (h_n : ∃ n, a n = 128) :
  ∃ n, a n = 128 ∧ n = 8 := 
sorry

end arithmetic_sequence_n_equals_8_l2020_202034


namespace minimum_sides_of_polygon_l2020_202058

theorem minimum_sides_of_polygon (θ : ℝ) (hθ : θ = 25.5) : ∃ n : ℕ, n = 240 ∧ ∀ k : ℕ, (k * θ) % 360 = 0 → k = n := 
by
  -- The proof goes here
  sorry

end minimum_sides_of_polygon_l2020_202058


namespace stephan_cannot_afford_laptop_l2020_202074

noncomputable def initial_laptop_price : ℝ := sorry

theorem stephan_cannot_afford_laptop (P₀ : ℝ) (h_rate : 0 < 0.06) (h₁ : initial_laptop_price = P₀) : 
  56358 < P₀ * (1.06)^2 :=
by 
  sorry

end stephan_cannot_afford_laptop_l2020_202074


namespace intersection_M_N_l2020_202089

def M (x : ℝ) : Prop := (x - 3) / (x + 1) > 0
def N (x : ℝ) : Prop := 3 * x + 2 > 0

theorem intersection_M_N :
  {x : ℝ | M x} ∩ {x : ℝ | N x} = {x : ℝ | 3 < x} :=
by
  sorry

end intersection_M_N_l2020_202089


namespace simplify_and_evaluate_l2020_202007

noncomputable def given_expression (a : ℝ) : ℝ :=
  (a-4) / a / ((a+2) / (a^2 - 2 * a) - (a-1) / (a^2 - 4 * a + 4))

theorem simplify_and_evaluate (a : ℝ) (h : a^2 - 4 * a + 3 = 0) : given_expression a = 1 := by
  sorry

end simplify_and_evaluate_l2020_202007


namespace calculate_expression_l2020_202071

open Complex

def B : Complex := 5 - 2 * I
def N : Complex := -3 + 2 * I
def T : Complex := 2 * I
def Q : ℂ := 3

theorem calculate_expression : B - N + T - 2 * Q = 2 - 2 * I := by
  sorry

end calculate_expression_l2020_202071


namespace sequence_geometric_l2020_202017

theorem sequence_geometric {a_n : ℕ → ℕ} (S : ℕ → ℕ) (a1 a2 a3 : ℕ) 
(hS : ∀ n, S n = 2 * a_n n - a_n 1) 
(h_arith : 2 * (a_n 2 + 1) = a_n 3 + a_n 1) : 
  ∀ n, a_n n = 2 ^ n :=
sorry

end sequence_geometric_l2020_202017


namespace speed_of_first_train_l2020_202020

noncomputable def speed_of_second_train : ℝ := 40 -- km/h
noncomputable def length_of_first_train : ℝ := 125 -- m
noncomputable def length_of_second_train : ℝ := 125.02 -- m
noncomputable def time_to_pass_each_other : ℝ := 1.5 / 60 -- hours (converted from minutes)

theorem speed_of_first_train (V1 V2 : ℝ) 
  (h1 : V2 = speed_of_second_train)
  (h2 : 125 + 125.02 = 250.02) 
  (h3 : 1.5 / 60 = 0.025) :
  V1 - V2 = 10.0008 → V1 = 50 :=
by 
  sorry

end speed_of_first_train_l2020_202020


namespace books_loaned_out_l2020_202006

/-- 
Given:
- There are 75 books in a special collection at the beginning of the month.
- By the end of the month, 70 percent of books that were loaned out are returned.
- There are 60 books in the special collection at the end of the month.
Prove:
- The number of books loaned out during the month is 50.
-/
theorem books_loaned_out (x : ℝ) (h1 : 75 - 0.3 * x = 60) : x = 50 :=
by
  sorry

end books_loaned_out_l2020_202006


namespace canned_food_total_bins_l2020_202047

theorem canned_food_total_bins :
  let soup_bins := 0.125
  let vegetable_bins := 0.125
  let pasta_bins := 0.5
  soup_bins + vegetable_bins + pasta_bins = 0.75 := 
by
  sorry

end canned_food_total_bins_l2020_202047


namespace calc_expression_l2020_202098

theorem calc_expression : 112 * 5^4 * 3^2 = 630000 := by
  sorry

end calc_expression_l2020_202098


namespace percentage_of_boys_from_schoolA_study_science_l2020_202060

variable (T : ℝ) -- Total number of boys in the camp
variable (schoolA_boys : ℝ)
variable (science_boys : ℝ)

noncomputable def percentage_science_boys := (science_boys / schoolA_boys) * 100

theorem percentage_of_boys_from_schoolA_study_science 
  (h1 : schoolA_boys = 0.20 * T)
  (h2 : science_boys = schoolA_boys - 56)
  (h3 : T = 400) :
  percentage_science_boys science_boys schoolA_boys = 30 := 
by sorry

end percentage_of_boys_from_schoolA_study_science_l2020_202060


namespace car_speeds_l2020_202002

noncomputable def distance_between_places : ℝ := 135
noncomputable def departure_time_diff : ℝ := 4 -- large car departs 4 hours before small car
noncomputable def arrival_time_diff : ℝ := 0.5 -- small car arrives 30 minutes earlier than large car
noncomputable def speed_ratio : ℝ := 5 / 2 -- ratio of speeds (small car : large car)

theorem car_speeds (v_small v_large : ℝ) (h1 : v_small / v_large = speed_ratio) :
    v_small = 45 ∧ v_large = 18 :=
sorry

end car_speeds_l2020_202002


namespace triangle_inequality_l2020_202062

theorem triangle_inequality (a b c : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) :
  a / (b + c - a) + b / (c + a - b) + c / (a + b - c) ≥ 3 :=
sorry

end triangle_inequality_l2020_202062
