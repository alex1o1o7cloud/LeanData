import Mathlib

namespace rearrangement_impossible_l1487_148757

-- Define the primary problem conditions and goal
theorem rearrangement_impossible :
  ¬ ∃ (f : Fin 100 → Fin 51), 
    (∀ k : Fin 51, ∃ i j : Fin 100, 
      f i = k ∧ f j = k ∧ (i < j ∧ j.val - i.val = k.val + 1)) :=
sorry

end rearrangement_impossible_l1487_148757


namespace other_student_in_sample_18_l1487_148711

theorem other_student_in_sample_18 (class_size sample_size : ℕ) (all_students : Finset ℕ) (sample_students : List ℕ)
  (h_class_size : class_size = 60)
  (h_sample_size : sample_size = 4)
  (h_all_students : all_students = Finset.range 60) -- students are numbered from 1 to 60
  (h_sample : sample_students = [3, 33, 48])
  (systematic_sampling : ℕ → ℕ → List ℕ) -- systematic_sampling function that generates the sample based on first element and k
  (k : ℕ) (h_k : k = class_size / sample_size) :
  systematic_sampling 3 k = [3, 18, 33, 48] := 
  sorry

end other_student_in_sample_18_l1487_148711


namespace solve_for_y_l1487_148706

theorem solve_for_y (y : ℝ) (h : (y * (y^5)^(1/4))^(1/3) = 4) : y = 2^(8/3) :=
by {
  sorry
}

end solve_for_y_l1487_148706


namespace root_of_quadratic_property_l1487_148760

theorem root_of_quadratic_property (m : ℝ) (h : m^2 - 2 * m - 1 = 0) :
  m^2 + (1 / m^2) = 6 :=
sorry

end root_of_quadratic_property_l1487_148760


namespace arithmetic_sequence_T_n_bound_l1487_148734

open Nat

theorem arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) (h2 : a 2 = 6) (h3_h6 : a 3 + a 6 = 27) :
  (∀ n, a n = 3 * n) := 
by
  sorry

theorem T_n_bound (a : ℕ → ℤ) (S : ℕ → ℤ) (T : ℕ → ℝ) (m : ℝ) (h_general_term : ∀ n, a n = 3 * n) 
  (h_S_n : ∀ n, S n = n^2 + n) (h_T_n : ∀ n, T n = (S n : ℝ) / (3 * (2 : ℝ)^(n-1)))
  (h_bound : ∀ n > 0, T n ≤ m) : 
  m ≥ 3/2 :=
by
  sorry

end arithmetic_sequence_T_n_bound_l1487_148734


namespace speed_of_sisters_sailboat_l1487_148787

variable (v_j : ℝ) (d : ℝ) (t_wait : ℝ)

-- Conditions
def janet_speed : Prop := v_j = 30
def lake_distance : Prop := d = 60
def janet_wait_time : Prop := t_wait = 3

-- Question to Prove
def sister_speed (v_s : ℝ) : Prop :=
  janet_speed v_j ∧ lake_distance d ∧ janet_wait_time t_wait →
  v_s = 12

-- The main theorem
theorem speed_of_sisters_sailboat (v_j d t_wait : ℝ) (h1 : janet_speed v_j) (h2 : lake_distance d) (h3 : janet_wait_time t_wait) :
  ∃ v_s : ℝ, sister_speed v_j d t_wait v_s :=
by
  sorry

end speed_of_sisters_sailboat_l1487_148787


namespace mailman_should_give_junk_mail_l1487_148701

-- Definitions from the conditions
def houses_in_block := 20
def junk_mail_per_house := 32

-- The mathematical equivalent proof problem statement in Lean 4
theorem mailman_should_give_junk_mail : 
  junk_mail_per_house * houses_in_block = 640 :=
  by sorry

end mailman_should_give_junk_mail_l1487_148701


namespace find_values_l1487_148789

theorem find_values (x y : ℤ) 
  (h1 : x / 5 + 7 = y / 4 - 7)
  (h2 : x / 3 - 4 = y / 2 + 4) : 
  x = -660 ∧ y = -472 :=
by 
  sorry

end find_values_l1487_148789


namespace quotient_is_six_l1487_148727

-- Definition of the given conditions
def S : Int := 476
def remainder : Int := 15
def difference : Int := 2395

-- Definition of the larger number based on the given conditions
def L : Int := S + difference

-- The statement we need to prove
theorem quotient_is_six : (L = S * 6 + remainder) := by
  sorry

end quotient_is_six_l1487_148727


namespace cos_squared_identity_l1487_148732

theorem cos_squared_identity (α : ℝ) (h : Real.tan (α + π / 4) = 3 / 4) :
    Real.cos (π / 4 - α) ^ 2 = 9 / 25 := by
  sorry

end cos_squared_identity_l1487_148732


namespace average_price_of_pencil_correct_l1487_148726

def average_price_of_pencil (n_pens n_pencils : ℕ) (total_cost pen_price : ℕ) : ℕ :=
  let pen_cost := n_pens * pen_price
  let pencil_cost := total_cost - pen_cost
  let avg_pencil_price := pencil_cost / n_pencils
  avg_pencil_price

theorem average_price_of_pencil_correct :
  average_price_of_pencil 30 75 450 10 = 2 :=
by
  -- The proof is omitted as per the instructions.
  sorry

end average_price_of_pencil_correct_l1487_148726


namespace repeated_two_digit_number_divisible_by_101_l1487_148786

theorem repeated_two_digit_number_divisible_by_101 (a b : ℕ) :
  (10 ≤ a ∧ a ≤ 99 ∧ 0 ≤ b ∧ b ≤ 9) →
  ∃ k, (100000 * a + 10000 * b + 1000 * a + 100 * b + 10 * a + b) = 101 * k :=
by
  intro h
  sorry

end repeated_two_digit_number_divisible_by_101_l1487_148786


namespace value_of_x_minus_y_l1487_148754

theorem value_of_x_minus_y (x y : ℝ) 
    (h1 : 3015 * x + 3020 * y = 3025) 
    (h2 : 3018 * x + 3024 * y = 3030) :
    x - y = 11.1167 :=
sorry

end value_of_x_minus_y_l1487_148754


namespace product_at_n_equals_three_l1487_148777

theorem product_at_n_equals_three : (3 - 2) * (3 - 1) * 3 * (3 + 1) * (3 + 2) = 120 := by
  sorry

end product_at_n_equals_three_l1487_148777


namespace sum_of_ages_l1487_148728

variable (A1 : ℝ) (A2 : ℝ) (A3 : ℝ) (A4 : ℝ) (A5 : ℝ) (A6 : ℝ) (A7 : ℝ)

noncomputable def age_first_scroll := 4080
noncomputable def age_difference := 2040

theorem sum_of_ages :
  let r := (age_difference:ℝ) / (age_first_scroll:ℝ)
  let A2 := (age_first_scroll:ℝ) + age_difference
  let A3 := A2 + (A2 - age_first_scroll) * r
  let A4 := A3 + (A3 - A2) * r
  let A5 := A4 + (A4 - A3) * r
  let A6 := A5 + (A5 - A4) * r
  let A7 := A6 + (A6 - A5) * r
  (age_first_scroll:ℝ) + A2 + A3 + A4 + A5 + A6 + A7 = 41023.75 := 
  by sorry

end sum_of_ages_l1487_148728


namespace roses_to_sister_l1487_148799

theorem roses_to_sister (total_roses roses_to_mother roses_to_grandmother roses_kept : ℕ) 
  (h1 : total_roses = 20)
  (h2 : roses_to_mother = 6)
  (h3 : roses_to_grandmother = 9)
  (h4 : roses_kept = 1) : 
  total_roses - (roses_to_mother + roses_to_grandmother + roses_kept) = 4 :=
by
  sorry

end roses_to_sister_l1487_148799


namespace point_on_x_axis_coordinates_l1487_148745

theorem point_on_x_axis_coordinates (a : ℝ) (hx : a - 3 = 0) : (a + 2, a - 3) = (5, 0) :=
by
  sorry

end point_on_x_axis_coordinates_l1487_148745


namespace polynomial_roots_l1487_148752

-- The statement that we need to prove
theorem polynomial_roots (a b : ℚ) (h : (2 + Real.sqrt 3) ^ 3 + 4 * (2 + Real.sqrt 3) ^ 2 + a * (2 + Real.sqrt 3) + b = 0) :
  ((Polynomial.C a * Polynomial.X ^ 3 + Polynomial.C (4 : ℚ) * Polynomial.X ^ 2 + Polynomial.C a * Polynomial.X + Polynomial.C b) = Polynomial.X ^ 3 + Polynomial.C (4 : ℚ) * Polynomial.X ^ 2 + Polynomial.C a * Polynomial.X + Polynomial.C b) →
  (2 - Real.sqrt 3) ^ 3 + 4 * (2 - Real.sqrt 3) ^ 2 + a * (2 - Real.sqrt 3) + b = 0 ∧ -8 ^ 3 + 4 * (-8) ^ 2 + a * (-8) + b = 0 := sorry

end polynomial_roots_l1487_148752


namespace system1_solution_system2_solution_l1487_148751

theorem system1_solution : 
  ∃ (x y : ℤ), 2 * x + 3 * y = -1 ∧ y = 4 * x - 5 ∧ x = 1 ∧ y = -1 := by 
    sorry

theorem system2_solution : 
  ∃ (x y : ℤ), 3 * x + 2 * y = 20 ∧ 4 * x - 5 * y = 19 ∧ x = 6 ∧ y = 1 := by 
    sorry

end system1_solution_system2_solution_l1487_148751


namespace quadratic_common_root_l1487_148788

theorem quadratic_common_root (b : ℤ) :
  (∃ x, 2 * x^2 + (3 * b - 1) * x - 3 = 0 ∧ 6 * x^2 - (2 * b - 3) * x - 1 = 0) ↔ b = 2 := 
sorry

end quadratic_common_root_l1487_148788


namespace partition_solution_l1487_148712

noncomputable def partitions (a m n x : ℝ) : Prop :=
  a = x + n * (a - m * x)

theorem partition_solution (a m n : ℝ) (h : n * m < 1) :
  partitions a m n (a * (1 - n) / (1 - n * m)) :=
by
  sorry

end partition_solution_l1487_148712


namespace larger_number_l1487_148759

theorem larger_number (x y : ℝ) (h1 : x + y = 30) (h2 : x - y = 4) : x = 17 :=
by
sorry

end larger_number_l1487_148759


namespace find_r_l1487_148722

theorem find_r (k r : ℝ) 
  (h1 : 7 = k * 3^r) 
  (h2 : 49 = k * 9^r) : 
  r = Real.log 7 / Real.log 3 :=
by
  sorry

end find_r_l1487_148722


namespace regular_polygon_sides_l1487_148793

theorem regular_polygon_sides (n : ℕ) (h1 : 360 / n = 18) : n = 20 :=
sorry

end regular_polygon_sides_l1487_148793


namespace geometric_series_common_ratio_l1487_148758

theorem geometric_series_common_ratio (a r S : ℝ) 
  (hS : S = a / (1 - r)) 
  (h64 : (a * r^4) / (1 - r) = S / 64) : 
  r = 1 / 2 :=
by
  sorry

end geometric_series_common_ratio_l1487_148758


namespace fractions_sum_simplified_l1487_148775

noncomputable def frac12over15 : ℚ := 12 / 15
noncomputable def frac7over9 : ℚ := 7 / 9
noncomputable def frac1and1over6 : ℚ := 1 + 1 / 6

theorem fractions_sum_simplified :
  frac12over15 + frac7over9 + frac1and1over6 = 247 / 90 :=
by
  -- This step will be left as a proof to complete.
  sorry

end fractions_sum_simplified_l1487_148775


namespace slope_of_line_through_midpoints_l1487_148716

theorem slope_of_line_through_midpoints :
  let P₁ := (1, 2)
  let P₂ := (3, 8)
  let P₃ := (4, 3)
  let P₄ := (7, 9)
  let M₁ := ( (P₁.1 + P₂.1)/2, (P₁.2 + P₂.2)/2 )
  let M₂ := ( (P₃.1 + P₄.1)/2, (P₃.2 + P₄.2)/2 )
  let slope := (M₂.2 - M₁.2) / (M₂.1 - M₁.1)
  slope = 2/7 :=
by
  sorry

end slope_of_line_through_midpoints_l1487_148716


namespace original_people_count_l1487_148709

theorem original_people_count (x : ℕ) 
  (H1 : (x - x / 3) / 2 = 15) : x = 45 := by
  sorry

end original_people_count_l1487_148709


namespace f_of_3_is_log2_3_l1487_148765

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_condition : ∀ x : ℝ, f (2 ^ x) = x

theorem f_of_3_is_log2_3 : f 3 = Real.log 3 / Real.log 2 := sorry

end f_of_3_is_log2_3_l1487_148765


namespace total_tickets_l1487_148798

theorem total_tickets (tickets_first_day tickets_second_day tickets_third_day : ℕ) 
  (h1 : tickets_first_day = 5 * 4) 
  (h2 : tickets_second_day = 32)
  (h3 : tickets_third_day = 28) :
  tickets_first_day + tickets_second_day + tickets_third_day = 80 := by
  sorry

end total_tickets_l1487_148798


namespace Tom_green_marbles_l1487_148708

-- Define the given variables
def Sara_green_marbles : Nat := 3
def Total_green_marbles : Nat := 7

-- The statement to be proven
theorem Tom_green_marbles : (Total_green_marbles - Sara_green_marbles) = 4 := by
  sorry

end Tom_green_marbles_l1487_148708


namespace part1_part2_part3_l1487_148738

-- Part 1
theorem part1 (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : (x + y) * (y + z) * (z + x) ≥ 8 * x * y * z :=
sorry

-- Part 2
theorem part2 (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : x^2 + y^2 + z^2 ≥ x * y + y * z + z * x :=
sorry

-- Part 3
theorem part3 (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : x ^ x * y ^ y * z ^ z ≥ (x * y * z) ^ ((x + y + z) / 3) :=
sorry

#print axioms part1
#print axioms part2
#print axioms part3

end part1_part2_part3_l1487_148738


namespace women_in_the_minority_l1487_148779

theorem women_in_the_minority (total_employees : ℕ) (female_employees : ℕ) (h : female_employees < total_employees * 20 / 100) : 
  (female_employees < total_employees / 2) :=
by
  sorry

end women_in_the_minority_l1487_148779


namespace length_of_OP_is_sqrt_200_div_3_l1487_148783

open Real

def square (a : ℝ) := a * a

theorem length_of_OP_is_sqrt_200_div_3 (KL MO MP OP : ℝ) (h₁ : KL = 10)
  (h₂: MO = MP) (h₃: square (10) = 100)
  (h₄ : 1 / 6 * 100 = 1 / 2 * (MO * MP)) : OP = sqrt (200/3) :=
by
  sorry

end length_of_OP_is_sqrt_200_div_3_l1487_148783


namespace calculate_total_students_l1487_148792

/-- Define the number of students who like basketball, cricket, and soccer. -/
def likes_basketball : ℕ := 7
def likes_cricket : ℕ := 10
def likes_soccer : ℕ := 8
def likes_all_three : ℕ := 2
def likes_basketball_and_cricket : ℕ := 5
def likes_basketball_and_soccer : ℕ := 4
def likes_cricket_and_soccer : ℕ := 3

/-- Calculate the number of students who like at least one sport using the principle of inclusion-exclusion. -/
def students_who_like_at_least_one_sport (b c s bc bs cs bcs : ℕ) : ℕ :=
  b + c + s - (bc + bs + cs) + bcs

theorem calculate_total_students :
  students_who_like_at_least_one_sport likes_basketball likes_cricket likes_soccer 
    (likes_basketball_and_cricket - likes_all_three) 
    (likes_basketball_and_soccer - likes_all_three) 
    (likes_cricket_and_soccer - likes_all_three) 
    likes_all_three = 21 := 
by
  sorry

end calculate_total_students_l1487_148792


namespace shirts_made_today_l1487_148762

def shirts_per_minute : ℕ := 6
def minutes_yesterday : ℕ := 12
def total_shirts : ℕ := 156
def shirts_yesterday : ℕ := shirts_per_minute * minutes_yesterday
def shirts_today : ℕ := total_shirts - shirts_yesterday

theorem shirts_made_today :
  shirts_today = 84 :=
by
  sorry

end shirts_made_today_l1487_148762


namespace first_discount_l1487_148731

theorem first_discount (P F : ℕ) (D₂ : ℝ) (D₁ : ℝ) 
  (hP : P = 150) 
  (hF : F = 105)
  (hD₂ : D₂ = 12.5)
  (hF_eq : F = P * (1 - D₁ / 100) * (1 - D₂ / 100)) : 
  D₁ = 20 :=
by
  sorry

end first_discount_l1487_148731


namespace magnitude_squared_l1487_148717

-- Let z be the complex number 3 + 4i
def z : ℂ := 3 + 4 * Complex.I

-- Prove that the magnitude of z squared equals 25
theorem magnitude_squared : Complex.abs z ^ 2 = 25 := by
  -- The term "by" starts the proof block, and "sorry" allows us to skip the proof details.
  sorry

end magnitude_squared_l1487_148717


namespace value_of_x_l1487_148729

theorem value_of_x (z : ℤ) (h1 : z = 100) (y : ℤ) (h2 : y = z / 10) (x : ℤ) (h3 : x = y / 3) : 
  x = 10 / 3 := 
by
  -- The proof is skipped
  sorry

end value_of_x_l1487_148729


namespace sum_mod_six_l1487_148761

theorem sum_mod_six (n : ℤ) : ((10 - 2 * n) + (4 * n + 2)) % 6 = 0 :=
by {
  sorry
}

end sum_mod_six_l1487_148761


namespace polynomial_independent_of_m_l1487_148743

theorem polynomial_independent_of_m (m : ℝ) (x : ℝ) (h : 6 * x^2 + (1 - 2 * m) * x + 7 * m = 6 * x^2 + x) : 
  x = 7 / 2 :=
by
  sorry

end polynomial_independent_of_m_l1487_148743


namespace harry_morning_ratio_l1487_148700

-- Define the total morning routine time
def total_morning_routine_time : ℕ := 45

-- Define the time taken to buy coffee and a bagel
def time_buying_coffee_and_bagel : ℕ := 15

-- Calculate the time spent reading the paper and eating
def time_reading_and_eating : ℕ :=
  total_morning_routine_time - time_buying_coffee_and_bagel

-- Define the ratio of the time spent reading and eating to buying coffee and a bagel
def ratio_reading_eating_to_buying_coffee_bagel : ℚ :=
  (time_reading_and_eating : ℚ) / (time_buying_coffee_and_bagel : ℚ)

-- State the theorem
theorem harry_morning_ratio : ratio_reading_eating_to_buying_coffee_bagel = 2 := 
by
  sorry

end harry_morning_ratio_l1487_148700


namespace david_profit_l1487_148715

theorem david_profit (weight : ℕ) (cost sell_price : ℝ) (h_weight : weight = 50) (h_cost : cost = 50) (h_sell_price : sell_price = 1.20) : 
  sell_price * weight - cost = 10 :=
by sorry

end david_profit_l1487_148715


namespace monotonic_increasing_interval_l1487_148707

noncomputable def log_base_1_div_3 (t : ℝ) := Real.log t / Real.log (1/3)

def quadratic (x : ℝ) := 4 + 3 * x - x^2

theorem monotonic_increasing_interval :
  ∃ (a b : ℝ), (∀ x, a < x ∧ x < b → (log_base_1_div_3 (quadratic x)) < (log_base_1_div_3 (quadratic (x + ε))) ∧
               ((-1 : ℝ) < x ∧ x < 4) ∧ (quadratic x > 0)) ↔ (a, b) = (3 / 2, 4) :=
by
  sorry

end monotonic_increasing_interval_l1487_148707


namespace sharks_in_Cape_May_August_l1487_148795

section
variable {D_J C_J D_A C_A : ℕ}

-- Given conditions
theorem sharks_in_Cape_May_August 
  (h1 : C_J = 2 * D_J) 
  (h2 : C_A = 5 + 3 * D_A) 
  (h3 : D_J = 23) 
  (h4 : D_A = D_J) : 
  C_A = 74 := 
by 
  -- Skipped the proof steps 
  sorry
end

end sharks_in_Cape_May_August_l1487_148795


namespace grid_blue_probability_l1487_148718

-- Define the problem in Lean
theorem grid_blue_probability :
  let n := 4
  let p_tile_blue := 1 / 2
  let invariant_prob := (p_tile_blue ^ (n / 2))
  let pair_prob := (p_tile_blue * p_tile_blue)
  let total_pairs := (n * n / 2 - n / 2)
  let final_prob := (invariant_prob ^ 2) * (pair_prob ^ total_pairs)
  final_prob = 1 / 65536 := by
  sorry

end grid_blue_probability_l1487_148718


namespace acute_triangle_condition_l1487_148736

theorem acute_triangle_condition (a b c : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  (|a^2 - b^2| < c^2 ∧ c^2 < a^2 + b^2) ↔ (a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ c^2 + a^2 > b^2) :=
sorry

end acute_triangle_condition_l1487_148736


namespace zero_intercept_and_distinct_roots_l1487_148772

noncomputable def Q (x a' b' c' d' : ℝ) : ℝ := x^4 + a' * x^3 + b' * x^2 + c' * x + d'

theorem zero_intercept_and_distinct_roots (a' b' c' d' : ℝ) (u v w : ℝ) (h_distinct : u ≠ v ∧ v ≠ w ∧ u ≠ w) (h_intercept_at_zero : d' = 0)
(h_Q_form : ∀ x, Q x a' b' c' d' = x * (x - u) * (x - v) * (x - w)) : c' ≠ 0 :=
by
  sorry

end zero_intercept_and_distinct_roots_l1487_148772


namespace rectangle_ratio_l1487_148771

open Real

theorem rectangle_ratio (A B C D E : Point) (rat : ℚ) : 
  let area_rect := 1
  let area_pentagon := (7 / 10 : ℚ)
  let area_triangle_AEC := 3 / 10
  let area_triangle_ECD := 1 / 5
  let x := 3 * EA
  let y := 2 * EA
  let diag_longer_side := sqrt (5 * EA ^ 2)
  let diag_shorter_side := EA * sqrt 5
  let ratio := sqrt 5 
  ( area_pentagon == area_rect * (7 / 10) ) →
  ( area_triangle_AEC + area_pentagon = area_rect ) →
  ( area_triangle_AEC == area_rect - area_pentagon ) →
  ( ratio == diag_longer_side / diag_shorter_side ) :=
  sorry

end rectangle_ratio_l1487_148771


namespace approximate_reading_l1487_148725

-- Define the given conditions
def arrow_location_between (a b : ℝ) : Prop := a < 42.3 ∧ 42.6 < b

-- Statement of the proof problem
theorem approximate_reading (a b : ℝ) (ha : arrow_location_between a b) :
  a = 42.3 :=
sorry

end approximate_reading_l1487_148725


namespace integral_curve_has_inflection_points_l1487_148733

theorem integral_curve_has_inflection_points (x y : ℝ) (f : ℝ → ℝ → ℝ) :
  f x y = y - x^2 + 2*x - 2 →
  (∃ y' y'' : ℝ, y' = f x y ∧ y'' = y - x^2 ∧ y'' = 0) ↔ y = x^2 :=
by
  sorry

end integral_curve_has_inflection_points_l1487_148733


namespace correct_fraction_simplification_l1487_148796

theorem correct_fraction_simplification (a b : ℝ) (h : a ≠ b) : 
  (∀ (c d : ℝ), (c ≠ d) → (a+2 = c → b+2 = d → (a+2)/d ≠ a/b))
  ∧ (∀ (e f : ℝ), (e ≠ f) → (a-2 = e → b-2 = f → (a-2)/f ≠ a/b))
  ∧ (∀ (g h : ℝ), (g ≠ h) → (a^2 = g → b^2 = h → a^2/h ≠ a/b))
  ∧ (a / b = ( (1/2)*a / (1/2)*b )) := 
sorry

end correct_fraction_simplification_l1487_148796


namespace complement_M_l1487_148774

def U : Set ℕ := {1, 2, 3, 4}

def M : Set ℕ := {x | (x - 1) * (x - 4) = 0}

theorem complement_M :
  (U \ M) = {2, 3} := by
  sorry

end complement_M_l1487_148774


namespace tim_out_of_pocket_cost_l1487_148714

noncomputable def totalOutOfPocketCost : ℝ :=
  let mriCost := 1200
  let xrayCost := 500
  let examinationCost := 400 * (45 / 60)
  let feeForBeingSeen := 150
  let consultationFee := 75
  let physicalTherapyCost := 100 * 8
  let totalCostBeforeInsurance := mriCost + xrayCost + examinationCost + feeForBeingSeen + consultationFee + physicalTherapyCost
  let insuranceCoverage := 0.70 * totalCostBeforeInsurance
  let outOfPocketCost := totalCostBeforeInsurance - insuranceCoverage
  outOfPocketCost

theorem tim_out_of_pocket_cost : totalOutOfPocketCost = 907.50 :=
  by
    -- Proof will be provided here
    sorry

end tim_out_of_pocket_cost_l1487_148714


namespace find_f_2_l1487_148766

noncomputable def f (a b x : ℝ) : ℝ := x^5 + a * x^3 + b * x - 8

theorem find_f_2 (a b : ℝ) (h : f a b (-2) = 10) : f a b 2 = -26 :=
by
  sorry

end find_f_2_l1487_148766


namespace video_minutes_per_week_l1487_148739

theorem video_minutes_per_week
  (daily_videos : ℕ := 3)
  (short_video_length : ℕ := 2)
  (long_video_multiplier : ℕ := 6)
  (days_in_week : ℕ := 7) :
  (2 * short_video_length + long_video_multiplier * short_video_length) * days_in_week = 112 := 
by 
  -- conditions
  let short_videos_per_day := 2
  let long_video_length := long_video_multiplier * short_video_length
  let daily_total := short_videos_per_day * short_video_length + long_video_length
  let weekly_total := daily_total * days_in_week
  -- proof
  sorry

end video_minutes_per_week_l1487_148739


namespace radius_of_larger_circle_l1487_148723

theorem radius_of_larger_circle (r R AC BC AB : ℝ)
  (h1 : R = 4 * r)
  (h2 : AC = 8 * r)
  (h3 : BC^2 + AB^2 = AC^2)
  (h4 : AB = 16) :
  R = 32 :=
by
  sorry

end radius_of_larger_circle_l1487_148723


namespace businesses_brandon_can_apply_to_l1487_148770

-- Definitions of the given conditions in the problem
variables (x y : ℕ)

-- Define the total, fired, and quit businesses
def total_businesses : ℕ := 72
def fired_businesses : ℕ := 36
def quit_businesses : ℕ := 24

-- Define the unique businesses Brandon can still apply to, considering common businesses and reapplications
def businesses_can_apply_to : ℕ := (12 + x) + y

-- The theorem to prove
theorem businesses_brandon_can_apply_to (x y : ℕ) : businesses_can_apply_to x y = 12 + x + y := by
  unfold businesses_can_apply_to
  sorry

end businesses_brandon_can_apply_to_l1487_148770


namespace lattice_point_count_l1487_148767

noncomputable def countLatticePoints (N : ℤ) : ℤ :=
  2 * N * (N + 1) + 1

theorem lattice_point_count (N : ℤ) (hN : 71 * N > 0) :
    ∃ P, P = countLatticePoints N := sorry

end lattice_point_count_l1487_148767


namespace arithmetic_sequence_n_equals_8_l1487_148750

theorem arithmetic_sequence_n_equals_8 :
  (∀ (a b c : ℕ), a + (1 / 4) * c = 2 * (1 / 2) * b) → ∃ n : ℕ, n = 8 :=
by 
  sorry

end arithmetic_sequence_n_equals_8_l1487_148750


namespace three_digit_number_cubed_sum_l1487_148791

theorem three_digit_number_cubed_sum {n : ℕ} (h1 : 100 ≤ n) (h2 : n ≤ 999) :
  (∃ a b c : ℕ, a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧ n = 100 * a + 10 * b + c ∧ n = a^3 + b^3 + c^3) ↔
  n = 153 ∨ n = 370 ∨ n = 371 ∨ n = 407 :=
by
  sorry

end three_digit_number_cubed_sum_l1487_148791


namespace never_prime_except_three_l1487_148747

theorem never_prime_except_three (p : ℕ) (hp : Nat.Prime p) :
  p^2 + 8 = 17 ∨ ∃ k, (k ≠ 1 ∧ k ≠ p^2 + 8 ∧ k ∣ (p^2 + 8)) := by
  sorry

end never_prime_except_three_l1487_148747


namespace calculate_power_l1487_148756

variable (x y : ℝ)

theorem calculate_power :
  (- (1 / 2) * x^2 * y)^3 = - (1 / 8) * x^6 * y^3 :=
sorry

end calculate_power_l1487_148756


namespace probability_of_valid_quadrilateral_l1487_148746

-- Define a regular octagon
def regular_octagon_sides : ℕ := 8

-- Total number of ways to choose 4 sides from 8 sides
def total_ways_choose_four_sides : ℕ := Nat.choose 8 4

-- Number of ways to choose 4 adjacent sides (invalid)
def invalid_adjacent_ways : ℕ := 8

-- Number of ways to choose 4 sides with 3 adjacent unchosen sides (invalid)
def invalid_three_adjacent_unchosen_ways : ℕ := 8 * 3

-- Total number of invalid ways
def total_invalid_ways : ℕ := invalid_adjacent_ways + invalid_three_adjacent_unchosen_ways

-- Total number of valid ways
def total_valid_ways : ℕ := total_ways_choose_four_sides - total_invalid_ways

-- Probability of forming a quadrilateral that contains the octagon
def probability_valid_quadrilateral : ℚ :=
  (total_valid_ways : ℚ) / (total_ways_choose_four_sides : ℚ)

-- Theorem statement
theorem probability_of_valid_quadrilateral :
  probability_valid_quadrilateral = 19 / 35 :=
by
  sorry

end probability_of_valid_quadrilateral_l1487_148746


namespace determine_a_l1487_148763

-- Given conditions
variable {a b : ℝ}
variable (h_neg : a < 0) (h_pos : b > 0) (h_max : ∀ x, -2 ≤ a * sin (b * x) ∧ a * sin (b * x) ≤ 2)

-- Statement to prove
theorem determine_a : a = -2 := by
  sorry

end determine_a_l1487_148763


namespace asbestos_tiles_width_l1487_148735

theorem asbestos_tiles_width (n : ℕ) (h : 0 < n) :
  let width_per_tile := 60
  let overlap := 10
  let effective_width := width_per_tile - overlap
  width_per_tile + (n - 1) * effective_width = 50 * n + 10 := by
sorry

end asbestos_tiles_width_l1487_148735


namespace monkey_tree_height_l1487_148713

theorem monkey_tree_height (hours: ℕ) (hop ft_per_hour : ℕ) (slip ft_per_hour : ℕ) (net_progress : ℕ) (final_hour : ℕ) (total_height : ℕ) :
  (hours = 18) ∧
  (hop = 3) ∧
  (slip = 2) ∧
  (net_progress = hop - slip) ∧
  (net_progress = 1) ∧
  (final_hour = 1) ∧
  (total_height = (hours - 1) * net_progress + hop) ∧
  (total_height = 20) :=
by
  sorry

end monkey_tree_height_l1487_148713


namespace fractions_order_l1487_148740

theorem fractions_order : (23 / 18) < (21 / 16) ∧ (21 / 16) < (25 / 19) :=
by
  sorry

end fractions_order_l1487_148740


namespace room_length_calculation_l1487_148703

-- Definitions of the problem conditions
def room_volume : ℝ := 10000
def room_width : ℝ := 10
def room_height : ℝ := 10

-- Statement to prove
theorem room_length_calculation : ∃ L : ℝ, L = room_volume / (room_width * room_height) ∧ L = 100 :=
by
  sorry

end room_length_calculation_l1487_148703


namespace tony_income_l1487_148785

-- Definitions for the given conditions
def investment : ℝ := 3200
def purchase_price : ℝ := 85
def dividend : ℝ := 6.640625

-- Theorem stating Tony's income based on the conditions
theorem tony_income : (investment / purchase_price) * dividend = 250 :=
by
  sorry

end tony_income_l1487_148785


namespace product_complex_numbers_l1487_148778

noncomputable def Q : ℂ := 3 + 4 * Complex.I
noncomputable def E : ℂ := 2 * Complex.I
noncomputable def D : ℂ := 3 - 4 * Complex.I
noncomputable def R : ℝ := 2

theorem product_complex_numbers : Q * E * D * (R : ℂ) = 100 * Complex.I := by
  sorry

end product_complex_numbers_l1487_148778


namespace geometric_sequence_a4_l1487_148773

-- Define the terms of the geometric sequence
variable {a : ℕ → ℝ}

-- Define the conditions of the problem
def a2_cond : Prop := a 2 = 2
def a6_cond : Prop := a 6 = 32

-- Define the theorem we want to prove
theorem geometric_sequence_a4 (a2_cond : a 2 = 2) (a6_cond : a 6 = 32) : a 4 = 8 := by
  sorry

end geometric_sequence_a4_l1487_148773


namespace symmetric_about_pi_over_4_l1487_148797

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.sin x + Real.cos x

theorem symmetric_about_pi_over_4 (a : ℝ) :
  (∀ x : ℝ, f a (x + π / 4) = f a (-(x + π / 4))) → a = 1 := by
  unfold f
  sorry

end symmetric_about_pi_over_4_l1487_148797


namespace sum_of_coefficients_zero_l1487_148769

open Real

theorem sum_of_coefficients_zero (a b c p1 p2 q1 q2 : ℝ)
  (h1 : ∃ p1 p2 : ℝ, p1 ≠ p2 ∧ a * p1^2 + b * p1 + c = 0 ∧ a * p2^2 + b * p2 + c = 0)
  (h2 : ∃ q1 q2 : ℝ, q1 ≠ q2 ∧ c * q1^2 + b * q1 + a = 0 ∧ c * q2^2 + b * q2 + a = 0)
  (h3 : q1 = p1 + (p2 - p1) / 2 ∧ p2 = p1 + (p2 - p1) ∧ q2 = p1 + 3 * (p2 - p1) / 2) :
  a + c = 0 := sorry

end sum_of_coefficients_zero_l1487_148769


namespace inequality_holds_l1487_148721

variable (x a : ℝ)

def tensor (x y : ℝ) : ℝ :=
  (1 - x) * (1 + y)

theorem inequality_holds (h : ∀ x : ℝ, tensor (x - a) (x + a) < 1) : -2 < a ∧ a < 0 := by
  sorry

end inequality_holds_l1487_148721


namespace arithmetic_sequence_eighth_term_l1487_148705

theorem arithmetic_sequence_eighth_term (a d : ℤ)
  (h₁ : a + 3 * d = 23)
  (h₂ : a + 5 * d = 47) :
  a + 7 * d = 71 :=
sorry

end arithmetic_sequence_eighth_term_l1487_148705


namespace min_sum_p_q_r_s_l1487_148748

theorem min_sum_p_q_r_s (p q r s : ℕ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) (hs : 0 < s)
    (h1 : 2 * p = 10 * p - 15 * q)
    (h2 : 2 * q = 6 * p - 9 * q)
    (h3 : 3 * r = 10 * r - 15 * s)
    (h4 : 3 * s = 6 * r - 9 * s) : p + q + r + s = 45 := by
  sorry

end min_sum_p_q_r_s_l1487_148748


namespace reduced_price_l1487_148755

theorem reduced_price (P : ℝ) (hP : P = 56)
    (original_qty : ℝ := 800 / P)
    (reduced_qty : ℝ := 800 / (0.65 * P))
    (diff_qty : ℝ := reduced_qty - original_qty)
    (difference_condition : diff_qty = 5) :
  0.65 * P = 36.4 :=
by
  rw [hP]
  sorry

end reduced_price_l1487_148755


namespace average_capacity_is_3_65_l1487_148719

/-- Define the capacities of the jars as a list--/
def jarCapacities : List ℚ := [2, 1/4, 8, 1.5, 0.75, 3, 10]

/-- Calculate the average jar capacity --/
def averageCapacity (capacities : List ℚ) : ℚ :=
  (capacities.sum) / (capacities.length)

/-- The average jar capacity for the given list of jar capacities is 3.65 liters. --/
theorem average_capacity_is_3_65 :
  averageCapacity jarCapacities = 3.65 := 
by
  unfold averageCapacity
  dsimp [jarCapacities]
  norm_num
  sorry

end average_capacity_is_3_65_l1487_148719


namespace find_x_l1487_148744

theorem find_x (x : ℝ) (h1: x > 0) (h2 : 1 / 2 * x * (3 * x) = 72) : x = 4 * Real.sqrt 3 :=
sorry

end find_x_l1487_148744


namespace find_k_value_l1487_148737

theorem find_k_value (k : ℝ) (x y : ℝ) (h1 : -3 * x + 2 * y = k) (h2 : 0.75 * x + y = 16) (h3 : x = -6) : k = 59 :=
by 
  sorry

end find_k_value_l1487_148737


namespace imo1989_q3_l1487_148742

theorem imo1989_q3 (a b : ℤ) (h1 : ¬ (∃ x : ℕ, a = x ^ 2))
                   (h2 : ¬ (∃ y : ℕ, b = y ^ 2))
                   (h3 : ∃ (x y z w : ℤ), x ^ 2 - a * y ^ 2 - b * z ^ 2 + a * b * w ^ 2 = 0 
                                           ∧ (x, y, z, w) ≠ (0, 0, 0, 0)) :
                   ∃ (x y z : ℤ), x ^ 2 - a * y ^ 2 - b * z ^ 2 = 0 ∧ (x, y, z) ≠ (0, 0, 0) := 
sorry

end imo1989_q3_l1487_148742


namespace sum_after_50_rounds_l1487_148710

def initial_states : List ℤ := [1, 0, -1]

def operation (n : ℤ) : ℤ :=
  match n with
  | 1   => n * n * n
  | 0   => n * n
  | -1  => -n
  | _ => n  -- although not necessary for current problem, this covers other possible states

def process_calculator (state : ℤ) (times: ℕ) : ℤ :=
  if state = 1 then state
  else if state = 0 then state
  else if state = -1 then state * (-1) ^ times
  else state

theorem sum_after_50_rounds :
  let final_states := initial_states.map (fun s => process_calculator s 50)
  final_states.sum = 2 := by
  simp only [initial_states, process_calculator]
  simp
  sorry

end sum_after_50_rounds_l1487_148710


namespace cubes_sum_expr_l1487_148794

variable {a b s p : ℝ}

theorem cubes_sum_expr (h1 : s = a + b) (h2 : p = a * b) : a^3 + b^3 = s^3 - 3 * s * p := by
  sorry

end cubes_sum_expr_l1487_148794


namespace max_total_cut_length_l1487_148702

theorem max_total_cut_length :
  let side_length := 30
  let num_pieces := 225
  let area_per_piece := (side_length ^ 2) / num_pieces
  let outer_perimeter := 4 * side_length
  let max_perimeter_per_piece := 10
  (num_pieces * max_perimeter_per_piece - outer_perimeter) / 2 = 1065 :=
by
  sorry

end max_total_cut_length_l1487_148702


namespace min_height_of_box_with_surface_area_condition_l1487_148704

theorem min_height_of_box_with_surface_area_condition {x : ℕ}  
(h : 2*x^2 + 4*x*(x + 6) ≥ 150) (hx: x ≥ 5) : (x + 6) = 11 := by
  sorry

end min_height_of_box_with_surface_area_condition_l1487_148704


namespace seq_eighth_term_l1487_148784

theorem seq_eighth_term : (8^2 + 2 * 8 - 1 = 79) :=
by
  sorry

end seq_eighth_term_l1487_148784


namespace percent_boys_in_class_l1487_148790

-- Define the conditions given in the problem
def initial_ratio (b g : ℕ) : Prop := b = 3 * g / 4

def total_students_after_new_girls (total : ℕ) (new_girls : ℕ) : Prop :=
  total = 42 ∧ new_girls = 4

-- Define the percentage calculation correctness
def percentage_of_boys (boys total : ℕ) (percentage : ℚ) : Prop :=
  percentage = (boys : ℚ) / (total : ℚ) * 100

-- State the theorem to be proven
theorem percent_boys_in_class
  (b g : ℕ)   -- Number of boys and initial number of girls
  (total new_girls : ℕ) -- Total students after new girls joined and number of new girls
  (percentage : ℚ) -- The percentage of boys in the class
  (h_initial_ratio : initial_ratio b g)
  (h_total_students : total_students_after_new_girls total new_girls)
  (h_goals : g + new_girls = total - b)
  (h_correct_calc : percentage = 35.71) :
  percentage_of_boys b total percentage :=
by
  sorry

end percent_boys_in_class_l1487_148790


namespace mink_babies_l1487_148781

theorem mink_babies (B : ℕ) (h_coats : 7 * 15 = 105)
    (h_minks: 30 + 30 * B = 210) :
  B = 6 :=
by
  sorry

end mink_babies_l1487_148781


namespace evaluate_4_over_04_eq_400_l1487_148782

noncomputable def evaluate_fraction : Float :=
  (0.4)^4 / (0.04)^3

theorem evaluate_4_over_04_eq_400 : evaluate_fraction = 400 :=
by
  sorry

end evaluate_4_over_04_eq_400_l1487_148782


namespace positive_integral_solution_exists_l1487_148741

theorem positive_integral_solution_exists :
  ∃ n : ℕ, n > 0 ∧
  ( (n * (n + 1) * (2 * n + 1)) * 100 = 27 * 6 * (n * (n + 1))^2 ) ∧ n = 5 :=
by {
  sorry
}

end positive_integral_solution_exists_l1487_148741


namespace perfect_squares_ending_in_5_or_6_lt_2000_l1487_148780

theorem perfect_squares_ending_in_5_or_6_lt_2000 :
  ∃ (n : ℕ), n = 9 ∧ ∀ k, 1 ≤ k ∧ k ≤ 44 → 
  (∃ m, m * m < 2000 ∧ (m % 10 = 5 ∨ m % 10 = 6)) :=
by
  sorry

end perfect_squares_ending_in_5_or_6_lt_2000_l1487_148780


namespace radius_range_of_circle_l1487_148768

theorem radius_range_of_circle (r : ℝ) :
  (∀ (x y : ℝ), (x - 3)^2 + (y + 5)^2 = r^2 → 
  (abs (4*x - 3*y - 2) = 1)) →
  4 < r ∧ r < 6 :=
by
  sorry

end radius_range_of_circle_l1487_148768


namespace minimize_fraction_sum_l1487_148720

theorem minimize_fraction_sum (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h_sum : a + b + c = 6) :
  (9 / a + 4 / b + 25 / c) ≥ 50 / 3 :=
sorry

end minimize_fraction_sum_l1487_148720


namespace angle_A_range_find_b_l1487_148749

-- Definitions based on problem conditions
variable {a b c S : ℝ}
variable {A B C : ℝ}
variable {x : ℝ}

-- First statement: range of values for A
theorem angle_A_range (h1 : c * b * Real.cos A ≤ 2 * Real.sqrt 3 * S)
                      (h2 : S = 1/2 * b * c * Real.sin A)
                      (h3 : 0 < A ∧ A < π) : π / 6 ≤ A ∧ A < π := 
sorry

-- Second statement: value of b
theorem find_b (h1 : Real.tan A = x ∧ Real.tan B = 2 * x ∧ Real.tan C = 3 * x)
               (h2 : x = 1)
               (h3 : c = 1) : b = 2 * Real.sqrt 2 / 3 :=
sorry

end angle_A_range_find_b_l1487_148749


namespace inequality_proof_l1487_148730

theorem inequality_proof (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a * b ≥ 1) :
  (a + 2 * b + 2 / (a + 1)) * (b + 2 * a + 2 / (b + 1)) ≥ 16 :=
by
  sorry

end inequality_proof_l1487_148730


namespace rationalize_denominator_l1487_148753

theorem rationalize_denominator :
  (Real.sqrt (5 / 12)) = ((Real.sqrt 15) / 6) :=
sorry

end rationalize_denominator_l1487_148753


namespace circle_equation_coefficients_l1487_148776

theorem circle_equation_coefficients (a : ℝ) (x y : ℝ) : 
  (a^2 * x^2 + (a + 2) * y^2 + 2 * a * x + a = 0) → (a = -1) :=
by 
  sorry

end circle_equation_coefficients_l1487_148776


namespace jana_walk_distance_l1487_148764

-- Define the time taken to walk one mile and the rest period
def walk_time_per_mile : ℕ := 24
def rest_time_per_mile : ℕ := 6

-- Define the total time spent per mile (walking + resting)
def total_time_per_mile : ℕ := walk_time_per_mile + rest_time_per_mile

-- Define the total available time
def total_available_time : ℕ := 78

-- Define the number of complete cycles of walking and resting within the total available time
def complete_cycles : ℕ := total_available_time / total_time_per_mile

-- Define the distance walked per cycle (in miles)
def distance_per_cycle : ℝ := 1.0

-- Define the total distance walked
def total_distance_walked : ℝ := complete_cycles * distance_per_cycle

-- The proof statement
theorem jana_walk_distance : total_distance_walked = 2.0 := by
  sorry

end jana_walk_distance_l1487_148764


namespace painted_cubes_count_l1487_148724

def total_painted_cubes : ℕ := 8 + 48

theorem painted_cubes_count : total_painted_cubes = 56 :=
by 
  -- Step 1: Define the number of cubes with 3 faces painted (8 corners)
  let corners := 8
  -- Step 2: Calculate the number of edge cubes with 2 faces painted
  let edge_middle_cubes_per_edge := 2
  let edges := 12
  let edge_cubes := edge_middle_cubes_per_edge * edges -- this should be 24
  -- Step 3: Calculate the number of face-interior cubes with 2 faces painted
  let face_cubes_per_face := 4
  let faces := 6
  let face_cubes := face_cubes_per_face * faces -- this should be 24
  -- Step 4: Sum them up to get total cubes with at least two faces painted
  let total_cubes := corners + edge_cubes + face_cubes
  show total_cubes = total_painted_cubes
  sorry

end painted_cubes_count_l1487_148724
