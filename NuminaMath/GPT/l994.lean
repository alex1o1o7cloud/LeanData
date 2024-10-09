import Mathlib

namespace xt_inequality_least_constant_l994_99482

theorem xt_inequality (x y z t : ℝ) (h : x ≤ y ∧ y ≤ z ∧ z ≤ t) (h_sum : x * y + x * z + x * t + y * z + y * t + z * t = 1) :
  x * t < 1 / 3 := sorry

theorem least_constant (x y z t : ℝ) (h : x ≤ y ∧ y ≤ z ∧ z ≤ t) (h_sum : x * y + x * z + x * t + y * z + y * t + z * t = 1) :
  ∃ C, ∀ (x t : ℝ), xt < C ∧ C = 1 / 3 := sorry

end xt_inequality_least_constant_l994_99482


namespace percentage_discount_l994_99435

theorem percentage_discount (discounted_price original_price : ℝ) (h1 : discounted_price = 560) (h2 : original_price = 700) :
  (original_price - discounted_price) / original_price * 100 = 20 :=
by
  simp [h1, h2]
  sorry

end percentage_discount_l994_99435


namespace area_of_rhombus_l994_99444

-- Defining conditions for the problem
def d1 : ℝ := 40   -- Length of the first diagonal in meters
def d2 : ℝ := 30   -- Length of the second diagonal in meters

-- Calculating the area of the rhombus
noncomputable def area : ℝ := (d1 * d2) / 2

-- Statement of the theorem
theorem area_of_rhombus : area = 600 := by
  sorry

end area_of_rhombus_l994_99444


namespace find_number_l994_99476

theorem find_number (x n : ℝ) (h1 : (3 / 2) * x - n = 15) (h2 : x = 12) : n = 3 :=
by
  sorry

end find_number_l994_99476


namespace shirt_price_l994_99406

theorem shirt_price (S : ℝ) (h : (5 * S + 5 * 3) / 2 = 10) : S = 1 :=
by
  sorry

end shirt_price_l994_99406


namespace find_two_digit_number_l994_99479

-- A type synonym for digit
def Digit := {n : ℕ // n < 10}

-- Define the conditions
variable (X Y : Digit)
-- The product of the digits is 8
def product_of_digits : Prop := X.val * Y.val = 8

-- When 18 is added, digits are reversed
def digits_reversed : Prop := 10 * X.val + Y.val + 18 = 10 * Y.val + X.val

-- The question translated to Lean: Prove that the two-digit number is 24
theorem find_two_digit_number (h1 : product_of_digits X Y) (h2 : digits_reversed X Y) : 10 * X.val + Y.val = 24 :=
  sorry

end find_two_digit_number_l994_99479


namespace Juanico_age_30_years_from_now_l994_99485

-- Definitions and hypothesis
def currentAgeGladys : ℕ := 30 -- Gladys's current age, since she will be 40 in 10 years
def currentAgeJuanico : ℕ := (1 / 2) * currentAgeGladys - 4 -- Juanico's current age based on Gladys's current age

theorem Juanico_age_30_years_from_now :
  currentAgeJuanico + 30 = 41 :=
by
  -- You would normally fill out the proof here, but we use 'sorry' to skip it.
  sorry

end Juanico_age_30_years_from_now_l994_99485


namespace problem1_problem2_problem3_problem4_l994_99471

theorem problem1 : -20 + (-14) - (-18) - 13 = -29 := by
  sorry

theorem problem2 : (-2) * 3 + (-5) - 4 / (-1/2) = -3 := by
  sorry

theorem problem3 : (-3/8 - 1/6 + 3/4) * (-24) = -5 := by
  sorry

theorem problem4 : -81 / (9/4) * abs (-4/9) - (-3)^3 / 27 = -15 := by
  sorry

end problem1_problem2_problem3_problem4_l994_99471


namespace Maria_students_l994_99403

variable (M J : ℕ)

def conditions : Prop :=
  (M = 4 * J) ∧ (M + J = 2500)

theorem Maria_students : conditions M J → M = 2000 :=
by
  intro h
  sorry

end Maria_students_l994_99403


namespace rectangle_sides_l994_99455

theorem rectangle_sides (n : ℕ) (hpos : n > 0)
  (h1 : (∃ (a : ℕ), (a^2 * n = n)))
  (h2 : (∃ (b : ℕ), (b^2 * (n + 98) = n))) :
  (∃ (l w : ℕ), l * w = n ∧ 
  ((n = 126 ∧ (l = 3 ∧ w = 42 ∨ l = 6 ∧ w = 21)) ∨
  (n = 1152 ∧ l = 24 ∧ w = 48))) :=
sorry

end rectangle_sides_l994_99455


namespace max_stamps_l994_99461

theorem max_stamps (price_per_stamp : ℕ) (total_money : ℕ) (h1 : price_per_stamp = 45) (h2 : total_money = 5000) : ∃ n : ℕ, n = 111 ∧ 45 * n ≤ 5000 ∧ ∀ m : ℕ, (45 * m ≤ 5000) → m ≤ n := 
by
  sorry

end max_stamps_l994_99461


namespace total_cost_l994_99447

variable (E P M : ℝ)

axiom condition1 : E + 3 * P + 2 * M = 240
axiom condition2 : 2 * E + 5 * P + 4 * M = 440

theorem total_cost : 3 * E + 4 * P + 6 * M = 520 := 
sorry

end total_cost_l994_99447


namespace squares_arrangement_l994_99487

noncomputable def arrangement_possible (n : ℕ) (cond : n ≥ 5) : Prop :=
  ∃ (position : ℕ → ℕ × ℕ),
    (∀ i, 1 ≤ i ∧ i ≤ n → 
        ∃ j k, j ≠ k ∧ 
             dist (position i) (position j) = 1 ∧
             dist (position i) (position k) = 1)

theorem squares_arrangement (n : ℕ) (hn : n ≥ 5) :
  arrangement_possible n hn :=
  sorry

end squares_arrangement_l994_99487


namespace student_count_l994_99436

theorem student_count 
( M S N : ℕ ) 
(h1 : N - M = 10) 
(h2 : N - S = 15) 
(h3 : N - (M + S - 7) = 2) : 
N = 34 :=
by
  sorry

end student_count_l994_99436


namespace ratio_of_boys_to_girls_l994_99442

theorem ratio_of_boys_to_girls (boys : ℕ) (students : ℕ) (h1 : boys = 42) (h2 : students = 48) : (boys : ℚ) / (students - boys : ℚ) = 7 / 1 := 
by
  sorry

end ratio_of_boys_to_girls_l994_99442


namespace julie_hourly_rate_l994_99486

variable (daily_hours : ℕ) (weekly_days : ℕ) (monthly_weeks : ℕ) (missed_days : ℕ) (monthly_salary : ℝ)

def total_monthly_hours : ℕ := daily_hours * weekly_days * monthly_weeks - daily_hours * missed_days

theorem julie_hourly_rate : 
    daily_hours = 8 → 
    weekly_days = 6 → 
    monthly_weeks = 4 → 
    missed_days = 1 → 
    monthly_salary = 920 → 
    (monthly_salary / total_monthly_hours daily_hours weekly_days monthly_weeks missed_days) = 5 := by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  norm_num
  sorry

end julie_hourly_rate_l994_99486


namespace product_of_roots_quadratic_l994_99474

noncomputable def product_of_roots (a b c : ℚ) : ℚ :=
  c / a

theorem product_of_roots_quadratic : product_of_roots 14 21 (-250) = -125 / 7 :=
by
  sorry

end product_of_roots_quadratic_l994_99474


namespace angle_conversion_l994_99451

-- Define the known conditions
def full_circle_vens : ℕ := 800
def full_circle_degrees : ℕ := 360
def given_angle_degrees : ℕ := 135
def expected_vens : ℕ := 300

-- Prove that an angle of 135 degrees corresponds to 300 vens.
theorem angle_conversion :
  (given_angle_degrees * full_circle_vens) / full_circle_degrees = expected_vens := by
  sorry

end angle_conversion_l994_99451


namespace investment_value_l994_99407

noncomputable def compound_interest (P : ℕ) (r : ℚ) (n : ℕ) : ℚ :=
  P * (1 + r)^n

theorem investment_value :
  ∀ (P : ℕ) (r : ℚ) (n : ℕ),
  P = 8000 →
  r = 0.05 →
  n = 3 →
  compound_interest P r n = 9250 := by
    intros P r n hP hr hn
    unfold compound_interest
    -- calculation steps would be here
    sorry

end investment_value_l994_99407


namespace part1_part2_l994_99467

noncomputable def f (x : ℝ) : ℝ := (Real.cos x) * (Real.cos (x - Real.pi / 3))

theorem part1 : f (2 * Real.pi / 3) = -1 / 4 :=
by
  sorry

theorem part2 :
  {x : ℝ | f x < 1 / 4} = {x : ℝ | ∃ k : ℤ, x ∈ Set.Ioo (k * Real.pi - 7 * Real.pi / 12) (k * Real.pi - Real.pi / 12)} :=
by
  sorry

end part1_part2_l994_99467


namespace zhang_bing_age_18_l994_99462

theorem zhang_bing_age_18 {x a : ℕ} (h1 : x < 2023) 
  (h2 : a = x - 1953)
  (h3 : a % 9 = 0)
  (h4 : a = (x % 10) + ((x / 10) % 10) + ((x / 100) % 10) + ((x / 1000) % 10)) :
  a = 18 :=
sorry

end zhang_bing_age_18_l994_99462


namespace MrSmithEnglishProof_l994_99475

def MrSmithLearningEnglish : Prop :=
  (∃ (decade: String) (age: String), 
    (decade = "1950's" ∧ age = "in his sixties") ∨ 
    (decade = "1950" ∧ age = "in the sixties") ∨ 
    (decade = "1950's" ∧ age = "over sixty"))
  
def correctAnswer : Prop :=
  MrSmithLearningEnglish →
  (∃ answer, answer = "D")

theorem MrSmithEnglishProof : correctAnswer :=
  sorry

end MrSmithEnglishProof_l994_99475


namespace problem_l994_99473

noncomputable def f (x : ℝ) : ℝ := x^3 + Real.sin x + 1

theorem problem (a : ℝ) (h : f a = 2) : f (-a) = 0 := 
  sorry

end problem_l994_99473


namespace min_soda_packs_90_l994_99483

def soda_packs (n : ℕ) : Prop :=
  ∃ (x y z : ℕ), 6 * x + 12 * y + 24 * z = n

theorem min_soda_packs_90 : (x y z : ℕ) → soda_packs 90 → x + y + z = 5 := by
  sorry

end min_soda_packs_90_l994_99483


namespace second_runner_stop_time_l994_99430

-- Definitions provided by the conditions
def pace_first := 8 -- pace of the first runner in minutes per mile
def pace_second := 7 -- pace of the second runner in minutes per mile
def time_elapsed := 56 -- time elapsed in minutes before the second runner stops
def distance_first := time_elapsed / pace_first -- distance covered by the first runner in miles
def distance_second := time_elapsed / pace_second -- distance covered by the second runner in miles
def distance_gap := distance_second - distance_first -- gap between the runners in miles

-- Statement of the proof problem
theorem second_runner_stop_time :
  8 = distance_gap * pace_first :=
by
sorry

end second_runner_stop_time_l994_99430


namespace probability_spinner_lands_in_shaded_region_l994_99432

theorem probability_spinner_lands_in_shaded_region :
  let total_regions := 4
  let shaded_regions := 3
  (shaded_regions: ℝ) / total_regions = 3 / 4 :=
by
  let total_regions := 4
  let shaded_regions := 3
  sorry

end probability_spinner_lands_in_shaded_region_l994_99432


namespace linear_eq_zero_l994_99484

variables {a b c d x y : ℝ}

theorem linear_eq_zero (h1 : a * x + b * y = 0) (h2 : c * x + d * y = 0) (h3 : a * d - c * b ≠ 0) :
  x = 0 ∧ y = 0 :=
by
  sorry

end linear_eq_zero_l994_99484


namespace quadratic_complete_square_l994_99426

theorem quadratic_complete_square (x m n : ℝ) 
  (h : 9 * x^2 - 36 * x - 81 = 0) :
  (x + m)^2 = n ∧ m + n = 11 :=
sorry

end quadratic_complete_square_l994_99426


namespace t_shaped_grid_sum_l994_99413

open Finset

theorem t_shaped_grid_sum :
  ∃ (a b c d e : ℕ), 
    a ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Finset ℕ) ∧
    b ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Finset ℕ) ∧
    c ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Finset ℕ) ∧
    d ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Finset ℕ) ∧
    e ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Finset ℕ) ∧
    (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (a ≠ e) ∧
    (b ≠ c) ∧ (b ≠ d) ∧ (b ≠ e) ∧
    (c ≠ d) ∧ (c ≠ e) ∧
    (d ≠ e) ∧
    a + b + c = 20 ∧
    d + e = 7 ∧
    (a + b + c + d + e + b) = 33 :=
sorry

end t_shaped_grid_sum_l994_99413


namespace older_brother_catches_younger_brother_l994_99400

theorem older_brother_catches_younger_brother
  (y_time_reach_school o_time_reach_school : ℕ) 
  (delay : ℕ) 
  (catchup_time : ℕ) 
  (h1 : y_time_reach_school = 25) 
  (h2 : o_time_reach_school = 15) 
  (h3 : delay = 8) 
  (h4 : catchup_time = 17):
  catchup_time = delay + ((8 * y_time_reach_school) / (o_time_reach_school - y_time_reach_school) * (y_time_reach_school / 25)) :=
by
  sorry

end older_brother_catches_younger_brother_l994_99400


namespace prove_k_eq_5_l994_99414

variable (a b k : ℕ)

theorem prove_k_eq_5 (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : (a^2 - 1 - b^2) / (a * b - 1) = k) : k = 5 :=
sorry

end prove_k_eq_5_l994_99414


namespace squirrels_more_than_nuts_l994_99459

theorem squirrels_more_than_nuts 
  (squirrels : ℕ) 
  (nuts : ℕ) 
  (h_squirrels : squirrels = 4) 
  (h_nuts : nuts = 2) 
  : squirrels - nuts = 2 :=
by
  sorry

end squirrels_more_than_nuts_l994_99459


namespace inequality_proof_l994_99433

theorem inequality_proof (x y : ℝ) (h1 : x^2 ≥ y) (h2 : y^2 ≥ x) : 
  (x / (y^2 + 1) + y / (x^2 + 1) ≤ 1) :=
sorry

end inequality_proof_l994_99433


namespace certain_fraction_is_half_l994_99402

theorem certain_fraction_is_half (n : ℕ) (fraction : ℚ) (h : (37 + 1/2) / fraction = 75) : fraction = 1/2 :=
by
    sorry

end certain_fraction_is_half_l994_99402


namespace two_zeros_range_l994_99415

noncomputable def f (x k : ℝ) : ℝ := x * Real.exp x - k

theorem two_zeros_range (k : ℝ) : -1 / Real.exp 1 < k ∧ k < 0 → ∃! x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 k = 0 ∧ f x2 k = 0 :=
by
  sorry

end two_zeros_range_l994_99415


namespace midpoint_plane_distance_l994_99422

noncomputable def midpoint_distance (A B : ℝ) (dA dB : ℝ) : ℝ :=
  (dA + dB) / 2

theorem midpoint_plane_distance (A B : ℝ) (dA dB : ℝ) (hA : dA = 1) (hB : dB = 3) :
  midpoint_distance A B dA dB = 1 ∨ midpoint_distance A B dA dB = 2 :=
by
  sorry

end midpoint_plane_distance_l994_99422


namespace common_tangent_x_eq_neg1_l994_99449
open Real

-- Definitions of circles C₁ and C₂
def circle1 := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}
def circle2 := {p : ℝ × ℝ | (p.1 - 3)^2 + (p.2 - 4)^2 = 16}

-- Statement of the problem
theorem common_tangent_x_eq_neg1 :
  ∀ (x : ℝ) (y : ℝ),
    (x, y) ∈ circle1 ∧ (x, y) ∈ circle2 → x = -1 :=
sorry

end common_tangent_x_eq_neg1_l994_99449


namespace vertex_parabola_is_parabola_l994_99420

variables {a c : ℝ} (h_a : 0 < a) (h_c : 0 < c)

theorem vertex_parabola_is_parabola :
  ∀ (x y : ℝ), (∃ b : ℝ, x = -b / (2 * a) ∧ y = a * (-b / (2 * a)) ^ 2 + b * (-b / (2 * a)) + c) ↔ y = -a * x ^ 2 + c :=
by sorry

end vertex_parabola_is_parabola_l994_99420


namespace remainder_of_a55_l994_99408

def concatenate_integers (n : ℕ) : ℕ :=
  -- Function to concatenate integers from 1 to n into a single number.
  -- This is a placeholder, actual implementation may vary.
  sorry

theorem remainder_of_a55 (n : ℕ) (hn : n = 55) :
  concatenate_integers n % 55 = 0 := by
  -- Proof is omitted, provided as a guideline.
  sorry

end remainder_of_a55_l994_99408


namespace combined_transform_is_correct_l994_99492

def dilation_matrix (k : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![k, 0; 0, k]

def reflection_x_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  !![1, 0; 0, -1]

def combined_transform (dilation_factor : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  dilation_matrix dilation_factor * reflection_x_matrix

theorem combined_transform_is_correct :
  combined_transform 5 = !![5, 0; 0, -5] :=
by
  sorry

end combined_transform_is_correct_l994_99492


namespace tens_digit_less_than_5_probability_l994_99405

theorem tens_digit_less_than_5_probability 
  (n : ℕ) 
  (hn : 10000 ≤ n ∧ n ≤ 99999)
  (h_even : ∃ k, n % 10 = 2 * k ∧ k < 5) :
  (∃ p, 0 ≤ p ∧ p ≤ 1 ∧ p = 1 / 2) :=
by
  sorry

end tens_digit_less_than_5_probability_l994_99405


namespace condition_on_a_and_b_l994_99421

variable (x a b : ℝ)

def f (x : ℝ) : ℝ := x^2 - 4*x + 3

theorem condition_on_a_and_b
  (h1 : a > 0)
  (h2 : b > 0) :
  (∀ x : ℝ, |f x + 3| < a ↔ |x - 1| < b) ↔ (b^2 + 2*b + 3 ≤ a) :=
sorry

end condition_on_a_and_b_l994_99421


namespace modulus_of_z_l994_99466

open Complex

theorem modulus_of_z (z : ℂ) (h : (1 - I) * z = 2 + 2 * I) : abs z = 2 := 
sorry

end modulus_of_z_l994_99466


namespace intersection_of_sets_l994_99481

-- Define the sets A and B
def A : Set ℝ := {x | |x| ≤ 2}
def B : Set ℝ := {x | 3 * x - 2 ≥ 1}

-- Prove that A ∩ B = {x | 1 ≤ x ∧ x ≤ 2}
theorem intersection_of_sets : A ∩ B = {x | 1 ≤ x ∧ x ≤ 2} :=
by sorry

end intersection_of_sets_l994_99481


namespace linear_equation_value_m_l994_99478

theorem linear_equation_value_m (m : ℝ) (h : ∀ x : ℝ, 2 * x^(m - 1) + 3 = 0 → x ≠ 0) : m = 2 :=
sorry

end linear_equation_value_m_l994_99478


namespace vasya_numbers_l994_99425

theorem vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) : (x = 1 / 2 ∧ y = -1) ∨ (x = -1 ∧ y = 1 / 2) :=
by sorry

end vasya_numbers_l994_99425


namespace find_alpha_l994_99453

theorem find_alpha (α : ℝ) (h1 : α ∈ Set.Ioo (Real.pi / 2) (3 * Real.pi / 2))
  (h2 : ∃ k : ℝ, (Real.cos α, Real.sin α) = k • (-3, -3)) :
  α = 3 * Real.pi / 4 :=
by
  sorry

end find_alpha_l994_99453


namespace gingerbreads_per_tray_l994_99489

theorem gingerbreads_per_tray (x : ℕ) (h : 4 * x + 3 * 20 = 160) : x = 25 :=
by
  sorry

end gingerbreads_per_tray_l994_99489


namespace XiaoZhang_four_vcd_probability_l994_99497

noncomputable def probability_four_vcd (zhang_vcd zhang_dvd wang_vcd wang_dvd : ℕ) : ℚ :=
  (4 * 2 / (7 * 3)) + (3 * 1 / (7 * 3))

theorem XiaoZhang_four_vcd_probability :
  probability_four_vcd 4 3 2 1 = 11 / 21 :=
by
  sorry

end XiaoZhang_four_vcd_probability_l994_99497


namespace unique_triple_gcd_square_l994_99493

theorem unique_triple_gcd_square (m n l : ℕ) (H1 : m + n = Nat.gcd m n ^ 2)
                                  (H2 : m + l = Nat.gcd m l ^ 2)
                                  (H3 : n + l = Nat.gcd n l ^ 2) : (m, n, l) = (2, 2, 2) :=
by
  sorry

end unique_triple_gcd_square_l994_99493


namespace greatest_two_digit_prod_12_l994_99456

theorem greatest_two_digit_prod_12 : ∃(n : ℕ), n < 100 ∧ n ≥ 10 ∧
  (∃(d1 d2 : ℕ), n = 10 * d1 + d2 ∧ d1 * d2 = 12) ∧ ∀(k : ℕ), k < 100 ∧ k ≥ 10 ∧ (∃(d1 d2 : ℕ), k = 10 * d1 + d2 ∧ d1 * d2 = 12) → k ≤ 62 :=
by
  sorry

end greatest_two_digit_prod_12_l994_99456


namespace equation_represents_two_intersecting_lines_l994_99499

theorem equation_represents_two_intersecting_lines :
  (∀ x y : ℝ, x^3 * (x + y - 2) = y^3 * (x + y - 2) ↔
    (x = y ∨ y = 2 - x)) :=
by sorry

end equation_represents_two_intersecting_lines_l994_99499


namespace distance_from_circumcenter_to_orthocenter_l994_99423

variables {A B C A1 H O : Type}

-- Condition Definitions
variable (acute_triangle : Prop)
variable (is_altitude : Prop)
variable (is_orthocenter : Prop)
variable (AH_dist : ℝ := 3)
variable (A1H_dist : ℝ := 2)
variable (circum_radius : ℝ := 4)

-- Prove the distance from O to H
theorem distance_from_circumcenter_to_orthocenter
  (h1 : acute_triangle)
  (h2 : is_altitude)
  (h3 : is_orthocenter)
  (h4 : AH_dist = 3)
  (h5 : A1H_dist = 2)
  (h6 : circum_radius = 4) : 
  ∃ (d : ℝ), d = 2 := 
sorry

end distance_from_circumcenter_to_orthocenter_l994_99423


namespace part1_part2_part3_l994_99458

noncomputable def f (x m : ℝ) : ℝ :=
  -x^2 + m*x - m

-- Part (1)
theorem part1 (m : ℝ) : (∀ x, f x m ≤ 0) → (m = 0 ∨ m = 4) :=
sorry

-- Part (2)
theorem part2 (m : ℝ) : (∀ x, -1 ≤ x ∧ x ≤ 0 → f x m ≤ f (-1) m) → (m ≤ -2) :=
sorry

-- Part (3)
theorem part3 : ∃ (m : ℝ), (∀ x, 2 ≤ x ∧ x ≤ 3 → (2 ≤ f x m ∧ f x m ≤ 3)) → m = 6 :=
sorry

end part1_part2_part3_l994_99458


namespace quadratic_solution_l994_99401

-- Definitions come from the conditions of the problem
def satisfies_equation (y : ℝ) : Prop := 6 * y^2 + 2 = 4 * y + 12

-- Statement of the proof
theorem quadratic_solution (y : ℝ) (hy : satisfies_equation y) : (12 * y - 2)^2 = 324 ∨ (12 * y - 2)^2 = 196 := 
sorry

end quadratic_solution_l994_99401


namespace find_m_l994_99470

open Real

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

theorem find_m :
  let a := (-sqrt 3, m)
  let b := (2, 1)
  (dot_product a b = 0) → m = 2 * sqrt 3 :=
by
  sorry

end find_m_l994_99470


namespace difference_between_max_and_min_area_l994_99472

noncomputable def max_area (l w : ℕ) : ℕ :=
  if 2 * l + 2 * w = 60 then l * w else 0

noncomputable def min_area (l w : ℕ) : ℕ :=
  if 2 * l + 2 * w = 60 then l * w else 0

theorem difference_between_max_and_min_area :
  ∃ (l_max l_min w_max w_min : ℕ),
    2 * l_max + 2 * w_max = 60 ∧
    2 * l_min + 2 * w_min = 60 ∧
    (l_max * w_max - l_min * w_min = 196) :=
by
  sorry

end difference_between_max_and_min_area_l994_99472


namespace four_digit_integer_l994_99416

theorem four_digit_integer (a b c d : ℕ) (h1 : a + b + c + d = 18)
  (h2 : b + c = 11) (h3 : a - d = 1) (h4 : 11 ∣ (1000 * a + 100 * b + 10 * c + d)) :
  1000 * a + 100 * b + 10 * c + d = 4653 :=
by sorry

end four_digit_integer_l994_99416


namespace prob_second_shot_l994_99491

theorem prob_second_shot (P_A : ℝ) (P_AB : ℝ) (p : ℝ) : 
  P_A = 0.75 → 
  P_AB = 0.6 → 
  P_A * p = P_AB → 
  p = 0.8 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  sorry

end prob_second_shot_l994_99491


namespace problem1_solution_set_problem2_range_of_a_l994_99437

section Problem1

def f1 (x : ℝ) : ℝ := |x - 4| + |x - 2|

theorem problem1_solution_set (a : ℝ) (h : a = 2) :
  { x : ℝ | f1 x > 10 } = { x : ℝ | x > 8 ∨ x < -2 } := sorry

end Problem1


section Problem2

def f2 (x a : ℝ) : ℝ := |x - 4| + |x - a|

theorem problem2_range_of_a (f_geq : ∀ x : ℝ, f2 x a ≥ 1) :
  a ≥ 5 ∨ a ≤ 3 := sorry

end Problem2

end problem1_solution_set_problem2_range_of_a_l994_99437


namespace complex_number_solution_l994_99446

open Complex

noncomputable def findComplexNumber (z : ℂ) : Prop :=
  abs (z - 2) = abs (z + 4) ∧ abs (z - 2) = abs (z - Complex.I * 2)

theorem complex_number_solution :
  ∃ z : ℂ, findComplexNumber z ∧ z = -1 + Complex.I :=
by
  sorry

end complex_number_solution_l994_99446


namespace probability_of_same_color_pairs_left_right_l994_99429

-- Define the counts of different pairs
def total_pairs := 15
def black_pairs := 8
def red_pairs := 4
def white_pairs := 3

-- Define the total number of shoes
def total_shoes := 30

-- Define the total ways to choose any 2 shoes out of total_shoes
def total_ways := Nat.choose total_shoes 2

-- Define the ways to choose one left and one right for each color
def black_ways := black_pairs * black_pairs
def red_ways := red_pairs * red_pairs
def white_ways := white_pairs * white_pairs

-- Define the total favorable outcomes for same color pairs
def total_favorable := black_ways + red_ways + white_ways

-- Define the probability
def probability := (total_favorable, total_ways)

-- Statement to prove
theorem probability_of_same_color_pairs_left_right :
  probability = (89, 435) :=
by
  sorry

end probability_of_same_color_pairs_left_right_l994_99429


namespace codys_grandmother_age_l994_99480

theorem codys_grandmother_age (cody_age : ℕ) (grandmother_factor : ℕ) (h1 : cody_age = 14) (h2 : grandmother_factor = 6) :
  grandmother_factor * cody_age = 84 :=
by
  sorry

end codys_grandmother_age_l994_99480


namespace train_speed_l994_99439

theorem train_speed (L V : ℝ) (h1 : L = V * 10) (h2 : L + 500 = V * 35) : V = 20 :=
by {
  -- Proof is skipped with sorry
  sorry
}

end train_speed_l994_99439


namespace expression_varies_l994_99460

variables {x : ℝ}

noncomputable def expression (x : ℝ) : ℝ :=
  (3 * x^2 + 4 * x - 5) / ((x + 1) * (x - 3)) - (8 + x) / ((x + 1) * (x - 3))

theorem expression_varies (h1 : x ≠ -1) (h2 : x ≠ 3) : 
  ∃ x₀ x₁ : ℝ, x₀ ≠ x₁ ∧ 
  expression x₀ ≠ expression x₁ :=
by
  sorry

end expression_varies_l994_99460


namespace red_paint_cans_l994_99410

theorem red_paint_cans (total_cans : ℕ) (ratio_red_blue : ℕ) (ratio_blue : ℕ) (h_ratio : ratio_red_blue = 4) (h_blue : ratio_blue = 1) (h_total_cans : total_cans = 50) : 
  (total_cans * ratio_red_blue) / (ratio_red_blue + ratio_blue) = 40 :=
by {
  -- Proof steps would go here
  sorry
}

end red_paint_cans_l994_99410


namespace units_digit_7_power_2023_l994_99498

theorem units_digit_7_power_2023 : ∃ d, d = 7^2023 % 10 ∧ d = 3 := 
by 
  sorry

end units_digit_7_power_2023_l994_99498


namespace lines_intersect_l994_99488

variables {s v : ℝ}

def line1 (s : ℝ) : ℝ × ℝ :=
  (3 - 2 * s, 4 + 3 * s)

def line2 (v : ℝ) : ℝ × ℝ :=
  (1 - 3 * v, 5 + 2 * v)

theorem lines_intersect :
  ∃ s v : ℝ, line1 s = line2 v ∧ line1 s = (25 / 13, 73 / 13) :=
by
  sorry

end lines_intersect_l994_99488


namespace roger_daily_goal_l994_99464

-- Conditions
def steps_in_30_minutes : ℕ := 2000
def time_to_reach_goal_min : ℕ := 150
def time_interval_min : ℕ := 30

-- Theorem to prove
theorem roger_daily_goal : steps_in_30_minutes * (time_to_reach_goal_min / time_interval_min) = 10000 := by
  sorry

end roger_daily_goal_l994_99464


namespace quadratic_real_roots_iff_range_k_quadratic_real_roots_specific_value_k_l994_99450

theorem quadratic_real_roots_iff_range_k (k : ℝ) :
  (∃ x1 x2 : ℝ, x1^2 - 4 * x1 + k + 1 = 0 ∧ x2^2 - 4 * x2 + k + 1 = 0 ∧ x1 ≠ x2) ↔ k ≤ 3 :=
by
  sorry

theorem quadratic_real_roots_specific_value_k (k : ℝ) (x1 x2 : ℝ) :
  x1^2 - 4 * x1 + k + 1 = 0 →
  x2^2 - 4 * x2 + k + 1 = 0 →
  x1 ≠ x2 →
  (3 / x1 + 3 / x2 = x1 * x2 - 4) →
  k = -3 :=
by
  sorry

end quadratic_real_roots_iff_range_k_quadratic_real_roots_specific_value_k_l994_99450


namespace janet_earnings_eur_l994_99440

noncomputable def usd_to_eur (usd : ℚ) : ℚ :=
  usd * 0.85

def janet_earnings_usd : ℚ :=
  (130 * 0.25) + (90 * 0.30) + (30 * 0.40)

theorem janet_earnings_eur : usd_to_eur janet_earnings_usd = 60.78 :=
  by
    sorry

end janet_earnings_eur_l994_99440


namespace plane_equation_l994_99431

noncomputable def equation_of_plane (x y z : ℝ) :=
  3 * x + 2 * z - 1

theorem plane_equation :
  ∀ (x y z : ℝ), 
    (∃ (p : ℝ × ℝ × ℝ), p = (1, 2, -1) ∧ 
                         (∃ (n : ℝ × ℝ × ℝ), n = (3, 0, 2) ∧ 
                                              equation_of_plane x y z = 0)) :=
by
  -- The statement setup is done. The proof is not included as per instructions.
  sorry

end plane_equation_l994_99431


namespace sequence_solution_l994_99424

theorem sequence_solution (a : ℕ → ℤ) :
  a 0 = -1 →
  a 1 = 1 →
  (∀ n ≥ 2, a n = 2 * a (n - 1) + 3 * a (n - 2) + 3^n) →
  ∀ n, a n = (1 / 16) * ((4 * n - 3) * 3^(n + 1) - 7 * (-1)^n) :=
by
  -- Detailed proof steps will go here.
  sorry

end sequence_solution_l994_99424


namespace basketball_team_lineup_l994_99428

-- Define the problem conditions
def total_players : ℕ := 12
def twins : ℕ := 2
def lineup_size : ℕ := 5
def remaining_players : ℕ := total_players - twins
def positions_to_fill : ℕ := lineup_size - twins

-- Define the combination function as provided in the standard libraries
def combination (n k : ℕ) : ℕ :=
  if k > n then 0 else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- State the theorem translating to the proof problem
theorem basketball_team_lineup : combination remaining_players positions_to_fill = 120 := 
sorry

end basketball_team_lineup_l994_99428


namespace john_and_mike_safe_weight_l994_99465

def weight_bench_max_support : ℕ := 1000
def safety_margin_percentage : ℕ := 20
def john_weight : ℕ := 250
def mike_weight : ℕ := 180

def safety_margin : ℕ := (safety_margin_percentage * weight_bench_max_support) / 100
def max_safe_weight : ℕ := weight_bench_max_support - safety_margin
def combined_weight : ℕ := john_weight + mike_weight
def weight_on_bar_together : ℕ := max_safe_weight - combined_weight

theorem john_and_mike_safe_weight :
  weight_on_bar_together = 370 := by
  sorry

end john_and_mike_safe_weight_l994_99465


namespace certain_number_proof_l994_99457

noncomputable def certain_number : ℝ := 30

theorem certain_number_proof (h1: 0.60 * 50 = 30) (h2: 30 = 0.40 * certain_number + 18) : 
  certain_number = 30 := 
sorry

end certain_number_proof_l994_99457


namespace max_value_of_f_l994_99443

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem max_value_of_f : ∃ x_max : ℝ, (∀ x : ℝ, f x ≤ f x_max) ∧ f (Real.exp 1) = 1 / Real.exp 1 := by
  sorry

end max_value_of_f_l994_99443


namespace rhombus_perimeter_l994_99412

theorem rhombus_perimeter (d1 d2 : ℕ) (h1 : d1 = 16) (h2 : d2 = 30) :
  4 * (Nat.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)) = 68 :=
by
  sorry

end rhombus_perimeter_l994_99412


namespace system_of_equations_solution_l994_99427

theorem system_of_equations_solution (x y z : ℝ) (hx : x = Real.exp (Real.log y))
(hy : y = Real.exp (Real.log z)) (hz : z = Real.exp (Real.log x)) : x = y ∧ y = z ∧ z = x ∧ x = Real.exp 1 :=
by
  sorry

end system_of_equations_solution_l994_99427


namespace unique_solution_l994_99445

def satisfies_equation (m n : ℕ) : Prop :=
  15 * m * n = 75 - 5 * m - 3 * n

theorem unique_solution : satisfies_equation 1 6 ∧ ∀ (m n : ℕ), m > 0 → n > 0 → satisfies_equation m n → (m, n) = (1, 6) :=
by {
  sorry
}

end unique_solution_l994_99445


namespace frisbee_sales_l994_99438

/-- A sporting goods store sold some frisbees, with $3 and $4 price points.
The total receipts from frisbee sales were $204. The fewest number of $4 frisbees that could have been sold is 24.
Prove the total number of frisbees sold is 60. -/
theorem frisbee_sales (x y : ℕ) (h1 : 3 * x + 4 * y = 204) (h2 : 24 ≤ y) : x + y = 60 :=
by {
  -- Proof skipped
  sorry
}

end frisbee_sales_l994_99438


namespace polynomial_mult_of_6_l994_99441

theorem polynomial_mult_of_6 (P : Polynomial ℤ)
  (h1 : 6 ∣ P.eval 2)
  (h2 : 6 ∣ P.eval 3) : 6 ∣ P.eval 5 := 
sorry

end polynomial_mult_of_6_l994_99441


namespace expected_number_of_shots_l994_99417

def probability_hit : ℝ := 0.8
def probability_miss := 1 - probability_hit
def max_shots : ℕ := 3

theorem expected_number_of_shots : ∃ ξ : ℝ, ξ = 1.24 := by
  sorry

end expected_number_of_shots_l994_99417


namespace min_sum_squares_l994_99490

variable {a b c t : ℝ}

def min_value_of_sum_squares (a b c : ℝ) (t : ℝ) : ℝ :=
  a^2 + b^2 + c^2

theorem min_sum_squares (h : a + b + c = t) : min_value_of_sum_squares a b c t ≥ t^2 / 3 :=
by
  sorry

end min_sum_squares_l994_99490


namespace cloth_cost_l994_99452

theorem cloth_cost
  (L : ℕ)
  (C : ℚ)
  (hL : L = 10)
  (h_condition : L * C = (L + 4) * (C - 1)) :
  10 * C = 35 := by
  sorry

end cloth_cost_l994_99452


namespace complete_square_variant_l994_99418

theorem complete_square_variant (x : ℝ) :
    3 * x^2 + 4 * x + 1 = 0 → (x + 2 / 3) ^ 2 = 1 / 9 :=
by
  intro h
  sorry

end complete_square_variant_l994_99418


namespace divisible_by_five_l994_99495

theorem divisible_by_five (a : ℤ) (h : ¬ (5 ∣ a)) :
    (5 ∣ (a^2 - 1)) ↔ ¬ (5 ∣ (a^2 + 1)) :=
by
  -- Begin the proof here (proof not required according to instructions)
  sorry

end divisible_by_five_l994_99495


namespace correct_calculation_l994_99419

theorem correct_calculation (a : ℝ) :
  (¬ (a^2 + a^2 = a^4)) ∧ (¬ (a^2 * a^3 = a^6)) ∧ (¬ ((a + 1)^2 = a^2 + 1)) ∧ ((-a^2)^2 = a^4) :=
by
  sorry

end correct_calculation_l994_99419


namespace inequality_pos_distinct_l994_99409

theorem inequality_pos_distinct (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
    (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) :
    (a + b + c) * (1/a + 1/b + 1/c) > 9 := by
  sorry

end inequality_pos_distinct_l994_99409


namespace pyramid_surface_area_l994_99434

noncomputable def total_surface_area : Real :=
  let ab := 14
  let bc := 8
  let pf := 15
  let base_area := ab * bc
  let fm := ab / 2
  let pm_ab := Real.sqrt (pf^2 + fm^2)
  let pm_bc := Real.sqrt (pf^2 + (bc / 2)^2)
  base_area + 2 * (ab / 2 * pm_ab) + 2 * (bc / 2 * pm_bc)

theorem pyramid_surface_area :
  total_surface_area = 112 + 14 * Real.sqrt 274 + 8 * Real.sqrt 241 := by
  sorry

end pyramid_surface_area_l994_99434


namespace max_xy_value_l994_99463

theorem max_xy_value (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 2 * x + y = 2) : xy ≤ 1 / 2 := 
by
  sorry

end max_xy_value_l994_99463


namespace total_cost_l994_99477

theorem total_cost
  (permits_cost : ℕ)
  (contractor_hourly_rate : ℕ)
  (contractor_days : ℕ)
  (contractor_hours_per_day : ℕ)
  (inspector_discount : ℕ)
  (h_pc : permits_cost = 250)
  (h_chr : contractor_hourly_rate = 150)
  (h_cd : contractor_days = 3)
  (h_chpd : contractor_hours_per_day = 5)
  (h_id : inspector_discount = 80)
  (contractor_total_hours : ℕ := contractor_days * contractor_hours_per_day)
  (contractor_total_cost : ℕ := contractor_total_hours * contractor_hourly_rate)
  (inspector_cost : ℕ := contractor_total_cost - (contractor_total_cost * inspector_discount / 100))
  (total_cost : ℕ := permits_cost + contractor_total_cost + inspector_cost) :
  total_cost = 2950 :=
by
  sorry

end total_cost_l994_99477


namespace tank_cost_correct_l994_99404

noncomputable def tankPlasteringCost (l w d cost_per_m2 : ℝ) : ℝ :=
  let long_walls_area := 2 * (l * d)
  let short_walls_area := 2 * (w * d)
  let bottom_area := l * w
  let total_area := long_walls_area + short_walls_area + bottom_area
  total_area * cost_per_m2

theorem tank_cost_correct :
  tankPlasteringCost 25 12 6 0.75 = 558 := by
  sorry

end tank_cost_correct_l994_99404


namespace evaluate_expression_l994_99411

def x : ℝ := 2
def y : ℝ := 4

theorem evaluate_expression : y * (y - 2 * x) = 0 := by
  sorry

end evaluate_expression_l994_99411


namespace all_options_valid_l994_99496

-- Definition of the line equation
def line_eq (x y : ℝ) : Prop := y = 2 * x - 4

-- Definitions of parameterizations for each option
def option_A (t : ℝ) : ℝ × ℝ := ⟨2 + (-1) * t, 0 + (-2) * t⟩
def option_B (t : ℝ) : ℝ × ℝ := ⟨6 + 4 * t, 8 + 8 * t⟩
def option_C (t : ℝ) : ℝ × ℝ := ⟨1 + 1 * t, -2 + 2 * t⟩
def option_D (t : ℝ) : ℝ × ℝ := ⟨0 + 0.5 * t, -4 + 1 * t⟩
def option_E (t : ℝ) : ℝ × ℝ := ⟨-2 + (-2) * t, -8 + (-4) * t⟩

-- The main statement to prove
theorem all_options_valid :
  (∀ t, line_eq (option_A t).1 (option_A t).2) ∧
  (∀ t, line_eq (option_B t).1 (option_B t).2) ∧
  (∀ t, line_eq (option_C t).1 (option_C t).2) ∧
  (∀ t, line_eq (option_D t).1 (option_D t).2) ∧
  (∀ t, line_eq (option_E t).1 (option_E t).2) :=
by sorry -- proof omitted

end all_options_valid_l994_99496


namespace original_average_score_of_class_l994_99468

theorem original_average_score_of_class {A : ℝ} 
  (num_students : ℝ) 
  (grace_marks : ℝ) 
  (new_average : ℝ) 
  (h1 : num_students = 35) 
  (h2 : grace_marks = 3) 
  (h3 : new_average = 40)
  (h_total_new : 35 * new_average = 35 * A + 35 * grace_marks) :
  A = 37 :=
by 
  -- Placeholder for proof
  sorry

end original_average_score_of_class_l994_99468


namespace man_buys_article_for_20_l994_99494

variable (SP : ℝ) (G : ℝ) (CP : ℝ)

theorem man_buys_article_for_20 (hSP : SP = 25) (hG : G = 0.25) (hEquation : SP = CP * (1 + G)) : CP = 20 :=
by
  sorry

end man_buys_article_for_20_l994_99494


namespace percentage_of_red_non_honda_cars_l994_99448

-- Define the conditions
def total_cars : ℕ := 900
def honda_cars : ℕ := 500
def red_per_100_honda_cars : ℕ := 90
def red_percent_total := 60

-- Define the question we want to answer
theorem percentage_of_red_non_honda_cars : 
  let red_honda_cars := (red_per_100_honda_cars / 100 : ℚ) * honda_cars
  let total_red_cars := (red_percent_total / 100 : ℚ) * total_cars
  let red_non_honda_cars := total_red_cars - red_honda_cars
  let non_honda_cars := total_cars - honda_cars
  (red_non_honda_cars / non_honda_cars) * 100 = (22.5 : ℚ) :=
by
  sorry

end percentage_of_red_non_honda_cars_l994_99448


namespace find_p_l994_99454

theorem find_p (p q : ℝ) (h1 : p + 2 * q = 1) (h2 : p > 0) (h3 : q > 0) (h4 : 10 * p^9 * q = 45 * p^8 * q^2): 
  p = 9 / 13 :=
by
  sorry

end find_p_l994_99454


namespace chord_length_of_intersection_l994_99469

theorem chord_length_of_intersection 
  (x y : ℝ) (h_line : 2 * x - y - 1 = 0) (h_circle : (x - 2)^2 + (y + 2)^2 = 9) : 
  ∃ l, l = 4 := 
sorry

end chord_length_of_intersection_l994_99469
