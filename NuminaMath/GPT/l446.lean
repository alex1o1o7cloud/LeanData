import Mathlib

namespace max_partitioned_test_plots_is_78_l446_44651

def field_length : ℕ := 52
def field_width : ℕ := 24
def total_fence : ℕ := 1994
def gcd_field_dimensions : ℕ := Nat.gcd field_length field_width

-- Since gcd_field_dimensions divides both 52 and 24 and gcd_field_dimensions = 4
def possible_side_lengths : List ℕ := [1, 2, 4]

noncomputable def max_square_plots : ℕ :=
  let max_plots (a : ℕ) : ℕ := (field_length / a) * (field_width / a)
  let valid_fence (a : ℕ) : Bool :=
    let vertical_fence := (field_length / a - 1) * field_width
    let horizontal_fence := (field_width / a - 1) * field_length
    vertical_fence + horizontal_fence ≤ total_fence
  let valid_lengths := possible_side_lengths.filter valid_fence
  valid_lengths.map max_plots |>.maximum? |>.getD 0

theorem max_partitioned_test_plots_is_78 : max_square_plots = 78 := by
  sorry

end max_partitioned_test_plots_is_78_l446_44651


namespace common_difference_divisible_by_p_l446_44637

variable (a : ℕ → ℕ) (p : ℕ)

-- Define that the sequence a is an arithmetic progression with common difference d
def is_arithmetic_progression (d : ℕ) : Prop :=
  ∀ i : ℕ, a (i + 1) = a i + d

-- Define that the sequence a is strictly increasing
def is_increasing_arithmetic_progression : Prop :=
  ∀ i j : ℕ, i < j → a i < a j

-- Define that all elements a_i are prime numbers
def all_primes : Prop :=
  ∀ i : ℕ, Nat.Prime (a i)

-- Define that the first element of the sequence is greater than p
def first_element_greater_than_p : Prop :=
  a 1 > p

-- Combining all conditions
def conditions (d : ℕ) : Prop :=
  is_arithmetic_progression a d ∧ is_increasing_arithmetic_progression a ∧ all_primes a ∧ first_element_greater_than_p a p ∧ Nat.Prime p

-- Statement to prove: common difference is divisible by p
theorem common_difference_divisible_by_p (d : ℕ) (h : conditions a p d) : p ∣ d :=
sorry

end common_difference_divisible_by_p_l446_44637


namespace box_volume_correct_l446_44664

-- Define the dimensions of the obelisk
def obelisk_height : ℕ := 15
def base_length : ℕ := 8
def base_width : ℕ := 10

-- Define the dimension and volume goal for the cube-shaped box
def box_side_length : ℕ := obelisk_height
def box_volume : ℕ := box_side_length ^ 3

-- The proof goal
theorem box_volume_correct : box_volume = 3375 := 
by sorry

end box_volume_correct_l446_44664


namespace books_added_l446_44684

theorem books_added (initial_books sold_books current_books added_books : ℕ)
  (h1 : initial_books = 4)
  (h2 : sold_books = 3)
  (h3 : current_books = 11)
  (h4 : added_books = current_books - (initial_books - sold_books)) :
  added_books = 10 :=
by
  rw [h1, h2, h3] at h4
  simp at h4
  exact h4

end books_added_l446_44684


namespace hogwarts_school_students_l446_44616

def total_students_at_school (participants boys : ℕ) (boy_participants girl_non_participants : ℕ) : Prop :=
  participants = 246 ∧ boys = 255 ∧ boy_participants = girl_non_participants + 11 → (boys + (participants - boy_participants + girl_non_participants)) = 490

theorem hogwarts_school_students : total_students_at_school 246 255 (boy_participants) girl_non_participants := 
 sorry

end hogwarts_school_students_l446_44616


namespace infant_weight_in_4th_month_l446_44695

-- Given conditions
def a : ℕ := 3000
def x : ℕ := 4
def y : ℕ := a + 700 * x

-- Theorem stating the weight of the infant in the 4th month equals 5800 grams
theorem infant_weight_in_4th_month : y = 5800 := by
  sorry

end infant_weight_in_4th_month_l446_44695


namespace find_x_l446_44680

def Hiram_age := 40
def Allyson_age := 28
def Twice_Allyson_age := 2 * Allyson_age
def Four_less_than_twice_Allyson_age := Twice_Allyson_age - 4

theorem find_x (x : ℤ) : Hiram_age + x = Four_less_than_twice_Allyson_age → x = 12 := 
by
  intros h -- introducing the assumption 
  sorry

end find_x_l446_44680


namespace parameter_a_values_l446_44687

theorem parameter_a_values (a : ℝ) :
  (∃ x y : ℝ, |x + y + 8| + |x - y + 8| = 16 ∧ ((|x| - 8)^2 + (|y| - 15)^2 = a) ∧
    (∀ x₁ y₁ x₂ y₂ : ℝ, |x₁ + y₁ + 8| + |x₁ - y₁ + 8| = 16 →
      (|x₁| - 8)^2 + (|y₁| - 15)^2 = a →
      |x₂ + y₂ + 8| + |x₂ - y₂ + 8| = 16 →
      (|x₂| - 8)^2 + (|y₂| - 15)^2 = a →
      (x₁, y₁) = (x₂, y₂) ∨ (x₁, y₁) = (y₂, x₂))) ↔ a = 49 ∨ a = 289 :=
by sorry

end parameter_a_values_l446_44687


namespace weight_loss_percentage_l446_44690

theorem weight_loss_percentage 
  (weight_before weight_after : ℝ) 
  (h_before : weight_before = 800) 
  (h_after : weight_after = 640) : 
  (weight_before - weight_after) / weight_before * 100 = 20 := 
by
  sorry

end weight_loss_percentage_l446_44690


namespace find_y_l446_44604

theorem find_y (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h1 : x = 2 + 1 / y) (h2 : y = 2 + 1 / x) :
  y = 1 + Real.sqrt 2 ∨ y = 1 - Real.sqrt 2 :=
by
  sorry

end find_y_l446_44604


namespace find_positive_integers_divisors_l446_44691

theorem find_positive_integers_divisors :
  ∃ n_list : List ℕ, 
    (∀ n ∈ n_list, n > 0 ∧ (n * (n + 1)) / 2 ∣ 10 * n) ∧ n_list.length = 5 :=
sorry

end find_positive_integers_divisors_l446_44691


namespace find_d1_over_d2_l446_44662

variables {k c1 c2 d1 d2 : ℝ}
variables (c1_nonzero : c1 ≠ 0) (c2_nonzero : c2 ≠ 0) 
variables (d1_nonzero : d1 ≠ 0) (d2_nonzero : d2 ≠ 0)
variables (h1 : c1 * d1 = k) (h2 : c2 * d2 = k)
variables (h3 : c1 / c2 = 3 / 4)

theorem find_d1_over_d2 : d1 / d2 = 4 / 3 :=
sorry

end find_d1_over_d2_l446_44662


namespace positive_integer_solutions_of_inequality_l446_44682

theorem positive_integer_solutions_of_inequality :
  {x : ℕ | 2 * (x - 1) < 7 - x ∧ x > 0} = {1, 2} :=
by
  sorry

end positive_integer_solutions_of_inequality_l446_44682


namespace probability_intersection_l446_44688

variables (A B : Type → Prop)

-- Assuming we have a measure space (probability) P
variables {P : Type → Prop}

-- Given probabilities
def p_A := 0.65
def p_B := 0.55
def p_Ac_Bc := 0.20

-- The theorem to be proven
theorem probability_intersection :
  (p_A + p_B - (1 - p_Ac_Bc) = 0.40) :=
by
  sorry

end probability_intersection_l446_44688


namespace tangent_line_intersects_y_axis_at_10_l446_44650

-- Define the curve y = x^2 + 11
def curve (x : ℝ) : ℝ := x^2 + 11

-- Define the derivative of the curve
def curve_derivative (x : ℝ) : ℝ := 2 * x

-- Define the point of tangency
def point_of_tangency : ℝ × ℝ := (1, 12)

-- Define the tangent line at point_of_tangency
def tangent_line (x : ℝ) : ℝ :=
  let slope := curve_derivative point_of_tangency.1
  let y_intercept := point_of_tangency.2 - slope * point_of_tangency.1
  slope * x + y_intercept

-- Theorem stating the y-coordinate of the intersection of the tangent line with the y-axis
theorem tangent_line_intersects_y_axis_at_10 :
  tangent_line 0 = 10 :=
by
  sorry

end tangent_line_intersects_y_axis_at_10_l446_44650


namespace car_new_speed_l446_44626

theorem car_new_speed (original_speed : ℝ) (supercharge_percent : ℝ) (weight_cut_speed_increase : ℝ) :
  original_speed = 150 → supercharge_percent = 0.30 → weight_cut_speed_increase = 10 → 
  original_speed * (1 + supercharge_percent) + weight_cut_speed_increase = 205 :=
by
  intros h_orig h_supercharge h_weight
  rw [h_orig, h_supercharge]
  sorry

end car_new_speed_l446_44626


namespace largest_packet_size_gcd_l446_44654

theorem largest_packet_size_gcd:
    ∀ (n1 n2 : ℕ), n1 = 36 → n2 = 60 → Nat.gcd n1 n2 = 12 :=
by
  intros n1 n2 h1 h2
  -- Sorry is added because the proof is not required as per the instructions
  sorry

end largest_packet_size_gcd_l446_44654


namespace sequence_is_arithmetic_l446_44639

theorem sequence_is_arithmetic 
  (a_n : ℕ → ℤ) 
  (h : ∀ n : ℕ, a_n n = n + 1) 
  : ∀ n : ℕ, a_n (n + 1) - a_n n = 1 :=
by
  sorry

end sequence_is_arithmetic_l446_44639


namespace min_disks_required_l446_44612

def num_files : ℕ := 35
def disk_size : ℕ := 2
def file_size_0_9 : ℕ := 4
def file_size_0_8 : ℕ := 15
def file_size_0_5 : ℕ := num_files - file_size_0_9 - file_size_0_8

-- Prove the minimum number of disks required to store all files.
theorem min_disks_required 
  (n : ℕ) 
  (disk_storage : ℕ)
  (num_files_0_9 : ℕ)
  (num_files_0_8 : ℕ)
  (num_files_0_5 : ℕ) :
  n = num_files → disk_storage = disk_size → num_files_0_9 = file_size_0_9 → num_files_0_8 = file_size_0_8 → num_files_0_5 = file_size_0_5 → 
  ∃ (d : ℕ), d = 15 :=
by 
  intros H1 H2 H3 H4 H5
  sorry

end min_disks_required_l446_44612


namespace commute_distance_l446_44672

theorem commute_distance (D : ℝ)
  (h1 : ∀ t : ℝ, t > 0 → t = D / 45)
  (h2 : ∀ t : ℝ, t > 0 → t = D / 30)
  (h3 : D / 45 + D / 30 = 1) :
  D = 18 :=
by
  sorry

end commute_distance_l446_44672


namespace solve_inequality_l446_44628

theorem solve_inequality (x : ℝ) : 3 * x^2 + 2 * x - 3 > 12 - 2 * x → x < -3 ∨ x > 5 / 3 :=
sorry

end solve_inequality_l446_44628


namespace linear_equation_unique_l446_44608

theorem linear_equation_unique (x y : ℝ) : 
  (3 * x = 2 * y) ∧ 
  ¬(3 * x - 6 = x) ∧ 
  ¬(x - 1 / y = 0) ∧ 
  ¬(2 * x - 3 * y = x * y) :=
by
  sorry

end linear_equation_unique_l446_44608


namespace m_condition_sufficient_not_necessary_l446_44655

-- Define the function f(x) and its properties
def f (m : ℝ) (x : ℝ) : ℝ := abs (x * (m * x + 2))

-- Define the condition for the function being increasing on (0, ∞)
def is_increasing_on_positives (m : ℝ) :=
  ∀ x y : ℝ, 0 < x → x < y → f m x < f m y

-- Prove that if m > 0, then the function is increasing on (0, ∞)
lemma m_gt_0_sufficient (m : ℝ) (h : 0 < m) : is_increasing_on_positives m :=
sorry

-- Show that the condition is indeed sufficient but not necessary
theorem m_condition_sufficient_not_necessary :
  ∀ m : ℝ, (0 < m → is_increasing_on_positives m) ∧ (is_increasing_on_positives m → 0 < m) :=
sorry

end m_condition_sufficient_not_necessary_l446_44655


namespace min_value_f_l446_44670

noncomputable def f (x : ℝ) : ℝ := Real.sin x + 1/2 * Real.cos (2 * x) - 1

theorem min_value_f : ∃ x : ℝ, f x = -5/2 := sorry

end min_value_f_l446_44670


namespace parabola_symmetric_points_l446_44627

theorem parabola_symmetric_points (a : ℝ) (h : 0 < a) :
  (∃ (P Q : ℝ × ℝ), (P ≠ Q) ∧ ((P.fst + P.snd = 0) ∧ (Q.fst + Q.snd = 0)) ∧
    (P.snd = a * P.fst ^ 2 - 1) ∧ (Q.snd = a * Q.fst ^ 2 - 1)) ↔ (3 / 4 < a) := 
sorry

end parabola_symmetric_points_l446_44627


namespace value_of_a_l446_44685

theorem value_of_a (a : ℕ) (h : ∀ x, ((a - 2) * x > a - 2) ↔ (x < 1)) : a = 0 ∨ a = 1 := by
  sorry

end value_of_a_l446_44685


namespace find_a_l446_44646

theorem find_a (x1 x2 x3 x4 x5 x6 x7 : ℝ)
  (h1 : x1 = 180)
  (h2 : x2 = 182)
  (h3 : x3 = 173)
  (h4 : x4 = 175)
  (h6 : x6 = 178)
  (h7 : x7 = 176)
  (h_avg : (x1 + x2 + x3 + x4 + x5 + x6 + x7) / 7 = 178) : x5 = 182 := by
  sorry

end find_a_l446_44646


namespace workers_together_time_l446_44697

theorem workers_together_time (hA : ℝ) (hB : ℝ) (jobA_time : hA = 10) (jobB_time : hB = 12) : 
  1 / ((1 / hA) + (1 / hB)) = (60 / 11) :=
by
  -- skipping the proof details
  sorry

end workers_together_time_l446_44697


namespace root_intervals_l446_44663

noncomputable def f (a b c x : ℝ) : ℝ := (x - a) * (x - b) + (x - b) * (x - c) + (x - c) * (x - a)

theorem root_intervals (a b c : ℝ) (h : a < b ∧ b < c) :
  ∃ r1 r2 : ℝ, (a < r1 ∧ r1 < b ∧ f a b c r1 = 0) ∧ (b < r2 ∧ r2 < c ∧ f a b c r2 = 0) :=
sorry

end root_intervals_l446_44663


namespace soup_problem_l446_44625

def cans_needed_for_children (children : ℕ) (children_per_can : ℕ) : ℕ :=
  children / children_per_can

def remaining_cans (initial_cans used_cans : ℕ) : ℕ :=
  initial_cans - used_cans

def half_cans (cans : ℕ) : ℕ :=
  cans / 2

def adults_fed (cans : ℕ) (adults_per_can : ℕ) : ℕ :=
  cans * adults_per_can

theorem soup_problem
  (initial_cans : ℕ)
  (children_fed : ℕ)
  (children_per_can : ℕ)
  (adults_per_can : ℕ)
  (reserved_fraction : ℕ)
  (hreserved : reserved_fraction = 2)
  (hintial : initial_cans = 8)
  (hchildren : children_fed = 24)
  (hchildren_per_can : children_per_can = 6)
  (hadults_per_can : adults_per_can = 4) :
  adults_fed (half_cans (remaining_cans initial_cans (cans_needed_for_children children_fed children_per_can))) adults_per_can = 8 :=
by
  sorry

end soup_problem_l446_44625


namespace find_m_l446_44640

theorem find_m (a0 a1 a2 a3 a4 a5 a6 m : ℝ) 
  (h1 : (1 + m) ^ 6 = a0 + a1 + a2 + a3 + a4 + a5 + a6) 
  (h2 : a0 + a1 + a2 + a3 + a4 + a5 + a6 = 64) :
  m = 1 ∨ m = -3 := 
  sorry

end find_m_l446_44640


namespace value_of_expression_l446_44624

-- Let's define the sequences and sums based on the conditions in a)
def sum_of_evens (n : ℕ) : ℕ :=
  n * (n + 1)

def sum_of_multiples_of_three (p : ℕ) : ℕ :=
  3 * (p * (p + 1)) / 2

def sum_of_odds (m : ℕ) : ℕ :=
  m * m

-- Now let's formulate the problem statement as a theorem.
theorem value_of_expression : 
  sum_of_evens 200 - sum_of_multiples_of_three 100 - sum_of_odds 148 = 3146 :=
  by
  sorry

end value_of_expression_l446_44624


namespace least_expensive_trip_is_1627_44_l446_44692

noncomputable def least_expensive_trip_cost : ℝ :=
  let distance_DE := 4500
  let distance_DF := 4000
  let distance_EF := Real.sqrt (distance_DE ^ 2 - distance_DF ^ 2)
  let cost_bus (distance : ℝ) : ℝ := distance * 0.20
  let cost_plane (distance : ℝ) : ℝ := distance * 0.12 + 120
  let cost_DE := min (cost_bus distance_DE) (cost_plane distance_DE)
  let cost_EF := min (cost_bus distance_EF) (cost_plane distance_EF)
  let cost_DF := min (cost_bus distance_DF) (cost_plane distance_DF)
  cost_DE + cost_EF + cost_DF

theorem least_expensive_trip_is_1627_44 :
  least_expensive_trip_cost = 1627.44 := sorry

end least_expensive_trip_is_1627_44_l446_44692


namespace bounds_of_F_and_G_l446_44689

noncomputable def F (a b c x : ℝ) : ℝ := a * x^2 + b * x + c
noncomputable def G (a b c x : ℝ) : ℝ := c * x^2 + b * x + a

theorem bounds_of_F_and_G {a b c : ℝ}
  (hF0 : |F a b c 0| ≤ 1)
  (hF1 : |F a b c 1| ≤ 1)
  (hFm1 : |F a b c (-1)| ≤ 1) :
  (∀ x, |x| ≤ 1 → |F a b c x| ≤ 5/4) ∧
  (∀ x, |x| ≤ 1 → |G a b c x| ≤ 2) :=
by
  sorry

end bounds_of_F_and_G_l446_44689


namespace find_f_2017_l446_44674

theorem find_f_2017 (f : ℕ → ℕ) (H1 : ∀ x y : ℕ, f (x * y + 1) = f x * f y - f y - x + 2) (H2 : f 0 = 1) : f 2017 = 2018 :=
sorry

end find_f_2017_l446_44674


namespace class_boys_count_l446_44605

theorem class_boys_count
    (x y : ℕ)
    (h1 : x + y = 20)
    (h2 : (1 / 3 : ℚ) * x = (1 / 2 : ℚ) * y) :
    x = 12 :=
by
  sorry

end class_boys_count_l446_44605


namespace calculate_ab_plus_cd_l446_44657

theorem calculate_ab_plus_cd (a b c d : ℝ) 
  (h1 : a + b + c = 5)
  (h2 : a + b + d = -1)
  (h3 : a + c + d = 8)
  (h4 : b + c + d = 12) :
  a * b + c * d = 27 :=
by
  sorry -- Proof to be filled in.

end calculate_ab_plus_cd_l446_44657


namespace birds_find_more_than_half_millet_on_sunday_l446_44696

noncomputable def seed_millet_fraction : ℕ → ℚ
| 0 => 2 * 0.2 -- initial amount on Day 1 (Monday)
| (n+1) => 0.7 * seed_millet_fraction n + 0.4

theorem birds_find_more_than_half_millet_on_sunday :
  let dayMillets : ℕ := 7
  let total_seeds : ℚ := 2
  let half_seeds : ℚ := total_seeds / 2
  (seed_millet_fraction dayMillets > half_seeds) := by
    sorry

end birds_find_more_than_half_millet_on_sunday_l446_44696


namespace sort_mail_together_time_l446_44660

-- Definitions of work rates
def mail_handler_work_rate : ℚ := 1 / 3
def assistant_work_rate : ℚ := 1 / 6

-- Definition to calculate combined work time
def combined_time (rate1 rate2 : ℚ) : ℚ := 1 / (rate1 + rate2)

-- Statement to prove
theorem sort_mail_together_time :
  combined_time mail_handler_work_rate assistant_work_rate = 2 := by
  -- Proof goes here
  sorry

end sort_mail_together_time_l446_44660


namespace rectangle_perimeter_l446_44693

theorem rectangle_perimeter {y x : ℝ} (hxy : x < y) : 
  2 * (y - x) + 2 * x = 2 * y :=
by
  sorry

end rectangle_perimeter_l446_44693


namespace simplify_expression_l446_44618

theorem simplify_expression (w : ℝ) : 2 * w + 4 * w + 6 * w + 8 * w + 10 * w + 12 = 30 * w + 12 :=
by
  sorry

end simplify_expression_l446_44618


namespace equilateral_triangle_ab_l446_44607

noncomputable def a : ℝ := 25 * Real.sqrt 3
noncomputable def b : ℝ := 5 * Real.sqrt 3

theorem equilateral_triangle_ab
  (a_val : a = 25 * Real.sqrt 3)
  (b_val : b = 5 * Real.sqrt 3)
  (h1 : Complex.abs (a + 15 * Complex.I) = 25)
  (h2 : Complex.abs (b + 45 * Complex.I) = 45)
  (h3 : Complex.abs ((a - b) + (15 - 45) * Complex.I) = 30) :
  a * b = 375 := 
sorry

end equilateral_triangle_ab_l446_44607


namespace max_abs_sum_of_squares_eq_2_sqrt_2_l446_44673

theorem max_abs_sum_of_squares_eq_2_sqrt_2 (x y : ℝ) (h : x^2 + y^2 = 4) : |x| + |y| ≤ 2 * Real.sqrt 2 :=
by
  sorry

end max_abs_sum_of_squares_eq_2_sqrt_2_l446_44673


namespace min_value_of_expression_l446_44698

theorem min_value_of_expression (n : ℕ) (h_pos : n > 0) : n = 8 → (n / 2 + 32 / n) = 8 :=
by sorry

end min_value_of_expression_l446_44698


namespace train_cross_time_l446_44606

noncomputable def train_length : ℕ := 1200 -- length of the train in meters
noncomputable def platform_length : ℕ := train_length -- length of the platform equals the train length
noncomputable def speed_kmh : ℝ := 144 -- speed in km/hr
noncomputable def speed_ms : ℝ := speed_kmh * (1000 / 3600) -- converting speed to m/s

-- the formula to calculate the crossing time
noncomputable def time_to_cross_platform : ℝ := 
  2 * train_length / speed_ms

theorem train_cross_time : time_to_cross_platform = 60 := by
  sorry

end train_cross_time_l446_44606


namespace odd_function_periodic_example_l446_44642

theorem odd_function_periodic_example (f : ℝ → ℝ) 
  (h_odd : ∀ x, f (-x) = -f x)
  (h_period : ∀ x, f (x + 2) = -f x) 
  (h_segment : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = 2 * x) :
  f (10 * Real.sqrt 3) = 36 - 20 * Real.sqrt 3 := 
sorry

end odd_function_periodic_example_l446_44642


namespace height_difference_l446_44681

theorem height_difference :
  let janet_height := 3.6666666666666665
  let sister_height := 2.3333333333333335
  janet_height - sister_height = 1.333333333333333 :=
by
  sorry

end height_difference_l446_44681


namespace batch_of_pizza_dough_makes_three_pizzas_l446_44634

theorem batch_of_pizza_dough_makes_three_pizzas
  (pizza_dough_time : ℕ)
  (baking_time : ℕ)
  (total_time_minutes : ℕ)
  (oven_capacity : ℕ)
  (total_pizzas : ℕ) 
  (number_of_batches : ℕ)
  (one_batch_pizzas : ℕ) :
  pizza_dough_time = 30 →
  baking_time = 30 →
  total_time_minutes = 300 →
  oven_capacity = 2 →
  total_pizzas = 12 →
  total_time_minutes = total_pizzas / oven_capacity * baking_time + number_of_batches * pizza_dough_time →
  number_of_batches = total_time_minutes / 30 →
  one_batch_pizzas = total_pizzas / number_of_batches →
  one_batch_pizzas = 3 :=
by
  intros
  sorry

end batch_of_pizza_dough_makes_three_pizzas_l446_44634


namespace image_center_coordinates_l446_44668

-- Define the point reflecting across the x-axis
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

-- Define the point translation by adding some units to the y-coordinate
def translate_y (p : ℝ × ℝ) (dy : ℝ) : ℝ × ℝ :=
  (p.1, p.2 + dy)

-- Define the initial point and translation
def initial_point : ℝ × ℝ := (3, -4)
def translation_units : ℝ := 5

-- Prove the final coordinates of the image of the center of circle Q
theorem image_center_coordinates : translate_y (reflect_x initial_point) translation_units = (3, 9) :=
  sorry

end image_center_coordinates_l446_44668


namespace f_a_minus_2_lt_0_l446_44613

theorem f_a_minus_2_lt_0 (f : ℝ → ℝ) (m a : ℝ) (h1 : ∀ x, f x = (m + 1 - x) * (x - m + 1)) (h2 : f a > 0) : f (a - 2) < 0 := 
sorry

end f_a_minus_2_lt_0_l446_44613


namespace number_of_integers_l446_44638

theorem number_of_integers (n : ℤ) : 
  (16 < n^2) → (n^2 < 121) → n = -10 ∨ n = -9 ∨ n = -8 ∨ n = -7 ∨ n = -6 ∨ n = -5 ∨ n = 5 ∨ n = 6 ∨ n = 7 ∨ n = 8 ∨ n = 9 ∨ n = 10 := 
by
  sorry

end number_of_integers_l446_44638


namespace distinct_integer_roots_l446_44629

theorem distinct_integer_roots (a : ℤ) : 
  (∃ u v : ℤ, u ≠ v ∧ (u + v = -a) ∧ (u * v = 2 * a)) ↔ a = -1 ∨ a = 9 :=
by
  sorry

end distinct_integer_roots_l446_44629


namespace strawberries_final_count_l446_44603

def initial_strawberries := 300
def buckets := 5
def strawberries_per_bucket := initial_strawberries / buckets
def strawberries_removed_per_bucket := 20
def redistributed_in_first_two := 15
def redistributed_in_third := 25

-- Defining the final counts after redistribution
def final_strawberries_first := strawberries_per_bucket - strawberries_removed_per_bucket + redistributed_in_first_two
def final_strawberries_second := strawberries_per_bucket - strawberries_removed_per_bucket + redistributed_in_first_two
def final_strawberries_third := strawberries_per_bucket - strawberries_removed_per_bucket + redistributed_in_third
def final_strawberries_fourth := strawberries_per_bucket - strawberries_removed_per_bucket
def final_strawberries_fifth := strawberries_per_bucket - strawberries_removed_per_bucket

theorem strawberries_final_count :
  final_strawberries_first = 55 ∧
  final_strawberries_second = 55 ∧
  final_strawberries_third = 65 ∧
  final_strawberries_fourth = 40 ∧
  final_strawberries_fifth = 40 := by
  sorry

end strawberries_final_count_l446_44603


namespace correct_calculated_value_l446_44653

theorem correct_calculated_value (x : ℝ) (h : 3 * x - 5 = 103) : x / 3 - 5 = 7 := 
by 
  sorry

end correct_calculated_value_l446_44653


namespace total_area_of_frequency_histogram_l446_44694

theorem total_area_of_frequency_histogram (f : ℝ → ℝ) (h_f : ∀ x, 0 ≤ f x ∧ f x ≤ 1) (integral_f_one : ∫ x, f x = 1) :
  ∫ x, f x = 1 := 
sorry

end total_area_of_frequency_histogram_l446_44694


namespace point_reflection_correct_l446_44686

def point_reflection_y_axis (x y z : ℝ) : ℝ × ℝ × ℝ :=
  (-x, y, -z)

theorem point_reflection_correct :
  point_reflection_y_axis (-3) 5 2 = (3, 5, -2) :=
by
  -- The proof would go here
  sorry

end point_reflection_correct_l446_44686


namespace speed_of_man_l446_44611

theorem speed_of_man :
  let L := 500 -- Length of the train in meters
  let t := 29.997600191984642 -- Time in seconds
  let V_train_kmh := 63 -- Speed of train in km/hr
  let V_train := (63 * 1000) / 3600 -- Speed of train converted to m/s
  let V_relative := L / t -- Relative speed of train w.r.t man
  
  V_train - V_relative = 0.833 := by
  sorry

end speed_of_man_l446_44611


namespace find_a_range_l446_44665

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  if x < 1 then -(x - 1) ^ 2 else (3 - a) * x + 4 * a

theorem find_a_range (a : ℝ) :
  (∀ (x₁ x₂ : ℝ), x₁ ≠ x₂ → (f x₁ a - f x₂ a) / (x₁ - x₂) > 0) ↔ (-1 ≤ a ∧ a < 3) :=
sorry

end find_a_range_l446_44665


namespace hakimi_age_is_40_l446_44661

variable (H : ℕ)
variable (Jared_age : ℕ) (Molly_age : ℕ := 30)
variable (total_age : ℕ := 120)

theorem hakimi_age_is_40 (h1 : Jared_age = H + 10) (h2 : H + Jared_age + Molly_age = total_age) : H = 40 :=
by
  sorry

end hakimi_age_is_40_l446_44661


namespace solve_fractional_equation_l446_44656

theorem solve_fractional_equation (x : ℝ) (h₀ : x ≠ 1) :
  (x^2 - x + 2) / (x - 1) = x + 3 ↔ x = 5 / 3 :=
by
  sorry

end solve_fractional_equation_l446_44656


namespace arithmetic_sequence_nth_term_l446_44678

theorem arithmetic_sequence_nth_term (a₁ a₂ a₃ : ℤ) (x : ℤ) (n : ℕ)
  (h₁ : a₁ = 3 * x - 4)
  (h₂ : a₂ = 6 * x - 14)
  (h₃ : a₃ = 4 * x + 3)
  (h₄ : ∀ k : ℕ, a₁ + (k - 1) * ((a₂ - a₁) + (a₃ - a₂) / 2) = 3012) :
  n = 247 :=
by {
  -- Proof to be provided
  sorry
}

end arithmetic_sequence_nth_term_l446_44678


namespace isosceles_triangle_base_angle_l446_44635

-- Define the problem and the given conditions
theorem isosceles_triangle_base_angle (A B C : ℝ)
(h_triangle : A + B + C = 180)
(h_isosceles : (A = B ∨ B = C ∨ C = A))
(h_ratio : (A = B / 2 ∨ B = C / 2 ∨ C = A / 2)) :
(A = 45 ∨ A = 72) ∨ (B = 45 ∨ B = 72) ∨ (C = 45 ∨ C = 72) :=
sorry

end isosceles_triangle_base_angle_l446_44635


namespace floor_add_frac_eq_154_l446_44648

theorem floor_add_frac_eq_154 (r : ℝ) (h : ⌊r⌋ + r = 15.4) : r = 7.4 := 
sorry

end floor_add_frac_eq_154_l446_44648


namespace y_squared_in_range_l446_44602

theorem y_squared_in_range (y : ℝ) 
  (h : (Real.sqrt (Real.sqrt (y + 16)) - Real.sqrt (Real.sqrt (y - 16)) = 2)) :
  270 ≤ y^2 ∧ y^2 ≤ 280 :=
sorry

end y_squared_in_range_l446_44602


namespace simplify_expression_l446_44615

theorem simplify_expression (a : ℝ) (h₀ : a ≥ 0) (h₁ : a ≠ 1) (h₂ : a ≠ 1 + Real.sqrt 2) (h₃ : a ≠ 1 - Real.sqrt 2) :
  (1 + 2 * a ^ (1 / 4) - a ^ (1 / 2)) / (1 - a + 4 * a ^ (3 / 4) - 4 * a ^ (1 / 2)) +
  (a ^ (1 / 4) - 2) / (a ^ (1 / 4) - 1) ^ 2 = 1 / (a ^ (1 / 4) - 1) :=
by
  sorry

end simplify_expression_l446_44615


namespace inequality_proof_l446_44614

theorem inequality_proof (a b : ℝ) (h : a - |b| > 0) : b + a > 0 :=
sorry

end inequality_proof_l446_44614


namespace n_is_power_of_p_l446_44620

-- Given conditions as definitions
variables {x y p n k l : ℕ}
variables (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < p) (h4 : 0 < n) (h5 : 0 < k)
variables (h6 : x^n + y^n = p^k) (h7 : odd n) (h8 : n > 1) (h9 : prime p) (h10 : odd p)

-- The theorem to be proved
theorem n_is_power_of_p : ∃ l : ℕ, n = p^l :=
  sorry

end n_is_power_of_p_l446_44620


namespace quadruplet_babies_l446_44675

variable (a b c : ℕ)
variable (h1 : b = 3 * c)
variable (h2 : a = 5 * b)
variable (h3 : 2 * a + 3 * b + 4 * c = 1500)

theorem quadruplet_babies : 4 * c = 136 := by
  sorry

end quadruplet_babies_l446_44675


namespace integer_solutions_inequality_system_l446_44649

theorem integer_solutions_inequality_system :
  {x : ℤ | 2 * (x - 1) ≤ x + 3 ∧ (x + 1) / 3 < x - 1} = {3, 4, 5} :=
by
  sorry

end integer_solutions_inequality_system_l446_44649


namespace sequence_general_term_and_sum_sum_tn_bound_l446_44699

theorem sequence_general_term_and_sum (c : ℝ) (h₁ : c = 1) 
  (f : ℕ → ℝ) (hf : ∀ x, f x = (1 / 3) ^ x) :
  (∀ n, a_n = -2 / 3 ^ n) ∧ (∀ n, b_n = 2 * n - 1) :=
by {
  sorry
}

theorem sum_tn_bound (h₂ : ∀ n > 0, T_n = (1 / 2) * (1 - 1 / (2 * n + 1))) :
  ∃ n, T_n > 1005 / 2014 ∧ n = 252 :=
by {
  sorry
}

end sequence_general_term_and_sum_sum_tn_bound_l446_44699


namespace geom_sequence_a7_l446_44647

theorem geom_sequence_a7 (a : ℕ → ℝ) (r : ℝ) 
  (h_geom : ∀ n : ℕ, a (n+1) = a n * r) 
  (h_a1 : a 1 = 8) 
  (h_a4_eq : a 4 = a 3 * a 5) : 
  a 7 = 1 / 8 :=
by
  sorry

end geom_sequence_a7_l446_44647


namespace regular_polygon_exterior_angle_l446_44622

theorem regular_polygon_exterior_angle (n : ℕ) (h : 72 = 360 / n) : n = 5 :=
by
  sorry

end regular_polygon_exterior_angle_l446_44622


namespace bananas_needed_to_make_yogurts_l446_44677

theorem bananas_needed_to_make_yogurts 
    (slices_per_yogurt : ℕ) 
    (slices_per_banana: ℕ) 
    (number_of_yogurts: ℕ) 
    (total_needed_slices: ℕ) 
    (bananas_needed: ℕ) 
    (h1: slices_per_yogurt = 8)
    (h2: slices_per_banana = 10)
    (h3: number_of_yogurts = 5)
    (h4: total_needed_slices = number_of_yogurts * slices_per_yogurt)
    (h5: bananas_needed = total_needed_slices / slices_per_banana): 
    bananas_needed = 4 := 
by
    sorry

end bananas_needed_to_make_yogurts_l446_44677


namespace sum_of_altitudes_less_than_sum_of_sides_l446_44679

theorem sum_of_altitudes_less_than_sum_of_sides 
  (a b c h_a h_b h_c K : ℝ) 
  (triangle_area : K = (1/2) * a * h_a)
  (h_a_def : h_a = 2 * K / a) 
  (h_b_def : h_b = 2 * K / b)
  (h_c_def : h_c = 2 * K / c) : 
  h_a + h_b + h_c < a + b + c := by
  sorry

end sum_of_altitudes_less_than_sum_of_sides_l446_44679


namespace replaced_solution_percentage_l446_44631

theorem replaced_solution_percentage (y x z w : ℝ) 
  (h1 : x = 0.5)
  (h2 : y = 80)
  (h3 : z = 0.5 * y)
  (h4 : w = 50) 
  :
  (40 + 0.5 * x) = 50 → x = 20 :=
by
  sorry

end replaced_solution_percentage_l446_44631


namespace value_of_fraction_l446_44609

variable {x y : ℝ}

theorem value_of_fraction (h1 : y > x) (h2 : x > 0) (h3 : x / y + y / x = 4) : 
  (x + y) / (x - y) = Real.sqrt 3 := 
sorry

end value_of_fraction_l446_44609


namespace simplify_expression_l446_44636

theorem simplify_expression :
  (3^4 + 3^2) / (3^3 - 3) = 15 / 4 :=
by {
  sorry
}

end simplify_expression_l446_44636


namespace jane_last_day_vases_l446_44658

theorem jane_last_day_vases (vases_per_day : ℕ) (total_vases : ℕ) (days : ℕ) (day_arrange_total: days = 17) (vases_per_day_is_25 : vases_per_day = 25) (total_vases_is_378 : total_vases = 378) :
  (vases_per_day * (days - 1) >= total_vases) → (total_vases - vases_per_day * (days - 1)) = 0 :=
by
  intros h
  -- adding this line below to match condition ": (total_vases - vases_per_day * (days - 1)) = 0"
  sorry

end jane_last_day_vases_l446_44658


namespace multiply_fractions_l446_44641

theorem multiply_fractions :
  (1 / 3 : ℚ) * (3 / 5) * (5 / 6) = 1 / 6 :=
by
  sorry

end multiply_fractions_l446_44641


namespace union_A_B_complement_intersection_A_B_l446_44630

def A : Set ℝ := {x | 3 ≤ x ∧ x < 10}

def B : Set ℝ := {x | 2 * x - 8 ≥ 0}

theorem union_A_B : A ∪ B = { x | x ≥ 3 } := 
by
  sorry

theorem complement_intersection_A_B : (A ∩ B)ᶜ = { x | x < 4 } ∪ { x | x ≥ 10 } := 
by
  sorry

end union_A_B_complement_intersection_A_B_l446_44630


namespace min_value_a_plus_2b_l446_44623

theorem min_value_a_plus_2b (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) 
(h_condition : (a + b) / (a * b) = 1) : a + 2 * b ≥ 3 + 2 * Real.sqrt 2 :=
by
  sorry

end min_value_a_plus_2b_l446_44623


namespace odd_phone_calls_are_even_l446_44652

theorem odd_phone_calls_are_even (n : ℕ) : Even (2 * n) :=
by
  sorry

end odd_phone_calls_are_even_l446_44652


namespace arithmetic_seq_formula_sum_first_n_terms_l446_44683

/-- Define the given arithmetic sequence an -/
def arithmetic_seq (a1 d : ℤ) : ℕ → ℤ
| 0       => a1
| (n + 1) => arithmetic_seq a1 d n + d

variable {a3 a7 : ℤ}
variable (a3_eq : arithmetic_seq 1 2 2 = 5)
variable (a7_eq : arithmetic_seq 1 2 6 = 13)

/-- Define the sequence bn -/
def b_seq (n : ℕ) : ℚ :=
  1 / ((2 * n + 1) * (arithmetic_seq 1 2 n))

/-- Define the sum of the first n terms of the sequence bn -/
def sum_b_seq : ℕ → ℚ
| 0       => 0
| (n + 1) => sum_b_seq n + b_seq (n + 1)
          
theorem arithmetic_seq_formula:
  ∀ (n : ℕ), arithmetic_seq 1 2 n = 2 * n - 1 :=
by
  intros
  sorry

theorem sum_first_n_terms:
  ∀ (n : ℕ), sum_b_seq n = n / (2 * n + 1) :=
by
  intros
  sorry

end arithmetic_seq_formula_sum_first_n_terms_l446_44683


namespace range_of_m_l446_44610

theorem range_of_m (x y : ℝ) (m : ℝ) (h1 : x^2 + y^2 = 9) (h2 : |x| + |y| ≥ m) :
    m ≤ 3 / 2 := 
sorry

end range_of_m_l446_44610


namespace gain_percent_is_25_l446_44619

theorem gain_percent_is_25 (C S : ℝ) (h : 50 * C = 40 * S) : (S - C) / C * 100 = 25 :=
  sorry

end gain_percent_is_25_l446_44619


namespace sum_of_factors_36_eq_91_l446_44667

/-- A helper definition to list all whole-number factors of 36 -/
def factors_of_36 : List ℕ := [1, 2, 3, 4, 6, 9, 12, 18, 36]

/-- The sum_of_factors function computes the sum of factors of a given number -/
def sum_of_factors (n : ℕ) (factors : List ℕ) : ℕ :=
  factors.foldl (· + ·) 0

/-- The main theorem stating that the sum of the whole-number factors of 36 is 91 -/
theorem sum_of_factors_36_eq_91 : sum_of_factors 36 factors_of_36 = 91 := by
  sorry

end sum_of_factors_36_eq_91_l446_44667


namespace blue_marble_difference_l446_44643

theorem blue_marble_difference :
  ∃ a b : ℕ, (10 * a = 10 * b) ∧ (3 * a + b = 80) ∧ (7 * a - 9 * b = 40) := by
  sorry

end blue_marble_difference_l446_44643


namespace relationship_f_3x_ge_f_2x_l446_44600

/-- Given a quadratic function f(x) = ax^2 + bx + c with a > 0, and
    satisfying the symmetry condition f(1-x) = f(1+x) for any x ∈ ℝ,
    the relationship f(3^x) ≥ f(2^x) holds. -/
theorem relationship_f_3x_ge_f_2x (a b c : ℝ) (h_a : a > 0) (symm_cond : ∀ x : ℝ, (a * (1 - x)^2 + b * (1 - x) + c) = (a * (1 + x)^2 + b * (1 + x) + c)) :
  ∀ x : ℝ, (a * (3^x)^2 + b * 3^x + c) ≥ (a * (2^x)^2 + b * 2^x + c) :=
sorry

end relationship_f_3x_ge_f_2x_l446_44600


namespace f_1996x_eq_1996_f_x_l446_44601

theorem f_1996x_eq_1996_f_x (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (x^3 + y^3) = (x + y) * (f x ^ 2 - f x * f y + f y ^ 2)) :
  ∀ x : ℝ, f (1996 * x) = 1996 * f x :=
by
  sorry

end f_1996x_eq_1996_f_x_l446_44601


namespace equal_cubes_l446_44666

theorem equal_cubes (a : ℤ) : -(a ^ 3) = (-a) ^ 3 :=
by
  sorry

end equal_cubes_l446_44666


namespace a_range_l446_44645

theorem a_range (x y z a : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_sum : x + y + z = 1)
  (h_eq : a / (x * y * z) = (1 / x) + (1 / y) + (1 / z) - 2) :
  0 < a ∧ a ≤ 7 / 27 :=
sorry

end a_range_l446_44645


namespace solve_for_y_l446_44644

theorem solve_for_y (y : ℝ) (h : 3 * y ^ (1 / 4) - 5 * (y / y ^ (3 / 4)) = 2 + y ^ (1 / 4)) : y = 16 / 81 :=
by
  sorry

end solve_for_y_l446_44644


namespace floor_e_minus_3_eq_negative_one_l446_44617

theorem floor_e_minus_3_eq_negative_one 
  (e : ℝ) 
  (h : 2 < e ∧ e < 3) : 
  (⌊e - 3⌋ = -1) :=
by
  sorry

end floor_e_minus_3_eq_negative_one_l446_44617


namespace edward_spent_13_l446_44633

-- Define the initial amount of money Edward had
def initial_amount : ℕ := 19
-- Define the current amount of money Edward has now
def current_amount : ℕ := 6
-- Define the amount of money Edward spent
def amount_spent : ℕ := initial_amount - current_amount

-- The proof we need to show
theorem edward_spent_13 : amount_spent = 13 := by
  -- The proof goes here.
  sorry

end edward_spent_13_l446_44633


namespace right_triangle_area_l446_44659

theorem right_triangle_area (x : ℝ) (h : 3 * x + 4 * x = 10) : 
  (1 / 2) * (3 * x) * (4 * x) = 24 :=
sorry

end right_triangle_area_l446_44659


namespace positive_number_solution_l446_44621

theorem positive_number_solution (x : ℚ) (hx : 0 < x) (h : x * x^2 * (1 / x) = 100 / 81) : x = 10 / 9 :=
sorry

end positive_number_solution_l446_44621


namespace cost_of_cookies_l446_44676

theorem cost_of_cookies (diane_has : ℕ) (needs_more : ℕ) (cost : ℕ) :
  diane_has = 27 → needs_more = 38 → cost = 65 :=
by
  sorry

end cost_of_cookies_l446_44676


namespace range_of_inverse_proportion_l446_44671

theorem range_of_inverse_proportion (x : ℝ) (h : 3 < x) :
    -1 < -3 / x ∧ -3 / x < 0 :=
by
  sorry

end range_of_inverse_proportion_l446_44671


namespace intersection_M_N_l446_44632

open Set

def M := {x : ℝ | x^2 - 2 * x - 3 ≤ 0}
def N := {x : ℝ | 0 < x}
def intersection := {x : ℝ | 0 < x ∧ x ≤ 3}

theorem intersection_M_N : M ∩ N = intersection := by
  sorry

end intersection_M_N_l446_44632


namespace eval_expression_at_minus_3_l446_44669

theorem eval_expression_at_minus_3 :
  (5 + 2 * x * (x + 2) - 4^2) / (x - 4 + x^2) = -5 / 2 :=
by
  let x := -3
  sorry

end eval_expression_at_minus_3_l446_44669
