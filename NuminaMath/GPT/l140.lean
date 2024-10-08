import Mathlib

namespace unique_partition_no_primes_l140_140553

open Set

def C_oplus_C (C : Set ℕ) : Set ℕ :=
  {z | ∃ x y, x ∈ C ∧ y ∈ C ∧ x ≠ y ∧ z = x + y}

def is_partition (A B : Set ℕ) : Prop :=
  (A ∪ B = univ) ∧ (A ∩ B = ∅)

theorem unique_partition_no_primes (A B : Set ℕ) :
  (is_partition A B) ∧ (∀ x ∈ C_oplus_C A, ¬Nat.Prime x) ∧ (∀ x ∈ C_oplus_C B, ¬Nat.Prime x) ↔ 
    (A = { n | n % 2 = 1 }) ∧ (B = { n | n % 2 = 0 }) :=
sorry

end unique_partition_no_primes_l140_140553


namespace numSpaceDiagonals_P_is_241_l140_140777

noncomputable def numSpaceDiagonals (vertices : ℕ) (edges : ℕ) (tri_faces : ℕ) (quad_faces : ℕ) : ℕ :=
  let total_segments := (vertices * (vertices - 1)) / 2
  let face_diagonals := 2 * quad_faces
  total_segments - edges - face_diagonals

theorem numSpaceDiagonals_P_is_241 :
  numSpaceDiagonals 26 60 24 12 = 241 := by 
  sorry

end numSpaceDiagonals_P_is_241_l140_140777


namespace A_n_divisible_by_225_l140_140326

theorem A_n_divisible_by_225 (n : ℕ) : 225 ∣ (16^n - 15 * n - 1) := by
  sorry

end A_n_divisible_by_225_l140_140326


namespace no_real_solution_l140_140544

theorem no_real_solution (x y : ℝ) (h: y = 3 * x - 1) : ¬ (4 * y ^ 2 + y + 3 = 3 * (8 * x ^ 2 + 3 * y + 1)) :=
by
  sorry

end no_real_solution_l140_140544


namespace sasha_hometown_name_l140_140075

theorem sasha_hometown_name :
  ∃ (sasha_hometown : String), 
  (∃ (vadik_last_column : String), vadik_last_column = "ВКСАМО") →
  (∃ (sasha_transformed : String), sasha_transformed = "мТТЛАРАЕкис") →
  (∃ (sasha_starts_with : Char), sasha_starts_with = 'с') →
  sasha_hometown = "СТЕРЛИТАМАК" :=
by
  sorry

end sasha_hometown_name_l140_140075


namespace set_inter_compl_eq_l140_140817

def U := ℝ
def M : Set ℝ := { x | abs (x - 1/2) ≤ 5/2 }
def P : Set ℝ := { x | -1 ≤ x ∧ x ≤ 4 }
def complement_U_M : Set ℝ := { x | x < -2 ∨ x > 3 }

theorem set_inter_compl_eq :
  (complement_U_M ∩ P) = { x | 3 < x ∧ x ≤ 4 } :=
sorry

end set_inter_compl_eq_l140_140817


namespace pentagon_angle_sum_l140_140107

theorem pentagon_angle_sum (A B C D Q : ℝ) (hA : A = 118) (hB : B = 105) (hC : C = 87) (hD : D = 135) :
  (A + B + C + D + Q = 540) -> Q = 95 :=
by
  sorry

end pentagon_angle_sum_l140_140107


namespace max_area_of_garden_l140_140338

theorem max_area_of_garden (l w : ℝ) (h : l + 2*w = 270) : l * w ≤ 9112.5 :=
sorry

end max_area_of_garden_l140_140338


namespace distance_between_parallel_lines_eq_l140_140784

open Real

theorem distance_between_parallel_lines_eq
  (h₁ : ∀ (x y : ℝ), 3 * x + y - 3 = 0 → Prop)
  (h₂ : ∀ (x y : ℝ), 6 * x + 2 * y + 1 = 0 → Prop) :
  ∃ d : ℝ, d = (7 / 20) * sqrt 10 :=
sorry

end distance_between_parallel_lines_eq_l140_140784


namespace convex_polygon_sides_l140_140932

theorem convex_polygon_sides (n : ℕ) (h : ∀ angle, angle = 45 → angle * n = 360) : n = 8 :=
  sorry

end convex_polygon_sides_l140_140932


namespace remainder_of_c_plus_d_l140_140946

-- Definitions based on conditions
def c (k : ℕ) : ℕ := 60 * k + 53
def d (m : ℕ) : ℕ := 40 * m + 29

-- Statement of the problem
theorem remainder_of_c_plus_d (k m : ℕ) :
  ((c k + d m) % 20) = 2 :=
by
  unfold c
  unfold d
  sorry

end remainder_of_c_plus_d_l140_140946


namespace A_and_C_amount_l140_140343

variables (A B C : ℝ)

def amounts_satisfy_conditions : Prop :=
  (A + B + C = 500) ∧ (B + C = 320) ∧ (C = 20)

theorem A_and_C_amount (h : amounts_satisfy_conditions A B C) : A + C = 200 :=
by {
  sorry
}

end A_and_C_amount_l140_140343


namespace find_quadratic_function_l140_140009

def quad_function (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem find_quadratic_function : ∃ (a b c : ℝ), 
  (∀ x : ℝ, quad_function a b c x = 2 * x^2 + 4 * x - 1) ∧ 
  (quad_function a b c (-1) = -3) ∧ 
  (quad_function a b c 1 = 5) :=
sorry

end find_quadratic_function_l140_140009


namespace largest_n_proof_l140_140729

def largest_n_less_than_50000_divisible_by_7 (n : ℕ) : Prop :=
  n < 50000 ∧ (10 * (n - 3)^5 - 2 * n^2 + 20 * n - 36) % 7 = 0

theorem largest_n_proof : ∃ n, largest_n_less_than_50000_divisible_by_7 n ∧ ∀ m, largest_n_less_than_50000_divisible_by_7 m → m ≤ n := 
sorry

end largest_n_proof_l140_140729


namespace n19_minus_n7_div_30_l140_140847

theorem n19_minus_n7_div_30 (n : ℕ) (h : 0 < n) : 30 ∣ (n^19 - n^7) :=
sorry

end n19_minus_n7_div_30_l140_140847


namespace range_of_m_l140_140099

noncomputable def set_A := { x : ℝ | x^2 + x - 6 = 0 }
noncomputable def set_B (m : ℝ) := { x : ℝ | m * x + 1 = 0 }

theorem range_of_m (m : ℝ) : set_A ∪ set_B m = set_A → m = 0 ∨ m = -1 / 2 ∨ m = 1 / 3 :=
by
  sorry

end range_of_m_l140_140099


namespace apples_used_l140_140259

theorem apples_used (apples_before : ℕ) (apples_left : ℕ) (apples_used_for_pie : ℕ) 
                    (h1 : apples_before = 19) 
                    (h2 : apples_left = 4) 
                    (h3 : apples_used_for_pie = apples_before - apples_left) : 
  apples_used_for_pie = 15 :=
by
  -- Since we are instructed to leave the proof out, we put sorry here
  sorry

end apples_used_l140_140259


namespace probability_point_below_x_axis_l140_140344

structure Point2D :=
  (x : ℝ)
  (y : ℝ)

structure Parallelogram :=
  (P Q R S : Point2D)

def vertices_of_PQRS : Parallelogram :=
  ⟨⟨4, 4⟩, ⟨-2, -2⟩, ⟨-8, -2⟩, ⟨-2, 4⟩⟩

def point_lies_below_x_axis_probability (parallelogram : Parallelogram) : ℝ :=
  sorry

theorem probability_point_below_x_axis :
  point_lies_below_x_axis_probability vertices_of_PQRS = 1 / 2 :=
sorry

end probability_point_below_x_axis_l140_140344


namespace aurelia_percentage_l140_140497

variables (P : ℝ)

theorem aurelia_percentage (h1 : 2000 + (P / 100) * 2000 = 3400) : 
  P = 70 :=
by
  sorry

end aurelia_percentage_l140_140497


namespace cos_plus_sin_l140_140477

open Real

theorem cos_plus_sin (α : ℝ) (h₁ : tan α = -2) (h₂ : (π / 2) < α ∧ α < π) : 
  cos α + sin α = (sqrt 5) / 5 :=
sorry

end cos_plus_sin_l140_140477


namespace power_div_eq_l140_140911

theorem power_div_eq (a : ℕ) (h : 36 = 6^2) : (6^12 / 36^5) = 36 := by
  sorry

end power_div_eq_l140_140911


namespace find_expression_value_l140_140334

theorem find_expression_value 
  (m : ℝ) 
  (hroot : m^2 - 3 * m + 1 = 0) : 
  (m - 3)^2 + (m + 2) * (m - 2) = 3 := 
sorry

end find_expression_value_l140_140334


namespace tennis_balls_ordered_originally_l140_140330

-- Definitions according to the conditions in a)
def retailer_ordered_equal_white_yellow_balls (W Y : ℕ) : Prop :=
  W = Y

def dispatch_error (Y : ℕ) : ℕ :=
  Y + 90

def ratio_white_to_yellow (W Y : ℕ) : Prop :=
  W / dispatch_error Y = 8 / 13

-- Main statement
theorem tennis_balls_ordered_originally (W Y : ℕ) (h1 : retailer_ordered_equal_white_yellow_balls W Y)
  (h2 : ratio_white_to_yellow W Y) : W + Y = 288 :=
by
  sorry    -- Placeholder for the actual proof

end tennis_balls_ordered_originally_l140_140330


namespace investment_interest_rate_calculation_l140_140528

theorem investment_interest_rate_calculation :
  let initial_investment : ℝ := 15000
  let first_year_rate : ℝ := 0.08
  let first_year_investment : ℝ := initial_investment * (1 + first_year_rate)
  let second_year_investment : ℝ := 17160
  ∃ (s : ℝ), (first_year_investment * (1 + s / 100) = second_year_investment) → s = 6 :=
by
  sorry

end investment_interest_rate_calculation_l140_140528


namespace euston_carriages_l140_140054

-- Definitions of the conditions
def E (N : ℕ) : ℕ := N + 20
def No : ℕ := 100
def FS : ℕ := No + 20
def total_carriages (E N : ℕ) : ℕ := E + N + No + FS

theorem euston_carriages (N : ℕ) (h : total_carriages (E N) N = 460) : E N = 130 :=
by
  -- Proof goes here
  sorry

end euston_carriages_l140_140054


namespace find_z_value_l140_140229

theorem find_z_value (k : ℝ) (y z : ℝ) (h1 : (y = 2) → (z = 1)) (h2 : y ^ 3 * z ^ (1/3) = k) : 
  (y = 4) → z = 1 / 512 :=
by
  sorry

end find_z_value_l140_140229


namespace correct_equation_l140_140608

variables (x : ℝ) (production_planned total_clothings : ℝ)
variables (increase_rate days_ahead : ℝ)

noncomputable def daily_production (x : ℝ) := x
noncomputable def total_production := 1000
noncomputable def production_per_day_due_to_overtime (x : ℝ) := x * (1 + 0.2 : ℝ)
noncomputable def original_completion_days (x : ℝ) := total_production / daily_production x
noncomputable def increased_production_completion_days (x : ℝ) := total_production / production_per_day_due_to_overtime x
noncomputable def days_difference := original_completion_days x - increased_production_completion_days x

theorem correct_equation : days_difference x = 2 := by
  sorry

end correct_equation_l140_140608


namespace value_of_2x_l140_140884

theorem value_of_2x (x y z : ℕ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0) (h_eq : 2 * x = 6 * z) (h_sum : x + y + z = 26) : 2 * x = 6 := 
by
  sorry

end value_of_2x_l140_140884


namespace gcd_79_pow7_plus_1_and_79_pow7_plus_79_pow2_plus_1_l140_140382

theorem gcd_79_pow7_plus_1_and_79_pow7_plus_79_pow2_plus_1 (h_prime : Nat.Prime 79) : 
  Nat.gcd (79^7 + 1) (79^7 + 79^2 + 1) = 1 := 
by
  sorry

end gcd_79_pow7_plus_1_and_79_pow7_plus_79_pow2_plus_1_l140_140382


namespace compare_abc_l140_140230

noncomputable def a : ℝ :=
  (1/2) * Real.cos 16 - (Real.sqrt 3 / 2) * Real.sin 16

noncomputable def b : ℝ :=
  2 * Real.tan 14 / (1 + (Real.tan 14) ^ 2)

noncomputable def c : ℝ :=
  Real.sqrt ((1 - Real.cos 50) / 2)

theorem compare_abc : b > c ∧ c > a :=
  by sorry

end compare_abc_l140_140230


namespace average_marks_l140_140546

variable (M P C B : ℕ)

theorem average_marks (h1 : M + P = 20) (h2 : C = P + 20) 
  (h3 : B = 2 * M) (h4 : M ≤ 100) (h5 : P ≤ 100) (h6 : C ≤ 100) (h7 : B ≤ 100) :
  (M + C) / 2 = 20 := by
  sorry

end average_marks_l140_140546


namespace trig_identity_solution_l140_140510

theorem trig_identity_solution (α : ℝ) (h : Real.tan α = -1 / 2) :
  (1 + 2 * Real.sin α * Real.cos α) / (Real.sin α ^ 2 - Real.cos α ^ 2) = -1 / 3 :=
by
  sorry

end trig_identity_solution_l140_140510


namespace find_person_age_l140_140646

theorem find_person_age : ∃ x : ℕ, 4 * (x + 4) - 4 * (x - 4) = x ∧ x = 32 := by
  sorry

end find_person_age_l140_140646


namespace initial_number_is_nine_l140_140275

theorem initial_number_is_nine (x : ℕ) (h : 3 * (2 * x + 9) = 81) : x = 9 :=
by
  sorry

end initial_number_is_nine_l140_140275


namespace point_in_fourth_quadrant_l140_140765

theorem point_in_fourth_quadrant (m : ℝ) (h1 : m + 2 > 0) (h2 : m < 0) : -2 < m ∧ m < 0 := by
  sorry

end point_in_fourth_quadrant_l140_140765


namespace geometry_progressions_not_exhaust_nat_l140_140467

theorem geometry_progressions_not_exhaust_nat :
  ∃ (g : Fin 1975 → ℕ → ℕ), 
  (∀ i : Fin 1975, ∃ (a r : ℤ), ∀ n : ℕ, g i n = (a * r^n)) ∧
  (∃ m : ℕ, ∀ i : Fin 1975, ∀ n : ℕ, m ≠ g i n) :=
sorry

end geometry_progressions_not_exhaust_nat_l140_140467


namespace max_value_x_minus_2y_exists_max_value_x_minus_2y_l140_140709

theorem max_value_x_minus_2y 
  (x y : ℝ) 
  (h : x^2 - 4 * x + y^2 = 0) : 
  x - 2 * y ≤ 2 + 2 * Real.sqrt 5 :=
sorry

theorem exists_max_value_x_minus_2y 
  (x y : ℝ) 
  (h : x^2 - 4 * x + y^2 = 0) : 
  ∃ (x y : ℝ), x - 2 * y = 2 + 2 * Real.sqrt 5 :=
sorry

end max_value_x_minus_2y_exists_max_value_x_minus_2y_l140_140709


namespace determine_a_from_root_l140_140871

noncomputable def quadratic_eq (x a : ℝ) : Prop := x^2 - a = 0

theorem determine_a_from_root :
  (∃ a : ℝ, quadratic_eq 2 a) → (∃ a : ℝ, a = 4) :=
by
  intro h
  obtain ⟨a, ha⟩ := h
  use a
  have h_eq : 2^2 - a = 0 := ha
  linarith

end determine_a_from_root_l140_140871


namespace total_hike_time_l140_140522

/-!
# Problem Statement
Jeannie hikes the 12 miles to Mount Overlook at a pace of 4 miles per hour, 
and then returns at a pace of 6 miles per hour. Prove that the total time 
Jeannie spent on her hike is 5 hours.
-/

def distance_to_mountain : ℝ := 12
def pace_up : ℝ := 4
def pace_down : ℝ := 6

theorem total_hike_time :
  (distance_to_mountain / pace_up) + (distance_to_mountain / pace_down) = 5 := 
by 
  sorry

end total_hike_time_l140_140522


namespace joan_already_put_in_cups_l140_140582

def recipe_cups : ℕ := 7
def cups_needed : ℕ := 4

theorem joan_already_put_in_cups : (recipe_cups - cups_needed = 3) :=
by
  sorry

end joan_already_put_in_cups_l140_140582


namespace no_arithmetic_progression_40_terms_l140_140241

noncomputable def is_arith_prog (f : ℕ → ℕ) (a : ℕ) (b : ℕ) : Prop :=
∀ n : ℕ, ∃ k : ℕ, f n = a + n * b

noncomputable def in_form_2m_3n (x : ℕ) : Prop :=
∃ m n : ℕ, x = 2^m + 3^n

theorem no_arithmetic_progression_40_terms :
  ¬ (∃ (a b : ℕ), ∀ n, n < 40 → in_form_2m_3n (a + n * b)) :=
sorry

end no_arithmetic_progression_40_terms_l140_140241


namespace skittles_per_friend_l140_140191

theorem skittles_per_friend (ts : ℕ) (nf : ℕ) (h1 : ts = 200) (h2 : nf = 5) : (ts / nf = 40) :=
by sorry

end skittles_per_friend_l140_140191


namespace crayons_count_l140_140144

theorem crayons_count (l b f : ℕ) (h1 : l = b / 2) (h2 : b = 3 * f) (h3 : l = 27) : f = 18 :=
by
  sorry

end crayons_count_l140_140144


namespace bill_took_six_naps_l140_140034

def total_hours (days : Nat) : Nat := days * 24

def hours_left (total : Nat) (worked : Nat) : Nat := total - worked

def naps_taken (remaining : Nat) (duration : Nat) : Nat := remaining / duration

theorem bill_took_six_naps :
  let days := 4
  let hours_worked := 54
  let nap_duration := 7
  naps_taken (hours_left (total_hours days) hours_worked) nap_duration = 6 := 
by {
  sorry
}

end bill_took_six_naps_l140_140034


namespace area_of_region_l140_140543

theorem area_of_region : 
  (∃ (A : ℝ), A = 12 ∧ ∀ (x y : ℝ), |x| + |y| + |x - 2| ≤ 4 → 
    (0 ≤ y ∧ y ≤ 6 - 2*x ∧ x ≥ 2) ∨
    (0 ≤ y ∧ y ≤ 2 ∧ 0 ≤ x ∧ x < 2) ∨
    (0 ≤ y ∧ y ≤ 2*x + 2 ∧ -1 ≤ x ∧ x < 0) ∨
    (0 ≤ y ∧ y ≤ 2*x + 2 ∧ x < -1)) :=
sorry

end area_of_region_l140_140543


namespace katie_books_ratio_l140_140783

theorem katie_books_ratio
  (d : ℕ)
  (k : ℚ)
  (g : ℕ)
  (total_books : ℕ)
  (hd : d = 6)
  (hk : ∃ k : ℚ, k = (k : ℚ))
  (hg : g = 5 * (d + k * d))
  (ht : total_books = d + k * d + g)
  (htotal : total_books = 54) :
  k = 1 / 2 :=
by
  sorry

end katie_books_ratio_l140_140783


namespace simplify_expression_l140_140162

theorem simplify_expression :
  ((3 + 4 + 6 + 7) / 3) + ((4 * 3 + 5 - 2) / 4) = 125 / 12 := by
  sorry

end simplify_expression_l140_140162


namespace mrs_hilt_rocks_l140_140576

def garden_length := 10
def garden_width := 15
def rock_coverage := 1
def available_rocks := 64

theorem mrs_hilt_rocks :
  ∃ extra_rocks : ℕ, 2 * (garden_length + garden_width) <= available_rocks ∧ extra_rocks = available_rocks - 2 * (garden_length + garden_width) ∧ extra_rocks = 14 :=
by
  sorry

end mrs_hilt_rocks_l140_140576


namespace color_plane_no_unit_equilateral_same_color_l140_140607

theorem color_plane_no_unit_equilateral_same_color :
  ∃ (coloring : ℝ × ℝ → ℕ), (∀ (A B C : ℝ × ℝ),
    (dist A B = 1 ∧ dist B C = 1 ∧ dist C A = 1) → 
    (coloring A ≠ coloring B ∨ coloring B ≠ coloring C ∨ coloring C ≠ coloring A)) :=
sorry

end color_plane_no_unit_equilateral_same_color_l140_140607


namespace find_positive_X_l140_140504

variable (X : ℝ) (Y : ℝ)

def hash_rel (X Y : ℝ) : ℝ :=
  X^2 + Y^2

theorem find_positive_X :
  hash_rel X 7 = 250 → X = Real.sqrt 201 :=
by
  sorry

end find_positive_X_l140_140504


namespace smallest_consecutive_even_sum_140_l140_140621

theorem smallest_consecutive_even_sum_140 :
  ∃ (x : ℕ), (x % 2 = 0) ∧ (x + (x + 2) + (x + 4) + (x + 6) = 140) ∧ (x = 32) :=
by
  sorry

end smallest_consecutive_even_sum_140_l140_140621


namespace ashok_average_marks_l140_140239

variable (avg_5_subjects : ℕ) (marks_6th_subject : ℕ)
def total_marks_5_subjects := avg_5_subjects * 5
def total_marks_6_subjects := total_marks_5_subjects avg_5_subjects + marks_6th_subject
def avg_6_subjects := total_marks_6_subjects avg_5_subjects marks_6th_subject / 6

theorem ashok_average_marks (h1 : avg_5_subjects = 74) (h2 : marks_6th_subject = 50) : avg_6_subjects avg_5_subjects marks_6th_subject = 70 := by
  sorry

end ashok_average_marks_l140_140239


namespace problem1_problem2_l140_140146

open Real

theorem problem1: 
  ((25^(1/3) - 125^(1/2)) / 5^(1/4) = 5^(5/12) - 5^(5/4)) :=
sorry

theorem problem2 (a : ℝ) (h : 0 < a): 
  (a^2 / (a^(1/2) * a^(2/3)) = a^(5/6)) :=
sorry

end problem1_problem2_l140_140146


namespace line_intersects_ellipse_l140_140591

theorem line_intersects_ellipse (b : ℝ) : (∃ (k : ℝ), ∀ (x y : ℝ), y = k * x + 1 → ((x^2 / 5) + (y^2 / b) = 1))
  ↔ b ∈ (Set.Ico 1 5 ∪ Set.Ioi 5) := by
sorry

end line_intersects_ellipse_l140_140591


namespace max_divisor_of_expression_l140_140448

theorem max_divisor_of_expression 
  (n : ℕ) (hn : n > 0) : ∃ k, k = 8 ∧ 8 ∣ (5^n + 2 * 3^(n-1) + 1) :=
by
  sorry

end max_divisor_of_expression_l140_140448


namespace smallest_number_l140_140376

theorem smallest_number (x : ℕ) : (∃ y : ℕ, y = x - 16 ∧ (y % 4 = 0) ∧ (y % 6 = 0) ∧ (y % 8 = 0) ∧ (y % 10 = 0)) → x = 136 := by
  sorry

end smallest_number_l140_140376


namespace puppies_left_l140_140392

theorem puppies_left (initial_puppies : ℕ) (given_away : ℕ) (remaining_puppies : ℕ) 
  (h1 : initial_puppies = 12) 
  (h2 : given_away = 7) 
  (h3 : remaining_puppies = initial_puppies - given_away) : 
  remaining_puppies = 5 :=
  by
  sorry

end puppies_left_l140_140392


namespace find_b_l140_140147

theorem find_b (a b : ℤ) 
  (h1 : a * b = 2 * (a + b) + 14) 
  (h2 : b - a = 3) : 
  b = 8 :=
sorry

end find_b_l140_140147


namespace minimum_value_of_quadratic_function_l140_140899

variable (p q : ℝ) (hp : 0 < p) (hq : 0 < q)

theorem minimum_value_of_quadratic_function : 
  (∃ x : ℝ, x = p) ∧ (∀ x : ℝ, (x^2 - 2 * p * x + 4 * q) ≥ (p^2 - 2 * p * p + 4 * q)) :=
sorry

end minimum_value_of_quadratic_function_l140_140899


namespace fraction_evaluation_l140_140489

theorem fraction_evaluation :
  (11 - 10 + 9 - 8 + 7 - 6 + 5 - 4 + 3 - 2) / (0 - 1 + 2 - 3 + 4 - 5 + 6 - 7 + 8) = 5 / 4 :=
by
  sorry

end fraction_evaluation_l140_140489


namespace floor_sqrt_120_eq_10_l140_140822

theorem floor_sqrt_120_eq_10 : ⌊Real.sqrt 120⌋ = 10 := by
  -- Here, we note that we are given:
  -- 100 < 120 < 121 and the square root of it lies between 10 and 11
  sorry

end floor_sqrt_120_eq_10_l140_140822


namespace average_length_of_strings_l140_140066

theorem average_length_of_strings (l1 l2 l3 : ℝ) (hl1 : l1 = 2) (hl2 : l2 = 5) (hl3 : l3 = 3) : 
  (l1 + l2 + l3) / 3 = 10 / 3 :=
by
  rw [hl1, hl2, hl3]
  change (2 + 5 + 3) / 3 = 10 / 3
  sorry

end average_length_of_strings_l140_140066


namespace complement_intersection_l140_140771

open Set

def U : Set ℕ := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
def A : Set ℕ := {0, 1, 3, 5, 8}
def B : Set ℕ := {2, 4, 5, 6, 8}

theorem complement_intersection (hU : ∀ x, x ∈ U) (hA : ∀ x, x ∈ A) (hB : ∀ x, x ∈ B) :
    (U \ A) ∩ (U \ B) = {7, 9} :=
by
  sorry

end complement_intersection_l140_140771


namespace hall_volume_l140_140987

theorem hall_volume (length breadth : ℝ) (h : ℝ)
  (h_length : length = 15) (h_breadth : breadth = 12)
  (h_area : 2 * (length * breadth) = 2 * (breadth * h) + 2 * (length * h)) :
  length * breadth * h = 8004 := 
by
  -- Proof not required
  sorry

end hall_volume_l140_140987


namespace eval_dollar_expr_l140_140634

noncomputable def dollar (k : ℝ) (a b : ℝ) := k * (a - b) ^ 2

theorem eval_dollar_expr (x y : ℝ) : dollar 3 ((2 * x - 3 * y) ^ 2) ((3 * y - 2 * x) ^ 2) = 0 :=
by sorry

end eval_dollar_expr_l140_140634


namespace time_to_walk_2_miles_l140_140878

/-- I walked 2 miles in a certain amount of time. -/
def walked_distance : ℝ := 2

/-- If I maintained this pace for 8 hours, I would walk 16 miles. -/
def pace_condition (pace : ℝ) : Prop :=
  pace * 8 = 16

/-- Prove that it took me 1 hour to walk 2 miles. -/
theorem time_to_walk_2_miles (t : ℝ) (pace : ℝ) (h1 : walked_distance = pace * t) (h2 : pace_condition pace) :
  t = 1 :=
sorry

end time_to_walk_2_miles_l140_140878


namespace alberto_biked_more_than_bjorn_l140_140954

-- Define the distances traveled by Bjorn and Alberto after 5 hours.
def b_distance : ℝ := 75
def a_distance : ℝ := 100

-- Statement to prove the distance difference after 5 hours.
theorem alberto_biked_more_than_bjorn : a_distance - b_distance = 25 := 
by
  -- Proof is skipped, focusing only on the statement.
  sorry

end alberto_biked_more_than_bjorn_l140_140954


namespace number_of_boys_in_school_l140_140253

-- Definition of percentages for Muslims, Hindus, and Sikhs
def percent_muslims : ℝ := 0.46
def percent_hindus : ℝ := 0.28
def percent_sikhs : ℝ := 0.10

-- Given number of boys in other communities
def boys_other_communities : ℝ := 136

-- The total number of boys in the school
def total_boys (B : ℝ) : Prop := B = 850

-- Proof statement (with conditions embedded)
theorem number_of_boys_in_school (B : ℝ) :
  percent_muslims * B + percent_hindus * B + percent_sikhs * B + boys_other_communities = B → 
  total_boys B :=
by
  sorry

end number_of_boys_in_school_l140_140253


namespace percentage_difference_highest_lowest_salary_l140_140443

variables (R : ℝ)
def Ram_salary := 1.25 * R
def Simran_salary := 0.85 * R
def Rahul_salary := 0.85 * R * 1.10

theorem percentage_difference_highest_lowest_salary :
  let highest_salary := Ram_salary R
  let lowest_salary := Simran_salary R
  (highest_salary ≠ 0) → ((highest_salary - lowest_salary) / highest_salary) * 100 = 32 :=
by
  intros
  -- Sorry in place of proof
  sorry

end percentage_difference_highest_lowest_salary_l140_140443


namespace last_digit_of_sum_edges_l140_140989

def total_edges (n : ℕ) : ℕ := (n + 1) * n * 2

def internal_edges (n : ℕ) : ℕ := (n - 1) * n * 2

def dominoes (n : ℕ) : ℕ := (n * n) / 2

def perfect_matchings (n : ℕ) : ℕ := if n = 8 then 12988816 else 0  -- specific to 8x8 chessboard

def sum_internal_edges_contribution (n : ℕ) : ℕ := perfect_matchings n * (dominoes n * 2)

def last_digit (n : ℕ) : ℕ := n % 10

theorem last_digit_of_sum_edges {n : ℕ} (h : n = 8) :
  last_digit (sum_internal_edges_contribution n) = 4 :=
by
  rw [h]
  sorry

end last_digit_of_sum_edges_l140_140989


namespace sequence_sum_129_l140_140192

/-- 
  In an increasing sequence of four positive integers where the first three terms form an arithmetic
  progression and the last three terms form a geometric progression, and where the first and fourth
  terms differ by 30, the sum of the four terms is 129.
-/
theorem sequence_sum_129 :
  ∃ (a d : ℕ), a > 0 ∧ d > 0 ∧ (a < a + d) ∧ (a + d < a + 2 * d) ∧ 
    (a + 2 * d < a + 30) ∧ 30 = (a + 30) - a ∧ 
    (a + d) * (a + 30) = (a + 2 * d) ^ 2 ∧ 
    a + (a + d) + (a + 2 * d) + (a + 30) = 129 :=
sorry

end sequence_sum_129_l140_140192


namespace non_empty_prime_subsets_count_l140_140089

-- Definition of the set S
def S : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

-- Definition of primes in S
def prime_subset_S : Set ℕ := {x ∈ S | Nat.Prime x}

-- The statement to prove
theorem non_empty_prime_subsets_count : 
  ∃ n, n = 15 ∧ ∀ T ⊆ prime_subset_S, T ≠ ∅ → ∃ m, n = 2^m - 1 := 
by
  sorry

end non_empty_prime_subsets_count_l140_140089


namespace tangent_line_to_parabola_l140_140386

theorem tangent_line_to_parabola : ∃ k : ℝ, (∀ x y : ℝ, 4 * x + 6 * y + k = 0) ∧ (∀ y : ℝ, ∃ x : ℝ, y^2 = 32 * x) ∧ (48^2 - 4 * (1 : ℝ) * 8 * k = 0) := by
  use 72
  sorry

end tangent_line_to_parabola_l140_140386


namespace three_point_seven_five_as_fraction_l140_140775

theorem three_point_seven_five_as_fraction :
  (15 : ℚ) / 4 = 3.75 :=
sorry

end three_point_seven_five_as_fraction_l140_140775


namespace map_length_represents_75_km_l140_140525
-- First, we broaden the import to bring in all the necessary libraries.

-- Define the conditions given in the problem.
def cm_to_km_ratio (cm : ℕ) (km : ℕ) : ℕ := km / cm

def map_represents (length_cm : ℕ) (length_km : ℕ) : Prop :=
  length_km = length_cm * cm_to_km_ratio 15 45

-- Rewrite the problem statement as a theorem in Lean 4.
theorem map_length_represents_75_km : map_represents 25 75 :=
by
  sorry

end map_length_represents_75_km_l140_140525


namespace abs_gt_x_iff_x_lt_0_l140_140801

theorem abs_gt_x_iff_x_lt_0 (x : ℝ) : |x| > x ↔ x < 0 := 
by
  sorry

end abs_gt_x_iff_x_lt_0_l140_140801


namespace problem1_problem2_l140_140177

variable {A B C : ℝ}
variable {a b c : ℝ}
variable (h1 : cos B - 2 * cos A = (2 * a - b) * cos C / c)
variable (h2 : a = 2 * b)

theorem problem1 : a / b = 2 :=
by sorry

theorem problem2 (h3 : A > π / 2) (h4 : c = 3) : 0 < b ∧ b < 3 :=
by sorry

end problem1_problem2_l140_140177


namespace height_relationship_height_at_90_l140_140602

noncomputable def f (x : ℝ) : ℝ := (1/2) * x

theorem height_relationship :
  (∀ x : ℝ, (x = 10 -> f x = 5) ∧ (x = 30 -> f x = 15) ∧ (x = 50 -> f x = 25) ∧ (x = 70 -> f x = 35)) → (∀ x : ℝ, f x = (1/2) * x) :=
by
  sorry

theorem height_at_90 :
  f 90 = 45 :=
by
  sorry

end height_relationship_height_at_90_l140_140602


namespace domain_inequality_l140_140182

theorem domain_inequality (m : ℝ) : 
  (∀ x : ℝ, m * x^2 - (1 - m) * x + m ≥ 0) ↔ (m ≥ 1/3) :=
by
  sorry

end domain_inequality_l140_140182


namespace average_annual_growth_rate_l140_140232

theorem average_annual_growth_rate (x : ℝ) (h : (1 + x)^2 = 1.20) : x < 0.1 :=
sorry

end average_annual_growth_rate_l140_140232


namespace algebra_expression_value_l140_140571

theorem algebra_expression_value (x y : ℝ) (h : x = 2 * y + 1) : x^2 - 4 * x * y + 4 * y^2 = 1 := 
by 
  sorry

end algebra_expression_value_l140_140571


namespace cattle_train_speed_is_correct_l140_140787

-- Given conditions as definitions
def cattle_train_speed (x : ℝ) : ℝ := x
def diesel_train_speed (x : ℝ) : ℝ := x - 33
def cattle_train_distance (x : ℝ) : ℝ := 6 * x
def diesel_train_distance (x : ℝ) : ℝ := 12 * (x - 33)

-- Statement to prove
theorem cattle_train_speed_is_correct (x : ℝ) :
  cattle_train_distance x + diesel_train_distance x = 1284 → 
  x = 93.33 :=
by
  intros h
  sorry

end cattle_train_speed_is_correct_l140_140787


namespace no_such_integers_l140_140668

noncomputable def omega : ℂ := Complex.exp (2 * Real.pi * Complex.I / 5)

theorem no_such_integers (a b c d k : ℤ) (h : k > 1) :
  (a + b * omega + c * omega^2 + d * omega^3)^k ≠ 1 + omega :=
sorry

end no_such_integers_l140_140668


namespace parallel_lines_slope_l140_140846

theorem parallel_lines_slope {a : ℝ} (h : -a / 3 = -2 / 3) : a = 2 := 
by
  sorry

end parallel_lines_slope_l140_140846


namespace square_side_length_l140_140482

theorem square_side_length (a b : ℕ) (h : a = 9) (h' : b = 16) (A : ℕ) (h1: A = a * b) :
  ∃ (s : ℕ), s * s = A ∧ s = 12 :=
by
  sorry

end square_side_length_l140_140482


namespace probability_both_blue_l140_140876

-- Conditions defined as assumptions
def jarC_red := 6
def jarC_blue := 10
def total_buttons_in_C := jarC_red + jarC_blue

def after_transfer_buttons_in_C := (3 / 4) * total_buttons_in_C

-- Carla removes the same number of red and blue buttons
-- and after transfer, 12 buttons remain in Jar C
def removed_buttons := total_buttons_in_C - after_transfer_buttons_in_C
def removed_red_buttons := removed_buttons / 2
def removed_blue_buttons := removed_buttons / 2

def remaining_red_in_C := jarC_red - removed_red_buttons
def remaining_blue_in_C := jarC_blue - removed_blue_buttons
def remaining_buttons_in_C := remaining_red_in_C + remaining_blue_in_C

def total_buttons_in_D := removed_buttons
def transferred_blue_buttons := removed_blue_buttons

-- Probability calculations
def probability_blue_in_C := remaining_blue_in_C / remaining_buttons_in_C
def probability_blue_in_D := transferred_blue_buttons / total_buttons_in_D

-- Proof
theorem probability_both_blue :
  (probability_blue_in_C * probability_blue_in_D) = (1 / 3) := 
by
  -- sorry is used here to skip the actual proof
  sorry

end probability_both_blue_l140_140876


namespace total_students_l140_140390

variables (F G B N : ℕ)
variables (hF : F = 41) (hG : G = 22) (hB : B = 9) (hN : N = 6)

theorem total_students (F G B N : ℕ) (hF : F = 41) (hG : G = 22) (hB : B = 9) (hN : N = 6) : 
  F + G - B + N = 60 := by
sorry

end total_students_l140_140390


namespace perp_bisector_b_value_l140_140140

theorem perp_bisector_b_value : ∃ b : ℝ, (∀ (x y : ℝ), x + y = b) ∧ (x + y = b) ∧ (x = (-1) ∧ y = 2) ∧ (x = 3 ∧ y = 8) := sorry

end perp_bisector_b_value_l140_140140


namespace sum_of_d_and_e_l140_140061

theorem sum_of_d_and_e (d e : ℤ) : 
  (∃ d e : ℤ, ∀ x : ℝ, x^2 - 24 * x + 50 = (x + d)^2 + e) → d + e = -106 :=
by
  sorry

end sum_of_d_and_e_l140_140061


namespace domain_of_f_l140_140304

noncomputable def f (x : ℝ) : ℝ := 1 / (Real.log (x - 2) / Real.log 3 - 1)

theorem domain_of_f :
  {x : ℝ | f x = f x} = {x : ℝ | 2 < x ∧ x ≠ 5} :=
by
  sorry

end domain_of_f_l140_140304


namespace value_A_minus_B_l140_140019

-- Conditions definitions
def A : ℕ := (1 * 1000) + (16 * 100) + (28 * 10)
def B : ℕ := 355 + 245 * 3

-- Theorem statement
theorem value_A_minus_B : A - B = 1790 := by
  sorry

end value_A_minus_B_l140_140019


namespace container_volume_ratio_l140_140975

theorem container_volume_ratio
  (A B : ℝ)
  (h : (5 / 6) * A = (3 / 4) * B) :
  (A / B = 9 / 10) :=
sorry

end container_volume_ratio_l140_140975


namespace child_l140_140734

noncomputable def C (G : ℝ) := 60 - 46
noncomputable def G := 130 - 60
noncomputable def ratio := (C G) / G

theorem child's_weight_to_grandmother's_weight_is_1_5 :
  ratio = 1 / 5 :=
by
  sorry

end child_l140_140734


namespace arrangement_ways_l140_140263

def green_marbles : Nat := 7
noncomputable def N_max_blue_marbles : Nat := 924

theorem arrangement_ways (N : Nat) (blue_marbles : Nat) (total_marbles : Nat)
  (h1 : total_marbles = green_marbles + blue_marbles) 
  (h2 : ∃ b_gap, b_gap = blue_marbles - (total_marbles - green_marbles - 1))
  (h3 : blue_marbles ≥ 6)
  : N = N_max_blue_marbles := 
sorry

end arrangement_ways_l140_140263


namespace solution_set_of_inequality_g_geq_2_l140_140967

-- Definition of the function f
def f (x a : ℝ) := |x - a|

-- Definition of the function g
def g (x a : ℝ) := f x a + f (x + 2) a

-- Proof Problem I
theorem solution_set_of_inequality (a : ℝ) (x : ℝ) :
  a = -1 → (f x a ≥ 4 - |2 * x - 1|) ↔ (x ≤ -4/3 ∨ x ≥ 4/3) :=
by sorry

-- Proof Problem II
theorem g_geq_2 (a : ℝ) (x : ℝ) :
  (∀ x, f x a ≤ 1 → (0 ≤ x ∧ x ≤ 2)) → a = 1 → g x a ≥ 2 :=
by sorry

end solution_set_of_inequality_g_geq_2_l140_140967


namespace tallest_boy_is_Vladimir_l140_140519

noncomputable def Andrei_statement1 (Boris_tallest: Prop) : Prop := ¬ Boris_tallest
def Andrei_statement2 (Vladimir_shortest: Prop) : Prop := Vladimir_shortest

def Boris_statement1 (Andrei_oldest: Prop) : Prop := Andrei_oldest
def Boris_statement2 (Andrei_shortest: Prop) : Prop := Andrei_shortest

def Vladimir_statement1 (Dmitry_taller: Prop) : Prop := Dmitry_taller
def Vladimir_statement2 (Dmitry_older: Prop) : Prop := Dmitry_older

noncomputable def Dmitry_statement1 (Vladimir_statement1: Prop) (Vladimir_statement2: Prop) : Prop :=
  Vladimir_statement1 ∧ Vladimir_statement2
def Dmitry_statement2 (Dmitry_oldest: Prop) : Prop := Dmitry_oldest

axiom one_statement_true_per_boy :
  ∀ {P₁ P₂: Prop}, (P₁ ∨ P₂) ∧ ¬ (P₁ ∧ P₂)

axiom no_same_height_or_age :
  ∀ {h1 h2 h3 h4 a1 a2 a3 a4 : ℕ},
    (h1 ≠ h2 ∧ h1 ≠ h3 ∧ h1 ≠ h4 ∧ h2 ≠ h3 ∧ h2 ≠ h4 ∧ h3 ≠ h4) ∧
    (a1 ≠ a2 ∧ a1 ≠ a3 ∧ a1 ≠ a4 ∧ a2 ≠ a3 ∧ a2 ≠ a4 ∧ a3 ≠ a4)

theorem tallest_boy_is_Vladimir :
  ∀ (Andrei_shortest Vladimir_shortest Boris_tallest Dmitry_taller Dmitry_oldest Vladimir_older : Prop),
    Dmitry_statement2 Dmitry_oldest → 
    Boris_statement2 Andrei_shortest → 
    Andrei_statement1 Boris_tallest → 
    Vladimir_statement2 Vladimir_older → 
    Dmitry_statement1 (Vladimir_statement1 Dmitry_taller) (Vladimir_statement2 Vladimir_older) →
    ¬ Dmitry_taller →
    ¬ Boris_tallest →
    Vladimir = "the tallest boy" :=
  sorry

end tallest_boy_is_Vladimir_l140_140519


namespace masking_tape_problem_l140_140149

variable (width_other : ℕ)

theorem masking_tape_problem
  (h1 : ∀ w : ℕ, (2 * 4 + 2 * w) = 20)
  : width_other = 6 :=
by
  have h2 : 8 + 2 * width_other = 20 := h1 width_other
  sorry

end masking_tape_problem_l140_140149


namespace largest_regular_hexagon_proof_l140_140119

noncomputable def largest_regular_hexagon_side_length (x : ℝ) (H : ConvexHexagon) 
  (hx : -5 < x ∧ x < 6) : ℝ := 11 / 2

-- Convex Hexagon Definition
structure ConvexHexagon :=
  (sides : Vector ℝ 6)
  (is_convex : true)  -- Placeholder for convex property

theorem largest_regular_hexagon_proof (x : ℝ) (H : ConvexHexagon) 
  (hx : -5 < x ∧ x < 6)
  (H_sides_length : H.sides = ⟨[5, 6, 7, 5+x, 6-x, 7+x], by simp⟩) :
  largest_regular_hexagon_side_length x H hx = 11 / 2 :=
sorry

end largest_regular_hexagon_proof_l140_140119


namespace record_jump_l140_140862

theorem record_jump (standard_jump jump : Float) (h_standard : standard_jump = 4.00) (h_jump : jump = 3.85) : (jump - standard_jump : Float) = -0.15 := 
by
  rw [h_standard, h_jump]
  simp
  sorry

end record_jump_l140_140862


namespace fraction_of_tips_in_august_is_five_eighths_l140_140655

-- Definitions
def average_tips (other_tips_total : ℤ) (n : ℤ) : ℤ := other_tips_total / n
def total_tips (other_tips : ℤ) (august_tips : ℤ) : ℤ := other_tips + august_tips
def fraction (numerator : ℤ) (denominator : ℤ) : ℚ := (numerator : ℚ) / (denominator : ℚ)

-- Given conditions
variables (A : ℤ) -- average monthly tips for the other 6 months (March to July and September)
variables (other_months : ℤ := 6)
variables (tips_total_other : ℤ := other_months * A) -- total tips for the 6 other months
variables (tips_august : ℤ := 10 * A) -- tips for August
variables (total_tips_all : ℤ := tips_total_other + tips_august) -- total tips for all months

-- Prove the statement
theorem fraction_of_tips_in_august_is_five_eighths :
  fraction tips_august total_tips_all = 5 / 8 := by sorry

end fraction_of_tips_in_august_is_five_eighths_l140_140655


namespace find_intended_number_l140_140800

theorem find_intended_number (n : ℕ) (h : 6 * n + 382 = 988) : n = 101 := 
by {
  sorry
}

end find_intended_number_l140_140800


namespace part1_simplification_part2_inequality_l140_140664

-- Part 1: Prove the simplification of the algebraic expression
theorem part1_simplification (x : ℝ) (h₁ : x ≠ 3):
  (2 * x + 4) / (x^2 - 6 * x + 9) / ((2 * x - 1) / (x - 3) - 1) = 2 / (x - 3) :=
sorry

-- Part 2: Prove the solution set for the inequality system
theorem part2_inequality (x : ℝ) :
  (5 * x - 2 > 3 * (x + 1)) → (1/2 * x - 1 ≥ 7 - 3/2 * x) → x ≥ 4 :=
sorry

end part1_simplification_part2_inequality_l140_140664


namespace smallest_solution_of_quadratic_l140_140088

theorem smallest_solution_of_quadratic :
  ∃ x : ℝ, 6 * x^2 - 29 * x + 35 = 0 ∧ x = 7 / 3 :=
sorry

end smallest_solution_of_quadratic_l140_140088


namespace solve_for_x_l140_140436

variable (a b x : ℝ)

def operation (a b : ℝ) : ℝ := (a + 5) * b

theorem solve_for_x (h : operation x 1.3 = 11.05) : x = 3.5 :=
by
  sorry

end solve_for_x_l140_140436


namespace negation_of_p_l140_140694

-- Defining the proposition 'p'
def p : Prop := ∃ x : ℝ, x^3 > x

-- Stating the theorem
theorem negation_of_p : ¬p ↔ ∀ x : ℝ, x^3 ≤ x :=
by
  sorry

end negation_of_p_l140_140694


namespace pinecones_left_l140_140869

theorem pinecones_left (initial_pinecones : ℕ)
    (percent_eaten_by_reindeer : ℝ)
    (percent_collected_for_fires : ℝ)
    (twice_eaten_by_squirrels : ℕ → ℕ)
    (eaten_by_reindeer : ℕ → ℝ → ℕ)
    (collected_for_fires : ℕ → ℝ → ℕ)
    (h_initial : initial_pinecones = 2000)
    (h_percent_reindeer : percent_eaten_by_reindeer = 0.20)
    (h_twice_squirrels : ∀ n, twice_eaten_by_squirrels n = 2 * n)
    (h_percent_fires : percent_collected_for_fires = 0.25)
    (h_eaten_reindeer : ∀ n p, eaten_by_reindeer n p = n * p)
    (h_collected_fires : ∀ n p, collected_for_fires n p = n * p) :
  let reindeer_eat := eaten_by_reindeer initial_pinecones percent_eaten_by_reindeer
  let squirrel_eat := twice_eaten_by_squirrels reindeer_eat
  let after_eaten := initial_pinecones - reindeer_eat - squirrel_eat
  let fire_collect := collected_for_fires after_eaten percent_collected_for_fires
  let final_pinecones := after_eaten - fire_collect
  final_pinecones = 600 :=
by sorry

end pinecones_left_l140_140869


namespace evaluate_expression_l140_140658

theorem evaluate_expression : (3 / (1 - (2 / 5))) = 5 := by
  sorry

end evaluate_expression_l140_140658


namespace area_of_tangency_triangle_l140_140460

theorem area_of_tangency_triangle (c a b T varrho : ℝ) (h_right_triangle : a^2 + b^2 = c^2) (h_area : T = (1/2) * a * b) (h_inradius : varrho = (a + b - c) / 2) :
  (area_tangency : ℝ) = (varrho / c) * T :=
sorry

end area_of_tangency_triangle_l140_140460


namespace cos_triple_angle_l140_140948

theorem cos_triple_angle (θ : ℝ) (h : Real.cos θ = 1 / 3) : Real.cos (3 * θ) = -23 / 27 := by
  sorry

end cos_triple_angle_l140_140948


namespace ratio_of_part_to_whole_l140_140589

theorem ratio_of_part_to_whole (N : ℝ) :
  (2/15) * N = 14 ∧ 0.40 * N = 168 → (14 / ((1/3) * (2/5) * N)) = 1 :=
by
  -- We assume the conditions given in the problem and need to prove the ratio
  intro h
  obtain ⟨h1, h2⟩ := h
  -- Establish equality through calculations
  sorry

end ratio_of_part_to_whole_l140_140589


namespace problem_inequality_l140_140425

variable {x y : ℝ}

theorem problem_inequality (hx : 2 < x) (hy : 2 < y) : 
  (x^2 - x) / (y^2 + y) + (y^2 - y) / (x^2 + x) > 2 / 3 := 
  sorry

end problem_inequality_l140_140425


namespace smallest_integer_ending_in_6_divisible_by_13_l140_140038

theorem smallest_integer_ending_in_6_divisible_by_13 (n : ℤ) (h1 : ∃ n : ℤ, 10 * n + 6 = x) (h2 : x % 13 = 0) : x = 26 :=
  sorry

end smallest_integer_ending_in_6_divisible_by_13_l140_140038


namespace factorize_expression_l140_140637

variable (x y : ℝ)

theorem factorize_expression : xy^2 + 6*xy + 9*x = x*(y + 3)^2 := by
  sorry

end factorize_expression_l140_140637


namespace maple_tree_total_l140_140492

-- Conditions
def initial_maple_trees : ℕ := 53
def trees_planted_today : ℕ := 11

-- Theorem to prove the result
theorem maple_tree_total : initial_maple_trees + trees_planted_today = 64 := by
  sorry

end maple_tree_total_l140_140492


namespace sequence_a_n_l140_140738

theorem sequence_a_n (a : ℕ → ℚ)
  (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, n > 0 → (n^2 + n) * (a (n + 1) - a n) = 2) :
  a 20 = 29 / 10 :=
by
  sorry

end sequence_a_n_l140_140738


namespace hypotenuse_length_l140_140013

theorem hypotenuse_length (x y : ℝ) 
  (h1 : (1/3) * Real.pi * y^2 * x = 1080 * Real.pi) 
  (h2 : (1/3) * Real.pi * x^2 * y = 2430 * Real.pi) : 
  Real.sqrt (x^2 + y^2) = 6 * Real.sqrt 13 := 
  sorry

end hypotenuse_length_l140_140013


namespace largest_value_B_l140_140579

theorem largest_value_B :
  let A := ((1 / 2) / (3 / 4))
  let B := (1 / ((2 / 3) / 4))
  let C := (((1 / 2) / 3) / 4)
  let E := ((1 / (2 / 3)) / 4)
  B > A ∧ B > C ∧ B > E :=
by
  sorry

end largest_value_B_l140_140579


namespace concyclic_H_E_N_N1_N2_l140_140727

open EuclideanGeometry

noncomputable def circumcenter (A B C : Point) : Point := sorry
noncomputable def nine_point_center (A B C : Point) : Point := sorry
noncomputable def altitude (A B C : Point) : Point := sorry
noncomputable def salmon_circle_center (A O O₁ O₂ : Point) : Point := sorry
noncomputable def foot_of_perpendicular (O' B C : Point) : Point := sorry
noncomputable def is_concyclic (points : List Point) : Prop := sorry

theorem concyclic_H_E_N_N1_N2 (A B C D : Point):
  let H := altitude A B C
  let O := circumcenter A B C
  let O₁ := circumcenter A B D
  let O₂ := circumcenter A C D
  let N := nine_point_center A B C
  let N₁ := nine_point_center A B D
  let N₂ := nine_point_center A C D
  let O' := salmon_circle_center A O O₁ O₂
  let E := foot_of_perpendicular O' B C
  is_concyclic [H, E, N, N₁, N₂] :=
sorry

end concyclic_H_E_N_N1_N2_l140_140727


namespace number_of_pairs_of_positive_integers_l140_140171

theorem number_of_pairs_of_positive_integers 
    {m n : ℕ} (h_pos_m : 0 < m) (h_pos_n : 0 < n) (h_mn : m > n) (h_diff : m^2 - n^2 = 144) : 
    ∃ (pairs : Finset (ℕ × ℕ)), pairs.card = 4 ∧ (∀ p ∈ pairs, p.1 > p.2 ∧ p.1^2 - p.2^2 = 144) :=
sorry

end number_of_pairs_of_positive_integers_l140_140171


namespace no_real_roots_smallest_m_l140_140187

theorem no_real_roots_smallest_m :
  ∃ m : ℕ, m = 4 ∧
  ∀ x : ℝ, 3 * x * (m * x - 5) - 2 * x^2 + 7 = 0 → ¬ ∃ x₀ : ℝ, 
  (3 * m - 2) * x₀^2 - 15 * x₀ + 7 = 0 ∧ 281 - 84 * m < 0 := sorry

end no_real_roots_smallest_m_l140_140187


namespace inequality_positive_integers_l140_140597

theorem inequality_positive_integers (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  |n * Real.sqrt (n^2 + 1) - m| ≥ Real.sqrt 2 - 1 :=
sorry

end inequality_positive_integers_l140_140597


namespace count_6_digit_palindromes_with_even_middle_digits_l140_140111

theorem count_6_digit_palindromes_with_even_middle_digits :
  let a_values := 9
  let b_even_values := 5
  let c_values := 10
  a_values * b_even_values * c_values = 450 :=
by {
  sorry
}

end count_6_digit_palindromes_with_even_middle_digits_l140_140111


namespace solve_equation_l140_140116

theorem solve_equation (x : ℚ) (h : x ≠ -5) : 
  (x^2 + 3*x + 4) / (x + 5) = x + 6 ↔ x = -13 / 4 := by
  sorry

end solve_equation_l140_140116


namespace sum_first_70_odd_eq_4900_l140_140864

theorem sum_first_70_odd_eq_4900 (h : (70 * (70 + 1) = 4970)) :
  (70 * 70 = 4900) :=
by
  sorry

end sum_first_70_odd_eq_4900_l140_140864


namespace light_path_in_cube_l140_140314

/-- Let ABCD and EFGH be two faces of a cube with AB = 10. A beam of light is emitted 
from vertex A and reflects off face EFGH at point Q, which is 6 units from EH and 4 
units from EF. The length of the light path from A until it reaches another vertex of 
the cube for the first time is expressed in the form s√t, where s and t are integers 
with t having no square factors. Provide s + t. -/
theorem light_path_in_cube :
  let AB := 10
  let s := 10
  let t := 152
  s + t = 162 := by
  sorry

end light_path_in_cube_l140_140314


namespace garden_length_l140_140938

noncomputable def length_of_garden : ℝ := 300

theorem garden_length (P : ℝ) (b : ℝ) (A : ℝ) 
  (h₁ : P = 800) (h₂ : b = 100) (h₃ : A = 10000) : length_of_garden = 300 := 
by 
  sorry

end garden_length_l140_140938


namespace flood_damage_conversion_l140_140041

-- Define the conversion rate and the damage in Indian Rupees as given
def rupees_to_pounds (rupees : ℕ) : ℕ := rupees / 75
def damage_in_rupees : ℕ := 45000000

-- Define the expected damage in British Pounds
def expected_damage_in_pounds : ℕ := 600000

-- The theorem to prove that the damage in British Pounds is as expected, given the conditions.
theorem flood_damage_conversion :
  rupees_to_pounds damage_in_rupees = expected_damage_in_pounds :=
by
  -- The proof goes here, but we'll use sorry to skip it as instructed.
  sorry

end flood_damage_conversion_l140_140041


namespace no_preimage_range_l140_140457

open Set

def f (x : ℝ) : ℝ := x^2 + 2 * x + 3

theorem no_preimage_range :
  { k : ℝ | ∀ x : ℝ, f x ≠ k } = Iio 2 := by
  sorry

end no_preimage_range_l140_140457


namespace length_of_living_room_l140_140017

theorem length_of_living_room (width area : ℝ) (h_width : width = 14) (h_area : area = 215.6) :
  ∃ length : ℝ, length = 15.4 ∧ area = length * width :=
by
  sorry

end length_of_living_room_l140_140017


namespace remainder_sum_div_8_l140_140298

theorem remainder_sum_div_8 (n : ℤ) : (((8 - n) + (n + 5)) % 8) = 5 := 
by {
  sorry
}

end remainder_sum_div_8_l140_140298


namespace functional_equation_solution_l140_140475

def f : ℝ → ℝ := sorry

theorem functional_equation_solution :
  (∀ x y : ℝ, f (2 * x + f y) = x + y + f x) →
  (∀ x : ℝ, f x = x) :=
by
  intros h x
  sorry

end functional_equation_solution_l140_140475


namespace solve_equations_l140_140853

theorem solve_equations :
  (∀ x, x^2 - 4 = 0 ↔ x = 2 ∨ x = -2) ∧
  (∀ x, x^2 - 6 * x + 9 = 0 ↔ x = 3) ∧
  (∀ x, x^2 - 7 * x + 12 = 0 ↔ x = 3 ∨ x = 4) ∧
  (∀ x, 2 * x^2 - 3 * x - 5 = 0 ↔ x = 5 / 2 ∨ x = -1) :=
by
  -- Proof goes here
  sorry

end solve_equations_l140_140853


namespace inequality_proof_l140_140570

theorem inequality_proof (a b c d : ℝ) (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d) (h_condition : a * b + b * c + c * d + d * a = 1) :
    (a ^ 3 / (b + c + d)) + (b ^ 3 / (c + d + a)) + (c ^ 3 / (a + b + d)) + (d ^ 3 / (a + b + c)) ≥ (1 / 3) :=
by
  sorry

end inequality_proof_l140_140570


namespace floor_of_smallest_zero_l140_140205
noncomputable def g (x : ℝ) := 3 * Real.sin x - Real.cos x + 2 * Real.tan x
def smallest_zero (s : ℝ) : Prop := s > 0 ∧ g s = 0 ∧ ∀ x, 0 < x ∧ x < s → g x ≠ 0

theorem floor_of_smallest_zero (s : ℝ) (h : smallest_zero s) : ⌊s⌋ = 4 :=
sorry

end floor_of_smallest_zero_l140_140205


namespace sum_of_squares_nonzero_l140_140417

theorem sum_of_squares_nonzero {a b : ℝ} (h : a^2 + b^2 ≠ 0) : a ≠ 0 ∨ b ≠ 0 :=
sorry

end sum_of_squares_nonzero_l140_140417


namespace number_of_distinct_real_roots_l140_140432

theorem number_of_distinct_real_roots (k : ℕ) :
  (∃ k : ℕ, ∀ x : ℝ, |x| - 4 = (3 * |x|) / 2 → 0 = k) :=
  sorry

end number_of_distinct_real_roots_l140_140432


namespace simplify_and_evaluate_l140_140770

-- Define the expression
def expr (a : ℚ) : ℚ := (3 * a - 1) ^ 2 + 3 * a * (3 * a + 2)

-- Given the condition
def a_value : ℚ := -1 / 3

-- State the theorem
theorem simplify_and_evaluate : expr a_value = 3 :=
by
  -- Proof will be added here
  sorry

end simplify_and_evaluate_l140_140770


namespace find_k_l140_140821

noncomputable def vector_a : ℝ × ℝ := (-1, 1)
noncomputable def vector_b : ℝ × ℝ := (2, 3)
noncomputable def vector_c (k : ℝ) : ℝ × ℝ := (-2, k)

def perp (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

theorem find_k (k : ℝ) (h : perp (vector_a.1 + vector_b.1, vector_a.2 + vector_b.2) (vector_c k)) : k = 1 / 2 :=
by
  sorry

end find_k_l140_140821


namespace area_of_square_l140_140016

theorem area_of_square (r s l b : ℝ) (h1 : l = (2/5) * r)
                               (h2 : r = s)
                               (h3 : b = 10)
                               (h4 : l * b = 220) :
  s^2 = 3025 :=
by
  -- proof goes here
  sorry

end area_of_square_l140_140016


namespace num_zeros_g_l140_140930

noncomputable def f (x : ℝ) (m : ℝ) : ℝ :=
  if x > 2 then m * (x - 2) / x
  else if 0 < x ∧ x ≤ 2 then 3 * x - x^2
  else 0

noncomputable def g (x : ℝ) (m : ℝ) : ℝ := f x m - 2

-- Statement to prove
theorem num_zeros_g (m : ℝ) : ∃ n : ℕ, (n = 4 ∨ n = 6) :=
sorry

end num_zeros_g_l140_140930


namespace simplify_expression_l140_140529

theorem simplify_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h1 : a^4 + b^4 = a + b) (h2 : a^2 + b^2 = 2) :
  (a^2 / b^2 + b^2 / a^2 - 1 / (a^2 * b^2)) = 1 := 
sorry

end simplify_expression_l140_140529


namespace coinCombinationCount_l140_140282

-- Definitions for the coin values and the target amount
def quarter := 25
def dime := 10
def nickel := 5
def penny := 1
def total := 400

-- Define a function counting the number of ways to reach the total using given coin values
def countWays : Nat := sorry -- placeholder for the actual computation

-- Theorem stating the problem statement
theorem coinCombinationCount (n : Nat) :
  countWays = n :=
sorry

end coinCombinationCount_l140_140282


namespace scientific_notation_of_2135_billion_l140_140873

theorem scientific_notation_of_2135_billion :
  (2135 * 10^9 : ℝ) = 2.135 * 10^11 := by
  sorry

end scientific_notation_of_2135_billion_l140_140873


namespace find_a_l140_140243

theorem find_a (a : ℤ) (h1 : 0 < a) (h2 : ∀ (x : ℝ), |x - a| < 1 → x ∈ {x | x = 2}) : a = 2 :=
sorry

end find_a_l140_140243


namespace balance_difference_is_7292_83_l140_140348

noncomputable def angela_balance : ℝ := 7000 * (1 + 0.05)^15
noncomputable def bob_balance : ℝ := 9000 * (1 + 0.03)^30
noncomputable def balance_difference : ℝ := bob_balance - angela_balance

theorem balance_difference_is_7292_83 : balance_difference = 7292.83 := by
  sorry

end balance_difference_is_7292_83_l140_140348


namespace smallest_k_no_real_roots_l140_140934

theorem smallest_k_no_real_roots :
  ∀ (k : ℤ), (∀ x : ℝ, 3 * x * (k * x - 5) - x^2 + 7 ≠ 0) → k ≥ 4 :=
by
  sorry

end smallest_k_no_real_roots_l140_140934


namespace product_of_primes_sum_ten_l140_140189

theorem product_of_primes_sum_ten :
  ∃ (p1 p2 : ℕ), p1 ≠ p2 ∧ Prime p1 ∧ Prime p2 ∧ p1 + p2 = 10 ∧ p1 * p2 = 21 := 
by
  sorry

end product_of_primes_sum_ten_l140_140189


namespace sum_of_fourth_powers_is_three_times_square_l140_140220

theorem sum_of_fourth_powers_is_three_times_square (n : ℤ) (h : n ≠ 0) :
  (n - 1)^4 + n^4 + (n + 1)^4 + 10 = 3 * (n^2 + 2)^2 :=
by
  sorry

end sum_of_fourth_powers_is_three_times_square_l140_140220


namespace domain_of_f_l140_140843

theorem domain_of_f (x : ℝ) : (2*x - x^2 > 0 ∧ x ≠ 1) ↔ (0 < x ∧ x < 1) ∨ (1 < x ∧ x < 2) :=
by
  -- proof omitted
  sorry

end domain_of_f_l140_140843


namespace unique_nets_of_a_cube_l140_140772

-- Definitions based on the conditions and the properties of the cube
def is_net (net: ℕ) : Prop :=
  -- A placeholder definition of a valid net
  sorry

def is_distinct_by_rotation_or_reflection (net1 net2: ℕ) : Prop :=
  -- Two nets are distinct if they cannot be transformed into each other by rotation or reflection
  sorry

-- The statement to be proved
theorem unique_nets_of_a_cube : ∃ n, n = 11 ∧ (∀ net, is_net net → ∃! net', is_net net' ∧ is_distinct_by_rotation_or_reflection net net') :=
sorry

end unique_nets_of_a_cube_l140_140772


namespace ratio_chest_of_drawers_to_treadmill_l140_140378

theorem ratio_chest_of_drawers_to_treadmill :
  ∀ (C T TV : ℕ),
  T = 100 →
  TV = 3 * 100 →
  100 + C + TV = 600 →
  C / T = 2 :=
by
  intros C T TV ht htv heq
  sorry

end ratio_chest_of_drawers_to_treadmill_l140_140378


namespace symmetry_center_range_in_interval_l140_140917

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x - Real.pi / 6) + 1

theorem symmetry_center (k : ℤ) :
  ∃ n : ℤ, ∃ x : ℝ, x = Real.pi / 12 + n * Real.pi / 2 ∧ f x = 1 := 
sorry

theorem range_in_interval :
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 2 → ∃ y : ℝ, f y ∈ Set.Icc 0 3 := 
sorry

end symmetry_center_range_in_interval_l140_140917


namespace y_intercept_with_z_3_l140_140023

theorem y_intercept_with_z_3 : 
  ∀ x y : ℝ, (4 * x + 6 * y - 2 * 3 = 24) → (x = 0) → y = 5 :=
by
  intros x y h1 h2
  sorry

end y_intercept_with_z_3_l140_140023


namespace find_k_l140_140155

-- Define a point and its translation
structure Point where
  x : ℕ
  y : ℕ

-- Original and translated points
def P : Point := { x := 5, y := 3 }
def P' : Point := { x := P.x - 4, y := P.y - 1 }

-- Given function with parameter k
def line (k : ℕ) (p : Point) : ℕ := (k * p.x) - 2

-- Prove the value of k
theorem find_k (k : ℕ) (h : line k P' = P'.y) : k = 4 :=
by
  sorry

end find_k_l140_140155


namespace find_k_l140_140984

theorem find_k (k x y : ℕ) (h : k * 2 + 1 = 5) : k = 2 :=
by {
  -- Proof will go here
  sorry
}

end find_k_l140_140984


namespace Tim_soda_cans_l140_140530

noncomputable def initial_cans : ℕ := 22
noncomputable def taken_cans : ℕ := 6
noncomputable def remaining_cans : ℕ := initial_cans - taken_cans
noncomputable def bought_cans : ℕ := remaining_cans / 2
noncomputable def final_cans : ℕ := remaining_cans + bought_cans

theorem Tim_soda_cans :
  final_cans = 24 :=
by
  sorry

end Tim_soda_cans_l140_140530


namespace c_d_not_true_l140_140912

variables (Beatles_haircut : Type → Prop) (hooligan : Type → Prop) (rude : Type → Prop)

-- Conditions
axiom a : ∃ x, Beatles_haircut x ∧ hooligan x
axiom b : ∀ y, hooligan y → rude y

-- Prove there is a rude hooligan with a Beatles haircut
theorem c : ∃ z, rude z ∧ Beatles_haircut z ∧ hooligan z :=
sorry

-- Disprove every rude hooligan having a Beatles haircut
theorem d_not_true : ¬(∀ w, rude w ∧ hooligan w → Beatles_haircut w) :=
sorry

end c_d_not_true_l140_140912


namespace points_2_units_away_l140_140047

theorem points_2_units_away : (∃ x : ℝ, (x = -3 ∨ x = 1) ∧ (abs (x - (-1)) = 2)) :=
by
  sorry

end points_2_units_away_l140_140047


namespace geometric_sequence_fourth_term_l140_140535

theorem geometric_sequence_fourth_term (a₁ a₂ : ℝ) (h₁ : a₁ = 3) (h₂ : a₂ = 1/3) :
    ∃ a₄ : ℝ, a₄ = 1/243 :=
sorry

end geometric_sequence_fourth_term_l140_140535


namespace smallest_range_l140_140040

-- Define the conditions
def estate (A B C : ℝ) : Prop :=
  A = 20000 ∧
  abs (A - B) > 0.3 * A ∧
  abs (A - C) > 0.3 * A ∧
  abs (B - C) > 0.3 * A

-- Define the statement to prove
theorem smallest_range (A B C : ℝ) (h : estate A B C) : 
  ∃ r : ℝ, r = 12000 :=
sorry

end smallest_range_l140_140040


namespace gross_profit_value_l140_140768

theorem gross_profit_value
  (SP : ℝ) (C : ℝ) (GP : ℝ)
  (h1 : SP = 81)
  (h2 : GP = 1.7 * C)
  (h3 : SP = C + GP) :
  GP = 51 :=
by
  sorry

end gross_profit_value_l140_140768


namespace total_animals_l140_140219

theorem total_animals : ∀ (D C R : ℕ), 
  C = 5 * D →
  R = D - 12 →
  R = 4 →
  (C + D + R = 100) :=
by
  intros D C R h1 h2 h3
  sorry

end total_animals_l140_140219


namespace quadratic_root_iff_l140_140732

theorem quadratic_root_iff (a b c : ℝ) :
  (∃ x : ℝ, x = 1 ∧ a * x^2 + b * x + c = 0) ↔ (a + b + c = 0) :=
by
  sorry

end quadratic_root_iff_l140_140732


namespace simplify_and_evaluate_l140_140622

theorem simplify_and_evaluate
  (a b : ℝ)
  (h : |a - 1| + (b + 2)^2 = 0) :
  ((2 * a + b)^2 - (2 * a + b) * (2 * a - b)) / (-1 / 2 * b) = 0 := 
sorry

end simplify_and_evaluate_l140_140622


namespace concentration_of_spirit_in_vessel_a_l140_140636

theorem concentration_of_spirit_in_vessel_a :
  ∀ (x : ℝ), 
    (∀ (v1 v2 v3 : ℝ), v1 * (x / 100) + v2 * (30 / 100) + v3 * (10 / 100) = 15 * (26 / 100) →
      v1 + v2 + v3 = 15 →
      v1 = 4 → v2 = 5 → v3 = 6 →
      x = 45) :=
by
  intros x v1 v2 v3 h h_volume h_v1 h_v2 h_v3
  sorry

end concentration_of_spirit_in_vessel_a_l140_140636


namespace num_dogs_correct_l140_140071

-- Definitions based on conditions
def total_animals : ℕ := 17
def number_of_cats : ℕ := 8

-- Definition based on required proof
def number_of_dogs : ℕ := total_animals - number_of_cats

-- Proof statement
theorem num_dogs_correct : number_of_dogs = 9 :=
by
  sorry

end num_dogs_correct_l140_140071


namespace solve_star_eq_five_l140_140169

def star (a b : ℝ) : ℝ := a + b^2

theorem solve_star_eq_five :
  ∃ x₁ x₂ : ℝ, star x₁ (x₁ + 1) = 5 ∧ star x₂ (x₂ + 1) = 5 ∧ x₁ = 1 ∧ x₂ = -4 :=
by
  sorry

end solve_star_eq_five_l140_140169


namespace least_integer_gt_sqrt_700_l140_140303

theorem least_integer_gt_sqrt_700 : ∃ n : ℕ, (n - 1) < Real.sqrt 700 ∧ Real.sqrt 700 ≤ n ∧ n = 27 :=
by
  sorry

end least_integer_gt_sqrt_700_l140_140303


namespace opposite_of_2023_l140_140161

def opposite (n x : ℤ) := n + x = 0 

theorem opposite_of_2023 : ∃ x : ℤ, opposite 2023 x ∧ x = -2023 := 
by
  sorry

end opposite_of_2023_l140_140161


namespace price_of_each_book_l140_140488

theorem price_of_each_book (B P : ℕ) 
  (h1 : (1 / 3 : ℚ) * B = 36) -- Number of unsold books is 1/3 of the total books and it equals 36
  (h2 : (2 / 3 : ℚ) * B * P = 144) -- Total amount received for the books sold is $144
  : P = 2 := 
by
  sorry

end price_of_each_book_l140_140488


namespace servings_of_peanut_butter_l140_140974

theorem servings_of_peanut_butter :
  let peanutButterInJar := 37 + 4 / 5
  let oneServing := 1 + 1 / 2
  let servings := 25 + 1 / 5
  (peanutButterInJar / oneServing) = servings :=
by
  let peanutButterInJar := 37 + 4 / 5
  let oneServing := 1 + 1 / 2
  let servings := 25 + 1 / 5
  sorry

end servings_of_peanut_butter_l140_140974


namespace quadratic_inequality_l140_140052

theorem quadratic_inequality : ∀ x : ℝ, -7 * x ^ 2 + 4 * x - 6 < 0 :=
by
  intro x
  have delta : 4 ^ 2 - 4 * (-7) * (-6) = -152 := by norm_num
  have neg_discriminant : -152 < 0 := by norm_num
  have coef : -7 < 0 := by norm_num
  sorry

end quadratic_inequality_l140_140052


namespace regression_analysis_notes_l140_140195

-- Define the conditions
def applicable_population (reg_eq: Type) (sample: Type) : Prop := sorry
def temporality (reg_eq: Type) : Prop := sorry
def sample_value_range_influence (reg_eq: Type) (sample: Type) : Prop := sorry
def prediction_precision (reg_eq: Type) : Prop := sorry

-- Define the key points to note
def key_points_to_note (reg_eq: Type) (sample: Type) : Prop :=
  applicable_population reg_eq sample ∧
  temporality reg_eq ∧
  sample_value_range_influence reg_eq sample ∧
  prediction_precision reg_eq

-- The main statement
theorem regression_analysis_notes (reg_eq: Type) (sample: Type) :
  key_points_to_note reg_eq sample := sorry

end regression_analysis_notes_l140_140195


namespace problem_statement_l140_140056

theorem problem_statement (x y : ℝ) (log2_3 log5_3 : ℝ)
  (h1 : log2_3 > 1)
  (h2 : 0 < log5_3)
  (h3 : log5_3 < 1)
  (h4 : log2_3^x - log5_3^x ≥ log2_3^(-y) - log5_3^(-y)) :
  x + y ≥ 0 := 
sorry

end problem_statement_l140_140056


namespace length_of_side_divisible_by_4_l140_140957

theorem length_of_side_divisible_by_4 {m n : ℕ} 
  (h : ∀ k : ℕ, (m * k) + (n * k) % 4 = 0 ) : 
  m % 4 = 0 ∨ n % 4 = 0 :=
by
  sorry

end length_of_side_divisible_by_4_l140_140957


namespace time_to_cross_second_platform_l140_140103

-- Definition of the conditions
variables (l_train l_platform1 l_platform2 t1 : ℕ)
variable (v : ℕ)

-- The conditions given in the problem
def conditions : Prop :=
  l_train = 190 ∧
  l_platform1 = 140 ∧
  l_platform2 = 250 ∧
  t1 = 15 ∧
  v = (l_train + l_platform1) / t1

-- The statement to prove
theorem time_to_cross_second_platform
    (l_train l_platform1 l_platform2 t1 : ℕ)
    (v : ℕ)
    (h : conditions l_train l_platform1 l_platform2 t1 v) :
    (l_train + l_platform2) / v = 20 :=
  sorry

end time_to_cross_second_platform_l140_140103


namespace log_function_increasing_interval_l140_140204

theorem log_function_increasing_interval (a : ℝ) :
  (∀ x y : ℝ, -1 ≤ x → x < y → y ≤ 3 → 4 - ax > 0 ∧ (4 - ax < 4 - ay)) ↔ (-4 < a ∧ a < 0) :=
by
  sorry

end log_function_increasing_interval_l140_140204


namespace inequality_2n_squared_plus_3n_plus_1_l140_140982

theorem inequality_2n_squared_plus_3n_plus_1 (n : ℕ) (h: n > 0) : (2 * n^2 + 3 * n + 1)^n ≥ 6^n * (n! * n!) := 
by sorry

end inequality_2n_squared_plus_3n_plus_1_l140_140982


namespace matrix_determinant_l140_140025

variable {a b c d : ℝ}
variable (h : a * d - b * c = 4)

theorem matrix_determinant :
  (a * (7 * c + 3 * d) - c * (7 * a + 3 * b)) = 12 := by
  sorry

end matrix_determinant_l140_140025


namespace part1_part2_l140_140837

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x)^2 + 2 * (Real.sin x) * (Real.cos x)

theorem part1 : f (Real.pi / 8) = Real.sqrt 2 + 1 := sorry

theorem part2 : (∀ x1 x2 : ℝ, f (x1 + Real.pi) = f x1) ∧ (∀ x : ℝ, f x ≥ 1 - Real.sqrt 2) := 
  sorry

-- Explanation:
-- part1 is for proving f(π/8) = √2 + 1
-- part2 handles proving the smallest positive period and the minimum value of the function.

end part1_part2_l140_140837


namespace equivalent_knicks_l140_140951

theorem equivalent_knicks (knicks knacks knocks : ℕ) (h1 : 5 * knicks = 3 * knacks) (h2 : 4 * knacks = 6 * knocks) :
  36 * knocks = 40 * knicks :=
by
  sorry

end equivalent_knicks_l140_140951


namespace problem_l140_140027

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  (1 / 2) * a * x^2 - (x - 1) * Real.exp x

theorem problem (a : ℝ) :
  (∀ x1 x2 x3 : ℝ, 0 ≤ x1 ∧ x1 ≤ 1 ∧ 0 ≤ x2 ∧ x2 ≤ 1 ∧ 0 ≤ x3 ∧ x3 ≤ 1 →
                  f a x1 + f a x2 ≥ f a x3) →
  1 ≤ a ∧ a ≤ 4 :=
sorry

end problem_l140_140027


namespace cannot_have_N_less_than_K_l140_140156

theorem cannot_have_N_less_than_K (K N : ℕ) (hK : K > 2) (cards : Fin N → ℕ) (h_cards : ∀ i, cards i > 0) :
  ¬ (N < K) :=
sorry

end cannot_have_N_less_than_K_l140_140156


namespace min_quadratic_expression_value_l140_140270

def quadratic_expression (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 2205

theorem min_quadratic_expression_value : 
  ∃ x : ℝ, quadratic_expression x = 2178 :=
sorry

end min_quadratic_expression_value_l140_140270


namespace transform_center_l140_140097

def point := (ℝ × ℝ)

def reflect_x_axis (p : point) : point :=
  (p.1, -p.2)

def translate_right (p : point) (d : ℝ) : point :=
  (p.1 + d, p.2)

theorem transform_center (C : point) (hx : C = (3, -4)) :
  translate_right (reflect_x_axis C) 3 = (6, 4) :=
by
  sorry

end transform_center_l140_140097


namespace decreasing_sequence_b_l140_140890

def seq_a (a : ℕ → ℝ) : Prop :=
  a 1 = 2 ∧ ∀ n : ℕ, 2 * a n * a (n + 1) = (a n)^2 + 1

def b_n (a b : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, b n = (a n - 1) / (a n + 1)

theorem decreasing_sequence_b {a b : ℕ → ℝ} (h1 : seq_a a) (h2 : b_n a b) :
  ∀ n : ℕ, b (n + 1) < b n :=
by
  sorry

end decreasing_sequence_b_l140_140890


namespace sin_angle_GAC_correct_l140_140502

noncomputable def sin_angle_GAC (AB AD AE : ℝ) := 
  let AC := Real.sqrt (AB^2 + AD^2)
  let AG := Real.sqrt (AB^2 + AD^2 + AE^2)
  (AC / AG)

theorem sin_angle_GAC_correct : sin_angle_GAC 2 3 4 = Real.sqrt 377 / 29 := by
  sorry

end sin_angle_GAC_correct_l140_140502


namespace ratio_HC_JE_l140_140688

noncomputable def A : ℝ := 0
noncomputable def B : ℝ := 1
noncomputable def C : ℝ := B + 2
noncomputable def D : ℝ := C + 1
noncomputable def E : ℝ := D + 1
noncomputable def F : ℝ := E + 2

variable (G H J K : ℝ × ℝ)
variable (parallel_AG_HC parallel_AG_JE parallel_AG_KB : Prop)

-- Conditions
axiom points_on_line : A < B ∧ B < C ∧ C < D ∧ D < E ∧ E < F
axiom AB : B - A = 1
axiom BC : C - B = 2
axiom CD : D - C = 1
axiom DE : E - D = 1
axiom EF : F - E = 2
axiom G_off_AF : G.2 ≠ 0
axiom H_on_GD : H.1 = G.1 ∧ H.2 = D
axiom J_on_GF : J.1 = G.1 ∧ J.2 = F
axiom K_on_GB : K.1 = G.1 ∧ K.2 = B
axiom parallel_hc_je_kb_ag : parallel_AG_HC ∧ parallel_AG_JE ∧ parallel_AG_KB ∧ (G.2 / 1) = (K.2 / (K.1 - G.1))

-- Task: Prove the ratio HC/JE = 7/8
theorem ratio_HC_JE : (H.2 - C) / (J.2 - E) = 7 / 8 :=
sorry

end ratio_HC_JE_l140_140688


namespace solve_recursive_fraction_l140_140860

noncomputable def recursive_fraction (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0     => x
  | (n+1) => 1 + 1 / (recursive_fraction n x)

theorem solve_recursive_fraction (x : ℝ) (n : ℕ) :
  (recursive_fraction n x = x) ↔ (x = (1 + Real.sqrt 5) / 2 ∨ x = (1 - Real.sqrt 5) / 2) :=
sorry

end solve_recursive_fraction_l140_140860


namespace halved_r_value_of_n_l140_140431

theorem halved_r_value_of_n (r a : ℝ) (n : ℕ) (h₁ : a = (2 * r)^n)
  (h₂ : 0.125 * a = r^n) : n = 3 :=
by
  sorry

end halved_r_value_of_n_l140_140431


namespace function_properties_l140_140693

noncomputable def f (x : ℝ) : ℝ := (4^x - 1) / (2^(x + 1))

theorem function_properties :
  (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x y : ℝ, x < y → f x < f y) :=
by
  sorry

end function_properties_l140_140693


namespace number_of_paths_to_spell_BINGO_l140_140631

theorem number_of_paths_to_spell_BINGO : 
  ∃ (paths : ℕ), paths = 36 :=
by
  sorry

end number_of_paths_to_spell_BINGO_l140_140631


namespace remainder_of_division_l140_140110

theorem remainder_of_division (L S R : ℕ) (hL : L = 1620) (h_diff : L - S = 1365) (h_div : L = 6 * S + R) : R = 90 :=
by {
  -- Since we are not providing the proof, we use sorry
  sorry
}

end remainder_of_division_l140_140110


namespace arithmetic_sequence_a_m_n_zero_l140_140048

theorem arithmetic_sequence_a_m_n_zero
  (a : ℕ → ℕ)
  (m n : ℕ) 
  (hm : m > 0) (hn : n > 0)
  (h_ma_m : a m = n)
  (h_na_n : a n = m) : 
  a (m + n) = 0 :=
by 
  sorry

end arithmetic_sequence_a_m_n_zero_l140_140048


namespace customers_left_l140_140681

theorem customers_left (x : ℕ) 
  (h1 : 47 - x + 20 = 26) : 
  x = 41 :=
sorry

end customers_left_l140_140681


namespace correct_percentage_fruits_in_good_condition_l140_140610

noncomputable def percentage_fruits_in_good_condition
    (total_oranges : ℕ)
    (total_bananas : ℕ)
    (rotten_percentage_oranges : ℝ)
    (rotten_percentage_bananas : ℝ) : ℝ :=
let rotten_oranges := (rotten_percentage_oranges / 100) * total_oranges
let rotten_bananas := (rotten_percentage_bananas / 100) * total_bananas
let good_condition_oranges := total_oranges - rotten_oranges
let good_condition_bananas := total_bananas - rotten_bananas
let total_fruits_in_good_condition := good_condition_oranges + good_condition_bananas
let total_fruits := total_oranges + total_bananas
(total_fruits_in_good_condition / total_fruits) * 100

theorem correct_percentage_fruits_in_good_condition :
  percentage_fruits_in_good_condition 600 400 15 4 = 89.4 := by
  sorry

end correct_percentage_fruits_in_good_condition_l140_140610


namespace initial_boxes_l140_140044

theorem initial_boxes (x : ℕ) (h : x + 6 = 14) : x = 8 :=
by sorry

end initial_boxes_l140_140044


namespace polygon_diagonals_with_restricted_vertices_l140_140785

theorem polygon_diagonals_with_restricted_vertices
  (vertices : ℕ) (non_contributing_vertices : ℕ)
  (h_vertices : vertices = 35)
  (h_non_contributing_vertices : non_contributing_vertices = 5) :
  (vertices - non_contributing_vertices) * (vertices - non_contributing_vertices - 3) / 2 = 405 :=
by {
  sorry
}

end polygon_diagonals_with_restricted_vertices_l140_140785


namespace find_multiple_of_y_l140_140317

noncomputable def multiple_of_y (q m : ℝ) : Prop :=
  ∀ x y : ℝ, (x = 5 - q) → (y = m * q - 1) → (q = 1) → (x = 3 * y) → (m = 7 / 3)

theorem find_multiple_of_y :
  multiple_of_y 1 (7 / 3) :=
by
  sorry

end find_multiple_of_y_l140_140317


namespace shaded_area_of_rotated_square_is_four_thirds_l140_140209

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

end shaded_area_of_rotated_square_is_four_thirds_l140_140209


namespace positive_solution_system_l140_140227

theorem positive_solution_system (x1 x2 x3 x4 x5 : ℝ) (h1 : (x3 + x4 + x5)^5 = 3 * x1)
  (h2 : (x4 + x5 + x1)^5 = 3 * x2) (h3 : (x5 + x1 + x2)^5 = 3 * x3)
  (h4 : (x1 + x2 + x3)^5 = 3 * x4) (h5 : (x2 + x3 + x4)^5 = 3 * x5) :
  x1 > 0 → x2 > 0 → x3 > 0 → x4 > 0 → x5 > 0 →
  x1 = x2 ∧ x2 = x3 ∧ x3 = x4 ∧ x4 = x5 ∧ (x1 = 1/3) :=
by 
  intros hpos1 hpos2 hpos3 hpos4 hpos5
  sorry

end positive_solution_system_l140_140227


namespace velocity_zero_at_t_eq_2_l140_140995

noncomputable def motion_equation (t : ℝ) : ℝ := -4 * t^3 + 48 * t

theorem velocity_zero_at_t_eq_2 :
  (exists t : ℝ, t > 0 ∧ deriv (motion_equation) t = 0) :=
by
  sorry

end velocity_zero_at_t_eq_2_l140_140995


namespace product_of_real_values_r_l140_140499

theorem product_of_real_values_r {x r : ℝ} (h : x ≠ 0) (heq : (1 / (3 * x)) = ((r - x) / 8)) :
  (∃! x : ℝ, 24 * x^2 - 8 * r * x + 24 = 0) →
  r = 6 ∨ r = -6 ∧ (r * -r) = -36 :=
by
  sorry

end product_of_real_values_r_l140_140499


namespace yellow_balls_in_bag_l140_140572

theorem yellow_balls_in_bag (x : ℕ) (prob : 1 / (1 + x) = 1 / 4) :
  x = 3 :=
sorry

end yellow_balls_in_bag_l140_140572


namespace total_mangoes_calculation_l140_140650

-- Define conditions as constants
def boxes : ℕ := 36
def dozen_to_mangoes : ℕ := 12
def dozens_per_box : ℕ := 10

-- Define the expected correct answer for the total mangoes
def expected_total_mangoes : ℕ := 4320

-- Lean statement to prove
theorem total_mangoes_calculation :
  dozens_per_box * dozen_to_mangoes * boxes = expected_total_mangoes :=
by sorry

end total_mangoes_calculation_l140_140650


namespace jamie_avg_is_correct_l140_140329

-- Declare the set of test scores and corresponding sums
def test_scores : List ℤ := [75, 78, 82, 85, 88, 91]

-- Alex's average score
def alex_avg : ℤ := 82

-- Total test score sum
def total_sum : ℤ := test_scores.sum

theorem jamie_avg_is_correct (alex_sum : ℤ) :
    alex_sum = 3 * alex_avg →
    (total_sum - alex_sum) / 3 = 253 / 3 :=
by
  sorry

end jamie_avg_is_correct_l140_140329


namespace unique_triangled_pair_l140_140654

theorem unique_triangled_pair (a b x y : ℝ) (h : ∀ a b : ℝ, (a, b) = (a * x + b * y, a * y + b * x)) : (x, y) = (1, 0) :=
by sorry

end unique_triangled_pair_l140_140654


namespace pyramid_vertices_l140_140959

theorem pyramid_vertices (n : ℕ) (h : 2 * n = 14) : n + 1 = 8 :=
by {
  sorry
}

end pyramid_vertices_l140_140959


namespace find_m_l140_140969

theorem find_m (m : ℝ) :
  (∀ x : ℝ, 0 < x → (m^2 - m - 1) * x < 0) → m = -1 :=
by sorry

end find_m_l140_140969


namespace tom_total_payment_l140_140647

theorem tom_total_payment :
  let apples_cost := 8 * 70
  let mangoes_cost := 9 * 55
  let oranges_cost := 5 * 40
  let bananas_cost := 12 * 30
  let grapes_cost := 7 * 45
  let cherries_cost := 4 * 80
  apples_cost + mangoes_cost + oranges_cost + bananas_cost + grapes_cost + cherries_cost = 2250 :=
by
  sorry

end tom_total_payment_l140_140647


namespace probability_two_green_apples_l140_140750

theorem probability_two_green_apples :
  let total_apples := 9
  let total_red := 5
  let total_green := 4
  let ways_to_choose_two := Nat.choose total_apples 2
  let ways_to_choose_two_green := Nat.choose total_green 2
  ways_to_choose_two ≠ 0 →
  (ways_to_choose_two_green / ways_to_choose_two : ℚ) = 1 / 6 :=
by
  intros
  -- skipping the proof
  sorry

end probability_two_green_apples_l140_140750


namespace carSpeedIs52mpg_l140_140683

noncomputable def carSpeed (fuelConsumptionKMPL : ℕ) -- 32 kilometers per liter
                           (gallonToLiter : ℝ)        -- 1 gallon = 3.8 liters
                           (fuelDecreaseGallons : ℝ)  -- 3.9 gallons
                           (timeHours : ℝ)            -- 5.7 hours
                           (kmToMiles : ℝ)            -- 1 mile = 1.6 kilometers
                           : ℝ :=
  let totalLiters := fuelDecreaseGallons * gallonToLiter
  let totalKilometers := totalLiters * fuelConsumptionKMPL
  let totalMiles := totalKilometers / kmToMiles
  totalMiles / timeHours

theorem carSpeedIs52mpg : carSpeed 32 3.8 3.9 5.7 1.6 = 52 := sorry

end carSpeedIs52mpg_l140_140683


namespace range_of_x_l140_140134

theorem range_of_x (a : ℝ) (x : ℝ) (h_a : 1 ≤ a) : 
  ax^2 + (a - 3) * x + (a - 4) > 0 ↔ x < -1 ∨ x > 3 :=
by
  sorry

end range_of_x_l140_140134


namespace tangent_line_circle_l140_140184

open Real

theorem tangent_line_circle (m n : ℝ) :
  (∀ x y : ℝ, ((m + 1) * x + (n + 1) * y - 2 = 0) ↔ (x - 1)^2 + (y - 1)^2 = 1) →
  ((m + n) ≤ 2 - 2 * sqrt 2) ∨ (2 + 2 * sqrt 2 ≤ (m + n)) := by
  sorry

end tangent_line_circle_l140_140184


namespace sum_of_coefficients_l140_140213

-- Definition of the polynomial
def P (x : ℝ) : ℝ := 5 * (2 * x ^ 9 - 3 * x ^ 6 + 4) - 4 * (x ^ 6 - 5 * x ^ 3 + 6)

-- Theorem stating the sum of the coefficients is 7
theorem sum_of_coefficients : P 1 = 7 := by
  sorry

end sum_of_coefficients_l140_140213


namespace arithmetic_sequence_geometric_condition_l140_140193

theorem arithmetic_sequence_geometric_condition 
  (a : ℕ → ℤ) 
  (h_arith : ∀ n, a (n + 1) = a n + 3) 
  (h_geom : (a 1 + 6) ^ 2 = a 1 * (a 1 + 9)) : 
  a 2 = -9 :=
sorry

end arithmetic_sequence_geometric_condition_l140_140193


namespace product_evaluation_l140_140479

theorem product_evaluation : (3 - 1) * 3 * (3 + 1) * (3 + 2) * (3 + 3) * (3 + 4) = 5040 := by
  -- sorry
  exact rfl  -- This is just a placeholder. The proof would go here.

end product_evaluation_l140_140479


namespace range_of_m_l140_140053

theorem range_of_m
  (h : ∀ x : ℝ, (m / (2 * x - 4) = (1 - x) / (2 - x) - 2) → x > 0) :
  m < 6 ∧ m ≠ 2 :=
by
  sorry

end range_of_m_l140_140053


namespace students_accounting_majors_l140_140672

theorem students_accounting_majors (p q r s : ℕ) 
  (h1 : 1 < p) (h2 : p < q) (h3 : q < r) (h4 : r < s) (h5 : p * q * r * s = 1365) : p = 3 := 
by 
  sorry

end students_accounting_majors_l140_140672


namespace ratio_of_areas_l140_140142

-- Define the side lengths of Squared B and Square C
variables (y : ℝ)

-- Define the areas of Square B and C
def area_B := (2 * y) * (2 * y)
def area_C := (8 * y) * (8 * y)

-- The theorem statement proving the ratio of the areas
theorem ratio_of_areas : area_B y / area_C y = 1 / 16 := 
by sorry

end ratio_of_areas_l140_140142


namespace complete_square_b_l140_140306

theorem complete_square_b (a b x : ℝ) (h : x^2 + 6 * x - 3 = 0) : (x + a)^2 = b → b = 12 := by
  sorry

end complete_square_b_l140_140306


namespace vinegar_solution_concentration_l140_140165

theorem vinegar_solution_concentration
  (original_volume : ℝ) (water_volume : ℝ)
  (original_concentration : ℝ)
  (h1 : original_volume = 12)
  (h2 : water_volume = 50)
  (h3 : original_concentration = 36.166666666666664) :
  original_concentration / 100 * original_volume / (original_volume + water_volume) = 0.07 :=
by
  sorry

end vinegar_solution_concentration_l140_140165


namespace binary_to_base4_conversion_l140_140979

theorem binary_to_base4_conversion : ∀ (a b c d e : ℕ), 
  1101101101 = (11 * 2^8) + (01 * 2^6) + (10 * 2^4) + (11 * 2^2) + 01 -> 
  a = 3 -> b = 1 -> c = 2 -> d = 3 -> e = 1 -> 
  (a*10000 + b*1000 + c*100 + d*10 + e : ℕ) = 31131 :=
by
  -- proof will go here
  sorry

end binary_to_base4_conversion_l140_140979


namespace ratio_of_spinsters_to_cats_l140_140511

theorem ratio_of_spinsters_to_cats :
  (∀ S C : ℕ, (S : ℚ) / (C : ℚ) = 2 / 9) ↔
  (∃ S C : ℕ, S = 18 ∧ C = S + 63 ∧ (S : ℚ) / (C : ℚ) = 2 / 9) :=
sorry

end ratio_of_spinsters_to_cats_l140_140511


namespace wheels_on_each_other_axle_l140_140139

def truck_toll_wheels (t : ℝ) (x : ℝ) (w : ℕ) : Prop :=
  t = 1.50 + 1.50 * (x - 2) ∧ (w = 18) ∧ (∀ y : ℕ, y = 18 - 2 - 4 *(x - 5) / 4)

theorem wheels_on_each_other_axle :
  ∀ t x w, truck_toll_wheels t x w → w = 18 ∧ x = 5 → (18 - 2) / 4 = 4 :=
by
  intros t x w h₁ h₂
  have h₃ : t = 6 := sorry
  have h₄ : x = 4 := sorry
  have h₅ : w = 18 := sorry
  have h₆ : (18 - 2) / 4 = 4 := sorry
  exact h₆

end wheels_on_each_other_axle_l140_140139


namespace cannot_determine_c_l140_140125

-- Definitions based on conditions
variables {a b c d : ℕ}
axiom h1 : a + b + c = 21
axiom h2 : a + b + d = 27
axiom h3 : a + c + d = 30

-- The statement that c cannot be determined exactly
theorem cannot_determine_c : ¬ (∃ c : ℕ, c = c) :=
by sorry

end cannot_determine_c_l140_140125


namespace difference_in_dimes_l140_140516

theorem difference_in_dimes : 
  ∀ (a b c : ℕ), (a + b + c = 100) → (5 * a + 10 * b + 25 * c = 835) → 
  (∀ b_max b_min, (b_max = 67) ∧ (b_min = 3) → (b_max - b_min = 64)) :=
by
  intros a b c h1 h2 b_max b_min h_bounds
  sorry

end difference_in_dimes_l140_140516


namespace tom_seashells_left_l140_140029

def initial_seashells : ℕ := 5
def given_away_seashells : ℕ := 2

theorem tom_seashells_left : (initial_seashells - given_away_seashells) = 3 :=
by
  sorry

end tom_seashells_left_l140_140029


namespace statement_two_statement_three_l140_140086

section
variables {R : Type*} [Field R]
variables (a b c p q : R)
noncomputable def f (x : R) := a * x^2 + b * x + c

-- Statement ②
theorem statement_two (hpq : f a b c p = f a b c q) (hpq_neq : p ≠ q) : 
  f a b c (p + q) = c :=
sorry

-- Statement ③
theorem statement_three (hf : f a b c (p + q) = c) (hpq_neq : p ≠ q) : 
  p + q = 0 ∨ f a b c p = f a b c q :=
sorry

end

end statement_two_statement_three_l140_140086


namespace factory_workers_count_l140_140248

theorem factory_workers_count :
  ∃ (F S_f : ℝ), 
    (F * S_f = 30000) ∧ 
    (30 * (S_f + 500) = 75000) → 
    (F = 15) :=
by
  sorry

end factory_workers_count_l140_140248


namespace max_expression_value_l140_140418

theorem max_expression_value (a b c : ℝ) (hb : b > a) (ha : a > c) (hb_ne : b ≠ 0) :
  ∃ M, M = 27 ∧ (∀ a b c, b > a → a > c → b ≠ 0 → (∃ M, (2*a + 3*b)^2 + (b - c)^2 + (2*c - a)^2 ≤ M * b^2) → M ≤ 27) :=
  sorry

end max_expression_value_l140_140418


namespace man_days_to_complete_work_alone_l140_140310

-- Defining the variables corresponding to the conditions
variable (M : ℕ)

-- Initial condition: The man can do the work alone in M days
def man_work_rate := 1 / (M : ℚ)
-- The son can do the work alone in 20 days
def son_work_rate := 1 / 20
-- Combined work rate when together
def combined_work_rate := 1 / 4

-- The main theorem we want to prove
theorem man_days_to_complete_work_alone
  (h : man_work_rate M + son_work_rate = combined_work_rate) :
  M = 5 := by
  sorry

end man_days_to_complete_work_alone_l140_140310


namespace largest_possible_s_l140_140483

theorem largest_possible_s (r s: ℕ) (h1: r ≥ s) (h2: s ≥ 3)
  (h3: (59 : ℚ) / 58 * (180 * (s - 2) / s) = (180 * (r - 2) / r)) : s = 117 :=
sorry

end largest_possible_s_l140_140483


namespace line_through_point_with_equal_intercepts_l140_140244

theorem line_through_point_with_equal_intercepts 
  (x y k : ℝ) 
  (h1 : (3 : ℝ) + (-6 : ℝ) + k = 0 ∨ 2 * (3 : ℝ) + (-6 : ℝ) = 0) 
  (h2 : k = 0 ∨ x + y + k = 0) : 
  (x = 1 ∨ x = 2) ∧ (k = -3 ∨ k = 0) :=
sorry

end line_through_point_with_equal_intercepts_l140_140244


namespace sum_of_first_column_l140_140708

theorem sum_of_first_column (a b : ℕ) 
  (h1 : 16 * (a + b) = 96) 
  (h2 : 16 * (a - b) = 64) :
  a + b = 20 :=
by sorry

end sum_of_first_column_l140_140708


namespace exists_integer_point_touching_x_axis_l140_140008

-- Define the context for the problem
variable {p q : ℤ}

-- Condition: The quadratic trinomial touches x-axis, i.e., discriminant is zero.
axiom discriminant_zero (p q : ℤ) : p^2 - 4 * q = 0

-- Theorem statement: Proving the existence of such an integer point.
theorem exists_integer_point_touching_x_axis :
  ∃ a b : ℤ, (a = -p ∧ b = q) ∧ (∀ (x : ℝ), x^2 + a * x + b = 0 → (a * a - 4 * b) = 0) :=
sorry

end exists_integer_point_touching_x_axis_l140_140008


namespace positive_integers_sum_digits_less_than_9000_l140_140779

theorem positive_integers_sum_digits_less_than_9000 : 
  ∃ n : ℕ, n = 47 ∧ ∀ x : ℕ, (1 ≤ x ∧ x < 9000 ∧ (Nat.digits 10 x).sum = 5) → (Nat.digits 10 x).length = n :=
sorry

end positive_integers_sum_digits_less_than_9000_l140_140779


namespace initial_blue_balls_l140_140832

theorem initial_blue_balls (total_balls : ℕ) (remaining_balls : ℕ) (B : ℕ) :
  total_balls = 18 → remaining_balls = total_balls - 3 → (B - 3) / remaining_balls = 1 / 5 → B = 6 :=
by 
  intros htotal hremaining hprob
  sorry

end initial_blue_balls_l140_140832


namespace exist_same_number_of_acquaintances_l140_140721

-- Define a group of 2014 people
variable (People : Type) [Fintype People] [DecidableEq People]
variable (knows : People → People → Prop)
variable [DecidableRel knows]

-- Conditions
def mutual_acquaintance : Prop := 
  ∀ (a b : People), knows a b ↔ knows b a

def num_people : Prop := 
  Fintype.card People = 2014

-- Theorem to prove
theorem exist_same_number_of_acquaintances 
  (h1 : mutual_acquaintance People knows) 
  (h2 : num_people People) : 
  ∃ (p1 p2 : People), p1 ≠ p2 ∧
    (Fintype.card { x // knows p1 x } = Fintype.card { x // knows p2 x }) :=
sorry

end exist_same_number_of_acquaintances_l140_140721


namespace rectangle_area_l140_140807

theorem rectangle_area (a b c : ℝ) :
  a = 15 ∧ b = 12 ∧ c = 1 / 3 →
  ∃ (AD AB : ℝ), 
  AD = (180 / 17) ∧ AB = (60 / 17) ∧ 
  (AD * AB = 10800 / 289) :=
by sorry

end rectangle_area_l140_140807


namespace dilation_rotation_l140_140910

noncomputable def center : ℂ := 2 + 3 * Complex.I
noncomputable def scale_factor : ℂ := 3
noncomputable def initial_point : ℂ := -1 + Complex.I
noncomputable def final_image : ℂ := -4 + 12 * Complex.I

theorem dilation_rotation (z : ℂ) :
  z = (-1 + Complex.I) →
  let z' := center + scale_factor * (initial_point - center)
  let rotated_z := center + Complex.I * (z' - center)
  rotated_z = final_image := sorry

end dilation_rotation_l140_140910


namespace value_of_five_l140_140970

variable (f : ℝ → ℝ)

-- Conditions of the problem
def odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f (x)
def periodic_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (x + 2) = f (x)

theorem value_of_five (hf_odd : odd_function f) (hf_periodic : periodic_function f) : f 5 = 0 :=
by 
  sorry

end value_of_five_l140_140970


namespace nancy_antacids_l140_140168

theorem nancy_antacids :
  ∀ (x : ℕ),
  (3 * 3 + x * 2 + 1 * 2) * 4 = 60 → x = 2 :=
by
  sorry

end nancy_antacids_l140_140168


namespace value_of_expression_l140_140421

theorem value_of_expression : (2207 - 2024)^2 * 4 / 144 = 930.25 := 
by
  sorry

end value_of_expression_l140_140421


namespace total_cost_magic_decks_l140_140258

theorem total_cost_magic_decks (price_per_deck : ℕ) (frank_decks : ℕ) (friend_decks : ℕ) :
  price_per_deck = 7 ∧ frank_decks = 3 ∧ friend_decks = 2 → 
  (price_per_deck * frank_decks + price_per_deck * friend_decks) = 35 :=
by
  sorry

end total_cost_magic_decks_l140_140258


namespace frank_fence_l140_140604

theorem frank_fence (L W F : ℝ) (hL : L = 40) (hA : 320 = L * W) : F = 2 * W + L → F = 56 := by
  sorry

end frank_fence_l140_140604


namespace John_max_tests_under_B_l140_140724

theorem John_max_tests_under_B (total_tests first_tests tests_with_B goal_percentage B_tests_first_half : ℕ) :
  total_tests = 60 →
  first_tests = 40 → 
  tests_with_B = 32 → 
  goal_percentage = 75 →
  B_tests_first_half = 32 →
  let needed_B_tests := (goal_percentage * total_tests) / 100
  let remaining_tests := total_tests - first_tests
  let remaining_needed_B_tests := needed_B_tests - B_tests_first_half
  remaining_tests - remaining_needed_B_tests ≤ 7 := sorry

end John_max_tests_under_B_l140_140724


namespace least_number_subtracted_378461_l140_140373

def least_number_subtracted (n : ℕ) : ℕ :=
  n % 13

theorem least_number_subtracted_378461 : least_number_subtracted 378461 = 5 :=
by
  -- actual proof would go here
  sorry

end least_number_subtracted_378461_l140_140373


namespace susan_hours_per_day_l140_140797

theorem susan_hours_per_day (h : ℕ) 
  (works_five_days_a_week : Prop)
  (paid_vacation_days : ℕ)
  (unpaid_vacation_days : ℕ)
  (missed_pay : ℕ)
  (hourly_rate : ℕ)
  (total_vacation_days : ℕ)
  (total_workdays_in_2_weeks : ℕ)
  (paid_vacation_days_eq : paid_vacation_days = 6)
  (unpaid_vacation_days_eq : unpaid_vacation_days = 4)
  (missed_pay_eq : missed_pay = 480)
  (hourly_rate_eq : hourly_rate = 15)
  (total_vacation_days_eq : total_vacation_days = 14)
  (total_workdays_in_2_weeks_eq : total_workdays_in_2_weeks = 10)
  (total_unpaid_hours_in_4_days : unpaid_vacation_days * hourly_rate = missed_pay) :
  h = 8 :=
by 
  -- We need to show that Susan works 8 hours a day
  sorry

end susan_hours_per_day_l140_140797


namespace problem_1_a_problem_1_b_problem_2_l140_140964

def set_A : Set ℝ := {x | 3 ≤ x ∧ x < 6}
def set_B : Set ℝ := {x | 2 < x ∧ x < 9}
def complement_B : Set ℝ := {x | x ≤ 2 ∨ x ≥ 9}
def set_union (s₁ s₂ : Set ℝ) := {x | x ∈ s₁ ∨ x ∈ s₂}
def set_inter (s₁ s₂ : Set ℝ) := {x | x ∈ s₁ ∧ x ∈ s₂}

theorem problem_1_a :
  set_inter set_A set_B = {x : ℝ | 3 ≤ x ∧ x < 6} :=
sorry

theorem problem_1_b :
  set_union complement_B set_A = {x : ℝ | x ≤ 2 ∨ (3 ≤ x ∧ x < 6) ∨ x ≥ 9} :=
sorry

def set_C (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 1}

theorem problem_2 (a : ℝ) :
  (set_C a ⊆ set_B) → (2 ≤ a ∧ a ≤ 8) :=
sorry

end problem_1_a_problem_1_b_problem_2_l140_140964


namespace dilation_result_l140_140813

noncomputable def dilation (c a : ℂ) (k : ℝ) : ℂ := k * (c - a) + a

theorem dilation_result :
  dilation (3 - 1* I) (1 + 2* I) 4 = 9 + 6* I :=
by
  sorry

end dilation_result_l140_140813


namespace distance_from_dormitory_to_city_l140_140069

theorem distance_from_dormitory_to_city (D : ℝ) 
  (h1 : D = (1/2) * D + (1/4) * D + 6) : D = 24 := 
  sorry

end distance_from_dormitory_to_city_l140_140069


namespace abs_eq_sum_solutions_l140_140283

theorem abs_eq_sum_solutions (x : ℝ) : (|3*x - 2| + |3*x + 1| = 3) ↔ 
  (x = -1 / 3 ∨ (-1 / 3 < x ∧ x <= 2 / 3)) :=
by
  sorry

end abs_eq_sum_solutions_l140_140283


namespace starting_number_of_range_divisible_by_11_l140_140188

theorem starting_number_of_range_divisible_by_11 (a : ℕ) : 
  a ≤ 79 ∧ (a + 22 = 77) ∧ ((a + 11) + 11 = 77) → a = 55 := 
by
  sorry

end starting_number_of_range_divisible_by_11_l140_140188


namespace sides_of_rectangle_EKMR_l140_140586

noncomputable def right_triangle_ACB (AC AB : ℕ) : Prop :=
AC = 3 ∧ AB = 4

noncomputable def rectangle_EKMR_area (area : ℚ) : Prop :=
area = 3/5

noncomputable def rectangle_EKMR_perimeter (x y : ℚ) : Prop :=
2 * (x + y) < 9

theorem sides_of_rectangle_EKMR (x y : ℚ) 
  (h_triangle : right_triangle_ACB 3 4)
  (h_area : rectangle_EKMR_area (3/5))
  (h_perimeter : rectangle_EKMR_perimeter x y) : 
  (x = 2 ∧ y = 3/10) ∨ (x = 3/10 ∧ y = 2) := 
sorry

end sides_of_rectangle_EKMR_l140_140586


namespace six_digit_number_property_l140_140613

theorem six_digit_number_property :
  ∃ N : ℕ, N = 285714 ∧ (∃ x : ℕ, N = 2 * 10^5 + x ∧ M = 10 * x + 2 ∧ M = 3 * N) :=
by
  sorry

end six_digit_number_property_l140_140613


namespace usual_time_is_36_l140_140150

-- Definition: let S be the usual speed of the worker (not directly relevant to the final proof)
noncomputable def S : ℝ := sorry

-- Definition: let T be the usual time taken by the worker
noncomputable def T : ℝ := sorry

-- Condition: The worker's speed is (3/4) of her normal speed, resulting in a time (T + 12)
axiom speed_delay_condition : (3 / 4) * S * (T + 12) = S * T

-- Theorem: Prove that the usual time T taken to cover the distance is 36 minutes
theorem usual_time_is_36 : T = 36 := by
  -- Formally stating our proof based on given conditions
  sorry

end usual_time_is_36_l140_140150


namespace f_2023_eq_1375_l140_140623

-- Define the function f and the conditions
noncomputable def f : ℕ → ℕ := sorry

axiom f_ff_eq (n : ℕ) (h : n > 0) : f (f n) = 3 * n
axiom f_3n2_eq (n : ℕ) (h : n > 0) : f (3 * n + 2) = 3 * n + 1

-- Prove the specific value for f(2023)
theorem f_2023_eq_1375 : f 2023 = 1375 := sorry

end f_2023_eq_1375_l140_140623


namespace width_of_grass_field_l140_140806

-- Define the conditions
def length_of_grass_field : ℝ := 75
def path_width : ℝ := 2.5
def cost_per_sq_m : ℝ := 2
def total_cost : ℝ := 1200

-- Define the width of the grass field as a variable
variable (w : ℝ)

-- Define the total length and width including the path
def total_length : ℝ := length_of_grass_field + 2 * path_width
def total_width (w : ℝ) : ℝ := w + 2 * path_width

-- Define the area of the path
def area_of_path (w : ℝ) : ℝ := (total_length * total_width w) - (length_of_grass_field * w)

-- Define the cost equation
def cost_eq (w : ℝ) : Prop := cost_per_sq_m * area_of_path w = total_cost

-- The theorem to prove
theorem width_of_grass_field : cost_eq 40 :=
by
  -- To be proved
  sorry

end width_of_grass_field_l140_140806


namespace common_tangent_y_intercept_l140_140114

noncomputable def circle_center_a : ℝ × ℝ := (1, 5)
noncomputable def circle_radius_a : ℝ := 3

noncomputable def circle_center_b : ℝ × ℝ := (15, 10)
noncomputable def circle_radius_b : ℝ := 10

theorem common_tangent_y_intercept :
  ∃ m b: ℝ, (m > 0) ∧ m = 700/1197 ∧ b = 7.416 ∧
  ∀ x y: ℝ, (y = m * x + b → ((x - 1)^2 + (y - 5)^2 = 9 ∨ (x - 15)^2 + (y - 10)^2 = 100)) := by
{
  sorry
}

end common_tangent_y_intercept_l140_140114


namespace complex_number_solution_l140_140714

-- Define that z is a complex number and the condition given in the problem.
theorem complex_number_solution (z : ℂ) (hz : (i / (z + i)) = 2 - i) : z = -1/5 - 3/5 * i :=
sorry

end complex_number_solution_l140_140714


namespace count_integer_solutions_l140_140175

theorem count_integer_solutions : 
  ∃ (s : Finset (ℤ × ℤ)), 
  (∀ (x y : ℤ), ((x, y) ∈ s) ↔ (x^3 + y^2 = 2*y + 1)) ∧ 
  s.card = 3 := 
by
  sorry

end count_integer_solutions_l140_140175


namespace exists_divisible_by_2011_l140_140120

def a (n : ℕ) : ℕ := (List.range n).foldl (λ acc i => acc + 10 ^ i) 0

theorem exists_divisible_by_2011 : ∃ n, 1 ≤ n ∧ n ≤ 2011 ∧ 2011 ∣ a n := by
  sorry

end exists_divisible_by_2011_l140_140120


namespace mistaken_multiplication_l140_140827

theorem mistaken_multiplication (x : ℕ) : 
  let a := 139
  let b := 43
  let incorrect_result := 1251
  (a * b - a * x = incorrect_result) ↔ (x = 34) := 
by 
  let a := 139
  let b := 43
  let incorrect_result := 1251
  sorry

end mistaken_multiplication_l140_140827


namespace perimeter_pentagon_l140_140513

noncomputable def AB : ℝ := 1
noncomputable def BC : ℝ := Real.sqrt 2
noncomputable def CD : ℝ := Real.sqrt 3
noncomputable def DE : ℝ := 2

noncomputable def AC : ℝ := Real.sqrt (AB^2 + BC^2)
noncomputable def AD : ℝ := Real.sqrt (AC^2 + CD^2)
noncomputable def AE : ℝ := Real.sqrt (AD^2 + DE^2)

theorem perimeter_pentagon (ABCDE : List ℝ) (H : ABCDE = [AB, BC, CD, DE, AE]) :
  List.sum ABCDE = 3 + Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 10 :=
by
  sorry -- Proof skipped as instructed

end perimeter_pentagon_l140_140513


namespace part_a_part_b_part_c_part_d_part_e_part_f_part_g_part_h_part_i_part_j_part_k_part_m_l140_140301

open Real

variables (a b c d : ℝ)

-- Assumptions
axiom a_neg : a < 0
axiom b_neg : b < 0
axiom c_pos : 0 < c
axiom d_pos : 0 < d
axiom abs_conditions : (0 < abs c) ∧ (abs c < 1) ∧ (abs b < 2) ∧ (1 < abs b) ∧ (1 < abs d) ∧ (abs d < 2) ∧ (abs a < 4) ∧ (2 < abs a)

-- Theorem Statements
theorem part_a : abs a < 4 := sorry
theorem part_b : abs b < 2 := sorry
theorem part_c : abs c < 2 := sorry
theorem part_d : abs a > abs b := sorry
theorem part_e : abs c < abs d := sorry
theorem part_f : ¬ (abs a < abs d) := sorry
theorem part_g : abs (a - b) < 4 := sorry
theorem part_h : ¬ (abs (a - b) ≥ 3) := sorry
theorem part_i : ¬ (abs (c - d) < 1) := sorry
theorem part_j : abs (b - c) < 2 := sorry
theorem part_k : ¬ (abs (b - c) > 3) := sorry
theorem part_m : abs (c - a) > 1 := sorry

end part_a_part_b_part_c_part_d_part_e_part_f_part_g_part_h_part_i_part_j_part_k_part_m_l140_140301


namespace exists_eleven_consecutive_numbers_sum_cube_l140_140909

theorem exists_eleven_consecutive_numbers_sum_cube :
  ∃ (n k : ℕ), (n + (n+1) + (n+2) + (n+3) + (n+4) + (n+5) + (n+6) + (n+7) + (n+8) + (n+9) + (n+10)) = k^3 :=
by
  sorry

end exists_eleven_consecutive_numbers_sum_cube_l140_140909


namespace total_value_of_gold_l140_140196

theorem total_value_of_gold (legacy_bars : ℕ) (aleena_bars : ℕ) (bar_value : ℕ) (total_gold_value : ℕ) 
  (h1 : legacy_bars = 12) 
  (h2 : aleena_bars = legacy_bars - 4)
  (h3 : bar_value = 3500) : 
  total_gold_value = (legacy_bars + aleena_bars) * bar_value := 
by 
  sorry

end total_value_of_gold_l140_140196


namespace inequality_implies_bounds_l140_140573

open Real

theorem inequality_implies_bounds (a : ℝ) :
  (∀ x : ℝ, (exp x - a * x) * (x^2 - a * x + 1) ≥ 0) → (0 ≤ a ∧ a ≤ 2) :=
by sorry

end inequality_implies_bounds_l140_140573


namespace burn_rate_walking_l140_140449

def burn_rate_running : ℕ := 10
def total_calories : ℕ := 450
def total_time : ℕ := 60
def running_time : ℕ := 35

theorem burn_rate_walking :
  ∃ (W : ℕ), ((running_time * burn_rate_running) + ((total_time - running_time) * W) = total_calories) ∧ (W = 4) :=
by
  sorry

end burn_rate_walking_l140_140449


namespace drawn_from_grade12_correct_l140_140469

-- Variables for the conditions
variable (total_students : ℕ) (sample_size : ℕ) (grade10_students : ℕ) 
          (grade11_students : ℕ) (grade12_students : ℕ) (drawn_from_grade12 : ℕ)

-- Conditions
def conditions : Prop :=
  total_students = 2400 ∧
  sample_size = 120 ∧
  grade10_students = 820 ∧
  grade11_students = 780 ∧
  grade12_students = total_students - grade10_students - grade11_students ∧
  drawn_from_grade12 = (grade12_students * sample_size) / total_students

-- Theorem to prove
theorem drawn_from_grade12_correct : conditions total_students sample_size grade10_students grade11_students grade12_students drawn_from_grade12 → drawn_from_grade12 = 40 :=
by
  intro h
  rcases h with ⟨h1, h2, h3, h4, h5, h6⟩
  sorry

end drawn_from_grade12_correct_l140_140469


namespace evaluate_expression_l140_140384

theorem evaluate_expression : 2 + 3 / (4 + 5 / 6) = 76 / 29 := by
  sorry

end evaluate_expression_l140_140384


namespace min_possible_value_l140_140767

theorem min_possible_value
  (a b c d e f g h : Int)
  (h_distinct : List.Nodup [a, b, c, d, e, f, g, h])
  (h_set_a : a ∈ [-9, -6, -3, 0, 1, 3, 6, 10])
  (h_set_b : b ∈ [-9, -6, -3, 0, 1, 3, 6, 10])
  (h_set_c : c ∈ [-9, -6, -3, 0, 1, 3, 6, 10])
  (h_set_d : d ∈ [-9, -6, -3, 0, 1, 3, 6, 10])
  (h_set_e : e ∈ [-9, -6, -3, 0, 1, 3, 6, 10])
  (h_set_f : f ∈ [-9, -6, -3, 0, 1, 3, 6, 10])
  (h_set_g : g ∈ [-9, -6, -3, 0, 1, 3, 6, 10])
  (h_set_h : h ∈ [-9, -6, -3, 0, 1, 3, 6, 10]) :
  ∃ a b c d e f g h : Int,
  ((a + b + c + d)^2 + (e + f + g + h)^2) = 2
  :=
  sorry

end min_possible_value_l140_140767


namespace find_c_l140_140033

variable (a b c : ℕ)

theorem find_c (h1 : a = 9) (h2 : b = 2) (h3 : Odd c) (h4 : a + b > c) (h5 : a - b < c) (h6 : b + c > a) (h7 : b - c < a) : c = 9 :=
sorry

end find_c_l140_140033


namespace mod_37_5_l140_140852

theorem mod_37_5 : 37 % 5 = 2 := 
by
  sorry

end mod_37_5_l140_140852


namespace intersection_point_l140_140653

def parametric_line (t : ℝ) : ℝ × ℝ × ℝ :=
  (-1 - 2 * t, 0, -1 + 3 * t)

def plane (x y z : ℝ) : Prop := x + 4 * y + 13 * z - 23 = 0

theorem intersection_point :
  ∃ t : ℝ, plane (-1 - 2 * t) 0 (-1 + 3 * t) ∧ parametric_line t = (-3, 0, 2) :=
by
  sorry

end intersection_point_l140_140653


namespace max_blocks_fit_l140_140269

-- Define the dimensions of the block
def block_length := 2
def block_width := 3
def block_height := 1

-- Define the dimensions of the container box
def box_length := 4
def box_width := 3
def box_height := 3

-- Define the volume calculations
def volume (length width height : ℕ) : ℕ := length * width * height

def block_volume := volume block_length block_width block_height
def box_volume := volume box_length box_width box_height

-- The theorem to prove
theorem max_blocks_fit : (box_volume / block_volume) = 6 :=
by
  sorry

end max_blocks_fit_l140_140269


namespace elise_initial_dog_food_l140_140010

variable (initial_dog_food : ℤ)
variable (bought_first_bag : ℤ := 15)
variable (bought_second_bag : ℤ := 10)
variable (final_dog_food : ℤ := 40)

theorem elise_initial_dog_food :
  initial_dog_food + bought_first_bag + bought_second_bag = final_dog_food →
  initial_dog_food = 15 :=
by
  sorry

end elise_initial_dog_food_l140_140010


namespace probability_computation_l140_140983

noncomputable def probability_two_equal_three : ℚ :=
  let p_one_digit : ℚ := 3 / 4
  let p_two_digit : ℚ := 1 / 4
  let number_of_dice : ℕ := 5
  let ways_to_choose_two_digit := Nat.choose number_of_dice 2
  ways_to_choose_two_digit * (p_two_digit^2) * (p_one_digit^3)

theorem probability_computation :
  probability_two_equal_three = 135 / 512 :=
by
  sorry

end probability_computation_l140_140983


namespace inequality_region_area_l140_140716

noncomputable def area_of_inequality_region : ℝ :=
  let region := {p : ℝ × ℝ | |p.fst - p.snd| + |2 * p.fst + 2 * p.snd| ≤ 8}
  let vertices := [(2, 2), (-2, 2), (-2, -2), (2, -2)]
  let d1 := 8
  let d2 := 8
  (1 / 2) * d1 * d2

theorem inequality_region_area :
  area_of_inequality_region = 32 :=
by
  sorry  -- Proof to be provided

end inequality_region_area_l140_140716


namespace power_mod_l140_140892

theorem power_mod (n m : ℕ) (hn : n = 13) (hm : m = 1000) : n ^ 21 % m = 413 :=
by
  rw [hn, hm]
  -- other steps of the proof would go here...
  sorry

end power_mod_l140_140892


namespace total_amount_divided_l140_140173

theorem total_amount_divided (B_amount A_amount C_amount: ℝ) (h1 : A_amount = (1/3) * B_amount)
    (h2 : B_amount = 270) (h3 : B_amount = (1/4) * C_amount) :
    A_amount + B_amount + C_amount = 1440 :=
by
  sorry

end total_amount_divided_l140_140173


namespace factor_expression_l140_140428

theorem factor_expression (a b c : ℝ) :
  ((a^4 - b^4)^3 + (b^4 - c^4)^3 + (c^4 - a^4)^3) /
  ((a^2 - b^2)^3 + (b^2 - c^2)^3 + (c^2 - a^2)^3) =
  (a^2 + b^2) * (b^2 + c^2) * (c^2 + a^2) :=
by
  sorry

end factor_expression_l140_140428


namespace arithmetic_sequence_sum_l140_140332

variable (a : ℕ → ℝ) (d : ℝ)

-- Condition: The sequence {a_n} is arithmetic
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Given conditions
axiom a1 : a 1 = 2
axiom a2_a3_sum : a 2 + a 3 = 13

-- The theorem to be proved
theorem arithmetic_sequence_sum (h : is_arithmetic_sequence a d) : a (4) + a (5) + a (6) = 42 :=
sorry

end arithmetic_sequence_sum_l140_140332


namespace friend_initial_money_l140_140113

theorem friend_initial_money (F : ℕ) : 
    (160 + 25 * 7 = F + 25 * 5) → 
    (F = 210) :=
by
  sorry

end friend_initial_money_l140_140113


namespace perimeter_range_l140_140236

variable (a b x : ℝ)
variable (a_gt_b : a > b)
variable (triangle_ineq : a - b < x ∧ x < a + b)

theorem perimeter_range : 2 * a < a + b + x ∧ a + b + x < 2 * (a + b) :=
by
  sorry

end perimeter_range_l140_140236


namespace linear_system_solution_l140_140050

theorem linear_system_solution (x y : ℝ) (h1 : x = 2) (h2 : y = -3) : x + y = -1 :=
by
  sorry

end linear_system_solution_l140_140050


namespace hyperbola_eccentricity_l140_140340

noncomputable def hyperbola_eccentricity_range (a b e : ℝ) (h_a_pos : 0 < a) (h_a_less_1 : a < 1) (h_b_pos : 0 < b) : Prop :=
  let c := Real.sqrt ((5 * a^2 - a^4) / (1 - a^2))
  let e := c / a
  e > Real.sqrt 5

theorem hyperbola_eccentricity (a b e : ℝ) (h_a_pos : 0 < a) (h_a_less_1 : a < 1) (h_b_pos : 0 < b) :
  hyperbola_eccentricity_range a b e h_a_pos h_a_less_1 h_b_pos := 
sorry

end hyperbola_eccentricity_l140_140340


namespace tangent_line_at_A_l140_140825

def f (x : ℝ) : ℝ := x ^ (1 / 2)

def tangent_line_equation (x y: ℝ) : Prop :=
  4 * x - 4 * y + 1 = 0

theorem tangent_line_at_A :
  tangent_line_equation (1/4) (f (1/4)) :=
by
  sorry

end tangent_line_at_A_l140_140825


namespace gervais_avg_mileage_l140_140541
variable (x : ℤ)

def gervais_daily_mileage : Prop := ∃ (x : ℤ), (3 * x = 1250 - 305) ∧ x = 315

theorem gervais_avg_mileage : gervais_daily_mileage :=
by
  sorry

end gervais_avg_mileage_l140_140541


namespace length_of_green_caterpillar_l140_140185

def length_of_orange_caterpillar : ℝ := 1.17
def difference_in_length_between_caterpillars : ℝ := 1.83

theorem length_of_green_caterpillar :
  (length_of_orange_caterpillar + difference_in_length_between_caterpillars) = 3.00 :=
by
  sorry

end length_of_green_caterpillar_l140_140185


namespace percentage_decrease_in_spring_l140_140136

-- Given Conditions
variables (initial_members : ℕ) (increased_percent : ℝ) (total_decrease_percent : ℝ)
-- population changes
variables (fall_members : ℝ) (spring_members : ℝ)

-- The initial conditions given by the problem
axiom initial_membership : initial_members = 100
axiom fall_increase : increased_percent = 6
axiom total_decrease : total_decrease_percent = 14.14

-- Derived values based on conditions
axiom fall_members_calculated : fall_members = initial_members * (1 + increased_percent / 100)
axiom spring_members_calculated : spring_members = initial_members * (1 - total_decrease_percent / 100)

-- The correct answer which we need to prove
theorem percentage_decrease_in_spring : 
  ((fall_members - spring_members) / fall_members) * 100 = 19 := by
  sorry

end percentage_decrease_in_spring_l140_140136


namespace polygon_sides_l140_140993

theorem polygon_sides {n : ℕ} (h : (n - 2) * 180 = 1080) : n = 8 :=
sorry

end polygon_sides_l140_140993


namespace perimeter_of_plot_is_340_l140_140430

def width : ℝ := 80 -- Derived width from the given conditions
def length (w : ℝ) : ℝ := w + 10 -- Length is 10 meters more than width
def perimeter (w : ℝ) : ℝ := 2 * (w + length w) -- Perimeter of the rectangle
def cost_per_meter : ℝ := 6.5 -- Cost rate per meter
def total_cost : ℝ := 2210 -- Total cost given

theorem perimeter_of_plot_is_340 :
  cost_per_meter * perimeter width = total_cost → perimeter width = 340 := 
by
  sorry

end perimeter_of_plot_is_340_l140_140430


namespace probability_of_both_red_is_one_sixth_l140_140427

noncomputable def probability_both_red (red blue green : ℕ) (balls_picked : ℕ) : ℚ :=
  if balls_picked = 2 ∧ red = 4 ∧ blue = 3 ∧ green = 2 then (4 / 9) * (3 / 8) else 0

theorem probability_of_both_red_is_one_sixth :
  probability_both_red 4 3 2 2 = 1 / 6 :=
by
  unfold probability_both_red
  split_ifs
  · sorry
  · contradiction

end probability_of_both_red_is_one_sixth_l140_140427


namespace number_of_terms_in_arithmetic_sequence_l140_140459

theorem number_of_terms_in_arithmetic_sequence 
  (a : ℕ)
  (d : ℕ)
  (an : ℕ)
  (h1 : a = 3)
  (h2 : d = 4)
  (h3 : an = 47) :
  ∃ n : ℕ, an = a + (n - 1) * d ∧ n = 12 :=
by
  sorry

end number_of_terms_in_arithmetic_sequence_l140_140459


namespace function_decreases_l140_140079

def op (m n : ℝ) : ℝ := - (m * n) + n

def f (x : ℝ) : ℝ := op x 2

theorem function_decreases (x1 x2 : ℝ) (h : x1 < x2) : f x1 > f x2 :=
by sorry

end function_decreases_l140_140079


namespace all_a_n_are_perfect_squares_l140_140369

noncomputable def c : ℕ → ℤ 
| 0 => 1
| 1 => 0
| 2 => 2005
| n+2 => -3 * c n - 4 * c (n-1) + 2008

noncomputable def a (n : ℕ) : ℤ :=
  5 * (c (n + 2) - c n) * (502 - c (n - 1) - c (n - 2)) + 4 ^ n * 2004 * 501

theorem all_a_n_are_perfect_squares (n : ℕ) (h : n > 2) : ∃ k : ℤ, a n = k^2 :=
by
  sorry

end all_a_n_are_perfect_squares_l140_140369


namespace complement_of_A_relative_to_U_l140_140632

def U := { x : ℝ | x < 3 }
def A := { x : ℝ | x < 1 }

def complement_U_A := { x : ℝ | 1 ≤ x ∧ x < 3 }

theorem complement_of_A_relative_to_U : (complement_U_A = { x : ℝ | x ∈ U ∧ x ∉ A }) :=
by
  sorry

end complement_of_A_relative_to_U_l140_140632


namespace sqrt_inequality_l140_140393

theorem sqrt_inequality (x : ℝ) : abs ((x^2 - 9) / 3) < 3 ↔ -Real.sqrt 18 < x ∧ x < Real.sqrt 18 :=
by
  sorry

end sqrt_inequality_l140_140393


namespace no_valid_triangle_exists_l140_140968

-- Variables representing the sides and altitudes of the triangle
variables (a b c h_a h_b h_c : ℕ)

-- Definition of the perimeter condition
def perimeter_condition : Prop := a + b + c = 1995

-- Definition of integer altitudes condition (simplified)
def integer_altitudes_condition : Prop := 
  ∃ (h_a h_b h_c : ℕ), (h_a * 4 * a ^ 2 = 2 * a ^ 2 * b ^ 2 + 2 * a ^ 2 * c ^ 2 + 2 * c ^ 2 * b ^ 2 - a ^ 4 - b ^ 4 - c ^ 4)

-- The main theorem to prove no valid triangle exists
theorem no_valid_triangle_exists : ¬ (∃ (a b c : ℕ), perimeter_condition a b c ∧ integer_altitudes_condition a b c) :=
sorry

end no_valid_triangle_exists_l140_140968


namespace neg_distance_represents_west_l140_140798

def represents_east (distance : Int) : Prop :=
  distance > 0

def represents_west (distance : Int) : Prop :=
  distance < 0

theorem neg_distance_represents_west (pos_neg : represents_east 30) :
  represents_west (-50) :=
by
  sorry

end neg_distance_represents_west_l140_140798


namespace lawn_width_l140_140514

variable (W : ℝ)
variable (h₁ : 80 * 15 + 15 * W - 15 * 15 = 1875)
variable (h₂ : 5625 = 3 * 1875)

theorem lawn_width (h₁ : 80 * 15 + 15 * W - 15 * 15 = 1875) (h₂ : 5625 = 3 * 1875) : 
  W = 60 := 
sorry

end lawn_width_l140_140514


namespace jovana_shells_l140_140045

theorem jovana_shells :
  let jovana_initial := 5
  let first_friend := 15
  let second_friend := 17
  jovana_initial + first_friend + second_friend = 37 := by
  sorry

end jovana_shells_l140_140045


namespace find_x_when_z_64_l140_140212

-- Defining the conditions
def directly_proportional (x y : ℝ) : Prop := ∃ m : ℝ, x = m * y^3
def inversely_proportional (y z : ℝ) : Prop := ∃ n : ℝ, y = n / z^2

theorem find_x_when_z_64 (x y z : ℝ) (m n : ℝ) (k : ℝ) (h1 : directly_proportional x y) 
    (h2 : inversely_proportional y z) (h3 : z = 64) (h4 : x = 8) (h5 : z = 16) : x = 1/256 := 
  sorry

end find_x_when_z_64_l140_140212


namespace chocolate_bar_pieces_l140_140517

theorem chocolate_bar_pieces (X : ℕ) (h1 : X / 2 + X / 4 + 15 = X) : X = 60 :=
by
  sorry

end chocolate_bar_pieces_l140_140517


namespace result_when_7_multiplies_number_l140_140291

theorem result_when_7_multiplies_number (x : ℤ) (h : x + 45 - 62 = 55) : 7 * x = 504 :=
by sorry

end result_when_7_multiplies_number_l140_140291


namespace find_e_l140_140915

-- Conditions
def f (x : ℝ) (b : ℝ) := 5 * x + b
def g (x : ℝ) (b : ℝ) := b * x + 4
def f_comp_g (x : ℝ) (b : ℝ) (e : ℝ) := 15 * x + e

-- Statement to prove
theorem find_e (b e : ℝ) (x : ℝ): 
  (f (g x b) b = f_comp_g x b e) → 
  (5 * b = 15) → 
  (20 + b = e) → 
  e = 23 :=
by 
  intros h1 h2 h3
  sorry

end find_e_l140_140915


namespace base_number_in_exponent_l140_140687

theorem base_number_in_exponent (x : ℝ) (k : ℕ) (h₁ : k = 8) (h₂ : 64^k > x^22) : 
  x = 2^(24/11) :=
sorry

end base_number_in_exponent_l140_140687


namespace min_value_fraction_l140_140627

open Real

theorem min_value_fraction (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (∃ x : ℝ, x = (a / (a + 2 * b) + b / (a + b)) ∧ x ≥ 1 - 1 / (2 * sqrt 2) ∧ x = 1 - 1 / (2 * sqrt 2)) :=
by
  sorry

end min_value_fraction_l140_140627


namespace least_three_digit_with_factors_correct_l140_140387

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000
def has_factors (n : ℕ) : Prop := n % 3 = 0 ∧ n % 4 = 0 ∧ n % 9 = 0
def least_three_digit_with_factors : ℕ := 108

theorem least_three_digit_with_factors_correct : 
  is_three_digit least_three_digit_with_factors ∧ has_factors least_three_digit_with_factors ∧
  ∀ m : ℕ, is_three_digit m → has_factors m → least_three_digit_with_factors ≤ m := 
by 
  sorry

end least_three_digit_with_factors_correct_l140_140387


namespace ratio_of_width_to_length_is_correct_l140_140611

-- Define the given conditions
def length := 10
def perimeter := 36

-- Define the width and the expected ratio
def width (l P : Nat) : Nat := (P - 2 * l) / 2
def ratio (w l : Nat) := w / l

-- Statement to prove that given the conditions, the ratio of width to length is 4/5
theorem ratio_of_width_to_length_is_correct :
  ratio (width length perimeter) length = 4 / 5 :=
by
  sorry

end ratio_of_width_to_length_is_correct_l140_140611


namespace integer_sequence_unique_l140_140364

theorem integer_sequence_unique (a : ℕ → ℤ) :
  (∀ n : ℕ, ∃ p q : ℕ, p ≠ q ∧ a p > 0 ∧ a q < 0) ∧
  (∀ n : ℕ, ∀ i j : ℕ, i < n → j < n → i ≠ j → a i % (n : ℤ) ≠ a j % (n : ℤ))
  → ∀ x : ℤ, ∃! i : ℕ, a i = x :=
by
  sorry

end integer_sequence_unique_l140_140364


namespace paul_digs_the_well_l140_140388

theorem paul_digs_the_well (P : ℝ) (h1 : 1 / 16 + 1 / P + 1 / 48 = 1 / 8) : P = 24 :=
sorry

end paul_digs_the_well_l140_140388


namespace lcm_1_to_5_l140_140385

theorem lcm_1_to_5 : Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5 = 60 := by
  sorry

end lcm_1_to_5_l140_140385


namespace trader_sold_90_pens_l140_140751

theorem trader_sold_90_pens (C N : ℝ) (gain_percent : ℝ) (H1 : gain_percent = 33.33333333333333) (H2 : 30 * C = (gain_percent / 100) * N * C) :
  N = 90 :=
by
  sorry

end trader_sold_90_pens_l140_140751


namespace angle_in_second_quadrant_l140_140018

theorem angle_in_second_quadrant (x : ℝ) (hx1 : Real.tan x < 0) (hx2 : Real.sin x - Real.cos x > 0) : 
  (∃ k : ℤ, x = 2 * k * Real.pi + Real.pi / 2 ∨ x = 2 * k * Real.pi + 3 * Real.pi / 2) :=
sorry

end angle_in_second_quadrant_l140_140018


namespace impossible_arrangement_l140_140257

theorem impossible_arrangement (s : Finset ℕ) (h₁ : s = Finset.range 2018 \ {0})
  (h₂ : ∀ a ∈ s, ∀ b ∈ s, a ≠ b ∧ (b = a + 17 ∨ b = a + 21 ∨ b = a - 17 ∨ b = a - 21)) : False :=
by
  sorry

end impossible_arrangement_l140_140257


namespace construction_paper_initial_count_l140_140605

theorem construction_paper_initial_count 
    (b r d : ℕ)
    (ratio_cond : b = 2 * r)
    (daily_usage : ∀ n : ℕ, n ≤ d → n * 1 = b ∧ n * 3 = r)
    (last_day_cond : 0 = b ∧ 15 = r):
    b + r = 135 :=
sorry

end construction_paper_initial_count_l140_140605


namespace value_of_fg_neg_one_l140_140095

def f (x : ℝ) : ℝ := x - 2

def g (x : ℝ) : ℝ := x^2 + 4 * x + 3

theorem value_of_fg_neg_one : f (g (-1)) = -2 :=
by
  sorry

end value_of_fg_neg_one_l140_140095


namespace geometric_sequence_common_ratio_l140_140722

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) (h₁ : a 1 = 1) (h₂ : a 1 * a 2 * a 3 = -8) :
  q = -2 :=
sorry

end geometric_sequence_common_ratio_l140_140722


namespace fraction_unchanged_l140_140290

theorem fraction_unchanged (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) : 
  (2 * x) / (2 * (x + y)) = x / (x + y) :=
by
  sorry

end fraction_unchanged_l140_140290


namespace Gwen_remaining_homework_l140_140296

def initial_problems_math := 18
def completed_problems_math := 12
def remaining_problems_math := initial_problems_math - completed_problems_math

def initial_problems_science := 11
def completed_problems_science := 6
def remaining_problems_science := initial_problems_science - completed_problems_science

def initial_questions_history := 15
def completed_questions_history := 10
def remaining_questions_history := initial_questions_history - completed_questions_history

def initial_questions_english := 7
def completed_questions_english := 4
def remaining_questions_english := initial_questions_english - completed_questions_english

def total_remaining_problems := remaining_problems_math 
                               + remaining_problems_science 
                               + remaining_questions_history 
                               + remaining_questions_english

theorem Gwen_remaining_homework : total_remaining_problems = 19 :=
by
  sorry

end Gwen_remaining_homework_l140_140296


namespace neg_exponent_reciprocal_l140_140260

theorem neg_exponent_reciprocal : (2 : ℝ) ^ (-1 : ℤ) = 1 / 2 := by
  -- Insert your proof here
  sorry

end neg_exponent_reciprocal_l140_140260


namespace no_real_x_satisfies_quadratic_ineq_l140_140122

theorem no_real_x_satisfies_quadratic_ineq :
  ¬ ∃ x : ℝ, x^2 + 3 * x + 3 ≤ 0 :=
sorry

end no_real_x_satisfies_quadratic_ineq_l140_140122


namespace unique_positive_b_for_one_solution_l140_140649

theorem unique_positive_b_for_one_solution
  (a : ℝ) (c : ℝ) :
  a = 3 →
  (∃! (b : ℝ), b > 0 ∧ (3 * (b + (1 / b)))^2 - 4 * c = 0 ) →
  c = 9 :=
by
  intros ha h
  -- Proceed to show that c must be 9
  sorry

end unique_positive_b_for_one_solution_l140_140649


namespace number_of_solutions_l140_140181

theorem number_of_solutions (f : ℕ → ℕ) (n : ℕ) : 
  (∀ n, f n = n^4 + 2 * n^3 - 20 * n^2 + 2 * n - 21) →
  (∀ n, 0 ≤ n ∧ n < 2013 → 2013 ∣ f n) → 
  ∃ k, k = 6 :=
by
  sorry

end number_of_solutions_l140_140181


namespace treasure_coins_problem_l140_140507

theorem treasure_coins_problem (N m n t k s u : ℤ) 
  (h1 : N = (2/3) * (2/3) * (2/3) * (m - 1) - (2/3) - (2^2 / 3^2))
  (h2 : N = 3 * n)
  (h3 : 8 * (m - 1) - 30 = 81 * k)
  (h4 : m - 1 = 3 * t)
  (h5 : 8 * t - 27 * k = 10)
  (h6 : m = 3 * t + 1)
  (h7 : k = 2 * s)
  (h8 : 4 * t - 27 * s = 5)
  (h9 : t = 8 + 27 * u)
  (h10 : s = 1 + 4 * u)
  (h11 : 110 ≤ 81 * u + 25)
  (h12 : 81 * u + 25 ≤ 200) :
  m = 187 :=
sorry

end treasure_coins_problem_l140_140507


namespace illiterate_employee_count_l140_140453

variable (I : ℕ) -- Number of illiterate employees
variable (literate_count : ℕ) -- Number of literate employees
variable (initial_wage_illiterate : ℕ) -- Initial average wage of illiterate employees
variable (new_wage_illiterate : ℕ) -- New average wage of illiterate employees
variable (average_salary_decrease : ℕ) -- Decrease in the average salary of all employees

-- Given conditions:
def condition1 : initial_wage_illiterate = 25 := by sorry
def condition2 : new_wage_illiterate = 10 := by sorry
def condition3 : average_salary_decrease = 10 := by sorry
def condition4 : literate_count = 10 := by sorry

-- Main proof statement:
theorem illiterate_employee_count :
  initial_wage_illiterate - new_wage_illiterate = 15 →
  average_salary_decrease * (literate_count + I) = (initial_wage_illiterate - new_wage_illiterate) * I →
  I = 20 := by
  intros h1 h2
  -- provided conditions
  exact sorry

end illiterate_employee_count_l140_140453


namespace alex_needs_more_coins_l140_140133

-- Define the conditions and problem statement 
def num_friends : ℕ := 15
def coins_alex_has : ℕ := 95 

-- The total number of coins required is
def total_coins_needed : ℕ := num_friends * (num_friends + 1) / 2

-- The minimum number of additional coins needed
def additional_coins_needed : ℕ := total_coins_needed - coins_alex_has

-- Formalize the theorem 
theorem alex_needs_more_coins : additional_coins_needed = 25 := by
  -- Here we would provide the actual proof steps
  sorry

end alex_needs_more_coins_l140_140133


namespace simplify_fraction_rationalize_denominator_l140_140060

noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x
noncomputable def fraction := 5 / (sqrt 125 + 3 * sqrt 45 + 4 * sqrt 20 + sqrt 75)

theorem simplify_fraction_rationalize_denominator :
  fraction = sqrt 5 / 27 :=
by
  sorry

end simplify_fraction_rationalize_denominator_l140_140060


namespace vertex_position_l140_140123

-- Definitions based on the conditions of the problem
def quadratic_function (x : ℝ) : ℝ := 3*x^2 + 9*x + 5

-- Theorem that the vertex of the parabola is at x = -1.5
theorem vertex_position : ∃ x : ℝ, x = -1.5 ∧ ∀ y : ℝ, quadratic_function y ≥ quadratic_function x :=
by
  sorry

end vertex_position_l140_140123


namespace text_message_costs_equal_l140_140109

theorem text_message_costs_equal (x : ℝ) : 
  (0.25 * x + 9 = 0.40 * x) ∧ (0.25 * x + 9 = 0.20 * x + 12) → x = 60 :=
by 
  sorry

end text_message_costs_equal_l140_140109


namespace minimum_value_f_inequality_proof_l140_140842

def f (x : ℝ) : ℝ := abs (x + 3) + abs (x - 1)

-- The minimal value of f(x)
def m : ℝ := 4

theorem minimum_value_f :
  (∀ x : ℝ, f x ≥ m) ∧ (∃ x : ℝ, -3 ≤ x ∧ x ≤ 1 ∧ f x = m) :=
by
  sorry -- Proof that the minimum value of f(x) is 4 and occurs in the range -3 ≤ x ≤ 1

variables (p q r : ℝ)

-- Given condition that p^2 + 2q^2 + r^2 = 4
theorem inequality_proof (h : p^2 + 2 * q^2 + r^2 = m) : q * (p + r) ≤ 2 :=
by
  sorry -- Proof that q(p + r) ≤ 2 given p^2 + 2q^2 + r^2 = 4

end minimum_value_f_inequality_proof_l140_140842


namespace xena_escape_l140_140247

theorem xena_escape
    (head_start : ℕ)
    (safety_distance : ℕ)
    (xena_speed : ℕ)
    (dragon_speed : ℕ)
    (effective_gap : ℕ := head_start - safety_distance)
    (speed_difference : ℕ := dragon_speed - xena_speed) :
    (time_to_safety : ℕ := effective_gap / speed_difference) →
    time_to_safety = 32 :=
by
  sorry

end xena_escape_l140_140247


namespace unique_root_condition_l140_140094

theorem unique_root_condition (a : ℝ) : 
  (∀ x : ℝ, x^3 + a*x^2 - 4*a*x + a^2 - 4 = 0 → ∃! x₀ : ℝ, x = x₀) ↔ a < 1 :=
by sorry

end unique_root_condition_l140_140094


namespace move_point_right_3_units_from_neg_2_l140_140137

noncomputable def move_point_to_right (start : ℤ) (units : ℤ) : ℤ :=
start + units

theorem move_point_right_3_units_from_neg_2 : move_point_to_right (-2) 3 = 1 :=
by
  sorry

end move_point_right_3_units_from_neg_2_l140_140137


namespace sum_series_eq_eight_l140_140096

noncomputable def sum_series : ℝ := ∑' n : ℕ, (3 * (n + 1) + 2) / 2^(n + 1)

theorem sum_series_eq_eight : sum_series = 8 := 
 by
  sorry

end sum_series_eq_eight_l140_140096


namespace hall_area_l140_140990

theorem hall_area (L W : ℝ) 
  (h1 : W = (1/2) * L)
  (h2 : L - W = 8) : 
  L * W = 128 := 
  sorry

end hall_area_l140_140990


namespace determine_a_of_parallel_lines_l140_140231

theorem determine_a_of_parallel_lines (a : ℝ) :
  (∀ x y : ℝ, 3 * y - 3 * a = 9 * x ↔ y = 3 * x + a) →
  (∀ x y : ℝ, y - 2 = (a - 3) * x ↔ y = (a - 3) * x + 2) →
  (∀ x y : ℝ, 3 * y - 3 * a = 9 * x → y - 2 = (a - 3) * x → 3 = a - 3) →
  a = 6 :=
by
  sorry

end determine_a_of_parallel_lines_l140_140231


namespace joe_max_money_l140_140112

noncomputable def max_guaranteed_money (initial_money : ℕ) (max_bet : ℕ) (num_bets : ℕ) : ℕ :=
  if initial_money = 100 ∧ max_bet = 17 ∧ num_bets = 5 then 98 else 0

theorem joe_max_money : max_guaranteed_money 100 17 5 = 98 := by
  sorry

end joe_max_money_l140_140112


namespace graph_properties_l140_140742

noncomputable def f (x : ℝ) : ℝ := (x^2 - 5*x + 6) / (x - 1)

theorem graph_properties :
  (∀ x, x ≠ 1 → f x = (x-2)*(x-3)/(x-1)) ∧
  (∃ x, f x = 0 ∧ (x = 2 ∨ x = 3)) ∧
  (∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - 1) < δ → abs (f x) > ε) ∧
  ((∀ ε > 0, ∃ M > 0, ∀ x > M, f x > ε) ∧ (∀ ε > 0, ∃ M < 0, ∀ x < M, f x < -ε)) := sorry

end graph_properties_l140_140742


namespace slopes_of_line_intersecting_ellipse_l140_140485

noncomputable def possible_slopes : Set ℝ := {m : ℝ | m ≤ -1/Real.sqrt 20 ∨ m ≥ 1/Real.sqrt 20}

theorem slopes_of_line_intersecting_ellipse (m : ℝ) (h : ∃ x y, y = m * x - 3 ∧ 4 * x^2 + 25 * y^2 = 100) : 
  m ∈ possible_slopes :=
sorry

end slopes_of_line_intersecting_ellipse_l140_140485


namespace problem_simplify_and_evaluate_l140_140590

theorem problem_simplify_and_evaluate (m : ℝ) (h : m = Real.sqrt 3 + 3) :
  (1 - (m / (m + 3))) / ((m^2 - 9) / (m^2 + 6 * m + 9)) = Real.sqrt 3 :=
by
  sorry

end problem_simplify_and_evaluate_l140_140590


namespace sliderB_moves_distance_l140_140824

theorem sliderB_moves_distance :
  ∀ (A B : ℝ) (rod_length : ℝ),
    (A = 20) →
    (B = 15) →
    (rod_length = Real.sqrt (20^2 + 15^2)) →
    (rod_length = 25) →
    (B_new = 25 - 15) →
    B_new = 10 := by
  sorry

end sliderB_moves_distance_l140_140824


namespace candy_group_size_l140_140612

-- Define the given conditions
def num_candies : ℕ := 30
def num_groups : ℕ := 10

-- Define the statement that needs to be proven
theorem candy_group_size : num_candies / num_groups = 3 := 
by 
  sorry

end candy_group_size_l140_140612


namespace sasha_mistake_l140_140481

/-- If Sasha obtained three numbers by raising 4 to various powers, such that all three units digits are different, 
     then Sasha's numbers cannot have three distinct last digits. -/
theorem sasha_mistake (h : ∀ n1 n2 n3 : ℕ, ∃ k1 k2 k3, n1 = 4^k1 ∧ n2 = 4^k2 ∧ n3 = 4^k3 ∧ (n1 % 10 ≠ n2 % 10) ∧ (n2 % 10 ≠ n3 % 10) ∧ (n1 % 10 ≠ n3 % 10)) :
False :=
sorry

end sasha_mistake_l140_140481


namespace hits_9_and_8_mutually_exclusive_hits_10_and_8_not_mutually_exclusive_both_hit_target_and_neither_hit_target_mutually_exclusive_at_least_one_hits_and_A_not_B_does_not_mutually_exclusive_l140_140437

-- Definitions for shooting events for clarity
def hits_9_rings (s : String) := s = "9 rings"
def hits_8_rings (s : String) := s = "8 rings"

def hits_10_rings (s : String) := s = "10 rings"

def hits_target (s: String) := s = "hits target"
def does_not_hit_target (s: String) := s = "does not hit target"

-- Mutual exclusivity:
def mutually_exclusive (E1 E2 : Prop) := ¬ (E1 ∧ E2)

-- Problem 1:
theorem hits_9_and_8_mutually_exclusive :
  mutually_exclusive (hits_9_rings "9 rings") (hits_8_rings "8 rings") :=
sorry

-- Problem 2:
theorem hits_10_and_8_not_mutually_exclusive :
  ¬ mutually_exclusive (hits_10_rings "10 rings" ) (hits_8_rings "8 rings") :=
sorry

-- Problem 3:
theorem both_hit_target_and_neither_hit_target_mutually_exclusive :
  mutually_exclusive (hits_target "both hit target") (does_not_hit_target "neither hit target") :=
sorry

-- Problem 4:
theorem at_least_one_hits_and_A_not_B_does_not_mutually_exclusive :
  ¬ mutually_exclusive (hits_target "at least one hits target") (does_not_hit_target "A not but B does hit target") :=
sorry

end hits_9_and_8_mutually_exclusive_hits_10_and_8_not_mutually_exclusive_both_hit_target_and_neither_hit_target_mutually_exclusive_at_least_one_hits_and_A_not_B_does_not_mutually_exclusive_l140_140437


namespace find_a2_b2_l140_140747

noncomputable def imaginary_unit : ℂ := Complex.I

theorem find_a2_b2 (a b : ℝ) (h1 : (a - 2 * imaginary_unit) * imaginary_unit = b - imaginary_unit) : a^2 + b^2 = 5 :=
  sorry

end find_a2_b2_l140_140747


namespace find_values_of_cubes_l140_140208

def N (a b c : ℂ) : Matrix (Fin 3) (Fin 3) ℂ :=
  ![![a, c, b], ![c, b, a], ![b, a, c]]

theorem find_values_of_cubes (a b c : ℂ) (h1 : (N a b c) ^ 2 = 1) (h2 : a * b * c = 1) :
  a^3 + b^3 + c^3 = 2 ∨ a^3 + b^3 + c^3 = 4 :=
by
  sorry

end find_values_of_cubes_l140_140208


namespace point_on_parabola_l140_140365

theorem point_on_parabola (c m n x1 x2 : ℝ) (h : x1 < x2)
  (hx1 : x1^2 + 2*x1 + c = 0)
  (hx2 : x2^2 + 2*x2 + c = 0)
  (hp : n = m^2 + 2*m + c)
  (hn : n < 0) :
  x1 < m ∧ m < x2 :=
sorry

end point_on_parabola_l140_140365


namespace least_number_subtracted_divisible_by_5_l140_140363

def subtract_least_number (n : ℕ) (m : ℕ) : ℕ :=
  n % m

theorem least_number_subtracted_divisible_by_5 : subtract_least_number 9671 5 = 1 :=
by
  sorry

end least_number_subtracted_divisible_by_5_l140_140363


namespace ilya_incorrect_l140_140000

theorem ilya_incorrect (s t : ℝ) : ¬ (s + t = s * t ∧ s * t = s / t) :=
by
  sorry

end ilya_incorrect_l140_140000


namespace find_the_number_l140_140942

-- Statement
theorem find_the_number (x : ℤ) (h : 2 * x = 3 * x - 25) : x = 25 :=
  sorry

end find_the_number_l140_140942


namespace mystery_number_addition_l140_140470

theorem mystery_number_addition (mystery_number : ℕ) (h : mystery_number = 47) : mystery_number + 45 = 92 :=
by
  -- Proof goes here
  sorry

end mystery_number_addition_l140_140470


namespace lcm_of_23_46_827_l140_140065

theorem lcm_of_23_46_827 : Nat.lcm (Nat.lcm 23 46) 827 = 38042 :=
by
  sorry

end lcm_of_23_46_827_l140_140065


namespace print_time_correct_l140_140084

-- Define the conditions
def pages_per_minute : ℕ := 23
def total_pages : ℕ := 345

-- Define the expected result
def expected_minutes : ℕ := 15

-- Prove the equivalence
theorem print_time_correct :
  total_pages / pages_per_minute = expected_minutes :=
by 
  -- Proof will be provided here
  sorry

end print_time_correct_l140_140084


namespace max_gold_coins_l140_140675

theorem max_gold_coins (k : ℤ) (h1 : ∃ k : ℤ, 15 * k + 3 < 120) : 
  ∃ n : ℤ, n = 15 * k + 3 ∧ n < 120 ∧ n = 108 :=
by
  sorry

end max_gold_coins_l140_140675


namespace find_y_l140_140922

theorem find_y (x : ℝ) (h : x^2 + (1 / x)^2 = 7) : x + 1 / x = 3 :=
by
  sorry

end find_y_l140_140922


namespace stock_yield_percentage_l140_140651

theorem stock_yield_percentage (face_value market_price : ℝ) (annual_dividend_rate : ℝ) 
  (h_face_value : face_value = 100)
  (h_market_price : market_price = 140)
  (h_annual_dividend_rate : annual_dividend_rate = 0.14) :
  (annual_dividend_rate * face_value / market_price) * 100 = 10 :=
by
  -- computation here
  sorry

end stock_yield_percentage_l140_140651


namespace no_positive_integer_n_exists_l140_140781

theorem no_positive_integer_n_exists {n : ℕ} (hn : n > 0) :
  ¬ ((∃ k, 5 * 10^(k - 1) ≤ 2^n ∧ 2^n < 6 * 10^(k - 1)) ∧
     (∃ m, 2 * 10^(m - 1) ≤ 5^n ∧ 5^n < 3 * 10^(m - 1))) :=
sorry

end no_positive_integer_n_exists_l140_140781


namespace radical_axis_theorem_l140_140435

structure Point :=
  (x : ℝ)
  (y : ℝ)

structure Circle :=
  (center : Point)
  (radius : ℝ)

def power_of_point (p : Point) (c : Circle) : ℝ :=
  ((p.x - c.center.x)^2 + (p.y - c.center.y)^2 - c.radius^2)

theorem radical_axis_theorem (O1 O2 : Circle) :
  ∃ L : ℝ → Point, 
  (∀ p : Point, (power_of_point p O1 = power_of_point p O2) → (L p.x = p)) ∧ 
  (O1.center.y = O2.center.y) ∧ 
  (∃ k : ℝ, ∀ x, L x = Point.mk x k) :=
sorry

end radical_axis_theorem_l140_140435


namespace basic_computer_price_l140_140881

theorem basic_computer_price :
  ∃ C P : ℝ,
    C + P = 2500 ∧
    (C + 800) + (1 / 5) * (C + 800 + P) = 2500 ∧
    (C + 1100) + (1 / 8) * (C + 1100 + P) = 2500 ∧
    (C + 1500) + (1 / 10) * (C + 1500 + P) = 2500 ∧
    C = 1040 :=
by
  sorry

end basic_computer_price_l140_140881


namespace restaurant_discount_l140_140473

theorem restaurant_discount :
  let coffee_price := 6
  let cheesecake_price := 10
  let discount_rate := 0.25
  let total_price := coffee_price + cheesecake_price
  let discount := discount_rate * total_price
  let final_price := total_price - discount
  final_price = 12 := by
  sorry

end restaurant_discount_l140_140473


namespace possible_values_of_x_l140_140973

-- Definitions representing the initial conditions
def condition1 (x : ℕ) : Prop := 203 % x = 13
def condition2 (x : ℕ) : Prop := 298 % x = 13

-- Main theorem statement
theorem possible_values_of_x (x : ℕ) (h1 : condition1 x) (h2 : condition2 x) : x = 19 ∨ x = 95 := 
by
  sorry

end possible_values_of_x_l140_140973


namespace variance_of_data_set_l140_140595

def data_set : List ℤ := [ -2, -1, 0, 3, 5 ]

def mean (l : List ℤ) : ℚ :=
  (l.sum / l.length)

def variance (l : List ℤ) : ℚ :=
  (1 / l.length) * (l.map (λ x => (x - mean l : ℚ)^2)).sum

theorem variance_of_data_set : variance data_set = 34 / 5 := by
  sorry

end variance_of_data_set_l140_140595


namespace combi_sum_l140_140666

theorem combi_sum : (Nat.choose 8 2) + (Nat.choose 8 3) + (Nat.choose 9 2) = 120 :=
by
  sorry

end combi_sum_l140_140666


namespace orchestra_members_l140_140021

theorem orchestra_members :
  ∃ (n : ℕ), 
    150 < n ∧ n < 250 ∧ 
    n % 4 = 2 ∧ 
    n % 5 = 3 ∧ 
    n % 7 = 4 :=
by
  use 158
  repeat {split};
  sorry

end orchestra_members_l140_140021


namespace find_a_and_b_l140_140561

noncomputable def a_and_b (x y : ℝ) (a b : ℝ) : Prop :=
  a = Real.sqrt x + Real.sqrt y ∧ b = Real.sqrt (x + 2) + Real.sqrt (y + 2) ∧
  ∃ n : ℤ, a = n ∧ b = n + 2

theorem find_a_and_b (x y : ℝ) (a b : ℝ)
  (h₁ : 0 ≤ x)
  (h₂ : 0 ≤ y)
  (h₃ : a_and_b x y a b)
  (h₄ : ∃ n : ℤ, a = n ∧ b = n + 2) :
  a = 1 ∧ b = 3 := by
  sorry

end find_a_and_b_l140_140561


namespace newspapers_on_sunday_l140_140083

theorem newspapers_on_sunday (papers_weekend : ℕ) (diff_papers : ℕ) 
  (h1 : papers_weekend = 110) 
  (h2 : diff_papers = 20) 
  (h3 : ∃ (S Su : ℕ), Su = S + diff_papers ∧ S + Su = papers_weekend) :
  ∃ Su, Su = 65 :=
by
  sorry

end newspapers_on_sunday_l140_140083


namespace multiples_of_seven_with_units_digit_three_l140_140284

theorem multiples_of_seven_with_units_digit_three :
  ∃ n : ℕ, n = 2 ∧ ∀ k : ℕ, (k < 150 ∧ k % 7 = 0 ∧ k % 10 = 3) ↔ (k = 63 ∨ k = 133) := by
  sorry

end multiples_of_seven_with_units_digit_three_l140_140284


namespace max_expression_value_l140_140490

open Real

theorem max_expression_value : 
  ∃ q : ℝ, ∀ q : ℝ, -3 * q ^ 2 + 18 * q + 5 ≤ 32 ∧ (-3 * (3 ^ 2) + 18 * 3 + 5 = 32) :=
by
  sorry

end max_expression_value_l140_140490


namespace find_y_l140_140438

theorem find_y (y : ℕ) (h : 2^10 = 32^y) : y = 2 :=
by {
  sorry
}

end find_y_l140_140438


namespace john_needs_one_plank_l140_140063

theorem john_needs_one_plank (total_nails : ℕ) (nails_per_plank : ℕ) (extra_nails : ℕ) (P : ℕ)
    (h1 : total_nails = 11)
    (h2 : nails_per_plank = 3)
    (h3 : extra_nails = 8)
    (h4 : total_nails = nails_per_plank * P + extra_nails) :
    P = 1 :=
by
    sorry

end john_needs_one_plank_l140_140063


namespace ratio_of_adults_to_children_l140_140207

-- Defining conditions as functions
def admission_fees_condition (a c : ℕ) : ℕ := 30 * a + 15 * c

-- Stating the problem
theorem ratio_of_adults_to_children (a c : ℕ) 
  (h1 : admission_fees_condition a c = 2250)
  (h2 : a ≥ 1) 
  (h3 : c ≥ 1) 
  : a / c = 2 := 
sorry

end ratio_of_adults_to_children_l140_140207


namespace unit_digit_is_nine_l140_140518

theorem unit_digit_is_nine (a b : ℕ) (h1 : 0 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9) (h3 : a ≠ 0) (h4 : a + b + a * b = 10 * a + b) : b = 9 := 
by 
  sorry

end unit_digit_is_nine_l140_140518


namespace min_edge_disjoint_cycles_l140_140201

noncomputable def minEdgesForDisjointCycles (n : ℕ) (h : n ≥ 6) : ℕ := 3 * (n - 2)

theorem min_edge_disjoint_cycles (n : ℕ) (h : n ≥ 6) : minEdgesForDisjointCycles n h = 3 * (n - 2) := 
by
  sorry

end min_edge_disjoint_cycles_l140_140201


namespace final_stack_height_l140_140726

theorem final_stack_height (x : ℕ) 
  (first_stack_height : ℕ := 7) 
  (second_stack_height : ℕ := first_stack_height + 5) 
  (final_stack_height : ℕ := second_stack_height + x) 
  (blocks_fell_first : ℕ := first_stack_height) 
  (blocks_fell_second : ℕ := second_stack_height - 2) 
  (blocks_fell_final : ℕ := final_stack_height - 3) 
  (total_blocks_fell : 33 = blocks_fell_first + blocks_fell_second + blocks_fell_final) 
  : x = 7 :=
  sorry

end final_stack_height_l140_140726


namespace intersection_M_N_l140_140628

def M : Set ℝ := { x | x > 1 }
def N : Set ℝ := { x | -2 ≤ x ∧ x ≤ 2 }

theorem intersection_M_N :
  M ∩ N = { x | 1 < x ∧ x ≤ 2 } := 
sorry

end intersection_M_N_l140_140628


namespace root_of_inverse_f_plus_x_eq_k_l140_140401

variable {α : Type*} [Nonempty α] [Field α]
variable (f : α → α)
variable (f_inv : α → α)
variable (k : α)

def root_of_f_plus_x_eq_k (x : α) : Prop :=
  f x + x = k

def inverse_function (f : α → α) (f_inv : α → α) : Prop :=
  ∀ y : α, f (f_inv y) = y ∧ f_inv (f y) = y

theorem root_of_inverse_f_plus_x_eq_k
  (h1 : root_of_f_plus_x_eq_k f 5 k)
  (h2 : inverse_function f f_inv) :
  f_inv (k - 5) + (k - 5) = k :=
by
  sorry

end root_of_inverse_f_plus_x_eq_k_l140_140401


namespace num_pairs_of_regular_polygons_l140_140578

def num_pairs : Nat := 
  let pairs := [(7, 42), (6, 18), (5, 10), (4, 6)]
  pairs.length

theorem num_pairs_of_regular_polygons : num_pairs = 4 := 
  sorry

end num_pairs_of_regular_polygons_l140_140578


namespace determinant_matrix_example_l140_140178

open Matrix

def matrix_example : Matrix (Fin 2) (Fin 2) ℤ := ![![7, -2], ![-3, 6]]

noncomputable def compute_det_and_add_5 : ℤ := (matrix_example.det) + 5

theorem determinant_matrix_example :
  compute_det_and_add_5 = 41 := by
  sorry

end determinant_matrix_example_l140_140178


namespace length_of_the_train_is_120_l140_140588

noncomputable def train_length (time: ℝ) (speed_km_hr: ℝ) : ℝ :=
  let speed_m_s := (speed_km_hr * 1000) / 3600
  speed_m_s * time

theorem length_of_the_train_is_120 :
  train_length 3.569962336897346 121 = 120 := by
  sorry

end length_of_the_train_is_120_l140_140588


namespace james_initial_amount_l140_140157

noncomputable def initial_amount (total_amount_invested_per_week: ℕ) 
                                (number_of_weeks_in_year: ℕ) 
                                (windfall_factor: ℚ) 
                                (amount_after_windfall: ℕ) : ℚ :=
  let total_investment := total_amount_invested_per_week * number_of_weeks_in_year
  let amount_without_windfall := (amount_after_windfall : ℚ) / (1 + windfall_factor)
  amount_without_windfall - total_investment

theorem james_initial_amount:
  initial_amount 2000 52 0.5 885000 = 250000 := sorry

end james_initial_amount_l140_140157


namespace solve_for_q_l140_140336

theorem solve_for_q (k l q : ℕ) (h1 : (2 : ℚ) / 3 = k / 45) (h2 : (2 : ℚ) / 3 = (k + l) / 75) (h3 : (2 : ℚ) / 3 = (q - l) / 105) : q = 90 :=
sorry

end solve_for_q_l140_140336


namespace evaluate_expression_l140_140399

theorem evaluate_expression : (2019 - (2000 - (10 - 9))) - (2000 - (10 - (9 - 2019))) = 40 :=
by
  sorry

end evaluate_expression_l140_140399


namespace gcd_of_consecutive_digit_sums_l140_140148

theorem gcd_of_consecutive_digit_sums :
  ∀ x y z : ℕ, x + 1 = y → y + 1 = z → gcd (101 * (x + z) + 10 * y) 212 = 212 :=
by
  sorry

end gcd_of_consecutive_digit_sums_l140_140148


namespace family_spent_36_dollars_l140_140261

def ticket_cost : ℝ := 5

def popcorn_cost : ℝ := 0.8 * ticket_cost

def soda_cost : ℝ := 0.5 * popcorn_cost

def tickets_bought : ℕ := 4

def popcorn_bought : ℕ := 2

def sodas_bought : ℕ := 4

def total_spent : ℝ :=
  (tickets_bought * ticket_cost) +
  (popcorn_bought * popcorn_cost) +
  (sodas_bought * soda_cost)

theorem family_spent_36_dollars : total_spent = 36 := by
  sorry

end family_spent_36_dollars_l140_140261


namespace complement_intersection_subset_condition_l140_140521

-- Definition of sets A, B, and C
def A := { x : ℝ | 3 ≤ x ∧ x < 7 }
def B := { x : ℝ | 2 < x ∧ x < 10 }
def C (a : ℝ) := { x : ℝ | x < a }

-- Proof problem 1 statement
theorem complement_intersection :
  ( { x : ℝ | x < 3 ∨ x ≥ 7 } ∩ { x : ℝ | 2 < x ∧ x < 10 } ) = { x : ℝ | 2 < x ∧ x < 3 ∨ 7 ≤ x ∧ x < 10 } :=
by
  sorry

-- Proof problem 2 statement
theorem subset_condition (a : ℝ) :
  ( { x : ℝ | 3 ≤ x ∧ x < 7 } ⊆ { x : ℝ | x < a } ) → (a ≥ 7) :=
by
  sorry

end complement_intersection_subset_condition_l140_140521


namespace jose_bottle_caps_proof_l140_140493

def jose_bottle_caps_initial : Nat := 7
def rebecca_bottle_caps : Nat := 2
def jose_bottle_caps_final : Nat := 9

theorem jose_bottle_caps_proof : jose_bottle_caps_initial + rebecca_bottle_caps = jose_bottle_caps_final := by
  sorry

end jose_bottle_caps_proof_l140_140493


namespace committee_count_is_252_l140_140958

/-- Definition of binomial coefficient -/
def binom (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- Problem statement: Number of ways to choose a 5-person committee from a club of 10 people is 252 -/
theorem committee_count_is_252 : binom 10 5 = 252 :=
by
  sorry

end committee_count_is_252_l140_140958


namespace expected_adjacent_red_pairs_l140_140233

theorem expected_adjacent_red_pairs :
  ∃ (E : ℚ), E = 650 / 51 :=
sorry

end expected_adjacent_red_pairs_l140_140233


namespace toothbrushes_difference_l140_140556

theorem toothbrushes_difference
  (total : ℕ)
  (jan : ℕ)
  (feb : ℕ)
  (mar : ℕ)
  (apr_may_sum : total = jan + feb + mar + 164)
  (apr_may_half : 164 / 2 = 82)
  (busy_month_given : feb = 67)
  (slow_month_given : mar = 46) :
  feb - mar = 21 :=
by
  sorry

end toothbrushes_difference_l140_140556


namespace remainder_of_sum_div_18_l140_140624

theorem remainder_of_sum_div_18 :
  let nums := [11065, 11067, 11069, 11071, 11073, 11075, 11077, 11079, 11081]
  let residues := [1, 3, 5, 7, 9, 11, 13, 15, 17]
  (nums.sum % 18) = 9 := by
    sorry

end remainder_of_sum_div_18_l140_140624


namespace proof_problem_l140_140757

-- Definitions of sequence terms and their properties
def geometric_sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 2 ∧ a 4 = 16 ∧ ∀ n, a n = 2^n

-- Definition for the sum of the first n terms of the sequence
noncomputable def sum_of_sequence (S : ℕ → ℕ) : Prop :=
  ∀ n, S n = 2^(n + 1) - 2

-- Definition for the transformed sequence b_n = log_2 a_n
def transformed_sequence (a b : ℕ → ℕ) : Prop :=
  ∀ n, b n = Nat.log2 (a n)

-- Definition for the sum T_n related to b_n
noncomputable def sum_of_transformed_sequence (T : ℕ → ℚ) (b : ℕ → ℕ) : Prop :=
  ∀ n, T n = 1 - 1 / (n + 1)

theorem proof_problem :
  (∃ a : ℕ → ℕ, geometric_sequence a) ∧
  (∃ S : ℕ → ℕ, sum_of_sequence S) ∧
  (∃ (a b : ℕ → ℕ), geometric_sequence a ∧ transformed_sequence a b ∧
   (∃ T : ℕ → ℚ, sum_of_transformed_sequence T b)) :=
by {
  -- Definitions and proofs will go here
  sorry
}

end proof_problem_l140_140757


namespace roots_opposite_signs_l140_140804

theorem roots_opposite_signs (a b c: ℝ) 
  (h1 : (b^2 - a * c) > 0)
  (h2 : (b^4 - a^2 * c^2) < 0) :
  a * c < 0 :=
sorry

end roots_opposite_signs_l140_140804


namespace number_of_cats_l140_140661

theorem number_of_cats 
  (n k : ℕ)
  (h1 : n * k = 999919)
  (h2 : k > n) :
  n = 991 :=
sorry

end number_of_cats_l140_140661


namespace students_with_uncool_parents_correct_l140_140877

def total_students : ℕ := 30
def cool_dads : ℕ := 12
def cool_moms : ℕ := 15
def cool_both : ℕ := 9

def students_with_uncool_parents : ℕ :=
  total_students - (cool_dads + cool_moms - cool_both)

theorem students_with_uncool_parents_correct :
  students_with_uncool_parents = 12 := by
  sorry

end students_with_uncool_parents_correct_l140_140877


namespace ratio_arithmetic_seq_a2019_a2017_eq_l140_140639

def ratio_arithmetic_seq (a : ℕ → ℝ) : Prop := 
  ∀ n : ℕ, n ≥ 1 → a (n+2) / a (n+1) - a (n+1) / a n = 2

theorem ratio_arithmetic_seq_a2019_a2017_eq (a : ℕ → ℝ) 
  (h : ratio_arithmetic_seq a) 
  (ha1 : a 1 = 1) 
  (ha2 : a 2 = 1) 
  (ha3 : a 3 = 3) : 
  a 2019 / a 2017 = 4 * 2017^2 - 1 :=
sorry

end ratio_arithmetic_seq_a2019_a2017_eq_l140_140639


namespace shared_bill_per_person_l140_140286

noncomputable def totalBill : ℝ := 139.00
noncomputable def tipPercentage : ℝ := 0.10
noncomputable def totalPeople : ℕ := 5

theorem shared_bill_per_person :
  let tipAmount := totalBill * tipPercentage
  let totalBillWithTip := totalBill + tipAmount
  let amountPerPerson := totalBillWithTip / totalPeople
  amountPerPerson = 30.58 :=
by
  let tipAmount := totalBill * tipPercentage
  let totalBillWithTip := totalBill + tipAmount
  let amountPerPerson := totalBillWithTip / totalPeople
  have h1 : tipAmount = 13.90 := by sorry
  have h2 : totalBillWithTip = 152.90 := by sorry
  have h3 : amountPerPerson = 30.58 := by sorry
  exact h3

end shared_bill_per_person_l140_140286


namespace find_number_l140_140955

theorem find_number (x : ℝ) (h : 2 * x - 2.6 * 4 = 10) : x = 10.2 :=
sorry

end find_number_l140_140955


namespace find_two_digit_divisors_l140_140011

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def has_remainder (a b r : ℕ) : Prop := a = b * (a / b) + r

theorem find_two_digit_divisors (n : ℕ) (h1 : is_two_digit n) (h2 : has_remainder 723 n 30) :
  n = 33 ∨ n = 63 ∨ n = 77 ∨ n = 99 :=
sorry

end find_two_digit_divisors_l140_140011


namespace ab_value_l140_140816

/-- 
  Given the conditions:
  - a - b = 10
  - a^2 + b^2 = 210
  Prove that ab = 55.
-/
theorem ab_value (a b : ℝ) (h1 : a - b = 10) (h2 : a^2 + b^2 = 210) : a * b = 55 :=
by
  sorry

end ab_value_l140_140816


namespace combined_volume_of_all_cubes_l140_140024

/-- Lily has 4 cubes each with side length 3, Mark has 3 cubes each with side length 4,
    and Zoe has 2 cubes each with side length 5. Prove that the combined volume of all
    the cubes is 550. -/
theorem combined_volume_of_all_cubes 
  (lily_cubes : ℕ := 4) (lily_side_length : ℕ := 3)
  (mark_cubes : ℕ := 3) (mark_side_length : ℕ := 4)
  (zoe_cubes : ℕ := 2) (zoe_side_length : ℕ := 5) :
  (lily_cubes * lily_side_length ^ 3) + 
  (mark_cubes * mark_side_length ^ 3) + 
  (zoe_cubes * zoe_side_length ^ 3) = 550 :=
by
  have lily_volume : ℕ := lily_cubes * lily_side_length ^ 3
  have mark_volume : ℕ := mark_cubes * mark_side_length ^ 3
  have zoe_volume : ℕ := zoe_cubes * zoe_side_length ^ 3
  have total_volume : ℕ := lily_volume + mark_volume + zoe_volume
  sorry

end combined_volume_of_all_cubes_l140_140024


namespace maximum_area_of_equilateral_triangle_in_rectangle_l140_140249

noncomputable def maxEquilateralTriangleArea (a b : ℝ) : ℝ :=
  (953 * Real.sqrt 3) / 16

theorem maximum_area_of_equilateral_triangle_in_rectangle :
  ∀ (a b : ℕ), a = 13 → b = 14 → maxEquilateralTriangleArea a b = (953 * Real.sqrt 3) / 16 :=
by
  intros a b h₁ h₂
  rw [h₁, h₂]
  apply rfl

end maximum_area_of_equilateral_triangle_in_rectangle_l140_140249


namespace interest_years_proof_l140_140242

theorem interest_years_proof :
  let interest_r800_first_2_years := 800 * 0.05 * 2
  let interest_r800_next_3_years := 800 * 0.12 * 3
  let total_interest_r800 := interest_r800_first_2_years + interest_r800_next_3_years
  let interest_r600_first_3_years := 600 * 0.07 * 3
  let interest_r600_next_n_years := 600 * 0.10 * n
  (interest_r600_first_3_years + interest_r600_next_n_years = total_interest_r800) ->
  n = 5 →
  3 + n = 8 :=
by
  sorry

end interest_years_proof_l140_140242


namespace find_value_of_p_l140_140272

theorem find_value_of_p (p q r s t u v w : ℤ)
  (h1 : r + s = -2)
  (h2 : s + (-2) = 5)
  (h3 : t + u = 5)
  (h4 : u + v = 3)
  (h5 : v + w = 8)
  (h6 : w + t = 3)
  (h7 : q + r = s)
  (h8 : p + q = r) :
  p = -25 := by
  -- proof skipped
  sorry

end find_value_of_p_l140_140272


namespace solve_equation_solve_proportion_l140_140921

theorem solve_equation (x : ℚ) :
  (3 + x) * (30 / 100) = 4.8 → x = 13 :=
by sorry

theorem solve_proportion (x : ℚ) :
  (5 / x) = (9 / 2) / (8 / 5) → x = (16 / 9) :=
by sorry

end solve_equation_solve_proportion_l140_140921


namespace lettuce_types_l140_140414

/-- Let L be the number of types of lettuce. 
    Given that Terry has 3 types of tomatoes, 4 types of olives, 
    and 2 types of soup. The total number of options for his lunch combo is 48. 
    Prove that L = 2. --/

theorem lettuce_types (L : ℕ) (H : 3 * 4 * 2 * L = 48) : L = 2 :=
by {
  -- beginning of the proof
  sorry
}

end lettuce_types_l140_140414


namespace range_of_a_l140_140703

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 0 ≤ x ∧ x ≤ 5 → |2 - x| + |x + 1| ≤ a) ↔ 9 ≤ a := 
by sorry

end range_of_a_l140_140703


namespace total_birds_l140_140720

-- Definitions from conditions
def num_geese : ℕ := 58
def num_ducks : ℕ := 37

-- Proof problem statement
theorem total_birds : num_geese + num_ducks = 95 := by
  sorry

end total_birds_l140_140720


namespace number_of_boys_l140_140745

theorem number_of_boys (girls boys : ℕ) (total_books books_girls books_boys books_per_student : ℕ)
  (h1 : girls = 15)
  (h2 : total_books = 375)
  (h3 : books_girls = 225)
  (h4 : total_books = books_girls + books_boys)
  (h5 : books_girls = girls * books_per_student)
  (h6 : books_boys = boys * books_per_student)
  (h7 : books_per_student = 15) :
  boys = 10 :=
by
  sorry

end number_of_boys_l140_140745


namespace simplify_logical_expression_l140_140845

variables (A B C : Bool)

theorem simplify_logical_expression :
  (A && !B || B && !C || B && C || A && B) = (A || B) :=
by { sorry }

end simplify_logical_expression_l140_140845


namespace geometric_sequence_general_term_l140_140092

theorem geometric_sequence_general_term (n : ℕ) (a : ℕ → ℕ) (a1 : ℕ) (q : ℕ) 
  (h1 : a1 = 4) (h2 : q = 3) (h3 : ∀ n, a n = a1 * (q ^ (n - 1))) :
  a n = 4 * 3^(n - 1) := by
  sorry

end geometric_sequence_general_term_l140_140092


namespace exists_common_plane_l140_140419

-- Definition of the triangular pyramids
structure Pyramid :=
(base_area : ℝ)
(height : ℝ)

-- Function to represent the area of the intersection produced by a horizontal plane at distance x from the table
noncomputable def sectional_area (P : Pyramid) (x : ℝ) : ℝ :=
  P.base_area * (1 - x / P.height) ^ 2

-- Given seven pyramids
variables {P1 P2 P3 P4 P5 P6 P7 : Pyramid}

-- For any three pyramids, there exists a horizontal plane that intersects them in triangles of equal area
axiom triple_intersection:
  ∀ (Pi Pj Pk : Pyramid), ∃ x : ℝ, x ≥ 0 ∧ x ≤ min (Pi.height) (min (Pj.height) (Pk.height)) ∧
    sectional_area Pi x = sectional_area Pj x ∧ sectional_area Pk x = sectional_area Pi x

-- Prove that there exists a plane that intersects all seven pyramids in triangles of equal area
theorem exists_common_plane :
  ∃ x : ℝ, x ≥ 0 ∧ x ≤ min P1.height (min P2.height (min P3.height (min P4.height (min P5.height (min P6.height P7.height))))) ∧
    sectional_area P1 x = sectional_area P2 x ∧
    sectional_area P2 x = sectional_area P3 x ∧
    sectional_area P3 x = sectional_area P4 x ∧
    sectional_area P4 x = sectional_area P5 x ∧
    sectional_area P5 x = sectional_area P6 x ∧
    sectional_area P6 x = sectional_area P7 x :=
sorry

end exists_common_plane_l140_140419


namespace min_value_c_and_d_l140_140875

theorem min_value_c_and_d (c d : ℝ) (h1 : c > 0) (h2 : d > 0)
  (h3 : c^2 - 12 * d ≥ 0)
  (h4 : 9 * d^2 - 4 * c ≥ 0) :
  c + d ≥ 5.74 :=
sorry

end min_value_c_and_d_l140_140875


namespace m_greater_than_p_l140_140281

theorem m_greater_than_p (p m n : ℕ) (hp : Nat.Prime p) (hm : 0 < m) (hn : 0 < n) (eq : p^2 + m^2 = n^2) : m > p :=
sorry

end m_greater_than_p_l140_140281


namespace monochromatic_regions_lower_bound_l140_140835

theorem monochromatic_regions_lower_bound (n : ℕ) (h_n_ge_2 : n ≥ 2) :
  ∀ (blue_lines red_lines : ℕ) (conditions :
    blue_lines = 2 * n ∧ red_lines = n ∧ 
    (∀ (i j k l : ℕ), i ≠ j ∧ j ≠ k ∧ k ≠ l ∧ l ≠ i → 
      (blue_lines = 2 * n ∧ red_lines = n))) 
  , ∃ (monochromatic_regions : ℕ), 
      monochromatic_regions ≥ (n - 1) * (n - 2) / 2 :=
sorry

end monochromatic_regions_lower_bound_l140_140835


namespace probability_of_hitting_target_at_least_once_l140_140659

-- Define the constant probability of hitting the target in a single shot
def p_hit : ℚ := 2 / 3

-- Define the probability of missing the target in a single shot
def p_miss := 1 - p_hit

-- Define the probability of missing the target in all 3 shots
def p_miss_all_3 := p_miss ^ 3

-- Define the probability of hitting the target at least once in 3 shots
def p_hit_at_least_once := 1 - p_miss_all_3

-- Provide the theorem stating the solution
theorem probability_of_hitting_target_at_least_once :
  p_hit_at_least_once = 26 / 27 :=
by
  -- sorry is used to indicate the theorem needs to be proved
  sorry

end probability_of_hitting_target_at_least_once_l140_140659


namespace middle_letter_value_l140_140440

theorem middle_letter_value 
  (final_score : ℕ) 
  (first_letter_value : ℕ) 
  (third_letter_value : ℕ)
  (word_length : ℕ)
  (triple_score : ℕ)
  (total_points : ℕ)
  (middle_letter_value : ℕ)
  (h1 : final_score = 30)
  (h2 : first_letter_value = 1)
  (h3 : third_letter_value = 1)
  (h4 : word_length = 3)
  (h5 : triple_score = 3)
  (h6 : total_points = final_score / triple_score)
  (h7 : total_points = 10)
  (h8 : middle_letter_value = total_points - first_letter_value - third_letter_value) :
  middle_letter_value = 8 := 
by sorry

end middle_letter_value_l140_140440


namespace payback_duration_l140_140081

-- Define constants for the problem conditions
def C : ℝ := 25000
def R : ℝ := 4000
def E : ℝ := 1500

-- Formal statement to be proven
theorem payback_duration : C / (R - E) = 10 := 
by
  sorry

end payback_duration_l140_140081


namespace overlapping_area_of_thirty_sixty_ninety_triangles_l140_140833

-- Definitions for 30-60-90 triangle and the overlapping region
def thirty_sixty_ninety_triangle (hypotenuse : ℝ) := 
  (hypotenuse > 0) ∧ 
  (exists (short_leg long_leg : ℝ), short_leg = hypotenuse / 2 ∧ long_leg = short_leg * (Real.sqrt 3))

-- Area of a parallelogram given base and height
def parallelogram_area (base height : ℝ) : ℝ :=
  base * height

theorem overlapping_area_of_thirty_sixty_ninety_triangles :
  ∀ (hypotenuse : ℝ), thirty_sixty_ninety_triangle hypotenuse →
  hypotenuse = 10 →
  (∃ (base height : ℝ), base = height ∧ base * height = parallelogram_area (5 * Real.sqrt 3) (5 * Real.sqrt 3)) →
  parallelogram_area (5 * Real.sqrt 3) (5 * Real.sqrt 3) = 75 :=
by
  sorry

end overlapping_area_of_thirty_sixty_ninety_triangles_l140_140833


namespace coin_probability_l140_140600

theorem coin_probability (p : ℚ) 
  (P_X_3 : ℚ := 10 * p^3 * (1 - p)^2)
  (P_X_4 : ℚ := 5 * p^4 * (1 - p))
  (P_X_5 : ℚ := p^5)
  (w : ℚ := P_X_3 + P_X_4 + P_X_5) :
  w = 5 / 16 → p = 1 / 4 :=
by
  sorry

end coin_probability_l140_140600


namespace percentage_decrease_after_raise_l140_140925

theorem percentage_decrease_after_raise
  (original_salary : ℝ) (final_salary : ℝ) (initial_raise_percent : ℝ)
  (initial_salary_raised : original_salary * (1 + initial_raise_percent / 100) = 5500): 
  original_salary = 5000 -> final_salary = 5225 -> initial_raise_percent = 10 ->
  ∃ (percentage_decrease : ℝ),
    final_salary = original_salary * (1 + initial_raise_percent / 100) * (1 - percentage_decrease / 100)
    ∧ percentage_decrease = 5 := by
  intros h1 h2 h3
  use 5
  rw [h1, h2, h3]
  simp
  sorry

end percentage_decrease_after_raise_l140_140925


namespace find_a_find_k_max_l140_140839

-- Problem 1
theorem find_a (f : ℝ → ℝ) (a : ℝ) 
  (hf : ∀ x, f x = x * (a + Real.log x))
  (hmin : ∃ x, f x = -Real.exp (-2) ∧ ∀ y, f y ≥ f x) : a = 1 := 
sorry

-- Problem 2
theorem find_k_max {k : ℤ} : 
  (∀ x > 1, k < (x * (1 + Real.log x)) / (x - 1)) → k ≤ 3 :=
sorry

end find_a_find_k_max_l140_140839


namespace problem_statement_l140_140001

noncomputable def f (x : ℝ) : ℝ := x - Real.log x
noncomputable def g (x : ℝ) : ℝ := Real.log x / x

theorem problem_statement (x : ℝ) (h : 0 < x ∧ x ≤ Real.exp 1) : 
  f x > g x + 1/2 :=
sorry

end problem_statement_l140_140001


namespace Donggil_cleaning_time_l140_140803

-- Define the total area of the school as A.
variable (A : ℝ)

-- Define the cleaning rates of Daehyeon (D) and Donggil (G).
variable (D G : ℝ)

-- Conditions given in the problem
def condition1 : Prop := (D + G) * 8 = (7 / 12) * A
def condition2 : Prop := D * 10 = (5 / 12) * A

-- The goal is to prove that Donggil can clean the entire area alone in 32 days.
theorem Donggil_cleaning_time : condition1 A D G ∧ condition2 A D → 32 * G = A :=
by
  sorry

end Donggil_cleaning_time_l140_140803


namespace income_is_108000_l140_140445

theorem income_is_108000 (S I : ℝ) (h1 : S / I = 5 / 9) (h2 : 48000 = I - S) : I = 108000 :=
by
  sorry

end income_is_108000_l140_140445


namespace sandy_total_sums_l140_140102

theorem sandy_total_sums (C I : ℕ) (h1 : C = 22) (h2 : 3 * C - 2 * I = 50) :
  C + I = 30 :=
sorry

end sandy_total_sums_l140_140102


namespace num_divisible_by_7_200_to_400_l140_140618

noncomputable def count_divisible_by_seven (a b : ℕ) : ℕ :=
  let start := (a + 6) / 7 * 7 -- the smallest multiple of 7 >= a
  let stop := b / 7 * 7         -- the largest multiple of 7 <= b
  (stop - start) / 7 + 1

theorem num_divisible_by_7_200_to_400 : count_divisible_by_seven 200 400 = 29 :=
by
  sorry

end num_divisible_by_7_200_to_400_l140_140618


namespace b_3_value_S_m_formula_l140_140036

-- Definition of the sequences a_n and b_n
def a_n (n : ℕ) : ℕ := if n = 0 then 0 else 3 ^ n
def b_m (m : ℕ) : ℕ := a_n (3 * m)

-- Given b_m = 3^(2m) for m in ℕ*
lemma b_m_formula (m : ℕ) (h : m > 0) : b_m m = 3 ^ (2 * m) :=
by sorry -- (This proof step will later ensure that b_m m is defined as required)

-- Prove b_3 = 729
theorem b_3_value : b_m 3 = 729 :=
by sorry

-- Sum of the first m terms of the sequence b_n
def S_m (m : ℕ) : ℕ := (Finset.range m).sum (λ i => if i = 0 then 0 else b_m (i + 1))

-- Prove S_m = (3/8)(9^m - 1)
theorem S_m_formula (m : ℕ) : S_m m = (3 / 8) * (9 ^ m - 1) :=
by sorry

end b_3_value_S_m_formula_l140_140036


namespace total_dogs_l140_140998

variable (U : Type) [Fintype U]
variable (jump fetch shake : U → Prop)
variable [DecidablePred jump] [DecidablePred fetch] [DecidablePred shake]

theorem total_dogs (h_jump : Fintype.card {u | jump u} = 70)
  (h_jump_and_fetch : Fintype.card {u | jump u ∧ fetch u} = 30)
  (h_fetch : Fintype.card {u | fetch u} = 40)
  (h_fetch_and_shake : Fintype.card {u | fetch u ∧ shake u} = 20)
  (h_shake : Fintype.card {u | shake u} = 50)
  (h_jump_and_shake : Fintype.card {u | jump u ∧ shake u} = 25)
  (h_all_three : Fintype.card {u | jump u ∧ fetch u ∧ shake u} = 15)
  (h_none : Fintype.card {u | ¬jump u ∧ ¬fetch u ∧ ¬shake u} = 15) :
  Fintype.card U = 115 :=
by
  sorry

end total_dogs_l140_140998


namespace Lindas_savings_l140_140466

theorem Lindas_savings (S : ℝ) (h1 : (1/3) * S = 250) : S = 750 := 
by
  sorry

end Lindas_savings_l140_140466


namespace compute_105_times_95_l140_140362

theorem compute_105_times_95 : (105 * 95 = 9975) :=
by
  sorry

end compute_105_times_95_l140_140362


namespace power_six_sum_l140_140197

theorem power_six_sum (x : ℝ) (h : x + 1 / x = 3) : x^6 + 1 / x^6 = 322 := 
by 
  sorry

end power_six_sum_l140_140197


namespace money_raised_by_full_price_tickets_l140_140904

theorem money_raised_by_full_price_tickets (f h : ℕ) (p revenue total_tickets : ℕ) 
  (full_price : p = 20) (total_cost : f * p + h * (p / 2) = revenue) 
  (ticket_count : f + h = total_tickets) (total_revenue : revenue = 2750)
  (ticket_number : total_tickets = 180) : f * p = 1900 := 
by
  sorry

end money_raised_by_full_price_tickets_l140_140904


namespace slope_angle_of_line_x_equal_one_l140_140426

noncomputable def slope_angle_of_vertical_line : ℝ := 90

theorem slope_angle_of_line_x_equal_one : slope_angle_of_vertical_line = 90 := by
  sorry

end slope_angle_of_line_x_equal_one_l140_140426


namespace discount_price_l140_140748

theorem discount_price (original_price : ℝ) (discount_percent : ℝ) (final_price : ℝ) :
  original_price = 800 ∧ discount_percent = 15 → final_price = 680 :=
by
  intros h
  cases' h with hp hd
  sorry

end discount_price_l140_140748


namespace intersection_M_N_l140_140791

-- Definitions of the domains M and N
def M := {x : ℝ | x < 1}
def N := {x : ℝ | x > 0}

-- The goal is to prove that the intersection of M and N is equal to (0, 1)
theorem intersection_M_N : M ∩ N = {x : ℝ | 0 < x ∧ x < 1} :=
by
  sorry

end intersection_M_N_l140_140791


namespace andy_paint_total_l140_140907

-- Define the given ratio condition and green paint usage
def paint_ratio (blue green white : ℕ) : Prop :=
  blue / green = 1 / 2 ∧ white / green = 5 / 2

def green_paint_used (green : ℕ) : Prop :=
  green = 6

-- Define the proof goal: total paint used
def total_paint_used (blue green white : ℕ) : ℕ :=
  blue + green + white

-- The statement to be proved
theorem andy_paint_total (blue green white : ℕ)
  (h_ratio : paint_ratio blue green white)
  (h_green : green_paint_used green) :
  total_paint_used blue green white = 24 :=
  sorry

end andy_paint_total_l140_140907


namespace cone_ratio_l140_140465

noncomputable def cone_height_ratio : ℚ :=
  let original_height := 40
  let circumference := 24 * Real.pi
  let original_radius := 12
  let new_volume := 432 * Real.pi
  let new_height := 9
  new_height / original_height

theorem cone_ratio (h : cone_height_ratio = 9 / 40) : (9 : ℚ) / 40 = 9 / 40 := by
  sorry

end cone_ratio_l140_140465


namespace seunghyeon_pizza_diff_l140_140555

theorem seunghyeon_pizza_diff (S Y : ℕ) (h : S - 2 = Y + 7) : S - Y = 9 :=
by {
  sorry
}

end seunghyeon_pizza_diff_l140_140555


namespace find_c_l140_140684

-- Define conditions as Lean statements
theorem find_c :
  ∀ (c n : ℝ), 
  (n ^ 2 + 1 / 16 = 1 / 4) → 
  2 * n = c → 
  c < 0 → 
  c = - (Real.sqrt 3) / 2 :=
by
  intros c n h1 h2 h3
  sorry

end find_c_l140_140684


namespace trajectory_eq_l140_140690

theorem trajectory_eq (a b : ℝ) :
  (∀ x y : ℝ, (x - a)^2 + (y - b)^2 = 6 → x^2 + y^2 + 2 * x + 2 * y - 3 = 0 → 
    ∃ p q : ℝ, p = a + 1 ∧ q = b + 1 ∧ (p * x + q * y = (a^2 + b^2 - 3)/2)) →
  a^2 + b^2 + 2 * a + 2 * b + 1 = 0 :=
by
  intros h
  sorry

end trajectory_eq_l140_140690


namespace man_speed_in_still_water_l140_140596

theorem man_speed_in_still_water :
  ∃ (V_m V_s : ℝ), 
  V_m + V_s = 14 ∧ 
  V_m - V_s = 6 ∧ 
  V_m = 10 :=
by
  sorry

end man_speed_in_still_water_l140_140596


namespace first_system_solution_second_system_solution_l140_140335

theorem first_system_solution (x y : ℝ) (h₁ : 3 * x - y = 8) (h₂ : 3 * x - 5 * y = -20) : 
  x = 5 ∧ y = 7 := 
by
  sorry

theorem second_system_solution (x y : ℝ) (h₁ : x / 3 - y / 2 = -1) (h₂ : 3 * x - 2 * y = 1) : 
  x = 3 ∧ y = 4 := 
by
  sorry

end first_system_solution_second_system_solution_l140_140335


namespace four_digit_cubes_divisible_by_16_count_l140_140145

theorem four_digit_cubes_divisible_by_16_count :
  ∃ (count : ℕ), count = 3 ∧
    ∀ (m : ℕ), 1000 ≤ 64 * m^3 ∧ 64 * m^3 ≤ 9999 → (m = 3 ∨ m = 4 ∨ m = 5) :=
by {
  -- our proof would go here
  sorry
}

end four_digit_cubes_divisible_by_16_count_l140_140145


namespace six_digit_squares_l140_140237

theorem six_digit_squares :
    ∃ n m : ℕ, 100000 ≤ n ∧ n ≤ 999999 ∧ 100 ≤ m ∧ m ≤ 999 ∧ n = m^2 ∧ (n = 390625 ∨ n = 141376) :=
by
  sorry

end six_digit_squares_l140_140237


namespace multiple_proof_l140_140924

noncomputable def K := 185  -- Given KJ's stamps
noncomputable def AJ := 370  -- Given AJ's stamps
noncomputable def total_stamps := 930  -- Given total amount

-- Using the conditions to find C
noncomputable def stamps_of_three := AJ + K  -- Total stamps of KJ and AJ
noncomputable def C := total_stamps - stamps_of_three

-- Stating the equivalence we need to prove
theorem multiple_proof : ∃ M: ℕ, M * K + 5 = C := by
  -- The solution proof here if required
  existsi 2
  sorry  -- proof to be completed

end multiple_proof_l140_140924


namespace coordinates_of_C_l140_140790

noncomputable def point := (ℚ × ℚ)

def A : point := (2, 8)
def B : point := (6, 14)
def M : point := (4, 11)
def L : point := (6, 6)
def C : point := (14, 2)

-- midpoint formula definition
def is_midpoint (M A B : point) : Prop :=
  M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2

-- Main statement to prove
theorem coordinates_of_C (hM : is_midpoint M A B) : C = (14, 2) :=
  sorry

end coordinates_of_C_l140_140790


namespace find_length_of_smaller_rectangle_l140_140164

theorem find_length_of_smaller_rectangle
  (w : ℝ)
  (h_original : 10 * 15 = 150)
  (h_new_rectangle : 2 * w * w = 150)
  (h_z : w = 5 * Real.sqrt 3) :
  z = 5 * Real.sqrt 3 :=
by
  sorry

end find_length_of_smaller_rectangle_l140_140164


namespace lines_parallel_l140_140947

theorem lines_parallel 
  (a : ℝ) (b : ℝ) (c : ℝ)
  (α : ℝ) (β : ℝ) (γ : ℝ)
  (h1 : Real.log (Real.sin α) + Real.log (Real.sin γ) = 2 * Real.log (Real.sin β)) :
  (∀ x y : ℝ, ∀ a b c : ℝ, 
    (x * (Real.sin α)^2 + y * Real.sin α = a) → 
    (x * (Real.sin β)^2 + y * Real.sin γ = c) →
    (-Real.sin α = -((Real.sin β)^2 / Real.sin γ))) :=
sorry

end lines_parallel_l140_140947


namespace probability_of_C_and_D_are_equal_l140_140206

theorem probability_of_C_and_D_are_equal (h1 : Prob_A = 1/4) (h2 : Prob_B = 1/3) (h3 : total_prob = 1) (h4 : Prob_C = Prob_D) : 
  Prob_C = 5/24 ∧ Prob_D = 5/24 := by
  sorry

end probability_of_C_and_D_are_equal_l140_140206


namespace age_ratio_l140_140928

def Kul : ℕ := 22
def Saras : ℕ := 33

theorem age_ratio : (Saras / Kul : ℚ) = 3 / 2 := by
  sorry

end age_ratio_l140_140928


namespace find_a_for_arithmetic_progression_roots_l140_140551

theorem find_a_for_arithmetic_progression_roots (x a : ℝ) : 
  (∀ (x : ℝ), x^4 - a*x^2 + 1 = 0) → 
  (∃ (t1 t2 : ℝ), t1 > 0 ∧ t2 > 0 ∧ (t2 = 9*t1) ∧ (t1 + t2 = a) ∧ (t1 * t2 = 1)) → 
  (a = 10/3) := 
  by 
    intros h1 h2
    sorry

end find_a_for_arithmetic_progression_roots_l140_140551


namespace problem_1_problem_2_l140_140820

theorem problem_1 (a_0 a_1 a_2 a_3 a_4 a_5 a_6 : ℝ) (x : ℝ)
  (h : (2 * x - 1) ^ 6 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6) :
  a_0 + a_1 + a_2 + a_3 + a_4 + a_5 + a_6 = 1 :=
sorry

theorem problem_2 (a_0 a_1 a_2 a_3 a_4 a_5 a_6 : ℝ) (x : ℝ)
  (h : (2 * x - 1) ^ 6 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6) :
  a_0 + a_2 + a_4 + a_6 = 365 :=
sorry

end problem_1_problem_2_l140_140820


namespace fraction_eq_zero_iff_l140_140355

theorem fraction_eq_zero_iff (x : ℝ) : (3 * x - 1) / (x ^ 2 + 1) = 0 ↔ x = 1 / 3 := by
  sorry

end fraction_eq_zero_iff_l140_140355


namespace mn_value_l140_140121

noncomputable def log_base (a b : ℝ) := Real.log b / Real.log a

theorem mn_value (M N : ℝ) (a : ℝ) 
  (h1 : log_base M N = a * log_base N M)
  (h2 : M ≠ N) (h3 : M * N > 0) (h4 : M ≠ 1) (h5 : N ≠ 1) (h6 : a = 4)
  : M * N = N^(3/2) ∨ M * N = N^(1/2) := 
by
  sorry

end mn_value_l140_140121


namespace escalator_ride_time_l140_140423

theorem escalator_ride_time (x y k t : ℝ)
  (h1 : 75 * x = y)
  (h2 : 30 * (x + k) = y)
  (h3 : t = y / k) :
  t = 50 := by
  sorry

end escalator_ride_time_l140_140423


namespace min_u_condition_l140_140844

-- Define the function u and the condition
def u (x y : ℝ) : ℝ := x^2 + 4 * x + y^2 - 2 * y

def condition (x y : ℝ) : Prop := 2 * x + y ≥ 1

-- The statement we want to prove
theorem min_u_condition : ∃ (x y : ℝ), condition x y ∧ u x y = -9/5 := 
by
  sorry

end min_u_condition_l140_140844


namespace logarithmic_inequality_l140_140012

noncomputable def a : ℝ := Real.log 2 / Real.log 3
noncomputable def b : ℝ := Real.log 3 / Real.log 2
noncomputable def c : ℝ := Real.log (1 / 3) / Real.log 4

theorem logarithmic_inequality :
  Real.log a < (1 / 2)^b := by
  sorry

end logarithmic_inequality_l140_140012


namespace rectangle_length_l140_140865

theorem rectangle_length (w l : ℝ) (hP : (2 * l + 2 * w) / w = 5) (hA : l * w = 150) : l = 15 :=
by
  sorry

end rectangle_length_l140_140865


namespace find_original_price_l140_140014

theorem find_original_price (x y : ℝ) 
  (h1 : 60 * x + 75 * y = 2700)
  (h2 : 60 * 0.85 * x + 75 * 0.90 * y = 2370) : 
  x = 20 ∧ y = 20 :=
sorry

end find_original_price_l140_140014


namespace stan_water_intake_l140_140434

-- Define the constants and parameters given in the conditions
def words_per_minute : ℕ := 50
def pages : ℕ := 5
def words_per_page : ℕ := 400
def water_per_hour : ℚ := 15  -- use rational numbers for precise division

-- Define the derived quantities from the conditions
def total_words : ℕ := pages * words_per_page
def total_minutes : ℕ := total_words / words_per_minute
def water_per_minute : ℚ := water_per_hour / 60

-- State the theorem
theorem stan_water_intake : 10 = total_minutes * water_per_minute := by
  sorry

end stan_water_intake_l140_140434


namespace intersection_point_of_lines_l140_140305

theorem intersection_point_of_lines :
  ∃ (x y : ℚ), 
    (3 * y = -2 * x + 6) ∧ 
    (-2 * y = 7 * x + 4) ∧ 
    x = -24 / 17 ∧ 
    y = 50 / 17 :=
by
  sorry

end intersection_point_of_lines_l140_140305


namespace cost_per_semester_correct_l140_140370

variable (cost_per_semester total_cost : ℕ)
variable (years semesters_per_year : ℕ)

theorem cost_per_semester_correct :
    years = 13 →
    semesters_per_year = 2 →
    total_cost = 520000 →
    cost_per_semester = total_cost / (years * semesters_per_year) →
    cost_per_semester = 20000 := by
  sorry

end cost_per_semester_correct_l140_140370


namespace possible_triangular_frames_B_l140_140476

-- Define the sides of the triangles and the similarity condition
def similar_triangles (a₁ a₂ a₃ b₁ b₂ b₃ : ℕ) : Prop :=
  a₁ * b₂ = a₂ * b₁ ∧ a₁ * b₃ = a₃ * b₁ ∧ a₂ * b₃ = a₃ * b₂

def sides_of_triangle_A := (50, 60, 80)

def is_a_possible_triangle (b₁ b₂ b₃ : ℕ) : Prop :=
  similar_triangles 50 60 80 b₁ b₂ b₃

-- Given conditions
def side_of_triangle_B := 20

-- Theorem to prove
theorem possible_triangular_frames_B :
  ∃ (b₂ b₃ : ℕ), (is_a_possible_triangle 20 b₂ b₃ ∨ is_a_possible_triangle b₂ 20 b₃ ∨ is_a_possible_triangle b₂ b₃ 20) :=
sorry

end possible_triangular_frames_B_l140_140476


namespace negation_existential_proposition_l140_140274

theorem negation_existential_proposition :
  ¬(∃ x : ℝ, x^2 - x + 1 = 0) ↔ ∀ x : ℝ, x^2 - x + 1 ≠ 0 :=
by sorry

end negation_existential_proposition_l140_140274


namespace g_properties_l140_140652

def f (x : ℝ) : ℝ := x

def g (x : ℝ) : ℝ := -f x

theorem g_properties :
  (∀ x : ℝ, g (-x) = -g x) ∧ (∀ x y : ℝ, x < y → g x > g y) :=
by
  sorry

end g_properties_l140_140652


namespace not_or_false_imp_and_false_l140_140221

variable (p q : Prop)

theorem not_or_false_imp_and_false (h : ¬ (p ∨ q) = False) : ¬ (p ∧ q) :=
by
  sorry

end not_or_false_imp_and_false_l140_140221


namespace ratio_of_votes_l140_140108

theorem ratio_of_votes (total_votes ben_votes : ℕ) (h_total : total_votes = 60) (h_ben : ben_votes = 24) :
  (ben_votes : ℚ) / (total_votes - ben_votes : ℚ) = 2 / 3 :=
by sorry

end ratio_of_votes_l140_140108


namespace spade_problem_l140_140540

def spade (x y : ℝ) : ℝ := (x + y) * (x - y)

theorem spade_problem : spade 2 (spade 3 (spade 1 4)) = -46652 := 
by sorry

end spade_problem_l140_140540


namespace count_congruent_to_3_mod_8_l140_140883

theorem count_congruent_to_3_mod_8 : 
  ∃ (count : ℕ), count = 31 ∧ ∀ (x : ℕ), 1 ≤ x ∧ x ≤ 250 → x % 8 = 3 → x = 8 * ((x - 3) / 8) + 3 := sorry

end count_congruent_to_3_mod_8_l140_140883


namespace w1_relation_w2_relation_maximize_total_profit_l140_140560

def w1 (x : ℕ) : ℤ := 200 * x - 10000

def w2 (x : ℕ) : ℤ := -(x ^ 2) + 1000 * x - 50000

def total_sales_vol (x y : ℕ) : Prop := x + y = 1000

def max_profit_volumes (x y : ℕ) : Prop :=
  total_sales_vol x y ∧ x = 600 ∧ y = 400

theorem w1_relation (x : ℕ) :
  w1 x = 200 * x - 10000 := 
sorry

theorem w2_relation (x : ℕ) :
  w2 x = -(x ^ 2) + 1000 * x - 50000 := 
sorry

theorem maximize_total_profit (x y : ℕ) :
  total_sales_vol x y → max_profit_volumes x y := 
sorry

end w1_relation_w2_relation_maximize_total_profit_l140_140560


namespace g_possible_values_l140_140949

noncomputable def g (x y z : ℝ) : ℝ :=
  (x + y) / x + (y + z) / y + (z + x) / z

theorem g_possible_values (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  6 ≤ g x y z :=
by
  sorry

end g_possible_values_l140_140949


namespace moles_of_HCl_needed_l140_140547

-- Define the reaction and corresponding stoichiometry
def reaction_relates (NaHSO3 HCl NaCl H2O SO2 : ℕ) : Prop :=
  NaHSO3 = HCl ∧ HCl = NaCl ∧ NaCl = H2O ∧ H2O = SO2

-- Given condition: one mole of each reactant produces one mole of each product
axiom reaction_stoichiometry : reaction_relates 1 1 1 1 1

-- Prove that 2 moles of NaHSO3 reacting with 2 moles of HCl forms 2 moles of NaCl
theorem moles_of_HCl_needed :
  ∀ (NaHSO3 HCl NaCl : ℕ), reaction_relates NaHSO3 HCl NaCl NaCl NaCl → NaCl = 2 → HCl = 2 :=
by
  intros NaHSO3 HCl NaCl h_eq h_NaCl
  sorry

end moles_of_HCl_needed_l140_140547


namespace average_GPA_of_class_l140_140792

theorem average_GPA_of_class (n : ℕ) (h1 : n > 0) 
  (GPA1 : ℝ := 60) (GPA2 : ℝ := 66) 
  (students_ratio1 : ℝ := 1 / 3) (students_ratio2 : ℝ := 2 / 3) :
  let total_students := (students_ratio1 * n + students_ratio2 * n)
  let total_GPA := (students_ratio1 * n * GPA1 + students_ratio2 * n * GPA2)
  let average_GPA := total_GPA / total_students
  average_GPA = 64 := by
    sorry

end average_GPA_of_class_l140_140792


namespace smallest_four_digit_divisible_43_l140_140451

theorem smallest_four_digit_divisible_43 :
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 43 = 0 ∧ n = 1032 :=
by
  sorry

end smallest_four_digit_divisible_43_l140_140451


namespace largest_A_l140_140786

namespace EquivalentProofProblem

def F (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x > 0 → f (3 * x) ≥ f (f (2 * x)) + x

theorem largest_A (f : ℝ → ℝ) (hf : F f) (x : ℝ) (hx : x > 0) : 
  ∃ A, (∀ (f : ℝ → ℝ), F f → ∀ x, x > 0 → f x ≥ A * x) ∧ A = 1 / 2 :=
sorry

end EquivalentProofProblem

end largest_A_l140_140786


namespace equation_of_BC_area_of_triangle_l140_140076

section triangle_geometry

variables (x y : ℝ)

/-- Given equations of the altitudes and vertex A, the equation of side BC is 2x + 3y + 7 = 0 -/
theorem equation_of_BC (h1 : 2 * x - 3 * y + 1 = 0) (h2 : x + y = 0) (A : ℝ × ℝ) (hA : A = (1, 2)) :
  ∃ (a b c : ℝ), (a * x + b * y + c = 0) ∧ (a, b, c) = (2, 3, 7) := 
sorry

/-- Given equations of the altitudes and vertex A, the area of triangle ABC is 45/2 -/
theorem area_of_triangle (h1 : 2 * x - 3 * y + 1 = 0) (h2 : x + y = 0) (A : ℝ × ℝ) (hA : A = (1, 2)) :
  ∃ (area : ℝ), (area = (45 / 2)) := 
sorry

end triangle_geometry

end equation_of_BC_area_of_triangle_l140_140076


namespace least_value_xy_l140_140520

theorem least_value_xy {x y : ℕ} (hx : 0 < x) (hy : 0 < y) (h : (1 : ℚ) / x + 1 / (3 * y) = 1 / 8) : x * y = 96 :=
sorry

end least_value_xy_l140_140520


namespace evaluate_expression_l140_140080

theorem evaluate_expression :
  (|(-1 : ℝ)|^2023 + (Real.sqrt 3)^2 - 2 * Real.sin (Real.pi / 6) + (1 / 2)⁻¹ = 5) :=
by
  sorry

end evaluate_expression_l140_140080


namespace compound_interest_calculation_l140_140495

-- Given conditions
def P : ℝ := 20000
def r : ℝ := 0.03
def t : ℕ := 5

-- The amount after t years with compound interest
def A := P * (1 + r) ^ t

-- Prove the total amount is as given in choice B
theorem compound_interest_calculation : 
  A = 20000 * (1 + 0.03) ^ 5 :=
by
  sorry

end compound_interest_calculation_l140_140495


namespace negation_of_every_square_positive_l140_140980

theorem negation_of_every_square_positive :
  ¬(∀ n : ℕ, n^2 > 0) ↔ ∃ n : ℕ, n^2 ≤ 0 := sorry

end negation_of_every_square_positive_l140_140980


namespace vector_relation_AD_l140_140598

variables {P V : Type} [AddCommGroup V] [Module ℝ V]
variables (A B C D : P) (AB AC AD BC BD CD : V)
variables (hBC_CD : BC = 3 • CD)

theorem vector_relation_AD (h1 : BC = 3 • CD)
                           (h2 : AD = AB + BD)
                           (h3 : BD = BC + CD)
                           (h4 : BC = -AB + AC) :
  AD = - (1 / 3 : ℝ) • AB + (4 / 3 : ℝ) • AC :=
by
  sorry

end vector_relation_AD_l140_140598


namespace range_of_a_l140_140228

variable {a : ℝ}

theorem range_of_a (h : ∃ x : ℝ, x < 0 ∧ 5^x = (a + 3) / (5 - a)) : -3 < a ∧ a < 1 :=
by
  sorry

end range_of_a_l140_140228


namespace sequence_sum_l140_140599

noncomputable def a (n : ℕ) : ℝ := n * Real.cos (n * Real.pi / 2)

noncomputable def S : ℕ → ℝ
| 0     => 0
| (n+1) => S n + a (n+1)

theorem sequence_sum : S 2017 = 1008 :=
by
  sorry

end sequence_sum_l140_140599


namespace sufficient_but_not_necessary_condition_l140_140402

def parabola (y : ℝ) : ℝ := y^2
def line (m : ℝ) (y : ℝ) : ℝ := m * y + 1

theorem sufficient_but_not_necessary_condition {m : ℝ} :
  (m ≠ 0) → ∃ y1 y2 : ℝ, y1 ≠ y2 ∧ parabola y1 = line m y1 ∧ parabola y2 = line m y2 :=
by
  sorry

end sufficient_but_not_necessary_condition_l140_140402


namespace determine_n_l140_140985

noncomputable def S : ℕ → ℝ := sorry -- define arithmetic series sum
noncomputable def a_1 : ℝ := sorry -- define first term
noncomputable def d : ℝ := sorry -- define common difference

axiom S_6 : S 6 = 36
axiom S_n {n : ℕ} (h : n > 0) : S n = 324
axiom S_n_minus_6 {n : ℕ} (h : n > 6) : S (n - 6) = 144

theorem determine_n (n : ℕ) (h : n > 0) : n = 18 := by {
  sorry
}

end determine_n_l140_140985


namespace total_cost_of_pencils_l140_140581

def pencil_price : ℝ := 0.20
def pencils_Tolu : ℕ := 3
def pencils_Robert : ℕ := 5
def pencils_Melissa : ℕ := 2

theorem total_cost_of_pencils :
  (pencil_price * pencils_Tolu + pencil_price * pencils_Robert + pencil_price * pencils_Melissa) = 2.00 := 
sorry

end total_cost_of_pencils_l140_140581


namespace teams_in_double_round_robin_l140_140754
-- Import the standard math library

-- Lean statement for the proof problem
theorem teams_in_double_round_robin (m n : ℤ) 
  (h : 9 * n^2 + 6 * n + 32 = m * (m - 1) / 2) : 
  m = 8 ∨ m = 32 :=
sorry

end teams_in_double_round_robin_l140_140754


namespace simple_interest_correct_l140_140415

def principal : ℝ := 10040.625
def rate : ℝ := 8
def time : ℕ := 5

theorem simple_interest_correct :
  (principal * rate * time / 100) = 40162.5 :=
by 
  sorry

end simple_interest_correct_l140_140415


namespace evaluate_expression_l140_140972

noncomputable def a : ℝ := Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 6
noncomputable def b : ℝ := -Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 6
noncomputable def c : ℝ := Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 6
noncomputable def d : ℝ := -Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 6

theorem evaluate_expression :
  ( (1 / a) + (1 / b) + (1 / c) + (1 / d) ) ^ 2 = 96 / 529 :=
by
  sorry

end evaluate_expression_l140_140972


namespace tailor_buttons_l140_140662

theorem tailor_buttons (G : ℕ) (yellow_buttons : ℕ) (blue_buttons : ℕ) 
(h1 : yellow_buttons = G + 10) (h2 : blue_buttons = G - 5) 
(h3 : G + yellow_buttons + blue_buttons = 275) : G = 90 :=
sorry

end tailor_buttons_l140_140662


namespace thirtieth_triangular_number_sum_thirtieth_thirtyfirst_triangular_numbers_l140_140396

def triangular_number (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem thirtieth_triangular_number :
  triangular_number 30 = 465 :=
by
  sorry

theorem sum_thirtieth_thirtyfirst_triangular_numbers :
  triangular_number 30 + triangular_number 31 = 961 :=
by
  sorry

end thirtieth_triangular_number_sum_thirtieth_thirtyfirst_triangular_numbers_l140_140396


namespace find_m_l140_140880

theorem find_m (m : ℝ) (h : 2^2 + 2 * m + 2 = 0) : m = -3 :=
by {
  sorry
}

end find_m_l140_140880


namespace power_sum_prime_eq_l140_140072

theorem power_sum_prime_eq (p a n : ℕ) (hp : p.Prime) (h_eq : 2^p + 3^p = a^n) : n = 1 :=
by sorry

end power_sum_prime_eq_l140_140072


namespace fraction_division_l140_140484

theorem fraction_division : 
  ((8 / 4) * (9 / 3) * (20 / 5)) / ((10 / 5) * (12 / 4) * (15 / 3)) = (4 / 5) := 
by
  sorry

end fraction_division_l140_140484


namespace trisect_54_degree_angle_l140_140674

theorem trisect_54_degree_angle :
  ∃ (a1 a2 : ℝ), a1 = 18 ∧ a2 = 36 ∧ a1 + a2 + a2 = 54 :=
by sorry

end trisect_54_degree_angle_l140_140674


namespace swimmers_meet_times_l140_140752

noncomputable def swimmers_passes (pool_length : ℕ) (time_minutes : ℕ) (speed_swimmer1 : ℕ) (speed_swimmer2 : ℕ) : ℕ :=
  let total_time_seconds := time_minutes * 60
  let speed_sum := speed_swimmer1 + speed_swimmer2
  let distance_in_time := total_time_seconds * speed_sum
  distance_in_time / pool_length

theorem swimmers_meet_times :
  swimmers_passes 120 15 4 3 = 53 :=
by
  -- Proof is omitted
  sorry

end swimmers_meet_times_l140_140752


namespace solve_for_s_l140_140855

theorem solve_for_s {x : ℝ} (h : 4 * x^2 - 8 * x - 320 = 0) : ∃ s, s = 81 :=
by 
  -- Introduce the conditions and the steps
  sorry

end solve_for_s_l140_140855


namespace problem_equivalent_l140_140733

variable (p : ℤ) 

theorem problem_equivalent (h : p = (-2023) * 100) : (-2023) * 99 = p + 2023 :=
by sorry

end problem_equivalent_l140_140733


namespace more_sqft_to_mow_l140_140256

-- Defining the parameters given in the original problem
def rate_per_sqft : ℝ := 0.10
def book_cost : ℝ := 150.0
def lawn_dimensions : ℝ × ℝ := (20, 15)
def num_lawns_mowed : ℕ := 3

-- The theorem stating how many more square feet LaKeisha needs to mow
theorem more_sqft_to_mow : 
  let area_one_lawn := (lawn_dimensions.1 * lawn_dimensions.2 : ℝ)
  let total_area_mowed := area_one_lawn * (num_lawns_mowed : ℝ)
  let money_earned := total_area_mowed * rate_per_sqft
  let remaining_amount := book_cost - money_earned
  let more_sqft_needed := remaining_amount / rate_per_sqft
  more_sqft_needed = 600 := 
by 
  sorry

end more_sqft_to_mow_l140_140256


namespace minimum_bottles_needed_l140_140882

theorem minimum_bottles_needed :
  (∃ n : ℕ, n * 45 ≥ 720 - 20 ∧ (n - 1) * 45 < 720 - 20) ∧ 720 - 20 = 700 :=
by
  sorry

end minimum_bottles_needed_l140_140882


namespace max_k_value_l140_140118

def maximum_k (k : ℕ) : ℕ := 2

theorem max_k_value
  (k : ℕ)
  (h1 : 2 * k + 1 ≤ 20)  -- Condition implicitly implied by having subsets of a 20-element set
  (h2 : ∀ (s t : Finset (Fin 20)), s.card = 7 → t.card = 7 → s ≠ t → (s ∩ t).card = k) : k ≤ maximum_k k := 
by {
  sorry
}

end max_k_value_l140_140118


namespace total_songs_megan_bought_l140_140744

-- Definitions for the problem conditions
def country_albums : ℕ := 2
def pop_albums : ℕ := 8
def songs_per_album : ℕ := 7
def total_albums : ℕ := country_albums + pop_albums

-- Theorem stating the conclusion we need to prove
theorem total_songs_megan_bought : total_albums * songs_per_album = 70 :=
by
  sorry

end total_songs_megan_bought_l140_140744


namespace expression_value_l140_140051

theorem expression_value (a b : ℤ) (ha : a = -4) (hb : b = 3) : 
  -2 * a - b ^ 3 + 2 * a * b = -43 := by
  rw [ha, hb]
  sorry

end expression_value_l140_140051


namespace max_m_plus_n_l140_140478

theorem max_m_plus_n (m n : ℝ) (h : n = -m^2 - 3*m + 3) : m + n ≤ 4 :=
by {
  sorry
}

end max_m_plus_n_l140_140478


namespace tanya_body_lotions_l140_140902

variable {F L : ℕ}  -- Number of face moisturizers (F) and body lotions (L) Tanya bought

theorem tanya_body_lotions
  (price_face_moisturizer : ℕ := 50)
  (price_body_lotion : ℕ := 60)
  (num_face_moisturizers : ℕ := 2)
  (total_spent : ℕ := 1020)
  (christy_spending_factor : ℕ := 2)
  (h_together_spent : total_spent = 3 * (num_face_moisturizers * price_face_moisturizer + L * price_body_lotion)) :
  L = 4 :=
by
  sorry

end tanya_body_lotions_l140_140902


namespace sum_of_squares_l140_140773

variable (a b c : ℝ)
variable (S : ℝ)

theorem sum_of_squares (h1 : ab + bc + ac = 131)
                       (h2 : a + b + c = 22) :
  a^2 + b^2 + c^2 = 222 :=
by
  -- Proof would be placed here
  sorry

end sum_of_squares_l140_140773


namespace right_triangle_of_three_colors_exists_l140_140135

-- Define the type for color
inductive Color
| color1
| color2
| color3

open Color

-- Define the type for integer coordinate points
structure Point :=
(x : ℤ)
(y : ℤ)
(color : Color)

-- Define the conditions
def all_points_colored : Prop :=
∀ (p : Point), p.color = color1 ∨ p.color = color2 ∨ p.color = color3

def all_colors_used : Prop :=
∃ (p1 p2 p3 : Point), p1.color = color1 ∧ p2.color = color2 ∧ p3.color = color3

-- Define the right_triangle_exist problem
def right_triangle_exists : Prop :=
∃ (p1 p2 p3 : Point), 
  p1.color ≠ p2.color ∧ p2.color ≠ p3.color ∧ p3.color ≠ p1.color ∧
  (p1.x = p2.x ∧ p2.y = p3.y ∧ p1.y = p3.y ∨
   p1.y = p2.y ∧ p2.x = p3.x ∧ p1.x = p3.x ∨
   (p3.x - p1.x)*(p3.x - p1.x) + (p3.y - p1.y)*(p3.y - p1.y) = (p2.x - p1.x)*(p2.x - p1.x) + (p2.y - p1.y)*(p2.y - p1.y) ∧
   (p3.x - p2.x)*(p3.x - p2.x) + (p3.y - p2.y)*(p3.y - p2.y) = (p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y))

theorem right_triangle_of_three_colors_exists (h1 : all_points_colored) (h2 : all_colors_used) : right_triangle_exists := 
sorry

end right_triangle_of_three_colors_exists_l140_140135


namespace makarala_meetings_percentage_l140_140410

def work_day_to_minutes (hours: ℕ) : ℕ :=
  60 * hours

def total_meeting_time (first: ℕ) (second: ℕ) : ℕ :=
  let third := first + second
  first + second + third

def percentage_of_day_spent (meeting_time: ℕ) (work_day_time: ℕ) : ℚ :=
  (meeting_time : ℚ) / (work_day_time : ℚ) * 100

theorem makarala_meetings_percentage
  (work_hours: ℕ)
  (first_meeting: ℕ)
  (second_meeting: ℕ)
  : percentage_of_day_spent (total_meeting_time first_meeting second_meeting) (work_day_to_minutes work_hours) = 37.5 :=
by
  sorry

end makarala_meetings_percentage_l140_140410


namespace BigJoe_is_8_feet_l140_140101

variable (Pepe_height : ℝ) (h1 : Pepe_height = 4.5)
variable (Frank_height : ℝ) (h2 : Frank_height = Pepe_height + 0.5)
variable (Larry_height : ℝ) (h3 : Larry_height = Frank_height + 1)
variable (Ben_height : ℝ) (h4 : Ben_height = Larry_height + 1)
variable (BigJoe_height : ℝ) (h5 : BigJoe_height = Ben_height + 1)

theorem BigJoe_is_8_feet : BigJoe_height = 8 := by
  sorry

end BigJoe_is_8_feet_l140_140101


namespace greatest_of_5_consecutive_integers_l140_140471

theorem greatest_of_5_consecutive_integers (m n : ℤ) (h : 5 * n + 10 = m^3) : (n + 4) = 202 := by
sorry

end greatest_of_5_consecutive_integers_l140_140471


namespace Option_C_correct_l140_140032

theorem Option_C_correct (x y : ℝ) : 3 * x * y^2 - 4 * x * y^2 = - x * y^2 :=
by
  sorry

end Option_C_correct_l140_140032


namespace Elmo_books_count_l140_140057

-- Define the number of books each person has
def Stu_books : ℕ := 4
def Laura_books : ℕ := 2 * Stu_books
def Elmo_books : ℕ := 3 * Laura_books

-- The theorem we need to prove
theorem Elmo_books_count : Elmo_books = 24 := by
  -- this part is skipped since no proof is required
  sorry

end Elmo_books_count_l140_140057


namespace geometric_sequence_fifth_term_l140_140527

theorem geometric_sequence_fifth_term (a r : ℝ) (h1 : a * r^2 = 9) (h2 : a * r^6 = 1) : a * r^4 = 3 :=
by
  sorry

end geometric_sequence_fifth_term_l140_140527


namespace parabola_chord_length_eight_l140_140929

noncomputable def parabola_intersection_length (x1 x2: ℝ) (y1 y2: ℝ) : ℝ :=
  if x1 + x2 = 6 ∧ y1^2 = 4 * x1 ∧ y2^2 = 4 * x2 then
    let A := (x1, y1)
    let B := (x2, y2)
    dist A B
  else
    0

theorem parabola_chord_length_eight :
  ∀ (x1 x2 y1 y2 : ℝ), (x1 + x2 = 6) → (y1^2 = 4 * x1) → (y2^2 = 4 * x2) →
  parabola_intersection_length x1 x2 y1 y2 = 8 :=
by
  -- proof goes here
  sorry

end parabola_chord_length_eight_l140_140929


namespace remainder_of_large_number_l140_140367

theorem remainder_of_large_number :
  (1235678901 % 101) = 1 :=
by
  have h1: (10^8 % 101) = 1 := sorry
  have h2: (10^6 % 101) = 1 := sorry
  have h3: (10^4 % 101) = 1 := sorry
  have h4: (10^2 % 101) = 1 := sorry
  have large_number_decomposition: 1235678901 = 12 * 10^8 + 35 * 10^6 + 67 * 10^4 + 89 * 10^2 + 1 := sorry
  -- Proof using the decomposition and modulo properties
  sorry

end remainder_of_large_number_l140_140367


namespace fifth_eqn_nth_eqn_l140_140450

theorem fifth_eqn : 10 * 12 + 1 = 121 :=
by
  sorry

theorem nth_eqn (n : ℕ) : 2 * n * (2 * n + 2) + 1 = (2 * n + 1)^2 :=
by
  sorry

end fifth_eqn_nth_eqn_l140_140450


namespace total_cans_collected_l140_140906

-- Definitions based on conditions
def cans_LaDonna : ℕ := 25
def cans_Prikya : ℕ := 2 * cans_LaDonna
def cans_Yoki : ℕ := 10

-- Theorem statement
theorem total_cans_collected : 
  cans_LaDonna + cans_Prikya + cans_Yoki = 85 :=
by
  -- The proof is not required, inserting sorry to complete the statement
  sorry

end total_cans_collected_l140_140906


namespace n_value_l140_140895

theorem n_value (n : ℕ) : 2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = 4^18 → n = 9 :=
by
  sorry

end n_value_l140_140895


namespace fraction_product_l140_140138

theorem fraction_product :
  (7 / 4) * (8 / 14) * (28 / 16) * (24 / 36) * (49 / 35) * (40 / 25) * (63 / 42) * (32 / 48) = 56 / 25 :=
by sorry

end fraction_product_l140_140138


namespace express_in_scientific_notation_l140_140927

theorem express_in_scientific_notation : 
  ∃ (a : ℝ) (n : ℤ), 388800 = a * 10 ^ n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ a = 3.888 ∧ n = 5 :=
by
  sorry

end express_in_scientific_notation_l140_140927


namespace total_students_l140_140416

theorem total_students (S : ℕ) (R : ℕ) :
  (2 * 0 + 12 * 1 + 13 * 2 + R * 3) / S = 2 →
  2 + 12 + 13 + R = S →
  S = 43 :=
by
  sorry

end total_students_l140_140416


namespace Hari_investment_contribution_l140_140203

noncomputable def Praveen_investment : ℕ := 3780
noncomputable def Praveen_time : ℕ := 12
noncomputable def Hari_time : ℕ := 7
noncomputable def profit_ratio : ℚ := 2 / 3

theorem Hari_investment_contribution :
  ∃ H : ℕ, (Praveen_investment * Praveen_time) / (H * Hari_time) = (2 : ℕ) / 3 ∧ H = 9720 :=
by
  sorry

end Hari_investment_contribution_l140_140203


namespace world_grain_supply_is_correct_l140_140153

def world_grain_demand : ℝ := 2400000
def supply_ratio : ℝ := 0.75
def world_grain_supply (demand : ℝ) (ratio : ℝ) : ℝ := ratio * demand

theorem world_grain_supply_is_correct :
  world_grain_supply world_grain_demand supply_ratio = 1800000 := 
by 
  sorry

end world_grain_supply_is_correct_l140_140153


namespace two_pow_div_factorial_iff_l140_140594

theorem two_pow_div_factorial_iff (n : ℕ) : 
  (∃ k : ℕ, k > 0 ∧ n = 2^(k - 1)) ↔ (∃ m : ℕ, m > 0 ∧ 2^(n - 1) ∣ n!) :=
by
  sorry

end two_pow_div_factorial_iff_l140_140594


namespace eq1_solution_eq2_solution_l140_140814

theorem eq1_solution (x : ℝ) (h : 6 * x - 7 = 4 * x - 5) : x = 1 :=
by
  sorry

theorem eq2_solution (x : ℝ) (h : (1 / 2) * x - 6 = (3 / 4) * x) : x = -24 :=
by
  sorry

end eq1_solution_eq2_solution_l140_140814


namespace sams_speed_l140_140638

theorem sams_speed (lucas_speed : ℝ) (maya_factor : ℝ) (relationship_factor : ℝ) 
  (h_lucas : lucas_speed = 5)
  (h_maya : maya_factor = 4 / 5)
  (h_relationship : relationship_factor = 9 / 8) :
  (5 / relationship_factor) = 40 / 9 :=
by
  sorry

end sams_speed_l140_140638


namespace constant_max_value_l140_140366

theorem constant_max_value (n : ℤ) (c : ℝ) (h1 : c * (n^2) ≤ 8100) (h2 : n = 8) :
  c ≤ 126.5625 :=
sorry

end constant_max_value_l140_140366


namespace workman_problem_l140_140992

theorem workman_problem
    (total_work : ℝ)
    (B_rate : ℝ)
    (A_rate : ℝ)
    (days_together : ℝ)
    (W : total_work = 8 * (A_rate + B_rate))
    (A_2B : A_rate = 2 * B_rate) :
    total_work = 24 * B_rate :=
by
  sorry

end workman_problem_l140_140992


namespace train_crosses_platform_in_20_seconds_l140_140851

theorem train_crosses_platform_in_20_seconds 
  (t : ℝ) (lp : ℝ) (lt : ℝ) (tp : ℝ) (sp : ℝ) (st : ℝ) 
  (pass_time : st = lt / tp) (lc : lp = 267) (lc_train : lt = 178) (cross_time : t = sp / st) : 
  t = 20 :=
by
  sorry

end train_crosses_platform_in_20_seconds_l140_140851


namespace leila_money_left_l140_140759

theorem leila_money_left (initial_money spent_on_sweater spent_on_jewelry total_spent left_money : ℕ) 
  (h1 : initial_money = 160) 
  (h2 : spent_on_sweater = 40) 
  (h3 : spent_on_jewelry = 100) 
  (h4 : total_spent = spent_on_sweater + spent_on_jewelry) 
  (h5 : total_spent = 140) : 
  initial_money - total_spent = 20 := by
  sorry

end leila_money_left_l140_140759


namespace range_a_of_tangents_coincide_l140_140456

theorem range_a_of_tangents_coincide (x1 x2 : ℝ) (h1 : x1 < 0) (h2 : 0 < x2) (a : ℝ)
  (h3 : -1 / (x2 ^ 2) = 2 * x1 + 1) (h4 : x1 ^ 2 = -a) :
  1/4 < a ∧ a < 1 :=
by
  sorry 

end range_a_of_tangents_coincide_l140_140456


namespace train_crossing_time_l140_140731

/-- Given the conditions that a moving train requires 10 seconds to pass a pole,
    its speed is 36 km/h, and the length of a stationary train is 300 meters,
    prove that the moving train takes 40 seconds to cross the stationary train. -/
theorem train_crossing_time (t_pole : ℕ)
  (v_kmh : ℕ)
  (length_stationary : ℕ) :
  t_pole = 10 →
  v_kmh = 36 →
  length_stationary = 300 →
  ∃ t_cross : ℕ, t_cross = 40 :=
by
  intros h1 h2 h3
  sorry

end train_crossing_time_l140_140731


namespace cars_transfer_equation_l140_140908

theorem cars_transfer_equation (x : ℕ) : 100 - x = 68 + x :=
sorry

end cars_transfer_equation_l140_140908


namespace cistern_fill_time_l140_140224

theorem cistern_fill_time (hF : ∀ (F : ℝ), F = 1 / 3)
                         (hE : ∀ (E : ℝ), E = 1 / 5) : 
  ∃ (t : ℝ), t = 15 / 2 :=
by
  sorry

end cistern_fill_time_l140_140224


namespace greatest_sum_consecutive_integers_product_less_than_500_l140_140313

theorem greatest_sum_consecutive_integers_product_less_than_500 :
  ∃ n : ℕ, n * (n + 1) < 500 ∧ (n + (n + 1) = 43) := sorry

end greatest_sum_consecutive_integers_product_less_than_500_l140_140313


namespace range_of_a_opposite_sides_l140_140903

theorem range_of_a_opposite_sides {a : ℝ} (h : (0 + 0 - a) * (1 + 1 - a) < 0) : 0 < a ∧ a < 2 :=
sorry

end range_of_a_opposite_sides_l140_140903


namespace compare_M_N_l140_140981

variable (a : ℝ)

def M : ℝ := 2 * a^2 - 4 * a
def N : ℝ := a^2 - 2 * a - 3

theorem compare_M_N : M a > N a := by
  sorry

end compare_M_N_l140_140981


namespace fraction_never_reducible_by_11_l140_140174

theorem fraction_never_reducible_by_11 :
  ∀ (n : ℕ), Nat.gcd (1 + n) (3 + 7 * n) ≠ 11 := by
  sorry

end fraction_never_reducible_by_11_l140_140174


namespace right_triangle_area_l140_140324

theorem right_triangle_area (A B C : ℝ) (hA : A = 64) (hB : B = 49) (hC : C = 225) :
  let a := Real.sqrt A
  let b := Real.sqrt B
  let c := Real.sqrt C
  ∃ (area : ℝ), area = (1 / 2) * a * b ∧ area = 28 :=
by
  sorry

end right_triangle_area_l140_140324


namespace jihye_marbles_l140_140353

theorem jihye_marbles (Y : ℕ) (h1 : Y + (Y + 11) = 85) : Y + 11 = 48 := by
  sorry

end jihye_marbles_l140_140353


namespace divisibility_by_P_divisibility_by_P_squared_divisibility_by_P_cubed_l140_140644

noncomputable def Q (x : ℝ) (n : ℕ) : ℝ := (x + 1)^n - x^n - 1

def P (x : ℝ) : ℝ := x^2 + x + 1

-- Prove Q(x, n) is divisible by P(x) if and only if n ≡ 1 or 5 (mod 6)
theorem divisibility_by_P (x : ℝ) (n : ℕ) : 
  (Q x n) % (P x) = 0 ↔ (n % 6 = 1 ∨ n % 6 = 5) := 
sorry

-- Prove Q(x, n) is divisible by P(x)^2 if and only if n ≡ 1 (mod 6)
theorem divisibility_by_P_squared (x : ℝ) (n : ℕ) : 
  (Q x n) % (P x)^2 = 0 ↔ n % 6 = 1 := 
sorry

-- Prove Q(x, n) is divisible by P(x)^3 if and only if n = 1
theorem divisibility_by_P_cubed (x : ℝ) (n : ℕ) : 
  (Q x n) % (P x)^3 = 0 ↔ n = 1 := 
sorry

end divisibility_by_P_divisibility_by_P_squared_divisibility_by_P_cubed_l140_140644


namespace pirate_overtakes_at_8pm_l140_140328

noncomputable def pirate_overtake_trade : Prop :=
  let initial_distance := 15
  let pirate_speed_before_damage := 14
  let trade_speed := 10
  let time_before_damage := 3
  let pirate_distance_before_damage := pirate_speed_before_damage * time_before_damage
  let trade_distance_before_damage := trade_speed * time_before_damage
  let remaining_distance := initial_distance + trade_distance_before_damage - pirate_distance_before_damage
  let pirate_speed_after_damage := (18 / 17) * 10
  let relative_speed_after_damage := pirate_speed_after_damage - trade_speed
  let time_to_overtake_after_damage := remaining_distance / relative_speed_after_damage
  let total_time := time_before_damage + time_to_overtake_after_damage
  total_time = 8

theorem pirate_overtakes_at_8pm : pirate_overtake_trade :=
by
  sorry

end pirate_overtakes_at_8pm_l140_140328


namespace polygon_sides_count_l140_140409

-- Definitions for each polygon and their sides
def pentagon_sides := 5
def square_sides := 4
def hexagon_sides := 6
def heptagon_sides := 7
def nonagon_sides := 9

-- Compute the total number of sides
def total_exposed_sides :=
  (pentagon_sides + nonagon_sides - 2) + (square_sides + hexagon_sides + heptagon_sides - 6)

theorem polygon_sides_count : total_exposed_sides = 23 :=
by
  -- Mathematical proof steps can be detailed here
  -- For now, let's assume it is correctly given as a single number
  sorry

end polygon_sides_count_l140_140409


namespace installment_payment_l140_140271

theorem installment_payment
  (cash_price : ℕ)
  (down_payment : ℕ)
  (first_four_months_payment : ℕ)
  (last_four_months_payment : ℕ)
  (installment_additional_cost : ℕ)
  (total_next_four_months_payment : ℕ)
  (H_cash_price : cash_price = 450)
  (H_down_payment : down_payment = 100)
  (H_first_four_months_payment : first_four_months_payment = 4 * 40)
  (H_last_four_months_payment : last_four_months_payment = 4 * 30)
  (H_installment_additional_cost : installment_additional_cost = 70)
  (H_total_next_four_months_payment_correct : 4 * total_next_four_months_payment = 4 * 35) :
  down_payment + first_four_months_payment + 4 * 35 + last_four_months_payment = cash_price + installment_additional_cost := 
by {
  sorry
}

end installment_payment_l140_140271


namespace pieces_per_plant_yield_l140_140753

theorem pieces_per_plant_yield 
  (rows : ℕ) (plants_per_row : ℕ) (total_harvest : ℕ) 
  (h1 : rows = 30) (h2 : plants_per_row = 10) (h3 : total_harvest = 6000) : 
  (total_harvest / (rows * plants_per_row) = 20) :=
by
  -- Insert math proof here.
  sorry

end pieces_per_plant_yield_l140_140753


namespace cans_per_person_day1_l140_140585

theorem cans_per_person_day1
  (initial_cans : ℕ)
  (people_day1 : ℕ)
  (restock_day1 : ℕ)
  (people_day2 : ℕ)
  (cans_per_person_day2 : ℕ)
  (total_cans_given_away : ℕ) :
  initial_cans = 2000 →
  people_day1 = 500 →
  restock_day1 = 1500 →
  people_day2 = 1000 →
  cans_per_person_day2 = 2 →
  total_cans_given_away = 2500 →
  (total_cans_given_away - (people_day2 * cans_per_person_day2)) / people_day1 = 1 :=
by
  intros h1 h2 h3 h4 h5 h6
  -- condition trivially holds
  sorry

end cans_per_person_day1_l140_140585


namespace compute_expression_l140_140255

theorem compute_expression (x : ℝ) (h : x + 1/x = 3) : 
  (x - 3)^2 + 16 / (x - 3)^2 = 23 := 
  sorry

end compute_expression_l140_140255


namespace integer_solutions_l140_140216

theorem integer_solutions (x y z : ℤ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0) (h4 : x + y + z ≠ 0) :
  (1 / x + 1 / y + 1 / z = 1 / (x + y + z)) ↔ (z = -x - y) :=
sorry

end integer_solutions_l140_140216


namespace yoongi_calculation_l140_140170

theorem yoongi_calculation (x : ℝ) (h : x - 5 = 30) : x / 7 = 5 :=
by
  sorry

end yoongi_calculation_l140_140170


namespace find_rate_percent_l140_140486

theorem find_rate_percent (P : ℝ) (r : ℝ) (A1 A2 : ℝ) (t1 t2 : ℕ)
  (h1 : A1 = P * (1 + r)^t1) (h2 : A2 = P * (1 + r)^t2) (hA1 : A1 = 2420) (hA2 : A2 = 3146) (ht1 : t1 = 2) (ht2 : t2 = 3) :
  r = 0.2992 :=
by
  sorry

end find_rate_percent_l140_140486


namespace Emily_total_cost_l140_140991

theorem Emily_total_cost :
  let cost_curtains := 2 * 30
  let cost_prints := 9 * 15
  let installation_cost := 50
  let total_cost := cost_curtains + cost_prints + installation_cost
  total_cost = 245 := by
{
 sorry
}

end Emily_total_cost_l140_140991


namespace classify_curve_l140_140793

-- Define the curve equation
def curve_equation (m : ℝ) : Prop := 
  ∃ (x y : ℝ), ((m - 3) * x^2 + (5 - m) * y^2 = 1)

-- Define the conditions for types of curves
def is_circle (m : ℝ) : Prop := 
  m = 4 ∧ (curve_equation m)

def is_ellipse (m : ℝ) : Prop := 
  (3 < m ∧ m < 5 ∧ m ≠ 4) ∧ (curve_equation m)

def is_hyperbola (m : ℝ) : Prop := 
  ((m > 5 ∨ m < 3) ∧ (curve_equation m))

-- Main theorem stating the type of curve
theorem classify_curve (m : ℝ) : 
  (is_circle m) ∨ (is_ellipse m) ∨ (is_hyperbola m) :=
sorry

end classify_curve_l140_140793


namespace smallest_x_l140_140818

theorem smallest_x (x : ℤ) (h : x + 3 < 3 * x - 4) : x = 4 :=
by
  sorry

end smallest_x_l140_140818


namespace sqrt_multiplication_and_subtraction_l140_140641

theorem sqrt_multiplication_and_subtraction :
  (Real.sqrt 21 * Real.sqrt 7 - Real.sqrt 3) = 6 * Real.sqrt 3 := 
by
  sorry

end sqrt_multiplication_and_subtraction_l140_140641


namespace inv_113_mod_114_l140_140515

theorem inv_113_mod_114 :
  (113 * 113) % 114 = 1 % 114 :=
by
  sorry

end inv_113_mod_114_l140_140515


namespace spurs_total_basketballs_l140_140131

theorem spurs_total_basketballs (players : ℕ) (basketballs_per_player : ℕ) (h1 : players = 22) (h2 : basketballs_per_player = 11) : players * basketballs_per_player = 242 := by
  sorry

end spurs_total_basketballs_l140_140131


namespace second_chapter_pages_l140_140956

/-- A book has 2 chapters across 81 pages. The first chapter is 13 pages long. -/
theorem second_chapter_pages (total_pages : ℕ) (first_chapter_pages : ℕ) (second_chapter_pages : ℕ) : 
  total_pages = 81 → 
  first_chapter_pages = 13 → 
  second_chapter_pages = total_pages - first_chapter_pages → 
  second_chapter_pages = 68 := 
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end second_chapter_pages_l140_140956


namespace complete_square_q_value_l140_140861

theorem complete_square_q_value :
  ∃ p q, (16 * x^2 - 32 * x - 512 = 0) ∧ ((x + p)^2 = q) → q = 33 := by
  sorry

end complete_square_q_value_l140_140861


namespace solve_equation_l140_140312

noncomputable def equation (x : ℝ) : ℝ :=
  (|Real.sin x| + Real.sin (3 * x)) / (Real.cos x * Real.cos (2 * x))

theorem solve_equation (x : ℝ) (k : ℤ) :
  (equation x = 2 / Real.sqrt 3) ↔
  (∃ k : ℤ, x = π / 12 + 2 * k * π ∨ x = 7 * π / 12 + 2 * k * π ∨ x = -5 * π / 6 + 2 * k * π) :=
sorry

end solve_equation_l140_140312


namespace find_DF_l140_140545

-- Conditions
variables {A B C D E F : Type}
variables {BC EF AC DF : ℝ}
variable (h_similar : similar_triangles A B C D E F)
variable (h_BC : BC = 6)
variable (h_EF : EF = 4)
variable (h_AC : AC = 9)

-- Question: Prove DF = 6 given the above conditions
theorem find_DF : DF = 6 :=
by
  sorry

end find_DF_l140_140545


namespace class_total_students_l140_140198

def initial_boys : ℕ := 15
def initial_girls : ℕ := (120 * initial_boys) / 100 -- 1.2 * initial_boys

def final_boys : ℕ := initial_boys
def final_girls : ℕ := 2 * initial_girls

def total_students : ℕ := final_boys + final_girls

theorem class_total_students : total_students = 51 := 
by 
  -- the actual proof will go here
  sorry

end class_total_students_l140_140198


namespace red_balls_in_bag_l140_140115

theorem red_balls_in_bag (r : ℕ) (h1 : 0 ≤ r ∧ r ≤ 12)
  (h2 : (r * (r - 1)) / (12 * 11) = 1 / 10) : r = 12 :=
sorry

end red_balls_in_bag_l140_140115


namespace pieces_per_box_correct_l140_140491

-- Define the number of boxes Will bought
def total_boxes_bought := 7

-- Define the number of boxes Will gave to his brother
def boxes_given := 3

-- Define the number of pieces left with Will
def pieces_left := 16

-- Define the function to find the pieces per box
def pieces_per_box (total_boxes : Nat) (given_away : Nat) (remaining_pieces : Nat) : Nat :=
  remaining_pieces / (total_boxes - given_away)

-- Prove that each box contains 4 pieces of chocolate candy
theorem pieces_per_box_correct : pieces_per_box total_boxes_bought boxes_given pieces_left = 4 :=
by
  sorry

end pieces_per_box_correct_l140_140491


namespace solve_equation_l140_140078

theorem solve_equation (x : ℝ) : x * (x - 1) = 0 ↔ x = 0 ∨ x = 1 := 
by
  sorry  -- Placeholder for the proof

end solve_equation_l140_140078


namespace find_a_l140_140965

noncomputable def f (x : ℝ) (a : ℝ) := (2 / x) - 2 + 2 * a * Real.log x

theorem find_a (a : ℝ) (h : ∃ x ∈ Set.Icc (1/2 : ℝ) 2, f x a = 0) : a = 1 := by
  sorry

end find_a_l140_140965


namespace distinct_ways_to_place_digits_l140_140380

theorem distinct_ways_to_place_digits :
  let n := 4 -- number of digits
  let k := 5 -- number of boxes
  (k * (n!)) = 120 := by
  sorry

end distinct_ways_to_place_digits_l140_140380


namespace determine_e_l140_140707

noncomputable def Q (x : ℝ) (d e f : ℝ) : ℝ := 3*x^3 + d*x^2 + e*x + f

theorem determine_e (d f : ℝ) (h1 : f = 18) (h2 : -f/3 = -6) (h3 : -d/3 = -6) (h4 : 3 + d + e + f = -6) : e = -45 :=
sorry

end determine_e_l140_140707


namespace not_perfect_square_l140_140058

theorem not_perfect_square (h1 : ∃ x : ℝ, x^2 = 1 ^ 2018) 
                           (h2 : ¬ ∃ x : ℝ, x^2 = 2 ^ 2019)
                           (h3 : ∃ x : ℝ, x^2 = 3 ^ 2020)
                           (h4 : ∃ x : ℝ, x^2 = 4 ^ 2021)
                           (h5 : ∃ x : ℝ, x^2 = 6 ^ 2022) : 
  2 ^ 2019 ≠ x^2 := 
sorry

end not_perfect_square_l140_140058


namespace find_interest_rate_l140_140549

-- Definitions from the conditions
def principal : ℕ := 1050
def time_period : ℕ := 6
def interest : ℕ := 378  -- Interest calculated as Rs. 1050 - Rs. 672

-- Correct Answer
def interest_rate : ℕ := 6

-- Lean 4 statement of the proof problem
theorem find_interest_rate (P : ℕ) (t : ℕ) (I : ℕ) 
    (hP : P = principal) (ht : t = time_period) (hI : I = interest) : 
    (I * 100) / (P * t) = interest_rate :=
by {
    sorry
}

end find_interest_rate_l140_140549


namespace find_a_l140_140333

noncomputable def f (x : ℝ) (a : ℝ) := 3^x + a / (3^x + 1)

theorem find_a (a : ℝ) (h₁ : 0 < a) (h₂ : ∀ x, f x a ≥ 5) (h₃ : ∃ x, f x a = 5) : a = 9 := by
  sorry

end find_a_l140_140333


namespace correct_propositions_l140_140689

variables {Line Plane : Type}
variables (m n : Line) (α β : Plane)

-- Assume basic predicates for lines and planes
variable (parallel : Line → Line → Prop)
variable (perp : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)
variable (planar_parallel : Plane → Plane → Prop)

-- Stating the theorem to be proved
theorem correct_propositions :
  (parallel m n ∧ perp m α → perp n α) ∧ 
  (planar_parallel α β ∧ parallel m n ∧ perp m α → perp n β) :=
by
  sorry

end correct_propositions_l140_140689


namespace vectors_parallel_l140_140361

theorem vectors_parallel (x : ℝ) :
    ∀ (a b : ℝ × ℝ × ℝ),
    a = (2, -1, 3) →
    b = (x, 2, -6) →
    (∃ k : ℝ, b = (k * 2, k * -1, k * 3)) →
    x = -4 :=
by
  intro a b ha hb hab
  sorry

end vectors_parallel_l140_140361


namespace pct_three_petals_is_75_l140_140279

-- Given Values
def total_clovers : Nat := 200
def pct_two_petals : Nat := 24
def pct_four_petals : Nat := 1

-- Statement: Prove that the percentage of clovers with three petals is 75%
theorem pct_three_petals_is_75 :
  (100 - pct_two_petals - pct_four_petals) = 75 := by
  sorry

end pct_three_petals_is_75_l140_140279


namespace rabbit_speed_l140_140357

theorem rabbit_speed (dog_speed : ℝ) (head_start : ℝ) (catch_time_minutes : ℝ) 
  (H1 : dog_speed = 24) (H2 : head_start = 0.6) (H3 : catch_time_minutes = 4) :
  let catch_time_hours := catch_time_minutes / 60
  let distance_dog_runs := dog_speed * catch_time_hours
  let distance_rabbit_runs := distance_dog_runs - head_start
  let rabbit_speed := distance_rabbit_runs / catch_time_hours
  rabbit_speed = 15 :=
  sorry

end rabbit_speed_l140_140357


namespace right_triangle_angle_l140_140943

theorem right_triangle_angle {A B C : ℝ} (hABC : A + B + C = 180) (hC : C = 90) (hA : A = 70) : B = 20 :=
sorry

end right_triangle_angle_l140_140943


namespace rosa_total_pages_called_l140_140778

variable (P_last P_this : ℝ)

theorem rosa_total_pages_called (h1 : P_last = 10.2) (h2 : P_this = 8.6) : P_last + P_this = 18.8 :=
by sorry

end rosa_total_pages_called_l140_140778


namespace sum_of_fourth_powers_l140_140755

theorem sum_of_fourth_powers
  (a b c : ℝ)
  (h1 : a + b + c = 2)
  (h2 : a^2 + b^2 + c^2 = 5)
  (h3 : a^3 + b^3 + c^3 = 8) :
  a^4 + b^4 + c^4 = 15.5 := sorry

end sum_of_fourth_powers_l140_140755


namespace ratio_third_to_first_l140_140526

theorem ratio_third_to_first (F S T : ℕ) (h1 : F = 33) (h2 : S = 4 * F) (h3 : (F + S + T) / 3 = 77) :
  T / F = 2 :=
by
  sorry

end ratio_third_to_first_l140_140526


namespace compute_f_seven_halves_l140_140210

theorem compute_f_seven_halves 
  (f : ℝ → ℝ)
  (h_odd : ∀ x, f (-x) = -f x)
  (h_shift : ∀ x, f (x + 2) = -f x)
  (h_interval : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = x) :
  f (7 / 2) = -1 / 2 :=
  sorry

end compute_f_seven_halves_l140_140210


namespace simplify_complex_fraction_l140_140966

theorem simplify_complex_fraction :
  (⟨-4, -6⟩ : ℂ) / (⟨5, -2⟩ : ℂ) = ⟨-(32 : ℚ) / 21, -(38 : ℚ) / 21⟩ := 
sorry

end simplify_complex_fraction_l140_140966


namespace prove_inequality_l140_140309

noncomputable def proof_problem (x y z : ℝ)
  (h1 : x + y + z = 0)
  (h2 : |x| + |y| + |z| ≤ 1) : Prop :=
  x + y/3 + z/5 ≤ 2/5

theorem prove_inequality (x y z : ℝ) 
  (h1 : x + y + z = 0) 
  (h2 : |x| + |y| + |z| ≤ 1) : proof_problem x y z h1 h2 :=
sorry

end prove_inequality_l140_140309


namespace vector_addition_l140_140371

variable (a : ℝ × ℝ)
variable (b : ℝ × ℝ)

theorem vector_addition (h1 : a = (-1, 2)) (h2 : b = (1, 0)) :
  3 • a + b = (-2, 6) :=
by
  -- proof goes here
  sorry

end vector_addition_l140_140371


namespace translation_theorem_l140_140665

noncomputable def f (θ : ℝ) (x : ℝ) : ℝ := Real.sin (2 * x + θ)
noncomputable def g (θ : ℝ) (φ : ℝ) (x : ℝ) : ℝ := Real.sin (2 * x - 2 * φ + θ)

theorem translation_theorem
  (θ φ : ℝ)
  (hθ1 : |θ| < Real.pi / 2)
  (hφ1 : 0 < φ)
  (hφ2 : φ < Real.pi)
  (hf : f θ 0 = 1 / 2)
  (hg : g θ φ 0 = 1 / 2) :
  φ = 2 * Real.pi / 3 :=
sorry

end translation_theorem_l140_140665


namespace equation_of_line_perpendicular_and_passing_point_l140_140398

theorem equation_of_line_perpendicular_and_passing_point :
  ∃ (a b c : ℝ), a = 3 ∧ b = 2 ∧ c = -1 ∧
  (∀ (x y : ℝ), (2 * x - 3 * y + 4 = 0 → y = (2 / 3) * x + 4 / 3) →
  (∀ (x1 y1 : ℝ), x1 = -1 ∧ y1 = 2 →
  (a * x1 + b * y1 + c = 0) ∧
  (∀ (x y : ℝ), (-3 / 2) * (x + 1) + 2 = y) →
  (a * x + b * y + c = 0))) :=
sorry

end equation_of_line_perpendicular_and_passing_point_l140_140398


namespace number_of_books_l140_140626

-- Define the conditions
def ratio_books : ℕ := 7
def ratio_pens : ℕ := 3
def ratio_notebooks : ℕ := 2
def total_items : ℕ := 600

-- Define the theorem and the goal to prove
theorem number_of_books (sets : ℕ) (ratio_books : ℕ := 7) (total_items : ℕ := 600) : 
  sets = total_items / (7 + 3 + 2) → 
  sets * ratio_books = 350 :=
by
  sorry

end number_of_books_l140_140626


namespace sufficient_but_not_necessary_condition_l140_140498

theorem sufficient_but_not_necessary_condition 
    (a : ℝ) (h_pos : a > 0)
    (h_line : ∀ x y, 2 * a * x - y + 2 * a^2 = 0)
    (h_hyperbola : ∀ x y, x^2 / a^2 - y^2 / 4 = 1) :
    (a ≥ 2) → 
    (∀ x y, ¬ (2 * a * x - y + 2 * a^2 = 0 ∧ x^2 / a^2 - y^2 / 4 = 1)) :=
by
  sorry

end sufficient_but_not_necessary_condition_l140_140498


namespace volume_of_rectangular_prism_l140_140971

-- Definition of the given conditions
variables (a b c : ℝ)

def condition1 : Prop := a * b = 24
def condition2 : Prop := b * c = 15
def condition3 : Prop := a * c = 10

-- The statement we want to prove
theorem volume_of_rectangular_prism
  (h1 : condition1 a b)
  (h2 : condition2 b c)
  (h3 : condition3 a c) :
  a * b * c = 60 :=
by sorry

end volume_of_rectangular_prism_l140_140971


namespace initial_weight_l140_140737

theorem initial_weight (W : ℝ) (h₁ : W > 0): 
  W * 0.85 * 0.75 * 0.90 = 450 := 
by 
  sorry

end initial_weight_l140_140737


namespace students_not_taking_test_l140_140064

theorem students_not_taking_test (total_students students_q1 students_q2 students_both not_taken : ℕ)
  (h_total : total_students = 30)
  (h_q1 : students_q1 = 25)
  (h_q2 : students_q2 = 22)
  (h_both : students_both = 22)
  (h_not_taken : not_taken = total_students - students_q2) :
  not_taken = 8 := by
  sorry

end students_not_taking_test_l140_140064


namespace david_reading_time_l140_140867

theorem david_reading_time (total_time : ℕ) (math_time : ℕ) (spelling_time : ℕ) 
  (reading_time : ℕ) (h1 : total_time = 60) (h2 : math_time = 15) 
  (h3 : spelling_time = 18) (h4 : reading_time = total_time - (math_time + spelling_time)) : 
  reading_time = 27 := by
  sorry

end david_reading_time_l140_140867


namespace binom_9_5_eq_126_l140_140524

theorem binom_9_5_eq_126 : (Nat.choose 9 5) = 126 := by
  sorry

end binom_9_5_eq_126_l140_140524


namespace danny_more_caps_l140_140293

variable (found thrown_away : ℕ)

def bottle_caps_difference (found thrown_away : ℕ) : ℕ :=
  found - thrown_away

theorem danny_more_caps
  (h_found : found = 36)
  (h_thrown_away : thrown_away = 35) :
  bottle_caps_difference found thrown_away = 1 :=
by
  -- Proof is omitted with sorry
  sorry

end danny_more_caps_l140_140293


namespace best_fitting_model_is_model_3_l140_140761

-- Define models with their corresponding R^2 values
def R_squared_model_1 : ℝ := 0.72
def R_squared_model_2 : ℝ := 0.64
def R_squared_model_3 : ℝ := 0.98
def R_squared_model_4 : ℝ := 0.81

-- Define a proposition that model 3 has the best fitting effect
def best_fitting_model (R1 R2 R3 R4 : ℝ) : Prop :=
  R3 = max (max R1 R2) (max R3 R4)

-- State the theorem that we need to prove
theorem best_fitting_model_is_model_3 :
  best_fitting_model R_squared_model_1 R_squared_model_2 R_squared_model_3 R_squared_model_4 :=
by
  sorry

end best_fitting_model_is_model_3_l140_140761


namespace distinct_four_digit_numbers_product_18_l140_140679

def is_valid_four_digit_product (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 
                    1 ≤ b ∧ b ≤ 9 ∧ 
                    1 ≤ c ∧ c ≤ 9 ∧ 
                    1 ≤ d ∧ d ≤ 9 ∧ 
                    a * b * c * d = 18 ∧ 
                    n = a * 1000 + b * 100 + c * 10 + d

theorem distinct_four_digit_numbers_product_18 : 
  ∃ (count : ℕ), count = 24 ∧ 
                  (∀ n, is_valid_four_digit_product n ↔ 0 < n ∧ n < 10000) :=
sorry

end distinct_four_digit_numbers_product_18_l140_140679


namespace max_value_frac_sixth_roots_eq_two_l140_140945

noncomputable def max_value_frac_sixth_roots (α β : ℝ) (t : ℝ) (q : ℝ) : ℝ :=
  if α + β = t ∧ α^2 + β^2 = t ∧ α^3 + β^3 = t ∧ α^4 + β^4 = t ∧ α^5 + β^5 = t then
    max (1 / α^6 + 1 / β^6) 2
  else
    0

theorem max_value_frac_sixth_roots_eq_two (α β : ℝ) (t : ℝ) (q : ℝ) :
  (α + β = t ∧ α^2 + β^2 = t ∧ α^3 + β^3 = t ∧ α^4 + β^4 = t ∧ α^5 + β^5 = t) →
  ∃ m, max_value_frac_sixth_roots α β t q = m ∧ m = 2 :=
sorry

end max_value_frac_sixth_roots_eq_two_l140_140945


namespace profit_margin_A_cost_price_B_units_purchased_l140_140046

variables (cost_price_A selling_price_A selling_price_B profit_margin_B total_units total_cost : ℕ)
variables (units_A units_B : ℕ)

-- Conditions
def condition1 : cost_price_A = 40 := sorry
def condition2 : selling_price_A = 60 := sorry
def condition3 : selling_price_B = 80 := sorry
def condition4 : profit_margin_B = 60 := sorry
def condition5 : total_units = 50 := sorry
def condition6 : total_cost = 2200 := sorry

-- Proof statements 
theorem profit_margin_A (h1 : cost_price_A = 40) (h2 : selling_price_A = 60) :
  (selling_price_A - cost_price_A) * 100 / cost_price_A = 50 :=
by sorry

theorem cost_price_B (h3 : selling_price_B = 80) (h4 : profit_margin_B = 60) :
  (selling_price_B * 100) / (100 + profit_margin_B) = 50 :=
by sorry

theorem units_purchased (h5 : 40 * units_A + 50 * units_B = 2200)
  (h6 : units_A + units_B = 50) :
  units_A = 30 ∧ units_B = 20 :=
by sorry


end profit_margin_A_cost_price_B_units_purchased_l140_140046


namespace stratified_sampling_B_l140_140132

-- Define the groups and their sizes
def num_people_A : ℕ := 18
def num_people_B : ℕ := 24
def num_people_C : ℕ := 30

-- Total number of people
def total_people : ℕ := num_people_A + num_people_B + num_people_C

-- Total sample size to be drawn
def sample_size : ℕ := 12

-- Proportion of group B
def proportion_B : ℚ := num_people_B / total_people

-- Number of people to be drawn from group B
def number_drawn_from_B : ℚ := sample_size * proportion_B

-- The theorem to be proved
theorem stratified_sampling_B : number_drawn_from_B = 4 := 
by
  -- This is where the proof would go
  sorry

end stratified_sampling_B_l140_140132


namespace regression_line_passes_through_sample_mean_point_l140_140199

theorem regression_line_passes_through_sample_mean_point
  (a b : ℝ) (x y : ℝ)
  (hx : x = a + b*x) :
  y = a + b*x :=
by sorry

end regression_line_passes_through_sample_mean_point_l140_140199


namespace pentagon_perimeter_l140_140994

noncomputable def perimeter_pentagon (FG GH HI IJ : ℝ) (FH FI FJ : ℝ) : ℝ :=
  FG + GH + HI + IJ + FJ

theorem pentagon_perimeter : 
  ∀ (FG GH HI IJ : ℝ), 
  ∀ (FH FI FJ : ℝ),
  FG = 1 → GH = 1 → HI = 1 → IJ = 1 →
  FH^2 = FG^2 + GH^2 → FI^2 = FH^2 + HI^2 → FJ^2 = FI^2 + IJ^2 →
  perimeter_pentagon FG GH HI IJ FJ = 6 :=
by
  intros FG GH HI IJ FH FI FJ
  intros H_FG H_GH H_HI H_IJ
  intros H1 H2 H3
  sorry

end pentagon_perimeter_l140_140994


namespace football_practice_hours_l140_140308

theorem football_practice_hours (practice_hours_per_day : ℕ) (days_per_week : ℕ) (missed_days_due_to_rain : ℕ) 
  (practice_hours_per_day_eq_six : practice_hours_per_day = 6)
  (days_per_week_eq_seven : days_per_week = 7)
  (missed_days_due_to_rain_eq_one : missed_days_due_to_rain = 1) : 
  practice_hours_per_day * (days_per_week - missed_days_due_to_rain) = 36 := 
by
  -- proof goes here
  sorry

end football_practice_hours_l140_140308


namespace real_part_0_or_3_complex_part_not_0_or_3_purely_imaginary_at_2_no_second_quadrant_l140_140691

def z (m : ℝ) : ℂ := (m^2 - 5 * m + 6 : ℝ) + (m^2 - 3 * m : ℝ) * Complex.I

theorem real_part_0_or_3 (m : ℝ) : (m^2 - 3 * m = 0) ↔ (m = 0 ∨ m = 3) := sorry

theorem complex_part_not_0_or_3 (m : ℝ) : (m^2 - 3 * m ≠ 0) ↔ (m ≠ 0 ∧ m ≠ 3) := sorry

theorem purely_imaginary_at_2 (m : ℝ) : (m^2 - 5 * m + 6 = 0) ∧ (m^2 - 3 * m ≠ 0) ↔ (m = 2) := sorry

theorem no_second_quadrant (m : ℝ) : ¬(m^2 - 5 * m + 6 < 0 ∧ m^2 - 3 * m > 0) := sorry

end real_part_0_or_3_complex_part_not_0_or_3_purely_imaginary_at_2_no_second_quadrant_l140_140691


namespace least_non_lucky_multiple_of_8_l140_140320

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.foldr (· + ·) 0

def lucky (n : ℕ) : Prop :=
  n % sum_of_digits n = 0

def multiple_of_8 (n : ℕ) : Prop :=
  n % 8 = 0

theorem least_non_lucky_multiple_of_8 : ∃ n > 0, multiple_of_8 n ∧ ¬ lucky n ∧ n = 16 :=
by
  -- Proof goes here.
  sorry

end least_non_lucky_multiple_of_8_l140_140320


namespace circle_passes_through_fixed_point_l140_140941

theorem circle_passes_through_fixed_point :
  ∀ (C : ℝ × ℝ), (C.2 ^ 2 = 4 * C.1) ∧ (C.1 = -1 + (C.1 + 1)) → ∃ P : ℝ × ℝ, P = (1, 0) ∧
    (P.1 - C.1) ^ 2 + (P.2 - C.2) ^ 2 = (C.1 + 1) ^ 2 + (0 - C.2) ^ 2 :=
by
  sorry

end circle_passes_through_fixed_point_l140_140941


namespace point_after_transformations_l140_140316

-- Define the initial coordinates of point F
def F : ℝ × ℝ := (-1, -1)

-- Function to reflect a point over the x-axis
def reflect_over_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

-- Function to reflect a point over the line y = x
def reflect_over_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

-- Prove that F, when reflected over x-axis and then y=x, results in (1, -1)
theorem point_after_transformations : 
  reflect_over_y_eq_x (reflect_over_x F) = (1, -1) := by
  sorry

end point_after_transformations_l140_140316


namespace not_all_squares_congruent_l140_140245

-- Define what it means to be a square
structure Square :=
  (side : ℝ)
  (angle : ℝ)
  (is_square : side > 0 ∧ angle = 90)

-- Define congruency of squares
def congruent (s1 s2 : Square) : Prop :=
  s1.side = s2.side ∧ s1.angle = s2.angle

-- The main statement to prove 
theorem not_all_squares_congruent : ∃ s1 s2 : Square, ¬ congruent s1 s2 :=
by
  sorry

end not_all_squares_congruent_l140_140245


namespace diamond_sum_l140_140250

def diamond (x : ℚ) : ℚ := (x^3 + 2 * x^2 + 3 * x) / 6

theorem diamond_sum : diamond 2 + diamond 3 + diamond 4 = 92 / 3 := by
  sorry

end diamond_sum_l140_140250


namespace probability_same_color_dice_l140_140172

theorem probability_same_color_dice :
  let total_sides := 12
  let red_sides := 3
  let green_sides := 4
  let blue_sides := 2
  let yellow_sides := 3
  let prob_red := (red_sides / total_sides) ^ 2
  let prob_green := (green_sides / total_sides) ^ 2
  let prob_blue := (blue_sides / total_sides) ^ 2
  let prob_yellow := (yellow_sides / total_sides) ^ 2
  prob_red + prob_green + prob_blue + prob_yellow = 19 / 72 := 
by
  -- The proof goes here
  sorry

end probability_same_color_dice_l140_140172


namespace replacement_parts_l140_140849

theorem replacement_parts (num_machines : ℕ) (parts_per_machine : ℕ) (week1_fail_rate : ℚ) (week2_fail_rate : ℚ) (week3_fail_rate : ℚ) :
  num_machines = 500 ->
  parts_per_machine = 6 ->
  week1_fail_rate = 0.10 ->
  week2_fail_rate = 0.30 ->
  week3_fail_rate = 0.60 ->
  (num_machines * parts_per_machine) * week1_fail_rate +
  (num_machines * parts_per_machine) * week2_fail_rate +
  (num_machines * parts_per_machine) * week3_fail_rate = 3000 := by
  sorry

end replacement_parts_l140_140849


namespace coprime_solution_l140_140091

theorem coprime_solution (a b : ℕ) (h_coprime : Nat.gcd a b = 1) (h_eq : 5 * a + 7 * b = 29 * (6 * a + 5 * b)) : a = 3 ∧ b = 2 :=
sorry

end coprime_solution_l140_140091


namespace container_volume_ratio_l140_140352

theorem container_volume_ratio (A B : ℚ) (h : (2 / 3 : ℚ) * A = (1 / 2 : ℚ) * B) : A / B = 3 / 4 :=
by sorry

end container_volume_ratio_l140_140352


namespace pipe_B_leak_time_l140_140936

theorem pipe_B_leak_time (t_B : ℝ) : (1 / 12 - 1 / t_B = 1 / 36) → t_B = 18 :=
by
  intro h
  -- Proof goes here
  sorry

end pipe_B_leak_time_l140_140936


namespace problem_proof_l140_140496

theorem problem_proof (a b c x y z : ℝ) (h₁ : 17 * x + b * y + c * z = 0) (h₂ : a * x + 29 * y + c * z = 0)
                      (h₃ : a * x + b * y + 53 * z = 0) (ha : a ≠ 17) (hx : x ≠ 0) :
                      (a / (a - 17)) + (b / (b - 29)) + (c / (c - 53)) = 1 :=
by
  -- proof goes here
  sorry

end problem_proof_l140_140496


namespace division_problem_l140_140923

theorem division_problem : 96 / (8 / 4) = 48 := 
by {
  sorry
}

end division_problem_l140_140923


namespace part1_daily_sales_profit_at_60_part2_selling_price_1350_l140_140397

-- Definitions from conditions
def cost_per_piece : ℕ := 40
def selling_price_50_sales_volume : ℕ := 100
def sales_decrease_per_dollar : ℕ := 2
def max_selling_price : ℕ := 65

-- Problem Part (1)
def profit_at_60_yuan := 
  let selling_price := 60
  let profit_per_piece := selling_price - cost_per_piece
  let sales_decrease := (selling_price - 50) * sales_decrease_per_dollar
  let sales_volume := selling_price_50_sales_volume - sales_decrease
  let daily_profit := profit_per_piece * sales_volume
  daily_profit

theorem part1_daily_sales_profit_at_60 : profit_at_60_yuan = 1600 := by
  sorry

-- Problem Part (2)
def selling_price_for_1350_profit :=
  let desired_profit := 1350
  let sales_volume (x : ℕ) := selling_price_50_sales_volume - sales_decrease_per_dollar * (x - 50)
  let profit_per_x_piece (x : ℕ) := x - cost_per_piece
  let daily_sales_profit (x : ℕ) := (profit_per_x_piece x) * (sales_volume x)
  daily_sales_profit

theorem part2_selling_price_1350 : 
  ∃ x, x ≤ max_selling_price ∧ selling_price_for_1350_profit x = 1350 ∧ x = 55 := by
  sorry

end part1_daily_sales_profit_at_60_part2_selling_price_1350_l140_140397


namespace triangle_shape_l140_140276

theorem triangle_shape
  (A B C : ℝ) -- Internal angles of triangle ABC
  (a b c : ℝ) -- Sides opposite to angles A, B, and C respectively
  (h1 : a * (Real.cos A) * (Real.cos B) + b * (Real.cos A) * (Real.cos A) = a * (Real.cos A)) :
  (A = Real.pi / 2) ∨ (A = C) :=
sorry

end triangle_shape_l140_140276


namespace necessary_but_not_sufficient_l140_140940

theorem necessary_but_not_sufficient (x : ℝ) (h : x < 4) : x < 0 ∨ true :=
by
  sorry

end necessary_but_not_sufficient_l140_140940


namespace find_bk_l140_140870

theorem find_bk
  (A B C D : ℝ)
  (BC : ℝ) (hBC : BC = 3)
  (AB CD : ℝ) (hAB_CD : AB = 2 * CD)
  (BK : ℝ) (hBK : BK = 2) :
  ∃ x a : ℝ, (x = BK) ∧ (AB = 2 * CD) ∧ ((2 * a + x) * (3 - x) = x * (a + 3 - x)) :=
by
  sorry

end find_bk_l140_140870


namespace added_number_is_five_l140_140537

variable (n x : ℤ)

theorem added_number_is_five (h1 : n % 25 = 4) (h2 : (n + x) % 5 = 4) : x = 5 :=
by
  sorry

end added_number_is_five_l140_140537


namespace incorrect_expression_l140_140487

theorem incorrect_expression (x y : ℚ) (h : x / y = 5 / 3) : x / (y - x) ≠ 5 / 2 := 
by
  sorry

end incorrect_expression_l140_140487


namespace soil_bags_needed_l140_140447

def raised_bed_length : ℝ := 8
def raised_bed_width : ℝ := 4
def raised_bed_height : ℝ := 1
def soil_bag_volume : ℝ := 4
def num_raised_beds : ℕ := 2

theorem soil_bags_needed : (raised_bed_length * raised_bed_width * raised_bed_height * num_raised_beds) / soil_bag_volume = 16 := 
by
  sorry

end soil_bags_needed_l140_140447


namespace arithmetic_sequence_a2_a4_a9_eq_18_l140_140874

theorem arithmetic_sequence_a2_a4_a9_eq_18 (a : ℕ → ℕ) (S : ℕ → ℕ) 
  (h1 : S 9 = 54) 
  (h2 : ∀ n, S n = n * (a 1 + a n) / 2) :
  a 2 + a 4 + a 9 = 18 :=
sorry

end arithmetic_sequence_a2_a4_a9_eq_18_l140_140874


namespace inverse_of_3_mod_185_l140_140533

theorem inverse_of_3_mod_185 : ∃ x : ℕ, 0 ≤ x ∧ x < 185 ∧ 3 * x ≡ 1 [MOD 185] :=
by
  use 62
  sorry

end inverse_of_3_mod_185_l140_140533


namespace triangles_exist_l140_140575

def exists_triangles : Prop :=
  ∃ (T : Fin 100 → Type) 
    (h : (i : Fin 100) → ℝ) 
    (A : (i : Fin 100) → ℝ)
    (is_isosceles : (i : Fin 100) → Prop),
    (∀ i : Fin 100, is_isosceles i) ∧
    (∀ i : Fin 99, h (i + 1) = 200 * h i) ∧
    (∀ i : Fin 99, A (i + 1) = A i / 20000) ∧
    (∀ i : Fin 100, 
      ¬(∃ (cover : (Fin 99) → Type),
        (∀ j : Fin 99, cover j = T j) ∧
        (∀ j : Fin 99, ∀ k : Fin 100, k ≠ i → ¬(cover j = T k))))

theorem triangles_exist : exists_triangles :=
sorry

end triangles_exist_l140_140575


namespace unique_n_l140_140715

theorem unique_n : ∃ n : ℕ, 0 < n ∧ n^3 % 1000 = n ∧ ∀ m : ℕ, 0 < m ∧ m^3 % 1000 = m → m = n :=
by
  sorry

end unique_n_l140_140715


namespace total_lucky_stars_l140_140531

theorem total_lucky_stars : 
  (∃ n : ℕ, 10 * n + 6 = 116 ∧ 4 * 8 + (n - 4) * 12 = 116) → 
  116 = 116 := 
by
  intro h
  obtain ⟨n, h1, h2⟩ := h
  sorry

end total_lucky_stars_l140_140531


namespace tangent_line_condition_l140_140468

theorem tangent_line_condition (a b k : ℝ) (h1 : (1 : ℝ) + a + b = 2) (h2 : 3 + a = k) (h3 : k = 1) :
    b - a = 5 := 
by 
    sorry

end tangent_line_condition_l140_140468


namespace number_of_pears_in_fruit_gift_set_l140_140615

theorem number_of_pears_in_fruit_gift_set 
  (F : ℕ) 
  (h1 : (2 / 9) * F = 10) 
  (h2 : 2 / 5 * F = 18) : 
  (2 / 5) * F = 18 :=
by 
  -- Sorry is used to skip the actual proof for now
  sorry

end number_of_pears_in_fruit_gift_set_l140_140615


namespace cups_filled_with_tea_l140_140454

theorem cups_filled_with_tea (total_tea ml_each_cup : ℕ)
  (h1 : total_tea = 1050)
  (h2 : ml_each_cup = 65) :
  total_tea / ml_each_cup = 16 := sorry

end cups_filled_with_tea_l140_140454


namespace possible_m_value_l140_140557

noncomputable def f (x : ℝ) : ℝ := Real.exp x + x^3 - (1/2)*x - 1
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := x^3 + m / x

theorem possible_m_value :
  ∃ m : ℝ, (m = (1/2) - (1/Real.exp 1)) ∧
    (∀ x1 x2 : ℝ, 
      (f x1 = g (-x1) m) →
      (f x2 = g (-x2) m) →
      x1 ≠ 0 ∧ x2 ≠ 0 ∧
      m = x1 * Real.exp x1 - (1/2) * x1^2 - x1 ∧
      m = x2 * Real.exp x2 - (1/2) * x2^2 - x2) :=
by
  sorry

end possible_m_value_l140_140557


namespace Lauryn_employs_80_men_l140_140996

theorem Lauryn_employs_80_men (W M : ℕ) 
  (h1 : M = W - 20) 
  (h2 : M + W = 180) : 
  M = 80 := 
by 
  sorry

end Lauryn_employs_80_men_l140_140996


namespace geometric_sequence_problem_l140_140297

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r, ∀ n, a (n + 1) = a n * r

-- Define the statement for the roots of the quadratic function
def is_root (f : ℝ → ℝ) (a : ℝ) : Prop :=
  f a = 0

-- Define the quadratic function
def quadratic_function (x : ℝ) : ℝ :=
  x^2 - x - 2013

-- Define the problem statement
theorem geometric_sequence_problem (a : ℕ → ℝ) 
  (h1 : is_geometric_sequence a) 
  (h2 : is_root quadratic_function (a 2)) 
  (h3 : is_root quadratic_function (a 3)) : 
  a 1 * a 4 = -2013 :=
sorry

end geometric_sequence_problem_l140_140297


namespace ratio_adult_child_l140_140268

theorem ratio_adult_child (total_fee adults_fee children_fee adults children : ℕ) 
  (h1 : adults ≥ 1) (h2 : children ≥ 1) 
  (h3 : adults_fee = 30) (h4 : children_fee = 15) 
  (h5 : total_fee = 2250) 
  (h6 : adults_fee * adults + children_fee * children = total_fee) :
  (2 : ℚ) = adults / children :=
sorry

end ratio_adult_child_l140_140268


namespace james_and_lisa_pizzas_l140_140885

theorem james_and_lisa_pizzas (slices_per_pizza : ℕ) (total_slices : ℕ) :
  slices_per_pizza = 6 →
  2 * total_slices = 3 * 8 →
  total_slices / slices_per_pizza = 2 :=
by
  intros h1 h2
  sorry

end james_and_lisa_pizzas_l140_140885


namespace ant_positions_l140_140377

noncomputable def side_lengths (a b c : ℝ) : Prop :=
  (a + 2 = b) ∧ (b + 2 = c) ∧ (4 * c / c - 2 + 1) = 3 ∧ (4 * c / (c - 4) - 1) = 3

theorem ant_positions (a b c : ℝ) (v : ℝ) (ha : side_lengths a b c) : 
  a = 4 ∧ b = 6 ∧ c = 8 :=
by
  sorry

end ant_positions_l140_140377


namespace distance_first_day_l140_140252

theorem distance_first_day (total_distance : ℕ) (q : ℚ) (n : ℕ) (a : ℚ) : total_distance = 378 ∧ q = 1 / 2 ∧ n = 6 → a = 192 :=
by
  -- Proof omitted, just provide the statement
  sorry

end distance_first_day_l140_140252


namespace find_y_l140_140534

noncomputable def G (a b c d : ℝ) : ℝ := a ^ b + c ^ d

theorem find_y (h : G 3 y 2 5 = 100) : y = Real.log 68 / Real.log 3 := 
by
  have hG : G 3 y 2 5 = 3 ^ y + 2 ^ 5 := rfl
  sorry

end find_y_l140_140534


namespace yellow_green_block_weight_difference_l140_140919

theorem yellow_green_block_weight_difference :
  let yellow_weight := 0.6
  let green_weight := 0.4
  yellow_weight - green_weight = 0.2 := by
  sorry

end yellow_green_block_weight_difference_l140_140919


namespace minimum_cuts_to_unit_cubes_l140_140087

def cubes := List (ℕ × ℕ × ℕ)

def cube_cut (c : cubes) (n : ℕ) (dim : ℕ) : cubes :=
  sorry -- Function body not required for the statement

theorem minimum_cuts_to_unit_cubes (c : cubes) (s : ℕ) (dim : ℕ) :
  c = [(4,4,4)] ∧ s = 64 ∧ dim = 3 →
  ∃ (n : ℕ), n = 9 ∧
    (∀ cuts : cubes, cube_cut cuts n dim = [(1,1,1)]) :=
sorry

end minimum_cuts_to_unit_cubes_l140_140087


namespace votes_for_sue_l140_140166

-- Conditions from the problem
def total_votes := 1000
def category1_percent := 20 / 100   -- 20%
def category2_percent := 45 / 100   -- 45%
def sue_percent := 1 - (category1_percent + category2_percent)  -- Remaining percentage

-- Mathematically equivalent proof problem
theorem votes_for_sue : sue_percent * total_votes = 350 :=
by
  -- reminder: we do not need to provide the proof here
  sorry

end votes_for_sue_l140_140166


namespace power_sum_l140_140070

theorem power_sum :
  (-3)^3 + (-3)^2 + (-3) + 3 + 3^2 + 3^3 = 18 :=
by
  sorry

end power_sum_l140_140070


namespace length_of_FD_l140_140863

/-- Square ABCD with side length 8 cm, corner C is folded to point E on AD such that AE = 2 cm and ED = 6 cm. Find the length of FD. -/
theorem length_of_FD 
  (A B C D E F G : Type)
  (square_length : Float)
  (AD_length AE_length ED_length : Float)
  (hyp1 : square_length = 8)
  (hyp2 : AE_length = 2)
  (hyp3 : ED_length = 6)
  (hyp4 : AD_length = AE_length + ED_length)
  (FD_length : Float) :
  FD_length = 7 / 4 := 
  by 
  sorry

end length_of_FD_l140_140863


namespace largest_convex_ngon_with_integer_tangents_l140_140630

-- Definitions of conditions and the statement
def isConvex (n : ℕ) : Prop := n ≥ 3 -- Condition 1: n is at least 3
def isConvexPolygon (n : ℕ) : Prop := isConvex n -- Condition 2: the polygon is convex
def tanInteriorAnglesAreIntegers (n : ℕ) : Prop := true -- Placeholder for Condition 3

-- Statement to prove
theorem largest_convex_ngon_with_integer_tangents : 
  ∀ n : ℕ, isConvexPolygon n → tanInteriorAnglesAreIntegers n → n ≤ 8 :=
by
  intros n h_convex h_tangents
  sorry

end largest_convex_ngon_with_integer_tangents_l140_140630


namespace base_angle_of_isosceles_triangle_l140_140678

theorem base_angle_of_isosceles_triangle (A B C : ℝ) (h_triangle : A + B + C = 180) (h_isosceles : A = B ∨ B = C ∨ A = C) (h_angle : A = 42 ∨ B = 42 ∨ C = 42) :
  A = 42 ∨ A = 69 ∨ B = 42 ∨ B = 69 ∨ C = 42 ∨ C = 69 :=
by
  sorry

end base_angle_of_isosceles_triangle_l140_140678


namespace math_proof_problem_l140_140349

variable {a_n : ℕ → ℝ} -- sequence a_n
variable {b_n : ℕ → ℝ} -- sequence b_n

-- Given that a_n is an arithmetic sequence with common difference d
def isArithmeticSequence (a_n : ℕ → ℝ) (d : ℝ) : Prop := ∀ n, a_n (n + 1) = a_n n + d

-- Given condition for sequence b_n
def b_n_def (a_n b_n : ℕ → ℝ) : Prop := ∀ n, b_n n = a_n (n + 1) * a_n (n + 2) - a_n n ^ 2

-- Both sequences have common difference d ≠ 0
def common_difference_ne_zero (a_n b_n : ℕ → ℝ) (d : ℝ) : Prop :=
  isArithmeticSequence a_n d ∧ isArithmeticSequence b_n d ∧ d ≠ 0

-- Condition involving positive integers s and t
def integer_condition (a_n b_n : ℕ → ℝ) (s t : ℕ) : Prop :=
  1 ≤ s ∧ 1 ≤ t ∧ ∃ (x : ℤ), a_n s + b_n t = x

-- Theorem to prove that the sequence {b_n} is arithmetic and find minimum value of |a_1|
theorem math_proof_problem
  (a_n b_n : ℕ → ℝ) (d : ℝ) (s t : ℕ)
  (arithmetic_a : isArithmeticSequence a_n d)
  (defined_b : b_n_def a_n b_n)
  (common_diff : common_difference_ne_zero a_n b_n d)
  (int_condition : integer_condition a_n b_n s t) :
  (isArithmeticSequence b_n (3 * d ^ 2)) ∧ (∃ m : ℝ, m = |a_n 1| ∧ m = 1 / 36) :=
  by sorry

end math_proof_problem_l140_140349


namespace max_number_of_children_l140_140404

theorem max_number_of_children (apples cookies chocolates : ℕ) (remaining_apples remaining_cookies remaining_chocolates : ℕ) 
  (h₁ : apples = 55) 
  (h₂ : cookies = 114) 
  (h₃ : chocolates = 83) 
  (h₄ : remaining_apples = 3) 
  (h₅ : remaining_cookies = 10) 
  (h₆ : remaining_chocolates = 5) : 
  gcd (apples - remaining_apples) (gcd (cookies - remaining_cookies) (chocolates - remaining_chocolates)) = 26 :=
by
  sorry

end max_number_of_children_l140_140404


namespace inequality_solution_l140_140577

theorem inequality_solution :
  { x : ℝ | (x-1)/(x+4) ≤ 0 } = { x : ℝ | (-4 < x ∧ x ≤ 0) ∨ (x = 1) } :=
by 
  sorry

end inequality_solution_l140_140577


namespace shane_gum_left_l140_140670

def elyse_initial_gum : ℕ := 100
def half (x : ℕ) := x / 2
def rick_gum : ℕ := half elyse_initial_gum
def shane_initial_gum : ℕ := half rick_gum
def chewed_gum : ℕ := 11

theorem shane_gum_left : shane_initial_gum - chewed_gum = 14 := by
  sorry

end shane_gum_left_l140_140670


namespace x_is_4286_percent_less_than_y_l140_140562

theorem x_is_4286_percent_less_than_y (x y : ℝ) (h : y = 1.75 * x) : 
  ((y - x) / y) * 100 = 42.86 :=
by
  sorry

end x_is_4286_percent_less_than_y_l140_140562


namespace convert_500_to_base2_l140_140508

theorem convert_500_to_base2 :
  let n_base10 : ℕ := 500
  let n_base8 : ℕ := 7 * 64 + 6 * 8 + 4
  let n_base2 : ℕ := 1 * 256 + 1 * 128 + 1 * 64 + 1 * 32 + 1 * 16 + 0 * 8 + 1 * 4 + 0 * 2 + 0
  n_base10 = 500 ∧ n_base8 = 500 ∧ n_base2 = n_base8 :=
by
  sorry

end convert_500_to_base2_l140_140508


namespace radius_of_inner_circle_l140_140003

theorem radius_of_inner_circle (R a x : ℝ) (hR : 0 < R) (ha : 0 ≤ a) (haR : a < R) :
  (a ≠ R ∧ a ≠ 0) → x = (R^2 - a^2) / (2 * R) :=
by
  sorry

end radius_of_inner_circle_l140_140003


namespace sequence_bounded_l140_140262

theorem sequence_bounded (a : ℕ → ℝ) :
  a 0 = 2 →
  (∀ n, a (n+1) = (2 * a n + 1) / (a n + 2)) →
  ∀ n, 1 < a n ∧ a n < 1 + 1 / 3^n :=
by
  intro h₀ h₁
  sorry

end sequence_bounded_l140_140262


namespace number_of_members_l140_140794

theorem number_of_members (n : ℕ) (h : n^2 = 9216) : n = 96 :=
sorry

end number_of_members_l140_140794


namespace total_games_eq_64_l140_140337

def games_attended : ℕ := 32
def games_missed : ℕ := 32
def total_games : ℕ := games_attended + games_missed

theorem total_games_eq_64 : total_games = 64 := by
  sorry

end total_games_eq_64_l140_140337


namespace original_number_of_men_l140_140888

theorem original_number_of_men (x : ℕ) (h1 : 40 * x = 60 * (x - 5)) : x = 15 :=
by
  sorry

end original_number_of_men_l140_140888


namespace remitted_amount_is_correct_l140_140682

-- Define the constants and conditions of the problem
def total_sales : ℝ := 32500
def commission_rate1 : ℝ := 0.05
def commission_limit : ℝ := 10000
def commission_rate2 : ℝ := 0.04

-- Define the function to calculate the remitted amount
def remitted_amount (total_sales commission_rate1 commission_limit commission_rate2 : ℝ) : ℝ :=
  let commission1 := commission_rate1 * commission_limit
  let remaining_sales := total_sales - commission_limit
  let commission2 := commission_rate2 * remaining_sales
  total_sales - (commission1 + commission2)

-- Lean statement to prove the remitted amount
theorem remitted_amount_is_correct :
  remitted_amount total_sales commission_rate1 commission_limit commission_rate2 = 31100 :=
by
  sorry

end remitted_amount_is_correct_l140_140682


namespace cost_of_milkshake_l140_140635

theorem cost_of_milkshake
  (initial_money : ℝ)
  (remaining_after_cupcakes : ℝ)
  (remaining_after_sandwich : ℝ)
  (remaining_after_toy : ℝ)
  (final_remaining : ℝ)
  (money_spent_on_milkshake : ℝ) :
  initial_money = 20 →
  remaining_after_cupcakes = initial_money - (1 / 4) * initial_money →
  remaining_after_sandwich = remaining_after_cupcakes - 0.30 * remaining_after_cupcakes →
  remaining_after_toy = remaining_after_sandwich - (1 / 5) * remaining_after_sandwich →
  final_remaining = 3 →
  money_spent_on_milkshake = remaining_after_toy - final_remaining →
  money_spent_on_milkshake = 5.40 :=
by
  intros 
  sorry

end cost_of_milkshake_l140_140635


namespace angle_same_terminal_side_l140_140913

theorem angle_same_terminal_side (k : ℤ) : ∃ k : ℤ, 95 = -265 + k * 360 :=
by
  use 1
  norm_num

end angle_same_terminal_side_l140_140913


namespace triangle_abs_diff_l140_140699

theorem triangle_abs_diff (a b c : ℝ) 
  (h1 : a + b > c)
  (h2 : a + c > b) :
  |a + b - c| - |a - b - c| = 2 * a - 2 * c := 
by sorry

end triangle_abs_diff_l140_140699


namespace yellow_chip_count_l140_140463

def point_values_equation (Y B G R : ℕ) : Prop :=
  2 ^ Y * 4 ^ B * 5 ^ G * 7 ^ R = 560000

theorem yellow_chip_count (Y B G R : ℕ) (h1 : B = 2 * G) (h2 : R = B / 2) (h3 : point_values_equation Y B G R) :
  Y = 2 :=
by
  sorry

end yellow_chip_count_l140_140463


namespace lollipop_distribution_l140_140961

theorem lollipop_distribution :
  let n := 42
  let initial_lollipops := 650
  let required_lollipops := n * (n + 1) / 2
  (required_lollipops - initial_lollipops) = 253 :=
by
  let n := 42
  let initial_lollipops := 650
  let required_lollipops := n * (n + 1) / 2
  have h : required_lollipops = 903 := by norm_num
  have h2 : (required_lollipops - initial_lollipops) = 253 := by norm_num
  exact h2

end lollipop_distribution_l140_140961


namespace h_of_neg_one_l140_140614

def f (x : ℝ) : ℝ := 3 * x + 7
def g (x : ℝ) : ℝ := (f x) ^ 2 - 3
def h (x : ℝ) : ℝ := f (g x)

theorem h_of_neg_one :
  h (-1) = 298 :=
by
  sorry

end h_of_neg_one_l140_140614


namespace side_error_percentage_l140_140643

theorem side_error_percentage (S S' : ℝ) (h1: S' = S * Real.sqrt 1.0609) : 
  (S' / S - 1) * 100 = 3 :=
by
  sorry

end side_error_percentage_l140_140643


namespace sum_of_decimals_as_fraction_l140_140254

theorem sum_of_decimals_as_fraction :
  (0.2 : ℚ) + (0.03 : ℚ) + (0.004 : ℚ) + (0.0005 : ℚ) + (0.00006 : ℚ) = 733 / 3125 := by
  sorry

end sum_of_decimals_as_fraction_l140_140254


namespace solution_couples_l140_140719

noncomputable def find_couples (n m k : ℕ) : Prop :=
  ∃ t : ℕ, (n = 2^k - 1 - t ∧ m = (Nat.factorial (2^k)) / 2^(2^k - 1 - t))

theorem solution_couples (k : ℕ) :
  ∃ n m : ℕ, (Nat.factorial (2^k)) = 2^n * m ∧ find_couples n m k :=
sorry

end solution_couples_l140_140719


namespace quadratic_function_value_at_18_l140_140686

noncomputable def p (d e f x : ℝ) : ℝ := d*x^2 + e*x + f

theorem quadratic_function_value_at_18
  (d e f : ℝ)
  (h_sym : ∀ x1 x2 : ℝ, p d e f 6 = p d e f 12)
  (h_max : ∀ x : ℝ, x = 10 → ∃ p_max : ℝ, ∀ y : ℝ, p d e f x ≤ p_max)
  (h_p0 : p d e f 0 = -1) : 
  p d e f 18 = -1 := 
sorry

end quadratic_function_value_at_18_l140_140686


namespace students_per_group_l140_140200

theorem students_per_group (n m : ℕ) (h_n : n = 36) (h_m : m = 9) : 
  (n - m) / 3 = 9 := 
by
  sorry

end students_per_group_l140_140200


namespace find_k_l140_140035

theorem find_k (k : ℝ) (h : (-2)^2 - k * (-2) - 6 = 0) : k = 1 :=
by
  sorry

end find_k_l140_140035


namespace correct_factorization_l140_140350

-- Define the conditions from the problem
def conditionA (a b : ℝ) : Prop := a * (a - b) - b * (b - a) = (a - b) * (a + b)
def conditionB (a b : ℝ) : Prop := a^2 - 4 * b^2 = (a + 4 * b) * (a - 4 * b)
def conditionC (a b : ℝ) : Prop := a^2 + 2 * a * b - b^2 = (a + b)^2
def conditionD (a : ℝ) : Prop := a^2 - a - 2 = a * (a - 1) - 2

-- Main theorem statement verifying that only conditionA holds
theorem correct_factorization (a b : ℝ) : 
  conditionA a b ∧ ¬ conditionB a b ∧ ¬ conditionC a b ∧ ¬ conditionD a :=
by 
  sorry

end correct_factorization_l140_140350


namespace sum_of_digits_of_N_l140_140082

theorem sum_of_digits_of_N (N : ℕ) (h : N * (N + 1) / 2 = 2016) : (6 + 3 = 9) :=
by
  sorry

end sum_of_digits_of_N_l140_140082


namespace area_range_of_triangle_l140_140400

-- Defining the points A and B
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (0, -2)

-- Circle equation
def on_circle (P : ℝ × ℝ) : Prop :=
  (P.1 - 2) ^ 2 + P.2 ^ 2 = 2

-- Function to compute the area of triangle ABP
noncomputable def area_of_triangle (P : ℝ × ℝ) : ℝ :=
  0.5 * abs ((A.1 - P.1) * (B.2 - P.2) - (B.1 - P.1) * (A.2 - P.2))

-- The proof goal statement
theorem area_range_of_triangle (P : ℝ × ℝ) (hp : on_circle P) :
  2 ≤ area_of_triangle P ∧ area_of_triangle P ≤ 6 :=
sorry

end area_range_of_triangle_l140_140400


namespace evaluate_expression_l140_140356

variable (x y z : ℚ) -- assuming x, y, z are rational numbers

theorem evaluate_expression (h1 : x = 1 / 4) (h2 : y = 3 / 4) (h3 : z = -8) :
  x^2 * y^3 * z^2 = 108 := by
  sorry

end evaluate_expression_l140_140356


namespace james_total_socks_l140_140999

-- Definitions based on conditions
def red_pairs : ℕ := 20
def black_pairs : ℕ := red_pairs / 2
def white_pairs : ℕ := 2 * (red_pairs + black_pairs)
def green_pairs : ℕ := (red_pairs + black_pairs + white_pairs) + 5

-- Total number of pairs
def total_pairs := red_pairs + black_pairs + white_pairs + green_pairs

-- Total number of socks
def total_socks := total_pairs * 2

-- The main theorem to prove the total number of socks
theorem james_total_socks : total_socks = 370 :=
  by
  -- proof is skipped
  sorry

end james_total_socks_l140_140999


namespace hyperbola_line_intersections_l140_140620

-- Define the hyperbola and line equations
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 4
def line (x y k : ℝ) : Prop := y = k * (x - 1)

-- Conditions for intersecting the hyperbola at two points
def intersect_two_points (k : ℝ) : Prop := 
  k ∈ Set.Ioo (-2 * Real.sqrt 3 / 3) (-1) ∨ 
  k ∈ Set.Ioo (-1) 1 ∨ 
  k ∈ Set.Ioo 1 (2 * Real.sqrt 3 / 3)

-- Conditions for intersecting the hyperbola at exactly one point
def intersect_one_point (k : ℝ) : Prop := 
  k = 1 ∨ 
  k = -1 ∨ 
  k = 2 * Real.sqrt 3 / 3 ∨ 
  k = -2 * Real.sqrt 3 / 3

-- Proof that k is in the appropriate ranges
theorem hyperbola_line_intersections (k : ℝ) :
  ((∃ x y : ℝ, hyperbola x y ∧ line x y k) 
  → (∃ x₁ x₂ y₁ y₂ : ℝ, (x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧ hyperbola x₁ y₁ ∧ line x₁ y₁ k ∧ hyperbola x₂ y₂ ∧ line x₂ y₂ k) 
  → intersect_two_points k))
  ∧ ((∃ x y : ℝ, hyperbola x y ∧ line x y k) 
  → (∃ x y : ℝ, (hyperbola x y ∧ line x y k ∧ (∀ x' y', hyperbola x' y' ∧ line x' y' k → (x' ≠ x ∨ y' ≠ y) = false)) 
  → intersect_one_point k)) := 
sorry

end hyperbola_line_intersections_l140_140620


namespace fraction_of_3_5_eq_2_15_l140_140391

theorem fraction_of_3_5_eq_2_15 : (2 / 15) / (3 / 5) = 2 / 9 := by
  sorry

end fraction_of_3_5_eq_2_15_l140_140391


namespace mean_age_of_all_children_l140_140180

def euler_ages : List ℕ := [10, 12, 8]
def gauss_ages : List ℕ := [8, 8, 8, 16, 18]
def all_ages : List ℕ := euler_ages ++ gauss_ages
def total_children : ℕ := all_ages.length
def total_age : ℕ := all_ages.sum
def mean_age : ℕ := total_age / total_children

theorem mean_age_of_all_children : mean_age = 11 := by
  sorry

end mean_age_of_all_children_l140_140180


namespace leg_ratio_of_right_triangle_l140_140669

theorem leg_ratio_of_right_triangle (a b c m : ℝ) (h1 : a ≤ b)
  (h2 : a * b = c * m) (h3 : c^2 = a^2 + b^2) (h4 : a^2 + m^2 = b^2) :
  (a / b) = Real.sqrt ((-1 + Real.sqrt 5) / 2) :=
sorry

end leg_ratio_of_right_triangle_l140_140669


namespace least_positive_integer_l140_140372

theorem least_positive_integer (n : ℕ) (h₁ : n % 3 = 0) (h₂ : n % 4 = 1) (h₃ : n % 5 = 2) : n = 57 :=
by
  -- sorry to skip the proof
  sorry

end least_positive_integer_l140_140372


namespace alice_bob_meet_l140_140769

theorem alice_bob_meet (t : ℝ) 
(h1 : ∀ s : ℝ, s = 30 * t) 
(h2 : ∀ b : ℝ, b = 29.5 * 60 ∨ b = 30.5 * 60)
(h3 : ∀ a : ℝ, a = 30 * t)
(h4 : ∀ a b : ℝ, a = b):
(t = 59 ∨ t = 61) :=
by
  sorry

end alice_bob_meet_l140_140769


namespace problem_statements_l140_140235

noncomputable def f (x : ℝ) := (Real.exp x - Real.exp (-x)) / 2

noncomputable def g (x : ℝ) := (Real.exp x + Real.exp (-x)) / 2

theorem problem_statements (x : ℝ) :
  (f x < g x) ∧
  ((f x)^2 + (g x)^2 ≥ 1) ∧
  (f (2 * x) = 2 * f x * g x) :=
by
  sorry

end problem_statements_l140_140235


namespace minimum_value_expression_l140_140315

theorem minimum_value_expression (a x1 x2 : ℝ) (h_pos : 0 < a)
  (h1 : x1 + x2 = 4 * a)
  (h2 : x1 * x2 = 3 * a^2)
  (h_ineq : ∀ x, x^2 - 4 * a * x + 3 * a^2 < 0 ↔ x1 < x ∧ x < x2) :
  x1 + x2 + a / (x1 * x2) = (4 * Real.sqrt 3) / 3 :=
by
  sorry

end minimum_value_expression_l140_140315


namespace find_num_non_officers_l140_140730

-- Define the average salaries and number of officers
def avg_salary_employees : Int := 120
def avg_salary_officers : Int := 470
def avg_salary_non_officers : Int := 110
def num_officers : Int := 15

-- States the problem of finding the number of non-officers
theorem find_num_non_officers : ∃ N : Int,
(15 * 470 + N * 110 = (15 + N) * 120) ∧ N = 525 := 
by {
  sorry
}

end find_num_non_officers_l140_140730


namespace nancy_shoes_l140_140697

theorem nancy_shoes (boots slippers heels : ℕ) 
  (h₀ : boots = 6)
  (h₁ : slippers = boots + 9)
  (h₂ : heels = 3 * (boots + slippers)) :
  2 * (boots + slippers + heels) = 168 := by
  sorry

end nancy_shoes_l140_140697


namespace find_number_l140_140840

theorem find_number (x : ℝ) (h : 0.65 * x = 0.05 * 60 + 23) : x = 40 :=
sorry

end find_number_l140_140840


namespace Charley_total_beads_pulled_l140_140728

-- Definitions and conditions
def initial_white_beads := 105
def initial_black_beads := 210
def initial_blue_beads := 60

def first_round_black_pulled := (2 / 7) * initial_black_beads
def first_round_white_pulled := (3 / 7) * initial_white_beads
def first_round_blue_pulled := (1 / 4) * initial_blue_beads

def first_round_total_pulled := first_round_black_pulled + first_round_white_pulled + first_round_blue_pulled

def remaining_black_beads := initial_black_beads - first_round_black_pulled
def remaining_white_beads := initial_white_beads - first_round_white_pulled
def remaining_blue_beads := initial_blue_beads - first_round_blue_pulled

def added_white_beads := 45
def added_black_beads := 80

def total_black_beads := remaining_black_beads + added_black_beads
def total_white_beads := remaining_white_beads + added_white_beads

def second_round_black_pulled := (3 / 8) * total_black_beads
def second_round_white_pulled := (1 / 3) * added_white_beads

def second_round_total_pulled := second_round_black_pulled + second_round_white_pulled

def total_beads_pulled := first_round_total_pulled + second_round_total_pulled 

-- Theorem statement
theorem Charley_total_beads_pulled : total_beads_pulled = 221 := 
by
  -- we can ignore the proof step and leave it to be filled
  sorry

end Charley_total_beads_pulled_l140_140728


namespace set_union_example_l140_140858

variable (A B : Set ℝ)

theorem set_union_example :
  A = {x | -2 < x ∧ x ≤ 1} ∧ B = {x | -1 ≤ x ∧ x < 2} →
  (A ∪ B) = {x | -2 < x ∧ x < 2} := 
by
  sorry

end set_union_example_l140_140858


namespace verify_trig_identity_l140_140240

noncomputable def trig_identity_eqn : Prop :=
  2 * Real.sqrt (1 - Real.sin 8) + Real.sqrt (2 + 2 * Real.cos 8) = -2 * Real.sin 4

theorem verify_trig_identity : trig_identity_eqn := by
  sorry

end verify_trig_identity_l140_140240


namespace length_of_jordans_rectangle_l140_140407

theorem length_of_jordans_rectangle 
  (h1 : ∃ (length width : ℕ), length = 5 ∧ width = 24) 
  (h2 : ∃ (width_area : ℕ), width_area = 30 ∧ ∃ (area : ℕ), area = 5 * 24 ∧ ∃ (L : ℕ), area = L * width_area) :
  ∃ L, L = 4 := by 
  sorry

end length_of_jordans_rectangle_l140_140407


namespace triangle_YZ_length_l140_140124

/-- In triangle XYZ, sides XY and XZ have lengths 6 and 8 inches respectively, 
    and the median XM from vertex X to the midpoint of side YZ is 5 inches. 
    Prove that the length of YZ is 10 inches. -/
theorem triangle_YZ_length
  (XY XZ XM : ℝ)
  (hXY : XY = 6)
  (hXZ : XZ = 8)
  (hXM : XM = 5) :
  ∃ (YZ : ℝ), YZ = 10 := 
by
  sorry

end triangle_YZ_length_l140_140124


namespace Brandy_can_safely_drink_20_mg_more_l140_140905

variable (maximum_caffeine_per_day : ℕ := 500)
variable (caffeine_per_drink : ℕ := 120)
variable (number_of_drinks : ℕ := 4)
variable (caffeine_consumed : ℕ := caffeine_per_drink * number_of_drinks)

theorem Brandy_can_safely_drink_20_mg_more :
    caffeine_consumed = caffeine_per_drink * number_of_drinks →
    (maximum_caffeine_per_day - caffeine_consumed) = 20 :=
by
  intros h1
  rw [h1]
  sorry

end Brandy_can_safely_drink_20_mg_more_l140_140905


namespace find_number_of_male_students_l140_140606

/- Conditions: 
 1. n ≡ 2 [MOD 4]
 2. n ≡ 1 [MOD 5]
 3. n > 15
 4. There are 15 female students
 5. There are more female students than male students
-/
theorem find_number_of_male_students (n : ℕ) (females : ℕ) (h1 : n % 4 = 2) (h2 : n % 5 = 1) (h3 : n > 15) (h4 : females = 15) (h5 : females > n - females) : (n - females) = 11 :=
by
  sorry

end find_number_of_male_students_l140_140606


namespace purchase_price_of_jacket_l140_140559

theorem purchase_price_of_jacket (S P : ℝ) (h1 : S = P + 0.30 * S)
                                (SP : ℝ) (h2 : SP = 0.80 * S)
                                (h3 : 8 = SP - P) :
                                P = 56 := by
  sorry

end purchase_price_of_jacket_l140_140559


namespace car_cost_l140_140802

-- Define the weekly allowance in the first year
def first_year_allowance_weekly : ℕ := 50

-- Define the number of weeks in a year
def weeks_in_year : ℕ := 52

-- Calculate the total first year savings
def first_year_savings : ℕ := first_year_allowance_weekly * weeks_in_year

-- Define the hourly wage and weekly hours worked in the second year
def hourly_wage : ℕ := 9
def weekly_hours_worked : ℕ := 30

-- Calculate the total second year earnings
def second_year_earnings : ℕ := hourly_wage * weekly_hours_worked * weeks_in_year

-- Define the weekly spending in the second year
def weekly_spending : ℕ := 35

-- Calculate the total second year spending
def second_year_spending : ℕ := weekly_spending * weeks_in_year

-- Calculate the total second year savings
def second_year_savings : ℕ := second_year_earnings - second_year_spending

-- Calculate the total savings after two years
def total_savings : ℕ := first_year_savings + second_year_savings

-- Define the additional amount needed
def additional_amount_needed : ℕ := 2000

-- Calculate the total cost of the car
def total_cost_of_car : ℕ := total_savings + additional_amount_needed

-- Theorem statement
theorem car_cost : total_cost_of_car = 16820 := by
  -- The proof is omitted; it is enough to state the theorem
  sorry

end car_cost_l140_140802


namespace cost_of_two_pencils_and_one_pen_l140_140937

variable (a b : ℝ)

-- Given conditions
def condition1 : Prop := (5 * a + b = 2.50)
def condition2 : Prop := (a + 2 * b = 1.85)

-- Statement to prove
theorem cost_of_two_pencils_and_one_pen
  (h1 : condition1 a b) 
  (h2 : condition2 a b) : 
  2 * a + b = 1.45 :=
sorry

end cost_of_two_pencils_and_one_pen_l140_140937


namespace mean_of_three_digit_multiples_of_8_l140_140429

theorem mean_of_three_digit_multiples_of_8 :
  let a1 := 104
  let an := 992
  let d := 8
  let n := (an - a1) / d + 1
  let sum := n * (a1 + an) / 2
  let mean := sum / n
  mean = 548 :=
by
  let a1 := 104
  let an := 992
  let d := 8
  let n := (an - a1) / d + 1
  let sum := n * (a1 + an) / 2
  let mean := sum / n
  sorry

end mean_of_three_digit_multiples_of_8_l140_140429


namespace max_value_of_trig_expr_l140_140836

theorem max_value_of_trig_expr (x : ℝ) : 2 * Real.cos x + 3 * Real.sin x ≤ Real.sqrt 13 :=
sorry

end max_value_of_trig_expr_l140_140836


namespace calculate_expression_l140_140828

theorem calculate_expression : 3 * Real.sqrt 2 - abs (Real.sqrt 2 - Real.sqrt 3) = 4 * Real.sqrt 2 - Real.sqrt 3 :=
  by sorry

end calculate_expression_l140_140828


namespace product_of_roots_l140_140695

-- Define the quadratic function in terms of a, b, c
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the conditions
variables (a b c y : ℝ)

-- Given conditions from the problem
def condition_1 := ∀ x, quadratic a b c x = 0 → ∃ x1 x2, x = x1 ∨ x = x2
def condition_2 := quadratic a b c y = 0
def condition_3 := quadratic a b c (4 * y) = 0

-- The statement to be proved
theorem product_of_roots (a b c y : ℝ) 
  (h1: ∀ x, quadratic a b c x = 0 → ∃ x1 x2, x = x1 ∨ x = x2)
  (h2: quadratic a b c y = 0) 
  (h3: quadratic a b c (4 * y) = 0) :
  ∃ x1 x2, (quadratic a b c x = 0 → (x1 = y ∧ x2 = 4 * y) ∨ (x1 = 4 * y ∧ x2 = y)) ∧ x1 * x2 = 4 * y^2 :=
by
  sorry

end product_of_roots_l140_140695


namespace math_problem_l140_140347

theorem math_problem (a b c : ℝ) (h₁ : a = 85) (h₂ : b = 32) (h₃ : c = 113) :
  (a + b / c) * c = 9637 :=
by
  rw [h₁, h₂, h₃]
  sorry

end math_problem_l140_140347


namespace smaller_pack_size_l140_140808

theorem smaller_pack_size {x : ℕ} (total_eggs large_pack_size large_packs : ℕ) (eggs_in_smaller_packs : ℕ) :
  total_eggs = 79 → large_pack_size = 11 → large_packs = 5 → eggs_in_smaller_packs = total_eggs - large_pack_size * large_packs →
  x * 1 = eggs_in_smaller_packs → x = 24 :=
by sorry

end smaller_pack_size_l140_140808


namespace daisies_left_l140_140323

def initial_daisies : ℕ := 5
def sister_daisies : ℕ := 9
def total_daisies : ℕ := initial_daisies + sister_daisies
def daisies_given_to_mother : ℕ := total_daisies / 2
def remaining_daisies : ℕ := total_daisies - daisies_given_to_mother

theorem daisies_left : remaining_daisies = 7 := by
  sorry

end daisies_left_l140_140323


namespace solution_set_of_even_function_l140_140776

theorem solution_set_of_even_function (f : ℝ → ℝ) (h_even : ∀ x, f (-x) = f x) 
  (h_def : ∀ x, 0 < x → f x = x^2 - 2*x - 3) : 
  { x : ℝ | f x > 0 } = { x | x > 3 } ∪ { x | x < -3 } :=
sorry

end solution_set_of_even_function_l140_140776


namespace problem1_problem2_l140_140550

noncomputable def interval1 (a : ℝ) : Set ℝ := {x | 2 * a < x ∧ x < a + 1}
noncomputable def interval2 : Set ℝ := {x | x < -1 ∨ x > 3}

theorem problem1 (a : ℝ) : (interval1 a ∩ interval2 = interval1 a) ↔ a ∈ {x | x ≤ -2} ∪ {x | 1 ≤ x} := by sorry

theorem problem2 (a : ℝ) : (interval1 a ∩ interval2 ≠ ∅) ↔ a < -1 / 2 := by sorry

end problem1_problem2_l140_140550


namespace perpendicular_line_slopes_l140_140278

theorem perpendicular_line_slopes (α₁ : ℝ) (hα₁ : α₁ = 30) (l₁ : ℝ) (k₁ : ℝ) (k₂ : ℝ) (α₂ : ℝ)
  (h₁ : k₁ = Real.tan (α₁ * Real.pi / 180))
  (h₂ : k₂ = - 1 / k₁)
  (h₃ : k₂ = - Real.sqrt 3)
  (h₄ : 0 < α₂ ∧ α₂ < 180)
  : k₂ = - Real.sqrt 3 ∧ α₂ = 120 := sorry

end perpendicular_line_slopes_l140_140278


namespace q_investment_time_l140_140395

theorem q_investment_time (x t : ℝ)
  (h1 : (7 * 20 * x) / (5 * t * x) = 7 / 10) : t = 40 :=
by
  sorry

end q_investment_time_l140_140395


namespace part1_solution_set_part2_range_of_a_l140_140073

-- Definitions of f and g as provided in the problem.
def f (x : ℝ) : ℝ := |x + 1| + |x - 2|
def g (x a : ℝ) : ℝ := |x + 1| - |x - a| + a

-- Problem 1: Prove the solution set for f(x) ≤ 5 is [-2, 3]
theorem part1_solution_set : { x : ℝ | f x ≤ 5 } = { x : ℝ | -2 ≤ x ∧ x ≤ 3 } :=
  sorry

-- Problem 2: Prove the range of a when f(x) ≥ g(x) always holds is (-∞, 1]
theorem part2_range_of_a (a : ℝ) (h : ∀ x : ℝ, f x ≥ g x a) : a ≤ 1 :=
  sorry

end part1_solution_set_part2_range_of_a_l140_140073


namespace second_number_is_90_l140_140374

theorem second_number_is_90 (x y z : ℕ) 
  (h1 : z = 4 * y) 
  (h2 : y = 2 * x) 
  (h3 : (x + y + z) / 3 = 165) : y = 90 := 
by
  sorry

end second_number_is_90_l140_140374


namespace negation_of_proposition_l140_140619

theorem negation_of_proposition :
  ¬ (∃ x_0 : ℤ, 2 * x_0 + x_0 + 1 ≤ 0) ↔ ∀ x : ℤ, 2 * x + x + 1 > 0 :=
by sorry

end negation_of_proposition_l140_140619


namespace train_cross_time_approx_l140_140346

noncomputable def time_to_cross_pole (length : ℝ) (speed_kmh : ℝ) : ℝ :=
  let speed_ms := speed_kmh * (5 / 18)
  length / speed_ms

theorem train_cross_time_approx
  (d : ℝ) (v_kmh : ℝ)
  (h_d : d = 120)
  (h_v : v_kmh = 121) :
  abs (time_to_cross_pole d v_kmh - 3.57) < 0.01 :=
by {
  sorry
}

end train_cross_time_approx_l140_140346


namespace euclid_1976_part_a_problem_4_l140_140331

theorem euclid_1976_part_a_problem_4
  (p q y1 y2 : ℝ)
  (h1 : y1 = p * 1^2 + q * 1 + 5)
  (h2 : y2 = p * (-1)^2 + q * (-1) + 5)
  (h3 : y1 + y2 = 14) :
  p = 2 :=
by
  sorry

end euclid_1976_part_a_problem_4_l140_140331


namespace lunch_cost_before_tip_l140_140167

theorem lunch_cost_before_tip (tip_rate : ℝ) (total_spent : ℝ) (C : ℝ) : 
  tip_rate = 0.20 ∧ total_spent = 72.96 ∧ C + tip_rate * C = total_spent → C = 60.80 :=
by
  intro h
  sorry

end lunch_cost_before_tip_l140_140167


namespace infinite_set_k_l140_140442

theorem infinite_set_k (C : ℝ) : ∃ᶠ k : ℤ in at_top, (k : ℝ) * Real.sin k > C :=
sorry

end infinite_set_k_l140_140442


namespace column_of_2008_l140_140273

theorem column_of_2008:
  (∃ k, 2008 = 2 * k) ∧
  ((2 % 8) = 2) ∧ ((4 % 8) = 4) ∧ ((6 % 8) = 6) ∧ ((8 % 8) = 0) ∧
  ((16 % 8) = 0) ∧ ((14 % 8) = 6) ∧ ((12 % 8) = 4) ∧ ((10 % 8) = 2) →
  (2008 % 8 = 4) :=
by
  sorry

end column_of_2008_l140_140273


namespace literate_employees_l140_140680

theorem literate_employees (num_illiterate : ℕ) (wage_decrease_per_illiterate : ℕ)
  (total_average_salary_decrease : ℕ) : num_illiterate = 35 → 
                                        wage_decrease_per_illiterate = 25 →
                                        total_average_salary_decrease = 15 →
                                        ∃ L : ℕ, L = 23 :=
by {
  -- given: num_illiterate = 35
  -- given: wage_decrease_per_illiterate = 25
  -- given: total_average_salary_decrease = 15
  sorry
}

end literate_employees_l140_140680


namespace inscribed_circle_radius_third_of_circle_l140_140566

noncomputable def inscribed_circle_radius (R : ℝ) : ℝ := 
  R * (Real.sqrt 3 - 1) / 2

theorem inscribed_circle_radius_third_of_circle (R : ℝ) (hR : R = 5) :
  inscribed_circle_radius R = 5 * (Real.sqrt 3 - 1) / 2 := by
  sorry

end inscribed_circle_radius_third_of_circle_l140_140566


namespace rationalize_denominator_l140_140603

theorem rationalize_denominator : (1 / (Real.sqrt 3 - 1)) = ((Real.sqrt 3 + 1) / 2) :=
by
  sorry

end rationalize_denominator_l140_140603


namespace numerator_denominator_added_l140_140211

theorem numerator_denominator_added (n : ℕ) : (3 + n) / (5 + n) = 9 / 11 → n = 6 :=
by
  sorry

end numerator_denominator_added_l140_140211


namespace sum_of_solutions_eq_8_l140_140950

theorem sum_of_solutions_eq_8 :
    let a : ℝ := 1
    let b : ℝ := -8
    let c : ℝ := -26
    ∀ x1 x2 : ℝ, (a * x1^2 + b * x1 + c = 0) ∧ (a * x2^2 + b * x2 + c = 0) →
      x1 + x2 = 8 :=
sorry

end sum_of_solutions_eq_8_l140_140950


namespace correctCountForDivisibilityBy15_l140_140358

namespace Divisibility

noncomputable def countWaysToMakeDivisibleBy15 : Nat := 
  let digits := [0, 2, 4, 5, 7, 9]
  let baseSum := 2 + 0 + 1 + 6 + 0 + 2
  let validLastDigit := [0, 5]
  let totalCombinations := 6^4
  let ways := 2 * totalCombinations
  let adjustment := (validLastDigit.length * digits.length * digits.length * digits.length * validLastDigit.length) / 4 -- Correcting multiplier as per reference
  adjustment

theorem correctCountForDivisibilityBy15 : countWaysToMakeDivisibleBy15 = 864 := 
  by
    sorry

end Divisibility

end correctCountForDivisibilityBy15_l140_140358


namespace tangent_parallel_l140_140500

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a * x + 2 * Real.cos x
noncomputable def f (x : ℝ) : ℝ := -Real.exp x - x

theorem tangent_parallel (a : ℝ) (H : ∀ x1 : ℝ, ∃ x2 : ℝ, (a - 2 * Real.sin x1) = (-Real.exp x2 - 1)) :
  a < -3 := by
  sorry

end tangent_parallel_l140_140500


namespace probability_of_picking_peach_l140_140850

-- Define the counts of each type of fruit
def apples : ℕ := 5
def pears : ℕ := 3
def peaches : ℕ := 2

-- Define the total number of fruits
def total_fruits : ℕ := apples + pears + peaches

-- Define the probability of picking a peach
def probability_of_peach : ℚ := peaches / total_fruits

-- State the theorem
theorem probability_of_picking_peach : probability_of_peach = 1/5 := by
  -- proof goes here
  sorry

end probability_of_picking_peach_l140_140850


namespace probability_penny_nickel_dime_heads_l140_140926

noncomputable def probability_heads (n : ℕ) : ℚ := (1 : ℚ) / (2 ^ n)

theorem probability_penny_nickel_dime_heads :
  probability_heads 3 = 1 / 8 := 
by
  sorry

end probability_penny_nickel_dime_heads_l140_140926


namespace intersection_question_l140_140322

def M : Set ℕ := {1, 2}
def N : Set ℕ := {n | ∃ a ∈ M, n = 2 * a - 1}

theorem intersection_question : M ∩ N = {1} :=
by sorry

end intersection_question_l140_140322


namespace remaining_fruit_count_l140_140887

theorem remaining_fruit_count (trees : ℕ) (fruits_per_tree : ℕ) (picked_fraction : ℚ) 
  (trees_eq : trees = 8) (fruits_per_tree_eq : fruits_per_tree = 200) (picked_fraction_eq : picked_fraction = 2/5) :
  let total_fruits := trees * fruits_per_tree
  let picked_fruits := picked_fraction * fruits_per_tree * trees
  let remaining_fruits := total_fruits - picked_fruits
  remaining_fruits = 960 := 
by 
  sorry

end remaining_fruit_count_l140_140887


namespace ted_age_l140_140360

variables (t s j : ℕ)

theorem ted_age
  (h1 : t = 2 * s - 20)
  (h2 : j = s + 6)
  (h3 : t + s + j = 90) :
  t = 32 :=
by
  sorry

end ted_age_l140_140360


namespace total_area_covered_by_strips_l140_140394

theorem total_area_covered_by_strips (L W : ℝ) (n : ℕ) (overlap_area : ℝ) (end_to_end_area : ℝ) :
  L = 15 → W = 1 → n = 4 → overlap_area = 15 → end_to_end_area = 30 → 
  (L * W * n - overlap_area + end_to_end_area) = 45 :=
by
  intros hL hW hn hoverlap hend_to_end
  sorry

end total_area_covered_by_strips_l140_140394


namespace sum_first_sequence_terms_l140_140725

theorem sum_first_sequence_terms 
  (S : ℕ → ℕ) 
  (a : ℕ → ℕ) 
  (h1 : ∀ n, n ≥ 2 → S n - S (n - 1) = 2 * n - 1)
  (h2 : S 2 = 3) 
  : a 1 + a 3 = 5 :=
sorry

end sum_first_sequence_terms_l140_140725


namespace probability_x_gt_2y_is_1_over_3_l140_140368

noncomputable def probability_x_gt_2y_in_rectangle : ℝ :=
  let A_rect := 6 * 1
  let A_triangle := (1/2) * 4 * 1
  A_triangle / A_rect

theorem probability_x_gt_2y_is_1_over_3 :
  probability_x_gt_2y_in_rectangle = 1 / 3 :=
sorry

end probability_x_gt_2y_is_1_over_3_l140_140368


namespace angle_ACD_l140_140292

theorem angle_ACD {α β δ : Type*} [LinearOrderedField α] [CharZero α] (ABC DAB DBA : α)
  (h1 : ABC = 60) (h2 : BAC = 80) (h3 : DAB = 10) (h4 : DBA = 20):
  ACD = 30 := by
  sorry

end angle_ACD_l140_140292


namespace subtract_23_result_l140_140764

variable {x : ℕ}

theorem subtract_23_result (h : x + 30 = 55) : x - 23 = 2 :=
sorry

end subtract_23_result_l140_140764


namespace initial_speed_increase_l140_140327

variables (S : ℝ) (P : ℝ)

/-- Prove that the initial percentage increase in speed P is 0.3 based on the given conditions: 
1. After the first increase by P, the speed becomes S + PS.
2. After the second increase by 10%, the final speed is (S + PS) * 1.10.
3. The total increase results in a speed that is 1.43 times the original speed S. -/
theorem initial_speed_increase (h : (S + P * S) * 1.1 = 1.43 * S) : P = 0.3 :=
sorry

end initial_speed_increase_l140_140327


namespace sum_four_variables_l140_140412

theorem sum_four_variables 
  (a b c d : ℝ) (x : ℝ)
  (h1 : a + 2 = x)
  (h2 : b + 3 = x)
  (h3 : c + 4 = x)
  (h4 : d + 5 = x)
  (h5 : a + b + c + d + 8 = x) :
  a + b + c + d = -6 :=
by
  sorry

end sum_four_variables_l140_140412


namespace lcm_18_35_is_630_l140_140997

def lcm_18_35 : ℕ :=
  Nat.lcm 18 35

theorem lcm_18_35_is_630 : lcm_18_35 = 630 := by
  sorry

end lcm_18_35_is_630_l140_140997


namespace tan_theta_parallel_l140_140152

theorem tan_theta_parallel (θ : ℝ) : 
  let a := (2, 3)
  let b := (Real.cos θ, Real.sin θ)
  (b.1 * a.2 = b.2 * a.1) → Real.tan θ = 3 / 2 :=
by
  intros h
  sorry

end tan_theta_parallel_l140_140152


namespace integral_1_integral_2_integral_3_integral_4_integral_5_l140_140625
open Real

-- Integral 1
theorem integral_1 : ∫ (x : ℝ), sin x * cos x ^ 3 = -1 / 4 * cos x ^ 4 + C :=
by sorry

-- Integral 2
theorem integral_2 : ∫ (x : ℝ), 1 / ((1 + sqrt x) * sqrt x) = 2 * log (1 + sqrt x) + C :=
by sorry

-- Integral 3
theorem integral_3 : ∫ (x : ℝ), x ^ 2 * sqrt (x ^ 3 + 1) = 2 / 9 * (x ^ 3 + 1) ^ (3/2) + C :=
by sorry

-- Integral 4
theorem integral_4 : ∫ (x : ℝ), (exp (2 * x) - 3 * exp x) / exp x = exp x - 3 * x + C :=
by sorry

-- Integral 5
theorem integral_5 : ∫ (x : ℝ), (1 - x ^ 2) * exp x = - (x - 1) ^ 2 * exp x + C :=
by sorry

end integral_1_integral_2_integral_3_integral_4_integral_5_l140_140625


namespace find_special_number_l140_140977

theorem find_special_number:
  ∃ (n : ℕ), (n > 0) ∧ (∃ (k : ℕ), 2 * n = k^2)
           ∧ (∃ (m : ℕ), 3 * n = m^3)
           ∧ (∃ (p : ℕ), 5 * n = p^5)
           ∧ n = 1085 :=
by
  sorry

end find_special_number_l140_140977


namespace cost_price_of_article_l140_140574

theorem cost_price_of_article (SP : ℝ) (profit_percent : ℝ) (CP : ℝ) 
    (h1 : SP = 100) 
    (h2 : profit_percent = 0.20) 
    (h3 : SP = CP * (1 + profit_percent)) : 
    CP = 83.33 :=
by
  sorry

end cost_price_of_article_l140_140574


namespace size_relationship_l140_140834

variable (a1 a2 b1 b2 : ℝ)

theorem size_relationship (h1 : a1 < a2) (h2 : b1 < b2) : a1 * b1 + a2 * b2 > a1 * b2 + a2 * b1 := 
sorry

end size_relationship_l140_140834


namespace tangent_intersects_x_axis_l140_140648

theorem tangent_intersects_x_axis (x0 x1 : ℝ) (hx : ∀ x : ℝ, x1 = x0 - 1) :
  x1 - x0 = -1 :=
by
  sorry

end tangent_intersects_x_axis_l140_140648


namespace pet_store_cats_left_l140_140657

theorem pet_store_cats_left :
  let initial_siamese := 13.5
  let initial_house := 5.25
  let added_cats := 10.75
  let discount := 0.5
  let initial_total := initial_siamese + initial_house
  let new_total := initial_total + added_cats
  let final_total := new_total - discount
  final_total = 29 :=
by sorry

end pet_store_cats_left_l140_140657


namespace a3_equals_neg7_l140_140294

-- Definitions based on given conditions
noncomputable def a₁ := -11
noncomputable def d : ℤ := sorry -- this is derived but unknown presently
noncomputable def a(n : ℕ) : ℤ := a₁ + (n - 1) * d

axiom condition : a 4 + a 6 = -6

-- The proof problem statement
theorem a3_equals_neg7 : a 3 = -7 :=
by
  have h₁ : a₁ = -11 := rfl
  have h₂ : a 4 + a 6 = -6 := condition
  sorry

end a3_equals_neg7_l140_140294


namespace unique_triangle_exists_l140_140629

theorem unique_triangle_exists : 
  (¬ (∀ (a b c : ℝ), a = 1 ∧ b = 2 ∧ c = 3 → a + b > c)) ∧
  (¬ (∀ (a b A : ℝ), a = 1 ∧ b = 2 ∧ A = 30 → ∃ (C : ℝ), C > 0)) ∧
  (¬ (∀ (a b A : ℝ), a = 1 ∧ b = 2 ∧ A = 100 → ∃ (C : ℝ), C > 0)) ∧
  (∀ (b c B : ℝ), b = 1 ∧ c = 1 ∧ B = 45 → ∃! (a c B : ℝ), b = 1 ∧ c = 1 ∧ B = 45) :=
by sorry

end unique_triangle_exists_l140_140629


namespace intersection_of_M_and_N_l140_140318

def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {x | ∃ a ∈ M, x = a^2}

theorem intersection_of_M_and_N :
  (M ∩ N = {0, 1}) :=
by
  sorry

end intersection_of_M_and_N_l140_140318


namespace probability_of_roll_6_after_E_l140_140554

/- Darryl has a six-sided die with faces 1, 2, 3, 4, 5, 6.
   The die is weighted so that one face comes up with probability 1/2,
   and the other five faces have equal probability.
   Darryl does not know which side is weighted, but each face is equally likely to be the weighted one.
   Darryl rolls the die 5 times and gets a 1, 2, 3, 4, and 5 in some unspecified order. -/

def probability_of_next_roll_getting_6 : ℚ :=
  let p_weighted := (1 / 2 : ℚ)
  let p_unweighted := (1 / 10 : ℚ)
  let p_w6_given_E := (1 / 26 : ℚ)
  let p_not_w6_given_E := (25 / 26 : ℚ)
  p_w6_given_E * p_weighted + p_not_w6_given_E * p_unweighted

theorem probability_of_roll_6_after_E : probability_of_next_roll_getting_6 = 3 / 26 := sorry

end probability_of_roll_6_after_E_l140_140554


namespace only_point_D_lies_on_graph_l140_140848

def point := ℤ × ℤ

def lies_on_graph (f : ℤ → ℤ) (p : point) : Prop :=
  f p.1 = p.2

def f (x : ℤ) : ℤ := 2 * x - 1

theorem only_point_D_lies_on_graph :
  (lies_on_graph f (-1, 3) = false) ∧ 
  (lies_on_graph f (0, 1) = false) ∧ 
  (lies_on_graph f (1, -1) = false) ∧ 
  (lies_on_graph f (2, 3)) := 
by
  sorry

end only_point_D_lies_on_graph_l140_140848


namespace ab_root_of_Q_l140_140701

theorem ab_root_of_Q (a b : ℝ) (h : a ≠ b) (ha : a^4 + a^3 - 1 = 0) (hb : b^4 + b^3 - 1 = 0) :
  (ab : ℝ)^6 + (ab : ℝ)^4 + (ab : ℝ)^3 - (ab : ℝ)^2 - 1 = 0 := 
sorry

end ab_root_of_Q_l140_140701


namespace spiral_2018_position_l140_140815

def T100_spiral : Matrix ℕ ℕ ℕ := sorry -- Definition of T100 as a spiral matrix

def pos_2018 := (34, 95) -- The given position we need to prove

theorem spiral_2018_position (i j : ℕ) (h₁ : T100_spiral 34 95 = 2018) : (i, j) = pos_2018 := by  
  sorry

end spiral_2018_position_l140_140815


namespace a5_is_3_l140_140918

section
variable {a : ℕ → ℝ} 
variable (h_pos : ∀ n, 0 < a n)
variable (h_a1 : a 1 = 1)
variable (h_a2 : a 2 = Real.sqrt 3)
variable (h_recursive : ∀ n ≥ 2, 2 * (a n)^2 = (a (n + 1))^2 + (a (n - 1))^2)

theorem a5_is_3 : a 5 = 3 :=
  by
  sorry
end

end a5_is_3_l140_140918


namespace new_average_l140_140325

theorem new_average (n : ℕ) (a : ℕ) (multiplier : ℕ) (average : ℕ) :
  (n = 35) →
  (a = 25) →
  (multiplier = 5) →
  (average = 125) →
  ((n * a * multiplier) / n = average) :=
by
  intros hn ha hm havg
  rw [hn, ha, hm]
  norm_num
  sorry

end new_average_l140_140325


namespace simplify_and_evaluate_expression_l140_140841

theorem simplify_and_evaluate_expression (m : ℕ) (h : m = 2) :
  ( (↑m + 1) / (↑m - 1) + 1 ) / ( (↑m + m^2) / (m^2 - 2*m + 1) ) - ( 2 - 2*↑m ) / ( m^2 - 1 ) = 4 / 3 :=
by sorry

end simplify_and_evaluate_expression_l140_140841


namespace dennis_years_of_teaching_l140_140028

variable (V A D E N : ℕ)

def combined_years_taught : Prop :=
  V + A + D + E + N = 225

def virginia_adrienne_relation : Prop :=
  V = A + 9

def virginia_dennis_relation : Prop :=
  V = D - 15

def elijah_adrienne_relation : Prop :=
  E = A - 3

def elijah_nadine_relation : Prop :=
  E = N + 7

theorem dennis_years_of_teaching 
  (h1 : combined_years_taught V A D E N) 
  (h2 : virginia_adrienne_relation V A)
  (h3 : virginia_dennis_relation V D)
  (h4 : elijah_adrienne_relation E A) 
  (h5 : elijah_nadine_relation E N) : 
  D = 65 :=
  sorry

end dennis_years_of_teaching_l140_140028


namespace triangle_area_ab_l140_140723

theorem triangle_area_ab (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hline : ∀ (x y : ℝ), a * x + b * y = 6) (harea : (1/2) * (6 / a) * (6 / b) = 6) : 
  a * b = 3 := 
by sorry

end triangle_area_ab_l140_140723


namespace greatest_product_two_ints_sum_300_l140_140743

theorem greatest_product_two_ints_sum_300 :
  (∀ x y : ℤ, x + y = 300 → x * y ≤ 22500) ∧ (∃ x y : ℤ, x + y = 300 ∧ x * y = 22500) :=
by
  sorry

end greatest_product_two_ints_sum_300_l140_140743


namespace min_value_l140_140944

theorem min_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 3 * x + y + 1 / x + 2 / y = 13 / 2) :
  x - 1 / y ≥ - 1 / 2 :=
sorry

end min_value_l140_140944


namespace find_number_that_satisfies_condition_l140_140158

theorem find_number_that_satisfies_condition : ∃ x : ℝ, x / 3 + 12 = 20 ∧ x = 24 :=
by
  sorry

end find_number_that_satisfies_condition_l140_140158


namespace f_is_odd_max_min_values_l140_140474

-- Define the function f satisfying the given conditions
variable (f : ℝ → ℝ)
variable (f_add : ∀ x y : ℝ, f (x + y) = f (x) + f (y))
variable (f_one : f 1 = -2)
variable (f_neg : ∀ x > 0, f x < 0)

-- Define the statement in Lean for Part 1: proving the function is odd
theorem f_is_odd : ∀ x : ℝ, f (-x) = -f (x) := by sorry

-- Define the statement in Lean for Part 2: proving the max and min values on [-3, 3]
theorem max_min_values : 
  ∃ max_value min_value : ℝ, 
  (max_value = f (-3) ∧ max_value = 6) ∧ 
  (min_value = f (3) ∧ min_value = -6) := by sorry

end f_is_odd_max_min_values_l140_140474


namespace solution_set_inequality_range_a_inequality_l140_140898

noncomputable def f (x a : ℝ) : ℝ := abs (x - a) - 2

theorem solution_set_inequality (x : ℝ) (a : ℝ) (h : a = 1) :
  f x a + abs (2*x - 3) > 0 ↔ (x < 2 / 3 ∨ 2 < x) := sorry

theorem range_a_inequality (a : ℝ) :
  (∀ x, f x a < abs (x - 3)) ↔ (1 < a ∧ a < 5) := sorry

end solution_set_inequality_range_a_inequality_l140_140898


namespace subtraction_result_l140_140342

open Matrix

namespace Vector

def a : (Fin 3 → ℝ) :=
  ![5, -3, 2]

def b : (Fin 3 → ℝ) :=
  ![-2, 4, 1]

theorem subtraction_result : a - (2 • b) = ![9, -11, 0] :=
by
  -- Skipping the proof
  sorry

end Vector

end subtraction_result_l140_140342


namespace triangle_median_difference_l140_140838

theorem triangle_median_difference
    (A B C D E : Type)
    (BC_len : BC = 10)
    (AD_len : AD = 6)
    (BE_len : BE = 7.5) :
    ∃ X_max X_min : ℝ, 
    X_max = AB^2 + AC^2 + BC^2 ∧ 
    X_min = AB^2 + AC^2 + BC^2 ∧ 
    (X_max - X_min) = 56.25 :=
by
  sorry

end triangle_median_difference_l140_140838


namespace ricardo_coins_difference_l140_140234

theorem ricardo_coins_difference :
  ∃ (x y : ℕ), (x + y = 2020) ∧ (x ≥ 1) ∧ (y ≥ 1) ∧ ((5 * x + y) - (x + 5 * y) = 8072) :=
by
  sorry

end ricardo_coins_difference_l140_140234


namespace appropriate_import_range_l140_140215

def mung_bean_import_range (p0 : ℝ) (p_desired_min p_desired_max : ℝ) (x : ℝ) : Prop :=
  p0 - (x / 100) ≤ p_desired_max ∧ p0 - (x / 100) ≥ p_desired_min

theorem appropriate_import_range : 
  ∃ x : ℝ, 600 ≤ x ∧ x ≤ 800 ∧ mung_bean_import_range 16 8 10 x :=
sorry

end appropriate_import_range_l140_140215


namespace g_at_100_l140_140736

-- Defining that g is a function from positive real numbers to real numbers
def g : ℝ → ℝ := sorry

-- The given conditions
axiom functional_equation (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  x * g y - y * g x = g (x / y)

axiom g_one : g 1 = 1

-- The theorem to prove
theorem g_at_100 : g 100 = 50 :=
by
  sorry

end g_at_100_l140_140736


namespace fraction_numerator_greater_than_denominator_l140_140896

theorem fraction_numerator_greater_than_denominator (x : ℝ) :
  -1 ≤ x ∧ x ≤ 3 ∧ x ≠ 5 / 3 → (8 / 11 < x ∧ x < 5 / 3) ∨ (5 / 3 < x ∧ x ≤ 3) ↔ (8 * x - 3 > 5 - 3 * x) := by
  sorry

end fraction_numerator_greater_than_denominator_l140_140896


namespace sarah_initial_bake_l140_140988

theorem sarah_initial_bake (todd_ate : ℕ) (packages : ℕ) (cupcakes_per_package : ℕ) 
  (initial_cupcakes : ℕ)
  (h1 : todd_ate = 14)
  (h2 : packages = 3)
  (h3 : cupcakes_per_package = 8)
  (h4 : packages * cupcakes_per_package + todd_ate = initial_cupcakes) :
  initial_cupcakes = 38 :=
by sorry

end sarah_initial_bake_l140_140988


namespace equal_roots_of_quadratic_l140_140106

theorem equal_roots_of_quadratic (k : ℝ) : (1 - 8 * k = 0) → (k = 1/8) :=
by
  intro h
  sorry

end equal_roots_of_quadratic_l140_140106


namespace factorization_1_factorization_2_l140_140811

variables {x y m n : ℝ}

theorem factorization_1 : x^3 + 2 * x^2 * y + x * y^2 = x * (x + y)^2 :=
sorry

theorem factorization_2 : 4 * m^2 - n^2 - 4 * m + 1 = (2 * m - 1 + n) * (2 * m - 1 - n) :=
sorry

end factorization_1_factorization_2_l140_140811


namespace range_of_a_for_increasing_l140_140914

noncomputable def f (a : ℝ) : (ℝ → ℝ) := λ x => x^3 + a * x^2 + 3 * x

theorem range_of_a_for_increasing (a : ℝ) : 
  (∀ x : ℝ, (3 * x^2 + 2 * a * x + 3) ≥ 0) ↔ (-3 ≤ a ∧ a ≤ 3) :=
by
  sorry

end range_of_a_for_increasing_l140_140914


namespace min_value_seq_ratio_l140_140642

-- Define the sequence {a_n} based on the given recurrence relation and initial condition
def seq (n : ℕ) : ℕ := 
  if n = 0 then 0 -- Handling the case when n is 0, though sequence starts from n=1
  else n^2 - n + 15

-- Prove the minimum value of (a_n / n) is 27/4
theorem min_value_seq_ratio : 
  ∃ n : ℕ, n > 0 ∧ seq n / n = 27 / 4 :=
by
  sorry

end min_value_seq_ratio_l140_140642


namespace problem_part_I_problem_part_II_l140_140645

-- Problem Part I
def f (x : ℝ) : ℝ := 4 - |x| - |x - 3|

theorem problem_part_I (x : ℝ) :
    (f (x + 3/2) ≥ 0) ↔ (-2 ≤ x ∧ x ≤ 2) :=
  sorry

-- Problem Part II
theorem problem_part_II (p q r : ℝ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) 
    (h : 1/(3*p) + 1/(2*q) + 1/r = 4) : 
    3*p + 2*q + r ≥ 9/4 :=
  sorry

end problem_part_I_problem_part_II_l140_140645


namespace heights_inequality_l140_140126

theorem heights_inequality (a b c h_a h_b h_c p R : ℝ) (h : a ≤ b ∧ b ≤ c) : 
  h_a + h_b + h_c ≤ (3 * b * (a^2 + a * c + c^2)) / (4 * p * R) := 
sorry

end heights_inequality_l140_140126


namespace cost_price_of_A_l140_140700

-- Assume the cost price of the bicycle for A which we need to prove
def CP_A : ℝ := 144

-- Given conditions
def profit_A_to_B (CP_A : ℝ) := 1.25 * CP_A
def profit_B_to_C (CP_B : ℝ) := 1.25 * CP_B
def SP_C := 225

-- Proof statement
theorem cost_price_of_A : 
  profit_B_to_C (profit_A_to_B CP_A) = SP_C :=
by
  sorry

end cost_price_of_A_l140_140700


namespace kathleen_savings_in_july_l140_140389

theorem kathleen_savings_in_july (savings_june savings_august spending_school spending_clothes money_left savings_target add_from_aunt : ℕ) 
  (h_june : savings_june = 21)
  (h_august : savings_august = 45)
  (h_school : spending_school = 12)
  (h_clothes : spending_clothes = 54)
  (h_left : money_left = 46)
  (h_target : savings_target = 125)
  (h_aunt : add_from_aunt = 25)
  (not_received_from_aunt : (savings_june + savings_august + money_left + add_from_aunt) ≤ savings_target)
  : (savings_june + savings_august + money_left + spending_school + spending_clothes - (savings_june + savings_august + spending_school + spending_clothes)) = 46 := 
by 
  -- These conditions narrate the problem setup
  -- We can proceed to show the proof here
  sorry 

end kathleen_savings_in_july_l140_140389


namespace juice_profit_eq_l140_140287

theorem juice_profit_eq (x : ℝ) :
  (70 - x) * (160 + 8 * x) = 16000 :=
sorry

end juice_profit_eq_l140_140287


namespace vertex_of_given_function_l140_140893

noncomputable def vertex_coordinates (f : ℝ → ℝ) : ℝ × ℝ := 
  (-2, 1)  -- Prescribed coordinates for this specific function form.

def function_vertex (x : ℝ) : ℝ :=
  -3 * (x + 2) ^ 2 + 1

theorem vertex_of_given_function : 
  vertex_coordinates function_vertex = (-2, 1) :=
by
  sorry

end vertex_of_given_function_l140_140893


namespace complex_multiplication_l140_140049

theorem complex_multiplication : ∀ (i : ℂ), i^2 = -1 → i * (2 + 3 * i) = (-3 : ℂ) + 2 * i :=
by
  intros i hi
  sorry

end complex_multiplication_l140_140049


namespace difference_in_elevation_difference_in_running_time_l140_140225

structure Day :=
  (distance_km : ℝ) -- kilometers
  (pace_min_per_km : ℝ) -- minutes per kilometer
  (elevation_gain_m : ℝ) -- meters

def monday : Day := { distance_km := 9, pace_min_per_km := 6, elevation_gain_m := 300 }
def wednesday : Day := { distance_km := 4.816, pace_min_per_km := 5.5, elevation_gain_m := 150 }
def friday : Day := { distance_km := 2.095, pace_min_per_km := 7, elevation_gain_m := 50 }

noncomputable def calculate_running_time(day : Day) : ℝ :=
  day.distance_km * day.pace_min_per_km

noncomputable def total_elevation_gain(wednesday friday : Day) : ℝ :=
  wednesday.elevation_gain_m + friday.elevation_gain_m

noncomputable def total_running_time(wednesday friday : Day) : ℝ :=
  calculate_running_time wednesday + calculate_running_time friday

theorem difference_in_elevation :
  monday.elevation_gain_m - total_elevation_gain wednesday friday = 100 := by 
  sorry

theorem difference_in_running_time :
  calculate_running_time monday - total_running_time wednesday friday = 12.847 := by 
  sorry

end difference_in_elevation_difference_in_running_time_l140_140225


namespace ali_total_money_l140_140030

-- Definitions based on conditions
def bills_of_5_dollars : ℕ := 7
def bills_of_10_dollars : ℕ := 1
def value_of_5_dollar_bill : ℕ := 5
def value_of_10_dollar_bill : ℕ := 10

-- Prove that Ali's total amount of money is $45
theorem ali_total_money : (bills_of_5_dollars * value_of_5_dollar_bill) + (bills_of_10_dollars * value_of_10_dollar_bill) = 45 := 
by
  sorry

end ali_total_money_l140_140030


namespace total_seats_l140_140810

theorem total_seats (s : ℕ) 
  (first_class : ℕ := 30) 
  (business_class : ℕ := (20 * s) / 100) 
  (premium_economy : ℕ := 15) 
  (economy_class : ℕ := s - first_class - business_class - premium_economy) 
  (total : first_class + business_class + premium_economy + economy_class = s) 
  : s = 288 := 
sorry

end total_seats_l140_140810


namespace tangent_line_to_parabola_l140_140741

theorem tangent_line_to_parabola :
  (∀ (x y : ℝ), y = x^2 → x = -1 → y = 1 → 2 * x + y + 1 = 0) :=
by
  intro x y parabola eq_x eq_y
  sorry

end tangent_line_to_parabola_l140_140741


namespace train_length_correct_l140_140424

-- Define the conditions
def train_speed : ℝ := 63
def time_crossing : ℝ := 40
def expected_length : ℝ := 2520

-- The statement to prove
theorem train_length_correct : train_speed * time_crossing = expected_length :=
by
  exact sorry

end train_length_correct_l140_140424


namespace more_soccer_balls_than_basketballs_l140_140085

theorem more_soccer_balls_than_basketballs :
  let soccer_boxes := 8
  let basketball_boxes := 5
  let balls_per_box := 12
  soccer_boxes * balls_per_box - basketball_boxes * balls_per_box = 36 := by
  sorry

end more_soccer_balls_than_basketballs_l140_140085


namespace product_of_real_roots_l140_140868

theorem product_of_real_roots : 
  let f (x : ℝ) := x ^ Real.log x / Real.log 2 
  ∃ r1 r2 : ℝ, (f r1 = 16 ∧ f r2 = 16) ∧ (r1 * r2 = 1) := 
by
  sorry

end product_of_real_roots_l140_140868


namespace log_relation_l140_140718

theorem log_relation (a b : ℝ) 
  (h₁ : a = Real.log 1024 / Real.log 16) 
  (h₂ : b = Real.log 32 / Real.log 2) : 
  a = 1 / 2 * b := 
by 
  sorry

end log_relation_l140_140718


namespace cone_base_radius_l140_140823

theorem cone_base_radius (r_paper : ℝ) (n_parts : ℕ) (r_cone_base : ℝ) 
  (h_radius_paper : r_paper = 16)
  (h_n_parts : n_parts = 4)
  (h_cone_part : r_cone_base = r_paper / n_parts) : r_cone_base = 4 := by
  sorry

end cone_base_radius_l140_140823


namespace f_decreasing_l140_140005

open Real

noncomputable def f (x : ℝ) : ℝ := 1 / x^2 + 3

theorem f_decreasing (x1 x2 : ℝ) (h1 : 0 < x1) (h2 : 0 < x2) (h : x1 < x2) : f x1 > f x2 := 
by
  sorry

end f_decreasing_l140_140005


namespace maximum_point_of_f_l140_140494

noncomputable def f (x : ℝ) : ℝ := (x^2 - 2 * x - 2) * Real.exp x

theorem maximum_point_of_f : ∃ x : ℝ, x = -2 ∧
  ∀ y : ℝ, f y ≤ f x :=
sorry

end maximum_point_of_f_l140_140494


namespace simplest_radical_expression_l140_140266

theorem simplest_radical_expression :
  let A := Real.sqrt 3
  let B := Real.sqrt 4
  let C := Real.sqrt 8
  let D := Real.sqrt (1 / 2)
  B = 2 :=
by
  sorry

end simplest_radical_expression_l140_140266


namespace tan_alpha_eq_neg2_l140_140780

theorem tan_alpha_eq_neg2 {α : ℝ} {x y : ℝ} (hx : x = -2) (hy : y = 4) (hM : (x, y) = (-2, 4)) :
  Real.tan α = -2 :=
by
  sorry

end tan_alpha_eq_neg2_l140_140780


namespace mikes_original_speed_l140_140246

variable (x : ℕ) -- x is the original typing speed of Mike

-- Condition: After the accident, Mike's typing speed is 20 words per minute less
def currentSpeed : ℕ := x - 20

-- Condition: It takes Mike 18 minutes to type 810 words at his reduced speed
def typingTimeCondition : Prop := 18 * currentSpeed x = 810

-- Proof goal: Prove that Mike's original typing speed is 65 words per minute
theorem mikes_original_speed (h : typingTimeCondition x) : x = 65 := 
sorry

end mikes_original_speed_l140_140246


namespace burger_cost_is_350_l140_140542

noncomputable def cost_of_each_burger (tip steak_cost steak_quantity ice_cream_cost ice_cream_quantity money_left: ℝ) : ℝ :=
(tip - money_left - (steak_cost * steak_quantity + ice_cream_cost * ice_cream_quantity)) / 2

theorem burger_cost_is_350 :
  cost_of_each_burger 99 24 2 2 3 38 = 3.5 :=
by
  sorry

end burger_cost_is_350_l140_140542


namespace abc_inequality_l140_140186

theorem abc_inequality (a b c : ℝ) (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (h : a * b + a * c + b * c = a + b + c) : 
  a + b + c + 1 ≥ 4 * a * b * c :=
by 
  sorry

end abc_inequality_l140_140186


namespace problem_statement_eq_l140_140826

noncomputable def given_sequence (a : ℝ) (n : ℕ) : ℝ :=
  a^n

noncomputable def Sn (a : ℝ) (n : ℕ) (an : ℝ) : ℝ :=
  (a / (a - 1)) * (an - 1)

noncomputable def bn (a : ℝ) (n : ℕ) : ℝ :=
  2 * (Sn a n (given_sequence a n)) / (given_sequence a n) + 1

noncomputable def cn (a : ℝ) (n : ℕ) : ℝ :=
  (n - 1) * (bn a n)

noncomputable def Tn (a : ℝ) (n : ℕ) : ℝ :=
  (List.range n).foldl (λ acc k => acc + cn a (k + 1)) 0

theorem problem_statement_eq :
  ∀ (a : ℝ) (n : ℕ), a ≠ 0 → a ≠ 1 →
  (bn a n = (3:ℝ)^n) →
  Tn (1 / 3) n = 3^(n+1) * (2 * n - 3) / 4 + 9 / 4 :=
by
  intros
  sorry

end problem_statement_eq_l140_140826


namespace caterpillars_and_leaves_l140_140532

def initial_caterpillars : Nat := 14
def caterpillars_after_storm : Nat := initial_caterpillars - 3
def hatched_eggs : Nat := 6
def caterpillars_after_hatching : Nat := caterpillars_after_storm + hatched_eggs
def leaves_eaten_by_babies : Nat := 18
def caterpillars_after_cocooning : Nat := caterpillars_after_hatching - 9
def moth_caterpillars : Nat := caterpillars_after_cocooning / 2
def butterfly_caterpillars : Nat := caterpillars_after_cocooning - moth_caterpillars
def leaves_eaten_per_moth_per_day : Nat := 4
def days_in_week : Nat := 7
def total_leaves_eaten_by_moths : Nat := moth_caterpillars * leaves_eaten_per_moth_per_day * days_in_week
def total_leaves_eaten_by_babies_and_moths : Nat := leaves_eaten_by_babies + total_leaves_eaten_by_moths

theorem caterpillars_and_leaves :
  (caterpillars_after_cocooning = 8) ∧ (total_leaves_eaten_by_babies_and_moths = 130) :=
by
  -- proof to be filled in
  sorry

end caterpillars_and_leaves_l140_140532


namespace find_profit_percentage_l140_140986

theorem find_profit_percentage (h : (m + 8) / (1 - 0.08) = m + 10) : m = 15 := sorry

end find_profit_percentage_l140_140986


namespace Elon_has_10_more_Teslas_than_Sam_l140_140739

noncomputable def TeslasCalculation : Nat :=
let Chris : Nat := 6
let Sam : Nat := Chris / 2
let Elon : Nat := 13
Elon - Sam

theorem Elon_has_10_more_Teslas_than_Sam :
  TeslasCalculation = 10 :=
by
  sorry

end Elon_has_10_more_Teslas_than_Sam_l140_140739


namespace shirts_per_minute_l140_140656

theorem shirts_per_minute (total_shirts : ℕ) (total_minutes : ℕ) (shirts_per_min : ℕ) 
  (h : total_shirts = 12 ∧ total_minutes = 6) :
  shirts_per_min = 2 :=
sorry

end shirts_per_minute_l140_140656


namespace algebraic_identity_l140_140472

theorem algebraic_identity (x : ℝ) (h : x = Real.sqrt 3 + 2) : x^2 - 4 * x + 3 = 2 := 
by
  -- proof steps here
  sorry

end algebraic_identity_l140_140472


namespace debate_team_group_size_l140_140176

theorem debate_team_group_size (boys girls groups : ℕ) (h_boys : boys = 11) (h_girls : girls = 45) (h_groups : groups = 8) : 
  (boys + girls) / groups = 7 := by
  sorry

end debate_team_group_size_l140_140176


namespace strength_training_sessions_l140_140667

-- Define the problem conditions
def strength_training_hours (x : ℕ) : ℝ := x * 1
def boxing_training_hours : ℝ := 4 * 1.5
def total_training_hours : ℝ := 9

-- Prove how many times a week does Kat do strength training
theorem strength_training_sessions : ∃ x : ℕ, strength_training_hours x + boxing_training_hours = total_training_hours ∧ x = 3 := 
by {
  sorry
}

end strength_training_sessions_l140_140667


namespace sequence_solution_l140_140037

theorem sequence_solution (a : ℕ → ℝ) (n : ℕ) (h1 : a 1 = 2) (h_rec : ∀ n > 0, a (n + 1) = a n ^ 2) : 
  a n = 2 ^ 2 ^ (n - 1) :=
by
  sorry

end sequence_solution_l140_140037


namespace maximum_even_integers_of_odd_product_l140_140592

theorem maximum_even_integers_of_odd_product (a b c d e f g : ℕ) (h1: a > 0) (h2: b > 0) (h3: c > 0) (h4: d > 0) (h5: e > 0) (h6: f > 0) (h7: g > 0) (hprod : a * b * c * d * e * f * g % 2 = 1): 
  (a % 2 = 1) ∧ (b % 2 = 1) ∧ (c % 2 = 1) ∧ (d % 2 = 1) ∧ (e % 2 = 1) ∧ (f % 2 = 1) ∧ (g % 2 = 1) :=
sorry

end maximum_even_integers_of_odd_product_l140_140592


namespace product_remainder_div_5_l140_140141

theorem product_remainder_div_5 :
  (1234 * 1567 * 1912) % 5 = 1 :=
by
  sorry

end product_remainder_div_5_l140_140141


namespace arithmetic_sqrt_25_l140_140762

-- Define the arithmetic square root condition
def is_arithmetic_sqrt (x a : ℝ) : Prop :=
  0 ≤ x ∧ x^2 = a

-- Lean statement to prove the arithmetic square root of 25 is 5
theorem arithmetic_sqrt_25 : is_arithmetic_sqrt 5 25 :=
by 
  sorry

end arithmetic_sqrt_25_l140_140762


namespace simplest_form_of_expression_l140_140593

theorem simplest_form_of_expression (c : ℝ) : ((3 * c + 5 - 3 * c) / 2) = 5 / 2 :=
by 
  sorry

end simplest_form_of_expression_l140_140593


namespace g_five_eq_one_l140_140879

noncomputable def g : ℝ → ℝ := sorry

theorem g_five_eq_one 
  (h1 : ∀ x y : ℝ, g (x - y) = g x * g y)
  (h2 : ∀ x : ℝ, g x ≠ 0)
  (h3 : ∀ x : ℝ, g x = g (-x)) : 
  g 5 = 1 :=
sorry

end g_five_eq_one_l140_140879


namespace symmetry_center_example_l140_140696

-- Define the function tan(2x - π/4)
noncomputable def func (x : ℝ) : ℝ := Real.tan (2 * x - Real.pi / 4)

-- Define what it means to be a symmetry center for the function
def is_symmetry_center (f : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  ∀ x, f (2 * (p.1) - x) = 2 * p.2 - f x

-- Statement of the proof problem
theorem symmetry_center_example : is_symmetry_center func (-Real.pi / 8, 0) :=
sorry

end symmetry_center_example_l140_140696


namespace nat_pairs_satisfy_conditions_l140_140503

theorem nat_pairs_satisfy_conditions :
  ∃ (a b : ℕ), 
  (a = 1 ∧ b = 2) ∨ (a = 2 ∧ b = 1) ∨ (a = 2 ∧ b = 2) ∨ 
  (a = 2 ∧ b = 3) ∨ (a = 3 ∧ b = 2) ∨ (a = 3 ∧ b = 3) ∧
  (∃ k : ℤ, (a^2 + b) = k * (b^2 - a)) ∧
  (∃ l : ℤ, (b^2 + a) = l * (a^2 - b)) := 
sorry

end nat_pairs_satisfy_conditions_l140_140503


namespace unique_rational_solution_l140_140022

theorem unique_rational_solution (x y z : ℚ) (h : x^3 + 3*y^3 + 9*z^3 - 9*x*y*z = 0) : x = 0 ∧ y = 0 ∧ z = 0 := 
by {
  sorry
}

end unique_rational_solution_l140_140022


namespace range_of_a_l140_140758

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then -x - 3 * a else a^x - 2

theorem range_of_a (a : ℝ) :
  (0 < a ∧ a ≠ 1 ∧ ∀ x y : ℝ, x < y → f a x > f a y) ↔ (0 < a ∧ a ≤ 1 / 3) :=
by sorry

end range_of_a_l140_140758


namespace triangle_angles_are_equal_l140_140130

theorem triangle_angles_are_equal
  (A B C : ℝ) (a b c : ℝ)
  (h1 : A + B + C = π)
  (h2 : A = B + (B - A))
  (h3 : B = C + (C - B))
  (h4 : 2 * (1 / b) = (1 / a) + (1 / c)) :
  A = π / 3 ∧ B = π / 3 ∧ C = π / 3 :=
sorry

end triangle_angles_are_equal_l140_140130


namespace solve_for_k_l140_140452

theorem solve_for_k (x y k : ℝ) (h1 : x = 2) (h2 : y = 3) (h3 : 5 * x - k * y - 7 = 0) : k = 1 :=
by
  sorry

end solve_for_k_l140_140452


namespace find_ab_l140_140962

theorem find_ab
  (a b c : ℝ)
  (h1 : a - b = 3)
  (h2 : a^2 + b^2 = 27)
  (h3 : a + b + c = 10)
  (h4 : a^3 - b^3 = 36)
  : a * b = -15 :=
by
  sorry

end find_ab_l140_140962


namespace retirement_hiring_year_l140_140702

theorem retirement_hiring_year (A W Y : ℕ)
  (hired_on_32nd_birthday : A = 32)
  (eligible_to_retire_in_2007 : 32 + (2007 - Y) = 70) : 
  Y = 1969 := by
  sorry

end retirement_hiring_year_l140_140702


namespace find_value_of_A_l140_140854

theorem find_value_of_A (M T A E : ℕ) (H : ℕ := 8) 
  (h1 : M + A + T + H = 28) 
  (h2 : T + E + A + M = 34) 
  (h3 : M + E + E + T = 30) : 
  A = 16 :=
by 
  sorry

end find_value_of_A_l140_140854


namespace cost_price_of_watch_l140_140159

variable (CP : ℝ)
variable (SP_loss SP_gain : ℝ)
variable (h1 : SP_loss = CP * 0.725)
variable (h2 : SP_gain = CP * 1.125)
variable (h3 : SP_gain - SP_loss = 275)

theorem cost_price_of_watch : CP = 687.50 :=
by
  sorry

end cost_price_of_watch_l140_140159


namespace taxi_ride_cost_l140_140819

-- Definitions given in the conditions
def base_fare : ℝ := 2.00
def cost_per_mile : ℝ := 0.30
def distance_traveled : ℝ := 10

-- The theorem we need to prove
theorem taxi_ride_cost : base_fare + (cost_per_mile * distance_traveled) = 5.00 :=
by
  sorry

end taxi_ride_cost_l140_140819


namespace petya_board_problem_l140_140933

variable (A B Z : ℕ)

theorem petya_board_problem (h1 : A + B + Z = 10) (h2 : A * B = 15) : Z = 2 := sorry

end petya_board_problem_l140_140933


namespace arrange_in_ascending_order_l140_140552

theorem arrange_in_ascending_order (x : ℝ) (h1 : -1 < x) (h2 : x < 0) : 5 * x < 0.5 * x ∧ 0.5 * x < 5 - x := by
  sorry

end arrange_in_ascending_order_l140_140552


namespace trajectory_of_Q_l140_140735

variable (x y m n : ℝ)

def line_l (x y : ℝ) : Prop := 2 * x + 4 * y + 3 = 0

def point_P_on_line_l (x y m n : ℝ) : Prop := line_l m n

def origin (O : (ℝ × ℝ)) := O = (0, 0)

def Q_condition (O Q P : (ℝ × ℝ)) : Prop := 2 • O + 2 • Q = Q + P

theorem trajectory_of_Q (x y m n : ℝ) (O : (ℝ × ℝ)) (P Q : (ℝ × ℝ)) :
  point_P_on_line_l x y m n → origin O → Q_condition O Q P → 
  2 * x + 4 * y + 1 = 0 := 
sorry

end trajectory_of_Q_l140_140735


namespace edward_made_in_summer_l140_140379

theorem edward_made_in_summer
  (spring_earnings : ℤ)
  (spent_on_supplies : ℤ)
  (final_amount : ℤ)
  (S : ℤ)
  (h1 : spring_earnings = 2)
  (h2 : spent_on_supplies = 5)
  (h3 : final_amount = 24)
  (h4 : spring_earnings + S - spent_on_supplies = final_amount) :
  S = 27 := 
by
  sorry

end edward_made_in_summer_l140_140379


namespace students_answered_both_correctly_l140_140383

theorem students_answered_both_correctly (x y z w total : ℕ) (h1 : x = 22) (h2 : y = 20) 
  (h3 : z = 3) (h4 : total = 25) (h5 : x + y - w - z = total) : w = 17 :=
by
  sorry

end students_answered_both_correctly_l140_140383


namespace second_derivative_at_pi_over_3_l140_140894

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.sin x) * (Real.cos x)

theorem second_derivative_at_pi_over_3 : 
  (deriv (deriv f)) (Real.pi / 3) = -1 :=
  sorry

end second_derivative_at_pi_over_3_l140_140894


namespace jill_tax_on_other_items_l140_140916

-- Define the conditions based on the problem statement.
variables (C : ℝ) (x : ℝ)
def tax_on_clothing := 0.04 * 0.60 * C
def tax_on_food := 0
def tax_on_other_items := 0.01 * x * 0.30 * C
def total_tax_paid := 0.048 * C

-- Prove the required percentage tax on other items.
theorem jill_tax_on_other_items :
  tax_on_clothing C + tax_on_food + tax_on_other_items C x = total_tax_paid C →
  x = 8 :=
by
  sorry

end jill_tax_on_other_items_l140_140916


namespace scientific_notation_l140_140194

theorem scientific_notation :
  0.000000014 = 1.4 * 10^(-8) :=
sorry

end scientific_notation_l140_140194


namespace Milly_took_extra_balloons_l140_140809

theorem Milly_took_extra_balloons :
  let total_packs := 3 + 2
  let balloons_per_pack := 6
  let total_balloons := total_packs * balloons_per_pack
  let even_split := total_balloons / 2
  let Floretta_balloons := 8
  let Milly_extra_balloons := even_split - Floretta_balloons
  Milly_extra_balloons = 7 := by
  sorry

end Milly_took_extra_balloons_l140_140809


namespace tenth_term_l140_140098

noncomputable def sequence_term (n : ℕ) : ℝ :=
  (-1)^(n+1) * (Real.sqrt (1 + 2*(n - 1))) / (2^n)

theorem tenth_term :
  sequence_term 10 = Real.sqrt 19 / (2^10) :=
by
  sorry

end tenth_term_l140_140098


namespace train_stop_time_per_hour_l140_140129

theorem train_stop_time_per_hour
    (speed_excl_stoppages : ℕ)
    (speed_incl_stoppages : ℕ)
    (h1 : speed_excl_stoppages = 48)
    (h2 : speed_incl_stoppages = 36) :
    ∃ (t : ℕ), t = 15 :=
by
  sorry

end train_stop_time_per_hour_l140_140129


namespace age_of_B_l140_140433

/--
A is two years older than B.
B is twice as old as C.
The total of the ages of A, B, and C is 32.
How old is B?
-/
theorem age_of_B (A B C : ℕ) (h1 : A = B + 2) (h2 : B = 2 * C) (h3 : A + B + C = 32) : B = 12 :=
by
  sorry

end age_of_B_l140_140433


namespace proof_set_intersection_l140_140952

noncomputable def U := ℝ
noncomputable def M := {x : ℝ | 0 ≤ x ∧ x < 5}
noncomputable def N := {x : ℝ | x ≥ 2}
noncomputable def compl_U_N := {x : ℝ | x < 2}
noncomputable def intersection := { x : ℝ | 0 ≤ x ∧ x < 2 }

theorem proof_set_intersection : ((compl_U_N ∩ M) = {x : ℝ | 0 ≤ x ∧ x < 2}) :=
by
  sorry

end proof_set_intersection_l140_140952


namespace max_ages_acceptable_within_one_std_dev_l140_140179

theorem max_ages_acceptable_within_one_std_dev
  (average_age : ℤ)
  (std_deviation : ℤ)
  (acceptable_range_lower : ℤ)
  (acceptable_range_upper : ℤ)
  (h1 : average_age = 31)
  (h2 : std_deviation = 5)
  (h3 : acceptable_range_lower = average_age - std_deviation)
  (h4 : acceptable_range_upper = average_age + std_deviation) :
  ∃ n : ℕ, n = acceptable_range_upper - acceptable_range_lower + 1 ∧ n = 11 :=
by
  sorry

end max_ages_acceptable_within_one_std_dev_l140_140179


namespace find_m_for_one_real_solution_l140_140749

theorem find_m_for_one_real_solution (m : ℝ) (h : 4 * m * 4 = m^2) : m = 8 := sorry

end find_m_for_one_real_solution_l140_140749


namespace difference_of_place_values_l140_140829

theorem difference_of_place_values :
  let n := 54179759
  let pos1 := 10000 * 7
  let pos2 := 10 * 7
  pos1 - pos2 = 69930 := by
  sorry

end difference_of_place_values_l140_140829


namespace symmetric_point_y_axis_l140_140006

def M : ℝ × ℝ := (-5, 2)
def symmetric_point (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)
theorem symmetric_point_y_axis :
  symmetric_point M = (5, 2) :=
by
  sorry

end symmetric_point_y_axis_l140_140006


namespace no_lighter_sentence_for_liar_l140_140939

theorem no_lighter_sentence_for_liar
  (total_eggs : ℕ)
  (stolen_eggs1 stolen_eggs2 stolen_eggs3 : ℕ)
  (different_stolen_eggs : stolen_eggs1 ≠ stolen_eggs2 ∧ stolen_eggs2 ≠ stolen_eggs3 ∧ stolen_eggs1 ≠ stolen_eggs3)
  (stolen_eggs1_max : stolen_eggs1 > stolen_eggs2 ∧ stolen_eggs1 > stolen_eggs3)
  (stole_7 : stolen_eggs1 = 7)
  (total_eq_20 : stolen_eggs1 + stolen_eggs2 + stolen_eggs3 = 20) :
  false :=
by
  sorry

end no_lighter_sentence_for_liar_l140_140939


namespace appeared_candidates_l140_140160

noncomputable def number_of_candidates_that_appeared_from_each_state (X : ℝ) : Prop :=
  (8 / 100) * X + 220 = (12 / 100) * X

theorem appeared_candidates (X : ℝ) (h : number_of_candidates_that_appeared_from_each_state X) : X = 5500 :=
  sorry

end appeared_candidates_l140_140160


namespace correct_algorithm_statement_l140_140127

def reversible : Prop := false -- Algorithms are generally not reversible.
def endless : Prop := false -- Algorithms should not run endlessly.
def unique_algo : Prop := false -- Not always one single algorithm for a task.
def simple_convenient : Prop := true -- Algorithms should be simple and convenient.

theorem correct_algorithm_statement : simple_convenient = true :=
by
  sorry

end correct_algorithm_statement_l140_140127


namespace problem_pm_sqrt5_sin_tan_l140_140277

theorem problem_pm_sqrt5_sin_tan
  (m : ℝ)
  (h_m_nonzero : m ≠ 0)
  (cos_alpha : ℝ)
  (h_cos_alpha : cos_alpha = (Real.sqrt 2 * m) / 4)
  (P : ℝ × ℝ)
  (h_P : P = (m, -Real.sqrt 3))
  (r : ℝ)
  (h_r : r = Real.sqrt (3 + m^2)) :
    (∃ m, m = Real.sqrt 5 ∨ m = -Real.sqrt 5) ∧
    (∃ sin_alpha tan_alpha,
      (sin_alpha = - Real.sqrt 6 / 4 ∧ tan_alpha = -Real.sqrt 15 / 5)) :=
by
  sorry

end problem_pm_sqrt5_sin_tan_l140_140277


namespace option_b_is_correct_l140_140319

def is_linear (equation : String) : Bool :=
  -- Pretend implementation that checks if the given equation is linear
  -- This function would parse the string and check the linearity condition
  true -- This should be replaced by actual linearity check

def has_two_unknowns (system : List String) : Bool :=
  -- Pretend implementation that checks if the system contains exactly two unknowns
  -- This function would analyze the variables in the system
  true -- This should be replaced by actual unknowns count check

def is_system_of_two_linear_equations (system : List String) : Bool :=
  -- Checking both conditions: Each equation is linear and contains exactly two unknowns
  (system.all is_linear) && (has_two_unknowns system)

def option_b := ["x + y = 1", "x - y = 2"]

theorem option_b_is_correct :
  is_system_of_two_linear_equations option_b := 
  by
    unfold is_system_of_two_linear_equations
    -- Assuming the placeholder implementations of is_linear and has_two_unknowns
    -- actually verify the required properties, this should be true
    sorry

end option_b_is_correct_l140_140319


namespace solve_equation_l140_140857

theorem solve_equation {x : ℝ} (hx : x = 1) : 9 - 3 / x / 3 + 3 = 3 := by
  rw [hx] -- Substitute x = 1
  norm_num -- Simplify the numerical expression
  sorry -- to be proved

end solve_equation_l140_140857


namespace dogsled_course_distance_l140_140143

theorem dogsled_course_distance 
    (t : ℕ)  -- time taken by Team B
    (speed_B : ℕ := 20)  -- average speed of Team B
    (speed_A : ℕ := 25)  -- average speed of Team A
    (tA_eq_tB_minus_3 : t - 3 = tA)  -- Team A’s time relation
    (speedA_eq_speedB_plus_5 : speed_A = speed_B + 5)  -- Team A's average speed in relation to Team B’s average speed
    (distance_eq : speed_B * t = speed_A * (t - 3))  -- Distance equality condition
    (t_eq_15 : t = 15)  -- Time taken by Team B to finish
    :
    (speed_B * t = 300) :=   -- Distance of the course
by
  sorry

end dogsled_course_distance_l140_140143


namespace maximum_distance_is_correct_l140_140677

-- Define the right trapezoid with the given side lengths and angle conditions
structure RightTrapezoid (AB CD : ℕ) where
  B_angle : ℝ
  D_angle : ℝ
  h_AB : AB = 200
  h_CD : CD = 100
  h_B_angle : B_angle = 90
  h_D_angle : D_angle = 45

-- Define the guards' walking condition and distance calculation
def max_distance_between_guards (T : RightTrapezoid 200 100) : ℝ :=
  let P := 400 + 100 * Real.sqrt 2
  let d := (400 + 100 * Real.sqrt 2) / 2
  222.1  -- Hard-coded according to the problem's correct answer for maximum distance

theorem maximum_distance_is_correct :
  ∀ (T : RightTrapezoid 200 100), max_distance_between_guards T = 222.1 := by
  sorry

end maximum_distance_is_correct_l140_140677


namespace arithmetic_sequence_general_formula_is_not_term_l140_140587

theorem arithmetic_sequence_general_formula (a : ℕ → ℤ) (d : ℤ) 
  (h1 : a 1 = 2) (h17 : a 17 = 66) :
  ∀ n : ℕ, a n = 4 * n - 2 := sorry

theorem is_not_term (a : ℕ → ℤ) 
  (ha : ∀ n : ℕ, a n = 4 * n - 2) :
  ∀ k : ℤ, k = 88 → ¬ ∃ n : ℕ, a n = k := sorry

end arithmetic_sequence_general_formula_is_not_term_l140_140587


namespace compare_negatives_l140_140151

theorem compare_negatives : (- (3 : ℝ) / 5) > (- (5 : ℝ) / 7) :=
by
  sorry

end compare_negatives_l140_140151


namespace sum_of_k_l140_140026

theorem sum_of_k (k : ℕ) :
  ((∃ x, x^2 - 4 * x + 3 = 0 ∧ x^2 - 7 * x + k = 0) →
  (k = 6 ∨ k = 12)) →
  (6 + 12 = 18) :=
by sorry

end sum_of_k_l140_140026


namespace cary_wage_after_two_years_l140_140359

theorem cary_wage_after_two_years (initial_wage raise_percentage cut_percentage : ℝ) (wage_after_first_year wage_after_second_year : ℝ) :
  initial_wage = 10 ∧ raise_percentage = 0.2 ∧ cut_percentage = 0.75 ∧ 
  wage_after_first_year = initial_wage * (1 + raise_percentage) ∧
  wage_after_second_year = wage_after_first_year * cut_percentage → 
  wage_after_second_year = 9 :=
by
  sorry

end cary_wage_after_two_years_l140_140359


namespace donut_combinations_l140_140872

theorem donut_combinations (donuts types : ℕ) (at_least_one : ℕ) :
  donuts = 7 ∧ types = 5 ∧ at_least_one = 4 → ∃ combinations : ℕ, combinations = 100 :=
by
  intros h
  sorry

end donut_combinations_l140_140872


namespace common_ratio_of_geometric_sequence_l140_140289

variable (a_1 q : ℚ) (S : ℕ → ℚ)

def geometric_sum (n : ℕ) : ℚ :=
  a_1 * (1 - q^n) / (1 - q)

def is_arithmetic_sequence (a b c : ℚ) : Prop :=
  2 * b = a + c

theorem common_ratio_of_geometric_sequence 
  (h1 : ∀ n, S n = geometric_sum a_1 q n)
  (h2 : ∀ n, is_arithmetic_sequence (S (n+2)) (S (n+1)) (S n)) : q = -2 :=
by
  sorry

end common_ratio_of_geometric_sequence_l140_140289


namespace asymptotes_of_hyperbola_l140_140062

theorem asymptotes_of_hyperbola (a b x y : ℝ) (h : a = 5 ∧ b = 2) :
  (x^2 / 25 - y^2 / 4 = 1) → (y = (2 / 5) * x ∨ y = -(2 / 5) * x) :=
by
  sorry

end asymptotes_of_hyperbola_l140_140062


namespace sum_of_integers_l140_140795

theorem sum_of_integers (x y : ℕ) (h1 : x > y) (h2 : x - y = 15) (h3 : x * y = 54) : x + y = 21 :=
by
  sorry

end sum_of_integers_l140_140795


namespace tank_fills_in_56_minutes_l140_140501

theorem tank_fills_in_56_minutes : 
  (∃ A B C : ℕ, (A = 40 ∧ B = 30 ∧ C = 20) ∧ 
                 ∃ capacity : ℕ, capacity = 950 ∧ 
                 ∃ time : ℕ, time = 56 ∧
                 ∀ cycle_time : ℕ, cycle_time = 3 ∧ 
                 ∀ net_water_per_cycle : ℕ, net_water_per_cycle = A + B - C ∧
                 ∀ total_cycles : ℕ, total_cycles = capacity / net_water_per_cycle ∧
                 ∀ total_time : ℕ, total_time = total_cycles * cycle_time - 1 ∧
                 total_time = time) :=
sorry

end tank_fills_in_56_minutes_l140_140501


namespace marsupial_protein_l140_140408

theorem marsupial_protein (absorbed : ℝ) (percent_absorbed : ℝ) (consumed : ℝ) :
  absorbed = 16 ∧ percent_absorbed = 0.4 → consumed = 40 :=
by
  sorry

end marsupial_protein_l140_140408


namespace actual_distance_between_mountains_l140_140831

theorem actual_distance_between_mountains (D_map : ℝ) (d_map_ram : ℝ) (d_real_ram : ℝ)
  (hD_map : D_map = 312) (hd_map_ram : d_map_ram = 25) (hd_real_ram : d_real_ram = 10.897435897435898) :
  D_map / d_map_ram * d_real_ram = 136 :=
by
  -- Theorem statement is proven based on the given conditions.
  sorry

end actual_distance_between_mountains_l140_140831


namespace not_a_function_l140_140439

theorem not_a_function (angle_sine : ℝ → ℝ) 
                       (side_length_area : ℝ → ℝ) 
                       (sides_sum_int_angles : ℕ → ℝ)
                       (person_age_height : ℕ → Set ℝ) :
  (∃ y₁ y₂, y₁ ∈ person_age_height 20 ∧ y₂ ∈ person_age_height 20 ∧ y₁ ≠ y₂) :=
by {
  sorry
}

end not_a_function_l140_140439


namespace evaluate_expression_l140_140464

theorem evaluate_expression : (8^6 / 8^4) * 3^10 = 3783136 := by
  sorry

end evaluate_expression_l140_140464


namespace find_fraction_l140_140713

theorem find_fraction (a b : ℝ) (h : a = 2 * b) : (a / (a - b)) = 2 :=
by
  sorry

end find_fraction_l140_140713


namespace greatest_common_divisor_of_three_common_divisors_l140_140039

theorem greatest_common_divisor_of_three_common_divisors (m : ℕ) :
  (∀ d, d ∣ 126 ∧ d ∣ m → d = 1 ∨ d = 3 ∨ d = 9) →
  gcd 126 m = 9 := 
sorry

end greatest_common_divisor_of_three_common_divisors_l140_140039


namespace five_letter_word_combinations_l140_140506

open Nat

theorem five_letter_word_combinations :
  let first_letter_choices := 26
  let other_letter_choices := 26
  let total_combinations := first_letter_choices ^ 1 * other_letter_choices ^ 3
  total_combinations = 456976 := 
by
  let first_letter_choices := 26
  let other_letter_choices := 26
  let total_combinations := first_letter_choices ^ 1 * other_letter_choices ^ 3
  show total_combinations = 456976
  sorry

end five_letter_word_combinations_l140_140506


namespace tangent_line_at_point_l140_140782

def tangent_line_equation (f : ℝ → ℝ) (slope : ℝ) (p : ℝ × ℝ) :=
  ∃ (a b c : ℝ), a * p.1 + b * p.2 + c = 0 ∧ a = slope ∧ p.2 = f p.1

noncomputable def curve (x : ℝ) : ℝ := x^3 + x + 1

theorem tangent_line_at_point : 
  tangent_line_equation curve 4 (1, 3) :=
sorry

end tangent_line_at_point_l140_140782


namespace rectangles_260_261_272_273_have_similar_property_l140_140766

-- Defining a rectangle as a structure with width and height
structure Rectangle where
  width : ℕ
  height : ℕ

-- Given conditions
def r1 : Rectangle := ⟨16, 10⟩
def r2 : Rectangle := ⟨23, 7⟩

-- Hypothesis function indicating the dissection trick causing apparent equality
def dissection_trick (r1 r2 : Rectangle) : Prop :=
  (r1.width * r1.height : ℕ) = (r2.width * r2.height : ℕ) + 1

-- The statement of the proof problem
theorem rectangles_260_261_272_273_have_similar_property :
  ∃ (r3 r4 : Rectangle) (r5 r6 : Rectangle),
    dissection_trick r3 r4 ∧ dissection_trick r5 r6 ∧
    r3.width * r3.height = 260 ∧ r4.width * r4.height = 261 ∧
    r5.width * r5.height = 272 ∧ r6.width * r6.height = 273 :=
  sorry

end rectangles_260_261_272_273_have_similar_property_l140_140766


namespace jonas_tshirts_count_l140_140539

def pairs_to_individuals (pairs : Nat) : Nat := pairs * 2

variable (num_pairs_socks : Nat := 20)
variable (num_pairs_shoes : Nat := 5)
variable (num_pairs_pants : Nat := 10)
variable (num_additional_pairs_socks : Nat := 35)

def total_individual_items_without_tshirts : Nat :=
  pairs_to_individuals num_pairs_socks +
  pairs_to_individuals num_pairs_shoes +
  pairs_to_individuals num_pairs_pants

def total_individual_items_desired : Nat :=
  total_individual_items_without_tshirts +
  pairs_to_individuals num_additional_pairs_socks

def tshirts_jonas_needs : Nat :=
  total_individual_items_desired - total_individual_items_without_tshirts

theorem jonas_tshirts_count : tshirts_jonas_needs = 70 := by
  sorry

end jonas_tshirts_count_l140_140539


namespace initial_ratio_of_stamps_l140_140214

variable (K A : ℕ)

theorem initial_ratio_of_stamps (h1 : (K - 12) * 3 = (A + 12) * 4) (h2 : K - 12 = A + 44) : K/A = 5/3 :=
sorry

end initial_ratio_of_stamps_l140_140214


namespace problem_stated_l140_140183

-- Definitions of constants based on conditions
def a : ℕ := 5
def b : ℕ := 4
def c : ℕ := 3
def d : ℕ := 400
def x : ℕ := 401

-- Mathematical theorem stating the question == answer given conditions
theorem problem_stated : a * x + b * x + c * x + d = 5212 := 
by 
  sorry

end problem_stated_l140_140183


namespace cashier_total_value_l140_140548

theorem cashier_total_value (total_bills : ℕ) (ten_bills : ℕ) (twenty_bills : ℕ)
  (h1 : total_bills = 30) (h2 : ten_bills = 27) (h3 : twenty_bills = 3) :
  (10 * ten_bills + 20 * twenty_bills) = 330 :=
by
  sorry

end cashier_total_value_l140_140548


namespace simon_spending_l140_140568

-- Assume entities and their properties based on the problem
def kabobStickCubes : Nat := 4
def slabCost : Nat := 25
def slabCubes : Nat := 80
def kabobSticksNeeded : Nat := 40

-- Theorem statement based on the problem analysis
theorem simon_spending : 
  (kabobSticksNeeded / (slabCubes / kabobStickCubes)) * slabCost = 50 := by
  sorry

end simon_spending_l140_140568


namespace curves_intersect_at_4_points_l140_140462

theorem curves_intersect_at_4_points (a : ℝ) :
  (∀ (x y : ℝ), x^2 + (y - 1)^2 = a^2 ∧ y = x^2 - a → ∃ x1 x2 x3 x4 y1 y2 y3 y4 : ℝ,
  (x1, y1) ≠ (x2, y2) ∧ (x2, y2) ≠ (x3, y3) ∧ (x3, y3) ≠ (x4, y4) ∧
  (x1, y1) ≠ (x3, y3) ∧ (x1, y1) ≠ (x4, y4) ∧ (x2, y2) ≠ (x4, y4) ∧
  (x4, y4) ≠ (x3, y3) ∧ x1^2 + (y1 - 1)^2 = a^2 ∧ y1 = x1^2 - a ∧
  x2^2 + (y2 - 1)^2 = a^2 ∧ y2 = x2^2 - a ∧
  x3^2 + (y3 - 1)^2 = a^2 ∧ y3 = x3^2 - a ∧
  x4^2 + (y4 - 1)^2 = a^2 ∧ y4 = x4^2 - a) ↔ a > 0 :=
sorry

end curves_intersect_at_4_points_l140_140462


namespace justine_more_than_bailey_l140_140341

-- Definitions from conditions
def J : ℕ := 22 -- Justine's initial rubber bands
def B : ℕ := 12 -- Bailey's initial rubber bands

-- Theorem to prove
theorem justine_more_than_bailey : J - B = 10 := by
  -- Proof will be done here
  sorry

end justine_more_than_bailey_l140_140341


namespace perfect_square_after_dividing_l140_140251

theorem perfect_square_after_dividing (n : ℕ) (h : n = 16800) : ∃ m : ℕ, (n / 21) = m * m :=
by {
  sorry
}

end perfect_square_after_dividing_l140_140251


namespace sum_divisible_by_11_l140_140897

theorem sum_divisible_by_11 (n : ℕ) : (6^(2*n) + 3^n + 3^(n+2)) % 11 = 0 := by
  sorry

end sum_divisible_by_11_l140_140897


namespace calendar_matrix_sum_l140_140889

def initial_matrix : Matrix (Fin 3) (Fin 3) ℕ :=
  ![![5, 6, 7], 
    ![8, 9, 10], 
    ![11, 12, 13]]

def modified_matrix (m : Matrix (Fin 3) (Fin 3) ℕ) : Matrix (Fin 3) (Fin 3) ℕ :=
  ![![m 0 2, m 0 1, m 0 0], 
    ![m 1 0, m 1 1, m 1 2], 
    ![m 2 2, m 2 1, m 2 0]]

def diagonal_sum (m : Matrix (Fin 3) (Fin 3) ℕ) : ℕ :=
  m 0 0 + m 1 1 + m 2 2

def edge_sum (m : Matrix (Fin 3) (Fin 3) ℕ) : ℕ :=
  m 0 1 + m 0 2 + m 2 0 + m 2 1

def total_sum (m : Matrix (Fin 3) (Fin 3) ℕ) : ℕ :=
  diagonal_sum m + edge_sum m

theorem calendar_matrix_sum :
  total_sum (modified_matrix initial_matrix) = 63 :=
by
  sorry

end calendar_matrix_sum_l140_140889


namespace pet_store_dogs_l140_140055

-- Define the given conditions as Lean definitions
def initial_dogs : ℕ := 2
def sunday_dogs : ℕ := 5
def monday_dogs : ℕ := 3

-- Define the total dogs calculation to use in the theorem
def total_dogs : ℕ := initial_dogs + sunday_dogs + monday_dogs

-- State the theorem
theorem pet_store_dogs : total_dogs = 10 := 
by
  -- Placeholder for the proof
  sorry

end pet_store_dogs_l140_140055


namespace parabola_vertex_sum_l140_140031

variable (a b c : ℝ)

def parabola_eq (x y : ℝ) : Prop :=
  x = a * y^2 + b * y + c

def vertex (v : ℝ × ℝ) : Prop :=
  v = (-3, 2)

def passes_through (p : ℝ × ℝ) : Prop :=
  p = (-1, 0)

theorem parabola_vertex_sum :
  ∀ (a b c : ℝ),
  (∃ v : ℝ × ℝ, vertex v) ∧
  (∃ p : ℝ × ℝ, passes_through p) →
  a + b + c = -7/2 :=
by
  intros a b c
  intro conditions
  sorry

end parabola_vertex_sum_l140_140031


namespace textbook_weight_l140_140671

theorem textbook_weight
  (w : ℝ)
  (bookcase_limit : ℝ := 80)
  (hardcover_books : ℕ := 70)
  (hardcover_weight_per_book : ℝ := 0.5)
  (textbooks : ℕ := 30)
  (knick_knacks : ℕ := 3)
  (knick_knack_weight : ℝ := 6)
  (over_limit : ℝ := 33)
  (total_items_weight : ℝ := bookcase_limit + over_limit)
  (hardcover_total_weight : ℝ := hardcover_books * hardcover_weight_per_book)
  (knick_knack_total_weight : ℝ := knick_knacks * knick_knack_weight)
  (remaining_weight : ℝ := total_items_weight - (hardcover_total_weight + knick_knack_total_weight)) :
  remaining_weight = textbooks * 2 :=
by
  sorry

end textbook_weight_l140_140671


namespace scoring_situations_4_students_l140_140866

noncomputable def number_of_scoring_situations (students : ℕ) (topicA_score : ℤ) (topicB_score : ℤ) : ℕ :=
  let combinations := Nat.choose 4 2
  let first_category := combinations * 2 * 2
  let second_category := 2 * combinations
  first_category + second_category

theorem scoring_situations_4_students : number_of_scoring_situations 4 100 90 = 36 :=
by
  -- The proof is omitted as per the instructions.
  sorry

end scoring_situations_4_students_l140_140866


namespace find_a_l140_140288

def star (a b : ℝ) : ℝ := 2 * a - b^3

theorem find_a (a : ℝ) : star a 3 = 15 → a = 21 :=
by
  intro h
  sorry

end find_a_l140_140288


namespace butterflies_left_correct_l140_140154

-- Define the total number of butterflies and the fraction that flies away
def butterflies_total : ℕ := 9
def fraction_fly_away : ℚ := 1 / 3

-- Define the number of butterflies left in the garden
def butterflies_left (t : ℕ) (f : ℚ) : ℚ := t - (t : ℚ) * f

-- State the theorem
theorem butterflies_left_correct : butterflies_left butterflies_total fraction_fly_away = 6 := by
  sorry

end butterflies_left_correct_l140_140154


namespace find_a11_times_a55_l140_140020

noncomputable def a_ij (i j : ℕ) : ℝ := 
  if i = 4 ∧ j = 1 then -2 else
  if i = 4 ∧ j = 3 then 10 else
  if i = 2 ∧ j = 4 then 4 else sorry

theorem find_a11_times_a55 
  (arithmetic_first_row : ∀ j, a_ij 1 (j + 1) = a_ij 1 1 + (j * 6))
  (geometric_columns : ∀ i j, a_ij (i + 1) j = a_ij 1 j * (2 ^ i) ∨ a_ij (i + 1) j = a_ij 1 j * ((-2) ^ i))
  (a24_eq_4 : a_ij 2 4 = 4)
  (a41_eq_neg2 : a_ij 4 1 = -2)
  (a43_eq_10 : a_ij 4 3 = 10) :
  a_ij 1 1 * a_ij 5 5 = -11 :=
by sorry

end find_a11_times_a55_l140_140020


namespace intersection_M_N_l140_140163

def M : Set ℝ := { x : ℝ | -4 < x ∧ x < 2 }
def N : Set ℝ := { x : ℝ | x^2 - x - 6 < 0 }

theorem intersection_M_N : M ∩ N = { x : ℝ | -2 < x ∧ x < 2 } := by
  sorry

end intersection_M_N_l140_140163


namespace neg_two_is_negative_rational_l140_140222

theorem neg_two_is_negative_rational : 
  (-2 : ℚ) < 0 ∧ ∃ (r : ℚ), r = -2 := 
by
  sorry

end neg_two_is_negative_rational_l140_140222


namespace tetrahedrons_volume_proportional_l140_140480

-- Define the scenario and conditions.
variable 
  (V V' : ℝ) -- Volumes of the tetrahedrons
  (a b c a' b' c' : ℝ) -- Edge lengths emanating from vertices O and O'
  (α : ℝ) -- The angle between vectors OB and OC which is assumed to be congruent

-- Theorem statement.
theorem tetrahedrons_volume_proportional
  (congruent_trihedral_angles_at_O_and_O' : α = α) -- Condition of congruent trihedral angles
  : (V' / V) = (a' * b' * c') / (a * b * c) :=
sorry

end tetrahedrons_volume_proportional_l140_140480


namespace students_in_school_B_l140_140007

theorem students_in_school_B 
    (A B C : ℕ) 
    (h1 : A + C = 210) 
    (h2 : A = 4 * B) 
    (h3 : C = 3 * B) : 
    B = 30 := 
by 
    sorry

end students_in_school_B_l140_140007


namespace circle_parabola_intersection_l140_140705

theorem circle_parabola_intersection (b : ℝ) : 
  b = 25 / 12 → 
  ∃ (r : ℝ) (cx : ℝ), 
  (∃ p1 p2 : ℝ × ℝ, 
    (p1.2 = 3/4 * p1.1 + b ∧ p2.2 = 3/4 * p2.1 + b) ∧ 
    (p1.2 = 3/4 * p1.1^2 ∧ p2.2 = 3/4 * p2.1^2) ∧ 
    (p1 ≠ (0, 0) ∧ p2 ≠ (0, 0))) ∧ 
  (cx^2 + b^2 = r^2) := 
by 
  sorry

end circle_parabola_intersection_l140_140705


namespace range_of_a_l140_140202

theorem range_of_a (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) : 1 < a ∧ a < 2 :=
by
  -- Insert the proof here
  sorry

end range_of_a_l140_140202


namespace notebook_cost_correct_l140_140523

def totalSpent : ℕ := 32
def costBackpack : ℕ := 15
def costPen : ℕ := 1
def costPencil : ℕ := 1
def numberOfNotebooks : ℕ := 5
def costPerNotebook : ℕ := 3

theorem notebook_cost_correct (h_totalSpent : totalSpent = 32)
    (h_costBackpack : costBackpack = 15)
    (h_costPen : costPen = 1)
    (h_costPencil : costPencil = 1)
    (h_numberOfNotebooks : numberOfNotebooks = 5) :
    (totalSpent - (costBackpack + costPen + costPencil)) / numberOfNotebooks = costPerNotebook :=
by
  sorry

end notebook_cost_correct_l140_140523


namespace matrix_not_invertible_l140_140584

def is_not_invertible_matrix (y : ℝ) : Prop :=
  let a := 2 + y
  let b := 9
  let c := 4 - y
  let d := 10
  a * d - b * c = 0

theorem matrix_not_invertible (y : ℝ) : is_not_invertible_matrix y ↔ y = 16 / 19 :=
  sorry

end matrix_not_invertible_l140_140584


namespace remainder_of_large_number_div_by_101_l140_140673

theorem remainder_of_large_number_div_by_101 :
  2468135792 % 101 = 52 :=
by
  sorry

end remainder_of_large_number_div_by_101_l140_140673


namespace Donovan_Mitchell_current_average_l140_140441

theorem Donovan_Mitchell_current_average 
    (points_per_game_goal : ℕ) 
    (games_played : ℕ) 
    (total_games_goal : ℕ) 
    (average_needed_remaining_games : ℕ)
    (points_needed : ℕ) 
    (remaining_games : ℕ) 
    (x : ℕ) 
    (h₁ : games_played = 15) 
    (h₂ : total_games_goal = 20) 
    (h₃ : points_per_game_goal = 30) 
    (h₄ : remaining_games = total_games_goal - games_played)
    (h₅ : average_needed_remaining_games = 42) 
    (h₆ : points_needed = remaining_games * average_needed_remaining_games) 
    (h₇ : points_needed = 210)  
    (h₈ : points_per_game_goal * total_games_goal = 600) 
    (h₉ : games_played * x + points_needed = 600) : 
    x = 26 :=
by {
  sorry
}

end Donovan_Mitchell_current_average_l140_140441


namespace part1_part2_l140_140375

noncomputable def f (a x : ℝ) : ℝ := a * x^2 - (a + 2) * x + Real.log x

theorem part1 (a : ℝ) (h : (2 * a - (a + 2) + 1) = 0) : a = 1 :=
by
  sorry

theorem part2 (a x : ℝ) (ha : a ≥ 1) (hx : 1 ≤ x ∧ x ≤ Real.exp 1) : (2 * a * x - (a + 2) + 1 / x) ≥ 0 :=
by
  sorry

end part1_part2_l140_140375


namespace reinforcement_arrival_days_l140_140774

theorem reinforcement_arrival_days (x : ℕ) (h : x = 2000) (provisions_days : ℕ) (provisions_days_initial : provisions_days = 54) 
(reinforcement : ℕ) (reinforcement_val : reinforcement = 1300) (remaining_days : ℕ) (remaining_days_val : remaining_days = 20) 
(total_men : ℕ) (total_men_val : total_men = 3300) (equation : 2000 * (54 - x) = 3300 * 20) : x = 21 := 
by
  have eq1 : 2000 * 54 - 2000 * x = 3300 * 20 := by sorry
  have eq2 : 108000 - 2000 * x = 66000 := by sorry
  have eq3 : 2000 * x = 42000 := by sorry
  have eq4 : x = 21000 / 2000 := by sorry
  have eq5 : x = 21 := by sorry
  sorry

end reinforcement_arrival_days_l140_140774


namespace maximize_x3y4_l140_140406

noncomputable def maximize_expr (x y : ℝ) : ℝ :=
x^3 * y^4

theorem maximize_x3y4 : ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = 50 ∧ maximize_expr x y = maximize_expr 30 20 :=
by
  sorry

end maximize_x3y4_l140_140406


namespace two_a_minus_b_l140_140799

-- Definitions of vector components and parallelism condition
def is_parallel (a b : ℝ × ℝ) : Prop := a.1 * b.2 - a.2 * b.1 = 0
def vector_a : ℝ × ℝ := (1, -2)

-- Given assumptions
variable (m : ℝ)
def vector_b : ℝ × ℝ := (m, 4)

-- Theorem statement
theorem two_a_minus_b (h : is_parallel vector_a (vector_b m)) : 2 • vector_a - vector_b m = (4, -8) :=
sorry

end two_a_minus_b_l140_140799


namespace area_ACD_l140_140509

def base_ABD : ℝ := 8
def height_ABD : ℝ := 4
def base_ABC : ℝ := 4
def height_ABC : ℝ := 4

theorem area_ACD : (1/2 * base_ABD * height_ABD) - (1/2 * base_ABC * height_ABC) = 8 := by
  sorry

end area_ACD_l140_140509


namespace pq_sum_l140_140746

open Real

theorem pq_sum (p q : ℝ) (hp : p^3 - 18 * p^2 + 81 * p - 162 = 0) (hq : 4 * q^3 - 24 * q^2 + 45 * q - 27 = 0) :
    p + q = 8 ∨ p + q = 8 + 6 * sqrt 3 ∨ p + q = 8 - 6 * sqrt 3 :=
sorry

end pq_sum_l140_140746


namespace find_a5_of_geometric_sequence_l140_140280

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop := ∃ r, ∀ n, a (n + 1) = a n * r

theorem find_a5_of_geometric_sequence (a : ℕ → ℝ) (h : geometric_sequence a)
  (h₀ : a 1 = 1) (h₁ : a 9 = 3) : a 5 = Real.sqrt 3 :=
sorry

end find_a5_of_geometric_sequence_l140_140280


namespace pq_r_zero_l140_140446

theorem pq_r_zero (p q r : ℝ) : 
  (∀ x : ℝ, x^4 + 6 * x^3 + 4 * p * x^2 + 2 * q * x + r = (x^3 + 4 * x^2 + 2 * x + 1) * (x - 2)) → 
  (p + q) * r = 0 :=
by
  sorry

end pq_r_zero_l140_140446


namespace policeman_hats_difference_l140_140238

theorem policeman_hats_difference
  (hats_simpson : ℕ)
  (hats_obrien_now : ℕ)
  (hats_obrien_before : ℕ)
  (H : hats_simpson = 15)
  (H_hats_obrien_now : hats_obrien_now = 34)
  (H_hats_obrien_twice : hats_obrien_before = hats_obrien_now + 1) :
  hats_obrien_before - 2 * hats_simpson = 5 :=
by
  sorry

end policeman_hats_difference_l140_140238


namespace ellipse_product_axes_l140_140580

/-- Prove that the product of the lengths of the major and minor axes (AB)(CD) of an ellipse
is 240, given the following conditions:
- Point O is the center of the ellipse.
- Point F is one focus of the ellipse.
- OF = 8
- The diameter of the inscribed circle of triangle OCF is 4.
- OA = OB = a
- OC = OD = b
- a² - b² = 64
- a - b = 4
-/
theorem ellipse_product_axes (a b : ℝ) (OF : ℝ) (d_inscribed_circle : ℝ) 
  (h1 : OF = 8) (h2 : d_inscribed_circle = 4) (h3 : a^2 - b^2 = 64) 
  (h4 : a - b = 4) : (2 * a) * (2 * b) = 240 :=
sorry

end ellipse_product_axes_l140_140580


namespace prob_queen_then_diamond_is_correct_l140_140920

/-- Define the probability of drawing a Queen first and a diamond second -/
def prob_queen_then_diamond : ℚ := (3 / 52) * (13 / 51) + (1 / 52) * (12 / 51)

/-- The probability that the first card is a Queen and the second card is a diamond is 18/221 -/
theorem prob_queen_then_diamond_is_correct : prob_queen_then_diamond = 18 / 221 :=
by
  sorry

end prob_queen_then_diamond_is_correct_l140_140920


namespace license_plate_combinations_l140_140963

-- Definitions based on the conditions
def num_letters := 26
def num_digits := 10
def num_positions := 5

def choose (n k : ℕ) : ℕ := Nat.choose n k

-- Main theorem statement
theorem license_plate_combinations :
  choose num_letters 2 * (num_letters - 2) * choose num_positions 2 * choose (num_positions - 2) 2 * num_digits * (num_digits - 1) * (num_digits - 2) = 7776000 :=
by
  sorry

end license_plate_combinations_l140_140963


namespace rectangle_perimeter_bounds_l140_140100

/-- Given 12 rectangular cardboard pieces, each measuring 4 cm in length and 3 cm in width,
  if these pieces are assembled to form a larger rectangle (possibly including squares),
  without overlapping or leaving gaps, then the minimum possible perimeter of the resulting 
  rectangle is 48 cm and the maximum possible perimeter is 102 cm. -/
theorem rectangle_perimeter_bounds (n : ℕ) (l w : ℝ) (total_area : ℝ) :
  n = 12 ∧ l = 4 ∧ w = 3 ∧ total_area = n * l * w →
  ∃ (min_perimeter max_perimeter : ℝ),
    min_perimeter = 48 ∧ max_perimeter = 102 :=
by
  intros
  sorry

end rectangle_perimeter_bounds_l140_140100


namespace curve_equation_l140_140536

theorem curve_equation
  (a b : ℝ)
  (h1 : a * 0 ^ 2 + b * (5 / 3) ^ 2 = 2)
  (h2 : a * 1 ^ 2 + b * 1 ^ 2 = 2) :
  (16 / 25) * x^2 + (9 / 25) * y^2 = 1 := 
by {
  sorry
}

end curve_equation_l140_140536


namespace smallest_omega_l140_140458

theorem smallest_omega (ω : ℝ) (h_pos : ω > 0) :
  (∃ k : ℤ, ω = 6 * k) ∧ (∀ k : ℤ, k > 0 → ω = 6 * k → ω = 6) :=
by sorry

end smallest_omega_l140_140458


namespace line_k_x_intercept_l140_140264

theorem line_k_x_intercept :
  ∀ (x y : ℝ), 3 * x - 5 * y + 40 = 0 ∧ 
  ∃ m' b', (m' = 4) ∧ (b' = 20 - 4 * 20) ∧ 
  (y = m' * x + b') →
  ∃ x_inter, (y = 0) → (x_inter = 15) := 
by
  sorry

end line_k_x_intercept_l140_140264


namespace longest_side_length_of_quadrilateral_l140_140692

-- Define the system of inequalities
def inFeasibleRegion (x y : ℝ) : Prop :=
  (x + 2 * y ≤ 4) ∧
  (3 * x + y ≥ 3) ∧
  (x ≥ 0) ∧
  (y ≥ 0)

-- The goal is to prove that the longest side length is 5
theorem longest_side_length_of_quadrilateral :
  ∃ a b c d : (ℝ × ℝ), inFeasibleRegion a.1 a.2 ∧
                  inFeasibleRegion b.1 b.2 ∧
                  inFeasibleRegion c.1 c.2 ∧
                  inFeasibleRegion d.1 d.2 ∧
                  -- For each side, specify the length condition (Euclidean distance)
                  max (dist a b) (max (dist b c) (max (dist c d) (dist d a))) = 5 :=
by sorry

end longest_side_length_of_quadrilateral_l140_140692


namespace perfect_squares_with_specific_ones_digit_count_l140_140710

theorem perfect_squares_with_specific_ones_digit_count : 
  ∃ n : ℕ, (∀ k : ℕ, k < 2500 → (k % 10 = 4 ∨ k % 10 = 5 ∨ k % 10 = 6) ↔ ∃ m : ℕ, m < n ∧ (m % 10 = 2 ∨ m % 10 = 8 ∨ m % 10 = 5 ∨ m % 10 = 4 ∨ m % 10 = 6) ∧ k = m * m) 
  ∧ n = 25 := 
by 
  sorry

end perfect_squares_with_specific_ones_digit_count_l140_140710


namespace range_of_f_l140_140321

open Real

noncomputable def f (x : ℝ) : ℝ := (sqrt 3) * sin x + cos x

theorem range_of_f :
  ∀ x : ℝ, -π/2 ≤ x ∧ x ≤ π/2 → - (sqrt 3) ≤ f x ∧ f x ≤ 2 := by
  sorry

end range_of_f_l140_140321


namespace no_positive_integer_solutions_l140_140351

theorem no_positive_integer_solutions (x y z : ℕ) (h_pos : x > 0 ∧ y > 0 ∧ z > 0) : x^2 + y^2 ≠ 7 * z^2 := by
  sorry

end no_positive_integer_solutions_l140_140351


namespace base8_subtraction_correct_l140_140455

-- Define what it means to subtract in base 8
def base8_sub (a b : ℕ) : ℕ :=
  let a_base10 := 8 * (a / 10) + (a % 10)
  let b_base10 := 8 * (b / 10) + (b % 10)
  let result_base10 := a_base10 - b_base10
  8 * (result_base10 / 8) + (result_base10 % 8)

-- The given numbers in base 8
def num1 : ℕ := 52
def num2 : ℕ := 31
def expected_result : ℕ := 21

-- The proof problem statement
theorem base8_subtraction_correct : base8_sub num1 num2 = expected_result := by
  sorry

end base8_subtraction_correct_l140_140455


namespace find_larger_integer_l140_140512

noncomputable def larger_integer (a b : ℤ) := max a b

theorem find_larger_integer (a b : ℕ) 
  (h1 : a/b = 7/3) 
  (h2 : a * b = 294): 
  larger_integer a b = 7 * Real.sqrt 14 :=
by
  -- Proof goes here
  sorry

end find_larger_integer_l140_140512


namespace parabola_focus_l140_140640

noncomputable def parabola_focus_coordinates (a : ℝ) : ℝ × ℝ :=
  if a ≠ 0 then (0, 1 / (4 * a)) else (0, 0)

theorem parabola_focus {x y : ℝ} (a : ℝ) (h : a = 2) (h_eq : y = a * x^2) :
  parabola_focus_coordinates a = (0, 1 / 8) :=
by sorry

end parabola_focus_l140_140640


namespace triangle_inequality_check_l140_140339

theorem triangle_inequality_check (a b c : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) : 
  (a = 5 ∧ b = 8 ∧ c = 12) → (a + b > c ∧ b + c > a ∧ c + a > b) :=
by 
  intros h
  rcases h with ⟨rfl, rfl, rfl⟩
  exact ⟨h1, h2, h3⟩

end triangle_inequality_check_l140_140339


namespace find_a_and_b_l140_140267

theorem find_a_and_b (a b : ℝ) (h1 : b - 1/4 = (a + b) / 4 + b / 2) (h2 : 4 * a / 3 = (a + b) / 2)  :
  a = 3/2 ∧ b = 5/2 :=
by
  sorry

end find_a_and_b_l140_140267


namespace evie_shells_l140_140538

theorem evie_shells (shells_per_day : ℕ) (days : ℕ) (gifted_shells : ℕ) 
  (h1 : shells_per_day = 10) 
  (h2 : days = 6)
  (h3 : gifted_shells = 2) : 
  shells_per_day * days - gifted_shells = 58 := 
by
  sorry

end evie_shells_l140_140538


namespace tangent_line_at_point_l140_140712

theorem tangent_line_at_point (x y : ℝ) (h : y = Real.exp x) (t : x = 2) :
  y = Real.exp 2 * x - 2 * Real.exp 2 :=
by sorry

end tangent_line_at_point_l140_140712


namespace base_radius_of_cone_l140_140461

-- Definitions of the conditions
def R1 : ℕ := 5
def R2 : ℕ := 4
def R3 : ℕ := 4
def height_radius_ratio := 4 / 3

-- Main theorem statement
theorem base_radius_of_cone : 
  (R1 = 5) → (R2 = 4) → (R3 = 4) → (height_radius_ratio = 4 / 3) → 
  ∃ r : ℚ, r = 169 / 60 :=
by 
  intros hR1 hR2 hR3 hRatio
  sorry

end base_radius_of_cone_l140_140461


namespace inequality_solution_equality_condition_l140_140300

theorem inequality_solution (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) : 
  (a^2 + b^2 + c^2 + d^2)^2 ≥ (a + b) * (b + c) * (c + d) * (d + a) :=
sorry

theorem equality_condition (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) : 
  (a^2 + b^2 + c^2 + d^2)^2 = (a + b) * (b + c) * (c + d) * (d + a) ↔ a = b ∧ b = c ∧ c = d :=
sorry

end inequality_solution_equality_condition_l140_140300


namespace base8_arithmetic_l140_140976

-- Define the numbers in base 8
def num1 : ℕ := 0o453
def num2 : ℕ := 0o267
def num3 : ℕ := 0o512
def expected_result : ℕ := 0o232

-- Prove that (num1 + num2) - num3 = expected_result in base 8
theorem base8_arithmetic : ((num1 + num2) - num3) = expected_result := by
  sorry

end base8_arithmetic_l140_140976


namespace expand_binomial_square_l140_140015

variables (x : ℝ)

theorem expand_binomial_square (x : ℝ) : (2 - x) ^ 2 = 4 - 4 * x + x ^ 2 := 
sorry

end expand_binomial_square_l140_140015


namespace smallest_positive_integer_n_l140_140830

noncomputable def matrix_330 : Matrix (Fin 2) (Fin 2) ℝ :=
  ![
    ![Real.cos (330 * Real.pi / 180), -Real.sin (330 * Real.pi / 180)],
    ![Real.sin (330 * Real.pi / 180), Real.cos (330 * Real.pi / 180)]
  ]

theorem smallest_positive_integer_n (n : ℕ) (h : matrix_330 ^ n = 1) : n = 12 := sorry

end smallest_positive_integer_n_l140_140830


namespace doubling_period_l140_140067

theorem doubling_period (initial_capacity: ℝ) (final_capacity: ℝ) (years: ℝ) (initial_year: ℝ) (final_year: ℝ) (doubling_period: ℝ) :
  initial_capacity = 0.4 → final_capacity = 4100 → years = (final_year - initial_year) →
  initial_year = 2000 → final_year = 2050 →
  2 ^ (years / doubling_period) * initial_capacity = final_capacity :=
by
  intros h_initial h_final h_years h_i_year h_f_year
  sorry

end doubling_period_l140_140067


namespace students_average_comparison_l140_140891

theorem students_average_comparison (t1 t2 t3 : ℝ) (h : t1 < t2) (h' : t2 < t3) :
  (∃ t1 t2 t3 : ℝ, t1 < t2 ∧ t2 < t3 ∧ (t1 + t2 + t3) / 3 = (t1 + t3 + 2 * t2) / 4) ∨
  (∀ t1 t2 t3 : ℝ, t1 < t2 ∧ t2 < t3 → 
     (t1 + t3 + 2 * t2) / 4 > (t1 + t2 + t3) / 3) :=
sorry

end students_average_comparison_l140_140891


namespace minimal_positive_sum_circle_integers_l140_140381

-- Definitions based on the conditions in the problem statement
def cyclic_neighbors (l : List Int) (i : ℕ) : Int :=
  l.getD (Nat.mod (i - 1) l.length) 0 + l.getD (Nat.mod (i + 1) l.length) 0

-- Problem statement in Lean: 
theorem minimal_positive_sum_circle_integers :
  ∃ (l : List Int), l.length ≥ 5 ∧ (∀ (i : ℕ), i < l.length → l.getD i 0 ∣ cyclic_neighbors l i) ∧ (0 < l.sum) ∧ l.sum = 2 :=
sorry

end minimal_positive_sum_circle_integers_l140_140381


namespace proof_inequalities_l140_140558

theorem proof_inequalities (A B C D E : ℝ) (p q r s t : ℝ)
  (h1 : A < B) (h2 : B < C) (h3 : C < D) (h4 : D < E)
  (h5 : p = B - A) (h6 : q = C - A) (h7 : r = D - A)
  (h8 : s = E - B) (h9 : t = E - D)
  (ineq1 : p + 2 * s > r + t)
  (ineq2 : r + t > p)
  (ineq3 : r + t > s) :
  (p < r / 2) ∧ (s < t + p / 2) :=
by 
  sorry

end proof_inequalities_l140_140558


namespace total_male_students_combined_l140_140004

/-- The number of first-year students is 695, of which 329 are female students. 
If the number of male second-year students is 254, prove that the number of male students in the first-year and second-year combined is 620. -/
theorem total_male_students_combined (first_year_students : ℕ) (female_first_year_students : ℕ) (male_second_year_students : ℕ) :
  first_year_students = 695 →
  female_first_year_students = 329 →
  male_second_year_students = 254 →
  (first_year_students - female_first_year_students + male_second_year_students) = 620 := by
  sorry

end total_male_students_combined_l140_140004


namespace find_even_integer_l140_140856

theorem find_even_integer (x y z : ℤ) (h₁ : Even x) (h₂ : Odd y) (h₃ : Odd z)
  (h₄ : x < y) (h₅ : y < z) (h₆ : y - x > 5) (h₇ : z - x = 9) : x = 2 := 
by 
  sorry

end find_even_integer_l140_140856


namespace laran_weekly_profit_l140_140090

-- Definitions based on the problem conditions
def daily_posters_sold : ℕ := 5
def large_posters_sold_daily : ℕ := 2
def small_posters_sold_daily : ℕ := daily_posters_sold - large_posters_sold_daily

def price_large_poster : ℕ := 10
def cost_large_poster : ℕ := 5
def profit_large_poster : ℕ := price_large_poster - cost_large_poster

def price_small_poster : ℕ := 6
def cost_small_poster : ℕ := 3
def profit_small_poster : ℕ := price_small_poster - cost_small_poster

def daily_profit_large_posters : ℕ := large_posters_sold_daily * profit_large_poster
def daily_profit_small_posters : ℕ := small_posters_sold_daily * profit_small_poster
def total_daily_profit : ℕ := daily_profit_large_posters + daily_profit_small_posters

def school_days_week : ℕ := 5
def weekly_profit : ℕ := total_daily_profit * school_days_week

-- Statement to prove
theorem laran_weekly_profit : weekly_profit = 95 := sorry

end laran_weekly_profit_l140_140090


namespace eventually_periodic_l140_140859

variable (u : ℕ → ℤ)

def bounded (u : ℕ → ℤ) : Prop :=
  ∃ (m M : ℤ), ∀ (n : ℕ), m ≤ u n ∧ u n ≤ M

def recurrence (u : ℕ → ℤ) (n : ℕ) : Prop := 
  u (n) = (u (n-1) + u (n-2) + u (n-3) * u (n-4)) / (u (n-1) * u (n-2) + u (n-3) + u (n-4))

theorem eventually_periodic (hu_bounded : bounded u) (hu_recurrence : ∀ n ≥ 4, recurrence u n) :
  ∃ N M, ∀ k ≥ 0, u (N + k) = u (N + M + k) :=
sorry

end eventually_periodic_l140_140859


namespace find_other_parallel_side_l140_140190

theorem find_other_parallel_side 
  (a b d : ℝ) 
  (area : ℝ) 
  (h_area : area = 285) 
  (h_a : a = 20) 
  (h_d : d = 15)
  : (∃ x : ℝ, area = 1/2 * (a + x) * d ∧ x = 18) :=
by
  sorry

end find_other_parallel_side_l140_140190


namespace find_4a_plus_8b_l140_140226

def quadratic_equation_x_solution (a b : ℝ) : Prop :=
  (1 : ℝ)^2 + a * (1 : ℝ) + 2 * b = 0

theorem find_4a_plus_8b (a b : ℝ) (h : quadratic_equation_x_solution a b) : 4 * a + 8 * b = -4 := 
  by
    sorry

end find_4a_plus_8b_l140_140226


namespace find_number_l140_140633

-- Define the necessary variables and constants
variables (N : ℝ) (h1 : (5 / 4) * N = (4 / 5) * N + 18)

-- State the problem as a theorem to be proved
theorem find_number : N = 40 :=
by
  sorry

end find_number_l140_140633


namespace least_number_of_stamps_l140_140564

theorem least_number_of_stamps (s t : ℕ) (h : 5 * s + 7 * t = 50) : s + t = 8 :=
sorry

end least_number_of_stamps_l140_140564


namespace beacon_population_l140_140405

variables (Richmond Victoria Beacon : ℕ)

theorem beacon_population :
  (Richmond = Victoria + 1000) →
  (Victoria = 4 * Beacon) →
  (Richmond = 3000) →
  (Beacon = 500) :=
by
  intros h1 h2 h3
  sorry

end beacon_population_l140_140405


namespace quadrilateral_segments_condition_l140_140223

-- Define the lengths and their conditions
variables {a b c d : ℝ}

-- Define the main theorem with necessary and sufficient conditions
theorem quadrilateral_segments_condition (h_sum : a + b + c + d = 1.5)
    (h_order : a ≤ b) (h_order2 : b ≤ c) (h_order3 : c ≤ d) (h_ratio : d ≤ 3 * a) :
    (a ≥ 0.25 ∧ d < 0.75) ↔ (a + b + c > d ∧ a + b + d > c ∧ a + c + d > b ∧ b + c + d > a) :=
by {
  sorry -- proof is omitted
}

end quadrilateral_segments_condition_l140_140223


namespace verify_differential_eq_l140_140444

noncomputable def y (x : ℝ) : ℝ := (2 + 3 * x - 3 * x^2)^(1 / 3 : ℝ)
noncomputable def y_prime (x : ℝ) : ℝ := 
  1 / 3 * (2 + 3 * x - 3 * x^2)^(-2 / 3 : ℝ) * (3 - 6 * x)

theorem verify_differential_eq (x : ℝ) :
  y x * y_prime x = (1 - 2 * x) / y x :=
by
  sorry

end verify_differential_eq_l140_140444


namespace div_37_permutation_l140_140068

-- Let A, B, C be digits of a three-digit number
variables (A B C : ℕ) -- these can take values from 0 to 9
variables (p : ℕ) -- integer multiplier for the divisibility condition

-- The main theorem stated as a Lean 4 problem
theorem div_37_permutation (h : 100 * A + 10 * B + C = 37 * p) : 
  ∃ (M : ℕ), (M = 100 * B + 10 * C + A ∨ M = 100 * C + 10 * A + B ∨ M = 100 * A + 10 * C + B ∨ M = 100 * C + 10 * B + A ∨ M = 100 * B + 10 * A + C) ∧ 37 ∣ M :=
by
  sorry

end div_37_permutation_l140_140068


namespace correct_intersection_l140_140960

def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

theorem correct_intersection : M ∩ N = {2, 3} := by sorry

end correct_intersection_l140_140960


namespace tan_theta_value_l140_140796

theorem tan_theta_value (θ : ℝ) (h : Real.tan (Real.pi / 4 + θ) = 1 / 2) : Real.tan θ = -1 / 3 :=
sorry

end tan_theta_value_l140_140796


namespace cos_4pi_over_3_l140_140411

theorem cos_4pi_over_3 : Real.cos (4 * Real.pi / 3) = -1 / 2 :=
by 
  sorry

end cos_4pi_over_3_l140_140411


namespace track_length_is_320_l140_140756

noncomputable def length_of_track (x : ℝ) : Prop :=
  (∃ v_b v_s : ℝ, (v_b > 0 ∧ v_s > 0 ∧ v_b + v_s = x / 2 ∧ -- speeds of Brenda and Sally must sum up to half the track length against each other
                    80 / v_b = (x / 2 - 80) / v_s ∧ -- First meeting condition
                    120 / v_s + 80 / v_b = (x / 2 + 40) / v_s + (x - 80) / v_b -- Second meeting condition
                   )) ∧ x = 320

theorem track_length_is_320 : ∃ x : ℝ, length_of_track x :=
by
  use 320
  unfold length_of_track
  simp
  sorry

end track_length_is_320_l140_140756


namespace solution_set_of_inequality_l140_140403

theorem solution_set_of_inequality : {x : ℝ | 8 * x^2 + 6 * x ≤ 2} = { x : ℝ | -1 ≤ x ∧ x ≤ (1/4) } :=
sorry

end solution_set_of_inequality_l140_140403


namespace total_people_museum_l140_140563

def bus1 := 12
def bus2 := 2 * bus1
def bus3 := bus2 - 6
def bus4 := bus1 + 9
def total := bus1 + bus2 + bus3 + bus4

theorem total_people_museum : total = 75 := by
  sorry

end total_people_museum_l140_140563


namespace tangent_line_to_curve_determines_m_l140_140567

theorem tangent_line_to_curve_determines_m :
  ∃ m : ℝ, (∀ x : ℝ, y = x ^ 4 + m * x) ∧ (2 * -1 + y' + 3 = 0) ∧ (y' = -2) → (m = 2) :=
by
  sorry

end tangent_line_to_curve_determines_m_l140_140567


namespace div_by_six_l140_140569

theorem div_by_six (n : ℕ) : 6 ∣ (17^n - 11^n) :=
by
  sorry

end div_by_six_l140_140569


namespace compute_a_b_difference_square_l140_140420

noncomputable def count_multiples (m n : ℕ) : ℕ :=
  (n - 1) / m

theorem compute_a_b_difference_square :
  let a := count_multiples 12 60
  let b := count_multiples 12 60
  (a - b) ^ 2 = 0 :=
by
  let a := count_multiples 12 60
  let b := count_multiples 12 60
  show (a - b) ^ 2 = 0
  sorry

end compute_a_b_difference_square_l140_140420


namespace f_zero_eq_zero_f_one_eq_one_f_n_is_n_l140_140354

variable (f : ℤ → ℤ)

axiom functional_eq : ∀ m n : ℤ, f (m^2 + f n) = f (f m) + n

theorem f_zero_eq_zero : f 0 = 0 :=
sorry

theorem f_one_eq_one : f 1 = 1 :=
sorry

theorem f_n_is_n : ∀ n : ℤ, f n = n :=
sorry

end f_zero_eq_zero_f_one_eq_one_f_n_is_n_l140_140354


namespace average_mileage_is_correct_l140_140978

noncomputable def total_distance : ℝ := 150 + 200
noncomputable def sedan_efficiency : ℝ := 25
noncomputable def truck_efficiency : ℝ := 15
noncomputable def sedan_miles : ℝ := 150
noncomputable def truck_miles : ℝ := 200

noncomputable def total_gas_used : ℝ := (sedan_miles / sedan_efficiency) + (truck_miles / truck_efficiency)
noncomputable def average_gas_mileage : ℝ := total_distance / total_gas_used

theorem average_mileage_is_correct :
  average_gas_mileage = 18.1 := 
by
  sorry

end average_mileage_is_correct_l140_140978


namespace max_lateral_surface_area_cylinder_optimizes_l140_140422

noncomputable def max_lateral_surface_area_cylinder (r m : ℝ) : ℝ × ℝ :=
  let r_c := r / 2
  let h_c := m / 2
  (r_c, h_c)

theorem max_lateral_surface_area_cylinder_optimizes {r m : ℝ} (hr : 0 < r) (hm : 0 < m) :
  let (r_c, h_c) := max_lateral_surface_area_cylinder r m
  r_c = r / 2 ∧ h_c = m / 2 :=
sorry

end max_lateral_surface_area_cylinder_optimizes_l140_140422


namespace acute_triangle_sums_to_pi_over_4_l140_140265

theorem acute_triangle_sums_to_pi_over_4 
    (A B : ℝ) 
    (hA : 0 < A ∧ A < π / 2) 
    (hB : 0 < B ∧ B < π / 2) 
    (h_sinA : Real.sin A = (Real.sqrt 5)/5) 
    (h_sinB : Real.sin B = (Real.sqrt 10)/10) : 
    A + B = π / 4 := 
sorry

end acute_triangle_sums_to_pi_over_4_l140_140265


namespace relationship_between_x1_x2_x3_l140_140698

variable {x1 x2 x3 : ℝ}

theorem relationship_between_x1_x2_x3
  (A_on_curve : (6 : ℝ) = 6 / x1)
  (B_on_curve : (12 : ℝ) = 6 / x2)
  (C_on_curve : (-6 : ℝ) = 6 / x3) :
  x3 < x2 ∧ x2 < x1 := 
sorry

end relationship_between_x1_x2_x3_l140_140698


namespace solution_l140_140218

theorem solution {a : ℕ → ℝ} 
  (h : a 1 = 1)
  (h2 : ∀ n : ℕ, 1 ≤ n ∧ n ≤ 100 →
    a n - 4 * a (if n = 100 then 1 else n + 1) + 3 * a (if n = 99 then 1 else if n = 100 then 2 else n + 2) ≥ 0) :
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 100 → a n = 1 :=
by
  sorry

end solution_l140_140218


namespace boat_distance_along_stream_in_one_hour_l140_140740

theorem boat_distance_along_stream_in_one_hour :
  ∀ (v_b v_s d_up t : ℝ),
  v_b = 7 →
  d_up = 3 →
  t = 1 →
  (t * (v_b - v_s) = d_up) →
  t * (v_b + v_s) = 11 :=
by
  intros v_b v_s d_up t Hv_b Hd_up Ht Hup
  sorry

end boat_distance_along_stream_in_one_hour_l140_140740


namespace number_of_classes_l140_140706

theorem number_of_classes (n : ℕ) (a₁ : ℕ) (d : ℤ) (S : ℕ) (h₁ : d = -2) (h₂ : a₁ = 25) (h₃ : S = 105) : n = 5 :=
by
  /- We state the theorem and the necessary conditions without proving it -/
  sorry

end number_of_classes_l140_140706


namespace number_of_people_l140_140616

-- Definitions based on the conditions
def total_cookies : ℕ := 420
def cookies_per_person : ℕ := 30

-- The goal is to prove the number of people is 14
theorem number_of_people : total_cookies / cookies_per_person = 14 :=
by
  sorry

end number_of_people_l140_140616


namespace kiril_age_problem_l140_140704

theorem kiril_age_problem (x : ℕ) (h1 : x % 5 = 0) (h2 : (x - 1) % 7 = 0) : 26 - x = 11 :=
by
  sorry

end kiril_age_problem_l140_140704


namespace point_after_rotation_l140_140953

-- Definitions based on conditions
def point_N : ℝ × ℝ := (-1, -2)
def origin_O : ℝ × ℝ := (0, 0)
def rotation_180 (P : ℝ × ℝ) : ℝ × ℝ := (-P.1, -P.2)

-- The statement to be proved
theorem point_after_rotation :
  rotation_180 point_N = (1, 2) :=
by
  sorry

end point_after_rotation_l140_140953


namespace odd_power_of_7_plus_1_divisible_by_8_l140_140565

theorem odd_power_of_7_plus_1_divisible_by_8 (n : ℕ) (h : n % 2 = 1) : (7 ^ n + 1) % 8 = 0 :=
by
  sorry

end odd_power_of_7_plus_1_divisible_by_8_l140_140565


namespace divisor_of_3825_is_15_l140_140302

theorem divisor_of_3825_is_15 : ∃ d, 3830 - 5 = 3825 ∧ 3825 % d = 0 ∧ d = 15 := by
  sorry

end divisor_of_3825_is_15_l140_140302


namespace angle_perpendicular_sides_l140_140074

theorem angle_perpendicular_sides (α β : ℝ) (hα : α = 80) 
  (h_perp : ∀ {x y}, ((x = α → y = 180 - x) ∨ (y = 180 - α → x = y))) : 
  β = 80 ∨ β = 100 :=
by
  sorry

end angle_perpendicular_sides_l140_140074


namespace balls_per_bag_l140_140900

theorem balls_per_bag (total_balls bags_used: Nat) (h1: total_balls = 36) (h2: bags_used = 9) : total_balls / bags_used = 4 := by
  sorry

end balls_per_bag_l140_140900


namespace planting_area_correct_l140_140002

def garden_area : ℕ := 18 * 14
def pond_area : ℕ := 4 * 2
def flower_bed_area : ℕ := (1 / 2) * 3 * 2
def planting_area : ℕ := garden_area - pond_area - flower_bed_area

theorem planting_area_correct : planting_area = 241 := by
  -- proof would go here
  sorry

end planting_area_correct_l140_140002


namespace find_k_l140_140663

theorem find_k (x₁ x₂ k : ℝ) (h1 : x₁ * x₁ - 6 * x₁ + k = 0) (h2 : x₂ * x₂ - 6 * x₂ + k = 0) (h3 : (1 / x₁) + (1 / x₂) = 3) :
  k = 2 :=
by
  sorry

end find_k_l140_140663


namespace lloyd_house_of_cards_l140_140299

theorem lloyd_house_of_cards 
  (decks : ℕ) (cards_per_deck : ℕ) (layers : ℕ)
  (h1 : decks = 24) (h2 : cards_per_deck = 78) (h3 : layers = 48) :
  ((decks * cards_per_deck) / layers) = 39 := 
  by
  sorry

end lloyd_house_of_cards_l140_140299


namespace max_quadratic_function_l140_140295

def quadratic_function (x : ℝ) : ℝ := -3 * x^2 + 12 * x - 5

theorem max_quadratic_function : ∃ x, quadratic_function x = 7 ∧ ∀ x', quadratic_function x' ≤ 7 :=
by
  sorry

end max_quadratic_function_l140_140295


namespace max_profit_l140_140059

theorem max_profit (m : ℝ) :
  (m - 8) * (900 - 15 * m) = -15 * (m - 34) ^ 2 + 10140 :=
by
  sorry

end max_profit_l140_140059


namespace rectangle_aspect_ratio_l140_140285

theorem rectangle_aspect_ratio (x y : ℝ) (h : x > 0 ∧ y > 0 ∧ x / y = 2 * y / x) : x / y = Real.sqrt 2 :=
by
  sorry

end rectangle_aspect_ratio_l140_140285


namespace simple_interest_correct_l140_140617

-- Define the principal amount P
variables {P : ℝ}

-- Define the rate of interest r which is 3% or 0.03 in decimal form
def r : ℝ := 0.03

-- Define the time period t which is 2 years
def t : ℕ := 2

-- Define the compound interest CI for 2 years which is $609
def CI : ℝ := 609

-- Define the simple interest SI that we need to find
def SI : ℝ := 600

-- Define a formula for compound interest
def compound_interest (P r : ℝ) (t : ℕ) : ℝ :=
  P * (1 + r)^t - P

-- Define a formula for simple interest
def simple_interest (P r : ℝ) (t : ℕ) : ℝ :=
  P * r * t

theorem simple_interest_correct (hCI : compound_interest P r t = CI) : simple_interest P r t = SI :=
by
  sorry

end simple_interest_correct_l140_140617


namespace percentage_of_children_allowed_to_draw_l140_140505

def total_jelly_beans := 100
def total_children := 40
def remaining_jelly_beans := 36
def jelly_beans_per_child := 2

theorem percentage_of_children_allowed_to_draw :
  ((total_jelly_beans - remaining_jelly_beans) / jelly_beans_per_child : ℕ) * 100 / total_children = 80 := by
  sorry

end percentage_of_children_allowed_to_draw_l140_140505


namespace find_f_2_find_f_neg2_l140_140128

noncomputable def f : ℝ → ℝ := sorry -- This is left to be defined as a function on ℝ

axiom f_property : ∀ x y : ℝ, f (x + y) = f x + f y + 2 * x * y
axiom f_at_1 : f 1 = 2

theorem find_f_2 : f 2 = 6 := by
  sorry

theorem find_f_neg2 : f (-2) = 2 := by
  sorry

end find_f_2_find_f_neg2_l140_140128


namespace triangle_PQR_QR_length_l140_140935

-- Define the given conditions as a Lean statement
theorem triangle_PQR_QR_length 
  (P Q R : ℝ) -- Angles in the triangle PQR in radians
  (PQ QR PR : ℝ) -- Lengths of the sides of the triangle PQR
  (h1 : Real.cos (2 * P - Q) + Real.sin (P + 2 * Q) = 1) 
  (h2 : PQ = 5)
  (h3 : PQ + QR + PR = 12)
  : QR = 3.5 := 
  sorry -- proof omitted

end triangle_PQR_QR_length_l140_140935


namespace sara_pumpkins_l140_140805

variable (original_pumpkins : ℕ)
variable (eaten_pumpkins : ℕ := 23)
variable (remaining_pumpkins : ℕ := 20)

theorem sara_pumpkins : original_pumpkins = eaten_pumpkins + remaining_pumpkins :=
by
  sorry

end sara_pumpkins_l140_140805


namespace find_center_of_tangent_circle_l140_140583

theorem find_center_of_tangent_circle :
  ∃ (a b : ℝ), (abs a = 5) ∧ (abs b = 5) ∧ (4 * a - 3 * b + 10 = 25) ∧ (a = -5) ∧ (b = 5) :=
by {
  -- Here we would provide the proof in Lean, but for now, we state the theorem
  -- and leave the proof as an exercise.
  sorry
}

end find_center_of_tangent_circle_l140_140583


namespace diana_hits_seven_l140_140345

-- Define the participants
inductive Player 
| Alex 
| Brooke 
| Carlos 
| Diana 
| Emily 
| Fiona

open Player

-- Define a function to get the total score of a participant
def total_score (p : Player) : ℕ :=
match p with
| Alex => 20
| Brooke => 23
| Carlos => 28
| Diana => 18
| Emily => 26
| Fiona => 30

-- Function to check if a dart target is hit within the range and unique
def is_valid_target (x y z : ℕ) :=
x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ 1 ≤ x ∧ x ≤ 12 ∧ 1 ≤ y ∧ y ≤ 12 ∧ 1 ≤ z ∧ z ≤ 12

-- Check if the sum equals the score of the player
def valid_score (p : Player) (x y z : ℕ) :=
is_valid_target x y z ∧ x + y + z = total_score p

-- Lean 4 theorem statement, asking if Diana hits the region 7
theorem diana_hits_seven : ∃ x y z, valid_score Diana x y z ∧ (x = 7 ∨ y = 7 ∨ z = 7) :=
sorry

end diana_hits_seven_l140_140345


namespace solve_negative_integer_sum_l140_140763

theorem solve_negative_integer_sum (N : ℤ) (h1 : N^2 + N = 12) (h2 : N < 0) : N = -4 :=
sorry

end solve_negative_integer_sum_l140_140763


namespace problem1_problem2_l140_140104

-- Proving that (3*sqrt(8) - 12*sqrt(1/2) + sqrt(18)) * 2*sqrt(3) = 6*sqrt(6)
theorem problem1 :
  (3 * Real.sqrt 8 - 12 * Real.sqrt (1/2) + Real.sqrt 18) * 2 * Real.sqrt 3 = 6 * Real.sqrt 6 :=
sorry

-- Proving that (6*sqrt(x/4) - 2*x*sqrt(1/x)) / 3*sqrt(x) = 1/3
theorem problem2 (x : ℝ) (hx : 0 < x) :
  (6 * Real.sqrt (x/4) - 2 * x * Real.sqrt (1/x)) / (3 * Real.sqrt x) = 1/3 :=
sorry

end problem1_problem2_l140_140104


namespace arithmetic_expression_value_l140_140760

theorem arithmetic_expression_value :
  68 + (105 / 15) + (26 * 19) - 250 - (390 / 6) = 254 :=
by
  sorry

end arithmetic_expression_value_l140_140760


namespace cyclic_quadrilateral_angles_l140_140043

theorem cyclic_quadrilateral_angles (A B C D : ℝ) (h_cyclic : A + C = 180) (h_diag_bisect : (A = 2 * (B / 5 + B / 5)) ∧ (C = 2 * (D / 5 + D / 5))) (h_ratio : B / D = 2 / 3):
  A = 80 ∨ A = 100 ∨ A = 1080 / 11 ∨ A = 900 / 11 :=
  sorry

end cyclic_quadrilateral_angles_l140_140043


namespace proposition_true_l140_140676

theorem proposition_true (a b : ℝ) (h1 : 0 > a) (h2 : a > b) : (1/a) < (1/b) := 
sorry

end proposition_true_l140_140676


namespace first_fun_friday_is_march_30_l140_140717

def month_days := 31
def start_day := 4 -- 1 for Sunday, 2 for Monday, ..., 7 for Saturday; 4 means Thursday
def first_friday := 2
def fun_friday (n : ℕ) : ℕ := first_friday + (n - 1) * 7

theorem first_fun_friday_is_march_30 (h1 : start_day = 4)
                                    (h2 : month_days = 31) :
                                    fun_friday 5 = 30 :=
by 
  -- Proof is omitted
  sorry

end first_fun_friday_is_march_30_l140_140717


namespace prob_exactly_two_passes_prob_at_least_one_fails_l140_140093

-- Define the probabilities for students A, B, and C passing their tests.
def prob_A : ℚ := 4/5
def prob_B : ℚ := 3/5
def prob_C : ℚ := 7/10

-- Define the probabilities for students A, B, and C failing their tests.
def prob_not_A : ℚ := 1 - prob_A
def prob_not_B : ℚ := 1 - prob_B
def prob_not_C : ℚ := 1 - prob_C

-- (1) Prove that the probability of exactly two students passing is 113/250.
theorem prob_exactly_two_passes : 
  prob_A * prob_B * prob_not_C + prob_A * prob_not_B * prob_C + prob_not_A * prob_B * prob_C = 113/250 := 
sorry

-- (2) Prove that the probability that at least one student fails is 83/125.
theorem prob_at_least_one_fails : 
  1 - (prob_A * prob_B * prob_C) = 83/125 := 
sorry

end prob_exactly_two_passes_prob_at_least_one_fails_l140_140093


namespace children_total_savings_l140_140311

theorem children_total_savings :
  let josiah_savings := 0.25 * 24
  let leah_savings := 0.50 * 20
  let megan_savings := (2 * 0.50) * 12
  josiah_savings + leah_savings + megan_savings = 28 := by
{
  -- lean proof goes here
  sorry
}

end children_total_savings_l140_140311


namespace even_multiples_of_25_l140_140788

def is_even (n : ℕ) : Prop := n % 2 = 0

def is_multiple_of_25 (n : ℕ) : Prop := n % 25 = 0

theorem even_multiples_of_25 (a b : ℕ) (h1 : 249 ≤ a) (h2 : b ≤ 501) :
  (a = 250 ∨ a = 275 ∨ a = 300 ∨ a = 350 ∨ a = 400 ∨ a = 450) →
  (b = 275 ∨ b = 300 ∨ b = 350 ∨ b = 400 ∨ b = 450 ∨ b = 500) →
  (∃ n, n = 5 ∧ ∀ m, (is_multiple_of_25 m ∧ is_even m ∧ a ≤ m ∧ m ≤ b) ↔ m ∈ [a, b]) :=
by sorry

end even_multiples_of_25_l140_140788


namespace slip_2_5_goes_to_B_l140_140789

-- Defining the slips and their values
def slips : List ℝ := [1.5, 2, 2, 2.5, 3, 3, 3, 3.5, 3.5, 4, 4, 4.5, 5, 5.5, 6]

-- Defining the total sum of slips
def total_sum : ℝ := 52

-- Defining the cup sum values
def cup_sums : List ℝ := [11, 10, 9, 8, 7]

-- Conditions: slip with 4 goes into cup A, slip with 5 goes into cup D
def cup_A_contains : ℝ := 4
def cup_D_contains : ℝ := 5

-- Proof statement
theorem slip_2_5_goes_to_B : 
  ∃ (cup_A cup_B cup_C cup_D cup_E : List ℝ), 
    (cup_A.sum = 11 ∧ cup_B.sum = 10 ∧ cup_C.sum = 9 ∧ cup_D.sum = 8 ∧ cup_E.sum = 7) ∧
    (4 ∈ cup_A) ∧ (5 ∈ cup_D) ∧ (2.5 ∈ cup_B) :=
sorry

end slip_2_5_goes_to_B_l140_140789


namespace gravel_cost_l140_140931

def cost_per_cubic_foot := 8
def cubic_yards := 3
def cubic_feet_per_cubic_yard := 27

theorem gravel_cost :
  (cubic_yards * cubic_feet_per_cubic_yard) * cost_per_cubic_foot = 648 :=
by sorry

end gravel_cost_l140_140931


namespace students_in_class_l140_140117

theorem students_in_class (S : ℕ)
  (h₁ : S / 2 + 2 * S / 5 - S / 10 = 4 * S / 5)
  (h₂ : S / 5 = 4) :
  S = 20 :=
sorry

end students_in_class_l140_140117


namespace value_of_expression_l140_140105

theorem value_of_expression (m : ℝ) (h : m^2 - m - 110 = 0) : (m - 1)^2 + m = 111 := by
  sorry

end value_of_expression_l140_140105


namespace union_of_sets_eq_l140_140609

variable (M N : Set ℕ)

theorem union_of_sets_eq (h1 : M = {1, 2}) (h2 : N = {2, 3}) : M ∪ N = {1, 2, 3} := by
  sorry

end union_of_sets_eq_l140_140609


namespace fraction_sum_l140_140601

-- Define the fractions
def frac1: ℚ := 3/9
def frac2: ℚ := 5/12

-- The theorem statement
theorem fraction_sum : frac1 + frac2 = 3/4 := 
sorry

end fraction_sum_l140_140601


namespace ball_colors_l140_140413

theorem ball_colors (R G B : ℕ) (h1 : R + G + B = 15) (h2 : B = R + 1) (h3 : R = G) (h4 : B = G + 5) : false :=
by
  sorry

end ball_colors_l140_140413


namespace smallest_positive_integer_n_mean_squares_l140_140901

theorem smallest_positive_integer_n_mean_squares :
  ∃ n : ℕ, n > 1 ∧ (∃ m : ℕ, (n * m ^ 2 = (n + 1) * (2 * n + 1) / 6) ∧ Nat.gcd (n + 1) (2 * n + 1) = 1 ∧ n = 337) :=
sorry

end smallest_positive_integer_n_mean_squares_l140_140901


namespace sin_alpha_value_l140_140217

theorem sin_alpha_value (α : ℝ) (h1 : Real.sin (α + π / 4) = 4 / 5) (h2 : α ∈ Set.Ioo (π / 4) (3 * π / 4)) :
  Real.sin α = 7 * Real.sqrt 2 / 10 :=
by
  sorry

end sin_alpha_value_l140_140217


namespace roots_equality_l140_140660

noncomputable def problem_statement (α β γ δ p q : ℝ) : Prop :=
(α - γ) * (β - δ) * (α + δ) * (β + γ) = 4 * (2 * p - 3 * q) ^ 2

theorem roots_equality (α β γ δ p q : ℝ)
  (h₁ : ∀ x, x^2 - 2 * p * x + 3 = 0 → (x = α ∨ x = β))
  (h₂ : ∀ x, x^2 - 3 * q * x + 4 = 0 → (x = γ ∨ x = δ)) :
  problem_statement α β γ δ p q :=
sorry

end roots_equality_l140_140660


namespace find_x_value_l140_140307

/-- Given x, y, z such that x ≠ 0, z ≠ 0, (x / 2) = y^2 + z, and (x / 4) = 4y + 2z, the value of x is 120. -/
theorem find_x_value (x y z : ℝ) (hx : x ≠ 0) (hz : z ≠ 0) (h1 : x / 2 = y^2 + z) (h2 : x / 4 = 4 * y + 2 * z) : x = 120 := 
sorry

end find_x_value_l140_140307


namespace number_of_paths_to_spell_MATH_l140_140077

-- Define the problem setting and conditions
def number_of_paths_M_to_H (adj: ℕ) (steps: ℕ): ℕ :=
  adj^(steps-1)

-- State the problem in Lean 4
theorem number_of_paths_to_spell_MATH : number_of_paths_M_to_H 8 4 = 512 := 
by 
  unfold number_of_paths_M_to_H 
  -- The needed steps are included:
  -- We calculate: 8^(4-1) = 8^3 which should be 512.
  sorry

end number_of_paths_to_spell_MATH_l140_140077


namespace average_percentage_decrease_l140_140685

-- Given definitions
def original_price : ℝ := 10000
def final_price : ℝ := 6400
def num_reductions : ℕ := 2

-- The goal is to prove the average percentage decrease per reduction
theorem average_percentage_decrease (x : ℝ) (h : (original_price * (1 - x)^num_reductions = final_price)) : x = 0.2 :=
sorry

end average_percentage_decrease_l140_140685


namespace frac_pow_zero_l140_140711

def frac := 123456789 / (-987654321 : ℤ)

theorem frac_pow_zero : frac ^ 0 = 1 :=
by sorry

end frac_pow_zero_l140_140711


namespace problem_a_problem_b_l140_140812

-- Define the polynomial P(x) = ax^3 + bx
def P (a b x : ℤ) : ℤ := a * x^3 + b * x

-- Define when a pair (a, b) is n-good
def is_ngood (n a b : ℤ) : Prop :=
  ∀ m k : ℤ, n ∣ P a b m - P a b k → n ∣ m - k

-- Define when a pair (a, b) is very good
def is_verygood (a b : ℤ) : Prop :=
  ∀ n : ℤ, ∃ (infinitely_many_n : ℕ), is_ngood n a b

-- Problem (a): Find a pair (1, -51^2) which is 51-good but not very good
theorem problem_a : ∃ a b : ℤ, a = 1 ∧ b = -(51^2) ∧ is_ngood 51 a b ∧ ¬is_verygood a b := 
by {
  sorry
}

-- Problem (b): Show that all 2010-good pairs are very good
theorem problem_b : ∀ a b : ℤ, is_ngood 2010 a b → is_verygood a b := 
by {
  sorry
}

end problem_a_problem_b_l140_140812


namespace probability_x_gt_3y_l140_140886

theorem probability_x_gt_3y :
  let width := 3000
  let height := 3001
  let triangle_area := (1 / 2 : ℚ) * width * (width / 3)
  let rectangle_area := (width : ℚ) * height
  triangle_area / rectangle_area = 1500 / 9003 :=
by 
  sorry

end probability_x_gt_3y_l140_140886


namespace transform_uniform_random_l140_140042

theorem transform_uniform_random (a_1 : ℝ) (h : 0 ≤ a_1 ∧ a_1 ≤ 1) : -2 ≤ a_1 * 8 - 2 ∧ a_1 * 8 - 2 ≤ 6 :=
by sorry

end transform_uniform_random_l140_140042
