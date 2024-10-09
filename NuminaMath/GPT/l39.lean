import Mathlib

namespace intersection_point_of_diagonals_l39_3923

noncomputable def intersection_of_diagonals (k m b : Real) : Real × Real :=
  let A := (0, b)
  let B := (0, -b)
  let C := (2 * b / (k - m), 2 * b * k / (k - m) - b)
  let D := (-2 * b / (k - m), -2 * b * k / (k - m) + b)
  (0, 0)

theorem intersection_point_of_diagonals (k m b : Real) :
  intersection_of_diagonals k m b = (0, 0) :=
sorry

end intersection_point_of_diagonals_l39_3923


namespace theta_half_quadrant_l39_3982

open Real

theorem theta_half_quadrant (θ : ℝ) (k : ℤ) 
  (h1 : 2 * k * π + 3 * π / 2 ≤ θ ∧ θ ≤ 2 * k * π + 2 * π) 
  (h2 : |cos (θ / 2)| = -cos (θ / 2)) : 
  k * π + 3 * π / 4 ≤ θ / 2 ∧ θ / 2 ≤ k * π + π ∧ cos (θ / 2) < 0 := 
sorry

end theta_half_quadrant_l39_3982


namespace ratio_of_u_to_v_l39_3941

theorem ratio_of_u_to_v (b : ℚ) (hb : b ≠ 0) (u v : ℚ)
  (hu : u = -b / 8) (hv : v = -b / 12) :
  u / v = 3 / 2 :=
by sorry

end ratio_of_u_to_v_l39_3941


namespace zoo_gorilla_percentage_l39_3990

theorem zoo_gorilla_percentage :
  ∀ (visitors_per_hour : ℕ) (open_hours : ℕ) (gorilla_visitors : ℕ) (total_visitors : ℕ)
    (percentage : ℕ),
  visitors_per_hour = 50 → open_hours = 8 → gorilla_visitors = 320 →
  total_visitors = visitors_per_hour * open_hours →
  percentage = (gorilla_visitors * 100) / total_visitors →
  percentage = 80 :=
by
  intros visitors_per_hour open_hours gorilla_visitors total_visitors percentage
  intros h1 h2 h3 h4 h5
  rw [h1, h2] at h4
  rw [h3, h4] at h5
  exact h5

end zoo_gorilla_percentage_l39_3990


namespace problem_quadratic_inequality_l39_3904

theorem problem_quadratic_inequality
  (a : ℝ) (b : ℝ) (c : ℝ)
  (h1 : 0 < a)
  (h2 : a ≤ 4/9)
  (h3 : b = -a)
  (h4 : c = -2*a + 1)
  (h5 : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 2 → 0 ≤ a*x^2 + b*x + c ∧ a*x^2 + b*x + c ≤ 1) :
  3*a + 2*b + c ≠ 1/3 ∧ 3*a + 2*b + c ≠ 5/4 :=
by
  sorry

end problem_quadratic_inequality_l39_3904


namespace factor_polynomial_l39_3939

theorem factor_polynomial (x : ℝ) :
  (x^3 - 12 * x + 16) = (x + 4) * ((x - 2)^2) :=
by
  sorry

end factor_polynomial_l39_3939


namespace k_value_opposite_solutions_l39_3946

theorem k_value_opposite_solutions (k x1 x2 : ℝ) 
  (h1 : 3 * (2 * x1 - 1) = 1 - 2 * x1)
  (h2 : 8 - k = 2 * (x2 + 1))
  (opposite : x2 = -x1) :
  k = 7 :=
by sorry

end k_value_opposite_solutions_l39_3946


namespace complement_set_M_l39_3988

-- Definitions of sets based on given conditions
def universal_set : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

def set_M : Set ℝ := {x | x^2 - x ≤ 0}

-- The proof statement that we need to prove
theorem complement_set_M :
  {x | 1 < x ∧ x ≤ 2} = universal_set \ set_M := by
  sorry

end complement_set_M_l39_3988


namespace ned_time_left_to_diffuse_bomb_l39_3912

-- Conditions
def building_flights : Nat := 20
def time_per_flight : Nat := 11
def bomb_timer : Nat := 72
def time_spent_running : Nat := 165

-- Main statement
theorem ned_time_left_to_diffuse_bomb : 
  (bomb_timer - (building_flights - (time_spent_running / time_per_flight)) * time_per_flight) = 17 :=
by
  sorry

end ned_time_left_to_diffuse_bomb_l39_3912


namespace Helga_articles_written_this_week_l39_3938

def articles_per_30_minutes : ℕ := 5
def work_hours_per_day : ℕ := 4
def work_days_per_week : ℕ := 5
def extra_hours_thursday : ℕ := 2
def extra_hours_friday : ℕ := 3

def articles_per_hour : ℕ := articles_per_30_minutes * 2
def regular_daily_articles : ℕ := articles_per_hour * work_hours_per_day
def regular_weekly_articles : ℕ := regular_daily_articles * work_days_per_week
def extra_thursday_articles : ℕ := articles_per_hour * extra_hours_thursday
def extra_friday_articles : ℕ := articles_per_hour * extra_hours_friday
def extra_weekly_articles : ℕ := extra_thursday_articles + extra_friday_articles
def total_weekly_articles : ℕ := regular_weekly_articles + extra_weekly_articles

theorem Helga_articles_written_this_week : total_weekly_articles = 250 := by
  sorry

end Helga_articles_written_this_week_l39_3938


namespace tribe_leadership_choices_l39_3909

open Nat

theorem tribe_leadership_choices (n m k l : ℕ) (h : n = 15) : 
  (choose 14 2 * choose 12 3 * choose 9 3 * 15 = 27392400) := 
  by sorry

end tribe_leadership_choices_l39_3909


namespace area_of_enclosing_square_is_100_l39_3955

noncomputable def radius : ℝ := 5

noncomputable def diameter_of_circle (r : ℝ) : ℝ := 2 * r

noncomputable def side_length_of_square (d : ℝ) : ℝ := d

noncomputable def area_of_square (s : ℝ) : ℝ := s * s

theorem area_of_enclosing_square_is_100 :
  area_of_square (side_length_of_square (diameter_of_circle radius)) = 100 :=
by
  sorry

end area_of_enclosing_square_is_100_l39_3955


namespace problem_1_l39_3944

noncomputable def derivative_y (a x y : ℝ) (h : y^3 - 3 * y + 2 * a * x = 0) : ℝ :=
  (2 * a) / (3 * (1 - y^2))

theorem problem_1 (a x y : ℝ) (h : y^3 - 3 * y + 2 * a * x = 0) :
  derivative_y a x y h = (2 * a) / (3 * (1 - y^2)) :=
sorry

end problem_1_l39_3944


namespace total_people_selected_l39_3953

-- Define the number of residents in each age group
def residents_21_to_35 : Nat := 840
def residents_36_to_50 : Nat := 700
def residents_51_to_65 : Nat := 560

-- Define the number of people selected from the 36 to 50 age group
def selected_36_to_50 : Nat := 100

-- Define the total number of residents
def total_residents : Nat := residents_21_to_35 + residents_36_to_50 + residents_51_to_65

-- Theorem: Prove that the total number of people selected in this survey is 300
theorem total_people_selected : (100 : ℕ) / (700 : ℕ) * (residents_21_to_35 + residents_36_to_50 + residents_51_to_65) = 300 :=
  by 
    sorry

end total_people_selected_l39_3953


namespace minimum_fencing_l39_3934

variable (a b z : ℝ)

def area_condition : Prop := a * b = 50
def length_condition : Prop := a + 2 * b = z

theorem minimum_fencing (h1 : area_condition a b) (h2 : length_condition a b z) : z ≥ 20 := 
  sorry

end minimum_fencing_l39_3934


namespace jane_project_time_l39_3915

theorem jane_project_time
  (J : ℝ)
  (work_rate_jane_ashley : ℝ := 1 / J + 1 / 40)
  (time_together : ℝ := 15.2 - 8)
  (work_done_together : ℝ := time_together * work_rate_jane_ashley)
  (ashley_alone_time : ℝ := 8)
  (work_done_ashley : ℝ := ashley_alone_time / 40)
  (jane_alone_time : ℝ := 4)
  (work_done_jane_alone : ℝ := jane_alone_time / J) :
  7.2 * (1 / J + 1 / 40) + 8 / 40 + 4 / J = 1 ↔ J = 18.06 :=
by 
  sorry

end jane_project_time_l39_3915


namespace smallest_blocks_required_l39_3950

theorem smallest_blocks_required (L H : ℕ) (block_height block_long block_short : ℕ) 
  (vert_joins_staggered : Prop) (consistent_end_finish : Prop) : 
  L = 120 → H = 10 → block_height = 1 → block_long = 3 → block_short = 1 → 
  (vert_joins_staggered) → (consistent_end_finish) → 
  ∃ n, n = 415 :=
by
  sorry

end smallest_blocks_required_l39_3950


namespace OQ_value_l39_3951

variables {X Y Z N O Q R : Type}
variables [MetricSpace X] [MetricSpace Y] [MetricSpace Z]
variables [MetricSpace N] [MetricSpace O] [MetricSpace Q] [MetricSpace R]
variables (XY YZ XN NY ZO XO OZ YN XR OQ RQ : ℝ)
variables (triangle_XYZ : Triangle X Y Z)
variables (X_equal_midpoint_XY : XY = 540)
variables (Y_equal_midpoint_YZ : YZ = 360)
variables (XN_equal_NY : XN = NY)
variables (ZO_is_angle_bisector : is_angle_bisector Z O X Y)
variables (intersection_YN_ZO : Q = intersection YN ZO)
variables (N_midpoint_RQ : is_midpoint N R Q)
variables (XR_value : XR = 216)

theorem OQ_value : OQ = 216 := sorry

end OQ_value_l39_3951


namespace triangle_side_lengths_l39_3901

noncomputable def radius_inscribed_circle := 4/3
def sum_of_heights := 13

theorem triangle_side_lengths :
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  ∃ (h_a h_b h_c : ℕ), h_a ≠ h_b ∧ h_b ≠ h_c ∧ h_a ≠ h_c ∧
  h_a + h_b + h_c = sum_of_heights ∧
  r * (a + b + c) = 8 ∧ -- (since Δ = r * s, where s = (a + b + c)/2)
  1 / 2 * a * h_a = 1 / 2 * b * h_b ∧
  1 / 2 * b * h_b = 1 / 2 * c * h_c ∧
  a = 6 ∧ b = 4 ∧ c = 3 :=
sorry

end triangle_side_lengths_l39_3901


namespace office_needs_24_pencils_l39_3986

noncomputable def number_of_pencils (total_cost : ℝ) (cost_per_pencil : ℝ) (cost_per_folder : ℝ) (number_of_folders : ℕ) : ℝ :=
  (total_cost - (number_of_folders * cost_per_folder)) / cost_per_pencil

theorem office_needs_24_pencils :
  number_of_pencils 30 0.5 0.9 20 = 24 :=
by
  sorry

end office_needs_24_pencils_l39_3986


namespace isosceles_trapezoid_side_length_is_five_l39_3965

noncomputable def isosceles_trapezoid_side_length (b1 b2 area : ℝ) : ℝ :=
  let h := 2 * area / (b1 + b2)
  let base_diff_half := (b2 - b1) / 2
  Real.sqrt (h^2 + base_diff_half^2)
  
theorem isosceles_trapezoid_side_length_is_five :
  isosceles_trapezoid_side_length 6 12 36 = 5 := by
  sorry

end isosceles_trapezoid_side_length_is_five_l39_3965


namespace area_of_trapezoid_RSQT_l39_3940
-- Import the required library

-- Declare the geometrical setup and given areas
variables (PQ PR : ℝ)
variable (PQR_area : ℝ)
variable (small_triangle_area : ℝ)
variable (num_small_triangles : ℕ)
variable (inner_triangle_area : ℝ)
variable (trapezoid_RSQT_area : ℝ)

-- Define the conditions from part a)
def isosceles_triangle : Prop := PQ = PR
def triangle_PQR_area_given : Prop := PQR_area = 75
def small_triangle_area_given : Prop := small_triangle_area = 3
def num_small_triangles_given : Prop := num_small_triangles = 9
def inner_triangle_area_given : Prop := inner_triangle_area = 5 * small_triangle_area

-- Define the target statement (question == answer)
theorem area_of_trapezoid_RSQT :
  isosceles_triangle PQ PR ∧
  triangle_PQR_area_given PQR_area ∧
  small_triangle_area_given small_triangle_area ∧
  num_small_triangles_given num_small_triangles ∧
  inner_triangle_area_given small_triangle_area inner_triangle_area → 
  trapezoid_RSQT_area = 60 :=
sorry

end area_of_trapezoid_RSQT_l39_3940


namespace niu_fraction_property_l39_3978

open Nat

-- Given mn <= 2009, where m, n are positive integers and (n/m) is in lowest terms
-- Prove that for adjacent terms in the sequence, m_k n_{k+1} - m_{k+1} n_k = 1.

noncomputable def is_numerator_denom_pair_in_seq (m n : ℕ) : Bool :=
  m > 0 ∧ n > 0 ∧ m * n ≤ 2009

noncomputable def are_sorted_adjacent_in_seq (m_k n_k m_k1 n_k1 : ℕ) : Bool :=
  m_k * n_k1 - m_k1 * n_k = 1

theorem niu_fraction_property :
  ∀ (m_k n_k m_k1 n_k1 : ℕ),
  is_numerator_denom_pair_in_seq m_k n_k →
  is_numerator_denom_pair_in_seq m_k1 n_k1 →
  m_k < m_k1 →
  are_sorted_adjacent_in_seq m_k n_k m_k1 n_k1
:=
sorry

end niu_fraction_property_l39_3978


namespace original_price_of_dish_l39_3994

theorem original_price_of_dish :
  let P : ℝ := 40
  (0.9 * P + 0.15 * P) - (0.9 * P + 0.15 * 0.9 * P) = 0.60 → P = 40 := by
  intros P h
  sorry

end original_price_of_dish_l39_3994


namespace starting_number_of_range_l39_3960

theorem starting_number_of_range (N : ℕ) : ∃ (start : ℕ), 
  (∀ n, n ≥ start ∧ n ≤ 200 → ∃ k, 8 * k = n) ∧ -- All numbers between start and 200 inclusive are multiples of 8
  (∃ k, k = (200 / 8) ∧ 25 - k = 13.5) ∧ -- There are 13.5 multiples of 8 in the range
  start = 84 := 
sorry

end starting_number_of_range_l39_3960


namespace placemat_length_l39_3999

noncomputable def calculate_placemat_length
    (R : ℝ)
    (num_mats : ℕ)
    (mat_width : ℝ)
    (overlap_ratio : ℝ) : ℝ := 
    let circumference := 2 * Real.pi * R
    let arc_length := circumference / num_mats
    let angle := 2 * Real.pi / num_mats
    let chord_length := 2 * R * Real.sin (angle / 2)
    let effective_mat_length := chord_length / (1 - overlap_ratio * 2)
    effective_mat_length

theorem placemat_length (R : ℝ) (num_mats : ℕ) (mat_width : ℝ) (overlap_ratio : ℝ): 
    R = 5 ∧ num_mats = 8 ∧ mat_width = 2 ∧ overlap_ratio = (1 / 4)
    → calculate_placemat_length R num_mats mat_width overlap_ratio = 7.654 :=
by
  sorry

end placemat_length_l39_3999


namespace thomas_percentage_l39_3969

/-- 
Prove that if Emmanuel gets 100 jelly beans out of a total of 200 jelly beans, and 
Barry and Emmanuel share the remainder in a 4:5 ratio, then Thomas takes 10% 
of the jelly beans.
-/
theorem thomas_percentage (total_jelly_beans : ℕ) (emmanuel_jelly_beans : ℕ)
  (barry_ratio : ℕ) (emmanuel_ratio : ℕ) (thomas_percentage : ℕ) :
  total_jelly_beans = 200 → emmanuel_jelly_beans = 100 → barry_ratio = 4 → emmanuel_ratio = 5 →
  thomas_percentage = 10 :=
by
  intros;
  sorry

end thomas_percentage_l39_3969


namespace radius_of_inscribed_circle_l39_3943

theorem radius_of_inscribed_circle (r1 r2 : ℝ) (AC BC AB : ℝ) 
  (h1 : AC = 2 * r1)
  (h2 : BC = 2 * r2)
  (h3 : AB = 2 * Real.sqrt (r1^2 + r2^2)) : 
  (r1 + r2 - Real.sqrt (r1^2 + r2^2)) = ((2 * r1 + 2 * r2 - 2 * Real.sqrt (r1^2 + r2^2)) / 2) := 
by
  sorry

end radius_of_inscribed_circle_l39_3943


namespace jen_problem_correct_answer_l39_3947

-- Definitions based on the conditions
def sum_178_269 : ℤ := 178 + 269
def round_to_nearest_hundred (n : ℤ) : ℤ :=
  if n % 100 >= 50 then n - (n % 100) + 100 else n - (n % 100)

-- Prove the statement
theorem jen_problem_correct_answer :
  round_to_nearest_hundred sum_178_269 = 400 :=
by
  have h1 : sum_178_269 = 447 := rfl
  have h2 : round_to_nearest_hundred 447 = 400 := by sorry
  exact h2

end jen_problem_correct_answer_l39_3947


namespace pics_per_album_eq_five_l39_3933

-- Definitions based on conditions
def pics_from_phone : ℕ := 5
def pics_from_camera : ℕ := 35
def total_pics : ℕ := pics_from_phone + pics_from_camera
def num_albums : ℕ := 8

-- Statement to prove
theorem pics_per_album_eq_five : total_pics / num_albums = 5 := by
  sorry

end pics_per_album_eq_five_l39_3933


namespace magician_earnings_l39_3976

theorem magician_earnings (price_per_deck : ℕ) (initial_decks : ℕ) (decks_remaining : ℕ) (money_earned : ℕ) : 
    price_per_deck = 7 →
    initial_decks = 16 →
    decks_remaining = 8 →
    money_earned = (initial_decks - decks_remaining) * price_per_deck →
    money_earned = 56 :=
by
  intros hp hi hd he
  rw [hp, hi, hd] at he
  exact he

end magician_earnings_l39_3976


namespace total_students_is_30_l39_3921

def students_per_bed : ℕ := 2 

def beds_per_room : ℕ := 2 

def students_per_couch : ℕ := 1 

def rooms_booked : ℕ := 6 

def total_students := (students_per_bed * beds_per_room + students_per_couch) * rooms_booked

theorem total_students_is_30 : total_students = 30 := by
  sorry

end total_students_is_30_l39_3921


namespace maximum_at_vertex_l39_3910

def quadratic_function (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem maximum_at_vertex (a b c x_0 : ℝ) (h_a : a < 0) (h_x0 : 2 * a * x_0 + b = 0) :
  ∀ x : ℝ, quadratic_function a b c x ≤ quadratic_function a b c x_0 :=
sorry

end maximum_at_vertex_l39_3910


namespace num_five_digit_numbers_is_correct_l39_3998

-- Define the set of digits and their repetition as given in the conditions
def digits : Multiset ℕ := {1, 3, 3, 5, 8}

-- Calculate the permutation with repetitions
noncomputable def num_five_digit_numbers : ℕ := (digits.card.factorial) / 
  (Multiset.count 1 digits).factorial / 
  (Multiset.count 3 digits).factorial / 
  (Multiset.count 5 digits).factorial / 
  (Multiset.count 8 digits).factorial

-- Theorem stating the final result
theorem num_five_digit_numbers_is_correct : num_five_digit_numbers = 60 :=
by
  -- Proof is omitted
  sorry

end num_five_digit_numbers_is_correct_l39_3998


namespace remainder_sum_modulo_eleven_l39_3932

theorem remainder_sum_modulo_eleven :
  (88132 + 88133 + 88134 + 88135 + 88136 + 88137 + 88138 + 88139 + 88140 + 88141) % 11 = 1 :=
by
  sorry

end remainder_sum_modulo_eleven_l39_3932


namespace max_score_top_three_teams_l39_3948

theorem max_score_top_three_teams : 
  ∀ (teams : Finset String) (points : String → ℕ), 
    teams.card = 6 →
    (∀ team, team ∈ teams → (points team = 0 ∨ points team = 1 ∨ points team = 3)) →
    ∃ top_teams : Finset String, top_teams.card = 3 ∧ 
    (∀ team, team ∈ top_teams → points team = 24) := 
by sorry

end max_score_top_three_teams_l39_3948


namespace function_periodicity_l39_3926

theorem function_periodicity
  (f : ℝ → ℝ)
  (H_odd : ∀ x, f (-x) = -f x)
  (H_even_shift : ∀ x, f (x + 2) = f (-x + 2))
  (H_val_neg1 : f (-1) = -1)
  : f 2017 + f 2016 = 1 := 
sorry

end function_periodicity_l39_3926


namespace boat_stream_speeds_l39_3931

variable (x y : ℝ)

theorem boat_stream_speeds (h1 : 20 + x ≠ 0) (h2 : 40 - y ≠ 0) :
  380 = 7 * x + 13 * y ↔ 
  26 * (40 - y) = 14 * (20 + x) :=
by { sorry }

end boat_stream_speeds_l39_3931


namespace neg_70kg_represents_subtract_70kg_l39_3917

theorem neg_70kg_represents_subtract_70kg (add_30kg : Int) (concept_opposite : ∀ (x : Int), x = -(-x)) :
  -70 = -70 := 
by
  sorry

end neg_70kg_represents_subtract_70kg_l39_3917


namespace smallest_sum_divisible_by_5_l39_3961

-- Definition of a prime number
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition of four consecutive primes greater than 5
def four_consecutive_primes_greater_than_five (a b c d : ℕ) : Prop :=
  is_prime a ∧ is_prime b ∧ is_prime c ∧ is_prime d ∧ a > 5 ∧ b > 5 ∧ c > 5 ∧ d > 5 ∧ 
  b = a + 4 ∧ c = b + 6 ∧ d = c + 2

-- The statement to prove
theorem smallest_sum_divisible_by_5 :
  (∃ a b c d : ℕ, four_consecutive_primes_greater_than_five a b c d ∧ (a + b + c + d) % 5 = 0 ∧
   ∀ x y z w : ℕ, four_consecutive_primes_greater_than_five x y z w → (x + y + z + w) % 5 = 0 → a + b + c + d ≤ x + y + z + w) →
  (∃ a b c d : ℕ, four_consecutive_primes_greater_than_five a b c d ∧ (a + b + c + d) = 60) :=
by
  sorry

end smallest_sum_divisible_by_5_l39_3961


namespace minimum_experiments_fractional_method_l39_3972

/--
A pharmaceutical company needs to optimize the cultivation temperature for a certain medicinal liquid through bioassay.
The experimental range is set from 29℃ to 63℃, with an accuracy requirement of ±1℃.
Prove that the minimum number of experiments required to ensure the best cultivation temperature is found using the fractional method is 7.
-/
theorem minimum_experiments_fractional_method
  (range_start : ℕ)
  (range_end : ℕ)
  (accuracy : ℕ)
  (fractional_method : ∀ (range_start range_end accuracy: ℕ), ℕ) :
  range_start = 29 → range_end = 63 → accuracy = 1 → fractional_method range_start range_end accuracy = 7 := by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end minimum_experiments_fractional_method_l39_3972


namespace banks_investments_count_l39_3966

-- Conditions
def revenue_per_investment_banks := 500
def revenue_per_investment_elizabeth := 900
def number_of_investments_elizabeth := 5
def extra_revenue_elizabeth := 500

-- Total revenue calculations
def total_revenue_elizabeth := number_of_investments_elizabeth * revenue_per_investment_elizabeth
def total_revenue_banks := total_revenue_elizabeth - extra_revenue_elizabeth

-- Number of investments for Mr. Banks
def number_of_investments_banks := total_revenue_banks / revenue_per_investment_banks

theorem banks_investments_count : number_of_investments_banks = 8 := by
  sorry

end banks_investments_count_l39_3966


namespace range_of_function_l39_3906

open Real

theorem range_of_function (x : ℝ) (h : 0 < x ∧ x < π / 2) :
  ∃ y, y = sin x - 2 * cos x + 32 / (125 * sin x * (1 - cos x)) ∧ y ≥ 2 / 5 :=
sorry

end range_of_function_l39_3906


namespace solve_equation1_solve_equation2_l39_3935

def equation1 (x : ℝ) := (x - 1) ^ 2 = 4
def equation2 (x : ℝ) := 2 * x ^ 3 = -16

theorem solve_equation1 (x : ℝ) (h : equation1 x) : x = 3 ∨ x = -1 := 
sorry

theorem solve_equation2 (x : ℝ) (h : equation2 x) : x = -2 := 
sorry

end solve_equation1_solve_equation2_l39_3935


namespace sequence_factorial_l39_3911

theorem sequence_factorial (a : ℕ → ℕ) (h₁ : a 1 = 1) (h₂ : ∀ n : ℕ, n > 0 → a n = n * a (n - 1)) :
  ∀ n : ℕ, a n = Nat.factorial n :=
by
  sorry

end sequence_factorial_l39_3911


namespace total_treat_value_is_339100_l39_3980

def hotel_cost (cost_per_night : ℕ) (nights : ℕ) (discount : ℕ) : ℕ :=
  let total_cost := cost_per_night * nights
  total_cost - (total_cost * discount / 100)

def car_cost (base_price : ℕ) (tax : ℕ) : ℕ :=
  base_price + (base_price * tax / 100)

def house_cost (car_base_price : ℕ) (multiplier : ℕ) (property_tax : ℕ) : ℕ :=
  let house_value := car_base_price * multiplier
  house_value + (house_value * property_tax / 100)

def yacht_cost (hotel_value : ℕ) (car_value : ℕ) (multiplier : ℕ) (discount : ℕ) : ℕ :=
  let combined_value := hotel_value + car_value
  let yacht_value := combined_value * multiplier
  yacht_value - (yacht_value * discount / 100)

def gold_coins_cost (yacht_value : ℕ) (multiplier : ℕ) (tax : ℕ) : ℕ :=
  let gold_value := yacht_value * multiplier
  gold_value + (gold_value * tax / 100)

theorem total_treat_value_is_339100 :
  let hotel_value := hotel_cost 4000 2 5
  let car_value := car_cost 30000 10
  let house_value := house_cost 30000 4 2
  let yacht_value := yacht_cost 8000 30000 2 7
  let gold_coins_value := gold_coins_cost 76000 3 3
  hotel_value + car_value + house_value + yacht_value + gold_coins_value = 339100 :=
by sorry

end total_treat_value_is_339100_l39_3980


namespace compute_4_star_3_l39_3975

def custom_op (a b : ℕ) : ℕ := a^2 - a * b + b^2

theorem compute_4_star_3 : custom_op 4 3 = 13 :=
by
  sorry

end compute_4_star_3_l39_3975


namespace constant_speed_total_distance_l39_3920

def travel_time : ℝ := 5.5
def distance_per_hour : ℝ := 100
def speed := distance_per_hour

theorem constant_speed : ∀ t : ℝ, (1 ≤ t) ∧ (t ≤ travel_time) → speed = distance_per_hour := 
by sorry

theorem total_distance : speed * travel_time = 550 :=
by sorry

end constant_speed_total_distance_l39_3920


namespace river_depth_is_correct_l39_3937

noncomputable def depth_of_river (width : ℝ) (flow_rate_kmph : ℝ) (volume_per_min : ℝ) : ℝ :=
  let flow_rate_mpm := (flow_rate_kmph * 1000) / 60
  let cross_sectional_area := volume_per_min / flow_rate_mpm
  cross_sectional_area / width

theorem river_depth_is_correct :
  depth_of_river 65 6 26000 = 4 :=
by
  -- Steps to compute depth (converted from solution)
  sorry

end river_depth_is_correct_l39_3937


namespace part1_part2_l39_3918

noncomputable def f (x : ℝ) : ℝ := |x - 1| - 1
noncomputable def g (x : ℝ) : ℝ := -|x + 1| - 4

theorem part1 (x : ℝ) : f x ≤ 1 ↔ -1 ≤ x ∧ x ≤ 3 :=
by
  sorry

theorem part2 (m : ℝ) : (∀ x : ℝ, f x - g x ≥ m + 1) ↔ m ≤ 4 :=
by
  sorry

end part1_part2_l39_3918


namespace votes_difference_l39_3903

theorem votes_difference (V : ℝ) (h1 : 0.62 * V = 899) :
  |(0.62 * V) - (0.38 * V)| = 348 :=
by
  -- The solution goes here
  sorry

end votes_difference_l39_3903


namespace find_real_solutions_l39_3930

theorem find_real_solutions (x : ℝ) :
  (1 / ((x - 2) * (x - 3)) + 1 / ((x - 3) * (x - 4)) + 1 / ((x - 4) * (x - 5)) = 1 / 8) ↔ (x = 7 ∨ x = -2) := 
by
  sorry

end find_real_solutions_l39_3930


namespace membership_percentage_change_l39_3914

theorem membership_percentage_change :
  let initial_membership := 100.0
  let first_fall_membership := initial_membership * 1.04
  let first_spring_membership := first_fall_membership * 0.95
  let second_fall_membership := first_spring_membership * 1.07
  let second_spring_membership := second_fall_membership * 0.97
  let third_fall_membership := second_spring_membership * 1.05
  let third_spring_membership := third_fall_membership * 0.81
  let final_membership := third_spring_membership
  let total_percentage_change := ((final_membership - initial_membership) / initial_membership) * 100.0
  total_percentage_change = -12.79 :=
by
  sorry

end membership_percentage_change_l39_3914


namespace speed_ratio_l39_3997

theorem speed_ratio (va vb : ℝ) (L : ℝ) (h : va = vb * k) (head_start : vb * (L - 0.05 * L) = vb * L) : 
    (va / vb) = (1 / 0.95) :=
by
  sorry

end speed_ratio_l39_3997


namespace cos_675_eq_sqrt2_div_2_l39_3981

theorem cos_675_eq_sqrt2_div_2 : Real.cos (675 * Real.pi / 180) = Real.sqrt 2 / 2 :=
by 
  sorry

end cos_675_eq_sqrt2_div_2_l39_3981


namespace ratio_of_b_plus_e_over_c_plus_f_l39_3991

theorem ratio_of_b_plus_e_over_c_plus_f 
  (a b c d e f : ℝ)
  (h1 : a + b = 2 * a + c)
  (h2 : a - 2 * b = 4 * c)
  (h3 : a + b + c = 21)
  (h4 : d + e = 3 * d + f)
  (h5 : d - 2 * e = 5 * f)
  (h6 : d + e + f = 32) :
  (b + e) / (c + f) = -3.99 :=
sorry

end ratio_of_b_plus_e_over_c_plus_f_l39_3991


namespace sector_angle_measure_l39_3928

-- Define the variables
variables (r α : ℝ)

-- Define the conditions
def perimeter_condition := (2 * r + r * α = 4)
def area_condition := (1 / 2 * α * r^2 = 1)

-- State the theorem
theorem sector_angle_measure (h1 : perimeter_condition r α) (h2 : area_condition r α) : α = 2 :=
sorry

end sector_angle_measure_l39_3928


namespace vector_parallel_solution_l39_3979

theorem vector_parallel_solution 
  (x : ℝ) 
  (a : ℝ × ℝ) 
  (b : ℝ × ℝ) 
  (ha : a = (2, 3)) 
  (hb : b = (x, -9)) 
  (h_parallel : ∃ k : ℝ, b = (k * a.1, k * a.2)) : 
  x = -6 :=
by
  sorry

end vector_parallel_solution_l39_3979


namespace sum_of_digits_least_N_l39_3971

def P (N k : ℕ) : ℚ :=
  (N + 1 - 2 * ⌈(2 * N : ℚ) / 5⌉) / (N + 1)

theorem sum_of_digits_least_N (k : ℕ) (h_k : k = 2) (h1 : ∀ N, P N k < 8 / 10 ) :
  ∃ N : ℕ, (N % 10) + (N / 10) = 1 ∧ (P N k < 8 / 10) ∧ (∀ M : ℕ, M < N → P M k ≥ 8 / 10) := by
  sorry

end sum_of_digits_least_N_l39_3971


namespace range_of_m_l39_3925

theorem range_of_m (m : ℝ) : (∃ x y : ℝ, 2 * x^2 - 3 * x + m = 0 ∧ 2 * y^2 - 3 * y + m = 0) → m ≤ 9 / 8 :=
by
  intro h
  -- We need to implement the proof here
  sorry

end range_of_m_l39_3925


namespace value_of_y_l39_3957

theorem value_of_y (y m : ℕ) (h1 : ((1 ^ m) / (y ^ m)) * (1 ^ 16 / 4 ^ 16) = 1 / (2 * 10 ^ 31)) (h2 : m = 31) : 
  y = 5 := 
sorry

end value_of_y_l39_3957


namespace value_of_fraction_l39_3905

variables (w x y : ℝ)

theorem value_of_fraction (h1 : w / x = 1 / 3) (h2 : w / y = 3 / 4) : (x + y) / y = 13 / 4 :=
sorry

end value_of_fraction_l39_3905


namespace chris_and_fiona_weight_l39_3995

theorem chris_and_fiona_weight (c d e f : ℕ) (h1 : c + d = 330) (h2 : d + e = 290) (h3 : e + f = 310) : c + f = 350 :=
by
  sorry

end chris_and_fiona_weight_l39_3995


namespace trapezoid_segment_ratio_l39_3936

theorem trapezoid_segment_ratio (s l : ℝ) (h₁ : 3 * s + l = 1) (h₂ : 2 * l + 6 * s = 2) :
  l = 2 * s :=
by
  sorry

end trapezoid_segment_ratio_l39_3936


namespace maximum_of_f_attain_maximum_of_f_l39_3996

def f (x : ℝ) : ℝ := x^3 + 3 * x^2 - 4

theorem maximum_of_f : ∀ x : ℝ, f x ≤ 0 :=
sorry

theorem attain_maximum_of_f : ∃ x : ℝ, f x = 0 :=
sorry

end maximum_of_f_attain_maximum_of_f_l39_3996


namespace solve_fractional_equation_l39_3985

theorem solve_fractional_equation
  (x : ℝ)
  (h1 : x ≠ 0)
  (h2 : x ≠ 2)
  (h_eq : 2 / x - 1 / (x - 2) = 0) : 
  x = 4 := by
  sorry

end solve_fractional_equation_l39_3985


namespace zinc_weight_in_mixture_l39_3970

theorem zinc_weight_in_mixture (total_weight : ℝ) (zinc_ratio : ℝ) (copper_ratio : ℝ) (total_parts : ℝ) (fraction_zinc : ℝ) (weight_zinc : ℝ) :
  zinc_ratio = 9 ∧ copper_ratio = 11 ∧ total_weight = 70 ∧ total_parts = zinc_ratio + copper_ratio ∧
  fraction_zinc = zinc_ratio / total_parts ∧ weight_zinc = fraction_zinc * total_weight →
  weight_zinc = 31.5 :=
by
  intros h
  sorry

end zinc_weight_in_mixture_l39_3970


namespace driving_time_is_correct_l39_3962

-- Define conditions
def flight_departure : ℕ := 20 * 60 -- 8:00 pm in minutes since 0:00
def checkin_time : ℕ := flight_departure - 2 * 60 -- 2 hours early
def latest_leave_time : ℕ := 17 * 60 -- 5:00 pm in minutes since 0:00
def additional_time : ℕ := 15 -- 15 minutes to park and make their way to the terminal

-- Define question
def driving_time : ℕ := checkin_time - additional_time - latest_leave_time

-- Prove the expected answer
theorem driving_time_is_correct : driving_time = 45 :=
by
  -- omitting the proof
  sorry

end driving_time_is_correct_l39_3962


namespace num_toys_purchased_min_selling_price_l39_3902

variable (x m : ℕ)

-- Given conditions
axiom cond1 : 1500 / x + 5 = 3500 / (2 * x)
axiom cond2 : 150 * m - 5000 >= 1150

-- Required proof
theorem num_toys_purchased : x = 50 :=
by
  sorry

theorem min_selling_price : m >= 41 :=
by
  sorry

end num_toys_purchased_min_selling_price_l39_3902


namespace find_x_for_y_equals_six_l39_3984

variable (x y k : ℚ)

-- Conditions
def varies_inversely_as_square := x = k / y^2
def initial_condition := (y = 3 ∧ x = 1)

-- Problem Statement
theorem find_x_for_y_equals_six (h₁ : varies_inversely_as_square x y k) (h₂ : initial_condition x y) :
  ∃ k, (k = 9 ∧ x = k / 6^2 ∧ x = 1 / 4) :=
sorry

end find_x_for_y_equals_six_l39_3984


namespace bottle_total_height_l39_3907

theorem bottle_total_height (r1 r2 water_height_up water_height_down : ℝ) (h_r1 : r1 = 1) (h_r2 : r2 = 3) (h_water_height_up : water_height_up = 20) (h_water_height_down : water_height_down = 28) : 
    ∃ x : ℝ, (π * r1^2 * (x - water_height_up) = 9 * π * (x - water_height_down) ∧ x = 29) := 
by 
    sorry

end bottle_total_height_l39_3907


namespace sector_central_angle_l39_3949

noncomputable def sector_angle (R L : ℝ) : ℝ := L / R

theorem sector_central_angle :
  ∃ R L : ℝ, 
    (R > 0) ∧ 
    (L > 0) ∧ 
    (1 / 2 * L * R = 5) ∧ 
    (2 * R + L = 9) ∧ 
    (sector_angle R L = 8 / 5 ∨ sector_angle R L = 5 / 2) :=
sorry

end sector_central_angle_l39_3949


namespace total_steps_l39_3959

theorem total_steps (steps_per_floor : ℕ) (n : ℕ) (m : ℕ) (h : steps_per_floor = 20) (hm : m = 11) (hn : n = 1) : 
  steps_per_floor * (m - n) = 200 :=
by
  sorry

end total_steps_l39_3959


namespace p_nonnegative_iff_equal_l39_3922

def p (a b c x : ℝ) : ℝ := (x - a) * (x - b) + (x - b) * (x - c) + (x - c) * (x - a)

theorem p_nonnegative_iff_equal (a b c : ℝ) : (∀ x : ℝ, p a b c x ≥ 0) ↔ a = b ∧ b = c :=
by
  sorry

end p_nonnegative_iff_equal_l39_3922


namespace range_of_a_l39_3913

-- Define the even function property
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- Define the monotonically increasing property on [0, ∞)
def mono_increasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → 0 ≤ y → x ≤ y → f x ≤ f y

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) :
  even_function f →
  mono_increasing_on_nonneg f →
  (f (Real.log a / Real.log 2) + f (Real.log a / Real.log (1/2)) ≤ 2 * f 1) →
  (0 < a ∧ a ≤ 2) :=
by
  intros h_even h_mono h_ineq
  sorry

end range_of_a_l39_3913


namespace rectangle_horizontal_length_l39_3945

variable (squareside rectheight : ℕ)

-- Condition: side of the square is 80 cm, vertical side length of the rectangle is 100 cm
def square_side_length := 80
def rect_vertical_length := 100

-- Question: Calculate the horizontal length of the rectangle
theorem rectangle_horizontal_length :
  (4 * square_side_length) = (2 * rect_vertical_length + 2 * rect_horizontal_length) -> rect_horizontal_length = 60 := by
  sorry

end rectangle_horizontal_length_l39_3945


namespace matrix_vector_computation_l39_3989

-- Setup vectors and their corresponding matrix multiplication results
variables {R : Type*} [Field R]
variables {M : Matrix (Fin 2) (Fin 2) R} {u z : Fin 2 → R}

-- Conditions given in (a)
def condition1 : M.mulVec u = ![3, -4] :=
  sorry

def condition2 : M.mulVec z = ![-1, 6] :=
  sorry

-- Statement equivalent to the proof problem given in (c)
theorem matrix_vector_computation :
  M.mulVec (3 • u - 2 • z) = ![11, -24] :=
by
  -- Use the conditions to prove the theorem
  sorry

end matrix_vector_computation_l39_3989


namespace negation_of_exists_l39_3992

theorem negation_of_exists (x : ℝ) : ¬(∃ x_0 : ℝ, |x_0| + x_0^2 < 0) ↔ ∀ x : ℝ, |x| + x^2 ≥ 0 :=
by
  sorry

end negation_of_exists_l39_3992


namespace largest_n_value_l39_3977

theorem largest_n_value (n : ℕ) (h1: n < 100000) (h2: (9 * (n - 3)^6 - n^3 + 16 * n - 27) % 7 = 0) : n = 99996 := 
sorry

end largest_n_value_l39_3977


namespace stacy_history_paper_pages_l39_3954

def stacy_paper := 1 -- Number of pages Stacy writes per day
def days_to_finish := 12 -- Number of days Stacy has to finish the paper

theorem stacy_history_paper_pages : stacy_paper * days_to_finish = 12 := by
  sorry

end stacy_history_paper_pages_l39_3954


namespace find_salary_l39_3974

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

end find_salary_l39_3974


namespace ratio_of_areas_l39_3956

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables {A B C M : V}

-- Define the collinearity condition point M in the triangle plane with respect to vectors AB and AC
def point_condition (A B C M : V) : Prop :=
  5 • (M - A) = (B - A) + 3 • (C - A)

-- Define an area ratio function
def area_ratio_triangles (A B C M : V) [AddCommGroup V] [Module ℝ V] : ℝ :=
  sorry  -- Implementation of area ratio comparison, abstracted out for the given problem statement

-- The theorem to prove
theorem ratio_of_areas (hM : point_condition A B C M) : area_ratio_triangles A B C M = 3 / 5 :=
sorry

end ratio_of_areas_l39_3956


namespace ratio_of_cats_l39_3964

-- Definitions from conditions
def total_animals_anthony := 12
def fraction_cats_anthony := 2 / 3
def extra_dogs_leonel := 7
def total_animals_both := 27

-- Calculate number of cats and dogs Anthony has
def cats_anthony := fraction_cats_anthony * total_animals_anthony
def dogs_anthony := total_animals_anthony - cats_anthony

-- Calculate number of dogs Leonel has
def dogs_leonel := dogs_anthony + extra_dogs_leonel

-- Calculate number of cats Leonel has
def cats_leonel := total_animals_both - (cats_anthony + dogs_anthony + dogs_leonel)

-- Prove the desired ratio
theorem ratio_of_cats : (cats_leonel / cats_anthony) = (1 / 2) := by
  -- Insert proof steps here
  sorry

end ratio_of_cats_l39_3964


namespace new_students_admitted_l39_3967

-- Definitions of the conditions
def original_students := 35
def increase_in_expenses := 42
def decrease_in_average_expense := 1
def original_expenditure := 420

-- Main statement: proving the number of new students admitted
theorem new_students_admitted : ∃ x : ℕ, 
  (original_expenditure + increase_in_expenses = 11 * (original_students + x)) ∧ 
  (x = 7) := 
sorry

end new_students_admitted_l39_3967


namespace problem_statement_l39_3987

theorem problem_statement (x y z : ℝ) (h : (x - z)^2 - 4 * (x - y) * (y - z) = 0) : x + z - 2 * y = 0 :=
sorry

end problem_statement_l39_3987


namespace binomial_coeff_divisibility_l39_3919

theorem binomial_coeff_divisibility (n k : ℕ) (hn : 0 < n) (hk : 0 < k) : n ∣ (Nat.choose n k) * Nat.gcd n k :=
sorry

end binomial_coeff_divisibility_l39_3919


namespace radian_measure_of_acute_angle_l39_3927

theorem radian_measure_of_acute_angle 
  (r1 r2 r3 : ℝ) (h1 : r1 = 4) (h2 : r2 = 3) (h3 : r3 = 2)
  (θ : ℝ) (S U : ℝ) 
  (hS : S = U * 9 / 14) (h_total_area : (π * r1^2) + (π * r2^2) + (π * r3^2) = S + U) :
  θ = 1827 * π / 3220 :=
by
  -- proof goes here
  sorry

end radian_measure_of_acute_angle_l39_3927


namespace reciprocal_of_mixed_number_l39_3900

def mixed_number := -1 - (4 / 5)

def reciprocal (x : ℚ) : ℚ := 1 / x

theorem reciprocal_of_mixed_number : reciprocal mixed_number = -5 / 9 := 
by
  sorry

end reciprocal_of_mixed_number_l39_3900


namespace opposite_of_number_l39_3993

-- Define the original number
def original_number : ℚ := -1 / 6

-- Statement to prove
theorem opposite_of_number : -original_number = 1 / 6 := by
  -- This is where the proof would go
  sorry

end opposite_of_number_l39_3993


namespace xy_in_N_l39_3908

def M := {x : ℤ | ∃ m : ℤ, x = 3 * m + 1}
def N := {y : ℤ | ∃ n : ℤ, y = 3 * n + 2}

theorem xy_in_N (x y : ℤ) (hx : x ∈ M) (hy : y ∈ N) : (x * y) ∈ N :=
by
  sorry

end xy_in_N_l39_3908


namespace remainder_div_by_13_l39_3942

-- Define conditions
variable (N : ℕ)
variable (k : ℕ)

-- Given condition
def condition := N = 39 * k + 19

-- Goal statement
theorem remainder_div_by_13 (h : condition N k) : N % 13 = 6 :=
sorry

end remainder_div_by_13_l39_3942


namespace line_equation_perpendicular_l39_3952

def is_perpendicular (a1 b1 a2 b2 : ℝ) : Prop :=
  a1 * a2 + b1 * b2 = 0

theorem line_equation_perpendicular (c : ℝ) :
  (∃ k : ℝ, x - 2 * y + k = 0) ∧ is_perpendicular 2 1 1 (-2) → x - 2 * y - 3 = 0 := by
  sorry

end line_equation_perpendicular_l39_3952


namespace binomial_coefficient_and_factorial_l39_3963

open Nat

/--
  Given:
    - The binomial coefficient definition: Nat.choose n k = n! / (k! * (n - k)!)
    - The factorial definition: Nat.factorial n = n * (n - 1) * ... * 1
  Prove:
    Nat.choose 60 3 * Nat.factorial 10 = 124467072000
-/
theorem binomial_coefficient_and_factorial :
  Nat.choose 60 3 * Nat.factorial 10 = 124467072000 :=
by
  sorry

end binomial_coefficient_and_factorial_l39_3963


namespace brick_height_l39_3929

variable {l w : ℕ} (SA : ℕ)

theorem brick_height (h : ℕ) (l_eq : l = 10) (w_eq : w = 4) (SA_eq : SA = 136) 
    (surface_area_eq : SA = 2 * (l * w + l * h + w * h)) : h = 2 :=
by
  sorry

end brick_height_l39_3929


namespace katy_books_l39_3916

theorem katy_books (x : ℕ) (h : x + 2 * x + (2 * x - 3) = 37) : x = 8 :=
by
  sorry

end katy_books_l39_3916


namespace problem1_problem2_l39_3973

-- Problem (I)
theorem problem1 (a b : ℝ) (h : a ≥ b ∧ b > 0) : 2 * a^3 - b^3 ≥ 2 * a * b^2 - a^2 * b := sorry

-- Problem (II)
theorem problem2 (a b c x y z : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < x ∧ 0 < y ∧ 0 < z)
  (h1 : a^2 + b^2 + c^2 = 10) 
  (h2 : x^2 + y^2 + z^2 = 40) 
  (h3 : a * x + b * y + c * z = 20) : 
  (a + b + c) / (x + y + z) = 1 / 2 := sorry

end problem1_problem2_l39_3973


namespace division_result_is_correct_l39_3924

def division_result : ℚ := 132 / 6 / 3

theorem division_result_is_correct : division_result = 22 / 3 :=
by
  -- here, we would include the proof steps, but for now, we'll put sorry
  sorry

end division_result_is_correct_l39_3924


namespace beam_count_represents_number_of_beams_l39_3983

def price := 6210
def transport_cost_per_beam := 3
def beam_condition (x : ℕ) : Prop := 
  transport_cost_per_beam * x * (x - 1) = price

theorem beam_count_represents_number_of_beams (x : ℕ) :
  beam_condition x → (∃ n : ℕ, x = n) := 
sorry

end beam_count_represents_number_of_beams_l39_3983


namespace neighbor_to_johnson_yield_ratio_l39_3958

-- Definitions
def johnsons_yield (months : ℕ) : ℕ := 80 * (months / 2)
def neighbors_yield_per_hectare (x : ℕ) (months : ℕ) : ℕ := 80 * x * (months / 2)
def total_neighor_yield (x : ℕ) (months : ℕ) : ℕ := 2 * neighbors_yield_per_hectare x months

-- Theorem statement
theorem neighbor_to_johnson_yield_ratio
  (x : ℕ)
  (h1 : johnsons_yield 6 = 240)
  (h2 : total_neighor_yield x 6 = 480 * x)
  (h3 : johnsons_yield 6 + total_neighor_yield x 6 = 1200)
  : x = 2 := by
sorry

end neighbor_to_johnson_yield_ratio_l39_3958


namespace cookies_count_l39_3968

theorem cookies_count :
  ∀ (Tom Lucy Millie Mike Frank : ℕ), 
  (Tom = 16) →
  (Lucy = Nat.sqrt Tom) →
  (Millie = 2 * Lucy) →
  (Mike = 3 * Millie) →
  (Frank = Mike / 2 - 3) →
  Frank = 9 :=
by
  intros Tom Lucy Millie Mike Frank hTom hLucy hMillie hMike hFrank
  have h1 : Tom = 16 := hTom
  have h2 : Lucy = Nat.sqrt Tom := hLucy
  have h3 : Millie = 2 * Lucy := hMillie
  have h4 : Mike = 3 * Millie := hMike
  have h5 : Frank = Mike / 2 - 3 := hFrank
  sorry

end cookies_count_l39_3968
