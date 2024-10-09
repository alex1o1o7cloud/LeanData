import Mathlib

namespace first_train_speed_l1352_135234

noncomputable def speed_first_train (length1 length2 : ℝ) (speed2 time : ℝ) : ℝ :=
  let distance := (length1 + length2) / 1000
  let time_hours := time / 3600
  (distance / time_hours) - speed2

theorem first_train_speed :
  speed_first_train 100 280 30 18.998480121590273 = 42 :=
by
  sorry

end first_train_speed_l1352_135234


namespace no_solution_to_system_l1352_135254

theorem no_solution_to_system :
  ¬ ∃ (x y : ℝ), (3 * x - 4 * y = 8) ∧ (6 * x - 8 * y = 18) :=
by
  sorry

end no_solution_to_system_l1352_135254


namespace ratio_of_arithmetic_sequence_sums_l1352_135212

theorem ratio_of_arithmetic_sequence_sums :
  let a1 := 2
  let d1 := 3
  let l1 := 41
  let n1 := (l1 - a1) / d1 + 1
  let sum1 := n1 / 2 * (a1 + l1)

  let a2 := 4
  let d2 := 4
  let l2 := 60
  let n2 := (l2 - a2) / d2 + 1
  let sum2 := n2 / 2 * (a2 + l2)
  sum1 / sum2 = 301 / 480 :=
by
  sorry

end ratio_of_arithmetic_sequence_sums_l1352_135212


namespace new_table_capacity_is_six_l1352_135246

-- Definitions based on the conditions
def total_tables : ℕ := 40
def extra_new_tables : ℕ := 12
def total_customers : ℕ := 212
def original_table_capacity : ℕ := 4

-- Main statement to prove
theorem new_table_capacity_is_six (O N C : ℕ) 
  (h1 : O + N = total_tables)
  (h2 : N = O + extra_new_tables)
  (h3 : O * original_table_capacity + N * C = total_customers) :
  C = 6 :=
sorry

end new_table_capacity_is_six_l1352_135246


namespace shaded_percentage_seven_by_seven_grid_l1352_135298

theorem shaded_percentage_seven_by_seven_grid :
  let total_squares := 49
  let shaded_squares := 7
  let shaded_fraction := shaded_squares / total_squares
  let shaded_percentage := shaded_fraction * 100
  shaded_percentage = 14.29 := by
  sorry

end shaded_percentage_seven_by_seven_grid_l1352_135298


namespace kimiko_watched_4_videos_l1352_135223

/-- Kimiko's videos. --/
def first_video_length := 120
def second_video_length := 270
def last_two_video_length := 60
def total_time_watched := 510

theorem kimiko_watched_4_videos :
  first_video_length + second_video_length + last_two_video_length + last_two_video_length = total_time_watched → 
  4 = 4 :=
by
  intro h
  sorry

end kimiko_watched_4_videos_l1352_135223


namespace shelby_initial_money_l1352_135269

-- Definitions based on conditions
def cost_of_first_book : ℕ := 8
def cost_of_second_book : ℕ := 4
def cost_of_each_poster : ℕ := 4
def number_of_posters : ℕ := 2

-- Number to prove (initial money)
def initial_money : ℕ := 20

-- Theorem statement
theorem shelby_initial_money :
    (cost_of_first_book + cost_of_second_book + (number_of_posters * cost_of_each_poster)) = initial_money := by
    sorry

end shelby_initial_money_l1352_135269


namespace transformations_map_figure_l1352_135292

noncomputable def count_transformations : ℕ := sorry

theorem transformations_map_figure :
  count_transformations = 3 :=
sorry

end transformations_map_figure_l1352_135292


namespace rebecca_end_of_day_money_eq_l1352_135218

-- Define the costs for different services
def haircut_cost   := 30
def perm_cost      := 40
def dye_job_cost   := 60
def extension_cost := 80

-- Define the supply costs for the services
def haircut_supply_cost   := 5
def dye_job_supply_cost   := 10
def extension_supply_cost := 25

-- Today's appointments
def num_haircuts   := 5
def num_perms      := 3
def num_dye_jobs   := 2
def num_extensions := 1

-- Additional incomes and expenses
def tips           := 75
def daily_expenses := 45

-- Calculate the total earnings and costs
def total_service_revenue : ℕ := 
  num_haircuts * haircut_cost +
  num_perms * perm_cost +
  num_dye_jobs * dye_job_cost +
  num_extensions * extension_cost

def total_revenue : ℕ := total_service_revenue + tips

def total_supply_cost : ℕ := 
  num_haircuts * haircut_supply_cost +
  num_dye_jobs * dye_job_supply_cost +
  num_extensions * extension_supply_cost

def end_of_day_money : ℕ := total_revenue - total_supply_cost - daily_expenses

-- Lean statement to prove Rebecca will have $430 at the end of the day
theorem rebecca_end_of_day_money_eq : end_of_day_money = 430 := by
  sorry

end rebecca_end_of_day_money_eq_l1352_135218


namespace simplified_sum_l1352_135259

theorem simplified_sum :
  (-1 : ℤ) ^ 2002 + (-1 : ℤ) ^ 2003 + 2 ^ 2004 - 2 ^ 2003 = 2 ^ 2003 := 
by 
  sorry -- Proof skipped

end simplified_sum_l1352_135259


namespace least_number_of_homeowners_l1352_135204

theorem least_number_of_homeowners (total_members : ℕ) 
(num_men : ℕ) (num_women : ℕ) 
(homeowners_men : ℕ) (homeowners_women : ℕ) 
(h_total : total_members = 5000)
(h_men_women : num_men + num_women = total_members) 
(h_percentage_men : homeowners_men = 15 * num_men / 100)
(h_percentage_women : homeowners_women = 25 * num_women / 100):
  homeowners_men + homeowners_women = 4 :=
sorry

end least_number_of_homeowners_l1352_135204


namespace tricia_age_l1352_135299

theorem tricia_age :
  ∀ (T A Y E K R V : ℕ),
    T = 1 / 3 * A →
    A = 1 / 4 * Y →
    Y = 2 * E →
    K = 1 / 3 * E →
    R = K + 10 →
    R = V - 2 →
    V = 22 →
    T = 5 :=
by sorry

end tricia_age_l1352_135299


namespace pyramid_x_value_l1352_135208

theorem pyramid_x_value (x y : ℝ) 
  (h1 : 150 = 10 * x)
  (h2 : 225 = x * 15)
  (h3 : 1800 = 150 * y * 225) :
  x = 15 :=
sorry

end pyramid_x_value_l1352_135208


namespace biology_marks_l1352_135288

theorem biology_marks 
  (e m p c : ℤ) 
  (avg : ℚ) 
  (marks_biology : ℤ)
  (h1 : e = 70) 
  (h2 : m = 63) 
  (h3 : p = 80)
  (h4 : c = 63)
  (h5 : avg = 68.2) 
  (h6 : avg * 5 = (e + m + p + c + marks_biology)) : 
  marks_biology = 65 :=
sorry

end biology_marks_l1352_135288


namespace total_bulbs_needed_l1352_135283

-- Definitions according to the conditions.
variables (T S M L XL : ℕ)

-- Conditions
variables (cond1 : L = 2 * M)
variables (cond2 : S = 5 * M / 4)  -- since 1.25M = 5/4M
variables (cond3 : XL = S - T)
variables (cond4 : 4 * T = 3 * M) -- equivalent to T / M = 3 / 4
variables (cond5 : 2 * S + 3 * M = 4 * L + 5 * XL)
variables (cond6 : XL = 14)

-- Prove total bulbs needed
theorem total_bulbs_needed :
  T + 2 * S + 3 * M + 4 * L + 5 * XL = 469 :=
sorry

end total_bulbs_needed_l1352_135283


namespace smallest_integer_inequality_l1352_135255

theorem smallest_integer_inequality :
  (∃ n : ℤ, ∀ x y z : ℝ, (x + y + z)^2 ≤ (n:ℝ) * (x^2 + y^2 + z^2)) ∧
  ∀ m : ℤ, (∀ x y z : ℝ, (x + y + z)^2 ≤ (m:ℝ) * (x^2 + y^2 + z^2)) → 3 ≤ m :=
  sorry

end smallest_integer_inequality_l1352_135255


namespace maximize_xyplusxzplusyzplusy2_l1352_135231

theorem maximize_xyplusxzplusyzplusy2 (x y z : ℝ) (h1 : x + 2 * y + z = 7) (h2 : y ≥ 0) :
  xy + xz + yz + y^2 ≤ 10.5 :=
sorry

end maximize_xyplusxzplusyzplusy2_l1352_135231


namespace angle_between_north_and_south_southeast_l1352_135210

-- Given a circular floor pattern with 12 equally spaced rays
def num_rays : ℕ := 12
def total_degrees : ℕ := 360

-- Proving each central angle measure
def central_angle_measure : ℕ := total_degrees / num_rays

-- Define rays of interest
def segments_between_rays : ℕ := 5

-- Prove the angle between the rays pointing due North and South-Southeast
theorem angle_between_north_and_south_southeast :
  (segments_between_rays * central_angle_measure) = 150 := by
  sorry

end angle_between_north_and_south_southeast_l1352_135210


namespace sum_of_first_and_last_l1352_135219

noncomputable section

variables {A B C D E F G H I : ℕ}

theorem sum_of_first_and_last :
  (D = 8) →
  (A + B + C + D = 50) →
  (B + C + D + E = 50) →
  (C + D + E + F = 50) →
  (D + E + F + G = 50) →
  (E + F + G + H = 50) →
  (F + G + H + I = 50) →
  (A + I = 92) :=
by
  intros hD h1 h2 h3 h4 h5 h6
  sorry

end sum_of_first_and_last_l1352_135219


namespace smallest_positive_angle_l1352_135289

theorem smallest_positive_angle :
  ∃ y : ℝ, 0 < y ∧ y < 90 ∧ (6 * Real.sin y * (Real.cos y)^3 - 6 * (Real.sin y)^3 * Real.cos y = 3 / 2) ∧ y = 22.5 :=
by
  sorry

end smallest_positive_angle_l1352_135289


namespace find_f_neg12_add_f_14_l1352_135225

noncomputable def f (x : ℝ) : ℝ := 1 + Real.log (Real.sqrt (x^2 - 2*x + 2) - x + 1)

theorem find_f_neg12_add_f_14 : f (-12) + f 14 = 2 :=
by
  -- The hard part, the actual proof, is left as sorry.
  sorry

end find_f_neg12_add_f_14_l1352_135225


namespace c_difference_correct_l1352_135227

noncomputable def find_c_difference (a b c : ℝ) (h1 : a + b + c = 3) (h2 : a^2 + b^2 + c^2 = 20) : ℝ :=
  2 * Real.sqrt 34

theorem c_difference_correct (a b c : ℝ) (h1 : a + b + c = 3) (h2 : a^2 + b^2 + c^2 = 20) :
  find_c_difference a b c h1 h2 = 2 * Real.sqrt 34 := 
sorry

end c_difference_correct_l1352_135227


namespace value_of_a_l1352_135207

noncomputable def M : Set ℝ := {x | x^2 = 2}
noncomputable def N (a : ℝ) : Set ℝ := {x | a*x = 1}

theorem value_of_a (a : ℝ) : N a ⊆ M → a = 0 ∨ a = -Real.sqrt 2 / 2 ∨ a = Real.sqrt 2 / 2 :=
by
  intro h
  sorry

end value_of_a_l1352_135207


namespace archie_needs_sod_l1352_135209

-- Define the dimensions of the backyard
def backyard_length : ℕ := 20
def backyard_width : ℕ := 13

-- Define the dimensions of the shed
def shed_length : ℕ := 3
def shed_width : ℕ := 5

-- Statement: Prove that the area of the backyard minus the area of the shed equals 245 square yards
theorem archie_needs_sod : 
  backyard_length * backyard_width - shed_length * shed_width = 245 := 
by sorry

end archie_needs_sod_l1352_135209


namespace mike_weekly_avg_time_l1352_135285

theorem mike_weekly_avg_time :
  let mon_wed_fri_tv := 4 -- hours per day on Mon, Wed, Fri
  let tue_thu_tv := 3 -- hours per day on Tue, Thu
  let weekend_tv := 5 -- hours per day on weekends
  let num_mon_wed_fri := 3 -- days
  let num_tue_thu := 2 -- days
  let num_weekend := 2 -- days
  let num_days_week := 7 -- days
  let num_video_game_days := 3 -- days
  let weeks := 4 -- weeks
  let mon_wed_fri_total := mon_wed_fri_tv * num_mon_wed_fri
  let tue_thu_total := tue_thu_tv * num_tue_thu
  let weekend_total := weekend_tv * num_weekend
  let weekly_tv_time := mon_wed_fri_total + tue_thu_total + weekend_total
  let daily_avg_tv_time := weekly_tv_time / num_days_week
  let daily_video_game_time := daily_avg_tv_time / 2
  let weekly_video_game_time := daily_video_game_time * num_video_game_days
  let total_tv_time_4_weeks := weekly_tv_time * weeks
  let total_video_game_time_4_weeks := weekly_video_game_time * weeks
  let total_time_4_weeks := total_tv_time_4_weeks + total_video_game_time_4_weeks
  let weekly_avg_time := total_time_4_weeks / weeks
  weekly_avg_time = 34 := sorry

end mike_weekly_avg_time_l1352_135285


namespace marla_drive_time_l1352_135296

theorem marla_drive_time (x : ℕ) (h_total : x + 70 + x = 110) : x = 20 :=
sorry

end marla_drive_time_l1352_135296


namespace tangent_parallel_l1352_135232

theorem tangent_parallel (a b : ℝ) 
  (h1 : b = (1 / 3) * a^3 - (1 / 2) * a^2 + 1) 
  (h2 : (a^2 - a) = 2) : 
  a = 2 ∨ a = -1 :=
by {
  -- proof skipped
  sorry
}

end tangent_parallel_l1352_135232


namespace coin_die_sum_probability_l1352_135233

theorem coin_die_sum_probability : 
  let coin_sides := [5, 15]
  let die_sides := [1, 2, 3, 4, 5, 6]
  let ben_age := 18
  (1 / 2 : ℚ) * (1 / 6 : ℚ) = 1 / 12 :=
by
  sorry

end coin_die_sum_probability_l1352_135233


namespace solve_cos_sin_eq_one_l1352_135217

open Real

theorem solve_cos_sin_eq_one (n : ℕ) (hn : n > 0) :
  {x : ℝ | cos x ^ n - sin x ^ n = 1} = {x : ℝ | ∃ k : ℤ, x = k * π} :=
by
  sorry

end solve_cos_sin_eq_one_l1352_135217


namespace prism_faces_eq_nine_l1352_135272

-- Define the condition: a prism with 21 edges
def prism_edges (n : ℕ) := n = 21

-- Define the number of sides on each polygonal base
def num_sides (L : ℕ) := 3 * L = 21

-- Define the total number of faces
def total_faces (F : ℕ) (L : ℕ) := F = L + 2

-- The theorem we want to prove
theorem prism_faces_eq_nine (n L F : ℕ) 
  (h1 : prism_edges n)
  (h2 : num_sides L)
  (h3 : total_faces F L) :
  F = 9 := 
sorry

end prism_faces_eq_nine_l1352_135272


namespace john_experience_when_mike_started_l1352_135247

-- Definitions from the conditions
variable (J O M : ℕ)
variable (h1 : J = 20) -- James currently has 20 years of experience
variable (h2 : O - 8 = 2 * (J - 8)) -- 8 years ago, John had twice as much experience as James
variable (h3 : J + O + M = 68) -- Combined experience is 68 years

-- Theorem to prove
theorem john_experience_when_mike_started : O - M = 16 := 
by
  -- Proof steps go here
  sorry

end john_experience_when_mike_started_l1352_135247


namespace trigonometric_identity_l1352_135273

open Real

theorem trigonometric_identity (α : ℝ) (h1 : cos α = -4 / 5) (h2 : π < α ∧ α < (3 * π / 2)) :
    (1 + tan (α / 2)) / (1 - tan (α / 2)) = -1 / 2 := by
  sorry

end trigonometric_identity_l1352_135273


namespace face_opposite_of_E_l1352_135213

-- Definitions of faces and their relationships
inductive Face : Type
| A | B | C | D | E | F | x

open Face

-- Adjacency relationship
def is_adjacent_to (f1 f2 : Face) : Prop :=
(f1 = x ∧ (f2 = A ∨ f2 = B ∨ f2 = C ∨ f2 = D)) ∨
(f2 = x ∧ (f1 = A ∨ f1 = B ∨ f1 = C ∨ f1 = D)) ∨
(f1 = E ∧ (f2 = A ∨ f2 = B ∨ f2 = C ∨ f2 = D)) ∨
(f2 = E ∧ (f1 = A ∨ f1 = B ∨ f1 = C ∨ f1 = D))

-- Non-adjacency relationship
def is_opposite (f1 f2 : Face) : Prop :=
∀ f : Face, is_adjacent_to f1 f → ¬ is_adjacent_to f2 f

-- Theorem to prove that F is opposite of E
theorem face_opposite_of_E : is_opposite E F :=
sorry

end face_opposite_of_E_l1352_135213


namespace correct_option_D_l1352_135293

theorem correct_option_D : -2 = -|-2| := 
by 
  sorry

end correct_option_D_l1352_135293


namespace area_difference_equal_28_5_l1352_135290

noncomputable def square_side_length (d: ℝ) : ℝ := d / Real.sqrt 2
noncomputable def square_area (d: ℝ) : ℝ := (square_side_length d) ^ 2
noncomputable def circle_radius (D: ℝ) : ℝ := D / 2
noncomputable def circle_area (D: ℝ) : ℝ := Real.pi * (circle_radius D) ^ 2
noncomputable def area_difference (d D : ℝ) : ℝ := |circle_area D - square_area d|

theorem area_difference_equal_28_5 :
  ∀ (d D : ℝ), d = 10 → D = 10 → area_difference d D = 28.5 :=
by
  intros d D hd hD
  rw [hd, hD]
  -- Remaining steps involve computing the known areas and their differences
  sorry

end area_difference_equal_28_5_l1352_135290


namespace intersection_of_A_and_B_l1352_135297

def A : Set ℝ := {x | ∃ y, y = Real.sqrt (4 - x)}
def B : Set ℝ := {x | x > 1}

theorem intersection_of_A_and_B :
  A ∩ B = {x | 1 < x ∧ x ≤ 4} :=
sorry

end intersection_of_A_and_B_l1352_135297


namespace triangle_rectangle_ratio_l1352_135235

-- Definitions of the perimeter conditions and the relationship between length and width of the rectangle.
def equilateral_triangle_side_length (t : ℕ) : Prop :=
  3 * t = 24

def rectangle_dimensions (l w : ℕ) : Prop :=
  2 * l + 2 * w = 24 ∧ l = 2 * w

-- The main theorem stating the desired ratio.
theorem triangle_rectangle_ratio (t l w : ℕ) 
  (ht : equilateral_triangle_side_length t) (hlw : rectangle_dimensions l w) : t / w = 2 :=
by
  sorry

end triangle_rectangle_ratio_l1352_135235


namespace servings_per_pie_l1352_135250

theorem servings_per_pie (serving_apples : ℝ) (guests : ℕ) (pies : ℕ) (apples_per_guest : ℝ)
  (H_servings: serving_apples = 1.5) 
  (H_guests: guests = 12)
  (H_pies: pies = 3)
  (H_apples_per_guest: apples_per_guest = 3) :
  (guests * apples_per_guest) / (serving_apples * pies) = 8 :=
by
  rw [H_servings, H_guests, H_pies, H_apples_per_guest]
  sorry

end servings_per_pie_l1352_135250


namespace sum_possible_values_l1352_135295

def abs_eq_2023 (a : ℤ) : Prop := abs a = 2023
def abs_eq_2022 (b : ℤ) : Prop := abs b = 2022
def greater_than (a b : ℤ) : Prop := a > b

theorem sum_possible_values (a b : ℤ) (h1 : abs_eq_2023 a) (h2 : abs_eq_2022 b) (h3 : greater_than a b) :
  a + b = 1 ∨ a + b = 4045 := 
sorry

end sum_possible_values_l1352_135295


namespace multiples_of_10_between_11_and_103_l1352_135281

def countMultiplesOf10 (lower_bound upper_bound : Nat) : Nat :=
  Nat.div (upper_bound - lower_bound) 10 + 1

theorem multiples_of_10_between_11_and_103 : 
  countMultiplesOf10 11 103 = 9 :=
by
  sorry

end multiples_of_10_between_11_and_103_l1352_135281


namespace bianca_points_earned_l1352_135266

-- Define the constants and initial conditions
def points_per_bag : ℕ := 5
def total_bags : ℕ := 17
def not_recycled_bags : ℕ := 8

-- Define a function to calculate the number of recycled bags
def recycled_bags (total: ℕ) (not_recycled: ℕ) : ℕ :=
  total - not_recycled

-- Define a function to calculate the total points earned
def total_points_earned (bags: ℕ) (points_per_bag: ℕ) : ℕ :=
  bags * points_per_bag

-- State the theorem
theorem bianca_points_earned : total_points_earned (recycled_bags total_bags not_recycled_bags) points_per_bag = 45 :=
by
  sorry

end bianca_points_earned_l1352_135266


namespace letters_with_both_l1352_135248

/-
In a certain alphabet, some letters contain a dot and a straight line. 
36 letters contain a straight line but do not contain a dot. 
The alphabet has 60 letters, all of which contain either a dot or a straight line or both. 
There are 4 letters that contain a dot but do not contain a straight line. 
-/
def L_no_D : ℕ := 36
def D_no_L : ℕ := 4
def total_letters : ℕ := 60

theorem letters_with_both (DL : ℕ) : 
  total_letters = D_no_L + L_no_D + DL → 
  DL = 20 :=
by
  intros h
  sorry

end letters_with_both_l1352_135248


namespace new_train_distance_l1352_135211

-- Given conditions
def distance_older_train : ℝ := 200
def percent_more : ℝ := 0.20

-- Conclusion to prove
theorem new_train_distance : (distance_older_train * (1 + percent_more)) = 240 := by
  -- Placeholder to indicate that we are skipping the actual proof steps
  sorry

end new_train_distance_l1352_135211


namespace valid_n_values_l1352_135243

theorem valid_n_values (n x y : ℤ) (h1 : n * (x - 3) = y + 3) (h2 : x + n = 3 * (y - n)) :
  n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 7 :=
sorry

end valid_n_values_l1352_135243


namespace initial_men_colouring_l1352_135215

theorem initial_men_colouring (M : ℕ) : 
  (∀ m : ℕ, ∀ d : ℕ, ∀ l : ℕ, m * d = 48 * 2 → 8 * 0.75 = 6 → M = 4) :=
by
  sorry

end initial_men_colouring_l1352_135215


namespace evaluate_f_at_2_l1352_135277

def f (x : ℝ) : ℝ := x^3 - 2*x^2 + 3*x - 1

theorem evaluate_f_at_2 : f 2 = 5 := by
  sorry

end evaluate_f_at_2_l1352_135277


namespace dad_real_age_l1352_135200

theorem dad_real_age (x : ℝ) (h : (5/7) * x = 35) : x = 49 :=
by
  sorry

end dad_real_age_l1352_135200


namespace range_of_m_value_of_m_l1352_135202

variables (m p x : ℝ)

-- Conditions: The quadratic equation x^2 - 2x + m - 1 = 0 must have two real roots.
def discriminant (m : ℝ) := (-2)^2 - 4 * 1 * (m - 1)

-- Part 1: Finding the range of values for m
theorem range_of_m (h : discriminant m ≥ 0) : m ≤ 2 := 
by sorry

-- Additional Condition: p is a real root of the equation x^2 - 2x + m - 1 = 0
def is_root (p m : ℝ) := p^2 - 2 * p + m - 1 = 0

-- Another condition: (p^2 - 2p + 3)(m + 4) = 7
def satisfies_condition (p m : ℝ) := (p^2 - 2 * p + 3) * (m + 4) = 7

-- Part 2: Finding the value of m given p is a real root and satisfies (p^2 - 2p + 3)(m + 4) = 7
theorem value_of_m (h1 : is_root p m) (h2 : satisfies_condition p m) : m = -3 := 
by sorry

end range_of_m_value_of_m_l1352_135202


namespace algebraic_expression_value_l1352_135221

theorem algebraic_expression_value (x : ℝ) (h : x^2 + 3*x - 1 = 0) :
  (x - 3)^2 - (2*x + 1)*(2*x - 1) - 3*x = 7 :=
sorry

end algebraic_expression_value_l1352_135221


namespace tickets_difference_l1352_135253

theorem tickets_difference :
  let tickets_won := 48.5
  let yoyo_cost := 11.7
  let keychain_cost := 6.3
  let plush_toy_cost := 16.2
  let total_cost := yoyo_cost + keychain_cost + plush_toy_cost
  let tickets_left := tickets_won - total_cost
  tickets_won - tickets_left = total_cost :=
by
  sorry

end tickets_difference_l1352_135253


namespace chess_team_boys_l1352_135206

variable {B G : ℕ}

theorem chess_team_boys
    (h1 : B + G = 30)
    (h2 : 1/3 * G + B = 18) :
    B = 12 :=
by
  sorry

end chess_team_boys_l1352_135206


namespace sum_of_coordinates_l1352_135205

-- Definitions based on conditions
variable (f k : ℝ → ℝ)
variable (h₁ : f 4 = 8)
variable (h₂ : ∀ x, k x = (f x) ^ 3)

-- Statement of the theorem
theorem sum_of_coordinates : 4 + k 4 = 516 := by
  -- Proof would go here
  sorry

end sum_of_coordinates_l1352_135205


namespace probability_heads_mod_coin_l1352_135214

theorem probability_heads_mod_coin (p : ℝ) (h : 20 * p ^ 3 * (1 - p) ^ 3 = 1 / 20) : p = (1 - Real.sqrt 0.6816) / 2 :=
by
  sorry

end probability_heads_mod_coin_l1352_135214


namespace prob_at_least_6_heads_eq_l1352_135274

-- define the number of coin flips
def n := 8

-- define the number of possible outcomes (2^n)
def total_outcomes := 2 ^ n

-- define the binomial coefficients for cases: 6 heads, 7 heads, 8 heads
def binom_8_6 := Nat.choose 8 6
def binom_8_7 := Nat.choose 8 7
def binom_8_8 := Nat.choose 8 8

-- calculate the favorable outcomes for at least 6 heads
def favorable_outcomes := binom_8_6 + binom_8_7 + binom_8_8

-- define the probability of getting at least 6 heads
def probability := (favorable_outcomes : ℚ) / total_outcomes

theorem prob_at_least_6_heads_eq : probability = 37 / 256 := by
  sorry

end prob_at_least_6_heads_eq_l1352_135274


namespace line_equation_intersections_l1352_135275

theorem line_equation_intersections (m b k : ℝ) (h1 : b ≠ 0) 
  (h2 : m * 2 + b = 7) (h3 : abs (k^2 + 8*k + 7 - (m*k + b)) = 4) :
  m = 6 ∧ b = -5 :=
by {
  sorry
}

end line_equation_intersections_l1352_135275


namespace exists_prime_not_dividing_difference_l1352_135260

theorem exists_prime_not_dividing_difference {m : ℕ} (hm : m ≠ 1) : 
  ∃ p : ℕ, Nat.Prime p ∧ ∀ n : ℕ, ¬ p ∣ (n^n - m) := 
sorry

end exists_prime_not_dividing_difference_l1352_135260


namespace no_monotonically_decreasing_l1352_135257

variable (f : ℝ → ℝ)

theorem no_monotonically_decreasing (x1 x2 : ℝ) (h1 : ∃ x1 x2, x1 < x2 ∧ f x1 ≤ f x2) : ∀ x1 x2, x1 < x2 → f x1 > f x2 → False :=
by
  intros x1 x2 h2 h3
  obtain ⟨a, b, h4, h5⟩ := h1
  have contra := h5
  sorry

end no_monotonically_decreasing_l1352_135257


namespace intersection_of_A_and_B_l1352_135256

variable (x y : ℝ)

def A := {y : ℝ | ∃ x > 1, y = Real.log x / Real.log 2}
def B := {y : ℝ | ∃ x > 1, y = (1 / 2) ^ x}

theorem intersection_of_A_and_B :
  (A ∩ B) = {y : ℝ | 0 < y ∧ y < 1 / 2} :=
by sorry

end intersection_of_A_and_B_l1352_135256


namespace solve_inequality_system_l1352_135237

theorem solve_inequality_system (x : ℝ) 
  (h1 : 2 * (x - 1) < x + 3)
  (h2 : (x + 1) / 3 - x < 3) : 
  -4 < x ∧ x < 5 := 
  sorry

end solve_inequality_system_l1352_135237


namespace symmetric_points_l1352_135228

theorem symmetric_points (m n : ℤ) (h1 : m - 1 = -3) (h2 : 1 = n - 1) : m + n = 0 := by
  sorry

end symmetric_points_l1352_135228


namespace largest_multiple_of_7_smaller_than_neg_55_l1352_135278

theorem largest_multiple_of_7_smaller_than_neg_55 : ∃ m : ℤ, m % 7 = 0 ∧ m < -55 ∧ ∀ n : ℤ, n % 7 = 0 → n < -55 → n ≤ m :=
sorry

end largest_multiple_of_7_smaller_than_neg_55_l1352_135278


namespace average_book_width_l1352_135240

-- Define the widths of the books as given in the problem conditions
def widths : List ℝ := [3, 7.5, 1.25, 0.75, 4, 12]

-- Define the number of books from the problem conditions
def num_books : ℝ := 6

-- We prove that the average width of the books is equal to 4.75
theorem average_book_width : (widths.sum / num_books) = 4.75 :=
by
  sorry

end average_book_width_l1352_135240


namespace find_number_l1352_135239

theorem find_number (x : ℤ) : 45 - (28 - (x - (15 - 16))) = 55 ↔ x = 37 :=
by
  sorry

end find_number_l1352_135239


namespace total_investment_amount_l1352_135241

-- Define the conditions
def total_interest_in_one_year : ℝ := 1023
def invested_at_6_percent : ℝ := 8200
def interest_rate_6_percent : ℝ := 0.06
def interest_rate_7_5_percent : ℝ := 0.075

-- Define the equation based on the conditions
def interest_from_6_percent_investment : ℝ := invested_at_6_percent * interest_rate_6_percent

def total_investment_is_correct (T : ℝ) : Prop :=
  let interest_from_7_5_percent_investment := (T - invested_at_6_percent) * interest_rate_7_5_percent
  interest_from_6_percent_investment + interest_from_7_5_percent_investment = total_interest_in_one_year

-- Statement to prove
theorem total_investment_amount : total_investment_is_correct 15280 :=
by
  unfold total_investment_is_correct
  unfold interest_from_6_percent_investment
  simp
  sorry

end total_investment_amount_l1352_135241


namespace taylor_one_basket_in_three_tries_l1352_135201

theorem taylor_one_basket_in_three_tries (P_no_make : ℚ) (h : P_no_make = 1/3) : 
  (∃ P_make : ℚ, P_make = 1 - P_no_make ∧ P_make * P_no_make * P_no_make * 3 = 2/9) := 
by
  sorry

end taylor_one_basket_in_three_tries_l1352_135201


namespace geometric_sequence_a3_l1352_135270

theorem geometric_sequence_a3 (q : ℝ) (a : ℕ → ℝ) (h1 : a 1 = 1) (h2 : a 5 = 4) (h3 : ∀ n, a n = a 1 * q ^ (n - 1)) : a 3 = 2 :=
by
  sorry

end geometric_sequence_a3_l1352_135270


namespace g_neither_even_nor_odd_l1352_135229

noncomputable def g (x : ℝ) : ℝ := 3 ^ (x^2 - 3) - |x| + Real.sin x

theorem g_neither_even_nor_odd : ∀ x : ℝ, g x ≠ g (-x) ∧ g x ≠ -g (-x) := 
by
  intro x
  sorry

end g_neither_even_nor_odd_l1352_135229


namespace cube_volume_from_surface_area_l1352_135291

theorem cube_volume_from_surface_area (A : ℕ) (h1 : A = 864) : 
  ∃ V : ℕ, V = 1728 :=
by
  sorry

end cube_volume_from_surface_area_l1352_135291


namespace walking_time_proof_l1352_135203

-- Define the conditions from the problem
def bus_ride : ℕ := 75
def train_ride : ℕ := 360
def total_trip_time : ℕ := 480

-- Define the walking time as variable
variable (W : ℕ)

-- State the theorem as a Lean statement
theorem walking_time_proof :
  bus_ride + W + 2 * W + train_ride = total_trip_time → W = 15 :=
by
  intros h
  sorry

end walking_time_proof_l1352_135203


namespace calculate_49_squared_l1352_135265

theorem calculate_49_squared : 
  ∀ (a b : ℕ), a = 50 → b = 2 → (a - b)^2 = a^2 - 2 * a * b + b^2 → (49^2 = 50^2 - 196) :=
by
  intro a b h1 h2 h3
  sorry

end calculate_49_squared_l1352_135265


namespace bonus_percentage_correct_l1352_135245

/-
Tom serves 10 customers per hour and works for 8 hours, earning 16 bonus points.
We need to find the percentage of bonus points per customer served.
-/

def customers_per_hour : ℕ := 10
def hours_worked : ℕ := 8
def total_bonus_points : ℕ := 16

def total_customers_served : ℕ := customers_per_hour * hours_worked
def bonus_percentage : ℕ := (total_bonus_points * 100) / total_customers_served

theorem bonus_percentage_correct : bonus_percentage = 20 := by
  sorry

end bonus_percentage_correct_l1352_135245


namespace amy_final_money_l1352_135222

theorem amy_final_money :
  let initial_money := 2
  let chore_payment := 5 * 13
  let birthday_gift := 3
  let toy_cost := 12
  let remaining_money := initial_money + chore_payment + birthday_gift - toy_cost
  let grandparents_reward := 2 * remaining_money
  remaining_money + grandparents_reward = 174 := 
by
  sorry

end amy_final_money_l1352_135222


namespace correct_operation_l1352_135280

theorem correct_operation (a b : ℝ) : (a^3 * b)^2 = a^6 * b^2 :=
sorry

end correct_operation_l1352_135280


namespace find_a_l1352_135238

noncomputable def f (x a : ℝ) : ℝ := x - Real.log (x + 2) + Real.exp (x - a) + 4 * Real.exp (a - x)

theorem find_a (a : ℝ) :
  (∃ x₀ : ℝ, f x₀ a = 3) → a = -Real.log 2 - 1 :=
by
  sorry

end find_a_l1352_135238


namespace min_value_x_plus_y_l1352_135267

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : 4 / x + 9 / y = 1) : x + y = 25 :=
sorry

end min_value_x_plus_y_l1352_135267


namespace calculateBooksRemaining_l1352_135261

noncomputable def totalBooksRemaining
    (initialBooks : ℕ)
    (n : ℕ)
    (a₁ : ℕ)
    (d : ℕ)
    (borrowedBooks : ℕ)
    (returnedBooks : ℕ) : ℕ :=
  let sumDonations := n * (2 * a₁ + (n - 1) * d) / 2
  let totalAfterDonations := initialBooks + sumDonations
  totalAfterDonations - borrowedBooks + returnedBooks

theorem calculateBooksRemaining :
  totalBooksRemaining 1000 15 2 2 350 270 = 1160 :=
by
  sorry

end calculateBooksRemaining_l1352_135261


namespace find_k_l1352_135249

theorem find_k (k : ℝ) (h : (2 * (7:ℝ)^2) + 3 * 7 - k = 0) : k = 119 := by
  sorry

end find_k_l1352_135249


namespace other_asymptote_l1352_135242

theorem other_asymptote (a b : ℝ) :
  (∀ x y : ℝ, y = 2 * x → y - b = a * (x - (-4))) ∧
  (∀ c d : ℝ, c = -4) →
  ∃ m b' : ℝ, m = -1/2 ∧ b' = -10 ∧ ∀ x y : ℝ, y = m * x + b' :=
by
  sorry

end other_asymptote_l1352_135242


namespace painters_workdays_l1352_135258

theorem painters_workdays (five_painters_days : ℝ) (four_painters_days : ℝ) : 
  (5 * five_painters_days = 9) → (4 * four_painters_days = 9) → (four_painters_days = 2.25) :=
by
  intros h1 h2
  sorry

end painters_workdays_l1352_135258


namespace problem_statement_l1352_135276

theorem problem_statement {n d : ℕ} (hn : 0 < n) (hd : 0 < d) (h1 : d ∣ n) (h2 : d^2 * n + 1 ∣ n^2 + d^2) :
  n = d^2 :=
sorry

end problem_statement_l1352_135276


namespace johns_equation_l1352_135263

theorem johns_equation (a b c d e : ℤ) (ha : a = 2) (hb : b = 3) 
  (hc : c = 4) (hd : d = 5) : 
  a - (b - (c * (d - e))) = a - b - c * d + e ↔ e = 8 := 
by
  sorry

end johns_equation_l1352_135263


namespace martha_painting_rate_l1352_135264

noncomputable def martha_square_feet_per_hour
  (width1 : ℕ) (width2 : ℕ) (height : ℕ) (coats : ℕ) (total_hours : ℕ) 
  (pair1_walls : ℕ) (pair2_walls : ℕ) : ℕ :=
  let pair1_total_area := width1 * height * pair1_walls
  let pair2_total_area := width2 * height * pair2_walls
  let total_area := pair1_total_area + pair2_total_area
  let total_paint_area := total_area * coats
  total_paint_area / total_hours

theorem martha_painting_rate :
  martha_square_feet_per_hour 12 16 10 3 42 2 2 = 40 :=
by
  -- Proof goes here
  sorry

end martha_painting_rate_l1352_135264


namespace average_age_of_group_l1352_135226

theorem average_age_of_group :
  let n_graders := 40
  let n_parents := 50
  let n_teachers := 10
  let avg_age_graders := 12
  let avg_age_parents := 35
  let avg_age_teachers := 45
  let total_individuals := n_graders + n_parents + n_teachers
  let total_age := n_graders * avg_age_graders + n_parents * avg_age_parents + n_teachers * avg_age_teachers
  (total_age : ℚ) / total_individuals = 26.8 :=
by
  sorry

end average_age_of_group_l1352_135226


namespace combined_votes_l1352_135224

theorem combined_votes {A B : ℕ} (h1 : A = 14) (h2 : 2 * B = A) : A + B = 21 := 
by 
sorry

end combined_votes_l1352_135224


namespace remaining_pages_l1352_135287

def original_book_pages : ℕ := 93
def pages_read_saturday : ℕ := 30
def pages_read_sunday : ℕ := 20

theorem remaining_pages :
  original_book_pages - (pages_read_saturday + pages_read_sunday) = 43 := by
  sorry

end remaining_pages_l1352_135287


namespace weight_of_replaced_student_l1352_135284

variable (W : ℝ) -- total weight of the original 10 students
variable (new_student_weight : ℝ := 60) -- weight of the new student
variable (weight_decrease_per_student : ℝ := 6) -- average weight decrease per student

theorem weight_of_replaced_student (replaced_student_weight : ℝ) :
  (W - replaced_student_weight + new_student_weight = W - 10 * weight_decrease_per_student) →
  replaced_student_weight = 120 := by
  sorry

end weight_of_replaced_student_l1352_135284


namespace nested_g_of_2_l1352_135230

def g (x : ℤ) : ℤ := x^2 - 4*x + 3

theorem nested_g_of_2 : g (g (g (g (g (g 2))))) = 1394486148248 := by
  sorry

end nested_g_of_2_l1352_135230


namespace determinant_value_l1352_135268

variable (a1 b1 b2 c1 c2 c3 d1 d2 d3 d4 : ℝ)

def matrix_det : ℝ :=
  Matrix.det ![
    ![a1, b1, c1, d1],
    ![a1, b2, c2, d2],
    ![a1, b2, c3, d3],
    ![a1, b2, c3, d4]
  ]

theorem determinant_value : 
  matrix_det a1 b1 b2 c1 c2 c3 d1 d2 d3 d4 = 
  a1 * (b2 - b1) * (c3 - c2) * (d4 - d3) :=
by
  sorry

end determinant_value_l1352_135268


namespace f_decreasing_on_positive_reals_f_greater_than_2_div_x_plus_2_l1352_135220

noncomputable def f (x : ℝ) : ℝ := if x > 0 then (Real.log (1 + x)) / x else 0

theorem f_decreasing_on_positive_reals :
  ∀ x y : ℝ, 0 < x → 0 < y → x < y → f y < f x :=
sorry

theorem f_greater_than_2_div_x_plus_2 :
  ∀ x : ℝ, 0 < x → f x > 2 / (x + 2) :=
sorry

end f_decreasing_on_positive_reals_f_greater_than_2_div_x_plus_2_l1352_135220


namespace exists_five_positive_integers_sum_20_product_420_l1352_135236
-- Import the entirety of Mathlib to ensure all necessary definitions are available

-- Lean statement for the proof problem
theorem exists_five_positive_integers_sum_20_product_420 :
  ∃ (a b c d e : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ a + b + c + d + e = 20 ∧ a * b * c * d * e = 420 :=
sorry

end exists_five_positive_integers_sum_20_product_420_l1352_135236


namespace find_max_value_l1352_135279

theorem find_max_value (f : ℝ → ℝ) (h₀ : f 0 = -5) (h₁ : ∀ x, deriv f x = 4 * x^3 - 4 * x) :
  ∃ x, f x = -5 ∧ (∀ y, f y ≤ f x) ∧ x = 0 :=
sorry

end find_max_value_l1352_135279


namespace fruit_prices_l1352_135244

theorem fruit_prices (x y : ℝ) 
  (h₁ : 3 * x + 2 * y = 40) 
  (h₂ : 2 * x + 3 * y = 35) : 
  x = 10 ∧ y = 5 :=
by
  sorry

end fruit_prices_l1352_135244


namespace soaked_part_solution_l1352_135216

theorem soaked_part_solution 
  (a b : ℝ) (c : ℝ) 
  (h : c * (2/3) * a * b = 2 * a^2 * b^3 + (1/3) * a^3 * b^2) :
  c = 3 * a * b^2 + (1/2) * a^2 * b :=
by
  sorry

end soaked_part_solution_l1352_135216


namespace modulus_of_complex_z_l1352_135286

open Complex

theorem modulus_of_complex_z (z : ℂ) (h : z * (2 - 3 * I) = 6 + 4 * I) : 
  Complex.abs z = 2 * Real.sqrt 313 / 13 :=
by
  sorry

end modulus_of_complex_z_l1352_135286


namespace at_least_one_angle_ge_60_l1352_135251

theorem at_least_one_angle_ge_60 (A B C : ℝ) (hA : A < 60) (hB : B < 60) (hC : C < 60) (h_sum : A + B + C = 180) : false :=
sorry

end at_least_one_angle_ge_60_l1352_135251


namespace geometric_sequence_S6_div_S3_l1352_135252

theorem geometric_sequence_S6_div_S3 (a : ℕ → ℝ) 
  (S : ℕ → ℝ)
  (h1 : a 1 + a 3 = 5 / 4)
  (h2 : a 2 + a 4 = 5 / 2)
  (hS : ∀ n, S n = a 1 * (1 - (2:ℝ) ^ n) / (1 - 2)) :
  S 6 / S 3 = 9 :=
by
  sorry

end geometric_sequence_S6_div_S3_l1352_135252


namespace minimum_dot_product_l1352_135271

noncomputable def ellipse (x y : ℝ) : Prop := (x^2 / 36) + (y^2 / 9) = 1

def K : (ℝ × ℝ) := (2, 0)

def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

def vector_sub (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 - v.1, u.2 - v.2)

theorem minimum_dot_product (M N : ℝ × ℝ) (hM : ellipse M.1 M.2) (hN : ellipse N.1 N.2) (h : dot_product (vector_sub M K) (vector_sub N K) = 0) :
  ∃ α β : ℝ, 0 ≤ α ∧ α < 2 * Real.pi ∧ 0 ≤ β ∧ β < 2 * Real.pi ∧ M = (6 * Real.cos α, 3 * Real.sin α) ∧ N = (6 * Real.cos β, 3 * Real.sin β) ∧
  (∃ C : ℝ, C = 23 / 3 ∧ ∀ M N, ellipse M.1 M.2 → ellipse N.1 N.2 → dot_product (vector_sub M K) (vector_sub N K) = 0 → dot_product (vector_sub M K) (vector_sub (vector_sub M N) K) >= C) :=
sorry

end minimum_dot_product_l1352_135271


namespace smallest_q_exists_l1352_135294

noncomputable def p_q_r_are_consecutive_terms (p q r : ℝ) : Prop :=
∃ d : ℝ, p = q - d ∧ r = q + d

theorem smallest_q_exists
  (p q r : ℝ)
  (h1 : p_q_r_are_consecutive_terms p q r)
  (h2 : p > 0) 
  (h3 : q > 0) 
  (h4 : r > 0)
  (h5 : p * q * r = 216) :
  q = 6 :=
sorry

end smallest_q_exists_l1352_135294


namespace tan_alpha_minus_pi_over_4_l1352_135282

theorem tan_alpha_minus_pi_over_4 
  (α β : ℝ) 
  (h1 : Real.tan (α + β) = 2) 
  (h2 : Real.tan (β + π/4) = 3) 
  : Real.tan (α - π/4) = -1 / 7 :=
by
  sorry

end tan_alpha_minus_pi_over_4_l1352_135282


namespace right_triangle_area_l1352_135262

theorem right_triangle_area
  (hypotenuse : ℝ) (leg1 : ℝ) (leg2 : ℝ)
  (hypotenuse_eq : hypotenuse = 13)
  (leg1_eq : leg1 = 5)
  (pythagorean_eq : hypotenuse^2 = leg1^2 + leg2^2) :
  (1 / 2) * leg1 * leg2 = 30 :=
by
  sorry

end right_triangle_area_l1352_135262
