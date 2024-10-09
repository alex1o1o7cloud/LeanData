import Mathlib

namespace problem_1_problem_2_l1937_193791

-- Definitions for set A and B when a = 3 for (1)
def A : Set ℝ := { x | -x^2 + 5 * x - 4 > 0 }
def B : Set ℝ := { x | x^2 - 5 * x + 6 ≤ 0 }

-- Theorem for (1)
theorem problem_1 : A ∪ (Bᶜ) = Set.univ := sorry

-- Function to describe B based on a for (2)
def B_a (a : ℝ) : Set ℝ := { x | x^2 - (a + 2) * x + 2 * a ≤ 0 }
def A_set : Set ℝ := { x | -x^2 + 5 * x - 4 > 0 }

-- Theorem for (2)
theorem problem_2 (a : ℝ) : (1 < a ∧ a < 4) → (A_set ∩ B_a a ≠ ∅ ∧ B_a a ⊆ A_set ∧ B_a a ≠ A_set) := sorry

end problem_1_problem_2_l1937_193791


namespace fraction_disliking_but_liking_l1937_193780

-- Definitions based on conditions
def total_students : ℕ := 100
def like_dancing : ℕ := 70
def dislike_dancing : ℕ := total_students - like_dancing

def say_they_like_dancing (like_dancing : ℕ) : ℕ := (70 * like_dancing) / 100
def say_they_dislike_dancing (like_dancing : ℕ) : ℕ := like_dancing - say_they_like_dancing like_dancing

def dislike_and_say_dislike (dislike_dancing : ℕ) : ℕ := (80 * dislike_dancing) / 100
def say_dislike_but_like (like_dancing : ℕ) : ℕ := say_they_dislike_dancing like_dancing

def total_say_dislike : ℕ := dislike_and_say_dislike dislike_dancing + say_dislike_but_like like_dancing

noncomputable def fraction_like_but_say_dislike : ℚ := (say_dislike_but_like like_dancing : ℚ) / (total_say_dislike : ℚ)

theorem fraction_disliking_but_liking : fraction_like_but_say_dislike = 46.67 / 100 := 
by sorry

end fraction_disliking_but_liking_l1937_193780


namespace system_solution_l1937_193746

theorem system_solution:
  let k := 115 / 12 
  ∃ x y z: ℝ, 
    x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ 
    (x + k * y + 5 * z = 0) ∧
    (4 * x + k * y - 3 * z = 0) ∧
    (3 * x + 5 * y - 4 * z = 0) ∧ 
    ((1 : ℝ) / 15 = (x * z) / (y * y)) := 
by sorry

end system_solution_l1937_193746


namespace cookies_batches_needed_l1937_193753

noncomputable def number_of_recipes (total_students : ℕ) (attendance_drop : ℝ) (cookies_per_batch : ℕ) : ℕ :=
  let remaining_students := (total_students : ℝ) * (1 - attendance_drop)
  let total_cookies := remaining_students * 2
  let recipes_needed := total_cookies / cookies_per_batch
  (Nat.ceil recipes_needed : ℕ)

theorem cookies_batches_needed :
  number_of_recipes 150 0.40 18 = 10 :=
by
  sorry

end cookies_batches_needed_l1937_193753


namespace part1_part2_l1937_193731

def set_A := {x : ℝ | x^2 + 2*x - 8 = 0}
def set_B (a : ℝ) := {x : ℝ | x^2 + 2*(a+1)*x + 2*a^2 - 2 = 0}

theorem part1 (a : ℝ) (h : a = 1) : 
  (set_A ∩ set_B a) = {-4} := by
  sorry

theorem part2 (a : ℝ) : 
  (set_A ∩ (set_B a) = set_B a) → (a < -1 ∨ a > 3) := by
  sorry

end part1_part2_l1937_193731


namespace exercise_l1937_193758

theorem exercise (x y z : ℝ)
  (h1 : x + y + z = 30)
  (h2 : x * y * z = 343)
  (h3 : 1/x + 1/y + 1/z = 3/5) : x^2 + y^2 + z^2 = 488.4 :=
sorry

end exercise_l1937_193758


namespace hawks_points_l1937_193786

theorem hawks_points (E H : ℕ) (h1 : E + H = 82) (h2 : E = H + 6) : H = 38 :=
sorry

end hawks_points_l1937_193786


namespace megatech_astrophysics_degrees_l1937_193774

theorem megatech_astrophysics_degrees :
  let microphotonics := 10
  let home_electronics := 24
  let food_additives := 15
  let gmo := 29
  let industrial_lubricants := 8
  let total_percentage := microphotonics + home_electronics + food_additives + gmo + industrial_lubricants
  let astrophysics_percentage := 100 - total_percentage
  let total_degrees := 360
  let astrophysics_degrees := (astrophysics_percentage / 100) * total_degrees
  astrophysics_degrees = 50.4 :=
by
  sorry

end megatech_astrophysics_degrees_l1937_193774


namespace sum_of_two_integers_l1937_193790

theorem sum_of_two_integers (x y : ℕ) (h₁ : x^2 + y^2 = 145) (h₂ : x * y = 40) : x + y = 15 := 
by
  -- Proof omitted
  sorry

end sum_of_two_integers_l1937_193790


namespace kathleen_money_left_l1937_193732

def june_savings : ℕ := 21
def july_savings : ℕ := 46
def august_savings : ℕ := 45

def school_supplies_expenses : ℕ := 12
def new_clothes_expenses : ℕ := 54

def total_savings : ℕ := june_savings + july_savings + august_savings
def total_expenses : ℕ := school_supplies_expenses + new_clothes_expenses

def total_money_left : ℕ := total_savings - total_expenses

theorem kathleen_money_left : total_money_left = 46 :=
by
  sorry

end kathleen_money_left_l1937_193732


namespace nonneg_reals_inequality_l1937_193717

theorem nonneg_reals_inequality (a b c d : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : 0 ≤ d) (h₄ : a^2 + b^2 + c^2 + d^2 = 1) : 
  a + b + c + d - 1 ≥ 16 * a * b * c * d := 
by 
  sorry

end nonneg_reals_inequality_l1937_193717


namespace smallest_n_l1937_193778

theorem smallest_n (a b c n : ℕ) (h1 : n = 100 * a + 10 * b + c)
  (h2 : n = a + b + c + a * b + b * c + a * c + a * b * c)
  (h3 : n >= 100 ∧ n < 1000)
  (h4 : a ≥ 1 ∧ a ≤ 9)
  (h5 : b ≥ 0 ∧ b ≤ 9)
  (h6 : c ≥ 0 ∧ c ≤ 9) :
  n = 199 :=
sorry

end smallest_n_l1937_193778


namespace basketball_third_quarter_points_l1937_193764

noncomputable def teamA_points (a r : ℕ) : ℕ :=
a + a*r + a*r^2 + a*r^3

noncomputable def teamB_points (b d : ℕ) : ℕ :=
b + (b + d) + (b + 2*d) + (b + 3*d)

theorem basketball_third_quarter_points (a b d : ℕ) (r : ℕ) 
    (h1 : r > 1) (h2 : d > 0) (h3 : a * (r^4 - 1) / (r - 1) = 4 * b + 6 * d + 3)
    (h4 : a * (r^4 - 1) / (r - 1) ≤ 100) (h5 : 4 * b + 6 * d ≤ 100) :
    a * r^2 + b + 2 * d = 60 :=
sorry

end basketball_third_quarter_points_l1937_193764


namespace cylinder_volume_l1937_193735

theorem cylinder_volume (r h : ℝ) (h_radius : r = 1) (h_height : h = 2) : (π * r^2 * h) = 2 * π :=
by
  sorry

end cylinder_volume_l1937_193735


namespace larger_root_of_quadratic_eq_l1937_193733

theorem larger_root_of_quadratic_eq : 
  ∀ x : ℝ, x^2 - 13 * x + 36 = 0 → x = 9 ∨ x = 4 := by
  sorry

end larger_root_of_quadratic_eq_l1937_193733


namespace count_equilateral_triangles_in_hexagonal_lattice_l1937_193710

-- Definitions based on conditions in problem (hexagonal lattice setup)
def hexagonal_lattice (dist : ℕ) : Prop :=
  -- Define properties of the points in hexagonal lattice
  -- Placeholder for actual structure defining the hexagon and surrounding points
  sorry

def equilateral_triangles (n : ℕ) : Prop :=
  -- Define a method to count equilateral triangles in the given lattice setup
  sorry

-- Theorem stating that 10 equilateral triangles can be formed in the lattice
theorem count_equilateral_triangles_in_hexagonal_lattice (dist : ℕ) (h : dist = 1 ∨ dist = 2) :
  equilateral_triangles 10 :=
by
  -- Proof to be completed
  sorry

end count_equilateral_triangles_in_hexagonal_lattice_l1937_193710


namespace hybrids_with_full_headlights_l1937_193716

-- Definitions for the conditions
def total_cars : ℕ := 600
def percentage_hybrids : ℕ := 60
def percentage_one_headlight : ℕ := 40

-- The proof statement
theorem hybrids_with_full_headlights :
  (percentage_hybrids * total_cars) / 100 - (percentage_one_headlight * (percentage_hybrids * total_cars) / 100) / 100 = 216 :=
by
  sorry

end hybrids_with_full_headlights_l1937_193716


namespace arthur_muffins_l1937_193704

variable (arthur_baked : ℕ)
variable (james_baked : ℕ := 1380)
variable (times_as_many : ℕ := 12)

theorem arthur_muffins : arthur_baked * times_as_many = james_baked -> arthur_baked = 115 := by
  sorry

end arthur_muffins_l1937_193704


namespace exponentiation_problem_l1937_193772

theorem exponentiation_problem : 10^6 * (10^2)^3 / 10^4 = 10^8 := 
by 
  sorry

end exponentiation_problem_l1937_193772


namespace even_function_has_specific_m_l1937_193797

theorem even_function_has_specific_m (m : ℝ) (f : ℝ → ℝ) (h_def : ∀ x : ℝ, f x = x^2 + (m - 1) * x - 3) (h_even : ∀ x : ℝ, f x = f (-x)) :
  m = 1 :=
by
  sorry

end even_function_has_specific_m_l1937_193797


namespace water_supply_days_l1937_193741

theorem water_supply_days (C V : ℕ) 
  (h1: C = 75 * (V + 10))
  (h2: C = 60 * (V + 20)) : 
  (C / V) = 100 := 
sorry

end water_supply_days_l1937_193741


namespace right_triangle_inequality_l1937_193720

theorem right_triangle_inequality (a b c : ℝ) (h : a^2 + b^2 = c^2) :
  a^4 + b^4 < c^4 :=
by
  sorry

end right_triangle_inequality_l1937_193720


namespace additional_rocks_needed_l1937_193756

-- Define the dimensions of the garden
def length (garden : Type) : ℕ := 15
def width (garden : Type) : ℕ := 10
def rock_cover (rock : Type) : ℕ := 1

-- Define the number of rocks Mrs. Hilt has
def rocks_possessed (mrs_hilt : Type) : ℕ := 64

-- Define the perimeter of the garden
def perimeter (garden : Type) : ℕ :=
  2 * (length garden + width garden)

-- Define the number of rocks required for the first layer
def rocks_first_layer (garden : Type) : ℕ :=
  perimeter garden

-- Define the number of rocks required for the second layer (only longer sides)
def rocks_second_layer (garden : Type) : ℕ :=
  2 * length garden

-- Define the total number of rocks needed
def total_rocks_needed (garden : Type) : ℕ :=
  rocks_first_layer garden + rocks_second_layer garden

-- Prove the number of additional rocks Mrs. Hilt needs
theorem additional_rocks_needed (garden : Type) (mrs_hilt : Type):
  total_rocks_needed garden - rocks_possessed mrs_hilt = 16 := by
  sorry

end additional_rocks_needed_l1937_193756


namespace total_stickers_l1937_193714

theorem total_stickers :
  (20.0 : ℝ) + (26.0 : ℝ) + (20.0 : ℝ) + (6.0 : ℝ) + (58.0 : ℝ) = 130.0 := by
  sorry

end total_stickers_l1937_193714


namespace sales_difference_l1937_193799
noncomputable def max_min_difference (sales : List ℕ) : ℕ :=
  (sales.maximum.getD 0) - (sales.minimum.getD 0)

theorem sales_difference :
  max_min_difference [1200, 1450, 1950, 1700] = 750 :=
by
  sorry

end sales_difference_l1937_193799


namespace geometric_sequence_l1937_193776

-- Define the set and its properties
variable (A : Set ℕ) (a : ℕ → ℕ) (n : ℕ)
variable (h1 : 1 ≤ a 1) 
variable (h2 : ∀ (i : ℕ), 1 ≤ i → i < n → a i < a (i + 1))
variable (h3 : n ≥ 5)
variable (h4 : ∀ (i j : ℕ), 1 ≤ i → i ≤ j → j ≤ n → (a i) * (a j) ∈ A ∨ (a i) / (a j) ∈ A)

-- Statement to prove that the sequence forms a geometric sequence
theorem geometric_sequence : 
  ∃ (c : ℕ), c > 1 ∧ ∀ (i : ℕ), 1 ≤ i → i ≤ n → a i = c^(i-1) := sorry

end geometric_sequence_l1937_193776


namespace optimal_sampling_methods_l1937_193702

/-
We define the conditions of the problem.
-/
def households := 500
def high_income_households := 125
def middle_income_households := 280
def low_income_households := 95
def sample_households := 100

def soccer_players := 12
def sample_soccer_players := 3

/-
We state the goal as a theorem.
-/
theorem optimal_sampling_methods :
  (sample_households == 100) ∧
  (sample_soccer_players == 3) ∧
  (high_income_households + middle_income_households + low_income_households == households) →
  ("stratified" = "stratified" ∧ "random" = "random") :=
by
  -- Sorry to skip the proof
  sorry

end optimal_sampling_methods_l1937_193702


namespace problem_solution_l1937_193745

noncomputable def time_without_distraction : ℝ :=
  let rate_A := 1 / 10
  let rate_B := 0.75 * rate_A
  let rate_C := 0.5 * rate_A
  let combined_rate := rate_A + rate_B + rate_C
  1 / combined_rate

noncomputable def time_with_distraction : ℝ :=
  let rate_A := 0.9 * (1 / 10)
  let rate_B := 0.9 * (0.75 * (1 / 10))
  let rate_C := 0.9 * (0.5 * (1 / 10))
  let combined_rate := rate_A + rate_B + rate_C
  1 / combined_rate

theorem problem_solution :
  time_without_distraction = 40 / 9 ∧
  time_with_distraction = 44.44 / 9 := by
  sorry

end problem_solution_l1937_193745


namespace equation_is_true_l1937_193707

theorem equation_is_true :
  10 * 6 - (9 - 3) * 2 = 48 :=
by
  sorry

end equation_is_true_l1937_193707


namespace new_ratio_after_adding_ten_l1937_193738

theorem new_ratio_after_adding_ten 
  (x : ℕ) 
  (h_ratio : 3 * x = 15) 
  (new_smaller : ℕ := x + 10) 
  (new_larger : ℕ := 15) 
  : new_smaller / new_larger = 1 :=
by sorry

end new_ratio_after_adding_ten_l1937_193738


namespace minimum_time_to_serve_tea_equals_9_l1937_193729

def boiling_water_time : Nat := 8
def washing_teapot_time : Nat := 1
def washing_teacups_time : Nat := 2
def fetching_tea_leaves_time : Nat := 2
def brewing_tea_time : Nat := 1

theorem minimum_time_to_serve_tea_equals_9 :
  boiling_water_time + brewing_tea_time = 9 := by
  sorry

end minimum_time_to_serve_tea_equals_9_l1937_193729


namespace inequality_holds_l1937_193743

theorem inequality_holds (a b : ℝ) (h1 : -1 < a) (h2 : a < b) (h3 : b < 1) :
  -2 < a - b ∧ a - b < 0 := 
by
  sorry

end inequality_holds_l1937_193743


namespace largest_consecutive_odd_number_sum_75_l1937_193737

theorem largest_consecutive_odd_number_sum_75 (a b c : ℤ) 
    (h1 : a + b + c = 75) 
    (h2 : b = a + 2) 
    (h3 : c = b + 2) : 
    c = 27 :=
by
  sorry

end largest_consecutive_odd_number_sum_75_l1937_193737


namespace circle_center_coordinates_l1937_193718

theorem circle_center_coordinates :
  ∃ (h k : ℝ), (x^2 + y^2 - 2*x + 4*y = 0) → (x - h)^2 + (y + k)^2 = 5 :=
sorry

end circle_center_coordinates_l1937_193718


namespace cuboid_second_edge_l1937_193715

variable (x : ℝ)

theorem cuboid_second_edge (h1 : 4 * x * 6 = 96) : x = 4 := by
  sorry

end cuboid_second_edge_l1937_193715


namespace root_of_quadratic_l1937_193721

theorem root_of_quadratic (x m : ℝ) (h : x = -1 ∧ x^2 + m*x - 1 = 0) : m = 0 :=
sorry

end root_of_quadratic_l1937_193721


namespace find_width_of_room_l1937_193725

theorem find_width_of_room
    (length : ℝ) (area : ℝ)
    (h1 : length = 12) (h2 : area = 96) :
    ∃ width : ℝ, width = 8 ∧ area = length * width :=
by
  sorry

end find_width_of_room_l1937_193725


namespace range_of_a_l1937_193785

theorem range_of_a (a b c : ℝ) 
  (h1 : a^2 - b*c - 8*a + 7 = 0)
  (h2 : b^2 + c^2 + b*c - 6*a + 6 = 0) : 
  1 ≤ a ∧ a ≤ 9 :=
sorry

end range_of_a_l1937_193785


namespace part1_complement_intersection_part2_range_m_l1937_193763

open Set

-- Define set A
def A : Set ℝ := { x | -1 ≤ x ∧ x < 4 }

-- Define set B parameterized by m
def B (m : ℝ) : Set ℝ := { x | m ≤ x ∧ x ≤ m + 2 }

-- Part (1): Prove the complement of the intersection for m = 3
theorem part1_complement_intersection :
  ∀ x : ℝ, x ∉ (A ∩ B 3) ↔ x < 3 ∨ x ≥ 4 :=
by
  sorry

-- Part (2): Prove the range of m for A ∩ B = ∅
theorem part2_range_m (m : ℝ) :
  (A ∩ B m = ∅) ↔ m < -3 ∨ m ≥ 4 :=
by
  sorry

end part1_complement_intersection_part2_range_m_l1937_193763


namespace range_of_function_l1937_193767

theorem range_of_function : 
  ∀ y : ℝ, 
  (∃ x : ℝ, y = x^2 + 1) ↔ (y ≥ 1) :=
by
  sorry

end range_of_function_l1937_193767


namespace speed_W_B_l1937_193784

-- Definitions for the conditions
def distance_W_B (D : ℝ) := 2 * D
def average_speed := 36
def speed_B_C := 20

-- The problem statement to be verified in Lean
theorem speed_W_B (D : ℝ) (S : ℝ) (h1: distance_W_B D = 2 * D) (h2: S ≠ 0 ∧ D ≠ 0)
(h3: (3 * D) / ((2 * D) / S + D / speed_B_C) = average_speed) : S = 60 := by
sorry

end speed_W_B_l1937_193784


namespace sum_of_cubics_l1937_193726

noncomputable def root_polynomial (x : ℝ) := 5 * x^3 + 2003 * x + 3005

theorem sum_of_cubics (a b c : ℝ)
  (h1 : root_polynomial a = 0)
  (h2 : root_polynomial b = 0)
  (h3 : root_polynomial c = 0) :
  (a + b)^3 + (b + c)^3 + (c + a)^3 = 1803 :=
sorry

end sum_of_cubics_l1937_193726


namespace table_height_l1937_193789

theorem table_height (r s x y l : ℝ)
  (h1 : x + l - y = 32)
  (h2 : y + l - x = 28) :
  l = 30 :=
by
  sorry

end table_height_l1937_193789


namespace johnny_worked_hours_l1937_193734

theorem johnny_worked_hours (total_earned hourly_wage hours_worked : ℝ) 
(h1 : total_earned = 16.5) (h2 : hourly_wage = 8.25) (h3 : total_earned / hourly_wage = hours_worked) : 
hours_worked = 2 := 
sorry

end johnny_worked_hours_l1937_193734


namespace new_boarders_day_scholars_ratio_l1937_193705

theorem new_boarders_day_scholars_ratio
  (initial_boarders : ℕ)
  (initial_day_scholars : ℕ)
  (ratio_boarders_day_scholars : ℕ → ℕ → Prop)
  (additional_boarders : ℕ)
  (new_boarders : ℕ)
  (new_ratio : ℕ → ℕ → Prop)
  (r1 r2 : ℕ)
  (h1 : ratio_boarders_day_scholars 7 16)
  (h2 : initial_boarders = 560)
  (h3 : initial_day_scholars = 1280)
  (h4 : additional_boarders = 80)
  (h5 : new_boarders = initial_boarders + additional_boarders)
  (h6 : new_ratio new_boarders initial_day_scholars) :
  new_ratio r1 r2 → r1 = 1 ∧ r2 = 2 :=
by {
    sorry
}

end new_boarders_day_scholars_ratio_l1937_193705


namespace games_attended_this_month_l1937_193751

theorem games_attended_this_month 
  (games_last_month games_next_month total_games games_this_month : ℕ)
  (h1 : games_last_month = 17)
  (h2 : games_next_month = 16)
  (h3 : total_games = 44)
  (h4 : games_last_month + games_this_month + games_next_month = total_games) : 
  games_this_month = 11 := by
  sorry

end games_attended_this_month_l1937_193751


namespace triangle_area_l1937_193752

-- Define the conditions of the problem
variables (a b c : ℝ) (C : ℝ)
axiom cond1 : c^2 = a^2 + b^2 - 2 * a * b + 6
axiom cond2 : C = Real.pi / 3

-- Define the goal
theorem triangle_area : 
  (1 / 2) * a * b * Real.sin C = (3 * Real.sqrt 3) / 2 :=
by
  sorry

end triangle_area_l1937_193752


namespace sqrt_221_between_15_and_16_l1937_193728

theorem sqrt_221_between_15_and_16 : 15 < Real.sqrt 221 ∧ Real.sqrt 221 < 16 := by
  sorry

end sqrt_221_between_15_and_16_l1937_193728


namespace expression_is_nonnegative_l1937_193761

noncomputable def expression_nonnegative (a b c d e : ℝ) : Prop :=
  (a - b) * (a - c) * (a - d) * (a - e) +
  (b - a) * (b - c) * (b - d) * (b - e) +
  (c - a) * (c - b) * (c - d) * (c - e) +
  (d - a) * (d - b) * (d - c) * (d - e) +
  (e - a) * (e - b) * (e - c) * (e - d) ≥ 0

theorem expression_is_nonnegative (a b c d e : ℝ) : expression_nonnegative a b c d e := 
by 
  sorry

end expression_is_nonnegative_l1937_193761


namespace percentage_difference_l1937_193708

variable {P Q : ℝ}

theorem percentage_difference (P Q : ℝ) : (100 * (Q - P)) / Q = ((Q - P) / Q) * 100 :=
by
  sorry

end percentage_difference_l1937_193708


namespace surplus_by_end_of_week_is_14_estimated_monthly_income_is_1860_l1937_193701

-- Given conditions
def income_per_day : List Int := [65, 68, 50, 66, 50, 75, 74]
def expenditure_per_day : List Int := [-60, -64, -63, -58, -60, -64, -65]

-- Part 1: Proving the surplus by the end of the week is 14 yuan
theorem surplus_by_end_of_week_is_14 :
  List.sum income_per_day + List.sum expenditure_per_day = 14 :=
by
  sorry

-- Part 2: Proving the estimated income needed per month to maintain normal expenses is 1860 yuan
theorem estimated_monthly_income_is_1860 :
  (List.sum (List.map Int.natAbs expenditure_per_day) / 7) * 30 = 1860 :=
by
  sorry

end surplus_by_end_of_week_is_14_estimated_monthly_income_is_1860_l1937_193701


namespace complex_product_l1937_193777

theorem complex_product (i : ℂ) (h : i^2 = -1) :
  (3 - 4 * i) * (2 + 7 * i) = 34 + 13 * i :=
sorry

end complex_product_l1937_193777


namespace question1_question2_l1937_193762

theorem question1 (x : ℝ) : (1 < x^2 - 3*x + 1 ∧ x^2 - 3*x + 1 < 9 - x) ↔ (-2 < x ∧ x < 0) ∨ (3 < x ∧ x < 4) :=
by sorry

theorem question2 (x a : ℝ) : ((x - a)/(x - a^2) < 0)
  ↔ (a = 0 ∨ a = 1 → false)
  ∨ (0 < a ∧ a < 1 ∧ a^2 < x ∧ x < a)
  ∨ ((a < 0 ∨ a > 1) ∧ a < x ∧ x < a^2) :=
by sorry

end question1_question2_l1937_193762


namespace square_B_perimeter_l1937_193765

theorem square_B_perimeter :
  ∀ (sideA sideB : ℝ), (4 * sideA = 24) → (sideB^2 = (sideA^2) / 4) → (4 * sideB = 12) :=
by
  sorry

end square_B_perimeter_l1937_193765


namespace cube_sum_l1937_193773

-- Definitions
variable (ω : ℂ) (h1 : ω^3 = 1) (h2 : ω^2 + ω + 1 = 0) -- nonreal root

-- Theorem statement
theorem cube_sum : (2 - ω + ω^2)^3 + (2 + ω - ω^2)^3 = -2 :=
by 
  sorry

end cube_sum_l1937_193773


namespace problem_l1937_193748

theorem problem (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 1) :
  1 / (a ^ 3 * (b + c)) + 1 / (b ^ 3 * (c + a)) + 1 / (c ^ 3 * (a + b)) ≥ 3 / 2 :=
by sorry

end problem_l1937_193748


namespace total_candidates_l1937_193723

theorem total_candidates (T : ℝ) 
  (h1 : 0.45 * T = T * 0.45)
  (h2 : 0.38 * T = T * 0.38)
  (h3 : 0.22 * T = T * 0.22)
  (h4 : 0.12 * T = T * 0.12)
  (h5 : 0.09 * T = T * 0.09)
  (h6 : 0.10 * T = T * 0.10)
  (h7 : 0.05 * T = T * 0.05)
  (h_passed_english_alone : T - (0.45 * T - 0.12 * T - 0.10 * T + 0.05 * T) = 720) :
  T = 1000 :=
by
  sorry

end total_candidates_l1937_193723


namespace shorter_diagonal_of_rhombus_l1937_193719

variable (d s : ℝ)  -- d for shorter diagonal, s for the side length of the rhombus

theorem shorter_diagonal_of_rhombus 
  (h1 : ∀ (s : ℝ), s = 39)
  (h2 : ∀ (a b : ℝ), a^2 + b^2 = s^2)
  (h3 : ∀ (d a : ℝ), (d / 2)^2 + a^2 = 39^2)
  (h4 : 72 / 2 = 36)
  : d = 30 := 
by 
  sorry

end shorter_diagonal_of_rhombus_l1937_193719


namespace weight_of_33rd_weight_l1937_193730

theorem weight_of_33rd_weight :
  ∃ a : ℕ → ℕ, (∀ k, a k < a (k+1)) ∧
               (∀ k ≤ 29, a k + a (k+3) = a (k+1) + a (k+2)) ∧
               a 2 = 9 ∧
               a 8 = 33 ∧
               a 32 = 257 :=
sorry

end weight_of_33rd_weight_l1937_193730


namespace total_rainfall_2003_and_2004_l1937_193727

noncomputable def average_rainfall_2003 : ℝ := 45
noncomputable def months_in_year : ℕ := 12
noncomputable def percent_increase : ℝ := 0.05

theorem total_rainfall_2003_and_2004 :
  let rainfall_2004 := average_rainfall_2003 * (1 + percent_increase)
  let total_rainfall_2003 := average_rainfall_2003 * months_in_year
  let total_rainfall_2004 := rainfall_2004 * months_in_year
  total_rainfall_2003 = 540 ∧ total_rainfall_2004 = 567 := 
by 
  sorry

end total_rainfall_2003_and_2004_l1937_193727


namespace min_transport_cost_l1937_193798

/- Definitions for the problem conditions -/
def villageA_vegetables : ℕ := 80
def villageB_vegetables : ℕ := 60
def destinationX_requirement : ℕ := 65
def destinationY_requirement : ℕ := 75

def cost_A_to_X : ℕ := 50
def cost_A_to_Y : ℕ := 30
def cost_B_to_X : ℕ := 60
def cost_B_to_Y : ℕ := 45

def W (x : ℕ) : ℕ :=
  cost_A_to_X * x +
  cost_A_to_Y * (villageA_vegetables - x) +
  cost_B_to_X * (destinationX_requirement - x) +
  cost_B_to_Y * (x - 5) + 6075 - 225

/- Prove that the minimum total cost W is 6100 -/
theorem min_transport_cost : ∃ (x : ℕ), 5 ≤ x ∧ x ≤ 65 ∧ W x = 6100 :=
by sorry

end min_transport_cost_l1937_193798


namespace ella_days_11_years_old_l1937_193747

theorem ella_days_11_years_old (x y z : ℕ) (h1 : 40 * x + 44 * y + 48 * (180 - x - y) = 7920) (h2 : x + y + z = 180) (h3 : 2 * x + y = 180) : y = 60 :=
by {
  -- proof can be derived from the given conditions
  sorry
}

end ella_days_11_years_old_l1937_193747


namespace hyeyoung_walked_correct_l1937_193794

/-- The length of the promenade near Hyeyoung's house is 6 kilometers (km). -/
def promenade_length : ℕ := 6

/-- Hyeyoung walked from the starting point to the halfway point of the trail. -/
def hyeyoung_walked : ℕ := promenade_length / 2

/-- The distance Hyeyoung walked is 3 kilometers (km). -/
theorem hyeyoung_walked_correct : hyeyoung_walked = 3 := by
  sorry

end hyeyoung_walked_correct_l1937_193794


namespace average_weight_of_16_boys_is_50_25_l1937_193759

theorem average_weight_of_16_boys_is_50_25
  (W : ℝ)
  (h1 : 8 * 45.15 = 361.2)
  (h2 : 24 * 48.55 = 1165.2)
  (h3 : 16 * W + 361.2 = 1165.2) :
  W = 50.25 :=
sorry

end average_weight_of_16_boys_is_50_25_l1937_193759


namespace find_f_of_conditions_l1937_193722

theorem find_f_of_conditions (f : ℝ → ℝ) :
  (f 1 = 1) →
  (∀ x y : ℝ, f (x + y) = 3^y * f x + 2^x * f y) →
  (∀ x : ℝ, f x = 3^x - 2^x) :=
by
  intros h1 h2
  sorry

end find_f_of_conditions_l1937_193722


namespace bus_stops_time_per_hour_l1937_193744

theorem bus_stops_time_per_hour 
  (avg_speed_without_stoppages : ℝ) 
  (avg_speed_with_stoppages : ℝ) 
  (h1 : avg_speed_without_stoppages = 75) 
  (h2 : avg_speed_with_stoppages = 40) : 
  ∃ (stoppage_time : ℝ), stoppage_time = 28 :=
by
  sorry

end bus_stops_time_per_hour_l1937_193744


namespace cover_square_with_rectangles_l1937_193749

theorem cover_square_with_rectangles :
  ∃ (n : ℕ), 
    ∀ (a b : ℕ), 
      (a = 3) ∧ 
      (b = 4) ∧ 
      (n = (12 * 12) / (a * b)) ∧ 
      (144 = n * (a * b)) ∧ 
      (3 * 4 = a * b) 
  → 
    n = 12 :=
by
  sorry

end cover_square_with_rectangles_l1937_193749


namespace speed_for_remaining_distance_l1937_193706

theorem speed_for_remaining_distance
  (t_total : ℝ) (v1 : ℝ) (d_total : ℝ)
  (t_total_def : t_total = 1.4)
  (v1_def : v1 = 4)
  (d_total_def : d_total = 5.999999999999999) :
  ∃ v2 : ℝ, v2 = 5 := 
by
  sorry

end speed_for_remaining_distance_l1937_193706


namespace sin_cos_relationship_l1937_193711

theorem sin_cos_relationship (α : ℝ) (h1 : Real.pi / 2 < α) (h2 : α < Real.pi) : 
  Real.sin α - Real.cos α > 1 :=
sorry

end sin_cos_relationship_l1937_193711


namespace bananas_to_oranges_equivalence_l1937_193709

theorem bananas_to_oranges_equivalence :
  (3 / 4 : ℚ) * 16 = 12 ->
  (2 / 5 : ℚ) * 10 = 4 :=
by
  intros h
  sorry

end bananas_to_oranges_equivalence_l1937_193709


namespace max_value_vector_sum_l1937_193788

theorem max_value_vector_sum (α β : ℝ) :
  let a := (Real.cos α, Real.sin α)
  let b := (Real.sin β, -Real.cos β)
  |(a.1 + b.1, a.2 + b.2)| ≤ 2 := by
  sorry

end max_value_vector_sum_l1937_193788


namespace find_number_l1937_193766

theorem find_number (x : ℝ) (h : (x / 4) + 3 = 5) : x = 8 :=
by
  sorry

end find_number_l1937_193766


namespace exists_distinct_pure_powers_l1937_193795

-- Definitions and conditions
def is_pure_kth_power (k m : ℕ) : Prop := ∃ t : ℕ, m = t ^ k

-- The main theorem statement
theorem exists_distinct_pure_powers (n : ℕ) (hn : 0 < n) :
  ∃ (a : Fin n → ℕ),
    (∀ i j : Fin n, i ≠ j → a i ≠ a j) ∧ 
    is_pure_kth_power 2009 (Finset.univ.sum a) ∧ 
    is_pure_kth_power 2010 (Finset.univ.prod a) :=
sorry

end exists_distinct_pure_powers_l1937_193795


namespace speed_limit_inequality_l1937_193757

theorem speed_limit_inequality (v : ℝ) : (v ≤ 40) :=
sorry

end speed_limit_inequality_l1937_193757


namespace no_integer_solutions_quadratic_l1937_193713

theorem no_integer_solutions_quadratic (n : ℤ) (s : ℕ) (pos_odd_s : s % 2 = 1) :
  ¬ ∃ x : ℤ, x^2 - 16 * n * x + 7^s = 0 :=
sorry

end no_integer_solutions_quadratic_l1937_193713


namespace stamps_total_l1937_193779

theorem stamps_total (x y : ℕ) (hx : x = 34) (hy : y = x + 44) : x + y = 112 :=
by sorry

end stamps_total_l1937_193779


namespace correct_equation_l1937_193760

-- Definitions of the conditions
def contributes_5_coins (x : ℕ) (P : ℕ) : Prop :=
  5 * x + 45 = P

def contributes_7_coins (x : ℕ) (P : ℕ) : Prop :=
  7 * x + 3 = P

-- Mathematical proof problem
theorem correct_equation 
(x : ℕ) (P : ℕ) (h1 : contributes_5_coins x P) (h2 : contributes_7_coins x P) : 
5 * x + 45 = 7 * x + 3 := 
by
  sorry

end correct_equation_l1937_193760


namespace john_subtracts_79_l1937_193775

theorem john_subtracts_79 (x : ℕ) (h : x = 40) : (x - 1)^2 = x^2 - 79 :=
by sorry

end john_subtracts_79_l1937_193775


namespace area_triangle_ABC_l1937_193782

noncomputable def area_trapezoid (AB CD height : ℝ) : ℝ :=
  (AB + CD) * height / 2

noncomputable def area_triangle (base height : ℝ) : ℝ :=
  base * height / 2

variable (AB CD height area_ABCD : ℝ)
variables (h0 : CD = 3 * AB) (h1 : area_trapezoid AB CD height = 24)

theorem area_triangle_ABC : area_triangle AB height = 6 :=
by
  sorry

end area_triangle_ABC_l1937_193782


namespace abs_inequality_solution_set_l1937_193742

theorem abs_inequality_solution_set (x : ℝ) : -1 < x ∧ x < 1 ↔ |2*x - 1| - |x - 2| < 0 := by
  sorry

end abs_inequality_solution_set_l1937_193742


namespace order_options_count_l1937_193736

/-- Define the number of options for each category -/
def drinks : ℕ := 3
def salads : ℕ := 2
def pizzas : ℕ := 5

/-- The theorem statement that we aim to prove -/
theorem order_options_count : drinks * salads * pizzas = 30 :=
by
  sorry -- Proof is skipped as instructed

end order_options_count_l1937_193736


namespace power_equivalence_l1937_193740

theorem power_equivalence (m n : ℕ) (hm_pos : 0 < m) (hn_pos : 0 < n) (x y : ℕ) 
  (hx : 2^m = x) (hy : 2^(2 * n) = y) : 4^(m + 2 * n) = x^2 * y^2 := 
by 
  sorry

end power_equivalence_l1937_193740


namespace expand_and_simplify_l1937_193787

theorem expand_and_simplify (x : ℝ) : 3 * (x - 3) * (x + 10) + 2 * x = 3 * x^2 + 23 * x - 90 :=
by
  sorry

end expand_and_simplify_l1937_193787


namespace value_of_a_squared_plus_b_squared_l1937_193724

theorem value_of_a_squared_plus_b_squared (a b : ℝ) 
  (h1 : (a + b) ^ 2 = 8) 
  (h2 : (a - b) ^ 2 = 12) : 
  a^2 + b^2 = 10 :=
sorry

end value_of_a_squared_plus_b_squared_l1937_193724


namespace trig_identity_proof_l1937_193783

theorem trig_identity_proof 
  (α p q : ℝ)
  (hp : p ≠ 0) (hq : q ≠ 0)
  (tangent : Real.tan α = p / q) :
  Real.sin (2 * α) = 2 * p * q / (p^2 + q^2) ∧
  Real.cos (2 * α) = (q^2 - p^2) / (q^2 + p^2) ∧
  Real.tan (2 * α) = (2 * p * q) / (q^2 - p^2) :=
by
  sorry

end trig_identity_proof_l1937_193783


namespace inequality_solution_l1937_193793

variable {x : ℝ}

theorem inequality_solution (h : 0 ≤ x ∧ x ≤ 2 * Real.pi) : 
  (2 * Real.cos x ≤ |Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x))| 
  ∧ |Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x))| ≤ Real.sqrt 2) ↔ 
  (Real.pi / 4 ≤ x ∧ x ≤ 7 * Real.pi / 4) :=
sorry

end inequality_solution_l1937_193793


namespace power_function_value_l1937_193792

noncomputable def f (x : ℝ) : ℝ := x^2

theorem power_function_value :
  f 3 = 9 :=
by
  -- Since f(x) = x^2 and f passes through (-2, 4)
  -- f(x) = x^2, so f(3) = 3^2 = 9
  sorry

end power_function_value_l1937_193792


namespace percentage_of_x_is_y_l1937_193739

theorem percentage_of_x_is_y (x y : ℝ) (h : 0.5 * (x - y) = 0.4 * (x + y)) : y = 0.1111 * x := 
sorry

end percentage_of_x_is_y_l1937_193739


namespace neg_A_is_square_of_int_l1937_193768

theorem neg_A_is_square_of_int (x y z : ℤ) (A : ℤ) (h1 : A = x * y + y * z + z * x) 
  (h2 : A = (x + 1) * (y - 2) + (y - 2) * (z - 2) + (z - 2) * (x + 1)) : ∃ k : ℤ, -A = k^2 :=
by
  sorry

end neg_A_is_square_of_int_l1937_193768


namespace value_2_std_dev_less_than_mean_l1937_193750

def mean : ℝ := 16.5
def std_dev : ℝ := 1.5

theorem value_2_std_dev_less_than_mean :
  mean - 2 * std_dev = 13.5 := by
  sorry

end value_2_std_dev_less_than_mean_l1937_193750


namespace hcf_of_three_numbers_l1937_193796

def hcf (a b : ℕ) : ℕ := gcd a b

theorem hcf_of_three_numbers :
  let a := 136
  let b := 144
  let c := 168
  hcf (hcf a b) c = 8 :=
by
  sorry

end hcf_of_three_numbers_l1937_193796


namespace correct_calculation_result_l1937_193771

theorem correct_calculation_result 
  (P : Polynomial ℝ := -x^2 + x - 1) :
  (P + -3 * x) = (-x^2 - 2 * x - 1) :=
by
  -- Since this is just the proof statement, sorry is used to skip the proof.
  sorry

end correct_calculation_result_l1937_193771


namespace abc_divisibility_l1937_193712

theorem abc_divisibility (a b c : ℕ) (h1 : 1 < a) (h2 : a < b) (h3 : b < c) : 
  (a - 1) * (b - 1) * (c - 1) ∣ (a * b * c - 1) ↔ (a = 2 ∧ b = 4 ∧ c = 8) ∨ (a = 3 ∧ b = 5 ∧ c = 15) :=
by {
  sorry  -- proof to be filled in
}

end abc_divisibility_l1937_193712


namespace subtraction_problem_digits_sum_l1937_193703

theorem subtraction_problem_digits_sum :
  ∃ (K L M N : ℕ), K < 10 ∧ L < 10 ∧ M < 10 ∧ N < 10 ∧ 
  ((6000 + K * 100 + 0 + L) - (900 + N * 10 + 4) = 2011) ∧ 
  (K + L + M + N = 17) :=
by
  sorry

end subtraction_problem_digits_sum_l1937_193703


namespace binomial_coefficient_multiple_of_4_l1937_193700

theorem binomial_coefficient_multiple_of_4 :
  ∃ (S : Finset ℕ), (∀ k ∈ S, 0 ≤ k ∧ k ≤ 2014 ∧ (Nat.choose 2014 k) % 4 = 0) ∧ S.card = 991 :=
sorry

end binomial_coefficient_multiple_of_4_l1937_193700


namespace column_of_1000_is_C_l1937_193770

def column_of_integer (n : ℕ) : String :=
  ["A", "B", "C", "D", "E", "F", "E", "D", "C", "B"].get! ((n - 2) % 10)

theorem column_of_1000_is_C :
  column_of_integer 1000 = "C" :=
by
  sorry

end column_of_1000_is_C_l1937_193770


namespace evaluate_expression_eq_l1937_193769

theorem evaluate_expression_eq :
  let x := 2
  let y := -3
  let z := 7
  x^2 + y^2 - z^2 - 2 * x * y + 3 * z = -15 := by
    sorry

end evaluate_expression_eq_l1937_193769


namespace find_fraction_l1937_193754

def f (x : ℕ) : ℕ := 3 * x + 2
def g (x : ℕ) : ℕ := 2 * x - 3

theorem find_fraction : (f (g (f 2))) / (g (f (g 2))) = 41 / 7 := 
by 
  sorry

end find_fraction_l1937_193754


namespace arithmetic_sequence_problem_l1937_193781

-- Define sequence and sum properties
def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
  ∀ n, S n = (n * (a 1 + a n)) / 2

/- Theorem Statement -/
theorem arithmetic_sequence_problem (a : ℕ → ℤ) (S : ℕ → ℤ) (d : ℤ) 
  (h_seq : arithmetic_sequence a d) 
  (h_initial : a 1 = 31) 
  (h_S_eq : S 10 = S 22) :
  -- Part 1: Find S_n
  (∀ n, S n = 32 * n - n ^ 2) ∧
  -- Part 2: Maximum sum occurs at n = 16 and is 256
  (∀ n, S n ≤ 256 ∧ (S 16 = 256 → ∀ m ≠ 16, S m < 256)) :=
by
  -- proof to be provided here
  sorry

end arithmetic_sequence_problem_l1937_193781


namespace value_of_fraction_l1937_193755

variable (x y : ℝ)
variable (hx : x ≠ 0)
variable (hy : y ≠ 0)
variable (h : (4 * x + y) / (x - 4 * y) = -3)

theorem value_of_fraction : (x + 4 * y) / (4 * x - y) = 39 / 37 := by
  sorry

end value_of_fraction_l1937_193755
