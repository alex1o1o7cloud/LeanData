import Mathlib

namespace NUMINAMATH_GPT_power_C_50_l1231_123161

def matrixC : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![2, 1], ![-4, -1]]

theorem power_C_50 :
  matrixC ^ 50 = ![![4^49 + 1, 4^49], ![-4^50, -2 * 4^49 + 1]] :=
by
  sorry

end NUMINAMATH_GPT_power_C_50_l1231_123161


namespace NUMINAMATH_GPT_juwella_read_more_last_night_l1231_123178

-- Definitions of the conditions
def pages_three_nights_ago : ℕ := 15
def book_pages : ℕ := 100
def pages_tonight : ℕ := 20
def pages_two_nights_ago : ℕ := 2 * pages_three_nights_ago
def total_pages_before_tonight : ℕ := book_pages - pages_tonight
def pages_last_night : ℕ := total_pages_before_tonight - pages_three_nights_ago - pages_two_nights_ago

theorem juwella_read_more_last_night :
  pages_last_night - pages_two_nights_ago = 5 :=
by
  sorry

end NUMINAMATH_GPT_juwella_read_more_last_night_l1231_123178


namespace NUMINAMATH_GPT_find_m_l1231_123117

def vector_a : ℝ × ℝ := (1, 2)
def vector_b (m : ℝ) : ℝ × ℝ := (-1, m)
def is_perpendicular (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

theorem find_m (m : ℝ) (h : is_perpendicular vector_a (vector_b m)) : m = 1 / 2 :=
by 
  sorry

end NUMINAMATH_GPT_find_m_l1231_123117


namespace NUMINAMATH_GPT_number_of_ones_and_zeros_not_perfect_square_l1231_123104

open Int

theorem number_of_ones_and_zeros_not_perfect_square (k : ℕ) : 
  let N := (10^k) * (10^300 - 1) / 9
  ¬ ∃ m : ℤ, m^2 = N :=
by
  sorry

end NUMINAMATH_GPT_number_of_ones_and_zeros_not_perfect_square_l1231_123104


namespace NUMINAMATH_GPT_find_angle_A_l1231_123156

theorem find_angle_A (a b c : ℝ) (A : ℝ) (h : a^2 = b^2 - b * c + c^2) : A = 60 :=
sorry

end NUMINAMATH_GPT_find_angle_A_l1231_123156


namespace NUMINAMATH_GPT_blue_markers_count_l1231_123130

-- Definitions based on given conditions
def total_markers : ℕ := 3343
def red_markers : ℕ := 2315

-- Main statement to prove
theorem blue_markers_count : total_markers - red_markers = 1028 := by
  sorry

end NUMINAMATH_GPT_blue_markers_count_l1231_123130


namespace NUMINAMATH_GPT_train_has_96_cars_l1231_123151

def train_cars_count (cars_in_15_seconds : Nat) (time_for_15_seconds : Nat) (total_time_seconds : Nat) : Nat :=
  total_time_seconds * cars_in_15_seconds / time_for_15_seconds

theorem train_has_96_cars :
  train_cars_count 8 15 180 = 96 :=
by
  sorry

end NUMINAMATH_GPT_train_has_96_cars_l1231_123151


namespace NUMINAMATH_GPT_expand_simplify_expression_l1231_123105

theorem expand_simplify_expression (a : ℝ) :
  (a + 1) * (a + 3) * (a + 4) * (a + 5) * (a + 6) = 
  a^5 + 19 * a^4 + 137 * a^3 + 461 * a^2 + 702 * a + 360 :=
by
  sorry

end NUMINAMATH_GPT_expand_simplify_expression_l1231_123105


namespace NUMINAMATH_GPT_swimming_pool_width_l1231_123131

theorem swimming_pool_width (L D1 D2 V : ℝ) (W : ℝ) (h : L = 12) (h1 : D1 = 1) (h2 : D2 = 4) (hV : V = 270) : W = 9 :=
  by
    -- We begin by stating the formula for the volume of 
    -- a trapezoidal prism: Volume = (1/2) * (D1 + D2) * L * W
    
    -- According to the problem, we have the following conditions:
    have hVolume : V = (1/2) * (D1 + D2) * L * W :=
      by sorry

    -- Substitute the provided values into the volume equation:
    -- 270 = (1/2) * (1 + 4) * 12 * W
    
    -- Simplify and solve for W
    simp at hVolume
    exact sorry

end NUMINAMATH_GPT_swimming_pool_width_l1231_123131


namespace NUMINAMATH_GPT_cylinder_volume_options_l1231_123175

theorem cylinder_volume_options (length width : ℝ) (h₀ : length = 4) (h₁ : width = 2) :
  ∃ V, (V = (4 / π) ∨ V = (8 / π)) :=
by
  sorry

end NUMINAMATH_GPT_cylinder_volume_options_l1231_123175


namespace NUMINAMATH_GPT_simultaneous_equations_solution_exists_l1231_123159

theorem simultaneous_equations_solution_exists (m : ℝ) : 
  (∃ (x y : ℝ), y = m * x + 2 ∧ y = (3 * m - 2) * x + 5) ↔ m ≠ 1 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_simultaneous_equations_solution_exists_l1231_123159


namespace NUMINAMATH_GPT_count_four_digit_numbers_with_digit_sum_4_l1231_123146

theorem count_four_digit_numbers_with_digit_sum_4 : 
  ∃ n : ℕ, (∀ (x1 x2 x3 x4 : ℕ), 
    x1 + x2 + x3 + x4 = 4 ∧ x1 ≥ 1 ∧ x2 ≥ 0 ∧ x3 ≥ 0 ∧ x4 ≥ 0 →
    n = 20) :=
sorry

end NUMINAMATH_GPT_count_four_digit_numbers_with_digit_sum_4_l1231_123146


namespace NUMINAMATH_GPT_problem_1_l1231_123137

theorem problem_1 (α : ℝ) (k : ℤ) (n : ℕ) (hk : k > 0) (hα : α ≠ k * Real.pi) (hn : n > 0) :
  n = 1 → (0.5 + Real.cos α) = (0.5 + Real.cos α) :=
by
  sorry

end NUMINAMATH_GPT_problem_1_l1231_123137


namespace NUMINAMATH_GPT_min_value_of_2x_plus_y_l1231_123116

theorem min_value_of_2x_plus_y 
  (x y : ℝ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (h : 1 / (x + 1) + 1 / (x + 2 * y) = 1) : 
  (2 * x + y) = 1 / 2 + Real.sqrt 3 := 
sorry

end NUMINAMATH_GPT_min_value_of_2x_plus_y_l1231_123116


namespace NUMINAMATH_GPT_tenth_day_of_month_is_monday_l1231_123113

def total_run_minutes_in_month (hours : ℕ) : ℕ := hours * 60

def run_minutes_per_week (runs_per_week : ℕ) (minutes_per_run : ℕ) : ℕ := 
  runs_per_week * minutes_per_run

def weeks_in_month (total_minutes : ℕ) (minutes_per_week : ℕ) : ℕ := 
  total_minutes / minutes_per_week

def identify_day_of_week (first_day : ℕ) (target_day : ℕ) : ℕ := 
  (first_day + target_day - 1) % 7

theorem tenth_day_of_month_is_monday :
  let hours := 5
  let runs_per_week := 3
  let minutes_per_run := 20
  let first_day := 6 -- Assuming 0=Sunday, ..., 6=Saturday
  let target_day := 10
  total_run_minutes_in_month hours = 300 ∧
  run_minutes_per_week runs_per_week minutes_per_run = 60 ∧
  weeks_in_month 300 60 = 5 ∧
  identify_day_of_week first_day target_day = 1 := -- 1 represents Monday
sorry

end NUMINAMATH_GPT_tenth_day_of_month_is_monday_l1231_123113


namespace NUMINAMATH_GPT_max_value_inequality_l1231_123170

theorem max_value_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (abc * (a + b + c)) / ((a + b)^2 * (b + c)^3) ≤ 1 / 4 :=
sorry

end NUMINAMATH_GPT_max_value_inequality_l1231_123170


namespace NUMINAMATH_GPT_smallest_positive_perfect_cube_has_divisor_l1231_123167

theorem smallest_positive_perfect_cube_has_divisor (p q r s : ℕ) (hp : Prime p) (hq : Prime q)
  (hr : Prime r) (hs : Prime s) (hpqrs : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s) :
  ∃ n : ℕ, n = (p * q * r * s^2)^3 ∧ ∀ m : ℕ, (m = p^2 * q^3 * r^4 * s^5 → m ∣ n) :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_perfect_cube_has_divisor_l1231_123167


namespace NUMINAMATH_GPT_scientific_notation_of_4212000_l1231_123177

theorem scientific_notation_of_4212000 :
  4212000 = 4.212 * 10^6 :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_of_4212000_l1231_123177


namespace NUMINAMATH_GPT_ratio_equivalence_l1231_123192

theorem ratio_equivalence (x : ℕ) : 
  (10 * 60 = 600) →
  (15 : ℕ) / 5 = x / 600 →
  x = 1800 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_ratio_equivalence_l1231_123192


namespace NUMINAMATH_GPT_angle_greater_difference_l1231_123189

theorem angle_greater_difference (A B C : ℕ) (h1 : B = 5 * A) (h2 : A + B + C = 180) (h3 : A = 24) 
: C - A = 12 := 
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_angle_greater_difference_l1231_123189


namespace NUMINAMATH_GPT_xyz_value_l1231_123162

theorem xyz_value (x y z : ℚ)
  (h1 : x + y + z = 1)
  (h2 : x + y - z = 2)
  (h3 : x - y - z = 3) :
  x * y * z = 1/2 :=
by
  sorry

end NUMINAMATH_GPT_xyz_value_l1231_123162


namespace NUMINAMATH_GPT_students_not_in_biology_l1231_123180

theorem students_not_in_biology (total_students : ℕ) (percentage_in_biology : ℚ) 
  (h1 : total_students = 880) (h2 : percentage_in_biology = 27.5 / 100) : 
  total_students - (total_students * percentage_in_biology) = 638 := 
by
  sorry

end NUMINAMATH_GPT_students_not_in_biology_l1231_123180


namespace NUMINAMATH_GPT_distance_from_M0_to_plane_is_sqrt77_l1231_123153

structure Point3D :=
  (x : ℝ) (y : ℝ) (z : ℝ)

def M1 : Point3D := ⟨1, 0, 2⟩
def M2 : Point3D := ⟨1, 2, -1⟩
def M3 : Point3D := ⟨2, -2, 1⟩
def M0 : Point3D := ⟨-5, -9, 1⟩

noncomputable def distance_to_plane (P : Point3D) (A B C : Point3D) : ℝ := sorry

theorem distance_from_M0_to_plane_is_sqrt77 : 
  distance_to_plane M0 M1 M2 M3 = Real.sqrt 77 := sorry

end NUMINAMATH_GPT_distance_from_M0_to_plane_is_sqrt77_l1231_123153


namespace NUMINAMATH_GPT_asymptotic_minimal_eccentricity_l1231_123140

noncomputable def hyperbola (m : ℝ) (x y : ℝ) : Prop :=
  x^2 / m - y^2 / (m^2 + 4) = 1

noncomputable def eccentricity (m : ℝ) : ℝ :=
  Real.sqrt (m + 4 / m + 1)

theorem asymptotic_minimal_eccentricity :
  ∃ (m : ℝ), m = 2 ∧ hyperbola m x y → ∀ x y, y = 2 * x ∨ y = -2 * x :=
by
  sorry

end NUMINAMATH_GPT_asymptotic_minimal_eccentricity_l1231_123140


namespace NUMINAMATH_GPT_positive_difference_of_squares_l1231_123157

theorem positive_difference_of_squares 
  (a b : ℕ)
  (h1 : a + b = 60)
  (h2 : a - b = 16) : a^2 - b^2 = 960 :=
by
  sorry

end NUMINAMATH_GPT_positive_difference_of_squares_l1231_123157


namespace NUMINAMATH_GPT_part1_point_A_value_of_m_part1_area_ABC_part2_max_ordinate_P_l1231_123188

noncomputable def quadratic_function (m x : ℝ) : ℝ := (m - 2) * x ^ 2 - x - m ^ 2 + 6 * m - 7

theorem part1_point_A_value_of_m (m : ℝ) (h : quadratic_function m (-1) = 2) : m = 5 :=
sorry

theorem part1_area_ABC (area : ℝ) 
  (h₁ : quadratic_function 5 (1 : ℝ) = 0) 
  (h₂ : quadratic_function 5 (-2/3 : ℝ) = 0) : area = 5 / 3 :=
sorry

theorem part2_max_ordinate_P (m : ℝ) (h : - (m - 3) ^ 2 + 2 ≤ 2) : m = 3 :=
sorry

end NUMINAMATH_GPT_part1_point_A_value_of_m_part1_area_ABC_part2_max_ordinate_P_l1231_123188


namespace NUMINAMATH_GPT_vacation_expenses_split_l1231_123121

theorem vacation_expenses_split
  (A : ℝ) (B : ℝ) (C : ℝ) (a : ℝ) (b : ℝ)
  (hA : A = 180)
  (hB : B = 240)
  (hC : C = 120)
  (ha : a = 0)
  (hb : b = 0)
  : a - b = 0 := 
by
  sorry

end NUMINAMATH_GPT_vacation_expenses_split_l1231_123121


namespace NUMINAMATH_GPT_hens_count_l1231_123174

theorem hens_count (H C : ℕ) (h1 : H + C = 48) (h2 : 2 * H + 4 * C = 136) : H = 28 :=
by
  sorry

end NUMINAMATH_GPT_hens_count_l1231_123174


namespace NUMINAMATH_GPT_trig_expression_evaluation_l1231_123141

theorem trig_expression_evaluation
  (α : ℝ)
  (h_tan_α : Real.tan α = 3) :
  (Real.sin (π - α) - Real.sin (π / 2 + α)) / 
  (Real.cos (3 * π / 2 - α) + 2 * Real.cos (-π + α)) = -2 / 5 := by
  sorry

end NUMINAMATH_GPT_trig_expression_evaluation_l1231_123141


namespace NUMINAMATH_GPT_sarah_earnings_l1231_123142

-- Conditions
def monday_hours : ℚ := 1 + 3 / 4
def wednesday_hours : ℚ := 65 / 60
def thursday_hours : ℚ := 2 + 45 / 60
def friday_hours : ℚ := 45 / 60
def saturday_hours : ℚ := 2

def weekday_rate : ℚ := 4
def weekend_rate : ℚ := 6

-- Definition for total earnings
def total_weekday_earnings : ℚ :=
  (monday_hours + wednesday_hours + thursday_hours + friday_hours) * weekday_rate

def total_weekend_earnings : ℚ :=
  saturday_hours * weekend_rate

def total_earnings : ℚ :=
  total_weekday_earnings + total_weekend_earnings

-- Statement to prove
theorem sarah_earnings : total_earnings = 37.3332 := by
  sorry

end NUMINAMATH_GPT_sarah_earnings_l1231_123142


namespace NUMINAMATH_GPT_initial_number_of_people_l1231_123150

theorem initial_number_of_people (P : ℕ) : P * 10 = (P + 1) * 5 → P = 1 :=
by sorry

end NUMINAMATH_GPT_initial_number_of_people_l1231_123150


namespace NUMINAMATH_GPT_polygon_sides_l1231_123132

theorem polygon_sides (h1 : 1260 - 360 = 900) (h2 : (n - 2) * 180 = 900) : n = 7 :=
by 
  sorry

end NUMINAMATH_GPT_polygon_sides_l1231_123132


namespace NUMINAMATH_GPT_stadium_seating_and_revenue_l1231_123155

   def children := 52
   def adults := 29
   def seniors := 15
   def seats_A := 40
   def seats_B := 30
   def seats_C := 25
   def price_A := 10
   def price_B := 15
   def price_C := 20
   def total_seats := 95

   def revenue_A := seats_A * price_A
   def revenue_B := seats_B * price_B
   def revenue_C := seats_C * price_C
   def total_revenue := revenue_A + revenue_B + revenue_C

   theorem stadium_seating_and_revenue :
     (children <= seats_B + seats_C) ∧
     (adults + seniors <= seats_A + seats_C) ∧
     (children + adults + seniors > total_seats) →
     (revenue_A = 400) ∧
     (revenue_B = 450) ∧
     (revenue_C = 500) ∧
     (total_revenue = 1350) :=
   by
     sorry
   
end NUMINAMATH_GPT_stadium_seating_and_revenue_l1231_123155


namespace NUMINAMATH_GPT_find_triples_l1231_123124

theorem find_triples (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + b^2 = c^2) ∧ (a^3 + b^3 + 1 = (c-1)^3) ↔ (a = 6 ∧ b = 8 ∧ c = 10) ∨ (a = 8 ∧ b = 6 ∧ c = 10) :=
by
  sorry

end NUMINAMATH_GPT_find_triples_l1231_123124


namespace NUMINAMATH_GPT_number_of_large_posters_is_5_l1231_123160

theorem number_of_large_posters_is_5
  (total_posters : ℕ)
  (small_posters_ratio : ℚ)
  (medium_posters_ratio : ℚ)
  (h_total : total_posters = 50)
  (h_small_ratio : small_posters_ratio = 2 / 5)
  (h_medium_ratio : medium_posters_ratio = 1 / 2) :
  (total_posters * (1 - small_posters_ratio - medium_posters_ratio)) = 5 :=
by sorry

end NUMINAMATH_GPT_number_of_large_posters_is_5_l1231_123160


namespace NUMINAMATH_GPT_total_lemonade_poured_l1231_123152

def lemonade_poured (first: ℝ) (second: ℝ) (third: ℝ) := first + second + third

theorem total_lemonade_poured :
  lemonade_poured 0.25 0.4166666666666667 0.25 = 0.917 :=
by
  sorry

end NUMINAMATH_GPT_total_lemonade_poured_l1231_123152


namespace NUMINAMATH_GPT_boxes_needed_l1231_123169

theorem boxes_needed (total_oranges boxes_capacity : ℕ) (h1 : total_oranges = 94) (h2 : boxes_capacity = 8) : 
  (total_oranges + boxes_capacity - 1) / boxes_capacity = 12 := 
by
  sorry

end NUMINAMATH_GPT_boxes_needed_l1231_123169


namespace NUMINAMATH_GPT_ratio_of_boys_to_girls_l1231_123100

theorem ratio_of_boys_to_girls (total_students : ℕ) (girls : ℕ) (boys : ℕ)
  (h_total : total_students = 1040)
  (h_girls : girls = 400)
  (h_boys : boys = total_students - girls) :
  (boys / Nat.gcd boys girls = 8) ∧ (girls / Nat.gcd boys girls = 5) :=
sorry

end NUMINAMATH_GPT_ratio_of_boys_to_girls_l1231_123100


namespace NUMINAMATH_GPT_smallest_n_for_multiples_of_2015_l1231_123126

theorem smallest_n_for_multiples_of_2015 (n : ℕ) (hn : 0 < n)
  (h5 : (2^n - 1) % 5 = 0)
  (h13 : (2^n - 1) % 13 = 0)
  (h31 : (2^n - 1) % 31 = 0) : n = 60 := by
  sorry

end NUMINAMATH_GPT_smallest_n_for_multiples_of_2015_l1231_123126


namespace NUMINAMATH_GPT_min_value_of_c_l1231_123165

variable {a b c : ℝ}
variables (a_pos : a > 0) (b_pos : b > 0)
variable (hyperbola : ∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1)
variable (semi_focal_dist : c = Real.sqrt (a^2 + b^2))
variable (distance_condition : ∀ (d : ℝ), d = a * b / c = 1 / 3 * c + 1)

theorem min_value_of_c : c = 6 := 
sorry

end NUMINAMATH_GPT_min_value_of_c_l1231_123165


namespace NUMINAMATH_GPT_average_lifespan_is_1013_l1231_123122

noncomputable def first_factory_lifespan : ℕ := 980
noncomputable def second_factory_lifespan : ℕ := 1020
noncomputable def third_factory_lifespan : ℕ := 1032

noncomputable def total_samples : ℕ := 100

noncomputable def first_samples : ℕ := (1 * total_samples) / 4
noncomputable def second_samples : ℕ := (2 * total_samples) / 4
noncomputable def third_samples : ℕ := (1 * total_samples) / 4

noncomputable def weighted_average_lifespan : ℕ :=
  ((first_factory_lifespan * first_samples) + (second_factory_lifespan * second_samples) + (third_factory_lifespan * third_samples)) / total_samples

theorem average_lifespan_is_1013 : weighted_average_lifespan = 1013 := by
  sorry

end NUMINAMATH_GPT_average_lifespan_is_1013_l1231_123122


namespace NUMINAMATH_GPT_working_mom_work_percent_l1231_123133

theorem working_mom_work_percent :
  let awake_hours := 16
  let work_hours := 8
  (work_hours / awake_hours) * 100 = 50 :=
by
  sorry

end NUMINAMATH_GPT_working_mom_work_percent_l1231_123133


namespace NUMINAMATH_GPT_books_not_sold_l1231_123143

theorem books_not_sold (X : ℕ) (H1 : (2/3 : ℝ) * X * 4 = 288) : (1 / 3 : ℝ) * X = 36 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_books_not_sold_l1231_123143


namespace NUMINAMATH_GPT_weight_of_first_lift_l1231_123172

-- Definitions as per conditions
variables (x y : ℝ)
def condition1 : Prop := x + y = 1800
def condition2 : Prop := 2 * x = y + 300

-- Prove that the weight of Joe's first lift is 700 pounds
theorem weight_of_first_lift (h1 : condition1 x y) (h2 : condition2 x y) : x = 700 :=
by
  sorry

end NUMINAMATH_GPT_weight_of_first_lift_l1231_123172


namespace NUMINAMATH_GPT_part1_part2_l1231_123198

open Set

variable (U : Set ℤ := {x | -3 ≤ x ∧ x ≤ 3})
variable (A : Set ℤ := {1, 2, 3})
variable (B : Set ℤ := {-1, 0, 1})
variable (C : Set ℤ := {-2, 0, 2})

theorem part1 : A ∪ (B ∩ C) = {0, 1, 2, 3} := by
  sorry

theorem part2 : A ∩ Uᶜ ∪ (B ∪ C) = {3} := by
  sorry

end NUMINAMATH_GPT_part1_part2_l1231_123198


namespace NUMINAMATH_GPT_volume_ratio_of_cube_and_cuboid_l1231_123136

theorem volume_ratio_of_cube_and_cuboid :
  let edge_length_meter := 1
  let edge_length_cm := edge_length_meter * 100 -- Convert meter to centimeters
  let cube_volume := edge_length_cm^3
  let cuboid_width := 50
  let cuboid_length := 50
  let cuboid_height := 20
  let cuboid_volume := cuboid_width * cuboid_length * cuboid_height
  cube_volume = 20 * cuboid_volume := 
by
  sorry

end NUMINAMATH_GPT_volume_ratio_of_cube_and_cuboid_l1231_123136


namespace NUMINAMATH_GPT_ratio_unchanged_l1231_123119

-- Define the initial ratio
def initial_ratio (a b : ℕ) : ℚ := a / b

-- Define the new ratio after transformation
def new_ratio (a b : ℕ) : ℚ := (3 * a) / (b / (1/3))

-- The theorem stating that the ratio remains unchanged
theorem ratio_unchanged (a b : ℕ) (hb : b ≠ 0) :
  initial_ratio a b = new_ratio a b :=
by
  sorry

end NUMINAMATH_GPT_ratio_unchanged_l1231_123119


namespace NUMINAMATH_GPT_find_line_eq_l1231_123134

-- Define the type for the line equation
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the given conditions
def given_point : ℝ × ℝ := (-3, -1)
def given_parallel_line : Line := { a := 1, b := -3, c := -1 }

-- Define what it means for two lines to be parallel
def are_parallel (L1 L2 : Line) : Prop :=
  L1.a * L2.b = L1.b * L2.a

-- Define what it means for a point to lie on the line
def lies_on_line (P : ℝ × ℝ) (L : Line) : Prop :=
  L.a * P.1 + L.b * P.2 + L.c = 0

-- Define the result line we need to prove
def result_line : Line := { a := 1, b := -3, c := 0 }

-- The final theorem statement
theorem find_line_eq : 
  ∃ (L : Line), are_parallel L given_parallel_line ∧ lies_on_line given_point L ∧ L = result_line := 
sorry

end NUMINAMATH_GPT_find_line_eq_l1231_123134


namespace NUMINAMATH_GPT_complex_div_conjugate_l1231_123147

theorem complex_div_conjugate (a b : ℂ) (h1 : a = 2 - I) (h2 : b = 1 + 2 * I) :
    a / b = -I := by
  sorry

end NUMINAMATH_GPT_complex_div_conjugate_l1231_123147


namespace NUMINAMATH_GPT_intersection_A_B_l1231_123176

def A : Set ℝ := {x | x < 3 * x - 1}
def B : Set ℝ := {x | -1 < x ∧ x < 3}

theorem intersection_A_B : (A ∩ B) = {x | x > 1 / 2 ∧ x < 3} :=
by sorry

end NUMINAMATH_GPT_intersection_A_B_l1231_123176


namespace NUMINAMATH_GPT_panda_bamboo_digestion_l1231_123190

theorem panda_bamboo_digestion (h : 16 = 0.40 * x) : x = 40 :=
by sorry

end NUMINAMATH_GPT_panda_bamboo_digestion_l1231_123190


namespace NUMINAMATH_GPT_purpose_of_LB_full_nutrient_medium_l1231_123154

/--
Given the experiment "Separation of Microorganisms in Soil Using Urea as a Nitrogen Source",
which involves both experimental and control groups with the following conditions:
- The variable in the experiment is the difference in the medium used.
- The experimental group uses a medium with urea as the only nitrogen source (selective medium).
- The control group uses a full-nutrient medium.

Prove that the purpose of preparing LB full-nutrient medium is to observe the types and numbers
of soil microorganisms that can grow under full-nutrient conditions.
-/
theorem purpose_of_LB_full_nutrient_medium
  (experiment: String) (experimental_variable: String) (experimental_group: String) (control_group: String)
  (H1: experiment = "Separation of Microorganisms in Soil Using Urea as a Nitrogen Source")
  (H2: experimental_variable = "medium")
  (H3: experimental_group = "medium with urea as the only nitrogen source (selective medium)")
  (H4: control_group = "full-nutrient medium") :
  purpose_of_preparing_LB_full_nutrient_medium = "observe the types and numbers of soil microorganisms that can grow under full-nutrient conditions" :=
sorry

end NUMINAMATH_GPT_purpose_of_LB_full_nutrient_medium_l1231_123154


namespace NUMINAMATH_GPT_find_integers_l1231_123149

theorem find_integers (x : ℤ) (h₁ : x ≠ 3) (h₂ : (x - 3) ∣ (x ^ 3 - 3)) :
  x = -21 ∨ x = -9 ∨ x = -5 ∨ x = -3 ∨ x = -1 ∨ x = 1 ∨ x = 2 ∨ x = 4 ∨ x = 5 ∨
  x = 7 ∨ x = 9 ∨ x = 11 ∨ x = 15 ∨ x = 27 :=
sorry

end NUMINAMATH_GPT_find_integers_l1231_123149


namespace NUMINAMATH_GPT_solution_set_for_f_gt_0_l1231_123166

noncomputable def f (x : ℝ) : ℝ := sorry

theorem solution_set_for_f_gt_0
  (odd_f : ∀ x : ℝ, f (-x) = -f x)
  (f_one_eq_zero : f 1 = 0)
  (ineq_f : ∀ x : ℝ, x > 0 → (x * (deriv^[2] f x) - f x) / x^2 > 0) :
  { x : ℝ | f x > 0 } = { x : ℝ | -1 < x ∧ x < 0 } ∪ { x : ℝ | 1 < x } :=
sorry

end NUMINAMATH_GPT_solution_set_for_f_gt_0_l1231_123166


namespace NUMINAMATH_GPT_sum_a6_a7_a8_is_32_l1231_123195

noncomputable def geom_seq (q : ℝ) (a₁ : ℝ) (n : ℕ) : ℝ := a₁ * q^(n-1)

theorem sum_a6_a7_a8_is_32 (q a₁ : ℝ) 
  (h1 : geom_seq q a₁ 1 + geom_seq q a₁ 2 + geom_seq q a₁ 3 = 1)
  (h2 : geom_seq q a₁ 2 + geom_seq q a₁ 3 + geom_seq q a₁ 4 = 2) :
  geom_seq q a₁ 6 + geom_seq q a₁ 7 + geom_seq q a₁ 8 = 32 :=
sorry

end NUMINAMATH_GPT_sum_a6_a7_a8_is_32_l1231_123195


namespace NUMINAMATH_GPT_no_infinite_seq_pos_int_l1231_123129

theorem no_infinite_seq_pos_int : 
  ¬∃ (a : ℕ → ℕ), 
  (∀ n : ℕ, 0 < a n) ∧ 
  ∀ n : ℕ, a (n+1) ^ 2 ≥ 2 * a n * a (n+2) :=
by
  sorry

end NUMINAMATH_GPT_no_infinite_seq_pos_int_l1231_123129


namespace NUMINAMATH_GPT_max_three_digit_sum_l1231_123145

theorem max_three_digit_sum (A B C : ℕ) (hA : A < 10) (hB : B < 10) (hC : C < 10) (h1 : A ≠ B) (h2 : B ≠ C) (h3 : C ≠ A) :
  110 * A + 10 * B + 3 * C ≤ 981 :=
sorry

end NUMINAMATH_GPT_max_three_digit_sum_l1231_123145


namespace NUMINAMATH_GPT_segments_in_proportion_l1231_123127

theorem segments_in_proportion (a b c d : ℝ) (ha : a = 1) (hb : b = 4) (hc : c = 2) (h : a / b = c / d) : d = 8 := 
by 
  sorry

end NUMINAMATH_GPT_segments_in_proportion_l1231_123127


namespace NUMINAMATH_GPT_coupon_probability_l1231_123197

theorem coupon_probability : 
  (Nat.choose 6 6 * Nat.choose 11 3) / Nat.choose 17 9 = 3 / 442 := 
by
  sorry

end NUMINAMATH_GPT_coupon_probability_l1231_123197


namespace NUMINAMATH_GPT_solution_set_of_equation_l1231_123102

theorem solution_set_of_equation (x : ℝ) : 
  (abs (2 * x - 1) = abs x + abs (x - 1)) ↔ (x ≤ 0 ∨ x ≥ 1) := 
by 
  sorry

end NUMINAMATH_GPT_solution_set_of_equation_l1231_123102


namespace NUMINAMATH_GPT_expand_expression_l1231_123110

variable (x y z : ℝ)

theorem expand_expression : (x + 5) * (3 * y + 2 * z + 15) = 3 * x * y + 2 * x * z + 15 * x + 15 * y + 10 * z + 75 := by
  sorry

end NUMINAMATH_GPT_expand_expression_l1231_123110


namespace NUMINAMATH_GPT_distance_apart_l1231_123111

def race_total_distance : ℕ := 1000
def distance_Arianna_ran : ℕ := 184

theorem distance_apart :
  race_total_distance - distance_Arianna_ran = 816 :=
by
  sorry

end NUMINAMATH_GPT_distance_apart_l1231_123111


namespace NUMINAMATH_GPT_y_time_to_complete_work_l1231_123109

-- Definitions of the conditions
def work_rate_x := 1 / 40
def work_done_by_x_in_8_days := 8 * work_rate_x
def remaining_work := 1 - work_done_by_x_in_8_days
def y_completion_time := 32
def work_rate_y := remaining_work / y_completion_time

-- Lean theorem
theorem y_time_to_complete_work :
  y_completion_time * work_rate_y = 1 →
  (1 / work_rate_y = 40) :=
by
  sorry

end NUMINAMATH_GPT_y_time_to_complete_work_l1231_123109


namespace NUMINAMATH_GPT_no_14_non_square_rectangles_l1231_123135

theorem no_14_non_square_rectangles (side_len : ℕ) 
    (h_side_len : side_len = 9) 
    (num_rectangles : ℕ) 
    (h_num_rectangles : num_rectangles = 14) 
    (min_side_len : ℕ → ℕ → Prop) 
    (h_min_side_len : ∀ l w, min_side_len l w → l ≥ 2 ∧ w ≥ 2) : 
    ¬ (∀ l w, min_side_len l w → l ≠ w) :=
by {
    sorry
}

end NUMINAMATH_GPT_no_14_non_square_rectangles_l1231_123135


namespace NUMINAMATH_GPT_race_distance_A_beats_C_l1231_123139

variables (race_distance1 race_distance2 race_distance3 : ℕ)
           (distance_AB distance_BC distance_AC : ℕ)

theorem race_distance_A_beats_C :
  race_distance1 = 500 →
  race_distance2 = 500 →
  distance_AB = 50 →
  distance_BC = 25 →
  distance_AC = 58 →
  race_distance3 = 400 :=
by
  sorry

end NUMINAMATH_GPT_race_distance_A_beats_C_l1231_123139


namespace NUMINAMATH_GPT_sandy_age_l1231_123114

variables (S M J : ℕ)

def Q1 : Prop := S = M - 14  -- Sandy is younger than Molly by 14 years
def Q2 : Prop := J = S + 6  -- John is older than Sandy by 6 years
def Q3 : Prop := 7 * M = 9 * S  -- The ratio of Sandy's age to Molly's age is 7:9
def Q4 : Prop := 5 * J = 6 * S  -- The ratio of Sandy's age to John's age is 5:6

theorem sandy_age (h1 : Q1 S M) (h2 : Q2 S J) (h3 : Q3 S M) (h4 : Q4 S J) : S = 49 :=
by sorry

end NUMINAMATH_GPT_sandy_age_l1231_123114


namespace NUMINAMATH_GPT_sandra_money_left_l1231_123185

def sandra_savings : ℕ := 10
def mother_gift : ℕ := 4
def father_gift : ℕ := 2 * mother_gift
def candy_cost : ℚ := 0.5
def jelly_bean_cost : ℚ := 0.2
def num_candies : ℕ := 14
def num_jelly_beans : ℕ := 20

def total_money : ℕ := sandra_savings + mother_gift + father_gift
def total_candy_cost : ℚ := num_candies * candy_cost
def total_jelly_bean_cost : ℚ := num_jelly_beans * jelly_bean_cost
def total_cost : ℚ := total_candy_cost + total_jelly_bean_cost
def money_left : ℚ := total_money - total_cost

theorem sandra_money_left : money_left = 11 := by
  sorry

end NUMINAMATH_GPT_sandra_money_left_l1231_123185


namespace NUMINAMATH_GPT_inequality_solution_l1231_123179

theorem inequality_solution {x : ℝ} : (x + 1) / x > 1 ↔ x > 0 := 
sorry

end NUMINAMATH_GPT_inequality_solution_l1231_123179


namespace NUMINAMATH_GPT_expression_zero_denominator_nonzero_l1231_123193

theorem expression_zero (x : ℝ) : 
  (2 * x - 6) = 0 ↔ x = 3 :=
by {
  sorry
  }

theorem denominator_nonzero (x : ℝ) : 
  x = 3 → (5 * x + 10) ≠ 0 :=
by {
  sorry
  }

end NUMINAMATH_GPT_expression_zero_denominator_nonzero_l1231_123193


namespace NUMINAMATH_GPT_probability_of_making_pro_shot_l1231_123108

-- Define the probabilities given in the problem
def P_free_throw : ℚ := 4 / 5
def P_high_school_3 : ℚ := 1 / 2
def P_at_least_one : ℚ := 0.9333333333333333

-- Define the unknown probability for professional 3-pointer
def P_pro := 1 / 3

-- Calculate the probability of missing each shot
def P_miss_free_throw : ℚ := 1 - P_free_throw
def P_miss_high_school_3 : ℚ := 1 - P_high_school_3
def P_miss_pro : ℚ := 1 - P_pro

-- Define the probability of missing all shots
def P_miss_all := P_miss_free_throw * P_miss_high_school_3 * P_miss_pro

-- Now state what needs to be proved
theorem probability_of_making_pro_shot :
  (1 - P_miss_all = P_at_least_one) → P_pro = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_making_pro_shot_l1231_123108


namespace NUMINAMATH_GPT_leopards_points_l1231_123164

variables (x y : ℕ)

theorem leopards_points (h₁ : x + y = 50) (h₂ : x - y = 28) : y = 11 := by
  sorry

end NUMINAMATH_GPT_leopards_points_l1231_123164


namespace NUMINAMATH_GPT_sin_double_angle_l1231_123144

theorem sin_double_angle (α : ℝ) (h : Real.sin α + Real.cos α = Real.sqrt 3 / 3) : 
  Real.sin (2 * α) = -2 / 3 := 
sorry

end NUMINAMATH_GPT_sin_double_angle_l1231_123144


namespace NUMINAMATH_GPT_jordan_probability_l1231_123184

-- Definitions based on conditions.
def total_students := 28
def enrolled_in_french := 20
def enrolled_in_spanish := 23
def enrolled_in_both := 17

-- Calculate students enrolled only in one language.
def only_french := enrolled_in_french - enrolled_in_both
def only_spanish := enrolled_in_spanish - enrolled_in_both

-- Calculation of combinations.
def total_combinations := Nat.choose total_students 2
def only_french_combinations := Nat.choose only_french 2
def only_spanish_combinations := Nat.choose only_spanish 2

-- Probability calculations.
def prob_both_one_language := (only_french_combinations + only_spanish_combinations) / total_combinations

def prob_both_languages : ℚ := 1 - prob_both_one_language

theorem jordan_probability :
  prob_both_languages = (20 : ℚ) / 21 := by
  sorry

end NUMINAMATH_GPT_jordan_probability_l1231_123184


namespace NUMINAMATH_GPT_reduced_rate_fraction_l1231_123187

-- Definitions
def hours_in_a_week := 7 * 24
def hours_with_reduced_rates_on_weekdays := (12 * 5)
def hours_with_reduced_rates_on_weekends := (24 * 2)

-- Question in form of theorem
theorem reduced_rate_fraction :
  (hours_with_reduced_rates_on_weekdays + hours_with_reduced_rates_on_weekends) / hours_in_a_week = 9 / 14 := 
by
  sorry

end NUMINAMATH_GPT_reduced_rate_fraction_l1231_123187


namespace NUMINAMATH_GPT_range_of_f1_3_l1231_123158

noncomputable def f (a b : ℝ) (x y : ℝ) : ℝ :=
  a * (x^3 + 3 * x) + b * (y^2 + 2 * y + 1)

theorem range_of_f1_3 (a b : ℝ)
  (h1 : 1 ≤ f a b 1 2 ∧ f a b 1 2 ≤ 2)
  (h2 : 2 ≤ f a b 3 4 ∧ f a b 3 4 ≤ 5):
  3 / 2 ≤ f a b 1 3 ∧ f a b 1 3 ≤ 4 :=
sorry

end NUMINAMATH_GPT_range_of_f1_3_l1231_123158


namespace NUMINAMATH_GPT_part1_part2_l1231_123181

open Set

variable {R : Type} [OrderedRing R]

def U : Set R := univ
def A : Set R := {x | x^2 - 2*x - 3 > 0}
def B : Set R := {x | 4 - x^2 <= 0}

theorem part1 : A ∩ B = {x | -2 ≤ x ∧ x < -1} :=
sorry

theorem part2 : (U \ A) ∪ (U \ B) = {x | x < -2 ∨ x > -1} :=
sorry

end NUMINAMATH_GPT_part1_part2_l1231_123181


namespace NUMINAMATH_GPT_find_M_l1231_123120

theorem find_M (x y z M : ℝ) 
  (h1 : x + y + z = 120) 
  (h2 : x - 10 = M) 
  (h3 : y + 10 = M) 
  (h4 : z / 10 = M) : 
  M = 10 := 
by
  sorry

end NUMINAMATH_GPT_find_M_l1231_123120


namespace NUMINAMATH_GPT_find_total_worth_of_stock_l1231_123191

theorem find_total_worth_of_stock (X : ℝ)
  (h1 : 0.20 * X * 0.10 = 0.02 * X)
  (h2 : 0.80 * X * 0.05 = 0.04 * X)
  (h3 : 0.04 * X - 0.02 * X = 200) :
  X = 10000 :=
sorry

end NUMINAMATH_GPT_find_total_worth_of_stock_l1231_123191


namespace NUMINAMATH_GPT_blueberry_pies_correct_l1231_123182

def total_pies := 36
def apple_pie_ratio := 3
def blueberry_pie_ratio := 4
def cherry_pie_ratio := 5

-- Total parts in the ratio
def total_ratio_parts := apple_pie_ratio + blueberry_pie_ratio + cherry_pie_ratio

-- Number of pies per part
noncomputable def pies_per_part := total_pies / total_ratio_parts

-- Number of blueberry pies
noncomputable def blueberry_pies := blueberry_pie_ratio * pies_per_part

theorem blueberry_pies_correct : blueberry_pies = 12 := 
by
  sorry

end NUMINAMATH_GPT_blueberry_pies_correct_l1231_123182


namespace NUMINAMATH_GPT_product_combination_count_l1231_123112

-- Definitions of the problem

-- There are 6 different types of cookies
def num_cookies : Nat := 6

-- There are 4 different types of milk
def num_milks : Nat := 4

-- Charlie will not order more than one of the same type
def charlie_order_limit : Nat := 1

-- Delta will only order cookies, including repeats of types
def delta_only_cookies : Bool := true

-- Prove that there are 2531 ways for Charlie and Delta to leave the store with 4 products collectively
theorem product_combination_count : 
  (number_of_ways : Nat) = 2531 
  := sorry

end NUMINAMATH_GPT_product_combination_count_l1231_123112


namespace NUMINAMATH_GPT_smallest_five_digit_congruent_two_mod_seventeen_l1231_123196

theorem smallest_five_digit_congruent_two_mod_seventeen : 
  ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ n % 17 = 2 ∧ n = 10013 :=
by
  sorry

end NUMINAMATH_GPT_smallest_five_digit_congruent_two_mod_seventeen_l1231_123196


namespace NUMINAMATH_GPT_election_winner_votes_l1231_123163

theorem election_winner_votes (V : ℝ) : (0.62 * V = 806) → (0.62 * V) - (0.38 * V) = 312 → 0.62 * V = 806 :=
by
  intro hWin hDiff
  exact hWin

end NUMINAMATH_GPT_election_winner_votes_l1231_123163


namespace NUMINAMATH_GPT_approx_num_chars_in_ten_thousand_units_l1231_123107

-- Define the number of characters in the book
def num_chars : ℕ := 731017

-- Define the conversion factor from characters to units of 'ten thousand'
def ten_thousand : ℕ := 10000

-- Define the number of characters in units of 'ten thousand'
def chars_in_ten_thousand_units : ℚ := num_chars / ten_thousand

-- Define the rounded number of units to the nearest whole number
def rounded_chars_in_ten_thousand_units : ℤ := round chars_in_ten_thousand_units

-- Theorem to state the approximate number of characters in units of 'ten thousand' is 73
theorem approx_num_chars_in_ten_thousand_units : rounded_chars_in_ten_thousand_units = 73 := 
by sorry

end NUMINAMATH_GPT_approx_num_chars_in_ten_thousand_units_l1231_123107


namespace NUMINAMATH_GPT_rohan_salary_l1231_123103

variable (S : ℝ)

theorem rohan_salary (h₁ : (0.20 * S = 2500)) : S = 12500 :=
by
  sorry

end NUMINAMATH_GPT_rohan_salary_l1231_123103


namespace NUMINAMATH_GPT_hcf_of_two_numbers_l1231_123138

theorem hcf_of_two_numbers (H L P : ℕ) (h1 : L = 160) (h2 : P = 2560) (h3 : H * L = P) : H = 16 :=
by
  sorry

end NUMINAMATH_GPT_hcf_of_two_numbers_l1231_123138


namespace NUMINAMATH_GPT_k_value_if_function_not_in_first_quadrant_l1231_123168

theorem k_value_if_function_not_in_first_quadrant : 
  ∀ k : ℝ, (∀ x : ℝ, x > 0 → (k - 2) * x ^ (|k|) + k ≤ 0) → k = -1 :=
by
  sorry

end NUMINAMATH_GPT_k_value_if_function_not_in_first_quadrant_l1231_123168


namespace NUMINAMATH_GPT_race_course_length_l1231_123106

variable (v_A v_B d : ℝ)

theorem race_course_length (h1 : v_A = 4 * v_B) (h2 : (d - 60) / v_B = d / v_A) : d = 80 := by
  sorry

end NUMINAMATH_GPT_race_course_length_l1231_123106


namespace NUMINAMATH_GPT_complex_number_problem_l1231_123171

variables {a b c x y z : ℂ}

theorem complex_number_problem (h1 : a = (b + c) / (x - 2))
    (h2 : b = (c + a) / (y - 2))
    (h3 : c = (a + b) / (z - 2))
    (h4 : x * y + y * z + z * x = 67)
    (h5 : x + y + z = 2010) :
    x * y * z = -5892 :=
sorry

end NUMINAMATH_GPT_complex_number_problem_l1231_123171


namespace NUMINAMATH_GPT_max_cylinder_volume_l1231_123123

/-- Given a rectangle with perimeter 18 cm, when rotating it around one side to form a cylinder, 
    the maximum volume of the cylinder and the corresponding side length of the rectangle. -/
theorem max_cylinder_volume (x y : ℝ) (h_perimeter : 2 * (x + y) = 18) (hx : x > 0) (hy : y > 0)
  (h_cylinder_volume : ∃ (V : ℝ), V = π * x * (y / 2)^2) :
  (x = 3 ∧ y = 6 ∧ ∀ V, V = 108 * π) := sorry

end NUMINAMATH_GPT_max_cylinder_volume_l1231_123123


namespace NUMINAMATH_GPT_ratio_problem_l1231_123194

variable (a b c d : ℝ)

theorem ratio_problem (h1 : a / b = 3) (h2 : b / c = 1 / 4) (h3 : c / d = 5) : d / a = 4 / 15 := 
sorry

end NUMINAMATH_GPT_ratio_problem_l1231_123194


namespace NUMINAMATH_GPT_cos_45_minus_cos_90_eq_sqrt2_over_2_l1231_123115

theorem cos_45_minus_cos_90_eq_sqrt2_over_2 :
  (Real.cos (45 * Real.pi / 180) - Real.cos (90 * Real.pi / 180)) = (Real.sqrt 2 / 2) :=
by
  have h1 : Real.cos (90 * Real.pi / 180) = 0 := by sorry
  have h2 : Real.cos (45 * Real.pi / 180) = Real.sqrt 2 / 2 := by sorry
  sorry

end NUMINAMATH_GPT_cos_45_minus_cos_90_eq_sqrt2_over_2_l1231_123115


namespace NUMINAMATH_GPT_present_age_ratio_l1231_123128

-- Define the variables and the conditions
variable (S M : ℕ)

-- Condition 1: Sandy's present age is 84 because she was 78 six years ago
def present_age_sandy := S = 84

-- Condition 2: Sixteen years from now, the ratio of their ages is 5:2
def age_ratio_16_years := (S + 16) * 2 = 5 * (M + 16)

-- The goal: The present age ratio of Sandy to Molly is 7:2
theorem present_age_ratio {S M : ℕ} (h1 : S = 84) (h2 : (S + 16) * 2 = 5 * (M + 16)) : S / M = 7 / 2 :=
by
  -- Integrating conditions
  have hS : S = 84 := h1
  have hR : (S + 16) * 2 = 5 * (M + 16) := h2
  -- We need a proof here, but we'll skip it for now
  sorry

end NUMINAMATH_GPT_present_age_ratio_l1231_123128


namespace NUMINAMATH_GPT_price_reduction_for_1920_profit_maximum_profit_calculation_l1231_123186

-- Definitions based on given conditions
def cost_price : ℝ := 12
def base_price : ℝ := 20
def base_quantity_sold : ℝ := 240
def increment_per_dollar : ℝ := 40

-- Profit function
def profit (x : ℝ) : ℝ := (base_price - cost_price - x) * (base_quantity_sold + increment_per_dollar * x)

-- Prove price reduction for $1920 profit per day
theorem price_reduction_for_1920_profit : ∃ x : ℝ, profit x = 1920 ∧ x = 8 := by
  sorry

-- Prove maximum profit calculation
theorem maximum_profit_calculation : ∃ x y : ℝ, x = 4 ∧ y = 2560 ∧ ∀ z, profit z ≤ y := by
  sorry

end NUMINAMATH_GPT_price_reduction_for_1920_profit_maximum_profit_calculation_l1231_123186


namespace NUMINAMATH_GPT_sum_of_ages_l1231_123173

theorem sum_of_ages {l t : ℕ} (h1 : t > l) (h2 : t * t * l = 72) : t + t + l = 14 :=
sorry

end NUMINAMATH_GPT_sum_of_ages_l1231_123173


namespace NUMINAMATH_GPT_pow_sum_geq_pow_prod_l1231_123101

theorem pow_sum_geq_pow_prod (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  x^5 + y^5 ≥ x^4 * y + x * y^4 :=
 by sorry

end NUMINAMATH_GPT_pow_sum_geq_pow_prod_l1231_123101


namespace NUMINAMATH_GPT_good_deed_done_by_C_l1231_123183

def did_good (A B C : Prop) := 
  (¬A ∧ ¬B ∧ C) ∨ (¬A ∧ B ∧ ¬C) ∨ (A ∧ ¬B ∧ ¬C)

def statement_A (B : Prop) := B
def statement_B (B : Prop) := ¬B
def statement_C (C : Prop) := ¬C

theorem good_deed_done_by_C (A B C : Prop)
  (h_deed : (did_good A B C))
  (h_statement : (statement_A B ∧ ¬statement_B B ∧ ¬statement_C C) ∨ 
                      (¬statement_A B ∧ statement_B B ∧ ¬statement_C C) ∨ 
                      (¬statement_A B ∧ ¬statement_B B ∧ statement_C C)) :
  C :=
by 
  sorry

end NUMINAMATH_GPT_good_deed_done_by_C_l1231_123183


namespace NUMINAMATH_GPT_scientific_notation_conversion_l1231_123199

theorem scientific_notation_conversion : 
  ∀ (n : ℝ), n = 1.8 * 10^8 → n = 180000000 :=
by
  intros n h
  sorry

end NUMINAMATH_GPT_scientific_notation_conversion_l1231_123199


namespace NUMINAMATH_GPT_reconstruct_quadrilateral_l1231_123148

def quadrilateralVectors (W W' X X' Y Y' Z Z' : ℝ) :=
  (W - Z = W/2 + Z'/2) ∧
  (X - Y = Y'/2 + X'/2) ∧
  (Y - X = Y'/2 + X'/2) ∧
  (Z - W = W/2 + Z'/2)

theorem reconstruct_quadrilateral (W W' X X' Y Y' Z Z' : ℝ) :
  quadrilateralVectors W W' X X' Y Y' Z Z' →
  W = (1/2) * W' + 0 * X' + 0 * Y' + (1/2) * Z' :=
sorry

end NUMINAMATH_GPT_reconstruct_quadrilateral_l1231_123148


namespace NUMINAMATH_GPT_min_value_expression_l1231_123125

theorem min_value_expression (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 2/x + 3/y = 1) :
  x/2 + y/3 = 4 :=
sorry

end NUMINAMATH_GPT_min_value_expression_l1231_123125


namespace NUMINAMATH_GPT_players_on_team_are_4_l1231_123118

noncomputable def number_of_players (score_old_record : ℕ) (rounds : ℕ) (score_first_9_rounds : ℕ) (final_round_diff : ℕ) :=
  let points_needed := score_old_record * rounds
  let points_final_needed := score_old_record - final_round_diff
  let total_points_needed := points_needed * 1
  let final_round_points_needed := total_points_needed - score_first_9_rounds
  let P := final_round_points_needed / points_final_needed
  P

theorem players_on_team_are_4 :
  number_of_players 287 10 10440 27 = 4 :=
by
  sorry

end NUMINAMATH_GPT_players_on_team_are_4_l1231_123118
