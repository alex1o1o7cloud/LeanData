import Mathlib

namespace NUMINAMATH_GPT_largest_multiple_of_15_less_than_500_l161_16182

theorem largest_multiple_of_15_less_than_500 : ∃ (n : ℕ), (n < 500) ∧ (15 ∣ n) ∧ (∀ m : ℕ, (m < 500) ∧ (15 ∣ m) → m ≤ n) ∧ n = 495 :=
by
  sorry

end NUMINAMATH_GPT_largest_multiple_of_15_less_than_500_l161_16182


namespace NUMINAMATH_GPT_possible_values_of_N_l161_16170

def is_valid_N (N : ℕ) : Prop :=
  (N > 22) ∧ (N ≤ 25)

theorem possible_values_of_N :
  {N : ℕ | is_valid_N N} = {23, 24, 25} :=
by
  sorry

end NUMINAMATH_GPT_possible_values_of_N_l161_16170


namespace NUMINAMATH_GPT_range_of_a_l161_16120

variable (a : ℝ) (f : ℝ → ℝ)
axiom func_def : ∀ x, f x = a^x
axiom a_pos : a > 0
axiom a_ne_one : a ≠ 1
axiom decreasing : ∀ m n : ℝ, m > n → f m < f n

theorem range_of_a : 0 < a ∧ a < 1 :=
sorry

end NUMINAMATH_GPT_range_of_a_l161_16120


namespace NUMINAMATH_GPT_cube_side_length_l161_16178

theorem cube_side_length (n : ℕ) (h : (6 * n^2) / (6 * n^3) = 1 / 3) : n = 3 :=
sorry

end NUMINAMATH_GPT_cube_side_length_l161_16178


namespace NUMINAMATH_GPT_boat_travel_times_l161_16130

theorem boat_travel_times (d_AB d_BC : ℕ) 
  (t_against_current t_with_current t_total_A t_total_C : ℕ) 
  (h_AB : d_AB = 3) (h_BC : d_BC = 3) 
  (h_against_current : t_against_current = 10) 
  (h_with_current : t_with_current = 8)
  (h_total_A : t_total_A = 24)
  (h_total_C : t_total_C = 72) :
  (t_total_A = 24 ∨ t_total_A = 72) ∧ (t_total_C = 24 ∨ t_total_C = 72) := 
by 
  sorry

end NUMINAMATH_GPT_boat_travel_times_l161_16130


namespace NUMINAMATH_GPT_solve_xy_l161_16148

theorem solve_xy : ∃ (x y : ℝ), x = 1 / 3 ∧ y = 2 / 3 ∧ x^2 + (1 - y)^2 + (x - y)^2 = 1 / 3 :=
by
  use 1 / 3, 2 / 3
  sorry

end NUMINAMATH_GPT_solve_xy_l161_16148


namespace NUMINAMATH_GPT_find_s_t_l161_16162

theorem find_s_t 
  (FG GH EH : ℝ)
  (angleE angleF : ℝ)
  (h1 : FG = 10)
  (h2 : GH = 15)
  (h3 : EH = 12)
  (h4 : angleE = 45)
  (h5 : angleF = 45)
  (s t : ℕ)
  (h6 : 12 + 7.5 * Real.sqrt 2 = s + Real.sqrt t) :
  s + t = 5637 :=
sorry

end NUMINAMATH_GPT_find_s_t_l161_16162


namespace NUMINAMATH_GPT_store_price_reduction_l161_16143

theorem store_price_reduction 
    (initial_price : ℝ) (initial_sales : ℕ) (price_reduction : ℝ)
    (sales_increase_factor : ℝ) (target_profit : ℝ)
    (x : ℝ) : (initial_price, initial_price - price_reduction, x) = (80, 50, 12) →
    sales_increase_factor = 20 →
    target_profit = 7920 →
    (30 - x) * (200 + sales_increase_factor * x / 2) = 7920 →
    x = 12 ∧ (initial_price - x) = 68 :=
by 
    intros h₁ h₂ h₃ h₄
    sorry

end NUMINAMATH_GPT_store_price_reduction_l161_16143


namespace NUMINAMATH_GPT_largest_n_l161_16136

def a_n (n : ℕ) (d_a : ℤ) : ℤ := 1 + (n-1) * d_a
def b_n (n : ℕ) (d_b : ℤ) : ℤ := 3 + (n-1) * d_b

theorem largest_n (d_a d_b : ℤ) (n : ℕ) :
  (a_n n d_a * b_n n d_b = 2304 ∧ a_n 1 d_a = 1 ∧ b_n 1 d_b = 3) 
  → n ≤ 20 := 
sorry

end NUMINAMATH_GPT_largest_n_l161_16136


namespace NUMINAMATH_GPT_integer_solutions_m3_eq_n3_plus_n_l161_16137

theorem integer_solutions_m3_eq_n3_plus_n (m n : ℤ) (h : m^3 = n^3 + n) : m = 0 ∧ n = 0 :=
sorry

end NUMINAMATH_GPT_integer_solutions_m3_eq_n3_plus_n_l161_16137


namespace NUMINAMATH_GPT_clean_car_time_l161_16128

theorem clean_car_time (t_outside : ℕ) (t_inside : ℕ) (h_outside : t_outside = 80) (h_inside : t_inside = t_outside / 4) : 
  t_outside + t_inside = 100 := 
by 
  sorry

end NUMINAMATH_GPT_clean_car_time_l161_16128


namespace NUMINAMATH_GPT_candy_mixture_price_l161_16123

theorem candy_mixture_price
  (price_first_per_kg : ℝ) (price_second_per_kg : ℝ) (weight_ratio : ℝ) (weight_second : ℝ) 
  (h1 : price_first_per_kg = 10) 
  (h2 : price_second_per_kg = 15) 
  (h3 : weight_ratio = 3) 
  : (price_first_per_kg * weight_ratio * weight_second + price_second_per_kg * weight_second) / 
    (weight_ratio * weight_second + weight_second) = 11.25 :=
by
  sorry

end NUMINAMATH_GPT_candy_mixture_price_l161_16123


namespace NUMINAMATH_GPT_value_range_f_in_0_to_4_l161_16111

noncomputable def f (x : ℝ) : ℝ := x^2 - 2*x + 2

theorem value_range_f_in_0_to_4 :
  ∀ (x : ℝ), (0 < x ∧ x ≤ 4) → (1 ≤ f x ∧ f x ≤ 10) :=
sorry

end NUMINAMATH_GPT_value_range_f_in_0_to_4_l161_16111


namespace NUMINAMATH_GPT_find_m_l161_16115

def a (m : ℝ) : ℝ × ℝ := (2 * m - 1, 3)
def b : ℝ × ℝ := (1, -1)
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem find_m (m : ℝ) (h : dot_product (a m) b = 2) : m = 3 :=
by sorry

end NUMINAMATH_GPT_find_m_l161_16115


namespace NUMINAMATH_GPT_num_pigs_on_farm_l161_16150

variables (P : ℕ)
def cows := 2 * P - 3
def goats := (2 * P - 3) + 6
def total_animals := P + cows P + goats P

theorem num_pigs_on_farm (h : total_animals P = 50) : P = 10 :=
sorry

end NUMINAMATH_GPT_num_pigs_on_farm_l161_16150


namespace NUMINAMATH_GPT_gregory_current_age_l161_16121

-- Given conditions
variables (D G y : ℕ)
axiom dm_is_three_times_greg_was (x : ℕ) : D = 3 * y
axiom future_age_sum : D + (3 * y) = 49
axiom greg_age_difference x y : D - (3 * y) = (3 * y) - x

-- Prove statement: Gregory's current age is 14
theorem gregory_current_age : G = 14 := by
  sorry

end NUMINAMATH_GPT_gregory_current_age_l161_16121


namespace NUMINAMATH_GPT_total_rats_l161_16155

theorem total_rats (Elodie_rats Hunter_rats Kenia_rats : ℕ) 
  (h1 : Elodie_rats = 30) 
  (h2 : Elodie_rats = Hunter_rats + 10)
  (h3 : Kenia_rats = 3 * (Elodie_rats + Hunter_rats)) :
  Elodie_rats + Hunter_rats + Kenia_rats = 200 :=
by
  sorry

end NUMINAMATH_GPT_total_rats_l161_16155


namespace NUMINAMATH_GPT_distinct_right_angles_l161_16151

theorem distinct_right_angles (n : ℕ) (h : n > 0) : 
  ∃ (a b c d : ℕ), (a + b + c + d ≥ 4 * (Int.sqrt n)) ∧ (a * c ≥ n) ∧ (b * d ≥ n) :=
by sorry

end NUMINAMATH_GPT_distinct_right_angles_l161_16151


namespace NUMINAMATH_GPT_coffee_shop_sales_l161_16117

def number_of_coffee_customers : Nat := 7
def price_per_coffee : Nat := 5

def number_of_tea_customers : Nat := 8
def price_per_tea : Nat := 4

def total_sales : Nat :=
  (number_of_coffee_customers * price_per_coffee)
  + (number_of_tea_customers * price_per_tea)

theorem coffee_shop_sales : total_sales = 67 := by
  sorry

end NUMINAMATH_GPT_coffee_shop_sales_l161_16117


namespace NUMINAMATH_GPT_elisa_target_amount_l161_16116

def elisa_current_amount : ℕ := 37
def elisa_additional_amount : ℕ := 16

theorem elisa_target_amount : elisa_current_amount + elisa_additional_amount = 53 :=
by
  sorry

end NUMINAMATH_GPT_elisa_target_amount_l161_16116


namespace NUMINAMATH_GPT_chris_pennies_count_l161_16168

theorem chris_pennies_count (a c : ℤ) 
  (h1 : c + 2 = 4 * (a - 2)) 
  (h2 : c - 2 = 3 * (a + 2)) : 
  c = 62 := 
by 
  -- The actual proof is omitted
  sorry

end NUMINAMATH_GPT_chris_pennies_count_l161_16168


namespace NUMINAMATH_GPT_subset_A_imp_range_a_disjoint_A_imp_range_a_l161_16129

-- Definition of sets A and B
def A : Set ℝ := {x | x^2 - 6*x + 8 < 0}
def B (a : ℝ) : Set ℝ := {x | (x - a)*(x - 3*a) < 0}

-- Proof problem for Question 1
theorem subset_A_imp_range_a (a : ℝ) (h : A ⊆ B a) : 
  (4 / 3) ≤ a ∧ a ≤ 2 ∧ a ≠ 0 :=
sorry

-- Proof problem for Question 2
theorem disjoint_A_imp_range_a (a : ℝ) (h : A ∩ B a = ∅) : 
  a ≤ (2 / 3) ∨ a ≥ 4 :=
sorry

end NUMINAMATH_GPT_subset_A_imp_range_a_disjoint_A_imp_range_a_l161_16129


namespace NUMINAMATH_GPT_least_x_value_l161_16124

variable (a b : ℕ)
variable (positive_int_a : 0 < a)
variable (positive_int_b : 0 < b)
variable (h : 2 * a^5 = 3 * b^2)

theorem least_x_value (h : 2 * a^5 = 3 * b^2) (positive_int_a : 0 < a) (positive_int_b : 0 < b) : ∃ x, x = 15552 ∧ x = 2 * a^5 ∧ x = 3 * b^2 :=
sorry

end NUMINAMATH_GPT_least_x_value_l161_16124


namespace NUMINAMATH_GPT_solution_existence_l161_16165

theorem solution_existence (m : ℤ) :
  (∀ x y : ℤ, 2 * x + (m - 1) * y = 3 ∧ (m + 1) * x + 4 * y = -3) ↔
  (m = -3 ∨ m = 3 → 
    (m = -3 → ∃ x y : ℤ, 2 * x + (m - 1) * y = 3 ∧ (m + 1) * x + 4 * y = -3) ∧
    (m = 3 → ¬∃ x y : ℤ, 2 * x + (m - 1) * y = 3 ∧ (m + 1) * x + 4 * y = -3)) := by
  sorry

end NUMINAMATH_GPT_solution_existence_l161_16165


namespace NUMINAMATH_GPT_volume_pyramid_ABC_l161_16119

structure Point where
  x : ℝ
  y : ℝ

def triangle_volume (A B C : Point) : ℝ :=
  -- The implementation would calculate the volume of the pyramid formed
  -- by folding along the midpoint sides.
  sorry

theorem volume_pyramid_ABC :
  let A := Point.mk 0 0
  let B := Point.mk 30 0
  let C := Point.mk 20 15
  triangle_volume A B C = 900 :=
by
  -- To be filled with the proof
  sorry

end NUMINAMATH_GPT_volume_pyramid_ABC_l161_16119


namespace NUMINAMATH_GPT_allocation_ways_l161_16191

theorem allocation_ways (programs : Finset ℕ) (grades : Finset ℕ) (h_programs : programs.card = 6) (h_grades : grades.card = 4) : 
  ∃ ways : ℕ, ways = 1080 := 
by 
  sorry

end NUMINAMATH_GPT_allocation_ways_l161_16191


namespace NUMINAMATH_GPT_sum_of_cubes_of_consecutive_even_integers_l161_16198

theorem sum_of_cubes_of_consecutive_even_integers 
    (x y z : ℕ) 
    (h1 : x % 2 = 0) 
    (h2 : y % 2 = 0) 
    (h3 : z % 2 = 0) 
    (h4 : y = x + 2) 
    (h5 : z = y + 2) 
    (h6 : x * y * z = 12 * (x + y + z)) : 
  x^3 + y^3 + z^3 = 8568 := 
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_sum_of_cubes_of_consecutive_even_integers_l161_16198


namespace NUMINAMATH_GPT_ted_and_mike_seeds_l161_16108

noncomputable def ted_morning_seeds (T : ℕ) (mike_morning_seeds : ℕ) (mike_afternoon_seeds : ℕ) (total_seeds : ℕ) : Prop :=
  mike_morning_seeds = 50 ∧
  mike_afternoon_seeds = 60 ∧
  total_seeds = 250 ∧
  T + (mike_afternoon_seeds - 20) + (mike_morning_seeds + mike_afternoon_seeds) = total_seeds ∧
  2 * mike_morning_seeds = T

theorem ted_and_mike_seeds :
  ∃ T : ℕ, ted_morning_seeds T 50 60 250 :=
by {
  sorry
}

end NUMINAMATH_GPT_ted_and_mike_seeds_l161_16108


namespace NUMINAMATH_GPT_cone_radius_l161_16176

open Real

theorem cone_radius
  (l : ℝ) (L : ℝ) (h_l : l = 5) (h_L : L = 15 * π) :
  ∃ r : ℝ, L = π * r * l ∧ r = 3 :=
by
  sorry

end NUMINAMATH_GPT_cone_radius_l161_16176


namespace NUMINAMATH_GPT_last_digit_2019_digit_number_l161_16146

theorem last_digit_2019_digit_number :
  ∃ n : ℕ → ℕ,  
    (∀ k, 0 ≤ k → k < 2018 → (n k * 10 + n (k + 1)) % 13 = 0) ∧ 
    n 0 = 6 ∧ 
    n 2018 = 2 :=
sorry

end NUMINAMATH_GPT_last_digit_2019_digit_number_l161_16146


namespace NUMINAMATH_GPT_divide_into_parts_l161_16184

theorem divide_into_parts (x y : ℚ) (h_sum : x + y = 10) (h_diff : y - x = 5) : 
  x = 5 / 2 ∧ y = 15 / 2 := 
sorry

end NUMINAMATH_GPT_divide_into_parts_l161_16184


namespace NUMINAMATH_GPT_largest_x_fraction_l161_16158

theorem largest_x_fraction (x : ℝ) (h : (⌊x⌋ : ℝ) / x = 11 / 12) : x ≤ 120 / 11 := by
  sorry

end NUMINAMATH_GPT_largest_x_fraction_l161_16158


namespace NUMINAMATH_GPT_no_primes_in_sequence_l161_16147

def P : ℕ := 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43 * 47 * 53 * 59 * 61

theorem no_primes_in_sequence :
  ∀ n : ℕ, 2 ≤ n ∧ n ≤ 59 → ¬ Nat.Prime (P + n) :=
by
  sorry

end NUMINAMATH_GPT_no_primes_in_sequence_l161_16147


namespace NUMINAMATH_GPT_find_sample_size_l161_16193

theorem find_sample_size
  (teachers : ℕ := 200)
  (male_students : ℕ := 1200)
  (female_students : ℕ := 1000)
  (sampled_females : ℕ := 80)
  (total_people := teachers + male_students + female_students)
  (ratio : sampled_females / female_students = n / total_people)
  : n = 192 := 
by
  sorry

end NUMINAMATH_GPT_find_sample_size_l161_16193


namespace NUMINAMATH_GPT_mom_age_when_jayson_born_l161_16175

theorem mom_age_when_jayson_born (jayson_age dad_age mom_age : ℕ) 
  (h1 : jayson_age = 10) 
  (h2 : dad_age = 4 * jayson_age)
  (h3 : mom_age = dad_age - 2) :
  mom_age - jayson_age = 28 :=
by
  sorry

end NUMINAMATH_GPT_mom_age_when_jayson_born_l161_16175


namespace NUMINAMATH_GPT_selling_price_per_pound_l161_16106

-- Definitions based on conditions
def cost_per_pound_type1 : ℝ := 2.00
def cost_per_pound_type2 : ℝ := 3.00
def weight_type1 : ℝ := 64
def weight_type2 : ℝ := 16
def total_weight : ℝ := 80

-- The selling price per pound of the mixture
theorem selling_price_per_pound :
  let total_cost := (weight_type1 * cost_per_pound_type1) + (weight_type2 * cost_per_pound_type2)
  (total_cost / total_weight) = 2.20 :=
by
  sorry

end NUMINAMATH_GPT_selling_price_per_pound_l161_16106


namespace NUMINAMATH_GPT_cos_45_deg_l161_16131

theorem cos_45_deg : Real.cos (Real.pi / 4) = Real.sqrt 2 / 2 :=
by
  sorry

end NUMINAMATH_GPT_cos_45_deg_l161_16131


namespace NUMINAMATH_GPT_least_number_of_tiles_l161_16187

-- Definitions for classroom dimensions
def classroom_length : ℕ := 624 -- in cm
def classroom_width : ℕ := 432 -- in cm

-- Definitions for tile dimensions
def rectangular_tile_length : ℕ := 60
def rectangular_tile_width : ℕ := 80
def triangular_tile_base : ℕ := 40
def triangular_tile_height : ℕ := 40

-- Definition for the area calculation
def area (length width : ℕ) : ℕ := length * width
def area_triangular_tile (base height : ℕ) : ℕ := (base * height) / 2

-- Define the area of the classroom and tiles
def classroom_area : ℕ := area classroom_length classroom_width
def rectangular_tile_area : ℕ := area rectangular_tile_length rectangular_tile_width
def triangular_tile_area : ℕ := area_triangular_tile triangular_tile_base triangular_tile_height

-- Define the number of tiles required
def number_of_rectangular_tiles : ℕ := (classroom_area + rectangular_tile_area - 1) / rectangular_tile_area -- ceiling division in lean
def number_of_triangular_tiles : ℕ := (classroom_area + triangular_tile_area - 1) / triangular_tile_area -- ceiling division in lean

-- Define the minimum number of tiles required
def minimum_number_of_tiles : ℕ := min number_of_rectangular_tiles number_of_triangular_tiles

-- The main theorem establishing the least number of tiles required
theorem least_number_of_tiles : minimum_number_of_tiles = 57 := by
    sorry

end NUMINAMATH_GPT_least_number_of_tiles_l161_16187


namespace NUMINAMATH_GPT_find_m_for_min_value_l161_16118

theorem find_m_for_min_value :
  ∃ (m : ℝ), ( ∀ x : ℝ, (y : ℝ) = m * x^2 - 4 * x + 1 → (∃ x_min : ℝ, (∀ x : ℝ, (m * x_min^2 - 4 * x_min + 1 ≤ m * x^2 - 4 * x + 1) → y = -3))) :=
sorry

end NUMINAMATH_GPT_find_m_for_min_value_l161_16118


namespace NUMINAMATH_GPT_second_projection_at_given_distance_l161_16156

structure Point :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

structure Line :=
  (point : Point)
  (direction : Point) -- Assume direction is given as a vector

def is_parallel (line1 line2 : Line) : Prop :=
  -- Function to check if two lines are parallel
  sorry

def distance (point1 point2 : Point) : ℝ := 
  -- Function to compute the distance between two points
  sorry

def first_projection_exists (M : Point) (a : Line) : Prop :=
  -- Check the projection outside the line a
  sorry

noncomputable def second_projection
  (M : Point)
  (a : Line)
  (d : ℝ)
  (h_parallel : is_parallel a (Line.mk ⟨0, 0, 0⟩ ⟨1, 0, 0⟩))
  (h_projection : first_projection_exists M a) :
  Point :=
  sorry

theorem second_projection_at_given_distance
  (M : Point)
  (a : Line)
  (d : ℝ)
  (h_parallel : is_parallel a (Line.mk ⟨0, 0, 0⟩ ⟨1, 0, 0⟩))
  (h_projection : first_projection_exists M a) :
  distance (second_projection M a d h_parallel h_projection) a.point = d :=
  sorry

end NUMINAMATH_GPT_second_projection_at_given_distance_l161_16156


namespace NUMINAMATH_GPT_modulo_calculation_l161_16107

theorem modulo_calculation : (68 * 97 * 113) % 25 = 23 := by
  sorry

end NUMINAMATH_GPT_modulo_calculation_l161_16107


namespace NUMINAMATH_GPT_cyclists_meet_time_l161_16104

theorem cyclists_meet_time 
  (v1 v2 : ℕ) (C : ℕ) (h1 : v1 = 7) (h2 : v2 = 8) (hC : C = 675) : 
  C / (v1 + v2) = 45 :=
by
  sorry

end NUMINAMATH_GPT_cyclists_meet_time_l161_16104


namespace NUMINAMATH_GPT_exist_column_remove_keeps_rows_distinct_l161_16100

theorem exist_column_remove_keeps_rows_distinct 
    (n : ℕ) 
    (table : Fin n → Fin n → Char) 
    (h_diff_rows : ∀ i j : Fin n, i ≠ j → ∃ k : Fin n, table i k ≠ table j k) 
    : ∃ col_to_remove : Fin n, ∀ i j : Fin n, i ≠ j → (table i ≠ table j) :=
sorry

end NUMINAMATH_GPT_exist_column_remove_keeps_rows_distinct_l161_16100


namespace NUMINAMATH_GPT_min_eccentricity_sum_l161_16189

def circle_O1 (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 16
def circle_O2 (x y r : ℝ) : Prop := x^2 + y^2 = r^2 ∧ 0 < r ∧ r < 2

def moving_circle_tangent (e1 e2 : ℝ) (r : ℝ) : Prop :=
  e1 = 2 / (4 - r) ∧ e2 = 2 / (4 + r)

theorem min_eccentricity_sum : ∃ (e1 e2 : ℝ) (r : ℝ), 
  circle_O1 x y ∧ circle_O2 x y r ∧ moving_circle_tangent e1 e2 r ∧
    e1 > e2 ∧ (e1 + 2 * e2) = (3 + 2 * Real.sqrt 2) / 4 :=
sorry

end NUMINAMATH_GPT_min_eccentricity_sum_l161_16189


namespace NUMINAMATH_GPT_solution_correct_l161_16112

noncomputable def solution_set : Set ℝ :=
  {x | (x < -2) ∨ (-1 < x ∧ x < 0) ∨ (1 < x)}

theorem solution_correct (x : ℝ) :
  (1 / (x * (x + 1)) - 1 / ((x + 1) * (x + 2))) < 1 / 4 ↔ (x < -2) ∨ (-1 < x ∧ x < 0) ∨ (1 < x) :=
by sorry

end NUMINAMATH_GPT_solution_correct_l161_16112


namespace NUMINAMATH_GPT_sum_opposite_signs_eq_zero_l161_16133

theorem sum_opposite_signs_eq_zero (x y : ℝ) (h : x * y < 0) : x + y = 0 :=
sorry

end NUMINAMATH_GPT_sum_opposite_signs_eq_zero_l161_16133


namespace NUMINAMATH_GPT_geometric_sequence_k_squared_l161_16183

theorem geometric_sequence_k_squared (a : ℕ → ℝ) (r : ℝ) (h_geom : ∀ n, a (n+1) = a n * r) (h5 : a 5 * a 8 * a 11 = k) : 
  k^2 = a 5 * a 6 * a 7 * a 9 * a 10 * a 11 := by
  sorry

end NUMINAMATH_GPT_geometric_sequence_k_squared_l161_16183


namespace NUMINAMATH_GPT_one_of_a_b_c_is_zero_l161_16145

theorem one_of_a_b_c_is_zero
  (a b c : ℝ)
  (h1 : (a + b) * (b + c) * (c + a) = a * b * c)
  (h2 : (a^9 + b^9) * (b^9 + c^9) * (c^9 + a^9) = (a * b * c)^9) :
  a = 0 ∨ b = 0 ∨ c = 0 :=
by
  sorry

end NUMINAMATH_GPT_one_of_a_b_c_is_zero_l161_16145


namespace NUMINAMATH_GPT_power_function_constant_l161_16188

theorem power_function_constant (k α : ℝ)
  (h : (1 / 2 : ℝ) ^ α * k = (Real.sqrt 2 / 2)) : k + α = 3 / 2 := by
  sorry

end NUMINAMATH_GPT_power_function_constant_l161_16188


namespace NUMINAMATH_GPT_deepak_present_age_l161_16186

-- Let R be Rahul's current age and D be Deepak's current age
variables (R D : ℕ)

-- Given conditions
def ratio_condition : Prop := (4 : ℚ) / 3 = (R : ℚ) / D
def rahul_future_age_condition : Prop := R + 6 = 50

-- Prove Deepak's present age D is 33 years
theorem deepak_present_age : ratio_condition R D ∧ rahul_future_age_condition R → D = 33 := 
sorry

end NUMINAMATH_GPT_deepak_present_age_l161_16186


namespace NUMINAMATH_GPT_find_number_l161_16160

theorem find_number : ∀ (x : ℝ), (0.15 * 0.30 * 0.50 * x = 99) → (x = 4400) :=
by
  intro x
  intro h
  sorry

end NUMINAMATH_GPT_find_number_l161_16160


namespace NUMINAMATH_GPT_sequences_meet_at_2017_l161_16177

-- Define the sequences for Paul and Penny
def paul_sequence (n : ℕ) : ℕ := 3 * n - 2
def penny_sequence (m : ℕ) : ℕ := 2022 - 5 * m

-- Statement to be proven
theorem sequences_meet_at_2017 : ∃ n m : ℕ, paul_sequence n = 2017 ∧ penny_sequence m = 2017 := by
  sorry

end NUMINAMATH_GPT_sequences_meet_at_2017_l161_16177


namespace NUMINAMATH_GPT_lowest_die_exactly_3_prob_l161_16114

noncomputable def fair_die_prob_at_least (n : ℕ) : ℚ :=
  if h : 1 ≤ n ∧ n ≤ 6 then (6 - n + 1) / 6 else 0

noncomputable def prob_lowest_die_exactly_3 : ℚ :=
  let p_at_least_3 := fair_die_prob_at_least 3
  let p_at_least_4 := fair_die_prob_at_least 4
  (p_at_least_3 ^ 4) - (p_at_least_4 ^ 4)

theorem lowest_die_exactly_3_prob :
  prob_lowest_die_exactly_3 = 175 / 1296 := by
  sorry

end NUMINAMATH_GPT_lowest_die_exactly_3_prob_l161_16114


namespace NUMINAMATH_GPT_compare_neg_fractions_l161_16192

theorem compare_neg_fractions : - (4 / 3 : ℚ) < - (5 / 4 : ℚ) := 
by sorry

end NUMINAMATH_GPT_compare_neg_fractions_l161_16192


namespace NUMINAMATH_GPT_range_of_y_l161_16164

theorem range_of_y (y : ℝ) (hy : y < 0) (h : ⌈y⌉ * ⌊y⌋ = 132) : -12 < y ∧ y < -11 := 
by 
  sorry

end NUMINAMATH_GPT_range_of_y_l161_16164


namespace NUMINAMATH_GPT_geometric_sequence_sum_a_l161_16102

theorem geometric_sequence_sum_a (a : ℤ) (S : ℕ → ℤ) (a_n : ℕ → ℤ) 
  (h1 : ∀ n : ℕ, S n = 2^n + a)
  (h2 : ∀ n : ℕ, a_n n = if n = 1 then S 1 else S n - S (n - 1)) :
  a = -1 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_a_l161_16102


namespace NUMINAMATH_GPT_triangular_weight_l161_16153

theorem triangular_weight (c t : ℝ) (h1 : c + t = 3 * c) (h2 : 4 * c + t = t + c + 90) : t = 60 := 
by sorry

end NUMINAMATH_GPT_triangular_weight_l161_16153


namespace NUMINAMATH_GPT_solve_arithmetic_sequence_l161_16167

theorem solve_arithmetic_sequence (x : ℝ) 
  (term1 term2 term3 : ℝ)
  (h1 : term1 = 3 / 4)
  (h2 : term2 = 2 * x - 3)
  (h3 : term3 = 7 * x) 
  (h_arith : term2 - term1 = term3 - term2) :
  x = -9 / 4 :=
by
  sorry

end NUMINAMATH_GPT_solve_arithmetic_sequence_l161_16167


namespace NUMINAMATH_GPT_number_of_even_divisors_of_factorial_eight_l161_16127

-- Definition of 8! and its prime factorization
def factorial_eight : ℕ := 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1
def prime_factorization_factorial_eight : Prop :=
  factorial_eight = 2^7 * 3^2 * 5 * 7

-- The main theorem statement
theorem number_of_even_divisors_of_factorial_eight :
  prime_factorization_factorial_eight →
  ∃ n, n = 7 * 3 * 2 * 2 ∧
  (∀ d, d ∣ factorial_eight → (∃ a b c d, 1 ≤ a ∧ a ≤ 7 ∧ 0 ≤ b ∧ b ≤ 2 ∧ 0 ≤ c ∧ c ≤ 1 ∧ 0 ≤ d ∧ d ≤ 1 ∧ d = 2^a * 3^b * 5^c * 7^d) →
  (7 * 3 * 2 * 2 = n)) :=
by
  intro h
  use 84
  sorry

end NUMINAMATH_GPT_number_of_even_divisors_of_factorial_eight_l161_16127


namespace NUMINAMATH_GPT_cherry_sodas_correct_l161_16161

/-
A cooler is filled with 24 cans of cherry soda and orange pop. 
There are twice as many cans of orange pop as there are of cherry soda. 
Prove that the number of cherry sodas is 8.
-/
def num_cherry_sodas (C O : ℕ) : Prop :=
  O = 2 * C ∧ C + O = 24 → C = 8

theorem cherry_sodas_correct (C O : ℕ) (h : O = 2 * C ∧ C + O = 24) : C = 8 :=
by
  sorry

end NUMINAMATH_GPT_cherry_sodas_correct_l161_16161


namespace NUMINAMATH_GPT_Amy_finish_time_l161_16125

-- Definitions and assumptions based on conditions
def Patrick_time : ℕ := 60
def Manu_time : ℕ := Patrick_time + 12
def Amy_time : ℕ := Manu_time / 2

-- Theorem statement to be proved
theorem Amy_finish_time : Amy_time = 36 :=
by
  sorry

end NUMINAMATH_GPT_Amy_finish_time_l161_16125


namespace NUMINAMATH_GPT_equilateral_cannot_be_obtuse_l161_16181

-- Additional definitions for clarity and mathematical rigor.
def is_equilateral (a b c : ℝ) : Prop := a = b ∧ b = c ∧ c = a
def is_obtuse (A B C : ℝ) : Prop := 
    (A > 90 ∧ B < 90 ∧ C < 90) ∨ 
    (B > 90 ∧ A < 90 ∧ C < 90) ∨
    (C > 90 ∧ A < 90 ∧ B < 90)

-- Theorem statement
theorem equilateral_cannot_be_obtuse (a b c : ℝ) (A B C : ℝ) :
  is_equilateral a b c → 
  (A + B + C = 180) → 
  (A = B ∧ B = C) → 
  ¬ is_obtuse A B C :=
by { sorry } -- Proof is not necessary as per instruction.

end NUMINAMATH_GPT_equilateral_cannot_be_obtuse_l161_16181


namespace NUMINAMATH_GPT_number_of_students_with_no_pets_l161_16138

-- Define the number of students in the class
def total_students : ℕ := 25

-- Define the number of students with cats
def students_with_cats : ℕ := (3 * total_students) / 5

-- Define the number of students with dogs
def students_with_dogs : ℕ := (20 * total_students) / 100

-- Define the number of students with elephants
def students_with_elephants : ℕ := 3

-- Calculate the number of students with no pets
def students_with_no_pets : ℕ := total_students - (students_with_cats + students_with_dogs + students_with_elephants)

-- Statement to be proved
theorem number_of_students_with_no_pets : students_with_no_pets = 2 :=
sorry

end NUMINAMATH_GPT_number_of_students_with_no_pets_l161_16138


namespace NUMINAMATH_GPT_cost_of_bananas_l161_16172

theorem cost_of_bananas (A B : ℕ) (h1 : 2 * A + B = 7) (h2 : A + B = 5) : B = 3 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_bananas_l161_16172


namespace NUMINAMATH_GPT_min_value_fraction_l161_16122

theorem min_value_fraction (x y : ℝ) (hx : -3 ≤ x ∧ x ≤ 1) (hy : -1 ≤ y ∧ y ≤ 3) : 
  ∃ v, v = (x + y) / x ∧ v = -2 := 
by 
  sorry

end NUMINAMATH_GPT_min_value_fraction_l161_16122


namespace NUMINAMATH_GPT_largest_consecutive_positive_elements_l161_16174

theorem largest_consecutive_positive_elements (a : ℕ → ℝ)
  (h₁ : ∀ n ≥ 2, a n = a (n-1) + a (n+2)) :
  ∃ m, m = 5 ∧ ∀ k < m, a k > 0 :=
sorry

end NUMINAMATH_GPT_largest_consecutive_positive_elements_l161_16174


namespace NUMINAMATH_GPT_ab_not_divisible_by_5_then_neither_divisible_l161_16180

theorem ab_not_divisible_by_5_then_neither_divisible (a b : ℕ) : ¬(¬(5 ∣ a) ∧ ¬(5 ∣ b)) → ¬(5 ∣ (a * b)) :=
by
  -- Mathematical statement for proof by contradiction:
  have H1: ¬(¬(5 ∣ a) ∧ ¬(5 ∣ b)) := sorry
  -- Rest of the proof would go here  
  sorry

end NUMINAMATH_GPT_ab_not_divisible_by_5_then_neither_divisible_l161_16180


namespace NUMINAMATH_GPT_joes_mean_score_is_88_83_l161_16166

def joesQuizScores : List ℕ := [88, 92, 95, 81, 90, 87]

noncomputable def mean (lst : List ℕ) : ℝ := (lst.sum : ℝ) / lst.length

theorem joes_mean_score_is_88_83 :
  mean joesQuizScores = 88.83 := 
sorry

end NUMINAMATH_GPT_joes_mean_score_is_88_83_l161_16166


namespace NUMINAMATH_GPT_ellen_bought_chairs_l161_16134

-- Define the conditions
def cost_per_chair : ℕ := 15
def total_amount_spent : ℕ := 180

-- State the theorem to be proven
theorem ellen_bought_chairs :
  (total_amount_spent / cost_per_chair = 12) := 
sorry

end NUMINAMATH_GPT_ellen_bought_chairs_l161_16134


namespace NUMINAMATH_GPT_stormi_needs_more_money_l161_16163

theorem stormi_needs_more_money
  (cars_washed : ℕ) (price_per_car : ℕ)
  (lawns_mowed : ℕ) (price_per_lawn : ℕ)
  (bike_cost : ℕ)
  (h1 : cars_washed = 3)
  (h2 : price_per_car = 10)
  (h3 : lawns_mowed = 2)
  (h4 : price_per_lawn = 13)
  (h5 : bike_cost = 80) : 
  bike_cost - (cars_washed * price_per_car + lawns_mowed * price_per_lawn) = 24 := by
  sorry

end NUMINAMATH_GPT_stormi_needs_more_money_l161_16163


namespace NUMINAMATH_GPT_sum_of_translated_parabolas_l161_16113

noncomputable def parabola_equation (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

noncomputable def reflected_parabola (a b c : ℝ) (x : ℝ) : ℝ := - (a * x^2 + b * x + c)

noncomputable def translated_right (a b c : ℝ) (x : ℝ) : ℝ := parabola_equation a b c (x - 3)

noncomputable def translated_left (a b c : ℝ) (x : ℝ) : ℝ := reflected_parabola a b c (x + 3)

theorem sum_of_translated_parabolas (a b c x : ℝ) : 
  (translated_right a b c x) + (translated_left a b c x) = -12 * a * x - 6 * b :=
sorry

end NUMINAMATH_GPT_sum_of_translated_parabolas_l161_16113


namespace NUMINAMATH_GPT_train_cable_car_distance_and_speeds_l161_16157
-- Import necessary libraries

-- Defining the variables and conditions
variables (s v1 v2 : ℝ)
variables (half_hour_sym_dist additional_distance quarter_hour_meet : ℝ)

-- Defining the conditions
def conditions :=
  (half_hour_sym_dist = v1 * (1 / 2) + v2 * (1 / 2)) ∧
  (additional_distance = 2 / v2) ∧
  (quarter_hour_meet = 1 / 4) ∧
  (v1 + v2 = 2 * s) ∧
  (v2 * (additional_distance + half_hour_sym_dist) = (v1 * (additional_distance + half_hour_sym_dist) - s)) ∧
  ((v1 + v2) * (half_hour_sym_dist + additional_distance + quarter_hour_meet) = 2 * s)

-- Proving the statement
theorem train_cable_car_distance_and_speeds
  (h : conditions s v1 v2 half_hour_sym_dist additional_distance quarter_hour_meet) :
  s = 24 ∧ v1 = 40 ∧ v2 = 8 := sorry

end NUMINAMATH_GPT_train_cable_car_distance_and_speeds_l161_16157


namespace NUMINAMATH_GPT_number_of_divisors_of_8_fact_l161_16140

theorem number_of_divisors_of_8_fact: 
  let n := 8
  let fact := Nat.factorial n
  fact = (2^7) * (3^2) * (5^1) * (7^1) -> 
  (7 + 1) * (2 + 1) * (1 + 1) * (1 + 1) = 96 := by
  sorry

end NUMINAMATH_GPT_number_of_divisors_of_8_fact_l161_16140


namespace NUMINAMATH_GPT_largest_three_digit_number_divisible_by_8_l161_16141

-- Define the properties of a number being a three-digit number
def isThreeDigitNumber (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

-- Define the property of a number being divisible by 8
def isDivisibleBy8 (n : ℕ) : Prop := n % 8 = 0

-- The theorem we want to prove: the largest three-digit number divisible by 8 is 992
theorem largest_three_digit_number_divisible_by_8 : ∃ n, isThreeDigitNumber n ∧ isDivisibleBy8 n ∧ (∀ m, isThreeDigitNumber m ∧ isDivisibleBy8 m → m ≤ 992) :=
  sorry

end NUMINAMATH_GPT_largest_three_digit_number_divisible_by_8_l161_16141


namespace NUMINAMATH_GPT_number_of_solutions_l161_16199

theorem number_of_solutions (n : ℕ) (hn : 0 < n) :
  ∃ k : ℕ, k = 2 + 4 * n ∧ (∃ (x y : ℤ), x ^ 2 + 2016 * y ^ 2 = 2017 ^ n) :=
by
  sorry

end NUMINAMATH_GPT_number_of_solutions_l161_16199


namespace NUMINAMATH_GPT_speed_of_man_in_still_water_l161_16179

theorem speed_of_man_in_still_water 
  (V_m V_s : ℝ)
  (h1 : 6 = V_m + V_s)
  (h2 : 4 = V_m - V_s) : 
  V_m = 5 := 
by 
  sorry

end NUMINAMATH_GPT_speed_of_man_in_still_water_l161_16179


namespace NUMINAMATH_GPT_coordinates_of_P_l161_16185

-- Define the conditions and the question as a Lean theorem
theorem coordinates_of_P (m : ℝ) (P : ℝ × ℝ) (h1 : P = (m + 3, m + 1)) (h2 : P.2 = 0) :
  P = (2, 0) := 
sorry

end NUMINAMATH_GPT_coordinates_of_P_l161_16185


namespace NUMINAMATH_GPT_square_side_4_FP_length_l161_16144

theorem square_side_4_FP_length (EF GH EP FP GP : ℝ) :
  EF = 4 ∧ GH = 4 ∧ EP = 4 ∧ GP = 4 ∧
  (1 / 2) * EP * 2 = 4 → FP = 2 * Real.sqrt 5 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_square_side_4_FP_length_l161_16144


namespace NUMINAMATH_GPT_grain_storage_bins_total_l161_16132

theorem grain_storage_bins_total
  (b20 : ℕ) (b20_tonnage : ℕ) (b15_tonnage : ℕ) (total_capacity : ℕ) (b20_count : ℕ)
  (h_b20_capacity : b20_count * b20_tonnage = b20)
  (h_total_capacity : b20 + (total_capacity - b20) = total_capacity)
  (h_b20_given : b20_count = 12)
  (h_b20_tonnage : b20_tonnage = 20)
  (h_b15_tonnage : b15_tonnage = 15)
  (h_total_capacity_given : total_capacity = 510) :
  ∃ b_total : ℕ, b_total = b20_count + ((total_capacity - (b20_count * b20_tonnage)) / b15_tonnage) ∧ b_total = 30 :=
by
  sorry

end NUMINAMATH_GPT_grain_storage_bins_total_l161_16132


namespace NUMINAMATH_GPT_exam_rule_l161_16110

variable (P R Q : Prop)

theorem exam_rule (hp : P ∧ R → Q) : ¬ Q → ¬ P ∨ ¬ R :=
by
  sorry

end NUMINAMATH_GPT_exam_rule_l161_16110


namespace NUMINAMATH_GPT_perfect_square_n_l161_16149

theorem perfect_square_n (n : ℤ) (h1 : n > 0) (h2 : ∃ k : ℤ, n^2 + 19 * n + 48 = k^2) : n = 33 :=
sorry

end NUMINAMATH_GPT_perfect_square_n_l161_16149


namespace NUMINAMATH_GPT_value_of_x_plus_y_l161_16139

theorem value_of_x_plus_y (x y : ℝ) (h : |x - 1| + (y - 2)^2 = 0) : x + y = 3 := by
  sorry

end NUMINAMATH_GPT_value_of_x_plus_y_l161_16139


namespace NUMINAMATH_GPT_value_of_f_at_5_l161_16154

def f (x : ℤ) : ℤ := x^3 - x^2 + x

theorem value_of_f_at_5 : f 5 = 105 := by
  sorry

end NUMINAMATH_GPT_value_of_f_at_5_l161_16154


namespace NUMINAMATH_GPT_solve_equation_l161_16126

theorem solve_equation : ∀ x : ℝ, 3 * x * (x - 1) = 2 * x - 2 ↔ (x = 1 ∨ x = 2 / 3) := 
by 
  intro x
  sorry

end NUMINAMATH_GPT_solve_equation_l161_16126


namespace NUMINAMATH_GPT_find_b_l161_16109

variables {a b : ℝ}

theorem find_b (h1 : (x - 3) * (x - a) = x^2 - b * x - 10) : b = -1/3 :=
  sorry

end NUMINAMATH_GPT_find_b_l161_16109


namespace NUMINAMATH_GPT_interval_of_decrease_l161_16135

noncomputable def f (x : ℝ) := x * Real.exp x + 1

theorem interval_of_decrease : {x : ℝ | x < -1} = {x : ℝ | (x + 1) * Real.exp x < 0} :=
by
  sorry

end NUMINAMATH_GPT_interval_of_decrease_l161_16135


namespace NUMINAMATH_GPT_solution_l161_16142

-- Define M and N according to the given conditions
def M : Set ℝ := {x | x < 0 ∨ x > 2}
def N : Set ℝ := {x | x ≥ 1}

-- Define the complement of M in Real numbers
def complementM : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

-- Define the union of the complement of M and N
def problem_statement : Set ℝ := complementM ∪ N

-- State the theorem
theorem solution :
  problem_statement = { x | x ≥ 0 } :=
by
  sorry

end NUMINAMATH_GPT_solution_l161_16142


namespace NUMINAMATH_GPT_inhabitable_fraction_l161_16101

theorem inhabitable_fraction 
  (total_land_fraction : ℚ)
  (inhabitable_land_fraction : ℚ)
  (h1 : total_land_fraction = 1 / 3)
  (h2 : inhabitable_land_fraction = 3 / 4):
  total_land_fraction * inhabitable_land_fraction = 1 / 4 := 
by
  sorry

end NUMINAMATH_GPT_inhabitable_fraction_l161_16101


namespace NUMINAMATH_GPT_max_of_inverse_power_sums_l161_16105

theorem max_of_inverse_power_sums (s p r1 r2 : ℝ) 
  (h_eq_roots : r1 + r2 = s ∧ r1 * r2 = p)
  (h_eq_powers : ∀ n : ℕ, 1 ≤ n ∧ n ≤ 2023 → r1^n + r2^n = s) :
  1 / r1^(2024:ℕ) + 1 / r2^(2024:ℕ) ≤ 2 :=
sorry

end NUMINAMATH_GPT_max_of_inverse_power_sums_l161_16105


namespace NUMINAMATH_GPT_length_of_base_AD_l161_16196

-- Definitions based on the conditions
def isosceles_trapezoid (A B C D : Type) : Prop := sorry -- Implementation of an isosceles trapezoid
def length_of_lateral_side (A B C D : Type) : ℝ := 40 -- The lateral side is 40 cm
def angle_BAC (A B C D : Type) : ℝ := 45 -- The angle ∠BAC is 45 degrees
def bisector_O_center (O A B D M : Type) : Prop := sorry -- Implementation that O is the center of circumscribed circle and lies on bisector

-- Main theorem based on the derived problem statement
theorem length_of_base_AD (A B C D O M : Type) 
  (h_iso_trapezoid : isosceles_trapezoid A B C D)
  (h_length_lateral : length_of_lateral_side A B C D = 40)
  (h_angle_BAC : angle_BAC A B C D = 45)
  (h_O_center_bisector : bisector_O_center O A B D M)
  : ℝ :=
  20 * (Real.sqrt 6 + Real.sqrt 2)

end NUMINAMATH_GPT_length_of_base_AD_l161_16196


namespace NUMINAMATH_GPT_revenue_change_l161_16190

theorem revenue_change (x : ℝ) 
  (increase_in_1996 : ∀ R : ℝ, R * (1 + x/100) > R) 
  (decrease_in_1997 : ∀ R : ℝ, R * (1 + x/100) * (1 - x/100) < R * (1 + x/100)) 
  (decrease_from_1995_to_1997 : ∀ R : ℝ, R * (1 + x/100) * (1 - x/100) = R * 0.96): 
  x = 20 :=
by
  sorry

end NUMINAMATH_GPT_revenue_change_l161_16190


namespace NUMINAMATH_GPT_germination_percentage_l161_16173

theorem germination_percentage :
  ∀ (seeds_plot1 seeds_plot2 germination_rate1 germination_rate2 : ℝ),
    seeds_plot1 = 300 →
    seeds_plot2 = 200 →
    germination_rate1 = 0.30 →
    germination_rate2 = 0.35 →
    ((germination_rate1 * seeds_plot1 + germination_rate2 * seeds_plot2) / (seeds_plot1 + seeds_plot2)) * 100 = 32 :=
by
  intros seeds_plot1 seeds_plot2 germination_rate1 germination_rate2 h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_germination_percentage_l161_16173


namespace NUMINAMATH_GPT_sum_mnp_is_405_l161_16152

theorem sum_mnp_is_405 :
  let C1_radius := 4
  let C2_radius := 10
  let C3_radius := C1_radius + C2_radius
  let chord_length := (8 * Real.sqrt 390) / 7
  ∃ m n p : ℕ,
    m * Real.sqrt n / p = chord_length ∧
    m.gcd p = 1 ∧
    (∀ k : ℕ, k^2 ∣ n → k = 1) ∧
    m + n + p = 405 :=
by
  sorry

end NUMINAMATH_GPT_sum_mnp_is_405_l161_16152


namespace NUMINAMATH_GPT_path_area_and_cost_correct_l161_16171

-- Define the given conditions
def length_field : ℝ := 75
def width_field : ℝ := 55
def path_width : ℝ := 2.5
def cost_per_sq_meter : ℝ := 7

-- Calculate new dimensions including the path
def length_including_path : ℝ := length_field + 2 * path_width
def width_including_path : ℝ := width_field + 2 * path_width

-- Calculate areas
def area_entire_field : ℝ := length_including_path * width_including_path
def area_grass_field : ℝ := length_field * width_field
def area_path : ℝ := area_entire_field - area_grass_field

-- Calculate cost
def cost_of_path : ℝ := area_path * cost_per_sq_meter

theorem path_area_and_cost_correct : 
  area_path = 675 ∧ cost_of_path = 4725 :=
by
  sorry

end NUMINAMATH_GPT_path_area_and_cost_correct_l161_16171


namespace NUMINAMATH_GPT_ed_pets_count_l161_16194

theorem ed_pets_count : 
  let dogs := 2 
  let cats := 3 
  let fish := 2 * (cats + dogs) 
  let birds := dogs * cats 
  dogs + cats + fish + birds = 21 := 
by
  sorry

end NUMINAMATH_GPT_ed_pets_count_l161_16194


namespace NUMINAMATH_GPT_sum_of_primes_less_than_20_l161_16169

theorem sum_of_primes_less_than_20 :
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19 = 77) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_primes_less_than_20_l161_16169


namespace NUMINAMATH_GPT_carla_needs_30_leaves_l161_16195

-- Definitions of the conditions
def items_per_day : Nat := 5
def total_days : Nat := 10
def total_bugs : Nat := 20

-- Maths problem to be proved
theorem carla_needs_30_leaves :
  let total_items := items_per_day * total_days
  let required_leaves := total_items - total_bugs
  required_leaves = 30 :=
by
  sorry

end NUMINAMATH_GPT_carla_needs_30_leaves_l161_16195


namespace NUMINAMATH_GPT_solve_m_problem_l161_16103

theorem solve_m_problem :
  (∃ x : ℝ, -1 < x ∧ x < 1 ∧ x^2 - x - m = 0) →
  m ∈ Set.Ico (-1/4 : ℝ) 2 :=
sorry

end NUMINAMATH_GPT_solve_m_problem_l161_16103


namespace NUMINAMATH_GPT_ice_cream_children_count_ice_cream_girls_count_l161_16159

-- Proof Problem for part (a)
theorem ice_cream_children_count (n : ℕ) (h : 3 * n = 24) : n = 8 := sorry

-- Proof Problem for part (b)
theorem ice_cream_girls_count (x y : ℕ) (h : x + y = 8) 
  (hx_even : x % 2 = 0) (hy_even : y % 2 = 0) (hx_pos : x > 0) (hxy : x < y) : y = 6 := sorry

end NUMINAMATH_GPT_ice_cream_children_count_ice_cream_girls_count_l161_16159


namespace NUMINAMATH_GPT_perimeter_of_regular_polygon_l161_16197

theorem perimeter_of_regular_polygon
  (side_length : ℕ)
  (exterior_angle : ℕ)
  (h1 : exterior_angle = 90)
  (h2 : side_length = 7) :
  4 * side_length = 28 :=
by
  sorry

end NUMINAMATH_GPT_perimeter_of_regular_polygon_l161_16197
