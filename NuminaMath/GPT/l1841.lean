import Mathlib

namespace NUMINAMATH_GPT_total_pages_read_l1841_184143

-- Define the average pages read by Lucas for the first four days.
def day1_4_avg : ℕ := 42

-- Define the average pages read by Lucas for the next two days.
def day5_6_avg : ℕ := 50

-- Define the pages read on the last day.
def day7 : ℕ := 30

-- Define the total number of days for which measurement is provided.
def total_days : ℕ := 7

-- Prove that the total number of pages Lucas read is 298.
theorem total_pages_read : 
  4 * day1_4_avg + 2 * day5_6_avg + day7 = 298 := 
by 
  sorry

end NUMINAMATH_GPT_total_pages_read_l1841_184143


namespace NUMINAMATH_GPT_helicopter_rental_cost_l1841_184174

theorem helicopter_rental_cost
  (hours_per_day : ℕ)
  (total_days : ℕ)
  (total_cost : ℕ)
  (H1 : hours_per_day = 2)
  (H2 : total_days = 3)
  (H3 : total_cost = 450) :
  total_cost / (hours_per_day * total_days) = 75 :=
by
  sorry

end NUMINAMATH_GPT_helicopter_rental_cost_l1841_184174


namespace NUMINAMATH_GPT_lukas_avg_points_per_game_l1841_184154

theorem lukas_avg_points_per_game (total_points games_played : ℕ) (h_total_points : total_points = 60) (h_games_played : games_played = 5) :
  (total_points / games_played = 12) :=
by
  sorry

end NUMINAMATH_GPT_lukas_avg_points_per_game_l1841_184154


namespace NUMINAMATH_GPT_function_values_at_mean_l1841_184109

noncomputable def f (x : ℝ) : ℝ := x^2 - 10 * x + 16

theorem function_values_at_mean (x₁ x₂ : ℝ) (h₁ : x₁ = 8) (h₂ : x₂ = 2) :
  let x' := (x₁ + x₂) / 2
  let x'' := Real.sqrt (x₁ * x₂)
  f x' = -9 ∧ f x'' = -8 := by
  let x' := (x₁ + x₂) / 2
  let x'' := Real.sqrt (x₁ * x₂)
  have hx' : x' = 5 := sorry
  have hx'' : x'' = 4 := sorry
  have hf_x' : f x' = -9 := sorry
  have hf_x'' : f x'' = -8 := sorry
  exact ⟨hf_x', hf_x''⟩

end NUMINAMATH_GPT_function_values_at_mean_l1841_184109


namespace NUMINAMATH_GPT_find_number_l1841_184126

theorem find_number (x : ℝ) (h : (x / 4) + 9 = 15) : x = 24 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l1841_184126


namespace NUMINAMATH_GPT_combined_PPF_two_females_combined_PPF_two_males_combined_PPF_male_female_l1841_184169

-- Definition: Combined PPF for two females
theorem combined_PPF_two_females (K : ℝ) (h : K ≤ 40) :
  (∀ K₁ K₂, K = K₁ + K₂ →  40 - 2 * K₁ + 40 - 2 * K₂ = 80 - 2 * K) := sorry

-- Definition: Combined PPF for two males
theorem combined_PPF_two_males (K : ℝ) (h : K ≤ 16) :
  (∀ K₁ K₂, K₁ = 0.5 * K → K₂ = 0.5 * K → 64 - K₁^2 + 64 - K₂^2 = 128 - 0.5 * K^2) := sorry

-- Definition: Combined PPF for one male and one female (piecewise)
theorem combined_PPF_male_female (K : ℝ) :
  (K ≤ 1 → (∀ K₁ K₂, K₁ = K → K₂ = 0 → 64 - K₁^2 + 40 - 2 * K₂ = 104 - K^2)) ∧
  (1 < K ∧ K ≤ 21 → (∀ K₁ K₂, K₁ = 1 → K₂ = K - 1 → 64 - K₁^2 + 40 - 2 * K₂ = 105 - 2 * K)) ∧
  (21 < K ∧ K ≤ 28 → (∀ K₁ K₂, K₁ = K - 20 → K₂ = 20 → 64 - K₁^2 + 40 - 2 * K₂ = 40 * K - K^2 - 336)) := sorry

end NUMINAMATH_GPT_combined_PPF_two_females_combined_PPF_two_males_combined_PPF_male_female_l1841_184169


namespace NUMINAMATH_GPT_determine_function_f_l1841_184106

noncomputable def f (c x : ℝ) : ℝ := c ^ (1 / Real.log x)

theorem determine_function_f (f : ℝ → ℝ) (c : ℝ) (Hc : c > 1) :
  (∀ x, 1 < x → 1 < f x) →
  (∀ (x y : ℝ) (u v : ℝ), 1 < x → 1 < y → 0 < u → 0 < v →
    f (x ^ 4 * y ^ v) ≤ (f x) ^ (1 / (4 * u)) * (f y) ^ (1 / (4 * v))) →
  (∀ x : ℝ, 1 < x → f x = c ^ (1 / Real.log x)) :=
by
  sorry

end NUMINAMATH_GPT_determine_function_f_l1841_184106


namespace NUMINAMATH_GPT_cannot_be_simultaneous_squares_l1841_184164

theorem cannot_be_simultaneous_squares (x y : ℕ) :
  ¬ (∃ a b : ℤ, x^2 + y = a^2 ∧ y^2 + x = b^2) :=
by
  sorry

end NUMINAMATH_GPT_cannot_be_simultaneous_squares_l1841_184164


namespace NUMINAMATH_GPT_parabola_directrix_l1841_184166

theorem parabola_directrix
  (p : ℝ) (hp : p > 0)
  (O : ℝ × ℝ := (0,0))
  (Focus_F : ℝ × ℝ := (p / 2, 0))
  (Point_P : ℝ × ℝ)
  (Point_Q : ℝ × ℝ)
  (H1 : Point_P.1 = p / 2 ∧ Point_P.2^2 = 2 * p * Point_P.1)
  (H2 : Point_P.1 = Point_P.1) -- This comes out of the perpendicularity of PF to x-axis
  (H3 : Point_Q.2 = 0)
  (H4 : ∃ k_OP slope_OP, slope_OP = 2 ∧ ∃ k_PQ slope_PQ, slope_PQ = -1 / 2 ∧ k_OP * k_PQ = -1)
  (H5 : abs (Point_Q.1 - Focus_F.1) = 6) :
  x = -3 / 2 := 
sorry

end NUMINAMATH_GPT_parabola_directrix_l1841_184166


namespace NUMINAMATH_GPT_magic_square_y_value_l1841_184185

/-- In a magic square, where the sum of three entries in any row, column, or diagonal is the same value.
    Given the entries as shown below, prove that \(y = -38\).
    The entries are: 
    - \( y \) at position (1,1)
    - 23 at position (1,2)
    - 101 at position (1,3)
    - 4 at position (2,1)
    The remaining positions are denoted as \( a, b, c, d, e \).
-/
theorem magic_square_y_value :
    ∃ y a b c d e: ℤ,
        y + 4 + c = y + 23 + 101 ∧ -- Condition from first column and first row
        23 + a + d = 101 + b + 4 ∧ -- Condition from middle column and diagonal
        c + d + e = 101 + b + e ∧ -- Condition from bottom row and rightmost column
        y + 23 + 101 = 4 + a + b → -- Condition from top row
        y = -38 := 
by
    sorry

end NUMINAMATH_GPT_magic_square_y_value_l1841_184185


namespace NUMINAMATH_GPT_quadratic_solution_product_l1841_184101

theorem quadratic_solution_product :
  let r := 9 / 2
  let s := -11
  (r + 4) * (s + 4) = -119 / 2 :=
by
  -- Define the quadratic equation and its solutions
  let r := 9 / 2
  let s := -11

  -- Prove the statement
  sorry

end NUMINAMATH_GPT_quadratic_solution_product_l1841_184101


namespace NUMINAMATH_GPT_unique_n_value_l1841_184105

def is_n_table (n : ℕ) (A : Matrix (Fin n) (Fin n) ℕ) : Prop :=
  ∃ i j, 
    (∀ k : Fin n, A i j ≥ A i k) ∧   -- Max in its row
    (∀ k : Fin n, A i j ≤ A k j)     -- Min in its column

theorem unique_n_value 
  {n : ℕ} (h : 2 ≤ n) 
  (A : Matrix (Fin n) (Fin n) ℕ) 
  (hA : ∀ i j, A i j ∈ Finset.range (n^2)) -- Each number appears exactly once
  (hn : is_n_table n A) : 
  ∃! a, ∃ i j, A i j = a ∧ 
           (∀ k : Fin n, a ≥ A i k) ∧ 
           (∀ k : Fin n, a ≤ A k j) := 
sorry

end NUMINAMATH_GPT_unique_n_value_l1841_184105


namespace NUMINAMATH_GPT_minimum_travel_time_l1841_184103

structure TravelSetup where
  distance_ab : ℝ
  number_of_people : ℕ
  number_of_bicycles : ℕ
  speed_cyclist : ℝ
  speed_pedestrian : ℝ
  unattended_rule : Prop

theorem minimum_travel_time (setup : TravelSetup) : setup.distance_ab = 45 → 
                                                    setup.number_of_people = 3 → 
                                                    setup.number_of_bicycles = 2 → 
                                                    setup.speed_cyclist = 15 → 
                                                    setup.speed_pedestrian = 5 → 
                                                    setup.unattended_rule → 
                                                    ∃ t : ℝ, t = 3 := 
by
  intros
  sorry

end NUMINAMATH_GPT_minimum_travel_time_l1841_184103


namespace NUMINAMATH_GPT_orange_preference_percentage_l1841_184160

theorem orange_preference_percentage 
  (red blue green yellow purple orange : ℕ)
  (total : ℕ)
  (h_red : red = 75)
  (h_blue : blue = 80)
  (h_green : green = 50)
  (h_yellow : yellow = 45)
  (h_purple : purple = 60)
  (h_orange : orange = 55)
  (h_total : total = red + blue + green + yellow + purple + orange) :
  (orange * 100) / total = 15 :=
by
sorry

end NUMINAMATH_GPT_orange_preference_percentage_l1841_184160


namespace NUMINAMATH_GPT_martinez_family_combined_height_l1841_184177

def chiquita_height := 5
def mr_martinez_height := chiquita_height + 2
def mrs_martinez_height := chiquita_height - 1
def son_height := chiquita_height + 3
def combined_height := chiquita_height + mr_martinez_height + mrs_martinez_height + son_height

theorem martinez_family_combined_height : combined_height = 24 :=
by
  sorry

end NUMINAMATH_GPT_martinez_family_combined_height_l1841_184177


namespace NUMINAMATH_GPT_ratio_of_black_to_white_tiles_l1841_184140

theorem ratio_of_black_to_white_tiles
  (original_width : ℕ)
  (original_height : ℕ)
  (original_black_tiles : ℕ)
  (original_white_tiles : ℕ)
  (border_width : ℕ)
  (border_height : ℕ)
  (extended_width : ℕ)
  (extended_height : ℕ)
  (new_white_tiles : ℕ)
  (total_white_tiles : ℕ)
  (total_black_tiles : ℕ)
  (ratio_black_to_white : ℚ)
  (h1 : original_width = 5)
  (h2 : original_height = 6)
  (h3 : original_black_tiles = 12)
  (h4 : original_white_tiles = 18)
  (h5 : border_width = 1)
  (h6 : border_height = 1)
  (h7 : extended_width = original_width + 2 * border_width)
  (h8 : extended_height = original_height + 2 * border_height)
  (h9 : new_white_tiles = (extended_width * extended_height) - (original_width * original_height))
  (h10 : total_white_tiles = original_white_tiles + new_white_tiles)
  (h11 : total_black_tiles = original_black_tiles)
  (h12 : ratio_black_to_white = total_black_tiles / total_white_tiles) :
  ratio_black_to_white = 3 / 11 := 
sorry

end NUMINAMATH_GPT_ratio_of_black_to_white_tiles_l1841_184140


namespace NUMINAMATH_GPT_democrats_ratio_l1841_184178

theorem democrats_ratio (F M: ℕ) 
  (h_total_participants : F + M = 810)
  (h_female_democrats : 135 * 2 = F)
  (h_male_democrats : (1 / 4) * M = 135) : 
  (270 / 810 = 1 / 3) :=
by 
  sorry

end NUMINAMATH_GPT_democrats_ratio_l1841_184178


namespace NUMINAMATH_GPT_christina_rearrangements_l1841_184167

-- define the main conditions
def rearrangements (n : Nat) : Nat := Nat.factorial n

def half (n : Nat) : Nat := n / 2

def time_for_first_half (r : Nat) : Nat := r / 12

def time_for_second_half (r : Nat) : Nat := r / 18

def total_time_in_minutes (t1 t2 : Nat) : Nat := t1 + t2

def total_time_in_hours (t : Nat) : Nat := t / 60

-- statement proving that the total time will be 420 hours
theorem christina_rearrangements : 
  rearrangements 9 = 362880 →
  half (rearrangements 9) = 181440 →
  time_for_first_half 181440 = 15120 →
  time_for_second_half 181440 = 10080 →
  total_time_in_minutes 15120 10080 = 25200 →
  total_time_in_hours 25200 = 420 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_christina_rearrangements_l1841_184167


namespace NUMINAMATH_GPT_shares_of_c_l1841_184171

theorem shares_of_c (a b c : ℝ) (h1 : 3 * a = 4 * b) (h2 : 4 * b = 7 * c) (h3 : a + b + c = 427): 
  c = 84 :=
by {
  sorry
}

end NUMINAMATH_GPT_shares_of_c_l1841_184171


namespace NUMINAMATH_GPT_masha_problem_l1841_184149

noncomputable def sum_arithmetic_series (a l n : ℕ) : ℕ :=
  (n * (a + l)) / 2

theorem masha_problem : 
  let a_even := 372
  let l_even := 506
  let n_even := 67
  let a_odd := 373
  let l_odd := 505
  let n_odd := 68
  let S_even := sum_arithmetic_series a_even l_even n_even
  let S_odd := sum_arithmetic_series a_odd l_odd n_odd
  S_odd - S_even = 439 := 
by sorry

end NUMINAMATH_GPT_masha_problem_l1841_184149


namespace NUMINAMATH_GPT_inverse_proportion_points_l1841_184114

theorem inverse_proportion_points (x1 x2 y1 y2 : ℝ)
  (h1 : x1 < 0)
  (h2 : x2 > 0)
  (h3 : y1 = -8 / x1)
  (h4 : y2 = -8 / x2) :
  y2 < 0 ∧ 0 < y1 :=
by
  sorry

end NUMINAMATH_GPT_inverse_proportion_points_l1841_184114


namespace NUMINAMATH_GPT_trapezoid_inscribed_circles_radii_l1841_184117

open Real

variables (a b m n : ℝ)
noncomputable def r := (a * sqrt b) / (sqrt a + sqrt b)
noncomputable def R := (b * sqrt a) / (sqrt a + sqrt b)

theorem trapezoid_inscribed_circles_radii
  (h : a < b)
  (hM : m = sqrt (a * b))
  (hN : m = sqrt (a * b)) :
  (r a b = (a * sqrt b) / (sqrt a + sqrt b)) ∧
  (R a b = (b * sqrt a) / (sqrt a + sqrt b)) :=
by
  sorry

end NUMINAMATH_GPT_trapezoid_inscribed_circles_radii_l1841_184117


namespace NUMINAMATH_GPT_distance_PQ_parallel_x_max_distance_PQ_l1841_184157

open Real

def parabola (x : ℝ) : ℝ := x^2

/--
1. When PQ is parallel to the x-axis, find the distance from point O to PQ.
-/
theorem distance_PQ_parallel_x (m : ℝ) (h₁ : m ≠ 0) (h₂ : parabola m = 1) : 
  ∃ d : ℝ, d = 1 := by
  sorry

/--
2. Find the maximum value of the distance from point O to PQ.
-/
theorem max_distance_PQ (a b : ℝ) (h₁ : a * b = -1) (h₂ : ∀ x, ∃ y, y = a * x + b) :
  ∃ d : ℝ, d = 1 := by
  sorry

end NUMINAMATH_GPT_distance_PQ_parallel_x_max_distance_PQ_l1841_184157


namespace NUMINAMATH_GPT_solve_for_a_l1841_184173

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if h : x >= 0 then 4^x else 2^(a - x)

theorem solve_for_a (a : ℝ) (h : a ≠ 1) (h_f_eq : f a (1 - a) = f a (a - 1)) : a = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_a_l1841_184173


namespace NUMINAMATH_GPT_parallelogram_sides_l1841_184121

theorem parallelogram_sides (x y : ℝ) 
  (h1 : 3 * x + 6 = 15) 
  (h2 : 10 * y - 2 = 12) :
  x + y = 4.4 := 
sorry

end NUMINAMATH_GPT_parallelogram_sides_l1841_184121


namespace NUMINAMATH_GPT_w_identity_l1841_184111

theorem w_identity (w : ℝ) (h_pos : w > 0) (h_eq : w - 1 / w = 5) : (w + 1 / w) ^ 2 = 29 := by
  sorry

end NUMINAMATH_GPT_w_identity_l1841_184111


namespace NUMINAMATH_GPT_total_material_ordered_l1841_184184

theorem total_material_ordered :
  12.468 + 4.6278 + 7.9101 + 8.3103 + 5.6327 = 38.9499 :=
by
  sorry

end NUMINAMATH_GPT_total_material_ordered_l1841_184184


namespace NUMINAMATH_GPT_bob_has_17_pennies_l1841_184102

-- Definitions based on the problem conditions
variable (a b : ℕ)
def condition1 : Prop := b + 1 = 4 * (a - 1)
def condition2 : Prop := b - 2 = 2 * (a + 2)

-- The main statement to be proven
theorem bob_has_17_pennies (a b : ℕ) (h1 : condition1 a b) (h2 : condition2 a b) : b = 17 :=
by
  sorry

end NUMINAMATH_GPT_bob_has_17_pennies_l1841_184102


namespace NUMINAMATH_GPT_survey_population_l1841_184161

-- Definitions based on conditions
def number_of_packages := 10
def dozens_per_package := 10
def sets_per_dozen := 12

-- Derived from conditions
def total_sets := number_of_packages * dozens_per_package * sets_per_dozen

-- Populations for the proof
def population_quality : ℕ := total_sets
def population_satisfaction : ℕ := total_sets

-- Proof statement
theorem survey_population:
  (population_quality = 1200) ∧ (population_satisfaction = 1200) := by
  sorry

end NUMINAMATH_GPT_survey_population_l1841_184161


namespace NUMINAMATH_GPT_spends_at_arcade_each_weekend_l1841_184122

def vanessa_savings : ℕ := 20
def parents_weekly_allowance : ℕ := 30
def dress_cost : ℕ := 80
def weeks : ℕ := 3

theorem spends_at_arcade_each_weekend (arcade_weekend_expense : ℕ) :
  (vanessa_savings + weeks * parents_weekly_allowance - dress_cost = weeks * parents_weekly_allowance - arcade_weekend_expense * weeks) →
  arcade_weekend_expense = 30 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_spends_at_arcade_each_weekend_l1841_184122


namespace NUMINAMATH_GPT_area_of_trapezoid_l1841_184170

noncomputable def triangle_XYZ_is_isosceles : Prop := 
  ∃ (X Y Z : Type) (XY XZ : ℝ), XY = XZ

noncomputable def identical_smaller_triangles (area : ℝ) (num : ℕ) : Prop := 
  num = 9 ∧ area = 3

noncomputable def total_area_large_triangle (total_area : ℝ) : Prop := 
  total_area = 135

noncomputable def trapezoid_contains_smaller_triangles (contained : ℕ) : Prop :=
  contained = 4

theorem area_of_trapezoid (XYZ_area smaller_triangle_area : ℝ) 
    (num_smaller_triangles contained_smaller_triangles : ℕ) : 
    triangle_XYZ_is_isosceles → 
    identical_smaller_triangles smaller_triangle_area num_smaller_triangles →
    total_area_large_triangle XYZ_area →
    trapezoid_contains_smaller_triangles contained_smaller_triangles →
    (XYZ_area - contained_smaller_triangles * smaller_triangle_area) = 123 :=
by
  intros iso smaller_triangles total_area contained
  sorry

end NUMINAMATH_GPT_area_of_trapezoid_l1841_184170


namespace NUMINAMATH_GPT_sum_of_x_and_y_l1841_184146

-- Define integers x and y
variables (x y : ℤ)

-- Define conditions
def condition1 : Prop := x - y = 200
def condition2 : Prop := y = 250

-- Define the main statement
theorem sum_of_x_and_y (h1 : condition1 x y) (h2 : condition2 y) : x + y = 700 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_x_and_y_l1841_184146


namespace NUMINAMATH_GPT_buttons_pattern_total_buttons_sum_l1841_184192

-- Define the sequence of the number of buttons in each box
def buttons_in_box (n : ℕ) : ℕ := 3^(n-1)

-- Define the sum of buttons up to the n-th box
def total_buttons (n : ℕ) : ℕ := (3^n - 1) / 2

-- Theorem statements to prove
theorem buttons_pattern (n : ℕ) : buttons_in_box n = 3^(n-1) := by
  sorry

theorem total_buttons_sum (n : ℕ) : total_buttons n = (3^n - 1) / 2 := by
  sorry

end NUMINAMATH_GPT_buttons_pattern_total_buttons_sum_l1841_184192


namespace NUMINAMATH_GPT_smallest_angle_of_trapezoid_l1841_184127

theorem smallest_angle_of_trapezoid (a d : ℝ) :
  (a + (a + d) + (a + 2 * d) + (a + 3 * d) = 360) → 
  (a + 3 * d = 150) → 
  a = 15 :=
by
  sorry

end NUMINAMATH_GPT_smallest_angle_of_trapezoid_l1841_184127


namespace NUMINAMATH_GPT_inf_coprime_naturals_l1841_184181

theorem inf_coprime_naturals (a b : ℤ) (h : a ≠ b) : 
  ∃ᶠ n in Filter.atTop, Nat.gcd (Int.natAbs (a + n)) (Int.natAbs (b + n)) = 1 := 
sorry

end NUMINAMATH_GPT_inf_coprime_naturals_l1841_184181


namespace NUMINAMATH_GPT_simplify_fraction_l1841_184118

theorem simplify_fraction (c : ℚ) : (6 + 5 * c) / 9 + 3 = (33 + 5 * c) / 9 := 
sorry

end NUMINAMATH_GPT_simplify_fraction_l1841_184118


namespace NUMINAMATH_GPT_maximum_expression_value_l1841_184110

theorem maximum_expression_value (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x^2 - 2 * x * y + 3 * y^2 = 9) :
  x^2 + 2 * x * y + 3 * y^2 ≤ 33 :=
sorry

end NUMINAMATH_GPT_maximum_expression_value_l1841_184110


namespace NUMINAMATH_GPT_total_tiles_needed_l1841_184188

-- Define the dimensions of the dining room
def dining_room_length : ℕ := 15
def dining_room_width : ℕ := 20

-- Define the width of the border
def border_width : ℕ := 2

-- Areas for one-foot by one-foot border tiles
def one_foot_tile_border_tiles : ℕ :=
  2 * (dining_room_width + (dining_room_width - 2 * border_width)) + 
  2 * ((dining_room_length - 2) + (dining_room_length - 2 * border_width))

-- Dimensions of the inner area
def inner_length : ℕ := dining_room_length - 2 * border_width
def inner_width : ℕ := dining_room_width - 2 * border_width

-- Area for two-foot by two-foot tiles
def inner_area : ℕ := inner_length * inner_width
def two_foot_tile_inner_tiles : ℕ := inner_area / 4

-- Total number of tiles
def total_tiles : ℕ := one_foot_tile_border_tiles + two_foot_tile_inner_tiles

-- Prove that the total number of tiles needed is 168
theorem total_tiles_needed : total_tiles = 168 := sorry

end NUMINAMATH_GPT_total_tiles_needed_l1841_184188


namespace NUMINAMATH_GPT_find_a4b4_l1841_184130

theorem find_a4b4 
  (a1 a2 a3 a4 b1 b2 b3 b4 : ℝ)
  (h1 : a1 * b1 + a2 * b3 = 1)
  (h2 : a1 * b2 + a2 * b4 = 0)
  (h3 : a3 * b1 + a4 * b3 = 0)
  (h4 : a3 * b2 + a4 * b4 = 1)
  (h5 : a2 * b3 = 7) :
  a4 * b4 = -6 :=
sorry

end NUMINAMATH_GPT_find_a4b4_l1841_184130


namespace NUMINAMATH_GPT_multiply_expression_l1841_184168

variable (y : ℝ)

theorem multiply_expression : 
  (16 * y^3) * (12 * y^5) * (1 / (4 * y)^3) = 3 * y^5 := by
  sorry

end NUMINAMATH_GPT_multiply_expression_l1841_184168


namespace NUMINAMATH_GPT_election_votes_total_l1841_184131

-- Definitions representing the conditions
def CandidateAVotes (V : ℕ) := 45 * V / 100
def CandidateBVotes (V : ℕ) := 35 * V / 100
def CandidateCVotes (V : ℕ) := 20 * V / 100

-- Main theorem statement
theorem election_votes_total (V : ℕ) (h1: CandidateAVotes V = 45 * V / 100) (h2: CandidateBVotes V = 35 * V / 100) (h3: CandidateCVotes V = 20 * V / 100)
  (h4: CandidateAVotes V - CandidateBVotes V = 1800) : V = 18000 :=
  sorry

end NUMINAMATH_GPT_election_votes_total_l1841_184131


namespace NUMINAMATH_GPT_find_n_l1841_184115

theorem find_n (n : ℕ) (h₁ : 3 * n + 4 = 13) : n = 3 :=
by 
  sorry

end NUMINAMATH_GPT_find_n_l1841_184115


namespace NUMINAMATH_GPT_no_valid_n_l1841_184133

theorem no_valid_n : ¬ ∃ (n : ℕ), (n > 0) ∧ (100 ≤ n / 4) ∧ (n / 4 ≤ 999) ∧ (100 ≤ 4 * n) ∧ (4 * n ≤ 999) :=
by {
  sorry
}

end NUMINAMATH_GPT_no_valid_n_l1841_184133


namespace NUMINAMATH_GPT_inclination_of_line_l1841_184186

theorem inclination_of_line (α : ℝ) (h1 : ∃ l : ℝ, ∀ x y : ℝ, x + y + 1 = 0 → y = -x - 1) : α = 135 :=
by
  sorry

end NUMINAMATH_GPT_inclination_of_line_l1841_184186


namespace NUMINAMATH_GPT_find_c_for_given_radius_l1841_184128

theorem find_c_for_given_radius (c : ℝ) : (∃ x y : ℝ, (x^2 - 2 * x + y^2 + 6 * y + c = 0) ∧ ((x - 1)^2 + (y + 3)^2 = 25)) → c = -15 :=
by
  sorry

end NUMINAMATH_GPT_find_c_for_given_radius_l1841_184128


namespace NUMINAMATH_GPT_minimum_c_value_l1841_184108

theorem minimum_c_value
  (a b c k : ℕ) (h1 : b = a + k) (h2 : c = b + k) (h3 : a < b) (h4 : b < c) (h5 : k > 0) :
  c = 6005 :=
sorry

end NUMINAMATH_GPT_minimum_c_value_l1841_184108


namespace NUMINAMATH_GPT_least_stamps_l1841_184135

theorem least_stamps (s t : ℕ) (h : 5 * s + 7 * t = 48) : s + t = 8 :=
by sorry

end NUMINAMATH_GPT_least_stamps_l1841_184135


namespace NUMINAMATH_GPT_rectangle_perimeter_is_36_l1841_184139

theorem rectangle_perimeter_is_36 (a b : ℕ) (h : a ≠ b) (h1 : a * b = 2 * (2 * a + 2 * b) - 8) : 2 * (a + b) = 36 :=
  sorry

end NUMINAMATH_GPT_rectangle_perimeter_is_36_l1841_184139


namespace NUMINAMATH_GPT_triangle_area_is_54_l1841_184199

-- Define the sides of the triangle
def side1 : ℕ := 9
def side2 : ℕ := 12
def side3 : ℕ := 15

-- Verify that it is a right triangle using the Pythagorean theorem
def isRightTriangle (a b c : ℕ) : Prop := a * a + b * b = c * c

-- Define the area calculation for a right triangle
def areaRightTriangle (a b : ℕ) : ℕ := Nat.div (a * b) 2

-- State the theorem (Problem) to prove
theorem triangle_area_is_54 :
  isRightTriangle side1 side2 side3 ∧ areaRightTriangle side1 side2 = 54 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_is_54_l1841_184199


namespace NUMINAMATH_GPT_area_of_triangle_ABC_l1841_184172

theorem area_of_triangle_ABC (A B C : ℝ) (a b c : ℝ) 
  (h1 : b = 2) (h2 : c = 3) (h3 : C = 2 * B): 
  ∃ S : ℝ, S = 1/2 * b * c * (Real.sin A) ∧ S = 15 * (Real.sqrt 7) / 16 :=
by
  sorry

end NUMINAMATH_GPT_area_of_triangle_ABC_l1841_184172


namespace NUMINAMATH_GPT_perpendicular_and_intersection_l1841_184145

variables (x y : ℚ)

def line1 := 4 * y - 3 * x = 15
def line4 := 3 * y + 4 * x = 15

theorem perpendicular_and_intersection :
  (4 * y - 3 * x = 15) ∧ (3 * y + 4 * x = 15) →
  let m1 := (3 : ℚ) / 4
  let m4 := -(4 : ℚ) / 3
  m1 * m4 = -1 ∧
  ∃ x y : ℚ, 4*y - 3*x = 15 ∧ 3*y + 4*x = 15 ∧ x = 15/32 ∧ y = 35/8 :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_and_intersection_l1841_184145


namespace NUMINAMATH_GPT_factor_correct_l1841_184194

noncomputable def p (b : ℝ) : ℝ := 221 * b^2 + 17 * b
def factored_form (b : ℝ) : ℝ := 17 * b * (13 * b + 1)

theorem factor_correct (b : ℝ) : p b = factored_form b := by
  sorry

end NUMINAMATH_GPT_factor_correct_l1841_184194


namespace NUMINAMATH_GPT_complex_square_simplification_l1841_184175

theorem complex_square_simplification (i : ℂ) (h : i^2 = -1) : (4 - 3 * i)^2 = 7 - 24 * i :=
by {
  sorry
}

end NUMINAMATH_GPT_complex_square_simplification_l1841_184175


namespace NUMINAMATH_GPT_jelly_beans_in_jar_y_l1841_184150

-- Definitions of the conditions
def total_beans : ℕ := 1200
def number_beans_in_jar_y (y : ℕ) := y
def number_beans_in_jar_x (y : ℕ) := 3 * y - 400

-- The main theorem to be proven
theorem jelly_beans_in_jar_y (y : ℕ) :
  number_beans_in_jar_x y + number_beans_in_jar_y y = total_beans → 
  y = 400 := 
by
  sorry

end NUMINAMATH_GPT_jelly_beans_in_jar_y_l1841_184150


namespace NUMINAMATH_GPT_basic_printer_total_price_l1841_184156

theorem basic_printer_total_price (C P : ℝ) (hC : C = 1500) (hP : P = (1/3) * (C + 500 + P)) : C + P = 2500 := 
by
  sorry

end NUMINAMATH_GPT_basic_printer_total_price_l1841_184156


namespace NUMINAMATH_GPT_cyclist_speed_north_l1841_184190

theorem cyclist_speed_north (v : ℝ) :
  (∀ d t : ℝ, d = 50 ∧ t = 1 ∧ 40 * t + v * t = d) → v = 10 :=
by
  sorry

end NUMINAMATH_GPT_cyclist_speed_north_l1841_184190


namespace NUMINAMATH_GPT_necessary_not_sufficient_l1841_184159

-- Define the function y = x^2 - 2ax + 1
def quadratic_function (a x : ℝ) : ℝ := x^2 - 2 * a * x + 1

-- Define strict monotonicity on the interval [1, +∞)
def strictly_increasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ I → y ∈ I → x < y → f x < f y

-- Define the condition for the function to be strictly increasing on [1, +∞)
def condition_strict_increasing (a : ℝ) : Prop :=
  strictly_increasing_on (quadratic_function a) (Set.Ici 1)

-- The condition to prove
theorem necessary_not_sufficient (a : ℝ) :
  condition_strict_increasing a → (a ≤ 0) := sorry

end NUMINAMATH_GPT_necessary_not_sufficient_l1841_184159


namespace NUMINAMATH_GPT_second_expression_l1841_184141

variable (a b : ℕ)

theorem second_expression (h : 89 = ((2 * a + 16) + b) / 2) (ha : a = 34) : b = 94 :=
by
  sorry

end NUMINAMATH_GPT_second_expression_l1841_184141


namespace NUMINAMATH_GPT_exists_term_not_of_form_l1841_184152

theorem exists_term_not_of_form (a d : ℕ) (h_seq : ∀ i j : ℕ, (i < 40 ∧ j < 40 ∧ i ≠ j) → a + i * d ≠ a + j * d)
  (pos_a : a > 0) (pos_d : d > 0) 
  : ∃ h : ℕ, h < 40 ∧ ¬ ∃ k l : ℕ, a + h * d = 2^k + 3^l :=
by {
  sorry
}

end NUMINAMATH_GPT_exists_term_not_of_form_l1841_184152


namespace NUMINAMATH_GPT_fraction_equals_half_l1841_184151

def numerator : ℤ := 1 - 2 + 4 - 8 + 16 - 32 + 64
def denominator : ℤ := 2 - 4 + 8 - 16 + 32 - 64 + 128

theorem fraction_equals_half : (numerator : ℚ) / (denominator : ℚ) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_fraction_equals_half_l1841_184151


namespace NUMINAMATH_GPT_find_number_of_children_l1841_184100

-- Definitions based on conditions
def decorative_spoons : Nat := 2
def new_set_large_spoons : Nat := 10
def new_set_tea_spoons : Nat := 15
def total_spoons : Nat := 39
def spoons_per_child : Nat := 3
def new_set_spoons := new_set_large_spoons + new_set_tea_spoons

-- The main statement to prove the number of children
theorem find_number_of_children (C : Nat) :
  3 * C + decorative_spoons + new_set_spoons = total_spoons → C = 4 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_find_number_of_children_l1841_184100


namespace NUMINAMATH_GPT_power_inequality_l1841_184119

theorem power_inequality (n : ℕ) (x : ℝ) (hn : n ≥ 2) (hx : abs x < 1) : 
  2^n > (1 - x)^n + (1 + x)^n := 
sorry

end NUMINAMATH_GPT_power_inequality_l1841_184119


namespace NUMINAMATH_GPT_time_to_sweep_one_room_l1841_184163

theorem time_to_sweep_one_room (x : ℕ) :
  (10 * x) = (2 * 9 + 6 * 2) → x = 3 := by
  sorry

end NUMINAMATH_GPT_time_to_sweep_one_room_l1841_184163


namespace NUMINAMATH_GPT_num_intersections_circle_line_eq_two_l1841_184182

theorem num_intersections_circle_line_eq_two :
  ∃ (points : Finset (ℝ × ℝ)), {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = 25 ∧ p.1 = 3} = points ∧ points.card = 2 :=
by
  sorry

end NUMINAMATH_GPT_num_intersections_circle_line_eq_two_l1841_184182


namespace NUMINAMATH_GPT_linear_eq_m_value_l1841_184147

theorem linear_eq_m_value (x m : ℝ) (h : 2 * x + m = 5) (hx : x = 1) : m = 3 :=
by
  -- Here we would carry out the proof steps
  sorry

end NUMINAMATH_GPT_linear_eq_m_value_l1841_184147


namespace NUMINAMATH_GPT_sum_of_coefficients_l1841_184138

-- Define the polynomial P(x)
def P (x : ℝ) : ℝ := 5 * (2 * x^8 - 3 * x^5 + 9 * x^3 - 6) + 4 * (7 * x^6 - 2 * x^3 + 8)

-- Statement to prove that the sum of the coefficients of P(x) is 62
theorem sum_of_coefficients : P 1 = 62 := sorry

end NUMINAMATH_GPT_sum_of_coefficients_l1841_184138


namespace NUMINAMATH_GPT_ellipse_eccentricity_half_l1841_184116

-- Definitions and assumptions
variable (a b c e : ℝ)
variable (h₁ : a = 2 * c)
variable (h₂ : b = sqrt 3 * c)
variable (eccentricity_def : e = c / a)

-- Theorem statement
theorem ellipse_eccentricity_half : e = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_ellipse_eccentricity_half_l1841_184116


namespace NUMINAMATH_GPT_cosine_identity_example_l1841_184144

theorem cosine_identity_example {α : ℝ} (h : Real.sin (π / 3 - α) = 1 / 3) : Real.cos (π / 3 + 2 * α) = -7 / 9 :=
by sorry

end NUMINAMATH_GPT_cosine_identity_example_l1841_184144


namespace NUMINAMATH_GPT_abc_value_l1841_184112

variables (a b c d e f : ℝ)
variables (h1 : b * c * d = 65)
variables (h2 : c * d * e = 750)
variables (h3 : d * e * f = 250)
variables (h4 : (a * f) / (c * d) = 0.6666666666666666)

theorem abc_value : a * b * c = 130 :=
by { sorry }

end NUMINAMATH_GPT_abc_value_l1841_184112


namespace NUMINAMATH_GPT_max_value_of_z_l1841_184129

theorem max_value_of_z (x y : ℝ) (hx : 0 ≤ x) (hx1 : x ≤ 1) (hy : 0 ≤ y) (hy1 : y ≤ 1) :
  x^2 + y^2 ≤ 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_max_value_of_z_l1841_184129


namespace NUMINAMATH_GPT_solve_for_x_l1841_184187

variable {a b c x : ℝ}
variable (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
variable (h_eq : (x - a - b) / c + (x - b - c) / a + (x - c - a) / b = 3)

theorem solve_for_x (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_eq : (x - a - b) / c + (x - b - c) / a + (x - c - a) / b = 3) : 
  x = a + b + c :=
sorry

end NUMINAMATH_GPT_solve_for_x_l1841_184187


namespace NUMINAMATH_GPT_train_crosses_second_platform_in_20_sec_l1841_184162

theorem train_crosses_second_platform_in_20_sec
  (length_train : ℝ)
  (length_first_platform : ℝ)
  (time_first_platform : ℝ)
  (length_second_platform : ℝ)
  (time_second_platform : ℝ):

  length_train = 100 ∧
  length_first_platform = 350 ∧
  time_first_platform = 15 ∧
  length_second_platform = 500 →
  time_second_platform = 20 := by
  sorry

end NUMINAMATH_GPT_train_crosses_second_platform_in_20_sec_l1841_184162


namespace NUMINAMATH_GPT_twins_age_l1841_184165

theorem twins_age (x : ℕ) (h : (x + 1) * (x + 1) = x * x + 5) : x = 2 :=
by
  sorry

end NUMINAMATH_GPT_twins_age_l1841_184165


namespace NUMINAMATH_GPT_find_area_of_plot_l1841_184136

def area_of_plot (B : ℝ) (L : ℝ) (A : ℝ) : Prop :=
  L = 0.75 * B ∧ B = 21.908902300206645 ∧ A = L * B

theorem find_area_of_plot (B L A : ℝ) (h : area_of_plot B L A) : A = 360 := by
  sorry

end NUMINAMATH_GPT_find_area_of_plot_l1841_184136


namespace NUMINAMATH_GPT_difference_between_new_and_original_l1841_184158

variables (x y : ℤ) -- Declaring variables x and y as integers

-- The original number is represented as 10*x + y, and the new number after swapping is 10*y + x.
-- We need to prove that the difference between the new number and the original number is -9*x + 9*y.
theorem difference_between_new_and_original (x y : ℤ) :
  (10 * y + x) - (10 * x + y) = -9 * x + 9 * y :=
by
  sorry -- Proof placeholder

end NUMINAMATH_GPT_difference_between_new_and_original_l1841_184158


namespace NUMINAMATH_GPT_major_axis_length_of_ellipse_l1841_184120

theorem major_axis_length_of_ellipse :
  ∀ {y x : ℝ},
  (y^2 / 25 + x^2 / 15 = 1) → 
  2 * Real.sqrt 25 = 10 :=
by
  intro y x h
  sorry

end NUMINAMATH_GPT_major_axis_length_of_ellipse_l1841_184120


namespace NUMINAMATH_GPT_sum_sequence_formula_l1841_184113

-- Define the sequence terms as a function.
def seq_term (x a : ℕ) (n : ℕ) : ℕ :=
x ^ (n + 1) + (n + 1) * a

-- Define the sum of the first nine terms of the sequence.
def sum_first_nine_terms (x a : ℕ) : ℕ :=
(x * (x ^ 9 - 1)) / (x - 1) + 45 * a

-- State the theorem to prove that the sum S is as expected.
theorem sum_sequence_formula (x a : ℕ) (h : x ≠ 1) : 
  sum_first_nine_terms x a = (x ^ 10 - x) / (x - 1) + 45 * a := by
  sorry

end NUMINAMATH_GPT_sum_sequence_formula_l1841_184113


namespace NUMINAMATH_GPT_bie_l1841_184153

noncomputable def surface_area_of_sphere (PA AB AC : ℝ) (hPA_AB : PA = AB) (hPA : PA = 2) (hAC : AC = 4) (r : ℝ) : ℝ :=
  let PC := Real.sqrt (PA ^ 2 + AC ^ 2)
  let radius := PC / 2
  4 * Real.pi * radius ^ 2

theorem bie'zhi_tetrahedron_surface_area
  (PA AB AC : ℝ)
  (hPA_AB : PA = AB)
  (hPA : PA = 2)
  (hAC : AC = 4)
  (PC : ℝ := Real.sqrt (PA ^ 2 + AC ^ 2))
  (r : ℝ := PC / 2)
  (surface_area : ℝ := 4 * Real.pi * r ^ 2)
  :
  surface_area = 20 * Real.pi := 
sorry

end NUMINAMATH_GPT_bie_l1841_184153


namespace NUMINAMATH_GPT_bipin_chandan_age_ratio_l1841_184124

-- Define the condition statements
def AlokCurrentAge : Nat := 5
def BipinCurrentAge : Nat := 6 * AlokCurrentAge
def ChandanCurrentAge : Nat := 7 + 3

-- Define the ages after 10 years
def BipinAgeAfter10Years : Nat := BipinCurrentAge + 10
def ChandanAgeAfter10Years : Nat := ChandanCurrentAge + 10

-- Define the ratio and the statement to prove
def AgeRatio := BipinAgeAfter10Years / ChandanAgeAfter10Years

-- The theorem to prove the ratio is 2
theorem bipin_chandan_age_ratio : AgeRatio = 2 := by
  sorry

end NUMINAMATH_GPT_bipin_chandan_age_ratio_l1841_184124


namespace NUMINAMATH_GPT_eleventh_term_of_sequence_l1841_184198

def inversely_proportional_sequence (a : ℕ → ℝ) (c : ℝ) : Prop :=
  ∀ n : ℕ, a n * a (n + 1) = c

theorem eleventh_term_of_sequence :
  ∃ a : ℕ → ℝ,
    (a 1 = 3) ∧
    (a 2 = 6) ∧
    inversely_proportional_sequence a 18 ∧
    a 11 = 3 :=
by
  sorry

end NUMINAMATH_GPT_eleventh_term_of_sequence_l1841_184198


namespace NUMINAMATH_GPT_smallest_positive_integer_x_l1841_184195

def smallest_x (x : ℕ) : Prop :=
  x > 0 ∧ (450 * x) % 625 = 0

theorem smallest_positive_integer_x :
  ∃ x : ℕ, smallest_x x ∧ ∀ y : ℕ, smallest_x y → x ≤ y ∧ x = 25 :=
by {
  sorry
}

end NUMINAMATH_GPT_smallest_positive_integer_x_l1841_184195


namespace NUMINAMATH_GPT_remaining_area_is_correct_l1841_184176

-- Define the given conditions:
def original_length : ℕ := 25
def original_width : ℕ := 35
def square_side : ℕ := 7

-- Define a function to calculate the area of the original cardboard:
def area_original : ℕ := original_length * original_width

-- Define a function to calculate the area of one square corner:
def area_corner : ℕ := square_side * square_side

-- Define a function to calculate the total area removed:
def total_area_removed : ℕ := 4 * area_corner

-- Define a function to calculate the remaining area:
def area_remaining : ℕ := area_original - total_area_removed

-- The theorem we want to prove:
theorem remaining_area_is_correct : area_remaining = 679 := by
  -- Here, we would provide the proof if required, but we use sorry for now.
  sorry

end NUMINAMATH_GPT_remaining_area_is_correct_l1841_184176


namespace NUMINAMATH_GPT_least_number_of_teams_l1841_184104

theorem least_number_of_teams
  (total_athletes : ℕ)
  (max_team_size : ℕ)
  (h_total : total_athletes = 30)
  (h_max : max_team_size = 12) :
  ∃ (number_of_teams : ℕ) (team_size : ℕ),
    number_of_teams * team_size = total_athletes ∧
    team_size ≤ max_team_size ∧
    number_of_teams = 3 :=
by
  sorry

end NUMINAMATH_GPT_least_number_of_teams_l1841_184104


namespace NUMINAMATH_GPT_range_of_a_l1841_184191

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0) ∧ (∃ x : ℝ, x^2 + 2 * a * x + a + 2 = 0) → a ≤ -1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1841_184191


namespace NUMINAMATH_GPT_minimum_positive_period_of_f_is_pi_l1841_184197

noncomputable def f (x : ℝ) : ℝ :=
  (Real.cos (x + Real.pi / 4))^2 - (Real.sin (x + Real.pi / 4))^2

theorem minimum_positive_period_of_f_is_pi :
  ∃ T > 0, (∀ x, f (x + T) = f x) ∧ (∀ T', T' > 0 ∧ (∀ x, f (x + T') = f x) → T' ≥ T) ∧ T = Real.pi :=
sorry

end NUMINAMATH_GPT_minimum_positive_period_of_f_is_pi_l1841_184197


namespace NUMINAMATH_GPT_mass_percentage_O_in_CaO_l1841_184155

theorem mass_percentage_O_in_CaO :
  (16.00 / (40.08 + 16.00)) * 100 = 28.53 :=
by
  sorry

end NUMINAMATH_GPT_mass_percentage_O_in_CaO_l1841_184155


namespace NUMINAMATH_GPT_find_value_l1841_184107

variables (x1 x2 y1 y2 : ℝ)

def condition1 := x1 ^ 2 + 5 * x2 ^ 2 = 10
def condition2 := x2 * y1 - x1 * y2 = 5
def condition3 := x1 * y1 + 5 * x2 * y2 = Real.sqrt 105

theorem find_value (h1 : condition1 x1 x2) (h2 : condition2 x1 x2 y1 y2) (h3 : condition3 x1 x2 y1 y2) :
  y1 ^ 2 + 5 * y2 ^ 2 = 23 :=
sorry

end NUMINAMATH_GPT_find_value_l1841_184107


namespace NUMINAMATH_GPT_goose_eggs_count_l1841_184193

theorem goose_eggs_count (E : ℕ) 
  (hatch_ratio : ℝ := 1/4)
  (survival_first_month_ratio : ℝ := 4/5)
  (survival_first_year_ratio : ℝ := 3/5)
  (survived_first_year : ℕ := 120) :
  ((survival_first_year_ratio * (survival_first_month_ratio * hatch_ratio * E)) = survived_first_year) → E = 1000 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_goose_eggs_count_l1841_184193


namespace NUMINAMATH_GPT_tangential_quadrilateral_perpendicular_diagonals_l1841_184179

-- Define what it means for a quadrilateral to be tangential
def is_tangential_quadrilateral (a b c d : ℝ) : Prop :=
  a + c = b + d

-- Define what it means for a quadrilateral to be a kite
def is_kite (a b c d : ℝ) : Prop :=
  a = b ∧ c = d

-- Define what it means for the diagonals of a quadrilateral to be perpendicular
def diagonals_perpendicular (a b c d : ℝ) : Prop :=
  sorry -- Actual geometric definition needs to be elaborated

-- Main statement to prove
theorem tangential_quadrilateral_perpendicular_diagonals (a b c d : ℝ) :
  is_tangential_quadrilateral a b c d → 
  (diagonals_perpendicular a b c d ↔ is_kite a b c d) := 
sorry

end NUMINAMATH_GPT_tangential_quadrilateral_perpendicular_diagonals_l1841_184179


namespace NUMINAMATH_GPT_katy_brownies_l1841_184196

-- Define the conditions
def ate_monday : ℕ := 5
def ate_tuesday : ℕ := 2 * ate_monday

-- Define the question
def total_brownies : ℕ := ate_monday + ate_tuesday

-- State the proof problem
theorem katy_brownies : total_brownies = 15 := by
  sorry

end NUMINAMATH_GPT_katy_brownies_l1841_184196


namespace NUMINAMATH_GPT_angle_between_plane_and_base_l1841_184180

variable (α k : ℝ)
variable (hα : ∀ S A B C : ℝ, S ≠ A ∧ S ≠ B ∧ S ≠ C ∧ A ≠ B ∧ A ≠ C ∧ B ≠ C)
variable (h_ratio : ∀ A D S : ℝ, AD / DS = k)

theorem angle_between_plane_and_base (α k : ℝ) 
  (hα : ∀ S A B C : ℝ, S ≠ A ∧ S ≠ B ∧ S ≠ C ∧ A ≠ B ∧ A ≠ C ∧ B ≠ C)
  (h_ratio : ∀ A D S : ℝ, AD / DS = k) 
  : ∃ γ : ℝ, γ = Real.arctan (k / (k + 3) * Real.tan α) :=
by
  sorry

end NUMINAMATH_GPT_angle_between_plane_and_base_l1841_184180


namespace NUMINAMATH_GPT_find_v_3_l1841_184183

def u (x : ℤ) : ℤ := 4 * x - 9

def v (z : ℤ) : ℤ := z^2 + 4 * z - 1

theorem find_v_3 : v 3 = 20 := by
  sorry

end NUMINAMATH_GPT_find_v_3_l1841_184183


namespace NUMINAMATH_GPT_ratio_of_milk_and_water_l1841_184123

theorem ratio_of_milk_and_water (x y : ℝ) (hx : 9 * x = 9 * y) : 
  let total_milk := (7 * x + 8 * y)
  let total_water := (2 * x + y)
  (total_milk / total_water) = 5 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_milk_and_water_l1841_184123


namespace NUMINAMATH_GPT_problem_evaluation_l1841_184189

theorem problem_evaluation : (726 * 726) - (725 * 727) = 1 := 
by 
  sorry

end NUMINAMATH_GPT_problem_evaluation_l1841_184189


namespace NUMINAMATH_GPT_base8_operations_l1841_184132

def add_base8 (a b : ℕ) : ℕ :=
  let sum := (a + b) % 8
  sum

def subtract_base8 (a b : ℕ) : ℕ :=
  let diff := (a + 8 - b) % 8
  diff

def step1 := add_base8 672 156
def step2 := subtract_base8 step1 213

theorem base8_operations :
  step2 = 0645 :=
by
  sorry

end NUMINAMATH_GPT_base8_operations_l1841_184132


namespace NUMINAMATH_GPT_ratio_m_q_l1841_184148

theorem ratio_m_q (m n p q : ℚ) (h1 : m / n = 25) (h2 : p / n = 5) (h3 : p / q = 1 / 15) : 
  m / q = 1 / 3 :=
by 
  sorry

end NUMINAMATH_GPT_ratio_m_q_l1841_184148


namespace NUMINAMATH_GPT_height_of_cuboid_l1841_184125

theorem height_of_cuboid (A l w : ℝ) (h : ℝ) (hA : A = 442) (hl : l = 7) (hw : w = 8) : h = 11 :=
by
  sorry

end NUMINAMATH_GPT_height_of_cuboid_l1841_184125


namespace NUMINAMATH_GPT_apple_capacity_l1841_184142

/-- Question: What is the largest possible number of apples that can be held by the 6 boxes and 4 extra trays?
 Conditions:
 - Paul has 6 boxes.
 - Each box contains 12 trays.
 - Paul has 4 extra trays.
 - Each tray can hold 8 apples.
 Answer: 608 apples
-/
theorem apple_capacity :
  let boxes := 6
  let trays_per_box := 12
  let extra_trays := 4
  let apples_per_tray := 8
  let total_trays := (boxes * trays_per_box) + extra_trays
  let total_apples_capacity := total_trays * apples_per_tray
  total_apples_capacity = 608 := 
by
  sorry

end NUMINAMATH_GPT_apple_capacity_l1841_184142


namespace NUMINAMATH_GPT_q_value_at_2_l1841_184137

def q (x d e : ℤ) : ℤ := x^2 + d*x + e

theorem q_value_at_2 (d e : ℤ) 
  (h1 : ∃ p : ℤ → ℤ, ∀ x, x^4 + 8*x^2 + 49 = (q x d e) * (p x))
  (h2 : ∃ r : ℤ → ℤ, ∀ x, 2*x^4 + 5*x^2 + 36*x + 7 = (q x d e) * (r x)) :
  q 2 d e = 5 := 
sorry

end NUMINAMATH_GPT_q_value_at_2_l1841_184137


namespace NUMINAMATH_GPT_flower_growth_l1841_184134

theorem flower_growth (total_seeds : ℕ) (seeds_per_bed : ℕ) (max_grow_per_bed : ℕ) (h1 : total_seeds = 55) (h2 : seeds_per_bed = 15) (h3 : max_grow_per_bed = 60) : total_seeds ≤ 55 :=
by
  -- use the given conditions
  have h4 : total_seeds = 55 := h1
  sorry -- Proof goes here, omitted as instructed

end NUMINAMATH_GPT_flower_growth_l1841_184134
