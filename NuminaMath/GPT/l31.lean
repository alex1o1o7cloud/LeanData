import Mathlib

namespace NUMINAMATH_GPT_totalBirdsOnFence_l31_3129

/-
Statement: Given initial birds and additional birds joining, the total number
           of birds sitting on the fence is 10.
Conditions:
1. Initially, there are 4 birds.
2. 6 more birds join them.
3. There are 46 storks on the fence, but they do not affect the number of birds.
-/

def initialBirds : Nat := 4
def additionalBirds : Nat := 6

theorem totalBirdsOnFence : initialBirds + additionalBirds = 10 := by
  sorry

end NUMINAMATH_GPT_totalBirdsOnFence_l31_3129


namespace NUMINAMATH_GPT_hare_height_l31_3146

theorem hare_height (camel_height_ft : ℕ) (hare_height_in_inches : ℕ) :
  (camel_height_ft = 28) ∧ (hare_height_in_inches * 24 = camel_height_ft * 12) → hare_height_in_inches = 14 :=
by
  sorry

end NUMINAMATH_GPT_hare_height_l31_3146


namespace NUMINAMATH_GPT_correct_reaction_for_phosphoric_acid_l31_3132

-- Define the reactions
def reaction_A := "H₂ + 2OH⁻ - 2e⁻ = 2H₂O"
def reaction_B := "H₂ - 2e⁻ = 2H⁺"
def reaction_C := "O₂ + 4H⁺ + 4e⁻ = 2H₂O"
def reaction_D := "O₂ + 2H₂O + 4e⁻ = 4OH⁻"

-- Define the condition that the electrolyte used is phosphoric acid
def electrolyte := "phosphoric acid"

-- Define the correct reaction
def correct_negative_electrode_reaction := reaction_B

-- Theorem to state that given the conditions above, the correct reaction is B
theorem correct_reaction_for_phosphoric_acid :
  (∃ r, r = reaction_B ∧ electrolyte = "phosphoric acid") :=
by
  sorry

end NUMINAMATH_GPT_correct_reaction_for_phosphoric_acid_l31_3132


namespace NUMINAMATH_GPT_average_score_l31_3165

theorem average_score (m n : ℝ) (hm : m ≥ 0) (hn : n ≥ 0) :
  (20 * m + 23 * n) / (20 + 23) = 20 / 43 * m + 23 / 43 * n := sorry

end NUMINAMATH_GPT_average_score_l31_3165


namespace NUMINAMATH_GPT_book_pages_total_l31_3181

theorem book_pages_total
  (pages_read_first_day : ℚ) (total_pages : ℚ) (pages_read_second_day : ℚ)
  (rem_read_ratio : ℚ) (read_ratio_mult : ℚ)
  (book_ratio: ℚ) (read_pages_ratio: ℚ)
  (read_second_day_ratio: ℚ):
  pages_read_first_day = 1 / 6 →
  pages_read_second_day = 42 →
  rem_read_ratio = 3 →
  read_ratio_mult = (2 / 6) →
  book_ratio = 3 / 5 →
  read_pages_ratio = 2 / 5 →
  read_second_day_ratio = (2 / 5 - 1 / 6) →
  total_pages = pages_read_second_day / read_second_day_ratio  →
  total_pages = 126 :=
by sorry

end NUMINAMATH_GPT_book_pages_total_l31_3181


namespace NUMINAMATH_GPT_water_added_l31_3153

theorem water_added (initial_fullness : ℝ) (fullness_after : ℝ) (capacity : ℝ) 
  (h_initial : initial_fullness = 0.30) (h_after : fullness_after = 3/4) (h_capacity : capacity = 100) : 
  fullness_after * capacity - initial_fullness * capacity = 45 := 
by 
  sorry

end NUMINAMATH_GPT_water_added_l31_3153


namespace NUMINAMATH_GPT_Alexei_finished_ahead_of_Sergei_by_1_9_km_l31_3110

noncomputable def race_distance : ℝ := 10
noncomputable def v_A : ℝ := 1  -- speed of Alexei
noncomputable def v_V : ℝ := 0.9 * v_A  -- speed of Vitaly
noncomputable def v_S : ℝ := 0.81 * v_A  -- speed of Sergei

noncomputable def distance_Alexei_finished_Ahead_of_Sergei : ℝ :=
race_distance - (0.81 * race_distance)

theorem Alexei_finished_ahead_of_Sergei_by_1_9_km :
  distance_Alexei_finished_Ahead_of_Sergei = 1.9 :=
by
  simp [race_distance, v_A, v_V, v_S, distance_Alexei_finished_Ahead_of_Sergei]
  sorry

end NUMINAMATH_GPT_Alexei_finished_ahead_of_Sergei_by_1_9_km_l31_3110


namespace NUMINAMATH_GPT_female_athletes_in_sample_l31_3175

theorem female_athletes_in_sample (total_athletes : ℕ) (male_athletes : ℕ) (sample_size : ℕ)
  (total_athletes_eq : total_athletes = 98)
  (male_athletes_eq : male_athletes = 56)
  (sample_size_eq : sample_size = 28)
  : (sample_size * (total_athletes - male_athletes) / total_athletes) = 12 :=
by
  sorry

end NUMINAMATH_GPT_female_athletes_in_sample_l31_3175


namespace NUMINAMATH_GPT_shortest_side_of_similar_triangle_l31_3143

theorem shortest_side_of_similar_triangle (h1 : ∀ (a b c : ℝ), a^2 + b^2 = c^2)
  (h2 : 15^2 + b^2 = 34^2) (h3 : ∃ (k : ℝ), k = 68 / 34) :
  ∃ s : ℝ, s = 2 * Real.sqrt 931 :=
by
  sorry

end NUMINAMATH_GPT_shortest_side_of_similar_triangle_l31_3143


namespace NUMINAMATH_GPT_rhombus_area_l31_3113

theorem rhombus_area (d1 d2 : ℝ) (h_d1 : d1 = 25) (h_d2 : d2 = 50) :
  (d1 * d2) / 2 = 625 := 
by
  sorry

end NUMINAMATH_GPT_rhombus_area_l31_3113


namespace NUMINAMATH_GPT_problem_l31_3164

theorem problem (a b : ℕ) (ha : 2^a ∣ 180) (h2 : ∀ n, 2^n ∣ 180 → n ≤ a) (hb : 5^b ∣ 180) (h5 : ∀ n, 5^n ∣ 180 → n ≤ b) : (1 / 3) ^ (b - a) = 3 := by
  sorry

end NUMINAMATH_GPT_problem_l31_3164


namespace NUMINAMATH_GPT_problem_lean_l31_3157

theorem problem_lean (x y : ℝ) (h₁ : (|x + 2| ≥ 0) ∧ (|y - 4| ≥ 0)) : 
  (|x + 2| = 0 ∧ |y - 4| = 0) → x + y - 3 = -1 :=
by sorry

end NUMINAMATH_GPT_problem_lean_l31_3157


namespace NUMINAMATH_GPT_arrange_f_values_l31_3185

noncomputable def f : ℝ → ℝ := sorry -- Assuming the actual definition is not necessary

-- The function f is even
def even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

-- The function f is strictly decreasing on (-∞, 0)
def strictly_decreasing_on_negative (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2 : ℝ, x1 < 0 ∧ x2 < 0 ∧ x1 ≠ x2 → (x1 < x2 ↔ f x1 > f x2)

theorem arrange_f_values (f : ℝ → ℝ)
  (h_even : even_function f)
  (h_decreasing : strictly_decreasing_on_negative f) :
  f 3 < f (-2) ∧ f (-2) < f 1 :=
by
  -- The actual proof would go here.
  sorry

end NUMINAMATH_GPT_arrange_f_values_l31_3185


namespace NUMINAMATH_GPT_intersection_of_M_and_N_l31_3145

def M : Set ℝ := {x | -3 < x ∧ x < 1}
def N : Set ℝ := {-3, -2, -1, 0, 1}

theorem intersection_of_M_and_N : M ∩ N = {-2, -1, 0} := sorry

end NUMINAMATH_GPT_intersection_of_M_and_N_l31_3145


namespace NUMINAMATH_GPT_cube_volume_increase_l31_3173

variable (a : ℝ) (h : a ≥ 0)

theorem cube_volume_increase :
  ((2 * a) ^ 3) = 8 * (a ^ 3) :=
by sorry

end NUMINAMATH_GPT_cube_volume_increase_l31_3173


namespace NUMINAMATH_GPT_total_books_of_gwen_l31_3114

theorem total_books_of_gwen 
  (mystery_shelves : ℕ) (picture_shelves : ℕ) (books_per_shelf : ℕ)
  (h1 : mystery_shelves = 3) (h2 : picture_shelves = 5) (h3 : books_per_shelf = 9) : 
  mystery_shelves * books_per_shelf + picture_shelves * books_per_shelf = 72 :=
by
  -- Given:
  -- 1. Gwen had 3 shelves of mystery books.
  -- 2. Each shelf had 9 books.
  -- 3. Gwen had 5 shelves of picture books.
  -- 4. Each shelf had 9 books.
  -- Prove:
  -- The total number of books Gwen had is 72.
  sorry

end NUMINAMATH_GPT_total_books_of_gwen_l31_3114


namespace NUMINAMATH_GPT_benjamin_billboards_l31_3194

theorem benjamin_billboards (B : ℕ) (h1 : 20 + 23 + B = 60) : B = 17 :=
by
  sorry

end NUMINAMATH_GPT_benjamin_billboards_l31_3194


namespace NUMINAMATH_GPT_base4_to_base10_conversion_l31_3140

theorem base4_to_base10_conversion : (2 * 4^4 + 1 * 4^3 + 0 * 4^2 + 1 * 4^1 + 2 * 4^0 = 582) :=
by {
  -- The proof is omitted
  sorry
}

end NUMINAMATH_GPT_base4_to_base10_conversion_l31_3140


namespace NUMINAMATH_GPT_total_feed_per_week_l31_3152

-- Define the conditions
def daily_feed_per_pig : ℕ := 10
def number_of_pigs : ℕ := 2
def days_per_week : ℕ := 7

-- Theorem statement
theorem total_feed_per_week : daily_feed_per_pig * number_of_pigs * days_per_week = 140 := 
  sorry

end NUMINAMATH_GPT_total_feed_per_week_l31_3152


namespace NUMINAMATH_GPT_fans_with_all_vouchers_l31_3172

theorem fans_with_all_vouchers (total_fans : ℕ) 
    (soda_interval : ℕ) (popcorn_interval : ℕ) (hotdog_interval : ℕ)
    (h1 : soda_interval = 60) (h2 : popcorn_interval = 80) (h3 : hotdog_interval = 100)
    (h4 : total_fans = 4500)
    (h5 : Nat.lcm soda_interval (Nat.lcm popcorn_interval hotdog_interval) = 1200) :
    (total_fans / Nat.lcm soda_interval (Nat.lcm popcorn_interval hotdog_interval)) = 3 := 
by
    sorry

end NUMINAMATH_GPT_fans_with_all_vouchers_l31_3172


namespace NUMINAMATH_GPT_hotel_room_mistake_l31_3103

theorem hotel_room_mistake (a b c : ℕ) (ha : 1 ≤ a ∧ a ≤ 9) (hb : 0 ≤ b ∧ b ≤ 9) (hc : 0 ≤ c ∧ c ≤ 9) :
  100 * a + 10 * b + c = (a + 1) * (b + 1) * c → false := by sorry

end NUMINAMATH_GPT_hotel_room_mistake_l31_3103


namespace NUMINAMATH_GPT_center_of_symmetry_l31_3182

-- Define the given conditions
def has_axis_symmetry_x (F : Set (ℝ × ℝ)) : Prop := 
  ∀ (x y : ℝ), (x, y) ∈ F → (-x, y) ∈ F

def has_axis_symmetry_y (F : Set (ℝ × ℝ)) : Prop :=
  ∀ (x y : ℝ), (x, y) ∈ F → (x, -y) ∈ F
  
-- Define the central proof goal
theorem center_of_symmetry (F : Set (ℝ × ℝ)) (H1: has_axis_symmetry_x F) (H2: has_axis_symmetry_y F) :
  ∀ (x y : ℝ), (x, y) ∈ F → (-x, -y) ∈ F :=
sorry

end NUMINAMATH_GPT_center_of_symmetry_l31_3182


namespace NUMINAMATH_GPT_g_expression_l31_3134

def f (x : ℝ) : ℝ := 2 * x + 3

def g (x : ℝ) : ℝ := sorry

theorem g_expression :
  (∀ x : ℝ, g (x + 2) = f x) → ∀ x : ℝ, g x = 2 * x - 1 :=
by
  sorry

end NUMINAMATH_GPT_g_expression_l31_3134


namespace NUMINAMATH_GPT_combined_sum_is_115_over_3_l31_3131

def geometric_series_sum (a : ℚ) (r : ℚ) : ℚ :=
  if h : abs r < 1 then a / (1 - r) else 0

def arithmetic_series_sum (a : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  n * (2 * a + (n - 1) * d) / 2

noncomputable def combined_series_sum : ℚ :=
  let geo_sum := geometric_series_sum 5 (-1/2)
  let arith_sum := arithmetic_series_sum 3 2 5
  geo_sum + arith_sum

theorem combined_sum_is_115_over_3 : combined_series_sum = 115 / 3 := 
  sorry

end NUMINAMATH_GPT_combined_sum_is_115_over_3_l31_3131


namespace NUMINAMATH_GPT_solve_for_k_l31_3136

theorem solve_for_k (x y : ℤ) (h₁ : x = 1) (h₂ : y = k) (h₃ : 2 * x + y = 6) : k = 4 :=
by 
  sorry

end NUMINAMATH_GPT_solve_for_k_l31_3136


namespace NUMINAMATH_GPT_height_of_water_a_height_of_water_b_height_of_water_c_l31_3189

noncomputable def edge_length : ℝ := 10  -- Edge length of the cube in cm.
noncomputable def angle_deg : ℝ := 20   -- Angle in degrees.

noncomputable def volume_a : ℝ := 100  -- Volume in cm^3 for case a)
noncomputable def height_a : ℝ := 2.53  -- Height in cm for case a)

noncomputable def volume_b : ℝ := 450  -- Volume in cm^3 for case b)
noncomputable def height_b : ℝ := 5.94  -- Height in cm for case b)

noncomputable def volume_c : ℝ := 900  -- Volume in cm^3 for case c)
noncomputable def height_c : ℝ := 10.29  -- Height in cm for case c)

theorem height_of_water_a :
  ∀ (edge_length angle_deg volume_a : ℝ), volume_a = 100 → height_a = 2.53 := by 
  sorry

theorem height_of_water_b :
  ∀ (edge_length angle_deg volume_b : ℝ), volume_b = 450 → height_b = 5.94 := by 
  sorry

theorem height_of_water_c :
  ∀ (edge_length angle_deg volume_c : ℝ), volume_c = 900 → height_c = 10.29 := by 
  sorry

end NUMINAMATH_GPT_height_of_water_a_height_of_water_b_height_of_water_c_l31_3189


namespace NUMINAMATH_GPT_area_regular_octagon_l31_3192

theorem area_regular_octagon (AB BC: ℝ) (hAB: AB = 2) (hBC: BC = 2) :
  let side_length := 2 * Real.sqrt 2
  let triangle_area := (AB * AB) / 2
  let total_triangle_area := 4 * triangle_area
  let side_length_rect := 4 + 2 * Real.sqrt 2
  let rect_area := side_length_rect * side_length_rect
  let octagon_area := rect_area - total_triangle_area
  octagon_area = 16 + 8 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_GPT_area_regular_octagon_l31_3192


namespace NUMINAMATH_GPT_some_number_is_105_l31_3102

def find_some_number (a : ℕ) (num : ℕ) : Prop :=
  a ^ 3 = 21 * 25 * num * 7

theorem some_number_is_105 (a : ℕ) (num : ℕ) (h : a = 105) (h_eq : find_some_number a num) : num = 105 :=
by
  sorry

end NUMINAMATH_GPT_some_number_is_105_l31_3102


namespace NUMINAMATH_GPT_cream_ratio_l31_3160

theorem cream_ratio (j : ℝ) (jo : ℝ) (jc : ℝ) (joc : ℝ) (jdrank : ℝ) (jodrank : ℝ) :
  j = 15 ∧ jo = 15 ∧ jc = 3 ∧ joc = 2.5 ∧ jdrank = 0 ∧ jodrank = 0.5 →
  j + jc - jdrank = jc ∧ jo + jc - jodrank = joc →
  (jc / joc) = (6 / 5) :=
  by
  sorry

end NUMINAMATH_GPT_cream_ratio_l31_3160


namespace NUMINAMATH_GPT_total_investment_amount_l31_3167

theorem total_investment_amount 
    (x : ℝ) 
    (h1 : 6258.0 * 0.08 + x * 0.065 = 678.87) : 
    x + 6258.0 = 9000.0 :=
sorry

end NUMINAMATH_GPT_total_investment_amount_l31_3167


namespace NUMINAMATH_GPT_initial_ducks_count_l31_3177

theorem initial_ducks_count (D : ℕ) 
  (h1 : ∃ (G : ℕ), G = 2 * D - 10) 
  (h2 : ∃ (D_new : ℕ), D_new = D + 4) 
  (h3 : ∃ (G_new : ℕ), G_new = 2 * D - 20) 
  (h4 : ∀ (D_new G_new : ℕ), G_new = D_new + 1) : 
  D = 25 := by
  sorry

end NUMINAMATH_GPT_initial_ducks_count_l31_3177


namespace NUMINAMATH_GPT_sum_xy_sum_inv_squared_geq_nine_four_l31_3130

variable {x y z : ℝ}

theorem sum_xy_sum_inv_squared_geq_nine_four (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x * y + y * z + z * x) * (1 / (x + y)^2 + 1 / (y + z)^2 + 1 / (z + x)^2) ≥ 9 / 4 :=
by sorry

end NUMINAMATH_GPT_sum_xy_sum_inv_squared_geq_nine_four_l31_3130


namespace NUMINAMATH_GPT_find_ABC_l31_3139

noncomputable def problem (A B C : ℕ) : Prop :=
  A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧
  A < 8 ∧ B < 8 ∧ C < 6 ∧
  (A * 8 + B + C = 8 * 2 + C) ∧
  (A * 8 + B + B * 8 + A = C * 8 + C) ∧
  (100 * A + 10 * B + C = 246)

theorem find_ABC : ∃ A B C : ℕ, problem A B C := sorry

end NUMINAMATH_GPT_find_ABC_l31_3139


namespace NUMINAMATH_GPT_possible_values_of_m_l31_3191

def f (x a m : ℝ) := abs (x - a) + m * abs (x + a)

theorem possible_values_of_m {a m : ℝ} (h1 : 0 < m) (h2 : m < 1) 
  (h3 : ∀ x : ℝ, f x a m ≥ 2)
  (h4 : a ≤ -5 ∨ a ≥ 5) : m = 1 / 5 :=
by 
  sorry

end NUMINAMATH_GPT_possible_values_of_m_l31_3191


namespace NUMINAMATH_GPT_number_of_non_representable_l31_3198

theorem number_of_non_representable :
  ∀ (a b : ℕ), Nat.gcd a b = 1 →
  (∃ n : ℕ, ¬ ∃ x y : ℕ, n = a * x + b * y) :=
sorry

end NUMINAMATH_GPT_number_of_non_representable_l31_3198


namespace NUMINAMATH_GPT_problem1_problem2_l31_3199

-- Problem 1
theorem problem1 (a : ℝ) : 2 * a + 3 * a - 4 * a = a :=
by sorry

-- Problem 2
theorem problem2 : 
  - (1 : ℝ) ^ 2022 + (27 / 4) * (- (1 / 3) - 1) / ((-3) ^ 2) + abs (-1) = -1 :=
by sorry

end NUMINAMATH_GPT_problem1_problem2_l31_3199


namespace NUMINAMATH_GPT_alex_seashells_l31_3180

theorem alex_seashells (mimi_seashells kyle_seashells leigh_seashells alex_seashells : ℕ) 
    (h1 : mimi_seashells = 2 * 12) 
    (h2 : kyle_seashells = 2 * mimi_seashells) 
    (h3 : leigh_seashells = kyle_seashells / 3) 
    (h4 : alex_seashells = 3 * leigh_seashells) : 
  alex_seashells = 48 := by
  sorry

end NUMINAMATH_GPT_alex_seashells_l31_3180


namespace NUMINAMATH_GPT_find_value_of_f2_plus_g3_l31_3115

def f (x : ℝ) : ℝ := 3 * x - 2
def g (x : ℝ) : ℝ := x^2 + 2

theorem find_value_of_f2_plus_g3 : f (2 + g 3) = 37 :=
by
  simp [f, g]
  norm_num
  done

end NUMINAMATH_GPT_find_value_of_f2_plus_g3_l31_3115


namespace NUMINAMATH_GPT_eval_power_l31_3120

theorem eval_power (h : 81 = 3^4) : 81^(5/4) = 243 := by
  sorry

end NUMINAMATH_GPT_eval_power_l31_3120


namespace NUMINAMATH_GPT_geometric_sequence_sum_l31_3151

-- Define the geometric sequence {a_n}
def a (n : ℕ) : ℕ := 2 * (1 ^ (n - 1))

-- Define the sum of the first n terms, s_n
def s (n : ℕ) : ℕ := (Finset.range n).sum (a)

-- The transformed sequence {a_n + 1} assumed also geometric
def b (n : ℕ) : ℕ := a n + 1

-- Lean theorem that s_n = 2n
theorem geometric_sequence_sum (n : ℕ) (h₁ : a 1 = 2) (h₂ : ∀ n, (b (n + 1)) * (b (n + 1)) = (b n * b (n + 2))) : 
  s n = 2 * n :=
sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l31_3151


namespace NUMINAMATH_GPT_cuboid_distance_properties_l31_3183

theorem cuboid_distance_properties (cuboid : Type) :
  (∃ P : cuboid → ℝ, ∀ V1 V2 : cuboid, P V1 = P V2) ∧
  ¬ (∃ Q : cuboid → ℝ, ∀ E1 E2 : cuboid, Q E1 = Q E2) ∧
  ¬ (∃ R : cuboid → ℝ, ∀ F1 F2 : cuboid, R F1 = R F2) := 
sorry

end NUMINAMATH_GPT_cuboid_distance_properties_l31_3183


namespace NUMINAMATH_GPT_total_books_l31_3124

theorem total_books (hbooks : ℕ) (fbooks : ℕ) (gbooks : ℕ)
  (Harry_books : hbooks = 50)
  (Flora_books : fbooks = 2 * hbooks)
  (Gary_books : gbooks = hbooks / 2) :
  hbooks + fbooks + gbooks = 175 := by
  sorry

end NUMINAMATH_GPT_total_books_l31_3124


namespace NUMINAMATH_GPT_two_p_plus_q_l31_3159

variable {p q : ℚ}  -- Variables are rationals

theorem two_p_plus_q (h : p / q = 5 / 4) : 2 * p + q = 7 * q / 2 :=
by sorry

end NUMINAMATH_GPT_two_p_plus_q_l31_3159


namespace NUMINAMATH_GPT_sum_of_side_lengths_l31_3137

theorem sum_of_side_lengths (p q r : ℕ) (h : p = 8 ∧ q = 1 ∧ r = 5) 
    (area_ratio : 128 / 50 = 64 / 25) 
    (side_length_ratio : 8 / 5 = Real.sqrt (128 / 50)) :
    p + q + r = 14 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_side_lengths_l31_3137


namespace NUMINAMATH_GPT_find_b_l31_3127

theorem find_b (b : ℝ) (h_floor : b + ⌊b⌋ = 22.6) : b = 11.6 :=
sorry

end NUMINAMATH_GPT_find_b_l31_3127


namespace NUMINAMATH_GPT_red_balls_removed_to_certain_event_l31_3168

theorem red_balls_removed_to_certain_event (total_balls red_balls yellow_balls : ℕ) (m : ℕ)
  (total_balls_eq : total_balls = 8)
  (red_balls_eq : red_balls = 3)
  (yellow_balls_eq : yellow_balls = 5)
  (certain_event_A : ∀ remaining_red_balls remaining_yellow_balls,
    remaining_red_balls = red_balls - m → remaining_yellow_balls = yellow_balls →
    remaining_red_balls = 0) : m = 3 :=
by
  sorry

end NUMINAMATH_GPT_red_balls_removed_to_certain_event_l31_3168


namespace NUMINAMATH_GPT_mathd_inequality_l31_3141

theorem mathd_inequality (x y z : ℝ) (hx : x ≥ 1) (hy : y ≥ 1) (hz : z ≥ 1) : 
  (x^3 + 2*y^2 + 3*z) * (4*y^3 + 5*z^2 + 6*x) * (7*z^3 + 8*x^2 + 9*y) ≥ 720 * (x * y + y * z + z * x) :=
by
  sorry

end NUMINAMATH_GPT_mathd_inequality_l31_3141


namespace NUMINAMATH_GPT_perimeter_of_square_B_l31_3178

theorem perimeter_of_square_B
  (perimeter_A : ℝ)
  (h_perimeter_A : perimeter_A = 36)
  (area_ratio : ℝ)
  (h_area_ratio : area_ratio = 1 / 3)
  : ∃ (perimeter_B : ℝ), perimeter_B = 12 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_perimeter_of_square_B_l31_3178


namespace NUMINAMATH_GPT_log_function_domain_l31_3116

theorem log_function_domain :
  { x : ℝ | x^2 - 2 * x - 3 > 0 } = { x | x > 3 } ∪ { x | x < -1 } :=
by {
  sorry
}

end NUMINAMATH_GPT_log_function_domain_l31_3116


namespace NUMINAMATH_GPT_part1_part2_l31_3187

noncomputable def f (a x : ℝ) : ℝ := a - 1/x - Real.log x

theorem part1 (a : ℝ) :
  a = 2 → ∃ m b : ℝ, (∀ x : ℝ, f a x = x * m + b) ∧ (∀ y : ℝ, f a 1 = y → b = y ∧ m = 0) :=
by
  sorry

theorem part2 (a : ℝ) :
  (∃! x : ℝ, f a x = 0) → a = 1 :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l31_3187


namespace NUMINAMATH_GPT_total_value_of_item_l31_3125

theorem total_value_of_item (V : ℝ) 
  (h1 : 0.07 * (V - 1000) = 109.20) : 
  V = 2560 :=
sorry

end NUMINAMATH_GPT_total_value_of_item_l31_3125


namespace NUMINAMATH_GPT_grasshopper_catched_in_finite_time_l31_3126

theorem grasshopper_catched_in_finite_time :
  ∀ (x0 y0 x1 y1 : ℤ),
  ∃ (T : ℕ), ∃ (x y : ℤ), 
  ((x = x0 + x1 * T) ∧ (y = y0 + y1 * T)) ∧ -- The hunter will catch the grasshopper at this point
  ((∀ t : ℕ, t ≤ T → (x ≠ x0 + x1 * t ∨ y ≠ y0 + y1 * t) → (x = x0 + x1 * t ∧ y = y0 + y1 * t))) :=
sorry

end NUMINAMATH_GPT_grasshopper_catched_in_finite_time_l31_3126


namespace NUMINAMATH_GPT_circle_tangent_values_l31_3155

theorem circle_tangent_values (m : ℝ) :
  (∀ x y : ℝ, ((x - m)^2 + (y + 2)^2 = 9) → ((x + 1)^2 + (y - m)^2 = 4)) → 
  m = 2 ∨ m = -5 :=
by
  sorry

end NUMINAMATH_GPT_circle_tangent_values_l31_3155


namespace NUMINAMATH_GPT_no_sum_of_consecutive_integers_to_420_l31_3188

noncomputable def perfect_square (n : ℕ) : Prop :=
∃ k : ℕ, k * k = n

def sum_sequence (n a : ℕ) : ℕ :=
n * a + n * (n - 1) / 2

theorem no_sum_of_consecutive_integers_to_420 
  (h1 : 420 > 0)
  (h2 : ∀ (n a : ℕ), n ≥ 2 → sum_sequence n a = 420 → perfect_square a)
  (h3 : ∃ n a, n ≥ 2 ∧ sum_sequence n a = 420 ∧ perfect_square a) :
  false :=
by
  sorry

end NUMINAMATH_GPT_no_sum_of_consecutive_integers_to_420_l31_3188


namespace NUMINAMATH_GPT_min_sum_of_factors_l31_3144

theorem min_sum_of_factors (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_prod : a * b * c = 2310) : 
  a + b + c ≥ 42 :=
sorry

end NUMINAMATH_GPT_min_sum_of_factors_l31_3144


namespace NUMINAMATH_GPT_calculate_b_l31_3106

open Real

theorem calculate_b (b : ℝ) (h : ∫ x in e..b, 2 / x = 6) : b = exp 4 := 
sorry

end NUMINAMATH_GPT_calculate_b_l31_3106


namespace NUMINAMATH_GPT_find_second_sum_l31_3108

def total_sum : ℝ := 2691
def interest_rate_first : ℝ := 0.03
def interest_rate_second : ℝ := 0.05
def time_first : ℝ := 8
def time_second : ℝ := 3

theorem find_second_sum (x second_sum : ℝ) 
  (H : x + second_sum = total_sum)
  (H_interest : x * interest_rate_first * time_first = second_sum * interest_rate_second * time_second) :
  second_sum = 1656 :=
sorry

end NUMINAMATH_GPT_find_second_sum_l31_3108


namespace NUMINAMATH_GPT_system_inconsistent_l31_3142

-- Define the coefficient matrix and the augmented matrices.
def A : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![1, -2, 3], ![2, 3, -1], ![3, 1, 2]]

def B1 : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![5, -2, 3], ![7, 3, -1], ![10, 1, 2]]

-- Calculate the determinants.
noncomputable def delta : ℤ := A.det
noncomputable def delta1 : ℤ := B1.det

-- The main theorem statement: the system is inconsistent if Δ = 0 and Δ1 ≠ 0.
theorem system_inconsistent (h₁ : delta = 0) (h₂ : delta1 ≠ 0) : False :=
sorry

end NUMINAMATH_GPT_system_inconsistent_l31_3142


namespace NUMINAMATH_GPT_bryan_total_earnings_l31_3176

-- Declare the data given in the problem:
def num_emeralds : ℕ := 3
def num_rubies : ℕ := 2
def num_sapphires : ℕ := 3

def price_emerald : ℝ := 1785
def price_ruby : ℝ := 2650
def price_sapphire : ℝ := 2300

-- Calculate the total earnings from each type of stone:
def total_emeralds : ℝ := num_emeralds * price_emerald
def total_rubies : ℝ := num_rubies * price_ruby
def total_sapphires : ℝ := num_sapphires * price_sapphire

-- Calculate the overall total earnings:
def total_earnings : ℝ := total_emeralds + total_rubies + total_sapphires

-- Prove that Bryan got 17555 dollars in total:
theorem bryan_total_earnings : total_earnings = 17555 := by
  simp [total_earnings, total_emeralds, total_rubies, total_sapphires, num_emeralds, num_rubies, num_sapphires, price_emerald, price_ruby, price_sapphire]
  sorry

end NUMINAMATH_GPT_bryan_total_earnings_l31_3176


namespace NUMINAMATH_GPT_compare_31_17_compare_33_63_compare_82_26_compare_29_80_l31_3150

-- Definition and proof obligation for each comparison

theorem compare_31_17 : 31^11 < 17^14 := sorry

theorem compare_33_63 : 33^75 > 63^60 := sorry

theorem compare_82_26 : 82^33 > 26^44 := sorry

theorem compare_29_80 : 29^31 > 80^23 := sorry

end NUMINAMATH_GPT_compare_31_17_compare_33_63_compare_82_26_compare_29_80_l31_3150


namespace NUMINAMATH_GPT_a_range_l31_3138

noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2
def g (x a : ℝ) : ℝ := 2 * x + a

theorem a_range :
  (∃ (x1 x2 : ℝ), x1 ∈ Set.Icc (1 / 2) 2 ∧ x2 ∈ Set.Icc (1 / 2) 2 ∧ f x1 = g x2 a) ↔ -5 ≤ a ∧ a ≤ 0 := 
by 
  sorry

end NUMINAMATH_GPT_a_range_l31_3138


namespace NUMINAMATH_GPT_jillian_largest_apartment_l31_3158

noncomputable def largest_apartment_size (budget : ℝ) (rate : ℝ) : ℝ :=
  budget / rate

theorem jillian_largest_apartment : largest_apartment_size 720 1.20 = 600 := by
  sorry

end NUMINAMATH_GPT_jillian_largest_apartment_l31_3158


namespace NUMINAMATH_GPT_angle_AFE_is_80_degrees_l31_3170

-- Defining the setup and given conditions
def point := ℝ × ℝ  -- defining a 2D point
noncomputable def A : point := (0, 0)
noncomputable def B : point := (1, 0)
noncomputable def C : point := (1, 1)
noncomputable def D : point := (0, 1)
noncomputable def E : point := (-1, 1.732)  -- Place E such that angle CDE ≈ 130 degrees

-- Conditions
def angle_CDE := 130
def DF_over_DE := 2  -- DF = 2 * DE
noncomputable def F : point := (0.5, 1)  -- This is an example position; real positioning depends on more details

-- Proving that the angle AFE is 80 degrees
theorem angle_AFE_is_80_degrees :
  ∃ (AFE : ℝ), AFE = 80 := sorry

end NUMINAMATH_GPT_angle_AFE_is_80_degrees_l31_3170


namespace NUMINAMATH_GPT_least_number_to_subtract_l31_3147

theorem least_number_to_subtract (n m : ℕ) (h : n = 56783421) (d : m = 569) : (n % m) = 56783421 % 569 := 
by sorry

end NUMINAMATH_GPT_least_number_to_subtract_l31_3147


namespace NUMINAMATH_GPT_min_right_triangles_cover_equilateral_triangle_l31_3118

theorem min_right_triangles_cover_equilateral_triangle :
  let side_length_equilateral := 12
  let legs_right_triangle := 1
  let area_equilateral := (Real.sqrt 3 / 4) * side_length_equilateral ^ 2
  let area_right_triangle := (1 / 2) * legs_right_triangle * legs_right_triangle
  let triangles_needed := area_equilateral / area_right_triangle
  triangles_needed = 72 * Real.sqrt 3 := 
by 
  sorry

end NUMINAMATH_GPT_min_right_triangles_cover_equilateral_triangle_l31_3118


namespace NUMINAMATH_GPT_modulus_of_complex_l31_3190

open Complex

theorem modulus_of_complex (z : ℂ) (h : z = 1 - (1 / Complex.I)) : Complex.abs z = Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_modulus_of_complex_l31_3190


namespace NUMINAMATH_GPT_number_of_valid_sequences_l31_3111

-- Define the sequence and conditions
def is_valid_sequence (a : Fin 9 → ℝ) : Prop :=
  a 0 = 1 ∧ a 8 = 1 ∧
  ∀ i : Fin 8, (a (i + 1) / a i) ∈ ({2, 1, -1/2} : Set ℝ)

-- The main problem statement
theorem number_of_valid_sequences : ∃ n, n = 491 ∧ ∀ a : Fin 9 → ℝ, is_valid_sequence a ↔ n = 491 := 
sorry

end NUMINAMATH_GPT_number_of_valid_sequences_l31_3111


namespace NUMINAMATH_GPT_length_of_AB_l31_3154

-- Definitions of the given entities
def is_on_parabola (A : ℝ × ℝ) : Prop := A.2^2 = 4 * A.1
def focus : ℝ × ℝ := (1, 0)
def line_through_focus (l : ℝ × ℝ → Prop) : Prop := l focus

-- The theorem we need to prove
theorem length_of_AB (A B : ℝ × ℝ) (l : ℝ × ℝ → Prop)
  (h1 : is_on_parabola A)
  (h2 : is_on_parabola B)
  (h3 : line_through_focus l)
  (h4 : l A)
  (h5 : l B)
  (h6 : A.1 + B.1 = 10 / 3) :
  dist A B = 16 / 3 :=
sorry

end NUMINAMATH_GPT_length_of_AB_l31_3154


namespace NUMINAMATH_GPT_prob_two_sunny_days_l31_3112

-- Define the probability of rain and sunny
def probRain : ℚ := 3 / 4
def probSunny : ℚ := 1 / 4

-- Define the problem statement
theorem prob_two_sunny_days : (10 * (probSunny^2) * (probRain^3)) = 135 / 512 := 
by
  sorry

end NUMINAMATH_GPT_prob_two_sunny_days_l31_3112


namespace NUMINAMATH_GPT_statement_II_and_IV_true_l31_3100

-- Definitions based on the problem's conditions
def AllNewEditions (P : Type) (books : P → Prop) := ∀ x, books x

-- Condition that the statement "All books in the library are new editions." is false
def NotAllNewEditions (P : Type) (books : P → Prop) := ¬ (AllNewEditions P books)

-- Statements to analyze
def SomeBookNotNewEdition (P : Type) (books : P → Prop) := ∃ x, ¬ books x
def NotAllBooksNewEditions (P : Type) (books : P → Prop) := ∃ x, ¬ books x

-- The theorem to prove
theorem statement_II_and_IV_true 
  (P : Type) 
  (books : P → Prop) 
  (h : NotAllNewEditions P books) : 
  SomeBookNotNewEdition P books ∧ NotAllBooksNewEditions P books :=
  by
    sorry

end NUMINAMATH_GPT_statement_II_and_IV_true_l31_3100


namespace NUMINAMATH_GPT_product_of_possible_values_l31_3179

theorem product_of_possible_values (N : ℤ) (M L : ℤ) 
(h1 : M = L + N)
(h2 : M - 3 = L + N - 3)
(h3 : L + 5 = L + 5)
(h4 : |(L + N - 3) - (L + 5)| = 4) :
N = 12 ∨ N = 4 → (12 * 4 = 48) :=
by sorry

end NUMINAMATH_GPT_product_of_possible_values_l31_3179


namespace NUMINAMATH_GPT_find_x_y_l31_3195

theorem find_x_y (x y : ℤ) (hx : 0 < x) (hy : 0 < y) (h : (x + y * Complex.I)^2 = (7 + 24 * Complex.I)) :
  x + y * Complex.I = 4 + 3 * Complex.I :=
by
  sorry

end NUMINAMATH_GPT_find_x_y_l31_3195


namespace NUMINAMATH_GPT_find_a9_l31_3197

noncomputable def a : ℕ → ℕ
| 0     => 0
| 1     => 1
| (n+1) => a n + n

theorem find_a9 : a 9 = 37 := by
  sorry

end NUMINAMATH_GPT_find_a9_l31_3197


namespace NUMINAMATH_GPT_relationship_y1_y2_y3_l31_3171

theorem relationship_y1_y2_y3 :
  ∀ (y1 y2 y3 : ℝ), y1 = 6 ∧ y2 = 3 ∧ y3 = -2 → y1 > y2 ∧ y2 > y3 :=
by 
  intros y1 y2 y3 h
  sorry

end NUMINAMATH_GPT_relationship_y1_y2_y3_l31_3171


namespace NUMINAMATH_GPT_grassy_area_percentage_l31_3135

noncomputable def percentage_grassy_area (park_area path1_area path2_area intersection_area : ℝ) : ℝ :=
  let covered_by_paths := path1_area + path2_area - intersection_area
  let grassy_area := park_area - covered_by_paths
  (grassy_area / park_area) * 100

theorem grassy_area_percentage (park_area : ℝ) (path1_area : ℝ) (path2_area : ℝ) (intersection_area : ℝ) 
  (h1 : park_area = 4000) (h2 : path1_area = 400) (h3 : path2_area = 250) (h4 : intersection_area = 25) : 
  percentage_grassy_area park_area path1_area path2_area intersection_area = 84.375 :=
by
  rw [percentage_grassy_area, h1, h2, h3, h4]
  simp
  sorry

end NUMINAMATH_GPT_grassy_area_percentage_l31_3135


namespace NUMINAMATH_GPT_figure_representation_l31_3109

theorem figure_representation (x y : ℝ) : 
  |x| + |y| ≤ 2 * Real.sqrt (x^2 + y^2) ∧ 2 * Real.sqrt (x^2 + y^2) ≤ 3 * max (|x|) (|y|) → 
  Figure2 :=
sorry

end NUMINAMATH_GPT_figure_representation_l31_3109


namespace NUMINAMATH_GPT_mark_eggs_supply_l31_3117

theorem mark_eggs_supply 
  (dozens_per_day : ℕ)
  (eggs_per_dozen : ℕ)
  (extra_eggs : ℕ)
  (days_in_week : ℕ)
  (H1 : dozens_per_day = 5)
  (H2 : eggs_per_dozen = 12)
  (H3 : extra_eggs = 30)
  (H4 : days_in_week = 7) :
  (dozens_per_day * eggs_per_dozen + extra_eggs) * days_in_week = 630 :=
by
  sorry

end NUMINAMATH_GPT_mark_eggs_supply_l31_3117


namespace NUMINAMATH_GPT_find_value_of_M_l31_3104

theorem find_value_of_M (M : ℝ) (h : 0.2 * M = 0.6 * 1230) : M = 3690 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_value_of_M_l31_3104


namespace NUMINAMATH_GPT_scientific_notation_of_0_000000023_l31_3107

theorem scientific_notation_of_0_000000023 : 
  0.000000023 = 2.3 * 10^(-8) :=
sorry

end NUMINAMATH_GPT_scientific_notation_of_0_000000023_l31_3107


namespace NUMINAMATH_GPT_weight_of_five_single_beds_l31_3119

-- Define the problem conditions and the goal
theorem weight_of_five_single_beds :
  ∃ S D : ℝ, (2 * S + 4 * D = 100) ∧ (D = S + 10) → (5 * S = 50) :=
by
  sorry

end NUMINAMATH_GPT_weight_of_five_single_beds_l31_3119


namespace NUMINAMATH_GPT_sum_of_digits_l31_3148

theorem sum_of_digits (a b c d : ℕ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
                      (h_range : 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ 1 ≤ c ∧ c ≤ 9 ∧ 1 ≤ d ∧ d ≤ 9)
                      (h_product : a * b * c * d = 810) :
  a + b + c + d = 23 := sorry

end NUMINAMATH_GPT_sum_of_digits_l31_3148


namespace NUMINAMATH_GPT_find_horses_l31_3101

theorem find_horses {x : ℕ} :
  (841 : ℝ) = 8 * (x : ℝ) + 16 * 9 + 18 * 6 → 348 = 16 * 9 →
  x = 73 :=
by
  intros h₁ h₂
  sorry

end NUMINAMATH_GPT_find_horses_l31_3101


namespace NUMINAMATH_GPT_surface_area_of_sphere_l31_3123

theorem surface_area_of_sphere (a b c : ℝ) (h1 : a = 1) (h2 : b = 2) (h3 : c = 2)
  (h4 : ∀ d, d = Real.sqrt (a^2 + b^2 + c^2)) : 
  4 * Real.pi * (d / 2)^2 = 9 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_surface_area_of_sphere_l31_3123


namespace NUMINAMATH_GPT_equal_roots_implies_c_value_l31_3105

theorem equal_roots_implies_c_value (c : ℝ) 
  (h : ∃ x : ℝ, (x^2 + 6 * x - c = 0) ∧ (2 * x + 6 = 0)) :
  c = -9 :=
sorry

end NUMINAMATH_GPT_equal_roots_implies_c_value_l31_3105


namespace NUMINAMATH_GPT_range_of_a_l31_3163

theorem range_of_a (M N : Set ℝ) (a : ℝ) 
(hM : M = {x : ℝ | x < 2}) 
(hN : N = {x : ℝ | x < a}) 
(hSubset : M ⊆ N) : 
  2 ≤ a := 
sorry

end NUMINAMATH_GPT_range_of_a_l31_3163


namespace NUMINAMATH_GPT_exponentiation_power_rule_l31_3166

theorem exponentiation_power_rule (a : ℝ) : (a^2)^3 = a^6 :=
by sorry

end NUMINAMATH_GPT_exponentiation_power_rule_l31_3166


namespace NUMINAMATH_GPT_set_intersection_correct_l31_3122

def set_A := {x : ℝ | x + 1 > 0}
def set_B := {x : ℝ | x - 3 < 0}
def set_intersection := {x : ℝ | -1 < x ∧ x < 3}

theorem set_intersection_correct : (set_A ∩ set_B) = set_intersection :=
by
  sorry

end NUMINAMATH_GPT_set_intersection_correct_l31_3122


namespace NUMINAMATH_GPT_monotonicity_f_inequality_proof_l31_3169

noncomputable def f (x : ℝ) : ℝ := Real.log x - x + 1

theorem monotonicity_f :
  (∀ x : ℝ, 0 < x ∧ x < 1 → f x < f (x + ε)) ∧ (∀ x : ℝ, 1 < x → f (x - ε) > f x) := 
sorry

theorem inequality_proof (x : ℝ) (hx : 1 < x) :
  1 < (x - 1) / Real.log x ∧ (x - 1) / Real.log x < x :=
sorry

end NUMINAMATH_GPT_monotonicity_f_inequality_proof_l31_3169


namespace NUMINAMATH_GPT_halfway_between_one_eighth_and_one_tenth_l31_3193

theorem halfway_between_one_eighth_and_one_tenth :
  (1 / 8 + 1 / 10) / 2 = 9 / 80 :=
by
  sorry

end NUMINAMATH_GPT_halfway_between_one_eighth_and_one_tenth_l31_3193


namespace NUMINAMATH_GPT_recurring_to_fraction_l31_3121

theorem recurring_to_fraction : ∀ (x : ℚ), x = 0.5656 ∧ 100 * x = 56.5656 → x = 56 / 99 :=
by
  intros x hx
  sorry

end NUMINAMATH_GPT_recurring_to_fraction_l31_3121


namespace NUMINAMATH_GPT_transform_f_to_shift_left_l31_3161

theorem transform_f_to_shift_left (f : ℝ → ℝ) :
  ∀ x : ℝ, f (2 * x - 1) = f (2 * (x - 1) + 1) := by
  sorry

end NUMINAMATH_GPT_transform_f_to_shift_left_l31_3161


namespace NUMINAMATH_GPT_days_to_complete_work_l31_3128

theorem days_to_complete_work :
  ∀ (M B: ℝ) (D: ℝ),
    (M = 2 * B)
    → (13 * M + 24 * B) * 4 = (12 * M + 16 * B) * D
    → D = 5 :=
by
  intros M B D h1 h2
  sorry

end NUMINAMATH_GPT_days_to_complete_work_l31_3128


namespace NUMINAMATH_GPT_line_x_intercept_l31_3162

-- Define the given points
def Point1 : ℝ × ℝ := (10, 3)
def Point2 : ℝ × ℝ := (-10, -7)

-- Define the x-intercept problem
theorem line_x_intercept (x : ℝ) : 
  ∃ m b : ℝ, (Point1.2 = m * Point1.1 + b) ∧ (Point2.2 = m * Point2.1 + b) ∧ (0 = m * x + b) → x = 4 :=
by
  sorry

end NUMINAMATH_GPT_line_x_intercept_l31_3162


namespace NUMINAMATH_GPT_find_m_l31_3174

theorem find_m (m : ℝ) : (∀ x : ℝ, 0 < x → x < 2 → - (1/2)*x^2 + 2*x > -m*x) ↔ m = -1 := 
sorry

end NUMINAMATH_GPT_find_m_l31_3174


namespace NUMINAMATH_GPT_compound_interest_l31_3184

theorem compound_interest 
  (P : ℝ) (r : ℝ) (t : ℕ) : P = 500 → r = 0.02 → t = 3 → (P * (1 + r)^t) - P = 30.60 :=
by
  intros P_invest rate years
  simp [P_invest, rate, years]
  sorry

end NUMINAMATH_GPT_compound_interest_l31_3184


namespace NUMINAMATH_GPT_vector_parallel_condition_l31_3196

def OA : ℝ × ℝ := (3, -4)
def OB : ℝ × ℝ := (6, -3)
def OC (m : ℝ) : ℝ × ℝ := (2 * m, m + 1)

def AB (OA OB : ℝ × ℝ) : ℝ × ℝ := (OB.1 - OA.1, OB.2 - OA.2)

theorem vector_parallel_condition (m : ℝ) (h_parallel : AB OA OB = (3, 1) ∧ 
    (∀ k : ℝ, 2*m = 3*k ∧ m + 1 = k)) : m = -3 :=
by
  sorry

end NUMINAMATH_GPT_vector_parallel_condition_l31_3196


namespace NUMINAMATH_GPT_sin_cos_quotient_l31_3149

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x
noncomputable def f_prime (x : ℝ) : ℝ := Real.cos x - Real.sin x

theorem sin_cos_quotient 
  (x : ℝ)
  (h : f_prime x = 3 * f x) 
  : (Real.sin x ^ 2 - 3) / (Real.cos x ^ 2 + 1) = -14 / 9 := 
by 
  sorry

end NUMINAMATH_GPT_sin_cos_quotient_l31_3149


namespace NUMINAMATH_GPT_max_cos_a_correct_l31_3186

noncomputable def max_cos_a (a b : ℝ) (h : Real.cos (a + b) = Real.cos a + Real.cos b) : ℝ :=
  Real.sqrt 3 - 1

theorem max_cos_a_correct (a b : ℝ) (h : Real.cos (a + b) = Real.cos a + Real.cos b) :
  max_cos_a a b h = Real.sqrt 3 - 1 :=
sorry

end NUMINAMATH_GPT_max_cos_a_correct_l31_3186


namespace NUMINAMATH_GPT_polynomial_evaluation_l31_3156

theorem polynomial_evaluation (x : ℝ) (h₁ : 0 < x) (h₂ : x^2 - 2 * x - 15 = 0) :
  x^3 - 2 * x^2 - 8 * x + 16 = 51 :=
sorry

end NUMINAMATH_GPT_polynomial_evaluation_l31_3156


namespace NUMINAMATH_GPT_minimum_value_of_a_plus_5b_l31_3133

theorem minimum_value_of_a_plus_5b :
  ∀ (a b : ℝ), a > 0 → b > 0 → (1 / a + 5 / b = 1) → a + 5 * b ≥ 36 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_a_plus_5b_l31_3133
