import Mathlib

namespace abs_diff_101st_term_l1629_162949

theorem abs_diff_101st_term 
  (C D : ℕ → ℤ)
  (hC_start : C 0 = 20)
  (hD_start : D 0 = 20)
  (hC_diff : ∀ n, C (n + 1) = C n + 12)
  (hD_diff : ∀ n, D (n + 1) = D n - 6) :
  |C 100 - D 100| = 1800 :=
by
  sorry

end abs_diff_101st_term_l1629_162949


namespace makenna_garden_larger_by_132_l1629_162965

-- Define the dimensions of Karl's garden
def length_karl : ℕ := 22
def width_karl : ℕ := 50

-- Define the dimensions of Makenna's garden including the walking path
def length_makenna_total : ℕ := 30
def width_makenna_total : ℕ := 46
def walking_path_width : ℕ := 1

-- Define the area calculation functions
def area (length : ℕ) (width : ℕ) : ℕ := length * width

-- Calculate the areas
def area_karl : ℕ := area length_karl width_karl
def area_makenna : ℕ := area (length_makenna_total - 2 * walking_path_width) (width_makenna_total - 2 * walking_path_width)

-- Define the theorem to prove
theorem makenna_garden_larger_by_132 :
  area_makenna = area_karl + 132 :=
by
  -- We skip the proof part
  sorry

end makenna_garden_larger_by_132_l1629_162965


namespace find_number_l1629_162994

theorem find_number (x : ℝ) (h : (5/3) * x = 45) : x = 27 :=
by
  sorry

end find_number_l1629_162994


namespace students_not_taking_music_nor_art_l1629_162959

theorem students_not_taking_music_nor_art (total_students music_students art_students both_students neither_students : ℕ) 
  (h_total : total_students = 500) 
  (h_music : music_students = 50) 
  (h_art : art_students = 20) 
  (h_both : both_students = 10) 
  (h_neither : neither_students = total_students - (music_students + art_students - both_students)) : 
  neither_students = 440 :=
by
  sorry

end students_not_taking_music_nor_art_l1629_162959


namespace solve_inequality_l1629_162969

theorem solve_inequality (x : ℝ) : (3 * x - 5) / 2 > 2 * x → x < -5 :=
by
  sorry

end solve_inequality_l1629_162969


namespace Jeongyeon_record_is_1_44_m_l1629_162937

def Eunseol_record_in_cm : ℕ := 100 + 35
def Jeongyeon_record_in_cm : ℕ := Eunseol_record_in_cm + 9
def Jeongyeon_record_in_m : ℚ := Jeongyeon_record_in_cm / 100

theorem Jeongyeon_record_is_1_44_m : Jeongyeon_record_in_m = 1.44 := by
  sorry

end Jeongyeon_record_is_1_44_m_l1629_162937


namespace odd_even_shift_composition_l1629_162955

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def is_even_function_shifted (f : ℝ → ℝ) (shift : ℝ) : Prop :=
  ∀ x : ℝ, f (x + shift) = f (-x + shift)

theorem odd_even_shift_composition
  (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_even_shift : is_even_function_shifted f 3)
  (h_f1 : f 1 = 1) :
  f 6 + f 11 = -1 := by
  sorry

end odd_even_shift_composition_l1629_162955


namespace no_valid_arrangement_l1629_162941

open Nat

theorem no_valid_arrangement :
  ¬ ∃ (f : Fin 30 → ℕ), 
    (∀ (i : Fin 30), 1 ≤ f i ∧ f i ≤ 30) ∧ 
    (∀ (i : Fin 30), ∃ n : ℕ, (f i + f (i + 1) % 30) = n^2) ∧ 
    (∀ i1 i2, i1 ≠ i2 → f i1 ≠ f i2) :=
  sorry

end no_valid_arrangement_l1629_162941


namespace ordered_triples_count_l1629_162957

open Real

theorem ordered_triples_count :
  ∃ (S : Finset (ℝ × ℝ × ℝ)),
    (∀ (a b c : ℝ), (a, b, c) ∈ S ↔ (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ ab = c ∧ bc = a + b ∧ ca = b)) ∧
    S.card = 2 := 
sorry

end ordered_triples_count_l1629_162957


namespace point_not_on_graph_l1629_162917

theorem point_not_on_graph : ∀ (x y : ℝ), (x, y) = (-1, 1) → ¬ (∃ z : ℝ, z ≠ -1 ∧ y = z / (z + 1)) :=
by {
  sorry
}

end point_not_on_graph_l1629_162917


namespace circle_center_radius_l1629_162968

theorem circle_center_radius :
  ∀ (x y : ℝ), (x + 1) ^ 2 + (y - 2) ^ 2 = 9 ↔ (x = -1 ∧ y = 2 ∧ ∃ r : ℝ, r = 3) :=
by
  sorry

end circle_center_radius_l1629_162968


namespace parabola_focus_l1629_162929

theorem parabola_focus (a : ℝ) (h : a ≠ 0) (h_directrix : ∀ x y : ℝ, y^2 = a * x → x = -1) : 
    ∃ x y : ℝ, (y = 0 ∧ x = 1 ∧ y^2 = a * x) :=
sorry

end parabola_focus_l1629_162929


namespace point_inside_circle_l1629_162906

theorem point_inside_circle (a : ℝ) :
  ((1 - a) ^ 2 + (1 + a) ^ 2 < 4) → (-1 < a ∧ a < 1) :=
by
  sorry

end point_inside_circle_l1629_162906


namespace extreme_values_a_4_find_a_minimum_minus_5_l1629_162995

noncomputable def f (x a : ℝ) : ℝ := 2 * x^2 - a * x + 5

theorem extreme_values_a_4 :
  (∀ x, x ∈ Set.Icc (-1:ℝ) 2 -> f x 4 ≤ 11) ∧ (∃ x, x ∈ Set.Icc (-1:ℝ) 2 ∧ f x 4 = 11) ∧
  (∀ x, x ∈ Set.Icc (-1:ℝ) 2 -> f x 4 ≥ 3) ∧ (∃ x, x ∈ Set.Icc (-1:ℝ) 2 ∧ f x 4 = 3) :=
  sorry

theorem find_a_minimum_minus_5 :
  ∀ (a : ℝ), (∃ x, x ∈ Set.Icc (-1:ℝ) 2 ∧ f x a = -5) -> (a = -12 ∨ a = 9) :=
  sorry

end extreme_values_a_4_find_a_minimum_minus_5_l1629_162995


namespace fraction_identity_l1629_162920

theorem fraction_identity (a b : ℝ) (h : a / b = 5 / 2) : (a + 2 * b) / (a - b) = 3 :=
by sorry

end fraction_identity_l1629_162920


namespace sufficient_but_not_necessary_condition_l1629_162990

noncomputable def is_increasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (∀ x y : ℝ, x < y → (a - 1) * (a ^ x) < (a - 1) * (a ^ y) → a > 1) ∧
  (¬ (∀ c : ℝ, is_increasing_function (λ x => (c - 1) * (c ^ x)) → c > 1)) :=
sorry

end sufficient_but_not_necessary_condition_l1629_162990


namespace parabola_and_line_solutions_l1629_162925

-- Definition of the parabola with its focus
def parabola_with_focus (p : ℝ) : Prop :=
  (∃ (y x : ℝ), y^2 = 2 * p * x) ∧ (∃ (x : ℝ), x = 1 / 2)

-- Definitions of conditions for intersection and orthogonal vectors
def line_intersecting_parabola (slope t : ℝ) (p : ℝ) : Prop :=
  ∃ (x1 x2 y1 y2 : ℝ), 
  (y1 = 2 * x1 + t) ∧ (y2 = 2 * x2 + t) ∧
  (y1^2 = 2 * x1) ∧ (y2^2 = 2 * x2) ∧
  (x1 ≠ 0) ∧ (x2 ≠ 0) ∧
  (x1 * x2 = (t^2) / 4) ∧ (x1 * x2 + y1 * y2 = 0)

-- Lean statement for the proof problem
theorem parabola_and_line_solutions :
  ∀ p t : ℝ, 
  parabola_with_focus p → 
  (line_intersecting_parabola 2 t p → t = -4)
  → p = 1 :=
by
  intros p t h_parabola h_line
  sorry

end parabola_and_line_solutions_l1629_162925


namespace total_students_l1629_162928

theorem total_students (rank_right rank_left : ℕ) (h_right : rank_right = 18) (h_left : rank_left = 12) : rank_right + rank_left - 1 = 29 := 
by
  sorry

end total_students_l1629_162928


namespace three_digit_max_l1629_162976

theorem three_digit_max (n : ℕ) : 
  n % 9 = 1 ∧ n % 5 = 3 ∧ n % 7 = 2 ∧ 100 <= n ∧ n <= 999 → n = 793 :=
by
  sorry

end three_digit_max_l1629_162976


namespace turns_per_minute_l1629_162989

theorem turns_per_minute (x : ℕ) (h₁ : x > 0) (h₂ : 60 / x = (60 / (x + 5)) + 2) :
  60 / x = 6 ∧ 60 / (x + 5) = 4 :=
by sorry

end turns_per_minute_l1629_162989


namespace div_floor_factorial_l1629_162973

theorem div_floor_factorial (n q : ℕ) (hn : n ≥ 5) (hq : 2 ≤ q ∧ q ≤ n) :
  q - 1 ∣ (Nat.floor ((Nat.factorial (n - 1)) / q : ℚ)) :=
by
  sorry

end div_floor_factorial_l1629_162973


namespace can_cut_into_equal_parts_l1629_162945

-- We assume the existence of a shape S and some grid G along with a function cut
-- that cuts the shape S along grid G lines and returns two parts.
noncomputable def Shape := Type
noncomputable def Grid := Type
noncomputable def cut (S : Shape) (G : Grid) : Shape × Shape := sorry

-- We assume a function superimpose that checks whether two shapes can be superimposed
noncomputable def superimpose (S1 S2 : Shape) : Prop := sorry

-- Assume the given shape S and grid G
variable (S : Shape) (G : Grid)

-- The question rewritten as a Lean statement
theorem can_cut_into_equal_parts : ∃ (S₁ S₂ : Shape), cut S G = (S₁, S₂) ∧ superimpose S₁ S₂ := sorry

end can_cut_into_equal_parts_l1629_162945


namespace find_dinner_bill_l1629_162916

noncomputable def total_dinner_bill (B : ℝ) (silas_share : ℝ) (remaining_friends_pay : ℝ) (each_friend_pays : ℝ) :=
  silas_share = (1/2) * B ∧
  remaining_friends_pay = (1/2) * B + 0.10 * B ∧
  each_friend_pays = remaining_friends_pay / 5 ∧
  each_friend_pays = 18

theorem find_dinner_bill : ∃ B : ℝ, total_dinner_bill B ((1/2) * B) ((1/2) * B + 0.10 * B) (18) → B = 150 :=
by
  sorry

end find_dinner_bill_l1629_162916


namespace area_triangle_ABC_l1629_162988

noncomputable def area_of_triangle_ABC : ℝ :=
  let base_AB : ℝ := 6 - 0
  let height_AB : ℝ := 2 - 0
  let base_BC : ℝ := 6 - 3
  let height_BC : ℝ := 8 - 0
  let base_CA : ℝ := 3 - 0
  let height_CA : ℝ := 8 - 2
  let area_ratio : ℝ := 1 / 2
  let area_I' : ℝ := area_ratio * base_AB * height_AB
  let area_II' : ℝ := area_ratio * 8 * 6
  let area_III' : ℝ := area_ratio * 8 * 3
  let total_small_triangles : ℝ := area_I' + area_II' + area_III'
  let total_area_rectangle : ℝ := 6 * 8
  total_area_rectangle - total_small_triangles

theorem area_triangle_ABC : area_of_triangle_ABC = 6 := 
by
  sorry

end area_triangle_ABC_l1629_162988


namespace calc_value_l1629_162948

theorem calc_value (a b : ℝ) (h : b = 3 * a - 2) : 2 * b - 6 * a + 2 = -2 := 
by 
  sorry

end calc_value_l1629_162948


namespace how_many_strawberries_did_paul_pick_l1629_162987

-- Here, we will define the known quantities
def original_strawberries : Nat := 28
def total_strawberries : Nat := 63

-- The statement to prove
theorem how_many_strawberries_did_paul_pick : total_strawberries - original_strawberries = 35 :=
by
  unfold total_strawberries
  unfold original_strawberries
  calc
    63 - 28 = 35 := by norm_num

end how_many_strawberries_did_paul_pick_l1629_162987


namespace Jane_remaining_time_l1629_162924

noncomputable def JaneRate : ℚ := 1 / 4
noncomputable def RoyRate : ℚ := 1 / 5
noncomputable def workingTime : ℚ := 2
noncomputable def cakeFractionCompletedTogether : ℚ := (JaneRate + RoyRate) * workingTime
noncomputable def remainingCakeFraction : ℚ := 1 - cakeFractionCompletedTogether
noncomputable def timeForJaneToCompleteRemainingCake : ℚ := remainingCakeFraction / JaneRate

theorem Jane_remaining_time :
  timeForJaneToCompleteRemainingCake = 2 / 5 :=
by
  sorry

end Jane_remaining_time_l1629_162924


namespace gcd_1729_1314_l1629_162953

theorem gcd_1729_1314 : Nat.gcd 1729 1314 = 1 :=
by
  sorry

end gcd_1729_1314_l1629_162953


namespace actual_distance_in_km_l1629_162901

-- Given conditions
def scale_factor : ℕ := 200000
def map_distance_cm : ℚ := 3.5

-- Proof goal: the actual distance in kilometers
theorem actual_distance_in_km : (map_distance_cm * scale_factor) / 100000 = 7 := 
by
  sorry

end actual_distance_in_km_l1629_162901


namespace bicycle_cost_calculation_l1629_162942

theorem bicycle_cost_calculation 
  (CP_A CP_B CP_C : ℝ)
  (h1 : CP_B = 1.20 * CP_A)
  (h2 : CP_C = 1.25 * CP_B)
  (h3 : CP_C = 225) :
  CP_A = 150 :=
by
  sorry

end bicycle_cost_calculation_l1629_162942


namespace shaded_region_area_l1629_162962

-- Conditions given in the problem
def diameter (d : ℝ) := d = 4
def length_feet (l : ℝ) := l = 2

-- Proof statement
theorem shaded_region_area (d l : ℝ) (h1 : diameter d) (h2 : length_feet l) : 
  (l * 12 / d * (d / 2)^2 * π = 24 * π) := by
  sorry

end shaded_region_area_l1629_162962


namespace find_original_denominator_l1629_162930

variable (d : ℕ)

theorem find_original_denominator
  (h1 : ∀ n : ℕ, n = 3)
  (h2 : 3 + 7 = 10)
  (h3 : (10 : ℕ) = 1 * (d + 7) / 3) :
  d = 23 := by
  sorry

end find_original_denominator_l1629_162930


namespace dawn_bananas_l1629_162952

-- Definitions of the given conditions
def total_bananas : ℕ := 200
def lydia_bananas : ℕ := 60
def donna_bananas : ℕ := 40

-- Proof that Dawn has 100 bananas
theorem dawn_bananas : (total_bananas - donna_bananas) - lydia_bananas = 100 := by
  sorry

end dawn_bananas_l1629_162952


namespace pure_acid_total_is_3_8_l1629_162997

/-- Volume of Solution A in liters -/
def volume_A : ℝ := 8

/-- Concentration of Solution A (in decimals, i.e., 20% as 0.20) -/
def concentration_A : ℝ := 0.20

/-- Volume of Solution B in liters -/
def volume_B : ℝ := 5

/-- Concentration of Solution B (in decimals, i.e., 35% as 0.35) -/
def concentration_B : ℝ := 0.35

/-- Volume of Solution C in liters -/
def volume_C : ℝ := 3

/-- Concentration of Solution C (in decimals, i.e., 15% as 0.15) -/
def concentration_C : ℝ := 0.15

/-- Total amount of pure acid in the resulting mixture -/
def total_pure_acid : ℝ :=
  (volume_A * concentration_A) +
  (volume_B * concentration_B) +
  (volume_C * concentration_C)

theorem pure_acid_total_is_3_8 : total_pure_acid = 3.8 := by
  sorry

end pure_acid_total_is_3_8_l1629_162997


namespace max_page_number_with_given_fives_l1629_162970

theorem max_page_number_with_given_fives (plenty_digit_except_five : ℕ → ℕ) 
  (H0 : ∀ d ≠ 5, ∀ n, plenty_digit_except_five d = n)
  (H5 : plenty_digit_except_five 5 = 30) : ∃ (n : ℕ), n = 154 :=
by {
  sorry
}

end max_page_number_with_given_fives_l1629_162970


namespace polynomial_composite_l1629_162986

theorem polynomial_composite (x : ℕ) : ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ 4 * x^3 + 6 * x^2 + 4 * x + 1 = a * b :=
by
  sorry

end polynomial_composite_l1629_162986


namespace no_always_1x3_rectangle_l1629_162922

/-- From a sheet of graph paper measuring 8 x 8 cells, 12 rectangles of size 1 x 2 were cut out along the grid lines. 
Prove that it is not necessarily possible to always find a 1 x 3 checkered rectangle in the remaining part. -/
theorem no_always_1x3_rectangle (grid_size : ℕ) (rectangles_removed : ℕ) (rect_size : ℕ) :
  grid_size = 64 → rectangles_removed * rect_size = 24 → ¬ (∀ remaining_cells, remaining_cells ≥ 0 → remaining_cells ≤ 64 → ∃ (x y : ℕ), remaining_cells = x * y ∧ x = 1 ∧ y = 3) :=
  by
  intro h1 h2 h3
  /- Exact proof omitted for brevity -/
  sorry

end no_always_1x3_rectangle_l1629_162922


namespace second_number_added_is_5_l1629_162960

theorem second_number_added_is_5
  (x : ℕ) (h₁ : x = 3)
  (y : ℕ)
  (h₂ : (x + 1) * (x + 13) = (x + y) * (x + y)) :
  y = 5 :=
sorry

end second_number_added_is_5_l1629_162960


namespace shortest_fence_length_l1629_162985

-- We define the conditions given in the problem.
def triangle_side_length : ℕ := 50
def number_of_dotted_lines : ℕ := 13

-- We need to prove that the shortest total length of the fences required to protect all the cabbage from goats equals 650 meters.
theorem shortest_fence_length : number_of_dotted_lines * triangle_side_length = 650 :=
by
  -- The proof steps are omitted as per instructions.
  sorry

end shortest_fence_length_l1629_162985


namespace hulk_first_jump_more_than_500_l1629_162909

def hulk_jumping_threshold : Prop :=
  ∃ n : ℕ, (3^n > 500) ∧ (∀ m < n, 3^m ≤ 500)

theorem hulk_first_jump_more_than_500 : ∃ n : ℕ, n = 6 ∧ hulk_jumping_threshold :=
  sorry

end hulk_first_jump_more_than_500_l1629_162909


namespace manufacturing_cost_of_shoe_l1629_162998

theorem manufacturing_cost_of_shoe
  (transportation_cost_per_shoe : ℝ)
  (selling_price_per_shoe : ℝ)
  (gain_percentage : ℝ)
  (manufacturing_cost : ℝ)
  (H1 : transportation_cost_per_shoe = 5)
  (H2 : selling_price_per_shoe = 282)
  (H3 : gain_percentage = 0.20)
  (H4 : selling_price_per_shoe = (manufacturing_cost + transportation_cost_per_shoe) * (1 + gain_percentage)) :
  manufacturing_cost = 230 :=
sorry

end manufacturing_cost_of_shoe_l1629_162998


namespace division_remainder_l1629_162975

-- let f(r) = r^15 + r + 1
def f (r : ℝ) : ℝ := r^15 + r + 1

-- let g(r) = r^2 - 1
def g (r : ℝ) : ℝ := r^2 - 1

-- remainder polynomial b(r)
def b (r : ℝ) : ℝ := r + 1

-- Lean statement to prove that polynomial division of f(r) by g(r) 
-- yields the remainder b(r)
theorem division_remainder (r : ℝ) : (f r) % (g r) = b r :=
  sorry

end division_remainder_l1629_162975


namespace find_blue_yarn_count_l1629_162979

def scarves_per_yarn : ℕ := 3
def red_yarn_count : ℕ := 2
def yellow_yarn_count : ℕ := 4
def total_scarves : ℕ := 36

def scarves_from_red_and_yellow : ℕ :=
  red_yarn_count * scarves_per_yarn + yellow_yarn_count * scarves_per_yarn

def blue_scarves : ℕ :=
  total_scarves - scarves_from_red_and_yellow

def blue_yarn_count : ℕ :=
  blue_scarves / scarves_per_yarn

theorem find_blue_yarn_count :
  blue_yarn_count = 6 :=
by 
  sorry

end find_blue_yarn_count_l1629_162979


namespace joseph_cards_l1629_162908

theorem joseph_cards (cards_per_student : ℕ) (students : ℕ) (cards_left : ℕ) 
    (H1 : cards_per_student = 23)
    (H2 : students = 15)
    (H3 : cards_left = 12) 
    : (cards_per_student * students + cards_left = 357) := 
  by
  sorry

end joseph_cards_l1629_162908


namespace line_not_tangent_if_only_one_common_point_l1629_162981

theorem line_not_tangent_if_only_one_common_point (l p : ℝ) :
  (∃ y, y^2 = 2 * p * l) ∧ ¬ (∃ x : ℝ, y = l ∧ y^2 = 2 * p * x) := 
  sorry

end line_not_tangent_if_only_one_common_point_l1629_162981


namespace polynomial_divisibility_l1629_162938

theorem polynomial_divisibility (n : ℕ) (h : 0 < n) : 
  ∃ g : Polynomial ℚ, 
    (Polynomial.X + 1)^(2*n + 1) + Polynomial.X^(n + 2) = g * (Polynomial.X^2 + Polynomial.X + 1) := 
by
  sorry

end polynomial_divisibility_l1629_162938


namespace inequality_problem_l1629_162912

theorem inequality_problem (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : a - d > b - c :=
by
  sorry

end inequality_problem_l1629_162912


namespace find_c_l1629_162940

variable (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)

theorem find_c (h1 : x = 2.5 * y) (h2 : 2 * y = (c / 100) * x) : c = 80 :=
sorry

end find_c_l1629_162940


namespace steps_in_staircase_l1629_162933

theorem steps_in_staircase :
  ∃ n : ℕ, n % 3 = 1 ∧ n % 4 = 3 ∧ n % 5 = 4 ∧ n = 19 :=
by
  sorry

end steps_in_staircase_l1629_162933


namespace perimeter_square_C_l1629_162936

theorem perimeter_square_C (pA pB pC : ℕ) (hA : pA = 16) (hB : pB = 32) (hC : pC = (pA + pB) / 2) : pC = 24 := by
  sorry

end perimeter_square_C_l1629_162936


namespace product_of_last_two_digits_of_divisible_by_6_l1629_162902

-- Definitions
def is_divisible_by_6 (n : ℤ) : Prop := n % 6 = 0
def sum_of_last_two_digits (n : ℤ) (a b : ℤ) : Prop := (n % 100) = 10 * a + b

-- Theorem statement
theorem product_of_last_two_digits_of_divisible_by_6 (x a b : ℤ)
  (h1 : is_divisible_by_6 x)
  (h2 : sum_of_last_two_digits x a b)
  (h3 : a + b = 15) :
  (a * b = 54 ∨ a * b = 56) := 
sorry

end product_of_last_two_digits_of_divisible_by_6_l1629_162902


namespace sugar_in_lollipop_l1629_162923

-- Definitions based on problem conditions
def chocolate_bars := 14
def sugar_per_bar := 10
def total_sugar := 177

-- The theorem to prove
theorem sugar_in_lollipop : total_sugar - (chocolate_bars * sugar_per_bar) = 37 :=
by
  -- we are not providing the proof, hence using sorry
  sorry

end sugar_in_lollipop_l1629_162923


namespace probability_of_all_heads_or_tails_l1629_162943

def num_favorable_outcomes : ℕ := 2

def total_outcomes : ℕ := 2 ^ 5

def probability_all_heads_or_tails : ℚ := num_favorable_outcomes / total_outcomes

theorem probability_of_all_heads_or_tails :
  probability_all_heads_or_tails = 1 / 16 := by
  -- Proof goes here
  sorry

end probability_of_all_heads_or_tails_l1629_162943


namespace find_solutions_l1629_162919

theorem find_solutions (x y : ℕ) : 33 ^ x + 31 = 2 ^ y → (x = 0 ∧ y = 5) ∨ (x = 1 ∧ y = 6) := 
by
  sorry

end find_solutions_l1629_162919


namespace recommended_cooking_time_is_5_minutes_l1629_162958

-- Define the conditions
def time_cooked := 45 -- seconds
def time_remaining := 255 -- seconds

-- Define the total cooking time in seconds
def total_time_seconds := time_cooked + time_remaining

-- Define the conversion from seconds to minutes
def to_minutes (seconds : ℕ) : ℕ := seconds / 60

-- The main theorem to prove
theorem recommended_cooking_time_is_5_minutes :
  to_minutes total_time_seconds = 5 :=
by
  sorry

end recommended_cooking_time_is_5_minutes_l1629_162958


namespace slope_of_line_through_PQ_is_4_l1629_162946

theorem slope_of_line_through_PQ_is_4
  (a : ℕ → ℝ)
  (h_arith_seq : ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d)
  (h_a4 : a 4 = 15)
  (h_a9 : a 9 = 55) :
  let a3 := a 3
  let a8 := a 8
  (a 9 - a 4) / (9 - 4) = 8 → (a 8 - a 3) / (13 - 3) = 4 := by
  sorry

end slope_of_line_through_PQ_is_4_l1629_162946


namespace problem_f8_minus_f4_l1629_162927

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function : ∀ x, f (-x) = -f x
axiom periodic_function : ∀ x, f (x + 5) = f x
axiom f_at_1 : f 1 = 1
axiom f_at_2 : f 2 = 2

theorem problem_f8_minus_f4 : f 8 - f 4 = -1 :=
by sorry

end problem_f8_minus_f4_l1629_162927


namespace ratio_of_blue_to_red_l1629_162974

variable (B : ℕ) -- Number of blue lights

def total_white := 59
def total_colored := total_white - 5
def red_lights := 12
def green_lights := 6

def total_bought := red_lights + green_lights + B

theorem ratio_of_blue_to_red (h : total_bought = total_colored) :
  B / red_lights = 3 :=
by
  sorry

end ratio_of_blue_to_red_l1629_162974


namespace wrapping_paper_needs_l1629_162903

theorem wrapping_paper_needs :
  let first_present := 2
  let second_present := (3 / 4) * first_present
  let third_present := first_present + second_present
  first_present + second_present + third_present = 7 := by
  let first_present := 2
  let second_present := (3 / 4) * first_present
  let third_present := first_present + second_present
  sorry

end wrapping_paper_needs_l1629_162903


namespace equi_partite_complex_number_a_l1629_162983

-- A complex number z = 1 + (a-1)i
def z (a : ℝ) : ℂ := ⟨1, a - 1⟩

-- Definition of an equi-partite complex number
def is_equi_partite (z : ℂ) : Prop := z.re = z.im

-- The theorem to prove
theorem equi_partite_complex_number_a (a : ℝ) : is_equi_partite (z a) ↔ a = 2 := 
by
  sorry

end equi_partite_complex_number_a_l1629_162983


namespace derivative_not_in_second_quadrant_l1629_162944

-- Define the function f(x) and its derivative f'(x)
noncomputable def f (b c x : ℝ) : ℝ := x^2 + b * x + c
noncomputable def f_derivative (x : ℝ) : ℝ := 2 * x - 4

-- Given condition: Axis of symmetry is x = 2
def axis_of_symmetry (b : ℝ) : Prop := b = -4

-- Additional condition: behavior of the derivative and quadrant check
def not_in_second_quadrant (f' : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x < 0 → f' x < 0

-- The main theorem to be proved
theorem derivative_not_in_second_quadrant (b c : ℝ) (h : axis_of_symmetry b) :
  not_in_second_quadrant f_derivative :=
by {
  sorry
}

end derivative_not_in_second_quadrant_l1629_162944


namespace solve_equation_l1629_162921

theorem solve_equation :
  ∃ x : ℝ, (3 * x^2 / (x - 2)) - (4 * x + 11) / 5 + (7 - 9 * x) / (x - 2) + 2 = 0 :=
sorry

end solve_equation_l1629_162921


namespace exponent_problem_l1629_162961

theorem exponent_problem (a : ℝ) (m n : ℕ) (h1 : a ^ m = 3) (h2 : a ^ n = 2) : a ^ (m - 2 * n) = 3 / 4 := by
  sorry

end exponent_problem_l1629_162961


namespace right_angled_triangles_count_l1629_162992

theorem right_angled_triangles_count : 
  ∃ n : ℕ, n = 12 ∧ ∀ (a b c : ℕ), (a = 2016^(1/2)) → (a^2 + b^2 = c^2) →
  (∃ (n k : ℕ), (c - b) = n ∧ (c + b) = k ∧ 2 ∣ n ∧ 2 ∣ k ∧ (n * k = 2016)) :=
by {
  sorry
}

end right_angled_triangles_count_l1629_162992


namespace function_satisfies_conditions_l1629_162931

-- Define the conditions
def f (n : ℕ) : ℕ := n + 1

-- Prove that the function f satisfies the given conditions
theorem function_satisfies_conditions : 
  (f 0 = 1) ∧ (f 2012 = 2013) :=
by
  sorry

end function_satisfies_conditions_l1629_162931


namespace range_of_y_function_l1629_162996

def range_of_function : Set ℝ :=
  {y : ℝ | ∃ (x : ℝ), x ≠ -2 ∧ y = (x^2 + 5*x + 6)/(x+2)}

theorem range_of_y_function :
  range_of_function = {y : ℝ | y ≠ 1} :=
by
  sorry

end range_of_y_function_l1629_162996


namespace simplify_expression_l1629_162918

theorem simplify_expression (x y z : ℝ) (h1 : x ≠ 2) (h2 : y ≠ 3) (h3 : z ≠ 4) :
  (x - 2) / (4 - z) * (y - 3) / (2 - x) * (z - 4) / (3 - y) = -1 :=
by 
sorry

end simplify_expression_l1629_162918


namespace compute_expression_l1629_162935

-- The definition and conditions
def is_nonreal_root_of_unity (ω : ℂ) : Prop := ω ^ 3 = 1 ∧ ω ≠ 1

-- The statement
theorem compute_expression (ω : ℂ) (hω : is_nonreal_root_of_unity ω) : 
  (1 - 2 * ω + 2 * ω ^ 2) ^ 6 + (1 + 2 * ω - 2 * ω ^ 2) ^ 6 = 0 :=
sorry

end compute_expression_l1629_162935


namespace root_interval_range_l1629_162911

theorem root_interval_range (m : ℝ) :
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ 2 ∧ x^3 - 3*x + m = 0) → (-2 ≤ m ∧ m ≤ 2) :=
by
  sorry

end root_interval_range_l1629_162911


namespace pythagorean_triple_fits_l1629_162951

theorem pythagorean_triple_fits 
  (k : ℤ) (n : ℤ) : 
  (∃ k, (n = 5 * k ∨ n = 12 * k ∨ n = 13 * k) ∧ 
      (n = 62 ∨ n = 96 ∨ n = 120 ∨ n = 91 ∨ n = 390)) ↔ 
      (n = 120 ∨ n = 91) := by 
  sorry

end pythagorean_triple_fits_l1629_162951


namespace calculate_expression_l1629_162932

theorem calculate_expression : 
  2 * (3 + 1) * (3^2 + 1) * (3^4 + 1) * (3^8 + 1) * (3^16 + 1) * (3^32 + 1) * (3^64 + 1) + 1 = 3^128 :=
sorry

end calculate_expression_l1629_162932


namespace color_of_last_bead_l1629_162913

-- Define the sequence and length of repeated pattern
def bead_pattern : List String := ["red", "red", "orange", "yellow", "yellow", "yellow", "green", "green", "blue"]
def pattern_length : Nat := bead_pattern.length

-- Define the total number of beads in the bracelet
def total_beads : Nat := 85

-- State the theorem to prove the color of the last bead
theorem color_of_last_bead : bead_pattern.get? ((total_beads - 1) % pattern_length) = some "yellow" :=
by
  sorry

end color_of_last_bead_l1629_162913


namespace Victor_worked_hours_l1629_162984

theorem Victor_worked_hours (h : ℕ) (pay_rate : ℕ) (total_earnings : ℕ) 
  (H1 : pay_rate = 6) 
  (H2 : total_earnings = 60) 
  (H3 : 2 * (pay_rate * h) = total_earnings): 
  h = 5 := 
by 
  sorry

end Victor_worked_hours_l1629_162984


namespace least_possible_b_prime_l1629_162999

theorem least_possible_b_prime :
  ∃ b a : ℕ, Nat.Prime a ∧ Nat.Prime b ∧ 2 * a + b = 180 ∧ a > b ∧ b = 2 :=
by
  sorry

end least_possible_b_prime_l1629_162999


namespace function_increasing_range_l1629_162954

theorem function_increasing_range (a : ℝ) : 
    (∀ x : ℝ, x ≥ 4 → (2*x + 2*(a-1)) > 0) ↔ a ≥ -3 := 
by
  sorry

end function_increasing_range_l1629_162954


namespace set_difference_A_B_l1629_162978

-- Defining the sets A and B
def setA : Set ℝ := { x : ℝ | abs (4 * x - 1) > 9 }
def setB : Set ℝ := { x : ℝ | x >= 0 }

-- The theorem stating the result of set difference A - B
theorem set_difference_A_B : (setA \ setB) = { x : ℝ | x > 5/2 } :=
by
  -- Proof omitted
  sorry

end set_difference_A_B_l1629_162978


namespace find_f_4_l1629_162967

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation : ∀ x : ℝ, f x + 3 * f (1 - x) = 4 * x ^ 2

theorem find_f_4 : f 4 = 5.5 :=
by
  sorry

end find_f_4_l1629_162967


namespace range_of_a_l1629_162963

open Real

namespace PropositionProof

-- Define propositions p and q
def p (a x : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0
def q (x : ℝ) : Prop := (x^2 - x - 6 ≤ 0) ∨ (x^2 + 2*x - 8 > 0)

theorem range_of_a (a : ℝ) (h : a < 0) :
  (¬ ∀ x, ¬ p a x → ∀ x, ¬ q x) ↔ (a ≤ -4 ∨ -2/3 ≤ a ∧ a < 0) :=
sorry

end PropositionProof

end range_of_a_l1629_162963


namespace soldiers_to_add_l1629_162977

theorem soldiers_to_add (N : ℕ) (add : ℕ) 
    (h1 : N % 7 = 2)
    (h2 : N % 12 = 2)
    (h_add : add = 84 - N) :
    add = 82 :=
by
  sorry

end soldiers_to_add_l1629_162977


namespace remainder_of_expression_mod7_l1629_162934

theorem remainder_of_expression_mod7 :
  (7^6 + 8^7 + 9^8) % 7 = 5 :=
by
  sorry

end remainder_of_expression_mod7_l1629_162934


namespace textbook_weight_difference_l1629_162939

theorem textbook_weight_difference :
  let chem_weight := 7.125
  let geom_weight := 0.625
  chem_weight - geom_weight = 6.5 :=
by
  sorry

end textbook_weight_difference_l1629_162939


namespace shots_and_hits_l1629_162926

theorem shots_and_hits (n k : ℕ) (h₀ : 10 < n) (h₁ : n < 20) (h₂ : 5 * k = 3 * (n - k)) : (n = 16) ∧ (k = 6) :=
by {
  -- We state the result that we wish to prove
  sorry
}

end shots_and_hits_l1629_162926


namespace quadratic_inequality_solutions_l1629_162980

theorem quadratic_inequality_solutions (k : ℝ) :
  (0 < k ∧ k < 16) ↔ ∃ x : ℝ, x^2 - 8*x + k < 0 :=
sorry

end quadratic_inequality_solutions_l1629_162980


namespace chair_cost_l1629_162915

theorem chair_cost :
  (∃ (C : ℝ), 3 * C + 50 + 40 = 130 - 4) → 
  (∃ (C : ℝ), C = 12) :=
by
  sorry

end chair_cost_l1629_162915


namespace price_of_second_oil_l1629_162904

open Real

-- Define conditions
def litres_of_first_oil : ℝ := 10
def price_per_litre_first_oil : ℝ := 50
def litres_of_second_oil : ℝ := 5
def total_volume_of_mixture : ℝ := 15
def rate_of_mixture : ℝ := 55.67
def total_cost_of_mixture : ℝ := total_volume_of_mixture * rate_of_mixture

-- Define total cost of the first oil
def total_cost_first_oil : ℝ := litres_of_first_oil * price_per_litre_first_oil

-- Define total cost of the second oil in terms of unknown price P
def total_cost_second_oil (P : ℝ) : ℝ := litres_of_second_oil * P

-- Theorem to prove price per litre of the second oil
theorem price_of_second_oil : ∃ P : ℝ, total_cost_first_oil + (total_cost_second_oil P) = total_cost_of_mixture ∧ P = 67.01 :=
by
  sorry

end price_of_second_oil_l1629_162904


namespace greatest_third_term_of_arithmetic_sequence_l1629_162966

theorem greatest_third_term_of_arithmetic_sequence (a d : ℕ) (h : 4 * a + 6 * d = 46) : a + 2 * d ≤ 15 :=
sorry

end greatest_third_term_of_arithmetic_sequence_l1629_162966


namespace f_inequality_l1629_162907

variables {n1 n2 d : ℕ} (f : ℕ → ℕ → ℕ)

theorem f_inequality (hn1 : n1 > 0) (hn2 : n2 > 0) (hd : d > 0) :
  f (n1 * n2) d ≤ f n1 d + n1 * (f n2 d - 1) :=
sorry

end f_inequality_l1629_162907


namespace sufficient_but_not_necessary_condition_l1629_162982

theorem sufficient_but_not_necessary_condition
  (a : ℝ) :
  (a = 2 → (a - 1) * (a - 2) = 0)
  ∧ (∃ a : ℝ, (a - 1) * (a - 2) = 0 ∧ a ≠ 2) :=
by
  sorry

end sufficient_but_not_necessary_condition_l1629_162982


namespace Mark_less_than_Craig_l1629_162910

-- Definitions for the conditions
def Dave_weight : ℕ := 175
def Dave_bench_press : ℕ := Dave_weight * 3
def Craig_bench_press : ℕ := (20 * Dave_bench_press) / 100
def Mark_bench_press : ℕ := 55

-- The theorem to be proven
theorem Mark_less_than_Craig : Craig_bench_press - Mark_bench_press = 50 :=
by
  sorry

end Mark_less_than_Craig_l1629_162910


namespace number_of_odd_palindromes_l1629_162993

def is_palindrome (n : ℕ) : Prop :=
  let d0 := n % 10
  let d1 := (n / 10) % 10
  let d2 := n / 100
  n < 1000 ∧ n >= 100 ∧ d0 = d2

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

theorem number_of_odd_palindromes : ∃ n : ℕ, is_palindrome n ∧ is_odd n → n = 50 :=
by
  sorry

end number_of_odd_palindromes_l1629_162993


namespace hiker_distance_l1629_162947

variable (s t d : ℝ)
variable (h₁ : (s + 1) * (2 / 3 * t) = d)
variable (h₂ : (s - 1) * (t + 3) = d)

theorem hiker_distance  : d = 6 :=
by
  sorry

end hiker_distance_l1629_162947


namespace direction_vector_l1_l1629_162991

theorem direction_vector_l1
  (m : ℝ)
  (l₁ : ∀ x y : ℝ, (m + 3) * x + 4 * y + 3 * m - 5 = 0)
  (l₂ : ∀ x y : ℝ, 2 * x + (m + 6) * y - 8 = 0)
  (h_perp : ((m + 3) * 2 = -4 * (m + 6)))
  : ∃ v : ℝ × ℝ, v = (-1, -1/2) :=
by
  sorry

end direction_vector_l1_l1629_162991


namespace compound_h_atoms_l1629_162971

theorem compound_h_atoms 
  (weight_H : ℝ) (weight_C : ℝ) (weight_O : ℝ)
  (num_C : ℕ) (num_O : ℕ)
  (total_molecular_weight : ℝ)
  (atomic_weight_H : ℝ) (atomic_weight_C : ℝ) (atomic_weight_O : ℝ)
  (H_w_is_1 : atomic_weight_H = 1)
  (C_w_is_12 : atomic_weight_C = 12)
  (O_w_is_16 : atomic_weight_O = 16)
  (C_atoms_is_1 : num_C = 1)
  (O_atoms_is_3 : num_O = 3)
  (total_mw_is_62 : total_molecular_weight = 62)
  (mw_C : weight_C = num_C * atomic_weight_C)
  (mw_O : weight_O = num_O * atomic_weight_O)
  (mw_CO : weight_C + weight_O = 60)
  (H_weight_contrib : total_molecular_weight - (weight_C + weight_O) = weight_H)
  (H_atoms_calc : weight_H = 2 * atomic_weight_H) :
  2 = 2 :=
by 
  sorry

end compound_h_atoms_l1629_162971


namespace janet_time_to_home_l1629_162900

-- Janet's initial and final positions
def initial_position : ℕ × ℕ := (0, 0) -- (x, y)
def north_blocks : ℕ := 3
def west_multiplier : ℕ := 7
def south_blocks : ℕ := 8
def east_multiplier : ℕ := 2
def speed_blocks_per_minute : ℕ := 2

def west_blocks : ℕ := west_multiplier * north_blocks
def east_blocks : ℕ := east_multiplier * south_blocks

-- Net movement calculations
def net_south_blocks : ℕ := south_blocks - north_blocks
def net_west_blocks : ℕ := west_blocks - east_blocks

-- Time calculation
def total_blocks_to_home : ℕ := net_south_blocks + net_west_blocks
def time_to_home : ℕ := total_blocks_to_home / speed_blocks_per_minute

theorem janet_time_to_home : time_to_home = 5 := by
  -- Proof goes here
  sorry

end janet_time_to_home_l1629_162900


namespace compare_real_numbers_l1629_162914

theorem compare_real_numbers (a b : ℝ) : (a > b) ∨ (a = b) ∨ (a < b) :=
sorry

end compare_real_numbers_l1629_162914


namespace find_y_l1629_162964

theorem find_y (x y : ℝ) (h1 : (100 + 200 + 300 + x) / 4 = 250) (h2 : (300 + 150 + 100 + x + y) / 5 = 200) : y = 50 :=
by
  sorry

end find_y_l1629_162964


namespace trucks_needed_for_coal_transport_l1629_162905

def number_of_trucks (total_coal : ℕ) (capacity_per_truck : ℕ) (x : ℕ) : Prop :=
  capacity_per_truck * x = total_coal

theorem trucks_needed_for_coal_transport :
  number_of_trucks 47500 2500 19 :=
by
  sorry

end trucks_needed_for_coal_transport_l1629_162905


namespace area_shaded_smaller_dodecagon_area_in_circle_l1629_162956

-- Part (a) statement
theorem area_shaded_smaller (dodecagon_area : ℝ) (shaded_area : ℝ) 
  (h : shaded_area = (1 / 12) * dodecagon_area) :
  shaded_area = dodecagon_area / 12 :=
sorry

-- Part (b) statement
theorem dodecagon_area_in_circle (r : ℝ) (A : ℝ) 
  (h : r = 1) (h' : A = (1 / 2) * 12 * r ^ 2 * Real.sin (2 * Real.pi / 12)) :
  A = 3 :=
sorry

end area_shaded_smaller_dodecagon_area_in_circle_l1629_162956


namespace sum_first_100_terms_l1629_162950

def a (n : ℕ) : ℤ := (-1) ^ (n + 1) * n

def S (n : ℕ) : ℤ := Finset.sum (Finset.range n) (λ i => a (i + 1))

theorem sum_first_100_terms : S 100 = -50 := 
by 
  sorry

end sum_first_100_terms_l1629_162950


namespace find_first_day_income_l1629_162972

def income_4 (i2 i3 i4 i5 : ℕ) : ℕ := i2 + i3 + i4 + i5

def total_income_5 (average_income : ℕ) : ℕ := 5 * average_income

def income_1 (total : ℕ) (known : ℕ) : ℕ := total - known

theorem find_first_day_income (i2 i3 i4 i5 a income5 : ℕ) (h1 : income_4 i2 i3 i4 i5 = 1800)
  (h2 : a = 440)
  (h3 : total_income_5 a = income5)
  : income_1 income5 (income_4 i2 i3 i4 i5) = 400 := 
sorry

end find_first_day_income_l1629_162972
