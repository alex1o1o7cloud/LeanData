import Mathlib

namespace NUMINAMATH_GPT_proof_time_lent_to_C_l549_54933

theorem proof_time_lent_to_C :
  let P_B := 5000
  let R := 0.1
  let T_B := 2
  let Total_Interest := 2200
  let P_C := 3000
  let I_B := P_B * R * T_B
  let I_C := Total_Interest - I_B
  let T_C := I_C / (P_C * R)
  T_C = 4 :=
by
  sorry

end NUMINAMATH_GPT_proof_time_lent_to_C_l549_54933


namespace NUMINAMATH_GPT_initial_puppies_count_l549_54925

theorem initial_puppies_count (P : ℕ) (h1 : P - 2 + 3 = 8) : P = 7 :=
sorry

end NUMINAMATH_GPT_initial_puppies_count_l549_54925


namespace NUMINAMATH_GPT_find_function_p_t_additional_hours_l549_54943

variable (p0 : ℝ) (t k : ℝ)

-- Given condition: initial concentration decreased by 1/5 after one hour
axiom filtration_condition_1 : (p0 * ((4 : ℝ) / 5) ^ t = p0 * (Real.exp (-k * t)))
axiom filtration_condition_2 : (p0 * ((4 : ℝ) / 5) = p0 * (Real.exp (-k)))

-- Problem 1: Find the function p(t)
theorem find_function_p_t : ∃ k, ∀ t, p0 * ((4 : ℝ) / 5) ^ t = p0 * (Real.exp (-k * t)) := by
  sorry

-- Problem 2: Find the additional hours of filtration needed
theorem additional_hours (h : ∀ t, p0 * ((4 : ℝ) / 5) ^ t = p0 * (Real.exp (-k * t))) :
  ∀ t, p0 * ((4 : ℝ) / 5) ^ t ≤ (p0 / 1000) → t ≥ 30 := by
  sorry

end NUMINAMATH_GPT_find_function_p_t_additional_hours_l549_54943


namespace NUMINAMATH_GPT_unique_not_in_range_of_g_l549_54958

noncomputable def g (p q r s : ℝ) (x : ℝ) : ℝ := (p * x + q) / (r * x + s)

theorem unique_not_in_range_of_g (p q r s : ℝ) (hps_qr_zero : p * s + q * r = 0) 
  (hpr_rs_zero : p * r + r * s = 0) (hg3 : g p q r s 3 = 3) 
  (hg81 : g p q r s 81 = 81) (h_involution : ∀ x ≠ (-s / r), g p q r s (g p q r s x) = x) :
  ∀ x : ℝ, x ≠ 42 :=
sorry

end NUMINAMATH_GPT_unique_not_in_range_of_g_l549_54958


namespace NUMINAMATH_GPT_perimeter_of_figure_l549_54906

-- Given conditions
def side_length : Nat := 2
def num_horizontal_segments : Nat := 16
def num_vertical_segments : Nat := 10

-- Define a function to calculate the perimeter based on the given conditions
def calculate_perimeter (side_length : Nat) (num_horizontal_segments : Nat) (num_vertical_segments : Nat) : Nat :=
  (num_horizontal_segments * side_length) + (num_vertical_segments * side_length)

-- Statement to be proved
theorem perimeter_of_figure : calculate_perimeter side_length num_horizontal_segments num_vertical_segments = 52 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_perimeter_of_figure_l549_54906


namespace NUMINAMATH_GPT_domain_of_h_l549_54955

def domain_f : Set ℝ := {x | -10 ≤ x ∧ x ≤ 3}

def h_dom := {x | -3 * x ∈ domain_f}

theorem domain_of_h :
  h_dom = {x | x ≥ 10 / 3} :=
by
  sorry

end NUMINAMATH_GPT_domain_of_h_l549_54955


namespace NUMINAMATH_GPT_geometric_seq_sum_four_and_five_l549_54980

noncomputable def geom_seq (a₁ q : ℝ) (n : ℕ) := a₁ * q^(n-1)

theorem geometric_seq_sum_four_and_five :
  (∀ n, geom_seq a₁ q n > 0) →
  geom_seq a₁ q 3 = 4 →
  geom_seq a₁ q 6 = 1 / 2 →
  geom_seq a₁ q 4 + geom_seq a₁ q 5 = 3 :=
by
  sorry

end NUMINAMATH_GPT_geometric_seq_sum_four_and_five_l549_54980


namespace NUMINAMATH_GPT_age_difference_l549_54902

variable (A B C : ℕ)

theorem age_difference (h : A + B = B + C + 12) : A - C = 12 := 
sorry

end NUMINAMATH_GPT_age_difference_l549_54902


namespace NUMINAMATH_GPT_sand_bucket_capacity_l549_54929

theorem sand_bucket_capacity
  (sandbox_depth : ℝ)
  (sandbox_width : ℝ)
  (sandbox_length : ℝ)
  (sand_weight_per_cubic_foot : ℝ)
  (water_per_4_trips : ℝ)
  (water_bottle_ounces : ℝ)
  (water_bottle_cost : ℝ)
  (tony_total_money : ℝ)
  (tony_change : ℝ)
  (tony's_bucket_capacity : ℝ) :
  sandbox_depth = 2 →
  sandbox_width = 4 →
  sandbox_length = 5 →
  sand_weight_per_cubic_foot = 3 →
  water_per_4_trips = 3 →
  water_bottle_ounces = 15 →
  water_bottle_cost = 2 →
  tony_total_money = 10 →
  tony_change = 4 →
  tony's_bucket_capacity = 2 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry -- skipping the proof as per instructions

end NUMINAMATH_GPT_sand_bucket_capacity_l549_54929


namespace NUMINAMATH_GPT_aston_found_pages_l549_54987

-- Given conditions
def pages_per_comic := 25
def initial_untorn_comics := 5
def total_comics_now := 11

-- The number of pages Aston found on the floor
theorem aston_found_pages :
  (total_comics_now - initial_untorn_comics) * pages_per_comic = 150 := 
by
  sorry

end NUMINAMATH_GPT_aston_found_pages_l549_54987


namespace NUMINAMATH_GPT_arithmetic_sequence_ratio_l549_54912

-- Define conditions
def sum_ratios (A_n B_n : ℕ → ℚ) (n : ℕ) : Prop := (A_n n) / (B_n n) = (4 * n + 2) / (5 * n - 5)
def arithmetic_sequences (a_n b_n : ℕ → ℚ) : Prop :=
  ∃ A_n B_n : ℕ → ℚ,
    (∀ n, A_n n = n * (a_n 1) + (n * (n - 1) / 2) * (a_n 2 - a_n 1)) ∧
    (∀ n, B_n n = n * (b_n 1) + (n * (n - 1) / 2) * (b_n 2 - b_n 1)) ∧
    ∀ n, sum_ratios A_n B_n n

-- Theorem to be proven
theorem arithmetic_sequence_ratio
  (a_n b_n : ℕ → ℚ)
  (h : arithmetic_sequences a_n b_n) :
  (a_n 5 + a_n 13) / (b_n 5 + b_n 13) = 7 / 8 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_ratio_l549_54912


namespace NUMINAMATH_GPT_tan_y_l549_54960

theorem tan_y (a b y : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) (hy : 0 < y ∧ y < π / 2)
  (hsiny : Real.sin y = 2 * a * b / (a^2 + b^2)) :
  Real.tan y = 2 * a * b / (a^2 - b^2) :=
sorry

end NUMINAMATH_GPT_tan_y_l549_54960


namespace NUMINAMATH_GPT_comb_eq_l549_54992

theorem comb_eq {n : ℕ} (h : Nat.choose 18 n = Nat.choose 18 2) : n = 2 ∨ n = 16 :=
by
  sorry

end NUMINAMATH_GPT_comb_eq_l549_54992


namespace NUMINAMATH_GPT_orthocentric_tetrahedron_edge_tangent_iff_l549_54967

structure Tetrahedron :=
(V : Type*)
(a b c d e f : V)
(is_orthocentric : Prop)
(has_edge_tangent_sphere : Prop)
(face_equilateral : Prop)
(edges_converging_equal : Prop)

variable (T : Tetrahedron)

noncomputable def edge_tangent_iff_equilateral_edges_converging_equal : Prop :=
T.has_edge_tangent_sphere ↔ (T.face_equilateral ∧ T.edges_converging_equal)

-- Now create the theorem statement
theorem orthocentric_tetrahedron_edge_tangent_iff :
  T.is_orthocentric →
  (∀ a d b e c f p r : ℝ, 
    a + d = b + e ∧ b + e = c + f ∧ a^2 + d^2 = b^2 + e^2 ∧ b^2 + e^2 = c^2 + f^2 ) → 
    edge_tangent_iff_equilateral_edges_converging_equal T := 
by
  intros
  unfold edge_tangent_iff_equilateral_edges_converging_equal
  sorry

end NUMINAMATH_GPT_orthocentric_tetrahedron_edge_tangent_iff_l549_54967


namespace NUMINAMATH_GPT_gcd_pow_minus_one_l549_54963

theorem gcd_pow_minus_one (n m : ℕ) (hn : n = 1030) (hm : m = 1040) :
  Nat.gcd (2^n - 1) (2^m - 1) = 1023 := 
by
  sorry

end NUMINAMATH_GPT_gcd_pow_minus_one_l549_54963


namespace NUMINAMATH_GPT_xiaofang_final_score_l549_54924

def removeHighestLowestScores (scores : List ℕ) : List ℕ :=
  let max_score := scores.maximum.getD 0
  let min_score := scores.minimum.getD 0
  scores.erase max_score |>.erase min_score

def average (scores : List ℕ) : ℚ :=
  (scores.sum : ℚ) / scores.length

theorem xiaofang_final_score :
  let scores := [95, 94, 91, 88, 91, 90, 94, 93, 91, 92]
  average (removeHighestLowestScores scores) = 92 := by
  sorry

end NUMINAMATH_GPT_xiaofang_final_score_l549_54924


namespace NUMINAMATH_GPT_sodas_total_l549_54975

def morning_sodas : ℕ := 77
def afternoon_sodas : ℕ := 19
def total_sodas : ℕ := morning_sodas + afternoon_sodas

theorem sodas_total :
  total_sodas = 96 :=
by
  sorry

end NUMINAMATH_GPT_sodas_total_l549_54975


namespace NUMINAMATH_GPT_derivatives_at_zero_l549_54978

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x)

theorem derivatives_at_zero :
  f 0 = 1 ∧
  deriv f 0 = 0 ∧
  deriv (deriv f) 0 = -4 ∧
  deriv (deriv (deriv f)) 0 = 0 ∧
  deriv (deriv (deriv (deriv f))) 0 = 16 :=
by
  sorry

end NUMINAMATH_GPT_derivatives_at_zero_l549_54978


namespace NUMINAMATH_GPT_cubicsum_eq_neg36_l549_54954

noncomputable def roots (p q r : ℝ) := 
  ∃ l : ℝ, (p^3 - 12) / p = l ∧ (q^3 - 12) / q = l ∧ (r^3 - 12) / r = l

theorem cubicsum_eq_neg36 {p q r : ℝ} (h : p ≠ q ∧ q ≠ r ∧ p ≠ r) 
  (hl : roots p q r) :
  p^3 + q^3 + r^3 = -36 :=
sorry

end NUMINAMATH_GPT_cubicsum_eq_neg36_l549_54954


namespace NUMINAMATH_GPT_thirty_percent_of_x_l549_54948

noncomputable def x : ℝ := 160 / 0.40

theorem thirty_percent_of_x (h : 0.40 * x = 160) : 0.30 * x = 120 :=
sorry

end NUMINAMATH_GPT_thirty_percent_of_x_l549_54948


namespace NUMINAMATH_GPT_find_pictures_museum_l549_54991

-- Define the given conditions
def pictures_zoo : Nat := 24
def pictures_deleted : Nat := 14
def pictures_remaining : Nat := 22

-- Define the target: the number of pictures taken at the museum
def pictures_museum : Nat := 12

-- State the goal to be proved
theorem find_pictures_museum :
  pictures_zoo + pictures_museum - pictures_deleted = pictures_remaining :=
sorry

end NUMINAMATH_GPT_find_pictures_museum_l549_54991


namespace NUMINAMATH_GPT_evaluate_f_l549_54965

def f (x : ℝ) : ℝ := x^5 + 3 * x^3 + 4 * x

theorem evaluate_f (h : f 3 - f (-3) = 672) : True :=
by
  sorry

end NUMINAMATH_GPT_evaluate_f_l549_54965


namespace NUMINAMATH_GPT_find_first_number_l549_54932

theorem find_first_number (x : ℕ) (h1 : x + 35 = 62) : x = 27 := by
  sorry

end NUMINAMATH_GPT_find_first_number_l549_54932


namespace NUMINAMATH_GPT_rower_trip_time_to_Big_Rock_l549_54940

noncomputable def row_trip_time (rowing_speed_in_still_water : ℝ) (river_speed : ℝ) (distance_to_destination : ℝ) : ℝ :=
  let speed_upstream := rowing_speed_in_still_water - river_speed
  let speed_downstream := rowing_speed_in_still_water + river_speed
  let time_upstream := distance_to_destination / speed_upstream
  let time_downstream := distance_to_destination / speed_downstream
  time_upstream + time_downstream

theorem rower_trip_time_to_Big_Rock :
  row_trip_time 7 2 3.2142857142857144 = 1 :=
by
  sorry

end NUMINAMATH_GPT_rower_trip_time_to_Big_Rock_l549_54940


namespace NUMINAMATH_GPT_field_trip_seniors_l549_54931

theorem field_trip_seniors (n : ℕ) 
  (h1 : n < 300) 
  (h2 : n % 17 = 15) 
  (h3 : n % 19 = 12) : 
  n = 202 :=
  sorry

end NUMINAMATH_GPT_field_trip_seniors_l549_54931


namespace NUMINAMATH_GPT_find_a_not_perfect_square_l549_54950

theorem find_a_not_perfect_square :
  {a : ℕ | ∀ n : ℕ, n > 0 → ¬(∃ k : ℕ, n * (n + a) = k * k)} = {1, 2, 4} :=
sorry

end NUMINAMATH_GPT_find_a_not_perfect_square_l549_54950


namespace NUMINAMATH_GPT_smallest_diff_of_YZ_XY_l549_54913

theorem smallest_diff_of_YZ_XY (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 2509) (h4 : a + b > c) (h5 : b + c > a) (h6 : a + c > b) : b - a = 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_smallest_diff_of_YZ_XY_l549_54913


namespace NUMINAMATH_GPT_ratio_volumes_l549_54937

theorem ratio_volumes (hA rA hB rB : ℝ) (hA_def : hA = 30) (rA_def : rA = 15) (hB_def : hB = rA) (rB_def : rB = 2 * hA) :
    (1 / 3 * Real.pi * rA^2 * hA) / (1 / 3 * Real.pi * rB^2 * hB) = 1 / 24 :=
by
  -- skipping the proof
  sorry

end NUMINAMATH_GPT_ratio_volumes_l549_54937


namespace NUMINAMATH_GPT_Zach_scored_more_l549_54911

theorem Zach_scored_more :
  let Zach := 42
  let Ben := 21
  Zach - Ben = 21 :=
by
  let Zach := 42
  let Ben := 21
  exact rfl

end NUMINAMATH_GPT_Zach_scored_more_l549_54911


namespace NUMINAMATH_GPT_jerry_wants_to_raise_average_l549_54900

theorem jerry_wants_to_raise_average :
  let average_first_3_tests := 85
  let total_first_3_tests := 3 * average_first_3_tests
  let score_fourth_test := 93
  let total_after_fourth_test := total_first_3_tests + score_fourth_test
  let new_average := (total_after_fourth_test : ℤ) / 4
  new_average - average_first_3_tests = 2 :=
by
  let average_first_3_tests := 85
  let total_first_3_tests := 3 * average_first_3_tests
  let score_fourth_test := 93
  let total_after_fourth_test := total_first_3_tests + score_fourth_test
  let new_average := (total_after_fourth_test : ℤ) / 4
  have h : new_average - average_first_3_tests = 2 := by
    sorry
  exact h

end NUMINAMATH_GPT_jerry_wants_to_raise_average_l549_54900


namespace NUMINAMATH_GPT_shaded_cells_product_l549_54927

def product_eq (a b c : ℕ) (p : ℕ) : Prop := a * b * c = p

theorem shaded_cells_product :
  ∃ (a₁₁ a₁₂ a₁₃ a₂₁ a₂₂ a₂₃ a₃₁ a₃₂ a₃₃ : ℕ),
    product_eq a₁₁ a₁₂ a₁₃ 12 ∧
    product_eq a₂₁ a₂₂ a₂₃ 112 ∧
    product_eq a₃₁ a₃₂ a₃₃ 216 ∧
    product_eq a₁₁ a₂₁ a₃₁ 12 ∧
    product_eq a₁₂ a₂₂ a₃₂ 12 ∧
    (a₁₁ * a₂₂ * a₃₃ = 3 * 2 * 5) :=
sorry

end NUMINAMATH_GPT_shaded_cells_product_l549_54927


namespace NUMINAMATH_GPT_albert_wins_strategy_l549_54973

theorem albert_wins_strategy (n : ℕ) (h : n = 1999) : 
  ∃ strategy : (ℕ → ℕ), (∀ tokens : ℕ, tokens = n → tokens > 1 → 
  (∃ next_tokens : ℕ, next_tokens < tokens ∧ next_tokens ≥ 1 ∧ next_tokens ≥ tokens / 2) → 
  (∃ k, tokens = 2^k + 1) → strategy n = true) :=
sorry

end NUMINAMATH_GPT_albert_wins_strategy_l549_54973


namespace NUMINAMATH_GPT_num_mittens_per_box_eq_six_l549_54998

theorem num_mittens_per_box_eq_six 
    (num_boxes : ℕ) (scarves_per_box : ℕ) (total_clothing : ℕ)
    (h1 : num_boxes = 4) (h2 : scarves_per_box = 2) (h3 : total_clothing = 32) :
    (total_clothing - num_boxes * scarves_per_box) / num_boxes = 6 :=
by
  sorry

end NUMINAMATH_GPT_num_mittens_per_box_eq_six_l549_54998


namespace NUMINAMATH_GPT_umar_age_is_10_l549_54905

-- Define Ali's age
def Ali_age := 8

-- Define the age difference between Ali and Yusaf
def age_difference := 3

-- Define Yusaf's age based on the conditions
def Yusaf_age := Ali_age - age_difference

-- Define Umar's age which is twice Yusaf's age
def Umar_age := 2 * Yusaf_age

-- Prove that Umar's age is 10
theorem umar_age_is_10 : Umar_age = 10 :=
by
  -- Proof is skipped
  sorry

end NUMINAMATH_GPT_umar_age_is_10_l549_54905


namespace NUMINAMATH_GPT_cat_mouse_positions_after_247_moves_l549_54979

-- Definitions for Positions:
inductive Position
| TopLeft
| TopRight
| BottomRight
| BottomLeft
| TopMiddle
| RightMiddle
| BottomMiddle
| LeftMiddle

open Position

-- Function to calculate position of the cat
def cat_position (n : ℕ) : Position :=
  match n % 4 with
  | 0 => TopLeft
  | 1 => TopRight
  | 2 => BottomRight
  | 3 => BottomLeft
  | _ => TopLeft   -- This case is impossible since n % 4 is in {0, 1, 2, 3}

-- Function to calculate position of the mouse
def mouse_position (n : ℕ) : Position :=
  match n % 8 with
  | 0 => TopMiddle
  | 1 => TopRight
  | 2 => RightMiddle
  | 3 => BottomRight
  | 4 => BottomMiddle
  | 5 => BottomLeft
  | 6 => LeftMiddle
  | 7 => TopLeft
  | _ => TopMiddle -- This case is impossible since n % 8 is in {0, 1, .., 7}

-- Target theorem
theorem cat_mouse_positions_after_247_moves :
  cat_position 247 = BottomRight ∧ mouse_position 247 = LeftMiddle :=
by
  sorry

end NUMINAMATH_GPT_cat_mouse_positions_after_247_moves_l549_54979


namespace NUMINAMATH_GPT_units_digit_of_1583_pow_1246_l549_54990

theorem units_digit_of_1583_pow_1246 : 
  (1583^1246) % 10 = 9 := 
sorry

end NUMINAMATH_GPT_units_digit_of_1583_pow_1246_l549_54990


namespace NUMINAMATH_GPT_store_shelves_l549_54977

theorem store_shelves (initial_books sold_books books_per_shelf : ℕ) 
    (h_initial: initial_books = 27)
    (h_sold: sold_books = 6)
    (h_per_shelf: books_per_shelf = 7) :
    (initial_books - sold_books) / books_per_shelf = 3 := by
  sorry

end NUMINAMATH_GPT_store_shelves_l549_54977


namespace NUMINAMATH_GPT_three_pow_255_mod_7_l549_54999

theorem three_pow_255_mod_7 : 3^255 % 7 = 6 :=
by 
  have h1 : 3^1 % 7 = 3 := by norm_num
  have h2 : 3^2 % 7 = 2 := by norm_num
  have h3 : 3^3 % 7 = 6 := by norm_num
  have h4 : 3^4 % 7 = 4 := by norm_num
  have h5 : 3^5 % 7 = 5 := by norm_num
  have h6 : 3^6 % 7 = 1 := by norm_num
  sorry

end NUMINAMATH_GPT_three_pow_255_mod_7_l549_54999


namespace NUMINAMATH_GPT_units_digit_expression_l549_54944

theorem units_digit_expression :
  ((2 * 21 * 2019 + 2^5) - 4^3) % 10 = 6 := 
sorry

end NUMINAMATH_GPT_units_digit_expression_l549_54944


namespace NUMINAMATH_GPT_minimum_tetrahedra_partition_l549_54956

-- Definitions for the problem conditions
def cube_faces : ℕ := 6
def tetrahedron_faces : ℕ := 4

def face_constraint (cube_faces : ℕ) (tetrahedral_faces : ℕ) : Prop :=
  cube_faces * 2 = 12

def volume_constraint (cube_volume : ℝ) (tetrahedron_volume : ℝ) : Prop :=
  tetrahedron_volume < cube_volume / 6

-- Main proof statement
theorem minimum_tetrahedra_partition (cube_faces tetrahedron_faces : ℕ) (cube_volume tetrahedron_volume : ℝ) :
  face_constraint cube_faces tetrahedron_faces →
  volume_constraint cube_volume tetrahedron_volume →
  5 ≤ cube_faces * 2 / 3 :=
  sorry

end NUMINAMATH_GPT_minimum_tetrahedra_partition_l549_54956


namespace NUMINAMATH_GPT_determine_min_bottles_l549_54917

-- Define the capacities and constraints
def mediumBottleCapacity : ℕ := 80
def largeBottleCapacity : ℕ := 1200
def additionalBottles : ℕ := 5

-- Define the minimum number of medium-sized bottles Jasmine needs to buy
def minimumMediumBottles (mediumCapacity largeCapacity extras : ℕ) : ℕ :=
  let requiredBottles := largeCapacity / mediumCapacity
  requiredBottles

theorem determine_min_bottles :
  minimumMediumBottles mediumBottleCapacity largeBottleCapacity additionalBottles = 15 :=
by
  sorry

end NUMINAMATH_GPT_determine_min_bottles_l549_54917


namespace NUMINAMATH_GPT_orange_pear_difference_l549_54918

theorem orange_pear_difference :
  let O1 := 37
  let O2 := 10
  let O3 := 2 * O2
  let P1 := 30
  let P2 := 3 * P1
  let P3 := P2 + 4
  (O1 + O2 + O3 - (P1 + P2 + P3)) = -147 := 
by
  sorry

end NUMINAMATH_GPT_orange_pear_difference_l549_54918


namespace NUMINAMATH_GPT_prove_A_exactly_2_hits_prove_B_at_least_2_hits_prove_B_exactly_2_more_hits_A_l549_54922

noncomputable def probability_A_exactly_2_hits :=
  let p_A := 1/2
  let trials := 3
  (trials.choose 2) * (p_A ^ 2) * ((1 - p_A) ^ (trials - 2))

noncomputable def probability_B_at_least_2_hits :=
  let p_B := 2/3
  let trials := 3
  (trials.choose 2) * (p_B ^ 2) * ((1 - p_B) ^ (trials - 2)) + (trials.choose 3) * (p_B ^ 3)

noncomputable def probability_B_exactly_2_more_hits_A :=
  let p_A := 1/2
  let p_B := 2/3
  let trials := 3
  let B_2_A_0 := (trials.choose 2) * (p_B ^ 2) * (1 - p_B) * (trials.choose 0) * (p_A ^ 0) * ((1 - p_A) ^ trials)
  let B_3_A_1 := (trials.choose 3) * (p_B ^ 3) * (trials.choose 1) * (p_A ^ 1) * ((1 - p_A) ^ (trials - 1))
  B_2_A_0 + B_3_A_1

theorem prove_A_exactly_2_hits : probability_A_exactly_2_hits = 3/8 := sorry
theorem prove_B_at_least_2_hits : probability_B_at_least_2_hits = 20/27 := sorry
theorem prove_B_exactly_2_more_hits_A : probability_B_exactly_2_more_hits_A = 1/6 := sorry

end NUMINAMATH_GPT_prove_A_exactly_2_hits_prove_B_at_least_2_hits_prove_B_exactly_2_more_hits_A_l549_54922


namespace NUMINAMATH_GPT_problem_statement_l549_54970

noncomputable def g : ℝ → ℝ := sorry

axiom functional_eqn (x y : ℝ) : g (x * g y + 2 * x) = 2 * x * y + g x

def n : ℕ := sorry  -- n is the number of possible values of g(3)
def s : ℝ := sorry  -- s is the sum of all possible values of g(3)

theorem problem_statement : n * s = 0 := sorry

end NUMINAMATH_GPT_problem_statement_l549_54970


namespace NUMINAMATH_GPT_best_solved_completing_square_l549_54910

theorem best_solved_completing_square :
  ∀ (x : ℝ), x^2 - 2*x - 3 = 0 → (x - 1)^2 - 4 = 0 :=
sorry

end NUMINAMATH_GPT_best_solved_completing_square_l549_54910


namespace NUMINAMATH_GPT_remaining_speed_l549_54969

theorem remaining_speed (D : ℝ) (V : ℝ) 
  (h1 : 0.35 * D / 35 + 0.65 * D / V = D / 50) : V = 32.5 :=
by sorry

end NUMINAMATH_GPT_remaining_speed_l549_54969


namespace NUMINAMATH_GPT_space_needed_between_apple_trees_l549_54974

-- Definitions based on conditions
def apple_tree_width : ℕ := 10
def peach_tree_width : ℕ := 12
def space_between_peach_trees : ℕ := 15
def total_space : ℕ := 71
def number_of_apple_trees : ℕ := 2
def number_of_peach_trees : ℕ := 2

-- Lean 4 theorem statement
theorem space_needed_between_apple_trees :
  (total_space 
   - (number_of_peach_trees * peach_tree_width + space_between_peach_trees))
  - (number_of_apple_trees * apple_tree_width) 
  = 12 := by
  sorry

end NUMINAMATH_GPT_space_needed_between_apple_trees_l549_54974


namespace NUMINAMATH_GPT_coffee_amount_l549_54903

theorem coffee_amount (total_mass : ℕ) (coffee_ratio : ℕ) (milk_ratio : ℕ) (h_total_mass : total_mass = 4400) (h_coffee_ratio : coffee_ratio = 2) (h_milk_ratio : milk_ratio = 9) : 
  total_mass * coffee_ratio / (coffee_ratio + milk_ratio) = 800 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_coffee_amount_l549_54903


namespace NUMINAMATH_GPT_third_cyclist_speed_l549_54993

theorem third_cyclist_speed (s1 s3 : ℝ) :
  (∃ s1 s3 : ℝ,
    (∀ t : ℝ, t > 0 → (s1 > s3) ∧ (20 = abs (10 * t - s1 * t)) ∧ (5 = abs (s1 * t - s3 * t)) ∧ (s1 ≥ 10))) →
  (s3 = 25 ∨ s3 = 5) :=
by sorry

end NUMINAMATH_GPT_third_cyclist_speed_l549_54993


namespace NUMINAMATH_GPT_total_distance_between_alice_bob_l549_54934

-- Define the constants for Alice's and Bob's speeds and the time duration in terms of conditions.
def alice_speed := 1 / 12  -- miles per minute
def bob_speed := 3 / 20    -- miles per minute
def time_duration := 120   -- minutes

-- Statement: Prove that the total distance between Alice and Bob after 2 hours is 28 miles.
theorem total_distance_between_alice_bob : (alice_speed * time_duration) + (bob_speed * time_duration) = 28 :=
by
  sorry

end NUMINAMATH_GPT_total_distance_between_alice_bob_l549_54934


namespace NUMINAMATH_GPT_units_digit_expression_l549_54962

lemma units_digit_2_pow_2023 : (2 ^ 2023) % 10 = 8 := sorry

lemma units_digit_5_pow_2024 : (5 ^ 2024) % 10 = 5 := sorry

lemma units_digit_11_pow_2025 : (11 ^ 2025) % 10 = 1 := sorry

theorem units_digit_expression : ((2 ^ 2023) * (5 ^ 2024) * (11 ^ 2025)) % 10 = 0 :=
by 
  have h1 := units_digit_2_pow_2023
  have h2 := units_digit_5_pow_2024
  have h3 := units_digit_11_pow_2025
  sorry

end NUMINAMATH_GPT_units_digit_expression_l549_54962


namespace NUMINAMATH_GPT_combined_population_port_perry_lazy_harbor_l549_54988

theorem combined_population_port_perry_lazy_harbor 
  (PP LH W : ℕ)
  (h1 : PP = 7 * W)
  (h2 : PP = LH + 800)
  (h3 : W = 900) :
  PP + LH = 11800 :=
by
  sorry

end NUMINAMATH_GPT_combined_population_port_perry_lazy_harbor_l549_54988


namespace NUMINAMATH_GPT_largest_inscribed_parabola_area_l549_54914

noncomputable def maximum_parabolic_area_in_cone (r l : ℝ) : ℝ :=
  (l * r) / 2 * Real.sqrt 3

theorem largest_inscribed_parabola_area (r l : ℝ) : 
  ∃ t : ℝ, t = maximum_parabolic_area_in_cone r l :=
by
  let t_max := (l * r) / 2 * Real.sqrt 3
  use t_max
  sorry

end NUMINAMATH_GPT_largest_inscribed_parabola_area_l549_54914


namespace NUMINAMATH_GPT_min_value_x_y_l549_54995

theorem min_value_x_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 9 / (x + 1) + 1 / (y + 1) = 1) :
  x + y ≥ 14 :=
sorry

end NUMINAMATH_GPT_min_value_x_y_l549_54995


namespace NUMINAMATH_GPT_total_spent_is_correct_l549_54942

def cost_of_lunch : ℝ := 60.50
def tip_percentage : ℝ := 0.20
def tip_amount : ℝ := cost_of_lunch * tip_percentage
def total_amount_spent : ℝ := cost_of_lunch + tip_amount

theorem total_spent_is_correct : total_amount_spent = 72.60 := by
  -- placeholder for the proof
  sorry

end NUMINAMATH_GPT_total_spent_is_correct_l549_54942


namespace NUMINAMATH_GPT_am_gm_inequality_l549_54941

theorem am_gm_inequality (a b c d : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) (h_prod : a * b * c * d = 1) : 
  a^2 + b^2 + c^2 + d^2 + a * b + a * c + a * d + b * c + b * d + c * d ≥ 10 :=
by
  sorry

end NUMINAMATH_GPT_am_gm_inequality_l549_54941


namespace NUMINAMATH_GPT_rectangle_area_l549_54989

theorem rectangle_area (l w : ℝ) (h₁ : (2 * l + 2 * w) = 46) (h₂ : (l^2 + w^2) = 289) : l * w = 120 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_l549_54989


namespace NUMINAMATH_GPT_parallel_vectors_solution_l549_54964

noncomputable def vector_a : (ℝ × ℝ) := (1, 2)
noncomputable def vector_b (x : ℝ) : (ℝ × ℝ) := (x, -4)

def vectors_parallel (a b : (ℝ × ℝ)) : Prop := ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2

theorem parallel_vectors_solution (x : ℝ) (h : vectors_parallel vector_a (vector_b x)) : x = -2 :=
sorry

end NUMINAMATH_GPT_parallel_vectors_solution_l549_54964


namespace NUMINAMATH_GPT_range_of_a_l549_54936

noncomputable def f (x a : ℝ) : ℝ :=
  (1 / 2) * x^2 - 2 * x + a * Real.log x

theorem range_of_a (a : ℝ) :
  (0 < a ∧ 4 - 4 * a > 0) ↔ (0 < a ∧ a < 1) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l549_54936


namespace NUMINAMATH_GPT_attendance_difference_is_85_l549_54983

def saturday_attendance : ℕ := 80
def monday_attendance : ℕ := saturday_attendance - 20
def wednesday_attendance : ℕ := monday_attendance + 50
def friday_attendance : ℕ := saturday_attendance + monday_attendance
def thursday_attendance : ℕ := 45
def expected_audience : ℕ := 350

def total_attendance : ℕ := 
  saturday_attendance + 
  monday_attendance + 
  wednesday_attendance + 
  friday_attendance + 
  thursday_attendance

def more_people_attended_than_expected : ℕ :=
  total_attendance - expected_audience

theorem attendance_difference_is_85 : more_people_attended_than_expected = 85 := 
by
  unfold more_people_attended_than_expected
  unfold total_attendance
  unfold saturday_attendance
  unfold monday_attendance
  unfold wednesday_attendance
  unfold friday_attendance
  unfold thursday_attendance
  unfold expected_audience
  exact sorry

end NUMINAMATH_GPT_attendance_difference_is_85_l549_54983


namespace NUMINAMATH_GPT_complement_union_A_B_l549_54901

def U : Set ℕ := {x | x < 6}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {2, 4}

theorem complement_union_A_B :
  (U \ (A ∪ B)) = {0, 5} := by
  sorry

end NUMINAMATH_GPT_complement_union_A_B_l549_54901


namespace NUMINAMATH_GPT_ratio_of_term_to_difference_l549_54928

def arithmetic_progression_sum (n a d : ℕ) : ℕ := (n * (2 * a + (n - 1) * d)) / 2

theorem ratio_of_term_to_difference (a d : ℕ) 
  (h1: arithmetic_progression_sum 7 a d = arithmetic_progression_sum 3 a d + 20)
  (h2 : d ≠ 0) : a / d = 1 / 2 := 
by 
  sorry

end NUMINAMATH_GPT_ratio_of_term_to_difference_l549_54928


namespace NUMINAMATH_GPT_quadratic_same_roots_abs_l549_54985

theorem quadratic_same_roots_abs (d e : ℤ) : 
  (∀ x : ℤ, |x - 8| = 3 ↔ x = 11 ∨ x = 5) →
  (∀ x : ℤ, x^2 + d * x + e = 0 ↔ x = 11 ∨ x = 5) →
  (d, e) = (-16, 55) :=
by
  intro h₁ h₂
  have h₃ : ∀ x : ℤ, x^2 - 16 * x + 55 = 0 ↔ x = 11 ∨ x = 5 := sorry
  sorry

end NUMINAMATH_GPT_quadratic_same_roots_abs_l549_54985


namespace NUMINAMATH_GPT_total_candidates_l549_54986

def average_marks_all_candidates : ℕ := 35
def average_marks_passed_candidates : ℕ := 39
def average_marks_failed_candidates : ℕ := 15
def passed_candidates : ℕ := 100

theorem total_candidates (T : ℕ) (F : ℕ) 
  (h1 : 35 * T = 39 * passed_candidates + 15 * F)
  (h2 : T = passed_candidates + F) : T = 120 := 
  sorry

end NUMINAMATH_GPT_total_candidates_l549_54986


namespace NUMINAMATH_GPT_books_written_l549_54957

variable (Z F : ℕ)

theorem books_written (h1 : Z = 60) (h2 : Z = 4 * F) : Z + F = 75 := by
  sorry

end NUMINAMATH_GPT_books_written_l549_54957


namespace NUMINAMATH_GPT_amount_b_l549_54915

-- Definitions of the conditions
variables (a b : ℚ) 

def condition1 : Prop := a + b = 1210
def condition2 : Prop := (2 / 3) * a = (1 / 2) * b

-- The theorem to prove
theorem amount_b (h₁ : condition1 a b) (h₂ : condition2 a b) : b = 691.43 :=
sorry

end NUMINAMATH_GPT_amount_b_l549_54915


namespace NUMINAMATH_GPT_sum_digits_l549_54904

def repeat_pattern (d: ℕ) (n: ℕ) : ℕ :=
  let pattern := if d = 404 then 404 else if d = 707 then 707 else 0
  pattern * 10^(n / 3)

def N1 := repeat_pattern 404 101
def N2 := repeat_pattern 707 101
def P := N1 * N2

def thousands_digit (n: ℕ) : ℕ :=
  (n / 1000) % 10

def units_digit (n: ℕ) : ℕ :=
  n % 10

theorem sum_digits : thousands_digit P + units_digit P = 10 := by
  sorry

end NUMINAMATH_GPT_sum_digits_l549_54904


namespace NUMINAMATH_GPT_number_of_integers_satisfying_inequalities_l549_54997

theorem number_of_integers_satisfying_inequalities :
  ∃ (count : ℕ), count = 3 ∧
    (∀ x : ℤ, -4 * x ≥ x + 10 → -3 * x ≤ 15 → -5 * x ≥ 3 * x + 24 → 2 * x ≤ 18 →
      x = -5 ∨ x = -4 ∨ x = -3) :=
sorry

end NUMINAMATH_GPT_number_of_integers_satisfying_inequalities_l549_54997


namespace NUMINAMATH_GPT_simplify_expression_l549_54926

theorem simplify_expression (a b : ℚ) (ha : a = -1) (hb : b = 1/4) : 
  (a + 2 * b) ^ 2 + (a + 2 * b) * (a - 2 * b) = 1 := 
by 
  sorry

end NUMINAMATH_GPT_simplify_expression_l549_54926


namespace NUMINAMATH_GPT_eleven_not_sum_of_two_primes_l549_54921

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem eleven_not_sum_of_two_primes :
  ¬ ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p + q = 11 :=
by sorry

end NUMINAMATH_GPT_eleven_not_sum_of_two_primes_l549_54921


namespace NUMINAMATH_GPT_minimum_time_to_finish_route_l549_54976

-- Step (a): Defining conditions and necessary terms
def points : Nat := 12
def segments_between_points : ℕ := 17
def time_per_segment : ℕ := 10 -- in minutes
def total_time_in_minutes : ℕ := segments_between_points * time_per_segment -- Total time in minutes

-- Step (c): Proving the question == answer given conditions
theorem minimum_time_to_finish_route (K : ℕ) : K = 4 :=
by
  have time_in_hours : ℕ := total_time_in_minutes / 60
  have minimum_time : ℕ := 4
  sorry -- proof needed

end NUMINAMATH_GPT_minimum_time_to_finish_route_l549_54976


namespace NUMINAMATH_GPT_find_divisor_of_115_l549_54966

theorem find_divisor_of_115 (x : ℤ) (N : ℤ)
  (hN : N = 115)
  (h1 : N % 38 = 1)
  (h2 : N % x = 1) :
  x = 57 :=
by
  sorry

end NUMINAMATH_GPT_find_divisor_of_115_l549_54966


namespace NUMINAMATH_GPT_flowers_per_bug_l549_54981

theorem flowers_per_bug (bugs : ℝ) (flowers : ℝ) (h_bugs : bugs = 2.0) (h_flowers : flowers = 3.0) :
  flowers / bugs = 1.5 :=
by
  sorry

end NUMINAMATH_GPT_flowers_per_bug_l549_54981


namespace NUMINAMATH_GPT_problem_1_solution_set_problem_2_range_of_a_l549_54907

-- Define the function f(x)
def f (x a : ℝ) := |2 * x - a| + |x - 1|

-- Problem 1: Solution set of the inequality f(x) ≥ 2 when a = 3
theorem problem_1_solution_set :
  { x : ℝ | f x 3 ≥ 2 } = { x : ℝ | x ≤ 2/3 ∨ x ≥ 2 } :=
sorry

-- Problem 2: Range of a such that f(x) ≥ 5 - x for all x ∈ ℝ
theorem problem_2_range_of_a (a : ℝ) :
  (∀ x : ℝ, f x a ≥ 5 - x) ↔ a ≥ 6 :=
sorry

end NUMINAMATH_GPT_problem_1_solution_set_problem_2_range_of_a_l549_54907


namespace NUMINAMATH_GPT_geometric_sequence_ratio_l549_54959
-- Lean 4 Code

noncomputable def a_n (a : ℝ) (q : ℝ) (n : ℕ) : ℝ := a * q ^ n

theorem geometric_sequence_ratio
  (a : ℝ)
  (q : ℝ)
  (h_pos : a > 0)
  (h_q_neq_1 : q ≠ 1)
  (h_arith_seq : 2 * a_n a q 4 = a_n a q 2 + a_n a q 5)
  : (a_n a q 2 + a_n a q 3) / (a_n a q 3 + a_n a q 4) = (Real.sqrt 5 - 1) / 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_geometric_sequence_ratio_l549_54959


namespace NUMINAMATH_GPT_find_A_l549_54968

def is_valid_A (A : ℕ) : Prop :=
  A = 1 ∨ A = 2 ∨ A = 4 ∨ A = 7 ∨ A = 9

def number (A : ℕ) : ℕ :=
  3 * 100000 + 0 * 10000 + 5 * 1000 + 2 * 100 + 0 * 10 + A

theorem find_A (A : ℕ) (h_valid_A : is_valid_A A) : A = 1 ↔ Nat.Prime (number A) :=
by
  sorry

end NUMINAMATH_GPT_find_A_l549_54968


namespace NUMINAMATH_GPT_incorrect_option_A_l549_54920

theorem incorrect_option_A (x y : ℝ) :
  ¬(5 * x + y / 2 = (5 * x + y) / 2) :=
by sorry

end NUMINAMATH_GPT_incorrect_option_A_l549_54920


namespace NUMINAMATH_GPT_find_m_l549_54971

theorem find_m (S : ℕ → ℝ) (m : ℕ) (h1 : S m = -2) (h2 : S (m+1) = 0) (h3 : S (m+2) = 3) : m = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l549_54971


namespace NUMINAMATH_GPT_distance_EC_l549_54953

-- Define the points and given distances as conditions
structure Points :=
  (A B C D E : Type)

-- Distances between points
variables {Points : Type}
variables (dAB dBC dCD dDE dEA dEC : ℝ)
variables [Nonempty Points]

-- Specify conditions: distances in kilometers
def distances_given (dAB dBC dCD dDE dEA : ℝ) : Prop :=
  dAB = 30 ∧ dBC = 80 ∧ dCD = 236 ∧ dDE = 86 ∧ dEA = 40

-- Main theorem: prove that the distance from E to C is 63.4 km
theorem distance_EC (h : distances_given 30 80 236 86 40) : dEC = 63.4 :=
sorry

end NUMINAMATH_GPT_distance_EC_l549_54953


namespace NUMINAMATH_GPT_no_pos_int_lt_2000_7_times_digits_sum_l549_54930

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem no_pos_int_lt_2000_7_times_digits_sum :
  ∀ n : ℕ, n < 2000 → n = 7 * sum_of_digits n → False :=
by
  intros n h1 h2
  sorry

end NUMINAMATH_GPT_no_pos_int_lt_2000_7_times_digits_sum_l549_54930


namespace NUMINAMATH_GPT_lee_charge_per_action_figure_l549_54994

def cost_of_sneakers : ℕ := 90
def amount_saved : ℕ := 15
def action_figures_sold : ℕ := 10
def amount_left_after_purchase : ℕ := 25
def amount_charged_per_action_figure : ℕ := 10

theorem lee_charge_per_action_figure :
  (cost_of_sneakers - amount_saved + amount_left_after_purchase = 
  action_figures_sold * amount_charged_per_action_figure) :=
by
  -- The proof steps will go here, but they are not required in the statement.
  sorry

end NUMINAMATH_GPT_lee_charge_per_action_figure_l549_54994


namespace NUMINAMATH_GPT_figure8_squares_figure12_perimeter_no_figure_C_figure29_figureD_ratio_l549_54947

-- Given conditions:
def initial_squares : ℕ := 3
def initial_perimeter : ℕ := 8
def squares_per_step : ℕ := 2
def perimeter_per_step : ℕ := 4

-- Statement proving Figure 8 has 17 squares
theorem figure8_squares : 3 + 2 * (8 - 1) = 17 := by sorry

-- Statement proving Figure 12 has a perimeter of 52 cm
theorem figure12_perimeter : 8 + 4 * (12 - 1) = 52 := by sorry

-- Statement proving no positive integer C yields perimeter of 38 cm
theorem no_figure_C : ¬∃ C : ℕ, 8 + 4 * (C - 1) = 38 := by sorry
  
-- Statement proving closest D giving the ratio for perimeter between Figure 29 and Figure D
theorem figure29_figureD_ratio : (8 + 4 * (29 - 1)) * 11 = 4 * (8 + 4 * (81 - 1)) := by sorry

end NUMINAMATH_GPT_figure8_squares_figure12_perimeter_no_figure_C_figure29_figureD_ratio_l549_54947


namespace NUMINAMATH_GPT_polynomial_rewrite_l549_54909

theorem polynomial_rewrite (d : ℤ) (h : d ≠ 0) :
  let a := 20
  let b := 18
  let c := 18
  let e := 8
  (16 * d + 15 + 18 * d^2 + 5 * d^3) + (4 * d + 3 + 3 * d^3) = a * d + b + c * d^2 + e * d^3 ∧ a + b + c + e = 64 := 
by
  sorry

end NUMINAMATH_GPT_polynomial_rewrite_l549_54909


namespace NUMINAMATH_GPT_arc_length_problem_l549_54923

noncomputable def arc_length (r : ℝ) (theta : ℝ) : ℝ :=
  r * theta

theorem arc_length_problem :
  ∀ (r : ℝ) (theta_deg : ℝ), r = 1 ∧ theta_deg = 150 → 
  arc_length r (theta_deg * (Real.pi / 180)) = (5 * Real.pi / 6) :=
by
  intro r theta_deg h
  sorry

end NUMINAMATH_GPT_arc_length_problem_l549_54923


namespace NUMINAMATH_GPT_not_power_of_two_l549_54945

theorem not_power_of_two (m n : ℕ) (hm : m > 0) (hn : n > 0) : 
  ¬ ∃ k : ℕ, (36 * m + n) * (m + 36 * n) = 2 ^ k :=
sorry

end NUMINAMATH_GPT_not_power_of_two_l549_54945


namespace NUMINAMATH_GPT_friend_spent_l549_54908

theorem friend_spent (x you friend total: ℝ) (h1 : total = you + friend) (h2 : friend = you + 3) (h3 : total = 11) : friend = 7 := by
  sorry

end NUMINAMATH_GPT_friend_spent_l549_54908


namespace NUMINAMATH_GPT_percentage_reduction_is_58_perc_l549_54982

-- Define the conditions
def initial_price (P : ℝ) : ℝ := P
def discount_price (P : ℝ) : ℝ := 0.7 * P
def increased_price (P : ℝ) : ℝ := 1.2 * (discount_price P)
def clearance_price (P : ℝ) : ℝ := 0.5 * (increased_price P)

-- The statement of the proof problem
theorem percentage_reduction_is_58_perc (P : ℝ) (h : P > 0) :
  (1 - (clearance_price P / initial_price P)) * 100 = 58 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_percentage_reduction_is_58_perc_l549_54982


namespace NUMINAMATH_GPT_allocation_ways_l549_54946

/-- Defining the number of different balls and boxes -/
def num_balls : ℕ := 4
def num_boxes : ℕ := 3

/-- The theorem asserting the number of ways to place the balls into the boxes -/
theorem allocation_ways : (num_boxes ^ num_balls) = 81 := by
  sorry

end NUMINAMATH_GPT_allocation_ways_l549_54946


namespace NUMINAMATH_GPT_Kyle_fish_count_l549_54961

def Carla_fish := 8
def Total_fish := 36
def Kyle_fish := (Total_fish - Carla_fish) / 2

theorem Kyle_fish_count : Kyle_fish = 14 :=
by
  -- This proof will be provided later
  sorry

end NUMINAMATH_GPT_Kyle_fish_count_l549_54961


namespace NUMINAMATH_GPT_students_not_in_any_activity_l549_54996

def total_students : ℕ := 1500
def students_chorus : ℕ := 420
def students_band : ℕ := 780
def students_chorus_and_band : ℕ := 150
def students_drama : ℕ := 300
def students_drama_and_other : ℕ := 50

theorem students_not_in_any_activity :
  total_students - ((students_chorus + students_band - students_chorus_and_band) + (students_drama - students_drama_and_other)) = 200 :=
by
  sorry

end NUMINAMATH_GPT_students_not_in_any_activity_l549_54996


namespace NUMINAMATH_GPT_john_plays_periods_l549_54919

theorem john_plays_periods
  (PointsPer4Minutes : ℕ := 7)
  (PeriodDurationMinutes : ℕ := 12)
  (TotalPoints : ℕ := 42) :
  (TotalPoints / PointsPer4Minutes) / (PeriodDurationMinutes / 4) = 2 := by
  sorry

end NUMINAMATH_GPT_john_plays_periods_l549_54919


namespace NUMINAMATH_GPT_product_of_binomials_l549_54951

theorem product_of_binomials :
  (2*x^2 + 3*x - 4) * (x + 6) = 2*x^3 + 15*x^2 + 14*x - 24 :=
by {
  sorry
}

end NUMINAMATH_GPT_product_of_binomials_l549_54951


namespace NUMINAMATH_GPT_roots_magnitudes_less_than_one_l549_54935

theorem roots_magnitudes_less_than_one
  (A B C D : ℝ)
  (h1 : ∀ (r : ℝ), |r| < 1 → r ≠ 0 → (r ^ 2 + A * r + B = 0))
  (h2 : ∀ (r : ℝ), |r| < 1 → r ≠ 0 → (r ^ 2 + C * r + D = 0)) :
  ∀ (r : ℝ), |r| < 1 → r ≠ 0 → (r ^ 2 + (1 / 2 * (A + C)) * r + (1 / 2 * (B + D)) = 0) :=
by
  sorry

end NUMINAMATH_GPT_roots_magnitudes_less_than_one_l549_54935


namespace NUMINAMATH_GPT_trapezoid_area_eq_15_l549_54939

theorem trapezoid_area_eq_15 :
  let line1 := fun (x : ℝ) => 2 * x
  let line2 := fun (x : ℝ) => 8
  let line3 := fun (x : ℝ) => 2
  let y_axis := fun (y : ℝ) => 0
  let intersection_points := [
    (4, 8),   -- Intersection of line1 and line2
    (1, 2),   -- Intersection of line1 and line3
    (0, 8),   -- Intersection of y_axis and line2
    (0, 2)    -- Intersection of y_axis and line3
  ]
  let base1 := (4 - 0 : ℝ)  -- Length of top base 
  let base2 := (1 - 0 : ℝ)  -- Length of bottom base
  let height := (8 - 2 : ℝ) -- Vertical distance between line2 and line3
  (0.5 * (base1 + base2) * height = 15.0) := by
  sorry

end NUMINAMATH_GPT_trapezoid_area_eq_15_l549_54939


namespace NUMINAMATH_GPT_smallest_n_not_divisible_by_10_l549_54916

theorem smallest_n_not_divisible_by_10 :
  ∃ n : ℕ, n > 2016 ∧ (1^n + 2^n + 3^n + 4^n) % 10 ≠ 0 ∧ n = 2020 := by
  sorry

end NUMINAMATH_GPT_smallest_n_not_divisible_by_10_l549_54916


namespace NUMINAMATH_GPT_range_F_l549_54938

-- Define the function and its critical points
def F (x : ℝ) : ℝ := |2 * x + 4| - |x - 2|

theorem range_F : ∀ y : ℝ, y ∈ Set.range F ↔ -4 ≤ y := by
  sorry

end NUMINAMATH_GPT_range_F_l549_54938


namespace NUMINAMATH_GPT_compute_pow_l549_54949

theorem compute_pow (i : ℂ) (h : i^2 = -1) : (1 - i)^6 = 8 * i := by
  sorry

end NUMINAMATH_GPT_compute_pow_l549_54949


namespace NUMINAMATH_GPT_min_value_expression_l549_54952

theorem min_value_expression : ∃ x y : ℝ, (xy-2)^2 + (x^2 + y^2) = 4 :=
by
  sorry

end NUMINAMATH_GPT_min_value_expression_l549_54952


namespace NUMINAMATH_GPT_smallest_unrepresentable_integer_l549_54972

theorem smallest_unrepresentable_integer :
  ∃ n : ℕ, (∀ a b c d : ℕ, (a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) → 
  n ≠ (2^a - 2^b) / (2^c - 2^d)) ∧ n = 11 :=
by
  sorry

end NUMINAMATH_GPT_smallest_unrepresentable_integer_l549_54972


namespace NUMINAMATH_GPT_Tod_speed_is_25_mph_l549_54984

-- Definitions of the conditions
def miles_north : ℕ := 55
def miles_west : ℕ := 95
def hours_driven : ℕ := 6

-- The total distance travelled
def total_distance : ℕ := miles_north + miles_west

-- The speed calculation, dividing total distance by hours driven
def speed : ℕ := total_distance / hours_driven

-- The theorem to prove
theorem Tod_speed_is_25_mph : speed = 25 :=
by
  -- Proof of the theorem will be filled here, but for now using sorry
  sorry

end NUMINAMATH_GPT_Tod_speed_is_25_mph_l549_54984
