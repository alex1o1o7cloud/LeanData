import Mathlib

namespace NUMINAMATH_GPT_problem_l1493_149379

def f (x : ℝ) : ℝ := sorry -- We assume f is defined as per the given condition but do not provide an implementation.

theorem problem (h : ∀ x : ℝ, f (Real.cos x) = Real.cos (2 * x)) :
  f (Real.sin (Real.pi / 12)) = - (Real.sqrt 3) / 2 :=
by
  sorry -- The proof is omitted

end NUMINAMATH_GPT_problem_l1493_149379


namespace NUMINAMATH_GPT_rogers_parents_paid_percentage_l1493_149357

variables 
  (house_cost : ℝ)
  (down_payment_percentage : ℝ)
  (remaining_balance_owed : ℝ)
  (down_payment : ℝ := down_payment_percentage * house_cost)
  (remaining_balance_after_down : ℝ := house_cost - down_payment)
  (parents_payment : ℝ := remaining_balance_after_down - remaining_balance_owed)
  (percentage_paid_by_parents : ℝ := (parents_payment / remaining_balance_after_down) * 100)

theorem rogers_parents_paid_percentage
  (h1 : house_cost = 100000)
  (h2 : down_payment_percentage = 0.20)
  (h3 : remaining_balance_owed = 56000) :
  percentage_paid_by_parents = 30 :=
sorry

end NUMINAMATH_GPT_rogers_parents_paid_percentage_l1493_149357


namespace NUMINAMATH_GPT_sequence_formula_l1493_149340

theorem sequence_formula (a : ℕ → ℝ) (h₁ : a 1 = 3) (h₂ : ∀ n : ℕ, n > 0 → a (n + 1) = 4 * a n + 3) :
  ∀ n : ℕ, n > 0 → a n = 4 ^ n - 1 :=
by
  sorry

end NUMINAMATH_GPT_sequence_formula_l1493_149340


namespace NUMINAMATH_GPT_percent_non_union_women_l1493_149343

-- Definitions used in the conditions:
def total_employees := 100
def percent_men := 50 / 100
def percent_union := 60 / 100
def percent_union_men := 70 / 100

-- Calculate intermediate values
def num_men := total_employees * percent_men
def num_union := total_employees * percent_union
def num_union_men := num_union * percent_union_men
def num_non_union := total_employees - num_union
def num_non_union_men := num_men - num_union_men
def num_non_union_women := num_non_union - num_non_union_men

-- Statement of the problem in Lean
theorem percent_non_union_women : (num_non_union_women / num_non_union) * 100 = 80 := 
by {
  sorry
}

end NUMINAMATH_GPT_percent_non_union_women_l1493_149343


namespace NUMINAMATH_GPT_area_of_gray_region_l1493_149327

open Real

-- Define the circles and the radii.
def circleC_center : Prod Real Real := (5, 5)
def radiusC : Real := 5

def circleD_center : Prod Real Real := (15, 5)
def radiusD : Real := 5

-- The main theorem stating the area of the gray region bound by the circles and the x-axis.
theorem area_of_gray_region : 
  let area_rectangle := (10:Real) * (5:Real)
  let area_sectors := (2:Real) * ((1/4) * (5:Real)^2 * π)
  area_rectangle - area_sectors = 50 - 12.5 * π :=
by
  sorry

end NUMINAMATH_GPT_area_of_gray_region_l1493_149327


namespace NUMINAMATH_GPT_enlarged_decal_height_l1493_149328

theorem enlarged_decal_height (original_width original_height new_width : ℕ)
  (original_width_eq : original_width = 3)
  (original_height_eq : original_height = 2)
  (new_width_eq : new_width = 15)
  (proportions_consistent : ∀ h : ℕ, new_width * original_height = original_width * h) :
  ∃ new_height, new_height = 10 :=
by sorry

end NUMINAMATH_GPT_enlarged_decal_height_l1493_149328


namespace NUMINAMATH_GPT_original_price_of_dish_l1493_149313

theorem original_price_of_dish :
  let P : ℝ := 40
  (0.9 * P + 0.15 * P) - (0.9 * P + 0.15 * 0.9 * P) = 0.60 → P = 40 := by
  intros P h
  sorry

end NUMINAMATH_GPT_original_price_of_dish_l1493_149313


namespace NUMINAMATH_GPT_avg_class_weight_is_46_67_l1493_149326

-- Define the total number of students in section A
def num_students_a : ℕ := 40

-- Define the average weight of students in section A
def avg_weight_a : ℚ := 50

-- Define the total number of students in section B
def num_students_b : ℕ := 20

-- Define the average weight of students in section B
def avg_weight_b : ℚ := 40

-- Calculate the total weight of section A
def total_weight_a : ℚ := num_students_a * avg_weight_a

-- Calculate the total weight of section B
def total_weight_b : ℚ := num_students_b * avg_weight_b

-- Calculate the total weight of the entire class
def total_weight_class : ℚ := total_weight_a + total_weight_b

-- Calculate the total number of students in the entire class
def total_students_class : ℕ := num_students_a + num_students_b

-- Calculate the average weight of the entire class
def avg_weight_class : ℚ := total_weight_class / total_students_class

-- Theorem to prove
theorem avg_class_weight_is_46_67 :
  avg_weight_class = 46.67 := sorry

end NUMINAMATH_GPT_avg_class_weight_is_46_67_l1493_149326


namespace NUMINAMATH_GPT_five_digit_divisible_by_four_digit_l1493_149366

theorem five_digit_divisible_by_four_digit (x y z u v : ℕ) (h1 : 1 ≤ x) (h2 : x < 10) (h3 : y < 10) (h4 : z < 10) (h5 : u < 10) (h6 : v < 10)
  (h7 : (x * 10^4 + y * 10^3 + z * 10^2 + u * 10 + v) % (x * 10^3 + y * 10^2 + u * 10 + v) = 0) : 
  ∃ N, 10 ≤ N ∧ N ≤ 99 ∧ 
  x * 10^4 + y * 10^3 + z * 10^2 + u * 10 + v = N * 10^3 ∧
  10 * (x * 10^3 + y * 10^2 + u * 10 + v) = x * 10^4 + y * 10^3 + z * 10^2 + u * 10 + v :=
sorry

end NUMINAMATH_GPT_five_digit_divisible_by_four_digit_l1493_149366


namespace NUMINAMATH_GPT_find_f_at_3_l1493_149312

noncomputable def f (x : ℝ) : ℝ :=
  ((x + 1) * (x^2 + 1) * (x^4 + 1) * (x^8 + 1) * (x^16 + 1) * (x^32 + 1) - 1) / (x^(2^6 - 1) - 1)

theorem find_f_at_3 : f 3 = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_f_at_3_l1493_149312


namespace NUMINAMATH_GPT_possible_values_of_a_l1493_149394

theorem possible_values_of_a (a : ℚ) : 
  (a^2 = 9 * 16) ∨ (16 * a = 81) ∨ (9 * a = 256) → 
  a = 12 ∨ a = -12 ∨ a = 81 / 16 ∨ a = 256 / 9 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_possible_values_of_a_l1493_149394


namespace NUMINAMATH_GPT_stone_width_l1493_149363

theorem stone_width (length_hall breadth_hall : ℝ) (num_stones length_stone : ℝ) (total_area_hall total_area_stones area_stone : ℝ)
  (h1 : length_hall = 36) (h2 : breadth_hall = 15) (h3 : num_stones = 5400) (h4 : length_stone = 2) 
  (h5 : total_area_hall = length_hall * breadth_hall * (10 * 10))
  (h6 : total_area_stones = num_stones * area_stone) 
  (h7 : area_stone = length_stone * (5 : ℝ)) 
  (h8 : total_area_stones = total_area_hall) : 
  (5 : ℝ) = 5 :=  
by sorry

end NUMINAMATH_GPT_stone_width_l1493_149363


namespace NUMINAMATH_GPT_relationship_log2_2_pow_03_l1493_149370

noncomputable def log_base_2 (x : ℝ) : ℝ := Real.log x / Real.log 2

theorem relationship_log2_2_pow_03 : 
  log_base_2 0.3 < (0.3)^2 ∧ (0.3)^2 < 2^(0.3) :=
by
  sorry

end NUMINAMATH_GPT_relationship_log2_2_pow_03_l1493_149370


namespace NUMINAMATH_GPT_speed_ratio_l1493_149308

theorem speed_ratio (va vb : ℝ) (L : ℝ) (h : va = vb * k) (head_start : vb * (L - 0.05 * L) = vb * L) : 
    (va / vb) = (1 / 0.95) :=
by
  sorry

end NUMINAMATH_GPT_speed_ratio_l1493_149308


namespace NUMINAMATH_GPT_placemat_length_l1493_149301

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

end NUMINAMATH_GPT_placemat_length_l1493_149301


namespace NUMINAMATH_GPT_fourth_root_difference_l1493_149351

theorem fourth_root_difference : (81 : ℝ) ^ (1 / 4 : ℝ) - (1296 : ℝ) ^ (1 / 4 : ℝ) = -3 :=
by
  sorry

end NUMINAMATH_GPT_fourth_root_difference_l1493_149351


namespace NUMINAMATH_GPT_smallest_integer_condition_l1493_149347

theorem smallest_integer_condition (p d n x : ℕ) (h1 : 1 ≤ d ∧ d ≤ 9) 
  (h2 : x = 10^p * d + n) (h3 : x = 19 * n) : 
  x = 95 := by
  sorry

end NUMINAMATH_GPT_smallest_integer_condition_l1493_149347


namespace NUMINAMATH_GPT_find_ages_l1493_149310

theorem find_ages (J sister cousin : ℝ)
  (h1 : J + 9 = 3 * (J - 11))
  (h2 : sister = 2 * J)
  (h3 : cousin = (J + sister) / 2) :
  J = 21 ∧ sister = 42 ∧ cousin = 31.5 :=
by
  sorry

end NUMINAMATH_GPT_find_ages_l1493_149310


namespace NUMINAMATH_GPT_gcd_pow_sub_one_l1493_149381

theorem gcd_pow_sub_one (n m : ℕ) (h1 : n = 1005) (h2 : m = 1016) :
  (Nat.gcd (2^n - 1) (2^m - 1)) = 2047 := by
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_gcd_pow_sub_one_l1493_149381


namespace NUMINAMATH_GPT_hyperbola_equation_l1493_149354

theorem hyperbola_equation 
  (vertex : ℝ × ℝ) 
  (asymptote_slope : ℝ) 
  (h_vertex : vertex = (2, 0))
  (h_asymptote : asymptote_slope = Real.sqrt 2) : 
  (∀ x y : ℝ, x^2 / 4 - y^2 / 8 = 1) := 
by
    sorry

end NUMINAMATH_GPT_hyperbola_equation_l1493_149354


namespace NUMINAMATH_GPT_surface_area_of_cube_l1493_149332

theorem surface_area_of_cube (a : ℝ) : 
  let edge_length := 4 * a
  let face_area := edge_length ^ 2
  let total_surface_area := 6 * face_area
  total_surface_area = 96 * a^2 := by
  sorry

end NUMINAMATH_GPT_surface_area_of_cube_l1493_149332


namespace NUMINAMATH_GPT_alpha_value_l1493_149373

theorem alpha_value
  (β γ δ α : ℝ) 
  (h1 : β = 100)
  (h2 : γ = 30)
  (h3 : δ = 150)
  (h4 : α + β + γ + 0.5 * γ = 360) : 
  α = 215 :=
by
  sorry

end NUMINAMATH_GPT_alpha_value_l1493_149373


namespace NUMINAMATH_GPT_quadratic_identity_l1493_149395

variables {R : Type*} [CommRing R] [IsDomain R]

-- Define the quadratic polynomial P
def P (a b c x : R) : R := a * x^2 + b * x + c

-- Conditions as definitions in Lean
variables (a b c : R) (h₁ : P a b c a = 2021 * b * c)
                (h₂ : P a b c b = 2021 * c * a)
                (h₃ : P a b c c = 2021 * a * b)
                (dist : (a ≠ b) ∧ (b ≠ c) ∧ (a ≠ c))

-- The main theorem statement
theorem quadratic_identity : a + 2021 * b + c = 0 :=
sorry

end NUMINAMATH_GPT_quadratic_identity_l1493_149395


namespace NUMINAMATH_GPT_coordinates_of_point_A_l1493_149387

def f (x : ℝ) : ℝ := x^2 + 3 * x

theorem coordinates_of_point_A (a : ℝ) (b : ℝ) 
    (slope_condition : deriv f a = 7) 
    (point_condition : f a = b) : 
    a = 2 ∧ b = 10 := 
by {
    sorry
}

end NUMINAMATH_GPT_coordinates_of_point_A_l1493_149387


namespace NUMINAMATH_GPT_find_phi_monotone_interval_1_monotone_interval_2_l1493_149346

-- Definitions related to the function f
noncomputable def f (x φ a : ℝ) : ℝ :=
  Real.sin (x + φ) + a * Real.cos x

-- Problem Part 1: Given f(π/2) = √2 / 2, find φ
theorem find_phi (a : ℝ) (φ : ℝ) (h : |φ| < Real.pi / 2) (hf : f (π / 2) φ a = Real.sqrt 2 / 2) :
  φ = π / 4 ∨ φ = -π / 4 :=
  sorry

-- Problem Part 2 Condition 1: Given a = √3, φ = -π/3, find the monotonically increasing interval
theorem monotone_interval_1 :
  ∀ k : ℤ, ∀ x : ℝ, 
  ((-5 * π / 6) + 2 * k * π) ≤ x ∧ x ≤ (π / 6 + 2 * k * π) → 
  f x (-π / 3) (Real.sqrt 3) = Real.sin (x + π / 3) :=
  sorry

-- Problem Part 2 Condition 2: Given a = -1, φ = π/6, find the monotonically increasing interval
theorem monotone_interval_2 :
  ∀ k : ℤ, ∀ x : ℝ, 
  ((-π / 3) + 2 * k * π) ≤ x ∧ x ≤ ((2 * π / 3) + 2 * k * π) → 
  f x (π / 6) (-1) = Real.sin (x - π / 6) :=
  sorry

end NUMINAMATH_GPT_find_phi_monotone_interval_1_monotone_interval_2_l1493_149346


namespace NUMINAMATH_GPT_angle_same_terminal_side_l1493_149344

theorem angle_same_terminal_side (k : ℤ) : 
  ∃ (θ : ℤ), θ = -324 ∧ 
    ∀ α : ℤ, α = 36 + k * 360 → 
            ( (α % 360 = θ % 360) ∨ (α % 360 + 360 = θ % 360) ∨ (θ % 360 + 360 = α % 360)) :=
by
  sorry

end NUMINAMATH_GPT_angle_same_terminal_side_l1493_149344


namespace NUMINAMATH_GPT_fraction_of_area_above_line_l1493_149329

theorem fraction_of_area_above_line :
  let A := (3, 2)
  let B := (6, 0)
  let side_length := B.fst - A.fst
  let square_area := side_length ^ 2
  let triangle_base := B.fst - A.fst
  let triangle_height := A.snd
  let triangle_area := (1 / 2 : ℚ) * triangle_base * triangle_height
  let area_above_line := square_area - triangle_area
  let fraction_above_line := area_above_line / square_area
  fraction_above_line = (2 / 3 : ℚ) :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_area_above_line_l1493_149329


namespace NUMINAMATH_GPT_triangle_side_b_value_l1493_149334

theorem triangle_side_b_value (a b c : ℝ) (A B C : ℝ) (h1 : a = Real.sqrt 3) (h2 : A = 60) (h3 : C = 75) : b = Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_triangle_side_b_value_l1493_149334


namespace NUMINAMATH_GPT_marked_price_percentage_l1493_149369

theorem marked_price_percentage (L C M S : ℝ) 
  (h1 : C = 0.7 * L) 
  (h2 : C = 0.7 * S) 
  (h3 : S = 0.9 * M) 
  (h4 : S = L) 
  : M = (10 / 9) * L := 
by
  sorry

end NUMINAMATH_GPT_marked_price_percentage_l1493_149369


namespace NUMINAMATH_GPT_inverse_proportion_passing_through_l1493_149360

theorem inverse_proportion_passing_through (k : ℝ) :
  (∀ x y : ℝ, (y = k / x) → (x = 3 → y = 2)) → k = 6 := 
by
  sorry

end NUMINAMATH_GPT_inverse_proportion_passing_through_l1493_149360


namespace NUMINAMATH_GPT_find_a_and_b_l1493_149361

theorem find_a_and_b (a b : ℤ) (h : ∀ x : ℝ, x ≤ 0 → (a*x + 2)*(x^2 + 2*b) ≤ 0) : a = 1 ∧ b = -2 := 
by 
  -- Proof steps would go here, but they are omitted as per instructions.
  sorry

end NUMINAMATH_GPT_find_a_and_b_l1493_149361


namespace NUMINAMATH_GPT_find_k_l1493_149362

theorem find_k (k : ℤ) :
  (-x^2 - (k + 10)*x - 8 = -(x - 2)*(x - 4)) → k = -16 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l1493_149362


namespace NUMINAMATH_GPT_distance_between_trees_l1493_149322

theorem distance_between_trees (yard_length : ℕ) (number_of_trees : ℕ) (number_of_gaps : ℕ)
  (h1 : yard_length = 400) (h2 : number_of_trees = 26) (h3 : number_of_gaps = number_of_trees - 1) :
  yard_length / number_of_gaps = 16 := by
  sorry

end NUMINAMATH_GPT_distance_between_trees_l1493_149322


namespace NUMINAMATH_GPT_triangle_area_is_correct_l1493_149393

noncomputable def isosceles_triangle_area : Prop :=
  let side_large_square := 6 -- sides of the large square WXYZ
  let area_large_square := side_large_square * side_large_square
  let side_small_square := 2 -- sides of the smaller squares
  let BC := side_large_square - 2 * side_small_square -- length of BC
  let height_AM := side_large_square / 2 + side_small_square -- height of the triangle from A to M
  let area_ABC := (BC * height_AM) / 2 -- area of the triangle ABC
  area_large_square = 36 ∧ BC = 2 ∧ height_AM = 5 ∧ area_ABC = 5

theorem triangle_area_is_correct : isosceles_triangle_area := sorry

end NUMINAMATH_GPT_triangle_area_is_correct_l1493_149393


namespace NUMINAMATH_GPT_unique_solution_abc_l1493_149375

theorem unique_solution_abc (a b c : ℕ) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) 
(h1 : b ∣ 2^a - 1) 
(h2 : c ∣ 2^b - 1) 
(h3 : a ∣ 2^c - 1) : 
  a = 1 ∧ b = 1 ∧ c = 1 :=
sorry

end NUMINAMATH_GPT_unique_solution_abc_l1493_149375


namespace NUMINAMATH_GPT_total_bouncy_balls_l1493_149389

-- Definitions based on the conditions of the problem
def packs_of_red := 4
def packs_of_yellow := 8
def packs_of_green := 4
def balls_per_pack := 10

-- Theorem stating the conclusion to be proven
theorem total_bouncy_balls :
  (packs_of_red + packs_of_yellow + packs_of_green) * balls_per_pack = 160 := 
by
  sorry

end NUMINAMATH_GPT_total_bouncy_balls_l1493_149389


namespace NUMINAMATH_GPT_additional_carpet_needed_is_94_l1493_149367

noncomputable def area_room_a : ℝ := 4 * 20

noncomputable def area_room_b : ℝ := area_room_a / 2.5

noncomputable def total_area : ℝ := area_room_a + area_room_b

noncomputable def carpet_jessie_has : ℝ := 18

noncomputable def additional_carpet_needed : ℝ := total_area - carpet_jessie_has

theorem additional_carpet_needed_is_94 :
  additional_carpet_needed = 94 := by
  sorry

end NUMINAMATH_GPT_additional_carpet_needed_is_94_l1493_149367


namespace NUMINAMATH_GPT_infinite_solutions_l1493_149321

theorem infinite_solutions (b : ℤ) : 
  (∀ x : ℤ, 4 * (3 * x - b) = 3 * (4 * x + 16)) ↔ b = -12 := 
by sorry

end NUMINAMATH_GPT_infinite_solutions_l1493_149321


namespace NUMINAMATH_GPT_find_distance_l1493_149368

variable (D V : ℕ)

axiom normal_speed : V = 25
axiom time_difference : (D / V) - (D / (V + 5)) = 2

theorem find_distance : D = 300 :=
by
  sorry

end NUMINAMATH_GPT_find_distance_l1493_149368


namespace NUMINAMATH_GPT_area_of_square_with_diagonal_40_l1493_149349

theorem area_of_square_with_diagonal_40 {d : ℝ} (h : d = 40) : ∃ A : ℝ, A = 800 :=
by
  sorry

end NUMINAMATH_GPT_area_of_square_with_diagonal_40_l1493_149349


namespace NUMINAMATH_GPT_f_increasing_on_neg_inf_to_one_l1493_149397

def f (x : ℝ) : ℝ := -x^2 + 2 * x + 8

theorem f_increasing_on_neg_inf_to_one :
  ∀ x y : ℝ, x < y ∧ y ≤ 1 → f x < f y :=
sorry

end NUMINAMATH_GPT_f_increasing_on_neg_inf_to_one_l1493_149397


namespace NUMINAMATH_GPT_lucas_seq_mod_50_l1493_149314

def lucas_seq : ℕ → ℕ
| 0       => 2
| 1       => 5
| (n + 2) => lucas_seq n + lucas_seq (n + 1)

theorem lucas_seq_mod_50 : lucas_seq 49 % 5 = 0 := 
by
  sorry

end NUMINAMATH_GPT_lucas_seq_mod_50_l1493_149314


namespace NUMINAMATH_GPT_convert_cost_to_usd_l1493_149399

def sandwich_cost_gbp : Float := 15.0
def conversion_rate : Float := 1.3

theorem convert_cost_to_usd :
  (Float.round ((sandwich_cost_gbp * conversion_rate) * 100) / 100) = 19.50 :=
by
  sorry

end NUMINAMATH_GPT_convert_cost_to_usd_l1493_149399


namespace NUMINAMATH_GPT_calculate_expression_l1493_149315

variable {x y : ℝ}

theorem calculate_expression (hx : x ≠ 0) (hy : y ≠ 0) (h : x^2 = 1 / y^2) :
  (x^2 - 1 / x^2) * (y^2 + 1 / y^2) = x^4 - y^4 := by
  sorry

end NUMINAMATH_GPT_calculate_expression_l1493_149315


namespace NUMINAMATH_GPT_line_point_t_l1493_149386

theorem line_point_t (t : ℝ) : 
  (∃ t, (0, 3) = (0, 3) ∧ (-8, 0) = (-8, 0) ∧ (5 - 3) / t = 3 / 8) → (t = 16 / 3) :=
by
  sorry

end NUMINAMATH_GPT_line_point_t_l1493_149386


namespace NUMINAMATH_GPT_find_m_from_inequality_l1493_149385

theorem find_m_from_inequality :
  (∀ x, x^2 - (m+2)*x > 0 ↔ (x < 0 ∨ x > 2)) → m = 0 :=
by
  sorry

end NUMINAMATH_GPT_find_m_from_inequality_l1493_149385


namespace NUMINAMATH_GPT_Jen_visits_either_but_not_both_l1493_149359

-- Define the events and their associated probabilities
def P_Chile : ℝ := 0.30
def P_Madagascar : ℝ := 0.50

-- Define the probability of visiting both assuming independence
def P_both : ℝ := P_Chile * P_Madagascar

-- Define the probability of visiting either but not both
def P_either_but_not_both : ℝ := P_Chile + P_Madagascar - 2 * P_both

-- The problem statement
theorem Jen_visits_either_but_not_both : P_either_but_not_both = 0.65 := by
  /- The proof goes here -/
  sorry

end NUMINAMATH_GPT_Jen_visits_either_but_not_both_l1493_149359


namespace NUMINAMATH_GPT_evaluate_floor_of_negative_seven_halves_l1493_149383

def floor_of_negative_seven_halves : ℤ :=
  Int.floor (-7 / 2)

theorem evaluate_floor_of_negative_seven_halves :
  floor_of_negative_seven_halves = -4 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_floor_of_negative_seven_halves_l1493_149383


namespace NUMINAMATH_GPT_fraction_zero_x_value_l1493_149352

theorem fraction_zero_x_value (x : ℝ) (h : (x^2 - 4) / (x - 2) = 0) (h2 : x ≠ 2) : x = -2 :=
sorry

end NUMINAMATH_GPT_fraction_zero_x_value_l1493_149352


namespace NUMINAMATH_GPT_classification_of_square_and_cube_roots_l1493_149325

-- Define the three cases: positive, zero, and negative
inductive NumberCase
| positive 
| zero 
| negative 

-- Define the concept of "classification and discussion thinking"
def is_classification_and_discussion_thinking (cases : List NumberCase) : Prop :=
  cases = [NumberCase.positive, NumberCase.zero, NumberCase.negative]

-- The main statement to be proven
theorem classification_of_square_and_cube_roots :
  is_classification_and_discussion_thinking [NumberCase.positive, NumberCase.zero, NumberCase.negative] :=
by
  sorry

end NUMINAMATH_GPT_classification_of_square_and_cube_roots_l1493_149325


namespace NUMINAMATH_GPT_f_inv_f_inv_17_l1493_149353

noncomputable def f (x : ℝ) : ℝ := 4 * x - 3

noncomputable def f_inv (y : ℝ) : ℝ := (y + 3) / 4

theorem f_inv_f_inv_17 : f_inv (f_inv 17) = 2 := by
  sorry

end NUMINAMATH_GPT_f_inv_f_inv_17_l1493_149353


namespace NUMINAMATH_GPT_servings_per_bottle_l1493_149371

-- Definitions based on conditions
def total_guests : ℕ := 120
def servings_per_guest : ℕ := 2
def total_bottles : ℕ := 40

-- Theorem stating that given the conditions, the servings per bottle is 6
theorem servings_per_bottle : (total_guests * servings_per_guest) / total_bottles = 6 := by
  sorry

end NUMINAMATH_GPT_servings_per_bottle_l1493_149371


namespace NUMINAMATH_GPT_crayon_production_correct_l1493_149377

def numColors := 4
def crayonsPerColor := 2
def boxesPerHour := 5
def hours := 4

def crayonsPerBox := numColors * crayonsPerColor
def crayonsPerHour := boxesPerHour * crayonsPerBox
def totalCrayons := hours * crayonsPerHour

theorem crayon_production_correct :
  totalCrayons = 160 :=  
by
  sorry

end NUMINAMATH_GPT_crayon_production_correct_l1493_149377


namespace NUMINAMATH_GPT_maximum_of_f_attain_maximum_of_f_l1493_149307

def f (x : ℝ) : ℝ := x^3 + 3 * x^2 - 4

theorem maximum_of_f : ∀ x : ℝ, f x ≤ 0 :=
sorry

theorem attain_maximum_of_f : ∃ x : ℝ, f x = 0 :=
sorry

end NUMINAMATH_GPT_maximum_of_f_attain_maximum_of_f_l1493_149307


namespace NUMINAMATH_GPT_boys_in_class_is_120_l1493_149345

-- Definitions from conditions
def num_boys_in_class (number_of_girls number_of_boys : Nat) : Prop :=
  ∃ x : Nat, number_of_girls = 5 * x ∧ number_of_boys = 6 * x ∧
             (5 * x - 20) * 3 = 2 * (6 * x)

-- The theorem proving that given the conditions, the number of boys in the class is 120.
theorem boys_in_class_is_120 (number_of_girls number_of_boys : Nat) (h : num_boys_in_class number_of_girls number_of_boys) :
  number_of_boys = 120 :=
by
  sorry

end NUMINAMATH_GPT_boys_in_class_is_120_l1493_149345


namespace NUMINAMATH_GPT_darks_washing_time_l1493_149317

theorem darks_washing_time (x : ℕ) :
  (72 + x + 45) + (50 + 65 + 54) = 344 → x = 58 :=
by
  sorry

end NUMINAMATH_GPT_darks_washing_time_l1493_149317


namespace NUMINAMATH_GPT_find_b_of_perpendicular_lines_l1493_149339

theorem find_b_of_perpendicular_lines (b : ℝ) (h : 4 * b - 8 = 0) : b = 2 := 
by 
  sorry

end NUMINAMATH_GPT_find_b_of_perpendicular_lines_l1493_149339


namespace NUMINAMATH_GPT_mario_garden_total_blossoms_l1493_149320

def hibiscus_growth (initial_flowers growth_rate weeks : ℕ) : ℕ :=
  initial_flowers + growth_rate * weeks

def rose_growth (initial_flowers growth_rate weeks : ℕ) : ℕ :=
  initial_flowers + growth_rate * weeks

theorem mario_garden_total_blossoms :
  let weeks := 2
  let hibiscus1 := hibiscus_growth 2 3 weeks
  let hibiscus2 := hibiscus_growth (2 * 2) 4 weeks
  let hibiscus3 := hibiscus_growth (4 * (2 * 2)) 5 weeks
  let rose1 := rose_growth 3 2 weeks
  let rose2 := rose_growth 5 3 weeks
  hibiscus1 + hibiscus2 + hibiscus3 + rose1 + rose2 = 64 := 
by
  sorry

end NUMINAMATH_GPT_mario_garden_total_blossoms_l1493_149320


namespace NUMINAMATH_GPT_mary_needs_more_sugar_l1493_149355

theorem mary_needs_more_sugar 
  (sugar_needed flour_needed salt_needed already_added_flour : ℕ)
  (h1 : sugar_needed = 11)
  (h2 : flour_needed = 6)
  (h3 : salt_needed = 9)
  (h4 : already_added_flour = 12) :
  (sugar_needed - salt_needed) = 2 :=
by
  sorry

end NUMINAMATH_GPT_mary_needs_more_sugar_l1493_149355


namespace NUMINAMATH_GPT_find_x_for_sin_minus_cos_eq_sqrt2_l1493_149337

theorem find_x_for_sin_minus_cos_eq_sqrt2 (x : ℝ) (h1 : 0 ≤ x) (h2 : x < 2 * Real.pi) (h3 : Real.sin x - Real.cos x = Real.sqrt 2) : x = 3 * Real.pi / 4 :=
by
  sorry

end NUMINAMATH_GPT_find_x_for_sin_minus_cos_eq_sqrt2_l1493_149337


namespace NUMINAMATH_GPT_twenty_five_point_zero_six_million_in_scientific_notation_l1493_149333

theorem twenty_five_point_zero_six_million_in_scientific_notation :
  (25.06e6 : ℝ) = 2.506 * 10^7 :=
by
  -- The proof would go here, but we use sorry to skip the proof.
  sorry

end NUMINAMATH_GPT_twenty_five_point_zero_six_million_in_scientific_notation_l1493_149333


namespace NUMINAMATH_GPT_books_sold_on_wednesday_l1493_149380

theorem books_sold_on_wednesday
  (initial_stock : ℕ)
  (sold_monday : ℕ)
  (sold_tuesday : ℕ)
  (sold_thursday : ℕ)
  (sold_friday : ℕ)
  (percent_unsold : ℚ) :
  initial_stock = 900 →
  sold_monday = 75 →
  sold_tuesday = 50 →
  sold_thursday = 78 →
  sold_friday = 135 →
  percent_unsold = 55.333333333333336 →
  ∃ (sold_wednesday : ℕ), sold_wednesday = 64 :=
by
  sorry

end NUMINAMATH_GPT_books_sold_on_wednesday_l1493_149380


namespace NUMINAMATH_GPT_find_natural_numbers_satisfying_prime_square_l1493_149336

-- Define conditions as a Lean statement
theorem find_natural_numbers_satisfying_prime_square (n : ℕ) (h : ∃ p : ℕ, Prime p ∧ (2 * n^2 + 3 * n - 35 = p^2)) :
  n = 4 ∨ n = 12 :=
sorry

end NUMINAMATH_GPT_find_natural_numbers_satisfying_prime_square_l1493_149336


namespace NUMINAMATH_GPT_no_solution_intervals_l1493_149316

theorem no_solution_intervals :
    ¬ ∃ x : ℝ, (2 / 3 < x ∧ x < 4 / 3) ∧ (1 / 5 < x ∧ x < 3 / 5) :=
by
  sorry

end NUMINAMATH_GPT_no_solution_intervals_l1493_149316


namespace NUMINAMATH_GPT_triangle_angle_condition_l1493_149309

theorem triangle_angle_condition (a b h_3 : ℝ) (A C : ℝ) 
  (h : 1/(h_3^2) = 1/(a^2) + 1/(b^2)) :
  C = 90 ∨ |A - C| = 90 := 
sorry

end NUMINAMATH_GPT_triangle_angle_condition_l1493_149309


namespace NUMINAMATH_GPT_like_terms_expression_value_l1493_149323

theorem like_terms_expression_value (m n : ℤ) (h1 : m = 3) (h2 : n = 1) :
  3^2 * n - (2 * m * n^2 - 2 * (m^2 * n + 2 * m * n^2)) = 33 := by
  sorry

end NUMINAMATH_GPT_like_terms_expression_value_l1493_149323


namespace NUMINAMATH_GPT_find_x_plus_y_l1493_149350

theorem find_x_plus_y (x y : ℕ) 
  (h1 : 4^x = 16^(y + 1)) 
  (h2 : 5^(2 * y) = 25^(x - 2)) : 
  x + y = 2 := 
sorry

end NUMINAMATH_GPT_find_x_plus_y_l1493_149350


namespace NUMINAMATH_GPT_range_of_a_l1493_149358

noncomputable def setA (a : ℝ) : Set ℝ := {x | 2 * a ≤ x ∧ x ≤ a + 3}
noncomputable def setB : Set ℝ := {x | x < -1 ∨ x > 5}

theorem range_of_a (a : ℝ) : (setA a ∩ setB = ∅) ↔ (a > 3 ∨ (-1 / 2 ≤ a ∧ a ≤ 2)) := 
  sorry

end NUMINAMATH_GPT_range_of_a_l1493_149358


namespace NUMINAMATH_GPT_geometric_sequence_sum_l1493_149356

-- Define the relations for geometric sequences
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ (m n p q : ℕ), m + n = p + q → a m * a n = a p * a q

theorem geometric_sequence_sum (a : ℕ → ℝ) (h_geom : is_geometric_sequence a) (h_pos : ∀ n, a n > 0)
  (h_cond : a 4 * a 6 + 2 * a 5 * a 7 + a 6 * a 8 = 36) : a 5 + a 7 = 6 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l1493_149356


namespace NUMINAMATH_GPT_repair_time_calculation_l1493_149392

-- Assume amount of work is represented as units
def work_10_people_45_minutes := 10 * 45
def work_20_people_20_minutes := 20 * 20

-- Assuming the flood destroys 2 units per minute as calculated in the solution
def flood_rate := 2

-- Calculate total initial units of the dike
def dike_initial_units :=
  work_10_people_45_minutes - flood_rate * 45

-- Given 14 people are repairing the dam
def repair_rate_14_people := 14 - flood_rate

-- Statement to prove that 14 people need 30 minutes to repair the dam
theorem repair_time_calculation :
  dike_initial_units / repair_rate_14_people = 30 :=
by
  sorry

end NUMINAMATH_GPT_repair_time_calculation_l1493_149392


namespace NUMINAMATH_GPT_initial_percentage_water_is_80_l1493_149341

noncomputable def initial_kola_solution := 340
noncomputable def added_sugar := 3.2
noncomputable def added_water := 10
noncomputable def added_kola := 6.8
noncomputable def final_percentage_sugar := 14.111111111111112
noncomputable def percentage_kola := 6

theorem initial_percentage_water_is_80 :
  ∃ (W : ℝ), W = 80 :=
by
  sorry

end NUMINAMATH_GPT_initial_percentage_water_is_80_l1493_149341


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1493_149318

theorem solution_set_of_inequality (x : ℝ) : x * (1 - x) > 0 ↔ 0 < x ∧ x < 1 :=
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1493_149318


namespace NUMINAMATH_GPT_percentage_above_wholesale_correct_l1493_149303

variable (wholesale_cost retail_cost employee_payment : ℝ)
variable (employee_discount percentage_above_wholesale : ℝ)

theorem percentage_above_wholesale_correct :
  wholesale_cost = 200 → 
  employee_discount = 0.25 → 
  employee_payment = 180 → 
  retail_cost = wholesale_cost + (percentage_above_wholesale / 100) * wholesale_cost →
  employee_payment = (1 - employee_discount) * retail_cost →
  percentage_above_wholesale = 20 :=
by
  intros
  sorry

end NUMINAMATH_GPT_percentage_above_wholesale_correct_l1493_149303


namespace NUMINAMATH_GPT_find_takeoff_run_distance_l1493_149398

-- Define the conditions
def time_of_takeoff : ℝ := 15 -- seconds
def takeoff_speed_kmh : ℝ := 100 -- km/h

-- Define the conversions and proof problem
noncomputable def takeoff_speed_ms : ℝ := takeoff_speed_kmh * 1000 / 3600 -- conversion from km/h to m/s
noncomputable def acceleration : ℝ := takeoff_speed_ms / time_of_takeoff -- a = v / t

theorem find_takeoff_run_distance : 
  (1/2) * acceleration * (time_of_takeoff ^ 2) = 208 := by
  sorry

end NUMINAMATH_GPT_find_takeoff_run_distance_l1493_149398


namespace NUMINAMATH_GPT_total_snakes_owned_l1493_149304

theorem total_snakes_owned 
  (total_people : ℕ)
  (only_dogs only_cats only_birds only_snakes : ℕ)
  (cats_and_dogs birds_and_dogs birds_and_cats snakes_and_dogs snakes_and_cats snakes_and_birds : ℕ)
  (cats_dogs_snakes cats_dogs_birds cats_birds_snakes dogs_birds_snakes all_four_pets : ℕ)
  (h1 : total_people = 150)
  (h2 : only_dogs = 30)
  (h3 : only_cats = 25)
  (h4 : only_birds = 10)
  (h5 : only_snakes = 7)
  (h6 : cats_and_dogs = 15)
  (h7 : birds_and_dogs = 12)
  (h8 : birds_and_cats = 8)
  (h9 : snakes_and_dogs = 3)
  (h10 : snakes_and_cats = 4)
  (h11 : snakes_and_birds = 2)
  (h12 : cats_dogs_snakes = 5)
  (h13 : cats_dogs_birds = 4)
  (h14 : cats_birds_snakes = 6)
  (h15 : dogs_birds_snakes = 9)
  (h16 : all_four_pets = 10) : 
  7 + 3 + 4 + 2 + 5 + 6 + 9 + 10 = 46 := 
sorry

end NUMINAMATH_GPT_total_snakes_owned_l1493_149304


namespace NUMINAMATH_GPT_kanul_raw_material_expense_l1493_149311

theorem kanul_raw_material_expense
  (total_amount : ℝ)
  (machinery_cost : ℝ)
  (raw_materials_cost : ℝ)
  (cash_fraction : ℝ)
  (h_total_amount : total_amount = 137500)
  (h_machinery_cost : machinery_cost = 30000)
  (h_cash_fraction: cash_fraction = 0.20)
  (h_eq : total_amount = raw_materials_cost + machinery_cost + cash_fraction * total_amount) :
  raw_materials_cost = 80000 :=
by
  rw [h_total_amount, h_machinery_cost, h_cash_fraction] at h_eq
  sorry

end NUMINAMATH_GPT_kanul_raw_material_expense_l1493_149311


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1493_149302

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (1 ≤ x ∧ x ≤ 4) ↔ (1 ≤ x^2 ∧ x^2 ≤ 16) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1493_149302


namespace NUMINAMATH_GPT_find_common_ratio_l1493_149338

-- Declare the sequence and conditions
variables {a : ℕ → ℝ} 
variable {q : ℝ}

-- Conditions of the problem 
def positive_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  (∀ m n : ℕ, a m = a 0 * q ^ m) ∧ q > 0

def third_term_condition (a : ℕ → ℝ) (q : ℝ) : Prop :=
  a 3 + a 5 = 5

def fifth_term_seventh_term_condition (a : ℕ → ℝ) (q : ℝ) : Prop :=
  a 5 + a 7 = 20

-- The final lean statement proving the common ratio is 2
theorem find_common_ratio 
  (h1 : positive_geometric_sequence a q) 
  (h2 : third_term_condition a q) 
  (h3 : fifth_term_seventh_term_condition a q) : 
  q = 2 :=
sorry

end NUMINAMATH_GPT_find_common_ratio_l1493_149338


namespace NUMINAMATH_GPT_triangle_third_side_one_third_perimeter_l1493_149305

theorem triangle_third_side_one_third_perimeter
  (a b x y p c : ℝ)
  (h1 : x^2 - y^2 = a^2 - b^2)
  (h2 : p = (a + b + c) / 2)
  (h3 : x - y = 2 * (a - b)) :
  c = (a + b + c) / 3 := by
  sorry

end NUMINAMATH_GPT_triangle_third_side_one_third_perimeter_l1493_149305


namespace NUMINAMATH_GPT_cost_of_plastering_is_334_point_8_l1493_149324

def tank_length : ℝ := 25
def tank_width : ℝ := 12
def tank_depth : ℝ := 6
def cost_per_sq_meter : ℝ := 0.45

def bottom_area : ℝ := tank_length * tank_width
def long_wall_area : ℝ := 2 * (tank_length * tank_depth)
def short_wall_area : ℝ := 2 * (tank_width * tank_depth)
def total_surface_area : ℝ := bottom_area + long_wall_area + short_wall_area
def total_cost : ℝ := total_surface_area * cost_per_sq_meter

theorem cost_of_plastering_is_334_point_8 :
  total_cost = 334.8 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_plastering_is_334_point_8_l1493_149324


namespace NUMINAMATH_GPT_smallest_positive_period_axis_of_symmetry_minimum_value_on_interval_l1493_149372

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - 3 * Real.pi / 5)

theorem smallest_positive_period :
  ∃ T > 0, (∀ x, f (x + T) = f x) ∧
  ∀ T' > 0, (∀ x, f (x + T') = f x) → T' ≥ T := by
  sorry

theorem axis_of_symmetry :
  ∃ k : ℤ, (∀ x, f x = f (11 * Real.pi / 20 + k * Real.pi / 2)) := by
  sorry

theorem minimum_value_on_interval :
  ∃ x ∈ Set.Icc (0 : ℝ) (Real.pi / 2), f x = -1 := by
  sorry

end NUMINAMATH_GPT_smallest_positive_period_axis_of_symmetry_minimum_value_on_interval_l1493_149372


namespace NUMINAMATH_GPT_ab_sum_l1493_149364

theorem ab_sum (a b : ℝ) (h : |a - 4| + (b + 1)^2 = 0) : a + b = 3 :=
by
  sorry -- this is where the proof would go

end NUMINAMATH_GPT_ab_sum_l1493_149364


namespace NUMINAMATH_GPT_contradictory_goldbach_l1493_149374

theorem contradictory_goldbach : ¬ (∀ n : ℕ, 2 < n ∧ Even n → ∃ p q : ℕ, Prime p ∧ Prime q ∧ n = p + q) :=
sorry

end NUMINAMATH_GPT_contradictory_goldbach_l1493_149374


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1493_149391

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (∃ a, (1 + a) ^ 6 = 64) →
  (a = 1 → (1 + a) ^ 6 = 64) ∧ ¬(∀ a, ((1 + a) ^ 6 = 64 → a = 1)) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1493_149391


namespace NUMINAMATH_GPT_eval_expression_l1493_149331

theorem eval_expression (a b : ℤ) (h₁ : a = 4) (h₂ : b = -2) : -a - b^2 + a*b + a^2 = 0 := by
  sorry

end NUMINAMATH_GPT_eval_expression_l1493_149331


namespace NUMINAMATH_GPT_same_terminal_side_l1493_149382

theorem same_terminal_side (α : ℝ) (k : ℤ) (h : α = -51) : 
  ∃ (m : ℤ), α + m * 360 = k * 360 - 51 :=
by {
    sorry
}

end NUMINAMATH_GPT_same_terminal_side_l1493_149382


namespace NUMINAMATH_GPT_value_of_expression_l1493_149348

noncomputable def f : ℝ → ℝ
| x => if x > 0 then -1 else if x < 0 then 1 else 0

theorem value_of_expression (a b : ℝ) (h : a ≠ b) :
  (a + b + (a - b) * f (a - b)) / 2 = min a b := 
sorry

end NUMINAMATH_GPT_value_of_expression_l1493_149348


namespace NUMINAMATH_GPT_check_perfect_squares_l1493_149390

-- Define the prime factorizations of each option
def optionA := 3^3 * 4^5 * 7^7
def optionB := 3^4 * 4^4 * 7^6
def optionC := 3^6 * 4^3 * 7^8
def optionD := 3^5 * 4^6 * 7^5
def optionE := 3^4 * 4^6 * 7^7

-- Definition of a perfect square (all exponents in prime factorization are even)
def is_perfect_square (n : ℕ) : Prop :=
  ∀ p : ℕ, (p ^ 2 ∣ n) -> (p ∣ n)

-- The Lean statement asserting which options are perfect squares
theorem check_perfect_squares :
  (is_perfect_square optionB) ∧ (is_perfect_square optionC) ∧
  ¬(is_perfect_square optionA) ∧ ¬(is_perfect_square optionD) ∧ ¬(is_perfect_square optionE) :=
by sorry

end NUMINAMATH_GPT_check_perfect_squares_l1493_149390


namespace NUMINAMATH_GPT_solve_star_op_eq_l1493_149365

def star_op (a b : ℕ) : ℕ :=
  if a < b then b * b else b * b * b

theorem solve_star_op_eq :
  ∃ x : ℕ, 5 * star_op 5 x = 64 ∧ (x = 4 ∨ x = 8) :=
sorry

end NUMINAMATH_GPT_solve_star_op_eq_l1493_149365


namespace NUMINAMATH_GPT_real_root_exists_l1493_149330

theorem real_root_exists (a : ℝ) : 
    (∃ x : ℝ, x^4 - a * x^3 - x^2 - a * x + 1 = 0) ↔ (-1 / 2 ≤ a) := by
  sorry

end NUMINAMATH_GPT_real_root_exists_l1493_149330


namespace NUMINAMATH_GPT_range_a_of_abs_2x_minus_a_eq_1_two_real_solutions_l1493_149376

open Real

theorem range_a_of_abs_2x_minus_a_eq_1_two_real_solutions :
  {a : ℝ | ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (abs (2^x1 - a) = 1) ∧ (abs (2^x2 - a) = 1)} = {a : ℝ | 1 < a} :=
by
  sorry

end NUMINAMATH_GPT_range_a_of_abs_2x_minus_a_eq_1_two_real_solutions_l1493_149376


namespace NUMINAMATH_GPT_sum_of_given_geom_series_l1493_149388

-- Define the necessary conditions
def first_term (a : ℕ) := a = 2
def common_ratio (r : ℕ) := r = 3
def number_of_terms (n : ℕ) := n = 6

-- Define the sum of the geometric series
def sum_geom_series (a r n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1)

-- State the theorem
theorem sum_of_given_geom_series :
  first_term 2 → common_ratio 3 → number_of_terms 6 → sum_geom_series 2 3 6 = 728 :=
by
  intros h1 h2 h3
  rw [first_term] at h1
  rw [common_ratio] at h2
  rw [number_of_terms] at h3
  have h1 : 2 = 2 := by exact h1
  have h2 : 3 = 3 := by exact h2
  have h3 : 6 = 6 := by exact h3
  exact sorry

end NUMINAMATH_GPT_sum_of_given_geom_series_l1493_149388


namespace NUMINAMATH_GPT_find_b_l1493_149384

theorem find_b (b : ℝ) (h : ∃ (f_inv : ℝ → ℝ), (∀ x y, f_inv (2^x + b) = y) ∧ f_inv 5 = 2) :
    b = 1 := by
  sorry

end NUMINAMATH_GPT_find_b_l1493_149384


namespace NUMINAMATH_GPT_equation_of_perpendicular_line_l1493_149319

theorem equation_of_perpendicular_line (c : ℝ) :
  (∃ c : ℝ, ∀ x y : ℝ, x - 2 * y + c = 0 ∧ 2 * x + y - 5 = 0) → (x - 2 * y - 3 = 0) := 
by
  sorry

end NUMINAMATH_GPT_equation_of_perpendicular_line_l1493_149319


namespace NUMINAMATH_GPT_fraction_of_price_l1493_149342

theorem fraction_of_price (d : ℝ) : d * 0.65 * 0.70 = d * 0.455 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_price_l1493_149342


namespace NUMINAMATH_GPT_product_ne_sum_11_times_l1493_149378

def is_prime (n : ℕ) : Prop := ∀ m, m > 1 → m < n → n % m ≠ 0
def prime_sum_product_condition (a b c d : ℕ) : Prop := 
  a * b * c * d = 11 * (a + b + c + d)

theorem product_ne_sum_11_times (a b c d : ℕ)
  (ha : is_prime a) (hb : is_prime b) (hc : is_prime c) (hd : is_prime d)
  (h : prime_sum_product_condition a b c d) :
  (a + b + c + d ≠ 46) ∧ (a + b + c + d ≠ 47) ∧ (a + b + c + d ≠ 48) :=
by  
  sorry

end NUMINAMATH_GPT_product_ne_sum_11_times_l1493_149378


namespace NUMINAMATH_GPT_predict_monthly_savings_l1493_149396

noncomputable def sum_x_i := 80
noncomputable def sum_y_i := 20
noncomputable def sum_x_i_y_i := 184
noncomputable def sum_x_i_sq := 720
noncomputable def n := 10
noncomputable def x_bar := sum_x_i / n
noncomputable def y_bar := sum_y_i / n
noncomputable def b := (sum_x_i_y_i - n * x_bar * y_bar) / (sum_x_i_sq - n * x_bar^2)
noncomputable def a := y_bar - b * x_bar
noncomputable def regression_eqn(x: ℝ) := b * x + a

theorem predict_monthly_savings :
  regression_eqn 7 = 1.7 :=
by
  sorry

end NUMINAMATH_GPT_predict_monthly_savings_l1493_149396


namespace NUMINAMATH_GPT_satisfactory_fraction_l1493_149306

theorem satisfactory_fraction :
  let num_students_A := 8
  let num_students_B := 7
  let num_students_C := 6
  let num_students_D := 5
  let num_students_F := 4
  let satisfactory_grades := num_students_A + num_students_B + num_students_C
  let total_students := num_students_A + num_students_B + num_students_C + num_students_D + num_students_F
  satisfactory_grades / total_students = 7 / 10 :=
by
  let num_students_A := 8
  let num_students_B := 7
  let num_students_C := 6
  let num_students_D := 5
  let num_students_F := 4
  let satisfactory_grades := num_students_A + num_students_B + num_students_C
  let total_students := num_students_A + num_students_B + num_students_C + num_students_D + num_students_F
  have h1: satisfactory_grades = 21 := by sorry
  have h2: total_students = 30 := by sorry
  have fraction := (satisfactory_grades: ℚ) / total_students
  have simplified_fraction := fraction = 7 / 10
  exact sorry

end NUMINAMATH_GPT_satisfactory_fraction_l1493_149306


namespace NUMINAMATH_GPT_harmonic_sum_base_case_l1493_149300

theorem harmonic_sum_base_case : 1 + 1/2 + 1/3 < 2 := 
sorry

end NUMINAMATH_GPT_harmonic_sum_base_case_l1493_149300


namespace NUMINAMATH_GPT_simplify_expression_l1493_149335

variable (x y : ℝ)

theorem simplify_expression :
  (2 * x + 3 * y) ^ 2 - 2 * x * (2 * x - 3 * y) = 18 * x * y + 9 * y ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1493_149335
