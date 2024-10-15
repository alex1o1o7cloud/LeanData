import Mathlib

namespace NUMINAMATH_GPT_triangle_inequality_l2315_231548

theorem triangle_inequality (a : ℝ) :
  (3/2 < a) ∧ (a < 5) ↔ ((4 * a + 1 - (3 * a - 1) < 12 - a) ∧ (4 * a + 1 + (3 * a - 1) > 12 - a)) := 
by 
  sorry

end NUMINAMATH_GPT_triangle_inequality_l2315_231548


namespace NUMINAMATH_GPT_average_monthly_growth_rate_l2315_231544

variable (x : ℝ)

-- Conditions
def turnover_January : ℝ := 36
def turnover_March : ℝ := 48

-- Theorem statement that corresponds to the problem's conditions and question
theorem average_monthly_growth_rate :
  turnover_January * (1 + x)^2 = turnover_March :=
sorry

end NUMINAMATH_GPT_average_monthly_growth_rate_l2315_231544


namespace NUMINAMATH_GPT_midpoint_trajectory_l2315_231516

-- Define the parabola and line intersection conditions
def parabola (x y : ℝ) : Prop := x^2 = 4 * y

def line_through_focus (A B : ℝ × ℝ) (focus : ℝ × ℝ) : Prop :=
  ∃ m b : ℝ, (∀ P ∈ [A, B, focus], P.2 = m * P.1 + b)

def midpoint (A B M : ℝ × ℝ) : Prop :=
  M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

theorem midpoint_trajectory (A B M : ℝ × ℝ) (focus : ℝ × ℝ):
  (parabola A.1 A.2) ∧ (parabola B.1 B.2) ∧ (line_through_focus A B focus) ∧ (midpoint A B M)
  → (M.1 ^ 2 = 2 * M.2 - 2) :=
by
  sorry

end NUMINAMATH_GPT_midpoint_trajectory_l2315_231516


namespace NUMINAMATH_GPT_largest_divisor_consecutive_odd_l2315_231585

theorem largest_divisor_consecutive_odd (m n : ℤ) (h : ∃ k : ℤ, m = 2 * k + 1 ∧ n = 2 * k - 1) :
  ∃ d : ℤ, d = 8 ∧ ∀ m n : ℤ, (∃ k : ℤ, m = 2 * k + 1 ∧ n = 2 * k - 1) → d ∣ (m^2 - n^2) :=
by
  sorry

end NUMINAMATH_GPT_largest_divisor_consecutive_odd_l2315_231585


namespace NUMINAMATH_GPT_max_median_of_pos_integers_l2315_231535

theorem max_median_of_pos_integers
  (k m p r s t u : ℕ)
  (h_avg : (k + m + p + r + s + t + u) / 7 = 24)
  (h_order : k < m ∧ m < p ∧ p < r ∧ r < s ∧ s < t ∧ t < u)
  (h_t : t = 54)
  (h_km_sum : k + m ≤ 20)
  : r ≤ 53 :=
sorry

end NUMINAMATH_GPT_max_median_of_pos_integers_l2315_231535


namespace NUMINAMATH_GPT_find_largest_natural_number_l2315_231586

theorem find_largest_natural_number :
  ∃ n : ℕ, (forall m : ℕ, m > n -> m ^ 300 ≥ 3 ^ 500) ∧ n ^ 300 < 3 ^ 500 :=
  sorry

end NUMINAMATH_GPT_find_largest_natural_number_l2315_231586


namespace NUMINAMATH_GPT_ratio_of_segments_l2315_231518

-- Definitions and conditions as per part (a)
variables (a b c r s : ℝ)
variable (h₁ : a / b = 1 / 3)
variable (h₂ : a^2 = r * c)
variable (h₃ : b^2 = s * c)

-- The statement of the theorem directly addressing part (c)
theorem ratio_of_segments (a b c r s : ℝ) 
  (h₁ : a / b = 1 / 3)
  (h₂ : a^2 = r * c)
  (h₃ : b^2 = s * c) :
  r / s = 1 / 9 :=
  sorry

end NUMINAMATH_GPT_ratio_of_segments_l2315_231518


namespace NUMINAMATH_GPT_find_N_aN_bN_cN_dN_eN_l2315_231527

theorem find_N_aN_bN_cN_dN_eN:
  ∃ (a b c d e : ℝ) (N : ℝ),
    (a > 0) ∧ (b > 0) ∧ (c > 0) ∧ (d > 0) ∧ (e > 0) ∧
    (a^2 + b^2 + c^2 + d^2 + e^2 = 1000) ∧
    (N = c * (a + 3 * b + 4 * d + 6 * e)) ∧
    (N + a + b + c + d + e = 150 + 250 * Real.sqrt 62 + 10 * Real.sqrt 50) := by
  sorry

end NUMINAMATH_GPT_find_N_aN_bN_cN_dN_eN_l2315_231527


namespace NUMINAMATH_GPT_matrix_inverse_eq_l2315_231521

theorem matrix_inverse_eq (d k : ℚ) (A : Matrix (Fin 2) (Fin 2) ℚ) 
  (hA : A = ![![1, 4], ![6, d]]) 
  (hA_inv : A⁻¹ = k • A) :
  (d, k) = (-1, 1/25) :=
  sorry

end NUMINAMATH_GPT_matrix_inverse_eq_l2315_231521


namespace NUMINAMATH_GPT_valentines_count_l2315_231506

theorem valentines_count (x y : ℕ) (h : x * y = x + y + 52) : x * y = 108 :=
by sorry

end NUMINAMATH_GPT_valentines_count_l2315_231506


namespace NUMINAMATH_GPT_smallest_n_gt_15_l2315_231560

theorem smallest_n_gt_15 (n : ℕ) : n ≡ 4 [MOD 6] → n ≡ 3 [MOD 7] → n > 15 → n = 52 :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_gt_15_l2315_231560


namespace NUMINAMATH_GPT_downstream_distance_correct_l2315_231546

-- Definitions based on the conditions
def still_water_speed : ℝ := 22
def stream_speed : ℝ := 5
def travel_time : ℝ := 3

-- The effective speed downstream is the sum of the still water speed and the stream speed
def effective_speed_downstream : ℝ := still_water_speed + stream_speed

-- The distance covered downstream is the product of effective speed and travel time
def downstream_distance : ℝ := effective_speed_downstream * travel_time

-- The theorem to be proven
theorem downstream_distance_correct : downstream_distance = 81 := by
  sorry

end NUMINAMATH_GPT_downstream_distance_correct_l2315_231546


namespace NUMINAMATH_GPT_find_d_l2315_231592

noncomputable def f (x : ℝ) (a b c d : ℝ) : ℝ := x^4 + a*x^3 + b*x^2 + c*x + d

theorem find_d (a b c d : ℝ) (roots_negative_integers : ∀ x, f x a b c d = 0 → x < 0) (sum_is_2023 : a + b + c + d = 2023) :
  d = 17020 :=
sorry

end NUMINAMATH_GPT_find_d_l2315_231592


namespace NUMINAMATH_GPT_total_students_in_class_l2315_231577

theorem total_students_in_class (female_students : ℕ) (male_students : ℕ) (total_students : ℕ) 
  (h1 : female_students = 13) 
  (h2 : male_students = 3 * female_students) 
  (h3 : total_students = female_students + male_students) : 
    total_students = 52 := 
by
  sorry

end NUMINAMATH_GPT_total_students_in_class_l2315_231577


namespace NUMINAMATH_GPT_determine_value_of_y_l2315_231571

variable (s y : ℕ)
variable (h_pos : s > 30)
variable (h_eq : s * s = (s - 15) * (s + y))

theorem determine_value_of_y (h_pos : s > 30) (h_eq : s * s = (s - 15) * (s + y)) : 
  y = 15 * s / (s + 15) :=
by
  sorry

end NUMINAMATH_GPT_determine_value_of_y_l2315_231571


namespace NUMINAMATH_GPT_common_difference_of_arithmetic_sequence_l2315_231580

theorem common_difference_of_arithmetic_sequence (a_n : ℕ → ℤ) (S_n : ℕ → ℤ) (n d a_2 S_3 a_4 : ℤ) 
  (h1 : a_2 + S_3 = -4) (h2 : a_4 = 3)
  (h3 : ∀ n, S_n = n * (a_n + (a_n + (n - 1) * d)) / 2)
  : d = 2 := by
  sorry

end NUMINAMATH_GPT_common_difference_of_arithmetic_sequence_l2315_231580


namespace NUMINAMATH_GPT_simplify_expression_l2315_231597

variables {R : Type*} [CommRing R]

theorem simplify_expression (x y : R) : (3 * x^2 * y^3)^4 = 81 * x^8 * y^12 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l2315_231597


namespace NUMINAMATH_GPT_problem_solution_l2315_231526

theorem problem_solution :
  ∀ (x y : ℚ), 
  4 * x + y = 20 ∧ x + 2 * y = 17 → 
  5 * x^2 + 18 * x * y + 5 * y^2 = 696 + 5/7 := 
by 
  sorry

end NUMINAMATH_GPT_problem_solution_l2315_231526


namespace NUMINAMATH_GPT_expand_binomial_l2315_231530

theorem expand_binomial (x : ℝ) : (x + 3) * (x - 8) = x^2 - 5 * x - 24 :=
by
  sorry

end NUMINAMATH_GPT_expand_binomial_l2315_231530


namespace NUMINAMATH_GPT_baker_made_cakes_l2315_231552

-- Conditions
def cakes_sold : ℕ := 145
def cakes_left : ℕ := 72

-- Question and required proof
theorem baker_made_cakes : (cakes_sold + cakes_left = 217) :=
by
  sorry

end NUMINAMATH_GPT_baker_made_cakes_l2315_231552


namespace NUMINAMATH_GPT_intersection_with_unit_circle_l2315_231568

theorem intersection_with_unit_circle (α : ℝ) : 
    let x := Real.cos (α - Real.pi / 2)
    let y := Real.sin (α - Real.pi / 2)
    (x, y) = (Real.sin α, -Real.cos α) :=
by
  sorry

end NUMINAMATH_GPT_intersection_with_unit_circle_l2315_231568


namespace NUMINAMATH_GPT_mean_value_of_quadrilateral_angles_l2315_231541

theorem mean_value_of_quadrilateral_angles 
  (a1 a2 a3 a4 : ℝ) 
  (h_sum : a1 + a2 + a3 + a4 = 360) :
  (a1 + a2 + a3 + a4) / 4 = 90 :=
by
  sorry

end NUMINAMATH_GPT_mean_value_of_quadrilateral_angles_l2315_231541


namespace NUMINAMATH_GPT_value_of_x_plus_y_l2315_231512

theorem value_of_x_plus_y (x y : ℝ) (h1 : |x| = 3) (h2 : |y| = 4) (h3 : x * y > 0) : x + y = 7 ∨ x + y = -7 :=
by
  sorry

end NUMINAMATH_GPT_value_of_x_plus_y_l2315_231512


namespace NUMINAMATH_GPT_sum_of_integers_l2315_231529

theorem sum_of_integers (n m : ℕ) (h1 : n * (n + 1) = 120) (h2 : (m - 1) * m * (m + 1) = 120) : 
  (n + (n + 1) + (m - 1) + m + (m + 1)) = 36 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_integers_l2315_231529


namespace NUMINAMATH_GPT_calculate_triangle_area_l2315_231595

-- Define the side lengths of the triangle.
def side1 : ℕ := 13
def side2 : ℕ := 13
def side3 : ℕ := 24

-- Define the area calculation.
noncomputable def triangle_area : ℕ := 60

-- Statement of the theorem we wish to prove.
theorem calculate_triangle_area :
  ∃ (a b c : ℕ) (area : ℕ), a = side1 ∧ b = side2 ∧ c = side3 ∧ area = triangle_area :=
sorry

end NUMINAMATH_GPT_calculate_triangle_area_l2315_231595


namespace NUMINAMATH_GPT_find_x_l2315_231591

def bin_op (p1 p2 : ℤ × ℤ) : ℤ × ℤ :=
  (p1.1 - 2 * p2.1, p1.2 + 2 * p2.2)

theorem find_x :
  ∃ x y : ℤ, 
  bin_op (2, -4) (1, -3) = bin_op (x, y) (2, 1) ∧ x = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l2315_231591


namespace NUMINAMATH_GPT_no_nat_nums_x4_minus_y4_eq_x3_plus_y3_l2315_231519

theorem no_nat_nums_x4_minus_y4_eq_x3_plus_y3 : ∀ (x y : ℕ), x^4 - y^4 ≠ x^3 + y^3 :=
by
  intro x y
  sorry

end NUMINAMATH_GPT_no_nat_nums_x4_minus_y4_eq_x3_plus_y3_l2315_231519


namespace NUMINAMATH_GPT_lindsey_savings_in_october_l2315_231582

-- Definitions based on conditions
def savings_september := 50
def savings_november := 11
def spending_video_game := 87
def final_amount_left := 36
def mom_gift := 25

-- The theorem statement
theorem lindsey_savings_in_october (X : ℕ) 
  (h1 : savings_september + X + savings_november > 75) 
  (total_savings := savings_september + X + savings_november + mom_gift) 
  (final_condition : total_savings - spending_video_game = final_amount_left) : 
  X = 37 :=
by
  sorry

end NUMINAMATH_GPT_lindsey_savings_in_october_l2315_231582


namespace NUMINAMATH_GPT_max_n_satisfying_property_l2315_231536

theorem max_n_satisfying_property :
  ∃ n : ℕ, (0 < n) ∧ (∀ m : ℕ, Nat.gcd m n = 1 → m^6 % n = 1) ∧ n = 504 :=
by
  sorry

end NUMINAMATH_GPT_max_n_satisfying_property_l2315_231536


namespace NUMINAMATH_GPT_greatest_k_for_inquality_l2315_231528

theorem greatest_k_for_inquality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a^2 > b*c) :
    (a^2 - b*c)^2 > 4 * ((b^2 - c*a) * (c^2 - a*b)) :=
  sorry

end NUMINAMATH_GPT_greatest_k_for_inquality_l2315_231528


namespace NUMINAMATH_GPT_fraction_inequality_l2315_231584

variable (a b c : ℝ)

theorem fraction_inequality (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : c > a) (h5 : a > b) :
  (a / (c - a)) > (b / (c - b)) := 
sorry

end NUMINAMATH_GPT_fraction_inequality_l2315_231584


namespace NUMINAMATH_GPT_acute_angle_sum_l2315_231505

open Real

theorem acute_angle_sum (α β : ℝ) (hα : 0 < α ∧ α < π / 2)
                        (hβ : 0 < β ∧ β < π / 2)
                        (h1 : 3 * (sin α) ^ 2 + 2 * (sin β) ^ 2 = 1)
                        (h2 : 3 * sin (2 * α) - 2 * sin (2 * β) = 0) :
  α + 2 * β = π / 2 :=
sorry

end NUMINAMATH_GPT_acute_angle_sum_l2315_231505


namespace NUMINAMATH_GPT_otimes_calculation_l2315_231579

def otimes (x y : ℝ) : ℝ := x^2 + y^2

theorem otimes_calculation (x : ℝ) : otimes x (otimes x x) = x^2 + 4 * x^4 :=
by
  sorry

end NUMINAMATH_GPT_otimes_calculation_l2315_231579


namespace NUMINAMATH_GPT_parallelogram_diagonal_length_l2315_231574

-- Define a structure to represent a parallelogram
structure Parallelogram :=
  (side_length : ℝ) 
  (diagonal_length : ℝ)
  (perpendicular : Bool)

-- State the theorem about the relationship between the diagonals in a parallelogram
theorem parallelogram_diagonal_length (a b : ℝ) (P : Parallelogram) (h₀ : P.side_length = a) (h₁ : P.diagonal_length = b) (h₂ : P.perpendicular = true) : 
  ∃ (AC : ℝ), AC = Real.sqrt (4 * a^2 + b^2) :=
by
  sorry

end NUMINAMATH_GPT_parallelogram_diagonal_length_l2315_231574


namespace NUMINAMATH_GPT_three_digit_numbers_count_l2315_231558

theorem three_digit_numbers_count : 
  ∃ (count : ℕ), count = 3 ∧ 
  ∀ (n : ℕ), (100 ≤ n ∧ n < 1000) ∧ 
             (n / 100 = 9) ∧ 
             (∃ a b c, n = 100 * a + 10 * b + c ∧ a + b + c = 27) ∧ 
             (n % 2 = 0) → count = 3 :=
sorry

end NUMINAMATH_GPT_three_digit_numbers_count_l2315_231558


namespace NUMINAMATH_GPT_sheetrock_width_l2315_231540

theorem sheetrock_width (l A w : ℕ) (h_length : l = 6) (h_area : A = 30) (h_formula : A = l * w) : w = 5 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_sheetrock_width_l2315_231540


namespace NUMINAMATH_GPT_Isabella_total_items_l2315_231502

theorem Isabella_total_items (A_pants A_dresses I_pants I_dresses : ℕ) 
  (h1 : A_pants = 3 * I_pants) 
  (h2 : A_dresses = 3 * I_dresses)
  (h3 : A_pants = 21) 
  (h4 : A_dresses = 18) : 
  I_pants + I_dresses = 13 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_Isabella_total_items_l2315_231502


namespace NUMINAMATH_GPT_number_of_children_l2315_231522

theorem number_of_children (C : ℝ) 
  (h1 : 0.30 * C >= 0)
  (h2 : 0.20 * C >= 0)
  (h3 : 0.50 * C >= 0)
  (h4 : 0.70 * C = 42) : 
  C = 60 := by
  sorry

end NUMINAMATH_GPT_number_of_children_l2315_231522


namespace NUMINAMATH_GPT_product_of_five_consecutive_integers_divisible_by_120_l2315_231570

theorem product_of_five_consecutive_integers_divisible_by_120 (n : ℤ) : 
  120 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by 
  sorry

end NUMINAMATH_GPT_product_of_five_consecutive_integers_divisible_by_120_l2315_231570


namespace NUMINAMATH_GPT_f_at_five_l2315_231562

-- Define the function f with the property given in the condition
axiom f : ℝ → ℝ
axiom f_prop : ∀ x : ℝ, f (3 * x - 1) = x^2 + x + 1

-- Prove that f(5) = 7 given the properties above
theorem f_at_five : f 5 = 7 :=
by
  sorry

end NUMINAMATH_GPT_f_at_five_l2315_231562


namespace NUMINAMATH_GPT_container_volume_ratio_l2315_231566

variable (A B C : ℝ)

theorem container_volume_ratio (h1 : (4 / 5) * A = (3 / 5) * B) (h2 : (3 / 5) * B = (3 / 4) * C) :
  A / C = 15 / 16 :=
sorry

end NUMINAMATH_GPT_container_volume_ratio_l2315_231566


namespace NUMINAMATH_GPT_value_of_x_l2315_231538

theorem value_of_x (x : ℝ) (h : x = -x) : x = 0 := 
by 
  sorry

end NUMINAMATH_GPT_value_of_x_l2315_231538


namespace NUMINAMATH_GPT_find_x_l2315_231561

variable (c d : ℝ)

theorem find_x (x : ℝ) (h : x^2 + 4 * c^2 = (3 * d - x)^2) : 
  x = (9 * d^2 - 4 * c^2) / (6 * d) :=
sorry

end NUMINAMATH_GPT_find_x_l2315_231561


namespace NUMINAMATH_GPT_valid_permutations_l2315_231539

theorem valid_permutations (a : Fin 101 → ℕ) :
  (∀ k, a k ≥ 2 ∧ a k ≤ 102 ∧ (∃ j, a j = k + 2)) →
  (∀ k, a (k + 1) % (k + 1) = 0) →
  (∃ cycles : List (List ℕ), cycles = [[1, 102], [1, 2, 102], [1, 3, 102], [1, 6, 102], [1, 17, 102], [1, 34, 102], 
                                       [1, 51, 102], [1, 2, 6, 102], [1, 2, 34, 102], [1, 3, 6, 102], [1, 3, 51, 102], 
                                       [1, 17, 34, 102], [1, 17, 51, 102]]) :=
sorry

end NUMINAMATH_GPT_valid_permutations_l2315_231539


namespace NUMINAMATH_GPT_maximize_expression_l2315_231578

theorem maximize_expression :
  ∀ (a b c d e : ℕ),
    a ≠ b → a ≠ c → a ≠ d → a ≠ e → b ≠ c → b ≠ d → b ≠ e → c ≠ d → c ≠ e → d ≠ e →
    (a = 1 ∨ a = 2 ∨ a = 3 ∨ a = 4 ∨ a = 6) → 
    (b = 1 ∨ b = 2 ∨ b = 3 ∨ b = 4 ∨ b = 6) →
    (c = 1 ∨ c = 2 ∨ c = 3 ∨ c = 4 ∨ c = 6) →
    (d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 6) →
    (e = 1 ∨ e = 2 ∨ e = 3 ∨ e = 4 ∨ e = 6) →
    ((a : ℚ) / 2 + (d : ℚ) / e * (c / b)) ≤ 9 :=
by
  sorry

end NUMINAMATH_GPT_maximize_expression_l2315_231578


namespace NUMINAMATH_GPT_max_side_length_triangle_l2315_231587

def triangle_with_max_side_length (a b c : ℕ) (ha : a ≠ b ∧ b ≠ c ∧ c ≠ a) (hper : a + b + c = 30) : Prop :=
  a > b ∧ a > c ∧ a = 14

theorem max_side_length_triangle : ∃ a b c : ℕ, 
  a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ a + b + c = 30 ∧ a > b ∧ a > c ∧ a = 14 :=
sorry

end NUMINAMATH_GPT_max_side_length_triangle_l2315_231587


namespace NUMINAMATH_GPT_mika_saucer_surface_area_l2315_231531

noncomputable def surface_area_saucer (r h rim_thickness : ℝ) : ℝ :=
  let A_cap := 2 * Real.pi * r * h  -- Surface area of the spherical cap
  let R_outer := r
  let R_inner := r - rim_thickness
  let A_rim := Real.pi * (R_outer^2 - R_inner^2)  -- Area of the rim
  A_cap + A_rim

theorem mika_saucer_surface_area :
  surface_area_saucer 3 1.5 1 = 14 * Real.pi :=
sorry

end NUMINAMATH_GPT_mika_saucer_surface_area_l2315_231531


namespace NUMINAMATH_GPT_max_tulips_l2315_231581

theorem max_tulips (r y n : ℕ) (hr : r = y + 1) (hc : 50 * y + 31 * r ≤ 600) :
  r + y = 2 * y + 1 → n = 2 * y + 1 → n = 15 :=
by
  -- Given the constraints, we need to identify the maximum n given the costs and relationships.
  sorry

end NUMINAMATH_GPT_max_tulips_l2315_231581


namespace NUMINAMATH_GPT_gcd_polynomial_l2315_231590

theorem gcd_polynomial (b : ℕ) (h : 570 ∣ b) : Nat.gcd (5 * b^3 + 2 * b^2 + 5 * b + 95) b = 95 :=
by
  sorry

end NUMINAMATH_GPT_gcd_polynomial_l2315_231590


namespace NUMINAMATH_GPT_num_common_elements_1000_multiples_5_9_l2315_231554

def multiples_up_to (n k : ℕ) : ℕ := n / k

def num_common_elements_in_sets (k m n : ℕ) : ℕ :=
  multiples_up_to n (Nat.lcm k m)

theorem num_common_elements_1000_multiples_5_9 :
  num_common_elements_in_sets 5 9 5000 = 111 :=
by
  -- The proof is omitted as per instructions
  sorry

end NUMINAMATH_GPT_num_common_elements_1000_multiples_5_9_l2315_231554


namespace NUMINAMATH_GPT_find_m_l2315_231523

noncomputable def m_value (a b c d : Int) (Y : Int) : Int :=
  let l1_1 := a + b
  let l1_2 := b + c
  let l1_3 := c + d
  let l2_1 := l1_1 + l1_2
  let l2_2 := l1_2 + l1_3
  let l3 := l2_1 + l2_2
  if l3 = Y then a else 0

theorem find_m : m_value m 6 (-3) 4 20 = 7 := sorry

end NUMINAMATH_GPT_find_m_l2315_231523


namespace NUMINAMATH_GPT_max_area_inscribed_triangle_l2315_231515

/-- Let ΔABC be an inscribed triangle in the ellipse given by the equation
    (x^2 / 9) + (y^2 / 4) = 1, where the line segment AB passes through the 
    point (1, 0). Prove that the maximum area of ΔABC is (16 * sqrt 2) / 3. --/
theorem max_area_inscribed_triangle
  (A B C : ℝ × ℝ) 
  (hA : (A.1 ^ 2) / 9 + (A.2 ^ 2) / 4 = 1)
  (hB : (B.1 ^ 2) / 9 + (B.2 ^ 2) / 4 = 1)
  (hC : (C.1 ^ 2) / 9 + (C.2 ^ 2) / 4 = 1)
  (hAB : ∃ n : ℝ, ∀ x y : ℝ, (x, y) ∈ [A, B] → x = n * y + 1)
  : ∃ S : ℝ, S = ((16 : ℝ) * Real.sqrt 2) / 3 :=
sorry

end NUMINAMATH_GPT_max_area_inscribed_triangle_l2315_231515


namespace NUMINAMATH_GPT_john_splits_profit_correctly_l2315_231593

-- Conditions
def total_cookies : ℕ := 6 * 12
def revenue_per_cookie : ℝ := 1.5
def cost_per_cookie : ℝ := 0.25
def amount_per_charity : ℝ := 45

-- Computations based on conditions
def total_revenue : ℝ := total_cookies * revenue_per_cookie
def total_cost : ℝ := total_cookies * cost_per_cookie
def total_profit : ℝ := total_revenue - total_cost

-- Proof statement
theorem john_splits_profit_correctly : total_profit / amount_per_charity = 2 := by
  sorry

end NUMINAMATH_GPT_john_splits_profit_correctly_l2315_231593


namespace NUMINAMATH_GPT_find_second_number_l2315_231596

theorem find_second_number (a : ℕ) (c : ℕ) (x : ℕ) : 
  3 * a + 3 * x + 3 * c + 11 = 170 → a = 16 → c = 20 → x = 17 := 
by
  intros h1 h2 h3
  rw [h2, h3] at h1
  simp at h1
  sorry

end NUMINAMATH_GPT_find_second_number_l2315_231596


namespace NUMINAMATH_GPT_parallel_vectors_result_l2315_231537

noncomputable def a : ℝ × ℝ := (1, -2)
noncomputable def b (m : ℝ) : ℝ × ℝ := (m, 4)
noncomputable def m : ℝ := -1 / 2

theorem parallel_vectors_result :
  (b m).1 * a.2 = (b m).2 * a.1 →
  2 * a - b m = (4, -8) :=
by
  intro h
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_parallel_vectors_result_l2315_231537


namespace NUMINAMATH_GPT_cannot_be_20182017_l2315_231594

theorem cannot_be_20182017 (a b : ℤ) (h : a * b * (a + b) = 20182017) : False :=
by
  sorry

end NUMINAMATH_GPT_cannot_be_20182017_l2315_231594


namespace NUMINAMATH_GPT_total_time_on_road_l2315_231520

def driving_time_day1 (jade_time krista_time krista_delay lunch_break : ℝ) : ℝ :=
  jade_time + krista_time + krista_delay + lunch_break

def driving_time_day2 (jade_time krista_time break_time krista_refuel lunch_break : ℝ) : ℝ :=
  jade_time + krista_time + break_time + krista_refuel + lunch_break

def driving_time_day3 (jade_time krista_time krista_delay lunch_break : ℝ) : ℝ :=
  jade_time + krista_time + krista_delay + lunch_break

def total_driving_time (day1 day2 day3 : ℝ) : ℝ :=
  day1 + day2 + day3

theorem total_time_on_road :
  total_driving_time 
    (driving_time_day1 8 6 1 1) 
    (driving_time_day2 7 5 0.5 (1/3) 1) 
    (driving_time_day3 6 4 1 1) 
  = 42.3333 := 
  by 
    sorry

end NUMINAMATH_GPT_total_time_on_road_l2315_231520


namespace NUMINAMATH_GPT_sequence_sum_l2315_231511

theorem sequence_sum (a : ℕ → ℝ) (h_pos : ∀ n, a n > 0)
    (h1 : a 1 = 1)
    (h_rec : ∀ n, a (n + 2) = 1 / (a n + 1))
    (h6_2 : a 6 = a 2) :
    a 2016 + a 3 = (Real.sqrt 5) / 2 :=
by
  sorry

end NUMINAMATH_GPT_sequence_sum_l2315_231511


namespace NUMINAMATH_GPT_strawberries_weight_l2315_231545

theorem strawberries_weight (marco_weight dad_increase : ℕ) (h_marco: marco_weight = 30) (h_diff: marco_weight = dad_increase + 13) : marco_weight + (marco_weight - 13) = 47 :=
by
  sorry

end NUMINAMATH_GPT_strawberries_weight_l2315_231545


namespace NUMINAMATH_GPT_bailey_dog_treats_l2315_231555

-- Definitions based on conditions
def total_charges_per_card : Nat := 5
def number_of_cards : Nat := 4
def chew_toys : Nat := 2
def rawhide_bones : Nat := 10

-- Total number of items bought
def total_items : Nat := total_charges_per_card * number_of_cards

-- Definition of the number of dog treats
def dog_treats : Nat := total_items - (chew_toys + rawhide_bones)

-- Theorem to prove the number of dog treats
theorem bailey_dog_treats : dog_treats = 8 := by
  -- Proof is skipped with sorry
  sorry

end NUMINAMATH_GPT_bailey_dog_treats_l2315_231555


namespace NUMINAMATH_GPT_min_max_value_l2315_231501

theorem min_max_value
  (x₁ x₂ x₃ x₄ x₅ : ℝ)
  (h₁ : 0 ≤ x₁) (h₂ : 0 ≤ x₂) (h₃ : 0 ≤ x₃) (h₄ : 0 ≤ x₄) (h₅ : 0 ≤ x₅)
  (h_sum : x₁ + x₂ + x₃ + x₄ + x₅ = 1) :
  (min (max (x₁ + x₂) (max (x₂ + x₃) (max (x₃ + x₄) (x₄ + x₅)))) = 1 / 3) :=
sorry

end NUMINAMATH_GPT_min_max_value_l2315_231501


namespace NUMINAMATH_GPT_half_of_number_l2315_231500

theorem half_of_number (x : ℝ) (h : (4 / 15 * 5 / 7 * x - 4 / 9 * 2 / 5 * x = 8)) : (1 / 2 * x = 315) :=
sorry

end NUMINAMATH_GPT_half_of_number_l2315_231500


namespace NUMINAMATH_GPT_f_odd_solve_inequality_l2315_231569

noncomputable def f : ℝ → ℝ := sorry

axiom f_additive : ∀ x y : ℝ, f (x + y) = f x + f y
axiom f_positive : ∀ x : ℝ, x > 0 → f x > 0

theorem f_odd : ∀ x : ℝ, f (-x) = -f x := 
by
  sorry

theorem solve_inequality : {a : ℝ | f (a-4) + f (2*a+1) < 0} = {a | a < 1} := 
by
  sorry

end NUMINAMATH_GPT_f_odd_solve_inequality_l2315_231569


namespace NUMINAMATH_GPT_crease_length_l2315_231503

theorem crease_length (AB : ℝ) (h₁ : AB = 15)
  (h₂ : ∀ (area : ℝ) (folded_area : ℝ), folded_area = 0.25 * area) :
  ∃ (DE : ℝ), DE = 0.5 * AB :=
by
  use 7.5 -- DE
  sorry

end NUMINAMATH_GPT_crease_length_l2315_231503


namespace NUMINAMATH_GPT_always_composite_l2315_231588

theorem always_composite (p : ℕ) (hp : Nat.Prime p) : ¬Nat.Prime (p^2 + 35) ∧ ¬Nat.Prime (p^2 + 55) :=
by
  sorry

end NUMINAMATH_GPT_always_composite_l2315_231588


namespace NUMINAMATH_GPT_incident_reflected_eqs_l2315_231564

theorem incident_reflected_eqs {x y : ℝ} :
  (∃ (A B : ℝ × ℝ), A = (2, 3) ∧ B = (1, 1) ∧ 
   (∀ (P : ℝ × ℝ), (P = A ∨ P = B → (P.1 + P.2 + 1 = 0) → false)) ∧
   (∃ (line_inc line_ref : ℝ × ℝ × ℝ),
     line_inc = (5, -4, 2) ∧
     line_ref = (4, -5, 1))) :=
sorry

end NUMINAMATH_GPT_incident_reflected_eqs_l2315_231564


namespace NUMINAMATH_GPT_solve_equation_l2315_231575

theorem solve_equation (x : ℝ) : 
  3 * x * (x - 1) = 2 * x - 2 ↔ x = 1 ∨ x = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l2315_231575


namespace NUMINAMATH_GPT_divisor_is_13_l2315_231556

theorem divisor_is_13 (N D : ℕ) (h1 : N = 32) (h2 : (N - 6) / D = 2) : D = 13 := by
  sorry

end NUMINAMATH_GPT_divisor_is_13_l2315_231556


namespace NUMINAMATH_GPT_average_salary_correct_l2315_231504

/-- The salaries of A, B, C, D, and E. -/
def salary_A : ℕ := 8000
def salary_B : ℕ := 5000
def salary_C : ℕ := 11000
def salary_D : ℕ := 7000
def salary_E : ℕ := 9000

/-- The number of people. -/
def number_of_people : ℕ := 5

/-- The total salary is the sum of the salaries. -/
def total_salary : ℕ := salary_A + salary_B + salary_C + salary_D + salary_E

/-- The average salary is the total salary divided by the number of people. -/
def average_salary : ℕ := total_salary / number_of_people

/-- The average salary of A, B, C, D, and E is Rs. 8000. -/
theorem average_salary_correct : average_salary = 8000 := by
  sorry

end NUMINAMATH_GPT_average_salary_correct_l2315_231504


namespace NUMINAMATH_GPT_rainy_days_l2315_231551

theorem rainy_days (n R NR : ℤ) 
  (h1 : n * R + 4 * NR = 26)
  (h2 : 4 * NR - n * R = 14)
  (h3 : R + NR = 7) : 
  R = 2 := 
sorry

end NUMINAMATH_GPT_rainy_days_l2315_231551


namespace NUMINAMATH_GPT_time_to_cross_first_platform_l2315_231550

noncomputable def train_length : ℝ := 30
noncomputable def first_platform_length : ℝ := 180
noncomputable def second_platform_length : ℝ := 250
noncomputable def time_second_platform : ℝ := 20

noncomputable def train_speed : ℝ :=
(train_length + second_platform_length) / time_second_platform

noncomputable def time_first_platform : ℝ :=
(train_length + first_platform_length) / train_speed

theorem time_to_cross_first_platform :
  time_first_platform = 15 :=
by
  sorry

end NUMINAMATH_GPT_time_to_cross_first_platform_l2315_231550


namespace NUMINAMATH_GPT_range_of_m_l2315_231589

open Set

def setM (m : ℝ) : Set ℝ := { x | x ≤ m }
def setP : Set ℝ := { x | x ≥ -1 }

theorem range_of_m (m : ℝ) (h : setM m ∩ setP = ∅) : m < -1 := sorry

end NUMINAMATH_GPT_range_of_m_l2315_231589


namespace NUMINAMATH_GPT_max_a_b_l2315_231543

theorem max_a_b (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) (h_eq : a^2 + b^2 = 1) : a + b ≤ Real.sqrt 2 := sorry

end NUMINAMATH_GPT_max_a_b_l2315_231543


namespace NUMINAMATH_GPT_arithmetic_sequence_property_l2315_231510

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (h1 : ((a 6 - 1)^3 + 2013 * (a 6 - 1)^3 = 1))
  (h2 : ((a 2008 - 1)^3 = -2013 * (a 2008 - 1)^3))
  (sum_formula : ∀ n, S n = n * a n) : 
  S 2013 = 2013 ∧ a 2008 < a 6 := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_property_l2315_231510


namespace NUMINAMATH_GPT_four_sq_geq_prod_sum_l2315_231509

variable {α : Type*} [LinearOrderedField α]

theorem four_sq_geq_prod_sum (a b c d : α) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a^2 + b^2 + c^2 + d^2)^2 ≥ (a + b) * (b + c) * (c + d) * (d + a) :=
sorry

end NUMINAMATH_GPT_four_sq_geq_prod_sum_l2315_231509


namespace NUMINAMATH_GPT_profit_margin_in_terms_of_retail_price_l2315_231573

theorem profit_margin_in_terms_of_retail_price
  (k c P_R : ℝ) (h1 : ∀ C, P = k * C) (h2 : ∀ C, P_R = c * (P + C)) :
  P = (k / (c * (k + 1))) * P_R :=
by sorry

end NUMINAMATH_GPT_profit_margin_in_terms_of_retail_price_l2315_231573


namespace NUMINAMATH_GPT_linear_relationship_selling_price_maximize_profit_l2315_231525

theorem linear_relationship (k b : ℝ)
  (h₁ : 36 = 12 * k + b)
  (h₂ : 34 = 13 * k + b) :
  y = -2 * x + 60 :=
by
  sorry

theorem selling_price (p c x : ℝ)
  (h₁ : x ≥ 10)
  (h₂ : x ≤ 19)
  (h₃ : x - 10 = (192 / (y + 10))) :
  x = 18 :=
by
  sorry

theorem maximize_profit (x w : ℝ)
  (h_max : x = 19)
  (h_profit : w = -2 * x^2 + 80 * x - 600) :
  w = 198 :=
by
  sorry

end NUMINAMATH_GPT_linear_relationship_selling_price_maximize_profit_l2315_231525


namespace NUMINAMATH_GPT_largest_divisor_n4_minus_n2_l2315_231576

theorem largest_divisor_n4_minus_n2 (n : ℤ) : 12 ∣ (n^4 - n^2) :=
by
  sorry

end NUMINAMATH_GPT_largest_divisor_n4_minus_n2_l2315_231576


namespace NUMINAMATH_GPT_arithmetic_sequence_product_l2315_231508

theorem arithmetic_sequence_product (b : ℕ → ℤ) (d : ℤ) (h1 : ∀ n m, n < m → b n < b m) 
(h2 : ∀ n, b (n + 1) - b n = d) (h3 : b 3 * b 4 = 18) : b 2 * b 5 = -80 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_product_l2315_231508


namespace NUMINAMATH_GPT_boarders_joined_l2315_231599

theorem boarders_joined (initial_boarders : ℕ) (initial_day_scholars : ℕ) (final_ratio_num : ℕ) (final_ratio_denom : ℕ) (new_boarders : ℕ)
  (initial_ratio_boarders_to_day_scholars : initial_boarders * 16 = 7 * initial_day_scholars)
  (initial_boarders_eq : initial_boarders = 560)
  (final_ratio : (initial_boarders + new_boarders) * 2 = final_day_scholars)
  (day_scholars_eq : initial_day_scholars = 1280) : 
  new_boarders = 80 := by
  sorry

end NUMINAMATH_GPT_boarders_joined_l2315_231599


namespace NUMINAMATH_GPT_inscribed_sphere_radius_l2315_231532

variable (a b r : ℝ)

theorem inscribed_sphere_radius (ha : 0 < a) (hb : 0 < b) (hr : 0 < r)
 (h : ∃ A B C D : ℝˣ, true) : r < (a * b) / (2 * (a + b)) := 
sorry

end NUMINAMATH_GPT_inscribed_sphere_radius_l2315_231532


namespace NUMINAMATH_GPT_max_green_socks_l2315_231583

theorem max_green_socks (g y : ℕ) (h_t : g + y ≤ 2000) (h_prob : (g * (g - 1) + y * (y - 1) = (g + y) * (g + y - 1) / 3)) :
  g ≤ 19 := by
  sorry

end NUMINAMATH_GPT_max_green_socks_l2315_231583


namespace NUMINAMATH_GPT_simplify_expression_l2315_231533

theorem simplify_expression :
  1 + (1 / (1 + Real.sqrt 2)) - (1 / (1 - Real.sqrt 5)) =
  1 + ((-Real.sqrt 2 - Real.sqrt 5) / (1 + Real.sqrt 2 - Real.sqrt 5 - Real.sqrt 10)) :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l2315_231533


namespace NUMINAMATH_GPT_math_problem_l2315_231553

theorem math_problem 
  (x y : ℝ) 
  (h1 : x + y = -5) 
  (h2 : x * y = 3) :
  x * Real.sqrt (y / x) + y * Real.sqrt (x / y) = -2 * Real.sqrt 3 := 
sorry

end NUMINAMATH_GPT_math_problem_l2315_231553


namespace NUMINAMATH_GPT_minimum_value_of_abs_phi_l2315_231572

theorem minimum_value_of_abs_phi (φ : ℝ) :
  (∃ k : ℤ, φ = k * π - (13 * π) / 6) → 
  ∃ φ_min : ℝ, 0 ≤ φ_min ∧ φ_min = abs φ ∧ φ_min = π / 6 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_abs_phi_l2315_231572


namespace NUMINAMATH_GPT_vector_sum_solve_for_m_n_l2315_231567

-- Define the vectors
def a : ℝ × ℝ := (3, 2)
def b : ℝ × ℝ := (-1, 2)
def c : ℝ × ℝ := (4, 1)

-- Problem 1: Vector sum
theorem vector_sum : 3 • a + b - 2 • c = (0, 6) :=
by sorry

-- Problem 2: Solving for m and n
theorem solve_for_m_n (m n : ℝ) (hm : a = m • b + n • c) :
  m = 5 / 9 ∧ n = 8 / 9 :=
by sorry

end NUMINAMATH_GPT_vector_sum_solve_for_m_n_l2315_231567


namespace NUMINAMATH_GPT_problem_equivalent_l2315_231542

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x^5 + a * x^3 + b * x - 8

theorem problem_equivalent (a b : ℝ) (h : f (-2) a b = 10) : f 2 a b = -10 :=
by
  sorry

end NUMINAMATH_GPT_problem_equivalent_l2315_231542


namespace NUMINAMATH_GPT_base7_to_base10_245_l2315_231513

theorem base7_to_base10_245 : (2 * 7^2 + 4 * 7^1 + 5 * 7^0) = 131 := by
  sorry

end NUMINAMATH_GPT_base7_to_base10_245_l2315_231513


namespace NUMINAMATH_GPT_multiply_decimals_l2315_231514

noncomputable def real_num_0_7 : ℝ := 7 * 10⁻¹
noncomputable def real_num_0_3 : ℝ := 3 * 10⁻¹
noncomputable def real_num_0_21 : ℝ := 0.21

theorem multiply_decimals :
  real_num_0_7 * real_num_0_3 = real_num_0_21 :=
sorry

end NUMINAMATH_GPT_multiply_decimals_l2315_231514


namespace NUMINAMATH_GPT_tips_multiple_l2315_231559

variable (A T : ℝ) (x : ℝ)
variable (h1 : T = 7 * A)
variable (h2 : T / 4 = x * A)

theorem tips_multiple (A T : ℝ) (x : ℝ) (h1 : T = 7 * A) (h2 : T / 4 = x * A) : x = 1.75 := by
  sorry

end NUMINAMATH_GPT_tips_multiple_l2315_231559


namespace NUMINAMATH_GPT_total_mail_l2315_231557

def monday_mail := 65
def tuesday_mail := monday_mail + 10
def wednesday_mail := tuesday_mail - 5
def thursday_mail := wednesday_mail + 15

theorem total_mail : 
  monday_mail + tuesday_mail + wednesday_mail + thursday_mail = 295 := by
  sorry

end NUMINAMATH_GPT_total_mail_l2315_231557


namespace NUMINAMATH_GPT_determine_m_l2315_231507

theorem determine_m (m : ℝ) : (∀ x : ℝ, (m * x = 1 → x = 1 ∨ x = -1)) ↔ (m = 0 ∨ m = 1 ∨ m = -1) :=
by sorry

end NUMINAMATH_GPT_determine_m_l2315_231507


namespace NUMINAMATH_GPT_a6_equals_8_l2315_231549

-- Defining Sn as given in the condition
def S (n : ℕ) : ℤ :=
  if n = 0 then 0
  else n^2 - 3*n

-- Defining a_n in terms of the differences stated in the solution
def a (n : ℕ) : ℤ := S n - S (n-1)

-- The problem statement to prove
theorem a6_equals_8 : a 6 = 8 :=
by
  sorry

end NUMINAMATH_GPT_a6_equals_8_l2315_231549


namespace NUMINAMATH_GPT_mod_exp_sub_l2315_231565

theorem mod_exp_sub (a b k : ℕ) (h₁ : a ≡ 6 [MOD 7]) (h₂ : b ≡ 4 [MOD 7]) :
  (a ^ k - b ^ k) % 7 = 2 :=
sorry

end NUMINAMATH_GPT_mod_exp_sub_l2315_231565


namespace NUMINAMATH_GPT_max_marks_is_400_l2315_231547

-- Given conditions
def passing_mark (M : ℝ) : ℝ := 0.30 * M
def student_marks : ℝ := 80
def marks_failed_by : ℝ := 40
def pass_marks : ℝ := student_marks + marks_failed_by

-- Statement to prove
theorem max_marks_is_400 (M : ℝ) (h : passing_mark M = pass_marks) : M = 400 :=
by sorry

end NUMINAMATH_GPT_max_marks_is_400_l2315_231547


namespace NUMINAMATH_GPT_floor_inequality_solution_set_l2315_231524

/-- Let ⌊x⌋ denote the greatest integer less than or equal to x.
    Prove that the solution set of the inequality ⌊x⌋² - 5⌊x⌋ - 36 ≤ 0 is {x | -4 ≤ x < 10}. -/
theorem floor_inequality_solution_set (x : ℝ) :
  (⌊x⌋^2 - 5 * ⌊x⌋ - 36 ≤ 0) ↔ -4 ≤ x ∧ x < 10 := by
    sorry

end NUMINAMATH_GPT_floor_inequality_solution_set_l2315_231524


namespace NUMINAMATH_GPT_faye_pencils_l2315_231534

theorem faye_pencils :
  ∀ (packs : ℕ) (pencils_per_pack : ℕ) (rows : ℕ) (total_pencils pencils_per_row : ℕ),
  packs = 35 →
  pencils_per_pack = 4 →
  rows = 70 →
  total_pencils = packs * pencils_per_pack →
  pencils_per_row = total_pencils / rows →
  pencils_per_row = 2 :=
by
  intros packs pencils_per_pack rows total_pencils pencils_per_row
  intros packs_eq pencils_per_pack_eq rows_eq total_pencils_eq pencils_per_row_eq
  sorry

end NUMINAMATH_GPT_faye_pencils_l2315_231534


namespace NUMINAMATH_GPT_grunters_win_4_out_of_6_l2315_231563

/-- The Grunters have a probability of winning any given game as 60% --/
def p : ℚ := 3 / 5

/-- The Grunters have a probability of losing any given game as 40% --/
def q : ℚ := 1 - p

/-- The binomial coefficient for choosing exactly 4 wins out of 6 games --/
def binomial_6_4 : ℚ := Nat.choose 6 4

/-- The probability that the Grunters win exactly 4 out of the 6 games --/
def prob_4_wins : ℚ := binomial_6_4 * (p ^ 4) * (q ^ 2)

/--
The probability that the Grunters win exactly 4 out of the 6 games is
exactly $\frac{4860}{15625}$.
--/
theorem grunters_win_4_out_of_6 : prob_4_wins = 4860 / 15625 := by
  sorry

end NUMINAMATH_GPT_grunters_win_4_out_of_6_l2315_231563


namespace NUMINAMATH_GPT_problem_l2315_231598

noncomputable def trajectory_C (x y : ℝ) : Prop :=
  y^2 = -8 * x

theorem problem (P : ℝ × ℝ) (k : ℝ) (h : -1 < k ∧ k < 0) 
  (H1 : P.1 = -2 ∨ P.1 = 2)
  (H2 : trajectory_C P.1 P.2) :
  ∃ Q : ℝ × ℝ, Q.1 < -6 :=
  sorry

end NUMINAMATH_GPT_problem_l2315_231598


namespace NUMINAMATH_GPT_line_AB_eq_x_plus_3y_zero_l2315_231517

variable (x y : ℝ)

def circle1 := x^2 + y^2 - 4*x + 6*y = 0
def circle2 := x^2 + y^2 - 6*x = 0

theorem line_AB_eq_x_plus_3y_zero : 
  (∃ (A B : ℝ × ℝ), circle1 A.1 A.2 ∧ circle1 B.1 B.2 ∧ circle2 A.1 A.2 ∧ circle2 B.1 B.2 ∧ (A ≠ B)) → 
  (∀ (x y : ℝ), x + 3*y = 0) := 
by
  sorry

end NUMINAMATH_GPT_line_AB_eq_x_plus_3y_zero_l2315_231517
