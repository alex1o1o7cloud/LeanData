import Mathlib

namespace product_4_6_7_14_l1388_138806

theorem product_4_6_7_14 : 4 * 6 * 7 * 14 = 2352 := by
  sorry

end product_4_6_7_14_l1388_138806


namespace find_stickers_before_birthday_l1388_138835

variable (stickers_received : ℕ) (total_stickers : ℕ)

def stickers_before_birthday (stickers_received total_stickers : ℕ) : ℕ :=
  total_stickers - stickers_received

theorem find_stickers_before_birthday (h1 : stickers_received = 22) (h2 : total_stickers = 61) : 
  stickers_before_birthday stickers_received total_stickers = 39 :=
by 
  have h1 : stickers_received = 22 := h1
  have h2 : total_stickers = 61 := h2
  rw [h1, h2]
  rfl

end find_stickers_before_birthday_l1388_138835


namespace sampling_method_systematic_l1388_138888

theorem sampling_method_systematic 
  (inspect_interval : ℕ := 10)
  (products_interval : ℕ := 10)
  (position : ℕ) :
  inspect_interval = 10 ∧ products_interval = 10 → 
  (sampling_method = "Systematic Sampling") :=
by
  sorry

end sampling_method_systematic_l1388_138888


namespace find_y_value_l1388_138856

theorem find_y_value (t : ℝ) (x : ℝ) (y : ℝ)
  (h1 : x = 3 - 2 * t)
  (h2 : y = 3 * t + 6)
  (h3 : x = -6)
  : y = 19.5 :=
by {
  sorry
}

end find_y_value_l1388_138856


namespace polynomial_value_at_3_l1388_138870

def f (x : ℝ) : ℝ := 4*x^5 + 2*x^4 + 3.5*x^3 - 2.6*x^2 + 1.7*x - 0.8

theorem polynomial_value_at_3 : f 3 = 1209.4 := 
by
  sorry

end polynomial_value_at_3_l1388_138870


namespace pos_solution_sum_l1388_138883

theorem pos_solution_sum (c d : ℕ) (h_c_pos : 0 < c) (h_d_pos : 0 < d) :
  (∃ x : ℝ, x ^ 2 + 16 * x = 100 ∧ x = Real.sqrt c - d) → c + d = 172 :=
by
  intro h
  sorry

end pos_solution_sum_l1388_138883


namespace matrix_projection_ratios_l1388_138807

theorem matrix_projection_ratios (x y z : ℚ) (h : 
  (1 / 14 : ℚ) * x - (5 / 14 : ℚ) * y = x ∧
  - (5 / 14 : ℚ) * x + (24 / 14 : ℚ) * y = y ∧
  0 * x + 0 * y + 1 * z = z)
  : y / x = 13 / 5 ∧ z / x = 1 := 
by 
  sorry

end matrix_projection_ratios_l1388_138807


namespace f_even_l1388_138831

noncomputable def f : ℝ → ℝ := sorry

axiom f_not_identically_zero : ∃ x : ℝ, f x ≠ 0

axiom f_functional_eqn : ∀ a b : ℝ, 
  f (a + b) + f (a - b) = 2 * f a + 2 * f b

theorem f_even (x : ℝ) : f (-x) = f x :=
  sorry

end f_even_l1388_138831


namespace angela_problems_l1388_138863

theorem angela_problems (total_problems martha_problems : ℕ) (jenna_problems mark_problems : ℕ) 
    (h1 : total_problems = 20) 
    (h2 : martha_problems = 2)
    (h3 : jenna_problems = 4 * martha_problems - 2)
    (h4 : mark_problems = jenna_problems / 2) :
    total_problems - (martha_problems + jenna_problems + mark_problems) = 9 := 
sorry

end angela_problems_l1388_138863


namespace binom_12_10_eq_66_l1388_138887

noncomputable def binom (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binom_12_10_eq_66 : binom 12 10 = 66 := 
by
  -- We state the given condition for binomial coefficients.
  let binom_coeff := binom 12 10
  -- We will use the formula \binom{n}{k} = \frac{n!}{k!(n-k)!}.
  change binom_coeff = Nat.factorial 12 / (Nat.factorial 10 * Nat.factorial (12 - 10))
  -- Simplify using factorial values and basic arithmetic to reach the conclusion.
  sorry

end binom_12_10_eq_66_l1388_138887


namespace additional_width_is_25cm_l1388_138899

-- Definitions
def length_of_room_cm := 5000
def width_of_room_cm := 1100
def additional_width_cm := 25
def number_of_tiles := 9000
def side_length_of_tile_cm := 25

-- Statement to prove
theorem additional_width_is_25cm : additional_width_cm = 25 :=
by
  -- The proof is omitted, we assume the proof steps here
  sorry

end additional_width_is_25cm_l1388_138899


namespace find_q_l1388_138809

def f (q : ℝ) : ℝ := 3 * q - 3

theorem find_q (q : ℝ) : f (f q) = 210 → q = 74 / 3 := by
  sorry

end find_q_l1388_138809


namespace bob_weekly_income_increase_l1388_138890

theorem bob_weekly_income_increase
  (raise_per_hour : ℝ)
  (hours_per_week : ℝ)
  (benefit_reduction_per_month : ℝ)
  (weeks_per_month : ℝ)
  (h_raise : raise_per_hour = 0.50)
  (h_hours : hours_per_week = 40)
  (h_reduction : benefit_reduction_per_month = 60)
  (h_weeks : weeks_per_month = 4.33) :
  (raise_per_hour * hours_per_week - benefit_reduction_per_month / weeks_per_month) = 6.14 :=
by
  simp [h_raise, h_hours, h_reduction, h_weeks]
  norm_num
  sorry

end bob_weekly_income_increase_l1388_138890


namespace triangle_area_l1388_138898

def vec2 := ℝ × ℝ

def area_of_triangle (a b : vec2) : ℝ :=
  0.5 * |a.1 * b.2 - a.2 * b.1|

def a : vec2 := (2, -3)
def b : vec2 := (4, -1)

theorem triangle_area : area_of_triangle a b = 5 := by
  sorry

end triangle_area_l1388_138898


namespace probability_log_value_l1388_138880

noncomputable def f (x : ℝ) := Real.log x / Real.log 2 - 1

theorem probability_log_value (a : ℝ) (h1 : 1 ≤ a) (h2 : a ≤ 10) :
  (4 / 9 : ℝ) = 
    ((8 - 4) / (10 - 1) : ℝ) := by
  sorry

end probability_log_value_l1388_138880


namespace total_distance_is_1095_l1388_138895

noncomputable def totalDistanceCovered : ℕ :=
  let running_first_3_months := 3 * 3 * 10
  let running_next_3_months := 3 * 3 * 20
  let running_last_6_months := 3 * 6 * 30
  let total_running := running_first_3_months + running_next_3_months + running_last_6_months

  let swimming_first_6_months := 3 * 6 * 5
  let total_swimming := swimming_first_6_months

  let total_hiking := 13 * 15

  total_running + total_swimming + total_hiking

theorem total_distance_is_1095 : totalDistanceCovered = 1095 := by
  sorry

end total_distance_is_1095_l1388_138895


namespace find_x_l1388_138829

theorem find_x (x : ℤ) :
  3 < x ∧ x < 10 →
  5 < x ∧ x < 18 →
  -2 < x ∧ x < 9 →
  0 < x ∧ x < 8 →
  x + 1 < 9 →
  x = 6 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end find_x_l1388_138829


namespace distance_traveled_in_20_seconds_l1388_138842

-- Define the initial distance, common difference, and total time
def initial_distance : ℕ := 8
def common_difference : ℕ := 9
def total_time : ℕ := 20

-- Define the nth term of the arithmetic sequence
def nth_term (n : ℕ) : ℕ := initial_distance + (n - 1) * common_difference

-- Define the sum of the first n terms of the arithmetic sequence
def sum_of_terms (n : ℕ) : ℕ := n * (initial_distance + nth_term n) / 2

-- The main theorem to be proven
theorem distance_traveled_in_20_seconds : sum_of_terms 20 = 1870 := 
by sorry

end distance_traveled_in_20_seconds_l1388_138842


namespace y_coord_of_third_vertex_of_equilateral_l1388_138826

/-- Given two vertices of an equilateral triangle at (0, 6) and (10, 6), and the third vertex in the first quadrant,
    prove that the y-coordinate of the third vertex is 6 + 5 * sqrt 3. -/
theorem y_coord_of_third_vertex_of_equilateral (A B C : ℝ × ℝ)
  (hA : A = (0, 6)) (hB : B = (10, 6)) (hAB : dist A B = 10) (hC : C.2 > 6):
  C.2 = 6 + 5 * Real.sqrt 3 :=
sorry

end y_coord_of_third_vertex_of_equilateral_l1388_138826


namespace gcd_84_126_l1388_138824

-- Conditions
def a : ℕ := 84
def b : ℕ := 126

-- Theorem to prove gcd(a, b) = 42
theorem gcd_84_126 : Nat.gcd a b = 42 := by
  sorry

end gcd_84_126_l1388_138824


namespace mn_value_l1388_138812

theorem mn_value (m n : ℤ) (h1 : m = n + 2) (h2 : 2 * m + n = 4) : m * n = 0 := by
  sorry

end mn_value_l1388_138812


namespace sequence_inequality_for_k_l1388_138868

theorem sequence_inequality_for_k (k : ℝ) : 
  (∀ n : ℕ, 0 < n → (n + 1)^2 + k * (n + 1) + 2 > n^2 + k * n + 2) ↔ k > -3 :=
sorry

end sequence_inequality_for_k_l1388_138868


namespace exists_nat_number_reduce_by_57_l1388_138830

theorem exists_nat_number_reduce_by_57 :
  ∃ (N : ℕ), ∃ (k : ℕ) (a x : ℕ),
    N = 10^k * a + x ∧
    10^k * a + x = 57 * x ∧
    N = 7125 :=
sorry

end exists_nat_number_reduce_by_57_l1388_138830


namespace doug_marbles_l1388_138892

theorem doug_marbles (e_0 d_0 : ℕ) (h1 : e_0 = d_0 + 12) (h2 : e_0 - 20 = 17) : d_0 = 25 :=
by
  sorry

end doug_marbles_l1388_138892


namespace common_region_area_of_triangles_l1388_138878

noncomputable def area_of_common_region (a : ℝ) : ℝ :=
  (a^2 * (2 * Real.sqrt 3 - 3)) / Real.sqrt 3

theorem common_region_area_of_triangles (a : ℝ) (h : 0 < a) : 
  area_of_common_region a = (a^2 * (2 * Real.sqrt 3 - 3)) / Real.sqrt 3 :=
by
  sorry

end common_region_area_of_triangles_l1388_138878


namespace common_measure_largest_l1388_138801

theorem common_measure_largest {a b : ℕ} (h_a : a = 15) (h_b : b = 12): 
  (∀ c : ℕ, c ∣ a ∧ c ∣ b → c ≤ Nat.gcd a b) ∧ Nat.gcd a b = 3 := 
by
  sorry

end common_measure_largest_l1388_138801


namespace captivating_quadruples_count_l1388_138839

theorem captivating_quadruples_count :
  (∃ n : ℕ, n = 682) ↔ 
  ∃ (a b c d : ℕ), 1 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d ≤ 15 ∧ a + d < b + c :=
sorry

end captivating_quadruples_count_l1388_138839


namespace percent_both_correct_proof_l1388_138837

-- Define the problem parameters
def totalTestTakers := 100
def percentFirstCorrect := 80
def percentSecondCorrect := 75
def percentNeitherCorrect := 5

-- Define the target proof statement
theorem percent_both_correct_proof :
  percentFirstCorrect + percentSecondCorrect - percentFirstCorrect + percentNeitherCorrect = 60 := 
by 
  sorry

end percent_both_correct_proof_l1388_138837


namespace bs_sequence_bounded_iff_f_null_l1388_138845

def is_bs_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n = abs (a (n + 1) - a (n + 2))

def f_null (a : ℕ → ℝ) : Prop :=
  ∀ n k, a n * a k * (a n - a k) = 0

def bs_bounded (a : ℕ → ℝ) : Prop :=
  ∃ M, ∀ n, abs (a n) ≤ M

theorem bs_sequence_bounded_iff_f_null (a : ℕ → ℝ) :
  is_bs_sequence a →
  (bs_bounded a ↔ f_null a) := by
  sorry

end bs_sequence_bounded_iff_f_null_l1388_138845


namespace carl_profit_l1388_138821

-- Define the conditions
def price_per_watermelon : ℕ := 3
def watermelons_start : ℕ := 53
def watermelons_end : ℕ := 18

-- Define the number of watermelons sold
def watermelons_sold : ℕ := watermelons_start - watermelons_end

-- Define the profit
def profit : ℕ := watermelons_sold * price_per_watermelon

-- State the theorem about Carl's profit
theorem carl_profit : profit = 105 :=
by
  -- Proof can be filled in later
  sorry

end carl_profit_l1388_138821


namespace set_intersection_complement_l1388_138802

variable (U : Set ℕ)
variable (P Q : Set ℕ)

theorem set_intersection_complement {U : Set ℕ} {P Q : Set ℕ} 
  (hU : U = {1, 2, 3, 4, 5, 6}) 
  (hP : P = {1, 2, 3, 4}) 
  (hQ : Q = {3, 4, 5, 6}) : 
  P ∩ (U \ Q) = {1, 2} :=
by
  sorry

end set_intersection_complement_l1388_138802


namespace smallest_positive_integer_remainder_conditions_l1388_138885

theorem smallest_positive_integer_remainder_conditions :
  ∃ b : ℕ, (b % 4 = 3 ∧ b % 6 = 5) ∧ ∀ n : ℕ, (n % 4 = 3 ∧ n % 6 = 5) → n ≥ b := 
by
  have b := 23
  use b
  sorry

end smallest_positive_integer_remainder_conditions_l1388_138885


namespace geometric_sequence_product_l1388_138893

theorem geometric_sequence_product (a : ℕ → ℝ) (r : ℝ) (n : ℕ) (h : a 4 = 4) :
  a 2 * a 6 = 16 := by
  -- Definition of geomtric sequence
  -- a_n = a_0 * r^n
  -- Using the fact that the product of corresponding terms equidistant from two ends is constant
  sorry

end geometric_sequence_product_l1388_138893


namespace distance_qr_eq_b_l1388_138891

theorem distance_qr_eq_b
  (a b c : ℝ)
  (hP : b = c * Real.cosh (a / c))
  (hQ : ∃ Q : ℝ × ℝ, Q = (0, c) ∧ Q.2 = c * Real.cosh (Q.1 / c))
  : QR = b := by
  sorry

end distance_qr_eq_b_l1388_138891


namespace evaluate_expression_l1388_138800

theorem evaluate_expression : (3 + Real.sqrt 3 + 1 / (3 + Real.sqrt 3) + 1 / (Real.sqrt 3 - 3)) = 3 := by
  sorry

end evaluate_expression_l1388_138800


namespace nonzero_fraction_power_zero_l1388_138879

theorem nonzero_fraction_power_zero (a b : ℤ) (h : a ≠ 0 ∧ b ≠ 0) : ((a : ℚ) / b)^0 = 1 := 
by
  -- proof goes here
  sorry

end nonzero_fraction_power_zero_l1388_138879


namespace erased_length_l1388_138804

def original_length := 100 -- in cm
def final_length := 76 -- in cm

theorem erased_length : original_length - final_length = 24 :=
by
    sorry

end erased_length_l1388_138804


namespace problem_statement_l1388_138822

open Real

theorem problem_statement (x y z : ℝ)
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (a : ℝ := x + x⁻¹) (b : ℝ := y + y⁻¹) (c : ℝ := z + z⁻¹) :
  a > 2 ∧ b > 2 ∧ c > 2 :=
by sorry

end problem_statement_l1388_138822


namespace largest_possible_integer_smallest_possible_integer_l1388_138865

theorem largest_possible_integer : 3 * (15 + 20 / 4 + 1) = 63 := by
  sorry

theorem smallest_possible_integer : (3 * 15 + 20) / (4 + 1) = 13 := by
  sorry

end largest_possible_integer_smallest_possible_integer_l1388_138865


namespace snack_eaters_left_l1388_138810

theorem snack_eaters_left (initial_participants : ℕ)
    (snack_initial : ℕ)
    (new_outsiders1 : ℕ)
    (half_left1 : ℕ)
    (new_outsiders2 : ℕ)
    (left2 : ℕ)
    (half_left2 : ℕ)
    (h1 : initial_participants = 200)
    (h2 : snack_initial = 100)
    (h3 : new_outsiders1 = 20)
    (h4 : half_left1 = (snack_initial + new_outsiders1) / 2)
    (h5 : new_outsiders2 = 10)
    (h6 : left2 = 30)
    (h7 : half_left2 = (half_left1 + new_outsiders2 - left2) / 2) :
    half_left2 = 20 := 
  sorry

end snack_eaters_left_l1388_138810


namespace zero_ordered_triples_non_zero_satisfy_conditions_l1388_138850

theorem zero_ordered_triples_non_zero_satisfy_conditions :
  ∀ (a b c : ℝ), a ≠ 0 → b ≠ 0 → c ≠ 0 → a = b + c → b = c + a → c = a + b → a + b + c ≠ 0 :=
by
  sorry

end zero_ordered_triples_non_zero_satisfy_conditions_l1388_138850


namespace erasers_per_box_l1388_138854

theorem erasers_per_box (total_erasers : ℕ) (num_boxes : ℕ) (erasers_per_box : ℕ) : total_erasers = 40 → num_boxes = 4 → erasers_per_box = total_erasers / num_boxes → erasers_per_box = 10 :=
by
  intros h_total h_boxes h_div
  rw [h_total, h_boxes] at h_div
  norm_num at h_div
  exact h_div

end erasers_per_box_l1388_138854


namespace prism_volume_l1388_138884

-- Define the right triangular prism conditions

variables (AB BC AC : ℝ)
variable (S : ℝ)
variable (volume : ℝ)

-- Given conditions
axiom AB_eq_2 : AB = 2
axiom BC_eq_2 : BC = 2
axiom AC_eq_2sqrt3 : AC = 2 * Real.sqrt 3
axiom circumscribed_sphere_surface_area : S = 32 * Real.pi

-- Statement to prove
theorem prism_volume : volume = 4 * Real.sqrt 3 :=
sorry

end prism_volume_l1388_138884


namespace sum_of_a_and_b_l1388_138816

theorem sum_of_a_and_b (a b : ℕ) (h1 : b > 1) (h2 : a^b < 500) (h3 : ∀ c d : ℕ, d > 1 → c^d < 500 → c^d ≤ a^b) : a + b = 24 :=
sorry

end sum_of_a_and_b_l1388_138816


namespace alicia_read_more_books_than_ian_l1388_138886

def books_read : List Nat := [3, 5, 8, 6, 7, 4, 2, 1]

def alicia_books (books : List Nat) : Nat :=
  books.maximum?.getD 0

def ian_books (books : List Nat) : Nat :=
  books.minimum?.getD 0

theorem alicia_read_more_books_than_ian :
  alicia_books books_read - ian_books books_read = 7 :=
by
  -- By reviewing the given list of books read [3, 5, 8, 6, 7, 4, 2, 1]
  -- We find that alicia_books books_read = 8 and ian_books books_read = 1
  -- Thus, 8 - 1 = 7
  sorry

end alicia_read_more_books_than_ian_l1388_138886


namespace parabola_vertex_l1388_138815

theorem parabola_vertex :
  ∃ a k : ℝ, (∀ x y : ℝ, y^2 - 4*y + 2*x + 7 = 0 ↔ y = k ∧ x = a - (1/2)*(y - k)^2) ∧ a = -3/2 ∧ k = 2 :=
by
  sorry

end parabola_vertex_l1388_138815


namespace lioness_age_l1388_138833

theorem lioness_age (H L : ℕ) 
  (h1 : L = 2 * H) 
  (h2 : (H / 2 + 5) + (L / 2 + 5) = 19) : 
  L = 12 :=
sorry

end lioness_age_l1388_138833


namespace consecutive_number_other_17_l1388_138862

theorem consecutive_number_other_17 (a b : ℕ) (h1 : b = 17) (h2 : a + b = 35) (h3 : a + b % 5 = 0) : a = 18 :=
sorry

end consecutive_number_other_17_l1388_138862


namespace least_subtracted_divisible_by_5_l1388_138820

theorem least_subtracted_divisible_by_5 :
  ∃ n : ℕ, (568219 - n) % 5 = 0 ∧ n ≤ 4 ∧ (∀ m : ℕ, m < 4 → (568219 - m) % 5 ≠ 0) :=
sorry

end least_subtracted_divisible_by_5_l1388_138820


namespace circle_equation_center_at_1_2_passing_through_origin_l1388_138859

theorem circle_equation_center_at_1_2_passing_through_origin :
  ∃ (x y : ℝ), (x - 1)^2 + (y - 2)^2 = 5 ∧
                (0 - 1)^2 + (0 - 2)^2 = 5 :=
by
  sorry

end circle_equation_center_at_1_2_passing_through_origin_l1388_138859


namespace valid_pairs_for_area_18_l1388_138828

theorem valid_pairs_for_area_18 (w l : ℕ) (hw : 0 < w) (hl : 0 < l) (h_area : w * l = 18) (h_lt : w < l) :
  (w, l) = (1, 18) ∨ (w, l) = (2, 9) ∨ (w, l) = (3, 6) :=
sorry

end valid_pairs_for_area_18_l1388_138828


namespace tiffany_total_bags_l1388_138836

theorem tiffany_total_bags (monday_bags next_day_bags : ℕ) (h1 : monday_bags = 4) (h2 : next_day_bags = 8) :
  monday_bags + next_day_bags = 12 :=
by
  sorry

end tiffany_total_bags_l1388_138836


namespace speed_conversion_l1388_138814

theorem speed_conversion (speed_kmh : ℝ) (conversion_factor : ℝ) :
  speed_kmh = 1.3 → conversion_factor = (1000 / 3600) → speed_kmh * conversion_factor = 0.3611 :=
by
  intros h_speed h_factor
  rw [h_speed, h_factor]
  norm_num
  sorry

end speed_conversion_l1388_138814


namespace sum_of_geometric_sequence_l1388_138858

noncomputable def geometric_sequence_sum (a : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a * (1 - q^n) / (1 - q)

theorem sum_of_geometric_sequence
  (a_1 q : ℝ) 
  (h1 : a_1^2 * q^6 = 2 * a_1 * q^2)
  (h2 : (a_1 * q^3 + 2 * a_1 * q^6) / 2 = 5 / 4)
  : geometric_sequence_sum a_1 q 4 = 30 :=
by
  sorry

end sum_of_geometric_sequence_l1388_138858


namespace num_diamonds_F10_l1388_138877

def num_diamonds (n : ℕ) : ℕ :=
  if n = 1 then 4 else 4 * (3 * n - 2)

theorem num_diamonds_F10 : num_diamonds 10 = 112 := by
  sorry

end num_diamonds_F10_l1388_138877


namespace average_weight_l1388_138838

theorem average_weight (w : ℕ) : 
  (64 < w ∧ w ≤ 67) → w = 66 :=
by sorry

end average_weight_l1388_138838


namespace playground_dimensions_l1388_138857

theorem playground_dimensions 
  (a b : ℕ) 
  (h1 : (a - 2) * (b - 2) = 4) : a * b = 2 * a + 2 * b :=
by
  sorry

end playground_dimensions_l1388_138857


namespace max_area_rectangle_l1388_138817

theorem max_area_rectangle :
  ∃ (l w : ℕ), (2 * (l + w) = 40) ∧ (l ≥ w + 3) ∧ (l * w = 91) :=
by
  sorry

end max_area_rectangle_l1388_138817


namespace smallest_prime_factor_of_5_pow_5_minus_5_pow_3_l1388_138867

theorem smallest_prime_factor_of_5_pow_5_minus_5_pow_3 : Nat.Prime 2 ∧ (∀ p : ℕ, Nat.Prime p ∧ p ∣ (5^5 - 5^3) → p ≥ 2) := by
  sorry

end smallest_prime_factor_of_5_pow_5_minus_5_pow_3_l1388_138867


namespace exists_indices_l1388_138840

theorem exists_indices (a : ℕ → ℕ) 
  (h_seq_perm : ∀ n, ∃ m, a m = n) : 
  ∃ ℓ m, 1 < ℓ ∧ ℓ < m ∧ a 1 + a m = 2 * a ℓ :=
by
  sorry

end exists_indices_l1388_138840


namespace determine_m_range_l1388_138866

-- Define propositions P and Q
def P (t : ℝ) : Prop := ∃ x y : ℝ, (x^2 / (t + 2) + y^2 / (t - 10) = 1)
def Q (t m : ℝ) : Prop := 1 - m < t ∧ t < 1 + m ∧ m > 0

-- Define negation of propositions
def notP (t : ℝ) : Prop := ∀ x y : ℝ, (x^2 / (t + 2) + y^2 / (t - 10) ≠ 1)
def notQ (t m : ℝ) : Prop := ¬ (1 - m < t ∧ t < 1 + m)

-- Main problem: Determine the range of m where notP -> notQ is a sufficient but not necessary condition
theorem determine_m_range {m : ℝ} : (∃ t : ℝ, notP t → notQ t m) ↔ (0 < m ∧ m ≤ 3) := by
  sorry

end determine_m_range_l1388_138866


namespace f_is_periodic_l1388_138873

noncomputable def f (x : ℝ) : ℝ := x - ⌊x⌋

theorem f_is_periodic : ∀ x : ℝ, f (x + 1) = f x := by
  intro x
  sorry

end f_is_periodic_l1388_138873


namespace inequality_solution_set_l1388_138823

theorem inequality_solution_set :
  ∀ x : ℝ, 8 * x^3 + 9 * x^2 + 7 * x - 6 < 0 ↔ (( -6 < x ∧ x < -1/8) ∨ (-1/8 < x ∧ x < 1)) :=
sorry

end inequality_solution_set_l1388_138823


namespace russian_dolls_initial_purchase_l1388_138847

theorem russian_dolls_initial_purchase (cost_initial cost_discount : ℕ) (num_discount : ℕ) (savings : ℕ) :
  cost_initial = 4 → cost_discount = 3 → num_discount = 20 → savings = num_discount * cost_discount → 
  (savings / cost_initial) = 15 := 
by {
sorry
}

end russian_dolls_initial_purchase_l1388_138847


namespace intersection_M_N_l1388_138832

def M : Set ℤ := {-1, 0, 1}
def N : Set ℤ := {x | x^2 ≠ x}

theorem intersection_M_N:
  M ∩ N = {-1} := by
  sorry

end intersection_M_N_l1388_138832


namespace tangent_line_through_origin_to_circle_in_third_quadrant_l1388_138808

theorem tangent_line_through_origin_to_circle_in_third_quadrant :
  ∃ m : ℝ, (∀ x y : ℝ, y = m * x) ∧ (∀ x y : ℝ, x^2 + y^2 + 4 * x + 3 = 0) ∧ (x < 0 ∧ y < 0) ∧ y = -3 * x :=
sorry

end tangent_line_through_origin_to_circle_in_third_quadrant_l1388_138808


namespace calculate_total_prime_dates_l1388_138876

-- Define the prime months
def prime_months : List Nat := [2, 3, 5, 7, 11, 13]

-- Define the number of days in each month for a non-leap year
def days_in_month (month : Nat) : Nat :=
  if month = 2 then 28
  else if month = 3 then 31
  else if month = 5 then 31
  else if month = 7 then 31
  else if month = 11 then 30
  else if month = 13 then 31
  else 0

-- Define the prime days in a month
def prime_days : List Nat := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

-- Calculate the number of prime dates in a given month
def prime_dates_in_month (month : Nat) : Nat :=
  (prime_days.filter (λ d => d <= days_in_month month)).length

-- Calculate the total number of prime dates for the year
def total_prime_dates : Nat :=
  (prime_months.map prime_dates_in_month).sum

theorem calculate_total_prime_dates : total_prime_dates = 62 := by
  sorry

end calculate_total_prime_dates_l1388_138876


namespace intersection_of_A_and_B_l1388_138803

def U := Set ℝ
def A := {x : ℝ | -2 ≤ x ∧ x ≤ 3 }
def B := {x : ℝ | x < -1}
def C := {x : ℝ | -2 ≤ x ∧ x < -1}

theorem intersection_of_A_and_B : A ∩ B = C :=
by sorry

end intersection_of_A_and_B_l1388_138803


namespace popsicles_eaten_l1388_138825

theorem popsicles_eaten (total_time : ℕ) (interval : ℕ) (p : ℕ)
  (h_total_time : total_time = 6 * 60)
  (h_interval : interval = 20) :
  p = total_time / interval :=
sorry

end popsicles_eaten_l1388_138825


namespace degree_of_monomial_neg2x2y_l1388_138881

def monomial_degree (coeff : ℤ) (exp_x exp_y : ℕ) : ℕ :=
  exp_x + exp_y

theorem degree_of_monomial_neg2x2y :
  monomial_degree (-2) 2 1 = 3 :=
by
  -- Definition matching conditions given
  sorry

end degree_of_monomial_neg2x2y_l1388_138881


namespace slope_of_decreasing_linear_function_l1388_138875

theorem slope_of_decreasing_linear_function (m b : ℝ) :
  (∀ x y : ℝ, x < y → mx + b > my + b) → m < 0 :=
by
  intro h
  sorry

end slope_of_decreasing_linear_function_l1388_138875


namespace range_of_expression_l1388_138889

variable (a b c : ℝ)

theorem range_of_expression (h1 : -3 < b) (h2 : b < a) (h3 : a < -1) (h4 : -2 < c) (h5 : c < -1) :
  0 < (a - b) * c^2 ∧ (a - b) * c^2 < 8 :=
sorry

end range_of_expression_l1388_138889


namespace tom_cost_cheaper_than_jane_l1388_138896

def store_A_full_price : ℝ := 125
def store_A_discount_single : ℝ := 0.08
def store_A_discount_bulk : ℝ := 0.12
def store_A_tax_rate : ℝ := 0.07
def store_A_shipping_fee : ℝ := 10
def store_A_club_discount : ℝ := 0.05

def store_B_full_price : ℝ := 130
def store_B_discount_single : ℝ := 0.10
def store_B_discount_bulk : ℝ := 0.15
def store_B_tax_rate : ℝ := 0.05
def store_B_free_shipping_threshold : ℝ := 250
def store_B_club_discount : ℝ := 0.03

def tom_smartphones_qty : ℕ := 2
def jane_smartphones_qty : ℕ := 3

theorem tom_cost_cheaper_than_jane :
  let tom_cost := 
    let total := store_A_full_price * tom_smartphones_qty
    let discount := if tom_smartphones_qty ≥ 2 then store_A_discount_bulk else store_A_discount_single
    let price_after_discount := total * (1 - discount)
    let price_after_club_discount := price_after_discount * (1 - store_A_club_discount)
    let price_after_tax := price_after_club_discount * (1 + store_A_tax_rate) 
    price_after_tax + store_A_shipping_fee

  let jane_cost := 
    let total := store_B_full_price * jane_smartphones_qty
    let discount := if jane_smartphones_qty ≥ 3 then store_B_discount_bulk else store_B_discount_single
    let price_after_discount := total * (1 - discount)
    let price_after_club_discount := price_after_discount * (1 - store_B_club_discount)
    let price_after_tax := price_after_club_discount * (1 + store_B_tax_rate)
    let shipping_fee := if total > store_B_free_shipping_threshold then 0 else 0
    price_after_tax + shipping_fee
  
  jane_cost - tom_cost = 104.01 := 
by 
  sorry

end tom_cost_cheaper_than_jane_l1388_138896


namespace glass_panels_in_neighborhood_l1388_138851

def total_glass_panels_in_neighborhood := 
  let double_windows_downstairs : ℕ := 6
  let glass_panels_per_double_window_downstairs : ℕ := 4
  let single_windows_upstairs : ℕ := 8
  let glass_panels_per_single_window_upstairs : ℕ := 3
  let bay_windows : ℕ := 2
  let glass_panels_per_bay_window : ℕ := 6
  let houses : ℕ := 10

  let glass_panels_in_one_house : ℕ := 
    (double_windows_downstairs * glass_panels_per_double_window_downstairs) +
    (single_windows_upstairs * glass_panels_per_single_window_upstairs) +
    (bay_windows * glass_panels_per_bay_window)

  houses * glass_panels_in_one_house

theorem glass_panels_in_neighborhood : total_glass_panels_in_neighborhood = 600 := by
  -- Calculation steps skipped
  sorry

end glass_panels_in_neighborhood_l1388_138851


namespace regular_polygon_sides_l1388_138855

theorem regular_polygon_sides (interior_angle : ℝ) (h : interior_angle = 150) : 
  ∃ n : ℕ, n = 12 := by
  sorry

end regular_polygon_sides_l1388_138855


namespace arithmetic_sequence_a_eq_zero_l1388_138853

theorem arithmetic_sequence_a_eq_zero (a : ℝ) :
  (∀ n : ℕ, n > 0 → ∃ S : ℕ → ℝ, S n = (n^2 : ℝ) + 2 * n + a) →
  a = 0 :=
by
  sorry

end arithmetic_sequence_a_eq_zero_l1388_138853


namespace positive_quadratic_expression_l1388_138846

theorem positive_quadratic_expression (m : ℝ) :
  (∀ x : ℝ, (4 - m) * x^2 - 3 * x + 4 + m > 0) ↔ (- (Real.sqrt 55) / 2 < m ∧ m < (Real.sqrt 55) / 2) := 
sorry

end positive_quadratic_expression_l1388_138846


namespace evaluate_expression_l1388_138869

theorem evaluate_expression :
  (1 / (-5^3)^4) * (-5)^15 * 5^2 = -3125 :=
by
  sorry

end evaluate_expression_l1388_138869


namespace person_walk_rate_l1388_138882

theorem person_walk_rate (v : ℝ) (elevator_speed : ℝ) (length : ℝ) (time : ℝ) 
  (h1 : elevator_speed = 10) 
  (h2 : length = 112) 
  (h3 : time = 8) 
  (h4 : length = (v + elevator_speed) * time) 
  : v = 4 :=
by 
  sorry

end person_walk_rate_l1388_138882


namespace fifth_coordinate_is_14_l1388_138827

theorem fifth_coordinate_is_14
  (a : Fin 16 → ℝ)
  (h_1 : a 0 = 2)
  (h_16 : a 15 = 47)
  (h_avg : ∀ i : Fin 14, a (i + 1) = (a i + a (i + 2)) / 2) :
  a 4 = 14 :=
by
  sorry

end fifth_coordinate_is_14_l1388_138827


namespace exists_special_integer_l1388_138843

-- Define the mathematical conditions and the proof
theorem exists_special_integer (P : Finset ℕ) (hP : ∀ p ∈ P, Nat.Prime p) : 
  ∃ x : ℕ, 
    (∀ p ∈ P, ∃ a b : ℕ, 0 < a ∧ 0 < b ∧ x = a^p + b^p) ∧
    (∀ p ∉ P, ¬∃ a b : ℕ, 0 < a ∧ 0 < b ∧ x = a^p + b^p) :=
sorry

end exists_special_integer_l1388_138843


namespace n_squared_sum_of_squares_l1388_138852

theorem n_squared_sum_of_squares (n a b c : ℕ) (h : n = a^2 + b^2 + c^2) : 
  ∃ x y z : ℕ, n^2 = x^2 + y^2 + z^2 :=
by 
  sorry

end n_squared_sum_of_squares_l1388_138852


namespace solution_to_problem_l1388_138813

def f (x : ℝ) : ℝ := sorry

noncomputable def functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * y) = y^2 * f x + x^2 * f y

theorem solution_to_problem
  (f : ℝ → ℝ)
  (h : functional_equation f) :
  f 0 = 0 ∧ f 1 = 0 ∧ ∀ x : ℝ, f (-x) = f x :=
by
  sorry

end solution_to_problem_l1388_138813


namespace log_fraction_property_l1388_138864

noncomputable def log_base (a N : ℝ) : ℝ := Real.log N / Real.log a

theorem log_fraction_property :
  (log_base 3 4 / log_base 9 8) = 4 / 3 :=
by
  sorry

end log_fraction_property_l1388_138864


namespace checkout_speed_ratio_l1388_138841

theorem checkout_speed_ratio (n x y : ℝ) 
  (h1 : 40 * x = 20 * y + n)
  (h2 : 36 * x = 12 * y + n) : 
  x = 2 * y := 
sorry

end checkout_speed_ratio_l1388_138841


namespace expression_not_defined_at_x_l1388_138844

theorem expression_not_defined_at_x :
  ∃ (x : ℝ), x = 10 ∧ (x^3 - 30 * x^2 + 300 * x - 1000) = 0 := 
sorry

end expression_not_defined_at_x_l1388_138844


namespace true_proposition_l1388_138872

variable (p q : Prop)
variable (hp : p = true)
variable (hq : q = false)

theorem true_proposition : (¬p ∨ ¬q) = true := by
  sorry

end true_proposition_l1388_138872


namespace sequence_non_positive_l1388_138819

theorem sequence_non_positive
  (a : ℕ → ℝ) (n : ℕ)
  (h0 : a 0 = 0)
  (hn : a n = 0)
  (h : ∀ k, 1 ≤ k → k ≤ n - 1 → a (k - 1) - 2 * a k + a (k + 1) ≥ 0) :
  ∀ k, k ≤ n → a k ≤ 0 := 
sorry

end sequence_non_positive_l1388_138819


namespace multiply_123_32_125_l1388_138894

theorem multiply_123_32_125 : 123 * 32 * 125 = 492000 := by
  sorry

end multiply_123_32_125_l1388_138894


namespace cyclist_return_trip_average_speed_l1388_138849

theorem cyclist_return_trip_average_speed :
  let first_leg_distance := 12
  let second_leg_distance := 24
  let first_leg_speed := 8
  let second_leg_speed := 12
  let round_trip_time := 7.5
  let distance_to_destination := first_leg_distance + second_leg_distance
  let time_to_destination := (first_leg_distance / first_leg_speed) + (second_leg_distance / second_leg_speed)
  let return_trip_time := round_trip_time - time_to_destination
  let return_trip_distance := distance_to_destination
  (return_trip_distance / return_trip_time) = 9 := 
by
  sorry

end cyclist_return_trip_average_speed_l1388_138849


namespace hexagon_area_l1388_138848

theorem hexagon_area (s t_height : ℕ) (tri_area rect_area : ℕ) :
    s = 2 →
    t_height = 4 →
    tri_area = 1 / 2 * s * t_height →
    rect_area = (s + s + s) * (t_height + t_height) →
    rect_area - 4 * tri_area = 32 :=
by
  sorry

end hexagon_area_l1388_138848


namespace find_m_l1388_138834

def f (x m : ℝ) : ℝ := x^2 - 3 * x + m
def g (x m : ℝ) : ℝ := x^2 - 3 * x + 5 * m

theorem find_m :
  let m := 10 / 7
  3 * f 5 m = 2 * g 5 m :=
by
  sorry

end find_m_l1388_138834


namespace ratio_n_over_p_l1388_138874

theorem ratio_n_over_p (m n p : ℝ) (h1 : m ≠ 0) (h2 : n ≠ 0) (h3 : p ≠ 0) 
  (h4 : ∃ r1 r2 : ℝ, r1 + r2 = -p ∧ r1 * r2 = m ∧ 3 * r1 + 3 * r2 = -m ∧ 9 * r1 * r2 = n) :
  n / p = -27 := 
by
  sorry

end ratio_n_over_p_l1388_138874


namespace train_speed_is_30_kmh_l1388_138860

noncomputable def speed_of_train (train_length : ℝ) (cross_time : ℝ) (man_speed_kmh : ℝ) : ℝ :=
  let man_speed_ms := man_speed_kmh * (1000 / 3600)
  let relative_speed := train_length / cross_time
  let train_speed_ms := relative_speed + man_speed_ms
  train_speed_ms * (3600 / 1000)

theorem train_speed_is_30_kmh :
  speed_of_train 400 59.99520038396929 6 = 30 :=
by
  -- Using the approximation mentioned in the solution, hence no computation proof required.
  sorry

end train_speed_is_30_kmh_l1388_138860


namespace remainder_8547_div_9_l1388_138818

theorem remainder_8547_div_9 : 8547 % 9 = 6 :=
by
  sorry

end remainder_8547_div_9_l1388_138818


namespace find_x_l1388_138811

-- Definitions of binomial coefficients as conditions
def binomial (n k : ℕ) : ℕ := n.choose k

-- The specific conditions given
def C65_eq_6 : Prop := binomial 6 5 = 6
def C64_eq_15 : Prop := binomial 6 4 = 15

-- The theorem we need to prove: ∃ x, binomial 7 x = 21
theorem find_x (h1 : C65_eq_6) (h2 : C64_eq_15) : ∃ x, binomial 7 x = 21 :=
by
  -- Proof will go here
  sorry

end find_x_l1388_138811


namespace original_price_of_bag_l1388_138861

theorem original_price_of_bag (P : ℝ) 
  (h1 : ∀ x, 0 < x → x < 1 → x * 100 = 75)
  (h2 : 2 * (0.25 * P) = 3)
  : P = 6 :=
sorry

end original_price_of_bag_l1388_138861


namespace sum_of_tens_and_units_digit_l1388_138871

theorem sum_of_tens_and_units_digit (n : ℕ) (h : n = 11^2004 - 5) : 
  (n % 100 / 10) + (n % 10) = 9 :=
by
  sorry

end sum_of_tens_and_units_digit_l1388_138871


namespace J_3_15_10_eq_68_over_15_l1388_138805

def J (a b c : ℚ) : ℚ := a / b + b / c + c / a

theorem J_3_15_10_eq_68_over_15 : J 3 15 10 = 68 / 15 := by
  sorry

end J_3_15_10_eq_68_over_15_l1388_138805


namespace circle_area_from_diameter_endpoints_l1388_138897

theorem circle_area_from_diameter_endpoints :
  let C := (-2, 3)
  let D := (4, -1)
  let diameter := Real.sqrt ((4 - (-2))^2 + ((-1) - 3)^2)
  let radius := diameter / 2
  let area := Real.pi * radius^2
  C = (-2, 3) ∧ D = (4, -1) → area = 13 * Real.pi := by
    sorry

end circle_area_from_diameter_endpoints_l1388_138897
