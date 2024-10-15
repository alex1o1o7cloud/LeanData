import Mathlib

namespace NUMINAMATH_GPT_smallest_four_digit_number_divisible_by_smallest_primes_l1918_191812

theorem smallest_four_digit_number_divisible_by_smallest_primes : 
  ∃ n : ℕ, n >= 1000 ∧ (∀ p ∈ [2, 3, 5, 7, 11], p ∣ n) ∧ n = 2310 := 
by
  sorry

end NUMINAMATH_GPT_smallest_four_digit_number_divisible_by_smallest_primes_l1918_191812


namespace NUMINAMATH_GPT_problem1_solution_l1918_191821

theorem problem1_solution : ∀ x : ℝ, x^2 - 6 * x + 9 = (5 - 2 * x)^2 → (x = 8/3 ∨ x = 2) :=
sorry

end NUMINAMATH_GPT_problem1_solution_l1918_191821


namespace NUMINAMATH_GPT_distance_between_circle_centers_l1918_191814

-- Define the given side lengths of the triangle
def DE : ℝ := 12
def DF : ℝ := 15
def EF : ℝ := 9

-- Define the problem and assertion
theorem distance_between_circle_centers :
  ∃ d : ℝ, d = 12 * Real.sqrt 13 :=
sorry

end NUMINAMATH_GPT_distance_between_circle_centers_l1918_191814


namespace NUMINAMATH_GPT_geometric_sequence_common_ratio_l1918_191866

theorem geometric_sequence_common_ratio (S : ℕ → ℝ) (a : ℕ → ℝ)
  (q : ℝ) (h1 : a 1 = 2) (h2 : S 3 = 6)
  (geo_sum : ∀ n, S n = a 1 * (1 - q ^ n) / (1 - q)) :
  q = 1 ∨ q = -2 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_common_ratio_l1918_191866


namespace NUMINAMATH_GPT_profit_margin_increase_l1918_191861

theorem profit_margin_increase (CP : ℝ) (SP : ℝ) (NSP : ℝ) (initial_margin : ℝ) (desired_margin : ℝ) :
  initial_margin = 0.25 → desired_margin = 0.40 → SP = (1 + initial_margin) * CP → NSP = (1 + desired_margin) * CP →
  ((NSP - SP) / SP) * 100 = 12 := 
by 
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_profit_margin_increase_l1918_191861


namespace NUMINAMATH_GPT_rectangle_new_area_l1918_191816

theorem rectangle_new_area (original_area : ℝ) (new_length_factor : ℝ) (new_width_factor : ℝ) 
  (h1 : original_area = 560) (h2 : new_length_factor = 1.2) (h3 : new_width_factor = 0.85) : 
  new_length_factor * new_width_factor * original_area = 571 := 
by 
  sorry

end NUMINAMATH_GPT_rectangle_new_area_l1918_191816


namespace NUMINAMATH_GPT_inequality_inequation_l1918_191865

theorem inequality_inequation (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (h : x + y + z = 1) :
  x * y + y * z + z * x ≤ 2 / 7 + 9 * x * y * z / 7 :=
by
  sorry

end NUMINAMATH_GPT_inequality_inequation_l1918_191865


namespace NUMINAMATH_GPT_final_number_after_increase_l1918_191802

-- Define the original number and the percentage increase
def original_number : ℕ := 70
def increase_percentage : ℝ := 0.50

-- Define the function to calculate the final number after the increase
def final_number : ℝ := original_number * (1 + increase_percentage)

-- The proof statement that the final number is 105
theorem final_number_after_increase : final_number = 105 :=
by
  sorry

end NUMINAMATH_GPT_final_number_after_increase_l1918_191802


namespace NUMINAMATH_GPT_probability_of_cold_given_rhinitis_l1918_191876

/-- Define the events A and B as propositions --/
def A : Prop := sorry -- A represents having rhinitis
def B : Prop := sorry -- B represents having a cold

/-- Define the given probabilities as assumptions --/
axiom P_A : ℝ -- P(A) = 0.8
axiom P_A_and_B : ℝ -- P(A ∩ B) = 0.6

/-- Adding the conditions --/
axiom P_A_val : P_A = 0.8
axiom P_A_and_B_val : P_A_and_B = 0.6

/-- Define the conditional probability --/
noncomputable def P_B_given_A : ℝ := P_A_and_B / P_A

/-- The main theorem which states the problem --/
theorem probability_of_cold_given_rhinitis : P_B_given_A = 0.75 :=
by 
  sorry

end NUMINAMATH_GPT_probability_of_cold_given_rhinitis_l1918_191876


namespace NUMINAMATH_GPT_division_of_203_by_single_digit_l1918_191853

theorem division_of_203_by_single_digit (d : ℕ) (h : 1 ≤ d ∧ d < 10) : 
  ∃ q : ℕ, q = 203 / d ∧ (10 ≤ q ∧ q < 100 ∨ 100 ≤ q ∧ q < 1000) := 
by
  sorry

end NUMINAMATH_GPT_division_of_203_by_single_digit_l1918_191853


namespace NUMINAMATH_GPT_smallest_m_plus_n_l1918_191822

theorem smallest_m_plus_n (m n : ℕ) (hm : m > 0) (hn : n > 0) (h : 3 * m^3 = 5 * n^5) : m + n = 720 :=
by
  sorry

end NUMINAMATH_GPT_smallest_m_plus_n_l1918_191822


namespace NUMINAMATH_GPT_fraction_replaced_l1918_191834

theorem fraction_replaced :
  ∃ x : ℚ, (0.60 * (1 - x) + 0.25 * x = 0.35) ∧ x = 5 / 7 := by
    sorry

end NUMINAMATH_GPT_fraction_replaced_l1918_191834


namespace NUMINAMATH_GPT_cylinder_volume_rotation_l1918_191899

theorem cylinder_volume_rotation (length width : ℝ) (π : ℝ) (h : length = 4) (w : width = 2) (V : ℝ) :
  (V = π * (4^2) * width ∨ V = π * (2^2) * length) :=
by
  sorry

end NUMINAMATH_GPT_cylinder_volume_rotation_l1918_191899


namespace NUMINAMATH_GPT_radius_of_tangent_circle_l1918_191862

def is_tangent_coor_axes_and_leg (r : ℝ) : Prop :=
  -- Circle with radius r is tangent to coordinate axes and one leg of the triangle
  ∃ O B C : ℝ × ℝ, 
  -- Conditions: centers and tangency
  O = (r, r) ∧ 
  B = (0, 2) ∧ 
  C = (2, 0) ∧ 
  r = 1

theorem radius_of_tangent_circle :
  ∀ r : ℝ, is_tangent_coor_axes_and_leg r → r = 1 :=
by
  sorry

end NUMINAMATH_GPT_radius_of_tangent_circle_l1918_191862


namespace NUMINAMATH_GPT_extreme_value_at_one_l1918_191838

noncomputable def f (x : ℝ) (a : ℝ) := (x^2 + a) / (x + 1)

theorem extreme_value_at_one (a : ℝ) :
  (∃ x : ℝ, x = 1 ∧ (∀ ε > 0, ∃ δ > 0, ∀ y : ℝ, abs (y-1) < δ → abs (f y a - f 1 a) < ε)) →
  a = 3 :=
by
  sorry

end NUMINAMATH_GPT_extreme_value_at_one_l1918_191838


namespace NUMINAMATH_GPT_chameleons_all_red_l1918_191869

theorem chameleons_all_red (Y G R : ℕ) (total : ℕ) (P : Y = 7) (Q : G = 10) (R_cond : R = 17) (total_cond : Y + G + R = total) (total_value : total = 34) :
  ∃ x, x = R ∧ x = total ∧ ∀ z : ℕ, z ≠ 0 → total % 3 = z % 3 → ((R : ℕ) % 3 = z) :=
by
  sorry

end NUMINAMATH_GPT_chameleons_all_red_l1918_191869


namespace NUMINAMATH_GPT_average_weight_of_eight_boys_l1918_191815

theorem average_weight_of_eight_boys :
  let avg16 := 50.25
  let avg24 := 48.55
  let total_weight_16 := 16 * avg16
  let total_weight_all := 24 * avg24
  let W := (total_weight_all - total_weight_16) / 8
  W = 45.15 :=
by
  sorry

end NUMINAMATH_GPT_average_weight_of_eight_boys_l1918_191815


namespace NUMINAMATH_GPT_matrix_pow_A4_l1918_191882

-- Define the matrix A
def A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![2, -1], ![1, 1]]

-- State the theorem
theorem matrix_pow_A4 :
  A^4 = ![![0, -9], ![9, -9]] :=
by
  sorry -- Proof is omitted

end NUMINAMATH_GPT_matrix_pow_A4_l1918_191882


namespace NUMINAMATH_GPT_range_of_m_l1918_191846

variable {m x : ℝ}

theorem range_of_m (h : ∀ x, -1 < x ∧ x < 4 ↔ x > 2 * m ^ 2 - 3) : m ∈ [-1, 1] :=
sorry

end NUMINAMATH_GPT_range_of_m_l1918_191846


namespace NUMINAMATH_GPT_sum_of_number_and_conjugate_l1918_191884

theorem sum_of_number_and_conjugate :
  (15 - Real.sqrt 500) + (15 + Real.sqrt 500) = 30 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_number_and_conjugate_l1918_191884


namespace NUMINAMATH_GPT_compute_diameter_of_garden_roller_l1918_191830

noncomputable def diameter_of_garden_roller (length : ℝ) (area_per_revolution : ℝ) (pi : ℝ) :=
  let radius := (area_per_revolution / (2 * pi * length))
  2 * radius

theorem compute_diameter_of_garden_roller :
  diameter_of_garden_roller 3 (66 / 5) (22 / 7) = 1.4 := by
  sorry

end NUMINAMATH_GPT_compute_diameter_of_garden_roller_l1918_191830


namespace NUMINAMATH_GPT_apples_picked_per_tree_l1918_191883

-- Definitions
def num_trees : Nat := 4
def total_apples_picked : Nat := 28

-- Proving how many apples Rachel picked from each tree if the same number were picked from each tree
theorem apples_picked_per_tree (h : num_trees ≠ 0) :
  total_apples_picked / num_trees = 7 :=
by
  sorry

end NUMINAMATH_GPT_apples_picked_per_tree_l1918_191883


namespace NUMINAMATH_GPT_middle_number_is_9_l1918_191801

-- Define the problem conditions
variable (x y z : ℕ)

-- Lean proof statement
theorem middle_number_is_9 
  (h1 : x + y = 16)
  (h2 : x + z = 21)
  (h3 : y + z = 23)
  (h4 : x < y)
  (h5 : y < z) : y = 9 :=
by
  sorry

end NUMINAMATH_GPT_middle_number_is_9_l1918_191801


namespace NUMINAMATH_GPT_Luke_mowing_lawns_l1918_191874

theorem Luke_mowing_lawns (L : ℕ) (h1 : 18 + L = 27) : L = 9 :=
by
  sorry

end NUMINAMATH_GPT_Luke_mowing_lawns_l1918_191874


namespace NUMINAMATH_GPT_large_box_count_l1918_191858

variable (x y : ℕ)

theorem large_box_count (h₁ : x + y = 21) (h₂ : 120 * x + 80 * y = 2000) : x = 8 := by
  sorry

end NUMINAMATH_GPT_large_box_count_l1918_191858


namespace NUMINAMATH_GPT_fraction_to_decimal_l1918_191886

/-- The decimal equivalent of 1/4 is 0.25. -/
theorem fraction_to_decimal : (1 : ℝ) / 4 = 0.25 :=
by
  sorry

end NUMINAMATH_GPT_fraction_to_decimal_l1918_191886


namespace NUMINAMATH_GPT_probability_three_heads_in_a_row_l1918_191864

theorem probability_three_heads_in_a_row (h : ℝ) (p_head : h = 1/2) (ind_flips : ∀ (n : ℕ), true) : 
  (1/2 * 1/2 * 1/2 = 1/8) :=
by
  sorry

end NUMINAMATH_GPT_probability_three_heads_in_a_row_l1918_191864


namespace NUMINAMATH_GPT_hilt_books_difference_l1918_191845

noncomputable def original_price : ℝ := 11
noncomputable def discount_rate : ℝ := 0.20
noncomputable def discount_price (price : ℝ) (rate : ℝ) : ℝ := price * (1 - rate)
noncomputable def quantity : ℕ := 15
noncomputable def sale_price : ℝ := 25
noncomputable def tax_rate : ℝ := 0.10
noncomputable def price_with_tax (price : ℝ) (rate : ℝ) : ℝ := price * (1 + rate)

noncomputable def total_cost : ℝ := discount_price original_price discount_rate * quantity
noncomputable def total_revenue : ℝ := price_with_tax sale_price tax_rate * quantity
noncomputable def profit : ℝ := total_revenue - total_cost

theorem hilt_books_difference : profit = 280.50 :=
by
  sorry

end NUMINAMATH_GPT_hilt_books_difference_l1918_191845


namespace NUMINAMATH_GPT_train_speed_is_180_kmh_l1918_191836

-- Defining the conditions
def train_length : ℕ := 1500  -- meters
def platform_length : ℕ := 1500  -- meters
def crossing_time : ℕ := 1  -- minute

-- Function to compute the speed in km/hr
def speed_in_km_per_hr (length : ℕ) (time : ℕ) : ℕ :=
  let distance := length + length
  let speed_m_per_min := distance / time
  let speed_km_per_hr := speed_m_per_min * 60 / 1000
  speed_km_per_hr

-- The main theorem we need to prove
theorem train_speed_is_180_kmh :
  speed_in_km_per_hr train_length crossing_time = 180 :=
by
  sorry

end NUMINAMATH_GPT_train_speed_is_180_kmh_l1918_191836


namespace NUMINAMATH_GPT_sqrt_sum_4_pow_4_eq_32_l1918_191891

theorem sqrt_sum_4_pow_4_eq_32 : Real.sqrt (4^4 + 4^4 + 4^4 + 4^4) = 32 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_sum_4_pow_4_eq_32_l1918_191891


namespace NUMINAMATH_GPT_number_of_pies_l1918_191873

-- Definitions based on the conditions
def box_weight : ℕ := 120
def weight_for_applesauce : ℕ := box_weight / 2
def weight_per_pie : ℕ := 4
def remaining_weight : ℕ := box_weight - weight_for_applesauce

-- The proof problem statement
theorem number_of_pies : (remaining_weight / weight_per_pie) = 15 :=
by
  sorry

end NUMINAMATH_GPT_number_of_pies_l1918_191873


namespace NUMINAMATH_GPT_min_value_of_a_plus_b_minus_c_l1918_191833

open Real

theorem min_value_of_a_plus_b_minus_c (a b c : ℝ) :
  (∀ (x y : ℝ), 3 * x + 4 * y - 5 ≤ a * x + b * y + c ∧ a * x + b * y + c ≤ 3 * x + 4 * y + 5) →
  (∃ c_min, c_min = 2 ∧ ∀ c', c' = a + b - c → c' ≥ c_min) :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_a_plus_b_minus_c_l1918_191833


namespace NUMINAMATH_GPT_trapezoid_PR_length_l1918_191800

noncomputable def PR_length (PQ RS QS PR : ℝ) (angle_QSP angle_SRP : ℝ) : Prop :=
  PQ < RS ∧ 
  QS = 2 ∧ 
  angle_QSP = 30 ∧ 
  angle_SRP = 60 ∧ 
  RS / PQ = 7 / 3 ∧ 
  PR = 8 / 3

theorem trapezoid_PR_length (PQ RS QS PR : ℝ) 
  (angle_QSP angle_SRP : ℝ) 
  (h1 : PQ < RS) 
  (h2 : QS = 2) 
  (h3 : angle_QSP = 30) 
  (h4 : angle_SRP = 60) 
  (h5 : RS / PQ = 7 / 3) :
  PR = 8 / 3 := 
by
  sorry

end NUMINAMATH_GPT_trapezoid_PR_length_l1918_191800


namespace NUMINAMATH_GPT_problem1_problem2_l1918_191892

/-- Problem 1 -/
theorem problem1 (a b : ℝ) : (a^2 - b)^2 = a^4 - 2 * a^2 * b + b^2 :=
by
  sorry

/-- Problem 2 -/
theorem problem2 (x : ℝ) : (2 * x + 1) * (4 * x^2 - 1) * (2 * x - 1) = 16 * x^4 - 8 * x^2 + 1 :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1918_191892


namespace NUMINAMATH_GPT_accessories_per_doll_l1918_191854

theorem accessories_per_doll (n dolls accessories time_per_doll time_per_accessory total_time : ℕ)
  (h0 : dolls = 12000)
  (h1 : time_per_doll = 45)
  (h2 : time_per_accessory = 10)
  (h3 : total_time = 1860000)
  (h4 : time_per_doll + accessories * time_per_accessory = n)
  (h5 : dolls * n = total_time) :
  accessories = 11 :=
by
  sorry

end NUMINAMATH_GPT_accessories_per_doll_l1918_191854


namespace NUMINAMATH_GPT_average_M_possibilities_l1918_191810

theorem average_M_possibilities (M : ℝ) (h1 : 12 < M) (h2 : M < 25) :
    (12 = (8 + 15 + M) / 3) ∨ (15 = (8 + 15 + M) / 3) :=
  sorry

end NUMINAMATH_GPT_average_M_possibilities_l1918_191810


namespace NUMINAMATH_GPT_smallest_angle_measure_in_triangle_l1918_191828

theorem smallest_angle_measure_in_triangle (a b : ℝ) (c : ℝ) (h1 : a = 2) (h2 : b = 1) (h3 : c > 2 * Real.sqrt 2) :
  ∃ x : ℝ, x = 140 ∧ C < x :=
sorry

end NUMINAMATH_GPT_smallest_angle_measure_in_triangle_l1918_191828


namespace NUMINAMATH_GPT_chairperson_and_committee_ways_l1918_191835

-- Definitions based on conditions
def total_people : ℕ := 10
def ways_to_choose_chairperson : ℕ := total_people
def ways_to_choose_committee (remaining_people : ℕ) (committee_size : ℕ) : ℕ :=
  Nat.choose remaining_people committee_size

-- The resulting theorem
theorem chairperson_and_committee_ways :
  ways_to_choose_chairperson * ways_to_choose_committee (total_people - 1) 3 = 840 :=
by
  sorry

end NUMINAMATH_GPT_chairperson_and_committee_ways_l1918_191835


namespace NUMINAMATH_GPT_find_m_plus_n_l1918_191894

def num_fir_trees : ℕ := 4
def num_pine_trees : ℕ := 5
def num_acacia_trees : ℕ := 6

def num_non_acacia_trees : ℕ := num_fir_trees + num_pine_trees
def total_trees : ℕ := num_fir_trees + num_pine_trees + num_acacia_trees

def prob_no_two_acacia_adj : ℚ :=
  (Nat.choose (num_non_acacia_trees + 1) num_acacia_trees * Nat.choose num_non_acacia_trees num_fir_trees : ℚ) /
  Nat.choose total_trees num_acacia_trees

theorem find_m_plus_n : (prob_no_two_acacia_adj = 84/159) -> (84 + 159 = 243) :=
by {
  admit
}

end NUMINAMATH_GPT_find_m_plus_n_l1918_191894


namespace NUMINAMATH_GPT_find_m_l1918_191857

variables (AB AC AD : ℝ × ℝ)
variables (m : ℝ)

-- Definitions of vectors
def vector_AB : ℝ × ℝ := (-1, 2)
def vector_AC : ℝ × ℝ := (2, 3)
def vector_AD (m : ℝ) : ℝ × ℝ := (m, -3)

-- Conditions
def collinear (B C D : ℝ × ℝ) : Prop := ∃ k : ℝ, B = k • C ∨ C = k • D ∨ D = k • B

-- Main statement to prove
theorem find_m (h1 : vector_AB = (-1, 2))
               (h2 : vector_AC = (2, 3))
               (h3 : vector_AD m = (m, -3))
               (h4 : collinear vector_AB vector_AC (vector_AD m)) :
  m = -16 :=
sorry

end NUMINAMATH_GPT_find_m_l1918_191857


namespace NUMINAMATH_GPT_sequence_expression_l1918_191885

-- Given conditions
variable (S : ℕ → ℝ)
variable (a : ℕ → ℝ)
variable (h1 : ∀ n, S n = (1/4) * (a n + 1)^2)

-- Theorem statement
theorem sequence_expression (n : ℕ) : a n = 2 * n - 1 :=
sorry

end NUMINAMATH_GPT_sequence_expression_l1918_191885


namespace NUMINAMATH_GPT_farmer_payment_per_acre_l1918_191889

-- Define the conditions
def monthly_payment : ℝ := 300
def length_ft : ℝ := 360
def width_ft : ℝ := 1210
def sqft_per_acre : ℝ := 43560

-- Define the question and its correct answer
def payment_per_acre_per_month : ℝ := 30

-- Prove that the farmer pays $30 per acre per month
theorem farmer_payment_per_acre :
  (monthly_payment / ((length_ft * width_ft) / sqft_per_acre)) = payment_per_acre_per_month :=
by
  sorry

end NUMINAMATH_GPT_farmer_payment_per_acre_l1918_191889


namespace NUMINAMATH_GPT_area_ratio_of_regular_polygons_l1918_191848

noncomputable def area_ratio (r : ℝ) : ℝ :=
  let A6 := (3 * Real.sqrt 3 / 2) * r^2
  let s8 := r * Real.sqrt (2 - Real.sqrt 2)
  let A8 := 2 * (1 + Real.sqrt 2) * (s8 ^ 2)
  A8 / A6

theorem area_ratio_of_regular_polygons (r : ℝ) :
  area_ratio r = 4 * (1 + Real.sqrt 2) * (2 - Real.sqrt 2) / (3 * Real.sqrt 3) :=
  sorry

end NUMINAMATH_GPT_area_ratio_of_regular_polygons_l1918_191848


namespace NUMINAMATH_GPT_john_guests_count_l1918_191843

def venue_cost : ℕ := 10000
def cost_per_guest : ℕ := 500
def additional_fractional_guests : ℝ := 0.60
def total_cost_when_wife_gets_her_way : ℕ := 50000

theorem john_guests_count (G : ℕ) :
  venue_cost + cost_per_guest * (1 + additional_fractional_guests) * G = 
  total_cost_when_wife_gets_her_way →
  G = 50 :=
by
  sorry

end NUMINAMATH_GPT_john_guests_count_l1918_191843


namespace NUMINAMATH_GPT_fans_per_bleacher_l1918_191871

theorem fans_per_bleacher 
  (total_fans : ℕ) 
  (sets_of_bleachers : ℕ) 
  (h_total : total_fans = 2436) 
  (h_sets : sets_of_bleachers = 3) : 
  total_fans / sets_of_bleachers = 812 := 
by 
  sorry

end NUMINAMATH_GPT_fans_per_bleacher_l1918_191871


namespace NUMINAMATH_GPT_swim_speed_in_still_water_l1918_191875

-- Definitions from conditions
def downstream_speed (v_man v_stream : ℝ) : ℝ := v_man + v_stream
def upstream_speed (v_man v_stream : ℝ) : ℝ := v_man - v_stream

-- Question formatted as a proof problem
theorem swim_speed_in_still_water (v_man v_stream : ℝ)
  (h1 : downstream_speed v_man v_stream = 6)
  (h2 : upstream_speed v_man v_stream = 10) : v_man = 8 :=
by
  -- The proof will come here
  sorry

end NUMINAMATH_GPT_swim_speed_in_still_water_l1918_191875


namespace NUMINAMATH_GPT_common_root_equation_l1918_191872

theorem common_root_equation {m : ℝ} (x : ℝ) (h1 : m * x - 1000 = 1001) (h2 : 1001 * x = m - 1000 * x) : m = 2001 ∨ m = -2001 :=
by
  -- Skipping the proof details
  sorry

end NUMINAMATH_GPT_common_root_equation_l1918_191872


namespace NUMINAMATH_GPT_find_other_root_l1918_191813

theorem find_other_root (m : ℝ) (α : ℝ) :
  (Polynomial.C 1 * Polynomial.X^2 + Polynomial.C m * Polynomial.X - Polynomial.C 10 = 0) →
  (α = -5) →
  ∃ β : ℝ, (α + β = -m) ∧ (α * β = -10) :=
by 
  sorry

end NUMINAMATH_GPT_find_other_root_l1918_191813


namespace NUMINAMATH_GPT_remainder_div_741147_6_l1918_191826

theorem remainder_div_741147_6 : 741147 % 6 = 3 :=
by
  sorry

end NUMINAMATH_GPT_remainder_div_741147_6_l1918_191826


namespace NUMINAMATH_GPT_unknown_number_value_l1918_191824

theorem unknown_number_value (x n : ℝ) (h1 : 0.75 / x = n / 8) (h2 : x = 2) : n = 3 :=
by
  sorry

end NUMINAMATH_GPT_unknown_number_value_l1918_191824


namespace NUMINAMATH_GPT_sum_of_squares_and_product_pos_ints_l1918_191881

variable (x y : ℕ)

theorem sum_of_squares_and_product_pos_ints :
  x^2 + y^2 = 289 ∧ x * y = 120 → x + y = 23 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_sum_of_squares_and_product_pos_ints_l1918_191881


namespace NUMINAMATH_GPT_average_of_original_set_l1918_191817

theorem average_of_original_set
  (A : ℝ)
  (n : ℕ)
  (B : ℝ)
  (h1 : n = 7)
  (h2 : B = 5 * A)
  (h3 : B / n = 100)
  : A = 20 :=
by
  sorry

end NUMINAMATH_GPT_average_of_original_set_l1918_191817


namespace NUMINAMATH_GPT_expand_and_simplify_expression_l1918_191880

theorem expand_and_simplify_expression : 
  ∀ (x : ℝ), (3 * x - 4) * (2 * x + 6) = 6 * x^2 + 10 * x - 24 := 
by 
  intro x
  sorry

end NUMINAMATH_GPT_expand_and_simplify_expression_l1918_191880


namespace NUMINAMATH_GPT_Janice_age_l1918_191850

theorem Janice_age (x : ℝ) (h : x + 12 = 8 * (x - 2)) : x = 4 := by
  sorry

end NUMINAMATH_GPT_Janice_age_l1918_191850


namespace NUMINAMATH_GPT_quiz_competition_l1918_191831

theorem quiz_competition (x : ℕ) :
  (10 * x - 4 * (20 - x) ≥ 88) ↔ (x ≥ 12) :=
by 
  sorry

end NUMINAMATH_GPT_quiz_competition_l1918_191831


namespace NUMINAMATH_GPT_total_pencils_correct_total_erasers_correct_total_rulers_correct_total_sharpeners_correct_total_skittles_correct_l1918_191890

-- Given conditions
def kids_A := 7
def kids_B := 9
def kids_C := 5

def pencils_per_child_A := 4
def erasers_per_child_A := 2
def skittles_per_child_A := 13

def pencils_per_child_B := 6
def rulers_per_child_B := 1
def skittles_per_child_B := 8

def pencils_per_child_C := 3
def sharpeners_per_child_C := 1
def skittles_per_child_C := 15

-- Calculated totals
def total_pencils := kids_A * pencils_per_child_A + kids_B * pencils_per_child_B + kids_C * pencils_per_child_C
def total_erasers := kids_A * erasers_per_child_A
def total_rulers := kids_B * rulers_per_child_B
def total_sharpeners := kids_C * sharpeners_per_child_C
def total_skittles := kids_A * skittles_per_child_A + kids_B * skittles_per_child_B + kids_C * skittles_per_child_C

-- Proof obligations
theorem total_pencils_correct : total_pencils = 97 := by
  sorry

theorem total_erasers_correct : total_erasers = 14 := by
  sorry

theorem total_rulers_correct : total_rulers = 9 := by
  sorry

theorem total_sharpeners_correct : total_sharpeners = 5 := by
  sorry

theorem total_skittles_correct : total_skittles = 238 := by
  sorry

end NUMINAMATH_GPT_total_pencils_correct_total_erasers_correct_total_rulers_correct_total_sharpeners_correct_total_skittles_correct_l1918_191890


namespace NUMINAMATH_GPT_problem1_problem2_l1918_191898

-- Problem 1
theorem problem1 (x : ℝ) : (1 : ℝ) * (-2 * x^2)^3 + x^2 * x^4 - (-3 * x^3)^2 = -16 * x^6 := 
sorry

-- Problem 2
theorem problem2 (a b : ℝ) : (a - b)^2 * (b - a)^4 + (b - a)^3 * (a - b)^3 = 0 := 
sorry

end NUMINAMATH_GPT_problem1_problem2_l1918_191898


namespace NUMINAMATH_GPT_det_dilation_matrix_l1918_191844

section DilationMatrixProof

def E : Matrix (Fin 3) (Fin 3) ℝ := !![5, 0, 0; 0, 5, 0; 0, 0, 5]

theorem det_dilation_matrix :
  Matrix.det E = 125 :=
by {
  sorry
}

end DilationMatrixProof

end NUMINAMATH_GPT_det_dilation_matrix_l1918_191844


namespace NUMINAMATH_GPT_range_of_p_add_q_l1918_191888

theorem range_of_p_add_q (p q : ℝ) :
  (∀ x : ℝ, ¬(x^2 + 2 * p * x - (q^2 - 2) = 0)) → 
  (p + q) ∈ Set.Ioo (-2 : ℝ) (2 : ℝ) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_range_of_p_add_q_l1918_191888


namespace NUMINAMATH_GPT_proper_sets_exist_l1918_191803

def proper_set (weights : List ℕ) : Prop :=
  ∀ w : ℕ, (1 ≤ w ∧ w ≤ 500) → ∃ (used_weights : List ℕ), (used_weights ⊆ weights) ∧ (used_weights.sum = w ∧ ∀ (alternative_weights : List ℕ), (alternative_weights ⊆ weights ∧ alternative_weights.sum = w) → used_weights = alternative_weights)

theorem proper_sets_exist (weights : List ℕ) :
  (weights.sum = 500) → 
  ∃ (sets : List (List ℕ)), sets.length = 3 ∧ (∀ s ∈ sets, proper_set s) :=
by
  sorry

end NUMINAMATH_GPT_proper_sets_exist_l1918_191803


namespace NUMINAMATH_GPT_red_grapes_in_salad_l1918_191849

theorem red_grapes_in_salad {G R B : ℕ} 
  (h1 : R = 3 * G + 7)
  (h2 : B = G - 5)
  (h3 : G + R + B = 102) : R = 67 :=
sorry

end NUMINAMATH_GPT_red_grapes_in_salad_l1918_191849


namespace NUMINAMATH_GPT_peanuts_added_correct_l1918_191878

-- Define the initial and final number of peanuts
def initial_peanuts : ℕ := 4
def final_peanuts : ℕ := 12

-- Define the number of peanuts Mary added
def peanuts_added : ℕ := final_peanuts - initial_peanuts

-- State the theorem that proves the number of peanuts Mary added
theorem peanuts_added_correct : peanuts_added = 8 :=
by
  -- Add the proof here
  sorry

end NUMINAMATH_GPT_peanuts_added_correct_l1918_191878


namespace NUMINAMATH_GPT_p_implies_q_l1918_191825

theorem p_implies_q (x : ℝ) :
  (|2*x - 3| < 1) → (x*(x - 3) < 0) :=
by
  intros hp
  sorry

end NUMINAMATH_GPT_p_implies_q_l1918_191825


namespace NUMINAMATH_GPT_tan_add_l1918_191823

theorem tan_add (α β : ℝ) (h1 : Real.tan (α - π / 6) = 3 / 7) (h2 : Real.tan (π / 6 + β) = 2 / 5) : Real.tan (α + β) = 1 := by
  sorry

end NUMINAMATH_GPT_tan_add_l1918_191823


namespace NUMINAMATH_GPT_initial_plan_days_l1918_191804

-- Define the given conditions in Lean
variables (D : ℕ) -- Initial planned days for completing the job
variables (P : ℕ) -- Number of people initially hired
variables (Q : ℕ) -- Number of people fired
variables (W1 : ℚ) -- Portion of the work done before firing people
variables (D1 : ℕ) -- Days taken to complete W1 portion of work
variables (W2 : ℚ) -- Remaining portion of the work done after firing people
variables (D2 : ℕ) -- Days taken to complete W2 portion of work

-- Conditions from the problem
axiom h1 : P = 10
axiom h2 : Q = 2
axiom h3 : W1 = 1 / 4
axiom h4 : D1 = 20
axiom h5 : W2 = 3 / 4
axiom h6 : D2 = 75

-- The main theorem that proves the total initially planned days were 80
theorem initial_plan_days : D = 80 :=
sorry

end NUMINAMATH_GPT_initial_plan_days_l1918_191804


namespace NUMINAMATH_GPT_cos_eq_cos_of_n_l1918_191851

theorem cos_eq_cos_of_n (n : ℕ) (h1 : 0 ≤ n) (h2 : n ≤ 180) (h3 : Real.cos (n * Real.pi / 180) = Real.cos (283 * Real.pi / 180)) : n = 77 :=
by sorry

end NUMINAMATH_GPT_cos_eq_cos_of_n_l1918_191851


namespace NUMINAMATH_GPT_investment_rate_l1918_191896

theorem investment_rate (total_investment income1_rate income2_rate income_total remaining_investment expected_income : ℝ)
  (h1 : total_investment = 12000)
  (h2 : income1_rate = 0.03)
  (h3 : income2_rate = 0.045)
  (h4 : expected_income = 600)
  (h5 : (5000 * income1_rate + 4000 * income2_rate) = 330)
  (h6 : remaining_investment = total_investment - 5000 - 4000) :
  (remaining_investment * 0.09 = expected_income - (5000 * income1_rate + 4000 * income2_rate)) :=
by
  sorry

end NUMINAMATH_GPT_investment_rate_l1918_191896


namespace NUMINAMATH_GPT_measurement_units_correct_l1918_191808

structure Measurement (A : Type) where
  value : A
  unit : String

def height_of_desk : Measurement ℕ := ⟨70, "centimeters"⟩
def weight_of_apple : Measurement ℕ := ⟨240, "grams"⟩
def duration_of_soccer_game : Measurement ℕ := ⟨90, "minutes"⟩
def dad_daily_work_duration : Measurement ℕ := ⟨8, "hours"⟩

theorem measurement_units_correct :
  height_of_desk.unit = "centimeters" ∧
  weight_of_apple.unit = "grams" ∧
  duration_of_soccer_game.unit = "minutes" ∧
  dad_daily_work_duration.unit = "hours" :=
by
  sorry

end NUMINAMATH_GPT_measurement_units_correct_l1918_191808


namespace NUMINAMATH_GPT_min_expression_value_l1918_191887

theorem min_expression_value (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_sum : a + 2 * b = 1) : 
  ∃ x, (x = (a^2 + 1) / a + (2 * b^2 + 1) / b) ∧ x = 4 + 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_min_expression_value_l1918_191887


namespace NUMINAMATH_GPT_intersecting_chords_length_l1918_191897

theorem intersecting_chords_length
  (h1 : ∃ c1 c2 : ℝ, c1 = 8 ∧ c2 = x + 4 * x ∧ x = 2)
  (h2 : ∀ (a b c d : ℝ), a * b = c * d → a = 4 ∧ b = 4 ∧ c = x ∧ d = 4 * x ∧ x = 2):
  (10 : ℝ) = (x + 4 * x) := by
  sorry

end NUMINAMATH_GPT_intersecting_chords_length_l1918_191897


namespace NUMINAMATH_GPT_license_plate_difference_l1918_191840

theorem license_plate_difference :
  (26^3 * 10^4) - (26^4 * 10^3) = -281216000 :=
by
  sorry

end NUMINAMATH_GPT_license_plate_difference_l1918_191840


namespace NUMINAMATH_GPT_time_for_new_circle_l1918_191819

theorem time_for_new_circle 
  (rounds : ℕ) (time : ℕ) (k : ℕ) (original_time_per_round new_time_per_round : ℝ) 
  (h1 : rounds = 8) 
  (h2 : time = 40) 
  (h3 : k = 10) 
  (h4 : original_time_per_round = time / rounds)
  (h5 : new_time_per_round = original_time_per_round * k) :
  new_time_per_round = 50 :=
by {
  sorry
}

end NUMINAMATH_GPT_time_for_new_circle_l1918_191819


namespace NUMINAMATH_GPT_speed_boat_upstream_l1918_191827

-- Define the conditions provided in the problem
def V_b : ℝ := 8.5  -- Speed of the boat in still water (in km/hr)
def V_downstream : ℝ := 13 -- Speed of the boat downstream (in km/hr)
def V_s : ℝ := V_downstream - V_b  -- Speed of the stream (in km/hr), derived from V_downstream and V_b
def V_upstream (V_b : ℝ) (V_s : ℝ) : ℝ := V_b - V_s  -- Speed of the boat upstream (in km/hr)

-- Statement to prove: the speed of the boat upstream is 4 km/hr
theorem speed_boat_upstream :
  V_upstream V_b V_s = 4 :=
by
  -- This line is for illustration, replace with an actual proof
  sorry

end NUMINAMATH_GPT_speed_boat_upstream_l1918_191827


namespace NUMINAMATH_GPT_opposite_of_2023_is_minus_2023_l1918_191820

def opposite (x y : ℤ) : Prop := x + y = 0

theorem opposite_of_2023_is_minus_2023 : opposite 2023 (-2023) :=
by
  sorry

end NUMINAMATH_GPT_opposite_of_2023_is_minus_2023_l1918_191820


namespace NUMINAMATH_GPT_students_in_each_group_is_9_l1918_191895

-- Define the number of students trying out for the trivia teams
def total_students : ℕ := 36

-- Define the number of students who didn't get picked for the team
def students_not_picked : ℕ := 9

-- Define the number of groups the remaining students are divided into
def number_of_groups : ℕ := 3

-- Define the function that calculates the number of students in each group
def students_per_group (total students_not_picked number_of_groups : ℕ) : ℕ :=
  (total - students_not_picked) / number_of_groups

-- Theorem: Given the conditions, the number of students in each group is 9
theorem students_in_each_group_is_9 : students_per_group total_students students_not_picked number_of_groups = 9 := 
by 
  -- proof skipped
  sorry

end NUMINAMATH_GPT_students_in_each_group_is_9_l1918_191895


namespace NUMINAMATH_GPT_unknown_rate_of_blankets_l1918_191852

theorem unknown_rate_of_blankets (x : ℝ) :
  2 * 100 + 5 * 150 + 2 * x = 9 * 150 → x = 200 :=
by
  sorry

end NUMINAMATH_GPT_unknown_rate_of_blankets_l1918_191852


namespace NUMINAMATH_GPT_part1_daily_sales_profit_part2_maximum_daily_profit_l1918_191860

-- Definitions of initial conditions
def original_price : ℝ := 30
def original_sales_volume : ℝ := 60
def cost_price : ℝ := 15
def price_reduction_effect : ℝ := 10

-- Part 1: Prove the daily sales profit if the price is reduced by 2 yuan
def new_price_after_reduction (reduction : ℝ) : ℝ := original_price - reduction
def new_sales_volume (reduction : ℝ) : ℝ := original_sales_volume + reduction * price_reduction_effect
def profit_per_kg (selling_price : ℝ) : ℝ := selling_price - cost_price
def daily_sales_profit (reduction : ℝ) : ℝ := profit_per_kg (new_price_after_reduction reduction) * new_sales_volume reduction

theorem part1_daily_sales_profit : daily_sales_profit 2 = 1040 := by sorry

-- Part 2: Prove the selling price for maximum profit and the maximum profit
def profit_function (x : ℝ) : ℝ := (x - cost_price) * (original_sales_volume + (original_price - x) * price_reduction_effect)

theorem part2_maximum_daily_profit : 
  ∃ x, profit_function x = 1102.5 ∧ x = 51 / 2 := by sorry

end NUMINAMATH_GPT_part1_daily_sales_profit_part2_maximum_daily_profit_l1918_191860


namespace NUMINAMATH_GPT_heaviest_person_is_42_27_l1918_191856

-- Define the main parameters using the conditions
def heaviest_person_weight (M : ℕ → Real) (P : ℕ → Real) (Q : ℕ → Real) (H : P 2 = 7.7) (L : Q 3 = 4.8) (S : P 1 + P 2 + P 3 = 106.6) : Real :=
  let h := P 2 + 7.7
  h

-- State the theorem
theorem heaviest_person_is_42_27 (M : ℕ → Real) (P : ℕ → Real) (Q : ℕ → Real)
  (H : P 2 = 7.7) (L : Q 3 = 4.8) (S : P 1 + P 2 + P 3 = 106.6) :
  heaviest_person_weight M P Q H L S = 42.27 :=
sorry

end NUMINAMATH_GPT_heaviest_person_is_42_27_l1918_191856


namespace NUMINAMATH_GPT_simplify_polynomial_subtraction_l1918_191863

/--
  Given the polynomials (2 * x^6 + x^5 + 3 * x^4 + x^3 + 8) and (x^6 + 2 * x^5 - 2 * x^4 + x^2 + 5),
  prove that their difference simplifies to x^6 - x^5 + 5 * x^4 + x^3 - x^2 + 3.
-/
theorem simplify_polynomial_subtraction  (x : ℝ) :
  (2 * x^6 + x^5 + 3 * x^4 + x^3 + 8) - (x^6 + 2 * x^5 - 2 * x^4 + x^2 + 5) = x^6 - x^5 + 5 * x^4 + x^3 - x^2 + 3 :=
sorry

end NUMINAMATH_GPT_simplify_polynomial_subtraction_l1918_191863


namespace NUMINAMATH_GPT_consecutive_numbers_sum_39_l1918_191847

theorem consecutive_numbers_sum_39 (n : ℕ) (hn : n + (n + 1) = 39) : n + 1 = 20 :=
sorry

end NUMINAMATH_GPT_consecutive_numbers_sum_39_l1918_191847


namespace NUMINAMATH_GPT_tank_empty_time_l1918_191809

def tank_capacity : ℝ := 6480
def leak_time : ℝ := 6
def inlet_rate_per_minute : ℝ := 4.5
def inlet_rate_per_hour : ℝ := inlet_rate_per_minute * 60

theorem tank_empty_time : tank_capacity / (tank_capacity / leak_time - inlet_rate_per_hour) = 8 := 
by
  sorry

end NUMINAMATH_GPT_tank_empty_time_l1918_191809


namespace NUMINAMATH_GPT_log_relationship_l1918_191839

noncomputable def log_m (m x : ℝ) : ℝ := Real.log x / Real.log m

theorem log_relationship (m : ℝ) (h₀ : 0 < m) (h₁ : m < 1) :
  log_m m 0.3 > log_m m 0.5 :=
by
  sorry

end NUMINAMATH_GPT_log_relationship_l1918_191839


namespace NUMINAMATH_GPT_a_pow_m_minus_a_pow_n_divisible_by_30_l1918_191818

theorem a_pow_m_minus_a_pow_n_divisible_by_30
  (a m n k : ℕ)
  (h_n_ge_two : n ≥ 2)
  (h_m_gt_n : m > n)
  (h_m_n_diff : m = n + 4 * k) :
  30 ∣ (a ^ m - a ^ n) :=
sorry

end NUMINAMATH_GPT_a_pow_m_minus_a_pow_n_divisible_by_30_l1918_191818


namespace NUMINAMATH_GPT_minimum_x2y3z_l1918_191867

theorem minimum_x2y3z (x y z : ℕ) (h_pos : x > 0 ∧ y > 0 ∧ z > 0) 
  (h_eq : x^3 + y^3 + z^3 - 3 * x * y * z = 607) : 
  x + 2 * y + 3 * z ≥ 1215 :=
sorry

end NUMINAMATH_GPT_minimum_x2y3z_l1918_191867


namespace NUMINAMATH_GPT_cubic_identity_l1918_191805

theorem cubic_identity (x : ℝ) (h : x + (1/x) = -3) : x^3 + (1/x^3) = -18 :=
by
  sorry

end NUMINAMATH_GPT_cubic_identity_l1918_191805


namespace NUMINAMATH_GPT_temperature_on_Tuesday_l1918_191893

variable (T W Th F : ℝ)

theorem temperature_on_Tuesday :
  (T + W + Th) / 3 = 52 →
  (W + Th + F) / 3 = 54 →
  F = 53 →
  T = 47 := by
  intros h₁ h₂ h₃
  sorry

end NUMINAMATH_GPT_temperature_on_Tuesday_l1918_191893


namespace NUMINAMATH_GPT_tangent_line_eqn_l1918_191877

theorem tangent_line_eqn 
  (x y : ℝ)
  (H_curve : y = x^3 + 3 * x^2 - 5)
  (H_point : (x, y) = (-1, -3)) :
  (3 * x + y + 6 = 0) := 
sorry

end NUMINAMATH_GPT_tangent_line_eqn_l1918_191877


namespace NUMINAMATH_GPT_product_eq_1519000000_div_6561_l1918_191855

-- Given conditions
def P (X : ℚ) : ℚ := X - 5
def Q (X : ℚ) : ℚ := X + 5
def R (X : ℚ) : ℚ := X / 2
def S (X : ℚ) : ℚ := 2 * X

theorem product_eq_1519000000_div_6561 
  (X : ℚ) 
  (h : (P X) + (Q X) + (R X) + (S X) = 100) :
  (P X) * (Q X) * (R X) * (S X) = 1519000000 / 6561 := 
by sorry

end NUMINAMATH_GPT_product_eq_1519000000_div_6561_l1918_191855


namespace NUMINAMATH_GPT_charlie_has_54_crayons_l1918_191806

theorem charlie_has_54_crayons
  (crayons_Billie : ℕ)
  (crayons_Bobbie : ℕ)
  (crayons_Lizzie : ℕ)
  (crayons_Charlie : ℕ)
  (h1 : crayons_Billie = 18)
  (h2 : crayons_Bobbie = 3 * crayons_Billie)
  (h3 : crayons_Lizzie = crayons_Bobbie / 2)
  (h4 : crayons_Charlie = 2 * crayons_Lizzie) : 
  crayons_Charlie = 54 := 
sorry

end NUMINAMATH_GPT_charlie_has_54_crayons_l1918_191806


namespace NUMINAMATH_GPT_find_omega_l1918_191829

noncomputable def f (x : ℝ) (ω φ : ℝ) := Real.sin (ω * x + φ)

theorem find_omega (ω φ : ℝ) (hω : ω > 0) (hφ : 0 ≤ φ ∧ φ ≤ π)
  (h_even : ∀ x : ℝ, f x ω φ = f (-x) ω φ)
  (h_symm : ∀ x : ℝ, f (3 * π / 4 + x) ω φ = f (3 * π / 4 - x) ω φ)
  (h_mono : ∀ x1 x2 : ℝ, 0 ≤ x1 → x1 ≤ x2 → x2 ≤ π / 2 → f x1 ω φ ≤ f x2 ω φ) :
  ω = 2 / 3 ∨ ω = 2 :=
sorry

end NUMINAMATH_GPT_find_omega_l1918_191829


namespace NUMINAMATH_GPT_num_of_chords_l1918_191868

theorem num_of_chords (n : ℕ) (h : n = 8) : (n.choose 2) = 28 :=
by
  -- Proof of this theorem will be here
  sorry

end NUMINAMATH_GPT_num_of_chords_l1918_191868


namespace NUMINAMATH_GPT_find_k_l1918_191859

theorem find_k (k : ℝ) : (∃ x : ℝ, k * x^2 - 9 * x + 8 = 0 ∧ x = 1) → k = 1 :=
sorry

end NUMINAMATH_GPT_find_k_l1918_191859


namespace NUMINAMATH_GPT_solve_system_of_equations_l1918_191842

theorem solve_system_of_equations (x y : ℝ) (h1 : x + 2 * y = 6) (h2 : 2 * x + y = 21) : x + y = 9 := by
  sorry

end NUMINAMATH_GPT_solve_system_of_equations_l1918_191842


namespace NUMINAMATH_GPT_grandfather_7_times_older_after_8_years_l1918_191811

theorem grandfather_7_times_older_after_8_years :
  ∃ x : ℕ, ∀ (g_age ng_age : ℕ), 50 < g_age ∧ g_age < 90 ∧ g_age = 31 * ng_age → g_age + x = 7 * (ng_age + x) → x = 8 :=
by
  sorry

end NUMINAMATH_GPT_grandfather_7_times_older_after_8_years_l1918_191811


namespace NUMINAMATH_GPT_perpendicular_slope_l1918_191879

theorem perpendicular_slope :
  ∀ (x y : ℝ), 5 * x - 2 * y = 10 → y = ((5 : ℝ) / 2) * x - 5 → ∃ (m : ℝ), m = - (2 / 5) := by
  sorry

end NUMINAMATH_GPT_perpendicular_slope_l1918_191879


namespace NUMINAMATH_GPT_average_height_of_trees_l1918_191870

-- Define the heights of the trees
def height_tree1: ℕ := 1000
def height_tree2: ℕ := height_tree1 / 2
def height_tree3: ℕ := height_tree1 / 2
def height_tree4: ℕ := height_tree1 + 200

-- Calculate the total number of trees
def number_of_trees: ℕ := 4

-- Compute the total height climbed
def total_height: ℕ := height_tree1 + height_tree2 + height_tree3 + height_tree4

-- Define the average height
def average_height: ℕ := total_height / number_of_trees

-- The theorem statement
theorem average_height_of_trees: average_height = 800 := by
  sorry

end NUMINAMATH_GPT_average_height_of_trees_l1918_191870


namespace NUMINAMATH_GPT_algebraic_expression_value_l1918_191837

theorem algebraic_expression_value (x : ℝ) (h : x = Real.sqrt 23 - 1) : x^2 + 2 * x + 2 = 24 :=
by
  -- Start of the proof
  sorry -- Proof is omitted as per instructions

end NUMINAMATH_GPT_algebraic_expression_value_l1918_191837


namespace NUMINAMATH_GPT_polynomial_remainder_l1918_191832

theorem polynomial_remainder (x : ℝ) :
  (x^4 + 3 * x^2 - 4) % (x^2 + 2) = x^2 - 4 :=
sorry

end NUMINAMATH_GPT_polynomial_remainder_l1918_191832


namespace NUMINAMATH_GPT_work_completion_l1918_191807

theorem work_completion (a b : Type) (work_done_together work_done_by_a work_done_by_b : ℝ) 
  (h1 : work_done_together = 1 / 12) 
  (h2 : work_done_by_a = 1 / 20) 
  (h3 : work_done_by_b = work_done_together - work_done_by_a) : 
  work_done_by_b = 1 / 30 :=
by
  sorry

end NUMINAMATH_GPT_work_completion_l1918_191807


namespace NUMINAMATH_GPT_eggs_divided_l1918_191841

theorem eggs_divided (boxes : ℝ) (eggs_per_box : ℝ) (total_eggs : ℝ) :
  boxes = 2.0 → eggs_per_box = 1.5 → total_eggs = boxes * eggs_per_box → total_eggs = 3.0 :=
by
  intros
  sorry

end NUMINAMATH_GPT_eggs_divided_l1918_191841
