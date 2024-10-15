import Mathlib

namespace NUMINAMATH_GPT_even_num_Z_tetrominoes_l1_128

-- Definitions based on the conditions of the problem
def is_tiled_with_S_tetrominoes (P : Type) : Prop := sorry
def tiling_uses_S_Z_tetrominoes (P : Type) : Prop := sorry
def num_Z_tetrominoes (P : Type) : ℕ := sorry

-- The theorem statement
theorem even_num_Z_tetrominoes (P : Type) 
  (hTiledWithS : is_tiled_with_S_tetrominoes P) 
  (hTilingWithSZ : tiling_uses_S_Z_tetrominoes P) : num_Z_tetrominoes P % 2 = 0 :=
sorry

end NUMINAMATH_GPT_even_num_Z_tetrominoes_l1_128


namespace NUMINAMATH_GPT_find_initial_jellybeans_l1_187

-- Definitions of the initial conditions
def jellybeans_initial (x : ℝ) (days : ℕ) (remaining : ℝ) := 
  days = 4 ∧ remaining = 48 ∧ (0.7 ^ days) * x = remaining

-- The theorem to prove
theorem find_initial_jellybeans (x : ℝ) : 
  jellybeans_initial x 4 48 → x = 200 :=
sorry

end NUMINAMATH_GPT_find_initial_jellybeans_l1_187


namespace NUMINAMATH_GPT_john_total_payment_in_month_l1_150

def daily_pills : ℕ := 2
def cost_per_pill : ℝ := 1.5
def insurance_coverage : ℝ := 0.4
def days_in_month : ℕ := 30

theorem john_total_payment_in_month : john_payment = 54 :=
  let daily_cost := daily_pills * cost_per_pill
  let monthly_cost := daily_cost * days_in_month
  let insurance_paid := monthly_cost * insurance_coverage
  let john_payment := monthly_cost - insurance_paid
  sorry

end NUMINAMATH_GPT_john_total_payment_in_month_l1_150


namespace NUMINAMATH_GPT_probability_not_exceed_60W_l1_149

noncomputable def total_bulbs : ℕ := 250
noncomputable def bulbs_100W : ℕ := 100
noncomputable def bulbs_60W : ℕ := 50
noncomputable def bulbs_25W : ℕ := 50
noncomputable def bulbs_15W : ℕ := 50

noncomputable def probability_of_event (event : ℕ) (total : ℕ) : ℝ := 
  event / total

noncomputable def P_A : ℝ := probability_of_event bulbs_60W total_bulbs
noncomputable def P_B : ℝ := probability_of_event bulbs_25W total_bulbs
noncomputable def P_C : ℝ := probability_of_event bulbs_15W total_bulbs
noncomputable def P_D : ℝ := probability_of_event bulbs_100W total_bulbs

theorem probability_not_exceed_60W : 
  P_A + P_B + P_C = 3 / 5 :=
by
  sorry

end NUMINAMATH_GPT_probability_not_exceed_60W_l1_149


namespace NUMINAMATH_GPT_quadratic_function_n_neg_l1_126

theorem quadratic_function_n_neg (n : ℝ) :
  (∀ x : ℝ, x^2 + 3 * x + n = 0 → x > 0) → n < 0 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_function_n_neg_l1_126


namespace NUMINAMATH_GPT_relationship_between_abc_l1_130

noncomputable def a : Real := Real.sqrt 1.2
noncomputable def b : Real := Real.exp 0.1
noncomputable def c : Real := 1 + Real.log 1.1

theorem relationship_between_abc : b > a ∧ a > c :=
by {
  -- a = sqrt(1.2)
  -- b = exp(0.1)
  -- c = 1 + log(1.1)
  -- We need to prove: b > a > c
  sorry
}

end NUMINAMATH_GPT_relationship_between_abc_l1_130


namespace NUMINAMATH_GPT_susan_spending_ratio_l1_199

theorem susan_spending_ratio (initial_amount clothes_spent books_left books_spent left_after_clothes gcd_ratio : ℤ)
  (h1 : initial_amount = 600)
  (h2 : clothes_spent = initial_amount / 2)
  (h3 : left_after_clothes = initial_amount - clothes_spent)
  (h4 : books_left = 150)
  (h5 : books_spent = left_after_clothes - books_left)
  (h6 : gcd books_spent left_after_clothes = 150)
  (h7 : books_spent / gcd_ratio = 1)
  (h8 : left_after_clothes / gcd_ratio = 2) :
  books_spent / gcd books_spent left_after_clothes = 1 ∧ left_after_clothes / gcd books_spent left_after_clothes = 2 :=
sorry

end NUMINAMATH_GPT_susan_spending_ratio_l1_199


namespace NUMINAMATH_GPT_cosine_double_angle_l1_138

theorem cosine_double_angle
  (θ : ℝ)
  (cos_θ : ℝ)
  (h : cos_θ = 3 / 5) :
  2 * cos_θ^2 - 1 = -7 / 25 :=
by
  sorry

end NUMINAMATH_GPT_cosine_double_angle_l1_138


namespace NUMINAMATH_GPT_solve_quadratic1_solve_quadratic2_l1_167

theorem solve_quadratic1 :
  (∀ x, x^2 + x - 4 = 0 → x = ( -1 + Real.sqrt 17 ) / 2 ∨ x = ( -1 - Real.sqrt 17 ) / 2) := sorry

theorem solve_quadratic2 :
  (∀ x, (2*x + 1)^2 + 15 = 8*(2*x + 1) → x = 1 ∨ x = 2) := sorry

end NUMINAMATH_GPT_solve_quadratic1_solve_quadratic2_l1_167


namespace NUMINAMATH_GPT_visitors_correct_l1_137

def visitors_that_day : ℕ := 92
def visitors_previous_day : ℕ := 419
def total_visitors_before_that_day : ℕ := 522
def visitors_two_days_before : ℕ := total_visitors_before_that_day - visitors_previous_day - visitors_that_day

theorem visitors_correct : visitors_two_days_before = 11 := by
  -- Sorry, proof to be filled in
  sorry

end NUMINAMATH_GPT_visitors_correct_l1_137


namespace NUMINAMATH_GPT_geom_seq_a4_a5_a6_value_l1_125

theorem geom_seq_a4_a5_a6_value (a : ℕ → ℝ) (h_geom : ∃ r, 0 < r ∧ ∀ n, a (n + 1) = r * a n)
  (h_roots : ∃ x y, x * y = 16 ∧ x + y = 10 ∧ a 1 = x ∧ a 9 = y) :
  a 4 * a 5 * a 6 = 64 :=
by
  sorry

end NUMINAMATH_GPT_geom_seq_a4_a5_a6_value_l1_125


namespace NUMINAMATH_GPT_prime_square_mod_six_l1_103

theorem prime_square_mod_six (p : ℕ) (hp : Nat.Prime p) (h : p > 5) : p^2 % 6 = 1 :=
by
  sorry

end NUMINAMATH_GPT_prime_square_mod_six_l1_103


namespace NUMINAMATH_GPT_weight_of_new_person_l1_143

theorem weight_of_new_person (avg_increase : ℝ) (num_persons : ℕ) (old_weight : ℝ) (new_weight : ℝ) :
  avg_increase = 2.5 → num_persons = 8 → old_weight = 60 → 
  new_weight = old_weight + num_persons * avg_increase → new_weight = 80 :=
  by
    intros
    sorry

end NUMINAMATH_GPT_weight_of_new_person_l1_143


namespace NUMINAMATH_GPT_true_discount_correct_l1_152

noncomputable def true_discount (FV BD : ℝ) : ℝ :=
  (BD * FV) / (BD + FV)

theorem true_discount_correct :
  true_discount 270 54 = 45 :=
by
  sorry

end NUMINAMATH_GPT_true_discount_correct_l1_152


namespace NUMINAMATH_GPT_find_a_and_other_root_l1_145

-- Define the quadratic equation with a
def quadratic_eq (a x : ℝ) : ℝ := (a + 1) * x^2 + x - 1

-- Define the conditions where -1 is a root
def condition (a : ℝ) : Prop := quadratic_eq a (-1) = 0

theorem find_a_and_other_root (a : ℝ) :
  condition a → 
  (a = 1 ∧ ∃ x : ℝ, x ≠ -1 ∧ quadratic_eq 1 x = 0 ∧ x = 1 / 2) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_a_and_other_root_l1_145


namespace NUMINAMATH_GPT_abc_sum_16_l1_164

theorem abc_sum_16 (a b c : ℕ) (h1 : a ≥ 4) (h2 : b ≥ 4) (h3 : c ≥ 4) (h4 : a ≠ b ∨ b ≠ c ∨ a ≠ c)
  (h5 : 4 * a * b * c = (a + 3) * (b + 3) * (c + 3)) : a + b + c = 16 :=
by
  sorry

end NUMINAMATH_GPT_abc_sum_16_l1_164


namespace NUMINAMATH_GPT_angle_B_measure_l1_173

open Real EuclideanGeometry Classical

noncomputable def measure_angle_B (A C : ℝ) : ℝ := 180 - (180 - A - C)

theorem angle_B_measure
  (l m : ℝ → ℝ → Prop) -- parallel lines l and m (can be interpreted as propositions for simplicity)
  (h_parallel : ∀ x y, l x y → m x y → x = y) -- Lines l and m are parallel
  (A C : ℝ)
  (hA : A = 120)
  (hC : C = 70) :
  measure_angle_B A C = 130 := 
by
  sorry

end NUMINAMATH_GPT_angle_B_measure_l1_173


namespace NUMINAMATH_GPT_isabel_initial_candy_l1_106

theorem isabel_initial_candy (total_candy : ℕ) (candy_given : ℕ) (initial_candy : ℕ) :
  candy_given = 25 → total_candy = 93 → total_candy = initial_candy + candy_given → initial_candy = 68 :=
by
  intros h_candy_given h_total_candy h_eq
  rw [h_candy_given, h_total_candy] at h_eq
  sorry

end NUMINAMATH_GPT_isabel_initial_candy_l1_106


namespace NUMINAMATH_GPT_compute_c_plus_d_l1_142

theorem compute_c_plus_d (c d : ℝ) 
  (h1 : c^3 - 18 * c^2 + 25 * c - 75 = 0) 
  (h2 : 9 * d^3 - 72 * d^2 - 345 * d + 3060 = 0) : 
  c + d = 10 := 
sorry

end NUMINAMATH_GPT_compute_c_plus_d_l1_142


namespace NUMINAMATH_GPT_system_solution_a_l1_117

theorem system_solution_a (x y z : ℤ) (h1 : x^2 + x * y + y^2 = 7) (h2 : y^2 + y * z + z^2 = 13) (h3 : z^2 + z * x + x^2 = 19) :
  (x = 2 ∧ y = 1 ∧ z = 3) ∨ (x = -2 ∧ y = -1 ∧ z = -3) :=
sorry

end NUMINAMATH_GPT_system_solution_a_l1_117


namespace NUMINAMATH_GPT_min_distance_eq_5_l1_124

-- Define the conditions
def condition1 (a b : ℝ) : Prop := b = 4 * Real.log a - a^2
def condition2 (c d : ℝ) : Prop := d = 2 * c + 2

-- Define the function to prove the minimum value
def minValue (a b c d : ℝ) : ℝ := (a - c)^2 + (b - d)^2

-- The main theorem statement
theorem min_distance_eq_5 (a b c d : ℝ) (ha : a > 0) (h1: condition1 a b) (h2: condition2 c d) : 
  ∃ a c b d, minValue a b c d = 5 := 
sorry

end NUMINAMATH_GPT_min_distance_eq_5_l1_124


namespace NUMINAMATH_GPT_minimum_bench_sections_l1_170

theorem minimum_bench_sections (N : ℕ) (hN : 8 * N = 12 * N) : N = 3 :=
sorry

end NUMINAMATH_GPT_minimum_bench_sections_l1_170


namespace NUMINAMATH_GPT_fraction_of_grid_covered_by_triangle_l1_183

noncomputable def area_of_triangle (A B C : ℝ × ℝ) : ℝ :=
  0.5 * |(A.1 * (B.2 - C.2)) + (B.1 * (C.2 - A.2)) + (C.1 * (A.2 - B.2))|

noncomputable def area_of_grid : ℝ := 7 * 6

noncomputable def fraction_covered : ℝ :=
  area_of_triangle (-1, 2) (3, 5) (2, 2) / area_of_grid

theorem fraction_of_grid_covered_by_triangle : fraction_covered = (3 / 28) :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_grid_covered_by_triangle_l1_183


namespace NUMINAMATH_GPT_number_of_australians_l1_181

-- Conditions are given here as definitions
def total_people : ℕ := 49
def number_americans : ℕ := 16
def number_chinese : ℕ := 22

-- Goal is to prove the number of Australians is 11
theorem number_of_australians : total_people - (number_americans + number_chinese) = 11 := by
  sorry

end NUMINAMATH_GPT_number_of_australians_l1_181


namespace NUMINAMATH_GPT_plates_used_l1_155

def plates_per_course : ℕ := 2
def courses_breakfast : ℕ := 2
def courses_lunch : ℕ := 2
def courses_dinner : ℕ := 3
def courses_late_snack : ℕ := 3
def courses_per_day : ℕ := courses_breakfast + courses_lunch + courses_dinner + courses_late_snack
def plates_per_day : ℕ := courses_per_day * plates_per_course

def parents_and_siblings_stay : ℕ := 6
def grandparents_stay : ℕ := 4
def cousins_stay : ℕ := 3

def parents_and_siblings_count : ℕ := 5
def grandparents_count : ℕ := 2
def cousins_count : ℕ := 4

def plates_parents_and_siblings : ℕ := parents_and_siblings_count * plates_per_day * parents_and_siblings_stay
def plates_grandparents : ℕ := grandparents_count * plates_per_day * grandparents_stay
def plates_cousins : ℕ := cousins_count * plates_per_day * cousins_stay

def total_plates_used : ℕ := plates_parents_and_siblings + plates_grandparents + plates_cousins

theorem plates_used (expected : ℕ) : total_plates_used = expected :=
by
  sorry

end NUMINAMATH_GPT_plates_used_l1_155


namespace NUMINAMATH_GPT_valid_passwords_count_l1_184

def total_passwords : Nat := 10 ^ 5
def restricted_passwords : Nat := 10

theorem valid_passwords_count : total_passwords - restricted_passwords = 99990 := by
  sorry

end NUMINAMATH_GPT_valid_passwords_count_l1_184


namespace NUMINAMATH_GPT_max_expression_value_l1_110

theorem max_expression_value (a b c d : ℝ) 
  (h1 : -6.5 ≤ a ∧ a ≤ 6.5) 
  (h2 : -6.5 ≤ b ∧ b ≤ 6.5) 
  (h3 : -6.5 ≤ c ∧ c ≤ 6.5) 
  (h4 : -6.5 ≤ d ∧ d ≤ 6.5) : 
  a + 2*b + c + 2*d - a*b - b*c - c*d - d*a ≤ 182 :=
sorry

end NUMINAMATH_GPT_max_expression_value_l1_110


namespace NUMINAMATH_GPT_a_n_is_arithmetic_sequence_b_n_is_right_sequence_sum_first_n_terms_b_n_l1_115

noncomputable def a_n (n : ℕ) : ℕ := 3 * n

noncomputable def b_n (n : ℕ) : ℕ := 3 * n + 2^(n - 1)

noncomputable def S_n (n : ℕ) : ℕ := (3 * n * (n + 1) / 2) + (2^n - 1)

theorem a_n_is_arithmetic_sequence (n : ℕ) :
  (a_n 1 = 3) ∧ (a_n 4 = 12) ∧ (∀ n : ℕ, a_n n = 3 * n) :=
by
  sorry

theorem b_n_is_right_sequence (n : ℕ) :
  (b_n 1 = 4) ∧ (b_n 4 = 20) ∧ (∀ n : ℕ, b_n n = 3 * n + 2^(n - 1)) ∧ 
  (∀ n : ℕ, b_n n - a_n n = 2^(n - 1)) :=
by
  sorry

theorem sum_first_n_terms_b_n (n : ℕ) :
  S_n n = 3 * (n * (n + 1) / 2) + 2^n - 1 :=
by
  sorry

end NUMINAMATH_GPT_a_n_is_arithmetic_sequence_b_n_is_right_sequence_sum_first_n_terms_b_n_l1_115


namespace NUMINAMATH_GPT_find_triangle_C_coordinates_find_triangle_area_l1_134

noncomputable def triangle_C_coordinates (A B : (ℝ × ℝ)) (median_eq altitude_eq : (ℝ × ℝ × ℝ)) : Prop :=
  ∃ C : ℝ × ℝ, C = (3, 1) ∧
    let A := (1,2)
    let B := (3, 4)
    let median_eq := (2, 1, -7)
    let altitude_eq := (2, -1, -2)
    true

noncomputable def triangle_area (A B C : (ℝ × ℝ)) : Prop :=
  ∃ S : ℝ, S = 3 ∧
    let A := (1,2)
    let B := (3, 4)
    let C := (3, 1)
    true

theorem find_triangle_C_coordinates : triangle_C_coordinates (1,2) (3,4) (2, 1, -7) (2, -1, -2) :=
by { sorry }

theorem find_triangle_area : triangle_area (1,2) (3,4) (3,1) :=
by { sorry }

end NUMINAMATH_GPT_find_triangle_C_coordinates_find_triangle_area_l1_134


namespace NUMINAMATH_GPT_dot_product_not_sufficient_nor_necessary_for_parallel_l1_176

open Real

-- Definitions for plane vectors \overrightarrow{a} and \overrightarrow{b}
variables (a b : ℝ × ℝ)

-- Dot product definition for two plane vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Parallelism condition for plane vectors
def parallel (v1 v2 : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v1 = (k • v2) ∨ v2 = (k • v1)

-- Statement to be proved
theorem dot_product_not_sufficient_nor_necessary_for_parallel :
  ¬ (∀ a b : ℝ × ℝ, (dot_product a b > 0) ↔ (parallel a b)) :=
sorry

end NUMINAMATH_GPT_dot_product_not_sufficient_nor_necessary_for_parallel_l1_176


namespace NUMINAMATH_GPT_product_mod_9_l1_196

theorem product_mod_9 (a b c : ℕ) (h1 : a % 6 = 2) (h2 : b % 7 = 3) (h3 : c % 8 = 4) : (a * b * c) % 9 = 6 :=
by
  sorry

end NUMINAMATH_GPT_product_mod_9_l1_196


namespace NUMINAMATH_GPT_expand_and_simplify_l1_161

variable (y : ℝ)

theorem expand_and_simplify :
  -2 * (5 * y^3 - 4 * y^2 + 3 * y - 6) = -10 * y^3 + 8 * y^2 - 6 * y + 12 :=
  sorry

end NUMINAMATH_GPT_expand_and_simplify_l1_161


namespace NUMINAMATH_GPT_correct_product_l1_163

-- Definitions for conditions
def reversed_product (a b : ℕ) : Prop :=
  let reversed_a := (a % 10) * 10 + (a / 10)
  reversed_a * b = 204

theorem correct_product (a b : ℕ) (h : reversed_product a b) : a * b = 357 := 
by
  sorry

end NUMINAMATH_GPT_correct_product_l1_163


namespace NUMINAMATH_GPT_units_digit_of_sum_64_8_75_8_is_1_l1_174

def units_digit_in_base_8_sum (a b : ℕ) : ℕ :=
  (a + b) % 8

theorem units_digit_of_sum_64_8_75_8_is_1 :
  units_digit_in_base_8_sum 0o64 0o75 = 1 :=
sorry

end NUMINAMATH_GPT_units_digit_of_sum_64_8_75_8_is_1_l1_174


namespace NUMINAMATH_GPT_expanded_polynomial_correct_l1_151

noncomputable def polynomial_product (x : ℚ) : ℚ :=
  (2 * x^3 - 3 * x + 1) * (x^2 + 4 * x + 3)

theorem expanded_polynomial_correct (x : ℚ) : 
  polynomial_product x = 2 * x^5 + 8 * x^4 + 3 * x^3 - 11 * x^2 - 5 * x + 3 := 
by
  sorry

end NUMINAMATH_GPT_expanded_polynomial_correct_l1_151


namespace NUMINAMATH_GPT_factorize_quartic_l1_135

-- Specify that p and q are real numbers (ℝ)
variables {p q : ℝ}

-- Statement: For any real numbers p and q, the polynomial x^4 + p x^2 + q can always be factored into two quadratic polynomials.
theorem factorize_quartic (p q : ℝ) : 
  ∃ a b c d e f : ℝ, (x^2 + a * x + b) * (x^2 + c * x + d) = x^4 + p * x^2 + q :=
sorry

end NUMINAMATH_GPT_factorize_quartic_l1_135


namespace NUMINAMATH_GPT_find_ordered_pairs_l1_133

theorem find_ordered_pairs (x y : ℝ) :
  x^2 * y = 3 ∧ x + x * y = 4 → (x, y) = (1, 3) ∨ (x, y) = (3, 1 / 3) :=
sorry

end NUMINAMATH_GPT_find_ordered_pairs_l1_133


namespace NUMINAMATH_GPT_gcd_6051_10085_l1_123

theorem gcd_6051_10085 : Nat.gcd 6051 10085 = 2017 := by
  sorry

end NUMINAMATH_GPT_gcd_6051_10085_l1_123


namespace NUMINAMATH_GPT_negation_of_p_l1_102

theorem negation_of_p :
  (¬ (∀ x : ℝ, x^3 + 2 < 0)) = ∃ x : ℝ, x^3 + 2 ≥ 0 := 
  by sorry

end NUMINAMATH_GPT_negation_of_p_l1_102


namespace NUMINAMATH_GPT_steps_per_flight_l1_158

-- Define the problem conditions
def jack_flights_up := 3
def jack_flights_down := 6
def steps_height_inches := 8
def jack_height_change_feet := 24

-- Convert the height change to inches
def jack_height_change_inches := jack_height_change_feet * 12

-- Calculate the net flights down
def net_flights_down := jack_flights_down - jack_flights_up

-- Calculate total height change in inches for net flights
def total_height_change_inches := net_flights_down * jack_height_change_inches

-- Calculate the number of steps in each flight
def number_of_steps_per_flight :=
  total_height_change_inches / (steps_height_inches * net_flights_down)

theorem steps_per_flight :
  number_of_steps_per_flight = 108 :=
sorry

end NUMINAMATH_GPT_steps_per_flight_l1_158


namespace NUMINAMATH_GPT_max_value_on_ellipse_l1_157

theorem max_value_on_ellipse (b : ℝ) (hb : b > 0) :
  ∃ (M : ℝ), 
    (∀ (x y : ℝ), (x^2 / 4 + y^2 / b^2 = 1) → x^2 + 2 * y ≤ M) ∧
    ((b ≤ 4 → M = b^2 / 4 + 4) ∧ (b > 4 → M = 2 * b)) :=
  sorry

end NUMINAMATH_GPT_max_value_on_ellipse_l1_157


namespace NUMINAMATH_GPT_living_room_area_is_60_l1_113

-- Define the conditions
def carpet_length : ℝ := 4
def carpet_width : ℝ := 9
def carpet_area : ℝ := carpet_length * carpet_width
def coverage_fraction : ℝ := 0.60

-- Define the target area of the living room floor
def target_living_room_area (A : ℝ) : Prop :=
  coverage_fraction * A = carpet_area

-- State the Theorem
theorem living_room_area_is_60 (A : ℝ) (h : target_living_room_area A) : A = 60 := by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_living_room_area_is_60_l1_113


namespace NUMINAMATH_GPT_landscape_breadth_l1_146

theorem landscape_breadth (L B : ℕ)
  (h1 : B = 6 * L)
  (h2 : 4200 = (1 / 7 : ℚ) * 6 * L^2) :
  B = 420 := 
  sorry

end NUMINAMATH_GPT_landscape_breadth_l1_146


namespace NUMINAMATH_GPT_new_average_l1_136

variable (avg9 : ℝ) (score10 : ℝ) (n : ℕ)
variable (h : avg9 = 80) (h10 : score10 = 100) (n9 : n = 9)

theorem new_average (h : avg9 = 80) (h10 : score10 = 100) (n9 : n = 9) :
  ((n * avg9 + score10) / (n + 1)) = 82 :=
by
  rw [h, h10, n9]
  sorry

end NUMINAMATH_GPT_new_average_l1_136


namespace NUMINAMATH_GPT_exists_midpoint_with_integer_coordinates_l1_154

theorem exists_midpoint_with_integer_coordinates (points : Fin 5 → ℤ × ℤ) :
  ∃ (i j : Fin 5), i ≠ j ∧ ((points i).1 + (points j).1) % 2 = 0 ∧ ((points i).2 + (points j).2) % 2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_exists_midpoint_with_integer_coordinates_l1_154


namespace NUMINAMATH_GPT_gcd_solutions_l1_165

theorem gcd_solutions (x m n p: ℤ) (h_eq: x * (4 * x - 5) = 7) (h_gcd: Int.gcd m (Int.gcd n p) = 1)
  (h_form: ∃ x1 x2: ℤ, x1 = (m + Int.sqrt n) / p ∧ x2 = (m - Int.sqrt n) / p) : m + n + p = 150 :=
by
  have disc_eq : 25 + 112 = 137 :=
    by norm_num
  sorry

end NUMINAMATH_GPT_gcd_solutions_l1_165


namespace NUMINAMATH_GPT_find_base_of_exponent_l1_192

theorem find_base_of_exponent
  (x : ℝ)
  (h1 : 4 ^ (2 * x + 2) = (some_number : ℝ) ^ (3 * x - 1))
  (x_eq : x = 1) :
  some_number = 16 := 
by
  -- proof steps would go here
  sorry

end NUMINAMATH_GPT_find_base_of_exponent_l1_192


namespace NUMINAMATH_GPT_polynomial_divisibility_by_5_l1_182

theorem polynomial_divisibility_by_5
  (a b c d : ℤ)
  (divisible : ∀ x : ℤ, 5 ∣ (a * x ^ 3 + b * x ^ 2 + c * x + d)) :
  5 ∣ a ∧ 5 ∣ b ∧ 5 ∣ c ∧ 5 ∣ d :=
sorry

end NUMINAMATH_GPT_polynomial_divisibility_by_5_l1_182


namespace NUMINAMATH_GPT_minimum_possible_n_l1_190

theorem minimum_possible_n (n p : ℕ) (h1: p > 0) (h2: 15 * n - 45 = 105) : n = 10 :=
sorry

end NUMINAMATH_GPT_minimum_possible_n_l1_190


namespace NUMINAMATH_GPT_caloprian_lifespan_proof_l1_112

open Real

noncomputable def timeDilation (delta_t : ℝ) (v : ℝ) (c : ℝ) : ℝ :=
  delta_t * sqrt (1 - (v ^ 2) / (c ^ 2))

noncomputable def caloprianMinLifeSpan (d : ℝ) (v : ℝ) (c : ℝ) : ℝ :=
  let earth_time := (d / v) * 2
  timeDilation earth_time v c

theorem caloprian_lifespan_proof :
  caloprianMinLifeSpan 30 0.3 1 = 20 * sqrt 91 :=
sorry

end NUMINAMATH_GPT_caloprian_lifespan_proof_l1_112


namespace NUMINAMATH_GPT_find_list_price_l1_169

noncomputable def list_price (x : ℝ) (alice_price_diff bob_price_diff : ℝ) (alice_comm_fraction bob_comm_fraction : ℝ) : Prop :=
  alice_comm_fraction * (x - alice_price_diff) = bob_comm_fraction * (x - bob_price_diff)

theorem find_list_price : list_price 40 15 25 0.15 0.25 :=
by
  sorry

end NUMINAMATH_GPT_find_list_price_l1_169


namespace NUMINAMATH_GPT_sum_of_digits_base8_product_l1_159

theorem sum_of_digits_base8_product
  (a b : ℕ)
  (a_base8 : a = 3 * 8^1 + 4 * 8^0)
  (b_base8 : b = 2 * 8^1 + 2 * 8^0)
  (product : ℕ := a * b)
  (product_base8 : ℕ := (product / 64) * 8^2 + ((product / 8) % 8) * 8^1 + (product % 8)) :
  ((product_base8 / 8^2) + ((product_base8 / 8) % 8) + (product_base8 % 8)) = 1 * 8^1 + 6 * 8^0 :=
sorry

end NUMINAMATH_GPT_sum_of_digits_base8_product_l1_159


namespace NUMINAMATH_GPT_words_per_page_l1_188

theorem words_per_page (p : ℕ) (h1 : 150 * p ≡ 210 [MOD 221]) (h2 : p ≤ 120) : p = 195 := by
  sorry

end NUMINAMATH_GPT_words_per_page_l1_188


namespace NUMINAMATH_GPT_total_cost_l1_141

-- Given conditions
def pen_cost : ℕ := 4
def briefcase_cost : ℕ := 5 * pen_cost

-- Theorem stating the total cost Marcel paid for both items
theorem total_cost (pen_cost briefcase_cost : ℕ) (h_pen: pen_cost = 4) (h_briefcase: briefcase_cost = 5 * pen_cost) :
  pen_cost + briefcase_cost = 24 := by
  sorry

end NUMINAMATH_GPT_total_cost_l1_141


namespace NUMINAMATH_GPT_average_of_rstu_l1_178

theorem average_of_rstu (r s t u : ℝ) (h : (5 / 4) * (r + s + t + u) = 15) : (r + s + t + u) / 4 = 3 :=
by
  sorry

end NUMINAMATH_GPT_average_of_rstu_l1_178


namespace NUMINAMATH_GPT_student_correct_answers_l1_195

theorem student_correct_answers (C I : ℕ) (h1 : C + I = 100) (h2 : C - 2 * I = 73) : C = 91 :=
sorry

end NUMINAMATH_GPT_student_correct_answers_l1_195


namespace NUMINAMATH_GPT_repetend_of_frac_4_div_17_is_235294_l1_179

noncomputable def decimalRepetend_of_4_div_17 : String :=
  let frac := 4 / 17
  let repetend := "235294"
  repetend

theorem repetend_of_frac_4_div_17_is_235294 :
  (∃ n m : ℕ, (4 / 17 : ℚ) = n + (m / 10^6) ∧ m % 10^6 = 235294) :=
sorry

end NUMINAMATH_GPT_repetend_of_frac_4_div_17_is_235294_l1_179


namespace NUMINAMATH_GPT_positive_difference_of_squares_l1_104

theorem positive_difference_of_squares (a b : ℕ) (h1 : a + b = 70) (h2 : a - b = 20) : a^2 - b^2 = 1400 :=
by
sorry

end NUMINAMATH_GPT_positive_difference_of_squares_l1_104


namespace NUMINAMATH_GPT_points_for_victory_l1_139

theorem points_for_victory (V : ℕ) :
  (∃ (played total_games : ℕ) (points_after_games : ℕ) (remaining_games : ℕ) (needed_points : ℕ) 
     (draw_points defeat_points : ℕ) (minimum_wins : ℕ), 
     played = 5 ∧
     total_games = 20 ∧ 
     points_after_games = 12 ∧
     remaining_games = total_games - played ∧
     needed_points = 40 - points_after_games ∧
     draw_points = 1 ∧
     defeat_points = 0 ∧
     minimum_wins = 7 ∧
     7 * V ≥ needed_points ∧
     remaining_games = total_games - played ∧
     needed_points = 28) → V = 4 :=
sorry

end NUMINAMATH_GPT_points_for_victory_l1_139


namespace NUMINAMATH_GPT_rectangle_perimeter_l1_172

theorem rectangle_perimeter : 
  ∃ (x y a b : ℝ), 
  (x * y = 2016) ∧ 
  (a * b = 2016) ∧ 
  (x^2 + y^2 = 4 * (a^2 - b^2)) → 
  2 * (x + y) = 8 * Real.sqrt 1008 :=
sorry

end NUMINAMATH_GPT_rectangle_perimeter_l1_172


namespace NUMINAMATH_GPT_age_of_student_who_left_l1_197

variables
  (avg_age_students : ℝ)
  (num_students_before : ℕ)
  (num_students_after : ℕ)
  (age_teacher : ℝ)
  (new_avg_age_class : ℝ)

theorem age_of_student_who_left
  (h1 : avg_age_students = 14)
  (h2 : num_students_before = 45)
  (h3 : num_students_after = 44)
  (h4 : age_teacher = 45)
  (h5 : new_avg_age_class = 14.66)
: ∃ (age_student_left : ℝ), abs (age_student_left - 15.3) < 0.1 :=
sorry

end NUMINAMATH_GPT_age_of_student_who_left_l1_197


namespace NUMINAMATH_GPT_positive_integer_divisible_by_18_with_sqrt_between_24_and_24_5_l1_171

theorem positive_integer_divisible_by_18_with_sqrt_between_24_and_24_5 : 
  ∃ (x : ℕ), (x = 594) ∧ (18 ∣ x) ∧ (24 ≤ Real.sqrt (x) ∧ Real.sqrt (x) ≤ 24.5) := 
by 
  sorry

end NUMINAMATH_GPT_positive_integer_divisible_by_18_with_sqrt_between_24_and_24_5_l1_171


namespace NUMINAMATH_GPT_expression_value_l1_191

def α : ℝ := 60
def β : ℝ := 20
def AB : ℝ := 1

noncomputable def γ : ℝ := 180 - (α + β)

noncomputable def AC : ℝ := AB * (Real.sin γ / Real.sin β)
noncomputable def BC : ℝ := (Real.sin α / Real.sin γ) * AB

theorem expression_value : (1 / AC - BC) = 2 := by
  sorry

end NUMINAMATH_GPT_expression_value_l1_191


namespace NUMINAMATH_GPT_brad_running_speed_l1_105

variable (dist_between_homes : ℕ)
variable (maxwell_speed : ℕ)
variable (time_maxwell_walks : ℕ)
variable (maxwell_start_time : ℕ)
variable (brad_start_time : ℕ)

#check dist_between_homes = 94
#check maxwell_speed = 4
#check time_maxwell_walks = 10
#check brad_start_time = maxwell_start_time + 1

theorem brad_running_speed (dist_between_homes : ℕ) (maxwell_speed : ℕ) (time_maxwell_walks : ℕ) (maxwell_start_time : ℕ) (brad_start_time : ℕ) :
  dist_between_homes = 94 →
  maxwell_speed = 4 →
  time_maxwell_walks = 10 →
  brad_start_time = maxwell_start_time + 1 →
  (dist_between_homes - maxwell_speed * time_maxwell_walks) / (time_maxwell_walks - (brad_start_time - maxwell_start_time)) = 6 :=
by
  intros
  sorry

end NUMINAMATH_GPT_brad_running_speed_l1_105


namespace NUMINAMATH_GPT_line_inclination_angle_l1_194

theorem line_inclination_angle (θ : ℝ) : 
  (∃ θ : ℝ, ∀ x y : ℝ, x + y + 1 = 0 → θ = 3 * π / 4) := sorry

end NUMINAMATH_GPT_line_inclination_angle_l1_194


namespace NUMINAMATH_GPT_whole_number_M_l1_118

theorem whole_number_M (M : ℤ) (hM : 9 < (M : ℝ) / 4 ∧ (M : ℝ) / 4 < 10) : M = 37 ∨ M = 38 ∨ M = 39 := by
  sorry

end NUMINAMATH_GPT_whole_number_M_l1_118


namespace NUMINAMATH_GPT_expand_product_l1_156

-- Define the problem
theorem expand_product (x : ℝ) : (x - 3) * (x + 4) = x^2 + x - 12 :=
by
  sorry

end NUMINAMATH_GPT_expand_product_l1_156


namespace NUMINAMATH_GPT_total_surface_area_of_new_solid_l1_162

-- Define the heights of the pieces using the given conditions
def height_A := 1 / 4
def height_B := 1 / 5
def height_C := 1 / 6
def height_D := 1 / 7
def height_E := 1 / 8
def height_F := 1 - (height_A + height_B + height_C + height_D + height_E)

-- Assembling the pieces back in reverse order (F to A), encapsulate the total surface area calculation
theorem total_surface_area_of_new_solid : 
  (2 * (1 : ℝ)) + (2 * (1 * 1 : ℝ)) + (2 * (1 * 1 : ℝ)) = 6 :=
by
  sorry

end NUMINAMATH_GPT_total_surface_area_of_new_solid_l1_162


namespace NUMINAMATH_GPT_unique_integer_m_l1_175

theorem unique_integer_m :
  ∃! (m : ℤ), m - ⌊m / (2005 : ℝ)⌋ = 2005 :=
by
  --- Here belongs the proof part, but we leave it with a sorry
  sorry

end NUMINAMATH_GPT_unique_integer_m_l1_175


namespace NUMINAMATH_GPT_freshmen_minus_sophomores_eq_24_l1_121

def total_students := 800
def percent_juniors := 27 / 100
def percent_not_sophomores := 75 / 100
def number_seniors := 160

def number_juniors := percent_juniors * total_students
def number_not_sophomores := percent_not_sophomores * total_students
def number_sophomores := total_students - number_not_sophomores
def number_freshmen := total_students - (number_juniors + number_sophomores + number_seniors)

theorem freshmen_minus_sophomores_eq_24 :
  number_freshmen - number_sophomores = 24 :=
sorry

end NUMINAMATH_GPT_freshmen_minus_sophomores_eq_24_l1_121


namespace NUMINAMATH_GPT_perpendicular_lines_l1_177

theorem perpendicular_lines (a : ℝ) : 
  (∀ x y : ℝ, (x + a * y - a = 0) → (a * x - (2 * a - 3) * y - 1 = 0) → 
    (∀ x y : ℝ, ( -1 / a ) * ( -a / (2 * a - 3)) = 1 )) → a = 3 := 
by
  sorry

end NUMINAMATH_GPT_perpendicular_lines_l1_177


namespace NUMINAMATH_GPT_isosceles_trapezoid_AC_length_l1_153

noncomputable def length_of_AC (AB AD BC CD AC : ℝ) :=
  AB = 30 ∧ AD = 15 ∧ BC = 15 ∧ CD = 12 → AC = 23.32

theorem isosceles_trapezoid_AC_length :
  length_of_AC 30 15 15 12 23.32 := by
  sorry

end NUMINAMATH_GPT_isosceles_trapezoid_AC_length_l1_153


namespace NUMINAMATH_GPT_tonya_stamps_left_l1_109

theorem tonya_stamps_left 
    (stamps_per_matchbook : ℕ) 
    (matches_per_matchbook : ℕ) 
    (tonya_initial_stamps : ℕ) 
    (jimmy_initial_matchbooks : ℕ) 
    (stamps_per_match : ℕ) 
    (tonya_final_stamps_expected : ℕ)
    (h1 : stamps_per_matchbook = 1) 
    (h2 : matches_per_matchbook = 24) 
    (h3 : tonya_initial_stamps = 13) 
    (h4 : jimmy_initial_matchbooks = 5) 
    (h5 : stamps_per_match = 12)
    (h6 : tonya_final_stamps_expected = 3) :
    tonya_initial_stamps - jimmy_initial_matchbooks * (matches_per_matchbook / stamps_per_match) = tonya_final_stamps_expected :=
by
  sorry

end NUMINAMATH_GPT_tonya_stamps_left_l1_109


namespace NUMINAMATH_GPT_probability_of_green_tile_l1_111

theorem probability_of_green_tile :
  let total_tiles := 100
  let green_tiles := 14
  let probability := green_tiles / total_tiles
  probability = 7 / 50 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_green_tile_l1_111


namespace NUMINAMATH_GPT_subtract_decimal_l1_186

theorem subtract_decimal : 3.75 - 1.46 = 2.29 :=
by
  sorry

end NUMINAMATH_GPT_subtract_decimal_l1_186


namespace NUMINAMATH_GPT_length_of_plot_l1_147

theorem length_of_plot (breadth : ℝ) (cost_per_meter : ℝ) (total_cost : ℝ) 
  (h1 : cost_per_meter = 26.50) 
  (h2 : total_cost = 5300)
  (h3 : breadth + 20 = 60) :
  2 * ((breadth + 20) + breadth) = total_cost / cost_per_meter := 
by
  sorry

end NUMINAMATH_GPT_length_of_plot_l1_147


namespace NUMINAMATH_GPT_prime_factorization_min_x_l1_129

-- Define the conditions
variable (x y : ℕ) (a b e f : ℕ)

-- Given conditions: x and y are positive integers, and 5x^7 = 13y^11
axiom condition1 : 0 < x ∧ 0 < y
axiom condition2 : 5 * x^7 = 13 * y^11

-- Prove the mathematical equivalence
theorem prime_factorization_min_x (a b e f : ℕ) 
    (hx : 5 * x^7 = 13 * y^11)
    (h_prime : a = 13 ∧ b = 5 ∧ e = 6 ∧ f = 1) :
    a + b + e + f = 25 :=
sorry

end NUMINAMATH_GPT_prime_factorization_min_x_l1_129


namespace NUMINAMATH_GPT_numbers_less_than_reciprocal_l1_193

theorem numbers_less_than_reciprocal :
  (1 / 3 < 3) ∧ (1 / 2 < 2) ∧ ¬(1 < 1) ∧ ¬(2 < 1 / 2) ∧ ¬(3 < 1 / 3) :=
by
  sorry

end NUMINAMATH_GPT_numbers_less_than_reciprocal_l1_193


namespace NUMINAMATH_GPT_right_triangle_construction_condition_l1_166

theorem right_triangle_construction_condition (A B C : Point) (b d : ℝ) :
  AC = b → AC + BC - AB = d → b > d :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_right_triangle_construction_condition_l1_166


namespace NUMINAMATH_GPT_new_perimeter_after_adding_tiles_l1_131

-- Define the original condition as per the problem statement
def original_T_shape (n : ℕ) : Prop :=
  n = 6

def original_perimeter (p : ℕ) : Prop :=
  p = 12

-- Define hypothesis required to add three more tiles while sharing a side with existing tiles
def add_three_tiles_with_shared_side (original_tiles : ℕ) (new_tiles_added : ℕ) : Prop :=
  original_tiles + new_tiles_added = 9

-- Prove the new perimeter after adding three tiles to the original T-shaped figure
theorem new_perimeter_after_adding_tiles
  (n : ℕ) (p : ℕ) (new_tiles : ℕ) (new_p : ℕ)
  (h1 : original_T_shape n)
  (h2 : original_perimeter p)
  (h3 : add_three_tiles_with_shared_side n new_tiles)
  : new_p = 16 :=
sorry

end NUMINAMATH_GPT_new_perimeter_after_adding_tiles_l1_131


namespace NUMINAMATH_GPT_discarded_marble_weight_l1_180

-- Define the initial weight of the marble block and the weights of the statues
def initial_weight : ℕ := 80
def weight_statue_1 : ℕ := 10
def weight_statue_2 : ℕ := 18
def weight_statue_3 : ℕ := 15
def weight_statue_4 : ℕ := 15

-- The proof statement: the discarded weight of marble is 22 pounds.
theorem discarded_marble_weight :
  initial_weight - (weight_statue_1 + weight_statue_2 + weight_statue_3 + weight_statue_4) = 22 :=
by
  sorry

end NUMINAMATH_GPT_discarded_marble_weight_l1_180


namespace NUMINAMATH_GPT_sqrt_99_eq_9801_expr_2000_1999_2001_eq_1_l1_108

theorem sqrt_99_eq_9801 : 99^2 = 9801 := by
  sorry

theorem expr_2000_1999_2001_eq_1 : 2000^2 - 1999 * 2001 = 1 := by
  sorry

end NUMINAMATH_GPT_sqrt_99_eq_9801_expr_2000_1999_2001_eq_1_l1_108


namespace NUMINAMATH_GPT_perimeter_of_square_l1_100

theorem perimeter_of_square (s : ℕ) (h : s = 13) : 4 * s = 52 :=
by {
  sorry
}

end NUMINAMATH_GPT_perimeter_of_square_l1_100


namespace NUMINAMATH_GPT_complement_union_l1_122

variable (U : Set ℤ)
variable (A : Set ℤ)
variable (B : Set ℤ)

theorem complement_union (hU : U = {-2, -1, 0, 1, 2, 3})
                         (hA : A = {-1, 0, 1})
                         (hB : B = {1, 2}) :
  U \ (A ∪ B) = {-2, 3} :=
sorry

end NUMINAMATH_GPT_complement_union_l1_122


namespace NUMINAMATH_GPT_range_of_a_l1_198

section
  variable {x a : ℝ}

  -- Define set A
  def setA : Set ℝ := { x | x^2 - 4*x + 3 < 0 }

  -- Define set B
  def setB (a : ℝ) : Set ℝ := 
    { x | (2*x + a ≤ 0) ∧ (x^2 - 2*(a + 7)*x + 5 ≤ 0)}

  -- The proof problem statement
  theorem range_of_a (a : ℝ) : 
    (setA ⊆ setB a) ↔ (-4 ≤ a ∧ a ≤ -2) :=
  sorry
end

end NUMINAMATH_GPT_range_of_a_l1_198


namespace NUMINAMATH_GPT_problem_I_problem_II_l1_119

noncomputable def f (x : ℝ) : ℝ := x - 2 * Real.sin x

theorem problem_I :
  ∀ x ∈ Set.Icc 0 Real.pi, (f x) ≥ (f (Real.pi / 3) - Real.sqrt 3) ∧ (f x) ≤ f Real.pi :=
sorry

theorem problem_II :
  ∀ a : ℝ, ((∃ x : ℝ, (0 < x ∧ x < Real.pi / 2) ∧ f x < a * x) ↔ a > -1) :=
sorry

end NUMINAMATH_GPT_problem_I_problem_II_l1_119


namespace NUMINAMATH_GPT_polygon_sides_exterior_angle_l1_101

theorem polygon_sides_exterior_angle (n : ℕ) (h : 360 / 24 = n) : n = 15 := by
  sorry

end NUMINAMATH_GPT_polygon_sides_exterior_angle_l1_101


namespace NUMINAMATH_GPT_alpha_beta_property_l1_185

theorem alpha_beta_property
  (α β : ℝ)
  (hαβ_roots : ∀ x : ℝ, (x = α ∨ x = β) → x^2 + x - 2023 = 0) :
  α^2 + 2 * α + β = 2022 :=
by
  sorry

end NUMINAMATH_GPT_alpha_beta_property_l1_185


namespace NUMINAMATH_GPT_minimum_value_of_2a5_a4_l1_140

variable {a : ℕ → ℝ} {q : ℝ}

-- Defining that the given sequence is geometric, i.e., a_{n+1} = a_n * q
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n, a (n + 1) = a n * q

-- The condition given in the problem is
def condition (a : ℕ → ℝ) : Prop :=
2 * a 4 + a 3 - 2 * a 2 - a 1 = 8

-- The sequence is positive
def positive_sequence (a : ℕ → ℝ) : Prop :=
∀ n, a n > 0

theorem minimum_value_of_2a5_a4 (h_geom : is_geometric_sequence a q) (h_cond : condition a) (h_pos : positive_sequence a) (h_q : q > 0) :
  2 * a 5 + a 4 = 12 * Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_minimum_value_of_2a5_a4_l1_140


namespace NUMINAMATH_GPT_candy_problem_l1_120

theorem candy_problem (a : ℕ) (h₁ : a % 10 = 6) (h₂ : a % 15 = 11) (h₃ : 200 ≤ a) (h₄ : a ≤ 250) :
  a = 206 ∨ a = 236 :=
sorry

end NUMINAMATH_GPT_candy_problem_l1_120


namespace NUMINAMATH_GPT_next_term_in_geometric_sequence_l1_148

theorem next_term_in_geometric_sequence (y : ℝ) : 
  let a := 3
  let r := 4*y 
  let t4 := 192*y^3 
  r * t4 = 768*y^4 :=
by
  sorry

end NUMINAMATH_GPT_next_term_in_geometric_sequence_l1_148


namespace NUMINAMATH_GPT_intersection_M_N_eq_segment_l1_168

def M : Set ℝ := {x | x^2 + x - 6 < 0}
def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

theorem intersection_M_N_eq_segment : M ∩ N = {x | 1 ≤ x ∧ x < 2} := by
  sorry

end NUMINAMATH_GPT_intersection_M_N_eq_segment_l1_168


namespace NUMINAMATH_GPT_solve_quadratic_eq1_solve_quadratic_eq2_solve_quadratic_eq3_solve_quadratic_eq4_l1_114

-- Equation (1)
theorem solve_quadratic_eq1 (x : ℝ) : x^2 + 16 = 8*x ↔ x = 4 := by
  sorry

-- Equation (2)
theorem solve_quadratic_eq2 (x : ℝ) : 2*x^2 + 4*x - 3 = 0 ↔ 
  x = -1 + (Real.sqrt 10) / 2 ∨ x = -1 - (Real.sqrt 10) / 2 := by
  sorry

-- Equation (3)
theorem solve_quadratic_eq3 (x : ℝ) : x*(x - 1) = x ↔ x = 0 ∨ x = 2 := by
  sorry

-- Equation (4)
theorem solve_quadratic_eq4 (x : ℝ) : x*(x + 4) = 8*x - 3 ↔ x = 3 ∨ x = 1 := by
  sorry

end NUMINAMATH_GPT_solve_quadratic_eq1_solve_quadratic_eq2_solve_quadratic_eq3_solve_quadratic_eq4_l1_114


namespace NUMINAMATH_GPT_unique_y_for_star_l1_127

def star (x y : ℝ) : ℝ := 5 * x - 2 * y + 2 * x * y - 3

theorem unique_y_for_star : (∀ y : ℝ, star 4 y = 17 → y = 0) ∧ (∃! y : ℝ, star 4 y = 17) := by
  sorry

end NUMINAMATH_GPT_unique_y_for_star_l1_127


namespace NUMINAMATH_GPT_sum_of_solutions_l1_116

theorem sum_of_solutions (x1 x2 : ℝ) (h1 : (x1 - 8) ^ 2 = 49) (h2 : (x2 - 8) ^ 2 = 49) : x1 + x2 = 16 :=
sorry

end NUMINAMATH_GPT_sum_of_solutions_l1_116


namespace NUMINAMATH_GPT_find_a10_l1_160

variable {G : Type*} [LinearOrderedField G]
variable (a : ℕ → G)

-- Conditions
def geometric_sequence (a : ℕ → G) (r : G) := ∀ n, a (n + 1) = r * a n
def positive_terms (a : ℕ → G) := ∀ n, 0 < a n
def specific_condition (a : ℕ → G) := a 3 * a 11 = 16

theorem find_a10
  (h_geom : geometric_sequence a 2)
  (h_pos : positive_terms a)
  (h_cond : specific_condition a) :
  a 10 = 32 := by
  sorry

end NUMINAMATH_GPT_find_a10_l1_160


namespace NUMINAMATH_GPT_not_perfect_square_l1_107

theorem not_perfect_square (n : ℕ) : ¬ ∃ k : ℕ, 7 * n + 3 = k^2 := 
by
  sorry

end NUMINAMATH_GPT_not_perfect_square_l1_107


namespace NUMINAMATH_GPT_gcd_187_119_base5_l1_132

theorem gcd_187_119_base5 :
  ∃ b : Nat, Nat.gcd 187 119 = 17 ∧ 17 = 3 * 5 + 2 ∧ 3 = 0 * 5 + 3 ∧ b = 3 * 10 + 2 := by
  sorry

end NUMINAMATH_GPT_gcd_187_119_base5_l1_132


namespace NUMINAMATH_GPT_original_strength_of_class_l1_144

-- Definitions from the problem conditions
def average_age_original (x : ℕ) : ℕ := 40 * x
def total_students (x : ℕ) : ℕ := x + 17
def total_age_new_students : ℕ := 17 * 32
def new_average_age : ℕ := 36

-- Lean statement to prove that the original strength of the class is 17.
theorem original_strength_of_class :
  ∃ x : ℕ, average_age_original x + total_age_new_students = total_students x * new_average_age ∧ x = 17 :=
by
  sorry

end NUMINAMATH_GPT_original_strength_of_class_l1_144


namespace NUMINAMATH_GPT_quadratic_transformation_l1_189

theorem quadratic_transformation :
  ∀ (x : ℝ), (x^2 + 6*x - 2 = 0) → ((x + 3)^2 = 11) :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_quadratic_transformation_l1_189
