import Mathlib

namespace NUMINAMATH_GPT_arithmetic_sequence_geometric_subsequence_l2102_210231

theorem arithmetic_sequence_geometric_subsequence :
  ∀ (a : ℕ → ℝ), (∀ n, a (n + 1) = a n + 2) ∧ (a 1 * a 3 = a 2 ^ 2) → a 2 = 4 :=
by
  intros a h
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_geometric_subsequence_l2102_210231


namespace NUMINAMATH_GPT_class_average_correct_l2102_210236

def class_average_test_A : ℝ :=
  0.30 * 97 + 0.25 * 85 + 0.20 * 78 + 0.15 * 65 + 0.10 * 55

def class_average_test_B : ℝ :=
  0.30 * 93 + 0.25 * 80 + 0.20 * 75 + 0.15 * 70 + 0.10 * 60

theorem class_average_correct :
  round class_average_test_A = 81 ∧
  round class_average_test_B = 79 := 
by 
  sorry

end NUMINAMATH_GPT_class_average_correct_l2102_210236


namespace NUMINAMATH_GPT_number_is_209_given_base_value_is_100_l2102_210269

theorem number_is_209_given_base_value_is_100 (n : ℝ) (base_value : ℝ) (H : base_value = 100) (percentage : ℝ) (H1 : percentage = 2.09) : n = 209 :=
by
  sorry

end NUMINAMATH_GPT_number_is_209_given_base_value_is_100_l2102_210269


namespace NUMINAMATH_GPT_absolute_value_simplification_l2102_210299

theorem absolute_value_simplification (a b : ℝ) (ha : a < 0) (hb : b > 0) : |a - b| + |b - a| = -2 * a + 2 * b := 
by 
  sorry

end NUMINAMATH_GPT_absolute_value_simplification_l2102_210299


namespace NUMINAMATH_GPT_minimum_value_expression_l2102_210290

theorem minimum_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  4 * x^4 + 16 * y^4 + 36 * z^4 + 9 / (x * y * z) ≥ 24 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_expression_l2102_210290


namespace NUMINAMATH_GPT_order_numbers_l2102_210209

theorem order_numbers (a b c : ℕ) (h1 : a = 8^10) (h2 : b = 4^15) (h3 : c = 2^31) : b = a ∧ a < c :=
by {
  sorry
}

end NUMINAMATH_GPT_order_numbers_l2102_210209


namespace NUMINAMATH_GPT_find_a_l2102_210230

theorem find_a (k a : ℚ) (hk : 4 * k = 60) (ha : 15 * a - 5 = 60) : a = 13 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l2102_210230


namespace NUMINAMATH_GPT_evaluate_x_squared_plus_y_squared_l2102_210215

theorem evaluate_x_squared_plus_y_squared (x y : ℚ) (h1 : x + 2 * y = 20) (h2 : 3 * x + y = 19) : x^2 + y^2 = 401 / 5 :=
sorry

end NUMINAMATH_GPT_evaluate_x_squared_plus_y_squared_l2102_210215


namespace NUMINAMATH_GPT_problem_equiv_math_problem_l2102_210239
-- Lean Statement for the proof problem

variable {x y z : ℝ}

theorem problem_equiv_math_problem (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) 
  (eq1 : x^2 + x * y + y^2 / 3 = 25) 
  (eq2 : y^2 / 3 + z^2 = 9) 
  (eq3 : z^2 + z * x + x^2 = 16) :
  x * y + 2 * y * z + 3 * z * x = 24 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_problem_equiv_math_problem_l2102_210239


namespace NUMINAMATH_GPT_sprinted_further_than_jogged_l2102_210234

def sprint_distance1 := 0.8932
def sprint_distance2 := 0.7773
def sprint_distance3 := 0.9539
def sprint_distance4 := 0.5417
def sprint_distance5 := 0.6843

def jog_distance1 := 0.7683
def jog_distance2 := 0.4231
def jog_distance3 := 0.5733
def jog_distance4 := 0.625
def jog_distance5 := 0.6549

def total_sprint_distance := sprint_distance1 + sprint_distance2 + sprint_distance3 + sprint_distance4 + sprint_distance5
def total_jog_distance := jog_distance1 + jog_distance2 + jog_distance3 + jog_distance4 + jog_distance5

theorem sprinted_further_than_jogged :
  total_sprint_distance - total_jog_distance = 0.8058 :=
by
  sorry

end NUMINAMATH_GPT_sprinted_further_than_jogged_l2102_210234


namespace NUMINAMATH_GPT_valid_t_range_for_f_l2102_210271

theorem valid_t_range_for_f :
  (∀ x : ℝ, |x + 1| + |x - t| ≥ 2015) ↔ t ∈ (Set.Iic (-2016) ∪ Set.Ici 2014) := 
sorry

end NUMINAMATH_GPT_valid_t_range_for_f_l2102_210271


namespace NUMINAMATH_GPT_original_students_l2102_210237

theorem original_students (a b : ℕ) : 
  a + b = 92 ∧ a - 5 = 3 * (b + 5 - 32) → a = 45 ∧ b = 47 :=
by sorry

end NUMINAMATH_GPT_original_students_l2102_210237


namespace NUMINAMATH_GPT_minimum_value_expression_l2102_210241

noncomputable def minimum_expression (a b c : ℝ) : ℝ :=
  a^2 + 6 * a * b + 9 * b^2 + 3 * c^2

theorem minimum_value_expression (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b * c = 27) :
  minimum_expression a b c ≥ 126 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_expression_l2102_210241


namespace NUMINAMATH_GPT_rahim_sequence_final_value_l2102_210213

theorem rahim_sequence_final_value :
  ∃ (a : ℕ) (b : ℕ), a ^ b = 5 ^ 16 :=
sorry

end NUMINAMATH_GPT_rahim_sequence_final_value_l2102_210213


namespace NUMINAMATH_GPT_pet_purchase_ways_l2102_210207

-- Define the conditions
def number_of_puppies : Nat := 20
def number_of_kittens : Nat := 6
def number_of_hamsters : Nat := 8

def alice_choices : Nat := number_of_puppies

-- Define the problem statement in Lean
theorem pet_purchase_ways : 
  (number_of_puppies = 20) ∧ 
  (number_of_kittens = 6) ∧ 
  (number_of_hamsters = 8) → 
  (alice_choices * 2 * number_of_kittens * number_of_hamsters) = 1920 := 
by
  intros h
  sorry

end NUMINAMATH_GPT_pet_purchase_ways_l2102_210207


namespace NUMINAMATH_GPT_Juan_birth_year_proof_l2102_210216

-- Let BTC_year(n) be the year of the nth BTC competition.
def BTC_year (n : ℕ) : ℕ :=
  1990 + (n - 1) * 2

-- Juan's birth year given his age and the BTC he participated in.
def Juan_birth_year (current_year : ℕ) (age : ℕ) : ℕ :=
  current_year - age

-- Main proof problem statement.
theorem Juan_birth_year_proof :
  (BTC_year 5 = 1998) →
  (Juan_birth_year 1998 14 = 1984) :=
by
  intros
  sorry

end NUMINAMATH_GPT_Juan_birth_year_proof_l2102_210216


namespace NUMINAMATH_GPT_find_k_l2102_210214

def system_of_equations (x y k : ℝ) : Prop :=
  x - y = k - 3 ∧
  3 * x + 5 * y = 2 * k + 8 ∧
  x + y = 2

theorem find_k (x y k : ℝ) (h : system_of_equations x y k) : k = 1 := 
sorry

end NUMINAMATH_GPT_find_k_l2102_210214


namespace NUMINAMATH_GPT_hyperbola_focal_length_l2102_210246

-- Define the constants a^2 and b^2 based on the given hyperbola equation.
def a_squared : ℝ := 16
def b_squared : ℝ := 25

-- Define the constants a and b as the square roots of a^2 and b^2.
noncomputable def a : ℝ := Real.sqrt a_squared
noncomputable def b : ℝ := Real.sqrt b_squared

-- Define the constant c based on the relation c^2 = a^2 + b^2.
noncomputable def c : ℝ := Real.sqrt (a_squared + b_squared)

-- The focal length of the hyperbola is 2c.
noncomputable def focal_length : ℝ := 2 * c

-- The theorem that captures the statement of the problem.
theorem hyperbola_focal_length : focal_length = 2 * Real.sqrt 41 := by
  -- Proof omitted.
  sorry

end NUMINAMATH_GPT_hyperbola_focal_length_l2102_210246


namespace NUMINAMATH_GPT_sum_of_consecutive_integers_420_l2102_210291

theorem sum_of_consecutive_integers_420 : 
  ∃ (k n : ℕ) (h1 : k ≥ 2) (h2 : k * n + k * (k - 1) / 2 = 420), 
  ∃ K : Finset ℕ, K.card = 6 ∧ (∀ x ∈ K, k = x) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_consecutive_integers_420_l2102_210291


namespace NUMINAMATH_GPT_work_completion_time_l2102_210275

theorem work_completion_time (x : ℝ) (a_work_rate b_work_rate combined_work_rate : ℝ) :
  a_work_rate = 1 / 15 ∧
  b_work_rate = 1 / 20 ∧
  combined_work_rate = 1 / 7.2 ∧
  a_work_rate + b_work_rate + (1 / x) = combined_work_rate → 
  x = 45 := by
  sorry

end NUMINAMATH_GPT_work_completion_time_l2102_210275


namespace NUMINAMATH_GPT_Carter_gave_Marcus_58_cards_l2102_210298

-- Define the conditions as variables
def original_cards : ℕ := 210
def current_cards : ℕ := 268

-- Define the question as a function
def cards_given_by_carter (original current : ℕ) : ℕ := current - original

-- Statement that we need to prove
theorem Carter_gave_Marcus_58_cards : cards_given_by_carter original_cards current_cards = 58 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_Carter_gave_Marcus_58_cards_l2102_210298


namespace NUMINAMATH_GPT_Carson_skipped_times_l2102_210217

variable (length width total_circles actual_distance perimeter distance_skipped : ℕ)
variable (total_distance : ℕ)

def perimeter_calculation (length width : ℕ) : ℕ := 2 * (length + width)

def total_distance_calculation (total_circles perimeter : ℕ) : ℕ := total_circles * perimeter

def distance_skipped_calculation (total_distance actual_distance : ℕ) : ℕ := total_distance - actual_distance

def times_skipped_calculation (distance_skipped perimeter : ℕ) : ℕ := distance_skipped / perimeter

theorem Carson_skipped_times (h_length : length = 600) 
                             (h_width : width = 400) 
                             (h_total_circles : total_circles = 10) 
                             (h_actual_distance : actual_distance = 16000) 
                             (h_perimeter : perimeter = perimeter_calculation length width) 
                             (h_total_distance : total_distance = total_distance_calculation total_circles perimeter) 
                             (h_distance_skipped : distance_skipped = distance_skipped_calculation total_distance actual_distance) :
                             times_skipped_calculation distance_skipped perimeter = 2 := 
by
  simp [perimeter_calculation, total_distance_calculation, distance_skipped_calculation, times_skipped_calculation]
  sorry

end NUMINAMATH_GPT_Carson_skipped_times_l2102_210217


namespace NUMINAMATH_GPT_johns_total_spending_l2102_210284

theorem johns_total_spending
    (online_phone_price : ℝ := 2000)
    (phone_price_increase : ℝ := 0.02)
    (phone_case_price : ℝ := 35)
    (screen_protector_price : ℝ := 15)
    (accessories_discount : ℝ := 0.05)
    (sales_tax : ℝ := 0.06) :
    let store_phone_price := online_phone_price * (1 + phone_price_increase)
    let regular_accessories_price := phone_case_price + screen_protector_price
    let discounted_accessories_price := regular_accessories_price * (1 - accessories_discount)
    let pre_tax_total := store_phone_price + discounted_accessories_price
    let total_spending := pre_tax_total * (1 + sales_tax)
    total_spending = 2212.75 :=
by
    sorry

end NUMINAMATH_GPT_johns_total_spending_l2102_210284


namespace NUMINAMATH_GPT_rented_movie_cost_l2102_210204

def cost_of_tickets (c_ticket : ℝ) (n_tickets : ℕ) := c_ticket * n_tickets
def total_cost (cost_tickets cost_bought : ℝ) := cost_tickets + cost_bought
def remaining_cost (total_spent cost_so_far : ℝ) := total_spent - cost_so_far

theorem rented_movie_cost
  (c_ticket : ℝ)
  (n_tickets : ℕ)
  (c_bought : ℝ)
  (c_total : ℝ)
  (h1 : c_ticket = 10.62)
  (h2 : n_tickets = 2)
  (h3 : c_bought = 13.95)
  (h4 : c_total = 36.78) :
  remaining_cost c_total (total_cost (cost_of_tickets c_ticket n_tickets) c_bought) = 1.59 :=
by 
  sorry

end NUMINAMATH_GPT_rented_movie_cost_l2102_210204


namespace NUMINAMATH_GPT_AM_GM_HM_inequality_l2102_210273

theorem AM_GM_HM_inequality (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a ≠ b) : 
  (a + b) / 2 > Real.sqrt (a * b) ∧ Real.sqrt (a * b) > (2 * a * b) / (a + b) := 
sorry

end NUMINAMATH_GPT_AM_GM_HM_inequality_l2102_210273


namespace NUMINAMATH_GPT_not_all_less_than_two_l2102_210228

theorem not_all_less_than_two {a b c : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ¬(a + 1/b < 2 ∧ b + 1/c < 2 ∧ c + 1/a < 2) :=
sorry

end NUMINAMATH_GPT_not_all_less_than_two_l2102_210228


namespace NUMINAMATH_GPT_problem_statement_l2102_210297

open Real

noncomputable def f (ω varphi : ℝ) (x : ℝ) := 2 * sin (ω * x + varphi)

theorem problem_statement (ω varphi : ℝ) (x1 x2 : ℝ) (hω_pos : ω > 0) (hvarphi_abs : abs varphi < π / 2)
    (hf0 : f ω varphi 0 = -1) (hmonotonic : ∀ x y, π / 18 < x ∧ x < y ∧ y < π / 3 → f ω varphi x < f ω varphi y)
    (hshift : ∀ x, f ω varphi (x + π) = f ω varphi x)
    (hx1x2_interval : -17 * π / 12 < x1 ∧ x1 < -2 * π / 3 ∧ -17 * π / 12 < x2 ∧ x2 < -2 * π / 3 ∧ x1 ≠ x2)
    (heq_fx : f ω varphi x1 = f ω varphi x2) :
    f ω varphi (x1 + x2) = -1 :=
sorry

end NUMINAMATH_GPT_problem_statement_l2102_210297


namespace NUMINAMATH_GPT_minimum_value_of_expression_l2102_210296

noncomputable def expr (a b c : ℝ) : ℝ := 8 * a^3 + 27 * b^3 + 64 * c^3 + 27 / (8 * a * b * c)

theorem minimum_value_of_expression (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  expr a b c ≥ 18 * Real.sqrt 3 := 
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_expression_l2102_210296


namespace NUMINAMATH_GPT_cross_square_side_length_l2102_210260

theorem cross_square_side_length (A : ℝ) (s : ℝ) (h1 : A = 810) 
(h2 : (2 * (s / 2)^2 + 2 * (s / 4)^2) = A) : s = 36 := by
  sorry

end NUMINAMATH_GPT_cross_square_side_length_l2102_210260


namespace NUMINAMATH_GPT_average_age_increase_by_one_l2102_210282

-- Definitions based on the conditions.
def initial_average_age : ℕ := 14
def initial_students : ℕ := 10
def new_students_average_age : ℕ := 17
def new_students : ℕ := 5

-- Helper calculation for the total age of initial students.
def total_age_initial_students := initial_students * initial_average_age

-- Helper calculation for the total age of new students.
def total_age_new_students := new_students * new_students_average_age

-- Helper calculation for the total age of all students.
def total_age_all_students := total_age_initial_students + total_age_new_students

-- Helper calculation for the number of all students.
def total_students := initial_students + new_students

-- Calculate the new average age.
def new_average_age := total_age_all_students / total_students

-- The goal is to prove the increase in average age is 1 year.
theorem average_age_increase_by_one :
  new_average_age - initial_average_age = 1 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_average_age_increase_by_one_l2102_210282


namespace NUMINAMATH_GPT_transformed_solution_equiv_l2102_210278

noncomputable def quadratic_solution_set (f : ℝ → ℝ) : Set ℝ :=
  {x | f x > 0}

noncomputable def transformed_solution_set (f : ℝ → ℝ) : Set ℝ :=
  {x | f (10^x) > 0}

theorem transformed_solution_equiv (f : ℝ → ℝ) :
  quadratic_solution_set f = {x | x < -1 ∨ x > 1 / 2} →
  transformed_solution_set f = {x | x > -Real.log 2} :=
by sorry

end NUMINAMATH_GPT_transformed_solution_equiv_l2102_210278


namespace NUMINAMATH_GPT_solve_equation_l2102_210261

theorem solve_equation (x : ℝ) (h : -x^2 = (3 * x + 1) / (x + 3)) : x = -1 :=
sorry

end NUMINAMATH_GPT_solve_equation_l2102_210261


namespace NUMINAMATH_GPT_problem_solution_l2102_210220

theorem problem_solution (a b : ℝ) (ha : |a| = 5) (hb : b = -3) :
  a + b = 2 ∨ a + b = -8 :=
by sorry

end NUMINAMATH_GPT_problem_solution_l2102_210220


namespace NUMINAMATH_GPT_fraction_of_white_surface_area_l2102_210294

/-- A cube has edges of 4 inches and is constructed using 64 smaller cubes, each with edges of 1 inch.
Out of these smaller cubes, 56 are white and 8 are black. The 8 black cubes fully cover one face of the larger cube.
Prove that the fraction of the surface area of the larger cube that is white is 5/6. -/
theorem fraction_of_white_surface_area 
  (total_cubes : ℕ := 64)
  (white_cubes : ℕ := 56)
  (black_cubes : ℕ := 8)
  (total_surface_area : ℕ := 96)
  (black_face_area : ℕ := 16)
  (white_surface_area : ℕ := 80) :
  white_surface_area / total_surface_area = 5 / 6 :=
sorry

end NUMINAMATH_GPT_fraction_of_white_surface_area_l2102_210294


namespace NUMINAMATH_GPT_bottles_per_case_l2102_210276

theorem bottles_per_case (total_bottles_per_day : ℕ) (cases_required : ℕ) (bottles_per_case : ℕ)
  (h1 : total_bottles_per_day = 65000)
  (h2 : cases_required = 5000) :
  bottles_per_case = total_bottles_per_day / cases_required :=
by
  sorry

end NUMINAMATH_GPT_bottles_per_case_l2102_210276


namespace NUMINAMATH_GPT_num_integers_satisfy_l2102_210208

theorem num_integers_satisfy : 
  ∃ n : ℕ, (n = 7 ∧ ∀ k : ℤ, (k > -5 ∧ k < 3) → (k = -4 ∨ k = -3 ∨ k = -2 ∨ k = -1 ∨ k = 0 ∨ k = 1 ∨ k = 2)) := 
sorry

end NUMINAMATH_GPT_num_integers_satisfy_l2102_210208


namespace NUMINAMATH_GPT_abc_eq_1_l2102_210242

theorem abc_eq_1 (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
(h4 : a ≠ b) (h5 : b ≠ c) (h6 : c ≠ a)
(h7 : a + 1 / b^2 = b + 1 / c^2) (h8 : b + 1 / c^2 = c + 1 / a^2) :
  |a * b * c| = 1 :=
sorry

end NUMINAMATH_GPT_abc_eq_1_l2102_210242


namespace NUMINAMATH_GPT_highest_possible_relocation_preference_l2102_210274

theorem highest_possible_relocation_preference
  (total_employees : ℕ)
  (relocated_to_X_percent : ℝ)
  (relocated_to_Y_percent : ℝ)
  (prefer_X_percent : ℝ)
  (prefer_Y_percent : ℝ)
  (htotal : total_employees = 200)
  (hrelocated_to_X_percent : relocated_to_X_percent = 0.30)
  (hrelocated_to_Y_percent : relocated_to_Y_percent = 0.70)
  (hprefer_X_percent : prefer_X_percent = 0.60)
  (hprefer_Y_percent : prefer_Y_percent = 0.40) :
  ∃ (max_relocated_with_preference : ℕ), max_relocated_with_preference = 140 :=
by
  sorry

end NUMINAMATH_GPT_highest_possible_relocation_preference_l2102_210274


namespace NUMINAMATH_GPT_rectangular_prism_edge_properties_l2102_210244

-- Define a rectangular prism and the concept of parallel and perpendicular pairs of edges.
structure RectangularPrism :=
  (vertices : Fin 8 → Fin 3 → ℝ)
  -- Additional necessary conditions on the structure could be added here.

-- Define the number of parallel edges in a rectangular prism
def number_of_parallel_edge_pairs (rp : RectangularPrism) : ℕ :=
  -- Formula or logic to count parallel edge pairs.
  8 -- Placeholder for actual logic computation, based on problem conditions.

-- Define the number of perpendicular edges in a rectangular prism
def number_of_perpendicular_edge_pairs (rp : RectangularPrism) : ℕ :=
  -- Formula or logic to count perpendicular edge pairs.
  20 -- Placeholder for actual logic computation, based on problem conditions.

-- Theorem that asserts the requirement based on conditions
theorem rectangular_prism_edge_properties (rp : RectangularPrism) :
  number_of_parallel_edge_pairs rp = 8 ∧ number_of_perpendicular_edge_pairs rp = 20 :=
  by
    -- Placeholder proof that establishes the theorem
    sorry

end NUMINAMATH_GPT_rectangular_prism_edge_properties_l2102_210244


namespace NUMINAMATH_GPT_distance_between_bakery_and_butcher_shop_l2102_210249

variables (v1 v2 : ℝ) -- speeds of the butcher's and baker's son respectively
variables (x : ℝ) -- distance covered by the baker's son by the time they meet
variable (distance : ℝ) -- distance between the bakery and the butcher shop

-- Given conditions
def butcher_walks_500_more := x + 0.5
def butcher_time_left := 10 / 60
def baker_time_left := 22.5 / 60

-- Equivalent relationships
def v1_def := v1 = 6 * x
def v2_def := v2 = (8/3) * (x + 0.5)

-- Final proof problem
theorem distance_between_bakery_and_butcher_shop :
  (x + 0.5 + x) = 2.5 :=
sorry

end NUMINAMATH_GPT_distance_between_bakery_and_butcher_shop_l2102_210249


namespace NUMINAMATH_GPT_count_decorations_l2102_210205

/--
Define a function T(n) that determines the number of ways to decorate the window 
with n stripes according to the given conditions.
--/
def T : ℕ → ℕ
| 0       => 1 -- optional case for completeness
| 1       => 2
| 2       => 2
| (n + 1) => T n + T (n - 1)

theorem count_decorations : T 10 = 110 := by
  sorry

end NUMINAMATH_GPT_count_decorations_l2102_210205


namespace NUMINAMATH_GPT_percentage_in_quarters_l2102_210201

theorem percentage_in_quarters:
  let dimes : ℕ := 40
  let quarters : ℕ := 30
  let value_dimes : ℕ := dimes * 10
  let value_quarters : ℕ := quarters * 25
  let total_value : ℕ := value_dimes + value_quarters
  let percentage_quarters : ℚ := (value_quarters : ℚ) / total_value * 100
  percentage_quarters = 65.22 := sorry

end NUMINAMATH_GPT_percentage_in_quarters_l2102_210201


namespace NUMINAMATH_GPT_ordinate_of_point_A_l2102_210253

noncomputable def p : ℝ := 1 / 4
noncomputable def distance_to_focus (y₀ : ℝ) : ℝ := y₀ + p / 2

theorem ordinate_of_point_A :
  ∃ y₀ : ℝ, (distance_to_focus y₀ = 9 / 8) → y₀ = 1 :=
by
  -- Assume solution steps here
  sorry

end NUMINAMATH_GPT_ordinate_of_point_A_l2102_210253


namespace NUMINAMATH_GPT_general_formula_for_a_n_a_n_greater_than_b_n_find_n_in_geometric_sequence_l2102_210265

-- Defines the sequences and properties given in the problem
def sequences (a_n b_n S_n T_n : ℕ → ℕ) : Prop :=
  a_n 1 = 1 ∧ S_n 2 = 4 ∧ 
  (∀ n : ℕ, 3 * S_n (n + 1) = 2 * S_n n + S_n (n + 2) + a_n n)

-- (1) Prove the general formula for {a_n}
theorem general_formula_for_a_n
  (a_n b_n S_n T_n : ℕ → ℕ)
  (h : sequences a_n b_n S_n T_n) :
  ∀ n : ℕ, a_n n = 2 * n - 1 :=
sorry

-- (2) If {b_n} is an arithmetic sequence and ∀n ∈ ℕ, S_n > T_n, prove a_n > b_n
theorem a_n_greater_than_b_n
  (a_n b_n S_n T_n : ℕ → ℕ)
  (h : sequences a_n b_n S_n T_n)
  (arithmetic_b : ∃ d: ℕ, ∀ n: ℕ, b_n n = b_n 0 + n * d)
  (Sn_greater_Tn : ∀ (n : ℕ), S_n n > T_n n) :
  ∀ n : ℕ, a_n n > b_n n :=
sorry

-- (3) If {b_n} is a geometric sequence, find n such that (a_n + 2 * T_n) / (b_n + 2 * S_n) = a_k
theorem find_n_in_geometric_sequence
  (a_n b_n S_n T_n : ℕ → ℕ)
  (h : sequences a_n b_n S_n T_n)
  (geometric_b : ∃ r: ℕ, ∀ n: ℕ, b_n n = b_n 0 * r^n)
  (b1_eq_1 : b_n 1 = 1)
  (b2_eq_3 : b_n 2 = 3)
  (k : ℕ) :
  ∃ n : ℕ, (a_n n + 2 * T_n n) / (b_n n + 2 * S_n n) = a_n k := 
sorry

end NUMINAMATH_GPT_general_formula_for_a_n_a_n_greater_than_b_n_find_n_in_geometric_sequence_l2102_210265


namespace NUMINAMATH_GPT_f_derivative_at_1_intervals_of_monotonicity_l2102_210258

def f (x : ℝ) := x^3 - 3 * x^2 + 10
def f' (x : ℝ) := 3 * x^2 - 6 * x

theorem f_derivative_at_1 : f' 1 = -3 := by
  sorry

theorem intervals_of_monotonicity :
  (∀ x : ℝ, x < 0 → f' x > 0) ∧
  (∀ x : ℝ, 0 < x ∧ x < 2 → f' x < 0) ∧
  (∀ x : ℝ, x > 2 → f' x > 0) := by
  sorry

end NUMINAMATH_GPT_f_derivative_at_1_intervals_of_monotonicity_l2102_210258


namespace NUMINAMATH_GPT_max_rectangle_area_l2102_210292

theorem max_rectangle_area (x y : ℝ) (h : 2 * x + 2 * y = 60) : x * y ≤ 225 :=
sorry

end NUMINAMATH_GPT_max_rectangle_area_l2102_210292


namespace NUMINAMATH_GPT_fraction_of_problems_solved_by_Andrey_l2102_210240

theorem fraction_of_problems_solved_by_Andrey (N x : ℕ) 
  (h1 : 0 < N) 
  (h2 : x = N / 2)
  (Boris_solves : ∀ y : ℕ, y = N - x → y / 3 = (N - x) / 3)
  (remaining_problems : ∀ y : ℕ, y = (N - x) - (N - x) / 3 → y = 2 * (N - x) / 3) 
  (Viktor_solves : (2 * (N - x) / 3 = N / 3)) :
  x / N = 1 / 2 := 
by {
  sorry
}

end NUMINAMATH_GPT_fraction_of_problems_solved_by_Andrey_l2102_210240


namespace NUMINAMATH_GPT_problem1_problem2_l2102_210229

theorem problem1 (x : ℝ) : (x + 4) ^ 2 - 5 * (x + 4) = 0 → x = -4 ∨ x = 1 :=
by
  sorry

theorem problem2 (x : ℝ) : x ^ 2 - 2 * x - 15 = 0 → x = -3 ∨ x = 5 :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l2102_210229


namespace NUMINAMATH_GPT_percentage_decrease_in_selling_price_l2102_210206

theorem percentage_decrease_in_selling_price (S M : ℝ) 
  (purchase_price : S = 240 + M)
  (markup_percentage : M = 0.25 * S)
  (gross_profit : S - 16 = 304) : 
  (320 - 304) / 320 * 100 = 5 := 
by
  sorry

end NUMINAMATH_GPT_percentage_decrease_in_selling_price_l2102_210206


namespace NUMINAMATH_GPT_smallest_sum_of_squares_l2102_210263

theorem smallest_sum_of_squares (x y : ℤ) (h : x^2 - y^2 = 145) : 
  ∃ x y, x^2 - y^2 = 145 ∧ x^2 + y^2 = 433 :=
sorry

end NUMINAMATH_GPT_smallest_sum_of_squares_l2102_210263


namespace NUMINAMATH_GPT_maximum_and_minimum_values_l2102_210210

noncomputable def f (p q x : ℝ) : ℝ := x^3 - p * x^2 - q * x

theorem maximum_and_minimum_values
  (p q : ℝ)
  (h1 : f p q 1 = 0)
  (h2 : (deriv (f p q)) 1 = 0) :
  ∃ (max_val min_val : ℝ), max_val = 4 / 27 ∧ min_val = 0 := 
by {
  sorry
}

end NUMINAMATH_GPT_maximum_and_minimum_values_l2102_210210


namespace NUMINAMATH_GPT_scale_reading_l2102_210247

theorem scale_reading (x : ℝ) (h₁ : 3.25 < x) (h₂ : x < 3.5) : x = 3.3 :=
sorry

end NUMINAMATH_GPT_scale_reading_l2102_210247


namespace NUMINAMATH_GPT_unique_solution_exists_l2102_210224

theorem unique_solution_exists (k : ℚ) (h : k ≠ 0) : 
  (∀ x : ℚ, (x + 3) / (kx - 2) = x → x = -2) ↔ k = -3 / 4 := 
by
  sorry

end NUMINAMATH_GPT_unique_solution_exists_l2102_210224


namespace NUMINAMATH_GPT_max_product_l2102_210267

theorem max_product (x : ℤ) : x + (2000 - x) = 2000 → x * (2000 - x) ≤ 1000000 :=
by
  sorry

end NUMINAMATH_GPT_max_product_l2102_210267


namespace NUMINAMATH_GPT_min_value_expr_l2102_210223

theorem min_value_expr (x y z : ℝ) (h : x > 0 ∧ y > 0 ∧ z > 0) (h_xyz : x * y * z = 1) : 
  x^2 + 4 * x * y + 9 * y^2 + 8 * y * z + 3 * z^2 ≥ 9^(10/9) :=
sorry

end NUMINAMATH_GPT_min_value_expr_l2102_210223


namespace NUMINAMATH_GPT_range_of_a_l2102_210272

theorem range_of_a (a : ℝ) :
  (∃ (M : ℝ × ℝ), (M.1 - a)^2 + (M.2 - a + 2)^2 = 1 ∧
    (M.1)^2 + (M.2 - 2)^2 + (M.1)^2 + (M.2)^2 = 10) → 
  0 ≤ a ∧ a ≤ 3 := 
sorry

end NUMINAMATH_GPT_range_of_a_l2102_210272


namespace NUMINAMATH_GPT_original_price_of_saree_is_400_l2102_210243

-- Define the original price of the saree
variable (P : ℝ)

-- Define the sale price after successive discounts
def sale_price (P : ℝ) : ℝ := 0.80 * P * 0.95

-- We want to prove that the original price P is 400 given that the sale price is 304
theorem original_price_of_saree_is_400 (h : sale_price P = 304) : P = 400 :=
sorry

end NUMINAMATH_GPT_original_price_of_saree_is_400_l2102_210243


namespace NUMINAMATH_GPT_ratio_fenced_region_l2102_210259

theorem ratio_fenced_region (L W : ℝ) (k : ℝ) 
  (area_eq : L * W = 200)
  (fence_eq : 2 * W + L = 40)
  (mult_eq : L = k * W) :
  k = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_fenced_region_l2102_210259


namespace NUMINAMATH_GPT_bird_families_migration_l2102_210285

theorem bird_families_migration 
  (total_families : ℕ)
  (africa_families : ℕ)
  (asia_families : ℕ)
  (south_america_families : ℕ)
  (africa_days : ℕ)
  (asia_days : ℕ)
  (south_america_days : ℕ)
  (migrated_families : ℕ)
  (remaining_families : ℕ)
  (total_migration_time : ℕ)
  (H1 : total_families = 200)
  (H2 : africa_families = 60)
  (H3 : asia_families = 95)
  (H4 : south_america_families = 30)
  (H5 : africa_days = 7)
  (H6 : asia_days = 14)
  (H7 : south_america_days = 10)
  (H8 : migrated_families = africa_families + asia_families + south_america_families)
  (H9 : remaining_families = total_families - migrated_families)
  (H10 : total_migration_time = 
          africa_families * africa_days + 
          asia_families * asia_days + 
          south_america_families * south_america_days) :
  remaining_families = 15 ∧ total_migration_time = 2050 :=
by
  sorry

end NUMINAMATH_GPT_bird_families_migration_l2102_210285


namespace NUMINAMATH_GPT_side_length_uncovered_l2102_210257

theorem side_length_uncovered (L W : ℝ) (h₁ : L * W = 50) (h₂ : 2 * W + L = 25) : L = 20 :=
by {
  sorry
}

end NUMINAMATH_GPT_side_length_uncovered_l2102_210257


namespace NUMINAMATH_GPT_intersection_A_B_l2102_210293

namespace MathProof

open Set

def A := {y : ℝ | ∃ x : ℝ, y = x^2 - 2 * x}
def B := {y : ℝ | ∃ x : ℝ, y = -x^2 + 2 * x + 6}

theorem intersection_A_B : A ∩ B = Icc (-1 : ℝ) 7 :=
by
  sorry

end MathProof

end NUMINAMATH_GPT_intersection_A_B_l2102_210293


namespace NUMINAMATH_GPT_alice_bob_age_difference_18_l2102_210270

-- Define Alice's and Bob's ages with the given constraints
def is_odd (n : ℕ) : Prop := n % 2 = 1

def alice_age (a b : ℕ) : ℕ := 10 * a + b
def bob_age (a b : ℕ) : ℕ := 10 * b + a

theorem alice_bob_age_difference_18 (a b : ℕ) (ha : is_odd a) (hb : is_odd b)
  (h : alice_age a b + 7 = 3 * (bob_age a b + 7)) : alice_age a b - bob_age a b = 18 :=
sorry

end NUMINAMATH_GPT_alice_bob_age_difference_18_l2102_210270


namespace NUMINAMATH_GPT_arithmetic_sum_eight_terms_l2102_210212

theorem arithmetic_sum_eight_terms :
  ∀ (a d : ℤ) (n : ℕ), a = -3 → d = 6 → n = 8 → 
  (last_term = a + (n - 1) * d) →
  (last_term = 39) →
  (sum = (n * (a + last_term)) / 2) →
  sum = 144 :=
by
  intros a d n ha hd hn hlast_term hlast_term_value hsum
  sorry

end NUMINAMATH_GPT_arithmetic_sum_eight_terms_l2102_210212


namespace NUMINAMATH_GPT_point_transformations_l2102_210200

theorem point_transformations (a b : ℝ) (h : (a ≠ 2 ∨ b ≠ 3))
  (H1 : ∃ x y : ℝ, (x, y) = (2 - (b - 3), 3 + (a - 2)) ∧ (y, x) = (-4, 2)) :
  b - a = -6 :=
by
  sorry

end NUMINAMATH_GPT_point_transformations_l2102_210200


namespace NUMINAMATH_GPT_stacy_days_to_finish_l2102_210264

-- Definitions based on the conditions
def total_pages : ℕ := 81
def pages_per_day : ℕ := 27

-- The theorem statement
theorem stacy_days_to_finish : total_pages / pages_per_day = 3 := by
  -- the proof is omitted
  sorry

end NUMINAMATH_GPT_stacy_days_to_finish_l2102_210264


namespace NUMINAMATH_GPT_pizza_eating_group_l2102_210286

theorem pizza_eating_group (x y : ℕ) (h1 : 6 * x + 2 * y ≥ 49) (h2 : 7 * x + 3 * y ≤ 59) : x = 8 ∧ y = 2 := by
  sorry

end NUMINAMATH_GPT_pizza_eating_group_l2102_210286


namespace NUMINAMATH_GPT_age_of_second_replaced_man_l2102_210235

theorem age_of_second_replaced_man (avg_age_increase : ℕ) (avg_new_men_age : ℕ) (first_replaced_age : ℕ) (total_men : ℕ) (new_age_sum : ℕ) :
  avg_age_increase = 1 →
  avg_new_men_age = 34 →
  first_replaced_age = 21 →
  total_men = 12 →
  new_age_sum = 2 * avg_new_men_age →
  47 - (new_age_sum - (first_replaced_age + x)) = 12 →
  x = 35 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_age_of_second_replaced_man_l2102_210235


namespace NUMINAMATH_GPT_quiz_answer_key_count_l2102_210226

theorem quiz_answer_key_count :
  ∃ n : ℕ, n = 480 ∧
  (∃ tf_count : ℕ, tf_count = 30 ∧
   (∃ mc_count : ℕ, mc_count = 16 ∧ 
    n = tf_count * mc_count)) :=
    sorry

end NUMINAMATH_GPT_quiz_answer_key_count_l2102_210226


namespace NUMINAMATH_GPT_exist_indices_with_non_decreasing_subsequences_l2102_210252

theorem exist_indices_with_non_decreasing_subsequences
  (a b c : ℕ → ℕ) :
  (∀ n m : ℕ, n < m → ∃ p q : ℕ, q < p ∧ 
    a p ≥ a q ∧ 
    b p ≥ b q ∧ 
    c p ≥ c q) :=
  sorry

end NUMINAMATH_GPT_exist_indices_with_non_decreasing_subsequences_l2102_210252


namespace NUMINAMATH_GPT_watched_commercials_eq_100_l2102_210256

variable (x : ℕ) -- number of people who watched commercials
variable (s : ℕ := 27) -- number of subscribers
variable (rev_comm : ℝ := 0.50) -- revenue per commercial
variable (rev_sub : ℝ := 1.00) -- revenue per subscriber
variable (total_rev : ℝ := 77.00) -- total revenue

theorem watched_commercials_eq_100 (h : rev_comm * (x : ℝ) + rev_sub * (s : ℝ) = total_rev) : x = 100 := by
  sorry

end NUMINAMATH_GPT_watched_commercials_eq_100_l2102_210256


namespace NUMINAMATH_GPT_exactly_one_even_l2102_210262

theorem exactly_one_even (a b c : ℕ) : 
  (∀ x, ¬ (a = x ∧ b = x ∧ c = x) ∧ 
  (a % 2 = 0 ∨ b % 2 = 0 ∨ c % 2 = 0) ∧ 
  ¬ (a % 2 = 0 ∧ b % 2 = 0) ∧ 
  ¬ (a % 2 = 0 ∧ c % 2 = 0) ∧ 
  ¬ (b % 2 = 0 ∧ c % 2 = 0)) :=
by
  sorry

end NUMINAMATH_GPT_exactly_one_even_l2102_210262


namespace NUMINAMATH_GPT_line_through_point_parallel_l2102_210250

theorem line_through_point_parallel (x y : ℝ) : 
  (∃ c : ℝ, x - 2 * y + c = 0 ∧ ∃ p : ℝ × ℝ, p = (1, 0) ∧ x - 2 * p.2 + c = 0) → (x - 2 * y - 1 = 0) :=
by
  sorry

end NUMINAMATH_GPT_line_through_point_parallel_l2102_210250


namespace NUMINAMATH_GPT_perpendicular_condition_l2102_210233

theorem perpendicular_condition (a : ℝ) :
  (2 * a * x + (a - 1) * y + 2 = 0) ∧ ((a + 1) * x + 3 * a * y + 3 = 0) →
  (a = 1/5 ↔ ∃ x y: ℝ, ((- (2 * a / (a - 1))) * (-(a + 1) / (3 * a)) = -1)) :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_condition_l2102_210233


namespace NUMINAMATH_GPT_profit_is_eight_dollars_l2102_210288

-- Define the given quantities and costs
def total_bracelets : ℕ := 52
def bracelets_given_away : ℕ := 8
def cost_of_materials : ℝ := 3.00
def selling_price_per_bracelet : ℝ := 0.25

-- Define the number of bracelets sold
def bracelets_sold := total_bracelets - bracelets_given_away

-- Calculate the total money earned from selling the bracelets
def total_earnings := bracelets_sold * selling_price_per_bracelet

-- Calculate the profit made by Alice
def profit := total_earnings - cost_of_materials

-- Prove that the profit is $8.00
theorem profit_is_eight_dollars : profit = 8.00 := by
  sorry

end NUMINAMATH_GPT_profit_is_eight_dollars_l2102_210288


namespace NUMINAMATH_GPT_tencent_technological_innovation_basis_tencent_innovative_development_analysis_l2102_210268

-- Define the dialectical materialist basis conditions
variable (dialectical_negation essence_innovation development_perspective unity_of_opposites : Prop)

-- Define Tencent's emphasis on technological innovation
variable (tencent_innovation : Prop)

-- Define the relationship between Tencent's development and materialist view of development
variable (unity_of_things_developmental progressiveness_tortuosity quantitative_qualitative_changes : Prop)
variable (tencent_development : Prop)

-- Prove that Tencent's emphasis on technological innovation aligns with dialectical materialism
theorem tencent_technological_innovation_basis :
  dialectical_negation ∧ essence_innovation ∧ development_perspective ∧ unity_of_opposites → tencent_innovation :=
by sorry

-- Prove that Tencent's innovative development aligns with dialectical materialist view of development
theorem tencent_innovative_development_analysis :
  unity_of_things_developmental ∧ progressiveness_tortuosity ∧ quantitative_qualitative_changes → tencent_development :=
by sorry

end NUMINAMATH_GPT_tencent_technological_innovation_basis_tencent_innovative_development_analysis_l2102_210268


namespace NUMINAMATH_GPT_find_second_remainder_l2102_210254

theorem find_second_remainder (k m n r : ℕ) 
  (h1 : n = 12 * k + 56) 
  (h2 : n = 34 * m + r) 
  (h3 : (22 + r) % 12 = 10) : 
  r = 10 :=
sorry

end NUMINAMATH_GPT_find_second_remainder_l2102_210254


namespace NUMINAMATH_GPT_prism_volume_eq_400_l2102_210222

noncomputable def prism_volume (a b c : ℝ) : ℝ := a * b * c

theorem prism_volume_eq_400 
  (a b c : ℝ)
  (h1 : a * b = 40)
  (h2 : a * c = 50)
  (h3 : b * c = 80) :
  prism_volume a b c = 400 :=
by
  sorry

end NUMINAMATH_GPT_prism_volume_eq_400_l2102_210222


namespace NUMINAMATH_GPT_arrangement_possible_32_arrangement_possible_100_l2102_210218

-- Problem (1)
theorem arrangement_possible_32 : 
  ∃ (f : Fin 32 → Fin 32), ∀ (a b : Fin 32), ∀ (i : Fin 32), 
    a < b → i < b → f i = (a + b) / 2 → False := 
sorry

-- Problem (2)
theorem arrangement_possible_100 : 
  ∃ (f : Fin 100 → Fin 100), ∀ (a b : Fin 100), ∀ (i : Fin 100),
    a < b → i < b → f i = (a + b) / 2 → False := 
sorry


end NUMINAMATH_GPT_arrangement_possible_32_arrangement_possible_100_l2102_210218


namespace NUMINAMATH_GPT_compare_slopes_l2102_210255

noncomputable def f (p q r x : ℝ) := x^3 + p * x^2 + q * x + r

noncomputable def s (p q x : ℝ) := 3 * x^2 + 2 * p * x + q

theorem compare_slopes (p q r a b c : ℝ) (hb : b ≠ 0) (ha : a ≠ c) 
  (hfa : f p q r a = 0) (hfc : f p q r c = 0) : a > c → s p q a > s p q c := 
by
  sorry

end NUMINAMATH_GPT_compare_slopes_l2102_210255


namespace NUMINAMATH_GPT_correct_word_is_any_l2102_210283

def words : List String := ["other", "any", "none", "some"]

def is_correct_word (word : String) : Prop :=
  "Jane was asked a lot of questions, but she didn’t answer " ++ word ++ " of them." = 
    "Jane was asked a lot of questions, but she didn’t answer any of them."

theorem correct_word_is_any : is_correct_word "any" :=
by
  sorry

end NUMINAMATH_GPT_correct_word_is_any_l2102_210283


namespace NUMINAMATH_GPT_geometric_series_sum_eq_4_div_3_l2102_210245

theorem geometric_series_sum_eq_4_div_3 (a : ℝ) (r : ℝ) (h₀ : a = 1) (h₁ : r = 1 / 4) :
  ∑' n : ℕ, a * r^n = 4 / 3 := by
  sorry

end NUMINAMATH_GPT_geometric_series_sum_eq_4_div_3_l2102_210245


namespace NUMINAMATH_GPT_measure_of_angle_l2102_210248

theorem measure_of_angle (x : ℝ) (h1 : 90 = x + (3 * x + 10)) : x = 20 :=
by
  sorry

end NUMINAMATH_GPT_measure_of_angle_l2102_210248


namespace NUMINAMATH_GPT_boys_in_school_l2102_210281

theorem boys_in_school (B G1 G2 : ℕ) (h1 : G1 = 632) (h2 : G2 = G1 + 465) (h3 : G2 = B + 687) : B = 410 :=
by
  sorry

end NUMINAMATH_GPT_boys_in_school_l2102_210281


namespace NUMINAMATH_GPT_age_difference_l2102_210279

theorem age_difference (p f : ℕ) (hp : p = 11) (hf : f = 42) : f - p = 31 :=
by
  sorry

end NUMINAMATH_GPT_age_difference_l2102_210279


namespace NUMINAMATH_GPT_hundred_days_from_friday_is_sunday_l2102_210225

/-- Given that today is Friday, determine that 100 days from now is Sunday. -/
theorem hundred_days_from_friday_is_sunday (today : ℕ) (days_in_week : ℕ := 7) 
(friday : ℕ := 0) (sunday : ℕ := 2) : (((today + 100) % days_in_week) = sunday) :=
sorry

end NUMINAMATH_GPT_hundred_days_from_friday_is_sunday_l2102_210225


namespace NUMINAMATH_GPT_prism_faces_l2102_210289

theorem prism_faces (E : ℕ) (h : E = 18) : 
  ∃ F : ℕ, F = 8 :=
by
  have L : ℕ := E / 3
  have F : ℕ := L + 2
  use F
  sorry

end NUMINAMATH_GPT_prism_faces_l2102_210289


namespace NUMINAMATH_GPT_A_n_is_integer_l2102_210295

open Real

noncomputable def A_n (a b : ℕ) (θ : ℝ) (n : ℕ) : ℝ :=
  (a^2 + b^2)^n * sin (n * θ)

theorem A_n_is_integer (a b : ℕ) (h : a > b) (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < pi/2) (h_sin : sin θ = 2 * a * b / (a^2 + b^2)) :
  ∀ n : ℕ, ∃ k : ℤ, A_n a b θ n = k :=
by
  sorry

end NUMINAMATH_GPT_A_n_is_integer_l2102_210295


namespace NUMINAMATH_GPT_range_of_a_l2102_210202

theorem range_of_a (x a : ℝ) : (∃ x : ℝ,  |x + 2| + |x - 3| ≤ |a - 1| ) ↔ (a ≤ -4 ∨ a ≥ 6) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l2102_210202


namespace NUMINAMATH_GPT_minimum_area_of_cyclic_quadrilateral_l2102_210280

theorem minimum_area_of_cyclic_quadrilateral :
  ∀ (r1 r2 : ℝ), (r1 = 1) ∧ (r2 = 2) →
    ∃ (A : ℝ), A = 3 * Real.sqrt 3 ∧ 
    (∀ (q : ℝ) (circumscribed : q ≤ A),
      ∀ (p : Prop), (p = (∃ x y z w, 
        ∀ (cx : ℝ) (cy : ℝ) (cr : ℝ), 
          cr = r2 ∧ 
          (Real.sqrt ((x - cx)^2 + (y - cy)^2) = r2) ∧ 
          (Real.sqrt ((z - cx)^2 + (w - cy)^2) = r2) ∧ 
          (Real.sqrt ((x - cx)^2 + (w - cy)^2) = r1) ∧ 
          (Real.sqrt ((z - cx)^2 + (y - cy)^2) = r1)
      )) → q = A) :=
sorry

end NUMINAMATH_GPT_minimum_area_of_cyclic_quadrilateral_l2102_210280


namespace NUMINAMATH_GPT_find_m_l2102_210266

open Set

theorem find_m (m : ℝ) (A B : Set ℝ)
  (h1 : A = {-1, 3, 2 * m - 1})
  (h2 : B = {3, m})
  (h3 : B ⊆ A) : m = 1 ∨ m = -1 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l2102_210266


namespace NUMINAMATH_GPT_CNY_share_correct_l2102_210251

noncomputable def total_NWF : ℝ := 1388.01
noncomputable def deductions_method1 : List ℝ := [41.89, 2.77, 478.48, 554.91, 0.24]
noncomputable def previous_year_share_CNY : ℝ := 17.77
noncomputable def deductions_method2 : List (ℝ × String) := [(3.02, "EUR"), (0.2, "USD"), (34.47, "GBP"), (39.98, "others"), (0.02, "other")]

theorem CNY_share_correct :
  let CNY22 := total_NWF - (deductions_method1.foldl (λ a b => a + b) 0)
  let alpha22_CNY := (CNY22 / total_NWF) * 100
  let method2_result := 100 - (deductions_method2.foldl (λ a b => a + b.1) 0)
  alpha22_CNY = 22.31 ∧ method2_result = 22.31 := 
sorry

end NUMINAMATH_GPT_CNY_share_correct_l2102_210251


namespace NUMINAMATH_GPT_num_ordered_pairs_xy_eq_2200_l2102_210219

/-- There are 24 ordered pairs (x, y) such that xy = 2200. -/
theorem num_ordered_pairs_xy_eq_2200 : 
  ∃ (n : ℕ), n = 24 ∧ (∃ divisors : Finset ℕ, 
    (∀ d ∈ divisors, 2200 % d = 0) ∧ 
    (divisors.card = 24)) := 
sorry

end NUMINAMATH_GPT_num_ordered_pairs_xy_eq_2200_l2102_210219


namespace NUMINAMATH_GPT_convex_2k_vertices_l2102_210203

theorem convex_2k_vertices (k : ℕ) (h1 : 2 ≤ k) (h2 : k ≤ 50)
    (P : Finset (EuclideanSpace ℝ (Fin 2)))
    (hP : P.card = 100) (M : Finset (EuclideanSpace ℝ (Fin 2)))
    (hM : M.card = k) : 
  ∃ V : Finset (EuclideanSpace ℝ (Fin 2)), V.card = 2 * k ∧ ∀ m ∈ M, m ∈ convexHull ℝ V :=
by
  sorry

end NUMINAMATH_GPT_convex_2k_vertices_l2102_210203


namespace NUMINAMATH_GPT_find_number_l2102_210277

theorem find_number (x : ℤ) (h : 4 * x = 28) : x = 7 :=
sorry

end NUMINAMATH_GPT_find_number_l2102_210277


namespace NUMINAMATH_GPT_total_number_of_balls_is_twelve_l2102_210287

noncomputable def num_total_balls (a : ℕ) : Prop :=
(3 : ℚ) / a = (25 : ℚ) / 100

theorem total_number_of_balls_is_twelve : num_total_balls 12 :=
by sorry

end NUMINAMATH_GPT_total_number_of_balls_is_twelve_l2102_210287


namespace NUMINAMATH_GPT_largest_expr_is_a_squared_plus_b_squared_l2102_210221

noncomputable def largest_expression (a b : ℝ) (h₁ : a > 1) (h₂ : b > 1) (h₃ : a ≠ b) : Prop :=
  (a^2 + b^2 > a - b) ∧ (a^2 + b^2 > a + b) ∧ (a^2 + b^2 > 2 * a * b)

theorem largest_expr_is_a_squared_plus_b_squared (a b : ℝ) (h₁ : a > 1) (h₂ : b > 1) (h₃ : a ≠ b) : 
  largest_expression a b h₁ h₂ h₃ :=
by
  sorry

end NUMINAMATH_GPT_largest_expr_is_a_squared_plus_b_squared_l2102_210221


namespace NUMINAMATH_GPT_smoking_lung_cancer_problem_l2102_210211

-- Defining the confidence relationship
def smoking_related_to_lung_cancer (confidence: ℝ) := confidence > 0.99

-- Statement 4: Among 100 smokers, it is possible that not a single person has lung cancer.
def statement_4 (N: ℕ) (p: ℝ) := N = 100 ∧ 0 ≤ p ∧ p ≤ 1 ∧ p ^ 100 > 0

-- The main theorem statement in Lean 4
theorem smoking_lung_cancer_problem (confidence: ℝ) (N: ℕ) (p: ℝ) 
  (h1: smoking_related_to_lung_cancer confidence): 
  statement_4 N p :=
by
  sorry -- Proof goes here

end NUMINAMATH_GPT_smoking_lung_cancer_problem_l2102_210211


namespace NUMINAMATH_GPT_series_sum_l2102_210238

noncomputable def series_term (n : ℕ) : ℝ :=
  (4 * n + 2) / ((6 * n - 5)^2 * (6 * n + 1)^2)

theorem series_sum :
  (∑' n : ℕ, series_term (n + 1)) = 1 / 6 :=
by
  sorry

end NUMINAMATH_GPT_series_sum_l2102_210238


namespace NUMINAMATH_GPT_correct_statement_l2102_210232

-- Define the conditions as assumptions

/-- Condition 1: To understand the service life of a batch of new energy batteries, a sampling survey can be used. -/
def condition1 : Prop := True

/-- Condition 2: If the probability of winning a lottery is 2%, then buying 50 of these lottery tickets at once will definitely win. -/
def condition2 : Prop := False

/-- Condition 3: If the average of two sets of data, A and B, is the same, SA^2=2.3, SB^2=4.24, then set B is more stable. -/
def condition3 : Prop := False

/-- Condition 4: Rolling a die with uniform density and getting a score of 0 is a certain event. -/
def condition4 : Prop := False

-- The main theorem to prove the correct statement is A
theorem correct_statement : condition1 = True ∧ condition2 = False ∧ condition3 = False ∧ condition4 = False :=
by
  constructor; repeat { try { exact True.intro }; try { exact False.elim (by sorry) } }

end NUMINAMATH_GPT_correct_statement_l2102_210232


namespace NUMINAMATH_GPT_probability_of_winning_fifth_game_championship_correct_overall_probability_of_winning_championship_correct_l2102_210227

noncomputable def binomial (n k : ℕ) : ℝ :=
  (Nat.choose n k : ℝ)

noncomputable def probability_of_winning_fifth_game_championship : ℝ :=
  binomial 4 3 * 0.6^4 * 0.4

noncomputable def overall_probability_of_winning_championship : ℝ :=
  0.6^4 +
  binomial 4 3 * 0.6^4 * 0.4 +
  binomial 5 3 * 0.6^4 * 0.4^2 +
  binomial 6 3 * 0.6^4 * 0.4^3

theorem probability_of_winning_fifth_game_championship_correct :
  probability_of_winning_fifth_game_championship = 0.20736 := by
  sorry

theorem overall_probability_of_winning_championship_correct :
  overall_probability_of_winning_championship = 0.710208 := by
  sorry

end NUMINAMATH_GPT_probability_of_winning_fifth_game_championship_correct_overall_probability_of_winning_championship_correct_l2102_210227
