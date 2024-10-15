import Mathlib

namespace NUMINAMATH_GPT_average_of_four_variables_l2222_222266

theorem average_of_four_variables (x y z w : ℝ) (h : (5 / 2) * (x + y + z + w) = 25) :
  (x + y + z + w) / 4 = 2.5 :=
sorry

end NUMINAMATH_GPT_average_of_four_variables_l2222_222266


namespace NUMINAMATH_GPT_determine_m_if_root_exists_l2222_222251

def fractional_equation_has_root (x m : ℝ) : Prop :=
  (3 / (x - 4) + (x + m) / (4 - x) = 1)

theorem determine_m_if_root_exists (x : ℝ) (h : fractional_equation_has_root x m) : m = -1 :=
sorry

end NUMINAMATH_GPT_determine_m_if_root_exists_l2222_222251


namespace NUMINAMATH_GPT_hiring_manager_acceptance_l2222_222203

theorem hiring_manager_acceptance {k : ℤ} 
  (avg_age : ℤ) (std_dev : ℤ) (num_accepted_ages : ℤ) 
  (h_avg : avg_age = 20) (h_std_dev : std_dev = 8)
  (h_num_accepted : num_accepted_ages = 17) : 
  (20 + k * 8 - (20 - k * 8) + 1) = 17 → k = 1 :=
by
  intros
  sorry

end NUMINAMATH_GPT_hiring_manager_acceptance_l2222_222203


namespace NUMINAMATH_GPT_cloth_coloring_problem_l2222_222226

theorem cloth_coloring_problem (lengthOfCloth : ℕ) 
  (women_can_color_100m_in_1_day : 5 * 1 = 100) 
  (women_can_color_in_3_days : 6 * 3 = lengthOfCloth) : lengthOfCloth = 360 := 
sorry

end NUMINAMATH_GPT_cloth_coloring_problem_l2222_222226


namespace NUMINAMATH_GPT_peyton_juice_boxes_needed_l2222_222255

def juice_boxes_needed
  (john_juice_per_day : ℕ)
  (samantha_juice_per_day : ℕ)
  (heather_juice_mon_wed : ℕ)
  (heather_juice_tue_thu : ℕ)
  (heather_juice_fri : ℕ)
  (john_weeks : ℕ)
  (samantha_weeks : ℕ)
  (heather_weeks : ℕ)
  : ℕ :=
  let john_juice_per_week := john_juice_per_day * 5
  let samantha_juice_per_week := samantha_juice_per_day * 5
  let heather_juice_per_week := heather_juice_mon_wed * 2 + heather_juice_tue_thu * 2 + heather_juice_fri
  let john_total_juice := john_juice_per_week * john_weeks
  let samantha_total_juice := samantha_juice_per_week * samantha_weeks
  let heather_total_juice := heather_juice_per_week * heather_weeks
  john_total_juice + samantha_total_juice + heather_total_juice

theorem peyton_juice_boxes_needed :
  juice_boxes_needed 2 1 3 2 1 25 20 25 = 625 :=
by
  sorry

end NUMINAMATH_GPT_peyton_juice_boxes_needed_l2222_222255


namespace NUMINAMATH_GPT_smallest_n_cube_ends_with_2016_l2222_222204

theorem smallest_n_cube_ends_with_2016 : ∃ n : ℕ, (n^3 % 10000 = 2016) ∧ (∀ m : ℕ, (m^3 % 10000 = 2016) → n ≤ m) :=
sorry

end NUMINAMATH_GPT_smallest_n_cube_ends_with_2016_l2222_222204


namespace NUMINAMATH_GPT_tens_digit_of_3_pow_2023_l2222_222246

theorem tens_digit_of_3_pow_2023 : (3 ^ 2023 % 100) / 10 = 2 := 
sorry

end NUMINAMATH_GPT_tens_digit_of_3_pow_2023_l2222_222246


namespace NUMINAMATH_GPT_smallest_repeating_block_digits_l2222_222208

theorem smallest_repeating_block_digits (n : ℕ) (d : ℕ) (hd_pos : d > 0) (hd_coprime : Nat.gcd n d = 1)
  (h_fraction : (n : ℚ) / d = 8 / 11) : n = 2 :=
by
  -- proof will go here
  sorry

end NUMINAMATH_GPT_smallest_repeating_block_digits_l2222_222208


namespace NUMINAMATH_GPT_evening_temperature_l2222_222236

-- Define the given conditions
def t_noon : ℤ := 1
def d : ℤ := 3

-- The main theorem stating that the evening temperature is -2℃
theorem evening_temperature : t_noon - d = -2 := by
  sorry

end NUMINAMATH_GPT_evening_temperature_l2222_222236


namespace NUMINAMATH_GPT_average_salary_all_workers_l2222_222295

/-- The total number of workers in the workshop is 15 -/
def total_number_of_workers : ℕ := 15

/-- The number of technicians is 5 -/
def number_of_technicians : ℕ := 5

/-- The number of other workers is given by the total number minus technicians -/
def number_of_other_workers : ℕ := total_number_of_workers - number_of_technicians

/-- The average salary per head of the technicians is Rs. 800 -/
def average_salary_per_technician : ℕ := 800

/-- The average salary per head of the other workers is Rs. 650 -/
def average_salary_per_other_worker : ℕ := 650

/-- The total salary for all the workers -/
def total_salary : ℕ := (number_of_technicians * average_salary_per_technician) + (number_of_other_workers * average_salary_per_other_worker)

/-- The average salary per head of all the workers in the workshop is Rs. 700 -/
theorem average_salary_all_workers :
  total_salary / total_number_of_workers = 700 := by
  sorry

end NUMINAMATH_GPT_average_salary_all_workers_l2222_222295


namespace NUMINAMATH_GPT_quadratic_common_root_l2222_222267

theorem quadratic_common_root (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h1 : ∃ x, x^2 + a * x + b = 0 ∧ x^2 + c * x + a = 0)
  (h2 : ∃ x, x^2 + a * x + b = 0 ∧ x^2 + b * x + c = 0)
  (h3 : ∃ x, x^2 + b * x + c = 0 ∧ x^2 + c * x + a = 0) :
  a^2 + b^2 + c^2 = 6 :=
sorry

end NUMINAMATH_GPT_quadratic_common_root_l2222_222267


namespace NUMINAMATH_GPT_range_of_a_l2222_222218

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - x^3

theorem range_of_a (a : ℝ) :
  (∀ (x₁ x₂ : ℝ), 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 → f a x₂ - f a x₁ > x₂ - x₁) →
  a ≥ 4 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_range_of_a_l2222_222218


namespace NUMINAMATH_GPT_gumball_problem_l2222_222244

theorem gumball_problem:
  ∀ (total_gumballs given_to_Todd given_to_Alisha given_to_Bobby remaining_gumballs: ℕ),
    total_gumballs = 45 →
    given_to_Todd = 4 →
    given_to_Alisha = 2 * given_to_Todd →
    remaining_gumballs = 6 →
    given_to_Todd + given_to_Alisha + given_to_Bobby + remaining_gumballs = total_gumballs →
    given_to_Bobby = 45 - 18 →
    4 * given_to_Alisha - given_to_Bobby = 5 :=
by
  intros total_gumballs given_to_Todd given_to_Alisha given_to_Bobby remaining_gumballs ht hTodd hAlisha hRemaining hSum hBobby
  rw [ht, hTodd] at *
  rw [hAlisha, hRemaining] at *
  sorry

end NUMINAMATH_GPT_gumball_problem_l2222_222244


namespace NUMINAMATH_GPT_pyramid_four_triangular_faces_area_l2222_222297

theorem pyramid_four_triangular_faces_area 
  (base_edge : ℝ) (lateral_edge : ℝ) 
  (h_base : base_edge = 8)
  (h_lateral : lateral_edge = 7) :
  let h := Real.sqrt (lateral_edge ^ 2 - (base_edge / 2) ^ 2)
  let triangle_area := (1 / 2) * base_edge * h
  let total_area := 4 * triangle_area
  total_area = 16 * Real.sqrt 33 :=
by
  -- Definitions to introduce local values
  let half_base := base_edge / 2
  let h := Real.sqrt (lateral_edge ^ 2 - half_base ^ 2)
  let triangle_area := (1 / 2) * base_edge * h
  let total_area := 4 * triangle_area
  -- Assertion to compare calculated total area with given correct answer
  have h_eq : h = Real.sqrt 33 := by sorry
  have triangle_area_eq : triangle_area = 4 * Real.sqrt 33 := by sorry
  have total_area_eq : total_area = 16 * Real.sqrt 33 := by sorry
  exact total_area_eq

end NUMINAMATH_GPT_pyramid_four_triangular_faces_area_l2222_222297


namespace NUMINAMATH_GPT_total_distance_traveled_l2222_222273

theorem total_distance_traveled 
  (Vm : ℝ) (Vr : ℝ) (T_total : ℝ) (D : ℝ) 
  (H_Vm : Vm = 6) 
  (H_Vr : Vr = 1.2) 
  (H_T_total : T_total = 1) 
  (H_time_eq : D / (Vm - Vr) + D / (Vm + Vr) = T_total) 
  : 2 * D = 5.76 := 
by sorry

end NUMINAMATH_GPT_total_distance_traveled_l2222_222273


namespace NUMINAMATH_GPT_Harry_bought_five_packets_of_chili_pepper_l2222_222229

noncomputable def price_pumpkin : ℚ := 2.50
noncomputable def price_tomato : ℚ := 1.50
noncomputable def price_chili_pepper : ℚ := 0.90
noncomputable def packets_pumpkin : ℕ := 3
noncomputable def packets_tomato : ℕ := 4
noncomputable def total_spent : ℚ := 18
noncomputable def packets_chili_pepper (p : ℕ) := price_pumpkin * packets_pumpkin + price_tomato * packets_tomato + price_chili_pepper * p = total_spent

theorem Harry_bought_five_packets_of_chili_pepper :
  ∃ p : ℕ, packets_chili_pepper p ∧ p = 5 :=
by 
  sorry

end NUMINAMATH_GPT_Harry_bought_five_packets_of_chili_pepper_l2222_222229


namespace NUMINAMATH_GPT_solid_is_cylinder_l2222_222207

def solid_views (v1 v2 v3 : String) : Prop := 
  -- This definition makes a placeholder for the views of the solid.
  sorry

def is_cylinder (s : String) : Prop := 
  s = "Cylinder"

theorem solid_is_cylinder (v1 v2 v3 : String) (h : solid_views v1 v2 v3) :
  ∃ s : String, is_cylinder s :=
sorry

end NUMINAMATH_GPT_solid_is_cylinder_l2222_222207


namespace NUMINAMATH_GPT_f_1984_and_f_1985_l2222_222253

namespace Proof

variable {N M : Type} [AddMonoid M] [Zero M] (f : ℕ → M)

-- Conditions
axiom f_10 : f 10 = 0
axiom f_last_digit_3 {n : ℕ} : (n % 10 = 3) → f n = 0
axiom f_mn (m n : ℕ) : f (m * n) = f m + f n

-- Prove f(1984) = 0 and f(1985) = 0
theorem f_1984_and_f_1985 : f 1984 = 0 ∧ f 1985 = 0 :=
by
  sorry

end Proof

end NUMINAMATH_GPT_f_1984_and_f_1985_l2222_222253


namespace NUMINAMATH_GPT_value_of_M_l2222_222261

theorem value_of_M (x y z M : ℝ) (h1 : x + y + z = 90)
    (h2 : x - 5 = M)
    (h3 : y + 5 = M)
    (h4 : 5 * z = M) :
    M = 450 / 11 :=
by
    sorry

end NUMINAMATH_GPT_value_of_M_l2222_222261


namespace NUMINAMATH_GPT_ratio_of_counters_l2222_222239

theorem ratio_of_counters (C_K M_K C_total M_ratio : ℕ)
  (h1 : C_K = 40)
  (h2 : M_K = 50)
  (h3 : M_ratio = 4 * M_K)
  (h4 : C_total = C_K + M_ratio)
  (h5 : C_total = 320) :
  C_K ≠ 0 → (320 - M_ratio) / C_K = 3 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_counters_l2222_222239


namespace NUMINAMATH_GPT_pull_ups_of_fourth_student_l2222_222242

theorem pull_ups_of_fourth_student 
  (avg_pullups : ℕ) 
  (num_students : ℕ) 
  (pullups_first : ℕ) 
  (pullups_second : ℕ) 
  (pullups_third : ℕ) 
  (pullups_fifth : ℕ) 
  (H_avg : avg_pullups = 10) 
  (H_students : num_students = 5) 
  (H_first : pullups_first = 9) 
  (H_second : pullups_second = 12) 
  (H_third : pullups_third = 9) 
  (H_fifth : pullups_fifth = 8) : 
  ∃ (pullups_fourth : ℕ), pullups_fourth = 12 := by
  sorry

end NUMINAMATH_GPT_pull_ups_of_fourth_student_l2222_222242


namespace NUMINAMATH_GPT_x_in_M_sufficient_condition_for_x_in_N_l2222_222247

def M := {y : ℝ | ∃ x : ℝ, y = 2^x ∧ x < 0}
def N := {y : ℝ | ∃ x : ℝ, y = Real.sqrt ((1 - x) / x)}

theorem x_in_M_sufficient_condition_for_x_in_N :
  (∀ x, x ∈ M → x ∈ N) ∧ ¬ (∀ x, x ∈ N → x ∈ M) :=
by sorry

end NUMINAMATH_GPT_x_in_M_sufficient_condition_for_x_in_N_l2222_222247


namespace NUMINAMATH_GPT_number_of_apples_ratio_of_mixed_fruits_total_weight_of_oranges_l2222_222291

theorem number_of_apples (total_fruit : ℕ) (oranges_fraction : ℚ) (peaches_fraction : ℚ) (apples_mult : ℕ) (total_fruit_value : total_fruit = 56) :
    oranges_fraction = 1/4 → 
    peaches_fraction = 1/2 → 
    apples_mult = 5 → 
    (apples_mult * peaches_fraction * oranges_fraction * total_fruit) = 35 :=
by
  intros h1 h2 h3
  sorry

theorem ratio_of_mixed_fruits (total_fruit : ℕ) (oranges_fraction : ℚ) (peaches_fraction : ℚ) (mixed_fruits_mult : ℕ) (total_fruit_value : total_fruit = 56) :
    oranges_fraction = 1/4 → 
    peaches_fraction = 1/2 → 
    mixed_fruits_mult = 2 → 
    (mixed_fruits_mult * peaches_fraction * oranges_fraction * total_fruit) / total_fruit = 1/4 :=
by
  intros h1 h2 h3
  sorry

theorem total_weight_of_oranges (total_fruit : ℕ) (oranges_fraction : ℚ) (orange_weight : ℕ) (total_fruit_value : total_fruit = 56) :
    oranges_fraction = 1/4 → 
    orange_weight = 200 → 
    (orange_weight * oranges_fraction * total_fruit) = 2800 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_number_of_apples_ratio_of_mixed_fruits_total_weight_of_oranges_l2222_222291


namespace NUMINAMATH_GPT_cost_of_dog_l2222_222292

-- Given conditions
def dollars_misha_has : ℕ := 34
def dollars_misha_needs_earn : ℕ := 13

-- Formal statement of the mathematic proof
theorem cost_of_dog : dollars_misha_has + dollars_misha_needs_earn = 47 := by
  sorry

end NUMINAMATH_GPT_cost_of_dog_l2222_222292


namespace NUMINAMATH_GPT_unique_digit_for_prime_l2222_222224

theorem unique_digit_for_prime (B : ℕ) (hB : B < 10) (hprime : Nat.Prime (30420 * 10 + B)) : B = 1 :=
sorry

end NUMINAMATH_GPT_unique_digit_for_prime_l2222_222224


namespace NUMINAMATH_GPT_find_n_l2222_222220

theorem find_n (n x y k : ℕ) (h_coprime : Nat.gcd x y = 1) (h_eq : 3^n = x^k + y^k) : n = 2 :=
sorry

end NUMINAMATH_GPT_find_n_l2222_222220


namespace NUMINAMATH_GPT_book_contains_300_pages_l2222_222243

-- The given conditions
def total_digits : ℕ := 792
def digits_per_page_1_to_9 : ℕ := 9 * 1
def digits_per_page_10_to_99 : ℕ := 90 * 2
def remaining_digits : ℕ := total_digits - digits_per_page_1_to_9 - digits_per_page_10_to_99
def pages_with_3_digits : ℕ := remaining_digits / 3

-- The total number of pages
def total_pages : ℕ := 99 + pages_with_3_digits

theorem book_contains_300_pages : total_pages = 300 := by
  sorry

end NUMINAMATH_GPT_book_contains_300_pages_l2222_222243


namespace NUMINAMATH_GPT_arc_length_is_correct_l2222_222268

-- Define the radius and central angle as given
def radius := 16
def central_angle := 2

-- Define the arc length calculation
def arc_length (r : ℕ) (α : ℕ) := α * r

-- The theorem stating the mathematically equivalent proof problem
theorem arc_length_is_correct : arc_length radius central_angle = 32 :=
by sorry

end NUMINAMATH_GPT_arc_length_is_correct_l2222_222268


namespace NUMINAMATH_GPT_vacuum_upstairs_more_than_twice_downstairs_l2222_222281

theorem vacuum_upstairs_more_than_twice_downstairs 
  (x y : ℕ) 
  (h1 : 27 = 2 * x + y) 
  (h2 : x + 27 = 38) : 
  y = 5 :=
by 
  sorry

end NUMINAMATH_GPT_vacuum_upstairs_more_than_twice_downstairs_l2222_222281


namespace NUMINAMATH_GPT_simultaneous_equations_solution_exists_l2222_222298

theorem simultaneous_equations_solution_exists (m : ℝ) :
  ∃ x y : ℝ, y = 3 * m * x + 2 ∧ y = (3 * m - 2) * x + 5 :=
by
  sorry

end NUMINAMATH_GPT_simultaneous_equations_solution_exists_l2222_222298


namespace NUMINAMATH_GPT_discount_difference_is_24_l2222_222294

-- Definitions based on conditions
def smartphone_price : ℝ := 800
def single_discount_rate : ℝ := 0.25
def first_successive_discount_rate : ℝ := 0.20
def second_successive_discount_rate : ℝ := 0.10

-- Definitions of discounted prices
def single_discount_price (p : ℝ) (d1 : ℝ) : ℝ := p * (1 - d1)
def successive_discount_price (p : ℝ) (d1 : ℝ) (d2 : ℝ) : ℝ := 
  let intermediate_price := p * (1 - d1) 
  intermediate_price * (1 - d2)

-- Calculate the difference between the two final prices
def price_difference (p : ℝ) (d1 : ℝ) (d2 : ℝ) (d3 : ℝ) : ℝ :=
  (single_discount_price p d1) - (successive_discount_price p d2 d3)

theorem discount_difference_is_24 :
  price_difference smartphone_price single_discount_rate first_successive_discount_rate second_successive_discount_rate = 24 := 
sorry

end NUMINAMATH_GPT_discount_difference_is_24_l2222_222294


namespace NUMINAMATH_GPT_solve_real_triples_l2222_222210

theorem solve_real_triples (a b c : ℝ) :
  (a * (b^2 + c) = c * (c + a * b) ∧
   b * (c^2 + a) = a * (a + b * c) ∧
   c * (a^2 + b) = b * (b + c * a)) ↔ 
  (∃ (x : ℝ), (a = x) ∧ (b = x) ∧ (c = x)) ∨ 
  (b = 0 ∧ c = 0) :=
sorry

end NUMINAMATH_GPT_solve_real_triples_l2222_222210


namespace NUMINAMATH_GPT_train_b_speed_l2222_222271

/-- Two trains, A and B, start simultaneously from two stations 480 kilometers apart and meet after 2.5 hours. 
Train A travels at a speed of 102 kilometers per hour. What is the speed of train B in kilometers per hour? -/
theorem train_b_speed (d t : ℝ) (speedA speedB : ℝ) (h1 : d = 480) (h2 : t = 2.5) (h3 : speedA = 102)
  (h4 : speedA * t + speedB * t = d) : speedB = 90 := 
by
  sorry

end NUMINAMATH_GPT_train_b_speed_l2222_222271


namespace NUMINAMATH_GPT_range_of_a_l2222_222285

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x ≤ 4 → (2 * x + 2 * (a - 1)) ≤ 0) → a ≤ -3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l2222_222285


namespace NUMINAMATH_GPT_functional_equation_solution_l2222_222200

theorem functional_equation_solution {f : ℚ → ℚ} :
  (∀ x y z t : ℚ, x < y ∧ y < z ∧ z < t ∧ (y - x) = (z - y) ∧ (z - y) = (t - z) →
    f x + f t = f y + f z) → 
  ∃ c b : ℚ, ∀ q : ℚ, f q = c * q + b := 
by
  sorry

end NUMINAMATH_GPT_functional_equation_solution_l2222_222200


namespace NUMINAMATH_GPT_five_b_value_l2222_222254

theorem five_b_value (a b : ℚ) 
  (h1 : 3 * a + 4 * b = 4) 
  (h2 : a = b - 3) : 
  5 * b = 65 / 7 := 
by
  sorry

end NUMINAMATH_GPT_five_b_value_l2222_222254


namespace NUMINAMATH_GPT_simplify_identity_l2222_222260

theorem simplify_identity :
  ∀ θ : ℝ, θ = 160 → (1 / (Real.sqrt (1 + Real.tan (θ : ℝ) ^ 2))) = -Real.cos (θ : ℝ) :=
by
  intro θ h
  rw [h]
  sorry  

end NUMINAMATH_GPT_simplify_identity_l2222_222260


namespace NUMINAMATH_GPT_value_of_J_l2222_222296

theorem value_of_J (J : ℕ) : 32^4 * 4^4 = 2^J → J = 28 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_value_of_J_l2222_222296


namespace NUMINAMATH_GPT_find_a_l2222_222221

variable (m n a : ℝ)
variable (h1 : m = 2 * n + 5)
variable (h2 : m + a = 2 * (n + 1.5) + 5)

theorem find_a : a = 3 := by
  sorry

end NUMINAMATH_GPT_find_a_l2222_222221


namespace NUMINAMATH_GPT_width_of_house_l2222_222217

theorem width_of_house (L P_L P_W A_total : ℝ) (hL : L = 20.5) (hPL : P_L = 6) (hPW : P_W = 4.5) (hAtotal : A_total = 232) :
  ∃ W : ℝ, W = 10 :=
by
  have area_porch : ℝ := P_L * P_W
  have area_house := A_total - area_porch
  use area_house / L
  sorry

end NUMINAMATH_GPT_width_of_house_l2222_222217


namespace NUMINAMATH_GPT_xy_sum_value_l2222_222201

theorem xy_sum_value (x y : ℝ) (h1 : x + Real.cos y = 1010) (h2 : x + 1010 * Real.sin y = 1009) (h3 : (Real.pi / 4) ≤ y ∧ y ≤ (Real.pi / 2)) :
  x + y = 1010 + (Real.pi / 2) := 
by
  sorry

end NUMINAMATH_GPT_xy_sum_value_l2222_222201


namespace NUMINAMATH_GPT_avg_children_in_families_with_children_l2222_222287

-- Define the conditions
def num_families : ℕ := 15
def avg_children_per_family : ℤ := 3
def num_childless_families : ℕ := 3

-- Total number of children among all families
def total_children : ℤ := num_families * avg_children_per_family

-- Number of families with children
def num_families_with_children : ℕ := num_families - num_childless_families

-- Average number of children in families with children, to be proven equal 3.8 when rounded to the nearest tenth.
theorem avg_children_in_families_with_children : (total_children : ℚ) / num_families_with_children = 3.8 := by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_avg_children_in_families_with_children_l2222_222287


namespace NUMINAMATH_GPT_f_value_at_3_l2222_222214

noncomputable def f : ℝ → ℝ := sorry

theorem f_value_at_3 (h : ∀ x : ℝ, f x + 2 * f (1 - x) = 4 * x^2) : f 3 = -4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_f_value_at_3_l2222_222214


namespace NUMINAMATH_GPT_steve_distance_l2222_222277

theorem steve_distance (D : ℝ) (S : ℝ) 
  (h1 : 2 * S = 10)
  (h2 : (D / S) + (D / (2 * S)) = 6) :
  D = 20 :=
by
  sorry

end NUMINAMATH_GPT_steve_distance_l2222_222277


namespace NUMINAMATH_GPT_divisor_of_difference_is_62_l2222_222250

-- The problem conditions as definitions
def x : Int := 859622
def y : Int := 859560
def difference : Int := x - y

-- The proof statement
theorem divisor_of_difference_is_62 (d : Int) (h₁ : d ∣ y) (h₂ : d ∣ difference) : d = 62 := by
  sorry

end NUMINAMATH_GPT_divisor_of_difference_is_62_l2222_222250


namespace NUMINAMATH_GPT_louise_winning_strategy_2023x2023_l2222_222258

theorem louise_winning_strategy_2023x2023 :
  ∀ (n : ℕ), (n % 2 = 1) → (n = 2023) →
  ∃ (strategy : ℕ × ℕ → Prop),
    (∀ turn : ℕ, ∃ (i j : ℕ), i < n ∧ j < n ∧ strategy (i, j)) ∧
    (∃ i j : ℕ, strategy (i, j) ∧ (i = 0 ∧ j = 0)) :=
by
  sorry

end NUMINAMATH_GPT_louise_winning_strategy_2023x2023_l2222_222258


namespace NUMINAMATH_GPT_S_63_value_l2222_222278

noncomputable def b (n : ℕ) : ℚ := (3 + (-1)^(n-1))/2

noncomputable def a : ℕ → ℚ
| 0       => 0
| 1       => 2
| (n+2)   => if (n % 2 = 0) then - (a (n+1))/2 else 2 - 2*(a (n+1))

noncomputable def S : ℕ → ℚ
| 0       => 0
| (n+1)   => S n + a (n+1)

theorem S_63_value : S 63 = 464 := by
  sorry

end NUMINAMATH_GPT_S_63_value_l2222_222278


namespace NUMINAMATH_GPT_sandy_savings_last_year_l2222_222286

theorem sandy_savings_last_year (S : ℝ) (P : ℝ) 
(h1 : P / 100 * S = x)
(h2 : 1.10 * S = y)
(h3 : 0.10 * y = 0.11 * S)
(h4 : 0.11 * S = 1.8333333333333331 * x) :
P = 6 := by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_sandy_savings_last_year_l2222_222286


namespace NUMINAMATH_GPT_smallest_n_for_triangle_area_l2222_222230

theorem smallest_n_for_triangle_area :
  ∃ n : ℕ, 10 * n^4 - 8 * n^3 - 52 * n^2 + 32 * n - 24 > 10000 ∧ ∀ m : ℕ, 
  (m < n → ¬ (10 * m^4 - 8 * m^3 - 52 * m^2 + 32 * m - 24 > 10000)) :=
sorry

end NUMINAMATH_GPT_smallest_n_for_triangle_area_l2222_222230


namespace NUMINAMATH_GPT_minimize_xy_l2222_222269

theorem minimize_xy (x y : ℕ) (hx : x > 0) (hy : y > 0) (h_eq : 7 * x + 4 * y = 200) : (x * y = 172) :=
sorry

end NUMINAMATH_GPT_minimize_xy_l2222_222269


namespace NUMINAMATH_GPT_area_of_FDBG_l2222_222237

noncomputable def area_quadrilateral (AB AC : ℝ) (area_ABC : ℝ) : ℝ :=
  let AD := AB / 2
  let AE := AC / 2
  let sin_A := (2 * area_ABC) / (AB * AC)
  let area_ADE := (1 / 2) * AD * AE * sin_A
  let BC := (2 * area_ABC) / (AC * sin_A)
  let GC := BC / 3
  let area_AGC := (1 / 2) * AC * GC * sin_A
  area_ABC - (area_ADE + area_AGC)

theorem area_of_FDBG (AB AC : ℝ) (area_ABC : ℝ)
  (h1 : AB = 30)
  (h2 : AC = 15) 
  (h3 : area_ABC = 90) :
  area_quadrilateral AB AC area_ABC = 37.5 :=
by
  intros
  sorry

end NUMINAMATH_GPT_area_of_FDBG_l2222_222237


namespace NUMINAMATH_GPT_find_some_number_l2222_222206

noncomputable def some_number : ℝ := 1000
def expr_approx (a b c d : ℝ) := (a * b) / c = d

theorem find_some_number :
  expr_approx 3.241 14 some_number 0.045374000000000005 :=
by sorry

end NUMINAMATH_GPT_find_some_number_l2222_222206


namespace NUMINAMATH_GPT_inequality_solution_l2222_222283

theorem inequality_solution (a : ℝ) (x : ℝ) 
  (h₁ : 0 < a) 
  (h₂ : 1 < a) 
  (y₁ : ℝ := a^(2 * x + 1)) 
  (y₂ : ℝ := a^(-3 * x)) :
  y₁ > y₂ → x > - (1 / 5) :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l2222_222283


namespace NUMINAMATH_GPT_average_speed_is_65_l2222_222270

-- Definitions based on the problem's conditions
def speed_first_hour : ℝ := 100 -- 100 km in the first hour
def speed_second_hour : ℝ := 30 -- 30 km in the second hour
def total_distance : ℝ := speed_first_hour + speed_second_hour -- total distance
def total_time : ℝ := 2 -- total time in hours (1 hour + 1 hour)

-- Problem: prove that the average speed is 65 km/h
theorem average_speed_is_65 : (total_distance / total_time) = 65 := by
  sorry

end NUMINAMATH_GPT_average_speed_is_65_l2222_222270


namespace NUMINAMATH_GPT_scissors_total_l2222_222259

theorem scissors_total (original_scissors : ℕ) (added_scissors : ℕ) (total_scissors : ℕ) 
  (h1 : original_scissors = 39)
  (h2 : added_scissors = 13)
  (h3 : total_scissors = original_scissors + added_scissors) : total_scissors = 52 :=
by
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_scissors_total_l2222_222259


namespace NUMINAMATH_GPT_inverse_of_g_l2222_222212

noncomputable def u (x : ℝ) : ℝ := sorry
noncomputable def v (x : ℝ) : ℝ := sorry
noncomputable def w (x : ℝ) : ℝ := sorry

noncomputable def u_inv (x : ℝ) : ℝ := sorry
noncomputable def v_inv (x : ℝ) : ℝ := sorry
noncomputable def w_inv (x : ℝ) : ℝ := sorry

lemma u_inverse : ∀ x, u_inv (u x) = x ∧ u (u_inv x) = x := sorry
lemma v_inverse : ∀ x, v_inv (v x) = x ∧ v (v_inv x) = x := sorry
lemma w_inverse : ∀ x, w_inv (w x) = x ∧ w (w_inv x) = x := sorry

noncomputable def g (x : ℝ) : ℝ := v (u (w x))

noncomputable def g_inv (x : ℝ) : ℝ := w_inv (u_inv (v_inv x))

theorem inverse_of_g :
  ∀ x : ℝ, g_inv (g x) = x ∧ g (g_inv x) = x :=
by
  intro x
  -- proof omitted
  sorry

end NUMINAMATH_GPT_inverse_of_g_l2222_222212


namespace NUMINAMATH_GPT_largest_common_term_l2222_222299

/-- The arithmetic progression sequence1 --/
def sequence1 (n : ℕ) : ℤ := 4 + 5 * n

/-- The arithmetic progression sequence2 --/
def sequence2 (n : ℕ) : ℤ := 5 + 8 * n

/-- The common term condition for sequence1 --/
def common_term_condition1 (a : ℤ) : Prop := ∃ n : ℕ, a = sequence1 n

/-- The common term condition for sequence2 --/
def common_term_condition2 (a : ℤ) : Prop := ∃ n : ℕ, a = sequence2 n

/-- The largest common term less than 1000 --/
def is_largest_common_term (a : ℤ) : Prop :=
  common_term_condition1 a ∧ common_term_condition2 a ∧ a < 1000 ∧
  ∀ b : ℤ, common_term_condition1 b ∧ common_term_condition2 b ∧ b < 1000 → b ≤ a

/-- Lean theorem statement --/
theorem largest_common_term :
  ∃ a : ℤ, is_largest_common_term a ∧ a = 989 :=
sorry

end NUMINAMATH_GPT_largest_common_term_l2222_222299


namespace NUMINAMATH_GPT_milk_production_l2222_222279

variables (x α y z w β v : ℝ)

theorem milk_production :
  (w * v * β * y) / (α^2 * x * z) = β * y * w * v / (α^2 * x * z) := 
by
  sorry

end NUMINAMATH_GPT_milk_production_l2222_222279


namespace NUMINAMATH_GPT_not_perfect_square_7_301_l2222_222263

theorem not_perfect_square_7_301 :
  ¬ ∃ x : ℝ, x^2 = 7^301 := sorry

end NUMINAMATH_GPT_not_perfect_square_7_301_l2222_222263


namespace NUMINAMATH_GPT_strictly_increasing_and_symmetric_l2222_222205

open Real

noncomputable def f1 (x : ℝ) : ℝ := x^((1 : ℝ)/2)
noncomputable def f2 (x : ℝ) : ℝ := x^((1 : ℝ)/3)
noncomputable def f3 (x : ℝ) : ℝ := x^((2 : ℝ)/3)
noncomputable def f4 (x : ℝ) : ℝ := x^(-(1 : ℝ)/3)

theorem strictly_increasing_and_symmetric : 
  ∀ f : ℝ → ℝ,
  (f = f2) ↔ 
  ((∀ x : ℝ, 0 < x → f x = x^((1 : ℝ)/3) ∧ f (-x) = -(f x)) ∧ 
   (∀ x y : ℝ, 0 < x ∧ 0 < y → (x < y → f x < f y))) :=
sorry

end NUMINAMATH_GPT_strictly_increasing_and_symmetric_l2222_222205


namespace NUMINAMATH_GPT_next_divisor_of_4_digit_even_number_l2222_222241

theorem next_divisor_of_4_digit_even_number (n : ℕ) (h1 : 1000 ≤ n ∧ n < 10000)
  (h2 : n % 2 = 0) (hDiv : n % 221 = 0) :
  ∃ d, d > 221 ∧ d < n ∧ d % 13 = 0 ∧ d % 17 = 0 ∧ d = 442 :=
by
  use 442
  sorry

end NUMINAMATH_GPT_next_divisor_of_4_digit_even_number_l2222_222241


namespace NUMINAMATH_GPT_expression_value_l2222_222228

theorem expression_value (x : ℝ) (hx1 : x ≠ -1) (hx2 : x ≠ 2) :
  (2 * x ^ 2 - x) / ((x + 1) * (x - 2)) - (4 + x) / ((x + 1) * (x - 2)) = 2 := 
by
  sorry

end NUMINAMATH_GPT_expression_value_l2222_222228


namespace NUMINAMATH_GPT_sequence_6th_term_sequence_1994th_term_l2222_222265

def sequence_term (n : Nat) : Nat := n * (n + 1)

theorem sequence_6th_term:
  sequence_term 6 = 42 :=
by
  -- proof initially skipped
  sorry

theorem sequence_1994th_term:
  sequence_term 1994 = 3978030 :=
by
  -- proof initially skipped
  sorry

end NUMINAMATH_GPT_sequence_6th_term_sequence_1994th_term_l2222_222265


namespace NUMINAMATH_GPT_total_distance_of_the_race_l2222_222219

-- Define the given conditions
def A_beats_B_by_56_meters_or_7_seconds : Prop :=
  ∃ D : ℕ, ∀ S_B S_A : ℕ, S_B = 8 ∧ S_A = D / 8 ∧ D = S_B * (8 + 7)

-- Define the question and correct answer
theorem total_distance_of_the_race : A_beats_B_by_56_meters_or_7_seconds → ∃ D : ℕ, D = 120 :=
by
  sorry

end NUMINAMATH_GPT_total_distance_of_the_race_l2222_222219


namespace NUMINAMATH_GPT_solve_expression_l2222_222249

theorem solve_expression : 68 + (108 * 3) + (29^2) - 310 - (6 * 9) = 869 :=
by
  sorry

end NUMINAMATH_GPT_solve_expression_l2222_222249


namespace NUMINAMATH_GPT_complement_of_A_in_U_l2222_222288

noncomputable def U : Set ℝ := {x | (x - 2) / x ≤ 1}

noncomputable def A : Set ℝ := {x | 2 - x ≤ 1}

theorem complement_of_A_in_U :
  (U \ A) = {x | 0 < x ∧ x < 1} :=
by
  sorry

end NUMINAMATH_GPT_complement_of_A_in_U_l2222_222288


namespace NUMINAMATH_GPT_solve_for_k_l2222_222257

theorem solve_for_k (x k : ℝ) (h : x = -3) (h_eq : k * (x + 4) - 2 * k - x = 5) : k = -2 :=
by sorry

end NUMINAMATH_GPT_solve_for_k_l2222_222257


namespace NUMINAMATH_GPT_digit_is_9_if_divisible_by_11_l2222_222248

theorem digit_is_9_if_divisible_by_11 (d : ℕ) : 
  (678000 + 9000 + 800 + 90 + d) % 11 = 0 -> d = 9 := by
  sorry

end NUMINAMATH_GPT_digit_is_9_if_divisible_by_11_l2222_222248


namespace NUMINAMATH_GPT_lucas_journey_distance_l2222_222275

noncomputable def distance (D : ℝ) : ℝ :=
  let usual_speed := D / 150
  let distance_before_traffic := 2 * D / 5
  let speed_after_traffic := usual_speed - 1 / 2
  let time_before_traffic := distance_before_traffic / usual_speed
  let time_after_traffic := (3 * D / 5) / speed_after_traffic
  time_before_traffic + time_after_traffic

theorem lucas_journey_distance : ∃ D : ℝ, distance D = 255 ∧ D = 48.75 :=
sorry

end NUMINAMATH_GPT_lucas_journey_distance_l2222_222275


namespace NUMINAMATH_GPT_part1_l2222_222293

theorem part1 (a b : ℝ) (h1 : a ≠ b) (h2 : a^2 ≠ b^2) :
  (a^2 + a * b + b^2) / (a + b) - (a^2 - a * b + b^2) / (a - b) + (2 * b^2 - b^2 + a^2) / (a^2 - b^2) = 1 := 
sorry

end NUMINAMATH_GPT_part1_l2222_222293


namespace NUMINAMATH_GPT_largest_divisor_of_n_squared_divisible_by_72_l2222_222262

theorem largest_divisor_of_n_squared_divisible_by_72
    (n : ℕ) (h1 : n > 0) (h2 : 72 ∣ n^2) : 12 ∣ n :=
by {
    sorry
}

end NUMINAMATH_GPT_largest_divisor_of_n_squared_divisible_by_72_l2222_222262


namespace NUMINAMATH_GPT_window_ratio_area_l2222_222289

/-- Given a rectangle with semicircles at either end, if the ratio of AD to AB is 3:2,
    and AB is 30 inches, then the ratio of the area of the rectangle to the combined 
    area of the semicircles is 6 : π. -/
theorem window_ratio_area (AD AB r : ℝ) (h1 : AB = 30) (h2 : AD / AB = 3 / 2) (h3 : r = AB / 2) :
    (AD * AB) / (π * r^2) = 6 / π :=
by
  sorry

end NUMINAMATH_GPT_window_ratio_area_l2222_222289


namespace NUMINAMATH_GPT_no_combination_of_four_squares_equals_100_no_repeat_order_irrelevant_l2222_222222

theorem no_combination_of_four_squares_equals_100_no_repeat_order_irrelevant :
    ∀ (a b c d : ℕ), (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) →
                     (0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) →
                     (a^2 + b^2 + c^2 + d^2 = 100) → False := by
  sorry

end NUMINAMATH_GPT_no_combination_of_four_squares_equals_100_no_repeat_order_irrelevant_l2222_222222


namespace NUMINAMATH_GPT_units_digit_of_ksq_plus_2k_l2222_222272

def k := 2023^3 - 3^2023

theorem units_digit_of_ksq_plus_2k : (k^2 + 2^k) % 10 = 1 := 
  sorry

end NUMINAMATH_GPT_units_digit_of_ksq_plus_2k_l2222_222272


namespace NUMINAMATH_GPT_train_length_l2222_222274

theorem train_length (speed_kmph : ℝ) (time_sec : ℝ) (speed_ms : ℝ) (length_m : ℝ)
  (h1 : speed_kmph = 120) 
  (h2 : time_sec = 6)
  (h3 : speed_ms = 33.33)
  (h4 : length_m = 200) : 
  speed_kmph * 1000 / 3600 * time_sec = length_m :=
by
  sorry

end NUMINAMATH_GPT_train_length_l2222_222274


namespace NUMINAMATH_GPT_cube_side_length_l2222_222227

-- Given conditions for the problem
def surface_area (a : ℝ) : ℝ := 6 * a^2

-- Theorem statement
theorem cube_side_length (h : surface_area a = 864) : a = 12 :=
by
  sorry

end NUMINAMATH_GPT_cube_side_length_l2222_222227


namespace NUMINAMATH_GPT_expression_value_l2222_222282

theorem expression_value (a : ℝ) (h_nonzero : a ≠ 0) (h_ne_two : a ≠ 2) (h_ne_neg_two : a ≠ -2) (h_ne_neg_one : a ≠ -1) (h_eq_one : a = 1) :
  1 - (((a-2)/a) / ((a^2-4)/(a^2+a))) = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_expression_value_l2222_222282


namespace NUMINAMATH_GPT_probability_female_wears_glasses_l2222_222223

def prob_female_wears_glasses (total_females : ℕ) (females_no_glasses : ℕ) : ℚ :=
  let females_with_glasses := total_females - females_no_glasses
  females_with_glasses / total_females

theorem probability_female_wears_glasses :
  prob_female_wears_glasses 18 8 = 5 / 9 := by
  sorry  -- Proof is skipped

end NUMINAMATH_GPT_probability_female_wears_glasses_l2222_222223


namespace NUMINAMATH_GPT_find_symmetric_L_like_shape_l2222_222256

-- Define the L-like shape and its mirror image
def L_like_shape : Type := sorry  -- Placeholder for the actual geometry definition
def mirrored_L_like_shape : Type := sorry  -- Placeholder for the actual mirrored shape

-- Condition: The vertical symmetry function
def symmetric_about_vertical_line (shape1 shape2 : Type) : Prop :=
   sorry  -- Define what it means for shape1 to be symmetric to shape2

-- Given conditions (A to E as L-like shape variations)
def option_A : Type := sorry  -- An inverted L-like shape
def option_B : Type := sorry  -- An upside-down T-like shape
def option_C : Type := mirrored_L_like_shape  -- A mirrored L-like shape
def option_D : Type := sorry  -- A rotated L-like shape by 180 degrees
def option_E : Type := L_like_shape  -- An unchanged L-like shape

-- The theorem statement
theorem find_symmetric_L_like_shape :
  symmetric_about_vertical_line L_like_shape option_C :=
  sorry

end NUMINAMATH_GPT_find_symmetric_L_like_shape_l2222_222256


namespace NUMINAMATH_GPT_time_to_cover_length_l2222_222202

/-- Constants -/
def speed_escalator : ℝ := 10
def length_escalator : ℝ := 112
def speed_person : ℝ := 4

/-- Proof problem -/
theorem time_to_cover_length :
  (length_escalator / (speed_escalator + speed_person)) = 8 := by
  sorry

end NUMINAMATH_GPT_time_to_cover_length_l2222_222202


namespace NUMINAMATH_GPT_painters_completing_rooms_l2222_222209

theorem painters_completing_rooms (three_painters_three_rooms_three_hours : 3 * 3 * 3 ≥ 3 * 3) :
  9 * 3 * 9 ≥ 9 * 27 :=
by 
  sorry

end NUMINAMATH_GPT_painters_completing_rooms_l2222_222209


namespace NUMINAMATH_GPT_initial_amount_liquid_A_l2222_222276

theorem initial_amount_liquid_A (A B : ℝ) (h1 : A / B = 4)
    (h2 : (A / (B + 40)) = 2 / 3) : A = 32 := by
  sorry

end NUMINAMATH_GPT_initial_amount_liquid_A_l2222_222276


namespace NUMINAMATH_GPT_sets_equal_l2222_222280

theorem sets_equal (M N : Set ℝ) (hM : M = { x | x^2 = 1 }) (hN : N = { a | ∀ x ∈ M, a * x = 1 }) : M = N :=
sorry

end NUMINAMATH_GPT_sets_equal_l2222_222280


namespace NUMINAMATH_GPT_metal_rods_per_sheet_l2222_222234

theorem metal_rods_per_sheet :
  (∀ (metal_rod_for_sheets metal_rod_for_beams total_metal_rod num_sheet_per_panel num_panel num_rod_per_beam),
    (num_rod_per_beam = 4) →
    (total_metal_rod = 380) →
    (metal_rod_for_beams = num_panel * (2 * num_rod_per_beam)) →
    (metal_rod_for_sheets = total_metal_rod - metal_rod_for_beams) →
    (num_sheet_per_panel = 3) →
    (num_panel = 10) →
    (metal_rod_per_sheet = metal_rod_for_sheets / (num_panel * num_sheet_per_panel)) →
    metal_rod_per_sheet = 10
  ) := sorry

end NUMINAMATH_GPT_metal_rods_per_sheet_l2222_222234


namespace NUMINAMATH_GPT_simplify_expression_l2222_222211

theorem simplify_expression (y : ℝ) : 
  3 * y - 5 * y ^ 2 + 12 - (7 - 3 * y + 5 * y ^ 2) = -10 * y ^ 2 + 6 * y + 5 :=
by 
  sorry

end NUMINAMATH_GPT_simplify_expression_l2222_222211


namespace NUMINAMATH_GPT_dice_probability_exactly_four_twos_l2222_222235

theorem dice_probability_exactly_four_twos :
  let probability := (Nat.choose 8 4 : ℚ) * (1 / 8)^4 * (7 / 8)^4 
  probability = 168070 / 16777216 :=
by
  sorry

end NUMINAMATH_GPT_dice_probability_exactly_four_twos_l2222_222235


namespace NUMINAMATH_GPT_evaluate_expression_l2222_222232

theorem evaluate_expression :
  ((3^1 - 2 + 7^3 + 1 : ℚ)⁻¹ * 6) = (2 / 115) := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2222_222232


namespace NUMINAMATH_GPT_cross_country_hours_l2222_222231

-- Definitions based on the conditions from part a)
def total_hours_required : ℕ := 1500
def hours_day_flying : ℕ := 50
def hours_night_flying : ℕ := 9
def goal_months : ℕ := 6
def hours_per_month : ℕ := 220

-- Problem statement: prove she has already completed 1261 hours of cross-country flying
theorem cross_country_hours : 
  (goal_months * hours_per_month) - (hours_day_flying + hours_night_flying) = 1261 := 
by
  -- Proof omitted (using the solution steps)
  sorry

end NUMINAMATH_GPT_cross_country_hours_l2222_222231


namespace NUMINAMATH_GPT_distinct_sum_product_problem_l2222_222245

theorem distinct_sum_product_problem (S : ℤ) (hS : S ≥ 100) :
  ∃ a b c P : ℤ, a > b ∧ b > c ∧ a + b + c = S ∧ a * b * c = P ∧ 
    ¬(∀ x y z : ℤ, x > y ∧ y > z ∧ x + y + z = S → a = x ∧ b = y ∧ c = z) := 
sorry

end NUMINAMATH_GPT_distinct_sum_product_problem_l2222_222245


namespace NUMINAMATH_GPT_sin_3x_sin_x_solutions_l2222_222213

open Real

theorem sin_3x_sin_x_solutions :
  ∃ s : Finset ℝ, (∀ x ∈ s, 0 ≤ x ∧ x ≤ 2 * π ∧ sin (3 * x) = sin x) ∧ s.card = 7 := 
by sorry

end NUMINAMATH_GPT_sin_3x_sin_x_solutions_l2222_222213


namespace NUMINAMATH_GPT_shaded_region_area_l2222_222238

noncomputable def shaded_area (π_approx : ℝ := 3.14) (r : ℝ := 1) : ℝ :=
  let square_area := (r / Real.sqrt 2) ^ 2
  let quarter_circle_area := (π_approx * r ^ 2) / 4
  quarter_circle_area - square_area

theorem shaded_region_area :
  shaded_area = 0.285 :=
by
  sorry

end NUMINAMATH_GPT_shaded_region_area_l2222_222238


namespace NUMINAMATH_GPT_base8_minus_base7_base10_eq_l2222_222240

-- Definitions of the two numbers in their respective bases
def n1_base8 : ℕ := 305
def n2_base7 : ℕ := 165

-- Conversion of these numbers to base 10
def n1_base10 : ℕ := 3 * 8^2 + 0 * 8^1 + 5 * 8^0
def n2_base10 : ℕ := 1 * 7^2 + 6 * 7^1 + 5 * 7^0

-- Statement of the theorem to be proven
theorem base8_minus_base7_base10_eq :
  (n1_base10 - n2_base10 = 101) :=
  by
    -- The proof would go here
    sorry

end NUMINAMATH_GPT_base8_minus_base7_base10_eq_l2222_222240


namespace NUMINAMATH_GPT_boys_and_girls_equal_l2222_222290

theorem boys_and_girls_equal (m d M D : ℕ) (hm : m > 0) (hd : d > 0) (h1 : (M / m) ≠ (D / d)) (h2 : (M / m + D / d) / 2 = (M + D) / (m + d)) :
  m = d := 
sorry

end NUMINAMATH_GPT_boys_and_girls_equal_l2222_222290


namespace NUMINAMATH_GPT_octagon_properties_l2222_222215

-- Definitions for a regular octagon inscribed in a circle
def regular_octagon (r : ℝ) := ∀ (a b : ℝ), abs (a - b) = r
def side_length := 5
def inscribed_in_circle (r : ℝ) := ∃ (a b : ℝ), a * a + b * b = r * r

-- Main theorem statement
theorem octagon_properties (r : ℝ) (h : r = side_length) (h1 : regular_octagon r) (h2 : inscribed_in_circle r) :
  let arc_length := (5 * π) / 4
  let area_sector := (25 * π) / 8
  arc_length = (5 * π) / 4 ∧ area_sector = (25 * π) / 8 := by
  sorry

end NUMINAMATH_GPT_octagon_properties_l2222_222215


namespace NUMINAMATH_GPT_sin_75_equals_sqrt_1_plus_sin_2_equals_l2222_222225

noncomputable def sin_75 : ℝ := Real.sin (75 * Real.pi / 180)
noncomputable def sqrt_1_plus_sin_2 : ℝ := Real.sqrt (1 + Real.sin 2)

theorem sin_75_equals :
  sin_75 = (Real.sqrt 2 + Real.sqrt 6) / 4 := 
sorry

theorem sqrt_1_plus_sin_2_equals :
  sqrt_1_plus_sin_2 = Real.sin 1 + Real.cos 1 := 
sorry

end NUMINAMATH_GPT_sin_75_equals_sqrt_1_plus_sin_2_equals_l2222_222225


namespace NUMINAMATH_GPT_find_a_values_l2222_222216

noncomputable def system_has_exactly_three_solutions (a : ℝ) : Prop :=
  ∃ (x y : ℝ), 
    ((|y - 4| + |x + 12| - 3) * (x^2 + y^2 - 12) = 0) ∧ 
    ((x + 5)^2 + (y - 4)^2 = a)

theorem find_a_values : system_has_exactly_three_solutions 16 ∨ 
                        system_has_exactly_three_solutions (41 + 4 * Real.sqrt 123) :=
  by sorry

end NUMINAMATH_GPT_find_a_values_l2222_222216


namespace NUMINAMATH_GPT_discs_contain_equal_minutes_l2222_222264

theorem discs_contain_equal_minutes (total_time discs_capacity : ℕ) 
  (h1 : total_time = 520) (h2 : discs_capacity = 65) :
  ∃ discs_needed : ℕ, discs_needed = total_time / discs_capacity ∧ 
  ∀ (k : ℕ), k = total_time / discs_needed → k = 65 :=
by
  sorry

end NUMINAMATH_GPT_discs_contain_equal_minutes_l2222_222264


namespace NUMINAMATH_GPT_line_passes_fixed_point_l2222_222284

open Real

theorem line_passes_fixed_point
  (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : b < a)
  (M N : ℝ × ℝ)
  (hM : M.1^2 / a^2 + M.2^2 / b^2 = 1)
  (hN : N.1^2 / a^2 + N.2^2 / b^2 = 1)
  (hMAhNA : (M.1 + a) * (N.1 + a) + M.2 * N.2 = 0):
  ∃ (P : ℝ × ℝ), P = (a * (b^2 - a^2) / (a^2 + b^2), 0) ∧ (N.2 - M.2) * (P.1 - M.1) = (P.2 - M.2) * (N.1 - M.1) :=
sorry

end NUMINAMATH_GPT_line_passes_fixed_point_l2222_222284


namespace NUMINAMATH_GPT_harry_terry_difference_l2222_222252

theorem harry_terry_difference :
  let H := 12 - (3 + 6)
  let T := 12 - 3 + 6 * 2
  H - T = -18 :=
by
  sorry

end NUMINAMATH_GPT_harry_terry_difference_l2222_222252


namespace NUMINAMATH_GPT_tom_teaching_years_l2222_222233

def years_tom_has_been_teaching (x : ℝ) : Prop :=
  x + (1/2 * x - 5) = 70

theorem tom_teaching_years:
  ∃ x : ℝ, years_tom_has_been_teaching x ∧ x = 50 :=
by
  sorry

end NUMINAMATH_GPT_tom_teaching_years_l2222_222233
