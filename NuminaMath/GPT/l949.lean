import Mathlib

namespace NUMINAMATH_GPT_seeds_in_big_garden_is_correct_l949_94962

def total_seeds : ℕ := 41
def small_gardens : ℕ := 3
def seeds_per_small_garden : ℕ := 4

def seeds_in_small_gardens : ℕ := small_gardens * seeds_per_small_garden
def seeds_in_big_garden : ℕ := total_seeds - seeds_in_small_gardens

theorem seeds_in_big_garden_is_correct : seeds_in_big_garden = 29 := by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_seeds_in_big_garden_is_correct_l949_94962


namespace NUMINAMATH_GPT_directrix_of_parabola_l949_94925

open Real

noncomputable def parabola_directrix (a : ℝ) : ℝ := -a / 4

theorem directrix_of_parabola (a : ℝ) (h : a = 4) : parabola_directrix a = -4 :=
by
  sorry

end NUMINAMATH_GPT_directrix_of_parabola_l949_94925


namespace NUMINAMATH_GPT_domain_of_k_l949_94951

noncomputable def domain_of_h := Set.Icc (-10 : ℝ) 6

def h (x : ℝ) : Prop := x ∈ domain_of_h
def k (x : ℝ) : Prop := h (-3 * x + 1)

theorem domain_of_k : ∀ x : ℝ, k x ↔ x ∈ Set.Icc (-5/3) (11/3) :=
by
  intro x
  change (-3 * x + 1 ∈ Set.Icc (-10 : ℝ) 6) ↔ (x ∈ Set.Icc (-5/3 : ℝ) (11/3))
  sorry

end NUMINAMATH_GPT_domain_of_k_l949_94951


namespace NUMINAMATH_GPT_discount_offered_is_5_percent_l949_94901

noncomputable def cost_price : ℝ := 100

noncomputable def selling_price_with_discount : ℝ := cost_price * 1.216

noncomputable def selling_price_without_discount : ℝ := cost_price * 1.28

noncomputable def discount : ℝ := selling_price_without_discount - selling_price_with_discount

noncomputable def discount_percentage : ℝ := (discount / selling_price_without_discount) * 100

theorem discount_offered_is_5_percent : discount_percentage = 5 :=
by 
  sorry

end NUMINAMATH_GPT_discount_offered_is_5_percent_l949_94901


namespace NUMINAMATH_GPT_fraction_not_covered_correct_l949_94960

def area_floor : ℕ := 64
def width_rug : ℕ := 2
def length_rug : ℕ := 7
def area_rug := width_rug * length_rug
def area_not_covered := area_floor - area_rug
def fraction_not_covered := (area_not_covered : ℚ) / area_floor

theorem fraction_not_covered_correct :
  fraction_not_covered = 25 / 32 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_fraction_not_covered_correct_l949_94960


namespace NUMINAMATH_GPT_suitable_for_experimental_method_is_meters_run_l949_94914

-- Define the options as a type
inductive ExperimentalOption
| recommending_class_monitor_candidates
| surveying_classmates_birthdays
| meters_run_in_10_seconds
| avian_influenza_occurrences_world

-- Define a function that checks if an option is suitable for the experimental method
def is_suitable_for_experimental_method (option: ExperimentalOption) : Prop :=
  option = ExperimentalOption.meters_run_in_10_seconds

-- The theorem stating which option is suitable for the experimental method
theorem suitable_for_experimental_method_is_meters_run :
  is_suitable_for_experimental_method ExperimentalOption.meters_run_in_10_seconds :=
by
  sorry

end NUMINAMATH_GPT_suitable_for_experimental_method_is_meters_run_l949_94914


namespace NUMINAMATH_GPT_repeating_decimals_for_n_div_18_l949_94998

theorem repeating_decimals_for_n_div_18 :
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 20 → (¬ (∃ m : ℕ, m * 18 = n * (2^k * 5^l) ∧ 0 < k ∧ 0 < l)) :=
by
  sorry

end NUMINAMATH_GPT_repeating_decimals_for_n_div_18_l949_94998


namespace NUMINAMATH_GPT_arithmetic_expression_value_l949_94947

theorem arithmetic_expression_value : 4 * (8 - 3 + 2) - 7 = 21 := by
  sorry

end NUMINAMATH_GPT_arithmetic_expression_value_l949_94947


namespace NUMINAMATH_GPT_find_x_l949_94997

noncomputable def log_base_2 (x : ℝ) : ℝ := Real.log x / Real.log 2
noncomputable def log_base_3 (x : ℝ) : ℝ := Real.log x / Real.log 3
noncomputable def log_base_5 (x : ℝ) : ℝ := Real.log x / Real.log 5
noncomputable def log_base_4 (x : ℝ) : ℝ := Real.log x / Real.log 4

noncomputable def right_triangular_prism_area (x : ℝ) : ℝ := 
  let a := log_base_2 x
  let b := log_base_3 x
  let c := log_base_5 x
  (1/2) * a * b + 3 * c * (a + b + c)

noncomputable def right_triangular_prism_volume (x : ℝ) : ℝ := 
  let a := log_base_2 x
  let b := log_base_3 x
  let c := log_base_5 x
  (1/2) * a * b * c

noncomputable def rectangular_prism_area (x : ℝ) : ℝ := 
  let a := log_base_2 x
  let b := log_base_4 x
  2 * (a * b + a * a + b * a)

noncomputable def rectangular_prism_volume (x : ℝ) : ℝ := 
  let a := log_base_2 x
  let b := log_base_4 x
  a * b * a

theorem find_x (x : ℝ) (h : right_triangular_prism_area x + rectangular_prism_area x = right_triangular_prism_volume x + rectangular_prism_volume x) :
  x = 1152 := by
sorry

end NUMINAMATH_GPT_find_x_l949_94997


namespace NUMINAMATH_GPT_new_trailers_added_l949_94935

theorem new_trailers_added :
  let initial_trailers := 25
  let initial_average_age := 15
  let years_passed := 3
  let current_average_age := 12
  let total_initial_age := initial_trailers * (initial_average_age + years_passed)
  ∀ n : Nat, 
    ((25 * 18) + (n * 3) = (25 + n) * 12) →
    n = 17 := 
by
  intros
  sorry

end NUMINAMATH_GPT_new_trailers_added_l949_94935


namespace NUMINAMATH_GPT_cats_joined_l949_94939

theorem cats_joined (c : ℕ) (h : 1 + c + 2 * c + 6 * c = 37) : c = 4 :=
sorry

end NUMINAMATH_GPT_cats_joined_l949_94939


namespace NUMINAMATH_GPT_branches_and_ornaments_l949_94909

def numberOfBranchesAndOrnaments (b t : ℕ) : Prop :=
  (b = t - 1) ∧ (2 * b = t - 1)

theorem branches_and_ornaments : ∃ (b t : ℕ), numberOfBranchesAndOrnaments b t ∧ b = 3 ∧ t = 4 :=
by
  sorry

end NUMINAMATH_GPT_branches_and_ornaments_l949_94909


namespace NUMINAMATH_GPT_max_sum_of_abcd_l949_94956

noncomputable def abcd_product (a b c d : ℕ) : ℕ := a * b * c * d

theorem max_sum_of_abcd (a b c d : ℕ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d)
    (h4 : b ≠ c) (h5 : b ≠ d) (h6 : c ≠ d) (h7 : abcd_product a b c d = 1995) : 
    a + b + c + d ≤ 142 :=
sorry

end NUMINAMATH_GPT_max_sum_of_abcd_l949_94956


namespace NUMINAMATH_GPT_sets_are_equal_l949_94948

def X : Set ℝ := {x | ∃ n : ℤ, x = (2 * n + 1) * Real.pi}
def Y : Set ℝ := {y | ∃ k : ℤ, y = (4 * k + 1) * Real.pi ∨ y = (4 * k - 1) * Real.pi}

theorem sets_are_equal : X = Y :=
by sorry

end NUMINAMATH_GPT_sets_are_equal_l949_94948


namespace NUMINAMATH_GPT_Gerald_charge_per_chore_l949_94912

noncomputable def charge_per_chore (E SE SP C : ℕ) : ℕ :=
  let total_expenditure := E * SE
  let monthly_saving_goal := total_expenditure / SP
  monthly_saving_goal / C

theorem Gerald_charge_per_chore :
  charge_per_chore 100 4 8 5 = 10 :=
by
  sorry

end NUMINAMATH_GPT_Gerald_charge_per_chore_l949_94912


namespace NUMINAMATH_GPT_odd_function_iff_l949_94918

variable {α : Type*} [LinearOrderedField α]

noncomputable def f (a b x : α) : α := x * abs (x + a) + b

theorem odd_function_iff (a b : α) : 
  (∀ x : α, f a b (-x) = -f a b x) ↔ (a^2 + b^2 = 0) :=
by
  sorry

end NUMINAMATH_GPT_odd_function_iff_l949_94918


namespace NUMINAMATH_GPT_total_repair_cost_l949_94967

theorem total_repair_cost :
  let rate1 := 60
  let hours1 := 8
  let days1 := 14
  let rate2 := 75
  let hours2 := 6
  let days2 := 10
  let parts_cost := 3200
  let first_mechanic_cost := rate1 * hours1 * days1
  let second_mechanic_cost := rate2 * hours2 * days2
  let total_cost := first_mechanic_cost + second_mechanic_cost + parts_cost
  total_cost = 14420 := by
  sorry

end NUMINAMATH_GPT_total_repair_cost_l949_94967


namespace NUMINAMATH_GPT_price_decrease_is_50_percent_l949_94924

-- Original price is 50 yuan
def original_price : ℝ := 50

-- Price after 100% increase
def increased_price : ℝ := original_price * (1 + 1)

-- Required percentage decrease to return to original price
def required_percentage_decrease (x : ℝ) : ℝ := increased_price * (1 - x)

theorem price_decrease_is_50_percent : required_percentage_decrease 0.5 = 50 :=
  by 
    sorry

end NUMINAMATH_GPT_price_decrease_is_50_percent_l949_94924


namespace NUMINAMATH_GPT_jenny_questions_wrong_l949_94903

variable (j k l m : ℕ)

theorem jenny_questions_wrong
  (h1 : j + k = l + m)
  (h2 : j + m = k + l + 6)
  (h3 : l = 7) : j = 10 := by
  sorry

end NUMINAMATH_GPT_jenny_questions_wrong_l949_94903


namespace NUMINAMATH_GPT_find_x_l949_94900

theorem find_x (x : ℝ) (h : 0.40 * x = (1/3) * x + 110) : x = 1650 :=
sorry

end NUMINAMATH_GPT_find_x_l949_94900


namespace NUMINAMATH_GPT_max_f_value_l949_94989

noncomputable def f (x : ℝ) : ℝ := min (3 * x + 1) (min (- (4 / 3) * x + 3) ((1 / 3) * x + 9))

theorem max_f_value : ∃ x : ℝ, f x = 31 / 13 :=
by 
  sorry

end NUMINAMATH_GPT_max_f_value_l949_94989


namespace NUMINAMATH_GPT_cost_of_flowers_l949_94930

theorem cost_of_flowers 
  (interval : ℕ) (perimeter : ℕ) (cost_per_flower : ℕ)
  (h_interval : interval = 30)
  (h_perimeter : perimeter = 1500)
  (h_cost : cost_per_flower = 5000) :
  (perimeter / interval) * cost_per_flower = 250000 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_flowers_l949_94930


namespace NUMINAMATH_GPT_people_in_room_eq_33_l949_94928

variable (people chairs : ℕ)

def chairs_empty := 5
def chairs_total := 5 * 5
def chairs_occupied := (4 * chairs_total) / 5
def people_seated := 3 * people / 5

theorem people_in_room_eq_33 : 
    (people_seated = chairs_occupied ∧ chairs_total - chairs_occupied = chairs_empty)
    → people = 33 :=
by
  sorry

end NUMINAMATH_GPT_people_in_room_eq_33_l949_94928


namespace NUMINAMATH_GPT_trajectory_of_P_distance_EF_l949_94952

section Exercise

-- Define the curve C in polar coordinates
def curve_C (ρ' θ: ℝ) : Prop :=
  ρ' * Real.cos (θ + Real.pi / 4) = 1

-- Define the relationship between OP and OQ
def product_OP_OQ (ρ ρ' : ℝ) : Prop :=
  ρ * ρ' = Real.sqrt 2

-- Define the trajectory of point P (C1) as the goal
theorem trajectory_of_P (ρ θ: ℝ) (hC: curve_C ρ' θ) (hPQ: product_OP_OQ ρ ρ') :
  ρ = Real.cos θ - Real.sin θ :=
sorry

-- Define the coordinates and the curve C2
def curve_C2 (x y t: ℝ) : Prop :=
  x = 0.5 - Real.sqrt 2 / 2 * t ∧ y = Real.sqrt 2 / 2 * t

-- Define the line l in Cartesian coordinates that needs to be converted to polar
def line_l (x y: ℝ) : Prop :=
  y = -Real.sqrt 3 * x

-- Define the distance |EF| to be proved
theorem distance_EF (θ ρ_1 ρ_2: ℝ) (hx: curve_C2 (0.5 - Real.sqrt 2 / 2 * t) (Real.sqrt 2 / 2 * t) t)
  (hE: θ = 2 * Real.pi / 3 ∨ θ = -Real.pi / 3)
  (hρ1: ρ_1 = Real.cos (-Real.pi / 3) - Real.sin (-Real.pi / 3))
  (hρ2: ρ_2 = 0.5 * (Real.sqrt 3 + 1)) :
  |ρ_1 + ρ_2| = Real.sqrt 3 + 1 :=
sorry

end Exercise

end NUMINAMATH_GPT_trajectory_of_P_distance_EF_l949_94952


namespace NUMINAMATH_GPT_cylinder_surface_area_l949_94990

theorem cylinder_surface_area (h : ℝ) (r : ℝ) (h_eq : h = 12) (r_eq : r = 4) : 
  2 * π * r * (r + h) = 128 * π := 
by
  rw [h_eq, r_eq]
  sorry

end NUMINAMATH_GPT_cylinder_surface_area_l949_94990


namespace NUMINAMATH_GPT_simplify_fraction_l949_94902

theorem simplify_fraction (x : ℝ) (h : x ≠ -1) : (x + 1) / (x^2 + 2 * x + 1) = 1 / (x + 1) :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l949_94902


namespace NUMINAMATH_GPT_sum_of_squares_neq_fourth_powers_l949_94906

theorem sum_of_squares_neq_fourth_powers (m n : ℕ) : 
  m^2 + (m + 1)^2 ≠ n^4 + (n + 1)^4 :=
by 
  sorry

end NUMINAMATH_GPT_sum_of_squares_neq_fourth_powers_l949_94906


namespace NUMINAMATH_GPT_third_divisor_l949_94987

theorem third_divisor (x : ℕ) (h1 : x - 16 = 136) (h2 : ∃ y, y = x - 16) (h3 : 4 ∣ x) (h4 : 6 ∣ x) (h5 : 10 ∣ x) : 19 ∣ x := 
by
  sorry

end NUMINAMATH_GPT_third_divisor_l949_94987


namespace NUMINAMATH_GPT_max_min_values_l949_94977

noncomputable def f (x : ℝ) : ℝ := x^3 - 12 * x + 8

theorem max_min_values :
  ∃ x_max x_min : ℝ, x_max ∈ Set.Icc (-3 : ℝ) 3 ∧ x_min ∈ Set.Icc (-3 : ℝ) 3 ∧
    (∀ x ∈ Set.Icc (-3 : ℝ) 3, f x ≤ f x_max) ∧ (∀ x ∈ Set.Icc (-3 : ℝ) 3, f x_min ≤ f x) ∧
    f (-2) = 24 ∧ f 2 = -6 := sorry

end NUMINAMATH_GPT_max_min_values_l949_94977


namespace NUMINAMATH_GPT_distance_to_grandma_l949_94941

-- Definitions based on the conditions
def miles_per_gallon : ℕ := 20
def gallons_needed : ℕ := 5

-- The theorem statement to prove the distance is 100 miles
theorem distance_to_grandma : miles_per_gallon * gallons_needed = 100 := by
  sorry

end NUMINAMATH_GPT_distance_to_grandma_l949_94941


namespace NUMINAMATH_GPT_vector_addition_AC_l949_94982

def vector := (ℝ × ℝ)

def AB : vector := (0, 1)
def BC : vector := (1, 0)

def AC (AB BC : vector) : vector := (AB.1 + BC.1, AB.2 + BC.2) 

theorem vector_addition_AC (AB BC : vector) (h1 : AB = (0, 1)) (h2 : BC = (1, 0)) : 
  AC AB BC = (1, 1) :=
by
  sorry

end NUMINAMATH_GPT_vector_addition_AC_l949_94982


namespace NUMINAMATH_GPT_shaded_area_is_correct_l949_94971

-- Define the basic constants and areas
def grid_length : ℝ := 15
def grid_height : ℝ := 5
def total_grid_area : ℝ := grid_length * grid_height

def large_triangle_base : ℝ := 15
def large_triangle_height : ℝ := 3
def large_triangle_area : ℝ := 0.5 * large_triangle_base * large_triangle_height

def small_triangle_base : ℝ := 3
def small_triangle_height : ℝ := 4
def small_triangle_area : ℝ := 0.5 * small_triangle_base * small_triangle_height

-- Define the total shaded area
def shaded_area : ℝ := total_grid_area - large_triangle_area + small_triangle_area

-- Theorem stating that the shaded area is 58.5 square units
theorem shaded_area_is_correct : shaded_area = 58.5 := 
by 
  -- proof will be provided here
  sorry

end NUMINAMATH_GPT_shaded_area_is_correct_l949_94971


namespace NUMINAMATH_GPT_prob_first_red_light_third_intersection_l949_94955

noncomputable def red_light_at_third_intersection (p : ℝ) (h : p = 2/3) : ℝ :=
(1 - p) * (1 - (1/2)) * (1/2)

theorem prob_first_red_light_third_intersection (h : 2/3 = (2/3 : ℝ)) :
  red_light_at_third_intersection (2/3) h = 1/12 := sorry

end NUMINAMATH_GPT_prob_first_red_light_third_intersection_l949_94955


namespace NUMINAMATH_GPT_max_covered_squares_l949_94972

-- Definitions representing the conditions
def checkerboard_squares : ℕ := 1 -- side length of each square on the checkerboard
def card_side_len : ℕ := 2 -- side length of the card

-- Theorem statement representing the question and answer
theorem max_covered_squares : ∀ n, 
  (∃ board_side squared_len, 
    checkerboard_squares = 1 ∧ card_side_len = 2 ∧
    (board_side = checkerboard_squares ∧ squared_len = card_side_len) ∧
    n ≤ 16) →
  n = 16 :=
  sorry

end NUMINAMATH_GPT_max_covered_squares_l949_94972


namespace NUMINAMATH_GPT_verka_digit_sets_l949_94969

-- Define the main conditions as:
def is_three_digit_number (a b c : ℕ) : Prop :=
  let num1 := 100 * a + 10 * b + c
  let num2 := 100 * a + 10 * c + b
  let num3 := 100 * b + 10 * a + c
  let num4 := 100 * b + 10 * c + a
  let num5 := 100 * c + 10 * a + b
  let num6 := 100 * c + 10 * b + a
  num1 + num2 + num3 + num4 + num5 + num6 = 1221

-- Prove the main theorem
theorem verka_digit_sets :
  ∃ (a b c : ℕ), is_three_digit_number a a c ∧
                 ((a, c) = (1, 9) ∨ (a, c) = (2, 7) ∨ (a, c) = (3, 5) ∨ (a, c) = (4, 3) ∨ (a, c) = (5, 1)) :=
by sorry

end NUMINAMATH_GPT_verka_digit_sets_l949_94969


namespace NUMINAMATH_GPT_log_base2_probability_l949_94923

theorem log_base2_probability (n : ℕ) (h1 : 100 ≤ n ∧ n ≤ 999) (h2 : ∃ k : ℕ, n = 2^k) : 
  ∃ p : ℚ, p = 1/300 :=
  sorry

end NUMINAMATH_GPT_log_base2_probability_l949_94923


namespace NUMINAMATH_GPT_balls_in_box_l949_94957

def num_blue : Nat := 6
def num_red : Nat := 4
def num_green : Nat := 3 * num_blue
def num_yellow : Nat := 2 * num_red
def num_total : Nat := num_blue + num_red + num_green + num_yellow

theorem balls_in_box : num_total = 36 := by
  sorry

end NUMINAMATH_GPT_balls_in_box_l949_94957


namespace NUMINAMATH_GPT_initial_crayons_l949_94994

theorem initial_crayons {C : ℕ} (h : C + 12 = 53) : C = 41 :=
by
  -- This is where the proof would go, but we use sorry to skip it.
  sorry

end NUMINAMATH_GPT_initial_crayons_l949_94994


namespace NUMINAMATH_GPT_remainder_prod_mod_10_l949_94993

theorem remainder_prod_mod_10 :
  (2457 * 7963 * 92324) % 10 = 4 :=
  sorry

end NUMINAMATH_GPT_remainder_prod_mod_10_l949_94993


namespace NUMINAMATH_GPT_abcd_hife_value_l949_94953

theorem abcd_hife_value (a b c d e f g h i : ℝ) 
  (h1 : a / b = 1 / 3) 
  (h2 : b / c = 2) 
  (h3 : c / d = 1 / 2) 
  (h4 : d / e = 3) 
  (h5 : e / f = 1 / 10) 
  (h6 : f / g = 3 / 4) 
  (h7 : g / h = 1 / 5) 
  (h8 : h / i = 5) : 
  abcd / hife = 17.28 := sorry

end NUMINAMATH_GPT_abcd_hife_value_l949_94953


namespace NUMINAMATH_GPT_gcd_lcm_sum_8_12_l949_94991

-- Define the problem statement
theorem gcd_lcm_sum_8_12 : Nat.gcd 8 12 + Nat.lcm 8 12 = 28 := by
  sorry

end NUMINAMATH_GPT_gcd_lcm_sum_8_12_l949_94991


namespace NUMINAMATH_GPT_seq_formula_l949_94978

theorem seq_formula (a : ℕ → ℕ) (h₀ : a 1 = 2) (h₁ : ∀ n : ℕ, 0 < n → a (n + 1) = 2 * a n - 1) :
  ∀ n : ℕ, 0 < n → a n = 2 ^ (n - 1) + 1 := 
by 
  sorry

end NUMINAMATH_GPT_seq_formula_l949_94978


namespace NUMINAMATH_GPT_correct_multiplier_l949_94985

theorem correct_multiplier (x : ℕ) 
  (h1 : 137 * 34 + 1233 = 137 * x) : 
  x = 43 := 
by 
  sorry

end NUMINAMATH_GPT_correct_multiplier_l949_94985


namespace NUMINAMATH_GPT_sequence_difference_l949_94932

theorem sequence_difference :
  ∃ a : ℕ → ℕ,
    a 1 = 1 ∧ a 2 = 1 ∧
    (∀ n ≥ 1, (a (n + 2) : ℚ) / a (n + 1) - (a (n + 1) : ℚ) / a n = 1) ∧
    a 6 - a 5 = 96 :=
sorry

end NUMINAMATH_GPT_sequence_difference_l949_94932


namespace NUMINAMATH_GPT_value_of_k_l949_94958

theorem value_of_k (x y k : ℤ) (h1 : 2 * x - y = 5 * k + 6) (h2 : 4 * x + 7 * y = k) (h3 : x + y = 2024)
: k = 2023 :=
by
  sorry

end NUMINAMATH_GPT_value_of_k_l949_94958


namespace NUMINAMATH_GPT_min_value_of_expression_l949_94996

open Real

noncomputable def condition (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ (1 / (x + 2) + 1 / (y + 2) = 1 / 4)

theorem min_value_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1 / (x + 2) + 1 / (y + 2) = 1 / 4) :
  2 * x + 3 * y = 5 + 4 * sqrt 3 :=
sorry

end NUMINAMATH_GPT_min_value_of_expression_l949_94996


namespace NUMINAMATH_GPT_lattice_point_in_PQE_l949_94976

-- Define points and their integer coordinates
structure Point :=
  (x : ℤ)
  (y : ℤ)

-- Define a convex quadrilateral with integer coordinates
structure ConvexQuadrilateral :=
  (P : Point)
  (Q : Point)
  (R : Point)
  (S : Point)

-- Define the intersection point of diagonals as another point
def diagIntersection (quad: ConvexQuadrilateral) : Point := sorry

-- Define the condition for the sum of angles at P and Q being less than 180 degrees
def sumAnglesLessThan180 (quad : ConvexQuadrilateral) : Prop := sorry

-- Define a function to check if a point is a lattice point
def isLatticePoint (p : Point) : Prop := true  -- Since all points are lattice points by definition

-- Define the proof problem
theorem lattice_point_in_PQE (quad : ConvexQuadrilateral) (E : Point) :
  sumAnglesLessThan180 quad →
  ∃ p : Point, p ≠ quad.P ∧ p ≠ quad.Q ∧ isLatticePoint p ∧ sorry := sorry -- (prove the point is in PQE)

end NUMINAMATH_GPT_lattice_point_in_PQE_l949_94976


namespace NUMINAMATH_GPT_items_per_friend_l949_94944

theorem items_per_friend (pencils : ℕ) (erasers : ℕ) (friends : ℕ) 
    (pencils_eq : pencils = 35) 
    (erasers_eq : erasers = 5) 
    (friends_eq : friends = 5) : 
    (pencils + erasers) / friends = 8 := 
by
  sorry

end NUMINAMATH_GPT_items_per_friend_l949_94944


namespace NUMINAMATH_GPT_find_replacement_percentage_l949_94983

noncomputable def final_percentage_replacement_alcohol_solution (a₁ p₁ p₂ x : ℝ) : Prop :=
  let d := 0.4 -- gallons
  let final_solution := 1 -- gallon
  let initial_pure_alcohol := a₁ * p₁ / 100
  let remaining_pure_alcohol := initial_pure_alcohol - (d * p₁ / 100)
  let added_pure_alcohol := d * x / 100
  remaining_pure_alcohol + added_pure_alcohol = final_solution * p₂ / 100

theorem find_replacement_percentage :
  final_percentage_replacement_alcohol_solution 1 75 65 50 :=
by
  sorry

end NUMINAMATH_GPT_find_replacement_percentage_l949_94983


namespace NUMINAMATH_GPT_base_n_multiple_of_5_l949_94970

-- Define the polynomial f(n)
def f (n : ℕ) : ℕ := 4 + n + 3 * n^2 + 5 * n^3 + n^4 + 4 * n^5

-- The main theorem to be proven
theorem base_n_multiple_of_5 (n : ℕ) (h1 : 2 ≤ n) (h2 : n ≤ 100) : 
  f n % 5 ≠ 0 :=
by sorry

end NUMINAMATH_GPT_base_n_multiple_of_5_l949_94970


namespace NUMINAMATH_GPT_adult_ticket_cost_l949_94949

def num_total_tickets : ℕ := 510
def cost_senior_ticket : ℕ := 15
def total_receipts : ℤ := 8748
def num_senior_tickets : ℕ := 327
def num_adult_tickets : ℕ := num_total_tickets - num_senior_tickets
def revenue_senior : ℤ := num_senior_tickets * cost_senior_ticket
def revenue_adult (cost_adult_ticket : ℤ) : ℤ := num_adult_tickets * cost_adult_ticket

theorem adult_ticket_cost : 
  ∃ (cost_adult_ticket : ℤ), 
    revenue_adult cost_adult_ticket + revenue_senior = total_receipts ∧ 
    cost_adult_ticket = 21 :=
by
  sorry

end NUMINAMATH_GPT_adult_ticket_cost_l949_94949


namespace NUMINAMATH_GPT_max_sum_of_lengths_l949_94916

theorem max_sum_of_lengths (x y : ℕ) (hx : 1 < x) (hy : 1 < y) (hxy : x + 3 * y < 5000) :
  ∃ a b : ℕ, x = 2^a ∧ y = 2^b ∧ a + b = 20 := sorry

end NUMINAMATH_GPT_max_sum_of_lengths_l949_94916


namespace NUMINAMATH_GPT_weight_of_second_new_player_l949_94933

theorem weight_of_second_new_player 
  (total_weight_seven_players : ℕ)
  (average_weight_seven_players : ℕ)
  (total_players_with_new_players : ℕ)
  (average_weight_with_new_players : ℕ)
  (weight_first_new_player : ℕ)
  (W : ℕ) :
  total_weight_seven_players = 7 * average_weight_seven_players →
  total_players_with_new_players = 9 →
  average_weight_with_new_players = 106 →
  weight_first_new_player = 110 →
  (total_weight_seven_players + weight_first_new_player + W) / total_players_with_new_players = average_weight_with_new_players →
  W = 60 := 
by sorry

end NUMINAMATH_GPT_weight_of_second_new_player_l949_94933


namespace NUMINAMATH_GPT_total_vegetables_l949_94984

theorem total_vegetables (b k r : ℕ) (broccoli_weight_kg : ℝ) (broccoli_weight_g : ℝ) 
  (kohlrabi_mult : ℕ) (radish_mult : ℕ) :
  broccoli_weight_kg = 5 ∧ 
  broccoli_weight_g = 0.25 ∧ 
  kohlrabi_mult = 4 ∧ 
  radish_mult = 3 ∧ 
  b = broccoli_weight_kg / broccoli_weight_g ∧ 
  k = kohlrabi_mult * b ∧ 
  r = radish_mult * k →
  b + k + r = 340 := 
by
  sorry

end NUMINAMATH_GPT_total_vegetables_l949_94984


namespace NUMINAMATH_GPT_tesseract_hyper_volume_l949_94927

theorem tesseract_hyper_volume
  (a b c d : ℝ)
  (h1 : a * b * c = 72)
  (h2 : b * c * d = 75)
  (h3 : c * d * a = 48)
  (h4 : d * a * b = 50) :
  a * b * c * d = 3600 :=
sorry

end NUMINAMATH_GPT_tesseract_hyper_volume_l949_94927


namespace NUMINAMATH_GPT_geometric_seq_increasing_condition_not_sufficient_nor_necessary_l949_94979

-- Definitions based on conditions
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) := ∀ n : ℕ, a (n + 1) = q * a n
def monotonically_increasing (a : ℕ → ℝ) := ∀ n : ℕ, a n ≤ a (n + 1)
def common_ratio_gt_one (q : ℝ) := q > 1

-- Proof statement of the problem
theorem geometric_seq_increasing_condition_not_sufficient_nor_necessary 
    (a : ℕ → ℝ) (q : ℝ) 
    (h1 : geometric_sequence a q) : 
    ¬(common_ratio_gt_one q ↔ monotonically_increasing a) :=
sorry

end NUMINAMATH_GPT_geometric_seq_increasing_condition_not_sufficient_nor_necessary_l949_94979


namespace NUMINAMATH_GPT_total_pens_l949_94908

theorem total_pens (r : ℕ) (h1 : r > 10) (h2 : 357 % r = 0) (h3 : 441 % r = 0) :
  (357 / r) + (441 / r) = 38 :=
sorry

end NUMINAMATH_GPT_total_pens_l949_94908


namespace NUMINAMATH_GPT_mod11_residue_l949_94968

theorem mod11_residue :
  (305 % 11 = 8) →
  (44 % 11 = 0) →
  (176 % 11 = 0) →
  (18 % 11 = 7) →
  (305 + 7 * 44 + 9 * 176 + 6 * 18) % 11 = 6 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_mod11_residue_l949_94968


namespace NUMINAMATH_GPT_multiple_of_C_share_l949_94919

noncomputable def find_multiple (A B C : ℕ) (total : ℕ) (mult : ℕ) (h1 : 4 * A = mult * C) (h2 : 5 * B = mult * C) (h3 : A + B + C = total) : ℕ :=
  mult

theorem multiple_of_C_share (A B : ℕ) (h1 : 4 * A = 10 * 160) (h2 : 5 * B = 10 * 160) (h3 : A + B + 160 = 880) : find_multiple A B 160 880 10 h1 h2 h3 = 10 :=
by
  sorry

end NUMINAMATH_GPT_multiple_of_C_share_l949_94919


namespace NUMINAMATH_GPT_cost_difference_l949_94937

def cost_per_copy_X : ℝ := 1.25
def cost_per_copy_Y : ℝ := 2.75
def num_copies : ℕ := 80

theorem cost_difference :
  num_copies * cost_per_copy_Y - num_copies * cost_per_copy_X = 120 := sorry

end NUMINAMATH_GPT_cost_difference_l949_94937


namespace NUMINAMATH_GPT_xyz_equivalence_l949_94904

theorem xyz_equivalence (x y z a b : ℝ) (h₁ : 4^x = a) (h₂: 2^y = b) (h₃ : 8^z = a * b) : 3 * z = 2 * x + y :=
by
  -- Here, we leave the proof as an exercise
  sorry

end NUMINAMATH_GPT_xyz_equivalence_l949_94904


namespace NUMINAMATH_GPT_percentage_of_360_l949_94945

theorem percentage_of_360 (percentage : ℝ) : 
  (percentage / 100) * 360 = 93.6 → percentage = 26 := 
by
  intro h
  -- proof missing
  sorry

end NUMINAMATH_GPT_percentage_of_360_l949_94945


namespace NUMINAMATH_GPT_distinct_after_removal_l949_94954

variable (n : ℕ)
variable (subsets : Fin n → Finset (Fin n))

theorem distinct_after_removal :
  ∃ k : Fin n, ∀ i j : Fin n, i ≠ j → (subsets i \ {k}) ≠ (subsets j \ {k}) := by
  sorry

end NUMINAMATH_GPT_distinct_after_removal_l949_94954


namespace NUMINAMATH_GPT_sum_abc_eq_ten_l949_94950

theorem sum_abc_eq_ten (a b c : ℝ) (h : (a - 5)^2 + (b - 3)^2 + (c - 2)^2 = 0) : a + b + c = 10 :=
by
  sorry

end NUMINAMATH_GPT_sum_abc_eq_ten_l949_94950


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l949_94920

variable {a b : ℝ}

theorem sufficient_but_not_necessary (h : b < a ∧ a < 0) : 1 / a < 1 / b :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l949_94920


namespace NUMINAMATH_GPT_estimate_larger_than_difference_l949_94915

theorem estimate_larger_than_difference
  (u v δ γ : ℝ)
  (huv : u > v)
  (hδ : δ > 0)
  (hγ : γ > 0)
  (hδγ : δ > γ) : (u + δ) - (v - γ) > u - v := by
  sorry

end NUMINAMATH_GPT_estimate_larger_than_difference_l949_94915


namespace NUMINAMATH_GPT_solve_expression_l949_94992

noncomputable def expression : ℝ := 5 * 1.6 - 2 * 1.4 / 1.3

theorem solve_expression : expression = 5.8462 := 
by 
  sorry

end NUMINAMATH_GPT_solve_expression_l949_94992


namespace NUMINAMATH_GPT_number_of_poly_lines_l949_94959

def nonSelfIntersectingPolyLines (n : ℕ) : ℕ :=
  if n = 2 then 1
  else if n ≥ 3 then n * 2^(n - 3)
  else 0

theorem number_of_poly_lines (n : ℕ) (h : n > 1) :
  nonSelfIntersectingPolyLines n =
  if n = 2 then 1 else n * 2^(n - 3) :=
by sorry

end NUMINAMATH_GPT_number_of_poly_lines_l949_94959


namespace NUMINAMATH_GPT_less_than_n_repetitions_l949_94999

variable {n : ℕ} (a : Fin n.succ → ℕ)

def is_repetition (a : Fin n.succ → ℕ) (k l p : ℕ) : Prop :=
  p ≤ (l - k) / 2 ∧
  (∀ i : ℕ, k + 1 ≤ i ∧ i ≤ l - p → a ⟨i, sorry⟩ = a ⟨i + p, sorry⟩) ∧
  (k > 0 → a ⟨k, sorry⟩ ≠ a ⟨k + p, sorry⟩) ∧
  (l < n → a ⟨l - p + 1, sorry⟩ ≠ a ⟨l + 1, sorry⟩)

theorem less_than_n_repetitions (a : Fin n.succ → ℕ) :
  ∃ r : ℕ, r < n ∧ ∀ k l : ℕ, is_repetition a k l r → r < n :=
sorry

end NUMINAMATH_GPT_less_than_n_repetitions_l949_94999


namespace NUMINAMATH_GPT_solve_inequality_l949_94946

theorem solve_inequality : 
  {x : ℝ | (1 / (x^2 + 1)) > (4 / x) + (21 / 10)} = {x : ℝ | -2 < x ∧ x < 0} :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l949_94946


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l949_94934

-- Define that a sequence is arithmetic
def is_arithmetic_sequence (a : ℕ → ℝ) :=
  ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0

theorem arithmetic_sequence_sum (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_cond : a 2 + a 10 = 16) : a 4 + a 6 + a 8 = 24 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l949_94934


namespace NUMINAMATH_GPT_tangent_line_equation_l949_94980

theorem tangent_line_equation (y : ℝ → ℝ) (x : ℝ) (dy_dx : ℝ → ℝ) (tangent_eq : ℝ → ℝ → Prop):
  (∀ x, y x = x^2 + Real.log x) →
  (∀ x, dy_dx x = (deriv y) x) →
  (dy_dx 1 = 3) →
  (tangent_eq x (y x) ↔ (3 * x - y x - 2 = 0)) →
  tangent_eq 1 (y 1) :=
by
  intros y_def dy_dx_def slope_at_1 tangent_line_char
  sorry

end NUMINAMATH_GPT_tangent_line_equation_l949_94980


namespace NUMINAMATH_GPT_right_angled_triangle_max_area_l949_94974

theorem right_angled_triangle_max_area (a b : ℝ) (h : a + b = 4) : (1 / 2) * a * b ≤ 2 :=
by 
  sorry

end NUMINAMATH_GPT_right_angled_triangle_max_area_l949_94974


namespace NUMINAMATH_GPT_compare_y_values_l949_94940

theorem compare_y_values :
  let y₁ := 2 / (-2)
  let y₂ := 2 / (-1)
  y₁ > y₂ := by sorry

end NUMINAMATH_GPT_compare_y_values_l949_94940


namespace NUMINAMATH_GPT_odd_function_sum_zero_l949_94943

def odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g x

theorem odd_function_sum_zero (g : ℝ → ℝ) (a : ℝ) (h_odd : odd_function g) : 
  g a + g (-a) = 0 :=
by 
  sorry

end NUMINAMATH_GPT_odd_function_sum_zero_l949_94943


namespace NUMINAMATH_GPT_number_of_decks_bought_l949_94929

theorem number_of_decks_bought :
  ∃ T : ℕ, (8 * T + 5 * 8 = 64) ∧ T = 3 :=
by
  sorry

end NUMINAMATH_GPT_number_of_decks_bought_l949_94929


namespace NUMINAMATH_GPT_madeline_water_intake_l949_94917

-- Declare necessary data and conditions
def bottle_A : ℕ := 8
def bottle_B : ℕ := 12
def bottle_C : ℕ := 16

def goal_yoga : ℕ := 15
def goal_work : ℕ := 35
def goal_jog : ℕ := 20
def goal_evening : ℕ := 30

def intake_yoga : ℕ := 2 * bottle_A
def intake_work : ℕ := 3 * bottle_B
def intake_jog : ℕ := 2 * bottle_C
def intake_evening : ℕ := 2 * bottle_A + 2 * bottle_C

def total_intake : ℕ := intake_yoga + intake_work + intake_jog + intake_evening
def goal_total : ℕ := 100

-- Statement of the proof problem
theorem madeline_water_intake : total_intake = 132 ∧ total_intake - goal_total = 32 :=
by
  -- Calculation parts go here (not needed per instruction)
  sorry

end NUMINAMATH_GPT_madeline_water_intake_l949_94917


namespace NUMINAMATH_GPT_new_individuals_weight_l949_94963

variables (W : ℝ) (A B C : ℝ)

-- Conditions
def original_twelve_people_weight : ℝ := W
def weight_leaving_1 : ℝ := 64
def weight_leaving_2 : ℝ := 75
def weight_leaving_3 : ℝ := 81
def average_increase : ℝ := 3.6
def total_weight_increase : ℝ := 12 * average_increase
def weight_leaving_sum : ℝ := weight_leaving_1 + weight_leaving_2 + weight_leaving_3

-- Equation derived from the problem conditions
def new_individuals_weight_sum : ℝ := weight_leaving_sum + total_weight_increase

-- Theorem to prove
theorem new_individuals_weight :
  A + B + C = 263.2 :=
by
  sorry

end NUMINAMATH_GPT_new_individuals_weight_l949_94963


namespace NUMINAMATH_GPT_max_principals_in_10_years_l949_94911

theorem max_principals_in_10_years (term_length : ℕ) (period_length : ℕ) (max_principals : ℕ)
  (term_length_eq : term_length = 4) (period_length_eq : period_length = 10) :
  max_principals = 4 :=
by
  sorry

end NUMINAMATH_GPT_max_principals_in_10_years_l949_94911


namespace NUMINAMATH_GPT_decimal_subtraction_l949_94988

theorem decimal_subtraction (a b : ℝ) (h1 : a = 3.79) (h2 : b = 2.15) : a - b = 1.64 := by
  rw [h1, h2]
  -- This follows from the correct calculation rule
  sorry

end NUMINAMATH_GPT_decimal_subtraction_l949_94988


namespace NUMINAMATH_GPT_factorization_of_x4_plus_81_l949_94966

theorem factorization_of_x4_plus_81 :
  ∀ x : ℝ, x^4 + 81 = (x^2 - 3 * x + 4.5) * (x^2 + 3 * x + 4.5) :=
by
  intros x
  sorry

end NUMINAMATH_GPT_factorization_of_x4_plus_81_l949_94966


namespace NUMINAMATH_GPT_otimes_example_l949_94981

def otimes (a b : ℤ) : ℤ := a^2 - a * b

theorem otimes_example : otimes 4 (otimes 2 (-5)) = -40 := by
  sorry

end NUMINAMATH_GPT_otimes_example_l949_94981


namespace NUMINAMATH_GPT_polynomial_min_k_eq_l949_94926

theorem polynomial_min_k_eq {k : ℝ} :
  (∃ k : ℝ, ∀ x y : ℝ, 9 * x^2 - 12 * k * x * y + (2 * k^2 + 3) * y^2 - 6 * x - 9 * y + 12 >= 0)
  ↔ k = (Real.sqrt 3) / 4 :=
sorry

end NUMINAMATH_GPT_polynomial_min_k_eq_l949_94926


namespace NUMINAMATH_GPT_smallest_integer_cubing_y_eq_350_l949_94938

def y : ℕ := 2^3 * 3^5 * 4^5 * 5^4 * 6^3 * 7^5 * 8^2

theorem smallest_integer_cubing_y_eq_350 : ∃ z : ℕ, z * y = (2^23) * (3^9) * (5^6) * (7^6) → z = 350 :=
by
  sorry

end NUMINAMATH_GPT_smallest_integer_cubing_y_eq_350_l949_94938


namespace NUMINAMATH_GPT_hyperbola_center_l949_94931

theorem hyperbola_center (x y : ℝ) :
  (∃ h k, h = 2 ∧ k = -1 ∧ 
    (∀ x y, (3 * y + 3)^2 / 7^2 - (4 * x - 8)^2 / 6^2 = 1 ↔ 
      (y - (-1))^2 / ((7 / 3)^2) - (x - 2)^2 / ((3 / 2)^2) = 1)) :=
by sorry

end NUMINAMATH_GPT_hyperbola_center_l949_94931


namespace NUMINAMATH_GPT_alice_has_ball_after_two_turns_l949_94964

noncomputable def prob_alice_has_ball_after_two_turns : ℚ :=
  let p_A_B := (3 : ℚ) / 5 -- Probability Alice tosses to Bob
  let p_B_A := (1 : ℚ) / 3 -- Probability Bob tosses to Alice
  let p_A_A := (2 : ℚ) / 5 -- Probability Alice keeps the ball
  (p_A_B * p_B_A) + (p_A_A * p_A_A)

theorem alice_has_ball_after_two_turns :
  prob_alice_has_ball_after_two_turns = 9 / 25 :=
by
  -- skipping the proof
  sorry

end NUMINAMATH_GPT_alice_has_ball_after_two_turns_l949_94964


namespace NUMINAMATH_GPT_Emily_money_made_l949_94965

def price_per_bar : ℕ := 4
def total_bars : ℕ := 8
def bars_sold : ℕ := total_bars - 3
def money_made : ℕ := bars_sold * price_per_bar

theorem Emily_money_made : money_made = 20 :=
by
  sorry

end NUMINAMATH_GPT_Emily_money_made_l949_94965


namespace NUMINAMATH_GPT_intersection_points_of_graph_and_line_l949_94907

theorem intersection_points_of_graph_and_line (f : ℝ → ℝ) :
  (∀ x : ℝ, f x ≠ my_special_value) → (∀ x₁ x₂ : ℝ, f x₁ = f x₂ → x₁ = x₂) →
  ∃! x : ℝ, x = 1 ∧ ∃ y : ℝ, y = f x :=
by
  sorry

end NUMINAMATH_GPT_intersection_points_of_graph_and_line_l949_94907


namespace NUMINAMATH_GPT_polygon_sides_l949_94922

theorem polygon_sides (n : ℕ) (h : (n-3) * 180 < 2008 ∧ 2008 < (n-1) * 180) : 
  n = 14 :=
sorry

end NUMINAMATH_GPT_polygon_sides_l949_94922


namespace NUMINAMATH_GPT_koala_fiber_absorption_l949_94942

theorem koala_fiber_absorption (x : ℝ) (hx : 0.30 * x = 12) : x = 40 :=
by
  sorry

end NUMINAMATH_GPT_koala_fiber_absorption_l949_94942


namespace NUMINAMATH_GPT_lock_code_difference_l949_94913

theorem lock_code_difference :
  ∃ A B C D, A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
             (A = 4 ∧ B = 2 * C ∧ C = D) ∨
             (A = 9 ∧ B = 3 * C ∧ C = D) ∧
             (A * 100 + B * 10 + C - (D * 100 + (2 * D) * 10 + D)) = 541 :=
sorry

end NUMINAMATH_GPT_lock_code_difference_l949_94913


namespace NUMINAMATH_GPT_determine_n_l949_94986

theorem determine_n (x n : ℝ) : 
  (∃ c d : ℝ, G = (c * x + d) ^ 2) ∧ (G = (8 * x^2 + 24 * x + 3 * n) / 8) → n = 6 :=
by {
  sorry
}

end NUMINAMATH_GPT_determine_n_l949_94986


namespace NUMINAMATH_GPT_ranking_l949_94975

variables (score : string → ℝ)
variables (Hannah Cassie Bridget David : string)

-- Conditions based on the problem statement
axiom Hannah_shows_her_test_to_everyone : ∀ x, x ≠ Hannah → x = Cassie ∨ x = Bridget ∨ x = David
axiom David_shows_his_test_only_to_Bridget : ∀ x, x ≠ Bridget → x ≠ David
axiom Cassie_does_not_show_her_test : ∀ x, x = Hannah ∨ x = Bridget ∨ x = David → x ≠ Cassie

-- Statements based on what Cassie and Bridget claim
axiom Cassie_statement : score Cassie > min (score Hannah) (score Bridget)
axiom Bridget_statement : score David > score Bridget

-- Final ranking to be proved
theorem ranking : score David > score Bridget ∧ score Bridget > score Cassie ∧ score Cassie > score Hannah := sorry

end NUMINAMATH_GPT_ranking_l949_94975


namespace NUMINAMATH_GPT_percent_of_part_is_20_l949_94921

theorem percent_of_part_is_20 {Part Whole : ℝ} (hPart : Part = 14) (hWhole : Whole = 70) : (Part / Whole) * 100 = 20 :=
by
  rw [hPart, hWhole]
  have h : (14 : ℝ) / 70 = 0.2 := by norm_num
  rw [h]
  norm_num

end NUMINAMATH_GPT_percent_of_part_is_20_l949_94921


namespace NUMINAMATH_GPT_calculation_result_l949_94973

def initial_number : ℕ := 15
def subtracted_value : ℕ := 2
def added_value : ℕ := 4
def divisor : ℕ := 1
def second_divisor : ℕ := 2
def multiplier : ℕ := 8

theorem calculation_result : 
  (initial_number - subtracted_value + (added_value / divisor : ℕ)) / second_divisor * multiplier = 68 :=
by
  sorry

end NUMINAMATH_GPT_calculation_result_l949_94973


namespace NUMINAMATH_GPT_sharks_win_percentage_at_least_ninety_percent_l949_94910

theorem sharks_win_percentage_at_least_ninety_percent (N : ℕ) :
  let initial_games := 3
  let initial_shark_wins := 2
  let total_games := initial_games + N
  let total_shark_wins := initial_shark_wins + N
  total_shark_wins * 10 ≥ total_games * 9 ↔ N ≥ 7 :=
by
  intros
  sorry

end NUMINAMATH_GPT_sharks_win_percentage_at_least_ninety_percent_l949_94910


namespace NUMINAMATH_GPT_teacher_total_score_l949_94961

/-- Conditions -/
def written_test_score : ℝ := 80
def interview_score : ℝ := 60
def written_test_weight : ℝ := 0.6
def interview_weight : ℝ := 0.4

/-- Prove the total score -/
theorem teacher_total_score :
  written_test_score * written_test_weight + interview_score * interview_weight = 72 :=
by
  sorry

end NUMINAMATH_GPT_teacher_total_score_l949_94961


namespace NUMINAMATH_GPT_al_sandwich_combinations_l949_94936

def types_of_bread : ℕ := 5
def types_of_meat : ℕ := 6
def types_of_cheese : ℕ := 5

def restricted_turkey_swiss_combinations : ℕ := 5
def restricted_white_chicken_combinations : ℕ := 5
def restricted_rye_turkey_combinations : ℕ := 5

def total_sandwich_combinations : ℕ := types_of_bread * types_of_meat * types_of_cheese

def valid_sandwich_combinations : ℕ :=
  total_sandwich_combinations - restricted_turkey_swiss_combinations
  - restricted_white_chicken_combinations - restricted_rye_turkey_combinations

theorem al_sandwich_combinations : valid_sandwich_combinations = 135 := 
  by
  sorry

end NUMINAMATH_GPT_al_sandwich_combinations_l949_94936


namespace NUMINAMATH_GPT_avg_price_two_returned_theorem_l949_94905

-- Defining the initial conditions given in the problem
def avg_price_of_five (price: ℕ) (packets: ℕ) : Prop :=
  packets = 5 ∧ price = 20

def avg_price_of_three_remaining (price: ℕ) (packets: ℕ) : Prop :=
  packets = 3 ∧ price = 12
  
def cost_of_packets (price packets: ℕ) := price * packets

noncomputable def avg_price_two_returned (total_initial_cost total_remaining_cost: ℕ) :=
  (total_initial_cost - total_remaining_cost) / 2

-- The Lean 4 proof statement
theorem avg_price_two_returned_theorem (p1 p2 p3 p4: ℕ):
  avg_price_of_five p1 5 →
  avg_price_of_three_remaining p2 3 →
  cost_of_packets p1 5 = 100 →
  cost_of_packets p2 3 = 36 →
  avg_price_two_returned 100 36 = 32 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_avg_price_two_returned_theorem_l949_94905


namespace NUMINAMATH_GPT_no_solutions_triples_l949_94995

theorem no_solutions_triples (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a! + b^3 ≠ 18 + c^3 :=
by
  sorry

end NUMINAMATH_GPT_no_solutions_triples_l949_94995
