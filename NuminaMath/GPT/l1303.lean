import Mathlib

namespace NUMINAMATH_GPT_initial_puppies_correct_l1303_130353

def initial_puppies (total_puppies_after: ℝ) (bought_puppies: ℝ) : ℝ :=
  total_puppies_after - bought_puppies

theorem initial_puppies_correct : initial_puppies (4.2 * 5.0) 3.0 = 18.0 := by
  sorry

end NUMINAMATH_GPT_initial_puppies_correct_l1303_130353


namespace NUMINAMATH_GPT_fraction_evaluation_l1303_130378

theorem fraction_evaluation : (1 / (2 + 1 / (3 + 1 / 4))) = (13 / 30) := by
  sorry

end NUMINAMATH_GPT_fraction_evaluation_l1303_130378


namespace NUMINAMATH_GPT_max_sum_x_y_l1303_130330

theorem max_sum_x_y (x y : ℝ) (h1 : x^2 + y^2 = 7) (h2 : x^3 + y^3 = 10) : x + y ≤ 4 :=
sorry

end NUMINAMATH_GPT_max_sum_x_y_l1303_130330


namespace NUMINAMATH_GPT_n1_prime_n2_not_prime_l1303_130354

def n1 := 1163
def n2 := 16424
def N := 19101112
def N_eq : N = n1 * n2 := by decide

theorem n1_prime : Prime n1 := 
sorry

theorem n2_not_prime : ¬ Prime n2 :=
sorry

end NUMINAMATH_GPT_n1_prime_n2_not_prime_l1303_130354


namespace NUMINAMATH_GPT_rectangular_floor_paint_l1303_130394

theorem rectangular_floor_paint (a b : ℕ) (ha : a > 0) (hb : b > a) (h1 : a * b = 2 * (a - 4) * (b - 4) + 32) : 
  ∃ pairs : Finset (ℕ × ℕ), pairs.card = 3 ∧ ∀ (p : ℕ × ℕ), p ∈ pairs → b > a :=
by 
  sorry

end NUMINAMATH_GPT_rectangular_floor_paint_l1303_130394


namespace NUMINAMATH_GPT_betty_oranges_l1303_130331

theorem betty_oranges (boxes: ℕ) (oranges_per_box: ℕ) (h1: boxes = 3) (h2: oranges_per_box = 8) : boxes * oranges_per_box = 24 :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_betty_oranges_l1303_130331


namespace NUMINAMATH_GPT_evaluate_x2_plus_y2_l1303_130303

theorem evaluate_x2_plus_y2 (x y : ℝ) (h₁ : 3 * x + 2 * y = 20) (h₂ : 4 * x + 2 * y = 26) : x^2 + y^2 = 37 := by
  sorry

end NUMINAMATH_GPT_evaluate_x2_plus_y2_l1303_130303


namespace NUMINAMATH_GPT_smallest_apples_l1303_130321

theorem smallest_apples (A : ℕ) (h1 : A % 9 = 2) (h2 : A % 10 = 2) (h3 : A % 11 = 2) (h4 : A > 2) : A = 992 :=
sorry

end NUMINAMATH_GPT_smallest_apples_l1303_130321


namespace NUMINAMATH_GPT_twice_perimeter_of_square_l1303_130380

theorem twice_perimeter_of_square (s : ℝ) (h : s^2 = 625) : 2 * 4 * s = 200 :=
by sorry

end NUMINAMATH_GPT_twice_perimeter_of_square_l1303_130380


namespace NUMINAMATH_GPT_remainder_division_l1303_130340

def polynomial (x : ℝ) : ℝ := x^4 - 4 * x^2 + 7 * x - 8

theorem remainder_division : polynomial 3 = 58 :=
by
  sorry

end NUMINAMATH_GPT_remainder_division_l1303_130340


namespace NUMINAMATH_GPT_matrix_power_A_2023_l1303_130392

open Matrix

def A : Matrix (Fin 3) (Fin 3) ℚ :=
  ![
    ![0, -1, 0],
    ![1, 0, 0],
    ![0, 0, 1]
  ]

theorem matrix_power_A_2023 :
  A ^ 2023 = ![
    ![0, 1, 0],
    ![-1, 0, 0],
    ![0, 0, 1]
  ] :=
sorry

end NUMINAMATH_GPT_matrix_power_A_2023_l1303_130392


namespace NUMINAMATH_GPT_angle_B_is_60_l1303_130316

theorem angle_B_is_60 (A B C : ℝ) (h_seq : 2 * B = A + C) (h_sum : A + B + C = 180) : B = 60 := 
by 
  sorry

end NUMINAMATH_GPT_angle_B_is_60_l1303_130316


namespace NUMINAMATH_GPT_number_of_chickens_l1303_130384

def eggs_per_chicken : ℕ := 6
def eggs_per_carton : ℕ := 12
def full_cartons : ℕ := 10

theorem number_of_chickens :
  (full_cartons * eggs_per_carton) / eggs_per_chicken = 20 :=
by
  sorry

end NUMINAMATH_GPT_number_of_chickens_l1303_130384


namespace NUMINAMATH_GPT_prove_total_number_of_apples_l1303_130327

def avg_price (light_price heavy_price : ℝ) (light_proportion heavy_proportion : ℝ) : ℝ :=
  light_proportion * light_price + heavy_proportion * heavy_price

def weighted_avg_price (prices proportions : List ℝ) : ℝ :=
  (List.map (λ ⟨p, prop⟩ => p * prop) (List.zip prices proportions)).sum

noncomputable def total_num_apples (total_earnings weighted_price : ℝ) : ℝ :=
  total_earnings / weighted_price

theorem prove_total_number_of_apples : 
  let light_proportion := 0.6
  let heavy_proportion := 0.4
  let prices := [avg_price 0.4 0.6 light_proportion heavy_proportion, 
                 avg_price 0.1 0.15 light_proportion heavy_proportion,
                 avg_price 0.25 0.35 light_proportion heavy_proportion,
                 avg_price 0.15 0.25 light_proportion heavy_proportion,
                 avg_price 0.2 0.3 light_proportion heavy_proportion,
                 avg_price 0.05 0.1 light_proportion heavy_proportion]
  let proportions := [0.4, 0.2, 0.15, 0.1, 0.1, 0.05]
  let weighted_avg := weighted_avg_price prices proportions
  total_num_apples 120 weighted_avg = 392 :=
by
  sorry

end NUMINAMATH_GPT_prove_total_number_of_apples_l1303_130327


namespace NUMINAMATH_GPT_minimum_value_proof_l1303_130396

noncomputable def minimum_value (x : ℝ) (h : x > 1) : ℝ :=
  (x^2 + x + 1) / (x - 1)

theorem minimum_value_proof : ∃ x : ℝ, x > 1 ∧ minimum_value x (by sorry) = 3 + 2*Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_minimum_value_proof_l1303_130396


namespace NUMINAMATH_GPT_number_div_mult_l1303_130326

theorem number_div_mult (n : ℕ) (h : n = 4) : (n / 6) * 12 = 8 :=
by
  sorry

end NUMINAMATH_GPT_number_div_mult_l1303_130326


namespace NUMINAMATH_GPT_conic_section_union_l1303_130386

theorem conic_section_union : 
  ∀ (y x : ℝ), y^4 - 6*x^4 = 3*y^2 - 2 → 
  ( ( y^2 - 3*x^2 = 1 ∨ y^2 - 2*x^2 = 1 ) ∧ 
    ( y^2 - 2*x^2 = 2 ∨ y^2 - 3*x^2 = 2 ) ) :=
by
  sorry

end NUMINAMATH_GPT_conic_section_union_l1303_130386


namespace NUMINAMATH_GPT_quadratic_standard_form_l1303_130311

theorem quadratic_standard_form :
  ∀ x : ℝ, (2 * x - 1) ^ 2 = (x + 1) * (3 * x + 4) →
  (∃ a b c : ℝ, a = 1 ∧ b = -11 ∧ c = -3 ∧ a * x^2 + b * x + c = 0) :=
by
  intro x
  intro h
  sorry

end NUMINAMATH_GPT_quadratic_standard_form_l1303_130311


namespace NUMINAMATH_GPT_largest_multiple_of_18_with_8_and_0_digits_l1303_130332

theorem largest_multiple_of_18_with_8_and_0_digits :
  ∃ m : ℕ, (∀ d ∈ (m.digits 10), d = 8 ∨ d = 0) ∧ (m % 18 = 0) ∧ (m = 8888888880) ∧ (m / 18 = 493826048) :=
by sorry

end NUMINAMATH_GPT_largest_multiple_of_18_with_8_and_0_digits_l1303_130332


namespace NUMINAMATH_GPT_largest_whole_number_l1303_130344

theorem largest_whole_number (x : ℕ) : 8 * x < 120 → x ≤ 14 :=
by
  intro h
  -- prove the main statement here
  sorry

end NUMINAMATH_GPT_largest_whole_number_l1303_130344


namespace NUMINAMATH_GPT_problem1_l1303_130369

theorem problem1 (x : ℝ) (hx : x > 0) : (x + 1/x = 2) ↔ (x = 1) :=
by
  sorry

end NUMINAMATH_GPT_problem1_l1303_130369


namespace NUMINAMATH_GPT_max_sum_after_swap_l1303_130347

section
variables (a1 a2 a3 b1 b2 b3 c1 c2 c3 : ℕ)
  (h1 : 100 * a1 + 10 * b1 + c1 + 100 * a2 + 10 * b2 + c2 + 100 * a3 + 10 * b3 + c3 = 2019)
  (h2 : 1 ≤ a1 ∧ a1 ≤ 9 ∧ 0 ≤ b1 ∧ b1 ≤ 9 ∧ 0 ≤ c1 ∧ c1 ≤ 9)
  (h3 : 1 ≤ a2 ∧ a2 ≤ 9 ∧ 0 ≤ b2 ∧ b2 ≤ 9 ∧ 0 ≤ c2 ∧ c2 ≤ 9)
  (h4 : 1 ≤ a3 ∧ a3 ≤ 9 ∧ 0 ≤ b3 ∧ b3 ≤ 9 ∧ 0 ≤ c3 ∧ c3 ≤ 9)

theorem max_sum_after_swap : 100 * c1 + 10 * b1 + a1 + 100 * c2 + 10 * b2 + a2 + 100 * c3 + 10 * b3 + a3 ≤ 2118 := 
  sorry

end

end NUMINAMATH_GPT_max_sum_after_swap_l1303_130347


namespace NUMINAMATH_GPT_find_m_l1303_130328

-- Definitions for the sets A and B
def A (m : ℝ) : Set ℝ := {3, 4, 4 * m - 4}
def B (m : ℝ) : Set ℝ := {3, m^2}

-- Problem statement
theorem find_m {m : ℝ} (h : B m ⊆ A m) : m = -2 :=
sorry

end NUMINAMATH_GPT_find_m_l1303_130328


namespace NUMINAMATH_GPT_handshake_problem_l1303_130325

noncomputable def total_handshakes (num_companies : ℕ) (repr_per_company : ℕ) : ℕ :=
    let total_people := num_companies * repr_per_company
    let possible_handshakes_per_person := total_people - repr_per_company
    (total_people * possible_handshakes_per_person) / 2

theorem handshake_problem : total_handshakes 4 4 = 96 :=
by
  sorry

end NUMINAMATH_GPT_handshake_problem_l1303_130325


namespace NUMINAMATH_GPT_increased_volume_l1303_130314

theorem increased_volume (l w h : ℕ) 
  (volume_eq : l * w * h = 4500) 
  (surface_area_eq : l * w + l * h + w * h = 900) 
  (edges_sum_eq : l + w + h = 54) :
  (l + 1) * (w + 1) * (h + 1) = 5455 := 
by 
  sorry

end NUMINAMATH_GPT_increased_volume_l1303_130314


namespace NUMINAMATH_GPT_factor_of_M_l1303_130312

theorem factor_of_M (a b c : ℕ) (h₀ : a ≠ 0) (h₁ : a < 10) (h₂ : b < 10) (h₃ : c < 10) : 
  1 ∣ (101010 * a + 10001 * b + 100 * c) :=
sorry

end NUMINAMATH_GPT_factor_of_M_l1303_130312


namespace NUMINAMATH_GPT_magic_square_y_minus_x_l1303_130309

theorem magic_square_y_minus_x :
  ∀ (x y : ℝ), 
    (x - 2 = 2 * y + y) ∧ (x - 2 = -2 + y + 6) →
    y - x = -6 :=
by 
  intros x y h
  sorry

end NUMINAMATH_GPT_magic_square_y_minus_x_l1303_130309


namespace NUMINAMATH_GPT_minimize_circumscribed_sphere_radius_l1303_130310

noncomputable def cylinder_surface_area (r h : ℝ) : ℝ :=
  2 * Real.pi * r^2 + 2 * Real.pi * r * h

noncomputable def circumscribed_sphere_radius (r h : ℝ) : ℝ :=
  (r^2 + (1 / 2 * h)^2).sqrt

theorem minimize_circumscribed_sphere_radius (r : ℝ) (h : ℝ) (hr : cylinder_surface_area r h = 16 * Real.pi) : 
  r^2 = 8 * Real.sqrt 5 / 5 :=
sorry

end NUMINAMATH_GPT_minimize_circumscribed_sphere_radius_l1303_130310


namespace NUMINAMATH_GPT_average_marks_l1303_130362

theorem average_marks {n : ℕ} (h1 : 5 * 74 + 104 = n * 79) : n = 6 :=
by
  sorry

end NUMINAMATH_GPT_average_marks_l1303_130362


namespace NUMINAMATH_GPT_negation_of_existential_proposition_l1303_130319

theorem negation_of_existential_proposition : 
  (¬ ∃ x : ℝ, x^2 - x + 3 = 0) = (∀ x : ℝ, x^2 - x + 3 ≠ 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_existential_proposition_l1303_130319


namespace NUMINAMATH_GPT_complement_set_solution_l1303_130335

open Set Real

theorem complement_set_solution :
  let M := {x : ℝ | (1 + x) / (1 - x) > 0}
  compl M = {x : ℝ | x ≤ -1 ∨ x ≥ 1} :=
by
  sorry

end NUMINAMATH_GPT_complement_set_solution_l1303_130335


namespace NUMINAMATH_GPT_Moscow_Olympiad_1958_problem_l1303_130355

theorem Moscow_Olympiad_1958_problem :
  ∀ n : ℤ, 1155 ^ 1958 + 34 ^ 1958 ≠ n ^ 2 := 
by 
  sorry

end NUMINAMATH_GPT_Moscow_Olympiad_1958_problem_l1303_130355


namespace NUMINAMATH_GPT_age_of_youngest_child_l1303_130323

theorem age_of_youngest_child (x : ℕ) 
  (h : x + (x + 4) + (x + 8) + (x + 12) + (x + 16) + (x + 20) + (x + 24) = 112) :
  x = 4 :=
sorry

end NUMINAMATH_GPT_age_of_youngest_child_l1303_130323


namespace NUMINAMATH_GPT_circumcircle_diameter_of_triangle_l1303_130346

theorem circumcircle_diameter_of_triangle (a b c : ℝ) (A B C : ℝ) 
  (h_a : a = 1) 
  (h_B : B = π/4) 
  (h_area : (1/2) * a * c * Real.sin B = 2) : 
  (2 * b = 5 * Real.sqrt 2) := 
sorry

end NUMINAMATH_GPT_circumcircle_diameter_of_triangle_l1303_130346


namespace NUMINAMATH_GPT_number_of_moles_of_methanol_formed_l1303_130308

def ch4_to_co2 : ℚ := 1
def o2_to_co2 : ℚ := 2
def co2_prod_from_ch4 (ch4 : ℚ) : ℚ := ch4 * ch4_to_co2 / o2_to_co2

def co2_to_ch3oh : ℚ := 1
def h2_to_ch3oh : ℚ := 3
def ch3oh_prod_from_co2 (co2 h2 : ℚ) : ℚ :=
  min (co2 / co2_to_ch3oh) (h2 / h2_to_ch3oh)

theorem number_of_moles_of_methanol_formed :
  (ch3oh_prod_from_co2 (co2_prod_from_ch4 5) 10) = 10/3 :=
by
  sorry

end NUMINAMATH_GPT_number_of_moles_of_methanol_formed_l1303_130308


namespace NUMINAMATH_GPT_inequality_holds_l1303_130313

theorem inequality_holds (a b : ℕ) (ha : a > 1) (hb : b > 2) : a ^ b + 1 ≥ b * (a + 1) :=
sorry

end NUMINAMATH_GPT_inequality_holds_l1303_130313


namespace NUMINAMATH_GPT_zaim_larger_part_l1303_130334

theorem zaim_larger_part (x y : ℕ) (h_sum : x + y = 20) (h_prod : x * y = 96) : max x y = 12 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_zaim_larger_part_l1303_130334


namespace NUMINAMATH_GPT_speed_of_stream_l1303_130374

theorem speed_of_stream
  (V S : ℝ)
  (h1 : 27 = 9 * (V - S))
  (h2 : 81 = 9 * (V + S)) :
  S = 3 :=
by
  sorry

end NUMINAMATH_GPT_speed_of_stream_l1303_130374


namespace NUMINAMATH_GPT_time_to_write_all_rearrangements_l1303_130390

-- Define the problem conditions
def sophie_name_length := 6
def rearrangements_per_minute := 18

-- Define the factorial function for calculating permutations
noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Define the total number of rearrangements of Sophie's name
noncomputable def total_rearrangements := factorial sophie_name_length

-- Define the time in minutes to write all rearrangements
noncomputable def time_in_minutes := total_rearrangements / rearrangements_per_minute

-- Convert the time to hours
noncomputable def minutes_to_hours (minutes : ℕ) : ℚ := minutes / 60

-- Prove the time in hours to write all the rearrangements
theorem time_to_write_all_rearrangements : minutes_to_hours time_in_minutes = (2 : ℚ) / 3 := 
  sorry

end NUMINAMATH_GPT_time_to_write_all_rearrangements_l1303_130390


namespace NUMINAMATH_GPT_non_parallel_lines_implies_unique_solution_l1303_130345

variable (a1 b1 c1 a2 b2 c2 : ℝ)

def system_of_equations (x y : ℝ) := a1 * x + b1 * y = c1 ∧ a2 * x + b2 * y = c2

def lines_not_parallel := a1 * b2 ≠ a2 * b1

theorem non_parallel_lines_implies_unique_solution :
  lines_not_parallel a1 b1 a2 b2 → ∃! (x y : ℝ), system_of_equations a1 b1 c1 a2 b2 c2 x y :=
sorry

end NUMINAMATH_GPT_non_parallel_lines_implies_unique_solution_l1303_130345


namespace NUMINAMATH_GPT_triangle_side_b_eq_l1303_130360

   variable (a b c : Real) (A B C : Real)
   variable (cos_A sin_A : Real)
   variable (area : Real)
   variable (π : Real := Real.pi)

   theorem triangle_side_b_eq :
     cos_A = 1 / 3 →
     B = π / 6 →
     a = 4 * Real.sqrt 2 →
     sin_A = 2 * Real.sqrt 2 / 3 →
     b = (a * sin_B / sin_A) →
     b = 3 := sorry
   
end NUMINAMATH_GPT_triangle_side_b_eq_l1303_130360


namespace NUMINAMATH_GPT_cosine_of_half_pi_minus_double_alpha_l1303_130324

theorem cosine_of_half_pi_minus_double_alpha (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.tan α = 2) :
  Real.cos (π / 2 - 2 * α) = 4 / 5 :=
sorry

end NUMINAMATH_GPT_cosine_of_half_pi_minus_double_alpha_l1303_130324


namespace NUMINAMATH_GPT_middle_circle_radius_l1303_130358

theorem middle_circle_radius 
  (r1 r3 : ℝ) 
  (geometric_sequence: ∃ r2 : ℝ, r2 ^ 2 = r1 * r3) 
  (r1_val : r1 = 5) 
  (r3_val : r3 = 20) 
  : ∃ r2 : ℝ, r2 = 10 := 
by
  sorry

end NUMINAMATH_GPT_middle_circle_radius_l1303_130358


namespace NUMINAMATH_GPT_simplify_expression_l1303_130385

theorem simplify_expression (x : ℤ) (h1 : 2 * (x - 1) < x + 1) (h2 : 5 * x + 3 ≥ 2 * x) :
  (x = 2) → (2 / (x^2 + x) / (1 - (x - 1) / (x^2 - 1)) = 1 / 2) :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1303_130385


namespace NUMINAMATH_GPT_position_of_term_in_sequence_l1303_130301

theorem position_of_term_in_sequence 
    (a : ℕ → ℕ)
    (h1 : a 1 = 1)
    (h2 : ∀ n, a (n + 1) - a n = 7 * n) :
    ∃ n, a n = 35351 ∧ n = 101 :=
by
  sorry

end NUMINAMATH_GPT_position_of_term_in_sequence_l1303_130301


namespace NUMINAMATH_GPT_range_of_a_l1303_130318

theorem range_of_a (a : ℝ) : (∃ x : ℝ, a * x^2 + x + 1 < 0) ↔ (a < 1/4) := 
sorry

end NUMINAMATH_GPT_range_of_a_l1303_130318


namespace NUMINAMATH_GPT_bananas_to_pears_l1303_130368

theorem bananas_to_pears:
  (∀ b a o p : ℕ, 
    6 * b = 4 * a → 
    5 * a = 3 * o → 
    4 * o = 7 * p → 
    36 * b = 28 * p) :=
by
  intros b a o p h1 h2 h3
  -- We need to prove 36 * b = 28 * p under the given conditions
  sorry

end NUMINAMATH_GPT_bananas_to_pears_l1303_130368


namespace NUMINAMATH_GPT_cindy_added_pens_l1303_130388

-- Define the initial number of pens
def initial_pens : ℕ := 5

-- Define the number of pens given by Mike
def pens_from_mike : ℕ := 20

-- Define the number of pens given to Sharon
def pens_given_to_sharon : ℕ := 10

-- Define the final number of pens
def final_pens : ℕ := 40

-- Formulate the theorem regarding the pens added by Cindy
theorem cindy_added_pens :
  final_pens = initial_pens + pens_from_mike - pens_given_to_sharon + 25 :=
by
  sorry

end NUMINAMATH_GPT_cindy_added_pens_l1303_130388


namespace NUMINAMATH_GPT_find_ratio_of_sums_l1303_130342

variables {S : ℕ → ℝ} {a : ℕ → ℝ} {d : ℝ}

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

def sum_first_n_terms (S : ℕ → ℝ) (a : ℕ → ℝ) :=
  ∀ n, S n = n * (a 1 + a n) / 2

def ratio_condition (a : ℕ → ℝ) :=
  a 6 / a 5 = 9 / 11

theorem find_ratio_of_sums (seq : ∃ d, arithmetic_sequence a d)
    (sum_prop : sum_first_n_terms S a)
    (ratio_prop : ratio_condition a) :
  S 11 / S 9 = 1 :=
sorry

end NUMINAMATH_GPT_find_ratio_of_sums_l1303_130342


namespace NUMINAMATH_GPT_find_s_l1303_130393

noncomputable def area_of_parallelogram (s : ℝ) : ℝ :=
  (3 * s) * (s * Real.sin (Real.pi / 3))

theorem find_s (s : ℝ) (h1 : area_of_parallelogram s = 27 * Real.sqrt 3) : s = 3 * Real.sqrt 2 := 
  sorry

end NUMINAMATH_GPT_find_s_l1303_130393


namespace NUMINAMATH_GPT_shots_per_puppy_l1303_130304

-- Definitions
def num_pregnant_dogs : ℕ := 3
def puppies_per_dog : ℕ := 4
def cost_per_shot : ℕ := 5
def total_shot_cost : ℕ := 120

-- Total number of puppies
def total_puppies : ℕ := num_pregnant_dogs * puppies_per_dog

-- Total number of shots
def total_shots : ℕ := total_shot_cost / cost_per_shot

-- The theorem to prove
theorem shots_per_puppy : total_shots / total_puppies = 2 :=
by
  sorry

end NUMINAMATH_GPT_shots_per_puppy_l1303_130304


namespace NUMINAMATH_GPT_find_all_functions_l1303_130322

theorem find_all_functions 
  (f : ℤ → ℝ)
  (h1 : ∀ m n : ℤ, m < n → f m < f n)
  (h2 : ∀ m n : ℤ, ∃ k : ℤ, f m - f n = f k) :
  ∃ a t : ℝ, a > 0 ∧ (∀ n : ℤ, f n = a * (n + t)) :=
sorry

end NUMINAMATH_GPT_find_all_functions_l1303_130322


namespace NUMINAMATH_GPT_tan_sum_identity_l1303_130387

theorem tan_sum_identity :
  Real.tan (15 * Real.pi / 180) + Real.tan (30 * Real.pi / 180) + 
  Real.tan (15 * Real.pi / 180) * Real.tan (30 * Real.pi / 180) = 1 :=
by sorry

end NUMINAMATH_GPT_tan_sum_identity_l1303_130387


namespace NUMINAMATH_GPT_valid_ways_to_assign_volunteers_l1303_130307

noncomputable def validAssignments : ℕ := 
  (Nat.choose 5 2) * (Nat.choose 3 2) + (Nat.choose 5 1) * (Nat.choose 4 2)

theorem valid_ways_to_assign_volunteers : validAssignments = 60 := 
  by
    simp [validAssignments]
    sorry

end NUMINAMATH_GPT_valid_ways_to_assign_volunteers_l1303_130307


namespace NUMINAMATH_GPT_chickens_cheaper_than_eggs_l1303_130356

-- Define the initial costs of the chickens
def initial_cost_chicken1 : ℝ := 25
def initial_cost_chicken2 : ℝ := 30
def initial_cost_chicken3 : ℝ := 22
def initial_cost_chicken4 : ℝ := 35

-- Define the weekly feed costs for the chickens
def weekly_feed_cost_chicken1 : ℝ := 1.50
def weekly_feed_cost_chicken2 : ℝ := 1.30
def weekly_feed_cost_chicken3 : ℝ := 1.10
def weekly_feed_cost_chicken4 : ℝ := 0.90

-- Define the weekly egg production for the chickens
def weekly_egg_prod_chicken1 : ℝ := 4
def weekly_egg_prod_chicken2 : ℝ := 3
def weekly_egg_prod_chicken3 : ℝ := 5
def weekly_egg_prod_chicken4 : ℝ := 2

-- Define the cost of a dozen eggs at the store
def cost_per_dozen_eggs : ℝ := 2

-- Define total initial costs, total weekly feed cost, and weekly savings
def total_initial_cost : ℝ := initial_cost_chicken1 + initial_cost_chicken2 + initial_cost_chicken3 + initial_cost_chicken4
def total_weekly_feed_cost : ℝ := weekly_feed_cost_chicken1 + weekly_feed_cost_chicken2 + weekly_feed_cost_chicken3 + weekly_feed_cost_chicken4
def weekly_savings : ℝ := cost_per_dozen_eggs

-- Define the condition for the number of weeks (W) when the chickens become cheaper
def breakeven_weeks : ℝ := 40

theorem chickens_cheaper_than_eggs (W : ℕ) :
  total_initial_cost + W * total_weekly_feed_cost = W * weekly_savings :=
sorry

end NUMINAMATH_GPT_chickens_cheaper_than_eggs_l1303_130356


namespace NUMINAMATH_GPT_negation_proof_l1303_130306

theorem negation_proof :
  (¬ (∀ x : ℝ, x^2 - x + 1 > 0)) ↔ (∃ x : ℝ, x^2 - x + 1 ≤ 0) := sorry

end NUMINAMATH_GPT_negation_proof_l1303_130306


namespace NUMINAMATH_GPT_range_of_m_l1303_130341

theorem range_of_m (m : ℝ) : (∀ x : ℝ, |2009 * x + 1| ≥ |m - 1| - 2) → -1 ≤ m ∧ m ≤ 3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_range_of_m_l1303_130341


namespace NUMINAMATH_GPT_pocket_money_calculation_l1303_130373

theorem pocket_money_calculation
  (a b c d e : ℝ)
  (h1 : (a + b + c + d + e) / 5 = 2300)
  (h2 : (a + b) / 2 = 3000)
  (h3 : (b + c) / 2 = 2100)
  (h4 : (c + d) / 2 = 2750)
  (h5 : a = b + 800) :
  d = 3900 :=
by
  sorry

end NUMINAMATH_GPT_pocket_money_calculation_l1303_130373


namespace NUMINAMATH_GPT_decagon_diagonals_l1303_130383

-- The condition for the number of diagonals in a polygon
def number_of_diagonals (n : Nat) : Nat :=
  n * (n - 3) / 2

-- The specific proof statement for a decagon
theorem decagon_diagonals : number_of_diagonals 10 = 35 := by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_decagon_diagonals_l1303_130383


namespace NUMINAMATH_GPT_complement_of_supplement_of_35_degree_l1303_130382

def angle : ℝ := 35
def supplement (x : ℝ) : ℝ := 180 - x
def complement (x : ℝ) : ℝ := 90 - x

theorem complement_of_supplement_of_35_degree :
  complement (supplement angle) = -55 := by
  sorry

end NUMINAMATH_GPT_complement_of_supplement_of_35_degree_l1303_130382


namespace NUMINAMATH_GPT_find_last_number_l1303_130329

-- Definitions for the conditions
def avg_first_three (A B C : ℕ) : ℕ := (A + B + C) / 3
def avg_last_three (B C D : ℕ) : ℕ := (B + C + D) / 3
def sum_first_last (A D : ℕ) : ℕ := A + D

-- Proof problem statement
theorem find_last_number (A B C D : ℕ) 
  (h1 : avg_first_three A B C = 6)
  (h2 : avg_last_three B C D = 5)
  (h3 : sum_first_last A D = 11) : D = 4 :=
sorry

end NUMINAMATH_GPT_find_last_number_l1303_130329


namespace NUMINAMATH_GPT_find_kn_l1303_130300

theorem find_kn (k n : ℕ) (h_pos_k : k > 0) (h_pos_n : n > 0) (h_eq : k^2 - 2016 = 3^n) : k = 45 ∧ n = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_kn_l1303_130300


namespace NUMINAMATH_GPT_triangle_area_l1303_130350

-- Define the given conditions
def perimeter : ℝ := 60
def inradius : ℝ := 2.5

-- Prove the area of the triangle using the given inradius and perimeter
theorem triangle_area (p : ℝ) (r : ℝ) (h1 : p = 60) (h2 : r = 2.5) :
  (r * (p / 2)) = 75 := 
by
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_triangle_area_l1303_130350


namespace NUMINAMATH_GPT_measure_angle_R_l1303_130399

theorem measure_angle_R (P Q R : ℝ) (h1 : P + Q = 60) : R = 120 :=
by
  have sum_of_angles_in_triangle : P + Q + R = 180 := sorry
  rw [h1] at sum_of_angles_in_triangle
  linarith

end NUMINAMATH_GPT_measure_angle_R_l1303_130399


namespace NUMINAMATH_GPT_average_rate_of_change_l1303_130389

noncomputable def f (x : ℝ) := 2 * x + 1

theorem average_rate_of_change :
  (f 2 - f 1) / (2 - 1) = 2 :=
by
  sorry

end NUMINAMATH_GPT_average_rate_of_change_l1303_130389


namespace NUMINAMATH_GPT_cubes_difference_l1303_130336

theorem cubes_difference (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 25) : a^3 - b^3 = 99 := 
sorry

end NUMINAMATH_GPT_cubes_difference_l1303_130336


namespace NUMINAMATH_GPT_last_digit_of_a2009_div_a2006_is_6_l1303_130343
open Nat

def ratio_difference_sequence (a : ℕ → ℕ) (d : ℕ) : Prop :=
  ∀ n : ℕ, n > 0 → a (n + 2) * a n = (a (n + 1)) ^ 2 + d * a (n + 1)

theorem last_digit_of_a2009_div_a2006_is_6
  (a : ℕ → ℕ)
  (h1 : a 1 = 1)
  (h2 : a 2 = 1)
  (h3 : a 3 = 2)
  (d : ℕ)
  (h4 : ratio_difference_sequence a d) :
  (a 2009 / a 2006) % 10 = 6 :=
by
  sorry

end NUMINAMATH_GPT_last_digit_of_a2009_div_a2006_is_6_l1303_130343


namespace NUMINAMATH_GPT_arithmetic_sequence_properties_l1303_130391

def arithmetic_seq (a : ℕ → ℕ) : Prop :=
  ∃ d a_1, ∀ n, a n = a_1 + d * (n - 1)

def sum_n (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
  ∀ n, S n = n * (a 1 + a n) / 2

def sum_b (b : ℕ → ℕ) (T : ℕ → ℕ) : Prop :=
  ∀ n, T n = n^2 + n + (3^(n+1) - 3)/2

theorem arithmetic_sequence_properties :
  ∀ (a : ℕ → ℕ) (S : ℕ → ℕ),
    (arithmetic_seq a) →
    a 5 = 10 →
    S 7 = 56 →
    (∀ n, a n = 2 * n) ∧
    ∃ (b T : ℕ → ℕ), (∀ n, b n = a n + 3^n) ∧ sum_b b T :=
by
  intros a S ha h5 hS7
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_properties_l1303_130391


namespace NUMINAMATH_GPT_m_n_value_l1303_130364

theorem m_n_value (m n : ℝ)
  (h1 : m * (-1/2)^2 + n * (-1/2) - 1/m < 0)
  (h2 : m * 2^2 + n * 2 - 1/m < 0)
  (h3 : m < 0)
  (h4 : (-1/2 + 2 = -n/m))
  (h5 : (-1/2) * 2 = -1/m^2) :
  m - n = -5/2 :=
sorry

end NUMINAMATH_GPT_m_n_value_l1303_130364


namespace NUMINAMATH_GPT_al_sandwiches_count_l1303_130377

noncomputable def total_sandwiches (bread meat cheese : ℕ) : ℕ :=
  bread * meat * cheese

noncomputable def prohibited_combinations (bread_forbidden_combination cheese_forbidden_combination : ℕ) : ℕ := 
  bread_forbidden_combination + cheese_forbidden_combination

theorem al_sandwiches_count (bread meat cheese : ℕ) 
  (bread_forbidden_combination cheese_forbidden_combination : ℕ) 
  (h1 : bread = 5) 
  (h2 : meat = 7) 
  (h3 : cheese = 6) 
  (h4 : bread_forbidden_combination = 5) 
  (h5 : cheese_forbidden_combination = 6) : 
  total_sandwiches bread meat cheese - prohibited_combinations bread_forbidden_combination cheese_forbidden_combination = 199 :=
by
  sorry

end NUMINAMATH_GPT_al_sandwiches_count_l1303_130377


namespace NUMINAMATH_GPT_range_of_a_l1303_130366

theorem range_of_a (a : ℝ) (h : a > 0) : (∀ x : ℝ, x > 0 → 9 * x + a^2 / x ≥ a^2 + 8) → 2 ≤ a ∧ a ≤ 4 :=
by
  intros h1
  sorry

end NUMINAMATH_GPT_range_of_a_l1303_130366


namespace NUMINAMATH_GPT_months_decreasing_l1303_130352

noncomputable def stock_decrease (m : ℕ) : Prop :=
  2 * m + 2 * 8 = 18

theorem months_decreasing (m : ℕ) (h : stock_decrease m) : m = 1 :=
by
  exact sorry

end NUMINAMATH_GPT_months_decreasing_l1303_130352


namespace NUMINAMATH_GPT_minimum_spending_l1303_130375

noncomputable def box_volume (length width height : ℕ) : ℕ := length * width * height
noncomputable def total_boxes_needed (total_volume box_volume : ℕ) : ℕ := (total_volume + box_volume - 1) / box_volume
noncomputable def total_cost (num_boxes : ℕ) (price_per_box : ℝ) : ℝ := num_boxes * price_per_box

theorem minimum_spending
  (box_length box_width box_height : ℕ)
  (price_per_box : ℝ)
  (total_collection_volume : ℕ)
  (h1 : box_length = 20)
  (h2 : box_width = 20)
  (h3 : box_height = 15)
  (h4 : price_per_box = 0.90)
  (h5 : total_collection_volume = 3060000) :
  total_cost (total_boxes_needed total_collection_volume (box_volume box_length box_width box_height)) price_per_box = 459 :=
by
  rw [h1, h2, h3, h4, h5]
  have box_vol : box_volume 20 20 15 = 6000 := by norm_num [box_volume]
  have boxes_needed : total_boxes_needed 3060000 6000 = 510 := by norm_num [total_boxes_needed, box_volume, *]
  have cost : total_cost 510 0.90 = 459 := by norm_num [total_cost]
  exact cost

end NUMINAMATH_GPT_minimum_spending_l1303_130375


namespace NUMINAMATH_GPT_george_speed_second_segment_l1303_130320

theorem george_speed_second_segment 
  (distance_total : ℝ)
  (speed_normal : ℝ)
  (distance_first : ℝ)
  (speed_first : ℝ) : 
  distance_total = 1 ∧ 
  speed_normal = 3 ∧ 
  distance_first = 0.5 ∧ 
  speed_first = 2 →
  (distance_first / speed_first + 0.5 * speed_second = 1 / speed_normal → speed_second = 6) :=
sorry

end NUMINAMATH_GPT_george_speed_second_segment_l1303_130320


namespace NUMINAMATH_GPT_petrol_price_increase_l1303_130302

theorem petrol_price_increase
  (P P_new : ℝ)
  (C : ℝ)
  (h1 : P * C = P_new * (C * 0.7692307692307693))
  (h2 : C * (1 - 0.23076923076923073) = C * 0.7692307692307693) :
  ((P_new - P) / P) * 100 = 30 := 
  sorry

end NUMINAMATH_GPT_petrol_price_increase_l1303_130302


namespace NUMINAMATH_GPT_michael_needs_flour_l1303_130370

-- Define the given conditions
def total_flour : ℕ := 8
def measuring_cup : ℚ := 1/4
def scoops_to_remove : ℕ := 8

-- Prove the amount of flour Michael needs is 6 cups
theorem michael_needs_flour : 
  (total_flour - (scoops_to_remove * measuring_cup)) = 6 := 
by
  sorry

end NUMINAMATH_GPT_michael_needs_flour_l1303_130370


namespace NUMINAMATH_GPT_total_red_marbles_l1303_130349

theorem total_red_marbles (jessica_marbles sandy_marbles alex_marbles : ℕ) (dozen : ℕ)
  (h_jessica : jessica_marbles = 3 * dozen)
  (h_sandy : sandy_marbles = 4 * jessica_marbles)
  (h_alex : alex_marbles = jessica_marbles + 2 * dozen)
  (h_dozen : dozen = 12) :
  jessica_marbles + sandy_marbles + alex_marbles = 240 :=
by
  sorry

end NUMINAMATH_GPT_total_red_marbles_l1303_130349


namespace NUMINAMATH_GPT_parabola_passes_through_point_l1303_130351

theorem parabola_passes_through_point {x y : ℝ} (h_eq : y = (1/2) * x^2 - 2) :
  (x = 2 ∧ y = 0) :=
by
  sorry

end NUMINAMATH_GPT_parabola_passes_through_point_l1303_130351


namespace NUMINAMATH_GPT_vector_problem_solution_l1303_130395

variables (a b c : ℤ × ℤ) (m n : ℤ)

def parallel (v1 v2 : ℤ × ℤ) : Prop := v1.1 * v2.2 = v1.2 * v2.1
def perpendicular (v1 v2 : ℤ × ℤ) : Prop := v1.1 * v2.1 + v1.2 * v2.2 = 0

theorem vector_problem_solution
  (a_eq : a = (1, -2))
  (b_eq : b = (2, m - 1))
  (c_eq : c = (4, n))
  (h1 : parallel a b)
  (h2 : perpendicular b c) :
  m + n = -1 := by
  sorry

end NUMINAMATH_GPT_vector_problem_solution_l1303_130395


namespace NUMINAMATH_GPT_bin101_to_decimal_l1303_130365

-- Define the binary representation of 101 (base 2)
def bin101 : ℕ := 1 * 2^2 + 0 * 2^1 + 1 * 2^0

-- State the theorem that asserts the decimal value of 101 (base 2) is 5
theorem bin101_to_decimal : bin101 = 5 := by
  sorry

end NUMINAMATH_GPT_bin101_to_decimal_l1303_130365


namespace NUMINAMATH_GPT_sector_COD_area_ratio_l1303_130371

-- Define the given angles
def angle_AOC : ℝ := 30
def angle_DOB : ℝ := 45
def angle_AOB : ℝ := 180

-- Define the full circle angle
def full_circle_angle : ℝ := 360

-- Calculate the angle COD
def angle_COD : ℝ := angle_AOB - angle_AOC - angle_DOB

-- State the ratio of the area of sector COD to the area of the circle
theorem sector_COD_area_ratio :
  angle_COD / full_circle_angle = 7 / 24 := by
  sorry

end NUMINAMATH_GPT_sector_COD_area_ratio_l1303_130371


namespace NUMINAMATH_GPT_div_pow_sub_one_l1303_130398

theorem div_pow_sub_one (n : ℕ) (h : n > 1) : (n - 1) ^ 2 ∣ n ^ (n - 1) - 1 :=
sorry

end NUMINAMATH_GPT_div_pow_sub_one_l1303_130398


namespace NUMINAMATH_GPT_find_alpha_l1303_130372

noncomputable def isochronous_growth (k α x₁ x₂ y₁ y₂ : ℝ) : Prop :=
  y₁ = k * x₁^α ∧
  y₂ = k * x₂^α ∧
  x₂ = 16 * x₁ ∧
  y₂ = 8 * y₁

theorem find_alpha (k x₁ x₂ y₁ y₂ : ℝ) (h : isochronous_growth k (3/4) x₁ x₂ y₁ y₂) : 3/4 = 3/4 :=
by 
  sorry

end NUMINAMATH_GPT_find_alpha_l1303_130372


namespace NUMINAMATH_GPT_num_quarters_l1303_130361

theorem num_quarters (n q : ℕ) (avg_initial avg_new : ℕ) 
  (h1 : avg_initial = 10) 
  (h2 : avg_new = 12) 
  (h3 : avg_initial * n + 10 = avg_new * (n + 1)) :
  q = 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_num_quarters_l1303_130361


namespace NUMINAMATH_GPT_compare_magnitudes_l1303_130376

noncomputable def log_base_3_of_2 : ℝ := Real.log 2 / Real.log 3   -- def a
noncomputable def ln_2 : ℝ := Real.log 2                          -- def b
noncomputable def five_minus_pi : ℝ := 5 - Real.pi                -- def c

theorem compare_magnitudes :
  let a := log_base_3_of_2
  let b := ln_2
  let c := five_minus_pi
  c < a ∧ a < b :=
by
  sorry

end NUMINAMATH_GPT_compare_magnitudes_l1303_130376


namespace NUMINAMATH_GPT_pentagon_triangle_area_percentage_l1303_130305

def is_equilateral_triangle (s : ℝ) (area : ℝ) : Prop :=
  area = (s^2 * Real.sqrt 3) / 4

def is_square (s : ℝ) (area : ℝ) : Prop :=
  area = s^2

def pentagon_area (square_area triangle_area : ℝ) : ℝ :=
  square_area + triangle_area

noncomputable def percentage (triangle_area pentagon_area : ℝ) : ℝ :=
  (triangle_area / pentagon_area) * 100

theorem pentagon_triangle_area_percentage (s : ℝ) (h₁ : s > 0) :
  let square_area := s^2
  let triangle_area := (s^2 * Real.sqrt 3) / 4
  let pentagon_total_area := pentagon_area square_area triangle_area
  let triangle_percentage := percentage triangle_area pentagon_total_area
  triangle_percentage = (100 * (4 * Real.sqrt 3 - 3) / 13) :=
by
  sorry

end NUMINAMATH_GPT_pentagon_triangle_area_percentage_l1303_130305


namespace NUMINAMATH_GPT_natural_number_sets_solution_l1303_130381

theorem natural_number_sets_solution (x y n : ℕ) (h : (x! + y!) / n! = 3^n) : (x = 0 ∧ y = 2 ∧ n = 1) ∨ (x = 1 ∧ y = 2 ∧ n = 1) :=
by
  sorry

end NUMINAMATH_GPT_natural_number_sets_solution_l1303_130381


namespace NUMINAMATH_GPT_sufficient_not_necessary_condition_l1303_130338

-- Definitions of propositions p and q
def p (x : ℝ) : Prop := abs (x + 1) ≤ 4
def q (x : ℝ) : Prop := x^2 < 5 * x - 6

-- Definitions of negations of p and q
def not_p (x : ℝ) : Prop := x < -5 ∨ x > 3
def not_q (x : ℝ) : Prop := x ≤ 2 ∨ x ≥ 3

-- The theorem to prove
theorem sufficient_not_necessary_condition (x : ℝ) :
  (¬ p x → ¬ q x) ∧ (¬ q x → ¬ p x → False) := 
by
  sorry

end NUMINAMATH_GPT_sufficient_not_necessary_condition_l1303_130338


namespace NUMINAMATH_GPT_total_loaves_served_l1303_130339

-- Given conditions
def wheat_bread := 0.5
def white_bread := 0.4

-- Proof that total loaves served is 0.9
theorem total_loaves_served : wheat_bread + white_bread = 0.9 :=
by sorry

end NUMINAMATH_GPT_total_loaves_served_l1303_130339


namespace NUMINAMATH_GPT_kiwi_lemon_relationship_l1303_130397

open Nat

-- Define the conditions
def total_fruits : ℕ := 58
def mangoes : ℕ := 18
def pears : ℕ := 10
def pawpaws : ℕ := 12
def lemons_in_last_two_baskets : ℕ := 9

-- Define the question and the proof goal
theorem kiwi_lemon_relationship :
  ∃ (kiwis lemons : ℕ), 
  kiwis = lemons_in_last_two_baskets ∧ 
  lemons = lemons_in_last_two_baskets ∧ 
  kiwis + lemons = total_fruits - (mangoes + pears + pawpaws) :=
sorry

end NUMINAMATH_GPT_kiwi_lemon_relationship_l1303_130397


namespace NUMINAMATH_GPT_min_ab_value_l1303_130359

theorem min_ab_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : (1 / a) + (4 / b) = 1) : ab ≥ 16 :=
by
  sorry

end NUMINAMATH_GPT_min_ab_value_l1303_130359


namespace NUMINAMATH_GPT_cos_alpha_beta_half_l1303_130367

open Real

theorem cos_alpha_beta_half (α β : ℝ)
  (h1 : cos (α - β / 2) = -1 / 3)
  (h2 : sin (α / 2 - β) = 1 / 4)
  (h3 : 3 * π / 2 < α ∧ α < 2 * π)
  (h4 : π / 2 < β ∧ β < π) :
  cos ((α + β) / 2) = -(2 * sqrt 2 + sqrt 15) / 12 :=
by
  sorry

end NUMINAMATH_GPT_cos_alpha_beta_half_l1303_130367


namespace NUMINAMATH_GPT_math_problem_l1303_130333

variable {x y z : ℝ}
variable (hx : x > 0) (hy : y > 0) (hz : z > 0)
variable (h : x^2 + y^2 + z^2 = 1)

theorem math_problem : 
  (x / (1 - x^2)) + (y / (1 - y^2)) + (z / (1 - z^2)) ≥ 3 * Real.sqrt 3 / 2 :=
sorry

end NUMINAMATH_GPT_math_problem_l1303_130333


namespace NUMINAMATH_GPT_possible_denominators_count_l1303_130348

variable (a b c : ℕ)
-- Conditions
def is_digit (n : ℕ) : Prop := 0 ≤ n ∧ n ≤ 9
def no_two_zeros (a b c : ℕ) : Prop := ¬(a = 0 ∧ b = 0) ∧ ¬(b = 0 ∧ c = 0) ∧ ¬(a = 0 ∧ c = 0)
def none_is_eight (a b c : ℕ) : Prop := a ≠ 8 ∧ b ≠ 8 ∧ c ≠ 8

-- Theorem
theorem possible_denominators_count : 
  is_digit a ∧ is_digit b ∧ is_digit c ∧ no_two_zeros a b c ∧ none_is_eight a b c →
  ∃ denoms : Finset ℕ, denoms.card = 7 ∧ ∀ d ∈ denoms, 999 % d = 0 :=
by
  sorry

end NUMINAMATH_GPT_possible_denominators_count_l1303_130348


namespace NUMINAMATH_GPT_probability_same_gender_l1303_130337

theorem probability_same_gender :
  let males := 3
  let females := 2
  let total := males + females
  let total_ways := Nat.choose total 2
  let male_ways := Nat.choose males 2
  let female_ways := Nat.choose females 2
  let same_gender_ways := male_ways + female_ways
  let probability := (same_gender_ways : ℚ) / total_ways
  probability = 2 / 5 :=
by
  sorry

end NUMINAMATH_GPT_probability_same_gender_l1303_130337


namespace NUMINAMATH_GPT_average_eq_solution_l1303_130315

theorem average_eq_solution (x : ℝ) :
  (1 / 3) * ((2 * x + 4) + (4 * x + 6) + (5 * x + 3)) = 3 * x + 5 → x = 1 :=
by
  sorry

end NUMINAMATH_GPT_average_eq_solution_l1303_130315


namespace NUMINAMATH_GPT_perimeter_of_photo_l1303_130317

theorem perimeter_of_photo 
  (frame_width : ℕ)
  (frame_area : ℕ)
  (outer_edge_length : ℕ)
  (photo_perimeter : ℕ) :
  frame_width = 2 → 
  frame_area = 48 → 
  outer_edge_length = 10 →
  photo_perimeter = 16 :=
by
  intros h1 h2 h3
  -- proof goes here
  sorry

end NUMINAMATH_GPT_perimeter_of_photo_l1303_130317


namespace NUMINAMATH_GPT_fraction_equivalence_l1303_130379

theorem fraction_equivalence : (8 : ℝ) / (5 * 48) = 0.8 / (5 * 0.48) :=
  sorry

end NUMINAMATH_GPT_fraction_equivalence_l1303_130379


namespace NUMINAMATH_GPT_exists_positive_int_n_l1303_130357

theorem exists_positive_int_n (p a k : ℕ) 
  (hp : Nat.Prime p) (ha : 0 < a) (hk1 : p^a < k) (hk2 : k < 2 * p^a) :
  ∃ n : ℕ, n < p^(2 * a) ∧ (Nat.choose n k ≡ n [MOD p^a]) ∧ (n ≡ k [MOD p^a]) :=
sorry

end NUMINAMATH_GPT_exists_positive_int_n_l1303_130357


namespace NUMINAMATH_GPT_basketball_success_rate_l1303_130363

theorem basketball_success_rate (p : ℝ) (h : 1 - p^2 = 16 / 25) : p = 3 / 5 :=
sorry

end NUMINAMATH_GPT_basketball_success_rate_l1303_130363
