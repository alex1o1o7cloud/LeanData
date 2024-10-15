import Mathlib

namespace NUMINAMATH_GPT_problem_part1_problem_part2_l1155_115523

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x + 1) / Real.log a
noncomputable def g (a : ℝ) (t : ℝ) (x : ℝ) : ℝ := 2 * Real.log (2 * x + t) / Real.log a

theorem problem_part1 (a t : ℝ) (ha_pos : a > 0) (ha_ne_one : a ≠ 1) :
  f a 1 - g a t 1 = 0 → t = -2 + Real.sqrt 2 :=
sorry

theorem problem_part2 (a t : ℝ) (ha_bound : 0 < a ∧ a < 1) :
  (∀ x, 0 ≤ x ∧ x ≤ 15 → f a x ≥ g a t x) → t ≥ 1 :=
sorry

end NUMINAMATH_GPT_problem_part1_problem_part2_l1155_115523


namespace NUMINAMATH_GPT_find_y_l1155_115528

theorem find_y (x y : ℝ) (h1 : x = 4 * y) (h2 : (1 / 2) * x = 1) : y = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_y_l1155_115528


namespace NUMINAMATH_GPT_printer_diff_l1155_115569

theorem printer_diff (A B : ℚ) (hA : A * 60 = 35) (hAB : (A + B) * 24 = 35) : B - A = 7 / 24 := by
  sorry

end NUMINAMATH_GPT_printer_diff_l1155_115569


namespace NUMINAMATH_GPT_hands_in_class_not_including_peters_l1155_115525

def total_students : ℕ := 11
def hands_per_student : ℕ := 2
def peter_hands : ℕ := 2

theorem hands_in_class_not_including_peters :  (total_students * hands_per_student) - peter_hands = 20 :=
by
  sorry

end NUMINAMATH_GPT_hands_in_class_not_including_peters_l1155_115525


namespace NUMINAMATH_GPT_garden_breadth_l1155_115519

theorem garden_breadth (perimeter length breadth : ℕ) 
    (h₁ : perimeter = 680)
    (h₂ : length = 258)
    (h₃ : perimeter = 2 * (length + breadth)) : 
    breadth = 82 := 
sorry

end NUMINAMATH_GPT_garden_breadth_l1155_115519


namespace NUMINAMATH_GPT_find_n_mod_11_l1155_115518

theorem find_n_mod_11 : ∃ n : ℕ, 0 ≤ n ∧ n ≤ 10 ∧ n ≡ 50000 [MOD 11] ∧ n = 5 :=
sorry

end NUMINAMATH_GPT_find_n_mod_11_l1155_115518


namespace NUMINAMATH_GPT_negation_of_p_l1155_115515

variable (x : ℝ)

-- Define the original proposition p
def p := ∀ x, x^2 < 1 → x < 1

-- Define the negation of p
def neg_p := ∃ x₀, x₀^2 ≥ 1 ∧ x₀ < 1

-- State the theorem that negates p
theorem negation_of_p : ¬ p ↔ neg_p :=
by
  sorry

end NUMINAMATH_GPT_negation_of_p_l1155_115515


namespace NUMINAMATH_GPT_mailman_junk_mail_l1155_115596

variable (junk_mail_per_house : ℕ) (houses_per_block : ℕ)

theorem mailman_junk_mail (h1 : junk_mail_per_house = 2) (h2 : houses_per_block = 7) :
  junk_mail_per_house * houses_per_block = 14 :=
by
  sorry

end NUMINAMATH_GPT_mailman_junk_mail_l1155_115596


namespace NUMINAMATH_GPT_terry_lunch_combos_l1155_115561

def num_lettuce : ℕ := 2
def num_tomatoes : ℕ := 3
def num_olives : ℕ := 4
def num_soups : ℕ := 2

theorem terry_lunch_combos : num_lettuce * num_tomatoes * num_olives * num_soups = 48 :=
by
  sorry

end NUMINAMATH_GPT_terry_lunch_combos_l1155_115561


namespace NUMINAMATH_GPT_distance_Xiaolan_to_Xiaohong_reverse_l1155_115586

def Xiaohong_to_Xiaolan := 30
def Xiaolu_to_Xiaohong := 26
def Xiaolan_to_Xiaolu := 28

def total_perimeter : ℕ := Xiaohong_to_Xiaolan + Xiaolan_to_Xiaolu + Xiaolu_to_Xiaohong

theorem distance_Xiaolan_to_Xiaohong_reverse : total_perimeter - Xiaohong_to_Xiaolan = 54 :=
by
  rw [total_perimeter]
  norm_num
  sorry

end NUMINAMATH_GPT_distance_Xiaolan_to_Xiaohong_reverse_l1155_115586


namespace NUMINAMATH_GPT_edwards_final_money_l1155_115529

def small_lawn_rate : ℕ := 8
def medium_lawn_rate : ℕ := 12
def large_lawn_rate : ℕ := 15

def first_garden_rate : ℕ := 10
def second_garden_rate : ℕ := 12
def additional_garden_rate : ℕ := 15

def num_small_lawns : ℕ := 3
def num_medium_lawns : ℕ := 1
def num_large_lawns : ℕ := 1
def num_gardens_cleaned : ℕ := 5

def fuel_expense : ℕ := 10
def equipment_rental_expense : ℕ := 15
def initial_savings : ℕ := 7

theorem edwards_final_money : 
  (num_small_lawns * small_lawn_rate + 
   num_medium_lawns * medium_lawn_rate + 
   num_large_lawns * large_lawn_rate + 
   (first_garden_rate + second_garden_rate + (num_gardens_cleaned - 2) * additional_garden_rate) + 
   initial_savings - 
   (fuel_expense + equipment_rental_expense)) = 100 := 
  by 
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_edwards_final_money_l1155_115529


namespace NUMINAMATH_GPT_repeating_decimal_to_fraction_l1155_115582

theorem repeating_decimal_to_fraction : (∃ (x : ℚ), x = 0.4 + 4 / 9) :=
sorry

end NUMINAMATH_GPT_repeating_decimal_to_fraction_l1155_115582


namespace NUMINAMATH_GPT_most_prolific_mathematician_is_euler_l1155_115580

noncomputable def prolific_mathematician (collected_works_volume_count: ℕ) (publishing_organization: String) : String :=
  if collected_works_volume_count > 75 ∧ publishing_organization = "Swiss Society of Natural Sciences" then
    "Leonhard Euler"
  else
    "Unknown"

theorem most_prolific_mathematician_is_euler :
  prolific_mathematician 76 "Swiss Society of Natural Sciences" = "Leonhard Euler" :=
by
  sorry

end NUMINAMATH_GPT_most_prolific_mathematician_is_euler_l1155_115580


namespace NUMINAMATH_GPT_max_integer_a_for_real_roots_l1155_115506

theorem max_integer_a_for_real_roots (a : ℤ) :
  (((a - 1) * x^2 - 2 * x + 3 = 0) ∧ a ≠ 1) → a ≤ 0 ∧ (∀ b : ℤ, ((b - 1) * x^2 - 2 * x + 3 = 0) ∧ a ≠ 1 → b ≤ 0) :=
sorry

end NUMINAMATH_GPT_max_integer_a_for_real_roots_l1155_115506


namespace NUMINAMATH_GPT_exists_infinite_solutions_l1155_115553

noncomputable def infinite_solutions_exist (m : ℕ) : Prop := 
  ∃ (a b c : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧  (1 / a + 1 / b + 1 / c + 1 / (a * b * c) = m / (a + b + c))

theorem exists_infinite_solutions : infinite_solutions_exist 12 :=
  sorry

end NUMINAMATH_GPT_exists_infinite_solutions_l1155_115553


namespace NUMINAMATH_GPT_smallest_k_l1155_115502

theorem smallest_k :
  ∃ k : ℤ, k > 1 ∧ k % 13 = 1 ∧ k % 8 = 1 ∧ k % 4 = 1 ∧ k = 105 :=
by
  sorry

end NUMINAMATH_GPT_smallest_k_l1155_115502


namespace NUMINAMATH_GPT_area_of_sector_l1155_115524

theorem area_of_sector (r : ℝ) (n : ℝ) (h_r : r = 3) (h_n : n = 120) : 
  (n / 360) * π * r^2 = 3 * π :=
by
  rw [h_r, h_n] -- Plugin in the given values first
  norm_num     -- Normalize numerical expressions
  sorry        -- Placeholder for further simplification if needed. 

end NUMINAMATH_GPT_area_of_sector_l1155_115524


namespace NUMINAMATH_GPT_inequality_holds_l1155_115544

variable (b : ℝ)

theorem inequality_holds (b : ℝ) : (3 * b - 1) * (4 * b + 1) > (2 * b + 1) * (5 * b - 3) :=
by
  sorry

end NUMINAMATH_GPT_inequality_holds_l1155_115544


namespace NUMINAMATH_GPT_find_sale_month_4_l1155_115516

-- Define the given sales data
def sale_month_1: ℕ := 5124
def sale_month_2: ℕ := 5366
def sale_month_3: ℕ := 5808
def sale_month_5: ℕ := 6124
def sale_month_6: ℕ := 4579
def average_sale_per_month: ℕ := 5400

-- Define the goal: Sale in the fourth month
def sale_month_4: ℕ := 5399

-- Prove that the total sales conforms to the given average sale
theorem find_sale_month_4 :
  sale_month_1 + sale_month_2 + sale_month_3 + sale_month_4 + sale_month_5 + sale_month_6 = 6 * average_sale_per_month :=
by
  sorry

end NUMINAMATH_GPT_find_sale_month_4_l1155_115516


namespace NUMINAMATH_GPT_polygon_diagonalization_l1155_115507

theorem polygon_diagonalization (n : ℕ) (h : n ≥ 3) : 
  ∃ (triangles : ℕ), triangles = n - 2 ∧ 
  (∀ (polygons : ℕ), 3 ≤ polygons → polygons < n → ∃ k, k = polygons - 2) := 
by {
  -- base case
  sorry
}

end NUMINAMATH_GPT_polygon_diagonalization_l1155_115507


namespace NUMINAMATH_GPT_sqrt_sum_inequality_l1155_115568

variable (a b c d : ℝ)

theorem sqrt_sum_inequality
  (h1 : 0 < a)
  (h2 : a < b)
  (h3 : b < c)
  (h4 : c < d)
  (h5 : a + d = b + c) :
  Real.sqrt a + Real.sqrt d < Real.sqrt b + Real.sqrt c :=
by
  sorry

end NUMINAMATH_GPT_sqrt_sum_inequality_l1155_115568


namespace NUMINAMATH_GPT_gcd_of_a_and_b_is_one_l1155_115503

theorem gcd_of_a_and_b_is_one {a b : ℕ} (h1 : a > b) (h2 : Nat.gcd (a + b) (a - b) = 1) : Nat.gcd a b = 1 :=
by
  sorry

end NUMINAMATH_GPT_gcd_of_a_and_b_is_one_l1155_115503


namespace NUMINAMATH_GPT_ants_no_collision_probability_l1155_115584

-- Definitions
def cube_vertices : Finset ℕ := {0, 1, 2, 3, 4, 5, 6, 7}

def adjacent (v : ℕ) : Finset ℕ :=
  match v with
  | 0 => {1, 3, 4}
  | 1 => {0, 2, 5}
  | 2 => {1, 3, 6}
  | 3 => {0, 2, 7}
  | 4 => {0, 5, 7}
  | 5 => {1, 4, 6}
  | 6 => {2, 5, 7}
  | 7 => {3, 4, 6}
  | _ => ∅

-- Hypothesis: Each ant moves independently to one of the three adjacent vertices.

-- Result to prove
def X : ℕ := sorry  -- The number of valid ways ants can move without collisions

theorem ants_no_collision_probability : 
  ∃ X, (X / (3 : ℕ)^8 = X / 6561) :=
  by
    sorry

end NUMINAMATH_GPT_ants_no_collision_probability_l1155_115584


namespace NUMINAMATH_GPT_ticket_price_l1155_115541

variable (x : ℝ)

def tickets_condition1 := 3 * x
def tickets_condition2 := 5 * x
def total_spent := 3 * x + 5 * x

theorem ticket_price : total_spent x = 32 → x = 4 :=
by
  -- Proof steps will be provided here.
  sorry

end NUMINAMATH_GPT_ticket_price_l1155_115541


namespace NUMINAMATH_GPT_opposite_of_neg_2023_l1155_115543

theorem opposite_of_neg_2023 : -(-2023) = 2023 := 
by 
  sorry

end NUMINAMATH_GPT_opposite_of_neg_2023_l1155_115543


namespace NUMINAMATH_GPT_g_at_5_l1155_115530

-- Define the function g(x) that satisfies the given condition
def g (x : ℝ) : ℝ := sorry

-- Axiom stating that the function g satisfies the given equation for all x ∈ ℝ
axiom g_condition : ∀ x : ℝ, 3 * g x + 4 * g (1 - x) = 6 * x^2

-- The theorem to prove
theorem g_at_5 : g 5 = -66 / 7 :=
by
  -- Proof will be added here.
  sorry

end NUMINAMATH_GPT_g_at_5_l1155_115530


namespace NUMINAMATH_GPT_determine_m_l1155_115573

theorem determine_m (a b : ℝ) (m : ℝ) :
  (2 * (a ^ 2 - 2 * a * b - b ^ 2) - (a ^ 2 + m * a * b + 2 * b ^ 2)) = a ^ 2 - (4 + m) * a * b - 4 * b ^ 2 →
  ¬(∃ (c : ℝ), (a ^ 2 - (4 + m) * a * b - 4 * b ^ 2) = a ^ 2 + c * (a * b) + k) →
  m = -4 :=
sorry

end NUMINAMATH_GPT_determine_m_l1155_115573


namespace NUMINAMATH_GPT_abs_x_minus_1_le_1_is_equivalent_to_x_le_2_l1155_115563

theorem abs_x_minus_1_le_1_is_equivalent_to_x_le_2 (x : ℝ) :
  (|x - 1| ≤ 1) ↔ (x ≤ 2) := sorry

end NUMINAMATH_GPT_abs_x_minus_1_le_1_is_equivalent_to_x_le_2_l1155_115563


namespace NUMINAMATH_GPT_no_value_of_a_l1155_115517

theorem no_value_of_a (a : ℝ) (x y : ℝ) : ¬∃ x1 x2 : ℝ, (x1 ≠ x2) ∧ (x1^2 + y^2 + 2 * x1 = abs (x1 - a) - 1) ∧ (x2^2 + y^2 + 2 * x2 = abs (x2 - a) - 1) := 
by
  sorry

end NUMINAMATH_GPT_no_value_of_a_l1155_115517


namespace NUMINAMATH_GPT_geom_seq_q_eq_l1155_115592

theorem geom_seq_q_eq (a1 : ℕ := 2) (S3 : ℕ := 26) 
  (h1 : a1 = 2) 
  (h2 : S3 = 26) : 
  ∃ q : ℝ, (q = 3 ∨ q = -4) := by
  sorry

end NUMINAMATH_GPT_geom_seq_q_eq_l1155_115592


namespace NUMINAMATH_GPT_valid_m_values_l1155_115535

theorem valid_m_values :
  ∃ (m : ℕ), (m ∣ 720) ∧ (m ≠ 1) ∧ (m ≠ 720) ∧ ((720 / m) > 1) ∧ ((30 - 2) = 28) := 
sorry

end NUMINAMATH_GPT_valid_m_values_l1155_115535


namespace NUMINAMATH_GPT_find_x_l1155_115557

theorem find_x (x : ℝ) (hx_pos : 0 < x) (h: (x / 100) * x = 4) : x = 20 := by
  sorry

end NUMINAMATH_GPT_find_x_l1155_115557


namespace NUMINAMATH_GPT_path_area_and_cost_l1155_115504

theorem path_area_and_cost:
  let length_grass_field := 75
  let width_grass_field := 55
  let path_width := 3.5
  let cost_per_sq_meter := 2
  let length_with_path := length_grass_field + 2 * path_width
  let width_with_path := width_grass_field + 2 * path_width
  let area_with_path := length_with_path * width_with_path
  let area_grass_field := length_grass_field * width_grass_field
  let area_path := area_with_path - area_grass_field
  let cost_of_construction := area_path * cost_per_sq_meter
  area_path = 959 ∧ cost_of_construction = 1918 :=
by
  sorry

end NUMINAMATH_GPT_path_area_and_cost_l1155_115504


namespace NUMINAMATH_GPT_quadratic_no_real_roots_l1155_115540

theorem quadratic_no_real_roots (a b c d : ℝ)  :
  a^2 - 4 * b < 0 → c^2 - 4 * d < 0 → ( (a + c) / 2 )^2 - 4 * ( (b + d) / 2 ) < 0 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_no_real_roots_l1155_115540


namespace NUMINAMATH_GPT_simplify_frac_l1155_115538

theorem simplify_frac :
  (1 / (1 / (Real.sqrt 3 + 2) + 2 / (Real.sqrt 5 - 2))) = 
  (Real.sqrt 3 - 2 * Real.sqrt 5 - 2) :=
by
  sorry

end NUMINAMATH_GPT_simplify_frac_l1155_115538


namespace NUMINAMATH_GPT_ratio_of_flowers_given_l1155_115594

-- Definitions based on conditions
def Collin_flowers : ℕ := 25
def Ingrid_flowers_initial : ℕ := 33
def petals_per_flower : ℕ := 4
def Collin_petals_total : ℕ := 144

-- The ratio of the number of flowers Ingrid gave to Collin to the number of flowers Ingrid had initially
theorem ratio_of_flowers_given :
  let Ingrid_flowers_given := (Collin_petals_total - (Collin_flowers * petals_per_flower)) / petals_per_flower
  let ratio := Ingrid_flowers_given / Ingrid_flowers_initial
  ratio = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_flowers_given_l1155_115594


namespace NUMINAMATH_GPT_miles_total_instruments_l1155_115574

theorem miles_total_instruments :
  let fingers := 10
  let hands := 2
  let heads := 1
  let trumpets := fingers - 3
  let guitars := hands + 2
  let trombones := heads + 2
  let french_horns := guitars - 1
  (trumpets + guitars + trombones + french_horns) = 17 :=
by
  sorry

end NUMINAMATH_GPT_miles_total_instruments_l1155_115574


namespace NUMINAMATH_GPT_ellipse_focus_xaxis_l1155_115520

theorem ellipse_focus_xaxis (k : ℝ) (h : 1 - k > 2 + k ∧ 2 + k > 0) : -2 < k ∧ k < -1/2 :=
by sorry

end NUMINAMATH_GPT_ellipse_focus_xaxis_l1155_115520


namespace NUMINAMATH_GPT_remaining_bollards_to_be_installed_l1155_115556

-- Definitions based on the problem's conditions
def total_bollards_per_side := 4000
def sides := 2
def total_bollards := total_bollards_per_side * sides
def installed_fraction := 3 / 4
def installed_bollards := installed_fraction * total_bollards
def remaining_bollards := total_bollards - installed_bollards

-- The statement to be proved
theorem remaining_bollards_to_be_installed : remaining_bollards = 2000 :=
  by sorry

end NUMINAMATH_GPT_remaining_bollards_to_be_installed_l1155_115556


namespace NUMINAMATH_GPT_average_rainfall_correct_l1155_115551

-- Define the monthly rainfall
def january_rainfall := 150
def february_rainfall := 200
def july_rainfall := 366
def other_months_rainfall := 100

-- Calculate total yearly rainfall
def total_yearly_rainfall := 
  january_rainfall + 
  february_rainfall + 
  july_rainfall + 
  (9 * other_months_rainfall)

-- Calculate total hours in a year
def days_per_month := 30
def total_days_in_year := 12 * days_per_month
def hours_per_day := 24
def total_hours_in_year := total_days_in_year * hours_per_day

-- Calculate average rainfall per hour
def average_rainfall_per_hour := 
  total_yearly_rainfall / total_hours_in_year

theorem average_rainfall_correct :
  average_rainfall_per_hour = (101 / 540) := sorry

end NUMINAMATH_GPT_average_rainfall_correct_l1155_115551


namespace NUMINAMATH_GPT_sin_double_angle_l1155_115579

theorem sin_double_angle (A : ℝ) (h1 : π / 2 < A) (h2 : A < π) (h3 : Real.sin A = 4 / 5) : Real.sin (2 * A) = -24 / 25 := 
by 
  sorry

end NUMINAMATH_GPT_sin_double_angle_l1155_115579


namespace NUMINAMATH_GPT_trains_cross_time_l1155_115508

theorem trains_cross_time 
  (len_train1 len_train2 : ℕ) 
  (speed_train1_kmph speed_train2_kmph : ℕ) 
  (len_train1_eq : len_train1 = 200) 
  (len_train2_eq : len_train2 = 300) 
  (speed_train1_eq : speed_train1_kmph = 70) 
  (speed_train2_eq : speed_train2_kmph = 50) 
  : (500 / (120 * 1000 / 3600)) = 15 := 
by sorry

end NUMINAMATH_GPT_trains_cross_time_l1155_115508


namespace NUMINAMATH_GPT_polynomial_proof_l1155_115565

theorem polynomial_proof (x : ℝ) : 
  (2 * x^2 + 5 * x + 4) = (2 * x^2 + 5 * x - 2) + (10 * x + 6) :=
by sorry

end NUMINAMATH_GPT_polynomial_proof_l1155_115565


namespace NUMINAMATH_GPT_minimum_participants_l1155_115539

theorem minimum_participants 
  (x y z : ℕ)
  (h_andrei : 3 * x + 1 = 61)
  (h_dima : 4 * y + 1 = 61)
  (h_lenya : 5 * z + 1 = 61) : 
  x = 20 ∧ y = 15 ∧ z = 12 :=
by
  sorry

end NUMINAMATH_GPT_minimum_participants_l1155_115539


namespace NUMINAMATH_GPT_survey_respondents_l1155_115546

theorem survey_respondents (X Y : ℕ) (hX : X = 150) (ratio : X = 5 * Y) : X + Y = 180 :=
by
  sorry

end NUMINAMATH_GPT_survey_respondents_l1155_115546


namespace NUMINAMATH_GPT_avg_percentage_decrease_l1155_115513

theorem avg_percentage_decrease (x : ℝ) 
  (h : 16 * (1 - x)^2 = 9) : x = 0.25 :=
sorry

end NUMINAMATH_GPT_avg_percentage_decrease_l1155_115513


namespace NUMINAMATH_GPT_student_l1155_115567

theorem student's_incorrect_answer (D I : ℕ) (h1 : D / 36 = 58) (h2 : D / 87 = I) : I = 24 :=
sorry

end NUMINAMATH_GPT_student_l1155_115567


namespace NUMINAMATH_GPT_colleen_pencils_l1155_115575

theorem colleen_pencils (joy_pencils : ℕ) (pencil_cost : ℕ) (extra_cost : ℕ) (colleen_paid : ℕ)
  (H1 : joy_pencils = 30)
  (H2 : pencil_cost = 4)
  (H3 : extra_cost = 80)
  (H4 : colleen_paid = (joy_pencils * pencil_cost) + extra_cost) :
  colleen_paid / pencil_cost = 50 := 
by 
  -- Hints, if necessary
sorry

end NUMINAMATH_GPT_colleen_pencils_l1155_115575


namespace NUMINAMATH_GPT_rank_of_A_l1155_115527

def A : Matrix (Fin 3) (Fin 5) ℝ :=
  ![![1, 2, 3, 5, 8],
    ![0, 1, 4, 6, 9],
    ![0, 0, 1, 7, 10]]

theorem rank_of_A : A.rank = 3 :=
by sorry

end NUMINAMATH_GPT_rank_of_A_l1155_115527


namespace NUMINAMATH_GPT_not_all_on_C_implies_exists_not_on_C_l1155_115599

def F (x y : ℝ) : Prop := sorry  -- Define F according to specifics
def on_curve_C (x y : ℝ) : Prop := sorry -- Define what it means to be on curve C according to specifics

theorem not_all_on_C_implies_exists_not_on_C (h : ¬ ∀ x y : ℝ, F x y → on_curve_C x y) :
  ∃ x y : ℝ, F x y ∧ ¬ on_curve_C x y := sorry

end NUMINAMATH_GPT_not_all_on_C_implies_exists_not_on_C_l1155_115599


namespace NUMINAMATH_GPT_actual_distance_traveled_l1155_115547

theorem actual_distance_traveled (D : ℝ) (T : ℝ) (h1 : D = 15 * T) (h2 : D + 35 = 25 * T) : D = 52.5 := 
by
  sorry

end NUMINAMATH_GPT_actual_distance_traveled_l1155_115547


namespace NUMINAMATH_GPT_binary_multiplication_l1155_115590

/-- 
Calculate the product of two binary numbers and validate the result.
Given:
  a = 1101 in base 2,
  b = 111 in base 2,
Prove:
  a * b = 1011110 in base 2. 
-/
theorem binary_multiplication : 
  let a := 0b1101
  let b := 0b111
  a * b = 0b1011110 :=
by
  sorry

end NUMINAMATH_GPT_binary_multiplication_l1155_115590


namespace NUMINAMATH_GPT_cos_alpha_second_quadrant_l1155_115576

variable (α : Real)
variable (h₁ : α ∈ Set.Ioo (π / 2) π)
variable (h₂ : Real.sin α = 5 / 13)

theorem cos_alpha_second_quadrant : Real.cos α = -12 / 13 := by
  sorry

end NUMINAMATH_GPT_cos_alpha_second_quadrant_l1155_115576


namespace NUMINAMATH_GPT_intersection_with_y_axis_l1155_115578

theorem intersection_with_y_axis :
  (∃ y : ℝ, y = -(0 + 2)^2 + 6 ∧ (0, y) = (0, 2)) :=
by
  sorry

end NUMINAMATH_GPT_intersection_with_y_axis_l1155_115578


namespace NUMINAMATH_GPT_proof_problem_l1155_115512

variable (α β : ℝ) (a b : ℝ × ℝ) (m : ℝ)
variable (hα : 0 < α ∧ α < Real.pi / 4)
variable (hβ : β = Real.pi)
variable (ha_def : a = (Real.tan (α + β / 4) - 1, 0))
variable (hb : b = (Real.cos α, 2))
variable (ha_dot : a.1 * b.1 + a.2 * b.2 = m)

-- Proof statement
theorem proof_problem :
  (0 < α ∧ α < Real.pi / 4) ∧
  β = Real.pi ∧
  a = (Real.tan (α + β / 4) - 1, 0) ∧
  b = (Real.cos α, 2) ∧
  (a.1 * b.1 + a.2 * b.2 = m) →
  (2 * Real.cos α * Real.cos α + Real.sin (2 * (α + β))) / (Real.cos α - Real.sin β) = 2 * (m + 2) := by
  sorry

end NUMINAMATH_GPT_proof_problem_l1155_115512


namespace NUMINAMATH_GPT_exists_subset_no_double_l1155_115572

theorem exists_subset_no_double (s : Finset ℕ) (h₁ : s = Finset.range 3000) :
  ∃ t : Finset ℕ, t.card = 2000 ∧ (∀ x ∈ t, ∀ y ∈ t, x ≠ 2 * y ∧ y ≠ 2 * x) :=
by
  sorry

end NUMINAMATH_GPT_exists_subset_no_double_l1155_115572


namespace NUMINAMATH_GPT_burrito_count_l1155_115501

def burrito_orders (wraps beef_fillings chicken_fillings : ℕ) :=
  if wraps = 5 ∧ beef_fillings >= 4 ∧ chicken_fillings >= 3 then 25 else 0

theorem burrito_count : burrito_orders 5 4 3 = 25 := by
  sorry

end NUMINAMATH_GPT_burrito_count_l1155_115501


namespace NUMINAMATH_GPT_machine_b_finishes_in_12_hours_l1155_115581

noncomputable def machine_b_time : ℝ :=
  let rA := 1 / 4  -- rate of Machine A
  let rC := 1 / 6  -- rate of Machine C
  let rTotalTogether := 1 / 2  -- rate of all machines working together
  let rB := (rTotalTogether - rA - rC)  -- isolate the rate of Machine B
  1 / rB  -- time for Machine B to finish the job

theorem machine_b_finishes_in_12_hours : machine_b_time = 12 :=
by
  sorry

end NUMINAMATH_GPT_machine_b_finishes_in_12_hours_l1155_115581


namespace NUMINAMATH_GPT_value_of_expression_l1155_115560

variable (a b : ℝ)

def system_of_equations : Prop :=
  (2 * a - b = 12) ∧ (a + 2 * b = 8)

theorem value_of_expression (h : system_of_equations a b) : 3 * a + b = 20 :=
  sorry

end NUMINAMATH_GPT_value_of_expression_l1155_115560


namespace NUMINAMATH_GPT_problem_l1155_115549

theorem problem (A B : ℝ) (h₀ : 0 < A) (h₁ : 0 < B) (h₂ : B > A) (n c : ℝ) 
  (h₃ : B = A * (1 + n / 100)) (h₄ : A = B * (1 - c / 100)) :
  A * Real.sqrt (100 + n) = B * Real.sqrt (100 - c) :=
by
  sorry

end NUMINAMATH_GPT_problem_l1155_115549


namespace NUMINAMATH_GPT_range_of_k_for_empty_solution_set_l1155_115552

theorem range_of_k_for_empty_solution_set :
  ∀ (k : ℝ), (∀ (x : ℝ), k * x^2 - 2 * |x - 1| + 3 * k < 0 → False) ↔ k ≥ 1 :=
by sorry

end NUMINAMATH_GPT_range_of_k_for_empty_solution_set_l1155_115552


namespace NUMINAMATH_GPT_total_jumps_correct_l1155_115597

-- Define Ronald's jumps
def Ronald_jumps : ℕ := 157

-- Define the difference in jumps between Rupert and Ronald
def difference : ℕ := 86

-- Define Rupert's jumps
def Rupert_jumps : ℕ := Ronald_jumps + difference

-- Define the total number of jumps
def total_jumps : ℕ := Ronald_jumps + Rupert_jumps

-- State the main theorem we want to prove
theorem total_jumps_correct : total_jumps = 400 := 
by sorry

end NUMINAMATH_GPT_total_jumps_correct_l1155_115597


namespace NUMINAMATH_GPT_randy_mango_trees_l1155_115595

theorem randy_mango_trees (M C : ℕ) 
  (h1 : C = M / 2 - 5) 
  (h2 : M + C = 85) : 
  M = 60 := 
sorry

end NUMINAMATH_GPT_randy_mango_trees_l1155_115595


namespace NUMINAMATH_GPT_jason_initial_cards_l1155_115585

-- Definitions based on conditions
def cards_given_away : ℕ := 4
def cards_left : ℕ := 5

-- Theorem to prove
theorem jason_initial_cards : cards_given_away + cards_left = 9 :=
by sorry

end NUMINAMATH_GPT_jason_initial_cards_l1155_115585


namespace NUMINAMATH_GPT_smallest_portion_is_five_thirds_l1155_115587

theorem smallest_portion_is_five_thirds
    (a1 a2 a3 a4 a5 : ℚ)
    (h1 : a2 = a1 + 1)
    (h2 : a3 = a1 + 2)
    (h3 : a4 = a1 + 3)
    (h4 : a5 = a1 + 4)
    (h_sum : a1 + a2 + a3 + a4 + a5 = 100)
    (h_cond : (1 / 7) * (a3 + a4 + a5) = a1 + a2) :
    a1 = 5 / 3 :=
by
  sorry

end NUMINAMATH_GPT_smallest_portion_is_five_thirds_l1155_115587


namespace NUMINAMATH_GPT_mary_cut_10_roses_l1155_115591

-- Define the initial and final number of roses
def initial_roses : ℕ := 6
def final_roses : ℕ := 16

-- Define the number of roses cut as the difference between final and initial
def roses_cut : ℕ :=
  final_roses - initial_roses

-- Theorem stating the number of roses cut by Mary
theorem mary_cut_10_roses : roses_cut = 10 := by
  sorry

end NUMINAMATH_GPT_mary_cut_10_roses_l1155_115591


namespace NUMINAMATH_GPT_lily_has_26_dollars_left_for_coffee_l1155_115537

-- Define the initial amount of money Lily has
def initialMoney : ℕ := 60

-- Define the costs of items
def celeryCost : ℕ := 5
def cerealCost : ℕ := 12 / 2
def breadCost : ℕ := 8
def milkCost : ℕ := 10 * 9 / 10
def potatoCostEach : ℕ := 1
def numberOfPotatoes : ℕ := 6
def totalPotatoCost : ℕ := potatoCostEach * numberOfPotatoes

-- Define the total amount spent on the items
def totalSpent : ℕ := celeryCost + cerealCost + breadCost + milkCost + totalPotatoCost

-- Define the amount left for coffee
def amountLeftForCoffee : ℕ := initialMoney - totalSpent

-- The theorem to prove
theorem lily_has_26_dollars_left_for_coffee :
  amountLeftForCoffee = 26 := by
  sorry

end NUMINAMATH_GPT_lily_has_26_dollars_left_for_coffee_l1155_115537


namespace NUMINAMATH_GPT_remainder_is_3_l1155_115533

-- Define the polynomial p(x)
def p (x : ℝ) := x^3 - 3 * x + 5

-- Define the divisor d(x)
def d (x : ℝ) := x - 1

-- The theorem: remainder when p(x) is divided by d(x)
theorem remainder_is_3 : p 1 = 3 := by 
  sorry

end NUMINAMATH_GPT_remainder_is_3_l1155_115533


namespace NUMINAMATH_GPT_omega_range_l1155_115522

theorem omega_range (ω : ℝ) (a b : ℝ) (hω_pos : ω > 0) (h_range : π ≤ a ∧ a < b ∧ b ≤ 2 * π)
  (h_sin : Real.sin (ω * a) + Real.sin (ω * b) = 2) :
  ω ∈ Set.Icc (9 / 4 : ℝ) (5 / 2) ∪ Set.Ici (13 / 4) :=
by
  sorry

end NUMINAMATH_GPT_omega_range_l1155_115522


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_false_statement_l1155_115588

theorem arithmetic_sequence_sum_false_statement (a_n : ℕ → ℝ) (S : ℕ → ℝ)
  (h_arithmetic : ∀ n, a_n n.succ - a_n n = a_n 1 - a_n 0)
  (h_S : ∀ n, S n = (n + 1) * a_n 0 + (n * (n + 1) * (a_n 1 - a_n 0)) / 2)
  (h1 : S 6 < S 7) (h2 : S 7 = S 8) (h3 : S 8 > S 9) : ¬ (S 10 > S 6) :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_false_statement_l1155_115588


namespace NUMINAMATH_GPT_taylor_scores_l1155_115526

theorem taylor_scores 
    (yellow_scores : ℕ := 78)
    (white_ratio : ℕ := 7)
    (black_ratio : ℕ := 6)
    (total_ratio : ℕ := white_ratio + black_ratio)
    (difference_ratio : ℕ := white_ratio - black_ratio) :
    (2/3 : ℚ) * (yellow_scores * difference_ratio / total_ratio) = 4 := by
  sorry

end NUMINAMATH_GPT_taylor_scores_l1155_115526


namespace NUMINAMATH_GPT_Barbier_theorem_for_delta_curves_l1155_115571

def delta_curve (h : ℝ) : Type := sorry 
def can_rotate_freely_in_3gon (K : delta_curve h) : Prop := sorry
def length_of_curve (K : delta_curve h) : ℝ := sorry

theorem Barbier_theorem_for_delta_curves
  (K : delta_curve h)
  (h : ℝ)
  (H : can_rotate_freely_in_3gon K)
  : length_of_curve K = (2 * Real.pi * h) / 3 := 
sorry

end NUMINAMATH_GPT_Barbier_theorem_for_delta_curves_l1155_115571


namespace NUMINAMATH_GPT_sally_initial_orange_balloons_l1155_115548

variable (initial_orange_balloons : ℕ)  -- The initial number of orange balloons Sally had
variable (lost_orange_balloons : ℕ := 2)  -- The number of orange balloons Sally lost
variable (current_orange_balloons : ℕ := 7)  -- The number of orange balloons Sally currently has

theorem sally_initial_orange_balloons : 
  current_orange_balloons + lost_orange_balloons = initial_orange_balloons := 
by
  sorry

end NUMINAMATH_GPT_sally_initial_orange_balloons_l1155_115548


namespace NUMINAMATH_GPT_width_at_bottom_of_stream_l1155_115570

theorem width_at_bottom_of_stream 
    (top_width : ℝ) (area : ℝ) (height : ℝ) (bottom_width : ℝ) :
    top_width = 10 → area = 640 → height = 80 → 
    2 * area = height * (top_width + bottom_width) → 
    bottom_width = 6 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  -- Finding bottom width
  have h5 : 2 * 640 = 80 * (10 + bottom_width) := h4
  norm_num at h5
  linarith [h5]

#check width_at_bottom_of_stream

end NUMINAMATH_GPT_width_at_bottom_of_stream_l1155_115570


namespace NUMINAMATH_GPT_max_value_of_exp_sum_l1155_115521

theorem max_value_of_exp_sum (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_ab_pos : 0 < a * b) :
    ∃ θ : ℝ, a * Real.exp θ + b * Real.exp (-θ) = 2 * Real.sqrt (a * b) :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_exp_sum_l1155_115521


namespace NUMINAMATH_GPT_segment_length_tangent_circles_l1155_115589

theorem segment_length_tangent_circles
  (r1 r2 : ℝ)
  (h1 : r1 > 0)
  (h2 : r2 > 0)
  (h3 : 7 - 4 * Real.sqrt 3 ≤ r1 / r2)
  (h4 : r1 / r2 ≤ 7 + 4 * Real.sqrt 3)
  :
  ∃ d : ℝ, d^2 = (1 / 12) * (14 * r1 * r2 - r1^2 - r2^2) :=
sorry

end NUMINAMATH_GPT_segment_length_tangent_circles_l1155_115589


namespace NUMINAMATH_GPT_eq_d_is_quadratic_l1155_115555

def is_quadratic (eq : ℕ → ℤ) : Prop :=
  ∃ a b c, a ≠ 0 ∧ eq 2 = a ∧ eq 1 = b ∧ eq 0 = c

def eq_cond_1 (n : ℕ) : ℤ :=
  match n with
  | 2 => 1  -- x^2 coefficient
  | 1 => 0  -- x coefficient
  | 0 => -1 -- constant term
  | _ => 0

theorem eq_d_is_quadratic : is_quadratic eq_cond_1 :=
  sorry

end NUMINAMATH_GPT_eq_d_is_quadratic_l1155_115555


namespace NUMINAMATH_GPT_remainder_ab_mod_n_l1155_115566

theorem remainder_ab_mod_n (n : ℕ) (a c : ℤ) (h1 : a * c ≡ 1 [ZMOD n]) (h2 : b = a * c) :
    (a * b % n) = (a % n) :=
  by
  sorry

end NUMINAMATH_GPT_remainder_ab_mod_n_l1155_115566


namespace NUMINAMATH_GPT_two_pow_2023_mod_17_l1155_115554

theorem two_pow_2023_mod_17 : (2 ^ 2023) % 17 = 4 := 
by
  sorry

end NUMINAMATH_GPT_two_pow_2023_mod_17_l1155_115554


namespace NUMINAMATH_GPT_minimum_people_in_troupe_l1155_115532

-- Let n be the number of people in the troupe.
variable (n : ℕ)

-- Conditions: n must be divisible by 8, 10, and 12.
def is_divisible_by (m k : ℕ) := m % k = 0
def divides_all (n : ℕ) := is_divisible_by n 8 ∧ is_divisible_by n 10 ∧ is_divisible_by n 12

-- The minimum number of people in the troupe that can form groups of 8, 10, or 12 with none left over.
theorem minimum_people_in_troupe (n : ℕ) : divides_all n → n = 120 :=
by
  sorry

end NUMINAMATH_GPT_minimum_people_in_troupe_l1155_115532


namespace NUMINAMATH_GPT_probability_snow_at_least_once_l1155_115505

-- Define the probabilities given in the conditions
def p_day_1_3 : ℚ := 1 / 3
def p_day_4_7 : ℚ := 1 / 4
def p_day_8_10 : ℚ := 1 / 2

-- Define the complementary no-snow probabilities
def p_no_snow_day_1_3 : ℚ := 2 / 3
def p_no_snow_day_4_7 : ℚ := 3 / 4
def p_no_snow_day_8_10 : ℚ := 1 / 2

-- Compute the total probability of no snow for all ten days
def p_no_snow_all_days : ℚ :=
  (p_no_snow_day_1_3 ^ 3) * (p_no_snow_day_4_7 ^ 4) * (p_no_snow_day_8_10 ^ 3)

-- Define the proof problem: Calculate probability of at least one snow day
theorem probability_snow_at_least_once : (1 - p_no_snow_all_days) = 2277 / 2304 := by
  sorry

end NUMINAMATH_GPT_probability_snow_at_least_once_l1155_115505


namespace NUMINAMATH_GPT_monks_mantou_l1155_115500

theorem monks_mantou (x y : ℕ) (h1 : x + y = 100) (h2 : 3 * x + y / 3 = 100) :
  (3 * x + (100 - x) / 3 = 100) ∧ (x + y = 100 ∧ 3 * x + y / 3 = 100) :=
by
  sorry

end NUMINAMATH_GPT_monks_mantou_l1155_115500


namespace NUMINAMATH_GPT_pentagon_quadrilateral_sum_of_angles_l1155_115514

   theorem pentagon_quadrilateral_sum_of_angles
     (exterior_angle_pentagon : ℕ := 72)
     (interior_angle_pentagon : ℕ := 108)
     (sum_interior_angles_quadrilateral : ℕ := 360)
     (reflex_angle : ℕ := 252) :
     (sum_interior_angles_quadrilateral - reflex_angle = interior_angle_pentagon) :=
   by
     sorry
   
end NUMINAMATH_GPT_pentagon_quadrilateral_sum_of_angles_l1155_115514


namespace NUMINAMATH_GPT_probability_of_three_correct_deliveries_l1155_115531

-- Define a combination function
def combination (n k : ℕ) : ℕ :=
  n.choose k

-- Define the factorial function
def factorial : ℕ → ℕ
  | 0 => 1
  | (n + 1) => (n + 1) * factorial n

-- Define the problem with conditions and derive the required probability
theorem probability_of_three_correct_deliveries :
  (combination 5 3) / (factorial 5) = 1 / 12 := by
  sorry

end NUMINAMATH_GPT_probability_of_three_correct_deliveries_l1155_115531


namespace NUMINAMATH_GPT_crayons_left_l1155_115593

theorem crayons_left (initial_crayons : ℕ) (kiley_fraction : ℚ) (joe_fraction : ℚ) 
  (initial_crayons_eq : initial_crayons = 48) 
  (kiley_fraction_eq : kiley_fraction = 1/4) 
  (joe_fraction_eq : joe_fraction = 1/2): 
  (initial_crayons - initial_crayons * kiley_fraction - (initial_crayons - initial_crayons * kiley_fraction) * joe_fraction) = 18 := 
by 
  sorry

end NUMINAMATH_GPT_crayons_left_l1155_115593


namespace NUMINAMATH_GPT_exists_small_area_triangle_l1155_115511

structure LatticePoint where
  x : Int
  y : Int

def isValidPoint (p : LatticePoint) : Prop := 
  |p.x| ≤ 2 ∧ |p.y| ≤ 2

def noThreeCollinear (points : List LatticePoint) : Prop := 
  ∀ (p1 p2 p3 : LatticePoint), p1 ∈ points → p2 ∈ points → p3 ∈ points → 
  ((p2.x - p1.x) * (p3.y - p1.y) ≠ (p3.x - p1.x) * (p2.y - p1.y))

def triangleArea (p1 p2 p3 : LatticePoint) : ℝ :=
  0.5 * |(p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y) : ℝ)|

theorem exists_small_area_triangle
  (points : List LatticePoint)
  (h1 : ∀ p ∈ points, isValidPoint p)
  (h2 : noThreeCollinear points) :
  ∃ (p1 p2 p3 : LatticePoint), p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧ triangleArea p1 p2 p3 ≤ 2 :=
sorry

end NUMINAMATH_GPT_exists_small_area_triangle_l1155_115511


namespace NUMINAMATH_GPT_measure_of_C_angle_maximum_area_triangle_l1155_115510

-- Proof Problem 1: Measure of angle C
theorem measure_of_C_angle (A B C : ℝ) (a b c : ℝ)
  (h1 : 0 < C ∧ C < Real.pi)
  (m n : ℝ × ℝ)
  (h2 : m = (Real.sin A, Real.sin B))
  (h3 : n = (Real.cos B, Real.cos A))
  (h4 : m.1 * n.1 + m.2 * n.2 = -Real.sin (2 * C)) :
  C = 2 * Real.pi / 3 :=
sorry

-- Proof Problem 2: Maximum area of triangle ABC
theorem maximum_area_triangle (A B C : ℝ) (a b c S : ℝ)
  (h1 : c = 2 * Real.sqrt 3)
  (h2 : Real.cos C = -1 / 2)
  (h3 : S = 1 / 2 * a * b * Real.sin (2 * Real.pi / 3)): 
  S ≤ Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_measure_of_C_angle_maximum_area_triangle_l1155_115510


namespace NUMINAMATH_GPT_xy_relationship_l1155_115542

theorem xy_relationship : 
  (∀ x y, (x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 4) ∨ (x = 3 ∧ y = 9) ∨ (x = 4 ∧ y = 16) ∨ (x = 5 ∧ y = 25) 
  → y = x * x) :=
by {
  sorry
}

end NUMINAMATH_GPT_xy_relationship_l1155_115542


namespace NUMINAMATH_GPT_largest_unsatisfiable_group_l1155_115598

theorem largest_unsatisfiable_group :
  ∃ n : ℕ, (∀ a b c : ℕ, n ≠ 6 * a + 9 * b + 20 * c) ∧ (∀ m : ℕ, m > n → ∃ a b c : ℕ, m = 6 * a + 9 * b + 20 * c) ∧ n = 43 :=
by
  sorry

end NUMINAMATH_GPT_largest_unsatisfiable_group_l1155_115598


namespace NUMINAMATH_GPT_total_travel_time_l1155_115509

/-
Define the conditions:
1. Distance_1 is 150 miles,
2. Speed_1 is 50 mph,
3. Stop_time is 0.5 hours,
4. Distance_2 is 200 miles,
5. Speed_2 is 75 mph.

and prove that the total time equals 6.17 hours.
-/

theorem total_travel_time :
  let distance1 := 150
  let speed1 := 50
  let stop_time := 0.5
  let distance2 := 200
  let speed2 := 75
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2
  time1 + stop_time + time2 = 6.17 :=
by {
  -- sorry to skip the proof part
  sorry
}

end NUMINAMATH_GPT_total_travel_time_l1155_115509


namespace NUMINAMATH_GPT_smallest_percent_both_coffee_tea_l1155_115583

noncomputable def smallest_percent_coffee_tea (P_C P_T P_not_C_or_T : ℝ) : ℝ :=
  let P_C_or_T := 1 - P_not_C_or_T
  let P_C_and_T := P_C + P_T - P_C_or_T
  P_C_and_T

theorem smallest_percent_both_coffee_tea :
  smallest_percent_coffee_tea 0.9 0.85 0.15 = 0.9 :=
by
  sorry

end NUMINAMATH_GPT_smallest_percent_both_coffee_tea_l1155_115583


namespace NUMINAMATH_GPT_ellipse_foci_coordinates_l1155_115562

theorem ellipse_foci_coordinates :
  (∀ (x y : ℝ), (x^2 / 16 + y^2 / 25 = 1) → (∃ (c : ℝ), c = 3 ∧ (x = 0 ∧ (y = c ∨ y = -c)))) :=
by
  sorry

end NUMINAMATH_GPT_ellipse_foci_coordinates_l1155_115562


namespace NUMINAMATH_GPT_piecewise_linear_function_y_at_x_10_l1155_115564

theorem piecewise_linear_function_y_at_x_10
  (k1 k2 : ℝ)
  (y : ℝ → ℝ)
  (hx1 : ∀ x < 0, y x = k1 * x)
  (hx2 : ∀ x ≥ 0, y x = k2 * x)
  (h_y_pos : y 2 = 4)
  (h_y_neg : y (-5) = -20) :
  y 10 = 20 :=
by
  sorry

end NUMINAMATH_GPT_piecewise_linear_function_y_at_x_10_l1155_115564


namespace NUMINAMATH_GPT_expected_value_T_l1155_115558

def boys_girls_expected_value (M N : ℕ) : ℚ :=
  2 * ((M / (M + N : ℚ)) * (N / (M + N - 1 : ℚ)))

theorem expected_value_T (M N : ℕ) (hM : M = 10) (hN : N = 10) :
  boys_girls_expected_value M N = 20 / 19 :=
by 
  rw [hM, hN]
  sorry

end NUMINAMATH_GPT_expected_value_T_l1155_115558


namespace NUMINAMATH_GPT_unique_solution_for_divisibility_l1155_115536

theorem unique_solution_for_divisibility (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  (a^2 + b^2) ∣ (a^3 + 1) ∧ (a^2 + b^2) ∣ (b^3 + 1) → (a = 1 ∧ b = 1) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_unique_solution_for_divisibility_l1155_115536


namespace NUMINAMATH_GPT_distance_rowed_downstream_l1155_115534

-- Define the conditions
def speed_in_still_water (b s: ℝ) := b - s = 60 / 4
def speed_of_stream (s: ℝ) := s = 3
def time_downstream (t: ℝ) := t = 4

-- Define the function that computes the downstream speed
def downstream_speed (b s t: ℝ) := (b + s) * t

-- The theorem we want to prove
theorem distance_rowed_downstream (b s t : ℝ) 
    (h1 : speed_in_still_water b s)
    (h2 : speed_of_stream s)
    (h3 : time_downstream t) : 
    downstream_speed b s t = 84 := by
    sorry

end NUMINAMATH_GPT_distance_rowed_downstream_l1155_115534


namespace NUMINAMATH_GPT_ellipse_hyperbola_foci_l1155_115550

theorem ellipse_hyperbola_foci (c d : ℝ) 
  (h_ellipse : d^2 - c^2 = 25) 
  (h_hyperbola : c^2 + d^2 = 64) : |c * d| = Real.sqrt 868.5 := by
  sorry

end NUMINAMATH_GPT_ellipse_hyperbola_foci_l1155_115550


namespace NUMINAMATH_GPT_find_m_l1155_115559

theorem find_m (m : ℝ) (h : |m - 4| = |2 * m + 7|) : m = -11 ∨ m = -1 :=
sorry

end NUMINAMATH_GPT_find_m_l1155_115559


namespace NUMINAMATH_GPT_computer_price_increase_l1155_115545

theorem computer_price_increase (c : ℕ) (h : 2 * c = 540) : c + (c * 30 / 100) = 351 :=
by
  sorry

end NUMINAMATH_GPT_computer_price_increase_l1155_115545


namespace NUMINAMATH_GPT_johns_piano_total_cost_l1155_115577

theorem johns_piano_total_cost : 
  let piano_cost := 500
  let original_lessons_cost := 20 * 40
  let discount := (25 / 100) * original_lessons_cost
  let discounted_lessons_cost := original_lessons_cost - discount
  let sheet_music_cost := 75
  let maintenance_fees := 100
  let total_cost := piano_cost + discounted_lessons_cost + sheet_music_cost + maintenance_fees
  total_cost = 1275 := 
by
  let piano_cost := 500
  let original_lessons_cost := 800
  let discount := 200
  let discounted_lessons_cost := 600
  let sheet_music_cost := 75
  let maintenance_fees := 100
  let total_cost := piano_cost + discounted_lessons_cost + sheet_music_cost + maintenance_fees
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_johns_piano_total_cost_l1155_115577
