import Mathlib

namespace NUMINAMATH_GPT_complete_square_l2294_229481

theorem complete_square (y : ℝ) : y^2 + 12 * y + 40 = (y + 6)^2 + 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_complete_square_l2294_229481


namespace NUMINAMATH_GPT_meal_combinations_l2294_229450

theorem meal_combinations (MenuA_items : ℕ) (MenuB_items : ℕ) : MenuA_items = 15 ∧ MenuB_items = 12 → MenuA_items * MenuB_items = 180 :=
by
  sorry

end NUMINAMATH_GPT_meal_combinations_l2294_229450


namespace NUMINAMATH_GPT_eight_diamond_three_l2294_229438

def diamond (x y : ℤ) : ℤ := sorry

axiom diamond_zero (x : ℤ) : diamond x 0 = x
axiom diamond_comm (x y : ℤ) : diamond x y = diamond y x
axiom diamond_recursive (x y : ℤ) : diamond (x + 2) y = diamond x y + 2 * y + 3

theorem eight_diamond_three : diamond 8 3 = 39 :=
sorry

end NUMINAMATH_GPT_eight_diamond_three_l2294_229438


namespace NUMINAMATH_GPT_fruits_eaten_total_l2294_229439

variable (oranges_per_day : ℕ) (grapes_per_day : ℕ) (days : ℕ)

def total_fruits (oranges_per_day grapes_per_day days : ℕ) : ℕ :=
  (oranges_per_day * days) + (grapes_per_day * days)

theorem fruits_eaten_total 
  (h1 : oranges_per_day = 20)
  (h2 : grapes_per_day = 40) 
  (h3 : days = 30) : 
  total_fruits oranges_per_day grapes_per_day days = 1800 := 
by 
  sorry

end NUMINAMATH_GPT_fruits_eaten_total_l2294_229439


namespace NUMINAMATH_GPT_greatest_second_term_l2294_229486

-- Definitions and Conditions
def is_arithmetic_sequence (a d : ℕ) : Bool := (a > 0) && (d > 0)
def sum_four_terms (a d : ℕ) : Bool := (4 * a + 6 * d = 80)
def integer_d (a d : ℕ) : Bool := ((40 - 2 * a) % 3 = 0)

-- Theorem statement to prove
theorem greatest_second_term : ∃ a d : ℕ, is_arithmetic_sequence a d ∧ sum_four_terms a d ∧ integer_d a d ∧ (a + d = 19) :=
sorry

end NUMINAMATH_GPT_greatest_second_term_l2294_229486


namespace NUMINAMATH_GPT_last_digit_of_a_power_b_l2294_229415

-- Define the constants from the problem
def a : ℕ := 954950230952380948328708
def b : ℕ := 470128749397540235934750230

-- Define a helper function to calculate the last digit of a natural number
def last_digit (n : ℕ) : ℕ :=
  n % 10

-- Define the main statement to be proven
theorem last_digit_of_a_power_b : last_digit ((last_digit a) ^ (b % 4)) = 4 :=
by
  -- Here go the proof steps if we were to provide them
  sorry

end NUMINAMATH_GPT_last_digit_of_a_power_b_l2294_229415


namespace NUMINAMATH_GPT_simplify_expression_l2294_229422

noncomputable def cube_root (x : ℝ) : ℝ := x ^ (1/3 : ℝ)

theorem simplify_expression :
  (cube_root 512) * (cube_root 343) = 56 := by
  -- conditions
  let h1 : 512 = 2^9 := by rfl
  let h2 : 343 = 7^3 := by rfl
  -- goal
  sorry

end NUMINAMATH_GPT_simplify_expression_l2294_229422


namespace NUMINAMATH_GPT_triangle_property_l2294_229411

theorem triangle_property
  (A B C : ℝ) (a b c : ℝ)
  (h1 : a > b)
  (h2 : a = 5)
  (h3 : c = 6)
  (h4 : Real.sin B = 3 / 5) :
  (b = Real.sqrt 13 ∧ Real.sin A = 3 * Real.sqrt 13 / 13) →
  Real.sin (2 * A + π / 4) = 7 * Real.sqrt 2 / 26 :=
sorry

end NUMINAMATH_GPT_triangle_property_l2294_229411


namespace NUMINAMATH_GPT_MrSlinkums_total_count_l2294_229468

variable (T : ℕ)

-- Defining the conditions as given in the problem
def placed_on_shelves (T : ℕ) : ℕ := (20 * T) / 100
def storage (T : ℕ) : ℕ := (80 * T) / 100

-- Stating the main theorem to prove
theorem MrSlinkums_total_count 
    (h : storage T = 120) : 
    T = 150 :=
sorry

end NUMINAMATH_GPT_MrSlinkums_total_count_l2294_229468


namespace NUMINAMATH_GPT_walnut_trees_planted_l2294_229496

-- The number of walnut trees before planting
def walnut_trees_before : ℕ := 22

-- The number of walnut trees after planting
def walnut_trees_after : ℕ := 55

-- The number of walnut trees planted today
def walnut_trees_planted_today : ℕ := 33

-- Theorem statement to prove that the number of walnut trees planted today is 33
theorem walnut_trees_planted:
  walnut_trees_after - walnut_trees_before = walnut_trees_planted_today :=
by sorry

end NUMINAMATH_GPT_walnut_trees_planted_l2294_229496


namespace NUMINAMATH_GPT_part_one_part_two_l2294_229459

theorem part_one (a b : ℝ) (h : a ≠ 0) : |a + b| + |a - b| ≥ 2 * |a| :=
by sorry

theorem part_two (x : ℝ) : |x - 1| + |x - 2| ≤ 2 ↔ (1 / 2 : ℝ) ≤ x ∧ x ≤ (5 / 2 : ℝ) :=
by sorry

end NUMINAMATH_GPT_part_one_part_two_l2294_229459


namespace NUMINAMATH_GPT_candy_problem_l2294_229410

-- Define the given conditions
def numberOfStudents : Nat := 43
def piecesOfCandyPerStudent : Nat := 8

-- Formulate the problem statement
theorem candy_problem : numberOfStudents * piecesOfCandyPerStudent = 344 := by
  sorry

end NUMINAMATH_GPT_candy_problem_l2294_229410


namespace NUMINAMATH_GPT_common_ratio_of_geometric_series_l2294_229401

theorem common_ratio_of_geometric_series 
  (a1 q : ℝ) 
  (h1 : a1 + a1 * q^2 = 5) 
  (h2 : a1 * q + a1 * q^3 = 10) : 
  q = 2 := 
by 
  sorry

end NUMINAMATH_GPT_common_ratio_of_geometric_series_l2294_229401


namespace NUMINAMATH_GPT_ratio_x_y_l2294_229416

theorem ratio_x_y (x y : ℚ) (h : (3 * x - 2 * y) / (2 * x + y) = 3 / 4) : x / y = 11 / 6 :=
by
  sorry

end NUMINAMATH_GPT_ratio_x_y_l2294_229416


namespace NUMINAMATH_GPT_abs_neg_frac_l2294_229420

theorem abs_neg_frac : abs (-3 / 2) = 3 / 2 := 
by sorry

end NUMINAMATH_GPT_abs_neg_frac_l2294_229420


namespace NUMINAMATH_GPT_initial_average_customers_l2294_229484

theorem initial_average_customers (x A : ℕ) (h1 : x = 1) (h2 : (A + 120) / 2 = 90) : A = 60 := by
  sorry

end NUMINAMATH_GPT_initial_average_customers_l2294_229484


namespace NUMINAMATH_GPT_exists_indices_divisible_2019_l2294_229407

theorem exists_indices_divisible_2019 (x : Fin 2020 → ℤ) : 
  ∃ (i j : Fin 2020), i ≠ j ∧ (x j - x i) % 2019 = 0 := 
  sorry

end NUMINAMATH_GPT_exists_indices_divisible_2019_l2294_229407


namespace NUMINAMATH_GPT_find_unit_prices_minimize_total_cost_l2294_229495

def unit_prices_ (x y : ℕ) :=
  x + 2 * y = 40 ∧ 2 * x + 3 * y = 70
  
theorem find_unit_prices (x y: ℕ) (h: unit_prices_ x y): x = 20 ∧ y = 10 := 
  sorry

def total_cost (m: ℕ) := 20 * m + 10 * (60 - m)

theorem minimize_total_cost (m : ℕ) (h1 : 60 ≥ m) (h2 : m ≥ 20) : 
  total_cost m = 800 → m = 20 :=
  sorry

end NUMINAMATH_GPT_find_unit_prices_minimize_total_cost_l2294_229495


namespace NUMINAMATH_GPT_ratio_of_sold_phones_to_production_l2294_229409

def last_years_production : ℕ := 5000
def this_years_production : ℕ := 2 * last_years_production
def phones_left_in_factory : ℕ := 7500
def sold_phones : ℕ := this_years_production - phones_left_in_factory

theorem ratio_of_sold_phones_to_production : 
  (sold_phones : ℚ) / this_years_production = 1 / 4 := 
by
  sorry

end NUMINAMATH_GPT_ratio_of_sold_phones_to_production_l2294_229409


namespace NUMINAMATH_GPT_sum_first_four_terms_geo_seq_l2294_229488

theorem sum_first_four_terms_geo_seq (q : ℝ) (a_1 : ℝ)
  (h1 : q ≠ 1) 
  (h2 : a_1 * (a_1 * q) * (a_1 * q^2) = -1/8)
  (h3 : 2 * (a_1 * q^3) = (a_1 * q) + (a_1 * q^2)) :
  (a_1 + (a_1 * q) + (a_1 * q^2) + (a_1 * q^3)) = 5 / 8 :=
  sorry

end NUMINAMATH_GPT_sum_first_four_terms_geo_seq_l2294_229488


namespace NUMINAMATH_GPT_david_average_marks_l2294_229449

-- Define the individual marks
def english_marks : ℕ := 74
def mathematics_marks : ℕ := 65
def physics_marks : ℕ := 82
def chemistry_marks : ℕ := 67
def biology_marks : ℕ := 90
def total_marks : ℕ := english_marks + mathematics_marks + physics_marks + chemistry_marks + biology_marks
def num_subjects : ℕ := 5

-- Define the average marks
def average_marks : ℚ := total_marks / num_subjects

-- Assert the average marks calculation
theorem david_average_marks : average_marks = 75.6 := by
  sorry

end NUMINAMATH_GPT_david_average_marks_l2294_229449


namespace NUMINAMATH_GPT_solve_quadratic_eq_l2294_229466

theorem solve_quadratic_eq (x : ℝ) : 4 * x^2 - (x^2 - 2 * x + 1) = 0 ↔ x = 1 / 3 ∨ x = -1 := by
  sorry

end NUMINAMATH_GPT_solve_quadratic_eq_l2294_229466


namespace NUMINAMATH_GPT_min_value_of_function_l2294_229435

theorem min_value_of_function (x : ℝ) (hx : x > 3) :
  (x + (1 / (x - 3))) ≥ 5 :=
sorry

end NUMINAMATH_GPT_min_value_of_function_l2294_229435


namespace NUMINAMATH_GPT_berries_from_fourth_bush_l2294_229418

def number_of_berries (n : ℕ) : ℕ :=
  match n with
  | 1 => 3
  | 2 => 4
  | 3 => 7
  | 5 => 19
  | _ => sorry  -- Assume the given pattern

theorem berries_from_fourth_bush : number_of_berries 4 = 12 :=
by sorry

end NUMINAMATH_GPT_berries_from_fourth_bush_l2294_229418


namespace NUMINAMATH_GPT_savings_equal_after_25_weeks_l2294_229453

theorem savings_equal_after_25_weeks (x : ℝ) :
  (160 + 25 * x = 210 + 125) → x = 7 :=
by 
  apply sorry

end NUMINAMATH_GPT_savings_equal_after_25_weeks_l2294_229453


namespace NUMINAMATH_GPT_rectangle_area_90_l2294_229440

theorem rectangle_area_90 {x y : ℝ} (h1 : (x + 3) * (y - 1) = x * y) (h2 : (x - 3) * (y + 1.5) = x * y) : x * y = 90 := 
  sorry

end NUMINAMATH_GPT_rectangle_area_90_l2294_229440


namespace NUMINAMATH_GPT_common_ratio_of_geometric_sequence_l2294_229467

theorem common_ratio_of_geometric_sequence (S : ℕ → ℝ) (a_1 a_2 : ℝ) (q : ℝ)
  (h1 : S 3 = a_1 * (1 + q + q^2))
  (h2 : 2 * S 3 = 2 * a_1 + a_2) : 
  q = -1/2 := 
sorry

end NUMINAMATH_GPT_common_ratio_of_geometric_sequence_l2294_229467


namespace NUMINAMATH_GPT_circle_area_with_radius_8_l2294_229444

noncomputable def circle_radius : ℝ := 8
noncomputable def circle_area (r : ℝ) : ℝ := Real.pi * r^2

theorem circle_area_with_radius_8 :
  circle_area circle_radius = 64 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_circle_area_with_radius_8_l2294_229444


namespace NUMINAMATH_GPT_find_value_of_x_l2294_229485

theorem find_value_of_x (a b c d e f x : ℕ) (h1 : a ≠ 1 ∧ a ≠ 6 ∧ b ≠ 1 ∧ b ≠ 6 ∧ c ≠ 1 ∧ c ≠ 6 ∧ d ≠ 1 ∧ d ≠ 6 ∧ e ≠ 1 ∧ e ≠ 6 ∧ f ≠ 1 ∧ f ≠ 6 ∧ x ≠ 1 ∧ x ≠ 6)
  (h2 : a + x + d = 18)
  (h3 : b + x + f = 18)
  (h4 : c + x + 6 = 18)
  (h5 : a + b + c + d + e + f + x + 6 + 1 = 45) :
  x = 7 :=
sorry

end NUMINAMATH_GPT_find_value_of_x_l2294_229485


namespace NUMINAMATH_GPT_strawberries_per_jar_l2294_229400

-- Let's define the conditions
def betty_strawberries : ℕ := 16
def matthew_strawberries : ℕ := betty_strawberries + 20
def natalie_strawberries : ℕ := matthew_strawberries / 2
def total_strawberries : ℕ := betty_strawberries + matthew_strawberries + natalie_strawberries
def jars_of_jam : ℕ := 40 / 4

-- Now we need to prove that the number of strawberries used in one jar of jam is 7.
theorem strawberries_per_jar : total_strawberries / jars_of_jam = 7 := by
  sorry

end NUMINAMATH_GPT_strawberries_per_jar_l2294_229400


namespace NUMINAMATH_GPT_fabric_nguyen_needs_l2294_229406

-- Definitions for conditions
def fabric_per_pant : ℝ := 8.5
def total_pants : ℝ := 7
def yards_to_feet (yards : ℝ) : ℝ := yards * 3
def fabric_nguyen_has_yards : ℝ := 3.5

-- The proof we need to establish
theorem fabric_nguyen_needs : (total_pants * fabric_per_pant) - (yards_to_feet fabric_nguyen_has_yards) = 49 :=
by
  sorry

end NUMINAMATH_GPT_fabric_nguyen_needs_l2294_229406


namespace NUMINAMATH_GPT_shark_feed_l2294_229455

theorem shark_feed (S : ℝ) (h1 : S + S/2 + 5 * S = 26) : S = 4 := 
by sorry

end NUMINAMATH_GPT_shark_feed_l2294_229455


namespace NUMINAMATH_GPT_infinite_solutions_l2294_229434

theorem infinite_solutions (x y : ℝ) : ∃ x y : ℝ, x^3 + y^2 * x - 6 * x + 5 * y + 1 = 0 :=
sorry

end NUMINAMATH_GPT_infinite_solutions_l2294_229434


namespace NUMINAMATH_GPT_probability_no_rain_five_days_l2294_229493

noncomputable def probability_of_no_rain (rain_prob : ℚ) (days : ℕ) :=
  (1 - rain_prob) ^ days

theorem probability_no_rain_five_days :
  probability_of_no_rain (2/3) 5 = 1/243 :=
by sorry

end NUMINAMATH_GPT_probability_no_rain_five_days_l2294_229493


namespace NUMINAMATH_GPT_min_people_wearing_both_l2294_229469

theorem min_people_wearing_both (n : ℕ) (h1 : n % 3 = 0)
  (h_gloves : ∃ g, g = n / 3 ∧ g = 1) (h_hats : ∃ h, h = (2 * n) / 3 ∧ h = 2) :
  ∃ x, x = 0 := by
  sorry

end NUMINAMATH_GPT_min_people_wearing_both_l2294_229469


namespace NUMINAMATH_GPT_calculation_result_l2294_229497

theorem calculation_result :
  (2 : ℝ)⁻¹ - (1 / 2 : ℝ)^0 + (2 : ℝ)^2023 * (-0.5 : ℝ)^2023 = -3 / 2 := sorry

end NUMINAMATH_GPT_calculation_result_l2294_229497


namespace NUMINAMATH_GPT_jorge_goals_l2294_229470

theorem jorge_goals (g_last g_total g_this : ℕ) (h_last : g_last = 156) (h_total : g_total = 343) :
  g_this = g_total - g_last → g_this = 187 :=
by
  intro h
  rw [h_last, h_total] at h
  apply h

end NUMINAMATH_GPT_jorge_goals_l2294_229470


namespace NUMINAMATH_GPT_classroom_count_l2294_229403

-- Definitions for conditions
def average_age_all (sum_ages : ℕ) (num_people : ℕ) : ℕ := sum_ages / num_people
def average_age_excluding_teacher (sum_ages : ℕ) (num_people : ℕ) (teacher_age : ℕ) : ℕ :=
  (sum_ages - teacher_age) / (num_people - 1)

-- Theorem statement using the provided conditions
theorem classroom_count (x : ℕ) (h1 : average_age_all (11 * x) x = 11)
  (h2 : average_age_excluding_teacher (11 * x) x 30 = 10) : x = 20 :=
  sorry

end NUMINAMATH_GPT_classroom_count_l2294_229403


namespace NUMINAMATH_GPT_fourth_number_second_set_l2294_229419

theorem fourth_number_second_set :
  (∃ (x y : ℕ), (28 + x + 42 + 78 + 104) / 5 = 90 ∧ (128 + 255 + 511 + y + x) / 5 = 423 ∧ x = 198) →
  (y = 1023) :=
by
  sorry

end NUMINAMATH_GPT_fourth_number_second_set_l2294_229419


namespace NUMINAMATH_GPT_number_added_l2294_229498

theorem number_added (x y : ℝ) (h1 : x = 33) (h2 : x / 4 + y = 15) : y = 6.75 :=
by sorry

end NUMINAMATH_GPT_number_added_l2294_229498


namespace NUMINAMATH_GPT_average_minutes_per_day_l2294_229472

-- Definitions based on the conditions
variables (f : ℕ)
def third_graders := 6 * f
def fourth_graders := 2 * f
def fifth_graders := f

def third_graders_time := 10 * third_graders f
def fourth_graders_time := 12 * fourth_graders f
def fifth_graders_time := 15 * fifth_graders f

def total_students := third_graders f + fourth_graders f + fifth_graders f
def total_time := third_graders_time f + fourth_graders_time f + fifth_graders_time f

-- Proof statement
theorem average_minutes_per_day : total_time f / total_students f = 11 := sorry

end NUMINAMATH_GPT_average_minutes_per_day_l2294_229472


namespace NUMINAMATH_GPT_no_non_negative_solutions_l2294_229464

theorem no_non_negative_solutions (a b : ℕ) (h_diff : a ≠ b) (d := Nat.gcd a b) 
                                 (a' := a / d) (b' := b / d) (n := d * (a' * b' - a' - b')) :
  ¬ ∃ x y : ℕ, a * x + b * y = n := 
by
  sorry

end NUMINAMATH_GPT_no_non_negative_solutions_l2294_229464


namespace NUMINAMATH_GPT_proof_a_square_plus_a_plus_one_l2294_229456

theorem proof_a_square_plus_a_plus_one (a : ℝ) (h : 2 * (5 - a) * (6 + a) = 100) : a^2 + a + 1 = -19 := 
by 
  sorry

end NUMINAMATH_GPT_proof_a_square_plus_a_plus_one_l2294_229456


namespace NUMINAMATH_GPT_parallel_vectors_l2294_229447

variables (x : ℝ)

theorem parallel_vectors (h : (1 + x) / 2 = (1 - 3 * x) / -1) : x = 3 / 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_parallel_vectors_l2294_229447


namespace NUMINAMATH_GPT_min_pencils_for_each_color_max_pencils_remaining_each_color_max_red_pencils_to_ensure_five_remaining_l2294_229480

-- Condition Definitions
def blue := 5
def red := 9
def green := 6
def yellow := 4

-- Theorem Statements
theorem min_pencils_for_each_color :
  ∀ B R G Y : ℕ, blue = 5 ∧ red = 9 ∧ green = 6 ∧ yellow = 4 →
  ∃ min_pencils : ℕ, min_pencils = 21 := by
  sorry

theorem max_pencils_remaining_each_color :
  ∀ B R G Y : ℕ, blue = 5 ∧ red = 9 ∧ green = 6 ∧ yellow = 4 →
  ∃ max_pencils : ℕ, max_pencils = 3 := by
  sorry

theorem max_red_pencils_to_ensure_five_remaining :
  ∀ B R G Y : ℕ, blue = 5 ∧ red = 9 ∧ green = 6 ∧ yellow = 4 →
  ∃ max_red_pencils : ℕ, max_red_pencils = 4 := by
  sorry

end NUMINAMATH_GPT_min_pencils_for_each_color_max_pencils_remaining_each_color_max_red_pencils_to_ensure_five_remaining_l2294_229480


namespace NUMINAMATH_GPT_cricket_team_matches_l2294_229465

theorem cricket_team_matches 
  (M : ℕ) (W : ℕ) 
  (h1 : W = 20 * M / 100) 
  (h2 : (W + 80) * 100 = 52 * M) : 
  M = 250 :=
by
  sorry

end NUMINAMATH_GPT_cricket_team_matches_l2294_229465


namespace NUMINAMATH_GPT_find_roots_and_m_l2294_229405

theorem find_roots_and_m (m a : ℝ) (h_root : (-2)^2 - 4 * (-2) + m = 0) :
  m = -12 ∧ a = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_roots_and_m_l2294_229405


namespace NUMINAMATH_GPT_positive_difference_l2294_229452

theorem positive_difference (x y : ℚ) (h1 : x + y = 40) (h2 : 3 * y - 4 * x = 20) : y - x = 80 / 7 := by
  sorry

end NUMINAMATH_GPT_positive_difference_l2294_229452


namespace NUMINAMATH_GPT_value_of_sum_ratio_l2294_229460

theorem value_of_sum_ratio (w x y: ℝ) (hx: w / x = 1 / 3) (hy: w / y = 2 / 3) : (x + y) / y = 3 :=
sorry

end NUMINAMATH_GPT_value_of_sum_ratio_l2294_229460


namespace NUMINAMATH_GPT_probability_full_house_after_rerolling_l2294_229457

theorem probability_full_house_after_rerolling
  (a b c : ℕ)
  (h0 : a ≠ b)
  (h1 : c ≠ a)
  (h2 : c ≠ b) :
  (2 / 6 : ℚ) = (1 / 3 : ℚ) :=
by
  sorry

end NUMINAMATH_GPT_probability_full_house_after_rerolling_l2294_229457


namespace NUMINAMATH_GPT_no_solution_for_m_l2294_229474

theorem no_solution_for_m (m : ℝ) : ¬ ∃ x : ℝ, x ≠ 1 ∧ (mx - 1) / (x - 1) = 3 ↔ m = 1 ∨ m = 3 := 
by sorry

end NUMINAMATH_GPT_no_solution_for_m_l2294_229474


namespace NUMINAMATH_GPT_smallest_n_exists_square_smallest_n_exists_cube_l2294_229429

open Nat

-- Statement for part (a)
theorem smallest_n_exists_square (n x y : ℕ) : (∀ n x y, (x * (x + n) = y^2) → (∃ (x y : ℕ), n = 3 ∧ (x * (x + 3) = y^2))) := sorry

-- Statement for part (b)
theorem smallest_n_exists_cube (n x y : ℕ) : (∀ n x y, (x * (x + n) = y^3) → (∃ (x y : ℕ), n = 2 ∧ (x * (x + 2) = y^3))) := sorry

end NUMINAMATH_GPT_smallest_n_exists_square_smallest_n_exists_cube_l2294_229429


namespace NUMINAMATH_GPT_present_number_of_teachers_l2294_229492

theorem present_number_of_teachers (S T : ℕ) (h1 : S = 50 * T) (h2 : S + 50 = 25 * (T + 5)) : T = 3 := 
by 
  sorry

end NUMINAMATH_GPT_present_number_of_teachers_l2294_229492


namespace NUMINAMATH_GPT_x_k_expr_a_x_k_expr_b_x_k_expr_c_y_k_expr_a_y_k_expr_b_y_k_expr_c_l2294_229461

theorem x_k_expr_a (x : ℕ → ℝ) (y : ℕ → ℝ) (k : ℕ) (h1 : ∀ k, y k = (x (k + 1) - 3 * x k) / 2) (h2 : ∀ k, x k = (y (k + 1) - 3 * y k) / 4) : 
  x k = 6 * x (k - 1) - x (k - 2) := 
by sorry

theorem x_k_expr_b (x : ℕ → ℝ) (y : ℕ → ℝ) (k : ℕ) (h1 : ∀ k, y k = (x (k + 1) - 3 * x k) / 2) (h2 : ∀ k, x k = (y (k + 1) - 3 * y k) / 4) : 
  x k = 34 * x (k - 2) - x (k - 4) := 
by sorry

theorem x_k_expr_c (x : ℕ → ℝ) (y : ℕ → ℝ) (k : ℕ) (h1 : ∀ k, y k = (x (k + 1) - 3 * x k) / 2) (h2 : ∀ k, x k = (y (k + 1) - 3 * y k) / 4) : 
  x k = 198 * x (k - 3) - x (k - 6) := 
by sorry

theorem y_k_expr_a (x : ℕ → ℝ) (y : ℕ → ℝ) (k : ℕ) (h1 : ∀ k, y k = (x (k + 1) - 3 * x k) / 2) (h2 : ∀ k, x k = (y (k + 1) - 3 * y k) / 4) : 
  y k = 6 * y (k - 1) - y (k - 2) := 
by sorry

theorem y_k_expr_b (x : ℕ → ℝ) (y : ℕ → ℝ) (k : ℕ) (h1 : ∀ k, y k = (x (k + 1) - 3 * x k) / 2) (h2 : ∀ k, x k = (y (k + 1) - 3 * y k) / 4) : 
  y k = 34 * y (k - 2) - y (k - 4) := 
by sorry

theorem y_k_expr_c (x : ℕ → ℝ) (y : ℕ → ℝ) (k : ℕ) (h1 : ∀ k, y k = (x (k + 1) - 3 * x k) / 2) (h2 : ∀ k, x k = (y (k + 1) - 3 * y k) / 4) : 
  y k = 198 * y (k - 3) - y (k - 6) := 
by sorry

end NUMINAMATH_GPT_x_k_expr_a_x_k_expr_b_x_k_expr_c_y_k_expr_a_y_k_expr_b_y_k_expr_c_l2294_229461


namespace NUMINAMATH_GPT_book_length_ratio_is_4_l2294_229417

-- Define the initial conditions
def pages_when_6 : ℕ := 8
def age_when_start := 6
def multiple_at_twice_age := 5
def multiple_eight_years_after := 3
def current_pages : ℕ := 480

def pages_when_12 := pages_when_6 * multiple_at_twice_age
def pages_when_20 := pages_when_12 * multiple_eight_years_after

theorem book_length_ratio_is_4 :
  (current_pages : ℚ) / pages_when_20 = 4 := by
  -- We need to show the proof for the equality
  sorry

end NUMINAMATH_GPT_book_length_ratio_is_4_l2294_229417


namespace NUMINAMATH_GPT_W_3_7_eq_13_l2294_229454

-- Define the operation W
def W (x y : ℤ) : ℤ := y + 5 * x - x^2

-- State the theorem
theorem W_3_7_eq_13 : W 3 7 = 13 := by
  sorry

end NUMINAMATH_GPT_W_3_7_eq_13_l2294_229454


namespace NUMINAMATH_GPT_number_of_subjects_l2294_229451

variable (P C M : ℝ)

-- Given conditions
def conditions (P C M : ℝ) : Prop :=
  (P + C + M) / 3 = 75 ∧
  (P + M) / 2 = 90 ∧
  (P + C) / 2 = 70 ∧
  P = 95

-- Proposition with given conditions and the conclusion
theorem number_of_subjects (P C M : ℝ) (h : conditions P C M) : 
  (∃ n : ℕ, n = 3) :=
by
  sorry

end NUMINAMATH_GPT_number_of_subjects_l2294_229451


namespace NUMINAMATH_GPT_find_range_of_a_l2294_229413

-- Definitions and conditions
def pointA : ℝ × ℝ := (0, 3)
def lineL (x : ℝ) : ℝ := 2 * x - 4
def circleCenter (a : ℝ) : ℝ × ℝ := (a, 2 * a - 4)
def circleRadius : ℝ := 1

-- The range to prove
def valid_range (a : ℝ) : Prop :=
  0 ≤ a ∧ a ≤ 12 / 5

-- Main theorem
theorem find_range_of_a (a : ℝ) (M : ℝ × ℝ)
  (on_circle : (M.1 - (circleCenter a).1)^2 + (M.2 - (circleCenter a).2)^2 = circleRadius^2)
  (condition_MA_MD : (M.1 - pointA.1)^2 + (M.2 - pointA.2)^2 = 4 * M.1^2 + 4 * M.2^2) :
  valid_range a :=
sorry

end NUMINAMATH_GPT_find_range_of_a_l2294_229413


namespace NUMINAMATH_GPT_solution_set_inequality_l2294_229425

noncomputable def f : ℝ → ℝ := sorry

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
∀ ⦃a b⦄, a ∈ s → b ∈ s → a ≤ b → f a ≤ f b

def f_increasing_on_pos : Prop := is_increasing_on f (Set.Ioi 0)

def f_at_one_zero : Prop := f 1 = 0

theorem solution_set_inequality : 
    is_odd f →
    f_increasing_on_pos →
    f_at_one_zero →
    {x : ℝ | x * (f x - f (-x)) < 0} = {x : ℝ | -1 < x ∧ x < 0 ∨ 0 < x ∧ x < 1} :=
sorry

end NUMINAMATH_GPT_solution_set_inequality_l2294_229425


namespace NUMINAMATH_GPT_correct_completion_l2294_229477

-- Definitions of conditions
def sentence_template := "By the time he arrives, all the work ___, with ___ our teacher will be content."
def option_A := ("will be accomplished", "that")
def option_B := ("will have been accomplished", "which")
def option_C := ("will have accomplished", "it")
def option_D := ("had been accomplished", "him")

-- The actual proof statement
theorem correct_completion : (option_B.fst = "will have been accomplished") ∧ (option_B.snd = "which") :=
by
  sorry

end NUMINAMATH_GPT_correct_completion_l2294_229477


namespace NUMINAMATH_GPT_find_g4_l2294_229483

noncomputable def g : ℝ → ℝ := sorry

theorem find_g4 (h : ∀ x y : ℝ, x * g y = 2 * y * g x) (h₁ : g 10 = 5) : g 4 = 4 :=
sorry

end NUMINAMATH_GPT_find_g4_l2294_229483


namespace NUMINAMATH_GPT_intersection_is_2_to_inf_l2294_229478

-- Define the set A
def setA (x : ℝ) : Prop :=
 x > 1

-- Define the set B
def setB (y : ℝ) : Prop :=
 ∃ x : ℝ, y = Real.sqrt (x^2 + 2*x + 5)

-- Define the intersection of A and B
def setIntersection : Set ℝ :=
{ y | setA y ∧ setB y }

-- Statement to prove the intersection
theorem intersection_is_2_to_inf : setIntersection = { y | y ≥ 2 } :=
sorry -- Proof is omitted

end NUMINAMATH_GPT_intersection_is_2_to_inf_l2294_229478


namespace NUMINAMATH_GPT_problem_l2294_229408

noncomputable def f (x : ℝ) (a : ℝ) (α : ℝ) (b : ℝ) (β : ℝ) : ℝ :=
  a * Real.sin (Real.pi * x + α) + b * Real.cos (Real.pi * x + β)

theorem problem {a α b β : ℝ} (h : f 2001 a α b β = 3) : f 2012 a α b β = -3 := by
  sorry

end NUMINAMATH_GPT_problem_l2294_229408


namespace NUMINAMATH_GPT_larger_number_is_correct_l2294_229499

theorem larger_number_is_correct : ∃ L : ℝ, ∃ S : ℝ, S = 48 ∧ (L - S = (1 : ℝ) / (3 : ℝ) * L) ∧ L = 72 :=
by
  sorry

end NUMINAMATH_GPT_larger_number_is_correct_l2294_229499


namespace NUMINAMATH_GPT_inequality_holds_l2294_229463

-- Given conditions
variables {a b x y : ℝ}
variables (pos_a : 0 < a) (pos_b : 0 < b) (pos_x : 0 < x) (pos_y : 0 < y)
variable (h : a + b = 1)

-- Goal/Question
theorem inequality_holds : (a * x + b * y) * (b * x + a * y) ≥ x * y :=
by sorry

end NUMINAMATH_GPT_inequality_holds_l2294_229463


namespace NUMINAMATH_GPT_interest_rate_per_annum_l2294_229412

-- Given conditions
variables (BG TD t : ℝ) (FV r : ℝ)
axiom bg_eq : BG = 6
axiom td_eq : TD = 50
axiom t_eq : t = 1
axiom bankers_gain_eq : BG = FV * r * t - (FV - TD) * r * t

-- Proof problem
theorem interest_rate_per_annum : r = 0.12 :=
by sorry

end NUMINAMATH_GPT_interest_rate_per_annum_l2294_229412


namespace NUMINAMATH_GPT_PQ_is_10_5_l2294_229423

noncomputable def PQ_length_proof_problem : Prop := 
  ∃ (PQ : ℝ),
    PQ = 10.5 ∧ 
    ∃ (ST : ℝ) (SU : ℝ),
      ST = 4.5 ∧ SU = 7.5 ∧ 
      ∃ (QR : ℝ) (PR : ℝ),
        QR = 21 ∧ PR = 15 ∧ 
        ∃ (angle_PQR angle_STU : ℝ),
          angle_PQR = 120 ∧ angle_STU = 120 ∧ 
          PQ / ST = PR / SU

theorem PQ_is_10_5 :
  PQ_length_proof_problem := sorry

end NUMINAMATH_GPT_PQ_is_10_5_l2294_229423


namespace NUMINAMATH_GPT_video_down_votes_l2294_229489

theorem video_down_votes 
  (up_votes : ℕ)
  (ratio_up_down : up_votes / 1394 = 45 / 17)
  (up_votes_known : up_votes = 3690) : 
  3690 / 1394 = 45 / 17 :=
by
  sorry

end NUMINAMATH_GPT_video_down_votes_l2294_229489


namespace NUMINAMATH_GPT_required_remaining_speed_l2294_229476

-- Definitions for the given problem
variables (D T : ℝ) 

-- Given conditions from the problem
def speed_first_part (D T : ℝ) : Prop := 
  40 = (2 * D / 3) / (T / 3)

def remaining_distance_time (D T : ℝ) : Prop :=
  10 = (D / 3) / (2 * (2 * D / 3) / 40 / 3)

-- Theorem to be proved
theorem required_remaining_speed (D T : ℝ) 
  (h1 : speed_first_part D T)
  (h2 : remaining_distance_time D T) :
  10 = (D / 3) / (2 * (T / 3)) :=
  sorry  -- Proof is skipped

end NUMINAMATH_GPT_required_remaining_speed_l2294_229476


namespace NUMINAMATH_GPT_calculate_3_pow_5_mul_6_pow_5_l2294_229426

theorem calculate_3_pow_5_mul_6_pow_5 :
  3^5 * 6^5 = 34012224 := 
by 
  sorry

end NUMINAMATH_GPT_calculate_3_pow_5_mul_6_pow_5_l2294_229426


namespace NUMINAMATH_GPT_x_squared_plus_y_squared_l2294_229473

theorem x_squared_plus_y_squared (x y : ℝ) (h₀ : x + y = 10) (h₁ : x * y = 15) : x^2 + y^2 = 70 :=
by
  sorry

end NUMINAMATH_GPT_x_squared_plus_y_squared_l2294_229473


namespace NUMINAMATH_GPT_number_of_students_playing_soccer_l2294_229446

variable (total_students boys playing_soccer_girls not_playing_soccer_girls : ℕ)
variable (percentage_boys_playing_soccer : ℕ)

-- Conditions
axiom h1 : total_students = 470
axiom h2 : boys = 300
axiom h3 : not_playing_soccer_girls = 135
axiom h4 : percentage_boys_playing_soccer = 86
axiom h5 : playing_soccer_girls = 470 - 300 - not_playing_soccer_girls

-- Question: Prove that the number of students playing soccer is 250
theorem number_of_students_playing_soccer : 
  (playing_soccer_girls * 100) / (100 - percentage_boys_playing_soccer) = 250 :=
sorry

end NUMINAMATH_GPT_number_of_students_playing_soccer_l2294_229446


namespace NUMINAMATH_GPT_exists_seq_nat_lcm_decreasing_l2294_229402

-- Natural number sequence and conditions
def seq_nat_lcm_decreasing : Prop :=
  ∃ (a : Fin 100 → ℕ), 
  ((∀ i j : Fin 100, i < j → a i < a j) ∧
  (∀ (i : Fin 99), Nat.lcm (a i) (a (i + 1)) > Nat.lcm (a (i + 1)) (a (i + 2))))

theorem exists_seq_nat_lcm_decreasing : seq_nat_lcm_decreasing :=
  sorry

end NUMINAMATH_GPT_exists_seq_nat_lcm_decreasing_l2294_229402


namespace NUMINAMATH_GPT_actual_road_length_l2294_229414

theorem actual_road_length
  (scale_factor : ℕ → ℕ → Prop)
  (map_length_cm : ℕ)
  (actual_length_km : ℝ) : 
  (scale_factor 1 50000) →
  (map_length_cm = 15) →
  (actual_length_km = 7.5) :=
by
  sorry

end NUMINAMATH_GPT_actual_road_length_l2294_229414


namespace NUMINAMATH_GPT_fourth_friend_payment_l2294_229462

theorem fourth_friend_payment (a b c d : ℕ) 
  (h1 : a = (1 / 3) * (b + c + d)) 
  (h2 : b = (1 / 4) * (a + c + d)) 
  (h3 : c = (1 / 5) * (a + b + d))
  (h4 : a + b + c + d = 84) : 
  d = 40 := by
sorry

end NUMINAMATH_GPT_fourth_friend_payment_l2294_229462


namespace NUMINAMATH_GPT_green_faction_lies_more_l2294_229445

theorem green_faction_lies_more (r1 r2 r3 l1 l2 l3 : ℕ) 
  (h1 : r1 + r2 + r3 + l1 + l2 + l3 = 2016) 
  (h2 : r1 + l2 + l3 = 1208) 
  (h3 : r2 + l1 + l3 = 908) 
  (h4 : r3 + l1 + l2 = 608) :
  l3 - r3 = 100 :=
by
  sorry

end NUMINAMATH_GPT_green_faction_lies_more_l2294_229445


namespace NUMINAMATH_GPT_car_value_decrease_per_year_l2294_229437

theorem car_value_decrease_per_year 
  (initial_value : ℝ) (final_value : ℝ) (years : ℝ) (decrease_per_year : ℝ)
  (h1 : initial_value = 20000)
  (h2 : final_value = 14000)
  (h3 : years = 6)
  (h4 : initial_value - final_value = 6 * decrease_per_year) : 
  decrease_per_year = 1000 :=
sorry

end NUMINAMATH_GPT_car_value_decrease_per_year_l2294_229437


namespace NUMINAMATH_GPT_min_value_f_range_of_a_l2294_229490

-- Define the function f(x) with parameter a.
def f (x a : ℝ) := |x + a| + |x - a|

-- (Ⅰ) Statement: Prove that for a = 1, the minimum value of f(x) is 2.
theorem min_value_f (x : ℝ) : f x 1 ≥ 2 :=
  by sorry

-- (Ⅱ) Statement: Prove that if f(2) > 5, then the range of values for a is (-∞, -5/2) ∪ (5/2, +∞).
theorem range_of_a (a : ℝ) : f 2 a > 5 → a < -5 / 2 ∨ a > 5 / 2 :=
  by sorry

end NUMINAMATH_GPT_min_value_f_range_of_a_l2294_229490


namespace NUMINAMATH_GPT_division_quotient_l2294_229431

theorem division_quotient (dividend divisor remainder quotient : ℕ) 
  (h₁ : dividend = 95) (h₂ : divisor = 15) (h₃ : remainder = 5)
  (h₄ : dividend = divisor * quotient + remainder) : quotient = 6 :=
by
  sorry

end NUMINAMATH_GPT_division_quotient_l2294_229431


namespace NUMINAMATH_GPT_part1_part2_l2294_229491

section

variables {x m : ℝ}

def f (x m : ℝ) : ℝ := 3 * x^2 + (4 - m) * x - 6 * m
def g (x m : ℝ) : ℝ := 2 * x^2 - x - m

theorem part1 (m : ℝ) (h : m = 1) : 
  {x : ℝ | f x m > 0} = {x : ℝ | x < -2 ∨ x > 1} :=
sorry

theorem part2 (m : ℝ) (h : m > 0) : 
  {x : ℝ | f x m ≤ g x m} = {x : ℝ | -5 ≤ x ∧ x ≤ m} :=
sorry
     
end

end NUMINAMATH_GPT_part1_part2_l2294_229491


namespace NUMINAMATH_GPT_solve_for_k_l2294_229458

noncomputable def f (k x : ℝ) : ℝ := k * x^2 + (k-1) * x + 2

theorem solve_for_k (k : ℝ) :
  (∀ x : ℝ, f k x = f k (-x)) ↔ k = 1 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_k_l2294_229458


namespace NUMINAMATH_GPT_five_aliens_have_more_limbs_than_five_martians_l2294_229475

-- Definitions based on problem conditions

def number_of_alien_arms : ℕ := 3
def number_of_alien_legs : ℕ := 8

-- Martians have twice as many arms as Aliens and half as many legs
def number_of_martian_arms : ℕ := 2 * number_of_alien_arms
def number_of_martian_legs : ℕ := number_of_alien_legs / 2

-- Total limbs for five aliens and five martians
def total_limbs_for_aliens (n : ℕ) : ℕ := n * (number_of_alien_arms + number_of_alien_legs)
def total_limbs_for_martians (n : ℕ) : ℕ := n * (number_of_martian_arms + number_of_martian_legs)

-- The theorem to prove
theorem five_aliens_have_more_limbs_than_five_martians :
  total_limbs_for_aliens 5 - total_limbs_for_martians 5 = 5 :=
sorry

end NUMINAMATH_GPT_five_aliens_have_more_limbs_than_five_martians_l2294_229475


namespace NUMINAMATH_GPT_log3_of_7_eq_ab_l2294_229424

noncomputable def log3_of_2_eq_a (a : ℝ) : Prop := Real.log 2 / Real.log 3 = a
noncomputable def log2_of_7_eq_b (b : ℝ) : Prop := Real.log 7 / Real.log 2 = b

theorem log3_of_7_eq_ab (a b : ℝ) (h1 : log3_of_2_eq_a a) (h2 : log2_of_7_eq_b b) :
  Real.log 7 / Real.log 3 = a * b :=
sorry

end NUMINAMATH_GPT_log3_of_7_eq_ab_l2294_229424


namespace NUMINAMATH_GPT_sam_total_money_spent_l2294_229427

def value_of_pennies (n : ℕ) : ℝ := n * 0.01
def value_of_nickels (n : ℕ) : ℝ := n * 0.05
def value_of_dimes (n : ℕ) : ℝ := n * 0.10
def value_of_quarters (n : ℕ) : ℝ := n * 0.25

def total_money_spent : ℝ :=
  (value_of_pennies 5 + value_of_nickels 3) +  -- Monday
  (value_of_dimes 8 + value_of_quarters 4) +   -- Tuesday
  (value_of_nickels 7 + value_of_dimes 10 + value_of_quarters 2) +  -- Wednesday
  (value_of_pennies 20 + value_of_nickels 15 + value_of_dimes 12 + value_of_quarters 6) +  -- Thursday
  (value_of_pennies 45 + value_of_nickels 20 + value_of_dimes 25 + value_of_quarters 10)  -- Friday

theorem sam_total_money_spent : total_money_spent = 14.05 :=
by
  sorry

end NUMINAMATH_GPT_sam_total_money_spent_l2294_229427


namespace NUMINAMATH_GPT_sneakers_cost_l2294_229448

theorem sneakers_cost (rate_per_yard : ℝ) (num_yards_cut : ℕ) (total_earnings : ℝ) :
  rate_per_yard = 2.15 ∧ num_yards_cut = 6 ∧ total_earnings = rate_per_yard * num_yards_cut → 
  total_earnings = 12.90 :=
by
  sorry

end NUMINAMATH_GPT_sneakers_cost_l2294_229448


namespace NUMINAMATH_GPT_simplify_and_evaluate_l2294_229482

-- Problem statement with conditions translated into Lean
theorem simplify_and_evaluate (a : ℝ) (h : a = Real.sqrt 5 + 1) :
  (a / (a^2 - 2*a + 1)) / (1 + 1 / (a - 1)) = Real.sqrt 5 / 5 := sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l2294_229482


namespace NUMINAMATH_GPT_intersecting_diagonals_probability_l2294_229442

def probability_of_intersecting_diagonals_inside_dodecagon : ℚ :=
  let total_points := 12
  let total_segments := (total_points.choose 2)
  let sides := 12
  let diagonals := total_segments - sides
  let ways_to_choose_2_diagonals := (diagonals.choose 2)
  let ways_to_choose_4_points := (total_points.choose 4)
  let probability := (ways_to_choose_4_points : ℚ) / (ways_to_choose_2_diagonals : ℚ)
  probability

theorem intersecting_diagonals_probability (H : probability_of_intersecting_diagonals_inside_dodecagon = 165 / 477) : 
  probability_of_intersecting_diagonals_inside_dodecagon = 165 / 477 :=
  by
  sorry

end NUMINAMATH_GPT_intersecting_diagonals_probability_l2294_229442


namespace NUMINAMATH_GPT_sea_horses_count_l2294_229436

theorem sea_horses_count (S P : ℕ) 
  (h1 : S / P = 5 / 11) 
  (h2 : P = S + 85) 
  : S = 70 := sorry

end NUMINAMATH_GPT_sea_horses_count_l2294_229436


namespace NUMINAMATH_GPT_not_enough_money_l2294_229432

-- Define the prices of the books
def price_animal_world : Real := 21.8
def price_fairy_tale_stories : Real := 19.5

-- Define the total amount of money Xiao Ming has
def xiao_ming_money : Real := 40.0

-- Define the statement we want to prove
theorem not_enough_money : (price_animal_world + price_fairy_tale_stories) > xiao_ming_money := by
  sorry

end NUMINAMATH_GPT_not_enough_money_l2294_229432


namespace NUMINAMATH_GPT_greatest_divisor_of_630_lt_35_and_factor_of_90_l2294_229479

theorem greatest_divisor_of_630_lt_35_and_factor_of_90 : ∃ d : ℕ, d < 35 ∧ d ∣ 630 ∧ d ∣ 90 ∧ ∀ e : ℕ, (e < 35 ∧ e ∣ 630 ∧ e ∣ 90) → e ≤ d := 
sorry

end NUMINAMATH_GPT_greatest_divisor_of_630_lt_35_and_factor_of_90_l2294_229479


namespace NUMINAMATH_GPT_triangle_inequality_l2294_229404

-- Define the nondegenerate condition for the triangle's side lengths.
def nondegenerate_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

-- Define the perimeter condition for the triangle.
def triangle_perimeter (a b c : ℝ) (p : ℝ) : Prop :=
  a + b + c = p

-- The main theorem to prove the given inequality.
theorem triangle_inequality (a b c : ℝ) (h_non_deg : nondegenerate_triangle a b c) (h_perim : triangle_perimeter a b c 1) :
  abs ((a - b) / (c + a * b)) + abs ((b - c) / (a + b * c)) + abs ((c - a) / (b + a * c)) < 2 :=
by
  sorry

end NUMINAMATH_GPT_triangle_inequality_l2294_229404


namespace NUMINAMATH_GPT_max_dot_and_area_of_triangle_l2294_229494

noncomputable def triangle_data (A B C : ℝ) (m n : ℝ × ℝ) : Prop :=
  A + B + C = Real.pi ∧
  (m = (2, 2 * (Real.cos ((B + C) / 2))^2 - 1)) ∧
  (n = (Real.sin (A / 2), -1))

noncomputable def is_max_dot_product (A : ℝ) (m n : ℝ × ℝ) : Prop :=
  m.1 * n.1 + m.2 * n.2 = (if A = Real.pi / 3 then 3 / 2 else 0)

noncomputable def max_area (A B C : ℝ) : ℝ :=
  let a : ℝ := 2
  let b : ℝ := 2
  let c : ℝ := 2
  if A = Real.pi / 3 then (Real.sqrt 3) else 0

theorem max_dot_and_area_of_triangle {A B C : ℝ} {m n : ℝ × ℝ}
  (h_triangle : triangle_data A B C m n) :
  is_max_dot_product (Real.pi / 3) m n ∧ max_area A B C = Real.sqrt 3 := by sorry

end NUMINAMATH_GPT_max_dot_and_area_of_triangle_l2294_229494


namespace NUMINAMATH_GPT_problem_statement_l2294_229487

theorem problem_statement : 25 * 15 * 9 * 5.4 * 3.24 = 3 ^ 10 := 
by 
  sorry

end NUMINAMATH_GPT_problem_statement_l2294_229487


namespace NUMINAMATH_GPT_percentage_increase_in_overtime_rate_l2294_229471

def regular_rate : ℝ := 16
def regular_hours : ℝ := 40
def total_compensation : ℝ := 976
def total_hours_worked : ℝ := 52

theorem percentage_increase_in_overtime_rate :
  ((total_compensation - (regular_rate * regular_hours)) / (total_hours_worked - regular_hours) - regular_rate) / regular_rate * 100 = 75 :=
by
  sorry

end NUMINAMATH_GPT_percentage_increase_in_overtime_rate_l2294_229471


namespace NUMINAMATH_GPT_picture_distance_from_right_end_l2294_229430

def distance_from_right_end_of_wall (wall_width picture_width position_from_left : ℕ) : ℕ := 
  wall_width - (position_from_left + picture_width)

theorem picture_distance_from_right_end :
  ∀ (wall_width picture_width position_from_left : ℕ), 
  wall_width = 24 -> 
  picture_width = 4 -> 
  position_from_left = 5 -> 
  distance_from_right_end_of_wall wall_width picture_width position_from_left = 15 :=
by
  intros wall_width picture_width position_from_left hw hp hp_left
  rw [hw, hp, hp_left]
  sorry

end NUMINAMATH_GPT_picture_distance_from_right_end_l2294_229430


namespace NUMINAMATH_GPT_smallest_n_l2294_229441

-- Define the conditions as properties of integers
def connected (a b : ℕ): Prop := sorry -- Assume we have a definition for connectivity

def condition1 (a b n : ℕ) : Prop :=
  ¬connected a b → Nat.gcd (a^2 + b^2) n = 1

def condition2 (a b n : ℕ) : Prop :=
  connected a b → Nat.gcd (a^2 + b^2) n > 1

theorem smallest_n : ∃ n, n = 65 ∧ ∀ (a b : ℕ), condition1 a b n ∧ condition2 a b n := by
  sorry

end NUMINAMATH_GPT_smallest_n_l2294_229441


namespace NUMINAMATH_GPT_repeating_decimal_to_fraction_l2294_229428

theorem repeating_decimal_to_fraction :
  (0.3 + 0.206) = (5057 / 9990) :=
sorry

end NUMINAMATH_GPT_repeating_decimal_to_fraction_l2294_229428


namespace NUMINAMATH_GPT_probability_X_between_neg1_and_5_probability_X_le_8_probability_X_ge_5_probability_X_between_neg3_and_9_l2294_229421

noncomputable def normalCDF (z : ℝ) : ℝ :=
  sorry -- Assuming some CDF function for the sake of the example.

variable (X : ℝ → ℝ)
variable (μ : ℝ := 3)
variable (σ : ℝ := sqrt 4)

-- 1. Proof that P(-1 < X < 5) = 0.8185
theorem probability_X_between_neg1_and_5 : 
  ((-1 < X) ∧ (X < 5) → (normalCDF 1 - normalCDF (-2)) = 0.8185) :=
  sorry

-- 2. Proof that P(X ≤ 8) = 0.9938
theorem probability_X_le_8 : 
  (X ≤ 8 → normalCDF 2.5 = 0.9938) :=
  sorry

-- 3. Proof that P(X ≥ 5) = 0.1587
theorem probability_X_ge_5 : 
  (X ≥ 5 → (1 - normalCDF 1) = 0.1587) :=
  sorry

-- 4. Proof that P(-3 < X < 9) = 0.9972
theorem probability_X_between_neg3_and_9 : 
  ((-3 < X) ∧ (X < 9) → (2 * normalCDF 3 - 1) = 0.9972) :=
  sorry

end NUMINAMATH_GPT_probability_X_between_neg1_and_5_probability_X_le_8_probability_X_ge_5_probability_X_between_neg3_and_9_l2294_229421


namespace NUMINAMATH_GPT_gcd_256_180_720_l2294_229443

theorem gcd_256_180_720 : Int.gcd (Int.gcd 256 180) 720 = 36 := by
  sorry

end NUMINAMATH_GPT_gcd_256_180_720_l2294_229443


namespace NUMINAMATH_GPT_find_x_l2294_229433

noncomputable def a : ℝ × ℝ := (3, 4)
noncomputable def b : ℝ × ℝ := (2, 1)

def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem find_x (x : ℝ) (h : dot_product (a.1 + x * b.1, a.2 + x * b.2) (a.1 - b.1, a.2 - b.2) = 0) : x = -3 :=
  sorry

end NUMINAMATH_GPT_find_x_l2294_229433
