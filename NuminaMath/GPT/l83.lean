import Mathlib

namespace NUMINAMATH_GPT_roommates_condition_l83_8376

def f (x : ℝ) := 3 * x ^ 2 + 5 * x - 1
def g (x : ℝ) := 2 * x ^ 2 - 3 * x + 5

theorem roommates_condition : f 3 = 2 * g 3 + 5 := 
by {
  sorry
}

end NUMINAMATH_GPT_roommates_condition_l83_8376


namespace NUMINAMATH_GPT_total_viewing_time_l83_8385

theorem total_viewing_time :
  let original_times := [4, 6, 7, 5, 9]
  let new_species_times := [3, 7, 8, 10]
  let total_breaks := 8
  let break_time_per_animal := 2
  let total_time := (original_times.sum + new_species_times.sum) + (total_breaks * break_time_per_animal)
  total_time = 75 :=
by
  sorry

end NUMINAMATH_GPT_total_viewing_time_l83_8385


namespace NUMINAMATH_GPT_max_value_of_function_l83_8386

theorem max_value_of_function (x : ℝ) (h : x < 5 / 4) :
    (∀ y, y = 4 * x - 2 + 1 / (4 * x - 5) → y ≤ 1):=
sorry

end NUMINAMATH_GPT_max_value_of_function_l83_8386


namespace NUMINAMATH_GPT_problem_statement_l83_8380

theorem problem_statement (a b c : ℝ) (h1 : a + 2 * b + 3 * c = 12) (h2 : a^2 + b^2 + c^2 = a * b + a * c + b * c) :
  a + b^2 + c^3 = 14 :=
sorry

end NUMINAMATH_GPT_problem_statement_l83_8380


namespace NUMINAMATH_GPT_painting_area_l83_8382

theorem painting_area
  (wall_height : ℝ) (wall_length : ℝ)
  (window_height : ℝ) (window_length : ℝ)
  (door_height : ℝ) (door_length : ℝ)
  (cond1 : wall_height = 10) (cond2 : wall_length = 15)
  (cond3 : window_height = 3) (cond4 : window_length = 5)
  (cond5 : door_height = 2) (cond6 : door_length = 7) :
  wall_height * wall_length - window_height * window_length - door_height * door_length = 121 := 
by
  simp [cond1, cond2, cond3, cond4, cond5, cond6]
  sorry

end NUMINAMATH_GPT_painting_area_l83_8382


namespace NUMINAMATH_GPT_cube_root_110592_l83_8377

theorem cube_root_110592 :
  (∃ x : ℕ, x^3 = 110592) ∧ 
  10^3 = 1000 ∧ 11^3 = 1331 ∧ 12^3 = 1728 ∧ 13^3 = 2197 ∧ 14^3 = 2744 ∧ 
  15^3 = 3375 ∧ 20^3 = 8000 ∧ 21^3 = 9261 ∧ 22^3 = 10648 ∧ 23^3 = 12167 ∧ 
  24^3 = 13824 ∧ 25^3 = 15625 → 48^3 = 110592 :=
by
  sorry

end NUMINAMATH_GPT_cube_root_110592_l83_8377


namespace NUMINAMATH_GPT_calculate_first_year_sample_l83_8316

noncomputable def stratified_sampling : ℕ :=
  let total_sample_size := 300
  let first_grade_ratio := 4
  let second_grade_ratio := 5
  let third_grade_ratio := 5
  let fourth_grade_ratio := 6
  let total_ratio := first_grade_ratio + second_grade_ratio + third_grade_ratio + fourth_grade_ratio
  let first_grade_proportion := first_grade_ratio / total_ratio
  300 * first_grade_proportion

theorem calculate_first_year_sample :
  stratified_sampling = 60 :=
by sorry

end NUMINAMATH_GPT_calculate_first_year_sample_l83_8316


namespace NUMINAMATH_GPT_currency_notes_total_l83_8328

theorem currency_notes_total (num_50_notes total_amount remaining_amount num_100_notes : ℕ) 
  (h1 : remaining_amount = total_amount - (num_50_notes * 50))
  (h2 : num_50_notes = 3500 / 50)
  (h3 : total_amount = 5000)
  (h4 : remaining_amount = 1500)
  (h5 : num_100_notes = remaining_amount / 100) : 
  num_50_notes + num_100_notes = 85 :=
by sorry

end NUMINAMATH_GPT_currency_notes_total_l83_8328


namespace NUMINAMATH_GPT_larry_wins_probability_l83_8395

noncomputable def probability (n : ℕ) : ℝ :=
  if n % 2 = 1 then (1/2)^(n) else 0

noncomputable def inf_geometric_sum (a : ℝ) (r : ℝ) : ℝ :=
  a / (1 - r)

theorem larry_wins_probability :
  inf_geometric_sum (1/2) (1/4) = 2/3 :=
by
  sorry

end NUMINAMATH_GPT_larry_wins_probability_l83_8395


namespace NUMINAMATH_GPT_sum_of_roots_l83_8393

variables {a b c : ℝ}

-- Conditions
-- The polynomial with roots a, b, c
def poly (x : ℝ) : ℝ := 24 * x^3 - 36 * x^2 + 14 * x - 1

-- The roots are in (0, 1)
def in_interval (x : ℝ) : Prop := 0 < x ∧ x < 1

-- All roots are distinct
def distinct (a b c : ℝ) : Prop := a ≠ b ∧ b ≠ c ∧ c ≠ a

-- Main Theorem
theorem sum_of_roots :
  (∀ x, poly x = 0 → x = a ∨ x = b ∨ x = c) →
  in_interval a →
  in_interval b →
  in_interval c →
  distinct a b c →
  (1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c) = 3 / 2) :=
by
  intros
  sorry

end NUMINAMATH_GPT_sum_of_roots_l83_8393


namespace NUMINAMATH_GPT_yellow_side_probability_correct_l83_8307

-- Define the problem scenario
structure CardBox where
  total_cards : ℕ := 8
  green_green_cards : ℕ := 4
  green_yellow_cards : ℕ := 2
  yellow_yellow_cards : ℕ := 2

noncomputable def yellow_side_probability 
  (box : CardBox)
  (picked_is_yellow : Bool) : ℚ :=
  if picked_is_yellow then
    let total_yellow_sides := 2 * box.green_yellow_cards + 2 * box.yellow_yellow_cards
    let yellow_yellow_sides := 2 * box.yellow_yellow_cards
    yellow_yellow_sides / total_yellow_sides
  else 0

theorem yellow_side_probability_correct :
  yellow_side_probability {total_cards := 8, green_green_cards := 4, green_yellow_cards := 2, yellow_yellow_cards := 2} true = 2 / 3 :=
by 
  sorry

end NUMINAMATH_GPT_yellow_side_probability_correct_l83_8307


namespace NUMINAMATH_GPT_longest_side_of_triangle_l83_8304

theorem longest_side_of_triangle (x : ℝ) (a b c : ℝ)
  (h1 : a = 5)
  (h2 : b = 2 * x + 3)
  (h3 : c = 3 * x - 2)
  (h4 : a + b + c = 41) :
  c = 19 :=
by
  sorry

end NUMINAMATH_GPT_longest_side_of_triangle_l83_8304


namespace NUMINAMATH_GPT_average_of_consecutive_sequences_l83_8323

theorem average_of_consecutive_sequences (a b : ℕ) (h : b = (a + (a+1) + (a+2) + (a+3) + (a+4)) / 5) : 
    ((b + (b+1) + (b+2) + (b+3) + (b+4)) / 5) = a + 4 :=
by
  sorry

end NUMINAMATH_GPT_average_of_consecutive_sequences_l83_8323


namespace NUMINAMATH_GPT_relationship_among_sets_l83_8339

-- Definitions of the integer sets E, F, and G
def E := {e : ℝ | ∃ m : ℤ, e = m + 1 / 6}
def F := {f : ℝ | ∃ n : ℤ, f = n / 2 - 1 / 3}
def G := {g : ℝ | ∃ p : ℤ, g = p / 2 + 1 / 6}

-- The theorem statement capturing the relationship among E, F, and G
theorem relationship_among_sets : E ⊆ F ∧ F = G := by
  sorry

end NUMINAMATH_GPT_relationship_among_sets_l83_8339


namespace NUMINAMATH_GPT_quadratic_real_roots_l83_8343

theorem quadratic_real_roots (a b c : ℝ) (h : a * c < 0) : 
  ∃ x y : ℝ, a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 ∧ x ≠ y :=
by
  sorry

end NUMINAMATH_GPT_quadratic_real_roots_l83_8343


namespace NUMINAMATH_GPT_propositions_are_3_and_4_l83_8356

-- Conditions
def stmt_1 := "Is it fun to study math?"
def stmt_2 := "Do your homework well and strive to pass the math test next time;"
def stmt_3 := "2 is not a prime number"
def stmt_4 := "0 is a natural number"

-- Representation of a propositional statement
def isPropositional (stmt : String) : Bool :=
  stmt ≠ stmt_1 ∧ stmt ≠ stmt_2

-- The theorem proving the question given the conditions
theorem propositions_are_3_and_4 :
  isPropositional stmt_3 ∧ isPropositional stmt_4 :=
by
  -- Proof to be filled in later
  sorry

end NUMINAMATH_GPT_propositions_are_3_and_4_l83_8356


namespace NUMINAMATH_GPT_range_of_k_l83_8355

theorem range_of_k (k : ℝ) :
  (∃ x y : ℝ, k^2 * x^2 + y^2 - 4 * k * x + 2 * k * y + k^2 - 1 = 0 ∧ (x, y) = (0, 0)) →
  0 < |k| ∧ |k| < 1 :=
by
  intros
  sorry

end NUMINAMATH_GPT_range_of_k_l83_8355


namespace NUMINAMATH_GPT_candidate_failed_by_45_marks_l83_8303

-- Define the main parameters
def passing_percentage : ℚ := 45 / 100
def candidate_marks : ℝ := 180
def maximum_marks : ℝ := 500
def passing_marks : ℝ := passing_percentage * maximum_marks
def failing_marks : ℝ := passing_marks - candidate_marks

-- State the theorem to be proved
theorem candidate_failed_by_45_marks : failing_marks = 45 := by
  sorry

end NUMINAMATH_GPT_candidate_failed_by_45_marks_l83_8303


namespace NUMINAMATH_GPT_tangent_line_at_A_tangent_line_through_B_l83_8398

open Real

noncomputable def f (x : ℝ) : ℝ := 4 / x
noncomputable def f' (x : ℝ) : ℝ := -4 / (x^2)

theorem tangent_line_at_A : 
  ∃ m b, m = -1 ∧ b = 4 ∧ (∀ x, 1 ≤ x → (x + b = 4)) :=
sorry

theorem tangent_line_through_B :
  ∃ m b, m = 4 ∧ b = -8 ∧ (∀ x, 1 ≤ x → (4*x + b = 8)) :=
sorry

end NUMINAMATH_GPT_tangent_line_at_A_tangent_line_through_B_l83_8398


namespace NUMINAMATH_GPT_greatest_possible_positive_integer_difference_l83_8378

theorem greatest_possible_positive_integer_difference (x y : ℤ) (hx : 4 < x) (hx' : x < 6) (hy : 6 < y) (hy' : y < 10) :
  y - x = 4 :=
sorry

end NUMINAMATH_GPT_greatest_possible_positive_integer_difference_l83_8378


namespace NUMINAMATH_GPT_train_length_l83_8372

theorem train_length (v_kmph : ℝ) (t_s : ℝ) (L_p : ℝ) (L_t : ℝ) : 
  (v_kmph = 72) ∧ (t_s = 15) ∧ (L_p = 250) →
  L_t = 50 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_train_length_l83_8372


namespace NUMINAMATH_GPT_combined_length_of_all_CDs_l83_8375

-- Define the lengths of each CD based on the conditions
def length_cd1 := 1.5
def length_cd2 := 1.5
def length_cd3 := 2 * length_cd1
def length_cd4 := length_cd2 / 2
def length_cd5 := length_cd1 + length_cd2

-- Define the combined length of all CDs
def combined_length := length_cd1 + length_cd2 + length_cd3 + length_cd4 + length_cd5

-- State the theorem
theorem combined_length_of_all_CDs : combined_length = 9.75 := by
  sorry

end NUMINAMATH_GPT_combined_length_of_all_CDs_l83_8375


namespace NUMINAMATH_GPT_find_c_deg3_l83_8350

-- Define the polynomials f and g.
def f (x : ℚ) : ℚ := 2 - 10 * x + 4 * x^2 - 5 * x^3 + 7 * x^4
def g (x : ℚ) : ℚ := 5 - 3 * x - 8 * x^3 + 11 * x^4

-- The statement that needs proof.
theorem find_c_deg3 (c : ℚ) : (∀ x : ℚ, f x + c * g x ≠ 0 → f x + c * g x = 2 - 10 * x + 4 * x^2 - 5 * x^3 - c * 8 * x^3) ↔ c = -7 / 11 :=
sorry

end NUMINAMATH_GPT_find_c_deg3_l83_8350


namespace NUMINAMATH_GPT_find_monthly_salary_l83_8396

variables (x h_1 h_2 h_3 : ℕ)

theorem find_monthly_salary 
    (half_salary_bank : h_1 = x / 2)
    (half_remaining_mortgage : h_2 = (h_1 - 300) / 2)
    (half_remaining_expenses : h_3 = (h_2 + 300) / 2)
    (remaining_salary : h_3 = 800) :
  x = 7600 :=
sorry

end NUMINAMATH_GPT_find_monthly_salary_l83_8396


namespace NUMINAMATH_GPT_gina_initial_money_l83_8362

variable (M : ℝ)
variable (kept : ℝ := 170)

theorem gina_initial_money (h1 : M * 1 / 4 + M * 1 / 8 + M * 1 / 5 + kept = M) : 
  M = 400 :=
by
  sorry

end NUMINAMATH_GPT_gina_initial_money_l83_8362


namespace NUMINAMATH_GPT_width_of_barrier_l83_8305

theorem width_of_barrier (r1 r2 : ℝ) (h : 2 * π * r1 - 2 * π * r2 = 16 * π) : r1 - r2 = 8 :=
by
  -- The proof would be inserted here, but is not required as per instructions.
  sorry

end NUMINAMATH_GPT_width_of_barrier_l83_8305


namespace NUMINAMATH_GPT_alchemists_less_than_half_l83_8326

variable (k c a : ℕ)

theorem alchemists_less_than_half (h1 : k = c + a) (h2 : c > a) : a < k / 2 := by
  sorry

end NUMINAMATH_GPT_alchemists_less_than_half_l83_8326


namespace NUMINAMATH_GPT_inequality_ab_gt_ac_l83_8347

theorem inequality_ab_gt_ac {a b c : ℝ} (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) : a * b > a * c :=
sorry

end NUMINAMATH_GPT_inequality_ab_gt_ac_l83_8347


namespace NUMINAMATH_GPT_tangent_lines_parallel_to_4x_minus_1_l83_8371

noncomputable def f (x : ℝ) : ℝ := x^3 + x - 2

theorem tangent_lines_parallel_to_4x_minus_1 :
  ∃ (a b : ℝ), (f a = b ∧ 3 * a^2 + 1 = 4) → (b = 4 * a - 4 ∨ b = 4 * a) :=
by
  sorry

end NUMINAMATH_GPT_tangent_lines_parallel_to_4x_minus_1_l83_8371


namespace NUMINAMATH_GPT_solve_system_l83_8322

theorem solve_system :
  ∃ x y : ℝ, (x^2 + 3 * x * y = 18 ∧ x * y + 3 * y^2 = 6) ∧ ((x = 3 ∧ y = 1) ∨ (x = -3 ∧ y = -1)) :=
by
  sorry

end NUMINAMATH_GPT_solve_system_l83_8322


namespace NUMINAMATH_GPT_interior_angle_of_regular_nonagon_l83_8363

theorem interior_angle_of_regular_nonagon : 
  let n := 9
  let sum_of_interior_angles := 180 * (n - 2)
  (sum_of_interior_angles / n) = 140 := 
by
  let n := 9
  let sum_of_interior_angles := 180 * (n - 2)
  show sum_of_interior_angles / n = 140
  sorry

end NUMINAMATH_GPT_interior_angle_of_regular_nonagon_l83_8363


namespace NUMINAMATH_GPT_complex_division_identity_l83_8374

noncomputable def left_hand_side : ℂ := (-2 : ℂ) + (5 : ℂ) * Complex.I / (6 : ℂ) - (3 : ℂ) * Complex.I
noncomputable def right_hand_side : ℂ := - (9 : ℂ) / 15 + (8 : ℂ) / 15 * Complex.I

theorem complex_division_identity : left_hand_side = right_hand_side := 
by
  sorry

end NUMINAMATH_GPT_complex_division_identity_l83_8374


namespace NUMINAMATH_GPT_cost_of_lamp_and_flashlight_max_desk_lamps_l83_8366

-- Part 1: Cost of purchasing one desk lamp and one flashlight
theorem cost_of_lamp_and_flashlight (x : ℕ) (desk_lamp_cost flashlight_cost : ℕ) 
        (hx : desk_lamp_cost = x + 20)
        (hdesk : 400 = x / 2 * desk_lamp_cost)
        (hflash : 160 = x * flashlight_cost)
        (hnum : desk_lamp_cost = 2 * flashlight_cost) : 
        desk_lamp_cost = 25 ∧ flashlight_cost = 5 :=
sorry

-- Part 2: Maximum number of desk lamps Rongqing Company can purchase
theorem max_desk_lamps (a : ℕ) (desk_lamp_cost flashlight_cost : ℕ)
        (hc1 : desk_lamp_cost = 25)
        (hc2 : flashlight_cost = 5)
        (free_flashlight : ℕ := a) (required_flashlight : ℕ := 2 * a + 8) 
        (total_cost : ℕ := desk_lamp_cost * a + flashlight_cost * required_flashlight)
        (hcost : total_cost ≤ 670) :
        a ≤ 21 :=
sorry

end NUMINAMATH_GPT_cost_of_lamp_and_flashlight_max_desk_lamps_l83_8366


namespace NUMINAMATH_GPT_percentage_increase_l83_8318

theorem percentage_increase (x : ℝ) (h : 2 * x = 540) (new_price : ℝ) (h_new_price : new_price = 351) :
  ((new_price - x) / x) * 100 = 30 := by
  sorry

end NUMINAMATH_GPT_percentage_increase_l83_8318


namespace NUMINAMATH_GPT_cost_of_items_l83_8320

theorem cost_of_items (x : ℝ) (cost_caramel_apple cost_ice_cream_cone : ℝ) :
  3 * cost_caramel_apple + 4 * cost_ice_cream_cone = 2 ∧
  cost_caramel_apple = cost_ice_cream_cone + 0.25 →
  cost_ice_cream_cone = 0.17857 ∧ cost_caramel_apple = 0.42857 :=
sorry

end NUMINAMATH_GPT_cost_of_items_l83_8320


namespace NUMINAMATH_GPT_average_age_population_l83_8312

theorem average_age_population 
  (k : ℕ) 
  (hwomen : ℕ := 7 * k)
  (hmen : ℕ := 5 * k)
  (avg_women_age : ℕ := 40)
  (avg_men_age : ℕ := 30)
  (h_age_women : ℕ := avg_women_age * hwomen)
  (h_age_men : ℕ := avg_men_age * hmen) : 
  (h_age_women + h_age_men) / (hwomen + hmen) = 35 + 5/6 :=
by
  sorry -- proof will fill in here

end NUMINAMATH_GPT_average_age_population_l83_8312


namespace NUMINAMATH_GPT_n_consecutive_even_sum_l83_8367

theorem n_consecutive_even_sum (n k : ℕ) (hn : n > 2) (hk : k > 2) : 
  ∃ (a : ℕ), (n * (n - 1)^(k - 1)) = (2 * a + (2 * a + 2 * (n - 1))) / 2 * n :=
by
  sorry

end NUMINAMATH_GPT_n_consecutive_even_sum_l83_8367


namespace NUMINAMATH_GPT_cos_beta_value_l83_8336

theorem cos_beta_value (α β : ℝ) (hα : 0 < α ∧ α < π/2) (hβ : 0 < β ∧ β < π/2)
    (h1 : Real.sin α = 3/5) (h2 : Real.cos (α + β) = 5/13) : 
    Real.cos β = 56/65 := 
by
  sorry

end NUMINAMATH_GPT_cos_beta_value_l83_8336


namespace NUMINAMATH_GPT_harrys_fish_count_l83_8353

theorem harrys_fish_count : 
  let sam_fish := 7
  let joe_fish := 8 * sam_fish
  let harry_fish := 4 * joe_fish
  harry_fish = 224 :=
by
  sorry

end NUMINAMATH_GPT_harrys_fish_count_l83_8353


namespace NUMINAMATH_GPT_fraction_minimum_decimal_digits_l83_8311

def minimum_decimal_digits (n d : ℕ) : ℕ := sorry

theorem fraction_minimum_decimal_digits :
  minimum_decimal_digits 987654321 (2^28 * 5^3) = 28 :=
sorry

end NUMINAMATH_GPT_fraction_minimum_decimal_digits_l83_8311


namespace NUMINAMATH_GPT_complement_of_A_l83_8300

/-
Given:
1. Universal set U = {0, 1, 2, 3, 4}
2. Set A = {1, 2}

Prove:
C_U A = {0, 3, 4}
-/

section
  variable (U : Set ℕ) (A : Set ℕ)
  variable (hU : U = {0, 1, 2, 3, 4})
  variable (hA : A = {1, 2})

  theorem complement_of_A (C_UA : Set ℕ) (hCUA : C_UA = {0, 3, 4}) : 
    {x ∈ U | x ∉ A} = C_UA :=
  by
    sorry
end

end NUMINAMATH_GPT_complement_of_A_l83_8300


namespace NUMINAMATH_GPT_find_X_plus_Y_l83_8329

-- Statement of the problem translated from the given problem-solution pair.
theorem find_X_plus_Y (X Y : ℚ) :
  (∀ x : ℚ, x ≠ 5 → x ≠ 6 →
    (Y * x + 8) / (x^2 - 11 * x + 30) = X / (x - 5) + 7 / (x - 6)) →
  X + Y = -22 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_X_plus_Y_l83_8329


namespace NUMINAMATH_GPT_total_bill_correct_l83_8333

def first_family_adults := 2
def first_family_children := 3
def second_family_adults := 4
def second_family_children := 2
def third_family_adults := 3
def third_family_children := 4

def adult_meal_cost := 8
def child_meal_cost := 5
def drink_cost_per_person := 2

def calculate_total_cost 
  (adults1 : ℕ) (children1 : ℕ) 
  (adults2 : ℕ) (children2 : ℕ) 
  (adults3 : ℕ) (children3 : ℕ)
  (adult_cost : ℕ) (child_cost : ℕ)
  (drink_cost : ℕ) : ℕ := 
  let meal_cost1 := (adults1 * adult_cost) + (children1 * child_cost)
  let meal_cost2 := (adults2 * adult_cost) + (children2 * child_cost)
  let meal_cost3 := (adults3 * adult_cost) + (children3 * child_cost)
  let drink_cost1 := (adults1 + children1) * drink_cost
  let drink_cost2 := (adults2 + children2) * drink_cost
  let drink_cost3 := (adults3 + children3) * drink_cost
  meal_cost1 + drink_cost1 + meal_cost2 + drink_cost2 + meal_cost3 + drink_cost3
   
theorem total_bill_correct :
  calculate_total_cost
    first_family_adults first_family_children
    second_family_adults second_family_children
    third_family_adults third_family_children
    adult_meal_cost child_meal_cost drink_cost_per_person = 153 :=
  sorry

end NUMINAMATH_GPT_total_bill_correct_l83_8333


namespace NUMINAMATH_GPT_alpha_beta_purchase_ways_l83_8319

-- Definitions for the problem
def number_of_flavors : ℕ := 7
def number_of_milk_types : ℕ := 4
def total_products_to_purchase : ℕ := 5

-- Conditions
def alpha_max_per_flavor : ℕ := 2
def beta_only_cookies (x : ℕ) : Prop := x = number_of_flavors

-- Main theorem (statement only)
theorem alpha_beta_purchase_ways : 
  ∃ (ways : ℕ), 
    ways = 17922 ∧
    ∀ (alpha beta : ℕ), 
      alpha + beta = total_products_to_purchase →
      (alpha <= alpha_max_per_flavor * number_of_flavors ∧ beta <= total_products_to_purchase - alpha) :=
sorry

end NUMINAMATH_GPT_alpha_beta_purchase_ways_l83_8319


namespace NUMINAMATH_GPT_ratio_a2_a3_l83_8368

namespace SequenceProof

def a (n : ℕ) : ℤ := 3 - 2^n

theorem ratio_a2_a3 : a 2 / a 3 = 1 / 5 := by
  sorry

end SequenceProof

end NUMINAMATH_GPT_ratio_a2_a3_l83_8368


namespace NUMINAMATH_GPT_quadratic_difference_sum_l83_8358

theorem quadratic_difference_sum :
  let a := 2
  let b := -10
  let c := 3
  let Δ := b * b - 4 * a * c
  let root1 := (10 + Real.sqrt Δ) / (2 * a)
  let root2 := (10 - Real.sqrt Δ) / (2 * a)
  let diff := root1 - root2
  let m := 19  -- from the difference calculation
  let n := 1   -- from the simplified form
  m + n = 20 :=
by
  -- Placeholders for calculation and proof steps.
  sorry

end NUMINAMATH_GPT_quadratic_difference_sum_l83_8358


namespace NUMINAMATH_GPT_cannot_determine_total_inhabitants_without_additional_info_l83_8383

variable (T : ℝ) (M F : ℝ)

axiom inhabitants_are_males_females : M + F = 1
axiom twenty_percent_of_males_are_literate : M * 0.20 * T = 0.20 * M * T
axiom twenty_five_percent_of_all_literates : 0.25 = 0.25 * T / T
axiom thirty_two_five_percent_of_females_are_literate : F = 1 - M ∧ F * 0.325 * T = 0.325 * (1 - M) * T

theorem cannot_determine_total_inhabitants_without_additional_info :
  ∃ (T : ℝ), True ↔ False := by
  sorry

end NUMINAMATH_GPT_cannot_determine_total_inhabitants_without_additional_info_l83_8383


namespace NUMINAMATH_GPT_sqrt_expression_l83_8308

theorem sqrt_expression : 2 * Real.sqrt 3 - (3 * Real.sqrt 2 + Real.sqrt 3) = Real.sqrt 3 - 3 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_expression_l83_8308


namespace NUMINAMATH_GPT_cylinder_volume_ratio_l83_8340

noncomputable def volume_ratio (h1 h2 : ℝ) (c1 c2 : ℝ) : ℝ :=
  let r1 := c1 / (2 * Real.pi)
  let r2 := c2 / (2 * Real.pi)
  let V1 := Real.pi * r1^2 * h1
  let V2 := Real.pi * r2^2 * h2
  if V1 > V2 then V1 / V2 else V2 / V1

theorem cylinder_volume_ratio :
  volume_ratio 7 6 6 7 = 7 / 4 :=
by
  sorry

end NUMINAMATH_GPT_cylinder_volume_ratio_l83_8340


namespace NUMINAMATH_GPT_dave_tickets_l83_8359

-- Definitions based on given conditions
def initial_tickets : ℕ := 25
def spent_tickets : ℕ := 22
def additional_tickets : ℕ := 15

-- Proof statement to demonstrate Dave would have 18 tickets
theorem dave_tickets : initial_tickets - spent_tickets + additional_tickets = 18 := by
  sorry

end NUMINAMATH_GPT_dave_tickets_l83_8359


namespace NUMINAMATH_GPT_ratio_of_width_perimeter_is_3_16_l83_8346

-- We define the conditions
def length_of_room : ℕ := 25
def width_of_room : ℕ := 15

-- We define the calculation and verification of the ratio
theorem ratio_of_width_perimeter_is_3_16 :
  let P := 2 * (length_of_room + width_of_room)
  let ratio := width_of_room / P
  let a := 15 / Nat.gcd 15 80
  let b := 80 / Nat.gcd 15 80
  (a, b) = (3, 16) :=
by 
  -- The proof is skipped with sorry
  sorry

end NUMINAMATH_GPT_ratio_of_width_perimeter_is_3_16_l83_8346


namespace NUMINAMATH_GPT_remainder_check_l83_8349

theorem remainder_check (q : ℕ) (n : ℕ) (h1 : q = 3^19) (h2 : n = 1162261460) : q % n = 7 := by
  rw [h1, h2]
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_remainder_check_l83_8349


namespace NUMINAMATH_GPT_passengers_remaining_after_fourth_stop_l83_8310

theorem passengers_remaining_after_fourth_stop :
  let initial_passengers := 64
  let remaining_fraction := 2 / 3
  (initial_passengers * remaining_fraction * remaining_fraction * remaining_fraction * remaining_fraction = 1024 / 81) :=
by
  let initial_passengers := 64
  let remaining_fraction := 2 / 3
  have H1 : initial_passengers * remaining_fraction = 128 / 3 := sorry
  have H2 : (128 / 3) * remaining_fraction = 256 / 9 := sorry
  have H3 : (256 / 9) * remaining_fraction = 512 / 27 := sorry
  have H4 : (512 / 27) * remaining_fraction = 1024 / 81 := sorry
  exact H4

end NUMINAMATH_GPT_passengers_remaining_after_fourth_stop_l83_8310


namespace NUMINAMATH_GPT_percentage_reduction_in_production_l83_8341

theorem percentage_reduction_in_production :
  let daily_production_rate := 10
  let days_in_year := 365
  let total_production_first_year := daily_production_rate * days_in_year
  let total_production_second_year := 3285
  let reduction_in_production := total_production_first_year - total_production_second_year
  let percentage_reduction := (reduction_in_production * 100) / total_production_first_year
  percentage_reduction = 10 :=
by
  let daily_production_rate := 10
  let days_in_year := 365
  let total_production_first_year := daily_production_rate * days_in_year
  let total_production_second_year := 3285
  let reduction_in_production := total_production_first_year - total_production_second_year
  let percentage_reduction := (reduction_in_production * 100) / total_production_first_year
  sorry

end NUMINAMATH_GPT_percentage_reduction_in_production_l83_8341


namespace NUMINAMATH_GPT_stewart_farm_horse_food_l83_8387

theorem stewart_farm_horse_food 
  (ratio : ℚ) (food_per_horse : ℤ) (num_sheep : ℤ) (num_horses : ℤ)
  (h1 : ratio = 5 / 7)
  (h2 : food_per_horse = 230)
  (h3 : num_sheep = 40)
  (h4 : ratio * num_horses = num_sheep) : 
  (num_horses * food_per_horse = 12880) := 
sorry

end NUMINAMATH_GPT_stewart_farm_horse_food_l83_8387


namespace NUMINAMATH_GPT_find_ab_l83_8335

theorem find_ab (a b : ℝ) (h1 : a + b = 5) (h2 : a^3 + b^3 = 35) : a * b = 6 :=
sorry

end NUMINAMATH_GPT_find_ab_l83_8335


namespace NUMINAMATH_GPT_margie_drive_distance_l83_8332

theorem margie_drive_distance
  (miles_per_gallon : ℕ)
  (cost_per_gallon : ℕ)
  (dollar_amount : ℕ)
  (h₁ : miles_per_gallon = 32)
  (h₂ : cost_per_gallon = 4)
  (h₃ : dollar_amount = 20) :
  (dollar_amount / cost_per_gallon) * miles_per_gallon = 160 :=
by
  sorry

end NUMINAMATH_GPT_margie_drive_distance_l83_8332


namespace NUMINAMATH_GPT_range_of_k_for_one_solution_l83_8315

-- Definitions
def angle_B : ℝ := 60 -- Angle B in degrees
def side_b : ℝ := 12 -- Length of side b
def side_a (k : ℝ) : ℝ := k -- Length of side a (parameterized by k)

-- Theorem stating the range of k that makes the side_a have exactly one solution
theorem range_of_k_for_one_solution (k : ℝ) : (0 < k ∧ k <= 12) ∨ k = 8 * Real.sqrt 3 := 
sorry

end NUMINAMATH_GPT_range_of_k_for_one_solution_l83_8315


namespace NUMINAMATH_GPT_sum_binomial_coefficients_l83_8351

theorem sum_binomial_coefficients :
  let a := 1
  let b := 1
  let binomial := (2 * a + 2 * b)
  (binomial)^7 = 16384 := by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_sum_binomial_coefficients_l83_8351


namespace NUMINAMATH_GPT_number_of_cow_herds_l83_8352

theorem number_of_cow_herds 
    (total_cows : ℕ) 
    (cows_per_herd : ℕ) 
    (h1 : total_cows = 320)
    (h2 : cows_per_herd = 40) : 
    total_cows / cows_per_herd = 8 :=
by
  sorry

end NUMINAMATH_GPT_number_of_cow_herds_l83_8352


namespace NUMINAMATH_GPT_range_of_m_for_false_proposition_l83_8334

theorem range_of_m_for_false_proposition :
  (∀ x ∈ (Set.Icc 0 (Real.pi / 4)), Real.tan x < m) → False ↔ m ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_for_false_proposition_l83_8334


namespace NUMINAMATH_GPT_find_a_l83_8364

theorem find_a (f : ℤ → ℤ) (h1 : ∀ (x : ℤ), f (2 * x + 1) = 3 * x + 2) (h2 : f a = 2) : a = 1 := by
sorry

end NUMINAMATH_GPT_find_a_l83_8364


namespace NUMINAMATH_GPT_common_number_l83_8388

theorem common_number (a b c d e u v w : ℝ) (h1 : (a + b + c + d + e) / 5 = 7) 
                                            (h2 : (u + v + w) / 3 = 10) 
                                            (h3 : (a + b + c + d + e + u + v + w) / 8 = 8) 
                                            (h4 : a + b + c + d + e = 35) 
                                            (h5 : u + v + w = 30) 
                                            (h6 : a + b + c + d + e + u + v + w = 64) 
                                            (h7 : 35 + 30 = 65):
  d = u := 
by
  sorry

end NUMINAMATH_GPT_common_number_l83_8388


namespace NUMINAMATH_GPT_gcd_lcm_product_l83_8327

theorem gcd_lcm_product (a b : ℕ) (ha : a = 90) (hb : b = 150) : 
  Nat.gcd a b * Nat.lcm a b = 13500 := 
by 
  sorry

end NUMINAMATH_GPT_gcd_lcm_product_l83_8327


namespace NUMINAMATH_GPT_s_at_1_l83_8370

def t (x : ℚ) := 5 * x - 12
def s (y : ℚ) := (y + 12) / 5 ^ 2 + 5 * ((y + 12) / 5) - 4

theorem s_at_1 : s 1 = 394 / 25 := by
  sorry

end NUMINAMATH_GPT_s_at_1_l83_8370


namespace NUMINAMATH_GPT_points_on_octagon_boundary_l83_8301

def is_on_octagon_boundary (x y : ℝ) : Prop :=
  |x| + |y| + |x - 1| + |y - 1| = 4

theorem points_on_octagon_boundary :
  ∀ (x y : ℝ), is_on_octagon_boundary x y ↔ ((0 ≤ x ∧ x ≤ 1 ∧ (y = 2 ∨ y = -1)) ∨
                                             (0 ≤ y ∧ y ≤ 1 ∧ (x = 2 ∨ x = -1)) ∨
                                             (x ≥ 1 ∧ y ≥ 1 ∧ x + y = 3) ∨
                                             (x ≤ 1 ∧ y ≤ 1 ∧ x + y = 1) ∨
                                             (x ≥ 1 ∧ y ≤ -1 ∧ x + y = 1) ∨
                                             (x ≤ -1 ∧ y ≥ 1 ∧ x + y = 1) ∨
                                             (x ≤ -1 ∧ y ≤ 1 ∧ x + y = -1) ∨
                                             (x ≤ 1 ∧ y ≤ -1 ∧ x + y = -1)) :=
by
  sorry

end NUMINAMATH_GPT_points_on_octagon_boundary_l83_8301


namespace NUMINAMATH_GPT_compensation_problem_l83_8394

namespace CompensationProof

variables (a b c : ℝ)

def geometric_seq_with_ratio_1_by_2 (a b c : ℝ) : Prop :=
  c = (1/2) * b ∧ b = (1/2) * a

def total_compensation_eq (a b c : ℝ) : Prop :=
  4 * c + 2 * b + a = 50

theorem compensation_problem :
  total_compensation_eq a b c ∧ geometric_seq_with_ratio_1_by_2 a b c → c = 50 / 7 :=
sorry

end CompensationProof

end NUMINAMATH_GPT_compensation_problem_l83_8394


namespace NUMINAMATH_GPT_minimum_pawns_remaining_l83_8337

-- Define the initial placement and movement conditions
structure Chessboard :=
  (white_pawns : ℕ)
  (black_pawns : ℕ)
  (on_board : ℕ)

def valid_placement (cb : Chessboard) : Prop :=
  cb.white_pawns = 32 ∧ cb.black_pawns = 32 ∧ cb.on_board = 64

def can_capture (player_pawn : ℕ → ℕ → Prop) := 
  ∀ (wp bp : ℕ), 
  wp ≥ 0 ∧ bp ≥ 0 ∧ wp + bp = 64 →
  ∀ (p_wp p_bp : ℕ), 
  player_pawn wp p_wp ∧ player_pawn bp p_bp →
  p_wp + p_bp ≥ 2
  
-- Our theorem to prove
theorem minimum_pawns_remaining (cb : Chessboard) (player_pawn : ℕ → ℕ → Prop) :
  valid_placement cb →
  can_capture player_pawn →
  ∃ min_pawns : ℕ, min_pawns = 2 :=
by
  sorry

end NUMINAMATH_GPT_minimum_pawns_remaining_l83_8337


namespace NUMINAMATH_GPT_lines_perpendicular_l83_8342

variable (b : ℝ)

/-- Proof that if the given lines are perpendicular, then b must be 3 -/
theorem lines_perpendicular (h : b ≠ 0) :
    let l₁_slope := -3
    let l₂_slope := b / 9
    l₁_slope * l₂_slope = -1 → b = 3 :=
by
  intros slope_prod
  simp only [h]
  sorry

end NUMINAMATH_GPT_lines_perpendicular_l83_8342


namespace NUMINAMATH_GPT_find_n_l83_8330

theorem find_n (n : ℕ) (h : 4 ^ 6 = 8 ^ n) : n = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_n_l83_8330


namespace NUMINAMATH_GPT_regression_line_intercept_l83_8391

theorem regression_line_intercept
  (x : ℕ → ℝ)
  (y : ℕ → ℝ)
  (h_x_sum : x 1 + x 2 + x 3 + x 4 + x 5 + x 6 = 10)
  (h_y_sum : y 1 + y 2 + y 3 + y 4 + y 5 + y 6 = 4) :
  ∃ a : ℝ, (∀ i, y i = (1 / 4) * x i + a) → a = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_regression_line_intercept_l83_8391


namespace NUMINAMATH_GPT_toys_produced_each_day_l83_8389

theorem toys_produced_each_day (weekly_production : ℕ) (days_worked : ℕ) 
  (h1 : weekly_production = 5500) (h2 : days_worked = 4) : 
  (weekly_production / days_worked = 1375) :=
sorry

end NUMINAMATH_GPT_toys_produced_each_day_l83_8389


namespace NUMINAMATH_GPT_multiples_of_15_between_12_and_152_l83_8313

theorem multiples_of_15_between_12_and_152 : 
  ∃ n : ℕ, n = 10 ∧ ∀ m : ℕ, (m * 15 > 12 ∧ m * 15 < 152) ↔ (1 ≤ m ∧ m ≤ 10) :=
by
  sorry

end NUMINAMATH_GPT_multiples_of_15_between_12_and_152_l83_8313


namespace NUMINAMATH_GPT_coefficient_x2_expansion_l83_8397

theorem coefficient_x2_expansion : 
  let binomial_coeff (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k
  let expansion_coeff (a b : ℤ) (n k : ℕ) : ℤ := (b ^ k) * (binomial_coeff n k) * (a ^ (n - k))
  (expansion_coeff 1 (-2) 4 2) = 24 :=
by
  let binomial_coeff (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k
  let expansion_coeff (a b : ℤ) (n k : ℕ) : ℤ := (b ^ k) * (binomial_coeff n k) * (a ^ (n - k))
  have coeff : ℤ := expansion_coeff 1 (-2) 4 2
  sorry -- Proof goes here

end NUMINAMATH_GPT_coefficient_x2_expansion_l83_8397


namespace NUMINAMATH_GPT_watched_movies_count_l83_8348

theorem watched_movies_count {M : ℕ} (total_books total_movies read_books : ℕ) 
  (h1 : total_books = 15) (h2 : total_movies = 14) (h3 : read_books = 11) 
  (h4 : read_books = M + 1) : M = 10 :=
by
  sorry

end NUMINAMATH_GPT_watched_movies_count_l83_8348


namespace NUMINAMATH_GPT_mrs_bil_earnings_percentage_in_may_l83_8373

theorem mrs_bil_earnings_percentage_in_may
  (M F : ℝ)
  (h₁ : 1.10 * M / (1.10 * M + F) = 0.7196) :
  M / (M + F) = 0.70 :=
sorry

end NUMINAMATH_GPT_mrs_bil_earnings_percentage_in_may_l83_8373


namespace NUMINAMATH_GPT_no_solution_for_s_l83_8317

theorem no_solution_for_s : ∀ s : ℝ,
  (s^2 - 6 * s + 8) / (s^2 - 9 * s + 20) ≠ (s^2 - 3 * s - 18) / (s^2 - 2 * s - 15) :=
by
  intros s
  sorry

end NUMINAMATH_GPT_no_solution_for_s_l83_8317


namespace NUMINAMATH_GPT_cost_of_cookies_equal_3_l83_8381

def selling_price : ℝ := 1.5
def cost_price : ℝ := 1
def number_of_bracelets : ℕ := 12
def amount_left : ℝ := 3

theorem cost_of_cookies_equal_3 : 
  (selling_price - cost_price) * number_of_bracelets - amount_left = 3 := by
  sorry

end NUMINAMATH_GPT_cost_of_cookies_equal_3_l83_8381


namespace NUMINAMATH_GPT_arthur_reading_pages_l83_8306

theorem arthur_reading_pages :
  let total_goal : ℕ := 800
  let pages_read_from_500_book : ℕ := 500 * 80 / 100 -- 80% of 500 pages
  let pages_read_from_1000_book : ℕ := 1000 / 5 -- 1/5 of 1000 pages
  let total_pages_read : ℕ := pages_read_from_500_book + pages_read_from_1000_book
  let remaining_pages : ℕ := total_goal - total_pages_read
  remaining_pages = 200 :=
by
  -- placeholder for actual proof
  sorry

end NUMINAMATH_GPT_arthur_reading_pages_l83_8306


namespace NUMINAMATH_GPT_correct_blanks_l83_8354

def fill_in_blanks (category : String) (plural_noun : String) : String :=
  "For many, winning remains " ++ category ++ " dream, but they continue trying their luck as there're always " ++ plural_noun ++ " chances that they might succeed."

theorem correct_blanks :
  fill_in_blanks "a" "" = "For many, winning remains a dream, but they continue trying their luck as there're always chances that they might succeed." :=
sorry

end NUMINAMATH_GPT_correct_blanks_l83_8354


namespace NUMINAMATH_GPT_integer_values_not_satisfying_inequality_l83_8360

theorem integer_values_not_satisfying_inequality :
  (∃ x : ℤ, ¬(3 * x^2 + 17 * x + 28 > 25)) ∧ (∃ x1 x2 : ℤ, x1 = -2 ∧ x2 = -1) ∧
  ∀ x : ℤ, (x = -2 ∨ x = -1) -> ¬(3 * x^2 + 17 * x + 28 > 25) :=
by
  sorry

end NUMINAMATH_GPT_integer_values_not_satisfying_inequality_l83_8360


namespace NUMINAMATH_GPT_range_of_a_l83_8321

noncomputable def f (a x : ℝ) : ℝ := a * Real.log x - x^2 + (1 / 2) * a

theorem range_of_a (a : ℝ) :
  (∀ x > 0, f a x ≤ 0) ↔ (0 ≤ a ∧ a ≤ 2) :=
sorry

end NUMINAMATH_GPT_range_of_a_l83_8321


namespace NUMINAMATH_GPT_chuck_distance_l83_8379

theorem chuck_distance
  (total_time : ℝ) (out_speed : ℝ) (return_speed : ℝ) (D : ℝ)
  (h1 : total_time = 3)
  (h2 : out_speed = 16)
  (h3 : return_speed = 24)
  (h4 : D / out_speed + D / return_speed = total_time) :
  D = 28.80 :=
by
  sorry

end NUMINAMATH_GPT_chuck_distance_l83_8379


namespace NUMINAMATH_GPT_roots_of_polynomial_l83_8314

noncomputable def p (x : ℝ) : ℝ := x^3 - 3 * x^2 - x + 3

theorem roots_of_polynomial : {x : ℝ | p x = 0} = {1, -1, 3} :=
by
  sorry

end NUMINAMATH_GPT_roots_of_polynomial_l83_8314


namespace NUMINAMATH_GPT_expression_equals_384_l83_8325

noncomputable def problem_expression : ℤ :=
  2021^4 - 4 * 2023^4 + 6 * 2025^4 - 4 * 2027^4 + 2029^4

theorem expression_equals_384 : problem_expression = 384 := by
  sorry

end NUMINAMATH_GPT_expression_equals_384_l83_8325


namespace NUMINAMATH_GPT_total_dots_not_visible_l83_8399

-- Define the conditions and variables
def total_dots_one_die : Nat := 1 + 2 + 3 + 4 + 5 + 6
def number_of_dice : Nat := 4
def total_dots_all_dice : Nat := number_of_dice * total_dots_one_die
def visible_numbers : List Nat := [6, 6, 4, 4, 3, 2, 1]

-- The question can be formalized as proving that the total number of dots not visible is 58
theorem total_dots_not_visible :
  total_dots_all_dice - visible_numbers.sum = 58 :=
by
  -- Statement only, proof skipped
  sorry

end NUMINAMATH_GPT_total_dots_not_visible_l83_8399


namespace NUMINAMATH_GPT_value_of_k_range_of_k_l83_8392

noncomputable def quadratic_eq_has_real_roots (k : ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), x₁ ^ 2 + (2 - 2 * k) * x₁ + k ^ 2 = 0 ∧
    x₂ ^ 2 + (2 - 2 * k) * x₂ + k ^ 2 = 0

def roots_condition (x₁ x₂ : ℝ) : Prop :=
  |(x₁ + x₂)| + 1 = x₁ * x₂

theorem value_of_k (k : ℝ) :
  quadratic_eq_has_real_roots k →
  (∀ (x₁ x₂ : ℝ), roots_condition x₁ x₂ → x₁ ^ 2 + (2 - 2 * k) * x₁ + k ^ 2 = 0 →
                    x₂ ^ 2 + (2 - 2 * k) * x₂ + k ^ 2 = 0 → k = -3) :=
by sorry

theorem range_of_k :
  ∃ (k : ℝ), quadratic_eq_has_real_roots k → k ≤ 1 :=
by sorry

end NUMINAMATH_GPT_value_of_k_range_of_k_l83_8392


namespace NUMINAMATH_GPT_min_odd_integers_is_zero_l83_8384

noncomputable def minOddIntegers (a b c d e f : ℤ) : ℕ :=
  if h₁ : a + b = 22 ∧ a + b + c + d = 36 ∧ a + b + c + d + e + f = 50 then
    0
  else
    6 -- default, just to match type expectations

theorem min_odd_integers_is_zero (a b c d e f : ℤ)
  (h₁ : a + b = 22)
  (h₂ : a + b + c + d = 36)
  (h₃ : a + b + c + d + e + f = 50) :
  minOddIntegers a b c d e f = 0 :=
  sorry

end NUMINAMATH_GPT_min_odd_integers_is_zero_l83_8384


namespace NUMINAMATH_GPT_value_of_n_l83_8331

theorem value_of_n (n : ℕ) (k : ℕ) (h : k = 11) (eqn : (1/2)^n * (1/81)^k = 1/18^22) : n = 22 :=
by
  sorry

end NUMINAMATH_GPT_value_of_n_l83_8331


namespace NUMINAMATH_GPT_sqrt_div_sqrt_eq_sqrt_fraction_l83_8345

theorem sqrt_div_sqrt_eq_sqrt_fraction
  (x y : ℝ)
  (h : ((1 / 2) ^ 2 + (1 / 3) ^ 2) / ((1 / 3) ^ 2 + (1 / 6) ^ 2) = 13 * x / (47 * y)) :
  (Real.sqrt x / Real.sqrt y) = (Real.sqrt 47 / Real.sqrt 5) :=
by
  sorry

end NUMINAMATH_GPT_sqrt_div_sqrt_eq_sqrt_fraction_l83_8345


namespace NUMINAMATH_GPT_baker_initial_cakes_l83_8361

theorem baker_initial_cakes (sold : ℕ) (left : ℕ) (initial : ℕ) 
  (h_sold : sold = 41) (h_left : left = 13) : 
  sold + left = initial → initial = 54 :=
by
  intros
  exact sorry

end NUMINAMATH_GPT_baker_initial_cakes_l83_8361


namespace NUMINAMATH_GPT_last_integer_in_sequence_l83_8344

theorem last_integer_in_sequence : ∀ (n : ℕ), n = 1000000 → (∀ k : ℕ, n = k * 3 → k * 3 < n) → n = 1000000 :=
by
  intro n hn hseq
  have h := hseq 333333 sorry
  exact hn

end NUMINAMATH_GPT_last_integer_in_sequence_l83_8344


namespace NUMINAMATH_GPT_work_rate_c_l83_8390

variables (rate_a rate_b rate_c : ℚ)

-- Given conditions
axiom h1 : rate_a + rate_b = 1 / 15
axiom h2 : rate_a + rate_b + rate_c = 1 / 6

theorem work_rate_c : rate_c = 1 / 10 :=
by sorry

end NUMINAMATH_GPT_work_rate_c_l83_8390


namespace NUMINAMATH_GPT_vector_addition_example_l83_8357

theorem vector_addition_example : 
  let v1 := (⟨-5, 3⟩ : ℝ × ℝ)
  let v2 := (⟨7, -6⟩ : ℝ × ℝ)
  v1 + v2 = (⟨2, -3⟩ : ℝ × ℝ) := 
by {
  sorry
}

end NUMINAMATH_GPT_vector_addition_example_l83_8357


namespace NUMINAMATH_GPT_probability_more_than_70_l83_8302

-- Definitions based on problem conditions
def P_A : ℝ := 0.15
def P_B : ℝ := 0.45
def P_C : ℝ := 0.25

-- Theorem to state that the probability of scoring more than 70 points is 0.85
theorem probability_more_than_70 (hA : P_A = 0.15) (hB : P_B = 0.45) (hC : P_C = 0.25):
  P_A + P_B + P_C = 0.85 :=
by
  rw [hA, hB, hC]
  sorry

end NUMINAMATH_GPT_probability_more_than_70_l83_8302


namespace NUMINAMATH_GPT_anusha_share_l83_8324

theorem anusha_share (A B E D G X : ℝ) 
  (h1: 20 * A = X)
  (h2: 15 * B = X)
  (h3: 8 * E = X)
  (h4: 12 * D = X)
  (h5: 10 * G = X)
  (h6: A + B + E + D + G = 950) : 
  A = 112 := 
by 
  sorry

end NUMINAMATH_GPT_anusha_share_l83_8324


namespace NUMINAMATH_GPT_alice_bob_coffee_shop_spending_l83_8369

theorem alice_bob_coffee_shop_spending (A B : ℝ) (h1 : B = 0.5 * A) (h2 : A = B + 15) : A + B = 45 :=
by
  sorry

end NUMINAMATH_GPT_alice_bob_coffee_shop_spending_l83_8369


namespace NUMINAMATH_GPT_polarBearDailyFish_l83_8338

-- Define the conditions
def polarBearDailyTrout : ℝ := 0.2
def polarBearDailySalmon : ℝ := 0.4

-- Define the statement to be proven
theorem polarBearDailyFish : polarBearDailyTrout + polarBearDailySalmon = 0.6 :=
by
  sorry

end NUMINAMATH_GPT_polarBearDailyFish_l83_8338


namespace NUMINAMATH_GPT_solve_system_of_equations_l83_8365

variable (a b c : Real)

def K : Real := a * b * c + a^2 * c + c^2 * b + b^2 * a

theorem solve_system_of_equations 
    (h₁ : (a + b) * (a - b) * (b + c) * (b - c) * (c + a) * (c - a) ≠ 0)
    (h₂ : K a b c ≠ 0) :
    ∃ (x y z : Real), 
    x = b^2 - c^2 ∧
    y = c^2 - a^2 ∧
    z = a^2 - b^2 ∧
    (x / (b + c) + y / (c - a) = a + b) ∧
    (y / (c + a) + z / (a - b) = b + c) ∧
    (z / (a + b) + x / (b - c) = c + a) :=
by
  sorry

end NUMINAMATH_GPT_solve_system_of_equations_l83_8365


namespace NUMINAMATH_GPT_candy_in_each_box_l83_8309

theorem candy_in_each_box (C K : ℕ) (h1 : 6 * C + 4 * K = 90) (h2 : C = K) : C = 9 :=
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_candy_in_each_box_l83_8309
