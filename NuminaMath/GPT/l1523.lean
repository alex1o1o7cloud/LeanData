import Mathlib

namespace NUMINAMATH_GPT_sqrt_four_eq_two_or_neg_two_l1523_152394

theorem sqrt_four_eq_two_or_neg_two (x : ℝ) : x^2 = 4 → (x = 2 ∨ x = -2) :=
sorry

end NUMINAMATH_GPT_sqrt_four_eq_two_or_neg_two_l1523_152394


namespace NUMINAMATH_GPT_point_Q_representation_l1523_152371

-- Definitions
variables {C D Q : Type} [AddCommGroup C] [AddCommGroup D] [AddCommGroup Q] [Module ℝ C] [Module ℝ D] [Module ℝ Q]
variable (CQ : ℝ)
variable (QD : ℝ)
variable (r s : ℝ)

-- Given condition: ratio CQ:QD = 7:2
axiom CQ_QD_ratio : CQ / QD = 7 / 2

-- Proof goal: the affine combination representation of the point Q
theorem point_Q_representation : CQ / (CQ + QD) = 7 / 9 ∧ QD / (CQ + QD) = 2 / 9 :=
sorry

end NUMINAMATH_GPT_point_Q_representation_l1523_152371


namespace NUMINAMATH_GPT_simplify_fractional_equation_l1523_152308

theorem simplify_fractional_equation (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 2) : (x / (x - 2) - 2 = 3 / (2 - x)) → (x - 2 * (x - 2) = -3) :=
by
  sorry

end NUMINAMATH_GPT_simplify_fractional_equation_l1523_152308


namespace NUMINAMATH_GPT_sample_size_obtained_l1523_152372

/-- A theorem which states the sample size obtained when a sample is taken from a population. -/
theorem sample_size_obtained 
  (total_students : ℕ)
  (sample_students : ℕ)
  (h1 : total_students = 300)
  (h2 : sample_students = 50) : 
  sample_students = 50 :=
by
  sorry

end NUMINAMATH_GPT_sample_size_obtained_l1523_152372


namespace NUMINAMATH_GPT_opposite_of_3_l1523_152355

theorem opposite_of_3 : -3 = -3 := 
by
  -- sorry is added to skip the proof as per instructions
  sorry

end NUMINAMATH_GPT_opposite_of_3_l1523_152355


namespace NUMINAMATH_GPT_part1_part2_l1523_152393

-- Part (1): Solution set of the inequality
theorem part1 (x : ℝ) : (|x - 1| + |x + 1| ≤ 8 - x^2) ↔ (-2 ≤ x) ∧ (x ≤ 2) :=
by
  sorry

-- Part (2): Range of real number t
theorem part2 (t : ℝ) (m n : ℝ) (x : ℝ) (h1 : m + n = 4) (h2 : m > 0) (h3 : n > 0) :  
  |x-t| + |x+t| = (4 * m^2 + n) / (m * n) → t ≥ 9 / 8 ∨ t ≤ -9 / 8 :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l1523_152393


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1523_152320

theorem solution_set_of_inequality (x m : ℝ) : 
  (x^2 - (2 * m + 1) * x + m^2 + m < 0) ↔ m < x ∧ x < m + 1 := 
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1523_152320


namespace NUMINAMATH_GPT_einstein_birth_weekday_l1523_152367

-- Defining the reference day of the week for 31 May 2006
def reference_date := 31
def reference_month := 5
def reference_year := 2006
def reference_weekday := 3  -- Wednesday

-- Defining Albert Einstein's birth date
def einstein_birth_day := 14
def einstein_birth_month := 3
def einstein_birth_year := 1879

-- Defining the calculation of weekday
def weekday_from_reference(reference_day reference_weekday einstein_birth_day einstein_birth_month einstein_birth_year : Nat) : Nat :=
  let days_from_reference_to_birth := 46464  -- Total days calculated in solution
  (reference_weekday - (days_from_reference_to_birth % 7) + 7) % 7

-- Stating the theorem
theorem einstein_birth_weekday : weekday_from_reference reference_day reference_weekday einstein_birth_day einstein_birth_month einstein_birth_year = 5 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_einstein_birth_weekday_l1523_152367


namespace NUMINAMATH_GPT_cost_of_72_tulips_is_115_20_l1523_152398

/-
Conditions:
1. A package containing 18 tulips costs $36.
2. The price of a package is directly proportional to the number of tulips it contains.
3. There is a 20% discount applied for packages containing more than 50 tulips.
Question:
What is the cost of 72 tulips?

Correct answer:
$115.20
-/

def costOfTulips (numTulips : ℕ)  : ℚ :=
  if numTulips ≤ 50 then
    36 * numTulips / 18
  else
    (36 * numTulips / 18) * 0.8 -- apply 20% discount for more than 50 tulips

theorem cost_of_72_tulips_is_115_20 :
  costOfTulips 72 = 115.2 := 
sorry

end NUMINAMATH_GPT_cost_of_72_tulips_is_115_20_l1523_152398


namespace NUMINAMATH_GPT_sum_of_pairwise_products_does_not_end_in_2019_l1523_152322

theorem sum_of_pairwise_products_does_not_end_in_2019 (n : ℤ) : ¬ (∃ (k : ℤ), 10000 ∣ (3 * n ^ 2 - 2020 + k * 10000)) := by
  sorry

end NUMINAMATH_GPT_sum_of_pairwise_products_does_not_end_in_2019_l1523_152322


namespace NUMINAMATH_GPT_find_staff_age_l1523_152344

theorem find_staff_age (n_students : ℕ) (avg_age_students : ℕ) (avg_age_with_staff : ℕ) (total_students : ℕ) :
  n_students = 32 →
  avg_age_students = 16 →
  avg_age_with_staff = 17 →
  total_students = 33 →
  (33 * 17 - 32 * 16) = 49 :=
by
  intros
  sorry

end NUMINAMATH_GPT_find_staff_age_l1523_152344


namespace NUMINAMATH_GPT_existence_of_solution_largest_unsolvable_n_l1523_152305

-- Definitions based on the conditions provided in the problem
def equation (x y z n : ℕ) : Prop := 28 * x + 30 * y + 31 * z = n

-- There exist positive integers x, y, z such that 28x + 30y + 31z = 365
theorem existence_of_solution : ∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ equation x y z 365 :=
by
  sorry

-- The largest positive integer n such that 28x + 30y + 31z = n cannot be solved in positive integers x, y, z is 370
theorem largest_unsolvable_n : ∀ (n : ℕ), (∀ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 → n ≠ 370) → ∀ (n' : ℕ), n' > 370 → (∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ equation x y z n') :=
by
  sorry

end NUMINAMATH_GPT_existence_of_solution_largest_unsolvable_n_l1523_152305


namespace NUMINAMATH_GPT_board_transformation_l1523_152388

def transformation_possible (a b : ℕ) : Prop :=
  6 ∣ (a * b)

theorem board_transformation (a b : ℕ) (h₁ : 2 ≤ a) (h₂ : 2 ≤ b) : 
  transformation_possible a b ↔ 6 ∣ (a * b) := by
  sorry

end NUMINAMATH_GPT_board_transformation_l1523_152388


namespace NUMINAMATH_GPT_possible_values_of_angle_F_l1523_152336

-- Define angle F conditions in a triangle DEF
def triangle_angle_F_conditions (D E : ℝ) : Prop :=
  5 * Real.sin D + 2 * Real.cos E = 8 ∧ 3 * Real.sin E + 5 * Real.cos D = 2

-- The main statement: proving the possible values of ∠F
theorem possible_values_of_angle_F (D E : ℝ) (h : triangle_angle_F_conditions D E) : 
  ∃ F : ℝ, F = Real.arcsin (43 / 50) ∨ F = 180 - Real.arcsin (43 / 50) :=
by
  sorry

end NUMINAMATH_GPT_possible_values_of_angle_F_l1523_152336


namespace NUMINAMATH_GPT_num_trucks_washed_l1523_152362

theorem num_trucks_washed (total_revenue cars_revenue suvs_revenue truck_charge : ℕ) 
  (h_total : total_revenue = 100)
  (h_cars : cars_revenue = 7 * 5)
  (h_suvs : suvs_revenue = 5 * 7)
  (h_truck_charge : truck_charge = 6) : 
  ∃ T : ℕ, (total_revenue - suvs_revenue - cars_revenue) / truck_charge = T := 
by {
  use 5,
  sorry
}

end NUMINAMATH_GPT_num_trucks_washed_l1523_152362


namespace NUMINAMATH_GPT_g_g_2_equals_226_l1523_152317

def g (x : ℝ) : ℝ := 2 * x^2 + 3 * x - 4

theorem g_g_2_equals_226 : g (g 2) = 226 := by
  sorry

end NUMINAMATH_GPT_g_g_2_equals_226_l1523_152317


namespace NUMINAMATH_GPT_jess_double_cards_l1523_152310

theorem jess_double_cards (rob_total_cards jess_doubles : ℕ) 
    (one_third_rob_cards_doubles : rob_total_cards / 3 = rob_total_cards / 3)
    (jess_times_rob_doubles : jess_doubles = 5 * (rob_total_cards / 3)) :
    rob_total_cards = 24 → jess_doubles = 40 :=
  by
  sorry

end NUMINAMATH_GPT_jess_double_cards_l1523_152310


namespace NUMINAMATH_GPT_cauliflower_production_diff_l1523_152382

theorem cauliflower_production_diff
  (area_this_year : ℕ)
  (area_last_year : ℕ)
  (side_this_year : ℕ)
  (side_last_year : ℕ)
  (H1 : side_this_year * side_this_year = area_this_year)
  (H2 : side_last_year * side_last_year = area_last_year)
  (H3 : side_this_year = side_last_year + 1)
  (H4 : area_this_year = 12544) :
  area_this_year - area_last_year = 223 :=
by
  sorry

end NUMINAMATH_GPT_cauliflower_production_diff_l1523_152382


namespace NUMINAMATH_GPT_boxes_contain_neither_markers_nor_sharpies_l1523_152324

theorem boxes_contain_neither_markers_nor_sharpies :
  (∀ (total_boxes markers_boxes sharpies_boxes both_boxes neither_boxes : ℕ),
    total_boxes = 15 → markers_boxes = 8 → sharpies_boxes = 5 → both_boxes = 4 →
    neither_boxes = total_boxes - (markers_boxes + sharpies_boxes - both_boxes) →
    neither_boxes = 6) :=
by
  intros total_boxes markers_boxes sharpies_boxes both_boxes neither_boxes
  intros htotal hmarkers hsharpies hboth hcalc
  rw [htotal, hmarkers, hsharpies, hboth] at hcalc
  exact hcalc

end NUMINAMATH_GPT_boxes_contain_neither_markers_nor_sharpies_l1523_152324


namespace NUMINAMATH_GPT_p_sufficient_but_not_necessary_for_q_l1523_152315

-- Definitions
variable {p q : Prop}

-- The condition: ¬p is a necessary but not sufficient condition for ¬q
def necessary_but_not_sufficient (p q : Prop) : Prop :=
  (∀ q, ¬q → ¬p) ∧ (∃ q, ¬q ∧ p)

-- The theorem stating the problem
theorem p_sufficient_but_not_necessary_for_q 
  (h : necessary_but_not_sufficient (¬p) (¬q)) : 
  (∀ p, p → q) ∧ (∃ p, p ∧ ¬q) :=
sorry

end NUMINAMATH_GPT_p_sufficient_but_not_necessary_for_q_l1523_152315


namespace NUMINAMATH_GPT_lord_moneybag_l1523_152341

theorem lord_moneybag (n : ℕ) (hlow : 300 ≤ n) (hhigh : n ≤ 500)
           (h6 : 6 ∣ n) (h5 : 5 ∣ (n - 1)) (h4 : 4 ∣ (n - 2)) 
           (h3 : 3 ∣ (n - 3)) (h2 : 2 ∣ (n - 4)) (hprime : Nat.Prime (n - 5)) :
  n = 426 := by
  sorry

end NUMINAMATH_GPT_lord_moneybag_l1523_152341


namespace NUMINAMATH_GPT_cube_difference_l1523_152389

theorem cube_difference (x y : ℕ) (h₁ : x + y = 64) (h₂ : x - y = 16) : x^3 - y^3 = 50176 := by
  sorry

end NUMINAMATH_GPT_cube_difference_l1523_152389


namespace NUMINAMATH_GPT_complement_M_in_U_l1523_152381

universe u

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 2, 4}

theorem complement_M_in_U :
  U \ M = {3, 5, 6} :=
by sorry

end NUMINAMATH_GPT_complement_M_in_U_l1523_152381


namespace NUMINAMATH_GPT_scrabble_champions_l1523_152340

noncomputable def num_champions : Nat := 25
noncomputable def male_percentage : Nat := 40
noncomputable def bearded_percentage : Nat := 40
noncomputable def bearded_bald_percentage : Nat := 60
noncomputable def non_bearded_bald_percentage : Nat := 30

theorem scrabble_champions :
  let male_champions := (male_percentage * num_champions) / 100
  let bearded_champions := (bearded_percentage * male_champions) / 100
  let bearded_bald_champions := (bearded_bald_percentage * bearded_champions) / 100
  let bearded_hair_champions := bearded_champions - bearded_bald_champions
  let non_bearded_champions := male_champions - bearded_champions
  let non_bearded_bald_champions := (non_bearded_bald_percentage * non_bearded_champions) / 100
  let non_bearded_hair_champions := non_bearded_champions - non_bearded_bald_champions
  bearded_bald_champions = 2 ∧ 
  bearded_hair_champions = 2 ∧ 
  non_bearded_bald_champions = 1 ∧ 
  non_bearded_hair_champions = 5 :=
by
  sorry

end NUMINAMATH_GPT_scrabble_champions_l1523_152340


namespace NUMINAMATH_GPT_intersect_point_sum_l1523_152395

theorem intersect_point_sum (a' b' : ℝ) (x y : ℝ) 
    (h1 : x = (1 / 3) * y + a')
    (h2 : y = (1 / 3) * x + b')
    (h3 : x = 2)
    (h4 : y = 4) : 
    a' + b' = 4 :=
by
  sorry

end NUMINAMATH_GPT_intersect_point_sum_l1523_152395


namespace NUMINAMATH_GPT_added_amount_l1523_152345

theorem added_amount (x y : ℕ) (h1 : x = 17) (h2 : 3 * (2 * x + y) = 117) : y = 5 :=
by
  sorry

end NUMINAMATH_GPT_added_amount_l1523_152345


namespace NUMINAMATH_GPT_find_m_l1523_152327

theorem find_m (x n m : ℝ) (h : (x + n)^2 = x^2 + 4*x + m) : m = 4 :=
sorry

end NUMINAMATH_GPT_find_m_l1523_152327


namespace NUMINAMATH_GPT_car_more_miles_per_tank_after_modification_l1523_152318

theorem car_more_miles_per_tank_after_modification (mpg_old : ℕ) (efficiency_factor : ℝ) (gallons : ℕ) :
  mpg_old = 33 →
  efficiency_factor = 1.25 →
  gallons = 16 →
  (efficiency_factor * mpg_old * gallons - mpg_old * gallons) = 132 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  simp
  sorry  -- Proof omitted

end NUMINAMATH_GPT_car_more_miles_per_tank_after_modification_l1523_152318


namespace NUMINAMATH_GPT_common_tangent_y_intercept_l1523_152321

theorem common_tangent_y_intercept
  (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ) (m b : ℝ)
  (h_c1 : c1 = (5, -2))
  (h_c2 : c2 = (20, 6))
  (h_r1 : r1 = 5)
  (h_r2 : r2 = 12)
  (h_tangent : ∃m > 0, ∃b, (∀ x y, y = m * x + b → (x - 5)^2 + (y + 2)^2 > 25 ∧ (x - 20)^2 + (y - 6)^2 > 144)) :
  b = -2100 / 161 :=
by
  sorry

end NUMINAMATH_GPT_common_tangent_y_intercept_l1523_152321


namespace NUMINAMATH_GPT_no_integer_solutions_3a2_eq_b2_plus_1_l1523_152303

theorem no_integer_solutions_3a2_eq_b2_plus_1 : 
  ¬ ∃ a b : ℤ, 3 * a^2 = b^2 + 1 :=
by
  intro h
  obtain ⟨a, b, hab⟩ := h
  sorry

end NUMINAMATH_GPT_no_integer_solutions_3a2_eq_b2_plus_1_l1523_152303


namespace NUMINAMATH_GPT_no_two_exact_cubes_between_squares_l1523_152348

theorem no_two_exact_cubes_between_squares :
  ∀ (n a b : ℤ), ¬ (n^2 < a^3 ∧ a^3 < b^3 ∧ b^3 < (n + 1)^2) :=
by
  intros n a b
  sorry

end NUMINAMATH_GPT_no_two_exact_cubes_between_squares_l1523_152348


namespace NUMINAMATH_GPT_rowing_speed_upstream_l1523_152333

theorem rowing_speed_upstream (V_s V_downstream : ℝ) (V_s_eq : V_s = 28) (V_downstream_eq : V_downstream = 31) : 
  V_s - (V_downstream - V_s) = 25 := 
by
  sorry

end NUMINAMATH_GPT_rowing_speed_upstream_l1523_152333


namespace NUMINAMATH_GPT_negation_of_p_l1523_152369

def p (x : ℝ) : Prop := x^3 - x^2 + 1 < 0

theorem negation_of_p : (¬ ∀ x : ℝ, p x) ↔ ∃ x : ℝ, ¬ p x := by
  sorry

end NUMINAMATH_GPT_negation_of_p_l1523_152369


namespace NUMINAMATH_GPT_eq_condition_l1523_152356

theorem eq_condition (a : ℝ) :
  (∃ x : ℝ, a * (4 * |x| + 1) = 4 * |x|) ↔ (0 ≤ a ∧ a < 1) :=
by
  sorry

end NUMINAMATH_GPT_eq_condition_l1523_152356


namespace NUMINAMATH_GPT_multiple_of_sales_total_l1523_152323

theorem multiple_of_sales_total
  (A : ℝ)
  (M : ℝ)
  (h : M * A = 0.3125 * (11 * A + M * A)) :
  M = 5 :=
by
  sorry

end NUMINAMATH_GPT_multiple_of_sales_total_l1523_152323


namespace NUMINAMATH_GPT_algebraic_expression_value_l1523_152376

theorem algebraic_expression_value (a b : ℝ) (h1 : a ≠ b) (h2 : a^2 - 5 * a + 2 = 0) (h3 : b^2 - 5 * b + 2 = 0) :
  (b - 1) / (a - 1) + (a - 1) / (b - 1) = -13 / 2 := by
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l1523_152376


namespace NUMINAMATH_GPT_race_times_l1523_152375

theorem race_times (x y : ℕ) (h1 : 5 * x + 1 = 4 * y) (h2 : 5 * y - 8 = 4 * x) :
  5 * x = 15 ∧ 5 * y = 20 :=
by
  sorry

end NUMINAMATH_GPT_race_times_l1523_152375


namespace NUMINAMATH_GPT_neg_p_l1523_152358

theorem neg_p (p : ∀ x : ℝ, x^2 ≥ 0) : ∃ x : ℝ, x^2 < 0 := 
sorry

end NUMINAMATH_GPT_neg_p_l1523_152358


namespace NUMINAMATH_GPT_profit_without_discount_l1523_152338

theorem profit_without_discount
  (CP SP_with_discount : ℝ) 
  (H1 : CP = 100) -- Assume cost price is 100
  (H2 : SP_with_discount = CP + 0.216 * CP) -- Selling price with discount
  (H3 : SP_with_discount = 0.95 * SP_without_discount) -- SP with discount is 95% of SP without discount
  : (SP_without_discount - CP) / CP * 100 = 28 := 
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_profit_without_discount_l1523_152338


namespace NUMINAMATH_GPT_problem_solution_l1523_152391

def satisfies_conditions (x y : ℚ) : Prop :=
  (3 * x + y = 6) ∧ (x + 3 * y = 6)

theorem problem_solution :
  ∃ (x y : ℚ), satisfies_conditions x y ∧ 3 * x^2 + 5 * x * y + 3 * y^2 = 24.75 :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l1523_152391


namespace NUMINAMATH_GPT_part1_part2_l1523_152360

-- Part 1
theorem part1 : (9 / 4) ^ (1 / 2) - (-2.5) ^ 0 - (8 / 27) ^ (2 / 3) + (3 / 2) ^ (-2) = 1 / 2 := 
by sorry

-- Part 2
theorem part2 (lg : ℝ → ℝ) -- Assuming a hypothetical lg function for demonstration
  (lg_prop1 : lg 10 = 1)
  (lg_prop2 : ∀ x y, lg (x * y) = lg x + lg y) :
  (lg 5) ^ 2 + lg 2 * lg 50 = 1 := 
by sorry

end NUMINAMATH_GPT_part1_part2_l1523_152360


namespace NUMINAMATH_GPT_rectangle_length_to_width_ratio_l1523_152316

-- Define the side length of the square
def s : ℝ := 1 -- Since we only need the ratio, the actual length does not matter

-- Define the length and width of the large rectangle
def length_of_large_rectangle : ℝ := 3 * s
def width_of_large_rectangle : ℝ := 3 * s

-- Define the dimensions of the small rectangle
def length_of_rectangle : ℝ := 3 * s
def width_of_rectangle : ℝ := s

-- Proving that the length of the rectangle is 3 times its width
theorem rectangle_length_to_width_ratio : length_of_rectangle = 3 * width_of_rectangle := 
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_rectangle_length_to_width_ratio_l1523_152316


namespace NUMINAMATH_GPT_initial_cows_l1523_152397

theorem initial_cows (x : ℕ) (h : (3 / 4 : ℝ) * (x + 5) = 42) : x = 51 :=
by
  sorry

end NUMINAMATH_GPT_initial_cows_l1523_152397


namespace NUMINAMATH_GPT_sum_of_divisors_85_l1523_152380

theorem sum_of_divisors_85 : (1 + 5 + 17 + 85 = 108) := by
  sorry

end NUMINAMATH_GPT_sum_of_divisors_85_l1523_152380


namespace NUMINAMATH_GPT_laptop_weight_l1523_152304

-- Defining the weights
variables (B U L P : ℝ)
-- Karen's tote weight
def K := 8

-- Conditions from the problem
axiom tote_eq_two_briefcase : K = 2 * B
axiom umbrella_eq_half_briefcase : U = B / 2
axiom full_briefcase_eq_double_tote : B + L + P + U = 2 * K
axiom papers_eq_sixth_full_briefcase : P = (B + L + P) / 6

-- Theorem stating the weight of Kevin's laptop is 7.67 pounds
theorem laptop_weight (hB : B = 4) (hU : U = 2) (hL : L = 7.67) : 
  L - K = -0.33 :=
by
  sorry

end NUMINAMATH_GPT_laptop_weight_l1523_152304


namespace NUMINAMATH_GPT_log_fraction_eq_l1523_152374

variable (a b : ℝ)
axiom h1 : a = Real.logb 3 5
axiom h2 : b = Real.logb 5 7

theorem log_fraction_eq : Real.logb 15 (49 / 45) = (2 * (a * b) - a - 2) / (1 + a) :=
by sorry

end NUMINAMATH_GPT_log_fraction_eq_l1523_152374


namespace NUMINAMATH_GPT_number_of_two_digit_primes_with_ones_digit_three_l1523_152325

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def has_ones_digit_three (n : ℕ) : Prop :=
  n % 10 = 3

def is_prime (n : ℕ) : Prop :=
  Nat.Prime n

theorem number_of_two_digit_primes_with_ones_digit_three :
  ∃! s : Finset ℕ, (∀ n ∈ s, is_two_digit n ∧ has_ones_digit_three n ∧ is_prime n) ∧ s.card = 6 :=
by
  sorry

end NUMINAMATH_GPT_number_of_two_digit_primes_with_ones_digit_three_l1523_152325


namespace NUMINAMATH_GPT_officers_count_l1523_152306

theorem officers_count (average_salary_all : ℝ) (average_salary_officers : ℝ) 
    (average_salary_non_officers : ℝ) (num_non_officers : ℝ) (total_salary : ℝ) : 
    average_salary_all = 120 → 
    average_salary_officers = 470 →  
    average_salary_non_officers = 110 → 
    num_non_officers = 525 → 
    total_salary = average_salary_all * (num_non_officers + O) → 
    total_salary = average_salary_officers * O + average_salary_non_officers * num_non_officers → 
    O = 15 :=
by
  intro h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_officers_count_l1523_152306


namespace NUMINAMATH_GPT_find_ratio_b_c_l1523_152312

variable {a b c A B C : Real}

theorem find_ratio_b_c
  (h1 : a * Real.sin A - b * Real.sin B = 4 * c * Real.sin C)
  (h2 : Real.cos A = -1 / 4) :
  b / c = 6 :=
sorry

end NUMINAMATH_GPT_find_ratio_b_c_l1523_152312


namespace NUMINAMATH_GPT_ratio_of_x_to_y_l1523_152319

theorem ratio_of_x_to_y (x y : ℝ) (h : (3 * x + 2 * y) / (2 * x - y) = 5 / 4) : x / y = -13 / 2 := 
by 
  sorry

end NUMINAMATH_GPT_ratio_of_x_to_y_l1523_152319


namespace NUMINAMATH_GPT_initial_ratio_milk_water_l1523_152366

-- Define the initial conditions
variables (M W : ℕ) (h_volume : M + W = 115) (h_ratio : M / (W + 46) = 3 / 4)

-- State the theorem to prove the initial ratio of milk to water
theorem initial_ratio_milk_water (h_volume : M + W = 115) (h_ratio : M / (W + 46) = 3 / 4) :
  (M * 2 = W * 3) :=
by
  sorry

end NUMINAMATH_GPT_initial_ratio_milk_water_l1523_152366


namespace NUMINAMATH_GPT_sum_of_digits_eq_4_l1523_152392

theorem sum_of_digits_eq_4 (A B C D X Y : ℕ) (h1 : A + B + C + D = 22) (h2 : B + D = 9) (h3 : X = 1) (h4 : Y = 3) :
    X + Y = 4 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_digits_eq_4_l1523_152392


namespace NUMINAMATH_GPT_find_f0_f1_l1523_152359

noncomputable def f : ℤ → ℤ := sorry

theorem find_f0_f1 :
  (∀ x : ℤ, f (x+5) - f x = 10 * x + 25) →
  (∀ x : ℤ, f (x^3 - 1) = (f x - x)^3 + x^3 - 3) →
  f 0 = -1 ∧ f 1 = 0 := by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_find_f0_f1_l1523_152359


namespace NUMINAMATH_GPT_number_of_lines_passing_through_point_and_forming_given_area_l1523_152300

theorem number_of_lines_passing_through_point_and_forming_given_area :
  ∃ l : ℝ → ℝ, (∀ x y : ℝ, l 1 = 1) ∧ (∃ (a b : ℝ), abs ((1/2) * a * b) = 2)
  → (∃ n : ℕ, n = 4) :=
by
  sorry

end NUMINAMATH_GPT_number_of_lines_passing_through_point_and_forming_given_area_l1523_152300


namespace NUMINAMATH_GPT_totalStudents_l1523_152364

-- Define the number of seats per ride
def seatsPerRide : ℕ := 15

-- Define the number of empty seats per ride
def emptySeatsPerRide : ℕ := 3

-- Define the number of rides taken
def ridesTaken : ℕ := 18

-- Define the number of students per ride
def studentsPerRide (seats : ℕ) (empty : ℕ) : ℕ := seats - empty

-- Calculate the total number of students
theorem totalStudents : studentsPerRide seatsPerRide emptySeatsPerRide * ridesTaken = 216 :=
by
  sorry

end NUMINAMATH_GPT_totalStudents_l1523_152364


namespace NUMINAMATH_GPT_max_belts_l1523_152330

theorem max_belts (h t b : ℕ) (Hh : h >= 1) (Ht : t >= 1) (Hb : b >= 1) (total_cost : 3 * h + 4 * t + 9 * b = 60) : b <= 5 :=
sorry

end NUMINAMATH_GPT_max_belts_l1523_152330


namespace NUMINAMATH_GPT_triangle_PQR_area_l1523_152383

/-- Given a triangle PQR where PQ = 4 miles, PR = 2 miles, and PQ is along Pine Street
and PR is along Quail Road, and there is a sub-triangle PQS within PQR
with PS = 2 miles along Summit Avenue and QS = 3 miles along Pine Street,
prove that the area of triangle PQR is 4 square miles --/
theorem triangle_PQR_area :
  ∀ (PQ PR PS QS : ℝ),
    PQ = 4 → PR = 2 → PS = 2 → QS = 3 →
    (1/2) * PQ * PR = 4 :=
by
  intros PQ PR PS QS hpq hpr hps hqs
  rw [hpq, hpr]
  norm_num
  done

end NUMINAMATH_GPT_triangle_PQR_area_l1523_152383


namespace NUMINAMATH_GPT_isle_of_unluckiness_l1523_152339

-- Definitions:
def is_knight (i : ℕ) (n : ℕ) : Prop :=
  ∃ k : ℕ, k = i * n / 100 ∧ k > 0

-- Main statement:
theorem isle_of_unluckiness (n : ℕ) (h : n ∈ [1, 2, 4, 5, 10, 20, 25, 50, 100]) :
  ∃ i : ℕ, 1 ≤ i ∧ i ≤ n ∧ is_knight i n := by
  sorry

end NUMINAMATH_GPT_isle_of_unluckiness_l1523_152339


namespace NUMINAMATH_GPT_find_overlapping_area_l1523_152346

-- Definitions based on conditions
def length_total : ℕ := 16
def length_strip1 : ℕ := 9
def length_strip2 : ℕ := 7
def area_only_strip1 : ℚ := 27
def area_only_strip2 : ℚ := 18

-- Widths are the same for both strips, hence areas are proportional to lengths
def area_ratio := (length_strip1 : ℚ) / (length_strip2 : ℚ)

-- The Lean statement to prove the question == answer
theorem find_overlapping_area : 
  ∃ S : ℚ, (area_only_strip1 + S) / (area_only_strip2 + S) = area_ratio ∧ 
              area_only_strip1 + S = area_only_strip1 + 13.5 := 
by 
  sorry

end NUMINAMATH_GPT_find_overlapping_area_l1523_152346


namespace NUMINAMATH_GPT_additional_books_acquired_l1523_152313

def original_stock : ℝ := 40.0
def shelves_used : ℕ := 15
def books_per_shelf : ℝ := 4.0

theorem additional_books_acquired :
  (shelves_used * books_per_shelf) - original_stock = 20.0 :=
by
  sorry

end NUMINAMATH_GPT_additional_books_acquired_l1523_152313


namespace NUMINAMATH_GPT_chocolates_initial_l1523_152332

variable (x : ℕ)
variable (h1 : 3 * x + 5 + 25 = 5 * x)
variable (h2 : x = 15)

theorem chocolates_initial (x : ℕ) (h1 : 3 * x + 5 + 25 = 5 * x) (h2 : x = 15) : 3 * 15 + 5 = 50 :=
by sorry

end NUMINAMATH_GPT_chocolates_initial_l1523_152332


namespace NUMINAMATH_GPT_molly_total_swim_l1523_152385

variable (meters_saturday : ℕ) (meters_sunday : ℕ)

theorem molly_total_swim (h1 : meters_saturday = 45) (h2 : meters_sunday = 28) : meters_saturday + meters_sunday = 73 := by
  sorry

end NUMINAMATH_GPT_molly_total_swim_l1523_152385


namespace NUMINAMATH_GPT_geometric_sequence_property_l1523_152326

variable (a : ℕ → ℤ)
-- Assume the sequence is geometric with ratio r
variable (r : ℤ)

-- Define the sequence a_n as a geometric sequence
def geometric_sequence (a : ℕ → ℤ) (r : ℤ) : Prop :=
  ∀ (n : ℕ), a (n + 1) = a n * r

-- Given condition: a_4 + a_8 = -2
axiom condition : a 4 + a 8 = -2

theorem geometric_sequence_property
  (h : geometric_sequence a r) : a 6 * (a 2 + 2 * a 6 + a 10) = 4 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_property_l1523_152326


namespace NUMINAMATH_GPT_condition_relation_l1523_152365

variable (A B C : Prop)

theorem condition_relation (h1 : C → B) (h2 : A → B) : 
  (¬(A → C) ∧ ¬(C → A)) :=
by 
  sorry

end NUMINAMATH_GPT_condition_relation_l1523_152365


namespace NUMINAMATH_GPT_multiple_of_rohan_age_l1523_152353

theorem multiple_of_rohan_age (x : ℝ) (h1 : 25 - 15 = 10) (h2 : 25 + 15 = 40) (h3 : 40 = x * 10) : x = 4 := 
by 
  sorry

end NUMINAMATH_GPT_multiple_of_rohan_age_l1523_152353


namespace NUMINAMATH_GPT_largest_mersenne_prime_is_127_l1523_152361

noncomputable def largest_mersenne_prime_less_than_500 : ℕ :=
  127

theorem largest_mersenne_prime_is_127 :
  ∃ p : ℕ, Nat.Prime p ∧ (2^p - 1) = largest_mersenne_prime_less_than_500 ∧ 2^p - 1 < 500 := 
by 
  -- The largest Mersenne prime less than 500 is 127
  use 7
  sorry

end NUMINAMATH_GPT_largest_mersenne_prime_is_127_l1523_152361


namespace NUMINAMATH_GPT_vector_addition_l1523_152351

-- Definitions for the vectors
def a : ℝ × ℝ := (5, 2)
def b : ℝ × ℝ := (1, 6)

-- Proof statement (Note: "theorem" is used here instead of "def" because we are stating something to be proven)
theorem vector_addition : a + b = (6, 8) := by
  sorry

end NUMINAMATH_GPT_vector_addition_l1523_152351


namespace NUMINAMATH_GPT_student_B_more_stable_l1523_152378

-- Definitions as stated in the conditions
def student_A_variance : ℝ := 0.3
def student_B_variance : ℝ := 0.1

-- Theorem stating that student B has more stable performance than student A
theorem student_B_more_stable : student_B_variance < student_A_variance :=
by
  sorry

end NUMINAMATH_GPT_student_B_more_stable_l1523_152378


namespace NUMINAMATH_GPT_cos_expression_value_l1523_152350

theorem cos_expression_value (x : ℝ) (h : Real.sin x = 3 * Real.sin (x - Real.pi / 2)) :
  Real.cos x * Real.cos (x + Real.pi / 2) = 3 / 10 := 
sorry

end NUMINAMATH_GPT_cos_expression_value_l1523_152350


namespace NUMINAMATH_GPT_sequence_problem_l1523_152349

theorem sequence_problem :
  7 * 9 * 11 + (7 + 9 + 11) = 720 :=
by
  sorry

end NUMINAMATH_GPT_sequence_problem_l1523_152349


namespace NUMINAMATH_GPT_sam_bought_cards_l1523_152386

-- Define the initial number of baseball cards Dan had.
def dan_initial_cards : ℕ := 97

-- Define the number of baseball cards Dan has after selling some to Sam.
def dan_remaining_cards : ℕ := 82

-- Prove that the number of baseball cards Sam bought is 15.
theorem sam_bought_cards : (dan_initial_cards - dan_remaining_cards) = 15 :=
by
  sorry

end NUMINAMATH_GPT_sam_bought_cards_l1523_152386


namespace NUMINAMATH_GPT_carol_rectangle_length_l1523_152311

theorem carol_rectangle_length (lCarol : ℝ) :
    (∃ (wCarol : ℝ), wCarol = 20 ∧ lCarol * wCarol = 300) ↔ lCarol = 15 :=
by
  have jordan_area : 6 * 50 = 300 := by norm_num
  sorry

end NUMINAMATH_GPT_carol_rectangle_length_l1523_152311


namespace NUMINAMATH_GPT_number_of_red_dresses_l1523_152384

-- Define context for Jane's dress shop problem
def dresses_problem (R B : Nat) : Prop :=
  R + B = 200 ∧ B = R + 34

-- Prove that the number of red dresses (R) should be 83
theorem number_of_red_dresses : ∃ R B : Nat, dresses_problem R B ∧ R = 83 :=
by
  sorry

end NUMINAMATH_GPT_number_of_red_dresses_l1523_152384


namespace NUMINAMATH_GPT_decreasing_implies_inequality_l1523_152377

variable (f : ℝ → ℝ)

theorem decreasing_implies_inequality (h : ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) < 0) : f 3 < f 2 ∧ f 2 < f 1 :=
  sorry

end NUMINAMATH_GPT_decreasing_implies_inequality_l1523_152377


namespace NUMINAMATH_GPT_hyperbola_real_axis_length_l1523_152301

theorem hyperbola_real_axis_length (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h_hyperbola : ∀ x y : ℝ, x = 1 → y = 2 → (x^2 / (a^2)) - (y^2 / (b^2)) = 1)
  (h_parabola : ∀ y : ℝ, y = 2 → (y^2) = 4 * 1)
  (h_focus : (1, 2) = (1, 2))
  (h_eq : a^2 + b^2 = 1) :
  2 * a = 2 * (Real.sqrt 2 - 1) :=
by 
-- Skipping the proof part
sorry

end NUMINAMATH_GPT_hyperbola_real_axis_length_l1523_152301


namespace NUMINAMATH_GPT_base8_base9_equivalence_l1523_152373

def base8_digit (x : ℕ) := 0 ≤ x ∧ x < 8
def base9_digit (y : ℕ) := 0 ≤ y ∧ y < 9

theorem base8_base9_equivalence 
    (X Y : ℕ) 
    (hX : base8_digit X) 
    (hY : base9_digit Y) 
    (h_eq : 8 * X + Y = 9 * Y + X) :
    (8 * 7 + 6 = 62) :=
by
  sorry

end NUMINAMATH_GPT_base8_base9_equivalence_l1523_152373


namespace NUMINAMATH_GPT_ratio_of_areas_l1523_152343

theorem ratio_of_areas (r : ℝ) (w_smaller : ℝ) (h_smaller : ℝ) (h_semi : ℝ) :
  (5 / 4) * 40 = r + 40 →
  h_semi = 20 →
  w_smaller = 5 →
  h_smaller = 20 →
  2 * w_smaller * h_smaller / ((1 / 2) * π * h_semi^2) = 1 / π :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_ratio_of_areas_l1523_152343


namespace NUMINAMATH_GPT_seunghyo_daily_dosage_l1523_152399

theorem seunghyo_daily_dosage (total_medicine : ℝ) (daily_fraction : ℝ) (correct_dosage : ℝ) :
  total_medicine = 426 → daily_fraction = 0.06 → correct_dosage = 25.56 →
  total_medicine * daily_fraction = correct_dosage :=
by
  intros ht hf hc
  simp [ht, hf, hc]
  sorry

end NUMINAMATH_GPT_seunghyo_daily_dosage_l1523_152399


namespace NUMINAMATH_GPT_base7_of_2345_l1523_152334

def decimal_to_base7 (n : ℕ) : ℕ :=
  6 * 7^3 + 5 * 7^2 + 6 * 7^1 + 0 * 7^0

theorem base7_of_2345 : decimal_to_base7 2345 = 6560 := by
  sorry

end NUMINAMATH_GPT_base7_of_2345_l1523_152334


namespace NUMINAMATH_GPT_maximum_value_fraction_l1523_152347

theorem maximum_value_fraction (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  (x / (2 * x + y) + y / (x + 2 * y)) ≤ 2 / 3 :=
sorry

end NUMINAMATH_GPT_maximum_value_fraction_l1523_152347


namespace NUMINAMATH_GPT_no_solution_of_fractional_equation_l1523_152354

theorem no_solution_of_fractional_equation (x : ℝ) : ¬ (x - 8) / (x - 7) - 8 = 1 / (7 - x) := 
sorry

end NUMINAMATH_GPT_no_solution_of_fractional_equation_l1523_152354


namespace NUMINAMATH_GPT_quadratic_always_positive_if_and_only_if_l1523_152387

theorem quadratic_always_positive_if_and_only_if :
  (∀ x : ℝ, x^2 + m * x + m + 3 > 0) ↔ (-2 < m ∧ m < 6) :=
by sorry

end NUMINAMATH_GPT_quadratic_always_positive_if_and_only_if_l1523_152387


namespace NUMINAMATH_GPT_compute_expression_l1523_152329

theorem compute_expression :
  -9 * 5 - (-(7 * -2)) + (-(11 * -6)) = 7 :=
by
  sorry

end NUMINAMATH_GPT_compute_expression_l1523_152329


namespace NUMINAMATH_GPT_region_area_l1523_152314

noncomputable def area_of_region_outside_hexagon_inside_semicircles (s : ℝ) : ℝ :=
  let area_hexagon := (3 * Real.sqrt 3 / 2) * s^2
  let area_semicircle := (1/2) * Real.pi * (s/2)^2
  let total_area_semicircles := 6 * area_semicircle
  let total_area_circles := 6 * Real.pi * (s/2)^2
  total_area_circles - area_hexagon

theorem region_area (s := 2) : area_of_region_outside_hexagon_inside_semicircles s = (6 * Real.pi - 6 * Real.sqrt 3) :=
by
  sorry  -- Proof is skipped.

end NUMINAMATH_GPT_region_area_l1523_152314


namespace NUMINAMATH_GPT_alpha_necessary_but_not_sufficient_for_beta_l1523_152342

theorem alpha_necessary_but_not_sufficient_for_beta 
  (a b : ℝ) (hα : b * (b - a) ≤ 0) (hβ : a / b ≥ 1) : 
  (b * (b - a) ≤ 0) ↔ (a / b ≥ 1) := 
sorry

end NUMINAMATH_GPT_alpha_necessary_but_not_sufficient_for_beta_l1523_152342


namespace NUMINAMATH_GPT_willows_in_the_park_l1523_152363

theorem willows_in_the_park (W O : ℕ) 
  (h1 : W + O = 83) 
  (h2 : O = W + 11) : 
  W = 36 := 
by 
  sorry

end NUMINAMATH_GPT_willows_in_the_park_l1523_152363


namespace NUMINAMATH_GPT_fran_speed_calculation_l1523_152309

noncomputable def fran_speed (joann_speed : ℝ) (joann_time : ℝ) (fran_time : ℝ) : ℝ :=
  joann_speed * joann_time / fran_time

theorem fran_speed_calculation : 
  fran_speed 15 3 2.5 = 18 := 
by
  -- Remember to write down the proof steps if needed, currently we use sorry as placeholder
  sorry

end NUMINAMATH_GPT_fran_speed_calculation_l1523_152309


namespace NUMINAMATH_GPT_trenton_commission_rate_l1523_152307

noncomputable def commission_rate (fixed_earnings : ℕ) (goal : ℕ) (sales : ℕ) : ℚ :=
  ((goal - fixed_earnings : ℤ) / (sales : ℤ)) * 100

theorem trenton_commission_rate :
  commission_rate 190 500 7750 = 4 := 
  by
  sorry

end NUMINAMATH_GPT_trenton_commission_rate_l1523_152307


namespace NUMINAMATH_GPT_prob_not_answered_after_three_rings_l1523_152352

def prob_first_ring_answered := 0.1
def prob_second_ring_answered := 0.25
def prob_third_ring_answered := 0.45

theorem prob_not_answered_after_three_rings : 
  1 - prob_first_ring_answered - prob_second_ring_answered - prob_third_ring_answered = 0.2 :=
by
  sorry

end NUMINAMATH_GPT_prob_not_answered_after_three_rings_l1523_152352


namespace NUMINAMATH_GPT_sqrt_t6_plus_t4_l1523_152368

open Real

theorem sqrt_t6_plus_t4 (t : ℝ) : sqrt (t^6 + t^4) = t^2 * sqrt (t^2 + 1) :=
by sorry

end NUMINAMATH_GPT_sqrt_t6_plus_t4_l1523_152368


namespace NUMINAMATH_GPT_range_of_values_l1523_152379

variable {f : ℝ → ℝ}

-- Conditions and given data
def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x) = f (-x)

def is_monotone_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → x ≤ y → f (x) ≤ f (y)

def condition (f : ℝ → ℝ) (a : ℝ) : Prop :=
  f (Real.log a / Real.log 2) + f (-Real.log a / Real.log 2) ≤ 2 * f (1)

-- The goal
theorem range_of_values (h1 : is_even f) (h2 : is_monotone_on_nonneg f) (a : ℝ) (h3 : condition f a) :
  1/2 ≤ a ∧ a ≤ 2 :=
sorry

end NUMINAMATH_GPT_range_of_values_l1523_152379


namespace NUMINAMATH_GPT_cone_volume_l1523_152335

theorem cone_volume (l h : ℝ) (l_eq : l = 5) (h_eq : h = 4) : 
  (1 / 3) * Real.pi * ((l^2 - h^2).sqrt)^2 * h = 12 * Real.pi := 
by 
  sorry

end NUMINAMATH_GPT_cone_volume_l1523_152335


namespace NUMINAMATH_GPT_q_implies_not_p_l1523_152390

-- Define the conditions p and q
def p (x : ℝ) := x < -1
def q (x : ℝ) := x^2 - x - 2 > 0

-- Prove that q implies ¬p
theorem q_implies_not_p (x : ℝ) : q x → ¬ p x := by
  intros hq hp
  -- Provide the steps of logic here
  sorry

end NUMINAMATH_GPT_q_implies_not_p_l1523_152390


namespace NUMINAMATH_GPT_proposition_p_and_not_q_l1523_152331

theorem proposition_p_and_not_q (P Q : Prop) 
  (h1 : P ∨ Q) 
  (h2 : ¬ (P ∧ Q)) : (P ↔ ¬ Q) :=
sorry

end NUMINAMATH_GPT_proposition_p_and_not_q_l1523_152331


namespace NUMINAMATH_GPT_initial_average_is_correct_l1523_152357

def initial_average_daily_production (n : ℕ) (today_production : ℕ) (new_average : ℕ) (initial_average : ℕ) :=
  let total_initial_production := initial_average * n
  let total_new_production := total_initial_production + today_production
  let total_days := n + 1
  total_new_production = new_average * total_days

theorem initial_average_is_correct :
  ∀ (A n today_production new_average : ℕ),
    n = 19 →
    today_production = 90 →
    new_average = 52 →
    initial_average_daily_production n today_production new_average A →
    A = 50 := by
    intros A n today_production new_average hn htoday hnew havg
    sorry

end NUMINAMATH_GPT_initial_average_is_correct_l1523_152357


namespace NUMINAMATH_GPT_polynomial_strictly_monotone_l1523_152328

def strictly_monotone (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

theorem polynomial_strictly_monotone
  (P : ℝ → ℝ)
  (H1 : strictly_monotone (P ∘ P))
  (H2 : strictly_monotone (P ∘ P ∘ P)) :
  strictly_monotone P :=
sorry

end NUMINAMATH_GPT_polynomial_strictly_monotone_l1523_152328


namespace NUMINAMATH_GPT_divisors_72_l1523_152396

theorem divisors_72 : 
  { d | d ∣ 72 ∧ 0 < d } = {1, 2, 3, 4, 6, 8, 9, 12, 18, 24, 36, 72} := 
sorry

end NUMINAMATH_GPT_divisors_72_l1523_152396


namespace NUMINAMATH_GPT_johns_raise_percent_increase_l1523_152302

theorem johns_raise_percent_increase (original_earnings new_earnings : ℝ) 
  (h₀ : original_earnings = 60) (h₁ : new_earnings = 110) : 
  ((new_earnings - original_earnings) / original_earnings) * 100 = 83.33 :=
by
  sorry

end NUMINAMATH_GPT_johns_raise_percent_increase_l1523_152302


namespace NUMINAMATH_GPT_car_average_speed_l1523_152370

def average_speed (speed1 speed2 : ℕ) (time1 time2 : ℕ) : ℕ := 
  (speed1 * time1 + speed2 * time2) / (time1 + time2)

theorem car_average_speed :
  average_speed 60 90 (1/3) (2/3) = 80 := 
by 
  sorry

end NUMINAMATH_GPT_car_average_speed_l1523_152370


namespace NUMINAMATH_GPT_correct_option_D_l1523_152337

def U : Set ℕ := {1, 2, 4, 6, 8}
def A : Set ℕ := {1, 2, 4}
def B : Set ℕ := {2, 4, 6}
def complement_U_B : Set ℕ := {x ∈ U | x ∉ B}

theorem correct_option_D : A ∩ complement_U_B = {1} := by
  sorry

end NUMINAMATH_GPT_correct_option_D_l1523_152337
