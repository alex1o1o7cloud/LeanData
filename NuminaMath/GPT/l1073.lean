import Mathlib

namespace NUMINAMATH_GPT_range_of_a_l1073_107349

noncomputable def f (x : ℝ) := x * Real.log x

theorem range_of_a (a : ℝ) :
  (∀ x > 0, f x ≥ -x^2 + a*x - 6) → a ≤ 5 + Real.log 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1073_107349


namespace NUMINAMATH_GPT_calculate_expression_l1073_107358

theorem calculate_expression : 
  (10 - 9 * 8 + 7^2 / 2 - 3 * 4 + 6 - 5 = -48.5) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_calculate_expression_l1073_107358


namespace NUMINAMATH_GPT_base_length_of_isosceles_triangle_triangle_l1073_107389

section Geometry

variable {b m x : ℝ}

-- Define the conditions
def isosceles_triangle (b : ℝ) : Prop :=
∀ {A B C : ℝ}, A = b ∧ B = b -- representing an isosceles triangle with two equal sides

def segment_length (m : ℝ) : Prop :=
∀ {D E : ℝ}, D - E = m -- the segment length between points where bisectors intersect sides is m

-- The theorem we want to prove
theorem base_length_of_isosceles_triangle_triangle (h1 : isosceles_triangle b) (h2 : segment_length m) : x = b * m / (b - m) :=
sorry

end Geometry

end NUMINAMATH_GPT_base_length_of_isosceles_triangle_triangle_l1073_107389


namespace NUMINAMATH_GPT_scalene_triangle_angle_obtuse_l1073_107379

theorem scalene_triangle_angle_obtuse (a b c : ℝ) 
  (h_scalene : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_longest : a > b ∧ a > c)
  (h_obtuse_angle : a^2 > b^2 + c^2) : 
  ∃ A : ℝ, A = (Real.pi / 2) ∧ (b^2 + c^2 - a^2) / (2 * b * c) < 0 := 
sorry

end NUMINAMATH_GPT_scalene_triangle_angle_obtuse_l1073_107379


namespace NUMINAMATH_GPT_small_bottles_needed_l1073_107368

noncomputable def small_bottle_capacity := 40 -- in milliliters
noncomputable def large_bottle_capacity := 540 -- in milliliters
noncomputable def worst_case_small_bottle_capacity := 38 -- in milliliters

theorem small_bottles_needed :
  let n_bottles := Int.ceil (large_bottle_capacity / worst_case_small_bottle_capacity : ℚ)
  n_bottles = 15 :=
by
  sorry

end NUMINAMATH_GPT_small_bottles_needed_l1073_107368


namespace NUMINAMATH_GPT_cups_of_flour_already_put_in_correct_l1073_107384

-- Let F be the number of cups of flour Mary has already put in
def cups_of_flour_already_put_in (F : ℕ) : Prop :=
  let total_flour_needed := 12
  let cups_of_salt := 7
  let additional_flour_needed := cups_of_salt + 3
  F = total_flour_needed - additional_flour_needed

-- Theorem stating that F = 2
theorem cups_of_flour_already_put_in_correct (F : ℕ) : cups_of_flour_already_put_in F → F = 2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_cups_of_flour_already_put_in_correct_l1073_107384


namespace NUMINAMATH_GPT_smallest_natural_number_condition_l1073_107370

theorem smallest_natural_number_condition (N : ℕ) : 
  (∀ k : ℕ, (10^6 - 1) * k = (10^54 - 1) / 9 → k < N) →
  N = 111112 :=
by
  sorry

end NUMINAMATH_GPT_smallest_natural_number_condition_l1073_107370


namespace NUMINAMATH_GPT_problem_statement_l1073_107359

def S : Set Nat := {x | x ∈ Finset.range 13 \ Finset.range 1}

def n : Nat :=
  4^12 - 3 * 3^12 + 3 * 2^12

theorem problem_statement : n % 1000 = 181 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1073_107359


namespace NUMINAMATH_GPT_win_sector_area_l1073_107303

theorem win_sector_area (r : ℝ) (p : ℝ) (h_r : r = 12) (h_p : p = 1 / 3) :
  ∃ A : ℝ, A = 48 * π :=
by {
  sorry
}

end NUMINAMATH_GPT_win_sector_area_l1073_107303


namespace NUMINAMATH_GPT_largest_multiple_of_15_less_than_500_l1073_107314

theorem largest_multiple_of_15_less_than_500 : ∃ n : ℕ, (n > 0 ∧ 15 * n < 500) ∧ (∀ m : ℕ, m > n → 15 * m >= 500) ∧ 15 * n = 495 :=
by
  sorry

end NUMINAMATH_GPT_largest_multiple_of_15_less_than_500_l1073_107314


namespace NUMINAMATH_GPT_son_age_next_year_l1073_107353

-- Definitions based on the given conditions
def my_current_age : ℕ := 35
def son_current_age : ℕ := my_current_age / 5

-- Theorem statement to prove the answer
theorem son_age_next_year : son_current_age + 1 = 8 :=
by
  -- Skipping the proof with 'sorry'
  sorry

end NUMINAMATH_GPT_son_age_next_year_l1073_107353


namespace NUMINAMATH_GPT_original_price_of_trouser_l1073_107309

theorem original_price_of_trouser (sale_price : ℝ) (percent_decrease : ℝ) (original_price : ℝ) 
  (h1 : sale_price = 75) 
  (h2 : percent_decrease = 0.25) 
  (h3 : original_price - percent_decrease * original_price = sale_price) : 
  original_price = 100 :=
by
  sorry

end NUMINAMATH_GPT_original_price_of_trouser_l1073_107309


namespace NUMINAMATH_GPT_right_triangle_incenter_distance_l1073_107327

noncomputable def triangle_right_incenter_distance : ℝ :=
  let AB := 4 * Real.sqrt 2
  let BC := 6
  let AC := Real.sqrt (AB^2 + BC^2)
  let area := (1 / 2) * AB * BC
  let s := (AB + BC + AC) / 2
  let r := area / s
  r

theorem right_triangle_incenter_distance :
  let AB := 4 * Real.sqrt 2
  let BC := 6
  let AC := 2 * Real.sqrt 17
  let area := 12 * Real.sqrt 2
  let s := 2 * Real.sqrt 2 + 3 + Real.sqrt 17
  let BI := area / s
  BI = triangle_right_incenter_distance := sorry

end NUMINAMATH_GPT_right_triangle_incenter_distance_l1073_107327


namespace NUMINAMATH_GPT_percentage_of_apples_after_removal_l1073_107396

-- Declare the initial conditions as Lean definitions
def initial_apples : Nat := 12
def initial_oranges : Nat := 23
def removed_oranges : Nat := 15

-- Calculate the new totals
def new_oranges : Nat := initial_oranges - removed_oranges
def new_total_fruit : Nat := initial_apples + new_oranges

-- Define the expected percentage of apples as a real number
def expected_percentage_apples : Nat := 60

-- Prove that the percentage of apples after removing the specified number of oranges is 60%
theorem percentage_of_apples_after_removal :
  (initial_apples * 100 / new_total_fruit) = expected_percentage_apples := by
  sorry

end NUMINAMATH_GPT_percentage_of_apples_after_removal_l1073_107396


namespace NUMINAMATH_GPT_base_6_addition_l1073_107390

-- Definitions of base conversion and addition
def base_6_to_nat (n : ℕ) : ℕ :=
  n.div 100 * 36 + n.div 10 % 10 * 6 + n % 10

def nat_to_base_6 (n : ℕ) : ℕ :=
  let a := n.div 216
  let b := (n % 216).div 36
  let c := ((n % 216) % 36).div 6
  let d := n % 6
  a * 1000 + b * 100 + c * 10 + d

-- Conversion from base 6 to base 10 for the given numbers
def nat_256 := base_6_to_nat 256
def nat_130 := base_6_to_nat 130

-- The final theorem to prove
theorem base_6_addition : nat_to_base_6 (nat_256 + nat_130) = 1042 :=
by
  -- Proof omitted since it is not required
  sorry

end NUMINAMATH_GPT_base_6_addition_l1073_107390


namespace NUMINAMATH_GPT_original_price_l1073_107334

theorem original_price (a b x : ℝ) (h : (x - a) * 0.60 = b) : x = (5 / 3 * b) + a :=
  sorry

end NUMINAMATH_GPT_original_price_l1073_107334


namespace NUMINAMATH_GPT_cos_Z_value_l1073_107366

-- The conditions given in the problem
def sin_X := 4 / 5
def cos_Y := 3 / 5

-- The theorem we want to prove
theorem cos_Z_value (sin_X : ℝ) (cos_Y : ℝ) (hX : sin_X = 4/5) (hY : cos_Y = 3/5) : 
  ∃ cos_Z : ℝ, cos_Z = 7 / 25 :=
by
  -- Attach all conditions and solve
  sorry

end NUMINAMATH_GPT_cos_Z_value_l1073_107366


namespace NUMINAMATH_GPT_problem_solution_l1073_107350

def f (x m : ℝ) : ℝ :=
  3 * x ^ 2 + m * (m - 6) * x + 5

theorem problem_solution (m n : ℝ) :
  (f 1 m > 0) ∧ (∀ x : ℝ, -1 < x ∧ x < 4 → f x m < n) ↔ (m = 3 ∧ n = 17) :=
by sorry

end NUMINAMATH_GPT_problem_solution_l1073_107350


namespace NUMINAMATH_GPT_smallest_degree_measure_for_WYZ_l1073_107360

def angle_XYZ : ℝ := 130
def angle_XYW : ℝ := 100
def angle_WYZ : ℝ := angle_XYZ - angle_XYW

theorem smallest_degree_measure_for_WYZ : angle_WYZ = 30 :=
by
  sorry

end NUMINAMATH_GPT_smallest_degree_measure_for_WYZ_l1073_107360


namespace NUMINAMATH_GPT_one_third_greater_than_333_l1073_107340

theorem one_third_greater_than_333 :
  (1 : ℝ) / 3 > (333 : ℝ) / 1000 - 1 / 3000 :=
sorry

end NUMINAMATH_GPT_one_third_greater_than_333_l1073_107340


namespace NUMINAMATH_GPT_inequality_does_not_hold_l1073_107336

theorem inequality_does_not_hold (a b : ℝ) (h₁ : a < b) (h₂ : b < 0) :
  ¬ (1 / (a - 1) < 1 / b) :=
by
  sorry

end NUMINAMATH_GPT_inequality_does_not_hold_l1073_107336


namespace NUMINAMATH_GPT_grandparents_to_parents_ratio_l1073_107393

-- Definitions corresponding to the conditions
def wallet_cost : ℕ := 100
def betty_half_money : ℕ := wallet_cost / 2
def parents_contribution : ℕ := 15
def betty_needs_more : ℕ := 5
def grandparents_contribution : ℕ := 95 - (betty_half_money + parents_contribution)

-- The mathematical statement for the proof
theorem grandparents_to_parents_ratio :
  grandparents_contribution / parents_contribution = 2 := by
  sorry

end NUMINAMATH_GPT_grandparents_to_parents_ratio_l1073_107393


namespace NUMINAMATH_GPT_P_is_subtract_0_set_P_is_not_subtract_1_set_no_subtract_2_set_exists_all_subtract_1_sets_l1073_107382

def is_subtract_set (T : Set ℕ) (i : ℕ) := T ⊆ Set.univ ∧ T ≠ {1} ∧ (∀ {x y : ℕ}, x ∈ Set.univ → y ∈ Set.univ → x + y ∈ T → x * y - i ∈ T)

theorem P_is_subtract_0_set : is_subtract_set {1, 2} 0 := sorry

theorem P_is_not_subtract_1_set : ¬ is_subtract_set {1, 2} 1 := sorry

theorem no_subtract_2_set_exists : ¬∃ T : Set ℕ, is_subtract_set T 2 := sorry

theorem all_subtract_1_sets : ∀ T : Set ℕ, is_subtract_set T 1 ↔ T = {1, 3} ∨ T = {1, 3, 5} := sorry

end NUMINAMATH_GPT_P_is_subtract_0_set_P_is_not_subtract_1_set_no_subtract_2_set_exists_all_subtract_1_sets_l1073_107382


namespace NUMINAMATH_GPT_inappropriate_survey_method_l1073_107341

/-
Parameters:
- A: Using a sampling survey method to understand the water-saving awareness of middle school students in the city (appropriate).
- B: Investigating the capital city to understand the environmental pollution situation of the entire province (inappropriate due to lack of representativeness).
- C: Investigating the audience's evaluation of a movie by surveying those seated in odd-numbered seats (appropriate).
- D: Using a census method to understand the compliance rate of pilots' vision (appropriate).
-/

theorem inappropriate_survey_method (A B C D : Prop) 
  (hA : A = true)
  (hB : B = false)  -- This condition defines B as inappropriate
  (hC : C = true)
  (hD : D = true) : B = false :=
sorry

end NUMINAMATH_GPT_inappropriate_survey_method_l1073_107341


namespace NUMINAMATH_GPT_initial_pigs_l1073_107394

theorem initial_pigs (x : ℕ) (h : x + 86 = 150) : x = 64 :=
by
  sorry

end NUMINAMATH_GPT_initial_pigs_l1073_107394


namespace NUMINAMATH_GPT_part_I_part_II_l1073_107356

noncomputable def f (a x : ℝ) : ℝ := a * Real.exp x - x * Real.log x

theorem part_I (a : ℝ) :
  (∀ x > 0, 0 ≤ a * Real.exp x - (1 + Real.log x)) ↔ a ≥ 1 / Real.exp 1 :=
sorry

theorem part_II (a : ℝ) (h : a ≥ 2 / Real.exp 2) (x : ℝ) (hx : x > 0) :
  f a x > 0 :=
sorry

end NUMINAMATH_GPT_part_I_part_II_l1073_107356


namespace NUMINAMATH_GPT_tickets_spent_on_beanie_l1073_107377

variable (initial_tickets won_tickets tickets_left tickets_spent: ℕ)

theorem tickets_spent_on_beanie
  (h1 : initial_tickets = 49)
  (h2 : won_tickets = 6)
  (h3 : tickets_left = 30)
  (h4 : tickets_spent = initial_tickets + won_tickets - tickets_left) :
  tickets_spent = 25 :=
by
  sorry

end NUMINAMATH_GPT_tickets_spent_on_beanie_l1073_107377


namespace NUMINAMATH_GPT_commutative_otimes_l1073_107387

def otimes (a b : ℝ) : ℝ := a * b + a + b

theorem commutative_otimes (a b : ℝ) : otimes a b = otimes b a :=
by
  /- The proof will go here, but we omit it and use sorry. -/
  sorry

end NUMINAMATH_GPT_commutative_otimes_l1073_107387


namespace NUMINAMATH_GPT_increasing_sequence_range_l1073_107337

theorem increasing_sequence_range (a : ℝ) (f : ℝ → ℝ) (a_n : ℕ+ → ℝ) :
  (∀ n : ℕ+, a_n n = f n) →
  (∀ n m : ℕ+, n < m → a_n n < a_n m) →
  (∀ x : ℝ, f x = if  x ≤ 7 then (3 - a) * x - 3 else a ^ (x - 6) ) →
  2 < a ∧ a < 3 :=
by
  sorry

end NUMINAMATH_GPT_increasing_sequence_range_l1073_107337


namespace NUMINAMATH_GPT_num_valid_colorings_l1073_107346

namespace ColoringGrid

-- Definition of the grid and the constraint.
-- It's easier to represent with simply 9 nodes and adjacent constraints, however,
-- we will declare the conditions and result as discussed.

def Grid := Fin 3 × Fin 3
def Colors := Fin 2

-- Define adjacency relationship
def adjacent (a b : Grid) : Prop :=
  (a.1 = b.1 ∧ (a.2 = b.2 + 1 ∨ a.2 + 1 = b.2)) ∨
  (a.2 = b.2 ∧ (a.1 = b.1 + 1 ∨ a.1 + 1 = b.1))

-- Condition stating no two adjacent squares can share the same color
def valid_coloring (f : Grid → Colors) : Prop :=
  ∀ a b : Grid, adjacent a b → f a ≠ f b

-- The main theorem stating the number of valid colorings
theorem num_valid_colorings : ∃ (n : ℕ), n = 2 ∧ ∀ (f : Grid → Colors), valid_coloring f → n = 2 :=
by sorry

end ColoringGrid

end NUMINAMATH_GPT_num_valid_colorings_l1073_107346


namespace NUMINAMATH_GPT_chess_tournament_participants_and_days_l1073_107321

theorem chess_tournament_participants_and_days:
  ∃ n d : ℕ, 
    (n % 2 = 1) ∧
    (n * (n - 1) / 2 = 630) ∧
    (d = 34 / 2) ∧
    (n = 35) ∧
    (d = 17) :=
sorry

end NUMINAMATH_GPT_chess_tournament_participants_and_days_l1073_107321


namespace NUMINAMATH_GPT_intersection_l1073_107329

def setA : Set ℝ := { x : ℝ | x^2 - 2*x - 3 < 0 }
def setB : Set ℝ := { x : ℝ | x > 1 }

theorem intersection (x : ℝ) : x ∈ setA ∧ x ∈ setB ↔ 1 < x ∧ x < 3 := by
  sorry

end NUMINAMATH_GPT_intersection_l1073_107329


namespace NUMINAMATH_GPT_circus_juggling_l1073_107315

theorem circus_juggling (jugglers : ℕ) (balls_per_juggler : ℕ) (total_balls : ℕ)
  (h1 : jugglers = 5000)
  (h2 : balls_per_juggler = 12)
  (h3 : total_balls = jugglers * balls_per_juggler) :
  total_balls = 60000 :=
by
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_circus_juggling_l1073_107315


namespace NUMINAMATH_GPT_transformed_data_properties_l1073_107300

-- Definitions of the initial mean and variance
def initial_mean : ℝ := 2.8
def initial_variance : ℝ := 3.6

-- Definitions of transformation constants
def multiplier : ℝ := 2
def increment : ℝ := 60

-- New mean after transformation
def new_mean : ℝ := multiplier * initial_mean + increment

-- New variance after transformation
def new_variance : ℝ := (multiplier ^ 2) * initial_variance

-- Theorem statement
theorem transformed_data_properties :
  new_mean = 65.6 ∧ new_variance = 14.4 :=
by
  sorry

end NUMINAMATH_GPT_transformed_data_properties_l1073_107300


namespace NUMINAMATH_GPT_exists_x_for_every_n_l1073_107301

theorem exists_x_for_every_n (n : ℕ) (hn : 0 < n) : ∃ x : ℤ, 2^n ∣ (x^2 - 17) :=
sorry

end NUMINAMATH_GPT_exists_x_for_every_n_l1073_107301


namespace NUMINAMATH_GPT_find_b_in_geometric_sequence_l1073_107351

theorem find_b_in_geometric_sequence 
  (a b c : ℝ) 
  (q : ℝ) 
  (h1 : -1 * q^4 = -9) 
  (h2 : a = -1 * q) 
  (h3 : b = a * q) 
  (h4 : c = b * q) 
  (h5 : -9 = c * q) : 
  b = -3 :=
by
  sorry

end NUMINAMATH_GPT_find_b_in_geometric_sequence_l1073_107351


namespace NUMINAMATH_GPT_prime_b_plus_1_l1073_107325

def is_a_good (a b : ℕ) : Prop :=
  ∀ n : ℕ, n > 0 → a * n ≥ b → (Nat.choose (a * n) b - 1) % (a * n + 1) = 0

theorem prime_b_plus_1 (a b : ℕ) (h1 : is_a_good a b) (h2 : ¬ is_a_good a (b + 2)) : Nat.Prime (b + 1) :=
by
  sorry

end NUMINAMATH_GPT_prime_b_plus_1_l1073_107325


namespace NUMINAMATH_GPT_complex_power_difference_l1073_107332

theorem complex_power_difference (i : ℂ) (h : i^2 = -1) : (1 + i) ^ 18 - (1 - i) ^ 18 = 1024 * i :=
by
  sorry

end NUMINAMATH_GPT_complex_power_difference_l1073_107332


namespace NUMINAMATH_GPT_age_of_b_l1073_107322

theorem age_of_b (a b : ℕ) 
(h1 : a + 10 = 2 * (b - 10)) 
(h2 : a = b + 4) : 
b = 34 := 
sorry

end NUMINAMATH_GPT_age_of_b_l1073_107322


namespace NUMINAMATH_GPT_cat_food_sufficiency_l1073_107311

variable (L S : ℝ)

theorem cat_food_sufficiency (h : L + 4 * S = 14) : L + 3 * S >= 11 :=
sorry

end NUMINAMATH_GPT_cat_food_sufficiency_l1073_107311


namespace NUMINAMATH_GPT_investment_total_correct_l1073_107304

-- Define the initial investment, interest rate, and duration
def initial_investment : ℝ := 300
def monthly_interest_rate : ℝ := 0.10
def duration_in_months : ℝ := 2

-- Define the total amount after 2 months
noncomputable def total_after_two_months : ℝ := initial_investment * (1 + monthly_interest_rate) * (1 + monthly_interest_rate)

-- Define the correct answer
def correct_answer : ℝ := 363

-- The proof problem
theorem investment_total_correct :
  total_after_two_months = correct_answer :=
sorry

end NUMINAMATH_GPT_investment_total_correct_l1073_107304


namespace NUMINAMATH_GPT_revenue_ratio_l1073_107397

variable (R_d : ℝ) (R_n : ℝ) (R_j : ℝ)

theorem revenue_ratio
  (nov_cond : R_n = 2 / 5 * R_d)
  (jan_cond : R_j = 1 / 2 * R_n) :
  R_d = 10 / 3 * ((R_n + R_j) / 2) := by
  -- Proof steps go here
  sorry

end NUMINAMATH_GPT_revenue_ratio_l1073_107397


namespace NUMINAMATH_GPT_div_4800_by_125_expr_13_mul_74_add_27_mul_13_sub_13_l1073_107317

theorem div_4800_by_125 : 4800 / 125 = 38.4 :=
by
  sorry

theorem expr_13_mul_74_add_27_mul_13_sub_13 : 13 * 74 + 27 * 13 - 13 = 1300 :=
by
  sorry

end NUMINAMATH_GPT_div_4800_by_125_expr_13_mul_74_add_27_mul_13_sub_13_l1073_107317


namespace NUMINAMATH_GPT_height_at_15_inches_l1073_107348

-- Define the conditions
def parabolic_eq (a x : ℝ) : ℝ := a * x^2 + 24
noncomputable def a : ℝ := -2 / 75
def x : ℝ := 15
def expected_y : ℝ := 18

-- Lean 4 statement
theorem height_at_15_inches :
  parabolic_eq a x = expected_y :=
by
  sorry

end NUMINAMATH_GPT_height_at_15_inches_l1073_107348


namespace NUMINAMATH_GPT_quadratic_inequality_solutions_l1073_107378

theorem quadratic_inequality_solutions {k : ℝ} (h1 : 0 < k) (h2 : k < 16) :
  ∃ x : ℝ, x^2 - 8*x + k < 0 :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_solutions_l1073_107378


namespace NUMINAMATH_GPT_eel_count_l1073_107367

theorem eel_count 
  (x y z : ℕ)
  (h1 : y + z = 12)
  (h2 : x + z = 14)
  (h3 : x + y = 16) : 
  x + y + z = 21 := 
by 
  sorry

end NUMINAMATH_GPT_eel_count_l1073_107367


namespace NUMINAMATH_GPT_find_x_l1073_107361

theorem find_x (x y : ℕ) (h1 : y = 30) (h2 : x / y = 5 / 2) : x = 75 := by
  sorry

end NUMINAMATH_GPT_find_x_l1073_107361


namespace NUMINAMATH_GPT_cheetah_catches_deer_in_10_minutes_l1073_107323

noncomputable def deer_speed : ℝ := 50 -- miles per hour
noncomputable def cheetah_speed : ℝ := 60 -- miles per hour
noncomputable def time_difference : ℝ := 2 / 60 -- 2 minutes converted to hours
noncomputable def distance_deer : ℝ := deer_speed * time_difference
noncomputable def speed_difference : ℝ := cheetah_speed - deer_speed
noncomputable def catch_up_time : ℝ := distance_deer / speed_difference

theorem cheetah_catches_deer_in_10_minutes :
  catch_up_time * 60 = 10 :=
by
  sorry

end NUMINAMATH_GPT_cheetah_catches_deer_in_10_minutes_l1073_107323


namespace NUMINAMATH_GPT_find_number_l1073_107373

theorem find_number (N : ℝ) (h : (1 / 4) * (1 / 3) * (2 / 5) * N = 14) : 0.40 * N = 168 :=
sorry

end NUMINAMATH_GPT_find_number_l1073_107373


namespace NUMINAMATH_GPT_bacteria_initial_count_l1073_107344

theorem bacteria_initial_count (n : ℕ) :
  (∀ t : ℕ, t % 30 = 0 → n * 2^(t / 30) = 262144 → t = 240) → n = 1024 :=
by sorry

end NUMINAMATH_GPT_bacteria_initial_count_l1073_107344


namespace NUMINAMATH_GPT_lateral_surface_area_of_cone_l1073_107324

theorem lateral_surface_area_of_cone (diameter height : ℝ) (h_d : diameter = 2) (h_h : height = 2) :
  let radius := diameter / 2
  let slant_height := Real.sqrt (radius ^ 2 + height ^ 2)
  π * radius * slant_height = Real.sqrt 5 * π := 
  by
    sorry

end NUMINAMATH_GPT_lateral_surface_area_of_cone_l1073_107324


namespace NUMINAMATH_GPT_child_b_share_l1073_107302

def total_money : ℕ := 4320

def ratio_parts : List ℕ := [2, 3, 4, 5, 6]

def parts_sum (parts : List ℕ) : ℕ :=
  parts.foldl (· + ·) 0

def value_of_one_part (total : ℕ) (parts : ℕ) : ℕ :=
  total / parts

def b_share (value_per_part : ℕ) (b_parts : ℕ) : ℕ :=
  value_per_part * b_parts

theorem child_b_share :
  let total_money := 4320
  let ratio_parts := [2, 3, 4, 5, 6]
  let total_parts := parts_sum ratio_parts
  let one_part_value := value_of_one_part total_money total_parts
  let b_parts := 3
  b_share one_part_value b_parts = 648 := by
  let total_money := 4320
  let ratio_parts := [2, 3, 4, 5, 6]
  let total_parts := parts_sum ratio_parts
  let one_part_value := value_of_one_part total_money total_parts
  let b_parts := 3
  show b_share one_part_value b_parts = 648
  sorry

end NUMINAMATH_GPT_child_b_share_l1073_107302


namespace NUMINAMATH_GPT_exists_polynomial_f_divides_f_x2_sub_1_l1073_107372

open Polynomial

theorem exists_polynomial_f_divides_f_x2_sub_1 (n : ℕ) :
    ∃ f : Polynomial ℝ, degree f = n ∧ f ∣ (f.comp (X ^ 2 - 1)) :=
by {
  sorry
}

end NUMINAMATH_GPT_exists_polynomial_f_divides_f_x2_sub_1_l1073_107372


namespace NUMINAMATH_GPT_solve_problem1_solve_problem2_l1073_107326

noncomputable def problem1 (m n : ℝ) : Prop :=
  (m + n) ^ 2 - 10 * (m + n) + 25 = (m + n - 5) ^ 2

noncomputable def problem2 (x : ℝ) : Prop :=
  ((x ^ 2 - 6 * x + 8) * (x ^ 2 - 6 * x + 10) + 1) = (x - 3) ^ 4

-- Placeholder for proofs
theorem solve_problem1 (m n : ℝ) : problem1 m n :=
by
  sorry

theorem solve_problem2 (x : ℝ) : problem2 x :=
by
  sorry

end NUMINAMATH_GPT_solve_problem1_solve_problem2_l1073_107326


namespace NUMINAMATH_GPT_car_kilometers_per_gallon_l1073_107391

-- Define the given conditions as assumptions
variable (total_distance : ℝ) (total_gallons : ℝ)
-- Assume the given conditions
axiom h1 : total_distance = 180
axiom h2 : total_gallons = 4.5

-- The statement to be proven
theorem car_kilometers_per_gallon : (total_distance / total_gallons) = 40 :=
by
  -- Sorry is used to skip the proof
  sorry

end NUMINAMATH_GPT_car_kilometers_per_gallon_l1073_107391


namespace NUMINAMATH_GPT_principal_amount_l1073_107386

theorem principal_amount
(SI : ℝ) (R : ℝ) (T : ℝ) (P : ℝ)
(h₀ : SI = 800)
(h₁ : R = 0.08)
(h₂ : T = 1)
(h₃ : SI = P * R * T) : P = 10000 :=
by
  sorry

end NUMINAMATH_GPT_principal_amount_l1073_107386


namespace NUMINAMATH_GPT_ratio_expenditure_l1073_107369

variable (I : ℝ) -- Assume the income in the first year is I.

-- Conditions
def savings_first_year := 0.25 * I
def expenditure_first_year := 0.75 * I
def income_second_year := 1.25 * I
def savings_second_year := 2 * savings_first_year
def expenditure_second_year := income_second_year - savings_second_year
def total_expenditure_two_years := expenditure_first_year + expenditure_second_year

-- Statement to be proved
theorem ratio_expenditure 
  (savings_first_year : ℝ := 0.25 * I)
  (expenditure_first_year : ℝ := 0.75 * I)
  (income_second_year : ℝ := 1.25 * I)
  (savings_second_year : ℝ := 2 * savings_first_year)
  (expenditure_second_year : ℝ := income_second_year - savings_second_year)
  (total_expenditure_two_years : ℝ := expenditure_first_year + expenditure_second_year) :
  (total_expenditure_two_years / expenditure_first_year) = 2 := by
    sorry

end NUMINAMATH_GPT_ratio_expenditure_l1073_107369


namespace NUMINAMATH_GPT_factorized_sum_is_33_l1073_107381

theorem factorized_sum_is_33 (p q r : ℤ)
  (h1 : ∀ x : ℤ, x^2 + 21 * x + 110 = (x + p) * (x + q))
  (h2 : ∀ x : ℤ, x^2 - 23 * x + 132 = (x - q) * (x - r)) : 
  p + q + r = 33 := by
  sorry

end NUMINAMATH_GPT_factorized_sum_is_33_l1073_107381


namespace NUMINAMATH_GPT_problem_statement_l1073_107395

open Set

def U : Set ℝ := univ
def M : Set ℝ := {x | x < 1}
def N : Set ℝ := {x | -6 < x ∧ x < 1}

theorem problem_statement : M ∩ N = N := by
  ext x
  constructor
  · intro h
    exact h.2
  · intro h
    exact ⟨h.2, h⟩

end NUMINAMATH_GPT_problem_statement_l1073_107395


namespace NUMINAMATH_GPT_total_trip_time_l1073_107308

theorem total_trip_time (driving_time : ℕ) (stuck_time : ℕ) (total_time : ℕ) :
  (stuck_time = 2 * driving_time) → (driving_time = 5) → (total_time = driving_time + stuck_time) → total_time = 15 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_total_trip_time_l1073_107308


namespace NUMINAMATH_GPT_correct_calculation_l1073_107383

variable (a : ℝ)

theorem correct_calculation : (a^2)^3 = a^6 := 
by sorry

end NUMINAMATH_GPT_correct_calculation_l1073_107383


namespace NUMINAMATH_GPT_baskets_count_l1073_107307

theorem baskets_count (total_apples apples_per_basket : ℕ) (h1 : total_apples = 629) (h2 : apples_per_basket = 17) : (total_apples / apples_per_basket) = 37 :=
by
  sorry

end NUMINAMATH_GPT_baskets_count_l1073_107307


namespace NUMINAMATH_GPT_quadratic_inequality_always_positive_l1073_107357

theorem quadratic_inequality_always_positive (k : ℝ) : 
  (∀ x : ℝ, k * x^2 - k * x + 1 > 0) ↔ (0 ≤ k ∧ k < 4) :=
by sorry

end NUMINAMATH_GPT_quadratic_inequality_always_positive_l1073_107357


namespace NUMINAMATH_GPT_no_pairs_for_arithmetic_progression_l1073_107362

-- Define the problem in Lean
theorem no_pairs_for_arithmetic_progression :
  ¬ ∃ (a b : ℝ), (2 * a = 5 + b) ∧ (2 * b = a * (1 + b)) :=
sorry

end NUMINAMATH_GPT_no_pairs_for_arithmetic_progression_l1073_107362


namespace NUMINAMATH_GPT_numberOfSolutions_l1073_107312

noncomputable def numberOfRealPositiveSolutions(x : ℝ) : Prop := 
  (x^6 + 1) * (x^4 + x^2 + 1) = 6 * x^5

theorem numberOfSolutions : ∃! x : ℝ, numberOfRealPositiveSolutions x := 
by
  sorry

end NUMINAMATH_GPT_numberOfSolutions_l1073_107312


namespace NUMINAMATH_GPT_range_of_a_l1073_107385

theorem range_of_a (a : ℝ) : 
  (∀ (x : ℝ), |x + 3| - |x - 1| ≤ a ^ 2 - 3 * a) ↔ a ≤ -1 ∨ a ≥ 4 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1073_107385


namespace NUMINAMATH_GPT_cos_5theta_l1073_107320

theorem cos_5theta (θ : ℝ) (h : Real.cos θ = 1/3) : Real.cos (5 * θ) = 241/243 :=
by
  sorry

end NUMINAMATH_GPT_cos_5theta_l1073_107320


namespace NUMINAMATH_GPT_sequence_a_n_derived_conditions_derived_sequence_is_even_l1073_107352

-- Statement of the first problem
theorem sequence_a_n_derived_conditions (a : ℕ → ℝ) (b : ℕ → ℝ) (n : ℕ)
  (h1 : b 1 = a n)
  (h2 : ∀ k, 2 ≤ k ∧ k ≤ n → b k = a (k - 1) + a k - b (k - 1))
  (h3 : b 1 = 5 ∧ b 2 = -2 ∧ b 3 = 7 ∧ b 4 = 2):
  a 1 = 2 ∧ a 2 = 1 ∧ a 3 = 4 ∧ a 4 = 5 :=
sorry

-- Statement of the second problem
theorem derived_sequence_is_even (a : ℕ → ℝ) (b : ℕ → ℝ) (c : ℕ → ℝ) (n : ℕ)
  (h_even : n % 2 = 0)
  (h1 : b 1 = a n)
  (h2 : ∀ k, 2 ≤ k ∧ k ≤ n → b k = a (k - 1) + a k - b (k - 1))
  (h3 : c 1 = b n)
  (h4 : ∀ k, 2 ≤ k ∧ k ≤ n → c k = b (k - 1) + b k - c (k - 1)):
  ∀ i, 1 ≤ i ∧ i ≤ n → c i = a i :=
sorry

end NUMINAMATH_GPT_sequence_a_n_derived_conditions_derived_sequence_is_even_l1073_107352


namespace NUMINAMATH_GPT_part1_part2_l1073_107330

-- Part 1
noncomputable def f (x a : ℝ) := x * Real.log x - a * x^2 + a

theorem part1 (a : ℝ) : (∀ x : ℝ, 0 < x → f x a ≤ a) → a ≥ 1 / Real.exp 1 :=
by
  sorry

-- Part 2
theorem part2 (a : ℝ) (x₀ : ℝ) : 
  (∀ x : ℝ, f x₀ a < f x a → x = x₀) → a < 1 / 2 → 2 * a - 1 < f x₀ a ∧ f x₀ a < 0 :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l1073_107330


namespace NUMINAMATH_GPT_altitude_of_dolphin_l1073_107335

theorem altitude_of_dolphin (h_submarine : altitude_submarine = -50) (h_dolphin : distance_above_submarine = 10) : altitude_dolphin = -40 :=
by
  -- Altitude of the dolphin is the altitude of the submarine plus the distance above it
  have h_dolphin_altitude : altitude_dolphin = altitude_submarine + distance_above_submarine := sorry
  -- Substitute the values
  rw [h_submarine, h_dolphin] at h_dolphin_altitude
  -- Simplify the expression
  exact h_dolphin_altitude

end NUMINAMATH_GPT_altitude_of_dolphin_l1073_107335


namespace NUMINAMATH_GPT_find_p_l1073_107371

theorem find_p (a b p : ℝ) (h1: a ≠ 0) (h2: b ≠ 0) 
  (h3: a^2 - 4 * b = 0) 
  (h4: a + b = 5 * p) 
  (h5: a * b = 2 * p^3) : p = 3 := 
sorry

end NUMINAMATH_GPT_find_p_l1073_107371


namespace NUMINAMATH_GPT_outer_perimeter_l1073_107305

theorem outer_perimeter (F G H I J K L M N : ℕ) 
  (h_outer : F + G + H + I + J = 42) 
  (h_inner : K + L + M = 20) 
  (h_adjustment : N = 4) : 
  F + G + H + I + J - K - L - M + N = 26 := 
by 
  sorry

end NUMINAMATH_GPT_outer_perimeter_l1073_107305


namespace NUMINAMATH_GPT_symmetry_axis_g_l1073_107380

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 3)
noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * (x - Real.pi / 3) - Real.pi / 3)

theorem symmetry_axis_g :
  ∃ k : ℤ, (x = k * Real.pi / 2 + Real.pi / 4) := sorry

end NUMINAMATH_GPT_symmetry_axis_g_l1073_107380


namespace NUMINAMATH_GPT_find_sandwich_cost_l1073_107319

theorem find_sandwich_cost (S : ℝ) :
  3 * S + 2 * 4 = 26 → S = 6 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_sandwich_cost_l1073_107319


namespace NUMINAMATH_GPT_total_students_in_class_l1073_107306

def students_play_football : Nat := 26
def students_play_tennis : Nat := 20
def students_play_both : Nat := 17
def students_play_neither : Nat := 7

theorem total_students_in_class :
  (students_play_football + students_play_tennis - students_play_both + students_play_neither) = 36 :=
by
  sorry

end NUMINAMATH_GPT_total_students_in_class_l1073_107306


namespace NUMINAMATH_GPT_max_value_sqrt_sum_l1073_107333

open Real

noncomputable def max_sqrt_sum (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h_sum : x + y + z = 7) : ℝ :=
  sqrt (3 * x + 1) + sqrt (3 * y + 1) + sqrt (3 * z + 1)

theorem max_value_sqrt_sum (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h_sum : x + y + z = 7) :
  max_sqrt_sum x y z h1 h2 h3 h_sum ≤ 3 * sqrt 8 :=
sorry

end NUMINAMATH_GPT_max_value_sqrt_sum_l1073_107333


namespace NUMINAMATH_GPT_range_of_a_l1073_107399

variable (a : ℝ)
def is_second_quadrant (z : ℂ) : Prop := z.re < 0 ∧ z.im > 0
def z : ℂ := 4 - 2 * Complex.I

theorem range_of_a (ha : is_second_quadrant ((z + a * Complex.I) ^ 2)) : a > 6 := by
  sorry

end NUMINAMATH_GPT_range_of_a_l1073_107399


namespace NUMINAMATH_GPT_g_x_plus_three_l1073_107313

variable (x : ℝ)

def g (x : ℝ) : ℝ := x^2 - x

theorem g_x_plus_three : g (x + 3) = x^2 + 5 * x + 6 := by
  sorry

end NUMINAMATH_GPT_g_x_plus_three_l1073_107313


namespace NUMINAMATH_GPT_find_line_equation_through_two_points_find_circle_equation_tangent_to_x_axis_l1073_107375

open Real

-- Given conditions
def line_passes_through (x1 y1 x2 y2 : ℝ) (l : ℝ → ℝ → Prop) : Prop :=
  l x1 y1 ∧ l x2 y2

def circle_tangent_to_x_axis (center_x center_y : ℝ) (r : ℝ) (C : ℝ → ℝ → Prop) : Prop :=
  C center_x center_y ∧ center_y = r

-- We want to prove:
-- 1. The equation of line l is x - 2y = 0
theorem find_line_equation_through_two_points:
  ∃ l : ℝ → ℝ → Prop, line_passes_through 2 1 6 3 l ∧ (∀ x y, l x y ↔ x - 2 * y = 0) :=
  sorry

-- 2. The equation of circle C is (x - 2)^2 + (y - 1)^2 = 1
theorem find_circle_equation_tangent_to_x_axis:
  ∃ C : ℝ → ℝ → Prop, circle_tangent_to_x_axis 2 1 1 C ∧ (∀ x y, C x y ↔ (x - 2)^2 + (y - 1)^2 = 1) :=
  sorry

end NUMINAMATH_GPT_find_line_equation_through_two_points_find_circle_equation_tangent_to_x_axis_l1073_107375


namespace NUMINAMATH_GPT_problem_l1073_107331

variables {a b c : ℝ}

-- Given positive numbers a, b, c
axiom a_pos : 0 < a
axiom b_pos : 0 < b
axiom c_pos : 0 < c

-- Given conditions
axiom h1 : a * b + a + b = 3
axiom h2 : b * c + b + c = 3
axiom h3 : a * c + a + c = 3

-- Goal statement
theorem problem : (a + 1) * (b + 1) * (c + 1) = 8 := 
by 
  sorry

end NUMINAMATH_GPT_problem_l1073_107331


namespace NUMINAMATH_GPT_length_of_AB_l1073_107374

theorem length_of_AB :
  let ellipse := {p : ℝ × ℝ | p.1^2 / 4 + p.2^2 = 1}
  let focus := (Real.sqrt 3, 0)
  let line := {p : ℝ × ℝ | p.2 = p.1 - Real.sqrt 3}
  ∃ A B : ℝ × ℝ, A ∈ ellipse ∧ B ∈ ellipse ∧ A ∈ line ∧ B ∈ line ∧
  (dist A B = 8 / 5) :=
by
  sorry

end NUMINAMATH_GPT_length_of_AB_l1073_107374


namespace NUMINAMATH_GPT_factor_polynomial_l1073_107345

theorem factor_polynomial (x y : ℝ) : 2 * x^2 - 2 * y^2 = 2 * (x + y) * (x - y) := 
by sorry

end NUMINAMATH_GPT_factor_polynomial_l1073_107345


namespace NUMINAMATH_GPT_parabola_vertex_point_l1073_107316

theorem parabola_vertex_point (a b c : ℝ) : 
  (∀ x y : ℝ, y = a * x^2 + b * x + c → 
  ∃ k : ℝ, ∃ h : ℝ, y = a * (x - h)^2 + k ∧ h = 2 ∧ k = -1 ∧ 
  (∃ y₀ : ℝ, 7 = a * (0 - h)^2 + k) ∧ y₀ = 7) 
  → (a = 2 ∧ b = -8 ∧ c = 7) := by
  sorry

end NUMINAMATH_GPT_parabola_vertex_point_l1073_107316


namespace NUMINAMATH_GPT_largest_non_zero_ending_factor_decreasing_number_l1073_107365

theorem largest_non_zero_ending_factor_decreasing_number :
  ∃ n: ℕ, n = 180625 ∧ (n % 10 ≠ 0) ∧ (∃ m: ℕ, m < n ∧ (n % m = 0) ∧ (n / 10 ≤ m ∧ m * 10 > 0)) :=
by {
  sorry
}

end NUMINAMATH_GPT_largest_non_zero_ending_factor_decreasing_number_l1073_107365


namespace NUMINAMATH_GPT_compare_abc_l1073_107388

noncomputable def a : ℝ := 1 / (1 + Real.exp 2)
noncomputable def b : ℝ := 1 / Real.exp 1
noncomputable def c : ℝ := Real.log ((1 + Real.exp 2) / (Real.exp 2))

theorem compare_abc : b > c ∧ c > a := by
  sorry

end NUMINAMATH_GPT_compare_abc_l1073_107388


namespace NUMINAMATH_GPT_trig_expression_simplification_l1073_107339

theorem trig_expression_simplification (α : Real) :
  Real.cos (3/2 * Real.pi + 4 * α)
  + Real.sin (3 * Real.pi - 8 * α)
  - Real.sin (4 * Real.pi - 12 * α)
  = 4 * Real.cos (2 * α) * Real.cos (4 * α) * Real.sin (6 * α) :=
sorry

end NUMINAMATH_GPT_trig_expression_simplification_l1073_107339


namespace NUMINAMATH_GPT_max_possible_value_of_gcd_l1073_107392

theorem max_possible_value_of_gcd (n : ℕ) : gcd ((8^n - 1) / 7) ((8^(n+1) - 1) / 7) = 1 := by
  sorry

end NUMINAMATH_GPT_max_possible_value_of_gcd_l1073_107392


namespace NUMINAMATH_GPT_evaluate_expression_l1073_107364

theorem evaluate_expression : (723 * 723) - (722 * 724) = 1 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1073_107364


namespace NUMINAMATH_GPT_max_ways_to_ascend_descend_l1073_107347

theorem max_ways_to_ascend_descend :
  let east_paths := 2
  let west_paths := 1
  let south_paths := 3
  let north_paths := 4

  let descend_from_east := west_paths + south_paths + north_paths
  let descend_from_west := east_paths + south_paths + north_paths
  let descend_from_south := east_paths + west_paths + north_paths
  let descend_from_north := east_paths + west_paths + south_paths

  let ways_from_east := east_paths * descend_from_east
  let ways_from_west := west_paths * descend_from_west
  let ways_from_south := south_paths * descend_from_south
  let ways_from_north := north_paths * descend_from_north

  max ways_from_east (max ways_from_west (max ways_from_south ways_from_north)) = 24 := 
by
  -- Insert the proof here
  sorry

end NUMINAMATH_GPT_max_ways_to_ascend_descend_l1073_107347


namespace NUMINAMATH_GPT_B_work_days_l1073_107318

theorem B_work_days
  (A_work_rate : ℝ) (B_work_rate : ℝ) (A_days_worked : ℝ) (B_days_worked : ℝ)
  (total_work : ℝ) (remaining_work : ℝ) :
  A_work_rate = 1 / 15 →
  B_work_rate = total_work / 18 →
  A_days_worked = 5 →
  remaining_work = total_work - A_work_rate * A_days_worked →
  B_days_worked = 12 →
  remaining_work = B_work_rate * B_days_worked →
  total_work = 1 →
  B_days_worked = 12 →
  B_work_rate = total_work / 18 →
  B_days_alone = total_work / B_work_rate →
  B_days_alone = 18 := 
by
  intro hA_work_rate hB_work_rate hA_days_worked hremaining_work hB_days_worked hremaining_work_eq htotal_work hB_days_worked_again hsry_mul_inv hB_days_we_alone_eq
  sorry

end NUMINAMATH_GPT_B_work_days_l1073_107318


namespace NUMINAMATH_GPT_correctness_statement_l1073_107376

-- Given points A, B, C are on the specific parabola
variable (a c x1 x2 x3 y1 y2 y3 : ℝ)
variable (ha : a < 0) -- a < 0 since the parabola opens upwards
variable (hA : y1 = - (a / 4) * x1^2 + a * x1 + c)
variable (hB : y2 = a + c) -- B is the vertex
variable (hC : y3 = - (a / 4) * x3^2 + a * x3 + c)
variable (hOrder : y1 > y3 ∧ y3 ≥ y2)

theorem correctness_statement : abs (x1 - x2) > abs (x3 - x2) :=
sorry

end NUMINAMATH_GPT_correctness_statement_l1073_107376


namespace NUMINAMATH_GPT_problem1_problem2_l1073_107338

-- Define conditions for Problem 1
def problem1_cond (x : ℝ) : Prop :=
  x ≠ 0 ∧ 2 * x ≠ 1

-- Statement for Problem 1
theorem problem1 (x : ℝ) (h : problem1_cond x) :
  (2 / x = 3 / (2 * x - 1)) ↔ x = 2 := by
  sorry

-- Define conditions for Problem 2
def problem2_cond (x : ℝ) : Prop :=
  x ≠ 2 

-- Statement for Problem 2
theorem problem2 (x : ℝ) (h : problem2_cond x) :
  ((x - 3) / (x - 2) + 1 = 3 / (2 - x)) ↔ x = 1 := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1073_107338


namespace NUMINAMATH_GPT_monica_sees_121_individual_students_l1073_107354

def students_count : ℕ :=
  let class1 := 20
  let class2 := 25
  let class3 := 25
  let class4 := class1 / 2
  let class5 := 28
  let class6 := 28
  let total_spots := class1 + class2 + class3 + class4 + class5 + class6
  let overlap12 := 5
  let overlap45 := 3
  let overlap36 := 7
  total_spots - overlap12 - overlap45 - overlap36

theorem monica_sees_121_individual_students : students_count = 121 := by
  sorry

end NUMINAMATH_GPT_monica_sees_121_individual_students_l1073_107354


namespace NUMINAMATH_GPT_sequence_value_a1_l1073_107310

theorem sequence_value_a1 (a : ℕ → ℝ) 
  (h₁ : ∀ n, a (n + 1) = (1 / 2) * a n) 
  (h₂ : a 4 = 8) : a 1 = 64 :=
sorry

end NUMINAMATH_GPT_sequence_value_a1_l1073_107310


namespace NUMINAMATH_GPT_exponent_division_l1073_107355

theorem exponent_division (a : ℕ) (m n : ℕ) (h1 : 19 = a) (h2 : 11 = m) (h3 : 8 = n) : a^(m - n) = 6859 := by
  sorry

end NUMINAMATH_GPT_exponent_division_l1073_107355


namespace NUMINAMATH_GPT_basketball_games_count_l1073_107343

noncomputable def tokens_per_game : ℕ := 3
noncomputable def total_tokens : ℕ := 18
noncomputable def air_hockey_games : ℕ := 2
noncomputable def air_hockey_tokens := air_hockey_games * tokens_per_game
noncomputable def remaining_tokens := total_tokens - air_hockey_tokens

theorem basketball_games_count :
  (remaining_tokens / tokens_per_game) = 4 := by
  sorry

end NUMINAMATH_GPT_basketball_games_count_l1073_107343


namespace NUMINAMATH_GPT_rope_total_length_is_54m_l1073_107363

noncomputable def totalRopeLength : ℝ :=
  let horizontalDistance : ℝ := 16
  let heightAB : ℝ := 18
  let heightCD : ℝ := 30
  let ropeBC := Real.sqrt (horizontalDistance^2 + (heightCD - heightAB)^2)
  let ropeAC := Real.sqrt (horizontalDistance^2 + heightCD^2)
  ropeBC + ropeAC

theorem rope_total_length_is_54m : totalRopeLength = 54 := sorry

end NUMINAMATH_GPT_rope_total_length_is_54m_l1073_107363


namespace NUMINAMATH_GPT_distance_between_A_and_B_l1073_107328

variable (d : ℝ) -- Total distance between A and B

def car_speeds (vA vB t : ℝ) : Prop :=
vA = 80 ∧ vB = 100 ∧ t = 2

def total_covered_distance (vA vB t : ℝ) : ℝ :=
(vA + vB) * t

def percentage_distance (total_distance covered_distance : ℝ) : Prop :=
0.6 * total_distance = covered_distance

theorem distance_between_A_and_B (vA vB t : ℝ) (H1 : car_speeds vA vB t) 
  (H2 : percentage_distance d (total_covered_distance vA vB t)) : d = 600 := by
  sorry

end NUMINAMATH_GPT_distance_between_A_and_B_l1073_107328


namespace NUMINAMATH_GPT_ellipse_range_k_l1073_107342

theorem ellipse_range_k (k : ℝ) (h1 : 3 + k > 0) (h2 : 2 - k > 0) (h3 : k ≠ -1 / 2) :
  k ∈ Set.Ioo (-3 : ℝ) (-1 / 2) ∪ Set.Ioo (-1 / 2) (2 : ℝ) :=
sorry

end NUMINAMATH_GPT_ellipse_range_k_l1073_107342


namespace NUMINAMATH_GPT_ten_elements_sequence_no_infinite_sequence_l1073_107398

def is_valid_seq (a : ℕ → ℕ) : Prop :=
  ∀ n, (a (n + 1))^2 - 4 * (a n) * (a (n + 2)) ≥ 0

theorem ten_elements_sequence : 
  ∃ a : ℕ → ℕ, (a 9 + 1 = 10) ∧ is_valid_seq a :=
sorry

theorem no_infinite_sequence :
  ¬∃ a : ℕ → ℕ, is_valid_seq a ∧ ∀ n, a n ≥ 1 :=
sorry

end NUMINAMATH_GPT_ten_elements_sequence_no_infinite_sequence_l1073_107398
