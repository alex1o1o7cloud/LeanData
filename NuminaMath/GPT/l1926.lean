import Mathlib

namespace NUMINAMATH_GPT_find_a8_l1926_192676

variable (a : ℕ → ℤ)

def arithmetic_sequence : Prop :=
  ∀ n : ℕ, 2 * a (n + 1) = a n + a (n + 2)

theorem find_a8 (h1 : a 7 + a 9 = 16) (h2 : arithmetic_sequence a) : a 8 = 8 := by
  -- proof would go here
  sorry

end NUMINAMATH_GPT_find_a8_l1926_192676


namespace NUMINAMATH_GPT_different_purchasing_methods_l1926_192683

noncomputable def number_of_purchasing_methods (n_two_priced : ℕ) (n_one_priced : ℕ) (total_price : ℕ) : ℕ :=
  let combinations_two_price (k : ℕ) := Nat.choose n_two_priced k
  let combinations_one_price (k : ℕ) := Nat.choose n_one_priced k
  combinations_two_price 5 + (combinations_two_price 4 * combinations_one_price 2)

theorem different_purchasing_methods :
  number_of_purchasing_methods 8 3 10 = 266 :=
by
  sorry

end NUMINAMATH_GPT_different_purchasing_methods_l1926_192683


namespace NUMINAMATH_GPT_age_ratio_l1926_192615

/-- Given that Sandy's age after 6 years will be 30 years,
    and Molly's current age is 18 years, 
    prove that the current ratio of Sandy's age to Molly's age is 4:3. -/
theorem age_ratio (M S : ℕ) 
  (h1 : M = 18) 
  (h2 : S + 6 = 30) : 
  S / gcd S M = 4 ∧ M / gcd S M = 3 :=
by
  sorry

end NUMINAMATH_GPT_age_ratio_l1926_192615


namespace NUMINAMATH_GPT_determine_a_for_unique_solution_of_quadratic_l1926_192696

theorem determine_a_for_unique_solution_of_quadratic :
  {a : ℝ | ∃! x : ℝ, a * x^2 - 4 * x + 2 = 0} = {0, 2} :=
sorry

end NUMINAMATH_GPT_determine_a_for_unique_solution_of_quadratic_l1926_192696


namespace NUMINAMATH_GPT_age_of_fourth_child_l1926_192662

theorem age_of_fourth_child 
  (avg_age : ℕ) 
  (age1 age2 age3 : ℕ) 
  (age4 : ℕ)
  (h_avg : (age1 + age2 + age3 + age4) / 4 = avg_age) 
  (h1 : age1 = 6) 
  (h2 : age2 = 8) 
  (h3 : age3 = 11) 
  (h_avg_val : avg_age = 9) : 
  age4 = 11 := 
by 
  sorry

end NUMINAMATH_GPT_age_of_fourth_child_l1926_192662


namespace NUMINAMATH_GPT_angle_CAB_EQ_angle_EAD_l1926_192659

variable {A B C D E : Type}

-- Define the angles as variables for the pentagon ABCDE
variable (ABC ADE CEA BDA CAB EAD : ℝ)

-- Given conditions
axiom angle_ABC_EQ_angle_ADE : ABC = ADE
axiom angle_CEA_EQ_angle_BDA : CEA = BDA

-- Prove that angle CAB equals angle EAD
theorem angle_CAB_EQ_angle_EAD : CAB = EAD :=
by
  sorry

end NUMINAMATH_GPT_angle_CAB_EQ_angle_EAD_l1926_192659


namespace NUMINAMATH_GPT_evaluate_expression_l1926_192688

theorem evaluate_expression : 7899665 - 12 * 3 * 2 = 7899593 :=
by
  -- This proof is skipped.
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1926_192688


namespace NUMINAMATH_GPT_abc_sum_zero_l1926_192650

theorem abc_sum_zero
  (a b c : ℝ)
  (h1 : ∀ x: ℝ, (a * (c * x^2 + b * x + a)^2 + b * (c * x^2 + b * x + a) + c = x)) :
  (a + b + c = 0) :=
by
  sorry

end NUMINAMATH_GPT_abc_sum_zero_l1926_192650


namespace NUMINAMATH_GPT_simplify_fraction_l1926_192607

theorem simplify_fraction (a b c : ℕ) (h1 : 222 = 2 * 111) (h2 : 999 = 3 * 333) (h3 : 111 = 3 * 37) :
  (222 / 999 * 111) = 74 :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l1926_192607


namespace NUMINAMATH_GPT_train_speed_in_km_hr_l1926_192695

noncomputable def train_length : ℝ := 110
noncomputable def bridge_length : ℝ := 132
noncomputable def crossing_time : ℝ := 9.679225661947045
noncomputable def distance_covered : ℝ := train_length + bridge_length
noncomputable def speed_m_s : ℝ := distance_covered / crossing_time
noncomputable def speed_km_hr : ℝ := speed_m_s * 3.6

theorem train_speed_in_km_hr : speed_km_hr = 90.0216 := by
  sorry

end NUMINAMATH_GPT_train_speed_in_km_hr_l1926_192695


namespace NUMINAMATH_GPT_negation_of_existence_l1926_192614

variable (Triangle : Type) (has_circumcircle : Triangle → Prop)

theorem negation_of_existence :
  ¬ (∃ t : Triangle, ¬ has_circumcircle t) ↔ ∀ t : Triangle, has_circumcircle t :=
by sorry

end NUMINAMATH_GPT_negation_of_existence_l1926_192614


namespace NUMINAMATH_GPT_population_density_reduction_l1926_192639

theorem population_density_reduction (scale : ℕ) (real_world_population : ℕ) : 
  scale = 1000000 → real_world_population = 1000000000 → 
  real_world_population / (scale ^ 2) < 1 := 
by 
  intros scale_value rw_population_value
  have h1 : scale ^ 2 = 1000000000000 := by sorry
  have h2 : real_world_population / 1000000000000 = 1 / 1000 := by sorry
  sorry

end NUMINAMATH_GPT_population_density_reduction_l1926_192639


namespace NUMINAMATH_GPT_total_yearly_interest_l1926_192670

/-- Mathematical statement:
Given Nina's total inheritance of $12,000, with $5,000 invested at 6% interest and the remainder invested at 8% interest, the total yearly interest from both investments is $860.
-/
theorem total_yearly_interest (principal : ℕ) (principal_part : ℕ) (rate1 rate2 : ℚ) (interest_part1 interest_part2 : ℚ) (total_interest : ℚ) :
  principal = 12000 ∧ principal_part = 5000 ∧ rate1 = 0.06 ∧ rate2 = 0.08 ∧
  interest_part1 = (principal_part : ℚ) * rate1 ∧ interest_part2 = ((principal - principal_part) : ℚ) * rate2 →
  total_interest = interest_part1 + interest_part2 → 
  total_interest = 860 := by
  sorry

end NUMINAMATH_GPT_total_yearly_interest_l1926_192670


namespace NUMINAMATH_GPT_orange_balls_count_l1926_192656

theorem orange_balls_count (total_balls red_balls blue_balls yellow_balls green_balls pink_balls orange_balls : ℕ) 
(h_total : total_balls = 100)
(h_red : red_balls = 30)
(h_blue : blue_balls = 20)
(h_yellow : yellow_balls = 10)
(h_green : green_balls = 5)
(h_pink : pink_balls = 2 * green_balls)
(h_orange : orange_balls = 3 * pink_balls)
(h_sum : red_balls + blue_balls + yellow_balls + green_balls + pink_balls + orange_balls = total_balls) :
orange_balls = 30 :=
sorry

end NUMINAMATH_GPT_orange_balls_count_l1926_192656


namespace NUMINAMATH_GPT_initial_red_marbles_l1926_192605

theorem initial_red_marbles
    (r g : ℕ)
    (h1 : 3 * r = 5 * g)
    (h2 : 2 * (r - 15) = g + 18) :
    r = 34 := by
  sorry

end NUMINAMATH_GPT_initial_red_marbles_l1926_192605


namespace NUMINAMATH_GPT_unique_solutions_of_system_l1926_192630

def system_of_equations (x y : ℝ) : Prop :=
  (x - 1) * (x - 2) * (x - 3) = 0 ∧
  (|x - 1| + |y - 1|) * (|x - 2| + |y - 2|) * (|x - 3| + |y - 4|) = 0

theorem unique_solutions_of_system :
  ∀ (x y : ℝ), system_of_equations x y ↔ (x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 2) ∨ (x = 3 ∧ y = 4) :=
by sorry

end NUMINAMATH_GPT_unique_solutions_of_system_l1926_192630


namespace NUMINAMATH_GPT_probability_businessmen_wait_two_minutes_l1926_192673

theorem probability_businessmen_wait_two_minutes :
  let total_suitcases := 200
  let business_suitcases := 10
  let time_to_wait_seconds := 120
  let suitcases_in_120_seconds := time_to_wait_seconds / 2
  let prob := (Nat.choose 59 9) / (Nat.choose total_suitcases business_suitcases)
  suitcases_in_120_seconds = 60 ->
  prob = (Nat.choose 59 9) / (Nat.choose 200 10) :=
by 
  sorry

end NUMINAMATH_GPT_probability_businessmen_wait_two_minutes_l1926_192673


namespace NUMINAMATH_GPT_quadratic_has_real_roots_l1926_192681

theorem quadratic_has_real_roots (m : ℝ) : 
  (∃ x : ℝ, (m-2) * x^2 - 2 * x + 1 = 0) ↔ m ≤ 3 :=
by sorry

end NUMINAMATH_GPT_quadratic_has_real_roots_l1926_192681


namespace NUMINAMATH_GPT_emails_difference_l1926_192645

def morning_emails : ℕ := 6
def afternoon_emails : ℕ := 8

theorem emails_difference :
  afternoon_emails - morning_emails = 2 := 
by
  sorry

end NUMINAMATH_GPT_emails_difference_l1926_192645


namespace NUMINAMATH_GPT_square_side_length_range_l1926_192603

theorem square_side_length_range (a : ℝ) (h : a^2 = 30) : 5.4 < a ∧ a < 5.5 :=
sorry

end NUMINAMATH_GPT_square_side_length_range_l1926_192603


namespace NUMINAMATH_GPT_james_age_when_john_turned_35_l1926_192686

theorem james_age_when_john_turned_35 :
  ∀ (J : ℕ) (Tim : ℕ) (John : ℕ),
  (Tim = 5) →
  (Tim + 5 = 2 * John) →
  (Tim = 79) →
  (John = 35) →
  (J = John) →
  J = 35 :=
by
  intros J Tim John h1 h2 h3 h4 h5
  rw [h4] at h5
  exact h5

end NUMINAMATH_GPT_james_age_when_john_turned_35_l1926_192686


namespace NUMINAMATH_GPT_matrix_A_pow_50_l1926_192663

def A : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![1, 1],
  ![0, 1]
]

theorem matrix_A_pow_50 : A^50 = ![
  ![1, 50],
  ![0, 1]
] :=
sorry

end NUMINAMATH_GPT_matrix_A_pow_50_l1926_192663


namespace NUMINAMATH_GPT_max_value_of_xy_expression_l1926_192608

theorem max_value_of_xy_expression (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 4 * x + 3 * y < 60) : 
  xy * (60 - 4 * x - 3 * y) ≤ 2000 / 3 := 
sorry

end NUMINAMATH_GPT_max_value_of_xy_expression_l1926_192608


namespace NUMINAMATH_GPT_total_volume_cylinder_cone_sphere_l1926_192649

theorem total_volume_cylinder_cone_sphere (r h : ℝ) (π : ℝ)
  (hc : π * r^2 * h = 150 * π)
  (hv : ∀ (r h : ℝ) (π : ℝ), V_cone = 1/3 * π * r^2 * h)
  (hs : ∀ (r : ℝ) (π : ℝ), V_sphere = 4/3 * π * r^3) :
  V_total = 50 * π + (4/3 * π * (150^(2/3))) :=
by
  sorry

end NUMINAMATH_GPT_total_volume_cylinder_cone_sphere_l1926_192649


namespace NUMINAMATH_GPT_sin_minus_cos_value_l1926_192693

open Real

noncomputable def tan_alpha := sqrt 3
noncomputable def alpha_condition (α : ℝ) := π < α ∧ α < (3 / 2) * π

theorem sin_minus_cos_value (α : ℝ) (h1 : tan α = tan_alpha) (h2 : alpha_condition α) : 
  sin α - cos α = -((sqrt 3) - 1) / 2 := 
by 
  sorry

end NUMINAMATH_GPT_sin_minus_cos_value_l1926_192693


namespace NUMINAMATH_GPT_tobias_charges_for_mowing_l1926_192661

/-- Tobias is buying a new pair of shoes that costs $95.
He has been saving up his money each month for the past three months.
He gets a $5 allowance a month.
He mowed 4 lawns and shoveled 5 driveways.
He charges $7 to shovel a driveway.
After buying the shoes, he has $15 in change.
Prove that Tobias charges $15 to mow a lawn.
--/
theorem tobias_charges_for_mowing 
  (shoes_cost : ℕ)
  (monthly_allowance : ℕ)
  (months_saving : ℕ)
  (lawns_mowed : ℕ)
  (driveways_shoveled : ℕ)
  (charge_per_shovel : ℕ)
  (money_left : ℕ)
  (total_money_before_purchase : ℕ)
  (x : ℕ)
  (h1 : shoes_cost = 95)
  (h2 : monthly_allowance = 5)
  (h3 : months_saving = 3)
  (h4 : lawns_mowed = 4)
  (h5 : driveways_shoveled = 5)
  (h6 : charge_per_shovel = 7)
  (h7 : money_left = 15)
  (h8 : total_money_before_purchase = shoes_cost + money_left)
  (h9 : total_money_before_purchase = (months_saving * monthly_allowance) + (lawns_mowed * x) + (driveways_shoveled * charge_per_shovel)) :
  x = 15 := 
sorry

end NUMINAMATH_GPT_tobias_charges_for_mowing_l1926_192661


namespace NUMINAMATH_GPT_product_mod_m_l1926_192692

-- Define the constants
def a : ℕ := 2345
def b : ℕ := 1554
def m : ℕ := 700

-- Definitions derived from the conditions
def a_mod_m : ℕ := a % m
def b_mod_m : ℕ := b % m

-- The proof problem
theorem product_mod_m (a b m : ℕ) (h1 : a % m = 245) (h2 : b % m = 154) :
  (a * b) % m = 630 := by sorry

end NUMINAMATH_GPT_product_mod_m_l1926_192692


namespace NUMINAMATH_GPT_right_triangle_hypotenuse_length_l1926_192680

theorem right_triangle_hypotenuse_length 
    (AB AC x y : ℝ) 
    (P : AB = x) (Q : AC = y) 
    (ratio_AP_PB : AP / PB = 1 / 3) 
    (ratio_AQ_QC : AQ / QC = 2 / 1) 
    (BQ_length : BQ = 18) 
    (CP_length : CP = 24) : 
    BC = 24 := 
by 
  sorry

end NUMINAMATH_GPT_right_triangle_hypotenuse_length_l1926_192680


namespace NUMINAMATH_GPT_age_ratio_in_two_years_l1926_192668

-- Definitions based on conditions
def lennon_age_current : ℕ := 8
def ophelia_age_current : ℕ := 38
def lennon_age_in_two_years := lennon_age_current + 2
def ophelia_age_in_two_years := ophelia_age_current + 2

-- Statement to prove
theorem age_ratio_in_two_years : 
  (ophelia_age_in_two_years / gcd ophelia_age_in_two_years lennon_age_in_two_years) = 4 ∧
  (lennon_age_in_two_years / gcd ophelia_age_in_two_years lennon_age_in_two_years) = 1 := 
by 
  sorry

end NUMINAMATH_GPT_age_ratio_in_two_years_l1926_192668


namespace NUMINAMATH_GPT_expected_worth_flip_l1926_192685

/-- A biased coin lands on heads with probability 2/3 and on tails with probability 1/3.
Each heads flip gains $5, and each tails flip loses $9.
If three consecutive flips all result in tails, then an additional loss of $10 is applied.
Prove that the expected worth of a single coin flip is -1/27. -/
theorem expected_worth_flip :
  let P_heads := 2 / 3
  let P_tails := 1 / 3
  (P_heads * 5 + P_tails * -9) - (P_tails ^ 3 * 10) = -1 / 27 :=
by
  sorry

end NUMINAMATH_GPT_expected_worth_flip_l1926_192685


namespace NUMINAMATH_GPT_solve_for_a_l1926_192689

theorem solve_for_a (x a : ℤ) (h : x = 3) (heq : 2 * x - 10 = 4 * a) : a = -1 := by
  sorry

end NUMINAMATH_GPT_solve_for_a_l1926_192689


namespace NUMINAMATH_GPT_values_of_n_for_replaced_constant_l1926_192602

theorem values_of_n_for_replaced_constant (n : ℤ) (x : ℤ) :
  (∀ n : ℤ, 4 * n + x > 1 ∧ 4 * n + x < 60) → x = 8 → 
  (∀ n : ℤ, 4 * n + 8 > 1 ∧ 4 * n + 8 < 60) :=
by
  sorry

end NUMINAMATH_GPT_values_of_n_for_replaced_constant_l1926_192602


namespace NUMINAMATH_GPT_similarity_transformation_result_l1926_192652

-- Define the original coordinates of point A and the similarity ratio
def A : ℝ × ℝ := (2, 2)
def ratio : ℝ := 2

-- Define the similarity transformation that scales coordinates, optionally considering reflection
def similarity_transform (p : ℝ × ℝ) (r : ℝ) : ℝ × ℝ :=
  (r * p.1, r * p.2)

-- Use Lean to state the theorem based on the given conditions and expected answer
theorem similarity_transformation_result :
  similarity_transform A ratio = (4, 4) ∨ similarity_transform A (-ratio) = (-4, -4) :=
by
  sorry

end NUMINAMATH_GPT_similarity_transformation_result_l1926_192652


namespace NUMINAMATH_GPT_four_digit_sum_of_digits_divisible_by_101_l1926_192647

theorem four_digit_sum_of_digits_divisible_by_101 (a b c d : ℕ)
  (h1 : 1 ≤ a ∧ a ≤ 9)
  (h2 : 1 ≤ b ∧ b ≤ 9)
  (h3 : 1 ≤ c ∧ c ≤ 9)
  (h4 : 1 ≤ d ∧ d ≤ 9)
  (h_diff : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_div : (1001 * a + 110 * b + 110 * c + 1001 * d) % 101 = 0) :
  (a + d) % 101 = (b + c) % 101 :=
by
  sorry

end NUMINAMATH_GPT_four_digit_sum_of_digits_divisible_by_101_l1926_192647


namespace NUMINAMATH_GPT_rectangle_area_3650_l1926_192691

variables (L B : ℕ)

-- Conditions given in the problem
def condition1 : Prop := L - B = 23
def condition2 : Prop := 2 * (L + B) = 246

-- Prove that the area of the rectangle is 3650 m² given the conditions
theorem rectangle_area_3650 (h1 : condition1 L B) (h2 : condition2 L B) : L * B = 3650 := by
  sorry

end NUMINAMATH_GPT_rectangle_area_3650_l1926_192691


namespace NUMINAMATH_GPT_calculate_meals_l1926_192648

-- Given conditions
def meal_cost : ℕ := 7
def total_spent : ℕ := 21

-- The expected number of meals Olivia's dad paid for
def expected_meals : ℕ := 3

-- Proof statement
theorem calculate_meals : total_spent / meal_cost = expected_meals :=
by
  sorry
  -- Proof can be completed using arithmetic simplification.

end NUMINAMATH_GPT_calculate_meals_l1926_192648


namespace NUMINAMATH_GPT_joes_fast_food_cost_l1926_192600

noncomputable def cost_of_sandwich (n : ℕ) : ℝ := n * 4
noncomputable def cost_of_soda (m : ℕ) : ℝ := m * 1.50
noncomputable def total_cost (n m : ℕ) : ℝ :=
  if n >= 10 then cost_of_sandwich n - 5 + cost_of_soda m else cost_of_sandwich n + cost_of_soda m

theorem joes_fast_food_cost :
  total_cost 10 6 = 44 := by
  sorry

end NUMINAMATH_GPT_joes_fast_food_cost_l1926_192600


namespace NUMINAMATH_GPT_problem_final_value_l1926_192629

theorem problem_final_value (x y z : ℝ) (hz : z ≠ 0) 
  (h1 : 3 * x - 2 * y - 2 * z = 0) 
  (h2 : x - 4 * y + 8 * z = 0) :
  (3 * x^2 - 2 * x * y) / (y^2 + 4 * z^2) = 120 / 269 := 
by 
  sorry

end NUMINAMATH_GPT_problem_final_value_l1926_192629


namespace NUMINAMATH_GPT_harkamal_paid_amount_l1926_192619

variable (grapesQuantity : ℕ)
variable (grapesRate : ℕ)
variable (mangoesQuantity : ℕ)
variable (mangoesRate : ℕ)

theorem harkamal_paid_amount (h1 : grapesQuantity = 8) (h2 : grapesRate = 70) (h3 : mangoesQuantity = 9) (h4 : mangoesRate = 45) :
  (grapesQuantity * grapesRate + mangoesQuantity * mangoesRate) = 965 := by
  sorry

end NUMINAMATH_GPT_harkamal_paid_amount_l1926_192619


namespace NUMINAMATH_GPT_sum_of_first_twelve_terms_l1926_192699

section ArithmeticSequence

variables (a : ℕ → ℚ) (d : ℚ) (a₁ : ℚ)

-- General definition of the nth term in arithmetic progression
def arithmetic_term (n : ℕ) : ℚ := a₁ + (n - 1) * d

-- Given conditions in the problem
axiom fifth_term : arithmetic_term a₁ d 5 = 1
axiom seventeenth_term : arithmetic_term a₁ d 17 = 18

-- Define the sum of the first n terms in arithmetic sequence
def sum_arithmetic_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  n * (2 * a₁ + (n - 1) * d) / 2

-- Statement of the proof problem
theorem sum_of_first_twelve_terms : 
  sum_arithmetic_sequence a₁ d 12 = 37.5 := 
sorry

end ArithmeticSequence

end NUMINAMATH_GPT_sum_of_first_twelve_terms_l1926_192699


namespace NUMINAMATH_GPT_smallest_x_satisfying_abs_eq_l1926_192698

theorem smallest_x_satisfying_abs_eq (x : ℝ) 
  (h : |2 * x^2 + 3 * x - 1| = 33) : 
  x = (-3 - Real.sqrt 281) / 4 := 
sorry

end NUMINAMATH_GPT_smallest_x_satisfying_abs_eq_l1926_192698


namespace NUMINAMATH_GPT_no_integers_satisfy_equation_l1926_192641

theorem no_integers_satisfy_equation :
  ∀ (a b c : ℤ), a^2 + b^2 - 8 * c ≠ 6 := by
  sorry

end NUMINAMATH_GPT_no_integers_satisfy_equation_l1926_192641


namespace NUMINAMATH_GPT_TotalToysIsNinetyNine_l1926_192682

def BillHasToys : ℕ := 60
def HalfOfBillToys : ℕ := BillHasToys / 2
def AdditionalToys : ℕ := 9
def HashHasToys : ℕ := HalfOfBillToys + AdditionalToys
def TotalToys : ℕ := BillHasToys + HashHasToys

theorem TotalToysIsNinetyNine : TotalToys = 99 := by
  sorry

end NUMINAMATH_GPT_TotalToysIsNinetyNine_l1926_192682


namespace NUMINAMATH_GPT_sum_of_solutions_l1926_192644

theorem sum_of_solutions (x : ℝ) : 
  (∃ y z, x^2 + 2017 * x - 24 = 2017 ∧ y^2 + 2017 * y - 2041 = 0 ∧ z^2 + 2017 * z - 2041 = 0 ∧ y ≠ z) →
  y + z = -2017 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_solutions_l1926_192644


namespace NUMINAMATH_GPT_meaningful_expression_range_l1926_192658

theorem meaningful_expression_range (x : ℝ) : (∃ (y : ℝ), y = 5 / (x - 2)) ↔ x ≠ 2 := 
by
  sorry

end NUMINAMATH_GPT_meaningful_expression_range_l1926_192658


namespace NUMINAMATH_GPT_arithmetic_sequence_50th_term_l1926_192635

-- Definitions as per the conditions
def a_1 : ℤ := 48
def d : ℤ := -2
def n : ℕ := 50

-- Statement to prove the 50th term in the series
theorem arithmetic_sequence_50th_term : a_1 + (n - 1) * d = -50 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_50th_term_l1926_192635


namespace NUMINAMATH_GPT_coefficient_of_x2_in_expansion_of_x_minus_2_to_the_5_l1926_192610

theorem coefficient_of_x2_in_expansion_of_x_minus_2_to_the_5 :
  ∀ (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ),
  (x - 2) ^ 5 = a_0 + a_1*x + a_2*x^2 + a_3*x^3 + a_4*x^4 + a_5*x^5 → a_2 = -80 := by
  sorry

end NUMINAMATH_GPT_coefficient_of_x2_in_expansion_of_x_minus_2_to_the_5_l1926_192610


namespace NUMINAMATH_GPT_shadow_area_correct_l1926_192694

noncomputable def shadow_area (R : ℝ) : ℝ := 3 * Real.pi * R^2

theorem shadow_area_correct (R r d R' : ℝ)
  (h1 : r = (Real.sqrt 3) * R / 2)
  (h2 : d = (3 * R) / 2)
  (h3 : R' = ((3 * R * r) / d)) :
  shadow_area R = Real.pi * R' ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_shadow_area_correct_l1926_192694


namespace NUMINAMATH_GPT_prove_equal_values_l1926_192617

theorem prove_equal_values :
  (-2: ℝ)^3 = -(2: ℝ)^3 :=
by sorry

end NUMINAMATH_GPT_prove_equal_values_l1926_192617


namespace NUMINAMATH_GPT_triangle_area_is_correct_l1926_192687

noncomputable def triangle_area_inscribed_circle (r : ℝ) (θ1 θ2 θ3 : ℝ) : ℝ := 
  (1 / 2) * r^2 * (Real.sin θ1 + Real.sin θ2 + Real.sin θ3)

theorem triangle_area_is_correct :
  triangle_area_inscribed_circle (18 / Real.pi) (Real.pi / 3) (2 * Real.pi / 3) Real.pi =
  162 * Real.sqrt 3 / (Real.pi^2) :=
by sorry

end NUMINAMATH_GPT_triangle_area_is_correct_l1926_192687


namespace NUMINAMATH_GPT_prime_divides_binom_l1926_192643

-- We define that n is a prime number.
def is_prime (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

-- Lean statement for the problem
theorem prime_divides_binom {n k : ℕ} (h₁ : is_prime n) (h₂ : 0 < k) (h₃ : k < n) :
  n ∣ Nat.choose n k :=
sorry

end NUMINAMATH_GPT_prime_divides_binom_l1926_192643


namespace NUMINAMATH_GPT_minimum_people_to_save_cost_l1926_192678

-- Define the costs for the two event planners.
def cost_first_planner (x : ℕ) : ℕ := 120 + 18 * x
def cost_second_planner (x : ℕ) : ℕ := 250 + 15 * x

-- State the theorem to prove the minimum number of people required for the second event planner to be less expensive.
theorem minimum_people_to_save_cost : ∃ x : ℕ, cost_second_planner x < cost_first_planner x ∧ ∀ y : ℕ, y < x → cost_second_planner y ≥ cost_first_planner y :=
sorry

end NUMINAMATH_GPT_minimum_people_to_save_cost_l1926_192678


namespace NUMINAMATH_GPT_m_is_perfect_square_l1926_192613

theorem m_is_perfect_square (n : ℕ) (m : ℤ) (h1 : m = 2 + 2 * Int.sqrt (44 * n^2 + 1) ∧ Int.sqrt (44 * n^2 + 1) * Int.sqrt (44 * n^2 + 1) = 44 * n^2 + 1) :
  ∃ k : ℕ, m = k^2 :=
by
  sorry

end NUMINAMATH_GPT_m_is_perfect_square_l1926_192613


namespace NUMINAMATH_GPT_special_burger_cost_l1926_192651

/-
  Prices of individual items and meals:
  - Burger: $5
  - French Fries: $3
  - Soft Drink: $3
  - Kid’s Burger: $3
  - Kid’s French Fries: $2
  - Kid’s Juice Box: $2
  - Kids Meal: $5

  Mr. Parker purchases:
  - 2 special burger meals for adults
  - 2 special burger meals and 2 kids' meals for 4 children
  - Saving $10 by buying 6 meals instead of the individual items

  Goal: 
  - Prove that the cost of one special burger meal is $8.
-/

def price_burger : Nat := 5
def price_fries : Nat := 3
def price_drink : Nat := 3
def price_kid_burger : Nat := 3
def price_kid_fries : Nat := 2
def price_kid_juice : Nat := 2
def price_kids_meal : Nat := 5

def total_adults_cost : Nat :=
  2 * price_burger + 2 * price_fries + 2 * price_drink

def total_kids_cost : Nat :=
  2 * price_kid_burger + 2 * price_kid_fries + 2 * price_kid_juice

def total_individual_cost : Nat :=
  total_adults_cost + total_kids_cost

def total_meals_cost : Nat :=
  total_individual_cost - 10

def cost_kids_meals : Nat :=
  2 * price_kids_meal

def total_cost_4_meals : Nat :=
  total_meals_cost

def cost_special_burger_meal : Nat :=
  (total_cost_4_meals - cost_kids_meals) / 2

theorem special_burger_cost : cost_special_burger_meal = 8 := by
  sorry

end NUMINAMATH_GPT_special_burger_cost_l1926_192651


namespace NUMINAMATH_GPT_yuna_candy_days_l1926_192601

theorem yuna_candy_days (total_candies : ℕ) (daily_candies_week : ℕ) (days_week : ℕ) (remaining_candies : ℕ) (daily_candies_future : ℕ) :
  total_candies = 60 →
  daily_candies_week = 6 →
  days_week = 7 →
  remaining_candies = total_candies - (daily_candies_week * days_week) →
  daily_candies_future = 3 →
  remaining_candies / daily_candies_future = 6 :=
by
  intros h_total h_daily_week h_days_week h_remaining h_daily_future
  sorry

end NUMINAMATH_GPT_yuna_candy_days_l1926_192601


namespace NUMINAMATH_GPT_range_of_m_l1926_192604

theorem range_of_m (f g : ℝ → ℝ) (h1 : ∃ m : ℝ, ∀ x : ℝ, f x = m * (x - m) * (x + m + 3))
  (h2 : ∀ x : ℝ, g x = 2 ^ x - 4)
  (h3 : ∀ x : ℝ, f x < 0 ∨ g x < 0) :
  ∃ m : ℝ, -5 < m ∧ m < 0 :=
sorry

end NUMINAMATH_GPT_range_of_m_l1926_192604


namespace NUMINAMATH_GPT_distance_CD_l1926_192677

theorem distance_CD (d_north: ℝ) (d_east: ℝ) (d_south: ℝ) (d_west: ℝ) (distance_CD: ℝ) :
  d_north = 30 ∧ d_east = 80 ∧ d_south = 20 ∧ d_west = 30 → distance_CD = 50 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_distance_CD_l1926_192677


namespace NUMINAMATH_GPT_determine_digits_from_expression_l1926_192636

theorem determine_digits_from_expression (a b c x y z S : ℕ) 
  (hx : x = 100) (hy : y = 10) (hz : z = 1)
  (hS : S = a * x + b * y + c * z) :
  S = 100 * a + 10 * b + c :=
by
  -- Variables
  -- a, b, c : ℕ -- digits to find
  -- x, y, z : ℕ -- chosen numbers
  -- S : ℕ -- the given sum

  -- Assumptions
  -- hx : x = 100
  -- hy : y = 10
  -- hz : z = 1
  -- hS : S = a * x + b * y + c * z
  sorry

end NUMINAMATH_GPT_determine_digits_from_expression_l1926_192636


namespace NUMINAMATH_GPT_sqrt_27_eq_3_sqrt_3_l1926_192664

theorem sqrt_27_eq_3_sqrt_3 : Real.sqrt 27 = 3 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_27_eq_3_sqrt_3_l1926_192664


namespace NUMINAMATH_GPT_lowest_temperature_in_january_2023_l1926_192627

theorem lowest_temperature_in_january_2023 
  (T_Beijing T_Shanghai T_Shenzhen T_Jilin : ℝ)
  (h_Beijing : T_Beijing = -5)
  (h_Shanghai : T_Shanghai = 6)
  (h_Shenzhen : T_Shenzhen = 19)
  (h_Jilin : T_Jilin = -22) :
  T_Jilin < T_Beijing ∧ T_Jilin < T_Shanghai ∧ T_Jilin < T_Shenzhen :=
by
  sorry

end NUMINAMATH_GPT_lowest_temperature_in_january_2023_l1926_192627


namespace NUMINAMATH_GPT_find_value_of_x_l1926_192671
-- Import the broader Mathlib to bring in the entirety of the necessary library

-- Definitions for the conditions
variables {x y z : ℝ}

-- Assume the given conditions
axiom h1 : x = y
axiom h2 : y = 2 * z
axiom h3 : x * y * z = 256

-- Statement to prove
theorem find_value_of_x : x = 8 :=
by {
  -- Proof goes here
  sorry
}

end NUMINAMATH_GPT_find_value_of_x_l1926_192671


namespace NUMINAMATH_GPT_segment_length_abs_eq_cubrt_27_five_l1926_192665

theorem segment_length_abs_eq_cubrt_27_five : 
  (∀ x : ℝ, |x - (3 : ℝ)| = 5) → (8 - (-2) = 10) :=
by 
  intros;
  sorry

end NUMINAMATH_GPT_segment_length_abs_eq_cubrt_27_five_l1926_192665


namespace NUMINAMATH_GPT_part1_part2_l1926_192655

section
variable (x a : ℝ)

def p (a x : ℝ) : Prop :=
  x^2 - 4*a*x + 3*a^2 < 0 ∧ a > 0

def q (x : ℝ) : Prop :=
  (x - 3) / (x - 2) ≤ 0

theorem part1 (h1 : p 1 x ∧ q x) : 2 < x ∧ x < 3 := by
  sorry

theorem part2 (h2 : ∀ x, ¬p a x → ¬q x) : 1 < a ∧ a ≤ 2 := by
  sorry

end

end NUMINAMATH_GPT_part1_part2_l1926_192655


namespace NUMINAMATH_GPT_solid_circles_count_2006_l1926_192620

def series_of_circles (n : ℕ) : List Char :=
  if n ≤ 0 then []
  else if n % 5 == 0 then '●' :: series_of_circles (n - 1)
  else '○' :: series_of_circles (n - 1)

def count_solid_circles (l : List Char) : ℕ :=
  l.count '●'

theorem solid_circles_count_2006 : count_solid_circles (series_of_circles 2006) = 61 := 
by
  sorry

end NUMINAMATH_GPT_solid_circles_count_2006_l1926_192620


namespace NUMINAMATH_GPT_pow_equation_sum_l1926_192625

theorem pow_equation_sum (x y : ℕ) (hx : 2 ^ 11 * 6 ^ 5 = 4 ^ x * 3 ^ y) : x + y = 13 :=
  sorry

end NUMINAMATH_GPT_pow_equation_sum_l1926_192625


namespace NUMINAMATH_GPT_length_of_first_train_l1926_192666

theorem length_of_first_train
    (speed_first_train_kmph : ℝ) 
    (speed_second_train_kmph : ℝ) 
    (time_to_cross_seconds : ℝ) 
    (length_second_train_meters : ℝ)
    (H1 : speed_first_train_kmph = 120)
    (H2 : speed_second_train_kmph = 80)
    (H3 : time_to_cross_seconds = 9)
    (H4 : length_second_train_meters = 300.04) : 
    ∃ (length_first_train : ℝ), length_first_train = 200 :=
by 
    let relative_speed_m_per_s := (speed_first_train_kmph + speed_second_train_kmph) * 1000 / 3600
    let combined_length := relative_speed_m_per_s * time_to_cross_seconds
    let length_first_train := combined_length - length_second_train_meters
    use length_first_train
    sorry

end NUMINAMATH_GPT_length_of_first_train_l1926_192666


namespace NUMINAMATH_GPT_cindy_gave_25_pens_l1926_192684

theorem cindy_gave_25_pens (initial_pens mike_gave pens_given_sharon final_pens : ℕ) (h1 : initial_pens = 5) (h2 : mike_gave = 20) (h3 : pens_given_sharon = 19) (h4 : final_pens = 31) :
  final_pens = initial_pens + mike_gave - pens_given_sharon + 25 :=
by 
  -- Insert the proof here later
  sorry

end NUMINAMATH_GPT_cindy_gave_25_pens_l1926_192684


namespace NUMINAMATH_GPT_quadratic_trinomial_prime_l1926_192606

theorem quadratic_trinomial_prime (p x : ℤ) (hp : p > 1) (hx : 0 ≤ x ∧ x < p)
  (h_prime : Prime (x^2 - x + p)) : x = 0 ∨ x = 1 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_trinomial_prime_l1926_192606


namespace NUMINAMATH_GPT_evaluate_polynomial_at_neg2_l1926_192640

theorem evaluate_polynomial_at_neg2 : 2 * (-2)^4 + 3 * (-2)^3 + 5 * (-2)^2 + (-2) + 4 = 30 :=
by 
  sorry

end NUMINAMATH_GPT_evaluate_polynomial_at_neg2_l1926_192640


namespace NUMINAMATH_GPT_world_book_day_l1926_192672

theorem world_book_day
  (x y : ℕ)
  (h1 : x + y = 22)
  (h2 : x = 2 * y + 1) :
  x = 15 ∧ y = 7 :=
by {
  -- The proof is omitted as per the instructions
  sorry
}

end NUMINAMATH_GPT_world_book_day_l1926_192672


namespace NUMINAMATH_GPT_glucose_in_mixed_solution_l1926_192633

def concentration1 := 20 / 100  -- concentration of first solution in grams per cubic centimeter
def concentration2 := 30 / 100  -- concentration of second solution in grams per cubic centimeter
def volume1 := 80               -- volume of first solution in cubic centimeters
def volume2 := 50               -- volume of second solution in cubic centimeters

theorem glucose_in_mixed_solution :
  (concentration1 * volume1) + (concentration2 * volume2) = 31 := by
  sorry

end NUMINAMATH_GPT_glucose_in_mixed_solution_l1926_192633


namespace NUMINAMATH_GPT_tuesday_more_than_monday_l1926_192624

variable (M T W Th x : ℕ)

-- Conditions
def monday_dinners : M = 40 := by sorry
def tuesday_dinners : T = M + x := by sorry
def wednesday_dinners : W = T / 2 := by sorry
def thursday_dinners : Th = W + 3 := by sorry
def total_dinners : M + T + W + Th = 203 := by sorry

-- Proof problem: How many more dinners were sold on Tuesday than on Monday?
theorem tuesday_more_than_monday : x = 32 :=
by
  sorry

end NUMINAMATH_GPT_tuesday_more_than_monday_l1926_192624


namespace NUMINAMATH_GPT_tangent_line_at_0_l1926_192621

noncomputable def f (x : ℝ) : ℝ := Real.exp x + x^2 - x + Real.sin x

theorem tangent_line_at_0 :
  ∃ (m b : ℝ), ∀ (x : ℝ), f 0 = 1 ∧ (f' : ℝ → ℝ) 0 = 1 ∧ (f' x = Real.exp x + 2 * x - 1 + Real.cos x) ∧ 
  (m = 1) ∧ (b = (m * 0 + 1)) ∧ (∀ x : ℝ, y = m * x + b) :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_at_0_l1926_192621


namespace NUMINAMATH_GPT_order_of_nums_l1926_192618

variable (a b : ℝ)

theorem order_of_nums (h1 : a + b > 0) (h2 : b < 0) : a > -b ∧ -b > b ∧ b > -a := 
sorry

end NUMINAMATH_GPT_order_of_nums_l1926_192618


namespace NUMINAMATH_GPT_perfect_square_of_sides_of_triangle_l1926_192690

theorem perfect_square_of_sides_of_triangle 
  (a b c : ℤ) 
  (h1: a > 0 ∧ b > 0 ∧ c > 0)
  (h2: a + b > c ∧ b + c > a ∧ c + a > b)
  (gcd_abc: Int.gcd (Int.gcd a b) c = 1)
  (h3: (a^2 + b^2 - c^2) % (a + b - c) = 0)
  (h4: (b^2 + c^2 - a^2) % (b + c - a) = 0)
  (h5: (c^2 + a^2 - b^2) % (c + a - b) = 0) : 
  ∃ n : ℤ, n^2 = (a + b - c) * (b + c - a) * (c + a - b) ∨ 
  ∃ m : ℤ, m^2 = 2 * (a + b - c) * (b + c - a) * (c + a - b) := 
sorry

end NUMINAMATH_GPT_perfect_square_of_sides_of_triangle_l1926_192690


namespace NUMINAMATH_GPT_tail_length_l1926_192660

variable (Length_body Length_tail Length_head : ℝ)

-- Conditions
def tail_half_body (Length_tail Length_body : ℝ) := Length_tail = 1/2 * Length_body
def head_sixth_body (Length_head Length_body : ℝ) := Length_head = 1/6 * Length_body
def overall_length (Length_head Length_body Length_tail : ℝ) := Length_head + Length_body + Length_tail = 30

-- Theorem statement
theorem tail_length (h1 : tail_half_body Length_tail Length_body) 
                  (h2 : head_sixth_body Length_head Length_body) 
                  (h3 : overall_length Length_head Length_body Length_tail) : 
                  Length_tail = 6 := by
  sorry

end NUMINAMATH_GPT_tail_length_l1926_192660


namespace NUMINAMATH_GPT_window_treatments_cost_l1926_192679

-- Define the costs and the number of windows
def cost_sheers : ℝ := 40.00
def cost_drapes : ℝ := 60.00
def number_of_windows : ℕ := 3

-- Define the total cost calculation
def total_cost := (cost_sheers + cost_drapes) * number_of_windows

-- State the theorem that needs to be proved
theorem window_treatments_cost : total_cost = 300.00 :=
by
  sorry

end NUMINAMATH_GPT_window_treatments_cost_l1926_192679


namespace NUMINAMATH_GPT_lcm_852_1491_l1926_192616

theorem lcm_852_1491 : Nat.lcm 852 1491 = 5961 := by
  sorry

end NUMINAMATH_GPT_lcm_852_1491_l1926_192616


namespace NUMINAMATH_GPT_inequaliy_pos_real_abc_l1926_192634

theorem inequaliy_pos_real_abc (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_abc : a * b * c = 1) : 
  (a / (a * b + 1)) + (b / (b * c + 1)) + (c / (c * a + 1)) ≥ (3 / 2) := 
by
  sorry

end NUMINAMATH_GPT_inequaliy_pos_real_abc_l1926_192634


namespace NUMINAMATH_GPT_rectangle_quadrilateral_inequality_l1926_192675

theorem rectangle_quadrilateral_inequality 
  (a b c d : ℝ)
  (h_a : 0 < a) (h_a_bound : a < 3)
  (h_b : 0 < b) (h_b_bound : b < 4)
  (h_c : 0 < c) (h_c_bound : c < 3)
  (h_d : 0 < d) (h_d_bound : d < 4) :
  25 ≤ ((3 - a)^2 + a^2 + (4 - b)^2 + b^2 + (3 - c)^2 + c^2 + (4 - d)^2 + d^2) ∧
  ((3 - a)^2 + a^2 + (4 - b)^2 + b^2 + (3 - c)^2 + c^2 + (4 - d)^2 + d^2) < 50 :=
by 
  sorry

end NUMINAMATH_GPT_rectangle_quadrilateral_inequality_l1926_192675


namespace NUMINAMATH_GPT_unique_two_digit_solution_l1926_192638

theorem unique_two_digit_solution : ∃! (t : ℕ), 10 ≤ t ∧ t < 100 ∧ 13 * t % 100 = 52 := sorry

end NUMINAMATH_GPT_unique_two_digit_solution_l1926_192638


namespace NUMINAMATH_GPT_B_pow_2048_l1926_192697

open Real Matrix

noncomputable def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![cos (π / 4), 0, -sin (π / 4)],
    ![0, 1, 0],
    ![sin (π / 4), 0, cos (π / 4)]]

theorem B_pow_2048 :
  B ^ 2048 = (1 : Matrix (Fin 3) (Fin 3) ℝ) :=
by
  sorry

end NUMINAMATH_GPT_B_pow_2048_l1926_192697


namespace NUMINAMATH_GPT_calculate_interest_rate_l1926_192622

variables (A : ℝ) (R : ℝ)

-- Conditions as definitions in Lean 4
def compound_interest_condition (A : ℝ) (R : ℝ) : Prop :=
  (A * (1 + R)^20 = 4 * A)

-- Theorem statement
theorem calculate_interest_rate (A : ℝ) (R : ℝ) (h : compound_interest_condition A R) : 
  R = (4)^(1/20) - 1 := 
sorry

end NUMINAMATH_GPT_calculate_interest_rate_l1926_192622


namespace NUMINAMATH_GPT_train_pass_time_approx_l1926_192646

noncomputable def time_to_pass_platform
  (L_t L_p : ℝ)
  (V_t : ℝ) : ℝ :=
  (L_t + L_p) / (V_t * (1000 / 3600))

theorem train_pass_time_approx
  (L_t L_p V_t : ℝ)
  (hL_t : L_t = 720)
  (hL_p : L_p = 360)
  (hV_t : V_t = 75) :
  abs (time_to_pass_platform L_t L_p V_t - 51.85) < 0.01 := 
by
  rw [hL_t, hL_p, hV_t]
  sorry

end NUMINAMATH_GPT_train_pass_time_approx_l1926_192646


namespace NUMINAMATH_GPT_binomial_product_l1926_192669

open Nat

theorem binomial_product : (Nat.choose 10 3) * (Nat.choose 8 3) = 6720 := by
  sorry

end NUMINAMATH_GPT_binomial_product_l1926_192669


namespace NUMINAMATH_GPT_calculate_final_number_l1926_192611

theorem calculate_final_number (initial increment times : ℕ) (h₀ : initial = 540) (h₁ : increment = 10) (h₂ : times = 6) : initial + increment * times = 600 :=
by
  sorry

end NUMINAMATH_GPT_calculate_final_number_l1926_192611


namespace NUMINAMATH_GPT_total_games_played_l1926_192628

theorem total_games_played (jerry_wins dave_wins ken_wins : ℕ)
  (h1 : jerry_wins = 7)
  (h2 : dave_wins = jerry_wins + 3)
  (h3 : ken_wins = dave_wins + 5) :
  jerry_wins + dave_wins + ken_wins = 32 :=
by
  sorry

end NUMINAMATH_GPT_total_games_played_l1926_192628


namespace NUMINAMATH_GPT_find_width_of_jordan_rectangle_l1926_192623

theorem find_width_of_jordan_rectangle (width : ℕ) (h1 : 12 * 15 = 9 * width) : width = 20 :=
by
  sorry

end NUMINAMATH_GPT_find_width_of_jordan_rectangle_l1926_192623


namespace NUMINAMATH_GPT_find_fraction_l1926_192612

theorem find_fraction (n d : ℕ) (h1 : n / (d + 1) = 1 / 2) (h2 : (n + 1) / d = 1) : n / d = 2 / 3 := 
by 
  sorry

end NUMINAMATH_GPT_find_fraction_l1926_192612


namespace NUMINAMATH_GPT_red_cookies_count_l1926_192637

-- Definitions of the conditions
def total_cookies : ℕ := 86
def pink_cookies : ℕ := 50

-- The proof problem statement
theorem red_cookies_count : ∃ y : ℕ, y = total_cookies - pink_cookies := by
  use 36
  show 36 = total_cookies - pink_cookies
  sorry

end NUMINAMATH_GPT_red_cookies_count_l1926_192637


namespace NUMINAMATH_GPT_probability_prime_sum_30_l1926_192653

def prime_numbers_up_to_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def prime_pairs_summing_to_30 : List (ℕ × ℕ) := [(7, 23), (11, 19), (13, 17)]

def num_prime_pairs := (prime_numbers_up_to_30.length.choose 2)

theorem probability_prime_sum_30 :
  (prime_pairs_summing_to_30.length / num_prime_pairs : ℚ) = 1 / 15 :=
sorry

end NUMINAMATH_GPT_probability_prime_sum_30_l1926_192653


namespace NUMINAMATH_GPT_clubs_students_equal_l1926_192632

theorem clubs_students_equal
  (C E : ℕ)
  (h1 : ∃ N, N = 3 * C)
  (h2 : ∃ N, N = 3 * E) :
  C = E :=
by
  sorry

end NUMINAMATH_GPT_clubs_students_equal_l1926_192632


namespace NUMINAMATH_GPT_cost_of_superman_game_l1926_192674

-- Define the costs as constants
def cost_batman_game : ℝ := 13.60
def total_amount_spent : ℝ := 18.66

-- Define the theorem to prove the cost of the Superman game
theorem cost_of_superman_game : total_amount_spent - cost_batman_game = 5.06 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_superman_game_l1926_192674


namespace NUMINAMATH_GPT_points_on_circle_l1926_192626

theorem points_on_circle (t : ℝ) : 
  ( (2 - 3 * t^2) / (2 + t^2) )^2 + ( 3 * t / (2 + t^2) )^2 = 1 := 
by 
  sorry

end NUMINAMATH_GPT_points_on_circle_l1926_192626


namespace NUMINAMATH_GPT_chris_pounds_of_nuts_l1926_192631

theorem chris_pounds_of_nuts :
  ∀ (R : ℝ) (x : ℝ),
  (∃ (N : ℝ), N = 4 * R) →
  (∃ (total_mixture_cost : ℝ), total_mixture_cost = 3 * R + 4 * R * x) →
  (3 * R = 0.15789473684210525 * total_mixture_cost) →
  x = 4 :=
by
  intros R x hN htotal_mixture_cost hRA
  sorry

end NUMINAMATH_GPT_chris_pounds_of_nuts_l1926_192631


namespace NUMINAMATH_GPT_james_present_age_l1926_192654

-- Definitions and conditions
variables (D J : ℕ) -- Dan's and James's ages are natural numbers

-- Condition 1: The ratio between Dan's and James's ages
def ratio_condition : Prop := (D * 5 = J * 6)

-- Condition 2: In 4 years, Dan will be 28
def future_age_condition : Prop := (D + 4 = 28)

-- The proof goal: James's present age is 20
theorem james_present_age : ratio_condition D J ∧ future_age_condition D → J = 20 :=
by
  sorry

end NUMINAMATH_GPT_james_present_age_l1926_192654


namespace NUMINAMATH_GPT_sum_p_q_r_l1926_192657

def b (n : ℕ) : ℕ :=
if n < 1 then 0 else
if n < 2 then 2 else
if n < 4 then 4 else
if n < 7 then 6
else 6 -- Continue this pattern for illustration; an infinite structure would need proper handling for all n.

noncomputable def p := 2
noncomputable def q := 0
noncomputable def r := 0

theorem sum_p_q_r : p + q + r = 2 :=
by sorry

end NUMINAMATH_GPT_sum_p_q_r_l1926_192657


namespace NUMINAMATH_GPT_transform_expression_to_product_l1926_192642

variables (a b c d s: ℝ)

theorem transform_expression_to_product
  (h1 : d = a + b + c)
  (h2 : s = (a + b + c + d) / 2) :
    2 * (a^2 * b^2 + a^2 * c^2 + a^2 * d^2 + b^2 * c^2 + b^2 * d^2 + c^2 * d^2) -
    (a^4 + b^4 + c^4 + d^4) + 8 * a * b * c * d = 16 * (s - a) * (s - b) * (s - c) * (s - d) :=
by
  sorry

end NUMINAMATH_GPT_transform_expression_to_product_l1926_192642


namespace NUMINAMATH_GPT_arithmetic_sequence_property_l1926_192667

def arith_seq (a : ℕ → ℤ) (a1 a3 : ℤ) (d : ℤ) : Prop :=
  a 1 = a1 ∧ a 3 = a3 ∧ (a 3 - a 1) = 2 * d

theorem arithmetic_sequence_property :
  ∀ (a : ℕ → ℤ), ∃ d : ℤ, arith_seq a 1 (-3) d →
  (1 - (a 2) - a 3 - (a 4) - (a 5) = 17) :=
by
  intros a
  use -2
  simp [arith_seq, *]
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_property_l1926_192667


namespace NUMINAMATH_GPT_overlap_region_area_l1926_192609

noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  1/2 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

noncomputable def overlap_area : ℝ := 
  let A : ℝ × ℝ := (0, 0);
  let B : ℝ × ℝ := (6, 2);
  let C : ℝ × ℝ := (2, 6);
  let D : ℝ × ℝ := (6, 6);
  let E : ℝ × ℝ := (0, 2);
  let F : ℝ × ℝ := (2, 0);
  let P1 : ℝ × ℝ := (2, 2);
  let P2 : ℝ × ℝ := (4, 2);
  let P3 : ℝ × ℝ := (3, 3);
  let P4 : ℝ × ℝ := (2, 3);
  1/2 * abs (P1.1 * (P2.2 - P4.2) + P2.1 * (P3.2 - P1.2) + P3.1 * (P4.2 - P2.2) + P4.1 * (P1.2 - P3.2))

theorem overlap_region_area :
  let A : ℝ × ℝ := (0, 0);
  let B : ℝ × ℝ := (6, 2);
  let C : ℝ × ℝ := (2, 6);
  let D : ℝ × ℝ := (6, 6);
  let E : ℝ × ℝ := (0, 2);
  let F : ℝ × ℝ := (2, 0);
  triangle_area A B C > 0 →
  triangle_area D E F > 0 →
  overlap_area = 0.5 :=
by { sorry }

end NUMINAMATH_GPT_overlap_region_area_l1926_192609
