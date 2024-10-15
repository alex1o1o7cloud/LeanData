import Mathlib

namespace NUMINAMATH_GPT_elements_author_is_euclid_l1501_150197

def author_of_elements := "Euclid"

theorem elements_author_is_euclid : author_of_elements = "Euclid" :=
by
  rfl -- Reflexivity of equality, since author_of_elements is defined to be "Euclid".

end NUMINAMATH_GPT_elements_author_is_euclid_l1501_150197


namespace NUMINAMATH_GPT_B_speaks_truth_60_l1501_150147

variable (P_A P_B P_A_and_B : ℝ)

-- Given conditions
def A_speaks_truth_85 : Prop := P_A = 0.85
def both_speak_truth_051 : Prop := P_A_and_B = 0.51

-- Solution condition
noncomputable def B_speaks_truth_percentage : ℝ := P_A_and_B / P_A

-- Statement to prove
theorem B_speaks_truth_60 (hA : A_speaks_truth_85 P_A) (hAB : both_speak_truth_051 P_A_and_B) : B_speaks_truth_percentage P_A_and_B P_A = 0.6 :=
by
  rw [A_speaks_truth_85] at hA
  rw [both_speak_truth_051] at hAB
  unfold B_speaks_truth_percentage
  sorry

end NUMINAMATH_GPT_B_speaks_truth_60_l1501_150147


namespace NUMINAMATH_GPT_expected_rolls_to_2010_l1501_150160

noncomputable def expected_rolls_for_sum (n : ℕ) : ℝ :=
  if n = 2010 then 574.5238095 else sorry

theorem expected_rolls_to_2010 : expected_rolls_for_sum 2010 = 574.5238095 := sorry

end NUMINAMATH_GPT_expected_rolls_to_2010_l1501_150160


namespace NUMINAMATH_GPT_externally_tangent_circles_m_l1501_150105

def circle1_eqn (x y : ℝ) : Prop := x^2 + y^2 = 4

def circle2_eqn (x y m : ℝ) : Prop := x^2 + y^2 - 2 * m * x + m^2 - 1 = 0

theorem externally_tangent_circles_m (m : ℝ) :
  (∀ x y : ℝ, circle1_eqn x y) →
  (∀ x y : ℝ, circle2_eqn x y m) →
  m = 3 ∨ m = -3 :=
by sorry

end NUMINAMATH_GPT_externally_tangent_circles_m_l1501_150105


namespace NUMINAMATH_GPT_license_plate_count_l1501_150162

-- Define the number of letters and digits
def num_letters := 26
def num_digits := 10
def num_odd_digits := 5  -- (1, 3, 5, 7, 9)
def num_even_digits := 5  -- (0, 2, 4, 6, 8)

-- Calculate the number of possible license plates
theorem license_plate_count : 
  (num_letters ^ 3) * ((num_even_digits * num_odd_digits * num_digits) * 3) = 13182000 :=
by sorry

end NUMINAMATH_GPT_license_plate_count_l1501_150162


namespace NUMINAMATH_GPT_negation_of_proposition_l1501_150126

open Real

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, x > sin x) ↔ (∃ x : ℝ, x ≤ sin x) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_proposition_l1501_150126


namespace NUMINAMATH_GPT_janet_initial_crayons_proof_l1501_150193

-- Define the initial number of crayons Michelle has
def michelle_initial_crayons : ℕ := 2

-- Define the final number of crayons Michelle will have after receiving Janet's crayons
def michelle_final_crayons : ℕ := 4

-- Define the function that calculates Janet's initial crayons
def janet_initial_crayons (m_i m_f : ℕ) : ℕ := m_f - m_i

-- The Lean statement to prove Janet's initial number of crayons
theorem janet_initial_crayons_proof : janet_initial_crayons michelle_initial_crayons michelle_final_crayons = 2 :=
by
  -- Proof steps go here (we use sorry to skip the proof)
  sorry

end NUMINAMATH_GPT_janet_initial_crayons_proof_l1501_150193


namespace NUMINAMATH_GPT_adam_first_half_correct_l1501_150195

-- Define the conditions
def second_half_correct := 2
def points_per_question := 8
def final_score := 80

-- Define the number of questions Adam answered correctly in the first half
def first_half_correct :=
  (final_score - (second_half_correct * points_per_question)) / points_per_question

-- Statement to prove
theorem adam_first_half_correct : first_half_correct = 8 :=
by
  -- skipping the proof
  sorry

end NUMINAMATH_GPT_adam_first_half_correct_l1501_150195


namespace NUMINAMATH_GPT_focal_radii_l1501_150127

theorem focal_radii (a e x y : ℝ) (h1 : x + y = 2 * a) (h2 : x - y = 2 * e) : x = a + e ∧ y = a - e :=
by
  -- We will add here the actual proof, but for now, we leave it as a placeholder.
  sorry

end NUMINAMATH_GPT_focal_radii_l1501_150127


namespace NUMINAMATH_GPT_mul_99_101_square_98_l1501_150161

theorem mul_99_101 : 99 * 101 = 9999 := sorry

theorem square_98 : 98^2 = 9604 := sorry

end NUMINAMATH_GPT_mul_99_101_square_98_l1501_150161


namespace NUMINAMATH_GPT_smallest_positive_value_is_A_l1501_150122

noncomputable def expr_A : ℝ := 12 - 4 * Real.sqrt 8
noncomputable def expr_B : ℝ := 4 * Real.sqrt 8 - 12
noncomputable def expr_C : ℝ := 20 - 6 * Real.sqrt 10
noncomputable def expr_D : ℝ := 60 - 15 * Real.sqrt 16
noncomputable def expr_E : ℝ := 15 * Real.sqrt 16 - 60

theorem smallest_positive_value_is_A :
  expr_A = 12 - 4 * Real.sqrt 8 ∧ 
  expr_B = 4 * Real.sqrt 8 - 12 ∧ 
  expr_C = 20 - 6 * Real.sqrt 10 ∧ 
  expr_D = 60 - 15 * Real.sqrt 16 ∧ 
  expr_E = 15 * Real.sqrt 16 - 60 ∧ 
  expr_A > 0 ∧ 
  expr_A < expr_C := 
sorry

end NUMINAMATH_GPT_smallest_positive_value_is_A_l1501_150122


namespace NUMINAMATH_GPT_arctan_sum_l1501_150186

theorem arctan_sum (x y : ℝ) (hx : x = 3) (hy : y = 7) :
  Real.arctan (x / y) + Real.arctan (y / x) = Real.pi / 2 := 
by
  rw [hx, hy]
  sorry

end NUMINAMATH_GPT_arctan_sum_l1501_150186


namespace NUMINAMATH_GPT_remainder_of_power_division_l1501_150148

-- Define the main entities
def power : ℕ := 3
def exponent : ℕ := 19
def divisor : ℕ := 10

-- Define the proof problem
theorem remainder_of_power_division :
  (power ^ exponent) % divisor = 7 := 
  by 
    sorry

end NUMINAMATH_GPT_remainder_of_power_division_l1501_150148


namespace NUMINAMATH_GPT_paint_needed_for_720_statues_l1501_150199

noncomputable def paint_for_similar_statues (n : Nat) (h₁ h₂ : ℝ) (p₁ : ℝ) : ℝ :=
  let ratio := (h₂ / h₁) ^ 2
  n * (ratio * p₁)

theorem paint_needed_for_720_statues :
  paint_for_similar_statues 720 12 2 1 = 20 :=
by
  sorry

end NUMINAMATH_GPT_paint_needed_for_720_statues_l1501_150199


namespace NUMINAMATH_GPT_trig_signs_l1501_150181

-- The conditions formulated as hypotheses
theorem trig_signs (h1 : Real.pi / 2 < 2 ∧ 2 < 3 ∧ 3 < Real.pi ∧ Real.pi < 4 ∧ 4 < 3 * Real.pi / 2) : 
  Real.sin 2 * Real.cos 3 * Real.tan 4 < 0 := 
sorry

end NUMINAMATH_GPT_trig_signs_l1501_150181


namespace NUMINAMATH_GPT_notebook_cost_l1501_150176

theorem notebook_cost (s n c : ℕ) (h1 : s > 17) (h2 : n > 2 ∧ n % 2 = 0) (h3 : c > n) (h4 : s * c * n = 2013) : c = 61 :=
sorry

end NUMINAMATH_GPT_notebook_cost_l1501_150176


namespace NUMINAMATH_GPT_standard_deviation_is_one_l1501_150150

noncomputable def standard_deviation (μ : ℝ) (σ : ℝ) : Prop :=
  ∀ x : ℝ, (0.68 * μ ≤ x ∧ x ≤ 1.32 * μ) → σ = 1

theorem standard_deviation_is_one (a : ℝ) (σ : ℝ) :
  (0.68 * a ≤ a + σ ∧ a + σ ≤ 1.32 * a) → σ = 1 :=
by
  -- Proof omitted.
  sorry

end NUMINAMATH_GPT_standard_deviation_is_one_l1501_150150


namespace NUMINAMATH_GPT_terms_of_sequence_are_equal_l1501_150183

theorem terms_of_sequence_are_equal
    (n : ℤ)
    (h_n : n ≥ 2018)
    (a b : ℕ → ℕ)
    (h_a_distinct : ∀ i j, i ≠ j → a i ≠ a j)
    (h_b_distinct : ∀ i j, i ≠ j → b i ≠ b j)
    (h_a_bounds : ∀ i, a i ≤ 5 * n)
    (h_b_bounds : ∀ i, b i ≤ 5 * n)
    (h_arith_seq : ∀ i, (a (i + 1) * b i - a i * b (i + 1)) = (a 1 * b 0 - a 0 * b 1) * i) :
    ∀ i j, (a i * b j = a j * b i) := 
by 
  sorry

end NUMINAMATH_GPT_terms_of_sequence_are_equal_l1501_150183


namespace NUMINAMATH_GPT_least_number_to_divisible_l1501_150104

theorem least_number_to_divisible (x : ℕ) : 
  (∃ x, (1049 + x) % 25 = 0) ∧ (∀ y, y < x → (1049 + y) % 25 ≠ 0) ↔ x = 1 :=
by
  sorry

end NUMINAMATH_GPT_least_number_to_divisible_l1501_150104


namespace NUMINAMATH_GPT_find_g_3_l1501_150170

-- Definitions and conditions
variable (g : ℝ → ℝ)
variable (h : ∀ x : ℝ, g (x - 1) = 2 * x + 6)

-- Theorem: Proof problem corresponding to the problem
theorem find_g_3 : g 3 = 14 :=
by
  -- Insert proof here
  sorry

end NUMINAMATH_GPT_find_g_3_l1501_150170


namespace NUMINAMATH_GPT_bridge_length_l1501_150152

variable (speed : ℝ) (time_minutes : ℝ)
variable (time_hours : ℝ := time_minutes / 60)

theorem bridge_length (h1 : speed = 5) (h2 : time_minutes = 15) : 
  speed * time_hours = 1.25 := by
  sorry

end NUMINAMATH_GPT_bridge_length_l1501_150152


namespace NUMINAMATH_GPT_total_paintable_wall_area_l1501_150185

/-- 
  Conditions:
  - John's house has 4 bedrooms.
  - Each bedroom is 15 feet long, 12 feet wide, and 10 feet high.
  - Doorways, windows, and a fireplace occupy 85 square feet per bedroom.
  Question: Prove that the total paintable wall area is 1820 square feet.
--/
theorem total_paintable_wall_area 
  (num_bedrooms : ℕ)
  (length width height non_paintable_area : ℕ)
  (h_num_bedrooms : num_bedrooms = 4)
  (h_length : length = 15)
  (h_width : width = 12)
  (h_height : height = 10)
  (h_non_paintable_area : non_paintable_area = 85) :
  (num_bedrooms * ((2 * (length * height) + 2 * (width * height)) - non_paintable_area) = 1820) :=
by
  sorry

end NUMINAMATH_GPT_total_paintable_wall_area_l1501_150185


namespace NUMINAMATH_GPT_chris_money_before_birthday_l1501_150178

/-- Chris's total money now is $279 -/
def money_now : ℕ := 279

/-- Money received from Chris's grandmother is $25 -/
def money_grandmother : ℕ := 25

/-- Money received from Chris's aunt and uncle is $20 -/
def money_aunt_uncle : ℕ := 20

/-- Money received from Chris's parents is $75 -/
def money_parents : ℕ := 75

/-- Total money received for his birthday -/
def money_received : ℕ := money_grandmother + money_aunt_uncle + money_parents

/-- Money Chris had before his birthday -/
def money_before_birthday : ℕ := money_now - money_received

theorem chris_money_before_birthday : money_before_birthday = 159 := by
  sorry

end NUMINAMATH_GPT_chris_money_before_birthday_l1501_150178


namespace NUMINAMATH_GPT_closest_multiple_of_18_l1501_150155

def is_multiple_of_2 (n : ℤ) : Prop := n % 2 = 0
def is_multiple_of_9 (n : ℤ) : Prop := n % 9 = 0
def is_multiple_of_18 (n : ℤ) : Prop := is_multiple_of_2 n ∧ is_multiple_of_9 n

theorem closest_multiple_of_18 (n : ℤ) (h : n = 2509) : 
  ∃ k : ℤ, is_multiple_of_18 k ∧ (abs (2509 - k) = 7) :=
sorry

end NUMINAMATH_GPT_closest_multiple_of_18_l1501_150155


namespace NUMINAMATH_GPT_pool_capacity_l1501_150165

theorem pool_capacity:
  (∃ (V1 V2 : ℝ) (t : ℝ), 
    (V1 = t / 120) ∧ 
    (V2 = V1 + 50) ∧ 
    (V1 + V2 = t / 48) ∧ 
    t = 12000) := 
by 
  sorry

end NUMINAMATH_GPT_pool_capacity_l1501_150165


namespace NUMINAMATH_GPT_find_x_l1501_150145

theorem find_x (x : ℤ) (h : (2008 + x)^2 = x^2) : x = -1004 :=
sorry

end NUMINAMATH_GPT_find_x_l1501_150145


namespace NUMINAMATH_GPT_total_water_filled_jars_l1501_150180

theorem total_water_filled_jars (x : ℕ) (h : 4 * x + 2 * x + x = 14 * 4) : 3 * x = 24 :=
by
  sorry

end NUMINAMATH_GPT_total_water_filled_jars_l1501_150180


namespace NUMINAMATH_GPT_OHara_triple_example_l1501_150106

def is_OHara_triple (a b x : ℕ) : Prop :=
  (Real.sqrt a + Real.sqrt b = x)

theorem OHara_triple_example : is_OHara_triple 36 25 11 :=
by {
  sorry
}

end NUMINAMATH_GPT_OHara_triple_example_l1501_150106


namespace NUMINAMATH_GPT_number_of_chain_links_l1501_150156

noncomputable def length_of_chain (number_of_links : ℕ) : ℝ :=
  (number_of_links * (7 / 3)) + 1

theorem number_of_chain_links (n m : ℕ) (d : ℝ) (thickness : ℝ) (max_length min_length : ℕ) 
  (h1 : d = 2 + 1 / 3)
  (h2 : thickness = 0.5)
  (h3 : max_length = 36)
  (h4 : min_length = 22)
  (h5 : m = n + 6)
  : length_of_chain n = 22 ∧ length_of_chain m = 36 
  :=
  sorry

end NUMINAMATH_GPT_number_of_chain_links_l1501_150156


namespace NUMINAMATH_GPT_father_twice_as_old_in_years_l1501_150158

-- Conditions
def father_age : ℕ := 42
def son_age : ℕ := 14
def years : ℕ := 14

-- Proof statement
theorem father_twice_as_old_in_years : (father_age + years) = 2 * (son_age + years) :=
by
  -- Proof content is omitted as per the instruction.
  sorry

end NUMINAMATH_GPT_father_twice_as_old_in_years_l1501_150158


namespace NUMINAMATH_GPT_charlie_has_largest_final_answer_l1501_150115

theorem charlie_has_largest_final_answer :
  let alice := (15 - 2)^2 + 3
  let bob := 15^2 - 2 + 3
  let charlie := (15 - 2 + 3)^2
  charlie > alice ∧ charlie > bob :=
by
  -- Definitions of intermediate variables
  let alice := (15 - 2)^2 + 3
  let bob := 15^2 - 2 + 3
  let charlie := (15 - 2 + 3)^2
  -- Comparison assertions
  sorry

end NUMINAMATH_GPT_charlie_has_largest_final_answer_l1501_150115


namespace NUMINAMATH_GPT_multiply_fractions_l1501_150139

theorem multiply_fractions :
  (2 / 9) * (5 / 14) = 5 / 63 :=
by
  sorry

end NUMINAMATH_GPT_multiply_fractions_l1501_150139


namespace NUMINAMATH_GPT_find_prime_p_l1501_150167

theorem find_prime_p (p : ℕ) (hp : Nat.Prime p) (hp_plus_10 : Nat.Prime (p + 10)) (hp_plus_14 : Nat.Prime (p + 14)) : p = 3 := 
sorry

end NUMINAMATH_GPT_find_prime_p_l1501_150167


namespace NUMINAMATH_GPT_rides_ratio_l1501_150177

theorem rides_ratio (total_money rides_spent dessert_spent money_left : ℕ) 
  (h1 : total_money = 30) 
  (h2 : dessert_spent = 5) 
  (h3 : money_left = 10) 
  (h4 : total_money - money_left = rides_spent + dessert_spent) : 
  (rides_spent : ℚ) / total_money = 1 / 2 := 
sorry

end NUMINAMATH_GPT_rides_ratio_l1501_150177


namespace NUMINAMATH_GPT_absent_laborers_l1501_150143

theorem absent_laborers (L : ℝ) (A : ℝ) (hL : L = 17.5) (h_work_done : (L - A) / 10 = L / 6) : A = 14 :=
by
  sorry

end NUMINAMATH_GPT_absent_laborers_l1501_150143


namespace NUMINAMATH_GPT_fraction_addition_l1501_150154

theorem fraction_addition :
  (1 / 3 * 2 / 5) + 1 / 4 = 23 / 60 := 
  sorry

end NUMINAMATH_GPT_fraction_addition_l1501_150154


namespace NUMINAMATH_GPT_sequence_periodic_from_some_term_l1501_150121

def is_bounded (s : ℕ → ℤ) (M : ℤ) : Prop :=
  ∀ n, |s n| ≤ M

def is_periodic_from (s : ℕ → ℤ) (N : ℕ) (p : ℕ) : Prop :=
  ∀ n, s (N + n) = s (N + n + p)

theorem sequence_periodic_from_some_term (s : ℕ → ℤ) (M : ℤ) (h_bounded : is_bounded s M)
    (h_recurrence : ∀ n, s (n + 5) = (5 * s (n + 4) ^ 3 + s (n + 3) - 3 * s (n + 2) + s n) / (2 * s (n + 2) + s (n + 1) ^ 2 + s (n + 1) * s n)) :
    ∃ N p, is_periodic_from s N p := by
  sorry

end NUMINAMATH_GPT_sequence_periodic_from_some_term_l1501_150121


namespace NUMINAMATH_GPT_complex_in_third_quadrant_l1501_150151

open Complex

noncomputable def quadrant (z : ℂ) : ℕ :=
  if z.re > 0 ∧ z.im > 0 then 1
  else if z.re < 0 ∧ z.im > 0 then 2
  else if z.re < 0 ∧ z.im < 0 then 3
  else 4

theorem complex_in_third_quadrant (z : ℂ) (h : (2 + I) * z = -I) : quadrant z = 3 := by
  sorry

end NUMINAMATH_GPT_complex_in_third_quadrant_l1501_150151


namespace NUMINAMATH_GPT_sum_of_divisor_and_quotient_is_correct_l1501_150189

theorem sum_of_divisor_and_quotient_is_correct (divisor quotient : ℕ)
  (h1 : 1000 ≤ divisor ∧ divisor < 10000) -- Divisor is a four-digit number.
  (h2 : quotient * divisor + remainder = original_number) -- Division condition (could be more specific)
  (h3 : remainder < divisor) -- Remainder condition
  (h4 : original_number = 82502) -- Given original number
  : divisor + quotient = 723 := 
sorry

end NUMINAMATH_GPT_sum_of_divisor_and_quotient_is_correct_l1501_150189


namespace NUMINAMATH_GPT_evaluate_expression_l1501_150142

variable {c d : ℝ}

theorem evaluate_expression (h : c ≠ d ∧ c ≠ -d) :
  (c^4 - d^4) / (2 * (c^2 - d^2)) = (c^2 + d^2) / 2 :=
by sorry

end NUMINAMATH_GPT_evaluate_expression_l1501_150142


namespace NUMINAMATH_GPT_players_per_group_l1501_150196

theorem players_per_group (new_players : ℕ) (returning_players : ℕ) (groups : ℕ) 
  (h1 : new_players = 48) 
  (h2 : returning_players = 6) 
  (h3 : groups = 9) : 
  (new_players + returning_players) / groups = 6 :=
by
  sorry

end NUMINAMATH_GPT_players_per_group_l1501_150196


namespace NUMINAMATH_GPT_class_distances_l1501_150191

theorem class_distances (x y z : ℕ) 
  (h1 : y = x + 8)
  (h2 : z = 3 * x)
  (h3 : x + y + z = 108) : 
  x = 20 ∧ y = 28 ∧ z = 60 := 
  by sorry

end NUMINAMATH_GPT_class_distances_l1501_150191


namespace NUMINAMATH_GPT_range_of_m_l1501_150120

theorem range_of_m (m : ℝ) (h : (8 - m) / (m - 5) > 1) : 5 < m ∧ m < 13 / 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1501_150120


namespace NUMINAMATH_GPT_simplify_expression_l1501_150116

theorem simplify_expression (a b : ℤ) : 4 * a + 5 * b - a - 7 * b = 3 * a - 2 * b :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1501_150116


namespace NUMINAMATH_GPT_range_of_x_plus_2y_l1501_150159

theorem range_of_x_plus_2y {x y : ℝ} (hx : x > 0) (hy : y > 0) (h : 1/x + 2/y = 1) : x + 2 * y ≥ 9 :=
sorry

end NUMINAMATH_GPT_range_of_x_plus_2y_l1501_150159


namespace NUMINAMATH_GPT_new_circle_equation_l1501_150138

-- Define the initial conditions
def initial_circle_equation (x y : ℝ) : Prop := x^2 + 2*x + y^2 = 0
def radius_of_new_circle : ℝ := 2

-- Define the target equation of the circle
def target_circle_equation (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 4

-- The theorem statement
theorem new_circle_equation (x y : ℝ) :
  initial_circle_equation x y → target_circle_equation x y :=
sorry

end NUMINAMATH_GPT_new_circle_equation_l1501_150138


namespace NUMINAMATH_GPT_child_support_amount_l1501_150117

-- Definitions
def base_salary_1_3 := 30000
def base_salary_4_7 := 36000
def bonus_1 := 2000
def bonus_2 := 3000
def bonus_3 := 4000
def bonus_4 := 5000
def bonus_5 := 6000
def bonus_6 := 7000
def bonus_7 := 8000
def child_support_1_5 := 30 / 100
def child_support_6_7 := 25 / 100
def paid_total := 1200

-- Total Income per year
def income_year_1 := base_salary_1_3 + bonus_1
def income_year_2 := base_salary_1_3 + bonus_2
def income_year_3 := base_salary_1_3 + bonus_3
def income_year_4 := base_salary_4_7 + bonus_4
def income_year_5 := base_salary_4_7 + bonus_5
def income_year_6 := base_salary_4_7 + bonus_6
def income_year_7 := base_salary_4_7 + bonus_7

-- Child Support per year
def support_year_1 := child_support_1_5 * income_year_1
def support_year_2 := child_support_1_5 * income_year_2
def support_year_3 := child_support_1_5 * income_year_3
def support_year_4 := child_support_1_5 * income_year_4
def support_year_5 := child_support_1_5 * income_year_5
def support_year_6 := child_support_6_7 * income_year_6
def support_year_7 := child_support_6_7 * income_year_7

-- Total Support calculation
def total_owed := support_year_1 + support_year_2 + support_year_3 + 
                  support_year_4 + support_year_5 +
                  support_year_6 + support_year_7

-- Final amount owed
def amount_owed := total_owed - paid_total

-- Theorem statement
theorem child_support_amount :
  amount_owed = 75150 :=
sorry

end NUMINAMATH_GPT_child_support_amount_l1501_150117


namespace NUMINAMATH_GPT_find_a_and_b_l1501_150172

def star (a b : ℕ) : ℕ := a^b + a * b

theorem find_a_and_b (a b : ℕ) (h1 : 2 ≤ a) (h2 : 2 ≤ b) (h3 : star a b = 40) : a + b = 7 :=
by
  sorry

end NUMINAMATH_GPT_find_a_and_b_l1501_150172


namespace NUMINAMATH_GPT_sqrt_a_minus_b_squared_eq_one_l1501_150134

noncomputable def PointInThirdQuadrant (a b : ℝ) : Prop :=
  a < 0 ∧ b < 0

noncomputable def DistanceToYAxis (a : ℝ) : Prop :=
  abs a = 5

noncomputable def BCondition (b : ℝ) : Prop :=
  abs (b + 1) = 3

theorem sqrt_a_minus_b_squared_eq_one
    (a b : ℝ)
    (h1 : PointInThirdQuadrant a b)
    (h2 : DistanceToYAxis a)
    (h3 : BCondition b) :
    Real.sqrt ((a - b) ^ 2) = 1 := 
  sorry

end NUMINAMATH_GPT_sqrt_a_minus_b_squared_eq_one_l1501_150134


namespace NUMINAMATH_GPT_broken_perfect_spiral_shells_difference_l1501_150109

theorem broken_perfect_spiral_shells_difference :
  let perfect_shells := 17
  let broken_shells := 52
  let broken_spiral_shells := broken_shells / 2
  let not_spiral_perfect_shells := 12
  let spiral_perfect_shells := perfect_shells - not_spiral_perfect_shells
  broken_spiral_shells - spiral_perfect_shells = 21 := by
  sorry

end NUMINAMATH_GPT_broken_perfect_spiral_shells_difference_l1501_150109


namespace NUMINAMATH_GPT_range_of_independent_variable_l1501_150107

theorem range_of_independent_variable (x : ℝ) (hx : 1 - 2 * x ≥ 0) : x ≤ 0.5 :=
sorry

end NUMINAMATH_GPT_range_of_independent_variable_l1501_150107


namespace NUMINAMATH_GPT_batsman_average_l1501_150188

theorem batsman_average (avg_20 : ℕ) (avg_10 : ℕ) (total_matches_20 : ℕ) (total_matches_10 : ℕ) :
  avg_20 = 40 → avg_10 = 20 → total_matches_20 = 20 → total_matches_10 = 10 →
  (800 + 200) / 30 = 33.33 :=
by
  sorry

end NUMINAMATH_GPT_batsman_average_l1501_150188


namespace NUMINAMATH_GPT_solve_fraction_eq_l1501_150153

theorem solve_fraction_eq (x : ℝ) (h₁ : x ≠ 1) (h₂ : x ≠ 3) :
  (x = 0 ∧ (x^3 - 3*x^2) / (x^2 - 4*x + 3) + 2*x = 0) ∨ 
  (x = 2 / 3 ∧ (x^3 - 3*x^2) / (x^2 - 4*x + 3) + 2*x = 0) :=
sorry

end NUMINAMATH_GPT_solve_fraction_eq_l1501_150153


namespace NUMINAMATH_GPT_sequence_diff_n_l1501_150171

theorem sequence_diff_n {a : ℕ → ℕ} (h1 : a 1 = 1) 
(h2 : ∀ n : ℕ, a (n + 1) ≤ 2 * n) (n : ℕ) :
  ∃ p q : ℕ, a p - a q = n :=
sorry

end NUMINAMATH_GPT_sequence_diff_n_l1501_150171


namespace NUMINAMATH_GPT_cos_A_eq_find_a_l1501_150112

variable {A B C a b c : ℝ}

-- Proposition 1: If in triangle ABC, b^2 + c^2 - (sqrt 6) / 2 * b * c = a^2, then cos A = sqrt 6 / 4
theorem cos_A_eq (h : b ^ 2 + c ^ 2 - (Real.sqrt 6) / 2 * b * c = a ^ 2) : Real.cos A = Real.sqrt 6 / 4 :=
sorry

-- Proposition 2: Given b = sqrt 6, B = 2 * A, and b^2 + c^2 - (sqrt 6) / 2 * b * c = a^2, then a = 2
theorem find_a (h1 : b ^ 2 + c ^ 2 - (Real.sqrt 6) / 2 * b * c = a ^ 2) (h2 : B = 2 * A) (h3 : b = Real.sqrt 6) : a = 2 :=
sorry

end NUMINAMATH_GPT_cos_A_eq_find_a_l1501_150112


namespace NUMINAMATH_GPT_find_positive_real_unique_solution_l1501_150146

theorem find_positive_real_unique_solution (x : ℝ) (h : 0 < x ∧ (x - 6) / 16 = 6 / (x - 16)) : x = 22 :=
sorry

end NUMINAMATH_GPT_find_positive_real_unique_solution_l1501_150146


namespace NUMINAMATH_GPT_locust_population_doubling_time_l1501_150132

theorem locust_population_doubling_time 
  (h: ℕ)
  (initial_population : ℕ := 1000)
  (time_past : ℕ := 4)
  (future_time: ℕ := 10)
  (population_limit: ℕ := 128000) :
  1000 * 2 ^ ((10 + 4) / h) > 128000 → h = 2 :=
by
  sorry

end NUMINAMATH_GPT_locust_population_doubling_time_l1501_150132


namespace NUMINAMATH_GPT_square_of_chord_length_l1501_150131

/--
Given two circles with radii 10 and 7, and centers 15 units apart, if they intersect at a point P such that the chords QP and PR are of equal lengths, then the square of the length of chord QP is 289.
-/
theorem square_of_chord_length :
  ∀ (r1 r2 d x : ℝ), r1 = 10 → r2 = 7 → d = 15 →
  let cos_theta1 := (x^2 + r1^2 - r2^2) / (2 * r1 * x)
  let cos_theta2 := (x^2 + r2^2 - r1^2) / (2 * r2 * x)
  cos_theta1 = cos_theta2 →
  x^2 = 289 := 
by
  intros r1 r2 d x h1 h2 h3
  let cos_theta1 := (x^2 + r1^2 - r2^2) / (2 * r1 * x)
  let cos_theta2 := (x^2 + r2^2 - r1^2) / (2 * r2 * x)
  intro h4
  sorry

end NUMINAMATH_GPT_square_of_chord_length_l1501_150131


namespace NUMINAMATH_GPT_find_number_l1501_150187

theorem find_number (n : ℝ) (h : n / 0.04 = 400.90000000000003) : n = 16.036 := 
by
  sorry

end NUMINAMATH_GPT_find_number_l1501_150187


namespace NUMINAMATH_GPT_quadratic_equation_solution_unique_l1501_150149

theorem quadratic_equation_solution_unique (b : ℝ) (hb : b ≠ 0) (h1_sol : ∀ x1 x2 : ℝ, 2*b*x1^2 + 16*x1 + 5 = 0 → 2*b*x2^2 + 16*x2 + 5 = 0 → x1 = x2) :
  ∃ x : ℝ, x = -5/8 ∧ 2*b*x^2 + 16*x + 5 = 0 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_equation_solution_unique_l1501_150149


namespace NUMINAMATH_GPT_total_pencils_l1501_150173

theorem total_pencils (pencils_per_child : ℕ) (children : ℕ) (h1 : pencils_per_child = 2) (h2 : children = 9) : pencils_per_child * children = 18 :=
sorry

end NUMINAMATH_GPT_total_pencils_l1501_150173


namespace NUMINAMATH_GPT_value_of_fraction_l1501_150184

-- Lean 4 statement
theorem value_of_fraction (x y : ℝ) (h1 : 2 * x + y = 7) (h2 : x + 2 * y = 8) : (x + y) / 3 = 5 / 3 :=
by
  sorry

end NUMINAMATH_GPT_value_of_fraction_l1501_150184


namespace NUMINAMATH_GPT_prime_and_n_eq_m_minus_1_l1501_150125

theorem prime_and_n_eq_m_minus_1 (n m : ℕ) (h1 : n ≥ 2) (h2 : m ≥ 2)
  (h3 : ∀ k : ℕ, k ∈ Finset.range n.succ → k^n % m = 1) : Nat.Prime m ∧ n = m - 1 := 
sorry

end NUMINAMATH_GPT_prime_and_n_eq_m_minus_1_l1501_150125


namespace NUMINAMATH_GPT_wang_hao_height_is_158_l1501_150169

/-- Yao Ming's height in meters. -/
def yao_ming_height : ℝ := 2.29

/-- Wang Hao is 0.71 meters shorter than Yao Ming. -/
def height_difference : ℝ := 0.71

/-- Wang Hao's height in meters. -/
def wang_hao_height : ℝ := yao_ming_height - height_difference

theorem wang_hao_height_is_158 :
  wang_hao_height = 1.58 :=
by
  sorry

end NUMINAMATH_GPT_wang_hao_height_is_158_l1501_150169


namespace NUMINAMATH_GPT_even_n_if_fraction_is_integer_l1501_150168

theorem even_n_if_fraction_is_integer (n : ℕ) (h_pos : 0 < n) :
  (∃ a b : ℕ, 0 < b ∧ (a^2 + n^2) % (b^2 - n^2) = 0) → n % 2 = 0 := 
sorry

end NUMINAMATH_GPT_even_n_if_fraction_is_integer_l1501_150168


namespace NUMINAMATH_GPT_hyperbola_equation_center_origin_asymptote_l1501_150140

theorem hyperbola_equation_center_origin_asymptote
  (center_origin : ∀ x y : ℝ, x = 0 ∧ y = 0)
  (focus_parabola : ∃ x : ℝ, 4 * x^2 = 8 * x)
  (asymptote : ∀ x y : ℝ, x + y = 0):
  ∃ a b : ℝ, a^2 = 2 ∧ b^2 = 2 ∧ (x^2 / 2) - (y^2 / 2) = 1 := 
sorry

end NUMINAMATH_GPT_hyperbola_equation_center_origin_asymptote_l1501_150140


namespace NUMINAMATH_GPT_total_students_correct_l1501_150179

def third_grade_students := 203
def fourth_grade_students := third_grade_students + 125
def total_students := third_grade_students + fourth_grade_students

theorem total_students_correct :
  total_students = 531 :=
by
  -- We state that the total number of students is 531
  sorry

end NUMINAMATH_GPT_total_students_correct_l1501_150179


namespace NUMINAMATH_GPT_cone_volume_proof_l1501_150141

noncomputable def cone_volume (r : ℝ) (h : ℝ) : ℝ :=
  (1 / 3) * Real.pi * r^2 * h

theorem cone_volume_proof :
  (cone_volume 1 (Real.sqrt 3)) = (Real.sqrt 3 / 3) * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_cone_volume_proof_l1501_150141


namespace NUMINAMATH_GPT_lewis_found_20_items_l1501_150130

noncomputable def tanya_items : ℕ := 4

noncomputable def samantha_items : ℕ := 4 * tanya_items

noncomputable def lewis_items : ℕ := samantha_items + 4

theorem lewis_found_20_items : lewis_items = 20 := by
  sorry

end NUMINAMATH_GPT_lewis_found_20_items_l1501_150130


namespace NUMINAMATH_GPT_maximal_sum_of_xy_l1501_150113

theorem maximal_sum_of_xy (x y : ℤ) (h : x^2 + y^2 = 100) : ∃ (s : ℤ), s = 14 ∧ ∀ (u v : ℤ), u^2 + v^2 = 100 → u + v ≤ s :=
by sorry

end NUMINAMATH_GPT_maximal_sum_of_xy_l1501_150113


namespace NUMINAMATH_GPT_construct_quad_root_of_sums_l1501_150175

theorem construct_quad_root_of_sums (a b : ℝ) : ∃ c : ℝ, c = (a^4 + b^4)^(1/4) := 
by
  sorry

end NUMINAMATH_GPT_construct_quad_root_of_sums_l1501_150175


namespace NUMINAMATH_GPT_lizard_eyes_l1501_150110

theorem lizard_eyes (E W S : Nat) 
  (h1 : W = 3 * E) 
  (h2 : S = 7 * W) 
  (h3 : E = S + W - 69) : 
  E = 3 := 
by
  sorry

end NUMINAMATH_GPT_lizard_eyes_l1501_150110


namespace NUMINAMATH_GPT_coins_dimes_count_l1501_150129

theorem coins_dimes_count :
  ∃ (p n d q : ℕ), 
    p + n + d + q = 10 ∧ 
    p + 5 * n + 10 * d + 25 * q = 110 ∧ 
    p ≥ 1 ∧ n ≥ 1 ∧ d ≥ 1 ∧ q ≥ 2 ∧ d = 5 :=
by {
    sorry
}

end NUMINAMATH_GPT_coins_dimes_count_l1501_150129


namespace NUMINAMATH_GPT_fraction_result_l1501_150101

theorem fraction_result (x : ℚ) (h₁ : x * (3/4) = (1/6)) : (x - (1/12)) = (5/36) := 
sorry

end NUMINAMATH_GPT_fraction_result_l1501_150101


namespace NUMINAMATH_GPT_spring_work_done_l1501_150135

theorem spring_work_done (F : ℝ) (l : ℝ) (stretched_length : ℝ) (k : ℝ) (W : ℝ) 
  (hF : F = 10) (hl : l = 0.1) (hk : k = F / l) (h_stretched_length : stretched_length = 0.06) : 
  W = 0.18 :=
by
  sorry

end NUMINAMATH_GPT_spring_work_done_l1501_150135


namespace NUMINAMATH_GPT_margaret_time_correct_l1501_150166

def margaret_time : ℕ :=
  let n := 7
  let r := 15
  (Nat.factorial n) / r

theorem margaret_time_correct : margaret_time = 336 := by
  sorry

end NUMINAMATH_GPT_margaret_time_correct_l1501_150166


namespace NUMINAMATH_GPT_unique_solution_of_quadratics_l1501_150114

theorem unique_solution_of_quadratics (y : ℚ) 
    (h1 : 9 * y^2 + 8 * y - 3 = 0) 
    (h2 : 27 * y^2 + 35 * y - 12 = 0) : 
    y = 1 / 3 :=
sorry

end NUMINAMATH_GPT_unique_solution_of_quadratics_l1501_150114


namespace NUMINAMATH_GPT_equal_cost_l1501_150198

theorem equal_cost (x : ℝ) : (2.75 * x + 125 = 1.50 * x + 140) ↔ (x = 12) := 
by sorry

end NUMINAMATH_GPT_equal_cost_l1501_150198


namespace NUMINAMATH_GPT_max_roses_purchase_l1501_150124

/--
Given three purchasing options for roses:
1. Individual roses cost $5.30 each.
2. One dozen (12) roses cost $36.
3. Two dozen (24) roses cost $50.
Given a total budget of $680, prove that the maximum number of roses that can be purchased is 317.
-/
noncomputable def max_roses : ℝ := 317

/--
Prove that given the purchasing options and the budget, the maximum number of roses that can be purchased is 317.
-/
theorem max_roses_purchase (individual_cost dozen_cost two_dozen_cost budget : ℝ) 
  (h1 : individual_cost = 5.30) 
  (h2 : dozen_cost = 36) 
  (h3 : two_dozen_cost = 50) 
  (h4 : budget = 680) : 
  max_roses = 317 := 
sorry

end NUMINAMATH_GPT_max_roses_purchase_l1501_150124


namespace NUMINAMATH_GPT_axis_of_symmetry_parabola_l1501_150108

theorem axis_of_symmetry_parabola (x y : ℝ) : 
  (∃ k : ℝ, (y^2 = -8 * k) → (y^2 = -8 * x) → x = -1) :=
by
  sorry

end NUMINAMATH_GPT_axis_of_symmetry_parabola_l1501_150108


namespace NUMINAMATH_GPT_find_f_log_l1501_150102

def even_function (f : ℝ → ℝ) :=
  ∀ (x : ℝ), f x = f (-x)

def periodic_function (f : ℝ → ℝ) (p : ℝ) :=
  ∀ (x : ℝ), f (x + p) = f x

theorem find_f_log (f : ℝ → ℝ)
  (h_even : even_function f)
  (h_periodic : periodic_function f 2)
  (h_condition : ∀ (x : ℝ), -1 ≤ x ∧ x ≤ 0 → f x = 3 * x + 4 / 9) :
  f (Real.log 5 / Real.log (1 / 3)) = -5 / 9 :=
by
  sorry

end NUMINAMATH_GPT_find_f_log_l1501_150102


namespace NUMINAMATH_GPT_find_cost_10_pound_bag_l1501_150118

def cost_5_pound_bag : ℝ := 13.82
def cost_25_pound_bag : ℝ := 32.25
def minimum_required_weight : ℝ := 65
def maximum_required_weight : ℝ := 80
def least_possible_cost : ℝ := 98.75
def cost_10_pound_bag (cost : ℝ) : Prop :=
  ∃ n m l, 
    (n * 5 + m * 10 + l * 25 ≥ minimum_required_weight) ∧
    (n * 5 + m * 10 + l * 25 ≤ maximum_required_weight) ∧
    (n * cost_5_pound_bag + m * cost + l * cost_25_pound_bag = least_possible_cost)

theorem find_cost_10_pound_bag : cost_10_pound_bag 2 := 
by
  sorry

end NUMINAMATH_GPT_find_cost_10_pound_bag_l1501_150118


namespace NUMINAMATH_GPT_tire_price_l1501_150119

theorem tire_price (x : ℕ) (h : 4 * x + 5 = 485) : x = 120 :=
by
  sorry

end NUMINAMATH_GPT_tire_price_l1501_150119


namespace NUMINAMATH_GPT_sam_total_distance_l1501_150128

-- Definitions based on conditions
def first_half_distance : ℕ := 120
def first_half_time : ℕ := 3
def second_half_distance : ℕ := 80
def second_half_time : ℕ := 2
def sam_time : ℚ := 5.5

-- Marguerite's overall average speed
def marguerite_average_speed : ℚ := (first_half_distance + second_half_distance) / (first_half_time + second_half_time)

-- Theorem statement: Sam's total distance driven
theorem sam_total_distance : ∀ (d : ℚ), d = (marguerite_average_speed * sam_time) ↔ d = 220 := by
  intro d
  sorry

end NUMINAMATH_GPT_sam_total_distance_l1501_150128


namespace NUMINAMATH_GPT_final_price_calculation_l1501_150182

theorem final_price_calculation 
  (ticket_price : ℝ)
  (initial_discount : ℝ)
  (additional_discount : ℝ)
  (sales_tax : ℝ)
  (final_price : ℝ) 
  (h1 : ticket_price = 200) 
  (h2 : initial_discount = 0.25) 
  (h3 : additional_discount = 0.15) 
  (h4 : sales_tax = 0.07)
  (h5 : final_price = (ticket_price * (1 - initial_discount)) * (1 - additional_discount) * (1 + sales_tax)):
  final_price = 136.43 :=
by
  sorry

end NUMINAMATH_GPT_final_price_calculation_l1501_150182


namespace NUMINAMATH_GPT_compute_expr1_factorize_expr2_l1501_150194

-- Definition for Condition 1: None explicitly stated.

-- Theorem for Question 1
theorem compute_expr1 (y : ℝ) : (y - 1) * (y + 5) = y^2 + 4*y - 5 :=
by sorry

-- Definition for Condition 2: None explicitly stated.

-- Theorem for Question 2
theorem factorize_expr2 (x y : ℝ) : -x^2 + 4*x*y - 4*y^2 = -((x - 2*y)^2) :=
by sorry

end NUMINAMATH_GPT_compute_expr1_factorize_expr2_l1501_150194


namespace NUMINAMATH_GPT_c_divides_n_l1501_150164

theorem c_divides_n (a b c n : ℤ) (h : a * n^2 + b * n + c = 0) : c ∣ n :=
sorry

end NUMINAMATH_GPT_c_divides_n_l1501_150164


namespace NUMINAMATH_GPT_tan_alpha_plus_pi_div_4_l1501_150144

noncomputable def tan_plus_pi_div_4 (α : ℝ) : ℝ := Real.tan (α + Real.pi / 4)

theorem tan_alpha_plus_pi_div_4 (α : ℝ) 
  (h1 : α > Real.pi / 2) 
  (h2 : α < Real.pi) 
  (h3 : (Real.cos α, Real.sin α) • (Real.cos α ^ 2, Real.sin α - 1) = 1 / 5)
  : tan_plus_pi_div_4 α = -1 / 7 := sorry

end NUMINAMATH_GPT_tan_alpha_plus_pi_div_4_l1501_150144


namespace NUMINAMATH_GPT_value_of_q_when_p_is_smallest_l1501_150174

-- Definitions of primality
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m > 1, m < n → ¬ (n % m = 0)

-- smallest prime number
def smallest_prime : ℕ := 2

-- Given conditions
def p : ℕ := 3
def q : ℕ := 2 + 13 * p

-- The theorem to prove
theorem value_of_q_when_p_is_smallest :
  is_prime smallest_prime →
  is_prime q →
  smallest_prime = 2 →
  p = 3 →
  q = 41 :=
by sorry

end NUMINAMATH_GPT_value_of_q_when_p_is_smallest_l1501_150174


namespace NUMINAMATH_GPT_time_to_groom_rottweiler_l1501_150111

theorem time_to_groom_rottweiler
  (R : ℕ)  -- Time to groom a rottweiler
  (B : ℕ)  -- Time to groom a border collie
  (C : ℕ)  -- Time to groom a chihuahua
  (total_time_6R_9B_1C : 6 * R + 9 * B + C = 255)  -- Total time for grooming 6 rottweilers, 9 border collies, and 1 chihuahua
  (time_to_groom_border_collie : B = 10)  -- Time to groom a border collie is 10 minutes
  (time_to_groom_chihuahua : C = 45) :  -- Time to groom a chihuahua is 45 minutes
  R = 20 :=  -- Prove that it takes 20 minutes to groom a rottweiler
by
  sorry

end NUMINAMATH_GPT_time_to_groom_rottweiler_l1501_150111


namespace NUMINAMATH_GPT_a_range_of_proposition_l1501_150133

theorem a_range_of_proposition (a : ℝ) : (∀ x : ℝ, 2 < x ∧ x < 3 → x^2 + 5 <= a * x) ↔ a ∈ Set.Ici (2 * Real.sqrt 5) := by
  sorry

end NUMINAMATH_GPT_a_range_of_proposition_l1501_150133


namespace NUMINAMATH_GPT_max_value_l1501_150157

open Real

theorem max_value (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (ha1 : a ≤ 1) (hb1 : b ≤ 1) (hc1 : c ≤ 1/2) :
  sqrt (a * b * c) + sqrt ((1 - a) * (1 - b) * (1 - c)) ≤ (1 / sqrt 2) + (1 / 2) :=
sorry

end NUMINAMATH_GPT_max_value_l1501_150157


namespace NUMINAMATH_GPT_middle_school_soccer_league_l1501_150192

theorem middle_school_soccer_league (n : ℕ) (h : (n * (n - 1)) / 2 = 36) : n = 9 := 
  sorry

end NUMINAMATH_GPT_middle_school_soccer_league_l1501_150192


namespace NUMINAMATH_GPT_lines_are_skew_l1501_150136

def line1 (a t : ℝ) : ℝ × ℝ × ℝ := 
  (2 + 3 * t, 1 + 4 * t, a + 5 * t)
  
def line2 (u : ℝ) : ℝ × ℝ × ℝ := 
  (5 + 6 * u, 3 + 3 * u, 1 + 2 * u)

theorem lines_are_skew (a : ℝ) : (∀ t u : ℝ, line1 a t ≠ line2 u) ↔ a ≠ -4/5 :=
sorry

end NUMINAMATH_GPT_lines_are_skew_l1501_150136


namespace NUMINAMATH_GPT_find_shortest_height_l1501_150103

variable (T S P Q : ℝ)

theorem find_shortest_height (h1 : T = 77.75) (h2 : T = S + 9.5) (h3 : P = S + 5) (h4 : Q = P - 3) : S = 68.25 :=
  sorry

end NUMINAMATH_GPT_find_shortest_height_l1501_150103


namespace NUMINAMATH_GPT_range_of_a_l1501_150137

open Real

noncomputable def f (x a : ℝ) : ℝ := (exp x / 2) - (a / exp x)

def condition (x₁ x₂ a : ℝ) : Prop :=
  x₁ ≠ x₂ ∧ 1 ≤ x₁ ∧ x₁ ≤ 2 ∧ 1 ≤ x₂ ∧ x₂ ≤ 2 ∧ ((abs (f x₁ a) - abs (f x₂ a)) * (x₁ - x₂) > 0)

theorem range_of_a (a : ℝ) :
  (∀ (x₁ x₂ : ℝ), condition x₁ x₂ a) ↔ (- (exp 2) / 2 ≤ a ∧ a ≤ (exp 2) / 2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1501_150137


namespace NUMINAMATH_GPT_finish_fourth_task_l1501_150123

noncomputable def time_task_starts : ℕ := 12 -- Time in hours (12:00 PM)
noncomputable def time_task_ends : ℕ := 15 -- Time in hours (3:00 PM)
noncomputable def total_tasks : ℕ := 4 -- Total number of tasks
noncomputable def tasks_time (tasks: ℕ) := (time_task_ends - time_task_starts) * 60 / (total_tasks - 1) -- Time in minutes for each task

theorem finish_fourth_task : tasks_time 1 + ((total_tasks - 1) * tasks_time 1) = 240 := -- 4:00 PM expressed as 240 minutes from 12:00 PM
by
  sorry

end NUMINAMATH_GPT_finish_fourth_task_l1501_150123


namespace NUMINAMATH_GPT_walking_speed_of_A_l1501_150190

-- Given conditions
def B_speed := 20 -- kmph
def start_delay := 10 -- hours
def distance_covered := 200 -- km

-- Prove A's walking speed
theorem walking_speed_of_A (v : ℝ) (time_A : ℝ) (time_B : ℝ) :
  distance_covered = v * time_A ∧ distance_covered = B_speed * time_B ∧ time_B = time_A - start_delay → v = 10 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_walking_speed_of_A_l1501_150190


namespace NUMINAMATH_GPT_initial_water_amount_l1501_150163

theorem initial_water_amount (E D R F I : ℕ) 
  (hE : E = 2000) 
  (hD : D = 3500) 
  (hR : R = 350 * (30 / 10))
  (hF : F = 1550) 
  (h : I - (E + D) + R = F) : 
  I = 6000 :=
by
  sorry

end NUMINAMATH_GPT_initial_water_amount_l1501_150163


namespace NUMINAMATH_GPT_find_expression_l1501_150100

theorem find_expression 
  (E a : ℤ) 
  (h1 : (E + (3 * a - 8)) / 2 = 74) 
  (h2 : a = 28) : 
  E = 72 := 
by
  sorry

end NUMINAMATH_GPT_find_expression_l1501_150100
