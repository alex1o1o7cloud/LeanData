import Mathlib

namespace NUMINAMATH_GPT_total_number_of_coins_is_324_l325_32593

noncomputable def total_coins (total_sum : ℕ) (coins_20p : ℕ) (coins_25p_value : ℕ) : ℕ :=
    coins_20p + (coins_25p_value / 25)

theorem total_number_of_coins_is_324 (h_sum: 7100 = 71 * 100) (h_coins_20p: 200 * 20 = 4000) :
  total_coins 7100 200 3100 = 324 := by
  sorry

end NUMINAMATH_GPT_total_number_of_coins_is_324_l325_32593


namespace NUMINAMATH_GPT_farmer_profit_l325_32580

noncomputable def profit_earned : ℕ :=
  let pigs := 6
  let sale_price := 300
  let food_cost_per_month := 10
  let months_group1 := 12
  let months_group2 := 16
  let pigs_group1 := 3
  let pigs_group2 := 3
  let total_food_cost := (pigs_group1 * months_group1 * food_cost_per_month) + 
                         (pigs_group2 * months_group2 * food_cost_per_month)
  let total_revenue := pigs * sale_price
  total_revenue - total_food_cost

theorem farmer_profit : profit_earned = 960 := by
  unfold profit_earned
  sorry

end NUMINAMATH_GPT_farmer_profit_l325_32580


namespace NUMINAMATH_GPT_train_length_calculation_l325_32585

theorem train_length_calculation (len1 : ℝ) (speed1_kmph : ℝ) (speed2_kmph : ℝ) (crossing_time : ℝ) (len2 : ℝ) :
  len1 = 120.00001 → 
  speed1_kmph = 120 → 
  speed2_kmph = 80 → 
  crossing_time = 9 → 
  (len1 + len2) = ((speed1_kmph * 1000 / 3600 + speed2_kmph * 1000 / 3600) * crossing_time) → 
  len2 = 379.99949 :=
by
  intros hlen1 hspeed1 hspeed2 htime hdistance
  sorry

end NUMINAMATH_GPT_train_length_calculation_l325_32585


namespace NUMINAMATH_GPT_marcus_calzones_total_time_l325_32572

theorem marcus_calzones_total_time :
  let saute_onions_time := 20
  let saute_garlic_peppers_time := (1 / 4 : ℚ) * saute_onions_time
  let knead_time := 30
  let rest_time := 2 * knead_time
  let assemble_time := (1 / 10 : ℚ) * (knead_time + rest_time)
  let total_time := saute_onions_time + saute_garlic_peppers_time + knead_time + rest_time + assemble_time
  total_time = 124 :=
by
  let saute_onions_time := 20
  let saute_garlic_peppers_time := (1 / 4 : ℚ) * saute_onions_time
  let knead_time := 30
  let rest_time := 2 * knead_time
  let assemble_time := (1 / 10 : ℚ) * (knead_time + rest_time)
  let total_time := saute_onions_time + saute_garlic_peppers_time + knead_time + rest_time + assemble_time
  sorry

end NUMINAMATH_GPT_marcus_calzones_total_time_l325_32572


namespace NUMINAMATH_GPT_find_a_l325_32553

def A (a : ℤ) : Set ℤ := {-4, 2 * a - 1, a * a}
def B (a : ℤ) : Set ℤ := {a - 5, 1 - a, 9}

theorem find_a (a : ℤ) : (9 ∈ (A a ∩ B a)) ∧ (A a ∩ B a = {9}) ↔ a = -3 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l325_32553


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l325_32558

noncomputable def S (n : ℕ) : ℤ :=
  n * (-2012) + n * (n - 1) / 2 * (1 : ℤ)

theorem arithmetic_sequence_sum :
  (S 2012) / 2012 - (S 10) / 10 = 2002 → S 2017 = 2017 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l325_32558


namespace NUMINAMATH_GPT_simple_interest_rate_l325_32581

theorem simple_interest_rate :
  ∀ (P T F : ℝ), P = 1000 → T = 3 → F = 1300 → (F - P) = P * 0.1 * T :=
by
  intros P T F hP hT hF
  sorry

end NUMINAMATH_GPT_simple_interest_rate_l325_32581


namespace NUMINAMATH_GPT_gcd_of_102_and_238_l325_32519

theorem gcd_of_102_and_238 : Nat.gcd 102 238 = 34 := 
by 
  sorry

end NUMINAMATH_GPT_gcd_of_102_and_238_l325_32519


namespace NUMINAMATH_GPT_probability_at_least_one_die_less_3_l325_32521

-- Definitions
def total_outcomes_dice : ℕ := 64
def outcomes_no_die_less_3 : ℕ := 36
def favorable_outcomes : ℕ := total_outcomes_dice - outcomes_no_die_less_3
def probability : ℚ := favorable_outcomes / total_outcomes_dice

-- Theorem statement
theorem probability_at_least_one_die_less_3 :
  probability = 7 / 16 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_probability_at_least_one_die_less_3_l325_32521


namespace NUMINAMATH_GPT_arrange_books_l325_32596

-- We define the conditions about the number of books
def num_algebra_books : ℕ := 4
def num_calculus_books : ℕ := 5
def total_books : ℕ := num_algebra_books + num_calculus_books

-- The combination function which calculates binomial coefficients
def combination (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- The theorem stating that there are 126 ways to arrange the books
theorem arrange_books : combination total_books num_algebra_books = 126 :=
  by
    sorry

end NUMINAMATH_GPT_arrange_books_l325_32596


namespace NUMINAMATH_GPT_tomato_red_flesh_probability_l325_32531

theorem tomato_red_flesh_probability :
  (P_yellow_skin : ℝ) = 3 / 8 →
  (P_red_flesh_given_yellow_skin : ℝ) = 8 / 15 →
  (P_yellow_skin_given_not_red_flesh : ℝ) = 7 / 30 →
  (P_red_flesh : ℝ) = 1 / 4 := 
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_tomato_red_flesh_probability_l325_32531


namespace NUMINAMATH_GPT_calc_r_over_s_at_2_l325_32526

def r (x : ℝ) := 3 * (x - 4) * (x - 1)
def s (x : ℝ) := (x - 4) * (x + 3)

theorem calc_r_over_s_at_2 : (r 2) / (s 2) = 3 / 5 := by
  sorry

end NUMINAMATH_GPT_calc_r_over_s_at_2_l325_32526


namespace NUMINAMATH_GPT_problem_1_problem_2_l325_32545

noncomputable def a : ℝ := Real.sqrt 7 + 2
noncomputable def b : ℝ := Real.sqrt 7 - 2

theorem problem_1 : a^2 * b + b^2 * a = 6 * Real.sqrt 7 := by
  sorry

theorem problem_2 : a^2 + a * b + b^2 = 25 := by
  sorry

end NUMINAMATH_GPT_problem_1_problem_2_l325_32545


namespace NUMINAMATH_GPT_solve_for_y_l325_32535

theorem solve_for_y (y : ℚ) (h : (4 / 7) * (1 / 5) * y - 2 = 10) : y = 105 := by
  sorry

end NUMINAMATH_GPT_solve_for_y_l325_32535


namespace NUMINAMATH_GPT_oxygen_atoms_in_compound_l325_32589

theorem oxygen_atoms_in_compound (K_weight Br_weight O_weight molecular_weight : ℕ) 
    (hK : K_weight = 39) (hBr : Br_weight = 80) (hO : O_weight = 16) (hMW : molecular_weight = 168) 
    (n : ℕ) :
    168 = 39 + 80 + n * 16 → n = 3 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_oxygen_atoms_in_compound_l325_32589


namespace NUMINAMATH_GPT_xiaoli_estimate_greater_l325_32523

variable (p q a b : ℝ)

theorem xiaoli_estimate_greater (hpq : p > q) (hq0 : q > 0) (hab : a > b) : (p + a) - (q + b) > p - q := 
by 
  sorry

end NUMINAMATH_GPT_xiaoli_estimate_greater_l325_32523


namespace NUMINAMATH_GPT_power_sum_tenth_l325_32509

theorem power_sum_tenth (a b : ℝ) (h1 : a + b = 1)
    (h2 : a^2 + b^2 = 3)
    (h3 : a^3 + b^3 = 4)
    (h4 : a^4 + b^4 = 7)
    (h5 : a^5 + b^5 = 11) : 
    a^10 + b^10 = 123 := 
sorry

end NUMINAMATH_GPT_power_sum_tenth_l325_32509


namespace NUMINAMATH_GPT_bill_property_taxes_l325_32594

theorem bill_property_taxes 
  (take_home_salary sales_taxes gross_salary : ℕ)
  (income_tax_rate : ℚ)
  (take_home_salary_eq : take_home_salary = 40000)
  (sales_taxes_eq : sales_taxes = 3000)
  (gross_salary_eq : gross_salary = 50000)
  (income_tax_rate_eq : income_tax_rate = 0.1) :
  let income_taxes := (income_tax_rate * gross_salary) 
  let property_taxes := gross_salary - (income_taxes + sales_taxes + take_home_salary)
  property_taxes = 2000 := by
  sorry

end NUMINAMATH_GPT_bill_property_taxes_l325_32594


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l325_32557

theorem isosceles_triangle_perimeter (a b : ℝ)
  (h1 : b = 7)
  (h2 : a^2 - 8 * a + 15 = 0)
  (h3 : a * 2 > b)
  : 2 * a + b = 17 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l325_32557


namespace NUMINAMATH_GPT_find_n_range_l325_32547

theorem find_n_range (m n : ℝ) 
  (h_m : -Real.sqrt 3 ≤ m ∧ m ≤ Real.sqrt 3) :
  (∀ x y z : ℝ, 0 ≤ x^2 + 2 * y^2 + 3 * z^2 + 2 * x * y + 2 * m * z * x + 2 * n * y * z) ↔ 
  (m - Real.sqrt (3 - m^2) ≤ n ∧ n ≤ m + Real.sqrt (3 - m^2)) :=
by
  sorry

end NUMINAMATH_GPT_find_n_range_l325_32547


namespace NUMINAMATH_GPT_second_customer_headphones_l325_32539

theorem second_customer_headphones
  (H : ℕ)
  (M : ℕ)
  (x : ℕ)
  (H_eq : H = 30)
  (eq1 : 5 * M + 8 * H = 840)
  (eq2 : 3 * M + x * H = 480) :
  x = 4 :=
by
  sorry

end NUMINAMATH_GPT_second_customer_headphones_l325_32539


namespace NUMINAMATH_GPT_pensioners_painting_conditions_l325_32518

def boardCondition (A Z : ℕ) : Prop :=
(∀ x y, (∃ i j, i ≤ 1 ∧ j ≤ 1 ∧ (x + 3 = A) ∧ (i ≤ 2 ∧ j ≤ 4 ∨ i ≤ 4 ∧ j ≤ 2) → x + 2 * y = Z))

theorem pensioners_painting_conditions (A Z : ℕ) :
  (boardCondition A Z) ↔ (A = 0 ∧ Z = 0) ∨ (A = 9 ∧ Z = 8) :=
sorry

end NUMINAMATH_GPT_pensioners_painting_conditions_l325_32518


namespace NUMINAMATH_GPT_part1_part2_l325_32540

-- Problem part (1)
theorem part1 : (Real.sqrt 12 + Real.sqrt (4 / 3)) * Real.sqrt 3 = 8 := 
  sorry

-- Problem part (2)
theorem part2 : Real.sqrt 48 - Real.sqrt 54 / Real.sqrt 2 + (3 - Real.sqrt 3) * (3 + Real.sqrt 3) = Real.sqrt 3 + 6 := 
  sorry

end NUMINAMATH_GPT_part1_part2_l325_32540


namespace NUMINAMATH_GPT_green_square_area_percentage_l325_32568

noncomputable def flag_side_length (k: ℝ) : ℝ := k
noncomputable def cross_area_fraction : ℝ := 0.49
noncomputable def cross_area (k: ℝ) : ℝ := cross_area_fraction * k^2
noncomputable def cross_width (t: ℝ) : ℝ := t
noncomputable def green_square_side (x: ℝ) : ℝ := x
noncomputable def green_square_area (x: ℝ) : ℝ := x^2

theorem green_square_area_percentage (k: ℝ) (t: ℝ) (x: ℝ)
  (h1: x = 2 * t)
  (h2: 4 * t * (k - t) + x^2 = cross_area k)
  : green_square_area x / (k^2) * 100 = 6.01 :=
by
  sorry

end NUMINAMATH_GPT_green_square_area_percentage_l325_32568


namespace NUMINAMATH_GPT_larger_square_area_total_smaller_squares_area_l325_32501
noncomputable def largerSquareSideLengthFromCircleRadius (r : ℝ) : ℝ :=
  2 * (2 * r)

noncomputable def squareArea (side : ℝ) : ℝ :=
  side * side

theorem larger_square_area (r : ℝ) (h : r = 3) :
  squareArea (largerSquareSideLengthFromCircleRadius r) = 144 :=
by
  sorry

theorem total_smaller_squares_area (r : ℝ) (h : r = 3) :
  4 * squareArea (2 * r) = 144 :=
by
  sorry

end NUMINAMATH_GPT_larger_square_area_total_smaller_squares_area_l325_32501


namespace NUMINAMATH_GPT_jellybean_probability_l325_32500

/-- Abe holds 1 blue and 2 red jelly beans. 
    Bob holds 2 blue, 2 yellow, and 1 red jelly bean. 
    Each randomly picks a jelly bean to show the other. 
    What is the probability that the colors match? 
-/
theorem jellybean_probability :
  let abe_blue_prob := 1 / 3
  let bob_blue_prob := 2 / 5
  let abe_red_prob := 2 / 3
  let bob_red_prob := 1 / 5
  (abe_blue_prob * bob_blue_prob + abe_red_prob * bob_red_prob) = 4 / 15 :=
by
  sorry

end NUMINAMATH_GPT_jellybean_probability_l325_32500


namespace NUMINAMATH_GPT_cubic_roots_expression_l325_32524

theorem cubic_roots_expression (p q r : ℝ) 
  (h1 : p + q + r = 4) 
  (h2 : pq + pr + qr = 6) 
  (h3 : pqr = 3) : 
  p / (qr + 2) + q / (pr + 2) + r / (pq + 2) = 4 / 5 := 
by 
  sorry

end NUMINAMATH_GPT_cubic_roots_expression_l325_32524


namespace NUMINAMATH_GPT_hypotenuse_length_l325_32528

theorem hypotenuse_length (a b : ℕ) (h1 : a = 36) (h2 : b = 48) : 
  ∃ c : ℕ, c * c = a * a + b * b ∧ c = 60 := 
by 
  use 60
  sorry

end NUMINAMATH_GPT_hypotenuse_length_l325_32528


namespace NUMINAMATH_GPT_quadratic_sum_l325_32506

theorem quadratic_sum (x : ℝ) :
  (∃ a b c : ℝ, 6 * x^2 + 48 * x + 162 = a * (x + b) ^ 2 + c ∧ a + b + c = 76) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_sum_l325_32506


namespace NUMINAMATH_GPT_pascal_elements_sum_l325_32533

theorem pascal_elements_sum :
  (Nat.choose 20 4 + Nat.choose 20 5) = 20349 :=
by
  sorry

end NUMINAMATH_GPT_pascal_elements_sum_l325_32533


namespace NUMINAMATH_GPT_Kristy_baked_cookies_l325_32536

theorem Kristy_baked_cookies 
  (ate_by_Kristy : ℕ) (given_to_brother : ℕ) 
  (taken_by_first_friend : ℕ) (taken_by_second_friend : ℕ)
  (taken_by_third_friend : ℕ) (cookies_left : ℕ) 
  (h_K : ate_by_Kristy = 2) (h_B : given_to_brother = 1) 
  (h_F1 : taken_by_first_friend = 3) (h_F2 : taken_by_second_friend = 5)
  (h_F3 : taken_by_third_friend = 5) (h_L : cookies_left = 6) :
  ate_by_Kristy + given_to_brother 
  + taken_by_first_friend + taken_by_second_friend 
  + taken_by_third_friend + cookies_left = 22 := 
by
  sorry

end NUMINAMATH_GPT_Kristy_baked_cookies_l325_32536


namespace NUMINAMATH_GPT_commodity_price_difference_l325_32566

theorem commodity_price_difference (r : ℝ) (t : ℕ) :
  let P_X (t : ℕ) := 4.20 * (1 + (2*r + 10)/100)^(t - 2001)
  let P_Y (t : ℕ) := 4.40 * (1 + (r + 15)/100)^(t - 2001)
  P_X t = P_Y t + 0.90  ->
  ∃ t : ℕ, true :=
by
  sorry

end NUMINAMATH_GPT_commodity_price_difference_l325_32566


namespace NUMINAMATH_GPT_number_of_girls_attending_picnic_l325_32508

variables (g b : ℕ)

def hms_conditions : Prop :=
  g + b = 1500 ∧ (3 / 4 : ℝ) * g + (3 / 5 : ℝ) * b = 975

theorem number_of_girls_attending_picnic (h : hms_conditions g b) : (3 / 4 : ℝ) * g = 375 :=
sorry

end NUMINAMATH_GPT_number_of_girls_attending_picnic_l325_32508


namespace NUMINAMATH_GPT_tedra_harvested_2000kg_l325_32573

noncomputable def totalTomatoesHarvested : ℕ :=
  let wednesday : ℕ := 400
  let thursday : ℕ := wednesday / 2
  let total_wednesday_thursday := wednesday + thursday
  let remaining_friday : ℕ := 700
  let given_away_friday : ℕ := 700
  let friday := remaining_friday + given_away_friday
  total_wednesday_thursday + friday

theorem tedra_harvested_2000kg :
  totalTomatoesHarvested = 2000 := by
  sorry

end NUMINAMATH_GPT_tedra_harvested_2000kg_l325_32573


namespace NUMINAMATH_GPT_sports_field_perimeter_l325_32507

noncomputable def perimeter_of_sports_field (a b : ℝ) (h1 : a^2 + b^2 = 400) (h2 : a * b = 120) : ℝ :=
  2 * (a + b)

theorem sports_field_perimeter {a b : ℝ} (h1 : a^2 + b^2 = 400) (h2 : a * b = 120) :
  perimeter_of_sports_field a b h1 h2 = 51 := by
  sorry

end NUMINAMATH_GPT_sports_field_perimeter_l325_32507


namespace NUMINAMATH_GPT_first_term_of_arithmetic_sequence_l325_32527

theorem first_term_of_arithmetic_sequence :
  ∃ (a_1 : ℤ), ∀ (d n : ℤ), d = 3 / 4 ∧ n = 30 ∧ a_n = 63 / 4 → a_1 = -6 := by
  sorry

end NUMINAMATH_GPT_first_term_of_arithmetic_sequence_l325_32527


namespace NUMINAMATH_GPT_Mr_Blue_potato_yield_l325_32511

-- Definitions based on the conditions
def steps_length (steps : ℕ) : ℕ := steps * 3
def garden_length : ℕ := steps_length 18
def garden_width : ℕ := steps_length 25

def area_garden : ℕ := garden_length * garden_width
def yield_potatoes (area : ℕ) : ℚ := area * (3/4)

-- Statement of the proof
theorem Mr_Blue_potato_yield :
  yield_potatoes area_garden = 3037.5 := by
  sorry

end NUMINAMATH_GPT_Mr_Blue_potato_yield_l325_32511


namespace NUMINAMATH_GPT_total_pencils_l325_32564

theorem total_pencils (pencils_per_child : ℕ) (children : ℕ) (h1 : pencils_per_child = 2) (h2 : children = 11) : (pencils_per_child * children = 22) := 
by
  sorry

end NUMINAMATH_GPT_total_pencils_l325_32564


namespace NUMINAMATH_GPT_solve_equation_l325_32590

theorem solve_equation (x : ℝ) : (x + 2)^2 - 5 * (x + 2) = 0 ↔ (x = -2 ∨ x = 3) :=
by sorry

end NUMINAMATH_GPT_solve_equation_l325_32590


namespace NUMINAMATH_GPT_hours_per_day_is_8_l325_32569

-- Define the conditions
def hire_two_bodyguards (day_count : ℕ) (total_payment : ℕ) (hourly_rate : ℕ) (daily_hours : ℕ) : Prop :=
  2 * hourly_rate * day_count * daily_hours = total_payment

-- Define the correct answer
theorem hours_per_day_is_8 :
  hire_two_bodyguards 7 2240 20 8 :=
by
  -- Here, you would provide the step-by-step justification, but we use sorry since no proof is required.
  sorry

end NUMINAMATH_GPT_hours_per_day_is_8_l325_32569


namespace NUMINAMATH_GPT_range_of_a_l325_32554

theorem range_of_a (a : ℝ) : 
  (∀ x y z : ℝ, x^2 + y^2 + z^2 = 1 → |a - 1| ≥ x + 2 * y + 2 * z) →
  a ∈ Set.Iic (-2) ∪ Set.Ici 4 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l325_32554


namespace NUMINAMATH_GPT_number_of_pencils_l325_32512

theorem number_of_pencils 
  (P Pe M : ℕ)
  (h1 : Pe = P + 4)
  (h2 : M = P + 20)
  (h3 : P / 5 = Pe / 6)
  (h4 : Pe / 6 = M / 7) : 
  Pe = 24 :=
by
  sorry

end NUMINAMATH_GPT_number_of_pencils_l325_32512


namespace NUMINAMATH_GPT_cricket_bat_weight_l325_32578

-- Define the conditions as Lean definitions
def weight_of_basketball : ℕ := 36
def weight_of_basketballs (n : ℕ) := n * weight_of_basketball
def weight_of_cricket_bats (m : ℕ) := m * (weight_of_basketballs 4 / 8)

-- State the theorem and skip the proof
theorem cricket_bat_weight :
  weight_of_cricket_bats 1 = 18 :=
by
  sorry

end NUMINAMATH_GPT_cricket_bat_weight_l325_32578


namespace NUMINAMATH_GPT_value_of_m_l325_32534

theorem value_of_m : 
  (2 ^ 1999 - 2 ^ 1998 - 2 ^ 1997 + 2 ^ 1996 - 2 ^ 1995 = m * 2 ^ 1995) -> m = 5 :=
by 
  sorry

end NUMINAMATH_GPT_value_of_m_l325_32534


namespace NUMINAMATH_GPT_cosA_sinB_value_l325_32537

theorem cosA_sinB_value (A B : ℝ) (hA1 : 0 < A ∧ A < π / 2) (hB1 : 0 < B ∧ B < π / 2)
  (h_tan_eq : (4 + (Real.tan A)^2) * (5 + (Real.tan B)^2) = Real.sqrt 320 * Real.tan A * Real.tan B) :
  Real.cos A * Real.sin B = Real.sqrt 6 / 6 := sorry

end NUMINAMATH_GPT_cosA_sinB_value_l325_32537


namespace NUMINAMATH_GPT_problem_I_problem_II_1_problem_II_2_l325_32520

section
variables (boys_A girls_A boys_B girls_B : ℕ)
variables (total_students : ℕ)

-- Define the conditions
def conditions : Prop :=
  boys_A = 2 ∧ girls_A = 1 ∧ boys_B = 3 ∧ girls_B = 2 ∧ total_students = boys_A + girls_A + boys_B + girls_B

-- Problem (I)
theorem problem_I (h : conditions boys_A girls_A boys_B girls_B total_students) :
  ∃ arrangements, arrangements = 14400 := sorry

-- Problem (II.1)
theorem problem_II_1 (h : conditions boys_A girls_A boys_B girls_B total_students) :
  ∃ prob, prob = 13 / 14 := sorry

-- Problem (II.2)
theorem problem_II_2 (h : conditions boys_A girls_A boys_B girls_B total_students) :
  ∃ prob, prob = 6 / 35 := sorry
end

end NUMINAMATH_GPT_problem_I_problem_II_1_problem_II_2_l325_32520


namespace NUMINAMATH_GPT_sum_of_n_l325_32591

theorem sum_of_n (n : ℤ) (h : (36 : ℤ) % (2 * n - 1) = 0) :
  (n = 1 ∨ n = 2 ∨ n = 5) → 1 + 2 + 5 = 8 :=
by
  intros hn
  have h1 : n = 1 ∨ n = 2 ∨ n = 5 := hn
  sorry

end NUMINAMATH_GPT_sum_of_n_l325_32591


namespace NUMINAMATH_GPT_bethany_saw_16_portraits_l325_32586

variable (P S : ℕ)

def bethany_conditions : Prop :=
  S = 4 * P ∧ P + S = 80

theorem bethany_saw_16_portraits (P S : ℕ) (h : bethany_conditions P S) : P = 16 := by
  sorry

end NUMINAMATH_GPT_bethany_saw_16_portraits_l325_32586


namespace NUMINAMATH_GPT_quadratic_equation_roots_sum_and_difference_l325_32561

theorem quadratic_equation_roots_sum_and_difference :
  ∃ (p q : ℝ), 
    p + q = 7 ∧ 
    |p - q| = 9 ∧ 
    (∀ x, (x - p) * (x - q) = x^2 - 7 * x - 8) :=
sorry

end NUMINAMATH_GPT_quadratic_equation_roots_sum_and_difference_l325_32561


namespace NUMINAMATH_GPT_minimum_value_l325_32552

noncomputable def condition (x : ℝ) : Prop := (2 * x - 1) / 3 - 1 ≥ x - (5 - 3 * x) / 2

noncomputable def target_function (x : ℝ) : ℝ := abs (x - 1) - abs (x + 3)

theorem minimum_value :
  ∃ x : ℝ, condition x ∧ ∀ y : ℝ, condition y → target_function y ≥ target_function x :=
sorry

end NUMINAMATH_GPT_minimum_value_l325_32552


namespace NUMINAMATH_GPT_contrapositive_of_implication_l325_32503

theorem contrapositive_of_implication (p q : Prop) (h : p → q) : ¬q → ¬p :=
by {
  sorry
}

end NUMINAMATH_GPT_contrapositive_of_implication_l325_32503


namespace NUMINAMATH_GPT_neg_four_is_square_root_of_sixteen_l325_32515

/-
  Definitions:
  - A number y is a square root of x if y^2 = x.
  - A number y is an arithmetic square root of x if y ≥ 0 and y^2 = x.
-/

theorem neg_four_is_square_root_of_sixteen :
  -4 * -4 = 16 := 
by
  -- proof step is omitted
  sorry

end NUMINAMATH_GPT_neg_four_is_square_root_of_sixteen_l325_32515


namespace NUMINAMATH_GPT_total_amount_due_l325_32525

noncomputable def original_bill : ℝ := 500
noncomputable def late_charge_rate : ℝ := 0.02
noncomputable def annual_interest_rate : ℝ := 0.05

theorem total_amount_due (n : ℕ) (initial_amount : ℝ) (late_charge_rate : ℝ) (interest_rate : ℝ) : 
  initial_amount = 500 → 
  late_charge_rate = 0.02 → 
  interest_rate = 0.05 → 
  n = 3 → 
  (initial_amount * (1 + late_charge_rate)^n * (1 + interest_rate) = 557.13) :=
by
  intros h_initial_amount h_late_charge_rate h_interest_rate h_n
  sorry

end NUMINAMATH_GPT_total_amount_due_l325_32525


namespace NUMINAMATH_GPT_locus_of_vertices_l325_32599

theorem locus_of_vertices (t : ℝ) (x y : ℝ) (h : y = x^2 + t * x + 1) : y = 1 - x^2 :=
by
  sorry

end NUMINAMATH_GPT_locus_of_vertices_l325_32599


namespace NUMINAMATH_GPT_toilet_paper_production_per_day_l325_32542

theorem toilet_paper_production_per_day 
    (total_production_march : ℕ)
    (days_in_march : ℕ)
    (increase_factor : ℕ)
    (total_production : ℕ)
    (days : ℕ)
    (increase : ℕ)
    (production : ℕ) :
    total_production_march = total_production →
    days_in_march = days →
    increase_factor = increase →
    total_production = 868000 →
    days = 31 →
    increase = 3 →
    production = total_production / days →
    production / increase = 9333
:= by
  intros h1 h2 h3 h4 h5 h6 h7

  sorry

end NUMINAMATH_GPT_toilet_paper_production_per_day_l325_32542


namespace NUMINAMATH_GPT_circle_equation_l325_32541

-- Define the conditions
def chord_length_condition (a b r : ℝ) : Prop := r^2 = a^2 + 1
def arc_length_condition (b r : ℝ) : Prop := r^2 = 2 * b^2
def min_distance_condition (a b : ℝ) : Prop := a = b

-- The main theorem stating the final answer
theorem circle_equation (a b r : ℝ) (h1 : chord_length_condition a b r)
    (h2 : arc_length_condition b r) (h3 : min_distance_condition a b) :
    ((x - a)^2 + (y - a)^2 = 2) ∨ ((x + a)^2 + (y + a)^2 = 2) :=
sorry

end NUMINAMATH_GPT_circle_equation_l325_32541


namespace NUMINAMATH_GPT_radius_of_circle_zero_l325_32592

theorem radius_of_circle_zero :
  (∃ x y : ℝ, x^2 + 8 * x + y^2 - 10 * y + 41 = 0) →
  (0 : ℝ) = 0 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_radius_of_circle_zero_l325_32592


namespace NUMINAMATH_GPT_bears_per_shelf_l325_32562

def bears_initial : ℕ := 6

def shipment : ℕ := 18

def shelves : ℕ := 4

theorem bears_per_shelf : (bears_initial + shipment) / shelves = 6 := by
  sorry

end NUMINAMATH_GPT_bears_per_shelf_l325_32562


namespace NUMINAMATH_GPT_total_books_to_put_away_l325_32516

-- Definitions based on the conditions
def books_per_shelf := 4
def shelves_needed := 3

-- The proof problem translates to finding the total number of books
theorem total_books_to_put_away : shelves_needed * books_per_shelf = 12 := by
  sorry

end NUMINAMATH_GPT_total_books_to_put_away_l325_32516


namespace NUMINAMATH_GPT_max_a_correct_answers_l325_32577

theorem max_a_correct_answers : 
  ∃ (a b c x y z w : ℕ), 
  a + b + c + x + y + z + w = 39 ∧
  a = b + c ∧
  (a + x + y + w) = a + 5 + (x + y + w) ∧
  b + z = 2 * (c + z) ∧
  23 ≤ a :=
sorry

end NUMINAMATH_GPT_max_a_correct_answers_l325_32577


namespace NUMINAMATH_GPT_problem_statement_l325_32543

-- Given that f(x) is an even function.
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Definition of the main condition f(x) + f(2 - x) = 0.
def special_condition (f : ℝ → ℝ) : Prop := ∀ x, f x + f (2 - x) = 0

-- Theorem: Given the conditions, show that f(x) has a period of 4 and f(x-1) is odd.
theorem problem_statement {f : ℝ → ℝ} (h_even : is_even f) (h_cond : special_condition f) :
  (∀ x, f (4 + x) = f x) ∧ (∀ x, f (-x - 1) = -f (x - 1)) :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l325_32543


namespace NUMINAMATH_GPT_find_breadth_of_landscape_l325_32548

theorem find_breadth_of_landscape (L B A : ℕ) 
  (h1 : B = 8 * L)
  (h2 : 3200 = A / 9)
  (h3 : 3200 * 9 = A) :
  B = 480 :=
by 
  sorry

end NUMINAMATH_GPT_find_breadth_of_landscape_l325_32548


namespace NUMINAMATH_GPT_greatest_sum_l325_32588

-- stating the conditions
def condition1 (x y : ℝ) := x^2 + y^2 = 130
def condition2 (x y : ℝ) := x * y = 45

-- proving the result
theorem greatest_sum (x y : ℝ) 
  (h1 : condition1 x y) 
  (h2 : condition2 x y) : 
  x + y = 10 * Real.sqrt 2.2 :=
sorry

end NUMINAMATH_GPT_greatest_sum_l325_32588


namespace NUMINAMATH_GPT_find_s_l325_32575

def f (x s : ℝ) : ℝ := 3 * x^3 - 2 * x^2 + 4 * x + s

theorem find_s (s : ℝ) : f (-1) s = 0 → s = 9 :=
by
  sorry

end NUMINAMATH_GPT_find_s_l325_32575


namespace NUMINAMATH_GPT_pencil_cost_l325_32560

theorem pencil_cost (P : ℝ) (h1 : 24 * P + 18 = 30) : P = 0.5 :=
by
  sorry

end NUMINAMATH_GPT_pencil_cost_l325_32560


namespace NUMINAMATH_GPT_radius_of_cookie_l325_32576

theorem radius_of_cookie : 
  ∀ x y : ℝ, (x^2 + y^2 - 6.5 = x + 3 * y) → 
  ∃ (c : ℝ × ℝ) (r : ℝ), r = 3 ∧ (x - c.1)^2 + (y - c.2)^2 = r^2 :=
by {
  sorry
}

end NUMINAMATH_GPT_radius_of_cookie_l325_32576


namespace NUMINAMATH_GPT_charlie_age_l325_32538

variable (J C B : ℝ)

def problem_statement :=
  J = C + 12 ∧ C = B + 7 ∧ J = 3 * B → C = 18

theorem charlie_age : problem_statement J C B :=
by
  sorry

end NUMINAMATH_GPT_charlie_age_l325_32538


namespace NUMINAMATH_GPT_journey_distance_l325_32579

theorem journey_distance (D : ℝ) (h1 : (D / 40) + (D / 60) = 40) : D = 960 :=
by
  sorry

end NUMINAMATH_GPT_journey_distance_l325_32579


namespace NUMINAMATH_GPT_tile_rectangle_condition_l325_32513

theorem tile_rectangle_condition (k m n : ℕ) (hk : 0 < k) (hm : 0 < m) (hn : 0 < n) : 
  (∃ q, m = k * q) ∨ (∃ r, n = k * r) :=
sorry

end NUMINAMATH_GPT_tile_rectangle_condition_l325_32513


namespace NUMINAMATH_GPT_missing_files_correct_l325_32582

def total_files : ℕ := 60
def files_in_morning : ℕ := total_files / 2
def files_in_afternoon : ℕ := 15
def missing_files : ℕ := total_files - (files_in_morning + files_in_afternoon)

theorem missing_files_correct : missing_files = 15 := by
  sorry

end NUMINAMATH_GPT_missing_files_correct_l325_32582


namespace NUMINAMATH_GPT_opposite_of_8_is_neg_8_l325_32532

theorem opposite_of_8_is_neg_8 : - (8 : ℤ) = -8 :=
by
  sorry

end NUMINAMATH_GPT_opposite_of_8_is_neg_8_l325_32532


namespace NUMINAMATH_GPT_cups_of_oil_used_l325_32517

-- Define the required amounts
def total_liquid : ℝ := 1.33
def water_used : ℝ := 1.17

-- The statement we want to prove
theorem cups_of_oil_used : total_liquid - water_used = 0.16 := by
sorry

end NUMINAMATH_GPT_cups_of_oil_used_l325_32517


namespace NUMINAMATH_GPT_subset_condition_for_a_l325_32546

theorem subset_condition_for_a (a : ℝ) : 
  (∀ x y : ℝ, (x - 1)^2 + (y - 2)^2 ≤ 5 / 4 → (|x - 1| + 2 * |y - 2| ≤ a)) → a ≥ 5 / 2 :=
by
  intro H
  sorry

end NUMINAMATH_GPT_subset_condition_for_a_l325_32546


namespace NUMINAMATH_GPT_red_knights_fraction_magic_l325_32570

theorem red_knights_fraction_magic (total_knights red_knights blue_knights magical_knights : ℕ)
  (h1 : red_knights = (3 / 8 : ℚ) * total_knights)
  (h2 : blue_knights = total_knights - red_knights)
  (h3 : magical_knights = (1 / 4 : ℚ) * total_knights)
  (fraction_red_magic fraction_blue_magic : ℚ) 
  (h4 : fraction_red_magic = 3 * fraction_blue_magic)
  (h5 : magical_knights = red_knights * fraction_red_magic + blue_knights * fraction_blue_magic) :
  fraction_red_magic = 3 / 7 := 
by
  sorry

end NUMINAMATH_GPT_red_knights_fraction_magic_l325_32570


namespace NUMINAMATH_GPT_pasture_rent_share_l325_32574

theorem pasture_rent_share (x : ℕ) (H1 : (45 / (10 * x + 60 + 45)) * 245 = 63) : 
  x = 7 :=
by {
  sorry
}

end NUMINAMATH_GPT_pasture_rent_share_l325_32574


namespace NUMINAMATH_GPT_stratified_sampling_junior_teachers_l325_32567

theorem stratified_sampling_junior_teachers 
    (total_teachers : ℕ) (senior_teachers : ℕ) 
    (intermediate_teachers : ℕ) (junior_teachers : ℕ) 
    (sample_size : ℕ) 
    (H1 : total_teachers = 200)
    (H2 : senior_teachers = 20)
    (H3 : intermediate_teachers = 100)
    (H4 : junior_teachers = 80) 
    (H5 : sample_size = 50)
    : (junior_teachers * sample_size / total_teachers = 20) := 
  by 
    sorry

end NUMINAMATH_GPT_stratified_sampling_junior_teachers_l325_32567


namespace NUMINAMATH_GPT_chords_from_nine_points_l325_32544

theorem chords_from_nine_points : 
  ∀ (n r : ℕ), n = 9 → r = 2 → (Nat.choose n r) = 36 :=
by
  intros n r hn hr
  rw [hn, hr]
  -- Goal: Nat.choose 9 2 = 36
  sorry

end NUMINAMATH_GPT_chords_from_nine_points_l325_32544


namespace NUMINAMATH_GPT_negation_of_p_l325_32555

open Classical

-- Define proposition p
def p : Prop := ∀ x : ℝ, x^2 + x > 2

-- Define the negation of proposition p
def not_p : Prop := ∃ x : ℝ, x^2 + x ≤ 2

theorem negation_of_p : ¬p ↔ not_p :=
by sorry

end NUMINAMATH_GPT_negation_of_p_l325_32555


namespace NUMINAMATH_GPT_tulip_to_remaining_ratio_l325_32510

theorem tulip_to_remaining_ratio (total_flowers daisies sunflowers tulips remaining_tulips remaining_flowers : ℕ) 
  (h1 : total_flowers = 12) 
  (h2 : daisies = 2) 
  (h3 : sunflowers = 4) 
  (h4 : tulips = total_flowers - (daisies + sunflowers))
  (h5 : remaining_tulips = tulips)
  (h6 : remaining_flowers = remaining_tulips + sunflowers)
  (h7 : remaining_flowers = 10) : 
  tulips / remaining_flowers = 3 / 5 := 
by
  sorry

end NUMINAMATH_GPT_tulip_to_remaining_ratio_l325_32510


namespace NUMINAMATH_GPT_rectangle_area_l325_32505

theorem rectangle_area (d : ℝ) (w : ℝ) (h : w^2 + (3*w)^2 = d^2) : (3 * w ^ 2 = 3 * d ^ 2 / 10) :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_l325_32505


namespace NUMINAMATH_GPT_roots_of_equation_l325_32529

theorem roots_of_equation (x : ℝ) : (x - 3) ^ 2 = 4 ↔ (x = 5 ∨ x = 1) := by
  sorry

end NUMINAMATH_GPT_roots_of_equation_l325_32529


namespace NUMINAMATH_GPT_garden_enlargement_l325_32530

theorem garden_enlargement :
  let length := 60
  let width := 20
  let perimeter := 2 * (length + width)
  let side_square := perimeter / 4
  let area_rectangular := length * width
  let area_square := side_square * side_square
  area_square - area_rectangular = 400 := by
  -- initializing all definitions
  let length := 60
  let width := 20
  let perimeter := 2 * (length + width)
  let side_square := perimeter / 4
  let area_rectangular := length * width
  let area_square := side_square * side_square
  -- placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_garden_enlargement_l325_32530


namespace NUMINAMATH_GPT_alex_ride_time_l325_32559

theorem alex_ride_time
  (T : ℝ) -- time on flat ground
  (flat_speed : ℝ := 20) -- flat ground speed
  (uphill_speed : ℝ := 12) -- uphill speed
  (uphill_time : ℝ := 2.5) -- uphill time
  (downhill_speed : ℝ := 24) -- downhill speed
  (downhill_time : ℝ := 1.5) -- downhill time
  (walk_distance : ℝ := 8) -- distance walked
  (total_distance : ℝ := 164) -- total distance to the town
  (hup : uphill_speed * uphill_time = 30)
  (hdown : downhill_speed * downhill_time = 36)
  (hwalk : walk_distance = 8) :
  flat_speed * T + 30 + 36 + 8 = total_distance → T = 4.5 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_alex_ride_time_l325_32559


namespace NUMINAMATH_GPT_find_m_parallel_l325_32595

noncomputable def is_parallel (A1 B1 C1 A2 B2 C2 : ℝ) : Prop :=
  -(A1 / B1) = -(A2 / B2)

theorem find_m_parallel : ∃ m : ℝ, is_parallel (m-1) 3 m 1 (m+1) 2 ∧ m = -2 :=
by
  unfold is_parallel
  exists (-2 : ℝ)
  sorry

end NUMINAMATH_GPT_find_m_parallel_l325_32595


namespace NUMINAMATH_GPT_parallelogram_area_increase_l325_32550
open Real

/-- The area of the parallelogram increases by 600 square meters when the base is increased by 20 meters. -/
theorem parallelogram_area_increase :
  ∀ (base height new_base : ℝ), 
    base = 65 → height = 30 → new_base = base + 20 → 
    (new_base * height - base * height) = 600 := 
by
  sorry

end NUMINAMATH_GPT_parallelogram_area_increase_l325_32550


namespace NUMINAMATH_GPT_bicycle_cost_after_tax_l325_32502

theorem bicycle_cost_after_tax :
  let original_price := 300
  let first_discount := original_price * 0.40
  let price_after_first_discount := original_price - first_discount
  let second_discount := price_after_first_discount * 0.20
  let price_after_second_discount := price_after_first_discount - second_discount
  let tax := price_after_second_discount * 0.05
  price_after_second_discount + tax = 151.20 :=
by
  sorry

end NUMINAMATH_GPT_bicycle_cost_after_tax_l325_32502


namespace NUMINAMATH_GPT_contrapositive_l325_32504

theorem contrapositive (a b : ℝ) : (a ≠ 0 ∨ b ≠ 0) → a^2 + b^2 ≠ 0 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_contrapositive_l325_32504


namespace NUMINAMATH_GPT_cori_age_l325_32597

theorem cori_age (C A : ℕ) (hA : A = 19) (hEq : C + 5 = (A + 5) / 3) : C = 3 := by
  rw [hA] at hEq
  norm_num at hEq
  linarith

end NUMINAMATH_GPT_cori_age_l325_32597


namespace NUMINAMATH_GPT_geometric_sequence_formula_l325_32549

noncomputable def a_n (a_1 : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a_1 * q^n

theorem geometric_sequence_formula
  (a_1 q : ℝ)
  (h_pos : ∀ n : ℕ, a_n a_1 q n > 0)
  (h_4_eq : a_n a_1 q 4 = (a_n a_1 q 2)^2)
  (h_2_4_sum : a_n a_1 q 2 + a_n a_1 q 4 = 5 / 16) :
  ∀ n : ℕ, a_n a_1 q n = ((1 : ℝ) / 2) ^ n :=
sorry

end NUMINAMATH_GPT_geometric_sequence_formula_l325_32549


namespace NUMINAMATH_GPT_three_two_three_zero_zero_zero_zero_in_scientific_notation_l325_32571

theorem three_two_three_zero_zero_zero_zero_in_scientific_notation :
  3230000 = 3.23 * 10^6 :=
sorry

end NUMINAMATH_GPT_three_two_three_zero_zero_zero_zero_in_scientific_notation_l325_32571


namespace NUMINAMATH_GPT_constant_term_value_l325_32587

theorem constant_term_value :
  ∀ (x y z k : ℤ), (4 * x + y + z = 80) → (2 * x - y - z = 40) → (x = 20) → (3 * x + y - z = k) → (k = 60) :=
by 
  intros x y z k h₁ h₂ hx h₃
  sorry

end NUMINAMATH_GPT_constant_term_value_l325_32587


namespace NUMINAMATH_GPT_pure_imaginary_number_l325_32598

open Complex -- Use the Complex module for complex numbers

theorem pure_imaginary_number (a : ℝ) (h : (a - 1 : ℂ).re = 0) : a = 1 :=
by
  -- This part of the proof is omitted hence we put sorry
  sorry

end NUMINAMATH_GPT_pure_imaginary_number_l325_32598


namespace NUMINAMATH_GPT_omitted_digits_correct_l325_32522

theorem omitted_digits_correct :
  (287 * 23 = 6601) := by
  sorry

end NUMINAMATH_GPT_omitted_digits_correct_l325_32522


namespace NUMINAMATH_GPT_polynomial_divisibility_l325_32583

-- Definitions
def f (k l m n : ℕ) (x : ℂ) : ℂ :=
  x^(4 * k) + x^(4 * l + 1) + x^(4 * m + 2) + x^(4 * n + 3)

def g (x : ℂ) : ℂ :=
  x^3 + x^2 + x + 1

-- Theorem statement
theorem polynomial_divisibility (k l m n : ℕ) : ∀ x : ℂ, g x ∣ f k l m n x :=
  sorry

end NUMINAMATH_GPT_polynomial_divisibility_l325_32583


namespace NUMINAMATH_GPT_estimate_ratio_l325_32551

theorem estimate_ratio (A B : ℕ) (A_def : A = 1 * 2 * 7 + 2 * 4 * 14 + 3 * 6 * 21 + 4 * 8 * 28)
  (B_def : B = 1 * 3 * 5 + 2 * 6 * 10 + 3 * 9 * 15 + 4 * 12 * 20) : 0 < A / B ∧ A / B < 1 := by
  sorry

end NUMINAMATH_GPT_estimate_ratio_l325_32551


namespace NUMINAMATH_GPT_evaluate_five_iterates_of_f_at_one_l325_32514

def f (x : ℕ) : ℕ :=
if x % 2 = 0 then x / 2 else 5 * x + 1

theorem evaluate_five_iterates_of_f_at_one :
  f (f (f (f (f 1)))) = 4 := by
  sorry

end NUMINAMATH_GPT_evaluate_five_iterates_of_f_at_one_l325_32514


namespace NUMINAMATH_GPT_number_of_factors_of_60_l325_32584

theorem number_of_factors_of_60 : 
  ∃ n, n = 12 ∧ 
  (∀ p k : ℕ, p ∈ [2, 3, 5] → 60 = 2^2 * 3^1 * 5^1 → (∃ d : ℕ, d = (2 + 1) * (1 + 1) * (1 + 1) ∧ n = d)) :=
by sorry

end NUMINAMATH_GPT_number_of_factors_of_60_l325_32584


namespace NUMINAMATH_GPT_angle_EHG_65_l325_32563

/-- Quadrilateral $EFGH$ has $EF = FG = GH$, $\angle EFG = 80^\circ$, and $\angle FGH = 150^\circ$; and hence the degree measure of $\angle EHG$ is $65^\circ$. -/
theorem angle_EHG_65 {EF FG GH : ℝ} (h1 : EF = FG) (h2 : FG = GH) 
  (EFG : ℝ) (FGH : ℝ) (h3 : EFG = 80) (h4 : FGH = 150) : 
  ∃ EHG : ℝ, EHG = 65 :=
by
  sorry

end NUMINAMATH_GPT_angle_EHG_65_l325_32563


namespace NUMINAMATH_GPT_polynomial_power_degree_l325_32556

noncomputable def polynomial_degree (p : Polynomial ℝ) : ℕ := p.natDegree

theorem polynomial_power_degree : 
  polynomial_degree ((5 * X^3 - 4 * X + 7)^10) = 30 := by
  sorry

end NUMINAMATH_GPT_polynomial_power_degree_l325_32556


namespace NUMINAMATH_GPT_find_f2_l325_32565

-- Define the conditions
variable {f g : ℝ → ℝ} {a : ℝ}

-- Assume f is an odd function
axiom odd_f : ∀ x : ℝ, f (-x) = -f x

-- Assume g is an even function
axiom even_g : ∀ x : ℝ, g (-x) = g x

-- Condition given in the problem
axiom f_g_relation : ∀ x : ℝ, f x + g x = a^x - a^(-x) + 2

-- Condition that g(2) = a
axiom g_at_2 : g 2 = a

-- Condition for a
axiom a_cond : a > 0 ∧ a ≠ 1

-- Proof problem
theorem find_f2 : f 2 = 15 / 4 := by
  sorry

end NUMINAMATH_GPT_find_f2_l325_32565
