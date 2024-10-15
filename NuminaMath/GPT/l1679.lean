import Mathlib

namespace NUMINAMATH_GPT_find_m_and_n_l1679_167982

theorem find_m_and_n (a b c d m n : ℕ) (h1 : a^2 + b^2 + c^2 + d^2 = 1989) 
                    (h2 : a + b + c + d = m^2) 
                    (h3 : max a (max b (max c d)) = n^2) : 
                    m = 9 ∧ n = 6 := 
sorry

end NUMINAMATH_GPT_find_m_and_n_l1679_167982


namespace NUMINAMATH_GPT_number_is_16_l1679_167966

theorem number_is_16 (n : ℝ) (h : (1/2) * n + 5 = 13) : n = 16 :=
sorry

end NUMINAMATH_GPT_number_is_16_l1679_167966


namespace NUMINAMATH_GPT_value_of_f_neg_a_l1679_167993

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x + x^3 + 1

theorem value_of_f_neg_a (a : ℝ) (h : f a = 3) : f (-a) = -1 := 
by
  sorry

end NUMINAMATH_GPT_value_of_f_neg_a_l1679_167993


namespace NUMINAMATH_GPT_pos_int_solutions_l1679_167948

-- defining the condition for a positive integer solution to the equation
def is_pos_int_solution (x y : Int) : Prop :=
  5 * x + 2 * y = 25 ∧ x > 0 ∧ y > 0

-- stating the theorem for positive integer solutions of the equation
theorem pos_int_solutions : 
  ∃ x y : Int, is_pos_int_solution x y ∧ ((x = 1 ∧ y = 10) ∨ (x = 3 ∧ y = 5)) :=
by
  sorry

end NUMINAMATH_GPT_pos_int_solutions_l1679_167948


namespace NUMINAMATH_GPT_volume_of_hall_l1679_167978

-- Define the dimensions and areas conditions
def length_hall : ℝ := 15
def breadth_hall : ℝ := 12
def area_floor_ceiling : ℝ := 2 * (length_hall * breadth_hall)
def area_walls (h : ℝ) : ℝ := 2 * (length_hall * h) + 2 * (breadth_hall * h)

-- Given condition: The sum of the areas of the floor and ceiling is equal to the sum of the areas of the four walls
def condition (h : ℝ) : Prop := area_floor_ceiling = area_walls h

-- Define the volume of the hall
def volume_hall (h : ℝ) : ℝ := length_hall * breadth_hall * h

-- The theorem to be proven: given the condition, the volume equals 8004
theorem volume_of_hall : ∃ h : ℝ, condition h ∧ volume_hall h = 8004 := by
  sorry

end NUMINAMATH_GPT_volume_of_hall_l1679_167978


namespace NUMINAMATH_GPT_total_rainfall_2004_l1679_167977

theorem total_rainfall_2004 (avg_2003 : ℝ) (increment : ℝ) (months : ℕ) (total_2004 : ℝ) 
  (h1 : avg_2003 = 41.5) 
  (h2 : increment = 2) 
  (h3 : months = 12) 
  (h4 : total_2004 = avg_2003 + increment * months) :
  total_2004 = 522 :=
by 
  sorry

end NUMINAMATH_GPT_total_rainfall_2004_l1679_167977


namespace NUMINAMATH_GPT_JillTotalTaxPercentage_l1679_167955

noncomputable def totalTaxPercentage : ℝ :=
  let totalSpending (beforeDiscount : ℝ) : ℝ := 100
  let clothingBeforeDiscount : ℝ := 0.4 * totalSpending 100
  let foodBeforeDiscount : ℝ := 0.2 * totalSpending 100
  let electronicsBeforeDiscount : ℝ := 0.1 * totalSpending 100
  let cosmeticsBeforeDiscount : ℝ := 0.2 * totalSpending 100
  let householdBeforeDiscount : ℝ := 0.1 * totalSpending 100

  let clothingDiscount : ℝ := 0.1 * clothingBeforeDiscount
  let foodDiscount : ℝ := 0.05 * foodBeforeDiscount
  let electronicsDiscount : ℝ := 0.15 * electronicsBeforeDiscount

  let clothingAfterDiscount := clothingBeforeDiscount - clothingDiscount
  let foodAfterDiscount := foodBeforeDiscount - foodDiscount
  let electronicsAfterDiscount := electronicsBeforeDiscount - electronicsDiscount
  
  let taxOnClothing := 0.06 * clothingAfterDiscount
  let taxOnFood := 0.0 * foodAfterDiscount
  let taxOnElectronics := 0.1 * electronicsAfterDiscount
  let taxOnCosmetics := 0.08 * cosmeticsBeforeDiscount
  let taxOnHousehold := 0.04 * householdBeforeDiscount

  let totalTaxPaid := taxOnClothing + taxOnFood + taxOnElectronics + taxOnCosmetics + taxOnHousehold
  (totalTaxPaid / totalSpending 100) * 100

theorem JillTotalTaxPercentage :
  totalTaxPercentage = 5.01 := by
  sorry

end NUMINAMATH_GPT_JillTotalTaxPercentage_l1679_167955


namespace NUMINAMATH_GPT_cubic_inches_in_two_cubic_feet_l1679_167986

-- Define the conversion factor between feet and inches
def foot_to_inch : ℕ := 12
-- Define the conversion factor between cubic feet and cubic inches
def cubic_foot_to_cubic_inch : ℕ := foot_to_inch ^ 3

-- State the theorem to be proved
theorem cubic_inches_in_two_cubic_feet : 2 * cubic_foot_to_cubic_inch = 3456 :=
by
  -- Proof steps go here
  sorry

end NUMINAMATH_GPT_cubic_inches_in_two_cubic_feet_l1679_167986


namespace NUMINAMATH_GPT_c_horses_months_l1679_167983

theorem c_horses_months (cost_total Rs_a Rs_b num_horses_a num_months_a num_horses_b num_months_b num_horses_c amount_paid_b : ℕ) (x : ℕ) 
  (h1 : cost_total = 841) 
  (h2 : Rs_a = 12 * 8)
  (h3 : Rs_b = 16 * 9)
  (h4 : amount_paid_b = 348)
  (h5 : 96 * (amount_paid_b / Rs_b) + (18 * x) * (amount_paid_b / Rs_b) = cost_total - amount_paid_b) :
  x = 11 :=
sorry

end NUMINAMATH_GPT_c_horses_months_l1679_167983


namespace NUMINAMATH_GPT_distance_to_Rock_Mist_Mountains_l1679_167961

theorem distance_to_Rock_Mist_Mountains
  (d_Sky_Falls : ℕ) (d_Sky_Falls_eq : d_Sky_Falls = 8)
  (d_Rock_Mist : ℕ) (d_Rock_Mist_eq : d_Rock_Mist = 50 * d_Sky_Falls)
  (detour_Thunder_Pass : ℕ) (detour_Thunder_Pass_eq : detour_Thunder_Pass = 25) :
  d_Rock_Mist + detour_Thunder_Pass = 425 := by
  sorry

end NUMINAMATH_GPT_distance_to_Rock_Mist_Mountains_l1679_167961


namespace NUMINAMATH_GPT_fg_of_3_l1679_167974

def f (x : ℕ) : ℕ := x * x
def g (x : ℕ) : ℕ := x + 2

theorem fg_of_3 : f (g 3) = 25 := by
  sorry

end NUMINAMATH_GPT_fg_of_3_l1679_167974


namespace NUMINAMATH_GPT_value_of_a_star_b_l1679_167923

variable (a b : ℤ)

def operation_star (a b : ℤ) : ℚ :=
  1 / a + 1 / b

theorem value_of_a_star_b (h1 : a + b = 7) (h2 : a * b = 12) :
  operation_star a b = 7 / 12 := by
  sorry

end NUMINAMATH_GPT_value_of_a_star_b_l1679_167923


namespace NUMINAMATH_GPT_pete_flag_total_circle_square_l1679_167933

theorem pete_flag_total_circle_square : 
  let stars := 50
  let stripes := 13
  let circles := (stars / 2) - 3
  let squares := (stripes * 2) + 6
  circles + squares = 54 := 
by
  let stars := 50
  let stripes := 13
  let circles := (stars / 2) - 3
  let squares := (stripes * 2) + 6
  show circles + squares = 54
  sorry

end NUMINAMATH_GPT_pete_flag_total_circle_square_l1679_167933


namespace NUMINAMATH_GPT_new_quadratic_equation_has_square_roots_l1679_167975

theorem new_quadratic_equation_has_square_roots (p q : ℝ) (x : ℝ) :
  (x^2 + px + q = 0 → ∃ x1 x2 : ℝ, x^2 - (p^2 - 2 * q) * x + q^2 = 0 ∧ (x1^2 = x ∨ x2^2 = x)) :=
by sorry

end NUMINAMATH_GPT_new_quadratic_equation_has_square_roots_l1679_167975


namespace NUMINAMATH_GPT_find_c_l1679_167942

open Real

theorem find_c (a b c d : ℕ) (M : ℝ) (h1 : a > 1) (h2 : b > 1) (h3 : c > 1) (h4 : d > 1) (hM : M ≠ 1) :
  (M ^ (1 / a) * (M ^ (1 / b) * (M ^ (1 / c) * (M ^ (1 / d))))) ^ (1 / a * b * c * d) = (M ^ 37) ^ (1 / 48) →
  c = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_c_l1679_167942


namespace NUMINAMATH_GPT_sum_of_pqrstu_eq_22_l1679_167969

theorem sum_of_pqrstu_eq_22 (p q r s t : ℤ) 
  (h : (7 - p) * (7 - q) * (7 - r) * (7 - s) * (7 - t) = -48) : 
  p + q + r + s + t = 22 :=
sorry

end NUMINAMATH_GPT_sum_of_pqrstu_eq_22_l1679_167969


namespace NUMINAMATH_GPT_special_hash_value_l1679_167939

def special_hash (a b c d : ℝ) : ℝ :=
  d * b ^ 2 - 4 * a * c

theorem special_hash_value :
  special_hash 2 3 1 (1 / 2) = -3.5 :=
by
  -- Note: Insert proof here
  sorry

end NUMINAMATH_GPT_special_hash_value_l1679_167939


namespace NUMINAMATH_GPT_boat_speed_in_still_water_l1679_167901

theorem boat_speed_in_still_water (b s : ℝ) (h1 : b + s = 11) (h2 : b - s = 5) : b = 8 :=
by 
  sorry

end NUMINAMATH_GPT_boat_speed_in_still_water_l1679_167901


namespace NUMINAMATH_GPT_find_sum_of_squares_l1679_167906

theorem find_sum_of_squares (a b c m : ℤ) (h1 : a + b + c = 0) (h2 : a * b + b * c + a * c = -2023) (h3 : a * b * c = -m) : a^2 + b^2 + c^2 = 4046 := by
  sorry

end NUMINAMATH_GPT_find_sum_of_squares_l1679_167906


namespace NUMINAMATH_GPT_calc1_calc2_calc3_l1679_167976

theorem calc1 : -4 - 4 = -8 := by
  sorry

theorem calc2 : (-32) / 4 = -8 := by
  sorry

theorem calc3 : -(-2)^3 = 8 := by
  sorry

end NUMINAMATH_GPT_calc1_calc2_calc3_l1679_167976


namespace NUMINAMATH_GPT_average_percentage_reduction_l1679_167941

theorem average_percentage_reduction (x : ℝ) (hx : 0 < x ∧ x < 1)
  (initial_price final_price : ℝ)
  (h_initial : initial_price = 25)
  (h_final : final_price = 16)
  (h_reduction : final_price = initial_price * (1-x)^2) :
  x = 0.2 :=
by {
  --". Convert fraction \( = x / y \)", proof is omitted
  sorry
}

end NUMINAMATH_GPT_average_percentage_reduction_l1679_167941


namespace NUMINAMATH_GPT_english_only_students_l1679_167963

theorem english_only_students (T B G_total : ℕ) (hT : T = 40) (hB : B = 12) (hG_total : G_total = 22) :
  (T - (G_total - B) - B) = 18 :=
by {
  -- sorry is used to skip the proof
  sorry
}

end NUMINAMATH_GPT_english_only_students_l1679_167963


namespace NUMINAMATH_GPT_find_original_cost_price_l1679_167956

variables (P : ℝ) (A B C D E : ℝ)

-- Define the conditions as per the problem statement
def with_tax (P : ℝ) : ℝ := P * 1.10
def profit_60 (price : ℝ) : ℝ := price * 1.60
def profit_25 (price : ℝ) : ℝ := price * 1.25
def loss_15 (price : ℝ) : ℝ := price * 0.85
def profit_30 (price : ℝ) : ℝ := price * 1.30

-- The final price E is given.
def final_price (P : ℝ) : ℝ :=
  profit_30 
  (loss_15 
  (profit_25 
  (profit_60 
  (with_tax P))))

-- To find original cost price P given final price of Rs. 500.
theorem find_original_cost_price (h : final_price P = 500) : 
  P = 500 / 2.431 :=
by 
  sorry

end NUMINAMATH_GPT_find_original_cost_price_l1679_167956


namespace NUMINAMATH_GPT_express_train_leaves_6_hours_later_l1679_167988

theorem express_train_leaves_6_hours_later
  (V_g V_e : ℕ) (t : ℕ) (catch_up_time : ℕ)
  (goods_train_speed : V_g = 36)
  (express_train_speed : V_e = 90)
  (catch_up_in_4_hours : catch_up_time = 4)
  (distance_e : V_e * catch_up_time = 360)
  (distance_g : V_g * (t + catch_up_time) = 360) :
  t = 6 := by
  sorry

end NUMINAMATH_GPT_express_train_leaves_6_hours_later_l1679_167988


namespace NUMINAMATH_GPT_negation_of_statement_l1679_167903

theorem negation_of_statement :
  ¬ (∃ x_0 : ℝ, x_0^2 + 2 * x_0 + 2 ≤ 0) ↔ ∀ x : ℝ, x^2 + 2 * x + 2 > 0 := by
  sorry

end NUMINAMATH_GPT_negation_of_statement_l1679_167903


namespace NUMINAMATH_GPT_solve_inequality_l1679_167960

noncomputable def satisfies_inequality (x : ℝ) : Prop :=
  (x ^ 3 - 3 * x ^ 2 + 2 * x) / (x ^ 2 - 3 * x + 2) ≤ 0 ∧
  x ≠ 1 ∧ x ≠ 2

theorem solve_inequality :
  {x : ℝ | satisfies_inequality x} = {x : ℝ | x ≤ 0 ∧ x ≠ 1 ∧ x ≠ 2} :=
  sorry

end NUMINAMATH_GPT_solve_inequality_l1679_167960


namespace NUMINAMATH_GPT_at_least_one_true_l1679_167949

-- Definitions (Conditions)
variables (p q : Prop)

-- Statement
theorem at_least_one_true (h : p ∨ q) : p ∨ q := by
  sorry

end NUMINAMATH_GPT_at_least_one_true_l1679_167949


namespace NUMINAMATH_GPT_novel_writing_time_l1679_167943

theorem novel_writing_time :
  ∀ (total_words : ℕ) (first_half_speed second_half_speed : ℕ),
    total_words = 50000 →
    first_half_speed = 600 →
    second_half_speed = 400 →
    (total_words / 2 / first_half_speed + total_words / 2 / second_half_speed : ℚ) = 104.17 :=
by
  -- No proof is required, placeholder using sorry
  sorry

end NUMINAMATH_GPT_novel_writing_time_l1679_167943


namespace NUMINAMATH_GPT_range_a_inequality_l1679_167962

theorem range_a_inequality (a : ℝ) : (∀ x : ℝ, (a-2) * x^2 + 4 * (a-2) * x - 4 < 0) ↔ 1 < a ∧ a ≤ 2 :=
by {
    sorry
}

end NUMINAMATH_GPT_range_a_inequality_l1679_167962


namespace NUMINAMATH_GPT_simplify_sum_l1679_167968

theorem simplify_sum :
  -2^2004 + (-2)^2005 + 2^2006 - 2^2007 = -2^2004 - 2^2005 + 2^2006 - 2^2007 :=
by
  sorry

end NUMINAMATH_GPT_simplify_sum_l1679_167968


namespace NUMINAMATH_GPT_geometric_sequence_l1679_167907

theorem geometric_sequence (a b c r : ℤ) (h1 : b = a * r) (h2 : c = a * r^2) (h3 : c = a + 56) : b = 21 :=
by sorry

end NUMINAMATH_GPT_geometric_sequence_l1679_167907


namespace NUMINAMATH_GPT_M_subset_N_l1679_167952

noncomputable def M_set : Set ℝ := { x | ∃ (k : ℤ), x = k / 4 + 1 / 4 }
noncomputable def N_set : Set ℝ := { x | ∃ (k : ℤ), x = k / 8 - 1 / 4 }

theorem M_subset_N : M_set ⊆ N_set :=
sorry

end NUMINAMATH_GPT_M_subset_N_l1679_167952


namespace NUMINAMATH_GPT_solution_set_condition_l1679_167990

theorem solution_set_condition (a : ℝ) :
  (∀ x : ℝ, x * (x - a + 1) > a ↔ (x < -1 ∨ x > a)) → a > -1 :=
sorry

end NUMINAMATH_GPT_solution_set_condition_l1679_167990


namespace NUMINAMATH_GPT_prob_business_less25_correct_l1679_167987

def prob_male : ℝ := 0.4
def prob_female : ℝ := 0.6

def prob_science : ℝ := 0.3
def prob_arts : ℝ := 0.45
def prob_business : ℝ := 0.25

def prob_male_science_25plus : ℝ := 0.4
def prob_male_arts_25plus : ℝ := 0.5
def prob_male_business_25plus : ℝ := 0.35

def prob_female_science_25plus : ℝ := 0.3
def prob_female_arts_25plus : ℝ := 0.45
def prob_female_business_25plus : ℝ := 0.2

def prob_male_science_less25 : ℝ := 1 - prob_male_science_25plus
def prob_male_arts_less25 : ℝ := 1 - prob_male_arts_25plus
def prob_male_business_less25 : ℝ := 1 - prob_male_business_25plus

def prob_female_science_less25 : ℝ := 1 - prob_female_science_25plus
def prob_female_arts_less25 : ℝ := 1 - prob_female_arts_25plus
def prob_female_business_less25 : ℝ := 1 - prob_female_business_25plus

def prob_science_less25 : ℝ := prob_male * prob_science * prob_male_science_less25 + prob_female * prob_science * prob_female_science_less25
def prob_arts_less25 : ℝ := prob_male * prob_arts * prob_male_arts_less25 + prob_female * prob_arts * prob_female_arts_less25
def prob_business_less25 : ℝ := prob_male * prob_business * prob_male_business_less25 + prob_female * prob_business * prob_female_business_less25

theorem prob_business_less25_correct :
    prob_business_less25 = 0.185 :=
by
  -- Theorem statement to be proved (proof omitted)
  sorry

end NUMINAMATH_GPT_prob_business_less25_correct_l1679_167987


namespace NUMINAMATH_GPT_lumberjack_question_l1679_167900

def logs_per_tree (total_firewood : ℕ) (firewood_per_log : ℕ) (trees_chopped : ℕ) : ℕ :=
  total_firewood / firewood_per_log / trees_chopped

theorem lumberjack_question : logs_per_tree 500 5 25 = 4 := by
  sorry

end NUMINAMATH_GPT_lumberjack_question_l1679_167900


namespace NUMINAMATH_GPT_simplify_and_evaluate_expr_l1679_167909

theorem simplify_and_evaluate_expr (x : ℝ) (h : x = Real.sqrt 2 - 1) : 
  ((x + 3) * (x - 3) - x * (x - 2)) = 2 * Real.sqrt 2 - 11 := by
  rw [h]
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expr_l1679_167909


namespace NUMINAMATH_GPT_statement_1_equiv_statement_2_equiv_l1679_167917

-- Statement 1
variable (A B C : Prop)

theorem statement_1_equiv : ((A ∨ B) → C) ↔ (A → C) ∧ (B → C) :=
by
  sorry

-- Statement 2
theorem statement_2_equiv : (A → (B ∧ C)) ↔ (A → B) ∧ (A → C) :=
by
  sorry

end NUMINAMATH_GPT_statement_1_equiv_statement_2_equiv_l1679_167917


namespace NUMINAMATH_GPT_first_divisor_l1679_167936

theorem first_divisor (d x : ℕ) (h1 : ∃ k : ℕ, x = k * d + 11) (h2 : ∃ m : ℕ, x = 9 * m + 2) : d = 3 :=
sorry

end NUMINAMATH_GPT_first_divisor_l1679_167936


namespace NUMINAMATH_GPT_find_AD_l1679_167958

theorem find_AD
  (A B C D : Type)
  (BD BC CD AD : ℝ)
  (hBD : BD = 21)
  (hBC : BC = 30)
  (hCD : CD = 15)
  (hAngleBisect : true) -- Encode that D bisects the angle at C internally
  : AD = 35 := by
  sorry

end NUMINAMATH_GPT_find_AD_l1679_167958


namespace NUMINAMATH_GPT_range_of_a_l1679_167991

def A (x : ℝ) : Prop := (x - 1) * (x - 2) ≥ 0
def B (a x : ℝ) : Prop := x ≥ a

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, A x ∨ B a x) ↔ a ≤ 1 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1679_167991


namespace NUMINAMATH_GPT_costForFirstKgs_l1679_167935

noncomputable def applePrice (l : ℝ) (q : ℝ) (x : ℝ) (totalWeight : ℝ) : ℝ :=
  if totalWeight <= x then l * totalWeight else l * x + q * (totalWeight - x)

theorem costForFirstKgs (l q x : ℝ) :
  l = 10 ∧ q = 11 ∧ (applePrice l q x 33 = 333) ∧ (applePrice l q x 36 = 366) ∧ (applePrice l q 15 15 = 150) → x = 30 := 
by
  sorry

end NUMINAMATH_GPT_costForFirstKgs_l1679_167935


namespace NUMINAMATH_GPT_age_difference_l1679_167979

variables (P M Mo : ℕ)

def patrick_michael_ratio (P M : ℕ) : Prop := (P * 5 = M * 3)
def michael_monica_ratio (M Mo : ℕ) : Prop := (M * 4 = Mo * 3)
def sum_of_ages (P M Mo : ℕ) : Prop := (P + M + Mo = 88)

theorem age_difference (P M Mo : ℕ) : 
  patrick_michael_ratio P M → 
  michael_monica_ratio M Mo → 
  sum_of_ages P M Mo → 
  (Mo - P = 22) :=
by
  sorry

end NUMINAMATH_GPT_age_difference_l1679_167979


namespace NUMINAMATH_GPT_convert_ternary_to_octal_2101211_l1679_167922

def ternaryToOctal (n : List ℕ) : ℕ := 
  sorry

theorem convert_ternary_to_octal_2101211 :
  ternaryToOctal [2, 1, 0, 1, 2, 1, 1] = 444
  := sorry

end NUMINAMATH_GPT_convert_ternary_to_octal_2101211_l1679_167922


namespace NUMINAMATH_GPT_peregrine_falcon_dive_time_l1679_167997

theorem peregrine_falcon_dive_time 
  (bald_eagle_speed : ℝ := 100) 
  (peregrine_falcon_speed : ℝ := 2 * bald_eagle_speed) 
  (bald_eagle_time : ℝ := 30) : 
  peregrine_falcon_speed = 2 * bald_eagle_speed ∧ peregrine_falcon_speed / bald_eagle_speed = 2 →
  ∃ peregrine_falcon_time : ℝ, peregrine_falcon_time = 15 :=
by
  intro h
  use (bald_eagle_time / 2)
  sorry

end NUMINAMATH_GPT_peregrine_falcon_dive_time_l1679_167997


namespace NUMINAMATH_GPT_triangle_side_eq_median_l1679_167984

theorem triangle_side_eq_median (A B C : Type) (a b c : ℝ) (hAB : a = 2) (hAC : b = 3) (hBC_eq_median : c = (2 * (Real.sqrt (13 / 10)))) :
  c = (Real.sqrt 130) / 5 := by
  sorry

end NUMINAMATH_GPT_triangle_side_eq_median_l1679_167984


namespace NUMINAMATH_GPT_young_people_sampled_l1679_167981

def num_young_people := 800
def num_middle_aged_people := 1600
def num_elderly_people := 1400
def sampled_elderly_people := 70

-- Lean statement to prove the number of young people sampled
theorem young_people_sampled : 
  (sampled_elderly_people:ℝ) / num_elderly_people = (1 / 20:ℝ) ->
  num_young_people * (1 / 20:ℝ) = 40 := by
  sorry

end NUMINAMATH_GPT_young_people_sampled_l1679_167981


namespace NUMINAMATH_GPT_minimum_transportation_cost_l1679_167953

theorem minimum_transportation_cost :
  ∀ (x : ℕ), 
    (17 - x) + (x - 3) = 12 → 
    (18 - x) + (17 - x) = 14 → 
    (200 * x + 19300 = 19900) → 
    (x = 3) 
:= by sorry

end NUMINAMATH_GPT_minimum_transportation_cost_l1679_167953


namespace NUMINAMATH_GPT_seats_still_available_l1679_167904

theorem seats_still_available (total_seats : ℕ) (two_fifths_seats : ℕ) (one_tenth_seats : ℕ) 
  (h1 : total_seats = 500) 
  (h2 : two_fifths_seats = (2 * total_seats) / 5) 
  (h3 : one_tenth_seats = total_seats / 10) :
  total_seats - (two_fifths_seats + one_tenth_seats) = 250 :=
by 
  sorry

end NUMINAMATH_GPT_seats_still_available_l1679_167904


namespace NUMINAMATH_GPT_hyperbola_equation_focus_and_eccentricity_l1679_167994

theorem hyperbola_equation_focus_and_eccentricity (a b : ℝ)
  (h_focus : ∃ c : ℝ, c = 1 ∧ (∃ c_squared : ℝ, c_squared = c ^ 2))
  (h_eccentricity : ∃ e : ℝ, e = Real.sqrt 5 ∧ e = c / a)
  (h_b : b ^ 2 = c ^ 2 - a ^ 2) :
  5 * x^2 - (5 / 4) * y^2 = 1 :=
sorry

end NUMINAMATH_GPT_hyperbola_equation_focus_and_eccentricity_l1679_167994


namespace NUMINAMATH_GPT_sin_alpha_value_l1679_167992

-- Define the given conditions
def α : ℝ := sorry -- α is an acute angle
def β : ℝ := sorry -- β has an unspecified value

-- Given conditions translated to Lean
def condition1 : Prop := 2 * Real.tan (Real.pi - α) - 3 * Real.cos (Real.pi / 2 + β) + 5 = 0
def condition2 : Prop := Real.tan (Real.pi + α) + 6 * Real.sin (Real.pi + β) = 1

-- Acute angle condition
def α_acute : Prop := 0 < α ∧ α < Real.pi / 2

-- The proof statement
theorem sin_alpha_value (h1 : condition1) (h2 : condition2) (h3 : α_acute) : Real.sin α = 3 * Real.sqrt 10 / 10 :=
by sorry

end NUMINAMATH_GPT_sin_alpha_value_l1679_167992


namespace NUMINAMATH_GPT_find_constants_l1679_167929

open Set

variable {α : Type*} [LinearOrderedField α]

def Set_1 : Set α := {x | x^2 - 3*x + 2 = 0}

def Set_2 (a : α) : Set α := {x | x^2 - a*x + (a-1) = 0}

def Set_3 (m : α) : Set α := {x | x^2 - m*x + 2 = 0}

theorem find_constants (a m : α) :
  (Set_1 ∪ Set_2 a = Set_1) ∧ (Set_1 ∩ Set_2 a = Set_3 m) → 
  a = 3 ∧ m = 3 :=
by sorry

end NUMINAMATH_GPT_find_constants_l1679_167929


namespace NUMINAMATH_GPT_correct_operation_l1679_167995

theorem correct_operation (a b : ℝ) :
  (a + b) * (b - a) = b^2 - a^2 :=
by
  sorry

end NUMINAMATH_GPT_correct_operation_l1679_167995


namespace NUMINAMATH_GPT_range_of_t_range_of_a_l1679_167947

-- Proposition P: The curve equation represents an ellipse with foci on the x-axis
def propositionP (t : ℝ) : Prop := ∀ x y : ℝ, (x^2 / (4 - t) + y^2 / (t - 1) = 1)

-- Proof problem for t
theorem range_of_t (t : ℝ) (h : propositionP t) : 1 < t ∧ t < 5 / 2 := 
  sorry

-- Proposition Q: The inequality involving real number t
def propositionQ (t a : ℝ) : Prop := t^2 - (a + 3) * t + (a + 2) < 0

-- Proof problem for a
theorem range_of_a (a : ℝ) (h₁ : ∀ t : ℝ, propositionP t → propositionQ t a) 
                   (h₂ : ∃ t : ℝ, propositionQ t a ∧ ¬ propositionP t) :
  a > 1 / 2 :=
  sorry

end NUMINAMATH_GPT_range_of_t_range_of_a_l1679_167947


namespace NUMINAMATH_GPT_m_above_x_axis_m_on_line_l1679_167996

namespace ComplexNumberProblem

def above_x_axis (m : ℝ) : Prop :=
  m^2 - 2 * m - 15 > 0

def on_line (m : ℝ) : Prop :=
  2 * m^2 + 3 * m - 4 = 0

theorem m_above_x_axis (m : ℝ) : above_x_axis m → (m < -3 ∨ m > 5) :=
  sorry

theorem m_on_line (m : ℝ) : on_line m → 
  (m = (-3 + Real.sqrt 41) / 4) ∨ (m = (-3 - Real.sqrt 41) / 4) :=
  sorry

end ComplexNumberProblem

end NUMINAMATH_GPT_m_above_x_axis_m_on_line_l1679_167996


namespace NUMINAMATH_GPT_find_value_of_a_perpendicular_lines_l1679_167902

theorem find_value_of_a_perpendicular_lines :
  ∃ (a : ℝ), (∀ (x y : ℝ), y = a * x - 2 → y = 2 * x + 1 → 
  (a * 2 = -1)) → a = -1/2 :=
by
  sorry

end NUMINAMATH_GPT_find_value_of_a_perpendicular_lines_l1679_167902


namespace NUMINAMATH_GPT_find_ordered_pair_l1679_167916

theorem find_ordered_pair (s h : ℝ) :
  (∀ (u : ℝ), ∃ (x y : ℝ), x = s + 3 * u ∧ y = -3 + h * u ∧ y = 4 * x + 2) →
  (s, h) = (-5 / 4, 12) :=
by
  sorry

end NUMINAMATH_GPT_find_ordered_pair_l1679_167916


namespace NUMINAMATH_GPT_quadrilateral_offset_l1679_167945

-- Define the problem statement
theorem quadrilateral_offset
  (d : ℝ) (x : ℝ) (y : ℝ) (A : ℝ)
  (h₀ : d = 10) 
  (h₁ : y = 3) 
  (h₂ : A = 50) :
  x = 7 :=
by
  -- Assuming the given conditions
  have h₃ : A = 1/2 * d * x + 1/2 * d * y :=
  by
    -- specific formula for area of the quadrilateral
    sorry
  
  -- Given A = 50, d = 10, y = 3, solve for x to show x = 7
  sorry

end NUMINAMATH_GPT_quadrilateral_offset_l1679_167945


namespace NUMINAMATH_GPT_boat_speed_in_still_water_l1679_167918

-- Identifying the speeds of the boat in still water and the stream
variables (b s : ℝ)

-- Conditions stated in terms of equations
axiom boat_along_stream : b + s = 7
axiom boat_against_stream : b - s = 5

-- Prove that the boat speed in still water is 6 km/hr
theorem boat_speed_in_still_water : b = 6 :=
by
  sorry

end NUMINAMATH_GPT_boat_speed_in_still_water_l1679_167918


namespace NUMINAMATH_GPT_value_of_k_l1679_167913

noncomputable def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ := a + (n - 1) * d

theorem value_of_k
  (a d : ℝ)
  (a1_eq_1 : a = 1)
  (sum_9_eq_sum_4 : 9/2 * (2*a + 8*d) = 4/2 * (2*a + 3*d))
  (k : ℕ)
  (a_k_plus_a_4_eq_0 : arithmetic_sequence a d k + arithmetic_sequence a d 4 = 0) :
  k = 10 :=
by
  sorry

end NUMINAMATH_GPT_value_of_k_l1679_167913


namespace NUMINAMATH_GPT_initial_amount_is_1875_l1679_167989

-- Defining the conditions as given in the problem
def initial_amount : ℝ := sorry
def spent_on_clothes : ℝ := 250
def spent_on_food (remaining : ℝ) : ℝ := 0.35 * remaining
def spent_on_electronics (remaining : ℝ) : ℝ := 0.50 * remaining

-- Given conditions
axiom condition1 : initial_amount - spent_on_clothes = sorry
axiom condition2 : initial_amount - spent_on_clothes - spent_on_food (initial_amount - spent_on_clothes) = sorry
axiom condition3 : initial_amount - spent_on_clothes - spent_on_food (initial_amount - spent_on_clothes) - spent_on_electronics (initial_amount - spent_on_clothes - spent_on_food (initial_amount - spent_on_clothes)) = 200

-- Prove that initial amount is $1875
theorem initial_amount_is_1875 : initial_amount = 1875 :=
sorry

end NUMINAMATH_GPT_initial_amount_is_1875_l1679_167989


namespace NUMINAMATH_GPT_general_term_of_sequence_l1679_167912

theorem general_term_of_sequence (a : ℕ → ℝ) (h₁ : a 1 = 3) (h₂ : ∀ n : ℕ, n > 0 → a (n + 1) = (a n) ^ 2) :
  ∀ n : ℕ, n > 0 → a n = 3 ^ (2 ^ (n - 1)) :=
by
  intros n hn
  sorry

end NUMINAMATH_GPT_general_term_of_sequence_l1679_167912


namespace NUMINAMATH_GPT_pool_capacity_l1679_167944

-- Define the total capacity of the pool as a variable
variable (C : ℝ)

-- Define the conditions
def additional_water_needed (x : ℝ) : Prop :=
  x = 300

def increases_by_25_percent (x : ℝ) (y : ℝ) : Prop :=
  y = x * 0.25

-- State the proof problem
theorem pool_capacity :
  ∃ C : ℝ, additional_water_needed 300 ∧ increases_by_25_percent (0.75 * C) 300 ∧ C = 1200 :=
sorry

end NUMINAMATH_GPT_pool_capacity_l1679_167944


namespace NUMINAMATH_GPT_sum_of_cubes_l1679_167957

theorem sum_of_cubes {a b c : ℝ} (h1 : a + b + c = 5) (h2 : a * b + a * c + b * c = 7) (h3 : a * b * c = -18) : 
  a^3 + b^3 + c^3 = 29 :=
by
  -- The proof part is intentionally left out.
  sorry

end NUMINAMATH_GPT_sum_of_cubes_l1679_167957


namespace NUMINAMATH_GPT_exists_product_sum_20000_l1679_167924

theorem exists_product_sum_20000 :
  ∃ (k m : ℕ), 1 ≤ k ∧ k ≤ 999 ∧ 1 ≤ m ∧ m ≤ 999 ∧ k * (k + 1) + m * (m + 1) = 20000 :=
by 
  sorry

end NUMINAMATH_GPT_exists_product_sum_20000_l1679_167924


namespace NUMINAMATH_GPT_speed_in_still_water_l1679_167950

-- Definitions of the conditions
def downstream_condition (v_m v_s : ℝ) : Prop := v_m + v_s = 6
def upstream_condition (v_m v_s : ℝ) : Prop := v_m - v_s = 3

-- The theorem to be proven
theorem speed_in_still_water (v_m v_s : ℝ) 
  (h1 : downstream_condition v_m v_s) 
  (h2 : upstream_condition v_m v_s) : v_m = 4.5 :=
by
  sorry

end NUMINAMATH_GPT_speed_in_still_water_l1679_167950


namespace NUMINAMATH_GPT_distinguishable_arrangements_l1679_167931

theorem distinguishable_arrangements :
  let n := 9
  let n1 := 3
  let n2 := 2
  let n3 := 4
  (Nat.factorial n) / ((Nat.factorial n1) * (Nat.factorial n2) * (Nat.factorial n3)) = 1260 :=
by sorry

end NUMINAMATH_GPT_distinguishable_arrangements_l1679_167931


namespace NUMINAMATH_GPT_sally_pens_proof_l1679_167934

variable (p : ℕ)  -- define p as a natural number for pens each student received
variable (pensLeft : ℕ)  -- define pensLeft as a natural number for pens left after distributing to students

-- Function representing Sally giving pens to each student
def pens_after_giving_students (p : ℕ) : ℕ := 342 - 44 * p

-- Condition 1: Left half of the remainder in her locker
def locker_pens (p : ℕ) : ℕ := (pens_after_giving_students p) / 2

-- Condition 2: She took 17 pens home
def home_pens : ℕ := 17

-- Main proof statement
theorem sally_pens_proof :
  (locker_pens p + home_pens = pens_after_giving_students p) → p = 7 :=
by
  sorry

end NUMINAMATH_GPT_sally_pens_proof_l1679_167934


namespace NUMINAMATH_GPT_medium_pizza_slices_l1679_167965

theorem medium_pizza_slices (M : ℕ) 
  (small_pizza_slices : ℕ := 6)
  (large_pizza_slices : ℕ := 12)
  (total_pizzas : ℕ := 15)
  (small_pizzas : ℕ := 4)
  (medium_pizzas : ℕ := 5)
  (total_slices : ℕ := 136) :
  (small_pizzas * small_pizza_slices) + (medium_pizzas * M) + ((total_pizzas - small_pizzas - medium_pizzas) * large_pizza_slices) = total_slices → 
  M = 8 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_medium_pizza_slices_l1679_167965


namespace NUMINAMATH_GPT_probability_of_at_least_six_heads_is_correct_l1679_167970

-- Definitions for the given problem
def binomial_coefficient (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k

def total_possible_outcomes : ℕ :=
  2^8

def favorable_outcomes : ℕ :=
  binomial_coefficient 8 6 + binomial_coefficient 8 7 + binomial_coefficient 8 8

def probability_of_at_least_6_heads : ℚ :=
  favorable_outcomes / total_possible_outcomes

-- The proof statement
theorem probability_of_at_least_six_heads_is_correct :
  probability_of_at_least_6_heads = 37 / 256 :=
by sorry

end NUMINAMATH_GPT_probability_of_at_least_six_heads_is_correct_l1679_167970


namespace NUMINAMATH_GPT_second_pump_drain_time_l1679_167985

-- Definitions of the rates R1 and R2
def R1 : ℚ := 1 / 12  -- Rate of the first pump
def R2 : ℚ := 1 - R1  -- Rate of the second pump (from the combined rate equation)

-- The time it takes the second pump alone to drain the pond
def time_to_drain_second_pump := 1 / R2

-- The goal is to prove that this value is 12/11
theorem second_pump_drain_time : time_to_drain_second_pump = 12 / 11 := by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_second_pump_drain_time_l1679_167985


namespace NUMINAMATH_GPT_line_equation_l1679_167915

noncomputable def line_intersects_at_point (a1 a2 b1 b2 c1 c2 : ℝ) (p : ℝ × ℝ) : Prop :=
  p.1 * a1 + p.2 * b1 = c1 ∧ p.1 * a2 + p.2 * b2 = c2

noncomputable def point_on_line (a b c : ℝ) (p : ℝ × ℝ) : Prop :=
  a * p.1 + b * p.2 = c

theorem line_equation
  (p : ℝ × ℝ)
  (h1 : line_intersects_at_point 3 2 2 3 5 5 p)
  (h2 : point_on_line 0 1 (-5) p)
  : ∃ a b c : ℝ,  a * p.1 + b * p.2 + (-5) = 0 :=
sorry

end NUMINAMATH_GPT_line_equation_l1679_167915


namespace NUMINAMATH_GPT_unique_4_digit_number_l1679_167905

theorem unique_4_digit_number (P E R U : ℕ) 
  (hP : 0 ≤ P ∧ P < 10)
  (hE : 0 ≤ E ∧ E < 10)
  (hR : 0 ≤ R ∧ R < 10)
  (hU : 0 ≤ U ∧ U < 10)
  (hPERU : 1000 ≤ (P * 1000 + E * 100 + R * 10 + U) ∧ (P * 1000 + E * 100 + R * 10 + U) < 10000) 
  (h_eq : (P * 1000 + E * 100 + R * 10 + U) = (P + E + R + U) ^ U) : 
  (P = 4) ∧ (E = 9) ∧ (R = 1) ∧ (U = 3) ∧ (P * 1000 + E * 100 + R * 10 + U = 4913) :=
sorry

end NUMINAMATH_GPT_unique_4_digit_number_l1679_167905


namespace NUMINAMATH_GPT_number_of_pencil_boxes_l1679_167967

open Nat

def books_per_box : Nat := 46
def num_book_boxes : Nat := 19
def pencils_per_box : Nat := 170
def total_books_and_pencils : Nat := 1894

theorem number_of_pencil_boxes :
  (total_books_and_pencils - (num_book_boxes * books_per_box)) / pencils_per_box = 6 := 
by
  sorry

end NUMINAMATH_GPT_number_of_pencil_boxes_l1679_167967


namespace NUMINAMATH_GPT_interval_proof_l1679_167951

theorem interval_proof (x : ℝ) (h1 : 2 < 3 * x) (h2 : 3 * x < 3) (h3 : 2 < 4 * x) (h4 : 4 * x < 3) :
    (2 / 3) < x ∧ x < (3 / 4) :=
sorry

end NUMINAMATH_GPT_interval_proof_l1679_167951


namespace NUMINAMATH_GPT_problem1_problem2_l1679_167919

-- Problem 1
theorem problem1 (a b : ℝ) (h : a ≠ 0) : 
  (a - b^2 / a) / ((a^2 + 2 * a * b + b^2) / a) = (a - b) / (a + b) :=
by
  sorry

-- Problem 2
theorem problem2 (x : ℝ) : 
  (6 - 2 * x ≥ 4) ∧ ((1 + 2 * x) / 3 > x - 1) ↔ (x ≤ 1) :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1679_167919


namespace NUMINAMATH_GPT_complement_of_A_relative_to_U_l1679_167973

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {x | 1 ≤ x ∧ x ≤ 3}

theorem complement_of_A_relative_to_U : (U \ A) = {4, 5, 6} := 
by
  sorry

end NUMINAMATH_GPT_complement_of_A_relative_to_U_l1679_167973


namespace NUMINAMATH_GPT_grandma_contribution_l1679_167910

def trip_cost : ℝ := 485
def candy_bar_profit : ℝ := 1.25
def candy_bars_sold : ℕ := 188
def amount_earned_from_selling_candy_bars : ℝ := candy_bars_sold * candy_bar_profit
def amount_grandma_gave : ℝ := trip_cost - amount_earned_from_selling_candy_bars

theorem grandma_contribution :
  amount_grandma_gave = 250 := by
  sorry

end NUMINAMATH_GPT_grandma_contribution_l1679_167910


namespace NUMINAMATH_GPT_minimum_boxes_cost_300_muffins_l1679_167954

theorem minimum_boxes_cost_300_muffins :
  ∃ (L_used M_used S_used : ℕ), 
    L_used + M_used + S_used = 28 ∧ 
    (L_used = 10 ∧ M_used = 15 ∧ S_used = 3) ∧ 
    (L_used * 15 + M_used * 9 + S_used * 5 = 300) ∧ 
    (L_used * 5 + M_used * 3 + S_used * 2 = 101) ∧ 
    (L_used ≤ 10 ∧ M_used ≤ 15 ∧ S_used ≤ 25) :=
by
  -- The proof is omitted (theorem statement only).
  sorry

end NUMINAMATH_GPT_minimum_boxes_cost_300_muffins_l1679_167954


namespace NUMINAMATH_GPT_geom_seq_sum_a3_a4_a5_l1679_167925

-- Define the geometric sequence terms and sum condition
def geometric_seq (a1 q : ℕ) (n : ℕ) : ℕ :=
  a1 * q^(n - 1)

def sum_first_three (a1 q : ℕ) : ℕ :=
  a1 + a1 * q + a1 * q^2

-- Given conditions
def a1 : ℕ := 3
def S3 : ℕ := 21

-- Define the problem statement
theorem geom_seq_sum_a3_a4_a5 (q : ℕ) (h : sum_first_three a1 q = S3) (h_pos : ∀ n, geometric_seq a1 q n > 0) :
  geometric_seq a1 q 3 + geometric_seq a1 q 4 + geometric_seq a1 q 5 = 84 :=
by sorry

end NUMINAMATH_GPT_geom_seq_sum_a3_a4_a5_l1679_167925


namespace NUMINAMATH_GPT_number_composite_l1679_167914

theorem number_composite : ∃ a1 a2 : ℕ, a1 > 1 ∧ a2 > 1 ∧ 2^17 + 2^5 - 1 = a1 * a2 := 
by
  sorry

end NUMINAMATH_GPT_number_composite_l1679_167914


namespace NUMINAMATH_GPT_amare_fabric_needed_l1679_167932

-- Definitions for the conditions
def fabric_per_dress_yards : ℝ := 5.5
def number_of_dresses : ℕ := 4
def fabric_owned_feet : ℝ := 7
def yard_to_feet : ℝ := 3

-- Total fabric needed in yards
def total_fabric_needed_yards : ℝ := fabric_per_dress_yards * number_of_dresses

-- Total fabric needed in feet
def total_fabric_needed_feet : ℝ := total_fabric_needed_yards * yard_to_feet

-- Fabric still needed
def fabric_still_needed : ℝ := total_fabric_needed_feet - fabric_owned_feet

-- Proof
theorem amare_fabric_needed : fabric_still_needed = 59 := by
  sorry

end NUMINAMATH_GPT_amare_fabric_needed_l1679_167932


namespace NUMINAMATH_GPT_tan_neg_405_eq_neg_1_l1679_167937

theorem tan_neg_405_eq_neg_1 :
  Real.tan (Real.pi * -405 / 180) = -1 := 
sorry

end NUMINAMATH_GPT_tan_neg_405_eq_neg_1_l1679_167937


namespace NUMINAMATH_GPT_zero_not_in_range_of_g_l1679_167920

noncomputable def g (x : ℝ) : ℤ :=
  if x > -3 then ⌈(Real.cos x) / (x + 3)⌉
  else if x < -3 then ⌊(Real.cos x) / (x + 3)⌋
  else 0 -- arbitrary value since it's undefined

theorem zero_not_in_range_of_g :
  ¬ (∃ x : ℝ, g x = 0) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_zero_not_in_range_of_g_l1679_167920


namespace NUMINAMATH_GPT_area_of_octagon_in_square_l1679_167946

theorem area_of_octagon_in_square (perimeter : ℝ) (side_length : ℝ) (area_square : ℝ)
  (segment_length : ℝ) (area_triangle : ℝ) (total_area_triangles : ℝ) :
  perimeter = 144 →
  side_length = perimeter / 4 →
  segment_length = side_length / 3 →
  area_triangle = (segment_length * segment_length) / 2 →
  total_area_triangles = 4 * area_triangle →
  area_square = side_length * side_length →
  (area_square - total_area_triangles) = 1008 :=
by
  sorry

end NUMINAMATH_GPT_area_of_octagon_in_square_l1679_167946


namespace NUMINAMATH_GPT_max_remainder_209_lt_120_l1679_167930

theorem max_remainder_209_lt_120 : 
  ∃ n : ℕ, n < 120 ∧ (209 % n = 104) := 
sorry

end NUMINAMATH_GPT_max_remainder_209_lt_120_l1679_167930


namespace NUMINAMATH_GPT_expression_evaluation_l1679_167940

theorem expression_evaluation :
  (3 * Real.sqrt 12 - 2 * Real.sqrt (1 / 3) + Real.sqrt 48) / (2 * Real.sqrt 3) + (Real.sqrt (1 / 3))^2 = 5 :=
by
  sorry

end NUMINAMATH_GPT_expression_evaluation_l1679_167940


namespace NUMINAMATH_GPT_projectiles_initial_distance_l1679_167964

theorem projectiles_initial_distance (Projectile1_speed Projectile2_speed Time_to_meet : ℕ) 
  (h1 : Projectile1_speed = 444)
  (h2 : Projectile2_speed = 555)
  (h3 : Time_to_meet = 2) : 
  (Projectile1_speed + Projectile2_speed) * Time_to_meet = 1998 := by
  sorry

end NUMINAMATH_GPT_projectiles_initial_distance_l1679_167964


namespace NUMINAMATH_GPT_gasoline_tank_capacity_l1679_167927

theorem gasoline_tank_capacity (x : ℝ)
  (h1 : (7 / 8) * x - (1 / 2) * x = 12) : x = 32 := 
sorry

end NUMINAMATH_GPT_gasoline_tank_capacity_l1679_167927


namespace NUMINAMATH_GPT_number_of_disconnected_regions_l1679_167926

theorem number_of_disconnected_regions (n : ℕ) (h : 2 ≤ n) : 
  ∀ R : ℕ → ℕ, (R 1 = 2) → 
  (∀ k, R k = k^2 - k + 2 → R (k + 1) = (k + 1)^2 - (k + 1) + 2) → 
  R n = n^2 - n + 2 :=
sorry

end NUMINAMATH_GPT_number_of_disconnected_regions_l1679_167926


namespace NUMINAMATH_GPT_students_not_A_either_l1679_167911

-- Given conditions as definitions
def total_students : ℕ := 40
def students_A_history : ℕ := 10
def students_A_math : ℕ := 18
def students_A_both : ℕ := 6

-- Statement to prove
theorem students_not_A_either : (total_students - (students_A_history + students_A_math - students_A_both)) = 18 := 
by
  sorry

end NUMINAMATH_GPT_students_not_A_either_l1679_167911


namespace NUMINAMATH_GPT_iggy_wednesday_run_6_l1679_167938

open Nat

noncomputable def iggy_miles_wednesday : ℕ :=
  let total_time := 4 * 60    -- Iggy spends 4 hours running (240 minutes)
  let pace := 10              -- Iggy runs 1 mile in 10 minutes
  let monday := 3
  let tuesday := 4
  let thursday := 8
  let friday := 3
  let total_miles_other_days := monday + tuesday + thursday + friday
  let total_time_other_days := total_miles_other_days * pace
  let wednesday_time := total_time - total_time_other_days
  wednesday_time / pace

theorem iggy_wednesday_run_6 :
  iggy_miles_wednesday = 6 := by
  sorry

end NUMINAMATH_GPT_iggy_wednesday_run_6_l1679_167938


namespace NUMINAMATH_GPT_book_configurations_l1679_167908

theorem book_configurations : 
  (∃ (configurations : Finset ℕ), configurations = {1, 2, 3, 4, 5, 6, 7} ∧ configurations.card = 7) 
  ↔ 
  (∃ (n : ℕ), n = 7) :=
by 
  sorry

end NUMINAMATH_GPT_book_configurations_l1679_167908


namespace NUMINAMATH_GPT_each_half_month_has_15_days_l1679_167971

noncomputable def days_in_each_half (total_days : ℕ) (mean_profit_total: ℚ) 
  (mean_profit_first_half: ℚ) (mean_profit_last_half: ℚ) : ℕ :=
  let first_half_days := total_days / 2
  let second_half_days := total_days - first_half_days
  first_half_days

theorem each_half_month_has_15_days (total_days : ℕ) (mean_profit_total: ℚ) 
  (mean_profit_first_half: ℚ) (mean_profit_last_half: ℚ) :
  total_days = 30 → mean_profit_total = 350 → mean_profit_first_half = 275 → mean_profit_last_half = 425 → 
  days_in_each_half total_days mean_profit_total mean_profit_first_half mean_profit_last_half = 15 :=
by
  intros h_days h_total h_first h_last
  sorry

end NUMINAMATH_GPT_each_half_month_has_15_days_l1679_167971


namespace NUMINAMATH_GPT_findFirstCarSpeed_l1679_167998

noncomputable def firstCarSpeed (v : ℝ) (blackCarSpeed : ℝ) (initialGap : ℝ) (timeToCatchUp : ℝ) : Prop :=
  blackCarSpeed * timeToCatchUp = initialGap + v * timeToCatchUp → v = 30

theorem findFirstCarSpeed :
  firstCarSpeed 30 50 20 1 :=
by
  sorry

end NUMINAMATH_GPT_findFirstCarSpeed_l1679_167998


namespace NUMINAMATH_GPT_equation1_solution_equation2_solution_equation3_solution_l1679_167959

theorem equation1_solution :
  ∀ x : ℝ, x^2 - 2 * x - 99 = 0 ↔ x = 11 ∨ x = -9 :=
by
  sorry

theorem equation2_solution :
  ∀ x : ℝ, x^2 + 5 * x = 7 ↔ x = (-5 - Real.sqrt 53) / 2 ∨ x = (-5 + Real.sqrt 53) / 2 :=
by
  sorry

theorem equation3_solution :
  ∀ x : ℝ, 4 * x * (2 * x + 1) = 3 * (2 * x + 1) ↔ x = -1/2 ∨ x = 3/4 :=
by
  sorry

end NUMINAMATH_GPT_equation1_solution_equation2_solution_equation3_solution_l1679_167959


namespace NUMINAMATH_GPT_mrs_peterson_change_l1679_167980

def num_tumblers : ℕ := 10
def cost_per_tumbler : ℕ := 45
def num_bills : ℕ := 5
def value_per_bill : ℕ := 100

theorem mrs_peterson_change : 
  (num_bills * value_per_bill) - (num_tumblers * cost_per_tumbler) = 50 :=
by
  sorry

end NUMINAMATH_GPT_mrs_peterson_change_l1679_167980


namespace NUMINAMATH_GPT_fuel_at_40_min_fuel_l1679_167921

section FuelConsumption

noncomputable def fuel_consumption (x : ℝ) : ℝ := (1 / 128000) * x^3 - (3 / 80) * x + 8

noncomputable def total_fuel (x : ℝ) : ℝ := (fuel_consumption x) * (100 / x)

theorem fuel_at_40 : total_fuel 40 = 17.5 :=
by sorry

theorem min_fuel : total_fuel 80 = 11.25 ∧ ∀ x, (0 < x ∧ x ≤ 120) → total_fuel x ≥ total_fuel 80 :=
by sorry

end FuelConsumption

end NUMINAMATH_GPT_fuel_at_40_min_fuel_l1679_167921


namespace NUMINAMATH_GPT_g_triple_application_l1679_167972

def g (x : ℕ) : ℕ := 7 * x + 3

theorem g_triple_application : g (g (g 3)) = 1200 :=
by
  sorry

end NUMINAMATH_GPT_g_triple_application_l1679_167972


namespace NUMINAMATH_GPT_value_of_4x_l1679_167999

variable (x : ℤ)

theorem value_of_4x (h : 2 * x - 3 = 10) : 4 * x = 26 := 
by
  sorry

end NUMINAMATH_GPT_value_of_4x_l1679_167999


namespace NUMINAMATH_GPT_power_division_l1679_167928

theorem power_division (a b : ℕ) (h₁ : 64 = 8^2) (h₂ : a = 15) (h₃ : b = 7) : 8^a / 64^b = 8 :=
by
  -- Equivalent to 8^15 / 64^7 = 8, given that 64 = 8^2
  sorry

end NUMINAMATH_GPT_power_division_l1679_167928
