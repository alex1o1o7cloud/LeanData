import Mathlib

namespace standard_deviations_below_mean_l1657_165795

theorem standard_deviations_below_mean (μ σ x : ℝ) (hμ : μ = 14.5) (hσ : σ = 1.7) (hx : x = 11.1) :
    (μ - x) / σ = 2 := by
  sorry

end standard_deviations_below_mean_l1657_165795


namespace greatest_possible_value_of_y_l1657_165792

theorem greatest_possible_value_of_y 
  (x y : ℤ) 
  (h : x * y + 7 * x + 6 * y = -8) : 
  y ≤ 27 ∧ (exists x, x * y + 7 * x + 6 * y = -8) := 
sorry

end greatest_possible_value_of_y_l1657_165792


namespace angle_P_measure_l1657_165736

theorem angle_P_measure (P Q R S : ℝ) 
  (h1 : P = 3 * Q)
  (h2 : P = 4 * R)
  (h3 : P = 6 * S)
  (h_sum : P + Q + R + S = 360) : 
  P = 206 :=
by 
  sorry

end angle_P_measure_l1657_165736


namespace competition_end_time_l1657_165718

-- Definitions for the problem conditions
def start_time : ℕ := 15 * 60  -- 3:00 p.m. in minutes from midnight
def duration : ℕ := 1300       -- competition duration in minutes
def end_time : ℕ := start_time + duration

-- The expected end time in minutes from midnight, where 12:40 p.m. is (12*60 + 40) = 760 + 40 = 800 minutes from midnight.
def expected_end_time : ℕ := 12 * 60 + 40 

-- The theorem to prove
theorem competition_end_time : end_time = expected_end_time := by
  sorry

end competition_end_time_l1657_165718


namespace cost_of_fencing_l1657_165732

/-- The sides of a rectangular field are in the ratio 3:4.
If the area of the field is 10092 sq. m and the cost of fencing the field is 25 paise per meter,
then the cost of fencing the field is 101.5 rupees. --/
theorem cost_of_fencing (area : ℕ) (fencing_cost : ℝ) (ratio1 ratio2 perimeter : ℝ)
  (h_area : area = 10092)
  (h_ratio : ratio1 = 3 ∧ ratio2 = 4)
  (h_fencing_cost : fencing_cost = 0.25)
  (h_perimeter : perimeter = 406) :
  perimeter * fencing_cost = 101.5 := by
  sorry

end cost_of_fencing_l1657_165732


namespace problem_1_problem_2_l1657_165793

def f (x : ℝ) : ℝ := abs (2 * x + 3) + abs (2 * x - 1)

theorem problem_1 (x : ℝ) : (f x ≤ 5) ↔ (-7/4 ≤ x ∧ x ≤ 3/4) :=
by sorry

theorem problem_2 (m : ℝ) : (∃ x, f x < abs (m - 1)) ↔ (m > 5 ∨ m < -3) :=
by sorry

end problem_1_problem_2_l1657_165793


namespace misha_total_students_l1657_165796

-- Definitions based on the conditions
def misha_best_rank : ℕ := 75
def misha_worst_rank : ℕ := 75

-- Statement of the theorem to be proved
theorem misha_total_students (misha_is_best : misha_best_rank = 75) (misha_is_worst : misha_worst_rank = 75) : 
  (misha_best_rank - 1) + (misha_worst_rank - 1) + 1 = 149 :=
by
  sorry

end misha_total_students_l1657_165796


namespace red_lucky_stars_l1657_165723

theorem red_lucky_stars (x : ℕ) : (20 + x + 15 > 0) → (x / (20 + x + 15) : ℚ) = 0.5 → x = 35 := by
  sorry

end red_lucky_stars_l1657_165723


namespace min_M_value_l1657_165786

noncomputable def max_pq (p q : ℝ) : ℝ := if p ≥ q then p else q

noncomputable def M (x y : ℝ) : ℝ := max_pq (|x^2 + y + 1|) (|y^2 - x + 1|)

theorem min_M_value : (∀ x y : ℝ, M x y ≥ (3 : ℚ) / 4) ∧ (∃ x y : ℝ, M x y = (3 : ℚ) / 4) :=
sorry

end min_M_value_l1657_165786


namespace engineering_students_pass_percentage_l1657_165710

theorem engineering_students_pass_percentage :
  let num_male_students := 120
  let num_female_students := 100
  let perc_male_eng_students := 0.25
  let perc_female_eng_students := 0.20
  let perc_male_eng_pass := 0.20
  let perc_female_eng_pass := 0.25
  
  let num_male_eng_students := num_male_students * perc_male_eng_students
  let num_female_eng_students := num_female_students * perc_female_eng_students
  
  let num_male_eng_pass := num_male_eng_students * perc_male_eng_pass
  let num_female_eng_pass := num_female_eng_students * perc_female_eng_pass
  
  let total_eng_students := num_male_eng_students + num_female_eng_students
  let total_eng_pass := num_male_eng_pass + num_female_eng_pass
  
  (total_eng_pass / total_eng_students) * 100 = 22 :=
by
  sorry

end engineering_students_pass_percentage_l1657_165710


namespace increase_fraction_l1657_165781

theorem increase_fraction (A F : ℝ) 
  (h₁ : A = 83200) 
  (h₂ : A * (1 + F) ^ 2 = 105300) : 
  F = 0.125 :=
by
  sorry

end increase_fraction_l1657_165781


namespace num_students_in_class_l1657_165722

-- Define the conditions
variables (S : ℕ) (num_boys : ℕ) (num_boys_under_6ft : ℕ)

-- Assume the conditions given in the problem
axiom two_thirds_boys : num_boys = (2 * S) / 3
axiom three_fourths_under_6ft : num_boys_under_6ft = (3 * num_boys) / 4
axiom nineteen_boys_under_6ft : num_boys_under_6ft = 19

-- The statement we want to prove
theorem num_students_in_class : S = 38 :=
by
  -- Proof omitted (insert proof here)
  sorry

end num_students_in_class_l1657_165722


namespace second_group_students_l1657_165704

theorem second_group_students 
  (total_students : ℕ) 
  (first_group_students : ℕ) 
  (h1 : total_students = 71) 
  (h2 : first_group_students = 34) : 
  total_students - first_group_students = 37 :=
by 
  sorry

end second_group_students_l1657_165704


namespace loss_calculation_l1657_165740

-- Given conditions: 
-- The ratio of the amount of money Cara, Janet, and Jerry have is 4:5:6
-- The total amount of money they have is $75

theorem loss_calculation :
  let cara_ratio := 4
  let janet_ratio := 5
  let jerry_ratio := 6
  let total_ratio := cara_ratio + janet_ratio + jerry_ratio
  let total_money := 75
  let part_value := total_money / total_ratio
  let cara_money := cara_ratio * part_value
  let janet_money := janet_ratio * part_value
  let combined_money := cara_money + janet_money
  let selling_price := 0.80 * combined_money
  combined_money - selling_price = 9 :=
by
  sorry

end loss_calculation_l1657_165740


namespace fraction_of_AD_eq_BC_l1657_165729

theorem fraction_of_AD_eq_BC (x y : ℝ) (B C D A : ℝ) 
  (h1 : B < C) 
  (h2 : C < D)
  (h3 : D < A) 
  (hBD : B < D)
  (hCD : C < D)
  (hAD : A = D)
  (hAB : A - B = 3 * (D - B)) 
  (hAC : A - C = 7 * (D - C))
  (hx_eq : x = 2 * y) 
  (hADx : A - D = 4 * x)
  (hADy : A - D = 8 * y)
  : (C - B) = 1/8 * (A - D) := 
sorry

end fraction_of_AD_eq_BC_l1657_165729


namespace inequality_proof_l1657_165766

variable {a1 a2 a3 a4 a5 : ℝ}

theorem inequality_proof (h1 : 1 < a1) (h2 : 1 < a2) (h3 : 1 < a3) (h4 : 1 < a4) (h5 : 1 < a5) :
  16 * (a1 * a2 * a3 * a4 * a5 + 1) > (1 + a1) * (1 + a2) * (1 + a3) * (1 + a4) * (1 + a5) := 
sorry

end inequality_proof_l1657_165766


namespace tan_diff_eq_rat_l1657_165751

theorem tan_diff_eq_rat (A : ℝ × ℝ) (B : ℝ × ℝ) (hA : A = (1, 2)) (hB : B = (5, 1))
  (α β : ℝ)
  (hα : Real.tan α = 2) (hβ : Real.tan β = 1 / 5) :
  Real.tan (α - β) = 9 / 7 := by
  sorry

end tan_diff_eq_rat_l1657_165751


namespace milk_jars_good_for_sale_l1657_165745

noncomputable def good_whole_milk_jars : ℕ := 
  let initial_jars := 60 * 30
  let short_deliveries := 20 * 30 * 2
  let damaged_jars_1 := 3 * 5
  let damaged_jars_2 := 4 * 6
  let totally_damaged_cartons := 2 * 30
  let received_jars := initial_jars - short_deliveries - damaged_jars_1 - damaged_jars_2 - totally_damaged_cartons
  let spoilage := (5 * received_jars) / 100
  received_jars - spoilage

noncomputable def good_skim_milk_jars : ℕ := 
  let initial_jars := 40 * 40
  let short_delivery := 10 * 40
  let damaged_jars := 5 * 4
  let totally_damaged_carton := 1 * 40
  let received_jars := initial_jars - short_delivery - damaged_jars - totally_damaged_carton
  let spoilage := (3 * received_jars) / 100
  received_jars - spoilage

noncomputable def good_almond_milk_jars : ℕ := 
  let initial_jars := 30 * 20
  let short_delivery := 5 * 20
  let damaged_jars := 2 * 3
  let received_jars := initial_jars - short_delivery - damaged_jars
  let spoilage := (1 * received_jars) / 100
  received_jars - spoilage

theorem milk_jars_good_for_sale : 
  good_whole_milk_jars = 476 ∧
  good_skim_milk_jars = 1106 ∧
  good_almond_milk_jars = 489 :=
by
  sorry

end milk_jars_good_for_sale_l1657_165745


namespace adding_books_multiplying_books_l1657_165702

-- Define the conditions
def num_books_first_shelf : ℕ := 4
def num_books_second_shelf : ℕ := 5
def num_books_third_shelf : ℕ := 6

-- Define the first question and prove its correctness
theorem adding_books :
  num_books_first_shelf + num_books_second_shelf + num_books_third_shelf = 15 :=
by
  -- The proof steps would go here, but they are replaced with sorry for now
  sorry

-- Define the second question and prove its correctness
theorem multiplying_books :
  num_books_first_shelf * num_books_second_shelf * num_books_third_shelf = 120 :=
by
  -- The proof steps would go here, but they are replaced with sorry for now
  sorry

end adding_books_multiplying_books_l1657_165702


namespace sum_of_differences_l1657_165789

theorem sum_of_differences (x : ℝ) (h : (45 + x) / 2 = 38) : abs (x - 45) + abs (x - 30) = 15 := by
  sorry

end sum_of_differences_l1657_165789


namespace least_positive_integer_divisibility_l1657_165714

theorem least_positive_integer_divisibility :
  ∃ n > 1, (∀ k ∈ [2, 3, 4, 5, 6, 7, 8, 9], n % k = 1) ∧ n = 2521 :=
by
  sorry

end least_positive_integer_divisibility_l1657_165714


namespace range_of_m_l1657_165750

-- Definitions according to the problem conditions
def p (x : ℝ) : Prop := (-2 ≤ x ∧ x ≤ 10)
def q (x : ℝ) (m : ℝ) : Prop := (1 - m ≤ x ∧ x ≤ 1 + m) ∧ m > 0

-- Rephrasing the problem statement in Lean
theorem range_of_m (x : ℝ) (m : ℝ) :
  (∀ x, p x → q x m) → m ≥ 9 :=
sorry

end range_of_m_l1657_165750


namespace sections_in_orchard_l1657_165772

-- Conditions: Farmers harvest 45 sacks from each section daily, 360 sacks are harvested daily
def harvest_sacks_per_section : ℕ := 45
def total_sacks_harvested_daily : ℕ := 360

-- Statement: Prove that the number of sections is 8 given the conditions
theorem sections_in_orchard (h1 : harvest_sacks_per_section = 45) (h2 : total_sacks_harvested_daily = 360) :
  total_sacks_harvested_daily / harvest_sacks_per_section = 8 :=
sorry

end sections_in_orchard_l1657_165772


namespace frequency_interval_20_to_inf_l1657_165738

theorem frequency_interval_20_to_inf (sample_size : ℕ)
  (freq_5_10 : ℕ) (freq_10_15 : ℕ) (freq_15_20 : ℕ)
  (freq_20_25 : ℕ) (freq_25_30 : ℕ) (freq_30_35 : ℕ) :
  sample_size = 35 ∧
  freq_5_10 = 5 ∧
  freq_10_15 = 12 ∧
  freq_15_20 = 7 ∧
  freq_20_25 = 5 ∧
  freq_25_30 = 4 ∧
  freq_30_35 = 2 →
  (1 - (freq_5_10 + freq_10_15 + freq_15_20 : ℕ) / (sample_size : ℕ) : ℝ) = 11 / 35 :=
by sorry

end frequency_interval_20_to_inf_l1657_165738


namespace weight_of_new_person_l1657_165746

/-- The average weight of 10 persons increases by 7.2 kg when a new person
replaces one who weighs 65 kg. Prove that the weight of the new person is 137 kg. -/
theorem weight_of_new_person (W_new : ℝ) (W_old : ℝ) (n : ℝ) (increase : ℝ) 
  (h1 : W_old = 65) (h2 : n = 10) (h3 : increase = 7.2) 
  (h4 : W_new = W_old + n * increase) : W_new = 137 := 
by
  -- proof to be done later
  sorry

end weight_of_new_person_l1657_165746


namespace base_six_product_correct_l1657_165734

namespace BaseSixProduct

-- Definitions of the numbers in base six
def num1_base6 : ℕ := 1 * 6^2 + 3 * 6^1 + 2 * 6^0
def num2_base6 : ℕ := 1 * 6^1 + 4 * 6^0

-- Their product in base ten
def product_base10 : ℕ := num1_base6 * num2_base6

-- Convert the base ten product back to base six
def product_base6 : ℕ := 2 * 6^3 + 3 * 6^2 + 3 * 6^1 + 2 * 6^0

-- Theorem statement
theorem base_six_product_correct : product_base10 = 560 ∧ product_base6 = 2332 := by
  sorry

end BaseSixProduct

end base_six_product_correct_l1657_165734


namespace nature_of_roots_l1657_165707

noncomputable def P (x : ℝ) : ℝ := x^6 - 5 * x^5 - 7 * x^3 - 2 * x + 9

theorem nature_of_roots : 
  (∀ x < 0, P x > 0) ∧ ∃ x > 0, P 0 * P x < 0 := 
by {
  sorry
}

end nature_of_roots_l1657_165707


namespace petrol_expenses_l1657_165711

-- Definitions based on the conditions stated in the problem
def salary_saved (salary : ℝ) : ℝ := 0.10 * salary
def total_known_expenses : ℝ := 5000 + 1500 + 4500 + 2500 + 3940

-- Main theorem statement that needs to be proved
theorem petrol_expenses (salary : ℝ) (petrol : ℝ) :
  salary_saved salary = 2160 ∧ salary - 2160 = 19440 ∧ 
  5000 + 1500 + 4500 + 2500 + 3940 = total_known_expenses  →
  petrol = 2000 :=
sorry

end petrol_expenses_l1657_165711


namespace extreme_value_l1657_165754

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x - a * (x - 1)

theorem extreme_value (a : ℝ) (h : a > 0) :
  ∃ x : ℝ, f x a = a - Real.log a - 1 ∧ (∀ y : ℝ, f y a ≤ f x a) :=
sorry

end extreme_value_l1657_165754


namespace obtain_2001_from_22_l1657_165721

theorem obtain_2001_from_22 :
  ∃ (f : ℕ → ℕ), (∀ n, f (n + 1) = n ∨ f (n) = n + 1) ∧ (f 22 = 2001) := 
sorry

end obtain_2001_from_22_l1657_165721


namespace reciprocal_of_neg_seven_l1657_165799

theorem reciprocal_of_neg_seven : (1 : ℚ) / (-7 : ℚ) = -1 / 7 :=
by
  sorry

end reciprocal_of_neg_seven_l1657_165799


namespace stereos_production_fraction_l1657_165726

/-
Company S produces three kinds of stereos: basic, deluxe, and premium.
Of the stereos produced by Company S last month, 2/5 were basic, 3/10 were deluxe, and the rest were premium.
It takes 1.6 as many hours to produce a deluxe stereo as it does to produce a basic stereo, and 2.5 as many hours to produce a premium stereo as it does to produce a basic stereo.
Prove that the number of hours it took to produce the deluxe and premium stereos last month was 123/163 of the total number of hours it took to produce all the stereos.
-/

def stereos_production (total_stereos : ℕ) (basic_ratio deluxe_ratio : ℚ)
  (deluxe_time_multiplier premium_time_multiplier : ℚ) : ℚ :=
  let basic_stereos := total_stereos * basic_ratio
  let deluxe_stereos := total_stereos * deluxe_ratio
  let premium_stereos := total_stereos - basic_stereos - deluxe_stereos
  let basic_time := basic_stereos
  let deluxe_time := deluxe_stereos * deluxe_time_multiplier
  let premium_time := premium_stereos * premium_time_multiplier
  let total_time := basic_time + deluxe_time + premium_time
  (deluxe_time + premium_time) / total_time

-- Given values
def total_stereos : ℕ := 100
def basic_ratio : ℚ := 2 / 5
def deluxe_ratio : ℚ := 3 / 10
def deluxe_time_multiplier : ℚ := 1.6
def premium_time_multiplier : ℚ := 2.5

theorem stereos_production_fraction : stereos_production total_stereos basic_ratio deluxe_ratio deluxe_time_multiplier premium_time_multiplier = 123 / 163 := by
  sorry

end stereos_production_fraction_l1657_165726


namespace max_abs_x_y_l1657_165782

theorem max_abs_x_y (x y : ℝ) (h : 4 * x^2 + y^2 = 4) : |x| + |y| ≤ 2 :=
by sorry

end max_abs_x_y_l1657_165782


namespace inheritance_amount_l1657_165758

-- Define the conditions
def federal_tax_rate : ℝ := 0.2
def state_tax_rate : ℝ := 0.1
def total_taxes_paid : ℝ := 10500

-- Lean statement for the proof
theorem inheritance_amount (I : ℝ)
  (h1 : federal_tax_rate = 0.2)
  (h2 : state_tax_rate = 0.1)
  (h3 : total_taxes_paid = 10500)
  (taxes_eq : total_taxes_paid = (federal_tax_rate * I) + (state_tax_rate * (I - (federal_tax_rate * I))))
  : I = 37500 :=
sorry

end inheritance_amount_l1657_165758


namespace vertex_angle_measure_l1657_165706

-- Define the isosceles triangle and its properties
def is_isosceles_triangle (A B C : ℝ) (a b c : ℝ) :=
  (A = B ∨ B = C ∨ C = A) ∧ (a + b + c = 180)

-- Define the conditions based on the problem statement
def two_angles_sum_to_100 (x y : ℝ) := x + y = 100

-- The measure of the vertex angle
theorem vertex_angle_measure (A B C : ℝ) (a b c : ℝ) 
  (h1 : is_isosceles_triangle A B C a b c) (h2 : two_angles_sum_to_100 A B) :
  C = 20 ∨ C = 80 :=
sorry

end vertex_angle_measure_l1657_165706


namespace more_girls_than_boys_l1657_165731

theorem more_girls_than_boys (total students : ℕ) (girls boys : ℕ) (h1 : total = 41) (h2 : girls = 22) (h3 : girls + boys = total) : (girls - boys) = 3 :=
by
  sorry

end more_girls_than_boys_l1657_165731


namespace point_on_coordinate_axes_l1657_165785

theorem point_on_coordinate_axes (x y : ℝ) (h : x * y = 0) : (x = 0 ∨ y = 0) :=
by sorry

end point_on_coordinate_axes_l1657_165785


namespace triangle_area_proof_l1657_165780

noncomputable def area_of_triangle_ABC : ℝ :=
  let r1 := 1 / 18
  let r2 := 2 / 9
  let AL := 1 / 9
  let CM := 1 / 6
  let KN := 2 * Real.sqrt (r1 * r2)
  let AC := AL + KN + CM
  let area := 3 / 11
  area

theorem triangle_area_proof :
  let r1 := 1 / 18
  let r2 := 2 / 9
  let AL := 1 / 9
  let CM := 1 / 6
  let KN := 2 * Real.sqrt (r1 * r2)
  let AC := AL + KN + CM
  area_of_triangle_ABC = 3 / 11 :=
by
  sorry

end triangle_area_proof_l1657_165780


namespace ratio_of_perimeters_l1657_165739

theorem ratio_of_perimeters (d : ℝ) (s1 s2 P1 P2 : ℝ) (h1 : d^2 = 2 * s1^2)
  (h2 : (3 * d)^2 = 2 * s2^2) (h3 : P1 = 4 * s1) (h4 : P2 = 4 * s2) :
  P2 / P1 = 3 := 
by sorry

end ratio_of_perimeters_l1657_165739


namespace addition_subtraction_questions_l1657_165768

theorem addition_subtraction_questions (total_questions word_problems answered_questions add_sub_questions : ℕ)
  (h1 : total_questions = 45)
  (h2 : word_problems = 17)
  (h3 : answered_questions = total_questions - 7)
  (h4 : add_sub_questions = answered_questions - word_problems) : 
  add_sub_questions = 21 := 
by 
  -- the proof steps are skipped
  sorry

end addition_subtraction_questions_l1657_165768


namespace remaining_miles_l1657_165728

theorem remaining_miles (total_miles : ℕ) (driven_miles : ℕ) (h1: total_miles = 1200) (h2: driven_miles = 642) :
  total_miles - driven_miles = 558 :=
by
  sorry

end remaining_miles_l1657_165728


namespace number_of_sandwiches_l1657_165742

-- Definitions based on the conditions in the problem
def sandwich_cost : Nat := 3
def water_cost : Nat := 2
def total_cost : Nat := 11

-- Lean statement to prove the number of sandwiches bought is 3
theorem number_of_sandwiches (S : Nat) (h : sandwich_cost * S + water_cost = total_cost) : S = 3 :=
by
  sorry

end number_of_sandwiches_l1657_165742


namespace tables_chairs_legs_l1657_165708

theorem tables_chairs_legs (t : ℕ) (c : ℕ) (total_legs : ℕ) 
  (h1 : c = 8 * t) 
  (h2 : total_legs = 4 * c + 6 * t) 
  (h3 : total_legs = 798) : 
  t = 21 :=
by
  sorry

end tables_chairs_legs_l1657_165708


namespace intersection_point_on_y_eq_neg_x_l1657_165784

theorem intersection_point_on_y_eq_neg_x 
  (α β : ℝ)
  (h1 : ∃ x y : ℝ, (x / (Real.sin α + Real.sin β) + y / (Real.sin α + Real.cos β) = 1) ∧ 
                   (x / (Real.cos α + Real.sin β) + y / (Real.cos α + Real.cos β) = 1) ∧ 
                   (y = -x)) :
  Real.sin α + Real.cos α + Real.sin β + Real.cos β = 0 :=
sorry

end intersection_point_on_y_eq_neg_x_l1657_165784


namespace number_of_restaurants_l1657_165775

def first_restaurant_meals_per_day := 20
def second_restaurant_meals_per_day := 40
def third_restaurant_meals_per_day := 50
def total_meals_per_week := 770

theorem number_of_restaurants :
  (first_restaurant_meals_per_day * 7) + 
  (second_restaurant_meals_per_day * 7) + 
  (third_restaurant_meals_per_day * 7) = total_meals_per_week → 
  3 = 3 :=
by 
  intros h
  sorry

end number_of_restaurants_l1657_165775


namespace solve_for_y_l1657_165743

theorem solve_for_y (x y : ℝ) (h1 : 2 * x - y = 10) (h2 : x + 3 * y = 2) : y = -6 / 7 := 
by
  sorry

end solve_for_y_l1657_165743


namespace f_at_2_l1657_165712

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  x ^ 2017 + a * x ^ 3 - b / x - 8

theorem f_at_2 (a b : ℝ) (h : f (-2) a b = 10) : f 2 a b = -26 :=
by sorry

end f_at_2_l1657_165712


namespace sum_of_squares_of_roots_l1657_165733

theorem sum_of_squares_of_roots :
  ∀ (x₁ x₂ : ℝ), (∀ a b c : ℝ, (a ≠ 0) →
  6 * x₁ ^ 2 + 5 * x₁ - 4 = 0 ∧ 6 * x₂ ^ 2 + 5 * x₂ - 4 = 0 →
  x₁ ^ 2 + x₂ ^ 2 = 73 / 36) :=
by
  sorry

end sum_of_squares_of_roots_l1657_165733


namespace stadium_revenue_difference_l1657_165724

theorem stadium_revenue_difference :
  let total_capacity := 2000
  let vip_capacity := 200
  let standard_capacity := 1000
  let general_capacity := 800
  let vip_price := 50
  let standard_price := 30
  let general_price := 20
  let three_quarters (n : ℕ) := (3 * n) / 4
  let three_quarter_full := three_quarters total_capacity
  let vip_three_quarter := three_quarters vip_capacity
  let standard_three_quarter := three_quarters standard_capacity
  let general_three_quarter := three_quarters general_capacity
  let revenue_three_quarter := vip_three_quarter * vip_price + standard_three_quarter * standard_price + general_three_quarter * general_price
  let revenue_full := vip_capacity * vip_price + standard_capacity * standard_price + general_capacity * general_price
  revenue_three_quarter = 42000 ∧ (revenue_full - revenue_three_quarter) = 14000 :=
by
  sorry

end stadium_revenue_difference_l1657_165724


namespace range_of_m_l1657_165741

theorem range_of_m (x m : ℝ) (h1: |x - m| < 1) (h2: x^2 - 8 * x + 12 < 0) (h3: ∀ x, (x^2 - 8 * x + 12 < 0) → ((m - 1) < x ∧ x < (m + 1))) : 
  3 ≤ m ∧ m ≤ 5 := 
sorry

end range_of_m_l1657_165741


namespace quadrilateral_area_is_6_l1657_165760

structure Point where
  x : ℝ
  y : ℝ

def A : Point := ⟨1, 2⟩
def B : Point := ⟨1, 1⟩
def C : Point := ⟨3, 1⟩
def D : Point := ⟨5, 5⟩

def area_triangle (P Q R : Point) : ℝ :=
  0.5 * abs ((Q.x - P.x) * (R.y - P.y) - (R.x - P.x) * (Q.y - P.y))

def quadrilateral_area (A B C D : Point) : ℝ :=
  area_triangle A B C + area_triangle A C D

theorem quadrilateral_area_is_6 : quadrilateral_area A B C D = 6 :=
  sorry

end quadrilateral_area_is_6_l1657_165760


namespace angle_in_first_quadrant_l1657_165713

theorem angle_in_first_quadrant (α : ℝ) (h : 90 < α ∧ α < 180) : 0 < 180 - α ∧ 180 - α < 90 :=
by
  sorry

end angle_in_first_quadrant_l1657_165713


namespace two_a_minus_five_d_eq_zero_l1657_165763

variables {α : Type*} [Field α]

def f (a b c d x : α) : α :=
  (2*a*x + 3*b) / (4*c*x - 5*d)

theorem two_a_minus_five_d_eq_zero
  (a b c d : α) (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0)
  (hf : ∀ x, f a b c d (f a b c d x) = x) :
  2*a - 5*d = 0 :=
sorry

end two_a_minus_five_d_eq_zero_l1657_165763


namespace hostel_cost_l1657_165716

def first_week_rate : ℝ := 18
def additional_week_rate : ℝ := 12
def first_week_days : ℕ := 7
def total_days : ℕ := 23

theorem hostel_cost :
  (first_week_days * first_week_rate + 
  (total_days - first_week_days) / first_week_days * first_week_days * additional_week_rate + 
  (total_days - first_week_days) % first_week_days * additional_week_rate) = 318 := 
by
  sorry

end hostel_cost_l1657_165716


namespace det_matrixE_l1657_165701

def matrixE : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![5, 0], ![0, 5]]

theorem det_matrixE : (matrixE.det) = 25 := by
  sorry

end det_matrixE_l1657_165701


namespace Adam_picks_apples_days_l1657_165762

theorem Adam_picks_apples_days (total_apples remaining_apples daily_pick : ℕ) 
  (h1 : total_apples = 350) 
  (h2 : remaining_apples = 230) 
  (h3 : daily_pick = 4) : 
  (total_apples - remaining_apples) / daily_pick = 30 :=
by {
  sorry
}

end Adam_picks_apples_days_l1657_165762


namespace fill_40x41_table_l1657_165757

-- Define the condition on integers in the table
def valid_integer_filling (m n : ℕ) (table : ℕ → ℕ → ℕ) : Prop :=
  ∀ i j, i < m → j < n →
    table i j =
    ((if i > 0 then if table i j = table (i - 1) j then 1 else 0 else 0) +
    (if j > 0 then if table i j = table i (j - 1) then 1 else 0 else 0) +
    (if i < m - 1 then if table i j = table (i + 1) j then 1 else 0 else 0) +
    (if j < n - 1 then if table i j = table i (j + 1) then 1 else 0 else 0))

-- Define the specific problem for a 40 × 41 table.
theorem fill_40x41_table :
  ∃ (table : ℕ → ℕ → ℕ), valid_integer_filling 40 41 table :=
by
  sorry

end fill_40x41_table_l1657_165757


namespace john_paid_percentage_l1657_165765

theorem john_paid_percentage (SRP WP : ℝ) (h1 : SRP = 1.40 * WP) (h2 : ∀ P, P = (1 / 3) * SRP) : ((1 / 3) * SRP / SRP * 100) = 33.33 :=
by
  sorry

end john_paid_percentage_l1657_165765


namespace systematic_sampling_sequence_l1657_165797

theorem systematic_sampling_sequence :
  ∃ k : ℕ, ∃ b : ℕ, (∀ n : ℕ, n < 6 → (3 + n * k = b + n * 10)) ∧ (b = 3 ∨ b = 13 ∨ b = 23 ∨ b = 33 ∨ b = 43 ∨ b = 53) :=
sorry

end systematic_sampling_sequence_l1657_165797


namespace solve_equation_l1657_165737

theorem solve_equation : ∀ x : ℝ, 4 * x - 2 * x + 1 - 3 = 0 → x = 1 :=
by
  intro x
  intro h
  sorry

end solve_equation_l1657_165737


namespace area_triangle_l1657_165788

noncomputable def area_of_triangle_ABC (AB BC : ℝ) : ℝ := 
    (1 / 2) * AB * BC 

theorem area_triangle (AC : ℝ) (h1 : AC = 40)
    (h2 : ∃ B C : ℝ, B = (1/2) * AC ∧ C = B * Real.sqrt 3) :
    area_of_triangle_ABC ((1 / 2) * AC) (((1 / 2) * AC) * Real.sqrt 3) = 200 * Real.sqrt 3 := 
sorry

end area_triangle_l1657_165788


namespace kilometers_to_meters_kilograms_to_grams_l1657_165735

def km_to_meters (km: ℕ) : ℕ := km * 1000
def kg_to_grams (kg: ℕ) : ℕ := kg * 1000

theorem kilometers_to_meters (h: 3 = 3): km_to_meters 3 = 3000 := by {
 sorry
}

theorem kilograms_to_grams (h: 4 = 4): kg_to_grams 4 = 4000 := by {
 sorry
}

end kilometers_to_meters_kilograms_to_grams_l1657_165735


namespace combination_square_octagon_tiles_l1657_165783

-- Define the internal angles of the polygons
def internal_angle (shape : String) : Float :=
  match shape with
  | "Square"   => 90.0
  | "Pentagon" => 108.0
  | "Hexagon"  => 120.0
  | "Octagon"  => 135.0
  | _          => 0.0

-- Define the condition for the combination of two regular polygons to tile seamlessly
def can_tile (shape1 shape2 : String) : Bool :=
  let angle1 := internal_angle shape1
  let angle2 := internal_angle shape2
  angle1 + 2 * angle2 == 360.0

-- Define the tiling problem
theorem combination_square_octagon_tiles : can_tile "Square" "Octagon" = true :=
by {
  -- The proof of this theorem should show that Square and Octagon can indeed tile seamlessly
  sorry
}

end combination_square_octagon_tiles_l1657_165783


namespace xiaoming_pens_l1657_165776

theorem xiaoming_pens (P M : ℝ) (hP : P > 0) (hM : M > 0) :
  (M / (7 / 8 * P) - M / P = 13) → (M / P = 91) := 
by
  sorry

end xiaoming_pens_l1657_165776


namespace g_at_50_l1657_165798

variable (g : ℝ → ℝ)

axiom g_functional_eq (x y : ℝ) : g (x * y) = x * g y
axiom g_at_1 : g 1 = 40

theorem g_at_50 : g 50 = 2000 :=
by
  -- Placeholder for proof
  sorry

end g_at_50_l1657_165798


namespace problem1_range_of_f_problem2_range_of_m_l1657_165748

noncomputable def f (x : ℝ) : ℝ :=
  (Real.log x / Real.log 2 - 2) * (Real.log x / Real.log 4 - 1/2)

theorem problem1_range_of_f :
  Set.range (fun x => f x) ∩ Set.Icc 1 4 = Set.Icc (-1/8 : ℝ) 1 :=
sorry

theorem problem2_range_of_m :
  ∀ x, x ∈ Set.Icc 4 16 → f x > (m : ℝ) * (Real.log x / Real.log 4) ↔ m < 0 :=
sorry

end problem1_range_of_f_problem2_range_of_m_l1657_165748


namespace length_of_PQ_is_8_l1657_165761

-- Define the lengths of the sides and conditions
variables (PQ QR PS SR : ℕ) (perimeter : ℕ)

-- State the conditions
def conditions : Prop :=
  SR = 16 ∧
  perimeter = 40 ∧
  PQ = QR ∧ QR = PS

-- State the goal
theorem length_of_PQ_is_8 (h : conditions PQ QR PS SR perimeter) : PQ = 8 :=
sorry

end length_of_PQ_is_8_l1657_165761


namespace inequality_always_true_l1657_165774

theorem inequality_always_true (x : ℝ) : x^2 + 1 ≥ 2 * |x| :=
sorry

end inequality_always_true_l1657_165774


namespace first_term_geometric_sequence_l1657_165777

theorem first_term_geometric_sequence (a r : ℚ) 
  (h3 : a * r^(3-1) = 24)
  (h4 : a * r^(4-1) = 36) :
  a = 32 / 3 :=
by
  sorry

end first_term_geometric_sequence_l1657_165777


namespace sheets_per_day_l1657_165727

-- Definitions based on conditions
def total_sheets : ℕ := 60
def total_days_per_week : ℕ := 7
def days_off : ℕ := 2

-- Derived condition from the problem
def work_days_per_week : ℕ := total_days_per_week - days_off

-- The statement to prove
theorem sheets_per_day : total_sheets / work_days_per_week = 12 :=
by
  sorry

end sheets_per_day_l1657_165727


namespace floor_add_self_eq_20_5_iff_l1657_165700

theorem floor_add_self_eq_20_5_iff (s : ℝ) : (⌊s⌋₊ : ℝ) + s = 20.5 ↔ s = 10.5 :=
by
  sorry

end floor_add_self_eq_20_5_iff_l1657_165700


namespace prism_is_five_sided_l1657_165778

-- Definitions based on problem conditions
def prism_faces (total_faces base_faces : Nat) := total_faces = 7 ∧ base_faces = 2

-- Theorem to prove based on the conditions
theorem prism_is_five_sided (total_faces base_faces : Nat) (h : prism_faces total_faces base_faces) : total_faces - base_faces = 5 :=
sorry

end prism_is_five_sided_l1657_165778


namespace sequence_closed_form_l1657_165705

theorem sequence_closed_form (a : ℕ → ℤ) (h1 : a 1 = 1) (h_rec : ∀ n : ℕ, a (n + 1) = 2 * a n + 3) :
  ∀ n : ℕ, a n = 2^(n + 1) - 3 :=
by 
sorry

end sequence_closed_form_l1657_165705


namespace cut_wood_into_5_pieces_l1657_165791

-- Definitions
def pieces_to_cuts (pieces : ℕ) : ℕ := pieces - 1
def time_per_cut (total_time : ℕ) (cuts : ℕ) : ℕ := total_time / cuts
def total_time_for_pieces (pieces : ℕ) (time_per_cut : ℕ) : ℕ := (pieces_to_cuts pieces) * time_per_cut

-- Given conditions
def conditions : Prop :=
  pieces_to_cuts 4 = 3 ∧
  time_per_cut 24 (pieces_to_cuts 4) = 8

-- Problem statement
theorem cut_wood_into_5_pieces (h : conditions) : total_time_for_pieces 5 8 = 32 :=
by sorry

end cut_wood_into_5_pieces_l1657_165791


namespace percentage_decrease_l1657_165771

theorem percentage_decrease 
  (P0 : ℕ) (P2 : ℕ) (H0 : P0 = 10000) (H2 : P2 = 9600) 
  (P1 : ℕ) (H1 : P1 = P0 + (20 * P0) / 100) :
  ∃ (D : ℕ), P2 = P1 - (D * P1) / 100 ∧ D = 20 :=
by
  sorry

end percentage_decrease_l1657_165771


namespace sum_of_all_two_digit_numbers_l1657_165709

theorem sum_of_all_two_digit_numbers : 
  let digits := [0, 1, 2, 3, 4, 5]
  let tens_digits := [1, 2, 3, 4, 5]
  let num_ones_digits := digits.length
  let num_tens_digits := tens_digits.length
  let sum_tens_place := 10 * (tens_digits.sum) * num_ones_digits
  let sum_ones_place := (digits.sum) * num_tens_digits
  sum_tens_place + sum_ones_place = 975 :=
by 
  let digits := [0, 1, 2, 3, 4, 5]
  let tens_digits := [1, 2, 3, 4, 5]
  let num_ones_digits := digits.length
  let num_tens_digits := tens_digits.length
  let sum_tens_place := 10 * (tens_digits.sum) * num_ones_digits
  let sum_ones_place := (digits.sum) * num_tens_digits
  show sum_tens_place + sum_ones_place = 975
  sorry

end sum_of_all_two_digit_numbers_l1657_165709


namespace length_of_chord_l1657_165752

theorem length_of_chord (x y : ℝ) 
  (h1 : (x - 1)^2 + y^2 = 4) 
  (h2 : x + y + 1 = 0) 
  : ∃ (l : ℝ), l = 2 * Real.sqrt 2 := by
  sorry

end length_of_chord_l1657_165752


namespace number_divided_is_144_l1657_165730

theorem number_divided_is_144 (n divisor quotient remainder : ℕ) (h_divisor : divisor = 11) (h_quotient : quotient = 13) (h_remainder : remainder = 1) (h_division : n = (divisor * quotient) + remainder) : n = 144 :=
by
  sorry

end number_divided_is_144_l1657_165730


namespace farmer_sowed_correct_amount_l1657_165720

def initial_buckets : ℝ := 8.75
def final_buckets : ℝ := 6
def buckets_sowed : ℝ := initial_buckets - final_buckets

theorem farmer_sowed_correct_amount : buckets_sowed = 2.75 :=
by {
  sorry
}

end farmer_sowed_correct_amount_l1657_165720


namespace min_sum_ab_l1657_165756

theorem min_sum_ab (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : (1 / a) + (2 / b) = 2) :
  a + b ≥ (3 + 2 * Real.sqrt 2) / 2 :=
sorry

end min_sum_ab_l1657_165756


namespace george_max_pencils_l1657_165715

-- Define the conditions for the problem
def total_money : ℝ := 9.30
def pencil_cost : ℝ := 1.05
def discount_rate : ℝ := 0.10

-- Define the final statement to prove
theorem george_max_pencils (n : ℕ) :
  (n ≤ 8 ∧ pencil_cost * n ≤ total_money) ∨ 
  (n > 8 ∧ pencil_cost * (1 - discount_rate) * n ≤ total_money) →
  n ≤ 9 :=
by
  sorry

end george_max_pencils_l1657_165715


namespace CD_eq_CE_l1657_165717

theorem CD_eq_CE {Point : Type*} [MetricSpace Point]
  (A B C D E : Point) (m : Set Point)
  (hAm : A ∈ m) (hBm : B ∈ m) (hCm : C ∈ m)
  (hDm : D ∉ m) (hEm : E ∉ m) 
  (hAD_AE : dist A D = dist A E)
  (hBD_BE : dist B D = dist B E) :
  dist C D = dist C E :=
sorry

end CD_eq_CE_l1657_165717


namespace lizas_final_balance_l1657_165755

-- Define the initial condition and subsequent changes
def initial_balance : ℕ := 800
def rent_payment : ℕ := 450
def paycheck_deposit : ℕ := 1500
def electricity_bill : ℕ := 117
def internet_bill : ℕ := 100
def phone_bill : ℕ := 70

-- Calculate the final balance
def final_balance : ℕ :=
  let balance_after_rent := initial_balance - rent_payment
  let balance_after_paycheck := balance_after_rent + paycheck_deposit
  let balance_after_bills := balance_after_paycheck - (electricity_bill + internet_bill)
  balance_after_bills - phone_bill

-- Theorem to prove that the final balance is 1563
theorem lizas_final_balance : final_balance = 1563 :=
by
  sorry

end lizas_final_balance_l1657_165755


namespace final_grey_cats_l1657_165719

def initially_total_cats : Nat := 16
def initial_white_cats : Nat := 2
def percent_black_cats : Nat := 25
def black_cats_left_fraction : Nat := 2
def new_white_cats : Nat := 2
def new_grey_cats : Nat := 1

/- We will calculate the number of grey cats after all specified events -/
theorem final_grey_cats :
  let total_cats := initially_total_cats
  let white_cats := initial_white_cats + new_white_cats
  let black_cats := (percent_black_cats * total_cats / 100) / black_cats_left_fraction
  let initial_grey_cats := total_cats - white_cats - black_cats
  let final_grey_cats := initial_grey_cats + new_grey_cats
  final_grey_cats = 11 := by
  sorry

end final_grey_cats_l1657_165719


namespace debate_schedule_ways_l1657_165787

-- Definitions based on the problem conditions
def east_debaters : Fin 4 := 4
def west_debaters : Fin 4 := 4
def total_debates := east_debaters.val * west_debaters.val
def debates_per_session := 3
def sessions := 5
def rest_debates := total_debates - sessions * debates_per_session

-- Claim that the number of scheduling ways is the given number
theorem debate_schedule_ways : (Nat.factorial total_debates) / ((Nat.factorial debates_per_session) ^ sessions * Nat.factorial rest_debates) = 20922789888000 :=
by
  -- Proof is skipped with sorry
  sorry

end debate_schedule_ways_l1657_165787


namespace geometric_sequence_fifth_term_l1657_165770

theorem geometric_sequence_fifth_term : 
  let a₁ := (2 : ℝ)
  let a₂ := (1 / 4 : ℝ)
  let r := a₂ / a₁
  let a₅ := a₁ * r ^ (5 - 1)
  a₅ = 1 / 2048 :=
by
  let a₁ := (2 : ℝ)
  let a₂ := (1 / 4 : ℝ)
  let r := a₂ / a₁
  let a₅ := a₁ * r ^ (5 - 1)
  sorry

end geometric_sequence_fifth_term_l1657_165770


namespace maximum_n_value_l1657_165790

theorem maximum_n_value (a b c d : ℝ) (n : ℕ) (h₀ : a > b) (h₁ : b > c) (h₂ : c > d) 
(h₃ : (1 / (a - b)) + (1 / (b - c)) + (1 / (c - d)) ≥ (n / (a - d))) : n ≤ 9 :=
sorry

end maximum_n_value_l1657_165790


namespace some_seniors_not_club_members_l1657_165749

variables {People : Type} (Senior ClubMember : People → Prop) (Punctual : People → Prop)

-- Conditions:
def some_seniors_not_punctual := ∃ x, Senior x ∧ ¬Punctual x
def all_club_members_punctual := ∀ x, ClubMember x → Punctual x

-- Theorem statement to be proven:
theorem some_seniors_not_club_members (h1 : some_seniors_not_punctual Senior Punctual) (h2 : all_club_members_punctual ClubMember Punctual) : 
  ∃ x, Senior x ∧ ¬ ClubMember x :=
sorry

end some_seniors_not_club_members_l1657_165749


namespace pens_given_to_sharon_l1657_165779

def initial_pens : Nat := 20
def mikes_pens : Nat := 22
def final_pens : Nat := 65

def total_pens_after_mike : Nat := initial_pens + mikes_pens
def total_pens_after_cindy : Nat := total_pens_after_mike * 2

theorem pens_given_to_sharon :
  total_pens_after_cindy - final_pens = 19 :=
by
  sorry

end pens_given_to_sharon_l1657_165779


namespace avg_of_6_10_N_is_10_if_even_l1657_165769

theorem avg_of_6_10_N_is_10_if_even (N : ℕ) (h1 : 9 ≤ N) (h2 : N ≤ 17) (h3 : (6 + 10 + N) % 2 = 0) : (6 + 10 + N) / 3 = 10 :=
by
-- sorry is placed here since we are not including the actual proof
sorry

end avg_of_6_10_N_is_10_if_even_l1657_165769


namespace rosie_pie_count_l1657_165744

def total_apples : ℕ := 40
def initial_apples_required : ℕ := 3
def apples_per_pie : ℕ := 5

theorem rosie_pie_count : (total_apples - initial_apples_required) / apples_per_pie = 7 :=
by
  sorry

end rosie_pie_count_l1657_165744


namespace total_bugs_eaten_l1657_165767

-- Define the conditions
def gecko_eats : ℕ := 12
def lizard_eats : ℕ := gecko_eats / 2
def frog_eats : ℕ := lizard_eats * 3
def toad_eats : ℕ := frog_eats + (frog_eats / 2)

-- Define the proof
theorem total_bugs_eaten : gecko_eats + lizard_eats + frog_eats + toad_eats = 63 :=
by
  sorry

end total_bugs_eaten_l1657_165767


namespace percentage_difference_wages_l1657_165703

variables (W1 W2 : ℝ)
variables (h1 : W1 > 0) (h2 : W2 > 0)
variables (h3 : 0.40 * W2 = 1.60 * 0.20 * W1)

theorem percentage_difference_wages (W1 W2 : ℝ) (h1 : W1 > 0) (h2 : W2 > 0) (h3 : 0.40 * W2 = 1.60 * 0.20 * W1) :
  (W1 - W2) / W1 = 0.20 :=
by
  sorry

end percentage_difference_wages_l1657_165703


namespace xiaoyu_reading_days_l1657_165794

theorem xiaoyu_reading_days
  (h1 : ∀ (p d : ℕ), p = 15 → d = 24 → p * d = 360)
  (h2 : ∀ (p t : ℕ), t = 360 → p = 18 → t / p = 20) :
  ∀ d : ℕ, d = 20 :=
by
  sorry

end xiaoyu_reading_days_l1657_165794


namespace pile_limit_exists_l1657_165725

noncomputable def log_floor (b x : ℝ) : ℤ :=
  Int.floor (Real.log x / Real.log b)

theorem pile_limit_exists (k : ℝ) (hk : k < 2) : ∃ Nk : ℤ, 
  Nk = 2 * (log_floor (2 / k) 2 + 1) := 
  by
    sorry

end pile_limit_exists_l1657_165725


namespace equal_triangle_area_l1657_165747

theorem equal_triangle_area
  (ABC_area : ℝ)
  (AP PB : ℝ)
  (AB_area : ℝ)
  (PQ_BQ_equal : Prop)
  (AP_ratio: AP / (AP + PB) = 3 / 5)
  (ABC_area_val : ABC_area = 15)
  (AP_val : AP = 3)
  (PB_val : PB = 2)
  (PQ_BQ_equal : PQ_BQ_equal = true) :
  ∃ area, area = 9 ∧ area = 9 :=
by
  sorry

end equal_triangle_area_l1657_165747


namespace conor_work_times_per_week_l1657_165759

-- Definitions for the conditions
def vegetables_per_day (eggplants carrots potatoes : ℕ) : ℕ :=
  eggplants + carrots + potatoes

def total_vegetables_per_week (days vegetables_per_day : ℕ) : ℕ :=
  days * vegetables_per_day

-- Theorem statement to be proven
theorem conor_work_times_per_week :
  let eggplants := 12
  let carrots := 9
  let potatoes := 8
  let weekly_total := 116
  vegetables_per_day eggplants carrots potatoes = 29 →
  total_vegetables_per_week 4 29 = 116 →
  4 = weekly_total / 29 :=
by
  intros _ _ h1 h2
  sorry

end conor_work_times_per_week_l1657_165759


namespace mary_puts_back_correct_number_of_oranges_l1657_165773

namespace FruitProblem

def price_apple := 40
def price_orange := 60
def total_fruits := 10
def average_price_all := 56
def average_price_kept := 50

theorem mary_puts_back_correct_number_of_oranges :
  ∀ (A O O' T: ℕ),
  A + O = total_fruits →
  A * price_apple + O * price_orange = total_fruits * average_price_all →
  A = 2 →
  T = A + O' →
  A * price_apple + O' * price_orange = T * average_price_kept →
  O - O' = 6 :=
by
  sorry

end FruitProblem

end mary_puts_back_correct_number_of_oranges_l1657_165773


namespace sufficient_but_not_necessary_condition_l1657_165753

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x > 2 → x^2 + 2 * x - 8 > 0) ∧ (¬(x > 2) → ¬(x^2 + 2 * x - 8 > 0)) → false :=
by 
  sorry

end sufficient_but_not_necessary_condition_l1657_165753


namespace reciprocal_neg_2023_l1657_165764

theorem reciprocal_neg_2023 : (1 / (-2023: ℤ)) = - (1 / 2023) :=
by
  -- proof goes here
  sorry

end reciprocal_neg_2023_l1657_165764
