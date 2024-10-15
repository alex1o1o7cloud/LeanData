import Mathlib

namespace NUMINAMATH_GPT_set_of_x_satisfying_inequality_l277_27796

theorem set_of_x_satisfying_inequality : 
  {x : ℝ | (x - 2)^2 < 9} = {x : ℝ | -1 < x ∧ x < 5} :=
by
  sorry

end NUMINAMATH_GPT_set_of_x_satisfying_inequality_l277_27796


namespace NUMINAMATH_GPT_inequality_and_equality_condition_l277_27717

theorem inequality_and_equality_condition (x : ℝ)
  (h : x ∈ (Set.Iio 0 ∪ Set.Ioi 0)) :
  max 0 (Real.log (|x|)) ≥ 
      ((Real.sqrt 5 - 1) / (2 * Real.sqrt 5)) * Real.log (|x|) + 
      (1 / (2 * Real.sqrt 5)) * Real.log (|x^2 - 1|) + 
      (1 / 2) * Real.log ((Real.sqrt 5 + 1) / 2)
  ∧ (max 0 (Real.log (|x|)) = 
      ((Real.sqrt 5 - 1) / (2 * Real.sqrt 5)) * Real.log (|x|) + 
      (1 / (2 * Real.sqrt 5)) * Real.log (|x^2 - 1|) + 
      (1 / 2) * Real.log ((Real.sqrt 5 + 1) / 2) ↔ 
      x = (Real.sqrt 5 - 1) / 2 ∨ 
      x = -(Real.sqrt 5 - 1) / 2 ∨ 
      x = (Real.sqrt 5 + 1) / 2 ∨ 
      x = -(Real.sqrt 5 + 1) / 2) :=
by
  sorry

end NUMINAMATH_GPT_inequality_and_equality_condition_l277_27717


namespace NUMINAMATH_GPT_number_of_bottle_caps_l277_27793

-- Condition: 7 bottle caps weigh exactly one ounce
def weight_of_7_caps : ℕ := 1 -- ounce

-- Condition: Josh's collection weighs 18 pounds, and 1 pound = 16 ounces
def weight_of_collection_pounds : ℕ := 18 -- pounds
def weight_of_pound_in_ounces : ℕ := 16 -- ounces per pound

-- Condition: Question translated to proof statement
theorem number_of_bottle_caps :
  (weight_of_collection_pounds * weight_of_pound_in_ounces * 7 = 2016) :=
by
  sorry

end NUMINAMATH_GPT_number_of_bottle_caps_l277_27793


namespace NUMINAMATH_GPT_sphere_surface_area_l277_27759

theorem sphere_surface_area (r : ℝ) (hr : r = 3) : 4 * Real.pi * r^2 = 36 * Real.pi :=
by
  rw [hr]
  norm_num
  sorry

end NUMINAMATH_GPT_sphere_surface_area_l277_27759


namespace NUMINAMATH_GPT_average_price_per_share_l277_27722

-- Define the conditions
def Microtron_price_per_share := 36
def Dynaco_price_per_share := 44
def total_shares := 300
def Dynaco_shares_sold := 150

-- Define the theorem to be proved
theorem average_price_per_share : 
  (Dynaco_shares_sold * Dynaco_price_per_share + (total_shares - Dynaco_shares_sold) * Microtron_price_per_share) / total_shares = 40 :=
by
  -- Skip the actual proof here
  sorry

end NUMINAMATH_GPT_average_price_per_share_l277_27722


namespace NUMINAMATH_GPT_right_triangle_area_l277_27720

theorem right_triangle_area (a b c : ℝ) (h1 : a = 30) (h2 : c = 34) (h3 : a^2 + b^2 = c^2) :
  (1 / 2) * a * b = 240 :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_area_l277_27720


namespace NUMINAMATH_GPT_number_as_A_times_10_pow_N_integer_l277_27702

theorem number_as_A_times_10_pow_N_integer (A : ℝ) (N : ℝ) (hA1 : 1 ≤ A) (hA2 : A < 10) (hN : A * 10^N > 10) : ∃ (n : ℤ), N = n := 
sorry

end NUMINAMATH_GPT_number_as_A_times_10_pow_N_integer_l277_27702


namespace NUMINAMATH_GPT_remainder_of_product_mod_7_l277_27744

theorem remainder_of_product_mod_7 
  (a b c : ℕ) 
  (ha : a % 7 = 2) 
  (hb : b % 7 = 3) 
  (hc : c % 7 = 5) : 
  (a * b * c) % 7 = 2 :=
sorry

end NUMINAMATH_GPT_remainder_of_product_mod_7_l277_27744


namespace NUMINAMATH_GPT_melissa_earnings_from_sales_l277_27723

noncomputable def commission_earned (coupe_price suv_price commission_rate : ℕ) : ℕ :=
  (coupe_price + suv_price) * commission_rate / 100

theorem melissa_earnings_from_sales : 
  commission_earned 30000 60000 2 = 1800 :=
by
  sorry

end NUMINAMATH_GPT_melissa_earnings_from_sales_l277_27723


namespace NUMINAMATH_GPT_intersection_of_sets_l277_27769

theorem intersection_of_sets:
  let A := {-2, -1, 0, 1}
  let B := {x : ℤ | x^3 + 1 ≤ 0 }
  A ∩ B = {-2, -1} :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_sets_l277_27769


namespace NUMINAMATH_GPT_sum_of_solutions_l277_27729

theorem sum_of_solutions (y : ℝ) (h : y + 16 / y = 12) : y = 4 ∨ y = 8 → 4 + 8 = 12 :=
by sorry

end NUMINAMATH_GPT_sum_of_solutions_l277_27729


namespace NUMINAMATH_GPT_find_r_l277_27788

variable {x y r k : ℝ}

theorem find_r (h1 : y^2 + 4 * y + 4 + Real.sqrt (x + y + k) = 0)
               (h2 : r = |x * y|) :
    r = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_r_l277_27788


namespace NUMINAMATH_GPT_find_value_of_A_l277_27741

theorem find_value_of_A (A B : ℤ) (h1 : A - B = 144) (h2 : A = 3 * B - 14) : A = 223 :=
by
  sorry

end NUMINAMATH_GPT_find_value_of_A_l277_27741


namespace NUMINAMATH_GPT_fg_of_2_l277_27765

def g (x : ℝ) : ℝ := 2 * x^2
def f (x : ℝ) : ℝ := 2 * x - 1

theorem fg_of_2 : f (g 2) = 15 :=
by
  have h1 : g 2 = 8 := by sorry
  have h2 : f 8 = 15 := by sorry
  rw [h1]
  exact h2

end NUMINAMATH_GPT_fg_of_2_l277_27765


namespace NUMINAMATH_GPT_kamal_marks_physics_l277_27787

-- Define the marks in subjects
def marks_english := 66
def marks_mathematics := 65
def marks_chemistry := 62
def marks_biology := 75
def average_marks := 69
def number_of_subjects := 5

-- Calculate the total marks from the average
def total_marks := average_marks * number_of_subjects

-- Calculate the known total marks
def known_total_marks := marks_english + marks_mathematics + marks_chemistry + marks_biology

-- Define Kamal's marks in Physics
def marks_physics := total_marks - known_total_marks

-- Prove the marks in Physics are 77
theorem kamal_marks_physics : marks_physics = 77 := by
  sorry

end NUMINAMATH_GPT_kamal_marks_physics_l277_27787


namespace NUMINAMATH_GPT_Ariella_has_more_savings_l277_27792

variable (Daniella_savings: ℝ) (Ariella_future_savings: ℝ) (interest_rate: ℝ) (time_years: ℝ)
variable (initial_Ariella_savings: ℝ)

-- Conditions
axiom h1 : Daniella_savings = 400
axiom h2 : Ariella_future_savings = 720
axiom h3 : interest_rate = 0.10
axiom h4 : time_years = 2

-- Assume simple interest formula for future savings
axiom simple_interest : Ariella_future_savings = initial_Ariella_savings * (1 + interest_rate * time_years)

-- Show the difference in savings
theorem Ariella_has_more_savings : initial_Ariella_savings - Daniella_savings = 200 :=
by sorry

end NUMINAMATH_GPT_Ariella_has_more_savings_l277_27792


namespace NUMINAMATH_GPT_actual_average_height_l277_27773

theorem actual_average_height
  (incorrect_avg_height : ℝ)
  (n : ℕ)
  (incorrect_height : ℝ)
  (actual_height : ℝ)
  (h1 : incorrect_avg_height = 184)
  (h2 : n = 35)
  (h3 : incorrect_height = 166)
  (h4 : actual_height = 106) :
  let incorrect_total_height := incorrect_avg_height * n
  let difference := incorrect_height - actual_height
  let correct_total_height := incorrect_total_height - difference
  let correct_avg_height := correct_total_height / n
  correct_avg_height = 182.29 :=
by {
  sorry
}

end NUMINAMATH_GPT_actual_average_height_l277_27773


namespace NUMINAMATH_GPT_neither_cable_nor_vcr_fraction_l277_27798

variable (T : ℕ) -- Let T be the total number of housing units

def cableTV_fraction : ℚ := 1 / 5
def VCR_fraction : ℚ := 1 / 10
def both_fraction_given_cable : ℚ := 1 / 4

theorem neither_cable_nor_vcr_fraction : 
  (T : ℚ) * (1 - ((1 / 5) + ((1 / 10) - ((1 / 4) * (1 / 5))))) = (T : ℚ) * (3 / 4) :=
by sorry

end NUMINAMATH_GPT_neither_cable_nor_vcr_fraction_l277_27798


namespace NUMINAMATH_GPT_special_lines_count_l277_27761

noncomputable def count_special_lines : ℕ :=
  sorry

theorem special_lines_count :
  count_special_lines = 3 :=
by sorry

end NUMINAMATH_GPT_special_lines_count_l277_27761


namespace NUMINAMATH_GPT_min_value_is_nine_l277_27771

noncomputable def min_value_expression (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_sum : a + b + c = 9) :
  ℝ :=
  (a^2 + b^2) / (a + b) + (a^2 + c^2) / (a + c) + (b^2 + c^2) / (b + c)

theorem min_value_is_nine (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_sum : a + b + c = 9) :
  min_value_expression a b c h_pos h_sum = 9 :=
sorry

end NUMINAMATH_GPT_min_value_is_nine_l277_27771


namespace NUMINAMATH_GPT_area_of_given_triangle_l277_27713

noncomputable def area_of_triangle (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  (1 / 2) * |x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)|

theorem area_of_given_triangle :
  area_of_triangle (-2) 3 7 (-3) 4 6 = 31.5 :=
by
  sorry

end NUMINAMATH_GPT_area_of_given_triangle_l277_27713


namespace NUMINAMATH_GPT_min_correct_answers_for_score_above_60_l277_27706

theorem min_correct_answers_for_score_above_60 :
  ∃ (x : ℕ), 6 * x - 2 * (15 - x) > 60 ∧ x = 12 :=
by
  sorry

end NUMINAMATH_GPT_min_correct_answers_for_score_above_60_l277_27706


namespace NUMINAMATH_GPT_min_length_intersection_l277_27775

theorem min_length_intersection (m n : ℝ) (h_m1 : 0 ≤ m) (h_m2 : m + 7 / 10 ≤ 1) 
                                (h_n1 : 2 / 5 ≤ n) (h_n2 : n ≤ 1) : 
  ∃ (min_length : ℝ), min_length = 1 / 10 :=
by
  sorry

end NUMINAMATH_GPT_min_length_intersection_l277_27775


namespace NUMINAMATH_GPT_sum_of_products_is_50_l277_27703

theorem sum_of_products_is_50
  (a b c : ℝ)
  (h1 : a^2 + b^2 + c^2 = 156)
  (h2 : a + b + c = 16) :
  a * b + b * c + a * c = 50 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_products_is_50_l277_27703


namespace NUMINAMATH_GPT_sum_of_first_eleven_terms_l277_27786

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {d : ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop := ∃ d, ∀ n, a (n + 1) = a n + d

theorem sum_of_first_eleven_terms 
  (h_arith : is_arithmetic_sequence a)
  (h_S : ∀ n, S n = n * (a 1 + a n) / 2)
  (h_condition : 2 * a 7 - a 8 = 5) :
  S 11 = 55 :=
sorry

end NUMINAMATH_GPT_sum_of_first_eleven_terms_l277_27786


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l277_27708

theorem sufficient_but_not_necessary_condition (a b : ℝ) :
  (a > 1 ∧ b > 2) → (a + b > 3 ∧ a * b > 2) ∧ ¬((a + b > 3 ∧ a * b > 2) → (a > 1 ∧ b > 2)) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l277_27708


namespace NUMINAMATH_GPT_triangle_area_solutions_l277_27714

theorem triangle_area_solutions (ABC BDE : ℝ) (k : ℝ) (h₁ : BDE = k^2) : 
  S >= 4 * k^2 ∧ (if S = 4 * k^2 then solutions = 1 else solutions = 2) :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_solutions_l277_27714


namespace NUMINAMATH_GPT_domain_of_function_l277_27730

def f (x : ℝ) : ℝ := x^5 - 5 * x^4 + 10 * x^3 - 10 * x^2 + 5 * x - 1
def g (x : ℝ) : ℝ := x^2 - 9

theorem domain_of_function :
  {x : ℝ | g x ≠ 0} = {x : ℝ | x < -3} ∪ {x : ℝ | -3 < x ∧ x < 3} ∪ {x : ℝ | x > 3} :=
by
  sorry

end NUMINAMATH_GPT_domain_of_function_l277_27730


namespace NUMINAMATH_GPT_fourth_term_geometric_progression_l277_27776

theorem fourth_term_geometric_progression (x : ℝ) (h : ∀ n : ℕ, 0 < n → 
  (x ≠ 0 ∧ (2 * (x) + 2 * (n - 1)) ≠ 0 ∧ (3 * (x) + 3 * (n - 1)) ≠ 0)
  → ((2 * x + 2) / x) = (3 * x + 3) / (2 * x + 2)) : 
  ∃ r : ℝ, r = -13.5 := 
by 
  sorry

end NUMINAMATH_GPT_fourth_term_geometric_progression_l277_27776


namespace NUMINAMATH_GPT_g_zero_value_l277_27731

variables {R : Type*} [Ring R]

def polynomial_h (f g h : Polynomial R) : Prop :=
  h = f * g

def constant_term (p : Polynomial R) : R :=
  p.coeff 0

variables {f g h : Polynomial R}

theorem g_zero_value
  (Hf : constant_term f = 6)
  (Hh : constant_term h = -18)
  (H : polynomial_h f g h) :
  g.coeff 0 = -3 :=
by
  sorry

end NUMINAMATH_GPT_g_zero_value_l277_27731


namespace NUMINAMATH_GPT_orthocenter_PQR_l277_27760

structure Point3D :=
  (x : ℚ)
  (y : ℚ)
  (z : ℚ)

def orthocenter (P Q R : Point3D) : Point3D :=
  sorry

theorem orthocenter_PQR :
  orthocenter ⟨2, 3, 4⟩ ⟨6, 4, 2⟩ ⟨4, 5, 6⟩ = ⟨1/2, 13/2, 15/2⟩ :=
by {
  sorry
}

end NUMINAMATH_GPT_orthocenter_PQR_l277_27760


namespace NUMINAMATH_GPT_hospital_staff_total_l277_27707

def initial_doctors := 11
def initial_nurses := 18
def initial_medical_assistants := 9
def initial_interns := 6

def doctors_quit := 5
def nurses_quit := 2
def medical_assistants_quit := 3
def nurses_transferred := 2
def interns_transferred := 4
def doctors_vacation := 4
def nurses_vacation := 3

def new_doctors := 3
def new_nurses := 5

def remaining_doctors := initial_doctors - doctors_quit - doctors_vacation
def remaining_nurses := initial_nurses - nurses_quit - nurses_transferred - nurses_vacation
def remaining_medical_assistants := initial_medical_assistants - medical_assistants_quit
def remaining_interns := initial_interns - interns_transferred

def final_doctors := remaining_doctors + new_doctors
def final_nurses := remaining_nurses + new_nurses
def final_medical_assistants := remaining_medical_assistants
def final_interns := remaining_interns

def total_staff := final_doctors + final_nurses + final_medical_assistants + final_interns

theorem hospital_staff_total : total_staff = 29 := by
  unfold total_staff
  unfold final_doctors
  unfold final_nurses
  unfold final_medical_assistants
  unfold final_interns
  unfold remaining_doctors
  unfold remaining_nurses
  unfold remaining_medical_assistants
  unfold remaining_interns
  unfold initial_doctors initial_nurses initial_medical_assistants initial_interns
  unfold doctors_quit nurses_quit medical_assistants_quit nurses_transferred interns_transferred
  unfold doctors_vacation nurses_vacation
  unfold new_doctors new_nurses
  sorry

end NUMINAMATH_GPT_hospital_staff_total_l277_27707


namespace NUMINAMATH_GPT_domain_of_v_l277_27712

noncomputable def v (x : ℝ) : ℝ := 1 / (x^(1/3))

theorem domain_of_v : {x : ℝ | ∃ y, y = v x} = {x : ℝ | x ≠ 0} := by
  sorry

end NUMINAMATH_GPT_domain_of_v_l277_27712


namespace NUMINAMATH_GPT_second_train_start_time_l277_27780

-- Define the conditions as hypotheses
def station_distance : ℝ := 200
def speed_train_A : ℝ := 20
def speed_train_B : ℝ := 25
def meet_time : ℝ := 12 - 7 -- Time they meet after the first train starts, in hours.

-- The theorem statement corresponding to the proof problem
theorem second_train_start_time :
  ∃ T : ℝ, 0 <= T ∧ T <= 5 ∧ (5 * speed_train_A) + ((5 - T) * speed_train_B) = station_distance → T = 1 :=
by
  -- Placeholder for actual proof
  sorry

end NUMINAMATH_GPT_second_train_start_time_l277_27780


namespace NUMINAMATH_GPT_expression_equivalence_l277_27701

theorem expression_equivalence :
  (4 + 3) * (4^2 + 3^2) * (4^4 + 3^4) * (4^8 + 3^8) * (4^16 + 3^16) * (4^32 + 3^32) * (4^64 + 3^64) = 3^128 - 4^128 :=
by
  sorry

end NUMINAMATH_GPT_expression_equivalence_l277_27701


namespace NUMINAMATH_GPT_prime_numbers_satisfying_equation_l277_27735

theorem prime_numbers_satisfying_equation :
  ∀ p : ℕ, Nat.Prime p →
    (∃ x y : ℕ, 1 ≤ x ∧ 1 ≤ y ∧ x * (y^2 - p) + y * (x^2 - p) = 5 * p) →
    p = 2 ∨ p = 3 ∨ p = 7 := 
by 
  intro p hpprime h
  sorry

end NUMINAMATH_GPT_prime_numbers_satisfying_equation_l277_27735


namespace NUMINAMATH_GPT_units_digit_of_product_l277_27754

-- Define the three given even composite numbers
def a := 4
def b := 6
def c := 8

-- Define the product of the three numbers
def product := a * b * c

-- State the units digit of the product
theorem units_digit_of_product : product % 10 = 2 :=
by
  -- Proof is skipped here
  sorry

end NUMINAMATH_GPT_units_digit_of_product_l277_27754


namespace NUMINAMATH_GPT_symmetric_origin_coordinates_l277_27727

-- Given the coordinates (m, n) of point P
variables (m n : ℝ)
-- Define point P
def P := (m, n)

-- Define point P' which is symmetric to P with respect to the origin O
def P'_symmetric_origin : ℝ × ℝ := (-m, -n)

-- Prove that the coordinates of P' are (-m, -n)
theorem symmetric_origin_coordinates :
  P'_symmetric_origin m n = (-m, -n) :=
by
  -- Proof content goes here but we're skipping it with sorry
  sorry

end NUMINAMATH_GPT_symmetric_origin_coordinates_l277_27727


namespace NUMINAMATH_GPT_odd_number_expression_parity_l277_27782

theorem odd_number_expression_parity (o n : ℕ) (ho : ∃ k : ℕ, o = 2 * k + 1) :
  (o^2 + n * o) % 2 = 1 ↔ n % 2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_odd_number_expression_parity_l277_27782


namespace NUMINAMATH_GPT_original_selling_price_l277_27764

theorem original_selling_price (P : ℝ) (d1 d2 d3 t : ℝ) (final_price : ℝ) :
  d1 = 0.32 → -- first discount
  d2 = 0.10 → -- loyalty discount
  d3 = 0.05 → -- holiday discount
  t = 0.15 → -- state tax
  final_price = 650 → 
  1.15 * P * (1 - d1) * (1 - d2) * (1 - d3) = final_price →
  P = 722.57 :=
sorry

end NUMINAMATH_GPT_original_selling_price_l277_27764


namespace NUMINAMATH_GPT_min_value_S_l277_27747

noncomputable def S (x y : ℝ) : ℝ := 2 * x ^ 2 - x * y + y ^ 2 + 2 * x + 3 * y

theorem min_value_S : ∃ x y : ℝ, S x y = -4 ∧ ∀ (a b : ℝ), S a b ≥ -4 := 
by
  sorry

end NUMINAMATH_GPT_min_value_S_l277_27747


namespace NUMINAMATH_GPT_prob_white_ball_second_l277_27736

structure Bag :=
  (black_balls : ℕ)
  (white_balls : ℕ)

def total_balls (bag : Bag) := bag.black_balls + bag.white_balls

def prob_white_second_after_black_first (bag : Bag) : ℚ :=
  if bag.black_balls > 0 ∧ bag.white_balls > 0 ∧ total_balls bag > 1 then
    (bag.white_balls : ℚ) / (total_balls bag - 1)
  else 0

theorem prob_white_ball_second 
  (bag : Bag)
  (h_black : bag.black_balls = 4)
  (h_white : bag.white_balls = 3)
  (h_total : total_balls bag = 7) :
  prob_white_second_after_black_first bag = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_prob_white_ball_second_l277_27736


namespace NUMINAMATH_GPT_students_more_than_turtles_l277_27762

theorem students_more_than_turtles
  (students_per_classroom : ℕ)
  (turtles_per_classroom : ℕ)
  (number_of_classrooms : ℕ)
  (h1 : students_per_classroom = 20)
  (h2 : turtles_per_classroom = 3)
  (h3 : number_of_classrooms = 5) :
  (students_per_classroom * number_of_classrooms)
  - (turtles_per_classroom * number_of_classrooms) = 85 :=
by
  sorry

end NUMINAMATH_GPT_students_more_than_turtles_l277_27762


namespace NUMINAMATH_GPT_log_comparison_theorem_CauchySchwarz_inequality_theorem_trigonometric_minimum_theorem_l277_27758

noncomputable def log_comparison (n : ℕ) (hn : 0 < n) : Prop := 
  Real.log n / Real.log (n + 1) < Real.log (n + 1) / Real.log (n + 2)

theorem log_comparison_theorem (n : ℕ) (hn : 0 < n) : log_comparison n hn := 
  sorry

def inequality_CauchySchwarz (a b x y : ℝ) : Prop :=
  (a*a + b*b) * (x*x + y*y) ≥ (a*x + b*y) * (a*x + b*y)

theorem CauchySchwarz_inequality_theorem (a b x y : ℝ) : inequality_CauchySchwarz a b x y :=
  sorry

noncomputable def trigonometric_minimum (x : ℝ) : ℝ := 
  (Real.sin x)^2 + (Real.cos x)^2

theorem trigonometric_minimum_theorem : ∀ x : ℝ, trigonometric_minimum x ≥ 9 :=
  sorry

end NUMINAMATH_GPT_log_comparison_theorem_CauchySchwarz_inequality_theorem_trigonometric_minimum_theorem_l277_27758


namespace NUMINAMATH_GPT_find_solutions_l277_27767

-- Define the conditions
variable (n : ℕ)
noncomputable def valid_solution (a b c d : ℕ) : Prop := 
  a^2 + b^2 + c^2 + d^2 = 7 * 4^n

-- Define each possible solution
def sol1 : ℕ × ℕ × ℕ × ℕ := (5 * 2 ^ (n - 1), 2 ^ (n - 1), 2 ^ (n - 1), 2 ^ (n - 1))
def sol2 : ℕ × ℕ × ℕ × ℕ := (2 ^ (n + 1), 2 ^ n, 2 ^ n, 2 ^ n)
def sol3 : ℕ × ℕ × ℕ × ℕ := (3 * 2 ^ (n - 1), 3 * 2 ^ (n - 1), 3 * 2 ^ (n - 1), 2 ^ (n - 1))

-- State the theorem
theorem find_solutions (a b c d : ℕ) (n : ℕ) :
  valid_solution n a b c d →
  (a, b, c, d) = sol1 n ∨
  (a, b, c, d) = sol2 n ∨
  (a, b, c, d) = sol3 n :=
sorry

end NUMINAMATH_GPT_find_solutions_l277_27767


namespace NUMINAMATH_GPT_polyhedron_faces_same_edges_l277_27745

theorem polyhedron_faces_same_edges (n : ℕ) (h_n : n ≥ 4) : 
  ∃ (f1 f2 : ℕ), f1 ≠ f2 ∧ 3 ≤ f1 ∧ f1 ≤ n - 1 ∧ 3 ≤ f2 ∧ f2 ≤ n - 1 ∧ f1 = f2 := 
by
  sorry

end NUMINAMATH_GPT_polyhedron_faces_same_edges_l277_27745


namespace NUMINAMATH_GPT_unique_two_digit_number_l277_27709

theorem unique_two_digit_number (x y : ℕ) (h1 : 10 ≤ 10 * x + y ∧ 10 * x + y < 100) (h2 : 3 * y = 2 * x) (h3 : y + 3 = x) : 10 * x + y = 63 :=
by
  sorry

end NUMINAMATH_GPT_unique_two_digit_number_l277_27709


namespace NUMINAMATH_GPT_minimize_wire_length_l277_27725

theorem minimize_wire_length :
  ∃ (x : ℝ), (x > 0) ∧ (2 * (x + 4 / x) = 8) :=
by
  sorry

end NUMINAMATH_GPT_minimize_wire_length_l277_27725


namespace NUMINAMATH_GPT_g_diff_l277_27783

noncomputable section

-- Definition of g(n) as given in the problem statement
def g (n : ℕ) : ℝ :=
  (3 + 2 * Real.sqrt 3) / 6 * ((1 + Real.sqrt 3) / 2)^n +
  (3 - 2 * Real.sqrt 3) / 6 * ((1 - Real.sqrt 3) / 2)^n

-- The statement to prove g(n+2) - g(n) = -1/4 * g(n)
theorem g_diff (n : ℕ) : g (n + 2) - g n = -1 / 4 * g n :=
by
  sorry

end NUMINAMATH_GPT_g_diff_l277_27783


namespace NUMINAMATH_GPT_positive_difference_l277_27711

theorem positive_difference (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : |x - y| = 4 :=
sorry

end NUMINAMATH_GPT_positive_difference_l277_27711


namespace NUMINAMATH_GPT_child_height_at_age_10_l277_27752

theorem child_height_at_age_10 (x y : ℝ) (h : y = 7.19 * x + 73.93) (hx : x = 10) : abs (y - 145.83) < 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_child_height_at_age_10_l277_27752


namespace NUMINAMATH_GPT_arithmetic_square_root_of_16_l277_27755

theorem arithmetic_square_root_of_16 : ∃ x : ℝ, x^2 = 16 ∧ x > 0 ∧ x = 4 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_square_root_of_16_l277_27755


namespace NUMINAMATH_GPT_x_ge_y_l277_27795

variable (a : ℝ)

def x : ℝ := 2 * a * (a + 3)
def y : ℝ := (a - 3) * (a + 3)

theorem x_ge_y : x a ≥ y a := 
by 
  sorry

end NUMINAMATH_GPT_x_ge_y_l277_27795


namespace NUMINAMATH_GPT_remainder_of_poly_division_l277_27704

theorem remainder_of_poly_division :
  ∀ (x : ℂ), ((x + 1)^2048) % (x^2 - x + 1) = x + 1 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_poly_division_l277_27704


namespace NUMINAMATH_GPT_smallest_n_inequality_l277_27743

theorem smallest_n_inequality :
  ∃ n : ℕ, (∀ x y z : ℝ, (x^2 + y^2 + z^2)^2 ≤ n * (x^4 + y^4 + z^4)) ∧
    ∀ m : ℕ, m < n → ¬ (∀ x y z : ℝ, (x^2 + y^2 + z^2)^2 ≤ m * (x^4 + y^4 + z^4)) :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_inequality_l277_27743


namespace NUMINAMATH_GPT_algebraic_identity_l277_27781

variables {R : Type*} [CommRing R] (a b : R)

theorem algebraic_identity : 2 * (a - b) + 3 * b = 2 * a + b :=
by
  sorry

end NUMINAMATH_GPT_algebraic_identity_l277_27781


namespace NUMINAMATH_GPT_rosie_purchase_price_of_art_piece_l277_27756

-- Define the conditions as hypotheses
variables (P : ℝ)
variables (future_value increase : ℝ)

-- Given conditions
def conditions := future_value = 3 * P ∧ increase = 8000 ∧ increase = future_value - P

-- The statement to be proved
theorem rosie_purchase_price_of_art_piece (h : conditions P future_value increase) : P = 4000 :=
sorry

end NUMINAMATH_GPT_rosie_purchase_price_of_art_piece_l277_27756


namespace NUMINAMATH_GPT_square_of_sum_possible_l277_27794

theorem square_of_sum_possible (a b c : ℝ) : 
  ∃ d : ℝ, d = (a + b + c)^2 :=
sorry

end NUMINAMATH_GPT_square_of_sum_possible_l277_27794


namespace NUMINAMATH_GPT_distance_between_foci_of_ellipse_l277_27789

theorem distance_between_foci_of_ellipse (a b : ℝ) (ha : a = 2) (hb : b = 6) :
  ∀ (x y : ℝ), 9 * x^2 + y^2 = 36 → 2 * Real.sqrt (b^2 - a^2) = 8 * Real.sqrt 2 :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_distance_between_foci_of_ellipse_l277_27789


namespace NUMINAMATH_GPT_div_add_fraction_l277_27734

theorem div_add_fraction :
  (-75) / (-25) + 1/2 = 7/2 := by
  sorry

end NUMINAMATH_GPT_div_add_fraction_l277_27734


namespace NUMINAMATH_GPT_minor_premise_l277_27724

variables (A B C : Prop)

theorem minor_premise (hA : A) (hB : B) (hC : C) : B := 
by
  exact hB

end NUMINAMATH_GPT_minor_premise_l277_27724


namespace NUMINAMATH_GPT_Haleigh_needs_leggings_l277_27719

/-- Haleigh's pet animals -/
def dogs : Nat := 4
def cats : Nat := 3
def legs_per_dog : Nat := 4
def legs_per_cat : Nat := 4
def leggings_per_pair : Nat := 2

/-- The proof statement -/
theorem Haleigh_needs_leggings : (dogs * legs_per_dog + cats * legs_per_cat) / leggings_per_pair = 14 := by
  sorry

end NUMINAMATH_GPT_Haleigh_needs_leggings_l277_27719


namespace NUMINAMATH_GPT_two_truth_tellers_are_B_and_C_l277_27740

-- Definitions of students and their statements
def A_statement_false (A_said : Prop) (A_truth_teller : Prop) := ¬A_said = A_truth_teller
def B_statement_true (B_said : Prop) (B_truth_teller : Prop) := B_said = B_truth_teller
def C_statement_true (C_said : Prop) (C_truth_teller : Prop) := C_said = C_truth_teller
def D_statement_false (D_said : Prop) (D_truth_teller : Prop) := ¬D_said = D_truth_teller

-- Given statements
def A_said := ¬ (False : Prop)
def B_said := True
def C_said := B_said ∨ D_statement_false True True
def D_said := False

-- Define who is telling the truth
def A_truth_teller := False
def B_truth_teller := True
def C_truth_teller := True
def D_truth_teller := False

-- Proof problem statement
theorem two_truth_tellers_are_B_and_C :
  (A_statement_false A_said A_truth_teller) ∧
  (B_statement_true B_said B_truth_teller) ∧
  (C_statement_true C_said C_truth_teller) ∧
  (D_statement_false D_said D_truth_teller) →
  ((A_truth_teller = False) ∧
  (B_truth_teller = True) ∧
  (C_truth_teller = True) ∧
  (D_truth_teller = False)) := 
by {
  sorry
}

end NUMINAMATH_GPT_two_truth_tellers_are_B_and_C_l277_27740


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_cond_l277_27785

noncomputable
def geometric_sequence (a : ℕ → ℝ) (a1 : ℝ) (q : ℝ) : Prop :=
  a 1 = a1 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem necessary_but_not_sufficient_cond (a : ℕ → ℝ) (a1 : ℝ) (q : ℝ)
  (hseq : geometric_sequence a a1 q)
  (hpos : a1 > 0) :
  (q < 0 ↔ (∀ n : ℕ, a (2 * n + 1) + a (2 * n + 2) < 0)) :=
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_cond_l277_27785


namespace NUMINAMATH_GPT_gpa_of_entire_class_l277_27768

def students : ℕ := 200

def gpa1_num : ℕ := 18 * students / 100
def gpa2_num : ℕ := 27 * students / 100
def gpa3_num : ℕ := 22 * students / 100
def gpa4_num : ℕ := 12 * students / 100
def gpa5_num : ℕ := students - (gpa1_num + gpa2_num + gpa3_num + gpa4_num)

def gpa1 : ℕ := 58
def gpa2 : ℕ := 63
def gpa3 : ℕ := 69
def gpa4 : ℕ := 75
def gpa5 : ℕ := 85

def total_points : ℕ :=
  (gpa1_num * gpa1) + (gpa2_num * gpa2) + (gpa3_num * gpa3) + (gpa4_num * gpa4) + (gpa5_num * gpa5)

def class_gpa : ℚ := total_points / students

theorem gpa_of_entire_class :
  class_gpa = 69.48 := 
  by
  sorry

end NUMINAMATH_GPT_gpa_of_entire_class_l277_27768


namespace NUMINAMATH_GPT_monica_expected_winnings_l277_27770

def monica_die_winnings : List ℤ := [2, 3, 5, 7, 0, 0, 0, -4]

def expected_value (values : List ℤ) : ℚ :=
  (List.sum values) / (values.length : ℚ)

theorem monica_expected_winnings :
  expected_value monica_die_winnings = 1.625 := by
  sorry

end NUMINAMATH_GPT_monica_expected_winnings_l277_27770


namespace NUMINAMATH_GPT_michael_saves_more_l277_27715

-- Definitions for the conditions
def price_per_pair : ℝ := 50
def discount_a (price : ℝ) : ℝ := price + 0.6 * price
def discount_b (price : ℝ) : ℝ := 2 * price - 15

-- Statement to prove
theorem michael_saves_more (price : ℝ) (h : price = price_per_pair) : discount_b price - discount_a price = 5 :=
by
  sorry

end NUMINAMATH_GPT_michael_saves_more_l277_27715


namespace NUMINAMATH_GPT_max_right_angles_in_triangular_prism_l277_27705

theorem max_right_angles_in_triangular_prism 
  (n_triangles : ℕ) 
  (n_rectangles : ℕ) 
  (max_right_angles_triangle : ℕ) 
  (max_right_angles_rectangle : ℕ)
  (h1 : n_triangles = 2)
  (h2 : n_rectangles = 3)
  (h3 : max_right_angles_triangle = 1)
  (h4 : max_right_angles_rectangle = 4) : 
  (n_triangles * max_right_angles_triangle + n_rectangles * max_right_angles_rectangle = 14) :=
by
  sorry

end NUMINAMATH_GPT_max_right_angles_in_triangular_prism_l277_27705


namespace NUMINAMATH_GPT_number_of_pages_correct_number_of_ones_correct_l277_27749

noncomputable def number_of_pages (total_digits : ℕ) : ℕ :=
  let single_digit_odd_pages := 5
  let double_digit_odd_pages := 45
  let triple_digit_odd_pages := (total_digits - (single_digit_odd_pages + 2 * double_digit_odd_pages)) / 3
  single_digit_odd_pages + double_digit_odd_pages + triple_digit_odd_pages

theorem number_of_pages_correct : number_of_pages 125 = 60 :=
by sorry

noncomputable def number_of_ones (total_digits : ℕ) : ℕ :=
  let ones_in_units_place := 12
  let ones_in_tens_place := 18
  let ones_in_hundreds_place := 10
  ones_in_units_place + ones_in_tens_place + ones_in_hundreds_place

theorem number_of_ones_correct : number_of_ones 125 = 40 :=
by sorry

end NUMINAMATH_GPT_number_of_pages_correct_number_of_ones_correct_l277_27749


namespace NUMINAMATH_GPT_even_integers_count_form_3k_plus_4_l277_27774

theorem even_integers_count_form_3k_plus_4 
  (n : ℕ) (h1 : 20 ≤ n ∧ n ≤ 250)
  (h2 : ∃ k : ℕ, n = 3 * k + 4 ∧ Even n) : 
  ∃ N : ℕ, N = 39 :=
by {
  sorry
}

end NUMINAMATH_GPT_even_integers_count_form_3k_plus_4_l277_27774


namespace NUMINAMATH_GPT_right_triangle_divisibility_l277_27733

theorem right_triangle_divisibility (a b c : ℕ) (h : a^2 + b^2 = c^2) :
  (a % 3 = 0 ∨ b % 3 = 0) ∧ (a % 5 = 0 ∨ b % 5 = 0 ∨ c % 5 = 0) :=
by
  -- skipping the proof
  sorry

end NUMINAMATH_GPT_right_triangle_divisibility_l277_27733


namespace NUMINAMATH_GPT_remainder_3_pow_2040_mod_11_l277_27777

theorem remainder_3_pow_2040_mod_11 : (3 ^ 2040) % 11 = 1 := by
  have h1 : 3 % 11 = 3 := by norm_num
  have h2 : (3 ^ 2) % 11 = 9 := by norm_num
  have h3 : (3 ^ 3) % 11 = 5 := by norm_num
  have h4 : (3 ^ 4) % 11 = 4 := by norm_num
  have h5 : (3 ^ 5) % 11 = 1 := by norm_num
  have h_mod : 2040 % 5 = 0 := by norm_num
  sorry

end NUMINAMATH_GPT_remainder_3_pow_2040_mod_11_l277_27777


namespace NUMINAMATH_GPT_sum_series_eq_three_l277_27738

theorem sum_series_eq_three :
  (∑' k : ℕ, (9^k) / ((4^k - 3^k) * (4^(k + 1) - 3^(k + 1)))) = 3 :=
by 
  sorry

end NUMINAMATH_GPT_sum_series_eq_three_l277_27738


namespace NUMINAMATH_GPT_george_initial_amount_l277_27716

-- Definitions as per conditions
def cost_of_shirt : ℕ := 24
def cost_of_socks : ℕ := 11
def amount_left : ℕ := 65

-- Goal: Prove that the initial amount of money George had is 100
theorem george_initial_amount : (cost_of_shirt + cost_of_socks + amount_left) = 100 := 
by sorry

end NUMINAMATH_GPT_george_initial_amount_l277_27716


namespace NUMINAMATH_GPT_max_value_fraction_l277_27766

theorem max_value_fraction (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  (∀ x y : ℝ, (0 < x → 0 < y → (x / (2 * x + y) + y / (x + 2 * y)) ≤ 2 / 3)) :=
by
  sorry

end NUMINAMATH_GPT_max_value_fraction_l277_27766


namespace NUMINAMATH_GPT_cos_value_l277_27746

theorem cos_value (α : ℝ) (h : Real.sin (Real.pi / 4 - α) = 1 / 3) :
  Real.cos (Real.pi / 4 + α) = 1 / 3 :=
sorry

end NUMINAMATH_GPT_cos_value_l277_27746


namespace NUMINAMATH_GPT_count_irreducible_fractions_l277_27763

theorem count_irreducible_fractions (s : Finset ℕ) (h1 : ∀ n ∈ s, 15*n > 15/16) (h2 : ∀ n ∈ s, n < 1) (h3 : ∀ n ∈ s, Nat.gcd n 15 = 1) :
  s.card = 8 := 
sorry

end NUMINAMATH_GPT_count_irreducible_fractions_l277_27763


namespace NUMINAMATH_GPT_sum_of_g_is_zero_l277_27700

def g (x : ℝ) : ℝ := x^3 * (1 - x)^3

theorem sum_of_g_is_zero :
  (Finset.range 2022).sum (λ k => (-1)^(k + 1) * g ((k + 1 : ℝ) / 2023)) = 0 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_g_is_zero_l277_27700


namespace NUMINAMATH_GPT_set_union_intersection_l277_27710

def A : Set ℕ := {1, 2}
def B : Set ℕ := {1, 2, 3}
def C : Set ℕ := {2, 3, 4}

theorem set_union_intersection :
  (A ∩ B) ∪ C = {1, 2, 3, 4} := 
by
  sorry

end NUMINAMATH_GPT_set_union_intersection_l277_27710


namespace NUMINAMATH_GPT_time_after_2021_hours_l277_27721

-- Definition of starting time and day
def start_time : Nat := 20 * 60 + 21  -- converting 20:21 to minutes
def hours_per_day : Nat := 24
def minutes_per_hour : Nat := 60
def days_per_week : Nat := 7

-- Define the main statement
theorem time_after_2021_hours :
  let total_minutes := 2021 * minutes_per_hour
  let total_days := total_minutes / (hours_per_day * minutes_per_hour)
  let remaining_minutes := total_minutes % (hours_per_day * minutes_per_hour)
  let final_minutes := start_time + remaining_minutes
  let final_day := (total_days + 1) % days_per_week -- start on Monday (0), hence +1 for Tuesday
  final_minutes / minutes_per_hour = 1 ∧ final_minutes % minutes_per_hour = 21 ∧ final_day = 2 :=
by
  sorry

end NUMINAMATH_GPT_time_after_2021_hours_l277_27721


namespace NUMINAMATH_GPT_ratio_four_of_v_m_l277_27757

theorem ratio_four_of_v_m (m v : ℝ) (h : m < v) 
  (h_eq : 5 * (3 / 4 * m) = v - 1 / 4 * m) : v / m = 4 :=
sorry

end NUMINAMATH_GPT_ratio_four_of_v_m_l277_27757


namespace NUMINAMATH_GPT_cats_not_eating_either_l277_27778

theorem cats_not_eating_either (total_cats : ℕ) (cats_liking_apples : ℕ) (cats_liking_fish : ℕ) (cats_liking_both : ℕ)
  (h1 : total_cats = 75) (h2 : cats_liking_apples = 15) (h3 : cats_liking_fish = 55) (h4 : cats_liking_both = 8) :
  ∃ cats_not_eating_either : ℕ, cats_not_eating_either = total_cats - (cats_liking_apples - cats_liking_both + cats_liking_fish - cats_liking_both + cats_liking_both) ∧ cats_not_eating_either = 13 :=
by
  sorry

end NUMINAMATH_GPT_cats_not_eating_either_l277_27778


namespace NUMINAMATH_GPT_range_of_m_l277_27799

theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, |x - 1| + |x + m| ≤ 4) ↔ -5 ≤ m ∧ m ≤ 3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l277_27799


namespace NUMINAMATH_GPT_sequence_general_formula_l277_27726

-- Define conditions: The sum of the first n terms of the sequence is Sn = an - 3
variable {a : ℕ → ℕ}
variable {S : ℕ → ℕ}
axiom condition (n : ℕ) : S n = a n - 3

-- Define the main theorem to prove
theorem sequence_general_formula (n : ℕ) (hn : 0 < n) : a n = 2 * 3 ^ n :=
sorry

end NUMINAMATH_GPT_sequence_general_formula_l277_27726


namespace NUMINAMATH_GPT_decagon_diagonal_relation_l277_27737

-- Define side length, shortest diagonal, and longest diagonal in a regular decagon
variable (a b d : ℝ)
variable (h1 : a > 0) -- Side length must be positive
variable (h2 : b > 0) -- Shortest diagonal length must be positive
variable (h3 : d > 0) -- Longest diagonal length must be positive

theorem decagon_diagonal_relation (ha : d^2 = 5 * a^2) (hb : b^2 = 3 * a^2) : b^2 = a * d :=
sorry

end NUMINAMATH_GPT_decagon_diagonal_relation_l277_27737


namespace NUMINAMATH_GPT_product_increase_2022_l277_27751

theorem product_increase_2022 (a b c : ℕ) (h1 : a = 1) (h2 : b = 1) (h3 : c = 678) :
  (a - 3) * (b - 3) * (c - 3) = a * b * c + 2022 :=
by {
  -- The proof would go here, but it's not required per the instructions.
  sorry
}

end NUMINAMATH_GPT_product_increase_2022_l277_27751


namespace NUMINAMATH_GPT_parabola_line_intersection_length_l277_27750

def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def line (x y k : ℝ) : Prop := y = k * x - 1
def focus : ℝ × ℝ := (1, 0)

theorem parabola_line_intersection_length (k x1 x2 y1 y2 : ℝ)
  (h_focus : line 1 0 k)
  (h_parabola1 : parabola x1 y1)
  (h_parabola2 : parabola x2 y2)
  (h_line1 : line x1 y1 k)
  (h_line2 : line x2 y2 k) :
  k = 1 ∧ (x1 + x2 + 2) = 8 :=
by
  sorry

end NUMINAMATH_GPT_parabola_line_intersection_length_l277_27750


namespace NUMINAMATH_GPT_dividend_calculation_l277_27728

theorem dividend_calculation
  (divisor : Int)
  (quotient : Int)
  (remainder : Int)
  (dividend : Int)
  (h_divisor : divisor = 800)
  (h_quotient : quotient = 594)
  (h_remainder : remainder = -968)
  (h_dividend : dividend = (divisor * quotient) + remainder) :
  dividend = 474232 := by
  sorry

end NUMINAMATH_GPT_dividend_calculation_l277_27728


namespace NUMINAMATH_GPT_oscar_bus_ride_length_l277_27718

/-- Oscar's bus ride to school is some distance, and Charlie's bus ride is 0.25 mile.
Oscar's bus ride is 0.5 mile longer than Charlie's. Prove that Oscar's bus ride is 0.75 mile. -/
theorem oscar_bus_ride_length (charlie_ride : ℝ) (h1 : charlie_ride = 0.25) 
  (oscar_ride : ℝ) (h2 : oscar_ride = charlie_ride + 0.5) : oscar_ride = 0.75 :=
by sorry

end NUMINAMATH_GPT_oscar_bus_ride_length_l277_27718


namespace NUMINAMATH_GPT_number_of_real_solutions_l277_27742

noncomputable def system_of_equations (n : ℕ) (a b c : ℝ) (x : Fin n → ℝ) : Prop :=
∀ i : Fin n, a * (x i) ^ 2 + b * (x i) + c = x (⟨(i + 1) % n, sorry⟩)

theorem number_of_real_solutions
  (a b c : ℝ)
  (h : a ≠ 0)
  (n : ℕ)
  (x : Fin n → ℝ) :
  (b - 1) ^ 2 - 4 * a * c < 0 → ¬(∃ x : Fin n → ℝ, system_of_equations n a b c x) ∧
  (b - 1) ^ 2 - 4 * a * c = 0 → ∃! x : Fin n → ℝ, system_of_equations n a b c x ∧
  (b - 1) ^ 2 - 4 * a * c > 0 → ∃ x : Fin n → ℝ, ∃ y : Fin n → ℝ, x ≠ y ∧ system_of_equations n a b c x ∧ system_of_equations n a b c y := 
sorry

end NUMINAMATH_GPT_number_of_real_solutions_l277_27742


namespace NUMINAMATH_GPT_simplify_expression_l277_27790

-- Define the given expression
def expr : ℚ := (5^6 + 5^3) / (5^5 - 5^2)

-- State the proof problem
theorem simplify_expression : expr = 315 / 62 := 
by sorry

end NUMINAMATH_GPT_simplify_expression_l277_27790


namespace NUMINAMATH_GPT_avg_pages_hr_difference_l277_27772

noncomputable def avg_pages_hr_diff (total_pages_ryan : ℕ) (hours_ryan : ℕ) (books_brother : ℕ) (pages_per_book : ℕ) (hours_brother : ℕ) : ℚ :=
  (total_pages_ryan / hours_ryan : ℚ) - (books_brother * pages_per_book / hours_brother : ℚ)

theorem avg_pages_hr_difference :
  avg_pages_hr_diff 4200 78 15 250 90 = 12.18 :=
by
  sorry

end NUMINAMATH_GPT_avg_pages_hr_difference_l277_27772


namespace NUMINAMATH_GPT_members_who_play_both_sports_l277_27732

theorem members_who_play_both_sports 
  (N B T Neither BT : ℕ) 
  (h1 : N = 27)
  (h2 : B = 17)
  (h3 : T = 19)
  (h4 : Neither = 2)
  (h5 : BT = B + T - N + Neither) : 
  BT = 11 := 
by 
  have h6 : 17 + 19 - 27 + 2 = 11 := by norm_num
  rw [h2, h3, h1, h4, h6] at h5
  exact h5

end NUMINAMATH_GPT_members_who_play_both_sports_l277_27732


namespace NUMINAMATH_GPT_value_of_p_l277_27791

noncomputable def p_value_condition (p q : ℝ) (h1 : p + q = 1) (h2 : p > 0) (h3 : q > 0) : Prop :=
  (9 * p^8 * q = 36 * p^7 * q^2)

theorem value_of_p (p q : ℝ) (h1 : p + q = 1) (h2 : p > 0) (h3 : q > 0) (h4 : p_value_condition p q h1 h2 h3) :
  p = 4 / 5 :=
by
  sorry

end NUMINAMATH_GPT_value_of_p_l277_27791


namespace NUMINAMATH_GPT_probability_of_rain_at_least_once_l277_27797

theorem probability_of_rain_at_least_once 
  (P_sat : ℝ) (P_sun : ℝ) (P_mon : ℝ)
  (h_sat : P_sat = 0.30)
  (h_sun : P_sun = 0.60)
  (h_mon : P_mon = 0.50) :
  (1 - (1 - P_sat) * (1 - P_sun) * (1 - P_mon)) * 100 = 86 :=
by
  rw [h_sat, h_sun, h_mon]
  sorry

end NUMINAMATH_GPT_probability_of_rain_at_least_once_l277_27797


namespace NUMINAMATH_GPT_shape_is_cone_l277_27779

-- Define the spherical coordinate system and the condition
structure SphericalCoord where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

def shape (c : ℝ) (p : SphericalCoord) : Prop := p.φ ≤ c

-- The shape described by \(\exists c, \forall p \in SphericalCoord, shape c p\) is a cone
theorem shape_is_cone (c : ℝ) (p : SphericalCoord) : shape c p → (c ≥ 0 ∧ c ≤ π → shape c p = Cone) :=
by
  sorry

end NUMINAMATH_GPT_shape_is_cone_l277_27779


namespace NUMINAMATH_GPT_Derek_test_score_l277_27739

def Grant_score (John_score : ℕ) : ℕ := John_score + 10
def John_score (Hunter_score : ℕ) : ℕ := 2 * Hunter_score
def Hunter_score : ℕ := 45
def Sarah_score (Grant_score : ℕ) : ℕ := Grant_score - 5
def Derek_score (John_score Grant_score : ℕ) : ℕ := (John_score + Grant_score) / 2

theorem Derek_test_score :
  Derek_score (John_score Hunter_score) (Grant_score (John_score Hunter_score)) = 95 :=
  by
  -- proof here
  sorry

end NUMINAMATH_GPT_Derek_test_score_l277_27739


namespace NUMINAMATH_GPT_domain_of_v_l277_27753

-- Define the function v
noncomputable def v (x y : ℝ) : ℝ := 1 / (x^(2/3) - y^(2/3))

-- State the domain of v
def domain_v : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 ≠ p.2 }

-- State the main theorem
theorem domain_of_v :
  ∀ x y : ℝ, x ≠ y ↔ (x, y) ∈ domain_v :=
by
  intro x y
  -- We don't need to provide proof
  sorry

end NUMINAMATH_GPT_domain_of_v_l277_27753


namespace NUMINAMATH_GPT_average_weight_decrease_l277_27784

theorem average_weight_decrease :
  let original_avg := 102
  let new_weight := 40
  let original_boys := 30
  let total_boys := original_boys + 1
  (original_avg - ((original_boys * original_avg + new_weight) / total_boys)) = 2 :=
by
  sorry

end NUMINAMATH_GPT_average_weight_decrease_l277_27784


namespace NUMINAMATH_GPT_units_digit_of_52_cubed_plus_29_cubed_l277_27748

-- Define the units digit of a number n
def units_digit (n : ℕ) : ℕ := n % 10

-- Given conditions as definitions in Lean
def units_digit_of_2_cubed : ℕ := units_digit (2^3)  -- 8
def units_digit_of_9_cubed : ℕ := units_digit (9^3)  -- 9

-- The main theorem to prove
theorem units_digit_of_52_cubed_plus_29_cubed : units_digit (52^3 + 29^3) = 7 :=
by
  sorry

end NUMINAMATH_GPT_units_digit_of_52_cubed_plus_29_cubed_l277_27748
