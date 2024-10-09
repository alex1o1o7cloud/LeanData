import Mathlib

namespace train_overtake_distance_l1198_119815

theorem train_overtake_distance (speed_a speed_b hours_late time_to_overtake distance_a distance_b : ℝ) 
  (h1 : speed_a = 30)
  (h2 : speed_b = 38)
  (h3 : hours_late = 2) 
  (h4 : distance_a = speed_a * hours_late) 
  (h5 : distance_b = speed_b * time_to_overtake) 
  (h6 : time_to_overtake = distance_a / (speed_b - speed_a)) : 
  distance_b = 285 := sorry

end train_overtake_distance_l1198_119815


namespace average_speed_ratio_l1198_119836

def eddy_distance := 450 -- distance from A to B in km
def eddy_time := 3 -- time taken by Eddy in hours
def freddy_distance := 300 -- distance from A to C in km
def freddy_time := 4 -- time taken by Freddy in hours

def avg_speed (distance : ℕ) (time : ℕ) : ℕ := distance / time

def eddy_avg_speed := avg_speed eddy_distance eddy_time
def freddy_avg_speed := avg_speed freddy_distance freddy_time

def speed_ratio (speed1 : ℕ) (speed2 : ℕ) : ℕ × ℕ := (speed1 / (gcd speed1 speed2), speed2 / (gcd speed1 speed2))

theorem average_speed_ratio : speed_ratio eddy_avg_speed freddy_avg_speed = (2, 1) :=
by
  sorry

end average_speed_ratio_l1198_119836


namespace average_check_l1198_119860

variable (a b c d e f g x : ℕ)

def sum_natural (l : List ℕ) : ℕ := l.foldr (λ x y => x + y) 0

theorem average_check (h1 : a = 54) (h2 : b = 55) (h3 : c = 57) (h4 : d = 58) (h5 : e = 59) (h6 : f = 63) (h7 : g = 65) (h8 : x = 65) (avg : 60 * 8 = 480) :
    sum_natural [a, b, c, d, e, f, g, x] = 480 :=
by
  sorry

end average_check_l1198_119860


namespace find_missing_figure_l1198_119870

theorem find_missing_figure (x : ℝ) (h : 0.003 * x = 0.15) : x = 50 :=
sorry

end find_missing_figure_l1198_119870


namespace distance_amanda_to_kimberly_l1198_119863

-- Define the given conditions
def amanda_speed : ℝ := 2 -- miles per hour
def amanda_time : ℝ := 3 -- hours

-- Prove that the distance is 6 miles
theorem distance_amanda_to_kimberly : amanda_speed * amanda_time = 6 := by
  sorry

end distance_amanda_to_kimberly_l1198_119863


namespace mass_percentage_Cl_in_HClO2_is_51_78_l1198_119803

noncomputable def molar_mass_H : ℝ := 1.01
noncomputable def molar_mass_Cl : ℝ := 35.45
noncomputable def molar_mass_O : ℝ := 16.00

noncomputable def molar_mass_HClO2 : ℝ :=
  molar_mass_H + molar_mass_Cl + 2 * molar_mass_O

noncomputable def mass_percentage_Cl_in_HClO2 : ℝ :=
  (molar_mass_Cl / molar_mass_HClO2) * 100

theorem mass_percentage_Cl_in_HClO2_is_51_78 :
  mass_percentage_Cl_in_HClO2 = 51.78 := 
sorry

end mass_percentage_Cl_in_HClO2_is_51_78_l1198_119803


namespace simplify_product_l1198_119801

theorem simplify_product (b : ℤ) : (1 * 2 * b * 3 * b^2 * 4 * b^3 * 5 * b^4 * 6 * b^5) = 720 * b^15 := by
  sorry

end simplify_product_l1198_119801


namespace complex_magnitude_l1198_119822

theorem complex_magnitude (z : ℂ) (h : z * (2 - 4 * Complex.I) = 1 + 3 * Complex.I) :
  Complex.abs z = Real.sqrt 2 / 2 :=
by
  sorry

end complex_magnitude_l1198_119822


namespace parabola_chord_ratio_is_3_l1198_119827

noncomputable def parabola_chord_ratio (p : ℝ) (h : p > 0) : ℝ :=
  let focus_x := p / 2
  let a_x := (3 * p) / 2
  let b_x := p / 6
  let af := a_x + (p / 2)
  let bf := b_x + (p / 2)
  af / bf

theorem parabola_chord_ratio_is_3 (p : ℝ) (h : p > 0) : parabola_chord_ratio p h = 3 := by
  sorry

end parabola_chord_ratio_is_3_l1198_119827


namespace find_missing_number_l1198_119828

theorem find_missing_number (x : ℕ) : 
  (1 + 22 + 23 + 24 + 25 + 26 + x + 2) / 8 = 20 → x = 37 := by
  sorry

end find_missing_number_l1198_119828


namespace girl_weaves_on_tenth_day_l1198_119861

theorem girl_weaves_on_tenth_day 
  (a1 d : ℝ)
  (h1 : 7 * a1 + 21 * d = 28)
  (h2 : a1 + d + a1 + 4 * d + a1 + 7 * d = 15) :
  a1 + 9 * d = 10 :=
by sorry

end girl_weaves_on_tenth_day_l1198_119861


namespace emerson_distance_l1198_119869

theorem emerson_distance (d1 : ℕ) : 
  (d1 + 15 + 18 = 39) → d1 = 6 := 
by
  intro h
  have h1 : 33 = 39 - d1 := sorry -- Steps to manipulate equation to find d1
  sorry

end emerson_distance_l1198_119869


namespace joe_paint_usage_l1198_119829

theorem joe_paint_usage :
  let total_paint := 360
  let paint_first_week := total_paint * (1 / 4)
  let remaining_paint_after_first_week := total_paint - paint_first_week
  let paint_second_week := remaining_paint_after_first_week * (1 / 7)
  paint_first_week + paint_second_week = 128.57 :=
by
  sorry

end joe_paint_usage_l1198_119829


namespace bob_total_candies_l1198_119851

noncomputable def total_chewing_gums : ℕ := 45
noncomputable def total_chocolate_bars : ℕ := 60
noncomputable def total_assorted_candies : ℕ := 45

def chewing_gum_ratio_sam_bob : ℕ × ℕ := (2, 3)
def chocolate_bar_ratio_sam_bob : ℕ × ℕ := (3, 1)
def assorted_candy_ratio_sam_bob : ℕ × ℕ := (1, 1)

theorem bob_total_candies :
  let bob_chewing_gums := (total_chewing_gums * chewing_gum_ratio_sam_bob.snd) / (chewing_gum_ratio_sam_bob.fst + chewing_gum_ratio_sam_bob.snd)
  let bob_chocolate_bars := (total_chocolate_bars * chocolate_bar_ratio_sam_bob.snd) / (chocolate_bar_ratio_sam_bob.fst + chocolate_bar_ratio_sam_bob.snd)
  let bob_assorted_candies := (total_assorted_candies * assorted_candy_ratio_sam_bob.snd) / (assorted_candy_ratio_sam_bob.fst + assorted_candy_ratio_sam_bob.snd)
  bob_chewing_gums + bob_chocolate_bars + bob_assorted_candies = 64 := by
  sorry

end bob_total_candies_l1198_119851


namespace reciprocals_sum_l1198_119899

theorem reciprocals_sum (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a + b = 6 * a * b) : 
  (1 / a) + (1 / b) = 6 := 
sorry

end reciprocals_sum_l1198_119899


namespace time_to_pass_jogger_l1198_119864

noncomputable def jogger_speed_kmh : ℕ := 9
noncomputable def jogger_speed_ms : ℝ := jogger_speed_kmh * 1000 / 3600
noncomputable def train_length : ℕ := 130
noncomputable def jogger_ahead_distance : ℕ := 240
noncomputable def train_speed_kmh : ℕ := 45
noncomputable def train_speed_ms : ℝ := train_speed_kmh * 1000 / 3600
noncomputable def relative_speed : ℝ := train_speed_ms - jogger_speed_ms
noncomputable def total_distance_to_cover : ℕ := jogger_ahead_distance + train_length
noncomputable def time_taken_to_pass : ℝ := total_distance_to_cover / relative_speed

theorem time_to_pass_jogger : time_taken_to_pass = 37 := sorry

end time_to_pass_jogger_l1198_119864


namespace school_class_student_count_l1198_119892

theorem school_class_student_count
  (num_classes : ℕ) (num_students : ℕ)
  (h_classes : num_classes = 30)
  (h_students : num_students = 1000)
  (h_max_students_per_class : ∀(n : ℕ), n < 30 → ∀(s : ℕ), s ≤ 33 → s ≤ 1000 / 30) :
  ∃ c, c ≤ num_classes ∧ ∃s, s ≥ 34 :=
by
  sorry

end school_class_student_count_l1198_119892


namespace chocolate_chips_needed_l1198_119808

-- Define the variables used in the conditions
def cups_per_recipe := 2
def number_of_recipes := 23

-- State the theorem
theorem chocolate_chips_needed : (cups_per_recipe * number_of_recipes) = 46 := 
by sorry

end chocolate_chips_needed_l1198_119808


namespace no_x4_term_expansion_l1198_119807

-- Mathematical condition and properties
variable {R : Type*} [CommRing R]

theorem no_x4_term_expansion (a : R) (h : a ≠ 0) :
  ∃ a, (a = 8) := 
by 
  sorry

end no_x4_term_expansion_l1198_119807


namespace students_not_reading_novels_l1198_119853

theorem students_not_reading_novels
  (total_students : ℕ)
  (students_three_or_more_novels : ℕ)
  (students_two_novels : ℕ)
  (students_one_novel : ℕ)
  (h_total_students : total_students = 240)
  (h_students_three_or_more_novels : students_three_or_more_novels = 1 / 6 * 240)
  (h_students_two_novels : students_two_novels = 35 / 100 * 240)
  (h_students_one_novel : students_one_novel = 5 / 12 * 240)
  :
  total_students - (students_three_or_more_novels + students_two_novels + students_one_novel) = 16 :=
by
  sorry

end students_not_reading_novels_l1198_119853


namespace seq_sum_l1198_119824

theorem seq_sum (r : ℚ) (x y : ℚ) (h1 : r = 1 / 4)
    (h2 : 1024 * r = x) (h3 : x * r = y) : 
    x + y = 320 := by
  sorry

end seq_sum_l1198_119824


namespace abc_ineq_l1198_119809

theorem abc_ineq (a b c : ℝ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c ≥ 0) : 
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 := 
by 
  sorry

end abc_ineq_l1198_119809


namespace share_of_y_is_63_l1198_119839

theorem share_of_y_is_63 (x y z : ℝ) (h1 : y = 0.45 * x) (h2 : z = 0.50 * x) (h3 : x + y + z = 273) : y = 63 :=
by
  -- The proof will go here
  sorry

end share_of_y_is_63_l1198_119839


namespace range_of_a_l1198_119841

noncomputable def common_point_ellipse_parabola (a : ℝ) : Prop :=
  ∃ x y : ℝ, x^2 + 4 * (y - a)^2 = 4 ∧ x^2 = 2 * y

theorem range_of_a : ∀ a : ℝ, common_point_ellipse_parabola a → -1 ≤ a ∧ a ≤ 17 / 8 :=
by
  sorry

end range_of_a_l1198_119841


namespace measure_of_angle_B_l1198_119877

theorem measure_of_angle_B 
  (A B C: ℝ)
  (a b c: ℝ)
  (h1: A + B + C = π)
  (h2: B / A = C / B)
  (h3: b^2 - a^2 = a * c) : B = 2 * π / 7 :=
  sorry

end measure_of_angle_B_l1198_119877


namespace smallest_number_diminished_by_10_divisible_l1198_119843

theorem smallest_number_diminished_by_10_divisible :
  ∃ (x : ℕ), (x - 10) % 24 = 0 ∧ x = 34 :=
by
  sorry

end smallest_number_diminished_by_10_divisible_l1198_119843


namespace bob_km_per_gallon_l1198_119875

-- Define the total distance Bob can drive.
def total_distance : ℕ := 100

-- Define the total amount of gas in gallons Bob's car uses.
def total_gas : ℕ := 10

-- Define the expected kilometers per gallon
def expected_km_per_gallon : ℕ := 10

-- Define the statement we want to prove
theorem bob_km_per_gallon : total_distance / total_gas = expected_km_per_gallon :=
by 
  sorry

end bob_km_per_gallon_l1198_119875


namespace min_value_of_abc_l1198_119846

noncomputable def minimum_value_abc (a b c : ℝ) : Prop :=
  (a > 0) ∧ (b > 0) ∧ (c > 0) ∧ ((a + c) * (a + b) = 6 - 2 * Real.sqrt 5) → (2 * a + b + c ≥ 2 * Real.sqrt 5 - 2)

theorem min_value_of_abc (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : (a + c) * (a + b) = 6 - 2 * Real.sqrt 5) : 
  2 * a + b + c ≥ 2 * Real.sqrt 5 - 2 :=
by 
  sorry

end min_value_of_abc_l1198_119846


namespace grade_assignment_ways_l1198_119844

theorem grade_assignment_ways : (4^12 = 16777216) := 
by 
  sorry

end grade_assignment_ways_l1198_119844


namespace bus_problem_l1198_119833

-- Define the participants in 2005
def participants_2005 (k : ℕ) : ℕ := 27 * k + 19

-- Define the participants in 2006
def participants_2006 (k : ℕ) : ℕ := participants_2005 k + 53

-- Define the total number of buses needed in 2006
def buses_needed_2006 (k : ℕ) : ℕ := (participants_2006 k) / 27 + if (participants_2006 k) % 27 = 0 then 0 else 1

-- Define the total number of buses needed in 2005
def buses_needed_2005 (k : ℕ) : ℕ := k + 1

-- Define the additional buses needed in 2006 compared to 2005
def additional_buses_2006 (k : ℕ) := buses_needed_2006 k - buses_needed_2005 k

-- Define the number of people in the incomplete bus in 2006
def people_in_incomplete_bus_2006 (k : ℕ) := (participants_2006 k) % 27

-- The proof statement to be proved
theorem bus_problem (k : ℕ) : additional_buses_2006 k = 2 ∧ people_in_incomplete_bus_2006 k = 9 := by
  sorry

end bus_problem_l1198_119833


namespace train_passing_time_l1198_119811

theorem train_passing_time (length_of_train : ℝ) (speed_of_train_kmhr : ℝ) :
  length_of_train = 180 → speed_of_train_kmhr = 36 → (length_of_train / (speed_of_train_kmhr * (1000 / 3600))) = 18 :=
by
  intro h1 h2
  sorry

end train_passing_time_l1198_119811


namespace max_value_of_s_l1198_119867

theorem max_value_of_s (p q r s : ℝ) (h1 : p + q + r + s = 10)
  (h2 : p * q + p * r + p * s + q * r + q * s + r * s = 22) :
  s ≤ (5 + Real.sqrt 93) / 2 :=
sorry

end max_value_of_s_l1198_119867


namespace exactly_one_box_empty_count_l1198_119856

-- Define the setting with four different balls and four boxes.
def numberOfWaysExactlyOneBoxEmpty (balls : Finset ℕ) (boxes : Finset ℕ) : ℕ :=
  if (balls.card = 4 ∧ boxes.card = 4) then
     Nat.choose 4 2 * Nat.factorial 3
  else 0

theorem exactly_one_box_empty_count :
  numberOfWaysExactlyOneBoxEmpty {1, 2, 3, 4} {1, 2, 3, 4} = 144 :=
by
  -- The proof is omitted
  sorry

end exactly_one_box_empty_count_l1198_119856


namespace net_percentage_change_l1198_119816

-- Definitions based on given conditions
variables (P : ℝ) (P_post_decrease : ℝ) (P_post_increase : ℝ)

-- Conditions
def decreased_by_5_percent : Prop := P_post_decrease = P * (1 - 0.05)
def increased_by_10_percent : Prop := P_post_increase = P_post_decrease * (1 + 0.10)

-- Proof problem
theorem net_percentage_change (h1 : decreased_by_5_percent P P_post_decrease) (h2 : increased_by_10_percent P_post_decrease P_post_increase) : 
  ((P_post_increase - P) / P) * 100 = 4.5 :=
by
  -- The proof would go here
  sorry

end net_percentage_change_l1198_119816


namespace cost_of_stuffers_number_of_combinations_l1198_119819

noncomputable def candy_cane_cost : ℝ := 4 * 0.5
noncomputable def beanie_baby_cost : ℝ := 2 * 3
noncomputable def book_cost : ℝ := 5
noncomputable def toy_cost : ℝ := 3 * 1
noncomputable def gift_card_cost : ℝ := 10
noncomputable def one_child_stuffers_cost : ℝ := candy_cane_cost + beanie_baby_cost + book_cost + toy_cost + gift_card_cost
noncomputable def total_cost : ℝ := one_child_stuffers_cost * 4

def num_books := 5
def num_toys := 10
def toys_combinations : ℕ := Nat.choose num_toys 3
def total_combinations : ℕ := num_books * toys_combinations

theorem cost_of_stuffers (h : total_cost = 104) : total_cost = 104 := by
  sorry

theorem number_of_combinations (h : total_combinations = 600) : total_combinations = 600 := by
  sorry

end cost_of_stuffers_number_of_combinations_l1198_119819


namespace number_of_positive_real_solutions_l1198_119896

noncomputable def p (x : ℝ) : ℝ := x^12 + 5 * x^11 + 20 * x^10 + 1300 * x^9 - 1105 * x^8

theorem number_of_positive_real_solutions : ∃! x : ℝ, 0 < x ∧ p x = 0 :=
sorry

end number_of_positive_real_solutions_l1198_119896


namespace find_m_l1198_119887

-- Define the vectors a and b
def vec_a : ℝ × ℝ := (1, 2)
def vec_b (m : ℝ) : ℝ × ℝ := (2, m)

-- Define the addition of vectors
def vec_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 + v2.1, v1.2 + v2.2)

-- Define the dot product of vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := (v1.1 * v2.1) + (v1.2 * v2.2)

-- State the main theorem without proof
theorem find_m (m : ℝ) : dot_product (vec_add vec_a (vec_b m)) vec_a = 0 ↔ m = -7/2 := by
  sorry

end find_m_l1198_119887


namespace unique_prime_value_l1198_119842

def T : ℤ := 2161

theorem unique_prime_value :
  ∃ p : ℕ, (∃ n : ℤ, n^4 - 898 * n^2 + T - 2160 = p) ∧ Prime p ∧ (∀ q, (∃ n : ℤ, n^4 - 898 * n^2 + T - 2160 = q) → q = p) :=
  sorry

end unique_prime_value_l1198_119842


namespace time_to_reach_ship_l1198_119858

-- Define the conditions
def rate_of_descent := 30 -- feet per minute
def depth_to_ship := 2400 -- feet

-- Define the proof statement
theorem time_to_reach_ship : (depth_to_ship / rate_of_descent) = 80 :=
by
  -- The proof will be inserted here in practice
  sorry

end time_to_reach_ship_l1198_119858


namespace ming_estimate_less_l1198_119890

theorem ming_estimate_less (x y δ : ℝ) (h1 : x > y) (h2 : y > 0) (h3 : δ > 0) : 
  (x + δ) - (y + 2 * δ) < x - y :=
by 
  sorry

end ming_estimate_less_l1198_119890


namespace suit_cost_l1198_119810

theorem suit_cost :
  let shirt_cost := 15
  let pants_cost := 40
  let sweater_cost := 30
  let shirts := 4
  let pants := 2
  let sweaters := 2 
  let total_cost := shirts * shirt_cost + pants * pants_cost + sweaters * sweater_cost
  let discount_store := 0.80
  let discount_coupon := 0.90
  ∃ S, discount_coupon * discount_store * (total_cost + S) = 252 → S = 150 :=
by
  let shirt_cost := 15
  let pants_cost := 40
  let sweater_cost := 30
  let shirts := 4
  let pants := 2
  let sweaters := 2 
  let total_cost := shirts * shirt_cost + pants * pants_cost + sweaters * sweater_cost
  let discount_store := 0.80
  let discount_coupon := 0.90
  exists 150
  intro h
  sorry

end suit_cost_l1198_119810


namespace larger_jar_half_full_l1198_119821

-- Defining the capacities of the jars
variables (S L W : ℚ)

-- Conditions
def equal_amount_water (S L W : ℚ) : Prop :=
  W = (1/5 : ℚ) * S ∧ W = (1/4 : ℚ) * L

-- Question: What fraction will the larger jar be filled if the water from the smaller jar is added to it?
theorem larger_jar_half_full (S L W : ℚ) (h : equal_amount_water S L W) :
  (2 * W) / L = (1 / 2 : ℚ) :=
sorry

end larger_jar_half_full_l1198_119821


namespace eggs_per_hen_l1198_119878

theorem eggs_per_hen (total_chickens : ℕ) (num_roosters : ℕ) (non_laying_hens : ℕ) (total_eggs : ℕ) :
  total_chickens = 440 →
  num_roosters = 39 →
  non_laying_hens = 15 →
  total_eggs = 1158 →
  (total_eggs / (total_chickens - num_roosters - non_laying_hens) = 3) :=
by
  intros
  sorry

end eggs_per_hen_l1198_119878


namespace problem_l1198_119823

theorem problem (a b : ℝ) (h1 : a^2 - b^2 = 10) (h2 : a^4 + b^4 = 228) :
  a * b = 8 :=
sorry

end problem_l1198_119823


namespace min_value_of_expression_l1198_119883

open Real

theorem min_value_of_expression (x y z : ℝ) (h₀ : 0 < x) (h₁ : 0 < y) (h₂ : 0 < z) (h₃ : x * y * z = 18) :
  x^2 + 4*x*y + y^2 + 3*z^2 ≥ 63 :=
sorry

end min_value_of_expression_l1198_119883


namespace full_day_students_count_l1198_119854

-- Define the conditions
def total_students : ℕ := 80
def percentage_half_day_students : ℕ := 25

-- Define the statement to prove
theorem full_day_students_count :
  total_students - (total_students * percentage_half_day_students / 100) = 60 :=
by
  sorry

end full_day_students_count_l1198_119854


namespace loggers_count_l1198_119881

theorem loggers_count 
  (cut_rate : ℕ) 
  (forest_width : ℕ) 
  (forest_height : ℕ) 
  (tree_density : ℕ) 
  (days_per_month : ℕ) 
  (months : ℕ) 
  (total_loggers : ℕ)
  (total_trees : ℕ := forest_width * forest_height * tree_density) 
  (total_days : ℕ := days_per_month * months)
  (trees_cut_down_per_logger : ℕ := cut_rate * total_days) 
  (expected_loggers : ℕ := total_trees / trees_cut_down_per_logger) 
  (h1: cut_rate = 6)
  (h2: forest_width = 4)
  (h3: forest_height = 6)
  (h4: tree_density = 600)
  (h5: days_per_month = 30)
  (h6: months = 10)
  (h7: total_loggers = expected_loggers)
: total_loggers = 8 := 
by {
    sorry
}

end loggers_count_l1198_119881


namespace average_minutes_run_per_day_l1198_119865

theorem average_minutes_run_per_day (e : ℕ)
  (sixth_grade_avg : ℕ := 16)
  (seventh_grade_avg : ℕ := 18)
  (eighth_grade_avg : ℕ := 12)
  (sixth_graders : ℕ := 3 * e)
  (seventh_graders : ℕ := 2 * e)
  (eighth_graders : ℕ := e) :
  ((sixth_grade_avg * sixth_graders + seventh_grade_avg * seventh_graders + eighth_grade_avg * eighth_graders)
   / (sixth_graders + seventh_graders + eighth_graders) : ℕ) = 16 := 
by
  sorry

end average_minutes_run_per_day_l1198_119865


namespace perpendicular_tangent_lines_l1198_119876

def f (x : ℝ) : ℝ := x^3 + 1

noncomputable def tangent_line_eqs (x₀ : ℝ) (y₀ : ℝ) : Prop :=
  (3 * x₀ - y₀ - 1 = 0) ∨ (3 * x₀ - y₀ + 3 = 0)

theorem perpendicular_tangent_lines (x₀ : ℝ) (hx₀ : x₀ = 1 ∨ x₀ = -1) :
  tangent_line_eqs x₀ (f x₀) := by
  sorry

end perpendicular_tangent_lines_l1198_119876


namespace hyperbola_range_of_m_l1198_119814

theorem hyperbola_range_of_m (m : ℝ) : (∃ f : ℝ → ℝ → ℝ, ∀ x y: ℝ, f x y = (x^2 / (4 - m) - y^2 / (2 + m))) → (4 - m) * (2 + m) > 0 → -2 < m ∧ m < 4 :=
by
  intros h_eq h_cond
  sorry

end hyperbola_range_of_m_l1198_119814


namespace sphere_volume_equals_surface_area_l1198_119888

theorem sphere_volume_equals_surface_area (r : ℝ) (hr : r = 3) :
  (4 / 3) * π * r^3 = 4 * π * r^2 := by
  sorry

end sphere_volume_equals_surface_area_l1198_119888


namespace largest_among_abcd_l1198_119830

theorem largest_among_abcd (a b c d k : ℤ) (h : a - 1 = b + 2 ∧ b + 2 = c - 3 ∧ c - 3 = d + 4) :
  c = k + 3 ∧
  a = k + 1 ∧
  b = k - 2 ∧
  d = k - 4 ∧
  c > a ∧
  c > b ∧
  c > d :=
by
  sorry

end largest_among_abcd_l1198_119830


namespace Monica_saved_per_week_l1198_119832

theorem Monica_saved_per_week(amount_per_cycle : ℕ) (weeks_per_cycle : ℕ) (num_cycles : ℕ) (total_saved : ℕ) :
  num_cycles = 5 →
  weeks_per_cycle = 60 →
  (amount_per_cycle * num_cycles) = total_saved →
  total_saved = 4500 →
  total_saved / (weeks_per_cycle * num_cycles) = 75 := 
by
  intros
  sorry

end Monica_saved_per_week_l1198_119832


namespace find_b_l1198_119835

theorem find_b (a b c : ℚ) :
  -- Condition from the problem, equivalence of polynomials for all x
  ((4 : ℚ) * x^2 - 2 * x + 5 / 2) * (a * x^2 + b * x + c) =
    12 * x^4 - 8 * x^3 + 15 * x^2 - 5 * x + 5 / 2 →
  -- Given we found that a = 3 from the solution
  a = 3 →
  -- We need to prove that b = -1/2
  b = -1 / 2 :=
sorry

end find_b_l1198_119835


namespace exists_six_numbers_multiple_2002_l1198_119817

theorem exists_six_numbers_multiple_2002 (a : Fin 41 → ℕ) (h : Function.Injective a) :
  ∃ (i j k l m n : Fin 41),
    i ≠ j ∧ k ≠ l ∧ m ≠ n ∧
    (a i - a j) * (a k - a l) * (a m - a n) % 2002 = 0 := sorry

end exists_six_numbers_multiple_2002_l1198_119817


namespace f_zero_eq_zero_f_periodic_l1198_119813

def odd_function {α : Type*} [AddGroup α] (f : α → α) : Prop :=
∀ x, f (-x) = -f (x)

def symmetric_about (c : ℝ) (f : ℝ → ℝ) : Prop :=
∀ x, f (c + x) = f (c - x)

variable (f : ℝ → ℝ)
variables (h_odd : odd_function f) (h_sym : symmetric_about 1 f)

theorem f_zero_eq_zero : f 0 = 0 :=
sorry

theorem f_periodic : ∀ x, f (x + 4) = f x :=
sorry

end f_zero_eq_zero_f_periodic_l1198_119813


namespace sum_of_ages_l1198_119885

theorem sum_of_ages (rose_age mother_age : ℕ) (rose_age_eq : rose_age = 25) (mother_age_eq : mother_age = 75) : 
  rose_age + mother_age = 100 := 
by
  sorry

end sum_of_ages_l1198_119885


namespace total_students_l1198_119891

-- Definition of the problem conditions
def ratio_boys_girls : ℕ := 8
def ratio_girls : ℕ := 5
def number_girls : ℕ := 160

-- The main theorem statement
theorem total_students (b g : ℕ) (h1 : b * ratio_girls = g * ratio_boys_girls) (h2 : g = number_girls) :
  b + g = 416 :=
sorry

end total_students_l1198_119891


namespace monochromatic_triangle_in_K6_l1198_119882

theorem monochromatic_triangle_in_K6 :
  ∀ (color : Fin 6 → Fin 6 → Prop),
  (∀ (a b : Fin 6), a ≠ b → (color a b ↔ color b a)) →
  (∃ (x y z : Fin 6), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ (color x y = color y z ∧ color y z = color z x)) :=
by
  sorry

end monochromatic_triangle_in_K6_l1198_119882


namespace map_length_representation_l1198_119845

variable (x : ℕ)

theorem map_length_representation :
  (12 : ℕ) * x = 17 * (72 : ℕ) / 12
:=
sorry

end map_length_representation_l1198_119845


namespace no_valid_n_for_conditions_l1198_119879

theorem no_valid_n_for_conditions :
  ¬∃ n : ℕ, 1000 ≤ n / 4 ∧ n / 4 ≤ 9999 ∧ 1000 ≤ 4 * n ∧ 4 * n ≤ 9999 := by
  sorry

end no_valid_n_for_conditions_l1198_119879


namespace find_principal_sum_l1198_119889

theorem find_principal_sum
  (R : ℝ) (P : ℝ)
  (H1 : 0 < R)
  (H2 : 8 * 10 * P / 100 = 150) :
  P = 187.50 :=
by
  sorry

end find_principal_sum_l1198_119889


namespace zookeeper_fish_excess_l1198_119852

theorem zookeeper_fish_excess :
  let emperor_ratio := 3
  let adelie_ratio := 5
  let total_penguins := 48
  let total_ratio := emperor_ratio + adelie_ratio
  let emperor_penguins := (emperor_ratio / total_ratio) * total_penguins
  let adelie_penguins := (adelie_ratio / total_ratio) * total_penguins
  let emperor_fish_needed := emperor_penguins * 1.5
  let adelie_fish_needed := adelie_penguins * 2
  let total_fish_needed := emperor_fish_needed + adelie_fish_needed
  let fish_zookeeper_has := total_penguins * 2.5
  (fish_zookeeper_has - total_fish_needed = 33) :=
  
by {
  sorry
}

end zookeeper_fish_excess_l1198_119852


namespace product_calculation_l1198_119859

theorem product_calculation :
  12 * 0.5 * 3 * 0.2 * 5 = 18 := by
  sorry

end product_calculation_l1198_119859


namespace correct_propositions_count_l1198_119880

theorem correct_propositions_count (a b : ℝ) :
  (∀ a b, a > b → a + 1 > b + 1) ∧
  (∀ a b, a > b → a - 1 > b - 1) ∧
  (∀ a b, a > b → -2 * a < -2 * b) ∧
  (¬ ∀ a b, a > b → 2 * a < 2 * b) → 
  3 = 3 :=
by
  intro h
  sorry

end correct_propositions_count_l1198_119880


namespace inner_ring_speed_minimum_train_distribution_l1198_119893

theorem inner_ring_speed_minimum
  (l_inner : ℝ) (num_trains_inner : ℕ) (max_wait_inner : ℝ) (speed_min : ℝ) :
  l_inner = 30 →
  num_trains_inner = 9 →
  max_wait_inner = 10 →
  speed_min = 20 :=
by 
  sorry

theorem train_distribution
  (l_inner : ℝ) (speed_inner : ℝ) (speed_outer : ℝ) (total_trains : ℕ) (max_wait_diff : ℝ) (trains_inner : ℕ) (trains_outer : ℕ) :
  l_inner = 30 →
  speed_inner = 25 →
  speed_outer = 30 →
  total_trains = 18 →
  max_wait_diff = 1 →
  trains_inner = 10 →
  trains_outer = 8 :=
by 
  sorry

end inner_ring_speed_minimum_train_distribution_l1198_119893


namespace rachel_bella_total_distance_l1198_119874

theorem rachel_bella_total_distance:
  ∀ (distance_land distance_sea total_distance: ℕ), 
  distance_land = 451 → 
  distance_sea = 150 → 
  total_distance = distance_land + distance_sea → 
  total_distance = 601 := 
by 
  intros distance_land distance_sea total_distance h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end rachel_bella_total_distance_l1198_119874


namespace mixture_alcohol_quantity_l1198_119805

theorem mixture_alcohol_quantity:
  ∀ (A W : ℝ), 
    A / W = 4 / 3 ∧ A / (W + 7) = 4 / 5 → A = 14 :=
by
  intros A W h
  sorry

end mixture_alcohol_quantity_l1198_119805


namespace curve_cross_intersection_l1198_119812

theorem curve_cross_intersection : 
  ∃ (t_a t_b : ℝ), t_a ≠ t_b ∧ 
  (3 * t_a^2 + 1 = 3 * t_b^2 + 1) ∧
  (t_a^3 - 6 * t_a^2 + 4 = t_b^3 - 6 * t_b^2 + 4) ∧
  (3 * t_a^2 + 1 = 109 ∧ t_a^3 - 6 * t_a^2 + 4 = -428) := by
  sorry

end curve_cross_intersection_l1198_119812


namespace intersection_A_B_l1198_119898

def A : Set ℝ := {x | x > 3}
def B : Set ℝ := {x | (x - 1) / (x - 4) < 0}
def inter : Set ℝ := {x | 3 < x ∧ x < 4}

theorem intersection_A_B : A ∩ B = inter := 
by 
  sorry

end intersection_A_B_l1198_119898


namespace final_hair_length_l1198_119868

-- Define the initial conditions and the expected final result.
def initial_hair_length : ℕ := 14
def hair_growth (x : ℕ) : ℕ := x
def hair_cut : ℕ := 20

-- Prove that the final hair length is x - 6.
theorem final_hair_length (x : ℕ) : initial_hair_length + hair_growth x - hair_cut = x - 6 :=
by
  sorry

end final_hair_length_l1198_119868


namespace scientific_notation_of_12_06_million_l1198_119847

theorem scientific_notation_of_12_06_million :
  12.06 * 10^6 = 1.206 * 10^7 :=
sorry

end scientific_notation_of_12_06_million_l1198_119847


namespace inequality_proof_l1198_119826

theorem inequality_proof (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_sum : a + b + c = 3) :
  (ab / Real.sqrt (c^2 + 3)) + (bc / Real.sqrt (a^2 + 3)) + (ca / Real.sqrt (b^2 + 3)) ≤ 3 / 2 :=
by
  sorry

end inequality_proof_l1198_119826


namespace smallest_square_area_l1198_119831

variable (M N : ℝ)

/-- Given that the largest square has an area of 1 cm^2, the middle square has an area M cm^2, and the smallest square has a vertex on the side of the middle square, prove that the area of the smallest square N is equal to ((1 - M) / 2)^2. -/
theorem smallest_square_area (h1 : 1 ≥ 0)
  (h2 : 0 ≤ M ∧ M ≤ 1)
  (h3 : 0 ≤ N) :
  N = (1 - M) ^ 2 / 4 := sorry

end smallest_square_area_l1198_119831


namespace diet_sodas_sold_l1198_119871

theorem diet_sodas_sold (R D : ℕ) (h1 : R + D = 64) (h2 : R / D = 9 / 7) : D = 28 := 
by
  sorry

end diet_sodas_sold_l1198_119871


namespace solve_circle_tangent_and_intercept_l1198_119840

namespace CircleProblems

-- Condition: Circle equation
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 2 * x - 4 * y + 3 = 0

-- Problem 1: Equations of tangent lines with equal intercepts
def tangent_lines_with_equal_intercepts (l : ℝ → ℝ → Prop) : Prop :=
  (∀ x y : ℝ, l x y ↔ x + y + 1 = 0) ∨ (∀ x y : ℝ, l x y ↔ x + y - 3 = 0)

-- Problem 2: Equations of lines passing through origin and intercepted by the circle with a segment length of 2
def lines_intercepted_by_circle (l : ℝ → ℝ → Prop) : Prop :=
  (∀ x y : ℝ, l x y ↔ x = 0) ∨ (∀ x y : ℝ, l x y ↔ y = - (3 / 4) * x)

theorem solve_circle_tangent_and_intercept (l_tangent l_origin : ℝ → ℝ → Prop) :
  (∀ x y : ℝ, circle_eq x y → l_tangent x y) →
  tangent_lines_with_equal_intercepts l_tangent ∧ lines_intercepted_by_circle l_origin :=
by
  sorry

end CircleProblems

end solve_circle_tangent_and_intercept_l1198_119840


namespace square_of_other_leg_l1198_119873

variable {R : Type} [CommRing R]

theorem square_of_other_leg (a b c : R) (h1 : a^2 + b^2 = c^2) (h2 : c = a + 2) : b^2 = 4 * a + 4 :=
by
  sorry

end square_of_other_leg_l1198_119873


namespace marys_final_amount_l1198_119872

def simple_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  P * r * t

def final_amount (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  P + simple_interest P r t

theorem marys_final_amount 
  (P : ℝ := 200)
  (A_after_2_years : ℝ := 260)
  (t1 : ℝ := 2)
  (t2 : ℝ := 6)
  (r : ℝ := (A_after_2_years - P) / (P * t1)) :
  final_amount P r t2 = 380 := 
by
  sorry

end marys_final_amount_l1198_119872


namespace infinite_points_on_line_with_positive_rational_coordinates_l1198_119884

theorem infinite_points_on_line_with_positive_rational_coordinates :
  ∃ (S : Set (ℚ × ℚ)), (∀ p ∈ S, p.1 + p.2 = 4 ∧ 0 < p.1 ∧ 0 < p.2) ∧ S.Infinite :=
sorry

end infinite_points_on_line_with_positive_rational_coordinates_l1198_119884


namespace model_A_sampling_l1198_119820

theorem model_A_sampling (prod_A prod_B prod_C total_prod total_sampled : ℕ)
    (hA : prod_A = 1200) (hB : prod_B = 6000) (hC : prod_C = 2000)
    (htotal : total_prod = prod_A + prod_B + prod_C) (htotal_car : total_prod = 9200)
    (hsampled : total_sampled = 46) :
    (prod_A * total_sampled) / total_prod = 6 := by
  sorry

end model_A_sampling_l1198_119820


namespace final_number_l1198_119855

variables (crab goat bear cat hen : ℕ)

-- Given conditions
def row4_sum : Prop := 5 * crab = 10
def col5_sum : Prop := 4 * crab + goat = 11
def row2_sum : Prop := 2 * goat + crab + 2 * bear = 16
def col2_sum : Prop := cat + bear + 2 * goat + crab = 13
def col3_sum : Prop := 2 * crab + 2 * hen + goat = 17

-- Theorem statement
theorem final_number
  (hcrab : row4_sum crab)
  (hgoat_col5 : col5_sum crab goat)
  (hbear_row2 : row2_sum crab goat bear)
  (hcat_col2 : col2_sum cat crab bear goat)
  (hhen_col3 : col3_sum crab goat hen) :
  crab = 2 ∧ goat = 3 ∧ bear = 4 ∧ cat = 1 ∧ hen = 5 → (cat * 10000 + hen * 1000 + crab * 100 + bear * 10 + goat = 15243) :=
sorry

end final_number_l1198_119855


namespace average_of_c_and_d_l1198_119800

variable (c d e : ℝ)

theorem average_of_c_and_d
  (h1: (4 + 6 + 9 + c + d + e) / 6 = 20)
  (h2: e = c + 6) :
  (c + d) / 2 = 47.5 := by
sorry

end average_of_c_and_d_l1198_119800


namespace old_lamp_height_is_one_l1198_119802

def new_lamp_height : ℝ := 2.3333333333333335
def height_difference : ℝ := 1.3333333333333333
def old_lamp_height : ℝ := new_lamp_height - height_difference

theorem old_lamp_height_is_one :
  old_lamp_height = 1 :=
by
  sorry

end old_lamp_height_is_one_l1198_119802


namespace speed_with_current_l1198_119886

theorem speed_with_current (v : ℝ) (current_speed : ℝ) (against_current_speed : ℝ) 
  (h1 : current_speed = 2.8) 
  (h2 : against_current_speed = 9.4) 
  (h3 : against_current_speed = v - current_speed) 
  : (v + current_speed) = 15 := by
  sorry

end speed_with_current_l1198_119886


namespace ellipse_focus_value_k_l1198_119894

theorem ellipse_focus_value_k 
  (k : ℝ)
  (h : ∀ x y, 5 * x^2 + k * y^2 = 5 → abs y ≠ 2 → ∀ c : ℝ, c^2 = 4 → k = 1) :
  ∀ k : ℝ, (5 * (0:ℝ)^2 + k * (2:ℝ)^2 = 5) ∧ (5 * (0:ℝ)^2 + k * (-(2:ℝ))^2 = 5) → k = 1 := by
  sorry

end ellipse_focus_value_k_l1198_119894


namespace math_problem_l1198_119806

theorem math_problem 
  (x y : ℝ) 
  (h1 : 4 * x + y = 12) 
  (h2 : x + 4 * y = 18) : 
  20 * x^2 + 24 * x * y + 20 * y^2 = 468 := 
by
  sorry

end math_problem_l1198_119806


namespace Danny_shorts_washed_l1198_119895

-- Define the given conditions
def Cally_white_shirts : ℕ := 10
def Cally_colored_shirts : ℕ := 5
def Cally_shorts : ℕ := 7
def Cally_pants : ℕ := 6

def Danny_white_shirts : ℕ := 6
def Danny_colored_shirts : ℕ := 8
def Danny_pants : ℕ := 6

def total_clothes_washed : ℕ := 58

-- Calculate total clothes washed by Cally
def total_cally_clothes : ℕ := 
  Cally_white_shirts + Cally_colored_shirts + Cally_shorts + Cally_pants

-- Calculate total clothes washed by Danny (excluding shorts)
def total_danny_clothes_excl_shorts : ℕ := 
  Danny_white_shirts + Danny_colored_shirts + Danny_pants

-- Define the statement to be proven
theorem Danny_shorts_washed : 
  total_clothes_washed - (total_cally_clothes + total_danny_clothes_excl_shorts) = 10 := by
  sorry

end Danny_shorts_washed_l1198_119895


namespace evaluate_expression_l1198_119834

theorem evaluate_expression : 
  (1 / 2 + ((2 / 3 * (3 / 8)) + 4) - (8 / 16)) = (17 / 4) :=
by
  sorry

end evaluate_expression_l1198_119834


namespace Flora_initial_daily_milk_l1198_119862

def total_gallons : ℕ := 105
def total_weeks : ℕ := 3
def days_per_week : ℕ := 7
def total_days : ℕ := total_weeks * days_per_week
def extra_gallons_daily : ℕ := 2

theorem Flora_initial_daily_milk : 
  (total_gallons / total_days) = 5 := by
  sorry

end Flora_initial_daily_milk_l1198_119862


namespace partition_impossible_l1198_119897

def sum_of_list (l : List Int) : Int := l.foldl (· + ·) 0

theorem partition_impossible
  (l : List Int)
  (h : l = [-7, -4, -2, 3, 5, 9, 10, 18, 21, 33])
  (total_sum : Int := sum_of_list l)
  (target_diff : Int := 9) :
  ¬∃ (l1 l2 : List Int), 
    (l1 ++ l2 = l ∧ 
     sum_of_list l1 - sum_of_list l2 = target_diff ∧
     total_sum  = 86) := 
sorry

end partition_impossible_l1198_119897


namespace find_other_endpoint_l1198_119848

theorem find_other_endpoint (mx my x₁ y₁ x₂ y₂ : ℤ) 
  (h1 : mx = (x₁ + x₂) / 2) 
  (h2 : my = (y₁ + y₂) / 2) 
  (h3 : mx = 3) 
  (h4 : my = 4) 
  (h5 : x₁ = -2) 
  (h6 : y₁ = -5) : 
  x₂ = 8 ∧ y₂ = 13 := 
by
  sorry

end find_other_endpoint_l1198_119848


namespace find_value_l1198_119866

noncomputable def roots_of_equation (a b c : ℝ) : Prop :=
  10 * a^3 + 502 * a + 3010 = 0 ∧
  10 * b^3 + 502 * b + 3010 = 0 ∧
  10 * c^3 + 502 * c + 3010 = 0

theorem find_value (a b c : ℝ)
  (h : roots_of_equation a b c) :
  (a + b)^3 + (b + c)^3 + (c + a)^3 = 903 :=
by
  sorry

end find_value_l1198_119866


namespace inequality_solution_l1198_119825

theorem inequality_solution (x : ℝ) : (x - 1) / 3 > 2 → x > 7 :=
by
  intros h
  sorry

end inequality_solution_l1198_119825


namespace total_dresses_l1198_119818

theorem total_dresses (E M D S: ℕ) 
  (h1 : D = M + 12)
  (h2 : M = E / 2)
  (h3 : E = 16)
  (h4 : S = D - 5) : 
  E + M + D + S = 59 :=
by
  sorry

end total_dresses_l1198_119818


namespace smallest_sphere_radius_l1198_119804

theorem smallest_sphere_radius :
  ∃ (R : ℝ), (∀ (a b : ℝ), a = 14 → b = 12 → ∃ (h : ℝ), h = Real.sqrt (12^2 - (14 * Real.sqrt 2 / 2)^2) ∧ R = 7 * Real.sqrt 2 ∧ h ≤ R) :=
sorry

end smallest_sphere_radius_l1198_119804


namespace total_surfers_l1198_119837

theorem total_surfers (num_surfs_santa_monica : ℝ) (ratio_malibu : ℝ) (ratio_santa_monica : ℝ) (ratio_venice : ℝ) (ratio_huntington : ℝ) (ratio_newport : ℝ) :
    num_surfs_santa_monica = 36 ∧ ratio_malibu = 7 ∧ ratio_santa_monica = 4.5 ∧ ratio_venice = 3.5 ∧ ratio_huntington = 2 ∧ ratio_newport = 1.5 →
    (ratio_malibu * (num_surfs_santa_monica / ratio_santa_monica) +
     num_surfs_santa_monica +
     ratio_venice * (num_surfs_santa_monica / ratio_santa_monica) +
     ratio_huntington * (num_surfs_santa_monica / ratio_santa_monica) +
     ratio_newport * (num_surfs_santa_monica / ratio_santa_monica)) = 148 :=
by
  sorry

end total_surfers_l1198_119837


namespace all_three_use_media_l1198_119850

variable (U T R M T_and_M T_and_R R_and_M T_and_R_and_M : ℕ)

theorem all_three_use_media (hU : U = 180)
  (hT : T = 115)
  (hR : R = 110)
  (hM : M = 130)
  (hT_and_M : T_and_M = 85)
  (hT_and_R : T_and_R = 75)
  (hR_and_M : R_and_M = 95)
  (h_union : U = T + R + M - T_and_R - T_and_M - R_and_M + T_and_R_and_M) :
  T_and_R_and_M = 80 :=
by
  sorry

end all_three_use_media_l1198_119850


namespace corrected_mean_l1198_119838

theorem corrected_mean (n : ℕ) (mean : ℝ) (obs1 obs2 : ℝ) (inc1 inc2 cor1 cor2 : ℝ)
    (h_num_obs : n = 50)
    (h_initial_mean : mean = 36)
    (h_incorrect1 : inc1 = 23) (h_correct1 : cor1 = 34)
    (h_incorrect2 : inc2 = 55) (h_correct2 : cor2 = 45)
    : (mean * n + (cor1 - inc1) + (cor2 - inc2)) / n = 36.02 := 
by 
  -- Insert steps to prove the theorem here
  sorry

end corrected_mean_l1198_119838


namespace number_of_children_l1198_119849

theorem number_of_children (x : ℕ) : 3 * x + 12 = 5 * x - 10 → x = 11 :=
by
  intros h
  have : 3 * x + 12 = 5 * x - 10 := h
  sorry

end number_of_children_l1198_119849


namespace find_A_for_club_suit_l1198_119857

def club_suit (A B : ℝ) : ℝ := 3 * A + 2 * B^2 + 5

theorem find_A_for_club_suit :
  ∃ A : ℝ, club_suit A 3 = 73 ∧ A = 50 / 3 :=
sorry

end find_A_for_club_suit_l1198_119857
