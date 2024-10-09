import Mathlib

namespace final_price_is_correct_l1606_160687

-- Define the original price and percentages as constants
def original_price : ℝ := 160
def increase_percentage : ℝ := 0.25
def discount_percentage : ℝ := 0.25

-- Calculate increased price
def increased_price : ℝ := original_price * (1 + increase_percentage)
-- Calculate the discount on the increased price
def discount_amount : ℝ := increased_price * discount_percentage
-- Calculate final price after discount
def final_price : ℝ := increased_price - discount_amount

-- Statement of the theorem: prove final price is $150
theorem final_price_is_correct : final_price = 150 :=
by
  -- Proof would go here
  sorry

end final_price_is_correct_l1606_160687


namespace express_y_in_terms_of_x_l1606_160615

theorem express_y_in_terms_of_x (x y : ℝ) (h : 2 * x - y = 4) : y = 2 * x - 4 :=
by
  sorry

end express_y_in_terms_of_x_l1606_160615


namespace two_digit_numbers_condition_l1606_160613

theorem two_digit_numbers_condition : ∃ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧
    10 * a + b ≥ 10 ∧ 10 * a + b ≤ 99 ∧
    (10 * a + b) / (a + b) = (a + b) / 3 ∧ 
    (10 * a + b = 27 ∨ 10 * a + b = 48) := 
by
    sorry

end two_digit_numbers_condition_l1606_160613


namespace B_finish_work_in_10_days_l1606_160653

variable (W : ℝ) -- amount of work
variable (x : ℝ) -- number of days B can finish the work alone

theorem B_finish_work_in_10_days (h1 : ∀ A_rate, A_rate = W / 4)
                                (h2 : ∀ B_rate, B_rate = W / x)
                                (h3 : ∀ Work_done_together Remaining_work,
                                      Work_done_together = 2 * (W / 4 + W / x) ∧
                                      Remaining_work = W - Work_done_together ∧
                                      Remaining_work = (W / x) * 3.0000000000000004) :
  x = 10 :=
by
  sorry

end B_finish_work_in_10_days_l1606_160653


namespace air_quality_conditional_prob_l1606_160647

theorem air_quality_conditional_prob :
  let p1 := 0.8
  let p2 := 0.68
  let p := p2 / p1
  p = 0.85 :=
by
  sorry

end air_quality_conditional_prob_l1606_160647


namespace trader_marked_price_percentage_above_cost_price_l1606_160690

theorem trader_marked_price_percentage_above_cost_price 
  (CP MP SP : ℝ) 
  (discount loss : ℝ)
  (h_discount : discount = 0.07857142857142857)
  (h_loss : loss = 0.01)
  (h_SP_discount : SP = MP * (1 - discount))
  (h_SP_loss : SP = CP * (1 - loss)) :
  (MP / CP - 1) * 100 = 7.4285714285714 := 
sorry

end trader_marked_price_percentage_above_cost_price_l1606_160690


namespace vika_made_84_dollars_l1606_160667

-- Define the amount of money Saheed, Kayla, and Vika made
variable (S K V : ℕ)

-- Given conditions
def condition1 : Prop := S = 4 * K
def condition2 : Prop := K = V - 30
def condition3 : Prop := S = 216

-- Statement to prove
theorem vika_made_84_dollars (S K V : ℕ) (h1 : condition1 S K) (h2 : condition2 K V) (h3 : condition3 S) : 
  V = 84 :=
by sorry

end vika_made_84_dollars_l1606_160667


namespace candidate_percentage_l1606_160631

variables (P candidate_votes rival_votes total_votes : ℝ)

-- Conditions
def candidate_lost_by_2460 (candidate_votes rival_votes : ℝ) : Prop :=
  rival_votes = candidate_votes + 2460

def total_votes_cast (candidate_votes rival_votes total_votes : ℝ) : Prop :=
  candidate_votes + rival_votes = total_votes

-- Proof problem
theorem candidate_percentage (h1 : candidate_lost_by_2460 candidate_votes rival_votes)
                             (h2 : total_votes_cast candidate_votes rival_votes 8200) :
  P = 35 :=
sorry

end candidate_percentage_l1606_160631


namespace equal_circumradii_l1606_160699

-- Define the points and triangles involved
variable (A B C M : Type*) 

-- The circumcircle radius of a triangle is at least R
variable (R R1 R2 R3 : ℝ)

-- Hypotheses: the given conditions
variable (hR1 : R1 ≥ R)
variable (hR2 : R2 ≥ R)
variable (hR3 : R3 ≥ R)

-- The goal: to show that all four radii are equal
theorem equal_circumradii {A B C M : Type*} (R R1 R2 R3 : ℝ) 
    (hR1 : R1 ≥ R) 
    (hR2 : R2 ≥ R) 
    (hR3 : R3 ≥ R): 
    R1 = R ∧ R2 = R ∧ R3 = R := 
by 
  sorry

end equal_circumradii_l1606_160699


namespace find_sum_l1606_160657

theorem find_sum 
  (R : ℝ) -- Original interest rate
  (P : ℝ) -- Principal amount
  (h: (P * (R + 3) * 3 / 100) = ((P * R * 3 / 100) + 81)): 
  P = 900 :=
sorry

end find_sum_l1606_160657


namespace cube_volume_l1606_160696

theorem cube_volume (a : ℕ) (h1 : 9 * 12 * 3 = 324) (h2 : 108 * a^3 = 324) : a^3 = 27 :=
by {
  sorry
}

end cube_volume_l1606_160696


namespace larger_of_two_numbers_l1606_160664

theorem larger_of_two_numbers (A B : ℕ) (HCF : ℕ) (factor1 factor2 : ℕ) (h_hcf : HCF = 23) (h_factor1 : factor1 = 13) (h_factor2 : factor2 = 14)
(hA : A = HCF * factor1) (hB : B = HCF * factor2) :
  max A B = 322 :=
by
  sorry

end larger_of_two_numbers_l1606_160664


namespace octagon_perimeter_correct_l1606_160601

def octagon_perimeter (n : ℕ) (side_length : ℝ) : ℝ :=
  n * side_length

theorem octagon_perimeter_correct :
  octagon_perimeter 8 3 = 24 :=
by
  sorry

end octagon_perimeter_correct_l1606_160601


namespace initial_leaves_l1606_160610

theorem initial_leaves (l_0 : ℕ) (blown_away : ℕ) (leaves_left : ℕ) (h1 : blown_away = 244) (h2 : leaves_left = 112) (h3 : l_0 - blown_away = leaves_left) : l_0 = 356 :=
by
  sorry

end initial_leaves_l1606_160610


namespace count_triangles_l1606_160694

-- Define the conditions for the problem
def P (x1 x2 : ℕ) : Prop := 37 * x1 ≤ 2022 ∧ 37 * x2 ≤ 2022

def valid_points (x y : ℕ) : Prop := 37 * x + y = 2022

def area_multiple_of_3 (x1 x2 : ℕ): Prop :=
  (∃ k : ℤ, 3 * k = x1 - x2) ∧ x1 ≠ x2 ∧ P x1 x2

-- The final theorem to prove the number of such distinct triangles
theorem count_triangles : 
  (∃ (n : ℕ), n = 459 ∧ 
    ∃ x1 x2 : ℕ, area_multiple_of_3 x1 x2 ∧ x1 ≠ x2) :=
by
  sorry

end count_triangles_l1606_160694


namespace pizza_problem_l1606_160637

theorem pizza_problem
  (pizza_slices : ℕ)
  (total_pizzas : ℕ)
  (total_people : ℕ)
  (pepperoni_only_friend : ℕ)
  (remaining_pepperoni : ℕ)
  (equal_distribution : Prop)
  (h_cond1 : pizza_slices = 16)
  (h_cond2 : total_pizzas = 2)
  (h_cond3 : total_people = 4)
  (h_cond4 : pepperoni_only_friend = 1)
  (h_cond5 : remaining_pepperoni = 1)
  (h_cond6 : equal_distribution ∧ (pepperoni_only_friend ≤ total_people)) :
  ∃ cheese_slices_left : ℕ, cheese_slices_left = 7 := by
  sorry

end pizza_problem_l1606_160637


namespace largest_divisor_of_n_squared_divisible_by_72_l1606_160665

theorem largest_divisor_of_n_squared_divisible_by_72
    (n : ℕ) (h1 : n > 0) (h2 : 72 ∣ n^2) : 12 ∣ n :=
by {
    sorry
}

end largest_divisor_of_n_squared_divisible_by_72_l1606_160665


namespace eccentricity_of_ellipse_l1606_160692

noncomputable def ellipse (a b c : ℝ) :=
  (a > b) ∧ (b > 0) ∧ (a^2 = b^2 + c^2) ∧ (b = 2 * c)

theorem eccentricity_of_ellipse (a b c : ℝ) (h : ellipse a b c) :
  (c / a = Real.sqrt 5 / 5) :=
by
  sorry

end eccentricity_of_ellipse_l1606_160692


namespace sets_equal_l1606_160648

theorem sets_equal (M N : Set ℝ) (hM : M = { x | x^2 = 1 }) (hN : N = { a | ∀ x ∈ M, a * x = 1 }) : M = N :=
sorry

end sets_equal_l1606_160648


namespace sandy_savings_last_year_l1606_160661

theorem sandy_savings_last_year (S : ℝ) (P : ℝ) 
(h1 : P / 100 * S = x)
(h2 : 1.10 * S = y)
(h3 : 0.10 * y = 0.11 * S)
(h4 : 0.11 * S = 1.8333333333333331 * x) :
P = 6 := by
  -- proof goes here
  sorry

end sandy_savings_last_year_l1606_160661


namespace find_product_of_abc_l1606_160660

theorem find_product_of_abc :
  ∃ (a b c m : ℝ), 
    a + b + c = 195 ∧
    m = 8 * a ∧
    m = b - 10 ∧
    m = c + 10 ∧
    a * b * c = 95922 := by
  sorry

end find_product_of_abc_l1606_160660


namespace average_speed_is_65_l1606_160669

-- Definitions based on the problem's conditions
def speed_first_hour : ℝ := 100 -- 100 km in the first hour
def speed_second_hour : ℝ := 30 -- 30 km in the second hour
def total_distance : ℝ := speed_first_hour + speed_second_hour -- total distance
def total_time : ℝ := 2 -- total time in hours (1 hour + 1 hour)

-- Problem: prove that the average speed is 65 km/h
theorem average_speed_is_65 : (total_distance / total_time) = 65 := by
  sorry

end average_speed_is_65_l1606_160669


namespace sequence_x_value_l1606_160646

theorem sequence_x_value (p q r x : ℕ) 
  (h1 : 13 = 5 + p + q) 
  (h2 : r = p + q + 13) 
  (h3 : x = 13 + r + 40) : 
  x = 74 := 
by 
  sorry

end sequence_x_value_l1606_160646


namespace window_ratio_area_l1606_160686

/-- Given a rectangle with semicircles at either end, if the ratio of AD to AB is 3:2,
    and AB is 30 inches, then the ratio of the area of the rectangle to the combined 
    area of the semicircles is 6 : π. -/
theorem window_ratio_area (AD AB r : ℝ) (h1 : AB = 30) (h2 : AD / AB = 3 / 2) (h3 : r = AB / 2) :
    (AD * AB) / (π * r^2) = 6 / π :=
by
  sorry

end window_ratio_area_l1606_160686


namespace discs_contain_equal_minutes_l1606_160658

theorem discs_contain_equal_minutes (total_time discs_capacity : ℕ) 
  (h1 : total_time = 520) (h2 : discs_capacity = 65) :
  ∃ discs_needed : ℕ, discs_needed = total_time / discs_capacity ∧ 
  ∀ (k : ℕ), k = total_time / discs_needed → k = 65 :=
by
  sorry

end discs_contain_equal_minutes_l1606_160658


namespace spanish_peanuts_l1606_160629

variable (x : ℝ)

theorem spanish_peanuts :
  (10 * 3.50 + x * 3.00 = (10 + x) * 3.40) → x = 2.5 :=
by
  intro h
  sorry

end spanish_peanuts_l1606_160629


namespace length_of_first_train_is_270_l1606_160605

/-- 
Given:
1. Speed of the first train = 120 kmph
2. Speed of the second train = 80 kmph
3. Time to cross each other = 9 seconds
4. Length of the second train = 230.04 meters
  
Prove that the length of the first train is 270 meters.
-/
theorem length_of_first_train_is_270
  (speed_first_train : ℝ := 120)
  (speed_second_train : ℝ := 80)
  (time_to_cross : ℝ := 9)
  (length_second_train : ℝ := 230.04)
  (conversion_factor : ℝ := 1000/3600) :
  (length_second_train + (speed_first_train + speed_second_train) * conversion_factor * time_to_cross - length_second_train) = 270 :=
by
  sorry

end length_of_first_train_is_270_l1606_160605


namespace max_product_l1606_160621

theorem max_product (a b : ℕ) (h1: a + b = 100) 
    (h2: a % 3 = 2) (h3: b % 7 = 5) : a * b ≤ 2491 := by
  sorry

end max_product_l1606_160621


namespace factorize_expression_l1606_160697

variable (a b : ℝ)

theorem factorize_expression : a^2 - 4 * b^2 - 2 * a + 4 * b = (a + 2 * b - 2) * (a - 2 * b) := 
  sorry

end factorize_expression_l1606_160697


namespace expression_value_l1606_160627

theorem expression_value (a : ℝ) (h_nonzero : a ≠ 0) (h_ne_two : a ≠ 2) (h_ne_neg_two : a ≠ -2) (h_ne_neg_one : a ≠ -1) (h_eq_one : a = 1) :
  1 - (((a-2)/a) / ((a^2-4)/(a^2+a))) = 1 / 3 :=
by
  sorry

end expression_value_l1606_160627


namespace initial_amount_liquid_A_l1606_160639

theorem initial_amount_liquid_A (A B : ℝ) (h1 : A / B = 4)
    (h2 : (A / (B + 40)) = 2 / 3) : A = 32 := by
  sorry

end initial_amount_liquid_A_l1606_160639


namespace range_values_for_a_l1606_160693

def p (x : ℝ) : Prop := x^2 - 8 * x - 20 ≤ 0
def q (x a : ℝ) (ha : 0 < a) : Prop := x^2 - 2 * x + 1 - a^2 ≥ 0

theorem range_values_for_a (a : ℝ) : (∃ ha : 0 < a, (∀ x : ℝ, (¬ p x → q x a ha))) → (0 < a ∧ a ≤ 3) :=
by
  sorry

end range_values_for_a_l1606_160693


namespace part1_l1606_160675

theorem part1 (a b : ℝ) (h1 : a ≠ b) (h2 : a^2 ≠ b^2) :
  (a^2 + a * b + b^2) / (a + b) - (a^2 - a * b + b^2) / (a - b) + (2 * b^2 - b^2 + a^2) / (a^2 - b^2) = 1 := 
sorry

end part1_l1606_160675


namespace average_salary_all_workers_l1606_160666

/-- The total number of workers in the workshop is 15 -/
def total_number_of_workers : ℕ := 15

/-- The number of technicians is 5 -/
def number_of_technicians : ℕ := 5

/-- The number of other workers is given by the total number minus technicians -/
def number_of_other_workers : ℕ := total_number_of_workers - number_of_technicians

/-- The average salary per head of the technicians is Rs. 800 -/
def average_salary_per_technician : ℕ := 800

/-- The average salary per head of the other workers is Rs. 650 -/
def average_salary_per_other_worker : ℕ := 650

/-- The total salary for all the workers -/
def total_salary : ℕ := (number_of_technicians * average_salary_per_technician) + (number_of_other_workers * average_salary_per_other_worker)

/-- The average salary per head of all the workers in the workshop is Rs. 700 -/
theorem average_salary_all_workers :
  total_salary / total_number_of_workers = 700 := by
  sorry

end average_salary_all_workers_l1606_160666


namespace value_of_expression_l1606_160663

theorem value_of_expression (x y z : ℤ) (h1 : x = -3) (h2 : y = 5) (h3 : z = -4) :
  x^2 + y^2 - z^2 + 2*x*y = -12 :=
by
  -- proof goes here
  sorry

end value_of_expression_l1606_160663


namespace minimize_xy_l1606_160680

theorem minimize_xy (x y : ℕ) (hx : x > 0) (hy : y > 0) (h_eq : 7 * x + 4 * y = 200) : (x * y = 172) :=
sorry

end minimize_xy_l1606_160680


namespace steve_distance_l1606_160679

theorem steve_distance (D : ℝ) (S : ℝ) 
  (h1 : 2 * S = 10)
  (h2 : (D / S) + (D / (2 * S)) = 6) :
  D = 20 :=
by
  sorry

end steve_distance_l1606_160679


namespace length_of_second_train_l1606_160645

def first_train_length : ℝ := 290
def first_train_speed_kmph : ℝ := 120
def second_train_speed_kmph : ℝ := 80
def cross_time : ℝ := 9

noncomputable def first_train_speed_mps := (first_train_speed_kmph * 1000) / 3600
noncomputable def second_train_speed_mps := (second_train_speed_kmph * 1000) / 3600
noncomputable def relative_speed := first_train_speed_mps + second_train_speed_mps
noncomputable def total_distance_covered := relative_speed * cross_time
noncomputable def second_train_length := total_distance_covered - first_train_length

theorem length_of_second_train : second_train_length = 209.95 := by
  sorry

end length_of_second_train_l1606_160645


namespace total_distance_traveled_l1606_160678

theorem total_distance_traveled 
  (Vm : ℝ) (Vr : ℝ) (T_total : ℝ) (D : ℝ) 
  (H_Vm : Vm = 6) 
  (H_Vr : Vr = 1.2) 
  (H_T_total : T_total = 1) 
  (H_time_eq : D / (Vm - Vr) + D / (Vm + Vr) = T_total) 
  : 2 * D = 5.76 := 
by sorry

end total_distance_traveled_l1606_160678


namespace complement_of_A_in_U_l1606_160685

noncomputable def U : Set ℝ := {x | (x - 2) / x ≤ 1}

noncomputable def A : Set ℝ := {x | 2 - x ≤ 1}

theorem complement_of_A_in_U :
  (U \ A) = {x | 0 < x ∧ x < 1} :=
by
  sorry

end complement_of_A_in_U_l1606_160685


namespace find_m_l1606_160662

theorem find_m (m : ℝ) (a b : ℝ × ℝ)
  (ha : a = (3, m)) (hb : b = (1, -2))
  (h : a.1 * b.1 + a.2 * b.2 = b.1^2 + b.2^2) :
  m = -1 :=
by {
  sorry
}

end find_m_l1606_160662


namespace boys_and_girls_equal_l1606_160623

theorem boys_and_girls_equal (m d M D : ℕ) (hm : m > 0) (hd : d > 0) (h1 : (M / m) ≠ (D / d)) (h2 : (M / m + D / d) / 2 = (M + D) / (m + d)) :
  m = d := 
sorry

end boys_and_girls_equal_l1606_160623


namespace value_of_J_l1606_160635

theorem value_of_J (J : ℕ) : 32^4 * 4^4 = 2^J → J = 28 :=
by
  intro h
  sorry

end value_of_J_l1606_160635


namespace pyramid_four_triangular_faces_area_l1606_160681

theorem pyramid_four_triangular_faces_area 
  (base_edge : ℝ) (lateral_edge : ℝ) 
  (h_base : base_edge = 8)
  (h_lateral : lateral_edge = 7) :
  let h := Real.sqrt (lateral_edge ^ 2 - (base_edge / 2) ^ 2)
  let triangle_area := (1 / 2) * base_edge * h
  let total_area := 4 * triangle_area
  total_area = 16 * Real.sqrt 33 :=
by
  -- Definitions to introduce local values
  let half_base := base_edge / 2
  let h := Real.sqrt (lateral_edge ^ 2 - half_base ^ 2)
  let triangle_area := (1 / 2) * base_edge * h
  let total_area := 4 * triangle_area
  -- Assertion to compare calculated total area with given correct answer
  have h_eq : h = Real.sqrt 33 := by sorry
  have triangle_area_eq : triangle_area = 4 * Real.sqrt 33 := by sorry
  have total_area_eq : total_area = 16 * Real.sqrt 33 := by sorry
  exact total_area_eq

end pyramid_four_triangular_faces_area_l1606_160681


namespace SamaraSpentOnDetailing_l1606_160611

def costSamara (D : ℝ) : ℝ := 25 + 467 + D
def costAlberto : ℝ := 2457
def difference : ℝ := 1886

theorem SamaraSpentOnDetailing : 
  ∃ (D : ℝ), costAlberto = costSamara D + difference ∧ D = 79 := 
sorry

end SamaraSpentOnDetailing_l1606_160611


namespace max_wx_plus_xy_plus_yz_plus_wz_l1606_160636

theorem max_wx_plus_xy_plus_yz_plus_wz (w x y z : ℝ) (h_nonneg : 0 ≤ w ∧ 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z) (h_sum : w + x + y + z = 200) :
  wx + xy + yz + wz ≤ 10000 :=
sorry

end max_wx_plus_xy_plus_yz_plus_wz_l1606_160636


namespace range_of_a_l1606_160674

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x ≤ 4 → (2 * x + 2 * (a - 1)) ≤ 0) → a ≤ -3 :=
by
  sorry

end range_of_a_l1606_160674


namespace average_of_four_variables_l1606_160633

theorem average_of_four_variables (x y z w : ℝ) (h : (5 / 2) * (x + y + z + w) = 25) :
  (x + y + z + w) / 4 = 2.5 :=
sorry

end average_of_four_variables_l1606_160633


namespace geometry_problem_l1606_160684

theorem geometry_problem
  (A B C D E : Type*)
  (BAC ABC ACB ADE ADC AEB DEB CDE : ℝ)
  (h₁ : ABC = 72)
  (h₂ : ACB = 90)
  (h₃ : CDE = 36)
  (h₄ : ADC = 180)
  (h₅ : AEB = 180) :
  DEB = 162 :=
sorry

end geometry_problem_l1606_160684


namespace lucas_journey_distance_l1606_160626

noncomputable def distance (D : ℝ) : ℝ :=
  let usual_speed := D / 150
  let distance_before_traffic := 2 * D / 5
  let speed_after_traffic := usual_speed - 1 / 2
  let time_before_traffic := distance_before_traffic / usual_speed
  let time_after_traffic := (3 * D / 5) / speed_after_traffic
  time_before_traffic + time_after_traffic

theorem lucas_journey_distance : ∃ D : ℝ, distance D = 255 ∧ D = 48.75 :=
sorry

end lucas_journey_distance_l1606_160626


namespace negation_of_p_l1606_160677

variable (p : ∀ x : ℝ, x^2 + x - 6 ≤ 0)

theorem negation_of_p : (∃ x : ℝ, x^2 + x - 6 > 0) :=
sorry

end negation_of_p_l1606_160677


namespace largest_common_term_l1606_160671

/-- The arithmetic progression sequence1 --/
def sequence1 (n : ℕ) : ℤ := 4 + 5 * n

/-- The arithmetic progression sequence2 --/
def sequence2 (n : ℕ) : ℤ := 5 + 8 * n

/-- The common term condition for sequence1 --/
def common_term_condition1 (a : ℤ) : Prop := ∃ n : ℕ, a = sequence1 n

/-- The common term condition for sequence2 --/
def common_term_condition2 (a : ℤ) : Prop := ∃ n : ℕ, a = sequence2 n

/-- The largest common term less than 1000 --/
def is_largest_common_term (a : ℤ) : Prop :=
  common_term_condition1 a ∧ common_term_condition2 a ∧ a < 1000 ∧
  ∀ b : ℤ, common_term_condition1 b ∧ common_term_condition2 b ∧ b < 1000 → b ≤ a

/-- Lean theorem statement --/
theorem largest_common_term :
  ∃ a : ℤ, is_largest_common_term a ∧ a = 989 :=
sorry

end largest_common_term_l1606_160671


namespace percentage_difference_l1606_160603

theorem percentage_difference (x y z : ℝ) (h1 : y = 1.70 * x) (h2 : z = 1.50 * y) :
   x / z = 39.22 / 100 :=
by
  sorry

end percentage_difference_l1606_160603


namespace nonneg_real_inequality_l1606_160612

theorem nonneg_real_inequality (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) : a^3 + b^3 ≥ Real.sqrt (a * b) * (a^2 + b^2) := 
by
  sorry

end nonneg_real_inequality_l1606_160612


namespace triangle_no_solution_l1606_160688

def angleSumOfTriangle : ℝ := 180

def hasNoSolution (a b A : ℝ) : Prop :=
  A >= angleSumOfTriangle

theorem triangle_no_solution {a b A : ℝ} (ha : a = 181) (hb : b = 209) (hA : A = 121) :
  hasNoSolution a b A := sorry

end triangle_no_solution_l1606_160688


namespace problem1_problem2_l1606_160619

theorem problem1 (a b c : ℝ) (h1 : a = 5.42) (h2 : b = 3.75) (h3 : c = 0.58) :
  a - (b - c) = 2.25 :=
by sorry

theorem problem2 (d e f g h : ℝ) (h4 : d = 4 / 5) (h5 : e = 7.7) (h6 : f = 0.8) (h7 : g = 3.3) (h8 : h = 1) :
  d * e + f * g - d = 8 :=
by sorry

end problem1_problem2_l1606_160619


namespace train_b_speed_l1606_160638

/-- Two trains, A and B, start simultaneously from two stations 480 kilometers apart and meet after 2.5 hours. 
Train A travels at a speed of 102 kilometers per hour. What is the speed of train B in kilometers per hour? -/
theorem train_b_speed (d t : ℝ) (speedA speedB : ℝ) (h1 : d = 480) (h2 : t = 2.5) (h3 : speedA = 102)
  (h4 : speedA * t + speedB * t = d) : speedB = 90 := 
by
  sorry

end train_b_speed_l1606_160638


namespace fill_time_calculation_l1606_160607

-- Definitions based on conditions
def pool_volume : ℝ := 24000
def number_of_hoses : ℕ := 6
def water_per_hose_per_minute : ℝ := 3
def minutes_per_hour : ℝ := 60

-- Theorem statement translating the mathematically equivalent proof problem
theorem fill_time_calculation :
  pool_volume / (number_of_hoses * water_per_hose_per_minute * minutes_per_hour) = 22 :=
by
  sorry

end fill_time_calculation_l1606_160607


namespace units_digit_of_ksq_plus_2k_l1606_160640

def k := 2023^3 - 3^2023

theorem units_digit_of_ksq_plus_2k : (k^2 + 2^k) % 10 = 1 := 
  sorry

end units_digit_of_ksq_plus_2k_l1606_160640


namespace ben_total_distance_walked_l1606_160689

-- Definitions based on conditions
def walking_speed : ℝ := 4  -- 4 miles per hour.
def total_time : ℝ := 2  -- 2 hours.
def break_time : ℝ := 0.25  -- 0.25 hours (15 minutes).

-- Proof goal: Prove that the total distance walked is 7.0 miles.
theorem ben_total_distance_walked : (walking_speed * (total_time - break_time) = 7.0) :=
by
  sorry

end ben_total_distance_walked_l1606_160689


namespace find_ab_l1606_160651

theorem find_ab (a b : ℕ) (h : (Real.sqrt 30 - Real.sqrt 18) * (3 * Real.sqrt a + Real.sqrt b) = 12) : a = 2 ∧ b = 30 :=
sorry

end find_ab_l1606_160651


namespace milk_production_l1606_160641

variables (x α y z w β v : ℝ)

theorem milk_production :
  (w * v * β * y) / (α^2 * x * z) = β * y * w * v / (α^2 * x * z) := 
by
  sorry

end milk_production_l1606_160641


namespace hyperbola_asymptotes_angle_l1606_160652

noncomputable def angle_between_asymptotes 
  (a b : ℝ) (e : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : e = 2 * Real.sqrt 3 / 3) : ℝ :=
  2 * Real.arctan (b / a)

theorem hyperbola_asymptotes_angle (a b : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : e = 2 * Real.sqrt 3 / 3) 
  (b_eq : b = Real.sqrt (e^2 * a^2 - a^2)) : 
  angle_between_asymptotes a b e h1 h2 h3 = π / 3 := 
by
  -- proof omitted
  sorry
  
end hyperbola_asymptotes_angle_l1606_160652


namespace smallest_possible_value_of_AP_plus_BP_l1606_160600

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2)

theorem smallest_possible_value_of_AP_plus_BP :
  let A := (1, 0)
  let B := (-3, 4)
  ∃ P : ℝ × ℝ, (P.2 ^ 2 = 4 * P.1) ∧
  (distance A P + distance B P = 12) :=
by
  -- proof steps would go here
  sorry

end smallest_possible_value_of_AP_plus_BP_l1606_160600


namespace dessert_menu_count_l1606_160656

def Dessert : Type := {d : String // d = "cake" ∨ d = "pie" ∨ d = "ice cream" ∨ d = "pudding"}

def valid_menu (menu : Fin 7 → Dessert) : Prop :=
  (menu 0).1 ≠ (menu 1).1 ∧
  menu 1 = ⟨"ice cream", Or.inr (Or.inr (Or.inl rfl))⟩ ∧
  (menu 1).1 ≠ (menu 2).1 ∧
  (menu 2).1 ≠ (menu 3).1 ∧
  (menu 3).1 ≠ (menu 4).1 ∧
  (menu 4).1 ≠ (menu 5).1 ∧
  menu 5 = ⟨"cake", Or.inl rfl⟩ ∧
  (menu 5).1 ≠ (menu 6).1

def total_valid_menus : Nat :=
  4 * 1 * 3 * 3 * 3 * 1 * 3

theorem dessert_menu_count : ∃ (count : Nat), count = 324 ∧ count = total_valid_menus :=
  sorry

end dessert_menu_count_l1606_160656


namespace number_of_apples_ratio_of_mixed_fruits_total_weight_of_oranges_l1606_160624

theorem number_of_apples (total_fruit : ℕ) (oranges_fraction : ℚ) (peaches_fraction : ℚ) (apples_mult : ℕ) (total_fruit_value : total_fruit = 56) :
    oranges_fraction = 1/4 → 
    peaches_fraction = 1/2 → 
    apples_mult = 5 → 
    (apples_mult * peaches_fraction * oranges_fraction * total_fruit) = 35 :=
by
  intros h1 h2 h3
  sorry

theorem ratio_of_mixed_fruits (total_fruit : ℕ) (oranges_fraction : ℚ) (peaches_fraction : ℚ) (mixed_fruits_mult : ℕ) (total_fruit_value : total_fruit = 56) :
    oranges_fraction = 1/4 → 
    peaches_fraction = 1/2 → 
    mixed_fruits_mult = 2 → 
    (mixed_fruits_mult * peaches_fraction * oranges_fraction * total_fruit) / total_fruit = 1/4 :=
by
  intros h1 h2 h3
  sorry

theorem total_weight_of_oranges (total_fruit : ℕ) (oranges_fraction : ℚ) (orange_weight : ℕ) (total_fruit_value : total_fruit = 56) :
    oranges_fraction = 1/4 → 
    orange_weight = 200 → 
    (orange_weight * oranges_fraction * total_fruit) = 2800 :=
by
  intros h1 h2
  sorry

end number_of_apples_ratio_of_mixed_fruits_total_weight_of_oranges_l1606_160624


namespace inequality_solution_l1606_160628

theorem inequality_solution (a : ℝ) (x : ℝ) 
  (h₁ : 0 < a) 
  (h₂ : 1 < a) 
  (y₁ : ℝ := a^(2 * x + 1)) 
  (y₂ : ℝ := a^(-3 * x)) :
  y₁ > y₂ → x > - (1 / 5) :=
by
  sorry

end inequality_solution_l1606_160628


namespace inscribed_cube_volume_l1606_160614

noncomputable def side_length_of_inscribed_cube (d : ℝ) : ℝ :=
d / Real.sqrt 3

noncomputable def volume_of_inscribed_cube (s : ℝ) : ℝ :=
s^3

theorem inscribed_cube_volume :
  (volume_of_inscribed_cube (side_length_of_inscribed_cube 12)) = 192 * Real.sqrt 3 :=
by
  sorry

end inscribed_cube_volume_l1606_160614


namespace rational_function_value_l1606_160691

theorem rational_function_value (g : ℚ → ℚ) (h : ∀ x : ℚ, x ≠ 0 → 4 * g (x⁻¹) + 3 * g x / x = 2 * x^3) : g (-1) = -2 :=
sorry

end rational_function_value_l1606_160691


namespace find_a_l1606_160630

def f (a : ℝ) (x : ℝ) : ℝ := a * x ^ 3 + 3 * x ^ 2 + 2

def f_prime (a : ℝ) (x : ℝ) : ℝ := 3 * a * x ^ 2 + 6 * x

theorem find_a : (f_prime a (-1) = 3) → a = 3 :=
by
  sorry

end find_a_l1606_160630


namespace least_number_of_candles_l1606_160695

theorem least_number_of_candles (b : ℕ) :
  (b ≡ 5 [MOD 6]) ∧ (b ≡ 7 [MOD 8]) ∧ (b ≡ 3 [MOD 9]) → b = 119 :=
by
  -- Proof omitted
  sorry

end least_number_of_candles_l1606_160695


namespace S_63_value_l1606_160683

noncomputable def b (n : ℕ) : ℚ := (3 + (-1)^(n-1))/2

noncomputable def a : ℕ → ℚ
| 0       => 0
| 1       => 2
| (n+2)   => if (n % 2 = 0) then - (a (n+1))/2 else 2 - 2*(a (n+1))

noncomputable def S : ℕ → ℚ
| 0       => 0
| (n+1)   => S n + a (n+1)

theorem S_63_value : S 63 = 464 := by
  sorry

end S_63_value_l1606_160683


namespace avg_children_in_families_with_children_l1606_160672

-- Define the conditions
def num_families : ℕ := 15
def avg_children_per_family : ℤ := 3
def num_childless_families : ℕ := 3

-- Total number of children among all families
def total_children : ℤ := num_families * avg_children_per_family

-- Number of families with children
def num_families_with_children : ℕ := num_families - num_childless_families

-- Average number of children in families with children, to be proven equal 3.8 when rounded to the nearest tenth.
theorem avg_children_in_families_with_children : (total_children : ℚ) / num_families_with_children = 3.8 := by
  -- Proof is omitted
  sorry

end avg_children_in_families_with_children_l1606_160672


namespace not_perfect_square_7_301_l1606_160668

theorem not_perfect_square_7_301 :
  ¬ ∃ x : ℝ, x^2 = 7^301 := sorry

end not_perfect_square_7_301_l1606_160668


namespace simultaneous_equations_solution_exists_l1606_160649

theorem simultaneous_equations_solution_exists (m : ℝ) :
  ∃ x y : ℝ, y = 3 * m * x + 2 ∧ y = (3 * m - 2) * x + 5 :=
by
  sorry

end simultaneous_equations_solution_exists_l1606_160649


namespace f_of_2014_l1606_160682

theorem f_of_2014 (f : ℝ → ℝ)
  (h1 : ∀ x : ℝ, f (x + 4) = -f x + 2 * Real.sqrt 2)
  (h2 : ∀ x : ℝ, f (-x) = f x)
  : f 2014 = Real.sqrt 2 :=
sorry

end f_of_2014_l1606_160682


namespace courtyard_length_l1606_160643

theorem courtyard_length 
  (stone_area : ℕ) 
  (stones_total : ℕ) 
  (width : ℕ)
  (total_area : ℕ) 
  (L : ℕ) 
  (h1 : stone_area = 4)
  (h2 : stones_total = 135)
  (h3 : width = 18)
  (h4 : total_area = stones_total * stone_area)
  (h5 : total_area = L * width) :
  L = 30 :=
by
  -- Proof steps would go here
  sorry

end courtyard_length_l1606_160643


namespace find_quadratic_function_l1606_160654

theorem find_quadratic_function (a h k x y : ℝ) (vertex_y : ℝ) (intersect_y : ℝ)
    (hv : h = 1 ∧ k = 2)
    (hi : x = 0 ∧ y = 3) :
    (∀ x, y = a * (x - h) ^ 2 + k) → vertex_y = h ∧ intersect_y = k →
    y = x^2 - 2 * x + 3 :=
by
  sorry

end find_quadratic_function_l1606_160654


namespace vacuum_upstairs_more_than_twice_downstairs_l1606_160650

theorem vacuum_upstairs_more_than_twice_downstairs 
  (x y : ℕ) 
  (h1 : 27 = 2 * x + y) 
  (h2 : x + 27 = 38) : 
  y = 5 :=
by 
  sorry

end vacuum_upstairs_more_than_twice_downstairs_l1606_160650


namespace find_capacity_of_second_vessel_l1606_160604

noncomputable def capacity_of_second_vessel (x : ℝ) : Prop :=
  let alcohol_from_first_vessel := 0.25 * 2
  let alcohol_from_second_vessel := 0.40 * x
  let total_liquid := 2 + x
  let total_alcohol := alcohol_from_first_vessel + alcohol_from_second_vessel
  let new_concentration := (total_alcohol / 10) * 100
  2 + x = 8 ∧ new_concentration = 29

open scoped Real

theorem find_capacity_of_second_vessel : ∃ x : ℝ, capacity_of_second_vessel x ∧ x = 6 :=
by
  sorry

end find_capacity_of_second_vessel_l1606_160604


namespace find_speed_way_home_l1606_160606

theorem find_speed_way_home
  (speed_to_mother : ℝ)
  (average_speed : ℝ)
  (speed_to_mother_val : speed_to_mother = 130)
  (average_speed_val : average_speed = 109) :
  ∃ v : ℝ, v = 109 * 130 / 151 := by
  sorry

end find_speed_way_home_l1606_160606


namespace largest_perfect_square_factor_1760_l1606_160618

theorem largest_perfect_square_factor_1760 :
  ∃ n, (∃ k, n = k^2) ∧ n ∣ 1760 ∧ ∀ m, (∃ j, m = j^2) ∧ m ∣ 1760 → m ≤ n := by
  sorry

end largest_perfect_square_factor_1760_l1606_160618


namespace sequence_6th_term_sequence_1994th_term_l1606_160659

def sequence_term (n : Nat) : Nat := n * (n + 1)

theorem sequence_6th_term:
  sequence_term 6 = 42 :=
by
  -- proof initially skipped
  sorry

theorem sequence_1994th_term:
  sequence_term 1994 = 3978030 :=
by
  -- proof initially skipped
  sorry

end sequence_6th_term_sequence_1994th_term_l1606_160659


namespace tangent_line_at_point_l1606_160698

noncomputable def f (x : ℝ) : ℝ := (x + 1) * Real.log x - 4 * (x - 1)

theorem tangent_line_at_point (x y : ℝ) (h : f 1 = 0) (h' : deriv f 1 = -2) :
  2 * x + y - 2 = 0 :=
sorry

end tangent_line_at_point_l1606_160698


namespace sum_of_fractions_eq_one_l1606_160608

variable {a b c d : ℝ} (h_a : a ≠ 0) (h_b : b ≠ 0) (h_c : c ≠ 0) (h_d : d ≠ 0)
          (h_equiv : (a * d + b * c) / (b * d) = (a * c) / (b * d))

theorem sum_of_fractions_eq_one : b / a + d / c = 1 :=
by sorry

end sum_of_fractions_eq_one_l1606_160608


namespace arc_length_is_correct_l1606_160670

-- Define the radius and central angle as given
def radius := 16
def central_angle := 2

-- Define the arc length calculation
def arc_length (r : ℕ) (α : ℕ) := α * r

-- The theorem stating the mathematically equivalent proof problem
theorem arc_length_is_correct : arc_length radius central_angle = 32 :=
by sorry

end arc_length_is_correct_l1606_160670


namespace train_length_l1606_160625

theorem train_length (speed_kmph : ℝ) (time_sec : ℝ) (speed_ms : ℝ) (length_m : ℝ)
  (h1 : speed_kmph = 120) 
  (h2 : time_sec = 6)
  (h3 : speed_ms = 33.33)
  (h4 : length_m = 200) : 
  speed_kmph * 1000 / 3600 * time_sec = length_m :=
by
  sorry

end train_length_l1606_160625


namespace max_possible_value_of_a_l1606_160620

theorem max_possible_value_of_a (a b c d : ℕ) (h1 : a < 3 * b) (h2 : b < 4 * c) (h3 : c < 5 * d) (h4 : d < 150) : 
  a ≤ 8924 :=
by {
  sorry
}

end max_possible_value_of_a_l1606_160620


namespace rachel_removed_bottle_caps_l1606_160642

def original_bottle_caps : ℕ := 87
def remaining_bottle_caps : ℕ := 40

theorem rachel_removed_bottle_caps :
  original_bottle_caps - remaining_bottle_caps = 47 := by
  sorry

end rachel_removed_bottle_caps_l1606_160642


namespace painted_cube_probability_l1606_160617

-- Define the conditions
def cube_size : Nat := 5
def total_unit_cubes : Nat := cube_size ^ 3
def corner_cubes_with_three_faces : Nat := 1
def edges_with_two_faces : Nat := 3 * (cube_size - 2) -- 3 edges, each (5 - 2) = 3
def faces_with_one_face : Nat := 2 * (cube_size * cube_size - corner_cubes_with_three_faces - edges_with_two_faces)
def no_painted_faces_cubes : Nat := total_unit_cubes - corner_cubes_with_three_faces - faces_with_one_face

-- Compute the probability
def probability := (corner_cubes_with_three_faces * no_painted_faces_cubes) / (total_unit_cubes * (total_unit_cubes - 1) / 2)

-- The theorem statement
theorem painted_cube_probability :
  probability = (2 : ℚ) / 155 := 
by {
  sorry
}

end painted_cube_probability_l1606_160617


namespace parabola_and_hyperbola_tangent_l1606_160655

theorem parabola_and_hyperbola_tangent (m : ℝ) :
  (∀ (x y : ℝ), (y = x^2 + 6) → (y^2 - m * x^2 = 6) → (m = 12 + 10 * Real.sqrt 6 ∨ m = 12 - 10 * Real.sqrt 6)) :=
sorry

end parabola_and_hyperbola_tangent_l1606_160655


namespace three_point_two_four_two_times_twelve_div_one_hundred_equals_zero_point_three_eight_nine_zero_four_l1606_160609

theorem three_point_two_four_two_times_twelve_div_one_hundred_equals_zero_point_three_eight_nine_zero_four :
  (3.242 * 12) / 100 = 0.38904 :=
by 
  sorry

end three_point_two_four_two_times_twelve_div_one_hundred_equals_zero_point_three_eight_nine_zero_four_l1606_160609


namespace y_increases_as_x_increases_l1606_160616

-- Define the linear function y = (m^2 + 2)x
def linear_function (m x : ℝ) : ℝ := (m^2 + 2) * x

-- Prove that y increases as x increases
theorem y_increases_as_x_increases (m x1 x2 : ℝ) (h : x1 < x2) : linear_function m x1 < linear_function m x2 :=
by
  -- because m^2 + 2 is always positive, the function is strictly increasing
  have hm : 0 < m^2 + 2 := by linarith [pow_two_nonneg m]
  have hx : (m^2 + 2) * x1 < (m^2 + 2) * x2 := by exact (mul_lt_mul_left hm).mpr h
  exact hx

end y_increases_as_x_increases_l1606_160616


namespace apples_in_each_box_l1606_160602

variable (A : ℕ)
variable (ApplesSaturday : ℕ := 50 * A)
variable (ApplesSunday : ℕ := 25 * A)
variable (ApplesLeft : ℕ := 3 * A)
variable (ApplesSold : ℕ := 720)

theorem apples_in_each_box :
  (ApplesSaturday + ApplesSunday - ApplesSold = ApplesLeft) → A = 10 :=
by
  sorry

end apples_in_each_box_l1606_160602


namespace line_passes_fixed_point_l1606_160673

open Real

theorem line_passes_fixed_point
  (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : b < a)
  (M N : ℝ × ℝ)
  (hM : M.1^2 / a^2 + M.2^2 / b^2 = 1)
  (hN : N.1^2 / a^2 + N.2^2 / b^2 = 1)
  (hMAhNA : (M.1 + a) * (N.1 + a) + M.2 * N.2 = 0):
  ∃ (P : ℝ × ℝ), P = (a * (b^2 - a^2) / (a^2 + b^2), 0) ∧ (N.2 - M.2) * (P.1 - M.1) = (P.2 - M.2) * (N.1 - M.1) :=
sorry

end line_passes_fixed_point_l1606_160673


namespace problem_product_xyzw_l1606_160632

theorem problem_product_xyzw
    (x y z w : ℝ)
    (h1 : x + 1 / y = 1)
    (h2 : y + 1 / z + w = 1)
    (h3 : w = 2) :
    xyzw = -2 * y^2 + 2 * y :=
by
    sorry

end problem_product_xyzw_l1606_160632


namespace cost_of_dog_l1606_160644

-- Given conditions
def dollars_misha_has : ℕ := 34
def dollars_misha_needs_earn : ℕ := 13

-- Formal statement of the mathematic proof
theorem cost_of_dog : dollars_misha_has + dollars_misha_needs_earn = 47 := by
  sorry

end cost_of_dog_l1606_160644


namespace discount_difference_is_24_l1606_160676

-- Definitions based on conditions
def smartphone_price : ℝ := 800
def single_discount_rate : ℝ := 0.25
def first_successive_discount_rate : ℝ := 0.20
def second_successive_discount_rate : ℝ := 0.10

-- Definitions of discounted prices
def single_discount_price (p : ℝ) (d1 : ℝ) : ℝ := p * (1 - d1)
def successive_discount_price (p : ℝ) (d1 : ℝ) (d2 : ℝ) : ℝ := 
  let intermediate_price := p * (1 - d1) 
  intermediate_price * (1 - d2)

-- Calculate the difference between the two final prices
def price_difference (p : ℝ) (d1 : ℝ) (d2 : ℝ) (d3 : ℝ) : ℝ :=
  (single_discount_price p d1) - (successive_discount_price p d2 d3)

theorem discount_difference_is_24 :
  price_difference smartphone_price single_discount_rate first_successive_discount_rate second_successive_discount_rate = 24 := 
sorry

end discount_difference_is_24_l1606_160676


namespace quadratic_common_root_l1606_160634

theorem quadratic_common_root (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h1 : ∃ x, x^2 + a * x + b = 0 ∧ x^2 + c * x + a = 0)
  (h2 : ∃ x, x^2 + a * x + b = 0 ∧ x^2 + b * x + c = 0)
  (h3 : ∃ x, x^2 + b * x + c = 0 ∧ x^2 + c * x + a = 0) :
  a^2 + b^2 + c^2 = 6 :=
sorry

end quadratic_common_root_l1606_160634


namespace cost_price_of_computer_table_l1606_160622

theorem cost_price_of_computer_table (C SP : ℝ) (h1 : SP = 1.25 * C) (h2 : SP = 8340) :
  C = 6672 :=
by
  sorry

end cost_price_of_computer_table_l1606_160622
