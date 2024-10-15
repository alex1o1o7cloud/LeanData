import Mathlib

namespace NUMINAMATH_GPT_P_inequality_l465_46570

def P (n : ℕ) (x : ℝ) : ℝ := (Finset.range (n + 1)).sum (λ k => x^k)

theorem P_inequality (x : ℝ) (hx : 0 < x) :
  P 20 x * P 21 (x^2) ≤ P 20 (x^2) * P 22 x :=
by
  sorry

end NUMINAMATH_GPT_P_inequality_l465_46570


namespace NUMINAMATH_GPT_cart_distance_traveled_l465_46561

-- Define the problem parameters/conditions
def circumference_front : ℕ := 30
def circumference_back : ℕ := 33
def revolutions_difference : ℕ := 5

-- Define the question and the expected correct answer
theorem cart_distance_traveled :
  ∀ (R : ℕ), ((R + revolutions_difference) * circumference_front = R * circumference_back) → (R * circumference_back) = 1650 :=
by
  intro R h
  sorry

end NUMINAMATH_GPT_cart_distance_traveled_l465_46561


namespace NUMINAMATH_GPT_sum_first_four_terms_of_arithmetic_sequence_l465_46592

theorem sum_first_four_terms_of_arithmetic_sequence (a₈ a₉ a₁₀ : ℤ) (d : ℤ) (a₁ a₂ a₃ a₄ : ℤ) : 
  (a₈ = 21) →
  (a₉ = 17) →
  (a₁₀ = 13) →
  (d = a₉ - a₈) →
  (a₁ = a₈ - 7 * d) →
  (a₂ = a₁ + d) →
  (a₃ = a₂ + d) →
  (a₄ = a₃ + d) →
  a₁ + a₂ + a₃ + a₄ = 172 :=
by 
  intros h₁ h₂ h₃ h₄ h₅ h₆ h₇ h₈
  sorry

end NUMINAMATH_GPT_sum_first_four_terms_of_arithmetic_sequence_l465_46592


namespace NUMINAMATH_GPT_robot_transport_max_robots_l465_46530

section
variable {A B : ℕ}   -- Define the variables A and B
variable {m : ℕ}     -- Define the variable m

-- Part 1
theorem robot_transport (h1 : A = B + 30) (h2 : 1500 * B = 1000 * (B + 30)) : A = 90 ∧ B = 60 :=
by
  sorry

-- Part 2
theorem max_robots (h3 : 50000 * m + 30000 * (12 - m) ≤ 450000) : m ≤ 4 :=
by
  sorry
end

end NUMINAMATH_GPT_robot_transport_max_robots_l465_46530


namespace NUMINAMATH_GPT_jessica_threw_away_4_roses_l465_46557

def roses_thrown_away (a b c d : ℕ) : Prop :=
  (a + b) - d = c

theorem jessica_threw_away_4_roses :
  roses_thrown_away 2 25 23 4 :=
by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_jessica_threw_away_4_roses_l465_46557


namespace NUMINAMATH_GPT_problem_statement_l465_46596

variable (a b : ℝ)

open Real

noncomputable def inequality_holds (a b : ℝ) : Prop :=
  0 ≤ a ∧ 0 ≤ b ∧ a + b < 2 → (1 / (1 + a^2)) + (1 / (1 + b^2)) ≤ 2 / (1 + a * b)

noncomputable def equality_condition (a b : ℝ) : Prop :=
  0 ≤ a ∧ 0 ≤ b ∧ a + b < 2 → ((1 / (1 + a^2)) + (1 / (1 + b^2)) = 2 / (1 + a * b) ↔ a = b)

theorem problem_statement (a b : ℝ) : inequality_holds a b ∧ equality_condition a b :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l465_46596


namespace NUMINAMATH_GPT_married_men_fraction_l465_46510

-- define the total number of women
def W : ℕ := 7

-- define the number of single women
def single_women (W : ℕ) : ℕ := 3

-- define the probability of picking a single woman
def P_s : ℚ := single_women W / W

-- define number of married women
def married_women (W : ℕ) : ℕ := W - single_women W

-- define number of married men
def married_men (W : ℕ) : ℕ := married_women W

-- define total number of people
def total_people (W : ℕ) : ℕ := W + married_men W

-- define fraction of married men
def married_men_ratio (W : ℕ) : ℚ := married_men W / total_people W

-- theorem to prove that the ratio is 4/11
theorem married_men_fraction : married_men_ratio W = 4 / 11 := 
by 
  sorry

end NUMINAMATH_GPT_married_men_fraction_l465_46510


namespace NUMINAMATH_GPT_smallest_number_of_seats_required_l465_46563

theorem smallest_number_of_seats_required (total_chairs : ℕ) (condition : ∀ (N : ℕ), ∀ (seating : Finset ℕ),
  seating.card = N → (∀ x ∈ seating, (x + 1) % total_chairs ∈ seating ∨ (x + total_chairs - 1) % total_chairs ∈ seating)) :
  total_chairs = 100 → ∃ N : ℕ, N = 20 :=
by
  intros
  sorry

end NUMINAMATH_GPT_smallest_number_of_seats_required_l465_46563


namespace NUMINAMATH_GPT_find_nine_boxes_of_same_variety_l465_46520

theorem find_nine_boxes_of_same_variety (boxes : ℕ) (A B C : ℕ) (h_total : boxes = 25) (h_one_variety : boxes = A + B + C) 
  (hA : A ≤ 25) (hB : B ≤ 25) (hC : C ≤ 25) :
  (A ≥ 9) ∨ (B ≥ 9) ∨ (C ≥ 9) :=
sorry

end NUMINAMATH_GPT_find_nine_boxes_of_same_variety_l465_46520


namespace NUMINAMATH_GPT_triangle_angle_sum_l465_46581

theorem triangle_angle_sum {A B C : Type} 
  (angle_ABC : ℝ) (angle_BAC : ℝ) (angle_BCA : ℝ) (x : ℝ) 
  (h1: angle_ABC = 90) 
  (h2: angle_BAC = 3 * x) 
  (h3: angle_BCA = x + 10)
  : x = 20 :=
by
  sorry

end NUMINAMATH_GPT_triangle_angle_sum_l465_46581


namespace NUMINAMATH_GPT_brendan_remaining_money_l465_46591

-- Definitions given in the conditions
def weekly_pay (total_monthly_earnings : ℕ) (weeks_in_month : ℕ) : ℕ := total_monthly_earnings / weeks_in_month
def weekly_recharge_amount (weekly_pay : ℕ) : ℕ := weekly_pay / 2
def total_recharge_amount (weekly_recharge_amount : ℕ) (weeks_in_month : ℕ) : ℕ := weekly_recharge_amount * weeks_in_month
def remaining_money_after_car_purchase (total_monthly_earnings : ℕ) (car_cost : ℕ) : ℕ := total_monthly_earnings - car_cost
def total_remaining_money (remaining_money_after_car_purchase : ℕ) (total_recharge_amount : ℕ) : ℕ := remaining_money_after_car_purchase - total_recharge_amount

-- The actual statement to prove
theorem brendan_remaining_money
  (total_monthly_earnings : ℕ := 5000)
  (weeks_in_month : ℕ := 4)
  (car_cost : ℕ := 1500)
  (weekly_pay := weekly_pay total_monthly_earnings weeks_in_month)
  (weekly_recharge_amount := weekly_recharge_amount weekly_pay)
  (total_recharge_amount := total_recharge_amount weekly_recharge_amount weeks_in_month)
  (remaining_money_after_car_purchase := remaining_money_after_car_purchase total_monthly_earnings car_cost)
  (total_remaining_money := total_remaining_money remaining_money_after_car_purchase total_recharge_amount) :
  total_remaining_money = 1000 :=
sorry

end NUMINAMATH_GPT_brendan_remaining_money_l465_46591


namespace NUMINAMATH_GPT_maximum_area_of_inscribed_rectangle_l465_46589

theorem maximum_area_of_inscribed_rectangle (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ∃ (A : ℝ), A = (a * b) / 4 :=
by
  sorry -- placeholder for the proof

end NUMINAMATH_GPT_maximum_area_of_inscribed_rectangle_l465_46589


namespace NUMINAMATH_GPT_find_number_l465_46569

theorem find_number (N x : ℝ) (h : x = 9) (h1 : N - (5 / x) = 4 + (4 / x)) : N = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l465_46569


namespace NUMINAMATH_GPT_sum_volumes_spheres_l465_46508

theorem sum_volumes_spheres (l : ℝ) (h_l : l = 2) : 
  ∑' (n : ℕ), (4 / 3) * π * ((1 / (3 ^ (n + 1))) ^ 3) = (2 * π / 39) :=
by
  sorry

end NUMINAMATH_GPT_sum_volumes_spheres_l465_46508


namespace NUMINAMATH_GPT_total_age_difference_l465_46578

noncomputable def ages_difference (A B C : ℕ) : ℕ :=
  (A + B) - (B + C)

theorem total_age_difference (A B C : ℕ) (h₁ : A + B > B + C) (h₂ : C = A - 11) : ages_difference A B C = 11 :=
by
  sorry

end NUMINAMATH_GPT_total_age_difference_l465_46578


namespace NUMINAMATH_GPT_base_b_representation_l465_46538

theorem base_b_representation (b : ℕ) (h₁ : 1 * b + 5 = n) (h₂ : n^2 = 4 * b^2 + 3 * b + 3) : b = 7 :=
by {
  sorry
}

end NUMINAMATH_GPT_base_b_representation_l465_46538


namespace NUMINAMATH_GPT_funct_eq_x_l465_46546

theorem funct_eq_x (f : ℝ → ℝ) (h : ∀ x y : ℝ, f (x^4 + 4 * y^4) = f (x^2)^2 + 4 * y^3 * f y) : ∀ x : ℝ, f x = x := 
by 
  sorry

end NUMINAMATH_GPT_funct_eq_x_l465_46546


namespace NUMINAMATH_GPT_Emily_subtract_59_l465_46537

theorem Emily_subtract_59 : (30 - 1) ^ 2 = 30 ^ 2 - 59 := by
  sorry

end NUMINAMATH_GPT_Emily_subtract_59_l465_46537


namespace NUMINAMATH_GPT_find_a_when_lines_perpendicular_l465_46514

theorem find_a_when_lines_perpendicular (a : ℝ) : 
  (∃ x y : ℝ, ax + 3 * y - 1 = 0 ∧  2 * x + (a^2 - a) * y + 3 = 0) ∧ 
  (∃ m₁ m₂ : ℝ, m₁ = -a / 3 ∧ m₂ = -2 / (a^2 - a) ∧ m₁ * m₂ = -1)
  → a = 0 ∨ a = 5 / 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_a_when_lines_perpendicular_l465_46514


namespace NUMINAMATH_GPT_percentage_in_quarters_l465_46533

theorem percentage_in_quarters (dimes quarters nickels : ℕ) (value_dime value_quarter value_nickel : ℕ)
  (h_dimes : dimes = 40)
  (h_quarters : quarters = 30)
  (h_nickels : nickels = 10)
  (h_value_dime : value_dime = 10)
  (h_value_quarter : value_quarter = 25)
  (h_value_nickel : value_nickel = 5) :
  (quarters * value_quarter : ℚ) / ((dimes * value_dime + quarters * value_quarter + nickels * value_nickel) : ℚ) * 100 = 62.5 := 
  sorry

end NUMINAMATH_GPT_percentage_in_quarters_l465_46533


namespace NUMINAMATH_GPT_fraction_addition_l465_46586

theorem fraction_addition (d : ℤ) :
  (6 + 4 * d) / 9 + 3 / 2 = (39 + 8 * d) / 18 := sorry

end NUMINAMATH_GPT_fraction_addition_l465_46586


namespace NUMINAMATH_GPT_calculate_initial_income_l465_46595

noncomputable def initial_income : Float := 151173.52

theorem calculate_initial_income :
  let I := initial_income
  let children_distribution := 0.30 * I
  let eldest_child_share := (children_distribution / 6) + 0.05 * I
  let remaining_for_wife := 0.40 * I
  let remaining_after_distribution := I - (children_distribution + remaining_for_wife)
  let donation_to_orphanage := 0.10 * remaining_after_distribution
  let remaining_after_donation := remaining_after_distribution - donation_to_orphanage
  let federal_tax := 0.02 * remaining_after_donation
  let final_amount := remaining_after_donation - federal_tax
  final_amount = 40000 :=
by
  sorry

end NUMINAMATH_GPT_calculate_initial_income_l465_46595


namespace NUMINAMATH_GPT_calculate_expression_l465_46585

theorem calculate_expression : 
  2 - 1 / (2 - 1 / (2 + 2)) = 10 / 7 := 
by sorry

end NUMINAMATH_GPT_calculate_expression_l465_46585


namespace NUMINAMATH_GPT_smallest_digit_divisible_by_9_l465_46556

theorem smallest_digit_divisible_by_9 :
  ∃ (d : ℕ), (25 + d) % 9 = 0 ∧ (∀ e : ℕ, (25 + e) % 9 = 0 → e ≥ d) :=
by
  sorry

end NUMINAMATH_GPT_smallest_digit_divisible_by_9_l465_46556


namespace NUMINAMATH_GPT_chen_steps_recorded_correct_l465_46552

-- Define the standard for steps per day
def standard : ℕ := 5000

-- Define the steps walked by Xia
def xia_steps : ℕ := 6200

-- Define the recorded steps for Xia
def xia_recorded : ℤ := xia_steps - standard

-- Assert that Xia's recorded steps are +1200
lemma xia_steps_recorded_correct : xia_recorded = 1200 := by
  sorry

-- Define the steps walked by Chen
def chen_steps : ℕ := 4800

-- Define the recorded steps for Chen
def chen_recorded : ℤ := standard - chen_steps

-- State and prove that Chen's recorded steps are -200
theorem chen_steps_recorded_correct : chen_recorded = -200 :=
  sorry

end NUMINAMATH_GPT_chen_steps_recorded_correct_l465_46552


namespace NUMINAMATH_GPT_union_of_A_and_B_l465_46517

def setA : Set ℝ := {x : ℝ | x > 1 / 2}
def setB : Set ℝ := {x : ℝ | -1 < x ∧ x < 3}

theorem union_of_A_and_B : setA ∪ setB = {x : ℝ | -1 < x} :=
by
  sorry

end NUMINAMATH_GPT_union_of_A_and_B_l465_46517


namespace NUMINAMATH_GPT_exists_three_points_l465_46599

theorem exists_three_points (n : ℕ) (h : 3 ≤ n) (points : Fin n → EuclideanSpace ℝ (Fin 2))
  (distinct : ∀ i j : Fin n, i ≠ j → points i ≠ points j) :
  ∃ (A B C : Fin n),
    A ≠ B ∧ A ≠ C ∧ B ≠ C ∧ 
    1 ≤ dist (points A) (points B) / dist (points A) (points C) ∧ 
    dist (points A) (points B) / dist (points A) (points C) < (n + 1) / (n - 1) := 
sorry

end NUMINAMATH_GPT_exists_three_points_l465_46599


namespace NUMINAMATH_GPT_find_coefficients_sum_l465_46564

theorem find_coefficients_sum (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ) :
  (∀ x : ℝ, (2 * x - 3)^5 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5) →
  a_1 + 2 * a_2 + 3 * a_3 + 4 * a_4 + 5 * a_5 = 10 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_coefficients_sum_l465_46564


namespace NUMINAMATH_GPT_find_fourth_root_l465_46580

theorem find_fourth_root (b c α : ℝ)
  (h₁ : b * (-3)^4 + (b + 3 * c) * (-3)^3 + (c - 4 * b) * (-3)^2 + (19 - b) * (-3) - 2 = 0)
  (h₂ : b * 4^4 + (b + 3 * c) * 4^3 + (c - 4 * b) * 4^2 + (19 - b) * 4 - 2 = 0)
  (h₃ : b * 2^4 + (b + 3 * c) * 2^3 + (c - 4 * b) * 2^2 + (19 - b) * 2 - 2 = 0)
  (h₄ : (-3) + 4 + 2 + α = 2)
  : α = 1 :=
sorry

end NUMINAMATH_GPT_find_fourth_root_l465_46580


namespace NUMINAMATH_GPT_find_track_circumference_l465_46582

noncomputable def track_circumference : ℝ := 720

theorem find_track_circumference
  (A B : ℝ)
  (uA uB : ℝ)
  (h1 : A = 0)
  (h2 : B = track_circumference / 2)
  (h3 : ∀ t : ℝ, A + uA * t = B + uB * t ↔ t = 150 / uB)
  (h4 : ∀ t : ℝ, A + uA * t = B + uB * t ↔ t = (track_circumference - 90) / uA)
  (h5 : ∀ t : ℝ, A + uA * t = B + uB * t ↔ t = 1.5 * track_circumference / uA) :
  track_circumference = 720 :=
by sorry

end NUMINAMATH_GPT_find_track_circumference_l465_46582


namespace NUMINAMATH_GPT_repeating_decimal_fraction_sum_l465_46502

/-- The repeating decimal 3.171717... can be written as a fraction. When reduced to lowest
terms, the sum of the numerator and denominator of this fraction is 413. -/
theorem repeating_decimal_fraction_sum :
  let y := 3.17171717 -- The repeating decimal
  let frac_num := 314
  let frac_den := 99
  let sum := frac_num + frac_den
  y = frac_num / frac_den ∧ sum = 413 := by
  sorry

end NUMINAMATH_GPT_repeating_decimal_fraction_sum_l465_46502


namespace NUMINAMATH_GPT_value_of_g_at_3_l465_46568

def g (x : ℝ) := x^2 - 2*x + 1

theorem value_of_g_at_3 : g 3 = 4 :=
by
  sorry

end NUMINAMATH_GPT_value_of_g_at_3_l465_46568


namespace NUMINAMATH_GPT_find_pairs_l465_46534

theorem find_pairs (x y : ℕ) (hx_pos : x > 0) (hy_pos : y > 0) :
  y ∣ x^2 + 1 ∧ x^2 ∣ y^3 + 1 ↔ (x = 1 ∧ y = 1) ∨ (x = 1 ∧ y = 2) ∨ (x = 3 ∧ y = 2) :=
by
  sorry

end NUMINAMATH_GPT_find_pairs_l465_46534


namespace NUMINAMATH_GPT_binary_multiplication_addition_l465_46503

-- Define the binary representation of the given numbers
def b1101 : ℕ := 0b1101
def b111 : ℕ := 0b111
def b1011 : ℕ := 0b1011
def b1011010 : ℕ := 0b1011010

-- State the theorem
theorem binary_multiplication_addition :
  (b1101 * b111 + b1011) = b1011010 := 
sorry

end NUMINAMATH_GPT_binary_multiplication_addition_l465_46503


namespace NUMINAMATH_GPT_table_cost_l465_46567

variable (T : ℝ) -- Cost of the table
variable (C : ℝ) -- Cost of a chair

-- Conditions
axiom h1 : C = T / 7
axiom h2 : T + 4 * C = 220

theorem table_cost : T = 140 :=
by
  sorry

end NUMINAMATH_GPT_table_cost_l465_46567


namespace NUMINAMATH_GPT_fractionOf_Product_Of_Fractions_l465_46542

noncomputable def fractionOfProductOfFractions := 
  let a := (2 : ℚ) / 9 * (5 : ℚ) / 6 -- Define the product of the fractions
  let b := (3 : ℚ) / 4 -- Define another fraction
  a / b = 20 / 81 -- Statement to be proven

theorem fractionOf_Product_Of_Fractions: fractionOfProductOfFractions :=
by sorry

end NUMINAMATH_GPT_fractionOf_Product_Of_Fractions_l465_46542


namespace NUMINAMATH_GPT_tan_alpha_eq_neg_one_l465_46504

theorem tan_alpha_eq_neg_one (α : ℝ) (h1 : |Real.sin α| = |Real.cos α|)
    (h2 : π / 2 < α ∧ α < π) : Real.tan α = -1 :=
sorry

end NUMINAMATH_GPT_tan_alpha_eq_neg_one_l465_46504


namespace NUMINAMATH_GPT_intersection_condition_l465_46594

-- Define the lines
def line1 (x y : ℝ) := 2*x - 2*y - 3 = 0
def line2 (x y : ℝ) := 3*x - 5*y + 1 = 0
def line (a b x y : ℝ) := a*x - y + b = 0

-- Define the condition
def condition (a b : ℝ) := 17*a + 4*b = 11

-- Prove that the line l passes through the intersection point of l1 and l2 if and only if the condition holds
theorem intersection_condition (a b : ℝ) :
  (∃ x y : ℝ, line1 x y ∧ line2 x y ∧ line a b x y) ↔ condition a b :=
  sorry

end NUMINAMATH_GPT_intersection_condition_l465_46594


namespace NUMINAMATH_GPT_original_weight_of_beef_l465_46558

theorem original_weight_of_beef (weight_after_processing : ℝ) (loss_percentage : ℝ) :
  loss_percentage = 0.5 → weight_after_processing = 750 → 
  (750 : ℝ) / (1 - 0.5) = 1500 :=
by
  intros h_loss_percent h_weight_after
  sorry

end NUMINAMATH_GPT_original_weight_of_beef_l465_46558


namespace NUMINAMATH_GPT_average_rainfall_l465_46535

theorem average_rainfall (rainfall_Tuesday : ℝ) (rainfall_others : ℝ) (days_in_week : ℝ)
  (h1 : rainfall_Tuesday = 10.5) 
  (h2 : rainfall_Tuesday = rainfall_others)
  (h3 : days_in_week = 7) : 
  (rainfall_Tuesday + rainfall_others) / days_in_week = 3 :=
by
  sorry

end NUMINAMATH_GPT_average_rainfall_l465_46535


namespace NUMINAMATH_GPT_focus_with_greatest_y_coordinate_l465_46527

-- Define the conditions as hypotheses
def ellipse_major_axis : (ℝ × ℝ) := (0, 3)
def ellipse_minor_axis : (ℝ × ℝ) := (2, 0)
def ellipse_semi_major_axis : ℝ := 3
def ellipse_semi_minor_axis : ℝ := 2

-- Define the theorem to compute the coordinates of the focus with the greater y-coordinate
theorem focus_with_greatest_y_coordinate :
  let a := ellipse_semi_major_axis
  let b := ellipse_semi_minor_axis
  let c := Real.sqrt (a^2 - b^2)
  (0, c) = (0, (Real.sqrt 5) / 2) :=
by
  -- skipped proof
  sorry

end NUMINAMATH_GPT_focus_with_greatest_y_coordinate_l465_46527


namespace NUMINAMATH_GPT_infinitesolutions_k_l465_46531

-- Define the system of equations as given in the problem
def system_of_equations (x y k : ℝ) : Prop :=
  (3 * x - 4 * y = 5) ∧ (9 * x - 12 * y = k)

-- State the theorem that describes the condition for infinitely many solutions
theorem infinitesolutions_k (k : ℝ) :
  (∀ (x y : ℝ), system_of_equations x y k) ↔ k = 15 :=
by
  sorry

end NUMINAMATH_GPT_infinitesolutions_k_l465_46531


namespace NUMINAMATH_GPT_find_amplitude_l465_46560

noncomputable def amplitude_of_cosine (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) : ℝ :=
  a

theorem find_amplitude (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_max : amplitude_of_cosine a b h_a h_b = 3) :
  a = 3 :=
sorry

end NUMINAMATH_GPT_find_amplitude_l465_46560


namespace NUMINAMATH_GPT_range_of_a_l465_46539

def set1 : Set ℝ := {x | x ≤ 2}
def set2 (a : ℝ) : Set ℝ := {x | x > a}
variable (a : ℝ)

theorem range_of_a (h : set1 ∪ set2 a = Set.univ) : a ≤ 2 :=
by sorry

end NUMINAMATH_GPT_range_of_a_l465_46539


namespace NUMINAMATH_GPT_unique_solution_l465_46528

theorem unique_solution (a b : ℤ) (h : a > b ∧ b > 0) (hab : a * b - a - b = 1) : a = 3 ∧ b = 2 := by
  sorry

end NUMINAMATH_GPT_unique_solution_l465_46528


namespace NUMINAMATH_GPT_perpendicular_line_l465_46522

theorem perpendicular_line (x y : ℝ) (h : 2 * x + y - 10 = 0) : 
    (∃ k : ℝ, (x = 1 ∧ y = 2) → (k * (-2) = -1)) → 
    (∃ m b : ℝ, b = 3 ∧ m = 1/2) → 
    (x - 2 * y + 3 = 0) := 
sorry

end NUMINAMATH_GPT_perpendicular_line_l465_46522


namespace NUMINAMATH_GPT_average_chemistry_mathematics_l465_46554

variable {P C M : ℝ}

theorem average_chemistry_mathematics (h : P + C + M = P + 150) : (C + M) / 2 = 75 :=
by
  sorry

end NUMINAMATH_GPT_average_chemistry_mathematics_l465_46554


namespace NUMINAMATH_GPT_min_value_of_expr_l465_46559

theorem min_value_of_expr (n : ℕ) (hn : n > 0) : (n / 3) + (27 / n) = 6 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_expr_l465_46559


namespace NUMINAMATH_GPT_jason_has_21_toys_l465_46579

-- Definitions based on the conditions
def rachel_toys : ℕ := 1
def john_toys : ℕ := rachel_toys + 6
def jason_toys : ℕ := 3 * john_toys

-- The theorem to prove
theorem jason_has_21_toys : jason_toys = 21 := by
  -- Proof not needed, hence sorry
  sorry

end NUMINAMATH_GPT_jason_has_21_toys_l465_46579


namespace NUMINAMATH_GPT_find_positive_m_l465_46555

theorem find_positive_m (m : ℝ) : (∃ x : ℝ, 16 * x^2 + m * x + 4 = 0 ∧ ∀ y : ℝ, 16 * y^2 + m * y + 4 = 0 → x = y) ↔ m = 16 :=
by
  sorry

end NUMINAMATH_GPT_find_positive_m_l465_46555


namespace NUMINAMATH_GPT_smallest_x_for_multiple_l465_46550

theorem smallest_x_for_multiple (x : ℕ) (h₁: 450 = 2 * 3^2 * 5^2) (h₂: 800 = 2^6 * 5^2) : 
  ((450 * x) % 800 = 0) ↔ x ≥ 32 :=
by
  sorry

end NUMINAMATH_GPT_smallest_x_for_multiple_l465_46550


namespace NUMINAMATH_GPT_gcd_9011_4403_l465_46597

theorem gcd_9011_4403 : Nat.gcd 9011 4403 = 1 := 
by sorry

end NUMINAMATH_GPT_gcd_9011_4403_l465_46597


namespace NUMINAMATH_GPT_steve_book_earning_l465_46500

theorem steve_book_earning
  (total_copies : ℕ)
  (advance_copies : ℕ)
  (total_kept : ℝ)
  (agent_cut_percentage : ℝ)
  (copies : ℕ)
  (money_kept : ℝ)
  (x : ℝ)
  (h1 : total_copies = 1000000)
  (h2 : advance_copies = 100000)
  (h3 : total_kept = 1620000)
  (h4 : agent_cut_percentage = 0.10)
  (h5 : copies = total_copies - advance_copies)
  (h6 : money_kept = copies * (1 - agent_cut_percentage) * x)
  (h7 : money_kept = total_kept) :
  x = 2 := 
by 
  sorry

end NUMINAMATH_GPT_steve_book_earning_l465_46500


namespace NUMINAMATH_GPT_min_remainder_n_div_2005_l465_46590

theorem min_remainder_n_div_2005 (n : ℕ) (hn_pos : 0 < n) 
  (h1 : n % 902 = 602) (h2 : n % 802 = 502) (h3 : n % 702 = 402) :
  n % 2005 = 101 :=
sorry

end NUMINAMATH_GPT_min_remainder_n_div_2005_l465_46590


namespace NUMINAMATH_GPT_long_sleeve_shirts_correct_l465_46518

def total_shirts : ℕ := 9
def short_sleeve_shirts : ℕ := 4
def long_sleeve_shirts : ℕ := total_shirts - short_sleeve_shirts

theorem long_sleeve_shirts_correct : long_sleeve_shirts = 5 := by
  sorry

end NUMINAMATH_GPT_long_sleeve_shirts_correct_l465_46518


namespace NUMINAMATH_GPT_point_in_third_quadrant_l465_46523

theorem point_in_third_quadrant
  (a b : ℝ)
  (hne : a ≠ 0)
  (y_increase : ∀ x1 x2, x1 < x2 → -5 * a * x1 + b < -5 * a * x2 + b)
  (ab_pos : a * b > 0) : 
  a < 0 ∧ b < 0 :=
by
  sorry

end NUMINAMATH_GPT_point_in_third_quadrant_l465_46523


namespace NUMINAMATH_GPT_intersection_of_complement_l465_46571

open Set

theorem intersection_of_complement (U : Set ℕ) (A : Set ℕ) (B : Set ℕ) (hU : U = {1, 2, 3, 4, 5, 6})
  (hA : A = {1, 3, 4}) (hB : B = {2, 3, 4, 5}) : A ∩ (U \ B) = {1} :=
by
  rw [hU, hA, hB]
  -- Proof steps go here
  sorry

end NUMINAMATH_GPT_intersection_of_complement_l465_46571


namespace NUMINAMATH_GPT_compare_store_costs_l465_46566

-- Define the conditions mathematically
def StoreA_cost (x : ℕ) : ℝ := 5 * x + 125
def StoreB_cost (x : ℕ) : ℝ := 4.5 * x + 135

theorem compare_store_costs (x : ℕ) (h : x ≥ 5) : 
  5 * 15 + 125 = 200 ∧ 4.5 * 15 + 135 = 202.5 ∧ 200 < 202.5 := 
by
  -- Here the theorem states the claims to be proved.
  sorry

end NUMINAMATH_GPT_compare_store_costs_l465_46566


namespace NUMINAMATH_GPT_five_y_eq_45_over_7_l465_46549

theorem five_y_eq_45_over_7 (x y : ℚ) (h1 : 3 * x + 4 * y = 0) (h2 : x = y - 3) : 5 * y = 45 / 7 := by
  sorry

end NUMINAMATH_GPT_five_y_eq_45_over_7_l465_46549


namespace NUMINAMATH_GPT_sally_initial_cards_l465_46521

theorem sally_initial_cards (X : ℕ) (h1 : X + 41 + 20 = 88) : X = 27 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_sally_initial_cards_l465_46521


namespace NUMINAMATH_GPT_length_of_train_l465_46532

theorem length_of_train (speed_km_hr : ℝ) (platform_length_m : ℝ) (time_sec : ℝ) 
  (h1 : speed_km_hr = 72) (h2 : platform_length_m = 250) (h3 : time_sec = 30) : 
  ∃ (train_length : ℝ), train_length = 350 := 
by 
  -- Definitions of the given conditions
  let speed_m_per_s := speed_km_hr * (5 / 18)
  let total_distance := speed_m_per_s * time_sec
  let train_length := total_distance - platform_length_m
  -- Verifying the length of the train
  use train_length
  sorry

end NUMINAMATH_GPT_length_of_train_l465_46532


namespace NUMINAMATH_GPT_line_intersects_y_axis_at_0_2_l465_46553

theorem line_intersects_y_axis_at_0_2 :
  ∃ y : ℝ, (2, 8) ≠ (4, 14) ∧ ∀ x: ℝ, (3 * x + y = 2) ∧ x = 0 → y = 2 :=
by
  sorry

end NUMINAMATH_GPT_line_intersects_y_axis_at_0_2_l465_46553


namespace NUMINAMATH_GPT_max_items_per_cycle_l465_46598

theorem max_items_per_cycle (shirts : Nat) (pants : Nat) (sweaters : Nat) (jeans : Nat)
  (cycle_time : Nat) (total_time : Nat) 
  (h_shirts : shirts = 18)
  (h_pants : pants = 12)
  (h_sweaters : sweaters = 17)
  (h_jeans : jeans = 13)
  (h_cycle_time : cycle_time = 45)
  (h_total_time : total_time = 3 * 60) :
  (shirts + pants + sweaters + jeans) / (total_time / cycle_time) = 15 :=
by
  -- We will provide the proof here
  sorry

end NUMINAMATH_GPT_max_items_per_cycle_l465_46598


namespace NUMINAMATH_GPT_avg_height_of_class_is_168_6_l465_46506

noncomputable def avgHeightClass : ℕ → ℕ → ℕ → ℕ → ℚ :=
  λ n₁ h₁ n₂ h₂ => (n₁ * h₁ + n₂ * h₂) / (n₁ + n₂)

theorem avg_height_of_class_is_168_6 :
  avgHeightClass 40 169 10 167 = 168.6 := 
by 
  sorry

end NUMINAMATH_GPT_avg_height_of_class_is_168_6_l465_46506


namespace NUMINAMATH_GPT_largest_three_digit_sum_fifteen_l465_46593

theorem largest_three_digit_sum_fifteen : ∃ (a b c : ℕ), (a = 9 ∧ b = 6 ∧ c = 0 ∧ 100 * a + 10 * b + c = 960 ∧ a + b + c = 15 ∧ a < 10 ∧ b < 10 ∧ c < 10) := by
  sorry

end NUMINAMATH_GPT_largest_three_digit_sum_fifteen_l465_46593


namespace NUMINAMATH_GPT_train_overtakes_motorbike_in_80_seconds_l465_46541

-- Definitions of the given conditions
def speed_train_kmph : ℝ := 100
def speed_motorbike_kmph : ℝ := 64
def length_train_m : ℝ := 800.064

-- Definition to convert kmph to m/s
noncomputable def kmph_to_mps (speed_kmph : ℝ) : ℝ :=
  speed_kmph * 1000 / 3600

-- Relative speed in m/s
noncomputable def relative_speed_mps : ℝ :=
  kmph_to_mps (speed_train_kmph - speed_motorbike_kmph)

-- Time taken for the train to overtake the motorbike
noncomputable def time_to_overtake (distance_m : ℝ) (speed_mps : ℝ) : ℝ :=
  distance_m / speed_mps

-- The statement to be proved
theorem train_overtakes_motorbike_in_80_seconds :
  time_to_overtake length_train_m relative_speed_mps = 80.0064 :=
by
  sorry

end NUMINAMATH_GPT_train_overtakes_motorbike_in_80_seconds_l465_46541


namespace NUMINAMATH_GPT_problem_solution_l465_46573

noncomputable def greatest_integer_not_exceeding (z : ℝ) : ℤ := Int.floor z

theorem problem_solution (x : ℝ) (y : ℝ) 
  (h1 : y = 4 * greatest_integer_not_exceeding x + 4)
  (h2 : y = 5 * greatest_integer_not_exceeding (x - 3) + 7)
  (h3 : x > 3 ∧ ¬ ∃ (n : ℤ), x = ↑n) :
  64 < x + y ∧ x + y < 65 :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l465_46573


namespace NUMINAMATH_GPT_initial_cost_of_article_correct_l465_46576

noncomputable def initial_cost_of_article (final_cost : ℝ) : ℝ :=
  final_cost / (0.75 * 0.85 * 1.10 * 1.05)

theorem initial_cost_of_article_correct (final_cost : ℝ) (h : final_cost = 1226.25) :
  initial_cost_of_article final_cost = 1843.75 :=
by
  rw [h]
  norm_num
  rw [initial_cost_of_article]
  simp [initial_cost_of_article]
  norm_num
  sorry

end NUMINAMATH_GPT_initial_cost_of_article_correct_l465_46576


namespace NUMINAMATH_GPT_sally_purchased_20_fifty_cent_items_l465_46565

noncomputable def num_fifty_cent_items (x y z : ℕ) (h1 : x + y + z = 30) (h2 : 50 * x + 500 * y + 1000 * z = 10000) : ℕ :=
x

theorem sally_purchased_20_fifty_cent_items
  (x y z : ℕ)
  (h1 : x + y + z = 30)
  (h2 : 50 * x + 500 * y + 1000 * z = 10000)
  : num_fifty_cent_items x y z h1 h2 = 20 :=
sorry

end NUMINAMATH_GPT_sally_purchased_20_fifty_cent_items_l465_46565


namespace NUMINAMATH_GPT_usual_walk_time_l465_46526

theorem usual_walk_time (S T : ℝ)
  (h : S / (2/3 * S) = (T + 15) / T) : T = 30 :=
by
  sorry

end NUMINAMATH_GPT_usual_walk_time_l465_46526


namespace NUMINAMATH_GPT_scalene_triangle_height_ratio_l465_46524

theorem scalene_triangle_height_ratio {a b c : ℝ} (h1 : a > b ∧ b > c ∧ a > c)
  (h2 : a + c = 2 * b) : 
  1 / 3 < c / a ∧ c / a < 1 :=
by sorry

end NUMINAMATH_GPT_scalene_triangle_height_ratio_l465_46524


namespace NUMINAMATH_GPT_lowest_point_graph_l465_46536

theorem lowest_point_graph (x : ℝ) (h : x > -1) : ∃ y, y = (x^2 + 2*x + 2) / (x + 1) ∧ y ≥ 2 ∧ (x = 0 → y = 2) :=
  sorry

end NUMINAMATH_GPT_lowest_point_graph_l465_46536


namespace NUMINAMATH_GPT_blue_string_length_is_320_l465_46509

-- Define the lengths of the strings
def red_string_length := 8
def white_string_length := 5 * red_string_length
def blue_string_length := 8 * white_string_length

-- The main theorem to prove
theorem blue_string_length_is_320 : blue_string_length = 320 := by
  sorry

end NUMINAMATH_GPT_blue_string_length_is_320_l465_46509


namespace NUMINAMATH_GPT_pages_in_second_chapter_l465_46507

theorem pages_in_second_chapter
  (total_pages : ℕ)
  (first_chapter_pages : ℕ)
  (second_chapter_pages : ℕ)
  (h1 : total_pages = 93)
  (h2 : first_chapter_pages = 60)
  (h3: second_chapter_pages = total_pages - first_chapter_pages) :
  second_chapter_pages = 33 :=
by
  sorry

end NUMINAMATH_GPT_pages_in_second_chapter_l465_46507


namespace NUMINAMATH_GPT_product_of_ninth_and_tenth_l465_46584

def scores_first_8 := [7, 4, 3, 6, 8, 3, 1, 5]
def total_points_first_8 := scores_first_8.sum

def condition1 (ninth_game_points tenth_game_points : ℕ) : Prop :=
  ninth_game_points < 10 ∧ tenth_game_points < 10

def condition2 (ninth_game_points : ℕ) : Prop :=
  (total_points_first_8 + ninth_game_points) % 9 = 0

def condition3 (ninth_game_points tenth_game_points : ℕ) : Prop :=
  (total_points_first_8 + ninth_game_points + tenth_game_points) % 10 = 0

theorem product_of_ninth_and_tenth (ninth_game_points : ℕ) (tenth_game_points : ℕ) 
  (h1 : condition1 ninth_game_points tenth_game_points)
  (h2 : condition2 ninth_game_points)
  (h3 : condition3 ninth_game_points tenth_game_points) : 
  ninth_game_points * tenth_game_points = 40 :=
sorry

end NUMINAMATH_GPT_product_of_ninth_and_tenth_l465_46584


namespace NUMINAMATH_GPT_birds_more_than_storks_l465_46587

theorem birds_more_than_storks :
  let birds := 6
  let initial_storks := 3
  let additional_storks := 2
  let total_storks := initial_storks + additional_storks
  birds - total_storks = 1 := by
  sorry

end NUMINAMATH_GPT_birds_more_than_storks_l465_46587


namespace NUMINAMATH_GPT_find_original_number_l465_46545

-- Variables and assumptions
variables (N y x : ℕ)

-- Conditions
def crossing_out_condition : Prop := N = 10 * y + x
def sum_condition : Prop := N + y = 54321
def remainder_condition : Prop := x = 54321 % 11
def y_value : Prop := y = 4938

-- The final proof problem statement
theorem find_original_number (h1 : crossing_out_condition N y x) (h2 : sum_condition N y) 
  (h3 : remainder_condition x) (h4 : y_value y) : N = 49383 :=
sorry

end NUMINAMATH_GPT_find_original_number_l465_46545


namespace NUMINAMATH_GPT_friends_picked_strawberries_with_Lilibeth_l465_46516

-- Define the conditions
def Lilibeth_baskets : ℕ := 6
def strawberries_per_basket : ℕ := 50
def total_strawberries : ℕ := 1200

-- Define the calculation of strawberries picked by Lilibeth
def Lilibeth_strawberries : ℕ := Lilibeth_baskets * strawberries_per_basket

-- Define the calculation of strawberries picked by friends
def friends_strawberries : ℕ := total_strawberries - Lilibeth_strawberries

-- Define the number of friends who picked strawberries
def friends_picked_with_Lilibeth : ℕ := friends_strawberries / Lilibeth_strawberries

-- The theorem we need to prove
theorem friends_picked_strawberries_with_Lilibeth : friends_picked_with_Lilibeth = 3 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_friends_picked_strawberries_with_Lilibeth_l465_46516


namespace NUMINAMATH_GPT_half_angle_third_quadrant_l465_46544

theorem half_angle_third_quadrant (α : ℝ) (k : ℤ) (h1 : k * 360 + 180 < α) (h2 : α < k * 360 + 270) : 
  (∃ n : ℤ, n * 360 + 90 < (α / 2) ∧ (α / 2) < n * 360 + 135) ∨ 
  (∃ n : ℤ, n * 360 + 270 < (α / 2) ∧ (α / 2) < n * 360 + 315) := 
sorry

end NUMINAMATH_GPT_half_angle_third_quadrant_l465_46544


namespace NUMINAMATH_GPT_sum_of_integers_is_24_l465_46543

theorem sum_of_integers_is_24 (x y : ℕ) (hx : x > y) (h1 : x - y = 4) (h2 : x * y = 132) : x + y = 24 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_integers_is_24_l465_46543


namespace NUMINAMATH_GPT_find_A_l465_46575

def spadesuit (A B : ℝ) : ℝ := 4 * A + 3 * B + 6

theorem find_A (A : ℝ) (h : spadesuit A 5 = 59) : A = 9.5 :=
by sorry

end NUMINAMATH_GPT_find_A_l465_46575


namespace NUMINAMATH_GPT_estimated_value_of_n_l465_46574

-- Definitions from the conditions of the problem
def total_balls (n : ℕ) : ℕ := n + 18 + 9
def probability_of_yellow (n : ℕ) : ℚ := 18 / total_balls n

-- The theorem stating what we need to prove
theorem estimated_value_of_n : ∃ n : ℕ, probability_of_yellow n = 0.30 ∧ n = 42 :=
by {
  sorry
}

end NUMINAMATH_GPT_estimated_value_of_n_l465_46574


namespace NUMINAMATH_GPT_num_pairs_divisible_7_l465_46525

theorem num_pairs_divisible_7 (x y : ℕ) (hx : 1 ≤ x ∧ x ≤ 1000) (hy : 1 ≤ y ∧ y ≤ 1000)
  (divisible : (x^2 + y^2) % 7 = 0) : 
  (∃ k : ℕ, k = 20164) :=
sorry

end NUMINAMATH_GPT_num_pairs_divisible_7_l465_46525


namespace NUMINAMATH_GPT_hyperbola_asymptotes_l465_46548

noncomputable def hyperbola_eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 + (b^2 / a^2))

theorem hyperbola_asymptotes (a b : ℝ) (h : hyperbola_eccentricity a b = Real.sqrt 3) :
  (∀ x y : ℝ, (y = Real.sqrt 2 * x) ∨ (y = -Real.sqrt 2 * x)) :=
sorry

end NUMINAMATH_GPT_hyperbola_asymptotes_l465_46548


namespace NUMINAMATH_GPT_All_Yarns_are_Zorps_and_Xings_l465_46512

-- Define the basic properties
variables {α : Type}
variable (Zorp Xing Yarn Wit Vamp : α → Prop)

-- Given conditions
axiom all_Zorps_are_Xings : ∀ z, Zorp z → Xing z
axiom all_Yarns_are_Xings : ∀ y, Yarn y → Xing y
axiom all_Wits_are_Zorps : ∀ w, Wit w → Zorp w
axiom all_Yarns_are_Wits : ∀ y, Yarn y → Wit y
axiom all_Yarns_are_Vamps : ∀ y, Yarn y → Vamp y

-- Proof problem
theorem All_Yarns_are_Zorps_and_Xings : 
  ∀ y, Yarn y → (Zorp y ∧ Xing y) :=
sorry

end NUMINAMATH_GPT_All_Yarns_are_Zorps_and_Xings_l465_46512


namespace NUMINAMATH_GPT_d_share_l465_46513

theorem d_share (T : ℝ) (A B C D E : ℝ) 
  (h1 : A = 5 / 15 * T) 
  (h2 : B = 2 / 15 * T) 
  (h3 : C = 4 / 15 * T)
  (h4 : D = 3 / 15 * T)
  (h5 : E = 1 / 15 * T)
  (combined_AC : A + C = 3 / 5 * T)
  (diff_BE : B - E = 250) : 
  D = 750 :=
by
  sorry

end NUMINAMATH_GPT_d_share_l465_46513


namespace NUMINAMATH_GPT_transform_parabola_l465_46577

theorem transform_parabola (a b c : ℝ) (h : a ≠ 0) :
  ∃ (f : ℝ → ℝ), (∀ x, f (a * x^2 + b * x + c) = x^2) :=
sorry

end NUMINAMATH_GPT_transform_parabola_l465_46577


namespace NUMINAMATH_GPT_smallest_denominator_fraction_interval_exists_l465_46515

def interval (a b c d : ℕ) : Prop :=
a = 14 ∧ b = 73 ∧ c = 5 ∧ d = 26

theorem smallest_denominator_fraction_interval_exists :
  ∃ (a b c d : ℕ), 
    a / b < 19 / 99 ∧ b < 99 ∧
    19 / 99 < c / d ∧ d < 99 ∧
    interval a b c d :=
by
  sorry

end NUMINAMATH_GPT_smallest_denominator_fraction_interval_exists_l465_46515


namespace NUMINAMATH_GPT_weight_of_bowling_ball_l465_46529

variable (b c : ℝ)

axiom h1 : 5 * b = 2 * c
axiom h2 : 3 * c = 84

theorem weight_of_bowling_ball : b = 11.2 :=
by
  sorry

end NUMINAMATH_GPT_weight_of_bowling_ball_l465_46529


namespace NUMINAMATH_GPT_wyatt_headmaster_duration_l465_46501

def Wyatt_start_month : Nat := 3 -- March
def Wyatt_break_start_month : Nat := 7 -- July
def Wyatt_break_end_month : Nat := 12 -- December
def Wyatt_end_year : Nat := 2011

def months_worked_before_break : Nat := Wyatt_break_start_month - Wyatt_start_month -- March to June (inclusive, hence -1)
def break_duration : Nat := 6
def months_worked_after_break : Nat := 12 -- January to December 2011

def total_months_worked : Nat := months_worked_before_break + months_worked_after_break
theorem wyatt_headmaster_duration : total_months_worked = 16 :=
by
  sorry

end NUMINAMATH_GPT_wyatt_headmaster_duration_l465_46501


namespace NUMINAMATH_GPT_monotonic_increasing_on_interval_min_value_on_interval_max_value_on_interval_l465_46551

noncomputable def f (x : ℝ) : ℝ := 1 - (3 / (x + 2))

theorem monotonic_increasing_on_interval :
  ∀ (x₁ x₂ : ℝ), 3 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 5 → f x₁ < f x₂ := sorry

theorem min_value_on_interval :
  ∃ (x : ℝ), x = 3 ∧ f x = 2 / 5 := sorry

theorem max_value_on_interval :
  ∃ (x : ℝ), x = 5 ∧ f x = 4 / 7 := sorry

end NUMINAMATH_GPT_monotonic_increasing_on_interval_min_value_on_interval_max_value_on_interval_l465_46551


namespace NUMINAMATH_GPT_remainder_7459_div_9_l465_46562

theorem remainder_7459_div_9 : 7459 % 9 = 7 := 
by
  sorry

end NUMINAMATH_GPT_remainder_7459_div_9_l465_46562


namespace NUMINAMATH_GPT_teal_total_sales_l465_46588

variable (pum_pie_slices_per_pie : ℕ) (cus_pie_slices_per_pie : ℕ)
variable (pum_pie_price_per_slice : ℕ) (cus_pie_price_per_slice : ℕ)
variable (pum_pies_sold : ℕ) (cus_pies_sold : ℕ)

def total_slices_sold (slices_per_pie pies_sold : ℕ) : ℕ :=
  slices_per_pie * pies_sold

def total_sales (slices_sold price_per_slice : ℕ) : ℕ :=
  slices_sold * price_per_slice

theorem teal_total_sales
  (h1 : pum_pie_slices_per_pie = 8)
  (h2 : cus_pie_slices_per_pie = 6)
  (h3 : pum_pie_price_per_slice = 5)
  (h4 : cus_pie_price_per_slice = 6)
  (h5 : pum_pies_sold = 4)
  (h6 : cus_pies_sold = 5) :
  (total_sales (total_slices_sold pum_pie_slices_per_pie pum_pies_sold) pum_pie_price_per_slice) +
  (total_sales (total_slices_sold cus_pie_slices_per_pie cus_pies_sold) cus_pie_price_per_slice) = 340 :=
by
  sorry

end NUMINAMATH_GPT_teal_total_sales_l465_46588


namespace NUMINAMATH_GPT_no_triangle_satisfies_condition_l465_46572

theorem no_triangle_satisfies_condition (x y z : ℝ) (h_tri : x + y > z ∧ x + z > y ∧ y + z > x) :
  x^3 + y^3 + z^3 ≠ (x + y) * (y + z) * (z + x) :=
by
  sorry

end NUMINAMATH_GPT_no_triangle_satisfies_condition_l465_46572


namespace NUMINAMATH_GPT_arithmetic_sequence_fifth_term_l465_46505

theorem arithmetic_sequence_fifth_term (x y : ℝ) (h1 : x = 2) (h2 : y = 1) :
    let a1 := x^2 + y^2
    let a2 := x^2 - y^2
    let a3 := x^2 * y^2
    let a4 := x^2 / y^2
    let d := a2 - a1
    let a5 := a4 + d
    a5 = 2 := by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_fifth_term_l465_46505


namespace NUMINAMATH_GPT_particles_probability_computation_l465_46519

theorem particles_probability_computation : 
  let L0 := 32
  let R0 := 68
  let N := 100
  let a := 1
  let b := 2
  let P_all_on_left := (a:ℚ) / b
  100 * a + b = 102 := by
  sorry

end NUMINAMATH_GPT_particles_probability_computation_l465_46519


namespace NUMINAMATH_GPT_simplify_expression_l465_46547

theorem simplify_expression (x : ℝ) : 7 * x + 9 - 3 * x + 15 * 2 = 4 * x + 39 := 
by sorry

end NUMINAMATH_GPT_simplify_expression_l465_46547


namespace NUMINAMATH_GPT_number_of_tires_slashed_l465_46583

-- Definitions based on conditions
def cost_per_tire : ℤ := 250
def cost_window : ℤ := 700
def total_cost : ℤ := 1450

-- Proof statement
theorem number_of_tires_slashed : ∃ T : ℤ, cost_per_tire * T + cost_window = total_cost ∧ T = 3 := 
sorry

end NUMINAMATH_GPT_number_of_tires_slashed_l465_46583


namespace NUMINAMATH_GPT_smallest_number_is_27_l465_46540

theorem smallest_number_is_27 (a b c : ℕ) (h_mean : (a + b + c) / 3 = 30) (h_median : b = 28) (h_largest : c = b + 7) : a = 27 :=
by {
  sorry
}

end NUMINAMATH_GPT_smallest_number_is_27_l465_46540


namespace NUMINAMATH_GPT_circle_equation_correct_l465_46511

theorem circle_equation_correct (x y : ℝ) :
  let h : ℝ := -2
  let k : ℝ := 2
  let r : ℝ := 5
  ((x - h)^2 + (y - k)^2 = r^2) ↔ ((x + 2)^2 + (y - 2)^2 = 25) :=
by
  sorry

end NUMINAMATH_GPT_circle_equation_correct_l465_46511
