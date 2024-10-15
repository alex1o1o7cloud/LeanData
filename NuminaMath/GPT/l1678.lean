import Mathlib

namespace NUMINAMATH_GPT_turnip_weight_possible_l1678_167877

-- Define the weights of the 6 bags
def bag_weights : List ℕ := [13, 15, 16, 17, 21, 24]

-- Define the weight of the turnip bag
def is_turnip_bag (T : ℕ) : Prop :=
  T ∈ bag_weights ∧
  ∃ O : ℕ, 3 * O = (bag_weights.sum - T)

theorem turnip_weight_possible : ∀ T, is_turnip_bag T ↔ T = 13 ∨ T = 16 :=
by sorry

end NUMINAMATH_GPT_turnip_weight_possible_l1678_167877


namespace NUMINAMATH_GPT_johns_leisure_travel_miles_per_week_l1678_167848

-- Define the given conditions
def mpg : Nat := 30
def work_round_trip_miles : Nat := 20 * 2  -- 20 miles to work + 20 miles back home
def work_days_per_week : Nat := 5
def weekly_fuel_usage_gallons : Nat := 8

-- Define the property to prove
theorem johns_leisure_travel_miles_per_week :
  let work_miles_per_week := work_round_trip_miles * work_days_per_week
  let total_possible_miles := weekly_fuel_usage_gallons * mpg
  let leisure_miles := total_possible_miles - work_miles_per_week
  leisure_miles = 40 :=
by
  sorry

end NUMINAMATH_GPT_johns_leisure_travel_miles_per_week_l1678_167848


namespace NUMINAMATH_GPT_Rudolph_stop_signs_l1678_167899

def distance : ℕ := 5 + 2
def stopSignsPerMile : ℕ := 2
def totalStopSigns : ℕ := distance * stopSignsPerMile

theorem Rudolph_stop_signs :
  totalStopSigns = 14 := 
  by sorry

end NUMINAMATH_GPT_Rudolph_stop_signs_l1678_167899


namespace NUMINAMATH_GPT_compare_logs_l1678_167839

noncomputable def a : ℝ := Real.log 0.3 / Real.log 0.2
noncomputable def b : ℝ := Real.log 3 / Real.log 2
noncomputable def c : ℝ := Real.log 5 / Real.log 3

theorem compare_logs : b > c ∧ c > a :=
by
  sorry

end NUMINAMATH_GPT_compare_logs_l1678_167839


namespace NUMINAMATH_GPT_value_of_a_5_l1678_167863

-- Define the sequence with the general term formula
def a (n : ℕ) : ℕ := 4 * n - 3

-- Prove that the value of a_5 is 17
theorem value_of_a_5 : a 5 = 17 := by
  sorry

end NUMINAMATH_GPT_value_of_a_5_l1678_167863


namespace NUMINAMATH_GPT_perpendicular_vectors_x_value_l1678_167804

theorem perpendicular_vectors_x_value
  (a : ℝ × ℝ) (b : ℝ × ℝ) (h : a = (1, -2)) (hb : b = (-3, x))
  (h_perp : a.1 * b.1 + a.2 * b.2 = 0) :
  x = -3 / 2 := by
  sorry

end NUMINAMATH_GPT_perpendicular_vectors_x_value_l1678_167804


namespace NUMINAMATH_GPT_vertex_angle_isosceles_triangle_l1678_167836

theorem vertex_angle_isosceles_triangle (α : ℝ) (β : ℝ) (sum_of_angles : α + α + β = 180) (base_angle : α = 50) :
  β = 80 :=
by
  sorry

end NUMINAMATH_GPT_vertex_angle_isosceles_triangle_l1678_167836


namespace NUMINAMATH_GPT_functional_equation_solution_l1678_167823

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (f x + y) = f (x^2 - y) + 4 * f x * y) →
  (∀ x : ℝ, f x = 0 ∨ f x = x^2) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_functional_equation_solution_l1678_167823


namespace NUMINAMATH_GPT_unique_root_ln_eqn_l1678_167885

/-- For what values of the parameter \(a\) does the equation
   \(\ln(x - 2a) - 3(x - 2a)^2 + 2a = 0\) have a unique root? -/
theorem unique_root_ln_eqn (a : ℝ) :
  ∃! x : ℝ, (Real.log (x - 2 * a) - 3 * (x - 2 * a) ^ 2 + 2 * a = 0) ↔
  a = (1 + Real.log 6) / 4 :=
sorry

end NUMINAMATH_GPT_unique_root_ln_eqn_l1678_167885


namespace NUMINAMATH_GPT_completing_the_square_solution_l1678_167831

theorem completing_the_square_solution : ∀ x : ℝ, x^2 + 8 * x + 9 = 0 ↔ (x + 4)^2 = 7 := by
  sorry

end NUMINAMATH_GPT_completing_the_square_solution_l1678_167831


namespace NUMINAMATH_GPT_factorize_expression_l1678_167835

variable {R : Type} [CommRing R]

theorem factorize_expression (x y : R) : 
  4 * (x + y)^2 - (x^2 - y^2)^2 = (x + y)^2 * (2 + x - y) * (2 - x + y) := 
by 
  sorry

end NUMINAMATH_GPT_factorize_expression_l1678_167835


namespace NUMINAMATH_GPT_lowest_dropped_score_l1678_167887

theorem lowest_dropped_score (A B C D : ℕ)
  (h1 : (A + B + C + D) / 4 = 50)
  (h2 : (A + B + C) / 3 = 55) :
  D = 35 :=
by
  sorry

end NUMINAMATH_GPT_lowest_dropped_score_l1678_167887


namespace NUMINAMATH_GPT_Mary_works_hours_on_Tuesday_and_Thursday_l1678_167850

theorem Mary_works_hours_on_Tuesday_and_Thursday 
  (h_mon_wed_fri : ∀ (d : ℕ), d = 3 → 9 * d = 27)
  (weekly_earnings : ℕ)
  (hourly_rate : ℕ)
  (weekly_hours_mon_wed_fri : ℕ)
  (tue_thu_hours : ℕ) :
  weekly_earnings = 407 →
  hourly_rate = 11 →
  weekly_hours_mon_wed_fri = 9 * 3 →
  weekly_earnings - weekly_hours_mon_wed_fri * hourly_rate = tue_thu_hours * hourly_rate →
  tue_thu_hours = 10 :=
by
  intros hearnings hrate hweek hsub
  sorry

end NUMINAMATH_GPT_Mary_works_hours_on_Tuesday_and_Thursday_l1678_167850


namespace NUMINAMATH_GPT_find_a_l1678_167821

theorem find_a (a : ℝ) (h1 : ∀ θ : ℝ, x = a + 4 * Real.cos θ ∧ y = 1 + 4 * Real.sin θ)
  (h2 : ∃ p : ℝ × ℝ, (3 * p.1 + 4 * p.2 - 5 = 0 ∧ (∃ θ : ℝ, p = (a + 4 * Real.cos θ, 1 + 4 * Real.sin θ))))
  (h3 : ∀ (p1 p2 : ℝ × ℝ), 
        (3 * p1.1 + 4 * p1.2 - 5 = 0 ∧ 3 * p2.1 + 4 * p2.2 - 5 = 0) ∧
        (∃ θ1 : ℝ, p1 = (a + 4 * Real.cos θ1, 1 + 4 * Real.sin θ1)) ∧
        (∃ θ2 : ℝ, p2 = (a + 4 * Real.cos θ2, 1 + 4 * Real.sin θ2)) → p1 = p2) :
  a = 7 := by
  sorry

end NUMINAMATH_GPT_find_a_l1678_167821


namespace NUMINAMATH_GPT_complex_modulus_z_l1678_167815

-- Define the complex number z with given conditions
noncomputable def z : ℂ := (2 + Complex.I) / Complex.I + Complex.I

-- State the theorem to be proven
theorem complex_modulus_z : Complex.abs z = Real.sqrt 2 := 
sorry

end NUMINAMATH_GPT_complex_modulus_z_l1678_167815


namespace NUMINAMATH_GPT_remainder_of_expression_l1678_167873

theorem remainder_of_expression (x y u v : ℕ) (h : x = u * y + v) (Hv : 0 ≤ v ∧ v < y) :
  (if v + 2 < y then (x + 3 * u * y + 2) % y = v + 2
   else (x + 3 * u * y + 2) % y = v + 2 - y) :=
by sorry

end NUMINAMATH_GPT_remainder_of_expression_l1678_167873


namespace NUMINAMATH_GPT_turnips_bag_l1678_167892

theorem turnips_bag (weights : List ℕ) (h : weights = [13, 15, 16, 17, 21, 24])
  (turnips: ℕ)
  (is_turnip : turnips ∈ weights)
  (o c : ℕ)
  (h1 : o + c = 106 - turnips)
  (h2 : c = 2 * o) :
  turnips = 13 ∨ turnips = 16 := by
  sorry

end NUMINAMATH_GPT_turnips_bag_l1678_167892


namespace NUMINAMATH_GPT_quadratic_distinct_roots_l1678_167819

theorem quadratic_distinct_roots (m : ℝ) : 
  ((m - 2) * x ^ 2 + 2 * x + 1 = 0) → (m < 3 ∧ m ≠ 2) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_distinct_roots_l1678_167819


namespace NUMINAMATH_GPT_expression_equals_24_l1678_167880

-- Given values
def a := 7
def b := 4
def c := 1
def d := 7

-- Statement to prove
theorem expression_equals_24 : (a - b) * (c + d) = 24 := by
  sorry

end NUMINAMATH_GPT_expression_equals_24_l1678_167880


namespace NUMINAMATH_GPT_proof_op_l1678_167886

def op (A B : ℕ) : ℕ := (A * B) / 2

theorem proof_op (a b c : ℕ) : op (op 4 6) 9 = 54 := by
  sorry

end NUMINAMATH_GPT_proof_op_l1678_167886


namespace NUMINAMATH_GPT_relationship_y1_y2_l1678_167867

theorem relationship_y1_y2 (k y1 y2 : ℝ) 
  (h1 : y1 = (k^2 + 1) * (-3) - 5) 
  (h2 : y2 = (k^2 + 1) * 4 - 5) : 
  y1 < y2 :=
sorry

end NUMINAMATH_GPT_relationship_y1_y2_l1678_167867


namespace NUMINAMATH_GPT_maximize_parabola_area_l1678_167881

variable {a b : ℝ}

/--
The parabola y = ax^2 + bx is tangent to the line x + y = 4 within the first quadrant. 
Prove that the values of a and b that maximize the area S enclosed by this parabola and 
the x-axis are a = -1 and b = 3, and that the maximum value of S is 9/2.
-/
theorem maximize_parabola_area (hab_tangent : ∃ x y, y = a * x^2 + b * x ∧ y = 4 - x ∧ x > 0 ∧ y > 0) 
  (area_eqn : S = 1/6 * (b^3 / a^2)) : 
  a = -1 ∧ b = 3 ∧ S = 9/2 := 
sorry

end NUMINAMATH_GPT_maximize_parabola_area_l1678_167881


namespace NUMINAMATH_GPT_record_expenditure_l1678_167898

theorem record_expenditure (income recording expenditure : ℤ) (h : income = 100 ∧ recording = 100) :
  expenditure = -80 ↔ recording - expenditure = income - 80 :=
by
  sorry

end NUMINAMATH_GPT_record_expenditure_l1678_167898


namespace NUMINAMATH_GPT_solve_equation_l1678_167851

theorem solve_equation (x : ℝ) (h₁ : x ≠ -2) (h₂ : x ≠ 2)
  (h₃ : (3 * x + 6)/(x^2 + 5 * x + 6) = (3 - x)/(x - 2)) :
  x = 3 ∨ x = -3 :=
sorry

end NUMINAMATH_GPT_solve_equation_l1678_167851


namespace NUMINAMATH_GPT_problem_statement_l1678_167866

theorem problem_statement : (-1:ℤ) ^ 4 - (2 - (-3:ℤ) ^ 2) = 6 := by
  sorry  -- Proof will be provided separately

end NUMINAMATH_GPT_problem_statement_l1678_167866


namespace NUMINAMATH_GPT_option_B_correct_l1678_167854

theorem option_B_correct (a b : ℝ) (h : a < b) : a^3 < b^3 := sorry

end NUMINAMATH_GPT_option_B_correct_l1678_167854


namespace NUMINAMATH_GPT_determine_initial_sum_l1678_167820

def initial_sum_of_money (P r : ℝ) : Prop :=
  (600 = P + 2 * P * r) ∧ (700 = P + 2 * P * (r + 0.1))

theorem determine_initial_sum (P r : ℝ) (h : initial_sum_of_money P r) : P = 500 :=
by
  cases h with
  | intro h1 h2 =>
    sorry

end NUMINAMATH_GPT_determine_initial_sum_l1678_167820


namespace NUMINAMATH_GPT_hyperbola_asymptotes_l1678_167818

theorem hyperbola_asymptotes (x y : ℝ) : x^2 - 4 * y^2 = -1 → (x = 2 * y) ∨ (x = -2 * y) := 
by
  intro h
  sorry

end NUMINAMATH_GPT_hyperbola_asymptotes_l1678_167818


namespace NUMINAMATH_GPT_campaign_donation_ratio_l1678_167810

theorem campaign_donation_ratio (max_donation : ℝ) 
  (total_money : ℝ) 
  (percent_donations : ℝ) 
  (num_max_donors : ℕ) 
  (half_max_donation : ℝ) 
  (total_raised : ℝ) 
  (half_donation : ℝ) :
  total_money = total_raised * percent_donations →
  half_donation = max_donation / 2 →
  half_max_donation = num_max_donors * max_donation →
  total_money - half_max_donation = 1500 * half_donation →
  (1500 : ℝ) / (num_max_donors : ℝ) = 3 :=
sorry

end NUMINAMATH_GPT_campaign_donation_ratio_l1678_167810


namespace NUMINAMATH_GPT_frac_eq_l1678_167825

def my_at (a b : ℕ) := a * b + b^2
def my_hash (a b : ℕ) := a^2 + b + a * b^2

theorem frac_eq : my_at 4 3 / my_hash 4 3 = 21 / 55 :=
by
  sorry

end NUMINAMATH_GPT_frac_eq_l1678_167825


namespace NUMINAMATH_GPT_goods_train_speed_l1678_167837

theorem goods_train_speed
  (length_train : ℝ) (length_platform : ℝ) (time_seconds : ℝ)
  (h_train_length : length_train = 250.0416)
  (h_platform_length : length_platform = 270)
  (h_time : time_seconds = 26) :
  (length_train + length_platform) / time_seconds * 3.6 = 72 := by
    sorry

end NUMINAMATH_GPT_goods_train_speed_l1678_167837


namespace NUMINAMATH_GPT_find_number_of_ducks_l1678_167838

variable {D H : ℕ}

-- Definition of the conditions
def total_animals (D H : ℕ) : Prop := D + H = 11
def total_legs (D H : ℕ) : Prop := 2 * D + 4 * H = 30
def number_of_ducks (D : ℕ) : Prop := D = 7

-- Lean statement for the proof problem
theorem find_number_of_ducks (D H : ℕ) (h1 : total_animals D H) (h2 : total_legs D H) : number_of_ducks D :=
by
  sorry

end NUMINAMATH_GPT_find_number_of_ducks_l1678_167838


namespace NUMINAMATH_GPT_train_speed_l1678_167849

def train_length : ℝ := 250
def bridge_length : ℝ := 150
def time_to_cross : ℝ := 32

theorem train_speed :
  (train_length + bridge_length) / time_to_cross = 12.5 :=
by {
  sorry
}

end NUMINAMATH_GPT_train_speed_l1678_167849


namespace NUMINAMATH_GPT_slightly_used_crayons_count_l1678_167828

-- Definitions
def total_crayons := 120
def new_crayons := total_crayons * (1/3)
def broken_crayons := total_crayons * (20/100)
def slightly_used_crayons := total_crayons - new_crayons - broken_crayons

-- Theorem statement
theorem slightly_used_crayons_count :
  slightly_used_crayons = 56 :=
by
  sorry

end NUMINAMATH_GPT_slightly_used_crayons_count_l1678_167828


namespace NUMINAMATH_GPT_volume_percentage_error_l1678_167832

theorem volume_percentage_error (L W H : ℝ) (hL : L > 0) (hW : W > 0) (hH : H > 0) :
  let V_true := L * W * H
  let L_meas := 1.08 * L
  let W_meas := 1.12 * W
  let H_meas := 1.05 * H
  let V_calc := L_meas * W_meas * H_meas
  let percentage_error := ((V_calc - V_true) / V_true) * 100
  percentage_error = 25.424 :=
by
  sorry

end NUMINAMATH_GPT_volume_percentage_error_l1678_167832


namespace NUMINAMATH_GPT_vans_capacity_l1678_167814

-- Definitions based on the conditions
def num_students : ℕ := 22
def num_adults : ℕ := 2
def num_vans : ℕ := 3

-- The Lean statement (theorem to be proved)
theorem vans_capacity :
  (num_students + num_adults) / num_vans = 8 := 
by
  sorry

end NUMINAMATH_GPT_vans_capacity_l1678_167814


namespace NUMINAMATH_GPT_parabola_equation_l1678_167893

theorem parabola_equation (a : ℝ) : 
(∀ x y : ℝ, y = x → y = a * x^2)
∧ (∃ P : ℝ × ℝ, P = (2, 2) ∧ P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) 
  → A = (x₁, y₁) ∧ B = (x₂, y₂) ∧ y₁ = x₁ ∧ y₂ = x₂ ∧ x₂ = x₁ → 
  ∃ f : ℝ × ℝ, f.fst ≠ 0 ∧ f.snd = 0) →
  a = (1 : ℝ) / 7 := 
sorry

end NUMINAMATH_GPT_parabola_equation_l1678_167893


namespace NUMINAMATH_GPT_girls_boys_ratio_l1678_167841

-- Let g be the number of girls and b be the number of boys.
-- From the conditions, we have:
-- 1. Total students: g + b = 32
-- 2. More girls than boys: g = b + 6

theorem girls_boys_ratio
  (g b : ℕ) -- Declare number of girls and boys as natural numbers
  (h1 : g + b = 32) -- Total number of students
  (h2 : g = b + 6)  -- 6 more girls than boys
  : g = 19 ∧ b = 13 := 
sorry

end NUMINAMATH_GPT_girls_boys_ratio_l1678_167841


namespace NUMINAMATH_GPT_largest_D_l1678_167876

theorem largest_D (D : ℝ) : (∀ x y : ℝ, x^2 + 2 * y^2 + 3 ≥ D * (3 * x + 4 * y)) → D ≤ Real.sqrt (12 / 17) :=
by
  sorry

end NUMINAMATH_GPT_largest_D_l1678_167876


namespace NUMINAMATH_GPT_second_number_less_than_first_by_16_percent_l1678_167853

variable (X : ℝ)

theorem second_number_less_than_first_by_16_percent
  (h1 : X > 0)
  (first_num : ℝ := 0.75 * X)
  (second_num : ℝ := 0.63 * X) :
  (first_num - second_num) / first_num * 100 = 16 := by
  sorry

end NUMINAMATH_GPT_second_number_less_than_first_by_16_percent_l1678_167853


namespace NUMINAMATH_GPT_sum_three_consecutive_odd_integers_l1678_167843

theorem sum_three_consecutive_odd_integers (n : ℤ) 
  (h : n + (n + 4) = 150) : n + (n + 2) + (n + 4) = 225 :=
by
  sorry

end NUMINAMATH_GPT_sum_three_consecutive_odd_integers_l1678_167843


namespace NUMINAMATH_GPT_max_single_student_books_l1678_167878

-- Definitions and conditions
variable (total_students : ℕ) (no_books : ℕ) (one_book : ℕ) (two_books : ℕ)
variable (total_avg_books_per_student : ℕ)

-- Given data
def given_data : Prop :=
  total_students = 20 ∧ no_books = 2 ∧ one_book = 8 ∧
  two_books = 3 ∧ total_avg_books_per_student = 2

-- Maximum number of books any single student could borrow
theorem max_single_student_books (total_students no_books one_book two_books total_avg_books_per_student : ℕ) 
  (h : given_data total_students no_books one_book two_books total_avg_books_per_student) : 
  ∃ max_books_borrowed, max_books_borrowed = 8 :=
by
  sorry

end NUMINAMATH_GPT_max_single_student_books_l1678_167878


namespace NUMINAMATH_GPT_volume_of_larger_part_of_pyramid_proof_l1678_167845

noncomputable def volume_of_larger_part_of_pyramid (a b : ℝ) (inclined_angle : ℝ) (area_ratio : ℝ) : ℝ :=
let h_trapezoid := Real.sqrt ((2 * Real.sqrt 3)^2 - (Real.sqrt 3)^2 / 4)
let height_pyramid := (1 / 2) * h_trapezoid * Real.tan (inclined_angle)
let volume_total := (1 / 3) * (((a + b) / 2) * Real.sqrt ((a - b) ^ 2 + 4 * h_trapezoid ^ 2) * height_pyramid)
let volume_smaller := (1 / (5 + 7)) * 7 * volume_total
(volume_total - volume_smaller)

theorem volume_of_larger_part_of_pyramid_proof  :
  (volume_of_larger_part_of_pyramid 2 (Real.sqrt 3) (Real.pi / 6) (5 / 7) = 0.875) :=
by
sorry

end NUMINAMATH_GPT_volume_of_larger_part_of_pyramid_proof_l1678_167845


namespace NUMINAMATH_GPT_service_fee_calculation_l1678_167840

-- Problem definitions based on conditions
def cost_food : ℝ := 50
def tip : ℝ := 5
def total_spent : ℝ := 61
def service_fee_percentage (x : ℝ) : Prop := x = (12 / 50) * 100

-- The main statement to be proven, showing that the service fee percentage is 24%
theorem service_fee_calculation : service_fee_percentage 24 :=
by {
  sorry
}

end NUMINAMATH_GPT_service_fee_calculation_l1678_167840


namespace NUMINAMATH_GPT_sugar_per_batch_l1678_167889

variable (S : ℝ)

theorem sugar_per_batch :
  (8 * (4 + S) = 44) → (S = 1.5) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_sugar_per_batch_l1678_167889


namespace NUMINAMATH_GPT_determine_y_minus_x_l1678_167864

theorem determine_y_minus_x (x y : ℝ) (h1 : x + y = 360) (h2 : x / y = 3 / 5) : y - x = 90 := sorry

end NUMINAMATH_GPT_determine_y_minus_x_l1678_167864


namespace NUMINAMATH_GPT_crown_cost_before_tip_l1678_167897

theorem crown_cost_before_tip (total_paid : ℝ) (tip_percentage : ℝ) (crown_cost : ℝ) :
  total_paid = 22000 → tip_percentage = 0.10 → total_paid = crown_cost * (1 + tip_percentage) → crown_cost = 20000 :=
by
  sorry

end NUMINAMATH_GPT_crown_cost_before_tip_l1678_167897


namespace NUMINAMATH_GPT_total_road_signs_l1678_167806

def first_intersection_signs := 40
def second_intersection_signs := first_intersection_signs + (first_intersection_signs / 4)
def third_intersection_signs := 2 * second_intersection_signs
def fourth_intersection_signs := third_intersection_signs - 20

def total_signs := first_intersection_signs + second_intersection_signs + third_intersection_signs + fourth_intersection_signs

theorem total_road_signs : total_signs = 270 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_total_road_signs_l1678_167806


namespace NUMINAMATH_GPT_solve_for_x_l1678_167834

theorem solve_for_x (a b c x : ℝ) (h : x^2 + b^2 + c = (a + x)^2) : 
  x = (b^2 + c - a^2) / (2 * a) :=
by sorry

end NUMINAMATH_GPT_solve_for_x_l1678_167834


namespace NUMINAMATH_GPT_absolute_inequality_l1678_167895

theorem absolute_inequality (a b c : ℝ) (h : |a - c| < |b|) : |a| < |b| + |c| := 
sorry

end NUMINAMATH_GPT_absolute_inequality_l1678_167895


namespace NUMINAMATH_GPT_factorization_problem1_factorization_problem2_l1678_167857

variables {a b x y : ℝ}

theorem factorization_problem1 (a b x y : ℝ) : a * (x + y) - 2 * b * (x + y) = (x + y) * (a - 2 * b) :=
by sorry

theorem factorization_problem2 (a b : ℝ) : a^3 + 2 * a^2 * b + a * b^2 = a * (a + b)^2 :=
by sorry

end NUMINAMATH_GPT_factorization_problem1_factorization_problem2_l1678_167857


namespace NUMINAMATH_GPT_gcd_459_357_is_51_l1678_167891

-- Define the problem statement
theorem gcd_459_357_is_51 : Nat.gcd 459 357 = 51 :=
by
  -- Proof here
  sorry

end NUMINAMATH_GPT_gcd_459_357_is_51_l1678_167891


namespace NUMINAMATH_GPT_find_value_l1678_167847

variable (y : ℝ) (Q : ℝ)
axiom condition : 5 * (3 * y + 7 * Real.pi) = Q

theorem find_value : 10 * (6 * y + 14 * Real.pi) = 4 * Q :=
by
  sorry

end NUMINAMATH_GPT_find_value_l1678_167847


namespace NUMINAMATH_GPT_find_number_l1678_167884

theorem find_number (N : ℝ) (h1 : (4/5) * (3/8) * N = some_number)
                    (h2 : 2.5 * N = 199.99999999999997) :
  N = 79.99999999999999 := 
sorry

end NUMINAMATH_GPT_find_number_l1678_167884


namespace NUMINAMATH_GPT_distance_between_points_on_parabola_l1678_167852

theorem distance_between_points_on_parabola :
  ∀ (x1 x2 y1 y2 : ℝ), 
    (y1^2 = 4 * x1) → (y2^2 = 4 * x2) → (x2 = x1 + 2) → (|y2 - y1| = 4 * Real.sqrt x2 - 4 * Real.sqrt x1) →
    Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) = 8 :=
by
  intros x1 x2 y1 y2 h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_distance_between_points_on_parabola_l1678_167852


namespace NUMINAMATH_GPT_exam_score_impossible_l1678_167872

theorem exam_score_impossible (x y : ℕ) : 
  (5 * x + y = 97) ∧ (x + y ≤ 20) → false :=
by
  sorry

end NUMINAMATH_GPT_exam_score_impossible_l1678_167872


namespace NUMINAMATH_GPT_equilateral_triangle_of_condition_l1678_167858

theorem equilateral_triangle_of_condition (a b c : ℝ) (h : a^2 + b^2 + c^2 - a * b - b * c - c * a = 0) :
  a = b ∧ b = c := 
sorry

end NUMINAMATH_GPT_equilateral_triangle_of_condition_l1678_167858


namespace NUMINAMATH_GPT_least_sum_p_q_r_l1678_167824

theorem least_sum_p_q_r (p q r : ℕ) (hp : 1 < p) (hq : 1 < q) (hr : 1 < r)
  (h : 17 * (p + 1) = 28 * (q + 1) ∧ 28 * (q + 1) = 35 * (r + 1)) : p + q + r = 290 :=
  sorry

end NUMINAMATH_GPT_least_sum_p_q_r_l1678_167824


namespace NUMINAMATH_GPT_distance_range_l1678_167870

theorem distance_range (A_school_distance : ℝ) (B_school_distance : ℝ) (x : ℝ)
  (hA : A_school_distance = 3) (hB : B_school_distance = 2) :
  1 ≤ x ∧ x ≤ 5 :=
sorry

end NUMINAMATH_GPT_distance_range_l1678_167870


namespace NUMINAMATH_GPT_lydia_current_age_l1678_167875

def years_for_apple_tree_to_bear_fruit : ℕ := 7
def lydia_age_when_planted_tree : ℕ := 4
def lydia_age_when_eats_apple : ℕ := 11

theorem lydia_current_age 
  (h : lydia_age_when_eats_apple - lydia_age_when_planted_tree = years_for_apple_tree_to_bear_fruit) :
  lydia_age_when_eats_apple = 11 := 
by
  sorry

end NUMINAMATH_GPT_lydia_current_age_l1678_167875


namespace NUMINAMATH_GPT_cricket_target_runs_l1678_167856

def target_runs (first_10_overs_run_rate remaining_40_overs_run_rate : ℝ) : ℝ :=
  10 * first_10_overs_run_rate + 40 * remaining_40_overs_run_rate

theorem cricket_target_runs : target_runs 4.2 6 = 282 := by
  sorry

end NUMINAMATH_GPT_cricket_target_runs_l1678_167856


namespace NUMINAMATH_GPT_minimum_value_of_expression_l1678_167830

variable (a b c : ℝ)

noncomputable def expression (a b c : ℝ) := (a + b) / c + (a + c) / b + (b + c) / a

theorem minimum_value_of_expression (hp1 : 0 < a) (hp2 : 0 < b) (hp3 : 0 < c) (h1 : a = 2 * b) (h2 : a = 2 * c) :
  expression a b c = 9.25 := 
sorry

end NUMINAMATH_GPT_minimum_value_of_expression_l1678_167830


namespace NUMINAMATH_GPT_intersection_of_M_and_N_l1678_167861

-- Define the universal set
def U : Set ℝ := Set.univ

-- Define sets M and N
def M : Set ℝ := { x | x < 1 }
def N : Set ℝ := { x | 0 < x ∧ x < 2 }

-- The theorem to be proved
theorem intersection_of_M_and_N : M ∩ N = { x : ℝ | 0 < x ∧ x < 1 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_M_and_N_l1678_167861


namespace NUMINAMATH_GPT_value_of_expression_l1678_167800

theorem value_of_expression (x y : ℤ) (hx : x = -5) (hy : y = 8) : 2 * (x - y) ^ 2 - x * y = 378 :=
by
  rw [hx, hy]
  -- The proof goes here.
  sorry

end NUMINAMATH_GPT_value_of_expression_l1678_167800


namespace NUMINAMATH_GPT_sum_first_10_terms_arithmetic_seq_l1678_167813

theorem sum_first_10_terms_arithmetic_seq (a : ℕ → ℤ) (h : (a 4)^2 + (a 7)^2 + 2 * (a 4) * (a 7) = 9) :
  ∃ S, S = 10 * (a 4 + a 7) / 2 ∧ (S = 15 ∨ S = -15) := 
by
  sorry

end NUMINAMATH_GPT_sum_first_10_terms_arithmetic_seq_l1678_167813


namespace NUMINAMATH_GPT_largest_possible_median_l1678_167859

theorem largest_possible_median 
  (l : List ℕ)
  (h_l : l = [4, 5, 3, 7, 9, 6])
  (h_pos : ∀ n ∈ l, 0 < n)
  (additional : List ℕ)
  (h_additional_pos : ∀ n ∈ additional, 0 < n)
  (h_length : l.length + additional.length = 9) : 
  ∃ median, median = 7 :=
by
  sorry

end NUMINAMATH_GPT_largest_possible_median_l1678_167859


namespace NUMINAMATH_GPT_Q_subset_P_l1678_167871

-- Definitions of the sets P and Q
def P : Set ℝ := {x | x ≥ 5}
def Q : Set ℝ := {x | 5 ≤ x ∧ x ≤ 7}

-- Statement to prove the relationship between P and Q
theorem Q_subset_P : Q ⊆ P :=
by
  sorry

end NUMINAMATH_GPT_Q_subset_P_l1678_167871


namespace NUMINAMATH_GPT_common_difference_d_l1678_167802

theorem common_difference_d (a_1 d : ℝ) (h1 : a_1 + 2 * d = 4) (h2 : 9 * a_1 + 36 * d = 18) : d = -1 :=
by sorry

end NUMINAMATH_GPT_common_difference_d_l1678_167802


namespace NUMINAMATH_GPT_sum_divisible_by_10_l1678_167811

-- Define the problem statement
theorem sum_divisible_by_10 {n : ℕ} : (n^2 + (n+1)^2 + (n+2)^2 + (n+3)^2) % 10 = 0 ↔ ∃ t : ℕ, n = 5 * t + 1 :=
by sorry

end NUMINAMATH_GPT_sum_divisible_by_10_l1678_167811


namespace NUMINAMATH_GPT_packing_big_boxes_l1678_167868

def total_items := 8640
def items_per_small_box := 12
def small_boxes_per_big_box := 6

def num_big_boxes (total_items items_per_small_box small_boxes_per_big_box : ℕ) : ℕ :=
  (total_items / items_per_small_box) / small_boxes_per_big_box

theorem packing_big_boxes : num_big_boxes total_items items_per_small_box small_boxes_per_big_box = 120 :=
by
  sorry

end NUMINAMATH_GPT_packing_big_boxes_l1678_167868


namespace NUMINAMATH_GPT_difference_between_neutrons_and_electrons_l1678_167890

def proton_number : Nat := 118
def mass_number : Nat := 293

def number_of_neutrons : Nat := mass_number - proton_number
def number_of_electrons : Nat := proton_number

theorem difference_between_neutrons_and_electrons :
  (number_of_neutrons - number_of_electrons) = 57 := by
  sorry

end NUMINAMATH_GPT_difference_between_neutrons_and_electrons_l1678_167890


namespace NUMINAMATH_GPT_beth_speed_l1678_167817

noncomputable def beth_average_speed (jerry_speed : ℕ) (jerry_time_minutes : ℕ) (beth_extra_miles : ℕ) (beth_extra_time_minutes : ℕ) : ℚ :=
  let jerry_time_hours := jerry_time_minutes / 60
  let jerry_distance := jerry_speed * jerry_time_hours
  let beth_distance := jerry_distance + beth_extra_miles
  let beth_time_hours := (jerry_time_minutes + beth_extra_time_minutes) / 60
  beth_distance / beth_time_hours

theorem beth_speed {beth_avg_speed : ℚ}
  (jerry_speed : ℕ) (jerry_time_minutes : ℕ) (beth_extra_miles : ℕ) (beth_extra_time_minutes : ℕ)
  (h_jerry_speed : jerry_speed = 40)
  (h_jerry_time : jerry_time_minutes = 30)
  (h_beth_extra_miles : beth_extra_miles = 5)
  (h_beth_extra_time : beth_extra_time_minutes = 20) :
  beth_average_speed jerry_speed jerry_time_minutes beth_extra_miles beth_extra_time_minutes = 30 := 
by 
  -- Leaving out the proof steps
  sorry

end NUMINAMATH_GPT_beth_speed_l1678_167817


namespace NUMINAMATH_GPT_coprime_repeating_decimal_sum_l1678_167803

theorem coprime_repeating_decimal_sum (a b : ℕ) (h_coprime : Nat.gcd a b = 1) (h_fraction : (0.35 : ℝ) = a / b) : a + b = 19 :=
sorry

end NUMINAMATH_GPT_coprime_repeating_decimal_sum_l1678_167803


namespace NUMINAMATH_GPT_handshakes_4_handshakes_n_l1678_167829

-- Defining the number of handshakes for n people
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

-- Proving that the number of handshakes for 4 people is 6
theorem handshakes_4 : handshakes 4 = 6 := by
  sorry

-- Proving that the number of handshakes for n people is (n * (n - 1)) / 2
theorem handshakes_n (n : ℕ) : handshakes n = (n * (n - 1)) / 2 := by 
  sorry

end NUMINAMATH_GPT_handshakes_4_handshakes_n_l1678_167829


namespace NUMINAMATH_GPT_cards_thrown_away_l1678_167874

theorem cards_thrown_away (h1 : 3 * (52 / 2) + 3 * 52 - 200 = 34) : 34 = 34 :=
by sorry

end NUMINAMATH_GPT_cards_thrown_away_l1678_167874


namespace NUMINAMATH_GPT_geometric_sequence_third_term_l1678_167883

theorem geometric_sequence_third_term :
  ∀ (a_1 a_5 : ℚ) (r : ℚ), 
    a_1 = 1 / 2 →
    (a_1 * r^4) = a_5 →
    a_5 = 16 →
    (a_1 * r^2) = 2 := 
by
  intros a_1 a_5 r h1 h2 h3
  sorry

end NUMINAMATH_GPT_geometric_sequence_third_term_l1678_167883


namespace NUMINAMATH_GPT_total_jokes_l1678_167842

theorem total_jokes (jessy_jokes_saturday : ℕ) (alan_jokes_saturday : ℕ) 
  (jessy_next_saturday : ℕ) (alan_next_saturday : ℕ) (total_jokes_so_far : ℕ) :
  jessy_jokes_saturday = 11 → 
  alan_jokes_saturday = 7 → 
  jessy_next_saturday = 11 * 2 → 
  alan_next_saturday = 7 * 2 → 
  total_jokes_so_far = (jessy_jokes_saturday + alan_jokes_saturday) + (jessy_next_saturday + alan_next_saturday) → 
  total_jokes_so_far = 54 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4] at h5
  sorry

end NUMINAMATH_GPT_total_jokes_l1678_167842


namespace NUMINAMATH_GPT_activity_probability_l1678_167882

noncomputable def total_basic_events : ℕ := 3^4
noncomputable def favorable_events : ℕ := Nat.choose 4 2 * Nat.factorial 3

theorem activity_probability :
  (favorable_events : ℚ) / total_basic_events = 4 / 9 :=
by
  sorry

end NUMINAMATH_GPT_activity_probability_l1678_167882


namespace NUMINAMATH_GPT_sequence_expression_l1678_167869

theorem sequence_expression (a : ℕ → ℕ) (h₀ : a 1 = 33) (h₁ : ∀ n, a (n + 1) - a n = 2 * n) : 
  ∀ n, a n = n^2 - n + 33 :=
by
  sorry

end NUMINAMATH_GPT_sequence_expression_l1678_167869


namespace NUMINAMATH_GPT_no_pos_int_mult_5005_in_form_l1678_167894

theorem no_pos_int_mult_5005_in_form (i j : ℕ) (h₀ : 0 ≤ i) (h₁ : i < j) (h₂ : j ≤ 49) :
  ¬ ∃ k : ℕ, 5005 * k = 10^j - 10^i := by
  sorry

end NUMINAMATH_GPT_no_pos_int_mult_5005_in_form_l1678_167894


namespace NUMINAMATH_GPT_cone_base_circumference_l1678_167826

theorem cone_base_circumference (radius : ℝ) (angle : ℝ) (c_base : ℝ) :
  radius = 6 ∧ angle = 180 ∧ c_base = 6 * Real.pi →
  (c_base = (angle / 360) * (2 * Real.pi * radius)) :=
by
  intros h
  rcases h with ⟨h_radius, h_angle, h_c_base⟩
  rw [h_radius, h_angle]
  norm_num
  sorry

end NUMINAMATH_GPT_cone_base_circumference_l1678_167826


namespace NUMINAMATH_GPT_minimum_area_isosceles_trapezoid_l1678_167807

theorem minimum_area_isosceles_trapezoid (r x a d : ℝ) (h_circumscribed : a + d = 2 * x) (h_minimal : x ≥ 2 * r) :
  4 * r^2 ≤ (a + d) * r :=
by sorry

end NUMINAMATH_GPT_minimum_area_isosceles_trapezoid_l1678_167807


namespace NUMINAMATH_GPT_part1_part2_l1678_167865

noncomputable def f (a x : ℝ) := Real.exp (2 * x) - 4 * a * Real.exp x - 2 * a * x
noncomputable def g (a x : ℝ) := x^2 + 5 * a^2
noncomputable def F (a x : ℝ) := f a x + g a x

theorem part1 (a : ℝ) : (∀ x y : ℝ, x ≤ y → f a x ≤ f a y) ↔ a ≤ 0 :=
by sorry

theorem part2 (a : ℝ) : ∀ x : ℝ, F a x ≥ 4 * (1 - Real.log 2)^2 / 5 :=
by sorry

end NUMINAMATH_GPT_part1_part2_l1678_167865


namespace NUMINAMATH_GPT_James_delivers_2565_bags_in_a_week_l1678_167888

noncomputable def total_bags_delivered_in_a_week
  (days_15_bags : ℕ)
  (trips_per_day_15_bags : ℕ)
  (bags_per_trip_15 : ℕ)
  (days_20_bags : ℕ)
  (trips_per_day_20_bags : ℕ)
  (bags_per_trip_20 : ℕ) : ℕ :=
  (days_15_bags * trips_per_day_15_bags * bags_per_trip_15) + (days_20_bags * trips_per_day_20_bags * bags_per_trip_20)

theorem James_delivers_2565_bags_in_a_week :
  total_bags_delivered_in_a_week 3 25 15 4 18 20 = 2565 :=
by
  sorry

end NUMINAMATH_GPT_James_delivers_2565_bags_in_a_week_l1678_167888


namespace NUMINAMATH_GPT_domain_of_f_l1678_167827

noncomputable def f (x : ℝ) : ℝ := 1 / (x^2 - 3 * x + 2)

theorem domain_of_f :
  {x : ℝ | (x < 1) ∨ (1 < x ∧ x < 2) ∨ (x > 2)} = 
  {x : ℝ | f x ≠ 0} :=
sorry

end NUMINAMATH_GPT_domain_of_f_l1678_167827


namespace NUMINAMATH_GPT_division_of_powers_l1678_167879

theorem division_of_powers :
  (0.5 ^ 4) / (0.05 ^ 3) = 500 :=
by sorry

end NUMINAMATH_GPT_division_of_powers_l1678_167879


namespace NUMINAMATH_GPT_savings_calculation_l1678_167860

-- Define the conditions
def income := 17000
def ratio_income_expenditure := 5 / 4

-- Prove that the savings are Rs. 3400
theorem savings_calculation (h : income = 5 * 3400): (income - 4 * 3400) = 3400 :=
by sorry

end NUMINAMATH_GPT_savings_calculation_l1678_167860


namespace NUMINAMATH_GPT_solution_set_l1678_167844

theorem solution_set:
  (∃ x y : ℝ, x - y = 0 ∧ x^2 + y = 2) ↔ (∃ x y : ℝ, (x = 1 ∧ y = 1) ∨ (x = -2 ∧ y = -2)) :=
by
  sorry

end NUMINAMATH_GPT_solution_set_l1678_167844


namespace NUMINAMATH_GPT_geometric_mean_problem_l1678_167833

theorem geometric_mean_problem
  (a : Nat) (a1 : Nat) (a8 : Nat) (r : Rat) 
  (h1 : a1 = 6) (h2 : a8 = 186624) 
  (h3 : a8 = a1 * r^7) 
  : a = a1 * r^3 → a = 1296 := 
by
  sorry

end NUMINAMATH_GPT_geometric_mean_problem_l1678_167833


namespace NUMINAMATH_GPT_angle_of_inclination_45_l1678_167801

def plane (x y z : ℝ) : Prop := (x = y) ∧ (y = z)
def image_planes (x y : ℝ) : Prop := (x = 45 ∧ y = 45)

theorem angle_of_inclination_45 (t₁₂ : ℝ) :
  ∃ θ: ℝ, (plane t₁₂ t₁₂ t₁₂ → image_planes 45 45 → θ = 45) :=
sorry

end NUMINAMATH_GPT_angle_of_inclination_45_l1678_167801


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1678_167855

theorem solution_set_of_inequality :
  {x : ℝ | -6 * x^2 + 2 < x} = {x : ℝ | x < -2 / 3} ∪ {x : ℝ | x > 1 / 2} := 
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1678_167855


namespace NUMINAMATH_GPT_find_a7_in_arithmetic_sequence_l1678_167822

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n, a (n + 1) - a n = a 1 - a 0

theorem find_a7_in_arithmetic_sequence
  (a : ℕ → ℤ)
  (h_arith : arithmetic_sequence a)
  (h_a1 : a 1 = 2)
  (h_a3_a5 : a 3 + a 5 = 10) :
  a 7 = 8 :=
sorry

end NUMINAMATH_GPT_find_a7_in_arithmetic_sequence_l1678_167822


namespace NUMINAMATH_GPT_gcd_of_12347_and_9876_l1678_167816

theorem gcd_of_12347_and_9876 : Nat.gcd 12347 9876 = 7 :=
by
  sorry

end NUMINAMATH_GPT_gcd_of_12347_and_9876_l1678_167816


namespace NUMINAMATH_GPT_larger_integer_is_30_l1678_167809

-- Define the problem statement using the given conditions
theorem larger_integer_is_30 (a b : ℕ) (ha : a > 0) (hb : b > 0)
  (h1 : a / b = 5 / 2) (h2 : a * b = 360) :
  max a b = 30 :=
sorry

end NUMINAMATH_GPT_larger_integer_is_30_l1678_167809


namespace NUMINAMATH_GPT_relationship_among_p_q_a_b_l1678_167862

open Int

variables (a b p q : ℕ) (h1 : a > b) (h2 : Nat.gcd a b = p) (h3 : Nat.lcm a b = q)

theorem relationship_among_p_q_a_b : q ≥ a ∧ a > b ∧ b ≥ p :=
by
  sorry

end NUMINAMATH_GPT_relationship_among_p_q_a_b_l1678_167862


namespace NUMINAMATH_GPT_overall_percentage_loss_l1678_167846

noncomputable def original_price : ℝ := 100
noncomputable def increased_price : ℝ := original_price * 1.36
noncomputable def first_discount_price : ℝ := increased_price * 0.90
noncomputable def second_discount_price : ℝ := first_discount_price * 0.85
noncomputable def third_discount_price : ℝ := second_discount_price * 0.80
noncomputable def final_price_with_tax : ℝ := third_discount_price * 1.05
noncomputable def percentage_change : ℝ := ((final_price_with_tax - original_price) / original_price) * 100

theorem overall_percentage_loss : percentage_change = -12.6064 :=
by
  sorry

end NUMINAMATH_GPT_overall_percentage_loss_l1678_167846


namespace NUMINAMATH_GPT_calculate_lunch_break_duration_l1678_167808

noncomputable def paula_rate (p : ℝ) : Prop := p > 0
noncomputable def helpers_rate (h : ℝ) : Prop := h > 0
noncomputable def apprentice_rate (a : ℝ) : Prop := a > 0
noncomputable def lunch_break_duration (L : ℝ) : Prop := L >= 0

-- Monday's work equation
noncomputable def monday_work (p h a L : ℝ) (monday_work_done : ℝ) :=
  0.6 = (9 - L) * (p + h + a)

-- Tuesday's work equation
noncomputable def tuesday_work (h a L : ℝ) (tuesday_work_done : ℝ) :=
  0.3 = (7 - L) * (h + a)

-- Wednesday's work equation
noncomputable def wednesday_work (p a L : ℝ) (wednesday_work_done : ℝ) :=
  0.1 = (1.2 - L) * (p + a)

-- Final proof statement
theorem calculate_lunch_break_duration (p h a L : ℝ)
  (H1 : paula_rate p)
  (H2 : helpers_rate h)
  (H3 : apprentice_rate a)
  (H4 : lunch_break_duration L)
  (H5 : monday_work p h a L 0.6)
  (H6 : tuesday_work h a L 0.3)
  (H7 : wednesday_work p a L 0.1) :
  L = 1.4 :=
sorry

end NUMINAMATH_GPT_calculate_lunch_break_duration_l1678_167808


namespace NUMINAMATH_GPT_leak_drain_time_l1678_167896

theorem leak_drain_time (P L : ℕ → ℕ) (H1 : ∀ t, P t = 1 / 2) (H2 : ∀ t, P t - L t = 1 / 3) : 
  (1 / L 1) = 6 :=
by
  sorry

end NUMINAMATH_GPT_leak_drain_time_l1678_167896


namespace NUMINAMATH_GPT_find_xyz_squares_l1678_167812

theorem find_xyz_squares (x y z : ℤ)
  (h1 : x + y + z = 3)
  (h2 : x^3 + y^3 + z^3 = 3) :
  x^2 + y^2 + z^2 = 3 ∨ x^2 + y^2 + z^2 = 57 :=
sorry

end NUMINAMATH_GPT_find_xyz_squares_l1678_167812


namespace NUMINAMATH_GPT_exists_a_solution_iff_l1678_167805

theorem exists_a_solution_iff (b : ℝ) : 
  (∃ (a x y : ℝ), y = b - x^2 ∧ x^2 + y^2 + 2 * a^2 = 4 - 2 * a * (x + y)) ↔ 
  b ≥ -2 * Real.sqrt 2 - 1 / 4 := 
by 
  sorry

end NUMINAMATH_GPT_exists_a_solution_iff_l1678_167805
