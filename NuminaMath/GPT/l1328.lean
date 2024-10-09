import Mathlib

namespace average_other_students_l1328_132804

theorem average_other_students (total_students other_students : ℕ) (mean_score_first : ℕ) 
 (mean_score_class : ℕ) (mean_score_other : ℕ) (h1 : total_students = 20) (h2 : other_students = 10)
 (h3 : mean_score_first = 80) (h4 : mean_score_class = 70) :
 mean_score_other = 60 :=
by
  sorry

end average_other_students_l1328_132804


namespace find_a_l1328_132855

-- Define the domains of the functions f and g
def A : Set ℝ :=
  {x | x < -1 ∨ x ≥ 1}

def B (a : ℝ) : Set ℝ :=
  {x | 2 * a < x ∧ x < a + 1}

-- Restate the problem as a Lean proposition
theorem find_a (a : ℝ) (h : a < 1) (hb : B a ⊆ A) :
  a ∈ {x | x ≤ -2 ∨ (1 / 2 ≤ x ∧ x < 1)} :=
sorry

end find_a_l1328_132855


namespace cube_greater_than_quadratic_minus_linear_plus_one_l1328_132803

variable (x : ℝ)

theorem cube_greater_than_quadratic_minus_linear_plus_one (h : x > 1) :
  x^3 > x^2 - x + 1 := by
  sorry

end cube_greater_than_quadratic_minus_linear_plus_one_l1328_132803


namespace sum_of_coefficients_zero_l1328_132885

theorem sum_of_coefficients_zero (A B C D E F : ℝ) :
  (∀ x : ℝ,
    1 / (x * (x + 1) * (x + 2) * (x + 3) * (x + 4) * (x + 5)) =
      A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 4) + F / (x + 5)) →
  A + B + C + D + E + F = 0 :=
by
  intro h
  -- Proof omitted
  sorry

end sum_of_coefficients_zero_l1328_132885


namespace speed_of_stream_l1328_132806

-- Define the conditions as premises
def boat_speed_in_still_water : ℝ := 24
def travel_time_downstream : ℝ := 3
def distance_downstream : ℝ := 84

-- The effective speed downstream is the sum of the boat's speed and the speed of the stream
def effective_speed_downstream (stream_speed : ℝ) : ℝ :=
  boat_speed_in_still_water + stream_speed

-- The speed of the stream
theorem speed_of_stream (stream_speed : ℝ) :
  84 = effective_speed_downstream stream_speed * travel_time_downstream →
  stream_speed = 4 :=
by
  sorry

end speed_of_stream_l1328_132806


namespace max_value_of_x_plus_y_l1328_132841

variable (x y : ℝ)

-- Define the condition
def condition : Prop := x^2 + y + 3 * x - 3 = 0

-- Define the proof statement
theorem max_value_of_x_plus_y (hx : condition x y) : x + y ≤ 4 :=
sorry

end max_value_of_x_plus_y_l1328_132841


namespace selling_price_range_l1328_132846

theorem selling_price_range
  (unit_purchase_price : ℝ)
  (initial_selling_price : ℝ)
  (initial_sales_volume : ℝ)
  (price_increase_effect : ℝ)
  (daily_profit_threshold : ℝ)
  (x : ℝ) :
  unit_purchase_price = 8 →
  initial_selling_price = 10 →
  initial_sales_volume = 100 →
  price_increase_effect = 10 →
  daily_profit_threshold = 320 →
  (initial_selling_price - unit_purchase_price) * initial_sales_volume > daily_profit_threshold →
  12 < x → x < 16 →
  (x - unit_purchase_price) * (initial_sales_volume - price_increase_effect * (x - initial_selling_price)) > daily_profit_threshold :=
sorry

end selling_price_range_l1328_132846


namespace solution_set_abs_le_one_inteval_l1328_132822

theorem solution_set_abs_le_one_inteval (x : ℝ) : |x| ≤ 1 ↔ -1 ≤ x ∧ x ≤ 1 :=
by sorry

end solution_set_abs_le_one_inteval_l1328_132822


namespace exists_two_linear_functions_l1328_132816

-- Define the quadratic trinomials and their general forms
variables (a b c d e f : ℝ)
-- Assuming coefficients a and d are non-zero
variable (ha : a ≠ 0)
variable (hd : d ≠ 0)

-- Define the linear function
def ell (m n x : ℝ) : ℝ := m * x + n

-- Define the quadratic trinomials P(x) and Q(x) 
def P (x : ℝ) := a * x^2 + b * x + c
def Q (x : ℝ) := d * x^2 + e * x + f

-- Prove that there exist exactly two linear functions ell(x) that satisfy the condition for all x
theorem exists_two_linear_functions : 
  ∃ (m1 m2 n1 n2 : ℝ), 
  (∀ x, P a b c x = Q d e f (ell m1 n1 x)) ∧ 
  (∀ x, P a b c x = Q d e f (ell m2 n2 x)) := 
sorry

end exists_two_linear_functions_l1328_132816


namespace zero_point_interval_l1328_132831

noncomputable def f (x : ℝ) : ℝ := (4 / x) - (2^x)

theorem zero_point_interval : ∃ x : ℝ, (1 < x ∧ x < 1.5) ∧ f x = 0 :=
sorry

end zero_point_interval_l1328_132831


namespace chocolates_cost_l1328_132898

-- Define the conditions given in the problem.
def boxes_needed (candies_total : ℕ) (candies_per_box : ℕ) : ℕ := 
    candies_total / candies_per_box

def total_cost_without_discount (num_boxes : ℕ) (cost_per_box : ℕ) : ℕ := 
    num_boxes * cost_per_box

def discount (total_cost : ℕ) : ℕ := 
    total_cost * 10 / 100

def final_cost (total_cost : ℕ) (discount : ℕ) : ℕ :=
    total_cost - discount

-- Theorem stating the total cost of buying 660 chocolate after discount is $138.60
theorem chocolates_cost (candies_total : ℕ) (candies_per_box : ℕ) (cost_per_box : ℕ) : 
     candies_total = 660 ∧ candies_per_box = 30 ∧ cost_per_box = 7 → 
     final_cost (total_cost_without_discount (boxes_needed candies_total candies_per_box) cost_per_box) 
          (discount (total_cost_without_discount (boxes_needed candies_total candies_per_box) cost_per_box)) = 13860 := 
by 
    intros h
    let ⟨h1, h2, h3⟩ := h 
    sorry 

end chocolates_cost_l1328_132898


namespace evaluate_expression_l1328_132830

theorem evaluate_expression : (Real.sqrt ((Real.sqrt 2)^4))^6 = 64 := by
  sorry

end evaluate_expression_l1328_132830


namespace unique_intersection_l1328_132850

open Real

-- Defining the functions f and g as per the conditions
def f (b : ℝ) (x : ℝ) : ℝ := b * x^2 + 5 * x + 3
def g (x : ℝ) : ℝ := -2 * x - 2

-- The condition that the intersection occurs at one point translates to a specific b satisfying the discriminant condition.
theorem unique_intersection (b : ℝ) : (∃ x : ℝ, f b x = g x) ∧ (f b x = g x → ∀ y : ℝ, y ≠ x → f b y ≠ g y) ↔ b = 49 / 20 :=
by {
  sorry
}

end unique_intersection_l1328_132850


namespace cody_initial_marbles_l1328_132810

theorem cody_initial_marbles (x : ℕ) (h1 : x - 5 = 7) : x = 12 := by
  sorry

end cody_initial_marbles_l1328_132810


namespace express_in_scientific_notation_l1328_132894

theorem express_in_scientific_notation :
  (2370000 : ℝ) = 2.37 * 10^6 := 
by
  -- proof omitted
  sorry

end express_in_scientific_notation_l1328_132894


namespace runner_injury_point_l1328_132877

theorem runner_injury_point
  (v d : ℝ)
  (h1 : 2 * (40 - d) / v = d / v + 11)
  (h2 : 2 * (40 - d) / v = 22) :
  d = 20 := 
by
  sorry

end runner_injury_point_l1328_132877


namespace charles_cleaning_time_l1328_132808

theorem charles_cleaning_time :
  let Alice_time := 20
  let Bob_time := (3/4) * Alice_time
  let Charles_time := (2/3) * Bob_time
  Charles_time = 10 :=
by
  sorry

end charles_cleaning_time_l1328_132808


namespace area_of_square_eq_36_l1328_132857

theorem area_of_square_eq_36 :
  ∃ (s q : ℝ), q = 6 ∧ s = 10 ∧ (∃ p : ℝ, p = 24 ∧ (p / 4) * (p / 4) = 36) := 
by
  sorry

end area_of_square_eq_36_l1328_132857


namespace find_b_over_a_find_angle_B_l1328_132834

-- Definitions and main theorems
noncomputable def sides_in_triangle (A B C a b c : ℝ) : Prop :=
  a * (Real.sin A) * (Real.sin B) + b * (Real.cos A) ^ 2 = Real.sqrt 2 * a

noncomputable def cos_law_condition (a b c : ℝ) : Prop :=
  c^2 = b^2 + Real.sqrt 3 * a^2

theorem find_b_over_a {A B C a b c : ℝ} (h : sides_in_triangle A B C a b c) : b / a = Real.sqrt 2 :=
  sorry

theorem find_angle_B {A B C a b c : ℝ} (h1 : sides_in_triangle A B C a b c) (h2 : cos_law_condition a b c)
  (h3 : b / a = Real.sqrt 2) : B = Real.pi / 4 :=
  sorry

end find_b_over_a_find_angle_B_l1328_132834


namespace angle_A_is_60_degrees_l1328_132826

theorem angle_A_is_60_degrees
  (a b c : ℝ) (A : ℝ) 
  (h1 : (a + b + c) * (b + c - a) = 3 * b * c) 
  (h2 : 0 < A) (h3 : A < 180) : 
  A = 60 := 
  sorry

end angle_A_is_60_degrees_l1328_132826


namespace solve_for_x_l1328_132844

theorem solve_for_x (x : ℝ) (h : 3*x - 4*x + 5*x = 140) : x = 35 :=
by 
  sorry

end solve_for_x_l1328_132844


namespace log2_15_eq_formula_l1328_132863

theorem log2_15_eq_formula (a b : ℝ) (h1 : a = Real.log 6 / Real.log 3) (h2 : b = Real.log 20 / Real.log 5) :
  Real.log 15 / Real.log 2 = (2 * a + b - 3) / ((a - 1) * (b - 1)) :=
by
  sorry

end log2_15_eq_formula_l1328_132863


namespace alpha_beta_range_l1328_132832

theorem alpha_beta_range (α β : ℝ) (P : ℝ × ℝ)
  (h1 : α > 0) 
  (h2 : β > 0) 
  (h3 : P = (α, 3 * β))
  (circle_eq : (α - 1)^2 + 9 * (β^2) = 1) :
  1 < α + β ∧ α + β < 5 / 3 :=
sorry

end alpha_beta_range_l1328_132832


namespace total_bill_l1328_132805

def num_adults := 2
def num_children := 5
def cost_per_meal := 3

theorem total_bill : (num_adults + num_children) * cost_per_meal = 21 := 
by 
  sorry

end total_bill_l1328_132805


namespace range_of_a_l1328_132836

variable (a : ℝ) (x : ℝ)

theorem range_of_a
  (h1 : 2 * x < 3 * (x - 3) + 1)
  (h2 : (3 * x + 2) / 4 > x + a) :
  -11 / 4 ≤ a ∧ a < -5 / 2 :=
sorry

end range_of_a_l1328_132836


namespace math_pattern_l1328_132873

theorem math_pattern (n : ℕ) : (2 * n - 1) * (2 * n + 1) = (2 * n) ^ 2 - 1 :=
by
  sorry

end math_pattern_l1328_132873


namespace circle_areas_equal_l1328_132892

theorem circle_areas_equal :
  let r1 := 15
  let d2 := 30
  let r2 := d2 / 2
  let A1 := Real.pi * r1^2
  let A2 := Real.pi * r2^2
  A1 = A2 :=
by
  sorry

end circle_areas_equal_l1328_132892


namespace painting_price_after_5_years_l1328_132866

variable (P : ℝ)
-- Conditions on price changes over the years
def year1_price (P : ℝ) := P * 1.30
def year2_price (P : ℝ) := year1_price P * 0.80
def year3_price (P : ℝ) := year2_price P * 1.25
def year4_price (P : ℝ) := year3_price P * 0.90
def year5_price (P : ℝ) := year4_price P * 1.15

theorem painting_price_after_5_years (P : ℝ) :
  year5_price P = 1.3455 * P := by
  sorry

end painting_price_after_5_years_l1328_132866


namespace sedrich_more_jelly_beans_l1328_132871

-- Define the given conditions
def napoleon_jelly_beans : ℕ := 17
def mikey_jelly_beans : ℕ := 19
def sedrich_jelly_beans (x : ℕ) : ℕ := napoleon_jelly_beans + x

-- Define the main theorem to be proved
theorem sedrich_more_jelly_beans (x : ℕ) :
  2 * (napoleon_jelly_beans + sedrich_jelly_beans x) = 4 * mikey_jelly_beans → x = 4 :=
by
  -- Proving the theorem
  sorry

end sedrich_more_jelly_beans_l1328_132871


namespace expected_total_rain_correct_l1328_132845

-- Define the probabilities and rain amounts for one day.
def prob_sun : ℝ := 0.30
def prob_rain3 : ℝ := 0.40
def prob_rain8 : ℝ := 0.30
def rain_sun : ℝ := 0
def rain_three : ℝ := 3
def rain_eight : ℝ := 8
def days : ℕ := 7

-- Define the expected value of daily rain.
def E_daily_rain : ℝ :=
  prob_sun * rain_sun + prob_rain3 * rain_three + prob_rain8 * rain_eight

-- Define the expected total rain over seven days.
def E_total_rain : ℝ :=
  days * E_daily_rain

-- Statement of the proof problem.
theorem expected_total_rain_correct : E_total_rain = 25.2 := by
  -- Proof goes here
  sorry

end expected_total_rain_correct_l1328_132845


namespace like_terms_powers_eq_l1328_132820

theorem like_terms_powers_eq (m n : ℕ) :
  (-2 : ℝ) * (x : ℝ) * (y : ℝ) ^ m = (1 / 3 : ℝ) * (x : ℝ) ^ n * (y : ℝ) ^ 3 → m = 3 ∧ n = 1 :=
by
  sorry

end like_terms_powers_eq_l1328_132820


namespace triangle_side_count_l1328_132870

theorem triangle_side_count
    (x : ℝ)
    (h1 : x + 15 > 40)
    (h2 : x + 40 > 15)
    (h3 : 15 + 40 > x)
    (hx : ∃ (x : ℕ), 25 < x ∧ x < 55) :
    ∃ n : ℕ, n = 29 :=
by
  sorry

end triangle_side_count_l1328_132870


namespace ratio_x_y_l1328_132872

theorem ratio_x_y (x y : ℝ) (h : (1/x - 1/y) / (1/x + 1/y) = 2023) : (x + y) / (x - y) = -1 := 
by
  sorry

end ratio_x_y_l1328_132872


namespace second_smallest_packs_of_hot_dogs_l1328_132849

theorem second_smallest_packs_of_hot_dogs (n m : ℕ) (k : ℕ) :
  (12 * n ≡ 5 [MOD 10]) ∧ (10 * m ≡ 3 [MOD 12]) → n = 15 :=
by
  sorry

end second_smallest_packs_of_hot_dogs_l1328_132849


namespace tangent_line_circle_m_values_l1328_132835

theorem tangent_line_circle_m_values {m : ℝ} :
  (∀ (x y: ℝ), 3 * x + 4 * y + m = 0 → (x - 1)^2 + (y + 2)^2 = 4) →
  (m = 15 ∨ m = -5) :=
by
  sorry

end tangent_line_circle_m_values_l1328_132835


namespace min_time_shoe_horses_l1328_132880

variable (blacksmiths horses hooves_per_horse minutes_per_hoof : ℕ)
variable (total_time : ℕ)

theorem min_time_shoe_horses (h_blacksmiths : blacksmiths = 48) 
                            (h_horses : horses = 60)
                            (h_hooves_per_horse : hooves_per_horse = 4)
                            (h_minutes_per_hoof : minutes_per_hoof = 5)
                            (h_total_time : total_time = (horses * hooves_per_horse * minutes_per_hoof) / blacksmiths) :
                            total_time = 25 := 
by
  sorry

end min_time_shoe_horses_l1328_132880


namespace disk_max_areas_l1328_132852

-- Conditions Definition
def disk_divided (n : ℕ) : ℕ :=
  let radii := 3 * n
  let secant_lines := 2
  let total_areas := 9 * n
  total_areas

theorem disk_max_areas (n : ℕ) : disk_divided n = 9 * n :=
by
  sorry

end disk_max_areas_l1328_132852


namespace problem1_problem2_l1328_132853

open Set

-- Part (1)
theorem problem1 (a : ℝ) :
  (∀ x, x ∉ Icc (0 : ℝ) (2 : ℝ) → x ∈ Icc (a : ℝ) (3 - 2 * a : ℝ)) ∨ (∀ x, x ∈ Icc (a : ℝ) (3 - 2 * a : ℝ) → x ∉ Icc (0 : ℝ) (2 : ℝ)) → a ≤ 0 := 
sorry

-- Part (2)
theorem problem2 (a : ℝ) :
  (¬ ∀ x, x ∈ Icc (a : ℝ) (3 - 2 * a : ℝ) → x ∈ Icc (0 : ℝ) (2 : ℝ)) → (a < 0.5 ∨ a > 1) :=
sorry

end problem1_problem2_l1328_132853


namespace trajectory_description_l1328_132867

def trajectory_of_A (x y : ℝ) (m : ℝ) : Prop :=
  m * x^2 - y^2 = m ∧ y ≠ 0
  
theorem trajectory_description (x y m : ℝ) (h : m ≠ 0) :
  trajectory_of_A x y m →
    (m < -1 → (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (x^2 / a^2 + y^2 / b^2 = 1 ∧ ¬(x = -1 ∧ y = 0) ∧ ¬(x = 1 ∧ y = 0)))) ∧
    (m = -1 → (x^2 + y^2 = 1 ∧ ¬(x = -1 ∧ y = 0) ∧ ¬(x = 1 ∧ y = 0))) ∧
    (-1 < m ∧ m < 0 → (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (x^2 / a^2 + y^2 / b^2 = 1 ∧ ¬(x = -1 ∧ y = 0) ∧ ¬(x = 1 ∧ y = 0)))) ∧
    (m > 0 → (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (x^2 / a^2 - y^2 / b^2 = 1 ∧ ¬(x = -1 ∧ y = 0) ∧ ¬(x = 1 ∧ y = 0)))) :=
by
  intro h_trajectory
  sorry

end trajectory_description_l1328_132867


namespace minimum_value_l1328_132801

theorem minimum_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 20) :
  (∃ (m : ℝ), m = (1 / x ^ 2 + 1 / y ^ 2) ∧ m ≥ 2 / 25) :=
by
  sorry

end minimum_value_l1328_132801


namespace emily_trip_duration_same_l1328_132827

theorem emily_trip_duration_same (s : ℝ) (h_s_pos : 0 < s) : 
  let t1 := (90 : ℝ) / s
  let t2 := (360 : ℝ) / (4 * s)
  t2 = t1 := sorry

end emily_trip_duration_same_l1328_132827


namespace mixed_sum_in_range_l1328_132851

def mixed_to_improper (a : ℕ) (b c : ℕ) : ℚ := a + b / c

def mixed_sum (a1 a2 a3 b1 b2 b3 c1 c2 c3 : ℕ) : ℚ :=
  (mixed_to_improper a1 b1 c1) + (mixed_to_improper a2 b2 c2) + (mixed_to_improper a3 b3 c3)

theorem mixed_sum_in_range :
  11 < mixed_sum 1 4 6 3 1 2 8 3 21 ∧ mixed_sum 1 4 6 3 1 2 8 3 21 < 12 :=
by { sorry }

end mixed_sum_in_range_l1328_132851


namespace total_instruments_correct_l1328_132814

def fingers : Nat := 10
def hands : Nat := 2
def heads : Nat := 1

def trumpets := fingers - 3
def guitars := hands + 2
def trombones := heads + 2
def french_horns := guitars - 1
def violins := trumpets / 2
def saxophones := trombones / 3

theorem total_instruments_correct : 
  (trumpets + guitars = trombones + violins + saxophones) →
  trumpets + guitars + trombones + french_horns + violins + saxophones = 21 := by
  sorry

end total_instruments_correct_l1328_132814


namespace smallest_integer_solution_l1328_132818

theorem smallest_integer_solution (x : ℤ) :
  (7 - 5 * x < 12) → ∃ (n : ℤ), x = n ∧ n = 0 :=
by
  intro h
  sorry

end smallest_integer_solution_l1328_132818


namespace least_actual_square_area_l1328_132875

theorem least_actual_square_area :
  let side_measured := 7
  let lower_bound := 6.5
  let actual_area := lower_bound * lower_bound
  actual_area = 42.25 :=
by
  sorry

end least_actual_square_area_l1328_132875


namespace student_correct_answers_l1328_132817

theorem student_correct_answers (C I : ℕ) (h1 : C + I = 100) (h2 : C - 2 * I = 64) : C = 88 :=
by
  sorry

end student_correct_answers_l1328_132817


namespace variation_of_variables_l1328_132812

variables (k j : ℝ) (x y z : ℝ)

theorem variation_of_variables (h1 : x = k * y^2) (h2 : y = j * z^3) : ∃ m : ℝ, x = m * z^6 :=
by
  -- Placeholder for the proof
  sorry

end variation_of_variables_l1328_132812


namespace correct_answers_count_l1328_132884

theorem correct_answers_count
  (c w : ℕ)
  (h1 : c + w = 150)
  (h2 : 4 * c - 2 * w = 420) :
  c = 120 := by
  sorry

end correct_answers_count_l1328_132884


namespace shara_age_l1328_132889

-- Definitions derived from conditions
variables (S : ℕ) (J : ℕ)

-- Jaymee's age is twice Shara's age plus 2
def jaymee_age_relation : Prop := J = 2 * S + 2

-- Jaymee's age is given as 22
def jaymee_age : Prop := J = 22

-- The proof problem to prove Shara's age equals 10
theorem shara_age (h1 : jaymee_age_relation S J) (h2 : jaymee_age J) : S = 10 :=
by 
  sorry

end shara_age_l1328_132889


namespace distinct_nonzero_reals_xy_six_l1328_132838

theorem distinct_nonzero_reals_xy_six (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + 6/x = y + 6/y) (h_distinct : x ≠ y) : x * y = 6 := 
sorry

end distinct_nonzero_reals_xy_six_l1328_132838


namespace correct_statements_l1328_132807

-- Definitions
def p_A : ℚ := 1 / 2
def p_B : ℚ := 1 / 3

-- Statements to be verified
def statement1 := (p_A * (1 - p_B) + (1 - p_A) * p_B) = (1 / 2 + 1 / 3)
def statement2 := (p_A * p_B) = (1 / 2 * 1 / 3)
def statement3 := (p_A * (1 - p_B) + p_A * p_B) = (1 / 2 * 2 / 3 + 1 / 2 * 1 / 3)
def statement4 := (1 - (1 - p_A) * (1 - p_B)) = (1 - 1 / 2 * 2 / 3)

-- Theorem stating the correct sequence of statements
theorem correct_statements : (statement2 ∧ statement4) ∧ ¬(statement1 ∨ statement3) :=
by
  sorry

end correct_statements_l1328_132807


namespace friends_count_l1328_132829

variables (F : ℕ)
def cindy_initial_marbles : ℕ := 500
def marbles_per_friend : ℕ := 80
def marbles_given : ℕ := F * marbles_per_friend
def marbles_remaining := cindy_initial_marbles - marbles_given

theorem friends_count (h : 4 * marbles_remaining = 720) : F = 4 :=
by sorry

end friends_count_l1328_132829


namespace greatest_valid_number_l1328_132839

-- Define the conditions
def is_valid_number (n : ℕ) : Prop :=
  n < 200 ∧ Nat.gcd n 30 = 5

-- Formulate the proof problem
theorem greatest_valid_number : ∃ n, is_valid_number n ∧ (∀ m, is_valid_number m → m ≤ n) ∧ n = 185 := 
by
  sorry

end greatest_valid_number_l1328_132839


namespace find_y_l1328_132815

theorem find_y (y : ℝ) (hy : 0 < y) 
  (h : (Real.sqrt (12 * y)) * (Real.sqrt (6 * y)) * (Real.sqrt (18 * y)) * (Real.sqrt (9 * y)) = 27) : 
  y = 1 / 2 := 
sorry

end find_y_l1328_132815


namespace circle_center_radius_l1328_132802

theorem circle_center_radius {x y : ℝ} :
  (∃ r : ℝ, (x - 1)^2 + y^2 = r^2) ↔ (x^2 + y^2 - 2*x - 5 = 0) :=
by sorry

end circle_center_radius_l1328_132802


namespace total_snacks_l1328_132847

variable (peanuts : ℝ) (raisins : ℝ)

theorem total_snacks (h1 : peanuts = 0.1) (h2 : raisins = 0.4) : peanuts + raisins = 0.5 :=
by
  sorry

end total_snacks_l1328_132847


namespace greatest_xy_value_l1328_132887

theorem greatest_xy_value :
  ∃ (x y : ℕ), 0 < x ∧ 0 < y ∧ 7 * x + 5 * y = 200 ∧ x * y = 285 :=
by 
  sorry

end greatest_xy_value_l1328_132887


namespace arithmetic_mean_common_difference_l1328_132821

theorem arithmetic_mean_common_difference (a : ℕ → ℝ) (d : ℝ) 
    (h1 : ∀ n, a (n + 1) = a n + d) 
    (h2 : a 1 + a 4 = 2 * (a 2 + 1))
    : d = 2 := 
by 
  -- Proof is omitted as it is not required.
  sorry

end arithmetic_mean_common_difference_l1328_132821


namespace aitana_jayda_total_spending_l1328_132813

theorem aitana_jayda_total_spending (jayda_spent : ℤ) (more_fraction : ℚ) (jayda_spent_400 : jayda_spent = 400) (more_fraction_2_5 : more_fraction = 2 / 5) :
  jayda_spent + (jayda_spent + (more_fraction * jayda_spent)) = 960 :=
by
  sorry

end aitana_jayda_total_spending_l1328_132813


namespace people_per_apartment_l1328_132848

/-- A 25 story building has 4 apartments on each floor. 
There are 200 people in the building. 
Prove that each apartment houses 2 people. -/
theorem people_per_apartment (stories : ℕ) (apartments_per_floor : ℕ) (total_people : ℕ)
    (h_stories : stories = 25)
    (h_apartments_per_floor : apartments_per_floor = 4)
    (h_total_people : total_people = 200) :
  (total_people / (stories * apartments_per_floor)) = 2 :=
by
  sorry

end people_per_apartment_l1328_132848


namespace max_value_expression_l1328_132886

theorem max_value_expression (k : ℕ) (a b c : ℝ) (h : k > 0) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) (habc : a + b + c = 3 * k) :
  a^(3 * k - 1) * b + b^(3 * k - 1) * c + c^(3 * k - 1) * a + k^2 * a^k * b^k * c^k ≤ (3 * k - 1)^(3 * k - 1) :=
sorry

end max_value_expression_l1328_132886


namespace width_of_first_tv_is_24_l1328_132897

-- Define the conditions
def height_first_tv := 16
def cost_first_tv := 672
def width_new_tv := 48
def height_new_tv := 32
def cost_new_tv := 1152
def cost_per_sq_inch_diff := 1

-- Define the width of the first TV
def width_first_tv := 24

-- Define the areas
def area_first_tv (W : ℕ) := W * height_first_tv
def area_new_tv := width_new_tv * height_new_tv

-- Define the cost per square inch
def cost_per_sq_inch_first_tv (W : ℕ) := cost_first_tv / area_first_tv W
def cost_per_sq_inch_new_tv := cost_new_tv / area_new_tv

-- The proof statement
theorem width_of_first_tv_is_24 :
  cost_per_sq_inch_first_tv width_first_tv = cost_per_sq_inch_new_tv + cost_per_sq_inch_diff
  := by
    unfold cost_per_sq_inch_first_tv
    unfold area_first_tv
    unfold cost_per_sq_inch_new_tv
    unfold area_new_tv
    sorry -- proof to be filled in

end width_of_first_tv_is_24_l1328_132897


namespace redistribution_not_always_possible_l1328_132882

theorem redistribution_not_always_possible (a b : ℕ) (h : a ≠ b) :
  ¬(∃ k : ℕ, a - k = b + k ∧ 0 ≤ k ∧ k ≤ a ∧ k ≤ b) ↔ (a + b) % 2 = 1 := 
by 
  sorry

end redistribution_not_always_possible_l1328_132882


namespace actual_books_bought_l1328_132876

def initial_spending : ℕ := 180
def planned_books (x : ℕ) : Prop := initial_spending / x - initial_spending / (5 * x / 4) = 9

theorem actual_books_bought (x : ℕ) (hx : planned_books x) : (5 * x / 4) = 5 :=
by
  sorry

end actual_books_bought_l1328_132876


namespace carla_total_time_l1328_132861

def time_sharpening : ℝ := 15
def time_peeling : ℝ := 3 * time_sharpening
def time_chopping : ℝ := 0.5 * time_peeling
def time_breaks : ℝ := 2 * 5

def total_time : ℝ :=
  time_sharpening + time_peeling + time_chopping + time_breaks

theorem carla_total_time : total_time = 92.5 :=
by sorry

end carla_total_time_l1328_132861


namespace average_speed_including_stoppages_l1328_132858

/--
If the average speed of a bus excluding stoppages is 50 km/hr, and
the bus stops for 12 minutes per hour, then the average speed of the
bus including stoppages is 40 km/hr.
-/
theorem average_speed_including_stoppages
  (u : ℝ) (Δt : ℝ) (h₁ : u = 50) (h₂ : Δt = 12) : 
  (u * (60 - Δt) / 60) = 40 :=
by
  sorry

end average_speed_including_stoppages_l1328_132858


namespace johns_profit_l1328_132864

variable (numDucks : ℕ) (duckCost : ℕ) (duckWeight : ℕ) (sellPrice : ℕ)

def totalCost (numDucks duckCost : ℕ) : ℕ :=
  numDucks * duckCost

def totalWeight (numDucks duckWeight : ℕ) : ℕ :=
  numDucks * duckWeight

def totalRevenue (totalWeight sellPrice : ℕ) : ℕ :=
  totalWeight * sellPrice

def profit (totalRevenue totalCost : ℕ) : ℕ :=
  totalRevenue - totalCost

theorem johns_profit :
  totalCost 30 10 = 300 →
  totalWeight 30 4 = 120 →
  totalRevenue 120 5 = 600 →
  profit 600 300 = 300 :=
  by
    intros
    sorry

end johns_profit_l1328_132864


namespace whole_process_time_is_6_hours_l1328_132843

def folding_time_per_fold : ℕ := 5
def number_of_folds : ℕ := 4
def resting_time_per_rest : ℕ := 75
def number_of_rests : ℕ := 4
def mixing_time : ℕ := 10
def baking_time : ℕ := 30

def total_time_process_in_minutes : ℕ :=
  mixing_time + 
  (folding_time_per_fold * number_of_folds) + 
  (resting_time_per_rest * number_of_rests) + 
  baking_time

def total_time_process_in_hours : ℕ := total_time_process_in_minutes / 60

theorem whole_process_time_is_6_hours :
  total_time_process_in_hours = 6 :=
by sorry

end whole_process_time_is_6_hours_l1328_132843


namespace leo_assignment_third_part_time_l1328_132859

-- Define all the conditions as variables
def first_part_time : ℕ := 25
def first_break : ℕ := 10
def second_part_time : ℕ := 2 * first_part_time
def second_break : ℕ := 15
def total_time : ℕ := 150

-- The calculated total time of the first two parts and breaks
def time_spent_on_first_two_parts_and_breaks : ℕ :=
  first_part_time + first_break + second_part_time + second_break

-- The remaining time for the third part of the assignment
def third_part_time : ℕ :=
  total_time - time_spent_on_first_two_parts_and_breaks

-- The theorem to prove that the time Leo took to finish the third part is 50 minutes
theorem leo_assignment_third_part_time : third_part_time = 50 := by
  sorry

end leo_assignment_third_part_time_l1328_132859


namespace european_math_school_gathering_l1328_132888

theorem european_math_school_gathering :
  ∃ n : ℕ, n < 400 ∧ n % 17 = 16 ∧ n % 19 = 12 ∧ n = 288 :=
by
  sorry

end european_math_school_gathering_l1328_132888


namespace exists_composite_expression_l1328_132862

-- Define what it means for a number to be composite
def is_composite (m : ℕ) : Prop :=
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ a * b = m

-- Main theorem statement
theorem exists_composite_expression :
  ∃ n : ℕ, n > 0 ∧ ∀ k : ℕ, k > 0 → is_composite (n * 2^k + 1) :=
sorry

end exists_composite_expression_l1328_132862


namespace max_reached_at_2001_l1328_132828

noncomputable def a (n : ℕ) : ℝ := n^2 / 1.001^n

theorem max_reached_at_2001 : ∀ n : ℕ, a 2001 ≥ a n := 
sorry

end max_reached_at_2001_l1328_132828


namespace ratio_of_sums_l1328_132819

theorem ratio_of_sums (a b c d : ℚ) (h1 : b / a = 3) (h2 : d / b = 4) (h3 : c = (a + b) / 2) :
  (a + b + c) / (b + c + d) = 8 / 17 :=
by
  sorry

end ratio_of_sums_l1328_132819


namespace trains_meet_1050_km_from_delhi_l1328_132865

def distance_train_meet (t1_departure t2_departure : ℕ) (s1 s2 : ℕ) : ℕ :=
  let t_gap := t2_departure - t1_departure      -- Time difference between the departures in hours
  let d1 := s1 * t_gap                          -- Distance covered by the first train until the second train starts
  let relative_speed := s2 - s1                 -- Relative speed of the second train with respect to the first train
  d1 + s2 * (d1 / relative_speed)               -- Distance from Delhi where they meet

theorem trains_meet_1050_km_from_delhi :
  distance_train_meet 9 14 30 35 = 1050 := by
  -- Definitions based on the problem's conditions
  let t1 := 9          -- First train departs at 9 a.m.
  let t2 := 14         -- Second train departs at 2 p.m. (14:00 in 24-hour format)
  let s1 := 30         -- Speed of the first train in km/h
  let s2 := 35         -- Speed of the second train in km/h
  sorry -- proof to be filled in

end trains_meet_1050_km_from_delhi_l1328_132865


namespace addition_even_odd_is_odd_subtraction_even_odd_is_odd_squared_sum_even_odd_is_odd_l1328_132868

section OperationsAlwaysYieldOdd

def is_even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k
def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

theorem addition_even_odd_is_odd (a b : ℤ) (h₁ : is_even a) (h₂ : is_odd b) : is_odd (a + b) :=
sorry

theorem subtraction_even_odd_is_odd (a b : ℤ) (h₁ : is_even a) (h₂ : is_odd b) : is_odd (a - b) :=
sorry

theorem squared_sum_even_odd_is_odd (a b : ℤ) (h₁ : is_even a) (h₂ : is_odd b) : is_odd ((a + b) * (a + b)) :=
sorry

end OperationsAlwaysYieldOdd

end addition_even_odd_is_odd_subtraction_even_odd_is_odd_squared_sum_even_odd_is_odd_l1328_132868


namespace pieces_eaten_first_night_l1328_132879

def initial_candy_debby : ℕ := 32
def initial_candy_sister : ℕ := 42
def candy_after_first_night : ℕ := 39

theorem pieces_eaten_first_night :
  (initial_candy_debby + initial_candy_sister) - candy_after_first_night = 35 := by
  sorry

end pieces_eaten_first_night_l1328_132879


namespace value_of_neg2_neg4_l1328_132890

def operation (a b x y : ℤ) : ℤ := a * x - b * y

theorem value_of_neg2_neg4 (a b : ℤ) (h : operation a b 1 2 = 8) : operation a b (-2) (-4) = -16 := by
  sorry

end value_of_neg2_neg4_l1328_132890


namespace range_of_a_l1328_132825

theorem range_of_a (a : ℝ) : (4 - a < 0) → (a > 4) :=
by
  intros h
  sorry

end range_of_a_l1328_132825


namespace not_rented_two_bedroom_units_l1328_132893

theorem not_rented_two_bedroom_units (total_units : ℕ)
  (units_rented_ratio : ℚ)
  (total_rented_units : ℕ)
  (one_bed_room_rented_ratio two_bed_room_rented_ratio three_bed_room_rented_ratio : ℚ)
  (one_bed_room_rented_count two_bed_room_rented_count three_bed_room_rented_count : ℕ)
  (x : ℕ) 
  (total_two_bed_room_units rented_two_bed_room_units : ℕ)
  (units_ratio_condition : 2*x + 3*x + 4*x = total_rented_units)
  (total_units_condition : total_units = 1200)
  (ratio_condition : units_rented_ratio = 7/12)
  (rented_units_condition : total_rented_units = (7/12) * total_units)
  (one_bed_condition : one_bed_room_rented_ratio = 2/5)
  (two_bed_condition : two_bed_room_rented_ratio = 1/2)
  (three_bed_condition : three_bed_room_rented_ratio = 3/8)
  (one_bed_count : one_bed_room_rented_count = 2 * x)
  (two_bed_count : two_bed_room_rented_count = 3 * x)
  (three_bed_count : three_bed_room_rented_count = 4 * x)
  (x_value : x = total_rented_units / 9)
  (total_two_bed_units_calc : total_two_bed_room_units = 2 * two_bed_room_rented_count)
  : total_two_bed_room_units - two_bed_room_rented_count = 231 :=
  by
  sorry

end not_rented_two_bedroom_units_l1328_132893


namespace max_real_roots_among_polynomials_l1328_132891

noncomputable def largest_total_real_roots (a b c : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) : ℕ :=
  4  -- representing the largest total number of real roots

theorem max_real_roots_among_polynomials
  (a b c : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) :
  largest_total_real_roots a b c h_a h_b h_c = 4 :=
sorry

end max_real_roots_among_polynomials_l1328_132891


namespace log_expression_l1328_132878

theorem log_expression :
  (Real.log 2)^2 + Real.log 2 * Real.log 5 + Real.log 5 = 1 := by
  sorry

end log_expression_l1328_132878


namespace max_volume_solid_l1328_132896

-- Define volumes of individual cubes
def cube_volume (side: ℕ) : ℕ := side * side * side

-- Calculate the total number of cubes in the solid
def total_cubes (base_layer : ℕ) (second_layer : ℕ) : ℕ := base_layer + second_layer

-- Define the base layer and second layer cubes
def base_layer_cubes : ℕ := 4 * 4
def second_layer_cubes : ℕ := 2 * 2

-- Define the total volume of the solid
def total_volume (side_length : ℕ) (base_layer : ℕ) (second_layer : ℕ) : ℕ := 
  total_cubes base_layer second_layer * cube_volume side_length

theorem max_volume_solid :
  total_volume 3 base_layer_cubes second_layer_cubes = 540 := by
  sorry

end max_volume_solid_l1328_132896


namespace child_height_at_last_visit_l1328_132895

-- Definitions for the problem
def h_current : ℝ := 41.5 -- current height in inches
def Δh : ℝ := 3 -- height growth in inches

-- The proof statement
theorem child_height_at_last_visit : h_current - Δh = 38.5 := by
  sorry

end child_height_at_last_visit_l1328_132895


namespace number_of_soccer_campers_l1328_132840

-- Conditions as definitions in Lean
def total_campers : ℕ := 88
def basketball_campers : ℕ := 24
def football_campers : ℕ := 32
def soccer_campers : ℕ := total_campers - (basketball_campers + football_campers)

-- Theorem statement to prove
theorem number_of_soccer_campers : soccer_campers = 32 := by
  sorry

end number_of_soccer_campers_l1328_132840


namespace football_defeat_points_l1328_132881

theorem football_defeat_points (V D F : ℕ) (x : ℕ) :
    3 * V + D + x * F = 8 →
    27 + 6 * x = 32 →
    x = 0 :=
by
    intros h1 h2
    sorry

end football_defeat_points_l1328_132881


namespace inequality_division_l1328_132842

variable {a b c : ℝ}

theorem inequality_division (h1 : a > b) (h2 : b > 0) (h3 : c < 0) : 
  (a / (a - c)) > (b / (b - c)) := 
sorry

end inequality_division_l1328_132842


namespace parking_spots_first_level_l1328_132856

theorem parking_spots_first_level (x : ℕ) 
    (h1 : ∃ x, x + (x + 7) + (x + 13) + 14 = 46) : x = 4 :=
by
  sorry

end parking_spots_first_level_l1328_132856


namespace range_of_k_no_third_quadrant_l1328_132899

theorem range_of_k_no_third_quadrant (k : ℝ) : ¬(∃ x : ℝ, ∃ y : ℝ, x < 0 ∧ y < 0 ∧ y = k * x + 3) → k ≤ 0 := 
sorry

end range_of_k_no_third_quadrant_l1328_132899


namespace arithmetic_sequence_sum_l1328_132837

variable {a : ℕ → ℤ} 
variable {a_3 a_4 a_5 : ℤ}

-- Hypothesis: arithmetic sequence and given condition
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
∀ n, a (n+1) - a n = a 2 - a 1

theorem arithmetic_sequence_sum (h : is_arithmetic_sequence a) (h_sum : a_3 + a_4 + a_5 = 12) : 
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 28 := 
by 
  sorry

end arithmetic_sequence_sum_l1328_132837


namespace total_spending_eq_total_is_19_l1328_132854

variable (friend_spending your_spending total_spending : ℕ)

-- Conditions
def friend_spending_eq : friend_spending = 11 := by sorry
def friend_spent_more : friend_spending = your_spending + 3 := by sorry

-- Proof that total_spending is 19
theorem total_spending_eq : total_spending = friend_spending + your_spending :=
  by sorry

theorem total_is_19 : total_spending = 19 :=
  by sorry

end total_spending_eq_total_is_19_l1328_132854


namespace arithmetic_sequence_a1_geometric_sequence_sum_l1328_132833

-- Definition of the arithmetic sequence problem
theorem arithmetic_sequence_a1 (a_n s_n : ℕ) (d : ℕ) (h1 : a_n = 32) (h2 : s_n = 63) (h3 : d = 11) :
  ∃ a_1 : ℕ, a_1 = 10 :=
by
  sorry

-- Definition of the geometric sequence problem
theorem geometric_sequence_sum (a_1 q : ℕ) (h1 : a_1 = 1) (h2 : q = 2) (m : ℕ) :
  let a_m := a_1 * (q ^ (m - 1))
  let a_m_sq := a_m * a_m
  let sm'_sum := (1 - 4^m) / (1 - 4)
  sm'_sum = (4^m - 1) / 3 :=
by
  sorry

end arithmetic_sequence_a1_geometric_sequence_sum_l1328_132833


namespace find_x_range_l1328_132860

def tight_sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, 0 < n → 1/2 ≤ a (n+1) / a n ∧ a (n+1) / a n ≤ 2

theorem find_x_range
  (a : ℕ → ℝ)
  (h_tight : tight_sequence a)
  (h1 : a 1 = 1)
  (h2 : a 2 = 3 / 2)
  (h3 : ∃ x, a 3 = x)
  (h4 : a 4 = 4) :
  ∃ x, (2 : ℝ) ≤ x ∧ x ≤ (3 : ℝ) :=
sorry

end find_x_range_l1328_132860


namespace calculate_total_cups_l1328_132824

variable (butter : ℕ) (flour : ℕ) (sugar : ℕ) (total_cups : ℕ)

def ratio_condition : Prop :=
  3 * butter = 2 * sugar ∧ 3 * flour = 5 * sugar

def sugar_condition : Prop :=
  sugar = 9

def total_cups_calculation : Prop :=
  total_cups = butter + flour + sugar

theorem calculate_total_cups (h1 : ratio_condition butter flour sugar) (h2 : sugar_condition sugar) :
  total_cups_calculation butter flour sugar total_cups -> total_cups = 30 := by
  sorry

end calculate_total_cups_l1328_132824


namespace increase_percent_exceeds_l1328_132811

theorem increase_percent_exceeds (p q M : ℝ) (M_positive : 0 < M) (p_positive : 0 < p) (q_positive : 0 < q) (q_less_p : q < p) :
  (M * (1 + p / 100) * (1 + q / 100) > M) ↔ (0 < p ∧ 0 < q) :=
by
  sorry

end increase_percent_exceeds_l1328_132811


namespace problem_part1_problem_part2_problem_part3_l1328_132800

noncomputable def find_ab (a b : ℝ) : Prop :=
  (5 * a + b = 40) ∧ (30 * a + b = 140)

noncomputable def production_cost (x : ℕ) : Prop :=
  (4 * x + 20 + 7 * (100 - x) = 660)

noncomputable def transport_cost (m : ℝ) : Prop :=
  ∃ n : ℝ, 10 ≤ n ∧ n ≤ 20 ∧ (m - 2) * n + 130 = 150

theorem problem_part1 : ∃ (a b : ℝ), find_ab a b ∧ a = 4 ∧ b = 20 := 
  sorry

theorem problem_part2 : ∃ (x : ℕ), production_cost x ∧ x = 20 := 
  sorry

theorem problem_part3 : ∃ (m : ℝ), transport_cost m ∧ m = 4 := 
  sorry

end problem_part1_problem_part2_problem_part3_l1328_132800


namespace range_of_x_l1328_132883

theorem range_of_x {f : ℝ → ℝ} (h_even : ∀ x, f x = f (-x)) 
  (h_mono_dec : ∀ x y, 0 ≤ x → x ≤ y → f y ≤ f x)
  (h_f2 : f 2 = 0)
  (h_pos : ∀ x, f (x - 1) > 0) : 
  ∀ x, -1 < x ∧ x < 3 ↔ f (x - 1) > 0 :=
sorry

end range_of_x_l1328_132883


namespace isosceles_triangle_sin_vertex_angle_l1328_132874

theorem isosceles_triangle_sin_vertex_angle (A : ℝ) (hA : 0 < A ∧ A < π / 2) 
  (hSinA : Real.sin A = 5 / 13) : 
  Real.sin (2 * A) = 120 / 169 :=
by 
  -- This placeholder indicates where the proof would go
  sorry

end isosceles_triangle_sin_vertex_angle_l1328_132874


namespace identify_letter_R_l1328_132809

variable (x y : ℕ)

def date_A : ℕ := x + 2
def date_B : ℕ := x + 5
def date_E : ℕ := x

def y_plus_x := y + x
def combined_dates := date_A x + 2 * date_B x

theorem identify_letter_R (h1 : y_plus_x x y = combined_dates x) : 
  y = 2 * x + 12 ∧ ∃ (letter : String), letter = "R" := sorry

end identify_letter_R_l1328_132809


namespace xiaohua_apples_l1328_132823

theorem xiaohua_apples (x : ℕ) (h1 : ∃ n, (n = 4 * x + 20)) 
                       (h2 : (4 * x + 20 - 8 * (x - 1) > 0) ∧ (4 * x + 20 - 8 * (x - 1) < 8)) : 
                       4 * x + 20 = 44 := by
  sorry

end xiaohua_apples_l1328_132823


namespace nonnegative_integers_existence_l1328_132869

open Classical

theorem nonnegative_integers_existence (x y : ℕ) : 
  (∃ (a b c d : ℕ), x = a + 2 * b + 3 * c + 7 * d ∧ y = b + 2 * c + 5 * d) ↔ (5 * x ≥ 7 * y) :=
by
  sorry

end nonnegative_integers_existence_l1328_132869
