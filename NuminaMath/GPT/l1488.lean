import Mathlib

namespace quadratic_real_roots_iff_l1488_148865

/-- For the quadratic equation x^2 + 3x + m = 0 to have two real roots,
    the value of m must satisfy m ≤ 9/4. -/
theorem quadratic_real_roots_iff (m : ℝ) : (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 * x2 = m ∧ x1 + x2 = -3) ↔ m ≤ 9 / 4 :=
by
  sorry

end quadratic_real_roots_iff_l1488_148865


namespace math_problem_l1488_148873

theorem math_problem (a b c : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : c ≠ 0) (h₄ : a + b + c = 0) :
  (a^2 * b^2 / ((a^2 - b * c) * (b^2 - a * c)) +
  a^2 * c^2 / ((a^2 - b * c) * (c^2 - a * b)) +
  b^2 * c^2 / ((b^2 - a * c) * (c^2 - a * b))) = 1 :=
by
  sorry

end math_problem_l1488_148873


namespace husband_additional_payment_l1488_148827

theorem husband_additional_payment (total_medical_cost : ℝ) (total_salary : ℝ) 
                                  (half_medical_cost : ℝ) (deduction_from_salary : ℝ) 
                                  (remaining_salary : ℝ) (total_payment : ℝ)
                                  (each_share : ℝ) (amount_paid_by_husband : ℝ) : 
                                  
                                  total_medical_cost = 128 →
                                  total_salary = 160 →
                                  half_medical_cost = total_medical_cost / 2 →
                                  deduction_from_salary = half_medical_cost →
                                  remaining_salary = total_salary - deduction_from_salary →
                                  total_payment = remaining_salary + half_medical_cost →
                                  each_share = total_payment / 2 →
                                  amount_paid_by_husband = 64 →
                                  (each_share - amount_paid_by_husband) = 16 := by
  sorry

end husband_additional_payment_l1488_148827


namespace average_difference_l1488_148851

theorem average_difference (t : ℚ) (ht : t = 4) :
  let m := (13 + 16 + 10 + 15 + 11) / 5
  let n := (16 + t + 3 + 13) / 4
  m - n = 4 :=
by
  sorry

end average_difference_l1488_148851


namespace cost_of_tea_l1488_148836

theorem cost_of_tea (x : ℕ) (h1 : 9 * x < 1000) (h2 : 10 * x > 1100) : x = 111 :=
by
  sorry

end cost_of_tea_l1488_148836


namespace problem_l1488_148839

theorem problem (h : (0.00027 : ℝ) = 27 / 100000) : (10^5 - 10^3) * 0.00027 = 26.73 := by
  sorry

end problem_l1488_148839


namespace amount_charged_for_kids_l1488_148853

theorem amount_charged_for_kids (K A: ℝ) (H1: A = 2 * K) (H2: 8 * K + 10 * A = 84) : K = 3 :=
by
  sorry

end amount_charged_for_kids_l1488_148853


namespace find_a_b_l1488_148860

theorem find_a_b (a b : ℝ) (h : (a - 2) ^ 2 + |b + 4| = 0) : a + b = -2 :=
sorry

end find_a_b_l1488_148860


namespace area_ratio_l1488_148847

theorem area_ratio
  (a b c : ℕ)
  (h1 : 2 * (a + c) = 2 * 2 * (b + c))
  (h2 : a = 2 * b)
  (h3 : c = c) :
  (a * c) = 2 * (b * c) :=
by
  sorry

end area_ratio_l1488_148847


namespace abc_cubed_sum_l1488_148830

theorem abc_cubed_sum (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
    (h_eq : (a^3 + 12) / a = (b^3 + 12) / b ∧ (b^3 + 12) / b = (c^3 + 12) / c) : 
    a^3 + b^3 + c^3 = -36 :=
by sorry

end abc_cubed_sum_l1488_148830


namespace price_reduction_correct_l1488_148833

noncomputable def percentage_reduction (x : ℝ) : Prop :=
  (5000 * (1 - x)^2 = 4050)

theorem price_reduction_correct {x : ℝ} (h : percentage_reduction x) : x = 0.1 :=
by
  -- proof is omitted, so we use sorry
  sorry

end price_reduction_correct_l1488_148833


namespace proof_y_minus_x_l1488_148840

theorem proof_y_minus_x (x y : ℤ) (h1 : x + y = 540) (h2 : x = (4 * y) / 5) : y - x = 60 :=
sorry

end proof_y_minus_x_l1488_148840


namespace sin_cos_pi_12_eq_l1488_148869

theorem sin_cos_pi_12_eq:
  (Real.sin (Real.pi / 12)) * (Real.cos (Real.pi / 12)) = 1 / 4 :=
by
  sorry

end sin_cos_pi_12_eq_l1488_148869


namespace minimum_value_of_expr_l1488_148888

noncomputable def expr (x : ℝ) : ℝ := x + (1 / (x - 5))

theorem minimum_value_of_expr : ∀ (x : ℝ), x > 5 → expr x ≥ 7 ∧ (expr x = 7 ↔ x = 6) := 
by 
  sorry

end minimum_value_of_expr_l1488_148888


namespace bekah_days_left_l1488_148898

theorem bekah_days_left 
  (total_pages : ℕ)
  (pages_read : ℕ)
  (pages_per_day : ℕ)
  (remaining_pages : ℕ := total_pages - pages_read)
  (days_left : ℕ := remaining_pages / pages_per_day) :
  total_pages = 408 →
  pages_read = 113 →
  pages_per_day = 59 →
  days_left = 5 :=
by {
  sorry
}

end bekah_days_left_l1488_148898


namespace valerie_laptop_purchase_l1488_148813

/-- Valerie wants to buy a new laptop priced at $800. She receives $100 dollars from her parents,
$60 dollars from her uncle, and $40 dollars from her siblings for her graduation.
She also makes $20 dollars each week from tutoring. How many weeks must she save 
her tutoring income, along with her graduation money, to buy the laptop? -/
theorem valerie_laptop_purchase :
  let price_of_laptop : ℕ := 800
  let graduation_money : ℕ := 100 + 60 + 40
  let weekly_tutoring_income : ℕ := 20
  let remaining_amount_needed : ℕ := price_of_laptop - graduation_money
  let weeks_needed := remaining_amount_needed / weekly_tutoring_income
  weeks_needed = 30 :=
by
  sorry

end valerie_laptop_purchase_l1488_148813


namespace part_one_part_two_l1488_148802

-- Part 1:
-- Define the function f
def f (x : ℝ) : ℝ := abs (2 * x - 3) + abs (2 * x + 2)

-- Define the inequality problem
theorem part_one (x : ℝ) : f x < x + 5 ↔ 0 < x ∧ x < 2 :=
by sorry

-- Part 2:
-- Define the condition for part 2
theorem part_two (a : ℝ) : (∀ x : ℝ, f x > a + 4 / a) ↔ (a ∈ Set.Ioo 1 4 ∨ a < 0) :=
by sorry

end part_one_part_two_l1488_148802


namespace mia_stops_in_quarter_C_l1488_148820

def track_circumference : ℕ := 100 -- The circumference of the track in feet.
def total_distance_run : ℕ := 10560 -- The total distance Mia runs in feet.

-- Define the function to determine the quarter of the circle Mia stops in.
def quarter_mia_stops : ℕ :=
  let quarters := track_circumference / 4 -- Each quarter's length.
  let complete_laps := total_distance_run / track_circumference
  let remaining_distance := total_distance_run % track_circumference
  if remaining_distance < quarters then 1 -- Quarter A
  else if remaining_distance < 2 * quarters then 2 -- Quarter B
  else if remaining_distance < 3 * quarters then 3 -- Quarter C
  else 4 -- Quarter D

theorem mia_stops_in_quarter_C : quarter_mia_stops = 3 := by
  sorry

end mia_stops_in_quarter_C_l1488_148820


namespace solution_set_for_composed_function_l1488_148887

theorem solution_set_for_composed_function :
  ∀ x : ℝ, (∀ y : ℝ, y = 2 * x - 1 → (2 * y - 1) ≥ 1) ↔ x ≥ 1 := by
  sorry

end solution_set_for_composed_function_l1488_148887


namespace simplify_fraction_l1488_148895

theorem simplify_fraction (a b m : ℝ) (h1 : (a / b) ^ m = (a^m) / (b^m)) (h2 : (-1 : ℝ) ^ (0 : ℝ) = 1) :
  ( (81 / 16) ^ (3 / 4) ) - 1 = 19 / 8 :=
by
  sorry

end simplify_fraction_l1488_148895


namespace fraction_identity_l1488_148805

variable (a b : ℚ) (h : a / b = 2 / 3)

theorem fraction_identity : a / (a - b) = -2 :=
by
  sorry

end fraction_identity_l1488_148805


namespace fraction_of_largest_jar_filled_l1488_148808

theorem fraction_of_largest_jar_filled
  (C1 C2 C3 : ℝ)
  (h1 : C1 < C2)
  (h2 : C2 < C3)
  (h3 : C1 / 6 = C2 / 5)
  (h4 : C2 / 5 = C3 / 7) :
  (C1 / 6 + C2 / 5) / C3 = 2 / 7 := sorry

end fraction_of_largest_jar_filled_l1488_148808


namespace winner_percentage_l1488_148816

theorem winner_percentage (total_votes winner_votes : ℕ) (h1 : winner_votes = 744) (h2 : total_votes - winner_votes = 288) :
  (winner_votes : ℤ) * 100 / total_votes = 62 := 
by
  sorry

end winner_percentage_l1488_148816


namespace centroid_distance_l1488_148815

theorem centroid_distance
  (a b m : ℝ)
  (h_a_nonneg : 0 ≤ a)
  (h_b_nonneg : 0 ≤ b)
  (h_m_pos : 0 < m) :
  (∃ d : ℝ, d = m * (b + 2 * a) / (3 * (a + b))) :=
by
  sorry

end centroid_distance_l1488_148815


namespace find_m_if_even_l1488_148801

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def my_function (m : ℝ) (x : ℝ) : ℝ :=
  (m - 1) * x^2 + (m - 2) * x + (m^2 - 7 * m + 12)

theorem find_m_if_even (m : ℝ) :
  is_even_function (my_function m) → m = 2 := 
by
  sorry

end find_m_if_even_l1488_148801


namespace jar_marbles_difference_l1488_148823

theorem jar_marbles_difference (a b : ℕ) (h1 : 9 * a = 9 * b) (h2 : 2 * a + b = 135) : 8 * b - 7 * a = 45 := by
  sorry

end jar_marbles_difference_l1488_148823


namespace base_five_of_156_is_1111_l1488_148872

def base_five_equivalent (n : ℕ) : ℕ := sorry

theorem base_five_of_156_is_1111 :
  base_five_equivalent 156 = 1111 :=
sorry

end base_five_of_156_is_1111_l1488_148872


namespace average_weight_of_all_children_l1488_148834

theorem average_weight_of_all_children 
  (Boys: ℕ) (Girls: ℕ) (Additional: ℕ)
  (avgWeightBoys: ℚ) (avgWeightGirls: ℚ) (avgWeightAdditional: ℚ) :
  Boys = 8 ∧ Girls = 5 ∧ Additional = 3 ∧ 
  avgWeightBoys = 160 ∧ avgWeightGirls = 130 ∧ avgWeightAdditional = 145 →
  ((Boys * avgWeightBoys + Girls * avgWeightGirls + Additional * avgWeightAdditional) / (Boys + Girls + Additional) = 148) :=
by
  intros
  sorry

end average_weight_of_all_children_l1488_148834


namespace relationship_among_a_b_c_l1488_148863

noncomputable def a := Real.log 2 / 2
noncomputable def b := Real.log 3 / 3
noncomputable def c := Real.log 5 / 5

theorem relationship_among_a_b_c : c < a ∧ a < b := by
  sorry

end relationship_among_a_b_c_l1488_148863


namespace necessary_but_not_sufficient_condition_l1488_148854

variables {Point Line Plane : Type} 

-- Definitions for the problem conditions
def is_subset_of (a : Line) (α : Plane) : Prop := sorry
def parallel_plane (a : Line) (β : Plane) : Prop := sorry
def parallel_lines (a b : Line) : Prop := sorry
def parallel_planes (α β : Plane) : Prop := sorry

-- The statement of the problem
theorem necessary_but_not_sufficient_condition (a b : Line) (α β : Plane) 
  (h1 : is_subset_of a α) (h2 : is_subset_of b β) :
  (parallel_plane a β ∧ parallel_plane b α) ↔ 
  (¬ parallel_planes α β ∧ sorry) :=
sorry

end necessary_but_not_sufficient_condition_l1488_148854


namespace fraction_calculation_l1488_148879

theorem fraction_calculation : (36 - 12) / (12 - 4) = 3 :=
by
  sorry

end fraction_calculation_l1488_148879


namespace exists_three_distinct_integers_in_A_l1488_148893

noncomputable def A (m n : ℤ) : Set ℤ := { x^2 + m * x + n | x : ℤ }

theorem exists_three_distinct_integers_in_A (m n : ℤ) :
  ∃ a b c : ℤ, a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ a ∈ A m n ∧ b ∈ A m n ∧ c ∈ A m n ∧ a = b * c :=
by
  sorry

end exists_three_distinct_integers_in_A_l1488_148893


namespace find_integer_x_l1488_148855

theorem find_integer_x (x : ℤ) :
  (2 * x > 70 → x > 35 ∧ 4 * x > 25 → x > 6) ∧
  (x < 100 → x = 6) ∧ -- Given valid inequality relations and restrictions
  ((2 * x <= 70 ∨ x >= 100) ∨ (4 * x <= 25 ∨ x <= 5)) := -- Certain contradictions inherent

  by sorry

end find_integer_x_l1488_148855


namespace height_to_top_floor_l1488_148875

def total_height : ℕ := 1454
def antenna_spire_height : ℕ := 204

theorem height_to_top_floor : (total_height - antenna_spire_height) = 1250 := by
  sorry

end height_to_top_floor_l1488_148875


namespace cylindrical_to_rectangular_l1488_148882

theorem cylindrical_to_rectangular (r θ z : ℝ) (hr : r = 10) (hθ : θ = 3 * Real.pi / 4) (hz : z = 2) :
    ∃ (x y z' : ℝ), (x = r * Real.cos θ) ∧ (y = r * Real.sin θ) ∧ (z' = z) ∧ (x = -5 * Real.sqrt 2) ∧ (y = 5 * Real.sqrt 2) ∧ (z' = 2) :=
by
  sorry

end cylindrical_to_rectangular_l1488_148882


namespace proof_main_proof_l1488_148810

noncomputable def main_proof : Prop :=
  2 * Real.logb 5 10 + Real.logb 5 0.25 = 2

theorem proof_main_proof : main_proof :=
  by
    sorry

end proof_main_proof_l1488_148810


namespace smallest_inverse_defined_l1488_148881

theorem smallest_inverse_defined (n : ℤ) : n = 5 :=
by sorry

end smallest_inverse_defined_l1488_148881


namespace rectangle_area_from_square_l1488_148864

theorem rectangle_area_from_square 
  (square_area : ℕ) 
  (width_rect : ℕ) 
  (length_rect : ℕ) 
  (h_square_area : square_area = 36)
  (h_width_rect : width_rect * width_rect = square_area)
  (h_length_rect : length_rect = 3 * width_rect) :
  width_rect * length_rect = 108 :=
by
  sorry

end rectangle_area_from_square_l1488_148864


namespace total_length_infinite_sum_l1488_148841

-- Define the infinite sums
noncomputable def S1 : ℝ := ∑' n : ℕ, (1 / (3^n))
noncomputable def S2 : ℝ := (∑' n : ℕ, (1 / (5^n))) * Real.sqrt 3
noncomputable def S3 : ℝ := (∑' n : ℕ, (1 / (7^n))) * Real.sqrt 5

-- Define the total length
noncomputable def total_length : ℝ := S1 + S2 + S3

-- The statement of the theorem
theorem total_length_infinite_sum : total_length = (3 / 2) + (Real.sqrt 3 / 4) + (Real.sqrt 5 / 6) :=
by
  sorry

end total_length_infinite_sum_l1488_148841


namespace cost_price_is_1500_l1488_148831

-- Definitions for the given conditions
def selling_price : ℝ := 1200
def loss_percentage : ℝ := 20

-- Define the cost price such that the loss percentage condition is satisfied
def cost_price (C : ℝ) : Prop :=
  loss_percentage = ((C - selling_price) / C) * 100

-- The proof problem to be solved: 
-- Prove that the cost price of the radio is Rs. 1500
theorem cost_price_is_1500 : ∃ C, cost_price C ∧ C = 1500 :=
by
  sorry

end cost_price_is_1500_l1488_148831


namespace range_of_a_l1488_148883

noncomputable section

open Real

def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ∈ Set.Icc 0 3 then a - x else a * log x / log 2

theorem range_of_a (a : ℝ) (h : f a 2 < f a 4) : a > -2 := by
  sorry

end range_of_a_l1488_148883


namespace ellipse_x_intercept_other_l1488_148809

noncomputable def foci : (ℝ × ℝ) × (ℝ × ℝ) := ((0, 3), (4, 0))
noncomputable def sum_of_distances : ℝ := 7
noncomputable def first_intercept : (ℝ × ℝ) := (0, 0)

theorem ellipse_x_intercept_other 
  (foci : (ℝ × ℝ) × (ℝ × ℝ))
  (sum_of_distances : ℝ)
  (first_intercept : (ℝ × ℝ))
  (hx : foci = ((0, 3), (4, 0)))
  (d_sum : sum_of_distances = 7)
  (intercept : first_intercept = (0, 0)) :
  ∃ (x : ℝ), x > 0 ∧ ((x, 0) = (56 / 11, 0)) := 
sorry

end ellipse_x_intercept_other_l1488_148809


namespace not_and_implication_l1488_148892

variable (p q : Prop)

theorem not_and_implication : ¬ (p ∧ q) → (¬ p ∨ ¬ q) :=
by
  sorry

end not_and_implication_l1488_148892


namespace ant_to_vertices_probability_l1488_148884

noncomputable def event_A_probability : ℝ :=
  1 - (Real.sqrt 3 * Real.pi / 24)

theorem ant_to_vertices_probability :
  let side_length := 4
  let event_A := "the distance from the ant to all three vertices is more than 1"
  event_A_probability = 1 - Real.sqrt 3 * Real.pi / 24
:=
sorry

end ant_to_vertices_probability_l1488_148884


namespace find_LP_l1488_148838

variables (A B C K L P M : Type) 
variables {AC BC AK CK CL AM LP : ℕ}

-- Defining the given conditions
def conditions (AC BC AK CK : ℕ) (AM : ℕ) :=
  AC = 360 ∧ BC = 240 ∧ AK = CK ∧ AK = 180 ∧ AM = 144

-- The theorem statement: proving LP equals 57.6
theorem find_LP (h : conditions 360 240 180 180 144) : LP = 576 / 10 := 
by sorry

end find_LP_l1488_148838


namespace percent_value_in_quarters_l1488_148818

theorem percent_value_in_quarters (dimes quarters : ℕ) (dime_value quarter_value : ℕ) (dime_count quarter_count : ℕ) :
  dimes = 50 →
  quarters = 20 →
  dime_value = 10 →
  quarter_value = 25 →
  dime_count = dimes * dime_value →
  quarter_count = quarters * quarter_value →
  (quarter_count : ℚ) / (dime_count + quarter_count) * 100 = 50 :=
by
  intros
  sorry

end percent_value_in_quarters_l1488_148818


namespace problem_statement_l1488_148837

noncomputable def C_points_count (A B : (ℝ × ℝ)) : ℕ :=
  if A = (0, 0) ∧ B = (12, 0) then 4 else 0

theorem problem_statement :
  let A := (0, 0)
  let B := (12, 0)
  C_points_count A B = 4 :=
by
  sorry

end problem_statement_l1488_148837


namespace car_speed_is_90_mph_l1488_148812

-- Define the given conditions
def distance_yards : ℚ := 22
def time_seconds : ℚ := 0.5
def yards_per_mile : ℚ := 1760

-- Define the car's speed in miles per hour
noncomputable def car_speed_mph : ℚ := (distance_yards / yards_per_mile) * (3600 / time_seconds)

-- The theorem to be proven
theorem car_speed_is_90_mph : car_speed_mph = 90 := by
  sorry

end car_speed_is_90_mph_l1488_148812


namespace regular_hexagon_interior_angles_l1488_148804

theorem regular_hexagon_interior_angles (n : ℕ) (h : n = 6) :
  (n - 2) * 180 = 720 :=
by
  subst h
  rfl

end regular_hexagon_interior_angles_l1488_148804


namespace original_price_of_cycle_l1488_148868

theorem original_price_of_cycle (selling_price : ℝ) (loss_percentage : ℝ) (original_price : ℝ) 
  (h1 : selling_price = 1610)
  (h2 : loss_percentage = 30) 
  (h3 : selling_price = original_price * (1 - loss_percentage / 100)) : 
  original_price = 2300 := 
by 
  sorry

end original_price_of_cycle_l1488_148868


namespace systematic_sampling_first_group_l1488_148896

theorem systematic_sampling_first_group
  (a : ℕ → ℕ)
  (d : ℕ)
  (n : ℕ)
  (a₁ : ℕ)
  (a₁₆ : ℕ)
  (h₁ : d = 8)
  (h₂ : a 16 = a₁₆)
  (h₃ : a₁₆ = 125)
  (h₄ : a n = a₁ + (n - 1) * d) :
  a 1 = 5 :=
by
  sorry

end systematic_sampling_first_group_l1488_148896


namespace range_of_ab_l1488_148889

noncomputable def f (x : ℝ) : ℝ := abs (2 - x^2)

theorem range_of_ab (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : f a = f b) : 0 < a * b ∧ a * b < 2 :=
by
  sorry

end range_of_ab_l1488_148889


namespace equal_cylinder_volumes_l1488_148880

theorem equal_cylinder_volumes (x : ℝ) (hx : x > 0) :
  π * (5 + x) ^ 2 * 4 = π * 25 * (4 + x) → x = 35 / 4 :=
by
  sorry

end equal_cylinder_volumes_l1488_148880


namespace arithmetic_sequence_geometric_sum_l1488_148800

theorem arithmetic_sequence_geometric_sum (a1 : ℝ) (S : ℕ → ℝ)
  (h1 : ∀ (n : ℕ), S 1 = a1)
  (h2 : ∀ (n : ℕ), S 2 = 2 * a1 - 1)
  (h3 : ∀ (n : ℕ), S 4 = 4 * a1 - 6)
  (h4 : (2 * a1 - 1)^2 = a1 * (4 * a1 - 6)) 
  : a1 = -1/2 := 
sorry

end arithmetic_sequence_geometric_sum_l1488_148800


namespace henry_has_30_more_lollipops_than_alison_l1488_148871

noncomputable def num_lollipops_alison : ℕ := 60
noncomputable def num_lollipops_diane : ℕ := 2 * num_lollipops_alison
noncomputable def total_num_days : ℕ := 6
noncomputable def num_lollipops_per_day : ℕ := 45
noncomputable def total_lollipops : ℕ := total_num_days * num_lollipops_per_day
noncomputable def num_lollipops_total_ad : ℕ := num_lollipops_alison + num_lollipops_diane
noncomputable def num_lollipops_henry : ℕ := total_lollipops - num_lollipops_total_ad
noncomputable def lollipops_diff_henry_alison : ℕ := num_lollipops_henry - num_lollipops_alison

theorem henry_has_30_more_lollipops_than_alison :
  lollipops_diff_henry_alison = 30 :=
by
  unfold lollipops_diff_henry_alison
  unfold num_lollipops_henry
  unfold num_lollipops_total_ad
  unfold total_lollipops
  sorry

end henry_has_30_more_lollipops_than_alison_l1488_148871


namespace hanks_pancakes_needed_l1488_148811

/-- Hank's pancake calculation problem -/
theorem hanks_pancakes_needed 
    (pancakes_per_big_stack : ℕ := 5)
    (pancakes_per_short_stack : ℕ := 3)
    (big_stack_orders : ℕ := 6)
    (short_stack_orders : ℕ := 9) :
    (pancakes_per_short_stack * short_stack_orders) + (pancakes_per_big_stack * big_stack_orders) = 57 := by {
  sorry
}

end hanks_pancakes_needed_l1488_148811


namespace danielles_rooms_l1488_148829

variable (rooms_heidi rooms_danielle : ℕ)

theorem danielles_rooms 
  (h1 : rooms_heidi = 3 * rooms_danielle)
  (h2 : 2 = 1 / 9 * rooms_heidi) :
  rooms_danielle = 6 := by
  -- Proof omitted
  sorry

end danielles_rooms_l1488_148829


namespace maximal_value_of_product_l1488_148877

theorem maximal_value_of_product (m n : ℤ)
  (h1 : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (1 < x1 ∧ x1 < 3) ∧ (1 < x2 ∧ x2 < 3) ∧ 
    ∀ x : ℝ, (10 * x^2 + m * x + n) = 10 * (x - x1) * (x - x2)) :
  (∃ f1 f3 : ℝ, f1 = 10 * (1 - x1) * (1 - x2) ∧ f3 = 10 * (3 - x1) * (3 - x2) ∧ (f1 * f3 = 99)) := 
sorry

end maximal_value_of_product_l1488_148877


namespace solve_for_y_l1488_148885

noncomputable def log5 (x : ℝ) : ℝ := (Real.log x) / (Real.log 5)

theorem solve_for_y (y : ℝ) (h₀ : log5 ((2 * y + 10) / (3 * y - 6)) + log5 ((3 * y - 6) / (y - 4)) = 3) : 
  y = 170 / 41 :=
sorry

end solve_for_y_l1488_148885


namespace remainder_is_zero_l1488_148859

def f (x : ℝ) : ℝ := x^3 - 5 * x^2 + 2 * x + 8

theorem remainder_is_zero : f 2 = 0 := by
  sorry

end remainder_is_zero_l1488_148859


namespace incorrect_statement_D_l1488_148828

-- Definitions based on conditions
def length_of_spring (x : ℝ) : ℝ := 8 + 0.5 * x

-- Incorrect Statement (to be proved as incorrect)
def statement_D_incorrect : Prop :=
  ¬ (length_of_spring 30 = 23)

-- Main theorem statement
theorem incorrect_statement_D : statement_D_incorrect :=
by
  sorry

end incorrect_statement_D_l1488_148828


namespace find_z_l1488_148843

/-- x and y are positive integers. When x is divided by 9, the remainder is 2, 
and when x is divided by 7, the remainder is 4. When y is divided by 13, 
the remainder is 12. The least possible value of y - x is 14. 
Prove that the number that y is divided by to get a remainder of 3 is 22. -/
theorem find_z (x y z : ℕ) (hx9 : x % 9 = 2) (hx7 : x % 7 = 4) (hy13 : y % 13 = 12) (hyx : y = x + 14) 
: y % z = 3 → z = 22 := 
by 
  sorry

end find_z_l1488_148843


namespace arithmetic_seq_common_diff_l1488_148876

theorem arithmetic_seq_common_diff (a b : ℕ) (d : ℕ) (a1 a2 a8 a9 : ℕ) 
  (h1 : a1 + a8 = 10)
  (h2 : a2 + a9 = 18)
  (h3 : a2 = a1 + d)
  (h4 : a8 = a1 + 7 * d)
  (h5 : a9 = a1 + 8 * d)
  : d = 4 :=
by
  sorry

end arithmetic_seq_common_diff_l1488_148876


namespace equal_utilities_l1488_148845

-- Conditions
def utility (juggling coding : ℕ) : ℕ := juggling * coding

def wednesday_utility (s : ℕ) : ℕ := utility s (12 - s)
def thursday_utility (s : ℕ) : ℕ := utility (6 - s) (s + 4)

-- Theorem
theorem equal_utilities (s : ℕ) (h : wednesday_utility s = thursday_utility s) : s = 12 / 5 := 
by sorry

end equal_utilities_l1488_148845


namespace replaced_person_is_65_l1488_148891

-- Define the conditions of the problem context
variable (W : ℝ)
variable (avg_increase : ℝ := 3.5)
variable (num_persons : ℕ := 8)
variable (new_person_weight : ℝ := 93)

-- Express the given condition in the problem: 
-- The total increase in weight is given by the number of persons multiplied by the average increase in weight
def total_increase : ℝ := num_persons * avg_increase

-- Express the relationship between the new person's weight and the person who was replaced
def replaced_person_weight (W : ℝ) : ℝ := new_person_weight - total_increase

-- Stating the theorem to be proved
theorem replaced_person_is_65 : replaced_person_weight W = 65 := by
  sorry

end replaced_person_is_65_l1488_148891


namespace max_subjects_per_teacher_l1488_148899

theorem max_subjects_per_teacher (math_teachers physics_teachers chemistry_teachers min_teachers : ℕ)
  (h_math : math_teachers = 4)
  (h_physics : physics_teachers = 3)
  (h_chemistry : chemistry_teachers = 3)
  (h_min_teachers : min_teachers = 5) :
  (math_teachers + physics_teachers + chemistry_teachers) / min_teachers = 2 :=
by
  sorry

end max_subjects_per_teacher_l1488_148899


namespace find_second_largest_element_l1488_148846

open List

theorem find_second_largest_element 
(a1 a2 a3 a4 a5 : ℕ) 
(h_pos : 0 < a1 ∧ 0 < a2 ∧ 0 < a3 ∧ 0 < a4 ∧ 0 < a5) 
(h_sorted : a1 ≤ a2 ∧ a2 ≤ a3 ∧ a3 ≤ a4 ∧ a4 ≤ a5) 
(h_mean : (a1 + a2 + a3 + a4 + a5) / 5 = 15) 
(h_range : a5 - a1 = 24) 
(h_mode : a2 = 10 ∧ a3 = 10) 
(h_median : a3 = 10) 
(h_three_diff : (a1 ≠ a2 ∨ a1 ≠ a3 ∨ a1 ≠ a4 ∨ a1 ≠ a5) ∧ (a4 ≠ a5)) :
a4 = 11 :=
sorry

end find_second_largest_element_l1488_148846


namespace no_square_pair_l1488_148806

/-- 
Given integers a, b, and c, where c > 0, if a(a + 4) = c^2 and (a + 2 + c)(a + 2 - c) = 4, 
then the numbers a(a + 4) and b(b + 4) cannot both be squares.
-/
theorem no_square_pair (a b c : ℤ) (hc_pos : c > 0) (ha_eq : a * (a + 4) = c^2) 
  (hfac_eq : (a + 2 + c) * (a + 2 - c) = 4) : ¬(∃ d e : ℤ, d^2 = a * (a + 4) ∧ e^2 = b * (b + 4)) :=
by sorry

end no_square_pair_l1488_148806


namespace certain_number_l1488_148894

theorem certain_number (x : ℝ) (h : 0.65 * 40 = (4/5) * x + 6) : x = 25 :=
sorry

end certain_number_l1488_148894


namespace find_rate_per_kg_grapes_l1488_148866

-- Define the main conditions
def rate_per_kg_mango := 55
def total_payment := 985
def kg_grapes := 7
def kg_mangoes := 9

-- Define the problem statement
theorem find_rate_per_kg_grapes (G : ℝ) : 
  (kg_grapes * G + kg_mangoes * rate_per_kg_mango = total_payment) → 
  G = 70 :=
by
  sorry

end find_rate_per_kg_grapes_l1488_148866


namespace maximize_profit_l1488_148814

-- Define the relationships and constants
def P (x : ℝ) : ℝ := -750 * x + 15000
def material_cost_per_unit : ℝ := 4
def fixed_cost : ℝ := 7000

-- Define the profit function
def profit (x : ℝ) : ℝ := (x - material_cost_per_unit) * P x - fixed_cost

-- The statement of the problem, proving the maximization condition
theorem maximize_profit :
  ∃ x : ℝ, x = 12 ∧ profit 12 = 41000 := by
  sorry

end maximize_profit_l1488_148814


namespace transform_polynomial_to_y_l1488_148817

theorem transform_polynomial_to_y (x y : ℝ) (h : y = x + 1/x) :
  (x^6 + x^5 - 5*x^4 + x^3 + x + 1 = 0) → 
  (∃ (y_expr : ℝ), (x * y_expr = 0 ∨ (x = 0 ∧ y_expr = y_expr))) :=
sorry

end transform_polynomial_to_y_l1488_148817


namespace relationship_between_a_and_b_l1488_148835

open Real

theorem relationship_between_a_and_b
   (a b : ℝ)
   (ha : 0 < a ∧ a < 1)
   (hb : 0 < b ∧ b < 1)
   (hab : (1 - a) * b > 1 / 4) :
   a < b := 
sorry

end relationship_between_a_and_b_l1488_148835


namespace even_quadratic_iff_b_zero_l1488_148842

-- Define a quadratic function
def quadratic (a b c x : ℝ) : ℝ := a * x ^ 2 + b * x + c

-- State the theorem
theorem even_quadratic_iff_b_zero (a b c : ℝ) : 
  (∀ x : ℝ, quadratic a b c x = quadratic a b c (-x)) ↔ b = 0 := 
by
  sorry

end even_quadratic_iff_b_zero_l1488_148842


namespace fourth_term_of_sequence_l1488_148821

theorem fourth_term_of_sequence (x : ℤ) (h : x^2 - 2 * x - 3 < 0) (hx : x ∈ {n : ℤ | x^2 - 2 * x - 3 < 0}) :
  ∃ a_1 a_2 a_3 a_4 : ℤ, 
  (a_1 = x) ∧ (a_2 = x + 1) ∧ (a_3 = x + 2) ∧ (a_4 = x + 3) ∧ 
  (a_4 = 3 ∨ a_4 = -1) :=
by { sorry }

end fourth_term_of_sequence_l1488_148821


namespace negation_of_exists_l1488_148849

open Set Real

theorem negation_of_exists (x : Real) :
  ¬ (∃ x ∈ Icc 0 1, x^3 + x^2 > 1) ↔ ∀ x ∈ Icc 0 1, x^3 + x^2 ≤ 1 := 
by sorry

end negation_of_exists_l1488_148849


namespace max_elements_X_l1488_148826

structure GameState where
  fire : Nat
  stone : Nat
  metal : Nat

def canCreateX (state : GameState) (x : Nat) : Bool :=
  state.metal >= x ∧ state.fire >= 2 * x ∧ state.stone >= 3 * x

def maxCreateX (state : GameState) : Nat :=
  if h : canCreateX state 14 then 14 else 0 -- we would need to show how to actually maximizing the value

theorem max_elements_X : maxCreateX ⟨50, 50, 0⟩ = 14 := 
by 
  -- Proof would go here, showing via the conditions given above
  -- We would need to show no more than 14 can be created given the initial resources
  sorry

end max_elements_X_l1488_148826


namespace smallest_square_area_l1488_148867

theorem smallest_square_area (n : ℕ) (h : ∃ m : ℕ, 14 * n = m ^ 2) : n = 14 :=
sorry

end smallest_square_area_l1488_148867


namespace geometric_sequence_product_l1488_148844

theorem geometric_sequence_product (a₁ aₙ : ℝ) (n : ℕ) (hn : n > 0) (number_of_terms : n ≥ 1) :
  -- Conditions: First term, last term, number of terms
  ∃ P : ℝ, P = (a₁ * aₙ) ^ (n / 2) :=
sorry

end geometric_sequence_product_l1488_148844


namespace inverse_proposition_false_l1488_148874

-- Define the original proposition
def original_proposition (a b : ℝ) : Prop :=
  a = b → abs a = abs b

-- Define the inverse proposition
def inverse_proposition (a b : ℝ) : Prop :=
  abs a = abs b → a = b

-- The theorem to prove
theorem inverse_proposition_false : ∃ (a b : ℝ), abs a = abs b ∧ a ≠ b :=
sorry

end inverse_proposition_false_l1488_148874


namespace toy_poodle_height_l1488_148819

theorem toy_poodle_height 
  (SP MP TP : ℕ)
  (h1 : SP = MP + 8)
  (h2 : MP = TP + 6)
  (h3 : SP = 28) 
  : TP = 14 := 
    by sorry

end toy_poodle_height_l1488_148819


namespace two_person_subcommittees_from_eight_l1488_148878

theorem two_person_subcommittees_from_eight :
  (Nat.choose 8 2) = 28 :=
by
  sorry

end two_person_subcommittees_from_eight_l1488_148878


namespace angle_AC_B₁C₁_is_60_l1488_148886

-- Redefine the conditions of the problem using Lean definitions
-- We define a regular triangular prism, equilateral triangle condition,
-- and parallel lines relation.

structure TriangularPrism :=
  (A B C A₁ B₁ C₁ : Type)
  (is_regular : Prop) -- Property stating it is a regular triangular prism
  (base_is_equilateral : Prop) -- Property stating the base is an equilateral triangle
  (B₁C₁_parallel_to_BC : Prop) -- Property stating B₁C₁ is parallel to BC

-- Assume a regular triangular prism with the given properties
variable (prism : TriangularPrism)
axiom isRegularPrism : prism.is_regular
axiom baseEquilateral : prism.base_is_equilateral
axiom parallelLines : prism.B₁C₁_parallel_to_BC

-- Define the angle calculation statement in Lean 4
theorem angle_AC_B₁C₁_is_60 :
  ∃ (angle : ℝ), angle = 60 :=
by
  -- Proof is omitted using sorry
  exact ⟨60, sorry⟩

end angle_AC_B₁C₁_is_60_l1488_148886


namespace complement_union_in_universe_l1488_148861

variable (U : Set ℕ := {1, 2, 3, 4, 5})
variable (M : Set ℕ := {1, 3})
variable (N : Set ℕ := {1, 2})

theorem complement_union_in_universe :
  (U \ (M ∪ N)) = {4, 5} :=
by
  sorry

end complement_union_in_universe_l1488_148861


namespace impossibility_exchange_l1488_148856

theorem impossibility_exchange :
  ¬ ∃ (x y z : ℕ), (x + y + z = 10) ∧ (x + 3 * y + 5 * z = 25) := 
by
  sorry

end impossibility_exchange_l1488_148856


namespace difference_largest_smallest_l1488_148807

noncomputable def ratio_2_3_5 := 2 / 3
noncomputable def ratio_3_5 := 3 / 5
noncomputable def int_sum := 90

theorem difference_largest_smallest :
  ∃ (a b c : ℝ), 
    a + b + c = int_sum ∧
    b / a = ratio_2_3_5 ∧
    c / a = 5 / 2 ∧
    b / a = 3 / 2 ∧
    c - a = 12.846 := 
by
  sorry

end difference_largest_smallest_l1488_148807


namespace count_two_digit_integers_with_perfect_square_sum_l1488_148858

def valid_pairs : List (ℕ × ℕ) :=
[(2, 9), (3, 8), (4, 7), (5, 6), (6, 5), (7, 4), (8, 3), (9, 2)]

def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

def reversed_sum_is_perfect_square (n : ℕ) : Prop :=
  ∃ t u, n = 10 * t + u ∧ t + u = 11

theorem count_two_digit_integers_with_perfect_square_sum :
  Nat.card { n : ℕ // is_two_digit n ∧ reversed_sum_is_perfect_square n } = 8 := 
sorry

end count_two_digit_integers_with_perfect_square_sum_l1488_148858


namespace problem_statement_l1488_148862

theorem problem_statement : (2^2015 + 2^2011) / (2^2015 - 2^2011) = 17 / 15 :=
by
  -- Prove the equivalence as outlined above.
  sorry

end problem_statement_l1488_148862


namespace smallest_positive_number_div_conditions_is_perfect_square_l1488_148832

theorem smallest_positive_number_div_conditions_is_perfect_square :
  ∃ n : ℕ,
    (n % 11 = 10) ∧
    (n % 10 = 9) ∧
    (n % 9 = 8) ∧
    (n % 8 = 7) ∧
    (n % 7 = 6) ∧
    (n % 6 = 5) ∧
    (n % 5 = 4) ∧
    (n % 4 = 3) ∧
    (n % 3 = 2) ∧
    (n % 2 = 1) ∧
    (∃ k : ℕ, n = k * k) ∧
    n = 2782559 :=
by
  sorry

end smallest_positive_number_div_conditions_is_perfect_square_l1488_148832


namespace nesbitt_inequality_l1488_148848

variable (a b c : ℝ)
variable (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

theorem nesbitt_inequality (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  a / (b + c) + b / (c + a) + c / (a + b) ≥ 3 / 2 := 
by
  sorry

end nesbitt_inequality_l1488_148848


namespace num_of_valid_three_digit_numbers_l1488_148890

def valid_three_digit_numbers : ℕ :=
  let valid_numbers : List (ℕ × ℕ × ℕ) :=
    [(2, 3, 4), (4, 6, 8)]
  valid_numbers.length

theorem num_of_valid_three_digit_numbers :
  valid_three_digit_numbers = 2 :=
by
  sorry

end num_of_valid_three_digit_numbers_l1488_148890


namespace tan_sum_formula_l1488_148824

theorem tan_sum_formula (α β : ℝ) (h1 : Real.tan α = 2) (h2 : Real.tan β = 3) : 
  Real.tan (α + β) = -1 := by
sorry

end tan_sum_formula_l1488_148824


namespace range_of_2a_plus_b_l1488_148852

variable {a b c A B C : Real}
variable {sin cos : Real → Real}

theorem range_of_2a_plus_b (h1 : a^2 + b^2 + ab = 4) (h2 : c = 2) (h3 : a = c * sin A / sin C) (h4 : b = c * sin B / sin C) :
  2 < 2 * a + b ∧ 2 * a + b < 4 :=
by
  sorry

end range_of_2a_plus_b_l1488_148852


namespace remove_candies_even_distribution_l1488_148825

theorem remove_candies_even_distribution (candies friends : ℕ) (h_candies : candies = 30) (h_friends : friends = 4) :
  ∃ k, candies - k % friends = 0 ∧ k = 2 :=
by
  sorry

end remove_candies_even_distribution_l1488_148825


namespace lea_total_cost_example_l1488_148870

/-- Léa bought one book for $16, three binders for $2 each, and six notebooks for $1 each. -/
def total_cost (book_cost binders_cost notebooks_cost : ℕ) : ℕ :=
  book_cost + binders_cost + notebooks_cost

/-- Given the individual costs, prove the total cost of Léa's purchases is $28. -/
theorem lea_total_cost_example : total_cost 16 (3 * 2) (6 * 1) = 28 := by
  sorry

end lea_total_cost_example_l1488_148870


namespace molecular_weight_of_BaBr2_l1488_148857

theorem molecular_weight_of_BaBr2 
    (atomic_weight_Ba : ℝ)
    (atomic_weight_Br : ℝ)
    (moles : ℝ)
    (hBa : atomic_weight_Ba = 137.33)
    (hBr : atomic_weight_Br = 79.90) 
    (hmol : moles = 8) :
    (atomic_weight_Ba + 2 * atomic_weight_Br) * moles = 2377.04 :=
by 
  sorry

end molecular_weight_of_BaBr2_l1488_148857


namespace ratio_of_circle_areas_l1488_148850

variable (S L A : ℝ)

theorem ratio_of_circle_areas 
  (h1 : A = (3 / 5) * S)
  (h2 : A = (6 / 25) * L)
  : S / L = 2 / 5 :=
by
  sorry

end ratio_of_circle_areas_l1488_148850


namespace gage_skating_time_l1488_148803

theorem gage_skating_time :
  let gage_times_in_minutes1 := 1 * 60 + 15 -- 1 hour 15 minutes converted to minutes
  let gage_times_in_minutes2 := 2 * 60      -- 2 hours converted to minutes
  let total_skating_time_8_days := 5 * gage_times_in_minutes1 + 3 * gage_times_in_minutes2
  let required_total_time := 10 * 95       -- 10 days * 95 minutes per day
  required_total_time - total_skating_time_8_days = 215 :=
by
  sorry

end gage_skating_time_l1488_148803


namespace triangle_side_ratio_l1488_148897

variable (A B C : ℝ)  -- angles in radians
variable (a b c : ℝ)  -- sides of triangle

theorem triangle_side_ratio
  (h : a * Real.sin A * Real.sin B + b * Real.cos A ^ 2 = Real.sqrt 2 * a) :
  b / a = Real.sqrt 2 :=
by sorry

end triangle_side_ratio_l1488_148897


namespace point_inside_circle_l1488_148822

theorem point_inside_circle : 
  ∀ (x y : ℝ), 
  (x-2)^2 + (y-3)^2 = 4 → 
  (3-2)^2 + (2-3)^2 < 4 :=
by
  intro x y h
  sorry

end point_inside_circle_l1488_148822
