import Mathlib

namespace remainder_of_base12_integer_divided_by_9_l1798_179820

-- Define the base-12 integer
def base12_integer := 2 * 12^3 + 7 * 12^2 + 4 * 12 + 3

-- Define the condition for our problem
def divisor := 9

-- State the theorem to be proved
theorem remainder_of_base12_integer_divided_by_9 :
  base12_integer % divisor = 0 :=
sorry

end remainder_of_base12_integer_divided_by_9_l1798_179820


namespace yan_ratio_l1798_179857

variables (w x y : ℝ)

-- Given conditions
def yan_conditions : Prop :=
  w > 0 ∧ 
  x > 0 ∧ 
  y > 0 ∧ 
  (y / w = x / w + (x + y) / (7 * w))

-- The ratio of Yan's distance from his home to his distance from the stadium is 3/4
theorem yan_ratio (h : yan_conditions w x y) : 
  x / y = 3 / 4 :=
sorry

end yan_ratio_l1798_179857


namespace no_such_natural_numbers_l1798_179878

theorem no_such_natural_numbers :
  ¬(∃ (a b c : ℕ), a > 1 ∧ b > 1 ∧ c > 1 ∧
  (b ∣ a^2 - 1) ∧ (c ∣ a^2 - 1) ∧
  (a ∣ b^2 - 1) ∧ (c ∣ b^2 - 1) ∧
  (a ∣ c^2 - 1) ∧ (b ∣ c^2 - 1)) :=
by sorry

end no_such_natural_numbers_l1798_179878


namespace white_given_popped_l1798_179897

-- Define the conditions
def white_kernels : ℚ := 1 / 2
def yellow_kernels : ℚ := 1 / 3
def blue_kernels : ℚ := 1 / 6

def white_kernels_pop : ℚ := 3 / 4
def yellow_kernels_pop : ℚ := 1 / 2
def blue_kernels_pop : ℚ := 1 / 3

def probability_white_popped : ℚ := white_kernels * white_kernels_pop
def probability_yellow_popped : ℚ := yellow_kernels * yellow_kernels_pop
def probability_blue_popped : ℚ := blue_kernels * blue_kernels_pop

def probability_popped : ℚ := probability_white_popped + probability_yellow_popped + probability_blue_popped

-- The theorem to be proved
theorem white_given_popped : (probability_white_popped / probability_popped) = (27 / 43) := 
by sorry

end white_given_popped_l1798_179897


namespace find_complex_z_l1798_179863

theorem find_complex_z (z : ℂ) (i : ℂ) (hi : i * i = -1) (h : z / (1 - 2 * i) = i) :
  z = 2 + i :=
sorry

end find_complex_z_l1798_179863


namespace value_of_a_l1798_179806

theorem value_of_a (a x y : ℤ) (h1 : x = 2) (h2 : y = 1) (h3 : a * x - 3 * y = 1) : a = 2 := by
  sorry

end value_of_a_l1798_179806


namespace minimum_distance_from_midpoint_to_y_axis_l1798_179887

theorem minimum_distance_from_midpoint_to_y_axis (M N : ℝ × ℝ) (P : ℝ × ℝ)
  (hM : M.snd ^ 2 = M.fst) (hN : N.snd ^ 2 = N.fst)
  (hlength : (M.fst - N.fst)^2 + (M.snd - N.snd)^2 = 16)
  (hP : P = ((M.fst + N.fst) / 2, (M.snd + N.snd) / 2)) :
  abs P.fst = 7 / 4 :=
sorry

end minimum_distance_from_midpoint_to_y_axis_l1798_179887


namespace window_treatments_cost_l1798_179874

def cost_of_sheers (n : ℕ) (cost_per_pair : ℝ) : ℝ := n * cost_per_pair
def cost_of_drapes (n : ℕ) (cost_per_pair : ℝ) : ℝ := n * cost_per_pair
def total_cost (n : ℕ) (cost_sheers : ℝ) (cost_drapes : ℝ) : ℝ :=
  cost_of_sheers n cost_sheers + cost_of_drapes n cost_drapes

theorem window_treatments_cost :
  total_cost 3 40 60 = 300 :=
by
  sorry

end window_treatments_cost_l1798_179874


namespace distance_light_travels_250_years_l1798_179800

def distance_light_travels_one_year : ℝ := 5.87 * 10^12
def years : ℝ := 250

theorem distance_light_travels_250_years :
  distance_light_travels_one_year * years = 1.4675 * 10^15 :=
by
  sorry

end distance_light_travels_250_years_l1798_179800


namespace maxSUVMileage_l1798_179869

noncomputable def maxSUVDistance : ℝ := 217.12

theorem maxSUVMileage 
    (tripGal : ℝ) (mpgHighway : ℝ) (mpgCity : ℝ)
    (regularHighwayRatio : ℝ) (regularCityRatio : ℝ)
    (peakHighwayRatio : ℝ) (peakCityRatio : ℝ) :
    tripGal = 23 →
    mpgHighway = 12.2 →
    mpgCity = 7.6 →
    regularHighwayRatio = 0.4 →
    regularCityRatio = 0.6 →
    peakHighwayRatio = 0.25 →
    peakCityRatio = 0.75 →
    max ((tripGal * regularHighwayRatio * mpgHighway) + (tripGal * regularCityRatio * mpgCity))
        ((tripGal * peakHighwayRatio * mpgHighway) + (tripGal * peakCityRatio * mpgCity)) = maxSUVDistance :=
by
  intros
  -- Proof would go here
  sorry

end maxSUVMileage_l1798_179869


namespace find_a_if_f_is_odd_l1798_179815

noncomputable def f (a x : ℝ) : ℝ := (a * 2^x + a - 2) / (2^x + 1)

theorem find_a_if_f_is_odd :
  (∀ x : ℝ, f 1 x = -f 1 (-x)) ↔ (1 = 1) :=
by
  sorry

end find_a_if_f_is_odd_l1798_179815


namespace hotdogs_sold_l1798_179848

-- Definitions of initial and remaining hotdogs
def initial : ℕ := 99
def remaining : ℕ := 97

-- The statement that needs to be proven
theorem hotdogs_sold : initial - remaining = 2 :=
by
  sorry

end hotdogs_sold_l1798_179848


namespace class_student_difference_l1798_179891

theorem class_student_difference (A B : ℕ) (h : A - 4 = B + 4) : A - B = 8 := by
  sorry

end class_student_difference_l1798_179891


namespace abs_sum_plus_two_eq_sum_abs_l1798_179842

theorem abs_sum_plus_two_eq_sum_abs {a b c : ℤ} (h : |a + b + c| + 2 = |a| + |b| + |c|) :
  a^2 = 1 ∨ b^2 = 1 ∨ c^2 = 1 :=
sorry

end abs_sum_plus_two_eq_sum_abs_l1798_179842


namespace length_AE_l1798_179898

theorem length_AE (A B C D E : Type) 
  (AB AC AD AE : ℝ) 
  (angle_BAC : ℝ)
  (h1 : AB = 4.5) 
  (h2 : AC = 5) 
  (h3 : angle_BAC = 30) 
  (h4 : AD = 1.5) 
  (h5 : AD / AB = AE / AC) : 
  AE = 1.6667 := 
sorry

end length_AE_l1798_179898


namespace find_number_of_pencils_l1798_179886

-- Define the conditions
def number_of_people : Nat := 6
def notebooks_per_person : Nat := 9
def number_of_notebooks : Nat := number_of_people * notebooks_per_person
def pencils_multiplier : Nat := 6
def number_of_pencils : Nat := pencils_multiplier * number_of_notebooks

-- Prove the main statement
theorem find_number_of_pencils : number_of_pencils = 324 :=
by
  sorry

end find_number_of_pencils_l1798_179886


namespace steven_set_aside_pears_l1798_179851

theorem steven_set_aside_pears :
  ∀ (apples pears grapes neededSeeds seedPerApple seedPerPear seedPerGrape : ℕ),
    apples = 4 →
    grapes = 9 →
    neededSeeds = 60 →
    seedPerApple = 6 →
    seedPerPear = 2 →
    seedPerGrape = 3 →
    (neededSeeds - 3) = (apples * seedPerApple + grapes * seedPerGrape + pears * seedPerPear) →
    pears = 3 :=
by
  intros apples pears grapes neededSeeds seedPerApple seedPerPear seedPerGrape
  intros h_apple h_grape h_needed h_seedApple h_seedPear h_seedGrape
  intros h_totalSeeds
  sorry

end steven_set_aside_pears_l1798_179851


namespace marys_age_l1798_179807

variable (M R : ℕ) -- Define M (Mary's current age) and R (Rahul's current age) as natural numbers

theorem marys_age
  (h1 : R = M + 40)       -- Rahul is 40 years older than Mary
  (h2 : R + 30 = 3 * (M + 30))  -- In 30 years, Rahul will be three times as old as Mary
  : M = 20 := 
sorry  -- The proof goes here

end marys_age_l1798_179807


namespace unique_solution_pair_l1798_179827

open Real

theorem unique_solution_pair :
  ∃! (x y : ℝ), y = (x-1)^2 ∧ x * y - y = -3 :=
sorry

end unique_solution_pair_l1798_179827


namespace BDD1H_is_Spatial_in_Cube_l1798_179829

structure Point3D :=
(x : ℝ) (y : ℝ) (z : ℝ)

structure Cube :=
(A B C D A1 B1 C1 D1 : Point3D)
(midpoint_B1C1 : Point3D)
(middle_B1C1 : midpoint_B1C1 = ⟨(B1.x + C1.x) / 2, (B1.y + C1.y) / 2, (B1.z + C1.z) / 2⟩)

def is_not_planar (a b c d : Point3D) : Prop :=
¬ ∃ α β γ δ : ℝ, α * a.x + β * a.y + γ * a.z + δ = 0 ∧ 
                α * b.x + β * b.y + γ * b.z + δ = 0 ∧ 
                α * c.x + β * c.y + γ * c.z + δ = 0 ∧ 
                α * d.x + β * d.y + γ * d.z + δ = 0

def BDD1H_is_spatial (cube : Cube) : Prop :=
is_not_planar cube.B cube.D cube.D1 cube.midpoint_B1C1

theorem BDD1H_is_Spatial_in_Cube (cube : Cube) : BDD1H_is_spatial cube :=
sorry

end BDD1H_is_Spatial_in_Cube_l1798_179829


namespace comb_10_3_eq_120_l1798_179843

theorem comb_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end comb_10_3_eq_120_l1798_179843


namespace inverse_proportion_function_m_neg_l1798_179834

theorem inverse_proportion_function_m_neg
  (x : ℝ) (y : ℝ) (m : ℝ)
  (h1 : y = m / x)
  (h2 : (x < 0 → y > 0) ∧ (x > 0 → y < 0)) :
  m < 0 :=
sorry

end inverse_proportion_function_m_neg_l1798_179834


namespace carts_needed_each_day_last_two_days_l1798_179813

-- Define capacities as per conditions
def daily_capacity_large_truck : ℚ := 1 / (3 * 4)
def daily_capacity_small_truck : ℚ := 1 / (4 * 5)
def daily_capacity_cart : ℚ := 1 / (20 * 6)

-- Define the number of carts required each day in the last two days
def required_carts_last_two_days : ℚ :=
  let total_work_done_by_large_trucks := 2 * daily_capacity_large_truck * 2
  let total_work_done_by_small_trucks := 3 * daily_capacity_small_truck * 2
  let total_work_done_by_carts := 7 * daily_capacity_cart * 2
  let total_work_done := total_work_done_by_large_trucks + total_work_done_by_small_trucks + total_work_done_by_carts
  let remaining_work := 1 - total_work_done
  remaining_work / (2 * daily_capacity_cart)

-- Assertion of the number of carts required
theorem carts_needed_each_day_last_two_days :
  required_carts_last_two_days = 15 := by
  sorry

end carts_needed_each_day_last_two_days_l1798_179813


namespace kanul_initial_amount_l1798_179880

noncomputable def initial_amount : ℝ :=
  (5000 : ℝ) + 200 + 1200 + (11058.82 : ℝ) * 0.15 + 3000

theorem kanul_initial_amount (X : ℝ) 
  (raw_materials : ℝ := 5000) 
  (machinery : ℝ := 200) 
  (employee_wages : ℝ := 1200) 
  (maintenance_cost : ℝ := 0.15 * X)
  (remaining_balance : ℝ := 3000) 
  (expenses : ℝ := raw_materials + machinery + employee_wages + maintenance_cost) 
  (total_expenses : ℝ := expenses + remaining_balance) :
  X = total_expenses :=
by sorry

end kanul_initial_amount_l1798_179880


namespace product_xyz_one_l1798_179821

theorem product_xyz_one (x y z : ℝ) (h1 : x + 1/y = 2) (h2 : y + 1/z = 2) (h3 : z + 1/x = 2) : x * y * z = 1 := 
by {
    sorry
}

end product_xyz_one_l1798_179821


namespace efficiency_and_days_l1798_179849

noncomputable def sakshi_efficiency : ℝ := 1 / 25
noncomputable def tanya_efficiency : ℝ := 1.25 * sakshi_efficiency
noncomputable def ravi_efficiency : ℝ := 0.70 * sakshi_efficiency
noncomputable def combined_efficiency : ℝ := sakshi_efficiency + tanya_efficiency + ravi_efficiency
noncomputable def days_to_complete_work : ℝ := 1 / combined_efficiency

theorem efficiency_and_days:
  combined_efficiency = 29.5 / 250 ∧
  days_to_complete_work = 250 / 29.5 :=
by
  sorry

end efficiency_and_days_l1798_179849


namespace part_one_part_two_l1798_179814

noncomputable def problem_conditions (θ : ℝ) : Prop :=
  let sin_theta := Real.sin θ
  let cos_theta := Real.cos θ
  ∃ m : ℝ, (∀ x : ℝ, x^2 - (Real.sqrt 3 - 1) * x + m = 0 → (x = sin_theta ∨ x = cos_theta))

theorem part_one (θ: ℝ) (h: problem_conditions θ) : 
  let sin_theta := Real.sin θ
  let cos_theta := Real.cos θ
  let m := sin_theta * cos_theta
  m = (3 - 2 * Real.sqrt 3) / 2 :=
sorry

theorem part_two (θ: ℝ) (h: problem_conditions θ) : 
  let sin_theta := Real.sin θ
  let cos_theta := Real.cos θ
  let tan_theta := sin_theta / cos_theta
  (cos_theta - sin_theta * tan_theta) / (1 - tan_theta) = Real.sqrt 3 - 1 :=
sorry

end part_one_part_two_l1798_179814


namespace cos_trig_identity_l1798_179875

theorem cos_trig_identity (α : Real) 
  (h : Real.cos (Real.pi / 6 - α) = 3 / 5) : 
  Real.cos (5 * Real.pi / 6 + α) = - (3 / 5) :=
by
  sorry

end cos_trig_identity_l1798_179875


namespace family_snails_l1798_179868

def total_snails_family (n1 n2 n3 n4 : ℕ) (mother_find : ℕ) : ℕ :=
  n1 + n2 + n3 + mother_find

def first_ducklings_snails (num_ducklings : ℕ) (snails_per_duckling : ℕ) : ℕ :=
  num_ducklings * snails_per_duckling

def remaining_ducklings_snails (num_ducklings : ℕ) (mother_snails : ℕ) : ℕ :=
  num_ducklings * (mother_snails / 2)

def mother_find_snails (snails_group1 : ℕ) (snails_group2 : ℕ) : ℕ :=
  3 * (snails_group1 + snails_group2)

theorem family_snails : 
  ∀ (ducklings : ℕ) (group1_ducklings group2_ducklings : ℕ) 
    (snails1 snails2 : ℕ) 
    (total_ducklings : ℕ), 
    ducklings = 8 →
    group1_ducklings = 3 → 
    group2_ducklings = 3 → 
    snails1 = 5 →
    snails2 = 9 →
    total_ducklings = group1_ducklings + group2_ducklings + 2 →
    total_snails_family 
      (first_ducklings_snails group1_ducklings snails1)
      (first_ducklings_snails group2_ducklings snails2)
      (remaining_ducklings_snails 2 (mother_find_snails 
        (first_ducklings_snails group1_ducklings snails1)
        (first_ducklings_snails group2_ducklings snails2)))
      (mother_find_snails 
        (first_ducklings_snails group1_ducklings snails1)
        (first_ducklings_snails group2_ducklings snails2)) 
    = 294 :=
by intros; sorry

end family_snails_l1798_179868


namespace number_of_strikers_l1798_179895

theorem number_of_strikers (goalies defenders total_players midfielders strikers : ℕ)
  (h1 : goalies = 3)
  (h2 : defenders = 10)
  (h3 : midfielders = 2 * defenders)
  (h4 : total_players = 40)
  (h5 : total_players = goalies + defenders + midfielders + strikers) :
  strikers = 7 :=
by
  sorry

end number_of_strikers_l1798_179895


namespace cafeteria_B_turnover_higher_in_May_l1798_179866

noncomputable def initial_turnover (X a r : ℝ) : Prop :=
  ∃ (X a r : ℝ),
    (X + 8 * a = X * (1 + r) ^ 8) ∧
    ((X + 4 * a) < (X * (1 + r) ^ 4))

theorem cafeteria_B_turnover_higher_in_May (X a r : ℝ) :
    (X + 8 * a = X * (1 + r) ^ 8) → (X + 4 * a < X * (1 + r) ^ 4) :=
  sorry

end cafeteria_B_turnover_higher_in_May_l1798_179866


namespace circumscribed_circle_area_l1798_179882

theorem circumscribed_circle_area (x y c : ℝ)
  (h1 : x + y + c = 24)
  (h2 : x * y = 48)
  (h3 : x^2 + y^2 = c^2) :
  ∃ R : ℝ, (x + y + 2 * R = 24) ∧ (π * R^2 = 25 * π) := 
sorry

end circumscribed_circle_area_l1798_179882


namespace equal_roots_for_specific_k_l1798_179892

theorem equal_roots_for_specific_k (k : ℝ) :
  ((k - 1) * x^2 + 6 * x + 9 = 0) → (6^2 - 4*(k-1)*9 = 0) → (k = 2) :=
by sorry

end equal_roots_for_specific_k_l1798_179892


namespace find_solutions_l1798_179858

-- Defining the system of equations as conditions
def cond1 (a b : ℕ) := a * b + 2 * a - b = 58
def cond2 (b c : ℕ) := b * c + 4 * b + 2 * c = 300
def cond3 (c d : ℕ) := c * d - 6 * c + 4 * d = 101

-- Theorem to prove the solutions satisfy the system of equations
theorem find_solutions (a b c d : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0):
  cond1 a b ∧ cond2 b c ∧ cond3 c d ↔ (a, b, c, d) ∈ [(3, 26, 7, 13), (15, 2, 73, 7)] :=
by sorry

end find_solutions_l1798_179858


namespace solutions_periodic_with_same_period_l1798_179819

variable {y z : ℝ → ℝ}
variable (f g : ℝ → ℝ)

-- defining the conditions
variable (h1 : ∀ x, deriv y x = - (z x)^3)
variable (h2 : ∀ x, deriv z x = (y x)^3)
variable (h3 : y 0 = 1)
variable (h4 : z 0 = 0)
variable (h5 : ∀ x, y x = f x)
variable (h6 : ∀ x, z x = g x)

-- proving periodicity
theorem solutions_periodic_with_same_period : ∃ k > 0, (∀ x, f (x + k) = f x ∧ g (x + k) = g x) := by
  sorry

end solutions_periodic_with_same_period_l1798_179819


namespace input_equals_output_l1798_179876

theorem input_equals_output (x : ℝ) :
  (x ≤ 1 → 2 * x - 3 = x) ∨ (x > 1 → x^2 - 3 * x + 3 = x) ↔ x = 3 :=
by
  sorry

end input_equals_output_l1798_179876


namespace find_larger_number_l1798_179802

-- Definitions based on the conditions
variables (x y : ℕ)

-- Main theorem
theorem find_larger_number (h1 : x + y = 50) (h2 : x - y = 10) : x = 30 :=
by
  sorry

end find_larger_number_l1798_179802


namespace seventh_term_of_geometric_sequence_l1798_179846

theorem seventh_term_of_geometric_sequence :
  ∀ (a r : ℝ), (a * r ^ 3 = 16) → (a * r ^ 8 = 2) → (a * r ^ 6 = 2) :=
by
  intros a r h1 h2
  sorry

end seventh_term_of_geometric_sequence_l1798_179846


namespace range_of_a_l1798_179822

open Real

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 < x → 2 * x * log x ≥ -x^2 + a * x - 3) → a ≤ 4 :=
by 
  sorry

end range_of_a_l1798_179822


namespace percent_of_part_l1798_179809

variable (Part : ℕ) (Whole : ℕ)

theorem percent_of_part (hPart : Part = 70) (hWhole : Whole = 280) :
  (Part / Whole) * 100 = 25 := by
  sorry

end percent_of_part_l1798_179809


namespace total_amount_owed_l1798_179840

theorem total_amount_owed :
  ∃ (P remaining_balance processing_fee new_total discount: ℝ),
    0.05 * P = 50 ∧
    remaining_balance = P - 50 ∧
    processing_fee = 0.03 * remaining_balance ∧
    new_total = remaining_balance + processing_fee ∧
    discount = 0.10 * new_total ∧
    new_total - discount = 880.65 :=
sorry

end total_amount_owed_l1798_179840


namespace number_of_ants_in_section_correct_l1798_179805

noncomputable def ants_in_section := 
  let width_feet : ℝ := 600
  let length_feet : ℝ := 800
  let ants_per_square_inch : ℝ := 5
  let side_feet : ℝ := 200
  let feet_to_inches : ℝ := 12
  let side_inches := side_feet * feet_to_inches
  let area_section_square_inches := side_inches^2
  ants_per_square_inch * area_section_square_inches

theorem number_of_ants_in_section_correct :
  ants_in_section = 28800000 := 
by 
  unfold ants_in_section 
  sorry

end number_of_ants_in_section_correct_l1798_179805


namespace jaden_time_difference_l1798_179888

-- Define the conditions as hypotheses
def jaden_time_as_girl (distance : ℕ) (time : ℕ) : Prop :=
  distance = 20 ∧ time = 240

def jaden_time_as_woman (distance : ℕ) (time : ℕ) : Prop :=
  distance = 8 ∧ time = 240

-- Define the proof problem
theorem jaden_time_difference
  (d_girl t_girl d_woman t_woman : ℕ)
  (H_girl : jaden_time_as_girl d_girl t_girl)
  (H_woman : jaden_time_as_woman d_woman t_woman)
  : (t_woman / d_woman) - (t_girl / d_girl) = 18 :=
by
  sorry

end jaden_time_difference_l1798_179888


namespace customers_stayed_behind_l1798_179845

theorem customers_stayed_behind : ∃ x : ℕ, (x + (x + 5) = 11) ∧ x = 3 := by
  sorry

end customers_stayed_behind_l1798_179845


namespace johns_total_spending_l1798_179817

theorem johns_total_spending:
  ∀ (X : ℝ), (3/7 * X + 2/5 * X + 1/4 * X + 1/14 * X + 12 = X) → X = 80 :=
by
  intro X h
  sorry

end johns_total_spending_l1798_179817


namespace find_ax6_by6_l1798_179847

variable {a b x y : ℝ}

theorem find_ax6_by6
  (h1 : a * x + b * y = 5)
  (h2 : a * x^2 + b * y^2 = 12)
  (h3 : a * x^3 + b * y^3 = 30)
  (h4 : a * x^4 + b * y^4 = 80) :
  a * x^6 + b * y^6 = 1531.25 :=
sorry

end find_ax6_by6_l1798_179847


namespace double_recipe_total_l1798_179877

theorem double_recipe_total 
  (butter_ratio : ℕ) (flour_ratio : ℕ) (sugar_ratio : ℕ) 
  (flour_cups : ℕ) 
  (h_ratio : butter_ratio = 2) 
  (h_flour : flour_ratio = 5) 
  (h_sugar : sugar_ratio = 3) 
  (h_flour_cups : flour_cups = 15) : 
  2 * ((butter_ratio * (flour_cups / flour_ratio)) + flour_cups + (sugar_ratio * (flour_cups / flour_ratio))) = 60 := 
by 
  sorry

end double_recipe_total_l1798_179877


namespace range_of_hx_l1798_179837

open Real

theorem range_of_hx (h : ℝ → ℝ) (a b : ℝ) (H_def : ∀ x : ℝ, h x = 3 / (1 + 3 * x^4)) 
  (H_range : ∀ y : ℝ, (y > 0 ∧ y ≤ 3) ↔ ∃ x : ℝ, h x = y) : 
  a + b = 3 := 
sorry

end range_of_hx_l1798_179837


namespace angle_difference_parallelogram_l1798_179889

theorem angle_difference_parallelogram (A B : ℝ) (hA : A = 55) (h1 : A + B = 180) :
  B - A = 70 := 
by
  sorry

end angle_difference_parallelogram_l1798_179889


namespace angle_is_40_l1798_179890

theorem angle_is_40 (x : ℝ) 
  : (180 - x = 2 * (90 - x) + 40) → x = 40 :=
by
  sorry

end angle_is_40_l1798_179890


namespace greater_num_792_l1798_179850

theorem greater_num_792 (x y : ℕ) (h1 : x + y = 1443) (h2 : x - y = 141) : x = 792 :=
by
  sorry

end greater_num_792_l1798_179850


namespace smallest_M_value_l1798_179841

theorem smallest_M_value 
  (a b c d e : ℕ) 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h5 : 0 < e) 
  (h_sum : a + b + c + d + e = 2010) : 
  (∃ M, M = max (a+b) (max (b+c) (max (c+d) (d+e))) ∧ M = 671) :=
by
  sorry

end smallest_M_value_l1798_179841


namespace sum_of_distinct_nums_l1798_179838

theorem sum_of_distinct_nums (m n p q : ℕ) (hmn : m ≠ n) (hmp : m ≠ p) (hmq : m ≠ q) 
(hnp : n ≠ p) (hnq : n ≠ q) (hpq : p ≠ q) (pos_m : 0 < m) (pos_n : 0 < n) 
(pos_p : 0 < p) (pos_q : 0 < q) (h : (6 - m) * (6 - n) * (6 - p) * (6 - q) = 4) : 
  m + n + p + q = 24 :=
sorry

end sum_of_distinct_nums_l1798_179838


namespace megan_bottles_left_l1798_179801

-- Defining the initial conditions
def initial_bottles : Nat := 17
def bottles_drank : Nat := 3

-- Theorem stating that Megan has 14 bottles left
theorem megan_bottles_left : initial_bottles - bottles_drank = 14 := by
  sorry

end megan_bottles_left_l1798_179801


namespace relationship_of_abc_l1798_179881

theorem relationship_of_abc (a b c : ℝ) 
  (h1 : b + c = 6 - 4 * a + 3 * a^2) 
  (h2 : c - b = 4 - 4 * a + a^2) : 
  a < b ∧ b ≤ c := 
sorry

end relationship_of_abc_l1798_179881


namespace min_total_cost_of_container_l1798_179859

-- Definitions from conditions
def container_volume := 4 -- m^3
def container_height := 1 -- m
def cost_per_square_meter_base : ℝ := 20
def cost_per_square_meter_sides : ℝ := 10

-- Proving the minimum total cost
theorem min_total_cost_of_container :
  ∃ (a b : ℝ), a * b = container_volume ∧
                (20 * (a + b) + 20 * (a * b)) = 160 :=
by
  sorry

end min_total_cost_of_container_l1798_179859


namespace find_children_tickets_l1798_179804

variable (A C S : ℝ)

theorem find_children_tickets 
  (h1 : A + C + S = 600)
  (h2 : 6 * A + 4.5 * C + 5 * S = 3250) :
  C = (350 - S) / 1.5 := 
sorry

end find_children_tickets_l1798_179804


namespace tree_planting_total_l1798_179833

theorem tree_planting_total (t4 t5 t6 : ℕ) 
  (h1 : t4 = 30)
  (h2 : t5 = 2 * t4)
  (h3 : t6 = 3 * t5 - 30) : 
  t4 + t5 + t6 = 240 := 
by 
  sorry

end tree_planting_total_l1798_179833


namespace length_EF_l1798_179854

theorem length_EF
  (AB CD GH EF : ℝ)
  (h1 : AB = 180)
  (h2 : CD = 120)
  (h3 : AB = 2 * GH)
  (h4 : CD = 2 * EF) :
  EF = 45 :=
by
  sorry

end length_EF_l1798_179854


namespace train_crossing_time_l1798_179836

def train_length : ℝ := 150
def train_speed : ℝ := 179.99999999999997

theorem train_crossing_time : train_length / train_speed = 0.8333333333333333 := by
  sorry

end train_crossing_time_l1798_179836


namespace intersection_points_l1798_179852

theorem intersection_points (x y : ℝ) (h1 : x^2 - 4 * y^2 = 4) (h2 : x = 3 * y) : 
  (x, y) = (3, 1) ∨ (x, y) = (-3, -1) :=
sorry

end intersection_points_l1798_179852


namespace third_snail_time_l1798_179899

theorem third_snail_time
  (speed_first_snail : ℝ)
  (speed_second_snail : ℝ)
  (speed_third_snail : ℝ)
  (time_first_snail : ℝ)
  (distance : ℝ) :
  (speed_first_snail = 2) →
  (speed_second_snail = 2 * speed_first_snail) →
  (speed_third_snail = 5 * speed_second_snail) →
  (time_first_snail = 20) →
  (distance = speed_first_snail * time_first_snail) →
  (distance / speed_third_snail = 2) :=
by
  sorry

end third_snail_time_l1798_179899


namespace t_minus_d_l1798_179816

-- Define amounts paid by Tom, Dorothy, and Sammy
def tom_paid : ℕ := 140
def dorothy_paid : ℕ := 90
def sammy_paid : ℕ := 220

-- Define the total amount and required equal share
def total_paid : ℕ := tom_paid + dorothy_paid + sammy_paid
def equal_share : ℕ := total_paid / 3

-- Define the amounts t and d where Tom and Dorothy balance the costs by paying Sammy
def t : ℤ := equal_share - tom_paid -- Amount Tom gave to Sammy
def d : ℤ := equal_share - dorothy_paid -- Amount Dorothy gave to Sammy

-- Prove that t - d = -50
theorem t_minus_d : t - d = -50 := by
  sorry

end t_minus_d_l1798_179816


namespace perfect_square_eq_m_val_l1798_179831

theorem perfect_square_eq_m_val (m : ℝ) (h : ∃ a : ℝ, x^2 - m * x + 49 = (x - a)^2) : m = 14 ∨ m = -14 :=
by
  sorry

end perfect_square_eq_m_val_l1798_179831


namespace required_volume_proof_l1798_179860

-- Defining the conditions
def initial_volume : ℝ := 60
def initial_concentration : ℝ := 0.10
def final_concentration : ℝ := 0.15

-- Defining the equation
def required_volume (V : ℝ) : Prop :=
  (initial_concentration * initial_volume + V = final_concentration * (initial_volume + V))

-- Stating the proof problem
theorem required_volume_proof :
  ∃ V : ℝ, required_volume V ∧ V = 3 / 0.85 :=
by {
  -- Proof skipped
  sorry
}

end required_volume_proof_l1798_179860


namespace heat_capacity_at_100K_l1798_179828

noncomputable def heat_capacity (t : ℝ) : ℝ :=
  0.1054 + 0.000004 * t

theorem heat_capacity_at_100K :
  heat_capacity 100 = 0.1058 := 
by
  sorry

end heat_capacity_at_100K_l1798_179828


namespace blake_change_l1798_179823

theorem blake_change :
  let lollipop_count := 4
  let chocolate_count := 6
  let lollipop_cost := 2
  let chocolate_cost := 4 * lollipop_cost
  let total_received := 6 * 10
  let total_cost := (lollipop_count * lollipop_cost) + (chocolate_count * chocolate_cost)
  let change := total_received - total_cost
  change = 4 :=
by
  sorry

end blake_change_l1798_179823


namespace algebraic_expression_value_l1798_179811

theorem algebraic_expression_value (a : ℝ) (h : a^2 + a - 1 = 0) : a^2 + a + 1 = 2 :=
sorry

end algebraic_expression_value_l1798_179811


namespace probability_other_side_green_l1798_179808

-- Definitions based on the conditions
def Card : Type := ℕ
def num_cards : ℕ := 8
def blue_blue : ℕ := 4
def blue_green : ℕ := 2
def green_green : ℕ := 2

def total_green_sides : ℕ := (green_green * 2) + blue_green
def green_opposite_green_side : ℕ := green_green * 2

theorem probability_other_side_green (h_total_green_sides : total_green_sides = 6)
(h_green_opposite_green_side : green_opposite_green_side = 4) :
  (green_opposite_green_side / total_green_sides : ℚ) = 2 / 3 := 
by
  sorry

end probability_other_side_green_l1798_179808


namespace quadratic_condition_l1798_179861

theorem quadratic_condition (m : ℝ) (h1 : m^2 - 2 = 2) (h2 : m + 2 ≠ 0) : m = 2 :=
by
  sorry

end quadratic_condition_l1798_179861


namespace volume_of_spheres_l1798_179839

noncomputable def sphere_volume (a : ℝ) : ℝ :=
  (4 / 3) * Real.pi * ((3 * a - a * Real.sqrt 3) / 4)^3

theorem volume_of_spheres (a : ℝ) : 
  ∃ r : ℝ, r = (3 * a - a * Real.sqrt 3) / 4 ∧ 
  sphere_volume a = (4 / 3) * Real.pi * r^3 := 
sorry

end volume_of_spheres_l1798_179839


namespace option_d_satisfies_equation_l1798_179871

theorem option_d_satisfies_equation (x y z : ℤ) (h1 : x = z) (h2 : y = x + 1) : x * (x - y) + y * (y - z) + z * (z - x) = 2 :=
by
  sorry

end option_d_satisfies_equation_l1798_179871


namespace shirt_cost_l1798_179894

theorem shirt_cost (J S : ℝ) (h1 : 3 * J + 2 * S = 69) (h2 : 2 * J + 3 * S = 66) : S = 12 :=
by
  sorry

end shirt_cost_l1798_179894


namespace alan_total_cost_is_84_l1798_179824

def num_dark_cds : ℕ := 2
def num_avn_cds : ℕ := 1
def num_90s_cds : ℕ := 5
def price_avn_cd : ℕ := 12 -- in dollars
def price_dark_cd : ℕ := price_avn_cd * 2
def total_cost_other_cds : ℕ := num_dark_cds * price_dark_cd + num_avn_cds * price_avn_cd
def price_90s_cds : ℕ := ((40 : ℕ) * total_cost_other_cds) / 100
def total_cost_all_products : ℕ := num_dark_cds * price_dark_cd + num_avn_cds * price_avn_cd + price_90s_cds

theorem alan_total_cost_is_84 : total_cost_all_products = 84 := by
  sorry

end alan_total_cost_is_84_l1798_179824


namespace no_integer_coordinates_between_A_and_B_l1798_179879

section
variable (A B : ℤ × ℤ)
variable (Aeq : A = (2, 3))
variable (Beq : B = (50, 305))

theorem no_integer_coordinates_between_A_and_B :
  (∀ P : ℤ × ℤ, P.1 > 2 ∧ P.1 < 50 ∧ P.2 = (151 * P.1 - 230) / 24 → False) :=
by
  sorry
end

end no_integer_coordinates_between_A_and_B_l1798_179879


namespace seven_does_not_always_divide_l1798_179885

theorem seven_does_not_always_divide (n : ℤ) :
  ¬(7 ∣ (n ^ 2225 - n ^ 2005)) :=
by sorry

end seven_does_not_always_divide_l1798_179885


namespace calc_expression_l1798_179855

theorem calc_expression : 3 ^ 2022 * (1 / 3) ^ 2023 = 1 / 3 :=
by
  sorry

end calc_expression_l1798_179855


namespace total_erasers_l1798_179803

def cases : ℕ := 7
def boxes_per_case : ℕ := 12
def erasers_per_box : ℕ := 25

theorem total_erasers : cases * boxes_per_case * erasers_per_box = 2100 := by
  sorry

end total_erasers_l1798_179803


namespace slope_of_line_through_intersecting_points_of_circles_l1798_179884

theorem slope_of_line_through_intersecting_points_of_circles :
  let circle1 (x y : ℝ) := x^2 + y^2 - 6*x + 4*y - 5 = 0
  let circle2 (x y : ℝ) := x^2 + y^2 - 10*x + 16*y + 24 = 0
  ∀ (C D : ℝ × ℝ), circle1 C.1 C.2 → circle2 C.1 C.2 → circle1 D.1 D.2 → circle2 D.1 D.2 → 
  let dx := D.1 - C.1
  let dy := D.2 - C.2
  dx ≠ 0 → dy / dx = 1 / 3 :=
by
  intros
  sorry

end slope_of_line_through_intersecting_points_of_circles_l1798_179884


namespace subset_implies_range_a_intersection_implies_range_a_l1798_179835

noncomputable def setA : Set ℝ := {x | -1 < x ∧ x < 2}
noncomputable def setB (a : ℝ) : Set ℝ := {x | 2 * a - 1 < x ∧ x < 2 * a + 3}

theorem subset_implies_range_a (a : ℝ) : (setA ⊆ setB a) → (-1/2 ≤ a ∧ a ≤ 0) :=
by
  sorry

theorem intersection_implies_range_a (a : ℝ) : (setA ∩ setB a = ∅) → (a ≤ -2 ∨ a ≥ 3/2) :=
by
  sorry

end subset_implies_range_a_intersection_implies_range_a_l1798_179835


namespace incorrect_judgment_l1798_179896

theorem incorrect_judgment : (∀ x : ℝ, x^2 - 1 ≥ -1) ∧ (4 + 2 ≠ 7) :=
by 
  sorry

end incorrect_judgment_l1798_179896


namespace usual_time_to_school_l1798_179862

theorem usual_time_to_school (R T : ℝ) (h : (R * T = (6/5) * R * (T - 4))) : T = 24 :=
by 
  sorry

end usual_time_to_school_l1798_179862


namespace find_x_angle_l1798_179832

theorem find_x_angle (x : ℝ) (h : x + x + 140 = 360) : x = 110 :=
by
  sorry

end find_x_angle_l1798_179832


namespace AndyCoordinatesAfter1500Turns_l1798_179865

/-- Definition for Andy's movement rules given his starting position. -/
def AndyPositionAfterTurns (turns : ℕ) : ℤ × ℤ :=
  let rec move (x y : ℤ) (length : ℤ) (dir : ℕ) (remainingTurns : ℕ) : ℤ × ℤ :=
    match remainingTurns with
    | 0 => (x, y)
    | n+1 => 
        let (dx, dy) := match dir % 4 with
                        | 0 => (0, 1)
                        | 1 => (1, 0)
                        | 2 => (0, -1)
                        | _ => (-1, 0)
        move (x + dx * length) (y + dy * length) (length + 1) (dir + 1) n
  move (-30) 25 2 0 turns

theorem AndyCoordinatesAfter1500Turns :
  AndyPositionAfterTurns 1500 = (-280141, 280060) :=
by
  sorry

end AndyCoordinatesAfter1500Turns_l1798_179865


namespace alcohol_percentage_in_mixed_solution_l1798_179818

theorem alcohol_percentage_in_mixed_solution :
  let vol1 := 8
  let perc1 := 0.25
  let vol2 := 2
  let perc2 := 0.12
  let total_alcohol := (vol1 * perc1) + (vol2 * perc2)
  let total_volume := vol1 + vol2
  (total_alcohol / total_volume) * 100 = 22.4 := by
  sorry

end alcohol_percentage_in_mixed_solution_l1798_179818


namespace solve_inequality_l1798_179867

theorem solve_inequality (x : ℝ) : ((x + 3) ^ 2 < 1) ↔ (-4 < x ∧ x < -2) := by
  sorry

end solve_inequality_l1798_179867


namespace volume_difference_is_867_25_l1798_179810

noncomputable def charlie_volume : ℝ :=
  let h_C := 9
  let circumference_C := 7
  let r_C := circumference_C / (2 * Real.pi)
  let v_C := Real.pi * r_C^2 * h_C
  v_C

noncomputable def dana_volume : ℝ :=
  let h_D := 5
  let circumference_D := 10
  let r_D := circumference_D / (2 * Real.pi)
  let v_D := Real.pi * r_D^2 * h_D
  v_D

noncomputable def volume_difference : ℝ :=
  Real.pi * (abs (charlie_volume - dana_volume))

theorem volume_difference_is_867_25 : volume_difference = 867.25 := by
  sorry

end volume_difference_is_867_25_l1798_179810


namespace outer_boundary_diameter_l1798_179830

-- Define the given conditions
def fountain_diameter : ℝ := 12
def walking_path_width : ℝ := 6
def garden_ring_width : ℝ := 10

-- Define what we need to prove
theorem outer_boundary_diameter :
  2 * (fountain_diameter / 2 + garden_ring_width + walking_path_width) = 44 :=
by
  sorry

end outer_boundary_diameter_l1798_179830


namespace simplify_fraction_l1798_179825

theorem simplify_fraction (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : y - 1/x ≠ 0) :
  (x - 1/y) / (y - 1/x) = x / y :=
sorry

end simplify_fraction_l1798_179825


namespace vasya_gift_ways_l1798_179883

theorem vasya_gift_ways :
  let cars := 7
  let constructor_sets := 5
  (cars * constructor_sets) + (Nat.choose cars 2) + (Nat.choose constructor_sets 2) = 66 :=
by
  let cars := 7
  let constructor_sets := 5
  sorry

end vasya_gift_ways_l1798_179883


namespace cos_832_eq_cos_l1798_179893

theorem cos_832_eq_cos (n : ℤ) (h1 : 0 ≤ n) (h2 : n ≤ 180) (h3 : Real.cos (n * Real.pi / 180) = Real.cos (832 * Real.pi / 180)) : n = 112 := 
  sorry

end cos_832_eq_cos_l1798_179893


namespace find_digits_of_six_two_digit_sum_equals_528_l1798_179844

theorem find_digits_of_six_two_digit_sum_equals_528
  (a b c : ℕ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_digits : a < 10 ∧ b < 10 ∧ c < 10)
  (h_sum_six_numbers : (10 * a + b) + (10 * a + c) + (10 * b + c) + (10 * b + a) + (10 * c + a) + (10 * c + b) = 528) :
  (a = 7 ∧ b = 8 ∧ c = 9) := 
sorry

end find_digits_of_six_two_digit_sum_equals_528_l1798_179844


namespace beakers_with_copper_l1798_179870

theorem beakers_with_copper :
  ∀ (total_beakers no_copper_beakers beakers_with_copper drops_per_beaker total_drops_used : ℕ),
    total_beakers = 22 →
    no_copper_beakers = 7 →
    drops_per_beaker = 3 →
    total_drops_used = 45 →
    total_drops_used = drops_per_beaker * beakers_with_copper →
    total_beakers = beakers_with_copper + no_copper_beakers →
    beakers_with_copper = 15 := 
-- inserting the placeholder proof 'sorry'
sorry

end beakers_with_copper_l1798_179870


namespace pond_fish_approximation_l1798_179812

noncomputable def total_number_of_fish
  (tagged_first: ℕ) (total_caught_second: ℕ) (tagged_second: ℕ) : ℕ :=
  (tagged_first * total_caught_second) / tagged_second

theorem pond_fish_approximation :
  total_number_of_fish 60 50 2 = 1500 :=
by
  -- calculation of the total number of fish based on given conditions
  sorry

end pond_fish_approximation_l1798_179812


namespace kolya_advantageous_methods_l1798_179853

-- Define the context and conditions
variables (n : ℕ) (h₀ : n ≥ 2)
variables (a b : ℕ) (h₁ : a + b = 2*n + 1) (h₂ : a ≥ 2) (h₃ : b ≥ 2)

-- Define outcomes of the methods
def method1_outcome (a b : ℕ) := max a b + min (a - 1) (b - 1)
def method2_outcome (a b : ℕ) := min a b + min (a - 1) (b - 1)
def method3_outcome (a b : ℕ) := max (method1_outcome a b - 1) (method2_outcome a b - 1)

-- Prove which methods are the most and least advantageous
theorem kolya_advantageous_methods :
  method1_outcome a b >= method2_outcome a b ∧ method1_outcome a b >= method3_outcome a b :=
sorry

end kolya_advantageous_methods_l1798_179853


namespace find_values_of_a_and_b_l1798_179826

theorem find_values_of_a_and_b (a b : ℚ) (h1 : 4 * a + 2 * b = 92) (h2 : 6 * a - 4 * b = 60) : 
  a = 122 / 7 ∧ b = 78 / 7 :=
by {
  sorry
}

end find_values_of_a_and_b_l1798_179826


namespace no_solutions_then_a_eq_zero_l1798_179873

theorem no_solutions_then_a_eq_zero (a b : ℝ) :
  (∀ x y : ℝ, ¬ (y^2 = x^2 + a * x + b ∧ x^2 = y^2 + a * y + b)) → a = 0 :=
by
  sorry

end no_solutions_then_a_eq_zero_l1798_179873


namespace distance_between_A_and_B_l1798_179864

-- Definitions for the problem
def speed_fast_train := 65 -- speed of the first train in km/h
def speed_slow_train := 29 -- speed of the second train in km/h
def time_difference := 5   -- difference in hours

-- Given conditions and the final equation leading to the proof
theorem distance_between_A_and_B :
  ∃ (D : ℝ), D = 9425 / 36 :=
by
  existsi (9425 / 36 : ℝ)
  sorry

end distance_between_A_and_B_l1798_179864


namespace simplify_140_210_l1798_179856

noncomputable def simplify_fraction (num den : Nat) : Nat × Nat :=
  let d := Nat.gcd num den
  (num / d, den / d)

theorem simplify_140_210 :
  simplify_fraction 140 210 = (2, 3) :=
by
  have p140 : 140 = 2^2 * 5 * 7 := by rfl
  have p210 : 210 = 2 * 3 * 5 * 7 := by rfl
  sorry

end simplify_140_210_l1798_179856


namespace reciprocal_of_repeating_decimal_l1798_179872

theorem reciprocal_of_repeating_decimal :
  (1 / (0.33333333 : ℚ)) = 3 := by
  sorry

end reciprocal_of_repeating_decimal_l1798_179872
