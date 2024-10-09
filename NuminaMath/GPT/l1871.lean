import Mathlib

namespace find_max_z_l1871_187190

theorem find_max_z :
  ∃ (x y : ℝ), abs x + abs y ≤ 4 ∧ 2 * x + y ≤ 4 ∧ (2 * x - y) = (20 / 3) :=
by
  sorry

end find_max_z_l1871_187190


namespace total_carriages_l1871_187145

-- Definitions based on given conditions
def Euston_carriages := 130
def Norfolk_carriages := Euston_carriages - 20
def Norwich_carriages := 100
def Flying_Scotsman_carriages := Norwich_carriages + 20
def Victoria_carriages := Euston_carriages - 15
def Waterloo_carriages := Norwich_carriages * 2

-- Theorem to prove the total number of carriages is 775
theorem total_carriages : 
  Euston_carriages + Norfolk_carriages + Norwich_carriages + Flying_Scotsman_carriages + Victoria_carriages + Waterloo_carriages = 775 :=
by sorry

end total_carriages_l1871_187145


namespace breadth_of_rectangle_l1871_187107

theorem breadth_of_rectangle 
  (Perimeter Length Breadth : ℝ)
  (h_perimeter_eq : Perimeter = 2 * (Length + Breadth))
  (h_given_perimeter : Perimeter = 480)
  (h_given_length : Length = 140) :
  Breadth = 100 := 
by
  sorry

end breadth_of_rectangle_l1871_187107


namespace find_second_number_l1871_187198

theorem find_second_number 
  (h1 : (20 + 40 + 60) / 3 = (10 + x + 45) / 3 + 5) :
  x = 50 :=
sorry

end find_second_number_l1871_187198


namespace louis_never_reaches_target_l1871_187195

def stable (p : ℤ × ℤ) : Prop :=
  (p.1 + p.2) % 7 ≠ 0

def move1 (p : ℤ × ℤ) : ℤ × ℤ :=
  (p.2, p.1)

def move2 (p : ℤ × ℤ) : ℤ × ℤ :=
  (3 * p.1, -4 * p.2)

def move3 (p : ℤ × ℤ) : ℤ × ℤ :=
  (-2 * p.1, 5 * p.2)

def move4 (p : ℤ × ℤ) : ℤ × ℤ :=
  (p.1 + 1, p.2 + 6)

def move5 (p : ℤ × ℤ) : ℤ × ℤ :=
  (p.1 - 7, p.2)

-- Define the start and target points
def start : ℤ × ℤ := (0, 1)
def target : ℤ × ℤ := (0, 0)

theorem louis_never_reaches_target :
  ∀ p, (p = start → ¬ ∃ k, move1^[k] p = target) ∧
       (p = start → ¬ ∃ k, move2^[k] p = target) ∧
       (p = start → ¬ ∃ k, move3^[k] p = target) ∧
       (p = start → ¬ ∃ k, move4^[k] p = target) ∧
       (p = start → ¬ ∃ k, move5^[k] p = target) :=
by {
  sorry
}

end louis_never_reaches_target_l1871_187195


namespace joan_gave_sam_seashells_l1871_187159

-- Definitions of initial conditions
def initial_seashells : ℕ := 70
def remaining_seashells : ℕ := 27

-- Theorem statement
theorem joan_gave_sam_seashells : initial_seashells - remaining_seashells = 43 :=
by
  sorry

end joan_gave_sam_seashells_l1871_187159


namespace sin_600_eq_neg_sqrt_3_div_2_l1871_187157

theorem sin_600_eq_neg_sqrt_3_div_2 : Real.sin (600 * Real.pi / 180) = -Real.sqrt 3 / 2 := 
by
  -- proof to be provided here
  sorry

end sin_600_eq_neg_sqrt_3_div_2_l1871_187157


namespace height_of_parallelogram_l1871_187186

-- Define the problem statement
theorem height_of_parallelogram (A : ℝ) (b : ℝ) (h : ℝ) (h_eq : A = b * h) (A_val : A = 384) (b_val : b = 24) : h = 16 :=
by
  -- Skeleton proof, include the initial conditions and proof statement
  sorry

end height_of_parallelogram_l1871_187186


namespace two_a_plus_two_b_plus_two_c_l1871_187140

variable (a b c : ℝ)

-- Defining the conditions as the hypotheses
def condition1 : Prop := b + c = 15 - 4 * a
def condition2 : Prop := a + c = -18 - 4 * b
def condition3 : Prop := a + b = 10 - 4 * c

-- The theorem to prove
theorem two_a_plus_two_b_plus_two_c (h1 : condition1 a b c) (h2 : condition2 a b c) (h3 : condition3 a b c) :
  2 * a + 2 * b + 2 * c = 7 / 3 :=
by
  sorry

end two_a_plus_two_b_plus_two_c_l1871_187140


namespace sum_of_first_seven_primes_with_units_digit_3_lt_150_l1871_187143

def has_units_digit_3 (n : ℕ) : Prop :=
  n % 10 = 3

def is_less_than_150 (n : ℕ) : Prop :=
  n < 150

def first_seven_primes_with_units_digit_3 := [3, 13, 23, 43, 53, 73, 83]

theorem sum_of_first_seven_primes_with_units_digit_3_lt_150 :
  (has_units_digit_3 3) ∧ (is_less_than_150 3) ∧ (Prime 3) ∧
  (has_units_digit_3 13) ∧ (is_less_than_150 13) ∧ (Prime 13) ∧
  (has_units_digit_3 23) ∧ (is_less_than_150 23) ∧ (Prime 23) ∧
  (has_units_digit_3 43) ∧ (is_less_than_150 43) ∧ (Prime 43) ∧
  (has_units_digit_3 53) ∧ (is_less_than_150 53) ∧ (Prime 53) ∧
  (has_units_digit_3 73) ∧ (is_less_than_150 73) ∧ (Prime 73) ∧
  (has_units_digit_3 83) ∧ (is_less_than_150 83) ∧ (Prime 83) →
  (3 + 13 + 23 + 43 + 53 + 73 + 83 = 291) :=
by
  sorry

end sum_of_first_seven_primes_with_units_digit_3_lt_150_l1871_187143


namespace sqrt_difference_l1871_187182

theorem sqrt_difference :
  (Real.sqrt 63 - 7 * Real.sqrt (1 / 7)) = 2 * Real.sqrt 7 :=
by
  sorry

end sqrt_difference_l1871_187182


namespace find_x_l1871_187174

theorem find_x (x : ℝ) : abs (2 * x - 1) = 3 * x + 6 ∧ x + 2 > 0 ↔ x = -1 := 
by
  sorry

end find_x_l1871_187174


namespace minimize_expression_l1871_187142

theorem minimize_expression : ∃ c : ℝ, c = 6 ∧ ∀ x : ℝ, (3 / 4) * (x ^ 2) - 9 * x + 7 ≥ (3 / 4) * (6 ^ 2) - 9 * 6 + 7 :=
by
  sorry

end minimize_expression_l1871_187142


namespace tax_calculation_l1871_187110

variable (winnings : ℝ) (processing_fee : ℝ) (take_home : ℝ)
variable (tax_percentage : ℝ)

def given_conditions : Prop :=
  winnings = 50 ∧ processing_fee = 5 ∧ take_home = 35

def to_prove : Prop :=
  tax_percentage = 20

theorem tax_calculation (h : given_conditions winnings processing_fee take_home) : to_prove tax_percentage :=
by
  sorry

end tax_calculation_l1871_187110


namespace series_sum_equals_one_fourth_l1871_187170

noncomputable def series_term (n : ℕ) : ℝ :=
  3^n / (1 + 3^n + 3^(n+1) + 3^(2*n+1))

noncomputable def infinite_series_sum : ℝ :=
  ∑' (n : ℕ), series_term (n + 1)

theorem series_sum_equals_one_fourth :
  infinite_series_sum = 1 / 4 :=
by
  -- Proof goes here.
  sorry

end series_sum_equals_one_fourth_l1871_187170


namespace sum_mean_median_mode_l1871_187188

def numbers : List ℕ := [3, 5, 3, 0, 2, 5, 0, 2]

def mode (l : List ℕ) : ℝ := 4

def median (l : List ℕ) : ℝ := 2.5

def mean (l : List ℕ) : ℝ := 2.5

theorem sum_mean_median_mode : mean numbers + median numbers + mode numbers = 9 := by
  sorry

end sum_mean_median_mode_l1871_187188


namespace voting_total_participation_l1871_187133

theorem voting_total_participation:
  ∀ (x : ℝ),
  0.35 * x + 0.65 * x = x ∧
  0.65 * x = 0.45 * (x + 80) →
  (x + 80 = 260) :=
by
  intros x h
  sorry

end voting_total_participation_l1871_187133


namespace rational_combination_zero_eqn_l1871_187162

theorem rational_combination_zero_eqn (a b c : ℚ) (h : a + b * Real.sqrt 32 + c * Real.sqrt 34 = 0) :
  a = 0 ∧ b = 0 ∧ c = 0 :=
by
  sorry

end rational_combination_zero_eqn_l1871_187162


namespace solution_in_quadrant_I_l1871_187149

theorem solution_in_quadrant_I (k x y : ℝ) (h1 : 2 * x - y = 5) (h2 : k * x^2 + y = 4) (h4 : x > 0) (h5 : y > 0) : k > 0 :=
sorry

end solution_in_quadrant_I_l1871_187149


namespace determine_values_l1871_187153

-- Define variables and conditions
variable {x v w y z : ℕ}

-- Define the conditions
def condition1 := v * x = 8 * 9
def condition2 := y^2 = x^2 + 81
def condition3 := z^2 = 20^2 - x^2
def condition4 := w^2 = 8^2 + v^2
def condition5 := v * 20 = y * 8

-- Theorem to prove
theorem determine_values : 
  x = 12 ∧ y = 15 ∧ z = 16 ∧ v = 6 ∧ w = 10 :=
by
  -- Insert necessary logic or 
  -- produce proof steps here
  sorry

end determine_values_l1871_187153


namespace sector_area_correct_l1871_187126

noncomputable def sector_area (r α : ℝ) : ℝ :=
  (1 / 2) * r^2 * α

theorem sector_area_correct :
  sector_area 3 2 = 9 :=
by
  sorry

end sector_area_correct_l1871_187126


namespace loan_amount_needed_l1871_187196

-- Define the total cost of tuition.
def total_tuition : ℝ := 30000

-- Define the amount Sabina has saved.
def savings : ℝ := 10000

-- Define the grant coverage rate.
def grant_coverage_rate : ℝ := 0.4

-- Define the remainder of the tuition after using savings.
def remaining_tuition : ℝ := total_tuition - savings

-- Define the amount covered by the grant.
def grant_amount : ℝ := grant_coverage_rate * remaining_tuition

-- Define the loan amount Sabina needs to apply for.
noncomputable def loan_amount : ℝ := remaining_tuition - grant_amount

-- State the theorem to prove the loan amount needed.
theorem loan_amount_needed : loan_amount = 12000 := by
  sorry

end loan_amount_needed_l1871_187196


namespace pieces_eaten_first_night_l1871_187187

-- Define the initial numbers of candies
def debby_candies : Nat := 32
def sister_candies : Nat := 42
def candies_left : Nat := 39

-- Calculate the initial total number of candies
def initial_total_candies : Nat := debby_candies + sister_candies

-- Define the number of candies eaten the first night
def candies_eaten : Nat := initial_total_candies - candies_left

-- The problem statement with the proof goal
theorem pieces_eaten_first_night : candies_eaten = 35 := by
  sorry

end pieces_eaten_first_night_l1871_187187


namespace lunch_break_duration_l1871_187121

theorem lunch_break_duration
  (p h L : ℝ)
  (monday_eq : (9 - L) * (p + h) = 0.4)
  (tuesday_eq : (8 - L) * h = 0.33)
  (wednesday_eq : (12 - L) * p = 0.27) :
  L = 7.0 ∨ L * 60 = 420 :=
by
  sorry

end lunch_break_duration_l1871_187121


namespace num_real_roots_l1871_187199

theorem num_real_roots (f : ℝ → ℝ)
  (h_eq : ∀ x, f x = 2 * x ^ 3 - 6 * x ^ 2 + 7)
  (h_interval : ∀ x, 0 < x ∧ x < 2 → f x < 0 ∧ f (2 - x) > 0) : 
  ∃! x, 0 < x ∧ x < 2 ∧ f x = 0 :=
sorry

end num_real_roots_l1871_187199


namespace side_length_square_l1871_187154

theorem side_length_square (x : ℝ) (h1 : x^2 = 2 * (4 * x)) : x = 8 :=
by
  sorry

end side_length_square_l1871_187154


namespace cos_C_in_triangle_l1871_187163

theorem cos_C_in_triangle (A B C : ℝ) (h_triangle : A + B + C = π)
  (h_sinA : Real.sin A = 4/5) (h_cosB : Real.cos B = 3/5) :
  Real.cos C = 7/25 := 
sorry

end cos_C_in_triangle_l1871_187163


namespace magnified_diameter_l1871_187132

theorem magnified_diameter (diameter_actual : ℝ) (magnification_factor : ℕ) 
  (h_actual : diameter_actual = 0.005) (h_magnification : magnification_factor = 1000) :
  diameter_actual * magnification_factor = 5 :=
by 
  sorry

end magnified_diameter_l1871_187132


namespace example_3_is_analogical_reasoning_l1871_187177

-- Definitions based on the conditions of the problem:
def is_analogical_reasoning (reasoning: String): Prop :=
  reasoning = "from one specific case to another similar specific case"

-- Example of reasoning given in the problem.
def example_3 := "From the fact that the sum of the distances from a point inside an equilateral triangle to its three sides is a constant, it is concluded that the sum of the distances from a point inside a regular tetrahedron to its four faces is a constant."

-- Proof statement based on the conditions and correct answer.
theorem example_3_is_analogical_reasoning: is_analogical_reasoning example_3 :=
by 
  sorry

end example_3_is_analogical_reasoning_l1871_187177


namespace domain_of_f_l1871_187189

noncomputable def f (x : ℝ) : ℝ := (2 * x + 3) / Real.sqrt (x^2 - 4)

theorem domain_of_f :
  {x : ℝ | x^2 - 4 >= 0 ∧ x^2 - 4 ≠ 0} = {x : ℝ | x ≤ -2 ∨ x ≥ 2} :=
by
  sorry

end domain_of_f_l1871_187189


namespace number_of_roots_l1871_187118

noncomputable def f (a b c x : ℝ) : ℝ := x^3 + 2 * a * x^2 + 2 * b * x + 3 * c

theorem number_of_roots (a b c x₁ x₂ : ℝ) (h_extreme : x₁ ≠ x₂)
    (h_fx1 : f a b c x₁ = x₁) :
    (∃ (r : ℝ), 3 * (f a b c r)^2 + 4 * a * (f a b c r) + 2 * b = 0) :=
sorry

end number_of_roots_l1871_187118


namespace replaced_person_weight_l1871_187160

theorem replaced_person_weight :
  ∀ (avg_weight: ℝ), 
    10 * (avg_weight + 4) - 10 * avg_weight = 110 - 70 :=
by
  intros avg_weight
  sorry

end replaced_person_weight_l1871_187160


namespace infimum_of_function_l1871_187134

open Real

-- Definitions given in the conditions:
def even_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = f x
def periodic_function (f : ℝ → ℝ) := ∀ x : ℝ, f (1 - x) = f (1 + x)
def function_on_interval (f : ℝ → ℝ) := ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = -3 * x ^ 2 + 2

-- Proof problem statement:
theorem infimum_of_function (f : ℝ → ℝ) 
  (h_even : even_function f) 
  (h_periodic : periodic_function f) 
  (h_interval : function_on_interval f) : 
  ∃ M : ℝ, (∀ x : ℝ, f x ≥ M) ∧ M = -1 :=
by
  sorry

end infimum_of_function_l1871_187134


namespace inequality_for_positive_numbers_l1871_187125

theorem inequality_for_positive_numbers (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) : 
  (a + b) * (a^4 + b^4) ≥ (a^2 + b^2) * (a^3 + b^3) :=
sorry

end inequality_for_positive_numbers_l1871_187125


namespace demand_decrease_annual_l1871_187127

noncomputable def price_increase (P : ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  P * (1 + r / 100) ^ t

noncomputable def demand_maintenance (P : ℝ) (r : ℝ) (t : ℕ) (d : ℝ) : Prop :=
  let new_price := price_increase P r t
  (P * (1 + r / 100)) * (1 - d / 100) ≥ price_increase P 10 1

theorem demand_decrease_annual (P : ℝ) (r : ℝ) (t : ℕ) :
  price_increase P r t ≥ price_increase P 10 1 → ∃ d : ℝ, d = 1.66156 :=
by
  sorry

end demand_decrease_annual_l1871_187127


namespace range_of_k_roots_for_neg_k_l1871_187129

theorem range_of_k (k : ℝ) : (∃ x y : ℝ, x ≠ y ∧ (x^2 + (2*k + 1)*x + (k^2 - 1) = 0 ∧ y^2 + (2*k + 1)*y + (k^2 - 1) = 0)) ↔ k > -5 / 4 :=
by sorry

theorem roots_for_neg_k (k : ℤ) (h1 : k < 0) (h2 : k > -5 / 4) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 + (2*k + 1)*x1 + (k^2 - 1) = 0 ∧ x2^2 + (2*k + 1)*x2 + (k^2 - 1) = 0 ∧ x1 = 0 ∧ x2 = 1)) :=
by sorry

end range_of_k_roots_for_neg_k_l1871_187129


namespace man_work_alone_in_5_days_l1871_187192

theorem man_work_alone_in_5_days (d : ℕ) (h1 : ∀ m : ℕ, (1 / (m : ℝ)) + 1 / 20 = 1 / 4):
  d = 5 := by
  sorry

end man_work_alone_in_5_days_l1871_187192


namespace smallest_square_side_length_l1871_187165

theorem smallest_square_side_length :
  ∃ (n s : ℕ),  14 * n = s^2 ∧ s = 14 := 
by
  existsi 14, 14
  sorry

end smallest_square_side_length_l1871_187165


namespace consecutive_integer_sets_sum_100_l1871_187152

theorem consecutive_integer_sets_sum_100 :
  ∃ s : Finset (Finset ℕ), 
    (∀ seq ∈ s, (∀ x ∈ seq, x > 0) ∧ (seq.sum id = 100)) ∧
    (s.card = 2) :=
sorry

end consecutive_integer_sets_sum_100_l1871_187152


namespace avg_wx_half_l1871_187124

noncomputable def avg_wx {w x y : ℝ} (h1 : 5 / w + 5 / x = 5 / y) (h2 : w * x = y) : ℝ :=
(w + x) / 2

theorem avg_wx_half {w x y : ℝ} (h1 : 5 / w + 5 / x = 5 / y) (h2 : w * x = y) :
  avg_wx h1 h2 = 1 / 2 :=
sorry

end avg_wx_half_l1871_187124


namespace factorize_expression_l1871_187103

theorem factorize_expression (m : ℝ) : 3 * m^2 - 12 = 3 * (m + 2) * (m - 2) := 
sorry

end factorize_expression_l1871_187103


namespace Vasya_can_win_l1871_187158

-- We need this library to avoid any import issues and provide necessary functionality for rational numbers

theorem Vasya_can_win :
  let a := (1 : ℚ) / 2009
  let b := (1 : ℚ) / 2008
  (∃ x : ℚ, a + x = 1) ∨ (∃ x : ℚ, b + x = 1) := sorry

end Vasya_can_win_l1871_187158


namespace route_y_slower_by_2_4_minutes_l1871_187171
noncomputable def time_route_x : ℝ := (7 : ℝ) / (35 : ℝ)
noncomputable def time_downtown_y : ℝ := (1 : ℝ) / (10 : ℝ)
noncomputable def time_other_y : ℝ := (7 : ℝ) / (50 : ℝ)
noncomputable def time_route_y : ℝ := time_downtown_y + time_other_y

theorem route_y_slower_by_2_4_minutes :
  ((time_route_y - time_route_x) * 60) = 2.4 :=
by
  -- Provide the required proof here
  sorry

end route_y_slower_by_2_4_minutes_l1871_187171


namespace radius_of_circumscribed_sphere_l1871_187135

noncomputable def circumscribed_sphere_radius (a : ℝ) : ℝ :=
  a / Real.sqrt 3

theorem radius_of_circumscribed_sphere 
  (a : ℝ) 
  (h_base_side : 0 < a)
  (h_distance : ∃ d : ℝ, d = a * Real.sqrt 2 / 8) : 
  circumscribed_sphere_radius a = a / Real.sqrt 3 :=
sorry

end radius_of_circumscribed_sphere_l1871_187135


namespace binom_1293_1_eq_1293_l1871_187172

theorem binom_1293_1_eq_1293 : (Nat.choose 1293 1) = 1293 := 
  sorry

end binom_1293_1_eq_1293_l1871_187172


namespace max_area_100_max_fence_length_l1871_187116

noncomputable def maximum_allowable_area (x y : ℝ) : Prop :=
  40 * x + 2 * 45 * y + 20 * x * y ≤ 3200

theorem max_area_100 (x y S : ℝ) (h : maximum_allowable_area x y) :
  S <= 100 :=
sorry

theorem max_fence_length (x y : ℝ) (h : maximum_allowable_area x y) (h1 : x * y = 100) :
  x = 15 :=
sorry

end max_area_100_max_fence_length_l1871_187116


namespace pyramid_volume_is_sqrt3_l1871_187139

noncomputable def volume_of_pyramid := 
  let base_area : ℝ := 2 * Real.sqrt 3
  let angle_ABC : ℝ := 60
  let BC := 2
  let EC := BC
  let FB := BC / 2
  let height : ℝ := Real.sqrt 3
  let pyramid_volume := 1/3 * EC * FB * height
  pyramid_volume

theorem pyramid_volume_is_sqrt3 : volume_of_pyramid = Real.sqrt 3 :=
by sorry

end pyramid_volume_is_sqrt3_l1871_187139


namespace regression_value_l1871_187113

theorem regression_value (x : ℝ) (y : ℝ) (h : y = 4.75 * x + 2.57) (hx : x = 28) : y = 135.57 :=
by
  sorry

end regression_value_l1871_187113


namespace problem_inequality_l1871_187167

theorem problem_inequality (a b c d : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) (h5 : a * b * c * d = 1) : 
  a^2 + b^2 + c^2 + d^2 + a * b + a * c + a * d + b * c + b * d + c * d ≥ 10 := 
by 
  sorry

end problem_inequality_l1871_187167


namespace box_filled_with_cubes_no_leftover_l1871_187144

-- Define dimensions of the box
def box_length : ℝ := 50
def box_width : ℝ := 60
def box_depth : ℝ := 43

-- Define volumes of different types of cubes
def volume_box : ℝ := box_length * box_width * box_depth
def volume_small_cube : ℝ := 2^3
def volume_medium_cube : ℝ := 3^3
def volume_large_cube : ℝ := 5^3

-- Define the smallest number of each type of cube
def num_large_cubes : ℕ := 1032
def num_medium_cubes : ℕ := 0
def num_small_cubes : ℕ := 0

-- Theorem statement ensuring the number of cubes completely fills the box
theorem box_filled_with_cubes_no_leftover :
  num_large_cubes * volume_large_cube + num_medium_cubes * volume_medium_cube + num_small_cubes * volume_small_cube = volume_box :=
by
  sorry

end box_filled_with_cubes_no_leftover_l1871_187144


namespace length_of_one_side_of_regular_octagon_l1871_187115

-- Define the conditions of the problem
def is_regular_octagon (n : ℕ) (P : ℝ) (length_of_side : ℝ) : Prop :=
  n = 8 ∧ P = 72 ∧ length_of_side = P / n

-- State the theorem
theorem length_of_one_side_of_regular_octagon : is_regular_octagon 8 72 9 :=
by
  -- The proof is omitted; only the statement is required
  sorry

end length_of_one_side_of_regular_octagon_l1871_187115


namespace prob_event_A_given_B_l1871_187130

def EventA (visits : Fin 4 → Fin 4) : Prop :=
  Function.Injective visits

def EventB (visits : Fin 4 → Fin 4) : Prop :=
  visits 0 = 0

theorem prob_event_A_given_B :
  ∀ (visits : Fin 4 → Fin 4),
  (∃ f : (Fin 4 → Fin 4) → Prop, f visits → (EventA visits ∧ EventB visits)) →
  (∃ P : ℚ, P = 2 / 9) :=
by
  intros visits h
  -- Proof omitted
  sorry

end prob_event_A_given_B_l1871_187130


namespace find_a2_plus_b2_l1871_187191

theorem find_a2_plus_b2 (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h: 8 * a^a * b^b = 27 * a^b * b^a) : a^2 + b^2 = 117 := by
  sorry

end find_a2_plus_b2_l1871_187191


namespace second_train_length_l1871_187114

noncomputable def length_of_second_train (speed1_kmph speed2_kmph time_sec length1_m : ℝ) : ℝ :=
  let relative_speed_mps := (speed1_kmph + speed2_kmph) * (1000 / 3600)
  let total_distance := relative_speed_mps * time_sec
  total_distance - length1_m

theorem second_train_length :
  length_of_second_train 60 48 9.99920006399488 140 = 159.9760019198464 :=
by
  sorry

end second_train_length_l1871_187114


namespace asimov_books_l1871_187123

theorem asimov_books (h p : Nat) (condition1 : h + p = 12) (condition2 : 30 * h + 20 * p = 300) : h = 6 := by
  sorry

end asimov_books_l1871_187123


namespace race_result_l1871_187146

-- Define the contestants
inductive Contestants
| Alyosha
| Borya
| Vanya
| Grisha

open Contestants

-- Define their statements
def Alyosha_statement (place : Contestants → ℕ) : Prop :=
  place Alyosha ≠ 1 ∧ place Alyosha ≠ 4

def Borya_statement (place : Contestants → ℕ) : Prop :=
  place Borya ≠ 4

def Vanya_statement (place : Contestants → ℕ) : Prop :=
  place Vanya = 1

def Grisha_statement (place : Contestants → ℕ) : Prop :=
  place Grisha = 4

-- Define that exactly one statement is false and the rest are true
def three_true_one_false (place : Contestants → ℕ) : Prop :=
  (Alyosha_statement place ∧ ¬ Vanya_statement place ∧ Borya_statement place ∧ Grisha_statement place) ∨
  (¬ Alyosha_statement place ∧ Vanya_statement place ∧ Borya_statement place ∧ Grisha_statement place) ∨
  (Alyosha_statement place ∧ Vanya_statement place ∧ ¬ Borya_statement place ∧ Grisha_statement place) ∨
  (Alyosha_statement place ∧ Vanya_statement place ∧ Borya_statement place ∧ ¬ Grisha_statement place)

-- Define the conclusion: Vanya lied and Borya was first
theorem race_result (place : Contestants → ℕ) : 
  three_true_one_false place → 
  (¬ Vanya_statement place ∧ place Borya = 1) :=
sorry

end race_result_l1871_187146


namespace x_not_4_17_percent_less_than_z_x_is_8_0032_percent_less_than_z_l1871_187101

def y_is_60_percent_greater_than_x (x y : ℝ) : Prop :=
  y = 1.60 * x

def z_is_40_percent_less_than_y (y z : ℝ) : Prop :=
  z = 0.60 * y

theorem x_not_4_17_percent_less_than_z (x y z : ℝ) (h1 : y_is_60_percent_greater_than_x x y) (h2 : z_is_40_percent_less_than_y y z) : 
  x ≠ 0.9583 * z :=
by {
  sorry
}

theorem x_is_8_0032_percent_less_than_z (x y z : ℝ) (h1 : y_is_60_percent_greater_than_x x y) (h2 : z_is_40_percent_less_than_y y z) : 
  x = 0.919968 * z :=
by {
  sorry
}

end x_not_4_17_percent_less_than_z_x_is_8_0032_percent_less_than_z_l1871_187101


namespace number_of_5_letter_words_with_at_least_one_consonant_equals_7744_l1871_187119

def is_consonant (c : Char) : Prop :=
  c = 'B' ∨ c = 'C' ∨ c = 'D' ∨ c = 'F'

def count_5_letter_words_with_at_least_one_consonant : Nat :=
  let total_words := 6 ^ 5
  let vowel_words := 2 ^ 5
  total_words - vowel_words

theorem number_of_5_letter_words_with_at_least_one_consonant_equals_7744 :
  count_5_letter_words_with_at_least_one_consonant = 7744 :=
by
  sorry

end number_of_5_letter_words_with_at_least_one_consonant_equals_7744_l1871_187119


namespace probability_abs_diff_l1871_187108

variables (P : ℕ → ℚ) (m : ℚ)

def is_probability_distribution : Prop :=
  P 1 = m ∧ P 2 = 1/4 ∧ P 3 = 1/4 ∧ P 4 = 1/3 ∧ m + 1/4 + 1/4 + 1/3 = 1

theorem probability_abs_diff (h : is_probability_distribution P m) :
  P 1 + P 3 = 5 / 12 :=
by 
sorry

end probability_abs_diff_l1871_187108


namespace eve_stamp_collection_worth_l1871_187176

def total_value_of_collection (stamps_value : ℕ) (num_stamps : ℕ) (set_size : ℕ) (set_value : ℕ) (bonus_per_set : ℕ) : ℕ :=
  let value_per_stamp := set_value / set_size
  let total_value := value_per_stamp * num_stamps
  let num_complete_sets := num_stamps / set_size
  let total_bonus := num_complete_sets * bonus_per_set
  total_value + total_bonus

theorem eve_stamp_collection_worth :
  total_value_of_collection 21 21 7 28 5 = 99 := by
  rfl

end eve_stamp_collection_worth_l1871_187176


namespace same_terminal_side_l1871_187197

theorem same_terminal_side : ∃ k : ℤ, 36 + k * 360 = -324 :=
by
  use -1
  linarith

end same_terminal_side_l1871_187197


namespace largest_possible_b_l1871_187194

theorem largest_possible_b (a b c : ℕ) (h₁ : 1 < c) (h₂ : c < b) (h₃ : b < a) (h₄ : a * b * c = 360): b = 12 :=
by
  sorry

end largest_possible_b_l1871_187194


namespace comprehensive_survey_option_l1871_187169

def suitable_for_comprehensive_survey (survey : String) : Prop :=
  survey = "Survey on the components of the first large civil helicopter in China"

theorem comprehensive_survey_option (A B C D : String)
  (hA : A = "Survey on the number of waste batteries discarded in the city every day")
  (hB : B = "Survey on the quality of ice cream in the cold drink market")
  (hC : C = "Survey on the current mental health status of middle school students nationwide")
  (hD : D = "Survey on the components of the first large civil helicopter in China") :
  suitable_for_comprehensive_survey D :=
by
  sorry

end comprehensive_survey_option_l1871_187169


namespace additional_people_to_halve_speed_l1871_187166

variables (s : ℕ → ℝ)
variables (x : ℕ)

-- Given conditions
axiom speed_with_200_people : s 200 = 500
axiom speed_with_400_people : s 400 = 125
axiom speed_halved : ∀ n, s (n + x) = s n / 2

theorem additional_people_to_halve_speed : x = 100 :=
by
  sorry

end additional_people_to_halve_speed_l1871_187166


namespace intersection_sum_l1871_187128

-- Define the conditions
def condition_1 (k : ℝ) := k > 0
def line1 (x y k : ℝ) := 50 * x + k * y = 1240
def line2 (x y k : ℝ) := k * y = 8 * x + 544
def right_angles (k : ℝ) := (-50 / k) * (8 / k) = -1

-- Define the point of intersection
def point_of_intersection (m n : ℝ) (k : ℝ) := line1 m n k ∧ line2 m n k

-- Prove that m + n = 44 under the given conditions
theorem intersection_sum (m n k : ℝ) :
  condition_1 k →
  right_angles k →
  point_of_intersection m n k →
  m + n = 44 :=
by
  sorry

end intersection_sum_l1871_187128


namespace daily_rental_cost_l1871_187111

theorem daily_rental_cost (x : ℝ) (total_cost miles : ℝ)
  (cost_per_mile : ℝ) (daily_cost : ℝ) :
  total_cost = daily_cost + cost_per_mile * miles →
  total_cost = 46.12 →
  miles = 214 →
  cost_per_mile = 0.08 →
  daily_cost = 29 :=
by
  sorry

end daily_rental_cost_l1871_187111


namespace range_of_a_l1871_187150

theorem range_of_a(p q: Prop)
  (hp: p ↔ (a = 0 ∨ (0 < a ∧ a < 4)))
  (hq: q ↔ (-1 < a ∧ a < 3))
  (hpor: p ∨ q)
  (hpand: ¬(p ∧ q)):
  (-1 < a ∧ a < 0) ∨ (3 ≤ a ∧ a < 4) := by sorry

end range_of_a_l1871_187150


namespace area_between_circles_l1871_187106

noncomputable def k_value (θ : ℝ) : ℝ := Real.tan θ

theorem area_between_circles {θ k : ℝ} (h₁ : k = Real.tan θ) (h₂ : θ = 4/3) (h_area : (3 * θ / 2) = 2) :
  k = Real.tan (4/3) :=
sorry

end area_between_circles_l1871_187106


namespace terminal_side_quadrant_l1871_187147

-- Given conditions
variables {α : ℝ}
variable (h1 : Real.sin α > 0)
variable (h2 : Real.tan α < 0)

-- Conclusion to be proved
theorem terminal_side_quadrant (h1 : Real.sin α > 0) (h2 : Real.tan α < 0) : 
  (∃ k : ℤ, (k % 2 = 0 ∧ Real.pi * k / 2 < α / 2 ∧ α / 2 < Real.pi / 2 + Real.pi * k) ∨ 
            (k % 2 = 1 ∧ Real.pi * (k - 1) < α / 2 ∧ α / 2 < Real.pi / 4 + Real.pi * (k - 0.5))) :=
by
  sorry

end terminal_side_quadrant_l1871_187147


namespace f_2017_eq_one_l1871_187117

noncomputable def f (x : ℝ) (a : ℝ) (α : ℝ) (b : ℝ) (β : ℝ) : ℝ :=
  a * Real.sin (Real.pi * x + α) + b * Real.cos (Real.pi * x - β)

-- Given conditions
variables {a b α β : ℝ}
variable (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ α ≠ 0 ∧ β ≠ 0)
variable (h_f2016 : f 2016 a α b β = -1)

-- The goal
theorem f_2017_eq_one : f 2017 a α b β = 1 :=
sorry

end f_2017_eq_one_l1871_187117


namespace max_concentration_at_2_l1871_187180

noncomputable def concentration (t : ℝ) : ℝ := (20 * t) / (t^2 + 4)

theorem max_concentration_at_2 : ∃ t : ℝ, 0 ≤ t ∧ ∀ s : ℝ, (0 ≤ s → concentration s ≤ concentration t) ∧ t = 2 := 
by 
  sorry -- we add sorry to skip the actual proof

end max_concentration_at_2_l1871_187180


namespace geometric_sequence_a3_l1871_187131

theorem geometric_sequence_a3 :
  ∀ (a : ℕ → ℝ), a 1 = 2 → a 5 = 8 → (a 3 = 4 ∨ a 3 = -4) :=
by
  intros a h₁ h₅
  sorry

end geometric_sequence_a3_l1871_187131


namespace same_roots_condition_l1871_187136

-- Definition of quadratic equations with coefficients a1, b1, c1 and a2, b2, c2
variables (a1 b1 c1 a2 b2 c2 : ℝ)

-- The condition we need to prove
theorem same_roots_condition :
  (a1 ≠ 0 ∧ a2 ≠ 0) → 
  (a1 / a2 = b1 / b2 ∧ b1 / b2 = c1 / c2) 
    ↔ 
  ∀ x : ℝ, (a1 * x^2 + b1 * x + c1 = 0 ↔ a2 * x^2 + b2 * x + c2 = 0) :=
sorry

end same_roots_condition_l1871_187136


namespace boys_collected_200_insects_l1871_187156

theorem boys_collected_200_insects
  (girls_insects : ℕ)
  (groups : ℕ)
  (insects_per_group : ℕ)
  (total_insects : ℕ)
  (boys_insects : ℕ)
  (H1 : girls_insects = 300)
  (H2 : groups = 4)
  (H3 : insects_per_group = 125)
  (H4 : total_insects = groups * insects_per_group)
  (H5 : boys_insects = total_insects - girls_insects) :
  boys_insects = 200 :=
  by sorry

end boys_collected_200_insects_l1871_187156


namespace equation_solution_count_l1871_187120

open Real

theorem equation_solution_count :
  ∃ s : Finset ℝ, (∀ x ∈ s, 0 ≤ x ∧ x ≤ 2 * π ∧ sin (π / 4 * sin x) = cos (π / 4 * cos x)) ∧ s.card = 4 :=
by
  sorry

end equation_solution_count_l1871_187120


namespace angles_sum_132_l1871_187105

theorem angles_sum_132
  (D E F p q : ℝ)
  (hD : D = 38)
  (hE : E = 58)
  (hF : F = 36)
  (five_sided_angle_sum : D + E + (360 - p) + 90 + (126 - q) = 540) : 
  p + q = 132 := 
by
  sorry

end angles_sum_132_l1871_187105


namespace volume_in_30_minutes_l1871_187173

-- Define the conditions
def rate_of_pumping := 540 -- gallons per hour
def time_in_hours := 30 / 60 -- 30 minutes as a fraction of an hour

-- Define the volume pumped in 30 minutes
def volume_pumped := rate_of_pumping * time_in_hours

-- State the theorem
theorem volume_in_30_minutes : volume_pumped = 270 := by
  sorry

end volume_in_30_minutes_l1871_187173


namespace increasing_interval_of_f_l1871_187185

noncomputable def f (x : ℝ) : ℝ := x^2 - 6 * x

theorem increasing_interval_of_f :
  ∀ x : ℝ, 3 ≤ x → ∀ y : ℝ, 3 ≤ y → x < y → f x < f y := 
sorry

end increasing_interval_of_f_l1871_187185


namespace polynomial_has_at_most_one_integer_root_l1871_187151

theorem polynomial_has_at_most_one_integer_root (k : ℝ) :
  ∀ x y : ℤ, (x^3 - 24 * x + k = 0) ∧ (y^3 - 24 * y + k = 0) → x = y :=
by
  intros x y h
  sorry

end polynomial_has_at_most_one_integer_root_l1871_187151


namespace div_expression_calc_l1871_187112

theorem div_expression_calc :
  (3752 / (39 * 2) + 5030 / (39 * 10) = 61) :=
by
  sorry -- Proof of the theorem

end div_expression_calc_l1871_187112


namespace initial_floors_l1871_187122

-- Define the conditions given in the problem
def austin_time := 60 -- Time Austin takes in seconds to reach the ground floor
def jake_time := 90 -- Time Jake takes in seconds to reach the ground floor
def jake_steps_per_sec := 3 -- Jake descends 3 steps per second
def steps_per_floor := 30 -- There are 30 steps per floor

-- Define the total number of steps Jake descends
def total_jake_steps := jake_time * jake_steps_per_sec

-- Define the number of floors descended in terms of total steps and steps per floor
def num_floors := total_jake_steps / steps_per_floor

-- Theorem stating the number of floors is 9
theorem initial_floors : num_floors = 9 :=
by 
  -- Provide the basic proof structure
  sorry

end initial_floors_l1871_187122


namespace sum_of_abs_squared_series_correct_l1871_187181

noncomputable def sum_of_abs_squared_series (a r : ℝ) (h : |r| < 1) : ℝ :=
  a^2 / (1 - |r|^2)

theorem sum_of_abs_squared_series_correct (a r : ℝ) (h : |r| < 1) :
  sum_of_abs_squared_series a r h = a^2 / (1 - |r|^2) :=
by
  sorry

end sum_of_abs_squared_series_correct_l1871_187181


namespace necessary_and_sufficient_for_Sn_lt_an_l1871_187164

def arithmetic_seq (a : ℕ → ℤ) (d : ℤ) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
(n + 1) * a 0 + (n * (n + 1)) / 2

theorem necessary_and_sufficient_for_Sn_lt_an
  (a : ℕ → ℤ) (d : ℤ) (S : ℕ → ℤ)
  (h_arith_seq : arithmetic_seq a d)
  (h_d_neg : d < 0)
  (m n : ℕ)
  (h_pos_m : m ≥ 3)
  (h_am_eq_Sm : a m = S m) :
  n > m ↔ S n < a n := sorry

end necessary_and_sufficient_for_Sn_lt_an_l1871_187164


namespace henry_jill_age_ratio_l1871_187137

theorem henry_jill_age_ratio :
  ∀ (H J : ℕ), (H + J = 48) → (H = 29) → (J = 19) → ((H - 9) / (J - 9) = 2) :=
by
  intros H J h_sum h_henry h_jill
  sorry

end henry_jill_age_ratio_l1871_187137


namespace original_quantity_ghee_mixture_is_correct_l1871_187138

-- Define the variables
def percentage_ghee (x : ℝ) := 0.55 * x
def percentage_vanasapati (x : ℝ) := 0.35 * x
def percentage_palm_oil (x : ℝ) := 0.10 * x
def new_mixture_weight (x : ℝ) := x + 20
def final_vanasapati_percentage (x : ℝ) := 0.30 * (new_mixture_weight x)

-- State the theorem
theorem original_quantity_ghee_mixture_is_correct (x : ℝ) 
  (h1 : percentage_ghee x = 0.55 * x)
  (h2 : percentage_vanasapati x = 0.35 * x)
  (h3 : percentage_palm_oil x = 0.10 * x)
  (h4 : percentage_vanasapati x = final_vanasapati_percentage x) :
  x = 120 := 
sorry

end original_quantity_ghee_mixture_is_correct_l1871_187138


namespace solution_set_l1871_187183

theorem solution_set (x y : ℝ) : (x - 2 * y = 1) ∧ (x^3 - 6 * x * y - 8 * y^3 = 1) ↔ y = (x - 1) / 2 :=
by
  sorry

end solution_set_l1871_187183


namespace drop_perpendicular_l1871_187168

open Classical

-- Definitions for geometrical constructions on the plane
structure Point :=
(x : ℝ)
(y : ℝ)

structure Line :=
(p1 : Point)
(p2 : Point)

-- Condition 1: Drawing a line through two points
def draw_line (A B : Point) : Line := {
  p1 := A,
  p2 := B
}

-- Condition 2: Drawing a perpendicular line through a given point on a line
def draw_perpendicular (l : Line) (P : Point) : Line :=
-- Details of construction skipped, this function should return the perpendicular line
sorry

-- The problem: Given a point A and a line l not passing through A, construct the perpendicular from A to l
theorem drop_perpendicular : 
  ∀ (A : Point) (l : Line), ¬ (A = l.p1 ∨ A = l.p2) → ∃ (P : Point), ∃ (m : Line), (m = draw_perpendicular l P) ∧ (m.p1 = A) :=
by
  intros A l h
  -- Details of theorem-proof skipped, assert the existence of P and m as required
  sorry

end drop_perpendicular_l1871_187168


namespace find_A_plus_C_l1871_187141

-- This will bring in the entirety of the necessary library and supports the digit verification and operations.

-- Definitions of digits and constraints
variables {A B C D : ℕ}

-- Given conditions in the problem
def distinct_digits (A B C D : ℕ) : Prop :=
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧ 
  A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10

def multiplication_condition_1 (A B C D : ℕ) : Prop :=
  C * D = A

def multiplication_condition_2 (A B C D : ℕ) : Prop :=
  10 * B * D + C * D = 11 * C

-- The final problem statement
theorem find_A_plus_C (A B C D : ℕ) (h1 : distinct_digits A B C D) 
  (h2 : multiplication_condition_1 A B C D) 
  (h3 : multiplication_condition_2 A B C D) : 
  A + C = 10 :=
sorry

end find_A_plus_C_l1871_187141


namespace find_d_l1871_187148

def point_in_square (x y : ℝ) : Prop :=
  0 ≤ x ∧ x ≤ 3030 ∧ 0 ≤ y ∧ y ≤ 3030

def point_in_ellipse (x y : ℝ) : Prop :=
  (x^2 / 2020^2) + (y^2 / 4040^2) ≤ 1

def point_within_distance (d : ℝ) (x y : ℝ) : Prop :=
  (∃ (a b : ℤ), (x - a) ^ 2 + (y - b) ^ 2 ≤ d ^ 2)

theorem find_d :
  (∃ d : ℝ, (∀ x y : ℝ, point_in_square x y → point_in_ellipse x y → point_within_distance d x y) ∧ (d = 0.5)) :=
by
  sorry

end find_d_l1871_187148


namespace stock_return_to_original_l1871_187179

theorem stock_return_to_original (x : ℝ) (h : x > 0) :
  ∃ d : ℝ, d = 3 / 13 ∧ (x * 1.30 * (1 - d)) = x :=
by sorry

end stock_return_to_original_l1871_187179


namespace perp_line_slope_zero_l1871_187100

theorem perp_line_slope_zero {k : ℝ} (h : ∀ x : ℝ, ∃ y : ℝ, y = k * x + 1 ∧ x = 1 → false) : k = 0 :=
sorry

end perp_line_slope_zero_l1871_187100


namespace trapezoid_reassembly_area_conservation_l1871_187175

theorem trapezoid_reassembly_area_conservation
  {height length new_width : ℝ}
  (h1 : height = 9)
  (h2 : length = 16)
  (h3 : new_width = y)  -- each base of the trapezoid measures y.
  (div_trapezoids : ∀ (a b c : ℝ), 3 * a = height → a = 9 / 3)
  (area_conserved : length * height = (3 / 2) * (3 * (length + new_width)))
  : new_width = 16 :=
by
  -- The proof is skipped
  sorry

end trapezoid_reassembly_area_conservation_l1871_187175


namespace john_naps_70_days_l1871_187109

def total_naps_in_days (naps_per_week nap_duration days_in_week total_days : ℕ) : ℕ :=
  let total_weeks := total_days / days_in_week
  let total_naps := total_weeks * naps_per_week
  total_naps * nap_duration

theorem john_naps_70_days
  (naps_per_week : ℕ)
  (nap_duration : ℕ)
  (days_in_week : ℕ)
  (total_days : ℕ)
  (h_naps_per_week : naps_per_week = 3)
  (h_nap_duration : nap_duration = 2)
  (h_days_in_week : days_in_week = 7)
  (h_total_days : total_days = 70) :
  total_naps_in_days naps_per_week nap_duration days_in_week total_days = 60 :=
by
  rw [h_naps_per_week, h_nap_duration, h_days_in_week, h_total_days]
  sorry

end john_naps_70_days_l1871_187109


namespace least_positive_int_to_multiple_of_3_l1871_187155

theorem least_positive_int_to_multiple_of_3 (x : ℕ) (h : 575 + x ≡ 0 [MOD 3]) : x = 1 := 
by
  sorry

end least_positive_int_to_multiple_of_3_l1871_187155


namespace sin_angle_add_pi_over_4_l1871_187193

open Real

theorem sin_angle_add_pi_over_4 (α : ℝ) (h1 : (cos α = -3/5) ∧ (sin α = 4/5)) : sin (α + π / 4) = sqrt 2 / 10 :=
by
  sorry

end sin_angle_add_pi_over_4_l1871_187193


namespace kyle_vs_parker_l1871_187184

-- Define the distances thrown by Parker, Grant, and Kyle.
def parker_distance : ℕ := 16
def grant_distance : ℕ := (125 * parker_distance) / 100
def kyle_distance : ℕ := 2 * grant_distance

-- Prove that Kyle threw the ball 24 yards farther than Parker.
theorem kyle_vs_parker : kyle_distance - parker_distance = 24 := 
by
  -- Sorry for proof
  sorry

end kyle_vs_parker_l1871_187184


namespace equivalent_statements_l1871_187161

variables (P Q R : Prop)

theorem equivalent_statements :
  (P → (Q ∧ ¬R)) ↔ ((¬ Q ∨ R) → ¬ P) :=
sorry

end equivalent_statements_l1871_187161


namespace total_circles_l1871_187102

theorem total_circles (n : ℕ) (h1 : ∀ k : ℕ, k = n + 14 → n^2 = (k * (k + 1) / 2)) : 
  n = 35 → n^2 = 1225 :=
by
  sorry

end total_circles_l1871_187102


namespace total_sales_correct_l1871_187104

def normal_sales_per_month : ℕ := 21122
def additional_sales_in_june : ℕ := 3922
def sales_in_june : ℕ := normal_sales_per_month + additional_sales_in_june
def sales_in_july : ℕ := normal_sales_per_month
def total_sales : ℕ := sales_in_june + sales_in_july

theorem total_sales_correct :
  total_sales = 46166 :=
by
  -- Proof goes here
  sorry

end total_sales_correct_l1871_187104


namespace initial_overs_l1871_187178

variable (x : ℝ)

/-- 
Proof that the number of initial overs x is 10, given the conditions:
1. The run rate in the initial x overs was 3.2 runs per over.
2. The run rate in the remaining 50 overs was 5 runs per over.
3. The total target is 282 runs.
4. The runs scored in the remaining 50 overs should be 250 runs.
-/
theorem initial_overs (hx : 3.2 * x + 250 = 282) : x = 10 :=
sorry

end initial_overs_l1871_187178
