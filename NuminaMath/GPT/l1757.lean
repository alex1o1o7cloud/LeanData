import Mathlib

namespace NUMINAMATH_GPT_chi_square_association_l1757_175797

theorem chi_square_association (k : ℝ) :
  (k > 3.841 → (∃ A B, A ∧ B)) ∧ (k ≤ 2.076 → (∃ A B, ¬(A ∧ B))) :=
by
  sorry

end NUMINAMATH_GPT_chi_square_association_l1757_175797


namespace NUMINAMATH_GPT_find_C_monthly_income_l1757_175747

theorem find_C_monthly_income (A_m B_m C_m : ℝ) (h1 : A_m / B_m = 5 / 2) (h2 : B_m = 1.12 * C_m) (h3 : 12 * A_m = 504000) : C_m = 15000 :=
sorry

end NUMINAMATH_GPT_find_C_monthly_income_l1757_175747


namespace NUMINAMATH_GPT_freddy_total_call_cost_l1757_175798

def lm : ℕ := 45
def im : ℕ := 31
def lc : ℝ := 0.05
def ic : ℝ := 0.25

theorem freddy_total_call_cost : lm * lc + im * ic = 10.00 := by
  sorry

end NUMINAMATH_GPT_freddy_total_call_cost_l1757_175798


namespace NUMINAMATH_GPT_discount_percentage_correct_l1757_175717

-- Define the problem parameters as variables
variables (sale_price marked_price : ℝ) (discount_percentage : ℝ)

-- Provide the conditions from the problem
def conditions : Prop :=
  sale_price = 147.60 ∧ marked_price = 180

-- State the problem: Prove the discount percentage is 18%
theorem discount_percentage_correct (h : conditions sale_price marked_price) : 
  discount_percentage = 18 :=
by
  sorry

end NUMINAMATH_GPT_discount_percentage_correct_l1757_175717


namespace NUMINAMATH_GPT_highest_total_zits_l1757_175753

def zits_per_student_Swanson := 5
def students_Swanson := 25
def total_zits_Swanson := zits_per_student_Swanson * students_Swanson -- should be 125

def zits_per_student_Jones := 6
def students_Jones := 32
def total_zits_Jones := zits_per_student_Jones * students_Jones -- should be 192

def zits_per_student_Smith := 7
def students_Smith := 20
def total_zits_Smith := zits_per_student_Smith * students_Smith -- should be 140

def zits_per_student_Brown := 8
def students_Brown := 16
def total_zits_Brown := zits_per_student_Brown * students_Brown -- should be 128

def zits_per_student_Perez := 4
def students_Perez := 30
def total_zits_Perez := zits_per_student_Perez * students_Perez -- should be 120

theorem highest_total_zits : 
  total_zits_Jones = max total_zits_Swanson (max total_zits_Smith (max total_zits_Brown (max total_zits_Perez total_zits_Jones))) :=
by
  sorry

end NUMINAMATH_GPT_highest_total_zits_l1757_175753


namespace NUMINAMATH_GPT_swim_club_percentage_l1757_175789

theorem swim_club_percentage (P : ℕ) (total_members : ℕ) (not_passed_taken_course : ℕ) (not_passed_not_taken_course : ℕ) :
  total_members = 50 →
  not_passed_taken_course = 5 →
  not_passed_not_taken_course = 30 →
  (total_members - (total_members * P / 100) = not_passed_taken_course + not_passed_not_taken_course) →
  P = 30 :=
by
  sorry

end NUMINAMATH_GPT_swim_club_percentage_l1757_175789


namespace NUMINAMATH_GPT_equation_of_line_perpendicular_l1757_175736

theorem equation_of_line_perpendicular 
  (P : ℝ × ℝ) (hx : P.1 = -1) (hy : P.2 = 2)
  (a b c : ℝ) (h_line : 2 * a - 3 * b + 4 = 0)
  (l : ℝ → ℝ) (h_perpendicular : ∀ x, l x = -(3/2) * x)
  (h_passing : l (-1) = 2)
  : a * 3 + b * 2 - 1 = 0 :=
sorry

end NUMINAMATH_GPT_equation_of_line_perpendicular_l1757_175736


namespace NUMINAMATH_GPT_min_value_expr_l1757_175786

theorem min_value_expr : ∀ (x : ℝ), 0 < x ∧ x < 4 → ∃ y : ℝ, y = (1 / (4 - x) + 2 / x) ∧ y = (3 + 2 * Real.sqrt 2) / 4 :=
by
  sorry

end NUMINAMATH_GPT_min_value_expr_l1757_175786


namespace NUMINAMATH_GPT_evaluate_expression_l1757_175757

theorem evaluate_expression : 
  Int.ceil (7 / 3 : ℚ) + Int.floor (-7 / 3 : ℚ) + Int.ceil (4 / 5 : ℚ) + Int.floor (-4 / 5 : ℚ) = 0 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1757_175757


namespace NUMINAMATH_GPT_sqrt_simplify_l1757_175759

theorem sqrt_simplify :
  Real.sqrt 450 = 15 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_simplify_l1757_175759


namespace NUMINAMATH_GPT_find_x_value_l1757_175743

/-- Defining the conditions given in the problem -/
structure HenrikhConditions where
  x : ℕ
  walking_time_per_block : ℕ := 60
  bicycle_time_per_block : ℕ := 20
  skateboard_time_per_block : ℕ := 40
  added_time_walking_over_bicycle : ℕ := 480
  added_time_walking_over_skateboard : ℕ := 240

/-- Defining a hypothesis based on the conditions -/
noncomputable def henrikh (c : HenrikhConditions) : Prop :=
  c.walking_time_per_block * c.x = c.bicycle_time_per_block * c.x + c.added_time_walking_over_bicycle ∧
  c.walking_time_per_block * c.x = c.skateboard_time_per_block * c.x + c.added_time_walking_over_skateboard

/-- The theorem to be proved -/
theorem find_x_value (c : HenrikhConditions) (h : henrikh c) : c.x = 12 := by
  sorry

end NUMINAMATH_GPT_find_x_value_l1757_175743


namespace NUMINAMATH_GPT_diagonal_length_l1757_175716

theorem diagonal_length (d : ℝ) 
  (offset1 offset2 : ℝ) 
  (area : ℝ) 
  (h_offsets : offset1 = 11) 
  (h_offsets2 : offset2 = 9) 
  (h_area : area = 400) : d = 40 :=
by 
  sorry

end NUMINAMATH_GPT_diagonal_length_l1757_175716


namespace NUMINAMATH_GPT_B_work_time_l1757_175721

noncomputable def workRateA (W : ℝ): ℝ := W / 14
noncomputable def combinedWorkRate (W : ℝ): ℝ := W / 10

theorem B_work_time (W : ℝ) :
  ∃ T : ℝ, (W / T) = (combinedWorkRate W) - (workRateA W) ∧ T = 35 :=
by {
  use 35,
  sorry
}

end NUMINAMATH_GPT_B_work_time_l1757_175721


namespace NUMINAMATH_GPT_playground_girls_l1757_175714

theorem playground_girls (total_children boys girls : ℕ) (h1 : boys = 40) (h2 : total_children = 117) (h3 : total_children = boys + girls) : girls = 77 := 
by 
  sorry

end NUMINAMATH_GPT_playground_girls_l1757_175714


namespace NUMINAMATH_GPT_correct_option_is_D_l1757_175783

noncomputable def data : List ℕ := [7, 5, 3, 5, 10]

theorem correct_option_is_D :
  let mean := (7 + 5 + 3 + 5 + 10) / 5
  let variance := (1 / 5 : ℚ) * ((7 - mean) ^ 2 + (5 - mean) ^ 2 + (5 - mean) ^ 2 + (3 - mean) ^ 2 + (10 - mean) ^ 2)
  let mode := 5
  let median := 5
  mean = 6 ∧ variance ≠ 3.6 ∧ mode ≠ 10 ∧ median ≠ 3 :=
by
  sorry

end NUMINAMATH_GPT_correct_option_is_D_l1757_175783


namespace NUMINAMATH_GPT_climbing_difference_l1757_175793

theorem climbing_difference (rate_matt rate_jason time : ℕ) (h_rate_matt : rate_matt = 6) (h_rate_jason : rate_jason = 12) (h_time : time = 7) : 
  rate_jason * time - rate_matt * time = 42 :=
by
  sorry

end NUMINAMATH_GPT_climbing_difference_l1757_175793


namespace NUMINAMATH_GPT_problem1_problem2_l1757_175751

-- Problem 1
theorem problem1 (x : ℝ) (hx1 : x ≠ 1) (hx2 : x ≠ 0) :
  (x^2 + x) / (x^2 - 2 * x + 1) / (2 / (x - 1) - 1 / x) = x^2 / (x - 1) := by
  sorry

-- Problem 2
theorem problem2 (x : ℤ) (hx1 : x > 0) :
  (2 * x + 1) / 3 - (5 * x - 1) / 2 < 1 ∧ 
  (5 * x - 1 < 3 * (x + 2)) →
  x = 1 ∨ x = 2 ∨ x = 3 := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1757_175751


namespace NUMINAMATH_GPT_students_in_both_band_and_chorus_l1757_175740

-- Definitions for conditions
def total_students : ℕ := 300
def students_in_band : ℕ := 100
def students_in_chorus : ℕ := 120
def students_in_band_or_chorus : ℕ := 195

-- Theorem: Prove the number of students in both band and chorus
theorem students_in_both_band_and_chorus : ℕ :=
  students_in_band + students_in_chorus - students_in_band_or_chorus

example : students_in_both_band_and_chorus = 25 := by
  sorry

end NUMINAMATH_GPT_students_in_both_band_and_chorus_l1757_175740


namespace NUMINAMATH_GPT_find_max_term_of_sequence_l1757_175764

theorem find_max_term_of_sequence :
  ∃ m : ℕ, (m = 8) ∧ ∀ n : ℕ, (0 < n → n ≠ m → a_n = (n - 7) / (n - 5 * Real.sqrt 2)) :=
by
  sorry

end NUMINAMATH_GPT_find_max_term_of_sequence_l1757_175764


namespace NUMINAMATH_GPT_sqrt_expression_l1757_175756

theorem sqrt_expression : Real.sqrt (3^2 * 4^4) = 48 := by
  sorry

end NUMINAMATH_GPT_sqrt_expression_l1757_175756


namespace NUMINAMATH_GPT_range_of_k_l1757_175777

-- Definitions to use in statement
variable (k : ℝ)

-- Statement: Proving the range of k
theorem range_of_k (h : ∀ x : ℝ, k * x^2 - k * x - 1 < 0) : -4 < k ∧ k ≤ 0 :=
  sorry

end NUMINAMATH_GPT_range_of_k_l1757_175777


namespace NUMINAMATH_GPT_find_digit_B_l1757_175780

theorem find_digit_B (A B : ℕ) (h1 : 100 * A + 78 - (210 + B) = 364) : B = 4 :=
by sorry

end NUMINAMATH_GPT_find_digit_B_l1757_175780


namespace NUMINAMATH_GPT_tan_of_alpha_l1757_175792

theorem tan_of_alpha
  (α : ℝ)
  (h1 : Real.sin (α + Real.pi / 2) = 1 / 3)
  (h2 : 0 < α ∧ α < Real.pi / 2) : 
  Real.tan α = 2 * Real.sqrt 2 := 
sorry

end NUMINAMATH_GPT_tan_of_alpha_l1757_175792


namespace NUMINAMATH_GPT_new_monthly_savings_l1757_175702

-- Definitions based on conditions
def monthly_salary := 4166.67
def initial_savings_percent := 0.20
def expense_increase_percent := 0.10

-- Calculations
def initial_savings := initial_savings_percent * monthly_salary
def initial_expenses := (1 - initial_savings_percent) * monthly_salary
def increased_expenses := initial_expenses + expense_increase_percent * initial_expenses
def new_savings := monthly_salary - increased_expenses

-- Lean statement to prove the question equals the answer given conditions
theorem new_monthly_savings :
  new_savings = 499.6704 := 
by
  sorry

end NUMINAMATH_GPT_new_monthly_savings_l1757_175702


namespace NUMINAMATH_GPT_quotient_of_division_l1757_175718

theorem quotient_of_division (a b : ℕ) (r q : ℕ) (h1 : a = 1637) (h2 : b + 1365 = a) (h3 : a = b * q + r) (h4 : r = 5) : q = 6 :=
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_quotient_of_division_l1757_175718


namespace NUMINAMATH_GPT_pump_capacity_l1757_175719

-- Define parameters and assumptions
def tank_volume : ℝ := 1000
def fill_percentage : ℝ := 0.85
def fill_time : ℝ := 1
def num_pumps : ℝ := 8
def pump_efficiency : ℝ := 0.75
def required_fill_volume : ℝ := fill_percentage * tank_volume

-- Assumed total effective capacity must meet the required fill volume
theorem pump_capacity (C : ℝ) : 
  (num_pumps * pump_efficiency * C = required_fill_volume) → 
  C = 850.0 / 6.0 :=
by
  sorry

end NUMINAMATH_GPT_pump_capacity_l1757_175719


namespace NUMINAMATH_GPT_teacher_selection_l1757_175795

/-- A school has 150 teachers, including 15 senior teachers, 45 intermediate teachers, 
and 90 junior teachers. By stratified sampling, 30 teachers are selected to 
participate in the teachers' representative conference. 
--/

def total_teachers : ℕ := 150
def senior_teachers : ℕ := 15
def intermediate_teachers : ℕ := 45
def junior_teachers : ℕ := 90

def total_selected_teachers : ℕ := 30
def selected_senior_teachers : ℕ := 3
def selected_intermediate_teachers : ℕ := 9
def selected_junior_teachers : ℕ := 18

def ratio (a b : ℕ) : ℕ × ℕ := (a / (gcd a b), b / (gcd a b))

theorem teacher_selection :
  ratio senior_teachers (gcd senior_teachers total_teachers) = ratio intermediate_teachers (gcd intermediate_teachers total_teachers) ∧
  ratio intermediate_teachers (gcd intermediate_teachers total_teachers) = ratio junior_teachers (gcd junior_teachers total_teachers) →
  selected_senior_teachers / selected_intermediate_teachers / selected_junior_teachers = 1 / 3 / 6 → 
  selected_senior_teachers + selected_intermediate_teachers + selected_junior_teachers = 30 :=
sorry

end NUMINAMATH_GPT_teacher_selection_l1757_175795


namespace NUMINAMATH_GPT_increasing_function_implies_a_nonpositive_l1757_175773

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x - 1

theorem increasing_function_implies_a_nonpositive (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x ≤ f a y) → a ≤ 0 :=
by
  sorry

end NUMINAMATH_GPT_increasing_function_implies_a_nonpositive_l1757_175773


namespace NUMINAMATH_GPT_unique_prime_triple_l1757_175758

/-- A prime is an integer greater than 1 whose only positive integer divisors are itself and 1. -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

/-- Prove that the only triple of primes (p, q, r), such that p = q + 2 and q = r + 2 is (7, 5, 3). -/
theorem unique_prime_triple (p q r : ℕ) (hp : is_prime p) (hq : is_prime q) (hr : is_prime r) :
  (p = q + 2) ∧ (q = r + 2) → (p = 7 ∧ q = 5 ∧ r = 3) := by
  sorry

end NUMINAMATH_GPT_unique_prime_triple_l1757_175758


namespace NUMINAMATH_GPT_smallest_four_digit_congruent_one_mod_17_l1757_175755

theorem smallest_four_digit_congruent_one_mod_17 :
  ∃ (n : ℕ), 1000 ≤ n ∧ n % 17 = 1 ∧ n = 1003 :=
by
sorry

end NUMINAMATH_GPT_smallest_four_digit_congruent_one_mod_17_l1757_175755


namespace NUMINAMATH_GPT_xyz_not_divisible_by_3_l1757_175744

theorem xyz_not_divisible_by_3 (x y z : ℕ) (h1 : x % 2 = 1) (h2 : y % 2 = 1) (h3 : z % 2 = 1) 
  (h4 : Nat.gcd (Nat.gcd x y) z = 1) (h5 : (x^2 + y^2 + z^2) % (x + y + z) = 0) : 
  (x + y + z - 2) % 3 ≠ 0 :=
by
  sorry

end NUMINAMATH_GPT_xyz_not_divisible_by_3_l1757_175744


namespace NUMINAMATH_GPT_survey_students_l1757_175760

theorem survey_students (S F : ℕ) (h1 : F = 20 + 60) (h2 : F = 40 * S / 100) : S = 200 :=
by
  sorry

end NUMINAMATH_GPT_survey_students_l1757_175760


namespace NUMINAMATH_GPT_count_not_divisible_by_5_or_7_l1757_175796

theorem count_not_divisible_by_5_or_7 :
  let n := 1000
  let count_divisible_by (m : ℕ) := Nat.floor (999 / m)
  (999 - count_divisible_by 5 - count_divisible_by 7 + count_divisible_by 35) = 686 :=
by
  sorry

end NUMINAMATH_GPT_count_not_divisible_by_5_or_7_l1757_175796


namespace NUMINAMATH_GPT_apples_left_total_l1757_175770

-- Define the initial conditions
def FrankApples : ℕ := 36
def SusanApples : ℕ := 3 * FrankApples
def SusanLeft : ℕ := SusanApples / 2
def FrankLeft : ℕ := (2 / 3) * FrankApples

-- Define the total apples left
def total_apples_left (SusanLeft FrankLeft : ℕ) : ℕ := SusanLeft + FrankLeft

-- Given conditions transformed to Lean
theorem apples_left_total : 
  total_apples_left (SusanApples / 2) ((2 / 3) * FrankApples) = 78 := by
  sorry

end NUMINAMATH_GPT_apples_left_total_l1757_175770


namespace NUMINAMATH_GPT_fraction_torn_off_l1757_175750

theorem fraction_torn_off (P: ℝ) (A_remaining: ℝ) (fraction: ℝ):
  P = 32 → 
  A_remaining = 48 → 
  fraction = 1 / 4 :=
by 
  sorry

end NUMINAMATH_GPT_fraction_torn_off_l1757_175750


namespace NUMINAMATH_GPT_speed_of_current_l1757_175724

-- Define the context and variables
variables (m c : ℝ)
-- State the conditions
variables (h1 : m + c = 12) (h2 : m - c = 8)

-- State the goal which is to prove the speed of the current
theorem speed_of_current : c = 2 :=
by
  sorry

end NUMINAMATH_GPT_speed_of_current_l1757_175724


namespace NUMINAMATH_GPT_great_grandson_age_l1757_175761

theorem great_grandson_age (n : ℕ) : 
  ∃ n, (n * (n + 1)) / 2 = 666 :=
by
  -- Solution steps would go here
  sorry

end NUMINAMATH_GPT_great_grandson_age_l1757_175761


namespace NUMINAMATH_GPT_sqrt_meaningful_range_l1757_175787

theorem sqrt_meaningful_range (x : ℝ) (h : 0 ≤ x - 2) : x ≥ 2 :=
sorry

end NUMINAMATH_GPT_sqrt_meaningful_range_l1757_175787


namespace NUMINAMATH_GPT_expensive_feed_cost_l1757_175778

/-- Tim and Judy mix two kinds of feed for pedigreed dogs. They made 35 pounds of feed worth 0.36 dollars per pound by mixing one kind worth 0.18 dollars per pound with another kind. They used 17 pounds of the cheaper kind in the mix. What is the cost per pound of the more expensive kind of feed? --/
theorem expensive_feed_cost 
  (total_feed : ℝ := 35) 
  (avg_cost : ℝ := 0.36) 
  (cheaper_feed : ℝ := 17) 
  (cheaper_cost : ℝ := 0.18) 
  (total_cost : ℝ := total_feed * avg_cost) 
  (cheaper_total_cost : ℝ := cheaper_feed * cheaper_cost) 
  (expensive_feed : ℝ := total_feed - cheaper_feed) : 
  (total_cost - cheaper_total_cost) / expensive_feed = 0.53 :=
by
  sorry

end NUMINAMATH_GPT_expensive_feed_cost_l1757_175778


namespace NUMINAMATH_GPT_hannah_dog_food_l1757_175734

def dog_food_consumption : Prop :=
  let dog1 : ℝ := 1.5 * 2
  let dog2 : ℝ := (1.5 * 2) * 1
  let dog3 : ℝ := (dog2 + 2.5) * 3
  let dog4 : ℝ := 1.2 * (dog2 + 2.5) * 2
  let dog5 : ℝ := 0.8 * 1.5 * 4
  let total_food := dog1 + dog2 + dog3 + dog4 + dog5
  total_food = 40.5

theorem hannah_dog_food : dog_food_consumption :=
  sorry

end NUMINAMATH_GPT_hannah_dog_food_l1757_175734


namespace NUMINAMATH_GPT_probability_different_colors_l1757_175762

-- Define the number of chips of each color
def num_blue := 6
def num_red := 5
def num_yellow := 4
def num_green := 3

-- Total number of chips
def total_chips := num_blue + num_red + num_yellow + num_green

-- Probability of drawing a chip of different color
theorem probability_different_colors : 
  (num_blue / total_chips) * ((total_chips - num_blue) / total_chips) +
  (num_red / total_chips) * ((total_chips - num_red) / total_chips) +
  (num_yellow / total_chips) * ((total_chips - num_yellow) / total_chips) +
  (num_green / total_chips) * ((total_chips - num_green) / total_chips) =
  119 / 162 := 
sorry

end NUMINAMATH_GPT_probability_different_colors_l1757_175762


namespace NUMINAMATH_GPT_num_divisors_of_36_l1757_175788

theorem num_divisors_of_36 : (∃ (S : Finset ℤ), (∀ x, x ∈ S ↔ x ∣ 36) ∧ S.card = 18) :=
sorry

end NUMINAMATH_GPT_num_divisors_of_36_l1757_175788


namespace NUMINAMATH_GPT_product_of_numbers_l1757_175710

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 30) (h2 : x^3 + y^3 = 9450) : x * y = -585 :=
  sorry

end NUMINAMATH_GPT_product_of_numbers_l1757_175710


namespace NUMINAMATH_GPT_auntie_em_can_park_l1757_175720

-- Define the conditions as formal statements in Lean
def parking_lot_spaces : ℕ := 20
def cars_arriving : ℕ := 14
def suv_adjacent_spaces : ℕ := 2

-- Define the total number of ways to park 14 cars in 20 spaces
def total_ways_to_park : ℕ := Nat.choose parking_lot_spaces cars_arriving
-- Define the number of unfavorable configurations where the SUV cannot park
def unfavorable_configs : ℕ := Nat.choose (parking_lot_spaces - suv_adjacent_spaces + 1) (parking_lot_spaces - cars_arriving)

-- Final probability calculation
def probability_park_suv : ℚ := 1 - (unfavorable_configs / total_ways_to_park)

-- Mathematically equivalent statement to be proved
theorem auntie_em_can_park : probability_park_suv = 850 / 922 :=
by sorry

end NUMINAMATH_GPT_auntie_em_can_park_l1757_175720


namespace NUMINAMATH_GPT_weight_removed_l1757_175712

-- Definitions for the given conditions
def weight_sugar : ℕ := 16
def weight_salt : ℕ := 30
def new_combined_weight : ℕ := 42

-- The proof problem statement
theorem weight_removed : (weight_sugar + weight_salt) - new_combined_weight = 4 := by
  -- Proof will be provided here
  sorry

end NUMINAMATH_GPT_weight_removed_l1757_175712


namespace NUMINAMATH_GPT_intersection_singleton_one_l1757_175735

-- Define sets A and B according to the given conditions
def setA : Set ℤ := { x | 0 < x ∧ x < 4 }
def setB : Set ℤ := { x | (x+1)*(x-2) < 0 }

-- Statement to prove A ∩ B = {1}
theorem intersection_singleton_one : setA ∩ setB = {1} :=
by 
  sorry

end NUMINAMATH_GPT_intersection_singleton_one_l1757_175735


namespace NUMINAMATH_GPT_arithmetic_example_l1757_175709

theorem arithmetic_example : 4 * (9 - 6) - 8 = 4 := by
  sorry

end NUMINAMATH_GPT_arithmetic_example_l1757_175709


namespace NUMINAMATH_GPT_ratio_hooper_bay_to_other_harbors_l1757_175739

-- Definitions based on conditions
def other_harbors_lobster : ℕ := 80
def total_lobster : ℕ := 480
def combined_other_harbors_lobster := 2 * other_harbors_lobster
def hooper_bay_lobster := total_lobster - combined_other_harbors_lobster

-- The theorem to prove
theorem ratio_hooper_bay_to_other_harbors : hooper_bay_lobster / combined_other_harbors_lobster = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_hooper_bay_to_other_harbors_l1757_175739


namespace NUMINAMATH_GPT_hyperbola_equation_Q_on_fixed_circle_l1757_175779

-- Define the hyperbola and necessary conditions
def hyperbola (a : ℝ) (x y : ℝ) : Prop := x^2 / a^2 - y^2 / (3 * a^2) = 1

-- Given conditions
variables (a : ℝ) (h_pos : a > 0)
variables (F1 F2 : ℝ × ℝ)
variables (dist_F2_asymptote : ℝ) (h_dist : dist_F2_asymptote = sqrt 3)
variables (left_vertex : ℝ × ℝ) (right_branch_intersect : ℝ × ℝ)
variables (line_x_half : ℝ × ℝ)
variables (line_PF2 : ℝ × ℝ)
variables (point_Q : ℝ × ℝ)

-- Prove that the equation of the hyperbola is correct
theorem hyperbola_equation :
  hyperbola a x y ↔ x^2 - y^2 / 3 = 1 :=
sorry

-- Prove that point Q lies on a fixed circle
theorem Q_on_fixed_circle :
  dist point_Q F2 = 4 :=
sorry

end NUMINAMATH_GPT_hyperbola_equation_Q_on_fixed_circle_l1757_175779


namespace NUMINAMATH_GPT_inequality_for_abcd_one_l1757_175790

theorem inequality_for_abcd_one (a b c d : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d) (h_prod : a * b * c * d = 1) :
  (1 / (1 + a)) + (1 / (1 + b)) + (1 / (1 + c)) + (1 / (1 + d)) > 1 := 
by
  sorry

end NUMINAMATH_GPT_inequality_for_abcd_one_l1757_175790


namespace NUMINAMATH_GPT_num_rectangles_in_grid_l1757_175704

theorem num_rectangles_in_grid : 
  let width := 35
  let height := 44
  ∃ n, n = 87 ∧ 
  ∀ x y, (1 ≤ x ∧ x ≤ width) ∧ (1 ≤ y ∧ y ≤ height) → 
    n = (x * (x + 1) / 2) * (y * (y + 1) / 2) := 
by
  sorry

end NUMINAMATH_GPT_num_rectangles_in_grid_l1757_175704


namespace NUMINAMATH_GPT_engineering_department_men_l1757_175706

theorem engineering_department_men (total_students men_percentage women_count : ℕ) (h_percentage : men_percentage = 70) (h_women : women_count = 180) (h_total : total_students = (women_count * 100) / (100 - men_percentage)) : 
  (total_students * men_percentage / 100) = 420 :=
by
  sorry

end NUMINAMATH_GPT_engineering_department_men_l1757_175706


namespace NUMINAMATH_GPT_actual_price_of_food_l1757_175726

noncomputable def food_price (total_spent: ℝ) (tip_percent: ℝ) (tax_percent: ℝ) (discount_percent: ℝ) : ℝ :=
  let P := total_spent / ((1 + tip_percent) * (1 + tax_percent) * (1 - discount_percent))
  P

theorem actual_price_of_food :
  food_price 198 0.20 0.10 0.15 = 176.47 :=
by
  sorry

end NUMINAMATH_GPT_actual_price_of_food_l1757_175726


namespace NUMINAMATH_GPT_probability_of_grid_being_black_l1757_175741

noncomputable def probability_grid_black_after_rotation : ℚ := sorry

theorem probability_of_grid_being_black:
  probability_grid_black_after_rotation = 429 / 21845 :=
sorry

end NUMINAMATH_GPT_probability_of_grid_being_black_l1757_175741


namespace NUMINAMATH_GPT_train_speed_l1757_175742

-- Define the conditions as given in part (a)
def train_length : ℝ := 160
def crossing_time : ℝ := 6

-- Define the statement to prove
theorem train_speed :
  train_length / crossing_time = 26.67 :=
by
  sorry

end NUMINAMATH_GPT_train_speed_l1757_175742


namespace NUMINAMATH_GPT_smallest_arith_prog_l1757_175791

theorem smallest_arith_prog (a d : ℝ) 
  (h1 : (a - 2 * d) < (a - d) ∧ (a - d) < a ∧ a < (a + d) ∧ (a + d) < (a + 2 * d))
  (h2 : (a - 2 * d)^2 + (a - d)^2 + a^2 + (a + d)^2 + (a + 2 * d)^2 = 70)
  (h3 : (a - 2 * d)^3 + (a - d)^3 + a^3 + (a + d)^3 + (a + 2 * d)^3 = 0)
  : (a - 2 * d) = -2 * Real.sqrt 7 :=
sorry

end NUMINAMATH_GPT_smallest_arith_prog_l1757_175791


namespace NUMINAMATH_GPT_portia_high_school_students_l1757_175733

theorem portia_high_school_students (P L : ℕ) (h1 : P = 4 * L) (h2 : P + L = 2500) : P = 2000 := by
  sorry

end NUMINAMATH_GPT_portia_high_school_students_l1757_175733


namespace NUMINAMATH_GPT_system_has_three_solutions_l1757_175771

theorem system_has_three_solutions (a : ℝ) :
  (a = 4 ∨ a = 64 ∨ a = 51 + 10 * Real.sqrt 2) ↔
  ∃ (x y : ℝ), 
    (x = abs (y - Real.sqrt a) + Real.sqrt a - 4 
    ∧ (abs x - 6)^2 + (abs y - 8)^2 = 100) 
        ∧ (∃! x1 y1 : ℝ, (x1 = abs (y1 - Real.sqrt a) + Real.sqrt a - 4 
        ∧ (abs x1 - 6)^2 + (abs y1 - 8)^2 = 100)) :=
by
  sorry

end NUMINAMATH_GPT_system_has_three_solutions_l1757_175771


namespace NUMINAMATH_GPT_painting_time_l1757_175781

theorem painting_time (n₁ t₁ n₂ t₂ : ℕ) (h1 : n₁ = 8) (h2 : t₁ = 12) (h3 : n₂ = 6) (h4 : n₁ * t₁ = n₂ * t₂) : t₂ = 16 :=
by
  sorry

end NUMINAMATH_GPT_painting_time_l1757_175781


namespace NUMINAMATH_GPT_g_3_2_plus_g_3_5_l1757_175728

def g (x y : ℚ) : ℚ :=
  if x + y ≤ 5 then (x * y - x + 3) / (3 * x) else (x * y - y - 3) / (-3 * y)

theorem g_3_2_plus_g_3_5 : g 3 2 + g 3 5 = 1/5 := by
  sorry

end NUMINAMATH_GPT_g_3_2_plus_g_3_5_l1757_175728


namespace NUMINAMATH_GPT_find_a_l1757_175775

-- Given conditions as definitions.
def f (a x : ℝ) := a * x^3
def tangent_line (a : ℝ) (x : ℝ) : ℝ := 3 * x + a - 3

-- Problem statement in Lean 4.
theorem find_a (a : ℝ) (h_tangent : ∀ x : ℝ, f a 1 = 1 ∧ f a 1 = tangent_line a 1) : a = 1 := 
by sorry

end NUMINAMATH_GPT_find_a_l1757_175775


namespace NUMINAMATH_GPT_reece_climbs_15_times_l1757_175738

/-
Given:
1. Keaton's ladder height: 30 feet.
2. Keaton climbs: 20 times.
3. Reece's ladder is 4 feet shorter than Keaton's ladder.
4. Total length of ladders climbed by both is 11880 inches.

Prove:
Reece climbed his ladder 15 times.
-/

theorem reece_climbs_15_times :
  let keaton_ladder_feet := 30
  let keaton_climbs := 20
  let reece_ladder_feet := keaton_ladder_feet - 4
  let total_length_inches := 11880
  let feet_to_inches (feet : ℕ) := 12 * feet
  let keaton_ladder_inches := feet_to_inches keaton_ladder_feet
  let reece_ladder_inches := feet_to_inches reece_ladder_feet
  let keaton_total_climbed := keaton_ladder_inches * keaton_climbs
  let reece_total_climbed := total_length_inches - keaton_total_climbed
  let reece_climbs := reece_total_climbed / reece_ladder_inches
  reece_climbs = 15 :=
by
  sorry

end NUMINAMATH_GPT_reece_climbs_15_times_l1757_175738


namespace NUMINAMATH_GPT_sum_abc_l1757_175700

noncomputable def f (a b c : ℕ) (x : ℤ) : ℤ :=
  if x > 0 then a * x + 3
  else if x = 0 then a * b
  else b * x^2 + c

theorem sum_abc (a b c : ℕ) (h1 : f a b c 2 = 7) (h2 : f a b c 0 = 6) (h3 : f a b c (-1) = 8) :
  a + b + c = 10 :=
by {
  sorry
}

end NUMINAMATH_GPT_sum_abc_l1757_175700


namespace NUMINAMATH_GPT_power_function_properties_l1757_175731

theorem power_function_properties (α : ℝ) (h : (3 : ℝ) ^ α = 27) :
  (α = 3) →
  (∀ x : ℝ, (x ^ α) = x ^ 3) ∧
  (∀ x : ℝ, x ^ α = -(((-x) ^ α))) ∧
  (∀ x y : ℝ, x < y → x ^ α < y ^ α) ∧
  (∀ y : ℝ, ∃ x : ℝ, x ^ α = y) :=
by
  sorry

end NUMINAMATH_GPT_power_function_properties_l1757_175731


namespace NUMINAMATH_GPT_table_height_l1757_175772

variable (l h w : ℝ)

-- Given conditions:
def conditionA := l + h - w = 36
def conditionB := w + h - l = 30

-- Proof that height of the table h is 33 inches
theorem table_height {l h w : ℝ} 
  (h1 : l + h - w = 36) 
  (h2 : w + h - l = 30) : 
  h = 33 := 
by
  sorry

end NUMINAMATH_GPT_table_height_l1757_175772


namespace NUMINAMATH_GPT_set_intersection_is_correct_l1757_175784

def setA : Set ℝ := {x | x^2 - 4 * x > 0}
def setB : Set ℝ := {x | abs (x - 1) ≤ 2}
def setIntersection : Set ℝ := {x | -1 ≤ x ∧ x < 0}

theorem set_intersection_is_correct :
  setA ∩ setB = setIntersection := 
by
  sorry

end NUMINAMATH_GPT_set_intersection_is_correct_l1757_175784


namespace NUMINAMATH_GPT_sherry_needs_bananas_l1757_175729

/-
Conditions:
- Sherry wants to make 99 loaves.
- Her recipe makes enough batter for 3 loaves.
- The recipe calls for 1 banana per batch of 3 loaves.

Question:
- How many bananas does Sherry need?

Equivalent Proof Problem:
- Prove that given the conditions, the number of bananas needed is 33.
-/

def total_loaves : ℕ := 99
def loaves_per_batch : ℕ := 3
def bananas_per_batch : ℕ := 1

theorem sherry_needs_bananas :
  (total_loaves / loaves_per_batch) * bananas_per_batch = 33 :=
sorry

end NUMINAMATH_GPT_sherry_needs_bananas_l1757_175729


namespace NUMINAMATH_GPT_find_range_g_l1757_175722

noncomputable def g (x : ℝ) : ℝ := Real.exp x + Real.exp (-x) + abs x

theorem find_range_g :
  {x : ℝ | g (2 * x - 1) < g 3} = {x : ℝ | -1 < x ∧ x < 2} :=
by
  sorry

end NUMINAMATH_GPT_find_range_g_l1757_175722


namespace NUMINAMATH_GPT_largest_remainder_division_by_11_l1757_175708

theorem largest_remainder_division_by_11 (A B C : ℕ) (h : A = 11 * B + C) (hC : 0 ≤ C ∧ C < 11) : C ≤ 10 :=
  sorry

end NUMINAMATH_GPT_largest_remainder_division_by_11_l1757_175708


namespace NUMINAMATH_GPT_volume_of_mixture_l1757_175794

section
variable (Va Vb Vtotal : ℝ)

theorem volume_of_mixture :
  (Va / Vb = 3 / 2) →
  (800 * Va + 850 * Vb = 2460) →
  (Vtotal = Va + Vb) →
  Vtotal = 2.998 :=
by
  intros h1 h2 h3
  sorry
end

end NUMINAMATH_GPT_volume_of_mixture_l1757_175794


namespace NUMINAMATH_GPT_sum_of_digits_palindrome_l1757_175725

theorem sum_of_digits_palindrome 
  (r : ℕ) 
  (h1 : r ≤ 36) 
  (x p q : ℕ) 
  (h2 : 2 * q = 5 * p) 
  (h3 : x = p * r^3 + p * r^2 + q * r + q) 
  (h4 : ∃ (a b c : ℕ), (x * x = a * r^6 + b * r^5 + c * r^4 + 0 * r^3 + c * r^2 + b * r + a)) : 
  (2 * (a + b + c) = 36) := 
sorry

end NUMINAMATH_GPT_sum_of_digits_palindrome_l1757_175725


namespace NUMINAMATH_GPT_probability_of_event_A_l1757_175737

def total_balls : ℕ := 10
def white_balls : ℕ := 7
def black_balls : ℕ := 3

def event_A : Prop := (black_balls / total_balls) * (white_balls / (total_balls - 1)) = 7 / 30

theorem probability_of_event_A : event_A := by
  sorry

end NUMINAMATH_GPT_probability_of_event_A_l1757_175737


namespace NUMINAMATH_GPT_max_sum_ge_zero_l1757_175754

-- Definition for max and min functions for real numbers
noncomputable def max_real (x y : ℝ) := if x ≥ y then x else y
noncomputable def min_real (x y : ℝ) := if x ≤ y then x else y

-- Condition: a + b + c + d = 0
def sum_zero (a b c d : ℝ) := a + b + c + d = 0

-- Lean statement for Problem (a)
theorem max_sum_ge_zero (a b c d : ℝ) (h : sum_zero a b c d) : 
  max_real a b + max_real a c + max_real a d + max_real b c + max_real b d + max_real c d ≥ 0 :=
sorry

-- Lean statement for Problem (b)
def find_max_k : ℕ :=
2

end NUMINAMATH_GPT_max_sum_ge_zero_l1757_175754


namespace NUMINAMATH_GPT_cost_price_of_one_toy_l1757_175776

-- Definitions translating the conditions into Lean
def total_revenue (toys_sold : ℕ) (price_per_toy : ℕ) : ℕ := toys_sold * price_per_toy
def gain (cost_per_toy : ℕ) (toys_gained : ℕ) : ℕ := cost_per_toy * toys_gained

-- Given the conditions in the problem
def total_cost_price_of_sold_toys := 18 * (1300 : ℕ)
def gain_from_sale := 3 * (1300 : ℕ)
def selling_price := total_cost_price_of_sold_toys + gain_from_sale

-- The target theorem we want to prove
theorem cost_price_of_one_toy : (selling_price = 27300) → (1300 = 27300 / 21) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_cost_price_of_one_toy_l1757_175776


namespace NUMINAMATH_GPT_evaluate_expression_l1757_175732

def g (x : ℝ) := 3 * x^2 - 5 * x + 7

theorem evaluate_expression : 3 * g 2 + 2 * g (-4) = 177 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1757_175732


namespace NUMINAMATH_GPT_shaded_trapezoids_perimeter_l1757_175767

theorem shaded_trapezoids_perimeter :
  let l := 8
  let w := 6
  let half_diagonal_1 := (l^2 + w^2) / 2
  let perimeter := 2 * (w + (half_diagonal_1 / l))
  let total_perimeter := perimeter + perimeter + half_diagonal_1
  total_perimeter = 48 :=
by 
  sorry

end NUMINAMATH_GPT_shaded_trapezoids_perimeter_l1757_175767


namespace NUMINAMATH_GPT_MrsHiltReadTotalChapters_l1757_175701

-- Define the number of books and chapters per book
def numberOfBooks : ℕ := 4
def chaptersPerBook : ℕ := 17

-- Define the total number of chapters Mrs. Hilt read
def totalChapters (books : ℕ) (chapters : ℕ) : ℕ := books * chapters

-- The main statement to be proved
theorem MrsHiltReadTotalChapters : totalChapters numberOfBooks chaptersPerBook = 68 := by
  sorry

end NUMINAMATH_GPT_MrsHiltReadTotalChapters_l1757_175701


namespace NUMINAMATH_GPT_range_of_x_l1757_175774

theorem range_of_x (x : ℝ) (h : x > -2) : ∃ y : ℝ, y = x / (Real.sqrt (x + 2)) :=
by {
  sorry
}

end NUMINAMATH_GPT_range_of_x_l1757_175774


namespace NUMINAMATH_GPT_radius_of_circle_l1757_175727

-- Define the polar coordinates equation
def polar_circle (ρ θ : ℝ) : Prop := ρ = 6 * Real.cos θ

-- Define the conversion to Cartesian coordinates and the circle equation
def cartesian_circle (x y : ℝ) : Prop := (x - 3) ^ 2 + y ^ 2 = 9

-- Prove that given the polar coordinates equation, the radius of the circle is 3
theorem radius_of_circle : ∀ (ρ θ : ℝ), polar_circle ρ θ → ∃ r, r = 3 := by
  sorry

end NUMINAMATH_GPT_radius_of_circle_l1757_175727


namespace NUMINAMATH_GPT_negation_of_existential_l1757_175711

theorem negation_of_existential (h : ∃ x_0 : ℝ, x_0^3 - x_0^2 + 1 ≤ 0) : 
  ∀ x : ℝ, x^3 - x^2 + 1 > 0 :=
sorry

end NUMINAMATH_GPT_negation_of_existential_l1757_175711


namespace NUMINAMATH_GPT_sum_of_primes_between_1_and_20_l1757_175768

theorem sum_of_primes_between_1_and_20 :
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19) = 77 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_primes_between_1_and_20_l1757_175768


namespace NUMINAMATH_GPT_min_value_of_inverse_sum_l1757_175707

noncomputable def min_value (a b : ℝ) := ¬(1 ≤ a + 2*b)

theorem min_value_of_inverse_sum (a b : ℝ) (h : a + 2 * b = 1) (h_nonneg : 0 < a ∧ 0 < b) :
  (1 / a + 2 / b) ≥ 9 :=
sorry

end NUMINAMATH_GPT_min_value_of_inverse_sum_l1757_175707


namespace NUMINAMATH_GPT_Seojun_apples_decimal_l1757_175752

theorem Seojun_apples_decimal :
  let total_apples := 100
  let seojun_apples := 11
  seojun_apples / total_apples = 0.11 :=
by
  let total_apples := 100
  let seojun_apples := 11
  sorry

end NUMINAMATH_GPT_Seojun_apples_decimal_l1757_175752


namespace NUMINAMATH_GPT_simplify_and_evaluate_l1757_175782

theorem simplify_and_evaluate (a : ℝ) (h : a = 3) : ((2 * a / (a + 1) - 1) / ((a - 1)^2 / (a + 1))) = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l1757_175782


namespace NUMINAMATH_GPT_map_distance_ratio_l1757_175705

theorem map_distance_ratio (actual_distance_km : ℕ) (map_distance_cm : ℕ) (h1 : actual_distance_km = 6) (h2 : map_distance_cm = 20) : map_distance_cm / (actual_distance_km * 100000) = 1 / 30000 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_map_distance_ratio_l1757_175705


namespace NUMINAMATH_GPT_trigonometric_expression_l1757_175765

theorem trigonometric_expression (θ : ℝ) (h : Real.tan θ = -3) :
    2 / (3 * (Real.sin θ) ^ 2 - (Real.cos θ) ^ 2) = 10 / 13 :=
by
  -- sorry to skip the proof
  sorry

end NUMINAMATH_GPT_trigonometric_expression_l1757_175765


namespace NUMINAMATH_GPT_cubes_not_touching_tin_foil_volume_l1757_175769

-- Definitions for the conditions given
variables (l w h : ℕ)
-- Condition 1: Width is twice the length
def width_twice_length := w = 2 * l
-- Condition 2: Width is twice the height
def width_twice_height := w = 2 * h
-- Condition 3: The adjusted width for the inner structure in inches
def adjusted_width := w = 8

-- The theorem statement to prove the final answer
theorem cubes_not_touching_tin_foil_volume : 
  width_twice_length l w → 
  width_twice_height w h →
  adjusted_width w →
  l * w * h = 128 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_cubes_not_touching_tin_foil_volume_l1757_175769


namespace NUMINAMATH_GPT_find_value_l1757_175730

variable (a : ℝ) (h : a + 1/a = 7)

theorem find_value :
  a^2 + 1/a^2 = 47 :=
sorry

end NUMINAMATH_GPT_find_value_l1757_175730


namespace NUMINAMATH_GPT_three_times_greater_than_two_l1757_175746

theorem three_times_greater_than_two (x : ℝ) : 3 * x - 2 > 0 → 3 * x > 2 :=
by
  sorry

end NUMINAMATH_GPT_three_times_greater_than_two_l1757_175746


namespace NUMINAMATH_GPT_reciprocal_eq_self_is_one_or_neg_one_l1757_175748

/-- If a rational number equals its own reciprocal, then the number is either 1 or -1. -/
theorem reciprocal_eq_self_is_one_or_neg_one (x : ℚ) (h : x = 1 / x) : x = 1 ∨ x = -1 := 
by
  sorry

end NUMINAMATH_GPT_reciprocal_eq_self_is_one_or_neg_one_l1757_175748


namespace NUMINAMATH_GPT_expenditure_increase_36_percent_l1757_175715

theorem expenditure_increase_36_percent
  (m : ℝ) -- mass of the bread
  (p_bread : ℝ) -- price of the bread
  (p_crust : ℝ) -- price of the crust
  (h1 : p_crust = 1.2 * p_bread) -- condition: crust is 20% more expensive
  (h2 : p_crust = 1.2 * p_bread) -- condition: crust is 20% more expensive
  (h3 : ∃ (m_crust : ℝ), m_crust = 0.75 * m) -- condition: crust is 25% lighter in weight
  (h4 : ∃ (m_consumed_bread : ℝ), m_consumed_bread = 0.85 * m) -- condition: 15% of bread dries out
  (h5 : ∃ (m_consumed_crust : ℝ), m_consumed_crust = 0.75 * m) -- condition: crust is consumed completely
  : (17 / 15) * (1.2 : ℝ) = 1.36 := 
by sorry

end NUMINAMATH_GPT_expenditure_increase_36_percent_l1757_175715


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1757_175745

theorem solution_set_of_inequality (x : ℝ) : x^2 - |x| - 2 ≤ 0 ↔ -2 ≤ x ∧ x ≤ 2 := by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1757_175745


namespace NUMINAMATH_GPT_base_conversion_problem_l1757_175749

theorem base_conversion_problem (n d : ℕ) (hn : 0 < n) (hd : d < 10) 
  (h1 : 3 * n^2 + 2 * n + d = 263) (h2 : 3 * n^2 + 2 * n + 4 = 253 + 6 * d) : 
  n + d = 11 :=
by
  sorry

end NUMINAMATH_GPT_base_conversion_problem_l1757_175749


namespace NUMINAMATH_GPT_students_received_B_l1757_175766

/-!
# Problem Statement

Given:
1. In Mr. Johnson's class, 18 out of 30 students received a B.
2. Ms. Smith has 45 students in total, and the ratio of students receiving a B is the same as in Mr. Johnson's class.
Prove:
27 students in Ms. Smith's class received a B.
-/

theorem students_received_B (s1 s2 b1 : ℕ) (r1 : ℚ) (r2 : ℕ) (h₁ : s1 = 30) (h₂ : b1 = 18) (h₃ : s2 = 45) (h₄ : r1 = 3/5) 
(H : (b1 : ℚ) / s1 = r1) : r2 = 27 :=
by
  -- Conditions provided
  -- h₁ : s1 = 30
  -- h₂ : b1 = 18
  -- h₃ : s2 = 45
  -- h₄ : r1 = 3/5
  -- H : (b1 : ℚ) / s1 = r1
  sorry

end NUMINAMATH_GPT_students_received_B_l1757_175766


namespace NUMINAMATH_GPT_value_of_x_squared_plus_inverse_squared_l1757_175713

theorem value_of_x_squared_plus_inverse_squared (x : ℝ) (hx : x + 1 / x = 8) : x^2 + 1 / x^2 = 62 := 
sorry

end NUMINAMATH_GPT_value_of_x_squared_plus_inverse_squared_l1757_175713


namespace NUMINAMATH_GPT_triangle_side_length_l1757_175763

theorem triangle_side_length (A : ℝ) (AC BC AB : ℝ) 
  (hA : A = 60)
  (hAC : AC = 4)
  (hBC : BC = 2 * Real.sqrt 3) :
  AB = 2 :=
sorry

end NUMINAMATH_GPT_triangle_side_length_l1757_175763


namespace NUMINAMATH_GPT_max_m_n_l1757_175785

theorem max_m_n (m n: ℕ) (h: m + 3*n - 5 = 2 * Nat.lcm m n - 11 * Nat.gcd m n) : 
  m + n ≤ 70 :=
sorry

end NUMINAMATH_GPT_max_m_n_l1757_175785


namespace NUMINAMATH_GPT_chord_length_range_l1757_175723

variable {x y : ℝ}

def center : ℝ × ℝ := (4, 5)
def radius : ℝ := 13
def point : ℝ × ℝ := (1, 1)
def circle_eq (x y : ℝ) : Prop := (x - 4)^2 + (y - 5)^2 = 169

-- statement: prove the range of |AB| for specific conditions
theorem chord_length_range :
  ∀ line : (ℝ × ℝ) → (ℝ × ℝ) → Prop,
  (line center point → line (x, y) (x, y) ∧ circle_eq x y)
  → 24 ≤ abs (dist (x, y) (x, y)) ∧ abs (dist (x, y) (x, y)) ≤ 26 :=
by
  sorry

end NUMINAMATH_GPT_chord_length_range_l1757_175723


namespace NUMINAMATH_GPT_largest_divisor_of_square_difference_l1757_175799

theorem largest_divisor_of_square_difference (m n : ℤ) (hm : m % 2 = 0) (hn : n % 2 = 0) (h : n < m) : 
  ∃ d, ∀ m n, (m % 2 = 0) → (n % 2 = 0) → (n < m) → d ∣ (m^2 - n^2) ∧ ∀ k, (∀ m n, (m % 2 = 0) → (n % 2 = 0) → (n < m) → k ∣ (m^2 - n^2)) → k ≤ d :=
sorry

end NUMINAMATH_GPT_largest_divisor_of_square_difference_l1757_175799


namespace NUMINAMATH_GPT_better_sequence_is_BAB_l1757_175703

def loss_prob_andrei : ℝ := 0.4
def loss_prob_boris : ℝ := 0.3

def win_prob_andrei : ℝ := 1 - loss_prob_andrei
def win_prob_boris : ℝ := 1 - loss_prob_boris

def prob_qualify_ABA : ℝ :=
  win_prob_andrei * loss_prob_boris * win_prob_andrei +
  win_prob_andrei * win_prob_boris +
  loss_prob_andrei * win_prob_boris * win_prob_andrei

def prob_qualify_BAB : ℝ :=
  win_prob_boris * loss_prob_andrei * win_prob_boris +
  win_prob_boris * win_prob_andrei +
  loss_prob_boris * win_prob_andrei * win_prob_boris

theorem better_sequence_is_BAB : prob_qualify_BAB = 0.742 ∧ prob_qualify_BAB > prob_qualify_ABA :=
by 
  sorry

end NUMINAMATH_GPT_better_sequence_is_BAB_l1757_175703
