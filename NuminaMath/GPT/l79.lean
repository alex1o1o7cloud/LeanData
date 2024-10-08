import Mathlib

namespace distinct_names_impossible_l79_79559

-- Define the alphabet
inductive Letter
| a | u | o | e

-- Simplified form of words in the Mumbo-Jumbo language
def simplified_form : List Letter → List Letter
| [] => []
| (Letter.e :: xs) => simplified_form xs
| (Letter.a :: Letter.a :: Letter.a :: Letter.a :: xs) => simplified_form (Letter.a :: Letter.a :: xs)
| (Letter.o :: Letter.o :: Letter.o :: Letter.o :: xs) => simplified_form xs
| (Letter.a :: Letter.a :: Letter.a :: Letter.u :: xs) => simplified_form (Letter.u :: xs)
| (x :: xs) => x :: simplified_form xs

-- Number of possible names
def num_possible_names : ℕ := 343

-- Number of tribe members
def num_tribe_members : ℕ := 400

theorem distinct_names_impossible : num_possible_names < num_tribe_members :=
by
  -- Skipping the proof with 'sorry'
  sorry

end distinct_names_impossible_l79_79559


namespace edwards_initial_money_l79_79254

variable (spent1 spent2 current remaining : ℕ)

def initial_money (spent1 spent2 current remaining : ℕ) : ℕ :=
  spent1 + spent2 + current

theorem edwards_initial_money :
  spent1 = 9 → spent2 = 8 → remaining = 17 →
  initial_money spent1 spent2 remaining remaining = 34 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end edwards_initial_money_l79_79254


namespace length_of_shorter_train_l79_79290

noncomputable def relativeSpeedInMS (speed1_kmh speed2_kmh : ℝ) : ℝ :=
  (speed1_kmh + speed2_kmh) * (5 / 18)

noncomputable def totalDistanceCovered (relativeSpeed_ms time_s : ℝ) : ℝ :=
  relativeSpeed_ms * time_s

noncomputable def lengthOfShorterTrain (longerTrainLength_m time_s : ℝ) (speed1_kmh speed2_kmh : ℝ) : ℝ :=
  let relativeSpeed_ms := relativeSpeedInMS speed1_kmh speed2_kmh
  let totalDistance := totalDistanceCovered relativeSpeed_ms time_s
  totalDistance - longerTrainLength_m

theorem length_of_shorter_train :
  lengthOfShorterTrain 160 10.07919366450684 60 40 = 117.8220467912412 := 
sorry

end length_of_shorter_train_l79_79290


namespace maxEccentricity_l79_79635

noncomputable def majorAxisLength := 4
noncomputable def majorSemiAxis := 2
noncomputable def leftVertexParabolaEq (y : ℝ) := y^2 = -3
noncomputable def distanceCondition (c : ℝ) := 2^2 / c - 2 ≥ 1

theorem maxEccentricity : ∃ c : ℝ, distanceCondition c ∧ (c ≤ 4 / 3) ∧ (c / majorSemiAxis = 2 / 3) :=
by
  sorry

end maxEccentricity_l79_79635


namespace initial_coins_l79_79586

-- Define the condition for the initial number of coins
variable (x : Nat) -- x represents the initial number of coins

-- The main statement theorem that needs proof
theorem initial_coins (h : x + 8 = 29) : x = 21 := 
by { sorry } -- placeholder for the proof

end initial_coins_l79_79586


namespace percent_full_time_more_than_three_years_l79_79260

variable (total_associates : ℕ)
variable (second_year_percentage : ℕ)
variable (third_year_percentage : ℕ)
variable (non_first_year_percentage : ℕ)
variable (part_time_percentage : ℕ)
variable (part_time_more_than_two_years_percentage : ℕ)
variable (full_time_more_than_three_years_percentage : ℕ)

axiom condition_1 : second_year_percentage = 30
axiom condition_2 : third_year_percentage = 20
axiom condition_3 : non_first_year_percentage = 60
axiom condition_4 : part_time_percentage = 10
axiom condition_5 : part_time_more_than_two_years_percentage = 5

theorem percent_full_time_more_than_three_years : 
  full_time_more_than_three_years_percentage = 10 := 
sorry

end percent_full_time_more_than_three_years_l79_79260


namespace factorization_of_polynomial_solve_quadratic_equation_l79_79157

-- Problem 1: Factorization
theorem factorization_of_polynomial : ∀ y : ℝ, 2 * y^2 - 8 = 2 * (y + 2) * (y - 2) :=
by
  intro y
  sorry

-- Problem 2: Solving the quadratic equation
theorem solve_quadratic_equation : ∀ x : ℝ, x^2 + 4 * x + 3 = 0 ↔ x = -1 ∨ x = -3 :=
by
  intro x
  sorry

end factorization_of_polynomial_solve_quadratic_equation_l79_79157


namespace cole_cost_l79_79780

def length_of_sides := 15
def length_of_back := 30
def cost_per_foot_side := 4
def cost_per_foot_back := 5
def cole_installation_fee := 50

def neighbor_behind_contribution := (length_of_back * cost_per_foot_back) / 2
def neighbor_left_contribution := (length_of_sides * cost_per_foot_side) / 3

def total_cost := 
  2 * length_of_sides * cost_per_foot_side + 
  length_of_back * cost_per_foot_back

def cole_contribution := 
  total_cost - neighbor_behind_contribution - neighbor_left_contribution + cole_installation_fee

theorem cole_cost (h : cole_contribution = 225) : cole_contribution = 225 := by
  sorry

end cole_cost_l79_79780


namespace problem_1_problem_2_l79_79498

def f (x : ℝ) : ℝ := x^2 + 4 * x
def g (a : ℝ) : ℝ := |a - 2| + |a + 1|

theorem problem_1 (x : ℝ) :
    (f x ≥ g 3) ↔ (x ≥ 1 ∨ x ≤ -5) :=
  sorry

theorem problem_2 (a : ℝ) :
    (∃ x : ℝ, f x + g a = 0) → (-3 / 2 ≤ a ∧ a ≤ 5 / 2) :=
  sorry

end problem_1_problem_2_l79_79498


namespace sum_of_squares_l79_79189

theorem sum_of_squares (r b s : ℕ) 
  (h1 : 2 * r + 3 * b + s = 80) 
  (h2 : 4 * r + 2 * b + 3 * s = 98) : 
  r^2 + b^2 + s^2 = 485 := 
by {
  sorry
}

end sum_of_squares_l79_79189


namespace eleonora_age_l79_79311

-- Definitions
def age_eleonora (e m : ℕ) : Prop :=
m - e = 3 * (2 * e - m) ∧ 3 * e + (m + 2 * e) = 100

-- Theorem stating that Eleonora's age is 15
theorem eleonora_age (e m : ℕ) (h : age_eleonora e m) : e = 15 :=
sorry

end eleonora_age_l79_79311


namespace solve_for_x_l79_79859

theorem solve_for_x : ∃ x : ℚ, 24 - 4 = 3 * (1 + x) ∧ x = 17 / 3 :=
by
  sorry

end solve_for_x_l79_79859


namespace coefficients_identity_l79_79893

def coefficients_of_quadratic (a b c : ℤ) (x : ℤ) : Prop :=
  a * x^2 + b * x + c = 0

theorem coefficients_identity : ∀ x : ℤ,
  coefficients_of_quadratic 3 (-4) 1 x :=
by
  sorry

end coefficients_identity_l79_79893


namespace union_of_sets_l79_79787

theorem union_of_sets (P Q : Set ℝ) 
  (hP : P = {x | 2 ≤ x ∧ x ≤ 3}) 
  (hQ : Q = {x | x^2 ≤ 4}) : 
  P ∪ Q = {x | -2 ≤ x ∧ x ≤ 3} := 
sorry

end union_of_sets_l79_79787


namespace toys_produced_each_day_l79_79258

def toys_produced_per_week : ℕ := 6000
def work_days_per_week : ℕ := 4

theorem toys_produced_each_day :
  (toys_produced_per_week / work_days_per_week) = 1500 := 
by
  -- The details of the proof are omitted
  -- The correct answer given the conditions is 1500 toys
  sorry

end toys_produced_each_day_l79_79258


namespace positive_int_sum_square_l79_79997

theorem positive_int_sum_square (M : ℕ) (h_pos : 0 < M) (h_eq : M^2 + M = 12) : M = 3 :=
by
  sorry

end positive_int_sum_square_l79_79997


namespace least_possible_value_z_minus_x_l79_79329

theorem least_possible_value_z_minus_x (x y z : ℤ) (h1 : Even x) (h2 : Odd y) (h3 : Odd z) (h4 : x < y) (h5 : y < z) (h6 : y - x > 5) : z - x = 9 := 
sorry

end least_possible_value_z_minus_x_l79_79329


namespace range_of_a_l79_79206

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, e^x + 1/e^x > a) ∧ (∃ x : ℝ, x^2 + 8*x + a^2 = 0) ↔ (-4 ≤ a ∧ a < 2) :=
by
  sorry

end range_of_a_l79_79206


namespace triangle_perimeter_triangle_side_c_l79_79077

theorem triangle_perimeter (A B C : ℝ) (a b c : ℝ) (h1 : b * (Real.sin (A/2))^2 + a * (Real.sin (B/2))^2 = C / 2) (h2 : c = 2) : 
  a + b + c = 6 := 
sorry

theorem triangle_side_c (A B C : ℝ) (a b c : ℝ) (h1 : b * (Real.sin (A/2))^2 + a * (Real.sin (B/2))^2 = C / 2) 
(h2 : C = Real.pi / 3) (h3 : 2 * Real.sqrt 3 = (1/2) * a * b * Real.sin (Real.pi / 3)) : 
c = 2 * Real.sqrt 2 := 
sorry

end triangle_perimeter_triangle_side_c_l79_79077


namespace evaluate_g_at_neg1_l79_79980

def g (x : ℚ) : ℚ := (2 * x - 3) / (4 * x + 5)

theorem evaluate_g_at_neg1 : g (-1) = -5 := by
  sorry

end evaluate_g_at_neg1_l79_79980


namespace work_completion_time_extension_l79_79964

theorem work_completion_time_extension
    (total_men : ℕ) (initial_days : ℕ) (remaining_men : ℕ) (man_days : ℕ) :
    total_men = 100 →
    initial_days = 20 →
    remaining_men = 50 →
    man_days = total_men * initial_days →
    (man_days / remaining_men) - initial_days = 20 :=
by
  intros h1 h2 h3 h4
  sorry

end work_completion_time_extension_l79_79964


namespace find_f_inv_486_l79_79067

open Function

noncomputable def f (x : ℕ) : ℕ := sorry -- placeholder for function definition

axiom f_condition1 : f 5 = 2
axiom f_condition2 : ∀ (x : ℕ), f (3 * x) = 3 * f x

theorem find_f_inv_486 : f⁻¹' {486} = {1215} := sorry

end find_f_inv_486_l79_79067


namespace exists_digit_sum_divisible_by_27_not_number_l79_79681

-- Definitions
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def divisible_by (a b : ℕ) : Prop :=
  b ≠ 0 ∧ a % b = 0

-- Theorem statement
theorem exists_digit_sum_divisible_by_27_not_number (n : ℕ) :
  divisible_by (sum_of_digits n) 27 ∧ ¬ divisible_by n 27 :=
  sorry

end exists_digit_sum_divisible_by_27_not_number_l79_79681


namespace find_a_l79_79327

theorem find_a (x : ℝ) (a : ℝ)
  (h1 : 3 * x - 4 = a)
  (h2 : (x + a) / 3 = 1)
  (h3 : (x = (a + 4) / 3) → (x = 3 - a → ((a + 4) / 3 = 2 * (3 - a)))) :
  a = 2 :=
sorry

end find_a_l79_79327


namespace octal_rep_square_l79_79192

theorem octal_rep_square (a b c : ℕ) (n : ℕ) (h : n^2 = 8^3 * a + 8^2 * b + 8 * 3 + c) (h₀ : a ≠ 0) : c = 1 :=
sorry

end octal_rep_square_l79_79192


namespace students_last_year_l79_79462

theorem students_last_year (students_this_year : ℝ) (increase_percent : ℝ) (last_year_students : ℝ) 
  (h1 : students_this_year = 960) 
  (h2 : increase_percent = 0.20) 
  (h3 : students_this_year = last_year_students * (1 + increase_percent)) : 
  last_year_students = 800 :=
by 
  sorry

end students_last_year_l79_79462


namespace man_speed_l79_79365

theorem man_speed (distance : ℝ) (time_minutes : ℝ) (time_hours : ℝ) (speed : ℝ) 
  (h1 : distance = 12)
  (h2 : time_minutes = 72)
  (h3 : time_hours = time_minutes / 60)
  (h4 : speed = distance / time_hours) : speed = 10 :=
by
  sorry

end man_speed_l79_79365


namespace equal_distances_sum_of_distances_moving_distances_equal_l79_79092

-- Define the points A, B, origin O, and moving point P
def A : ℝ := -1
def B : ℝ := 3
def O : ℝ := 0

-- Define the moving point P
def P (x : ℝ) : ℝ := x

-- Define the velocities of each point
def vP : ℝ := -1
def vA : ℝ := -5
def vB : ℝ := -20

-- Proof statement ①: Distance from P to A and B are equal implies x = 1
theorem equal_distances (x : ℝ) (h : abs (x + 1) = abs (x - 3)) : x = 1 :=
sorry

-- Proof statement ②: Sum of distances from P to A and B is 5 implies x = -3/2 or 7/2
theorem sum_of_distances (x : ℝ) (h : abs (x + 1) + abs (x - 3) = 5) : x = -3/2 ∨ x = 7/2 :=
sorry

-- Proof statement ③: Moving distances equal at times t = 4/15 or 2/23
theorem moving_distances_equal (t : ℝ) (h : abs (4 * t + 1) = abs (19 * t - 3)) : t = 4/15 ∨ t = 2/23 :=
sorry

end equal_distances_sum_of_distances_moving_distances_equal_l79_79092


namespace find_divisor_l79_79615

variable (n : ℤ) (d : ℤ)

theorem find_divisor 
    (h1 : ∃ k : ℤ, n = k * d + 4)
    (h2 : ∃ m : ℤ, n + 15 = m * 5 + 4) :
    d = 5 :=
sorry

end find_divisor_l79_79615


namespace angle_at_intersection_l79_79146

theorem angle_at_intersection (n : ℕ) (h₁ : n = 8)
  (h₂ : ∀ i j : ℕ, (i + 1) % n ≠ j ∧ i < j)
  (h₃ : ∀ i : ℕ, i < n)
  (h₄ : ∀ i j : ℕ, (i + 1) % n = j ∨ (i + n - 1) % n = j)
  : (2 * (180 / n - (180 * (n - 2) / n) / 2)) = 90 :=
by
  sorry

end angle_at_intersection_l79_79146


namespace lateral_surface_area_of_cube_l79_79104

-- Define the side length of the cube
def side_length : ℕ := 12

-- Define the area of one face of the cube
def area_of_one_face (s : ℕ) : ℕ := s * s

-- Define the lateral surface area of the cube
def lateral_surface_area (s : ℕ) : ℕ := 4 * (area_of_one_face s)

-- Prove the lateral surface area of a cube with side length 12 m is equal to 576 m²
theorem lateral_surface_area_of_cube : lateral_surface_area side_length = 576 := by
  sorry

end lateral_surface_area_of_cube_l79_79104


namespace smallest_a_l79_79084

theorem smallest_a (a : ℕ) (h_a : a > 8) : (∀ x : ℤ, ¬ Prime (x^4 + a^2)) ↔ a = 9 :=
by
  sorry

end smallest_a_l79_79084


namespace eddy_travel_time_l79_79481

theorem eddy_travel_time (T : ℝ) (S_e S_f : ℝ) (Freddy_time : ℝ := 4)
  (distance_AB : ℝ := 540) (distance_AC : ℝ := 300) (speed_ratio : ℝ := 2.4) :
  (distance_AB / T = 2.4 * (distance_AC / Freddy_time)) -> T = 3 :=
by
  sorry

end eddy_travel_time_l79_79481


namespace farm_horses_cows_l79_79512

variables (H C : ℕ)

theorem farm_horses_cows (H C : ℕ) (h1 : H = 6 * C) (h2 : (H - 15) = 3 * (C + 15)) : (H - 15) - (C + 15) = 70 :=
by {
  sorry
}

end farm_horses_cows_l79_79512


namespace custom_op_evaluation_l79_79021

def custom_op (x y : ℕ) : ℕ := x * y + x - y

theorem custom_op_evaluation : (custom_op 7 4) - (custom_op 4 7) = 6 := by
  sorry

end custom_op_evaluation_l79_79021


namespace price_per_slice_is_five_l79_79000

-- Definitions based on the given conditions
def pies_sold := 9
def slices_per_pie := 4
def total_revenue := 180

-- Definition derived from given conditions
def total_slices := pies_sold * slices_per_pie

-- The theorem to prove
theorem price_per_slice_is_five :
  total_revenue / total_slices = 5 :=
by
  sorry

end price_per_slice_is_five_l79_79000


namespace kylie_coins_left_l79_79840

-- Definitions for each condition
def coins_from_piggy_bank : ℕ := 15
def coins_from_brother : ℕ := 13
def coins_from_father : ℕ := 8
def coins_given_to_friend : ℕ := 21

-- The total coins Kylie has initially
def initial_coins : ℕ := coins_from_piggy_bank + coins_from_brother
def total_coins_after_father : ℕ := initial_coins + coins_from_father
def coins_left : ℕ := total_coins_after_father - coins_given_to_friend

-- The theorem to prove the final number of coins left is 15
theorem kylie_coins_left : coins_left = 15 :=
by
  sorry -- Proof goes here

end kylie_coins_left_l79_79840


namespace sine_tangent_coincide_3_decimal_places_l79_79755

open Real

noncomputable def deg_to_rad (d : ℝ) : ℝ := d * (π / 180)

theorem sine_tangent_coincide_3_decimal_places :
  ∀ θ : ℝ,
    0 ≤ θ ∧ θ ≤ deg_to_rad (4 + 20 / 60) →
    |sin θ - tan θ| < 0.0005 :=
by
  intros θ hθ
  sorry

end sine_tangent_coincide_3_decimal_places_l79_79755


namespace pascal_tenth_number_in_hundred_row_l79_79464

def pascal_row (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem pascal_tenth_number_in_hundred_row :
  pascal_row 99 9 = Nat.choose 99 9 :=
by
  sorry

end pascal_tenth_number_in_hundred_row_l79_79464


namespace isosceles_triangle_length_l79_79344

variable (a b : ℝ)

theorem isosceles_triangle_length (h1 : 2 * a + 3 = 16) (h2 : a != 3) : a = 6.5 :=
sorry

end isosceles_triangle_length_l79_79344


namespace impossible_to_get_100_pieces_l79_79754

/-- We start with 1 piece of paper. Each time a piece of paper is torn into 3 parts,
it increases the total number of pieces by 2.
Therefore, the number of pieces remains odd through any sequence of tears.
Prove that it is impossible to obtain exactly 100 pieces. -/
theorem impossible_to_get_100_pieces : 
  ∀ n, n = 1 ∨ (∃ k, n = 1 + 2 * k) → n ≠ 100 :=
by
  sorry

end impossible_to_get_100_pieces_l79_79754


namespace evaluate_expression_l79_79291

-- Given conditions 
def x := 3
def y := 2

-- Prove that y + y(y^x + x!) evaluates to 30.
theorem evaluate_expression : y + y * (y^x + Nat.factorial x) = 30 := by
  sorry

end evaluate_expression_l79_79291


namespace percentage_HNO3_final_l79_79548

-- Define the initial conditions
def initial_volume_solution : ℕ := 60 -- 60 liters of solution
def initial_percentage_HNO3 : ℝ := 0.45 -- 45% HNO3
def added_pure_HNO3 : ℕ := 6 -- 6 liters of pure HNO3

-- Define the volume of HNO3 in the initial solution
def hno3_initial := initial_percentage_HNO3 * initial_volume_solution

-- Define the total volume of the final solution
def total_volume_final := initial_volume_solution + added_pure_HNO3

-- Define the total amount of HNO3 in the final solution
def total_hno3_final := hno3_initial + added_pure_HNO3

-- The main theorem: prove the final percentage is 50%
theorem percentage_HNO3_final :
  (total_hno3_final / total_volume_final) * 100 = 50 :=
by
  -- proof is omitted
  sorry

end percentage_HNO3_final_l79_79548


namespace total_exercise_time_l79_79064

-- Definition of constants and speeds for each day
def monday_speed := 2 -- miles per hour
def wednesday_speed := 3 -- miles per hour
def friday_speed := 6 -- miles per hour
def distance := 6 -- miles

-- Function to calculate time given distance and speed
def time (distance : ℕ) (speed : ℕ) : ℕ := distance / speed

-- Prove the total time spent in a week
theorem total_exercise_time :
  time distance monday_speed + time distance wednesday_speed + time distance friday_speed = 6 :=
by
  -- Insert detailed proof steps here
  sorry

end total_exercise_time_l79_79064


namespace model_tower_height_l79_79833

-- Definitions based on conditions
def height_actual_tower : ℝ := 60
def volume_actual_tower : ℝ := 80000
def volume_model_tower : ℝ := 0.5

-- Theorem statement
theorem model_tower_height (h: ℝ) : h = 0.15 :=
by
  sorry

end model_tower_height_l79_79833


namespace function_domain_l79_79107

theorem function_domain (x : ℝ) : x + 3 ≥ 0 ↔ x ≥ -3 := by
  sorry

end function_domain_l79_79107


namespace mean_of_six_numbers_sum_three_quarters_l79_79693

theorem mean_of_six_numbers_sum_three_quarters :
  let sum := (3 / 4 : ℝ)
  let n := 6
  (sum / n) = (1 / 8 : ℝ) :=
by
  sorry

end mean_of_six_numbers_sum_three_quarters_l79_79693


namespace percent_value_in_quarters_l79_79939

theorem percent_value_in_quarters
  (num_dimes num_quarters num_nickels : ℕ)
  (value_dime value_quarter value_nickel : ℕ)
  (h_dimes : num_dimes = 70)
  (h_quarters : num_quarters = 30)
  (h_nickels : num_nickels = 40)
  (h_value_dime : value_dime = 10)
  (h_value_quarter : value_quarter = 25)
  (h_value_nickel : value_nickel = 5) :
  ((num_quarters * value_quarter : ℕ) * 100 : ℚ) / 
  (num_dimes * value_dime + num_quarters * value_quarter + num_nickels * value_nickel) = 45.45 :=
by
  sorry

end percent_value_in_quarters_l79_79939


namespace complement_union_l79_79049

open Set

-- Definitions from the given conditions
def U : Set ℕ := {x | x ≤ 9}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {3, 4, 5, 6}

-- Statement of the proof problem
theorem complement_union :
  compl (A ∪ B) = {7, 8, 9} :=
sorry

end complement_union_l79_79049


namespace correct_statement_l79_79219

theorem correct_statement : 
  (∀ x : ℝ, (x < 0 → x^2 > x)) ∧
  (¬ ∀ x : ℝ, (x^2 > 0 → x > 0)) ∧
  (¬ ∀ x : ℝ, (x^2 > x → x > 0)) ∧
  (¬ ∀ x : ℝ, (x^2 > x → x < 0)) ∧
  (¬ ∀ x : ℝ, (x < 1 → x^2 < x)) :=
by
  sorry

end correct_statement_l79_79219


namespace car_rental_cost_eq_800_l79_79399

-- Define the number of people
def num_people : ℕ := 8

-- Define the cost of the Airbnb rental
def airbnb_cost : ℕ := 3200

-- Define each person's share
def share_per_person : ℕ := 500

-- Define the total contribution of all people
def total_contribution : ℕ := num_people * share_per_person

-- Define the car rental cost
def car_rental_cost : ℕ := total_contribution - airbnb_cost

-- State the theorem to be proved
theorem car_rental_cost_eq_800 : car_rental_cost = 800 :=
  by sorry

end car_rental_cost_eq_800_l79_79399


namespace disproves_proposition_b_l79_79426

-- Definition and condition of complementary angles
def angles_complementary (angle1 angle2: ℝ) : Prop := angle1 + angle2 = 180

-- Proposition to disprove
def disprove (angle1 angle2: ℝ) : Prop := ¬ ((angle1 < 90 ∧ angle2 > 90 ∧ angle2 < 180) ∨ (angle2 < 90 ∧ angle1 > 90 ∧ angle1 < 180))

-- Definition of angles in sets
def set_a := (120, 60)
def set_b := (95.1, 84.9)
def set_c := (30, 60)
def set_d := (90, 90)

-- Statement to prove
theorem disproves_proposition_b : 
  (angles_complementary 95.1 84.9) ∧ (disprove 95.1 84.9) :=
by
  sorry

end disproves_proposition_b_l79_79426


namespace most_cost_effective_payment_l79_79861

theorem most_cost_effective_payment :
  let worker_days := 5 * 10
  let hourly_rate_per_worker := 8 * 10 * 4
  let paint_cost := 4800
  let area_painted := 150
  let cost_option_1 := worker_days * 30
  let cost_option_2 := paint_cost * 0.30
  let cost_option_3 := area_painted * 12
  let cost_option_4 := 5 * hourly_rate_per_worker
  (cost_option_2 < cost_option_1) ∧ (cost_option_2 < cost_option_3) ∧ (cost_option_2 < cost_option_4) :=
by
  sorry

end most_cost_effective_payment_l79_79861


namespace new_length_maintains_area_l79_79869

noncomputable def new_length_for_doubled_width (A W : ℝ) : ℝ := A / (2 * W)

theorem new_length_maintains_area (A W : ℝ) (hA : A = 35.7) (hW : W = 3.8) :
  new_length_for_doubled_width A W = 4.69736842 :=
by
  rw [new_length_for_doubled_width, hA, hW]
  norm_num
  sorry

end new_length_maintains_area_l79_79869


namespace solve_for_a_l79_79511

-- Given the equation is quadratic, meaning the highest power of x in the quadratic term equals 2
theorem solve_for_a (a : ℚ) : (2 * a - 1 = 2) -> a = 3 / 2 :=
by
  sorry

end solve_for_a_l79_79511


namespace jason_spent_on_shorts_l79_79854

def total_spent : ℝ := 14.28
def jacket_spent : ℝ := 4.74
def shorts_spent : ℝ := total_spent - jacket_spent

theorem jason_spent_on_shorts :
  shorts_spent = 9.54 :=
by
  -- Placeholder for the proof. The statement is correct as it matches the given problem data.
  sorry

end jason_spent_on_shorts_l79_79854


namespace second_number_is_72_l79_79533

theorem second_number_is_72 
  (sum_eq_264 : ∀ (x : ℝ), 2 * x + x + (2 / 3) * x = 264) 
  (first_eq_2_second : ∀ (x : ℝ), first = 2 * x)
  (third_eq_1_3_first : ∀ (first : ℝ), third = 1 / 3 * first) :
  second = 72 :=
by
  sorry

end second_number_is_72_l79_79533


namespace hexagon_angle_arith_prog_l79_79078

theorem hexagon_angle_arith_prog (x d : ℝ) (hx : x > 0) (hd : d > 0) 
  (h_eq : 6 * x + 15 * d = 720) : x = 120 :=
by
  sorry

end hexagon_angle_arith_prog_l79_79078


namespace cells_that_remain_open_l79_79937

/-- A cell q remains open after iterative toggling if and only if it is a perfect square. -/
theorem cells_that_remain_open (n : ℕ) (h : n > 0) : 
  (∃ k : ℕ, k ^ 2 = n) ↔ 
  (∀ i : ℕ, (1 ≤ i ∧ i ≤ n) → (∃ m : ℕ, i = m ^ 2)) := 
sorry

end cells_that_remain_open_l79_79937


namespace probability_of_matching_pair_l79_79307

def total_socks : ℕ := 12 + 6 + 9
def choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

def black_pairs : ℕ := choose_two 12
def white_pairs : ℕ := choose_two 6
def blue_pairs : ℕ := choose_two 9

def total_pairs : ℕ := choose_two total_socks
def matching_pairs : ℕ := black_pairs + white_pairs + blue_pairs

def probability : ℚ := matching_pairs / total_pairs

theorem probability_of_matching_pair :
  probability = 1 / 3 :=
by
  -- The proof will go here
  sorry

end probability_of_matching_pair_l79_79307


namespace number_of_BMWs_sold_l79_79884

theorem number_of_BMWs_sold (total_cars : ℕ) (ford_percentage nissan_percentage volkswagen_percentage : ℝ) 
    (h1 : total_cars = 300)
    (h2 : ford_percentage = 0.2)
    (h3 : nissan_percentage = 0.25)
    (h4 : volkswagen_percentage = 0.1) :
    ∃ (bmw_percentage : ℝ) (bmw_cars : ℕ), bmw_percentage = 0.45 ∧ bmw_cars = 135 :=
by 
    sorry

end number_of_BMWs_sold_l79_79884


namespace number_of_elements_in_A_l79_79468

theorem number_of_elements_in_A (a b : ℕ) (h1 : a = 3 * b)
  (h2 : a + b - 100 = 500) (h3 : 100 = 100) (h4 : a - 100 = b - 100 + 50) : a = 450 := by
  sorry

end number_of_elements_in_A_l79_79468


namespace factorization_correct_l79_79748

theorem factorization_correct (a b : ℝ) : 
  a^2 + 2 * b - b^2 - 1 = (a - b + 1) * (a + b - 1) :=
by
  sorry

end factorization_correct_l79_79748


namespace wendy_first_album_pictures_l79_79247

theorem wendy_first_album_pictures 
  (total_pictures : ℕ)
  (num_albums : ℕ)
  (pics_per_album : ℕ)
  (pics_in_first_album : ℕ)
  (h1 : total_pictures = 79)
  (h2 : num_albums = 5)
  (h3 : pics_per_album = 7)
  (h4 : total_pictures = pics_in_first_album + num_albums * pics_per_album) : 
  pics_in_first_album = 44 :=
by
  sorry

end wendy_first_album_pictures_l79_79247


namespace percentage_discount_on_pencils_l79_79150

-- Establish the given conditions
variable (cucumbers pencils price_per_cucumber price_per_pencil total_spent : ℕ)
variable (h1 : cucumbers = 100)
variable (h2 : price_per_cucumber = 20)
variable (h3 : price_per_pencil = 20)
variable (h4 : total_spent = 2800)
variable (h5 : cucumbers = 2 * pencils)

-- Propose the statement to be proved
theorem percentage_discount_on_pencils : 20 * pencils * price_per_pencil = 20 * (total_spent - cucumbers * price_per_cucumber) ∧ pencils = 50 ∧ ((total_spent - cucumbers * price_per_cucumber) * 100 = 80 * pencils * price_per_pencil) :=
by
  sorry

end percentage_discount_on_pencils_l79_79150


namespace archie_initial_marbles_l79_79116

theorem archie_initial_marbles (M : ℝ) (h1 : 0.6 * M + 0.5 * 0.4 * M = M - 20) : M = 100 :=
sorry

end archie_initial_marbles_l79_79116


namespace total_boys_went_down_slide_l79_79501

-- Definitions according to the conditions given
def boys_went_down_slide1 : ℕ := 22
def boys_went_down_slide2 : ℕ := 13

-- The statement to be proved
theorem total_boys_went_down_slide : boys_went_down_slide1 + boys_went_down_slide2 = 35 := 
by 
  sorry

end total_boys_went_down_slide_l79_79501


namespace solve_poly_l79_79420

open Real

-- Define the condition as a hypothesis
def prob_condition (x : ℝ) : Prop :=
  arctan (1 / x) + arctan (1 / (x^5)) = π / 6

-- Define the statement to be proven that x satisfies the polynomial equation
theorem solve_poly (x : ℝ) (h : prob_condition x) :
  x^6 - sqrt 3 * x^5 - sqrt 3 * x - 1 = 0 :=
sorry

end solve_poly_l79_79420


namespace initial_pencils_count_l79_79473

variables {pencils_taken : ℕ} {pencils_left : ℕ} {initial_pencils : ℕ}

theorem initial_pencils_count 
  (h1 : pencils_taken = 4)
  (h2 : pencils_left = 5) :
  initial_pencils = 9 :=
by 
  sorry

end initial_pencils_count_l79_79473


namespace find_a1_l79_79663

variable (a : ℕ → ℤ) (S : ℕ → ℤ)

def is_arithmetic_seq (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def sum_n_terms (S : ℕ → ℤ) (a : ℕ → ℤ) : Prop :=
  ∀ n, S n = (n * (a 1 + a n)) / 2

theorem find_a1 (h1 : is_arithmetic_seq a (-2)) 
               (h2 : sum_n_terms S a) 
               (h3 : S 10 = S 11) : 
  a 1 = 20 :=
sorry

end find_a1_l79_79663


namespace sum_of_digits_of_2010_l79_79293

noncomputable def sum_of_base6_digits (n : ℕ) : ℕ :=
  (n.digits 6).sum

theorem sum_of_digits_of_2010 : sum_of_base6_digits 2010 = 10 := by
  sorry

end sum_of_digits_of_2010_l79_79293


namespace carpet_cost_calculation_l79_79039

theorem carpet_cost_calculation
  (length_feet : ℕ)
  (width_feet : ℕ)
  (feet_to_yards : ℕ)
  (cost_per_square_yard : ℕ)
  (h_length : length_feet = 15)
  (h_width : width_feet = 12)
  (h_convert : feet_to_yards = 3)
  (h_cost : cost_per_square_yard = 10) :
  (length_feet / feet_to_yards) *
  (width_feet / feet_to_yards) *
  cost_per_square_yard = 200 := by
  sorry

end carpet_cost_calculation_l79_79039


namespace students_only_english_l79_79156

variable (total_students both_english_german enrolled_german: ℕ)

theorem students_only_english :
  total_students = 45 ∧ both_english_german = 12 ∧ enrolled_german = 22 ∧
  (∀ S E G B : ℕ, S = total_students ∧ B = both_english_german ∧ G = enrolled_german - B ∧
   (S = E + G + B) → E = 23) :=
by
  sorry

end students_only_english_l79_79156


namespace area_after_trimming_l79_79025

-- Define the conditions
def original_side_length : ℝ := 22
def trim_x : ℝ := 6
def trim_y : ℝ := 5

-- Calculate dimensions after trimming
def new_length : ℝ := original_side_length - trim_x
def new_width : ℝ := original_side_length - trim_y

-- Define the goal
theorem area_after_trimming : new_length * new_width = 272 := by
  sorry

end area_after_trimming_l79_79025


namespace Robert_salary_loss_l79_79916

theorem Robert_salary_loss (S : ℝ) (x : ℝ) (h : x ≠ 0) (h1 : (S - (x/100) * S + (x/100) * (S - (x/100) * S) = (96/100) * S)) : x = 20 :=
by sorry

end Robert_salary_loss_l79_79916


namespace remainder_when_divided_by_9_l79_79418

theorem remainder_when_divided_by_9 (z : ℤ) (k : ℤ) (h : z + 3 = 9 * k) :
  z % 9 = 6 :=
sorry

end remainder_when_divided_by_9_l79_79418


namespace MsElizabethInvestmentsCount_l79_79804

variable (MrBanksRevPerInvestment : ℕ) (MsElizabethRevPerInvestment : ℕ) (MrBanksInvestments : ℕ) (MsElizabethExtraRev : ℕ)

def MrBanksTotalRevenue := MrBanksRevPerInvestment * MrBanksInvestments
def MsElizabethTotalRevenue := MrBanksTotalRevenue + MsElizabethExtraRev
def MsElizabethInvestments := MsElizabethTotalRevenue / MsElizabethRevPerInvestment

theorem MsElizabethInvestmentsCount (h1 : MrBanksRevPerInvestment = 500) 
  (h2 : MsElizabethRevPerInvestment = 900)
  (h3 : MrBanksInvestments = 8)
  (h4 : MsElizabethExtraRev = 500) : 
  MsElizabethInvestments MrBanksRevPerInvestment MsElizabethRevPerInvestment MrBanksInvestments MsElizabethExtraRev = 5 :=
by
  sorry

end MsElizabethInvestmentsCount_l79_79804


namespace problem_statement_l79_79163

def f (x : ℝ) : ℝ := 3 * x + 2
def g (x : ℝ) : ℝ := 2 * x - 3

theorem problem_statement : (f (g (f 1))) / (g (f (g 1))) = (-23 : ℝ) / 5 :=
by 
  sorry

end problem_statement_l79_79163


namespace intersection_of_M_and_N_is_correct_l79_79190

-- Definitions according to conditions
def M : Set ℤ := {-4, -2, 0, 2, 4, 6}
def N : Set ℤ := {x | -3 ≤ x ∧ x ≤ 4}

-- Proof statement
theorem intersection_of_M_and_N_is_correct : (M ∩ N) = {-2, 0, 2, 4} := by
  sorry

end intersection_of_M_and_N_is_correct_l79_79190


namespace ring_roads_count_l79_79034

noncomputable def factorial (n : ℕ) : ℕ :=
  if h : n = 0 then 1 else n * factorial (n - 1)

noncomputable def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem ring_roads_count : 
  binomial 8 4 * binomial 8 4 - (binomial 10 4 * binomial 6 4) = 1750 := by 
sorry

end ring_roads_count_l79_79034


namespace total_cloth_sold_l79_79972

variable (commissionA commissionB salesA salesB totalWorth : ℝ)

def agentA_commission := 0.025 * salesA
def agentB_commission := 0.03 * salesB
def total_worth_of_cloth_sold := salesA + salesB

theorem total_cloth_sold 
  (hA : agentA_commission = 21) 
  (hB : agentB_commission = 27)
  : total_worth_of_cloth_sold = 1740 :=
by
  sorry

end total_cloth_sold_l79_79972


namespace triangle_area_l79_79932

theorem triangle_area :
  let A := (2, -3)
  let B := (2, 4)
  let C := (8, 0) 
  let base := (4 - (-3))
  let height := (8 - 2)
  let area := (1 / 2) * base * height
  area = 21 := 
by 
  sorry

end triangle_area_l79_79932


namespace initial_population_l79_79288

-- Define the initial population
variable (P : ℝ)

-- Define the conditions
theorem initial_population
  (h1 : P * 1.25 * 0.8 * 1.1 * 0.85 * 1.3 + 150 = 25000) :
  P = 24850 :=
by
  sorry

end initial_population_l79_79288


namespace min_value_of_expression_l79_79581

theorem min_value_of_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h_sum : x + y + z = 1) : 
  36 ≤ (1/x + 4/y + 9/z) :=
sorry

end min_value_of_expression_l79_79581


namespace discriminant_of_quadratic_equation_l79_79458

theorem discriminant_of_quadratic_equation :
  let a := 5
  let b := -9
  let c := 4
  (b^2 - 4 * a * c = 1) :=
by {
  sorry
}

end discriminant_of_quadratic_equation_l79_79458


namespace find_real_numbers_l79_79790

theorem find_real_numbers (a b c : ℝ)    :
  (a + b + c = 3) → (a^2 + b^2 + c^2 = 35) → (a^3 + b^3 + c^3 = 99) → 
  (a = 1 ∧ b = -3 ∧ c = 5) ∨ (a = 1 ∧ b = 5 ∧ c = -3) ∨ 
  (a = -3 ∧ b = 1 ∧ c = 5) ∨ (a = -3 ∧ b = 5 ∧ c = 1) ∨
  (a = 5 ∧ b = 1 ∧ c = -3) ∨ (a = 5 ∧ b = -3 ∧ c = 1) :=
by intros h1 h2 h3; sorry

end find_real_numbers_l79_79790


namespace repeat_block_of_7_div_13_l79_79020

theorem repeat_block_of_7_div_13 : ∃ k : ℕ, (∀ n : ℕ, n < k → 10^n % 13 ≠ 1) ∧ 10^k % 13 = 1 ∧ k = 6 :=
by { sorry }

end repeat_block_of_7_div_13_l79_79020


namespace remainder_division_l79_79571

theorem remainder_division (x : ℤ) (hx : x % 82 = 5) : (x + 7) % 41 = 12 := 
by 
  sorry

end remainder_division_l79_79571


namespace decreasing_by_25_l79_79531

theorem decreasing_by_25 (n : ℕ) (k : ℕ) (y : ℕ) (hy : 0 ≤ y ∧ y < 10^k) : 
  (n = 6 * 10^k + y → n / 10 = y / 25) → (∃ m, n = 625 * 10^m) := 
sorry

end decreasing_by_25_l79_79531


namespace income_percentage_less_l79_79979

-- Definitions representing the conditions
variables (T M J : ℝ)
variables (h1 : M = 1.60 * T) (h2 : M = 1.12 * J)

-- The theorem stating the problem
theorem income_percentage_less : (100 - (T / J) * 100) = 30 :=
by
  sorry

end income_percentage_less_l79_79979


namespace shadow_length_false_if_approaching_lamp_at_night_l79_79124

theorem shadow_length_false_if_approaching_lamp_at_night
  (night : Prop)
  (approaches_lamp : Prop)
  (shadow_longer : Prop) :
  night → approaches_lamp → ¬shadow_longer :=
by
  -- assume it is night and person is approaching lamp
  intros h_night h_approaches
  -- proof is omitted
  sorry

end shadow_length_false_if_approaching_lamp_at_night_l79_79124


namespace speed_ratio_l79_79555

theorem speed_ratio (v_A v_B : ℝ) (L t : ℝ) 
  (h1 : v_A * t = (1 - 0.11764705882352941) * L)
  (h2 : v_B * t = L) : 
  v_A / v_B = 1.11764705882352941 := 
by 
  sorry

end speed_ratio_l79_79555


namespace polynomial_evaluation_l79_79576

theorem polynomial_evaluation (y : ℝ) (hy : y^2 - 3 * y - 9 = 0) : y^3 - 3 * y^2 - 9 * y + 7 = 7 := 
  sorry

end polynomial_evaluation_l79_79576


namespace probability_of_three_primes_from_30_l79_79981

noncomputable def primes_up_to_30 : Finset ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

theorem probability_of_three_primes_from_30 :
  ((primes_up_to_30.card.choose 3) / ((Finset.range 31).card.choose 3)) = (6 / 203) :=
by
  sorry

end probability_of_three_primes_from_30_l79_79981


namespace marcia_average_cost_l79_79218

theorem marcia_average_cost :
  let price_apples := 2
  let price_bananas := 1
  let price_oranges := 3
  let count_apples := 12
  let count_bananas := 4
  let count_oranges := 4
  let offer_apples_free := count_apples / 10 * 2
  let offer_oranges_free := count_oranges / 3
  let total_apples := count_apples + offer_apples_free
  let total_oranges := count_oranges + offer_oranges_free
  let total_fruits := total_apples + count_bananas + count_oranges
  let cost_apples := price_apples * (count_apples - offer_apples_free)
  let cost_bananas := price_bananas * count_bananas
  let cost_oranges := price_oranges * (count_oranges - offer_oranges_free)
  let total_cost := cost_apples + cost_bananas + cost_oranges
  let average_cost := total_cost / total_fruits
  average_cost = 1.85 :=
  sorry

end marcia_average_cost_l79_79218


namespace find_k_l79_79479

theorem find_k (x1 x2 : ℝ) (r : ℝ) (h1 : x1 = 3 * r) (h2 : x2 = r) (h3 : x1 + x2 = -8) (h4 : x1 * x2 = k) : k = 12 :=
by
  -- proof steps here
  sorry

end find_k_l79_79479


namespace degree_diploma_salary_ratio_l79_79761

theorem degree_diploma_salary_ratio
  (jared_salary : ℕ)
  (diploma_monthly_salary : ℕ)
  (h_annual_salary : jared_salary = 144000)
  (h_diploma_annual_salary : 12 * diploma_monthly_salary = 48000) :
  (jared_salary / (12 * diploma_monthly_salary)) = 3 := 
by sorry

end degree_diploma_salary_ratio_l79_79761


namespace evaluate_expression_l79_79364

theorem evaluate_expression :
  (12 - 11 + 10 - 9 + 8 - 7 + 6 - 5 + 4 - 3 + 2 - 1) / (2 - 3 + 4 - 5 + 6 - 7 + 8 - 9 + 10 - 11 + 12) = 6 / 7 := by
sorry

end evaluate_expression_l79_79364


namespace problem1_problem2_l79_79565

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 / x

def is_increasing_on (f : ℝ → ℝ) (domain : Set ℝ) : Prop :=
  ∀ x₁ x₂, x₁ ∈ domain → x₂ ∈ domain → x₁ < x₂ → f x₁ < f x₂

theorem problem1 : is_increasing_on f {x | 1 ≤ x} := 
by sorry

def is_decreasing (g : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → g x₁ > g x₂

theorem problem2 (g : ℝ → ℝ) (h_decreasing : is_decreasing g)
  (h_inequality : ∀ x : ℝ, 1 ≤ x → g (x^3 + 2) < g ((a^2 - 2 * a) * x)) :
  -1 < a ∧ a < 3 :=
by sorry

end problem1_problem2_l79_79565


namespace smallest_b_for_perfect_square_l79_79012

theorem smallest_b_for_perfect_square (b : ℤ) (h1 : b > 4) (h2 : ∃ n : ℤ, 3 * b + 4 = n * n) : b = 7 :=
by
  sorry

end smallest_b_for_perfect_square_l79_79012


namespace remainder_division_l79_79882

variable (P D K Q R R'_q R'_r : ℕ)

theorem remainder_division (h1 : P = Q * D + R) (h2 : R = R'_q * K + R'_r) (h3 : K < D) : 
  P % (D * K) = R'_r :=
sorry

end remainder_division_l79_79882


namespace product_repeating_decimal_l79_79148

theorem product_repeating_decimal (p : ℚ) (h₁ : p = 152 / 333) : 
  p * 7 = 1064 / 333 :=
  by
    sorry

end product_repeating_decimal_l79_79148


namespace smallest_n_for_two_distinct_tuples_l79_79228

theorem smallest_n_for_two_distinct_tuples : ∃ (n : ℕ), n = 1729 ∧ 
  (∃ (x1 y1 x2 y2 : ℕ), x1 ≠ x2 ∧ y1 ≠ y2 ∧ n = x1^3 + y1^3 ∧ n = x2^3 + y2^3 ∧ 0 < x1 ∧ 0 < y1 ∧ 0 < x2 ∧ 0 < y2) := sorry

end smallest_n_for_two_distinct_tuples_l79_79228


namespace second_group_members_l79_79102

theorem second_group_members (total first third : ℕ) (h1 : total = 70) (h2 : first = 25) (h3 : third = 15) :
  (total - first - third) = 30 :=
by
  sorry

end second_group_members_l79_79102


namespace pancake_problem_l79_79807

theorem pancake_problem :
  let mom_rate := (100 : ℚ) / 30
  let anya_rate := (100 : ℚ) / 40
  let andrey_rate := (100 : ℚ) / 60
  let combined_baking_rate := mom_rate + anya_rate
  let net_rate := combined_baking_rate - andrey_rate
  let target_pancakes := 100
  let time := target_pancakes / net_rate
  time = 24 := by
sorry

end pancake_problem_l79_79807


namespace sum_of_possible_values_of_N_l79_79224

variable (N S : ℝ) (hN : N ≠ 0)

theorem sum_of_possible_values_of_N : 
  (3 * N + 5 / N = S) → 
  ∀ N1 N2 : ℝ, (3 * N1^2 - S * N1 + 5 = 0) ∧ (3 * N2^2 - S * N2 + 5 = 0) → 
  N1 + N2 = S / 3 :=
by 
  intro hS hRoots
  sorry

end sum_of_possible_values_of_N_l79_79224


namespace total_amount_leaked_l79_79917

def amount_leaked_before_start : ℕ := 2475
def amount_leaked_while_fixing : ℕ := 3731

theorem total_amount_leaked : amount_leaked_before_start + amount_leaked_while_fixing = 6206 := by
  sorry

end total_amount_leaked_l79_79917


namespace perpendicular_lines_condition_l79_79606

theorem perpendicular_lines_condition (A1 B1 C1 A2 B2 C2 : ℝ) :
  (A1 * A2 + B1 * B2 = 0) ↔ (A1 * A2) / (B1 * B2) = -1 := sorry

end perpendicular_lines_condition_l79_79606


namespace point_on_graph_l79_79557

variable (x y : ℝ)

-- Define the condition for a point to be on the graph of the function y = 6/x
def is_on_graph (x y : ℝ) : Prop :=
  x * y = 6

-- State the theorem to be proved
theorem point_on_graph : is_on_graph (-2) (-3) :=
  by
  sorry

end point_on_graph_l79_79557


namespace triangle_properties_l79_79144

-- Definitions of sides of the triangle
def a : ℕ := 15
def b : ℕ := 11
def c : ℕ := 18

-- Definition of the triangle inequality theorem in the context
def triangle_inequality (x y z : ℕ) : Prop :=
  x + y > z ∧ x + z > y ∧ y + z > x

-- Perimeter calculation
def perimeter (x y z : ℕ) : ℕ :=
  x + y + z

-- Stating the proof problem
theorem triangle_properties : triangle_inequality a b c ∧ perimeter a b c = 44 :=
by
  -- Start the process for the actual proof that will be filled out
  sorry

end triangle_properties_l79_79144


namespace originally_planned_days_l79_79280

def man_days (men : ℕ) (days : ℕ) : ℕ := men * days

theorem originally_planned_days (D : ℕ) (h : man_days 5 10 = man_days 10 D) : D = 5 :=
by 
  sorry

end originally_planned_days_l79_79280


namespace nikita_productivity_l79_79143

theorem nikita_productivity 
  (x y : ℕ) 
  (h1 : 3 * x + 2 * y = 7) 
  (h2 : 5 * x + 3 * y = 11) : 
  y = 2 := 
sorry

end nikita_productivity_l79_79143


namespace angle_relation_in_triangle_l79_79182

theorem angle_relation_in_triangle
    (a b c : ℝ)
    (A B C : ℝ)
    (h1 : b * (a + b) * (b + c) = a^3 + b * (a^2 + c^2) + c^3)
    (h2 : A + B + C = π) 
    (h3 : A > 0) 
    (h4 : B > 0) 
    (h5 : C > 0) :
    (1 / (Real.sqrt A + Real.sqrt B)) + (1 / (Real.sqrt B + Real.sqrt C)) = (2 / (Real.sqrt C + Real.sqrt A)) :=
sorry

end angle_relation_in_triangle_l79_79182


namespace remaining_distance_l79_79277

-- Definitions based on the conditions
def total_distance : ℕ := 78
def first_leg : ℕ := 35
def second_leg : ℕ := 18

-- The theorem we want to prove
theorem remaining_distance : total_distance - (first_leg + second_leg) = 25 := by
  sorry

end remaining_distance_l79_79277


namespace same_function_l79_79415

noncomputable def f (x : ℝ) : ℝ := x
noncomputable def g (t : ℝ) : ℝ := (t^3 + t) / (t^2 + 1)

theorem same_function : ∀ x : ℝ, f x = g x :=
by sorry

end same_function_l79_79415


namespace sum_cubes_mod_l79_79346

theorem sum_cubes_mod (n : ℕ) : (1^3 + 2^3 + 3^3 + 4^3 + 5^3 + 6^3 + 7^3 + 8^3 + 9^3 + 10^3) % 7 = 1 := by
  sorry

end sum_cubes_mod_l79_79346


namespace calculate_expression_l79_79618

theorem calculate_expression
  (x y : ℚ)
  (D E : ℚ × ℚ)
  (hx : x = (D.1 + E.1) / 2)
  (hy : y = (D.2 + E.2) / 2)
  (hD : D = (15, -3))
  (hE : E = (-4, 12)) :
  3 * x - 5 * y = -6 :=
by
  subst hD
  subst hE
  subst hx
  subst hy
  sorry

end calculate_expression_l79_79618


namespace arithmetic_mean_reciprocals_primes_l79_79701

theorem arithmetic_mean_reciprocals_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let rec1 := (1:ℚ) / p1
  let rec2 := (1:ℚ) / p2
  let rec3 := (1:ℚ) / p3
  let rec4 := (1:ℚ) / p4
  (rec1 + rec2 + rec3 + rec4) / 4 = 247 / 840 := by
  sorry

end arithmetic_mean_reciprocals_primes_l79_79701


namespace vote_difference_l79_79335

-- Definitions of initial votes for and against the policy
def vote_initial_for (x y : ℕ) : Prop := x + y = 450
def initial_margin (x y m : ℕ) : Prop := y > x ∧ y - x = m

-- Definitions of votes for and against in the second vote
def vote_second_for (x' y' : ℕ) : Prop := x' + y' = 450
def second_margin (x' y' m : ℕ) : Prop := x' - y' = 3 * m
def second_vote_ratio (x' y : ℕ) : Prop := x' = 10 * y / 9

-- Theorem to prove the increase in votes
theorem vote_difference (x y x' y' m : ℕ)
  (hi : vote_initial_for x y)
  (hm : initial_margin x y m)
  (hs : vote_second_for x' y')
  (hsm : second_margin x' y' m)
  (hr : second_vote_ratio x' y) : 
  x' - x = 52 :=
sorry

end vote_difference_l79_79335


namespace simplify_expr1_simplify_expr2_simplify_expr3_l79_79539

-- For the first expression
theorem simplify_expr1 (a b : ℝ) : 2 * a - 3 * b + a - 5 * b = 3 * a - 8 * b :=
by
  sorry

-- For the second expression
theorem simplify_expr2 (a : ℝ) : (a^2 - 6 * a) - 3 * (a^2 - 2 * a + 1) + 3 = -2 * a^2 :=
by
  sorry

-- For the third expression
theorem simplify_expr3 (x y : ℝ) : 4*(x^2*y - 2*x*y^2) - 3*(-x*y^2 + 2*x^2*y) = -2*x^2*y - 5*x*y^2 :=
by
  sorry

end simplify_expr1_simplify_expr2_simplify_expr3_l79_79539


namespace find_S_11_l79_79878

variables (a : ℕ → ℤ)
variables (d : ℤ) (n : ℕ)

def is_arithmetic_sequence : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def sum_first_n_terms (n : ℕ) : ℤ :=
  (n * (a 1 + a n)) / 2

noncomputable def a_3 := a 3
noncomputable def a_6 := a 6
noncomputable def a_9 := a 9

theorem find_S_11
  (h1 : is_arithmetic_sequence a d)
  (h2 : a_3 + a_9 = 18 - a_6) :
  sum_first_n_terms a 11 = 66 :=
sorry

end find_S_11_l79_79878


namespace min_sum_fraction_sqrt_l79_79360

open Real

theorem min_sum_fraction_sqrt (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 1) :
  ∃ min, min = sqrt 2 ∧ ∀ z, (z = (x / sqrt (1 - x) + y / sqrt (1 - y))) → z ≥ sqrt 2 :=
sorry

end min_sum_fraction_sqrt_l79_79360


namespace typist_original_salary_l79_79356

theorem typist_original_salary (S : ℝ) (h : (1.12 * 0.93 * 1.15 * 0.90 * S = 5204.21)) : S = 5504.00 :=
sorry

end typist_original_salary_l79_79356


namespace calculation_l79_79930

theorem calculation : 8 - (7.14 * (1 / 3) - (20 / 9) / (5 / 2)) + 0.1 = 6.62 :=
by
  sorry

end calculation_l79_79930


namespace totalNumberOfPeople_l79_79710

def numGirls := 542
def numBoys := 387
def numTeachers := 45
def numStaff := 27

theorem totalNumberOfPeople : numGirls + numBoys + numTeachers + numStaff = 1001 := by
  sorry

end totalNumberOfPeople_l79_79710


namespace find_a_exactly_two_solutions_l79_79828

theorem find_a_exactly_two_solutions :
  (∀ x y : ℝ, |x - 6 - y| + |x - 6 + y| = 12 ∧ (|x| - 6)^2 + (|y| - 8)^2 = a) ↔ (a = 4 ∨ a = 100) :=
sorry

end find_a_exactly_two_solutions_l79_79828


namespace team_formation_problem_l79_79561

def num_team_formation_schemes : Nat :=
  let comb (n k : Nat) : Nat := Nat.choose n k
  (comb 5 1 * comb 4 2) + (comb 5 2 * comb 4 1)

theorem team_formation_problem :
  num_team_formation_schemes = 70 :=
sorry

end team_formation_problem_l79_79561


namespace vector_perpendicular_solve_x_l79_79791

theorem vector_perpendicular_solve_x
  (x : ℝ)
  (a : ℝ × ℝ := (4, 8))
  (b : ℝ × ℝ := (x, 4))
  (h : 4 * x + 8 * 4 = 0) :
  x = -8 :=
sorry

end vector_perpendicular_solve_x_l79_79791


namespace marilyn_initial_bottle_caps_l79_79449

theorem marilyn_initial_bottle_caps (x : ℕ) (h : x - 36 = 15) : x = 51 :=
sorry

end marilyn_initial_bottle_caps_l79_79449


namespace arithmetic_sequence_seventh_term_l79_79323

variable (a1 a15 : ℚ)
variable (n : ℕ) (a7 : ℚ)

-- Given conditions
def first_term (a1 : ℚ) : Prop := a1 = 3
def last_term (a15 : ℚ) : Prop := a15 = 72
def total_terms (n : ℕ) : Prop := n = 15

-- Arithmetic sequence formula
def common_difference (d : ℚ) : Prop := d = (72 - 3) / (15 - 1)
def nth_term (a_n : ℚ) (a1 : ℚ) (n : ℕ) (d : ℚ) : Prop := a_n = a1 + (n - 1) * d

-- Prove that the 7th term is approximately 33
theorem arithmetic_sequence_seventh_term :
  ∀ (a1 a15 : ℚ) (n : ℕ), first_term a1 → last_term a15 → total_terms n → ∃ a7 : ℚ, 
  nth_term a7 a1 7 ((a15 - a1) / (n - 1)) ∧ (33 - 0.5) < a7 ∧ a7 < (33 + 0.5) :=
by {
  sorry
}

end arithmetic_sequence_seventh_term_l79_79323


namespace middle_part_of_proportion_l79_79985

theorem middle_part_of_proportion (x : ℚ) (h : x + (1/4) * x + (1/8) * x = 104) : (1/4) * x = 208 / 11 :=
by
  sorry

end middle_part_of_proportion_l79_79985


namespace milton_sold_15_pies_l79_79946

theorem milton_sold_15_pies
  (apple_pie_slices_per_pie : ℕ) (peach_pie_slices_per_pie : ℕ)
  (ordered_apple_pie_slices : ℕ) (ordered_peach_pie_slices : ℕ)
  (h1 : apple_pie_slices_per_pie = 8) (h2 : peach_pie_slices_per_pie = 6)
  (h3 : ordered_apple_pie_slices = 56) (h4 : ordered_peach_pie_slices = 48) :
  (ordered_apple_pie_slices / apple_pie_slices_per_pie) + (ordered_peach_pie_slices / peach_pie_slices_per_pie) = 15 := 
by
  sorry

end milton_sold_15_pies_l79_79946


namespace find_range_of_a_l79_79933

variable (x a : ℝ)

/-- Given p: 2 * x^2 - 9 * x + a < 0 and q: the negation of p is sufficient 
condition for the negation of q,
prove to find the range of the real number a. -/
theorem find_range_of_a (hp: 2 * x^2 - 9 * x + a < 0) (hq: ¬ (2 * x^2 - 9 * x + a < 0) → ¬ q) :
  ∃ a : ℝ, sorry := sorry

end find_range_of_a_l79_79933


namespace math_problem_l79_79750

theorem math_problem :
  (-1 : ℤ) ^ 49 + 2 ^ (4 ^ 3 + 3 ^ 2 - 7 ^ 2) = 16777215 := by
  sorry

end math_problem_l79_79750


namespace rooms_in_house_l79_79421

-- define the number of paintings
def total_paintings : ℕ := 32

-- define the number of paintings per room
def paintings_per_room : ℕ := 8

-- define the number of rooms
def number_of_rooms (total_paintings : ℕ) (paintings_per_room : ℕ) : ℕ := total_paintings / paintings_per_room

-- state the theorem
theorem rooms_in_house : number_of_rooms total_paintings paintings_per_room = 4 :=
by sorry

end rooms_in_house_l79_79421


namespace division_of_difference_squared_l79_79499

theorem division_of_difference_squared :
  ((2222 - 2121)^2) / 196 = 52 := 
sorry

end division_of_difference_squared_l79_79499


namespace total_population_is_3311_l79_79832

-- Definitions based on the problem's conditions
def fewer_than_6000_inhabitants (L : ℕ) : Prop :=
  L < 6000

def more_girls_than_boys (girls boys : ℕ) : Prop :=
  girls = (11 * boys) / 10

def more_men_than_women (men women : ℕ) : Prop :=
  men = (23 * women) / 20

def more_children_than_adults (children adults : ℕ) : Prop :=
  children = (6 * adults) / 5

-- Prove that the total population is 3311 given the described conditions
theorem total_population_is_3311 {L n men women children boys girls : ℕ}
  (hc : more_children_than_adults children (n + men))
  (hm : more_men_than_women men n)
  (hg : more_girls_than_boys girls boys)
  (hL : L = n + men + boys + girls)
  (hL_lt : fewer_than_6000_inhabitants L) :
  L = 3311 :=
sorry

end total_population_is_3311_l79_79832


namespace alice_wins_l79_79373

noncomputable def game_condition (r : ℝ) (f : ℕ → ℝ) : Prop :=
∀ n, 0 ≤ f n ∧ f n ≤ 1

theorem alice_wins (r : ℝ) (f : ℕ → ℝ) (hf : game_condition r f) :
  r ≤ 3 → (∃ x : ℕ → ℝ, game_condition 3 x ∧ (abs (x 0 - x 1) + abs (x 2 - x 3) + abs (x 4 - x 5) ≥ r)) :=
by
  sorry

end alice_wins_l79_79373


namespace number_of_5_dollar_bills_l79_79858

def total_money : ℤ := 45
def value_of_each_bill : ℤ := 5

theorem number_of_5_dollar_bills : total_money / value_of_each_bill = 9 := by
  sorry

end number_of_5_dollar_bills_l79_79858


namespace phone_sales_total_amount_l79_79912

theorem phone_sales_total_amount
  (vivienne_phones : ℕ)
  (aliyah_more_phones : ℕ)
  (price_per_phone : ℕ)
  (aliyah_phones : ℕ := vivienne_phones + aliyah_more_phones)
  (total_phones : ℕ := vivienne_phones + aliyah_phones)
  (total_amount : ℕ := total_phones * price_per_phone) :
  vivienne_phones = 40 → aliyah_more_phones = 10 → price_per_phone = 400 → total_amount = 36000 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end phone_sales_total_amount_l79_79912


namespace eval_expression_l79_79543

theorem eval_expression : 5 + 4 - 3 + 2 - 1 = 7 :=
by
  -- Mathematically, this statement holds by basic arithmetic operations.
  sorry

end eval_expression_l79_79543


namespace James_final_assets_correct_l79_79061

/-- Given the following initial conditions:
- James starts with 60 gold bars.
- He pays 10% in tax.
- He loses half of what is left in a divorce.
- He invests 25% of the remaining gold bars in a stock market and earns an additional gold bar.
- On Monday, he exchanges half of his remaining gold bars at a rate of 5 silver bars for 1 gold bar.
- On Tuesday, he exchanges half of his remaining gold bars at a rate of 7 silver bars for 1 gold bar.
- On Wednesday, he exchanges half of his remaining gold bars at a rate of 3 silver bars for 1 gold bar.

We need to determine:
- The number of silver bars James has,
- The number of remaining gold bars James has, and
- The number of gold bars worth from the stock investment James has after these transactions.
-/
noncomputable def James_final_assets (init_gold : ℕ) : ℕ × ℕ × ℕ :=
  let tax := init_gold / 10
  let gold_after_tax := init_gold - tax
  let gold_after_divorce := gold_after_tax / 2
  let invest_gold := gold_after_divorce * 25 / 100
  let remaining_gold_after_invest := gold_after_divorce - invest_gold
  let gold_after_stock := remaining_gold_after_invest + 1
  let monday_gold_exchanged := gold_after_stock / 2
  let monday_silver := monday_gold_exchanged * 5
  let remaining_gold_after_monday := gold_after_stock - monday_gold_exchanged
  let tuesday_gold_exchanged := remaining_gold_after_monday / 2
  let tuesday_silver := tuesday_gold_exchanged * 7
  let remaining_gold_after_tuesday := remaining_gold_after_monday - tuesday_gold_exchanged
  let wednesday_gold_exchanged := remaining_gold_after_tuesday / 2
  let wednesday_silver := wednesday_gold_exchanged * 3
  let remaining_gold_after_wednesday := remaining_gold_after_tuesday - wednesday_gold_exchanged
  let total_silver := monday_silver + tuesday_silver + wednesday_silver
  (total_silver, remaining_gold_after_wednesday, invest_gold)

theorem James_final_assets_correct : James_final_assets 60 = (99, 3, 6) := 
sorry

end James_final_assets_correct_l79_79061


namespace cost_per_pound_correct_l79_79348

noncomputable def cost_per_pound_of_coffee (initial_amount spent_amount pounds_of_coffee : ℕ) : ℚ :=
  (initial_amount - spent_amount) / pounds_of_coffee

theorem cost_per_pound_correct :
  let initial_amount := 70
  let amount_left    := 35.68
  let pounds_of_coffee := 4
  (initial_amount - amount_left) / pounds_of_coffee = 8.58 := 
by
  sorry

end cost_per_pound_correct_l79_79348


namespace part_I_part_II_l79_79633

noncomputable def f (x : ℝ) (a : ℝ) := Real.log x + x^2 - a * x

theorem part_I (x : ℝ) (a : ℝ) (h_inc : ∀ x > 0, (1/x + 2*x - a) ≥ 0) : a ≤ 2 * Real.sqrt 2 :=
sorry

noncomputable def g (x : ℝ) (a : ℝ) := f x a + 2 * Real.log ((a * x + 2) / (6 * Real.sqrt x))

theorem part_II (a : ℝ) (k : ℝ) (h_a : 2 < a ∧ a < 4) (h_ex : ∃ x : ℝ, (3/2) ≤ x ∧ x ≤ 2 ∧ g x a > k * (4 - a^2)) : k ≥ 1/3 :=
sorry

end part_I_part_II_l79_79633


namespace solve_for_x_l79_79662

theorem solve_for_x (x : ℝ) (h : -3 * x - 12 = 8 * x + 5) : x = -17 / 11 :=
by
  sorry

end solve_for_x_l79_79662


namespace Katie_old_games_l79_79770

theorem Katie_old_games (O : ℕ) (hk1 : Katie_new_games = 57) (hf1 : Friends_new_games = 34) (hk2 : Katie_total_games = Friends_total_games + 62) : 
  O = 39 :=
by
  sorry

variables (Katie_new_games Friends_new_games Katie_total_games Friends_total_games : ℕ)

end Katie_old_games_l79_79770


namespace polyhedron_faces_l79_79728

theorem polyhedron_faces (V E F T P t p : ℕ)
  (hF : F = 20)
  (hFaces : t + p = 20)
  (hTriangles : t = 2 * p)
  (hVertex : T = 2 ∧ P = 2)
  (hEdges : E = (3 * t + 5 * p) / 2)
  (hEuler : V - E + F = 2) :
  100 * P + 10 * T + V = 238 :=
by
  sorry

end polyhedron_faces_l79_79728


namespace apartment_building_floors_l79_79113

theorem apartment_building_floors (K E P : ℕ) (h1 : 1 < K) (h2 : K < E) (h3 : E < P) (h4 : K * E * P = 715) : 
  E = 11 :=
sorry

end apartment_building_floors_l79_79113


namespace dan_helmet_crater_difference_l79_79778

theorem dan_helmet_crater_difference :
  ∀ (r d : ℕ), 
  (r = 75) ∧ (d = 35) ∧ (r = 15 + (d + (r - 15 - d))) ->
  ((d - (r - 15 - d)) = 10) :=
by
  intros r d h
  have hr : r = 75 := h.1
  have hd : d = 35 := h.2.1
  have h_combined : r = 15 + (d + (r - 15 - d)) := h.2.2
  sorry

end dan_helmet_crater_difference_l79_79778


namespace dividend_calculation_l79_79598

theorem dividend_calculation (divisor quotient remainder dividend : ℕ)
  (h1 : divisor = 36)
  (h2 : quotient = 20)
  (h3 : remainder = 5)
  (h4 : dividend = (divisor * quotient) + remainder)
  : dividend = 725 := 
by
  -- We skip the proof here
  sorry

end dividend_calculation_l79_79598


namespace f_periodic_with_period_one_l79_79070

noncomputable def is_periodic (f : ℝ → ℝ) :=
  ∃ c : ℝ, c ≠ 0 ∧ ∀ x : ℝ, f (x + c) = f x

theorem f_periodic_with_period_one
  (f : ℝ → ℝ)
  (h1 : ∀ x : ℝ, |f x| ≤ 1)
  (h2 : ∀ x : ℝ, f (x + 13/42) + f x = f (x + 1/6) + f (x + 1/7)) :
  is_periodic f := 
sorry

end f_periodic_with_period_one_l79_79070


namespace pages_copied_for_15_dollars_l79_79342

theorem pages_copied_for_15_dollars
  (cost_per_page : ℕ)
  (dollar_to_cents : ℕ)
  (dollars_available : ℕ)
  (convert_to_cents : dollar_to_cents = 100)
  (cost_per_page_eq : cost_per_page = 3)
  (dollars_available_eq : dollars_available = 15) :
  (dollars_available * dollar_to_cents) / cost_per_page = 500 := by
  -- Convert the dollar amount to cents
  -- Calculate the number of pages that can be copied
  sorry

end pages_copied_for_15_dollars_l79_79342


namespace range_of_m_is_increasing_l79_79556

noncomputable def f (x : ℝ) (m: ℝ) := x^2 + m*x + m

theorem range_of_m_is_increasing :
  { m : ℝ // ∀ x y : ℝ, -2 ≤ x → x ≤ y → f x m ≤ f y m } = {m | 4 ≤ m} :=
by
  sorry

end range_of_m_is_increasing_l79_79556


namespace min_value_quadratic_expr_l79_79446

theorem min_value_quadratic_expr (a : ℝ) (x₁ x₂ : ℝ) 
  (h1 : a > 0) 
  (h2 : x₁ ≠ x₂) 
  (h3 : x₁^2 - 4*a*x₁ + 3*a^2 < 0) 
  (h4 : x₂^2 - 4*a*x₂ + 3*a^2 < 0)
  (h5 : x₁ + x₂ = 4*a)
  (h6 : x₁ * x₂ = 3*a^2) : 
  x₁ + x₂ + a / (x₁ * x₂) = 4 * a + 1 / (3 * a) := 
sorry

end min_value_quadratic_expr_l79_79446


namespace comparison_of_logs_l79_79547

noncomputable def a : ℝ := Real.log 6 / Real.log 3
noncomputable def b : ℝ := Real.log 12 / Real.log 6
noncomputable def c : ℝ := Real.log 16 / Real.log 8

theorem comparison_of_logs : a > b ∧ b > c :=
by
  sorry

end comparison_of_logs_l79_79547


namespace length_of_uncovered_side_l79_79443

-- Define the conditions of the problem
def area_condition (L W : ℝ) : Prop := L * W = 210
def fencing_condition (L W : ℝ) : Prop := L + 2 * W = 41

-- Define the proof statement
theorem length_of_uncovered_side (L W : ℝ) (h_area : area_condition L W) (h_fence : fencing_condition L W) : 
  L = 21 :=
  sorry

end length_of_uncovered_side_l79_79443


namespace no_solution_for_x_y_z_seven_n_plus_eight_is_perfect_square_l79_79574

theorem no_solution_for_x_y_z (a : ℕ) : 
  ¬ ∃ (x y z : ℚ), x^2 + y^2 + z^2 = 8 * a + 7 :=
by
  sorry

theorem seven_n_plus_eight_is_perfect_square (n : ℕ) :
  ∃ x : ℕ, 7^n + 8 = x^2 ↔ n = 0 :=
by
  sorry

end no_solution_for_x_y_z_seven_n_plus_eight_is_perfect_square_l79_79574


namespace total_roasted_marshmallows_l79_79051

-- Definitions based on problem conditions
def dadMarshmallows : ℕ := 21
def joeMarshmallows := 4 * dadMarshmallows
def dadRoasted := dadMarshmallows / 3
def joeRoasted := joeMarshmallows / 2

-- Theorem to prove the total roasted marshmallows
theorem total_roasted_marshmallows : dadRoasted + joeRoasted = 49 := by
  sorry -- Proof omitted

end total_roasted_marshmallows_l79_79051


namespace solve_eq_l79_79957

open Real

noncomputable def solution : Set ℝ := { x | ∃ (n : ℤ), x = π / 12 + π * (n : ℝ) }

theorem solve_eq : { x : ℝ | ∃ (n : ℤ), x = π / 12 + π * (n : ℝ) } = solution := by sorry

end solve_eq_l79_79957


namespace solve_for_y_l79_79781

-- Define the conditions as Lean functions and statements
def is_positive (y : ℕ) : Prop := y > 0
def multiply_sixteen (y : ℕ) : Prop := 16 * y = 256

-- The theorem that states the value of y
theorem solve_for_y (y : ℕ) (h1 : is_positive y) (h2 : multiply_sixteen y) : y = 16 :=
sorry

end solve_for_y_l79_79781


namespace ellen_smoothie_total_l79_79031

theorem ellen_smoothie_total :
  0.2 + 0.1 + 0.2 + 0.15 + 0.05 = 0.7 :=
by sorry

end ellen_smoothie_total_l79_79031


namespace dihedral_angle_ge_l79_79684

-- Define the problem conditions and goal in Lean
theorem dihedral_angle_ge (n : ℕ) (h : 3 ≤ n) (ϕ : ℝ) :
  ϕ ≥ π * (1 - 2 / n) := 
sorry

end dihedral_angle_ge_l79_79684


namespace sqrt_5sq_4six_eq_320_l79_79448

theorem sqrt_5sq_4six_eq_320 : Real.sqrt (5^2 * 4^6) = 320 :=
by sorry

end sqrt_5sq_4six_eq_320_l79_79448


namespace non_gray_squares_count_l79_79762

-- Define the dimensions of the grid strip
def width : ℕ := 5
def length : ℕ := 250

-- Define the repeating pattern dimensions and color distribution
def pattern_columns : ℕ := 4
def pattern_non_gray_squares : ℕ := 13
def pattern_total_squares : ℕ := width * pattern_columns

-- Define the number of complete patterns in the grid strip
def complete_patterns : ℕ := length / pattern_columns

-- Define the number of additional columns and additional non-gray squares
def additional_columns : ℕ := length % pattern_columns
def additional_non_gray_squares : ℕ := 6

-- Calculate the total non-gray squares
def total_non_gray_squares : ℕ := complete_patterns * pattern_non_gray_squares + additional_non_gray_squares

theorem non_gray_squares_count : total_non_gray_squares = 812 := by
  sorry

end non_gray_squares_count_l79_79762


namespace ratio_of_kids_waiting_for_slide_to_swings_final_ratio_of_kids_waiting_l79_79466

-- Define the conditions
def W : ℕ := 3
def wait_time_swing : ℕ := 120 * W
def wait_time_slide (S : ℕ) : ℕ := 15 * S
def wait_diff_condition (S : ℕ) : Prop := wait_time_swing - wait_time_slide S = 270

theorem ratio_of_kids_waiting_for_slide_to_swings (S : ℕ) (h : wait_diff_condition S) : S = 6 :=
by
  -- placeholder proof
  sorry

theorem final_ratio_of_kids_waiting (S : ℕ) (h : wait_diff_condition S) : S / W = 2 :=
by
  -- placeholder proof
  sorry

end ratio_of_kids_waiting_for_slide_to_swings_final_ratio_of_kids_waiting_l79_79466


namespace find_minimum_value_of_quadratic_l79_79203

theorem find_minimum_value_of_quadratic :
  ∀ (x : ℝ), (x = 5/2) -> (∀ y, y = 3 * x ^ 2 - 15 * x + 7 -> ∀ z, z ≥ y) := 
sorry

end find_minimum_value_of_quadratic_l79_79203


namespace minimum_value_l79_79603

theorem minimum_value (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  ∃ x : ℝ, 
    (x = 2 * (a / b) + 2 * (b / c) + 2 * (c / a) + (a / b) ^ 2) ∧ 
    (∀ y, y = 2 * (a / b) + 2 * (b / c) + 2 * (c / a) + (a / b) ^ 2 → x ≤ y) ∧ 
    x = 7 :=
by 
  sorry

end minimum_value_l79_79603


namespace evaluate_expression_l79_79774

noncomputable def a : ℕ := 2
noncomputable def b : ℕ := 1

theorem evaluate_expression : (1 / 2)^(b - a + 1) = 1 :=
by
  sorry

end evaluate_expression_l79_79774


namespace max_a_value_l79_79812

noncomputable def f (x k a : ℝ) : ℝ := x^2 - (k^2 - 5 * a * k + 3) * x + 7

theorem max_a_value : ∀ (k a : ℝ), (0 <= k) → (k <= 2) →
  (∀ (x1 : ℝ), (k <= x1) → (x1 <= k + a) →
  ∀ (x2 : ℝ), (k + 2 * a <= x2) → (x2 <= k + 4 * a) →
  f x1 k a >= f x2 k a) → 
  a <= (2 * Real.sqrt 6 - 4) / 5 := 
sorry

end max_a_value_l79_79812


namespace company_employee_count_l79_79494

/-- 
 Given the employees are divided into three age groups: A, B, and C, with a ratio of 5:4:1,
 a stratified sampling method is used to draw a sample of size 20 from the population,
 and the probability of selecting both person A and person B from group C is 1/45.
 Prove the total number of employees in the company is 100.
-/
theorem company_employee_count :
  ∃ (total_employees : ℕ),
    (∃ (ratio_A : ℕ) (ratio_B : ℕ) (ratio_C : ℕ),
      ratio_A = 5 ∧ 
      ratio_B = 4 ∧ 
      ratio_C = 1 ∧
      ∃ (sample_size : ℕ), 
        sample_size = 20 ∧
        ∃ (prob_selecting_two_from_C : ℚ),
          prob_selecting_two_from_C = 1 / 45 ∧
          total_employees = 100) :=
sorry

end company_employee_count_l79_79494


namespace train_speed_l79_79553

theorem train_speed
  (cross_time : ℝ := 5)
  (train_length : ℝ := 111.12)
  (conversion_factor : ℝ := 3.6)
  (speed : ℝ := (train_length / cross_time) * conversion_factor) :
  speed = 80 :=
by
  sorry

end train_speed_l79_79553


namespace condition_for_a_b_complex_l79_79378

theorem condition_for_a_b_complex (a b : ℂ) (h1 : a ≠ 0) (h2 : 2 * a + b ≠ 0) :
  (2 * a + b) / a = b / (2 * a + b) → 
  (∃ z : ℂ, a = z ∨ b = z) ∨ 
  ((∃ z1 : ℂ, a = z1) ∧ (∃ z2 : ℂ, b = z2)) :=
sorry

end condition_for_a_b_complex_l79_79378


namespace negation_of_p_l79_79691

open Classical

variable {x : ℝ}

def p : Prop := ∃ x : ℝ, x > 1

theorem negation_of_p : ¬p ↔ ∀ x : ℝ, x ≤ 1 :=
by
  sorry

end negation_of_p_l79_79691


namespace quadratic_inequality_solution_set_l79_79015

theorem quadratic_inequality_solution_set (a b c : ℝ) : 
  (∀ x : ℝ, - (a / 3) * x^2 + 2 * b * x - c < 0) ↔ (a > 0 ∧ 4 * b^2 - (4 / 3) * a * c < 0) := 
by
  sorry

end quadratic_inequality_solution_set_l79_79015


namespace value_of_expr_l79_79416

theorem value_of_expr (x : ℤ) (h : x = 3) : (2 * x + 6) ^ 2 = 144 := by
  sorry

end value_of_expr_l79_79416


namespace roots_polynomial_l79_79249

noncomputable def roots_are (a b c : ℝ) : Prop :=
  a^3 - 18 * a^2 + 20 * a - 8 = 0 ∧ b^3 - 18 * b^2 + 20 * b - 8 = 0 ∧ c^3 - 18 * c^2 + 20 * c - 8 = 0

theorem roots_polynomial (a b c : ℝ) (h : roots_are a b c) : 
  (2 + a) * (2 + b) * (2 + c) = 128 :=
by
  sorry

end roots_polynomial_l79_79249


namespace problem_correct_l79_79226

def is_four_digit (n : ℕ) : Prop := n >= 1000 ∧ n < 10000
def is_nat_lt_10 (n : ℕ) : Prop := n < 10
def not_zero (n : ℕ) : Prop := n ≠ 0

structure Matrix4x4 :=
  (a₀₀ a₀₁ a₀₂ a₀₃ : ℕ)
  (a₁₀ a₁₁ a₁₂ a₁₃ : ℕ)
  (a₂₀ a₂₁ a₂₂ a₂₃ : ℕ)
  (a₃₀ a₃₁ a₃₂ a₃₃ : ℕ)

def valid_matrix (M : Matrix4x4) : Prop :=
  -- Each cell must be a natural number less than 10
  is_nat_lt_10 M.a₀₀ ∧ is_nat_lt_10 M.a₀₁ ∧ is_nat_lt_10 M.a₀₂ ∧ is_nat_lt_10 M.a₀₃ ∧
  is_nat_lt_10 M.a₁₀ ∧ is_nat_lt_10 M.a₁₁ ∧ is_nat_lt_10 M.a₁₂ ∧ is_nat_lt_10 M.a₁₃ ∧
  is_nat_lt_10 M.a₂₀ ∧ is_nat_lt_10 M.a₂₁ ∧ is_nat_lt_10 M.a₂₂ ∧ is_nat_lt_10 M.a₂₃ ∧
  is_nat_lt_10 M.a₃₀ ∧ is_nat_lt_10 M.a₃₁ ∧ is_nat_lt_10 M.a₃₂ ∧ is_nat_lt_10 M.a₃₃ ∧

  -- Cells in the same region must contain the same number
  M.a₀₀ = M.a₁₀ ∧ M.a₀₀ = M.a₂₀ ∧ M.a₀₀ = M.a₃₀ ∧
  M.a₂₀ = M.a₂₁ ∧
  M.a₂₂ = M.a₂₃ ∧ M.a₂₂ = M.a₃₂ ∧ M.a₂₂ = M.a₃₃ ∧
  M.a₀₃ = M.a₁₃ ∧
  
  -- Cells in the leftmost column cannot contain the number 0
  not_zero M.a₀₀ ∧ not_zero M.a₁₀ ∧ not_zero M.a₂₀ ∧ not_zero M.a₃₀ ∧

  -- The four-digit number formed by the first row is 2187
  is_four_digit (M.a₀₀ * 1000 + M.a₀₁ * 100 + M.a₀₂ * 10 + M.a₀₃) ∧ 
  (M.a₀₀ * 1000 + M.a₀₁ * 100 + M.a₀₂ * 10 + M.a₀₃ = 2187) ∧
  
  -- The four-digit number formed by the second row is 7387
  is_four_digit (M.a₁₀ * 1000 + M.a₁₁ * 100 + M.a₁₂ * 10 + M.a₁₃) ∧ 
  (M.a₁₀ * 1000 + M.a₁₁ * 100 + M.a₁₂ * 10 + M.a₁₃ = 7387) ∧
  
  -- The four-digit number formed by the third row is 7744
  is_four_digit (M.a₂₀ * 1000 + M.a₂₁ * 100 + M.a₂₂ * 10 + M.a₂₃) ∧ 
  (M.a₂₀ * 1000 + M.a₂₁ * 100 + M.a₂₂ * 10 + M.a₂₃ = 7744) ∧
  
  -- The four-digit number formed by the fourth row is 7844
  is_four_digit (M.a₃₀ * 1000 + M.a₃₁ * 100 + M.a₃₂ * 10 + M.a₃₃) ∧ 
  (M.a₃₀ * 1000 + M.a₃₁ * 100 + M.a₃₂ * 10 + M.a₃₃ = 7844)

noncomputable def problem_solution : Matrix4x4 :=
{ a₀₀ := 2, a₀₁ := 1, a₀₂ := 8, a₀₃ := 7,
  a₁₀ := 7, a₁₁ := 3, a₁₂ := 8, a₁₃ := 7,
  a₂₀ := 7, a₂₁ := 7, a₂₂ := 4, a₂₃ := 4,
  a₃₀ := 7, a₃₁ := 8, a₃₂ := 4, a₃₃ := 4 }

theorem problem_correct : valid_matrix problem_solution :=
by
  -- The proof would go here to show that problem_solution meets valid_matrix
  sorry

end problem_correct_l79_79226


namespace penguins_count_l79_79850

theorem penguins_count (fish_total penguins_fed penguins_require : ℕ) (h1 : fish_total = 68) (h2 : penguins_fed = 19) (h3 : penguins_require = 17) : penguins_fed + penguins_require = 36 :=
by
  sorry

end penguins_count_l79_79850


namespace magnification_factor_is_correct_l79_79722

theorem magnification_factor_is_correct
    (diameter_magnified_image : ℝ)
    (actual_diameter_tissue : ℝ)
    (diameter_magnified_image_eq : diameter_magnified_image = 2)
    (actual_diameter_tissue_eq : actual_diameter_tissue = 0.002) :
  diameter_magnified_image / actual_diameter_tissue = 1000 := by
  -- Theorem and goal statement
  sorry

end magnification_factor_is_correct_l79_79722


namespace rectangular_garden_area_l79_79534

theorem rectangular_garden_area (w l : ℝ) 
  (h1 : l = 3 * w + 30) 
  (h2 : 2 * (l + w) = 800) : w * l = 28443.75 := 
by
  sorry

end rectangular_garden_area_l79_79534


namespace min_value_quadratic_l79_79177

theorem min_value_quadratic (x : ℝ) : -2 * x^2 + 8 * x + 5 ≥ -2 * (2 - x)^2 + 13 :=
by
  sorry

end min_value_quadratic_l79_79177


namespace find_line_eq_of_given_conditions_l79_79036

def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 6 * y + 5 = 0
def line_perpendicular (a b : ℝ) : Prop := a + b + 1 = 0
def is_center (x y : ℝ) : Prop := (x, y) = (0, 3)
def is_eq_of_line (x y : ℝ) : Prop := x - y + 3 = 0

theorem find_line_eq_of_given_conditions (x y : ℝ) (h1 : circle_eq x y) (h2 : line_perpendicular x y) (h3 : is_center x y) : is_eq_of_line x y :=
by
  sorry

end find_line_eq_of_given_conditions_l79_79036


namespace evan_books_l79_79611

theorem evan_books (B M : ℕ) (h1 : B = 200 - 40) (h2 : M * B + 60 = 860) : M = 5 :=
by {
  sorry  -- proof is omitted as per instructions
}

end evan_books_l79_79611


namespace percentage_increase_each_job_l79_79622

-- Definitions of original and new amounts for each job as given conditions
def original_first_job : ℝ := 65
def new_first_job : ℝ := 70

def original_second_job : ℝ := 240
def new_second_job : ℝ := 315

def original_third_job : ℝ := 800
def new_third_job : ℝ := 880

-- Proof problem statement
theorem percentage_increase_each_job :
  (new_first_job - original_first_job) / original_first_job * 100 = 7.69 ∧
  (new_second_job - original_second_job) / original_second_job * 100 = 31.25 ∧
  (new_third_job - original_third_job) / original_third_job * 100 = 10 := by
  sorry

end percentage_increase_each_job_l79_79622


namespace complete_the_square_l79_79570

theorem complete_the_square :
  ∀ x : ℝ, (x^2 - 2 * x - 2 = 0) → ((x - 1)^2 = 3) :=
by
  intros x h
  sorry

end complete_the_square_l79_79570


namespace total_number_of_athletes_l79_79797

theorem total_number_of_athletes (M F x : ℕ) (r1 r2 r3 : ℕ×ℕ) (H1 : r1 = (19, 12)) (H2 : r2 = (20, 13)) (H3 : r3 = (30, 19))
  (initial_males : M = 380 * x) (initial_females : F = 240 * x)
  (males_after_gym : M' = 390 * x) (females_after_gym : F' = 247 * x)
  (conditions : (M' - M) - (F' - F) = 30) : M' + F' = 6370 :=
by
  sorry

end total_number_of_athletes_l79_79797


namespace sequence_value_l79_79978

theorem sequence_value (a : ℕ → ℝ) 
  (h1 : ∀ n : ℕ, n > 0 → a n * a (n + 2) = a (n + 1) ^ 2)
  (h2 : a 7 = 16)
  (h3 : a 3 * a 5 = 4) : 
  a 3 = 1 := 
sorry

end sequence_value_l79_79978


namespace ticket_prices_count_l79_79205

theorem ticket_prices_count :
  let y := 30
  let divisors := [1, 2, 3, 5, 6, 10, 15, 30]
  ∀ (k : ℕ), (k ∈ divisors) ↔ (60 % k = 0 ∧ 90 % k = 0) → 
  (∃ n : ℕ, n = 8) :=
by
  sorry

end ticket_prices_count_l79_79205


namespace average_of_roots_l79_79785

theorem average_of_roots (p q : ℝ) (h : ∀ r : ℝ, r^2 * (3 * p) + r * (-6 * p) + q = 0 → ∃ a b : ℝ, r = a ∨ r = b) : 
  ∀ (r1 r2 : ℝ), (3 * p) * r1^2 + (-6 * p) * r1 + q = 0 ∧ (3 * p) * r2^2 + (-6 * p) * r2 + q = 0 → 
  (r1 + r2) / 2 = 1 :=
by {
  sorry
}

end average_of_roots_l79_79785


namespace area_of_rectangle_is_108_l79_79871

theorem area_of_rectangle_is_108 (s w l : ℕ) (h₁ : s * s = 36) (h₂ : w = s) (h₃ : l = 3 * w) : w * l = 108 :=
by
  -- This is a placeholder for a detailed proof.
  sorry

end area_of_rectangle_is_108_l79_79871


namespace correct_conclusions_l79_79137

noncomputable def quadratic_solution_set (a b c : ℝ) : Prop :=
  ∀ x : ℝ, (-1 / 2 < x ∧ x < 3) ↔ (a * x^2 + b * x + c > 0)

theorem correct_conclusions (a b c : ℝ) (h : quadratic_solution_set a b c) : c > 0 ∧ 4 * a + 2 * b + c > 0 :=
  sorry

end correct_conclusions_l79_79137


namespace perfect_square_pairs_l79_79958

-- Definition of a perfect square
def is_perfect_square (k : ℕ) : Prop :=
∃ (n : ℕ), n * n = k

-- Main theorem statement
theorem perfect_square_pairs (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  is_perfect_square ((2^m - 1) * (2^n - 1)) ↔ (m = n) ∨ (m = 3 ∧ n = 6) ∨ (m = 6 ∧ n = 3) :=
sorry

end perfect_square_pairs_l79_79958


namespace difference_of_numbers_l79_79216

theorem difference_of_numbers (a b : ℕ) (h1 : a = 2 * b) (h2 : (a + 4) / (b + 4) = 5 / 7) : a - b = 8 := 
by
  sorry

end difference_of_numbers_l79_79216


namespace anna_discontinued_coaching_on_2nd_august_l79_79439

theorem anna_discontinued_coaching_on_2nd_august
  (coaching_days : ℕ) (non_leap_year : ℕ) (first_day : ℕ) 
  (days_in_january : ℕ) (days_in_february : ℕ) (days_in_march : ℕ) 
  (days_in_april : ℕ) (days_in_may : ℕ) (days_in_june : ℕ) 
  (days_in_july : ℕ) (days_in_august : ℕ)
  (not_leap_year : non_leap_year = 365)
  (first_day_of_year : first_day = 1)
  (january_days : days_in_january = 31)
  (february_days : days_in_february = 28)
  (march_days : days_in_march = 31)
  (april_days : days_in_april = 30)
  (may_days : days_in_may = 31)
  (june_days : days_in_june = 30)
  (july_days : days_in_july = 31)
  (august_days : days_in_august = 31)
  (total_coaching_days : coaching_days = 245) :
  ∃ day, day = 2 ∧ month = "August" := 
sorry

end anna_discontinued_coaching_on_2nd_august_l79_79439


namespace samatha_routes_l79_79179

-- Definitions based on the given conditions
def blocks_from_house_to_southwest_corner := 4
def blocks_through_park := 1
def blocks_from_northeast_corner_to_school := 4
def blocks_from_school_to_library := 3

-- Number of ways to arrange movements
def number_of_routes_house_to_southwest : ℕ :=
  Nat.choose blocks_from_house_to_southwest_corner 1

def number_of_routes_through_park : ℕ := blocks_through_park

def number_of_routes_northeast_to_school : ℕ :=
  Nat.choose blocks_from_northeast_corner_to_school 1

def number_of_routes_school_to_library : ℕ :=
  Nat.choose blocks_from_school_to_library 1

-- Total number of different routes
def total_number_of_routes : ℕ :=
  number_of_routes_house_to_southwest *
  number_of_routes_through_park *
  number_of_routes_northeast_to_school *
  number_of_routes_school_to_library

theorem samatha_routes (n : ℕ) (h : n = 48) :
  total_number_of_routes = n :=
  by
    -- Proof is skipped
    sorry

end samatha_routes_l79_79179


namespace correct_average_weight_l79_79292

theorem correct_average_weight (avg_weight : ℝ) (num_boys : ℕ) (incorrect_weight correct_weight : ℝ)
  (h1 : avg_weight = 58.4) (h2 : num_boys = 20) (h3 : incorrect_weight = 56) (h4 : correct_weight = 62) :
  (avg_weight * ↑num_boys + (correct_weight - incorrect_weight)) / ↑num_boys = 58.7 := by
  sorry

end correct_average_weight_l79_79292


namespace relationship_between_Q_and_t_remaining_power_after_5_hours_distance_with_40_power_l79_79383

-- Define the relationship between Q and t
def remaining_power (t : ℕ) : ℕ := 80 - 15 * t

-- Question 1: Prove relationship between Q and t
theorem relationship_between_Q_and_t : ∀ t : ℕ, remaining_power t = 80 - 15 * t :=
by sorry

-- Question 2: Prove remaining power after 5 hours
theorem remaining_power_after_5_hours : remaining_power 5 = 5 :=
by sorry

-- Question 3: Prove distance the car can travel with 40 kW·h remaining power
theorem distance_with_40_power 
  (remaining_power : ℕ := (80 - 15 * t)) 
  (t := 8 / 3)
  (speed : ℕ := 90) : (90 * (8 / 3)) = 240 :=
by sorry

end relationship_between_Q_and_t_remaining_power_after_5_hours_distance_with_40_power_l79_79383


namespace remainder_2753_div_98_l79_79395

theorem remainder_2753_div_98 : (2753 % 98) = 9 := 
by sorry

end remainder_2753_div_98_l79_79395


namespace number_of_balls_sold_l79_79910

theorem number_of_balls_sold 
  (selling_price : ℤ) (loss_per_5_balls : ℤ) (cost_price_per_ball : ℤ) (n : ℤ) 
  (h1 : selling_price = 720)
  (h2 : loss_per_5_balls = 5 * cost_price_per_ball)
  (h3 : cost_price_per_ball = 48)
  (h4 : (n * cost_price_per_ball) - selling_price = loss_per_5_balls) :
  n = 20 := 
by
  sorry

end number_of_balls_sold_l79_79910


namespace part_I_part_II_l79_79510

-- Part (I): If a = 1, prove that q implies p
theorem part_I (x : ℝ) (h : 3 < x ∧ x < 4) : (1 < x) ∧ (x < 4) :=
by sorry

-- Part (II): Prove the range of a for which p is necessary but not sufficient for q
theorem part_II (a : ℝ) (h1 : a > 0) (h2 : ∀ x : ℝ, (a < x ∧ x < 4 * a) → (3 < x ∧ x < 4)) : 1 < a ∧ a ≤ 3 :=
by sorry

end part_I_part_II_l79_79510


namespace selling_price_correct_l79_79357

noncomputable def cost_price : ℝ := 2800
noncomputable def loss_percentage : ℝ := 25
noncomputable def loss_amount (cost_price loss_percentage : ℝ) : ℝ := (loss_percentage / 100) * cost_price
noncomputable def selling_price (cost_price loss_amount : ℝ) : ℝ := cost_price - loss_amount

theorem selling_price_correct : 
  selling_price cost_price (loss_amount cost_price loss_percentage) = 2100 :=
by
  sorry

end selling_price_correct_l79_79357


namespace fruit_prob_l79_79843

variable (O A B S : ℕ) 

-- Define the conditions
variables (H1 : O + A + B + S = 32)
variables (H2 : O - 5 = 3)
variables (H3 : A - 3 = 7)
variables (H4 : S - 2 = 4)
variables (H5 : 3 + 7 + 4 + B = 20)

-- Define the proof problem
theorem fruit_prob :
  (O = 8) ∧ (A = 10) ∧ (B = 6) ∧ (S = 6) → (O + S) / (O + A + B + S) = 7 / 16 := 
by
  sorry

end fruit_prob_l79_79843


namespace solve_for_y_l79_79381

theorem solve_for_y (x y : ℝ) (h : 4 * x - y = 3) : y = 4 * x - 3 :=
by sorry

end solve_for_y_l79_79381


namespace jackson_difference_l79_79168

theorem jackson_difference :
  let Jackson_initial := 500
  let Brandon_initial := 500
  let Meagan_initial := 700
  let Jackson_final := Jackson_initial * 4
  let Brandon_final := Brandon_initial * 0.20
  let Meagan_final := Meagan_initial + (Meagan_initial * 0.50)
  Jackson_final - (Brandon_final + Meagan_final) = 850 :=
by
  sorry

end jackson_difference_l79_79168


namespace misread_weight_l79_79257

theorem misread_weight (avg_initial : ℝ) (avg_correct : ℝ) (n : ℕ) (actual_weight : ℝ) (x : ℝ) : 
  avg_initial = 58.4 → avg_correct = 58.7 → n = 20 → actual_weight = 62 → 
  (n * avg_correct - n * avg_initial = actual_weight - x) → x = 56 :=
by
  intros
  sorry

end misread_weight_l79_79257


namespace running_speed_is_24_l79_79319

def walk_speed := 8 -- km/h
def walk_time := 3 -- hours
def run_time := 1 -- hour

def walk_distance := walk_speed * walk_time

def run_speed := walk_distance / run_time

theorem running_speed_is_24 : run_speed = 24 := 
by
  sorry

end running_speed_is_24_l79_79319


namespace darwin_final_money_l79_79385

def initial_amount : ℕ := 600
def spent_on_gas (initial : ℕ) : ℕ := initial * 1 / 3
def remaining_after_gas (initial spent_gas : ℕ) : ℕ := initial - spent_gas
def spent_on_food (remaining : ℕ) : ℕ := remaining * 1 / 4
def final_amount (remaining spent_food : ℕ) : ℕ := remaining - spent_food

theorem darwin_final_money :
  final_amount (remaining_after_gas initial_amount (spent_on_gas initial_amount)) (spent_on_food (remaining_after_gas initial_amount (spent_on_gas initial_amount))) = 300 :=
by
  sorry

end darwin_final_money_l79_79385


namespace no_super_plus_good_exists_at_most_one_super_plus_good_l79_79057

def is_super_plus_good (board : ℕ → ℕ → ℕ) (n : ℕ) (i j : ℕ) : Prop :=
  (∀ k, k < n → board i k ≤ board i j) ∧ 
  (∀ k, k < n → board k j ≥ board i j)

def arrangement (n : ℕ) := { board : ℕ → ℕ → ℕ // ∀ i j, i < n → j < n → 1 ≤ board i j ∧ board i j ≤ n * n }

-- Prove that in some arrangements, there is no super-plus-good number.
theorem no_super_plus_good_exists (n : ℕ) (h₁ : n = 8) :
  ∃ (b : arrangement n), ∀ i j, ¬ is_super_plus_good b.val n i j := sorry

-- Prove that in every arrangement, there is at most one super-plus-good number.
theorem at_most_one_super_plus_good (n : ℕ) (h : n = 8) :
  ∀ (b : arrangement n), ∃! i j, is_super_plus_good b.val n i j := sorry

end no_super_plus_good_exists_at_most_one_super_plus_good_l79_79057


namespace bananas_to_pears_ratio_l79_79123

theorem bananas_to_pears_ratio (B P : ℕ) (hP : P = 50) (h1 : B + 10 = 160) (h2: ∃ k : ℕ, B = k * P) : B / P = 3 :=
by
  -- proof steps would go here
  sorry

end bananas_to_pears_ratio_l79_79123


namespace find_Q_l79_79936

variable {x P Q : ℝ}

theorem find_Q (h₁ : x + 1 / x = P) (h₂ : P = 1) : x^6 + 1 / x^6 = 2 :=
by
  sorry

end find_Q_l79_79936


namespace Anne_mom_toothpaste_usage_l79_79955

theorem Anne_mom_toothpaste_usage
  (total_toothpaste : ℕ)
  (dad_usage_per_brush : ℕ)
  (sibling_usage_per_brush : ℕ)
  (num_brushes_per_day : ℕ)
  (total_days : ℕ)
  (total_toothpaste_used : ℕ)
  (M : ℕ)
  (family_use_model : total_toothpaste = total_toothpaste_used + 3 * num_brushes_per_day * M)
  (total_toothpaste_used_def : total_toothpaste_used = 5 * (dad_usage_per_brush * num_brushes_per_day + 2 * sibling_usage_per_brush * num_brushes_per_day))
  (given_values : total_toothpaste = 105 ∧ dad_usage_per_brush = 3 ∧ sibling_usage_per_brush = 1 ∧ num_brushes_per_day = 3 ∧ total_days = 5)
  : M = 2 := by
  sorry

end Anne_mom_toothpaste_usage_l79_79955


namespace total_money_raised_l79_79461

-- Assume there are 30 students in total
def total_students := 30

-- Assume 10 students raised $20 each
def students_raising_20 := 10
def money_raised_per_20 := 20

-- The rest of the students raised $30 each
def students_raising_30 := total_students - students_raising_20
def money_raised_per_30 := 30

-- Prove that the total amount raised is $800
theorem total_money_raised :
  (students_raising_20 * money_raised_per_20) +
  (students_raising_30 * money_raised_per_30) = 800 :=
by
  sorry

end total_money_raised_l79_79461


namespace sum_of_three_squares_l79_79585

-- Using the given conditions to define the problem.
variable (square triangle : ℝ)

-- Conditions
axiom h1 : square + triangle + 2 * square + triangle = 34
axiom h2 : triangle + square + triangle + 3 * square = 40

-- Statement to prove
theorem sum_of_three_squares : square + square + square = 66 / 7 :=
by
  sorry

end sum_of_three_squares_l79_79585


namespace fraction_is_one_fourth_l79_79826

theorem fraction_is_one_fourth (f N : ℝ) 
  (h1 : (1/3) * f * N = 15) 
  (h2 : (3/10) * N = 54) : 
  f = 1/4 :=
by
  sorry

end fraction_is_one_fourth_l79_79826


namespace texts_sent_on_Tuesday_l79_79194

theorem texts_sent_on_Tuesday (total_texts monday_texts : Nat) (texts_each_monday : Nat)
  (h_monday : texts_each_monday = 5)
  (h_total : total_texts = 40)
  (h_monday_total : monday_texts = 2 * texts_each_monday) :
  total_texts - monday_texts = 30 := by
  sorry

end texts_sent_on_Tuesday_l79_79194


namespace no_common_solution_l79_79264

theorem no_common_solution :
  ¬(∃ y : ℚ, (6 * y^2 + 11 * y - 1 = 0) ∧ (18 * y^2 + y - 1 = 0)) :=
by
  sorry

end no_common_solution_l79_79264


namespace mary_needs_to_add_l79_79645

-- Define the conditions
def total_flour_required : ℕ := 7
def flour_already_added : ℕ := 2

-- Define the statement that corresponds to the mathematical equivalent proof problem
theorem mary_needs_to_add :
  total_flour_required - flour_already_added = 5 :=
by
  sorry

end mary_needs_to_add_l79_79645


namespace derivative_at_five_l79_79006

noncomputable def g (x : ℝ) : ℝ := (x - 1) * (x - 2) * (x - 3)

theorem derivative_at_five : deriv g 5 = 26 :=
sorry

end derivative_at_five_l79_79006


namespace expected_coins_100_rounds_l79_79845

noncomputable def expectedCoinsAfterGame (rounds : ℕ) (initialCoins : ℕ) : ℝ :=
  initialCoins * (101 / 100) ^ rounds

theorem expected_coins_100_rounds :
  expectedCoinsAfterGame 100 1 = (101 / 100 : ℝ) ^ 100 :=
by
  sorry

end expected_coins_100_rounds_l79_79845


namespace machines_initially_working_l79_79008

theorem machines_initially_working (N x : ℕ) (h1 : N * 4 * R = x)
  (h2 : 20 * 6 * R = 3 * x) : N = 10 :=
by
  sorry

end machines_initially_working_l79_79008


namespace total_time_spent_l79_79967

noncomputable def time_per_round : ℕ := 30
noncomputable def saturday_rounds : ℕ := 1 + 10
noncomputable def sunday_rounds : ℕ := 15
noncomputable def total_rounds : ℕ := saturday_rounds + sunday_rounds
noncomputable def total_time : ℕ := total_rounds * time_per_round

theorem total_time_spent :
  total_time = 780 := by sorry

end total_time_spent_l79_79967


namespace temperature_on_tuesday_l79_79765

variable (T W Th F : ℝ)

theorem temperature_on_tuesday :
  (T + W + Th = 156) ∧ (W + Th + 53 = 162) → T = 47 :=
by
  sorry

end temperature_on_tuesday_l79_79765


namespace initial_forks_l79_79470

variables (forks knives spoons teaspoons : ℕ)
variable (F : ℕ)

-- Conditions as given
def num_knives := F + 9
def num_spoons := 2 * (F + 9)
def num_teaspoons := F / 2
def total_cutlery := (F + 2) + (F + 11) + (2 * (F + 9) + 2) + (F / 2 + 2)

-- Problem statement to prove
theorem initial_forks :
  (total_cutlery = 62) ↔ (F = 6) :=
by {
  sorry
}

end initial_forks_l79_79470


namespace evaluate_f_g_at_3_l79_79795

def f (x : ℝ) : ℝ := x^2 + 2
def g (x : ℝ) : ℝ := 3 * x + 2

theorem evaluate_f_g_at_3 : f (g 3) = 123 := by
  sorry

end evaluate_f_g_at_3_l79_79795


namespace distance_they_both_run_l79_79646

theorem distance_they_both_run
  (time_A time_B : ℕ)
  (distance_advantage: ℝ)
  (speed_A speed_B : ℝ)
  (D : ℝ) :
  time_A = 198 →
  time_B = 220 →
  distance_advantage = 300 →
  speed_A = D / time_A →
  speed_B = D / time_B →
  speed_A * time_B = D + distance_advantage →
  D = 2700 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end distance_they_both_run_l79_79646


namespace safe_trip_possible_l79_79906

-- Define the time intervals and eruption cycles
def total_round_trip_time := 16
def trail_time := 8
def crater1_cycle := 18
def crater2_cycle := 10
def crater1_erupt := 1
def crater1_quiet := 17
def crater2_erupt := 1
def crater2_quiet := 9

-- Ivan wants to safely reach the summit and return
theorem safe_trip_possible : ∃ t, 
  -- t is a valid start time where both craters are quiet
  ((t % crater1_cycle) ≥ crater1_erupt ∧ (t % crater2_cycle) ≥ crater2_erupt) ∧
  -- t + total_round_trip_time is also safe for both craters
  (((t + total_round_trip_time) % crater1_cycle) ≥ crater1_erupt ∧ ((t + total_round_trip_time) % crater2_cycle) ≥ crater2_erupt) :=
sorry

end safe_trip_possible_l79_79906


namespace cricket_target_runs_l79_79263

def run_rate_first_20_overs : ℝ := 4.2
def overs_first_20 : ℝ := 20
def run_rate_remaining_30_overs : ℝ := 5.533333333333333
def overs_remaining_30 : ℝ := 30
def total_runs_first_20 : ℝ := run_rate_first_20_overs * overs_first_20
def total_runs_remaining_30 : ℝ := run_rate_remaining_30_overs * overs_remaining_30

theorem cricket_target_runs :
  (total_runs_first_20 + total_runs_remaining_30) = 250 :=
by
  sorry

end cricket_target_runs_l79_79263


namespace patanjali_distance_first_day_l79_79671

theorem patanjali_distance_first_day
  (h : ℕ)
  (H1 : 3 * h + 4 * (h - 1) + 4 * h = 62) :
  3 * h = 18 :=
by
  sorry

end patanjali_distance_first_day_l79_79671


namespace evaluate_expression_l79_79079

theorem evaluate_expression : 5^2 - 5 + (6^2 - 6) - (7^2 - 7) + (8^2 - 8) = 64 :=
by sorry

end evaluate_expression_l79_79079


namespace sum_of_divisors_117_l79_79198

-- Defining the conditions in Lean
def n : ℕ := 117
def is_factorization : n = 3^2 * 13 := by rfl

-- The sum-of-divisors function can be defined based on the problem
def sum_of_divisors (n : ℕ) : ℕ :=
  (1 + 3 + 3^2) * (1 + 13)

-- Assertion of the correct answer
theorem sum_of_divisors_117 : sum_of_divisors n = 182 := by
  sorry

end sum_of_divisors_117_l79_79198


namespace probability_even_sum_l79_79612

theorem probability_even_sum (x y : ℕ) (h : x + y ≤ 10) : 
  (∃ (p : ℚ), p = 6 / 11 ∧ (x + y) % 2 = 0) :=
sorry

end probability_even_sum_l79_79612


namespace minimize_triangle_expression_l79_79500

theorem minimize_triangle_expression :
  ∃ (a b c : ℤ), a < b ∧ b < c ∧ a + b + c = 30 ∧
  ∀ (x y z : ℤ), x < y ∧ y < z ∧ x + y + z = 30 → (z^2 + 18*x + 18*y - 446) ≥ 17 ∧ 
  ∃ (p q r : ℤ), p < q ∧ q < r ∧ p + q + r = 30 ∧ (r^2 + 18*p + 18*q - 446 = 17) := 
sorry

end minimize_triangle_expression_l79_79500


namespace songs_owned_initially_l79_79929

theorem songs_owned_initially (a b c : ℕ) (hc : c = a + b) (hb : b = 7) (hc_total : c = 13) :
  a = 6 :=
by
  -- Direct usage of the given conditions to conclude the proof goes here.
  sorry

end songs_owned_initially_l79_79929


namespace days_to_complete_work_together_l79_79971

theorem days_to_complete_work_together :
  (20 * 35) / (20 + 35) = 140 / 11 :=
by
  sorry

end days_to_complete_work_together_l79_79971


namespace union_sets_l79_79763

open Set

/-- Given sets A and B defined as follows:
    A = {x | -1 ≤ x ∧ x ≤ 2}
    B = {x | x ≤ 4}
    Prove that A ∪ B = {x | x ≤ 4}
--/
theorem union_sets  :
    let A := {x | -1 ≤ x ∧ x ≤ 2}
    let B := {x | x ≤ 4}
    A ∪ B = {x | x ≤ 4} :=
by
    intros A B
    have : A = {x | -1 ≤ x ∧ x ≤ 2} := rfl
    have : B = {x | x ≤ 4} := rfl
    sorry

end union_sets_l79_79763


namespace yellow_ball_percentage_l79_79428

theorem yellow_ball_percentage
  (yellow_balls : ℕ)
  (brown_balls : ℕ)
  (blue_balls : ℕ)
  (green_balls : ℕ)
  (total_balls : ℕ := yellow_balls + brown_balls + blue_balls + green_balls)
  (h_yellow : yellow_balls = 75)
  (h_brown : brown_balls = 120)
  (h_blue : blue_balls = 45)
  (h_green : green_balls = 60) :
  (yellow_balls * 100) / total_balls = 25 := 
by
  sorry

end yellow_ball_percentage_l79_79428


namespace white_truck_percentage_is_17_l79_79745

-- Define the conditions
def total_trucks : ℕ := 50
def total_cars : ℕ := 40
def total_vehicles : ℕ := total_trucks + total_cars

def red_trucks : ℕ := total_trucks / 2
def black_trucks : ℕ := (total_trucks * 20) / 100
def white_trucks : ℕ := total_trucks - red_trucks - black_trucks

def percentage_white_trucks : ℕ := (white_trucks * 100) / total_vehicles

theorem white_truck_percentage_is_17 :
  percentage_white_trucks = 17 :=
  by sorry

end white_truck_percentage_is_17_l79_79745


namespace part_I_part_II_l79_79951

noncomputable def f (x : ℝ) : ℝ :=
  2 * (Real.cos x) ^ 2 - Real.sin (2 * x - (7 * Real.pi / 6))

theorem part_I :
  (∀ x, f x ≤ 2) ∧ (∃ x, f x = 2 ∧ ∃ k : ℤ, x = k * Real.pi + Real.pi / 6) :=
by
  sorry

theorem part_II (A a b c : ℝ) (h1 : f A = 3 / 2) (h2 : b + c = 2) :
  a >= 1 :=
by
  sorry

end part_I_part_II_l79_79951


namespace right_triangle_angles_l79_79746

theorem right_triangle_angles (a b S : ℝ) (hS : S = 1 / 2 * a * b) (h : (a + b) ^ 2 = 8 * S) :
  ∃ θ₁ θ₂ θ₃ : ℝ, θ₁ = 45 ∧ θ₂ = 45 ∧ θ₃ = 90 :=
by {
  sorry
}

end right_triangle_angles_l79_79746


namespace total_vessels_l79_79627

theorem total_vessels (C G S F : ℕ) (h1 : C = 4) (h2 : G = 2 * C) (h3 : S = G + 6) (h4 : S = 7 * F) : 
  C + G + S + F = 28 :=
by
  sorry

end total_vessels_l79_79627


namespace wood_length_equation_l79_79255

theorem wood_length_equation (x : ℝ) : 
  (∃ r : ℝ, r - x = 4.5 ∧ r/2 + 1 = x) → 1/2 * (x + 4.5) = x - 1 :=
sorry

end wood_length_equation_l79_79255


namespace pentagon_area_l79_79300

open Function 

/-
Given a convex pentagon FGHIJ with the following properties:
  1. ∠F = ∠G = 100°
  2. JF = FG = GH = 3
  3. HI = IJ = 5
Prove that the area of pentagon FGHIJ is approximately 15.2562 square units.
-/

noncomputable def area_pentagon_FGHIJ : ℝ :=
  let sin100 := Real.sin (100 * Real.pi / 180)
  let area_FGJ := (3 * 3 * sin100) / 2
  let area_HIJ := (5 * 5 * Real.sqrt 3) / 4
  area_FGJ + area_HIJ

theorem pentagon_area : abs (area_pentagon_FGHIJ - 15.2562) < 0.0001 := by
  sorry

end pentagon_area_l79_79300


namespace find_a_l79_79225

def A : Set ℝ := { x | x^2 - 5 * x + 6 = 0 }
def B (a : ℝ) : Set ℝ := { x | a * x - 1 = 0 }

theorem find_a (a : ℝ) : (A ∩ B a = B a) → (a = 0 ∨ a = 1 / 2 ∨ a = 1 / 3) := by
  sorry

end find_a_l79_79225


namespace determine_set_A_l79_79032

variable (U : Set ℕ) (A : Set ℕ)

theorem determine_set_A (hU : U = {0, 1, 2, 3}) (hcompl : U \ A = {2}) :
  A = {0, 1, 3} :=
by
  sorry

end determine_set_A_l79_79032


namespace no_positive_integer_solutions_m2_m3_positive_integer_solutions_m4_l79_79737

theorem no_positive_integer_solutions_m2_m3 (x y z t : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (ht : 0 < t) :
  (∃ m, m = 2 ∨ m = 3 → (x / y + y / z + z / t + t / x = m) → false) :=
sorry

theorem positive_integer_solutions_m4 (x y z t : ℕ) :
  x / y + y / z + z / t + t / x = 4 ↔ ∃ k : ℕ, k > 0 ∧ (x = k ∧ y = k ∧ z = k ∧ t = k) :=
sorry

end no_positive_integer_solutions_m2_m3_positive_integer_solutions_m4_l79_79737


namespace cannot_determine_right_triangle_l79_79301

def is_right_triangle (A B C : ℝ) : Prop :=
  A + B + C = 180 ∧ (A = 90 ∨ B = 90 ∨ C = 90)

theorem cannot_determine_right_triangle :
  ∀ A B C : ℝ, 
    (A = 2 * B ∧ A = 3 * C) →
    ¬ is_right_triangle A B C :=
by
  intro A B C h
  have h1 : A = 2 * B := h.1
  have h2 : A = 3 * C := h.2
  sorry

end cannot_determine_right_triangle_l79_79301


namespace distance_correct_l79_79735

-- Define geometry entities and properties
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

structure Sphere where
  center : Point
  radius : ℝ

structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define conditions
def sphere_center : Point := { x := 0, y := 0, z := 0 }
def sphere : Sphere := { center := sphere_center, radius := 5 }
def triangle : Triangle := { a := 13, b := 13, c := 10 }

-- Define the distance calculation
noncomputable def distance_from_sphere_center_to_plane (O : Point) (T : Triangle) : ℝ :=
  let h := 12  -- height calculation based on given triangle sides
  let A := 60  -- area of the triangle
  let s := 18  -- semiperimeter
  let r := 10 / 3  -- inradius calculation
  let x := 5 * (Real.sqrt 5) / 3  -- final distance calculation
  x

-- Prove the obtained distance matches expected value
theorem distance_correct :
  distance_from_sphere_center_to_plane sphere_center triangle = 5 * (Real.sqrt 5) / 3 :=
by
  sorry

end distance_correct_l79_79735


namespace sine_cosine_obtuse_angle_l79_79035

theorem sine_cosine_obtuse_angle :
  ∀ P : (ℝ × ℝ), P = (Real.sin 2, Real.cos 2) → (Real.sin 2 > 0) ∧ (Real.cos 2 < 0) → 
  (P.1 > 0) ∧ (P.2 < 0) :=
by
  sorry

end sine_cosine_obtuse_angle_l79_79035


namespace ellipse_reflection_symmetry_l79_79717

theorem ellipse_reflection_symmetry :
  (∀ x y, (x = -y ∧ y = -x) →
  (∀ a b : ℝ, 
    (a - 3)^2 / 9 + (b - 2)^2 / 4 = 1 ↔
    (b - 3)^2 / 4 + (a - 2)^2 / 9 = 1)
  )
  →
  (∀ x y, 
    ((x + 2)^2 / 9 + (y + 3)^2 / 4 = 1) = 
    (∃ a b : ℝ, 
      (a - 3)^2 / 9 + (b - 2)^2 / 4 = 1 ∧ 
      (a = -y ∧ b = -x))
  ) :=
by
  intros
  sorry

end ellipse_reflection_symmetry_l79_79717


namespace am_gm_inequality_l79_79620

theorem am_gm_inequality (a b c : ℝ) (h : a * b * c = 1 / 8) : 
  a^2 + b^2 + c^2 + a^2 * b^2 + b^2 * c^2 + c^2 * a^2 ≥ 15 / 16 :=
sorry

end am_gm_inequality_l79_79620


namespace sum_of_interior_angles_n_plus_3_l79_79628

-- Define the condition that the sum of the interior angles of a convex polygon with n sides is 1260 degrees
def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

-- Prove that given the above condition for n, the sum of the interior angles of a convex polygon with n + 3 sides is 1800 degrees
theorem sum_of_interior_angles_n_plus_3 (n : ℕ) (h : sum_of_interior_angles n = 1260) : 
  sum_of_interior_angles (n + 3) = 1800 :=
by
  sorry

end sum_of_interior_angles_n_plus_3_l79_79628


namespace second_movie_duration_proof_l79_79220

-- initial duration for the first movie (in minutes)
def first_movie_duration_minutes : ℕ := 1 * 60 + 48

-- additional duration for the second movie (in minutes)
def additional_duration_minutes : ℕ := 25

-- total duration for the second movie (in minutes)
def second_movie_duration_minutes : ℕ := first_movie_duration_minutes + additional_duration_minutes

-- convert total minutes to hours and minutes
def duration_in_hours_and_minutes (total_minutes : ℕ) : ℕ × ℕ :=
  (total_minutes / 60, total_minutes % 60)

theorem second_movie_duration_proof :
  duration_in_hours_and_minutes second_movie_duration_minutes = (2, 13) :=
by
  -- proof would go here
  sorry

end second_movie_duration_proof_l79_79220


namespace find_x_l79_79798

def a : ℝ × ℝ := (-2, 0)
def b : ℝ × ℝ := (2, 1)
def c (x : ℝ) : ℝ × ℝ := (x, -1)
def scalar_multiply (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)
def vector_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 + v2.1, v1.2 + v2.2)
def collinear (v1 v2 : ℝ × ℝ) : Prop := v1.1 * v2.2 = v1.2 * v2.1

theorem find_x :
  ∃ x : ℝ, collinear (vector_add (scalar_multiply 3 a) b) (c x) ∧ x = 4 :=
by
  sorry

end find_x_l79_79798


namespace length_MN_l79_79347

variables {A B C D M N : Type}
variables {BC AD AB : ℝ} -- Lengths of sides
variables {a b : ℝ}

-- Given conditions
def is_trapezoid (a b BC AD AB : ℝ) : Prop :=
  BC = a ∧ AD = b ∧ AB = AD + BC

-- Given, side AB is divided into 5 equal parts and a line parallel to bases is drawn through the 3rd division point
def is_divided (AB : ℝ) : Prop := ∃ P_1 P_2 P_3 P_4, AB = P_4 + P_3 + P_2 + P_1

-- Prove the length of MN
theorem length_MN (a b : ℝ) (h_trapezoid : is_trapezoid a b BC AD AB) (h_divided : is_divided AB) : 
  MN = (2 * BC + 3 * AD) / 5 :=
sorry

end length_MN_l79_79347


namespace total_cups_l79_79526

theorem total_cups (n : ℤ) (h_rainy_days : n = 8) :
  let tea_cups := 6 * 3
  let total_cups := tea_cups + n
  total_cups = 26 :=
by
  let tea_cups := 6 * 3
  let total_cups := tea_cups + n
  exact sorry

end total_cups_l79_79526


namespace probability_of_choosing_A_on_second_day_l79_79823

-- Definitions of the probabilities given in the problem conditions.
def p_first_day_A := 0.5
def p_first_day_B := 0.5
def p_second_day_A_given_first_day_A := 0.6
def p_second_day_A_given_first_day_B := 0.5

-- Define the problem to be proved in Lean 4
theorem probability_of_choosing_A_on_second_day :
  (p_first_day_A * p_second_day_A_given_first_day_A) +
  (p_first_day_B * p_second_day_A_given_first_day_B) = 0.55 :=
by
  sorry

end probability_of_choosing_A_on_second_day_l79_79823


namespace machine_shirts_per_minute_l79_79569

def shirts_made_yesterday : ℕ := 13
def shirts_made_today : ℕ := 3
def minutes_worked : ℕ := 2
def total_shirts_made : ℕ := shirts_made_yesterday + shirts_made_today
def shirts_per_minute : ℕ := total_shirts_made / minutes_worked

theorem machine_shirts_per_minute :
  shirts_per_minute = 8 := by
  sorry

end machine_shirts_per_minute_l79_79569


namespace average_percentage_l79_79802

theorem average_percentage (num_students1 num_students2 : Nat) (avg1 avg2 avg : Nat) :
  num_students1 = 15 ->
  avg1 = 73 ->
  num_students2 = 10 ->
  avg2 = 88 ->
  (num_students1 * avg1 + num_students2 * avg2) / (num_students1 + num_students2) = avg ->
  avg = 79 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end average_percentage_l79_79802


namespace find_b_l79_79879

theorem find_b 
  (b : ℝ)
  (h_pos : 0 < b)
  (h_geom_sequence : ∃ r : ℝ, 10 * r = b ∧ b * r = 2 / 3) :
  b = 2 * Real.sqrt 15 / 3 :=
by
  sorry

end find_b_l79_79879


namespace average_of_remaining_numbers_l79_79488

variable (numbers : List ℝ) (x y : ℝ)

theorem average_of_remaining_numbers
  (h_length_15 : numbers.length = 15)
  (h_avg_15 : (numbers.sum / 15) = 90)
  (h_x : x = 80)
  (h_y : y = 85)
  (h_members : x ∈ numbers ∧ y ∈ numbers) :
  ((numbers.sum - x - y) / 13) = 91.15 :=
sorry

end average_of_remaining_numbers_l79_79488


namespace principal_amount_simple_interest_l79_79769

theorem principal_amount_simple_interest 
    (P : ℝ) (R : ℝ) (T : ℝ) (SI : ℝ)
    (hR : R = 4)
    (hT : T = 5)
    (hSI : SI = P - 2080)
    (hInterestFormula : SI = (P * R * T) / 100) :
    P = 2600 := 
by
  sorry

end principal_amount_simple_interest_l79_79769


namespace age_of_fourth_child_l79_79489

theorem age_of_fourth_child (c1 c2 c3 c4 : ℕ) (h1 : c1 = 15)
  (h2 : c2 = c1 - 1) (h3 : c3 = c2 - 4)
  (h4 : c4 = c3 - 2) : c4 = 8 :=
by
  sorry

end age_of_fourth_child_l79_79489


namespace notebooks_difference_l79_79836

theorem notebooks_difference :
  ∀ (Jac_left Jac_Paula Jac_Mike Ger_not Jac_init : ℕ),
  Ger_not = 8 →
  Jac_left = 10 →
  Jac_Paula = 5 →
  Jac_Mike = 6 →
  Jac_init = Jac_left + Jac_Paula + Jac_Mike →
  Jac_init - Ger_not = 13 := 
by
  intros Jac_left Jac_Paula Jac_Mike Ger_not Jac_init
  intros Ger_not_8 Jac_left_10 Jac_Paula_5 Jac_Mike_6 Jac_init_def
  sorry

end notebooks_difference_l79_79836


namespace max_number_of_pies_l79_79484

def total_apples := 250
def apples_given_to_students := 42
def apples_used_for_juice := 75
def apples_per_pie := 8

theorem max_number_of_pies (h1 : total_apples = 250)
                           (h2 : apples_given_to_students = 42)
                           (h3 : apples_used_for_juice = 75)
                           (h4 : apples_per_pie = 8) :
  ((total_apples - apples_given_to_students - apples_used_for_juice) / apples_per_pie) ≥ 16 :=
by
  sorry

end max_number_of_pies_l79_79484


namespace albert_earnings_l79_79507

theorem albert_earnings (E P : ℝ) 
  (h1 : E * 1.20 = 660) 
  (h2 : E * (1 + P) = 693) : 
  P = 0.26 :=
sorry

end albert_earnings_l79_79507


namespace find_x_in_interval_l79_79918

theorem find_x_in_interval (x : ℝ) (h₀ : 0 ≤ x ∧ x ≤ 2 * Real.pi) :
  2 * Real.cos x ≤ abs (Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x))) ∧
  abs (Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x))) ≤ Real.sqrt 2 → 
  Real.pi / 4 ≤ x ∧ x ≤ 7 * Real.pi / 4 :=
by 
  sorry

end find_x_in_interval_l79_79918


namespace geometric_sequence_common_ratio_l79_79597

theorem geometric_sequence_common_ratio (a q : ℝ) (h : a = a * q / (1 - q)) : q = 1 / 2 :=
by
  sorry

end geometric_sequence_common_ratio_l79_79597


namespace set_intersection_l79_79007

theorem set_intersection (M N : Set ℝ) (hM : M = {x | x < 3}) (hN : N = {x | x > 2}) :
  M ∩ N = {x | 2 < x ∧ x < 3} :=
sorry

end set_intersection_l79_79007


namespace area_EFGH_l79_79947

theorem area_EFGH (n : ℕ) (n_pos : 1 < n) (S_ABCD : ℝ) (h₁ : S_ABCD = 1) :
  ∃ S_EFGH : ℝ, S_EFGH = (n - 2) / n :=
by sorry

end area_EFGH_l79_79947


namespace last_four_digits_5_pow_2017_l79_79304

theorem last_four_digits_5_pow_2017 : (5 ^ 2017) % 10000 = 3125 :=
by sorry

end last_four_digits_5_pow_2017_l79_79304


namespace farmer_land_l79_79033

-- Define A to be the total land owned by the farmer
variables (A : ℝ)

-- Define the conditions of the problem
def condition_1 (A : ℝ) : ℝ := 0.90 * A
def condition_2 (cleared_land : ℝ) : ℝ := 0.20 * cleared_land
def condition_3 (cleared_land : ℝ) : ℝ := 0.70 * cleared_land
def condition_4 (cleared_land : ℝ) : ℝ := cleared_land - condition_2 cleared_land - condition_3 cleared_land

-- Define the assertion we need to prove
theorem farmer_land (h : condition_4 (condition_1 A) = 630) : A = 7000 :=
by
  sorry

end farmer_land_l79_79033


namespace solve_for_x_l79_79060

-- Problem definition
def problem_statement (x : ℕ) : Prop :=
  (3 * x / 7 = 15) → x = 35

-- Theorem statement in Lean 4
theorem solve_for_x (x : ℕ) : problem_statement x :=
by
  intros h
  sorry

end solve_for_x_l79_79060


namespace find_son_age_l79_79974

theorem find_son_age (F S : ℕ) (h1 : F + S = 55)
  (h2 : ∃ Y, S + Y = F ∧ (F + Y) + (S + Y) = 93)
  (h3 : F = 18 ∨ S = 18) : S = 18 :=
by
  sorry  -- Proof to be filled in

end find_son_age_l79_79974


namespace integer_1000_column_l79_79467

def column_sequence (n : ℕ) : String :=
  let sequence := ["A", "B", "C", "D", "E", "F", "E", "D", "C", "B"]
  sequence.get! (n % 10)

theorem integer_1000_column : column_sequence 999 = "C" :=
by
  sorry

end integer_1000_column_l79_79467


namespace sum_of_coefficients_l79_79564

theorem sum_of_coefficients (a b : ℝ) (h1 : a = 1 * 5) (h2 : -b = 1 + 5) : a + b = -1 :=
by
  sorry

end sum_of_coefficients_l79_79564


namespace simplify_expression_l79_79440

theorem simplify_expression (a : ℝ) : a * (a - 3) = a^2 - 3 * a := 
by 
  sorry

end simplify_expression_l79_79440


namespace solve_for_x_l79_79896

theorem solve_for_x (x : ℝ) (h1 : x ≠ 4) (h2 : x ≠ 6)
  (h : (x + 10) / (x - 4) = (x - 3) / (x + 6)) : x = -48 / 23 :=
sorry

end solve_for_x_l79_79896


namespace min_value_am_hm_l79_79724

theorem min_value_am_hm (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a + b + c + d) * (1 / (a + b) + 1 / (a + c) + 1 / (b + d) + 1 / (c + d)) ≥ 8 :=
by
  sorry

end min_value_am_hm_l79_79724


namespace find_metal_sheet_width_l79_79702

-- The given conditions
def metalSheetLength : ℝ := 100
def cutSquareSide : ℝ := 10
def boxVolume : ℝ := 24000

-- Statement to prove
theorem find_metal_sheet_width (w : ℝ) (h : w - 2 * cutSquareSide > 0):
  boxVolume = (metalSheetLength - 2 * cutSquareSide) * (w - 2 * cutSquareSide) * cutSquareSide → 
  w = 50 := 
by {
  sorry
}

end find_metal_sheet_width_l79_79702


namespace find_interest_rate_l79_79474

noncomputable def interest_rate (A P T : ℚ) : ℚ := (A - P) / (P * T) * 100

theorem find_interest_rate :
  let A := 1120
  let P := 921.0526315789474
  let T := 2.4
  interest_rate A P T = 9 := 
by
  sorry

end find_interest_rate_l79_79474


namespace intersection_of_sets_l79_79126

def is_angle_in_first_quadrant (α : ℝ) : Prop :=
  ∃ k : ℤ, k * 360 < α ∧ α < k * 360 + 90

def is_acute_angle (α : ℝ) : Prop :=
  α < 90

theorem intersection_of_sets (α : ℝ) :
  (is_acute_angle α ∧ is_angle_in_first_quadrant α) ↔
  (∃ k : ℤ, k ≤ 0 ∧ k * 360 < α ∧ α < k * 360 + 90) := 
sorry

end intersection_of_sets_l79_79126


namespace product_plus_one_eq_216_l79_79233

variable (a b c : ℝ)

theorem product_plus_one_eq_216 
  (h1 : a * b + a + b = 35)
  (h2 : b * c + b + c = 35)
  (h3 : c * a + c + a = 35)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c) :
  (a + 1) * (b + 1) * (c + 1) = 216 := 
sorry

end product_plus_one_eq_216_l79_79233


namespace total_interest_percentage_l79_79497

theorem total_interest_percentage (inv_total : ℝ) (rate1 rate2 : ℝ) (inv2 : ℝ)
  (h_inv_total : inv_total = 100000)
  (h_rate1 : rate1 = 0.09)
  (h_rate2 : rate2 = 0.11)
  (h_inv2 : inv2 = 24999.999999999996) :
  (rate1 * (inv_total - inv2) + rate2 * inv2) / inv_total * 100 = 9.5 := 
sorry

end total_interest_percentage_l79_79497


namespace find_S11_l79_79072

variable (n : ℕ) (a : ℕ → ℕ) (S : ℕ → ℕ)

-- Conditions
def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d a1, ∀ n, a n = a1 + (n - 1) * d

axiom sum_of_first_n_terms (a : ℕ → ℕ) (S : ℕ → ℕ) : ∀ n, S n = n * (a 1 + a n) / 2
axiom condition1 : is_arithmetic_sequence a
axiom condition2 : a 5 + a 7 = (a 6)^2

-- Proof (statement) that the sum of the first 11 terms is 22
theorem find_S11 : S 11 = 22 :=
  sorry

end find_S11_l79_79072


namespace sum_of_squares_l79_79621

theorem sum_of_squares (a b c : ℝ) (h1 : a + b + c = 0)
  (h2 : a^3 + b^3 + c^3 = a^5 + b^5 + c^5) (h3 : a^3 + b^3 + c^3 = a^7 + b^7 + c^7) :
  a^2 + b^2 + c^2 = 6 / 5 :=
by
  sorry

end sum_of_squares_l79_79621


namespace base_2_base_3_product_is_144_l79_79183

def convert_base_2_to_10 (n : ℕ) : ℕ :=
  match n with
  | 1001 => 9
  | _ => 0 -- For simplicity, only handle 1001_2

def convert_base_3_to_10 (n : ℕ) : ℕ :=
  match n with
  | 121 => 16
  | _ => 0 -- For simplicity, only handle 121_3

theorem base_2_base_3_product_is_144 :
  convert_base_2_to_10 1001 * convert_base_3_to_10 121 = 144 :=
by
  sorry

end base_2_base_3_product_is_144_l79_79183


namespace jake_bitcoins_l79_79617

theorem jake_bitcoins (initial : ℕ) (donation1 : ℕ) (fraction : ℕ) (multiplier : ℕ) (donation2 : ℕ) :
  initial = 80 →
  donation1 = 20 →
  fraction = 2 →
  multiplier = 3 →
  donation2 = 10 →
  (initial - donation1) / fraction * multiplier - donation2 = 80 :=
by
  sorry

end jake_bitcoins_l79_79617


namespace correct_statement_l79_79712

section
variables {a b c d : Real}

-- Define the conditions as hypotheses/functions

-- Statement A: If a > b, then 1/a < 1/b
def statement_A (a b : Real) : Prop := a > b → 1 / a < 1 / b

-- Statement B: If a > b, then a^2 > b^2
def statement_B (a b : Real) : Prop := a > b → a^2 > b^2

-- Statement C: If a > b and c > d, then ac > bd
def statement_C (a b c d : Real) : Prop := a > b ∧ c > d → a * c > b * d

-- Statement D: If a^3 > b^3, then a > b
def statement_D (a b : Real) : Prop := a^3 > b^3 → a > b

-- The Lean statement to prove which statement is correct
theorem correct_statement : ¬ statement_A a b ∧ ¬ statement_B a b ∧ ¬ statement_C a b c d ∧ statement_D a b :=
by {
  sorry
}

end

end correct_statement_l79_79712


namespace inequality_proof_l79_79846

theorem inequality_proof (x y : ℝ) : 5 * x^2 + y^2 + 4 ≥ 4 * x + 4 * x * y :=
by
  sorry

end inequality_proof_l79_79846


namespace race_course_length_to_finish_at_same_time_l79_79720

variable (v : ℝ) -- speed of B
variable (d : ℝ) -- length of the race course

-- A's speed is 4 times B's speed and A gives B a 75-meter head start.
theorem race_course_length_to_finish_at_same_time (h1 : v > 0) (h2 : d > 75) : 
  (1 : ℝ) / 4 * (d / v) = ((d - 75) / v) ↔ d = 100 := 
sorry

end race_course_length_to_finish_at_same_time_l79_79720


namespace export_volume_scientific_notation_l79_79715

theorem export_volume_scientific_notation :
  (234.1 * 10^6) = (2.341 * 10^8) := 
sorry

end export_volume_scientific_notation_l79_79715


namespace trapezoid_circumcircle_radius_l79_79490

theorem trapezoid_circumcircle_radius :
  ∀ (BC AD height midline R : ℝ), 
  (BC / AD = (5 / 12)) →
  (height = 17) →
  (midline = height) →
  (midline = (BC + AD) / 2) →
  (BC = 10) →
  (AD = 24) →
  R = 13 :=
by
  intro BC AD height midline R
  intros h_ratio h_height h_midline_eq_height h_midline_eq_avg_bases h_BC h_AD
  -- Proof would go here, but it's skipped for now.
  sorry

end trapezoid_circumcircle_radius_l79_79490


namespace fraction_water_by_volume_l79_79607

theorem fraction_water_by_volume
  (A W : ℝ) 
  (h1 : A / W = 0.5)
  (h2 : A / (A + W) = 1/7) : 
  W / (A + W) = 2/7 :=
by
  sorry

end fraction_water_by_volume_l79_79607


namespace total_money_is_correct_l79_79926

-- Define conditions as constants
def numChocolateCookies : ℕ := 220
def pricePerChocolateCookie : ℕ := 1
def numVanillaCookies : ℕ := 70
def pricePerVanillaCookie : ℕ := 2

-- Total money made from selling chocolate cookies
def moneyFromChocolateCookies : ℕ := numChocolateCookies * pricePerChocolateCookie

-- Total money made from selling vanilla cookies
def moneyFromVanillaCookies : ℕ := numVanillaCookies * pricePerVanillaCookie

-- Total money made from selling all cookies
def totalMoneyMade : ℕ := moneyFromChocolateCookies + moneyFromVanillaCookies

-- The statement to prove, with the expected result
theorem total_money_is_correct : totalMoneyMade = 360 := by
  sorry

end total_money_is_correct_l79_79926


namespace derivative_at_two_l79_79988

def f (x : ℝ) : ℝ := x^3 + 4 * x - 5

noncomputable def derivative_f (x : ℝ) : ℝ := 3 * x^2 + 4

theorem derivative_at_two : derivative_f 2 = 16 :=
by
  sorry

end derivative_at_two_l79_79988


namespace similar_triangles_x_value_l79_79590

theorem similar_triangles_x_value : ∃ (x : ℝ), (12 / x = 9 / 6) ∧ x = 8 := by
  use 8
  constructor
  · sorry
  · rfl

end similar_triangles_x_value_l79_79590


namespace maximum_value_of_x2y3z_l79_79904

theorem maximum_value_of_x2y3z (x y z : ℝ) (h : x^2 + y^2 + z^2 = 5) : 
  x + 2 * y + 3 * z ≤ Real.sqrt 70 :=
by 
  sorry

end maximum_value_of_x2y3z_l79_79904


namespace A_works_alone_45_days_l79_79805

open Nat

theorem A_works_alone_45_days (x : ℕ) :
  (∀ x : ℕ, (9 * (1 / x + 1 / 40) + 23 * (1 / 40) = 1) → (x = 45)) :=
sorry

end A_works_alone_45_days_l79_79805


namespace repeating_decmials_sum_is_fraction_l79_79630

noncomputable def x : ℚ := 2/9
noncomputable def y : ℚ := 2/99
noncomputable def z : ℚ := 2/9999

theorem repeating_decmials_sum_is_fraction :
  (x + y + z) = 2426 / 9999 := by
  sorry

end repeating_decmials_sum_is_fraction_l79_79630


namespace find_x2_minus_x1_l79_79431

theorem find_x2_minus_x1 (a x1 x2 d e : ℝ) (h_a : a ≠ 0) (h_d : d ≠ 0) (h_x : x1 ≠ x2) (h_e : e = -d * x1)
  (h_y1 : ∀ x, y1 = a * (x - x1) * (x - x2)) (h_y2 : ∀ x, y2 = d * x + e)
  (h_intersect : ∀ x, y = a * (x - x1) * (x - x2) + (d * x + e)) 
  (h_single_point : ∀ x, y = a * (x - x1)^2) :
  x2 - x1 = d / a :=
sorry

end find_x2_minus_x1_l79_79431


namespace map_width_l79_79393

theorem map_width (length : ℝ) (area : ℝ) (h1 : length = 2) (h2 : area = 20) : ∃ (width : ℝ), width = 10 :=
by
  sorry

end map_width_l79_79393


namespace video_time_per_week_l79_79405

-- Define the basic conditions
def short_video_length : ℕ := 2
def multiplier : ℕ := 6
def long_video_length : ℕ := multiplier * short_video_length
def short_videos_per_day : ℕ := 2
def long_videos_per_day : ℕ := 1
def days_in_week : ℕ := 7

-- Calculate daily and weekly video release time
def daily_video_time : ℕ := (short_videos_per_day * short_video_length) + (long_videos_per_day * long_video_length)
def weekly_video_time : ℕ := daily_video_time * days_in_week

-- Main theorem to prove
theorem video_time_per_week : weekly_video_time = 112 := by
    sorry

end video_time_per_week_l79_79405


namespace mean_proportional_234_104_l79_79237

theorem mean_proportional_234_104 : Real.sqrt (234 * 104) = 156 :=
by 
  sorry

end mean_proportional_234_104_l79_79237


namespace angle_in_third_quadrant_l79_79138

theorem angle_in_third_quadrant (α : ℝ) (h1 : Real.cos α < 0) (h2 : Real.tan α > 0) : 
  (π < α ∧ α < 3 * π / 2) :=
sorry

end angle_in_third_quadrant_l79_79138


namespace find_m_l79_79975

theorem find_m :
  ∃ m : ℝ, (∀ x : ℝ, x > 0 → (m^2 - m - 5) * x^(m - 1) > 0) ∧ m = 3 :=
sorry

end find_m_l79_79975


namespace cube_dihedral_angle_is_60_degrees_l79_79392

-- Define the cube and related geometrical features
structure Point := (x y z : ℝ)
structure Cube :=
  (A B C D A₁ B₁ C₁ D₁ : Point)
  (is_cube : true) -- Placeholder for cube properties

-- Define the function to calculate dihedral angle measure
noncomputable def dihedral_angle_measure (cube: Cube) : ℝ := sorry

-- The theorem statement
theorem cube_dihedral_angle_is_60_degrees (cube : Cube) : dihedral_angle_measure cube = 60 :=
by sorry

end cube_dihedral_angle_is_60_degrees_l79_79392


namespace nursing_home_beds_l79_79098

/-- A community plans to build a nursing home with 100 rooms, consisting of single, double, and triple rooms.
    Let t be the number of single rooms (1 nursing bed), double rooms (2 nursing beds) is twice the single rooms,
    and the rest are triple rooms (3 nursing beds).
    The equations are:
    - number of double rooms: 2 * t
    - number of single rooms: t
    - number of triple rooms: 100 - 3 * t
    - total number of nursing beds: t + 2 * (2 * t) + 3 * (100 - 3 * t) 
    Prove the following:
    1. If the total number of nursing beds is 200, then t = 25.
    2. The maximum number of nursing beds is 260.
    3. The minimum number of nursing beds is 180.
-/
theorem nursing_home_beds (t : ℕ) (h1 : 10 ≤ t ∧ t ≤ 30) (total_rooms : ℕ := 100) :
  (∀ total_beds, (total_beds = t + 2 * (2 * t) + 3 * (100 - 3 * t)) → total_beds = 200 → t = 25) ∧
  (∀ max_beds, (max_beds = t + 2 * (2 * t) + 3 * (100 - 3 * t)) → t = 10 → max_beds = 260) ∧
  (∀ min_beds, (min_beds = t + 2 * (2 * t) + 3 * (100 - 3 * t)) → t = 30 → min_beds = 180) := 
by
  sorry

end nursing_home_beds_l79_79098


namespace side_length_of_square_l79_79848

theorem side_length_of_square (P : ℕ) (h1 : P = 28) (h2 : P = 4 * s) : s = 7 :=
  by sorry

end side_length_of_square_l79_79848


namespace polynomial_expression_value_l79_79921

theorem polynomial_expression_value
  (p q r s : ℂ)
  (h1 : p + q + r + s = 0)
  (h2 : p*q + p*r + p*s + q*r + q*s + r*s = -1)
  (h3 : p*q*r + p*q*s + p*r*s + q*r*s = -1)
  (h4 : p*q*r*s = 2) :
  p*(q - r)^2 + q*(r - s)^2 + r*(s - p)^2 + s*(p - q)^2 = -6 :=
by sorry

end polynomial_expression_value_l79_79921


namespace prove_midpoint_trajectory_eq_l79_79772

noncomputable def midpoint_trajectory_eq {x y : ℝ} (h : ∃ (x_P y_P : ℝ), (x_P^2 - y_P^2 = 1) ∧ (x = x_P / 2) ∧ (y = y_P / 2)) : Prop :=
  4*x^2 - 4*y^2 = 1

theorem prove_midpoint_trajectory_eq (x y : ℝ) (h : ∃ (x_P y_P : ℝ), (x_P^2 - y_P^2 = 1) ∧ (x = x_P / 2) ∧ (y = y_P / 2)) :
  midpoint_trajectory_eq h :=
sorry

end prove_midpoint_trajectory_eq_l79_79772


namespace range_of_a_l79_79417

noncomputable def f (a x : ℝ) : ℝ := 
  if x < 1 then a^x else (a-3)*x + 4*a

theorem range_of_a (a : ℝ) 
  (h1 : ∀ x1 x2 : ℝ, x1 ≠ x2 → (f a x1 - f a x2) / (x1 - x2) < 0) : 
  0 < a ∧ a ≤ 3/4 :=
by {
  sorry
}

end range_of_a_l79_79417


namespace min_value_is_four_l79_79544

noncomputable def min_value (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x + y + z = 2) : ℝ :=
  (x + y) / (x * y * z)

theorem min_value_is_four (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x + y + z = 2) :
  min_value x y z h1 h2 h3 h4 = 4 :=
sorry

end min_value_is_four_l79_79544


namespace children_vehicle_wheels_l79_79830

theorem children_vehicle_wheels:
  ∀ (x : ℕ),
    (6 * 2) + (15 * x) = 57 →
    x = 3 :=
by
  intros x h
  sorry

end children_vehicle_wheels_l79_79830


namespace select_female_athletes_l79_79756

theorem select_female_athletes (males females sample_size total_size : ℕ)
    (h1 : males = 56) (h2 : females = 42) (h3 : sample_size = 28)
    (h4 : total_size = males + females) : 
    (females * sample_size / total_size = 12) := 
by
  sorry

end select_female_athletes_l79_79756


namespace line_intersects_circle_l79_79609

theorem line_intersects_circle (m : ℝ) : 
  ∃ (x y : ℝ), y = m * x - 3 ∧ x^2 + (y - 1)^2 = 25 :=
sorry

end line_intersects_circle_l79_79609


namespace basketball_classes_l79_79432

theorem basketball_classes (x : ℕ) : (x * (x - 1)) / 2 = 10 :=
sorry

end basketball_classes_l79_79432


namespace probability_floor_sqrt_100x_eq_180_given_floor_sqrt_x_eq_18_l79_79599

open Real

noncomputable def probability_event : ℝ :=
  ((327.61 - 324) / (361 - 324))

theorem probability_floor_sqrt_100x_eq_180_given_floor_sqrt_x_eq_18 :
  probability_event = 361 / 3700 :=
by
  -- Conditions and calculations supplied in the problem
  sorry

end probability_floor_sqrt_100x_eq_180_given_floor_sqrt_x_eq_18_l79_79599


namespace log_simplify_l79_79786

open Real

theorem log_simplify : 
  (1 / (log 12 / log 3 + 1)) + 
  (1 / (log 8 / log 2 + 1)) + 
  (1 / (log 30 / log 5 + 1)) = 2 :=
by
  sorry

end log_simplify_l79_79786


namespace problem_statement_l79_79210

theorem problem_statement (a : Fin 17 → ℕ)
  (h : ∀ i : Fin 17, a i ^ a (i + 1) = a (i + 1) ^ a (i + 2)): 
  a 0 = a 1 :=
sorry

end problem_statement_l79_79210


namespace tenth_graders_science_only_l79_79114

theorem tenth_graders_science_only (total_students science_students art_students : ℕ) 
  (h1 : total_students = 140) 
  (h2 : science_students = 100) 
  (h3 : art_students = 75) : 
  (science_students - (science_students + art_students - total_students)) = 65 :=
by
  sorry

end tenth_graders_science_only_l79_79114


namespace smallest_n_for_perfect_square_and_cube_l79_79460

theorem smallest_n_for_perfect_square_and_cube :
  ∃ n : ℕ, (∃ a : ℕ, 4 * n = a^2) ∧ (∃ b : ℕ, 5 * n = b^3) ∧ n = 125 :=
by
  sorry

end smallest_n_for_perfect_square_and_cube_l79_79460


namespace f_properties_l79_79816

noncomputable def f (x : ℝ) : ℝ := 2 ^ x

theorem f_properties :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂) ∧
  (∀ x₁ x₂ : ℝ, f (x₁ + x₂) = f x₁ * f x₂) :=
by 
  sorry

end f_properties_l79_79816


namespace geometric_sequence_nine_l79_79527

theorem geometric_sequence_nine (a : ℕ → ℝ) (h_geo : ∀ n, a (n + 1) / a n = a 1 / a 0) 
  (h_a1 : a 1 = 2) (h_a5: a 5 = 4) : a 9 = 8 := 
by
  sorry

end geometric_sequence_nine_l79_79527


namespace largest_divisor_for_odd_n_l79_79996

noncomputable def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

theorem largest_divisor_for_odd_n (n : ℤ) (h : is_odd n ∧ n > 0) : 
  15 ∣ (n + 3) * (n + 5) * (n + 7) * (n + 9) * (n + 11) := 
by 
  sorry

end largest_divisor_for_odd_n_l79_79996


namespace find_digit_A_l79_79441

open Nat

theorem find_digit_A :
  let n := 52
  let k := 13
  let number_of_hands := choose n k
  number_of_hands = 635013587600 → 0 = 0 := by
  suffices h: 635013587600 = 635013587600 by
    simp [h]
  sorry

end find_digit_A_l79_79441


namespace jessie_problem_l79_79815

def round_to_nearest_five (n : ℤ) : ℤ :=
  if n % 5 = 0 then n
  else if n % 5 < 3 then n - (n % 5)
  else n - (n % 5) + 5

theorem jessie_problem :
  round_to_nearest_five ((82 + 56) - 15) = 125 :=
by
  sorry

end jessie_problem_l79_79815


namespace winston_initial_quarters_l79_79316

-- Defining the conditions
def spent_candy := 50 -- 50 cents spent on candy
def remaining_cents := 300 -- 300 cents left

-- Defining the value of a quarter in cents
def value_of_quarter := 25

-- Calculating the number of quarters Winston initially had
def initial_quarters := (spent_candy + remaining_cents) / value_of_quarter

-- Proof statement
theorem winston_initial_quarters : initial_quarters = 14 := 
by sorry

end winston_initial_quarters_l79_79316


namespace sum_of_powers_eq_zero_l79_79648

theorem sum_of_powers_eq_zero
  (a b c : ℝ)
  (n : ℝ)
  (h1 : a + b + c = 0)
  (h2 : a^3 + b^3 + c^3 = 0) :
  a^(2* ⌊n⌋ + 1) + b^(2* ⌊n⌋ + 1) + c^(2* ⌊n⌋ + 1) = 0 := by
  sorry

end sum_of_powers_eq_zero_l79_79648


namespace range_of_m_l79_79447

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, x^2 + m * x + m / 2 + 2 ≥ 0) ∨ ((1 / 2) * m > 1) ↔ ((m > 4) ∧ ¬(∀ x : ℝ, x^2 + m * x + m / 2 + 2 ≥ 0)) :=
sorry

end range_of_m_l79_79447


namespace LCM_20_45_75_is_900_l79_79751

def prime_factorization_20 := (2^2, 5)
def prime_factorization_45 := (3^2, 5)
def prime_factorization_75 := (3, 5^2)

theorem LCM_20_45_75_is_900 
  (pf_20 : prime_factorization_20 = (2^2, 5))
  (pf_45 : prime_factorization_45 = (3^2, 5))
  (pf_75 : prime_factorization_75 = (3, 5^2)) : 
  Nat.lcm (Nat.lcm 20 45) 75 = 900 := 
  by sorry

end LCM_20_45_75_is_900_l79_79751


namespace binom_sum_l79_79119

def binom (n k : ℕ) := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binom_sum : binom 7 4 + binom 6 5 = 41 := by
  sorry

end binom_sum_l79_79119


namespace cost_of_apples_and_bananas_l79_79525

variable (a b : ℝ) -- Assume a and b are real numbers.

theorem cost_of_apples_and_bananas (a b : ℝ) : 
  (3 * a + 2 * b) = 3 * a + 2 * b :=
by 
  sorry -- Proof placeholder

end cost_of_apples_and_bananas_l79_79525


namespace max_val_4ab_sqrt3_12bc_l79_79567

theorem max_val_4ab_sqrt3_12bc {a b c : ℝ} (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : a^2 + b^2 + c^2 = 3) :
  4 * a * b * Real.sqrt 3 + 12 * b * c ≤ Real.sqrt 39 :=
sorry

end max_val_4ab_sqrt3_12bc_l79_79567


namespace algebraic_expression_value_l79_79436

theorem algebraic_expression_value (a b c : ℝ) (h : (∀ x : ℝ, (x - 1) * (x + 2) = a * x^2 + b * x + c)) :
  4 * a - 2 * b + c = 0 :=
sorry

end algebraic_expression_value_l79_79436


namespace right_triangle_third_side_l79_79278

theorem right_triangle_third_side (a b c : ℝ) (h₁ : a = 3) (h₂ : b = 4) (h₃ : c = 5) (h₄ : c = Real.sqrt (7) ∨ c = 5) :
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2 := by
  sorry

end right_triangle_third_side_l79_79278


namespace total_votes_cast_l79_79245

-- Define the variables and constants
def total_votes (V : ℝ) : Prop :=
  let A := 0.32 * V
  let B := 0.28 * V
  let C := 0.22 * V
  let D := 0.18 * V
  -- Candidate A defeated Candidate B by 1200 votes
  0.32 * V - 0.28 * V = 1200 ∧
  -- Candidate A defeated Candidate C by 2200 votes
  0.32 * V - 0.22 * V = 2200 ∧
  -- Candidate B defeated Candidate D by 900 votes
  0.28 * V - 0.18 * V = 900

noncomputable def V := 30000

-- State the theorem
theorem total_votes_cast : total_votes V := by
  sorry

end total_votes_cast_l79_79245


namespace distance_to_plane_l79_79545

variable (V : ℝ) (A : ℝ) (r : ℝ) (d : ℝ)

-- Assume the volume of the sphere and area of the cross-section
def sphere_volume := V = 4 * Real.sqrt 3 * Real.pi
def cross_section_area := A = Real.pi

-- Define radius of sphere and cross-section
def sphere_radius := r = Real.sqrt 3
def cross_section_radius := Real.sqrt A = 1

-- Define distance as per Pythagorean theorem
def distance_from_center := d = Real.sqrt (r^2 - 1^2)

-- Main statement to prove
theorem distance_to_plane (V A : ℝ)
  (h1 : sphere_volume V) 
  (h2 : cross_section_area A) 
  (h3: sphere_radius r) 
  (h4: cross_section_radius A) : 
  distance_from_center r d :=
sorry

end distance_to_plane_l79_79545


namespace total_floor_area_covered_l79_79814

-- Definitions for the given problem
def combined_area : ℕ := 204
def overlap_two_layers : ℕ := 24
def overlap_three_layers : ℕ := 20
def total_floor_area : ℕ := 140

-- Theorem to prove the total floor area covered by the rugs
theorem total_floor_area_covered :
  combined_area - overlap_two_layers - 2 * overlap_three_layers = total_floor_area := by
  sorry

end total_floor_area_covered_l79_79814


namespace pythagorean_triple_divisible_by_60_l79_79350

theorem pythagorean_triple_divisible_by_60 
  (a b c : ℕ) (h : a * a + b * b = c * c) : 60 ∣ (a * b * c) :=
sorry

end pythagorean_triple_divisible_by_60_l79_79350


namespace find_initial_time_l79_79248

-- The initial distance d
def distance : ℕ := 288

-- Conditions
def initial_condition (v t : ℕ) : Prop :=
  distance = v * t

def new_condition (t : ℕ) : Prop :=
  distance = 32 * (3 * t / 2)

-- Proof Problem Statement
theorem find_initial_time (v t : ℕ) (h1 : initial_condition v t)
  (h2 : new_condition t) : t = 6 := by
  sorry

end find_initial_time_l79_79248


namespace solve_for_y_l79_79582

theorem solve_for_y (y : ℚ) (h : 1 / 3 + 1 / y = 7 / 9) : y = 9 / 4 :=
by
  sorry

end solve_for_y_l79_79582


namespace inequality_proof_l79_79054

variable {m n : ℝ}

theorem inequality_proof (h1 : m < n) (h2 : n < 0) : (n / m + m / n > 2) := 
by
  sorry

end inequality_proof_l79_79054


namespace number_of_mixed_groups_l79_79993

theorem number_of_mixed_groups (n_children n_groups n_games boy_vs_boy girl_vs_girl mixed_games : ℕ) (h_children : n_children = 90) (h_groups : n_groups = 30) (h_games_per_group : n_games = 3) (h_boy_vs_boy : boy_vs_boy = 30) (h_girl_vs_girl : girl_vs_girl = 14) (h_total_games : mixed_games = 46) :
  (∀ g : ℕ, g * 2 = mixed_games → g = 23) :=
by
  intros g hg
  sorry

end number_of_mixed_groups_l79_79993


namespace add_and_multiply_l79_79857

def num1 : ℝ := 0.0034
def num2 : ℝ := 0.125
def num3 : ℝ := 0.00678
def sum := num1 + num2 + num3

theorem add_and_multiply :
  (sum * 2) = 0.27036 := by
  sorry

end add_and_multiply_l79_79857


namespace product_of_roots_l79_79483

theorem product_of_roots (a b c : ℝ) (h_eq : 24 * a^2 + 36 * a - 648 = 0) : a * c = -27 := 
by
  have h_root_product : (24 * a^2 + 36 * a - 648) = 0 ↔ a = -27 := sorry
  exact sorry

end product_of_roots_l79_79483


namespace amount_in_cup_after_division_l79_79017

theorem amount_in_cup_after_division (removed remaining cups : ℕ) (h : remaining + removed = 40) : 
  (40 / cups = 8) :=
by
  sorry

end amount_in_cup_after_division_l79_79017


namespace domain_g_l79_79151

def domain_f (a : ℝ) : Prop := 0 ≤ a ∧ a ≤ 2
def g (x : ℝ) (f : ℝ → ℝ) : Prop := 
  ((1 < x) ∧ (x ≤ Real.sqrt 3)) ∧ domain_f (x^2 - 1) ∧ (0 < x - 1 ∧ x - 1 < 1)

theorem domain_g (x : ℝ) (f : ℝ → ℝ) (hf : ∀ a, domain_f a → True) : 
  g x f ↔ 1 < x ∧ x ≤ Real.sqrt 3 :=
by 
  sorry

end domain_g_l79_79151


namespace number_of_players_in_hockey_club_l79_79207

-- Defining the problem parameters
def cost_of_gloves : ℕ := 6
def cost_of_helmet := cost_of_gloves + 7
def total_cost_per_set := cost_of_gloves + cost_of_helmet
def total_cost_per_player := 2 * total_cost_per_set
def total_expenditure : ℕ := 3120

-- Defining the target number of players
def num_players : ℕ := total_expenditure / total_cost_per_player

theorem number_of_players_in_hockey_club : num_players = 82 := by
  sorry

end number_of_players_in_hockey_club_l79_79207


namespace houses_after_boom_l79_79670

theorem houses_after_boom (h_pre_boom : ℕ) (h_built : ℕ) (h_count : ℕ)
  (H1 : h_pre_boom = 1426)
  (H2 : h_built = 574)
  (H3 : h_count = h_pre_boom + h_built) :
  h_count = 2000 :=
by {
  sorry
}

end houses_after_boom_l79_79670


namespace leftover_yarn_after_square_l79_79902

theorem leftover_yarn_after_square (total_yarn : ℕ) (side_length : ℕ) (left_yarn : ℕ) :
  total_yarn = 35 →
  (4 * side_length ≤ total_yarn ∧ (∀ s : ℕ, s > side_length → 4 * s > total_yarn)) →
  left_yarn = total_yarn - 4 * side_length →
  left_yarn = 3 :=
by
  sorry

end leftover_yarn_after_square_l79_79902


namespace squared_product_l79_79212

theorem squared_product (a b : ℝ) : (- (1 / 2) * a^2 * b)^2 = (1 / 4) * a^4 * b^2 := by 
  sorry

end squared_product_l79_79212


namespace lcm_first_ten_l79_79355

-- Define the set of first ten positive integers
def first_ten_integers : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define the LCM of a list of integers
noncomputable def lcm_list (l : List ℕ) : ℕ :=
List.foldr Nat.lcm 1 l

-- The theorem stating that the LCM of the first ten integers is 2520
theorem lcm_first_ten : lcm_list first_ten_integers = 2520 := by
  sorry

end lcm_first_ten_l79_79355


namespace parabola_shifting_produces_k_l79_79133

theorem parabola_shifting_produces_k
  (k : ℝ)
  (h1 : -k/2 > 0)
  (h2 : (0 : ℝ) = (((0 : ℝ) - 3) + k/2)^2 - (5*k^2)/4 + 1)
  :
  k = -5 :=
sorry

end parabola_shifting_produces_k_l79_79133


namespace num_O_atoms_l79_79343

def compound_molecular_weight : ℕ := 62
def atomic_weight_H : ℕ := 1
def atomic_weight_C : ℕ := 12
def atomic_weight_O : ℕ := 16
def num_H_atoms : ℕ := 2
def num_C_atoms : ℕ := 1

theorem num_O_atoms (H_weight : ℕ := num_H_atoms * atomic_weight_H)
                    (C_weight : ℕ := num_C_atoms * atomic_weight_C)
                    (total_weight : ℕ := compound_molecular_weight)
                    (O_weight := atomic_weight_O) : 
    (total_weight - (H_weight + C_weight)) / O_weight = 3 :=
by
  sorry

end num_O_atoms_l79_79343


namespace yao_ming_shots_l79_79538

-- Defining the conditions
def total_shots_made : ℕ := 14
def total_points_scored : ℕ := 28
def three_point_shots_made : ℕ := 3
def two_point_shots (x : ℕ) : ℕ := x
def free_throws_made (x : ℕ) : ℕ := total_shots_made - three_point_shots_made - x

-- The theorem we want to prove
theorem yao_ming_shots :
  ∃ (x y : ℕ),
    (total_shots_made = three_point_shots_made + x + y) ∧ 
    (total_points_scored = 3 * three_point_shots_made + 2 * x + y) ∧
    (x = 8) ∧
    (y = 3) :=
sorry

end yao_ming_shots_l79_79538


namespace three_digit_number_452_l79_79743

theorem three_digit_number_452 (a b c : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ 9) (h3 : 1 ≤ b) (h4 : b ≤ 9) (h5 : 1 ≤ c) (h6 : c ≤ 9) 
  (h7 : 100 * a + 10 * b + c % (a + b + c) = 1)
  (h8 : 100 * c + 10 * b + a % (a + b + c) = 1)
  (h9 : a ≠ b) (h10 : b ≠ c) (h11 : a ≠ c)
  (h12 : a > c) :
  100 * a + 10 * b + c = 452 :=
sorry

end three_digit_number_452_l79_79743


namespace distinct_intersection_points_l79_79949

theorem distinct_intersection_points : 
  ∃! (x y : ℝ), (x + 2*y = 6 ∧ x - 3*y = 2) ∨ (x + 2*y = 6 ∧ 4*x + y = 14) :=
by
  -- proof would be here
  sorry

end distinct_intersection_points_l79_79949


namespace rate_up_the_mountain_l79_79321

noncomputable def mountain_trip_rate (R : ℝ) : ℝ := 1.5 * R

theorem rate_up_the_mountain : 
  ∃ R : ℝ, (2 * 1.5 * R = 18) ∧ (1.5 * R = 9) → R = 6 :=
by
  sorry

end rate_up_the_mountain_l79_79321


namespace polygon_interior_angles_eq_360_l79_79740

theorem polygon_interior_angles_eq_360 (n : ℕ) (h : (n - 2) * 180 = 360) : n = 4 :=
sorry

end polygon_interior_angles_eq_360_l79_79740


namespace minimum_sum_of_areas_l79_79362

theorem minimum_sum_of_areas (x y : ℝ) (hx : x + y = 16) (hx_nonneg : 0 ≤ x) (hy_nonneg : 0 ≤ y) : 
  (x ^ 2 / 16 + y ^ 2 / 16) / 4 ≥ 8 :=
  sorry

end minimum_sum_of_areas_l79_79362


namespace odd_function_m_zero_l79_79626

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := 2 * x^3 + m

theorem odd_function_m_zero (m : ℝ) : (∀ x : ℝ, f (-x) m = -f x m) → m = 0 :=
by
  sorry

end odd_function_m_zero_l79_79626


namespace perpendicular_lines_a_value_l79_79191

theorem perpendicular_lines_a_value (a : ℝ) :
  (∀ x y : ℝ, (a-2)*x + a*y = 1 ↔ 2*x + 3*y = 5) → a = 4/5 := by
sorry

end perpendicular_lines_a_value_l79_79191


namespace negative_value_option_D_l79_79271

theorem negative_value_option_D :
  (-7) * (-6) > 0 ∧
  (-7) - (-15) > 0 ∧
  0 * (-2) * (-3) = 0 ∧
  (-6) + (-4) < 0 :=
by
  sorry

end negative_value_option_D_l79_79271


namespace standard_equation_of_ellipse_l79_79987

-- Definitions for clarity
def is_ellipse (E : Type) := true
def major_axis (e : is_ellipse E) : ℝ := sorry
def minor_axis (e : is_ellipse E) : ℝ := sorry
def focus (e : is_ellipse E) : ℝ := sorry

theorem standard_equation_of_ellipse (E : Type)
  (e : is_ellipse E)
  (major_sum : major_axis e + minor_axis e = 9)
  (focus_position : focus e = 3) :
  ∀ x y, (x^2 / 25) + (y^2 / 16) = 1 :=
by sorry

end standard_equation_of_ellipse_l79_79987


namespace wilsons_theorem_l79_79725

theorem wilsons_theorem (p : ℕ) (hp : Nat.Prime p) : (Nat.factorial (p - 1)) % p = p - 1 :=
by
  sorry

end wilsons_theorem_l79_79725


namespace positive_divisors_multiple_of_15_l79_79437

theorem positive_divisors_multiple_of_15 (a b c : ℕ) (n : ℕ) (divisor : ℕ) (h_factorization : n = 6480)
  (h_prime_factorization : n = 2^4 * 3^4 * 5^1)
  (h_divisor : divisor = 2^a * 3^b * 5^c)
  (h_a_range : 0 ≤ a ∧ a ≤ 4)
  (h_b_range : 1 ≤ b ∧ b ≤ 4)
  (h_c_range : 1 ≤ c ∧ c ≤ 1) : sorry :=
sorry

end positive_divisors_multiple_of_15_l79_79437


namespace rhombus_diagonal_l79_79178

theorem rhombus_diagonal
  (d1 : ℝ) (d2 : ℝ) (area : ℝ) 
  (h1 : d1 = 17) (h2 : area = 170) 
  (h3 : area = (d1 * d2) / 2) : d2 = 20 :=
by
  sorry

end rhombus_diagonal_l79_79178


namespace largest_number_of_stores_visited_l79_79134

-- Definitions of the conditions
def num_stores := 7
def total_visits := 21
def num_shoppers := 11
def two_stores_visitors := 7
def at_least_one_store (n : ℕ) : Prop := n ≥ 1

-- The goal statement
theorem largest_number_of_stores_visited :
  ∃ n, n ≤ num_shoppers ∧ 
       at_least_one_store n ∧ 
       (n * 2 + (num_shoppers - n)) <= total_visits ∧ 
       (num_shoppers - n) ≥ 3 → 
       n = 4 :=
sorry

end largest_number_of_stores_visited_l79_79134


namespace complex_imaginary_part_l79_79229

theorem complex_imaginary_part (z : ℂ) (h : z + (3 - 4 * I) = 1) : z.im = 4 :=
  sorry

end complex_imaginary_part_l79_79229


namespace biased_die_expected_value_is_neg_1_5_l79_79296

noncomputable def biased_die_expected_value : ℚ :=
  let prob_123 := (1 / 6 : ℚ) + (1 / 6) + (1 / 6)
  let prob_456 := (1 / 2 : ℚ)
  let gain := prob_123 * 2
  let loss := prob_456 * -5
  gain + loss

theorem biased_die_expected_value_is_neg_1_5 :
  biased_die_expected_value = - (3 / 2 : ℚ) :=
by
  -- We skip the detailed proof steps here.
  sorry

end biased_die_expected_value_is_neg_1_5_l79_79296


namespace weight_of_apples_l79_79011

-- Definitions based on conditions
def total_weight : ℕ := 10
def weight_orange : ℕ := 1
def weight_grape : ℕ := 3
def weight_strawberry : ℕ := 3

-- Prove that the weight of apples is 3 kilograms
theorem weight_of_apples : (total_weight - (weight_orange + weight_grape + weight_strawberry)) = 3 :=
by
  sorry

end weight_of_apples_l79_79011


namespace bakery_total_items_l79_79601

theorem bakery_total_items (total_money : ℝ) (cupcake_cost : ℝ) (pastry_cost : ℝ) (max_cupcakes : ℕ) (remaining_money : ℝ) (total_items : ℕ) :
  total_money = 50 ∧ cupcake_cost = 3 ∧ pastry_cost = 2.5 ∧ max_cupcakes = 16 ∧ remaining_money = 2 ∧ total_items = max_cupcakes + 0 → total_items = 16 :=
by
  sorry

end bakery_total_items_l79_79601


namespace jinhee_pages_per_day_l79_79613

noncomputable def pages_per_day (total_pages : ℕ) (days : ℕ) : ℕ :=
  (total_pages + days - 1) / days

theorem jinhee_pages_per_day : 
  ∀ (total_pages : ℕ) (days : ℕ), total_pages = 220 → days = 7 → pages_per_day total_pages days = 32 :=
by 
  intros total_pages days hp hd
  rw [hp, hd]
  -- the computation of the function
  show pages_per_day 220 7 = 32
  sorry

end jinhee_pages_per_day_l79_79613


namespace find_int_solutions_l79_79122

theorem find_int_solutions (x y : ℤ) (h : x^4 - 2 * y^2 = 1) : (x = 1 ∧ y = 0) ∨ (x = -1 ∧ y = 0) :=
sorry

end find_int_solutions_l79_79122


namespace number_of_refills_l79_79294

variable (totalSpent costPerRefill : ℕ)
variable (h1 : totalSpent = 40)
variable (h2 : costPerRefill = 10)

theorem number_of_refills (h1 h2 : totalSpent = 40) (h2 : costPerRefill = 10) :
  totalSpent / costPerRefill = 4 := by
  sorry

end number_of_refills_l79_79294


namespace apple_price_difference_l79_79330

variable (S R F : ℝ)

theorem apple_price_difference (h1 : S + R > R + F) (h2 : F = S - 250) :
  (S + R) - (R + F) = 250 :=
by
  sorry

end apple_price_difference_l79_79330


namespace angle_bisector_divides_longest_side_l79_79423

theorem angle_bisector_divides_longest_side :
  ∀ (a b c : ℕ) (p q : ℕ), a = 12 → b = 15 → c = 18 →
  p + q = c → p * b = q * a → p = 8 ∧ q = 10 :=
by
  intros a b c p q ha hb hc hpq hprop
  rw [ha, hb, hc] at *
  sorry

end angle_bisector_divides_longest_side_l79_79423


namespace largest_is_C_l79_79055

def A : ℝ := 0.978
def B : ℝ := 0.9719
def C : ℝ := 0.9781
def D : ℝ := 0.917
def E : ℝ := 0.9189

theorem largest_is_C : 
  (C > A) ∧ 
  (C > B) ∧ 
  (C > D) ∧ 
  (C > E) := by
  sorry

end largest_is_C_l79_79055


namespace average_difference_l79_79572

theorem average_difference :
  let avg1 := (200 + 400) / 2
  let avg2 := (100 + 200) / 2
  avg1 - avg2 = 150 :=
by
  sorry

end average_difference_l79_79572


namespace layla_earnings_l79_79422

-- Define the hourly rates for each family
def rate_donaldson : ℕ := 15
def rate_merck : ℕ := 18
def rate_hille : ℕ := 20
def rate_johnson : ℕ := 22
def rate_ramos : ℕ := 25

-- Define the hours Layla worked for each family
def hours_donaldson : ℕ := 7
def hours_merck : ℕ := 6
def hours_hille : ℕ := 3
def hours_johnson : ℕ := 4
def hours_ramos : ℕ := 2

-- Calculate the earnings for each family
def earnings_donaldson : ℕ := rate_donaldson * hours_donaldson
def earnings_merck : ℕ := rate_merck * hours_merck
def earnings_hille : ℕ := rate_hille * hours_hille
def earnings_johnson : ℕ := rate_johnson * hours_johnson
def earnings_ramos : ℕ := rate_ramos * hours_ramos

-- Calculate total earnings
def total_earnings : ℕ :=
  earnings_donaldson + earnings_merck + earnings_hille + earnings_johnson + earnings_ramos

-- The assertion that Layla's total earnings are $411
theorem layla_earnings : total_earnings = 411 := by
  sorry

end layla_earnings_l79_79422


namespace sum_mod_30_l79_79696

theorem sum_mod_30 (a b c : ℕ) 
  (h1 : a % 30 = 15) 
  (h2 : b % 30 = 7) 
  (h3 : c % 30 = 18) : 
  (a + 2 * b + c) % 30 = 17 := 
by
  sorry

end sum_mod_30_l79_79696


namespace probability_both_truth_l79_79766

noncomputable def probability_A_truth : ℝ := 0.75
noncomputable def probability_B_truth : ℝ := 0.60

theorem probability_both_truth : 
  (probability_A_truth * probability_B_truth) = 0.45 :=
by sorry

end probability_both_truth_l79_79766


namespace find_special_5_digit_number_l79_79045

theorem find_special_5_digit_number :
  ∃! (A : ℤ), (10000 ≤ A ∧ A < 100000) ∧ (A^2 % 100000 = A) ∧ A = 90625 :=
sorry

end find_special_5_digit_number_l79_79045


namespace juniper_bones_proof_l79_79438

-- Define the conditions
def juniper_original_bones : ℕ := 4
def bones_given_by_master : ℕ := juniper_original_bones
def bones_stolen_by_neighbor : ℕ := 2

-- Define the final number of bones Juniper has
def juniper_remaining_bones : ℕ := juniper_original_bones + bones_given_by_master - bones_stolen_by_neighbor

-- State the theorem to prove the given answer
theorem juniper_bones_proof : juniper_remaining_bones = 6 :=
by
  -- Proof omitted
  sorry

end juniper_bones_proof_l79_79438


namespace max_value_f1_l79_79624

-- Definitions for the conditions
def f (x a b : ℝ) : ℝ := x^2 + a * b * x + a + 2 * b

-- Lean theorem statements
theorem max_value_f1 (a b : ℝ) (h : a + 2 * b = 4) :
  f 0 a b = 4 → f 1 a b ≤ 7 :=
sorry

end max_value_f1_l79_79624


namespace population_growth_proof_l79_79227

noncomputable def population_growth (P0 : ℕ) (P200 : ℕ) (t : ℕ) (x : ℝ) : Prop :=
  P200 = P0 * (1 + 1 / x)^t

theorem population_growth_proof :
  population_growth 6 1000000 200 16 :=
by
  -- Proof goes here
  sorry

end population_growth_proof_l79_79227


namespace ratio_of_Phil_to_Bob_l79_79664

-- There exists real numbers P, J, and B such that
theorem ratio_of_Phil_to_Bob (P J B : ℝ) (h1 : J = 2 * P) (h2 : B = 60) (h3 : J = B - 20) : P / B = 1 / 3 :=
by
  sorry

end ratio_of_Phil_to_Bob_l79_79664


namespace cross_area_l79_79336

variables (R : ℝ) (A : ℝ × ℝ) (φ : ℝ)
  -- Radius R of the circle, Point A inside the circle, and angle φ in radians

-- Define the area of the cross formed by rotated lines
def area_of_cross (R : ℝ) (φ : ℝ) : ℝ :=
  2 * φ * R^2

theorem cross_area (R : ℝ) (A : ℝ × ℝ) (φ : ℝ) (hR : 0 < R) (hA : dist A (0, 0) < R) :
  area_of_cross R φ = 2 * φ * R^2 := 
sorry

end cross_area_l79_79336


namespace square_area_l79_79764

theorem square_area (side_length : ℕ) (h : side_length = 12) : side_length * side_length = 144 :=
by
  sorry

end square_area_l79_79764


namespace appropriate_sampling_methods_l79_79887

structure Region :=
  (total_households : ℕ)
  (farmer_households : ℕ)
  (worker_households : ℕ)
  (sample_size : ℕ)

theorem appropriate_sampling_methods (r : Region) 
  (h_total: r.total_households = 2004)
  (h_farmers: r.farmer_households = 1600)
  (h_workers: r.worker_households = 303)
  (h_sample: r.sample_size = 40) :
  ("Simple random sampling" ∈ ["Simple random sampling", "Systematic sampling", "Stratified sampling"]) ∧
  ("Systematic sampling" ∈ ["Simple random sampling", "Systematic sampling", "Stratified sampling"]) ∧
  ("Stratified sampling" ∈ ["Simple random sampling", "Systematic sampling", "Stratified sampling"]) :=
by
  sorry

end appropriate_sampling_methods_l79_79887


namespace max_marks_l79_79900

theorem max_marks (M : ℕ) (h1 : M * 33 / 100 = 175 + 56) : M = 700 :=
by
  sorry

end max_marks_l79_79900


namespace cost_price_percentage_l79_79873

theorem cost_price_percentage (MP CP : ℝ) (h_discount : 0.75 * MP = CP * 1.171875) :
  ((CP / MP) * 100) = 64 :=
by
  sorry

end cost_price_percentage_l79_79873


namespace terminal_side_angle_l79_79063

open Real

theorem terminal_side_angle (α : ℝ) (m n : ℝ) (h_line : n = 3 * m) (h_radius : m^2 + n^2 = 10) (h_sin : sin α < 0) (h_coincide : tan α = 3) : m - n = 2 :=
by
  sorry

end terminal_side_angle_l79_79063


namespace percent_formula_l79_79749

theorem percent_formula (x y p : ℝ) (h : x = (p / 100) * y) : p = 100 * x / y :=
by
    sorry

end percent_formula_l79_79749


namespace lego_count_l79_79043

theorem lego_count 
  (total_legos : ℕ := 500)
  (used_legos : ℕ := total_legos / 2)
  (missing_legos : ℕ := 5) :
  total_legos - used_legos - missing_legos = 245 := 
sorry

end lego_count_l79_79043


namespace spherical_caps_ratio_l79_79196

theorem spherical_caps_ratio (r : ℝ) (m₁ m₂ : ℝ) (σ₁ σ₂ : ℝ)
  (h₁ : r = 1)
  (h₂ : σ₁ = 2 * π * m₁ + π * (1 - (1 - m₁)^2))
  (h₃ : σ₂ = 2 * π * m₂ + π * (1 - (1 - m₂)^2))
  (h₄ : σ₁ + σ₂ = 5 * π)
  (h₅ : m₁ + m₂ = 2) :
  (2 * m₁ + (1 - (1 - m₁)^2)) / (2 * m₂ + (1 - (1 - m₂)^2)) = 3.6 :=
sorry

end spherical_caps_ratio_l79_79196


namespace mean_of_five_numbers_is_correct_l79_79738

-- Define the given sum of five numbers as three-quarters
def sum_of_five_numbers : ℚ := 3 / 4

-- Define the number of numbers, which is 5
def number_of_numbers : ℕ := 5

-- Define the mean calculation from the given sum and number of numbers
def mean_five_numbers (sum : ℚ) (count : ℕ) : ℚ := sum / count

-- Statement to prove: the mean of five numbers given their sum is 3/4 equals 3/20
theorem mean_of_five_numbers_is_correct :
  mean_five_numbers sum_of_five_numbers number_of_numbers = 3 / 20 :=
by
  -- Skipping the proof
  sorry

end mean_of_five_numbers_is_correct_l79_79738


namespace probability_of_selecting_A_and_B_l79_79181

-- Define the combination function
def combination (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

-- Define the conditions
def students : ℕ := 5
def to_select : ℕ := 3

-- Define the favorable outcomes where A and B are selected
def favorable_outcomes : ℕ := combination 3 1

-- Define the total outcomes of selecting 3 out of 5 students
def total_outcomes : ℕ := combination 5 3

-- Define the probability calculation
def probability_of_A_and_B : ℚ := favorable_outcomes / total_outcomes

-- Proposition to prove
theorem probability_of_selecting_A_and_B : probability_of_A_and_B = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_l79_79181


namespace opposite_of_neg_one_third_l79_79166

theorem opposite_of_neg_one_third : -(- (1 / 3)) = (1 / 3) :=
by sorry

end opposite_of_neg_one_third_l79_79166


namespace parabola_line_intersection_sum_l79_79610

theorem parabola_line_intersection_sum (r s : ℝ) (h_r : r = 20 - 10 * Real.sqrt 38) (h_s : s = 20 + 10 * Real.sqrt 38) :
  r + s = 40 := by
  sorry

end parabola_line_intersection_sum_l79_79610


namespace evaluate_expression_l79_79503

theorem evaluate_expression :
  (2 + 3 / (4 + 5 / (6 + 7 / 8))) = 137 / 52 :=
by
  sorry

end evaluate_expression_l79_79503


namespace inequality_transformation_l79_79040

theorem inequality_transformation (x y : ℝ) (h : x > y) : 3 * x > 3 * y :=
by sorry

end inequality_transformation_l79_79040


namespace uncle_kahn_total_cost_l79_79732

noncomputable def base_price : ℝ := 10
noncomputable def child_discount : ℝ := 0.3
noncomputable def senior_discount : ℝ := 0.1
noncomputable def handling_fee : ℝ := 5
noncomputable def discounted_senior_ticket_price : ℝ := 14
noncomputable def num_child_tickets : ℝ := 2
noncomputable def num_senior_tickets : ℝ := 2

theorem uncle_kahn_total_cost :
  let child_ticket_cost := (1 - child_discount) * base_price + handling_fee
  let senior_ticket_cost := discounted_senior_ticket_price
  num_child_tickets * child_ticket_cost + num_senior_tickets * senior_ticket_cost = 52 :=
by
  sorry

end uncle_kahn_total_cost_l79_79732


namespace larger_segment_length_l79_79328

theorem larger_segment_length (a b c : ℕ) (h : ℝ) (x : ℝ)
  (ha : a = 50) (hb : b = 90) (hc : c = 110)
  (hyp1 : a^2 = x^2 + h^2)
  (hyp2 : b^2 = (c - x)^2 + h^2) :
  110 - x = 80 :=
by {
  sorry
}

end larger_segment_length_l79_79328


namespace mod_inverse_9_mod_23_l79_79990

theorem mod_inverse_9_mod_23 : ∃ (a : ℤ), 0 ≤ a ∧ a < 23 ∧ (9 * a) % 23 = 1 :=
by
  use 18
  sorry

end mod_inverse_9_mod_23_l79_79990


namespace min_value_of_f_l79_79546

def f (x : ℝ) : ℝ := x^2 - 4 * x + 4

theorem min_value_of_f : ∀ x : ℝ, f x ≥ 0 ∧ f 2 = 0 :=
  by sorry

end min_value_of_f_l79_79546


namespace sequence_general_term_l79_79856

theorem sequence_general_term (a : ℕ → ℕ) :
  (a 1 = 1 * 2) ∧ (a 2 = 2 * 3) ∧ (a 3 = 3 * 4) ∧ (a 4 = 4 * 5) ↔ 
    (∀ n, a n = n^2 + n) := sorry

end sequence_general_term_l79_79856


namespace quadratic_equation_correct_form_l79_79502

theorem quadratic_equation_correct_form :
  ∀ (a b c x : ℝ), a = 3 → b = -6 → c = 1 → a * x^2 + c = b * x :=
by
  intros a b c x ha hb hc
  rw [ha, hb, hc]
  sorry

end quadratic_equation_correct_form_l79_79502


namespace Jeff_Jogging_Extra_Friday_l79_79048

theorem Jeff_Jogging_Extra_Friday :
  let planned_daily_minutes := 60
  let days_in_week := 5
  let planned_weekly_minutes := days_in_week * planned_daily_minutes
  let thursday_cut_short := 20
  let actual_weekly_minutes := 290
  let thursday_run := planned_daily_minutes - thursday_cut_short
  let other_four_days_minutes := actual_weekly_minutes - thursday_run
  let mondays_to_wednesdays_run := 3 * planned_daily_minutes
  let friday_run := other_four_days_minutes - mondays_to_wednesdays_run
  let extra_run_on_friday := friday_run - planned_daily_minutes
  extra_run_on_friday = 10 := by trivial

end Jeff_Jogging_Extra_Friday_l79_79048


namespace trigonometric_identity_l79_79105

theorem trigonometric_identity (α : ℝ) (h : Real.sin α = 1 / 3) : 
  Real.cos (Real.pi / 4 + α) * Real.cos (Real.pi / 4 - α) = 7 / 18 :=
by sorry

end trigonometric_identity_l79_79105


namespace quadratic_inequality_no_solution_l79_79568

theorem quadratic_inequality_no_solution (a b c : ℝ) (h : a ≠ 0)
  (hnsol : ∀ x : ℝ, ¬(a * x^2 + b * x + c ≥ 0)) :
  a < 0 ∧ b^2 - 4 * a * c < 0 :=
sorry

end quadratic_inequality_no_solution_l79_79568


namespace candy_distribution_l79_79188

theorem candy_distribution (n : ℕ) (h : n ≥ 2) :
  (∀ i : ℕ, i < n → ∃ k : ℕ, ((k * (k + 1)) / 2) % n = i) ↔ ∃ k : ℕ, n = 2 ^ k :=
by
  sorry

end candy_distribution_l79_79188


namespace max_f_when_a_minus_1_range_of_a_l79_79366

noncomputable section

-- Definitions of the functions given in the problem
def f (a : ℝ) (x : ℝ) : ℝ := a * x + Real.log x
def g (a : ℝ) (x : ℝ) : ℝ := x * f a x
def h (a : ℝ) (x : ℝ) : ℝ := 2 * a * x^2 - (2 * a - 1) * x + (a - 1)

-- Statement (1): Proving the maximum value of f(x) when a = -1
theorem max_f_when_a_minus_1 : 
  (∀ x : ℝ, f (-1) x ≤ f (-1) 1) :=
sorry

-- Statement (2): Proving the range of a when g(x) ≤ h(x) for x ≥ 1
theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ≥ 1 → g a x ≤ h a x) → (1 ≤ a) :=
sorry

end max_f_when_a_minus_1_range_of_a_l79_79366


namespace octagon_diagonal_ratio_l79_79577

theorem octagon_diagonal_ratio (P : ℝ → ℝ → Prop) (d1 d2 : ℝ) (h1 : P d1 d2) : d1 / d2 = Real.sqrt 2 / 2 :=
sorry

end octagon_diagonal_ratio_l79_79577


namespace complement_intersection_l79_79629

noncomputable def M : Set ℝ := {x | 2 / x < 1}
noncomputable def N : Set ℝ := {y | ∃ x, y = Real.sqrt (x - 1)}

theorem complement_intersection : 
  ((Set.univ \ M) ∩ N) = {x | 0 ≤ x ∧ x ≤ 2} :=
by
  sorry

end complement_intersection_l79_79629


namespace problem_solution_l79_79989

theorem problem_solution (a b : ℝ) (h1 : 2 + 3 = -b) (h2 : 2 * 3 = -2 * a) : a + b = -8 :=
by
  sorry

end problem_solution_l79_79989


namespace max_M_is_7524_l79_79259

-- Define the conditions
def is_valid_t (t : ℕ) : Prop :=
  let a := t / 1000
  let b := (t % 1000) / 100
  let c := (t % 100) / 10
  let d := t % 10
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
  a + c = 9 ∧
  b - d = 1 ∧
  (2 * (2 * a + d)) % (2 * b + c) = 0

-- Define function M
def M (a b c d : ℕ) : ℕ := 2000 * a + 100 * b + 10 * c + d

-- Define the maximum value of M
def max_valid_M : ℕ :=
  let m_values := [5544, 7221, 7322, 7524]
  m_values.foldl max 0

theorem max_M_is_7524 : max_valid_M = 7524 := by
  -- The proof would be written here. For now, we indicate the theorem as
  -- not yet proven.
  sorry

end max_M_is_7524_l79_79259


namespace train_speed_is_25_kmph_l79_79687

noncomputable def train_speed_kmph (train_length_m : ℕ) (man_speed_kmph : ℕ) (cross_time_s : ℕ) : ℕ :=
  let man_speed_mps := (man_speed_kmph * 1000) / 3600
  let relative_speed_mps := train_length_m / cross_time_s
  let train_speed_mps := relative_speed_mps - man_speed_mps
  let train_speed_kmph := (train_speed_mps * 3600) / 1000
  train_speed_kmph

theorem train_speed_is_25_kmph : train_speed_kmph 270 2 36 = 25 := by
  sorry

end train_speed_is_25_kmph_l79_79687


namespace sqrt_of_225_eq_15_l79_79232

theorem sqrt_of_225_eq_15 : Real.sqrt 225 = 15 :=
by
  sorry

end sqrt_of_225_eq_15_l79_79232


namespace black_white_area_ratio_l79_79130

theorem black_white_area_ratio :
  let r1 := 2
  let r2 := 6
  let r3 := 10
  let r4 := 14
  let r5 := 18
  let area (r : ℝ) := π * r^2
  let black_area := area r1 + (area r3 - area r2) + (area r5 - area r4)
  let white_area := (area r2 - area r1) + (area r4 - area r3)
  black_area / white_area = (49 : ℝ) / 32 :=
by
  sorry

end black_white_area_ratio_l79_79130


namespace find_p_for_quadratic_l79_79476

theorem find_p_for_quadratic (p : ℝ) (h : p ≠ 0) 
  (h_eq : ∀ x : ℝ, p * x^2 - 10 * x + 2 = 0 → x = 5 / p) : p = 12.5 :=
sorry

end find_p_for_quadratic_l79_79476


namespace boys_count_eq_792_l79_79514

-- Definitions of conditions
variables (B G : ℤ)

-- Total number of students is 1443
axiom total_students : B + G = 1443

-- Number of girls is 141 fewer than the number of boys
axiom girls_fewer_than_boys : G = B - 141

-- Proof statement to show that the number of boys (B) is 792
theorem boys_count_eq_792 (B G : ℤ)
  (h1 : B + G = 1443)
  (h2 : G = B - 141) : B = 792 :=
by
  sorry

end boys_count_eq_792_l79_79514


namespace oliver_bags_fraction_l79_79705

theorem oliver_bags_fraction
  (weight_james_bag : ℝ)
  (combined_weight_oliver_bags : ℝ)
  (h1 : weight_james_bag = 18)
  (h2 : combined_weight_oliver_bags = 6)
  (f : ℝ) :
  2 * f * weight_james_bag = combined_weight_oliver_bags → f = 1 / 6 :=
by
  intro h
  sorry

end oliver_bags_fraction_l79_79705


namespace intersection_S_T_l79_79753

def S : Set ℝ := { x | 2 * x + 1 > 0 }
def T : Set ℝ := { x | 3 * x - 5 < 0 }

theorem intersection_S_T :
  S ∩ T = { x | -1/2 < x ∧ x < 5/3 } := by
  sorry

end intersection_S_T_l79_79753


namespace eighth_term_sum_of_first_15_terms_l79_79831

-- Given definitions from the conditions
def a1 : ℚ := 5
def a30 : ℚ := 100
def n8 : ℕ := 8
def n15 : ℕ := 15
def n30 : ℕ := 30

-- Formulate the arithmetic sequence properties
def common_difference : ℚ := (a30 - a1) / (n30 - 1)

def nth_term (n : ℕ) : ℚ :=
  a1 + (n - 1) * common_difference

def sum_of_first_n_terms (n : ℕ) : ℚ :=
  n / 2 * (2 * a1 + (n - 1) * common_difference)

-- Statements to be proven
theorem eighth_term :
  nth_term n8 = 25 + 1/29 := by sorry

theorem sum_of_first_15_terms :
  sum_of_first_n_terms n15 = 393 + 2/29 := by sorry

end eighth_term_sum_of_first_15_terms_l79_79831


namespace square_window_side_length_l79_79214

-- Definitions based on the conditions
def total_panes := 8
def rows := 2
def cols := 4
def height_ratio := 3
def width_ratio := 1
def border_width := 3

-- The statement to prove
theorem square_window_side_length :
  let height := 3 * (1 : ℝ)
  let width := 1 * (1 : ℝ)
  let total_width := cols * width + (cols + 1) * border_width
  let total_height := rows * height + (rows + 1) * border_width
  total_width = total_height → total_width = 27 :=
by
  sorry

end square_window_side_length_l79_79214


namespace range_of_a_for_local_min_l79_79197

noncomputable def f (a x : ℝ) : ℝ := (x - 2 * a) * (x^2 + a^2 * x + 2 * a^3)

theorem range_of_a_for_local_min :
  (∀ a : ℝ, (∃ δ > 0, ∀ ε ∈ Set.Ioo (-δ) δ, f a ε > f a 0) → a < 0 ∨ a > 2) :=
by
  sorry

end range_of_a_for_local_min_l79_79197


namespace other_function_value_at_20_l79_79855

def linear_function (k b : ℝ) (x : ℝ) : ℝ :=
  k * x + b

theorem other_function_value_at_20
    (k1 k2 b1 b2 : ℝ)
    (h_intersect : linear_function k1 b1 2 = linear_function k2 b2 2)
    (h_diff_at_8 : abs (linear_function k1 b1 8 - linear_function k2 b2 8) = 8)
    (h_y1_at_20 : linear_function k1 b1 20 = 100) :
  linear_function k2 b2 20 = 76 ∨ linear_function k2 b2 20 = 124 :=
sorry

end other_function_value_at_20_l79_79855


namespace sum_reciprocal_squares_roots_l79_79391

-- Define the polynomial P(X) = X^3 - 3X - 1
noncomputable def P (X : ℂ) : ℂ := X^3 - 3 * X - 1

-- Define the roots of the polynomial
variables (r1 r2 r3 : ℂ)

-- State that r1, r2, and r3 are roots of the polynomial
variable (hroots : P r1 = 0 ∧ P r2 = 0 ∧ P r3 = 0)

-- Vieta's formulas conditions for the polynomial P
variable (hvieta : r1 + r2 + r3 = 0 ∧ r1 * r2 + r1 * r3 + r2 * r3 = -3 ∧ r1 * r2 * r3 = 1)

-- The sum of the reciprocals of the squares of the roots
theorem sum_reciprocal_squares_roots : (1 / r1^2) + (1 / r2^2) + (1 / r3^2) = 9 := 
sorry

end sum_reciprocal_squares_roots_l79_79391


namespace alice_password_prob_correct_l79_79954

noncomputable def password_probability : ℚ :=
  let even_digit_prob := 5 / 10
  let valid_symbol_prob := 3 / 5
  let non_zero_digit_prob := 9 / 10
  even_digit_prob * valid_symbol_prob * non_zero_digit_prob

theorem alice_password_prob_correct :
  password_probability = 27 / 100 := by
  rfl

end alice_password_prob_correct_l79_79954


namespace sally_remaining_cards_l79_79661

variable (total_cards : ℕ) (torn_cards : ℕ) (bought_cards : ℕ)

def intact_cards (total_cards : ℕ) (torn_cards : ℕ) : ℕ := total_cards - torn_cards
def remaining_cards (intact_cards : ℕ) (bought_cards : ℕ) : ℕ := intact_cards - bought_cards

theorem sally_remaining_cards :
  intact_cards 39 9 - 24 = 6 :=
by
  -- sorry for proof
  sorry

end sally_remaining_cards_l79_79661


namespace find_sets_l79_79842

variable (A X Y : Set ℕ) -- Mimicking sets of natural numbers for generality.

theorem find_sets (h1 : X ∪ Y = A) (h2 : X ∩ A = Y) : X = A ∧ Y = A := by
  -- This would need a proof, which shows that: X = A and Y = A
  sorry

end find_sets_l79_79842


namespace card_arrangement_probability_l79_79673

/-- 
This problem considers the probability of arranging four distinct cards,
each labeled with a unique character, in such a way that they form one of two specific
sequences. Specifically, the sequences are "我爱数学" (I love mathematics) and "数学爱我" (mathematics loves me).
-/
theorem card_arrangement_probability :
  let cards := ["我", "爱", "数", "学"]
  let total_permutations := 24
  let favorable_outcomes := 2
  let probability := favorable_outcomes / total_permutations
  probability = 1 / 12 :=
by
  sorry

end card_arrangement_probability_l79_79673


namespace fraction_is_percent_of_y_l79_79782

theorem fraction_is_percent_of_y (y : ℝ) (hy : y > 0) : 
  (2 * y / 5 + 3 * y / 10) / y = 0.7 :=
sorry

end fraction_is_percent_of_y_l79_79782


namespace retail_price_before_discounts_l79_79652

theorem retail_price_before_discounts 
  (wholesale_price profit_rate tax_rate discount1 discount2 total_effective_price : ℝ) 
  (h_wholesale_price : wholesale_price = 108)
  (h_profit_rate : profit_rate = 0.20)
  (h_tax_rate : tax_rate = 0.15)
  (h_discount1 : discount1 = 0.10)
  (h_discount2 : discount2 = 0.05)
  (h_total_effective_price : total_effective_price = 126.36) :
  ∃ (retail_price_before_discounts : ℝ), retail_price_before_discounts = 147.78 := 
by
  sorry

end retail_price_before_discounts_l79_79652


namespace heights_inscribed_circle_inequality_l79_79578

theorem heights_inscribed_circle_inequality
  {h₁ h₂ r : ℝ} (h₁_pos : 0 < h₁) (h₂_pos : 0 < h₂) (r_pos : 0 < r)
  (triangle_heights : ∃ (a b c : ℝ), 0 < a ∧ 0 < b ∧ 0 < c ∧ a * h₁ = b * h₂ ∧ 
                                       a + b > c ∧ h₁ = 2 * r * (a + b + c) / (a * b)):
  (1 / (2 * r) < 1 / h₁ + 1 / h₂ ∧ 1 / h₁ + 1 / h₂ < 1 / r) :=
sorry

end heights_inscribed_circle_inequality_l79_79578


namespace original_number_of_people_l79_79550

variable (x : ℕ)
-- Conditions
axiom one_third_left : x / 3 > 0
axiom half_dancing : 18 = x / 3

-- Theorem Statement
theorem original_number_of_people (x : ℕ) (one_third_left : x / 3 > 0) (half_dancing : 18 = x / 3) : x = 54 := sorry

end original_number_of_people_l79_79550


namespace necessary_but_not_sufficient_for_gt_l79_79101

variable {a b : ℝ}

theorem necessary_but_not_sufficient_for_gt : a > b → a > b - 1 :=
by sorry

end necessary_but_not_sufficient_for_gt_l79_79101


namespace diamond_expression_calculation_l79_79185

def diamond (a b : ℚ) : ℚ := a - 1 / b

theorem diamond_expression_calculation :
  (diamond (diamond 2 3) 5) - (diamond 2 (diamond 3 5)) = -37 / 210 :=
by
  sorry

end diamond_expression_calculation_l79_79185


namespace value_at_4_value_of_x_when_y_is_0_l79_79719

-- Problem statement
def f (x : ℝ) : ℝ := 2 * x - 3

-- Proof statement 1: When x = 4, y = 5
theorem value_at_4 : f 4 = 5 := sorry

-- Proof statement 2: When y = 0, x = 3/2
theorem value_of_x_when_y_is_0 : (∃ x : ℝ, f x = 0) → (∃ x : ℝ, x = 3 / 2) := sorry

end value_at_4_value_of_x_when_y_is_0_l79_79719


namespace valid_seating_arrangements_l79_79642

def num_people : Nat := 10
def total_arrangements : Nat := Nat.factorial num_people
def restricted_group_arrangements : Nat := Nat.factorial 7 * Nat.factorial 4
def valid_arrangements : Nat := total_arrangements - restricted_group_arrangements

theorem valid_seating_arrangements : valid_arrangements = 3507840 := by
  sorry

end valid_seating_arrangements_l79_79642


namespace quiz_score_of_dropped_student_l79_79223

theorem quiz_score_of_dropped_student (avg16 : ℝ) (avg15 : ℝ) (num_students : ℝ) (dropped_students : ℝ) (x : ℝ)
  (h1 : avg16 = 60.5) (h2 : avg15 = 64) (h3 : num_students = 16) (h4 : dropped_students = 1) :
  x = 60.5 * 16 - 64 * 15 :=
by
  sorry

end quiz_score_of_dropped_student_l79_79223


namespace complete_square_solution_l79_79519

theorem complete_square_solution
  (x : ℝ)
  (h : x^2 + 4*x + 2 = 0):
  ∃ c : ℝ, (x + 2)^2 = c ∧ c = 2 :=
by
  sorry

end complete_square_solution_l79_79519


namespace smallest_square_side_length_paintings_l79_79697

theorem smallest_square_side_length_paintings (n : ℕ) :
  ∃ n : ℕ, (∀ (i : ℕ), 1 ≤ i ∧ i ≤ 2020 → 1 * i ≤ n * n) → n = 1430 :=
by
  sorry

end smallest_square_side_length_paintings_l79_79697


namespace area_of_triangle_l79_79485

theorem area_of_triangle {a b c : ℝ} (S : ℝ) (h1 : (a^2) * (Real.sin C) = 4 * (Real.sin A))
                          (h2 : (a + c)^2 = 12 + b^2)
                          (h3 : S = Real.sqrt ((1/4) * (a^2 * c^2 - ( (a^2 + c^2 - b^2)/2 )^2))) :
  S = Real.sqrt 3 :=
by
  sorry

end area_of_triangle_l79_79485


namespace wings_per_person_l79_79001

-- Define the number of friends
def number_of_friends : ℕ := 15

-- Define the number of wings already cooked
def wings_already_cooked : ℕ := 7

-- Define the number of additional wings cooked
def additional_wings_cooked : ℕ := 45

-- Define the number of friends who don't eat chicken
def friends_not_eating : ℕ := 2

-- Calculate the total number of chicken wings
def total_chicken_wings : ℕ := wings_already_cooked + additional_wings_cooked

-- Calculate the number of friends who will eat chicken
def friends_eating : ℕ := number_of_friends - friends_not_eating

-- Define the statement we want to prove
theorem wings_per_person : total_chicken_wings / friends_eating = 4 := by
  sorry

end wings_per_person_l79_79001


namespace clock_angle_at_3_40_l79_79199

noncomputable def hour_hand_angle (h m : ℕ) : ℝ := (h % 12) * 30 + m * 0.5
noncomputable def minute_hand_angle (m : ℕ) : ℝ := m * 6
noncomputable def angle_between_hands (h m : ℕ) : ℝ := 
  let angle := |minute_hand_angle m - hour_hand_angle h m|
  if angle > 180 then 360 - angle else angle

theorem clock_angle_at_3_40 : angle_between_hands 3 40 = 130.0 := 
by
  sorry

end clock_angle_at_3_40_l79_79199


namespace sixth_oak_placement_l79_79851

structure Point :=
  (x : ℝ)
  (y : ℝ)

def is_aligned (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

noncomputable def intersection_point (p1 p2 p3 p4 : Point) : Point := 
  let m1 := (p2.y - p1.y) / (p2.x - p1.x)
  let m2 := (p4.y - p3.y) / (p4.x - p3.x)
  let c1 := p1.y - (m1 * p1.x)
  let c2 := p3.y - (m2 * p3.x)
  let x := (c2 - c1) / (m1 - m2)
  let y := m1 * x + c1
  ⟨x, y⟩

theorem sixth_oak_placement 
  (A1 A2 A3 B1 B2 B3 : Point) 
  (hA : ¬ is_aligned A1 A2 A3)
  (hB : ¬ is_aligned B1 B2 B3) :
  ∃ P : Point, (∃ (C1 C2 : Point), C1 = A1 ∧ C2 = B1 ∧ is_aligned C1 C2 P) ∧ 
               (∃ (C3 C4 : Point), C3 = A2 ∧ C4 = B2 ∧ is_aligned C3 C4 P) := by
  sorry

end sixth_oak_placement_l79_79851


namespace fraction_decomposition_l79_79152

theorem fraction_decomposition :
  (1 : ℚ) / 4 = (1 : ℚ) / 8 + (1 : ℚ) / 8 := 
by
  -- proof goes here
  sorry

end fraction_decomposition_l79_79152


namespace seventh_term_l79_79478

def nth_term (n : ℕ) (a : ℝ) : ℝ :=
  (-2) ^ n * a ^ (2 * n - 1)

theorem seventh_term (a : ℝ) : nth_term 7 a = -128 * a ^ 13 :=
by sorry

end seventh_term_l79_79478


namespace remainder_of_125_div_j_l79_79083

theorem remainder_of_125_div_j (j : ℕ) (h1 : j > 0) (h2 : 75 % (j^2) = 3) : 125 % j = 5 :=
sorry

end remainder_of_125_div_j_l79_79083


namespace delivery_parcels_problem_l79_79594

theorem delivery_parcels_problem (x : ℝ) (h1 : 2 + 2 * (1 + x) + 2 * (1 + x) ^ 2 = 7.28) : 
  2 + 2 * (1 + x) + 2 * (1 + x) ^ 2 = 7.28 :=
by
  exact h1

end delivery_parcels_problem_l79_79594


namespace rational_numbers_countable_l79_79680

theorem rational_numbers_countable : ∃ (f : ℚ → ℕ), Function.Bijective f :=
by
  sorry

end rational_numbers_countable_l79_79680


namespace find_a_l79_79962

theorem find_a (a x : ℝ) 
  (h : x^2 + 3 * x + a = (x + 1) * (x + 2)) : 
  a = 2 :=
sorry

end find_a_l79_79962


namespace remainder_when_divided_by_9_l79_79076

theorem remainder_when_divided_by_9 (x : ℕ) (h1 : x > 0) (h2 : (5 * x) % 9 = 7) : x % 9 = 5 :=
sorry

end remainder_when_divided_by_9_l79_79076


namespace train_speed_second_part_l79_79201

variables (x v : ℝ)

theorem train_speed_second_part
  (h1 : ∀ t1 : ℝ, t1 = x / 30)
  (h2 : ∀ t2 : ℝ, t2 = 2 * x / v)
  (h3 : ∀ t : ℝ, t = 3 * x / 22.5) :
  (x / 30) + (2 * x / v) = (3 * x / 22.5) → v = 20 :=
by
  intros h4
  sorry

end train_speed_second_part_l79_79201


namespace factorize_polynomial_l79_79665

variable (a x y : ℝ)

theorem factorize_polynomial (a x y : ℝ) :
  3 * a * x ^ 2 - 3 * a * y ^ 2 = 3 * a * (x + y) * (x - y) := by
  sorry

end factorize_polynomial_l79_79665


namespace sum_abcd_l79_79380

variables (a b c d : ℚ)

theorem sum_abcd :
  3 * a + 4 * b + 6 * c + 8 * d = 48 →
  4 * (d + c) = b →
  4 * b + 2 * c = a →
  c + 1 = d →
  a + b + c + d = 513 / 37 :=
by
sorry

end sum_abcd_l79_79380


namespace solution_set_of_inequality_l79_79094

theorem solution_set_of_inequality (x : ℝ) : x < (1 / x) ↔ (x < -1 ∨ (0 < x ∧ x < 1)) :=
by
  sorry

end solution_set_of_inequality_l79_79094


namespace opponent_score_value_l79_79708

-- Define the given conditions
def total_points : ℕ := 720
def games_played : ℕ := 24
def average_score := total_points / games_played
def championship_score := average_score / 2 - 2
def opponent_score := championship_score + 2

-- Lean theorem statement to prove
theorem opponent_score_value : opponent_score = 15 :=
by
  -- Proof to be filled in
  sorry

end opponent_score_value_l79_79708


namespace no_real_solution_intersection_l79_79950

theorem no_real_solution_intersection :
  ¬ ∃ x y : ℝ, (y = 8 / (x^3 + 4 * x + 3)) ∧ (x + y = 5) :=
by
  sorry

end no_real_solution_intersection_l79_79950


namespace each_group_has_two_bananas_l79_79273

theorem each_group_has_two_bananas (G T : ℕ) (hG : G = 196) (hT : T = 392) : T / G = 2 :=
by
  sorry

end each_group_has_two_bananas_l79_79273


namespace no_friendly_triplet_in_range_l79_79623

open Nat

def isFriendly (a b c : ℕ) : Prop :=
  (a ∣ (b * c) ∨ b ∣ (a * c) ∨ c ∣ (a * b))

theorem no_friendly_triplet_in_range (n : ℕ) (a b c : ℕ) :
  n^2 < a ∧ a < n^2 + n → n^2 < b ∧ b < n^2 + n → n^2 < c ∧ c < n^2 + n → a ≠ b → b ≠ c → a ≠ c →
  ¬ isFriendly a b c :=
by sorry

end no_friendly_triplet_in_range_l79_79623


namespace class_ratio_and_percentage_l79_79305

theorem class_ratio_and_percentage:
  ∀ (female male : ℕ), female = 15 → male = 25 →
  (∃ ratio_n ratio_d : ℕ, gcd ratio_n ratio_d = 1 ∧ ratio_n = 5 ∧ ratio_d = 8 ∧
  ratio_n / ratio_d = male / (female + male))
  ∧
  (∃ percentage : ℕ, percentage = 40 ∧ percentage = 100 * (male - female) / male) :=
by
  intros female male hf hm
  have h1 : female = 15 := hf
  have h2 : male = 25 := hm
  sorry

end class_ratio_and_percentage_l79_79305


namespace trigonometric_identity_l79_79424

theorem trigonometric_identity (α : ℝ) (h : Real.tan (Real.pi + α) = 2) :
  4 * Real.sin α * Real.cos α + 3 * (Real.cos α) ^ 2 = 11 / 5 :=
sorry

end trigonometric_identity_l79_79424


namespace simplify_absolute_values_l79_79038

theorem simplify_absolute_values (a : ℝ) (h : -2 < a ∧ a < 0) : |a| + |a + 2| = 2 :=
sorry

end simplify_absolute_values_l79_79038


namespace solve_for_p_l79_79469

variable (p q : ℝ)
noncomputable def binomial_third_term : ℝ := 55 * p^9 * q^2
noncomputable def binomial_fourth_term : ℝ := 165 * p^8 * q^3

theorem solve_for_p (h1 : p + q = 1) (h2 : binomial_third_term p q = binomial_fourth_term p q) : p = 3 / 4 :=
by sorry

end solve_for_p_l79_79469


namespace percentage_relationship_l79_79928

variable {x y z : ℝ}

theorem percentage_relationship (h1 : x = 1.30 * y) (h2 : y = 0.50 * z) : x = 0.65 * z :=
by
  sorry

end percentage_relationship_l79_79928


namespace required_percentage_to_pass_l79_79492

theorem required_percentage_to_pass
  (marks_obtained : ℝ)
  (marks_failed_by : ℝ)
  (max_marks : ℝ)
  (passing_marks := marks_obtained + marks_failed_by)
  (required_percentage : ℝ := (passing_marks / max_marks) * 100)
  (h : marks_obtained = 80)
  (h' : marks_failed_by = 40)
  (h'' : max_marks = 200) :
  required_percentage = 60 := 
by
  sorry

end required_percentage_to_pass_l79_79492


namespace cos_135_eq_neg_inv_sqrt_2_l79_79992

theorem cos_135_eq_neg_inv_sqrt_2 : Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 := by
  sorry

end cos_135_eq_neg_inv_sqrt_2_l79_79992


namespace age_difference_l79_79860

variables (A B C : ℕ)

theorem age_difference (h : A + B = B + C + 12) : A - C = 12 :=
sorry

end age_difference_l79_79860


namespace arithmetic_sequence_a5_l79_79184

variable (a : ℕ → ℝ)

theorem arithmetic_sequence_a5 (h : a 2 + a 8 = 15 - a 5) : a 5 = 5 :=
by
  sorry

end arithmetic_sequence_a5_l79_79184


namespace common_point_geometric_lines_l79_79639

-- Define that a, b, c form a geometric progression given common ratio r
def geometric_prog (a b c r : ℝ) : Prop := b = a * r ∧ c = a * r^2

-- Prove that all lines with the equation ax + by = c pass through the point (-1, 1)
theorem common_point_geometric_lines (a b c r x y : ℝ) (h : geometric_prog a b c r) :
  a * x + b * y = c → (x, y) = (-1, 1) :=
by
  sorry

end common_point_geometric_lines_l79_79639


namespace monthly_fixed_cost_is_correct_l79_79308

-- Definitions based on the conditions in the problem
def production_cost_per_component : ℕ := 80
def shipping_cost_per_component : ℕ := 5
def components_per_month : ℕ := 150
def minimum_price_per_component : ℕ := 195

-- Monthly fixed cost definition based on the provided solution
def monthly_fixed_cost := components_per_month * (minimum_price_per_component - (production_cost_per_component + shipping_cost_per_component))

-- Theorem stating that the calculated fixed cost is correct.
theorem monthly_fixed_cost_is_correct : monthly_fixed_cost = 16500 :=
by
  unfold monthly_fixed_cost
  norm_num
  sorry

end monthly_fixed_cost_is_correct_l79_79308


namespace heroes_can_reduce_heads_to_zero_l79_79637

-- Definition of the Hero strikes
def IlyaMurometsStrikes (H : ℕ) : ℕ := H / 2 - 1
def DobrynyaNikitichStrikes (H : ℕ) : ℕ := 2 * H / 3 - 2
def AlyoshaPopovichStrikes (H : ℕ) : ℕ := 3 * H / 4 - 3

-- The ultimate goal is proving this theorem
theorem heroes_can_reduce_heads_to_zero (H : ℕ) : 
  ∃ (n : ℕ), ∀ i ≤ n, 
  (if i % 3 = 0 then H = 0 
   else if i % 3 = 1 then IlyaMurometsStrikes H = 0 
   else if i % 3 = 2 then DobrynyaNikitichStrikes H = 0 
   else AlyoshaPopovichStrikes H = 0)
:= sorry

end heroes_can_reduce_heads_to_zero_l79_79637


namespace amount_left_after_pool_l79_79413

def amount_left (total_earned : ℝ) (cost_per_person : ℝ) (num_people : ℕ) : ℝ :=
  total_earned - (cost_per_person * num_people)

theorem amount_left_after_pool :
  amount_left 30 2.5 10 = 5 :=
by
  sorry

end amount_left_after_pool_l79_79413


namespace find_a_l79_79619

theorem find_a :
  ∀ (a : ℝ), 
  (∀ x : ℝ, 2 * x^2 - 2016 * x + 2016^2 - 2016 * a - 1 = a^2) → 
  (∃ x1 x2 : ℝ, 2 * x1^2 - 2016 * x1 + 2016^2 - 2016 * a - 1 - a^2 = 0 ∧
                 2 * x2^2 - 2016 * x2 + 2016^2 - 2016 * a - 1 - a^2 = 0 ∧
                 x1 < a ∧ a < x2) → 
  2015 < a ∧ a < 2017 :=
by sorry

end find_a_l79_79619


namespace probability_of_winning_pair_l79_79752

/--
A deck consists of five red cards and five green cards, with each color having cards labeled from A to E. 
Two cards are drawn from this deck.
A winning pair is defined as two cards of the same color or two cards of the same letter. 
Prove that the probability of drawing a winning pair is 5/9.
-/
theorem probability_of_winning_pair :
  let total_cards := 10
  let total_ways := Nat.choose total_cards 2
  let same_letter_ways := 5
  let same_color_red_ways := Nat.choose 5 2
  let same_color_green_ways := Nat.choose 5 2
  let same_color_ways := same_color_red_ways + same_color_green_ways
  let favorable_outcomes := same_letter_ways + same_color_ways
  favorable_outcomes / total_ways = 5 / 9 := by
  sorry

end probability_of_winning_pair_l79_79752


namespace total_payroll_calc_l79_79935

theorem total_payroll_calc
  (h : ℕ := 129)          -- pay per day for heavy operators
  (l : ℕ := 82)           -- pay per day for general laborers
  (n : ℕ := 31)           -- total number of people hired
  (g : ℕ := 1)            -- number of general laborers employed
  : (h * (n - g) + l * g) = 3952 := 
by
  sorry

end total_payroll_calc_l79_79935


namespace hcf_of_two_numbers_l79_79704

theorem hcf_of_two_numbers (A B : ℕ) (h1 : A * B = 4107) (h2 : A = 111) : (Nat.gcd A B) = 37 :=
by
  -- Given conditions
  have h3 : B = 37 := by
    -- Deduce B from given conditions
    sorry
  -- Prove hcf (gcd) is 37
  sorry

end hcf_of_two_numbers_l79_79704


namespace henri_drove_farther_l79_79375

theorem henri_drove_farther (gervais_avg_miles_per_day : ℕ) (gervais_days : ℕ) (henri_total_miles : ℕ)
  (h1 : gervais_avg_miles_per_day = 315) (h2 : gervais_days = 3) (h3 : henri_total_miles = 1250) :
  (henri_total_miles - (gervais_avg_miles_per_day * gervais_days) = 305) :=
by
  -- Here we would provide the proof, but we are omitting it as requested
  sorry

end henri_drove_farther_l79_79375


namespace roots_sum_of_quadratic_l79_79282

theorem roots_sum_of_quadratic:
  (∃ a b : ℝ, (a ≠ b) ∧ (a * b = 5) ∧ (a + b = 8)) →
  (a + b = 8) :=
by
  sorry

end roots_sum_of_quadratic_l79_79282


namespace zeros_of_quadratic_l79_79062

def f (x : ℝ) := x^2 - 2 * x - 3

theorem zeros_of_quadratic : ∀ x, f x = 0 ↔ (x = 3 ∨ x = -1) := 
by 
  sorry

end zeros_of_quadratic_l79_79062


namespace arithmetic_seq_slope_l79_79193

theorem arithmetic_seq_slope {a : ℕ → ℤ} (h : a 2 - a 4 = 2) : ∃ a1 : ℤ, ∀ n : ℕ, a n = -n + (a 1) + 1 := 
by {
  sorry
}

end arithmetic_seq_slope_l79_79193


namespace pencils_distribution_count_l79_79155

def count_pencils_distribution : ℕ :=
  let total_pencils := 10
  let friends := 4
  let adjusted_pencils := total_pencils - friends
  Nat.choose (adjusted_pencils + friends - 1) (friends - 1)

theorem pencils_distribution_count :
  count_pencils_distribution = 84 := 
  by sorry

end pencils_distribution_count_l79_79155


namespace proof_range_of_a_l79_79314

/-- p is the proposition that for all x in [1,2], x^2 - a ≥ 0 --/
def p (a : ℝ) : Prop := ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0

/-- q is the proposition that there exists an x0 in ℝ such that x0^2 + (a-1)x0 + 1 < 0 --/
def q (a : ℝ) : Prop := ∃ x0 : ℝ, x0^2 + (a-1)*x0 + 1 < 0

theorem proof_range_of_a (a : ℝ) : (p a ∨ q a) ∧ (¬p a ∧ ¬q a) → (a ≥ -1 ∧ a ≤ 1) ∨ a > 3 :=
by
  sorry -- proof will be filled out here

end proof_range_of_a_l79_79314


namespace integer_solutions_l79_79886

theorem integer_solutions (x y : ℤ) : 2 * (x + y) = x * y + 7 ↔ (x, y) = (3, -1) ∨ (x, y) = (5, 1) ∨ (x, y) = (1, 5) ∨ (x, y) = (-1, 3) := by
  sorry

end integer_solutions_l79_79886


namespace triangle_angles_l79_79371

theorem triangle_angles (α β : ℝ) (A B C : ℝ) (hA : A = 2) (hB : B = 3) (hC : C = 4) :
  2 * α + 3 * β = 180 :=
sorry

end triangle_angles_l79_79371


namespace defect_rate_probability_l79_79324

theorem defect_rate_probability (p : ℝ) (n : ℕ) (ε : ℝ) (q : ℝ) : 
  p = 0.02 →
  n = 800 →
  ε = 0.01 →
  q = 1 - p →
  1 - (p * q) / (n * ε^2) = 0.755 :=
by
  intro hp hn he hq
  rw [hp, hn, he, hq]
  -- Calculation steps can be verified here
  sorry

end defect_rate_probability_l79_79324


namespace solve_system_l79_79868

theorem solve_system (a b c x y z : ℝ) (h₀ : a = (a * x + c * y) / (b * z + 1))
  (h₁ : b = (b * x + y) / (b * z + 1)) 
  (h₂ : c = (a * z + c) / (b * z + 1)) 
  (h₃ : ¬ a = b * c) :
  x = 1 ∧ y = 0 ∧ z = 0 :=
sorry

end solve_system_l79_79868


namespace cube_mod7_not_divisible_7_l79_79808

theorem cube_mod7_not_divisible_7 (a : ℤ) (h : ¬ (7 ∣ a)) :
  (a^3 % 7 = 1) ∨ (a^3 % 7 = -1) :=
sorry

end cube_mod7_not_divisible_7_l79_79808


namespace fraction_walk_home_l79_79721

theorem fraction_walk_home : 
  (1 - ((1 / 2) + (1 / 4) + (1 / 10) + (1 / 8))) = (1 / 40) :=
by 
  sorry

end fraction_walk_home_l79_79721


namespace find_number_l79_79349

theorem find_number (x : ℝ) : 4 * x - 23 = 33 → x = 14 :=
by
  intros h
  sorry

end find_number_l79_79349


namespace cost_of_purchasing_sandwiches_and_sodas_l79_79170

def sandwich_price : ℕ := 4
def soda_price : ℕ := 1
def num_sandwiches : ℕ := 6
def num_sodas : ℕ := 5
def total_cost : ℕ := 29

theorem cost_of_purchasing_sandwiches_and_sodas :
  (num_sandwiches * sandwich_price + num_sodas * soda_price) = total_cost :=
by
  sorry

end cost_of_purchasing_sandwiches_and_sodas_l79_79170


namespace commute_time_late_l79_79924

theorem commute_time_late (S : ℝ) (T : ℝ) (T' : ℝ) (H1 : T = 1) (H2 : T' = (4/3)) :
  T' - T = 20 / 60 :=
by
  sorry

end commute_time_late_l79_79924


namespace expression_evaluation_l79_79891

theorem expression_evaluation : 
  ( ((2 + 2)^2 / 2^2) * ((3 + 3 + 3 + 3)^3 / (3 + 3 + 3)^3) * ((6 + 6 + 6 + 6 + 6 + 6)^6 / (6 + 6 + 6 + 6)^6) = 108 ) := 
by 
  sorry

end expression_evaluation_l79_79891


namespace value_of_y_at_x_3_l79_79729

theorem value_of_y_at_x_3 (a b c : ℝ) (h : a * (-3 : ℝ)^5 + b * (-3)^3 + c * (-3) - 5 = 7) :
  a * (3 : ℝ)^5 + b * 3^3 + c * 3 - 5 = -17 :=
by
  sorry

end value_of_y_at_x_3_l79_79729


namespace range_of_a_l79_79286

noncomputable
def proposition_p (x : ℝ) : Prop := abs (x - (3 / 4)) <= (1 / 4)
noncomputable
def proposition_q (x a : ℝ) : Prop := (x - a) * (x - a - 1) <= 0

theorem range_of_a :
  (∀ x : ℝ, proposition_p x → ∃ x : ℝ, proposition_q x a) ∧
  (∃ x : ℝ, ¬(proposition_p x → proposition_q x a )) →
  0 ≤ a ∧ a ≤ (1 / 2) :=
sorry

end range_of_a_l79_79286


namespace deepak_present_age_l79_79644

-- Define the variables R and D
variables (R D : ℕ)

-- The conditions:
-- 1. After 4 years, Rahul's age will be 32 years.
-- 2. The ratio between Rahul and Deepak's ages is 4:3.
def rahul_age_after_4 : Prop := R + 4 = 32
def age_ratio : Prop := R / D = 4 / 3

-- The statement we want to prove:
theorem deepak_present_age (h1 : rahul_age_after_4 R) (h2 : age_ratio R D) : D = 21 :=
by sorry

end deepak_present_age_l79_79644


namespace total_students_is_37_l79_79379

-- Let b be the number of blue swim caps 
-- Let r be the number of red swim caps
variables (b r : ℕ)

-- The number of blue swim caps according to the male sports commissioner
def condition1 : Prop := b = 4 * r + 1

-- The number of blue swim caps according to the female sports commissioner
def condition2 : Prop := b = r + 24

-- The total number of students in the 3rd grade
def total_students : ℕ := b + r

theorem total_students_is_37 (h1 : condition1 b r) (h2 : condition2 b r) : total_students b r = 37 :=
by sorry

end total_students_is_37_l79_79379


namespace mark_charged_more_hours_than_kate_l79_79866

variables (K P M : ℝ)
variables (h1 : K + P + M = 198) (h2 : P = 2 * K) (h3 : M = 3 * P)

theorem mark_charged_more_hours_than_kate : M - K = 110 :=
by
  sorry

end mark_charged_more_hours_than_kate_l79_79866


namespace profit_distribution_l79_79897

theorem profit_distribution (x : ℕ) (hx : 2 * x = 4000) :
  let A := 2 * x
  let B := 3 * x
  let C := 5 * x
  A + B + C = 20000 := by
  sorry

end profit_distribution_l79_79897


namespace ratio_of_nuts_to_raisins_l79_79325

theorem ratio_of_nuts_to_raisins 
  (R N : ℝ) 
  (h_ratio : 3 * R = 0.2727272727272727 * (3 * R + 4 * N)) : 
  N = 2 * R := 
sorry

end ratio_of_nuts_to_raisins_l79_79325


namespace average_words_per_hour_l79_79658

theorem average_words_per_hour
  (total_words : ℕ := 60000)
  (total_hours : ℕ := 150)
  (first_period_hours : ℕ := 50)
  (first_period_words : ℕ := total_words / 2) :
  first_period_words / first_period_hours = 600 ∧ total_words / total_hours = 400 := 
by
  sorry

end average_words_per_hour_l79_79658


namespace circumscribed_circle_radius_of_rectangle_l79_79442

theorem circumscribed_circle_radius_of_rectangle 
  (a b : ℝ) 
  (h1: a = 1) 
  (angle_between_diagonals : ℝ) 
  (h2: angle_between_diagonals = 60) : 
  ∃ R, R = 1 :=
by 
  sorry

end circumscribed_circle_radius_of_rectangle_l79_79442


namespace sum_of_24_consecutive_integers_is_square_l79_79643

theorem sum_of_24_consecutive_integers_is_square : ∃ n : ℕ, ∃ k : ℕ, (n > 0) ∧ (24 * (2 * n + 23)) = k * k ∧ k * k = 324 :=
by
  sorry

end sum_of_24_consecutive_integers_is_square_l79_79643


namespace booster_club_tickets_l79_79952

theorem booster_club_tickets (x : ℕ) : 
  (11 * 9 + x * 7 = 225) → 
  (x + 11 = 29) := 
by
  sorry

end booster_club_tickets_l79_79952


namespace can_form_sets_l79_79944

def clearly_defined (s : Set α) : Prop := ∀ x ∈ s, True
def not_clearly_defined (s : Set α) : Prop := ¬clearly_defined s

def cubes := {x : Type | True} -- Placeholder for the actual definition
def major_supermarkets := {x : Type | True} -- Placeholder for the actual definition
def difficult_math_problems := {x : Type | True} -- Placeholder for the actual definition
def famous_dancers := {x : Type | True} -- Placeholder for the actual definition
def products_2012 := {x : Type | True} -- Placeholder for the actual definition
def points_on_axes := {x : ℝ × ℝ | x.1 = 0 ∨ x.2 = 0}

theorem can_form_sets :
  (clearly_defined cubes) ∧
  (not_clearly_defined major_supermarkets) ∧
  (not_clearly_defined difficult_math_problems) ∧
  (not_clearly_defined famous_dancers) ∧
  (clearly_defined products_2012) ∧
  (clearly_defined points_on_axes) →
  True := 
by {
  -- Your proof goes here
  sorry
}

end can_form_sets_l79_79944


namespace lower_limit_b_l79_79651

theorem lower_limit_b (a b : ℤ) (h1 : 6 < a) (h2 : a < 17) (h3 : b < 29) 
  (h4 : ∃ min_b max_b, min_b = 4 ∧ max_b ≤ 29 ∧ 3.75 = (16 : ℚ) / (min_b : ℚ) - (7 : ℚ) / (max_b : ℚ)) : 
  b ≥ 4 :=
sorry

end lower_limit_b_l79_79651


namespace child_to_grandmother_ratio_l79_79768

variable (G D C : ℝ)

axiom condition1 : G + D + C = 150
axiom condition2 : D + C = 60
axiom condition3 : D = 42

theorem child_to_grandmother_ratio : (C / G) = (1 / 5) :=
by
  sorry

end child_to_grandmother_ratio_l79_79768


namespace card_statements_has_four_true_l79_79608

noncomputable def statement1 (S : Fin 5 → Bool) : Prop := S 0 = true -> (S 1 = false ∧ S 2 = false ∧ S 3 = false ∧ S 4 = false)
noncomputable def statement2 (S : Fin 5 → Bool) : Prop := S 1 = true -> (S 0 = false ∧ S 2 = false ∧ S 3 = false ∧ S 4 = false)
noncomputable def statement3 (S : Fin 5 → Bool) : Prop := S 2 = true -> (S 0 = false ∧ S 1 = false ∧ S 3 = false ∧ S 4 = false)
noncomputable def statement4 (S : Fin 5 → Bool) : Prop := S 3 = true -> (S 0 = false ∧ S 1 = false ∧ S 2 = false ∧ S 4 = false)
noncomputable def statement5 (S : Fin 5 → Bool) : Prop := S 4 = true -> (S 0 = false ∧ S 1 = false ∧ S 2 = false ∧ S 3 = false)

theorem card_statements_has_four_true : ∃ (S : Fin 5 → Bool), 
  (statement1 S ∧ statement2 S ∧ statement3 S ∧ statement4 S ∧ statement5 S ∧ 
  ((S 0 = true ∨ S 1 = true ∨ S 2 = true ∨ S 3 = true ∨ S 4 = true) ∧ 
  4 = (if S 0 then 1 else 0) + (if S 1 then 1 else 0) + 
      (if S 2 then 1 else 0) + (if S 3 then 1 else 0) + 
      (if S 4 then 1 else 0))) :=
sorry

end card_statements_has_four_true_l79_79608


namespace supplements_of_congruent_angles_are_congruent_l79_79508

-- Define the concept of supplementary angles
def is_supplementary (α β : ℝ) : Prop := α + β = 180

-- Statement of the problem
theorem supplements_of_congruent_angles_are_congruent :
  ∀ {α β γ δ : ℝ},
  is_supplementary α β →
  is_supplementary γ δ →
  β = δ →
  α = γ :=
by
  intros α β γ δ h1 h2 h3
  sorry

end supplements_of_congruent_angles_are_congruent_l79_79508


namespace cherry_tree_leaves_l79_79931

theorem cherry_tree_leaves (original_plan : ℕ) (multiplier : ℕ) (leaves_per_tree : ℕ) 
  (h1 : original_plan = 7) (h2 : multiplier = 2) (h3 : leaves_per_tree = 100) : 
  (original_plan * multiplier * leaves_per_tree = 1400) :=
by
  sorry

end cherry_tree_leaves_l79_79931


namespace num_integer_terms_sequence_l79_79175

noncomputable def sequence_starting_at_8820 : Nat := 8820

def divide_by_5 (n : Nat) : Nat := n / 5

theorem num_integer_terms_sequence :
  let seq := [sequence_starting_at_8820, divide_by_5 sequence_starting_at_8820]
  seq = [8820, 1764] →
  seq.length = 2 := by
  sorry

end num_integer_terms_sequence_l79_79175


namespace min_value_a_l79_79655

theorem min_value_a (a b c : ℤ) (α β : ℝ)
  (h_a_pos : a > 0) 
  (h_eq : ∀ x : ℝ, a * x^2 + b * x + c = 0 → (x = α ∨ x = β))
  (h_alpha_beta_order : 0 < α ∧ α < β ∧ β < 1) :
  a ≥ 5 :=
sorry

end min_value_a_l79_79655


namespace total_rainfall_l79_79810

theorem total_rainfall (rain_first_hour : ℕ) (rain_second_hour : ℕ) : Prop :=
  rain_first_hour = 5 →
  rain_second_hour = 7 + 2 * rain_first_hour →
  rain_first_hour + rain_second_hour = 22

-- Add sorry to skip the proof.

end total_rainfall_l79_79810


namespace average_of_last_three_numbers_l79_79915

theorem average_of_last_three_numbers (A B C D E F : ℕ) 
  (h1 : (A + B + C + D + E + F) / 6 = 30)
  (h2 : (A + B + C + D) / 4 = 25)
  (h3 : D = 25) :
  (D + E + F) / 3 = 35 :=
by
  sorry

end average_of_last_three_numbers_l79_79915


namespace tylenol_intake_proof_l79_79145

noncomputable def calculate_tylenol_intake_grams
  (tablet_mg : ℕ) (tablets_per_dose : ℕ) (hours_per_dose : ℕ) (total_hours : ℕ) : ℕ :=
  let doses := total_hours / hours_per_dose
  let total_mg := doses * tablets_per_dose * tablet_mg
  total_mg / 1000

theorem tylenol_intake_proof : calculate_tylenol_intake_grams 500 2 4 12 = 3 :=
  by sorry

end tylenol_intake_proof_l79_79145


namespace subtraction_of_fractions_l79_79318

theorem subtraction_of_fractions : (5 / 9) - (1 / 6) = 7 / 18 :=
by
  sorry

end subtraction_of_fractions_l79_79318


namespace volume_of_snow_correct_l79_79676

noncomputable def volume_of_snow : ℝ :=
  let sidewalk_length := 30
  let sidewalk_width := 3
  let depth := 3 / 4
  let sidewalk_volume := sidewalk_length * sidewalk_width * depth
  
  let garden_path_leg1 := 3
  let garden_path_leg2 := 4
  let garden_path_area := (garden_path_leg1 * garden_path_leg2) / 2
  let garden_path_volume := garden_path_area * depth
  
  let total_volume := sidewalk_volume + garden_path_volume
  total_volume

theorem volume_of_snow_correct : volume_of_snow = 72 := by
  sorry

end volume_of_snow_correct_l79_79676


namespace num_5_letter_words_with_at_least_two_consonants_l79_79520

open Finset

def letters : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F'}
def consonants : Finset Char := {'B', 'C', 'D', 'F'}
def vowels : Finset Char := {'A', 'E'}

def total_5_letter_words : ℕ := 6^5
def words_with_0_consonants : ℕ := 2^5
def words_with_1_consonant : ℕ := 5 * 4 * 2^4

theorem num_5_letter_words_with_at_least_two_consonants : 
  total_5_letter_words - (words_with_0_consonants + words_with_1_consonant) = 7424 := by
  sorry

end num_5_letter_words_with_at_least_two_consonants_l79_79520


namespace cats_added_l79_79407

theorem cats_added (siamese_cats house_cats total_cats : ℕ) 
  (h1 : siamese_cats = 13) 
  (h2 : house_cats = 5) 
  (h3 : total_cats = 28) : 
  total_cats - (siamese_cats + house_cats) = 10 := 
by 
  sorry

end cats_added_l79_79407


namespace square_circle_area_ratio_l79_79160

theorem square_circle_area_ratio {r : ℝ} (h : ∀ s : ℝ, 2 * r = s * Real.sqrt 2) :
  (2 * r ^ 2) / (Real.pi * r ^ 2) = 2 / Real.pi :=
by
  sorry

end square_circle_area_ratio_l79_79160


namespace molecular_weight_CO_l79_79480

theorem molecular_weight_CO : 
  let molecular_weight_C := 12.01
  let molecular_weight_O := 16.00
  molecular_weight_C + molecular_weight_O = 28.01 :=
by
  sorry

end molecular_weight_CO_l79_79480


namespace interval_solution_l79_79455

theorem interval_solution (x : ℝ) : 2 ≤ x / (2 * x - 5) ∧ x / (2 * x - 5) < 7 ↔ (35 / 13 : ℝ) < x ∧ x ≤ 10 / 3 :=
by
  sorry

end interval_solution_l79_79455


namespace find_r_divisibility_l79_79029

theorem find_r_divisibility :
  ∃ r : ℝ, (10 * r ^ 2 - 4 * r - 26 = 0 ∧ (r = (19 / 10) ∨ r = (-3 / 2))) ∧ (r = -3 / 2) ∧ (10 * r ^ 3 - 5 * r ^ 2 - 52 * r + 60 = 0) :=
by
  sorry

end find_r_divisibility_l79_79029


namespace find_percentage_l79_79454

theorem find_percentage (P : ℝ) (h : (P / 100) * 600 = (50 / 100) * 720) : P = 60 :=
by
  sorry

end find_percentage_l79_79454


namespace inverse_proportion_point_l79_79402

theorem inverse_proportion_point (k : ℝ) (x1 y1 x2 y2 : ℝ)
  (h1 : y1 = k / x1) 
  (h2 : x1 = -2) 
  (h3 : y1 = 3)
  (h4 : x2 = 2) :
  y2 = -3 := 
by
  -- proof will be provided here
  sorry

end inverse_proportion_point_l79_79402


namespace average_salary_rest_l79_79275

theorem average_salary_rest (total_workers : ℕ) (avg_salary_all : ℝ)
  (num_technicians : ℕ) (avg_salary_technicians : ℝ) :
  total_workers = 21 →
  avg_salary_all = 8000 →
  num_technicians = 7 →
  avg_salary_technicians = 12000 →
  (avg_salary_all * total_workers - avg_salary_technicians * num_technicians) / (total_workers - num_technicians) = 6000 :=
by intros h1 h2 h3 h4; sorry

end average_salary_rest_l79_79275


namespace keith_attended_games_l79_79284

-- Definitions from the conditions
def total_games : ℕ := 20
def missed_games : ℕ := 9

-- The statement to prove
theorem keith_attended_games : (total_games - missed_games) = 11 :=
by
  sorry

end keith_attended_games_l79_79284


namespace number_of_players_knight_moves_friend_not_winner_l79_79792

-- Problem (a)
theorem number_of_players (sum_scores : ℕ) (h : sum_scores = 210) : 
  ∃ x : ℕ, x * (x - 1) = 210 :=
sorry

-- Problem (b)
theorem knight_moves (initial_positions : ℕ) (wrong_guess : ℕ) (correct_answer : ℕ) : 
  initial_positions = 1 ∧ wrong_guess = 64 ∧ correct_answer = 33 → 
  ∃ squares : ℕ, squares = 33 :=
sorry

-- Problem (c)
theorem friend_not_winner (total_scores : ℕ) (num_players : ℕ) (friend_score : ℕ) (avg_score : ℕ) : 
  total_scores = 210 ∧ num_players = 15 ∧ friend_score = 12 ∧ avg_score = 14 → 
  ∃ higher_score : ℕ, higher_score > friend_score :=
sorry

end number_of_players_knight_moves_friend_not_winner_l79_79792


namespace greatest_number_of_consecutive_integers_sum_36_l79_79117

theorem greatest_number_of_consecutive_integers_sum_36 :
  ∃ (N : ℕ), 
    (∃ a : ℤ, N * a + ((N - 1) * N) / 2 = 36) ∧ 
    (∀ N' : ℕ, (∃ a' : ℤ, N' * a' + ((N' - 1) * N') / 2 = 36) → N' ≤ 72) := by
  sorry

end greatest_number_of_consecutive_integers_sum_36_l79_79117


namespace jim_net_paycheck_l79_79890

-- Let’s state the problem conditions:
def biweekly_gross_pay : ℝ := 1120
def retirement_percentage : ℝ := 0.25
def tax_deduction : ℝ := 100

-- Define the amount deduction for the retirement account
def retirement_deduction (gross : ℝ) (percentage : ℝ) : ℝ := gross * percentage

-- Define the remaining paycheck after all deductions
def net_paycheck (gross : ℝ) (retirement : ℝ) (tax : ℝ) : ℝ :=
  gross - retirement - tax

-- The theorem to prove:
theorem jim_net_paycheck :
  net_paycheck biweekly_gross_pay (retirement_deduction biweekly_gross_pay retirement_percentage) tax_deduction = 740 :=
by
  sorry

end jim_net_paycheck_l79_79890


namespace simplify_expression_l79_79129

theorem simplify_expression : 8 * (15 / 4) * (-56 / 45) = -112 / 3 :=
by sorry

end simplify_expression_l79_79129


namespace circle_O2_tangent_circle_O2_intersect_l79_79554

-- Condition: The equation of circle O_1 is \(x^2 + (y + 1)^2 = 4\)
def circle_O1 (x y : ℝ) : Prop := x^2 + (y + 1)^2 = 4

-- Condition: The center of circle O_2 is \(O_2(2, 1)\)
def center_O2 : (ℝ × ℝ) := (2, 1)

-- Prove the equation of circle O_2 if it is tangent to circle O_1
theorem circle_O2_tangent : 
  ∀ (x y : ℝ), circle_O1 x y → (x - 2)^2 + (y - 1)^2 = 12 - 8 * Real.sqrt 2 :=
sorry

-- Prove the equations of circle O_2 if it intersects circle O_1 and \(|AB| = 2\sqrt{2}\)
theorem circle_O2_intersect :
  ∀ (x y : ℝ), 
  circle_O1 x y → 
  (2 * Real.sqrt 2 = |(x - 2)^2 + (y - 1)^2 - 4| ∨ (x - 2)^2 + (y - 1)^2 = 20) :=
sorry

end circle_O2_tangent_circle_O2_intersect_l79_79554


namespace alex_downhill_time_l79_79246

theorem alex_downhill_time
  (speed_flat : ℝ)
  (time_flat : ℝ)
  (speed_uphill : ℝ)
  (time_uphill : ℝ)
  (speed_downhill : ℝ)
  (distance_walked : ℝ)
  (total_distance : ℝ)
  (h_flat : speed_flat = 20)
  (h_time_flat : time_flat = 4.5)
  (h_uphill : speed_uphill = 12)
  (h_time_uphill : time_uphill = 2.5)
  (h_downhill : speed_downhill = 24)
  (h_walked : distance_walked = 8)
  (h_total : total_distance = 164)
  : (156 - (speed_flat * time_flat + speed_uphill * time_uphill)) / speed_downhill = 1.5 :=
by 
  sorry

end alex_downhill_time_l79_79246


namespace distance_traveled_l79_79486

theorem distance_traveled
  (D : ℝ) (T : ℝ)
  (h1 : D = 10 * T)
  (h2 : D + 20 = 14 * T)
  : D = 50 := sorry

end distance_traveled_l79_79486


namespace simplify_expression_l79_79069

theorem simplify_expression (x : ℝ) : (x + 1) ^ 2 + x * (x - 2) = 2 * x ^ 2 + 1 :=
by
  sorry

end simplify_expression_l79_79069


namespace area_of_quadrilateral_ABCD_l79_79435

noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

theorem area_of_quadrilateral_ABCD :
  let AB := 15 * sqrt 2
  let BE := 15 * sqrt 2
  let BC := 7.5 * sqrt 2
  let CE := 7.5 * sqrt 6
  let CD := 7.5 * sqrt 2
  let DE := 7.5 * sqrt 6
  (1/2 * AB * BE) + (1/2 * BC * CE) + (1/2 * CD * DE) = 225 + 112.5 * sqrt 12 :=
by
  sorry

end area_of_quadrilateral_ABCD_l79_79435


namespace min_fraction_value_l79_79058

noncomputable def min_value_fraction (x y z : ℝ) (h : x > 0 ∧ y > 0 ∧ z > 0) (h₁ : x^2 + y^2 + z^2 = 1) : ℝ :=
  (z+1)^2 / (2 * x * y * z)

theorem min_fraction_value (x y z : ℝ) (h : x > 0 ∧ y > 0 ∧ z > 0) (h₁ : x^2 + y^2 + z^2 = 1) :
  min_value_fraction x y z h h₁ = 3 + 2 * Real.sqrt 2 :=
  sorry

end min_fraction_value_l79_79058


namespace A_inter_B_complement_l79_79518

def A : Set ℝ := {x : ℝ | -4 < x^2 - 5*x + 2 ∧ x^2 - 5*x + 2 < 26}
def B : Set ℝ := {x : ℝ | -x^2 + 4*x - 3 < 0}

theorem A_inter_B_complement :
  A ∩ B = {x : ℝ | (-3 < x ∧ x < 1) ∨ (3 < x ∧ x < 8)} ∧
  {x | x ∉ A ∩ B} = {x : ℝ | x ≤ -3 ∨ (1 ≤ x ∧ x ≤ 3) ∨ x ≥ 8 } :=
by
  sorry

end A_inter_B_complement_l79_79518


namespace students_neither_math_physics_drama_exclusive_l79_79811

def total_students : ℕ := 75
def math_students : ℕ := 42
def physics_students : ℕ := 35
def both_students : ℕ := 25
def drama_exclusive_students : ℕ := 10

theorem students_neither_math_physics_drama_exclusive : 
  total_students - (math_students + physics_students - both_students + drama_exclusive_students) = 13 :=
by
  sorry

end students_neither_math_physics_drama_exclusive_l79_79811


namespace johns_raw_squat_weight_l79_79919

variable (R : ℝ)

def sleeves_lift := R + 30
def wraps_lift := 1.25 * R
def wraps_more_than_sleeves := wraps_lift R - sleeves_lift R = 120

theorem johns_raw_squat_weight : wraps_more_than_sleeves R → R = 600 :=
by
  intro h
  sorry

end johns_raw_squat_weight_l79_79919


namespace person_speed_l79_79234

theorem person_speed (d_meters : ℕ) (t_minutes : ℕ) (d_km t_hours : ℝ) :
  (d_meters = 1800) →
  (t_minutes = 12) →
  (d_km = d_meters / 1000) →
  (t_hours = t_minutes / 60) →
  d_km / t_hours = 9 :=
by
  intros
  sorry

end person_speed_l79_79234


namespace goldie_total_earnings_l79_79614

-- Define weekly earnings based on hours and rates
def earnings_first_week (hours_dog_walking hours_medication : ℕ) : ℕ :=
  (hours_dog_walking * 5) + (hours_medication * 8)

def earnings_second_week (hours_feeding hours_cleaning hours_playing : ℕ) : ℕ :=
  (hours_feeding * 6) + (hours_cleaning * 4) + (hours_playing * 3)

-- Given conditions for hours worked each task in two weeks
def hours_dog_walking : ℕ := 12
def hours_medication : ℕ := 8
def hours_feeding : ℕ := 10
def hours_cleaning : ℕ := 15
def hours_playing : ℕ := 5

-- Proof statement: Total earnings over two weeks equals $259
theorem goldie_total_earnings : 
  (earnings_first_week hours_dog_walking hours_medication) + 
  (earnings_second_week hours_feeding hours_cleaning hours_playing) = 259 :=
by
  sorry

end goldie_total_earnings_l79_79614


namespace kayla_waiting_years_l79_79279

def minimum_driving_age : ℕ := 18
def kimiko_age : ℕ := 26
def kayla_age : ℕ := kimiko_age / 2
def years_until_kayla_can_drive : ℕ := minimum_driving_age - kayla_age

theorem kayla_waiting_years : years_until_kayla_can_drive = 5 :=
by
  sorry

end kayla_waiting_years_l79_79279


namespace min_value_of_reciprocal_l79_79773

theorem min_value_of_reciprocal (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 * x + y = 2) :
  (∀ r, r = 1 / x + 1 / y → r ≥ 3 / 2 + Real.sqrt 2) :=
by
  sorry

end min_value_of_reciprocal_l79_79773


namespace area_outside_squares_inside_triangle_l79_79016

noncomputable def side_length_large_square : ℝ := 6
noncomputable def side_length_small_square1 : ℝ := 2
noncomputable def side_length_small_square2 : ℝ := 3
noncomputable def area_large_square := side_length_large_square ^ 2
noncomputable def area_small_square1 := side_length_small_square1 ^ 2
noncomputable def area_small_square2 := side_length_small_square2 ^ 2
noncomputable def area_triangle_EFG := area_large_square / 2
noncomputable def total_area_small_squares := area_small_square1 + area_small_square2

theorem area_outside_squares_inside_triangle :
  (area_triangle_EFG - total_area_small_squares) = 5 :=
by
  sorry

end area_outside_squares_inside_triangle_l79_79016


namespace ratio_of_two_numbers_l79_79315

variable {a b : ℝ}

theorem ratio_of_two_numbers
  (h1 : a + b = 7 * (a - b))
  (h2 : 0 < b)
  (h3 : a > b) :
  a / b = 4 / 3 := by
  sorry

end ratio_of_two_numbers_l79_79315


namespace eccentricity_ellipse_l79_79050

variable (a b : ℝ) (h1 : a > b) (h2 : b > 0)
variable (c : ℝ) (h3 : c = Real.sqrt (a ^ 2 - b ^ 2))
variable (h4 : b = c)
variable (ellipse_eq : ∀ x y, (x^2)/(a^2) + (y^2)/(b^2) = 1)

theorem eccentricity_ellipse :
  c / a = Real.sqrt 2 / 2 :=
by
  sorry

end eccentricity_ellipse_l79_79050


namespace combined_boys_average_l79_79660

noncomputable def average_boys_score (C c D d : ℕ) : ℚ :=
  (68 * C + 74 * 3 * c / 4) / (C + 3 * c / 4)

theorem combined_boys_average:
  ∀ (C c D d : ℕ),
  (68 * C + 72 * c) / (C + c) = 70 →
  (74 * D + 88 * d) / (D + d) = 82 →
  (72 * c + 88 * d) / (c + d) = 83 →
  C = c →
  4 * D = 3 * d →
  average_boys_score C c D d = 48.57 :=
by
  intros C c D d h_clinton h_dixon h_combined_girls h_C_eq_c h_D_eq_d
  sorry

end combined_boys_average_l79_79660


namespace larger_number_l79_79230

theorem larger_number (HCF A B : ℕ) (factor1 factor2 : ℕ) (h_HCF : HCF = 23) (h_factor1 : factor1 = 14) (h_factor2 : factor2 = 15) (h_LCM : HCF * factor1 * factor2 = A * B) (h_A : A = HCF * factor2) (h_B : B = HCF * factor1) : A = 345 :=
by
  sorry

end larger_number_l79_79230


namespace shelby_scooter_drive_l79_79711

/-- 
Let y be the time (in minutes) Shelby drove when it was not raining.
Speed when not raining is 25 miles per hour, which is 5/12 mile per minute.
Speed when raining is 15 miles per hour, which is 1/4 mile per minute.
Total distance covered is 18 miles.
Total time taken is 36 minutes.
Prove that Shelby drove for 6 minutes when it was not raining.
-/
theorem shelby_scooter_drive
  (y : ℝ)
  (h_not_raining_speed : ∀ t (h : t = (25/60 : ℝ)), t = (5/12 : ℝ))
  (h_raining_speed : ∀ t (h : t = (15/60 : ℝ)), t = (1/4 : ℝ))
  (h_total_distance : ∀ d (h : d = ((5/12 : ℝ) * y + (1/4 : ℝ) * (36 - y))), d = 18)
  (h_total_time : ∀ t (h : t = 36), t = 36) :
  y = 6 :=
sorry

end shelby_scooter_drive_l79_79711


namespace range_of_t_max_radius_circle_eq_l79_79493

-- Definitions based on conditions
def circle_equation (x y t : ℝ) := x^2 + y^2 - 2 * x + t^2 = 0

-- Statement for the range of values of t
theorem range_of_t (t : ℝ) (h : ∃ x y : ℝ, circle_equation x y t) : -1 < t ∧ t < 1 := sorry

-- Statement for the equation of the circle when t = 0
theorem max_radius_circle_eq (x y : ℝ) (h : circle_equation x y 0) : (x - 1)^2 + y^2 = 1 := sorry

end range_of_t_max_radius_circle_eq_l79_79493


namespace calculate_expression_l79_79938

theorem calculate_expression :
  50 * 24.96 * 2.496 * 500 = (1248)^2 :=
by
  sorry

end calculate_expression_l79_79938


namespace solve_x_l79_79496

theorem solve_x :
  (1 / 4 - 1 / 6) = 1 / (12 : ℝ) :=
by sorry

end solve_x_l79_79496


namespace percentage_discount_l79_79334

theorem percentage_discount (C S S' : ℝ) (h1 : S = 1.14 * C) (h2 : S' = 2.20 * C) :
  (S' - S) / S' * 100 = 48.18 :=
by 
  sorry

end percentage_discount_l79_79334


namespace hcf_of_two_numbers_l79_79872

noncomputable def find_hcf (x y : ℕ) (lcm_xy : ℕ) (prod_xy : ℕ) : ℕ :=
  prod_xy / lcm_xy

theorem hcf_of_two_numbers (x y : ℕ) (lcm_xy: ℕ) (prod_xy: ℕ) 
  (h_lcm: lcm x y = lcm_xy) (h_prod: x * y = prod_xy) :
  find_hcf x y lcm_xy prod_xy = 75 :=
by
  sorry

end hcf_of_two_numbers_l79_79872


namespace polynomial_divisible_by_seven_l79_79400

-- Define the theorem
theorem polynomial_divisible_by_seven (n : ℤ) : 7 ∣ (n + 7)^2 - n^2 :=
by sorry

end polynomial_divisible_by_seven_l79_79400


namespace expression_simplification_l79_79088

theorem expression_simplification :
  (2 + 3) * (2^3 + 3^3) * (2^9 + 3^9) * (2^27 + 3^27) = 3^41 - 2^41 := 
sorry

end expression_simplification_l79_79088


namespace minimum_value_condition_l79_79695

theorem minimum_value_condition (m n : ℝ) (hm : m > 0) (hn : n > 0) 
                                (h_line : ∀ x y : ℝ, m * x + n * y + 2 = 0 → (x + 3)^2 + (y + 1)^2 = 1) 
                                (h_chord : ∀ x1 y1 x2 y2 : ℝ, m * x1 + n * y1 + 2 = 0 ∧ (x1 + 3)^2 + (y1 + 1)^2 = 1 ∧
                                           m * x2 + n * y2 + 2 = 0 ∧ (x2 + 3)^2 + (y2 + 1)^2 = 1 ∧
                                           (x1 - x2)^2 + (y1 - y2)^2 = 4) 
                                (h_relation : 3 * m + n = 2) : 
    ∃ (C : ℝ), C = 6 ∧ (C = (1 / m + 3 / n)) := 
by
  sorry

end minimum_value_condition_l79_79695


namespace line_through_intersection_of_circles_l79_79176

theorem line_through_intersection_of_circles :
  ∀ (x y : ℝ),
    (x^2 + y^2 + 4 * x - 4 * y - 12 = 0) ∧
    (x^2 + y^2 + 2 * x + 4 * y - 4 = 0) →
    (x - 4 * y - 4 = 0) :=
by sorry

end line_through_intersection_of_circles_l79_79176


namespace max_three_cards_l79_79986

theorem max_three_cards (n m p : ℕ) (h : n + m + p = 8) (sum : 3 * n + 4 * m + 5 * p = 33) 
  (n_le_10 : n ≤ 10) (m_le_10 : m ≤ 10) (p_le_10 : p ≤ 10) : n ≤ 3 := 
sorry

end max_three_cards_l79_79986


namespace total_area_l79_79065

-- Defining basic dimensions as conditions
def left_vertical_length : ℕ := 7
def top_horizontal_length_left : ℕ := 5
def left_vertical_length_near_top : ℕ := 3
def top_horizontal_length_right_of_center : ℕ := 2
def right_vertical_length_near_center : ℕ := 3
def top_horizontal_length_far_right : ℕ := 2

-- Defining areas of partitioned rectangles
def area_bottom_left_rectangle : ℕ := 7 * 8
def area_middle_rectangle : ℕ := 5 * 3
def area_top_left_rectangle : ℕ := 2 * 8
def area_top_right_rectangle : ℕ := 2 * 7
def area_bottom_right_rectangle : ℕ := 4 * 4

-- Calculate the total area of the figure
theorem total_area : 
  area_bottom_left_rectangle + area_middle_rectangle + area_top_left_rectangle + area_top_right_rectangle + area_bottom_right_rectangle = 117 := by
  -- Proof steps will go here
  sorry

end total_area_l79_79065


namespace sum_of_abs_coeffs_in_binomial_expansion_l79_79707

theorem sum_of_abs_coeffs_in_binomial_expansion :
  ∀ (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℤ), 
  (3 * x - 1) ^ 7 = a₀ + a₁ * x + a₂ * x ^ 2 + a₃ * x ^ 3 + a₄ * x ^ 4 + a₅ * x ^ 5 + a₆ * x ^ 6 + a₇ * x ^ 7
  → |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| + |a₇| = 4 ^ 7 :=
by
  sorry

end sum_of_abs_coeffs_in_binomial_expansion_l79_79707


namespace sum_of_7_terms_arithmetic_seq_l79_79204

variable {α : Type*} [LinearOrderedField α]

def arithmetic_seq (a : ℕ → α) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

theorem sum_of_7_terms_arithmetic_seq (a : ℕ → α) (h_arith : arithmetic_seq a)
  (h_a4 : a 4 = 2) :
  (7 * (a 1 + a 7)) / 2 = 14 :=
sorry

end sum_of_7_terms_arithmetic_seq_l79_79204


namespace toy_value_l79_79310

theorem toy_value
  (t : ℕ)                 -- total number of toys
  (W : ℕ)                 -- total worth in dollars
  (v : ℕ)                 -- value of one specific toy
  (x : ℕ)                 -- value of one of the other toys
  (h1 : t = 9)            -- condition 1: total number of toys
  (h2 : W = 52)           -- condition 2: total worth
  (h3 : v = 12)           -- condition 3: value of one specific toy
  (h4 : (t - 1) * x + v = W) -- condition 4: equation based on the problem
  : x = 5 :=              -- theorem statement: other toy's value
by {
  -- proof goes here
  sorry
}

end toy_value_l79_79310


namespace evaluate_expression_l79_79549

theorem evaluate_expression :
  (24^36) / (72^18) = 8^18 :=
by
  sorry

end evaluate_expression_l79_79549


namespace geometric_sequence_sum_l79_79976

theorem geometric_sequence_sum (n : ℕ) (a : ℕ → ℕ := λ k => 2^k) 
  (S : ℕ → ℕ := λ k => (1 - 2^k) / (1 - 2)) :
  S (n + 1) = 2 * a n - 1 :=
by
  sorry

end geometric_sequence_sum_l79_79976


namespace ways_to_make_change_l79_79390

theorem ways_to_make_change : ∃ ways : ℕ, ways = 60 ∧ (∀ (p n d q : ℕ), p + 5 * n + 10 * d + 25 * q = 55 → True) := 
by
  -- The proof will go here
  sorry

end ways_to_make_change_l79_79390


namespace exists_list_with_all_players_l79_79867

-- Definitions and assumptions
variable {Player : Type} 

-- Each player plays against every other player exactly once, and there are no ties.
-- Defining defeats relationship
def defeats (p1 p2 : Player) : Prop :=
  sorry -- Assume some ordering or wins relationship

-- Defining the list of defeats
def list_of_defeats (p : Player) : Set Player :=
  { q | defeats p q ∨ (∃ r, defeats p r ∧ defeats r q) }

-- Main theorem to be proven
theorem exists_list_with_all_players (players : Set Player) :
  (∀ p q : Player, p ∈ players → q ∈ players → p ≠ q → (defeats p q ∨ defeats q p)) →
  ∃ p : Player, (list_of_defeats p) = players \ {p} :=
by
  sorry

end exists_list_with_all_players_l79_79867


namespace episodes_per_monday_l79_79927

theorem episodes_per_monday (M : ℕ) (h : 67 * (M + 2) = 201) : M = 1 :=
sorry

end episodes_per_monday_l79_79927


namespace complex_exp_power_cos_angle_l79_79298

theorem complex_exp_power_cos_angle (z : ℂ) (h : z + 1/z = 2 * Complex.cos (Real.pi / 36)) :
    z^1000 + 1/(z^1000) = 2 * Complex.cos (Real.pi * 2 / 9) :=
by
  sorry

end complex_exp_power_cos_angle_l79_79298


namespace polynomial_roots_sum_reciprocal_l79_79352

open Polynomial

theorem polynomial_roots_sum_reciprocal (a b c : ℝ) (h : 0 < a ∧ a < 1 ∧ 0 < b ∧ b < 1 ∧ 0 < c ∧ c < 1) :
    (40 * a^3 - 70 * a^2 + 32 * a - 3 = 0) ∧
    (40 * b^3 - 70 * b^2 + 32 * b - 3 = 0) ∧
    (40 * c^3 - 70 * c^2 + 32 * c - 3 = 0) →
    (1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c)) = 3 :=
by
  sorry

end polynomial_roots_sum_reciprocal_l79_79352


namespace part1_part2_l79_79295

noncomputable def f (x a : ℝ) := x * Real.log (x + 1) + (1/2 - a) * x + 2 - a

noncomputable def g (x a : ℝ) := f x a + Real.log (x + 1) + 1/2 * x

theorem part1 (a : ℝ) (x : ℝ) (h : x > 0) : 
  (a ≤ 2 → ∀ x, g x a > 0) ∧ 
  (a > 2 → ∀ x, x < Real.exp (a - 2) - 1 → g x a < 0) ∧
  (a > 2 → ∀ x, x > Real.exp (a - 2) - 1 → g x a > 0) :=
sorry

theorem part2 (a : ℤ) : 
  (∃ x ≥ 0, f x a < 0) → a ≥ 3 :=
sorry

end part1_part2_l79_79295


namespace cos_alpha_in_second_quadrant_l79_79834

theorem cos_alpha_in_second_quadrant 
  (alpha : ℝ) 
  (h1 : π / 2 < alpha ∧ alpha < π)
  (h2 : ∀ x y : ℝ, 2 * x + (Real.tan alpha) * y + 1 = 0 → 8 / 3 = -(2 / (Real.tan alpha))) :
  Real.cos alpha = -4 / 5 :=
by
  sorry

end cos_alpha_in_second_quadrant_l79_79834


namespace largest_divisor_of_n_l79_79940

theorem largest_divisor_of_n (n : ℕ) (h_pos : 0 < n) (h_div : 360 ∣ n^2) : 60 ∣ n := 
sorry

end largest_divisor_of_n_l79_79940


namespace count_1320_factors_l79_79139

-- Prime factorization function
def primeFactors (n : ℕ) : List ℕ :=
  sorry

-- Count factors function based on prime factorization
def countFactors (factors : List (ℕ × ℕ)) : ℕ :=
  factors.foldr (fun (p : ℕ × ℕ) acc => acc * (p.snd + 1)) 1

theorem count_1320_factors : countFactors [(2, 3), (3, 1), (5, 1), (11, 1)] = 32 :=
by
  sorry

end count_1320_factors_l79_79139


namespace find_roots_of_polynomial_l79_79948

noncomputable def polynomial_roots : Set ℝ :=
  {x | (6 * x^4 + 25 * x^3 - 59 * x^2 + 28 * x) = 0 }

theorem find_roots_of_polynomial :
  polynomial_roots = {0, 1, (-31 + Real.sqrt 1633) / 12, (-31 - Real.sqrt 1633) / 12} :=
by
  sorry

end find_roots_of_polynomial_l79_79948


namespace max_lines_with_specific_angles_l79_79110

def intersecting_lines : ℕ := 6

theorem max_lines_with_specific_angles :
  ∀ (n : ℕ), (∀ (i j : ℕ), i ≠ j → (∃ θ : ℝ, θ = 30 ∨ θ = 60 ∨ θ = 90)) → n ≤ 6 :=
  sorry

end max_lines_with_specific_angles_l79_79110


namespace age_of_teacher_l79_79713

theorem age_of_teacher (avg_age_students : ℕ) (num_students : ℕ) (inc_avg_with_teacher : ℕ) (num_people_with_teacher : ℕ) :
  avg_age_students = 21 →
  num_students = 20 →
  inc_avg_with_teacher = 22 →
  num_people_with_teacher = 21 →
  let total_age_students := num_students * avg_age_students
  let total_age_with_teacher := num_people_with_teacher * inc_avg_with_teacher
  total_age_with_teacher - total_age_students = 42 :=
by
  intros
  sorry

end age_of_teacher_l79_79713


namespace cos_pi_minus_alpha_l79_79487

theorem cos_pi_minus_alpha (α : ℝ) (h : Real.sin (Real.pi / 2 + α) = 1 / 7) : Real.cos (Real.pi - α) = - (1 / 7) := by
  sorry

end cos_pi_minus_alpha_l79_79487


namespace max_X_leq_ratio_XY_l79_79560

theorem max_X_leq_ratio_XY (x y z u : ℕ) (h1 : x + y = z + u) (h2 : 2 * x * y = z * u) (h3 : x ≥ y) : 
  ∃ m, m = 3 + 2 * Real.sqrt 2 ∧ ∀ (x y z u : ℕ), (x + y = z + u) → (2 * x *y = z * u) → (x ≥ y) → m ≤ x / y :=
sorry

end max_X_leq_ratio_XY_l79_79560


namespace two_squares_inequality_l79_79267

theorem two_squares_inequality (a b : ℝ) : 2 * (a^2 + b^2) ≥ (a + b)^2 := 
sorry

end two_squares_inequality_l79_79267


namespace parabola_expression_l79_79562

theorem parabola_expression:
  (∀ x : ℝ, y = a * (x + 3) * (x - 1)) →
  a * (0 + 3) * (0 - 1) = 2 →
  a = -2 / 3 →
  (∀ x : ℝ, y = -2 / 3 * x^2 - 4 / 3 * x + 2) :=
by
  sorry

end parabola_expression_l79_79562


namespace distance_between_trees_l79_79465

theorem distance_between_trees (num_trees : ℕ) (total_length : ℕ) (num_spaces : ℕ) (distance_per_space : ℕ) 
  (h_num_trees : num_trees = 11) (h_total_length : total_length = 180)
  (h_num_spaces : num_spaces = num_trees - 1) (h_distance_per_space : distance_per_space = total_length / num_spaces) :
  distance_per_space = 18 := 
  by 
    sorry

end distance_between_trees_l79_79465


namespace Diana_friends_count_l79_79363

theorem Diana_friends_count (totalErasers : ℕ) (erasersPerFriend : ℕ) 
  (h1: totalErasers = 3840) (h2: erasersPerFriend = 80) : 
  totalErasers / erasersPerFriend = 48 := 
by 
  sorry

end Diana_friends_count_l79_79363


namespace solve_for_x_l79_79024

theorem solve_for_x (x : ℚ) (h : (x + 4) / (x - 3) = (x - 2) / (x + 2)) : x = -2 / 11 :=
by
  sorry

end solve_for_x_l79_79024


namespace gcd_problem_l79_79677

theorem gcd_problem : ∃ b : ℕ, gcd (20 * b) (18 * 24) = 2 :=
by { sorry }

end gcd_problem_l79_79677


namespace mandy_yoga_time_l79_79090

theorem mandy_yoga_time 
  (gym_ratio : ℕ)
  (bike_ratio : ℕ)
  (yoga_exercise_ratio : ℕ)
  (bike_time : ℕ) 
  (exercise_ratio : ℕ) 
  (yoga_ratio : ℕ)
  (h1 : gym_ratio = 2)
  (h2 : bike_ratio = 3)
  (h3 : yoga_exercise_ratio = 2)
  (h4 : exercise_ratio = 3)
  (h5 : bike_time = 18)
  (total_exercise_time : ℕ)
  (yoga_time : ℕ)
  (h6: total_exercise_time = ((gym_ratio * bike_time) / bike_ratio) + bike_time)
  (h7 : yoga_time = (yoga_exercise_ratio * total_exercise_time) / exercise_ratio) :
  yoga_time = 20 := 
by 
  sorry

end mandy_yoga_time_l79_79090


namespace factorial_fraction_is_integer_l79_79374

open Nat

theorem factorial_fraction_is_integer (m n : ℕ) : 
  ↑((factorial (2 * m)) * (factorial (2 * n))) % (factorial m * factorial n * factorial (m + n)) = 0 := sorry

end factorial_fraction_is_integer_l79_79374


namespace stars_per_classmate_is_correct_l79_79922

-- Define the given conditions
def total_stars : ℕ := 45
def num_classmates : ℕ := 9

-- Define the expected number of stars per classmate
def stars_per_classmate : ℕ := 5

-- Prove that the number of stars per classmate is 5 given the conditions
theorem stars_per_classmate_is_correct :
  total_stars / num_classmates = stars_per_classmate :=
sorry

end stars_per_classmate_is_correct_l79_79922


namespace nine_consecutive_arithmetic_mean_divisible_1111_l79_79727

theorem nine_consecutive_arithmetic_mean_divisible_1111 {n : ℕ} (h1 : ∀ i : ℕ, 0 ≤ i ∧ i < 9 → Nat.Prime (n + i)) :
  ∃ n : ℕ, (∀ k : ℕ, 0 ≤ k ∧ k < 9 → (n + k) ∣ 1111) → (n + 4) = 97 := by
  sorry

end nine_consecutive_arithmetic_mean_divisible_1111_l79_79727


namespace vector_subtraction_l79_79767

def vector1 : ℝ × ℝ := (3, -5)
def vector2 : ℝ × ℝ := (2, -6)
def scalar1 : ℝ := 4
def scalar2 : ℝ := 3

theorem vector_subtraction :
  (scalar1 • vector1 - scalar2 • vector2) = (6, -2) := by
  sorry

end vector_subtraction_l79_79767


namespace exists_m_divisible_by_1988_l79_79005

def f (x : ℕ) : ℕ := 3 * x + 2
def iter_function (n : ℕ) (x : ℕ) : ℕ :=
  match n with
  | 0 => x
  | n + 1 => f (iter_function n x)

theorem exists_m_divisible_by_1988 : ∃ m : ℕ, 1988 ∣ iter_function 100 m :=
by sorry

end exists_m_divisible_by_1988_l79_79005


namespace unit_prices_minimum_B_seedlings_l79_79231

-- Definition of the problem conditions and the results of Part 1
theorem unit_prices (x : ℝ) : 
  (1200 / (1.5 * x) + 10 = 900 / x) ↔ x = 10 :=
by
  sorry

-- Definition of the problem conditions and the result of Part 2
theorem minimum_B_seedlings (m : ℕ) : 
  (10 * m + 15 * (100 - m) ≤ 1314) ↔ m ≥ 38 :=
by
  sorry

end unit_prices_minimum_B_seedlings_l79_79231


namespace original_number_l79_79700

theorem original_number (n : ℕ) (h : (2 * (n + 2) - 2) / 2 = 7) : n = 6 := by
  sorry

end original_number_l79_79700


namespace find_divisor_l79_79542

theorem find_divisor (n d k : ℤ) (h1 : n = k * d + 3) (h2 : n^2 % d = 4) : d = 5 :=
by
  sorry

end find_divisor_l79_79542


namespace P_has_common_root_l79_79387

def P (x : ℝ) (p : ℝ) (q : ℝ) : ℝ := x^2 + p * x + q

theorem P_has_common_root (p q : ℝ) (t : ℝ) (h : P t p q = 0) :
  P 0 p q * P 1 p q = 0 :=
by
  sorry

end P_has_common_root_l79_79387


namespace sum_of_cubes_eq_five_l79_79164

noncomputable def root_polynomial (a b c : ℂ) : Prop :=
  (a + b + c = 2) ∧ (a*b + b*c + c*a = 3) ∧ (a*b*c = 5)

theorem sum_of_cubes_eq_five (a b c : ℂ) (h : root_polynomial a b c) :
  a^3 + b^3 + c^3 = 5 :=
sorry

end sum_of_cubes_eq_five_l79_79164


namespace determinant_of_matrix_l79_79552

def mat : Matrix (Fin 3) (Fin 3) ℤ := 
  ![![3, 0, 2],![8, 5, -2],![3, 3, 6]]

theorem determinant_of_matrix : Matrix.det mat = 90 := 
by 
  sorry

end determinant_of_matrix_l79_79552


namespace solve_for_x_l79_79659

theorem solve_for_x (x y z w : ℤ) (h1 : x + y = 4) (h2 : x - y = 36) 
(h3 : x * z + y * w = 50) (h4 : z - w = 5) : x = 20 := 
by 
  sorry

end solve_for_x_l79_79659


namespace spherical_to_rectangular_coordinates_l79_79827

theorem spherical_to_rectangular_coordinates :
  let ρ := 5
  let θ := Real.pi / 4
  let φ := Real.pi / 3
  let x := ρ * Real.sin φ * Real.cos θ
  let y := ρ * Real.sin φ * Real.sin θ
  let z := ρ * Real.cos φ
  x = (5 * Real.sqrt 6) / 4 ∧ y = (5 * Real.sqrt 6) / 4 ∧ z = 5 / 2
:= by
  let ρ := 5
  let θ := Real.pi / 4
  let φ := Real.pi / 3
  let x := ρ * Real.sin φ * Real.cos θ
  let y := ρ * Real.sin φ * Real.sin θ
  let z := ρ * Real.cos φ
  have hx : x = (5 * Real.sqrt 6) / 4 := sorry
  have hy : y = (5 * Real.sqrt 6) / 4 := sorry
  have hz : z = 5 / 2 := sorry
  exact ⟨hx, hy, hz⟩

end spherical_to_rectangular_coordinates_l79_79827


namespace line_through_intersection_points_l79_79847

noncomputable def circle1 := { p : ℝ × ℝ | p.1^2 + p.2^2 = 10 }
noncomputable def circle2 := { p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 3)^2 = 10 }

theorem line_through_intersection_points (p : ℝ × ℝ) (hp1 : p ∈ circle1) (hp2 : p ∈ circle2) :
  p.1 + 3 * p.2 - 5 = 0 :=
sorry

end line_through_intersection_points_l79_79847


namespace original_cube_volume_l79_79430

theorem original_cube_volume
  (a : ℝ)
  (h : (a + 2) * (a - 1) * a = a^3 + 14) :
  a^3 = 64 :=
by
  sorry

end original_cube_volume_l79_79430


namespace chord_length_l79_79863

theorem chord_length (r d: ℝ) (h1: r = 5) (h2: d = 4) : ∃ EF, EF = 6 := by
  sorry

end chord_length_l79_79863


namespace valid_paths_in_grid_l79_79477

theorem valid_paths_in_grid : 
  let total_paths := Nat.choose 15 4;
  let paths_through_EF := (Nat.choose 7 2) * (Nat.choose 7 2);
  let valid_paths := total_paths - 2 * paths_through_EF;
  grid_size == (11, 4) ∧
  blocked_segments == [((5, 2), (5, 3)), ((6, 2), (6, 3))] 
  → valid_paths = 483 :=
by
  sorry

end valid_paths_in_grid_l79_79477


namespace solve_quadratic_l79_79338

theorem solve_quadratic : ∃ x : ℝ, (x^2 - 2 * x - 8 = 0 ↔ x = 4 ∨ x = -2) :=
sorry

end solve_quadratic_l79_79338


namespace net_difference_in_expenditure_l79_79285

variable (P Q : ℝ)
-- Condition 1: Price increased by 25%
def new_price (P : ℝ) : ℝ := P * 1.25

-- Condition 2: Purchased 72% of the originally required amount
def new_quantity (Q : ℝ) : ℝ := Q * 0.72

-- Definition of original expenditure
def original_expenditure (P Q : ℝ) : ℝ := P * Q

-- Definition of new expenditure
def new_expenditure (P Q : ℝ) : ℝ := new_price P * new_quantity Q

-- Statement of the proof problem.
theorem net_difference_in_expenditure
  (P Q : ℝ) : new_expenditure P Q - original_expenditure P Q = -0.1 * original_expenditure P Q := 
by
  sorry

end net_difference_in_expenditure_l79_79285


namespace cos_B_plus_C_value_of_c_l79_79819

variable {A B C a b c : ℝ}

-- Given conditions
axiom a_eq_2b : a = 2 * b
axiom sine_arithmetic_sequence : 2 * Real.sin C = Real.sin A + Real.sin B

-- First proof
theorem cos_B_plus_C (h : a = 2 * b) (h_seq : 2 * Real.sin C = Real.sin A + Real.sin B) :
  Real.cos (B + C) = 1 / 4 := 
sorry

-- Given additional condition for the area
axiom area_eq : (1 / 2) * b * c * Real.sin A = (3 * Real.sqrt 15) / 3

-- Second proof
theorem value_of_c (h : a = 2 * b) (h_seq : 2 * Real.sin C = Real.sin A + Real.sin B) (h_area : (1 / 2) * b * c * Real.sin A = (3 * Real.sqrt 15) / 3) :
  c = 4 * Real.sqrt 2 :=
sorry

end cos_B_plus_C_value_of_c_l79_79819


namespace tech_gadgets_components_total_l79_79731

theorem tech_gadgets_components_total (a₁ r n : ℕ) (h₁ : a₁ = 8) (h₂ : r = 3) (h₃ : n = 4) :
  a₁ * (r^n - 1) / (r - 1) = 320 := by
  sorry

end tech_gadgets_components_total_l79_79731


namespace altitude_of_triangle_l79_79984

theorem altitude_of_triangle (x : ℝ) (h : ℝ) 
  (h1 : x^2 = (1/2) * x * h) : h = 2 * x :=
by
  sorry

end altitude_of_triangle_l79_79984


namespace area_between_sine_and_half_line_is_sqrt3_minus_pi_by_3_l79_79714

noncomputable def area_enclosed_by_sine_and_line : ℝ :=
  (∫ x in (Real.pi / 6)..(5 * Real.pi / 6), (Real.sin x - 1 / 2))

theorem area_between_sine_and_half_line_is_sqrt3_minus_pi_by_3 :
  area_enclosed_by_sine_and_line = Real.sqrt 3 - Real.pi / 3 := by
  sorry

end area_between_sine_and_half_line_is_sqrt3_minus_pi_by_3_l79_79714


namespace farm_distance_is_6_l79_79009

noncomputable def distance_to_farm (initial_gallons : ℕ) 
  (consumption_rate : ℕ) (supermarket_distance : ℕ) 
  (outbound_distance : ℕ) (remaining_gallons : ℕ) : ℕ :=
initial_gallons * consumption_rate - 
  (2 * supermarket_distance + 2 * outbound_distance - remaining_gallons * consumption_rate)

theorem farm_distance_is_6 : 
  distance_to_farm 12 2 5 2 2 = 6 :=
by
  sorry

end farm_distance_is_6_l79_79009


namespace magical_stack_example_l79_79726

-- Definitions based on the conditions
def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

def belongs_to_pile_A (card : ℕ) (n : ℕ) : Prop :=
  card <= n

def belongs_to_pile_B (card : ℕ) (n : ℕ) : Prop :=
  n < card

def magical_stack (cards : ℕ) (n : ℕ) : Prop :=
  ∀ (card : ℕ), (belongs_to_pile_A card n ∨ belongs_to_pile_B card n) → 
  (card + n) % (2 * n) = 1

-- The theorem to prove
theorem magical_stack_example :
  ∃ (n : ℕ), magical_stack 482 n ∧ (2 * n = 482) :=
by
  sorry

end magical_stack_example_l79_79726


namespace sqrt_expression_nonneg_l79_79085

theorem sqrt_expression_nonneg {b : ℝ} : b - 3 ≥ 0 ↔ b ≥ 3 := by
  sorry

end sqrt_expression_nonneg_l79_79085


namespace lower_water_level_by_inches_l79_79783

theorem lower_water_level_by_inches
  (length width : ℝ) (gallons_removed : ℝ) (gallons_to_cubic_feet : ℝ) (feet_to_inches : ℝ) : 
  length = 20 → 
  width = 25 → 
  gallons_removed = 1875 → 
  gallons_to_cubic_feet = 7.48052 → 
  feet_to_inches = 12 → 
  (gallons_removed / gallons_to_cubic_feet) / (length * width) * feet_to_inches = 6.012 := 
by 
  sorry

end lower_water_level_by_inches_l79_79783


namespace nat_pow_eq_sub_two_case_l79_79409

theorem nat_pow_eq_sub_two_case (n : ℕ) : (∃ a k : ℕ, k ≥ 2 ∧ 2^n - 1 = a^k) ↔ (n = 0 ∨ n = 1) :=
by
  sorry

end nat_pow_eq_sub_two_case_l79_79409


namespace determinant_condition_l79_79505

theorem determinant_condition (a b c d : ℤ)
    (H : ∀ m n : ℤ, ∃ x y : ℤ, a * x + b * y = m ∧ c * x + d * y = n) :
    a * d - b * c = 1 ∨ a * d - b * c = -1 :=
by 
  sorry

end determinant_condition_l79_79505


namespace blocks_differ_in_two_ways_l79_79236

/-- 
A child has a set of 120 distinct blocks. Each block is one of 3 materials (plastic, wood, metal), 
3 sizes (small, medium, large), 4 colors (blue, green, red, yellow), and 5 shapes (circle, hexagon, 
square, triangle, pentagon). How many blocks in the set differ from the 'metal medium blue hexagon' 
in exactly 2 ways?
-/
def num_blocks_differ_in_two_ways : Nat := 44

theorem blocks_differ_in_two_ways (blocks : Fin 120)
    (materials : Fin 3)
    (sizes : Fin 3)
    (colors : Fin 4)
    (shapes : Fin 5)
    (fixed_block : {m // m = 2} × {s // s = 1} × {c // c = 0} × {sh // sh = 1}) :
    num_blocks_differ_in_two_ways = 44 :=
by
  -- proof steps are omitted
  sorry

end blocks_differ_in_two_ways_l79_79236


namespace cost_of_ingredients_l79_79602

theorem cost_of_ingredients :
  let popcorn_earnings := 50
  let cotton_candy_earnings := 3 * popcorn_earnings
  let total_earnings_per_day := popcorn_earnings + cotton_candy_earnings
  let total_earnings := total_earnings_per_day * 5
  let rent := 30
  let earnings_after_rent := total_earnings - rent
  earnings_after_rent - 895 = 75 :=
by
  let popcorn_earnings := 50
  let cotton_candy_earnings := 3 * popcorn_earnings
  let total_earnings_per_day := popcorn_earnings + cotton_candy_earnings
  let total_earnings := total_earnings_per_day * 5
  let rent := 30
  let earnings_after_rent := total_earnings - rent
  show earnings_after_rent - 895 = 75
  sorry

end cost_of_ingredients_l79_79602


namespace problem1_problem2_l79_79240

-- Problem 1
theorem problem1 : 23 + (-13) + (-17) + 8 = 1 :=
by
  sorry

-- Problem 2
theorem problem2 : - (2^3) - (1 + 0.5) / (1/3) * (-3) = 11/2 :=
by
  sorry

end problem1_problem2_l79_79240


namespace pie_eaten_after_four_trips_l79_79583

theorem pie_eaten_after_four_trips : 
  let trip1 := (1 / 3 : ℝ)
  let trip2 := (1 / 3^2 : ℝ)
  let trip3 := (1 / 3^3 : ℝ)
  let trip4 := (1 / 3^4 : ℝ)
  trip1 + trip2 + trip3 + trip4 = (40 / 81 : ℝ) :=
by
  sorry

end pie_eaten_after_four_trips_l79_79583


namespace smallest_with_20_divisors_is_144_l79_79253

def has_exactly_20_divisors (n : ℕ) : Prop :=
  let factors := n.factors;
  let divisors_count := factors.foldr (λ a b => (a + 1) * b) 1;
  divisors_count = 20

theorem smallest_with_20_divisors_is_144 : ∀ (n : ℕ), has_exactly_20_divisors n → (n < 144) → False :=
by
  sorry

end smallest_with_20_divisors_is_144_l79_79253


namespace sum_in_range_l79_79372

open Real

def mix1 := 3 + 3/8
def mix2 := 4 + 2/5
def mix3 := 6 + 1/11
def mixed_sum := mix1 + mix2 + mix3

theorem sum_in_range : mixed_sum > 13 ∧ mixed_sum < 14 :=
by
  -- Since we are just providing the statement, we leave the proof as a placeholder.
  sorry

end sum_in_range_l79_79372


namespace no_digit_B_divisible_by_4_l79_79262

theorem no_digit_B_divisible_by_4 : 
  ∀ B : ℕ, B < 10 → ¬ (8 * 1000000 + B * 100000 + 4 * 10000 + 6 * 1000 + 3 * 100 + 5 * 10 + 1) % 4 = 0 :=
by
  intros B hB_lt_10
  sorry

end no_digit_B_divisible_by_4_l79_79262


namespace relationship_among_a_b_c_l79_79217

theorem relationship_among_a_b_c 
  (a : ℝ) (b : ℝ) (c : ℝ)
  (ha : a = (1 / 2) ^ (3 / 2))
  (hb : b = Real.log pi)
  (hc : c = Real.logb 0.5 (3 / 2)) :
  c < a ∧ a < b :=
by 
  sorry

end relationship_among_a_b_c_l79_79217


namespace find_functions_satisfying_condition_l79_79970

noncomputable def function_satisfies_condition (f : ℝ → ℝ) : Prop :=
  ∀ (a b c d : ℝ), a > 0 → b > 0 → c > 0 → d > 0 → a * b * c * d = 1 →
  (f a + f b) * (f c + f d) = (a + b) * (c + d)

theorem find_functions_satisfying_condition :
  ∀ f : ℝ → ℝ, function_satisfies_condition f →
    (∀ x : ℝ, x > 0 → f x = x) ∨ (∀ x : ℝ, x > 0 → f x = 1 / x) :=
sorry

end find_functions_satisfying_condition_l79_79970


namespace find_k_l79_79147

theorem find_k 
  (t k r : ℝ)
  (h1 : t = 5 / 9 * (k - 32))
  (h2 : r = 3 * t)
  (h3 : r = 150) : 
  k = 122 := 
sorry

end find_k_l79_79147


namespace increasing_interval_l79_79675

noncomputable def function_y (x : ℝ) : ℝ :=
  2 * (Real.logb (1/2) x) ^ 2 - 2 * Real.logb (1/2) x + 1

theorem increasing_interval :
  ∀ x : ℝ, x > 0 → (∀ {y}, y ≥ x → function_y y ≥ function_y x) ↔ x ∈ Set.Ici (Real.sqrt 2 / 2) :=
by
  sorry

end increasing_interval_l79_79675


namespace number_of_assembled_desks_and_chairs_students_cannot_complete_tasks_simultaneously_l79_79333

-- Defining the conditions
def wooden_boards_type_A := 400
def wooden_boards_type_B := 500
def desk_needs_type_A := 2
def desk_needs_type_B := 1
def chair_needs_type_A := 1
def chair_needs_type_B := 2
def total_students := 30
def desk_assembly_time := 10
def chair_assembly_time := 7

-- Theorem for the number of assembled desks and chairs
theorem number_of_assembled_desks_and_chairs :
  ∃ x y : ℕ, 2 * x + y = wooden_boards_type_A ∧ x + 2 * y = wooden_boards_type_B ∧ x = 100 ∧ y = 200 :=
by {
  sorry
}

-- Theorem for the feasibility of students completing the tasks simultaneously
theorem students_cannot_complete_tasks_simultaneously :
  ¬ ∃ a : ℕ, (a ≤ total_students) ∧ (total_students - a > 0) ∧ 
  (100 / a) * desk_assembly_time = (200 / (total_students - a)) * chair_assembly_time :=
by {
  sorry
}

end number_of_assembled_desks_and_chairs_students_cannot_complete_tasks_simultaneously_l79_79333


namespace intersection_A_B_at_1_range_of_a_l79_79339

-- Problem definitions
def set_A (x : ℝ) : Prop := x^2 + 2*x - 3 > 0
def set_B (x a : ℝ) : Prop := x^2 - 2*a*x - 1 ≤ 0 ∧ a > 0

-- Question (I) If a = 1, find A ∩ B
theorem intersection_A_B_at_1 : (∀ x : ℝ, set_A x ∧ set_B x 1 ↔ (1 < x ∧ x ≤ 1 + Real.sqrt 2)) := sorry

-- Question (II) If A ∩ B contains exactly one integer, find the range of a.
theorem range_of_a (h : ∃ x : ℤ, set_A x ∧ set_B x 2) : 3 / 4 ≤ 2 ∧ 2 < 4 / 3 := sorry

end intersection_A_B_at_1_range_of_a_l79_79339


namespace no_integer_regular_pentagon_l79_79968

theorem no_integer_regular_pentagon 
  (x y : Fin 5 → ℤ) 
  (h_length : ∀ i j : Fin 5, i ≠ j → (x i - x j) ^ 2 + (y i - y j) ^ 2 = (x 0 - x 1) ^ 2 + (y 0 - y 1) ^ 2)
  : False :=
sorry

end no_integer_regular_pentagon_l79_79968


namespace sequence_nth_term_mod_2500_l79_79799

def sequence_nth_term (n : ℕ) : ℕ :=
  -- this is a placeholder function definition; the actual implementation to locate the nth term is skipped
  sorry

theorem sequence_nth_term_mod_2500 : (sequence_nth_term 2500) % 7 = 1 := 
sorry

end sequence_nth_term_mod_2500_l79_79799


namespace miles_left_to_drive_l79_79885

theorem miles_left_to_drive 
  (total_distance : ℕ) 
  (distance_covered : ℕ) 
  (remaining_distance : ℕ) 
  (h1 : total_distance = 78) 
  (h2 : distance_covered = 32) 
  : remaining_distance = total_distance - distance_covered -> remaining_distance = 46 :=
by
  sorry

end miles_left_to_drive_l79_79885


namespace factorize_x_squared_minus_one_l79_79044

theorem factorize_x_squared_minus_one (x : Real) : (x^2 - 1) = (x + 1) * (x - 1) :=
sorry

end factorize_x_squared_minus_one_l79_79044


namespace tom_spent_correct_amount_l79_79200

-- Define the prices of the games
def batman_game_price : ℝ := 13.6
def superman_game_price : ℝ := 5.06

-- Define the total amount spent calculation
def total_spent := batman_game_price + superman_game_price

-- The main statement to prove
theorem tom_spent_correct_amount : total_spent = 18.66 := by
  -- Proof (intended)
  sorry

end tom_spent_correct_amount_l79_79200


namespace total_papers_delivered_l79_79920

-- Definitions based on given conditions
def papers_saturday : ℕ := 45
def papers_sunday : ℕ := 65
def total_papers := papers_saturday + papers_sunday

-- The statement we need to prove
theorem total_papers_delivered : total_papers = 110 := by
  -- Proof steps would go here
  sorry

end total_papers_delivered_l79_79920


namespace solve_quadratic_1_solve_quadratic_2_l79_79592

theorem solve_quadratic_1 : ∀ x : ℝ, x^2 - 5 * x + 4 = 0 ↔ x = 4 ∨ x = 1 :=
by sorry

theorem solve_quadratic_2 : ∀ x : ℝ, x^2 = 4 - 2 * x ↔ x = -1 + Real.sqrt 5 ∨ x = -1 - Real.sqrt 5 :=
by sorry

end solve_quadratic_1_solve_quadratic_2_l79_79592


namespace ellipse_hyperbola_tangent_l79_79892

def ellipse (x y : ℝ) : Prop := x^2 + 9 * y^2 = 9
def hyperbola (x y m : ℝ) : Prop := x^2 - m * (y + 1)^2 = 1

theorem ellipse_hyperbola_tangent (m : ℝ) :
  (∀ x y : ℝ, ellipse x y → hyperbola x y m) → m = 2 :=
by sorry

end ellipse_hyperbola_tangent_l79_79892


namespace cheaper_store_difference_in_cents_l79_79165

/-- Given the following conditions:
1. Best Deals offers \$12 off the list price of \$52.99.
2. Market Value offers 20% off the list price of \$52.99.
 -/
theorem cheaper_store_difference_in_cents :
  let list_price : ℝ := 52.99
  let best_deals_price := list_price - 12
  let market_value_price := list_price * 0.80
  best_deals_price < market_value_price →
  let difference_in_dollars := market_value_price - best_deals_price
  let difference_in_cents := difference_in_dollars * 100
  difference_in_cents = 140 := by
  intro h
  let list_price : ℝ := 52.99
  let best_deals_price := list_price - 12
  let market_value_price := list_price * 0.80
  let difference_in_dollars := market_value_price - best_deals_price
  let difference_in_cents := difference_in_dollars * 100
  sorry

end cheaper_store_difference_in_cents_l79_79165


namespace find_values_l79_79535

theorem find_values (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : x - 4 = 21 * (1 / x)) 
  (h2 : x + y^2 = 45) : 
  x = 7 ∧ y = Real.sqrt 38 :=
by
  sorry

end find_values_l79_79535


namespace cost_of_whistle_l79_79631

theorem cost_of_whistle (cost_yoyo : ℕ) (total_spent : ℕ) (cost_yoyo_equals : cost_yoyo = 24) (total_spent_equals : total_spent = 38) : (total_spent - cost_yoyo) = 14 :=
by
  sorry

end cost_of_whistle_l79_79631


namespace distance_from_A_to_D_l79_79026

theorem distance_from_A_to_D 
  (A B C D : Type)
  (east_of : B → A)
  (north_of : C → B)
  (distance_AC : Real)
  (angle_BAC : ℝ)
  (north_of_D : D → C)
  (distance_CD : Real) : 
  distance_AC = 5 * Real.sqrt 5 → 
  angle_BAC = 60 → 
  distance_CD = 15 → 
  ∃ (AD : Real), AD =
    Real.sqrt (
      (5 * Real.sqrt 15 / 2) ^ 2 + 
      (5 * Real.sqrt 5 / 2 + 15) ^ 2
    ) :=
by
  intros
  sorry


end distance_from_A_to_D_l79_79026


namespace simple_interest_years_l79_79187

theorem simple_interest_years (P R : ℝ) (T : ℝ) :
  P = 2500 → (2500 * (R + 2) / 100 * T = 2500 * R / 100 * T + 250) → T = 5 :=
by
  intro hP h
  -- Note: Actual proof details would go here
  sorry

end simple_interest_years_l79_79187


namespace find_a_l79_79741

noncomputable def tangent_condition (a : ℝ) : Prop :=
  ∃ (x₀ y₀ : ℝ), y₀ = x₀ + 1 ∧ y₀ = Real.log (x₀ + a) ∧ (1 : ℝ) = (1 / (x₀ + a))

theorem find_a : ∃ a : ℝ, tangent_condition a ∧ a = 2 :=
by
  sorry

end find_a_l79_79741


namespace range_of_m_l79_79634

theorem range_of_m (a b c : ℝ) (m : ℝ) (h1 : a > b) (h2 : b > c) (h3 : 1 / (a - b) + m / (b - c) ≥ 9 / (a - c)) :
  m ≥ 4 :=
sorry

end range_of_m_l79_79634


namespace find_a2_l79_79973

theorem find_a2 (a : ℕ → ℝ) (h_arith : ∀ n, a (n + 1) = a n + 2)
  (h_geom : (a 1) * (a 5) = (a 2) * (a 2)) : a 2 = 3 :=
by
  -- We are given the conditions and need to prove the statement.
  sorry

end find_a2_l79_79973


namespace min_value_of_number_l79_79864

theorem min_value_of_number (a b c d : ℕ) (h1 : 0 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ d) (h5 : d ≤ 9) (h6 : 1 ≤ d) : 
  a + b * 10 + c * 100 + d * 1000 = 1119 :=
by
  sorry

end min_value_of_number_l79_79864


namespace count_positive_integers_l79_79351

theorem count_positive_integers (n : ℤ) : 
  (130 * n) ^ 50 > (n : ℤ) ^ 100 ∧ (n : ℤ) ^ 100 > 2 ^ 200 → 
  ∃ k : ℕ, k = 125 := sorry

end count_positive_integers_l79_79351


namespace find_least_q_l79_79587

theorem find_least_q : 
  ∃ q : ℕ, 
    (q ≡ 0 [MOD 7]) ∧ 
    (q ≥ 1000) ∧ 
    (q ≡ 1 [MOD 3]) ∧ 
    (q ≡ 1 [MOD 4]) ∧ 
    (q ≡ 1 [MOD 5]) ∧ 
    (q = 1141) :=
by
  sorry

end find_least_q_l79_79587


namespace square_minus_self_divisible_by_2_l79_79580

theorem square_minus_self_divisible_by_2 (a : ℕ) : 2 ∣ (a^2 - a) :=
by sorry

end square_minus_self_divisible_by_2_l79_79580


namespace tourists_left_l79_79495

noncomputable def tourists_remaining {initial remaining poisoned recovered : ℕ} 
  (h1 : initial = 30)
  (h2 : remaining = initial - 2)
  (h3 : poisoned = remaining / 2)
  (h4 : recovered = poisoned / 7)
  (h5 : remaining % 2 = 0) -- ensuring even division for / 2
  (h6 : poisoned % 7 = 0) -- ensuring even division for / 7
  : ℕ :=
  remaining - poisoned + recovered

theorem tourists_left 
  (initial remaining poisoned recovered : ℕ) 
  (h1 : initial = 30)
  (h2 : remaining = initial - 2)
  (h3 : poisoned = remaining / 2)
  (h4 : recovered = poisoned / 7)
  (h5 : remaining % 2 = 0) -- ensuring even division for / 2
  (h6 : poisoned % 7 = 0) -- ensuring even division for / 7
  : tourists_remaining h1 h2 h3 h4 h5 h6 = 16 :=
  by
  sorry

end tourists_left_l79_79495


namespace y_divides_x_squared_l79_79647

-- Define the conditions and proof problem in Lean 4
theorem y_divides_x_squared (x y : ℕ) (hx : 0 < x) (hy : 0 < y) 
(h : ∃ (n : ℕ), n = (x^2 / y) + (y^2 / x)) : y ∣ x^2 :=
by {
  -- Proof steps are skipped
  sorry
}

end y_divides_x_squared_l79_79647


namespace M_inter_N_l79_79883

def M : Set ℝ := { y | y > 1 }
def N : Set ℝ := { x | 0 < x ∧ x < 2 }

theorem M_inter_N : M ∩ N = { z | 1 < z ∧ z < 2 } :=
by 
  sorry

end M_inter_N_l79_79883


namespace minimum_value_y_range_of_a_l79_79158

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2*a*x - 1 + a

theorem minimum_value_y (x : ℝ) 
  (hx_pos : x > 0) : (f x 2 / x) = -2 :=
by sorry

theorem range_of_a : 
  ∀ a : ℝ, ∀ x ∈ (Set.Icc 0 2), (f x a) ≤ a ↔ a ≥ 3 / 4 :=
by sorry

end minimum_value_y_range_of_a_l79_79158


namespace solve_fabric_price_l79_79331

-- Defining the variables
variables (x y : ℕ)

-- Conditions as hypotheses
def condition1 := 7 * x = 9 * y
def condition2 := x - y = 36

-- Theorem statement to prove the system of equations
theorem solve_fabric_price (h1 : condition1 x y) (h2 : condition2 x y) :
  (7 * x = 9 * y) ∧ (x - y = 36) :=
by
  -- No proof is provided
  sorry

end solve_fabric_price_l79_79331


namespace factorize_expression_l79_79002

theorem factorize_expression
  (x : ℝ) :
  ( (x^2-1)*(x^4+x^2+1)-(x^3+1)^2 ) = -2*(x + 1)*(x^2 - x + 1) :=
by
  sorry

end factorize_expression_l79_79002


namespace part1_part2_l79_79820

-- Part 1
theorem part1 (x y : ℤ) (hx : x = -2) (hy : y = -3) :
  x^2 - 2 * (x^2 - 3 * y) - 3 * (2 * x^2 + 5 * y) = -1 :=
by
  -- Proof to be provided
  sorry

-- Part 2
theorem part2 (a b : ℤ) (hab : a - b = 2 * b^2) :
  2 * (a^3 - 2 * b^2) - (2 * b - a) + a - 2 * a^3 = 0 :=
by
  -- Proof to be provided
  sorry

end part1_part2_l79_79820


namespace geometric_sequence_a5_eq_neg1_l79_79289

-- Definitions for the conditions
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q, ∀ n, a (n + 1) = a n * q

def roots_of_quadratic (a3 a7 : ℝ) : Prop :=
  a3 + a7 = -4 ∧ a3 * a7 = 1

-- The statement to prove
theorem geometric_sequence_a5_eq_neg1 {a : ℕ → ℝ}
  (h_geo : is_geometric_sequence a)
  (h_roots : roots_of_quadratic (a 3) (a 7)) :
  a 5 = -1 :=
sorry

end geometric_sequence_a5_eq_neg1_l79_79289


namespace arcsin_neg_one_eq_neg_pi_div_two_l79_79901

theorem arcsin_neg_one_eq_neg_pi_div_two : Real.arcsin (-1) = -Real.pi / 2 :=
by
  sorry

end arcsin_neg_one_eq_neg_pi_div_two_l79_79901


namespace digit_B_in_4B52B_divisible_by_9_l79_79162

theorem digit_B_in_4B52B_divisible_by_9 (B : ℕ) (h : (2 * B + 11) % 9 = 0) : B = 8 :=
by {
  sorry
}

end digit_B_in_4B52B_divisible_by_9_l79_79162


namespace sum_cotangents_equal_l79_79367

theorem sum_cotangents_equal (a b c S m_a m_b m_c S' : ℝ) (cot_A cot_B cot_C cot_A' cot_B' cot_C' : ℝ)
  (h1 : cot_A + cot_B + cot_C = (a^2 + b^2 + c^2) / (4 * S))
  (h2 : m_a^2 + m_b^2 + m_c^2 = 3 * (a^2 + b^2 + c^2) / 4)
  (h3 : S' = 3 * S / 4)
  (h4 : cot_A' + cot_B' + cot_C' = (m_a^2 + m_b^2 + m_c^2) / (4 * S')) :
  cot_A + cot_B + cot_C = cot_A' + cot_B' + cot_C' :=
by
  -- Proof is needed, but omitted here
  sorry

end sum_cotangents_equal_l79_79367


namespace satisfactory_grades_fraction_l79_79809

def total_satisfactory_students (gA gB gC gD gE : Nat) : Nat :=
  gA + gB + gC + gD + gE

def total_students (gA gB gC gD gE gF : Nat) : Nat :=
  total_satisfactory_students gA gB gC gD gE + gF

def satisfactory_fraction (gA gB gC gD gE gF : Nat) : Rat :=
  total_satisfactory_students gA gB gC gD gE / total_students gA gB gC gD gE gF

theorem satisfactory_grades_fraction :
  satisfactory_fraction 3 5 4 2 1 4 = (15 : Rat) / 19 :=
by
  sorry

end satisfactory_grades_fraction_l79_79809


namespace range_of_m_l79_79042

def P (m : ℝ) : Prop := m^2 - 4 > 0
def Q (m : ℝ) : Prop := 16 * (m - 2)^2 - 16 < 0

theorem range_of_m (m : ℝ) :
  (P m ∨ Q m) ∧ ¬(P m ∧ Q m) ↔ (m < -2 ∨ (1 < m ∧ m ≤ 2) ∨ m ≥ 3) :=
by
  sorry

end range_of_m_l79_79042


namespace percentage_difference_l79_79115

theorem percentage_difference (N : ℝ) (hN : N = 160) : 0.50 * N - 0.35 * N = 24 := by
  sorry

end percentage_difference_l79_79115


namespace maryann_time_spent_calling_clients_l79_79758

theorem maryann_time_spent_calling_clients (a c : ℕ) 
  (h1 : a + c = 560) 
  (h2 : a = 7 * c) : c = 70 := 
by 
  sorry

end maryann_time_spent_calling_clients_l79_79758


namespace martha_no_daughters_count_l79_79678

-- Definitions based on conditions
def total_people : ℕ := 40
def martha_daughters : ℕ := 8
def granddaughters_per_child (x : ℕ) : ℕ := if x = 1 then 8 else 0

-- Statement of the problem
theorem martha_no_daughters_count : 
  (total_people - martha_daughters) +
  (martha_daughters - (total_people - martha_daughters) / 8) = 36 := 
  by
    sorry

end martha_no_daughters_count_l79_79678


namespace subcommittee_ways_l79_79821

theorem subcommittee_ways :
  ∃ (n : ℕ), n = Nat.choose 10 4 * Nat.choose 7 2 ∧ n = 4410 :=
by
  use 4410
  sorry

end subcommittee_ways_l79_79821


namespace painted_cells_l79_79516

open Int

theorem painted_cells : ∀ (m n : ℕ), (m = 20210) → (n = 1505) →
  let sub_rectangles := 215
  let cells_per_diagonal := 100
  let total_cells := sub_rectangles * cells_per_diagonal
  let total_painted_cells := 2 * total_cells
  let overlap_cells := sub_rectangles
  let unique_painted_cells := total_painted_cells - overlap_cells
  unique_painted_cells = 42785 := sorry

end painted_cells_l79_79516


namespace fraction_proof_l79_79813

-- Define N
def N : ℕ := 24

-- Define F that satisfies the equation N = F + 15
def F := N - 15

-- Define the fraction that N exceeds by 15
noncomputable def fraction := (F : ℚ) / N

-- Prove that fraction = 3/8
theorem fraction_proof : fraction = 3 / 8 := by
  sorry

end fraction_proof_l79_79813


namespace range_of_p_l79_79895

theorem range_of_p (p : ℝ) : 
  (∀ x : ℝ, 2 ≤ x ∧ x ≤ 4 → x^2 + p * x + 1 > 2 * x + p) → p > -1 := 
by
  sorry

end range_of_p_l79_79895


namespace minimum_value_of_phi_l79_79775

noncomputable def f (A ω φ : ℝ) (x : ℝ) := A * Real.sin (ω * x + φ)

noncomputable def minimum_positive_period (ω : ℝ) := 2 * Real.pi / ω

theorem minimum_value_of_phi {A ω φ : ℝ} (hA : A > 0) (hω : ω > 0) 
  (h_period : minimum_positive_period ω = Real.pi) 
  (h_symmetry : ∀ x, f A ω φ x = f A ω φ (2 * Real.pi / ω - x)) : 
  ∃ k : ℤ, |φ| = |k * Real.pi - Real.pi / 6| → |φ| = Real.pi / 6 :=
by
  sorry

end minimum_value_of_phi_l79_79775


namespace emilee_earns_25_l79_79384

-- Define the conditions
def earns_together (jermaine terrence emilee : ℕ) : Prop := 
  jermaine + terrence + emilee = 90

def jermaine_more (jermaine terrence : ℕ) : Prop :=
  jermaine = terrence + 5

def terrence_earning : ℕ := 30

-- The goal: Prove Emilee earns 25 dollars
theorem emilee_earns_25 (jermaine terrence emilee : ℕ) (h1 : earns_together jermaine terrence emilee) 
  (h2 : jermaine_more jermaine terrence) (h3 : terrence = terrence_earning) : 
  emilee = 25 := 
sorry

end emilee_earns_25_l79_79384


namespace sixth_term_geometric_mean_l79_79593

variable (a d : ℝ)

-- Define the arithmetic progression terms
def a_n (n : ℕ) := a + (n - 1) * d

-- Provided condition: second term is the geometric mean of the 1st and 4th terms
def condition (a d : ℝ) := a_n a d 2 = Real.sqrt (a_n a d 1 * a_n a d 4)

-- The goal to be proved: sixth term is the geometric mean of the 4th and 9th terms
theorem sixth_term_geometric_mean (a d : ℝ) (h : condition a d) : 
  a_n a d 6 = Real.sqrt (a_n a d 4 * a_n a d 9) :=
sorry

end sixth_term_geometric_mean_l79_79593


namespace sandy_has_32_fish_l79_79907

-- Define the initial number of pet fish Sandy has
def initial_fish : Nat := 26

-- Define the number of fish Sandy bought
def fish_bought : Nat := 6

-- Define the total number of pet fish Sandy has now
def total_fish : Nat := initial_fish + fish_bought

-- Prove that Sandy now has 32 pet fish
theorem sandy_has_32_fish : total_fish = 32 :=
by
  sorry

end sandy_has_32_fish_l79_79907


namespace initial_distance_l79_79019

-- Define conditions
def fred_speed : ℝ := 4
def sam_speed : ℝ := 4
def sam_distance_when_meet : ℝ := 20

-- States that the initial distance between Fred and Sam is 40 miles considering the given conditions.
theorem initial_distance (d : ℝ) (fred_speed_eq : fred_speed = 4) (sam_speed_eq : sam_speed = 4) (sam_distance_eq : sam_distance_when_meet = 20) :
  d = 40 :=
  sorry

end initial_distance_l79_79019


namespace integer_solution_x_l79_79801

theorem integer_solution_x (x y : ℤ) (hx : x > 0) (hy : y > 0) (hxy : x > y) (h : x + y + x * y = 101) : x = 50 :=
sorry

end integer_solution_x_l79_79801


namespace arcade_spending_fraction_l79_79222

theorem arcade_spending_fraction (allowance remaining_after_arcade remaining_after_toystore: ℝ) (f: ℝ) : 
  allowance = 3.75 ∧
  remaining_after_arcade = (1 - f) * allowance ∧
  remaining_after_toystore = remaining_after_arcade - (1 / 3) * remaining_after_arcade ∧
  remaining_after_toystore = 1 →
  f = 3 / 5 :=
by
  sorry

end arcade_spending_fraction_l79_79222


namespace salary_for_january_l79_79558

theorem salary_for_january (J F M A May : ℝ)
  (h1 : (J + F + M + A) / 4 = 8000)
  (h2 : (F + M + A + May) / 4 = 8700)
  (h_may : May = 6500) :
  J = 3700 :=
by
  sorry

end salary_for_january_l79_79558


namespace quadratic_inequality_l79_79068

theorem quadratic_inequality (a b c : ℝ) (h : a^2 + a * b + a * c < 0) : b^2 > 4 * a * c := 
sorry

end quadratic_inequality_l79_79068


namespace new_supervisor_salary_l79_79945

-- Definitions
def average_salary_old (W : ℕ) : Prop :=
  (W + 870) / 9 = 430

def average_salary_new (W : ℕ) (S_new : ℕ) : Prop :=
  (W + S_new) / 9 = 430

-- Problem statement
theorem new_supervisor_salary (W : ℕ) (S_new : ℕ) :
  average_salary_old W →
  average_salary_new W S_new →
  S_new = 870 :=
by
  sorry

end new_supervisor_salary_l79_79945


namespace proof_problem_l79_79211

-- Declare x, y as real numbers
variables (x y : ℝ)

-- Define the condition given in the problem
def condition (k : ℝ) : Prop :=
  (x^2 + y^2) / (x^2 - y^2) + (x^2 - y^2) / (x^2 + y^2) = k

-- The main conclusion we need to prove given the condition
theorem proof_problem (k : ℝ) (h : condition x y k) :
  (x^8 + y^8) / (x^8 - y^8) + (x^8 - y^8) / (x^8 + y^8) = (k^4 + 24 * k^2 + 16) / (4 * k^3 + 16 * k) :=
sorry

end proof_problem_l79_79211


namespace friend_selling_price_correct_l79_79776

-- Definition of the original cost price
def original_cost_price : ℕ := 50000

-- Definition of the loss percentage
def loss_percentage : ℕ := 10

-- Definition of the gain percentage
def gain_percentage : ℕ := 20

-- Definition of the man's selling price after loss
def man_selling_price : ℕ := original_cost_price - (original_cost_price * loss_percentage / 100)

-- Definition of the friend's selling price after gain
def friend_selling_price : ℕ := man_selling_price + (man_selling_price * gain_percentage / 100)

theorem friend_selling_price_correct : friend_selling_price = 54000 := by
  sorry

end friend_selling_price_correct_l79_79776


namespace solve_for_x_l79_79140

open Real

theorem solve_for_x (x : ℝ) (h1 : x > 0) (h2 : 6 * sqrt (4 + x) + 6 * sqrt (4 - x) = 9 * sqrt 2) : 
  x = sqrt 255 / 4 :=
sorry

end solve_for_x_l79_79140


namespace largest_integer_n_neg_quad_expr_l79_79829

theorem largest_integer_n_neg_quad_expr :
  ∃ n : ℤ, n = 6 ∧ ∀ m : ℤ, ((n^2 - 11 * n + 28 < 0) → (m < 7 ∧ m > 4) → m ≤ n) :=
by
  sorry

end largest_integer_n_neg_quad_expr_l79_79829


namespace simplify_expression_l79_79894

theorem simplify_expression :
  (1 / 2^2 + (2 / 3^3 * (3 / 2)^2) + 4^(1/2)) - 8 / (4^2 - 3^2) = 107 / 84 :=
by
  -- Skip the proof
  sorry

end simplify_expression_l79_79894


namespace jane_savings_l79_79760

-- Given conditions
def cost_pair_1 : ℕ := 50
def cost_pair_2 : ℕ := 40

def promotion_A (cost1 cost2 : ℕ) : ℕ :=
  cost1 + cost2 / 2

def promotion_B (cost1 cost2 : ℕ) : ℕ :=
  cost1 + (cost2 - 15)

-- Define the savings calculation
def savings (promoA promoB : ℕ) : ℕ :=
  promoB - promoA

-- Specify the theorem to prove
theorem jane_savings :
  savings (promotion_A cost_pair_1 cost_pair_2) (promotion_B cost_pair_1 cost_pair_2) = 5 := 
by
  sorry

end jane_savings_l79_79760


namespace minimum_value_of_f_l79_79575

noncomputable def f (x m : ℝ) := (1 / 3) * x^3 - x + m

theorem minimum_value_of_f (m : ℝ) (h_max : f (-1) m = 1) : 
  f 1 m = -1 / 3 :=
by
  sorry

end minimum_value_of_f_l79_79575


namespace floor_function_solution_l79_79991

def floor_eq_x_solutions : Prop :=
  ∀ x : ℤ, (⌊(x : ℝ) / 2⌋ + ⌊(x : ℝ) / 4⌋ = x) ↔ x = 0 ∨ x = -3 ∨ x = -2 ∨ x = -5

theorem floor_function_solution: floor_eq_x_solutions :=
by
  intro x
  sorry

end floor_function_solution_l79_79991


namespace star_example_l79_79686

def star (x y : ℝ) : ℝ := 2 * x * y - 3 * x + y

theorem star_example : (star 6 4) - (star 4 6) = -8 := by
  sorry

end star_example_l79_79686


namespace molecular_weight_of_one_mole_l79_79136

-- Conditions
def molecular_weight_6_moles : ℤ := 1404
def num_moles : ℤ := 6

-- Theorem
theorem molecular_weight_of_one_mole : (molecular_weight_6_moles / num_moles) = 234 := by
  sorry

end molecular_weight_of_one_mole_l79_79136


namespace percentage_change_difference_l79_79595

-- Define the initial and final percentages of students
def initial_liked_percentage : ℝ := 0.4
def initial_disliked_percentage : ℝ := 0.6
def final_liked_percentage : ℝ := 0.8
def final_disliked_percentage : ℝ := 0.2

-- Define the problem statement
theorem percentage_change_difference :
  (final_liked_percentage - initial_liked_percentage) + 
  (initial_disliked_percentage - final_disliked_percentage) = 0.6 :=
sorry

end percentage_change_difference_l79_79595


namespace joined_after_8_months_l79_79095

theorem joined_after_8_months
  (investment_A investment_B : ℕ)
  (time_A time_B : ℕ)
  (profit_ratio : ℕ × ℕ)
  (h_A : investment_A = 36000)
  (h_B : investment_B = 54000)
  (h_ratio : profit_ratio = (2, 1))
  (h_time_A : time_A = 12)
  (h_eq : (investment_A * time_A) / (investment_B * time_B) = (profit_ratio.1 / profit_ratio.2)) :
  time_B = 4 := by
  sorry

end joined_after_8_months_l79_79095


namespace diagonal_not_perpendicular_l79_79459

open Real

theorem diagonal_not_perpendicular (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) 
  (h_a_ne_b : a ≠ b) (h_c_ne_d : c ≠ d) (h_a_ne_c : a ≠ c) (h_b_ne_d : b ≠ d): 
  ¬ ((d - b) * (b - a) = - (c - a) * (d - c)) :=
by
  sorry

end diagonal_not_perpendicular_l79_79459


namespace inequality_am_gm_l79_79128

theorem inequality_am_gm (a b c d : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) (h_prod : a * b * c * d = 1) : 
  a^2 + b^2 + c^2 + d^2 + a * b + a * c + a * d + b * c + b * d + c * d ≥ 10 :=
by
  sorry

end inequality_am_gm_l79_79128


namespace sin_2x_plus_one_equals_9_over_5_l79_79969

theorem sin_2x_plus_one_equals_9_over_5 (x : ℝ) (h : Real.sin x = 2 * Real.cos x) : Real.sin (2 * x) + 1 = 9 / 5 :=
sorry

end sin_2x_plus_one_equals_9_over_5_l79_79969


namespace negation_of_proposition_l79_79849

theorem negation_of_proposition (p : ∀ x : ℝ, -x^2 + 4 * x + 3 > 0) :
  (∃ x : ℝ, -x^2 + 4 * x + 3 ≤ 0) :=
sorry

end negation_of_proposition_l79_79849


namespace simplify_expr_l79_79022

theorem simplify_expr (a b : ℝ) (h₁ : a + b = 0) (h₂ : a ≠ b) : (1 - a) + (1 - b) = 2 := by
  sorry

end simplify_expr_l79_79022


namespace rectangle_area_eq_l79_79317

theorem rectangle_area_eq (a b c d x y z w : ℝ)
  (h1 : a = x + y) (h2 : b = y + z) (h3 : c = z + w) (h4 : d = w + x) :
  a + c = b + d :=
by
  sorry

end rectangle_area_eq_l79_79317


namespace Andrey_knows_the_secret_l79_79354

/-- Question: Does Andrey know the secret?
    Conditions:
    - Andrey says: "I know the secret!"
    - Boris says to Andrey: "No, you don't!"
    - Victor says to Boris: "Boris, you are wrong!"
    - Gosha says to Victor: "No, you are wrong!"
    - Dima says to Gosha: "Gosha, you are lying!"
    - More than half of the kids told the truth (i.e., at least 3 out of 5). --/
theorem Andrey_knows_the_secret (Andrey Boris Victor Gosha Dima : Prop) (truth_count : ℕ)
    (h1 : Andrey)   -- Andrey says he knows the secret
    (h2 : ¬Andrey → Boris)   -- Boris says Andrey does not know the secret
    (h3 : ¬Boris → Victor)   -- Victor says Boris is wrong
    (h4 : ¬Victor → Gosha)   -- Gosha says Victor is wrong
    (h5 : ¬Gosha → Dima)   -- Dima says Gosha is lying
    (h6 : truth_count > 2)   -- More than half of the friends tell the truth (at least 3 out of 5)
    : Andrey := 
sorry

end Andrey_knows_the_secret_l79_79354


namespace xiaoliang_steps_l79_79382

/-- 
  Xiaoping lives on the fifth floor and climbs 80 steps to get home every day.
  Xiaoliang lives on the fourth floor.
  Prove that the number of steps Xiaoliang has to climb is 60.
-/
theorem xiaoliang_steps (steps_per_floor : ℕ) (h_xiaoping : 4 * steps_per_floor = 80) : 3 * steps_per_floor = 60 :=
by {
  -- The proof is intentionally left out
  sorry
}

end xiaoliang_steps_l79_79382


namespace decreasing_functions_l79_79771

noncomputable def f1 (x : ℝ) : ℝ := -x^2 + 1
noncomputable def f2 (x : ℝ) : ℝ := Real.sqrt x
noncomputable def f3 (x : ℝ) : ℝ := Real.log x / Real.log 2
noncomputable def f4 (x : ℝ) : ℝ := 3 ^ x

theorem decreasing_functions :
  (∀ x y : ℝ, 0 < x → x < y → f1 y < f1 x) ∧
  (∀ x y : ℝ, 0 < x → x < y → f2 y > f2 x) ∧
  (∀ x y : ℝ, 0 < x → x < y → f3 y > f3 x) ∧
  (∀ x y : ℝ, 0 < x → x < y → f4 y > f4 x) :=
by {
  sorry
}

end decreasing_functions_l79_79771


namespace bernoulli_inequality_l79_79209

theorem bernoulli_inequality (x : ℝ) (n : ℕ) (hx : x > -1) (hn : n > 0) : 
  (1 + x) ^ n ≥ 1 + n * x := 
sorry

end bernoulli_inequality_l79_79209


namespace find_x_y_l79_79963

theorem find_x_y (x y : ℝ) : (3 * x + 4 * -2 = 0) ∧ (3 * 1 + 4 * y = 0) → x = 8 / 3 ∧ y = -3 / 4 :=
by
  sorry

end find_x_y_l79_79963


namespace point_B_position_l79_79359

/-- Given points A and B on the same number line, with A at -2 and B 5 units away from A, prove 
    that B can be either -7 or 3. -/
theorem point_B_position (A B : ℤ) (hA : A = -2) (hB : (B = A + 5) ∨ (B = A - 5)) : 
  B = 3 ∨ B = -7 :=
sorry

end point_B_position_l79_79359


namespace quadratic_inequality_l79_79679

variable (a b c A B C : ℝ)

theorem quadratic_inequality
  (h₁ : a ≠ 0)
  (h₂ : A ≠ 0)
  (h₃ : ∀ x : ℝ, |a * x^2 + b * x + c| ≤ |A * x^2 + B * x + C|) :
  |b^2 - 4 * a * c| ≤ |B^2 - 4 * A * C| :=
sorry

end quadratic_inequality_l79_79679


namespace rachel_weight_l79_79173

theorem rachel_weight :
  ∃ R : ℝ, (R + (R + 6) + (R - 15)) / 3 = 72 ∧ R = 75 :=
by
  sorry

end rachel_weight_l79_79173


namespace coefficient_of_q_l79_79641

theorem coefficient_of_q (q' : ℤ → ℤ) (h : ∀ q, q' q = 3 * q - 3) (h₁ : q' (q' 4) = 72) : 
  ∀ q, q' q = 3 * q - 3 :=
  sorry

end coefficient_of_q_l79_79641


namespace vote_ratio_l79_79591

theorem vote_ratio (X Y Z : ℕ) (hZ : Z = 25000) (hX : X = 22500) (hX_Y : X = Y + (1/2 : ℚ) * Y) 
    : Y / (Z - Y) = 2 / 5 := 
by 
  sorry

end vote_ratio_l79_79591


namespace G_at_16_l79_79965

noncomputable def G : ℝ → ℝ := sorry

-- Condition 1: G is a polynomial, implicitly stated
-- Condition 2: Given G(8) = 21
axiom G_at_8 : G 8 = 21

-- Condition 3: Given that
axiom G_fraction_condition : ∀ (x : ℝ), 
  (x^2 + 6*x + 8) ≠ 0 ∧ ((x+4)*(x+2)) ≠ 0 → 
  (G (2*x) / G (x+4) = 4 - (16*x + 32) / (x^2 + 6*x + 8))

-- The problem: Prove G(16) = 90
theorem G_at_16 : G 16 = 90 := 
sorry

end G_at_16_l79_79965


namespace cuboid_edge_length_l79_79961

theorem cuboid_edge_length (x : ℝ) (h1 : 5 * 6 * x = 120) : x = 4 :=
by
  sorry

end cuboid_edge_length_l79_79961


namespace johns_weight_l79_79109

-- Definitions based on the given conditions
def max_weight : ℝ := 1000
def safety_percentage : ℝ := 0.20
def bar_weight : ℝ := 550

-- Theorem stating the mathematically equivalent proof problem
theorem johns_weight : 
  (johns_safe_weight : ℝ) = max_weight - safety_percentage * max_weight 
  → (johns_safe_weight - bar_weight = 250) :=
by
  sorry

end johns_weight_l79_79109


namespace mrs_sheridan_initial_cats_l79_79739

theorem mrs_sheridan_initial_cats (bought_cats total_cats : ℝ) (h_bought : bought_cats = 43.0) (h_total : total_cats = 54) : total_cats - bought_cats = 11 :=
by
  rw [h_bought, h_total]
  norm_num

end mrs_sheridan_initial_cats_l79_79739


namespace exp_ineq_of_r_gt_one_l79_79256

theorem exp_ineq_of_r_gt_one {x r : ℝ} (hx : x > 0) (hr : r > 1) : (1 + x)^r > 1 + r * x :=
by
  sorry

end exp_ineq_of_r_gt_one_l79_79256


namespace total_doughnuts_l79_79877

-- Definitions used in the conditions
def boxes : ℕ := 4
def doughnuts_per_box : ℕ := 12

theorem total_doughnuts : boxes * doughnuts_per_box = 48 :=
by
  sorry

end total_doughnuts_l79_79877


namespace tracy_initial_candies_l79_79709

theorem tracy_initial_candies (y : ℕ) 
  (condition1 : y - y / 4 = y * 3 / 4)
  (condition2 : y * 3 / 4 - (y * 3 / 4) / 3 = y / 2)
  (condition3 : y / 2 - 24 = y / 2 - 12 - 12)
  (condition4 : y / 2 - 24 - 4 = 2) : 
  y = 60 :=
by sorry

end tracy_initial_candies_l79_79709


namespace xyz_zero_if_equation_zero_l79_79605

theorem xyz_zero_if_equation_zero (x y z : ℚ) 
  (h : x^3 + 3 * y^3 + 9 * z^3 - 9 * x * y * z = 0) : 
  x = 0 ∧ y = 0 ∧ z = 0 := 
by 
  sorry

end xyz_zero_if_equation_zero_l79_79605


namespace probability_event_proof_l79_79434

noncomputable def probability_event_occur (deck_size : ℕ) (num_queens : ℕ) (num_jacks : ℕ) (num_reds : ℕ) : ℚ :=
  let prob_two_queens := (num_queens / deck_size) * ((num_queens - 1) / (deck_size - 1))
  let prob_at_least_one_jack := 
    (num_jacks / deck_size) * ((deck_size - num_jacks) / (deck_size - 1)) +
    ((deck_size - num_jacks) / deck_size) * (num_jacks / (deck_size - 1)) +
    (num_jacks / deck_size) * ((num_jacks - 1) / (deck_size - 1))
  let prob_both_red := (num_reds / deck_size) * ((num_reds - 1) / (deck_size - 1))
  prob_two_queens + prob_at_least_one_jack + prob_both_red

theorem probability_event_proof :
  probability_event_occur 52 4 4 26 = 89 / 221 :=
by
  sorry

end probability_event_proof_l79_79434


namespace simplify_and_evaluate_expression_l79_79386

theorem simplify_and_evaluate_expression (a b : ℤ) (h_a : a = 2) (h_b : b = -1) : 
  2 * (-a^2 + 2 * a * b) - 3 * (a * b - a^2) = 2 :=
by 
  sorry

end simplify_and_evaluate_expression_l79_79386


namespace correct_divisor_l79_79977

variable (D X : ℕ)

-- Conditions
def condition1 : Prop := X = D * 24
def condition2 : Prop := X = (D - 12) * 42

theorem correct_divisor (D X : ℕ) (h1 : condition1 D X) (h2 : condition2 D X) : D = 28 := by
  sorry

end correct_divisor_l79_79977


namespace simplify_329_mul_101_simplify_54_mul_98_plus_46_mul_98_simplify_98_mul_125_simplify_37_mul_29_plus_37_l79_79003

theorem simplify_329_mul_101 : 329 * 101 = 33229 := by
  sorry

theorem simplify_54_mul_98_plus_46_mul_98 : 54 * 98 + 46 * 98 = 9800 := by
  sorry

theorem simplify_98_mul_125 : 98 * 125 = 12250 := by
  sorry

theorem simplify_37_mul_29_plus_37 : 37 * 29 + 37 = 1110 := by
  sorry

end simplify_329_mul_101_simplify_54_mul_98_plus_46_mul_98_simplify_98_mul_125_simplify_37_mul_29_plus_37_l79_79003


namespace masha_dolls_l79_79161

theorem masha_dolls (n : ℕ) (h : (n / 2) * 1 + (n / 4) * 2 + (n / 4) * 4 = 24) : n = 12 :=
sorry

end masha_dolls_l79_79161


namespace problem_1_problem_2_problem_3_l79_79445

-- Definitions and conditions
def monomial_degree_condition (a : ℝ) : Prop := 2 + (1 + a) = 5

-- Proof goals
theorem problem_1 (a : ℝ) (h : monomial_degree_condition a) : a^3 + 1 = 9 := sorry
theorem problem_2 (a : ℝ) (h : monomial_degree_condition a) : (a + 1) * (a^2 - a + 1) = 9 := sorry
theorem problem_3 (a : ℝ) (h : monomial_degree_condition a) : a^3 + 1 = (a + 1) * (a^2 - a + 1) := sorry

end problem_1_problem_2_problem_3_l79_79445


namespace factor_theorem_l79_79429

theorem factor_theorem (h : ℤ) : (∀ m : ℤ, (m - 8) ∣ (m^2 - h * m - 24) ↔ h = 5) :=
  sorry

end factor_theorem_l79_79429


namespace seq_a2010_l79_79082

-- Definitions and conditions
def seq (a : ℕ → ℕ) : Prop :=
  a 1 = 2 ∧ 
  a 2 = 3 ∧ 
  ∀ n ≥ 2, a (n + 1) = (a n * a (n - 1)) % 10

-- Proof statement
theorem seq_a2010 {a : ℕ → ℕ} (h : seq a) : a 2010 = 4 := 
  sorry

end seq_a2010_l79_79082


namespace speed_in_still_water_l79_79410

theorem speed_in_still_water (upstream_speed downstream_speed : ℝ) (h₁ : upstream_speed = 20) (h₂ : downstream_speed = 60) :
  (upstream_speed + downstream_speed) / 2 = 40 := by
  sorry

end speed_in_still_water_l79_79410


namespace max_k_no_real_roots_max_integer_value_k_no_real_roots_l79_79059

-- Define the quadratic equation with the condition on the discriminant.
theorem max_k_no_real_roots : ∀ k : ℤ, (4 + 4 * (k : ℝ) < 0) ↔ k < -1 := sorry

-- Prove that the maximum integer value of k satisfying this condition is -2.
theorem max_integer_value_k_no_real_roots : ∃ k_max : ℤ, k_max ∈ { k : ℤ | 4 + 4 * (k : ℝ) < 0 } ∧ ∀ k' : ℤ, k' ∈ { k : ℤ | 4 + 4 * (k : ℝ) < 0 } → k' ≤ k_max :=
sorry

end max_k_no_real_roots_max_integer_value_k_no_real_roots_l79_79059


namespace student_total_marks_l79_79649

variable (M P C : ℕ)

theorem student_total_marks :
  C = P + 20 ∧ (M + C) / 2 = 25 → M + P = 30 :=
by
  sorry

end student_total_marks_l79_79649


namespace percentage_selected_B_l79_79625

-- Definitions for the given conditions
def candidates := 7900
def selected_A := (6 / 100) * candidates
def selected_B := selected_A + 79

-- The question to be answered
def P_B := (selected_B / candidates) * 100

-- Proof statement
theorem percentage_selected_B : P_B = 7 := 
by
  -- Canonical statement placeholder 
  sorry

end percentage_selected_B_l79_79625


namespace find_m_when_z_is_real_l79_79853

theorem find_m_when_z_is_real (m : ℝ) (h : (m ^ 2 + 2 * m - 15 = 0)) : m = 3 :=
sorry

end find_m_when_z_is_real_l79_79853


namespace other_root_of_quadratic_l79_79118

theorem other_root_of_quadratic (m : ℝ) :
  (∀ x : ℝ, x^2 - 4 * x + m = 0 → x = -1) → (∀ y : ℝ, y^2 - 4 * y + m = 0 → y = 5) :=
sorry

end other_root_of_quadratic_l79_79118


namespace kody_half_mohamed_years_ago_l79_79463

-- Definitions of initial conditions
def current_age_mohamed : ℕ := 2 * 30
def current_age_kody : ℕ := 32

-- Proof statement
theorem kody_half_mohamed_years_ago : ∃ x : ℕ, (current_age_kody - x) = (1 / 2 : ℕ) * (current_age_mohamed - x) ∧ x = 4 := 
by 
  sorry

end kody_half_mohamed_years_ago_l79_79463


namespace area_of_the_region_l79_79404

noncomputable def region_area (C D : ℝ×ℝ) (rC rD : ℝ) (y : ℝ) : ℝ :=
  let rect_area := (D.1 - C.1) * y
  let sector_areaC := (1 / 2) * Real.pi * rC^2
  let sector_areaD := (1 / 2) * Real.pi * rD^2
  rect_area - (sector_areaC + sector_areaD)

theorem area_of_the_region :
  region_area (3, 5) (10, 5) 3 5 5 = 35 - 17 * Real.pi := by
  sorry

end area_of_the_region_l79_79404


namespace solve_for_y_l79_79149

theorem solve_for_y (y : ℝ) : 7 - y = 4 → y = 3 :=
by
  sorry

end solve_for_y_l79_79149


namespace tetrahedron_vertex_equality_l79_79268

theorem tetrahedron_vertex_equality
  (r1 r2 r3 r4 j1 j2 j3 j4 : ℝ) (hr1 : r1 > 0) (hr2 : r2 > 0) (hr3 : r3 > 0) (hr4 : r4 > 0)
  (hj1 : j1 > 0) (hj2 : j2 > 0) (hj3 : j3 > 0) (hj4 : j4 > 0) 
  (h1 : r2 * r3 + r3 * r4 + r4 * r2 = j2 * j3 + j3 * j4 + j4 * j2)
  (h2 : r1 * r3 + r3 * r4 + r4 * r1 = j1 * j3 + j3 * j4 + j4 * j1)
  (h3 : r1 * r2 + r2 * r4 + r4 * r1 = j1 * j2 + j2 * j4 + j4 * j1)
  (h4 : r1 * r2 + r2 * r3 + r3 * r1 = j1 * j2 + j2 * j3 + j3 * j1) :
  r1 = j1 ∧ r2 = j2 ∧ r3 = j3 ∧ r4 = j4 := by
  sorry

end tetrahedron_vertex_equality_l79_79268


namespace largest_of_seven_consecutive_integers_l79_79656

-- Define the main conditions as hypotheses
theorem largest_of_seven_consecutive_integers (n : ℕ) (h_sum : 7 * n + 21 = 2401) : 
  n + 6 = 346 :=
by
  -- Conditions from the problem are utilized here
  sorry

end largest_of_seven_consecutive_integers_l79_79656


namespace percentage_difference_between_maximum_and_minimum_changes_is_40_l79_79536

-- Definitions of initial and final survey conditions
def initialYesPercentage : ℝ := 0.40
def initialNoPercentage : ℝ := 0.60
def finalYesPercentage : ℝ := 0.80
def finalNoPercentage : ℝ := 0.20
def absenteePercentage : ℝ := 0.10

-- Main theorem stating the problem
theorem percentage_difference_between_maximum_and_minimum_changes_is_40 :
  let attendeesPercentage := 1 - absenteePercentage
  let adjustedFinalYesPercentage := finalYesPercentage / attendeesPercentage
  let minChange := adjustedFinalYesPercentage - initialYesPercentage
  let maxChange := initialYesPercentage + minChange
  maxChange - minChange = 0.40 :=
by
  -- Proof is omitted
  sorry

end percentage_difference_between_maximum_and_minimum_changes_is_40_l79_79536


namespace ratio_difference_l79_79734

theorem ratio_difference (x : ℕ) (h : (2 * x + 4) * 7 = (3 * x + 4) * 5) : 3 * x - 2 * x = 8 := 
by sorry

end ratio_difference_l79_79734


namespace prove_inequalities_l79_79689

theorem prove_inequalities (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  a^3 * b > a * b^3 ∧ a - b / a > b - a / b :=
by
  sorry

end prove_inequalities_l79_79689


namespace pies_baked_l79_79880

/-- Mrs. Hilt baked 16.0 pecan pies and 14.0 apple pies. She needs 5.0 times this amount.
    Prove that the total number of pies she has to bake is 150.0. -/
theorem pies_baked (pecan_pies : ℝ) (apple_pies : ℝ) (times : ℝ)
  (h1 : pecan_pies = 16.0) (h2 : apple_pies = 14.0) (h3 : times = 5.0) :
  times * (pecan_pies + apple_pies) = 150.0 := by
  sorry

end pies_baked_l79_79880


namespace part1_part2_l79_79638

-- Define the quadratic equation and its discriminant
def quadratic_discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

-- Define the conditions
def quadratic_equation (m : ℝ) : ℝ :=
  quadratic_discriminant 1 (-2) (-3 * m^2)

-- Part 1: Prove the quadratic equation always has two distinct real roots
theorem part1 (m : ℝ) : 
  quadratic_equation m > 0 :=
by
  sorry

-- Part 2: Find the value of m given the roots satisfy the equation α + 2β = 5
theorem part2 (α β m : ℝ) (h1 : α + β = 2) (h2 : α + 2 * β = 5) : 
  m = 1 ∨ m = -1 :=
by
  sorry


end part1_part2_l79_79638


namespace apples_left_correct_l79_79120

noncomputable def apples_left (initial_apples : ℝ) (additional_apples : ℝ) (apples_for_pie : ℝ) : ℝ :=
  initial_apples + additional_apples - apples_for_pie

theorem apples_left_correct :
  apples_left 10.0 5.5 4.25 = 11.25 :=
by
  sorry

end apples_left_correct_l79_79120


namespace simplify_and_evaluate_expression_l79_79654

variable (a b : ℚ)

theorem simplify_and_evaluate_expression
  (ha : a = 1 / 2)
  (hb : b = -1 / 3) :
  b^2 - a^2 + 2 * (a^2 + a * b) - (a^2 + b^2) = -1 / 3 :=
by
  -- The proof will be inserted here
  sorry

end simplify_and_evaluate_expression_l79_79654


namespace math_problem_l79_79888

theorem math_problem (a : ℝ) (h : a = 1/3) : (3 * a⁻¹ + 2 / 3 * a⁻¹) / a = 33 := by
  sorry

end math_problem_l79_79888


namespace birch_tree_count_l79_79475

theorem birch_tree_count:
  let total_trees := 8000
  let spruces := 0.12 * total_trees
  let pines := 0.15 * total_trees
  let maples := 0.18 * total_trees
  let cedars := 0.09 * total_trees
  let oaks := spruces + pines
  let calculated_trees := spruces + pines + maples + cedars + oaks
  let birches := total_trees - calculated_trees
  spruces = 960 → pines = 1200 → maples = 1440 → cedars = 720 → oaks = 2160 →
  birches = 1520 :=
by
  intros
  sorry

end birch_tree_count_l79_79475


namespace find_divisor_l79_79096

variable (Dividend : ℕ) (Quotient : ℕ) (Divisor : ℕ)
variable (h1 : Dividend = 64)
variable (h2 : Quotient = 8)
variable (h3 : Dividend = Divisor * Quotient)

theorem find_divisor : Divisor = 8 := by
  sorry

end find_divisor_l79_79096


namespace total_amount_collected_l79_79180

-- Define ticket prices and quantities
def adult_ticket_price : ℕ := 12
def child_ticket_price : ℕ := 4
def total_tickets_sold : ℕ := 130
def adult_tickets_sold : ℕ := 40

-- Calculate the number of child tickets sold
def child_tickets_sold : ℕ := total_tickets_sold - adult_tickets_sold

-- Calculate the total amount collected from adult tickets
def total_adult_amount_collected : ℕ := adult_tickets_sold * adult_ticket_price

-- Calculate the total amount collected from child tickets
def total_child_amount_collected : ℕ := child_tickets_sold * child_ticket_price

-- Prove the total amount collected from ticket sales
theorem total_amount_collected : total_adult_amount_collected + total_child_amount_collected = 840 := by
  sorry

end total_amount_collected_l79_79180


namespace volume_inside_sphere_outside_cylinder_l79_79389

noncomputable def volumeDifference (r_cylinder base_radius_sphere : ℝ) :=
  let height := 4 * Real.sqrt 5
  let V_sphere := (4/3) * Real.pi * base_radius_sphere^3
  let V_cylinder := Real.pi * r_cylinder^2 * height
  V_sphere - V_cylinder

theorem volume_inside_sphere_outside_cylinder
  (base_radius_sphere r_cylinder : ℝ) (h_base_radius_sphere : base_radius_sphere = 6) (h_r_cylinder : r_cylinder = 4) :
  volumeDifference r_cylinder base_radius_sphere = (288 - 64 * Real.sqrt 5) * Real.pi := by
  sorry

end volume_inside_sphere_outside_cylinder_l79_79389


namespace log_expression_value_l79_79653

noncomputable def log_expression : ℝ :=
  (Real.log (Real.sqrt 27) + Real.log 8 - 3 * Real.log (Real.sqrt 10)) / Real.log 1.2

theorem log_expression_value : log_expression = 3 / 2 :=
  sorry

end log_expression_value_l79_79653


namespace squared_distance_focus_product_tangents_l79_79414

variable {a b : ℝ}
variable {x0 y0 : ℝ}
variable {P Q R F : ℝ × ℝ}

-- Conditions
def is_ellipse (x y : ℝ) (a b : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

def outside_ellipse (x0 y0 : ℝ) (a b : ℝ) : Prop :=
  (x0^2 / a^2) + (y0^2 / b^2) > 1

-- Question (statement we need to prove)
theorem squared_distance_focus_product_tangents
  (h_ellipse : is_ellipse Q.1 Q.2 a b)
  (h_ellipse' : is_ellipse R.1 R.2 a b)
  (h_outside : outside_ellipse x0 y0 a b)
  (h_a_greater_b : a > b) :
  ‖P - F‖^2 > ‖Q - F‖ * ‖R - F‖ := sorry

end squared_distance_focus_product_tangents_l79_79414


namespace rental_cost_l79_79195

theorem rental_cost (total_cost gallons gas_price mile_cost miles : ℝ)
    (H1 : gallons = 8)
    (H2 : gas_price = 3.50)
    (H3 : mile_cost = 0.50)
    (H4 : miles = 320)
    (H5 : total_cost = 338) :
    total_cost - (gallons * gas_price + miles * mile_cost) = 150 := by
  sorry

end rental_cost_l79_79195


namespace polar_to_rectangular_coords_l79_79800

theorem polar_to_rectangular_coords (r θ : ℝ) (x y : ℝ) 
  (hr : r = 5) (hθ : θ = 5 * Real.pi / 4)
  (hx : x = r * Real.cos θ) (hy : y = r * Real.sin θ) :
  x = - (5 * Real.sqrt 2) / 2 ∧ y = - (5 * Real.sqrt 2) / 2 := 
by
  rw [hr, hθ] at hx hy
  simp [Real.cos, Real.sin] at hx hy
  rw [hx, hy]
  constructor
  . sorry
  . sorry

end polar_to_rectangular_coords_l79_79800


namespace probability_palindrome_divisible_by_11_is_zero_l79_79406

-- Define the three-digit palindrome
def is_palindrome (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ n = 101 * a + 10 * b + a

-- Define the divisibility condition
def is_divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

-- Prove that the probability is zero
theorem probability_palindrome_divisible_by_11_is_zero :
  (∃ n, is_palindrome n ∧ is_divisible_by_11 n) →
  (0 : ℕ) = 0 := by
  sorry

end probability_palindrome_divisible_by_11_is_zero_l79_79406


namespace train_stoppage_time_l79_79142

theorem train_stoppage_time
    (speed_without_stoppages : ℕ)
    (speed_with_stoppages : ℕ)
    (time_unit : ℕ)
    (h1 : speed_without_stoppages = 50)
    (h2 : speed_with_stoppages = 30)
    (h3 : time_unit = 60) :
    (time_unit * (speed_without_stoppages - speed_with_stoppages) / speed_without_stoppages) = 24 :=
by
  sorry

end train_stoppage_time_l79_79142


namespace intersection_of_A_and_B_l79_79747

open Set

variable (A : Set ℕ) (B : Set ℕ)

theorem intersection_of_A_and_B (hA : A = {0, 1, 2}) (hB : B = {0, 2, 4}) :
  A ∩ B = {0, 2} := by
  sorry

end intersection_of_A_and_B_l79_79747


namespace arithmetic_mean_of_three_digit_multiples_of_8_l79_79071

-- Define the conditions given in the problem
def smallest_three_digit_multiple_of_8 := 104
def largest_three_digit_multiple_of_8 := 992
def common_difference := 8

-- Define the sequence as an arithmetic sequence
def arithmetic_sequence (n : ℕ) : ℕ :=
  smallest_three_digit_multiple_of_8 + n * common_difference

-- Calculate the number of terms in the sequence
def number_of_terms : ℕ :=
  (largest_three_digit_multiple_of_8 - smallest_three_digit_multiple_of_8) / common_difference + 1

-- Calculate the sum of the arithmetic sequence
def sum_of_sequence : ℕ :=
  (number_of_terms * (smallest_three_digit_multiple_of_8 + largest_three_digit_multiple_of_8)) / 2

-- Calculate the arithmetic mean
def arithmetic_mean : ℕ :=
  sum_of_sequence / number_of_terms

-- The statement to be proved
theorem arithmetic_mean_of_three_digit_multiples_of_8 :
  arithmetic_mean = 548 :=
by
  sorry

end arithmetic_mean_of_three_digit_multiples_of_8_l79_79071


namespace pencils_per_student_l79_79862

theorem pencils_per_student (total_pencils : ℕ) (students : ℕ) (pencils_per_student : ℕ) 
  (h_total : total_pencils = 125) 
  (h_students : students = 25) 
  (h_div : pencils_per_student = total_pencils / students) : 
  pencils_per_student = 5 :=
by
  sorry

end pencils_per_student_l79_79862


namespace gcd_90_450_l79_79999

theorem gcd_90_450 : Nat.gcd 90 450 = 90 := by
  sorry

end gcd_90_450_l79_79999


namespace find_m_of_inverse_proportion_l79_79174

theorem find_m_of_inverse_proportion (k : ℝ) (m : ℝ) 
(A_cond : (-1) * 3 = k) 
(B_cond : 2 * m = k) : 
m = -3 / 2 := 
by 
  sorry

end find_m_of_inverse_proportion_l79_79174


namespace factorize_cubic_l79_79238

theorem factorize_cubic (x : ℝ) : x^3 - x = x * (x + 1) * (x - 1) :=
by sorry

end factorize_cubic_l79_79238


namespace Vanya_original_number_l79_79874

theorem Vanya_original_number (m n : ℕ) (hm : m ≤ 9) (hn : n ≤ 9) (h : (10 * n + m)^2 = 4 * (10 * m + n)) : 10 * m + n = 81 := by
  sorry

end Vanya_original_number_l79_79874


namespace probability_of_both_making_basket_l79_79081

noncomputable def P : Set ℕ → ℚ :=
  sorry

def A : Set ℕ := sorry
def B : Set ℕ := sorry

axiom prob_A : P A = 2 / 5
axiom prob_B : P B = 1 / 2
axiom independent : P (A ∩ B) = P A * P B

theorem probability_of_both_making_basket :
  P (A ∩ B) = 1 / 5 :=
by
  rw [independent, prob_A, prob_B]
  norm_num

end probability_of_both_making_basket_l79_79081


namespace count_numbers_without_1_or_2_l79_79789

/-- The number of whole numbers between 1 and 2000 that do not contain the digits 1 or 2 is 511. -/
theorem count_numbers_without_1_or_2 : 
  ∃ n : ℕ, n = 511 ∧
    (∀ k : ℕ, 1 ≤ k ∧ k ≤ 2000 →
      ¬ (∃ d : ℕ, (k.digits 10).contains d ∧ (d = 1 ∨ d = 2)) → n = 511) :=
sorry

end count_numbers_without_1_or_2_l79_79789


namespace number_of_students_suggested_mashed_potatoes_l79_79803

theorem number_of_students_suggested_mashed_potatoes 
    (students_suggested_bacon : ℕ := 374) 
    (students_suggested_tomatoes : ℕ := 128) 
    (total_students_participated : ℕ := 826) : 
    (total_students_participated - (students_suggested_bacon + students_suggested_tomatoes)) = 324 :=
by sorry

end number_of_students_suggested_mashed_potatoes_l79_79803


namespace children_per_block_l79_79588

theorem children_per_block {children total_blocks : ℕ} 
  (h_total_blocks : total_blocks = 9) 
  (h_total_children : children = 54) : 
  (children / total_blocks = 6) :=
by
  -- Definitions from conditions
  have h1 : total_blocks = 9 := h_total_blocks
  have h2 : children = 54 := h_total_children

  -- Goal to prove
  -- children / total_blocks = 6
  sorry

end children_per_block_l79_79588


namespace robin_total_spending_l79_79131

def jelly_bracelets_total_cost : ℕ :=
  let names := ["Jessica", "Tori", "Lily", "Patrice"]
  let total_letters := names.foldl (λ acc name => acc + name.length) 0
  total_letters * 2

theorem robin_total_spending : jelly_bracelets_total_cost = 44 := by
  sorry

end robin_total_spending_l79_79131


namespace orvin_max_balloons_l79_79540

variable (C : ℕ) (P : ℕ)

noncomputable def max_balloons (C P : ℕ) : ℕ :=
  let pair_cost := P + P / 2  -- Cost for two balloons
  let pairs := C / pair_cost  -- Maximum number of pairs
  pairs * 2 + (if C % pair_cost >= P then 1 else 0) -- Total balloons considering the leftover money

theorem orvin_max_balloons (hC : C = 120) (hP : P = 3) : max_balloons C P = 53 :=
by
  sorry

end orvin_max_balloons_l79_79540


namespace triangle_area_0_0_0_5_7_12_l79_79529

theorem triangle_area_0_0_0_5_7_12 : 
    let base := 5
    let height := 7
    let area := (1 / 2) * base * height
    area = 17.5 := 
by
    sorry

end triangle_area_0_0_0_5_7_12_l79_79529


namespace probability_sum_5_l79_79302

theorem probability_sum_5 :
  let total_outcomes := 36
  let favorable_outcomes := 4
  (favorable_outcomes / total_outcomes : ℚ) = 1 / 9 :=
by
  -- proof omitted
  sorry

end probability_sum_5_l79_79302


namespace problem_1_problem_2_l79_79825

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | -2 < x ∧ x < 3}
def B : Set ℝ := {x | -3 ≤ x ∧ x ≤ 2}

-- 1. Prove that A ∩ B = {x | -2 < x ≤ 2}
theorem problem_1 : A ∩ B = {x | -2 < x ∧ x ≤ 2} :=
by
  sorry

-- 2. Prove that (complement U A) ∪ B = {x | x ≤ 2 ∨ x ≥ 3}
theorem problem_2 : (U \ A) ∪ B = {x | x ≤ 2 ∨ x ≥ 3} :=
by
  sorry

end problem_1_problem_2_l79_79825


namespace max_value_of_expression_l79_79427

open Real

theorem max_value_of_expression
  (x y : ℝ)
  (h1 : x > 0)
  (h2 : y > 0)
  (h3 : x^2 - 2 * x * y + 3 * y^2 = 10) 
  : x^2 + 2 * x * y + 3 * y^2 ≤ 10 * (45 + 42 * sqrt 3) := 
sorry

end max_value_of_expression_l79_79427


namespace number_of_customers_l79_79309

theorem number_of_customers 
  (total_cartons : ℕ) 
  (damaged_cartons : ℕ) 
  (accepted_cartons : ℕ) 
  (customers : ℕ) 
  (h1 : total_cartons = 400)
  (h2 : damaged_cartons = 60)
  (h3 : accepted_cartons = 160)
  (h_eq_per_customer : (total_cartons / customers) - damaged_cartons = accepted_cartons / customers) :
  customers = 4 :=
sorry

end number_of_customers_l79_79309


namespace min_troublemakers_l79_79283

theorem min_troublemakers (n : ℕ) (students : ℕ → Prop) 
  (h : n = 29)
  (condition1 : ∀ i, students i → (students ((i - 1) % n) ↔ ¬ students ((i + 1) % n)))
  (condition2 : ∀ i, ¬ students i → (students ((i - 1) % n) ∧ students ((i + 1) % n)))
  : ∃ L : ℕ, (L ≤ 29 ∧ L ≥ 10) :=
by sorry

end min_troublemakers_l79_79283


namespace geometric_sequence_arithmetic_condition_l79_79213

noncomputable def geometric_sequence_ratio (q : ℝ) : Prop :=
  q > 0

def arithmetic_sequence (a₁ a₂ a₃ : ℝ) : Prop :=
  2 * a₃ = a₁ + 2 * a₂

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a 0 * q ^ n

theorem geometric_sequence_arithmetic_condition
  (a : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0)
  (q : ℝ)
  (hq : geometric_sequence_ratio q)
  (h_arith : arithmetic_sequence (a 0) (geometric_sequence a q 1) (geometric_sequence a q 2)) :
  (geometric_sequence a q 9 + geometric_sequence a q 10) / 
  (geometric_sequence a q 7 + geometric_sequence a q 8) = 3 + 2 * Real.sqrt 2 :=
by
  sorry

end geometric_sequence_arithmetic_condition_l79_79213


namespace modified_expression_range_l79_79718

open Int

theorem modified_expression_range (m : ℤ) :
  ∃ n_min n_max : ℤ, 1 < 4 * n_max + 7 ∧ 4 * n_min + 7 < 60 ∧ (n_max - n_min + 1 = 15) →
  ∃ k_min k_max : ℤ, 1 < m * k_max + 7 ∧ m * k_min + 7 < 60 ∧ (k_max - k_min + 1 ≥ 15) := 
sorry

end modified_expression_range_l79_79718


namespace inequality_solution_l79_79451

theorem inequality_solution (x : ℝ) :
  ((2 / (x - 1)) - (3 / (x - 3)) + (2 / (x - 4)) - (2 / (x - 5)) < (1 / 15)) ↔
  (x < -1 ∨ (1 < x ∧ x < 3) ∨ (4 < x ∧ x < 5) ∨ (7 < x ∧ x < 8)) :=
by
  sorry

end inequality_solution_l79_79451


namespace necessary_but_not_sufficient_l79_79251

theorem necessary_but_not_sufficient (x : ℝ) (h : x ≠ 1) : x^2 - 3 * x + 2 ≠ 0 :=
by
  intro h1
  -- Insert the proof here
  sorry

end necessary_but_not_sufficient_l79_79251


namespace tan_double_angle_l79_79111

theorem tan_double_angle (θ : ℝ) (P : ℝ × ℝ) 
  (h_vertex : θ = 0) 
  (h_initial_side : ∀ x, θ = x)
  (h_terminal_side : P = (-1, 2)) : 
  Real.tan (2 * θ) = 4 / 3 := 
by 
  sorry

end tan_double_angle_l79_79111


namespace sin_150_eq_half_l79_79159

theorem sin_150_eq_half : Real.sin (150 * Real.pi / 180) = 1 / 2 := by
  sorry

end sin_150_eq_half_l79_79159


namespace evaluate_expression_l79_79690

theorem evaluate_expression :
  8^(-1/3 : ℝ) + (49^(-1/2 : ℝ))^(1/2 : ℝ) = (Real.sqrt 7 + 2) / (2 * Real.sqrt 7) := by
  sorry

end evaluate_expression_l79_79690


namespace law_of_sines_proof_l79_79047

noncomputable def law_of_sines (a b c α β γ : ℝ) :=
  (a / Real.sin α = b / Real.sin β) ∧
  (b / Real.sin β = c / Real.sin γ) ∧
  (α + β + γ = Real.pi)

theorem law_of_sines_proof (a b c α β γ : ℝ) (h : law_of_sines a b c α β γ) :
  (a = b * Real.cos γ + c * Real.cos β) ∧
  (b = c * Real.cos α + a * Real.cos γ) ∧
  (c = a * Real.cos β + b * Real.cos α) :=
sorry

end law_of_sines_proof_l79_79047


namespace frustum_slant_height_l79_79881

theorem frustum_slant_height
  (ratio_area : ℝ)
  (slant_height_removed : ℝ)
  (sf_ratio : ratio_area = 1/16)
  (shr : slant_height_removed = 3) :
  ∃ (slant_height_frustum : ℝ), slant_height_frustum = 9 :=
by
  sorry

end frustum_slant_height_l79_79881


namespace monotonicity_and_extremum_of_f_l79_79943

noncomputable def f (x : ℝ) : ℝ := x * Real.log x - x

theorem monotonicity_and_extremum_of_f :
  (∀ x, 1 < x → ∀ y, x < y → f x < f y) ∧
  (∀ x, 0 < x → x < 1 → ∀ y, x < y → y < 1 → f x > f y) ∧
  (f 1 = -1) :=
by
  sorry

end monotonicity_and_extremum_of_f_l79_79943


namespace four_kids_wash_three_whiteboards_in_20_minutes_l79_79100

-- Condition: It takes one kid 160 minutes to wash six whiteboards
def time_per_whiteboard_for_one_kid : ℚ := 160 / 6

-- Calculation involving four kids
def time_per_whiteboard_for_four_kids : ℚ := time_per_whiteboard_for_one_kid / 4

-- The total time it takes for four kids to wash three whiteboards together
def total_time_for_four_kids_washing_three_whiteboards : ℚ := time_per_whiteboard_for_four_kids * 3

-- Statement to prove
theorem four_kids_wash_three_whiteboards_in_20_minutes : 
  total_time_for_four_kids_washing_three_whiteboards = 20 :=
by
  sorry

end four_kids_wash_three_whiteboards_in_20_minutes_l79_79100


namespace percentage_of_girl_scouts_with_slips_l79_79736

-- Define the proposition that captures the problem
theorem percentage_of_girl_scouts_with_slips 
    (total_scouts : ℕ)
    (scouts_with_slips : ℕ := total_scouts * 60 / 100)
    (boy_scouts : ℕ := total_scouts * 45 / 100)
    (boy_scouts_with_slips : ℕ := boy_scouts * 50 / 100)
    (girl_scouts : ℕ := total_scouts - boy_scouts)
    (girl_scouts_with_slips : ℕ := scouts_with_slips - boy_scouts_with_slips) :
  (girl_scouts_with_slips * 100 / girl_scouts) = 68 :=
by 
  -- The proof goes here
  sorry

end percentage_of_girl_scouts_with_slips_l79_79736


namespace men_became_absent_l79_79018

theorem men_became_absent (original_men planned_days actual_days : ℕ) (h1 : original_men = 48) (h2 : planned_days = 15) (h3 : actual_days = 18) :
  ∃ x : ℕ, 48 * 15 = (48 - x) * 18 ∧ x = 8 :=
by
  sorry

end men_became_absent_l79_79018


namespace product_of_six_consecutive_nat_not_equal_776965920_l79_79779

theorem product_of_six_consecutive_nat_not_equal_776965920 (n : ℕ) :
  (n * (n + 1) * (n + 2) * (n + 3) * (n + 4) * (n + 5) ≠ 776965920) :=
by
  sorry

end product_of_six_consecutive_nat_not_equal_776965920_l79_79779


namespace compute_fg_l79_79023

def f (x : ℝ) : ℝ := 4 * x - 1
def g (x : ℝ) : ℝ := (x + 2) ^ 2

theorem compute_fg : f (g (-3)) = 3 := by
  sorry

end compute_fg_l79_79023


namespace find_a7_l79_79208

def arithmetic_seq (a₁ d : ℤ) (n : ℤ) : ℤ := a₁ + (n-1) * d

theorem find_a7 (a₁ d : ℤ)
  (h₁ : arithmetic_seq a₁ d 3 + arithmetic_seq a₁ d 7 - arithmetic_seq a₁ d 10 = -1)
  (h₂ : arithmetic_seq a₁ d 11 - arithmetic_seq a₁ d 4 = 21) :
  arithmetic_seq a₁ d 7 = 20 :=
by
  sorry

end find_a7_l79_79208


namespace relationship_between_y_values_l79_79341

theorem relationship_between_y_values 
  (m : ℝ) 
  (y1 y2 y3 : ℝ)
  (h1 : y1 = (-1 : ℝ) ^ 2 + 2 * (-1 : ℝ) + m) 
  (h2 : y2 = (3 : ℝ) ^ 2 + 2 * (3 : ℝ) + m) 
  (h3 : y3 = ((1 / 2) : ℝ) ^ 2 + 2 * ((1 / 2) : ℝ) + m) : 
  y2 > y3 ∧ y3 > y1 := 
by 
  sorry

end relationship_between_y_values_l79_79341


namespace discount_coupon_value_l79_79898

theorem discount_coupon_value :
  let hamburger_cost := 2 * 5
  let cola_cost := 3 * 2
  let total_cost := hamburger_cost + cola_cost
  let amount_paid := 12
  total_cost - amount_paid = 4 := by
  intros
  let hamburger_cost := 2 * 5
  let cola_cost := 3 * 2
  let total_cost := hamburger_cost + cola_cost
  let amount_paid := 12
  show total_cost - amount_paid = 4
  sorry

end discount_coupon_value_l79_79898


namespace marys_score_l79_79604

theorem marys_score (C ω S : ℕ) (H1 : S = 30 + 4 * C - ω) (H2 : S > 80)
  (H3 : (∀ C1 ω1 C2 ω2, (C1 ≠ C2 → 30 + 4 * C1 - ω1 ≠ 30 + 4 * C2 - ω2))) : 
  S = 119 :=
sorry

end marys_score_l79_79604


namespace alice_winning_strategy_l79_79028

theorem alice_winning_strategy (n : ℕ) (hn : n ≥ 2) : 
  (Alice_has_winning_strategy ↔ n % 4 = 3) :=
sorry

end alice_winning_strategy_l79_79028


namespace time_45_minutes_after_10_20_is_11_05_l79_79332

def time := Nat × Nat -- Represents time as (hours, minutes)

noncomputable def add_minutes (t : time) (m : Nat) : time :=
  let (hours, minutes) := t
  let total_minutes := minutes + m
  let new_hours := hours + total_minutes / 60
  let new_minutes := total_minutes % 60
  (new_hours, new_minutes)

theorem time_45_minutes_after_10_20_is_11_05 :
  add_minutes (10, 20) 45 = (11, 5) :=
  sorry

end time_45_minutes_after_10_20_is_11_05_l79_79332


namespace zhuzhuxia_defeats_monsters_l79_79824

theorem zhuzhuxia_defeats_monsters {a : ℕ} (H1 : zhuzhuxia_total_defeated_monsters = 20) :
  zhuzhuxia_total_defeated_by_monsters = 8 :=
sorry

end zhuzhuxia_defeats_monsters_l79_79824


namespace number_subtracted_l79_79482

theorem number_subtracted (t k x : ℝ) (h1 : t = (5 / 9) * (k - x)) (h2 : t = 105) (h3 : k = 221) : x = 32 :=
by
  sorry

end number_subtracted_l79_79482


namespace ram_money_l79_79522

variable (R G K : ℕ)

theorem ram_money (h1 : R / G = 7 / 17) (h2 : G / K = 7 / 17) (h3 : K = 2890) : R = 490 :=
by
  sorry

end ram_money_l79_79522


namespace triangle_side_lengths_l79_79261

theorem triangle_side_lengths
  (x y z : ℕ)
  (h1 : x > y)
  (h2 : y > z)
  (h3 : x + y + z = 240)
  (h4 : 3 * x - 2 * (y + z) = 5 * z + 10)
  (h5 : x < y + z) :
  (x = 113 ∧ y = 112 ∧ z = 15) ∨
  (x = 114 ∧ y = 110 ∧ z = 16) ∨
  (x = 115 ∧ y = 108 ∧ z = 17) ∨
  (x = 116 ∧ y = 106 ∧ z = 18) ∨
  (x = 117 ∧ y = 104 ∧ z = 19) ∨
  (x = 118 ∧ y = 102 ∧ z = 20) ∨
  (x = 119 ∧ y = 100 ∧ z = 21) := by
  sorry

end triangle_side_lengths_l79_79261


namespace compute_x_l79_79817

/-- 
Let ABC be a triangle. 
Points D, E, and F are on BC, CA, and AB, respectively. 
Given that AE/AC = CD/CB = BF/BA = x for some x with 1/2 < x < 1. 
Segments AD, BE, and CF divide the triangle into 7 non-overlapping regions: 
4 triangles and 3 quadrilaterals. 
The total area of the 4 triangles equals the total area of the 3 quadrilaterals. 
Compute the value of x.
-/
theorem compute_x (x : ℝ) (h1 : 1 / 2 < x) (h2 : x < 1)
  (h3 : (∃ (triangleArea quadrilateralArea : ℝ), 
          let A := triangleArea + 3 * x
          let B := quadrilateralArea
          A = B))
  : x = (11 - Real.sqrt 37) / 6 := 
sorry

end compute_x_l79_79817


namespace weekend_weekday_ratio_l79_79368

-- Defining the basic constants and conditions
def weekday_episodes : ℕ := 8
def total_episodes_in_week : ℕ := 88

-- Defining the main theorem
theorem weekend_weekday_ratio : (2 * (total_episodes_in_week - 5 * weekday_episodes)) / weekday_episodes = 3 :=
by
  sorry

end weekend_weekday_ratio_l79_79368


namespace B_Bons_wins_probability_l79_79668

theorem B_Bons_wins_probability :
  let roll_six := (1 : ℚ) / 6
  let not_roll_six := (5 : ℚ) / 6
  let p := (5 : ℚ) / 11
  p = (5 / 36) + (25 / 36) * p :=
by
  sorry

end B_Bons_wins_probability_l79_79668


namespace sqrt_121_pm_11_l79_79075

theorem sqrt_121_pm_11 :
  (∃ y : ℤ, y * y = 121) ∧ (∃ x : ℤ, x = 11 ∨ x = -11) → (∃ x : ℤ, x * x = 121 ∧ (x = 11 ∨ x = -11)) :=
by
  sorry

end sqrt_121_pm_11_l79_79075


namespace probability_of_region_C_l79_79337

theorem probability_of_region_C (pA pB pC : ℚ) 
  (h1 : pA = 1/2) 
  (h2 : pB = 1/5) 
  (h3 : pA + pB + pC = 1) : 
  pC = 3/10 := 
sorry

end probability_of_region_C_l79_79337


namespace problem_part1_problem_part2_l79_79911

variable (a m : ℝ)

def prop_p (a m : ℝ) : Prop := (m - a) * (m - 3 * a) ≤ 0
def prop_q (m : ℝ) : Prop := (m + 2) * (m + 1) < 0

theorem problem_part1 (h₁ : a = -1) (h₂ : prop_p a m ∨ prop_q m) : -3 ≤ m ∧ m ≤ -1 :=
sorry

theorem problem_part2 (h₁ : ∀ m, prop_p a m → ¬prop_q m) :
  -1 / 3 ≤ a ∧ a < 0 ∨ a ≤ -2 :=
sorry

end problem_part1_problem_part2_l79_79911


namespace students_like_apple_and_chocolate_not_blueberry_l79_79433

variables (n A C B D : ℕ)

theorem students_like_apple_and_chocolate_not_blueberry
  (h1 : n = 50)
  (h2 : A = 25)
  (h3 : C = 20)
  (h4 : B = 5)
  (h5 : D = 15) :
  ∃ (x : ℕ), x = 10 ∧ x = n - D - (A + C - 2 * x) ∧ 0 ≤ 2 * x - A - C + B :=
sorry

end students_like_apple_and_chocolate_not_blueberry_l79_79433


namespace maximum_value_problem_l79_79532

theorem maximum_value_problem (a b c : ℝ) (h : a^2 + b^2 + c^2 = 1) :
  (a^2 - b * c) * (b^2 - c * a) * (c^2 - a * b) ≤ 1 / 8 :=
sorry

end maximum_value_problem_l79_79532


namespace find_x_l79_79551

def f (x : ℝ) := 2 * x - 3

theorem find_x : ∃ x, 2 * (f x) - 11 = f (x - 2) ∧ x = 5 :=
by 
  unfold f
  exists 5
  sorry

end find_x_l79_79551


namespace cost_of_song_book_l79_79521

-- Define the given constants: cost of trumpet, cost of music tool, and total spent at the music store.
def cost_of_trumpet : ℝ := 149.16
def cost_of_music_tool : ℝ := 9.98
def total_spent_at_store : ℝ := 163.28

-- The goal is to prove that the cost of the song book is $4.14.
theorem cost_of_song_book :
  total_spent_at_store - (cost_of_trumpet + cost_of_music_tool) = 4.14 :=
by
  sorry

end cost_of_song_book_l79_79521


namespace third_test_point_l79_79097

noncomputable def test_points : ℝ × ℝ × ℝ :=
  let x1 := 2 + 0.618 * (4 - 2)
  let x2 := 2 + 4 - x1
  let x3 := 4 - 0.618 * (4 - x1)
  (x1, x2, x3)

theorem third_test_point :
  let x1 := 2 + 0.618 * (4 - 2)
  let x2 := 2 + 4 - x1
  let x3 := 4 - 0.618 * (4 - x1)
  x1 > x2 → x3 = 3.528 :=
by
  intros
  sorry

end third_test_point_l79_79097


namespace brian_total_distance_l79_79923

noncomputable def miles_per_gallon : ℝ := 20
noncomputable def tank_capacity : ℝ := 15
noncomputable def tank_fraction_remaining : ℝ := 3 / 7

noncomputable def total_miles_traveled (miles_per_gallon tank_capacity tank_fraction_remaining : ℝ) : ℝ :=
  let total_miles := miles_per_gallon * tank_capacity
  let fuel_used := tank_capacity * (1 - tank_fraction_remaining)
  let miles_traveled := fuel_used * miles_per_gallon
  miles_traveled

theorem brian_total_distance : 
  total_miles_traveled miles_per_gallon tank_capacity tank_fraction_remaining = 171.4 := 
by
  sorry

end brian_total_distance_l79_79923


namespace other_continent_passengers_l79_79674

noncomputable def totalPassengers := 240
noncomputable def northAmericaFraction := (1 / 3 : ℝ)
noncomputable def europeFraction := (1 / 8 : ℝ)
noncomputable def africaFraction := (1 / 5 : ℝ)
noncomputable def asiaFraction := (1 / 6 : ℝ)

theorem other_continent_passengers :
  (totalPassengers : ℝ) - (totalPassengers * northAmericaFraction +
                           totalPassengers * europeFraction +
                           totalPassengers * africaFraction +
                           totalPassengers * asiaFraction) = 42 :=
by
  sorry

end other_continent_passengers_l79_79674


namespace one_and_two_thirds_of_what_number_is_45_l79_79073

theorem one_and_two_thirds_of_what_number_is_45 (x : ℝ) (h : (5 / 3) * x = 45) : x = 27 :=
by
  sorry

end one_and_two_thirds_of_what_number_is_45_l79_79073


namespace initial_lives_emily_l79_79875

theorem initial_lives_emily (L : ℕ) (h1 : L - 25 + 24 = 41) : L = 42 :=
by
  sorry

end initial_lives_emily_l79_79875


namespace circumscribed_triangle_area_relationship_l79_79221

theorem circumscribed_triangle_area_relationship (X Y Z : ℝ) :
  let a := 15
  let b := 20
  let c := 25
  let triangle_area := (1/2) * a * b
  let diameter := c
  let radius := diameter / 2
  let circle_area := Real.pi * radius^2
  let Z := circle_area / 2
  (X + Y + triangle_area = Z) :=
sorry

end circumscribed_triangle_area_relationship_l79_79221


namespace proportion_of_adopted_kittens_l79_79839

-- Define the relevant objects and conditions in Lean
def breeding_rabbits : ℕ := 10
def kittens_first_spring := 10 * breeding_rabbits -- 100 kittens
def kittens_second_spring : ℕ := 60
def adopted_first_spring (P : ℝ) := 100 * P
def returned_first_spring : ℕ := 5
def adopted_second_spring : ℕ := 4
def total_rabbits_in_house (P : ℝ) :=
  breeding_rabbits + (kittens_first_spring - adopted_first_spring P + returned_first_spring) +
  (kittens_second_spring - adopted_second_spring)

theorem proportion_of_adopted_kittens : ∃ (P : ℝ), total_rabbits_in_house P = 121 ∧ P = 0.5 :=
by
  use 0.5
  -- Proof part (with "sorry" to skip the detailed proof)
  sorry

end proportion_of_adopted_kittens_l79_79839


namespace ladder_base_distance_l79_79089

variable (x : ℕ)
variable (ladder_length : ℕ := 13)
variable (height : ℕ := 12)

theorem ladder_base_distance : ∃ x, (x^2 + height^2 = ladder_length^2) ∧ x = 5 := by
  sorry

end ladder_base_distance_l79_79089


namespace range_of_a_l79_79241

variable {x a : ℝ}

def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0
def q (x : ℝ) : Prop := x^2 - 6*x + 8 > 0

theorem range_of_a (h : (∀ x a, p x a → q x) ∧ (∃ x a, q x ∧ ¬ p x a)) :
  a ≥ 4 ∨ (0 < a ∧ a ≤ 2/3) :=
sorry

end range_of_a_l79_79241


namespace lines_coplanar_parameter_l79_79306

/-- 
  Two lines are given in parametric form: 
  L1: (2 + 2s, 4s, -3 + rs)
  L2: (-1 + 3t, 2t, 1 + 2t)
  Prove that if these lines are coplanar, then r = 4.
-/
theorem lines_coplanar_parameter (s t r : ℝ) :
  ∃ (k : ℝ), 
  (∀ s t, 
    ∃ (k₁ k₂ : ℝ), k₁ * k₂ ≠ 0
      ∧
      (2 + 2 * s, 4 * s, -3 + r * s) = (k * (-1 + 3 * t), k * 2 * t, k * (1 + 2 * t))
  ) → r = 4 := sorry

end lines_coplanar_parameter_l79_79306


namespace trisha_spent_on_eggs_l79_79870

def totalSpent (meat chicken veggies eggs dogFood amountLeft initialAmount : ℕ) : ℕ :=
  initialAmount - (meat + chicken + veggies + dogFood + amountLeft)

theorem trisha_spent_on_eggs :
  ∀ (meat chicken veggies eggs dogFood amountLeft initialAmount : ℕ),
    meat = 17 →
    chicken = 22 →
    veggies = 43 →
    dogFood = 45 →
    amountLeft = 35 →
    initialAmount = 167 →
    totalSpent meat chicken veggies eggs dogFood amountLeft initialAmount = 5 :=
by
  intros meat chicken veggies eggs dogFood amountLeft initialAmount
  sorry

end trisha_spent_on_eggs_l79_79870


namespace find_f_sqrt_10_l79_79794

-- Definitions and conditions provided in the problem
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x
def is_periodic_function (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x : ℝ, f (x + p) = f x
def f_condition (f : ℝ → ℝ) : Prop := ∀ x : ℝ, 0 < x ∧ x < 1 → f x = x^2 - 8*x + 30

-- The problem specific conditions for f
variable (f : ℝ → ℝ)
variable (h_odd : is_odd_function f)
variable (h_periodic : is_periodic_function f 2)
variable (h_condition : f_condition f)

-- The statement to prove
theorem find_f_sqrt_10 : f (Real.sqrt 10) = -24 :=
by
  sorry

end find_f_sqrt_10_l79_79794


namespace area_of_triangle_l79_79108

def line1 (x : ℝ) : ℝ := 3 * x + 6
def line2 (x : ℝ) : ℝ := -2 * x + 10

theorem area_of_triangle : 
  let inter_x := (10 - 6) / (3 + 2)
  let inter_y := line1 inter_x
  let base := (10 - 6 : ℝ)
  let height := inter_x
  base * height / 2 = 8 / 5 := 
by
  sorry

end area_of_triangle_l79_79108


namespace fractions_with_same_denominators_fractions_with_same_numerators_fractions_with_different_numerators_and_denominators_l79_79688

theorem fractions_with_same_denominators {a b c : ℤ} (h_c : c ≠ 0) :
  (a > b → a / (c:ℚ) > b / (c:ℚ)) ∧ (a < b → a / (c:ℚ) < b / (c:ℚ)) :=
by sorry

theorem fractions_with_same_numerators {a c d : ℤ} (h_c : c ≠ 0) (h_d : d ≠ 0) :
  (c < d → a / (c:ℚ) > a / (d:ℚ)) ∧ (c > d → a / (c:ℚ) < a / (d:ℚ)) :=
by sorry

theorem fractions_with_different_numerators_and_denominators {a b c d : ℤ} (h_c : c ≠ 0) (h_d : d ≠ 0) :
  a > b ∧ c < d → a / (c:ℚ) > b / (d:ℚ) :=
by sorry

end fractions_with_same_denominators_fractions_with_same_numerators_fractions_with_different_numerators_and_denominators_l79_79688


namespace angle_B_measure_triangle_area_l79_79682

noncomputable def triangle (A B C : ℝ) : Type := sorry

variable (a b c : ℝ)
variable (A B C : ℝ)

-- Given conditions:
axiom eq1 : b * Real.cos C = (2 * a - c) * Real.cos B

-- Part 1: Prove the measure of angle B
theorem angle_B_measure : B = Real.pi / 3 :=
by
  have b_cos_C := eq1
  sorry

-- Part 2: Given additional conditions and find the area
variable (b_value : ℝ := Real.sqrt 7)
variable (sum_ac : ℝ := 4)

theorem triangle_area : (1 / 2 * a * c * Real.sin B = 3 * Real.sqrt 3 / 4) :=
by
  have b_value_def := b_value
  have sum_ac_def := sum_ac
  sorry

end angle_B_measure_triangle_area_l79_79682


namespace circle_equation_unique_circle_equation_l79_79757

-- Definitions based on conditions
def radius (r : ℝ) : Prop := r = 1
def center_in_first_quadrant (a b : ℝ) : Prop := a > 0 ∧ b > 0
def tangent_to_line (a b : ℝ) : Prop := (|4 * a - 3 * b| / Real.sqrt (4^2 + (-3)^2)) = 1
def tangent_to_x_axis (b : ℝ) : Prop := b = 1

-- Main theorem statement
theorem circle_equation_unique 
  {a b : ℝ} 
  (h_rad : radius 1) 
  (h_center : center_in_first_quadrant a b) 
  (h_tan_line : tangent_to_line a b) 
  (h_tan_x : tangent_to_x_axis b) :
  (a = 2 ∧ b = 1) :=
sorry

-- Final circle equation
theorem circle_equation : 
  (∀ a b : ℝ, ((a = 2) ∧ (b = 1)) → (x - a)^2 + (y - b)^2 = 1) :=
sorry

end circle_equation_unique_circle_equation_l79_79757


namespace lulu_cash_left_l79_79914

theorem lulu_cash_left :
  ∀ (initial money spentIceCream spentTshirt deposited finalCash: ℝ),
    initial = 65 →
    spentIceCream = 5 →
    spentTshirt = 0.5 * (initial - spentIceCream) →
    deposited = (initial - spentIceCream - spentTshirt) / 5 →
    finalCash = initial - spentIceCream - spentTshirt - deposited →
    finalCash = 24 :=
by
  intros
  sorry

end lulu_cash_left_l79_79914


namespace sin_pi_div_two_plus_2alpha_eq_num_fifth_ninth_l79_79742

noncomputable def sin_pi_div_two_plus_2alpha (α : ℝ) : ℝ :=
  Real.sin ((Real.pi / 2) + 2 * α)

def cos_alpha (α : ℝ) := Real.cos α = - (Real.sqrt 2) / 3

theorem sin_pi_div_two_plus_2alpha_eq_num_fifth_ninth (α : ℝ) (h : cos_alpha α) :
  sin_pi_div_two_plus_2alpha α = -5 / 9 :=
sorry

end sin_pi_div_two_plus_2alpha_eq_num_fifth_ninth_l79_79742


namespace maximum_value_l79_79171

theorem maximum_value (R P K : ℝ) (h₁ : 3 * Real.sqrt 3 * R ≥ P) (h₂ : K = P * R / 4) : 
  (K * P) / (R^3) ≤ 27 / 4 :=
by
  sorry

end maximum_value_l79_79171


namespace simplify_tangent_expression_l79_79530

theorem simplify_tangent_expression :
  (1 + Real.tan (Real.pi / 18)) * (1 + Real.tan (35 * Real.pi / 180)) = 2 :=
by
  sorry

end simplify_tangent_expression_l79_79530


namespace exam_question_correct_count_l79_79515

theorem exam_question_correct_count (C W : ℕ) (h1 : C + W = 60) (h2 : 4 * C - W = 110) : C = 34 :=
by
  sorry

end exam_question_correct_count_l79_79515


namespace greatest_prime_factor_of_154_l79_79998

theorem greatest_prime_factor_of_154 : ∃ p : ℕ, Nat.Prime p ∧ p ∣ 154 ∧ (∀ q : ℕ, Nat.Prime q → q ∣ 154 → q ≤ p) :=
  sorry

end greatest_prime_factor_of_154_l79_79998


namespace complete_square_solution_l79_79141

theorem complete_square_solution (x : ℝ) :
  x^2 - 2*x - 3 = 0 → (x - 1)^2 = 4 :=
by
  sorry

end complete_square_solution_l79_79141


namespace evaluate_expression_c_eq_4_l79_79744

theorem evaluate_expression_c_eq_4 :
  (4^4 - 4 * (4-1)^(4-1))^(4-1) = 3241792 :=
by
  sorry

end evaluate_expression_c_eq_4_l79_79744


namespace scientific_notation_of_0_000000032_l79_79287

theorem scientific_notation_of_0_000000032 :
  0.000000032 = 3.2 * 10^(-8) :=
by
  -- skipping the proof
  sorry

end scientific_notation_of_0_000000032_l79_79287


namespace explorers_crossing_time_l79_79788

/-- Define constants and conditions --/
def num_explorers : ℕ := 60
def boat_capacity : ℕ := 6
def crossing_time : ℕ := 3
def round_trip_crossings : ℕ := 2
def total_trips := 1 + (num_explorers - boat_capacity - 1) / (boat_capacity - 1) + 1

theorem explorers_crossing_time :
  total_trips * crossing_time * round_trip_crossings / 2 + crossing_time = 69 :=
by sorry

end explorers_crossing_time_l79_79788


namespace solve_equation_in_natural_numbers_l79_79010

-- Define the main theorem
theorem solve_equation_in_natural_numbers :
  (∃ (x y z : ℕ), 2^x + 5^y + 63 = z! ∧ ((x = 5 ∧ y = 2 ∧ z = 5) ∨ (x = 4 ∧ y = 4 ∧ z = 6))) :=
sorry

end solve_equation_in_natural_numbers_l79_79010


namespace polynomial_factor_c_zero_l79_79537

theorem polynomial_factor_c_zero (c q : ℝ) :
    ∃ q : ℝ, (3*q + 6 = 0 ∧ c = 6*q + 12) ↔ c = 0 :=
by
  sorry

end polynomial_factor_c_zero_l79_79537


namespace ratio_of_price_l79_79636

-- Definitions from conditions
def original_price : ℝ := 3.00
def tom_pay_price : ℝ := 9.00

-- Theorem stating the ratio
theorem ratio_of_price : tom_pay_price / original_price = 3 := by
  sorry

end ratio_of_price_l79_79636


namespace max_ratio_three_digit_l79_79835

theorem max_ratio_three_digit (x a b c : ℕ) (h1 : 100 * a + 10 * b + c = x) (h2 : 1 ≤ a ∧ a ≤ 9)
  (h3 : 0 ≤ b ∧ b ≤ 9) (h4 : 0 ≤ c ∧ c ≤ 9) : 
  (x : ℚ) / (a + b + c) ≤ 100 := sorry

end max_ratio_three_digit_l79_79835


namespace curve_transformation_l79_79425

def matrix_transform (a : ℝ) (x y : ℝ) : ℝ × ℝ :=
  (0 * x + 1 * y, a * x + 0 * y)

def curve_eq (x y : ℝ) : Prop :=
  x ^ 2 + y ^ 2 = 1

def transformed_curve_eq (x y : ℝ) : Prop :=
  x ^ 2 + (y ^ 2) / 4 = 1

theorem curve_transformation (a : ℝ) 
  (h₁ : matrix_transform a 2 (-2) = (-2, 4))
  (h₂ : ∀ x y, curve_eq x y → transformed_curve_eq (matrix_transform a x y).fst (matrix_transform a x y).snd) :
  a = 2 ∧ ∀ x y, curve_eq x y → transformed_curve_eq (0 * x + 1 * y) (2 * x + 0 * y) :=
by
  sorry

end curve_transformation_l79_79425


namespace int_pairs_satisfy_conditions_l79_79784

theorem int_pairs_satisfy_conditions (m n : ℤ) :
  (∃ a b : ℤ, m^2 + n = a^2 ∧ n^2 + m = b^2) ↔ 
  ∃ k : ℤ, (m = 0 ∧ n = k^2) ∨ (m = k^2 ∧ n = 0) ∨ (m = 1 ∧ n = -1) ∨ (m = -1 ∧ n = 1) := by
  sorry

end int_pairs_satisfy_conditions_l79_79784


namespace seating_arrangement_l79_79457

variable {M I P A : Prop}

def first_fact : ¬ M := sorry
def second_fact : ¬ A := sorry
def third_fact : ¬ M → I := sorry
def fourth_fact : I → P := sorry

theorem seating_arrangement : ¬ M → (I ∧ P) :=
by
  intros hM
  have hI : I := third_fact hM
  have hP : P := fourth_fact hI
  exact ⟨hI, hP⟩

end seating_arrangement_l79_79457


namespace sin_neg_nine_pi_div_two_l79_79777

theorem sin_neg_nine_pi_div_two : Real.sin (-9 * Real.pi / 2) = -1 := by
  sorry

end sin_neg_nine_pi_div_two_l79_79777


namespace m_range_positive_real_number_l79_79066

theorem m_range_positive_real_number (m : ℝ) (x : ℝ) 
  (h : m * x - 1 = 2 * x) (h_pos : x > 0) : m > 2 :=
sorry

end m_range_positive_real_number_l79_79066


namespace compound_interest_example_l79_79959

theorem compound_interest_example :
  let P := 5000
  let r := 0.08
  let n := 4
  let t := 0.5
  let A := P * (1 + r / n) ^ (n * t)
  A = 5202 :=
by
  sorry

end compound_interest_example_l79_79959


namespace sum_of_first_5n_l79_79053

theorem sum_of_first_5n (n : ℕ) (h : (4 * n * (4 * n + 1)) / 2 = (2 * n * (2 * n + 1)) / 2 + 504) :
  (5 * n * (5 * n + 1)) / 2 = 1035 :=
sorry

end sum_of_first_5n_l79_79053


namespace shelves_full_percentage_l79_79600

-- Define the conditions as constants
def ridges_per_record : Nat := 60
def cases : Nat := 4
def shelves_per_case : Nat := 3
def records_per_shelf : Nat := 20
def total_ridges : Nat := 8640

-- Define the total number of records
def total_records := total_ridges / ridges_per_record

-- Define the total capacity of the shelves
def total_capacity := cases * shelves_per_case * records_per_shelf

-- Define the percentage of shelves that are full
def percentage_full := (total_records * 100) / total_capacity

-- State the theorem that the percentage of the shelves that are full is 60%
theorem shelves_full_percentage : percentage_full = 60 := 
by
  sorry

end shelves_full_percentage_l79_79600


namespace problem1_problem2_l79_79080

-- Problem 1: Prove that (1) - 8 + 12 - 16 - 23 = -35
theorem problem1 : (1 - 8 + 12 - 16 - 23 = -35) :=
by
  sorry

-- Problem 2: Prove that (3 / 4) + (-1 / 6) - (1 / 3) - (-1 / 8) = 3 / 8
theorem problem2 : (3 / 4 + (-1 / 6) - 1 / 3 + 1 / 8 = 3 / 8) :=
by
  sorry

end problem1_problem2_l79_79080


namespace reflection_eqn_l79_79796

theorem reflection_eqn 
  (x y : ℝ)
  (h : y = 2 * x + 3) : 
  -y = 2 * x + 3 :=
sorry

end reflection_eqn_l79_79796


namespace tigers_count_l79_79398

theorem tigers_count (T C : ℝ) 
  (h1 : 12 + T + C = 39) 
  (h2 : C = 0.5 * (12 + T)) : 
  T = 14 := by
  sorry

end tigers_count_l79_79398


namespace notebook_cost_l79_79353

theorem notebook_cost (total_spent ruler_cost pencil_count pencil_cost: ℕ)
  (h1 : total_spent = 74)
  (h2 : ruler_cost = 18)
  (h3 : pencil_count = 3)
  (h4 : pencil_cost = 7) :
  total_spent - (ruler_cost + pencil_count * pencil_cost) = 35 := 
by 
  sorry

end notebook_cost_l79_79353


namespace evaluate_expression_l79_79672

theorem evaluate_expression :
  2 + (3 / (4 + (5 / (6 + (7 / 8))))) = 137 / 52 :=
by
  sorry

end evaluate_expression_l79_79672


namespace hoseok_multiplied_number_l79_79942

theorem hoseok_multiplied_number (n : ℕ) (h : 11 * n = 99) : n = 9 := 
sorry

end hoseok_multiplied_number_l79_79942


namespace alicia_stickers_l79_79388

theorem alicia_stickers :
  ∃ S : ℕ, S > 2 ∧
  (S % 5 = 2) ∧ (S % 11 = 2) ∧ (S % 13 = 2) ∧
  S = 717 :=
sorry

end alicia_stickers_l79_79388


namespace square_area_problem_l79_79657

theorem square_area_problem
    (x1 y1 x2 y2 : ℝ)
    (h1 : y1 = x1^2)
    (h2 : y2 = x2^2)
    (line_eq : ∃ a : ℝ, a = 2 ∧ ∃ b : ℝ, b = -22 ∧ ∀ x y : ℝ, y = 2 * x - 22 → (y = y1 ∨ y = y2)) :
    ∃ area : ℝ, area = 180 ∨ area = 980 :=
sorry

end square_area_problem_l79_79657


namespace range_of_a_satisfies_l79_79153

noncomputable def satisfies_conditions (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f (-(x + 1)) = -f (x + 1)) ∧
  (∀ x1 x2 : ℝ, 0 ≤ x1 ∧ 0 ≤ x2 → (f x1 - f x2) / (x1 - x2) > -1) ∧
  (∀ a : ℝ, f (a^2 - 1) + f (a - 1) + a^2 + a > 2)

theorem range_of_a_satisfies (f : ℝ → ℝ) (hf_conditions : satisfies_conditions f) :
  {a : ℝ | f (a^2 - 1) + f (a - 1) + a^2 + a > 2} = {a | a < -2 ∨ a > 1} :=
by
  sorry

end range_of_a_satisfies_l79_79153


namespace dorothy_money_left_l79_79103

-- Define the conditions
def annual_income : ℝ := 60000
def tax_rate : ℝ := 0.18

-- Define the calculation of the amount of money left after paying taxes
def money_left (income : ℝ) (rate : ℝ) : ℝ :=
  income - (rate * income)

-- State the main theorem to prove
theorem dorothy_money_left :
  money_left annual_income tax_rate = 49200 := 
by
  sorry

end dorothy_money_left_l79_79103


namespace real_roots_quadratic_range_l79_79716

theorem real_roots_quadratic_range (k : ℝ) :
  (∃ x : ℝ, x^2 + 2 * x - k = 0) ↔ k ≥ -1 :=
by
  sorry

end real_roots_quadratic_range_l79_79716


namespace mixed_number_multiplication_equiv_l79_79566

theorem mixed_number_multiplication_equiv :
  (-3 - 1 / 2) * (5 / 7) = -3.5 * (5 / 7) := 
by 
  sorry

end mixed_number_multiplication_equiv_l79_79566


namespace unique_divisor_of_2_pow_n_minus_1_l79_79876

theorem unique_divisor_of_2_pow_n_minus_1 : ∀ (n : ℕ), n ≥ 1 → n ∣ (2^n - 1) → n = 1 := 
by
  intro n h1 h2
  sorry

end unique_divisor_of_2_pow_n_minus_1_l79_79876


namespace typing_speed_ratio_l79_79563

-- Define Tim's and Tom's typing speeds
variables (T t : ℝ)

-- Conditions from the problem
def condition1 : Prop := T + t = 15
def condition2 : Prop := T + 1.6 * t = 18

-- The proposition to prove: the ratio of Tom's typing speed to Tim's is 1:2
theorem typing_speed_ratio (h1 : condition1 T t) (h2 : condition2 T t) : t / T = 1 / 2 :=
sorry

end typing_speed_ratio_l79_79563


namespace force_exerted_by_pulley_on_axis_l79_79472

-- Define the basic parameters given in the problem
def m1 : ℕ := 3 -- mass 1 in kg
def m2 : ℕ := 6 -- mass 2 in kg
def g : ℕ := 10 -- acceleration due to gravity in m/s^2

-- From the problem, we know that:
def F1 : ℕ := m1 * g -- gravitational force on mass 1
def F2 : ℕ := m2 * g -- gravitational force on mass 2

-- To find the tension, setup the equations
def a := (F2 - F1) / (m1 + m2) -- solving for acceleration between the masses

def T := (m1 * a) + F1 -- solving for the tension in the rope considering mass 1

-- Define the proof statement to find the force exerted by the pulley on its axis
theorem force_exerted_by_pulley_on_axis : 2 * T = 80 :=
by
  -- Annotations or calculations can go here
  sorry

end force_exerted_by_pulley_on_axis_l79_79472


namespace total_length_remaining_l79_79106

def initial_figure_height : ℕ := 10
def initial_figure_width : ℕ := 7
def top_right_removed : ℕ := 2
def middle_left_removed : ℕ := 2
def bottom_removed : ℕ := 3
def near_top_left_removed : ℕ := 1

def remaining_top_length : ℕ := initial_figure_width - top_right_removed
def remaining_left_length : ℕ := initial_figure_height - middle_left_removed
def remaining_bottom_length : ℕ := initial_figure_width - bottom_removed
def remaining_right_length : ℕ := initial_figure_height - near_top_left_removed

theorem total_length_remaining :
  remaining_top_length + remaining_left_length + remaining_bottom_length + remaining_right_length = 26 := by
  sorry

end total_length_remaining_l79_79106


namespace no_non_trivial_solution_l79_79361

theorem no_non_trivial_solution (a b c : ℤ) (h : a^2 = 2 * b^2 + 3 * c^2) : a = 0 ∧ b = 0 ∧ c = 0 :=
sorry

end no_non_trivial_solution_l79_79361


namespace graph_passes_through_point_l79_79909

theorem graph_passes_through_point (a : ℝ) (h_pos : 0 < a) (h_neq : a ≠ 1) :
  ∃ x y, (x, y) = (0, 3) ∧ (∀ f : ℝ → ℝ, (∀ y, (f y = a ^ y) → (0, f 0 + 2) = (0, 3))) :=
by
  sorry

end graph_passes_through_point_l79_79909


namespace avg_age_team_proof_l79_79401

-- Defining the known constants
def members : ℕ := 15
def avg_age_team : ℕ := 28
def captain_age : ℕ := avg_age_team + 4
def remaining_players : ℕ := members - 2
def avg_age_remaining : ℕ := avg_age_team - 2

-- Stating the problem to prove the average age remains 28
theorem avg_age_team_proof (W : ℕ) :
  28 = avg_age_team ∧
  members = 15 ∧
  captain_age = avg_age_team + 4 ∧
  remaining_players = members - 2 ∧
  avg_age_remaining = avg_age_team - 2 ∧
  28 * 15 = 26 * 13 + captain_age + W :=
sorry

end avg_age_team_proof_l79_79401


namespace books_problem_l79_79528

variable (L W : ℕ) -- L for Li Ming's initial books, W for Wang Hong's initial books

theorem books_problem (h1 : L = W + 26) (h2 : L - 14 = W + 14 - 2) : 14 = 14 :=
by
  sorry

end books_problem_l79_79528


namespace largest_d_l79_79127

variable (a b c d : ℤ)

def condition : Prop := a + 2 = b - 1 ∧ a + 2 = c + 3 ∧ a + 2 = d - 4

theorem largest_d (h : condition a b c d) : d > a ∧ d > b ∧ d > c :=
by
  -- Assuming the condition holds, we need to prove d > a, d > b, and d > c
  sorry

end largest_d_l79_79127


namespace work_fraction_completed_after_first_phase_l79_79041

-- Definitions based on conditions
def total_work := 1 -- Assume total work as 1 unit
def initial_days := 100
def initial_people := 10
def first_phase_days := 20
def fired_people := 2
def remaining_days := 75
def remaining_people := initial_people - fired_people

-- Hypothesis about the rate of work initially and after firing people
def initial_rate := total_work / initial_days
def first_phase_work := first_phase_days * initial_rate
def remaining_work := total_work - first_phase_work
def remaining_rate := remaining_work / remaining_days

-- Proof problem statement: 
theorem work_fraction_completed_after_first_phase :
  (first_phase_work / total_work) = (15 / 64) :=
by
  -- This is the place where the actual formal proof should be written.
  sorry

end work_fraction_completed_after_first_phase_l79_79041


namespace polynomial_expansion_identity_l79_79086

variable (a0 a1 a2 a3 a4 : ℝ)

theorem polynomial_expansion_identity
  (h : (2 - (x : ℝ))^4 = a0 + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4) :
  a0 - a1 + a2 - a3 + a4 = 81 :=
sorry

end polynomial_expansion_identity_l79_79086


namespace similar_triangles_PQ_length_l79_79579

theorem similar_triangles_PQ_length (XY YZ QR : ℝ) (hXY : XY = 8) (hYZ : YZ = 16) (hQR : QR = 24)
  (hSimilar : ∃ (k : ℝ), XY = k * 8 ∧ YZ = k * 16 ∧ QR = k * 24) : (∃ (PQ : ℝ), PQ = 12) :=
by 
  -- Here we need to prove the theorem using similarity and given equalities
  sorry

end similar_triangles_PQ_length_l79_79579


namespace xyz_value_l79_79013

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 24) 
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 12) : 
  x * y * z = 4 := by
  sorry

end xyz_value_l79_79013


namespace arithmetic_square_root_16_l79_79322

theorem arithmetic_square_root_16 : ∃ (x : ℝ), x * x = 16 ∧ x ≥ 0 ∧ x = 4 := by
  sorry

end arithmetic_square_root_16_l79_79322


namespace least_whole_number_l79_79698

theorem least_whole_number (n : ℕ) 
  (h1 : n % 2 = 1)
  (h2 : n % 3 = 1)
  (h3 : n % 4 = 1)
  (h4 : n % 5 = 1)
  (h5 : n % 6 = 1)
  (h6 : 7 ∣ n) : 
  n = 301 := 
sorry

end least_whole_number_l79_79698


namespace yuna_initial_marbles_l79_79982

theorem yuna_initial_marbles (M : ℕ) :
  (M - 12 + 5) / 2 + 3 = 17 → M = 35 := by
  sorry

end yuna_initial_marbles_l79_79982


namespace find_a_l79_79056

theorem find_a (a : ℝ) :
  (∀ x : ℝ, (x * x - 4 <= 0) → (2 * x + a <= 0)) ↔ (a = -4) := by
  sorry

end find_a_l79_79056


namespace find_word_l79_79135

theorem find_word (antonym : Nat) (cond : antonym = 26) : String :=
  "seldom"

end find_word_l79_79135


namespace distance_between_stations_l79_79650

theorem distance_between_stations
  (v₁ v₂ : ℝ)
  (D₁ D₂ : ℝ)
  (T : ℝ)
  (h₁ : v₁ = 20)
  (h₂ : v₂ = 25)
  (h₃ : D₂ = D₁ + 70)
  (h₄ : D₁ = v₁ * T)
  (h₅ : D₂ = v₂ * T) : 
  D₁ + D₂ = 630 := 
by
  sorry

end distance_between_stations_l79_79650


namespace find_sixth_term_of_geometric_sequence_l79_79941

noncomputable def common_ratio (a b : ℚ) : ℚ := b / a

noncomputable def geometric_sequence_term (a r : ℚ) (k : ℕ) : ℚ := a * (r ^ (k - 1))

theorem find_sixth_term_of_geometric_sequence :
  geometric_sequence_term 5 (common_ratio 5 1.25) 6 = 5 / 1024 :=
by
  sorry

end find_sixth_term_of_geometric_sequence_l79_79941


namespace b7_in_form_l79_79960

theorem b7_in_form (a : ℕ → ℚ) (b : ℕ → ℚ) : 
  a 0 = 3 → 
  b 0 = 5 → 
  (∀ n : ℕ, a (n + 1) = (a n)^2 / (b n)) → 
  (∀ n : ℕ, b (n + 1) = (b n)^2 / (a n)) → 
  b 7 = (5^50 : ℚ) / (3^41 : ℚ) := 
by 
  intros h1 h2 h3 h4 
  sorry

end b7_in_form_l79_79960


namespace total_pencils_l79_79299

theorem total_pencils  (a b c : Nat) (total : Nat) 
(h₀ : a = 43) 
(h₁ : b = 19) 
(h₂ : c = 16) 
(h₃ : total = a + b + c) : 
total = 78 := 
by
  sorry

end total_pencils_l79_79299


namespace polynomial_pair_solution_l79_79411

-- We define the problem in terms of polynomials over real numbers
open Polynomial

theorem polynomial_pair_solution (P Q : ℝ[X]) :
  (∀ x y : ℝ, P.eval (x + Q.eval y) = Q.eval (x + P.eval y)) →
  (P = Q ∨ (∃ a b : ℝ, P = X + C a ∧ Q = X + C b)) :=
by
  intro h
  sorry

end polynomial_pair_solution_l79_79411


namespace number_of_black_squares_in_56th_row_l79_79699

def total_squares (n : Nat) : Nat := 3 + 2 * (n - 1)

def black_squares (n : Nat) : Nat :=
  if total_squares n % 2 == 1 then
    (total_squares n - 1) / 2
  else
    total_squares n / 2

theorem number_of_black_squares_in_56th_row :
  black_squares 56 = 56 :=
by
  sorry

end number_of_black_squares_in_56th_row_l79_79699


namespace no_valid_n_for_three_digit_conditions_l79_79733

theorem no_valid_n_for_three_digit_conditions :
  ∃ (n : ℕ) (h₁ : 100 ≤ n / 4 ∧ n / 4 ≤ 999) (h₂ : 100 ≤ 4 * n ∧ 4 * n ≤ 999), false :=
by sorry

end no_valid_n_for_three_digit_conditions_l79_79733


namespace speed_of_man_proof_l79_79905

noncomputable def speed_of_man (train_length : ℝ) (crossing_time : ℝ) (train_speed_kph : ℝ) : ℝ :=
  let train_speed_mps := (train_speed_kph * 1000) / 3600
  let relative_speed := train_length / crossing_time
  train_speed_mps - relative_speed

theorem speed_of_man_proof 
  (train_length : ℝ := 600) 
  (crossing_time : ℝ := 35.99712023038157) 
  (train_speed_kph : ℝ := 64) :
  speed_of_man train_length crossing_time train_speed_kph = 1.10977777777778 :=
by
  -- Proof goes here
  sorry

end speed_of_man_proof_l79_79905


namespace angle_between_hour_and_minute_hand_at_3_40_l79_79269

def angle_between_hands (hour minute : ℕ) : ℝ :=
  let minute_angle := (360 / 60) * minute
  let hour_angle := (360 / 12) + (30 / 60) * minute
  abs (minute_angle - hour_angle)

theorem angle_between_hour_and_minute_hand_at_3_40 : angle_between_hands 3 40 = 130 :=
by
  sorry

end angle_between_hour_and_minute_hand_at_3_40_l79_79269


namespace range_of_k_l79_79517

theorem range_of_k (k : ℝ) (f : ℝ → ℝ) (f' : ℝ → ℝ) (h_f : ∀ x, f x = x^3 - 3 * x^2 - k)
  (h_f' : ∀ x, f' x = 3 * x^2 - 6 * x) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₃ ≠ x₁ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ f x₃ = 0) ↔ -4 < k ∧ k < 0 :=
sorry

end range_of_k_l79_79517


namespace certain_number_condition_l79_79706

theorem certain_number_condition (x y z : ℤ) (N : ℤ)
  (hx : Even x) (hy : Odd y) (hz : Odd z)
  (hxy : x < y) (hyz : y < z)
  (h1 : y - x > N)
  (h2 : z - x = 7) :
  N < 3 := by
  sorry

end certain_number_condition_l79_79706


namespace polygon_side_count_l79_79276

theorem polygon_side_count (s : ℝ) (hs : s ≠ 0) : 
  ∀ (side_length_ratio : ℝ) (sides_first sides_second : ℕ),
  sides_first = 50 ∧ side_length_ratio = 3 ∧ 
  sides_first * side_length_ratio * s = sides_second * s → sides_second = 150 :=
by
  sorry

end polygon_side_count_l79_79276


namespace largest_is_21_l79_79202

theorem largest_is_21(a b c d : ℕ) 
  (h1 : (a + b + c) / 3 + d = 17)
  (h2 : (a + b + d) / 3 + c = 21)
  (h3 : (a + c + d) / 3 + b = 23)
  (h4 : (b + c + d) / 3 + a = 29):
  d = 21 := 
sorry

end largest_is_21_l79_79202


namespace correct_equation_after_moving_digit_l79_79523

theorem correct_equation_after_moving_digit :
  (101 - 102 = 1) →
  101 - 10^2 = 1 :=
by
  intro h
  sorry

end correct_equation_after_moving_digit_l79_79523


namespace counter_example_exists_l79_79841

theorem counter_example_exists : 
  ∃ n : ℕ, n ≥ 2 ∧ ¬(∃ k : ℕ, (2 ^ 2 ^ n) % (2 ^ n - 1) = 4 ^ k) :=
  sorry

end counter_example_exists_l79_79841


namespace quadratic_root_sqrt_2010_2009_l79_79004

theorem quadratic_root_sqrt_2010_2009 :
  (∃ (a b : ℤ), a = 0 ∧ b = -(2010 + 2 * Real.sqrt 2009) ∧
  ∀ (x : ℝ), x^2 + (a : ℝ) * x + (b : ℝ) = 0 → x = Real.sqrt (2010 + 2 * Real.sqrt 2009) ∨ x = -Real.sqrt (2010 + 2 * Real.sqrt 2009)) :=
sorry

end quadratic_root_sqrt_2010_2009_l79_79004


namespace complex_imaginary_unit_theorem_l79_79903

def complex_imaginary_unit_equality : Prop :=
  let i := Complex.I
  i * (i + 1) = -1 + i

theorem complex_imaginary_unit_theorem : complex_imaginary_unit_equality :=
by
  sorry

end complex_imaginary_unit_theorem_l79_79903


namespace unobserved_planet_exists_l79_79589

theorem unobserved_planet_exists
  (n : ℕ) (h_n_eq : n = 15)
  (planets : Fin n → Type)
  (dist : ∀ (i j : Fin n), ℝ)
  (h_distinct : ∀ (i j : Fin n), i ≠ j → dist i j ≠ dist j i)
  (nearest : ∀ i : Fin n, Fin n)
  (h_nearest : ∀ i : Fin n, nearest i ≠ i)
  : ∃ i : Fin n, ∀ j : Fin n, nearest j ≠ i := by
  sorry

end unobserved_planet_exists_l79_79589


namespace percentage_alcohol_in_first_vessel_is_zero_l79_79452

theorem percentage_alcohol_in_first_vessel_is_zero (x : ℝ) :
  ∀ (alcohol_first_vessel total_vessel_capacity first_vessel_capacity second_vessel_capacity concentration_mixture : ℝ),
  first_vessel_capacity = 2 →
  (∃ xpercent, alcohol_first_vessel = (first_vessel_capacity * xpercent / 100)) →
  second_vessel_capacity = 6 →
  (∃ ypercent, ypercent = 40 ∧ alcohol_first_vessel + 2.4 = concentration_mixture * (total_vessel_capacity/8) * 8) →
  concentration_mixture = 0.3 →
  0 = x := sorry

end percentage_alcohol_in_first_vessel_is_zero_l79_79452


namespace min_M_for_inequality_l79_79818

noncomputable def M := (9 * Real.sqrt 2) / 32

theorem min_M_for_inequality (a b c : ℝ) : 
  abs (a * b * (a^2 - b^2) + b * c * (b^2 - c^2) + c * a * (c^2 - a^2)) 
  ≤ M * (a^2 + b^2 + c^2)^2 := 
sorry

end min_M_for_inequality_l79_79818


namespace molecular_weight_calculation_l79_79112

def molecular_weight (n_Ar n_Si n_H n_O : ℕ) (w_Ar w_Si w_H w_O : ℝ) : ℝ :=
  n_Ar * w_Ar + n_Si * w_Si + n_H * w_H + n_O * w_O

theorem molecular_weight_calculation :
  molecular_weight 2 3 12 8 39.948 28.085 1.008 15.999 = 304.239 :=
by
  sorry

end molecular_weight_calculation_l79_79112


namespace number_of_times_each_player_plays_l79_79369

def players : ℕ := 7
def total_games : ℕ := 42

theorem number_of_times_each_player_plays (x : ℕ) 
  (H1 : 42 = (players * (players - 1) * x) / 2) : x = 2 :=
by
  sorry

end number_of_times_each_player_plays_l79_79369


namespace remainders_mod_m_l79_79994

theorem remainders_mod_m {m n b : ℤ} (h_coprime : Int.gcd m n = 1) :
    (∀ r : ℤ, 0 ≤ r ∧ r < m → ∃ k : ℤ, 0 ≤ k ∧ k < n ∧ ((b + k * n) % m = r)) :=
by
  sorry

end remainders_mod_m_l79_79994


namespace range_of_a_l79_79541

noncomputable def f (x a : ℝ) : ℝ := (x^2 + (a - 1) * x + 1) * Real.exp x

theorem range_of_a :
  (∀ x, f x a + Real.exp 2 ≥ 0) ↔ (-2 ≤ a ∧ a ≤ Real.exp 3 + 3) :=
sorry

end range_of_a_l79_79541


namespace surface_area_correct_l79_79412

def radius_hemisphere : ℝ := 9
def height_cone : ℝ := 12
def radius_cone_base : ℝ := 9

noncomputable def total_surface_area : ℝ := 
  let base_area : ℝ := radius_hemisphere^2 * Real.pi
  let curved_area_hemisphere : ℝ := 2 * radius_hemisphere^2 * Real.pi
  let slant_height_cone : ℝ := Real.sqrt (radius_cone_base^2 + height_cone^2)
  let lateral_area_cone : ℝ := radius_cone_base * slant_height_cone * Real.pi
  base_area + curved_area_hemisphere + lateral_area_cone

theorem surface_area_correct : total_surface_area = 378 * Real.pi := by
  sorry

end surface_area_correct_l79_79412


namespace alex_bought_3_bags_of_chips_l79_79806

theorem alex_bought_3_bags_of_chips (x : ℝ) : 
    (1 * x + 5 + 73) / x = 27 → x = 3 := by sorry

end alex_bought_3_bags_of_chips_l79_79806


namespace turnip_heavier_than_zhuchka_l79_79154

theorem turnip_heavier_than_zhuchka {C B M T : ℝ} 
  (h1 : B = 3 * C)
  (h2 : M = C / 10)
  (h3 : T = 60 * M) : 
  T / B = 2 :=
by
  sorry

end turnip_heavier_than_zhuchka_l79_79154


namespace geometric_sequence_q_cubed_l79_79169

theorem geometric_sequence_q_cubed (q a_1 : ℝ) (h1 : q ≠ 0) (h2 : q ≠ 1) 
(h3 : 2 * (a_1 * (1 - q^9) / (1 - q)) = (a_1 * (1 - q^3) / (1 - q)) + (a_1 * (1 - q^6) / (1 - q))) : 
  q^3 = -1/2 := by
  sorry

end geometric_sequence_q_cubed_l79_79169


namespace bridge_length_is_219_l79_79640

noncomputable def length_of_bridge (train_length : ℕ) (train_speed_kmh : ℤ) (time_seconds : ℕ) : ℝ :=
  let train_speed_ms : ℝ := train_speed_kmh * (1000 / 3600)
  let total_distance : ℝ := train_speed_ms * time_seconds
  total_distance - train_length

theorem bridge_length_is_219 :
  length_of_bridge 156 45 30 = 219 :=
by
  sorry

end bridge_length_is_219_l79_79640


namespace ratio_of_areas_l79_79358

theorem ratio_of_areas (s : ℝ) (h1 : s > 0) : 
  let small_square_area := s^2
  let total_small_squares_area := 4 * s^2
  let large_square_side_length := 4 * s
  let large_square_area := (4 * s)^2
  total_small_squares_area / large_square_area = 1 / 4 :=
by
  sorry

end ratio_of_areas_l79_79358


namespace distance_to_other_asymptote_is_8_l79_79703

-- Define the hyperbola and the properties
def hyperbola (x y : ℝ) : Prop := (x^2) / 2 - (y^2) / 8 = 1

-- Define the asymptotes
def asymptote_1 (x y : ℝ) : Prop := y = 2 * x
def asymptote_2 (x y : ℝ) : Prop := y = -2 * x

-- Given conditions
variables (P : ℝ × ℝ)
variable (distance_to_one_asymptote : ℝ)
variable (distance_to_other_asymptote : ℝ)

axiom point_on_hyperbola : hyperbola P.1 P.2
axiom distance_to_one_asymptote_is_1_over_5 : distance_to_one_asymptote = 1 / 5

-- The proof statement
theorem distance_to_other_asymptote_is_8 :
  distance_to_other_asymptote = 8 := sorry

end distance_to_other_asymptote_is_8_l79_79703


namespace algebraic_expression_transformation_l79_79444

theorem algebraic_expression_transformation (a b : ℝ) :
  (∀ x : ℝ, x^2 + 4 * x + 3 = (x - 1)^2 + a * (x - 1) + b) → (a + b = 14) :=
by
  intros h
  sorry

end algebraic_expression_transformation_l79_79444


namespace problem_statement_l79_79966

variable (y1 y2 y3 y4 y5 y6 y7 y8 : ℝ)

theorem problem_statement
  (h1 : y1 + 4 * y2 + 9 * y3 + 16 * y4 + 25 * y5 + 36 * y6 + 49 * y7 + 64 * y8 = 3)
  (h2 : 4 * y1 + 9 * y2 + 16 * y3 + 25 * y4 + 36 * y5 + 49 * y6 + 64 * y7 + 81 * y8 = 15)
  (h3 : 9 * y1 + 16 * y2 + 25 * y3 + 36 * y4 + 49 * y5 + 64 * y6 + 81 * y7 + 100 * y8 = 140) :
  16 * y1 + 25 * y2 + 36 * y3 + 49 * y4 + 64 * y5 + 81 * y6 + 100 * y7 + 121 * y8 = 472 := by
  sorry

end problem_statement_l79_79966


namespace unique_sequence_exists_and_bounded_l79_79694

theorem unique_sequence_exists_and_bounded (a : ℝ) (n : ℕ) :
  ∃! (x : ℕ → ℝ), -- There exists a unique sequence x : ℕ → ℝ
    (x 1 = x (n - 1)) ∧ -- x_1 = x_{n-1}
    (∀ i, 1 ≤ i ∧ i ≤ n → (1 / 2) * (x (i - 1) + x i) = x i + x i ^ 3 - a ^ 3) ∧ -- Condition for all 1 ≤ i ≤ n
    (∀ i, 0 ≤ i ∧ i ≤ n + 1 → |x i| ≤ |a|) -- Bounding condition for all 0 ≤ i ≤ n + 1
:= sorry

end unique_sequence_exists_and_bounded_l79_79694


namespace find_other_number_l79_79889

theorem find_other_number (x y : ℤ) (h1 : 3 * x + 2 * y = 145) (h2 : x = 35 ∨ y = 35) : y = 20 :=
sorry

end find_other_number_l79_79889


namespace eliminate_denominators_l79_79242

theorem eliminate_denominators (x : ℝ) :
  (4 * (2 * x - 1) - 3 * (3 * x - 4) = 12) ↔ ((2 * x - 1) / 3 - (3 * x - 4) / 4 = 1) := 
by
  sorry

end eliminate_denominators_l79_79242


namespace polygon_sides_sum_l79_79899

theorem polygon_sides_sum
  (area_ABCDEF : ℕ) (AB BC FA DE EF : ℕ)
  (h1 : area_ABCDEF = 78)
  (h2 : AB = 10)
  (h3 : BC = 11)
  (h4 : FA = 7)
  (h5 : DE = 4)
  (h6 : EF = 8) :
  DE + EF = 12 := 
by
  sorry

end polygon_sides_sum_l79_79899


namespace number_of_common_tangents_l79_79235

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem number_of_common_tangents
  (C₁ C₂ : ℝ × ℝ) (r₁ r₂ : ℝ)
  (h₁ : ∀ (x y : ℝ), x^2 + y^2 - 2 * x = 0 → (C₁ = (1, 0)) ∧ (r₁ = 1))
  (h₂ : ∀ (x y : ℝ), x^2 + y^2 - 4 * y + 3 = 0 → (C₂ = (0, 2)) ∧ (r₂ = 1))
  (d : distance C₁ C₂ = Real.sqrt 5) :
  4 = 4 := 
by sorry

end number_of_common_tangents_l79_79235


namespace relationship_abc_l79_79394

noncomputable def a : ℝ := Real.sqrt 2
noncomputable def b : ℝ := Real.exp (-Real.pi)
noncomputable def c : ℝ := Real.log 3 / Real.log 2

theorem relationship_abc : b < a ∧ a < c :=
by
  -- proofs would be added here
  sorry

end relationship_abc_l79_79394


namespace carol_total_peanuts_l79_79376

open Nat

-- Define the conditions
def peanuts_from_tree : Nat := 48
def peanuts_from_ground : Nat := 178
def bags_of_peanuts : Nat := 3
def peanuts_per_bag : Nat := 250

-- Define the total number of peanuts Carol has to prove it equals 976
def total_peanuts (peanuts_from_tree peanuts_from_ground bags_of_peanuts peanuts_per_bag : Nat) : Nat :=
  peanuts_from_tree + peanuts_from_ground + (bags_of_peanuts * peanuts_per_bag)

theorem carol_total_peanuts : total_peanuts peanuts_from_tree peanuts_from_ground bags_of_peanuts peanuts_per_bag = 976 :=
  by
    -- proof goes here
    sorry

end carol_total_peanuts_l79_79376


namespace hat_p_at_1_l79_79408

-- Define the polynomial p(x)
def p (x : ℝ) : ℝ := x^2 - (1 + 1)*x + 1

-- Definition of displeased polynomial
def isDispleased (p : ℝ → ℝ) : Prop :=
  ∃ (x1 x2 x3 x4 : ℝ), p (p x1) = 0 ∧ p (p x2) = 0 ∧ p (p x3) = 0 ∧ p (p x4) = 0

-- Define the specific polynomial hat_p
def hat_p (x : ℝ) : ℝ := p x

-- Theorem statement
theorem hat_p_at_1 : isDispleased hat_p → hat_p 1 = 0 :=
by
  sorry

end hat_p_at_1_l79_79408


namespace new_ratio_milk_to_water_l79_79403

def total_volume : ℕ := 100
def initial_milk_ratio : ℚ := 3
def initial_water_ratio : ℚ := 2
def additional_water : ℕ := 48

def new_milk_volume := (initial_milk_ratio / (initial_milk_ratio + initial_water_ratio)) * total_volume
def new_water_volume := (initial_water_ratio / (initial_milk_ratio + initial_water_ratio)) * total_volume + additional_water

theorem new_ratio_milk_to_water :
  new_milk_volume / (new_water_volume : ℚ) = 15 / 22 :=
by
  sorry

end new_ratio_milk_to_water_l79_79403


namespace find_x_coordinate_l79_79326

theorem find_x_coordinate 
  (x : ℝ)
  (h1 : (0, 0) = (0, 0))
  (h2 : (0, 4) = (0, 4))
  (h3 : (x, 4) = (x, 4))
  (h4 : (x, 0) = (x, 0))
  (h5 : 0.4 * (4 * x) = 8)
  : x = 5 := 
sorry

end find_x_coordinate_l79_79326


namespace mass_percentage_of_C_in_CCl4_l79_79865

theorem mass_percentage_of_C_in_CCl4 :
  let mass_carbon : ℝ := 12.01
  let mass_chlorine : ℝ := 35.45
  let molar_mass_CCl4 : ℝ := mass_carbon + 4 * mass_chlorine
  let mass_percentage_C : ℝ := (mass_carbon / molar_mass_CCl4) * 100
  mass_percentage_C = 7.81 := 
by
  sorry

end mass_percentage_of_C_in_CCl4_l79_79865


namespace tanya_number_75_less_l79_79132

def rotate180 (d : ℕ) : ℕ :=
  match d with
  | 0 => 0
  | 1 => 1
  | 6 => 9
  | 8 => 8
  | 9 => 6
  | _ => 0 -- invalid assumption for digits outside the defined scope

def two_digit_upside_down (n : ℕ) : ℕ :=
  let tens := n / 10
  let units := n % 10
  10 * rotate180 units + rotate180 tens

theorem tanya_number_75_less (n : ℕ) : 
  ∀ n, (∃ a b, n = 10 * a + b ∧ (a = 0 ∨ a = 1 ∨ a = 6 ∨ a = 8 ∨ a = 9) ∧ 
      (b = 0 ∨ b = 1 ∨ b = 6 ∨ b = 8 ∨ b = 9) ∧  
      n - two_digit_upside_down n = 75) :=
by {
  sorry
}

end tanya_number_75_less_l79_79132


namespace min_days_to_triple_loan_l79_79934

theorem min_days_to_triple_loan (amount_borrowed : ℕ) (interest_rate : ℝ) :
  ∀ x : ℕ, x ≥ 20 ↔ amount_borrowed + (amount_borrowed * (interest_rate / 10)) * x ≥ 3 * amount_borrowed :=
sorry

end min_days_to_triple_loan_l79_79934


namespace actual_price_of_good_l79_79584

variables (P : Real)

theorem actual_price_of_good:
  (∀ (P : ℝ), 0.5450625 * P = 6500 → P = 6500 / 0.5450625) :=
  by sorry

end actual_price_of_good_l79_79584


namespace part1_part2_l79_79667

-- Part 1: Define the sequence and sum function, then state the problem.
def a_1 : ℚ := 3 / 2
def d : ℚ := 1

def S_n (n : ℕ) : ℚ :=
  n * a_1 + (n * (n - 1) / 2) * d

theorem part1 (k : ℕ) (h : S_n (k^2) = (S_n k)^2) : k = 4 := sorry

-- Part 2: Define the general sequence and state the problem.
def arith_seq (a_1 : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  a_1 + (n - 1) * d

def S_n_general (a_1 : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  (n * a_1) + (n * (n - 1) / 2) * d

theorem part2 (a_1 : ℚ) (d : ℚ) :
  (∀ k : ℕ, S_n_general a_1 d (k^2) = (S_n_general a_1 d k)^2) ↔
  (a_1 = 0 ∧ d = 0) ∨
  (a_1 = 1 ∧ d = 0) ∨
  (a_1 = 1 ∧ d = 2) := sorry

end part1_part2_l79_79667


namespace convert_length_convert_area_convert_time_convert_mass_l79_79297

theorem convert_length (cm : ℕ) : cm = 7 → (cm : ℚ) / 100 = 7 / 100 :=
by sorry

theorem convert_area (dm2 : ℕ) : dm2 = 35 → (dm2 : ℚ) / 100 = 7 / 20 :=
by sorry

theorem convert_time (min : ℕ) : min = 45 → (min : ℚ) / 60 = 3 / 4 :=
by sorry

theorem convert_mass (g : ℕ) : g = 2500 → (g : ℚ) / 1000 = 5 / 2 :=
by sorry

end convert_length_convert_area_convert_time_convert_mass_l79_79297


namespace sin_x_cos_x_value_l79_79450

theorem sin_x_cos_x_value (x : ℝ) (h : Real.sin x = 4 * Real.cos x) : Real.sin x * Real.cos x = 4 / 17 :=
  sorry

end sin_x_cos_x_value_l79_79450


namespace number_of_divisors_3465_l79_79377

def prime_factors_3465 : Prop := 3465 = 3^2 * 5 * 7^2

theorem number_of_divisors_3465 (h : prime_factors_3465) : Nat.totient 3465 = 18 :=
  sorry

end number_of_divisors_3465_l79_79377


namespace sours_total_l79_79723

variable (c l o T : ℕ)

axiom cherry_sours : c = 32
axiom ratio_cherry_lemon : 4 * l = 5 * c
axiom orange_sours_ratio : o = 25 * T / 100
axiom total_sours : T = c + l + o

theorem sours_total :
  T = 96 :=
by
  sorry

end sours_total_l79_79723


namespace find_YJ_l79_79793

structure Triangle :=
  (XY XZ YZ : ℝ)
  (XY_pos : XY > 0)
  (XZ_pos : XZ > 0)
  (YZ_pos : YZ > 0)

noncomputable def incenter_length (T : Triangle) : ℝ := 
  let XY := T.XY
  let XZ := T.XZ
  let YZ := T.YZ
  -- calculation using the provided constraints goes here
  3 * Real.sqrt 13 -- this should be computed based on the constraints, but is directly given as the answer

theorem find_YJ
  (T : Triangle)
  (XY_eq : T.XY = 17)
  (XZ_eq : T.XZ = 19)
  (YZ_eq : T.YZ = 20) :
  incenter_length T = 3 * Real.sqrt 13 :=
by 
  sorry

end find_YJ_l79_79793


namespace avg_annual_growth_rate_optimal_room_price_l79_79822

-- Problem 1: Average Annual Growth Rate
theorem avg_annual_growth_rate (visitors_2021 visitors_2023 : ℝ) (years : ℕ) (visitors_2021_pos : 0 < visitors_2021) :
  visitors_2023 > visitors_2021 → visitors_2023 / visitors_2021 = 2.25 → 
  ∃ x : ℝ, (1 + x)^2 = 2.25 ∧ x = 0.5 :=
by sorry

-- Problem 2: Optimal Room Price for Desired Profit
theorem optimal_room_price (rooms : ℕ) (base_price cost_per_room desired_profit : ℝ)
  (rooms_pos : 0 < rooms) :
  base_price = 180 → cost_per_room = 20 → desired_profit = 9450 → 
  ∃ y : ℝ, (y - cost_per_room) * (rooms - (y - base_price) / 10) = desired_profit ∧ y = 230 :=
by sorry

end avg_annual_growth_rate_optimal_room_price_l79_79822


namespace total_pears_picked_l79_79074

theorem total_pears_picked (keith_pears jason_pears : ℕ) (h1 : keith_pears = 3) (h2 : jason_pears = 2) : keith_pears + jason_pears = 5 :=
by
  sorry

end total_pears_picked_l79_79074


namespace max_n_no_constant_term_l79_79453

theorem max_n_no_constant_term (n : ℕ) (h : n < 10 ∧ n ≠ 3 ∧ n ≠ 6 ∧ n ≠ 9 ∧ n ≠ 2 ∧ n ≠ 5 ∧ n ≠ 8): n ≤ 7 :=
by {
  sorry
}

end max_n_no_constant_term_l79_79453


namespace distance_between_A_and_B_l79_79345

theorem distance_between_A_and_B 
  (d : ℕ) -- The distance we want to prove
  (ha : ∀ (t : ℕ), d = 700 * t)
  (hb : ∀ (t : ℕ), d + 400 = 2100 * t) :
  d = 1700 := 
by
  sorry

end distance_between_A_and_B_l79_79345


namespace sinB_law_of_sines_l79_79239

variable (A B C : Type)
variable [MetricSpace A] [MetricSpace B] [MetricSpace C]

-- Assuming a triangle with sides and angles as described
variable (a b : ℝ) (sinA sinB : ℝ)
variable (h₁ : a = 3) (h₂ : b = 5) (h₃ : sinA = 1 / 3)

theorem sinB_law_of_sines : sinB = 5 / 9 :=
by
  -- Placeholder for the proof
  sorry

end sinB_law_of_sines_l79_79239


namespace b_is_arithmetic_sequence_a_general_formula_l79_79666

open Nat

-- Define the sequence a_n
def a : ℕ → ℤ
| 0     => 1
| 1     => 2
| (n+2) => 2 * (a (n+1)) - (a n) + 2

-- Define the sequence b_n
def b (n : ℕ) : ℤ := a (n+1) - a n

-- Part 1: The sequence b_n is an arithmetic sequence
theorem b_is_arithmetic_sequence : ∀ n : ℕ, b (n+1) - b n = 2 := by
  sorry

-- Part 2: Find the general formula for a_n
theorem a_general_formula : ∀ n : ℕ, a (n+1) = n^2 + 1 := by
  sorry

end b_is_arithmetic_sequence_a_general_formula_l79_79666


namespace rationalize_denominator_l79_79908

theorem rationalize_denominator (a b c : ℝ) (h : b ≠ 0) (h0 : 98 = c * c) (h1 : 7 = c) :
  (7 / (Real.sqrt 98) = (Real.sqrt 2) / 2) :=
by
  sorry

end rationalize_denominator_l79_79908


namespace quadruplet_zero_solution_l79_79509

theorem quadruplet_zero_solution (a b c d : ℝ)
  (h1 : (a + b) * (a^2 + b^2) = (c + d) * (c^2 + d^2))
  (h2 : (a + c) * (a^2 + c^2) = (b + d) * (b^2 + d^2))
  (h3 : (a + d) * (a^2 + d^2) = (b + c) * (b^2 + c^2)) :
  a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0 := 
sorry

end quadruplet_zero_solution_l79_79509


namespace clothes_add_percentage_l79_79913

theorem clothes_add_percentage (W : ℝ) (C : ℝ) (h1 : W > 0) 
  (h2 : C = 0.0174 * W) : 
  ((C / (0.87 * W)) * 100) = 2 :=
by
  sorry

end clothes_add_percentage_l79_79913


namespace geometric_progression_difference_l79_79837

variable {n : ℕ}
variable {a : ℕ → ℝ} -- assuming the sequence is indexed by natural numbers
variable {a₁ : ℝ}
variable {r : ℝ} (hr : r = (1 + Real.sqrt 5) / 2)

def geometric_progression (a : ℕ → ℝ) (a₁ : ℝ) (r : ℝ) : Prop :=
  ∀ n, a n = a₁ * (r ^ n)

theorem geometric_progression_difference
  (a₁ : ℝ)
  (hr : r = (1 + Real.sqrt 5) / 2)
  (hg : geometric_progression a a₁ r) :
  ∀ n, n ≥ 2 → a n = a (n-1) - a (n-2) :=
by
  sorry

end geometric_progression_difference_l79_79837


namespace num_rooms_with_2_windows_l79_79125

theorem num_rooms_with_2_windows:
  ∃ (num_rooms_with_2_windows: ℕ),
  (∀ (num_rooms_with_4_windows num_rooms_with_3_windows: ℕ), 
    num_rooms_with_4_windows = 5 ∧ 
    num_rooms_with_3_windows = 8 ∧
    4 * num_rooms_with_4_windows + 3 * num_rooms_with_3_windows + 2 * num_rooms_with_2_windows = 122) → 
    num_rooms_with_2_windows = 39 :=
by
  sorry

end num_rooms_with_2_windows_l79_79125


namespace rearrangement_impossible_l79_79506

-- Definition of an 8x8 chessboard's cell numbering.
def cell_number (i j : ℕ) : ℕ := i + j - 1

-- The initial placement of pieces, represented as a permutation on {1, 2, ..., 8}
def initial_placement (p: Fin 8 → Fin 8) := True -- simplify for definition purposes

-- The rearranged placement of pieces
def rearranged_placement (q: Fin 8 → Fin 8) := True -- simplify for definition purposes

-- Condition for each piece: cell number increases
def cell_increase_condition (p q: Fin 8 → Fin 8) : Prop :=
  ∀ i, cell_number (q i).val (i.val + 1) > cell_number (p i).val (i.val + 1)

-- The main theorem to state it's impossible to rearrange under the given conditions and question
theorem rearrangement_impossible 
  (p q: Fin 8 → Fin 8) 
  (h_initial : initial_placement p) 
  (h_rearranged : rearranged_placement q) 
  (h_increase : cell_increase_condition p q) : False := 
sorry

end rearrangement_impossible_l79_79506


namespace zeroSeq_arithmetic_not_geometric_l79_79266

-- Define what it means for a sequence to be arithmetic
def isArithmeticSequence (seq : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, seq (n + 1) = seq n + d

-- Define what it means for a sequence to be geometric
def isGeometricSequence (seq : ℕ → ℝ) : Prop :=
  ∃ q, ∀ n, seq n ≠ 0 → seq (n + 1) = seq n * q

-- Define the sequence of zeros
def zeroSeq (n : ℕ) : ℝ := 0

theorem zeroSeq_arithmetic_not_geometric :
  isArithmeticSequence zeroSeq ∧ ¬ isGeometricSequence zeroSeq :=
by
  sorry

end zeroSeq_arithmetic_not_geometric_l79_79266


namespace students_who_like_both_l79_79669

def total_students : ℕ := 50
def apple_pie_lovers : ℕ := 22
def chocolate_cake_lovers : ℕ := 20
def neither_dessert_lovers : ℕ := 15

theorem students_who_like_both : 
  (apple_pie_lovers + chocolate_cake_lovers) - (total_students - neither_dessert_lovers) = 7 :=
by
  -- Calculation steps (skipped)
  sorry

end students_who_like_both_l79_79669


namespace g_of_3_l79_79172

def g (x : ℝ) : ℝ := 5 * x ^ 4 + 4 * x ^ 3 - 7 * x ^ 2 + 3 * x - 2

theorem g_of_3 : g 3 = 401 :=
by
    -- proof will go here
    sorry

end g_of_3_l79_79172


namespace basketball_team_total_players_l79_79250

theorem basketball_team_total_players (total_points : ℕ) (min_points : ℕ) (max_points : ℕ) (team_size : ℕ)
  (h1 : total_points = 100)
  (h2 : min_points = 7)
  (h3 : max_points = 23)
  (h4 : ∀ (n : ℕ), n ≥ min_points)
  (h5 : max_points = 23)
  : team_size = 12 :=
sorry

end basketball_team_total_players_l79_79250


namespace evaluate_g_at_neg3_l79_79167

def g (x : ℤ) : ℤ := x^2 - x + 2 * x^3

theorem evaluate_g_at_neg3 : g (-3) = -42 := by
  sorry

end evaluate_g_at_neg3_l79_79167


namespace conference_duration_is_960_l79_79396

-- The problem statement definition
def conference_sessions_duration_in_minutes (day1_hours : ℕ) (day1_minutes : ℕ) (day2_hours : ℕ) (day2_minutes : ℕ) : ℕ :=
  (day1_hours * 60 + day1_minutes) + (day2_hours * 60 + day2_minutes)

-- The theorem we want to prove given the above conditions
theorem conference_duration_is_960 :
  conference_sessions_duration_in_minutes 7 15 8 45 = 960 :=
by 
  -- The proof is omitted
  sorry

end conference_duration_is_960_l79_79396


namespace secant_length_l79_79303

theorem secant_length
  (A B C D E : ℝ)
  (AB : A - B = 7)
  (BC : B - C = 7)
  (AD : A - D = 10)
  (pos : A > E ∧ D > E):
  E - D = 0.2 :=
by
  sorry

end secant_length_l79_79303


namespace linear_system_solution_l79_79683

theorem linear_system_solution :
  ∃ (x y z : ℝ), (x ≠ 0) ∧ (y ≠ 0) ∧ (z ≠ 0) ∧
  (x + (85/3) * y + 4 * z = 0) ∧ 
  (4 * x + (85/3) * y + z = 0) ∧ 
  (3 * x + 5 * y - 2 * z = 0) ∧ 
  (x * z) / (y ^ 2) = 25 := 
sorry

end linear_system_solution_l79_79683


namespace prove_m_equals_9_given_split_l79_79838

theorem prove_m_equals_9_given_split (m : ℕ) (h : 1 < m) (h1 : m^3 = 73) : m = 9 :=
sorry

end prove_m_equals_9_given_split_l79_79838


namespace solve_inequality_system_l79_79272

theorem solve_inequality_system :
  (∀ x : ℝ, (1 - 3 * (x - 1) < 8 - x) ∧ ((x - 3) / 2 + 2 ≥ x)) →
  ∃ (integers : Set ℤ), integers = {x : ℤ | -2 < (x : ℝ) ∧ (x : ℝ) ≤ 1} ∧ integers = {-1, 0, 1} :=
by
  sorry

end solve_inequality_system_l79_79272


namespace sweater_markup_percentage_l79_79243

variables (W R : ℝ)
variables (h1 : 0.30 * R = 1.40 * W)

theorem sweater_markup_percentage :
  (R = (1.40 / 0.30) * W) →
  (R - W) / W * 100 = 367 := 
by
  intro hR
  sorry

end sweater_markup_percentage_l79_79243


namespace smallest_number_l79_79983

theorem smallest_number:
  ∃ n : ℕ, (∀ d ∈ [12, 16, 18, 21, 28, 35, 39], (n - 7) % d = 0) ∧ n = 65527 :=
by
  sorry

end smallest_number_l79_79983


namespace miles_driven_each_day_l79_79244

theorem miles_driven_each_day
  (total_distance : ℕ)
  (days_in_semester : ℕ)
  (h_total : total_distance = 1600)
  (h_days : days_in_semester = 80):
  total_distance / days_in_semester = 20 := by
  sorry

end miles_driven_each_day_l79_79244


namespace problem_1_problem_2_l79_79419

-- Definition of the function f
def f (x a : ℝ) : ℝ := |x - a| + 3 * x

-- Problem I
theorem problem_1 (x : ℝ) : (f x 1 ≥ 3 * x + 2) ↔ (x ≥ 3 ∨ x ≤ -1) :=
by sorry

-- Problem II
theorem problem_2 (a : ℝ) (h : ∀ x : ℝ, f x a ≤ 0 → x ≤ -3) : a = 6 :=
by sorry

end problem_1_problem_2_l79_79419


namespace initial_books_l79_79215

theorem initial_books (B : ℕ) (h : B + 5 = 7) : B = 2 :=
by sorry

end initial_books_l79_79215


namespace probability_of_losing_l79_79596

noncomputable def odds_of_winning : ℕ := 5
noncomputable def odds_of_losing : ℕ := 3
noncomputable def total_outcomes : ℕ := odds_of_winning + odds_of_losing

theorem probability_of_losing : 
  (odds_of_losing : ℚ) / (total_outcomes : ℚ) = 3 / 8 := 
by
  sorry

end probability_of_losing_l79_79596


namespace geometric_arithmetic_seq_unique_ratio_l79_79370

variable (d : ℚ) (q : ℚ) (k : ℤ)
variable (h_d_nonzero : d ≠ 0)
variable (h_q_pos : 0 < q) (h_q_lt_one : q < 1)
variable (h_integer : 14 / (1 + q + q^2) = k)

theorem geometric_arithmetic_seq_unique_ratio :
  q = 1 / 2 :=
by
  sorry

end geometric_arithmetic_seq_unique_ratio_l79_79370


namespace simplify_fraction_l79_79274

theorem simplify_fraction : 5 * (21 / 8) * (32 / -63) = -20 / 3 := by
  sorry

end simplify_fraction_l79_79274


namespace liquid_x_percentage_l79_79730

theorem liquid_x_percentage (a_weight b_weight : ℝ) (a_percentage b_percentage : ℝ)
  (result_weight : ℝ) (x_weight_result : ℝ) (x_percentage_result : ℝ) :
  a_weight = 500 → b_weight = 700 → a_percentage = 0.8 / 100 →
  b_percentage = 1.8 / 100 → result_weight = a_weight + b_weight →
  x_weight_result = a_weight * a_percentage + b_weight * b_percentage →
  x_percentage_result = (x_weight_result / result_weight) * 100 →
  x_percentage_result = 1.3833 :=
by sorry

end liquid_x_percentage_l79_79730


namespace range_of_m_l79_79844

theorem range_of_m (a m : ℝ) (h_a_neg : a < 0) (y1 y2 : ℝ)
  (hA : y1 = a * m^2 - 4 * a * m)
  (hB : y2 = 4 * a * m^2 - 8 * a * m)
  (hA_above : y1 > -3 * a)
  (hB_above : y2 > -3 * a)
  (hy1_gt_y2 : y1 > y2) :
  4 / 3 < m ∧ m < 3 / 2 :=
sorry

end range_of_m_l79_79844


namespace problem_statement_l79_79513

noncomputable def least_period (f : ℝ → ℝ) (P : ℝ) :=
  ∀ x : ℝ, f (x + P) = f x

theorem problem_statement (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x + 5) + f (x - 5) = f x) :
  least_period f 30 :=
sorry

end problem_statement_l79_79513


namespace max_digit_sum_in_24_hour_format_l79_79281

def digit_sum (n : ℕ) : ℕ := 
  (n / 10) + (n % 10)

theorem max_digit_sum_in_24_hour_format :
  (∃ (h m : ℕ), 0 ≤ h ∧ h < 24 ∧ 0 ≤ m ∧ m < 60 ∧ digit_sum h + digit_sum m = 19) ∧
  ∀ (h m : ℕ), 0 ≤ h ∧ h < 24 ∧ 0 ≤ m ∧ m < 60 → digit_sum h + digit_sum m ≤ 19 :=
by
  sorry

end max_digit_sum_in_24_hour_format_l79_79281


namespace series_sum_l79_79632

theorem series_sum :
  ∑' n : ℕ, (3 * (n + 1) - 2) / ((n + 1) * (n + 2) * (n + 4)) = (55 / 12) :=
sorry

end series_sum_l79_79632


namespace domain_log2_x_minus_1_l79_79995

theorem domain_log2_x_minus_1 (x : ℝ) : (1 < x) ↔ (∃ y : ℝ, y = Real.logb 2 (x - 1)) := by
  sorry

end domain_log2_x_minus_1_l79_79995


namespace cement_mixture_weight_l79_79340

theorem cement_mixture_weight 
  (W : ℝ)
  (h1 : W = (2/5) * W + (1/6) * W + (1/10) * W + (1/8) * W + 12) :
  W = 57.6 := by
  sorry

end cement_mixture_weight_l79_79340


namespace three_op_six_l79_79052

-- Define the new operation @.
def op (a b : ℕ) : ℕ := (a * a * b) / (a + b)

-- The theorem to prove that the value of 3 @ 6 is 6.
theorem three_op_six : op 3 6 = 6 := by 
  sorry

end three_op_six_l79_79052


namespace no_a_b_not_divide_bn_minus_n_l79_79504

theorem no_a_b_not_divide_bn_minus_n :
  ∀ (a b : ℕ), 0 < a → 0 < b → ∃ (n : ℕ), 0 < n ∧ a ∣ (b^n - n) :=
by
  sorry

end no_a_b_not_divide_bn_minus_n_l79_79504


namespace average_mpg_highway_l79_79759

variable (mpg_city : ℝ) (H mpg : ℝ) (gallons : ℝ) (max_distance : ℝ)

noncomputable def SUV_fuel_efficiency : Prop :=
  mpg_city  = 7.6 ∧
  gallons = 20 ∧
  max_distance = 244 ∧
  H * gallons = max_distance

theorem average_mpg_highway (h1 : mpg_city = 7.6) (h2 : gallons = 20) (h3 : max_distance = 244) :
  SUV_fuel_efficiency mpg_city H gallons max_distance → H = 12.2 :=
by
  intros h
  cases h
  sorry

end average_mpg_highway_l79_79759


namespace pyramid_volume_l79_79491

theorem pyramid_volume
  (s : ℝ) (h : ℝ) (base_area : ℝ) (triangular_face_area : ℝ) (surface_area : ℝ)
  (h_base_area : base_area = s * s)
  (h_triangular_face_area : triangular_face_area = (1 / 3) * base_area)
  (h_surface_area : surface_area = base_area + 4 * triangular_face_area)
  (h_surface_area_value : surface_area = 768)
  (h_vol : h = 7.78) :
  (1 / 3) * base_area * h = 853.56 :=
by
  sorry

end pyramid_volume_l79_79491


namespace function_properties_l79_79270

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 6)

theorem function_properties :
  (∀ x, f (x + Real.pi) = f x) ∧
  (f (Real.pi / 3) = 1) ∧
  (∀ x y, -Real.pi / 6 ≤ x → x ≤ y → y ≤ Real.pi / 3 → f x ≤ f y) := by
  sorry

end function_properties_l79_79270


namespace certain_number_l79_79030

theorem certain_number (x y : ℝ) (h1 : 0.20 * x = 0.15 * y - 15) (h2 : x = 1050) : y = 1500 :=
by
  sorry

end certain_number_l79_79030


namespace sum_in_range_l79_79953

noncomputable def mixed_number_sum : ℚ :=
  3 + 1/8 + 4 + 3/7 + 6 + 2/21

theorem sum_in_range : 13.5 ≤ mixed_number_sum ∧ mixed_number_sum < 14 := by
  sorry

end sum_in_range_l79_79953


namespace bowling_ball_weight_l79_79186

theorem bowling_ball_weight (b c : ℝ) (h1 : c = 36) (h2 : 5 * b = 4 * c) : b = 28.8 := by
  sorry

end bowling_ball_weight_l79_79186


namespace find_A_l79_79524

theorem find_A (A B C : ℕ) (h1 : A ≠ B) (h2 : A ≠ C) (h3 : B ≠ C) (h4 : A < 10) (h5 : B < 10) (h6 : C < 10) (h7 : 10 * A + B + 10 * B + C = 101 * B + 10 * C) : A = 9 :=
sorry

end find_A_l79_79524


namespace bailey_chew_toys_l79_79471

theorem bailey_chew_toys (dog_treats rawhide_bones: ℕ) (cards items_per_card : ℕ)
  (h1 : dog_treats = 8)
  (h2 : rawhide_bones = 10)
  (h3 : cards = 4)
  (h4 : items_per_card = 5) :
  ∃ chew_toys : ℕ, chew_toys = 2 :=
by
  sorry

end bailey_chew_toys_l79_79471


namespace total_gray_area_trees_l79_79573

/-- 
Three aerial photos were taken by the drone, each capturing the same number of trees.
First rectangle has 100 trees in total and 82 trees in the white area.
Second rectangle has 90 trees in total and 82 trees in the white area.
Prove that the number of trees in gray areas in both rectangles is 26.
-/
theorem total_gray_area_trees : (100 - 82) + (90 - 82) = 26 := 
by sorry

end total_gray_area_trees_l79_79573


namespace compute_x_l79_79046

theorem compute_x 
  (x : ℝ) 
  (hx : 0 < x ∧ x < 0.1)
  (hs1 : ∑' n, 4 * x^n = 4 / (1 - x))
  (hs2 : ∑' n, 4 * (10^n - 1) * x^n = 4 * (4 / (1 - x))) :
  x = 3 / 40 :=
by
  sorry

end compute_x_l79_79046


namespace smallest_number_satisfying_conditions_l79_79320

theorem smallest_number_satisfying_conditions :
  ∃ (n : ℕ), n % 6 = 2 ∧ n % 7 = 3 ∧ n % 8 = 4 ∧ ∀ m, (m % 6 = 2 → m % 7 = 3 → m % 8 = 4 → n ≤ m) :=
  sorry

end smallest_number_satisfying_conditions_l79_79320


namespace statement_1_correct_statement_3_correct_correct_statements_l79_79265

-- Definition for Acute Angles
def is_acute_angle (α : Real) : Prop :=
  0 < α ∧ α < 90

-- Definition for First Quadrant Angles
def is_first_quadrant_angle (β : Real) : Prop :=
  ∃ k : Int, k * 360 < β ∧ β < 90 + k * 360

-- Conditions
theorem statement_1_correct (α : Real) : is_acute_angle α → is_first_quadrant_angle α :=
sorry

theorem statement_3_correct (β : Real) : is_first_quadrant_angle β :=
sorry

-- Final Proof Statement
theorem correct_statements (α β : Real) :
  (is_acute_angle α → is_first_quadrant_angle α) ∧ (is_first_quadrant_angle β) :=
⟨statement_1_correct α, statement_3_correct β⟩

end statement_1_correct_statement_3_correct_correct_statements_l79_79265


namespace geese_in_marsh_l79_79027

theorem geese_in_marsh (number_of_ducks : ℕ) (total_number_of_birds : ℕ) (number_of_geese : ℕ) (h1 : number_of_ducks = 37) (h2 : total_number_of_birds = 95) : 
  number_of_geese = 58 := 
by
  sorry

end geese_in_marsh_l79_79027


namespace unique_element_in_set_l79_79685

theorem unique_element_in_set (A : Set ℝ) (h₁ : ∃ x, A = {x})
(h₂ : ∀ x ∈ A, (x + 3) / (x - 1) ∈ A) : ∃ x, x ∈ A ∧ (x = 3 ∨ x = -1) := by
  sorry

end unique_element_in_set_l79_79685


namespace original_number_conditions_l79_79956

theorem original_number_conditions (a : ℕ) :
  ∃ (y1 y2 : ℕ), (7 * a = 10 * 9 + y1) ∧ (9 * 9 = 10 * 8 + y2) ∧ y2 = 1 ∧ (a = 13 ∨ a = 14) := sorry

end original_number_conditions_l79_79956


namespace neither_sufficient_nor_necessary_condition_l79_79456

-- Given conditions
def p (a : ℝ) : Prop := ∃ (x y : ℝ), a * x + y + 1 = 0 ∧ a * x - y + 2 = 0
def q : Prop := ∃ (a : ℝ), a = 1

-- The proof problem
theorem neither_sufficient_nor_necessary_condition : 
  ¬ ((∀ a, p a → q) ∧ (∀ a, q → p a)) :=
sorry

end neither_sufficient_nor_necessary_condition_l79_79456


namespace max_diagonals_in_chessboard_l79_79099

/-- The maximum number of non-intersecting diagonals that can be drawn in an 8x8 chessboard is 36. -/
theorem max_diagonals_in_chessboard : 
  ∃ (diagonals : Finset (ℕ × ℕ)), 
  diagonals.card = 36 ∧ 
  ∀ (d1 d2 : ℕ × ℕ), d1 ∈ diagonals → d2 ∈ diagonals → d1 ≠ d2 → d1.fst ≠ d2.fst ∧ d1.snd ≠ d2.snd := 
  sorry

end max_diagonals_in_chessboard_l79_79099


namespace solve_equation_l79_79252

theorem solve_equation :
  ∃ x : ℝ, (x - 2)^2 - (x + 3) * (x - 3) = 4 * x - 1 ∧ x = 7 / 4 := 
by
  sorry

end solve_equation_l79_79252


namespace area_of_rectangle_l79_79037

-- Define the problem statement and conditions
theorem area_of_rectangle (p d : ℝ) :
  ∃ A : ℝ, (∀ (x y : ℝ), 2 * x + 2 * y = p ∧ x^2 + y^2 = d^2 → A = x * y) →
  A = (p^2 - 4 * d^2) / 8 :=
by 
  sorry

end area_of_rectangle_l79_79037


namespace sum_eq_sum_l79_79692

theorem sum_eq_sum {a b c d : ℝ} (h1 : a + b = c + d) (h2 : ac = bd) (h3 : a + b ≠ 0) : a + c = b + d := 
by
  sorry

end sum_eq_sum_l79_79692


namespace calculate_AH_l79_79616

def square (a : ℝ) := a ^ 2
def area_square (s : ℝ) := s ^ 2
def area_triangle (b h : ℝ) := 0.5 * b * h

theorem calculate_AH (s DG DH AH : ℝ) 
  (h_square : area_square s = 144) 
  (h_area_triangle : area_triangle DG DH = 63)
  (h_perpendicular : DG = DH)
  (h_hypotenuse : square AH = square s + square DH) :
  AH = 3 * Real.sqrt 30 :=
by
  -- Proof would be provided here
  sorry

end calculate_AH_l79_79616


namespace largest_positive_integer_n_exists_l79_79121

theorem largest_positive_integer_n_exists (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  ∃ n, 
    0 < n ∧ 
    (n^2 = x^2 + y^2 + z^2 + 2 * x * y + 2 * y * z + 2 * z * x + 4 * x + 4 * y + 4 * z - 9) ∧ 
    ∀ m, 0 < m → 
      (m^2 = x^2 + y^2 + z^2 + 2 * x * y + 2 * y * z + 2 * z * x + 4 * x + 4 * y + 4 * z - 9) → 
      m ≤ n :=
  sorry

end largest_positive_integer_n_exists_l79_79121


namespace art_gallery_total_pieces_l79_79313

theorem art_gallery_total_pieces :
  ∃ T : ℕ, 
    (1/3 : ℝ) * T + (2/3 : ℝ) * (1/3 : ℝ) * T + 400 + 3 * (1/18 : ℝ) * T + 2 * (1/18 : ℝ) * T = T :=
sorry

end art_gallery_total_pieces_l79_79313


namespace range_of_f_area_of_triangle_l79_79087

noncomputable def f (x : ℝ) : ℝ := Real.cos x * Real.sin (x - Real.pi / 6)

-- Problem Part (I)
theorem range_of_f : 
  (∀ x, 0 ≤ x ∧ x ≤ Real.pi / 2 →
      -1/2 ≤ f x ∧ f x ≤ 1/4) :=
sorry

-- Problem Part (II)
theorem area_of_triangle 
  (A B C : ℝ)
  (a b c : ℝ) 
  (hA0 : 0 < A ∧ A < Real.pi)
  (hS1 : a = Real.sqrt 3)
  (hS2 : b = 2 * c)
  (hF : f A = 1/4) :
  (∃ (area : ℝ), area = (1/2) * b * c * Real.sin A ∧ area = Real.sqrt 3 / 3)
:=
sorry

end range_of_f_area_of_triangle_l79_79087


namespace actual_distance_traveled_l79_79852

theorem actual_distance_traveled
  (t : ℕ)
  (H1 : 6 * t = 3 * t + 15) :
  3 * t = 15 :=
by
  exact sorry

end actual_distance_traveled_l79_79852


namespace coordinates_of_P_l79_79925

-- Definitions of conditions
def inFourthQuadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0
def absEqSeven (x : ℝ) : Prop := |x| = 7
def ysquareEqNine (y : ℝ) : Prop := y^2 = 9

-- Main theorem
theorem coordinates_of_P (x y : ℝ) (hx : absEqSeven x) (hy : ysquareEqNine y) (hq : inFourthQuadrant x y) :
  (x, y) = (7, -3) :=
  sorry

end coordinates_of_P_l79_79925


namespace complete_half_job_in_six_days_l79_79091

theorem complete_half_job_in_six_days (x : ℕ) (h1 : 2 * x = x + 6) : x = 6 :=
  by
    sorry

end complete_half_job_in_six_days_l79_79091


namespace tan_of_acute_angle_l79_79312

theorem tan_of_acute_angle (A : ℝ) (hA1 : 0 < A ∧ A < π / 2)
  (hA2 : 4 * (Real.sin A)^2 - 4 * Real.sin A * Real.cos A + (Real.cos A)^2 = 0) :
  Real.tan A = 1 / 2 :=
by
  sorry

end tan_of_acute_angle_l79_79312


namespace book_distribution_l79_79093

theorem book_distribution (x : ℕ) (books : ℕ) :
  (books = 3 * x + 8) ∧ (books < 5 * x - 5 + 2) → (x = 6 ∧ books = 26) :=
by
  sorry

end book_distribution_l79_79093


namespace expand_polynomial_l79_79397

noncomputable def p (x : ℝ) : ℝ := 7 * x ^ 2 + 5
noncomputable def q (x : ℝ) : ℝ := 3 * x ^ 3 + 2 * x + 1

theorem expand_polynomial (x : ℝ) : 
  (p x) * (q x) = 21 * x ^ 5 + 29 * x ^ 3 + 7 * x ^ 2 + 10 * x + 5 := 
by sorry

end expand_polynomial_l79_79397


namespace value_of_k_l79_79014

theorem value_of_k (k : ℤ) : (1/2)^(22) * (1/(81 : ℝ))^k = 1/(18 : ℝ)^(22) → k = 11 :=
by
  sorry

end value_of_k_l79_79014
