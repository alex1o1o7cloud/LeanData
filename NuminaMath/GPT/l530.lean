import Mathlib

namespace sleep_hours_l530_53069

-- Define the times Isaac wakes up, goes to sleep, and takes naps
def monday : ℝ := 16 - 9
def tuesday_night : ℝ := 12 - 6.5
def tuesday_nap : ℝ := 1
def wednesday : ℝ := 9.75 - 7.75
def thursday_night : ℝ := 15.5 - 8
def thursday_nap : ℝ := 1.5
def friday : ℝ := 12 - 7.25
def saturday : ℝ := 12.75 - 9
def sunday_night : ℝ := 10.5 - 8.5
def sunday_nap : ℝ := 2

noncomputable def total_sleep : ℝ := 
  monday +
  (tuesday_night + tuesday_nap) +
  wednesday +
  (thursday_night + thursday_nap) +
  friday +
  saturday +
  (sunday_night + sunday_nap)

theorem sleep_hours (total_sleep : ℝ) : total_sleep = 36.75 := 
by
  -- Here, you would provide the steps used to add up the hours, but we will skip with sorry
  sorry

end sleep_hours_l530_53069


namespace books_in_special_collection_l530_53089

theorem books_in_special_collection (B : ℕ) :
  (∃ returned not_returned loaned_out_end  : ℝ, 
    loaned_out_end = 54 ∧ 
    returned = 0.65 * 60.00000000000001 ∧ 
    not_returned = 60.00000000000001 - returned ∧ 
    B = loaned_out_end + not_returned) → 
  B = 75 :=
by 
  intro h
  sorry

end books_in_special_collection_l530_53089


namespace EFGH_perimeter_l530_53090

noncomputable def perimeter_rectangle_EFGH (WE EX WY XZ : ℕ) : Rat :=
  let WX := Real.sqrt (WE ^ 2 + EX ^ 2)
  let p := 15232
  let q := 100
  p / q

theorem EFGH_perimeter :
  let WE := 12
  let EX := 16
  let WY := 24
  let XZ := 32
  perimeter_rectangle_EFGH WE EX WY XZ = 15232 / 100 :=
by
  sorry

end EFGH_perimeter_l530_53090


namespace num_zeros_in_binary_l530_53068

namespace BinaryZeros

def expression : ℕ := ((18 * 8192 + 8 * 128 - 12 * 16) / 6) + (4 * 64) + (3 ^ 5) - (25 * 2)

def binary_zeros (n : ℕ) : ℕ :=
  (Nat.digits 2 n).count 0

theorem num_zeros_in_binary :
  binary_zeros expression = 6 :=
by
  sorry

end BinaryZeros

end num_zeros_in_binary_l530_53068


namespace number_of_dice_l530_53091

theorem number_of_dice (n : ℕ) (h : (1 / 6 : ℝ) ^ (n - 1) = 0.0007716049382716049) : n = 5 :=
sorry

end number_of_dice_l530_53091


namespace sin_theta_value_l530_53075

open Real

theorem sin_theta_value
  (θ : ℝ)
  (h1 : θ ∈ Set.Ioo (3 * π / 4) (5 * π / 4))
  (h2 : sin (θ - π / 4) = 5 / 13) :
  sin θ = - (7 * sqrt 2) / 26 :=
  sorry

end sin_theta_value_l530_53075


namespace least_number_to_subtract_997_l530_53096

theorem least_number_to_subtract_997 (x : ℕ) (h : x = 997) 
  : ∃ y : ℕ, ∀ m (h₁ : m = (997 - y)), 
    m % 5 = 3 ∧ m % 9 = 3 ∧ m % 11 = 3 ∧ y = 4 :=
by
  -- Proof omitted
  sorry

end least_number_to_subtract_997_l530_53096


namespace solution_set_of_inequality_l530_53003

theorem solution_set_of_inequality
  (f : ℝ → ℝ)
  (h_decreasing : ∀ x y, x < y → f x > f y)
  (hA : f 0 = -2)
  (hB : f (-3) = 2) :
  { x : ℝ | |f (x - 2)| > 2 } = { x : ℝ | x < -1 } ∪ { x : ℝ | x > 2 } :=
by
  sorry

end solution_set_of_inequality_l530_53003


namespace candy_in_each_bag_l530_53073

theorem candy_in_each_bag (total_candy : ℕ) (bags : ℕ) (h1 : total_candy = 16) (h2 : bags = 2) : total_candy / bags = 8 :=
by {
    sorry
}

end candy_in_each_bag_l530_53073


namespace book_total_pages_l530_53044

theorem book_total_pages (x : ℝ) 
  (h1 : ∀ d1 : ℝ, d1 = x * (1/6) + 10)
  (h2 : ∀ remaining1 : ℝ, remaining1 = x - d1)
  (h3 : ∀ d2 : ℝ, d2 = remaining1 * (1/5) + 12)
  (h4 : ∀ remaining2 : ℝ, remaining2 = remaining1 - d2)
  (h5 : ∀ d3 : ℝ, d3 = remaining2 * (1/4) + 14)
  (h6 : ∀ remaining3 : ℝ, remaining3 = remaining2 - d3)
  (h7 : remaining3 = 52) : x = 169 := sorry

end book_total_pages_l530_53044


namespace tangent_line_circle_l530_53065

theorem tangent_line_circle (m : ℝ) (h : m > 0) : 
  (∀ x y : ℝ, x + y = 2 ↔ x^2 + y^2 = m) → m = 2 :=
by
  intro h_tangent
  sorry

end tangent_line_circle_l530_53065


namespace total_students_l530_53057

/-- Definition of the problem's conditions as Lean statements -/
def left_col := 8
def right_col := 14
def front_row := 7
def back_row := 15

/-- The total number of columns calculated from Eunji's column positions -/
def total_columns := left_col + right_col - 1
/-- The total number of rows calculated from Eunji's row positions -/
def total_rows := front_row + back_row - 1

/-- Lean statement showing the total number of students given the conditions -/
theorem total_students : total_columns * total_rows = 441 := by
  sorry

end total_students_l530_53057


namespace initial_population_l530_53015

theorem initial_population (P : ℝ) (h : P * 1.21 = 12000) : P = 12000 / 1.21 :=
by sorry

end initial_population_l530_53015


namespace probability_prime_factor_of_120_l530_53076

open Nat

theorem probability_prime_factor_of_120 : 
  let s := Finset.range 61
  let primes := {2, 3, 5}
  let prime_factors_of_5_fact := primes ∩ s
  (prime_factors_of_5_fact.card : ℚ) / s.card = 1 / 20 :=
by
  sorry

end probability_prime_factor_of_120_l530_53076


namespace binders_required_l530_53083

variables (b1 b2 B1 B2 d1 d2 b3 : ℕ)

def binding_rate_per_binder_per_day : ℚ := B1 / (↑b1 * d1)

def books_per_binder_in_d2_days : ℚ := binding_rate_per_binder_per_day b1 B1 d1 * ↑d2

def binding_rate_for_b2_binders : ℚ := B2 / ↑b2

theorem binders_required (b1 b2 B1 B2 d1 d2 b3 : ℕ)
  (h1 : binding_rate_per_binder_per_day b1 B1 d1 = binding_rate_for_b2_binders b2 B2)
  (h2 : books_per_binder_in_d2_days b1 B1 d1 d2 = binding_rate_for_b2_binders b2 B2) :
  b3 = b2 :=
sorry

end binders_required_l530_53083


namespace solve_fractional_equation_for_c_l530_53040

theorem solve_fractional_equation_for_c :
  (∃ c : ℝ, (c - 37) / 3 = (3 * c + 7) / 8) → c = -317 := by
sorry

end solve_fractional_equation_for_c_l530_53040


namespace sum_of_special_integers_l530_53063

theorem sum_of_special_integers :
  let a := 0
  let b := 1
  let c := -1
  a + b + c = 0 := by
  sorry

end sum_of_special_integers_l530_53063


namespace solution_set_l530_53027

-- Define the two conditions as hypotheses
variables (x : ℝ)

def condition1 : Prop := x + 6 ≤ 8
def condition2 : Prop := x - 7 < 2 * (x - 3)

-- The statement to prove
theorem solution_set (h1 : condition1 x) (h2 : condition2 x) : -1 < x ∧ x ≤ 2 :=
by
  sorry

end solution_set_l530_53027


namespace depth_of_canal_l530_53016

/-- The cross-section of a canal is a trapezium with a top width of 12 meters, 
a bottom width of 8 meters, and an area of 840 square meters. 
Prove that the depth of the canal is 84 meters.
-/
theorem depth_of_canal (top_width bottom_width area : ℝ) (h : ℝ) :
  top_width = 12 → bottom_width = 8 → area = 840 → 1 / 2 * (top_width + bottom_width) * h = area → h = 84 :=
by
  intros ht hb ha h_area
  sorry

end depth_of_canal_l530_53016


namespace area_of_L_shape_is_58_l530_53093

-- Define the dimensions of the large rectangle
def large_rectangle_length : ℕ := 10
def large_rectangle_width : ℕ := 7

-- Define the dimensions of the smaller rectangle to be removed
def small_rectangle_length : ℕ := 4
def small_rectangle_width : ℕ := 3

-- Define the area of the large rectangle
def area_large_rectangle : ℕ := large_rectangle_length * large_rectangle_width

-- Define the area of the small rectangle
def area_small_rectangle : ℕ := small_rectangle_length * small_rectangle_width

-- Define the area of the "L" shaped region
def area_L_shape : ℕ := area_large_rectangle - area_small_rectangle

-- Prove that the area of the "L" shaped region is 58 square units
theorem area_of_L_shape_is_58 : area_L_shape = 58 := by
  sorry

end area_of_L_shape_is_58_l530_53093


namespace net_displacement_east_of_A_total_fuel_consumed_l530_53094

def distances : List Int := [22, -3, 4, -2, -8, -17, -2, 12, 7, -5]
def fuel_consumption_per_km : ℝ := 0.07

theorem net_displacement_east_of_A :
  List.sum distances = 8 := by
  sorry

theorem total_fuel_consumed :
  List.sum (distances.map Int.natAbs) * fuel_consumption_per_km = 5.74 := by
  sorry

end net_displacement_east_of_A_total_fuel_consumed_l530_53094


namespace solve_abs_eqn_l530_53024

theorem solve_abs_eqn (y : ℝ) : (|y - 4| + 3 * y = 11) ↔ (y = 3.5) := by
  sorry

end solve_abs_eqn_l530_53024


namespace total_wheels_in_parking_lot_l530_53025

-- Definitions (conditions)
def cars := 14
def wheels_per_car := 4
def missing_wheels_per_missing_car := 1
def missing_cars := 2

def bikes := 5
def wheels_per_bike := 2

def unicycles := 3
def wheels_per_unicycle := 1

def twelve_wheeler_trucks := 2
def wheels_per_twelve_wheeler_truck := 12
def damaged_wheels_per_twelve_wheeler_truck := 3
def damaged_twelve_wheeler_trucks := 1

def eighteen_wheeler_trucks := 1
def wheels_per_eighteen_wheeler_truck := 18

-- The total wheels calculation proof
theorem total_wheels_in_parking_lot :
  ((cars * wheels_per_car - missing_cars * missing_wheels_per_missing_car) +
   (bikes * wheels_per_bike) +
   (unicycles * wheels_per_unicycle) +
   (twelve_wheeler_trucks * wheels_per_twelve_wheeler_truck - damaged_twelve_wheeler_trucks * damaged_wheels_per_twelve_wheeler_truck) +
   (eighteen_wheeler_trucks * wheels_per_eighteen_wheeler_truck)) = 106 := by
  sorry

end total_wheels_in_parking_lot_l530_53025


namespace man_speed_in_still_water_l530_53048

theorem man_speed_in_still_water 
  (V_u : ℕ) (V_d : ℕ) 
  (hu : V_u = 34) 
  (hd : V_d = 48) : 
  V_s = (V_u + V_d) / 2 :=
by
  sorry

end man_speed_in_still_water_l530_53048


namespace positive_integers_sum_of_squares_l530_53085

theorem positive_integers_sum_of_squares
  (a b c d : ℤ)
  (h1 : a^2 + b^2 + c^2 + d^2 = 90)
  (h2 : a + b + c + d = 16) :
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d := 
by
  sorry

end positive_integers_sum_of_squares_l530_53085


namespace sufficient_but_not_necessary_l530_53047

theorem sufficient_but_not_necessary (x : ℝ) :
  (x^2 > 1) → (1 / x < 1) ∧ ¬(1 / x < 1 → x^2 > 1) :=
by
  sorry

end sufficient_but_not_necessary_l530_53047


namespace problem_solution_l530_53074

theorem problem_solution
  (a1 a2 a3: ℝ)
  (a_arith_seq : ∃ d, a1 = 1 + d ∧ a2 = a1 + d ∧ a3 = a2 + d ∧ 9 = a3 + d)
  (b1 b2 b3: ℝ)
  (b_geo_seq : ∃ r, r > 0 ∧ b1 = -9 * r ∧ b2 = b1 * r ∧ b3 = b2 * r ∧ -1 = b3 * r) :
  (b2 / (a1 + a3) = -3 / 10) :=
by
  -- Placeholder for the proof, not required in this context
  sorry

end problem_solution_l530_53074


namespace base_b_arithmetic_l530_53060

theorem base_b_arithmetic (b : ℕ) (h1 : 4 + 3 = 7) (h2 : 6 + 2 = 8) (h3 : 4 + 6 = 10) (h4 : 3 + 4 + 1 = 8) : b = 9 :=
  sorry

end base_b_arithmetic_l530_53060


namespace avg_salary_increases_by_150_l530_53006

def avg_salary_increase
  (emp_avg_salary : ℕ) (num_employees : ℕ) (mgr_salary : ℕ) : ℕ :=
  let total_salary_employees := emp_avg_salary * num_employees
  let total_salary_with_mgr := total_salary_employees + mgr_salary
  let new_avg_salary := total_salary_with_mgr / (num_employees + 1)
  new_avg_salary - emp_avg_salary

theorem avg_salary_increases_by_150 :
  avg_salary_increase 1800 15 4200 = 150 :=
by
  sorry

end avg_salary_increases_by_150_l530_53006


namespace union_complement_l530_53028

-- Definitions of the sets
def U : Set ℕ := {x | x > 0 ∧ x ≤ 5}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {1, 2, 4}

-- Statement of the proof problem
theorem union_complement : A ∪ (U \ B) = {1, 3, 5} := by
  sorry

end union_complement_l530_53028


namespace cube_less_than_three_times_square_l530_53009

theorem cube_less_than_three_times_square (x : ℤ) : x^3 < 3 * x^2 → x = 1 ∨ x = 2 :=
by
  sorry

end cube_less_than_three_times_square_l530_53009


namespace quadratic_solution_l530_53061

theorem quadratic_solution (x : ℝ) (h : 2 * x ^ 2 - 2 = 0) : x = 1 ∨ x = -1 :=
sorry

end quadratic_solution_l530_53061


namespace range_of_a_l530_53037

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - a * x + 2 * a > 0) ↔ (0 < a ∧ a < 8) :=
by
  sorry

end range_of_a_l530_53037


namespace algebraic_expression_correct_l530_53018

theorem algebraic_expression_correct (a b : ℝ) (h : a = 7 - 3 * b) : a^2 + 6 * a * b + 9 * b^2 = 49 := 
by sorry

end algebraic_expression_correct_l530_53018


namespace sum_of_all_x_l530_53050

theorem sum_of_all_x (x1 x2 : ℝ) (h1 : (x1 + 5)^2 = 81) (h2 : (x2 + 5)^2 = 81) : x1 + x2 = -10 :=
by
  sorry

end sum_of_all_x_l530_53050


namespace digit_for_multiple_of_9_l530_53022

theorem digit_for_multiple_of_9 (d : ℕ) : (23450 + d) % 9 = 0 ↔ d = 4 := by
  sorry

end digit_for_multiple_of_9_l530_53022


namespace pure_imaginary_k_l530_53008

theorem pure_imaginary_k (k : ℝ) :
  (2 * k^2 - 3 * k - 2 = 0) → (k^2 - 2 * k ≠ 0) → k = -1 / 2 :=
by
  intro hr hi
  -- Proof will go here.
  sorry

end pure_imaginary_k_l530_53008


namespace mangoes_rate_l530_53005

theorem mangoes_rate (grapes_weight mangoes_weight total_amount grapes_rate mango_rate : ℕ)
  (h1 : grapes_weight = 7)
  (h2 : grapes_rate = 68)
  (h3 : total_amount = 908)
  (h4 : mangoes_weight = 9)
  (h5 : total_amount - grapes_weight * grapes_rate = mangoes_weight * mango_rate) :
  mango_rate = 48 :=
by
  sorry

end mangoes_rate_l530_53005


namespace points_per_game_l530_53007

theorem points_per_game (total_points games : ℕ) (h1 : total_points = 91) (h2 : games = 13) :
  total_points / games = 7 :=
by
  sorry

end points_per_game_l530_53007


namespace prism_faces_l530_53095

theorem prism_faces (E V F : ℕ) (n : ℕ) 
  (h1 : E + V = 40) 
  (h2 : E = 3 * F - 6) 
  (h3 : V - E + F = 2)
  (h4 : V = 2 * n)
  : F = 10 := 
by
  sorry

end prism_faces_l530_53095


namespace initial_mixture_volume_l530_53053

variable (p q : ℕ) (x : ℕ)

theorem initial_mixture_volume :
  (3 * x) + (2 * x) = 5 * x →
  (3 * x) / (2 * x + 12) = 3 / 4 →
  5 * x = 30 :=
by
  sorry

end initial_mixture_volume_l530_53053


namespace john_weight_loss_percentage_l530_53086

def john_initial_weight := 220
def john_final_weight_after_gain := 200
def weight_gain := 2

theorem john_weight_loss_percentage : 
  ∃ P : ℝ, (john_initial_weight - (P / 100) * john_initial_weight + weight_gain = john_final_weight_after_gain) ∧ P = 10 :=
sorry

end john_weight_loss_percentage_l530_53086


namespace initial_pieces_of_fruit_l530_53004

-- Definitions for the given problem
def pieces_eaten_in_first_four_days : ℕ := 5
def pieces_kept_for_next_week : ℕ := 2
def pieces_brought_to_school : ℕ := 3

-- Problem statement
theorem initial_pieces_of_fruit 
  (pieces_eaten : ℕ)
  (pieces_kept : ℕ)
  (pieces_brought : ℕ)
  (h1 : pieces_eaten = pieces_eaten_in_first_four_days)
  (h2 : pieces_kept = pieces_kept_for_next_week)
  (h3 : pieces_brought = pieces_brought_to_school) :
  pieces_eaten + pieces_kept + pieces_brought = 10 := 
sorry

end initial_pieces_of_fruit_l530_53004


namespace students_with_all_three_pets_l530_53054

variable (x y z : ℕ)
variable (total_students : ℕ := 40)
variable (dog_students : ℕ := total_students * 5 / 8)
variable (cat_students : ℕ := total_students * 1 / 4)
variable (other_students : ℕ := 8)
variable (no_pet_students : ℕ := 6)
variable (only_dog_students : ℕ := 12)
variable (only_other_students : ℕ := 3)
variable (cat_other_no_dog_students : ℕ := 10)

theorem students_with_all_three_pets :
  (x + y + z + 10 + 3 + 12 = total_students - no_pet_students) →
  (x + z + 10 = dog_students) →
  (10 + z = cat_students) →
  (y + z + 10 = other_students) →
  z = 0 :=
by
  -- Provide proof here
  sorry

end students_with_all_three_pets_l530_53054


namespace hours_to_seconds_l530_53002

theorem hours_to_seconds : 
  (3.5 * 60 * 60) = 12600 := 
by 
  sorry

end hours_to_seconds_l530_53002


namespace line_through_point_perpendicular_l530_53023

theorem line_through_point_perpendicular :
  ∃ (a b : ℝ), ∀ (x : ℝ), y = - (3 / 2) * x + 8 ∧ y - 2 = - (3 / 2) * (x - 4) ∧ 2*x - 3*y = 6 → y = - (3 / 2) * x + 8 :=
by 
  sorry

end line_through_point_perpendicular_l530_53023


namespace maximum_possible_shortest_piece_length_l530_53012

theorem maximum_possible_shortest_piece_length :
  ∃ (A B C D E : ℝ), A ≤ B ∧ B ≤ C ∧ C ≤ D ∧ D ≤ E ∧ 
  C = 140 ∧ (A + B + C + D + E = 640) ∧ A = 80 :=
by
  sorry

end maximum_possible_shortest_piece_length_l530_53012


namespace Zilla_savings_l530_53034

/-- Zilla's monthly savings based on her spending distributions -/
theorem Zilla_savings
  (rent : ℚ) (monthly_earnings_percentage : ℚ)
  (other_expenses_fraction : ℚ) (monthly_rent : ℚ)
  (monthly_expenses : ℚ) (total_monthly_earnings : ℚ)
  (half_monthly_earnings : ℚ) (savings : ℚ)
  (h1 : rent = 133)
  (h2 : monthly_earnings_percentage = 0.07)
  (h3 : other_expenses_fraction = 0.5)
  (h4 : total_monthly_earnings = monthly_rent / monthly_earnings_percentage)
  (h5 : half_monthly_earnings = total_monthly_earnings * other_expenses_fraction)
  (h6 : savings = total_monthly_earnings - (monthly_rent + half_monthly_earnings))
  : savings = 817 :=
sorry

end Zilla_savings_l530_53034


namespace sum_divisible_by_5_and_7_remainder_12_l530_53077

theorem sum_divisible_by_5_and_7_remainder_12 :
  let a := 105
  let d := 35
  let n := 2013
  let S := (n * (2 * a + (n - 1) * d)) / 2
  S % 12 = 3 :=
by
  sorry

end sum_divisible_by_5_and_7_remainder_12_l530_53077


namespace bus_remaining_distance_l530_53019

noncomputable def final_distance (z x : ℝ) : ℝ :=
  z - (z * x / 5)

theorem bus_remaining_distance (z : ℝ) :
  (z / 2) / (z - 19.2) = x ∧ (z - 12) / (z / 2) = x → final_distance z x = 6.4 :=
by
  intro h
  sorry

end bus_remaining_distance_l530_53019


namespace smallest_integer_cube_ends_in_576_l530_53062

theorem smallest_integer_cube_ends_in_576 : ∃ n : ℕ, n > 0 ∧ n^3 % 1000 = 576 ∧ ∀ m : ℕ, m > 0 → m^3 % 1000 = 576 → m ≥ n := 
by
  sorry

end smallest_integer_cube_ends_in_576_l530_53062


namespace p_plus_q_eq_10_l530_53035

theorem p_plus_q_eq_10 (p q : ℕ) (hp : p > q) (hpq1 : p < 10) (hpq2 : q < 10)
  (h : p.factorial / q.factorial = 840) : p + q = 10 :=
by
  sorry

end p_plus_q_eq_10_l530_53035


namespace min_value_expression_min_value_expression_achieved_at_1_l530_53020

noncomputable def min_value_expr (a b : ℝ) (n : ℕ) : ℝ :=
  (1 / (1 + a^n)) + (1 / (1 + b^n))

theorem min_value_expression (a b : ℝ) (n : ℕ) (h1 : a + b = 2) (h2 : 0 < a) (h3 : 0 < b) : 
  (min_value_expr a b n) ≥ 1 :=
sorry

theorem min_value_expression_achieved_at_1 (n : ℕ) :
  (min_value_expr 1 1 n = 1) :=
sorry

end min_value_expression_min_value_expression_achieved_at_1_l530_53020


namespace rectangle_circles_l530_53055

theorem rectangle_circles (p q : Prop) (hp : p) (hq : ¬ q) : p ∨ q :=
by sorry

end rectangle_circles_l530_53055


namespace value_of_y_l530_53038

theorem value_of_y (x y : ℝ) (h1 : x * y = 9) (h2 : x / y = 36) : y = 1 / 2 :=
by
  sorry

end value_of_y_l530_53038


namespace rectangle_width_l530_53087

theorem rectangle_width (L W : ℝ) 
  (h1 : L * W = 300)
  (h2 : 2 * L + 2 * W = 70) : 
  W = 15 :=
by 
  -- We prove the width W of the rectangle is 15 meters.
  sorry

end rectangle_width_l530_53087


namespace pool_capacity_l530_53043

theorem pool_capacity (C : ℝ) (h1 : C * 0.70 = C * 0.40 + 300)
  (h2 : 300 = C * 0.30) : C = 1000 :=
sorry

end pool_capacity_l530_53043


namespace percentage_unloaded_at_second_store_l530_53021

theorem percentage_unloaded_at_second_store
  (initial_weight : ℝ)
  (percent_unloaded_first : ℝ)
  (remaining_weight_after_deliveries : ℝ)
  (remaining_weight_after_first : ℝ)
  (weight_unloaded_second : ℝ)
  (percent_unloaded_second : ℝ) :
  initial_weight = 50000 →
  percent_unloaded_first = 0.10 →
  remaining_weight_after_deliveries = 36000 →
  remaining_weight_after_first = initial_weight * (1 - percent_unloaded_first) →
  weight_unloaded_second = remaining_weight_after_first - remaining_weight_after_deliveries →
  percent_unloaded_second = (weight_unloaded_second / remaining_weight_after_first) * 100 →
  percent_unloaded_second = 20 :=
by
  intros _
  sorry

end percentage_unloaded_at_second_store_l530_53021


namespace identify_quadratic_equation_l530_53046

/-- Proving which equation is a quadratic equation from given options -/
def is_quadratic_equation (eq : String) : Prop :=
  eq = "sqrt(x^2)=2" ∨ eq = "x^2 - x - 2" ∨ eq = "1/x^2 - 2=0" ∨ eq = "x^2=0"

theorem identify_quadratic_equation :
  ∀ (eq : String), is_quadratic_equation eq → eq = "x^2=0" :=
by
  intro eq h
  -- add proof steps here
  sorry

end identify_quadratic_equation_l530_53046


namespace verify_digits_l530_53045

theorem verify_digits :
  ∀ (a b c d e f g h : ℕ), 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧
  d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧
  e ≠ f ∧ e ≠ g ∧ e ≠ h ∧
  f ≠ g ∧ f ≠ h ∧
  g ≠ h ∧
  a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧ f < 10 ∧ g < 10 ∧ h < 10 →
  (10 * a + b) - (10 * c + d) = 10 * e + d →
  e * f = 10 * d + c →
  (10 * g + d) + (10 * g + b) = 10 * h + c →
  a = 9 ∧ b = 8 ∧ c = 2 ∧ d = 4 ∧ e = 7 ∧ f = 6 ∧ g = 1 ∧ h = 3 :=
by
  intros a b c d e f g h
  intros h1 h2 h3
  sorry

end verify_digits_l530_53045


namespace tree_initial_height_l530_53088

theorem tree_initial_height (H : ℝ) (C : ℝ) (P : H + 6 = (H + 4) + 1/4 * (H + 4) ∧ C = 1) : H = 4 :=
by
  let H := 4
  sorry

end tree_initial_height_l530_53088


namespace find_constants_l530_53080

theorem find_constants
  (k m n : ℝ)
  (h : -x^3 + (k + 7) * x^2 + m * x - 8 = -(x - 2) * (x - 4) * (x - n)) :
  k = 7 ∧ m = 2 ∧ n = 1 :=
sorry

end find_constants_l530_53080


namespace exists_positive_integer_m_such_that_sqrt_8m_is_integer_l530_53017

theorem exists_positive_integer_m_such_that_sqrt_8m_is_integer :
  ∃ (m : ℕ), m > 0 ∧ ∃ (k : ℕ), 8 * m = k^2 :=
by
  sorry

end exists_positive_integer_m_such_that_sqrt_8m_is_integer_l530_53017


namespace common_term_sequence_7n_l530_53032

theorem common_term_sequence_7n (n : ℕ) : 
  ∃ a_n : ℕ, a_n = (7 / 9) * (10^n - 1) :=
by
  sorry

end common_term_sequence_7n_l530_53032


namespace leo_weight_proof_l530_53066

def Leo_s_current_weight (L K : ℝ) := 
  L + 10 = 1.5 * K ∧ L + K = 170 → L = 98

theorem leo_weight_proof : ∀ (L K : ℝ), L + 10 = 1.5 * K ∧ L + K = 170 → L = 98 := 
by 
  intros L K h
  sorry

end leo_weight_proof_l530_53066


namespace roots_imply_sum_l530_53079

theorem roots_imply_sum (a b c x1 x2 : ℝ) (hneq : a ≠ 0) (hroots : a * x1 ^ 2 + b * x1 + c = 0 ∧ a * x2 ^ 2 + b * x2 + c = 0) :
  x1 + x2 = -b / a :=
sorry

end roots_imply_sum_l530_53079


namespace remy_water_usage_l530_53030

theorem remy_water_usage :
  ∃ R : ℕ, (Remy = 3 * R + 1) ∧ 
    (Riley = R + (3 * R + 1) - 2) ∧ 
    (R + (3 * R + 1) + (R + (3 * R + 1) - 2) = 48) ∧ 
    (Remy = 19) :=
sorry

end remy_water_usage_l530_53030


namespace estimated_red_balls_l530_53052

theorem estimated_red_balls
  (total_balls : ℕ)
  (total_draws : ℕ)
  (red_draws : ℕ)
  (h_total_balls : total_balls = 12)
  (h_total_draws : total_draws = 200)
  (h_red_draws : red_draws = 50) :
  red_draws * total_balls = total_draws * 3 :=
by
  sorry

end estimated_red_balls_l530_53052


namespace engineer_thought_of_l530_53092

def isProperDivisor (n k : ℕ) : Prop :=
  k ≠ 1 ∧ k ≠ n ∧ k ∣ n

def transformDivisors (n m : ℕ) : Prop :=
  ∀ k, isProperDivisor n k → isProperDivisor m (k + 1)

theorem engineer_thought_of (n : ℕ) :
  (∀ m : ℕ, n = 2^2 ∨ n = 2^3 → transformDivisors n m → (m % 2 = 1)) :=
by
  sorry

end engineer_thought_of_l530_53092


namespace ironed_clothing_count_l530_53011

theorem ironed_clothing_count : 
  (4 * 2 + 5 * 3) + (3 * 3 + 4 * 2) + (2 * 1 + 3 * 1) = 45 := by
  sorry

end ironed_clothing_count_l530_53011


namespace systematic_sampling_first_two_numbers_l530_53033

theorem systematic_sampling_first_two_numbers
  (sample_size : ℕ) (population_size : ℕ) (last_sample_number : ℕ)
  (h1 : sample_size = 50) (h2 : population_size = 8000) (h3 : last_sample_number = 7900) :
  ∃ first second : ℕ, first = 60 ∧ second = 220 :=
by
  -- Proof to be provided.
  sorry

end systematic_sampling_first_two_numbers_l530_53033


namespace totalGoals_l530_53036

-- Define the conditions
def louieLastMatchGoals : Nat := 4
def louiePreviousGoals : Nat := 40
def gamesPerSeason : Nat := 50
def seasons : Nat := 3
def brotherGoalsPerGame := 2 * louieLastMatchGoals

-- Define the properties derived from the conditions
def totalBrotherGoals : Nat := brotherGoalsPerGame * gamesPerSeason * seasons
def totalLouieGoals : Nat := louiePreviousGoals + louieLastMatchGoals

-- State what needs to be proved
theorem totalGoals : louiePreviousGoals + louieLastMatchGoals + brotherGoalsPerGame * gamesPerSeason * seasons = 1244 := by
  sorry

end totalGoals_l530_53036


namespace cube_coloring_schemes_l530_53039

theorem cube_coloring_schemes (colors : Finset ℕ) (h : colors.card = 6) :
  ∃ schemes : Nat, schemes = 230 :=
by
  sorry

end cube_coloring_schemes_l530_53039


namespace repeating_decimal_sum_l530_53059

open Real

noncomputable def repeating_decimal_to_fraction (d: ℕ) : ℚ :=
  if d = 3 then 1/3 else if d = 7 then 7/99 else if d = 9 then 1/111 else 0 -- specific case of 3, 7, 9.

theorem repeating_decimal_sum:
  let x := repeating_decimal_to_fraction 3
  let y := repeating_decimal_to_fraction 7
  let z := repeating_decimal_to_fraction 9
  x + y + z = 499 / 1189 :=
by
  sorry -- Proof is omitted

end repeating_decimal_sum_l530_53059


namespace dhoni_dishwasher_spending_l530_53099

noncomputable def percentage_difference : ℝ := 0.25 - 0.225
noncomputable def percentage_less_than : ℝ := (percentage_difference / 0.25) * 100

theorem dhoni_dishwasher_spending :
  (percentage_difference / 0.25) * 100 = 10 :=
by sorry

end dhoni_dishwasher_spending_l530_53099


namespace largest_common_value_l530_53071

theorem largest_common_value (a : ℕ) (h1 : a % 4 = 3) (h2 : a % 9 = 5) (h3 : a < 600) :
  a = 599 :=
sorry

end largest_common_value_l530_53071


namespace volleyball_team_lineup_l530_53013

theorem volleyball_team_lineup : 
  let team_members := 10
  let lineup_positions := 6
  10 * 9 * 8 * 7 * 6 * 5 = 151200 := by sorry

end volleyball_team_lineup_l530_53013


namespace gcd_f_100_f_101_l530_53097

def f (x : ℕ) : ℕ := x^2 - 2*x + 2023

theorem gcd_f_100_f_101 : Nat.gcd (f 100) (f 101) = 1 := by
  sorry

end gcd_f_100_f_101_l530_53097


namespace apples_total_l530_53042

theorem apples_total
    (cecile_apples : ℕ := 15)
    (diane_apples_more : ℕ := 20) :
    (cecile_apples + (cecile_apples + diane_apples_more)) = 50 :=
by
  sorry

end apples_total_l530_53042


namespace number_of_mango_trees_l530_53010

-- Define the conditions
variable (M : Nat) -- Number of mango trees
def num_papaya_trees := 2
def papayas_per_tree := 10
def mangos_per_tree := 20
def total_fruits := 80

-- Prove that the number of mango trees M is equal to 3
theorem number_of_mango_trees : 20 + (mangos_per_tree * M) = total_fruits -> M = 3 :=
by
  intro h
  sorry

end number_of_mango_trees_l530_53010


namespace prove_q_l530_53070

theorem prove_q 
  (p q : ℝ)
  (h : (∀ x, (x + 3) * (x + p) = x^2 + q * x + 12)) : 
  q = 7 :=
sorry

end prove_q_l530_53070


namespace sum_of_other_endpoint_coordinates_l530_53072

/-- 
  Given that (9, -15) is the midpoint of the segment with one endpoint (7, 4),
  find the sum of the coordinates of the other endpoint.
-/
theorem sum_of_other_endpoint_coordinates : 
  ∃ x y : ℤ, ((7 + x) / 2 = 9 ∧ (4 + y) / 2 = -15) ∧ (x + y = -23) :=
by
  sorry

end sum_of_other_endpoint_coordinates_l530_53072


namespace total_distance_covered_l530_53029

-- Define the distances for each segment of Biker Bob's journey
def distance1 : ℕ := 45 -- 45 miles west
def distance2 : ℕ := 25 -- 25 miles northwest
def distance3 : ℕ := 35 -- 35 miles south
def distance4 : ℕ := 50 -- 50 miles east

-- Statement to prove that the total distance covered is 155 miles
theorem total_distance_covered : distance1 + distance2 + distance3 + distance4 = 155 :=
by
  -- This is where the proof would go
  sorry

end total_distance_covered_l530_53029


namespace find_f_7_l530_53098

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function : ∀ x : ℝ, f (-x) = -f x
axiom function_period : ∀ x : ℝ, f (x + 2) = -f x
axiom function_value_range : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = x

theorem find_f_7 : f 7 = -1 := by
  sorry

end find_f_7_l530_53098


namespace find_star_l530_53049

theorem find_star :
  ∃ (star : ℤ), 45 - ( 28 - ( 37 - ( 15 - star ) ) ) = 56 ∧ star = 17 :=
by
  sorry

end find_star_l530_53049


namespace probability_of_unique_color_and_number_l530_53067

-- Defining the sets of colors and numbers
inductive Color
| red
| yellow
| blue

inductive Number
| one
| two
| three

-- Defining a ball as a combination of a Color and a Number
structure Ball :=
(color : Color)
(number : Number)

-- Setting up the list of 9 balls
def allBalls : List Ball :=
  [⟨Color.red, Number.one⟩, ⟨Color.red, Number.two⟩, ⟨Color.red, Number.three⟩,
   ⟨Color.yellow, Number.one⟩, ⟨Color.yellow, Number.two⟩, ⟨Color.yellow, Number.three⟩,
   ⟨Color.blue, Number.one⟩, ⟨Color.blue, Number.two⟩, ⟨Color.blue, Number.three⟩]

-- Proving the probability calculation as a theorem
noncomputable def probability_neither_same_color_nor_number : ℕ → ℕ → ℚ :=
  λ favorable total => favorable / total

theorem probability_of_unique_color_and_number :
  probability_neither_same_color_nor_number
    (6) -- favorable outcomes
    (84) -- total outcomes
  = 1 / 14 := by
  sorry

end probability_of_unique_color_and_number_l530_53067


namespace find_x_l530_53078

-- Definitions of the conditions in Lean 4
def angle_sum_180 (A B C : ℝ) : Prop := A + B + C = 180
def angle_BAC_eq_90 (A : ℝ) : Prop := A = 90
def angle_BCA_eq_2x (C x : ℝ) : Prop := C = 2 * x
def angle_ABC_eq_3x (B x : ℝ) : Prop := B = 3 * x

-- The theorem we need to prove
theorem find_x (A B C x : ℝ) 
  (h1 : angle_sum_180 A B C) 
  (h2 : angle_BAC_eq_90 A)
  (h3 : angle_BCA_eq_2x C x) 
  (h4 : angle_ABC_eq_3x B x) : x = 18 :=
by 
  sorry

end find_x_l530_53078


namespace store_loss_l530_53058

noncomputable def calculation (x y : ℕ) : ℤ :=
  let revenue : ℕ := 60 * 2
  let cost : ℕ := x + y
  revenue - cost

theorem store_loss (x y : ℕ) (hx : (60 - x) * 2 = x) (hy : (y - 60) * 2 = y) :
  calculation x y = -40 := by
    sorry

end store_loss_l530_53058


namespace total_pay_is_880_l530_53051

theorem total_pay_is_880 (X_pay Y_pay : ℝ) 
  (hY : Y_pay = 400)
  (hX : X_pay = 1.2 * Y_pay):
  X_pay + Y_pay = 880 :=
by
  sorry

end total_pay_is_880_l530_53051


namespace hcf_of_two_numbers_l530_53064

theorem hcf_of_two_numbers (A B : ℕ) (h1 : Nat.lcm A B = 750) (h2 : A * B = 18750) : Nat.gcd A B = 25 :=
by
  sorry

end hcf_of_two_numbers_l530_53064


namespace minimum_value_of_expression_l530_53026

theorem minimum_value_of_expression {a b : ℝ} (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + b = 2) : 
  (1 / (2 * a) + 1 / b) ≥ (3 + 2 * Real.sqrt 2) / 4 := 
sorry

end minimum_value_of_expression_l530_53026


namespace problem1_l530_53014

noncomputable def sqrt7_minus_1_pow_0 : ℝ := (Real.sqrt 7 - 1)^0
noncomputable def minus_half_pow_neg_2 : ℝ := (-1 / 2)^(-2 : ℤ)
noncomputable def sqrt3_tan_30 : ℝ := Real.sqrt 3 * Real.tan (Real.pi / 6)

theorem problem1 : sqrt7_minus_1_pow_0 - minus_half_pow_neg_2 + sqrt3_tan_30 = -2 := by
  sorry

end problem1_l530_53014


namespace r_at_6_l530_53001

-- Define the monic quintic polynomial r(x) with given conditions
def r (x : ℝ) : ℝ :=
  (x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 5) + x^2 + 2 

-- Given conditions
axiom r_1 : r 1 = 3
axiom r_2 : r 2 = 7
axiom r_3 : r 3 = 13
axiom r_4 : r 4 = 21
axiom r_5 : r 5 = 31

-- Proof goal
theorem r_at_6 : r 6 = 158 :=
by
  sorry

end r_at_6_l530_53001


namespace domain_of_func_1_domain_of_func_2_domain_of_func_3_domain_of_func_4_l530_53031
-- Import the necessary library.

-- Define the domains for the given functions.
def domain_func_1 (x : ℝ) : Prop := true

def domain_func_2 (x : ℝ) : Prop := 1 ≤ x ∧ x ≤ 2

def domain_func_3 (x : ℝ) : Prop := x ≥ -3 ∧ x ≠ 1

def domain_func_4 (x : ℝ) : Prop := 2 ≤ x ∧ x ≤ 5 ∧ x ≠ 3

-- Prove the domains of each function.
theorem domain_of_func_1 : ∀ x : ℝ, domain_func_1 x :=
by sorry

theorem domain_of_func_2 : ∀ x : ℝ, domain_func_2 x ↔ (1 ≤ x ∧ x ≤ 2) :=
by sorry

theorem domain_of_func_3 : ∀ x : ℝ, domain_func_3 x ↔ (x ≥ -3 ∧ x ≠ 1) :=
by sorry

theorem domain_of_func_4 : ∀ x : ℝ, domain_func_4 x ↔ (2 ≤ x ∧ x ≤ 5 ∧ x ≠ 3) :=
by sorry

end domain_of_func_1_domain_of_func_2_domain_of_func_3_domain_of_func_4_l530_53031


namespace acrobat_eq_two_lambs_l530_53056

variables (ACROBAT DOG BARREL SPOOL LAMB : ℝ)

axiom acrobat_dog_eq_two_barrels : ACROBAT + DOG = 2 * BARREL
axiom dog_eq_two_spools : DOG = 2 * SPOOL
axiom lamb_spool_eq_barrel : LAMB + SPOOL = BARREL

theorem acrobat_eq_two_lambs : ACROBAT = 2 * LAMB :=
by
  sorry

end acrobat_eq_two_lambs_l530_53056


namespace minimum_toothpicks_for_5_squares_l530_53082

theorem minimum_toothpicks_for_5_squares :
  let single_square_toothpicks := 4
  let additional_shared_side_toothpicks := 3
  ∃ n, n = single_square_toothpicks + 4 * additional_shared_side_toothpicks ∧ n = 15 :=
by
  sorry

end minimum_toothpicks_for_5_squares_l530_53082


namespace sum_of_roots_l530_53081

-- Define the quadratic equation
def quadratic_eq (a b c x : ℝ) : Prop :=
  a * x^2 + b * x + c = 0

-- Prove that the sum of the roots of the given quadratic equation is 6
theorem sum_of_roots :
  (quadratic_eq 1 (-6) 9) x → (quadratic_eq 1 (-6) 9) y → x ≠ y → x + y = 6 :=
by
  sorry

end sum_of_roots_l530_53081


namespace area_triangle_le_quarter_l530_53000

theorem area_triangle_le_quarter (S : ℝ) (S₁ S₂ S₃ S₄ S₅ S₆ S₇ : ℝ)
  (h₁ : S₃ + (S₂ + S₇) = S / 2)
  (h₂ : S₁ + S₆ + (S₂ + S₇) = S / 2) :
  S₁ ≤ S / 4 :=
by
  -- Proof skipped
  sorry

end area_triangle_le_quarter_l530_53000


namespace find_s_l530_53041

theorem find_s (s t : ℝ) (h1 : 8 * s + 4 * t = 160) (h2 : t = 2 * s - 3) : s = 10.75 :=
by
  sorry

end find_s_l530_53041


namespace points_player_1_after_13_rotations_l530_53084

variable (table : List ℕ) (players : Fin 16 → ℕ)

axiom round_rotating_table : table = [0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1]
axiom points_player_5 : players 5 = 72
axiom points_player_9 : players 9 = 84

theorem points_player_1_after_13_rotations : players 1 = 20 := 
  sorry

end points_player_1_after_13_rotations_l530_53084
