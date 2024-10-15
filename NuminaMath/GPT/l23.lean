import Mathlib

namespace NUMINAMATH_GPT_sum_of_all_possible_x_l23_2314

theorem sum_of_all_possible_x : 
  (∀ x : ℝ, |x - 5| - 4 = -1 → (x = 8 ∨ x = 2)) → ( ∃ (x1 x2 : ℝ), (x1 = 8) ∧ (x2 = 2) ∧ (x1 + x2 = 10) ) :=
by
  admit

end NUMINAMATH_GPT_sum_of_all_possible_x_l23_2314


namespace NUMINAMATH_GPT_marys_next_birthday_l23_2385

noncomputable def calculate_marys_age (d j s m TotalAge : ℝ) (H1 : j = 1.15 * d) (H2 : s = 1.30 * d) (H3 : m = 1.25 * s) (H4 : j + d + s + m = TotalAge) : ℝ :=
  m + 1

theorem marys_next_birthday (d j s m TotalAge : ℝ) 
  (H1 : j = 1.15 * d)
  (H2 : s = 1.30 * d)
  (H3 : m = 1.25 * s)
  (H4 : j + d + s + m = TotalAge)
  (H5 : TotalAge = 80) :
  calculate_marys_age d j s m TotalAge H1 H2 H3 H4 = 26 :=
sorry

end NUMINAMATH_GPT_marys_next_birthday_l23_2385


namespace NUMINAMATH_GPT_larger_integer_is_21_l23_2368

theorem larger_integer_is_21 (a b : ℕ) (h₀ : 0 < b) (h₁ : a / b = 7 / 3) (h₂ : a * b = 189) : a = 21 ∨ b = 21 :=
by
  sorry

end NUMINAMATH_GPT_larger_integer_is_21_l23_2368


namespace NUMINAMATH_GPT_units_digit_of_result_is_3_l23_2389

def hundreds_digit_relation (c : ℕ) (a : ℕ) : Prop :=
  a = 2 * c - 3

def original_number_expression (a b c : ℕ) : ℕ :=
  100 * a + 10 * b + c

def reversed_number_expression (a b c : ℕ) : ℕ :=
  100 * c + 10 * b + a + 50

def subtraction_result (orig rev : ℕ) : ℕ :=
  orig - rev

def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_of_result_is_3 (a b c : ℕ) (h : hundreds_digit_relation c a) :
  units_digit (subtraction_result (original_number_expression a b c)
                                  (reversed_number_expression a b c)) = 3 :=
by
  sorry

end NUMINAMATH_GPT_units_digit_of_result_is_3_l23_2389


namespace NUMINAMATH_GPT_remainder_12401_163_l23_2370

theorem remainder_12401_163 :
  let original_number := 12401
  let divisor := 163
  let quotient := 76
  let remainder := 13
  original_number = divisor * quotient + remainder :=
by
  sorry

end NUMINAMATH_GPT_remainder_12401_163_l23_2370


namespace NUMINAMATH_GPT_no_integer_solutions_for_square_polynomial_l23_2321

theorem no_integer_solutions_for_square_polynomial :
  (∀ x : ℤ, ∃ k : ℤ, k^2 = x^4 + 5*x^3 + 10*x^2 + 5*x + 25 → false) :=
by
  sorry

end NUMINAMATH_GPT_no_integer_solutions_for_square_polynomial_l23_2321


namespace NUMINAMATH_GPT_g_4_minus_g_7_l23_2326

theorem g_4_minus_g_7 (g : ℝ → ℝ) (h_linear : ∀ x y : ℝ, g (x + y) = g x + g y)
  (h_diff : ∀ k : ℝ, g (k + 1) - g k = 5) : g 4 - g 7 = -15 :=
by
  sorry

end NUMINAMATH_GPT_g_4_minus_g_7_l23_2326


namespace NUMINAMATH_GPT_wxyz_sum_l23_2327

noncomputable def wxyz (w x y z : ℕ) := 2^w * 3^x * 5^y * 7^z

theorem wxyz_sum (w x y z : ℕ) (h : wxyz w x y z = 1260) : w + 2 * x + 3 * y + 4 * z = 13 :=
sorry

end NUMINAMATH_GPT_wxyz_sum_l23_2327


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l23_2354

theorem necessary_but_not_sufficient (x : ℝ) : (1 - x) * (1 + |x|) > 0 -> x < 2 :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l23_2354


namespace NUMINAMATH_GPT_central_angle_of_sector_l23_2363

theorem central_angle_of_sector (r α : ℝ) (h_arc_length : α * r = 5) (h_area : 0.5 * α * r^2 = 5): α = 5 / 2 := by
  sorry

end NUMINAMATH_GPT_central_angle_of_sector_l23_2363


namespace NUMINAMATH_GPT_total_cable_cost_neighborhood_l23_2305

-- Define the number of east-west streets and their length
def ew_streets : ℕ := 18
def ew_length_per_street : ℕ := 2

-- Define the number of north-south streets and their length
def ns_streets : ℕ := 10
def ns_length_per_street : ℕ := 4

-- Define the cable requirements and cost
def cable_per_mile_of_street : ℕ := 5
def cable_cost_per_mile : ℕ := 2000

-- Calculate total length of east-west streets
def ew_total_length : ℕ := ew_streets * ew_length_per_street

-- Calculate total length of north-south streets
def ns_total_length : ℕ := ns_streets * ns_length_per_street

-- Calculate total length of all streets
def total_street_length : ℕ := ew_total_length + ns_total_length

-- Calculate total length of cable required
def total_cable_length : ℕ := total_street_length * cable_per_mile_of_street

-- Calculate total cost of the cable
def total_cost : ℕ := total_cable_length * cable_cost_per_mile

-- The statement to prove
theorem total_cable_cost_neighborhood : total_cost = 760000 :=
by
  sorry

end NUMINAMATH_GPT_total_cable_cost_neighborhood_l23_2305


namespace NUMINAMATH_GPT_tips_fraction_of_income_l23_2340

theorem tips_fraction_of_income
  (S T : ℝ)
  (h1 : T = (2 / 4) * S) :
  T / (S + T) = 1 / 3 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_tips_fraction_of_income_l23_2340


namespace NUMINAMATH_GPT_tan_60_eq_sqrt3_l23_2382

theorem tan_60_eq_sqrt3 : Real.tan (Real.pi / 3) = Real.sqrt 3 := 
sorry

end NUMINAMATH_GPT_tan_60_eq_sqrt3_l23_2382


namespace NUMINAMATH_GPT_clock_angle_7_35_l23_2338

theorem clock_angle_7_35 : 
  let minute_hand_angle := (35 / 60) * 360
  let hour_hand_angle := 7 * 30 + (35 / 60) * 30
  let angle_between := hour_hand_angle - minute_hand_angle
  angle_between = 17.5 := by
sorry

end NUMINAMATH_GPT_clock_angle_7_35_l23_2338


namespace NUMINAMATH_GPT_represent_1947_as_squares_any_integer_as_squares_l23_2356

theorem represent_1947_as_squares :
  ∃ (a b c : ℤ), 1947 = a * a - b * b - c * c :=
by
  use 488, 486, 1
  sorry

theorem any_integer_as_squares (n : ℤ) :
  ∃ (a b c d : ℤ), n = a * a + b * b + c * c + d * d :=
by
  sorry

end NUMINAMATH_GPT_represent_1947_as_squares_any_integer_as_squares_l23_2356


namespace NUMINAMATH_GPT_length_of_string_for_circle_l23_2341

theorem length_of_string_for_circle (A : ℝ) (pi_approx : ℝ) (extra_length : ℝ) (hA : A = 616) (hpi : pi_approx = 22 / 7) (hextra : extra_length = 5) :
  ∃ (length : ℝ), length = 93 :=
by {
  sorry
}

end NUMINAMATH_GPT_length_of_string_for_circle_l23_2341


namespace NUMINAMATH_GPT_xiao_ming_kite_payment_l23_2386

/-- Xiao Ming has multiple 1 yuan, 2 yuan, and 5 yuan banknotes. 
    He wants to buy a kite priced at 18 yuan using no more than 10 of these banknotes
    and must use at least two different denominations.
    Prove that there are exactly 11 different ways he can pay. -/
theorem xiao_ming_kite_payment : 
  ∃ (combinations : Nat), 
    (∀ (c1 c2 c5 : Nat), (c1 * 1 + c2 * 2 + c5 * 5 = 18) → 
    (c1 + c2 + c5 ≤ 10) → 
    ((c1 > 0 ∧ c2 > 0) ∨ (c1 > 0 ∧ c5 > 0) ∨ (c2 > 0 ∧ c5 > 0)) →
    combinations = 11) :=
sorry

end NUMINAMATH_GPT_xiao_ming_kite_payment_l23_2386


namespace NUMINAMATH_GPT_length_YW_l23_2345

-- Definitions of the sides of the triangle
def XY := 6
def YZ := 8
def XZ := 10

-- The total perimeter of triangle XYZ
def perimeter : ℕ := XY + YZ + XZ

-- Each ant travels half the perimeter
def halfPerimeter : ℕ := perimeter / 2

-- Distance one ant travels from X to W through Y
def distanceXtoW : ℕ := XY + 6

-- Prove that the distance segment YW is 6
theorem length_YW : distanceXtoW = halfPerimeter := by sorry

end NUMINAMATH_GPT_length_YW_l23_2345


namespace NUMINAMATH_GPT_sqrt_expression_evaluation_l23_2310

theorem sqrt_expression_evaluation :
  (Real.sqrt 48 - 6 * Real.sqrt (1 / 3) - Real.sqrt 18 / Real.sqrt 6) = Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_expression_evaluation_l23_2310


namespace NUMINAMATH_GPT_g_1_5_l23_2309

noncomputable def g (x : ℝ) : ℝ := sorry

axiom g_defined (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) : g x ≠ 0

axiom g_zero : g 0 = 0

axiom g_mono (x y : ℝ) (hx : 0 ≤ x ∧ x < y ∧ y ≤ 1) : g x ≤ g y

axiom g_symmetry (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) : g (1 - x) = 1 - g x

axiom g_scaling (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) : g (x/4) = g x / 2

theorem g_1_5 : g (1 / 5) = 1 / 4 := 
sorry

end NUMINAMATH_GPT_g_1_5_l23_2309


namespace NUMINAMATH_GPT_Lin_peels_15_potatoes_l23_2357

-- Define the conditions
def total_potatoes : Nat := 60
def homer_rate : Nat := 2 -- potatoes per minute
def christen_rate : Nat := 3 -- potatoes per minute
def lin_rate : Nat := 4 -- potatoes per minute
def christen_join_time : Nat := 6 -- minutes
def lin_join_time : Nat := 9 -- minutes

-- Prove that Lin peels 15 potatoes
theorem Lin_peels_15_potatoes :
  ∃ (lin_potatoes : Nat), lin_potatoes = 15 :=
by
  sorry

end NUMINAMATH_GPT_Lin_peels_15_potatoes_l23_2357


namespace NUMINAMATH_GPT_determine_k_l23_2379

theorem determine_k (k : ℝ) :
  (∀ x : ℝ, (x - 3) * (x - 5) = k - 4 * x) ↔ k = 11 :=
by
  sorry

end NUMINAMATH_GPT_determine_k_l23_2379


namespace NUMINAMATH_GPT_hardcover_volumes_l23_2348

theorem hardcover_volumes (h p : ℕ) (h1 : h + p = 10) (h2 : 25 * h + 15 * p = 220) : h = 7 :=
by sorry

end NUMINAMATH_GPT_hardcover_volumes_l23_2348


namespace NUMINAMATH_GPT_marbles_exchange_l23_2392

-- Define the initial number of marbles for Drew and Marcus
variables {D M x : ℕ}

-- Conditions
axiom Drew_initial (D M : ℕ) : D = M + 24
axiom Drew_after_give (D x : ℕ) : D - x = 25
axiom Marcus_after_receive (M x : ℕ) : M + x = 25

-- The goal is to prove: x = 12
theorem marbles_exchange : ∀ {D M x : ℕ}, D = M + 24 ∧ D - x = 25 ∧ M + x = 25 → x = 12 :=
by 
    sorry

end NUMINAMATH_GPT_marbles_exchange_l23_2392


namespace NUMINAMATH_GPT_problem1_problem2_l23_2390

-- Problem 1
theorem problem1 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a + b + c = 1) :
  a * b + b * c + c * a ≤ 1 / 3 :=
sorry

-- Problem 2
theorem problem2 (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a ≥ b) :
  2 * a ^ 3 - b ^ 3 ≥ 2 * a * b ^ 2 - a ^ 2 * b :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l23_2390


namespace NUMINAMATH_GPT_alternate_interior_angles_equal_l23_2377

-- Defining the parallel lines and the third intersecting line
def Line : Type := sorry  -- placeholder type for a line

-- Predicate to check if lines are parallel
def parallel (l1 l2 : Line) : Prop := sorry

-- Predicate to represent a line intersecting another
def intersects (l1 l2 : Line) : Prop := sorry

-- Function to get interior alternate angles formed by the intersection
def alternate_interior_angles (l1 l2 : Line) (l3 : Line) : Prop := sorry

-- Theorem statement
theorem alternate_interior_angles_equal
  (l1 l2 l3 : Line)
  (h1 : parallel l1 l2)
  (h2 : intersects l3 l1)
  (h3 : intersects l3 l2) :
  alternate_interior_angles l1 l2 l3 :=
sorry

end NUMINAMATH_GPT_alternate_interior_angles_equal_l23_2377


namespace NUMINAMATH_GPT_apples_per_pie_l23_2329

theorem apples_per_pie (total_apples : ℕ) (unripe_apples : ℕ) (pies : ℕ) (ripe_apples : ℕ)
  (H1 : total_apples = 34)
  (H2 : unripe_apples = 6)
  (H3 : pies = 7)
  (H4 : ripe_apples = total_apples - unripe_apples) :
  ripe_apples / pies = 4 := by
  sorry

end NUMINAMATH_GPT_apples_per_pie_l23_2329


namespace NUMINAMATH_GPT_homework_total_l23_2300

theorem homework_total :
  let math_pages := 20
  let reading_pages := math_pages - (30 * math_pages / 100)
  let science_pages := 2 * reading_pages
  math_pages + reading_pages + science_pages = 62 :=
by
  let math_pages := 20
  let reading_pages := math_pages - (30 * math_pages / 100)
  let science_pages := 2 * reading_pages
  show math_pages + reading_pages + science_pages = 62
  sorry

end NUMINAMATH_GPT_homework_total_l23_2300


namespace NUMINAMATH_GPT_cone_volume_l23_2339

theorem cone_volume (r h : ℝ) (π : ℝ) (V : ℝ) :
    r = 3 → h = 4 → π = Real.pi → V = (1/3) * π * r^2 * h → V = 37.68 :=
by
  sorry

end NUMINAMATH_GPT_cone_volume_l23_2339


namespace NUMINAMATH_GPT_simplify_polynomial_l23_2335

variable (y : ℤ)

theorem simplify_polynomial :
  (3 * y - 2) * (5 * y^12 + 3 * y^11 + 6 * y^10 + 2 * y^9 + 4) = 
  15 * y^13 - y^12 + 12 * y^11 - 6 * y^10 - 4 * y^9 + 12 * y - 8 :=
by
  sorry

end NUMINAMATH_GPT_simplify_polynomial_l23_2335


namespace NUMINAMATH_GPT_andrea_fewer_apples_l23_2320

theorem andrea_fewer_apples {total_apples given_to_zenny kept_by_yanna given_to_andrea : ℕ} 
  (h1 : total_apples = 60) 
  (h2 : given_to_zenny = 18) 
  (h3 : kept_by_yanna = 36) 
  (h4 : given_to_andrea = total_apples - kept_by_yanna - given_to_zenny) : 
  (given_to_andrea + 12 = given_to_zenny) := 
sorry

end NUMINAMATH_GPT_andrea_fewer_apples_l23_2320


namespace NUMINAMATH_GPT_equation1_solution_equation2_solution_l23_2319

-- Equation 1: x^2 + 2x - 8 = 0 has solutions x = -4 and x = 2.
theorem equation1_solution (x : ℝ) : x^2 + 2 * x - 8 = 0 ↔ x = -4 ∨ x = 2 := by
  sorry

-- Equation 2: 2(x+3)^2 = x(x+3) has solutions x = -3 and x = -6.
theorem equation2_solution (x : ℝ) : 2 * (x + 3)^2 = x * (x + 3) ↔ x = -3 ∨ x = -6 := by
  sorry

end NUMINAMATH_GPT_equation1_solution_equation2_solution_l23_2319


namespace NUMINAMATH_GPT_find_distance_between_PQ_l23_2322

-- Defining distances and speeds
def distance_by_first_train (t : ℝ) : ℝ := 50 * t
def distance_by_second_train (t : ℝ) : ℝ := 40 * t
def distance_between_PQ (t : ℝ) : ℝ := distance_by_first_train t + (distance_by_first_train t - 100)

-- Main theorem stating the problem
theorem find_distance_between_PQ : ∃ t : ℝ, distance_by_first_train t - distance_by_second_train t = 100 ∧ distance_between_PQ t = 900 := 
sorry

end NUMINAMATH_GPT_find_distance_between_PQ_l23_2322


namespace NUMINAMATH_GPT_initial_pipes_count_l23_2391

theorem initial_pipes_count (n r : ℝ) 
  (h1 : n * r = 1 / 12) 
  (h2 : (n + 10) * r = 1 / 4) : 
  n = 5 := 
by 
  sorry

end NUMINAMATH_GPT_initial_pipes_count_l23_2391


namespace NUMINAMATH_GPT_alice_score_record_l23_2337

def total_points : ℝ := 72
def average_points_others : ℝ := 4.7
def others_count : ℕ := 7

def total_points_others : ℝ := others_count * average_points_others
def alice_points : ℝ := total_points - total_points_others

theorem alice_score_record : alice_points = 39.1 :=
by {
  -- Proof should be inserted here
  sorry
}

end NUMINAMATH_GPT_alice_score_record_l23_2337


namespace NUMINAMATH_GPT_reciprocal_of_neg_2023_l23_2306

theorem reciprocal_of_neg_2023 : ∀ x : ℝ, x = -2023 → (1 / x = - (1 / 2023)) :=
by
  intros x hx
  rw [hx]
  sorry

end NUMINAMATH_GPT_reciprocal_of_neg_2023_l23_2306


namespace NUMINAMATH_GPT_average_salary_rest_workers_l23_2315

-- Define the conditions
def total_workers : Nat := 21
def average_salary_all_workers : ℝ := 8000
def number_of_technicians : Nat := 7
def average_salary_technicians : ℝ := 12000

-- Define the task
theorem average_salary_rest_workers :
  let number_of_rest := total_workers - number_of_technicians
  let total_salary_all := average_salary_all_workers * total_workers
  let total_salary_technicians := average_salary_technicians * number_of_technicians
  let total_salary_rest := total_salary_all - total_salary_technicians
  let average_salary_rest := total_salary_rest / number_of_rest
  average_salary_rest = 6000 :=
by
  sorry

end NUMINAMATH_GPT_average_salary_rest_workers_l23_2315


namespace NUMINAMATH_GPT_crabapple_recipients_sequences_l23_2381

-- Define the number of students in Mrs. Crabapple's class
def num_students : ℕ := 12

-- Define the number of class meetings per week
def num_meetings : ℕ := 5

-- Define the total number of different sequences
def total_sequences : ℕ := num_students ^ num_meetings

-- The target theorem to prove
theorem crabapple_recipients_sequences :
  total_sequences = 248832 := by
  sorry

end NUMINAMATH_GPT_crabapple_recipients_sequences_l23_2381


namespace NUMINAMATH_GPT_pigeon_problem_l23_2397

theorem pigeon_problem (x y : ℕ) :
  (1 / 6 : ℝ) * (x + y) = y - 1 ∧ x - 1 = y + 1 → x = 4 ∧ y = 2 :=
by
  sorry

end NUMINAMATH_GPT_pigeon_problem_l23_2397


namespace NUMINAMATH_GPT_sequence_sum_l23_2388

open BigOperators

-- Define the general term
def term (n : ℕ) : ℚ := n * (1 - (1 / n))

-- Define the index range for the sequence
def index_range : Finset ℕ := Finset.range 9 \ {0, 1}

-- Lean statement of the problem
theorem sequence_sum : ∑ n in index_range, term (n + 2) = 45 := by
  sorry

end NUMINAMATH_GPT_sequence_sum_l23_2388


namespace NUMINAMATH_GPT_select_pairs_eq_l23_2369

open Set

-- Definitions for sets A and B
def A : Set ℕ := {1, 2}
def B : Set ℕ := {1, 2, 3}

-- Statement of the theorem
theorem select_pairs_eq :
  {p | p.1 ∈ A ∧ p.2 ∈ B} = {(1,1), (1,2), (1,3), (2,1), (2,2), (2,3)} :=
by sorry

end NUMINAMATH_GPT_select_pairs_eq_l23_2369


namespace NUMINAMATH_GPT_harry_weekly_earnings_l23_2317

def dogs_walked_per_day : Nat → Nat
| 1 => 7  -- Monday
| 2 => 12 -- Tuesday
| 3 => 7  -- Wednesday
| 4 => 9  -- Thursday
| 5 => 7  -- Friday
| _ => 0  -- Other days (not relevant for this problem)

def payment_per_dog : Nat := 5

def daily_earnings (day : Nat) : Nat :=
  dogs_walked_per_day day * payment_per_dog

def total_weekly_earnings : Nat :=
  (daily_earnings 1) + (daily_earnings 2) + (daily_earnings 3) +
  (daily_earnings 4) + (daily_earnings 5)

theorem harry_weekly_earnings : total_weekly_earnings = 210 :=
by
  sorry

end NUMINAMATH_GPT_harry_weekly_earnings_l23_2317


namespace NUMINAMATH_GPT_closest_to_2010_l23_2387

theorem closest_to_2010 :
  let A := 2008 * 2012
  let B := 1000 * Real.pi
  let C := 58 * 42
  let D := (48.3 ^ 2 - 2 * 8.3 * 48.3 + 8.3 ^ 2)
  abs (2010 - D) < abs (2010 - A) ∧
  abs (2010 - D) < abs (2010 - B) ∧
  abs (2010 - D) < abs (2010 - C) :=
by
  sorry

end NUMINAMATH_GPT_closest_to_2010_l23_2387


namespace NUMINAMATH_GPT_find_f_neg2_l23_2347

-- Condition (1): f is an even function on ℝ
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

-- Condition (2): f(x) = x^2 + 1 for x > 0
def function_defined_for_positive_x {f : ℝ → ℝ} (h_even : even_function f): Prop :=
  ∀ x : ℝ, x > 0 → f x = x^2 + 1

-- Proof problem: prove that given the conditions, f(-2) = 5
theorem find_f_neg2 (f : ℝ → ℝ) 
  (h_even : even_function f) 
  (h_pos : function_defined_for_positive_x h_even) : 
  f (-2) = 5 := 
sorry

end NUMINAMATH_GPT_find_f_neg2_l23_2347


namespace NUMINAMATH_GPT_sum_of_digits_9ab_l23_2312

def a : ℕ := 999
def b : ℕ := 666

theorem sum_of_digits_9ab : 
  let n := 9 * a * b
  (n.digits 10).sum = 36 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_digits_9ab_l23_2312


namespace NUMINAMATH_GPT_smallest_positive_integer_l23_2376

theorem smallest_positive_integer (n : ℕ) :
  (∃ n : ℕ, n > 0 ∧ n % 30 = 0 ∧ n % 40 = 0 ∧ n % 16 ≠ 0 ∧ ∀ m : ℕ, (m > 0 ∧ m % 30 = 0 ∧ m % 40 = 0 ∧ m % 16 ≠ 0) → n ≤ m) ↔ n = 120 :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_integer_l23_2376


namespace NUMINAMATH_GPT_parabola_tangent_y_intercept_correct_l23_2393

noncomputable def parabola_tangent_y_intercept (a : ℝ) : Prop :=
  let C := fun x : ℝ => x^2
  let slope := 2 * a
  let tangent_line := fun x : ℝ => slope * (x - a) + C a
  let Q := (0, tangent_line 0)
  Q = (0, -a^2)

-- Statement of the problem as a Lean theorem
theorem parabola_tangent_y_intercept_correct (a : ℝ) (h : a > 0) :
  parabola_tangent_y_intercept a := 
by 
  sorry

end NUMINAMATH_GPT_parabola_tangent_y_intercept_correct_l23_2393


namespace NUMINAMATH_GPT_sets_of_bleachers_l23_2371

def totalFans : ℕ := 2436
def fansPerSet : ℕ := 812

theorem sets_of_bleachers (n : ℕ) (h : totalFans = n * fansPerSet) : n = 3 :=
by {
    sorry
}

end NUMINAMATH_GPT_sets_of_bleachers_l23_2371


namespace NUMINAMATH_GPT_value_of_8x_minus_5_squared_l23_2316

theorem value_of_8x_minus_5_squared (x : ℝ) (h : 8 * x ^ 2 + 7 = 12 * x + 17) : (8 * x - 5) ^ 2 = 465 := 
sorry

end NUMINAMATH_GPT_value_of_8x_minus_5_squared_l23_2316


namespace NUMINAMATH_GPT_number_of_ways_to_select_books_l23_2318

theorem number_of_ways_to_select_books :
  let bag1 := 4
  let bag2 := 5
  bag1 * bag2 = 20 :=
by
  sorry

end NUMINAMATH_GPT_number_of_ways_to_select_books_l23_2318


namespace NUMINAMATH_GPT_sum_smallest_largest_even_integers_l23_2330

theorem sum_smallest_largest_even_integers (n : ℕ) (h_odd : n % 2 = 1) (b z : ℤ)
  (h_mean : z = b + n - 1) : (b + (b + 2 * (n - 1))) = 2 * z :=
by
  sorry

end NUMINAMATH_GPT_sum_smallest_largest_even_integers_l23_2330


namespace NUMINAMATH_GPT_total_passengers_landed_l23_2361

theorem total_passengers_landed (on_time late : ℕ) (h_on_time : on_time = 14507) (h_late : late = 213) :
  on_time + late = 14720 :=
by
  sorry

end NUMINAMATH_GPT_total_passengers_landed_l23_2361


namespace NUMINAMATH_GPT_find_x_solution_l23_2324

theorem find_x_solution (b x : ℝ) (hb : b > 1) (hx : x > 0) 
    (h_eq : (4 * x)^(Real.log 4 / Real.log b) - (5 * x)^(Real.log 5 / Real.log b) = 0) : 
    x = 1 / 5 :=
by
  sorry

end NUMINAMATH_GPT_find_x_solution_l23_2324


namespace NUMINAMATH_GPT_product_of_fractions_is_27_l23_2383

theorem product_of_fractions_is_27 :
  (1/3) * (9/1) * (1/27) * (81/1) * (1/243) * (729/1) = 27 :=
by
  sorry

end NUMINAMATH_GPT_product_of_fractions_is_27_l23_2383


namespace NUMINAMATH_GPT_explain_education_policy_l23_2350

theorem explain_education_policy :
  ∃ (reason1 reason2 : String), reason1 ≠ reason2 ∧
    (reason1 = "International Agreements: Favorable foreign credit terms or reciprocal educational benefits" ∧
     reason2 = "Addressing Demographic Changes: Attracting educated youth for future economic contributions")
    ∨
    (reason2 = "International Agreements: Favorable foreign credit terms or reciprocal educational benefits" ∧
     reason1 = "Addressing Demographic Changes: Attracting educated youth for future economic contributions") :=
by
  sorry

end NUMINAMATH_GPT_explain_education_policy_l23_2350


namespace NUMINAMATH_GPT_michael_robots_l23_2375

-- Conditions
def tom_robots := 3
def times_more := 4

-- Theorem to prove
theorem michael_robots : (times_more * tom_robots) + tom_robots = 15 := by
  sorry

end NUMINAMATH_GPT_michael_robots_l23_2375


namespace NUMINAMATH_GPT_equation_of_parabola_l23_2366

def parabola_passes_through_point (a h : ℝ) : Prop :=
  2 = a * (8^2) + h

def focus_x_coordinate (a h : ℝ) : Prop :=
  h + (1 / (4 * a)) = 3

theorem equation_of_parabola :
  ∃ (a h : ℝ), parabola_passes_through_point a h ∧ focus_x_coordinate a h ∧
    (∀ x y : ℝ, x = (15 / 256) * y^2 - (381 / 128)) :=
sorry

end NUMINAMATH_GPT_equation_of_parabola_l23_2366


namespace NUMINAMATH_GPT_least_number_of_coins_l23_2399

theorem least_number_of_coins (n : ℕ) : 
  (n % 7 = 3) ∧ (n % 5 = 4) ∧ (∀ m : ℕ, (m % 7 = 3) ∧ (m % 5 = 4) → n ≤ m) → n = 24 :=
by
  sorry

end NUMINAMATH_GPT_least_number_of_coins_l23_2399


namespace NUMINAMATH_GPT_grain_distance_l23_2384

theorem grain_distance
    (d : ℝ) (v_church : ℝ) (v_cathedral : ℝ)
    (h_d : d = 400) (h_v_church : v_church = 20) (h_v_cathedral : v_cathedral = 25) :
    ∃ x : ℝ, x = 1600 / 9 ∧ v_church * x = v_cathedral * (d - x) :=
by
  sorry

end NUMINAMATH_GPT_grain_distance_l23_2384


namespace NUMINAMATH_GPT_square_of_99_is_9801_l23_2325

theorem square_of_99_is_9801 : 99 ^ 2 = 9801 := 
by
  sorry

end NUMINAMATH_GPT_square_of_99_is_9801_l23_2325


namespace NUMINAMATH_GPT_min_a2_b2_c2_l23_2301

theorem min_a2_b2_c2 (a b c : ℕ) (h : a + 2 * b + 3 * c = 73) : a^2 + b^2 + c^2 ≥ 381 :=
by sorry

end NUMINAMATH_GPT_min_a2_b2_c2_l23_2301


namespace NUMINAMATH_GPT_minimal_divisors_at_kth_place_l23_2372

open Nat

theorem minimal_divisors_at_kth_place (n k : ℕ) (hnk : n ≥ k) (S : ℕ) (hS : ∃ d : ℕ, d ≥ n ∧ d = S ∧ ∀ i, i ≤ d → exists m, m = d):
  ∃ (min_div : ℕ), min_div = ⌈ (n : ℝ) / k ⌉ :=
by
  sorry

end NUMINAMATH_GPT_minimal_divisors_at_kth_place_l23_2372


namespace NUMINAMATH_GPT_order_of_A_B_C_D_l23_2333

def A := Nat.factorial 8 ^ Nat.factorial 8
def B := 8 ^ (8 ^ 8)
def C := 8 ^ 88
def D := 8 ^ 64

theorem order_of_A_B_C_D : D < C ∧ C < B ∧ B < A := by
  sorry

end NUMINAMATH_GPT_order_of_A_B_C_D_l23_2333


namespace NUMINAMATH_GPT_unique_prime_pair_l23_2355

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem unique_prime_pair :
  ∀ p : ℕ, is_prime p ∧ is_prime (p + 1) → p = 2 := by
  sorry

end NUMINAMATH_GPT_unique_prime_pair_l23_2355


namespace NUMINAMATH_GPT_unguarded_area_eq_225_l23_2398

-- Define the basic conditions of the problem in Lean
structure Room where
  side_length : ℕ
  unguarded_fraction : ℚ
  deriving Repr

-- Define the specific room used in the problem
def problemRoom : Room :=
  { side_length := 10,
    unguarded_fraction := 9/4 }

-- Define the expected unguarded area in square meters
def expected_unguarded_area (r : Room) : ℚ :=
  r.unguarded_fraction * (r.side_length ^ 2)

-- Prove that the unguarded area is 225 square meters
theorem unguarded_area_eq_225 (r : Room) (h : r = problemRoom) : expected_unguarded_area r = 225 := by
  -- The proof in this case is omitted.
  sorry

end NUMINAMATH_GPT_unguarded_area_eq_225_l23_2398


namespace NUMINAMATH_GPT_tan_45_eq_one_l23_2343

theorem tan_45_eq_one : Real.tan (Real.pi / 4) = 1 := 
by
  sorry

end NUMINAMATH_GPT_tan_45_eq_one_l23_2343


namespace NUMINAMATH_GPT_polynomial_no_real_roots_l23_2359

def f (x : ℝ) : ℝ := 4 * x ^ 8 - 2 * x ^ 7 + x ^ 6 - 3 * x ^ 4 + x ^ 2 - x + 1

theorem polynomial_no_real_roots : ∀ x : ℝ, f x > 0 := by
  sorry

end NUMINAMATH_GPT_polynomial_no_real_roots_l23_2359


namespace NUMINAMATH_GPT_correct_set_of_equations_l23_2334

-- Define the digits x and y as integers
def digits (x y : ℕ) := x + y = 8

-- Conditions
def condition_1 (x y : ℕ) := 10*y + x + 18 = 10*x + y

theorem correct_set_of_equations : 
  ∃ (x y : ℕ), digits x y ∧ condition_1 x y :=
sorry

end NUMINAMATH_GPT_correct_set_of_equations_l23_2334


namespace NUMINAMATH_GPT_sqrt_eq_pm_4_l23_2353

theorem sqrt_eq_pm_4 : {x : ℝ | x * x = 16} = {4, -4} :=
by sorry

end NUMINAMATH_GPT_sqrt_eq_pm_4_l23_2353


namespace NUMINAMATH_GPT_work_ratio_l23_2302

theorem work_ratio 
  (m b : ℝ) 
  (h : 7 * m + 2 * b = 6 * (m + b)) : 
  m / b = 4 := 
sorry

end NUMINAMATH_GPT_work_ratio_l23_2302


namespace NUMINAMATH_GPT_eight_machines_produce_ninety_six_bottles_in_three_minutes_l23_2311

-- Define the initial conditions
def rate_per_machine: ℕ := 16 / 4 -- bottles per minute per machine

def total_bottles_8_machines_3_minutes: ℕ := 8 * rate_per_machine * 3

-- Prove the question
theorem eight_machines_produce_ninety_six_bottles_in_three_minutes:
  total_bottles_8_machines_3_minutes = 96 :=
by
  sorry

end NUMINAMATH_GPT_eight_machines_produce_ninety_six_bottles_in_three_minutes_l23_2311


namespace NUMINAMATH_GPT_find_angle_C_find_area_of_triangle_l23_2378

-- Given triangle ABC with sides a, b, and c opposite to angles A, B, and C respectively
-- And given conditions: c * cos B = (2a - b) * cos C

variable (a b c : ℝ) (A B C : ℝ)
variable (h1 : c * Real.cos B = (2 * a - b) * Real.cos C)
variable (h2 : c = 2)
variable (h3 : a + b + c = 2 * Real.sqrt 3 + 2)

-- Prove that angle C = π / 3
theorem find_angle_C : C = Real.pi / 3 :=
by sorry

-- Given angle C, side c, and perimeter, prove the area of triangle ABC
theorem find_area_of_triangle (h4 : C = Real.pi / 3) : 
  1 / 2 * a * b * Real.sin C = 2 * Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_GPT_find_angle_C_find_area_of_triangle_l23_2378


namespace NUMINAMATH_GPT_geometric_series_first_term_l23_2352

theorem geometric_series_first_term 
  (S : ℝ) (r : ℝ) (a : ℝ)
  (h_sum : S = 40) (h_ratio : r = 1/4) :
  S = a / (1 - r) → a = 30 := by
  sorry

end NUMINAMATH_GPT_geometric_series_first_term_l23_2352


namespace NUMINAMATH_GPT_cos_squared_plus_twice_sin_double_alpha_l23_2380

theorem cos_squared_plus_twice_sin_double_alpha (α : ℝ) (h : Real.tan α = 3 / 4) :
  Real.cos α ^ 2 + 2 * Real.sin (2 * α) = 64 / 25 :=
by
  sorry

end NUMINAMATH_GPT_cos_squared_plus_twice_sin_double_alpha_l23_2380


namespace NUMINAMATH_GPT_arithmetic_geometric_common_ratio_l23_2323

theorem arithmetic_geometric_common_ratio (a₁ r : ℝ) 
  (h₁ : a₁ + a₁ * r^2 = 10) 
  (h₂ : a₁ * (1 + r + r^2 + r^3) = 15) : 
  r = 1/2 ∨ r = -1/2 :=
by {
  sorry
}

end NUMINAMATH_GPT_arithmetic_geometric_common_ratio_l23_2323


namespace NUMINAMATH_GPT_min_value_x_plus_y_l23_2394

theorem min_value_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) 
  (h : (2 * x + Real.sqrt (4 * x^2 + 1)) * (Real.sqrt (y^2 + 4) - 2) ≥ y) : 
  x + y >= 2 := 
by
  sorry

end NUMINAMATH_GPT_min_value_x_plus_y_l23_2394


namespace NUMINAMATH_GPT_slope_condition_l23_2328

theorem slope_condition {m : ℝ} : 
  (4 - m) / (m + 2) = 1 → m = 1 :=
by
  sorry

end NUMINAMATH_GPT_slope_condition_l23_2328


namespace NUMINAMATH_GPT_john_saves_water_l23_2304

-- Define the conditions
def old_water_per_flush : ℕ := 5
def num_flushes_per_day : ℕ := 15
def reduction_percentage : ℕ := 80
def days_in_june : ℕ := 30

-- Define the savings calculation
def water_saved_in_june : ℕ :=
  let old_daily_usage := old_water_per_flush * num_flushes_per_day
  let old_june_usage := old_daily_usage * days_in_june
  let new_water_per_flush := old_water_per_flush * (100 - reduction_percentage) / 100
  let new_daily_usage := new_water_per_flush * num_flushes_per_day
  let new_june_usage := new_daily_usage * days_in_june
  old_june_usage - new_june_usage

-- The proof problem statement
theorem john_saves_water : water_saved_in_june = 1800 := 
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_john_saves_water_l23_2304


namespace NUMINAMATH_GPT_smallest_multiple_of_40_gt_100_l23_2364

theorem smallest_multiple_of_40_gt_100 :
  ∃ x : ℕ, 0 < x ∧ 40 * x > 100 ∧ ∀ y : ℕ, 0 < y ∧ 40 * y > 100 → x ≤ y → 40 * x = 120 :=
by
  sorry

end NUMINAMATH_GPT_smallest_multiple_of_40_gt_100_l23_2364


namespace NUMINAMATH_GPT_two_f_eq_eight_over_four_plus_x_l23_2307

noncomputable def f : ℝ → ℝ := sorry

theorem two_f_eq_eight_over_four_plus_x (f_def : ∀ x > 0, f (2 * x) = 2 / (2 + x)) :
  ∀ x > 0, 2 * f x = 8 / (4 + x) :=
by
  sorry

end NUMINAMATH_GPT_two_f_eq_eight_over_four_plus_x_l23_2307


namespace NUMINAMATH_GPT_min_colors_to_distinguish_keys_l23_2346

def min_colors_needed (n : Nat) : Nat :=
  if n <= 2 then n
  else if n >= 6 then 2
  else 3

theorem min_colors_to_distinguish_keys (n : Nat) :
  (n ≤ 2 → min_colors_needed n = n) ∧
  (3 ≤ n ∧ n ≤ 5 → min_colors_needed n = 3) ∧
  (n ≥ 6 → min_colors_needed n = 2) :=
by
  sorry

end NUMINAMATH_GPT_min_colors_to_distinguish_keys_l23_2346


namespace NUMINAMATH_GPT_find_c_plus_d_l23_2351

def is_smallest_two_digit_multiple_of_5 (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ ∃ k : ℕ, n = 5 * k ∧ ∀ m : ℕ, (10 ≤ m ∧ m < 100 ∧ ∃ k', m = 5 * k') → n ≤ m

def is_smallest_three_digit_multiple_of_7 (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ ∃ k : ℕ, n = 7 * k ∧ ∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ ∃ k', m = 7 * k') → n ≤ m

theorem find_c_plus_d :
  ∃ c d : ℕ, is_smallest_two_digit_multiple_of_5 c ∧ is_smallest_three_digit_multiple_of_7 d ∧ c + d = 115 :=
by
  sorry

end NUMINAMATH_GPT_find_c_plus_d_l23_2351


namespace NUMINAMATH_GPT_jackson_meat_left_l23_2360

theorem jackson_meat_left (total_meat : ℕ) (meatballs_fraction : ℚ) (spring_rolls_meat : ℕ) :
  total_meat = 20 →
  meatballs_fraction = 1/4 →
  spring_rolls_meat = 3 →
  total_meat - (meatballs_fraction * total_meat + spring_rolls_meat) = 12 := by
  intros ht hm hs
  sorry

end NUMINAMATH_GPT_jackson_meat_left_l23_2360


namespace NUMINAMATH_GPT_trigonometric_identity_l23_2396

theorem trigonometric_identity :
  (1 / Real.cos (70 * Real.pi / 180) - Real.sqrt 3 / Real.sin (70 * Real.pi / 180))
  = (4 * Real.sin (10 * Real.pi / 180) / Real.sin (40 * Real.pi / 180)) :=
by sorry

end NUMINAMATH_GPT_trigonometric_identity_l23_2396


namespace NUMINAMATH_GPT_find_x_l23_2349

def F (a : ℕ) (b : ℕ) (c : ℕ) (d : ℕ) : ℕ := a^b + c * d

theorem find_x (x : ℕ) : F 3 x 5 9 = 500 → x = 6 := 
by 
  sorry

end NUMINAMATH_GPT_find_x_l23_2349


namespace NUMINAMATH_GPT_distance_between_towns_proof_l23_2332

noncomputable def distance_between_towns : ℕ :=
  let distance := 300
  let time_after_departure := 2
  let remaining_distance := 40
  let speed_difference := 10
  let total_distance_covered := distance - remaining_distance
  let speed_slower_train := 60
  let speed_faster_train := speed_slower_train + speed_difference
  let relative_speed := speed_slower_train + speed_faster_train
  distance

theorem distance_between_towns_proof 
  (distance : ℕ) 
  (time_after_departure : ℕ) 
  (remaining_distance : ℕ) 
  (speed_difference : ℕ) 
  (h1 : distance = 300) 
  (h2 : time_after_departure = 2) 
  (h3 : remaining_distance = 40) 
  (h4 : speed_difference = 10) 
  (speed_slower_train speed_faster_train relative_speed : ℕ)
  (h_speed_faster : speed_faster_train = speed_slower_train + speed_difference)
  (h_relative_speed : relative_speed = speed_slower_train + speed_faster_train) :
  distance = 300 :=
by {
  sorry
}

end NUMINAMATH_GPT_distance_between_towns_proof_l23_2332


namespace NUMINAMATH_GPT_x_varies_z_pow_l23_2395

variable (k j : ℝ)
variable (y z : ℝ)

-- Given conditions
def x_varies_y_squared (x : ℝ) := x = k * y^2
def y_varies_z_cuberoot_squared := y = j * z^(2/3)

-- To prove: 
theorem x_varies_z_pow (x : ℝ) (h1 : x_varies_y_squared k y x) (h2 : y_varies_z_cuberoot_squared j z y) : ∃ m : ℝ, x = m * z^(4/3) :=
by
  sorry

end NUMINAMATH_GPT_x_varies_z_pow_l23_2395


namespace NUMINAMATH_GPT_parallel_condition_sufficient_not_necessary_l23_2303

noncomputable def a (x : ℝ) : ℝ × ℝ := (1, x - 1)
noncomputable def b (x : ℝ) : ℝ × ℝ := (x + 1, 3)

theorem parallel_condition_sufficient_not_necessary (x : ℝ) :
  (x = 2) → (a x = b x) ∨ (a (-2) = b (-2)) :=
by sorry

end NUMINAMATH_GPT_parallel_condition_sufficient_not_necessary_l23_2303


namespace NUMINAMATH_GPT_find_angle_D_l23_2367

theorem find_angle_D 
  (A B C D : ℝ)
  (h1 : A + B = 180)
  (h2 : C = D)
  (h3 : A = 50) :
  D = 25 := 
by
  sorry

end NUMINAMATH_GPT_find_angle_D_l23_2367


namespace NUMINAMATH_GPT_double_inputs_revenue_l23_2374

theorem double_inputs_revenue (A K L : ℝ) (α1 α2 : ℝ) (hα1 : α1 = 0.6) (hα2 : α2 = 0.5) (hα1_bound : 0 < α1 ∧ α1 < 1) (hα2_bound : 0 < α2 ∧ α2 < 1) :
  A * (2 * K) ^ α1 * (2 * L) ^ α2 > 2 * (A * K ^ α1 * L ^ α2) :=
by
  sorry

end NUMINAMATH_GPT_double_inputs_revenue_l23_2374


namespace NUMINAMATH_GPT_calculate_angle_C_l23_2313

variable (A B C : ℝ)

theorem calculate_angle_C (h1 : A = C - 40) (h2 : B = 2 * A) (h3 : A + B + C = 180) :
  C = 75 :=
by
  sorry

end NUMINAMATH_GPT_calculate_angle_C_l23_2313


namespace NUMINAMATH_GPT_common_chord_of_circles_is_x_eq_y_l23_2344

theorem common_chord_of_circles_is_x_eq_y :
  ∀ x y : ℝ, (x^2 + y^2 - 4 * x - 3 = 0) ∧ (x^2 + y^2 - 4 * y - 3 = 0) → (x = y) :=
by
  sorry

end NUMINAMATH_GPT_common_chord_of_circles_is_x_eq_y_l23_2344


namespace NUMINAMATH_GPT_find_age_difference_l23_2358

variable (a b c : ℕ)

theorem find_age_difference (h : a + b = b + c + 20) : c = a - 20 :=
by
  sorry

end NUMINAMATH_GPT_find_age_difference_l23_2358


namespace NUMINAMATH_GPT_square_of_binomial_l23_2365

theorem square_of_binomial {a r s : ℚ} 
  (h1 : r^2 = a)
  (h2 : 2 * r * s = 18)
  (h3 : s^2 = 16) : 
  a = 81 / 16 :=
by sorry

end NUMINAMATH_GPT_square_of_binomial_l23_2365


namespace NUMINAMATH_GPT_mowers_mow_l23_2331

theorem mowers_mow (mowers hectares days mowers_new days_new : ℕ)
  (h1 : 3 * 3 * days = 3 * hectares)
  (h2 : 5 * days_new = 5 * (days_new * hectares / days)) :
  5 * days_new * (hectares / (3 * days)) = 25 / 3 :=
sorry

end NUMINAMATH_GPT_mowers_mow_l23_2331


namespace NUMINAMATH_GPT_email_scam_check_l23_2342

-- Define the condition for receiving an email about winning a car
def received_email (info: String) : Prop :=
  info = "You received an email informing you that you have won a car. You are asked to provide your mobile phone number for contact and to transfer 150 rubles to a bank card to cover the postage fee for sending the invitation letter."

-- Define what indicates a scam
def is_scam (info: String) : Prop :=
  info = "Request for mobile number already known to the sender and an upfront payment."

-- Proving that the information in the email implies it is a scam
theorem email_scam_check (info: String) (h1: received_email info) : is_scam info :=
by
  sorry

end NUMINAMATH_GPT_email_scam_check_l23_2342


namespace NUMINAMATH_GPT_jack_years_after_son_death_l23_2362

noncomputable def jackAdolescenceTime (L : Real) : Real := (1 / 6) * L
noncomputable def jackFacialHairTime (L : Real) : Real := (1 / 12) * L
noncomputable def jackMarriageTime (L : Real) : Real := (1 / 7) * L
noncomputable def jackSonBornTime (L : Real) (marriageTime : Real) : Real := marriageTime + 5
noncomputable def jackSonLifetime (L : Real) : Real := (1 / 2) * L
noncomputable def jackSonDeathTime (bornTime : Real) (sonLifetime : Real) : Real := bornTime + sonLifetime
noncomputable def yearsAfterSonDeath (L : Real) (sonDeathTime : Real) : Real := L - sonDeathTime

theorem jack_years_after_son_death : 
  yearsAfterSonDeath 84 
    (jackSonDeathTime (jackSonBornTime 84 (jackMarriageTime 84)) (jackSonLifetime 84)) = 4 :=
by
  sorry

end NUMINAMATH_GPT_jack_years_after_son_death_l23_2362


namespace NUMINAMATH_GPT_helly_half_planes_helly_convex_polygons_l23_2373

-- Helly's theorem for half-planes
theorem helly_half_planes (n : ℕ) (H : Fin n → Set ℝ) 
  (h : ∀ (i j k : Fin n), (H i ∩ H j ∩ H k).Nonempty) : 
  (⋂ i, H i).Nonempty :=
sorry

-- Helly's theorem for convex polygons
theorem helly_convex_polygons (n : ℕ) (P : Fin n → Set ℝ) 
  (h : ∀ (i j k : Fin n), (P i ∩ P j ∩ P k).Nonempty) : 
  (⋂ i, P i).Nonempty :=
sorry

end NUMINAMATH_GPT_helly_half_planes_helly_convex_polygons_l23_2373


namespace NUMINAMATH_GPT_requiredSheetsOfPaper_l23_2308

-- Define the conditions
def englishAlphabetLetters : ℕ := 26
def timesWrittenPerLetter : ℕ := 3
def sheetsOfPaperPerLetter (letters : ℕ) (times : ℕ) : ℕ := letters * times

-- State the theorem equivalent to the original math problem
theorem requiredSheetsOfPaper : sheetsOfPaperPerLetter englishAlphabetLetters timesWrittenPerLetter = 78 := by
  sorry

end NUMINAMATH_GPT_requiredSheetsOfPaper_l23_2308


namespace NUMINAMATH_GPT_arithmetic_sequence_value_l23_2336

theorem arithmetic_sequence_value (a : ℕ → ℝ) 
  (h1 : ∀ n : ℕ, a n = a 0 + n * (a 1 - a 0)) -- definition of arithmetic sequence
  (h2 : a 2 + a 10 = -12) -- given that a_2 + a_{10} = -12
  (h3 : a_2 = -6) -- given that a_6 is the average of a_2 and a_{10}
  : a 6 = -6 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_value_l23_2336
