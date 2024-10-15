import Mathlib

namespace NUMINAMATH_GPT_volleyball_ranking_l1347_134764

-- Define type for place
inductive Place where
  | first : Place
  | second : Place
  | third : Place

-- Define type for teams
inductive Team where
  | A : Team
  | B : Team
  | C : Team

open Place Team

-- Given conditions as hypotheses
def LiMing_prediction_half_correct (p : Place → Team → Prop) : Prop :=
  (p first A ∨ p third A) ∧ (p first B ∨ p third B) ∧ 
  ¬ (p first A ∧ p third A) ∧ ¬ (p first B ∧ p third B)

def ZhangHua_prediction_half_correct (p : Place → Team → Prop) : Prop :=
  (p third A ∨ p first C) ∧ (p third A ∨ p first A) ∧ 
  ¬ (p third A ∧ p first A) ∧ ¬ (p first C ∧ p third C)

def WangQiang_prediction_half_correct (p : Place → Team → Prop) : Prop :=
  (p second C ∨ p third B) ∧ (p second C ∨ p third C) ∧ 
  ¬ (p second C ∧ p third C) ∧ ¬ (p third B ∧ p second B)

-- Final proof problem
theorem volleyball_ranking (p : Place → Team → Prop) :
    (LiMing_prediction_half_correct p) →
    (ZhangHua_prediction_half_correct p) →
    (WangQiang_prediction_half_correct p) →
    p first C ∧ p second A ∧ p third B :=
  by
    sorry

end NUMINAMATH_GPT_volleyball_ranking_l1347_134764


namespace NUMINAMATH_GPT_charlie_certain_instrument_l1347_134717

theorem charlie_certain_instrument :
  ∃ (x : ℕ), (1 + 2 + x) + (2 + 1 + 0) = 7 → x = 1 :=
by
  sorry

end NUMINAMATH_GPT_charlie_certain_instrument_l1347_134717


namespace NUMINAMATH_GPT_percentage_markup_l1347_134780

theorem percentage_markup (P : ℝ) : 
  (∀ (n : ℕ) (cost price total_earned : ℝ),
    n = 50 →
    cost = 1 →
    price = 1 + P / 100 →
    total_earned = 60 →
    n * price = total_earned) →
  P = 20 :=
by
  intro h
  have h₁ := h 50 1 (1 + P / 100) 60 rfl rfl rfl rfl
  sorry  -- Placeholder for proof steps

end NUMINAMATH_GPT_percentage_markup_l1347_134780


namespace NUMINAMATH_GPT_sprint_team_total_miles_l1347_134794

-- Define the number of people and miles per person as constants
def numberOfPeople : ℕ := 250
def milesPerPerson : ℝ := 7.5

-- Assertion to prove the total miles
def totalMilesRun : ℝ := numberOfPeople * milesPerPerson

-- Proof statement
theorem sprint_team_total_miles : totalMilesRun = 1875 := 
by 
  -- Proof to be filled in
  sorry

end NUMINAMATH_GPT_sprint_team_total_miles_l1347_134794


namespace NUMINAMATH_GPT_calculate_expression_l1347_134710

theorem calculate_expression (a b c : ℤ) (ha : a = 3) (hb : b = 7) (hc : c = 2) :
  ((a * b - c) - (a + b * c)) - ((a * c - b) - (a - b * c)) = -8 :=
by
  rw [ha, hb, hc]  -- Substitute a, b, c with 3, 7, 2 respectively
  sorry  -- Placeholder for the proof

end NUMINAMATH_GPT_calculate_expression_l1347_134710


namespace NUMINAMATH_GPT_fraction_goldfish_preference_l1347_134769

theorem fraction_goldfish_preference
  (students_per_class : ℕ)
  (students_prefer_golfish_miss_johnson : ℕ)
  (students_prefer_golfish_ms_henderson : ℕ)
  (students_prefer_goldfish_total : ℕ)
  (miss_johnson_fraction : ℚ)
  (ms_henderson_fraction : ℚ)
  (total_students_prefer_goldfish_feldstein : ℕ)
  (feldstein_fraction : ℚ) :
  miss_johnson_fraction = 1/6 ∧
  ms_henderson_fraction = 1/5 ∧
  students_per_class = 30 ∧
  students_prefer_golfish_miss_johnson = miss_johnson_fraction * students_per_class ∧
  students_prefer_golfish_ms_henderson = ms_henderson_fraction * students_per_class ∧
  students_prefer_goldfish_total = 31 ∧
  students_prefer_goldfish_total = students_prefer_golfish_miss_johnson + students_prefer_golfish_ms_henderson + total_students_prefer_goldfish_feldstein ∧
  feldstein_fraction * students_per_class = total_students_prefer_goldfish_feldstein
  →
  feldstein_fraction = 2 / 3 :=
by 
  sorry

end NUMINAMATH_GPT_fraction_goldfish_preference_l1347_134769


namespace NUMINAMATH_GPT_allan_balloons_l1347_134708

theorem allan_balloons (a j t : ℕ) (h1 : t = 6) (h2 : j = 4) (h3 : t = a + j) : a = 2 := by
  sorry

end NUMINAMATH_GPT_allan_balloons_l1347_134708


namespace NUMINAMATH_GPT_poles_needed_to_enclose_plot_l1347_134784

-- Defining the lengths of the sides
def side1 : ℕ := 15
def side2 : ℕ := 22
def side3 : ℕ := 40
def side4 : ℕ := 30
def side5 : ℕ := 18

-- Defining the distance between poles
def dist_first_three_sides : ℕ := 4
def dist_last_two_sides : ℕ := 5

-- Defining the function to calculate required poles for a side
def calculate_poles (length : ℕ) (distance : ℕ) : ℕ :=
  (length / distance) + 1

-- Total poles needed before adjustment
def total_poles_before_adjustment : ℕ :=
  calculate_poles side1 dist_first_three_sides +
  calculate_poles side2 dist_first_three_sides +
  calculate_poles side3 dist_first_three_sides +
  calculate_poles side4 dist_last_two_sides +
  calculate_poles side5 dist_last_two_sides

-- Adjustment for shared poles at corners
def total_poles : ℕ :=
  total_poles_before_adjustment - 5

-- The theorem to prove
theorem poles_needed_to_enclose_plot : total_poles = 29 := by
  sorry

end NUMINAMATH_GPT_poles_needed_to_enclose_plot_l1347_134784


namespace NUMINAMATH_GPT_houses_before_boom_l1347_134745

theorem houses_before_boom (T B H : ℕ) (hT : T = 2000) (hB : B = 574) : H = 1426 := by
  sorry

end NUMINAMATH_GPT_houses_before_boom_l1347_134745


namespace NUMINAMATH_GPT_table_capacity_l1347_134735

theorem table_capacity :
  ∀ (n_invited no_show tables : ℕ), n_invited = 47 → no_show = 7 → tables = 8 → 
  (n_invited - no_show) / tables = 5 := by
  intros n_invited no_show tables h_invited h_no_show h_tables
  sorry

end NUMINAMATH_GPT_table_capacity_l1347_134735


namespace NUMINAMATH_GPT_quadratic_inequality_l1347_134750

theorem quadratic_inequality 
  (a b c : ℝ) 
  (h₁ : ∀ x : ℝ, |x| ≤ 1 → |a * x^2 + b * x + c| ≤ 1)
  (x : ℝ) 
  (hx : |x| ≤ 1) : 
  |c * x^2 + b * x + a| ≤ 2 := 
sorry

end NUMINAMATH_GPT_quadratic_inequality_l1347_134750


namespace NUMINAMATH_GPT_pure_imaginary_number_a_l1347_134703

theorem pure_imaginary_number_a (a : ℝ) 
  (h1 : a^2 + 2 * a - 3 = 0)
  (h2 : a^2 - 4 * a + 3 ≠ 0) : a = -3 :=
sorry

end NUMINAMATH_GPT_pure_imaginary_number_a_l1347_134703


namespace NUMINAMATH_GPT_largest_room_width_l1347_134788

theorem largest_room_width (w : ℕ) :
  (w * 30 - 15 * 8 = 1230) → (w = 45) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_largest_room_width_l1347_134788


namespace NUMINAMATH_GPT_inequality_proof_l1347_134761

-- Conditions: a > b and c > d
variables {a b c d : ℝ}

-- The main statement to prove: d - a < c - b with given conditions
theorem inequality_proof (h1 : a > b) (h2 : c > d) : d - a < c - b := 
sorry

end NUMINAMATH_GPT_inequality_proof_l1347_134761


namespace NUMINAMATH_GPT_work_rate_c_l1347_134702

theorem work_rate_c (A B C : ℝ) (h1 : A + B = 1 / 4) (h2 : B + C = 1 / 6) (h3 : C + A = 1 / 3) :
    1 / C = 8 :=
by
  sorry

end NUMINAMATH_GPT_work_rate_c_l1347_134702


namespace NUMINAMATH_GPT_payment_is_variable_l1347_134786

variable (x y : ℕ)

def price_of_pen : ℕ := 3

theorem payment_is_variable (x y : ℕ) (h : y = price_of_pen * x) : 
  (price_of_pen = 3) ∧ (∃ n : ℕ, y = 3 * n) :=
by 
  sorry

end NUMINAMATH_GPT_payment_is_variable_l1347_134786


namespace NUMINAMATH_GPT_tan_45_eq_one_l1347_134713

theorem tan_45_eq_one (Q : ℝ × ℝ) (h1 : Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2)) : Real.tan (Real.pi / 4) = 1 := by
  sorry

end NUMINAMATH_GPT_tan_45_eq_one_l1347_134713


namespace NUMINAMATH_GPT_arithmetic_sequence_a6_l1347_134751

-- Definitions representing the conditions
def arithmetic_sequence (a_n : ℕ → ℝ) : Prop :=
∀ n m : ℕ, a_n (n + m) = a_n n + a_n m + n

def sum_of_first_n_terms (S : ℕ → ℝ) (a_n : ℕ → ℝ) : Prop :=
∀ n : ℕ, S n = (n / 2) * (2 * a_n 1 + (n - 1) * (a_n 2 - a_n 1))

theorem arithmetic_sequence_a6 (S : ℕ → ℝ) (a_n : ℕ → ℝ) 
  (h_seq : arithmetic_sequence a_n)
  (h_sum : sum_of_first_n_terms S a_n)
  (h_cond : S 9 - S 2 = 35) : 
  a_n 6 = 5 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a6_l1347_134751


namespace NUMINAMATH_GPT_polynomial_remainder_l1347_134771

theorem polynomial_remainder (x : ℝ) : 
  (x - 1)^100 + (x - 2)^200 = (x^2 - 3 * x + 2) * (some_q : ℝ) + 1 :=
sorry

end NUMINAMATH_GPT_polynomial_remainder_l1347_134771


namespace NUMINAMATH_GPT_perpendicular_lines_foot_l1347_134798

variables (a b c : ℝ)

theorem perpendicular_lines_foot (h1 : a * -2/20 = -1)
  (h2_foot_l1 : a * 1 + 4 * c - 2 = 0)
  (h3_foot_l2 : 2 * 1 - 5 * c + b = 0) :
  a + b + c = -4 :=
sorry

end NUMINAMATH_GPT_perpendicular_lines_foot_l1347_134798


namespace NUMINAMATH_GPT_total_students_appeared_l1347_134770

variable (T : ℝ) -- total number of students

def fraction_failed := 0.65
def num_failed := 546

theorem total_students_appeared :
  0.65 * T = 546 → T = 840 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_total_students_appeared_l1347_134770


namespace NUMINAMATH_GPT_solution_l1347_134799

noncomputable def x : ℕ := 13

theorem solution : (3 * x) - (36 - x) = 16 := by
  sorry

end NUMINAMATH_GPT_solution_l1347_134799


namespace NUMINAMATH_GPT_seashells_problem_l1347_134768

theorem seashells_problem
  (F : ℕ)
  (h : (150 - F) / 2 = 55) :
  F = 40 :=
  sorry

end NUMINAMATH_GPT_seashells_problem_l1347_134768


namespace NUMINAMATH_GPT_relation_among_a_b_c_l1347_134721

noncomputable def a : ℝ := (1/2)^(1/3)
noncomputable def b : ℝ := Real.log (1/3) / Real.log 2
noncomputable def c : ℝ := Real.log 3 / Real.log 2

theorem relation_among_a_b_c : c > a ∧ a > b :=
by
  -- Prove that c > a and a > b
  sorry

end NUMINAMATH_GPT_relation_among_a_b_c_l1347_134721


namespace NUMINAMATH_GPT_car_catches_up_in_6_hours_l1347_134785

-- Conditions
def speed_truck := 40 -- km/h
def speed_car_initial := 50 -- km/h
def speed_car_increment := 5 -- km/h
def distance_between := 135 -- km

-- Solution: car catches up in 6 hours
theorem car_catches_up_in_6_hours : 
  ∃ n : ℕ, n = 6 ∧ (n * speed_truck + distance_between) ≤ (n * speed_car_initial + (n * (n - 1) / 2 * speed_car_increment)) := 
by
  sorry

end NUMINAMATH_GPT_car_catches_up_in_6_hours_l1347_134785


namespace NUMINAMATH_GPT_book_pages_l1347_134726

-- Define the number of pages Sally reads on weekdays and weekends
def pages_on_weekdays : ℕ := 10
def pages_on_weekends : ℕ := 20

-- Define the number of weekdays and weekends in 2 weeks
def weekdays_in_two_weeks : ℕ := 5 * 2
def weekends_in_two_weeks : ℕ := 2 * 2

-- Total number of pages read in 2 weeks
def total_pages_read (pages_on_weekdays : ℕ) (pages_on_weekends : ℕ) (weekdays_in_two_weeks : ℕ) (weekends_in_two_weeks : ℕ) : ℕ :=
  (pages_on_weekdays * weekdays_in_two_weeks) + (pages_on_weekends * weekends_in_two_weeks)

-- Prove the number of pages in the book
theorem book_pages : total_pages_read 10 20 10 4 = 180 := by
  sorry

end NUMINAMATH_GPT_book_pages_l1347_134726


namespace NUMINAMATH_GPT_highest_value_of_a_l1347_134732

theorem highest_value_of_a (a : ℕ) (h : 0 ≤ a ∧ a ≤ 9) : (365 * 10 ^ 3 + a * 10 ^ 2 + 16) % 8 = 0 → a = 8 := by
  sorry

end NUMINAMATH_GPT_highest_value_of_a_l1347_134732


namespace NUMINAMATH_GPT_annual_decrease_rate_l1347_134714

theorem annual_decrease_rate
  (P0 : ℕ := 8000)
  (P2 : ℕ := 6480) :
  ∃ r : ℝ, 8000 * (1 - r / 100)^2 = 6480 ∧ r = 10 :=
by
  use 10
  sorry

end NUMINAMATH_GPT_annual_decrease_rate_l1347_134714


namespace NUMINAMATH_GPT_largest_positive_integer_l1347_134718

def binary_operation (n : Int) : Int := n - (n * 5)

theorem largest_positive_integer (n : Int) : (∀ m : Int, m > 0 → n - (n * 5) < -19 → m ≤ n) 
  ↔ n = 5 := 
by
  sorry

end NUMINAMATH_GPT_largest_positive_integer_l1347_134718


namespace NUMINAMATH_GPT_proof_of_expression_l1347_134796

theorem proof_of_expression (a : ℝ) (h : a^2 + a + 1 = 2) : (5 - a) * (6 + a) = 29 :=
by {
  sorry
}

end NUMINAMATH_GPT_proof_of_expression_l1347_134796


namespace NUMINAMATH_GPT_quadratic_eq_real_roots_l1347_134759

theorem quadratic_eq_real_roots (m : ℝ) : 
  (∃ x : ℝ, x^2 - 3*x + 2*m = 0) → m ≤ 9 / 8 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_eq_real_roots_l1347_134759


namespace NUMINAMATH_GPT_calculate_expression_l1347_134756

theorem calculate_expression : -1^4 * 8 - 2^3 / (-4) * (-7 + 5) = -12 := 
by 
  /-
  In Lean, we typically perform arithmetic simplifications step by step;
  however, for the purpose of this example, only stating the goal:
  -/
  sorry

end NUMINAMATH_GPT_calculate_expression_l1347_134756


namespace NUMINAMATH_GPT_count_even_fibonacci_first_2007_l1347_134707

def fibonacci (n : Nat) : Nat :=
  if h : n = 0 then 0
  else if h : n = 1 then 1
  else fibonacci (n - 1) + fibonacci (n - 2)

def fibonacci_parity : List Bool := List.map (fun x => fibonacci x % 2 = 0) (List.range 2008)

def count_even (l : List Bool) : Nat :=
  l.foldl (fun acc x => if x then acc + 1 else acc) 0

theorem count_even_fibonacci_first_2007 : count_even (fibonacci_parity.take 2007) = 669 :=
sorry

end NUMINAMATH_GPT_count_even_fibonacci_first_2007_l1347_134707


namespace NUMINAMATH_GPT_min_x2_plus_y2_l1347_134755

theorem min_x2_plus_y2 (x y : ℝ) (h : (x + 5)^2 + (y - 12)^2 = 14^2) : x^2 + y^2 ≥ 1 :=
sorry

end NUMINAMATH_GPT_min_x2_plus_y2_l1347_134755


namespace NUMINAMATH_GPT_angle_SVU_l1347_134747

theorem angle_SVU (TU SV SU : ℝ) (angle_STU_T : ℝ) (angle_STU_S : ℝ) :
  TU = SV → angle_STU_T = 75 → angle_STU_S = 30 →
  TU = SU → SU = SV → S_V_U = 65 :=
by
  intros H1 H2 H3 H4 H5
  -- skip proof
  sorry

end NUMINAMATH_GPT_angle_SVU_l1347_134747


namespace NUMINAMATH_GPT_equation_roots_l1347_134705

theorem equation_roots (x : ℝ) : x * x = 2 * x ↔ (x = 0 ∨ x = 2) := by
  sorry

end NUMINAMATH_GPT_equation_roots_l1347_134705


namespace NUMINAMATH_GPT_solution_m_value_l1347_134790

theorem solution_m_value (m : ℝ) : 
  (m^2 - 5*m + 4 > 0) ∧ (m^2 - 2*m = 0) ↔ m = 0 :=
by
  sorry

end NUMINAMATH_GPT_solution_m_value_l1347_134790


namespace NUMINAMATH_GPT_quadratic_m_value_l1347_134781

theorem quadratic_m_value (m : ℤ) (hm1 : |m| = 2) (hm2 : m ≠ 2) : m = -2 :=
sorry

end NUMINAMATH_GPT_quadratic_m_value_l1347_134781


namespace NUMINAMATH_GPT_interest_after_5_years_l1347_134719

noncomputable def initial_amount : ℝ := 2000
noncomputable def interest_rate : ℝ := 0.08
noncomputable def duration : ℕ := 5
noncomputable def final_amount : ℝ := initial_amount * (1 + interest_rate) ^ duration
noncomputable def interest_earned : ℝ := final_amount - initial_amount

theorem interest_after_5_years : interest_earned = 938.66 := by
  sorry

end NUMINAMATH_GPT_interest_after_5_years_l1347_134719


namespace NUMINAMATH_GPT_car_miles_per_gallon_l1347_134748

-- Define the conditions
def distance_home : ℕ := 220
def additional_distance : ℕ := 100
def total_distance : ℕ := distance_home + additional_distance
def tank_capacity : ℕ := 16 -- in gallons
def miles_per_gallon : ℕ := total_distance / tank_capacity

-- State the goal
theorem car_miles_per_gallon : miles_per_gallon = 20 := by
  sorry

end NUMINAMATH_GPT_car_miles_per_gallon_l1347_134748


namespace NUMINAMATH_GPT_cans_in_third_bin_l1347_134795

noncomputable def num_cans_in_bin (n : ℕ) : ℕ :=
  match n with
  | 1 => 2
  | 2 => 4
  | 3 => 7
  | 4 => 11
  | 5 => 16
  | _ => sorry

theorem cans_in_third_bin :
  num_cans_in_bin 3 = 7 :=
sorry

end NUMINAMATH_GPT_cans_in_third_bin_l1347_134795


namespace NUMINAMATH_GPT_new_percentage_of_water_l1347_134763

noncomputable def initial_weight : ℝ := 100
noncomputable def initial_percentage_water : ℝ := 99 / 100
noncomputable def initial_weight_water : ℝ := initial_weight * initial_percentage_water
noncomputable def initial_weight_non_water : ℝ := initial_weight - initial_weight_water
noncomputable def new_weight : ℝ := 25

theorem new_percentage_of_water :
  ((new_weight - initial_weight_non_water) / new_weight) * 100 = 96 :=
by
  sorry

end NUMINAMATH_GPT_new_percentage_of_water_l1347_134763


namespace NUMINAMATH_GPT_product_approximation_l1347_134752

-- Define the approximation condition
def approxProduct (x y : ℕ) (approxX approxY : ℕ) : ℕ :=
  approxX * approxY

-- State the theorem
theorem product_approximation :
  let x := 29
  let y := 32
  let approxX := 30
  let approxY := 30
  approxProduct x y approxX approxY = 900 := by
  sorry

end NUMINAMATH_GPT_product_approximation_l1347_134752


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1347_134778

-- Definitions
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Problem conditions
theorem solution_set_of_inequality (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_neg : ∀ x, x < 0 → f x = x + 2) :
  { x : ℝ | 2 * f x - 1 < 0 } = { x : ℝ | x < -3/2 ∨ (0 ≤ x ∧ x < 5/2) } :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1347_134778


namespace NUMINAMATH_GPT_general_term_arithmetic_sequence_l1347_134704

-- Consider an arithmetic sequence {a_n}
variable (a : ℕ → ℤ)

-- Conditions
def a1 : Prop := a 1 = 1
def a3 : Prop := a 3 = -3
def is_arithmetic_sequence : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, n ≥ 1 → a (n + 1) = a n + d

-- Theorem statement
theorem general_term_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) 
  (h1 : a 1 = 1) (h3 : a 3 = -3) (h_arith : ∀ n : ℕ, n ≥ 1 → a (n + 1) = a n + d) :
  ∀ n : ℕ, a n = 3 - 2 * n :=
by
  sorry  -- proof is not required

end NUMINAMATH_GPT_general_term_arithmetic_sequence_l1347_134704


namespace NUMINAMATH_GPT_sum_of_tangent_points_l1347_134779

noncomputable def f (x : ℝ) : ℝ := 
  max (max (-7 * x - 19) (3 * x - 1)) (5 * x + 3)

theorem sum_of_tangent_points :
  ∃ x4 x5 x6 : ℝ, 
  (∃ q : ℝ → ℝ, 
    (∀ x, q x = f x ∨ (q x - (-7 * x - 19)) = b * (x - x4)^2
    ∨ (q x - (3 * x - 1)) = b * (x - x5)^2 
    ∨ (q x - (5 * x + 3)) = b * (x - x6)^2)) ∧
  x4 + x5 + x6 = -3.2 :=
sorry

end NUMINAMATH_GPT_sum_of_tangent_points_l1347_134779


namespace NUMINAMATH_GPT_abs_inequality_solution_l1347_134765

theorem abs_inequality_solution (x : ℝ) : 
  (|2 * x + 1| > 3) ↔ (x > 1 ∨ x < -2) :=
sorry

end NUMINAMATH_GPT_abs_inequality_solution_l1347_134765


namespace NUMINAMATH_GPT_simplify_expression_l1347_134715

noncomputable def cube_root (x : ℝ) := x^(1/3)

theorem simplify_expression :
  cube_root (8 + 27) * cube_root (8 + cube_root 27) = cube_root 385 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1347_134715


namespace NUMINAMATH_GPT_ellipse_properties_l1347_134701

theorem ellipse_properties 
  (foci1 foci2 : ℝ × ℝ) 
  (point_on_ellipse : ℝ × ℝ) 
  (h k a b : ℝ) 
  (a_pos : a > 0) 
  (b_pos : b > 0) 
  (ellipse_condition : foci1 = (-4, 1) ∧ foci2 = (-4, 5) ∧ point_on_ellipse = (1, 3))
  (ellipse_eqn : (x y : ℝ) → ((x - h)^2 / a^2) + ((y - k)^2 / b^2) = 1) :
  a + k = 8 :=
by
  sorry

end NUMINAMATH_GPT_ellipse_properties_l1347_134701


namespace NUMINAMATH_GPT_nails_sum_is_correct_l1347_134774

-- Define the fractions for sizes 2d, 3d, 5d, and 8d
def fraction_2d : ℚ := 1 / 6
def fraction_3d : ℚ := 2 / 15
def fraction_5d : ℚ := 1 / 10
def fraction_8d : ℚ := 1 / 8

-- Define the expected answer
def expected_fraction : ℚ := 21 / 40

-- The theorem to prove
theorem nails_sum_is_correct : fraction_2d + fraction_3d + fraction_5d + fraction_8d = expected_fraction :=
by
  -- The proof is not required as per the instructions
  sorry

end NUMINAMATH_GPT_nails_sum_is_correct_l1347_134774


namespace NUMINAMATH_GPT_problem_1_and_2_problem_1_infinite_solutions_l1347_134731

open Nat

theorem problem_1_and_2 (k : ℕ) (a b c : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) :
  (a^2 + b^2 + c^2 = k * a * b * c) →
  (k = 1 ∨ k = 3) :=
sorry

theorem problem_1_infinite_solutions (k : ℕ) (h_k : k = 1 ∨ k = 3) :
  ∃ (a_n b_n c_n : ℕ) (n : ℕ), 
  a_n > 0 ∧ b_n > 0 ∧ c_n > 0 ∧
  (a_n^2 + b_n^2 + c_n^2 = k * a_n * b_n * c_n) ∧
  ∀ x y : ℕ, (x = a_n ∧ y = b_n) ∨ (x = a_n ∧ y = c_n) ∨ (x = b_n ∧ y = c_n) →
    ∃ p q : ℕ, x * y = p^2 + q^2 :=
sorry

end NUMINAMATH_GPT_problem_1_and_2_problem_1_infinite_solutions_l1347_134731


namespace NUMINAMATH_GPT_Tim_carrots_count_l1347_134723

theorem Tim_carrots_count (initial_potatoes new_potatoes initial_carrots final_potatoes final_carrots : ℕ) 
  (h_ratio : 3 * final_potatoes = 4 * final_carrots)
  (h_initial_potatoes : initial_potatoes = 32)
  (h_new_potatoes : new_potatoes = 28)
  (h_final_potatoes : final_potatoes = initial_potatoes + new_potatoes)
  (h_initial_ratio : 3 * 32 = 4 * initial_carrots) : 
  final_carrots = 45 :=
by {
  sorry
}

end NUMINAMATH_GPT_Tim_carrots_count_l1347_134723


namespace NUMINAMATH_GPT_find_u_l1347_134777

theorem find_u (u : ℝ) : (∃ x : ℝ, x = ( -15 - Real.sqrt 145 ) / 8 ∧ 4 * x^2 + 15 * x + u = 0) ↔ u = 5 := by
  sorry

end NUMINAMATH_GPT_find_u_l1347_134777


namespace NUMINAMATH_GPT_arithmetic_sequence_problem_l1347_134793

-- Define the arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

-- Use the given specific conditions
theorem arithmetic_sequence_problem 
  (a : ℕ → ℤ)
  (h1 : is_arithmetic_sequence a)
  (h2 : a 2 * a 3 = 21) : 
  a 1 * a 4 = -11 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_problem_l1347_134793


namespace NUMINAMATH_GPT_math_problem_l1347_134716

theorem math_problem (m : ℝ) (h : m^2 - m = 2) : (m - 1)^2 + (m + 2) * (m - 2) = 1 := 
by sorry

end NUMINAMATH_GPT_math_problem_l1347_134716


namespace NUMINAMATH_GPT_total_population_calculation_l1347_134709

theorem total_population_calculation :
  ∀ (total_lions total_leopards adult_lions adult_leopards : ℕ)
  (female_lions male_lions female_leopards male_leopards : ℕ)
  (adult_elephants baby_elephants total_elephants total_zebras : ℕ),
  total_lions = 200 →
  total_lions = 2 * total_leopards →
  adult_lions = 3 * total_lions / 4 →
  adult_leopards = 3 * total_leopards / 5 →
  female_lions = 3 * total_lions / 5 →
  male_lions = 2 * total_lions / 5 →
  female_leopards = 2 * total_leopards / 3 →
  male_leopards = total_leopards / 3 →
  adult_elephants = (adult_lions + adult_leopards) / 2 →
  baby_elephants = 100 →
  total_elephants = adult_elephants + baby_elephants →
  total_zebras = adult_elephants + total_leopards →
  total_lions + total_leopards + total_elephants + total_zebras = 710 :=
by sorry

end NUMINAMATH_GPT_total_population_calculation_l1347_134709


namespace NUMINAMATH_GPT_triangle_angle_problem_l1347_134720

open Real

-- Define degrees to radians conversion (if necessary)
noncomputable def degrees (d : ℝ) : ℝ := d * π / 180

-- Define the problem conditions and goal
theorem triangle_angle_problem
  (x y : ℝ)
  (h1 : degrees 3 * x + degrees y = degrees 90) :
  x = 18 ∧ y = 36 := by
  sorry

end NUMINAMATH_GPT_triangle_angle_problem_l1347_134720


namespace NUMINAMATH_GPT_simplify_decimal_l1347_134776

theorem simplify_decimal : (3416 / 1000 : ℚ) = 427 / 125 := by
  sorry

end NUMINAMATH_GPT_simplify_decimal_l1347_134776


namespace NUMINAMATH_GPT_smallest_positive_period_max_min_value_interval_l1347_134749

noncomputable def f (x : ℝ) : ℝ :=
  2 * (Real.sin (x + Real.pi / 3))^2 - (Real.cos x)^2 + (Real.sin x)^2

theorem smallest_positive_period : (∀ x : ℝ, f (x + Real.pi) = f x) :=
by sorry

theorem max_min_value_interval :
  (∀ x ∈ Set.Icc (-Real.pi / 3) (Real.pi / 6), 
    f x ≤ 3 / 2 ∧ f x ≥ 0 ∧ 
    (f (-Real.pi / 6) = 0) ∧ 
    (f (Real.pi / 6) = 3 / 2)) :=
by sorry

end NUMINAMATH_GPT_smallest_positive_period_max_min_value_interval_l1347_134749


namespace NUMINAMATH_GPT_distance_to_water_source_l1347_134782

theorem distance_to_water_source (d : ℝ) :
  (¬(d ≥ 8)) ∧ (¬(d ≤ 7)) ∧ (¬(d ≤ 5)) → 7 < d ∧ d < 8 :=
by
  sorry

end NUMINAMATH_GPT_distance_to_water_source_l1347_134782


namespace NUMINAMATH_GPT_sin_cos_half_angle_sum_l1347_134727

theorem sin_cos_half_angle_sum 
  (θ : ℝ)
  (hcos : Real.cos θ = -7/25) 
  (hθ : θ ∈ Set.Ioo (-Real.pi) 0) : 
  Real.sin (θ/2) + Real.cos (θ/2) = -1/5 := 
sorry

end NUMINAMATH_GPT_sin_cos_half_angle_sum_l1347_134727


namespace NUMINAMATH_GPT_principle_calculation_l1347_134797

noncomputable def calculate_principal (A : ℝ) (R : ℝ) (T : ℝ) : ℝ :=
  A / (1 + (R * T))

theorem principle_calculation :
  calculate_principal 1456 0.05 2.4 = 1300 :=
by
  sorry

end NUMINAMATH_GPT_principle_calculation_l1347_134797


namespace NUMINAMATH_GPT_min_value_expression_l1347_134787

theorem min_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x * y * z = 48) :
  x^2 + 4 * x * y + 4 * y^2 + 3 * z^2 ≥ 144 :=
sorry

end NUMINAMATH_GPT_min_value_expression_l1347_134787


namespace NUMINAMATH_GPT_range_of_a_l1347_134706

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, a * x^2 - a * x - 2 ≤ 0) ↔ -8 ≤ a ∧ a ≤ 0 := sorry

end NUMINAMATH_GPT_range_of_a_l1347_134706


namespace NUMINAMATH_GPT_find_angle_A_find_value_of_c_l1347_134746

variable {a b c A B C : ℝ}

-- Define the specific conditions as Lean 'variables' and 'axioms'
-- Condition: In triangle ABC, the sides opposite to angles A, B and C are a, b, and c respectively.
axiom triangle_ABC_sides : b = 2 * (a * Real.cos B - c)

-- Part (1): Prove the value of angle A
theorem find_angle_A (h : b = 2 * (a * Real.cos B - c)) : A = (2 * Real.pi) / 3 :=
by
  sorry

-- Condition: a * cos C = sqrt 3 and b = 1
axiom cos_C_value : a * Real.cos C = Real.sqrt 3
axiom b_value : b = 1

-- Part (2): Prove the value of c
theorem find_value_of_c (h1 : a * Real.cos C = Real.sqrt 3) (h2 : b = 1) : c = 2 * Real.sqrt 3 - 2 :=
by
  sorry

end NUMINAMATH_GPT_find_angle_A_find_value_of_c_l1347_134746


namespace NUMINAMATH_GPT_sin_double_angle_l1347_134733

theorem sin_double_angle (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.sin α = 3 / 5) : 
  Real.sin (2 * α) = 24 / 25 :=
by sorry

end NUMINAMATH_GPT_sin_double_angle_l1347_134733


namespace NUMINAMATH_GPT_find_k_l1347_134753

theorem find_k (k : ℚ) : 
  ((3, -8) ≠ (k, 20)) ∧ 
  (∃ m, (4 * m = -3) ∧ (20 - (-8) = m * (k - 3))) → 
  k = -103/3 := 
by
  sorry

end NUMINAMATH_GPT_find_k_l1347_134753


namespace NUMINAMATH_GPT_y_intercept_3x_minus_4y_eq_12_l1347_134725

theorem y_intercept_3x_minus_4y_eq_12 :
  (- 4 * -3) = 12 :=
by
  sorry

end NUMINAMATH_GPT_y_intercept_3x_minus_4y_eq_12_l1347_134725


namespace NUMINAMATH_GPT_bucket_full_weight_l1347_134737

variable {a b x y : ℝ}

theorem bucket_full_weight (h1 : x + 2/3 * y = a) (h2 : x + 1/2 * y = b) : 
  (x + y) = 3 * a - 2 * b := 
sorry

end NUMINAMATH_GPT_bucket_full_weight_l1347_134737


namespace NUMINAMATH_GPT_tree_last_tree_height_difference_l1347_134730

noncomputable def treeHeightDifference : ℝ :=
  let t1 := 1000
  let t2 := 500
  let t3 := 500
  let avgHeight := 800
  let lastTreeHeight := 4 * avgHeight - (t1 + t2 + t3)
  lastTreeHeight - t1

theorem tree_last_tree_height_difference :
  treeHeightDifference = 200 := sorry

end NUMINAMATH_GPT_tree_last_tree_height_difference_l1347_134730


namespace NUMINAMATH_GPT_average_sale_over_six_months_l1347_134792

theorem average_sale_over_six_months : 
  let s1 := 3435
  let s2 := 3920
  let s3 := 3855
  let s4 := 4230
  let s5 := 3560
  let s6 := 2000
  let total_sale := s1 + s2 + s3 + s4 + s5 + s6
  let average_sale := total_sale / 6
  average_sale = 3500 :=
by
  let s1 := 3435
  let s2 := 3920
  let s3 := 3855
  let s4 := 4230
  let s5 := 3560
  let s6 := 2000
  let total_sale := s1 + s2 + s3 + s4 + s5 + s6
  let average_sale := total_sale / 6
  show average_sale = 3500
  sorry

end NUMINAMATH_GPT_average_sale_over_six_months_l1347_134792


namespace NUMINAMATH_GPT_find_n_l1347_134722

theorem find_n (n : ℕ) : (10^n = (10^5)^3) → n = 15 :=
by sorry

end NUMINAMATH_GPT_find_n_l1347_134722


namespace NUMINAMATH_GPT_remainder_when_divided_by_x_minus_2_l1347_134757

-- Define the polynomial f(x)
def f (x : ℝ) : ℝ := x^4 - 8 * x^3 + 12 * x^2 + 20 * x - 10

-- State the theorem about the remainder when f(x) is divided by x-2
theorem remainder_when_divided_by_x_minus_2 : f 2 = 30 := by
  -- This is where the proof would go, but we use sorry to skip the proof.
  sorry

end NUMINAMATH_GPT_remainder_when_divided_by_x_minus_2_l1347_134757


namespace NUMINAMATH_GPT_ryan_spanish_hours_l1347_134743

theorem ryan_spanish_hours (S : ℕ) (h : 7 = S + 3) : S = 4 :=
sorry

end NUMINAMATH_GPT_ryan_spanish_hours_l1347_134743


namespace NUMINAMATH_GPT_yellow_tint_percentage_l1347_134772

theorem yellow_tint_percentage {V₀ V₁ V_t red_pct yellow_pct : ℝ} 
  (hV₀ : V₀ = 40)
  (hRed : red_pct = 0.20)
  (hYellow : yellow_pct = 0.25)
  (hAdd : V₁ = 10) :
  (yellow_pct * V₀ + V₁) / (V₀ + V₁) = 0.40 :=
by
  sorry

end NUMINAMATH_GPT_yellow_tint_percentage_l1347_134772


namespace NUMINAMATH_GPT_find_number_l1347_134773

theorem find_number (x : ℕ) (h : x + 18 = 44) : x = 26 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l1347_134773


namespace NUMINAMATH_GPT_number_of_people_who_purchased_only_book_A_l1347_134758

-- Define the conditions and the problem
theorem number_of_people_who_purchased_only_book_A 
    (total_A : ℕ) (total_B : ℕ) (both_AB : ℕ) (only_B : ℕ) :
    (total_A = 2 * total_B) → 
    (both_AB = 500) → 
    (both_AB = 2 * only_B) → 
    (total_B = only_B + both_AB) → 
    (total_A - both_AB = 1000) :=
by
  sorry

end NUMINAMATH_GPT_number_of_people_who_purchased_only_book_A_l1347_134758


namespace NUMINAMATH_GPT_find_principal_amount_l1347_134760

-- Definitions of the conditions
def rate_of_interest : ℝ := 0.20
def time_period : ℕ := 2
def interest_difference : ℝ := 144

-- Definitions for Simple Interest (SI) and Compound Interest (CI)
def simple_interest (P : ℝ) : ℝ := P * rate_of_interest * time_period
def compound_interest (P : ℝ) : ℝ := P * (1 + rate_of_interest)^time_period - P

-- Statement to prove the principal amount given the conditions
theorem find_principal_amount (P : ℝ) : 
    compound_interest P - simple_interest P = interest_difference → P = 3600 := by
    sorry

end NUMINAMATH_GPT_find_principal_amount_l1347_134760


namespace NUMINAMATH_GPT_solve_for_x_l1347_134736

theorem solve_for_x (x : ℕ) (h : x + 1 = 2) : x = 1 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l1347_134736


namespace NUMINAMATH_GPT_find_multiplier_l1347_134775

theorem find_multiplier (x : ℕ) (h1 : 268 * x = 19832) (h2 : 2.68 * 0.74 = 1.9832) : x = 74 :=
sorry

end NUMINAMATH_GPT_find_multiplier_l1347_134775


namespace NUMINAMATH_GPT_increase_75_by_150_percent_l1347_134789

noncomputable def original_number : Real := 75
noncomputable def percentage_increase : Real := 1.5
noncomputable def increase_amount : Real := original_number * percentage_increase
noncomputable def result : Real := original_number + increase_amount

theorem increase_75_by_150_percent : result = 187.5 := by
  sorry

end NUMINAMATH_GPT_increase_75_by_150_percent_l1347_134789


namespace NUMINAMATH_GPT_noodles_initial_l1347_134724

-- Definitions of our conditions
def given_away : ℝ := 12.0
def noodles_left : ℝ := 42.0
def initial_noodles : ℝ := 54.0

-- Theorem statement
theorem noodles_initial (a b : ℝ) (x : ℝ) (h₁ : a = 12.0) (h₂ : b = 42.0) (h₃ : x = a + b) : x = initial_noodles :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_noodles_initial_l1347_134724


namespace NUMINAMATH_GPT_total_length_of_figure_2_segments_l1347_134767

-- Definitions based on conditions
def rectangle_length : ℕ := 10
def rectangle_breadth : ℕ := 6
def square_side : ℕ := 4
def interior_segment : ℕ := rectangle_breadth / 2

-- Summing up the lengths of segments in Figure 2
def total_length_of_segments : ℕ :=
  square_side + 2 * rectangle_length + interior_segment

-- Mathematical proof problem statement
theorem total_length_of_figure_2_segments :
  total_length_of_segments = 27 :=
sorry

end NUMINAMATH_GPT_total_length_of_figure_2_segments_l1347_134767


namespace NUMINAMATH_GPT_cade_marbles_now_l1347_134740

def original_marbles : ℝ := 87.0
def added_marbles : ℝ := 8.0
def total_marbles : ℝ := original_marbles + added_marbles

theorem cade_marbles_now : total_marbles = 95.0 :=
by
  sorry

end NUMINAMATH_GPT_cade_marbles_now_l1347_134740


namespace NUMINAMATH_GPT_height_cylinder_l1347_134728

variables (r_c h_c r_cy h_cy : ℝ)
variables (V_cone V_cylinder : ℝ)
variables (r_c_val : r_c = 15)
variables (h_c_val : h_c = 20)
variables (r_cy_val : r_cy = 30)
variables (V_cone_eq : V_cone = (1/3) * π * r_c^2 * h_c)
variables (V_cylinder_eq : V_cylinder = π * r_cy^2 * h_cy)

theorem height_cylinder : h_cy = 1.67 :=
by
  rw [r_c_val, h_c_val, r_cy_val] at *
  have V_cone := V_cone_eq
  have V_cylinder := V_cylinder_eq
  sorry

end NUMINAMATH_GPT_height_cylinder_l1347_134728


namespace NUMINAMATH_GPT_sum_of_squares_l1347_134791

theorem sum_of_squares (x y : ℝ) (h1 : x * y = 120) (h2 : x + y = 23) : x^2 + y^2 = 289 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_l1347_134791


namespace NUMINAMATH_GPT_eval_otimes_l1347_134754

def otimes (a b : ℝ) : ℝ := 2 * a + 5 * b

theorem eval_otimes : otimes 4 2 = 18 :=
by
  sorry

end NUMINAMATH_GPT_eval_otimes_l1347_134754


namespace NUMINAMATH_GPT_smallest_palindrome_not_five_digit_l1347_134783

theorem smallest_palindrome_not_five_digit (n : ℕ) (h : 100 ≤ n ∧ n ≤ 999 ∧ n % 10 = n / 100 ∧ n / 10 % 10 = n / 100 ∧ 103 * n < 10000) :
  n = 707 := by
sorry

end NUMINAMATH_GPT_smallest_palindrome_not_five_digit_l1347_134783


namespace NUMINAMATH_GPT_largest_fraction_proof_l1347_134712

theorem largest_fraction_proof 
  (w x y z : ℕ)
  (hw : 0 < w)
  (hx : w < x)
  (hy : x < y)
  (hz : y < z)
  (w_eq : w = 1)
  (x_eq : x = y - 1)
  (z_eq : z = y + 1)
  (y_eq : y = x!) : 
  (max (max (w + z) (w + x)) (max (x + z) (max (x + y) (y + z))) = 5 / 3) := 
sorry

end NUMINAMATH_GPT_largest_fraction_proof_l1347_134712


namespace NUMINAMATH_GPT_quadratic_equal_real_roots_l1347_134766

theorem quadratic_equal_real_roots (m : ℝ) (h : ∃ x : ℝ, x^2 - 4 * x + m = 1 ∧ 
                              (∀ y : ℝ, y ≠ x → y^2 - 4 * y + m ≠ 1)) : m = 5 :=
by sorry

end NUMINAMATH_GPT_quadratic_equal_real_roots_l1347_134766


namespace NUMINAMATH_GPT_unit_price_ratio_l1347_134744

theorem unit_price_ratio (v p : ℝ) (hv : 0 < v) (hp : 0 < p) :
  (1.1 * p / (1.4 * v)) / (0.85 * p / (1.3 * v)) = 13 / 11 :=
by
  sorry

end NUMINAMATH_GPT_unit_price_ratio_l1347_134744


namespace NUMINAMATH_GPT_last_box_weight_l1347_134700

theorem last_box_weight (a b c : ℕ) (h1 : a = 2) (h2 : b = 11) (h3 : a + b + c = 18) : c = 5 :=
by
  sorry

end NUMINAMATH_GPT_last_box_weight_l1347_134700


namespace NUMINAMATH_GPT_product_of_two_numbers_l1347_134762

theorem product_of_two_numbers (x y : ℝ) (h1 : x - y = 11) (h2 : x^2 + y^2 = 205) : x * y = 42 :=
by
  sorry

end NUMINAMATH_GPT_product_of_two_numbers_l1347_134762


namespace NUMINAMATH_GPT_f_2010_plus_f_2011_l1347_134734

-- Definition of f being an odd function
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f (x)

-- Conditions in Lean 4
variables (f : ℝ → ℝ)

axiom f_odd : odd_function f
axiom f_symmetry : ∀ x, f (1 + x) = f (1 - x)
axiom f_1 : f 1 = 2

-- The theorem to be proved
theorem f_2010_plus_f_2011 : f (2010) + f (2011) = -2 :=
by
  sorry

end NUMINAMATH_GPT_f_2010_plus_f_2011_l1347_134734


namespace NUMINAMATH_GPT_piecewise_function_continuity_l1347_134741

theorem piecewise_function_continuity :
  (∃ a c : ℝ, (2 * a * 2 + 4 = 2^2 - 2) ∧ (4 - 2 = 3 * (-2) - c) ∧ a + c = -17 / 2) :=
by
  sorry

end NUMINAMATH_GPT_piecewise_function_continuity_l1347_134741


namespace NUMINAMATH_GPT_candles_must_be_odd_l1347_134739

theorem candles_must_be_odd (n k : ℕ) (h : n * k = (n * (n + 1)) / 2) : n % 2 = 1 :=
by
  -- Given that the total burn time for all n candles = k * n
  -- And the sum of the first n natural numbers = (n * (n + 1)) / 2
  -- We have the hypothesis h: n * k = (n * (n + 1)) / 2
  -- We need to prove that n must be odd
  sorry

end NUMINAMATH_GPT_candles_must_be_odd_l1347_134739


namespace NUMINAMATH_GPT_range_of_a_l1347_134729

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x + 3

theorem range_of_a (a : ℝ) : (∀ x ∈ Set.Icc (0 : ℝ) a, f x ≤ 3) ∧ (∃ x ∈ Set.Icc (0 : ℝ) a, f x = 3) ∧ (∀ x ∈ Set.Icc (0 : ℝ) a, f x ≥ 2) ∧ (∃ x ∈ Set.Icc (0 : ℝ) a, f x = 2) ↔ 1 ≤ a ∧ a ≤ 2 := 
by 
  sorry

end NUMINAMATH_GPT_range_of_a_l1347_134729


namespace NUMINAMATH_GPT_min_equal_area_triangles_l1347_134711

theorem min_equal_area_triangles (chessboard_area missing_area : ℕ) (total_area : ℕ := chessboard_area - missing_area) 
(H1 : chessboard_area = 64) (H2 : missing_area = 1) : 
∃ n : ℕ, n = 18 ∧ (total_area = 63) → total_area / ((7:ℕ)/2) = n := 
sorry

end NUMINAMATH_GPT_min_equal_area_triangles_l1347_134711


namespace NUMINAMATH_GPT_spadesuit_problem_l1347_134738

-- Define the spadesuit operation
def spadesuit (a b : ℝ) : ℝ := abs (a - b)

-- Theorem statement
theorem spadesuit_problem : spadesuit (spadesuit 2 3) (spadesuit 6 (spadesuit 9 4)) = 0 := 
sorry

end NUMINAMATH_GPT_spadesuit_problem_l1347_134738


namespace NUMINAMATH_GPT_train_crossing_time_is_correct_l1347_134742

noncomputable def train_crossing_time (train_length bridge_length : ℝ) (train_speed_kmph : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  total_distance / train_speed_mps

theorem train_crossing_time_is_correct :
  train_crossing_time 250 180 120 = 12.9 :=
by
  sorry

end NUMINAMATH_GPT_train_crossing_time_is_correct_l1347_134742
