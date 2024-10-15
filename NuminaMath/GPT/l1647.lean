import Mathlib

namespace NUMINAMATH_GPT_tom_finishes_in_6_years_l1647_164778

/-- Combined program years for BS and Ph.D. -/
def BS_years : ℕ := 3
def PhD_years : ℕ := 5

/-- Total combined program time -/
def total_program_years : ℕ := BS_years + PhD_years

/-- Tom's time multiplier -/
def tom_time_multiplier : ℚ := 3 / 4

/-- Tom's total time to finish the program -/
def tom_total_time : ℚ := tom_time_multiplier * total_program_years

theorem tom_finishes_in_6_years : tom_total_time = 6 := 
by 
  -- implementation of the proof is to be filled in here
  sorry

end NUMINAMATH_GPT_tom_finishes_in_6_years_l1647_164778


namespace NUMINAMATH_GPT_wrapping_paper_area_l1647_164755

theorem wrapping_paper_area (l w h : ℝ) (hlw : l > w) (hwh : w > h) (hl : l = 2 * w) : 
    (∃ a : ℝ, a = 5 * w^2 + h^2) :=
by 
  sorry

end NUMINAMATH_GPT_wrapping_paper_area_l1647_164755


namespace NUMINAMATH_GPT_solution_set_f_inequality_l1647_164718

noncomputable def f (x : ℝ) : ℝ := 
if x > 0 then 1 - 2^(-x)
else if x < 0 then 2^x - 1
else 0

def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = -f x

theorem solution_set_f_inequality : 
  is_odd_function f →
  {x | f x < -1/2} = {x | x < -1} := 
by
  sorry

end NUMINAMATH_GPT_solution_set_f_inequality_l1647_164718


namespace NUMINAMATH_GPT_tan_3theta_l1647_164757

-- Let θ be an angle such that tan θ = 3.
variable (θ : ℝ)
noncomputable def tan_theta : ℝ := 3

-- Claim: tan(3 * θ) = 9/13
theorem tan_3theta :
  Real.tan (3 * θ) = 9 / 13 :=
by
  sorry

end NUMINAMATH_GPT_tan_3theta_l1647_164757


namespace NUMINAMATH_GPT_simplify_expression_l1647_164788

theorem simplify_expression (w : ℝ) : 3 * w + 6 * w + 9 * w + 12 * w + 15 * w + 18 = 45 * w + 18 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1647_164788


namespace NUMINAMATH_GPT_trapezoid_triangle_area_ratio_l1647_164713

/-- Given a trapezoid with triangles ABC and ADC such that the ratio of their areas is 4:1 and AB + CD = 150 cm.
Prove that the length of segment AB is 120 cm. --/
theorem trapezoid_triangle_area_ratio
  (h ABC_area ADC_area : ℕ)
  (AB CD : ℕ)
  (h_ratio : ABC_area / ADC_area = 4)
  (area_ABC : ABC_area = AB * h / 2)
  (area_ADC : ADC_area = CD * h / 2)
  (h_sum : AB + CD = 150) :
  AB = 120 := 
sorry

end NUMINAMATH_GPT_trapezoid_triangle_area_ratio_l1647_164713


namespace NUMINAMATH_GPT_c_in_terms_of_t_l1647_164760

theorem c_in_terms_of_t (t a b c : ℝ) (h_t_ne_zero : t ≠ 0)
    (h1 : t^3 + a * t = 0)
    (h2 : b * t^2 + c = 0)
    (h3 : 3 * t^2 + a = 2 * b * t) :
    c = -t^3 :=
by
sorry

end NUMINAMATH_GPT_c_in_terms_of_t_l1647_164760


namespace NUMINAMATH_GPT_largest_number_among_l1647_164733

theorem largest_number_among (π: ℝ) (sqrt_2: ℝ) (neg_2: ℝ) (three: ℝ)
  (h1: 3.14 ≤ π)
  (h2: 1 < sqrt_2 ∧ sqrt_2 < 2)
  (h3: neg_2 < 1)
  (h4: 3 < π) :
  (neg_2 < sqrt_2) ∧ (sqrt_2 < 3) ∧ (3 < π) :=
by {
  sorry
}

end NUMINAMATH_GPT_largest_number_among_l1647_164733


namespace NUMINAMATH_GPT_seventh_observation_is_eight_l1647_164774

theorem seventh_observation_is_eight
  (s₆ : ℕ)
  (a₆ : ℕ)
  (s₇ : ℕ)
  (a₇ : ℕ)
  (h₁ : s₆ = 6 * a₆)
  (h₂ : a₆ = 15)
  (h₃ : s₇ = 7 * a₇)
  (h₄ : a₇ = 14) :
  s₇ - s₆ = 8 :=
by
  -- Place proof here
  sorry

end NUMINAMATH_GPT_seventh_observation_is_eight_l1647_164774


namespace NUMINAMATH_GPT_smallest_N_divisibility_l1647_164737

theorem smallest_N_divisibility :
  ∃ N : ℕ, 
    (N + 2) % 2 = 0 ∧
    (N + 3) % 3 = 0 ∧
    (N + 4) % 4 = 0 ∧
    (N + 5) % 5 = 0 ∧
    (N + 6) % 6 = 0 ∧
    (N + 7) % 7 = 0 ∧
    (N + 8) % 8 = 0 ∧
    (N + 9) % 9 = 0 ∧
    (N + 10) % 10 = 0 ∧
    N = 2520 := 
sorry

end NUMINAMATH_GPT_smallest_N_divisibility_l1647_164737


namespace NUMINAMATH_GPT_point_in_fourth_quadrant_l1647_164792

theorem point_in_fourth_quadrant (m n : ℝ) (h₁ : m < 0) (h₂ : n > 0) : 
  2 * n - m > 0 ∧ -n + m < 0 := by
  sorry

end NUMINAMATH_GPT_point_in_fourth_quadrant_l1647_164792


namespace NUMINAMATH_GPT_plates_added_before_topple_l1647_164783

theorem plates_added_before_topple (init_plates add_first add_total : ℕ) (h : init_plates = 27) (h1 : add_first = 37) (h2 : add_total = 83) : 
  add_total - (init_plates + add_first) = 19 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_plates_added_before_topple_l1647_164783


namespace NUMINAMATH_GPT_range_of_m_l1647_164748

theorem range_of_m (m : ℝ) : 
  (∀ x y : ℝ, 0 < x → 0 < y → x < y → x^(m-1) > y^(m-1)) → m < 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1647_164748


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1647_164777

theorem solution_set_of_inequality :
  ∀ (x : ℝ), abs (2 * x + 1) < 3 ↔ -2 < x ∧ x < 1 :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1647_164777


namespace NUMINAMATH_GPT_total_cars_l1647_164706

-- Definitions for the conditions
def cathy_cars : ℕ := 5
def lindsey_cars : ℕ := cathy_cars + 4
def carol_cars : ℕ := 2 * cathy_cars
def susan_cars : ℕ := carol_cars - 2

-- Lean theorem statement
theorem total_cars : cathy_cars + lindsey_cars + carol_cars + susan_cars = 32 := by
  sorry

end NUMINAMATH_GPT_total_cars_l1647_164706


namespace NUMINAMATH_GPT_sectors_not_equal_l1647_164735

theorem sectors_not_equal (a1 a2 a3 a4 a5 a6 : ℕ) :
  ¬(∃ k : ℕ, (∀ n : ℕ, n = k) ↔
    ∃ m, (a1 + m) = k ∧ (a2 + m) = k ∧ (a3 + m) = k ∧ 
         (a4 + m) = k ∧ (a5 + m) = k ∧ (a6 + m) = k) :=
sorry

end NUMINAMATH_GPT_sectors_not_equal_l1647_164735


namespace NUMINAMATH_GPT_socorro_training_days_l1647_164744

def total_hours := 5
def minutes_per_hour := 60
def total_training_minutes := total_hours * minutes_per_hour

def minutes_multiplication_per_day := 10
def minutes_division_per_day := 20
def daily_training_minutes := minutes_multiplication_per_day + minutes_division_per_day

theorem socorro_training_days:
  total_training_minutes / daily_training_minutes = 10 :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_socorro_training_days_l1647_164744


namespace NUMINAMATH_GPT_certain_number_x_l1647_164710

theorem certain_number_x (p q x : ℕ) (hp : p > 1) (hq : q > 1)
  (h_eq : x * (p + 1) = 21 * (q + 1)) 
  (h_sum : p + q = 36) : x = 245 := 
by 
  sorry

end NUMINAMATH_GPT_certain_number_x_l1647_164710


namespace NUMINAMATH_GPT_parallelogram_altitude_base_ratio_l1647_164709

theorem parallelogram_altitude_base_ratio 
  (area base : ℕ) (h : ℕ) 
  (h_base : base = 9)
  (h_area : area = 162)
  (h_area_eq : area = base * h) : 
  h / base = 2 := 
by 
  -- placeholder for the proof
  sorry

end NUMINAMATH_GPT_parallelogram_altitude_base_ratio_l1647_164709


namespace NUMINAMATH_GPT_quadratic_inequality_iff_abs_a_le_2_l1647_164714

theorem quadratic_inequality_iff_abs_a_le_2 (a : ℝ) :
  (|a| ≤ 2) ↔ (∀ x : ℝ, x^2 + a * x + 1 ≥ 0) :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_iff_abs_a_le_2_l1647_164714


namespace NUMINAMATH_GPT_garrett_cats_count_l1647_164704

def number_of_cats_sheridan : ℕ := 11
def difference_in_cats : ℕ := 13

theorem garrett_cats_count (G : ℕ) (h : G - number_of_cats_sheridan = difference_in_cats) : G = 24 :=
by
  sorry

end NUMINAMATH_GPT_garrett_cats_count_l1647_164704


namespace NUMINAMATH_GPT_bug_crawl_distance_l1647_164773

-- Define the conditions
def initial_position : ℤ := -2
def first_move : ℤ := -6
def second_move : ℤ := 5

-- Define the absolute difference function (distance on a number line)
def abs_diff (a b : ℤ) : ℤ :=
  abs (b - a)

-- Define the total distance crawled function
def total_distance (p1 p2 p3 : ℤ) : ℤ :=
  abs_diff p1 p2 + abs_diff p2 p3

-- Prove that total distance starting at -2, moving to -6, and then to 5 is 15 units
theorem bug_crawl_distance : total_distance initial_position first_move second_move = 15 := by
  sorry

end NUMINAMATH_GPT_bug_crawl_distance_l1647_164773


namespace NUMINAMATH_GPT_solve_inequalities_l1647_164753

theorem solve_inequalities (x : ℝ) (h1 : x - 2 > 1) (h2 : x < 4) : 3 < x ∧ x < 4 :=
  sorry

end NUMINAMATH_GPT_solve_inequalities_l1647_164753


namespace NUMINAMATH_GPT_total_pages_of_book_l1647_164700

-- Definitions for the conditions
def firstChapterPages : Nat := 66
def secondChapterPages : Nat := 35
def thirdChapterPages : Nat := 24

-- Theorem stating the main question and answer
theorem total_pages_of_book : firstChapterPages + secondChapterPages + thirdChapterPages = 125 := by
  -- Proof will be provided here
  sorry

end NUMINAMATH_GPT_total_pages_of_book_l1647_164700


namespace NUMINAMATH_GPT_digit_sum_of_nines_l1647_164752

theorem digit_sum_of_nines (k : ℕ) (n : ℕ) (h : n = 9 * (10^k - 1) / 9):
  (8 + 9 * (k - 1) + 1 = 500) → k = 55 := 
by 
  sorry

end NUMINAMATH_GPT_digit_sum_of_nines_l1647_164752


namespace NUMINAMATH_GPT_problem_statement_l1647_164763

theorem problem_statement
  (g : ℝ → ℝ)
  (p q r s : ℝ)
  (h_roots : ∃ n1 n2 n3 n4 : ℕ, 
                ∀ x, g x = (x + 2 * n1) * (x + 2 * n2) * (x + 2 * n3) * (x + 2 * n4))
  (h_pqrs : p + q + r + s = 2552)
  (h_g : ∀ x, g x = x^4 + p * x^3 + q * x^2 + r * x + s) :
  s = 3072 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1647_164763


namespace NUMINAMATH_GPT_problem_equivalent_l1647_164717

theorem problem_equivalent :
  ∀ m n : ℤ, |m - n| = n - m ∧ |m| = 4 ∧ |n| = 3 → m + n = -1 ∨ m + n = -7 :=
by
  intros m n h
  have h1 : |m - n| = n - m := h.1
  have h2 : |m| = 4 := h.2.1
  have h3 : |n| = 3 := h.2.2
  sorry

end NUMINAMATH_GPT_problem_equivalent_l1647_164717


namespace NUMINAMATH_GPT_problem_l1647_164715

-- Define the problem conditions and the statement that needs to be proved
theorem problem:
  ∀ (x : ℝ), (x ∈ Set.Icc (-1) m) ∧ ((1 - (-1)) / (m - (-1)) = 2 / 5) → m = 4 := by
  sorry

end NUMINAMATH_GPT_problem_l1647_164715


namespace NUMINAMATH_GPT_periodic_sequence_a2019_l1647_164711

theorem periodic_sequence_a2019 :
  (∃ (a : ℕ → ℤ),
    a 1 = 1 ∧ a 2 = 1 ∧ a 3 = -1 ∧ 
    (∀ n : ℕ, n ≥ 4 → a n = a (n-1) * a (n-3)) ∧
    a 2019 = -1) :=
sorry

end NUMINAMATH_GPT_periodic_sequence_a2019_l1647_164711


namespace NUMINAMATH_GPT_avg_salary_of_employees_is_1500_l1647_164747

-- Definitions for conditions
def num_employees : ℕ := 20
def num_people_incl_manager : ℕ := 21
def manager_salary : ℝ := 4650
def salary_increase : ℝ := 150

-- Definition for average salary of employees excluding the manager
def avg_salary_employees (A : ℝ) : Prop :=
    21 * (A + salary_increase) = 20 * A + manager_salary

-- The target proof statement
theorem avg_salary_of_employees_is_1500 :
  ∃ A : ℝ, avg_salary_employees A ∧ A = 1500 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_avg_salary_of_employees_is_1500_l1647_164747


namespace NUMINAMATH_GPT_no_infinite_positive_integer_sequence_l1647_164720

theorem no_infinite_positive_integer_sequence (a : ℕ → ℕ) :
  ¬(∀ n, a (n - 1) ^ 2 ≥ 2 * a n * a (n + 2)) :=
sorry

end NUMINAMATH_GPT_no_infinite_positive_integer_sequence_l1647_164720


namespace NUMINAMATH_GPT_students_in_first_bus_l1647_164707

theorem students_in_first_bus (total_buses : ℕ) (avg_students_per_bus : ℕ) 
(avg_remaining_students : ℕ) (num_remaining_buses : ℕ) 
(h1 : total_buses = 6) 
(h2 : avg_students_per_bus = 28) 
(h3 : avg_remaining_students = 26) 
(h4 : num_remaining_buses = 5) :
  (total_buses * avg_students_per_bus - num_remaining_buses * avg_remaining_students = 38) :=
by
  sorry

end NUMINAMATH_GPT_students_in_first_bus_l1647_164707


namespace NUMINAMATH_GPT_number_is_375_l1647_164780

theorem number_is_375 (x : ℝ) (h : (40 / 100) * x = (30 / 100) * 50) : x = 37.5 :=
sorry

end NUMINAMATH_GPT_number_is_375_l1647_164780


namespace NUMINAMATH_GPT_jellybeans_condition_l1647_164703

theorem jellybeans_condition (n : ℕ) (h1 : n ≥ 150) (h2 : n % 15 = 14) : n = 164 :=
sorry

end NUMINAMATH_GPT_jellybeans_condition_l1647_164703


namespace NUMINAMATH_GPT_divisible_by_72_l1647_164759

theorem divisible_by_72 (a b : ℕ) (h1 : 0 ≤ a ∧ a < 10) (h2 : 0 ≤ b ∧ b < 10) :
  (b = 2 ∧ a = 3) → (a * 10000 + 6 * 1000 + 7 * 100 + 9 * 10 + b) % 72 = 0 :=
by
  sorry

end NUMINAMATH_GPT_divisible_by_72_l1647_164759


namespace NUMINAMATH_GPT_hyperbola_foci_coords_l1647_164789

theorem hyperbola_foci_coords :
  ∀ x y, (x^2) / 8 - (y^2) / 17 = 1 → (x, y) = (5, 0) ∨ (x, y) = (-5, 0) :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_foci_coords_l1647_164789


namespace NUMINAMATH_GPT_negation_of_proposition_l1647_164771

variable (f : ℕ+ → ℕ)

theorem negation_of_proposition :
  (¬ ∀ n : ℕ+, f n ≤ n) ↔ (∃ n : ℕ+, f n > n) :=
by sorry

end NUMINAMATH_GPT_negation_of_proposition_l1647_164771


namespace NUMINAMATH_GPT_neither_5_nor_6_nice_1200_l1647_164756

def is_k_nice (N k : ℕ) : Prop := N % k = 1

def count_k_nice_up_to (k n : ℕ) : ℕ :=
(n + (k - 1)) / k

def count_neither_5_nor_6_nice_up_to (n : ℕ) : ℕ :=
  let count_5_nice := count_k_nice_up_to 5 n
  let count_6_nice := count_k_nice_up_to 6 n
  let count_5_and_6_nice := count_k_nice_up_to 30 n
  n - (count_5_nice + count_6_nice - count_5_and_6_nice)

theorem neither_5_nor_6_nice_1200 : count_neither_5_nor_6_nice_up_to 1200 = 800 := 
by
  sorry

end NUMINAMATH_GPT_neither_5_nor_6_nice_1200_l1647_164756


namespace NUMINAMATH_GPT_sphere_volume_l1647_164726

theorem sphere_volume (S : ℝ) (hS : S = 4 * π) : ∃ V : ℝ, V = (4 / 3) * π := 
by
  sorry

end NUMINAMATH_GPT_sphere_volume_l1647_164726


namespace NUMINAMATH_GPT_sums_of_adjacent_cells_l1647_164751

theorem sums_of_adjacent_cells (N : ℕ) (h : N ≥ 2) :
  ∃ (f : ℕ → ℕ → ℝ), (∀ i j, 1 ≤ i ∧ i < N → 1 ≤ j ∧ j < N → 
    (∃ (s : ℕ), 1 ≤ s ∧ s ≤ 2*(N-1)*N ∧ s = f i j + f (i + 1) j) ∧
    (∃ (s : ℕ), 1 ≤ s ∧ s ≤ 2*(N-1)*N ∧ s = f i j + f i (j + 1))) := sorry

end NUMINAMATH_GPT_sums_of_adjacent_cells_l1647_164751


namespace NUMINAMATH_GPT_max_tan2alpha_l1647_164795

variable (α β : ℝ)
variable (hα : 0 < α ∧ α < Real.pi / 2)
variable (hβ : 0 < β ∧ β < Real.pi / 2)
variable (h : Real.tan (α + β) = 2 * Real.tan β)

theorem max_tan2alpha : 
    Real.tan (2 * α) = 4 * Real.sqrt 2 / 7 := 
by 
  sorry

end NUMINAMATH_GPT_max_tan2alpha_l1647_164795


namespace NUMINAMATH_GPT_find_g5_l1647_164731

variable (g : ℝ → ℝ)

-- Formal definition of the condition for the function g in the problem statement.
def functional_eq_condition :=
  ∀ x : ℝ, g x + 3 * g (2 - x) = 4 * x^2

-- The main statement to prove g(5) = 1 under the given condition.
theorem find_g5 (h : functional_eq_condition g) :
  g 5 = 1 :=
sorry

end NUMINAMATH_GPT_find_g5_l1647_164731


namespace NUMINAMATH_GPT_number_of_mandatory_questions_correct_l1647_164786

-- Definitions and conditions
def num_mandatory_questions (x : ℕ) (k : ℕ) (y : ℕ) (m : ℕ) : Prop :=
  (3 * k - 2 * (x - k) + 5 * m = 49) ∧
  (k + m = 15) ∧
  (y = 25 - x)

-- Proof statement
theorem number_of_mandatory_questions_correct :
  ∃ x k y m : ℕ, num_mandatory_questions x k y m ∧ x = 13 :=
by
  sorry

end NUMINAMATH_GPT_number_of_mandatory_questions_correct_l1647_164786


namespace NUMINAMATH_GPT_midpoint_sum_eq_six_l1647_164781

theorem midpoint_sum_eq_six :
  let x1 := 6
  let y1 := 12
  let x2 := 0
  let y2 := -6
  let midpoint_x := (x1 + x2) / 2 
  let midpoint_y := (y1 + y2) / 2 
  (midpoint_x + midpoint_y) = 6 :=
by
  let x1 := 6
  let y1 := 12
  let x2 := 0
  let y2 := -6
  let midpoint_x := (x1 + x2) / 2 
  let midpoint_y := (y1 + y2) / 2 
  sorry

end NUMINAMATH_GPT_midpoint_sum_eq_six_l1647_164781


namespace NUMINAMATH_GPT_total_revenue_correct_l1647_164721

-- Definitions and conditions
def number_of_fair_tickets : ℕ := 60
def price_per_fair_ticket : ℕ := 15
def price_per_baseball_ticket : ℕ := 10
def number_of_baseball_tickets : ℕ := number_of_fair_tickets / 3

-- Calculate revenues
def revenue_from_fair_tickets : ℕ := number_of_fair_tickets * price_per_fair_ticket
def revenue_from_baseball_tickets : ℕ := number_of_baseball_tickets * price_per_baseball_ticket
def total_revenue : ℕ := revenue_from_fair_tickets + revenue_from_baseball_tickets

-- Proof statement
theorem total_revenue_correct : total_revenue = 1100 := by
  sorry

end NUMINAMATH_GPT_total_revenue_correct_l1647_164721


namespace NUMINAMATH_GPT_chewbacca_pack_size_l1647_164782

/-- Given Chewbacca has 20 pieces of cherry gum and 30 pieces of grape gum,
if losing one pack of cherry gum keeps the ratio of cherry to grape gum the same
as when finding 5 packs of grape gum, determine the number of pieces x in each 
complete pack of gum. We show that x = 14. -/
theorem chewbacca_pack_size :
  ∃ (x : ℕ), (20 - x) * (30 + 5 * x) = 20 * 30 ∧ ∀ (y : ℕ), (20 - y) * (30 + 5 * y) = 600 → y = 14 :=
by
  sorry

end NUMINAMATH_GPT_chewbacca_pack_size_l1647_164782


namespace NUMINAMATH_GPT_geometric_series_first_term_l1647_164762

theorem geometric_series_first_term (r : ℚ) (S : ℚ) (a : ℚ) (h_r : r = 1 / 4) (h_S : S = 80) (h_sum : S = a / (1 - r)) : a = 60 :=
by 
  sorry

end NUMINAMATH_GPT_geometric_series_first_term_l1647_164762


namespace NUMINAMATH_GPT_store_total_profit_l1647_164768

theorem store_total_profit
  (purchase_price : ℕ)
  (selling_price_total : ℕ)
  (max_selling_price : ℕ)
  (profit : ℕ)
  (N : ℕ)
  (selling_price_per_card : ℕ)
  (h1 : purchase_price = 21)
  (h2 : selling_price_total = 1457)
  (h3 : max_selling_price = 2 * purchase_price)
  (h4 : selling_price_per_card * N = selling_price_total)
  (h5 : selling_price_per_card ≤ max_selling_price)
  (h_profit : profit = (selling_price_per_card - purchase_price) * N)
  : profit = 470 :=
sorry

end NUMINAMATH_GPT_store_total_profit_l1647_164768


namespace NUMINAMATH_GPT_probability_first_prize_both_distribution_of_X_l1647_164702

-- Definitions for the conditions
def total_students : ℕ := 500
def male_students : ℕ := 200
def female_students : ℕ := 300

def male_first_prize : ℕ := 10
def female_first_prize : ℕ := 25

def male_second_prize : ℕ := 15
def female_second_prize : ℕ := 25

def male_third_prize : ℕ := 15
def female_third_prize : ℕ := 40

-- Part (1): Prove the probability that both selected students receive the first prize is 1/240.
theorem probability_first_prize_both :
  (male_first_prize / male_students : ℚ) * (female_first_prize / female_students : ℚ) = 1 / 240 := 
sorry

-- Part (2): Prove the distribution of X.
def P_male_award : ℚ := (male_first_prize + male_second_prize + male_third_prize) / male_students
def P_female_award : ℚ := (female_first_prize + female_second_prize + female_third_prize) / female_students

theorem distribution_of_X :
  ∀ X : ℕ, X = 0 ∧ ((1 - P_male_award) * (1 - P_female_award) = 28 / 50) ∨ 
           X = 1 ∧ ((1 - P_male_award) * (1 - P_female_award) + (P_male_award * (1 - P_female_award)) + ((1 - P_male_award) * P_female_award) = 19 / 50) ∨ 
           X = 2 ∧ (P_male_award * P_female_award = 3 / 50) :=
sorry

end NUMINAMATH_GPT_probability_first_prize_both_distribution_of_X_l1647_164702


namespace NUMINAMATH_GPT_range_of_m_l1647_164779

open Real

def p (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 1 > m
def q (m : ℝ) : Prop := (2 - m) > 0

theorem range_of_m (m : ℝ) : (p m ∨ q m) ∧ ¬ (p m ∧ q m) → 1 ≤ m ∧ m < 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1647_164779


namespace NUMINAMATH_GPT_total_marbles_l1647_164787

-- Define the number of marbles Mary has
def marblesMary : Nat := 9 

-- Define the number of marbles Joan has
def marblesJoan : Nat := 3 

-- Theorem to prove the total number of marbles
theorem total_marbles : marblesMary + marblesJoan = 12 := 
by sorry

end NUMINAMATH_GPT_total_marbles_l1647_164787


namespace NUMINAMATH_GPT_nadia_flower_shop_l1647_164705

theorem nadia_flower_shop (roses lilies cost_per_rose cost_per_lily cost_roses cost_lilies total_cost : ℕ)
  (h1 : roses = 20)
  (h2 : lilies = 3 * roses / 4)
  (h3 : cost_per_rose = 5)
  (h4 : cost_per_lily = 2 * cost_per_rose)
  (h5 : cost_roses = roses * cost_per_rose)
  (h6 : cost_lilies = lilies * cost_per_lily)
  (h7 : total_cost = cost_roses + cost_lilies) :
  total_cost = 250 :=
by
  sorry

end NUMINAMATH_GPT_nadia_flower_shop_l1647_164705


namespace NUMINAMATH_GPT_candles_shared_equally_l1647_164796

theorem candles_shared_equally :
  ∀ (Aniyah Ambika Bree Caleb : ℕ),
  Aniyah = 6 * Ambika → Ambika = 4 → Bree = 0 → Caleb = 0 →
  (Aniyah + Ambika + Bree + Caleb) / 4 = 7 :=
by
  intros Aniyah Ambika Bree Caleb h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_candles_shared_equally_l1647_164796


namespace NUMINAMATH_GPT_minimum_squares_and_perimeter_l1647_164767

theorem minimum_squares_and_perimeter 
  (length width : ℕ) 
  (h_length : length = 90) 
  (h_width : width = 42) 
  (h_gcd : Nat.gcd length width = 6) 
  : 
  ((length / Nat.gcd length width) * (width / Nat.gcd length width) = 105) ∧ 
  (105 * (4 * Nat.gcd length width) = 2520) := 
by 
  sorry

end NUMINAMATH_GPT_minimum_squares_and_perimeter_l1647_164767


namespace NUMINAMATH_GPT_cos_triple_angle_l1647_164770

theorem cos_triple_angle (θ : ℝ) (h : Real.cos θ = -1 / 3) : Real.cos (3 * θ) = 23 / 27 :=
by
  sorry

end NUMINAMATH_GPT_cos_triple_angle_l1647_164770


namespace NUMINAMATH_GPT_summation_values_l1647_164729

theorem summation_values (x y : ℝ) (h1 : x = y * (3 - y) ^ 2) (h2 : y = x * (3 - x) ^ 2) : 
  x + y = 0 ∨ x + y = 3 ∨ x + y = 4 ∨ x + y = 5 ∨ x + y = 8 :=
sorry

end NUMINAMATH_GPT_summation_values_l1647_164729


namespace NUMINAMATH_GPT_initial_ratio_proof_l1647_164722

variable (p q : ℕ) -- Define p and q as non-negative integers

-- Condition: The initial total volume of the mixture is 30 liters
def initial_volume (p q : ℕ) : Prop := p + q = 30

-- Condition: Adding 12 liters of q changes the ratio to 3:4
def new_ratio (p q : ℕ) : Prop := p * 4 = (q + 12) * 3

-- The final goal: prove the initial ratio is 3:2
def initial_ratio (p q : ℕ) : Prop := p * 2 = q * 3

-- The main proof problem statement
theorem initial_ratio_proof (p q : ℕ) 
  (h1 : initial_volume p q) 
  (h2 : new_ratio p q) : initial_ratio p q :=
  sorry

end NUMINAMATH_GPT_initial_ratio_proof_l1647_164722


namespace NUMINAMATH_GPT_percentage_decrease_of_original_number_is_30_l1647_164712

theorem percentage_decrease_of_original_number_is_30 :
  ∀ (original_number : ℕ) (difference : ℕ) (percent_increase : ℚ) (percent_decrease : ℚ),
  original_number = 40 →
  percent_increase = 0.25 →
  difference = 22 →
  original_number + percent_increase * original_number - (original_number - percent_decrease * original_number) = difference →
  percent_decrease = 0.30 :=
by
  intros original_number difference percent_increase percent_decrease h_original h_increase h_diff h_eq
  sorry

end NUMINAMATH_GPT_percentage_decrease_of_original_number_is_30_l1647_164712


namespace NUMINAMATH_GPT_change_given_l1647_164761

-- Define the given conditions
def oranges_cost := 40
def apples_cost := 50
def mangoes_cost := 60
def initial_amount := 300

-- Calculate total cost of fruits
def total_fruits_cost := oranges_cost + apples_cost + mangoes_cost

-- Define the given change
def given_change := initial_amount - total_fruits_cost

-- Prove that the given change is equal to 150
theorem change_given (h_oranges : oranges_cost = 40)
                     (h_apples : apples_cost = 50)
                     (h_mangoes : mangoes_cost = 60)
                     (h_initial : initial_amount = 300) :
  given_change = 150 :=
by
  -- Proof is omitted, indicated by sorry
  sorry

end NUMINAMATH_GPT_change_given_l1647_164761


namespace NUMINAMATH_GPT_keith_picked_p_l1647_164794

-- Definitions of the given conditions
def p_j : ℕ := 46  -- Jason's pears
def p_m : ℕ := 12  -- Mike's pears
def p_t : ℕ := 105 -- Total pears picked

-- The theorem statement
theorem keith_picked_p : p_t - (p_j + p_m) = 47 := by
  -- Proof part will be handled later
  sorry

end NUMINAMATH_GPT_keith_picked_p_l1647_164794


namespace NUMINAMATH_GPT_blue_notebook_cost_l1647_164730

theorem blue_notebook_cost
  (total_spent : ℕ)
  (total_notebooks : ℕ)
  (red_notebooks : ℕ)
  (red_notebook_cost : ℕ)
  (green_notebooks : ℕ)
  (green_notebook_cost : ℕ)
  (blue_notebook_cost : ℕ)
  (h₀ : total_spent = 37)
  (h₁ : total_notebooks = 12)
  (h₂ : red_notebooks = 3)
  (h₃ : red_notebook_cost = 4)
  (h₄ : green_notebooks = 2)
  (h₅ : green_notebook_cost = 2)
  (h₆ : total_spent = red_notebooks * red_notebook_cost + green_notebooks * green_notebook_cost + blue_notebook_cost * (total_notebooks - red_notebooks - green_notebooks)) :
  blue_notebook_cost = 3 := by
  sorry

end NUMINAMATH_GPT_blue_notebook_cost_l1647_164730


namespace NUMINAMATH_GPT_tan_arctan_five_twelfths_l1647_164790

theorem tan_arctan_five_twelfths : Real.tan (Real.arctan (5 / 12)) = 5 / 12 :=
by
  sorry

end NUMINAMATH_GPT_tan_arctan_five_twelfths_l1647_164790


namespace NUMINAMATH_GPT_find_n_coins_l1647_164716

def num_coins : ℕ := 5

theorem find_n_coins (n : ℕ) (h : (n^2 + n + 2) = 2^n) : n = num_coins :=
by {
  -- Proof to be filled in
  sorry
}

end NUMINAMATH_GPT_find_n_coins_l1647_164716


namespace NUMINAMATH_GPT_find_first_offset_l1647_164776

theorem find_first_offset
  (area : ℝ)
  (diagonal : ℝ)
  (offset2 : ℝ)
  (first_offset : ℝ)
  (h_area : area = 225)
  (h_diagonal : diagonal = 30)
  (h_offset2 : offset2 = 6)
  (h_formula : area = (diagonal * (first_offset + offset2)) / 2)
  : first_offset = 9 := by
  sorry

end NUMINAMATH_GPT_find_first_offset_l1647_164776


namespace NUMINAMATH_GPT_int_values_satisfying_l1647_164766

theorem int_values_satisfying (x : ℤ) : (∃ k : ℤ, (5 * x + 2) = 17 * k) ↔ (∃ m : ℤ, x = 17 * m + 3) :=
by
  sorry

end NUMINAMATH_GPT_int_values_satisfying_l1647_164766


namespace NUMINAMATH_GPT_R_depends_on_a_d_m_l1647_164740

theorem R_depends_on_a_d_m (a d m : ℝ) :
    let s1 := (m / 2) * (2 * a + (m - 1) * d)
    let s2 := m * (2 * a + (2 * m - 1) * d)
    let s3 := 2 * m * (2 * a + (4 * m - 1) * d)
    let R := s3 - 2 * s2 + s1
    R = m * (a + 12 * m * d - (d / 2)) := by
  sorry

end NUMINAMATH_GPT_R_depends_on_a_d_m_l1647_164740


namespace NUMINAMATH_GPT_prime_pairs_l1647_164798

theorem prime_pairs (p q : ℕ) : 
  p < 2005 → q < 2005 → 
  Prime p → Prime q → 
  (q ∣ p^2 + 4) → 
  (p ∣ q^2 + 4) → 
  (p = 2 ∧ q = 2) :=
by sorry

end NUMINAMATH_GPT_prime_pairs_l1647_164798


namespace NUMINAMATH_GPT_min_x_given_conditions_l1647_164797

theorem min_x_given_conditions :
  ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ (100 : ℚ) / 151 < y / x ∧ y / x < (200 : ℚ) / 251 ∧ x = 3 :=
by
  sorry

end NUMINAMATH_GPT_min_x_given_conditions_l1647_164797


namespace NUMINAMATH_GPT_smallest_and_largest_group_sizes_l1647_164749

theorem smallest_and_largest_group_sizes (S T : Finset ℕ) (hS : S.card + T.card = 20)
  (h_union: (S ∪ T) = (Finset.range 21) \ {0}) (h_inter: S ∩ T = ∅)
  (sum_S : S.sum id = 210 - T.sum id) (prod_T : T.prod id = 210 - S.sum id) :
  T.card = 3 ∨ T.card = 5 := 
sorry

end NUMINAMATH_GPT_smallest_and_largest_group_sizes_l1647_164749


namespace NUMINAMATH_GPT_fraction_simplification_l1647_164754

theorem fraction_simplification
  (a b c x : ℝ)
  (hb : b ≠ 0)
  (hxc : c ≠ 0)
  (h : x = a / b)
  (ha : a ≠ c * b) :
  (a + c * b) / (a - c * b) = (x + c) / (x - c) :=
by
  sorry

end NUMINAMATH_GPT_fraction_simplification_l1647_164754


namespace NUMINAMATH_GPT_cars_and_tourists_l1647_164708

theorem cars_and_tourists (n t : ℕ) (h : n * t = 737) : n = 11 ∧ t = 67 ∨ n = 67 ∧ t = 11 :=
by
  sorry

end NUMINAMATH_GPT_cars_and_tourists_l1647_164708


namespace NUMINAMATH_GPT_passing_marks_l1647_164723

theorem passing_marks :
  ∃ P T : ℝ, (0.2 * T = P - 40) ∧ (0.3 * T = P + 20) ∧ P = 160 :=
by
  sorry

end NUMINAMATH_GPT_passing_marks_l1647_164723


namespace NUMINAMATH_GPT_distance_between_stations_l1647_164734

-- distance calculation definitions
def distance (rate time : ℝ) := rate * time

-- problem conditions as definitions
def rate_slow := 20 -- km/hr
def rate_fast := 25 -- km/hr
def extra_distance := 50 -- km

-- final statement
theorem distance_between_stations :
  ∃ (D : ℝ) (T : ℝ),
    (distance rate_slow T = D) ∧
    (distance rate_fast T = D + extra_distance) ∧
    (D + (D + extra_distance) = 450) :=
by
  sorry

end NUMINAMATH_GPT_distance_between_stations_l1647_164734


namespace NUMINAMATH_GPT_parabola_trajectory_l1647_164772

theorem parabola_trajectory :
  ∀ P : ℝ × ℝ, (dist P (0, -1) + 1 = dist P (0, 3)) ↔ (P.1 ^ 2 = -8 * P.2) := by
  sorry

end NUMINAMATH_GPT_parabola_trajectory_l1647_164772


namespace NUMINAMATH_GPT_find_absolute_value_l1647_164775

theorem find_absolute_value (h k : ℤ) (h1 : 3 * (-3)^3 - h * (-3) + k = 0) (h2 : 3 * 2^3 - h * 2 + k = 0) : |3 * h - 2 * k| = 27 :=
by
  sorry

end NUMINAMATH_GPT_find_absolute_value_l1647_164775


namespace NUMINAMATH_GPT_price_difference_is_300_cents_l1647_164724

noncomputable def list_price : ℝ := 59.99
noncomputable def tech_bargains_price : ℝ := list_price - 15
noncomputable def digital_deal_price : ℝ := 0.7 * list_price
noncomputable def price_difference : ℝ := tech_bargains_price - digital_deal_price
noncomputable def price_difference_in_cents : ℝ := price_difference * 100

theorem price_difference_is_300_cents :
  price_difference_in_cents = 300 := by
  sorry

end NUMINAMATH_GPT_price_difference_is_300_cents_l1647_164724


namespace NUMINAMATH_GPT_g_pi_over_4_eq_neg_sqrt2_over_4_l1647_164728

noncomputable def g (x : Real) : Real := 
  Real.sqrt (5 * (Real.sin x)^4 + 4 * (Real.cos x)^2) - 
  Real.sqrt (6 * (Real.cos x)^4 + 4 * (Real.sin x)^2)

theorem g_pi_over_4_eq_neg_sqrt2_over_4 :
  g (Real.pi / 4) = - (Real.sqrt 2) / 4 := 
sorry

end NUMINAMATH_GPT_g_pi_over_4_eq_neg_sqrt2_over_4_l1647_164728


namespace NUMINAMATH_GPT_system_of_equations_implies_quadratic_l1647_164738

theorem system_of_equations_implies_quadratic (x y : ℝ) :
  (3 * x^2 + 9 * x + 4 * y + 2 = 0) ∧ (3 * x + y + 4 = 0) → (y^2 + 11 * y - 14 = 0) := by
  sorry

end NUMINAMATH_GPT_system_of_equations_implies_quadratic_l1647_164738


namespace NUMINAMATH_GPT_points_on_line_l1647_164719

theorem points_on_line (b m n : ℝ) (hA : m = -(-5) + b) (hB : n = -(4) + b) :
  m > n :=
by
  sorry

end NUMINAMATH_GPT_points_on_line_l1647_164719


namespace NUMINAMATH_GPT_compute_difference_of_squares_l1647_164739

theorem compute_difference_of_squares :
  262^2 - 258^2 = 2080 := 
by
  sorry

end NUMINAMATH_GPT_compute_difference_of_squares_l1647_164739


namespace NUMINAMATH_GPT_line_exists_symmetric_diagonals_l1647_164732

-- Define the initial conditions
def Circle_C1 (x y : ℝ) : Prop := x^2 + y^2 + 6 * x = 0
def Line_l1 (x y : ℝ) : Prop := y = 2 * x + 1

-- Define the symmetric circle C about the line l1
def Symmetric_Circle (x y : ℝ) : Prop := (x - 1)^2 + (y + 2)^2 = 9

-- Define the origion and intersection points
def Point_O : (ℝ × ℝ) := (0, 0)
def Point_Intersection (l : ℝ → ℝ) (A B : ℝ × ℝ) : Prop := ∃ x_A y_A x_B y_B : ℝ,
  l x_A = y_A ∧ l x_B = y_B ∧ Symmetric_Circle x_A y_A ∧ Symmetric_Circle x_B y_B

-- Define diagonal equality condition
def Diagonals_Equal (O A S B : ℝ × ℝ) : Prop := 
  let (xO, yO) := O
  let (xA, yA) := A
  let (xS, yS) := S
  let (xB, yB) := B
  (xA - xO)^2 + (yA - yO)^2 = (xB - xS)^2 + (yB - yS)^2

-- Prove existence of line where diagonals are equal and find the equation
theorem line_exists_symmetric_diagonals :
  ∃ l : ℝ → ℝ, (l (-1) = 0) ∧
    (∃ (A B S : ℝ × ℝ), Point_Intersection l A B ∧ Diagonals_Equal Point_O A S B) ∧
    (∀ x : ℝ, l x = x + 1) :=
by
  sorry

end NUMINAMATH_GPT_line_exists_symmetric_diagonals_l1647_164732


namespace NUMINAMATH_GPT_number_of_hens_is_50_l1647_164764

def number_goats : ℕ := 45
def number_camels : ℕ := 8
def number_keepers : ℕ := 15
def extra_feet : ℕ := 224

def total_heads (number_hens number_goats number_camels number_keepers : ℕ) : ℕ :=
  number_hens + number_goats + number_camels + number_keepers

def total_feet (number_hens number_goats number_camels number_keepers : ℕ) : ℕ :=
  2 * number_hens + 4 * number_goats + 4 * number_camels + 2 * number_keepers

theorem number_of_hens_is_50 (H : ℕ) :
  total_feet H number_goats number_camels number_keepers = (total_heads H number_goats number_camels number_keepers) + extra_feet → H = 50 :=
sorry

end NUMINAMATH_GPT_number_of_hens_is_50_l1647_164764


namespace NUMINAMATH_GPT_find_a_b_l1647_164791

def f (x a b : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

def f_derivative (x a b : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem find_a_b (a b : ℝ) (h1 : f 1 a b = 10) (h2 : f_derivative 1 a b = 0) : a = 4 ∧ b = -11 :=
sorry

end NUMINAMATH_GPT_find_a_b_l1647_164791


namespace NUMINAMATH_GPT_new_person_weight_l1647_164745

-- Define the conditions of the problem
variables (avg_weight : ℝ) (weight_replaced_person : ℝ) (num_persons : ℕ)
variable (weight_increase : ℝ)

-- Given conditions
def condition (avg_weight weight_replaced_person : ℝ) (num_persons : ℕ) (weight_increase : ℝ) : Prop :=
  num_persons = 10 ∧ weight_replaced_person = 60 ∧ weight_increase = 5

-- The proof problem
theorem new_person_weight (avg_weight weight_replaced_person : ℝ) (num_persons : ℕ)
  (weight_increase : ℝ) (h : condition avg_weight weight_replaced_person num_persons weight_increase) :
  weight_replaced_person + num_persons * weight_increase = 110 :=
sorry

end NUMINAMATH_GPT_new_person_weight_l1647_164745


namespace NUMINAMATH_GPT_pythagorean_theorem_l1647_164743

theorem pythagorean_theorem (a b c : ℝ) : (a^2 + b^2 = c^2) ↔ (a^2 + b^2 = c^2) :=
by sorry

end NUMINAMATH_GPT_pythagorean_theorem_l1647_164743


namespace NUMINAMATH_GPT_area_of_border_l1647_164742

theorem area_of_border (height_painting width_painting border_width : ℕ)
    (area_painting framed_height framed_width : ℕ)
    (H1 : height_painting = 12)
    (H2 : width_painting = 15)
    (H3 : border_width = 3)
    (H4 : area_painting = height_painting * width_painting)
    (H5 : framed_height = height_painting + 2 * border_width)
    (H6 : framed_width = width_painting + 2 * border_width)
    (area_framed : ℕ)
    (H7 : area_framed = framed_height * framed_width) :
    area_framed - area_painting = 198 := 
sorry

end NUMINAMATH_GPT_area_of_border_l1647_164742


namespace NUMINAMATH_GPT_effective_rate_proof_l1647_164758

noncomputable def nominal_rate : ℝ := 0.08
noncomputable def compounding_periods : ℕ := 2
noncomputable def effective_annual_rate (i : ℝ) (n : ℕ) : ℝ := (1 + i / n) ^ n - 1

theorem effective_rate_proof :
  effective_annual_rate nominal_rate compounding_periods = 0.0816 :=
by
  sorry

end NUMINAMATH_GPT_effective_rate_proof_l1647_164758


namespace NUMINAMATH_GPT_integer_solution_unique_l1647_164799

theorem integer_solution_unique (n : ℤ) : (⌊(n^2 : ℤ) / 5⌋ - ⌊n / 2⌋^2 = 3) ↔ n = 5 :=
by
  sorry

end NUMINAMATH_GPT_integer_solution_unique_l1647_164799


namespace NUMINAMATH_GPT_intersection_A_B_l1647_164750

def A : Set ℤ := {0, 1, 2}
def B : Set ℤ := {x | x - 2 < 0}

theorem intersection_A_B : A ∩ B = {0, 1} :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l1647_164750


namespace NUMINAMATH_GPT_printer_paper_last_days_l1647_164727

def packs : Nat := 2
def sheets_per_pack : Nat := 240
def prints_per_day : Nat := 80
def total_sheets : Nat := packs * sheets_per_pack
def number_of_days : Nat := total_sheets / prints_per_day

theorem printer_paper_last_days :
  number_of_days = 6 :=
by
  sorry

end NUMINAMATH_GPT_printer_paper_last_days_l1647_164727


namespace NUMINAMATH_GPT_find_x_value_l1647_164725

noncomputable def check_x (x : ℝ) : Prop :=
  (0 < x) ∧ (Real.sqrt (12 * x) * Real.sqrt (5 * x) * Real.sqrt (6 * x) * Real.sqrt (10 * x) = 10)

theorem find_x_value (x : ℝ) (h : check_x x) : x = 1 / 6 :=
by 
  sorry

end NUMINAMATH_GPT_find_x_value_l1647_164725


namespace NUMINAMATH_GPT_triangle_ABC_area_l1647_164765

open Real

-- Define points A, B, and C
structure Point :=
  (x: ℝ)
  (y: ℝ)

def A : Point := ⟨-1, 2⟩
def B : Point := ⟨8, 2⟩
def C : Point := ⟨6, -1⟩

-- Function to calculate the area of a triangle given vertices A, B, and C
noncomputable def triangle_area (A B C : Point) : ℝ := 
  abs ((B.x - A.x) * (C.y - A.y) - (C.x - A.x) * (B.y - A.y)) / 2

-- The statement to be proved
theorem triangle_ABC_area : triangle_area A B C = 13.5 :=
by
  sorry

end NUMINAMATH_GPT_triangle_ABC_area_l1647_164765


namespace NUMINAMATH_GPT_probability_of_inverse_proportion_l1647_164701

def points : List (ℝ × ℝ) :=
  [(0.5, -4.5), (1, -4), (1.5, -3.5), (2, -3), (2.5, -2.5), (3, -2), (3.5, -1.5),
   (4, -1), (4.5, -0.5), (5, 0)]

def inverse_proportion_pairs : List ((ℝ × ℝ) × (ℝ × ℝ)) :=
  [((0.5, -4.5), (4.5, -0.5)), ((1, -4), (4, -1)), ((1.5, -3.5), (3.5, -1.5)), ((2, -3), (3, -2))]

theorem probability_of_inverse_proportion:
  let num_pairs := List.length points * (List.length points - 1)
  let favorable_pairs := 2 * List.length inverse_proportion_pairs
  favorable_pairs / num_pairs = (4 : ℚ) / 45 := by
  sorry

end NUMINAMATH_GPT_probability_of_inverse_proportion_l1647_164701


namespace NUMINAMATH_GPT_solve_for_x_l1647_164736

theorem solve_for_x (x: ℚ) (h: (3/5 - 1/4) = 4/x) : x = 80/7 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1647_164736


namespace NUMINAMATH_GPT_necessarily_true_statement_l1647_164784

-- Define the four statements as propositions
def Statement1 (d : ℕ) : Prop := d = 2
def Statement2 (d : ℕ) : Prop := d ≠ 3
def Statement3 (d : ℕ) : Prop := d = 5
def Statement4 (d : ℕ) : Prop := d % 2 = 0

-- The main theorem stating that given one of the statements is false, Statement3 is necessarily true
theorem necessarily_true_statement (d : ℕ) 
  (h1 : Not (Statement1 d ∧ Statement2 d ∧ Statement3 d ∧ Statement4 d) 
    ∨ Not (Statement1 d ∧ Statement2 d ∧ Statement3 d ∧ ¬ Statement4 d) 
    ∨ Not (Statement1 d ∧ Statement2 d ∧ ¬ Statement3 d ∧ Statement4 d) 
    ∨ Not (Statement1 d ∧ ¬ Statement2 d ∧ Statement3 d ∧ Statement4 d)):
  Statement2 d :=
sorry

end NUMINAMATH_GPT_necessarily_true_statement_l1647_164784


namespace NUMINAMATH_GPT_b_has_infinite_solutions_l1647_164785

noncomputable def b_value_satisfies_infinite_solutions : Prop :=
  ∃ b : ℚ, (∀ x : ℚ, 4 * (3 * x - b) = 3 * (4 * x + 7)) → b = -21 / 4

theorem b_has_infinite_solutions : b_value_satisfies_infinite_solutions :=
  sorry

end NUMINAMATH_GPT_b_has_infinite_solutions_l1647_164785


namespace NUMINAMATH_GPT_sum_of_possible_radii_l1647_164741

-- Define the geometric and algebraic conditions of the problem
noncomputable def circleTangentSum (r : ℝ) : Prop :=
  let center_C := (r, r)
  let center_other := (3, 3)
  let radius_other := 2
  (∃ r : ℝ, (r > 0) ∧ ((center_C.1 - center_other.1)^2 + (center_C.2 - center_other.2)^2 = (r + radius_other)^2))

-- Define the theorem statement
theorem sum_of_possible_radii : ∃ r1 r2 : ℝ, r1 ≠ r2 ∧ circleTangentSum r1 ∧ circleTangentSum r2 ∧ r1 + r2 = 16 :=
sorry

end NUMINAMATH_GPT_sum_of_possible_radii_l1647_164741


namespace NUMINAMATH_GPT_range_of_m_l1647_164793

-- Define the quadratic function
def quadratic_function (x m : ℝ) : ℝ := (x - m) ^ 2 - 1

-- State the main theorem
theorem range_of_m (m : ℝ) :
  (∀ x ≤ 3, quadratic_function x m ≥ quadratic_function (x + 1) m) ↔ m ≥ 3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1647_164793


namespace NUMINAMATH_GPT_circle_equation_line_equation_l1647_164769

noncomputable def circle_C (x y : ℝ) : Prop :=
  x^2 + y^2 - 8 * x + 6 * y = 0

noncomputable def point_O : ℝ × ℝ := (0, 0)
noncomputable def point_A : ℝ × ℝ := (1, 1)
noncomputable def point_B : ℝ × ℝ := (4, 2)

theorem circle_equation :
  circle_C point_O.1 point_O.2 ∧
  circle_C point_A.1 point_A.2 ∧
  circle_C point_B.1 point_B.2 :=
by sorry

noncomputable def line_l_case1 (x : ℝ) : Prop :=
  x = 3 / 2

noncomputable def line_l_case2 (x y : ℝ) : Prop :=
  8 * x + 6 * y - 39 = 0

noncomputable def center_C : ℝ × ℝ := (4, -3)
noncomputable def radius_C : ℝ := 5

noncomputable def point_through_l : ℝ × ℝ := (3 / 2, 9 / 2)

theorem line_equation : 
(∀ (M N : ℝ × ℝ), circle_C M.1 M.2 ∧ circle_C N.1 N.2 → ∃ C_slave : Prop, 
(C_slave → 
((line_l_case1 (point_through_l.1)) ∨ 
(line_l_case2 point_through_l.1 point_through_l.2)))) :=
by sorry

end NUMINAMATH_GPT_circle_equation_line_equation_l1647_164769


namespace NUMINAMATH_GPT_magnolia_trees_below_threshold_l1647_164746

-- Define the initial number of trees and the function describing the decrease
def initial_tree_count (N₀ : ℕ) (t : ℕ) : ℝ := N₀ * (0.8 ^ t)

-- Define the year when the number of trees is less than 25% of initial trees
theorem magnolia_trees_below_threshold (N₀ : ℕ) : (t : ℕ) -> initial_tree_count N₀ t < 0.25 * N₀ -> t > 14 := 
-- Provide the required statement but omit the actual proof with "sorry"
by sorry

end NUMINAMATH_GPT_magnolia_trees_below_threshold_l1647_164746
