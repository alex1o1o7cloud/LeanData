import Mathlib

namespace NUMINAMATH_GPT_solution_set_of_equation_l1557_155775

theorem solution_set_of_equation :
  {p : ℝ × ℝ | p.1 * p.2 + 1 = p.1 + p.2} = {p : ℝ × ℝ | p.1 = 1 ∨ p.2 = 1} :=
by 
  sorry

end NUMINAMATH_GPT_solution_set_of_equation_l1557_155775


namespace NUMINAMATH_GPT_moles_Cl2_combined_l1557_155743

-- Condition Definitions
def moles_C2H6 := 2
def moles_HCl_formed := 2
def balanced_reaction (C2H6 Cl2 C2H4Cl2 HCl : ℝ) : Prop :=
  C2H6 + Cl2 = C2H4Cl2 + 2 * HCl

-- Mathematical Equivalent Proof Problem Statement
theorem moles_Cl2_combined (C2H6 Cl2 HCl C2H4Cl2 : ℝ) (h1 : C2H6 = 2) 
(h2 : HCl = 2) (h3 : balanced_reaction C2H6 Cl2 C2H4Cl2 HCl) :
  Cl2 = 1 :=
by
  -- The proof is stated here.
  sorry

end NUMINAMATH_GPT_moles_Cl2_combined_l1557_155743


namespace NUMINAMATH_GPT_different_prime_factors_of_factorial_eq_10_l1557_155770

-- First, define n as 30
def n : ℕ := 30

-- Define a list of primes less than 30
def primesLessThanN : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- This is the theorem stating the number of distinct prime factors of 30!
theorem different_prime_factors_of_factorial_eq_10 : 
  (primesLessThanN.filter (Nat.Prime)).length = 10 := by 
  sorry

end NUMINAMATH_GPT_different_prime_factors_of_factorial_eq_10_l1557_155770


namespace NUMINAMATH_GPT_horner_method_correct_l1557_155776

-- Define the polynomial function using Horner's method
def f (x : ℤ) : ℤ := (((((x - 8) * x + 60) * x + 16) * x + 96) * x + 240) * x + 64

-- Define the value to be plugged into the polynomial
def x_val : ℤ := 2

-- Compute v_0, v_1, and v_2 according to the Horner's method
def v0 : ℤ := 1
def v1 : ℤ := v0 * x_val - 8
def v2 : ℤ := v1 * x_val + 60

-- Formal statement of the proof problem
theorem horner_method_correct :
  v2 = 48 := by
  -- Insert proof here
  sorry

end NUMINAMATH_GPT_horner_method_correct_l1557_155776


namespace NUMINAMATH_GPT_yogurt_cost_l1557_155769

-- Definitions from the conditions
def milk_cost : ℝ := 1.5
def fruit_cost : ℝ := 2
def milk_needed_per_batch : ℝ := 10
def fruit_needed_per_batch : ℝ := 3
def batches : ℕ := 3

-- Using the conditions, we state the theorem
theorem yogurt_cost :
  (milk_needed_per_batch * milk_cost + fruit_needed_per_batch * fruit_cost) * batches = 63 :=
by
  -- Skipping the proof
  sorry

end NUMINAMATH_GPT_yogurt_cost_l1557_155769


namespace NUMINAMATH_GPT_find_mystery_number_l1557_155780

theorem find_mystery_number (x : ℕ) (h : x + 45 = 92) : x = 47 :=
sorry

end NUMINAMATH_GPT_find_mystery_number_l1557_155780


namespace NUMINAMATH_GPT_count_integers_between_sqrt5_and_sqrt50_l1557_155774

theorem count_integers_between_sqrt5_and_sqrt50 
  (h1 : 2 < Real.sqrt 5 ∧ Real.sqrt 5 < 3)
  (h2 : 7 < Real.sqrt 50 ∧ Real.sqrt 50 < 8) : 
  ∃ n : ℕ, n = 5 := 
sorry

end NUMINAMATH_GPT_count_integers_between_sqrt5_and_sqrt50_l1557_155774


namespace NUMINAMATH_GPT_total_parcel_boxes_l1557_155762

theorem total_parcel_boxes (a b c d : ℕ) (row_boxes column_boxes total_boxes : ℕ)
  (h_left : a = 7) (h_right : b = 13)
  (h_front : c = 8) (h_back : d = 14)
  (h_row : row_boxes = a - 1 + 1 + b) -- boxes in a row: (a - 1) + 1 (parcel itself) + b
  (h_column : column_boxes = c - 1 + 1 + d) -- boxes in a column: (c -1) + 1(parcel itself) + d
  (h_total : total_boxes = row_boxes * column_boxes) :
  total_boxes = 399 := by
  sorry

end NUMINAMATH_GPT_total_parcel_boxes_l1557_155762


namespace NUMINAMATH_GPT_solution_exists_iff_divisor_form_l1557_155729

theorem solution_exists_iff_divisor_form (n : ℕ) (hn_pos : 0 < n) (hn_odd : n % 2 = 1) :
  (∃ x y : ℕ, 0 < x ∧ 0 < y ∧ 4 * x * y = n * (x + y)) ↔
    (∃ k : ℕ, n % (4 * k + 3) = 0) :=
by
  sorry

end NUMINAMATH_GPT_solution_exists_iff_divisor_form_l1557_155729


namespace NUMINAMATH_GPT_full_capacity_l1557_155730

def oil_cylinder_capacity (C : ℝ) :=
  (4 / 5) * C - (3 / 4) * C = 4

theorem full_capacity : oil_cylinder_capacity 80 :=
by
  simp [oil_cylinder_capacity]
  sorry

end NUMINAMATH_GPT_full_capacity_l1557_155730


namespace NUMINAMATH_GPT_cost_of_carton_l1557_155706

-- Definition of given conditions
def totalCost : ℝ := 4.88
def numberOfCartons : ℕ := 4
def costPerCarton : ℝ := 1.22

-- The proof statement
theorem cost_of_carton
  (h : totalCost = 4.88) 
  (n : numberOfCartons = 4) :
  totalCost / numberOfCartons = costPerCarton := 
sorry

end NUMINAMATH_GPT_cost_of_carton_l1557_155706


namespace NUMINAMATH_GPT_torn_pages_are_112_and_113_l1557_155748

theorem torn_pages_are_112_and_113 (n k : ℕ) (S S' : ℕ) 
  (h1 : S = n * (n + 1) / 2)
  (h2 : S' = S - (k - 1) - k)
  (h3 : S' = 15000) :
  (k = 113) ∧ (k - 1 = 112) :=
by
  sorry

end NUMINAMATH_GPT_torn_pages_are_112_and_113_l1557_155748


namespace NUMINAMATH_GPT_nina_ants_count_l1557_155749

theorem nina_ants_count 
  (spiders : ℕ) 
  (eyes_per_spider : ℕ) 
  (eyes_per_ant : ℕ) 
  (total_eyes : ℕ) 
  (total_spider_eyes : ℕ) 
  (total_ant_eyes : ℕ) 
  (ants : ℕ) 
  (h1 : spiders = 3) 
  (h2 : eyes_per_spider = 8) 
  (h3 : eyes_per_ant = 2) 
  (h4 : total_eyes = 124) 
  (h5 : total_spider_eyes = spiders * eyes_per_spider) 
  (h6 : total_ant_eyes = total_eyes - total_spider_eyes) 
  (h7 : ants = total_ant_eyes / eyes_per_ant) : 
  ants = 50 := by
  sorry

end NUMINAMATH_GPT_nina_ants_count_l1557_155749


namespace NUMINAMATH_GPT_intersection_of_M_and_N_l1557_155773

def set_M (x : ℝ) : Prop := 1 - 2 / x < 0
def set_N (x : ℝ) : Prop := -1 ≤ x
def set_Intersection (x : ℝ) : Prop := 0 < x ∧ x < 2

theorem intersection_of_M_and_N :
  ∀ x, (set_M x ∧ set_N x) ↔ set_Intersection x :=
by sorry

end NUMINAMATH_GPT_intersection_of_M_and_N_l1557_155773


namespace NUMINAMATH_GPT_cubic_root_sqrt_equation_l1557_155758

theorem cubic_root_sqrt_equation (x : ℝ) (h1 : 3 - x = y^3) (h2 : x - 2 = z^2) (h3 : y + z = 1) : 
  x = 3 ∨ x = 2 ∨ x = 11 :=
sorry

end NUMINAMATH_GPT_cubic_root_sqrt_equation_l1557_155758


namespace NUMINAMATH_GPT_total_students_l1557_155759

def numStudents (skiing scavenger : ℕ) : ℕ :=
  skiing + scavenger

theorem total_students (skiing scavenger : ℕ) (h1 : skiing = 2 * scavenger) (h2 : scavenger = 4000) :
  numStudents skiing scavenger = 12000 :=
by
  sorry

end NUMINAMATH_GPT_total_students_l1557_155759


namespace NUMINAMATH_GPT_inequality_solution_fractional_equation_solution_l1557_155713

-- Proof Problem 1
theorem inequality_solution (x : ℝ) : (1 - x) / 3 - x < 3 - (x + 2) / 4 → x > -2 :=
by
  sorry

-- Proof Problem 2
theorem fractional_equation_solution (x : ℝ) : (x - 2) / (2 * x - 1) + 1 = 3 / (2 * (1 - 2 * x)) → false :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_fractional_equation_solution_l1557_155713


namespace NUMINAMATH_GPT_value_x_when_y2_l1557_155757

theorem value_x_when_y2 (x : ℝ) (h1 : ∃ (x : ℝ), y = 1 / (5 * x + 2)) (h2 : y = 2) : x = -3 / 10 := by
  sorry

end NUMINAMATH_GPT_value_x_when_y2_l1557_155757


namespace NUMINAMATH_GPT_min_fraction_sum_l1557_155710

theorem min_fraction_sum (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 1) : 
  (∃ (z : ℝ), z = (1 / (x + 1)) + (4 / (y + 2)) ∧ z = 9 / 4) :=
by 
  sorry

end NUMINAMATH_GPT_min_fraction_sum_l1557_155710


namespace NUMINAMATH_GPT_max_sum_of_factors_l1557_155792

theorem max_sum_of_factors (p q : ℕ) (hpq : p * q = 100) : p + q ≤ 101 :=
sorry

end NUMINAMATH_GPT_max_sum_of_factors_l1557_155792


namespace NUMINAMATH_GPT_minimal_primes_ensuring_first_player_win_l1557_155752

-- Define primes less than or equal to 100
def primes_le_100 : List Nat := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

-- Define function to get the last digit of a number
def last_digit (n : Nat) : Nat := n % 10

-- Define function to get the first digit of a number
def first_digit (n : Nat) : Nat :=
  let rec first_digit_aux (m : Nat) :=
    if m < 10 then m else first_digit_aux (m / 10)
  first_digit_aux n

-- Define a condition that checks if a prime number follows the game rule
def follows_rule (a b : Nat) : Bool :=
  last_digit a = first_digit b

theorem minimal_primes_ensuring_first_player_win :
  ∃ (p1 p2 p3 : Nat),
  p1 ∈ primes_le_100 ∧
  p2 ∈ primes_le_100 ∧
  p3 ∈ primes_le_100 ∧
  follows_rule p1 p2 ∧
  follows_rule p2 p3 ∧
  p1 = 19 ∧ p2 = 97 ∧ p3 = 79 :=
sorry

end NUMINAMATH_GPT_minimal_primes_ensuring_first_player_win_l1557_155752


namespace NUMINAMATH_GPT_jack_grassy_time_is_6_l1557_155772

def jack_sandy_time := 19
def jill_total_time := 32
def jill_time_delay := 7
def jack_total_time : ℕ := jill_total_time - jill_time_delay
def jack_grassy_time : ℕ := jack_total_time - jack_sandy_time

theorem jack_grassy_time_is_6 : jack_grassy_time = 6 := by 
  have h1: jack_total_time = 25 := by sorry
  have h2: jack_grassy_time = 6 := by sorry
  exact h2

end NUMINAMATH_GPT_jack_grassy_time_is_6_l1557_155772


namespace NUMINAMATH_GPT_average_weight_of_all_boys_l1557_155712

theorem average_weight_of_all_boys (total_boys_16 : ℕ) (avg_weight_boys_16 : ℝ)
  (total_boys_8 : ℕ) (avg_weight_boys_8 : ℝ) 
  (h1 : total_boys_16 = 16) (h2 : avg_weight_boys_16 = 50.25)
  (h3 : total_boys_8 = 8) (h4 : avg_weight_boys_8 = 45.15) : 
  (total_boys_16 * avg_weight_boys_16 + total_boys_8 * avg_weight_boys_8) / (total_boys_16 + total_boys_8) = 48.55 :=
by
  sorry

end NUMINAMATH_GPT_average_weight_of_all_boys_l1557_155712


namespace NUMINAMATH_GPT_not_perfect_square_l1557_155786

-- Definitions and Conditions
def N (k : ℕ) : ℕ := (10^300 - 1) / 9 * 10^k

-- Proof Statement
theorem not_perfect_square (k : ℕ) : ¬∃ (m: ℕ), m * m = N k := 
sorry

end NUMINAMATH_GPT_not_perfect_square_l1557_155786


namespace NUMINAMATH_GPT_part_3_l1557_155799

noncomputable def f (x : ℝ) (m : ℝ) := Real.log x - m * x^2
noncomputable def g (x : ℝ) (m : ℝ) := (1/2) * m * x^2 + x
noncomputable def F (x : ℝ) (m : ℝ) := f x m + g x m

theorem part_3 (x₁ x₂ : ℝ) (m : ℝ) (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) (hm : m = -2)
  (hF : F x₁ m + F x₂ m + x₁ * x₂ = 0) : x₁ + x₂ ≥ (Real.sqrt 5 - 1) / 2 :=
sorry

end NUMINAMATH_GPT_part_3_l1557_155799


namespace NUMINAMATH_GPT_converse_inverse_contrapositive_count_l1557_155709

theorem converse_inverse_contrapositive_count
  (a b : ℝ) : (a = 0 → ab = 0) →
  (if (ab = 0 → a = 0) then 1 else 0) +
  (if (a ≠ 0 → ab ≠ 0) then 1 else 0) +
  (if (ab ≠ 0 → a ≠ 0) then 1 else 0) = 1 :=
sorry

end NUMINAMATH_GPT_converse_inverse_contrapositive_count_l1557_155709


namespace NUMINAMATH_GPT_area_H1H2H3_eq_four_l1557_155731

section TriangleArea

variables {P D E F H1 H2 H3 : Type*}

-- Definitions of midpoints, centroid, etc. can be implicit in Lean's formalism if necessary
-- We'll represent the area relation directly

-- Assume P is inside triangle DEF
def point_inside_triangle (P D E F : Type*) : Prop :=
sorry  -- Details are abstracted

-- Assume H1, H2, H3 are centroids of triangles PDE, PEF, PFD respectively
def is_centroid (H1 H2 H3 P D E F : Type*) : Prop :=
sorry  -- Details are abstracted

-- Given the area of triangle DEF
def area_DEF : ℝ := 12

-- Define the area function for the triangle formed by specific points
def area_triangle (A B C : Type*) : ℝ :=
sorry  -- Actual computation is abstracted

-- Mathematical statement to be proven
theorem area_H1H2H3_eq_four (P D E F H1 H2 H3 : Type*)
  (h_inside : point_inside_triangle P D E F)
  (h_centroid : is_centroid H1 H2 H3 P D E F)
  (h_area_DEF : area_triangle D E F = area_DEF) :
  area_triangle H1 H2 H3 = 4 :=
sorry

end TriangleArea

end NUMINAMATH_GPT_area_H1H2H3_eq_four_l1557_155731


namespace NUMINAMATH_GPT_how_many_ducks_did_john_buy_l1557_155747

def cost_price_per_duck : ℕ := 10
def weight_per_duck : ℕ := 4
def selling_price_per_pound : ℕ := 5
def profit : ℕ := 300

theorem how_many_ducks_did_john_buy (D : ℕ) (h : 10 * D - 10 * D + 10 * D = profit) : D = 30 :=
by 
  sorry

end NUMINAMATH_GPT_how_many_ducks_did_john_buy_l1557_155747


namespace NUMINAMATH_GPT_chocolate_bar_weight_l1557_155796

theorem chocolate_bar_weight :
  let square_weight := 6
  let triangles_count := 16
  let squares_count := 32
  let triangle_weight := square_weight / 2
  let total_square_weight := squares_count * square_weight
  let total_triangles_weight := triangles_count * triangle_weight
  total_square_weight + total_triangles_weight = 240 := 
by
  sorry

end NUMINAMATH_GPT_chocolate_bar_weight_l1557_155796


namespace NUMINAMATH_GPT_find_b_for_real_root_l1557_155724

noncomputable def polynomial_has_real_root (b : ℝ) : Prop :=
  ∃ x : ℝ, x^4 + b * x^3 - 2 * x^2 + b * x + 2 = 0

theorem find_b_for_real_root :
  ∀ b : ℝ, polynomial_has_real_root b → b ≤ 0 := by
  sorry

end NUMINAMATH_GPT_find_b_for_real_root_l1557_155724


namespace NUMINAMATH_GPT_boat_downstream_travel_time_l1557_155744

theorem boat_downstream_travel_time (D : ℝ) (V_b : ℝ) (T_u : ℝ) (V_c : ℝ) (T_d : ℝ) : 
  D = 300 ∧ V_b = 105 ∧ T_u = 5 ∧ (300 = (105 - V_c) * 5) ∧ (300 = (105 + V_c) * T_d) → T_d = 2 :=
by
  sorry

end NUMINAMATH_GPT_boat_downstream_travel_time_l1557_155744


namespace NUMINAMATH_GPT_evaluate_fraction_l1557_155777

theorem evaluate_fraction : (1 - 1/4) / (1 - 1/3) = 9/8 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_fraction_l1557_155777


namespace NUMINAMATH_GPT_xiao_wang_original_plan_l1557_155725

theorem xiao_wang_original_plan (p d1 extra_pages : ℕ) (original_days : ℝ) (x : ℝ) 
  (h1 : p = 200)
  (h2 : d1 = 5)
  (h3 : extra_pages = 5)
  (h4 : original_days = p / x)
  (h5 : original_days - 1 = d1 + (p - (d1 * x)) / (x + extra_pages)) :
  x = 20 := 
  sorry

end NUMINAMATH_GPT_xiao_wang_original_plan_l1557_155725


namespace NUMINAMATH_GPT_adult_ticket_cost_l1557_155787

-- Definitions from the conditions
def total_amount : ℕ := 35
def child_ticket_cost : ℕ := 3
def num_children : ℕ := 9

-- The amount spent on children’s tickets
def total_child_ticket_cost : ℕ := num_children * child_ticket_cost

-- The remaining amount after purchasing children’s tickets
def remaining_amount : ℕ := total_amount - total_child_ticket_cost

-- The adult ticket cost should be equal to the remaining amount
theorem adult_ticket_cost : remaining_amount = 8 :=
by sorry

end NUMINAMATH_GPT_adult_ticket_cost_l1557_155787


namespace NUMINAMATH_GPT_volume_of_region_l1557_155761

theorem volume_of_region :
    ∀ (x y z : ℝ), 
    |x - y + z| + |x - y - z| ≤ 12 ∧ x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 
    → true := by
    sorry

end NUMINAMATH_GPT_volume_of_region_l1557_155761


namespace NUMINAMATH_GPT_happy_dictionary_problem_l1557_155728

def smallest_positive_integer : ℕ := 1
def largest_negative_integer : ℤ := -1
def smallest_abs_rational : ℚ := 0

theorem happy_dictionary_problem : 
  smallest_positive_integer - largest_negative_integer + smallest_abs_rational = 2 := 
by
  sorry

end NUMINAMATH_GPT_happy_dictionary_problem_l1557_155728


namespace NUMINAMATH_GPT_avg_visitors_per_day_l1557_155782

theorem avg_visitors_per_day :
  let visitors := [583, 246, 735, 492, 639]
  (visitors.sum / visitors.length) = 539 := by
  sorry

end NUMINAMATH_GPT_avg_visitors_per_day_l1557_155782


namespace NUMINAMATH_GPT_find_N_l1557_155702

theorem find_N : 
  (1993 + 1994 + 1995 + 1996 + 1997) / N = (3 + 4 + 5 + 6 + 7) / 5 → 
  N = 1995 :=
by
  sorry

end NUMINAMATH_GPT_find_N_l1557_155702


namespace NUMINAMATH_GPT_find_b_l1557_155793

theorem find_b (a b : ℝ) (h1 : a * (a - 4) = 21) (h2 : b * (b - 4) = 21) (h3 : a + b = 4) (h4 : a ≠ b) :
  b = -3 :=
sorry

end NUMINAMATH_GPT_find_b_l1557_155793


namespace NUMINAMATH_GPT_evaluate_101_times_101_l1557_155745

theorem evaluate_101_times_101 : 101 * 101 = 10201 :=
by sorry

end NUMINAMATH_GPT_evaluate_101_times_101_l1557_155745


namespace NUMINAMATH_GPT_fill_tank_in_6_hours_l1557_155767

theorem fill_tank_in_6_hours (A B : ℝ) (hA : A = 1 / 10) (hB : B = 1 / 15) : (1 / (A + B)) = 6 :=
by 
  sorry

end NUMINAMATH_GPT_fill_tank_in_6_hours_l1557_155767


namespace NUMINAMATH_GPT_problem_statement_l1557_155783

namespace ProofProblems

open Set

def U : Set ℕ := {2, 3, 4, 5, 6}
def M : Set ℕ := {3, 4, 5}
def N : Set ℕ := {2, 4, 5, 6}

theorem problem_statement : M ∪ N = U := sorry

end ProofProblems

end NUMINAMATH_GPT_problem_statement_l1557_155783


namespace NUMINAMATH_GPT_door_cranking_time_l1557_155764

-- Define the given conditions
def run_time_with_backpack : ℝ := 7 * 60 + 23  -- 443 seconds
def run_time_without_backpack : ℝ := 5 * 60 + 58  -- 358 seconds
def total_time : ℝ := 874  -- 874 seconds

-- Define the Lean statement of the proof
theorem door_cranking_time :
  (run_time_with_backpack + run_time_without_backpack) + (total_time - (run_time_with_backpack + run_time_without_backpack)) = total_time ∧
  (total_time - (run_time_with_backpack + run_time_without_backpack)) = 73 :=
by
  sorry

end NUMINAMATH_GPT_door_cranking_time_l1557_155764


namespace NUMINAMATH_GPT_product_of_reciprocals_is_9_over_4_l1557_155726

noncomputable def product_of_reciprocals (a b : ℝ) : ℝ :=
  (1 / a) * (1 / b)

theorem product_of_reciprocals_is_9_over_4 (a b : ℝ) (h : a + b = 3 * a * b) (ha : a ≠ 0) (hb : b ≠ 0) : 
  product_of_reciprocals a b = 9 / 4 :=
sorry

end NUMINAMATH_GPT_product_of_reciprocals_is_9_over_4_l1557_155726


namespace NUMINAMATH_GPT_y2_over_x2_plus_x2_over_y2_eq_9_over_4_l1557_155765

theorem y2_over_x2_plus_x2_over_y2_eq_9_over_4 (x y : ℝ) 
  (h : (1 / x) - (1 / (2 * y)) = (1 / (2 * x + y))) : 
  (y^2 / x^2) + (x^2 / y^2) = 9 / 4 := 
by 
  sorry

end NUMINAMATH_GPT_y2_over_x2_plus_x2_over_y2_eq_9_over_4_l1557_155765


namespace NUMINAMATH_GPT_fewer_vip_tickets_sold_l1557_155798

-- Definitions based on the conditions
variables (V G : ℕ)
def tickets_sold := V + G = 320
def total_cost := 40 * V + 10 * G = 7500

-- The main statement to prove
theorem fewer_vip_tickets_sold :
  tickets_sold V G → total_cost V G → G - V = 34 := 
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_fewer_vip_tickets_sold_l1557_155798


namespace NUMINAMATH_GPT_sector_area_l1557_155797

theorem sector_area (θ : ℝ) (r : ℝ) (hθ : θ = π / 3) (hr : r = 4) : 
  (1/2) * (r * θ) * r = 8 * π / 3 :=
by
  -- Implicitly use the given values of θ and r by substituting them in the expression.
  sorry

end NUMINAMATH_GPT_sector_area_l1557_155797


namespace NUMINAMATH_GPT_determine_b_l1557_155754

theorem determine_b (a b c : ℕ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (eq_radicals: Real.sqrt (4 * a + 4 * b / c) = 2 * a * Real.sqrt (b / c)) : 
  b = c + 1 :=
sorry

end NUMINAMATH_GPT_determine_b_l1557_155754


namespace NUMINAMATH_GPT_range_of_a_l1557_155701

variable {α : Type}

def A (x : ℝ) : Prop := 1 ≤ x ∧ x < 5
def B (x a : ℝ) : Prop := -a < x ∧ x ≤ a + 3

theorem range_of_a (a : ℝ) :
  (∀ x, B x a → A x) → a ≤ -1 := by
  sorry

end NUMINAMATH_GPT_range_of_a_l1557_155701


namespace NUMINAMATH_GPT_proof_quotient_l1557_155766

/-- Let x be in the form (a + b * sqrt c) / d -/
def x_form (a b c d : ℤ) (x : ℝ) : Prop := x = (a + b * Real.sqrt c) / d

/-- Main theorem -/
theorem proof_quotient (a b c d : ℤ) (x : ℝ) (h_eq : 4 * x / 5 + 2 = 5 / x) (h_form : x_form a b c d x) : (a * c * d) / b = -20 := by
  sorry

end NUMINAMATH_GPT_proof_quotient_l1557_155766


namespace NUMINAMATH_GPT_relationship_between_x_b_a_l1557_155746

variable {x b a : ℝ}

theorem relationship_between_x_b_a 
  (hx : x < 0) (hb : b < 0) (ha : a < 0)
  (hxb : x < b) (hba : b < a) : x^2 > b * x ∧ b * x > b^2 :=
by sorry

end NUMINAMATH_GPT_relationship_between_x_b_a_l1557_155746


namespace NUMINAMATH_GPT_fraction_spent_on_sandwich_l1557_155716
    
theorem fraction_spent_on_sandwich 
  (x : ℚ)
  (h1 : 90 * x + 90 * (1/6) + 90 * (1/2) + 12 = 90) : 
  x = 1/5 :=
by
  sorry

end NUMINAMATH_GPT_fraction_spent_on_sandwich_l1557_155716


namespace NUMINAMATH_GPT_value_of_m_solve_system_relationship_x_y_l1557_155723

-- Part 1: Prove the value of m is 1
theorem value_of_m (x : ℝ) (m : ℝ) (h1 : 2 - x = x + 4) (h2 : m * (1 - x) = x + 3) : m = 1 := sorry

-- Part 2: Solve the system of equations given m = 1
theorem solve_system (x y : ℝ) (h1 : 3 * x + 2 * 1 = - y) (h2 : 2 * x + 2 * y = 1 - 1) : x = -1 ∧ y = 1 := sorry

-- Part 3: Relationship between x and y regardless of m
theorem relationship_x_y (x y m : ℝ) (h1 : 3 * x + y = -2 * m) (h2 : 2 * x + 2 * y = m - 1) : 7 * x + 5 * y = -2 := sorry

end NUMINAMATH_GPT_value_of_m_solve_system_relationship_x_y_l1557_155723


namespace NUMINAMATH_GPT_final_score_l1557_155708

def dart1 : ℕ := 50
def dart2 : ℕ := 0
def dart3 : ℕ := dart1 / 2

theorem final_score : dart1 + dart2 + dart3 = 75 := by
  sorry

end NUMINAMATH_GPT_final_score_l1557_155708


namespace NUMINAMATH_GPT_neg_mul_reverses_inequality_l1557_155714

theorem neg_mul_reverses_inequality (a b : ℝ) (h : a < b) : -3 * a > -3 * b :=
  sorry

end NUMINAMATH_GPT_neg_mul_reverses_inequality_l1557_155714


namespace NUMINAMATH_GPT_Samuel_fraction_spent_l1557_155717

variable (totalAmount receivedRatio remainingAmount : ℕ)
variable (h1 : totalAmount = 240)
variable (h2 : receivedRatio = 3 / 4)
variable (h3 : remainingAmount = 132)

theorem Samuel_fraction_spent (spend : ℚ) : 
  (spend = (1 / 5)) :=
by
  sorry

end NUMINAMATH_GPT_Samuel_fraction_spent_l1557_155717


namespace NUMINAMATH_GPT_range_of_a_l1557_155789

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (2 - a) * x^2 + 2 * (2 - a) * x + 4 ≥ 0) → (-2 ≤ a ∧ a ≤ 2) := 
sorry

end NUMINAMATH_GPT_range_of_a_l1557_155789


namespace NUMINAMATH_GPT_trig_identity_l1557_155778

theorem trig_identity : 
  (Real.sin (12 * Real.pi / 180)) * (Real.sin (48 * Real.pi / 180)) * 
  (Real.sin (72 * Real.pi / 180)) * (Real.sin (84 * Real.pi / 180)) = 1 / 32 :=
by sorry

end NUMINAMATH_GPT_trig_identity_l1557_155778


namespace NUMINAMATH_GPT_number_of_days_worked_l1557_155727

-- Definitions based on the given conditions and question
def total_hours_worked : ℕ := 15
def hours_worked_each_day : ℕ := 3

-- The statement we need to prove:
theorem number_of_days_worked : 
  (total_hours_worked / hours_worked_each_day) = 5 :=
by
  sorry

end NUMINAMATH_GPT_number_of_days_worked_l1557_155727


namespace NUMINAMATH_GPT_trendy_haircut_cost_l1557_155707

theorem trendy_haircut_cost (T : ℝ) (H1 : 5 * 5 * 7 + 3 * 6 * 7 + 2 * T * 7 = 413) : T = 8 :=
by linarith

end NUMINAMATH_GPT_trendy_haircut_cost_l1557_155707


namespace NUMINAMATH_GPT_least_three_digit_with_factors_l1557_155711

theorem least_three_digit_with_factors (n : ℕ) :
  (n ≥ 100 ∧ n < 1000 ∧ 2 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n ∧ 3 ∣ n) → n = 210 := by
  sorry

end NUMINAMATH_GPT_least_three_digit_with_factors_l1557_155711


namespace NUMINAMATH_GPT_apple_tree_bears_fruit_in_7_years_l1557_155736

def age_planted : ℕ := 4
def age_eats : ℕ := 11
def time_to_bear_fruit : ℕ := age_eats - age_planted

theorem apple_tree_bears_fruit_in_7_years :
  time_to_bear_fruit = 7 :=
by
  sorry

end NUMINAMATH_GPT_apple_tree_bears_fruit_in_7_years_l1557_155736


namespace NUMINAMATH_GPT_alice_operations_terminate_l1557_155738

theorem alice_operations_terminate (a : List ℕ) (h_pos : ∀ x ∈ a, x > 0) : 
(∀ x y z, (x, y) = (y + 1, x) ∨ (x, y) = (x - 1, x) → ∃ n, (x :: y :: z).sum ≤ n) :=
by sorry

end NUMINAMATH_GPT_alice_operations_terminate_l1557_155738


namespace NUMINAMATH_GPT_original_class_size_l1557_155760

/-- Let A be the average age of the original adult class, which is 40 years. -/
def A : ℕ := 40

/-- Let B be the average age of the 8 new students, which is 32 years. -/
def B : ℕ := 32

/-- Let C be the decreased average age of the class after the new students join, which is 36 years. -/
def C : ℕ := 36

/-- The original number of students in the adult class is N. -/
def N : ℕ := 8

/-- The equation representing the total age of the class after the new students join. -/
theorem original_class_size :
  (A * N) + (B * 8) = C * (N + 8) ↔ N = 8 := by
  sorry

end NUMINAMATH_GPT_original_class_size_l1557_155760


namespace NUMINAMATH_GPT_general_form_of_quadratic_equation_l1557_155779

noncomputable def quadratic_equation_general_form (x : ℝ) : Prop :=
  (x + 3) * (x - 1) = 2 * x - 4

theorem general_form_of_quadratic_equation (x : ℝ) :
  quadratic_equation_general_form x → x^2 + 1 = 0 :=
sorry

end NUMINAMATH_GPT_general_form_of_quadratic_equation_l1557_155779


namespace NUMINAMATH_GPT_kim_boxes_on_thursday_l1557_155722

theorem kim_boxes_on_thursday (Tues Wed Thurs : ℕ) 
(h1 : Tues = 4800)
(h2 : Tues = 2 * Wed)
(h3 : Wed = 2 * Thurs) : Thurs = 1200 :=
by
  sorry

end NUMINAMATH_GPT_kim_boxes_on_thursday_l1557_155722


namespace NUMINAMATH_GPT_a_1000_value_l1557_155705

theorem a_1000_value :
  ∃ (a : ℕ → ℤ), 
    (a 1 = 2010) ∧
    (a 2 = 2011) ∧
    (∀ n : ℕ, n ≥ 1 → a n + a (n + 1) + a (n + 2) = 2 * n + 3) ∧
    (a 1000 = 2676) :=
by {
  -- sorry is used to skip the proof
  sorry 
}

end NUMINAMATH_GPT_a_1000_value_l1557_155705


namespace NUMINAMATH_GPT_base_5_representation_l1557_155715

theorem base_5_representation (n : ℕ) (h : n = 84) : 
  ∃ (a b c : ℕ), 
  a < 5 ∧ b < 5 ∧ c < 5 ∧ 
  n = a * 5^2 + b * 5^1 + c * 5^0 ∧ 
  a = 3 ∧ b = 1 ∧ c = 4 :=
by 
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_base_5_representation_l1557_155715


namespace NUMINAMATH_GPT_relationship_abc_l1557_155756

variables {a b c : ℝ}

-- Given conditions
def condition1 (a b c : ℝ) : Prop := 0 < a ∧ 0 < b ∧ 0 < c ∧ (11/6 : ℝ) * c < a + b ∧ a + b < 2 * c
def condition2 (a b c : ℝ) : Prop := (3/2 : ℝ) * a < b + c ∧ b + c < (5/3 : ℝ) * a
def condition3 (a b c : ℝ) : Prop := (5/2 : ℝ) * b < a + c ∧ a + c < (11/4 : ℝ) * b

-- Proof statement
theorem relationship_abc (a b c : ℝ) (h1 : condition1 a b c) (h2 : condition2 a b c) (h3 : condition3 a b c) :
  b < c ∧ c < a :=
by
  sorry

end NUMINAMATH_GPT_relationship_abc_l1557_155756


namespace NUMINAMATH_GPT_problem_statement_l1557_155763

open Complex

theorem problem_statement (x y : ℝ) (i : ℂ) (h_i : i = Complex.I) (h : x + (y - 2) * i = 2 / (1 + i)) : x + y = 2 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1557_155763


namespace NUMINAMATH_GPT_percentage_increase_l1557_155732

theorem percentage_increase (R W : ℕ) (hR : R = 36) (hW : W = 20) : 
  ((R - W) / W : ℚ) * 100 = 80 := 
by 
  sorry

end NUMINAMATH_GPT_percentage_increase_l1557_155732


namespace NUMINAMATH_GPT_order_of_values_l1557_155742

noncomputable def a : ℝ := 21.2
noncomputable def b : ℝ := Real.sqrt 450 - 0.8
noncomputable def c : ℝ := 2 * Real.logb 5 2

theorem order_of_values : c < b ∧ b < a := by 
  sorry

end NUMINAMATH_GPT_order_of_values_l1557_155742


namespace NUMINAMATH_GPT_solve_ratios_l1557_155768

theorem solve_ratios (q m n : ℕ) (h1 : 7 / 9 = n / 108) (h2 : 7 / 9 = (m + n) / 126) (h3 : 7 / 9 = (q - m) / 162) : q = 140 :=
by
  sorry

end NUMINAMATH_GPT_solve_ratios_l1557_155768


namespace NUMINAMATH_GPT_max_volume_48cm_square_l1557_155735

def volume_of_box (x : ℝ) := x * (48 - 2 * x)^2

theorem max_volume_48cm_square : 
  ∃ x : ℝ, 0 < x ∧ x < 24 ∧ (∀ y : ℝ, 0 < y ∧ y < 24 → volume_of_box x ≥ volume_of_box y) ∧ x = 8 :=
sorry

end NUMINAMATH_GPT_max_volume_48cm_square_l1557_155735


namespace NUMINAMATH_GPT_percentage_increase_in_savings_l1557_155719

theorem percentage_increase_in_savings (I : ℝ) (hI : 0 < I) :
  let E := 0.75 * I
  let S := I - E
  let I_new := 1.20 * I
  let E_new := 0.825 * I
  let S_new := I_new - E_new
  ((S_new - S) / S) * 100 = 50 :=
by
  let E := 0.75 * I
  let S := I - E
  let I_new := 1.20 * I
  let E_new := 0.825 * I
  let S_new := I_new - E_new
  sorry

end NUMINAMATH_GPT_percentage_increase_in_savings_l1557_155719


namespace NUMINAMATH_GPT_z_when_y_six_l1557_155788

theorem z_when_y_six
    (k : ℝ)
    (h1 : ∀ y (z : ℝ), y^2 * Real.sqrt z = k)
    (h2 : ∃ (y : ℝ) (z : ℝ), y = 3 ∧ z = 4 ∧ y^2 * Real.sqrt z = k) :
  ∃ z : ℝ, y = 6 ∧ z = 1 / 4 := 
sorry

end NUMINAMATH_GPT_z_when_y_six_l1557_155788


namespace NUMINAMATH_GPT_money_problem_l1557_155781

variable {c d : ℝ}

theorem money_problem (h1 : 3 * c - 2 * d < 30) (h2 : 4 * c + d = 60) : 
  c < 150 / 11 ∧ d > 60 / 11 := 
by 
  sorry

end NUMINAMATH_GPT_money_problem_l1557_155781


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1557_155739

theorem solution_set_of_inequality (x : ℝ) (h : x ≠ 2) :
  (1 / (x - 2) > -2) ↔ (x < 3 / 2 ∨ x > 2) :=
by sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1557_155739


namespace NUMINAMATH_GPT_fraction_addition_correct_l1557_155718

theorem fraction_addition_correct : (3 / 5 : ℚ) + (2 / 5) = 1 := 
by
  sorry

end NUMINAMATH_GPT_fraction_addition_correct_l1557_155718


namespace NUMINAMATH_GPT_find_y_when_x_4_l1557_155753

-- Definitions and conditions
variables (x y : ℝ)
def inversely_proportional (x y : ℝ) (K : ℝ) : Prop := x * y = K

-- Main theorem
theorem find_y_when_x_4 
  (K : ℝ) (h1 : inversely_proportional 20 10 K) (h2 : 20 + 10 = 30) (h3 : 20 - 10 = 10) 
  (hx : 4 * y = K) : y = 50 := 
sorry

end NUMINAMATH_GPT_find_y_when_x_4_l1557_155753


namespace NUMINAMATH_GPT_find_number_of_cows_l1557_155720

-- Definitions for the problem
def number_of_legs (cows chickens : ℕ) := 4 * cows + 2 * chickens
def twice_the_heads_plus_12 (cows chickens : ℕ) := 2 * (cows + chickens) + 12

-- Main statement to prove
theorem find_number_of_cows (h : ℕ) : ∃ c : ℕ, number_of_legs c h = twice_the_heads_plus_12 c h ∧ c = 6 := 
by
  -- Sorry is used as a placeholder for the proof
  sorry

end NUMINAMATH_GPT_find_number_of_cows_l1557_155720


namespace NUMINAMATH_GPT_income_of_person_l1557_155733

theorem income_of_person (x: ℝ) (h : 9 * x - 8 * x = 2000) : 9 * x = 18000 :=
by
  sorry

end NUMINAMATH_GPT_income_of_person_l1557_155733


namespace NUMINAMATH_GPT_max_sum_of_squares_l1557_155794

theorem max_sum_of_squares (a b c d : ℝ) 
  (h1 : a + b = 17) 
  (h2 : ab + c + d = 86) 
  (h3 : ad + bc = 180) 
  (h4 : cd = 110) : 
  a^2 + b^2 + c^2 + d^2 ≤ 258 :=
sorry

end NUMINAMATH_GPT_max_sum_of_squares_l1557_155794


namespace NUMINAMATH_GPT_simplify_and_evaluate_l1557_155750

theorem simplify_and_evaluate (x : ℝ) (hx : x = Real.sqrt 2 + 1) :
  (x + 1) / x / (x - (1 + x^2) / (2 * x)) = Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l1557_155750


namespace NUMINAMATH_GPT_ratio_of_josh_to_brad_l1557_155785

theorem ratio_of_josh_to_brad (J D B : ℝ) (h1 : J + D + B = 68) (h2 : J = (3 / 4) * D) (h3 : D = 32) :
  (J / B) = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_josh_to_brad_l1557_155785


namespace NUMINAMATH_GPT_combined_weight_of_boxes_l1557_155784

def weight_box1 : ℝ := 2
def weight_box2 : ℝ := 11
def weight_box3 : ℝ := 5

theorem combined_weight_of_boxes : weight_box1 + weight_box2 + weight_box3 = 18 := by
  sorry

end NUMINAMATH_GPT_combined_weight_of_boxes_l1557_155784


namespace NUMINAMATH_GPT_compute_expression_l1557_155791

theorem compute_expression : (3 + 5) ^ 2 + (3 ^ 2 + 5 ^ 2) = 98 := by
  sorry

end NUMINAMATH_GPT_compute_expression_l1557_155791


namespace NUMINAMATH_GPT_function_value_at_minus_one_l1557_155795

theorem function_value_at_minus_one :
  ( -(1:ℝ)^4 + -(1:ℝ)^3 + (1:ℝ) ) / ( -(1:ℝ)^2 + (1:ℝ) ) = 1 / 2 :=
by sorry

end NUMINAMATH_GPT_function_value_at_minus_one_l1557_155795


namespace NUMINAMATH_GPT_total_balls_in_bag_l1557_155734

theorem total_balls_in_bag (R G B T : ℕ) 
  (hR : R = 907) 
  (hRatio : 15 * T = 15 * R + 13 * R + 17 * R)
  : T = 2721 :=
sorry

end NUMINAMATH_GPT_total_balls_in_bag_l1557_155734


namespace NUMINAMATH_GPT_g_neither_even_nor_odd_l1557_155703

noncomputable def g (x : ℝ) : ℝ := ⌈x⌉ - 1 / 2

theorem g_neither_even_nor_odd :
  (¬ ∀ x, g x = g (-x)) ∧ (¬ ∀ x, g (-x) = -g x) :=
by
  sorry

end NUMINAMATH_GPT_g_neither_even_nor_odd_l1557_155703


namespace NUMINAMATH_GPT_painting_time_l1557_155741

-- Definitions based on the conditions
def num_people1 := 8
def num_houses1 := 3
def time1 := 12
def num_people2 := 9
def num_houses2 := 4
def k := (num_people1 * time1) / num_houses1

-- The statement we want to prove
theorem painting_time : (num_people2 * t = k * num_houses2) → (t = 128 / 9) :=
by sorry

end NUMINAMATH_GPT_painting_time_l1557_155741


namespace NUMINAMATH_GPT_vector_addition_l1557_155740

-- Defining the vectors a and b
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (1, 3)

-- Stating the problem: proving the sum of vectors a and b
theorem vector_addition : a + b = (3, 4) := 
by 
  -- Proof is not required as per the instructions
  sorry

end NUMINAMATH_GPT_vector_addition_l1557_155740


namespace NUMINAMATH_GPT_christopher_age_l1557_155737

variables (C G : ℕ)

theorem christopher_age :
  (C = 2 * G) ∧ (C - 9 = 5 * (G - 9)) → C = 24 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_christopher_age_l1557_155737


namespace NUMINAMATH_GPT_least_x_divisible_by_3_l1557_155700

theorem least_x_divisible_by_3 : ∃ x : ℕ, (∀ y : ℕ, (2 + 3 + 5 + 7 + y) % 3 = 0 → y = 1) :=
by
  sorry

end NUMINAMATH_GPT_least_x_divisible_by_3_l1557_155700


namespace NUMINAMATH_GPT_angle_A_range_l1557_155721

-- Definitions from the conditions
variable (A B C : ℝ)
variable (a b c : ℝ)
axiom triangle_scalene : a ≠ b ∧ b ≠ c ∧ c ≠ a
axiom longest_side_a : a > b ∧ a > c
axiom inequality_a : a^2 < b^2 + c^2

-- Target proof statement
theorem angle_A_range (triangle_scalene : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (longest_side_a : a > b ∧ a > c)
  (inequality_a : a^2 < b^2 + c^2) : 60 < A ∧ A < 90 := 
sorry

end NUMINAMATH_GPT_angle_A_range_l1557_155721


namespace NUMINAMATH_GPT_greatest_integer_gcd_is_4_l1557_155755

theorem greatest_integer_gcd_is_4 : 
  ∀ (n : ℕ), n < 150 ∧ (Nat.gcd n 24 = 4) → n ≤ 148 := 
by
  sorry

end NUMINAMATH_GPT_greatest_integer_gcd_is_4_l1557_155755


namespace NUMINAMATH_GPT_calc_expression_l1557_155790

theorem calc_expression : 
  (Real.sqrt 16 - 4 * (Real.sqrt 2) / 2 + abs (- (Real.sqrt 3 * Real.sqrt 6)) + (-1) ^ 2023) = 
  (3 + Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_GPT_calc_expression_l1557_155790


namespace NUMINAMATH_GPT_find_number_l1557_155704

theorem find_number {x : ℝ} 
  (h : 973 * x - 739 * x = 110305) : 
  x = 471.4 := 
by 
  sorry

end NUMINAMATH_GPT_find_number_l1557_155704


namespace NUMINAMATH_GPT_largest_possible_d_plus_r_l1557_155771

theorem largest_possible_d_plus_r :
  ∃ d r : ℕ, 0 < d ∧ 468 % d = r ∧ 636 % d = r ∧ 867 % d = r ∧ d + r = 27 := by
  sorry

end NUMINAMATH_GPT_largest_possible_d_plus_r_l1557_155771


namespace NUMINAMATH_GPT_boys_bought_balloons_l1557_155751

def initial_balloons : ℕ := 3 * 12  -- Clown initially has 3 dozen balloons, i.e., 36 balloons
def girls_balloons : ℕ := 12        -- 12 girls buy a balloon each
def balloons_remaining : ℕ := 21     -- Clown is left with 21 balloons

def boys_balloons : ℕ :=
  initial_balloons - balloons_remaining - girls_balloons

theorem boys_bought_balloons :
  boys_balloons = 3 :=
by
  sorry

end NUMINAMATH_GPT_boys_bought_balloons_l1557_155751
