import Mathlib

namespace NUMINAMATH_GPT_jake_later_than_austin_by_20_seconds_l1250_125091

theorem jake_later_than_austin_by_20_seconds :
  (9 * 30) / 3 - 60 = 20 :=
by
  sorry

end NUMINAMATH_GPT_jake_later_than_austin_by_20_seconds_l1250_125091


namespace NUMINAMATH_GPT_total_yards_run_l1250_125032

theorem total_yards_run (Malik_yards_per_game : ℕ) (Josiah_yards_per_game : ℕ) (Darnell_yards_per_game : ℕ) (games : ℕ) 
  (hM : Malik_yards_per_game = 18) (hJ : Josiah_yards_per_game = 22) (hD : Darnell_yards_per_game = 11) (hG : games = 4) : 
  Malik_yards_per_game * games + Josiah_yards_per_game * games + Darnell_yards_per_game * games = 204 := by
  sorry

end NUMINAMATH_GPT_total_yards_run_l1250_125032


namespace NUMINAMATH_GPT_find_n_values_l1250_125001

theorem find_n_values (n : ℤ) (hn : ∃ x y : ℤ, x ≠ y ∧ x^2 - 6*x - 4*n^2 - 32*n = 0 ∧ y^2 - 6*y - 4*n^2 - 32*n = 0):
  n = 10 ∨ n = 0 ∨ n = -8 ∨ n = -18 := 
sorry

end NUMINAMATH_GPT_find_n_values_l1250_125001


namespace NUMINAMATH_GPT_four_digit_number_divisible_by_18_l1250_125054

theorem four_digit_number_divisible_by_18 : ∃ n : ℕ, (n % 2 = 0) ∧ (10 + n) % 9 = 0 ∧ n = 8 :=
by
  sorry

end NUMINAMATH_GPT_four_digit_number_divisible_by_18_l1250_125054


namespace NUMINAMATH_GPT_variance_of_planted_trees_l1250_125046

def number_of_groups := 10

def planted_trees : List ℕ := [5, 5, 5, 6, 6, 6, 6, 7, 7, 7]

noncomputable def mean (xs : List ℕ) : ℚ :=
  (xs.sum : ℚ) / (xs.length : ℚ)

noncomputable def variance (xs : List ℕ) : ℚ :=
  let m := mean xs
  (xs.map (λ x => (x - m) ^ 2)).sum / (xs.length : ℚ)

theorem variance_of_planted_trees :
  variance planted_trees = 0.6 := sorry

end NUMINAMATH_GPT_variance_of_planted_trees_l1250_125046


namespace NUMINAMATH_GPT_first_digit_of_base16_representation_l1250_125060

-- Firstly we define the base conversion from base 4 to base 10 and from base 10 to base 16.
-- For simplicity, we assume that the required functions exist and skip their implementations.

-- Assume base 4 to base 10 conversion function
def base4_to_base10 (n : String) : Nat :=
  sorry

-- Assume base 10 to base 16 conversion function that gives the first digit
def first_digit_base16 (n : Nat) : Nat :=
  sorry

-- Given the base 4 number as string
def y_base4 : String := "20313320132220312031"

-- Define the final statement
theorem first_digit_of_base16_representation :
  first_digit_base16 (base4_to_base10 y_base4) = 5 :=
by
  sorry

end NUMINAMATH_GPT_first_digit_of_base16_representation_l1250_125060


namespace NUMINAMATH_GPT_ratio_of_students_l1250_125005

-- Define the conditions
def total_students : Nat := 800
def students_spaghetti : Nat := 320
def students_fettuccine : Nat := 160

-- The proof problem
theorem ratio_of_students (h1 : students_spaghetti = 320) (h2 : students_fettuccine = 160) :
  students_spaghetti / students_fettuccine = 2 := by
  sorry

end NUMINAMATH_GPT_ratio_of_students_l1250_125005


namespace NUMINAMATH_GPT_man_speed_is_correct_l1250_125022

noncomputable def speed_of_man (train_speed_kmh : ℝ) (train_length_m : ℝ) (time_to_pass_s : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let relative_speed_ms := train_length_m / time_to_pass_s
  let man_speed_ms := relative_speed_ms - train_speed_ms
  man_speed_ms * 3600 / 1000

theorem man_speed_is_correct : 
  speed_of_man 60 110 5.999520038396929 = 6.0024 := 
by
  sorry

end NUMINAMATH_GPT_man_speed_is_correct_l1250_125022


namespace NUMINAMATH_GPT_coconut_tree_difference_l1250_125020

-- Define the known quantities
def mango_trees : ℕ := 60
def total_trees : ℕ := 85
def half_mango_trees : ℕ := 30 -- half of 60
def coconut_trees : ℕ := 25 -- 85 - 60

-- Define the proof statement
theorem coconut_tree_difference : (half_mango_trees - coconut_trees) = 5 := by
  -- The proof steps are given
  sorry

end NUMINAMATH_GPT_coconut_tree_difference_l1250_125020


namespace NUMINAMATH_GPT_area_union_example_l1250_125024

noncomputable def area_union_square_circle (s r : ℝ) : ℝ :=
  let A_square := s ^ 2
  let A_circle := Real.pi * r ^ 2
  let A_overlap := (1 / 4) * A_circle
  A_square + A_circle - A_overlap

theorem area_union_example : (area_union_square_circle 10 10) = 100 + 75 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_area_union_example_l1250_125024


namespace NUMINAMATH_GPT_number_of_possible_routes_l1250_125092

def f (x y : ℕ) : ℕ :=
  if y = 2 then sorry else sorry -- Here you need the exact definition of f(x, y)

theorem number_of_possible_routes (n : ℕ) (h : n > 0) : 
  f n 2 = (1 / 2 : ℚ) * (n^2 + 3 * n + 2) := 
by 
  sorry

end NUMINAMATH_GPT_number_of_possible_routes_l1250_125092


namespace NUMINAMATH_GPT_distinct_positive_and_conditions_l1250_125066

theorem distinct_positive_and_conditions (a b : ℕ) (h_distinct: a ≠ b) (h_pos1: 0 < a) (h_pos2: 0 < b) (h_eq: a^3 - b^3 = a^2 - b^2) : 
  ∃ (c : ℕ), c = 9 * a * b ∧ (c = 1 ∨ c = 2 ∨ c = 3) :=
by
  sorry

end NUMINAMATH_GPT_distinct_positive_and_conditions_l1250_125066


namespace NUMINAMATH_GPT_no_three_digit_number_l1250_125012

theorem no_three_digit_number :
  ¬ ∃ (a b c : ℕ), (1 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) ∧ (0 ≤ c ∧ c ≤ 9) ∧ (100 * a + 10 * b + c = 3 * (100 * b + 10 * c + a)) :=
by
  sorry

end NUMINAMATH_GPT_no_three_digit_number_l1250_125012


namespace NUMINAMATH_GPT_dwarf_heights_l1250_125089

-- Define the heights of the dwarfs.
variables (F J M : ℕ)

-- Given conditions
def condition1 : Prop := J + F = M
def condition2 : Prop := M + F = J + 34
def condition3 : Prop := M + J = F + 72

-- Proof statement
theorem dwarf_heights
  (h1 : condition1 F J M)
  (h2 : condition2 F J M)
  (h3 : condition3 F J M) :
  F = 17 ∧ J = 36 ∧ M = 53 :=
by
  sorry

end NUMINAMATH_GPT_dwarf_heights_l1250_125089


namespace NUMINAMATH_GPT_probability_XOX_OXO_l1250_125086

open Nat

/-- Setting up the math problem to be proved -/
def X : Finset ℕ := {1, 2, 3, 4}
def O : Finset ℕ := {5, 6, 7}

def totalArrangements : ℕ := choose 7 4

def favorableArrangements : ℕ := 1

theorem probability_XOX_OXO : (favorableArrangements : ℚ) / (totalArrangements : ℚ) = 1 / 35 := by
  have h_total : totalArrangements = 35 := by sorry
  have h_favorable : favorableArrangements = 1 := by sorry
  rw [h_total, h_favorable]
  norm_num

end NUMINAMATH_GPT_probability_XOX_OXO_l1250_125086


namespace NUMINAMATH_GPT_number_of_solutions_l1250_125068

theorem number_of_solutions :
  (∃ (a b c : ℕ), 4 * a = 6 * c ∧ 168 * a = 6 * a * b * c) → 
  ∃ (s : Finset ℕ), s.card = 6 :=
by sorry

end NUMINAMATH_GPT_number_of_solutions_l1250_125068


namespace NUMINAMATH_GPT_largest_integer_n_l1250_125035

-- Define the condition for existence of positive integers x, y, z that satisfy the given equation
def condition (n : ℕ) : Prop :=
  ∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ n^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 5*x + 5*y + 5*z - 10

-- State that the largest such integer n is 4
theorem largest_integer_n : ∀ (n : ℕ), condition n → n ≤ 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_largest_integer_n_l1250_125035


namespace NUMINAMATH_GPT_sphere_radius_eq_three_l1250_125033

theorem sphere_radius_eq_three (r : ℝ) (h : 4 / 3 * π * r ^ 3 = 4 * π * r ^ 2) : r = 3 :=
by
  sorry

end NUMINAMATH_GPT_sphere_radius_eq_three_l1250_125033


namespace NUMINAMATH_GPT_incorrect_gcd_statement_l1250_125000

theorem incorrect_gcd_statement :
  ¬(gcd 85 357 = 34) ∧ (gcd 16 12 = 4) ∧ (gcd 78 36 = 6) ∧ (gcd 105 315 = 105) :=
by
  sorry

end NUMINAMATH_GPT_incorrect_gcd_statement_l1250_125000


namespace NUMINAMATH_GPT_third_year_award_count_l1250_125071

-- Define the variables and conditions
variables (x x1 x2 x3 x4 x5 : ℕ)

-- The conditions and definition for the problem
def conditions : Prop :=
  (x1 = x) ∧
  (x5 = 3 * x) ∧
  (x1 < x2) ∧
  (x2 < x3) ∧
  (x3 < x4) ∧
  (x4 < x5) ∧
  (x1 + x2 + x3 + x4 + x5 = 27)

-- The theorem statement
theorem third_year_award_count (h : conditions x x1 x2 x3 x4 x5) : x3 = 5 :=
sorry

end NUMINAMATH_GPT_third_year_award_count_l1250_125071


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l1250_125014

def setA : Set ℝ := { x | x - 2 ≥ 0 }
def setB : Set ℝ := { x | 0 < Real.log x / Real.log 2 ∧ Real.log x / Real.log 2 < 2 }

theorem intersection_of_A_and_B :
  setA ∩ setB = { x | 2 ≤ x ∧ x < 4 } :=
sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l1250_125014


namespace NUMINAMATH_GPT_fraction_of_girls_is_one_half_l1250_125027

def fraction_of_girls (total_students_jasper : ℕ) (ratio_jasper : ℕ × ℕ) (total_students_brookstone : ℕ) (ratio_brookstone : ℕ × ℕ) : ℚ :=
  let (boys_ratio_jasper, girls_ratio_jasper) := ratio_jasper
  let (boys_ratio_brookstone, girls_ratio_brookstone) := ratio_brookstone
  let girls_jasper := (total_students_jasper * girls_ratio_jasper) / (boys_ratio_jasper + girls_ratio_jasper)
  let girls_brookstone := (total_students_brookstone * girls_ratio_brookstone) / (boys_ratio_brookstone + girls_ratio_brookstone)
  let total_girls := girls_jasper + girls_brookstone
  let total_students := total_students_jasper + total_students_brookstone
  total_girls / total_students

theorem fraction_of_girls_is_one_half :
  fraction_of_girls 360 (7, 5) 240 (3, 5) = 1 / 2 :=
  sorry

end NUMINAMATH_GPT_fraction_of_girls_is_one_half_l1250_125027


namespace NUMINAMATH_GPT_solution_set_of_tan_eq_two_l1250_125053

open Real

theorem solution_set_of_tan_eq_two :
  {x | ∃ k : ℤ, x = k * π + (-1 : ℤ) ^ k * arctan 2} = {x | tan x = 2} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_tan_eq_two_l1250_125053


namespace NUMINAMATH_GPT_eggs_left_over_l1250_125057

def david_eggs : ℕ := 44
def elizabeth_eggs : ℕ := 52
def fatima_eggs : ℕ := 23
def carton_size : ℕ := 12

theorem eggs_left_over : 
  (david_eggs + elizabeth_eggs + fatima_eggs) % carton_size = 11 :=
by sorry

end NUMINAMATH_GPT_eggs_left_over_l1250_125057


namespace NUMINAMATH_GPT_salary_spending_l1250_125076

theorem salary_spending (S_A S_B : ℝ) (P_A P_B : ℝ) 
  (h1 : S_A = 4500) 
  (h2 : S_A + S_B = 6000)
  (h3 : P_B = 0.85) 
  (h4 : S_A * (1 - P_A) = S_B * (1 - P_B)) : 
  P_A = 0.95 :=
by
  -- Start proofs here
  sorry

end NUMINAMATH_GPT_salary_spending_l1250_125076


namespace NUMINAMATH_GPT_intersection_M_N_l1250_125031

-- Definitions for sets M and N
def set_M : Set ℝ := {x | abs x < 1}
def set_N : Set ℝ := {x | x^2 <= x}

-- The theorem stating the intersection of M and N
theorem intersection_M_N : {x : ℝ | x ∈ set_M ∧ x ∈ set_N} = {x : ℝ | 0 <= x ∧ x < 1} :=
by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l1250_125031


namespace NUMINAMATH_GPT_find_b_l1250_125030

-- Define the conditions as constants
def x := 36 -- angle a in degrees
def y := 44 -- given
def z := 52 -- given
def w := 48 -- angle b we need to find

-- Define the problem as a theorem
theorem find_b : x + w + y + z = 180 :=
by
  -- Substitute the given values and show the sum
  have h : 36 + 48 + 44 + 52 = 180 := by norm_num
  exact h

end NUMINAMATH_GPT_find_b_l1250_125030


namespace NUMINAMATH_GPT_solve_system_of_equations_solve_system_of_inequalities_l1250_125039

-- For the system of equations
theorem solve_system_of_equations (x y : ℝ) (h1 : 3 * x + 4 * y = 2) (h2 : 2 * x - y = 5) : 
    x = 2 ∧ y = -1 :=
sorry

-- For the system of inequalities
theorem solve_system_of_inequalities (x : ℝ) 
    (h1 : x - 3 * (x - 1) < 7) 
    (h2 : x - 2 ≤ (2 * x - 3) / 3) :
    -2 < x ∧ x ≤ 3 :=
sorry

end NUMINAMATH_GPT_solve_system_of_equations_solve_system_of_inequalities_l1250_125039


namespace NUMINAMATH_GPT_solve_for_x_l1250_125082

def f (x : ℝ) : ℝ := 2 * x - 3

theorem solve_for_x : ∃ (x : ℝ), 2 * (f x) - 11 = f (x - 2) :=
by
  use 5
  have h1 : f 5 = 2 * 5 - 3 := rfl
  have h2 : f (5 - 2) = 2 * (5 - 2) - 3 := rfl
  simp [f] at *
  exact sorry

end NUMINAMATH_GPT_solve_for_x_l1250_125082


namespace NUMINAMATH_GPT_find_discount_l1250_125036

noncomputable def children_ticket_cost : ℝ := 4.25
noncomputable def adult_ticket_cost : ℝ := children_ticket_cost + 3.25
noncomputable def total_cost_without_discount : ℝ := 2 * adult_ticket_cost + 4 * children_ticket_cost
noncomputable def total_spent : ℝ := 30
noncomputable def discount_received : ℝ := total_cost_without_discount - total_spent

theorem find_discount :
  discount_received = 2 := by
  sorry

end NUMINAMATH_GPT_find_discount_l1250_125036


namespace NUMINAMATH_GPT_range_of_a_l1250_125044

theorem range_of_a (hP : ¬ ∃ x : ℝ, x^2 + 2 * a * x + a ≤ 0) : 0 < a ∧ a < 1 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1250_125044


namespace NUMINAMATH_GPT_range_of_a_opposite_sides_l1250_125050

theorem range_of_a_opposite_sides (a : ℝ) :
  (3 * (-2) - 2 * 1 - a) * (3 * 1 - 2 * 1 - a) < 0 ↔ -8 < a ∧ a < 1 := by
  sorry

end NUMINAMATH_GPT_range_of_a_opposite_sides_l1250_125050


namespace NUMINAMATH_GPT_possible_slopes_l1250_125021

theorem possible_slopes (k : ℝ) (H_pos : k > 0) :
  (∃ x1 x2 : ℤ, (x1 + x2 : ℝ) = k ∧ (x1 * x2 : ℝ) = -2020) ↔ 
  k = 81 ∨ k = 192 ∨ k = 399 ∨ k = 501 ∨ k = 1008 ∨ k = 2019 := 
by
  sorry

end NUMINAMATH_GPT_possible_slopes_l1250_125021


namespace NUMINAMATH_GPT_divisors_log_sum_eq_l1250_125081

open BigOperators

/-- Given the sum of the base-10 logarithms of the divisors of \( 10^{2n} = 4752 \), prove that \( n = 12 \). -/
theorem divisors_log_sum_eq (n : ℕ) (h : ∑ a in Finset.range (2*n + 1), ∑ b in Finset.range (2*n + 1), 
  (a * Real.log (2) / Real.log (10) + b * Real.log (5) / Real.log (10)) = 4752) : n = 12 :=
by {
  sorry
}

end NUMINAMATH_GPT_divisors_log_sum_eq_l1250_125081


namespace NUMINAMATH_GPT_C_completion_time_l1250_125067

noncomputable def racer_time (v_C : ℝ) : ℝ := 100 / v_C

theorem C_completion_time
  (v_A v_B v_C : ℝ)
  (h1 : 100 / v_A = 10)
  (h2 : 85 / v_B = 10)
  (h3 : 90 / v_C = 100 / v_B) :
  racer_time v_C = 13.07 :=
by
  sorry

end NUMINAMATH_GPT_C_completion_time_l1250_125067


namespace NUMINAMATH_GPT_time_to_cross_approx_l1250_125070

-- Define train length, tunnel length, speed in km/hr, conversion factors, and the final equation
def length_of_train : ℕ := 415
def length_of_tunnel : ℕ := 285
def speed_in_kmph : ℕ := 63
def km_to_m : ℕ := 1000
def hr_to_sec : ℕ := 3600

-- Convert speed to m/s
def speed_in_mps : ℚ := (speed_in_kmph * km_to_m) / hr_to_sec

-- Calculate total distance
def total_distance : ℕ := length_of_train + length_of_tunnel

-- Calculate the time to cross the tunnel in seconds
def time_to_cross : ℚ := total_distance / speed_in_mps

theorem time_to_cross_approx : abs (time_to_cross - 40) < 0.1 :=
sorry

end NUMINAMATH_GPT_time_to_cross_approx_l1250_125070


namespace NUMINAMATH_GPT_meat_needed_l1250_125011

theorem meat_needed (meat_per_hamburger : ℚ) (h_meat : meat_per_hamburger = (3 : ℚ) / 8) : 
  (24 * meat_per_hamburger) = 9 :=
by
  sorry

end NUMINAMATH_GPT_meat_needed_l1250_125011


namespace NUMINAMATH_GPT_roller_coaster_ticket_cost_l1250_125045

def ferrisWheelCost : ℕ := 6
def logRideCost : ℕ := 7
def initialTickets : ℕ := 2
def ticketsToBuy : ℕ := 16

def totalTicketsNeeded : ℕ := initialTickets + ticketsToBuy
def ridesCost : ℕ := ferrisWheelCost + logRideCost
def rollerCoasterCost : ℕ := totalTicketsNeeded - ridesCost

theorem roller_coaster_ticket_cost :
  rollerCoasterCost = 5 :=
by
  sorry

end NUMINAMATH_GPT_roller_coaster_ticket_cost_l1250_125045


namespace NUMINAMATH_GPT_painting_frame_ratio_l1250_125094

theorem painting_frame_ratio (x l : ℝ) (h1 : x > 0) (h2 : l > 0) 
  (h3 : (2 / 3) * x * x = (x + 2 * l) * ((3 / 2) * x + 2 * l) - x * (3 / 2) * x) :
  (x + 2 * l) / ((3 / 2) * x + 2 * l) = 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_painting_frame_ratio_l1250_125094


namespace NUMINAMATH_GPT_transformer_coils_flawless_l1250_125025

theorem transformer_coils_flawless (x y : ℕ) (hx : x + y = 8200)
  (hdef : (2 * x / 100) + (3 * y / 100) = 216) :
  ((x = 3000 ∧ y = 5200) ∧ ((x * 98 / 100) = 2940) ∧ ((y * 97 / 100) = 5044)) :=
by
  sorry

end NUMINAMATH_GPT_transformer_coils_flawless_l1250_125025


namespace NUMINAMATH_GPT_incorrect_sum_Sn_l1250_125056

-- Define the geometric sequence sum formula
def Sn (a r : ℕ) (n : ℕ) : ℕ := a * (1 - r^n) / (1 - r)

-- Define the given values
def S1 : ℕ := 8
def S2 : ℕ := 20
def S3 : ℕ := 36
def S4 : ℕ := 65

-- The main proof statement
theorem incorrect_sum_Sn : 
  ∃ (a r : ℕ), 
  a = 8 ∧ 
  Sn a r 1 = S1 ∧ 
  Sn a r 2 = S2 ∧ 
  Sn a r 3 ≠ S3 ∧ 
  Sn a r 4 = S4 :=
by sorry

end NUMINAMATH_GPT_incorrect_sum_Sn_l1250_125056


namespace NUMINAMATH_GPT_marilyn_bananas_l1250_125098

-- Defining the conditions
def boxes : ℕ := 8
def bananas_per_box : ℕ := 5

-- The statement that Marilyn has 40 bananas
theorem marilyn_bananas : boxes * bananas_per_box = 40 :=
by
  sorry

end NUMINAMATH_GPT_marilyn_bananas_l1250_125098


namespace NUMINAMATH_GPT_math_problem_l1250_125075

noncomputable def exponential_sequence (a : ℕ → ℝ) : Prop :=
∀ n, a n = 2 * 3^(n - 1)

noncomputable def geometric_sum (a : ℕ → ℝ) (n : ℕ) : ℝ :=
(2 * 3^n - 2) / 2

theorem math_problem 
  (a : ℕ → ℝ) (b : ℕ → ℕ) (c : ℕ → ℝ) (S T P : ℕ → ℝ)
  (h1 : exponential_sequence a)
  (h2 : a 1 * a 3 = 36)
  (h3 : a 3 + a 4 = 9 * (a 1 + a 2))
  (h4 : ∀ n, S n + 1 = 3^(b n))
  (h5 : ∀ n, T n = (2 * n - 1) * 3^n / 2 + 1 / 2)
  (h6 : ∀ n, c n = a n / ((a n + 1) * (a (n + 1) + 1)))
  (h7 : ∀ n, P (2 * n) = 1 / 6 - 1 / (4 * 3^(2 * n) + 2)) :
  (∀ n, a n = 2 * 3^(n - 1)) ∧
  ∀ n, b n = n ∧
  ∀ n, a n * b n = 2 * n * 3^(n - 1) ∧
  ∃ n, T n = (2 * n - 1) * 3^n / 2 + 1 / 2 ∧
  P (2 * n) = 1 / 6 - 1 / (4 * 3^(2 * n) + 2) :=
by sorry

end NUMINAMATH_GPT_math_problem_l1250_125075


namespace NUMINAMATH_GPT_seating_arrangement_l1250_125017

def num_ways_to_seat (A B C D E F : Type) (chairs : List (Option Type)) : Nat := sorry

theorem seating_arrangement {A B C D E F : Type} :
  ∀ (chairs : List (Option Type)),
    (A ≠ B ∧ A ≠ C ∧ F ≠ B) → num_ways_to_seat A B C D E F chairs = 28 :=
by
  sorry

end NUMINAMATH_GPT_seating_arrangement_l1250_125017


namespace NUMINAMATH_GPT_initial_price_of_article_l1250_125043

theorem initial_price_of_article (P : ℝ) (h : 0.4025 * P = 620) : P = 620 / 0.4025 :=
by
  sorry

end NUMINAMATH_GPT_initial_price_of_article_l1250_125043


namespace NUMINAMATH_GPT_sin_eq_sin_sinx_l1250_125019

noncomputable def S (x : ℝ) := Real.sin x - x

theorem sin_eq_sin_sinx (x : ℝ) (h : 0 ≤ x ∧ x ≤ Real.arcsin 742) :
  ∃! x, Real.sin x = Real.sin (Real.sin x) :=
by
  sorry

end NUMINAMATH_GPT_sin_eq_sin_sinx_l1250_125019


namespace NUMINAMATH_GPT_expression_value_l1250_125004

theorem expression_value (b : ℝ) (hb : b = 1 / 3) :
    (3 * b⁻¹ - b⁻¹ / 3) / b^2 = 72 :=
sorry

end NUMINAMATH_GPT_expression_value_l1250_125004


namespace NUMINAMATH_GPT_sum_of_fifth_powers_l1250_125042

theorem sum_of_fifth_powers (a b c d : ℝ) (h1 : a + b = c + d) (h2 : a^3 + b^3 = c^3 + d^3) : a^5 + b^5 = c^5 + d^5 := sorry

end NUMINAMATH_GPT_sum_of_fifth_powers_l1250_125042


namespace NUMINAMATH_GPT_find_angle_BCD_l1250_125028

-- Defining the given conditions in the problem
def angleA : ℝ := 100
def angleD : ℝ := 120
def angleE : ℝ := 80
def angleABC : ℝ := 140
def pentagonInteriorAngleSum : ℝ := 540

-- Statement: Prove that the measure of ∠ BCD is 100 degrees given the conditions
theorem find_angle_BCD (h1 : angleA = 100) (h2 : angleD = 120) (h3 : angleE = 80) 
                       (h4 : angleABC = 140) (h5 : pentagonInteriorAngleSum = 540) :
    (angleBCD : ℝ) = 100 :=
sorry

end NUMINAMATH_GPT_find_angle_BCD_l1250_125028


namespace NUMINAMATH_GPT_solve_a_b_powers_l1250_125002

theorem solve_a_b_powers :
  ∃ a b : ℂ, (a + b = 1) ∧ 
             (a^2 + b^2 = 3) ∧ 
             (a^3 + b^3 = 4) ∧ 
             (a^4 + b^4 = 7) ∧ 
             (a^5 + b^5 = 11) ∧ 
             (a^10 + b^10 = 93) :=
sorry

end NUMINAMATH_GPT_solve_a_b_powers_l1250_125002


namespace NUMINAMATH_GPT_area_fraction_above_line_l1250_125072

-- Define the points of the rectangle
def A := (2,0)
def B := (7,0)
def C := (7,4)
def D := (2,4)

-- Define the points used for the line
def P := (2,1)
def Q := (7,3)

-- The area of the rectangle
def rect_area := (7 - 2) * 4

-- The fraction of the area of the rectangle above the line
theorem area_fraction_above_line : 
  ∀ A B C D P Q, 
    A = (2,0) → B = (7,0) → C = (7,4) → D = (2,4) →
    P = (2,1) → Q = (7,3) →
    (rect_area = 20) → 1 - ((1/2) * 5 * 2 / 20) = 3 / 4 :=
by
  intros A B C D P Q
  intros hA hB hC hD hP hQ h_area
  sorry

end NUMINAMATH_GPT_area_fraction_above_line_l1250_125072


namespace NUMINAMATH_GPT_initial_condition_proof_move_to_1_proof_move_to_2_proof_recurrence_relation_proof_p_99_proof_p_100_proof_l1250_125016

variable (p : ℕ → ℚ)

-- Given conditions
axiom initial_condition : p 0 = 1
axiom move_to_1 : p 1 = 1 / 2
axiom move_to_2 : p 2 = 3 / 4
axiom recurrence_relation : ∀ n : ℕ, 2 ≤ n → n ≤ 99 → p n - p (n - 1) = - 1 / 2 * (p (n - 1) - p (n - 2))
axiom p_99_cond : p 99 = 2 / 3 - 1 / (3 * 2^99)
axiom p_100_cond : p 100 = 1 / 3 + 1 / (3 * 2^99)

-- Proof that initial conditions are met
theorem initial_condition_proof : p 0 = 1 :=
sorry

theorem move_to_1_proof : p 1 = 1 / 2 :=
sorry

theorem move_to_2_proof : p 2 = 3 / 4 :=
sorry

-- Proof of the recurrence relation
theorem recurrence_relation_proof : ∀ n : ℕ, 2 ≤ n → n ≤ 99 → p n - p (n - 1) = - 1 / 2 * (p (n - 1) - p (n - 2)) :=
sorry

-- Proof of p_99
theorem p_99_proof : p 99 = 2 / 3 - 1 / (3 * 2^99) :=
sorry

-- Proof of p_100
theorem p_100_proof : p 100 = 1 / 3 + 1 / (3 * 2^99) :=
sorry

end NUMINAMATH_GPT_initial_condition_proof_move_to_1_proof_move_to_2_proof_recurrence_relation_proof_p_99_proof_p_100_proof_l1250_125016


namespace NUMINAMATH_GPT_original_number_is_neg2_l1250_125037

theorem original_number_is_neg2 (x : ℚ) (h : 2 - 1/x = 5/2) : x = -2 :=
sorry

end NUMINAMATH_GPT_original_number_is_neg2_l1250_125037


namespace NUMINAMATH_GPT_tan_sum_l1250_125052

theorem tan_sum (x y : ℝ)
  (h1 : Real.sin x + Real.sin y = 72 / 65)
  (h2 : Real.cos x + Real.cos y = 96 / 65) : 
  Real.tan x + Real.tan y = 868 / 112 := 
by sorry

end NUMINAMATH_GPT_tan_sum_l1250_125052


namespace NUMINAMATH_GPT_fraction_representation_of_3_36_l1250_125097

theorem fraction_representation_of_3_36 : (336 : ℚ) / 100 = 84 / 25 := 
by sorry

end NUMINAMATH_GPT_fraction_representation_of_3_36_l1250_125097


namespace NUMINAMATH_GPT_number_of_true_propositions_l1250_125063

-- Let's state the propositions
def original_proposition (P Q : Prop) := P → Q
def converse_proposition (P Q : Prop) := Q → P
def inverse_proposition (P Q : Prop) := ¬P → ¬Q
def contrapositive_proposition (P Q : Prop) := ¬Q → ¬P

-- Main statement we need to prove
theorem number_of_true_propositions (P Q : Prop) (hpq : original_proposition P Q) 
  (hc: contrapositive_proposition P Q) (hev: converse_proposition P Q)  (hbv: inverse_proposition P Q) : 
  (¬(P ↔ Q) ∨ (¬¬P ↔ ¬¬Q) ∨ (¬Q → ¬P) ∨ (P → Q)) := sorry

end NUMINAMATH_GPT_number_of_true_propositions_l1250_125063


namespace NUMINAMATH_GPT_trucks_and_goods_l1250_125026

variable (x : ℕ) -- Number of trucks
variable (goods : ℕ) -- Total tons of goods

-- Conditions
def condition1 : Prop := goods = 3 * x + 5
def condition2 : Prop := goods = 4 * (x - 5)

theorem trucks_and_goods (h1 : condition1 x goods) (h2 : condition2 x goods) : x = 25 ∧ goods = 80 :=
by
  sorry

end NUMINAMATH_GPT_trucks_and_goods_l1250_125026


namespace NUMINAMATH_GPT_decreasing_function_l1250_125096

def f (a x : ℝ) : ℝ := a * x^3 - x

theorem decreasing_function (a : ℝ) 
  (h : ∀ x y : ℝ, x < y → f a y ≤ f a x) : a ≤ 0 :=
by
  sorry

end NUMINAMATH_GPT_decreasing_function_l1250_125096


namespace NUMINAMATH_GPT_smallest_z_l1250_125074

theorem smallest_z 
  (x y z : ℕ) 
  (h_pos_x : x > 0) 
  (h_pos_y : y > 0) 
  (h1 : x + y = z) 
  (h2 : x * y < z^2) 
  (ineq : (27^z) * (5^x) > (3^24) * (2^y)) :
  z = 10 :=
by
  sorry

end NUMINAMATH_GPT_smallest_z_l1250_125074


namespace NUMINAMATH_GPT_find_y_value_l1250_125013

theorem find_y_value (k : ℝ) (h1 : ∀ (x : ℝ), y = k * x) 
(h2 : y = 4 ∧ x = 2) : 
(∀ (x : ℝ), x = -2 → y = -4) := 
by 
  sorry

end NUMINAMATH_GPT_find_y_value_l1250_125013


namespace NUMINAMATH_GPT_percentage_increase_in_price_l1250_125088

theorem percentage_increase_in_price (initial_price : ℝ) (total_cost : ℝ) (num_family_members : ℕ) 
  (pounds_per_person : ℝ) (new_price : ℝ) (percentage_increase : ℝ) :
  initial_price = 1.6 → 
  total_cost = 16 → 
  num_family_members = 4 → 
  pounds_per_person = 2 → 
  (total_cost / (num_family_members * pounds_per_person)) = new_price → 
  percentage_increase = ((new_price - initial_price) / initial_price) * 100 → 
  percentage_increase = 25 :=
by
  intros h_initial h_total h_members h_pounds h_new_price h_percentage
  sorry

end NUMINAMATH_GPT_percentage_increase_in_price_l1250_125088


namespace NUMINAMATH_GPT_gcd_306_522_l1250_125062

theorem gcd_306_522 : Nat.gcd 306 522 = 18 := 
  by sorry

end NUMINAMATH_GPT_gcd_306_522_l1250_125062


namespace NUMINAMATH_GPT_abs_neg_2023_l1250_125073

theorem abs_neg_2023 : |(-2023 : ℤ)| = 2023 :=
by
  sorry

end NUMINAMATH_GPT_abs_neg_2023_l1250_125073


namespace NUMINAMATH_GPT_john_total_distance_l1250_125006

def speed : ℕ := 45
def time1 : ℕ := 2
def time2 : ℕ := 3

theorem john_total_distance:
  speed * (time1 + time2) = 225 := by
  sorry

end NUMINAMATH_GPT_john_total_distance_l1250_125006


namespace NUMINAMATH_GPT_pumps_280_gallons_in_30_minutes_l1250_125048

def hydraflow_rate_per_hour := 560 -- gallons per hour
def time_fraction_in_hour := 1 / 2

theorem pumps_280_gallons_in_30_minutes : hydraflow_rate_per_hour * time_fraction_in_hour = 280 := by
  sorry

end NUMINAMATH_GPT_pumps_280_gallons_in_30_minutes_l1250_125048


namespace NUMINAMATH_GPT_field_trip_fraction_l1250_125051

theorem field_trip_fraction (b g : ℕ) (hb : g = b)
  (girls_trip_fraction : ℚ := 4/5)
  (boys_trip_fraction : ℚ := 3/4) :
  girls_trip_fraction * g / (girls_trip_fraction * g + boys_trip_fraction * b) = 16 / 31 :=
by {
  sorry
}

end NUMINAMATH_GPT_field_trip_fraction_l1250_125051


namespace NUMINAMATH_GPT_min_value_inequality_equality_condition_l1250_125047

theorem min_value_inequality (a b : ℝ) (ha : 1 < a) (hb : 1 < b) :
  (b^2 / (a - 1) + a^2 / (b - 1)) ≥ 8 :=
sorry

theorem equality_condition (a b : ℝ) (ha : 1 < a) (hb : 1 < b) :
  (b^2 / (a - 1) + a^2 / (b - 1) = 8) ↔ ((a = 2) ∧ (b = 2)) :=
sorry

end NUMINAMATH_GPT_min_value_inequality_equality_condition_l1250_125047


namespace NUMINAMATH_GPT_systematic_sampling_interval_l1250_125038

theorem systematic_sampling_interval 
  (N : ℕ) (n : ℕ) (hN : N = 630) (hn : n = 45) :
  N / n = 14 :=
by {
  sorry
}

end NUMINAMATH_GPT_systematic_sampling_interval_l1250_125038


namespace NUMINAMATH_GPT_internet_bill_is_100_l1250_125079

theorem internet_bill_is_100 (initial_amount rent paycheck electricity_bill phone_bill final_amount internet_bill : ℝ)
  (h1 : initial_amount = 800)
  (h2 : rent = 450)
  (h3 : paycheck = 1500)
  (h4 : electricity_bill = 117)
  (h5 : phone_bill = 70)
  (h6 : final_amount = 1563)
  (h7 : initial_amount - rent + paycheck - electricity_bill - internet_bill - phone_bill = final_amount) :
  internet_bill = 100 :=
by
  sorry

end NUMINAMATH_GPT_internet_bill_is_100_l1250_125079


namespace NUMINAMATH_GPT_sqrt_factorial_mul_squared_l1250_125061

theorem sqrt_factorial_mul_squared :
  (Nat.sqrt (Nat.factorial 5 * Nat.factorial 4)) ^ 2 = 2880 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_factorial_mul_squared_l1250_125061


namespace NUMINAMATH_GPT_equation_of_line_parallel_to_x_axis_l1250_125010

theorem equation_of_line_parallel_to_x_axis (x: ℝ) :
  ∃ (y: ℝ), (y-2=0) ∧ ∀ (P: ℝ × ℝ), (P = (1, 2)) → P.2 = 2 := 
by
  sorry

end NUMINAMATH_GPT_equation_of_line_parallel_to_x_axis_l1250_125010


namespace NUMINAMATH_GPT_roof_length_width_diff_l1250_125058

variable (w l : ℝ)
variable (h1 : l = 4 * w)
variable (h2 : l * w = 676)

theorem roof_length_width_diff :
  l - w = 39 :=
by
  sorry

end NUMINAMATH_GPT_roof_length_width_diff_l1250_125058


namespace NUMINAMATH_GPT_swimming_pool_time_l1250_125095

theorem swimming_pool_time
  (A B C : ℝ)
  (h1 : A + B = 1 / 3)
  (h2 : A + C = 1 / 6)
  (h3 : B + C = 1 / 4.5) :
  1 / (A + B + C) = 2.25 :=
by
  sorry

end NUMINAMATH_GPT_swimming_pool_time_l1250_125095


namespace NUMINAMATH_GPT_sum_of_ages_twins_l1250_125040

-- Define that Evan has two older twin sisters and their ages are such that the product of all three ages is 162
def twin_sisters_ages (a : ℕ) (b : ℕ) (c : ℕ) : Prop :=
  a * b * c = 162

-- Given the above definition, we need to prove the sum of these ages is 20
theorem sum_of_ages_twins (a b c : ℕ) (h : twin_sisters_ages a b c) (ha : b = c) : a + b + c = 20 :=
by 
  sorry

end NUMINAMATH_GPT_sum_of_ages_twins_l1250_125040


namespace NUMINAMATH_GPT_power_equivalence_l1250_125077

theorem power_equivalence (p : ℕ) (hp : 81^10 = 3^p) : p = 40 :=
by {
  -- Proof steps would go here
  sorry
}

end NUMINAMATH_GPT_power_equivalence_l1250_125077


namespace NUMINAMATH_GPT_product_of_three_consecutive_integers_surrounding_twin_primes_divisible_by_240_l1250_125083

theorem product_of_three_consecutive_integers_surrounding_twin_primes_divisible_by_240
 (p : ℕ) (prime_p : Prime p) (prime_p_plus_2 : Prime (p + 2)) (p_gt_7 : p > 7) :
  240 ∣ ((p - 1) * p * (p + 1)) := by
  sorry

end NUMINAMATH_GPT_product_of_three_consecutive_integers_surrounding_twin_primes_divisible_by_240_l1250_125083


namespace NUMINAMATH_GPT_paul_total_vertical_distance_l1250_125085

def total_vertical_distance
  (n_stories : ℕ)
  (trips_per_day : ℕ)
  (days_in_week : ℕ)
  (height_per_story : ℕ)
  : ℕ :=
  let trips_per_week := trips_per_day * days_in_week
  let distance_per_trip := n_stories * height_per_story
  trips_per_week * distance_per_trip

theorem paul_total_vertical_distance :
  total_vertical_distance 5 6 7 10 = 2100 :=
by
  -- Proof is omitted.
  sorry

end NUMINAMATH_GPT_paul_total_vertical_distance_l1250_125085


namespace NUMINAMATH_GPT_boy_reaches_early_l1250_125023

-- Given conditions
def usual_time : ℚ := 42
def rate_multiplier : ℚ := 7 / 6

-- Derived variables
def new_time : ℚ := (6 / 7) * usual_time
def early_time : ℚ := usual_time - new_time

-- The statement to prove
theorem boy_reaches_early : early_time = 6 := by
  sorry

end NUMINAMATH_GPT_boy_reaches_early_l1250_125023


namespace NUMINAMATH_GPT_fraction_multiplication_l1250_125008

theorem fraction_multiplication : ((1 / 2) * (1 / 3) * (1 / 6) * 72 = 2) :=
by
  sorry

end NUMINAMATH_GPT_fraction_multiplication_l1250_125008


namespace NUMINAMATH_GPT_continuous_at_4_l1250_125009

noncomputable def f (x : ℝ) := 3 * x^2 - 3

theorem continuous_at_4 : 
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x - 4| < δ → |f x - f 4| < ε :=
by
  sorry

end NUMINAMATH_GPT_continuous_at_4_l1250_125009


namespace NUMINAMATH_GPT_wire_length_unique_l1250_125087

noncomputable def distance_increment := (5 / 3)

theorem wire_length_unique (d L : ℝ) 
  (h1 : L = 25 * d) 
  (h2 : L = 24 * (d + distance_increment)) :
  L = 1000 := by
  sorry

end NUMINAMATH_GPT_wire_length_unique_l1250_125087


namespace NUMINAMATH_GPT_Mr_Tom_invested_in_fund_X_l1250_125018

theorem Mr_Tom_invested_in_fund_X (a b : ℝ) (h1 : a + b = 100000) (h2 : 0.17 * b = 0.23 * a + 200) : a = 42000 := 
by
  sorry

end NUMINAMATH_GPT_Mr_Tom_invested_in_fund_X_l1250_125018


namespace NUMINAMATH_GPT_union_A_B_l1250_125007

open Set Real

def A : Set ℝ := {x | x^2 - x - 2 < 0}
def B : Set ℝ := {y | ∃ x : ℝ, y = sin x}

theorem union_A_B : A ∪ B = Ico (-1 : ℝ) 2 := by
  sorry

end NUMINAMATH_GPT_union_A_B_l1250_125007


namespace NUMINAMATH_GPT_rectangle_area_constant_l1250_125003

noncomputable def k (d : ℝ) : ℝ :=
  let x := d / Real.sqrt 29
  10 / 29

theorem rectangle_area_constant (d : ℝ) : 
  let k := 10 / 29
  let length := 5 * (d / Real.sqrt 29)
  let width := 2 * (d / Real.sqrt 29)
  let diagonal := d
  let area := length * width
  area = k * d^2 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_constant_l1250_125003


namespace NUMINAMATH_GPT_required_butter_l1250_125055

-- Define the given conditions
variables (butter sugar : ℕ)
def recipe_butter : ℕ := 25
def recipe_sugar : ℕ := 125
def used_sugar : ℕ := 1000

-- State the theorem
theorem required_butter (h1 : butter = recipe_butter) (h2 : sugar = recipe_sugar) :
  (used_sugar * recipe_butter) / recipe_sugar = 200 := 
by 
  sorry

end NUMINAMATH_GPT_required_butter_l1250_125055


namespace NUMINAMATH_GPT_find_a11_l1250_125099

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem find_a11 (a : ℕ → ℝ)
  (h1 : geometric_sequence a)
  (h2 : a 1 * a 4 = 20)
  (h3 : a 0 + a 5 = 9) :
  a 10 = 25 / 4 :=
sorry

end NUMINAMATH_GPT_find_a11_l1250_125099


namespace NUMINAMATH_GPT_original_price_of_dish_l1250_125049

variable (P : ℝ)

def john_paid (P : ℝ) : ℝ := 0.9 * P + 0.15 * P
def jane_paid (P : ℝ) : ℝ := 0.9 * P + 0.135 * P

theorem original_price_of_dish (h : john_paid P = jane_paid P + 1.26) : P = 84 := by
  sorry

end NUMINAMATH_GPT_original_price_of_dish_l1250_125049


namespace NUMINAMATH_GPT_munchausen_forest_l1250_125041

theorem munchausen_forest (E B : ℕ) (h : B = 10 * E) : B > E := by sorry

end NUMINAMATH_GPT_munchausen_forest_l1250_125041


namespace NUMINAMATH_GPT_quadratic_function_difference_zero_l1250_125029

theorem quadratic_function_difference_zero
  (a b c x1 x2 x3 x4 x5 p q : ℝ)
  (h1 : a ≠ 0)
  (h2 : a * x1^2 + b * x1 + c = 5)
  (h3 : a * (x2 + x3 + x4 + x5)^2 + b * (x2 + x3 + x4 + x5) + c = 5)
  (h4 : x1 ≠ x2 + x3 + x4 + x5)
  (h5 : a * (x1 + x2)^2 + b * (x1 + x2) + c = p)
  (h6 : a * (x3 + x4 + x5)^2 + b * (x3 + x4 + x5) + c = q) :
  p - q = 0 := 
sorry

end NUMINAMATH_GPT_quadratic_function_difference_zero_l1250_125029


namespace NUMINAMATH_GPT_bus_people_count_l1250_125064

-- Define the initial number of people on the bus
def initial_people_on_bus : ℕ := 34

-- Define the number of people who got off the bus
def people_got_off : ℕ := 11

-- Define the number of people who got on the bus
def people_got_on : ℕ := 24

-- Define the final number of people on the bus
def final_people_on_bus : ℕ := (initial_people_on_bus - people_got_off) + people_got_on

-- Theorem: The final number of people on the bus is 47.
theorem bus_people_count : final_people_on_bus = 47 := by
  sorry

end NUMINAMATH_GPT_bus_people_count_l1250_125064


namespace NUMINAMATH_GPT_tetrahedron_volume_l1250_125059

noncomputable def volume_of_tetrahedron (S1 S2 a α : ℝ) : ℝ :=
  (2 * S1 * S2 * Real.sin α) / (3 * a)

theorem tetrahedron_volume (S1 S2 a α : ℝ) :
  a > 0 → S1 > 0 → S2 > 0 → α ≥ 0 → α ≤ Real.pi → volume_of_tetrahedron S1 S2 a α =
  (2 * S1 * S2 * Real.sin α) / (3 * a) := 
by
  intros
  -- The proof is omitted here.
  sorry

end NUMINAMATH_GPT_tetrahedron_volume_l1250_125059


namespace NUMINAMATH_GPT_evaluate_expression_l1250_125093

theorem evaluate_expression (x : ℝ) : 
  x * (x * (x * (3 - x) - 5) + 12) + 2 = -x^4 + 3 * x^3 - 5 * x^2 + 12 * x + 2 := 
by 
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1250_125093


namespace NUMINAMATH_GPT_sum_of_interiors_l1250_125078

theorem sum_of_interiors (n : ℕ) (h : 180 * (n - 2) = 1620) : 180 * ((n + 3) - 2) = 2160 :=
by sorry

end NUMINAMATH_GPT_sum_of_interiors_l1250_125078


namespace NUMINAMATH_GPT_find_sin_B_l1250_125015

variables (a b c : ℝ) (A B C : ℝ)

def sin_law_abc (a b : ℝ) (sinA : ℝ) (sinB : ℝ) : Prop := 
  (a / sinA) = (b / sinB)

theorem find_sin_B {a b : ℝ} (sinA : ℝ) 
  (ha : a = 3) 
  (hb : b = 5) 
  (hA : sinA = 1 / 3) :
  ∃ sinB : ℝ, (sinB = 5 / 9) ∧ sin_law_abc a b sinA sinB :=
by
  use 5 / 9
  simp [sin_law_abc, ha, hb, hA]
  sorry

end NUMINAMATH_GPT_find_sin_B_l1250_125015


namespace NUMINAMATH_GPT_smallest_possible_value_other_integer_l1250_125090

theorem smallest_possible_value_other_integer (x : ℕ) (n : ℕ) (h_pos : x > 0)
  (h_gcd : ∃ m, Nat.gcd m n = x + 3 ∧ m = 30) 
  (h_lcm : Nat.lcm 30 n = x * (x + 3)) :
  n = 162 := 
by sorry

end NUMINAMATH_GPT_smallest_possible_value_other_integer_l1250_125090


namespace NUMINAMATH_GPT_frac_x_y_value_l1250_125084

theorem frac_x_y_value (x y : ℝ) (h1 : 3 < (2 * x - y) / (x + 2 * y))
(h2 : (2 * x - y) / (x + 2 * y) < 7) (h3 : ∃ (t : ℤ), x = t * y) : x / y = -4 := by
  sorry

end NUMINAMATH_GPT_frac_x_y_value_l1250_125084


namespace NUMINAMATH_GPT_negation_if_then_l1250_125034

theorem negation_if_then (x : ℝ) : ¬ (x > 2 → x > 1) ↔ (x ≤ 2 → x ≤ 1) :=
by 
  sorry

end NUMINAMATH_GPT_negation_if_then_l1250_125034


namespace NUMINAMATH_GPT_molecular_weight_calc_l1250_125065

theorem molecular_weight_calc (total_weight : ℕ) (num_moles : ℕ) (one_mole_weight : ℕ) :
  total_weight = 1170 → num_moles = 5 → one_mole_weight = total_weight / num_moles → one_mole_weight = 234 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_molecular_weight_calc_l1250_125065


namespace NUMINAMATH_GPT_sum_of_reciprocals_l1250_125080

noncomputable def reciprocal_sum (x y : ℝ) : ℝ :=
  (1 / x) + (1 / y)

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 40) (h2 : x * y = 375) :
  reciprocal_sum x y = 8 / 75 :=
by
  unfold reciprocal_sum
  -- Intermediate steps would go here, but we'll use sorry to denote the proof is omitted.
  sorry

end NUMINAMATH_GPT_sum_of_reciprocals_l1250_125080


namespace NUMINAMATH_GPT_fewer_servings_per_day_l1250_125069

theorem fewer_servings_per_day :
  ∀ (daily_consumption servings_old servings_new: ℕ),
    daily_consumption = 64 →
    servings_old = 8 →
    servings_new = 16 →
    (daily_consumption / servings_old) - (daily_consumption / servings_new) = 4 :=
by
  intros daily_consumption servings_old servings_new h1 h2 h3
  sorry

end NUMINAMATH_GPT_fewer_servings_per_day_l1250_125069
