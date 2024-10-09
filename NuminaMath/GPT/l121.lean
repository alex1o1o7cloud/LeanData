import Mathlib

namespace value_of_2_68_times_0_74_l121_12100

theorem value_of_2_68_times_0_74 : 
  (268 * 74 = 19732) → (2.68 * 0.74 = 1.9732) :=
by intro h1; sorry

end value_of_2_68_times_0_74_l121_12100


namespace breaks_difference_l121_12177

-- James works for 240 minutes
def total_work_time : ℕ := 240

-- He takes a water break every 20 minutes
def water_break_interval : ℕ := 20

-- He takes a sitting break every 120 minutes
def sitting_break_interval : ℕ := 120

-- Calculate the number of water breaks James takes
def number_of_water_breaks : ℕ := total_work_time / water_break_interval

-- Calculate the number of sitting breaks James takes
def number_of_sitting_breaks : ℕ := total_work_time / sitting_break_interval

-- Prove the difference between the number of water breaks and sitting breaks is 10
theorem breaks_difference :
  number_of_water_breaks - number_of_sitting_breaks = 10 :=
by
  -- calculate number_of_water_breaks = 12
  -- calculate number_of_sitting_breaks = 2
  -- check the difference 12 - 2 = 10
  sorry

end breaks_difference_l121_12177


namespace muffins_division_l121_12190

theorem muffins_division (total_muffins total_people muffins_per_person : ℕ) 
  (h1 : total_muffins = 20) (h2 : total_people = 5) (h3 : muffins_per_person = total_muffins / total_people) : 
  muffins_per_person = 4 := 
by
  sorry

end muffins_division_l121_12190


namespace sum_of_squares_l121_12131

theorem sum_of_squares (b j s : ℕ) 
  (h1 : 3 * b + 2 * j + 4 * s = 70)
  (h2 : 4 * b + 3 * j + 2 * s = 88) : 
  b^2 + j^2 + s^2 = 405 := 
sorry

end sum_of_squares_l121_12131


namespace sum_of_7_and_2_terms_l121_12132

open Nat

variable {α : Type*} [Field α]

-- Definitions
def is_arithmetic_sequence (a : ℕ → α) (d : α) : Prop :=
  ∀ n : ℕ, a (n + 1) - a n = d
  
def is_geometric_sequence (a : ℕ → α) : Prop :=
  ∀ m n k : ℕ, m < n → n < k → a n * a n = a m * a k
  
def sum_first_n_terms (a : ℕ → α) (n : ℕ) : α :=
  (n * (a 0 + a n)) / 2

-- Given Conditions
variable (a : ℕ → α) 
variable (d : α)

-- Checked. Arithmetic sequence with non-zero common difference
axiom h1 : is_arithmetic_sequence a d

-- Known values provided in the problem statement
axiom h2 : a 1 = 6

-- Terms forming a geometric sequence
axiom h3 : is_geometric_sequence a

-- The goal is to find the sum of the first 7 terms and the first 2 terms
theorem sum_of_7_and_2_terms : sum_first_n_terms a 7 + sum_first_n_terms a 2 = 80 := 
by {
  -- Proof will be here
  sorry
}

end sum_of_7_and_2_terms_l121_12132


namespace min_rice_pounds_l121_12109

variable {o r : ℝ}

theorem min_rice_pounds (h1 : o ≥ 8 + r / 3) (h2 : o ≤ 2 * r) : r ≥ 5 :=
sorry

end min_rice_pounds_l121_12109


namespace remaining_distance_is_one_l121_12106

def total_distance_to_grandma : ℕ := 78
def initial_distance_traveled : ℕ := 35
def bakery_detour : ℕ := 7
def pie_distance : ℕ := 18
def gift_detour : ℕ := 3
def next_travel_distance : ℕ := 12
def scenic_detour : ℕ := 2

def total_distance_traveled : ℕ :=
  initial_distance_traveled + bakery_detour + pie_distance + gift_detour + next_travel_distance + scenic_detour

theorem remaining_distance_is_one :
  total_distance_to_grandma - total_distance_traveled = 1 := by
  sorry

end remaining_distance_is_one_l121_12106


namespace digit_b_divisible_by_5_l121_12169

theorem digit_b_divisible_by_5 (B : ℕ) (h : B = 0 ∨ B = 5) : 
  (∃ n : ℕ, (947 * 10 + B) = 5 * n) ↔ (B = 0 ∨ B = 5) :=
by {
  sorry
}

end digit_b_divisible_by_5_l121_12169


namespace total_area_correct_l121_12135

-- Define the given conditions
def dust_covered_area : ℕ := 64535
def untouched_area : ℕ := 522

-- Define the total area of prairie by summing covered and untouched areas
def total_prairie_area : ℕ := dust_covered_area + untouched_area

-- State the theorem we need to prove
theorem total_area_correct : total_prairie_area = 65057 := by
  sorry

end total_area_correct_l121_12135


namespace cos_double_angle_unit_circle_l121_12180

theorem cos_double_angle_unit_circle (α y₀ : ℝ) (h : (1/2)^2 + y₀^2 = 1) : 
  Real.cos (2 * α) = -1/2 :=
by 
  -- The proof is omitted
  sorry

end cos_double_angle_unit_circle_l121_12180


namespace oxygen_atom_count_l121_12184

-- Definitions and conditions
def molecular_weight_C : ℝ := 12.01
def molecular_weight_H : ℝ := 1.008
def molecular_weight_O : ℝ := 16.00

def num_carbon_atoms : ℕ := 4
def num_hydrogen_atoms : ℕ := 1
def total_molecular_weight : ℝ := 65.0

-- Theorem statement
theorem oxygen_atom_count : 
  ∃ (num_oxygen_atoms : ℕ), 
  num_oxygen_atoms * molecular_weight_O = total_molecular_weight - (num_carbon_atoms * molecular_weight_C + num_hydrogen_atoms * molecular_weight_H) 
  ∧ num_oxygen_atoms = 1 :=
by
  sorry

end oxygen_atom_count_l121_12184


namespace remainder_addition_l121_12182

theorem remainder_addition (m : ℕ) (k : ℤ) (h : m = 9 * k + 4) : (m + 2025) % 9 = 4 := by
  sorry

end remainder_addition_l121_12182


namespace order_y1_y2_y3_l121_12146

-- Defining the parabolic function and the points A, B, C
def parabola (a x : ℝ) : ℝ :=
  a * x^2 - 2 * a * x + 3

-- Points A, B, C
def y1 (a : ℝ) : ℝ := parabola a (-1)
def y2 (a : ℝ) : ℝ := parabola a 2
def y3 (a : ℝ) : ℝ := parabola a 4

-- Assumption: a > 0
variables (a : ℝ) (h : a > 0)

-- The theorem to prove
theorem order_y1_y2_y3 : 
  y2 a < y1 a ∧ y1 a < y3 a :=
sorry

end order_y1_y2_y3_l121_12146


namespace bus_commutes_three_times_a_week_l121_12127

-- Define the commuting times
def bike_time := 30
def bus_time := bike_time + 10
def friend_time := bike_time * (1 - (2/3))
def total_weekly_time := 160

-- Define the number of times taking the bus as a variable
variable (b : ℕ)

-- The equation for total commuting time
def commuting_time_eq := bike_time + bus_time * b + friend_time = total_weekly_time

-- The proof statement: b should be equal to 3
theorem bus_commutes_three_times_a_week (h : commuting_time_eq b) : b = 3 := sorry

end bus_commutes_three_times_a_week_l121_12127


namespace perpendicular_lines_parallel_l121_12179

noncomputable def line := Type
noncomputable def plane := Type

variables (m n : line) (α : plane)

def parallel (l1 l2 : line) : Prop := sorry -- Definition of parallel lines
def perpendicular (l : line) (α : plane) : Prop := sorry -- Definition of perpendicular line to a plane

theorem perpendicular_lines_parallel (h1 : perpendicular m α) (h2 : perpendicular n α) : parallel m n :=
sorry

end perpendicular_lines_parallel_l121_12179


namespace min_value_expression_l121_12165

variable (a b c : ℝ)
variable (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
variable (h_eq : a * b * c = 64)

theorem min_value_expression :
  a^2 + 6 * a * b + 9 * b^2 + 3 * c^2 ≥ 192 :=
by {
  sorry
}

end min_value_expression_l121_12165


namespace pyramid_base_edge_length_l121_12170

noncomputable def edge_length_of_pyramid_base : ℝ :=
  let R := 4 -- radius of the hemisphere
  let h := 12 -- height of the pyramid
  let base_length := 6 -- edge-length of the base of the pyramid to be proved
  -- assume necessary geometric configurations of the pyramid and sphere
  base_length

theorem pyramid_base_edge_length :
  ∀ R h base_length, R = 4 → h = 12 → edge_length_of_pyramid_base = base_length → base_length = 6 :=
by
  intros R h base_length hR hH hBaseLength
  have R_spec : R = 4 := hR
  have h_spec : h = 12 := hH
  have base_length_spec : edge_length_of_pyramid_base = base_length := hBaseLength
  sorry

end pyramid_base_edge_length_l121_12170


namespace quadratic_root_other_l121_12103

theorem quadratic_root_other (a : ℝ) (h : (3 : ℝ)*3 - 2*3 + a = 0) : 
  ∃ (b : ℝ), b = -1 ∧ (b : ℝ)*b - 2*b + a = 0 :=
by
  sorry

end quadratic_root_other_l121_12103


namespace books_sold_correct_l121_12153

-- Define the initial number of books, number of books added, and the final number of books.
def initial_books : ℕ := 41
def added_books : ℕ := 2
def final_books : ℕ := 10

-- Define the number of books sold.
def sold_books : ℕ := initial_books + added_books - final_books

-- The theorem we need to prove: the number of books sold is 33.
theorem books_sold_correct : sold_books = 33 := by
  sorry

end books_sold_correct_l121_12153


namespace prod_of_consecutive_nums_divisible_by_504_l121_12166

theorem prod_of_consecutive_nums_divisible_by_504
  (a : ℕ)
  (h : ∃ b : ℕ, a = b ^ 3) :
  (a^3 - 1) * a^3 * (a^3 + 1) % 504 = 0 := 
sorry

end prod_of_consecutive_nums_divisible_by_504_l121_12166


namespace medicine_price_after_discount_l121_12119

theorem medicine_price_after_discount :
  ∀ (price : ℝ) (discount : ℝ), price = 120 → discount = 0.3 → 
  (price - price * discount) = 84 :=
by
  intros price discount h1 h2
  rw [h1, h2]
  sorry

end medicine_price_after_discount_l121_12119


namespace worst_is_father_l121_12126

-- Definitions for players
inductive Player
| father
| sister
| daughter
| son
deriving DecidableEq

open Player

def opposite_sex (p1 p2 : Player) : Bool :=
match p1, p2 with
| father, sister => true
| father, daughter => true
| sister, father => true
| daughter, father => true
| son, sister => true
| son, daughter => true
| daughter, son => true
| sister, son => true
| _, _ => false 

-- Problem conditions
variables (worst best : Player)
variable (twins : Player → Player)
variable (worst_best_twins : twins worst = best)
variable (worst_twin_conditions : opposite_sex (twins worst) best)

-- Goal: Prove that the worst player is the father
theorem worst_is_father : worst = Player.father := by
  sorry

end worst_is_father_l121_12126


namespace probability_of_two_mathematicians_living_contemporarily_l121_12191

noncomputable def probability_of_contemporary_lifespan : ℚ :=
  let total_area := 500 * 500
  let triangle_area := 0.5 * 380 * 380
  let non_overlap_area := 2 * triangle_area
  let overlap_area := total_area - non_overlap_area
  overlap_area / total_area

theorem probability_of_two_mathematicians_living_contemporarily :
  probability_of_contemporary_lifespan = 2232 / 5000 :=
by
  -- The actual proof would go here
  sorry

end probability_of_two_mathematicians_living_contemporarily_l121_12191


namespace driver_total_miles_per_week_l121_12108

theorem driver_total_miles_per_week :
  let distance_monday_to_saturday := (30 * 3 + 25 * 4 + 40 * 2) * 6
  let distance_sunday := 35 * (5 - 1)
  distance_monday_to_saturday + distance_sunday = 1760 := by
  sorry

end driver_total_miles_per_week_l121_12108


namespace union_of_sets_l121_12129

theorem union_of_sets (A B : Set α) : A ∪ B = { x | x ∈ A ∨ x ∈ B } :=
by
  sorry

end union_of_sets_l121_12129


namespace quadratic_has_real_root_l121_12196

theorem quadratic_has_real_root (p : ℝ) : 
  ∃ x : ℝ, 3 * (p + 2) * x^2 - p * x - (4 * p + 7) = 0 :=
sorry

end quadratic_has_real_root_l121_12196


namespace intersection_of_A_and_B_l121_12156

def A : Set ℝ := { x | x^2 - x - 2 ≥ 0 }
def B : Set ℝ := { x | -2 ≤ x ∧ x < 2 }

theorem intersection_of_A_and_B :
  A ∩ B = { x | -2 ≤ x ∧ x ≤ -1 } := by
-- The proof would go here
sorry

end intersection_of_A_and_B_l121_12156


namespace area_of_rhombus_is_375_l121_12152

-- define the given diagonals
def diagonal1 := 25
def diagonal2 := 30

-- define the formula for the area of a rhombus
def area_of_rhombus (d1 d2 : ℕ) : ℕ := (d1 * d2) / 2

-- state the theorem
theorem area_of_rhombus_is_375 : area_of_rhombus diagonal1 diagonal2 = 375 := 
by 
  -- The proof is omitted as per the requirement
  sorry

end area_of_rhombus_is_375_l121_12152


namespace employee_n_salary_l121_12168

theorem employee_n_salary (x : ℝ) (h : x + 1.2 * x = 583) : x = 265 := sorry

end employee_n_salary_l121_12168


namespace stable_number_divisible_by_11_l121_12114

/-- Definition of a stable number as a three-digit number (cen, ten, uni) where
    each digit is non-zero, and the sum of any two digits is greater than the remaining digit.
-/
def is_stable_number (cen ten uni : ℕ) : Prop :=
cen ≠ 0 ∧ ten ≠ 0 ∧ uni ≠ 0 ∧
(cen + ten > uni) ∧ (cen + uni > ten) ∧ (ten + uni > cen)

/-- Function F defined for a stable number (cen ten uni). -/
def F (cen ten uni : ℕ) : ℕ := 10 * ten + cen + uni

/-- Function Q defined for a stable number (cen ten uni). -/
def Q (cen ten uni : ℕ) : ℕ := 10 * cen + ten + uni

/-- Statement to prove: Given a stable number s = 100a + 101b + 30 where 1 ≤ a ≤ 5 and 1 ≤ b ≤ 4,
    the expression 5 * F(s) + 2 * Q(s) is divisible by 11.
-/
theorem stable_number_divisible_by_11 (a b cen ten uni : ℕ)
  (h_a : 1 ≤ a ∧ a ≤ 5)
  (h_b : 1 ≤ b ∧ b ≤ 4)
  (h_s : 100 * a + 101 * b + 30 = 100 * cen + 10 * ten + uni)
  (h_stable : is_stable_number cen ten uni) :
  (5 * F cen ten uni + 2 * Q cen ten uni) % 11 = 0 :=
sorry

end stable_number_divisible_by_11_l121_12114


namespace monotonic_increasing_interval_l121_12124

noncomputable def log_base := (1 / 4 : ℝ)

def quad_expression (x : ℝ) : ℝ := -x^2 + 2*x + 3

def is_defined (x : ℝ) : Prop := quad_expression x > 0

theorem monotonic_increasing_interval : ∀ (x : ℝ), 
  is_defined x → 
  ∃ (a b : ℝ), 1 < a ∧ a ≤ x ∧ x < b ∧ b < 3 :=
by
  sorry

end monotonic_increasing_interval_l121_12124


namespace find_m_value_l121_12193

theorem find_m_value
  (x y : ℤ)
  (h1 : x = 2)
  (h2 : y = m)
  (h3 : 3 * x + 2 * y = 10) : 
  m = 2 :=
by
  sorry

end find_m_value_l121_12193


namespace center_digit_is_two_l121_12105

theorem center_digit_is_two :
  ∃ (a b : ℕ), (a^2 < 1000 ∧ b^2 < 1000 ∧ (a^2 ≠ b^2) ∧
  (∀ d, d ∈ [a^2 / 100, (a^2 / 10) % 10, a^2 % 10] → d ∈ [2, 3, 4, 5, 6]) ∧
  (∀ d, d ∈ [b^2 / 100, (b^2 / 10) % 10, b^2 % 10] → d ∈ [2, 3, 4, 5, 6])) ∧
  (∀ d, (d ∈ [2, 3, 4, 5, 6]) → (d ∈ [a^2 / 100, (a^2 / 10) % 10, a^2 % 10] ∨ d ∈ [b^2 / 100, (b^2 / 10) % 10, b^2 % 10])) ∧
  2 = (a^2 / 10) % 10 ∨ 2 = (b^2 / 10) % 10 :=
sorry -- no proof needed, just the statement

end center_digit_is_two_l121_12105


namespace water_wasted_per_hour_l121_12175

def drips_per_minute : ℝ := 10
def volume_per_drop : ℝ := 0.05

def drops_per_hour : ℝ := 60 * drips_per_minute
def total_volume : ℝ := drops_per_hour * volume_per_drop

theorem water_wasted_per_hour : total_volume = 30 :=
by
  sorry

end water_wasted_per_hour_l121_12175


namespace solve_system_of_inequalities_l121_12110

theorem solve_system_of_inequalities (x : ℝ) :
  4*x^2 - 27*x + 18 > 0 ∧ x^2 + 4*x + 4 > 0 ↔ (x < 3/4 ∨ x > 6) ∧ x ≠ -2 :=
by
  sorry

end solve_system_of_inequalities_l121_12110


namespace line_through_P_parallel_to_AB_circumcircle_of_triangle_OAB_l121_12128

-- Define the points A, B and P
def A : ℝ × ℝ := (4, 0)
def B : ℝ × ℝ := (0, 2)
def P : ℝ × ℝ := (2, 3)
def O : ℝ × ℝ := (0, 0)

-- Define the functions and theorems for the problem
theorem line_through_P_parallel_to_AB :
  ∃ k b : ℝ, ∀ x y : ℝ, ((y = k * x + b) ↔ (x + 2 * y - 8 = 0)) :=
sorry

theorem circumcircle_of_triangle_OAB :
  ∃ cx cy r : ℝ, (cx, cy) = (2, 1) ∧ r^2 = 5 ∧ ∀ x y : ℝ, ((x - cx)^2 + (y - cy)^2 = r^2) ↔ ((x - 2)^2 + (y - 1)^2 = 5) :=
sorry

end line_through_P_parallel_to_AB_circumcircle_of_triangle_OAB_l121_12128


namespace baron_not_lying_l121_12116

def sum_of_digits (n : ℕ) : ℕ :=
  if n = 0 then 0 else (n % 10) + sum_of_digits (n / 10)

theorem baron_not_lying : 
  ∃ a b : ℕ, 
  (a ≠ b ∧ a ≥ 10^9 ∧ a < 10^10 ∧ b ≥ 10^9 ∧ b < 10^10 ∧ a % 10 ≠ 0 ∧ b % 10 ≠ 0 ∧ 
  (a + sum_of_digits (a * a) = b + sum_of_digits (b * b))) :=
  sorry

end baron_not_lying_l121_12116


namespace visitors_not_enjoyed_not_understood_l121_12150

theorem visitors_not_enjoyed_not_understood (V E U : ℕ) (hv_v : V = 520)
  (hu_e : E = U) (he : E = 3 * V / 4) : (V / 4) = 130 :=
by
  rw [hv_v] at he
  sorry

end visitors_not_enjoyed_not_understood_l121_12150


namespace remainder_of_sum_of_primes_mod_eighth_prime_l121_12107

def sum_first_seven_primes : ℕ := 2 + 3 + 5 + 7 + 11 + 13 + 17

def eighth_prime : ℕ := 19

theorem remainder_of_sum_of_primes_mod_eighth_prime : sum_first_seven_primes % eighth_prime = 1 := by
  sorry

end remainder_of_sum_of_primes_mod_eighth_prime_l121_12107


namespace exists_decreasing_lcm_sequence_l121_12122

theorem exists_decreasing_lcm_sequence :
  ∃ (a : Fin 100 → ℕ), 
    (∀ i j, i < j → a i < a j) ∧ 
    (∀ i : Fin 99, Nat.lcm (a i) (a (i + 1)) > Nat.lcm (a (i + 1)) (a (i + 2))) :=
sorry

end exists_decreasing_lcm_sequence_l121_12122


namespace find_58th_digit_in_fraction_l121_12159

def decimal_representation_of_fraction : ℕ := sorry

theorem find_58th_digit_in_fraction:
  (decimal_representation_of_fraction = 4) := sorry

end find_58th_digit_in_fraction_l121_12159


namespace inequality_abc_l121_12141

open Real

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / sqrt (a^2 + 8 * b * c)) + (b / sqrt (b^2 + 8 * c * a)) + (c / sqrt (c^2 + 8 * a * b)) ≥ 1 :=
sorry

end inequality_abc_l121_12141


namespace min_value_2xy_minus_2x_minus_y_l121_12143

theorem min_value_2xy_minus_2x_minus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1/x + 2/y = 1) :
  2 * x * y - 2 * x - y ≥ 8 :=
sorry

end min_value_2xy_minus_2x_minus_y_l121_12143


namespace average_of_4_8_N_l121_12194

-- Define the condition for N
variable (N : ℝ) (cond : 7 < N ∧ N < 15)

-- State the theorem to prove
theorem average_of_4_8_N (N : ℝ) (h : 7 < N ∧ N < 15) :
  (frac12 + N) / 3 = 7 ∨ (12 + N) / 3 = 9 :=
sorry

end average_of_4_8_N_l121_12194


namespace sum_of_last_two_digits_l121_12115

theorem sum_of_last_two_digits (x y : ℕ) : 
  x = 8 → y = 12 → (x^25 + y^25) % 100 = 0 := 
by
  intros hx hy
  sorry

end sum_of_last_two_digits_l121_12115


namespace distance_between_points_l121_12199

def point1 : ℝ × ℝ := (3.5, -2)
def point2 : ℝ × ℝ := (7.5, 5)

theorem distance_between_points : 
  Real.sqrt ((point2.1 - point1.1)^2 + (point2.2 - point1.2)^2) = Real.sqrt 65 := by
  sorry

end distance_between_points_l121_12199


namespace num_pairs_sold_l121_12121

theorem num_pairs_sold : 
  let total_amount : ℤ := 735
  let avg_price : ℝ := 9.8
  let num_pairs : ℝ := total_amount / avg_price
  num_pairs = 75 :=
by
  let total_amount : ℤ := 735
  let avg_price : ℝ := 9.8
  let num_pairs : ℝ := total_amount / avg_price
  exact sorry

end num_pairs_sold_l121_12121


namespace tom_books_l121_12195

theorem tom_books (books_may books_june books_july : ℕ) (h_may : books_may = 2) (h_june : books_june = 6) (h_july : books_july = 10) : 
books_may + books_june + books_july = 18 := by
sorry

end tom_books_l121_12195


namespace distance_to_y_axis_parabola_midpoint_l121_12140

noncomputable def distance_from_midpoint_to_y_axis (x1 x2 : ℝ) : ℝ :=
  (x1 + x2) / 2

theorem distance_to_y_axis_parabola_midpoint :
  ∀ (x1 y1 x2 y2 : ℝ), y1^2 = x1 → y2^2 = x2 → 
  abs (x1 + 1 / 4) + abs (x2 + 1 / 4) = 3 →
  abs (distance_from_midpoint_to_y_axis x1 x2) = 5 / 4 :=
by
  intros x1 y1 x2 y2 h1 h2 h3
  sorry

end distance_to_y_axis_parabola_midpoint_l121_12140


namespace num_possible_lists_l121_12148

theorem num_possible_lists :
  let binA_balls := 8
  let binB_balls := 5
  let total_lists := binA_balls * binB_balls
  total_lists = 40 := by
{
  let binA_balls := 8
  let binB_balls := 5
  let total_lists := binA_balls * binB_balls
  show total_lists = 40
  exact rfl
}

end num_possible_lists_l121_12148


namespace toothbrush_count_l121_12158

theorem toothbrush_count (T A : ℕ) (h1 : 53 + 67 + 46 = 166)
  (h2 : 67 - 36 = 31) (h3 : A = 31) (h4 : T = 166 + 2 * A) :
  T = 228 :=
  by 
  -- Using Lean's sorry keyword to skip the proof
  sorry

end toothbrush_count_l121_12158


namespace crumbs_triangle_area_l121_12197

theorem crumbs_triangle_area :
  ∀ (table_length table_width : ℝ) (crumbs : ℕ),
    table_length = 2 ∧ table_width = 1 ∧ crumbs = 500 →
    ∃ (triangle_area : ℝ), (triangle_area < 0.005 ∧ ∃ (a b c : Type), a ≠ b ∧ b ≠ c ∧ a ≠ c) :=
by
  sorry

end crumbs_triangle_area_l121_12197


namespace perpendicular_length_GH_from_centroid_l121_12137

theorem perpendicular_length_GH_from_centroid
  (A B C D E F G : ℝ)
  -- Conditions for distances from vertices to the line RS
  (hAD : AD = 12)
  (hBE : BE = 12)
  (hCF : CF = 18)
  -- Define the coordinates based on the vertical distances to line RS
  (yA : A = 12)
  (yB : B = 12)
  (yC : C = 18)
  -- Define the centroid G of triangle ABC based on the average of the y-coordinates
  (yG : G = (A + B + C) / 3)
  : G = 14 :=
by
  sorry

end perpendicular_length_GH_from_centroid_l121_12137


namespace proof_rewritten_eq_and_sum_l121_12171

-- Define the given equation
def given_eq (x : ℝ) : Prop := 64 * x^2 + 80 * x - 72 = 0

-- Define the rewritten form of the equation
def rewritten_eq (x : ℝ) : Prop := (8 * x + 5)^2 = 97

-- Define the correctness of rewriting the equation
def correct_rewrite (x : ℝ) : Prop :=
  given_eq x → rewritten_eq x

-- Define the correct value of a + b + c
def correct_sum : Prop :=
  8 + 5 + 97 = 110

-- The final theorem statement
theorem proof_rewritten_eq_and_sum (x : ℝ) : correct_rewrite x ∧ correct_sum :=
by
  sorry

end proof_rewritten_eq_and_sum_l121_12171


namespace exists_point_P_equal_distance_squares_l121_12160

-- Define the points in the plane representing the vertices of the triangles
variables {A1 A2 A3 B1 B2 B3 C1 C2 C3 : ℝ × ℝ}
-- Define the function that calculates the square distance between two points
def sq_distance (P Q : ℝ × ℝ) : ℝ :=
  (P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2

-- Define the proof statement
theorem exists_point_P_equal_distance_squares :
  ∃ P : ℝ × ℝ,
    sq_distance P A1 + sq_distance P A2 + sq_distance P A3 =
    sq_distance P B1 + sq_distance P B2 + sq_distance P B3 ∧
    sq_distance P A1 + sq_distance P A2 + sq_distance P A3 =
    sq_distance P C1 + sq_distance P C2 + sq_distance P C3 := sorry

end exists_point_P_equal_distance_squares_l121_12160


namespace smallest_c_l121_12133

theorem smallest_c {a b c : ℤ} (h1 : a < b) (h2 : b < c) 
  (h3 : 2 * b = a + c)
  (h4 : a^2 = c * b) : c = 4 :=
by
  -- We state the theorem here without proof. 
  -- The actual proof steps are omitted and replaced by sorry.
  sorry

end smallest_c_l121_12133


namespace incorrect_option_D_l121_12163

variable {p q : Prop}

theorem incorrect_option_D (hp : ¬p) (hq : q) : ¬(¬q) := 
by 
  sorry  

end incorrect_option_D_l121_12163


namespace cars_count_l121_12188

theorem cars_count
  (distance : ℕ)
  (time_between_cars : ℕ)
  (total_time_hours : ℕ)
  (cars_per_hour : ℕ)
  (expected_cars_count : ℕ) :
  distance = 3 →
  time_between_cars = 20 →
  total_time_hours = 10 →
  cars_per_hour = 3 →
  expected_cars_count = total_time_hours * cars_per_hour →
  expected_cars_count = 30 :=
by
  intros h1 h2 h3 h4 h5
  rw [h3, h4] at h5
  exact h5


end cars_count_l121_12188


namespace sum_due_in_years_l121_12172

theorem sum_due_in_years 
  (D : ℕ)
  (S : ℕ)
  (r : ℚ)
  (H₁ : D = 168)
  (H₂ : S = 768)
  (H₃ : r = 14 / 100) :
  ∃ t : ℕ, t = 2 := 
by
  sorry

end sum_due_in_years_l121_12172


namespace balls_in_drawers_l121_12174

theorem balls_in_drawers (n k : ℕ) (h_n : n = 5) (h_k : k = 2) : (k ^ n) = 32 :=
by
  rw [h_n, h_k]
  sorry

end balls_in_drawers_l121_12174


namespace athlete_total_heartbeats_l121_12189

/-
  An athlete's heart rate starts at 140 beats per minute at the beginning of a race
  and increases by 5 beats per minute for each subsequent mile. How many times does
  the athlete's heart beat during a 10-mile race if the athlete runs at a pace of
  6 minutes per mile?
-/

def athlete_heartbeats (initial_rate : ℕ) (increase_rate : ℕ) (miles : ℕ) (minutes_per_mile : ℕ) : ℕ :=
  let n := miles
  let a := initial_rate
  let l := initial_rate + (increase_rate * (miles - 1))
  let S := (n * (a + l)) / 2
  S * minutes_per_mile

theorem athlete_total_heartbeats :
  athlete_heartbeats 140 5 10 6 = 9750 :=
sorry

end athlete_total_heartbeats_l121_12189


namespace molecularWeight_correct_l121_12101

noncomputable def molecularWeight (nC nH nO nN: ℤ) 
    (wC wH wO wN : ℚ) : ℚ := nC * wC + nH * wH + nO * wO + nN * wN

theorem molecularWeight_correct : 
    molecularWeight 5 12 3 1 12.01 1.008 16.00 14.01 = 134.156 := by
  sorry

end molecularWeight_correct_l121_12101


namespace find_b_value_l121_12181

theorem find_b_value
    (k1 k2 b : ℝ)
    (y1 y2 : ℝ → ℝ)
    (a n : ℝ)
    (h1 : ∀ x, y1 x = k1 / x)
    (h2 : ∀ x, y2 x = k2 * x + b)
    (intersection_A : y1 1 = 4)
    (intersection_B : y2 a = 1 ∧ y1 a = 1)
    (translated_C_y1 : y1 (-1) = n + 6)
    (translated_C_y2 : y2 1 = n)
    (k1k2_nonzero : k1 ≠ 0 ∧ k2 ≠ 0)
    (sum_k1_k2 : k1 + k2 = 0) :
  b = -6 :=
sorry

end find_b_value_l121_12181


namespace charts_per_associate_professor_l121_12111

theorem charts_per_associate_professor (A B C : ℕ) 
  (h1 : A + B = 6) 
  (h2 : 2 * A + B = 10) 
  (h3 : C * A + 2 * B = 8) : 
  C = 1 :=
by
  sorry

end charts_per_associate_professor_l121_12111


namespace max_ab_value_l121_12173

variable (a b c : ℝ)

-- Conditions
axiom h1 : 0 < a ∧ a < 1
axiom h2 : 0 < b ∧ b < 1
axiom h3 : 0 < c ∧ c < 1
axiom h4 : 3 * a + 2 * b = 1

-- Goal
theorem max_ab_value : ab = 1 / 24 :=
by
  sorry

end max_ab_value_l121_12173


namespace intersection_of_sets_A_B_l121_12186

def set_A : Set ℝ := { x : ℝ | x^2 - 2*x - 3 > 0 }
def set_B : Set ℝ := { x : ℝ | -2 < x ∧ x ≤ 2 }
def set_intersection : Set ℝ := { x : ℝ | -2 < x ∧ x < -1 }

theorem intersection_of_sets_A_B :
  (set_A ∩ set_B) = set_intersection :=
  sorry

end intersection_of_sets_A_B_l121_12186


namespace calculate_value_l121_12164

theorem calculate_value : (245^2 - 225^2) / 20 = 470 :=
by
  sorry

end calculate_value_l121_12164


namespace new_number_formed_l121_12134

theorem new_number_formed (h t u : ℕ) (Hh : h < 10) (Ht : t < 10) (Hu : u < 10) :
  let original_number := 100 * h + 10 * t + u
  let new_number := 2000 + 10 * original_number
  new_number = 1000 * (h + 2) + 100 * t + 10 * u :=
by
  -- Proof would go here
  sorry

end new_number_formed_l121_12134


namespace find_x_value_l121_12149

theorem find_x_value (x : ℝ) (h : 0.75 / x = 5 / 6) : x = 0.9 := 
by 
  sorry

end find_x_value_l121_12149


namespace fraction_of_blueberry_tart_l121_12178

/-- Let total leftover tarts be 0.91.
    Let the tart filled with cherries be 0.08.
    Let the tart filled with peaches be 0.08.
    Prove that the fraction of the tart filled with blueberries is 0.75. --/
theorem fraction_of_blueberry_tart (H_total : Real) (H_cherry : Real) (H_peach : Real)
  (H1 : H_total = 0.91) (H2 : H_cherry = 0.08) (H3 : H_peach = 0.08) :
  (H_total - (H_cherry + H_peach)) = 0.75 :=
sorry

end fraction_of_blueberry_tart_l121_12178


namespace cricket_innings_l121_12112

theorem cricket_innings (n : ℕ) 
  (average_run : ℕ := 40) 
  (next_innings_run : ℕ := 84) 
  (new_average_run : ℕ := 44) :
  (40 * n + 84) / (n + 1) = 44 ↔ n = 10 := 
by
  sorry

end cricket_innings_l121_12112


namespace notebook_cost_l121_12118

theorem notebook_cost
  (initial_amount : ℝ)
  (notebook_count : ℕ)
  (pen_count : ℕ)
  (pen_cost : ℝ)
  (remaining_amount : ℝ)
  (total_spent : ℝ)
  (notebook_cost : ℝ) :
  initial_amount = 15 →
  notebook_count = 2 →
  pen_count = 2 →
  pen_cost = 1.5 →
  remaining_amount = 4 →
  total_spent = initial_amount - remaining_amount →
  total_spent = notebook_count * notebook_cost + pen_count * pen_cost →
  notebook_cost = 4 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end notebook_cost_l121_12118


namespace frac_wx_l121_12185

theorem frac_wx (x y z w : ℚ) (h1 : x / y = 5) (h2 : y / z = 1 / 2) (h3 : z / w = 7) : w / x = 2 / 35 :=
by
  sorry

end frac_wx_l121_12185


namespace george_coin_distribution_l121_12136

theorem george_coin_distribution (a b c : ℕ) (h₁ : a = 1050) (h₂ : b = 1260) (h₃ : c = 210) :
  Nat.gcd (Nat.gcd a b) c = 210 :=
by
  sorry

end george_coin_distribution_l121_12136


namespace abs_five_minus_sqrt_pi_l121_12198

theorem abs_five_minus_sqrt_pi : |5 - Real.sqrt Real.pi| = 3.22755 := by
  sorry

end abs_five_minus_sqrt_pi_l121_12198


namespace tan_alpha_plus_cot_alpha_l121_12139

theorem tan_alpha_plus_cot_alpha (α : Real) (h : Real.sin (2 * α) = 3 / 4) : 
  Real.tan α + 1 / Real.tan α = 8 / 3 :=
  sorry

end tan_alpha_plus_cot_alpha_l121_12139


namespace edward_garage_sale_games_l121_12162

variables
  (G_total : ℕ) -- total number of games
  (G_good : ℕ) -- number of good games
  (G_bad : ℕ) -- number of bad games
  (G_friend : ℕ) -- number of games bought from a friend
  (G_garage : ℕ) -- number of games bought at the garage sale

-- The conditions
def total_games (G_total : ℕ) (G_good : ℕ) (G_bad : ℕ) : Prop :=
  G_total = G_good + G_bad

def garage_sale_games (G_total : ℕ) (G_friend : ℕ) (G_garage : ℕ) : Prop :=
  G_total = G_friend + G_garage

-- The theorem to be proved
theorem edward_garage_sale_games
  (G_total : ℕ) 
  (G_good : ℕ) 
  (G_bad : ℕ)
  (G_friend : ℕ) 
  (G_garage : ℕ) 
  (h1 : total_games G_total G_good G_bad)
  (h2 : G_good = 24)
  (h3 : G_bad = 31)
  (h4 : G_friend = 41) :
  G_garage = 14 :=
by
  sorry

end edward_garage_sale_games_l121_12162


namespace neg_p_is_true_neg_q_is_true_l121_12167

theorem neg_p_is_true : ∃ m : ℝ, ∀ x : ℝ, (x^2 + x - m = 0 → False) :=
sorry

theorem neg_q_is_true : ∀ x : ℝ, (x^2 + x + 1 > 0) :=
sorry

end neg_p_is_true_neg_q_is_true_l121_12167


namespace quadratic_inequality_solution_l121_12154

theorem quadratic_inequality_solution (x : ℝ) : 
  (x - 2) * (x + 2) < 5 ↔ -3 < x ∧ x < 3 :=
by sorry

end quadratic_inequality_solution_l121_12154


namespace p_necessary_not_sufficient_for_q_l121_12155

open Classical

variable (p q : Prop)

theorem p_necessary_not_sufficient_for_q (h1 : ¬(p → q)) (h2 : ¬q → ¬p) : (¬(p → q) ∧ (¬q → ¬p) ∧ (¬p → ¬q ∧ ¬(¬q → p))) := by
  sorry

end p_necessary_not_sufficient_for_q_l121_12155


namespace inequality_solution_subset_l121_12187

theorem inequality_solution_subset {x a : ℝ} : (∀ x, |x| > a * x + 1 → x ≤ 0) ↔ a ≥ 1 :=
by sorry

end inequality_solution_subset_l121_12187


namespace m_gt_n_l121_12113

variable (m n : ℝ)

-- Definition of points A and B lying on the line y = -2x + 1
def point_A_on_line : Prop := m = -2 * (-1) + 1
def point_B_on_line : Prop := n = -2 * 3 + 1

-- Theorem stating that m > n given the conditions
theorem m_gt_n (hA : point_A_on_line m) (hB : point_B_on_line n) : m > n :=
by
  -- To avoid the proof part, which we skip as per instructions
  sorry

end m_gt_n_l121_12113


namespace length_of_segment_BD_is_sqrt_3_l121_12102

open Real

-- Define the triangle ABC and the point D according to the problem conditions
def triangle_ABC (A B C : ℝ × ℝ) :=
  B.1 = 0 ∧ B.2 = 0 ∧
  (B.1 - A.1) ^ 2 + (B.2 - A.2) ^ 2 = 3 ∧
  (C.1 - B.1) ^ 2 + (C.2 - B.2) ^ 2 = 7 ∧
  C.2 = 0 ∧ (A.1 - C.1) ^ 2 + A.2 ^ 2 = 10

def point_D (A B C D : ℝ × ℝ) :=
  ∃ BD DC : ℝ, BD + DC = sqrt 7 ∧
  BD / DC = sqrt 3 / sqrt 7 ∧
  D.1 = BD / sqrt 7 ∧ D.2 = 0

-- The theorem to prove
theorem length_of_segment_BD_is_sqrt_3 (A B C D : ℝ × ℝ)
  (h₁ : triangle_ABC A B C)
  (h₂ : point_D A B C D) :
  (sqrt ((D.1 - B.1) ^ 2 + (D.2 - B.2) ^ 2)) = sqrt 3 :=
sorry

end length_of_segment_BD_is_sqrt_3_l121_12102


namespace cube_edge_length_and_volume_l121_12125

variable (edge_length : ℕ)

def cube_edge_total_length (edge_length : ℕ) : ℕ := edge_length * 12
def cube_volume (edge_length : ℕ) : ℕ := edge_length * edge_length * edge_length

theorem cube_edge_length_and_volume (h : cube_edge_total_length edge_length = 96) :
  edge_length = 8 ∧ cube_volume edge_length = 512 :=
by
  sorry

end cube_edge_length_and_volume_l121_12125


namespace change_in_mean_l121_12144

theorem change_in_mean {a b c d : ℝ} 
  (h1 : (a + b + c + d) / 4 = 10)
  (h2 : (b + c + d) / 3 = 11)
  (h3 : (a + c + d) / 3 = 12)
  (h4 : (a + b + d) / 3 = 13) : 
  ((a + b + c) / 3) = 4 := by 
  sorry

end change_in_mean_l121_12144


namespace area_of_triangles_l121_12176

theorem area_of_triangles
  (ABC_area : ℝ)
  (AD : ℝ)
  (DB : ℝ)
  (h_AD_DB : AD + DB = 7)
  (h_equal_areas : ABC_area = 12) :
  (∃ ABE_area : ℝ, ABE_area = 36 / 7) ∧ (∃ DBF_area : ℝ, DBF_area = 36 / 7) :=
by
  sorry

end area_of_triangles_l121_12176


namespace side_length_square_correct_l121_12117

noncomputable def side_length_square (time_seconds : ℕ) (speed_kmph : ℕ) : ℕ := sorry

theorem side_length_square_correct (time_seconds : ℕ) (speed_kmph : ℕ) (h_time : time_seconds = 24) 
  (h_speed : speed_kmph = 12) : side_length_square time_seconds speed_kmph = 20 :=
sorry

end side_length_square_correct_l121_12117


namespace krystiana_earnings_l121_12120

def earning_building1_first_floor : ℝ := 5 * 15 * 0.8
def earning_building1_second_floor : ℝ := 6 * 25 * 0.75
def earning_building1_third_floor : ℝ := 9 * 30 * 0.5
def earning_building1_fourth_floor : ℝ := 4 * 60 * 0.85
def earnings_building1 : ℝ := earning_building1_first_floor + earning_building1_second_floor + earning_building1_third_floor + earning_building1_fourth_floor

def earning_building2_first_floor : ℝ := 7 * 20 * 0.9
def earning_building2_second_floor : ℝ := (25 + 30 + 35 + 40 + 45 + 50 + 55 + 60) * 0.7
def earning_building2_third_floor : ℝ := 6 * 60 * 0.6
def earnings_building2 : ℝ := earning_building2_first_floor + earning_building2_second_floor + earning_building2_third_floor

def total_earnings : ℝ := earnings_building1 + earnings_building2

theorem krystiana_earnings : total_earnings = 1091.5 := by
  sorry

end krystiana_earnings_l121_12120


namespace sarah_bought_new_shirts_l121_12138

-- Define the given conditions
def original_shirts : ℕ := 9
def total_shirts : ℕ := 17

-- The proof statement: Prove that the number of new shirts is 8
theorem sarah_bought_new_shirts : total_shirts - original_shirts = 8 := by
  sorry

end sarah_bought_new_shirts_l121_12138


namespace similar_triangle_shortest_side_l121_12104

theorem similar_triangle_shortest_side (a b c: ℝ) (d e f: ℝ) :
  a = 21 ∧ b = 20 ∧ c = 29 ∧ d = 87 ∧ c^2 = a^2 + b^2 ∧ d / c = 3 → e = 60 :=
by
  sorry

end similar_triangle_shortest_side_l121_12104


namespace find_sum_of_a_and_b_l121_12151

theorem find_sum_of_a_and_b (a b : ℝ) (h1 : 0.005 * a = 0.65) (h2 : 0.0125 * b = 1.04) : a + b = 213.2 :=
  sorry

end find_sum_of_a_and_b_l121_12151


namespace solve_system_l121_12161

theorem solve_system : ∃ x y : ℝ, 2 * x - y = 3 ∧ 3 * x + 2 * y = 8 ∧ x = 2 ∧ y = 1 :=
by
  sorry

end solve_system_l121_12161


namespace cost_to_paint_cube_l121_12183

def side_length := 30 -- in feet
def cost_per_kg := 40 -- Rs. per kg
def coverage_per_kg := 20 -- sq. ft. per kg

def area_of_one_face := side_length * side_length
def total_surface_area := 6 * area_of_one_face
def paint_required := total_surface_area / coverage_per_kg
def total_cost := paint_required * cost_per_kg

theorem cost_to_paint_cube : total_cost = 10800 := 
by
  -- proof here would follow the solution steps provided in the solution part, which are omitted
  sorry

end cost_to_paint_cube_l121_12183


namespace find_f_2017_l121_12130

noncomputable def f : ℝ → ℝ := sorry

axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom f_period : ∀ x : ℝ, f (x + 2) = -f x
axiom f_neg1 : f (-1) = -3

theorem find_f_2017 : f 2017 = 3 := 
by
  sorry

end find_f_2017_l121_12130


namespace rectangle_area_ratio_l121_12147

theorem rectangle_area_ratio (a b c d : ℝ) 
  (h1 : a / c = 3 / 4) (h2 : b / d = 3 / 4) :
  (a * b) / (c * d) = 9 / 16 := 
  sorry

end rectangle_area_ratio_l121_12147


namespace positive_correlation_not_proportional_l121_12123

/-- Two quantities x and y depend on each other, and when one increases, the other also increases.
    This general relationship is denoted as a function g such that for any x₁, x₂,
    if x₁ < x₂ then g(x₁) < g(x₂). This implies a positive correlation but not necessarily proportionality. 
    We will prove that this does not imply a proportional relationship (y = kx). -/
theorem positive_correlation_not_proportional (g : ℝ → ℝ) 
(h_increasing: ∀ x₁ x₂ : ℝ, x₁ < x₂ → g x₁ < g x₂) :
¬ ∃ k : ℝ, ∀ x : ℝ, g x = k * x :=
sorry

end positive_correlation_not_proportional_l121_12123


namespace find_c_l121_12192

-- Define the function f(x)
def f (x c : ℝ) : ℝ := x * (x - c) ^ 2

-- Define the first derivative of f(x)
def f_prime (x c : ℝ) : ℝ := 3 * x ^ 2 - 4 * c * x + c ^ 2

-- Define the condition that f(x) has a local maximum at x = 2
def is_local_max (f' : ℝ → ℝ) (x0 : ℝ) : Prop :=
  f' x0 = 0 ∧ (∀ x, x < x0 → f' x > 0) ∧ (∀ x, x > x0 → f' x < 0)

-- The main theorem stating the equivalent proof problem
theorem find_c (c : ℝ) : is_local_max (f_prime 2) 2 → c = 6 := 
  sorry

end find_c_l121_12192


namespace min_value_of_expr_l121_12145

-- Define the expression
def expr (x y : ℝ) : ℝ := (x * y + 1)^2 + (x - y)^2

-- Statement to prove that the minimum value of the expression is 1
theorem min_value_of_expr : ∃ x y : ℝ, expr x y = 1 ∧ ∀ a b : ℝ, expr a b ≥ 1 :=
by
  -- Here the proof would be provided, but we leave it as sorry as per instructions.
  sorry

end min_value_of_expr_l121_12145


namespace problem_statement_l121_12142

theorem problem_statement (a b : ℝ) (h : a + b = 1) : 
  ((∀ (a b : ℝ), a + b = 1 → ab ≤ 1/4) ∧ 
   (∀ (a b : ℝ), ¬(ab ≤ 1/4) → ¬(a + b = 1)) ∧ 
   ¬(∀ (a b : ℝ), ab ≤ 1/4 → a + b = 1) ∧ 
   ¬(∀ (a b : ℝ), ¬(a + b = 1) → ¬(ab ≤ 1/4))) := 
sorry

end problem_statement_l121_12142


namespace square_of_leg_l121_12157

theorem square_of_leg (a c b : ℝ) (h1 : c = 2 * a + 1) (h2 : a^2 + b^2 = c^2) : b^2 = 3 * a^2 + 4 * a + 1 :=
by
  sorry

end square_of_leg_l121_12157
