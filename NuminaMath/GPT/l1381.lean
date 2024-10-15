import Mathlib

namespace NUMINAMATH_GPT_union_complements_eq_l1381_138116

-- Definitions as per conditions
def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 1, 2, 3}
def B : Set ℕ := {2, 3, 4}

-- Define complements
def C_UA : Set ℕ := U \ A
def C_UB : Set ℕ := U \ B

-- The proof statement
theorem union_complements_eq :
  (C_UA ∪ C_UB) = {0, 1, 4} :=
by
  sorry

end NUMINAMATH_GPT_union_complements_eq_l1381_138116


namespace NUMINAMATH_GPT_part1_part2_l1381_138188

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := Real.exp x * (Real.log x + k)
noncomputable def f_prime (x : ℝ) (k : ℝ) : ℝ := Real.exp x * (Real.log x + k) + Real.exp x / x
noncomputable def g (x : ℝ) (k : ℝ) : ℝ := f_prime x k - 2 * (f x k + Real.exp x)
noncomputable def phi (x : ℝ) : ℝ := Real.exp x / x

theorem part1 (h : f_prime 1 k = 0) : k = -1 := sorry

theorem part2 (t : ℝ) (h_g_le_phi : ∀ x > 0, g x (-1) ≤ t * phi x) : t ≥ 1 + 1 / Real.exp 2 := sorry

end NUMINAMATH_GPT_part1_part2_l1381_138188


namespace NUMINAMATH_GPT_workers_task_solution_l1381_138140

-- Defining the variables for the number of days worked by A and B
variables (x y : ℕ)

-- Defining the total earnings for A and B
def total_earnings_A := 30
def total_earnings_B := 14

-- Condition: B worked 3 days less than A
def condition1 := y = x - 3

-- Daily wages of A and B
def daily_wage_A := total_earnings_A / x
def daily_wage_B := total_earnings_B / y

-- New scenario conditions
def new_days_A := x - 2
def new_days_B := y + 5

-- New total earnings in the scenario where they work changed days
def new_earnings_A := new_days_A * daily_wage_A
def new_earnings_B := new_days_B * daily_wage_B

-- Final proof to show the number of days worked and daily wages satisfying the conditions
theorem workers_task_solution 
  (h1 : y = x - 3)
  (h2 : new_earnings_A = new_earnings_B) 
  (hx : x = 10)
  (hy : y = 7) 
  (wageA : daily_wage_A = 3) 
  (wageB : daily_wage_B = 2) : 
  x = 10 ∧ y = 7 ∧ daily_wage_A = 3 ∧ daily_wage_B = 2 :=
by {
  sorry  -- Proof is skipped as instructed
}

end NUMINAMATH_GPT_workers_task_solution_l1381_138140


namespace NUMINAMATH_GPT_smallest_altitude_leq_three_l1381_138166

theorem smallest_altitude_leq_three (a b c : ℝ) (r : ℝ) 
  (ha : a = max a (max b c)) 
  (r_eq : r = 1) 
  (area_eq : ∀ (S : ℝ), S = (a + b + c) / 2 ∧ S = a * h / 2) :
  ∃ h : ℝ, h ≤ 3 :=
by
  sorry

end NUMINAMATH_GPT_smallest_altitude_leq_three_l1381_138166


namespace NUMINAMATH_GPT_simplify_product_l1381_138101

theorem simplify_product (a : ℝ) : (1 * 2 * a * 3 * a^2 * 4 * a^3 * 5 * a^4) = 120 * a^10 := by
  sorry

end NUMINAMATH_GPT_simplify_product_l1381_138101


namespace NUMINAMATH_GPT_vasya_made_a_mistake_l1381_138191

theorem vasya_made_a_mistake :
  ∀ x : ℝ, x^4 - 3*x^3 - 2*x^2 - 4*x + 1 = 0 → ¬ x < 0 :=
by sorry

end NUMINAMATH_GPT_vasya_made_a_mistake_l1381_138191


namespace NUMINAMATH_GPT_min_value_f_l1381_138190

def f (x : ℝ) : ℝ := 5 * x^2 + 10 * x + 20

theorem min_value_f : ∃ (x : ℝ), f x = 15 :=
by
  sorry

end NUMINAMATH_GPT_min_value_f_l1381_138190


namespace NUMINAMATH_GPT_average_age_of_9_l1381_138164

theorem average_age_of_9 : 
  ∀ (avg_20 avg_5 age_15 : ℝ),
  avg_20 = 15 →
  avg_5 = 14 →
  age_15 = 86 →
  (9 * (69/9)) = 7.67 :=
by
  intros avg_20 avg_5 age_15 avg_20_val avg_5_val age_15_val
  -- The proof is skipped
  sorry

end NUMINAMATH_GPT_average_age_of_9_l1381_138164


namespace NUMINAMATH_GPT_crates_sold_on_monday_l1381_138150

variable (M : ℕ)
variable (h : M + 2 * M + (2 * M - 2) + M = 28)

theorem crates_sold_on_monday : M = 5 :=
by
  sorry

end NUMINAMATH_GPT_crates_sold_on_monday_l1381_138150


namespace NUMINAMATH_GPT_arith_seq_100th_term_l1381_138196

noncomputable def arithSeq (a : ℤ) (n : ℕ) : ℤ :=
  a - 1 + (n - 1) * ((a + 1) - (a - 1))

theorem arith_seq_100th_term (a : ℤ) : arithSeq a 100 = 197 := by
  sorry

end NUMINAMATH_GPT_arith_seq_100th_term_l1381_138196


namespace NUMINAMATH_GPT_count_four_digit_numbers_divisible_by_17_and_end_in_17_l1381_138110

theorem count_four_digit_numbers_divisible_by_17_and_end_in_17 :
  ∃ S : Finset ℕ, S.card = 5 ∧ ∀ n ∈ S, 1000 ≤ n ∧ n < 10000 ∧ n % 17 = 0 ∧ n % 100 = 17 :=
by
  sorry

end NUMINAMATH_GPT_count_four_digit_numbers_divisible_by_17_and_end_in_17_l1381_138110


namespace NUMINAMATH_GPT_smallest_four_digit_div_by_53_l1381_138118

theorem smallest_four_digit_div_by_53 : ∃ n : ℕ, n % 53 = 0 ∧ 1000 ≤ n ∧ n ≤ 9999 ∧ ∀ m : ℕ, (m % 53 = 0 ∧ 1000 ≤ m ∧ m ≤ 9999) → n ≤ m ∧ n = 1007 :=
sorry

end NUMINAMATH_GPT_smallest_four_digit_div_by_53_l1381_138118


namespace NUMINAMATH_GPT_maximum_area_of_triangle_ABQ_l1381_138185

open Real

structure Point3D where
  x : ℝ
  y : ℝ

def circle_C (Q : Point3D) : Prop := (Q.x - 3)^2 + (Q.y - 4)^2 = 4

def A := Point3D.mk 1 0
def B := Point3D.mk (-1) 0

noncomputable def area_triangle (P Q R : Point3D) : ℝ :=
  (1 / 2) * abs ((P.x * (Q.y - R.y)) + (Q.x * (R.y - P.y)) + (R.x * (P.y - Q.y)))

theorem maximum_area_of_triangle_ABQ : ∀ (Q : Point3D), circle_C Q → area_triangle A B Q ≤ 6 := by
  sorry

end NUMINAMATH_GPT_maximum_area_of_triangle_ABQ_l1381_138185


namespace NUMINAMATH_GPT_fraction_of_students_with_buddy_l1381_138144

variable (s n : ℕ)

theorem fraction_of_students_with_buddy (h : n = 4 * s / 3) : 
  (n / 4 + s / 3) / (n + s) = 2 / 7 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_students_with_buddy_l1381_138144


namespace NUMINAMATH_GPT_expression_meaningful_if_not_three_l1381_138199

-- Definition of meaningful expression
def meaningful_expr (x : ℝ) : Prop := (x ≠ 3)

theorem expression_meaningful_if_not_three (x : ℝ) :
  (∃ y : ℝ, y = 1 / (x - 3)) ↔ meaningful_expr x := by
  sorry

end NUMINAMATH_GPT_expression_meaningful_if_not_three_l1381_138199


namespace NUMINAMATH_GPT_triangle_perimeter_l1381_138182

-- Define the ratios
def ratio1 : ℚ := 1 / 2
def ratio2 : ℚ := 1 / 3
def ratio3 : ℚ := 1 / 4

-- Define the longest side
def longest_side : ℚ := 48

-- Compute the perimeter given the conditions
theorem triangle_perimeter (ratio1 ratio2 ratio3 : ℚ) (longest_side : ℚ) 
  (h_ratio1 : ratio1 = 1 / 2) (h_ratio2 : ratio2 = 1 / 3) (h_ratio3 : ratio3 = 1 / 4)
  (h_longest_side : longest_side = 48) : 
  (longest_side * 6/ (ratio1 * 12 + ratio2 * 12 + ratio3 * 12)) = 104 := by
  sorry

end NUMINAMATH_GPT_triangle_perimeter_l1381_138182


namespace NUMINAMATH_GPT_face_opposite_A_is_F_l1381_138156

structure Cube where
  adjacency : String → String → Prop
  exists_face : ∃ a b c d e f : String, True

variable 
  (C : Cube)
  (adjA_B : C.adjacency "A" "B")
  (adjA_C : C.adjacency "A" "C")
  (adjB_D : C.adjacency "B" "D")

theorem face_opposite_A_is_F : 
  ∃ f : String, f = "F" ∧ ∀ g : String, (C.adjacency "A" g → g ≠ "F") :=
by 
  sorry

end NUMINAMATH_GPT_face_opposite_A_is_F_l1381_138156


namespace NUMINAMATH_GPT_blocks_left_l1381_138149

/-- Problem: Randy has 78 blocks. He uses 19 blocks to build a tower. Prove that he has 59 blocks left. -/
theorem blocks_left (initial_blocks : ℕ) (used_blocks : ℕ) (remaining_blocks : ℕ) : initial_blocks = 78 → used_blocks = 19 → remaining_blocks = initial_blocks - used_blocks → remaining_blocks = 59 :=
by
  sorry

end NUMINAMATH_GPT_blocks_left_l1381_138149


namespace NUMINAMATH_GPT_three_numbers_sum_div_by_three_l1381_138194

theorem three_numbers_sum_div_by_three (s : Fin 7 → ℕ) : 
  ∃ (a b c : Fin 7), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ (s a + s b + s c) % 3 = 0 := 
sorry

end NUMINAMATH_GPT_three_numbers_sum_div_by_three_l1381_138194


namespace NUMINAMATH_GPT_mean_median_sum_is_11_l1381_138165

theorem mean_median_sum_is_11 (m n : ℕ) (h1 : m + 5 < n)
  (h2 : (m + (m + 3) + (m + 5) + n + (n + 1) + (2 * n - 1)) / 6 = n)
  (h3 : (m + 5 + n) / 2 = n) : m + n = 11 := by
  sorry

end NUMINAMATH_GPT_mean_median_sum_is_11_l1381_138165


namespace NUMINAMATH_GPT_keith_spent_on_cards_l1381_138136

theorem keith_spent_on_cards :
  let digimon_card_cost := 4.45
  let num_digimon_packs := 4
  let baseball_card_cost := 6.06
  let total_spent := num_digimon_packs * digimon_card_cost + baseball_card_cost
  total_spent = 23.86 :=
by
  sorry

end NUMINAMATH_GPT_keith_spent_on_cards_l1381_138136


namespace NUMINAMATH_GPT_fraction_of_garden_occupied_by_flowerbeds_is_correct_l1381_138103

noncomputable def garden_fraction_occupied : ℚ :=
  let garden_length := 28
  let garden_shorter_length := 18
  let triangle_leg := (garden_length - garden_shorter_length) / 2
  let triangle_area := 1 / 2 * triangle_leg^2
  let flowerbeds_area := 2 * triangle_area
  let garden_width : ℚ := 5  -- Assuming the height of the trapezoid as part of the garden rest
  let garden_area := garden_length * garden_width
  flowerbeds_area / garden_area

theorem fraction_of_garden_occupied_by_flowerbeds_is_correct :
  garden_fraction_occupied = 5 / 28 := by
  sorry

end NUMINAMATH_GPT_fraction_of_garden_occupied_by_flowerbeds_is_correct_l1381_138103


namespace NUMINAMATH_GPT_janet_practiced_days_l1381_138114

theorem janet_practiced_days (total_miles : ℕ) (miles_per_day : ℕ) (days_practiced : ℕ) :
  total_miles = 72 ∧ miles_per_day = 8 → days_practiced = total_miles / miles_per_day → days_practiced = 9 :=
by
  sorry

end NUMINAMATH_GPT_janet_practiced_days_l1381_138114


namespace NUMINAMATH_GPT_machines_needed_l1381_138155

theorem machines_needed (x Y : ℝ) (R : ℝ) :
  (4 * R * 6 = x) → (M * R * 6 = Y) → M = 4 * Y / x :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_machines_needed_l1381_138155


namespace NUMINAMATH_GPT_interest_rate_l1381_138187

-- Define the given conditions
def principal : ℝ := 4000
def total_interest : ℝ := 630.50
def future_value : ℝ := principal + total_interest
def time : ℝ := 1.5  -- 1 1/2 years
def times_compounded : ℝ := 2  -- Compounded half yearly

-- Statement to prove the annual interest rate
theorem interest_rate (P A t n : ℝ) (hP : P = principal) (hA : A = future_value) 
    (ht : t = time) (hn : n = times_compounded) :
    ∃ r : ℝ, A = P * (1 + r / n) ^ (n * t) ∧ r = 0.1 := 
by 
  sorry

end NUMINAMATH_GPT_interest_rate_l1381_138187


namespace NUMINAMATH_GPT_adam_tickets_left_l1381_138195

def tickets_left (total_tickets : ℕ) (ticket_cost : ℕ) (total_spent : ℕ) : ℕ :=
  total_tickets - total_spent / ticket_cost

theorem adam_tickets_left :
  tickets_left 13 9 81 = 4 := 
by
  sorry

end NUMINAMATH_GPT_adam_tickets_left_l1381_138195


namespace NUMINAMATH_GPT_last_card_in_box_l1381_138127

-- Define the zigzag pattern
def card_position (n : Nat) : Nat :=
  let cycle_pos := n % 12
  if cycle_pos = 0 then
    12
  else
    cycle_pos

def box_for_card (pos : Nat) : Nat :=
  if pos ≤ 7 then
    pos
  else
    14 - pos

theorem last_card_in_box : box_for_card (card_position 2015) = 3 := by
  sorry

end NUMINAMATH_GPT_last_card_in_box_l1381_138127


namespace NUMINAMATH_GPT_min_value_of_function_l1381_138100

noncomputable def f (x y : ℝ) : ℝ := x^2 / (x + 2) + y^2 / (y + 1)

theorem min_value_of_function (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 1) : 
  f x y ≥ 1 / 4 :=
sorry

end NUMINAMATH_GPT_min_value_of_function_l1381_138100


namespace NUMINAMATH_GPT_solve_for_x_l1381_138146

theorem solve_for_x (x : ℚ) (h : (2 * x + 18) / (x - 6) = (2 * x - 4) / (x + 10)) : x = -26 / 9 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l1381_138146


namespace NUMINAMATH_GPT_geometric_sequence_a5_l1381_138183

theorem geometric_sequence_a5 (a : ℕ → ℝ) (q : ℝ) 
  (h1 : a 3 = 6)
  (h2 : a 3 + a 5 + a 7 = 78)
  (h_geom : ∀ n, a (n + 1) = a n * q) : 
  a 5 = 18 :=
by sorry

end NUMINAMATH_GPT_geometric_sequence_a5_l1381_138183


namespace NUMINAMATH_GPT_area_of_trapezium_l1381_138177

def length_parallel_side1 : ℝ := 20
def length_parallel_side2 : ℝ := 18
def distance_between_sides : ℝ := 15
def expected_area : ℝ := 285

theorem area_of_trapezium :
  (1 / 2) * (length_parallel_side1 + length_parallel_side2) * distance_between_sides = expected_area :=
by sorry

end NUMINAMATH_GPT_area_of_trapezium_l1381_138177


namespace NUMINAMATH_GPT_sqrt_expression_l1381_138126

theorem sqrt_expression : Real.sqrt ((4^2) * (5^6)) = 500 := by
  sorry

end NUMINAMATH_GPT_sqrt_expression_l1381_138126


namespace NUMINAMATH_GPT_range_of_a_l1381_138189

theorem range_of_a (a : ℝ) (h : ¬ (1^2 - 2*1 + a > 0)) : 1 ≤ a := sorry

end NUMINAMATH_GPT_range_of_a_l1381_138189


namespace NUMINAMATH_GPT_brenda_has_8_dollars_l1381_138168

-- Define the amounts of money each friend has
def emma_money : ℕ := 8
def daya_money : ℕ := emma_money + (25 * emma_money / 100) -- 25% more than Emma's money
def jeff_money : ℕ := 2 * daya_money / 5 -- Jeff has 2/5 of Daya's money
def brenda_money : ℕ := jeff_money + 4 -- Brenda has 4 more dollars than Jeff

-- The theorem stating the final question
theorem brenda_has_8_dollars : brenda_money = 8 :=
by
  sorry

end NUMINAMATH_GPT_brenda_has_8_dollars_l1381_138168


namespace NUMINAMATH_GPT_flour_needed_for_dozen_cookies_l1381_138129

/--
Matt uses 4 bags of flour, each weighing 5 pounds, to make a total of 120 cookies.
Prove that 2 pounds of flour are needed to make a dozen cookies.
-/
theorem flour_needed_for_dozen_cookies :
  ∀ (bags_of_flour : ℕ) (weight_per_bag : ℕ) (total_cookies : ℕ),
  bags_of_flour = 4 →
  weight_per_bag = 5 →
  total_cookies = 120 →
  (12 * (bags_of_flour * weight_per_bag)) / total_cookies = 2 :=
by
  sorry

end NUMINAMATH_GPT_flour_needed_for_dozen_cookies_l1381_138129


namespace NUMINAMATH_GPT_arithmetic_sequence_fourth_term_l1381_138138

-- Define the arithmetic sequence and conditions
variable {a : ℕ → ℕ}
variable {S : ℕ → ℕ}

-- Given conditions
def a₂ := 606
def S₄ := 3834

-- Problem statement
theorem arithmetic_sequence_fourth_term :
  (a 1 + a 2 + a 3 = 1818) →
  (a 4 = 2016) :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_fourth_term_l1381_138138


namespace NUMINAMATH_GPT_fraction_of_board_shaded_is_one_fourth_l1381_138174

def totalArea : ℕ := 16
def shadedTopLeft : ℕ := 4
def shadedBottomRight : ℕ := 4
def fractionShaded (totalArea shadedTopLeft shadedBottomRight : ℕ) : ℚ :=
  (shadedTopLeft + shadedBottomRight) / totalArea

theorem fraction_of_board_shaded_is_one_fourth :
  fractionShaded totalArea shadedTopLeft shadedBottomRight = 1 / 4 := by
  sorry

end NUMINAMATH_GPT_fraction_of_board_shaded_is_one_fourth_l1381_138174


namespace NUMINAMATH_GPT_F_8_not_true_F_6_might_be_true_l1381_138161

variable {n : ℕ}

-- Declare the proposition F
variable (F : ℕ → Prop)

-- Placeholder conditions
axiom condition1 : ¬ F 7
axiom condition2 : ∀ k : ℕ, k > 0 → (F k → F (k + 1))

-- Proof statements
theorem F_8_not_true : ¬ F 8 :=
by {
  sorry
}

theorem F_6_might_be_true : ¬ ¬ F 6 :=
by {
  sorry
}

end NUMINAMATH_GPT_F_8_not_true_F_6_might_be_true_l1381_138161


namespace NUMINAMATH_GPT_average_speed_uphill_l1381_138198

theorem average_speed_uphill (d : ℝ) (v : ℝ) :
  (2 * d) / ((d / v) + (d / 100)) = 9.523809523809524 → v = 5 :=
by
  intro h1
  sorry

end NUMINAMATH_GPT_average_speed_uphill_l1381_138198


namespace NUMINAMATH_GPT_find_x_condition_l1381_138115

theorem find_x_condition :
  ∀ x : ℝ, (x^2 - 1) / (x + 1) = 0 → x = 1 := by
  intros x h
  have num_zero : x^2 - 1 = 0 := by
    -- Proof that the numerator is zero
    sorry
  have denom_nonzero : x ≠ -1 := by
    -- Proof that the denominator is non-zero
    sorry
  have x_solves : x = 1 := by
    -- Final proof to show x = 1
    sorry
  exact x_solves

end NUMINAMATH_GPT_find_x_condition_l1381_138115


namespace NUMINAMATH_GPT_smallest_integer_solution_l1381_138112

theorem smallest_integer_solution (n : ℤ) (h : n^3 - 12 * n^2 + 44 * n - 48 ≤ 0) : n = 2 :=
sorry

end NUMINAMATH_GPT_smallest_integer_solution_l1381_138112


namespace NUMINAMATH_GPT_eyes_that_saw_the_plane_l1381_138111

theorem eyes_that_saw_the_plane (students : ℕ) (ratio : ℚ) (eyes_per_student : ℕ) 
  (h1 : students = 200) (h2 : ratio = 3 / 4) (h3 : eyes_per_student = 2) : 
  2 * (ratio * students) = 300 := 
by 
  -- the proof is omitted
  sorry

end NUMINAMATH_GPT_eyes_that_saw_the_plane_l1381_138111


namespace NUMINAMATH_GPT_sum_of_dihedral_angles_leq_90_l1381_138139
noncomputable section

-- Let θ1 and θ2 be angles formed by a line with two perpendicular planes
variable (θ1 θ2 : ℝ)

-- Define the condition stating the planes are perpendicular, and the line forms dihedral angles
def dihedral_angle_condition (θ1 θ2 : ℝ) : Prop := 
  θ1 ≥ 0 ∧ θ1 ≤ 90 ∧ θ2 ≥ 0 ∧ θ2 ≤ 90

-- The theorem statement capturing the problem
theorem sum_of_dihedral_angles_leq_90 
  (θ1 θ2 : ℝ) 
  (h : dihedral_angle_condition θ1 θ2) : 
  θ1 + θ2 ≤ 90 :=
sorry

end NUMINAMATH_GPT_sum_of_dihedral_angles_leq_90_l1381_138139


namespace NUMINAMATH_GPT_max_sum_arithmetic_prog_l1381_138193

theorem max_sum_arithmetic_prog (a d : ℝ) (S : ℕ → ℝ) 
  (h1 : S 3 = 327)
  (h2 : S 57 = 57)
  (hS : ∀ n, S n = (n / 2) * (2 * a + (n - 1) * d)) :
  ∃ max_S : ℝ, max_S = 1653 := by
  sorry

end NUMINAMATH_GPT_max_sum_arithmetic_prog_l1381_138193


namespace NUMINAMATH_GPT_race_distance_l1381_138192

theorem race_distance (T_A T_B : ℝ) (D : ℝ) (V_A V_B : ℝ)
  (h1 : T_A = 23)
  (h2 : T_B = 30)
  (h3 : V_A = D / 23)
  (h4 : V_B = (D - 56) / 30)
  (h5 : D = (D - 56) * (23 / 30) + 56) :
  D = 56 :=
by
  sorry

end NUMINAMATH_GPT_race_distance_l1381_138192


namespace NUMINAMATH_GPT_find_a_of_inequality_solution_l1381_138147

theorem find_a_of_inequality_solution (a : ℝ) :
  (∀ x : ℝ, 0 < x ∧ x < 1 ↔ x^2 - a * x < 0) → a = 1 := 
by 
  sorry

end NUMINAMATH_GPT_find_a_of_inequality_solution_l1381_138147


namespace NUMINAMATH_GPT_polynomial_coeff_sum_l1381_138117

theorem polynomial_coeff_sum (a_4 a_3 a_2 a_1 a_0 : ℝ) :
  (∀ x: ℝ, (x - 1) ^ 4 = a_4 * x^4 + a_3 * x^3 + a_2 * x^2 + a_1 * x + a_0) →
  a_4 - a_3 + a_2 - a_1 + a_0 = 16 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_polynomial_coeff_sum_l1381_138117


namespace NUMINAMATH_GPT_jill_more_than_jake_l1381_138158

-- Definitions from conditions
def jill_peaches := 12
def steven_peaches := jill_peaches + 15
def jake_peaches := steven_peaches - 16

-- Theorem to prove the question == answer given conditions
theorem jill_more_than_jake : jill_peaches - jake_peaches = 1 :=
by
  -- Proof steps would be here, but for the statement requirement we put sorry
  sorry

end NUMINAMATH_GPT_jill_more_than_jake_l1381_138158


namespace NUMINAMATH_GPT_functional_equation_solution_exists_l1381_138122

theorem functional_equation_solution_exists (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (f x + y) = 2 * x + f (f y - x)) →
  ∃ c : ℝ, ∀ x : ℝ, f x = x + c :=
by
  intro h
  sorry

end NUMINAMATH_GPT_functional_equation_solution_exists_l1381_138122


namespace NUMINAMATH_GPT_hyperbola_standard_eq_proof_l1381_138106

noncomputable def real_axis_length := 6
noncomputable def asymptote_slope := 3 / 2

def hyperbola_standard_eq (a b : ℝ) :=
  ∀ x y : ℝ, (y^2 / a^2 - x^2 / b^2 = 1)

theorem hyperbola_standard_eq_proof (a b : ℝ) 
  (h_a : 2 * a = real_axis_length)
  (h_b : a / b = asymptote_slope) :
  hyperbola_standard_eq 3 2 := 
by
  sorry

end NUMINAMATH_GPT_hyperbola_standard_eq_proof_l1381_138106


namespace NUMINAMATH_GPT_set_intersection_l1381_138135

def A := {x : ℝ | x^2 - 3*x ≥ 0}
def B := {x : ℝ | x < 1}
def intersection := {x : ℝ | x ≤ 0}

theorem set_intersection : A ∩ B = intersection :=
  sorry

end NUMINAMATH_GPT_set_intersection_l1381_138135


namespace NUMINAMATH_GPT_geometric_sequence_common_ratio_l1381_138159

theorem geometric_sequence_common_ratio
  (a_n : ℕ → ℝ)
  (q : ℝ)
  (h1 : a_n 3 = 7)
  (h2 : a_n 1 + a_n 2 + a_n 3 = 21) :
  q = 1 ∨ q = -1 / 2 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_common_ratio_l1381_138159


namespace NUMINAMATH_GPT_polynomial_sum_l1381_138107

def p (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def q (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def r (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2

theorem polynomial_sum (x : ℝ) : p x + q x + r x = -4 * x^2 + 12 * x - 12 := by
  sorry

end NUMINAMATH_GPT_polynomial_sum_l1381_138107


namespace NUMINAMATH_GPT_angle_inclusion_l1381_138125

-- Defining the sets based on the given conditions
def M : Set ℝ := { x | 0 < x ∧ x ≤ 90 }
def N : Set ℝ := { x | 0 < x ∧ x < 90 }
def P : Set ℝ := { x | 0 ≤ x ∧ x ≤ 90 }

-- The proof statement
theorem angle_inclusion : N ⊆ M ∧ M ⊆ P :=
by
  sorry

end NUMINAMATH_GPT_angle_inclusion_l1381_138125


namespace NUMINAMATH_GPT_total_artworks_l1381_138113

theorem total_artworks (students : ℕ) (group1_artworks : ℕ) (group2_artworks : ℕ) (total_students : students = 10) 
    (artwork_group1 : group1_artworks = 5 * 3) (artwork_group2 : group2_artworks = 5 * 4) : 
    group1_artworks + group2_artworks = 35 :=
by
  sorry

end NUMINAMATH_GPT_total_artworks_l1381_138113


namespace NUMINAMATH_GPT_typing_cost_equation_l1381_138181

def typing_cost (x : ℝ) : ℝ :=
  200 * x + 80 * 3 + 20 * 6

theorem typing_cost_equation (x : ℝ) (h : typing_cost x = 1360) : x = 5 :=
by
  sorry

end NUMINAMATH_GPT_typing_cost_equation_l1381_138181


namespace NUMINAMATH_GPT_sqrt_neg9_sq_l1381_138197

theorem sqrt_neg9_sq : Real.sqrt ((-9 : Real)^2) = 9 := 
by 
  sorry

end NUMINAMATH_GPT_sqrt_neg9_sq_l1381_138197


namespace NUMINAMATH_GPT_area_of_section_ABD_l1381_138171
-- Import everything from the Mathlib library

-- Define the conditions
def is_equilateral_triangle (a b c : ℝ) (ABC_angle : ℝ) : Prop := 
  a = b ∧ b = c ∧ ABC_angle = 60

def plane_angle (angle : ℝ) : Prop := 
  angle = 35 + 18/60

def volume_of_truncated_pyramid (volume : ℝ) : Prop := 
  volume = 15

-- The main theorem based on the above conditions
theorem area_of_section_ABD
  (a b c ABC_angle : ℝ)
  (S : ℝ)
  (V : ℝ)
  (h1 : is_equilateral_triangle a b c ABC_angle)
  (h2 : plane_angle S)
  (h3 : volume_of_truncated_pyramid V) :
  ∃ (area : ℝ), area = 16.25 :=
by
  -- skipping the proof
  sorry

end NUMINAMATH_GPT_area_of_section_ABD_l1381_138171


namespace NUMINAMATH_GPT_sum_of_three_numbers_l1381_138141

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 138) 
  (h2 : ab + bc + ca = 131) : 
  a + b + c = 20 := 
by sorry

end NUMINAMATH_GPT_sum_of_three_numbers_l1381_138141


namespace NUMINAMATH_GPT_red_ball_prob_gt_black_ball_prob_l1381_138121

theorem red_ball_prob_gt_black_ball_prob (m : ℕ) (h : 8 > m) : m ≠ 10 :=
by
  sorry

end NUMINAMATH_GPT_red_ball_prob_gt_black_ball_prob_l1381_138121


namespace NUMINAMATH_GPT_remainder_is_four_l1381_138104

def least_number : Nat := 174

theorem remainder_is_four (n : Nat) (m₁ m₂ : Nat) (h₁ : n = least_number / m₁ * m₁ + 4) 
(h₂ : n = least_number / m₂ * m₂ + 4) (h₃ : m₁ = 34) (h₄ : m₂ = 5) : 
  n % m₁ = 4 ∧ n % m₂ = 4 := 
by
  sorry

end NUMINAMATH_GPT_remainder_is_four_l1381_138104


namespace NUMINAMATH_GPT_inequality_proof_l1381_138180

theorem inequality_proof (a b c : ℝ) (hp : 0 < a ∧ 0 < b ∧ 0 < c) (hd : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
    (bc / a + ac / b + ab / c > a + b + c) :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1381_138180


namespace NUMINAMATH_GPT_mark_egg_supply_in_a_week_l1381_138157

def dozen := 12
def eggs_per_day_store1 := 5 * dozen
def eggs_per_day_store2 := 30
def daily_eggs_supplied := eggs_per_day_store1 + eggs_per_day_store2
def days_per_week := 7

theorem mark_egg_supply_in_a_week : daily_eggs_supplied * days_per_week = 630 := by
  sorry

end NUMINAMATH_GPT_mark_egg_supply_in_a_week_l1381_138157


namespace NUMINAMATH_GPT_trapezoid_angles_l1381_138128

-- Definition of the problem statement in Lean 4
theorem trapezoid_angles (A B C D : ℝ) (h1 : A = 60) (h2 : B = 130)
  (h3 : A + D = 180) (h4 : B + C = 180) (h_sum : A + B + C + D = 360) :
  C = 50 ∧ D = 120 :=
by
  sorry

end NUMINAMATH_GPT_trapezoid_angles_l1381_138128


namespace NUMINAMATH_GPT_sequence_sum_n_eq_21_l1381_138186

theorem sequence_sum_n_eq_21 (a : ℕ → ℕ) (S : ℕ → ℕ) (n : ℕ) 
  (h1 : a 1 = 1)
  (h2 : ∀ k, a (k + 1) = a k + 1)
  (h3 : ∀ n, S n = (n * (n + 1)) / 2)
  (h4 : S n = 21) :
  n = 6 :=
sorry

end NUMINAMATH_GPT_sequence_sum_n_eq_21_l1381_138186


namespace NUMINAMATH_GPT_cheapest_book_price_l1381_138109

theorem cheapest_book_price
  (n : ℕ) (c : ℕ) (d : ℕ)
  (h1 : n = 40)
  (h2 : d = 3)
  (h3 : c + d * 19 = 75) :
  c = 18 :=
sorry

end NUMINAMATH_GPT_cheapest_book_price_l1381_138109


namespace NUMINAMATH_GPT_mixed_fractions_product_l1381_138137

theorem mixed_fractions_product :
  ∃ X Y : ℤ, (5 * X + 1) / X * (2 * Y + 1) / 2 = 43 ∧ X = 17 ∧ Y = 8 :=
by
  use 17, 8
  simp
  sorry

end NUMINAMATH_GPT_mixed_fractions_product_l1381_138137


namespace NUMINAMATH_GPT_jason_borrowed_amount_l1381_138134

theorem jason_borrowed_amount (hours cycles value_per_cycle remaining_hrs remaining_value total_value: ℕ) : 
  hours = 39 → cycles = (hours / 7) → value_per_cycle = 28 → remaining_hrs = (hours % 7) →
  remaining_value = (1 + 2 + 3 + 4) →
  total_value = (cycles * value_per_cycle + remaining_value) →
  total_value = 150 := 
by {
  sorry
}

end NUMINAMATH_GPT_jason_borrowed_amount_l1381_138134


namespace NUMINAMATH_GPT_mouse_grasshopper_diff_l1381_138162

def grasshopper_jump: ℕ := 19
def frog_jump: ℕ := grasshopper_jump + 10
def mouse_jump: ℕ := frog_jump + 20

theorem mouse_grasshopper_diff:
  (mouse_jump - grasshopper_jump) = 30 :=
by
  sorry

end NUMINAMATH_GPT_mouse_grasshopper_diff_l1381_138162


namespace NUMINAMATH_GPT_length_of_second_offset_l1381_138133

theorem length_of_second_offset (d₁ d₂ h₁ A : ℝ) (h_d₁ : d₁ = 30) (h_h₁ : h₁ = 9) (h_A : A = 225):
  ∃ h₂, (A = (1/2) * d₁ * h₁ + (1/2) * d₁ * h₂) → h₂ = 6 := by
  sorry

end NUMINAMATH_GPT_length_of_second_offset_l1381_138133


namespace NUMINAMATH_GPT_hyperbola_sum_l1381_138131

noncomputable def h : ℝ := 3
noncomputable def k : ℝ := -4
noncomputable def a : ℝ := 4
noncomputable def c : ℝ := Real.sqrt 53
noncomputable def b : ℝ := Real.sqrt (c^2 - a^2)

theorem hyperbola_sum : h + k + a + b = 3 + Real.sqrt 37 :=
by
  -- sorry is used to skip the proof as per the instruction
  sorry
  -- exact calc
  --   h + k + a + b = 3 + (-4) + 4 + Real.sqrt 37 : by simp
  --             ... = 3 + Real.sqrt 37 : by simp

end NUMINAMATH_GPT_hyperbola_sum_l1381_138131


namespace NUMINAMATH_GPT_find_num_students_l1381_138173

variables (N T : ℕ)
variables (h1 : T = N * 80)
variables (h2 : 5 * 20 = 100)
variables (h3 : (T - 100) / (N - 5) = 90)

theorem find_num_students (h1 : T = N * 80) (h3 : (T - 100) / (N - 5) = 90) : N = 35 :=
sorry

end NUMINAMATH_GPT_find_num_students_l1381_138173


namespace NUMINAMATH_GPT_measure_of_angle_A_range_of_b2_add_c2_div_a2_l1381_138163

variable {A B C a b c : ℝ}
variable {S : ℝ}

theorem measure_of_angle_A
  (h1 : S = 1 / 2 * b * c * Real.sin A)
  (h2 : 4 * Real.sqrt 3 * S = a ^ 2 - (b - c) ^ 2)
  (h3 : a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos A) : 
  A = 2 * Real.pi / 3 :=
by
  sorry

theorem range_of_b2_add_c2_div_a2
  (h1 : S = 1 / 2 * b * c * Real.sin A)
  (h2 : 4 * Real.sqrt 3 * S = a ^ 2 - (b - c) ^ 2)
  (h3 : A = 2 * Real.pi / 3) : 
  2 / 3 ≤ (b ^ 2 + c ^ 2) / a ^ 2 ∧ (b ^ 2 + c ^ 2) / a ^ 2 < 1 :=
by
  sorry

end NUMINAMATH_GPT_measure_of_angle_A_range_of_b2_add_c2_div_a2_l1381_138163


namespace NUMINAMATH_GPT_find_cylinder_radius_l1381_138130

-- Define the problem conditions
def cone_diameter := 10
def cone_altitude := 12
def cylinder_height_eq_diameter (r: ℚ) := 2 * r

-- Define the cone and cylinder inscribed properties
noncomputable def inscribed_cylinder_radius (r : ℚ) : Prop :=
  (cylinder_height_eq_diameter r) ≤ cone_altitude ∧
  2 * r ≤ cone_diameter ∧
  cone_altitude - cylinder_height_eq_diameter r = (cone_altitude * r) / (cone_diameter / 2)

-- The proof goal
theorem find_cylinder_radius : ∃ r : ℚ, inscribed_cylinder_radius r ∧ r = 30/11 :=
by
  sorry

end NUMINAMATH_GPT_find_cylinder_radius_l1381_138130


namespace NUMINAMATH_GPT_equivalent_expression_l1381_138154

theorem equivalent_expression :
  (2 + 3) * (2^2 + 3^2) * (2^4 + 3^4) * (2^8 + 3^8) * (2^16 + 3^16) * (2^32 + 3^32) * (2^64 + 3^64) = 3^128 - 2^128 :=
sorry

end NUMINAMATH_GPT_equivalent_expression_l1381_138154


namespace NUMINAMATH_GPT_total_marks_eq_300_second_candidate_percentage_l1381_138179

-- Defining the conditions
def percentage_marks (total_marks : ℕ) : ℕ := 40
def fail_by (fail_marks : ℕ) : ℕ := 40
def passing_marks : ℕ := 160

-- The number of total marks in the exam computed from conditions
theorem total_marks_eq_300 : ∃ T, 0.40 * T = 120 :=
by
  use 300
  sorry

-- The percentage of marks the second candidate gets
theorem second_candidate_percentage : ∃ percent, percent = (180 / 300) * 100 :=
by
  use 60
  sorry

end NUMINAMATH_GPT_total_marks_eq_300_second_candidate_percentage_l1381_138179


namespace NUMINAMATH_GPT_trigonometric_identity_l1381_138184

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 2) : 
  (2 * Real.cos α + 3 * Real.sin α) / (3 * Real.cos α - Real.sin α) = 8 :=
by 
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l1381_138184


namespace NUMINAMATH_GPT_cheese_cookies_price_is_correct_l1381_138153

-- Define the problem conditions and constants
def total_boxes_per_carton : ℕ := 15
def total_packs_per_box : ℕ := 12
def discount_15_percent : ℝ := 0.15
def total_number_of_cartons : ℕ := 13
def total_cost_paid : ℝ := 2058

-- Calculate the expected price per pack
noncomputable def price_per_pack : ℝ :=
  let total_packs := total_boxes_per_carton * total_packs_per_box * total_number_of_cartons
  let total_cost_without_discount := total_cost_paid / (1 - discount_15_percent)
  total_cost_without_discount / total_packs

theorem cheese_cookies_price_is_correct : 
  abs (price_per_pack - 1.0347) < 0.0001 :=
by sorry

end NUMINAMATH_GPT_cheese_cookies_price_is_correct_l1381_138153


namespace NUMINAMATH_GPT_Lara_age_10_years_from_now_l1381_138151

theorem Lara_age_10_years_from_now (current_year_age : ℕ) (age_7_years_ago : ℕ)
  (h1 : age_7_years_ago = 9) (h2 : current_year_age = age_7_years_ago + 7) :
  current_year_age + 10 = 26 :=
by
  sorry

end NUMINAMATH_GPT_Lara_age_10_years_from_now_l1381_138151


namespace NUMINAMATH_GPT_minimize_cost_l1381_138170

theorem minimize_cost (x : ℝ) (h1 : 0 < x) (h2 : 400 / x * 40 ≤ 4 * x) : x = 20 :=
by
  sorry

end NUMINAMATH_GPT_minimize_cost_l1381_138170


namespace NUMINAMATH_GPT_population_growth_l1381_138123

theorem population_growth (P_present P_future : ℝ) (r : ℝ) (n : ℕ)
  (h1 : P_present = 7800)
  (h2 : P_future = 10860.72)
  (h3 : n = 2) :
  P_future = P_present * (1 + r / 100)^n → r = 18.03 :=
by sorry

end NUMINAMATH_GPT_population_growth_l1381_138123


namespace NUMINAMATH_GPT_john_income_increase_l1381_138175

theorem john_income_increase :
  let initial_job_income := 60
  let initial_freelance_income := 40
  let initial_online_sales_income := 20

  let new_job_income := 120
  let new_freelance_income := 60
  let new_online_sales_income := 35

  let weeks_per_month := 4

  let initial_monthly_income := (initial_job_income + initial_freelance_income + initial_online_sales_income) * weeks_per_month
  let new_monthly_income := (new_job_income + new_freelance_income + new_online_sales_income) * weeks_per_month
  
  let percentage_increase := 100 * (new_monthly_income - initial_monthly_income) / initial_monthly_income

  percentage_increase = 79.17 := by
  sorry

end NUMINAMATH_GPT_john_income_increase_l1381_138175


namespace NUMINAMATH_GPT_max_value_sin_cos_combination_l1381_138132

theorem max_value_sin_cos_combination :
  ∀ x : ℝ, (5 * Real.sin x + 12 * Real.cos x) ≤ 13 :=
by
  intro x
  sorry

end NUMINAMATH_GPT_max_value_sin_cos_combination_l1381_138132


namespace NUMINAMATH_GPT_selling_price_of_cycle_l1381_138167

theorem selling_price_of_cycle
  (cost_price : ℕ)
  (gain_percent_decimal : ℚ)
  (h_cp : cost_price = 850)
  (h_gpd : gain_percent_decimal = 27.058823529411764 / 100) :
  ∃ selling_price : ℚ, selling_price = cost_price * (1 + gain_percent_decimal) ∧ selling_price = 1080 := 
by
  use (cost_price * (1 + gain_percent_decimal))
  sorry

end NUMINAMATH_GPT_selling_price_of_cycle_l1381_138167


namespace NUMINAMATH_GPT_slower_bike_longer_time_by_1_hour_l1381_138176

/-- Speed of the slower bike in kmph -/
def speed_slow : ℕ := 60

/-- Speed of the faster bike in kmph -/
def speed_fast : ℕ := 64

/-- Distance both bikes travel in km -/
def distance : ℕ := 960

/-- Time taken to travel the distance by a bike going at a certain speed -/
def time (speed : ℕ) : ℕ :=
  distance / speed

/-- Proof that the slower bike takes 1 hour longer to cover the distance compared to the faster bike -/
theorem slower_bike_longer_time_by_1_hour : 
  (time speed_slow) = (time speed_fast) + 1 := by
sorry

end NUMINAMATH_GPT_slower_bike_longer_time_by_1_hour_l1381_138176


namespace NUMINAMATH_GPT_smallest_a_l1381_138142

theorem smallest_a (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 96 * a^2 = b^3) : a = 12 :=
by
  sorry

end NUMINAMATH_GPT_smallest_a_l1381_138142


namespace NUMINAMATH_GPT_scientific_notation_correct_l1381_138105

def million : ℝ := 10^6
def num : ℝ := 1.06
def num_in_million : ℝ := num * million
def scientific_notation : ℝ := 1.06 * 10^6

theorem scientific_notation_correct : num_in_million = scientific_notation :=
by 
  -- The proof is skipped, indicated by sorry
  sorry

end NUMINAMATH_GPT_scientific_notation_correct_l1381_138105


namespace NUMINAMATH_GPT_building_height_l1381_138108

theorem building_height
  (num_stories_1 : ℕ)
  (height_story_1 : ℕ)
  (num_stories_2 : ℕ)
  (height_story_2 : ℕ)
  (h1 : num_stories_1 = 10)
  (h2 : height_story_1 = 12)
  (h3 : num_stories_2 = 10)
  (h4 : height_story_2 = 15)
  :
  num_stories_1 * height_story_1 + num_stories_2 * height_story_2 = 270 :=
by
  sorry

end NUMINAMATH_GPT_building_height_l1381_138108


namespace NUMINAMATH_GPT_max_product_three_distinct_nats_sum_48_l1381_138120

open Nat

theorem max_product_three_distinct_nats_sum_48
  (a b c : ℕ) (h_distinct: a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_sum: a + b + c = 48) :
  a * b * c ≤ 4080 :=
sorry

end NUMINAMATH_GPT_max_product_three_distinct_nats_sum_48_l1381_138120


namespace NUMINAMATH_GPT_NaOH_combined_l1381_138143

theorem NaOH_combined (n : ℕ) (h : n = 54) : 
  (2 * n) / 2 = 54 :=
by
  sorry

end NUMINAMATH_GPT_NaOH_combined_l1381_138143


namespace NUMINAMATH_GPT_simplified_expression_value_l1381_138160

theorem simplified_expression_value (x : ℝ) (h : x = -2) :
  (x - 2)^2 - 4 * x * (x - 1) + (2 * x + 1) * (2 * x - 1) = 7 := 
  by
    -- We are given x = -2
    simp [h]
    -- sorry added to skip the actual solution in Lean
    sorry

end NUMINAMATH_GPT_simplified_expression_value_l1381_138160


namespace NUMINAMATH_GPT_thirty_k_divisor_of_929260_l1381_138152

theorem thirty_k_divisor_of_929260 (k : ℕ) (h1: 30^k ∣ 929260):
(3^k - k^3 = 2) :=
sorry

end NUMINAMATH_GPT_thirty_k_divisor_of_929260_l1381_138152


namespace NUMINAMATH_GPT_rectangles_in_grid_at_least_three_cells_l1381_138145

theorem rectangles_in_grid_at_least_three_cells :
  let number_of_rectangles (n : ℕ) := (n + 1).choose 2 * (n + 1).choose 2
  let single_cell_rectangles (n : ℕ) := n * n
  let one_by_two_or_two_by_one_rectangles (n : ℕ) := n * (n - 1) * 2
  let total_rectangles (n : ℕ) := number_of_rectangles n - (single_cell_rectangles n + one_by_two_or_two_by_one_rectangles n)
  total_rectangles 6 = 345 :=
by
  sorry

end NUMINAMATH_GPT_rectangles_in_grid_at_least_three_cells_l1381_138145


namespace NUMINAMATH_GPT_find_f_7_l1381_138178

noncomputable def f (a b c x : ℝ) : ℝ := a * x^7 + b * x^3 + c * x - 5

theorem find_f_7 (a b c : ℝ) (h : f a b c (-7) = 7) : f a b c 7 = -17 :=
by
  dsimp [f] at *
  sorry

end NUMINAMATH_GPT_find_f_7_l1381_138178


namespace NUMINAMATH_GPT_evaluate_expression_at_x_eq_3_l1381_138148

theorem evaluate_expression_at_x_eq_3 :
  (3 ^ 3) ^ (3 ^ 3) = 7625597484987 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_at_x_eq_3_l1381_138148


namespace NUMINAMATH_GPT_lowest_common_denominator_l1381_138172

theorem lowest_common_denominator (a b c : ℕ) (h1 : a = 9) (h2 : b = 4) (h3 : c = 18) : Nat.lcm (Nat.lcm a b) c = 36 :=
by
  -- Introducing the given conditions
  rw [h1, h2, h3]
  -- Compute the LCM of the provided values
  sorry

end NUMINAMATH_GPT_lowest_common_denominator_l1381_138172


namespace NUMINAMATH_GPT_third_twenty_third_wise_superior_number_l1381_138119

def wise_superior_number (x : ℕ) : Prop :=
  ∃ m n : ℕ, m > n ∧ m - n > 1 ∧ x = m^2 - n^2

theorem third_twenty_third_wise_superior_number :
  ∃ T_3 T_23 : ℕ, wise_superior_number T_3 ∧ wise_superior_number T_23 ∧ T_3 = 15 ∧ T_23 = 57 :=
by
  sorry

end NUMINAMATH_GPT_third_twenty_third_wise_superior_number_l1381_138119


namespace NUMINAMATH_GPT_max_value_a_l1381_138102

theorem max_value_a (x y : ℝ) (h₁ : x > 0) (h₂ : y > 0) (h₃ : x + y = 1) : 
  ∃ a, a = 16 ∧ (∀ x y, (x > 0 → y > 0 → x + y = 1 → a ≤ (1/x) + (9/y))) :=
by 
  use 16
  sorry

end NUMINAMATH_GPT_max_value_a_l1381_138102


namespace NUMINAMATH_GPT_isosceles_triangle_angles_l1381_138124

theorem isosceles_triangle_angles (A B C : ℝ)
    (h_iso : A = B ∨ B = C ∨ C = A)
    (h_one_angle : A = 36 ∨ B = 36 ∨ C = 36)
    (h_sum_angles : A + B + C = 180) :
  (A = 36 ∧ B = 36 ∧ C = 108) ∨
  (A = 72 ∧ B = 72 ∧ C = 36) :=
by 
  sorry

end NUMINAMATH_GPT_isosceles_triangle_angles_l1381_138124


namespace NUMINAMATH_GPT_negation_proposition_l1381_138169

theorem negation_proposition (x y : ℝ) :
  (¬ ∃ (x y : ℝ), 2 * x + 3 * y + 3 < 0) ↔ (∀ (x y : ℝ), 2 * x + 3 * y + 3 ≥ 0) :=
by {
  sorry
}

end NUMINAMATH_GPT_negation_proposition_l1381_138169
