import Mathlib

namespace percentage_increase_l577_57750

theorem percentage_increase (R W : ℕ) (hR : R = 36) (hW : W = 20) : 
  ((R - W) / W : ℚ) * 100 = 80 := 
by 
  sorry

end percentage_increase_l577_57750


namespace calc_expression_l577_57742

theorem calc_expression : 
  (Real.sqrt 16 - 4 * (Real.sqrt 2) / 2 + abs (- (Real.sqrt 3 * Real.sqrt 6)) + (-1) ^ 2023) = 
  (3 + Real.sqrt 2) :=
by
  sorry

end calc_expression_l577_57742


namespace total_obstacle_course_time_l577_57770

-- Definitions for the given conditions
def first_part_time : Nat := 7 * 60 + 23
def second_part_time : Nat := 73
def third_part_time : Nat := 5 * 60 + 58

-- State the main theorem
theorem total_obstacle_course_time :
  first_part_time + second_part_time + third_part_time = 874 :=
by
  sorry

end total_obstacle_course_time_l577_57770


namespace angle_C_is_110_degrees_l577_57799

def lines_are_parallel (l m : Type) : Prop := sorry
def angle_measure (A : Type) : ℝ := sorry
noncomputable def mangle (C : Type) : ℝ := sorry

theorem angle_C_is_110_degrees 
  (l m C D : Type) 
  (hlm : lines_are_parallel l m)
  (hCDl : lines_are_parallel C l)
  (hCDm : lines_are_parallel C m)
  (hA : angle_measure A = 100)
  (hB : angle_measure B = 150) :
  mangle C = 110 :=
by
  sorry

end angle_C_is_110_degrees_l577_57799


namespace find_radius_l577_57778

theorem find_radius (AB EO : ℝ) (AE BE : ℝ) (h1 : AB = AE + BE) (h2 : AE = 2 * BE) (h3 : EO = 7) :
  ∃ R : ℝ, R = 11 := by
  sorry

end find_radius_l577_57778


namespace vector_addition_l577_57711

-- Defining the vectors a and b
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (1, 3)

-- Stating the problem: proving the sum of vectors a and b
theorem vector_addition : a + b = (3, 4) := 
by 
  -- Proof is not required as per the instructions
  sorry

end vector_addition_l577_57711


namespace ellipse_equation_from_hyperbola_l577_57759

theorem ellipse_equation_from_hyperbola :
  (∃ (a b : ℝ), ∀ x y : ℝ, (x^2 / 4 - y^2 / 12 = 1) →
  (x^2 / 16 + y^2 / 12 = 1)) :=
by
  sorry

end ellipse_equation_from_hyperbola_l577_57759


namespace find_mystery_number_l577_57745

theorem find_mystery_number (x : ℕ) (h : x + 45 = 92) : x = 47 :=
sorry

end find_mystery_number_l577_57745


namespace money_problem_l577_57726

variable {c d : ℝ}

theorem money_problem (h1 : 3 * c - 2 * d < 30) (h2 : 4 * c + d = 60) : 
  c < 150 / 11 ∧ d > 60 / 11 := 
by 
  sorry

end money_problem_l577_57726


namespace range_of_a_l577_57741

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (2 - a) * x^2 + 2 * (2 - a) * x + 4 ≥ 0) → (-2 ≤ a ∧ a ≤ 2) := 
sorry

end range_of_a_l577_57741


namespace find_real_number_l577_57793

theorem find_real_number (x : ℝ) (h1 : 0 < x) (h2 : ⌊x⌋ * x = 72) : x = 9 :=
sorry

end find_real_number_l577_57793


namespace horner_method_correct_l577_57732

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

end horner_method_correct_l577_57732


namespace cubic_root_sqrt_equation_l577_57714

theorem cubic_root_sqrt_equation (x : ℝ) (h1 : 3 - x = y^3) (h2 : x - 2 = z^2) (h3 : y + z = 1) : 
  x = 3 ∨ x = 2 ∨ x = 11 :=
sorry

end cubic_root_sqrt_equation_l577_57714


namespace remainder_proof_l577_57795

def nums : List ℕ := [83, 84, 85, 86, 87, 88, 89, 90]
def mod : ℕ := 17

theorem remainder_proof : (nums.sum % mod) = 3 := by sorry

end remainder_proof_l577_57795


namespace max_volume_48cm_square_l577_57743

def volume_of_box (x : ℝ) := x * (48 - 2 * x)^2

theorem max_volume_48cm_square : 
  ∃ x : ℝ, 0 < x ∧ x < 24 ∧ (∀ y : ℝ, 0 < y ∧ y < 24 → volume_of_box x ≥ volume_of_box y) ∧ x = 8 :=
sorry

end max_volume_48cm_square_l577_57743


namespace positive_difference_of_squares_l577_57787

theorem positive_difference_of_squares (a b : ℕ) (h1 : a + b = 40) (h2 : a - b = 8) : a^2 - b^2 = 320 :=
by
  sorry

end positive_difference_of_squares_l577_57787


namespace apple_tree_bears_fruit_in_7_years_l577_57744

def age_planted : ℕ := 4
def age_eats : ℕ := 11
def time_to_bear_fruit : ℕ := age_eats - age_planted

theorem apple_tree_bears_fruit_in_7_years :
  time_to_bear_fruit = 7 :=
by
  sorry

end apple_tree_bears_fruit_in_7_years_l577_57744


namespace determine_b_l577_57716

theorem determine_b (a b c : ℕ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (eq_radicals: Real.sqrt (4 * a + 4 * b / c) = 2 * a * Real.sqrt (b / c)) : 
  b = c + 1 :=
sorry

end determine_b_l577_57716


namespace order_of_values_l577_57746

noncomputable def a : ℝ := 21.2
noncomputable def b : ℝ := Real.sqrt 450 - 0.8
noncomputable def c : ℝ := 2 * Real.logb 5 2

theorem order_of_values : c < b ∧ b < a := by 
  sorry

end order_of_values_l577_57746


namespace minimal_primes_ensuring_first_player_win_l577_57722

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

end minimal_primes_ensuring_first_player_win_l577_57722


namespace area_H1H2H3_eq_four_l577_57738

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

end area_H1H2H3_eq_four_l577_57738


namespace problem_inequality_l577_57776

variable {a b c : ℝ}

-- Assuming a, b, c are positive real numbers
variables (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)

-- Assuming abc = 1
variable (h_abc : a * b * c = 1)

theorem problem_inequality :
  (1 / (a^3 * (b + c))) + (1 / (b^3 * (c + a))) + (1 / (c^3 * (a + b))) ≥ (3 / 2) :=
by sorry

end problem_inequality_l577_57776


namespace not_perfect_square_l577_57728

-- Definitions and Conditions
def N (k : ℕ) : ℕ := (10^300 - 1) / 9 * 10^k

-- Proof Statement
theorem not_perfect_square (k : ℕ) : ¬∃ (m: ℕ), m * m = N k := 
sorry

end not_perfect_square_l577_57728


namespace total_marbles_in_bowls_l577_57754

theorem total_marbles_in_bowls :
  let second_bowl := 600
  let first_bowl := 3 / 4 * second_bowl
  let third_bowl := 1 / 2 * first_bowl
  let fourth_bowl := 1 / 3 * second_bowl
  first_bowl + second_bowl + third_bowl + fourth_bowl = 1475 :=
by
  sorry

end total_marbles_in_bowls_l577_57754


namespace similarity_coordinates_l577_57791

theorem similarity_coordinates {B B1 : ℝ × ℝ} 
  (h₁ : ∃ (k : ℝ), k = 2 ∧ 
         (∀ (x y : ℝ), B = (x, y) → ∀ (x₁ y₁ : ℝ), B1 = (x₁, y₁) → x₁ = x / k ∨ x₁ = x / -k) ∧ 
         (∀ (x y : ℝ), B = (x, y) → ∀ (x₁ y₁ : ℝ), B1 = (x₁, y₁) → y₁ = y / k ∨ y₁ = y / -k))
  (h₂ : B = (-4, -2)) :
  B1 = (-2, -1) ∨ B1 = (2, 1) :=
sorry

end similarity_coordinates_l577_57791


namespace pages_read_in_7_days_l577_57785

-- Definitions of the conditions
def total_hours : ℕ := 10
def days : ℕ := 5
def pages_per_hour : ℕ := 50
def reading_days : ℕ := 7

-- Compute intermediate steps
def hours_per_day : ℕ := total_hours / days
def pages_per_day : ℕ := pages_per_hour * hours_per_day

-- Lean statement to prove Tom reads 700 pages in 7 days
theorem pages_read_in_7_days :
  pages_per_day * reading_days = 700 :=
by
  -- We can add the intermediate steps here as sorry, as we will not do the proof
  sorry

end pages_read_in_7_days_l577_57785


namespace smallest_int_ending_in_9_divisible_by_11_l577_57768

theorem smallest_int_ending_in_9_divisible_by_11:
  ∃ x : ℕ, (∃ k : ℤ, x = 10 * k + 9) ∧ x % 11 = 0 ∧ x = 99 :=
by
  sorry

end smallest_int_ending_in_9_divisible_by_11_l577_57768


namespace reciprocal_power_l577_57758

theorem reciprocal_power (a : ℝ) (h : 1 / a = -1) : a ^ 2023 = -1 :=
by sorry

end reciprocal_power_l577_57758


namespace sequence_arithmetic_difference_neg1_l577_57782

variable (a : ℕ → ℝ)

theorem sequence_arithmetic_difference_neg1 (h : ∀ n, a (n + 1) + 1 = a n) : ∀ n, a (n + 1) - a n = -1 :=
by
  intro n
  specialize h n
  linarith

-- Assuming natural numbers starting from 1 (ℕ^*), which is not directly available in Lean.
-- So we use assumptions accordingly.

end sequence_arithmetic_difference_neg1_l577_57782


namespace partial_fraction_decomposition_l577_57771

theorem partial_fraction_decomposition (A B C : ℚ) :
  (∀ x : ℚ, x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1 →
    (-x^2 + 5*x - 6) / (x^3 - x) = A / x + (B*x + C) / (x^2 - 1)) →
  A = 6 ∧ B = -7 ∧ C = 5 :=
by
  intro h
  sorry

end partial_fraction_decomposition_l577_57771


namespace a_b_fifth_power_equals_c_d_fifth_power_cannot_conclude_fourth_powers_l577_57786

variables (a b c d : ℝ)

-- Given conditions
def first_condition : Prop := a + b = c + d
def second_condition : Prop := a^3 + b^3 = c^3 + d^3

-- Proof problem for part (a)
theorem a_b_fifth_power_equals_c_d_fifth_power 
  (h1 : first_condition a b c d) 
  (h2 : second_condition a b c d) : a^5 + b^5 = c^5 + d^5 := 
sorry

-- Proof problem for part (b)
theorem cannot_conclude_fourth_powers 
  (h1 : first_condition a b c d) 
  (h2 : second_condition a b c d) : ¬ (a^4 + b^4 = c^4 + d^4) :=
sorry

end a_b_fifth_power_equals_c_d_fifth_power_cannot_conclude_fourth_powers_l577_57786


namespace moles_Cl2_combined_l577_57747

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

end moles_Cl2_combined_l577_57747


namespace price_of_other_frisbees_l577_57769

-- Lean 4 Statement
theorem price_of_other_frisbees (P : ℝ) (x : ℕ) (h1 : x ≥ 40) (h2 : P * x + 4 * (60 - x) = 200) :
  P = 3 := 
  sorry

end price_of_other_frisbees_l577_57769


namespace combined_weight_of_boxes_l577_57702

def weight_box1 : ℝ := 2
def weight_box2 : ℝ := 11
def weight_box3 : ℝ := 5

theorem combined_weight_of_boxes : weight_box1 + weight_box2 + weight_box3 = 18 := by
  sorry

end combined_weight_of_boxes_l577_57702


namespace original_class_size_l577_57709

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

end original_class_size_l577_57709


namespace proof_quotient_l577_57735

/-- Let x be in the form (a + b * sqrt c) / d -/
def x_form (a b c d : ℤ) (x : ℝ) : Prop := x = (a + b * Real.sqrt c) / d

/-- Main theorem -/
theorem proof_quotient (a b c d : ℤ) (x : ℝ) (h_eq : 4 * x / 5 + 2 = 5 / x) (h_form : x_form a b c d x) : (a * c * d) / b = -20 := by
  sorry

end proof_quotient_l577_57735


namespace a_and_b_solution_l577_57797

noncomputable def solve_for_a_b (a b : ℕ) : Prop :=
  a > 0 ∧ (∀ b : ℤ, b > 0) ∧ (2 * a^b + 16 + 3 * a^b - 8) / 2 = 84 → a = 2 ∧ b = 5

theorem a_and_b_solution (a b : ℕ) (h : solve_for_a_b a b) : a = 2 ∧ b = 5 :=
sorry

end a_and_b_solution_l577_57797


namespace fill_tank_in_6_hours_l577_57703

theorem fill_tank_in_6_hours (A B : ℝ) (hA : A = 1 / 10) (hB : B = 1 / 15) : (1 / (A + B)) = 6 :=
by 
  sorry

end fill_tank_in_6_hours_l577_57703


namespace avg_visitors_per_day_l577_57727

theorem avg_visitors_per_day :
  let visitors := [583, 246, 735, 492, 639]
  (visitors.sum / visitors.length) = 539 := by
  sorry

end avg_visitors_per_day_l577_57727


namespace gasoline_price_increase_l577_57761

theorem gasoline_price_increase :
  ∀ (p_low p_high : ℝ), p_low = 14 → p_high = 23 → 
  ((p_high - p_low) / p_low) * 100 = 64.29 :=
by
  intro p_low p_high h_low h_high
  rw [h_low, h_high]
  sorry

end gasoline_price_increase_l577_57761


namespace find_y_when_x_4_l577_57719

-- Definitions and conditions
variables (x y : ℝ)
def inversely_proportional (x y : ℝ) (K : ℝ) : Prop := x * y = K

-- Main theorem
theorem find_y_when_x_4 
  (K : ℝ) (h1 : inversely_proportional 20 10 K) (h2 : 20 + 10 = 30) (h3 : 20 - 10 = 10) 
  (hx : 4 * y = K) : y = 50 := 
sorry

end find_y_when_x_4_l577_57719


namespace square_side_length_l577_57788

/-- 
If a square is drawn by joining the midpoints of the sides of a given square and repeating this process continues indefinitely,
and the sum of the areas of all the squares is 32 cm²,
then the length of the side of the first square is 4 cm. 
-/
theorem square_side_length (s : ℝ) (h : ∑' n : ℕ, (s^2) * (1 / 2)^n = 32) : s = 4 := 
by 
  sorry

end square_side_length_l577_57788


namespace sarah_interview_combinations_l577_57780

theorem sarah_interview_combinations : 
  (1 * 2 * (2 + 3) * 5 * 1) = 50 := 
by
  sorry

end sarah_interview_combinations_l577_57780


namespace christopher_age_l577_57792

variable (C G F : ℕ)

theorem christopher_age (h1 : G = C + 8) (h2 : F = C - 2) (h3 : C + G + F = 60) : C = 18 := by
  sorry

end christopher_age_l577_57792


namespace intersection_of_M_and_N_l577_57730

def set_M (x : ℝ) : Prop := 1 - 2 / x < 0
def set_N (x : ℝ) : Prop := -1 ≤ x
def set_Intersection (x : ℝ) : Prop := 0 < x ∧ x < 2

theorem intersection_of_M_and_N :
  ∀ x, (set_M x ∧ set_N x) ↔ set_Intersection x :=
by sorry

end intersection_of_M_and_N_l577_57730


namespace boat_downstream_travel_time_l577_57706

theorem boat_downstream_travel_time (D : ℝ) (V_b : ℝ) (T_u : ℝ) (V_c : ℝ) (T_d : ℝ) : 
  D = 300 ∧ V_b = 105 ∧ T_u = 5 ∧ (300 = (105 - V_c) * 5) ∧ (300 = (105 + V_c) * T_d) → T_d = 2 :=
by
  sorry

end boat_downstream_travel_time_l577_57706


namespace jack_grassy_time_is_6_l577_57729

def jack_sandy_time := 19
def jill_total_time := 32
def jill_time_delay := 7
def jack_total_time : ℕ := jill_total_time - jill_time_delay
def jack_grassy_time : ℕ := jack_total_time - jack_sandy_time

theorem jack_grassy_time_is_6 : jack_grassy_time = 6 := by 
  have h1: jack_total_time = 25 := by sorry
  have h2: jack_grassy_time = 6 := by sorry
  exact h2

end jack_grassy_time_is_6_l577_57729


namespace how_many_ducks_did_john_buy_l577_57724

def cost_price_per_duck : ℕ := 10
def weight_per_duck : ℕ := 4
def selling_price_per_pound : ℕ := 5
def profit : ℕ := 300

theorem how_many_ducks_did_john_buy (D : ℕ) (h : 10 * D - 10 * D + 10 * D = profit) : D = 30 :=
by 
  sorry

end how_many_ducks_did_john_buy_l577_57724


namespace children_of_exceptions_l577_57762

theorem children_of_exceptions (x y : ℕ) (h : 6 * x + 2 * y = 58) (hx : x = 8) : y = 5 :=
by
  sorry

end children_of_exceptions_l577_57762


namespace volume_of_region_l577_57749

theorem volume_of_region :
    ∀ (x y z : ℝ), 
    |x - y + z| + |x - y - z| ≤ 12 ∧ x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 
    → true := by
    sorry

end volume_of_region_l577_57749


namespace expand_product_l577_57781

theorem expand_product (y : ℝ) : (y + 3) * (y + 7) = y^2 + 10 * y + 21 := by
  sorry

end expand_product_l577_57781


namespace sum_of_cubes_eq_neg_27_l577_57783

theorem sum_of_cubes_eq_neg_27
  (a b c : ℝ)
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_condition : (a^3 + 9) / a = (b^3 + 9) / b ∧ (b^3 + 9) / b = (c^3 + 9) / c) :
  a^3 + b^3 + c^3 = -27 :=
sorry

end sum_of_cubes_eq_neg_27_l577_57783


namespace evaluate_101_times_101_l577_57707

theorem evaluate_101_times_101 : 101 * 101 = 10201 :=
by sorry

end evaluate_101_times_101_l577_57707


namespace full_capacity_l577_57737

def oil_cylinder_capacity (C : ℝ) :=
  (4 / 5) * C - (3 / 4) * C = 4

theorem full_capacity : oil_cylinder_capacity 80 :=
by
  simp [oil_cylinder_capacity]
  sorry

end full_capacity_l577_57737


namespace problem_statement_l577_57740

open Complex

theorem problem_statement (x y : ℝ) (i : ℂ) (h_i : i = Complex.I) (h : x + (y - 2) * i = 2 / (1 + i)) : x + y = 2 :=
by
  sorry

end problem_statement_l577_57740


namespace domain_of_sqrt_1_minus_2_cos_l577_57763

theorem domain_of_sqrt_1_minus_2_cos (x : ℝ) (k : ℤ) :
  1 - 2 * Real.cos x ≥ 0 ↔ ∃ k : ℤ, (π / 3 + 2 * k * π ≤ x ∧ x ≤ 5 * π / 3 + 2 * k * π) :=
by
  sorry

end domain_of_sqrt_1_minus_2_cos_l577_57763


namespace evaluate_fraction_l577_57718

theorem evaluate_fraction : (1 - 1/4) / (1 - 1/3) = 9/8 :=
by
  sorry

end evaluate_fraction_l577_57718


namespace christopher_age_l577_57720

variables (C G : ℕ)

theorem christopher_age :
  (C = 2 * G) ∧ (C - 9 = 5 * (G - 9)) → C = 24 :=
by
  intro h
  sorry

end christopher_age_l577_57720


namespace problem_part_1_problem_part_2_l577_57773
open Set Real

noncomputable def A (a : ℝ) : Set ℝ := {x | a < x ∧ x ≤ a^2 - 2}
noncomputable def B : Set ℝ := {x | 1 < x ∧ x < 5}

theorem problem_part_1 : A 3 ∪ B = {x | 1 < x ∧ x ≤ 7} := 
  by
  sorry

theorem problem_part_2 : (∀ a : ℝ, A a ∪ B = B → 2 < a ∧ a < sqrt 7) :=
  by 
  sorry

end problem_part_1_problem_part_2_l577_57773


namespace percentage_reduction_l577_57775

variable (P R : ℝ)
variable (ReducedPrice : R = 15)
variable (AmountMore : 900 / 15 - 900 / P = 6)

theorem percentage_reduction (ReducedPrice : R = 15) (AmountMore : 900 / 15 - 900 / P = 6) :
  (P - R) / P * 100 = 10 :=
by
  sorry

end percentage_reduction_l577_57775


namespace vertex_of_parabola_is_correct_l577_57765

theorem vertex_of_parabola_is_correct :
  ∀ x y : ℝ, y = -5 * (x + 2) ^ 2 - 6 → (x = -2 ∧ y = -6) :=
by
  sorry

end vertex_of_parabola_is_correct_l577_57765


namespace sequence_value_l577_57790

theorem sequence_value (x : ℕ) : 
  (5 - 2 = 1 * 3) ∧ 
  (11 - 5 = 2 * 3) ∧ 
  (20 - 11 = 3 * 3) ∧ 
  (x - 20 = 4 * 3) ∧ 
  (47 - x = 5 * 3) → 
  x = 32 :=
by 
  intros h 
  sorry

end sequence_value_l577_57790


namespace solve_ratios_l577_57704

theorem solve_ratios (q m n : ℕ) (h1 : 7 / 9 = n / 108) (h2 : 7 / 9 = (m + n) / 126) (h3 : 7 / 9 = (q - m) / 162) : q = 140 :=
by
  sorry

end solve_ratios_l577_57704


namespace dad_caught_more_l577_57756

theorem dad_caught_more {trouts_caleb : ℕ} (h₁ : trouts_caleb = 2) 
    (h₂ : ∃ trouts_dad : ℕ, trouts_dad = 3 * trouts_caleb) : 
    ∃ more_trouts : ℕ, more_trouts = 4 := by
  sorry

end dad_caught_more_l577_57756


namespace simplify_and_evaluate_l577_57717

theorem simplify_and_evaluate (x : ℝ) (hx : x = Real.sqrt 2 + 1) :
  (x + 1) / x / (x - (1 + x^2) / (2 * x)) = Real.sqrt 2 :=
by
  sorry

end simplify_and_evaluate_l577_57717


namespace soybeans_to_oil_l577_57777

theorem soybeans_to_oil 
    (kg_soybeans_to_tofu : ℝ)
    (kg_soybeans_to_oil : ℝ)
    (price_soybeans : ℝ)
    (price_tofu : ℝ)
    (price_oil : ℝ)
    (purchase_amount : ℝ)
    (sales_amount : ℝ)
    (amount_to_oil : ℝ)
    (used_soybeans_for_oil : ℝ) :
    kg_soybeans_to_tofu = 3 →
    kg_soybeans_to_oil = 6 →
    price_soybeans = 2 →
    price_tofu = 3 →
    price_oil = 15 →
    purchase_amount = 920 →
    sales_amount = 1800 →
    used_soybeans_for_oil = 360 →
    (6 * amount_to_oil) = 360 →
    15 * amount_to_oil + 3 * (460 - 6 * amount_to_oil) = 1800 :=
by sorry

end soybeans_to_oil_l577_57777


namespace odd_function_has_zero_l577_57796

variable {R : Type} [LinearOrderedField R]

def is_odd_function (f : R → R) := ∀ x : R, f (-x) = -f x

theorem odd_function_has_zero {f : R → R} (h : is_odd_function f) : ∃ x : R, f x = 0 :=
sorry

end odd_function_has_zero_l577_57796


namespace number_of_valid_integers_l577_57774

theorem number_of_valid_integers (n : ℕ) (h1 : n ≤ 2021) (h2 : ∀ m : ℕ, m^2 ≤ n → n < (m + 1)^2 → ((m^2 + 1) ∣ (n^2 + 1))) : 
  ∃ k, k = 47 :=
by
  sorry

end number_of_valid_integers_l577_57774


namespace greatest_integer_gcd_is_4_l577_57725

theorem greatest_integer_gcd_is_4 : 
  ∀ (n : ℕ), n < 150 ∧ (Nat.gcd n 24 = 4) → n ≤ 148 := 
by
  sorry

end greatest_integer_gcd_is_4_l577_57725


namespace art_piece_increase_is_correct_l577_57764

-- Define the conditions
def initial_price : ℝ := 4000
def future_multiplier : ℝ := 3
def future_price : ℝ := future_multiplier * initial_price

-- Define the goal
-- Proof that the increase in price is equal to $8000
theorem art_piece_increase_is_correct : future_price - initial_price = 8000 := 
by {
  -- We put sorry here to skip the actual proof
  sorry
}

end art_piece_increase_is_correct_l577_57764


namespace integer_values_count_l577_57753

theorem integer_values_count (x : ℤ) :
  ∃ k, (∀ n : ℤ, (3 ≤ Real.sqrt (3 * n + 1) ∧ Real.sqrt (3 * n + 1) < 5) ↔ ((n = 3) ∨ (n = 4) ∨ (n = 5) ∨ (n = 6) ∨ (n = 7)) ∧ k = 5) :=
by
  sorry

end integer_values_count_l577_57753


namespace largest_possible_d_plus_r_l577_57700

theorem largest_possible_d_plus_r :
  ∃ d r : ℕ, 0 < d ∧ 468 % d = r ∧ 636 % d = r ∧ 867 % d = r ∧ d + r = 27 := by
  sorry

end largest_possible_d_plus_r_l577_57700


namespace curve_equation_represents_line_l577_57757

noncomputable def curve_is_line (x y : ℝ) : Prop :=
(x^2 + y^2 - 2*x) * (x + y - 3)^(1/2) = 0

theorem curve_equation_represents_line (x y : ℝ) :
curve_is_line x y ↔ (x + y = 3) :=
by sorry

end curve_equation_represents_line_l577_57757


namespace only_n_equal_3_exists_pos_solution_l577_57779

theorem only_n_equal_3_exists_pos_solution :
  ∀ (n : ℕ), (∃ (x y z : ℕ), 0 < x ∧ 0 < y ∧ 0 < z ∧ x^3 + y^3 + z^3 = n * x^2 * y^2 * z^2) ↔ n = 3 := 
by
  sorry

end only_n_equal_3_exists_pos_solution_l577_57779


namespace intersection_two_elements_l577_57772

open Real Set

-- Definitions
def M (k : ℝ) : Set (ℝ × ℝ) := {p | ∃ x y : ℝ, p = (x, y) ∧ y = k * (x - 1) + 1}
def N : Set (ℝ × ℝ) := {p | ∃ x y : ℝ, p = (x, y) ∧ x^2 + y^2 - 2 * y = 0}

-- Statement of the problem
theorem intersection_two_elements (k : ℝ) (hk : k ≠ 0) :
  ∃ x1 y1 x2 y2 : ℝ,
    (x1, y1) ∈ M k ∧ (x1, y1) ∈ N ∧ 
    (x2, y2) ∈ M k ∧ (x2, y2) ∈ N ∧ 
    (x1, y1) ≠ (x2, y2) := sorry

end intersection_two_elements_l577_57772


namespace alyssa_initial_puppies_l577_57789

theorem alyssa_initial_puppies (gave_away has_left : ℝ) (h1 : gave_away = 8.5) (h2 : has_left = 12.5) :
    (gave_away + has_left = 21) :=
by
    sorry

end alyssa_initial_puppies_l577_57789


namespace boys_bought_balloons_l577_57721

def initial_balloons : ℕ := 3 * 12  -- Clown initially has 3 dozen balloons, i.e., 36 balloons
def girls_balloons : ℕ := 12        -- 12 girls buy a balloon each
def balloons_remaining : ℕ := 21     -- Clown is left with 21 balloons

def boys_balloons : ℕ :=
  initial_balloons - balloons_remaining - girls_balloons

theorem boys_bought_balloons :
  boys_balloons = 3 :=
by
  sorry

end boys_bought_balloons_l577_57721


namespace solution_set_of_inequality_l577_57710

theorem solution_set_of_inequality (x : ℝ) (h : x ≠ 2) :
  (1 / (x - 2) > -2) ↔ (x < 3 / 2 ∨ x > 2) :=
by sorry

end solution_set_of_inequality_l577_57710


namespace adult_ticket_cost_l577_57748

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

end adult_ticket_cost_l577_57748


namespace income_of_person_l577_57751

theorem income_of_person (x: ℝ) (h : 9 * x - 8 * x = 2000) : 9 * x = 18000 :=
by
  sorry

end income_of_person_l577_57751


namespace total_parcel_boxes_l577_57739

theorem total_parcel_boxes (a b c d : ℕ) (row_boxes column_boxes total_boxes : ℕ)
  (h_left : a = 7) (h_right : b = 13)
  (h_front : c = 8) (h_back : d = 14)
  (h_row : row_boxes = a - 1 + 1 + b) -- boxes in a row: (a - 1) + 1 (parcel itself) + b
  (h_column : column_boxes = c - 1 + 1 + d) -- boxes in a column: (c -1) + 1(parcel itself) + d
  (h_total : total_boxes = row_boxes * column_boxes) :
  total_boxes = 399 := by
  sorry

end total_parcel_boxes_l577_57739


namespace pieces_info_at_most_two_identical_digits_l577_57784

def num_pieces_of_information_with_at_most_two_positions_as_0110 : Nat :=
  (Nat.choose 4 2 + Nat.choose 4 1 + Nat.choose 4 0)

theorem pieces_info_at_most_two_identical_digits :
  num_pieces_of_information_with_at_most_two_positions_as_0110 = 11 :=
by
  sorry

end pieces_info_at_most_two_identical_digits_l577_57784


namespace yogurt_cost_l577_57715

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

end yogurt_cost_l577_57715


namespace total_balls_in_bag_l577_57733

theorem total_balls_in_bag (R G B T : ℕ) 
  (hR : R = 907) 
  (hRatio : 15 * T = 15 * R + 13 * R + 17 * R)
  : T = 2721 :=
sorry

end total_balls_in_bag_l577_57733


namespace value_x_when_y2_l577_57734

theorem value_x_when_y2 (x : ℝ) (h1 : ∃ (x : ℝ), y = 1 / (5 * x + 2)) (h2 : y = 2) : x = -3 / 10 := by
  sorry

end value_x_when_y2_l577_57734


namespace cricket_runs_l577_57794

variable (A B C D E : ℕ)

theorem cricket_runs
  (h1 : (A + B + C + D + E) = 180)
  (h2 : D = E + 5)
  (h3 : A = E + 8)
  (h4 : B = D + E)
  (h5 : B + C = 107) :
  E = 20 := by
  sorry

end cricket_runs_l577_57794


namespace eq_of_divisible_l577_57767

theorem eq_of_divisible (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h : a + b ∣ 5 * a + 3 * b) : a = b :=
sorry

end eq_of_divisible_l577_57767


namespace general_form_of_quadratic_equation_l577_57752

noncomputable def quadratic_equation_general_form (x : ℝ) : Prop :=
  (x + 3) * (x - 1) = 2 * x - 4

theorem general_form_of_quadratic_equation (x : ℝ) :
  quadratic_equation_general_form x → x^2 + 1 = 0 :=
sorry

end general_form_of_quadratic_equation_l577_57752


namespace painting_time_l577_57712

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

end painting_time_l577_57712


namespace relationship_between_x_b_a_l577_57723

variable {x b a : ℝ}

theorem relationship_between_x_b_a 
  (hx : x < 0) (hb : b < 0) (ha : a < 0)
  (hxb : x < b) (hba : b < a) : x^2 > b * x ∧ b * x > b^2 :=
by sorry

end relationship_between_x_b_a_l577_57723


namespace z_when_y_six_l577_57736

theorem z_when_y_six
    (k : ℝ)
    (h1 : ∀ y (z : ℝ), y^2 * Real.sqrt z = k)
    (h2 : ∃ (y : ℝ) (z : ℝ), y = 3 ∧ z = 4 ∧ y^2 * Real.sqrt z = k) :
  ∃ z : ℝ, y = 6 ∧ z = 1 / 4 := 
sorry

end z_when_y_six_l577_57736


namespace problem_statement_l577_57701

namespace ProofProblems

open Set

def U : Set ℕ := {2, 3, 4, 5, 6}
def M : Set ℕ := {3, 4, 5}
def N : Set ℕ := {2, 4, 5, 6}

theorem problem_statement : M ∪ N = U := sorry

end ProofProblems

end problem_statement_l577_57701


namespace eight_distinct_solutions_l577_57755

noncomputable def f (x : ℝ) : ℝ := x^2 - 2

theorem eight_distinct_solutions : 
  ∃ S : Finset ℝ, S.card = 8 ∧ ∀ x ∈ S, f (f (f x)) = x :=
sorry

end eight_distinct_solutions_l577_57755


namespace last_place_is_Fedya_l577_57798

def position_is_valid (position : ℕ) := position >= 1 ∧ position <= 4

variable (Misha Anton Petya Fedya : ℕ)

axiom Misha_statement: position_is_valid Misha → Misha ≠ 1 ∧ Misha ≠ 4
axiom Anton_statement: position_is_valid Anton → Anton ≠ 4
axiom Petya_statement: position_is_valid Petya → Petya = 1
axiom Fedya_statement: position_is_valid Fedya → Fedya = 4

theorem last_place_is_Fedya : ∃ (x : ℕ), x = Fedya ∧ Fedya = 4 :=
by
  sorry

end last_place_is_Fedya_l577_57798


namespace solution_set_of_equation_l577_57731

theorem solution_set_of_equation :
  {p : ℝ × ℝ | p.1 * p.2 + 1 = p.1 + p.2} = {p : ℝ × ℝ | p.1 = 1 ∨ p.2 = 1} :=
by 
  sorry

end solution_set_of_equation_l577_57731


namespace number_of_days_worked_l577_57705

-- Definitions based on the given conditions and question
def total_hours_worked : ℕ := 15
def hours_worked_each_day : ℕ := 3

-- The statement we need to prove:
theorem number_of_days_worked : 
  (total_hours_worked / hours_worked_each_day) = 5 :=
by
  sorry

end number_of_days_worked_l577_57705


namespace door_cranking_time_l577_57713

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

end door_cranking_time_l577_57713


namespace expression_value_l577_57766

theorem expression_value (a b c d m : ℝ) 
  (h1 : a = -b) 
  (h2 : c * d = 1) 
  (h3 : |m| = 3) :
  (a + b) / m - c * d + m = 2 ∨ (a + b) / m - c * d + m = -4 := 
by
  sorry

end expression_value_l577_57766


namespace janine_read_pages_in_two_months_l577_57760

theorem janine_read_pages_in_two_months :
  (let books_last_month := 5
   let books_this_month := 2 * books_last_month
   let total_books := books_last_month + books_this_month
   let pages_per_book := 10
   total_books * pages_per_book = 150) := by
   sorry

end janine_read_pages_in_two_months_l577_57760


namespace total_students_l577_57708

def numStudents (skiing scavenger : ℕ) : ℕ :=
  skiing + scavenger

theorem total_students (skiing scavenger : ℕ) (h1 : skiing = 2 * scavenger) (h2 : scavenger = 4000) :
  numStudents skiing scavenger = 12000 :=
by
  sorry

end total_students_l577_57708
