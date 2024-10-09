import Mathlib

namespace gail_has_two_ten_dollar_bills_l2087_208776

-- Define the given conditions
def total_amount : ℕ := 100
def num_five_bills : ℕ := 4
def num_twenty_bills : ℕ := 3
def value_five_bill : ℕ := 5
def value_twenty_bill : ℕ := 20
def value_ten_bill : ℕ := 10

-- The function to determine the number of ten-dollar bills
noncomputable def num_ten_bills : ℕ := 
  (total_amount - (num_five_bills * value_five_bill + num_twenty_bills * value_twenty_bill)) / value_ten_bill

-- Proof statement
theorem gail_has_two_ten_dollar_bills : num_ten_bills = 2 := by
  sorry

end gail_has_two_ten_dollar_bills_l2087_208776


namespace length_of_each_train_l2087_208724

theorem length_of_each_train (L : ℝ) (s1 : ℝ) (s2 : ℝ) (t : ℝ)
    (h1 : s1 = 46) (h2 : s2 = 36) (h3 : t = 144) (h4 : 2 * L = ((s1 - s2) * (5 / 18)) * t) :
    L = 200 := 
sorry

end length_of_each_train_l2087_208724


namespace area_of_right_triangle_ABC_l2087_208770

variable {A B C : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C]

noncomputable def area_triangle_ABC (AB BC : ℝ) (angleB : ℕ) (hangle : angleB = 90) (hAB : AB = 30) (hBC : BC = 40) : ℝ :=
  1 / 2 * AB * BC

theorem area_of_right_triangle_ABC (AB BC : ℝ) (angleB : ℕ) (hangle : angleB = 90) 
  (hAB : AB = 30) (hBC : BC = 40) : 
  area_triangle_ABC AB BC angleB hangle hAB hBC = 600 :=
by
  sorry

end area_of_right_triangle_ABC_l2087_208770


namespace solve_system_and_compute_l2087_208791

-- Given system of equations
variables {x y : ℝ}
variables (h1 : 2 * x + y = 4) (h2 : x + 2 * y = 5)

-- Statement to prove
theorem solve_system_and_compute :
  (x - y = -1) ∧ (x + y = 3) ∧ ((1/3 * (x^2 - y^2)) * (x^2 - 2*x*y + y^2) = -1) :=
by
  sorry

end solve_system_and_compute_l2087_208791


namespace train_speed_is_85_kmh_l2087_208799

noncomputable def speed_of_train_in_kmh (length_of_train : ℝ) (time_to_cross : ℝ) (speed_of_man_kmh : ℝ) : ℝ :=
  let speed_of_man_mps := speed_of_man_kmh * 1000 / 3600
  let relative_speed_mps := length_of_train / time_to_cross
  let speed_of_train_mps := relative_speed_mps - speed_of_man_mps
  speed_of_train_mps * 3600 / 1000

theorem train_speed_is_85_kmh
  (length_of_train : ℝ)
  (time_to_cross : ℝ)
  (speed_of_man_kmh : ℝ)
  (h1 : length_of_train = 150)
  (h2 : time_to_cross = 6)
  (h3 : speed_of_man_kmh = 5) :
  speed_of_train_in_kmh length_of_train time_to_cross speed_of_man_kmh = 85 :=
by
  sorry

end train_speed_is_85_kmh_l2087_208799


namespace tank_width_problem_l2087_208786

noncomputable def tank_width (cost_per_sq_meter : ℚ) (total_cost : ℚ) (length depth : ℚ) : ℚ :=
  let total_cost_in_paise := total_cost * 100
  let total_area := total_cost_in_paise / cost_per_sq_meter
  let w := (total_area - (2 * length * depth) - (2 * depth * 6)) / (length + 2 * depth)
  w

theorem tank_width_problem :
  tank_width 55 409.20 25 6 = 12 := 
by 
  sorry

end tank_width_problem_l2087_208786


namespace interval_increase_for_k_eq_2_range_of_k_if_f_leq_0_l2087_208796

noncomputable def f (x k : ℝ) : ℝ := Real.log x - k * x + 1

theorem interval_increase_for_k_eq_2 :
  ∃ k : ℝ, k = 2 → 
  ∃ a b : ℝ, 0 < b ∧ b = 1 / 2 ∧
  (∀ x : ℝ, 0 < x ∧ x < 1 / 2 → (Real.log x - 2 * x + 1 < Real.log x - 2 * x + 1)) := 
sorry

theorem range_of_k_if_f_leq_0 :
  ∀ (k : ℝ), (∀ x : ℝ, 0 < x → Real.log x - k * x + 1 ≤ 0) →
  ∃ k_min : ℝ, k_min = 1 ∧ k ≥ k_min :=
sorry

end interval_increase_for_k_eq_2_range_of_k_if_f_leq_0_l2087_208796


namespace percentage_increase_in_rectangle_area_l2087_208795

theorem percentage_increase_in_rectangle_area (L W : ℝ) :
  (1.35 * 1.35 * L * W - L * W) / (L * W) * 100 = 82.25 :=
by sorry

end percentage_increase_in_rectangle_area_l2087_208795


namespace carpenter_needs_80_woodblocks_l2087_208714

-- Define the number of logs the carpenter currently has
def existing_logs : ℕ := 8

-- Define the number of woodblocks each log can produce
def woodblocks_per_log : ℕ := 5

-- Define the number of additional logs needed
def additional_logs : ℕ := 8

-- Calculate the total number of woodblocks needed
def total_woodblocks_needed : ℕ := 
  (existing_logs * woodblocks_per_log) + (additional_logs * woodblocks_per_log)

-- Prove that the total number of woodblocks needed is 80
theorem carpenter_needs_80_woodblocks : total_woodblocks_needed = 80 := by
  sorry

end carpenter_needs_80_woodblocks_l2087_208714


namespace smallest_d_l2087_208723

theorem smallest_d (d : ℕ) (h : 3150 * d = k ^ 2) : d = 14 :=
by
  -- assuming the condition: 3150 = 2 * 3 * 5^2 * 7
  have h_factorization : 3150 = 2 * 3 * 5^2 * 7 := by sorry
  -- based on the computation and verification, the smallest d that satisfies the condition is 14
  sorry

end smallest_d_l2087_208723


namespace rectangles_in_grid_l2087_208787

-- Define a function that calculates the number of rectangles formed
def number_of_rectangles (n m : ℕ) : ℕ :=
  ((m + 2) * (m + 1) * (n + 2) * (n + 1)) / 4

-- Prove that the number_of_rectangles function correctly calculates the number of rectangles given n and m 
theorem rectangles_in_grid (n m : ℕ) :
  number_of_rectangles n m = ((m + 2) * (m + 1) * (n + 2) * (n + 1)) / 4 := 
by
  sorry

end rectangles_in_grid_l2087_208787


namespace optimal_discount_savings_l2087_208722

theorem optimal_discount_savings : 
  let total_amount := 15000
  let discount1 := 0.30
  let discount2 := 0.15
  let single_discount := 0.40
  let two_successive_discounts := total_amount * (1 - discount1) * (1 - discount2)
  let one_single_discount := total_amount * (1 - single_discount)
  one_single_discount - two_successive_discounts = 75 :=
by
  sorry

end optimal_discount_savings_l2087_208722


namespace solve_quadratic_equation_l2087_208709

theorem solve_quadratic_equation (x : ℝ) :
  (x^2 - 2 * x - 5 = 0) ↔ (x = 1 + Real.sqrt 6 ∨ x = 1 - Real.sqrt 6) := 
sorry

end solve_quadratic_equation_l2087_208709


namespace regular_polygon_properties_l2087_208778

theorem regular_polygon_properties
  (exterior_angle : ℝ := 18) :
  (∃ (n : ℕ), n = 20) ∧ (∃ (interior_angle : ℝ), interior_angle = 162) := 
by
  sorry

end regular_polygon_properties_l2087_208778


namespace triangle_area_l2087_208755

theorem triangle_area (f : ℝ → ℝ) (x1 x2 yIntercept base height area : ℝ)
  (h1 : ∀ x, f x = (x - 4)^2 * (x + 3))
  (h2 : f 0 = yIntercept)
  (h3 : x1 = -3)
  (h4 : x2 = 4)
  (h5 : base = x2 - x1)
  (h6 : height = yIntercept)
  (h7 : area = 1/2 * base * height) :
  area = 168 := sorry

end triangle_area_l2087_208755


namespace original_population_l2087_208720

theorem original_population (p: ℝ) :
  (p + 1500) * 0.85 = p - 45 -> p = 8800 :=
by
  sorry

end original_population_l2087_208720


namespace distinct_arrangements_balloon_l2087_208738

noncomputable def totalPermutations (n nl no : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial nl * Nat.factorial no)

theorem distinct_arrangements_balloon :
  totalPermutations 7 2 2 = 1260 := by 
  sorry

end distinct_arrangements_balloon_l2087_208738


namespace find_unknown_number_l2087_208737

theorem find_unknown_number (y : ℝ) (h : 25 / y = 80 / 100) : y = 31.25 :=
sorry

end find_unknown_number_l2087_208737


namespace units_digit_of_subtraction_is_seven_l2087_208761

theorem units_digit_of_subtraction_is_seven (a b c: ℕ) (h1: a = c + 3) (h2: b = 2 * c) :
  let original_number := 100 * a + 10 * b + c
  let reversed_number := 100 * c + 10 * b + a
  let result := original_number - reversed_number
  result % 10 = 7 :=
by
  let original_number := 100 * a + 10 * b + c
  let reversed_number := 100 * c + 10 * b + a
  let result := original_number - reversed_number
  sorry

end units_digit_of_subtraction_is_seven_l2087_208761


namespace symmetric_about_one_symmetric_about_two_l2087_208725

-- Part 1
theorem symmetric_about_one (rational_num_x : ℚ) (rational_num_r : ℚ) 
(h1 : 3 - 1 = 1 - rational_num_x) (hr1 : r = 3 - 1): 
  rational_num_x = -1 ∧ rational_num_r = 2 := 
by
  sorry

-- Part 2
theorem symmetric_about_two (a b : ℚ) (symmetric_radius : ℚ) 
(h2 : (a + b) / 2 = 2) (condition : |a| = 2 * |b|) : 
  symmetric_radius = 2 / 3 ∨ symmetric_radius = 6 := 
by
  sorry

end symmetric_about_one_symmetric_about_two_l2087_208725


namespace find_set_B_l2087_208728

open Set

variable (U : Finset ℕ) (A B : Finset ℕ)
variable (hU : U = {1, 2, 3, 4, 5, 6, 7})
variable (h1 : (U \ (A ∪ B)) = {1, 3})
variable (h2 : A ∩ (U \ B) = {2, 5})

theorem find_set_B : B = {4, 6, 7} := by
  sorry

end find_set_B_l2087_208728


namespace sum_of_fractions_eq_two_l2087_208750

theorem sum_of_fractions_eq_two : 
  (1 / 2) + (2 / 4) + (4 / 8) + (8 / 16) = 2 :=
by sorry

end sum_of_fractions_eq_two_l2087_208750


namespace negation_of_exists_l2087_208721

theorem negation_of_exists (P : ℝ → Prop) :
  (¬ ∃ x : ℝ, x^2 - x + 1 < 0) ↔ (∀ x : ℝ, x^2 - x + 1 ≥ 0) :=
by sorry

end negation_of_exists_l2087_208721


namespace second_train_speed_l2087_208732

theorem second_train_speed (d : ℝ) (s₁ : ℝ) (t₁ : ℝ) (t₂ : ℝ) (meet_time : ℝ) (total_distance : ℝ) :
  d = 110 ∧ s₁ = 20 ∧ t₁ = 3 ∧ t₂ = 2 ∧ meet_time = 10 ∧ total_distance = d →
  60 + 2 * (total_distance - 60) / 2 = 110 →
  (total_distance - 60) / 2 = 25 :=
by
  intro h1 h2
  sorry

end second_train_speed_l2087_208732


namespace hypotenuse_length_l2087_208708

theorem hypotenuse_length (a b c : ℝ) 
  (h1 : a^2 + b^2 = c^2) 
  (h2 : a^2 + b^2 + c^2 = 1800) : 
  c = 30 :=
sorry

end hypotenuse_length_l2087_208708


namespace sum_of_values_l2087_208779

theorem sum_of_values (N : ℝ) (h : N * (N + 4) = 8) : N + (4 - N - 8 / N) = -4 := 
sorry

end sum_of_values_l2087_208779


namespace parallel_line_slope_l2087_208705

theorem parallel_line_slope (a b c : ℝ) (m : ℝ) :
  (5 * a + 10 * b = -35) →
  (∃ m : ℝ, b = m * a + c) →
  m = -1/2 :=
by sorry

end parallel_line_slope_l2087_208705


namespace free_throw_percentage_l2087_208706

theorem free_throw_percentage (p : ℚ) :
  (1 - p)^2 + 2 * p * (1 - p) = 16 / 25 → p = 3 / 5 :=
by
  sorry

end free_throw_percentage_l2087_208706


namespace sum_of_midpoint_coords_l2087_208771

theorem sum_of_midpoint_coords (x1 y1 x2 y2 : ℝ) (hx1 : x1 = 3) (hy1 : y1 = 5) (hx2 : x2 = 11) (hy2 : y2 = 21) :
  ((x1 + x2) / 2 + (y1 + y2) / 2) = 20 :=
by
  sorry

end sum_of_midpoint_coords_l2087_208771


namespace area_of_quadrilateral_AXYD_l2087_208794

open Real

noncomputable def area_quadrilateral_AXYD: ℝ :=
  let A := (0, 0)
  let B := (20, 0)
  let C := (20, 12)
  let D := (0, 12)
  let Z := (20, 30)
  let E := (6, 6)
  let X := (2.5, 0)
  let Y := (9.5, 12)
  let base1 := (B.1 - X.1)  -- Length from B to X
  let base2 := (Y.1 - A.1)  -- Length from D to Y
  let height := (C.2 - A.2) -- Height common for both bases
  (base1 + base2) * height / 2

theorem area_of_quadrilateral_AXYD : area_quadrilateral_AXYD = 72 :=
by
  sorry

end area_of_quadrilateral_AXYD_l2087_208794


namespace tiles_needed_l2087_208752

theorem tiles_needed (A_classroom : ℝ) (side_length_tile : ℝ) (H_classroom : A_classroom = 56) (H_side_length : side_length_tile = 0.4) :
  A_classroom / (side_length_tile * side_length_tile) = 350 :=
by
  sorry

end tiles_needed_l2087_208752


namespace range_of_a_l2087_208747

theorem range_of_a (f : ℝ → ℝ)
  (h_odd : ∀ x, f (-x) = -f x)
  (h_mono : ∀ ⦃a b⦄, 0 ≤ a → a ≤ b → f a ≤ f b)
  (h_cond : ∀ a, f a < f (2 * a - 1) → a > 1) :
  ∀ a, f a < f (2 * a - 1) → 1 < a := 
sorry

end range_of_a_l2087_208747


namespace range_of_a_l2087_208758

theorem range_of_a (x y a : ℝ) (h1 : x < y) (h2 : (a - 3) * x > (a - 3) * y) : a < 3 :=
sorry

end range_of_a_l2087_208758


namespace total_sum_lent_l2087_208704

theorem total_sum_lent (x : ℝ) (second_part : ℝ) (total_sum : ℝ)
  (h1 : second_part = 1648)
  (h2 : (x * 3 / 100 * 8) = (second_part * 5 / 100 * 3))
  (h3 : total_sum = x + second_part) :
  total_sum = 2678 := 
  sorry

end total_sum_lent_l2087_208704


namespace minimum_value_f_x_l2087_208781

theorem minimum_value_f_x (x : ℝ) (h : 1 < x) : 
  x + (1 / (x - 1)) ≥ 3 :=
sorry

end minimum_value_f_x_l2087_208781


namespace find_X_d_minus_Y_d_l2087_208710

def digits_in_base_d (X Y d : ℕ) : Prop :=
  2 * d * X + X + Y = d^2 + 8 * d + 2 

theorem find_X_d_minus_Y_d (d X Y : ℕ) (h1 : digits_in_base_d X Y d) (h2 : d > 8) : X - Y = d - 8 :=
by 
  sorry

end find_X_d_minus_Y_d_l2087_208710


namespace lower_rent_amount_l2087_208707

-- Define the conditions and proof goal
variable (T R : ℕ)
variable (L : ℕ)

-- Condition 1: Total rent is $1000
def total_rent (T R : ℕ) (L : ℕ) := 60 * R + L * (T - R)

-- Condition 2: Reduction by 20% when 10 rooms are swapped
def reduced_rent (T R : ℕ) (L : ℕ) := 60 * (R - 10) + L * (T - R + 10)

-- Proof that the lower rent amount is $40 given the conditions
theorem lower_rent_amount (h1 : total_rent T R L = 1000)
                         (h2 : reduced_rent T R L = 800) : L = 40 :=
by
  sorry

end lower_rent_amount_l2087_208707


namespace arithmetic_geometric_sequence_l2087_208700

-- Let {a_n} be an arithmetic sequence
-- And let a_1, a_2, a_3 form a geometric sequence
-- Given that a_5 = 1, we aim to prove that a_10 = 1
theorem arithmetic_geometric_sequence (a : ℕ → ℝ)
  (h_arith : ∀ n, a (n + 1) - a n = a 2 - a 1)
  (h_geom : a 1 * a 3 = (a 2) ^ 2)
  (h_a5 : a 5 = 1) :
  a 10 = 1 :=
sorry

end arithmetic_geometric_sequence_l2087_208700


namespace total_cans_in_display_l2087_208762

-- Definitions and conditions
def first_term : ℕ := 30
def second_term : ℕ := 27
def nth_term : ℕ := 3
def common_difference : ℕ := second_term - first_term

-- Statement of the problem
theorem total_cans_in_display : 
  ∃ (n : ℕ), nth_term = first_term + (n - 1) * common_difference ∧
  (2 * 165 = n * (first_term + nth_term)) :=
by
  sorry

end total_cans_in_display_l2087_208762


namespace largest_angle_of_obtuse_isosceles_triangle_l2087_208733

variables (X Y Z : ℝ)

def is_triangle (X Y Z : ℝ) : Prop := X + Y + Z = 180
def is_isosceles_triangle (X Y : ℝ) : Prop := X = Y
def is_obtuse_triangle (X Y Z : ℝ) : Prop := X > 90 ∨ Y > 90 ∨ Z > 90

theorem largest_angle_of_obtuse_isosceles_triangle
  (X Y Z : ℝ)
  (h1 : is_triangle X Y Z)
  (h2 : is_isosceles_triangle X Y)
  (h3 : X = 30)
  (h4 : is_obtuse_triangle X Y Z) :
  Z = 120 :=
sorry

end largest_angle_of_obtuse_isosceles_triangle_l2087_208733


namespace fraction_girls_at_meet_l2087_208718

-- Define the conditions of the problem
def numStudentsMaplewood : ℕ := 300
def ratioBoysGirlsMaplewood : ℕ × ℕ := (3, 2)
def numStudentsRiverview : ℕ := 240
def ratioBoysGirlsRiverview : ℕ × ℕ := (3, 5)

-- Define the combined number of students and number of girls
def totalStudentsMaplewood := numStudentsMaplewood
def totalStudentsRiverview := numStudentsRiverview

def numGirlsMaplewood : ℕ :=
  let (b, g) := ratioBoysGirlsMaplewood
  (totalStudentsMaplewood * g) / (b + g)

def numGirlsRiverview : ℕ :=
  let (b, g) := ratioBoysGirlsRiverview
  (totalStudentsRiverview * g) / (b + g)

def totalGirls := numGirlsMaplewood + numGirlsRiverview
def totalStudents := totalStudentsMaplewood + totalStudentsRiverview

-- Formalize the actual proof statement
theorem fraction_girls_at_meet : 
  (totalGirls : ℚ) / totalStudents = 1 / 2 := by
  sorry

end fraction_girls_at_meet_l2087_208718


namespace trisect_chord_exists_l2087_208746

noncomputable def distance (O P : Point) : ℝ := sorry
def trisect (P : Point) (A B : Point) : Prop := 2 * (distance A P) = distance P B

-- Main theorem based on the given conditions and conclusions
theorem trisect_chord_exists (O P : Point) (r : ℝ) (hP_in_circle : distance O P < r) :
  (∃ A B : Point, trisect P A B) ↔ 
  (distance O P > r / 3 ∨ distance O P = r / 3) :=
by
  sorry

end trisect_chord_exists_l2087_208746


namespace initial_decaf_percentage_l2087_208744

theorem initial_decaf_percentage 
  (x : ℝ)
  (h1 : 0 ≤ x) (h2 : x ≤ 100) 
  (h3 : (x / 100 * 400) + 60 = 220) :
  x = 40 :=
by sorry

end initial_decaf_percentage_l2087_208744


namespace original_price_l2087_208766

theorem original_price (x : ℝ) (h1 : x > 0) (h2 : 1.12 * x - x = 270) : x = 2250 :=
by
  sorry

end original_price_l2087_208766


namespace find_cost_price_l2087_208763

variable (CP : ℝ)

def SP1 : ℝ := 0.80 * CP
def SP2 : ℝ := 1.06 * CP

axiom cond1 : SP2 - SP1 = 520

theorem find_cost_price : CP = 2000 :=
by
  sorry

end find_cost_price_l2087_208763


namespace find_p_value_l2087_208743

theorem find_p_value (D E F : ℚ) (α β : ℚ)
  (h₁: D ≠ 0) 
  (h₂: E^2 - 4*D*F ≥ 0) 
  (hαβ: D * (α^2 + β^2) + E * (α + β) + 2*F = 2*D^2 - E^2) :
  ∃ p : ℚ, (p = (2*D*F - E^2 - 2*D^2) / D^2) :=
sorry

end find_p_value_l2087_208743


namespace children_working_initially_l2087_208797

theorem children_working_initially (W C : ℝ) (n : ℕ) 
  (h1 : 10 * W = 1 / 5) 
  (h2 : n * C = 1 / 10) 
  (h3 : 5 * W + 10 * C = 1 / 5) : 
  n = 10 :=
by
  sorry

end children_working_initially_l2087_208797


namespace M_less_equal_fraction_M_M_greater_equal_fraction_M_M_less_equal_sum_M_l2087_208769

noncomputable def M : ℕ → ℕ → ℕ → ℝ := sorry

theorem M_less_equal_fraction_M (n k h : ℕ) : 
  M n k h ≤ (n / h) * M (n-1) (k-1) (h-1) :=
sorry

theorem M_greater_equal_fraction_M (n k h : ℕ) : 
  M n k h ≥ (n / (n - h)) * M (n-1) k k :=
sorry

theorem M_less_equal_sum_M (n k h : ℕ) : 
  M n k h ≤ M (n-1) (k-1) (h-1) + M (n-1) k h :=
sorry

end M_less_equal_fraction_M_M_greater_equal_fraction_M_M_less_equal_sum_M_l2087_208769


namespace percentage_pure_acid_l2087_208719

theorem percentage_pure_acid (volume_pure_acid total_volume: ℝ) (h1 : volume_pure_acid = 1.4) (h2 : total_volume = 4) : 
  (volume_pure_acid / total_volume) * 100 = 35 := 
by
  -- Given metric volumes of pure acid and total solution, we need to prove the percentage 
  -- Here, we assert the conditions and conclude the result
  sorry

end percentage_pure_acid_l2087_208719


namespace factor_expression_l2087_208739

theorem factor_expression (x : ℝ) : (x * (x + 3) + 2 * (x + 3)) = (x + 2) * (x + 3) :=
by
  sorry

end factor_expression_l2087_208739


namespace log_conditions_l2087_208757

noncomputable def log_base (a b : ℝ) : ℝ := Real.log b / Real.log a

theorem log_conditions (m n : ℝ) (h₁ : log_base m 9 < log_base n 9)
  (h₂ : log_base n 9 < 0) : 0 < m ∧ m < n ∧ n < 1 :=
sorry

end log_conditions_l2087_208757


namespace all_terms_perfect_squares_l2087_208753

def seq_x : ℕ → ℕ
| 0       => 1
| 1       => 1
| (n + 2) => 14 * seq_x (n + 1) - seq_x n - 4

theorem all_terms_perfect_squares : ∀ n, ∃ k, seq_x n = k^2 :=
by
  sorry

end all_terms_perfect_squares_l2087_208753


namespace proposition_statementC_l2087_208768

-- Definitions of each statement
def statementA := "Draw a parallel line to line AB"
def statementB := "Take a point C on segment AB"
def statementC := "The complement of equal angles are equal"
def statementD := "Is the perpendicular segment the shortest?"

-- Proving that among the statements A, B, C, and D, statement C is the proposition
theorem proposition_statementC : 
  (statementC = "The complement of equal angles are equal") :=
by
  -- We assume it directly from the equivalence given in the problem statement
  sorry

end proposition_statementC_l2087_208768


namespace locus_of_points_is_straight_line_l2087_208740

theorem locus_of_points_is_straight_line 
  (a R1 R2 : ℝ) 
  (h_nonzero_a : a ≠ 0)
  (h_positive_R1 : R1 > 0)
  (h_positive_R2 : R2 > 0) :
  ∃ x : ℝ, ∀ (y : ℝ),
  ((x + a)^2 + y^2 - R1^2 = (x - a)^2 + y^2 - R2^2) ↔ 
  x = (R1^2 - R2^2) / (4 * a) :=
by
  sorry

end locus_of_points_is_straight_line_l2087_208740


namespace empty_one_container_l2087_208730

theorem empty_one_container (a b c : ℕ) :
  ∃ a' b' c', (a' = 0 ∨ b' = 0 ∨ c' = 0) ∧
    (a' = a ∧ b' = b ∧ c' = c ∨
     (a' ≤ a ∧ b' ≤ b ∧ c' ≤ c ∧ (a + b + c = a' + b' + c')) ∧
     (∀ i j, i ≠ j → (i = 1 ∨ i = 2 ∨ i = 3) →
              (j = 1 ∨ j = 2 ∨ j = 3) →
              (if i = 1 then (if j = 2 then a' = a - a ∨ a' = a else (if j = 3 then a' = a - a ∨ a' = a else false))
               else if i = 2 then (if j = 1 then b' = b - b ∨ b' = b else (if j = 3 then b' = b - b ∨ b' = b else false))
               else (if j = 1 then c' = c - c ∨ c' = c else (if j = 2 then c' = c - c ∨ c' = c else false))))) :=
by
  sorry

end empty_one_container_l2087_208730


namespace ramsey_six_vertices_monochromatic_quadrilateral_l2087_208715

theorem ramsey_six_vertices_monochromatic_quadrilateral :
  ∀ (V : Type) (E : V → V → Prop), (∀ x y : V, x ≠ y → E x y ∨ ¬ E x y) →
  ∃ (u v w x : V), u ≠ v ∧ v ≠ w ∧ w ≠ x ∧ x ≠ u ∧ (E u v = E v w ∧ E v w = E w x ∧ E w x = E x u) :=
by sorry

end ramsey_six_vertices_monochromatic_quadrilateral_l2087_208715


namespace eval_operations_l2087_208760

def star (a b : ℤ) : ℤ := a + b - 1
def hash (a b : ℤ) : ℤ := a * b - 1

theorem eval_operations : star (star 6 8) (hash 3 5) = 26 := by
  sorry

end eval_operations_l2087_208760


namespace abscissa_of_tangent_point_l2087_208765

theorem abscissa_of_tangent_point (a : ℝ) (f : ℝ → ℝ)
  (h : ∀ x, f x = Real.exp x + a * Real.exp (-x))
  (h_odd : ∀ x, (D^[2] f x) = - (D^[2] f (-x)))
  (slope_cond : ∀ x, (D f x) = 3 / 2) : 
  ∃ x ∈ Set.Ioo (-Real.log 2) (Real.log 2), x = Real.log 2 :=
by
  sorry

end abscissa_of_tangent_point_l2087_208765


namespace product_of_terms_form_l2087_208756

theorem product_of_terms_form 
  (a b c d : ℝ) 
  (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : 0 ≤ d) :
  ∃ p q : ℝ, 
    (a + b * Real.sqrt 5) * (c + d * Real.sqrt 5) = p + q * Real.sqrt 5 
    ∧ 0 ≤ p 
    ∧ 0 ≤ q := 
by
  let p := a * c + 5 * b * d
  let q := a * d + b * c
  use p, q
  sorry

end product_of_terms_form_l2087_208756


namespace part_a_part_b_l2087_208790

theorem part_a (x : ℝ) (n : ℕ) (hx_pos : 0 < x) (hx_ne_one : x ≠ 1) (hn_pos : 0 < n) :
  Real.log x < n * (x ^ (1 / n) - 1) ∧ n * (x ^ (1 / n) - 1) < (x ^ (1 / n)) * Real.log x := sorry

theorem part_b (x : ℝ) (hx_pos : 0 < x) (hx_ne_one : x ≠ 1) :
  (Real.log x) = (Real.log x) := sorry

end part_a_part_b_l2087_208790


namespace total_days_to_finish_job_l2087_208774

noncomputable def workers_job_completion
  (initial_workers : ℕ)
  (additional_workers : ℕ)
  (initial_days : ℕ)
  (total_days : ℕ)
  (work_completion_days : ℕ)
  (remaining_work : ℝ)
  (additional_days_needed : ℝ)
  : ℝ :=
  initial_days + additional_days_needed

theorem total_days_to_finish_job
  (initial_workers : ℕ := 6)
  (additional_workers : ℕ := 4)
  (initial_days : ℕ := 3)
  (total_days : ℕ := 8)
  (work_completion_days : ℕ := 8)
  : workers_job_completion initial_workers additional_workers initial_days total_days work_completion_days (1 - (initial_days : ℝ) / work_completion_days) (remaining_work / (((initial_workers + additional_workers) : ℝ) / work_completion_days)) = 3.5 :=
  sorry

end total_days_to_finish_job_l2087_208774


namespace option_D_min_value_is_2_l2087_208767

noncomputable def funcD (x : ℝ) : ℝ :=
  (x^2 + 2) / Real.sqrt (x^2 + 1)

theorem option_D_min_value_is_2 :
  ∃ x : ℝ, funcD x = 2 :=
sorry

end option_D_min_value_is_2_l2087_208767


namespace ordered_pair_exists_l2087_208726

theorem ordered_pair_exists :
  ∃ p q : ℝ, 
  (3 + 8 * p = 2 - 3 * q) ∧ (-4 - 6 * p = -3 + 4 * q) ∧ (p = -1/14) ∧ (q = -1/7) :=
by
  sorry

end ordered_pair_exists_l2087_208726


namespace amit_work_days_l2087_208759

theorem amit_work_days (x : ℕ) (h : 2 * (1 / x : ℚ) + 16 * (1 / 20 : ℚ) = 1) : x = 10 :=
by {
  sorry
}

end amit_work_days_l2087_208759


namespace survey_steps_correct_l2087_208783

theorem survey_steps_correct :
  ∀ steps : (ℕ → ℕ), (steps 1 = 2) → (steps 2 = 4) → (steps 3 = 3) → (steps 4 = 1) → True :=
by
  intros steps h1 h2 h3 h4
  exact sorry

end survey_steps_correct_l2087_208783


namespace sum_even_then_diff_even_sum_odd_then_diff_odd_l2087_208793

theorem sum_even_then_diff_even (a b : ℤ) (h : (a + b) % 2 = 0) : (a - b) % 2 = 0 := by
  sorry

theorem sum_odd_then_diff_odd (a b : ℤ) (h : (a + b) % 2 = 1) : (a - b) % 2 = 1 := by
  sorry

end sum_even_then_diff_even_sum_odd_then_diff_odd_l2087_208793


namespace total_time_in_pool_is_29_minutes_l2087_208754

noncomputable def calculate_total_time_in_pool : ℝ :=
  let jerry := 3             -- Jerry's time in minutes
  let elaine := 2 * jerry    -- Elaine's time in minutes
  let george := elaine / 3    -- George's time in minutes
  let susan := 150 / 60      -- Susan's time in minutes
  let puddy := elaine / 2    -- Puddy's time in minutes
  let frank := elaine / 2    -- Frank's time in minutes
  let estelle := 0.1 * 60    -- Estelle's time in minutes
  let total_excluding_newman := jerry + elaine + george + susan + puddy + frank + estelle
  let newman := total_excluding_newman / 7   -- Newman's average time
  total_excluding_newman + newman

theorem total_time_in_pool_is_29_minutes : 
  calculate_total_time_in_pool = 29 :=
by
  sorry

end total_time_in_pool_is_29_minutes_l2087_208754


namespace longest_sticks_triangle_shortest_sticks_not_triangle_l2087_208713

-- Define the lengths of the six sticks in descending order
variables {a1 a2 a3 a4 a5 a6 : ℝ}

-- Assuming the conditions
axiom h1 : a1 ≥ a2
axiom h2 : a2 ≥ a3
axiom h3 : a3 ≥ a4
axiom h4 : a4 ≥ a5
axiom h5 : a5 ≥ a6
axiom h6 : a1 + a2 > a3

-- Proof problem 1: It is always possible to form a triangle from the three longest sticks.
theorem longest_sticks_triangle : a1 < a2 + a3 := by sorry

-- Assuming an additional condition for proof problem 2
axiom two_triangles_formed : ∃ b1 b2 b3 b4 b5 b6: ℝ, 
  ((b1 + b2 > b3 ∧ b1 + b3 > b2 ∧ b2 + b3 > b1) ∧
   (b4 + b5 > b6 ∧ b4 + b6 > b5 ∧ b5 + b6 > b4 ∧ 
    a1 = b1 ∧ a2 = b2 ∧ a3 = b3 ∧ a4 = b4 ∧ a5 = b5 ∧ a6 = b6))

-- Proof problem 2: It is not always possible to form a triangle from the three shortest sticks.
theorem shortest_sticks_not_triangle : ¬(a4 < a5 + a6 ∧ a5 < a4 + a6 ∧ a6 < a4 + a5) := by sorry

end longest_sticks_triangle_shortest_sticks_not_triangle_l2087_208713


namespace yogurt_amount_l2087_208711

namespace SmoothieProblem

def strawberries := 0.2 -- cups
def orange_juice := 0.2 -- cups
def total_ingredients := 0.5 -- cups

def yogurt_used := total_ingredients - (strawberries + orange_juice)

theorem yogurt_amount : yogurt_used = 0.1 :=
by
  unfold yogurt_used strawberries orange_juice total_ingredients
  norm_num
  sorry  -- Proof can be filled in as needed

end SmoothieProblem

end yogurt_amount_l2087_208711


namespace graph_t_intersects_x_axis_exists_integer_a_with_integer_points_on_x_axis_intersection_l2087_208741

open Real

def function_y (a x : ℝ) : ℝ := (4 * a + 2) * x^2 + (9 - 6 * a) * x - 4 * a + 4

theorem graph_t_intersects_x_axis (a : ℝ) : ∃ x : ℝ, function_y a x = 0 :=
by sorry

theorem exists_integer_a_with_integer_points_on_x_axis_intersection :
  ∃ (a : ℤ), 
  (∀ x : ℝ, (function_y a x = 0) → ∃ (x_int : ℤ), x = x_int) ∧ 
  (a = -2 ∨ a = -1 ∨ a = 0 ∨ a = 1) :=
by sorry

end graph_t_intersects_x_axis_exists_integer_a_with_integer_points_on_x_axis_intersection_l2087_208741


namespace find_number_l2087_208727

theorem find_number (x : ℤ) (N : ℤ) (h1 : 3 * x = (N - x) + 18) (hx : x = 11) : N = 26 :=
by
  sorry

end find_number_l2087_208727


namespace squares_of_roots_equation_l2087_208736

theorem squares_of_roots_equation (a b x : ℂ) 
  (h : ab * x^2 - (a + b) * x + 1 = 0) : 
  a^2 * b^2 * x^2 - (a^2 + b^2) * x + 1 = 0 :=
sorry

end squares_of_roots_equation_l2087_208736


namespace patsy_deviled_eggs_l2087_208735

-- Definitions based on given problem conditions
def guests : ℕ := 30
def appetizers_per_guest : ℕ := 6
def total_appetizers_needed : ℕ := appetizers_per_guest * guests
def pigs_in_blanket : ℕ := 2
def kebabs : ℕ := 2
def additional_appetizers_needed (already_planned : ℕ) : ℕ := 8 + already_planned
def already_planned_appetizers : ℕ := pigs_in_blanket + kebabs
def total_appetizers_planned : ℕ := additional_appetizers_needed already_planned_appetizers

-- The proof problem statement
theorem patsy_deviled_eggs : total_appetizers_needed = total_appetizers_planned * 12 → 
                            total_appetizers_planned = already_planned_appetizers + 8 →
                            (total_appetizers_planned - already_planned_appetizers) = 8 :=
by
  sorry

end patsy_deviled_eggs_l2087_208735


namespace new_class_mean_l2087_208792

theorem new_class_mean {X Y : ℕ} {mean_a mean_b : ℚ}
  (hx : X = 30) (hy : Y = 6) 
  (hmean_a : mean_a = 72) (hmean_b : mean_b = 78) :
  (X * mean_a + Y * mean_b) / (X + Y) = 73 := 
by 
  sorry

end new_class_mean_l2087_208792


namespace speed_of_sound_l2087_208772

theorem speed_of_sound (d₁ d₂ t : ℝ) (speed_car : ℝ) (speed_km_hr_to_m_s : ℝ) :
  d₁ = 1200 ∧ speed_car = 108 ∧ speed_km_hr_to_m_s = (speed_car * 1000 / 3600) ∧ t = 3.9669421487603307 →
  (d₁ + speed_km_hr_to_m_s * t) / t = 332.59 :=
by sorry

end speed_of_sound_l2087_208772


namespace common_factor_is_n_plus_1_l2087_208742

def polynomial1 (n : ℕ) : ℕ := n^2 - 1
def polynomial2 (n : ℕ) : ℕ := n^2 + n

theorem common_factor_is_n_plus_1 (n : ℕ) : 
  ∃ (d : ℕ), d ∣ polynomial1 n ∧ d ∣ polynomial2 n ∧ d = n + 1 := by
  sorry

end common_factor_is_n_plus_1_l2087_208742


namespace three_digit_number_exists_l2087_208731

theorem three_digit_number_exists : 
  ∃ (x y z : ℕ), 
  (1 ≤ x ∧ x ≤ 9) ∧ (1 ≤ y ∧ y ≤ 9) ∧ (0 ≤ z ∧ z ≤ 9) ∧ 
  (100 * x + 10 * z + y + 1 = 2 * (100 * y + 10 * z + x)) ∧ 
  (100 * x + 10 * z + y = 793) :=
by
  sorry

end three_digit_number_exists_l2087_208731


namespace quadratic_inequality_solution_l2087_208701

theorem quadratic_inequality_solution (a b c : ℝ) (h1 : a < 0)
  (h2 : ∀ x : ℝ, -2 < x ∧ x < 3 ↔ ax^2 + bx + c > 0) :
  ∀ x : ℝ, -3 < x ∧ x < 2 ↔ ax^2 - bx + c > 0 := 
sorry

end quadratic_inequality_solution_l2087_208701


namespace bad_iff_prime_l2087_208734

def a_n (n : ℕ) : ℕ := (2 * n)^2 + 1

def is_bad (n : ℕ) : Prop :=
  ¬∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ a_n n = a^2 + b^2

theorem bad_iff_prime (n : ℕ) : is_bad n ↔ Nat.Prime (a_n n) :=
by
  sorry

end bad_iff_prime_l2087_208734


namespace Sum_a2_a3_a7_l2087_208784

-- Definitions from the conditions
variable {a : ℕ → ℝ} -- Define the arithmetic sequence as a function from natural numbers to real numbers
variable {S : ℕ → ℝ} -- Define the sum of the first n terms as a function from natural numbers to real numbers

-- Given conditions
axiom Sn_formula : ∀ n : ℕ, S n = (n / 2) * (2 * a 1 + (n - 1) * (a 2 - a 1))
axiom S7_eq_42 : S 7 = 42

theorem Sum_a2_a3_a7 :
  a 2 + a 3 + a 7 = 18 :=
sorry

end Sum_a2_a3_a7_l2087_208784


namespace bird_average_l2087_208716

theorem bird_average (a b c : ℤ) (h1 : a = 7) (h2 : b = 11) (h3 : c = 9) :
  (a + b + c) / 3 = 9 :=
by
  sorry

end bird_average_l2087_208716


namespace gcd_pow_of_subtraction_l2087_208773

noncomputable def m : ℕ := 2^2100 - 1
noncomputable def n : ℕ := 2^1950 - 1

theorem gcd_pow_of_subtraction : Nat.gcd m n = 2^150 - 1 :=
by
  -- To be proven
  sorry

end gcd_pow_of_subtraction_l2087_208773


namespace factorable_polynomial_with_integer_coeffs_l2087_208712

theorem factorable_polynomial_with_integer_coeffs (m : ℤ) : 
  ∃ A B C D E F : ℤ, 
  (A * D = 1) ∧ (B * E = 0) ∧ (A * E + B * D = 5) ∧ 
  (A * F + C * D = 1) ∧ (B * F + C * E = 2 * m) ∧ (C * F = -10) ↔ m = 5 := sorry

end factorable_polynomial_with_integer_coeffs_l2087_208712


namespace exponentiation_problem_l2087_208703

theorem exponentiation_problem :
  (-0.125 ^ 2003) * (-8 ^ 2004) = -8 := 
sorry

end exponentiation_problem_l2087_208703


namespace accelerations_l2087_208764

open Real

namespace Problem

variables (m M g : ℝ) (a1 a2 : ℝ)

theorem accelerations (mass_condition : 4 * m + M ≠ 0):
  (a1 = 2 * ((2 * m + M) * g) / (4 * m + M)) ∧
  (a2 = ((2 * m + M) * g) / (4 * m + M)) :=
sorry

end Problem

end accelerations_l2087_208764


namespace total_seeds_l2087_208788

-- Definitions and conditions
def Bom_seeds : ℕ := 300
def Gwi_seeds : ℕ := Bom_seeds + 40
def Yeon_seeds : ℕ := 3 * Gwi_seeds
def Eun_seeds : ℕ := 2 * Gwi_seeds

-- Theorem statement
theorem total_seeds : Bom_seeds + Gwi_seeds + Yeon_seeds + Eun_seeds = 2340 :=
by
  -- Skipping the proof steps with sorry
  sorry

end total_seeds_l2087_208788


namespace geometric_sequence_a3_l2087_208785

theorem geometric_sequence_a3 (a : ℕ → ℝ) (q : ℝ) (n : ℕ) (h1 : a 1 = 1) (h2 : a 4 = 8)
  (h3 : ∀ k : ℕ, a (k + 1) = a k * q) : a 3 = 4 :=
sorry

end geometric_sequence_a3_l2087_208785


namespace arithmetic_sequence_first_term_l2087_208749

theorem arithmetic_sequence_first_term (a d : ℚ) 
  (h1 : 30 * (2 * a + 59 * d) = 500) 
  (h2 : 30 * (2 * a + 179 * d) = 2900) : 
  a = -34 / 3 := 
sorry

end arithmetic_sequence_first_term_l2087_208749


namespace range_a_l2087_208748

theorem range_a (a : ℝ) : (∀ x, x > 0 → x^2 - a * x + 1 > 0) → -2 < a ∧ a < 2 := by
  sorry

end range_a_l2087_208748


namespace problem_l2087_208777

theorem problem (m : ℝ) (h : m + 1/m = 6) : m^2 + 1/m^2 + 3 = 37 :=
by
  sorry

end problem_l2087_208777


namespace eval_abc_l2087_208775

theorem eval_abc (a b c : ℚ) (h1 : a = 1 / 2) (h2 : b = 3 / 4) (h3 : c = 8) :
  a^3 * b^2 * c = 9 / 16 :=
by
  sorry

end eval_abc_l2087_208775


namespace value_of_f_g3_l2087_208702

def g (x : ℝ) : ℝ := 4 * x - 5
def f (x : ℝ) : ℝ := 6 * x + 11

theorem value_of_f_g3 : f (g 3) = 53 := by
  sorry

end value_of_f_g3_l2087_208702


namespace find_LN_l2087_208782

noncomputable def LM : ℝ := 9
noncomputable def sin_N : ℝ := 3 / 5
noncomputable def LN : ℝ := 15

theorem find_LN (h₁ : sin_N = 3 / 5) (h₂ : LM = 9) (h₃ : sin_N = LM / LN) : LN = 15 :=
by
  sorry

end find_LN_l2087_208782


namespace problem_statement_l2087_208789

def p (x : ℝ) : ℝ := x^2 - x + 1

theorem problem_statement (α : ℝ) (h : p (p (p (p α))) = 0) :
  (p α - 1) * p α * p (p α) * p (p (p α)) = -1 :=
by
  sorry

end problem_statement_l2087_208789


namespace jennifer_cards_left_l2087_208798

-- Define the initial number of cards and the number of cards eaten
def initial_cards : ℕ := 72
def eaten_cards : ℕ := 61

-- Define the final number of cards
def final_cards (initial_cards eaten_cards : ℕ) : ℕ :=
  initial_cards - eaten_cards

-- Proposition stating that Jennifer has 11 cards left
theorem jennifer_cards_left : final_cards initial_cards eaten_cards = 11 :=
by
  -- Proof here
  sorry

end jennifer_cards_left_l2087_208798


namespace first_digit_power_l2087_208729

theorem first_digit_power (n : ℕ) (h : ∃ k : ℕ, 7 * 10^k ≤ 2^n ∧ 2^n < 8 * 10^k) :
  (∃ k' : ℕ, 1 * 10^k' ≤ 5^n ∧ 5^n < 2 * 10^k') :=
sorry

end first_digit_power_l2087_208729


namespace find_a9_l2087_208717

variable (a : ℕ → ℝ)  -- Define a sequence a_n.

-- Define the conditions for the arithmetic sequence.
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m, a (n + 1) - a n = a (m + 1) - a m

variables (h_arith_seq : is_arithmetic_sequence a)
          (h_a3 : a 3 = 8)   -- Condition a_3 = 8
          (h_a6 : a 6 = 5)   -- Condition a_6 = 5 

-- State the theorem.
theorem find_a9 : a 9 = 2 := by
  sorry

end find_a9_l2087_208717


namespace perpendicular_lines_l2087_208751

theorem perpendicular_lines (a : ℝ) 
  (h1 : (3 : ℝ) * y + (2 : ℝ) * x - 6 = 0) 
  (h2 : (4 : ℝ) * y + a * x - 5 = 0) : 
  a = -6 :=
sorry

end perpendicular_lines_l2087_208751


namespace find_digits_for_divisibility_l2087_208780

theorem find_digits_for_divisibility (d1 d2 : ℕ) (h1 : d1 < 10) (h2 : d2 < 10) :
  (32 * 10^7 + d1 * 10^6 + 35717 * 10 + d2) % 72 = 0 →
  d1 = 2 ∧ d2 = 6 :=
by
  sorry

end find_digits_for_divisibility_l2087_208780


namespace weight_of_B_l2087_208745

-- Definitions for the weights of A, B, and C
variable (A B C : ℝ)

-- Conditions given in the problem
def condition1 := (A + B + C) / 3 = 45
def condition2 := (A + B) / 2 = 40
def condition3 := (B + C) / 2 = 43

-- The theorem to prove that B = 31 under the given conditions
theorem weight_of_B : condition1 A B C → condition2 A B → condition3 B C → B = 31 := by
  intros
  sorry

end weight_of_B_l2087_208745
