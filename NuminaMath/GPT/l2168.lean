import Mathlib

namespace tangent_line_to_parabola_parallel_l2168_216877

theorem tangent_line_to_parabola_parallel (m : ℝ) :
  ∀ (x y : ℝ), (y = x^2) → (2*x - y + m = 0 → m = -1) :=
by
  sorry

end tangent_line_to_parabola_parallel_l2168_216877


namespace brownies_in_pan_l2168_216896

theorem brownies_in_pan : 
    ∀ (pan_length pan_width brownie_length brownie_width : ℕ), 
    pan_length = 24 -> 
    pan_width = 20 -> 
    brownie_length = 3 -> 
    brownie_width = 2 -> 
    (pan_length * pan_width) / (brownie_length * brownie_width) = 80 := 
by
  intros pan_length pan_width brownie_length brownie_width h1 h2 h3 h4
  sorry

end brownies_in_pan_l2168_216896


namespace anya_kolya_apples_l2168_216841

theorem anya_kolya_apples (A K : ℕ) (h1 : A = (K * 100) / (A + K)) (h2 : K = (A * 100) / (A + K)) : A = 50 ∧ K = 50 :=
sorry

end anya_kolya_apples_l2168_216841


namespace tree_planting_equation_l2168_216822

theorem tree_planting_equation (x : ℝ) (h : x > 0) : 
  180 / x - 180 / (1.5 * x) = 2 :=
sorry

end tree_planting_equation_l2168_216822


namespace problem1_solution_problem2_solution_l2168_216826

-- Problem 1
theorem problem1_solution (x y : ℝ) (h1 : y = 2 * x - 3) (h2 : 2 * x + y = 5) : 
  x = 2 ∧ y = 1 :=
  sorry

-- Problem 2
theorem problem2_solution (x y : ℝ) (h1 : 3 * x + 4 * y = 5) (h2 : 5 * x - 2 * y = 17) : 
  x = 3 ∧ y = -1 :=
  sorry

end problem1_solution_problem2_solution_l2168_216826


namespace base_height_l2168_216801

-- Define the height of the sculpture and the combined height.
def sculpture_height : ℚ := 2 + 10 / 12
def total_height : ℚ := 3 + 2 / 3

-- We want to prove that the base height is 5/6 feet.
theorem base_height :
  total_height - sculpture_height = 5 / 6 :=
by
  sorry

end base_height_l2168_216801


namespace coconut_to_almond_ratio_l2168_216833

-- Conditions
def number_of_coconut_candles (C : ℕ) : Prop :=
  ∃ L A : ℕ, L = 2 * C ∧ A = 10

-- Question
theorem coconut_to_almond_ratio (C : ℕ) (h : number_of_coconut_candles C) :
  ∃ r : ℚ, r = C / 10 := by
  sorry

end coconut_to_almond_ratio_l2168_216833


namespace exists_x0_l2168_216849

noncomputable def f (x : Real) (a : Real) : Real :=
  Real.exp x - a * Real.sin x

theorem exists_x0 (a : Real) (h : a = 1) :
  ∃ x0 ∈ Set.Ioo (-Real.pi / 2) 0, 1 < f x0 a ∧ f x0 a < Real.sqrt 2 :=
  sorry

end exists_x0_l2168_216849


namespace amount_b_l2168_216862

-- Definitions of the conditions
variables (a b : ℚ) 

def condition1 : Prop := a + b = 1210
def condition2 : Prop := (2 / 3) * a = (1 / 2) * b

-- The theorem to prove
theorem amount_b (h₁ : condition1 a b) (h₂ : condition2 a b) : b = 691.43 :=
sorry

end amount_b_l2168_216862


namespace divisor_of_136_l2168_216820

theorem divisor_of_136 (d : ℕ) (h : 136 = 9 * d + 1) : d = 15 := 
by {
  -- Since the solution steps are skipped, we use sorry to indicate a placeholder.
  sorry
}

end divisor_of_136_l2168_216820


namespace no_infinite_subset_of_natural_numbers_l2168_216890

theorem no_infinite_subset_of_natural_numbers {
  S : Set ℕ 
} (hS_infinite : S.Infinite) :
  ¬ (∀ a b : ℕ, a ∈ S → b ∈ S → a^2 - a * b + b^2 ∣ (a * b)^2) :=
sorry

end no_infinite_subset_of_natural_numbers_l2168_216890


namespace minimum_candies_l2168_216869

theorem minimum_candies (students : ℕ) (N : ℕ) (k : ℕ) : 
  students = 25 → 
  N = 25 * k → 
  (∀ n, 1 ≤ n → n ≤ students → ∃ m, n * k + m ≤ N) → 
  600 ≤ N := 
by
  intros hs hn hd
  sorry

end minimum_candies_l2168_216869


namespace binom_product_l2168_216834

open Nat

-- Define binomial coefficient
def binom (n k : ℕ) : ℕ := n.choose k

theorem binom_product :
  binom 10 3 * binom 8 3 = 6720 := by
  sorry

end binom_product_l2168_216834


namespace domain_of_sqrt_log_l2168_216878

noncomputable def domain_of_function : Set ℝ := 
  {x : ℝ | (-Real.sqrt 2) ≤ x ∧ x < -1 ∨ 1 < x ∧ x ≤ Real.sqrt 2}

theorem domain_of_sqrt_log : ∀ x : ℝ, 
  (∃ y : ℝ, y = Real.sqrt (Real.log (x^2 - 1) / Real.log (1/2)) ∧ 
  y ≥ 0) ↔ x ∈ domain_of_function := 
by
  sorry

end domain_of_sqrt_log_l2168_216878


namespace arrangement_with_A_in_middle_arrangement_with_A_at_end_B_not_at_end_arrangement_with_A_B_adjacent_not_adjacent_to_C_l2168_216844

-- Proof Problem 1
theorem arrangement_with_A_in_middle (products : Finset ℕ) (A : ℕ) (hA : A ∈ products) (arrangements : Finset (Fin 5 → ℕ)) :
  5 ∈ products ∧ (∀ a ∈ arrangements, a (Fin.mk 2 sorry) = A) →
  arrangements.card = 24 :=
by sorry

-- Proof Problem 2
theorem arrangement_with_A_at_end_B_not_at_end (products : Finset ℕ) (A B : ℕ) (hA : A ∈ products) (hB : B ∈ products) (arrangements : Finset (Fin 5 → ℕ)) :
  (5 ∈ products ∧ (∀ a ∈ arrangements, (a 0 = A ∨ a 4 = A) ∧ (a 1 ≠ B ∧ a 2 ≠ B ∧ a 3 ≠ B))) →
  arrangements.card = 36 :=
by sorry

-- Proof Problem 3
theorem arrangement_with_A_B_adjacent_not_adjacent_to_C (products : Finset ℕ) (A B C : ℕ) (hA : A ∈ products) (hB : B ∈ products) (hC : C ∈ products) (arrangements : Finset (Fin 5 → ℕ)) :
  (5 ∈ products ∧ (∀ a ∈ arrangements, ((a 0 = A ∧ a 1 = B) ∨ (a 1 = A ∧ a 2 = B) ∨ (a 2 = A ∧ a 3 = B) ∨ (a 3 = A ∧ a 4 = B)) ∧
   (a 0 ≠ A ∧ a 1 ≠ B ∧ a 2 ≠ C))) →
  arrangements.card = 36 :=
by sorry

end arrangement_with_A_in_middle_arrangement_with_A_at_end_B_not_at_end_arrangement_with_A_B_adjacent_not_adjacent_to_C_l2168_216844


namespace treadmill_time_saved_l2168_216881

theorem treadmill_time_saved:
  let monday_speed := 6
  let tuesday_speed := 4
  let wednesday_speed := 5
  let thursday_speed := 6
  let friday_speed := 3
  let distance := 3 
  let daily_times : List ℚ := 
    [distance/monday_speed, distance/tuesday_speed, distance/wednesday_speed, distance/thursday_speed, distance/friday_speed]
  let total_time := (daily_times.map (λ t => t)).sum
  let total_distance := 5 * distance 
  let uniform_speed := 5 
  let uniform_time := total_distance / uniform_speed 
  let time_difference := total_time - uniform_time 
  let time_in_minutes := time_difference * 60 
  time_in_minutes = 21 := 
by 
  sorry

end treadmill_time_saved_l2168_216881


namespace no_pos_int_lt_2000_7_times_digits_sum_l2168_216874

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem no_pos_int_lt_2000_7_times_digits_sum :
  ∀ n : ℕ, n < 2000 → n = 7 * sum_of_digits n → False :=
by
  intros n h1 h2
  sorry

end no_pos_int_lt_2000_7_times_digits_sum_l2168_216874


namespace pipeA_fill_time_l2168_216847

variable (t : ℕ) -- t is the time in minutes for Pipe A to fill the tank

-- Conditions
def pipeA_duration (t : ℕ) : Prop :=
  t > 0

def pipeB_duration (t : ℕ) : Prop :=
  t / 3 > 0

def combined_rate (t : ℕ) : Prop :=
  3 * (1 / (4 / t)) = t

-- Problem
theorem pipeA_fill_time (h1 : pipeA_duration t) (h2 : pipeB_duration t) (h3 : combined_rate t) : t = 12 :=
sorry

end pipeA_fill_time_l2168_216847


namespace odd_function_b_value_f_monotonically_increasing_l2168_216815

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := Real.log (Real.sqrt (4 * x^2 + b) + 2 * x)

-- part (1): Prove that if y = f(x) is an odd function, then b = 1
theorem odd_function_b_value :
  (∀ x : ℝ, f x b + f (-x) b = 0) → b = 1 := sorry

-- part (2): Prove that y = f(x) is monotonically increasing for all x in ℝ given b = 1
theorem f_monotonically_increasing (b : ℝ) :
  b = 1 → ∀ x1 x2 : ℝ, x1 < x2 → f x1 b < f x2 b := sorry

end odd_function_b_value_f_monotonically_increasing_l2168_216815


namespace M_lt_N_l2168_216845

/-- M is the coefficient of x^4 y^2 in the expansion of (x^2 + x + 2y)^5 -/
def M : ℕ := 120

/-- N is the sum of the coefficients in the expansion of (3/x - x)^7 -/
def N : ℕ := 128

/-- The relationship between M and N -/
theorem M_lt_N : M < N := by 
  dsimp [M, N]
  sorry

end M_lt_N_l2168_216845


namespace smallest_positive_angle_same_terminal_side_l2168_216879

theorem smallest_positive_angle_same_terminal_side 
  (k : ℤ) : ∃ α : ℝ, 0 < α ∧ α < 360 ∧ -2002 = α + k * 360 ∧ α = 158 :=
by
  sorry

end smallest_positive_angle_same_terminal_side_l2168_216879


namespace actual_distance_between_cities_l2168_216811

-- Define the scale and distance on the map as constants
def distance_on_map : ℝ := 20
def scale_inch_miles : ℝ := 12  -- Because 1 inch = 12 miles derived from the scale 0.5 inches = 6 miles

-- Define the actual distance calculation
def actual_distance (distance_inch : ℝ) (scale : ℝ) : ℝ :=
  distance_inch * scale

-- Example theorem to prove the actual distance between the cities
theorem actual_distance_between_cities :
  actual_distance distance_on_map scale_inch_miles = 240 := by
  sorry

end actual_distance_between_cities_l2168_216811


namespace smallest_four_digit_multiple_of_18_l2168_216838

theorem smallest_four_digit_multiple_of_18 : 
  ∃ (n : ℕ), 999 < n ∧ n < 10000 ∧ 18 ∣ n ∧ ∀ m : ℕ, 999 < m ∧ m < 10000 ∧ 18 ∣ m → n ≤ m ∧ n = 1008 := 
sorry

end smallest_four_digit_multiple_of_18_l2168_216838


namespace polynomial_rewrite_l2168_216853

theorem polynomial_rewrite (d : ℤ) (h : d ≠ 0) :
  let a := 20
  let b := 18
  let c := 18
  let e := 8
  (16 * d + 15 + 18 * d^2 + 5 * d^3) + (4 * d + 3 + 3 * d^3) = a * d + b + c * d^2 + e * d^3 ∧ a + b + c + e = 64 := 
by
  sorry

end polynomial_rewrite_l2168_216853


namespace perimeter_of_figure_l2168_216891

-- Given conditions
def side_length : Nat := 2
def num_horizontal_segments : Nat := 16
def num_vertical_segments : Nat := 10

-- Define a function to calculate the perimeter based on the given conditions
def calculate_perimeter (side_length : Nat) (num_horizontal_segments : Nat) (num_vertical_segments : Nat) : Nat :=
  (num_horizontal_segments * side_length) + (num_vertical_segments * side_length)

-- Statement to be proved
theorem perimeter_of_figure : calculate_perimeter side_length num_horizontal_segments num_vertical_segments = 52 :=
by
  -- The proof would go here
  sorry

end perimeter_of_figure_l2168_216891


namespace cost_of_schools_renovation_plans_and_min_funding_l2168_216852

-- Define costs of Type A and Type B schools
def cost_A : ℝ := 60
def cost_B : ℝ := 85

-- Initial conditions given in the problem
axiom initial_condition_1 : cost_A + 2 * cost_B = 230
axiom initial_condition_2 : 2 * cost_A + cost_B = 205

-- Variables for number of Type A and Type B schools to renovate
variables (x : ℕ) (y : ℕ)
-- Total schools to renovate
axiom total_schools : x + y = 6

-- National and local finance constraints
axiom national_finance_max : 60 * x + 85 * y ≤ 380
axiom local_finance_min : 10 * x + 15 * y ≥ 70

-- Proving the cost of one Type A and one Type B school
theorem cost_of_schools : cost_A = 60 ∧ cost_B = 85 := 
by {
  sorry
}

-- Proving the number of renovation plans and the least funding plan
theorem renovation_plans_and_min_funding :
  ∃ x y, (x + y = 6) ∧ 
         (10 * x + 15 * y ≥ 70) ∧ 
         (60 * x + 85 * y ≤ 380) ∧ 
         (x = 2 ∧ y = 4 ∨ x = 3 ∧ y = 3 ∨ x = 4 ∧ y = 2) ∧ 
         (∀ (a b : ℕ), (a + b = 6) ∧ 
                       (10 * a + 15 * b ≥ 70) ∧ 
                       (60 * a + 85 * b ≤ 380) → 
                       60 * a + 85 * b ≥ 410) :=
by {
  sorry
}

end cost_of_schools_renovation_plans_and_min_funding_l2168_216852


namespace initial_puppies_count_l2168_216888

theorem initial_puppies_count (P : ℕ) (h1 : P - 2 + 3 = 8) : P = 7 :=
sorry

end initial_puppies_count_l2168_216888


namespace fraction_of_total_amount_l2168_216813

theorem fraction_of_total_amount (p q r : ℕ) (h1 : p + q + r = 4000) (h2 : r = 1600) :
  r / (p + q + r) = 2 / 5 :=
by
  sorry

end fraction_of_total_amount_l2168_216813


namespace solids_with_triangular_front_view_l2168_216893

-- Definitions based on given conditions
inductive Solid
  | TriangularPyramid
  | SquarePyramid
  | TriangularPrism
  | SquarePrism
  | Cone
  | Cylinder

def can_have_triangular_front_view : Solid → Prop
  | Solid.TriangularPyramid => true
  | Solid.SquarePyramid => true
  | Solid.TriangularPrism => true
  | Solid.SquarePrism => false
  | Solid.Cone => true
  | Solid.Cylinder => false

-- Theorem statement
theorem solids_with_triangular_front_view :
  {s : Solid | can_have_triangular_front_view s} = 
  {Solid.TriangularPyramid, Solid.SquarePyramid, Solid.TriangularPrism, Solid.Cone} :=
by
  sorry

end solids_with_triangular_front_view_l2168_216893


namespace round_trip_time_l2168_216832

def boat_speed_still_water : ℝ := 16
def stream_speed : ℝ := 2
def distance_to_place : ℝ := 7560

theorem round_trip_time : (distance_to_place / (boat_speed_still_water + stream_speed) + distance_to_place / (boat_speed_still_water - stream_speed)) = 960 := by
  sorry

end round_trip_time_l2168_216832


namespace ratio_of_term_to_difference_l2168_216868

def arithmetic_progression_sum (n a d : ℕ) : ℕ := (n * (2 * a + (n - 1) * d)) / 2

theorem ratio_of_term_to_difference (a d : ℕ) 
  (h1: arithmetic_progression_sum 7 a d = arithmetic_progression_sum 3 a d + 20)
  (h2 : d ≠ 0) : a / d = 1 / 2 := 
by 
  sorry

end ratio_of_term_to_difference_l2168_216868


namespace problem_1_solution_set_problem_2_range_of_a_l2168_216892

-- Define the function f(x)
def f (x a : ℝ) := |2 * x - a| + |x - 1|

-- Problem 1: Solution set of the inequality f(x) ≥ 2 when a = 3
theorem problem_1_solution_set :
  { x : ℝ | f x 3 ≥ 2 } = { x : ℝ | x ≤ 2/3 ∨ x ≥ 2 } :=
sorry

-- Problem 2: Range of a such that f(x) ≥ 5 - x for all x ∈ ℝ
theorem problem_2_range_of_a (a : ℝ) :
  (∀ x : ℝ, f x a ≥ 5 - x) ↔ a ≥ 6 :=
sorry

end problem_1_solution_set_problem_2_range_of_a_l2168_216892


namespace greatest_remainder_le_11_l2168_216875

noncomputable def greatest_remainder (x : ℕ) : ℕ := x % 12

theorem greatest_remainder_le_11 (x : ℕ) (h : x % 12 ≠ 0) : greatest_remainder x = 11 :=
by
  sorry

end greatest_remainder_le_11_l2168_216875


namespace number_of_hens_l2168_216842

theorem number_of_hens (H C : ℕ) (h1 : H + C = 46) (h2 : 2 * H + 4 * C = 136) : H = 24 :=
by
  sorry

end number_of_hens_l2168_216842


namespace vector_dot_product_l2168_216829

def dot_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2

noncomputable def sin_deg (x : ℝ) : ℝ := Real.sin (x * Real.pi / 180)
noncomputable def cos_deg (x : ℝ) : ℝ := Real.cos (x * Real.pi / 180)

theorem vector_dot_product :
  let a := (sin_deg 55, sin_deg 35)
  let b := (sin_deg 25, sin_deg 65)
  dot_product a b = (Real.sqrt 3) / 2 :=
by
  sorry

end vector_dot_product_l2168_216829


namespace eleven_not_sum_of_two_primes_l2168_216858

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem eleven_not_sum_of_two_primes :
  ¬ ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p + q = 11 :=
by sorry

end eleven_not_sum_of_two_primes_l2168_216858


namespace machine_value_depletion_rate_l2168_216825

theorem machine_value_depletion_rate :
  ∃ r : ℝ, 700 * (1 - r)^2 = 567 ∧ r = 0.1 := 
by
  sorry

end machine_value_depletion_rate_l2168_216825


namespace prob_A_wins_correct_l2168_216840

noncomputable def prob_A_wins : ℚ :=
  let outcomes : ℕ := 3^3
  let win_one_draw_two : ℕ := 3
  let win_two_other : ℕ := 6
  let win_all : ℕ := 1
  let total_wins : ℕ := win_one_draw_two + win_two_other + win_all
  total_wins / outcomes

theorem prob_A_wins_correct :
  prob_A_wins = 10/27 :=
by
  sorry

end prob_A_wins_correct_l2168_216840


namespace gcd_lcm_sum_l2168_216836

-- Define the necessary components: \( A \) as the greatest common factor and \( B \) as the least common multiple of 16, 32, and 48
def A := Int.gcd (Int.gcd 16 32) 48
def B := Int.lcm (Int.lcm 16 32) 48

-- Statement that needs to be proved
theorem gcd_lcm_sum : A + B = 112 := by
  sorry

end gcd_lcm_sum_l2168_216836


namespace large_box_times_smaller_box_l2168_216872

noncomputable def large_box_volume (width length height : ℕ) : ℕ := width * length * height

noncomputable def small_box_volume (width length height : ℕ) : ℕ := width * length * height

theorem large_box_times_smaller_box :
  let large_volume := large_box_volume 30 20 5
  let small_volume := small_box_volume 6 4 1
  large_volume / small_volume = 125 :=
by
  let large_volume := large_box_volume 30 20 5
  let small_volume := small_box_volume 6 4 1
  show large_volume / small_volume = 125
  sorry

end large_box_times_smaller_box_l2168_216872


namespace distance_at_40_kmph_l2168_216828

theorem distance_at_40_kmph (x : ℝ) (h1 : x / 40 + (250 - x) / 60 = 5) : x = 100 := 
by
  sorry

end distance_at_40_kmph_l2168_216828


namespace arc_length_problem_l2168_216866

noncomputable def arc_length (r : ℝ) (theta : ℝ) : ℝ :=
  r * theta

theorem arc_length_problem :
  ∀ (r : ℝ) (theta_deg : ℝ), r = 1 ∧ theta_deg = 150 → 
  arc_length r (theta_deg * (Real.pi / 180)) = (5 * Real.pi / 6) :=
by
  intro r theta_deg h
  sorry

end arc_length_problem_l2168_216866


namespace bc_over_ad_l2168_216818

-- Define the rectangular prism
structure RectangularPrism :=
(length width height : ℝ)

-- Define the problem parameters
def B : RectangularPrism := ⟨2, 4, 5⟩

-- Define the volume form of S(r)
def volume (a b c d : ℝ) (r : ℝ) : ℝ := a * r^3 + b * r^2 + c * r + d

-- Prove that the relationship holds
theorem bc_over_ad (a b c d : ℝ) (r : ℝ) (h_a : a = (4 * π) / 3) (h_b : b = 11 * π) (h_c : c = 76) (h_d : d = 40) :
  (b * c) / (a * d) = 15.67 := by
  sorry

end bc_over_ad_l2168_216818


namespace length_of_football_field_l2168_216880

theorem length_of_football_field :
  ∃ x : ℝ, (4 * x + 500 = 1172) ∧ x = 168 :=
by
  use 168
  simp
  sorry

end length_of_football_field_l2168_216880


namespace rectangle_perimeter_l2168_216809

-- Definitions of the conditions
def side_length_square : ℕ := 75  -- side length of the square in mm
def height_sum (x y z : ℕ) : Prop := x + y + z = side_length_square  -- sum of heights of the rectangles

-- Perimeter definition
def perimeter (h : ℕ) (w : ℕ) : ℕ := 2 * (h + w)

-- Statement of the problem
theorem rectangle_perimeter (x y z : ℕ) (h_sum : height_sum x y z)
  (h1 : perimeter x side_length_square = (perimeter y side_length_square + perimeter z side_length_square) / 2)
  : perimeter x side_length_square = 200 := by
  sorry

end rectangle_perimeter_l2168_216809


namespace basketball_shots_l2168_216808

theorem basketball_shots (total_points total_3pt_shots: ℕ) 
  (h1: total_points = 26) 
  (h2: total_3pt_shots = 4) 
  (h3: ∀ points_from_3pt_shots, points_from_3pt_shots = 3 * total_3pt_shots) :
  let points_from_3pt_shots := 3 * total_3pt_shots
  let points_from_2pt_shots := total_points - points_from_3pt_shots
  let total_2pt_shots := points_from_2pt_shots / 2
  total_2pt_shots + total_3pt_shots = 11 :=
by
  sorry

end basketball_shots_l2168_216808


namespace male_students_count_l2168_216830

theorem male_students_count :
  ∃ (N M : ℕ), 
  (N % 4 = 2) ∧ 
  (N % 5 = 1) ∧ 
  (N = M + 15) ∧ 
  (15 > M) ∧ 
  (M = 11) :=
sorry

end male_students_count_l2168_216830


namespace smallest_positive_integer_remainder_l2168_216814

theorem smallest_positive_integer_remainder
  (b : ℕ) (h1 : b % 4 = 3) (h2 : b % 6 = 5) :
  b = 11 := by
  sorry

end smallest_positive_integer_remainder_l2168_216814


namespace net_change_of_Toronto_Stock_Exchange_l2168_216843

theorem net_change_of_Toronto_Stock_Exchange :
  let monday := -150
  let tuesday := 106
  let wednesday := -47
  let thursday := 182
  let friday := -210
  (monday + tuesday + wednesday + thursday + friday) = -119 :=
by
  let monday := -150
  let tuesday := 106
  let wednesday := -47
  let thursday := 182
  let friday := -210
  have h : (monday + tuesday + wednesday + thursday + friday) = -119 := sorry
  exact h

end net_change_of_Toronto_Stock_Exchange_l2168_216843


namespace simplify_expression_l2168_216855

theorem simplify_expression (a b : ℚ) (ha : a = -1) (hb : b = 1/4) : 
  (a + 2 * b) ^ 2 + (a + 2 * b) * (a - 2 * b) = 1 := 
by 
  sorry

end simplify_expression_l2168_216855


namespace arithmetic_sequence_ratio_l2168_216860

-- Define conditions
def sum_ratios (A_n B_n : ℕ → ℚ) (n : ℕ) : Prop := (A_n n) / (B_n n) = (4 * n + 2) / (5 * n - 5)
def arithmetic_sequences (a_n b_n : ℕ → ℚ) : Prop :=
  ∃ A_n B_n : ℕ → ℚ,
    (∀ n, A_n n = n * (a_n 1) + (n * (n - 1) / 2) * (a_n 2 - a_n 1)) ∧
    (∀ n, B_n n = n * (b_n 1) + (n * (n - 1) / 2) * (b_n 2 - b_n 1)) ∧
    ∀ n, sum_ratios A_n B_n n

-- Theorem to be proven
theorem arithmetic_sequence_ratio
  (a_n b_n : ℕ → ℚ)
  (h : arithmetic_sequences a_n b_n) :
  (a_n 5 + a_n 13) / (b_n 5 + b_n 13) = 7 / 8 :=
sorry

end arithmetic_sequence_ratio_l2168_216860


namespace find_a3_l2168_216810

-- Definitions from conditions
def arithmetic_sum (a1 a3 : ℕ) := (3 / 2) * (a1 + a3)
def common_difference := 2
def S3 := 12

-- Theorem to prove that a3 = 6
theorem find_a3 (a1 a3 : ℕ) (h₁ : arithmetic_sum a1 a3 = S3) (h₂ : a3 = a1 + common_difference * 2) : a3 = 6 :=
by
  sorry

end find_a3_l2168_216810


namespace tan_alpha_value_tan_beta_value_sum_angles_l2168_216863

open Real

noncomputable def tan_alpha (α : ℝ) : ℝ := sin α / cos α
noncomputable def tan_beta (β : ℝ) : ℝ := sin β / cos β

def conditions (α β : ℝ) :=
  α > 0 ∧ α < π / 2 ∧ β > 0 ∧ β < π / 2 ∧ 
  sin α = 1 / sqrt 10 ∧ tan β = 1 / 7

theorem tan_alpha_value (α β : ℝ) (h : conditions α β) : tan_alpha α = 1 / 3 := sorry

theorem tan_beta_value (α β : ℝ) (h : conditions α β) : tan_beta β = 1 / 7 := sorry

theorem sum_angles (α β : ℝ) (h : conditions α β) : 2 * α + β = π / 4 := sorry

end tan_alpha_value_tan_beta_value_sum_angles_l2168_216863


namespace ratio_of_selling_prices_l2168_216870

variable (CP : ℝ)
def SP1 : ℝ := CP * 1.6
def SP2 : ℝ := CP * 0.8

theorem ratio_of_selling_prices : SP2 / SP1 = 1 / 2 := 
by sorry

end ratio_of_selling_prices_l2168_216870


namespace parabola_focus_l2168_216827

theorem parabola_focus (a : ℝ) : (∀ x : ℝ, y = a * x^2) ∧ ∃ f : ℝ × ℝ, f = (0, 1) → a = (1/4) := 
sorry

end parabola_focus_l2168_216827


namespace area_of_quadrilateral_EFGH_l2168_216835

-- Define the properties of rectangle ABCD and the areas
def rectangle (A B C D : Type) := 
  ∃ (area : ℝ), area = 48

-- Define the positions of the points E, G, F, H
def points_positions (A D C B E G F H : Type) :=
  ∃ (one_third : ℝ) (two_thirds : ℝ), one_third = 1/3 ∧ two_thirds = 2/3

-- Define the area calculation for quadrilateral EFGH
def area_EFGH (area_ABCD : ℝ) (one_third : ℝ) : ℝ :=
  (one_third * one_third) * area_ABCD

-- The proof statement that area of EFGH is 5 1/3 square meters
theorem area_of_quadrilateral_EFGH 
  (A B C D E F G H : Type)
  (area_ABCD : ℝ)
  (one_third : ℝ) :
  rectangle A B C D →
  points_positions A D C B E G F H →
  area_ABCD = 48 →
  one_third = 1/3 →
  area_EFGH area_ABCD one_third = 16/3 :=
by
  intros h1 h2 h3 h4
  have h5 : area_EFGH area_ABCD one_third = 16/3 :=
  sorry
  exact h5

end area_of_quadrilateral_EFGH_l2168_216835


namespace jerry_wants_to_raise_average_l2168_216885

theorem jerry_wants_to_raise_average :
  let average_first_3_tests := 85
  let total_first_3_tests := 3 * average_first_3_tests
  let score_fourth_test := 93
  let total_after_fourth_test := total_first_3_tests + score_fourth_test
  let new_average := (total_after_fourth_test : ℤ) / 4
  new_average - average_first_3_tests = 2 :=
by
  let average_first_3_tests := 85
  let total_first_3_tests := 3 * average_first_3_tests
  let score_fourth_test := 93
  let total_after_fourth_test := total_first_3_tests + score_fourth_test
  let new_average := (total_after_fourth_test : ℤ) / 4
  have h : new_average - average_first_3_tests = 2 := by
    sorry
  exact h

end jerry_wants_to_raise_average_l2168_216885


namespace f_decreasing_interval_triangle_abc_l2168_216819

noncomputable def f (x : Real) : Real := 2 * (Real.sin x)^2 + Real.cos ((Real.pi) / 3 - 2 * x)

theorem f_decreasing_interval :
  ∃ (a b : Real), a = Real.pi / 3 ∧ b = 5 * Real.pi / 6 ∧ 
  ∀ x y, (a ≤ x ∧ x < y ∧ y ≤ b) → f y ≤ f x := 
sorry

variables {a b c : Real} (A B C : Real) 

theorem triangle_abc (h1 : A = Real.pi / 3) 
    (h2 : f A = 2)
    (h3 : a = 2 * b)
    (h4 : Real.sin C = 2 * Real.sin B):
  a / b = Real.sqrt 3 := 
sorry

end f_decreasing_interval_triangle_abc_l2168_216819


namespace possible_values_l2168_216812

def seq_condition (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, a n + a (n + 1) = 2 * a (n + 2) * a (n + 3) + 2016

theorem possible_values (a : ℕ → ℤ) (h : seq_condition a) :
  (a 1, a 2) = (0, 2016) ∨
  (a 1, a 2) = (-14, 70) ∨
  (a 1, a 2) = (-69, 15) ∨
  (a 1, a 2) = (-2015, 1) ∨
  (a 1, a 2) = (2016, 0) ∨
  (a 1, a 2) = (70, -14) ∨
  (a 1, a 2) = (15, -69) ∨
  (a 1, a 2) = (1, -2015) :=
sorry

end possible_values_l2168_216812


namespace C_is_a_liar_l2168_216876

def is_knight_or_liar (P : Prop) : Prop :=
P = true ∨ P = false

variable (A B C : Prop)

-- A, B and C can only be true (knight) or false (liar)
axiom a1 : is_knight_or_liar A
axiom a2 : is_knight_or_liar B
axiom a3 : is_knight_or_liar C

-- A says "B is a liar", meaning if A is a knight, B is a liar, and if A is a liar, B is a knight
axiom a4 : A = true → B = false
axiom a5 : A = false → B = true

-- B says "A and C are of the same type", meaning if B is a knight, A and C are of the same type, otherwise they are not
axiom a6 : B = true → (A = C)
axiom a7 : B = false → (A ≠ C)

-- Prove that C is a liar
theorem C_is_a_liar : C = false :=
by
  sorry

end C_is_a_liar_l2168_216876


namespace range_neg_square_l2168_216817

theorem range_neg_square (x : ℝ) (hx : -3 ≤ x ∧ x ≤ 1) : 
  -9 ≤ -x^2 ∧ -x^2 ≤ 0 :=
sorry

end range_neg_square_l2168_216817


namespace animals_remaining_correct_l2168_216800

-- Definitions from the conditions
def initial_cows : ℕ := 184
def initial_dogs : ℕ := initial_cows / 2

def cows_sold : ℕ := initial_cows / 4
def remaining_cows : ℕ := initial_cows - cows_sold

def dogs_sold : ℕ := (3 * initial_dogs) / 4
def remaining_dogs : ℕ := initial_dogs - dogs_sold

def total_remaining_animals : ℕ := remaining_cows + remaining_dogs

-- Theorem to be proved
theorem animals_remaining_correct : total_remaining_animals = 161 := 
by
  sorry

end animals_remaining_correct_l2168_216800


namespace xiaofang_final_score_l2168_216867

def removeHighestLowestScores (scores : List ℕ) : List ℕ :=
  let max_score := scores.maximum.getD 0
  let min_score := scores.minimum.getD 0
  scores.erase max_score |>.erase min_score

def average (scores : List ℕ) : ℚ :=
  (scores.sum : ℚ) / scores.length

theorem xiaofang_final_score :
  let scores := [95, 94, 91, 88, 91, 90, 94, 93, 91, 92]
  average (removeHighestLowestScores scores) = 92 := by
  sorry

end xiaofang_final_score_l2168_216867


namespace first_discount_percentage_l2168_216816

theorem first_discount_percentage (normal_price sale_price : ℝ) (second_discount : ℝ) (first_discount : ℝ) :
  normal_price = 149.99999999999997 →
  sale_price = 108 →
  second_discount = 0.20 →
  (1 - second_discount) * (1 - first_discount) * normal_price = sale_price →
  first_discount = 0.10 :=
by
  intros
  sorry

end first_discount_percentage_l2168_216816


namespace equidistant_point_quadrants_l2168_216821

theorem equidistant_point_quadrants (x y : ℝ) (h : 4 * x + 3 * y = 12) :
  (x > 0 ∧ y = 0) ∨ (x < 0 ∧ y > 0) :=
sorry

end equidistant_point_quadrants_l2168_216821


namespace Zach_scored_more_l2168_216859

theorem Zach_scored_more :
  let Zach := 42
  let Ben := 21
  Zach - Ben = 21 :=
by
  let Zach := 42
  let Ben := 21
  exact rfl

end Zach_scored_more_l2168_216859


namespace shaded_cells_product_l2168_216856

def product_eq (a b c : ℕ) (p : ℕ) : Prop := a * b * c = p

theorem shaded_cells_product :
  ∃ (a₁₁ a₁₂ a₁₃ a₂₁ a₂₂ a₂₃ a₃₁ a₃₂ a₃₃ : ℕ),
    product_eq a₁₁ a₁₂ a₁₃ 12 ∧
    product_eq a₂₁ a₂₂ a₂₃ 112 ∧
    product_eq a₃₁ a₃₂ a₃₃ 216 ∧
    product_eq a₁₁ a₂₁ a₃₁ 12 ∧
    product_eq a₁₂ a₂₂ a₃₂ 12 ∧
    (a₁₁ * a₂₂ * a₃₃ = 3 * 2 * 5) :=
sorry

end shaded_cells_product_l2168_216856


namespace Ksyusha_time_to_school_l2168_216823

/-- Ksyusha runs twice as fast as she walks (both speeds are constant).
On Tuesday, Ksyusha walked a distance twice the distance she ran, and she reached school in 30 minutes.
On Wednesday, Ksyusha ran twice the distance she walked.
Prove that it took her 24 minutes to get from home to school on Wednesday. -/
theorem Ksyusha_time_to_school
  (S v : ℝ)                  -- S for distance unit, v for walking speed
  (htuesday : 2 * (2 * S / v) + S / (2 * v) = 30) -- Condition on Tuesday's travel
  :
  (S / v + 2 * S / (2 * v) = 24)                 -- Claim about Wednesday's travel time
:=
  sorry

end Ksyusha_time_to_school_l2168_216823


namespace sum_of_a_and_b_l2168_216803

def otimes (x y : ℝ) : ℝ := x * (1 - y)

variable (a b : ℝ)

theorem sum_of_a_and_b :
  ({ x : ℝ | (x - a) * (1 - (x - b)) > 0 } = { x : ℝ | 2 < x ∧ x < 3 }) →
  a + b = 4 :=
by
  intro h
  have h_eq : ∀ x, (x - a) * ((1 : ℝ) - (x - b)) = (x - a) * (x - (b + 1)) := sorry
  have h_ineq : ∀ x, (x - a) * (x - (b + 1)) > 0 ↔ 2 < x ∧ x < 3 := sorry
  have h_set_eq : { x | (x - a) * ((1 : ℝ) - (x - b)) > 0 } = { x | 2 < x ∧ x < 3 } := sorry
  have h_roots_2_3 : (2 - a) * (2 - (b + 1)) = 0 ∧ (3 - a) * (3 - (b + 1)) = 0 := sorry
  have h_2_eq : 2 - a = 0 ∨ 2 - (b + 1) = 0 := sorry
  have h_3_eq : 3 - a = 0 ∨ 3 - (b + 1) = 0 := sorry
  have h_a_2 : a = 2 ∨ b + 1 = 2 := sorry
  have h_b_2 : b = 2 - 1 := sorry
  have h_a_3 : a = 3 ∨ b + 1 = 3 := sorry
  have h_b_3 : b = 3 - 1 := sorry
  sorry

end sum_of_a_and_b_l2168_216803


namespace people_born_in_country_l2168_216884

-- Define the conditions
def people_immigrated : ℕ := 16320
def new_people_total : ℕ := 106491

-- Define the statement to be proven
theorem people_born_in_country (people_born : ℕ) (h : people_born = new_people_total - people_immigrated) : 
    people_born = 90171 :=
  by
    -- This is where we would provide the proof, but we use sorry to skip the proof.
    sorry

end people_born_in_country_l2168_216884


namespace min_weight_of_automobile_l2168_216824

theorem min_weight_of_automobile (ferry_weight_tons: ℝ) (auto_max_weight: ℝ) 
  (max_autos: ℝ) (ferry_weight_pounds: ℝ) (min_auto_weight: ℝ) : 
  ferry_weight_tons = 50 → 
  auto_max_weight = 3200 → 
  max_autos = 62.5 → 
  ferry_weight_pounds = ferry_weight_tons * 2000 → 
  min_auto_weight = ferry_weight_pounds / max_autos → 
  min_auto_weight = 1600 :=
by
  intros
  sorry

end min_weight_of_automobile_l2168_216824


namespace best_solved_completing_square_l2168_216854

theorem best_solved_completing_square :
  ∀ (x : ℝ), x^2 - 2*x - 3 = 0 → (x - 1)^2 - 4 = 0 :=
sorry

end best_solved_completing_square_l2168_216854


namespace age_difference_l2168_216899

variable (A B C : ℕ)

theorem age_difference (h : A + B = B + C + 12) : A - C = 12 := 
sorry

end age_difference_l2168_216899


namespace complement_union_A_B_l2168_216886

def U : Set ℕ := {x | x < 6}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {2, 4}

theorem complement_union_A_B :
  (U \ (A ∪ B)) = {0, 5} := by
  sorry

end complement_union_A_B_l2168_216886


namespace number_of_pines_l2168_216848

theorem number_of_pines (total_trees : ℕ) (T B S : ℕ) (h_total : total_trees = 101)
  (h_poplar_spacing : ∀ T1 T2, T1 ≠ T2 → (T2 - T1) > 1)
  (h_birch_spacing : ∀ B1 B2, B1 ≠ B2 → (B2 - B1) > 2)
  (h_pine_spacing : ∀ S1 S2, S1 ≠ S2 → (S2 - S1) > 3) :
  S = 25 ∨ S = 26 :=
sorry

end number_of_pines_l2168_216848


namespace sum_digits_l2168_216887

def repeat_pattern (d: ℕ) (n: ℕ) : ℕ :=
  let pattern := if d = 404 then 404 else if d = 707 then 707 else 0
  pattern * 10^(n / 3)

def N1 := repeat_pattern 404 101
def N2 := repeat_pattern 707 101
def P := N1 * N2

def thousands_digit (n: ℕ) : ℕ :=
  (n / 1000) % 10

def units_digit (n: ℕ) : ℕ :=
  n % 10

theorem sum_digits : thousands_digit P + units_digit P = 10 := by
  sorry

end sum_digits_l2168_216887


namespace negation_proposition_l2168_216871

theorem negation_proposition :
  (¬ (∀ x : ℝ, x ≥ 0)) ↔ (∃ x : ℝ, x < 0) :=
by
  sorry

end negation_proposition_l2168_216871


namespace fraction_product_equals_l2168_216894

def frac1 := 7 / 4
def frac2 := 8 / 14
def frac3 := 9 / 6
def frac4 := 10 / 25
def frac5 := 28 / 21
def frac6 := 15 / 45
def frac7 := 32 / 16
def frac8 := 50 / 100

theorem fraction_product_equals : 
  (frac1 * frac2 * frac3 * frac4 * frac5 * frac6 * frac7 * frac8) = (4 / 5) := 
by
  sorry

end fraction_product_equals_l2168_216894


namespace linear_condition_l2168_216807

theorem linear_condition (a : ℝ) : a ≠ 0 ↔ ∃ (x y : ℝ), ax + y = -1 :=
by
  sorry

end linear_condition_l2168_216807


namespace abs_sum_lt_ineq_l2168_216857

theorem abs_sum_lt_ineq (x : ℝ) (a : ℝ) (h₀ : 0 < a) : 
  (∃ x : ℝ, |x - 4| + |x - 3| < a) ↔ (1 < a) :=
by
  sorry

end abs_sum_lt_ineq_l2168_216857


namespace solution_A_l2168_216804

def P : Set ℕ := {1, 2, 3, 4}

theorem solution_A (A : Set ℕ) (h1 : A ⊆ P) 
  (h2 : ∀ x ∈ A, 2 * x ∉ A) 
  (h3 : ∀ x ∈ (P \ A), 2 * x ∉ (P \ A)): 
    A = {2} ∨ A = {1, 4} ∨ A = {2, 3} ∨ A = {1, 3, 4} :=
sorry

end solution_A_l2168_216804


namespace exact_time_between_9_10_l2168_216850

theorem exact_time_between_9_10
  (t : ℝ)
  (h1 : 0 ≤ t ∧ t < 60)
  (h2 : |6 * (t + 5) - (270 + 0.5 * (t - 2))| = 180) :
  t = 10 + 3 / 4 :=
sorry

end exact_time_between_9_10_l2168_216850


namespace field_trip_seniors_l2168_216882

theorem field_trip_seniors (n : ℕ) 
  (h1 : n < 300) 
  (h2 : n % 17 = 15) 
  (h3 : n % 19 = 12) : 
  n = 202 :=
  sorry

end field_trip_seniors_l2168_216882


namespace bus_capacity_l2168_216831

def left_side_seats : ℕ := 15
def seats_difference : ℕ := 3
def people_per_seat : ℕ := 3
def back_seat_capacity : ℕ := 7

theorem bus_capacity : left_side_seats + (left_side_seats - seats_difference) * people_per_seat + back_seat_capacity = 88 := 
by
  sorry

end bus_capacity_l2168_216831


namespace chandler_tickets_total_cost_l2168_216802

theorem chandler_tickets_total_cost :
  let movie_ticket_cost := 30
  let num_movie_tickets := 8
  let num_football_tickets := 5
  let num_concert_tickets := 3
  let num_theater_tickets := 4
  let theater_ticket_cost := 40
  let discount := 0.10
  let total_movie_cost := num_movie_tickets * movie_ticket_cost
  let football_ticket_cost := total_movie_cost / 2
  let total_football_cost := num_football_tickets * football_ticket_cost
  let concert_ticket_cost := football_ticket_cost - 10
  let total_concert_cost := num_concert_tickets * concert_ticket_cost
  let discounted_theater_ticket_cost := theater_ticket_cost * (1 - discount)
  let total_theater_cost := num_theater_tickets * discounted_theater_ticket_cost
  let total_cost := total_movie_cost + total_football_cost + total_concert_cost + total_theater_cost
  total_cost = 1314 := by
  sorry

end chandler_tickets_total_cost_l2168_216802


namespace problem_solution_l2168_216897

theorem problem_solution : (6 * 7 * 8 * 9 * 10) / (6 + 7 + 8 + 9 + 10) = 756 := by
  sorry

end problem_solution_l2168_216897


namespace joey_needs_figures_to_cover_cost_l2168_216898

-- Definitions based on conditions
def cost_sneakers : ℕ := 92
def earnings_per_lawn : ℕ := 8
def lawns : ℕ := 3
def earnings_per_hour : ℕ := 5
def work_hours : ℕ := 10
def price_per_figure : ℕ := 9

-- Total earnings from mowing lawns
def earnings_lawns := lawns * earnings_per_lawn
-- Total earnings from job
def earnings_job := work_hours * earnings_per_hour
-- Total earnings from both
def total_earnings := earnings_lawns + earnings_job
-- Remaining amount to cover the cost
def remaining_amount := cost_sneakers - total_earnings

-- Correct answer based on the problem statement
def collectible_figures_needed := remaining_amount / price_per_figure

-- Lean 4 statement to prove the requirement
theorem joey_needs_figures_to_cover_cost :
  collectible_figures_needed = 2 := by
  sorry

end joey_needs_figures_to_cover_cost_l2168_216898


namespace smallest_number_groups_l2168_216837

theorem smallest_number_groups :
  ∃ x : ℕ, (∀ y : ℕ, (y % 12 = 0 ∧ y % 20 = 0 ∧ y % 6 = 0) → y ≥ x) ∧ 
           (x % 12 = 0 ∧ x % 20 = 0 ∧ x % 6 = 0) ∧ x = 60 :=
by
  sorry

end smallest_number_groups_l2168_216837


namespace swap_square_digit_l2168_216839

theorem swap_square_digit (n : ℕ) (h1 : n ≥ 10 ∧ n < 100) : 
  ∃ (x y : ℕ), n = 10 * x + y ∧ (x < 10 ∧ y < 10) ∧ (y * 100 + x * 10 + y^2 + 20 * x * y - 1) = n * n + 2 * n + 1 :=
by 
    sorry

end swap_square_digit_l2168_216839


namespace largest_inscribed_parabola_area_l2168_216861

noncomputable def maximum_parabolic_area_in_cone (r l : ℝ) : ℝ :=
  (l * r) / 2 * Real.sqrt 3

theorem largest_inscribed_parabola_area (r l : ℝ) : 
  ∃ t : ℝ, t = maximum_parabolic_area_in_cone r l :=
by
  let t_max := (l * r) / 2 * Real.sqrt 3
  use t_max
  sorry

end largest_inscribed_parabola_area_l2168_216861


namespace sand_bucket_capacity_l2168_216873

theorem sand_bucket_capacity
  (sandbox_depth : ℝ)
  (sandbox_width : ℝ)
  (sandbox_length : ℝ)
  (sand_weight_per_cubic_foot : ℝ)
  (water_per_4_trips : ℝ)
  (water_bottle_ounces : ℝ)
  (water_bottle_cost : ℝ)
  (tony_total_money : ℝ)
  (tony_change : ℝ)
  (tony's_bucket_capacity : ℝ) :
  sandbox_depth = 2 →
  sandbox_width = 4 →
  sandbox_length = 5 →
  sand_weight_per_cubic_foot = 3 →
  water_per_4_trips = 3 →
  water_bottle_ounces = 15 →
  water_bottle_cost = 2 →
  tony_total_money = 10 →
  tony_change = 4 →
  tony's_bucket_capacity = 2 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry -- skipping the proof as per instructions

end sand_bucket_capacity_l2168_216873


namespace total_pumped_volume_l2168_216805

def powerJetA_rate : ℕ := 360
def powerJetB_rate : ℕ := 540
def powerJetA_time : ℕ := 30
def powerJetB_time : ℕ := 45

def pump_volume (rate : ℕ) (minutes : ℕ) : ℕ :=
  rate * (minutes / 60)

theorem total_pumped_volume : 
  pump_volume powerJetA_rate powerJetA_time + pump_volume powerJetB_rate powerJetB_time = 585 := 
by
  sorry

end total_pumped_volume_l2168_216805


namespace range_of_ω_l2168_216806

noncomputable def f (x : ℝ) (ω : ℝ) (ϕ : ℝ) : ℝ :=
  Real.cos (ω * x + ϕ)

theorem range_of_ω :
  ∀ (ω : ℝ) (ϕ : ℝ),
    (0 < ω) →
    (-π ≤ ϕ) →
    (ϕ ≤ 0) →
    (∀ x, f x ω ϕ = -f (-x) ω ϕ) →
    (∀ x1 x2, (x1 < x2) → (-π/4 ≤ x1 ∧ x1 ≤ 3*π/16) ∧ (-π/4 ≤ x2 ∧ x2 ≤ 3*π/16) → f x1 ω ϕ ≤ f x2 ω ϕ) →
    (0 < ω ∧ ω ≤ 2) :=
by
  sorry

end range_of_ω_l2168_216806


namespace words_lost_due_to_prohibition_l2168_216895

-- Define the conditions given in the problem.
def number_of_letters := 64
def forbidden_letter := 7
def total_one_letter_words := number_of_letters
def total_two_letter_words := number_of_letters * number_of_letters

-- Define the forbidden letter loss calculation.
def one_letter_words_lost := 1
def two_letter_words_lost := number_of_letters + number_of_letters - 1

-- Define the total words lost calculation.
def total_words_lost := one_letter_words_lost + two_letter_words_lost

-- State the theorem to prove the number of words lost is 128.
theorem words_lost_due_to_prohibition : total_words_lost = 128 :=
by sorry

end words_lost_due_to_prohibition_l2168_216895


namespace sum_of_squares_of_consecutive_integers_is_perfect_square_l2168_216846

theorem sum_of_squares_of_consecutive_integers_is_perfect_square (x : ℤ) :
  ∃ k : ℤ, k ^ 2 = x ^ 2 + (x + 1) ^ 2 + (x ^ 2 * (x + 1) ^ 2) :=
by
  use (x^2 + x + 1)
  sorry

end sum_of_squares_of_consecutive_integers_is_perfect_square_l2168_216846


namespace least_xy_l2168_216889

theorem least_xy (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 1 / (3 * y) = 1 / 9) : x * y = 108 :=
by
  sorry

end least_xy_l2168_216889


namespace seated_students_count_l2168_216851

theorem seated_students_count :
  ∀ (S T standing_students total_attendees : ℕ),
    T = 30 →
    standing_students = 25 →
    total_attendees = 355 →
    total_attendees = S + T + standing_students →
    S = 300 :=
by
  intros S T standing_students total_attendees hT hStanding hTotalAttendees hEquation
  sorry

end seated_students_count_l2168_216851


namespace determine_min_bottles_l2168_216883

-- Define the capacities and constraints
def mediumBottleCapacity : ℕ := 80
def largeBottleCapacity : ℕ := 1200
def additionalBottles : ℕ := 5

-- Define the minimum number of medium-sized bottles Jasmine needs to buy
def minimumMediumBottles (mediumCapacity largeCapacity extras : ℕ) : ℕ :=
  let requiredBottles := largeCapacity / mediumCapacity
  requiredBottles

theorem determine_min_bottles :
  minimumMediumBottles mediumBottleCapacity largeBottleCapacity additionalBottles = 15 :=
by
  sorry

end determine_min_bottles_l2168_216883


namespace smallest_diff_of_YZ_XY_l2168_216865

theorem smallest_diff_of_YZ_XY (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 2509) (h4 : a + b > c) (h5 : b + c > a) (h6 : a + c > b) : b - a = 1 :=
by {
  sorry
}

end smallest_diff_of_YZ_XY_l2168_216865


namespace number_of_numbers_l2168_216864

theorem number_of_numbers (n S : ℕ) 
  (h1 : (S + 26) / n = 15)
  (h2 : (S + 36) / n = 16)
  : n = 10 :=
sorry

end number_of_numbers_l2168_216864
