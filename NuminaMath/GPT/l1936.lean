import Mathlib

namespace part_one_extreme_value_part_two_max_k_l1936_193677

noncomputable def f (x : ℝ) (k : ℝ) : ℝ :=
  x * Real.log x - k * (x - 1)

theorem part_one_extreme_value :
  ∃ x : ℝ, x > 0 ∧ ∀ y > 0, f y 1 ≥ f x 1 ∧ f x 1 = 0 := 
  sorry

theorem part_two_max_k :
  ∀ x : ℝ, ∃ k : ℕ, (1 < x) -> (f x (k:ℝ) + x > 0) ∧ k = 3 :=
  sorry

end part_one_extreme_value_part_two_max_k_l1936_193677


namespace correct_sampling_methods_l1936_193622

def reporter_A_sampling : String :=
  "systematic sampling"

def reporter_B_sampling : String :=
  "systematic sampling"

theorem correct_sampling_methods (constant_flow : Prop)
  (A_interview_method : ∀ t : ℕ, t % 10 = 0)
  (B_interview_method : ∀ n : ℕ, n % 1000 = 0) :
  reporter_A_sampling = "systematic sampling" ∧ reporter_B_sampling = "systematic sampling" :=
by
  sorry

end correct_sampling_methods_l1936_193622


namespace arithmetic_progression_sum_l1936_193600

theorem arithmetic_progression_sum (a d S n : ℤ) (h_a : a = 32) (h_d : d = -4) (h_S : S = 132) :
  (n = 6 ∨ n = 11) :=
by
  -- Start the proof here
  sorry

end arithmetic_progression_sum_l1936_193600


namespace problem_statement_l1936_193660

   noncomputable def g (x : ℝ) : ℝ := (3 * x + 4) / (x + 3)

   def T := {y : ℝ | ∃ (x : ℝ), x ≥ 0 ∧ y = g x}

   theorem problem_statement :
     (∃ N, (∀ y ∈ T, y ≤ N) ∧ N = 3 ∧ N ∉ T) ∧
     (∃ n, (∀ y ∈ T, y ≥ n) ∧ n = 4/3 ∧ n ∈ T) :=
   by
     sorry
   
end problem_statement_l1936_193660


namespace min_box_height_l1936_193650

noncomputable def height_of_box (x : ℝ) := x + 4

def surface_area (x : ℝ) : ℝ := 2 * x^2 + 4 * x * (x + 4)

theorem min_box_height (x h : ℝ) (h₁ : h = height_of_box x) (h₂ : surface_area x ≥ 130) : h ≥ 25 / 3 :=
by sorry

end min_box_height_l1936_193650


namespace courtyard_width_l1936_193653

def width_of_courtyard (w : ℝ) : Prop :=
  28 * 100 * 100 * w = 13788 * 22 * 12

theorem courtyard_width :
  ∃ w : ℝ, width_of_courtyard w ∧ abs (w - 13.012) < 0.001 :=
by
  sorry

end courtyard_width_l1936_193653


namespace shaded_region_area_l1936_193680

section

-- Define points and shapes
structure point := (x : ℝ) (y : ℝ)
def square_side_length : ℝ := 40
def square_area : ℝ := square_side_length * square_side_length

-- Points defining the square and triangles within it
def point_O : point := ⟨0, 0⟩
def point_A : point := ⟨15, 0⟩
def point_B : point := ⟨40, 25⟩
def point_C : point := ⟨40, 40⟩
def point_D1 : point := ⟨25, 40⟩
def point_E : point := ⟨0, 15⟩

-- Function to calculate the area of a triangle given base and height
def triangle_area (base height : ℝ) : ℝ := 0.5 * base * height

-- Areas of individual triangles
def triangle1_area : ℝ := triangle_area 15 15
def triangle2_area : ℝ := triangle_area 25 25
def triangle3_area : ℝ := triangle_area 15 15

-- Total area of the triangles
def total_triangles_area : ℝ := triangle1_area + triangle2_area + triangle3_area

-- Shaded area calculation
def shaded_area : ℝ := square_area - total_triangles_area

-- Statement of the theorem to be proven
theorem shaded_region_area : shaded_area = 1062.5 := by sorry

end

end shaded_region_area_l1936_193680


namespace non_visible_dots_l1936_193632

-- Define the configuration of the dice
def total_dots_on_one_die : ℕ := 1 + 2 + 3 + 4 + 5 + 6
def total_dots_on_two_dice : ℕ := 2 * total_dots_on_one_die
def visible_dots : ℕ := 2 + 3 + 5

-- The statement to prove
theorem non_visible_dots : total_dots_on_two_dice - visible_dots = 32 := by sorry

end non_visible_dots_l1936_193632


namespace probability_divisible_by_5_l1936_193629

def is_three_digit_integer (M : ℕ) : Prop :=
  100 ≤ M ∧ M < 1000

def ones_digit_is_4 (M : ℕ) : Prop :=
  (M % 10) = 4

theorem probability_divisible_by_5 (M : ℕ) (h1 : is_three_digit_integer M) (h2 : ones_digit_is_4 M) :
  (∃ p : ℚ, p = 0) :=
by
  sorry

end probability_divisible_by_5_l1936_193629


namespace mod_multiplication_example_l1936_193657

theorem mod_multiplication_example :
  (98 % 75) * (202 % 75) % 75 = 71 :=
by
  have h1 : 98 % 75 = 23 := by sorry
  have h2 : 202 % 75 = 52 := by sorry
  have h3 : 1196 % 75 = 71 := by sorry
  exact h3

end mod_multiplication_example_l1936_193657


namespace part1_part2_part3_l1936_193670

section ShoppingMall

variable (x y a b : ℝ)
variable (cpaA spaA cpaB spaB : ℝ)
variable (n total_y yuan : ℝ)

-- Conditions given in the problem
def cost_price_A := 160
def selling_price_A := 220
def cost_price_B := 120
def selling_price_B := 160
def total_clothing := 100
def min_A_clothing := 60
def max_budget := 15000
def discount_diff := 4
def max_profit_with_discount := 4950

-- Definitions applied from conditions
def profit_per_piece_A := selling_price_A - cost_price_A
def profit_per_piece_B := selling_price_B - cost_price_B

-- Question 1: Functional relationship between y and x
theorem part1 : 
  (∀ (x : ℝ), x ≥ 0 → x ≤ total_clothing → 
  y = profit_per_piece_A * x + profit_per_piece_B * (total_clothing - x)) →
  y = 20 * x + 4000 := 
sorry

-- Question 2: Maximum profit under given cost constraints
theorem part2 : 
  (min_A_clothing ≤ x ∧ x ≤ 75 ∧ 
  (cost_price_A * x + cost_price_B * (total_clothing - x) ≤ max_budget)) →
  y = 20 * 75 + 4000 → 
  y = 5500 :=
sorry

-- Question 3: Determine a under max profit condition
theorem part3 : 
  (a - b = discount_diff ∧ 0 < a ∧ a < 20 ∧ 
  (20 - a) * 75 + 4000 + 100 * a - 400 = max_profit_with_discount) →
  a = 9 :=
sorry

end ShoppingMall

end part1_part2_part3_l1936_193670


namespace length_of_circle_l1936_193639

-- Define initial speeds and conditions
variables (V1 V2 : ℝ)
variables (L : ℝ) -- Length of the circle

-- Conditions
def initial_condition : Prop := V1 - V2 = 3
def extra_laps_after_speed_increase : Prop := (V1 + 10) - V2 = V1 - V2 + 10

-- Statement representing the mathematical equivalence
theorem length_of_circle
  (h1 : initial_condition V1 V2) 
  (h2 : extra_laps_after_speed_increase V1 V2) :
  L = 1250 := 
sorry

end length_of_circle_l1936_193639


namespace negation_of_implication_l1936_193617

theorem negation_of_implication (a b c : ℝ) :
  ¬ (a > b → a + c > b + c) ↔ (a ≤ b → a + c ≤ b + c) :=
by sorry

end negation_of_implication_l1936_193617


namespace square_line_product_l1936_193613

theorem square_line_product (b : ℝ) (h1 : y = 3) (h2 : y = 7) (h3 := x = 2) (h4 : x = b) : 
  (b = 6 ∨ b = -2) → (6 * -2 = -12) :=
by
  sorry

end square_line_product_l1936_193613


namespace sum_of_interior_angles_6_find_n_from_300_degrees_l1936_193684

-- Definitions and statement for part 1:
def sum_of_interior_angles (n : ℕ) : ℕ :=
  (n - 2) * 180

theorem sum_of_interior_angles_6 :
  sum_of_interior_angles 6 = 720 := 
by
  sorry

-- Definitions and statement for part 2:
def find_n_from_angles (angle : ℕ) : ℕ := 
  (angle / 180) + 2

theorem find_n_from_300_degrees :
  find_n_from_angles 900 = 7 :=
by
  sorry

end sum_of_interior_angles_6_find_n_from_300_degrees_l1936_193684


namespace part1_part2_l1936_193641

noncomputable def f (x a : ℝ) : ℝ := abs (x + 2 * a) + abs (x - 1)

noncomputable def g (a : ℝ) : ℝ := abs ((1 : ℝ) / a + 2 * a) + abs ((1 : ℝ) / a - 1)

theorem part1 (x : ℝ) : f x 1 ≤ 5 ↔ -3 ≤ x ∧ x ≤ 2 := by
  sorry

theorem part2 (a : ℝ) (h : a ≠ 0) : g a ≤ 4 ↔ (1 / 2) ≤ a ∧ a ≤ (3 / 2) := by
  sorry

end part1_part2_l1936_193641


namespace Cartesian_eq_C2_correct_distance_AB_correct_l1936_193624

-- Part I: Proving the Cartesian equation of curve (C2)
noncomputable def equation_of_C2 (x y : ℝ) (α : ℝ) : Prop :=
  x = 4 * Real.cos α ∧ y = 4 + 4 * Real.sin α

def Cartesian_eq_C2 (x y : ℝ) : Prop :=
  x^2 + (y - 4)^2 = 16

theorem Cartesian_eq_C2_correct (x y α : ℝ) (h : equation_of_C2 x y α) : Cartesian_eq_C2 x y :=
by sorry

-- Part II: Proving the distance |AB| given polar equations
noncomputable def polar_eq_C1 (theta : ℝ) : ℝ :=
  4 * Real.sin theta

noncomputable def polar_eq_C2 (theta : ℝ) : ℝ :=
  8 * Real.sin theta

def distance_AB (rho1 rho2 : ℝ) : ℝ :=
  abs (rho1 - rho2)

theorem distance_AB_correct : distance_AB (polar_eq_C1 (π / 3)) (polar_eq_C2 (π / 3)) = 2 * Real.sqrt 3 :=
by sorry

end Cartesian_eq_C2_correct_distance_AB_correct_l1936_193624


namespace major_axis_length_l1936_193668

theorem major_axis_length (r : ℝ) (minor_axis : ℝ) (major_axis : ℝ) 
  (h1 : r = 2) 
  (h2 : minor_axis = 2 * r) 
  (h3 : major_axis = minor_axis + 0.75 * minor_axis) : 
  major_axis = 7 := 
by 
  sorry

end major_axis_length_l1936_193668


namespace at_least_one_not_less_than_two_l1936_193661

theorem at_least_one_not_less_than_two
  (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) :
  (2 ≤ (y / x + y / z)) ∨ (2 ≤ (z / x + z / y)) ∨ (2 ≤ (x / z + x / y)) :=
sorry

end at_least_one_not_less_than_two_l1936_193661


namespace nancy_age_l1936_193655

variable (n g : ℕ)

theorem nancy_age (h1 : g = 10 * n) (h2 : g - n = 45) : n = 5 :=
by
  sorry

end nancy_age_l1936_193655


namespace cubic_vs_square_ratio_l1936_193605

theorem cubic_vs_square_ratio 
  (s r : ℝ) 
  (hs : 0 < s) 
  (hr : 0 < r) 
  (h : r < s) : 
  (s^3 - r^3) / (s^3 + r^3) > (s^2 - r^2) / (s^2 + r^2) :=
by sorry

end cubic_vs_square_ratio_l1936_193605


namespace max_dinners_for_7_people_max_dinners_for_8_people_l1936_193683

def max_dinners_with_new_neighbors (n : ℕ) : ℕ :=
  if n = 7 ∨ n = 8 then 3 else 0

theorem max_dinners_for_7_people : max_dinners_with_new_neighbors 7 = 3 := sorry

theorem max_dinners_for_8_people : max_dinners_with_new_neighbors 8 = 3 := sorry

end max_dinners_for_7_people_max_dinners_for_8_people_l1936_193683


namespace min_brilliant_triple_product_l1936_193671

theorem min_brilliant_triple_product :
  ∃ a b c : ℕ, a > b ∧ b > c ∧ Prime a ∧ Prime b ∧ Prime c ∧ (a = b + 2 * c) ∧ (∃ k : ℕ, (a + b + c) = k^2) ∧ (a * b * c = 35651) :=
by
  sorry

end min_brilliant_triple_product_l1936_193671


namespace real_solution_l1936_193611

noncomputable def condition_1 (x : ℝ) : Prop := 
  4 ≤ x / (2 * x - 7)

noncomputable def condition_2 (x : ℝ) : Prop := 
  x / (2 * x - 7) < 10

noncomputable def solution_set : Set ℝ :=
  { x | (70 / 19 : ℝ) < x ∧ x ≤ 4 }

theorem real_solution (x : ℝ) : 
  (condition_1 x ∧ condition_2 x) ↔ x ∈ solution_set :=
sorry

end real_solution_l1936_193611


namespace total_ducks_in_lake_l1936_193603

/-- 
Problem: Determine the total number of ducks in the lake after more ducks join.

Conditions:
- Initially, there are 13 ducks in the lake.
- 20 more ducks come to join them.
-/

def initial_ducks : Nat := 13

def new_ducks : Nat := 20

theorem total_ducks_in_lake : initial_ducks + new_ducks = 33 := by
  sorry -- Proof to be filled in later

end total_ducks_in_lake_l1936_193603


namespace acute_angle_sum_l1936_193698

theorem acute_angle_sum (n : ℕ) (hn : n ≥ 4) (M m: ℕ) 
  (hM : M = 3) (hm : m = 0) : M + m = 3 := 
by 
  sorry

end acute_angle_sum_l1936_193698


namespace compare_neg_fractions_and_neg_values_l1936_193654

theorem compare_neg_fractions_and_neg_values :
  (- (3 : ℚ) / 4 > - (4 : ℚ) / 5) ∧ (-(-3 : ℤ) > -|(3 : ℤ)|) :=
by
  apply And.intro
  sorry
  sorry

end compare_neg_fractions_and_neg_values_l1936_193654


namespace count_valid_triangles_l1936_193640

/-- 
Define the problem constraints: scalene triangles with side lengths a, b, c, 
where a < b < c, a + c = 2b, and a + b + c ≤ 30.
-/
def is_valid_triangle (a b c : ℕ) : Prop :=
  a < b ∧ b < c ∧ a + c = 2 * b ∧ a + b + c ≤ 30

/-- 
Statement of the problem: Prove that there are 20 distinct triangles satisfying the above constraints. 
-/
theorem count_valid_triangles : ∃ n, n = 20 ∧ (∀ {a b c : ℕ}, is_valid_triangle a b c → n = 20) :=
sorry

end count_valid_triangles_l1936_193640


namespace tan_alpha_eq_neg_one_l1936_193602

-- Define the point P and the angle α
def P : ℝ × ℝ := (-1, 1)
def α : ℝ := sorry  -- α is the angle whose terminal side passes through P

-- Statement to be proved
theorem tan_alpha_eq_neg_one (h : (P.1, P.2) = (-1, 1)) : Real.tan α = -1 :=
by
  sorry

end tan_alpha_eq_neg_one_l1936_193602


namespace combined_proposition_range_l1936_193649

def p (a : ℝ) : Prop := ∀ x ∈ ({1, 2} : Set ℝ), 3 * x^2 - a ≥ 0

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0

theorem combined_proposition_range (a : ℝ) : 
  (p a ∧ q a) ↔ (a ≤ -2 ∨ (1 ≤ a ∧ a ≤ 3)) := 
  sorry

end combined_proposition_range_l1936_193649


namespace percentage_increase_overtime_rate_l1936_193664

theorem percentage_increase_overtime_rate :
  let regular_rate := 16
  let regular_hours_limit := 30
  let total_earnings := 760
  let total_hours_worked := 40
  let overtime_rate := 28 -- This is calculated as $280/10 from the solution.
  let increase_in_hourly_rate := overtime_rate - regular_rate
  let percentage_increase := (increase_in_hourly_rate / regular_rate) * 100
  percentage_increase = 75 :=
by {
  sorry
}

end percentage_increase_overtime_rate_l1936_193664


namespace problem_solution_l1936_193604

theorem problem_solution (n : ℕ) (h : n^3 - n = 5814) : (n % 2 = 0) :=
by sorry

end problem_solution_l1936_193604


namespace shooter_scores_l1936_193673

theorem shooter_scores
    (x y z : ℕ)
    (hx : x + y + z > 11)
    (hscore: 8 * x + 9 * y + 10 * z = 100) :
    (x + y + z = 12) ∧ ((x = 10 ∧ y = 0 ∧ z = 2) ∨ (x = 9 ∧ y = 2 ∧ z = 1) ∨ (x = 8 ∧ y = 4 ∧ z = 0)) :=
by
  sorry

end shooter_scores_l1936_193673


namespace arithmetic_sequence_a12_bound_l1936_193694

theorem arithmetic_sequence_a12_bound (a_1 d : ℤ) (h8 : a_1 + 7 * d ≥ 15) (h9 : a_1 + 8 * d ≤ 13) : 
  a_1 + 11 * d ≤ 7 :=
by
  sorry

end arithmetic_sequence_a12_bound_l1936_193694


namespace katya_attached_squares_perimeter_l1936_193686

theorem katya_attached_squares_perimeter :
  let p1 := 100 -- Perimeter of the larger square
  let p2 := 40  -- Perimeter of the smaller square
  let s1 := p1 / 4 -- Side length of the larger square
  let s2 := p2 / 4 -- Side length of the smaller square
  let combined_perimeter_without_internal_sides := p1 + p2
  let actual_perimeter := combined_perimeter_without_internal_sides - 2 * s2
  actual_perimeter = 120 :=
by
  sorry

end katya_attached_squares_perimeter_l1936_193686


namespace negation_example_l1936_193608

theorem negation_example :
  ¬ (∀ n : ℕ, (n^2 + n) % 2 = 0) ↔ ∃ n : ℕ, (n^2 + n) % 2 ≠ 0 :=
by
  sorry

end negation_example_l1936_193608


namespace evaluate_functions_l1936_193609

def f (x : ℝ) := x + 2
def g (x : ℝ) := 2 * x^2 - 4
def h (x : ℝ) := x + 1

theorem evaluate_functions : f (g (h 3)) = 30 := by
  sorry

end evaluate_functions_l1936_193609


namespace price_decrease_percentage_l1936_193643

theorem price_decrease_percentage (P₀ P₁ P₂ : ℝ) (x : ℝ) :
  P₀ = 1 → P₁ = P₀ * 1.25 → P₂ = P₁ * (1 - x / 100) → P₂ = 1 → x = 20 :=
by
  intros h₀ h₁ h₂ h₃
  sorry

end price_decrease_percentage_l1936_193643


namespace find_f_find_g_l1936_193638

-- Problem 1: Finding f(x) given f(x+1) = x^2 - 2x
theorem find_f (f : ℝ → ℝ) (h : ∀ x, f (x + 1) = x^2 - 2 * x) :
  ∀ x, f x = x^2 - 4 * x + 3 :=
sorry

-- Problem 2: Finding g(x) given roots and a point
theorem find_g (g : ℝ → ℝ) (h1 : g (-2) = 0) (h2 : g 3 = 0) (h3 : g 0 = -3) :
  ∀ x, g x = (1 / 2) * x^2 - (1 / 2) * x - 3 :=
sorry

end find_f_find_g_l1936_193638


namespace Sadie_l1936_193697

theorem Sadie's_homework_problems (T : ℝ) 
  (h1 : 0.40 * T = A) 
  (h2 : 0.5 * A = 28) 
  : T = 140 := 
by
  sorry

end Sadie_l1936_193697


namespace sequences_power_of_two_l1936_193625

open scoped Classical

theorem sequences_power_of_two (n : ℕ) (a b : Fin n → ℚ)
  (h1 : (∃ i j, i < j ∧ a i = a j) → ∀ i, a i = b i)
  (h2 : {p | ∃ (i j : Fin n), i < j ∧ (a i + a j = p)} = {q | ∃ (i j : Fin n), i < j ∧ (b i + b j = q)})
  (h3 : ∃ i j, i < j ∧ a i ≠ b i) :
  ∃ k : ℕ, n = 2 ^ k := 
sorry

end sequences_power_of_two_l1936_193625


namespace solve_eq1_solve_eq2_l1936_193659

-- Define the first equation
def eq1 (x : ℚ) : Prop := x / (x - 1) = 3 / (2*x - 2) - 2

-- Define the valid solution for the first equation
def sol1 : ℚ := 7 / 6

-- Theorem for the first equation
theorem solve_eq1 : eq1 sol1 :=
by
  sorry

-- Define the second equation
def eq2 (x : ℚ) : Prop := (5*x + 2) / (x^2 + x) = 3 / (x + 1)

-- Theorem for the second equation: there is no valid solution
theorem solve_eq2 : ¬ ∃ x : ℚ, eq2 x :=
by
  sorry

end solve_eq1_solve_eq2_l1936_193659


namespace perpendicular_line_eq_slope_intercept_l1936_193627

/-- Given the line 3x - 6y = 9, find the equation of the line perpendicular to it passing through the point (2, -3) in slope-intercept form. The resulting equation should be y = -2x + 1. -/
theorem perpendicular_line_eq_slope_intercept :
  ∃ (m b : ℝ), (m = -2) ∧ (b = 1) ∧ (∀ x y : ℝ, y = m * x + b ↔ (3 * x - 6 * y = 9) ∧ y = -2 * x + 1) :=
sorry

end perpendicular_line_eq_slope_intercept_l1936_193627


namespace train_length_l1936_193666

noncomputable def jogger_speed_kmh : ℝ := 9
noncomputable def train_speed_kmh : ℝ := 45
noncomputable def head_start : ℝ := 270
noncomputable def passing_time : ℝ := 39

noncomputable def kmh_to_ms (speed: ℝ) : ℝ := speed * (1000 / 3600)

theorem train_length (l : ℝ) 
  (v_j := kmh_to_ms jogger_speed_kmh)
  (v_t := kmh_to_ms train_speed_kmh)
  (d_h := head_start)
  (t := passing_time) :
  l = 120 :=
by 
  sorry

end train_length_l1936_193666


namespace sum_of_roots_l1936_193636

theorem sum_of_roots (x1 x2 : ℝ) (h : x1^2 + 5*x1 - 1 = 0 ∧ x2^2 + 5*x2 - 1 = 0) : x1 + x2 = -5 :=
sorry

end sum_of_roots_l1936_193636


namespace simplify_and_evaluate_l1936_193642

theorem simplify_and_evaluate (x : ℝ) (h : x = -3) :
  (1 - (1 / (x - 1))) / ((x ^ 2 - 4 * x + 4) / (x ^ 2 - 1)) = (2 / 5) :=
by
  rw [h]
  sorry

end simplify_and_evaluate_l1936_193642


namespace matrix_eq_l1936_193652

open Matrix

def matA : Matrix (Fin 2) (Fin 2) ℤ := ![![1, 3], ![4, 2]]
def matI : Matrix (Fin 2) (Fin 2) ℤ := 1

theorem matrix_eq (A : Matrix (Fin 2) (Fin 2) ℤ)
  (hA : A = ![![1, 3], ![4, 2]]) :
  A ^ 7 = 9936 * A ^ 2 + 12400 * 1 :=
  by
    sorry

end matrix_eq_l1936_193652


namespace at_least_one_ge_two_l1936_193675

theorem at_least_one_ge_two (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    max (a + 1/b) (max (b + 1/c) (c + 1/a)) ≥ 2 :=
sorry

end at_least_one_ge_two_l1936_193675


namespace girls_not_join_field_trip_l1936_193630

theorem girls_not_join_field_trip (total_students : ℕ) (number_of_boys : ℕ) (number_on_trip : ℕ)
  (h_total : total_students = 18)
  (h_boys : number_of_boys = 8)
  (h_equal : number_on_trip = number_of_boys) :
  total_students - number_of_boys - number_on_trip = 2 := by
sorry

end girls_not_join_field_trip_l1936_193630


namespace race_course_length_l1936_193628

theorem race_course_length (v : ℝ) (d : ℝ) (h1 : d = 7 * (d - 120)) : d = 140 :=
sorry

end race_course_length_l1936_193628


namespace quadratic_inequality_solution_l1936_193619

theorem quadratic_inequality_solution (x : ℝ) : 
  (x^2 < x + 6) ↔ (-2 < x ∧ x < 3) := 
by
  sorry

end quadratic_inequality_solution_l1936_193619


namespace compare_A_B_C_l1936_193658

-- Define the expressions A, B, and C
def A : ℚ := (2010 / 2009) + (2010 / 2011)
def B : ℚ := (2010 / 2011) + (2012 / 2011)
def C : ℚ := (2011 / 2010) + (2011 / 2012)

-- The statement asserting A is the greatest
theorem compare_A_B_C : A > B ∧ A > C := by
  sorry

end compare_A_B_C_l1936_193658


namespace remainder_product_mod_five_l1936_193688

-- Define the conditions as congruences
def num1 : ℕ := 14452
def num2 : ℕ := 15652
def num3 : ℕ := 16781

-- State the main theorem using the conditions and the given problem
theorem remainder_product_mod_five : 
  (num1 % 5 = 2) → 
  (num2 % 5 = 2) → 
  (num3 % 5 = 1) → 
  ((num1 * num2 * num3) % 5 = 4) :=
by
  intros
  sorry

end remainder_product_mod_five_l1936_193688


namespace candy_bar_cost_l1936_193646

theorem candy_bar_cost {initial_money left_money cost_bar : ℕ} 
                        (h_initial : initial_money = 4)
                        (h_left : left_money = 3)
                        (h_cost : cost_bar = initial_money - left_money) :
                        cost_bar = 1 :=
by 
  sorry -- Proof is not required as per the instructions

end candy_bar_cost_l1936_193646


namespace interest_rate_unique_l1936_193612

theorem interest_rate_unique (P r : ℝ) (h₁ : P * (1 + 3 * r) = 300) (h₂ : P * (1 + 8 * r) = 400) : r = 1 / 12 :=
by {
  sorry
}

end interest_rate_unique_l1936_193612


namespace total_telephone_bill_second_month_l1936_193692

variable (F C : ℝ)

-- Elvin's total telephone bill for January is 40 dollars
axiom january_bill : F + C = 40

-- The charge for calls in the second month is twice the charge for calls in January
axiom second_month_call_charge : ∃ C2, C2 = 2 * C

-- Proof that the total telephone bill for the second month is 40 + C
theorem total_telephone_bill_second_month : 
  ∃ S, S = F + 2 * C ∧ S = 40 + C :=
sorry

end total_telephone_bill_second_month_l1936_193692


namespace find_integer_n_l1936_193637

theorem find_integer_n : ∃ n : ℕ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ 123456 [MOD 8] ∧ n = 0 :=
by
  sorry

end find_integer_n_l1936_193637


namespace remainder_130_div_k_l1936_193691

theorem remainder_130_div_k (k : ℕ) (h_positive : k > 0)
  (h_remainder : 84 % (k*k) = 20) : 
  130 % k = 2 := 
by sorry

end remainder_130_div_k_l1936_193691


namespace minimum_trains_needed_l1936_193678

theorem minimum_trains_needed (n : ℕ) (h : 50 * n >= 645) : n = 13 :=
by
  sorry

end minimum_trains_needed_l1936_193678


namespace math_problem_l1936_193689

theorem math_problem :
  (1 / (1 / (1 / (1 / (3 + 2 : ℝ) - 1 : ℝ) - 1 : ℝ) - 1 : ℝ) - 1 : ℝ) = -13 / 9 :=
by
  -- proof goes here
  sorry

end math_problem_l1936_193689


namespace true_proposition_among_options_l1936_193699

theorem true_proposition_among_options :
  (∀ (x y : ℝ), (x > |y|) → (x > y)) ∧
  (¬ (∀ (x : ℝ), (x > 1) → (x^2 > 1))) ∧
  (¬ (∀ (x : ℤ), (x = 1) → (x^2 + x - 2 = 0))) ∧
  (¬ (∀ (x : ℝ), (x^2 > 0) → (x > 1))) :=
by
  sorry

end true_proposition_among_options_l1936_193699


namespace slope_tangent_line_at_zero_l1936_193607

noncomputable def f (x : ℝ) : ℝ := (2 * x - 5) / (x^2 + 1)

theorem slope_tangent_line_at_zero : 
  (deriv f 0) = 2 :=
sorry

end slope_tangent_line_at_zero_l1936_193607


namespace greatest_integer_a_l1936_193620

-- Define formal properties and state the main theorem.
theorem greatest_integer_a (a : ℤ) : (∀ x : ℝ, ¬(x^2 + (a:ℝ) * x + 15 = 0)) → (a ≤ 7) :=
by
  intro h
  sorry

end greatest_integer_a_l1936_193620


namespace chocolates_per_student_class_7B_l1936_193687

theorem chocolates_per_student_class_7B :
  (∃ (x : ℕ), 9 * x < 288 ∧ 10 * x > 300 ∧ x = 31) :=
by
  use 31
  -- proof steps omitted here
  sorry

end chocolates_per_student_class_7B_l1936_193687


namespace min_value_of_3x_plus_4y_l1936_193634

open Real

theorem min_value_of_3x_plus_4y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 3 * y = 5 * x * y) : 3 * x + 4 * y ≥ 5 :=
sorry

end min_value_of_3x_plus_4y_l1936_193634


namespace quadratic_inequality_solution_l1936_193651

theorem quadratic_inequality_solution 
  (a b : ℝ) 
  (h1 : (∀ x : ℝ, x^2 + a * x + b > 0 → (x < 3 ∨ x > 1))) :
  ∀ x : ℝ, a * x + b < 0 → x > 3 / 4 := 
by 
  sorry

end quadratic_inequality_solution_l1936_193651


namespace quadratic_inequality_solution_l1936_193618

theorem quadratic_inequality_solution (x : ℝ) :
  (x^2 - 5 * x - 6 > 0) ↔ (x < -1 ∨ x > 6) := 
by
  sorry

end quadratic_inequality_solution_l1936_193618


namespace find_a_2b_3c_l1936_193626

noncomputable def a : ℝ := 28
noncomputable def b : ℝ := 32
noncomputable def c : ℝ := -3

def ineq_condition (x : ℝ) : Prop := (x < -3) ∨ (abs (x - 30) ≤ 2)

theorem find_a_2b_3c (a b c : ℝ) (h₁ : a < b)
  (h₂ : ∀ x : ℝ, (x < -3 ∨ abs (x - 30) ≤ 2) ↔ ((x - a)*(x - b)/(x - c) ≤ 0)) :
  a + 2 * b + 3 * c = 83 :=
by
  sorry

end find_a_2b_3c_l1936_193626


namespace louis_current_age_l1936_193601

-- Define the constants for years to future and future age of Carla
def years_to_future : ℕ := 6
def carla_future_age : ℕ := 30

-- Define the sum of current ages
def sum_current_ages : ℕ := 55

-- State the theorem
theorem louis_current_age :
  ∃ (c l : ℕ), (c + years_to_future = carla_future_age) ∧ (c + l = sum_current_ages) ∧ (l = 31) :=
sorry

end louis_current_age_l1936_193601


namespace matthews_contribution_l1936_193695

theorem matthews_contribution 
  (total_cost : ℝ) (yen_amount : ℝ) (conversion_rate : ℝ)
  (h1 : total_cost = 18)
  (h2 : yen_amount = 2500)
  (h3 : conversion_rate = 140) :
  (total_cost - (yen_amount / conversion_rate)) = 0.143 :=
by sorry

end matthews_contribution_l1936_193695


namespace percentage_ownership_l1936_193610

theorem percentage_ownership (total students_cats students_dogs : ℕ) (h1 : total = 500) (h2 : students_cats = 75) (h3 : students_dogs = 125):
  (students_cats / total : ℝ) = 0.15 ∧
  (students_dogs / total : ℝ) = 0.25 :=
by
  sorry

end percentage_ownership_l1936_193610


namespace mass_percentage_of_H_in_H2O_is_11_19_l1936_193681

def mass_of_hydrogen : Float := 1.008
def mass_of_oxygen : Float := 16.00
def mass_of_H2O : Float := 2 * mass_of_hydrogen + mass_of_oxygen
def mass_percentage_hydrogen : Float :=
  (2 * mass_of_hydrogen / mass_of_H2O) * 100

theorem mass_percentage_of_H_in_H2O_is_11_19 :
  mass_percentage_hydrogen = 11.19 :=
  sorry

end mass_percentage_of_H_in_H2O_is_11_19_l1936_193681


namespace doubled_money_is_1_3_l1936_193662

-- Define the amounts of money Alice and Bob have
def alice_money := (2 : ℚ) / 5
def bob_money := (1 : ℚ) / 4

-- Define the total money before doubling
def total_money_before_doubling := alice_money + bob_money

-- Define the total money after doubling
def total_money_after_doubling := 2 * total_money_before_doubling

-- State the proposition to prove
theorem doubled_money_is_1_3 : total_money_after_doubling = 1.3 := by
  -- The proof will be filled in here
  sorry

end doubled_money_is_1_3_l1936_193662


namespace digit_appears_in_3n_l1936_193663

-- Define a function to check if a digit is in a number
def contains_digit (n : ℕ) (d : ℕ) : Prop :=
  ∃ k : ℕ, n / 10^k % 10 = d

-- Define the statement that n does not contain the digits 1, 2, or 9
def does_not_contain_1_2_9 (n : ℕ) : Prop :=
  ¬ (contains_digit n 1 ∨ contains_digit n 2 ∨ contains_digit n 9)

theorem digit_appears_in_3n (n : ℕ) (hn : 1 ≤ n) (h : does_not_contain_1_2_9 n) :
  contains_digit (3 * n) 1 ∨ contains_digit (3 * n) 2 ∨ contains_digit (3 * n) 9 :=
by
  sorry

end digit_appears_in_3n_l1936_193663


namespace maximum_value_of_expression_l1936_193679

theorem maximum_value_of_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a * b = 1) :
  (1 / (a + 9 * b) + 1 / (9 * a + b)) ≤ 5 / 24 := sorry

end maximum_value_of_expression_l1936_193679


namespace observations_decrement_l1936_193682

theorem observations_decrement (n : ℤ) (h_n_pos : n > 0) : 200 - 15 = 185 :=
by
  sorry

end observations_decrement_l1936_193682


namespace checkerboard_probability_l1936_193674

def total_squares (n : ℕ) : ℕ :=
  n * n

def perimeter_squares (n : ℕ) : ℕ :=
  4 * n - 4

def non_perimeter_squares (n : ℕ) : ℕ :=
  total_squares n - perimeter_squares n

def probability_non_perimeter_square (n : ℕ) : ℚ :=
  non_perimeter_squares n / total_squares n

theorem checkerboard_probability :
  probability_non_perimeter_square 10 = 16 / 25 :=
by
  sorry

end checkerboard_probability_l1936_193674


namespace slope_of_line_intersecting_hyperbola_l1936_193635

theorem slope_of_line_intersecting_hyperbola 
  (A B : ℝ × ℝ)
  (hA : A.1^2 - A.2^2 = 1)
  (hB : B.1^2 - B.2^2 = 1)
  (midpoint_condition : (A.1 + B.1) / 2 = 2 ∧ (A.2 + B.2) / 2 = 1) :
  (B.2 - A.2) / (B.1 - A.1) = 2 :=
by
  sorry

end slope_of_line_intersecting_hyperbola_l1936_193635


namespace solve_inequality_l1936_193616

theorem solve_inequality (x : ℝ) : x + 2 < 1 ↔ x < -1 := sorry

end solve_inequality_l1936_193616


namespace perpendicular_vectors_m_value_l1936_193614

theorem perpendicular_vectors_m_value : 
  ∀ (m : ℝ), ((2 : ℝ) * (1 : ℝ) + (m * (1 / 2)) + (1 * 2) = 0) → m = -8 :=
by
  intro m
  intro h
  sorry

end perpendicular_vectors_m_value_l1936_193614


namespace probability_XOXOX_l1936_193696

theorem probability_XOXOX (n_X n_O n_total : ℕ) (h_total : n_X + n_O = n_total)
  (h_X : n_X = 3) (h_O : n_O = 2) (h_total' : n_total = 5) :
  (1 / ↑(Nat.choose n_total n_X)) = (1 / 10) :=
by
  sorry

end probability_XOXOX_l1936_193696


namespace negation_prop_l1936_193615

open Classical

variable (x : ℝ)

theorem negation_prop :
    (∃ x : ℝ, x^2 + 2*x + 2 < 0) = False ↔
    (∀ x : ℝ, x^2 + 2*x + 2 ≥ 0) :=
by
    sorry

end negation_prop_l1936_193615


namespace responses_needed_l1936_193656

-- Define the given conditions
def rate : ℝ := 0.80
def num_mailed : ℕ := 375

-- Statement to prove
theorem responses_needed :
  rate * num_mailed = 300 := by
  sorry

end responses_needed_l1936_193656


namespace bus_driver_hours_l1936_193685

theorem bus_driver_hours (h : ℕ) (regular_rate : ℕ) (extra_rate1 : ℕ) (extra_rate2 : ℕ) (total_earnings : ℕ)
  (h1 : regular_rate = 14)
  (h2 : extra_rate1 = (14 + (14 * 35 / 100)))
  (h3: extra_rate2 = (14 + (14 * 75 / 100)))
  (h4: total_earnings = 1230)
  (h5: total_earnings = 40 * regular_rate + 10 * extra_rate1 + (h - 50) * extra_rate2)
  (condition : 50 < h) :
  h = 69 :=
by
  sorry

end bus_driver_hours_l1936_193685


namespace side_length_of_square_l1936_193665

theorem side_length_of_square (d : ℝ) (h_d : d = 2 * Real.sqrt 2) : 
  ∃ s : ℝ, d = s * Real.sqrt 2 ∧ s = 2 := by
  sorry

end side_length_of_square_l1936_193665


namespace total_insects_eaten_l1936_193645

theorem total_insects_eaten : 
  (5 * 6) + (3 * (2 * 6)) = 66 :=
by
  /- We'll calculate the total number of insects eaten by combining the amounts eaten by the geckos and lizards -/
  sorry

end total_insects_eaten_l1936_193645


namespace trig_identity_l1936_193669

theorem trig_identity (α : ℝ) (h : Real.tan α = 4) : 
  (1 + Real.cos (2 * α) + 8 * Real.sin α ^ 2) / Real.sin (2 * α) = 65 / 4 :=
by
  sorry

end trig_identity_l1936_193669


namespace find_two_digit_number_l1936_193647

theorem find_two_digit_number (x y : ℕ) (h1 : 10 * x + y = 4 * (x + y) + 3) (h2 : 10 * x + y = 3 * x * y + 5) : 10 * x + y = 23 :=
by {
  sorry
}

end find_two_digit_number_l1936_193647


namespace find_length_of_PB_l1936_193690

theorem find_length_of_PB
  (PA : ℝ) -- Define PA
  (h_PA : PA = 4) -- Condition PA = 4
  (PB : ℝ) -- Define PB
  (PT : ℝ) -- Define PT
  (h_PT : PT = PB - 2 * PA) -- Condition PT = PB - 2 * PA
  (h_power_of_a_point : PA * PB = PT^2) -- Condition PA * PB = PT^2
  : PB = 16 :=
sorry

end find_length_of_PB_l1936_193690


namespace convert_444_quinary_to_octal_l1936_193676

def quinary_to_decimal (n : ℕ) : ℕ :=
  let d2 := (n / 100) * 25
  let d1 := ((n % 100) / 10) * 5
  let d0 := (n % 10)
  d2 + d1 + d0

def decimal_to_octal (n : ℕ) : ℕ :=
  let r2 := (n / 64)
  let n2 := (n % 64)
  let r1 := (n2 / 8)
  let r0 := (n2 % 8)
  r2 * 100 + r1 * 10 + r0

theorem convert_444_quinary_to_octal :
  decimal_to_octal (quinary_to_decimal 444) = 174 := by
  sorry

end convert_444_quinary_to_octal_l1936_193676


namespace range_of_a_l1936_193672

theorem range_of_a (a : ℝ) : 
  (∀ (x : ℝ) (θ : ℝ), 0 ≤ θ ∧ θ ≤ π / 2 → 
    (x + 3 + 2 * (Real.sin θ) * (Real.cos θ))^2 + (x + a * (Real.sin θ) + a * (Real.cos θ))^2 ≥ 1 / 8) → 
  a ≥ 7 / 2 ∨ a ≤ Real.sqrt 6 :=
sorry

end range_of_a_l1936_193672


namespace time_to_travel_to_shop_l1936_193623

-- Define the distance and speed as given conditions
def distance : ℕ := 184
def speed : ℕ := 23

-- Define the time taken for the journey
def time_taken (d : ℕ) (s : ℕ) : ℕ := d / s

-- Statement to prove that the time taken is 8 hours
theorem time_to_travel_to_shop : time_taken distance speed = 8 := by
  -- The proof is omitted
  sorry

end time_to_travel_to_shop_l1936_193623


namespace perpendicular_vectors_l1936_193667

theorem perpendicular_vectors (b : ℚ) : 
  (4 * b - 15 = 0) → (b = 15 / 4) :=
by
  intro h
  sorry

end perpendicular_vectors_l1936_193667


namespace prob_B_takes_second_shot_correct_prob_A_takes_ith_shot_correct_expected_A_shots_correct_l1936_193606

noncomputable def prob_A_makes_shot : ℝ := 0.6
noncomputable def prob_B_makes_shot : ℝ := 0.8
noncomputable def prob_A_starts : ℝ := 0.5
noncomputable def prob_B_starts : ℝ := 0.5

noncomputable def prob_B_takes_second_shot : ℝ :=
  prob_A_starts * (1 - prob_A_makes_shot) + prob_B_starts * prob_B_makes_shot

theorem prob_B_takes_second_shot_correct :
  prob_B_takes_second_shot = 0.6 :=
  sorry

noncomputable def prob_A_takes_nth_shot (n : ℕ) : ℝ :=
  let p₁ := 0.5
  let recurring_prob := (1 / 6) * ((2 / 5)^(n-1))
  (1 / 3) + recurring_prob

theorem prob_A_takes_ith_shot_correct (i : ℕ) :
  prob_A_takes_nth_shot i = (1 / 3) + (1 / 6) * ((2 / 5)^(i - 1)) :=
  sorry

noncomputable def expected_A_shots (n : ℕ) : ℝ :=
  let geometric_sum := ((2 / 5)^n - 1) / (1 - (2 / 5))
  (1 / 6) * geometric_sum + (n / 3)

theorem expected_A_shots_correct (n : ℕ) :
  expected_A_shots n = (5 / 18) * (1 - (2 / 5)^n) + (n / 3) :=
  sorry

end prob_B_takes_second_shot_correct_prob_A_takes_ith_shot_correct_expected_A_shots_correct_l1936_193606


namespace investment_of_c_l1936_193648

variable (P_a P_b P_c C_a C_b C_c : ℝ)

theorem investment_of_c (h1 : P_b = 3500) 
                        (h2 : P_a - P_c = 1399.9999999999998) 
                        (h3 : C_a = 8000) 
                        (h4 : C_b = 10000) 
                        (h5 : P_a / C_a = P_b / C_b) 
                        (h6 : P_c / C_c = P_b / C_b) : 
                        C_c = 40000 := 
by 
  sorry

end investment_of_c_l1936_193648


namespace rectangle_ratio_l1936_193621

theorem rectangle_ratio (a b c d : ℝ)
  (h1 : (a * b) / (c * d) = 0.16)
  (h2 : a / c = b / d) :
  a / c = 0.4 ∧ b / d = 0.4 :=
by 
  sorry

end rectangle_ratio_l1936_193621


namespace green_paint_quarts_l1936_193693

theorem green_paint_quarts (blue green white : ℕ) (h_ratio : 3 = blue ∧ 2 = green ∧ 4 = white) 
  (h_white_paint : white = 12) : green = 6 := 
by
  sorry

end green_paint_quarts_l1936_193693


namespace combined_height_l1936_193644

theorem combined_height (h_John : ℕ) (h_Lena : ℕ) (h_Rebeca : ℕ)
  (cond1 : h_John = 152)
  (cond2 : h_John = h_Lena + 15)
  (cond3 : h_Rebeca = h_John + 6) :
  h_Lena + h_Rebeca = 295 :=
by
  sorry

end combined_height_l1936_193644


namespace planned_pigs_correct_l1936_193631

-- Define initial number of animals
def initial_cows : ℕ := 2
def initial_pigs : ℕ := 3
def initial_goats : ℕ := 6

-- Define planned addition of animals
def added_cows : ℕ := 3
def added_goats : ℕ := 2
def total_animals : ℕ := 21

-- Define the total planned number of pigs to verify:
def planned_pigs := 8

-- State the final number of pigs to be proven
theorem planned_pigs_correct : 
  initial_cows + initial_pigs + initial_goats + added_cows + planned_pigs + added_goats = total_animals :=
by
  sorry

end planned_pigs_correct_l1936_193631


namespace seating_arrangement_l1936_193633

theorem seating_arrangement (x y : ℕ) (h : 9 * x + 7 * y = 112) : x = 7 :=
by
  sorry

end seating_arrangement_l1936_193633
