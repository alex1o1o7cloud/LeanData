import Mathlib

namespace number_of_children_l45_4562

def male_adults : ℕ := 60
def female_adults : ℕ := 60
def total_people : ℕ := 200

def total_adults : ℕ := male_adults + female_adults

theorem number_of_children : total_people - total_adults = 80 :=
by sorry

end number_of_children_l45_4562


namespace proof_quadratic_conclusions_l45_4502

noncomputable def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Given points on the graph
def points_on_graph (a b c : ℝ) : Prop :=
  quadratic_function a b c (-1) = -2 ∧
  quadratic_function a b c 0 = -3 ∧
  quadratic_function a b c 1 = -4 ∧
  quadratic_function a b c 2 = -3 ∧
  quadratic_function a b c 3 = 0

-- Assertions based on the problem statement
def assertion_A (a b : ℝ) : Prop := 2 * a + b = 0

def assertion_C (a b c : ℝ) : Prop :=
  quadratic_function a b c 3 = 0 ∧ quadratic_function a b c (-1) = 0

def assertion_D (a b c : ℝ) (m : ℝ) (y1 y2 : ℝ) : Prop :=
  (quadratic_function a b c (m - 1) = y1) → 
  (quadratic_function a b c m = y2) → 
  (y1 < y2) → 
  (m > 3 / 2)

-- Final theorem statement to be proven
theorem proof_quadratic_conclusions (a b c : ℝ) (m y1 y2 : ℝ) :
  points_on_graph a b c →
  assertion_A a b →
  assertion_C a b c →
  assertion_D a b c m y1 y2 :=
by
  sorry

end proof_quadratic_conclusions_l45_4502


namespace largest_inscribed_rightangled_parallelogram_l45_4553

theorem largest_inscribed_rightangled_parallelogram (r : ℝ) (x y : ℝ) 
  (parallelogram_inscribed : x = 2 * r * Real.sin (45 * π / 180) ∧ y = 2 * r * Real.cos (45 * π / 180)) :
  x = r * Real.sqrt 2 ∧ y = r * Real.sqrt 2 := 
by 
  sorry

end largest_inscribed_rightangled_parallelogram_l45_4553


namespace mr_green_expected_produce_l45_4566

noncomputable def total_produce_yield (steps_length : ℕ) (steps_width : ℕ) (step_length : ℝ)
                                      (yield_carrots : ℝ) (yield_potatoes : ℝ): ℝ :=
  let length_feet := steps_length * step_length
  let width_feet := steps_width * step_length
  let area := length_feet * width_feet
  let yield_carrots_total := area * yield_carrots
  let yield_potatoes_total := area * yield_potatoes
  yield_carrots_total + yield_potatoes_total

theorem mr_green_expected_produce:
  total_produce_yield 18 25 3 0.4 0.5 = 3645 := by
  sorry

end mr_green_expected_produce_l45_4566


namespace computer_operations_in_three_hours_l45_4538

theorem computer_operations_in_three_hours :
  let additions_per_second := 12000
  let multiplications_per_second := 2 * additions_per_second
  let seconds_in_three_hours := 3 * 3600
  (additions_per_second + multiplications_per_second) * seconds_in_three_hours = 388800000 :=
by
  sorry

end computer_operations_in_three_hours_l45_4538


namespace factor_difference_of_squares_l45_4554

theorem factor_difference_of_squares (x : ℝ) : 49 - 16 * x^2 = (7 - 4 * x) * (7 + 4 * x) :=
by
  sorry

end factor_difference_of_squares_l45_4554


namespace domain_transformation_l45_4594

variable {α : Type*}
variable {f : α → α}
variable {x y : α}
variable (h₁ : ∀ x, -1 < x ∧ x < 1)

theorem domain_transformation (h₁ : ∀ x, -1 < x ∧ x < 1) : ∀ x, 0 < x ∧ x < 1 →
  ((-1 < (2 * x - 1) ∧ (2 * x - 1) < 1)) :=
by
  intro x
  intro h
  have h₂ : -1 < 2 * x - 1 := sorry
  have h₃ : 2 * x - 1 < 1 := sorry
  exact ⟨h₂, h₃⟩

end domain_transformation_l45_4594


namespace solution1_solution2_l45_4531

namespace MathProofProblem

-- Define the first system of equations
def system1 (x y : ℝ) : Prop :=
  4 * x - 2 * y = 14 ∧ 3 * x + 2 * y = 7

-- Prove the solution for the first system
theorem solution1 : ∃ (x y : ℝ), system1 x y ∧ x = 3 ∧ y = -1 := by
  sorry

-- Define the second system of equations
def system2 (x y : ℝ) : Prop :=
  y = x + 1 ∧ 2 * x + y = 10

-- Prove the solution for the second system
theorem solution2 : ∃ (x y : ℝ), system2 x y ∧ x = 3 ∧ y = 4 := by
  sorry

end MathProofProblem

end solution1_solution2_l45_4531


namespace ceil_sqrt_of_900_l45_4535

def isPerfectSquare (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

theorem ceil_sqrt_of_900 :
  isPerfectSquare 36 ∧ isPerfectSquare 25 ∧ (36 * 25 = 900) → 
  Int.ceil (Real.sqrt 900) = 30 :=
by
  intro h
  sorry

end ceil_sqrt_of_900_l45_4535


namespace smallest_x_multiple_of_53_l45_4540

theorem smallest_x_multiple_of_53 : ∃ (x : Nat), (x > 0) ∧ ( ∀ (n : Nat), (n > 0) ∧ ((3 * n + 43) % 53 = 0) → x ≤ n ) ∧ ((3 * x + 43) % 53 = 0) :=
sorry

end smallest_x_multiple_of_53_l45_4540


namespace Seokjin_total_fish_l45_4598

-- Define the conditions
def fish_yesterday := 10
def cost_yesterday := 3000
def additional_cost := 6000
def price_per_fish := cost_yesterday / fish_yesterday
def total_cost_today := cost_yesterday + additional_cost
def fish_today := total_cost_today / price_per_fish

-- Define the goal
theorem Seokjin_total_fish (h1 : fish_yesterday = 10)
                           (h2 : cost_yesterday = 3000)
                           (h3 : additional_cost = 6000)
                           (h4 : price_per_fish = cost_yesterday / fish_yesterday)
                           (h5 : total_cost_today = cost_yesterday + additional_cost)
                           (h6 : fish_today = total_cost_today / price_per_fish) :
  fish_yesterday + fish_today = 40 :=
by
  sorry

end Seokjin_total_fish_l45_4598


namespace solve_inequality_l45_4509

theorem solve_inequality (x : ℝ) : x^2 - 3 * x - 10 < 0 ↔ -2 < x ∧ x < 5 := 
by
  sorry

end solve_inequality_l45_4509


namespace lemma2_l45_4516

noncomputable def f (x a b : ℝ) := |x + a| - |x - b|

lemma lemma1 {x : ℝ} : f x 1 2 > 2 ↔ x > 3 / 2 := 
sorry

theorem lemma2 {a b : ℝ} (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : ∀ x : ℝ, f x a b ≤ 3):
  1 / a + 2 / b = (1 / 3) * (3 + 2 * Real.sqrt 2) := 
sorry

end lemma2_l45_4516


namespace iterate_fixed_point_l45_4519

theorem iterate_fixed_point {f : ℤ → ℤ} (a : ℤ) :
  (∀ n, f^[n] a = a → f a = a) ∧ (f a = a → f^[22000] a = a) :=
sorry

end iterate_fixed_point_l45_4519


namespace population_increase_difference_l45_4513

noncomputable def births_per_day : ℝ := 24 / 6
noncomputable def deaths_per_day : ℝ := 24 / 16
noncomputable def net_increase_per_day : ℝ := births_per_day - deaths_per_day
noncomputable def annual_increase_regular_year : ℝ := net_increase_per_day * 365
noncomputable def annual_increase_leap_year : ℝ := net_increase_per_day * 366

theorem population_increase_difference :
  annual_increase_leap_year - annual_increase_regular_year = 2.5 :=
by {
  sorry
}

end population_increase_difference_l45_4513


namespace arithmetic_sequence_sum_l45_4517

theorem arithmetic_sequence_sum :
  ∀ (a : ℕ → ℤ) (d : ℤ),
    a 1 = 1 →
    d ≠ 0 →
    (a 2 = a 1 + d) →
    (a 3 = a 1 + 2 * d) →
    (a 6 = a 1 + 5 * d) →
    (a 3)^2 = (a 2) * (a 6) →
    (1 + 2 * d)^2 = (1 + d) * (1 + 5 * d) →
    (6 / 2) * (2 * a 1 + (6 - 1) * d) = -24 := 
by intros a d h1 h2 h3 h4 h5 h6 h7
   sorry

end arithmetic_sequence_sum_l45_4517


namespace required_additional_amount_l45_4567

noncomputable def ryan_order_total : ℝ := 15.80 + 8.20 + 10.50 + 6.25 + 9.15
def minimum_free_delivery : ℝ := 50
def discount_threshold : ℝ := 30
def discount_rate : ℝ := 0.10

theorem required_additional_amount : 
  ∃ X : ℝ, ryan_order_total + X - discount_rate * (ryan_order_total + X) = minimum_free_delivery :=
sorry

end required_additional_amount_l45_4567


namespace cannot_tile_with_sphinxes_l45_4552

def triangle_side_length : ℕ := 6
def small_triangles_count : ℕ := 36
def upward_triangles_count : ℕ := 21
def downward_triangles_count : ℕ := 15

theorem cannot_tile_with_sphinxes (n : ℕ) (small_triangles : ℕ) (upward : ℕ) (downward : ℕ) :
  n = triangle_side_length →
  small_triangles = small_triangles_count →
  upward = upward_triangles_count →
  downward = downward_triangles_count →
  (upward % 2 ≠ 0) ∨ (downward % 2 ≠ 0) →
  ¬ (upward + downward = small_triangles ∧
     ∀ k, (k * 6) ≤ small_triangles →
     ∃ u d, u + d = k * 6 ∧ u % 2 = 0 ∧ d % 2 = 0) := 
by
  intros
  sorry

end cannot_tile_with_sphinxes_l45_4552


namespace cameron_total_questions_l45_4581

def usual_questions : Nat := 2

def group_a_questions : Nat := 
  let q1 := 2 * 1 -- 2 people who asked a single question each
  let q2 := 3 * usual_questions -- 3 people who asked two questions as usual
  let q3 := 1 * 5 -- 1 person who asked 5 questions
  q1 + q2 + q3

def group_b_questions : Nat :=
  let q1 := 1 * 0 -- 1 person asked no questions
  let q2 := 6 * 3 -- 6 people asked 3 questions each
  let q3 := 4 * usual_questions -- 4 people asked the usual number of questions
  q1 + q2 + q3

def group_c_questions : Nat :=
  let q1 := 1 * (usual_questions * 3) -- 1 person asked three times as many questions as usual
  let q2 := 1 * 1 -- 1 person asked only one question
  let q3 := 2 * 0 -- 2 members asked no questions
  let q4 := 4 * usual_questions -- The remaining tourists asked the usual 2 questions each
  q1 + q2 + q3 + q4

def group_d_questions : Nat :=
  let q1 := 1 * (usual_questions * 4) -- 1 individual asked four times as many questions as normal
  let q2 := 1 * 0 -- 1 person asked no questions at all
  let q3 := 3 * usual_questions -- The remaining tourists asked the usual number of questions
  q1 + q2 + q3

def group_e_questions : Nat :=
  let q1 := 3 * (usual_questions * 2) -- 3 people asked double the average number of questions
  let q2 := 2 * 0 -- 2 people asked none
  let q3 := 1 * 5 -- 1 tourist asked 5 questions
  let q4 := 3 * usual_questions -- The remaining tourists asked the usual number
  q1 + q2 + q3 + q4

def group_f_questions : Nat :=
  let q1 := 2 * 3 -- 2 individuals asked three questions each
  let q2 := 1 * 0 -- 1 person asked no questions
  let q3 := 4 * usual_questions -- The remaining tourists asked the usual number
  q1 + q2 + q3

def total_questions : Nat :=
  group_a_questions + group_b_questions + group_c_questions + group_d_questions + group_e_questions + group_f_questions

theorem cameron_total_questions : total_questions = 105 := by
  sorry

end cameron_total_questions_l45_4581


namespace y_intercept_of_line_b_l45_4544

-- Define the conditions
def line_parallel (m1 m2 : ℝ) : Prop := m1 = m2

def point_on_line (m b x y : ℝ) : Prop := y = m * x + b

-- Given conditions
variables (m b : ℝ)
variable (x₁ := 3)
variable (y₁ := -2)
axiom parallel_condition : line_parallel m (-3)
axiom point_condition : point_on_line m b x₁ y₁

-- Prove that the y-intercept b equals 7
theorem y_intercept_of_line_b : b = 7 :=
sorry

end y_intercept_of_line_b_l45_4544


namespace cube_less_than_three_times_l45_4564

theorem cube_less_than_three_times (x : ℤ) : x ^ 3 < 3 * x ↔ x = -3 ∨ x = -2 ∨ x = 1 :=
by
  sorry

end cube_less_than_three_times_l45_4564


namespace evaluate_expression_l45_4546

theorem evaluate_expression :
  3 * 307 + 4 * 307 + 2 * 307 + 307 * 307 = 97012 := by
  sorry

end evaluate_expression_l45_4546


namespace math_problem_l45_4587

theorem math_problem (p q r : ℝ) (h1 : p ≠ 0) (h2 : q ≠ 0) (h3 : r ≠ 0) (h4 : p + q + r = 0) :
    (p^2 * q^2 / ((p^2 - q * r) * (q^2 - p * r)) +
    p^2 * r^2 / ((p^2 - q * r) * (r^2 - p * q)) +
    q^2 * r^2 / ((q^2 - p * r) * (r^2 - p * q))) = 1 :=
by
  sorry

end math_problem_l45_4587


namespace x_plus_y_plus_z_equals_4_l45_4539

theorem x_plus_y_plus_z_equals_4 (x y z : ℝ) 
  (h1 : 2 * x + 3 * y + 4 * z = 10) 
  (h2 : y + 2 * z = 2) : 
  x + y + z = 4 :=
by
  sorry

end x_plus_y_plus_z_equals_4_l45_4539


namespace ratio_part_to_whole_l45_4528

variable (N : ℝ)

theorem ratio_part_to_whole :
  (1 / 1) * (1 / 3) * (2 / 5) * N = 10 →
  0.4 * N = 120 →
  (10 / ((1 / 3) * (2 / 5) * N) = 1 / 4) :=
by
  intros h1 h2
  sorry

end ratio_part_to_whole_l45_4528


namespace r4_plus_inv_r4_l45_4529

theorem r4_plus_inv_r4 (r : ℝ) (h : (r + (1 : ℝ) / r) ^ 2 = 5) : r ^ 4 + (1 : ℝ) / r ^ 4 = 7 := 
by
  -- Proof goes here
  sorry

end r4_plus_inv_r4_l45_4529


namespace sqrt_equiv_c_d_l45_4541

noncomputable def c : ℤ := 3
noncomputable def d : ℤ := 375

theorem sqrt_equiv_c_d : ∀ (x y : ℤ), x = 3^5 ∧ y = 5^3 → (∃ c d : ℤ, (c = 3 ∧ d = 375 ∧ x * y = c^4 * d))
    ∧ c + d = 378 := by sorry

end sqrt_equiv_c_d_l45_4541


namespace find_x1_l45_4595

variable (x1 x2 x3 : ℝ)

theorem find_x1 (h1 : 0 ≤ x3 ∧ x3 ≤ x2 ∧ x2 ≤ x1 ∧ x1 ≤ 0.8)
    (h2 : (1 - x1)^2 + (x1 - x2)^2 + (x2 - x3)^2 + x3^2 = 1 / 3) : 
    x1 = 3 / 4 :=
  sorry

end find_x1_l45_4595


namespace triangle_tan_inequality_l45_4559

theorem triangle_tan_inequality 
  {A B C : ℝ} 
  (h1 : π / 2 ≠ A) 
  (h2 : A ≥ B) 
  (h3 : B ≥ C) : 
  |Real.tan A| ≥ Real.tan B ∧ Real.tan B ≥ Real.tan C := 
  by
    sorry

end triangle_tan_inequality_l45_4559


namespace complex_coordinates_l45_4548

-- Define the imaginary unit
def i : ℂ := ⟨0, 1⟩

-- Define the complex number (1 + i)
def z1 : ℂ := 1 + i

-- Define the complex number i
def z2 : ℂ := i

-- The problem statement to be proven: the given complex number equals 1 - i
theorem complex_coordinates : (z1 / z2) = 1 - i :=
  sorry

end complex_coordinates_l45_4548


namespace line_A1_A2_condition_plane_A1_A2_A3_condition_plane_through_A3_A4_parallel_to_A1_A2_condition_l45_4570

section BarycentricCoordinates

variables {A1 A2 A3 A4 : Type} 

def barycentric_condition (x1 x2 x3 x4 : ℝ) : Prop :=
  x1 + x2 + x3 + x4 = 1

theorem line_A1_A2_condition (x1 x2 x3 x4 : ℝ) : 
  barycentric_condition x1 x2 x3 x4 → (x3 = 0 ∧ x4 = 0) ↔ (x1 + x2 = 1) :=
by
  sorry

theorem plane_A1_A2_A3_condition (x1 x2 x3 x4 : ℝ) :
  barycentric_condition x1 x2 x3 x4 → (x4 = 0) ↔ (x1 + x2 + x3 = 1) :=
by
  sorry

theorem plane_through_A3_A4_parallel_to_A1_A2_condition (x1 x2 x3 x4 : ℝ) :
  barycentric_condition x1 x2 x3 x4 → (x1 = -x2 ∧ x3 + x4 = 1) ↔ (x1 + x2 + x3 + x4 = 1) :=
by
  sorry

end BarycentricCoordinates

end line_A1_A2_condition_plane_A1_A2_A3_condition_plane_through_A3_A4_parallel_to_A1_A2_condition_l45_4570


namespace circle_center_eq_circle_center_is_1_3_2_l45_4585

-- Define the problem: Given the equation of the circle, prove the center is (1, 3/2)
theorem circle_center_eq (x y : ℝ) :
  16 * x^2 - 32 * x + 16 * y^2 - 48 * y + 100 = 0 ↔ (x - 1)^2 + (y - 3/2)^2 = 3 := sorry

-- Prove that the center of the circle from the given equation is (1, 3/2)
theorem circle_center_is_1_3_2 :
  ∃ x y : ℝ, (16 * x^2 - 32 * x + 16 * y^2 - 48 * y + 100 = 0) ∧ (x = 1) ∧ (y = 3 / 2) := sorry

end circle_center_eq_circle_center_is_1_3_2_l45_4585


namespace translate_graph_downward_3_units_l45_4506

def f (x : ℝ) : ℝ := 3 * x + 2
def g (x : ℝ) : ℝ := 3 * x - 1

theorem translate_graph_downward_3_units :
  ∀ x : ℝ, g x = f x - 3 :=
by
  sorry

end translate_graph_downward_3_units_l45_4506


namespace value_of_k_l45_4520

theorem value_of_k (x z k : ℝ) (h1 : 2 * x - (-1) + 3 * z = 9) 
                   (h2 : x + 2 * (-1) - z = k) 
                   (h3 : -x + (-1) + 4 * z = 6) : 
                   k = -3 :=
by
  sorry

end value_of_k_l45_4520


namespace find_fraction_l45_4599

-- Let f be a real number representing the fraction
theorem find_fraction (f : ℝ) (h : f * 12 + 5 = 11) : f = 1 / 2 := 
by
  sorry

end find_fraction_l45_4599


namespace compute_f_l45_4525

theorem compute_f (f : ℕ → ℚ) (h1 : f 1 = 1 / 3)
  (h2 : ∀ n : ℕ, n ≥ 2 → f n = (2 * (n - 1) - 1) / (2 * (n - 1) + 3) * f (n - 1)) :
  ∀ n : ℕ, n ≥ 1 → f n = 1 / ((2 * n - 1) * (2 * n + 1)) :=
by
  sorry

end compute_f_l45_4525


namespace intersection_x_coord_of_lines_l45_4504

theorem intersection_x_coord_of_lines (k b : ℝ) (h : k ≠ b) :
  ∃ x : ℝ, (kx + b = bx + k) ∧ x = 1 :=
by
  -- Proof is omitted.
  sorry

end intersection_x_coord_of_lines_l45_4504


namespace different_people_count_l45_4592

def initial_people := 9
def people_left := 6
def people_joined := 3
def total_different_people (initial_people people_left people_joined : ℕ) : ℕ :=
  initial_people + people_joined

theorem different_people_count :
  total_different_people initial_people people_left people_joined = 12 :=
by
  sorry

end different_people_count_l45_4592


namespace price_of_shirt_l45_4514

theorem price_of_shirt (T S : ℝ) 
  (h1 : T + S = 80.34) 
  (h2 : T = S - 7.43) : 
  T = 36.455 :=
by
  sorry

end price_of_shirt_l45_4514


namespace Tim_scored_30_l45_4573

-- Definitions and conditions
variables (Joe Tim Ken : ℕ)
variables (h1 : Tim = Joe + 20)
variables (h2 : Tim = Nat.div (Ken * 2) 2)
variables (h3 : Joe + Tim + Ken = 100)

-- Statement to prove
theorem Tim_scored_30 : Tim = 30 :=
by sorry

end Tim_scored_30_l45_4573


namespace alice_next_birthday_age_l45_4591

theorem alice_next_birthday_age (a b c : ℝ) 
  (h1 : a = 1.25 * b)
  (h2 : b = 0.7 * c)
  (h3 : a + b + c = 30) : a + 1 = 11 :=
by {
  sorry
}

end alice_next_birthday_age_l45_4591


namespace find_a_l45_4550

theorem find_a (a : ℝ) :
  let θ := 120
  let tan120 := -Real.sqrt 3
  (∀ x y: ℝ, 2 * x + a * y + 3 = 0) →
  a = 2 * Real.sqrt 3 / 3 :=
by
  sorry

end find_a_l45_4550


namespace sum_of_ages_l45_4579

theorem sum_of_ages (S F : ℕ) 
  (h1 : F - 18 = 3 * (S - 18)) 
  (h2 : F = 2 * S) : S + F = 108 := by 
  sorry

end sum_of_ages_l45_4579


namespace average_age_of_women_l45_4551

-- Defining the conditions
def average_age_of_men : ℝ := 40
def number_of_men : ℕ := 15
def increase_in_average : ℝ := 2.9
def ages_of_replaced_men : List ℝ := [26, 32, 41, 39]
def number_of_women : ℕ := 4

-- Stating the proof problem
theorem average_age_of_women :
  let total_age_of_men := average_age_of_men * number_of_men
  let total_age_of_replaced_men := ages_of_replaced_men.sum
  let new_average_age := average_age_of_men + increase_in_average
  let new_total_age_of_group := new_average_age * number_of_men
  let total_age_of_women := new_total_age_of_group - (total_age_of_men - total_age_of_replaced_men)
  let average_age_of_women := total_age_of_women / number_of_women
  average_age_of_women = 45.375 :=
sorry

end average_age_of_women_l45_4551


namespace inequality_proof_l45_4571

variable (a b : ℝ)

theorem inequality_proof (h1 : -1 < b) (h2 : b < 0) (h3 : a < 0) : 
  (a * b > a * b^2) ∧ (a * b^2 > a) := 
by
  sorry

end inequality_proof_l45_4571


namespace sum_of_drawn_numbers_is_26_l45_4572

theorem sum_of_drawn_numbers_is_26 :
  ∃ A B : ℕ, A > 1 ∧ A ≤ 50 ∧ B ≤ 50 ∧ A ≠ B ∧ Prime B ∧
           (150 * B + A = k^2) ∧ 1 ≤ B ∧ (B > 1 → A > 1 ∧ B = 2) ∧ A + B = 26 :=
by
  sorry

end sum_of_drawn_numbers_is_26_l45_4572


namespace solution_l45_4574

def mapping (x : ℝ) : ℝ := x^2

theorem solution (x : ℝ) : mapping x = 4 ↔ x = 2 ∨ x = -2 :=
by
  sorry

end solution_l45_4574


namespace evaluate_expression_l45_4577

variable (a b : ℤ)

-- Define the main expression
def main_expression (a b : ℤ) : ℤ :=
  (a - b)^2 + (a + 3 * b) * (a - 3 * b) - a * (a - 2 * b)

theorem evaluate_expression : main_expression (-1) 2 = -31 := by
  -- substituting the value and solving it in the proof block
  sorry

end evaluate_expression_l45_4577


namespace algebraic_expression_evaluation_l45_4501

theorem algebraic_expression_evaluation (x y : ℤ) (h1 : x = -2) (h2 : y = -4) : 2 * x^2 - y + 3 = 15 :=
by
  rw [h1, h2]
  sorry

end algebraic_expression_evaluation_l45_4501


namespace tan_product_min_value_l45_4545

theorem tan_product_min_value (α β γ : ℝ) (h1 : α > 0 ∧ α < π / 2) 
    (h2 : β > 0 ∧ β < π / 2) (h3 : γ > 0 ∧ γ < π / 2)
    (h4 : Real.cos α ^ 2 + Real.cos β ^ 2 + Real.cos γ ^ 2 = 1) : 
  (Real.tan α * Real.tan β * Real.tan γ) = 2 * Real.sqrt 2 := 
sorry

end tan_product_min_value_l45_4545


namespace barge_arrives_at_B_at_2pm_l45_4584

noncomputable def barge_arrival_time
  (constant_barge_speed : ℝ)
  (river_current_speed : ℝ)
  (distance_AB : ℝ)
  (time_depart_A : ℕ)
  (wait_time_B : ℝ)
  (time_return_A : ℝ) :
  ℝ := by
  sorry

theorem barge_arrives_at_B_at_2pm :
  ∀ (constant_barge_speed : ℝ), 
    (river_current_speed = 3) →
    (distance_AB = 60) →
    (time_depart_A = 9) →
    (wait_time_B = 2) →
    (time_return_A = 19 + 20 / 60) →
    barge_arrival_time constant_barge_speed river_current_speed distance_AB time_depart_A wait_time_B time_return_A = 14 := by
  sorry

end barge_arrives_at_B_at_2pm_l45_4584


namespace divisible_by_5_l45_4503

-- Problem statement: For which values of \( x \) is \( 2^x - 1 \) divisible by \( 5 \)?
-- Equivalent Proof Problem in Lean 4.

theorem divisible_by_5 (x : ℕ) : 
  (∃ t : ℕ, x = 6 * t + 1) ∨ (∃ t : ℕ, x = 6 * t + 4) ↔ (5 ∣ (2^x - 1)) :=
by sorry

end divisible_by_5_l45_4503


namespace find_a_l45_4515

noncomputable def f (x : ℝ) := x^2

theorem find_a (a : ℝ) (h : (1/2) * a^2 * (a/2) = 2) :
  a = 2 :=
sorry

end find_a_l45_4515


namespace largest_nonrepresentable_integer_l45_4578

theorem largest_nonrepresentable_integer :
  (∀ a b : ℕ, 8 * a + 15 * b ≠ 97) ∧ (∀ n : ℕ, n > 97 → ∃ a b : ℕ, n = 8 * a + 15 * b) :=
sorry

end largest_nonrepresentable_integer_l45_4578


namespace symmetric_line_x_axis_l45_4557

theorem symmetric_line_x_axis (x y : ℝ) : 
  let P := (x, y)
  let P' := (x, -y)
  (3 * x - 4 * y + 5 = 0) →  
  (3 * x + 4 * -y + 5 = 0) :=
by 
  sorry

end symmetric_line_x_axis_l45_4557


namespace equilateral_triangle_side_length_l45_4590

theorem equilateral_triangle_side_length (side_length_of_square : ℕ) (h : side_length_of_square = 21) :
    let total_length_of_string := 4 * side_length_of_square
    let side_length_of_triangle := total_length_of_string / 3
    side_length_of_triangle = 28 :=
by
  sorry

end equilateral_triangle_side_length_l45_4590


namespace find_m_real_find_m_imaginary_l45_4522

-- Define the real part condition
def real_part_condition (m : ℝ) : Prop :=
  m^2 - 3 * m - 4 = 0

-- Define the imaginary part condition
def imaginary_part_condition (m : ℝ) : Prop :=
  m^2 - 2 * m - 3 = 0 ∧ m^2 - 3 * m - 4 ≠ 0

-- Theorem for the first part
theorem find_m_real : ∀ (m : ℝ), (real_part_condition m) → (m = 4 ∨ m = -1) :=
by sorry

-- Theorem for the second part
theorem find_m_imaginary : ∀ (m : ℝ), (imaginary_part_condition m) → (m = 3) :=
by sorry

end find_m_real_find_m_imaginary_l45_4522


namespace num_male_rabbits_l45_4508

/-- 
There are 12 white rabbits and 9 black rabbits. 
There are 8 female rabbits. 
Prove that the number of male rabbits is 13.
-/
theorem num_male_rabbits (white_rabbits : ℕ) (black_rabbits : ℕ) (female_rabbits: ℕ) 
  (h_white : white_rabbits = 12) (h_black : black_rabbits = 9) (h_female : female_rabbits = 8) :
  (white_rabbits + black_rabbits - female_rabbits = 13) :=
by
  sorry

end num_male_rabbits_l45_4508


namespace find_m_n_sum_l45_4512

theorem find_m_n_sum (m n : ℝ) :
  ( ∀ x, -3 < x ∧ x < 6 → x^2 - m * x - 6 * n < 0 ) →
  m + n = 6 :=
by
  sorry

end find_m_n_sum_l45_4512


namespace emily_gave_away_l45_4511

variable (x : ℕ)

def emily_initial_books : ℕ := 7

def emily_books_after_giving_away (x : ℕ) : ℕ := 7 - x

def emily_books_after_buying_more (x : ℕ) : ℕ :=
  7 - x + 14

def emily_final_books : ℕ := 19

theorem emily_gave_away : (emily_books_after_buying_more x = emily_final_books) → x = 2 := by
  sorry

end emily_gave_away_l45_4511


namespace students_selected_milk_is_54_l45_4593

-- Define the parameters.
variable (total_students : ℕ)
variable (students_selected_soda students_selected_milk : ℕ)

-- Given conditions.
axiom h1 : students_selected_soda = 90
axiom h2 : students_selected_soda = (1 / 2) * total_students
axiom h3 : students_selected_milk = (3 / 5) * students_selected_soda

-- Prove that the number of students who selected milk is equal to 54.
theorem students_selected_milk_is_54 : students_selected_milk = 54 :=
by
  sorry

end students_selected_milk_is_54_l45_4593


namespace find_xyz_l45_4589

theorem find_xyz
  (a b c x y z : ℂ)
  (h1 : a ≠ 0)
  (h2 : b ≠ 0)
  (h3 : c ≠ 0)
  (h4 : x ≠ 0)
  (h5 : y ≠ 0)
  (h6 : z ≠ 0)
  (h7 : a = (b + c) / (x - 3))
  (h8 : b = (a + c) / (y - 3))
  (h9 : c = (a + b) / (z - 3))
  (h10 : x * y + x * z + y * z = 10)
  (h11 : x + y + z = 6) :
  x * y * z = 10 :=
sorry

end find_xyz_l45_4589


namespace original_triangle_area_l45_4556

theorem original_triangle_area :
  let S_perspective := (1 / 2) * 1 * 1 * Real.sin (Real.pi / 3)
  let S_ratio := Real.sqrt 2 / 4
  let S_perspective_value := Real.sqrt 3 / 4
  let S_original := S_perspective_value / S_ratio
  S_original = Real.sqrt 6 / 2 :=
by
  sorry

end original_triangle_area_l45_4556


namespace quadrant_iv_l45_4565

theorem quadrant_iv (x y : ℚ) (h1 : x = 1) (h2 : x - y = 12 / 5) (h3 : 6 * x + 5 * y = -1) :
  x = 1 ∧ y = -7 / 5 ∧ (12 / 5 > 0 ∧ -7 / 5 < 0) :=
by
  sorry

end quadrant_iv_l45_4565


namespace divisibility_by_10_l45_4576

theorem divisibility_by_10 (a : ℤ) (n : ℕ) (h : n ≥ 2) : 
  (a^(2^n + 1) - a) % 10 = 0 :=
by
  sorry

end divisibility_by_10_l45_4576


namespace probability_of_yellow_jelly_bean_l45_4549

theorem probability_of_yellow_jelly_bean (P_red P_orange P_yellow : ℝ) 
  (h1 : P_red = 0.2) 
  (h2 : P_orange = 0.5) 
  (h3 : P_red + P_orange + P_yellow = 1) : 
  P_yellow = 0.3 :=
sorry

end probability_of_yellow_jelly_bean_l45_4549


namespace age_ratio_in_two_years_l45_4597

variable (S M : ℕ)

-- Conditions
def sonCurrentAge : Prop := S = 18
def manCurrentAge : Prop := M = S + 20
def multipleCondition : Prop := ∃ k : ℕ, M + 2 = k * (S + 2)

-- Statement to prove
theorem age_ratio_in_two_years (h1 : sonCurrentAge S) (h2 : manCurrentAge S M) (h3 : multipleCondition S M) : 
  (M + 2) / (S + 2) = 2 := 
by
  sorry

end age_ratio_in_two_years_l45_4597


namespace sum_a3_a7_l45_4568

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_a3_a7 (a : ℕ → ℝ)
  (h₁ : arithmetic_sequence a)
  (h₂ : a 1 + a 9 + a 2 + a 8 = 20) :
  a 3 + a 7 = 10 :=
sorry

end sum_a3_a7_l45_4568


namespace find_x_l45_4537

theorem find_x :
  ∃ x : ℕ, x % 6 = 5 ∧ x % 7 = 6 ∧ x % 8 = 7 ∧ (∀ y : ℕ, y % 6 = 5 ∧ y % 7 = 6 ∧ y % 8 = 7 → y ≥ x) :=
sorry

end find_x_l45_4537


namespace remainder_of_6x_mod_9_l45_4533

theorem remainder_of_6x_mod_9 (x : ℕ) (h : x % 9 = 5) : (6 * x) % 9 = 3 :=
by
  sorry

end remainder_of_6x_mod_9_l45_4533


namespace airplane_time_in_air_l45_4543

-- Define conditions
def distance_seaport_island := 840  -- Total distance in km
def speed_icebreaker := 20          -- Speed of the icebreaker in km/h
def time_icebreaker := 22           -- Total time the icebreaker traveled in hours
def speed_airplane := 120           -- Speed of the airplane in km/h

-- Prove the time the airplane spent in the air
theorem airplane_time_in_air : (distance_seaport_island - speed_icebreaker * time_icebreaker) / speed_airplane = 10 / 3 := by
  -- This is where the proof steps would go, but we're placing sorry to skip it for now.
  sorry

end airplane_time_in_air_l45_4543


namespace train_time_l45_4507

theorem train_time (T : ℕ) (D : ℝ) (h1 : D = 48 * (T / 60)) (h2 : D = 60 * (40 / 60)) : T = 50 :=
by
  sorry

end train_time_l45_4507


namespace popsicle_melting_ratio_l45_4558

theorem popsicle_melting_ratio (S : ℝ) (r : ℝ) (h : r^5 = 32) : r = 2 :=
by
  sorry

end popsicle_melting_ratio_l45_4558


namespace amc_inequality_l45_4561

theorem amc_inequality (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 1) :
  (a / (b + c^2) + b / (c + a^2) + c / (a + b^2)) ≥ (9 / 4) :=
by
  sorry

end amc_inequality_l45_4561


namespace num_points_C_l45_4586

theorem num_points_C (
  A B : ℝ × ℝ)
  (C : ℝ × ℝ) 
  (hA : A = (2, 2))
  (hB : B = (-1, -2))
  (hC : (C.1 - 3)^2 + (C.2 + 5)^2 = 36)
  (h_area : 1/2 * (abs ((B.1 - A.1) * (C.2 - A.2) - (B.2 - A.2) * (C.1 - A.1))) = 5/2) :
  ∃ C1 C2 C3 : ℝ × ℝ,
    (C1.1 - 3)^2 + (C1.2 + 5)^2 = 36 ∧
    (C2.1 - 3)^2 + (C2.2 + 5)^2 = 36 ∧
    (C3.1 - 3)^2 + (C3.2 + 5)^2 = 36 ∧
    1/2 * (abs ((B.1 - A.1) * (C1.2 - A.2) - (B.2 - A.2) * (C1.1 - A.1))) = 5/2 ∧
    1/2 * (abs ((B.1 - A.1) * (C2.2 - A.2) - (B.2 - A.2) * (C2.1 - A.1))) = 5/2 ∧
    1/2 * (abs ((B.1 - A.1) * (C3.2 - A.2) - (B.2 - A.2) * (C3.1 - A.1))) = 5/2 ∧
    (C1 ≠ C2 ∧ C1 ≠ C3 ∧ C2 ≠ C3) :=
sorry

end num_points_C_l45_4586


namespace positive_number_and_cube_l45_4569

theorem positive_number_and_cube (n : ℕ) (h : n^2 + n = 210) (h_pos : n > 0) : n = 14 ∧ n^3 = 2744 :=
by sorry

end positive_number_and_cube_l45_4569


namespace find_abc_square_sum_l45_4526

theorem find_abc_square_sum (a b c : ℝ) 
  (h1 : a^2 + 3 * b = 9) 
  (h2 : b^2 + 5 * c = -8) 
  (h3 : c^2 + 7 * a = -18) : 
  a^2 + b^2 + c^2 = 20.75 := 
sorry

end find_abc_square_sum_l45_4526


namespace part1_extreme_value_part2_range_of_a_l45_4582

noncomputable def f (x : ℝ) : ℝ := x * Real.exp (x + 1)

theorem part1_extreme_value :
  ∃ x : ℝ, f x = -1 :=
  sorry

theorem part2_range_of_a :
  ∀ x > 0, ∃ a : ℝ, f x ≥ x + Real.log x + a + 1 → a ≤ 1 :=
  sorry

end part1_extreme_value_part2_range_of_a_l45_4582


namespace odd_terms_in_expansion_l45_4555

theorem odd_terms_in_expansion (p q : ℤ) (hp : p % 2 = 1) (hq : q % 2 = 1) : 
  (∃ k, k = 2) :=
sorry

end odd_terms_in_expansion_l45_4555


namespace frank_initial_boxes_l45_4560

theorem frank_initial_boxes (filled left : ℕ) (h_filled : filled = 8) (h_left : left = 5) : 
  filled + left = 13 := by
  sorry

end frank_initial_boxes_l45_4560


namespace find_function_satisfaction_l45_4575

theorem find_function_satisfaction :
  ∃ (a b : ℚ) (f : ℚ × ℚ → ℚ), (∀ (x y z : ℚ),
  f (x, y) + f (y, z) + f (z, x) = f (0, x + y + z)) ∧ 
  (∀ (x y : ℚ), f (x, y) = a * y^2 + 2 * a * x * y + b * y) := sorry

end find_function_satisfaction_l45_4575


namespace sum_of_a_b_c_d_l45_4527

theorem sum_of_a_b_c_d (a b c d : ℝ) (h1 : c + d = 12 * a) (h2 : c * d = -13 * b) (h3 : a + b = 12 * c) (h4 : a * b = -13 * d) (h_distinct : a ≠ c) : a + b + c + d = 2028 :=
  by 
  -- The proof will go here
  sorry

end sum_of_a_b_c_d_l45_4527


namespace sasha_remainder_l45_4530

theorem sasha_remainder (n a b c d : ℕ) 
  (h1 : n = 102 * a + b) 
  (h2 : n = 103 * c + d) 
  (h3 : a + d = 20)
  (hb : 0 ≤ b ∧ b ≤ 101) : 
  b = 20 :=
sorry

end sasha_remainder_l45_4530


namespace find_subtracted_value_l45_4524

-- Define the conditions
def chosen_number := 124
def result := 110

-- Lean statement to prove
theorem find_subtracted_value (x : ℕ) (y : ℕ) (h1 : x = chosen_number) (h2 : 2 * x - y = result) : y = 138 :=
by
  sorry

end find_subtracted_value_l45_4524


namespace average_percentage_decrease_l45_4563

theorem average_percentage_decrease (x : ℝ) (h : 0 < x ∧ x < 1) :
  (800 * (1 - x)^2 = 578) → x = 0.15 :=
by
  sorry

end average_percentage_decrease_l45_4563


namespace operation_B_correct_l45_4500

theorem operation_B_correct : 3 / Real.sqrt 3 = Real.sqrt 3 :=
  sorry

end operation_B_correct_l45_4500


namespace polynomial_remainder_x1012_l45_4534

theorem polynomial_remainder_x1012 (x : ℂ) : 
  (x^1012) % (x^3 - x^2 + x - 1) = 1 :=
sorry

end polynomial_remainder_x1012_l45_4534


namespace production_days_l45_4532

theorem production_days (n : ℕ) (h1 : (40 * n + 90) / (n + 1) = 45) : n = 9 :=
by
  sorry

end production_days_l45_4532


namespace cube_edge_ratio_l45_4523

theorem cube_edge_ratio (a b : ℕ) (h : a^3 = 27 * b^3) : a = 3 * b :=
sorry

end cube_edge_ratio_l45_4523


namespace flynn_tv_weeks_l45_4521

-- Define the conditions
def minutes_per_weekday := 30
def additional_hours_weekend := 2
def total_hours := 234
def minutes_per_hour := 60
def weekdays := 5

-- Define the total watching time per week in minutes
def total_weekday_minutes := minutes_per_weekday * weekdays
def total_weekday_hours := total_weekday_minutes / minutes_per_hour
def total_weekly_hours := total_weekday_hours + additional_hours_weekend

-- Create a theorem to prove the correct number of weeks
theorem flynn_tv_weeks : 
  (total_hours / total_weekly_hours) = 52 := 
by
  sorry

end flynn_tv_weeks_l45_4521


namespace bake_sale_cookies_l45_4580

theorem bake_sale_cookies (R O C : ℕ) (H1 : R = 42) (H2 : R = 6 * O) (H3 : R = 2 * C) : R + O + C = 70 := by
  sorry

end bake_sale_cookies_l45_4580


namespace correct_option_C_correct_option_D_l45_4596

-- definitions representing the conditions
def A_inequality (x : ℝ) : Prop := (2 * x + 1) / (3 - x) ≤ 0
def B_inequality (x : ℝ) : Prop := (2 * x + 1) * (3 - x) ≥ 0
def C_inequality (x : ℝ) : Prop := (2 * x + 1) / (3 - x) ≥ 0
def D_inequality (x : ℝ) : Prop := (2 * x + 1) / (x - 3) ≤ 0
def solution_set (x : ℝ) : Prop := (-1 / 2 ≤ x ∧ x < 3)

-- proving that option C is equivalent to the solution set
theorem correct_option_C : ∀ x : ℝ, C_inequality x ↔ solution_set x :=
by sorry

-- proving that option D is equivalent to the solution set
theorem correct_option_D : ∀ x : ℝ, D_inequality x ↔ solution_set x :=
by sorry

end correct_option_C_correct_option_D_l45_4596


namespace women_at_dance_event_l45_4510

theorem women_at_dance_event (men women : ℕ)
  (each_man_dances_with : ℕ)
  (each_woman_dances_with : ℕ)
  (total_men : men = 18)
  (dances_per_man : each_man_dances_with = 4)
  (dances_per_woman : each_woman_dances_with = 3)
  (total_dance_pairs : men * each_man_dances_with = 72) :
  women = 24 := 
  by {
    sorry
  }

end women_at_dance_event_l45_4510


namespace sin_120_eq_sqrt3_div_2_l45_4547

theorem sin_120_eq_sqrt3_div_2
  (h1 : 120 = 180 - 60)
  (h2 : ∀ θ, Real.sin (180 - θ) = Real.sin θ)
  (h3 : Real.sin 60 = Real.sqrt 3 / 2) :
  Real.sin 120 = Real.sqrt 3 / 2 :=
sorry

end sin_120_eq_sqrt3_div_2_l45_4547


namespace valid_sandwiches_bob_can_order_l45_4505

def total_breads := 5
def total_meats := 7
def total_cheeses := 6

def undesired_combinations_count : Nat :=
  let turkey_swiss := total_breads
  let roastbeef_rye := total_cheeses
  let roastbeef_swiss := total_breads
  turkey_swiss + roastbeef_rye + roastbeef_swiss

def total_sandwiches : Nat :=
  total_breads * total_meats * total_cheeses

def valid_sandwiches_count : Nat :=
  total_sandwiches - undesired_combinations_count

theorem valid_sandwiches_bob_can_order : valid_sandwiches_count = 194 := by
  sorry

end valid_sandwiches_bob_can_order_l45_4505


namespace quadratic_no_real_roots_l45_4588

theorem quadratic_no_real_roots (a : ℝ) :
  ¬ ∃ x : ℝ, x^2 - 2 * x - a = 0 → a < -1 :=
sorry

end quadratic_no_real_roots_l45_4588


namespace solve_dog_walking_minutes_l45_4518

-- Definitions based on the problem conditions
def cost_one_dog (x : ℕ) : ℕ := 20 + x
def cost_two_dogs : ℕ := 54
def cost_three_dogs : ℕ := 87
def total_earnings (x : ℕ) : ℕ := cost_one_dog x + cost_two_dogs + cost_three_dogs

-- Proving that the total earnings equal to 171 implies x = 10
theorem solve_dog_walking_minutes (x : ℕ) (h : total_earnings x = 171) : x = 10 :=
by
  -- The proof goes here
  sorry

end solve_dog_walking_minutes_l45_4518


namespace binom_1300_2_eq_l45_4583

theorem binom_1300_2_eq : Nat.choose 1300 2 = 844350 := by
  sorry

end binom_1300_2_eq_l45_4583


namespace curve_equation_l45_4542

theorem curve_equation :
  (∃ (x y : ℝ), 2 * x + y - 8 = 0 ∧ x - 2 * y + 1 = 0 ∧ x = 3 ∧ y = 2) ∧
  (∃ (C : ℝ), 
    8 * 3 + 6 * 2 + C = 0 ∧
    8 * x + 6 * y + C = 0 ∧
    4 * x + 3 * y - 18 = 0 ∧
    ∀ x y, 6 * x - 8 * y + 3 = 0 → 
    4 * x + 3 * y - 18 = 0) ∧
  (∃ (a : ℝ), ∀ x y, (x + 1)^2 + 1 = (x - 1)^2 + 9 →
    ((x - 2)^2 + y^2 = 10 ∧ a = 2)) :=
sorry

end curve_equation_l45_4542


namespace number_of_rocks_in_bucket_l45_4536

noncomputable def average_weight_rock : ℝ := 1.5
noncomputable def total_money_made : ℝ := 60
noncomputable def price_per_pound : ℝ := 4

theorem number_of_rocks_in_bucket : 
  let total_weight_rocks := total_money_made / price_per_pound
  let number_of_rocks := total_weight_rocks / average_weight_rock
  number_of_rocks = 10 :=
by
  sorry

end number_of_rocks_in_bucket_l45_4536
