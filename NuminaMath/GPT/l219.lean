import Mathlib

namespace annes_score_l219_219231

theorem annes_score (a b : ℕ) (h1 : a = b + 50) (h2 : (a + b) / 2 = 150) : a = 175 := 
by
  sorry

end annes_score_l219_219231


namespace sqrt_five_squared_times_seven_fourth_correct_l219_219803

noncomputable def sqrt_five_squared_times_seven_fourth : Prop :=
  sqrt (5^2 * 7^4) = 245

theorem sqrt_five_squared_times_seven_fourth_correct : sqrt_five_squared_times_seven_fourth := by
  sorry

end sqrt_five_squared_times_seven_fourth_correct_l219_219803


namespace sqrt_of_sixteen_l219_219029

theorem sqrt_of_sixteen (x : ℝ) (h : x^2 = 16) : x = 4 ∨ x = -4 := 
sorry

end sqrt_of_sixteen_l219_219029


namespace angle_coloring_min_colors_l219_219393

  theorem angle_coloring_min_colors (n : ℕ) : 
    (∃ c : ℕ, (c = 2 ↔ n % 2 = 0) ∧ (c = 3 ↔ n % 2 = 1)) :=
  by
    sorry
  
end angle_coloring_min_colors_l219_219393


namespace right_angled_triangles_count_l219_219103

theorem right_angled_triangles_count : 
  ∃ n : ℕ, n = 12 ∧ ∀ (a b c : ℕ), (a = 2016^(1/2)) → (a^2 + b^2 = c^2) →
  (∃ (n k : ℕ), (c - b) = n ∧ (c + b) = k ∧ 2 ∣ n ∧ 2 ∣ k ∧ (n * k = 2016)) :=
by {
  sorry
}

end right_angled_triangles_count_l219_219103


namespace minimum_stamps_combination_l219_219689

theorem minimum_stamps_combination (c f : ℕ) (h : 3 * c + 4 * f = 30) :
  c + f = 8 :=
sorry

end minimum_stamps_combination_l219_219689


namespace find_m_range_of_x_l219_219826

def f (m x : ℝ) : ℝ := (m^2 - 1) * x + m^2 - 3 * m + 2

theorem find_m (m : ℝ) (H_dec : m^2 - 1 < 0) (H_f1 : f m 1 = 0) : 
  m = 1 / 2 :=
sorry

theorem range_of_x (x : ℝ) :
  f (1 / 2) (x + 1) ≥ x^2 ↔ -3 / 4 ≤ x ∧ x ≤ 0 :=
sorry

end find_m_range_of_x_l219_219826


namespace find_f2_f5_sum_l219_219014

theorem find_f2_f5_sum
  (f : ℤ → ℤ)
  (a b : ℤ)
  (h1 : f 1 = 4)
  (h2 : ∀ z : ℤ, f z = 3 * z + 6)
  (h3 : ∀ x y : ℤ, f (x + y) = f x + f y + a * x * y + b) :
  f 2 + f 5 = 33 :=
sorry

end find_f2_f5_sum_l219_219014


namespace arithmetic_mean_l219_219336

variables (x y z : ℝ)

def condition1 : Prop := 1 / (x * y) = y / (z - x + 1)
def condition2 : Prop := 1 / (x * y) = 2 / (z + 1)

theorem arithmetic_mean (h1 : condition1 x y z) (h2 : condition2 x y z) : x = (z + y) / 2 :=
by
  sorry

end arithmetic_mean_l219_219336


namespace common_ratio_of_sequence_l219_219931

theorem common_ratio_of_sequence 
  (a1 a2 a3 a4 : ℤ)
  (h1 : a1 = 25)
  (h2 : a2 = -50)
  (h3 : a3 = 100)
  (h4 : a4 = -200)
  (is_geometric : ∀ (i : ℕ), a1 * (-2) ^ i = if i = 0 then a1 else if i = 1 then a2 else if i = 2 then a3 else a4) : 
  (-50 / 25 = -2) ∧ (100 / -50 = -2) ∧ (-200 / 100 = -2) :=
by 
  sorry

end common_ratio_of_sequence_l219_219931


namespace simplify_cos_difference_l219_219150

noncomputable def cos (x : ℝ) : ℝ := real.cos x

def c := cos (20 * real.pi / 180)  -- cos(20°)
def d := cos (40 * real.pi / 180)  -- cos(40°)

theorem simplify_cos_difference :
  c - d =
  -- The expression below is placeholder; real expression involves radicals and squares
  sorry :=
by
  let c := cos (20 * real.pi / 180)
  let d := cos (40 * real.pi / 180)
  have h1 : d = 2 * c^2 - 1 := sorry
  let sqrt3 : ℝ := real.sqrt 3
  have h2 : c = (1 / 2) * d + (sqrt3 / 2) * real.sqrt (1 - d^2) := sorry
  sorry

end simplify_cos_difference_l219_219150


namespace number_of_quarters_l219_219339

-- Defining constants for the problem
def value_dime : ℝ := 0.10
def value_nickel : ℝ := 0.05
def value_penny : ℝ := 0.01
def value_quarter : ℝ := 0.25

-- Given conditions
def total_dimes : ℝ := 3
def total_nickels : ℝ := 4
def total_pennies : ℝ := 200
def total_amount : ℝ := 5.00

-- Theorem stating the number of quarters found
theorem number_of_quarters :
  (total_amount - (total_dimes * value_dime + total_nickels * value_nickel + total_pennies * value_penny)) / value_quarter = 10 :=
by
  sorry

end number_of_quarters_l219_219339


namespace polynomial_divisibility_l219_219882

theorem polynomial_divisibility (P : Polynomial ℝ) (n : ℕ) (h_pos : 0 < n) :
  ∃ Q : Polynomial ℝ, (P * P + Q * Q) % (X * X + 1)^n = 0 :=
sorry

end polynomial_divisibility_l219_219882


namespace number_of_ways_to_represent_1500_l219_219731

theorem number_of_ways_to_represent_1500 :
  ∃ (count : ℕ), count = 30 ∧ ∀ (a b c : ℕ), a * b * c = 1500 :=
sorry

end number_of_ways_to_represent_1500_l219_219731


namespace fraction_of_girls_l219_219992

theorem fraction_of_girls (G T B : ℕ) (Fraction : ℚ)
  (h1 : Fraction * G = (1/3 : ℚ) * T)
  (h2 : (B : ℚ) / G = 1/2) :
  Fraction = 1/2 := by
  sorry

end fraction_of_girls_l219_219992


namespace max_value_x_plus_2y_l219_219875

variable (x y : ℝ)
variable (h1 : 4 * x + 3 * y ≤ 12)
variable (h2 : 3 * x + 6 * y ≤ 9)

theorem max_value_x_plus_2y : x + 2 * y ≤ 3 := by
  sorry

end max_value_x_plus_2y_l219_219875


namespace polynomial_root_solution_l219_219387

theorem polynomial_root_solution (a b c : ℝ) (h1 : (2:ℝ)^5 + 4*(2:ℝ)^4 + a*(2:ℝ)^2 = b*(2:ℝ) + 4*c) 
  (h2 : (-2:ℝ)^5 + 4*(-2:ℝ)^4 + a*(-2:ℝ)^2 = b*(-2:ℝ) + 4*c) :
  a = -48 ∧ b = 16 ∧ c = -32 :=
sorry

end polynomial_root_solution_l219_219387


namespace equal_cookies_per_person_l219_219698

theorem equal_cookies_per_person 
  (boxes : ℕ) (cookies_per_box : ℕ) (people : ℕ)
  (h1 : boxes = 7) (h2 : cookies_per_box = 10) (h3 : people = 5) :
  (boxes * cookies_per_box) / people = 14 :=
by sorry

end equal_cookies_per_person_l219_219698


namespace elder_person_age_l219_219202

open Nat

variable (y e : ℕ)

-- Conditions
def age_difference := e = y + 16
def age_relation := e - 6 = 3 * (y - 6)

theorem elder_person_age
  (h1 : age_difference y e)
  (h2 : age_relation y e) :
  e = 30 :=
sorry

end elder_person_age_l219_219202


namespace difference_between_waiter_and_twenty_less_l219_219690

-- Definitions for the given conditions
def total_slices : ℕ := 78
def ratio_buzz : ℕ := 5
def ratio_waiter : ℕ := 8
def total_ratio : ℕ := ratio_buzz + ratio_waiter
def slices_per_part : ℕ := total_slices / total_ratio
def buzz_share : ℕ := ratio_buzz * slices_per_part
def waiter_share : ℕ := ratio_waiter * slices_per_part
def twenty_less_waiter : ℕ := waiter_share - 20

-- The proof statement
theorem difference_between_waiter_and_twenty_less : 
  waiter_share - twenty_less_waiter = 20 :=
by sorry

end difference_between_waiter_and_twenty_less_l219_219690


namespace arrangement_count_l219_219187

def numArrangements : Nat := 15000

theorem arrangement_count (students events : ℕ) (nA nB : ℕ) 
  (A_ne_B : nA ≠ nB) 
  (all_students : students = 7) 
  (all_events : events = 5) 
  (one_event_per_student : ∀ (e : ℕ), e < events → ∃ s, s < students ∧ (∀ (s' : ℕ), s' < students → s' ≠ s → e ≠ s')) :
  numArrangements = 15000 := 
sorry

end arrangement_count_l219_219187


namespace customs_days_l219_219663

-- Definitions from the problem conditions
def navigation_days : ℕ := 21
def transport_days : ℕ := 7
def total_days : ℕ := 30

-- Proposition we need to prove
theorem customs_days (expected_days: ℕ) (ship_departure_days : ℕ) : expected_days = 2 → ship_departure_days = 30 → (navigation_days + expected_days + transport_days = total_days) → expected_days = 2 :=
by
  intros h_expected h_departure h_eq
  sorry

end customs_days_l219_219663


namespace additional_people_needed_l219_219665

def total_days := 50
def initial_people := 40
def days_passed := 25
def work_completed := 0.40

theorem additional_people_needed : 
  ∃ additional_people : ℕ, additional_people = 8 :=
by
  -- Placeholder for the actual proof skipped with 'sorry'
  sorry

end additional_people_needed_l219_219665


namespace decorations_count_l219_219946

-- Define the conditions as Lean definitions
def plastic_skulls := 12
def broomsticks := 4
def spiderwebs := 12
def pumpkins := 2 * spiderwebs
def large_cauldron := 1
def budget_more_decorations := 20
def left_to_put_up := 10

-- Define the total decorations
def decorations_already_up := plastic_skulls + broomsticks + spiderwebs + pumpkins + large_cauldron
def additional_decorations := budget_more_decorations + left_to_put_up
def total_decorations := decorations_already_up + additional_decorations

-- Prove the total number of decorations will be 83
theorem decorations_count : total_decorations = 83 := by 
  sorry

end decorations_count_l219_219946


namespace fractions_arith_l219_219372

theorem fractions_arith : (3 / 50) + (2 / 25) - (5 / 1000) = 0.135 := by
  sorry

end fractions_arith_l219_219372


namespace initial_percentage_l219_219056

variable (P : ℝ)

theorem initial_percentage (P : ℝ) 
  (h1 : 0 ≤ P ∧ P ≤ 100)
  (h2 : (7600 * (1 - P / 100) * 0.75) = 5130) :
  P = 10 :=
by
  sorry

end initial_percentage_l219_219056


namespace problem_a_problem_b_l219_219450

-- Define given points and lines
variables (A B P Q R L T K S : Type) 
variables (l : A) -- line through A
variables (a : A) -- line through A perpendicular to l
variables (b : B) -- line through B perpendicular to l
variables (PQ_intersects_a : Q) (PR_intersects_b : R)
variables (line_through_A_perp_BQ : L) (line_through_B_perp_AR : K)
variables (intersects_BQ_at_L : L) (intersects_BR_at_T : T)
variables (intersects_AR_at_K : K) (intersects_AQ_at_S : S)

-- Define collinearity properties
def collinear (X Y Z : Type) : Prop := sorry

-- Formalize the mathematical proofs as Lean theorems
theorem problem_a : collinear P T S :=
sorry

theorem problem_b : collinear P K L :=
sorry

end problem_a_problem_b_l219_219450


namespace simplify_expression_l219_219454

-- Defining the original expression
def original_expr (y : ℝ) : ℝ := 3 * y^3 - 7 * y^2 + 12 * y + 5 - (2 * y^3 - 4 + 3 * y^2 - 9 * y)

-- Defining the simplified expression
def simplified_expr (y : ℝ) : ℝ := y^3 - 10 * y^2 + 21 * y + 9

-- The statement to prove
theorem simplify_expression (y : ℝ) : original_expr y = simplified_expr y :=
by sorry

end simplify_expression_l219_219454


namespace sum_fourth_power_l219_219844

  theorem sum_fourth_power (x y z : ℝ) 
    (h1 : x + y + z = 2) 
    (h2 : x^2 + y^2 + z^2 = 6) 
    (h3 : x^3 + y^3 + z^3 = 8) : 
    x^4 + y^4 + z^4 = 26 := 
  by 
    sorry
  
end sum_fourth_power_l219_219844


namespace ellipse_through_points_parabola_equation_l219_219779

-- Ellipse Problem: Prove the standard equation
theorem ellipse_through_points (m n : ℝ) (m_pos : m > 0) (n_pos : n > 0) (m_ne_n : m ≠ n) :
  (m * 0^2 + n * (5/3)^2 = 1) ∧ (m * 1^2 + n * 1^2 = 1) →
  (m = 16 / 25 ∧ n = 9 / 25) → (m * x^2 + n * y^2 = 1) ↔ (16 * x^2 + 9 * y^2 = 225) :=
sorry

-- Parabola Problem: Prove the equation
theorem parabola_equation (p x y : ℝ) (p_pos : p > 0)
  (dist_focus : abs (x + p / 2) = 10) (dist_axis : y^2 = 36) :
  (p = 2 ∨ p = 18) →
  (y^2 = 2 * p * x) ↔ (y^2 = 4 * x ∨ y^2 = 36 * x) :=
sorry

end ellipse_through_points_parabola_equation_l219_219779


namespace total_distance_is_correct_l219_219784

noncomputable def magic_ball_total_distance : ℕ := sorry

theorem total_distance_is_correct : magic_ball_total_distance = 80 := sorry

end total_distance_is_correct_l219_219784


namespace evaluate_expression_l219_219106

variable (x y z : ℤ)

theorem evaluate_expression :
  x = 3 → y = 2 → z = 4 → 3 * x - 4 * y + 5 * z = 21 :=
by
  intros hx hy hz
  rw [hx, hy, hz]
  sorry

end evaluate_expression_l219_219106


namespace probability_A_not_losing_l219_219508

theorem probability_A_not_losing (P_draw : ℚ) (P_win_A : ℚ) (h1 : P_draw = 1/2) (h2 : P_win_A = 1/3) : 
  P_draw + P_win_A = 5/6 :=
by
  rw [h1, h2]
  norm_num

end probability_A_not_losing_l219_219508


namespace distinct_remainders_l219_219793

open Finset

noncomputable def B (A : Fin 100 → Fin 101) (k : Fin 100) := (range k.succ).sum (λ i, A i)

theorem distinct_remainders (A : Fin 100 → Fin 101) (hf : ∀ i, A i ∈ (range 100).map (coe : Fin 100 → Fin 101)) (hperm : ∀ i, A.to_fun i ∈ (range 100).image coe) :
  (range 100).image (λ n, (B A n).val % 100).card ≥ 11 :=
by sorry

end distinct_remainders_l219_219793


namespace cos_sin_eq_l219_219398

theorem cos_sin_eq (x : ℝ) (h : Real.cos x - 3 * Real.sin x = 2) :
  (Real.sin x + 3 * Real.cos x = (2 * Real.sqrt 6 - 3) / 5) ∨
  (Real.sin x + 3 * Real.cos x = -(2 * Real.sqrt 6 + 3) / 5) := 
by
  sorry

end cos_sin_eq_l219_219398


namespace find_number_l219_219813

theorem find_number
  (a b c : ℕ)
  (h1 : 0 ≤ a ∧ a ≤ 9)
  (h2 : 0 ≤ b ∧ b ≤ 9)
  (h3 : 0 ≤ c ∧ c ≤ 9)
  (h4 : 328 - (100 * a + 10 * b + c) = a + b + c) :
  100 * a + 10 * b + c = 317 :=
sorry

end find_number_l219_219813


namespace min_rectangle_perimeter_l219_219223

theorem min_rectangle_perimeter (x y : ℤ) (h1 : x * y = 50) (hx : 0 < x) (hy : 0 < y) : 
  (∀ x y, x * y = 50 → 2 * (x + y) ≥ 30) ∧ 
  ∃ x y, x * y = 50 ∧ 2 * (x + y) = 30 := 
by sorry

end min_rectangle_perimeter_l219_219223


namespace min_value_of_sum_of_squares_l219_219968

theorem min_value_of_sum_of_squares (x y z : ℝ) (h : x + 2 * y + z = 1) : 
    x^2 + y^2 + z^2 ≥ (1 / 6) := 
  sorry

noncomputable def min_val_xy2z (x y z : ℝ) (h : x + 2 * y + z = 1) : ℝ :=
  if h_sq : x^2 + y^2 + z^2 = 1 / 6 then (x^2 + y^2 + z^2) else if x = 1 / 6 ∧ z = 1 / 6 ∧ y = 1 / 3 then 1 / 6 else (1 / 6)

example (x y z : ℝ) (h : x + 2 * y + z = 1) : x^2 + y^2 + z^2 = min_val_xy2z x y z h :=
  sorry

end min_value_of_sum_of_squares_l219_219968


namespace infinitely_many_solutions_l219_219547

def circ (x y : ℝ) : ℝ := 4 * x - 3 * y + x * y

theorem infinitely_many_solutions : ∀ y : ℝ, circ 3 y = 12 := by
  sorry

end infinitely_many_solutions_l219_219547


namespace original_selling_price_l219_219515

theorem original_selling_price:
  ∀ (P : ℝ), (1.17 * P - 1.10 * P = 56) → (P > 0) → 1.10 * P = 880 :=
by
  intro P h₁ h₂
  sorry

end original_selling_price_l219_219515


namespace cost_price_correct_l219_219916

noncomputable def cost_price_per_meter (selling_price_per_meter : ℝ) (total_meters : ℝ) (loss_per_meter : ℝ) :=
  (selling_price_per_meter * total_meters + loss_per_meter * total_meters) / total_meters

theorem cost_price_correct :
  cost_price_per_meter 18000 500 5 = 41 :=
by 
  sorry

end cost_price_correct_l219_219916


namespace approximate_number_of_fish_in_pond_l219_219777

-- Define the conditions as hypotheses.
def tagged_fish_caught_first : ℕ := 50
def total_fish_caught_second : ℕ := 50
def tagged_fish_found_second : ℕ := 5

-- Define total fish in the pond.
def total_fish_in_pond (N : ℝ) : Prop :=
  tagged_fish_found_second / total_fish_caught_second = tagged_fish_caught_first / N

-- The statement to be proved.
theorem approximate_number_of_fish_in_pond (N : ℝ) (h : total_fish_in_pond N) : N = 500 :=
sorry

end approximate_number_of_fish_in_pond_l219_219777


namespace cos_double_angle_identity_l219_219571

open Real

theorem cos_double_angle_identity (α : ℝ) 
  (h : tan (α + π / 4) = 1 / 3) : cos (2 * α) = 3 / 5 :=
sorry

end cos_double_angle_identity_l219_219571


namespace fundraising_part1_fundraising_part2_l219_219856

-- Problem 1
theorem fundraising_part1 (x y : ℕ) 
(h1 : x + y = 60) 
(h2 : 100 * x + 80 * y = 5600) :
x = 40 ∧ y = 20 := 
by 
  sorry

-- Problem 2
theorem fundraising_part2 (a : ℕ) 
(h1 : 100 * a + 80 * (80 - a) ≤ 6890) :
a ≤ 24 := 
by 
  sorry

end fundraising_part1_fundraising_part2_l219_219856


namespace greatest_valid_number_l219_219500

-- Define the conditions
def is_valid_number (n : ℕ) : Prop :=
  n < 200 ∧ Nat.gcd n 30 = 5

-- Formulate the proof problem
theorem greatest_valid_number : ∃ n, is_valid_number n ∧ (∀ m, is_valid_number m → m ≤ n) ∧ n = 185 := 
by
  sorry

end greatest_valid_number_l219_219500


namespace distance_between_x_intercepts_l219_219219

theorem distance_between_x_intercepts 
  (m₁ m₂ : ℝ) (p : ℝ × ℝ) (h₁ : m₁ = 4) (h₂ : m₂ = -3) (h₃ : p = (8, 20)) : 
  let x₁ := (20 - (20 - m₁ * (8 - 8))) / m₁,
      x₂ := (20 - (20 - m₂ * (8 - 8))) / m₂ in
  |x₁ - x₂| = 35 / 3 :=
by
  sorry

end distance_between_x_intercepts_l219_219219


namespace wedding_chairs_total_l219_219817

theorem wedding_chairs_total :
  let first_section_rows := 5
  let first_section_chairs_per_row := 10
  let first_section_late_people := 15
  let first_section_extra_chairs_per_late := 2
  
  let second_section_rows := 8
  let second_section_chairs_per_row := 12
  let second_section_late_people := 25
  let second_section_extra_chairs_per_late := 3
  
  let third_section_rows := 4
  let third_section_chairs_per_row := 15
  let third_section_late_people := 8
  let third_section_extra_chairs_per_late := 1

  let fourth_section_rows := 6
  let fourth_section_chairs_per_row := 9
  let fourth_section_late_people := 12
  let fourth_section_extra_chairs_per_late := 1
  
  let total_original_chairs := 
    (first_section_rows * first_section_chairs_per_row) + 
    (second_section_rows * second_section_chairs_per_row) + 
    (third_section_rows * third_section_chairs_per_row) + 
    (fourth_section_rows * fourth_section_chairs_per_row)
  
  let total_extra_chairs :=
    (first_section_late_people * first_section_extra_chairs_per_late) + 
    (second_section_late_people * second_section_extra_chairs_per_late) + 
    (third_section_late_people * third_section_extra_chairs_per_late) + 
    (fourth_section_late_people * fourth_section_extra_chairs_per_late)
  
  total_original_chairs + total_extra_chairs = 385 :=
by
  sorry

end wedding_chairs_total_l219_219817


namespace quadratic_function_increasing_l219_219099

theorem quadratic_function_increasing (x : ℝ) : ((x - 1)^2 + 2 < (x + 1 - 1)^2 + 2) ↔ (x > 1) := by
  sorry

end quadratic_function_increasing_l219_219099


namespace problem1_problem2_l219_219519

-- Proof Problem 1
theorem problem1 (x : ℝ) : -x^2 + 4 * x + 5 < 0 ↔ x < -1 ∨ x > 5 :=
by sorry

-- Proof Problem 2
theorem problem2 (x a : ℝ) :
  if a = -1 then (x^2 + (1 - a) * x - a < 0 ↔ false) else
  if a > -1 then (x^2 + (1 - a) * x - a < 0 ↔ -1 < x ∧ x < a) else
  (x^2 + (1 - a) * x - a < 0 ↔ a < x ∧ x < -1) :=
by sorry

end problem1_problem2_l219_219519


namespace largest_four_digit_number_divisible_by_33_l219_219041

theorem largest_four_digit_number_divisible_by_33 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ (33 ∣ n) ∧ ∀ m : ℕ, (1000 ≤ m ∧ m < 10000 ∧ 33 ∣ m → m ≤ 9999) :=
by
  sorry

end largest_four_digit_number_divisible_by_33_l219_219041


namespace arithmetic_sqrt_9_l219_219458

theorem arithmetic_sqrt_9 :
  (∃ y : ℝ, y * y = 9 ∧ y ≥ 0) → (∃ y : ℝ, y = 3) :=
by
  sorry

end arithmetic_sqrt_9_l219_219458


namespace average_percentage_decrease_l219_219066

theorem average_percentage_decrease (x : ℝ) (h : 0 < x ∧ x < 1) :
  (800 * (1 - x)^2 = 578) → x = 0.15 :=
by
  sorry

end average_percentage_decrease_l219_219066


namespace parabola_distance_l219_219759

theorem parabola_distance (y : ℝ) (h : y ^ 2 = 24) : |-6 - 1| = 7 :=
by { sorry }

end parabola_distance_l219_219759


namespace hexagon_tiling_min_colors_l219_219762

theorem hexagon_tiling_min_colors :
  ∀ (s₁ s₂ : ℝ) (hex_area : ℝ) (tile_area : ℝ) (tiles_needed : ℕ) (n : ℕ),
    s₁ = 6 →
    s₂ = 0.5 →
    hex_area = (3 * Real.sqrt 3 / 2) * s₁^2 →
    tile_area = (Real.sqrt 3 / 4) * s₂^2 →
    tiles_needed = hex_area / tile_area →
    tiles_needed ≤ (Nat.choose n 3) →
    n ≥ 19 :=
by
  intros s₁ s₂ hex_area tile_area tiles_needed n
  intros s₁_eq s₂_eq hex_area_eq tile_area_eq tiles_needed_eq color_constraint
  sorry

end hexagon_tiling_min_colors_l219_219762


namespace smallest_n_for_Qn_l219_219994

theorem smallest_n_for_Qn (n : ℕ) : 
  (∃ n : ℕ, 1 / (n * (2 * n + 1)) < 1 / 2023 ∧ ∀ m < n, 1 / (m * (2 * m + 1)) ≥ 1 / 2023) ↔ n = 32 := by
sorry

end smallest_n_for_Qn_l219_219994


namespace chi_squared_confidence_level_l219_219112

theorem chi_squared_confidence_level 
  (chi_squared_value : ℝ)
  (p_value_3841 : ℝ)
  (p_value_5024 : ℝ)
  (h1 : chi_squared_value = 4.073)
  (h2 : p_value_3841 = 0.05)
  (h3 : p_value_5024 = 0.025)
  (h4 : 3.841 ≤ chi_squared_value ∧ chi_squared_value < 5.024) :
  ∃ confidence_level : ℝ, confidence_level = 0.95 :=
by 
  sorry

end chi_squared_confidence_level_l219_219112


namespace trigonometric_identity_solution_l219_219204

open Real

theorem trigonometric_identity_solution (k n l : ℤ) (x : ℝ) 
  (h : 2 * cos x ≠ sin x) : 
  (sin x ^ 3 + cos x ^ 3) / (2 * cos x - sin x) = cos (2 * x) ↔
  (∃ k : ℤ, x = (π / 2) * (2 * k + 1)) ∨
  (∃ n : ℤ, x = (π / 4) * (4 * n - 1)) ∨
  (∃ l : ℤ, x = arctan (1 / 2) + π * l) :=
sorry

end trigonometric_identity_solution_l219_219204


namespace complement_set_M_l219_219410

-- Definitions of sets based on given conditions
def universal_set : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

def set_M : Set ℝ := {x | x^2 - x ≤ 0}

-- The proof statement that we need to prove
theorem complement_set_M :
  {x | 1 < x ∧ x ≤ 2} = universal_set \ set_M := by
  sorry

end complement_set_M_l219_219410


namespace books_about_outer_space_l219_219340

variable (x : ℕ)

theorem books_about_outer_space :
  160 + 48 + 16 * x = 224 → x = 1 :=
by
  intro h
  sorry

end books_about_outer_space_l219_219340


namespace sufficient_but_not_necessary_l219_219088

theorem sufficient_but_not_necessary (a b c : ℝ) :
  (b^2 = a * c → (c ≠ 0 ∧ a ≠ 0 ∧ b * b = a * c) ∨ (b = 0)) ∧ 
  ¬ ((c ≠ 0 ∧ a ≠ 0 ∧ b * b = a * c) → b^2 = a * c) :=
by
  sorry

end sufficient_but_not_necessary_l219_219088


namespace P_T_S_collinear_P_K_L_collinear_l219_219448

-- Given conditions
variable (l : Line) (A B P Q R S T K L : Point)

-- Condition statements
axiom A_B_P_on_line_l : A ∈ l ∧ B ∈ l ∧ P ∈ l ∧ A ≠ B ∧ B ≠ P ∧ A ≠ P

axiom line_a : is_perpendicular (line_through A) l
axiom line_b : is_perpendicular (line_through B) l

axiom line_through_P : (line_through P) ≠ l
axiom Q_on_a_and_R_on_b : Q ∈ (line_through A) ∧ R ∈ (line_through B)

axiom line_perp_A_BQ : is_perpendicular (line_through A) (line_through B Q)
axiom L_on_BQ_and_T_on_BR : L ∈ (line_through B Q) ∧ T ∈ (line_through B R)

axiom line_perp_B_AR : is_perpendicular (line_through B) (line_through A R)
axiom K_on_AR_and_S_on_AQ : K ∈ (line_through A R) ∧ S ∈ (line_through A Q)

-- Prove (a): P, T, S collinear
theorem P_T_S_collinear : collinear P T S :=
sorry

-- Prove (b): P, K, L collinear
theorem P_K_L_collinear : collinear P K L :=
sorry

end P_T_S_collinear_P_K_L_collinear_l219_219448


namespace sum_of_series_l219_219694

noncomputable def seriesSum : ℝ := ∑' n : ℕ, (4 * (n + 1) + 1) / (3 ^ (n + 1))

theorem sum_of_series : seriesSum = 7 / 2 := by
  sorry

end sum_of_series_l219_219694


namespace base_salary_at_least_l219_219191

-- Definitions for the conditions.
def previous_salary : ℕ := 75000
def commission_rate : ℚ := 0.15
def sale_value : ℕ := 750
def min_sales_required : ℚ := 266.67

-- Calculate the commission per sale
def commission_per_sale : ℚ := commission_rate * sale_value

-- Calculate the total commission for the minimum sales required
def total_commission : ℚ := min_sales_required * commission_per_sale

-- The base salary S required to not lose money
theorem base_salary_at_least (S : ℚ) : S + total_commission ≥ previous_salary ↔ S ≥ 45000 := 
by
  -- Use sorry to skip the proof
  sorry

end base_salary_at_least_l219_219191


namespace negative_number_reciprocal_eq_self_l219_219722

theorem negative_number_reciprocal_eq_self (x : ℝ) (hx : x < 0) (h : 1 / x = x) : x = -1 :=
by
  sorry

end negative_number_reciprocal_eq_self_l219_219722


namespace expand_polynomial_l219_219551

theorem expand_polynomial :
  (3 * x ^ 2 - 4 * x + 3) * (-2 * x ^ 2 + 3 * x - 4) = -6 * x ^ 4 + 17 * x ^ 3 - 30 * x ^ 2 + 25 * x - 12 :=
by
  sorry

end expand_polynomial_l219_219551


namespace average_visitors_on_sundays_is_correct_l219_219360

noncomputable def average_visitors_sundays
  (num_sundays : ℕ) (num_non_sundays : ℕ) 
  (avg_non_sunday_visitors : ℕ) (avg_month_visitors : ℕ) : ℕ :=
  let total_month_days := num_sundays + num_non_sundays
  let total_visitors := avg_month_visitors * total_month_days
  let total_non_sunday_visitors := num_non_sundays * avg_non_sunday_visitors
  let total_sunday_visitors := total_visitors - total_non_sunday_visitors
  total_sunday_visitors / num_sundays

theorem average_visitors_on_sundays_is_correct :
  average_visitors_sundays 5 25 240 290 = 540 :=
by
  sorry

end average_visitors_on_sundays_is_correct_l219_219360


namespace greatest_value_x_plus_y_l219_219505

theorem greatest_value_x_plus_y (x y : ℝ) (h1 : x^2 + y^2 = 100) (h2 : x * y = 40) : x + y = 6 * Real.sqrt 5 ∨ x + y = -6 * Real.sqrt 5 :=
by
  sorry

end greatest_value_x_plus_y_l219_219505


namespace value_of_fraction_l219_219294

theorem value_of_fraction (x y : ℝ) (h : x / y = 7 / 3) : (x + y) / (x - y) = 5 / 2 :=
by
  sorry

end value_of_fraction_l219_219294


namespace normal_prob_ineq_l219_219602

noncomputable def X : ℝ → ℝ := sorry -- Define the random variable X following the normal distribution
noncomputable def normal_dist (μ σ : ℝ) : Measure (ℝ) := sorry -- Define the normal distribution measure

theorem normal_prob_ineq {X : ℝ → ℝ} {μ σ : ℝ} 
  (hX : X ~ normal_dist μ σ) 
  (hμ : μ = 100) 
  (hσ : σ > 0)
  (h_prob : ProbabilityTheory.probability (set.Ioc 80 120) = 3 / 4) :
  ProbabilityTheory.probability (set.Ioi 120) = 1 / 8 :=
sorry -- proof omitted

end normal_prob_ineq_l219_219602


namespace john_books_purchase_l219_219121

theorem john_books_purchase : 
  let john_money := 4575
  let book_price := 325
  john_money / book_price = 14 :=
by
  sorry

end john_books_purchase_l219_219121


namespace fruit_punch_total_l219_219159

section fruit_punch
variable (orange_punch : ℝ) (cherry_punch : ℝ) (apple_juice : ℝ) (total_punch : ℝ)

axiom h1 : orange_punch = 4.5
axiom h2 : cherry_punch = 2 * orange_punch
axiom h3 : apple_juice = cherry_punch - 1.5
axiom h4 : total_punch = orange_punch + cherry_punch + apple_juice

theorem fruit_punch_total : total_punch = 21 := sorry

end fruit_punch

end fruit_punch_total_l219_219159


namespace base_number_of_equation_l219_219719

theorem base_number_of_equation (n : ℕ) (h_n: n = 17)
  (h_eq: 2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = some_number^18) : some_number = 2 := by
  sorry

end base_number_of_equation_l219_219719


namespace dependence_of_Q_l219_219129

theorem dependence_of_Q (a d k : ℕ) :
    ∃ (Q : ℕ), Q = (2 * k * (2 * a + 4 * k * d - d)) 
                - (k * (2 * a + (2 * k - 1) * d)) 
                - (k / 2 * (2 * a + (k - 1) * d)) 
                → Q = k * a + 13 * k^2 * d := 
sorry

end dependence_of_Q_l219_219129


namespace acres_used_for_corn_l219_219673

theorem acres_used_for_corn (total_acres : ℕ) (beans_ratio : ℕ) (wheat_ratio : ℕ) (corn_ratio : ℕ) :
  total_acres = 1034 → beans_ratio = 5 → wheat_ratio = 2 → corn_ratio = 4 →
  let total_parts := beans_ratio + wheat_ratio + corn_ratio in
  let acres_per_part := total_acres / total_parts in
  let corn_acres := acres_per_part * corn_ratio in
  corn_acres = 376 :=
by
  intros
  let total_parts := beans_ratio + wheat_ratio + corn_ratio
  let acres_per_part := total_acres / total_parts
  let corn_acres := acres_per_part * corn_ratio
  show corn_acres = 376
  sorry

end acres_used_for_corn_l219_219673


namespace tan_double_angle_l219_219845

open Real

theorem tan_double_angle (α : ℝ) (h : (sin α + cos α) / (sin α - cos α) = 1 / 2) : tan (2 * α) = 3 / 4 := 
by 
  sorry

end tan_double_angle_l219_219845


namespace find_a1_and_d_l219_219732

-- Defining the arithmetic sequence and its properties
def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- Given conditions
def conditions (a : ℕ → ℤ) (a_1 : ℤ) (d : ℤ) : Prop :=
  (a 4 + a 5 + a 6 + a 7 = 56) ∧ (a 4 * a 7 = 187) ∧ (a 1 = a_1) ∧ is_arithmetic_sequence a d

-- Proving the solution
theorem find_a1_and_d :
  ∃ (a : ℕ → ℤ) (a_1 d : ℤ),
    conditions a a_1 d ∧ ((a_1 = 5 ∧ d = 2) ∨ (a_1 = 23 ∧ d = -2)) :=
by
  sorry

end find_a1_and_d_l219_219732


namespace rain_is_random_event_l219_219480

def is_random_event (p : ℝ) : Prop := p > 0 ∧ p < 1

theorem rain_is_random_event (p : ℝ) (h : p = 0.75) : is_random_event p :=
by
  -- Here we will provide the necessary proof eventually.
  sorry

end rain_is_random_event_l219_219480


namespace myrtle_hens_l219_219879

/-- Myrtle has some hens that lay 3 eggs a day. She was gone for 7 days and told her neighbor 
    to take as many as they would like. The neighbor took 12 eggs. Once home, Myrtle collected 
    the remaining eggs, dropping 5 on the way into her house. Myrtle has 46 eggs. Prove 
    that Myrtle has 3 hens. -/
theorem myrtle_hens (eggs_per_hen_per_day hens days neighbor_took dropped remaining_hens_eggs : ℕ) 
    (h1 : eggs_per_hen_per_day = 3) 
    (h2 : days = 7) 
    (h3 : neighbor_took = 12) 
    (h4 : dropped = 5) 
    (h5 : remaining_hens_eggs = 46) : 
    hens = 3 := 
by 
  sorry

end myrtle_hens_l219_219879


namespace julie_hours_per_week_school_year_l219_219122

-- Defining the assumptions
variable (summer_hours_per_week : ℕ) (summer_weeks : ℕ) (summer_earnings : ℝ)
variable (school_year_weeks : ℕ) (school_year_earnings : ℝ)

-- Assuming the given values
def assumptions : Prop :=
  summer_hours_per_week = 36 ∧ 
  summer_weeks = 10 ∧ 
  summer_earnings = 4500 ∧ 
  school_year_weeks = 45 ∧ 
  school_year_earnings = 4500

-- Proving that Julie must work 8 hours per week during the school year to make another $4500
theorem julie_hours_per_week_school_year : 
  assumptions summer_hours_per_week summer_weeks summer_earnings school_year_weeks school_year_earnings →
  (school_year_earnings / (summer_earnings / (summer_hours_per_week * summer_weeks)) / school_year_weeks = 8) :=
by
  sorry

end julie_hours_per_week_school_year_l219_219122


namespace land_profit_each_son_l219_219060

theorem land_profit_each_son :
  let hectares : ℝ := 3
  let m2_per_hectare : ℝ := 10000
  let total_sons : ℕ := 8
  let area_per_son := (hectares * m2_per_hectare) / total_sons
  let m2_per_portion : ℝ := 750
  let profit_per_portion : ℝ := 500
  let periods_per_year : ℕ := 12 / 3

  (area_per_son / m2_per_portion * profit_per_portion * periods_per_year = 10000) :=
by
  sorry

end land_profit_each_son_l219_219060


namespace three_f_x_expression_l219_219090

variable (f : ℝ → ℝ)
variable (h : ∀ x > 0, f (3 * x) = 3 / (3 + 2 * x))

theorem three_f_x_expression (x : ℝ) (hx : x > 0) : 3 * f x = 27 / (9 + 2 * x) :=
by sorry

end three_f_x_expression_l219_219090


namespace possible_denominators_count_l219_219888

variable (a b c : ℕ)
-- Conditions
def is_digit (n : ℕ) : Prop := 0 ≤ n ∧ n ≤ 9
def no_two_zeros (a b c : ℕ) : Prop := ¬(a = 0 ∧ b = 0) ∧ ¬(b = 0 ∧ c = 0) ∧ ¬(a = 0 ∧ c = 0)
def none_is_eight (a b c : ℕ) : Prop := a ≠ 8 ∧ b ≠ 8 ∧ c ≠ 8

-- Theorem
theorem possible_denominators_count : 
  is_digit a ∧ is_digit b ∧ is_digit c ∧ no_two_zeros a b c ∧ none_is_eight a b c →
  ∃ denoms : Finset ℕ, denoms.card = 7 ∧ ∀ d ∈ denoms, 999 % d = 0 :=
by
  sorry

end possible_denominators_count_l219_219888


namespace polynomial_pair_solution_l219_219254

-- We define the problem in terms of polynomials over real numbers
open Polynomial

theorem polynomial_pair_solution (P Q : ℝ[X]) :
  (∀ x y : ℝ, P.eval (x + Q.eval y) = Q.eval (x + P.eval y)) →
  (P = Q ∨ (∃ a b : ℝ, P = X + C a ∧ Q = X + C b)) :=
by
  intro h
  sorry

end polynomial_pair_solution_l219_219254


namespace volume_decreases_by_sixteen_point_sixty_seven_percent_l219_219015

variable {P V k : ℝ}

-- Stating the conditions
def inverse_proportionality (P V k : ℝ) : Prop :=
  P * V = k

def increased_pressure (P : ℝ) : ℝ :=
  1.2 * P

-- Theorem statement to prove the volume decrease percentage
theorem volume_decreases_by_sixteen_point_sixty_seven_percent (P V k : ℝ)
  (h1 : inverse_proportionality P V k)
  (h2 : P' = increased_pressure P) :
  V' = V / 1.2 ∧ (100 * (V - V') / V) = 16.67 :=
by
  sorry

end volume_decreases_by_sixteen_point_sixty_seven_percent_l219_219015


namespace number_of_moles_of_methanol_formed_l219_219404

def ch4_to_co2 : ℚ := 1
def o2_to_co2 : ℚ := 2
def co2_prod_from_ch4 (ch4 : ℚ) : ℚ := ch4 * ch4_to_co2 / o2_to_co2

def co2_to_ch3oh : ℚ := 1
def h2_to_ch3oh : ℚ := 3
def ch3oh_prod_from_co2 (co2 h2 : ℚ) : ℚ :=
  min (co2 / co2_to_ch3oh) (h2 / h2_to_ch3oh)

theorem number_of_moles_of_methanol_formed :
  (ch3oh_prod_from_co2 (co2_prod_from_ch4 5) 10) = 10/3 :=
by
  sorry

end number_of_moles_of_methanol_formed_l219_219404


namespace monkey_swinging_speed_l219_219785

namespace LamplighterMonkey

def running_speed : ℝ := 15
def running_time : ℝ := 5
def swinging_time : ℝ := 10
def total_distance : ℝ := 175

theorem monkey_swinging_speed : 
  (total_distance = running_speed * running_time + (running_speed / swinging_time) * swinging_time) → 
  (running_speed / swinging_time = 10) := 
by 
  intros h
  sorry

end LamplighterMonkey

end monkey_swinging_speed_l219_219785


namespace max_ab_bc_cd_da_l219_219741

theorem max_ab_bc_cd_da (a b c d : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : 0 ≤ d) (h_sum : a + b + c + d = 200) :
  ab + bc + cd + da ≤ 10000 :=
by sorry

end max_ab_bc_cd_da_l219_219741


namespace max_min_values_of_function_l219_219704

theorem max_min_values_of_function :
  ∀ (x : ℝ), -5 ≤ 4 * Real.sin x + 3 * Real.cos x ∧ 4 * Real.sin x + 3 * Real.cos x ≤ 5 :=
by
  sorry

end max_min_values_of_function_l219_219704


namespace total_distance_covered_l219_219652

variable (h : ℝ) (initial_height : ℝ := h) (bounce_ratio : ℝ := 0.8)

theorem total_distance_covered :
  initial_height + 2 * initial_height * bounce_ratio / (1 - bounce_ratio) = 13 * h :=
by 
  -- Proof omitted for now
  sorry

end total_distance_covered_l219_219652


namespace alyssa_went_to_13_games_last_year_l219_219684

theorem alyssa_went_to_13_games_last_year :
  ∀ (X : ℕ), (11 + X + 15 = 39) → X = 13 :=
by
  intros X h
  sorry

end alyssa_went_to_13_games_last_year_l219_219684


namespace white_to_brown_eggs_ratio_l219_219605

-- Define variables W and B (the initial numbers of white and brown eggs respectively)
variable (W B : ℕ)

-- Conditions: 
-- 1. All 5 brown eggs survived.
-- 2. Total number of eggs after dropping is 12.
def egg_conditions : Prop :=
  B = 5 ∧ (W + B) = 12

-- Prove the ratio of white eggs to brown eggs is 7/5 given these conditions.
theorem white_to_brown_eggs_ratio (h : egg_conditions W B) : W / B = 7 / 5 :=
by 
  sorry

end white_to_brown_eggs_ratio_l219_219605


namespace cosine_product_identity_l219_219616

open Real

theorem cosine_product_identity (α : ℝ) (n : ℕ) :
  (List.foldr (· * ·) 1 (List.map (λ k => cos (2^k * α)) (List.range (n + 1)))) =
  sin (2^(n + 1) * α) / (2^(n + 1) * sin α) :=
sorry

end cosine_product_identity_l219_219616


namespace total_time_correct_l219_219102

def greta_time : ℝ := 6.5
def george_time : ℝ := greta_time - 1.5
def gloria_time : ℝ := 2 * george_time
def gary_time : ℝ := (george_time + gloria_time) + 1.75
def gwen_time : ℝ := (greta_time + george_time) - 0.40 * (greta_time + george_time)
def total_time : ℝ := greta_time + george_time + gloria_time + gary_time + gwen_time

theorem total_time_correct : total_time = 45.15 := by
  sorry

end total_time_correct_l219_219102


namespace geom_seq_common_ratio_l219_219573

variable {a_n : ℕ → ℝ}
variable {q : ℝ}

def is_geom_seq (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

theorem geom_seq_common_ratio (h1 : a_n 0 + a_n 2 = 10)
                              (h2 : a_n 3 + a_n 5 = 5 / 4)
                              (h_geom : is_geom_seq a_n q) :
  q = 1 / 2 :=
by
  sorry

end geom_seq_common_ratio_l219_219573


namespace arithmetic_sqrt_9_l219_219457

theorem arithmetic_sqrt_9 :
  (∃ y : ℝ, y * y = 9 ∧ y ≥ 0) → (∃ y : ℝ, y = 3) :=
by
  sorry

end arithmetic_sqrt_9_l219_219457


namespace color_stamps_sold_l219_219883

theorem color_stamps_sold :
    let total_stamps : ℕ := 1102609
    let black_and_white_stamps : ℕ := 523776
    total_stamps - black_and_white_stamps = 578833 := 
by
  sorry

end color_stamps_sold_l219_219883


namespace find_fraction_l219_219650

theorem find_fraction (f : ℝ) (n : ℝ) (h : n = 180) (eqn : f * ((1 / 3) * (1 / 5) * n) + 6 = (1 / 15) * n) : f = 1 / 2 :=
by
  -- Definitions and assumptions provided above will be used here.
  sorry

end find_fraction_l219_219650


namespace total_time_iggy_runs_correct_l219_219114

noncomputable def total_time_iggy_runs : ℝ :=
  let monday_time := 3 * (10 + 1 + 0.5);
  let tuesday_time := 5 * (9 + 1 + 1);
  let wednesday_time := 7 * (12 - 2 + 2);
  let thursday_time := 10 * (8 + 2 + 4);
  let friday_time := 4 * (10 + 0.25);
  monday_time + tuesday_time + wednesday_time + thursday_time + friday_time

theorem total_time_iggy_runs_correct : total_time_iggy_runs = 354.5 := by
  sorry

end total_time_iggy_runs_correct_l219_219114


namespace find_OC_l219_219857

noncomputable section

open Real

structure Point where
  x : ℝ
  y : ℝ

def OA (A : Point) : ℝ := sqrt (A.x^2 + A.y^2)
def OB (B : Point) : ℝ := sqrt (B.x^2 + B.y^2)
def OD (D : Point) : ℝ := sqrt (D.x^2 + D.y^2)
def ratio_of_lengths (A B : Point) : ℝ := OA A / OB B

def find_D (A B : Point) : Point :=
  let ratio := ratio_of_lengths A B
  { x := (A.x + ratio * B.x) / (1 + ratio),
    y := (A.y + ratio * B.y) / (1 + ratio) }

-- Given conditions
def A : Point := ⟨0, 1⟩
def B : Point := ⟨-3, 4⟩
def C_magnitude : ℝ := 2

-- Goal to prove
theorem find_OC : Point :=
  let D := find_D A B
  let D_length := OD D
  let scale := C_magnitude / D_length
  { x := D.x * scale,
    y := D.y * scale }

example : find_OC = ⟨-sqrt 10 / 5, 3 * sqrt 10 / 5⟩ := by
  sorry

end find_OC_l219_219857


namespace people_left_on_beach_l219_219185

theorem people_left_on_beach : 
  ∀ (initial_first_row initial_second_row initial_third_row left_first_row left_second_row left_third_row : ℕ),
  initial_first_row = 24 →
  initial_second_row = 20 →
  initial_third_row = 18 →
  left_first_row = 3 →
  left_second_row = 5 →
  initial_first_row - left_first_row + (initial_second_row - left_second_row) + initial_third_row = 54 :=
by
  intros initial_first_row initial_second_row initial_third_row left_first_row left_second_row left_third_row
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  simp
  sorry

end people_left_on_beach_l219_219185


namespace shaded_fraction_in_fourth_square_l219_219354

theorem shaded_fraction_in_fourth_square : 
  ∀ (f : ℕ → ℕ), (f 1 = 1)
  ∧ (f 2 = 3)
  ∧ (f 3 = 5)
  ∧ (f 4 = f 3 + (3 - 1) + (5 - 3))
  ∧ (f 4 * 2 = 14)
  → (f 4 = 7)
  → (f 4 / 16 = 7 / 16) :=
sorry

end shaded_fraction_in_fourth_square_l219_219354


namespace find_second_number_l219_219053

theorem find_second_number (A B : ℝ) (h1 : A = 3200) (h2 : 0.10 * A = 0.20 * B + 190) : B = 650 :=
by
  sorry

end find_second_number_l219_219053


namespace center_of_circle_sum_eq_seven_l219_219891

theorem center_of_circle_sum_eq_seven 
  (h k : ℝ)
  (circle_eq : ∀ (x y : ℝ), x^2 + y^2 = 6 * x + 8 * y - 15 → (x - h)^2 + (y - k)^2 = 10) :
  h + k = 7 := 
sorry

end center_of_circle_sum_eq_seven_l219_219891


namespace simplify_expression_l219_219878

-- Define the main theorem
theorem simplify_expression 
  (a b x : ℝ) 
  (hx : x = 1 / a * Real.sqrt ((2 * a - b) / b))
  (hc1 : 0 < b / 2)
  (hc2 : b / 2 < a)
  (hc3 : a < b) : 
  (1 - a * x) / (1 + a * x) * Real.sqrt ((1 + b * x) / (1 - b * x)) = 1 :=
sorry

end simplify_expression_l219_219878


namespace difference_highest_lowest_score_l219_219728

-- Declare scores of each player
def Zach_score : ℕ := 42
def Ben_score : ℕ := 21
def Emma_score : ℕ := 35
def Leo_score : ℕ := 28

-- Calculate the highest and lowest scores
def highest_score : ℕ := max (max Zach_score Ben_score) (max Emma_score Leo_score)
def lowest_score : ℕ := min (min Zach_score Ben_score) (min Emma_score Leo_score)

-- Calculate the difference
def score_difference : ℕ := highest_score - lowest_score

theorem difference_highest_lowest_score : score_difference = 21 := 
by
  sorry

end difference_highest_lowest_score_l219_219728


namespace can_still_row_probability_l219_219530

/-- Define the probabilities for the left and right oars --/
def P_left1_work : ℚ := 3 / 5
def P_left2_work : ℚ := 2 / 5
def P_right1_work : ℚ := 4 / 5 
def P_right2_work : ℚ := 3 / 5

/-- Define the probabilities of the failures as complementary probabilities --/
def P_left1_fail : ℚ := 1 - P_left1_work
def P_left2_fail : ℚ := 1 - P_left2_work
def P_right1_fail : ℚ := 1 - P_right1_work
def P_right2_fail : ℚ := 1 - P_right2_work

/-- Define the probability of both left oars failing --/
def P_both_left_fail : ℚ := P_left1_fail * P_left2_fail

/-- Define the probability of both right oars failing --/
def P_both_right_fail : ℚ := P_right1_fail * P_right2_fail

/-- Define the probability of all four oars failing --/
def P_all_fail : ℚ := P_both_left_fail * P_both_right_fail

/-- Calculate the probability that at least one oar on each side works --/
def P_can_row : ℚ := 1 - (P_both_left_fail + P_both_right_fail - P_all_fail)

theorem can_still_row_probability :
  P_can_row = 437 / 625 :=
by {
  -- The proof is to be completed
  sorry
}

end can_still_row_probability_l219_219530


namespace number_of_sheep_l219_219067

variable (S H C : ℕ)

def ratio_constraint : Prop := 4 * H = 7 * S ∧ 5 * S = 4 * C

def horse_food_per_day (H : ℕ) : ℕ := 230 * H
def sheep_food_per_day (S : ℕ) : ℕ := 150 * S
def cow_food_per_day (C : ℕ) : ℕ := 300 * C

def total_horse_food : Prop := horse_food_per_day H = 12880
def total_sheep_food : Prop := sheep_food_per_day S = 9750
def total_cow_food : Prop := cow_food_per_day C = 15000

theorem number_of_sheep (h1 : ratio_constraint S H C)
                        (h2 : total_horse_food H)
                        (h3 : total_sheep_food S)
                        (h4 : total_cow_food C) :
  S = 98 :=
sorry

end number_of_sheep_l219_219067


namespace fundraising_part1_fundraising_part2_l219_219855

-- Problem 1
theorem fundraising_part1 (x y : ℕ) 
(h1 : x + y = 60) 
(h2 : 100 * x + 80 * y = 5600) :
x = 40 ∧ y = 20 := 
by 
  sorry

-- Problem 2
theorem fundraising_part2 (a : ℕ) 
(h1 : 100 * a + 80 * (80 - a) ≤ 6890) :
a ≤ 24 := 
by 
  sorry

end fundraising_part1_fundraising_part2_l219_219855


namespace minimize_cost_l219_219929

noncomputable def shipping_cost (x : ℝ) : ℝ := 5 * x
noncomputable def storage_cost (x : ℝ) : ℝ := 20 / x
noncomputable def total_cost (x : ℝ) : ℝ := shipping_cost x + storage_cost x

theorem minimize_cost : ∃ x : ℝ, x = 2 ∧ total_cost x = 20 :=
by
  use 2
  unfold total_cost
  unfold shipping_cost
  unfold storage_cost
  sorry

end minimize_cost_l219_219929


namespace base7_divisibility_rules_2_base7_divisibility_rules_3_l219_219482

def divisible_by_2 (d : Nat) : Prop :=
  d = 0 ∨ d = 2 ∨ d = 4

def divisible_by_3 (d : Nat) : Prop :=
  d = 0 ∨ d = 3

def last_digit_base7 (n : Nat) : Nat :=
  n % 7

theorem base7_divisibility_rules_2 (n : Nat) :
  (∃ k, n = 2 * k) ↔ divisible_by_2 (last_digit_base7 n) :=
by
  sorry

theorem base7_divisibility_rules_3 (n : Nat) :
  (∃ k, n = 3 * k) ↔ divisible_by_3 (last_digit_base7 n) :=
by
  sorry

end base7_divisibility_rules_2_base7_divisibility_rules_3_l219_219482


namespace moles_of_HCl_needed_l219_219842

theorem moles_of_HCl_needed : ∀ (moles_KOH : ℕ), moles_KOH = 2 →
  (moles_HCl : ℕ) → moles_HCl = 2 :=
by
  sorry

end moles_of_HCl_needed_l219_219842


namespace value_2x_y_l219_219824

theorem value_2x_y (x y : ℝ) (h : x^2 + y^2 - 2*x + 4*y + 5 = 0) : 2*x + y = 0 := 
by
  sorry

end value_2x_y_l219_219824


namespace total_dots_not_visible_l219_219819

def total_dots_on_dice (n : ℕ): ℕ := n * 21
def visible_dots : ℕ := 1 + 1 + 2 + 3 + 4 + 4 + 5 + 6
def total_dice : ℕ := 4

theorem total_dots_not_visible :
  total_dots_on_dice total_dice - visible_dots = 58 := by
  sorry

end total_dots_not_visible_l219_219819


namespace age_of_replaced_person_l219_219625

theorem age_of_replaced_person
    (T : ℕ) -- total age of the original group of 10 persons
    (age_person_replaced : ℕ) -- age of the person who was replaced
    (age_new_person : ℕ) -- age of the new person
    (h1 : age_new_person = 15)
    (h2 : (T / 10) - 3 = (T - age_person_replaced + age_new_person) / 10) :
    age_person_replaced = 45 :=
by
  sorry

end age_of_replaced_person_l219_219625


namespace solve_for_x_l219_219107

theorem solve_for_x (x : ℝ) (y : ℝ) (h1 : y = 2) (h2 : y = 1 / (4 * x + 2)) : x = -3/8 :=
by
  -- The proof will go here
  sorry

end solve_for_x_l219_219107


namespace birds_initial_count_l219_219887

theorem birds_initial_count (B : ℕ) (h1 : B + 21 = 35) : B = 14 :=
by
  sorry

end birds_initial_count_l219_219887


namespace who_is_who_l219_219910

-- Defining the structure and terms
structure Brother :=
  (name : String)
  (has_purple_card : Bool)

-- Conditions
def first_brother := Brother.mk "Tralalya" true
def second_brother := Brother.mk "Trulalya" false

/-- Proof that the names and cards of the brothers are as stated. -/
theorem who_is_who :
  ((first_brother.name = "Tralalya" ∧ first_brother.has_purple_card = false) ∧
   (second_brother.name = "Trulalya" ∧ second_brother.has_purple_card = true)) :=
by sorry

end who_is_who_l219_219910


namespace john_spends_on_memory_cards_l219_219434

theorem john_spends_on_memory_cards :
  (10 * (3 * 365)) / 50 * 60 = 13140 :=
by
  sorry

end john_spends_on_memory_cards_l219_219434


namespace liquid_levels_proof_l219_219645

noncomputable def liquid_levels (H : ℝ) : ℝ × ℝ :=
  let ρ_water := 1000
  let ρ_gasoline := 600
  -- x = level drop in the left vessel
  let x := (3 / 14) * H
  let h_left := 0.9 * H - x
  let h_right := H
  (h_left, h_right)

theorem liquid_levels_proof (H : ℝ) (h : ℝ) :
  H > 0 →
  h = 0.9 * H →
  liquid_levels H = (0.69 * H, H) :=
by
  intros
  sorry

end liquid_levels_proof_l219_219645


namespace find_t_l219_219101

open_locale big_operators

def vec2 := (ℝ × ℝ)

def dot_product (u v : vec2) : ℝ :=
  u.1 * v.1 + u.2 * v.2

def scalar_mult (t : ℝ) (u : vec2) : vec2 :=
  (t * u.1, t * u.2)

def vec_add (u v : vec2) : vec2 :=
  (u.1 + v.1, u.2 + v.2)

def a : vec2 := (1, -1)
def b : vec2 := (6, -4)

theorem find_t (t : ℝ) (h : dot_product a (vec_add (scalar_mult t a) b) = 0) : 
  t = -5 :=
by sorry

end find_t_l219_219101


namespace antiderivative_correct_l219_219395

def f (x : ℝ) : ℝ := 2 * x
def F (x : ℝ) : ℝ := x^2 + 2

theorem antiderivative_correct :
  (∀ x, f x = deriv (F) x) ∧ (F 1 = 3) :=
by
  sorry

end antiderivative_correct_l219_219395


namespace values_of_x_minus_y_l219_219084

theorem values_of_x_minus_y (x y : ℤ) (h1 : |x| = 5) (h2 : |y| = 3) (h3 : y > x) : x - y = -2 ∨ x - y = -8 :=
  sorry

end values_of_x_minus_y_l219_219084


namespace prove_f2_l219_219309

def func_condition (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ x y : ℝ, f (f x + y) = f (x ^ 2 - y) + 2 * c * f x * y

theorem prove_f2 (c : ℝ) (f : ℝ → ℝ)
  (hf : func_condition f c) :
  (f 2 = 0 ∨ f 2 = 4) ∧ (2 * (if f 2 = 0 then 4 else if f 2 = 4 then 4 else 0) = 8) :=
by {
  sorry
}

end prove_f2_l219_219309


namespace solve_quadratic_eq1_solve_quadratic_eq2_solve_quadratic_eq3_solve_quadratic_eq4_l219_219620

-- Equation (1)
theorem solve_quadratic_eq1 (x : ℝ) : x^2 + 16 = 8*x ↔ x = 4 := by
  sorry

-- Equation (2)
theorem solve_quadratic_eq2 (x : ℝ) : 2*x^2 + 4*x - 3 = 0 ↔ 
  x = -1 + (Real.sqrt 10) / 2 ∨ x = -1 - (Real.sqrt 10) / 2 := by
  sorry

-- Equation (3)
theorem solve_quadratic_eq3 (x : ℝ) : x*(x - 1) = x ↔ x = 0 ∨ x = 2 := by
  sorry

-- Equation (4)
theorem solve_quadratic_eq4 (x : ℝ) : x*(x + 4) = 8*x - 3 ↔ x = 3 ∨ x = 1 := by
  sorry

end solve_quadratic_eq1_solve_quadratic_eq2_solve_quadratic_eq3_solve_quadratic_eq4_l219_219620


namespace count_mappings_A_to_B_l219_219100

noncomputable def number_of_mappings : ℕ := Nat.choose 99 49

theorem count_mappings_A_to_B
  (A : Fin 100) (B : Fin 50)
  (f : A → B)
  (h1 : ∀ a1 a2 : A, a1 ≤ a2 → f a1 ≤ f a2)
  (h2 : ∀ b : B, ∃ a : A, f a = b) :
  number_of_mappings = Nat.choose 99 49 :=
by
  sorry

end count_mappings_A_to_B_l219_219100


namespace solve_equation_l219_219154

theorem solve_equation (x : ℚ) : 3 * (x - 2) = 2 - 5 * (x - 2) ↔ x = 9 / 4 := by
  sorry

end solve_equation_l219_219154


namespace manager_salary_l219_219166

theorem manager_salary
    (average_salary_employees : ℝ)
    (num_employees : ℕ)
    (increase_in_average_due_to_manager : ℝ)
    (total_salary_20_employees : ℝ)
    (new_average_salary : ℝ)
    (total_salary_with_manager : ℝ) :
  average_salary_employees = 1300 →
  num_employees = 20 →
  increase_in_average_due_to_manager = 100 →
  total_salary_20_employees = average_salary_employees * num_employees →
  new_average_salary = average_salary_employees + increase_in_average_due_to_manager →
  total_salary_with_manager = new_average_salary * (num_employees + 1) →
  total_salary_with_manager - total_salary_20_employees = 3400 :=
by 
  sorry

end manager_salary_l219_219166


namespace geom_seq_common_ratio_l219_219934

theorem geom_seq_common_ratio:
  ∃ r : ℝ, 
  r = -2 ∧ 
  (∀ n : ℕ, n = 0 → n = 3 →
  let a : ℕ → ℝ := λ n, if n = 0 then 25 else
                            if n = 1 then -50 else
                            if n = 2 then 100 else
                            if n = 3 then -200 else 0 in
  (a n = a 0 * r ^ n)) :=
by
  sorry

end geom_seq_common_ratio_l219_219934


namespace earn_2800_probability_l219_219001

def total_outcomes : ℕ := 7 ^ 4

def favorable_outcomes : ℕ :=
  (1 * 3 * 2 * 1) * 4 -- For each combination: \$1000, \$600, \$600, \$600; \$1000, \$1000, \$400, \$400; \$800, \$800, \$600, \$600; \$800, \$800, \$800, \$400

noncomputable def probability_of_earning_2800 : ℚ := favorable_outcomes / total_outcomes

theorem earn_2800_probability : probability_of_earning_2800 = 96 / 2401 := by
  sorry

end earn_2800_probability_l219_219001


namespace point_on_hyperbola_l219_219714

theorem point_on_hyperbola (x y : ℝ) (h_eqn : y = -4 / x) (h_point : x = -2 ∧ y = 2) : x * y = -4 := 
by
  intros
  sorry

end point_on_hyperbola_l219_219714


namespace platform_length_is_correct_l219_219048

noncomputable def length_of_platform (T : ℕ) (t_p t_s : ℕ) : ℕ :=
  let speed_of_train := T / t_s
  let distance_when_crossing_platform := speed_of_train * t_p
  distance_when_crossing_platform - T

theorem platform_length_is_correct :
  ∀ (T t_p t_s : ℕ),
  T = 300 → t_p = 33 → t_s = 18 →
  length_of_platform T t_p t_s = 250 :=
by
  intros T t_p t_s hT ht_p ht_s
  simp [length_of_platform, hT, ht_p, ht_s]
  sorry

end platform_length_is_correct_l219_219048


namespace elena_earnings_l219_219383

theorem elena_earnings (hourly_wage : ℝ) (hours_worked : ℝ) (h_wage : hourly_wage = 13.25) (h_hours : hours_worked = 4) : 
  hourly_wage * hours_worked = 53.00 := by
sorry

end elena_earnings_l219_219383


namespace no_such_triples_l219_219233

theorem no_such_triples : ¬ ∃ a b c : ℕ, 
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  Prime ((a-2)*(b-2)*(c-2)+12) ∧ 
  ((a-2)*(b-2)*(c-2)+12) ∣ (a^2 + b^2 + c^2 + a*b*c - 2017) := 
by sorry

end no_such_triples_l219_219233


namespace sufficient_but_not_necessary_condition_l219_219406

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (∀ x : ℝ, |x - 3/4| ≤ 1/4 → (x - a) * (x - (a + 1)) ≤ 0) ∧
  ¬(∀ x : ℝ, (x - a) * (x - (a + 1)) ≤ 0 → |x - 3/4| ≤ 1/4) ↔ (0 ≤ a ∧ a ≤ 1/2) :=
by
  sorry

end sufficient_but_not_necessary_condition_l219_219406


namespace parallelepiped_analogy_l219_219199

-- Define the possible plane figures
inductive PlaneFigure
| Triangle
| Trapezoid
| Parallelogram
| Rectangle

-- Define the concept of a parallelepiped
structure Parallelepiped : Type

-- The theorem asserting the parallelogram is the correct analogy
theorem parallelepiped_analogy : 
  ∀ (fig : PlaneFigure), 
    (fig = PlaneFigure.Parallelogram) ↔ 
    (fig = PlaneFigure.Parallelogram) :=
by sorry

end parallelepiped_analogy_l219_219199


namespace ab_squared_non_positive_l219_219982

theorem ab_squared_non_positive (a b : ℝ) (h : 7 * a + 9 * |b| = 0) : a * b^2 ≤ 0 :=
sorry

end ab_squared_non_positive_l219_219982


namespace sum_prime_numbers_l219_219710

theorem sum_prime_numbers (a b c : ℕ) (h1 : Nat.Prime a) (h2 : Nat.Prime b) (h3 : Nat.Prime c) (hEqn : a * b * c + a = 851) : 
  a + b + c = 50 :=
sorry

end sum_prime_numbers_l219_219710


namespace PQ_sum_l219_219126

-- Define the problem conditions
variable (P Q x : ℝ)
variable (h1 : (∀ x, x ≠ 3 → P / (x - 3) + Q * (x - 2) = (-4 * x^2 + 20 * x + 32) / (x - 3)))

-- Define the proof goal
theorem PQ_sum (h1 : (∀ x, x ≠ 3 → P / (x - 3) + Q * (x - 2) = (-4 * x^2 + 20 * x + 32) / (x - 3))) : P + Q = 52 :=
sorry

end PQ_sum_l219_219126


namespace total_hits_and_misses_l219_219642

theorem total_hits_and_misses (h : ℕ) (m : ℕ) (hc : m = 3 * h) (hm : m = 50) : h + m = 200 :=
by
  sorry

end total_hits_and_misses_l219_219642


namespace probability_of_matching_pair_l219_219384

theorem probability_of_matching_pair (blackSocks blueSocks : ℕ) (h_black : blackSocks = 12) (h_blue : blueSocks = 10) : 
  let totalSocks := blackSocks + blueSocks
  let totalWays := Nat.choose totalSocks 2
  let blackPairWays := Nat.choose blackSocks 2
  let bluePairWays := Nat.choose blueSocks 2
  let matchingPairWays := blackPairWays + bluePairWays
  totalWays = 231 ∧ matchingPairWays = 111 → (matchingPairWays : ℚ) / totalWays = 111 / 231 := 
by
  intros
  sorry

end probability_of_matching_pair_l219_219384


namespace each_member_score_l219_219539

def total_members : ℝ := 5.0
def members_didnt_show_up : ℝ := 2.0
def total_points_by_showed_up_members : ℝ := 6.0

theorem each_member_score
  (h1 : total_members - members_didnt_show_up = 3.0)
  (h2 : total_points_by_showed_up_members = 6.0) :
  total_points_by_showed_up_members / (total_members - members_didnt_show_up) = 2.0 :=
sorry

end each_member_score_l219_219539


namespace return_trip_time_l219_219534

variables (d p w : ℝ)
-- Condition 1: The outbound trip against the wind took 120 minutes.
axiom h1 : d = 120 * (p - w)
-- Condition 2: The return trip with the wind took 15 minutes less than it would in still air.
axiom h2 : d / (p + w) = d / p - 15

-- Translate the conclusion that needs to be proven in Lean 4
theorem return_trip_time (h1 : d = 120 * (p - w)) (h2 : d / (p + w) = d / p - 15) : (d / (p + w) = 15) ∨ (d / (p + w) = 85) :=
sorry

end return_trip_time_l219_219534


namespace cubic_polynomial_evaluation_l219_219127

theorem cubic_polynomial_evaluation (Q : ℚ → ℚ) (m : ℚ)
  (hQ0 : Q 0 = 2 * m) 
  (hQ1 : Q 1 = 5 * m) 
  (hQm1 : Q (-1) = 0) : 
  Q 2 + Q (-2) = 8 * m := 
by
  sorry

end cubic_polynomial_evaluation_l219_219127


namespace number_divided_by_005_l219_219258

theorem number_divided_by_005 (number : ℝ) (h : number / 0.05 = 1500) : number = 75 :=
sorry

end number_divided_by_005_l219_219258


namespace trainB_speed_l219_219192

variable (v : ℕ)

def trainA_speed : ℕ := 30
def time_gap : ℕ := 2
def distance_overtake : ℕ := 360

theorem trainB_speed (h :  v > trainA_speed) : v = 42 :=
by
  sorry

end trainB_speed_l219_219192


namespace upper_bound_y_l219_219989

theorem upper_bound_y 
  (U : ℤ) 
  (x y : ℤ)
  (h1 : 3 < x ∧ x < 6) 
  (h2 : 6 < y ∧ y < U) 
  (h3 : y - x = 4) : 
  U = 10 := 
sorry

end upper_bound_y_l219_219989


namespace right_triangle_area_l219_219545

theorem right_triangle_area (a_square_area b_square_area hypotenuse_square_area : ℝ)
  (ha : a_square_area = 36) (hb : b_square_area = 64) (hc : hypotenuse_square_area = 100)
  (leg1 leg2 hypotenuse : ℝ)
  (hleg1 : leg1 * leg1 = a_square_area)
  (hleg2 : leg2 * leg2 = b_square_area)
  (hhyp : hypotenuse * hypotenuse = hypotenuse_square_area) :
  (1/2) * leg1 * leg2 = 24 :=
by
  sorry

end right_triangle_area_l219_219545


namespace employee_salary_l219_219656

theorem employee_salary (x y : ℝ) (h1 : x + y = 770) (h2 : x = 1.2 * y) : y = 350 :=
by
  sorry

end employee_salary_l219_219656


namespace acres_used_for_corn_l219_219671

theorem acres_used_for_corn (total_acres : ℕ) (ratio_beans ratio_wheat ratio_corn : ℕ)
    (h_total : total_acres = 1034)
    (h_ratio : ratio_beans = 5 ∧ ratio_wheat = 2 ∧ ratio_corn = 4) : 
    ratio_corn * (total_acres / (ratio_beans + ratio_wheat + ratio_corn)) = 376 := 
by
  -- Proof goes here
  sorry

end acres_used_for_corn_l219_219671


namespace positive_difference_eq_30_l219_219906

theorem positive_difference_eq_30 : 
  let x1 := 12
      x2 := -18
  in |x1 - x2| = 30 := 
by
  sorry

end positive_difference_eq_30_l219_219906


namespace solution_set_of_inequality_l219_219092

/-- Given an even function f that is monotonically increasing on [0, ∞) with f(3) = 0,
    show that the solution set for xf(2x - 1) < 0 is (-∞, -1) ∪ (0, 2). -/
theorem solution_set_of_inequality
  (f : ℝ → ℝ)
  (h_even : ∀ x : ℝ, f x = f (-x))
  (h_mono : ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y)
  (h_value : f 3 = 0) :
  {x : ℝ | x * f (2*x - 1) < 0} = {x : ℝ | x < -1} ∪ {x : ℝ | 0 < x ∧ x < 2} :=
by sorry

end solution_set_of_inequality_l219_219092


namespace population_net_increase_one_day_l219_219778

-- Define the given rates and constants
def birth_rate := 10 -- people per 2 seconds
def death_rate := 2 -- people per 2 seconds
def seconds_per_day := 24 * 60 * 60 -- seconds

-- Define the expected net population increase per second
def population_increase_per_sec := (birth_rate / 2) - (death_rate / 2)

-- Define the expected net population increase per day
def expected_population_increase_per_day := population_increase_per_sec * seconds_per_day

theorem population_net_increase_one_day :
  expected_population_increase_per_day = 345600 := by
  -- This will skip the proof implementation.
  sorry

end population_net_increase_one_day_l219_219778


namespace balls_per_color_l219_219139

theorem balls_per_color (total_balls : ℕ) (total_colors : ℕ)
  (h1 : total_balls = 350) (h2 : total_colors = 10) : 
  total_balls / total_colors = 35 :=
by
  sorry

end balls_per_color_l219_219139


namespace Tamara_height_l219_219754

-- Define the conditions and goal as a theorem
theorem Tamara_height (K T : ℕ) (h1 : T = 3 * K - 4) (h2 : K + T = 92) : T = 68 :=
by
  sorry

end Tamara_height_l219_219754


namespace students_walk_home_fraction_l219_219796

theorem students_walk_home_fraction :
  (1 - (3 / 8 + 2 / 5 + 1 / 8 + 5 / 100)) = (1 / 20) :=
by 
  -- The detailed proof is complex and would require converting these fractions to a common denominator,
  -- performing the arithmetic operations carefully and using Lean's rational number properties. Thus,
  -- the full detailed proof can be written with further steps, but here we insert 'sorry' to focus on the statement.
  sorry

end students_walk_home_fraction_l219_219796


namespace total_birds_in_tree_l219_219658

def initial_birds := 14
def additional_birds := 21

theorem total_birds_in_tree : initial_birds + additional_birds = 35 := by
  sorry

end total_birds_in_tree_l219_219658


namespace candy_necklaces_l219_219737

theorem candy_necklaces (friends : ℕ) (candies_per_necklace : ℕ) (candies_per_block : ℕ)(blocks_needed : ℕ):
  friends = 8 →
  candies_per_necklace = 10 →
  candies_per_block = 30 →
  80 / 30 > 2.67 →
  blocks_needed = 3 :=
by
  intros
  sorry

end candy_necklaces_l219_219737


namespace machine_made_8_shirts_today_l219_219230

-- Define the conditions
def shirts_per_minute : ℕ := 2
def minutes_worked_today : ℕ := 4

-- Define the expected number of shirts made today
def shirts_made_today : ℕ := shirts_per_minute * minutes_worked_today

-- The theorem stating that the shirts made today should be 8
theorem machine_made_8_shirts_today : shirts_made_today = 8 := by
  sorry

end machine_made_8_shirts_today_l219_219230


namespace greatest_possible_NPMPP_l219_219832

theorem greatest_possible_NPMPP :
  ∃ (M N P PP : ℕ),
    0 ≤ M ∧ M ≤ 9 ∧
    M^2 % 10 = M ∧
    NPMPP = M * (1111 * M) ∧
    NPMPP = 89991 := by
  sorry

end greatest_possible_NPMPP_l219_219832


namespace probability_of_event_l219_219646

def is_uniform (a : ℝ) : Prop := 0 ≤ a ∧ a ≤ 1

theorem probability_of_event : 
  ∀ (a : ℝ), is_uniform a → ∀ (p : ℚ), (3 * a - 1 > 0) → p = 2 / 3 → 
  (∃ b, 0 ≤ b ∧ b ≤ 1 ∧ 3 * b - 1 > 0) := 
by
  intro a h_uniform p h_event h_prob
  sorry

end probability_of_event_l219_219646


namespace integer_solutions_count_l219_219177

theorem integer_solutions_count :
  ∃ (s : Finset ℤ), s.card = 6 ∧ ∀ x ∈ s, 4 < Real.sqrt (3 * x) ∧ Real.sqrt (3 * x) < 6 :=
by
  sorry

end integer_solutions_count_l219_219177


namespace value_of_business_l219_219789

-- Defining the conditions
def owns_shares : ℚ := 2/3
def sold_fraction : ℚ := 3/4 
def sold_amount : ℝ := 75000 

-- The final proof statement
theorem value_of_business : 
  (owns_shares * sold_fraction) * value = sold_amount →
  value = 150000 :=
by
  sorry

end value_of_business_l219_219789


namespace henry_friend_fireworks_l219_219577

-- Definitions of variables and conditions
variable 
  (F : ℕ) -- Number of fireworks Henry's friend bought

-- Main theorem statement
theorem henry_friend_fireworks (h1 : 6 + 2 + F = 11) : F = 3 :=
by
  sorry

end henry_friend_fireworks_l219_219577


namespace binomial_product_l219_219241

open Nat

theorem binomial_product :
  (Nat.choose 10 3) * (Nat.choose 8 3) = 6720 :=
by
  sorry

end binomial_product_l219_219241


namespace ab_equals_five_l219_219717

variable (a m b n : ℝ)

def arithmetic_seq (x y z : ℝ) : Prop :=
  2 * y = x + z

def geometric_seq (w x y z u : ℝ) : Prop :=
  x * x = w * y ∧ y * y = x * z ∧ z * z = y * u

theorem ab_equals_five
  (h1 : arithmetic_seq (-9) a (-1))
  (h2 : geometric_seq (-9) m b n (-1)) :
  a * b = 5 := sorry

end ab_equals_five_l219_219717


namespace solve_df1_l219_219283

variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)
variable (df1 : ℝ)

-- The condition given in the problem
axiom func_def : ∀ x, f x = 2 * x * df1 + (Real.log x)

-- Express the relationship from the derivative and solve for f'(1) = -1
theorem solve_df1 : df1 = -1 :=
by
  -- Here we will insert the proof steps in Lean, but they are omitted in this statement.
  sorry

end solve_df1_l219_219283


namespace market_value_of_10_percent_yielding_8_percent_stock_l219_219210

/-- 
Given:
1. The stock yields 8%.
2. It is a 10% stock, meaning the annual dividend per share is 10% of the face value.
3. Assume the face value of the stock is $100.

Prove:
The market value of the stock is $125.
-/
theorem market_value_of_10_percent_yielding_8_percent_stock
    (annual_dividend_per_share : ℝ)
    (face_value : ℝ)
    (dividend_yield : ℝ)
    (market_value_per_share : ℝ) 
    (h1 : face_value = 100)
    (h2 : annual_dividend_per_share = 0.10 * face_value)
    (h3 : dividend_yield = 8) :
    market_value_per_share = 125 := 
by
  /-
  Here, the following conditions are already given:
  1. face_value = 100
  2. annual_dividend_per_share = 0.10 * 100 = 10
  3. dividend_yield = 8
  
  We need to prove: market_value_per_share = 125
  -/
  sorry

end market_value_of_10_percent_yielding_8_percent_stock_l219_219210


namespace sum_of_digits_x_squared_l219_219677

theorem sum_of_digits_x_squared {r x p q : ℕ} (h_r : r ≤ 400) 
  (h_x_form : x = p * r^3 + p * r^2 + q * r + q) 
  (h_pq_condition : 7 * q = 17 * p) 
  (h_x2_form : ∃ (a b c : ℕ), x^2 = a * r^6 + b * r^5 + c * r^4 + d * r^3 + c * r^2 + b * r + a ∧ d = 0) :
  p + p + q + q = 400 := 
sorry

end sum_of_digits_x_squared_l219_219677


namespace greatest_integer_with_gcf_5_l219_219492

theorem greatest_integer_with_gcf_5 :
  ∃ n, n < 200 ∧ gcd n 30 = 5 ∧ ∀ m, m < 200 ∧ gcd m 30 = 5 → m ≤ n :=
by
  sorry

end greatest_integer_with_gcf_5_l219_219492


namespace min_next_score_to_increase_avg_l219_219311

def Liam_initial_scores : List ℕ := [72, 85, 78, 66, 90, 82]

def current_average (scores: List ℕ) : ℚ :=
  (scores.sum / scores.length : ℚ)

def next_score_requirement (initial_scores: List ℕ) (desired_increase: ℚ) : ℚ :=
  let current_avg := current_average initial_scores
  let desired_avg := current_avg + desired_increase
  let total_tests := initial_scores.length + 1
  let total_required := desired_avg * total_tests
  total_required - initial_scores.sum

theorem min_next_score_to_increase_avg :
  next_score_requirement Liam_initial_scores 5 = 115 := by
  sorry

end min_next_score_to_increase_avg_l219_219311


namespace greatest_integer_gcf_l219_219494

theorem greatest_integer_gcf (x : ℕ) : x < 200 ∧ (gcd x 30 = 5) → x = 185 :=
by sorry

end greatest_integer_gcf_l219_219494


namespace ben_is_10_l219_219070

-- Define the ages of the cousins
def ages : List ℕ := [6, 8, 10, 12, 14]

-- Define the conditions
def wentToPark (x y : ℕ) : Prop := x + y = 18
def wentToLibrary (x y : ℕ) : Prop := x + y < 20
def stayedHome (ben young : ℕ) : Prop := young = 6 ∧ ben ∈ ages ∧ ben ≠ 6 ∧ ben ≠ 12

-- The main theorem stating Ben's age
theorem ben_is_10 : ∃ ben, stayedHome ben 6 ∧ 
  (∃ x y, wentToPark x y ∧ x ∈ ages ∧ y ∈ ages ∧ x ≠ y ∧ x ≠ ben ∧ y ≠ ben) ∧
  (∃ x y, wentToLibrary x y ∧ x ∈ ages ∧ y ∈ ages ∧ x ≠ y ∧ x ≠ ben ∧ y ≠ ben) :=
by
  use 10
  -- Proof steps would go here
  sorry

end ben_is_10_l219_219070


namespace problem_statement_l219_219264

theorem problem_statement 
  (x y z : ℝ)
  (h1 : 5 = 0.25 * x)
  (h2 : 5 = 0.10 * y)
  (h3 : z = 2 * y) :
  x - z = -80 :=
sorry

end problem_statement_l219_219264


namespace inequality_proof_l219_219272

theorem inequality_proof (x y z : ℝ) (h1 : x ≥ y) (h2 : y ≥ z) (h3 : z > 0) :
  (x^2 * y / (y + z) + y^2 * z / (z + x) + z^2 * x / (x + y) ≥ 1 / 2 * (x^2 + y^2 + z^2)) :=
by sorry

end inequality_proof_l219_219272


namespace satisfy_equation_l219_219379

theorem satisfy_equation (a b c : ℤ) (h1 : a = c) (h2 : b - 1 = a) : a * (a - b) + b * (b - c) + c * (c - a) = 2 :=
by
  sorry

end satisfy_equation_l219_219379


namespace min_value_abs_function_l219_219169

theorem min_value_abs_function : ∀ x : ℝ, 4 ≤ x ∧ x ≤ 6 → (|x - 4| + |x - 6| = 2) :=
by
  sorry


end min_value_abs_function_l219_219169


namespace sin_double_angle_identity_l219_219400

variable (α : Real)

theorem sin_double_angle_identity (h : Real.sin (α + π / 6) = 1 / 3) : 
  Real.sin (2 * α + 5 * π / 6) = 7 / 9 :=
by
  sorry

end sin_double_angle_identity_l219_219400


namespace total_cakes_served_l219_219581

def Cakes_Monday_Lunch : ℕ := 5
def Cakes_Monday_Dinner : ℕ := 6
def Cakes_Sunday : ℕ := 3
def cakes_served_twice (n : ℕ) : ℕ := 2 * n
def cakes_thrown_away : ℕ := 4

theorem total_cakes_served : 
  Cakes_Sunday + Cakes_Monday_Lunch + Cakes_Monday_Dinner + 
  (cakes_served_twice (Cakes_Monday_Lunch + Cakes_Monday_Dinner) - cakes_thrown_away) = 32 := 
by 
  sorry

end total_cakes_served_l219_219581


namespace coordinates_of_B_l219_219334

structure Point where
  x : Float
  y : Float

def symmetricWithRespectToY (A B : Point) : Prop :=
  B.x = -A.x ∧ B.y = A.y

theorem coordinates_of_B (A B : Point) 
  (hA : A.x = 2 ∧ A.y = -5)
  (h_sym : symmetricWithRespectToY A B) :
  B.x = -2 ∧ B.y = -5 :=
by
  sorry

end coordinates_of_B_l219_219334


namespace center_of_circle_l219_219760

theorem center_of_circle (A B : ℝ × ℝ) (hA : A = (2, -3)) (hB : B = (10, 5)) :
    (A.1 + B.1) / 2 = 6 ∧ (A.2 + B.2) / 2 = 1 :=
by
  sorry

end center_of_circle_l219_219760


namespace initial_typists_count_l219_219423

theorem initial_typists_count 
  (typists_rate : ℕ → ℕ)
  (letters_in_20min : ℕ)
  (total_typists : ℕ)
  (letters_in_1hour : ℕ)
  (initial_typists : ℕ) 
  (h1 : letters_in_20min = 38)
  (h2 : letters_in_1hour = 171)
  (h3 : total_typists = 30)
  (h4 : ∀ t, 3 * (typists_rate t) = letters_in_1hour / total_typists)
  (h5 : ∀ t, typists_rate t = letters_in_20min / t) 
  : initial_typists = 20 := 
sorry

end initial_typists_count_l219_219423


namespace max_value_of_a_l219_219285

noncomputable def f (a b c d x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

theorem max_value_of_a (a b c d : ℝ) (h_deriv_bounds : ∀ x, 0 ≤ x → x ≤ 1 → abs (3 * a * x^2 + 2 * b * x + c) ≤ 1) (h_a_nonzero : a ≠ 0) :
  a ≤ 8 / 3 :=
sorry

end max_value_of_a_l219_219285


namespace point_Q_in_third_quadrant_l219_219422

theorem point_Q_in_third_quadrant (m : ℝ) :
  (2 * m + 4 = 0 → (m - 3, m).fst < 0 ∧ (m - 3, m).snd < 0) :=
by
  sorry

end point_Q_in_third_quadrant_l219_219422


namespace distance_x_intercepts_correct_l219_219218

noncomputable def distance_between_x_intercepts : ℝ :=
  let slope1 : ℝ := 4
  let slope2 : ℝ := -3
  let intercept_point : Prod ℝ ℝ := (8, 20)
  let line1 (x : ℝ) : ℝ := slope1 * (x - intercept_point.1) + intercept_point.2
  let line2 (x : ℝ) : ℝ := slope2 * (x - intercept_point.1) + intercept_point.2
  let x_intercept1 : ℝ := (0 - intercept_point.2) / slope1 + intercept_point.1
  let x_intercept2 : ℝ := (0 - intercept_point.2) / slope2 + intercept_point.1
  abs (x_intercept2 - x_intercept1)

theorem distance_x_intercepts_correct :
  distance_between_x_intercepts = 35 / 3 :=
sorry

end distance_x_intercepts_correct_l219_219218


namespace percentage_error_in_area_l219_219685

theorem percentage_error_in_area (s : ℝ) (x : ℝ) (h₁ : s' = 1.08 * s) 
  (h₂ : s^2 = (2 * A)) (h₃ : x^2 = (2 * A)) : 
  (abs ((1.1664 * s^2 - s^2) / s^2 * 100) - 17) ≤ 0.5 := 
sorry

end percentage_error_in_area_l219_219685


namespace sum_of_digits_l219_219859

def distinct_digits (a b c d : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

theorem sum_of_digits (a b c d : ℕ) (h_distinct : distinct_digits a b c d) (h_eqn : 100*a + 60 + b - (400 + 10*c + d) = 2) :
  a + b + c + d = 10 ∨ a + b + c + d = 18 ∨ a + b + c + d = 19 :=
sorry

end sum_of_digits_l219_219859


namespace parallelepiped_eq_l219_219588

-- Definitions of the variables and conditions
variables (a b c u v w : ℝ)

-- Prove the identity given the conditions:
theorem parallelepiped_eq :
  u * v * w = a * v * w + b * u * w + c * u * v :=
sorry

end parallelepiped_eq_l219_219588


namespace problem1_problem2_problem2_zero_problem2_neg_l219_219876

-- Definitions
def f (a x : ℝ) : ℝ := x^2 + a*x + a
def g (a x : ℝ) : ℝ := a*(f a x) - a^2*(x + 1) - 2*x

-- Problem 1
theorem problem1 (a : ℝ) :
  (∃ x1 x2 : ℝ, 0 < x1 ∧ x1 < x2 ∧ x2 < 1 ∧ f a x1 - x1 = 0 ∧ f a x2 - x2 = 0) →
  (0 < a ∧ a < 3 - 2*Real.sqrt 2) :=
sorry

-- Problem 2
theorem problem2 (a : ℝ) (h1 : a > 0) : 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → g a x ≥ 
    if a < 1 then a-2 
    else -1/a) :=
sorry

theorem problem2_zero (h2 : a = 0) : 
  g a 1 = -2 :=
sorry

theorem problem2_neg (a : ℝ) (h3 : a < 0) : 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → g a x ≥ a - 2) :=
sorry

end problem1_problem2_problem2_zero_problem2_neg_l219_219876


namespace apples_per_hour_l219_219447

def total_apples : ℕ := 15
def hours : ℕ := 3

theorem apples_per_hour : total_apples / hours = 5 := by
  sorry

end apples_per_hour_l219_219447


namespace arithmetic_square_root_of_9_is_3_l219_219462

-- Define the arithmetic square root property
def is_arithmetic_square_root (x : ℝ) (n : ℝ) : Prop :=
  x * x = n ∧ x ≥ 0

-- The main theorem: The arithmetic square root of 9 is 3
theorem arithmetic_square_root_of_9_is_3 : 
  is_arithmetic_square_root 3 9 :=
by
  -- This is where the proof would go, but since only the statement is required:
  sorry

end arithmetic_square_root_of_9_is_3_l219_219462


namespace total_pennies_l219_219002

theorem total_pennies (rachelle gretchen rocky max taylor : ℕ) (h_r : rachelle = 720) (h_g : gretchen = rachelle / 2)
  (h_ro : rocky = gretchen / 3) (h_m : max = rocky * 4) (h_t : taylor = max / 5) :
  rachelle + gretchen + rocky + max + taylor = 1776 := 
by
  sorry

end total_pennies_l219_219002


namespace probability_of_three_draws_l219_219213

noncomputable def box_chips : List ℕ := [1, 2, 3, 4, 5, 6, 7]

def valid_first_two_draws (a b : ℕ) : Prop :=
  a + b <= 7

def prob_three_draws_to_exceed_seven : ℚ :=
  1 / 6

theorem probability_of_three_draws :
  (∃ (draws : List ℕ), (draws.length = 3) ∧ (draws.sum > 7)
    ∧ (∀ x ∈ draws, x ∈ box_chips)
    ∧ (∀ (a b : ℕ), (a ∈ box_chips ∧ b ∈ box_chips) → valid_first_two_draws a b))
  → prob_three_draws_to_exceed_seven = 1 / 6 :=
sorry

end probability_of_three_draws_l219_219213


namespace negation_exists_ge_zero_l219_219474

theorem negation_exists_ge_zero (h : ∀ x > 0, x^2 - 3 * x + 2 < 0) :
  ∃ x > 0, x^2 - 3 * x + 2 ≥ 0 :=
sorry

end negation_exists_ge_zero_l219_219474


namespace probability_points_one_unit_apart_l219_219324

theorem probability_points_one_unit_apart :
  let total_points := 16
  let total_pairs := (total_points * (total_points - 1)) / 2
  let favorable_pairs := 12
  let probability := favorable_pairs / total_pairs
  probability = (1 : ℚ) / 10 :=
by
  sorry

end probability_points_one_unit_apart_l219_219324


namespace largest_divisor_of_n4_minus_n2_l219_219806

theorem largest_divisor_of_n4_minus_n2 :
  ∀ n : ℤ, 12 ∣ (n^4 - n^2) :=
by
  sorry

end largest_divisor_of_n4_minus_n2_l219_219806


namespace train_length_calculation_l219_219227

theorem train_length_calculation
  (speed_kmph : ℝ)
  (time_seconds : ℝ)
  (train_length : ℝ)
  (h1 : speed_kmph = 80)
  (h2 : time_seconds = 8.999280057595392)
  (h3 : train_length = (80 * 1000) / 3600 * 8.999280057595392) :
  train_length = 200 := by
  sorry

end train_length_calculation_l219_219227


namespace minimum_value_proof_l219_219424

noncomputable def minimum_value (a b : ℝ) (h : 0 < a ∧ 0 < b) : ℝ :=
  1 / (2 * a) + 1 / b

theorem minimum_value_proof (a b : ℝ) (h : 0 < a ∧ 0 < b)
  (line_bisects_circle : a + b = 1) : minimum_value a b h = (3 + 2 * Real.sqrt 2) / 2 := 
by
  sorry

end minimum_value_proof_l219_219424


namespace range_of_a_l219_219980

variable (a : ℝ)

def p : Prop := (0 < a) ∧ (a < 1)
def q : Prop := (a > (1 / 2))

theorem range_of_a (hpq_true: p a ∨ q a) (hpq_false: ¬ (p a ∧ q a)) :
  (0 < a ∧ a ≤ (1 / 2)) ∨ (a ≥ 1) :=
sorry

end range_of_a_l219_219980


namespace greatest_valid_number_l219_219499

-- Define the conditions
def is_valid_number (n : ℕ) : Prop :=
  n < 200 ∧ Nat.gcd n 30 = 5

-- Formulate the proof problem
theorem greatest_valid_number : ∃ n, is_valid_number n ∧ (∀ m, is_valid_number m → m ≤ n) ∧ n = 185 := 
by
  sorry

end greatest_valid_number_l219_219499


namespace integer_solutions_count_l219_219176

theorem integer_solutions_count :
  ∃ (s : Finset ℤ), s.card = 6 ∧ ∀ x ∈ s, 4 < Real.sqrt (3 * x) ∧ Real.sqrt (3 * x) < 6 :=
by
  sorry

end integer_solutions_count_l219_219176


namespace probability_guizhou_visit_is_9_div_14_l219_219900

noncomputable def probability_guizhou_visit : ℚ :=
  (Nat.choose 5 1 * Nat.choose 3 1 + Nat.choose 3 2) / Nat.choose 8 2

theorem probability_guizhou_visit_is_9_div_14 :
  probability_guizhou_visit = 9 / 14 :=
by
  -- Skipping the proof steps, but they must show the combinatorial calculations
  sorry

end probability_guizhou_visit_is_9_div_14_l219_219900


namespace largest_n_for_quadratic_neg_l219_219565

theorem largest_n_for_quadratic_neg (n : ℤ) : n^2 - 11 * n + 24 < 0 → n ≤ 7 :=
begin
  sorry
end

end largest_n_for_quadratic_neg_l219_219565


namespace acres_used_for_corn_l219_219672

theorem acres_used_for_corn (total_acres : ℕ) (ratio_beans ratio_wheat ratio_corn : ℕ)
    (h_total : total_acres = 1034)
    (h_ratio : ratio_beans = 5 ∧ ratio_wheat = 2 ∧ ratio_corn = 4) : 
    ratio_corn * (total_acres / (ratio_beans + ratio_wheat + ratio_corn)) = 376 := 
by
  -- Proof goes here
  sorry

end acres_used_for_corn_l219_219672


namespace kids_from_lawrence_county_go_to_camp_l219_219953

theorem kids_from_lawrence_county_go_to_camp : 
  (1201565 - 590796 = 610769) := 
by
  sorry

end kids_from_lawrence_county_go_to_camp_l219_219953


namespace L_shaped_figure_area_l219_219359

noncomputable def area_rectangle (length : ℕ) (width : ℕ) : ℕ :=
  length * width

theorem L_shaped_figure_area :
  let large_rect_length := 10
  let large_rect_width := 7
  let small_rect_length := 4
  let small_rect_width := 3
  area_rectangle large_rect_length large_rect_width - area_rectangle small_rect_length small_rect_width = 58 :=
by
  sorry

end L_shaped_figure_area_l219_219359


namespace range_of_b_l219_219902

theorem range_of_b (a b c m : ℝ) (h_ge_seq : c = b * b / a) (h_sum : a + b + c = m) (h_pos_a : a > 0) (h_pos_m : m > 0) : 
  (-m ≤ b ∧ b < 0) ∨ (0 < b ∧ b ≤ m / 3) :=
by
  sorry

end range_of_b_l219_219902


namespace total_days_needed_l219_219790

-- Define the conditions
def project1_questions : ℕ := 518
def project2_questions : ℕ := 476
def questions_per_day : ℕ := 142

-- Define the statement to prove
theorem total_days_needed :
  (project1_questions + project2_questions) / questions_per_day = 7 := by
  sorry

end total_days_needed_l219_219790


namespace coffee_table_price_l219_219589

theorem coffee_table_price :
  let sofa := 1250
  let armchairs := 2 * 425
  let rug := 350
  let bookshelf := 200
  let subtotal_without_coffee_table := sofa + armchairs + rug + bookshelf
  let C := 429.24
  let total_before_discount_and_tax := subtotal_without_coffee_table + C
  let discounted_total := total_before_discount_and_tax * 0.90
  let final_invoice_amount := discounted_total * 1.06
  final_invoice_amount = 2937.60 :=
by
  sorry

end coffee_table_price_l219_219589


namespace geometric_sequence_common_ratio_l219_219933

theorem geometric_sequence_common_ratio 
  (a1 a2 a3 a4 : ℝ) 
  (h1 : a1 = 25) 
  (h2 : a2 = -50) 
  (h3 : a3 = 100) 
  (h4 : a4 = -200)
  (h_geometric : a2 / a1 = a3 / a2 ∧ a3 / a2 = a4 / a3) : 
  a2 / a1 = -2 :=
by 
  have r1 : a2 / a1 = -2, sorry
  -- additional steps to complete proof here
  exact r1

end geometric_sequence_common_ratio_l219_219933


namespace ninth_observation_l219_219203

theorem ninth_observation (avg1 : ℝ) (avg2 : ℝ) (n1 n2 : ℝ) 
  (sum1 : n1 * avg1 = 120) 
  (sum2 : n2 * avg2 = 117) 
  (avg_decrease : avg1 - avg2 = 2) 
  (obs_count_change : n1 + 1 = n2) 
  : n2 * avg2 - n1 * avg1 = -3 :=
by
  sorry

end ninth_observation_l219_219203


namespace factorize_a_squared_minus_25_factorize_2x_squared_y_minus_8xy_plus_8y_l219_219555

-- Math Proof Problem 1
theorem factorize_a_squared_minus_25 (a : ℝ) : a^2 - 25 = (a + 5) * (a - 5) :=
by
  sorry

-- Math Proof Problem 2
theorem factorize_2x_squared_y_minus_8xy_plus_8y (x y : ℝ) : 2 * x^2 * y - 8 * x * y + 8 * y = 2 * y * (x - 2)^2 :=
by
  sorry

end factorize_a_squared_minus_25_factorize_2x_squared_y_minus_8xy_plus_8y_l219_219555


namespace hyperbola_asymptotes_l219_219713

theorem hyperbola_asymptotes (a b : ℝ) (he : 2 = Real.sqrt (a^2 + b^2) / a)
  (hyperbola_eq : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1) :
    ∀ x y, (Real.sqrt 3 * x + y = 0) ∨ (Real.sqrt 3 * x - y = 0) :=
by
  sorry

end hyperbola_asymptotes_l219_219713


namespace greatest_valid_number_l219_219497

-- Define the conditions
def is_valid_number (n : ℕ) : Prop :=
  n < 200 ∧ Nat.gcd n 30 = 5

-- Formulate the proof problem
theorem greatest_valid_number : ∃ n, is_valid_number n ∧ (∀ m, is_valid_number m → m ≤ n) ∧ n = 185 := 
by
  sorry

end greatest_valid_number_l219_219497


namespace smallest_period_range_interval_l219_219712

open Real

noncomputable def f (x : ℝ) : ℝ :=
  2 * cos x ^ 2 + 2 * sqrt 3 * sin x * cos x

theorem smallest_period (x : ℝ) : ∃ T > 0, ∀ t : ℝ, f (x + T) = f x := by
  use π
  sorry

theorem range_interval : set.image f (set.Icc (-π / 6) (π / 4)) = set.Icc 0 3 := by
  sorry

end smallest_period_range_interval_l219_219712


namespace coffee_mix_price_l219_219217

theorem coffee_mix_price (
  weight1 price1 weight2 price2 total_weight : ℝ)
  (h1 : weight1 = 9)
  (h2 : price1 = 2.15)
  (h3 : weight2 = 9)
  (h4 : price2 = 2.45)
  (h5 : total_weight = 18)
  :
  (weight1 * price1 + weight2 * price2) / total_weight = 2.30 :=
by
  sorry

end coffee_mix_price_l219_219217


namespace knight_probability_2023_moves_l219_219666

noncomputable def position_after_n_moves (n : ℕ) 
  (moves : ℕ → (ℤ × ℤ) → (ℤ × ℤ)) (start : ℤ × ℤ) : ℤ × ℤ :=
if n = 0 then start else moves n (position_after_n_moves (n - 1) moves start)

noncomputable def knight_moves (n : ℕ) (pos : ℤ × ℤ) : ℤ × ℤ :=
let (a, b) := pos in
match (n % 8) with
| 0 => (a + 1, b + 2)
| 1 => (a - 1, b + 2)
| 2 => (a + 1, b - 2)
| 3 => (a - 1, b - 2)
| 4 => (a + 2, b + 1)
| 5 => (a - 2, b + 1)
| 6 => (a + 2, b - 1)
| 7 => (a - 2, b - 1)
| _ => (a, b) -- impossible case for pattern matching completeness
end

noncomputable def probability_of_position (n : ℕ) (target: ℤ × ℤ) : ℝ :=
if target = (4, 5) then (1 / 32 - 1 / (2 ^ (n + 4))) else 0 -- simplified for the specific (4,5) case 

theorem knight_probability_2023_moves : 
  probability_of_position 2023 (4, 5) = (1 / 32 - 1 / 2 ^ 2027) :=
by sorry

end knight_probability_2023_moves_l219_219666


namespace angle_bisector_ratio_l219_219734

theorem angle_bisector_ratio (XY XZ YZ : ℝ) (hXY : XY = 8) (hXZ : XZ = 6) (hYZ : YZ = 4) :
  ∃ (Q : Point) (YQ QV : ℝ), YQ / QV = 2 :=
by
  sorry

end angle_bisector_ratio_l219_219734


namespace chess_player_total_games_l219_219356

noncomputable def total_games_played (W L : ℕ) : ℕ :=
  W + L

theorem chess_player_total_games :
  ∃ (W L : ℕ), W = 16 ∧ (L : ℚ) / W = 7 / 4 ∧ total_games_played W L = 44 :=
by
  sorry

end chess_player_total_games_l219_219356


namespace sue_initially_borrowed_six_movies_l219_219456

variable (M : ℕ)
variable (initial_books : ℕ := 15)
variable (returned_books : ℕ := 8)
variable (returned_movies_fraction : ℚ := 1/3)
variable (additional_books : ℕ := 9)
variable (total_items : ℕ := 20)

theorem sue_initially_borrowed_six_movies (hM : total_items = initial_books - returned_books + additional_books + (M - returned_movies_fraction * M)) : 
  M = 6 := by
  sorry

end sue_initially_borrowed_six_movies_l219_219456


namespace each_wolf_kills_one_deer_l219_219768

-- Definitions based on conditions
def hunting_wolves : Nat := 4
def additional_wolves : Nat := 16
def wolves_per_pack : Nat := hunting_wolves + additional_wolves
def meat_per_wolf_per_day : Nat := 8
def days_between_hunts : Nat := 5
def meat_per_wolf : Nat := meat_per_wolf_per_day * days_between_hunts
def total_meat_required : Nat := wolves_per_pack * meat_per_wolf
def meat_per_deer : Nat := 200
def deer_needed : Nat := total_meat_required / meat_per_deer
def deer_per_wolf_needed : Nat := deer_needed / hunting_wolves

-- Lean statement to prove
theorem each_wolf_kills_one_deer (hunting_wolves : Nat := 4) (additional_wolves : Nat := 16) 
    (meat_per_wolf_per_day : Nat := 8) (days_between_hunts : Nat := 5) 
    (meat_per_deer : Nat := 200) : deer_per_wolf_needed = 1 := 
by
  -- Proof required here
  sorry

end each_wolf_kills_one_deer_l219_219768


namespace inequality_lt_l219_219818

theorem inequality_lt (x y : ℝ) (h1 : x > y) (h2 : y > 0) (n k : ℕ) (h3 : n > k) :
  (x^k - y^k) ^ n < (x^n - y^n) ^ k := 
  sorry

end inequality_lt_l219_219818


namespace Tamara_height_l219_219753

-- Define the conditions and goal as a theorem
theorem Tamara_height (K T : ℕ) (h1 : T = 3 * K - 4) (h2 : K + T = 92) : T = 68 :=
by
  sorry

end Tamara_height_l219_219753


namespace geometric_mean_of_1_and_9_is_pm3_l219_219707

theorem geometric_mean_of_1_and_9_is_pm3 (a b c : ℝ) (h₀ : a = 1) (h₁ : b = 9) (h₂ : c^2 = a * b) : c = 3 ∨ c = -3 := by
  sorry

end geometric_mean_of_1_and_9_is_pm3_l219_219707


namespace radius_of_ball_l219_219521

theorem radius_of_ball (diameter depth : ℝ) (h₁ : diameter = 30) (h₂ : depth = 10) : 
  ∃ r : ℝ, r = 25 :=
by
  sorry

end radius_of_ball_l219_219521


namespace q1_correct_q2_correct_q3_correct_l219_219749

-- Given conditions
def roll_die := {n : ℕ // n ≥ 1 ∧ n ≤ 6}
def three_digit_number := (roll_die × roll_die × roll_die)

-- Question 1: Number of distinct three-digit numbers
noncomputable def distinct_three_digit_count : ℕ :=
  fintype.card { x : three_digit_number // x.1 ≠ x.2 ∧ x.1 ≠ x.3 ∧ x.2 ≠ x.3 }

theorem q1_correct : distinct_three_digit_count = 120 := 
by
  sorry

-- Question 2: Total number of three-digit numbers
noncomputable def total_three_digit_count : ℕ := 
  fintype.card three_digit_number

theorem q2_correct : total_three_digit_count = 216 := 
by
  sorry

-- Question 3: Number of three-digit numbers with exactly two identical digits
noncomputable def exactly_two_identical_digits_count : ℕ := 
  fintype.card { x : three_digit_number // (x.1 = x.2 ∧ x.1 ≠ x.3) ∨ (x.2 = x.3 ∧ x.2 ≠ x.1) ∨ (x.1 = x.3 ∧ x.1 ≠ x.2)}

theorem q3_correct : exactly_two_identical_digits_count = 90 := 
by
  sorry

end q1_correct_q2_correct_q3_correct_l219_219749


namespace greatest_integer_with_gcf_5_l219_219491

theorem greatest_integer_with_gcf_5 :
  ∃ n, n < 200 ∧ gcd n 30 = 5 ∧ ∀ m, m < 200 ∧ gcd m 30 = 5 → m ≤ n :=
by
  sorry

end greatest_integer_with_gcf_5_l219_219491


namespace only_root_is_4_l219_219631

noncomputable def equation_one (x : ℝ) : ℝ := (2 * x^2) / (x - 1) - (2 * x + 7) / 3 + (4 - 6 * x) / (x - 1) + 1

noncomputable def equation_two (x : ℝ) : ℝ := x^2 - 5 * x + 4

theorem only_root_is_4 (x : ℝ) (h: equation_one x = 0) (h_transformation: equation_two x = 0) : x = 4 := sorry

end only_root_is_4_l219_219631


namespace Kayla_picked_40_apples_l219_219739

variable (x : ℕ) (kayla kylie total : ℕ)
variables (h1 : total = kayla + kylie) (h2 : kayla = 1/4 * kylie) (h3 : total = 200)

theorem Kayla_picked_40_apples (x : ℕ) (hx1 : (5/4) * x = 200): 
  1/4 * x = 40 :=
by {
  have h4: x = 160, from sorry,
  rw h4,
  exact (show 1/4 * 160 = 40, by norm_num)
}

end Kayla_picked_40_apples_l219_219739


namespace acres_used_for_corn_l219_219667

-- Define the conditions in the problem:
def total_land : ℕ := 1034
def ratio_beans : ℕ := 5
def ratio_wheat : ℕ := 2
def ratio_corn : ℕ := 4
def total_ratio : ℕ := ratio_beans + ratio_wheat + ratio_corn

-- Proof problem statement: Prove the number of acres used for corn is 376 acres
theorem acres_used_for_corn : total_land * ratio_corn / total_ratio = 376 := by
  -- Proof goes here
  sorry

end acres_used_for_corn_l219_219667


namespace end_same_digit_l219_219829

theorem end_same_digit
  (a b : ℕ)
  (h : (2 * a + b) % 10 = (2 * b + a) % 10) :
  a % 10 = b % 10 :=
by
  sorry

end end_same_digit_l219_219829


namespace two_painters_days_l219_219142

-- Define the conditions and the proof problem
def five_painters_days : ℕ := 5
def days_per_five_painters : ℕ := 2
def total_painter_days : ℕ := five_painters_days * days_per_five_painters -- Total painter-days for the original scenario
def two_painters : ℕ := 2
def last_day_painter_half_day : ℕ := 1 -- Indicating that one painter works half a day on the last day
def last_day_work : ℕ := two_painters - last_day_painter_half_day / 2 -- Total work on the last day is equivalent to 1.5 painter-days

theorem two_painters_days : total_painter_days = 5 :=
by
  sorry -- Mathematical proof goes here

end two_painters_days_l219_219142


namespace parabola_intersection_points_l219_219036

theorem parabola_intersection_points :
  let parabola1 := λ x : ℝ => 4*x^2 + 3*x - 1
  let parabola2 := λ x : ℝ => x^2 + 8*x + 7
  ∃ (x₁ y₁ x₂ y₂ : ℝ), x₁ = -4/3 ∧ y₁ = -17/9 ∧
                        x₂ = 2 ∧ y₂ = 27 ∧
                        parabola1 x₁ = y₁ ∧ 
                        parabola2 x₁ = y₁ ∧
                        parabola1 x₂ = y₂ ∧
                        parabola2 x₂ = y₂ :=
by {
  sorry
}

end parabola_intersection_points_l219_219036


namespace find_k_l219_219840

variables (k : ℝ)
def vector_a : ℝ × ℝ := (1, 2)
def vector_b : ℝ × ℝ := (-3, 2)
def vector_k_a_plus_b (k : ℝ) : ℝ × ℝ := (k*1 + (-3), k*2 + 2)
def vector_a_minus_2b : ℝ × ℝ := (1 - 2*(-3), 2 - 2*2)

theorem find_k (h : (vector_k_a_plus_b k).fst * (vector_a_minus_2b).snd = (vector_k_a_plus_b k).snd * (vector_a_minus_2b).fst) : k = -1/2 :=
sorry

end find_k_l219_219840


namespace sufficient_not_necessary_l219_219089

def M : Set Int := {0, 1, 2}
def N : Set Int := {-1, 0, 1, 2}

theorem sufficient_not_necessary (a : Int) : a ∈ M → a ∈ N ∧ ¬(a ∈ N → a ∈ M) := by
  sorry

end sufficient_not_necessary_l219_219089


namespace data_transmission_time_l219_219954

def packet_size : ℕ := 256
def num_packets : ℕ := 100
def transmission_rate : ℕ := 200
def total_data : ℕ := num_packets * packet_size
def transmission_time_in_seconds : ℚ := total_data / transmission_rate
def transmission_time_in_minutes : ℚ := transmission_time_in_seconds / 60

theorem data_transmission_time :
  transmission_time_in_minutes = 2 :=
  sorry

end data_transmission_time_l219_219954


namespace number_is_76_l219_219782

theorem number_is_76 (x : ℝ) (h : (3 / 4) * x = x - 19) : x = 76 :=
sorry

end number_is_76_l219_219782


namespace expand_product_l219_219553

-- We need to state the problem as a theorem
theorem expand_product (y : ℝ) (hy : y ≠ 0) : (3 / 7) * (7 / y + 14 * y^3) = 3 / y + 6 * y^3 :=
by
  sorry -- Skipping the proof

end expand_product_l219_219553


namespace solve_equation_l219_219886

theorem solve_equation (x y z : ℕ) (hx : 0 ≤ x ∧ x ≤ 9) (hy : 0 ≤ y ∧ y ≤ 9)
  (hz : 0 ≤ z ∧ z ≤ 9) (h_eq : 1 / (x + y + z) = (x * 100 + y * 10 + z) / 1000) :
  x = 1 ∧ y = 2 ∧ z = 5 :=
by
  sorry

end solve_equation_l219_219886


namespace elaine_earnings_increase_l219_219124

variable (E : ℝ) -- Elaine's earnings last year
variable (P : ℝ) -- Percentage increase in earnings

-- Conditions
variable (rent_last_year : ℝ := 0.20 * E)
variable (earnings_this_year : ℝ := E * (1 + P / 100))
variable (rent_this_year : ℝ := 0.30 * earnings_this_year)
variable (multiplied_rent_last_year : ℝ := 1.875 * rent_last_year)

-- Theorem to be proven
theorem elaine_earnings_increase (h : rent_this_year = multiplied_rent_last_year) : P = 25 :=
by
  sorry

end elaine_earnings_increase_l219_219124


namespace highest_price_without_lowering_revenue_minimum_annual_sales_volume_and_price_l219_219057

noncomputable def original_price : ℝ := 25
noncomputable def original_sales_volume : ℝ := 80000
noncomputable def sales_volume_decrease_per_yuan_increase : ℝ := 2000

-- Question 1
theorem highest_price_without_lowering_revenue :
  ∀ (x : ℝ), 
  25 ≤ x ∧ (8 - (x - original_price) * 0.2) * x ≥ 25 * 8 → 
  x ≤ 40 :=
sorry

-- Question 2
noncomputable def tech_reform_fee (x : ℝ) : ℝ := (1 / 6) * (x^2 - 600)
noncomputable def fixed_promotion_fee : ℝ := 50
noncomputable def variable_promotion_fee (x : ℝ) : ℝ := (1 / 5) * x

theorem minimum_annual_sales_volume_and_price (x : ℝ) (a : ℝ) :
  x > 25 →
  (a * x ≥ 25 * 8 + fixed_promotion_fee + tech_reform_fee x + variable_promotion_fee x) →
  (a ≥ 10.2 ∧ x = 30) :=
sorry

end highest_price_without_lowering_revenue_minimum_annual_sales_volume_and_price_l219_219057


namespace number_of_valid_subcommittees_l219_219333

theorem number_of_valid_subcommittees : 
  let total_members := 12
  let professors := 5
  let subcommittee_size := 4
  let total_subcommittees := Nat.choose total_members subcommittee_size
  let zero_prof_subcommittees := Nat.choose (total_members - professors) subcommittee_size
  let one_prof_subcommittees := professors * Nat.choose (total_members - professors) (subcommittee_size - 1)
  let non_valid_subcommittees := zero_prof_subcommittees + one_prof_subcommittees
  valid_subcommittees := 285 in
  total_subcommittees - non_valid_subcommittees = valid_subcommittees :=
by 
  sorry

end number_of_valid_subcommittees_l219_219333


namespace common_factor_of_right_triangle_l219_219428

theorem common_factor_of_right_triangle (d : ℝ) 
  (h_triangle : (2*d)^2 + (4*d)^2 = (5*d)^2) 
  (h_side : 2*d = 45 ∨ 4*d = 45 ∨ 5*d = 45) : 
  d = 9 :=
sorry

end common_factor_of_right_triangle_l219_219428


namespace evaluate_expression_l219_219809

/- The mathematical statement to prove:

Evaluate the expression 2/10 + 4/20 + 6/30, then multiply the result by 3
and show that it equals to 9/5.
-/

theorem evaluate_expression : 
  (2 / 10 + 4 / 20 + 6 / 30) * 3 = 9 / 5 := 
by 
  sorry

end evaluate_expression_l219_219809


namespace part1_part2_l219_219574

-- Statement for part (1)
theorem part1 (m : ℝ) : 
  (∀ x1 x2 : ℝ, (m - 1) * x1^2 + 3 * x1 - 2 = 0 ∧ 
               (m - 1) * x2^2 + 3 * x2 - 2 = 0 ∧ x1 ≠ x2) ↔ m > -1/8 :=
sorry

-- Statement for part (2)
theorem part2 (m : ℝ) : 
  (∃ x : ℝ, (m - 1) * x^2 + 3 * x - 2 = 0 ∧ ∀ y : ℝ, (m - 1) * y^2 + 3 * y - 2 = 0 → y = x) ↔ 
  (m = 1 ∨ m = -1/8) :=
sorry

end part1_part2_l219_219574


namespace var_power_eight_l219_219292

variable (k j : ℝ)
variable {x y z : ℝ}

theorem var_power_eight (hx : x = k * y^4) (hy : y = j * z^2) : ∃ c : ℝ, x = c * z^8 :=
by
  sorry

end var_power_eight_l219_219292


namespace x_plus_inverse_x_eq_eight_imp_x4_plus_inverse_x4_eq_3842_l219_219985

theorem x_plus_inverse_x_eq_eight_imp_x4_plus_inverse_x4_eq_3842
  (x : ℝ) (h : x + 1/x = 8) : x^4 + 1/x^4 = 3842 :=
sorry

end x_plus_inverse_x_eq_eight_imp_x4_plus_inverse_x4_eq_3842_l219_219985


namespace sqrt_computation_l219_219798

open Real

theorem sqrt_computation : sqrt ((5: ℝ)^2 * (7: ℝ)^4) = 245 :=
by
  -- Proof here
  sorry

end sqrt_computation_l219_219798


namespace first_meeting_time_of_boys_l219_219903

theorem first_meeting_time_of_boys 
  (L : ℝ) (v1_kmh : ℝ) (v2_kmh : ℝ) (v1_ms v2_ms : ℝ) (rel_speed : ℝ) (t : ℝ)
  (hv1_km_to_ms : v1_ms = v1_kmh * 1000 / 3600)
  (hv2_km_to_ms : v2_ms = v2_kmh * 1000 / 3600)
  (hrel_speed : rel_speed = v1_ms + v2_ms)
  (hl : L = 4800)
  (hv1 : v1_kmh = 60)
  (hv2 : v2_kmh = 100)
  (ht : t = L / rel_speed) :
  t = 108 := by
  -- we're providing a placeholder for the proof
  sorry

end first_meeting_time_of_boys_l219_219903


namespace percentage_decrease_wages_l219_219123

theorem percentage_decrease_wages (W : ℝ) (P : ℝ) : 
  (0.20 * W * (1 - P / 100)) = 0.70 * (0.20 * W) → 
  P = 30 :=
by
  sorry

end percentage_decrease_wages_l219_219123


namespace min_value_Px_Py_l219_219546

def P (τ : ℝ) : ℝ := (τ + 1)^3

theorem min_value_Px_Py (x y : ℝ) (h : x + y = 0) : P x + P y = 2 :=
sorry

end min_value_Px_Py_l219_219546


namespace total_cost_l219_219536

def cost_of_items (x y : ℝ) : Prop :=
  (6 * x + 5 * y = 6.10) ∧ (3 * x + 4 * y = 4.60)

theorem total_cost (x y : ℝ) (h : cost_of_items x y) : 12 * x + 8 * y = 10.16 :=
by
  sorry

end total_cost_l219_219536


namespace friend_spent_11_l219_219913

-- Definitions of the conditions
def total_lunch_cost (you friend : ℝ) : Prop := you + friend = 19
def friend_spent_more (you friend : ℝ) : Prop := friend = you + 3

-- The theorem to prove
theorem friend_spent_11 (you friend : ℝ) 
  (h1 : total_lunch_cost you friend) 
  (h2 : friend_spent_more you friend) : 
  friend = 11 := 
by 
  sorry

end friend_spent_11_l219_219913


namespace roots_difference_l219_219248

theorem roots_difference (a b c : ℝ) (h_eq : a = 1) (h_b : b = -11) (h_c : c = 24) :
    let r1 := (-b + Real.sqrt (b ^ 2 - 4 * a * c)) / (2 * a)
    let r2 := (-b - Real.sqrt (b ^ 2 - 4 * a * c)) / (2 * a)
    r1 - r2 = 5 := 
by
  sorry

end roots_difference_l219_219248


namespace total_toothpicks_needed_l219_219531

/-- The number of toothpicks needed to construct both a large and smaller equilateral triangle 
    side by side, given the large triangle has a base of 100 small triangles and the smaller triangle 
    has a base of 50 small triangles -/
theorem total_toothpicks_needed 
  (base_large : ℕ) (base_small : ℕ) (shared_boundary : ℕ) 
  (h1 : base_large = 100) (h2 : base_small = 50) (h3 : shared_boundary = base_small) :
  3 * (100 * 101 / 2) / 2 + 3 * (50 * 51 / 2) / 2 - shared_boundary = 9462 := 
sorry

end total_toothpicks_needed_l219_219531


namespace solve_fractional_eq_l219_219621

theorem solve_fractional_eq (x : ℝ) (h1 : x ≠ -1) (h2 : x ≠ 1) : (x / (x + 1) - 1 = 3 / (x - 1)) → x = -1 / 2 :=
by
  sorry

end solve_fractional_eq_l219_219621


namespace sqrt_of_16_l219_219028

theorem sqrt_of_16 (x : ℝ) (hx : x^2 = 16) : x = 4 ∨ x = -4 := 
by
  sorry

end sqrt_of_16_l219_219028


namespace compute_binom_product_l219_219239

/-- Definition of binomial coefficient -/
def binomial (n k : ℕ) : ℕ := n.choose k

/-- The main theorem to prove -/
theorem compute_binom_product : binomial 10 3 * binomial 8 3 = 6720 :=
by
  sorry

end compute_binom_product_l219_219239


namespace student_ticket_price_l219_219157

-- Define the conditions
variables (S T : ℝ)
def condition1 := 4 * S + 3 * T = 79
def condition2 := 12 * S + 10 * T = 246

-- Prove that the price of a student ticket is 9 dollars, given the equations above
theorem student_ticket_price (h1 : condition1 S T) (h2 : condition2 S T) : T = 9 :=
sorry

end student_ticket_price_l219_219157


namespace parallelogram_area_l219_219592

open Matrix

noncomputable def u : Fin 2 → ℝ := ![7, -4]
noncomputable def z : Fin 2 → ℝ := ![8, -1]

theorem parallelogram_area :
  let matrix := ![u, z]
  |det (of fun (i j : Fin 2) => (matrix i) j)| = 25 :=
by
  sorry

end parallelogram_area_l219_219592


namespace marquita_gardens_l219_219606

open Nat

theorem marquita_gardens (num_mancino_gardens : ℕ) 
  (length_mancino_garden width_mancino_garden : ℕ) 
  (num_marquita_gardens : ℕ) 
  (length_marquita_garden width_marquita_garden : ℕ)
  (total_area : ℕ) 
  (h1 : num_mancino_gardens = 3)
  (h2 : length_mancino_garden = 16)
  (h3 : width_mancino_garden = 5)
  (h4 : length_marquita_garden = 8)
  (h5 : width_marquita_garden = 4)
  (h6 : total_area = 304)
  (hmancino_area : num_mancino_gardens * (length_mancino_garden * width_mancino_garden) = 3 * (16 * 5))
  (hcombined_area : total_area = num_mancino_gardens * (length_mancino_garden * width_mancino_garden) + num_marquita_gardens * (length_marquita_garden * width_marquita_garden)) :
  num_marquita_gardens = 2 :=
sorry

end marquita_gardens_l219_219606


namespace mean_of_remaining_students_l219_219296

noncomputable def mean_remaining_students (k : ℕ) (h : k > 18) (mean_class : ℚ) (mean_18_students : ℚ) : ℚ :=
  (12 * k - 360) / (k - 18)

theorem mean_of_remaining_students (k : ℕ) (h : k > 18) (mean_class_eq : mean_class = 12) (mean_18_eq : mean_18_students = 20) :
  mean_remaining_students k h mean_class mean_18_students = (12 * k - 360) / (k - 18) :=
by sorry

end mean_of_remaining_students_l219_219296


namespace equilibrium_proof_l219_219920

noncomputable def equilibrium_constant (Γ_eq B_eq : ℝ) : ℝ :=
(Γ_eq ^ 3) / (B_eq ^ 3)

theorem equilibrium_proof (Γ_eq B_eq : ℝ) (K_c : ℝ) (B_initial : ℝ) (Γ_initial : ℝ)
  (hΓ : Γ_eq = 0.25) (hB : B_eq = 0.15) (hKc : K_c = 4.63) 
  (ratio : Γ_eq = B_eq + B_initial) (hΓ_initial : Γ_initial = 0) :
  equilibrium_constant Γ_eq B_eq = K_c ∧ 
  B_initial = 0.4 ∧ 
  Γ_initial = 0 := 
by
  sorry

end equilibrium_proof_l219_219920


namespace f_bounded_by_inverse_l219_219825

theorem f_bounded_by_inverse (f : ℕ → ℝ) (h_pos : ∀ n, 0 < f n) (h_rec : ∀ n, (f n)^2 ≤ f n - f (n + 1)) :
  ∀ n, f n < 1 / (n + 1) :=
by
  sorry

end f_bounded_by_inverse_l219_219825


namespace negation_of_universal_sin_l219_219331

theorem negation_of_universal_sin (h : ∀ x : ℝ, Real.sin x > 0) : ∃ x : ℝ, Real.sin x ≤ 0 :=
sorry

end negation_of_universal_sin_l219_219331


namespace game_C_more_likely_than_game_D_l219_219787

-- Definitions for the probabilities
def p_heads : ℚ := 3 / 4
def p_tails : ℚ := 1 / 4

-- Game C probability
def p_game_C : ℚ := p_heads ^ 4

-- Game D probabilities for each scenario
def p_game_D_scenario1 : ℚ := (p_heads ^ 3) * (p_heads ^ 2)
def p_game_D_scenario2 : ℚ := (p_heads ^ 3) * (p_tails ^ 2)
def p_game_D_scenario3 : ℚ := (p_tails ^ 3) * (p_heads ^ 2)
def p_game_D_scenario4 : ℚ := (p_tails ^ 3) * (p_tails ^ 2)

-- Total probability for Game D
def p_game_D : ℚ :=
  p_game_D_scenario1 + p_game_D_scenario2 + p_game_D_scenario3 + p_game_D_scenario4

-- Proof statement
theorem game_C_more_likely_than_game_D : (p_game_C - p_game_D) = 11 / 256 := by
  sorry

end game_C_more_likely_than_game_D_l219_219787


namespace youngest_child_age_l219_219031

theorem youngest_child_age (x : ℕ) 
  (h : x + (x + 3) + (x + 6) + (x + 9) + (x + 12) = 50) : x = 4 := 
by 
  sorry

end youngest_child_age_l219_219031


namespace speed_of_X_l219_219937

theorem speed_of_X (t1 t2 Vx : ℝ) (h1 : t2 - t1 = 3) 
  (h2 : 3 * Vx + Vx * t1 = 60 * t1 + 30)
  (h3 : 3 * Vx + Vx * t2 + 30 = 60 * t2) : Vx = 60 :=
by sorry

end speed_of_X_l219_219937


namespace yolka_probability_correct_l219_219370

open ProbabilityTheory

noncomputable def yolka_meeting_probability : ℝ :=
  let anya_last := 1 / 3
  let borya_vasya_meet := (144 - (0.5 * (81 + 100))) / 144
  in anya_last * borya_vasya_meet

theorem yolka_probability_correct :
  yolka_meeting_probability = 0.124 := 
by sorry

end yolka_probability_correct_l219_219370


namespace parabola_focus_coordinates_l219_219629

theorem parabola_focus_coordinates (a : ℝ) (h : a ≠ 0) :
  ∃ x y : ℝ, y = 4 * a * x^2 → (x, y) = (0, 1 / (16 * a)) :=
by
  sorry

end parabola_focus_coordinates_l219_219629


namespace geometric_series_seventh_term_l219_219720

theorem geometric_series_seventh_term (a₁ a₁₀ : ℝ) (n : ℝ) (r : ℝ) :
  a₁ = 4 →
  a₁₀ = 93312 →
  n = 10 →
  a₁₀ = a₁ * r^(n-1) →
  (∃ (r : ℝ), r = 6) →
  4 * 6^(7-1) = 186624 := by
  intros a1_eq a10_eq n_eq an_eq exists_r
  sorry

end geometric_series_seventh_term_l219_219720


namespace brick_length_proof_l219_219214

-- Defining relevant parameters and conditions
def width_of_brick : ℝ := 10 -- width in cm
def height_of_brick : ℝ := 7.5 -- height in cm
def wall_length : ℝ := 26 -- length in m
def wall_width : ℝ := 2 -- width in m
def wall_height : ℝ := 0.75 -- height in m
def num_bricks : ℝ := 26000 

-- Defining known volumes for conversion
def volume_of_wall_m3 : ℝ := wall_length * wall_width * wall_height
def volume_of_wall_cm3 : ℝ := volume_of_wall_m3 * 1000000 -- converting m³ to cm³

-- Volume of one brick given the unknown length L
def volume_of_one_brick (L : ℝ) : ℝ := L * width_of_brick * height_of_brick

-- Total volume of bricks is the volume of one brick times the number of bricks
def total_volume_of_bricks (L : ℝ) : ℝ := volume_of_one_brick L * num_bricks

-- The length of the brick is found by equating the total volume of bricks to the volume of the wall
theorem brick_length_proof : ∃ L : ℝ, total_volume_of_bricks L = volume_of_wall_cm3 ∧ L = 20 :=
by
  existsi 20
  sorry

end brick_length_proof_l219_219214


namespace product_of_slopes_l219_219405

theorem product_of_slopes (p : ℝ) (hp : 0 < p) :
  let T := (p, 0)
  let parabola := fun x y => y^2 = 2*p*x
  let line := fun x y => y = x - p
  -- Define intersection points A and B on the parabola satisfying the line equation
  ∃ A B : ℝ × ℝ, 
  parabola A.1 A.2 ∧ line A.1 A.2 ∧
  parabola B.1 B.2 ∧ line B.1 B.2 ∧
  -- O is the origin
  let O := (0, 0)
  -- define slope function
  let slope (P Q : ℝ × ℝ) := (Q.2 - P.2) / (Q.1 - P.1)
  -- slopes of OA and OB
  let k_OA := slope O A
  let k_OB := slope O B
  -- product of slopes
  k_OA * k_OB = -2 := sorry

end product_of_slopes_l219_219405


namespace polynomial_expansion_p_eq_l219_219023

theorem polynomial_expansion_p_eq (p q : ℝ) (h1 : 10 * p^9 * q = 45 * p^8 * q^2) (h2 : p + 2 * q = 1) (hp : p > 0) (hq : q > 0) : p = 9 / 13 :=
by
  sorry

end polynomial_expansion_p_eq_l219_219023


namespace sqrt_five_squared_times_seven_fourth_correct_l219_219802

noncomputable def sqrt_five_squared_times_seven_fourth : Prop :=
  sqrt (5^2 * 7^4) = 245

theorem sqrt_five_squared_times_seven_fourth_correct : sqrt_five_squared_times_seven_fourth := by
  sorry

end sqrt_five_squared_times_seven_fourth_correct_l219_219802


namespace probability_diagonals_intersect_l219_219727

theorem probability_diagonals_intersect {n : ℕ} :
  (2 * n + 1 > 2) → 
  ∀ (total_diagonals : ℕ) (total_combinations : ℕ) (intersecting_pairs : ℕ),
    total_diagonals = 2 * n^2 - n - 1 →
    total_combinations = (total_diagonals * (total_diagonals - 1)) / 2 →
    intersecting_pairs = ((2 * n + 1) * n * (2 * n - 1) * (n - 1)) / 6 →
    (intersecting_pairs : ℚ) / (total_combinations : ℚ) = n * (2 * n - 1) / (3 * (2 * n^2 - n - 2)) := sorry

end probability_diagonals_intersect_l219_219727


namespace polygon_sides_in_arithmetic_progression_l219_219471

theorem polygon_sides_in_arithmetic_progression 
  (n : ℕ) 
  (d : ℕ := 3)
  (max_angle : ℕ := 150)
  (sum_of_interior_angles : ℕ := 180 * (n - 2)) 
  (a_n : ℕ := max_angle) : 
  (max_angle - d * (n - 1) + max_angle) * n / 2 = sum_of_interior_angles → 
  n = 28 :=
by 
  sorry

end polygon_sides_in_arithmetic_progression_l219_219471


namespace Jason_cards_l219_219119

theorem Jason_cards (initial_cards : ℕ) (cards_bought : ℕ) (remaining_cards : ℕ) 
  (h1 : initial_cards = 3) (h2 : cards_bought = 2) : remaining_cards = 1 :=
by
  sorry

end Jason_cards_l219_219119


namespace transformed_quadratic_l219_219319

theorem transformed_quadratic (a b c n x : ℝ) (h : a * x^2 + b * x + c = 0) :
  a * x^2 + n * b * x + n^2 * c = 0 :=
sorry

end transformed_quadratic_l219_219319


namespace problem_statement_l219_219286

def f (x : ℝ) : ℝ := x^3 + 1
def g (x : ℝ) : ℝ := 3 * x - 2

theorem problem_statement : f (g (f (g 2))) = 7189058 := by
  sorry

end problem_statement_l219_219286


namespace algebraic_expression_value_l219_219095

theorem algebraic_expression_value (m x n : ℝ)
  (h1 : (m + 3) * x ^ (|m| - 2) + 6 * m = 0)
  (h2 : n * x - 5 = x * (3 - n))
  (h3 : |m| = 2)
  (h4 : (m + 3) ≠ 0) :
  (m + x) ^ 2000 * (-m ^ 2 * n + x * n ^ 2) + 1 = 1 := by
  sorry

end algebraic_expression_value_l219_219095


namespace percentage_problem_l219_219678

theorem percentage_problem (N : ℕ) (P : ℕ) (h1 : N = 25) (h2 : N = (P * N / 100) + 21) : P = 16 :=
sorry

end percentage_problem_l219_219678


namespace min_AP_BP_l219_219872

-- Definitions based on conditions in the problem
def A : ℝ × ℝ := (2, 0)
def B : ℝ × ℝ := (7, 6)
def parabola (x y : ℝ) : Prop := y^2 = 8 * x

-- The theorem to prove the minimum value of AP + BP
theorem min_AP_BP
  (P : ℝ × ℝ)
  (hP_parabola : parabola P.1 P.2) :
  dist P A + dist P B ≥ 9 :=
sorry

end min_AP_BP_l219_219872


namespace problem_statement_l219_219972

-- Definitions of f and g based on given conditions
variables {f g : ℝ → ℝ} (hf : ∀ x : ℝ, f (-x) = -f x) (hg : ∀ x : ℝ, g (-x) = g x)
          (hdf : ∀ x : ℝ, x > 0 → deriv f x > 0) (hdg : ∀ x : ℝ, x > 0 → deriv g (-x) > 0)

theorem problem_statement :
  ∀ x : ℝ, x < 0 → deriv f x > 0 ∧ deriv g (-x) < 0 :=
by
  sorry

end problem_statement_l219_219972


namespace total_holes_dug_l219_219867

theorem total_holes_dug :
  (Pearl_digging_rate * 21 + Miguel_digging_rate * 21) = 26 :=
by
  -- Definitions based on conditions
  let Pearl_digging_rate := 4 / 7
  let Miguel_digging_rate := 2 / 3
  -- Sorry placeholder for the proof
  sorry

end total_holes_dug_l219_219867


namespace probability_one_instrument_l219_219517

theorem probability_one_instrument (total_people : ℕ) (at_least_one_instrument_ratio : ℚ) (two_or_more_instruments : ℕ)
  (h1 : total_people = 800) (h2 : at_least_one_instrument_ratio = 1 / 5) (h3 : two_or_more_instruments = 128) :
  (160 - 128) / 800 = 1 / 25 :=
by
  sorry

end probability_one_instrument_l219_219517


namespace sequence_general_term_l219_219397

theorem sequence_general_term (a : ℕ → ℝ) (h₁ : a 1 = 1)
  (h₂ : ∀ n : ℕ, n > 0 → a (n + 1) = 3 * a n / (a n + 3)) :
  ∀ n : ℕ, n > 0 → a n = 3 / (n + 2) := 
by 
  sorry

end sequence_general_term_l219_219397


namespace triangle_equilateral_if_condition_l219_219862

-- Define the given conditions
variables {A B C : ℝ} -- Angles
variables {a b c : ℝ} -- Opposite sides

-- Assume the condition that a/ cos(A) = b/ cos(B) = c/ cos(C)
def triangle_condition (A B C a b c : ℝ) : Prop :=
  a / Real.cos A = b / Real.cos B ∧ b / Real.cos B = c / Real.cos C

-- The theorem to prove under these conditions
theorem triangle_equilateral_if_condition (A B C a b c : ℝ) 
  (h : triangle_condition A B C a b c) : 
  A = B ∧ B = C :=
sorry

end triangle_equilateral_if_condition_l219_219862


namespace average_of_25_results_l219_219184

theorem average_of_25_results (first12_avg : ℕ -> ℕ -> ℕ)
                             (last12_avg : ℕ -> ℕ -> ℕ) 
                             (res13 : ℕ)
                             (avg_of_25 : ℕ) :
                             first12_avg 12 10 = 120
                             ∧ last12_avg 12 20 = 240
                             ∧ res13 = 90
                             ∧ avg_of_25 = (first12_avg 12 10 + last12_avg 12 20 + res13) / 25
                             → avg_of_25 = 18 := by
  sorry

end average_of_25_results_l219_219184


namespace probability_of_event_A_l219_219189

noncomputable def probability_both_pieces_no_less_than_three_meters (L : ℝ) (a : ℝ) (b : ℝ) : ℝ :=
  if h : L = a + b 
  then (if a ≥ 3 ∧ b ≥ 3 then (L - 2 * 3) / L else 0)
  else 0

theorem probability_of_event_A : 
  probability_both_pieces_no_less_than_three_meters 11 6 5 = 5 / 11 :=
by
  -- Additional context to ensure proper definition of the problem
  sorry

end probability_of_event_A_l219_219189


namespace trader_excess_donations_l219_219939

-- Define the conditions
def profit : ℤ := 1200
def allocation_percentage : ℤ := 60
def family_donation : ℤ := 250
def friends_donation : ℤ := (20 * family_donation) / 100 + family_donation
def total_family_friends_donation : ℤ := family_donation + friends_donation
def local_association_donation : ℤ := 15 * total_family_friends_donation / 10
def total_donations : ℤ := family_donation + friends_donation + local_association_donation
def allocated_amount : ℤ := allocation_percentage * profit / 100

-- Theorem statement (Question)
theorem trader_excess_donations : total_donations - allocated_amount = 655 :=
by
  sorry

end trader_excess_donations_l219_219939


namespace average_all_results_l219_219654

theorem average_all_results (s₁ s₂ : ℤ) (n₁ n₂ : ℤ) (h₁ : n₁ = 60) (h₂ : n₂ = 40) (avg₁ : s₁ / n₁ = 40) (avg₂ : s₂ / n₂ = 60) : 
  ((s₁ + s₂) / (n₁ + n₂) = 48) :=
sorry

end average_all_results_l219_219654


namespace units_digit_of_5_to_4_l219_219908

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_5_to_4 : units_digit (5^4) = 5 := by
  -- The definition ensures that 5^4 = 625 and the units digit is 5
  sorry

end units_digit_of_5_to_4_l219_219908


namespace female_employees_sampled_l219_219926

theorem female_employees_sampled
  (T : ℕ) -- Total number of employees
  (M : ℕ) -- Number of male employees
  (F : ℕ) -- Number of female employees
  (S_m : ℕ) -- Number of sampled male employees
  (H_T : T = 140)
  (H_M : M = 80)
  (H_F : F = 60)
  (H_Sm : S_m = 16) :
  ∃ S_f : ℕ, S_f = 12 :=
by
  sorry

end female_employees_sampled_l219_219926


namespace problem_solution_l219_219255

theorem problem_solution :
  { x : ℝ // (x / 4 ≤ 3 + x) ∧ (3 + x < -3 * (1 + x)) } = { x : ℝ // x ∈ Set.Ico (-4 : ℝ) (-(3 / 2) : ℝ) } :=
by
  sorry

end problem_solution_l219_219255


namespace negation_of_proposition_l219_219763

noncomputable def P (x : ℝ) : Prop := x^2 + 1 ≥ 0

theorem negation_of_proposition :
  (¬ ∀ x, x > 1 → P x) ↔ (∃ x, x > 1 ∧ ¬ P x) :=
sorry

end negation_of_proposition_l219_219763


namespace max_profit_l219_219780

noncomputable def fixed_cost : ℝ := 2.5
noncomputable def var_cost (x : ℕ) : ℝ :=
  if x < 80 then (x^2 + 10 * x) * 1e4
  else (51 * x - 1450) * 1e4
noncomputable def revenue (x : ℕ) : ℝ := 500 * x * 1e4
noncomputable def profit (x : ℕ) : ℝ := revenue x - var_cost x - fixed_cost * 1e4

theorem max_profit (x : ℕ) :
  (∀ y : ℕ, profit y ≤ 43200 * 1e4) ∧ profit 100 = 43200 * 1e4 := by
  sorry

end max_profit_l219_219780


namespace total_squares_in_4x4_grid_l219_219843

-- Define the grid size
def grid_size : ℕ := 4

-- Define a function to count the number of k x k squares in an n x n grid
def count_squares (n k : ℕ) : ℕ :=
  (n - k + 1) * (n - k + 1)

-- Total number of squares in a 4 x 4 grid
def total_squares (n : ℕ) : ℕ :=
  count_squares n 1 + count_squares n 2 + count_squares n 3 + count_squares n 4

-- The main theorem asserting the total number of squares in a 4 x 4 grid is 30
theorem total_squares_in_4x4_grid : total_squares grid_size = 30 := by
  sorry

end total_squares_in_4x4_grid_l219_219843


namespace car_mpg_city_l219_219662

theorem car_mpg_city (h c t : ℕ) (H1 : 560 = h * t) (H2 : 336 = c * t) (H3 : c = h - 6) : c = 9 :=
by
  sorry

end car_mpg_city_l219_219662


namespace chocolate_bar_percentage_l219_219743

theorem chocolate_bar_percentage (milk_chocolate dark_chocolate almond_chocolate white_chocolate : ℕ)
  (h1 : milk_chocolate = 25) (h2 : dark_chocolate = 25)
  (h3 : almond_chocolate = 25) (h4 : white_chocolate = 25) :
  (milk_chocolate * 100) / (milk_chocolate + dark_chocolate + almond_chocolate + white_chocolate) = 25 ∧
  (dark_chocolate * 100) / (milk_chocolate + dark_chocolate + almond_chocolate + white_chocolate) = 25 ∧
  (almond_chocolate * 100) / (milk_chocolate + dark_chocolate + almond_chocolate + white_chocolate) = 25 ∧
  (white_chocolate * 100) / (milk_chocolate + dark_chocolate + almond_chocolate + white_chocolate) = 25 :=
by
  sorry

end chocolate_bar_percentage_l219_219743


namespace time_saved_1200_miles_l219_219609

theorem time_saved_1200_miles
  (distance : ℕ)
  (speed1 speed2 : ℕ)
  (h_distance : distance = 1200)
  (h_speed1 : speed1 = 60)
  (h_speed2 : speed2 = 50) :
  (distance / speed2) - (distance / speed1) = 4 :=
by
  sorry

end time_saved_1200_miles_l219_219609


namespace diagonal_cubes_140_320_360_l219_219054

-- Define the problem parameters 
def length_x : ℕ := 140
def length_y : ℕ := 320
def length_z : ℕ := 360

-- Define the function to calculate the number of unit cubes the internal diagonal passes through.
def num_cubes_diagonal (x y z : ℕ) : ℕ :=
  x + y + z - Nat.gcd x y - Nat.gcd y z - Nat.gcd z x + Nat.gcd (Nat.gcd x y) z

-- The target theorem to be proven
theorem diagonal_cubes_140_320_360 :
  num_cubes_diagonal length_x length_y length_z = 760 :=
by
  sorry

end diagonal_cubes_140_320_360_l219_219054


namespace probability_of_product_is_29_over_36_l219_219441

open Classical

def probability_product_leq_36 :=
  let p_outcome := [1, 2, 3, 4, 5, 6]
  let m_outcome := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
  let pairs := [(p, m) | p ← p_outcome, m ← m_outcome]
  let valid_pairs := pairs.filter (λ pair => pair.1 * pair.2 ≤ 36)
  let total_prob := (valid_pairs.length : ℚ) / (pairs.length : ℚ)
  total_prob

theorem probability_of_product_is_29_over_36 :
  probability_product_leq_36 = 29 / 36 := 
by
  sorry

end probability_of_product_is_29_over_36_l219_219441


namespace value_of_f2008_plus_f2009_l219_219093

variable {f : ℤ → ℤ}

-- Conditions
axiom h1 : ∀ x : ℤ, f (-(x) + 2) = -f (x + 2)
axiom h2 : ∀ x : ℤ, f (6 - x) = f x
axiom h3 : f 3 = 2

-- The theorem to prove
theorem value_of_f2008_plus_f2009 : f 2008 + f 2009 = -2 :=
  sorry

end value_of_f2008_plus_f2009_l219_219093


namespace point_exists_l219_219597

theorem point_exists (P : Fin 1993 → ℤ × ℤ)
  (H0 : ∀ i : Fin 1993, (P i).1 ∈ Int ∧ (P i).2 ∈ Int)
  (H1 : ∀ i : Fin 1992, ∀ x y : ℚ, 
    (P i).1 ≤ x ∧ x ≤ (P (i+1)).1 ∧ 
    (P i).2 ≤ y ∧ y ≤ (P (i+1)).2 → 
    ((x ∉ Int) ∨ (y ∉ Int))): 
  ∃ i : Fin 1992, ∃ Q : ℚ × ℚ, 
  (Q.1 = (↑((P i).1) + ↑((P (i+1)).1)) / 2) ∧ 
  (Q.2 = (↑((P i).2) + ↑((P (i+1)).2)) / 2) ∧ 
  (Odd (2 * Q.1.num)) ∧ 
  (Odd (2 * Q.2.num)) :=
sorry

end point_exists_l219_219597


namespace digital_earth_storage_technology_matured_l219_219758

-- Definitions of conditions as technology properties
def NanoStorageTechnology : Prop := 
  -- Assume it has matured (based on solution analysis)
  sorry

def LaserHolographicStorageTechnology : Prop :=
  -- Assume it has matured (based on solution analysis)
  sorry

def ProteinStorageTechnology : Prop :=
  -- Assume it has matured (based on solution analysis)
  sorry

def DistributedStorageTechnology : Prop :=
  -- Assume it has matured (based on solution analysis)
  sorry

def VirtualStorageTechnology : Prop :=
  -- Assume it has not matured or is not relevant
  sorry

def SpatialStorageTechnology : Prop :=
  -- Assume it has not matured or is not relevant
  sorry

def VisualizationStorageTechnology : Prop :=
  -- Assume it has not matured or is not relevant
  sorry

-- Lean statement to prove the combination
theorem digital_earth_storage_technology_matured : 
  NanoStorageTechnology ∧ LaserHolographicStorageTechnology ∧ ProteinStorageTechnology ∧ DistributedStorageTechnology :=
by {
  sorry
}

end digital_earth_storage_technology_matured_l219_219758


namespace lcm_of_18_and_20_l219_219958

theorem lcm_of_18_and_20 : Nat.lcm 18 20 = 180 := by
  sorry

end lcm_of_18_and_20_l219_219958


namespace option_D_correct_l219_219510

theorem option_D_correct (y : ℝ) : -9 * y^2 + 16 * y^2 = 7 * y^2 :=
by sorry

end option_D_correct_l219_219510


namespace nine_a_eq_frac_minus_eighty_one_over_eleven_l219_219097

theorem nine_a_eq_frac_minus_eighty_one_over_eleven (a b : ℚ) 
  (h1 : 8 * a + 3 * b = 0) 
  (h2 : a = b - 3) : 
  9 * a = -81 / 11 := 
sorry

end nine_a_eq_frac_minus_eighty_one_over_eleven_l219_219097


namespace proof_solution_arithmetic_progression_l219_219583

noncomputable def system_has_solution (a b c m : ℝ) : Prop :=
  (m = 1 → a = b ∧ b = c) ∧
  (m = -2 → a + b + c = 0) ∧ 
  (m ≠ -2 ∧ m ≠ 1 → ∃ x y z : ℝ, x + y + m * z = a ∧ x + m * y + z = b ∧ m * x + y + z = c)

def abc_arithmetic_progression (a b c : ℝ) : Prop :=
  2 * b = a + c

theorem proof_solution_arithmetic_progression (a b c m : ℝ) : 
  system_has_solution a b c m → 
  (∃ x y z : ℝ, x + y + m * z = a ∧ x + m * y + z = b ∧ m * x + y + z = c ∧ 2 * y = x + z) ↔
  abc_arithmetic_progression a b c := 
by 
  sorry

end proof_solution_arithmetic_progression_l219_219583


namespace probability_12OA_l219_219858

-- Definitions based on conditions
def is_digit (s: Char) : Prop := s.isdigit

-- Vowels in the problem
def is_vowel (ch : Char) : Prop := ch ∈ ['A', 'E', 'I', 'O', 'U']

-- License plate validity given the conditions in the country of Mathlandia
def valid_license_plate (plate : List Char) : Prop :=
  plate.length = 4 ∧ is_digit plate.head ∧ is_digit plate.tail.head ∧
  ∃ p3 p4, p3 ∈ set_of (λ (c: Char), true) ∧ p4 ∈ set_of (λ (c: Char), true) ∧
  (is_vowel p3 ∨ is_vowel p4)

-- Total number of valid plates
def total_plates : ℕ := 21000

-- Specific plate "12OA"
def plate_12OA : List Char := ['1', '2', 'O', 'A']

-- Formal proof problem statement
theorem probability_12OA : 
  valid_license_plate plate_12OA →
  ∀ total_plates : ℕ, total_plates = 21000 →
  \frac{1}{total_plates} = \frac{1}{21000} :=
begin
  sorry
end

end probability_12OA_l219_219858


namespace konjok_gorbunok_should_act_l219_219721

def magical_power_retention (eat : ℕ → Prop) (sleep : ℕ → Prop) (seven_days : ℕ) : Prop :=
  ∀ t : ℕ, (0 ≤ t ∧ t ≤ seven_days) → ¬(eat t ∨ sleep t)

def retains_power (need_action : Prop) : Prop :=
  need_action

theorem konjok_gorbunok_should_act
  (eat : ℕ → Prop) (sleep : ℕ → Prop)
  (seven_days : ℕ)
  (h : magical_power_retention eat sleep seven_days)
  (before_start : ℕ → Prop) :
  retains_power (before_start seven_days) :=
by
  sorry

end konjok_gorbunok_should_act_l219_219721


namespace surface_area_geometric_mean_volume_geometric_mean_l219_219135

noncomputable def surfaces_areas_proof (r : ℝ) (π : ℝ) : Prop :=
  let F_1 := 6 * π * r^2
  let F_2 := 4 * π * r^2
  let F_3 := 9 * π * r^2
  F_1^2 = F_2 * F_3

noncomputable def volumes_proof (r : ℝ) (π : ℝ) : Prop :=
  let V_1 := 2 * π * r^3
  let V_2 := (4 / 3) * π * r^3
  let V_3 := π * r^3
  V_1^2 = V_2 * V_3

theorem surface_area_geometric_mean (r : ℝ) (π : ℝ) : surfaces_areas_proof r π := 
  sorry

theorem volume_geometric_mean (r : ℝ) (π : ℝ) : volumes_proof r π :=
  sorry

end surface_area_geometric_mean_volume_geometric_mean_l219_219135


namespace decorations_count_l219_219947

-- Define the conditions as Lean definitions
def plastic_skulls := 12
def broomsticks := 4
def spiderwebs := 12
def pumpkins := 2 * spiderwebs
def large_cauldron := 1
def budget_more_decorations := 20
def left_to_put_up := 10

-- Define the total decorations
def decorations_already_up := plastic_skulls + broomsticks + spiderwebs + pumpkins + large_cauldron
def additional_decorations := budget_more_decorations + left_to_put_up
def total_decorations := decorations_already_up + additional_decorations

-- Prove the total number of decorations will be 83
theorem decorations_count : total_decorations = 83 := by 
  sorry

end decorations_count_l219_219947


namespace value_of_expression_l219_219479

theorem value_of_expression : 2 * 2015 - 2015 = 2015 :=
by
  sorry

end value_of_expression_l219_219479


namespace integers_even_condition_l219_219111

-- Definitions based on conditions
def is_even (n : ℤ) : Prop := n % 2 = 0

def exactly_one_even (a b c : ℤ) : Prop :=
(is_even a ∧ ¬ is_even b ∧ ¬ is_even c) ∨
(¬ is_even a ∧ is_even b ∧ ¬ is_even c) ∨
(¬ is_even a ∧ ¬ is_even b ∧ is_even c)

-- Proof statement
theorem integers_even_condition (a b c : ℤ) (h : ¬ exactly_one_even a b c) :
  (¬ is_even a ∧ ¬ is_even b ∧ ¬ is_even c) ∨
  (is_even a ∧ is_even b) ∨
  (is_even a ∧ is_even c) ∨
  (is_even b ∧ is_even c) :=
sorry

end integers_even_condition_l219_219111


namespace bread_cost_l219_219162

theorem bread_cost {packs_meat packs_cheese sandwiches : ℕ} 
  (cost_meat cost_cheese cost_sandwich coupon_meat coupon_cheese total_cost : ℝ) 
  (h_meat_cost : cost_meat = 5.00) 
  (h_cheese_cost : cost_cheese = 4.00)
  (h_coupon_meat : coupon_meat = 1.00)
  (h_coupon_cheese : coupon_cheese = 1.00)
  (h_cost_sandwich : cost_sandwich = 2.00)
  (h_packs_meat : packs_meat = 2)
  (h_packs_cheese : packs_cheese = 2)
  (h_sandwiches : sandwiches = 10)
  (h_total_revenue : total_cost = sandwiches * cost_sandwich) :
  ∃ (bread_cost : ℝ), bread_cost = total_cost - ((packs_meat * cost_meat - coupon_meat) + (packs_cheese * cost_cheese - coupon_cheese)) :=
sorry

end bread_cost_l219_219162


namespace range_of_2m_plus_n_l219_219711

noncomputable def f (x : ℝ) : ℝ := abs (Real.log x / Real.log 3)

theorem range_of_2m_plus_n {m n : ℝ} (hmn : 0 < m ∧ m < n) (heq : f m = f n) :
  ∃ y, y ∈ Set.Ici (2 * Real.sqrt 2) ∧ (2 * m + n = y) :=
sorry

end range_of_2m_plus_n_l219_219711


namespace combined_points_correct_l219_219295

-- Definitions for the points scored by each player
def points_Lemuel := 7 * 2 + 5 * 3 + 4
def points_Marcus := 4 * 2 + 6 * 3 + 7
def points_Kevin := 9 * 2 + 4 * 3 + 5
def points_Olivia := 6 * 2 + 3 * 3 + 6

-- Definition for the combined points scored by both teams
def combined_points := points_Lemuel + points_Marcus + points_Kevin + points_Olivia

-- Theorem statement to prove combined points equals 128
theorem combined_points_correct : combined_points = 128 :=
by
  -- Lean proof goes here
  sorry

end combined_points_correct_l219_219295


namespace eve_discovers_secret_l219_219064

theorem eve_discovers_secret (x : ℕ) : ∃ (n : ℕ), ∃ (is_prime : ℕ → Prop), (∀ m : ℕ, (is_prime (x + n * m)) ∨ (¬is_prime (x + n * m))) :=
  sorry

end eve_discovers_secret_l219_219064


namespace compute_binom_product_l219_219240

/-- Definition of binomial coefficient -/
def binomial (n k : ℕ) : ℕ := n.choose k

/-- The main theorem to prove -/
theorem compute_binom_product : binomial 10 3 * binomial 8 3 = 6720 :=
by
  sorry

end compute_binom_product_l219_219240


namespace polynomial_bound_implies_l219_219475

theorem polynomial_bound_implies :
  (∀ x : ℝ, |x| ≤ 1 → |a * x^2 + b * x + c| ≤ 1) →
  (∀ x : ℝ, |x| ≤ 1 → |c * x^2 + b * x + a| ≤ 2) :=
by
  sorry

end polynomial_bound_implies_l219_219475


namespace find_number_l219_219229

theorem find_number (x : ℤ) (h : (x + 305) / 16 = 31) : x = 191 :=
sorry

end find_number_l219_219229


namespace simplify_expression_l219_219884

def expr_initial (y : ℝ) := 3*y + 4*y^2 + 2 - (7 - 3*y - 4*y^2)
def expr_simplified (y : ℝ) := 8*y^2 + 6*y - 5

theorem simplify_expression (y : ℝ) : expr_initial y = expr_simplified y :=
by
  sorry

end simplify_expression_l219_219884


namespace isosceles_triangle_side_l219_219988

theorem isosceles_triangle_side (a : ℝ) : 
  (10 - a = 7 ∨ 10 - a = 6) ↔ (a = 3 ∨ a = 4) := 
by sorry

end isosceles_triangle_side_l219_219988


namespace approximate_roots_l219_219957

noncomputable def f (x : ℝ) : ℝ := 0.3 * x^3 - 2 * x^2 - 0.2 * x + 0.5

theorem approximate_roots : 
  ∃ x₁ x₂ x₃ : ℝ, 
    (f x₁ = 0 ∧ |x₁ + 0.4| < 0.1) ∧ 
    (f x₂ = 0 ∧ |x₂ - 0.5| < 0.1) ∧ 
    (f x₃ = 0 ∧ |x₃ - 2.6| < 0.1) :=
by
  sorry

end approximate_roots_l219_219957


namespace nina_total_cost_l219_219315

-- Define the cost of the first pair of shoes
def first_pair_cost : ℕ := 22

-- Define the cost of the second pair of shoes
def second_pair_cost : ℕ := first_pair_cost + (first_pair_cost / 2)

-- Define the total cost for both pairs of shoes
def total_cost : ℕ := first_pair_cost + second_pair_cost

-- The formal statement of the problem
theorem nina_total_cost : total_cost = 55 := by
  sorry

end nina_total_cost_l219_219315


namespace other_root_is_minus_two_l219_219402

theorem other_root_is_minus_two (b : ℝ) (h : 1^2 + b * 1 - 2 = 0) : 
  ∃ (x : ℝ), x = -2 ∧ x^2 + b * x - 2 = 0 :=
by
  sorry

end other_root_is_minus_two_l219_219402


namespace solution_set_inequality_l219_219476

theorem solution_set_inequality (x : ℝ) (h : 0 < x ∧ x ≤ 1) : 
  ∀ (x : ℝ), (0 < x ∧ x ≤ 1 ↔ ∀ a > 0, ∀ b ≤ 1, (2/x + (1-x) ^ (1/2) ≥ 1 + (1-x)^(1/2))) := sorry

end solution_set_inequality_l219_219476


namespace real_values_of_a_l219_219814

noncomputable def P (x a b : ℝ) : ℝ := x^2 - 2 * a * x + b

theorem real_values_of_a (a b : ℝ) :
  (P 0 a b ≠ 0) →
  (P 1 a b ≠ 0) →
  (P 2 a b ≠ 0) →
  (P 1 a b / P 0 a b = P 2 a b / P 1 a b) →
  (∃ b, P x 1 b = 0) :=
by
  sorry

end real_values_of_a_l219_219814


namespace value_of_polynomial_l219_219418

variable {R : Type} [CommRing R]

theorem value_of_polynomial 
  (m : R) 
  (h : 2 * m^2 - 3 * m - 1 = 0) : 
  6 * m^2 - 9 * m + 2019 = 2022 := by
  sorry

end value_of_polynomial_l219_219418


namespace museum_ticket_cost_l219_219206

theorem museum_ticket_cost 
  (num_students : ℕ) (num_teachers : ℕ) 
  (student_ticket_cost : ℕ) (teacher_ticket_cost : ℕ)
  (h_students : num_students = 12) (h_teachers : num_teachers = 4)
  (h_student_cost : student_ticket_cost = 1) (h_teacher_cost : teacher_ticket_cost = 3) :
  num_students * student_ticket_cost + num_teachers * teacher_ticket_cost = 24 :=
by
  sorry

end museum_ticket_cost_l219_219206


namespace binomial_product_l219_219243

open Nat

theorem binomial_product :
  (Nat.choose 10 3) * (Nat.choose 8 3) = 6720 :=
by
  sorry

end binomial_product_l219_219243


namespace Annika_three_times_Hans_in_future_l219_219998

theorem Annika_three_times_Hans_in_future
  (hans_age_now : Nat)
  (annika_age_now : Nat)
  (x : Nat)
  (hans_future_age : Nat)
  (annika_future_age : Nat)
  (H1 : hans_age_now = 8)
  (H2 : annika_age_now = 32)
  (H3 : hans_future_age = hans_age_now + x)
  (H4 : annika_future_age = annika_age_now + x)
  (H5 : annika_future_age = 3 * hans_future_age) :
  x = 4 := 
  by
  sorry

end Annika_three_times_Hans_in_future_l219_219998


namespace S_3_eq_11_S_n_general_l219_219130

-- Define x_n as {1, 2, ..., n}
def x_n (n : ℕ) : Finset ℕ := Finset.range (n + 1) \ {0}

-- Define f(A) as the smallest element in A
def f (A : Finset ℕ) : ℕ := A.min' ⟨1, by simp [x_n]⟩

-- Define S_n as the sum of f(A) across all non-empty subsets of x_n
def S_n (n : ℕ) : ℕ := 
  (Finset.powerset (x_n n)).filter (λ A, ¬A = ∅).sum f

-- Theorems to prove the given results
theorem S_3_eq_11 : S_n 3 = 11 :=
by sorry

theorem S_n_general (n : ℕ) : S_n n = 2^(n+1) - n - 2 :=
by sorry

end S_3_eq_11_S_n_general_l219_219130


namespace positive_integer_solution_of_inequality_l219_219170

theorem positive_integer_solution_of_inequality (x : ℕ) (h : 0 < x) : (3 * x - 1) / 2 + 1 ≥ 2 * x → x = 1 :=
by
  intros
  sorry

end positive_integer_solution_of_inequality_l219_219170


namespace james_ate_slices_l219_219301

variable (NumPizzas : ℕ) (SlicesPerPizza : ℕ) (FractionEaten : ℚ)
variable (TotalSlices : ℕ := NumPizzas * SlicesPerPizza)
variable (JamesSlices : ℚ := FractionEaten * TotalSlices)

theorem james_ate_slices (h1 : NumPizzas = 2) (h2 : SlicesPerPizza = 6) (h3 : FractionEaten = 2 / 3) :
    JamesSlices = 8 := 
by 
  simp [JamesSlices, TotalSlices]
  rw [h1, h2, h3]
  norm_num
  sorry

end james_ate_slices_l219_219301


namespace time_to_cross_pole_is_2_5_l219_219864

noncomputable def time_to_cross_pole : ℝ :=
  let length_of_train := 100 -- meters
  let speed_km_per_hr := 144 -- km/hr
  let speed_m_per_s := speed_km_per_hr * 1000 / 3600 -- converting speed to m/s
  length_of_train / speed_m_per_s

theorem time_to_cross_pole_is_2_5 :
  time_to_cross_pole = 2.5 :=
by
  -- The Lean proof will be written here.
  -- Placeholder for the formal proof.
  sorry

end time_to_cross_pole_is_2_5_l219_219864


namespace maximum_sum_each_side_equals_22_l219_219956

theorem maximum_sum_each_side_equals_22 (A B C D : ℕ) :
  (∀ i, 1 ≤ i ∧ i ≤ 10)
  → (∀ S, S = A ∨ S = B ∨ S = C ∨ S = D ∧ A + B + C + D = 33)
  → (A + B + C + D + 55) / 4 = 22 :=
by
  sorry

end maximum_sum_each_side_equals_22_l219_219956


namespace find_g_25_l219_219085

noncomputable def g (x : ℝ) : ℝ := sorry

axiom h₁ : ∀ (x y : ℝ), x > 0 → y > 0 → g (x / y) = (y / x) * g x
axiom h₂ : g 50 = 4

theorem find_g_25 : g 25 = 4 / 25 :=
by {
  sorry
}

end find_g_25_l219_219085


namespace geometric_sequence_sum_l219_219427

theorem geometric_sequence_sum (a : ℕ → ℝ) (r : ℝ) 
  (h1 : ∀ n, a (n + 1) = r * a n)
  (h2 : 0 < r)
  (h3 : a 1 = 3)
  (h4 : a 1 + a 2 + a 3 = 21) :
  a 3 + a 4 + a 5 = 84 :=
sorry

end geometric_sequence_sum_l219_219427


namespace sufficiency_not_necessity_l219_219576

theorem sufficiency_not_necessity (x y : ℝ) :
  (x > 3 ∧ y > 3) → (x + y > 6 ∧ x * y > 9) ∧ (¬ (x + y > 6 ∧ x * y > 9 → x > 3 ∧ y > 3)) :=
by
  sorry

end sufficiency_not_necessity_l219_219576


namespace number_of_distinct_linear_recurrences_l219_219599

open BigOperators

/-
  Let p be a prime positive integer.
  Define a mod-p recurrence of degree n to be a sequence {a_k}_{k >= 0} of numbers modulo p 
  satisfying a relation of the form:

  ai+n = c_n-1 ai+n-1 + ... + c_1 ai+1 + c_0 ai
  for all i >= 0, where c_0, c_1, ..., c_n-1 are integers and c_0 not equivalent to 0 mod p.
  Compute the number of distinct linear recurrences of degree at most n in terms of p and n.
-/
theorem number_of_distinct_linear_recurrences (p n : ℕ) (hp : Nat.Prime p) : 
  ∃ d : ℕ, 
    (∀ {a : ℕ → ℕ} {c : ℕ → ℕ} (h : ∀ i, a (i + n) = ∑ j in Finset.range n, c j * a (i + j))
     (hc0 : c 0 ≠ 0), 
      d = (1 - n * (p - 1) / (p + 1) + p^2 * (p^(2 * n) - 1) / (p + 1)^2 : ℚ)) :=
  sorry

end number_of_distinct_linear_recurrences_l219_219599


namespace exponent_multiplication_l219_219341

theorem exponent_multiplication :
  (10^(3/4)) * (10^(-0.25)) * (10^(1.5)) = 10^2 :=
by sorry

end exponent_multiplication_l219_219341


namespace concert_tickets_l219_219771

theorem concert_tickets : ∃ (A B : ℕ), 8 * A + 425 * B = 3000000 ∧ A + B = 4500 ∧ A = 2900 := by
  sorry

end concert_tickets_l219_219771


namespace egg_count_l219_219643

theorem egg_count :
  ∃ x : ℕ, 
    (∀ e1 e10 e100 : ℤ, 
      (e1 = 1 ∨ e1 = -1) →
      (e10 = 10 ∨ e10 = -10) →
      (e100 = 100 ∨ e100 = -100) →
      7 * x + e1 + e10 + e100 = 3162) → 
    x = 439 :=
by 
  sorry

end egg_count_l219_219643


namespace melanie_total_amount_l219_219313

theorem melanie_total_amount :
  let g1 := 12
  let g2 := 15
  let g3 := 8
  let g4 := 10
  let g5 := 20
  g1 + g2 + g3 + g4 + g5 = 65 :=
by
  sorry

end melanie_total_amount_l219_219313


namespace smallest_k_49_divides_binom_l219_219373

theorem smallest_k_49_divides_binom : 
  ∃ k : ℕ, 0 < k ∧ 49 ∣ Nat.choose (2 * k) k ∧ (∀ m : ℕ, 0 < m ∧ 49 ∣ Nat.choose (2 * m) m → k ≤ m) ∧ k = 25 :=
by
  sorry

end smallest_k_49_divides_binom_l219_219373


namespace sally_initial_cards_l219_219750

def initial_baseball_cards (t w s a : ℕ) : Prop :=
  a = w + s + t

theorem sally_initial_cards :
  ∃ (initial_cards : ℕ), initial_baseball_cards 9 24 15 initial_cards ∧ initial_cards = 48 :=
by
  use 48
  sorry

end sally_initial_cards_l219_219750


namespace smallest_circle_radius_eq_l219_219430

open Real

-- Declaring the problem's conditions
def largestCircleRadius : ℝ := 10
def smallestCirclesCount : ℕ := 6
def congruentSmallerCirclesFitWithinLargerCircle (r : ℝ) : Prop :=
  3 * (2 * r) = 2 * largestCircleRadius

-- Stating the theorem to prove
theorem smallest_circle_radius_eq :
  ∃ r : ℝ, congruentSmallerCirclesFitWithinLargerCircle r ∧ r = 10 / 3 :=
by
  sorry

end smallest_circle_radius_eq_l219_219430


namespace triangle_properties_l219_219580

theorem triangle_properties (a b c : ℝ) 
  (h : |a - Real.sqrt 7| + Real.sqrt (b - 5) + (c - 4 * Real.sqrt 2)^2 = 0) :
  a = Real.sqrt 7 ∧ b = 5 ∧ c = 4 * Real.sqrt 2 ∧ a^2 + b^2 = c^2 := by
{
  sorry
}

end triangle_properties_l219_219580


namespace shopkeeper_gain_percentage_l219_219358

noncomputable def gain_percentage (false_weight: ℕ) (true_weight: ℕ) : ℝ :=
  (↑(true_weight - false_weight) / ↑false_weight) * 100

theorem shopkeeper_gain_percentage :
  gain_percentage 960 1000 = 4.166666666666667 := 
sorry

end shopkeeper_gain_percentage_l219_219358


namespace freken_bok_weight_l219_219306

variables (K F M : ℕ)

theorem freken_bok_weight 
  (h1 : K + F = M + 75) 
  (h2 : F + M = K + 45) : 
  F = 60 :=
sorry

end freken_bok_weight_l219_219306


namespace difference_between_picked_and_left_is_five_l219_219805

theorem difference_between_picked_and_left_is_five :
  let dave_sticks := 14
  let amy_sticks := 9
  let ben_sticks := 12
  let total_initial_sticks := 65
  let total_picked_up := dave_sticks + amy_sticks + ben_sticks
  let sticks_left := total_initial_sticks - total_picked_up
  total_picked_up - sticks_left = 5 :=
by
  sorry

end difference_between_picked_and_left_is_five_l219_219805


namespace negation_of_P_l219_219330

-- Defining the original proposition
def P : Prop := ∃ x₀ : ℝ, x₀^2 = 1

-- The problem is to prove the negation of the proposition
theorem negation_of_P : (¬P) ↔ (∀ x : ℝ, x^2 ≠ 1) :=
  by sorry

end negation_of_P_l219_219330


namespace compute_z_pow_7_l219_219125

namespace ComplexProof

noncomputable def z : ℂ := (Real.sqrt 3 + Complex.I) / 2

theorem compute_z_pow_7 : z ^ 7 = - (Real.sqrt 3 / 2) - (1 / 2) * Complex.I :=
by
  sorry

end ComplexProof

end compute_z_pow_7_l219_219125


namespace tangent_parallel_l219_219091

theorem tangent_parallel (a b : ℝ) 
  (h1 : b = (1 / 3) * a^3 - (1 / 2) * a^2 + 1) 
  (h2 : (a^2 - a) = 2) : 
  a = 2 ∨ a = -1 :=
by {
  -- proof skipped
  sorry
}

end tangent_parallel_l219_219091


namespace venus_speed_mph_l219_219484

theorem venus_speed_mph (speed_mps : ℝ) (seconds_per_hour : ℝ) (mph : ℝ) 
  (h1 : speed_mps = 21.9) 
  (h2 : seconds_per_hour = 3600)
  (h3 : mph = speed_mps * seconds_per_hour) : 
  mph = 78840 := 
  by 
  sorry

end venus_speed_mph_l219_219484


namespace path_exists_probability_l219_219300

-- Let the maze segments be randomly colored black or white
open ProbabilityTheory

noncomputable def maze_path_probability :=
  let event_a_b : Event (PathExists 'white A B) := sorry
  let event_c_d : Event (PathExists 'black C D) := sorry
  Pr[event_a_b] = 1 / 2 ∧ Pr[event_c_d] = 1 / 2

theorem path_exists_probability :
  maze_path_probability :=
by sorry

end path_exists_probability_l219_219300


namespace cistern_fill_time_l219_219525

variable (A_rate : ℚ) (B_rate : ℚ) (C_rate : ℚ)
variable (total_rate : ℚ := A_rate + C_rate - B_rate)

theorem cistern_fill_time (hA : A_rate = 1/7) (hB : B_rate = 1/9) (hC : C_rate = 1/12) :
  (1/total_rate) = 252/29 :=
by
  rw [hA, hB, hC]
  sorry

end cistern_fill_time_l219_219525


namespace max_sum_abc_divisible_by_13_l219_219852

theorem max_sum_abc_divisible_by_13 :
  ∃ (A B C : ℕ), A ≤ 9 ∧ B ≤ 9 ∧ C ≤ 9 ∧ 13 ∣ (2000 + 100 * A + 10 * B + C) ∧ (A + B + C = 26) :=
by
  sorry

end max_sum_abc_divisible_by_13_l219_219852


namespace problem_statement_l219_219132

theorem problem_statement 
  (x y z : ℝ) 
  (hx1 : x ≠ 1) 
  (hy1 : y ≠ 1) 
  (hz1 : z ≠ 1) 
  (hxyz : x * y * z = 1) : 
  x^2 / (x - 1)^2 + y^2 / (y - 1)^2 + z^2 / (z - 1)^2 ≥ 1 :=
sorry

end problem_statement_l219_219132


namespace john_overall_profit_l219_219201

-- Definitions based on conditions
def cost_grinder : ℕ := 15000
def cost_mobile : ℕ := 8000
def loss_percentage_grinder : ℚ := 4 / 100
def profit_percentage_mobile : ℚ := 15 / 100

-- Calculations based on the conditions
def loss_amount_grinder := cost_grinder * loss_percentage_grinder
def selling_price_grinder := cost_grinder - loss_amount_grinder
def profit_amount_mobile := cost_mobile * profit_percentage_mobile
def selling_price_mobile := cost_mobile + profit_amount_mobile
def total_cost_price := cost_grinder + cost_mobile
def total_selling_price := selling_price_grinder + selling_price_mobile

-- Overall profit calculation
def overall_profit := total_selling_price - total_cost_price

-- Proof statement to prove the overall profit
theorem john_overall_profit : overall_profit = 600 := by
  sorry

end john_overall_profit_l219_219201


namespace total_cars_l219_219034

theorem total_cars (Tommy_cars Jessie_cars : ℕ) (older_brother_cars : ℕ) 
  (h1 : Tommy_cars = 3) 
  (h2 : Jessie_cars = 3)
  (h3 : older_brother_cars = Tommy_cars + Jessie_cars + 5) : 
  Tommy_cars + Jessie_cars + older_brother_cars = 17 := by
  sorry

end total_cars_l219_219034


namespace band_row_lengths_l219_219055

theorem band_row_lengths (x y : ℕ) :
  (x * y = 90) → (5 ≤ x ∧ x ≤ 20) → (Even y) → False :=
by sorry

end band_row_lengths_l219_219055


namespace exists_positive_integer_solution_l219_219451

theorem exists_positive_integer_solution (m : ℕ) (hm : 0 < m) :
  ∃ n : ℕ, 0 < n ∧ n / m = ⌊(n^2 : ℝ)^(1/3)⌋ + ⌊(n : ℝ)^(1/2)⌋ + 1 := 
by
  sorry

end exists_positive_integer_solution_l219_219451


namespace inversion_properties_l219_219596

open EuclideanGeometry

variables {A B C H H_A H_B H_C : Point}

-- Conditions
axiom triangle_ABC : is_triangle A B C
axiom altitudes_feet : feet_of_altitudes A B C H_A H_B H_C
axiom orthocenter : orthocenter A B C H
axiom cyclic_quadrilateral_A : cyclic_quad C H_B H_C B
axiom cyclic_quadrilateral_B : cyclic_quad C H_B H H_A

-- Theorem we want to prove
theorem inversion_properties :
  (inversion_center_A : by inversion_center A B H_C)
  (inversion_center_H : by inversion_center H A H_A) : 
  inversion_center_A  ↔ B ↔ H_C ∧ 
  inversion_center_H ↔ A ↔ H_A :=
sorry

end inversion_properties_l219_219596


namespace football_team_total_players_l219_219032

/-- The conditions are:
1. There are some players on a football team.
2. 46 are throwers.
3. All throwers are right-handed.
4. One third of the rest of the team are left-handed.
5. There are 62 right-handed players in total.
And we need to prove that the total number of players on the football team is 70. 
--/

theorem football_team_total_players (P : ℕ) 
  (h_throwers : P >= 46) 
  (h_total_right_handed : 62 = 46 + 2 * (P - 46) / 3)
  (h_remainder_left_handed : 1 * (P - 46) / 3 = (P - 46) / 3) :
  P = 70 :=
by
  sorry

end football_team_total_players_l219_219032


namespace root_of_linear_eq_l219_219039

variable (a b : ℚ) -- Using rationals for coefficients

-- Define the linear equation
def linear_eq (x : ℚ) : Prop := a * x + b = 0

-- Define the root function
def root_function : ℚ := -b / a

-- State the goal
theorem root_of_linear_eq : linear_eq a b (root_function a b) :=
by
  unfold linear_eq
  unfold root_function
  sorry

end root_of_linear_eq_l219_219039


namespace curvature_at_2_radius_of_curvature_at_2_center_of_curvature_at_2_l219_219559

noncomputable def y := λ x : ℝ, 4 / x
noncomputable def curvature (x : ℝ) :=
  abs (8 / x^3) / (1 + (-4 / x^2)^2)^(3/2)
noncomputable def radius_of_curvature (x : ℝ) :=
  ((1 + (-4 / x^2)^2)^(3/2)) / (8 / x^3)
noncomputable def center_of_curvature_x (x : ℝ) :=
  x - (-4 / x^2 * (1 + (-4 / x^2)^2)) / (8 / x^3)
noncomputable def center_of_curvature_y (x : ℝ) :=
  y x + (1 + (-4 / x^2)^2) / (8 / x^3)

theorem curvature_at_2 : curvature 2 = Real.sqrt 2 / 4 :=
by
  sorry

theorem radius_of_curvature_at_2 : radius_of_curvature 2 = 2 * Real.sqrt 2 :=
by
  sorry

theorem center_of_curvature_at_2 : center_of_curvature_x 2 = 4 ∧ center_of_curvature_y 2 = 4 :=
by
  sorry

end curvature_at_2_radius_of_curvature_at_2_center_of_curvature_at_2_l219_219559


namespace value_of_x_l219_219290

-- Let a and b be real numbers.
variable (a b : ℝ)

-- Given conditions
def cond_1 : 10 * a = 6 * b := sorry
def cond_2 : 120 * a * b = 800 := sorry

theorem value_of_x (x : ℝ) (h1 : 10 * a = x) (h2 : 6 * b = x) (h3 : 120 * a * b = 800) : x = 20 :=
sorry

end value_of_x_l219_219290


namespace ricky_time_difference_l219_219006

noncomputable def old_man_time_per_mile : ℚ := 300 / 8
noncomputable def young_man_time_per_mile : ℚ := 160 / 12
noncomputable def time_difference : ℚ := old_man_time_per_mile - young_man_time_per_mile

theorem ricky_time_difference :
  time_difference = 24 := by
sorry

end ricky_time_difference_l219_219006


namespace buffalo_weight_rounding_l219_219788

theorem buffalo_weight_rounding
  (weight_kg : ℝ) (conversion_factor : ℝ) (expected_weight_lb : ℕ) :
  weight_kg = 850 →
  conversion_factor = 0.454 →
  expected_weight_lb = 1872 →
  Nat.floor (weight_kg / conversion_factor + 0.5) = expected_weight_lb :=
by
  intro h1 h2 h3
  sorry

end buffalo_weight_rounding_l219_219788


namespace equation_of_line_l219_219965

theorem equation_of_line {x y : ℝ} (b : ℝ) (h1 : ∀ x y, (3 * x + 4 * y - 7 = 0) → (y = -3/4 * x))
  (h2 : (1 / 2) * |b| * |(4 / 3) * b| = 24) : 
  ∃ b : ℝ, ∀ x, y = -3/4 * x + b := 
sorry

end equation_of_line_l219_219965


namespace square_distance_l219_219013

theorem square_distance (a b c d e f: ℝ) 
  (side_length : ℝ)
  (AB : a = 0 ∧ b = side_length)
  (BC : c = side_length ∧ d = 0)
  (BE_dist : (a - b)^2 + (b - b)^2 = 25)
  (AE_dist : a^2 + (c - b)^2 = 144)
  (DF_dist : (d)^2 + (d)^2 = 25)
  (CF_dist : (d - c)^2 + e^2 = 144) :
  (f - d)^2 + (e - a)^2 = 578 :=
by
  -- Required to bypass the proof steps
  sorry

end square_distance_l219_219013


namespace price_reduction_l219_219723

theorem price_reduction (p0 p1 p2 : ℝ) (H0 : p0 = 1) (H1 : p1 = 1.25 * p0) (H2 : p2 = 1.1 * p0) :
  ∃ x : ℝ, p2 = p1 * (1 - x / 100) ∧ x = 12 :=
  sorry

end price_reduction_l219_219723


namespace quoted_value_stock_l219_219660

-- Define the conditions
def face_value : ℕ := 100
def dividend_percentage : ℝ := 0.14
def yield_percentage : ℝ := 0.1

-- Define the computed dividend per share
def dividend_per_share : ℝ := dividend_percentage * face_value

-- State the theorem to prove the quoted value
theorem quoted_value_stock : (dividend_per_share / yield_percentage) * 100 = 140 :=
by
  sorry  -- Placeholder for the proof

end quoted_value_stock_l219_219660


namespace original_price_of_shirts_l219_219138

theorem original_price_of_shirts 
  (sale_price : ℝ) 
  (fraction_of_original : ℝ) 
  (original_price : ℝ) 
  (h1 : sale_price = 6) 
  (h2 : fraction_of_original = 0.25) 
  (h3 : sale_price = fraction_of_original * original_price) 
  : original_price = 24 := 
by 
  sorry

end original_price_of_shirts_l219_219138


namespace necessary_but_not_sufficient_l219_219083

-- Definitions used in the conditions
variable (a b : ℝ)

-- The Lean 4 theorem statement for the proof problem
theorem necessary_but_not_sufficient : (a > b - 1) ∧ ¬ (a > b) ↔ a > b := 
sorry

end necessary_but_not_sufficient_l219_219083


namespace largest_possible_distance_between_spheres_l219_219649

noncomputable def largest_distance_between_spheres : ℝ :=
  110 + Real.sqrt 1818

theorem largest_possible_distance_between_spheres :
  let center1 := (3, -5, 7)
  let radius1 := 15
  let center2 := (-10, 20, -25)
  let radius2 := 95
  ∀ A B : ℝ × ℝ × ℝ,
    (dist A center1 = radius1) →
    (dist B center2 = radius2) →
    dist A B ≤ largest_distance_between_spheres :=
  sorry

end largest_possible_distance_between_spheres_l219_219649


namespace commutative_l219_219647

variable (R : Type) [NonAssocRing R]
variable (star : R → R → R)

axiom assoc : ∀ x y z : R, star (star x y) z = star x (star y z)
axiom comm_left : ∀ x y z : R, star (star x y) z = star (star y z) x
axiom distinct : ∀ {x y : R}, x ≠ y → ∃ z : R, star z x ≠ star z y

theorem commutative (x y : R) : star x y = star y x := sorry

end commutative_l219_219647


namespace real_part_sum_l219_219268

-- Definitions of a and b as real numbers and i as the imaginary unit
variables (a b : ℝ)
def i := Complex.I

-- Condition given in the problem
def given_condition : Prop := (a + b * i) / (2 - i) = 3 + i

-- Statement to prove
theorem real_part_sum : given_condition a b → a + b = 20 := by
  sorry

end real_part_sum_l219_219268


namespace quadratic_roots_problem_l219_219774

theorem quadratic_roots_problem 
  (x y : ℤ) 
  (h1 : x + y = 10)
  (h2 : |x - y| = 12) :
  (x - 11) * (x + 1) = 0 :=
sorry

end quadratic_roots_problem_l219_219774


namespace probability_trial_ends_first_draw_adjusted_probability_given_white_higher_probability_scenario_l219_219686

/-- The initial problem setup -/
def bagA : List (Ball) := [red, red, red, red, red, red, red, red, red, white]
def bagB : List (Ball) := [red, red, white, white, white, white, white, white, white, white]

structure Experiment :=
  (bags : List (List Ball))
  (p_chosen_bag : Real)
  (p_red_ball_given_bag : List Real)
  (p_white_ball_given_bag : List Real)
  (prior_A : Real := 0.5)
  (prior_B : Real := 0.5)

noncomputable def example_expr : Experiment :=
  { bags := [bagA, bagB],
    p_chosen_bag := 0.5,
    p_red_ball_given_bag := [9/10, 2/10],
    p_white_ball_given_bag := [1/10, 8/10],
  }

-- The problem statements to prove:
theorem probability_trial_ends_first_draw (e : Experiment) : 
  e.p_chosen_bag * e.p_red_ball_given_bag[0] + e.p_chosen_bag * e.p_red_ball_given_bag[1] = 11 / 20 :=
by sorry

theorem adjusted_probability_given_white (e : Experiment) : 
  (e.p_white_ball_given_bag[0] * e.p_chosen_bag) / ((e.p_white_ball_given_bag[0] * e.p_chosen_bag) + (e.p_white_ball_given_bag[1] * e.p_chosen_bag)) = 1 / 9 :=
by sorry

theorem higher_probability_scenario (e : Experiment) :
  let P1 := (1 / 9 * 9 / 10) + (8 / 9 * 2 / 10) in 
  let P2 := (8 / 9 * 9 / 10) + (1 / 9 * 2 / 10) in 
  P2 > P1 :=
by sorry

end probability_trial_ends_first_draw_adjusted_probability_given_white_higher_probability_scenario_l219_219686


namespace inner_tetrahedron_volume_ratio_l219_219226

noncomputable def volume_ratio_of_tetrahedrons (s : ℝ) : ℝ :=
  let V_original := (s^3 * Real.sqrt 2) / 12
  let a := (Real.sqrt 6 / 9) * s
  let V_inner := (a^3 * Real.sqrt 2) / 12
  V_inner / V_original

theorem inner_tetrahedron_volume_ratio {s : ℝ} (hs : s > 0) : volume_ratio_of_tetrahedrons s = 1 / 243 :=
by
  sorry

end inner_tetrahedron_volume_ratio_l219_219226


namespace min_max_f_l219_219470

noncomputable def f (x : ℝ) : ℝ := Real.cos x + (x + 1) * Real.sin x + 1

theorem min_max_f : 
  let min_val := - (3 * Real.pi) / 2 in
  let max_val := (Real.pi / 2) + 2 in
  ∃ x_min ∈ Set.Icc 0 (2 * Real.pi), f x_min = min_val ∧
  ∃ x_max ∈ Set.Icc 0 (2 * Real.pi), f x_max = max_val :=
sorry

end min_max_f_l219_219470


namespace sum_of_tangency_points_l219_219247

-- Define the function f(x)
def f(x : ℝ) : ℝ := max (-7 * x - 50) (max (2 * x - 2) (6 * x + 4))

-- Assume that p(x) is a quadratic polynomial tangent to f at three distinct points
-- Define x_1, x_2, x_3 to be the x-coordinates of points of tangency
-- We need to prove that x_1 + x_2 + x_3 = -4.5 under these conditions

theorem sum_of_tangency_points (x1 x2 x3 : ℝ)
  (h1 : ∃ a b c, is_quadratic (λ x, a * x^2 + b * x + c) ∧
                    is_tangent (λ x, a * x^2 + b * x + c) f x1 ∧
                    is_tangent (λ x, a * x^2 + b * x + c) f x2 ∧
                    is_tangent (λ x, a * x^2 + b * x + c) f x3 ∧
                    x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3) :
  x1 + x2 + x3 = -4.5 := sorry

end sum_of_tangency_points_l219_219247


namespace predict_sales_amount_l219_219096

theorem predict_sales_amount :
  let x_data := [2, 4, 5, 6, 8]
  let y_data := [30, 40, 50, 60, 70]
  let b := 7
  let x := 10 -- corresponding to 10,000 yuan investment
  let a := 15 -- \hat{a} calculated from the regression equation and data points
  let regression (x : ℝ) := b * x + a
  regression x = 85 :=
by
  -- Proof skipped
  sorry

end predict_sales_amount_l219_219096


namespace geometric_sequence_condition_l219_219636

-- Definition of a geometric sequence
def is_geometric_sequence (x y z : ℤ) : Prop :=
  y ^ 2 = x * z

-- Lean 4 statement based on the condition and correct answer tuple
theorem geometric_sequence_condition (a : ℤ) :
  is_geometric_sequence 4 a 9 ↔ (a = 6 ∨ a = -6) :=
by 
  sorry

end geometric_sequence_condition_l219_219636


namespace salary_reduction_l219_219174

theorem salary_reduction (S : ℝ) (x : ℝ) 
  (H1 : S > 0) 
  (H2 : 1.25 * S * (1 - 0.01 * x) = 1.0625 * S) : 
  x = 15 := 
  sorry

end salary_reduction_l219_219174


namespace circle_symmetric_line_l219_219628

theorem circle_symmetric_line (a b : ℝ) (h : a < 2) (hb : b = -2) : a + b < 0 := by
  sorry

end circle_symmetric_line_l219_219628


namespace exponent_reciprocal_evaluation_l219_219518

theorem exponent_reciprocal_evaluation : (4⁻¹ - 3⁻¹)⁻¹ = -12 := by
  have h_4 := (show 4⁻¹ = 1 / 4 by norm_num)
  have h_3 := (show 3⁻¹ = 1 / 3 by norm_num)
  rw [h_4, h_3]
  have h := (show (1 / 4 - 1 / 3)⁻¹ = -12 by sorry)
  exact h

end exponent_reciprocal_evaluation_l219_219518


namespace parallelepiped_intersection_l219_219086

/-- Given a parallelepiped A B C D A₁ B₁ C₁ D₁.
    Point X is chosen on edge A₁ D₁, and point Y is chosen on edge B C.
    It is known that A₁ X = 5, B Y = 3, and B₁ C₁ = 14.
    The plane C₁ X Y intersects ray D A at point Z.
    Prove that D Z = 20. -/
theorem parallelepiped_intersection
  (A B C D A₁ B₁ C₁ D₁ X Y Z : ℝ)
  (h₁: A₁ - X = 5)
  (h₂: B - Y = 3)
  (h₃: B₁ - C₁ = 14) :
  D - Z = 20 :=
sorry

end parallelepiped_intersection_l219_219086


namespace gcd_polynomial_multiple_l219_219401

theorem gcd_polynomial_multiple (b : ℤ) (h : b % 2373 = 0) : Int.gcd (b^2 + 13 * b + 40) (b + 5) = 5 :=
by
  sorry

end gcd_polynomial_multiple_l219_219401


namespace train_length_l219_219655

theorem train_length
  (speed_km_hr : ℕ)
  (time_sec : ℕ)
  (length_train : ℕ)
  (length_platform : ℕ)
  (h_eq_len : length_train = length_platform)
  (h_speed : speed_km_hr = 108)
  (h_time : time_sec = 60) :
  length_train = 900 :=
by
  sorry

end train_length_l219_219655


namespace height_of_parabolic_arch_l219_219936

theorem height_of_parabolic_arch (a : ℝ) (x : ℝ) (k : ℝ) (h : ℝ) (s : ℝ) :
  k = 20 →
  s = 30 →
  a = - 4 / 45 →
  x = 3 →
  k = h →
  y = a * x^2 + k →
  h = 20 → 
  y = 19.2 :=
by
  -- Given the conditions, we'll prove using provided Lean constructs
  sorry

end height_of_parabolic_arch_l219_219936


namespace greatest_int_less_than_200_gcd_30_is_5_l219_219502

theorem greatest_int_less_than_200_gcd_30_is_5 : ∃ n, n < 200 ∧ gcd n 30 = 5 ∧ ∀ m, m < 200 ∧ gcd m 30 = 5 → m ≤ n :=
by
  sorry

end greatest_int_less_than_200_gcd_30_is_5_l219_219502


namespace min_max_f_l219_219467

noncomputable def f (x : ℝ) : ℝ := Math.cos x + (x + 1) * Math.sin x + 1

theorem min_max_f :
  ∃ a b : ℝ, a = -3 * Real.pi / 2 ∧ b = Real.pi / 2 + 2 ∧
  (∀ x ∈ Set.Icc 0 (2 * Real.pi), f x ≥ a) ∧ 
  (∃ y ∈ Set.Icc 0 (2 * Real.pi), f y = a) ∧
  (∀ x ∈ Set.Icc 0 (2 * Real.pi), f x ≤ b) ∧ 
  (∃ z ∈ Set.Icc 0 (2 * Real.pi), f z = b) :=
sorry

end min_max_f_l219_219467


namespace polynomial_divisibility_l219_219289

theorem polynomial_divisibility (p q : ℝ) :
    (∀ x, x = -2 ∨ x = 3 → (x^6 - x^5 + x^4 - p*x^3 + q*x^2 - 7*x - 35) = 0) →
    (p, q) = (6.86, -36.21) :=
by
  sorry

end polynomial_divisibility_l219_219289


namespace min_value_fraction_expression_l219_219275

theorem min_value_fraction_expression {a b : ℝ} (ha : a > 0) (hb : b > 0) (h : a + b = 1) : 
  (1 / a^2 - 1) * (1 / b^2 - 1) ≥ 9 := 
by
  sorry

end min_value_fraction_expression_l219_219275


namespace student_distribution_l219_219381

-- Definition to check the number of ways to distribute 7 students into two dormitories A and B
-- with each dormitory having at least 2 students equals 56.
theorem student_distribution (students dorms : Nat) (min_students : Nat) (dist_plans : Nat) :
  students = 7 → dorms = 2 → min_students = 2 → dist_plans = 56 → 
  true := sorry

end student_distribution_l219_219381


namespace graph_of_y_eq_neg2x_passes_quadrant_II_IV_l219_219894

-- Definitions
def linear_function (k : ℝ) (x : ℝ) : ℝ := k * x

def is_in_quadrant_II (x y : ℝ) : Prop := x < 0 ∧ y > 0

def is_in_quadrant_IV (x y : ℝ) : Prop := x > 0 ∧ y < 0

-- The main statement
theorem graph_of_y_eq_neg2x_passes_quadrant_II_IV :
  ∀ (x : ℝ), (is_in_quadrant_II x (linear_function (-2) x) ∨ 
               is_in_quadrant_IV x (linear_function (-2) x)) :=
by
  sorry

end graph_of_y_eq_neg2x_passes_quadrant_II_IV_l219_219894


namespace expression_value_l219_219153

theorem expression_value (a : ℝ) (h_nonzero : a ≠ 0) (h_ne_two : a ≠ 2) (h_ne_neg_two : a ≠ -2) (h_ne_neg_one : a ≠ -1) (h_eq_one : a = 1) :
  1 - (((a-2)/a) / ((a^2-4)/(a^2+a))) = 1 / 3 :=
by
  sorry

end expression_value_l219_219153


namespace street_trees_one_side_number_of_street_trees_l219_219047

-- Conditions
def road_length : ℕ := 2575
def interval : ℕ := 25
def trees_at_endpoints : ℕ := 2

-- Question: number of street trees on one side of the road
theorem street_trees_one_side (road_length interval : ℕ) (trees_at_endpoints : ℕ) : ℕ :=
  (road_length / interval) + 1

-- Proof of the provided problem
theorem number_of_street_trees : street_trees_one_side road_length interval trees_at_endpoints = 104 :=
by
  sorry

end street_trees_one_side_number_of_street_trees_l219_219047


namespace difference_of_numbers_l219_219478

theorem difference_of_numbers (a b : ℕ) (h1 : a + b = 12390) (h2 : b = 2 * a + 18) : b - a = 4142 :=
by {
  sorry
}

end difference_of_numbers_l219_219478


namespace calculate_expression_l219_219236

theorem calculate_expression:
  202.2 * 89.8 - 20.22 * 186 + 2.022 * 3570 - 0.2022 * 16900 = 18198 :=
by
  sorry

end calculate_expression_l219_219236


namespace find_constant_c_l219_219516

def f: ℝ → ℝ := sorry

noncomputable def constant_c := 8

theorem find_constant_c (h : ∀ x : ℝ, f x + 3 * f (constant_c - x) = x) (h2 : f 2 = 2) : 
  constant_c = 8 :=
sorry

end find_constant_c_l219_219516


namespace simplify_and_evaluate_l219_219322

noncomputable def a := 3

theorem simplify_and_evaluate : (a^2 / (a + 1) - 1 / (a + 1)) = 2 := by
  sorry

end simplify_and_evaluate_l219_219322


namespace total_widgets_sold_after_20_days_l219_219000

-- Definition of the arithmetic sequence
def widgets_sold_on_day (n : ℕ) : ℕ :=
  2 * n - 1

-- Sum of the first n terms of the sequence
def sum_of_widgets_sold (n : ℕ) : ℕ :=
  n * (widgets_sold_on_day 1 + widgets_sold_on_day n) / 2

-- Prove that the total widgets sold after 20 days is 400
theorem total_widgets_sold_after_20_days : sum_of_widgets_sold 20 = 400 :=
by
  sorry

end total_widgets_sold_after_20_days_l219_219000


namespace man_speed_in_still_water_l219_219361

noncomputable def speedInStillWater 
  (upstreamSpeedWithCurrentAndWind : ℝ)
  (downstreamSpeedWithCurrentAndWind : ℝ)
  (waterCurrentSpeed : ℝ)
  (windSpeedUpstream : ℝ) : ℝ :=
  (upstreamSpeedWithCurrentAndWind + waterCurrentSpeed + windSpeedUpstream + downstreamSpeedWithCurrentAndWind - waterCurrentSpeed + windSpeedUpstream) / 2
  
theorem man_speed_in_still_water :
  speedInStillWater 20 60 5 2.5 = 42.5 :=
  sorry

end man_speed_in_still_water_l219_219361


namespace binary_to_decimal_l219_219250

theorem binary_to_decimal : (11010 : ℕ) = 26 := by
  sorry

end binary_to_decimal_l219_219250


namespace candy_bar_cost_is_7_l219_219804

-- Define the conditions
def chocolate_cost : Nat := 3
def candy_additional_cost : Nat := 4

-- Define the expression for the cost of the candy bar
def candy_cost : Nat := chocolate_cost + candy_additional_cost

-- State the theorem to prove the cost of the candy bar
theorem candy_bar_cost_is_7 : candy_cost = 7 :=
by
  sorry

end candy_bar_cost_is_7_l219_219804


namespace age_sum_l219_219766

-- Defining the ages of Henry and Jill
def Henry_age : ℕ := 20
def Jill_age : ℕ := 13

-- The statement we need to prove
theorem age_sum : Henry_age + Jill_age = 33 := by
  -- Proof goes here
  sorry

end age_sum_l219_219766


namespace decimal_multiplication_l219_219691

theorem decimal_multiplication : (3.6 * 0.3 = 1.08) := by
  sorry

end decimal_multiplication_l219_219691


namespace quotient_is_six_l219_219893

-- Definition of the given conditions
def S : Int := 476
def remainder : Int := 15
def difference : Int := 2395

-- Definition of the larger number based on the given conditions
def L : Int := S + difference

-- The statement we need to prove
theorem quotient_is_six : (L = S * 6 + remainder) := by
  sorry

end quotient_is_six_l219_219893


namespace greatest_integer_with_gcd_30_eq_5_l219_219488

theorem greatest_integer_with_gcd_30_eq_5 :
  ∃ n : ℕ, n < 200 ∧ gcd n 30 = 5 ∧ ∀ m : ℕ, m < 200 ∧ gcd m 30 = 5 → m ≤ n :=
begin
  let n := 195,
  use n,
  split,
  { sorry }, -- Proof that n < 200
  split,
  { sorry }, -- Proof that gcd n 30 = 5
  { sorry }  -- Proof that n is the greatest integer satisfying the conditions
end

end greatest_integer_with_gcd_30_eq_5_l219_219488


namespace ladder_length_difference_l219_219868

theorem ladder_length_difference :
  ∀ (flights : ℕ) (flight_height rope ladder_total_height : ℕ),
    flights = 3 →
    flight_height = 10 →
    rope = (flights * flight_height) / 2 →
    ladder_total_height = 70 →
    ladder_total_height - (flights * flight_height + rope) = 25 →
    ladder_total_height - (flights * flight_height) - rope = 10 :=
by
  intros
  sorry

end ladder_length_difference_l219_219868


namespace num_ways_arrange_l219_219870

open Finset

def valid_combinations : Finset (Finset Nat) :=
  { {2, 5, 11, 3}, {3, 5, 6, 2}, {3, 6, 11, 5}, {5, 6, 11, 2} }

theorem num_ways_arrange : valid_combinations.card = 4 :=
  by
    sorry  -- proof of the statement

end num_ways_arrange_l219_219870


namespace hannahs_grapes_per_day_l219_219904

-- Definitions based on conditions
def oranges_per_day : ℕ := 20
def days : ℕ := 30
def total_fruits : ℕ := 1800
def total_oranges : ℕ := oranges_per_day * days

-- The math proof problem to be targeted
theorem hannahs_grapes_per_day : 
  (total_fruits - total_oranges) / days = 40 := 
by
  -- Proof to be filled in here
  sorry

end hannahs_grapes_per_day_l219_219904


namespace concentration_of_first_solution_l219_219659

theorem concentration_of_first_solution
  (C : ℝ)
  (h : 4 * (C / 100) + 0.2 = 0.36) :
  C = 4 :=
by
  sorry

end concentration_of_first_solution_l219_219659


namespace length_of_AB_l219_219861

theorem length_of_AB
  (height h : ℝ)
  (AB CD : ℝ)
  (ratio_AB_ADC : (1/2 * AB * h) / (1/2 * CD * h) = 5/4)
  (sum_AB_CD : AB + CD = 300) :
  AB = 166.67 :=
by
  -- The proof goes here.
  sorry

end length_of_AB_l219_219861


namespace valid_pair_l219_219297

-- Definitions of the animals
inductive Animal
| lion
| tiger
| leopard
| elephant

open Animal

-- Given conditions
def condition1 (selected : Animal → Prop) : Prop :=
  selected lion → selected tiger

def condition2 (selected : Animal → Prop) : Prop :=
  ¬selected leopard → ¬selected tiger

def condition3 (selected : Animal → Prop) : Prop :=
  selected leopard → ¬selected elephant

-- Main theorem to prove
theorem valid_pair (selected : Animal → Prop) (pair : Animal × Animal) :
  (pair = (tiger, leopard)) ↔ 
  (condition1 selected ∧ condition2 selected ∧ condition3 selected) :=
sorry

end valid_pair_l219_219297


namespace integral_value_l219_219975

theorem integral_value (a : ℝ) (h : -35 * a^3 = -280) : ∫ x in a..2 * Real.exp 1, 1 / x = 1 := by
  sorry

end integral_value_l219_219975


namespace simplify_fraction_l219_219010

theorem simplify_fraction (a b m n : ℕ) (h : a ≠ 0 ∧ b ≠ 0 ∧ m ≠ 0 ∧ n ≠ 0) : 
  (a^2 * b) / (m * n^2) / ((a * b) / (3 * m * n)) = 3 * a / n :=
by
  sorry

end simplify_fraction_l219_219010


namespace large_cube_side_length_l219_219866

theorem large_cube_side_length (s1 s2 s3 : ℝ) (h1 : s1 = 1) (h2 : s2 = 6) (h3 : s3 = 8) : 
  ∃ s_large : ℝ, s_large^3 = s1^3 + s2^3 + s3^3 ∧ s_large = 9 := 
by 
  use 9
  rw [h1, h2, h3]
  norm_num

end large_cube_side_length_l219_219866


namespace binomial_product_l219_219246

def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_product : binomial 10 3 * binomial 8 3 = 6720 := 
by 
  sorry

end binomial_product_l219_219246


namespace range_of_x_l219_219838

noncomputable def g (x : ℝ) : ℝ := 2^x + 2^(-x) + |x|

theorem range_of_x (x : ℝ) : g (2 * x - 1) < g 3 → -1 < x ∧ x < 2 := by
  sorry

end range_of_x_l219_219838


namespace fit_jack_apples_into_jill_basket_l219_219304

-- Conditions:
def jack_basket_full : ℕ := 12
def jack_basket_space : ℕ := 4
def jack_current_apples : ℕ := jack_basket_full - jack_basket_space
def jill_basket_capacity : ℕ := 2 * jack_basket_full

-- Proof statement:
theorem fit_jack_apples_into_jill_basket : jill_basket_capacity / jack_current_apples = 3 :=
by {
  sorry
}

end fit_jack_apples_into_jill_basket_l219_219304


namespace roots_of_polynomial_l219_219257

theorem roots_of_polynomial :
  ∀ x : ℝ, (x^3 - 3 * x^2 + 2 * x) * (x - 5) = 0 ↔ x = 0 ∨ x = 1 ∨ x = 2 ∨ x = 5 :=
by 
  sorry

end roots_of_polynomial_l219_219257


namespace greatest_divisor_450_90_l219_219775

open Nat

-- Define a condition for the set of divisors of given numbers which are less than a certain number.
def is_divisor (a : ℕ) (b : ℕ) : Prop := b % a = 0

def is_greatest_divisor (d : ℕ) (n : ℕ) (m : ℕ) (k : ℕ) : Prop :=
  is_divisor m d ∧ d < k ∧ ∀ (x : ℕ), x < k → is_divisor m x → x ≤ d

-- Define the proof problem.
theorem greatest_divisor_450_90 : is_greatest_divisor 18 450 90 30 := 
by
  sorry

end greatest_divisor_450_90_l219_219775


namespace contrapositive_statement_l219_219435

theorem contrapositive_statement (m : ℝ) (h : ¬ ∃ x : ℝ, x^2 = m) : m < 0 :=
sorry

end contrapositive_statement_l219_219435


namespace max_value_of_a_l219_219058

theorem max_value_of_a :
  ∀ (a : ℚ),
  (∀ (m : ℚ), 1/3 < m ∧ m < a →
   (∀ (x : ℤ), 0 < x ∧ x ≤ 200 →
    ¬ (∃ (y : ℤ), y = m * x + 3 ∨ y = m * x + 1))) →
  a = 68/201 :=
by
  sorry

end max_value_of_a_l219_219058


namespace largest_integer_n_neg_l219_219562

theorem largest_integer_n_neg (n : ℤ) : (n < 8 ∧ 3 < n) ∧ (n^2 - 11 * n + 24 < 0) → n ≤ 7 := by
  sorry

end largest_integer_n_neg_l219_219562


namespace arithmetic_square_root_of_9_is_3_l219_219460

-- Define the arithmetic square root property
def is_arithmetic_square_root (x : ℝ) (n : ℝ) : Prop :=
  x * x = n ∧ x ≥ 0

-- The main theorem: The arithmetic square root of 9 is 3
theorem arithmetic_square_root_of_9_is_3 : 
  is_arithmetic_square_root 3 9 :=
by
  -- This is where the proof would go, but since only the statement is required:
  sorry

end arithmetic_square_root_of_9_is_3_l219_219460


namespace total_museum_tickets_cost_l219_219207

theorem total_museum_tickets_cost (num_students num_teachers cost_student_ticket cost_teacher_ticket : ℕ) :
  num_students = 12 →
  num_teachers = 4 →
  cost_student_ticket = 1 →
  cost_teacher_ticket = 3 →
  num_students * cost_student_ticket + num_teachers * cost_teacher_ticket = 24 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end total_museum_tickets_cost_l219_219207


namespace mitchell_pencils_l219_219140

/-- Mitchell and Antonio have a combined total of 54 pencils.
Mitchell has 6 more pencils than Antonio. -/
theorem mitchell_pencils (A M : ℕ) 
  (h1 : M = A + 6)
  (h2 : M + A = 54) : M = 30 :=
by
  sorry

end mitchell_pencils_l219_219140


namespace poles_needed_to_enclose_plot_l219_219222

-- Defining the lengths of the sides
def side1 : ℕ := 15
def side2 : ℕ := 22
def side3 : ℕ := 40
def side4 : ℕ := 30
def side5 : ℕ := 18

-- Defining the distance between poles
def dist_first_three_sides : ℕ := 4
def dist_last_two_sides : ℕ := 5

-- Defining the function to calculate required poles for a side
def calculate_poles (length : ℕ) (distance : ℕ) : ℕ :=
  (length / distance) + 1

-- Total poles needed before adjustment
def total_poles_before_adjustment : ℕ :=
  calculate_poles side1 dist_first_three_sides +
  calculate_poles side2 dist_first_three_sides +
  calculate_poles side3 dist_first_three_sides +
  calculate_poles side4 dist_last_two_sides +
  calculate_poles side5 dist_last_two_sides

-- Adjustment for shared poles at corners
def total_poles : ℕ :=
  total_poles_before_adjustment - 5

-- The theorem to prove
theorem poles_needed_to_enclose_plot : total_poles = 29 := by
  sorry

end poles_needed_to_enclose_plot_l219_219222


namespace bouquet_daisies_percentage_l219_219533

theorem bouquet_daisies_percentage :
  (∀ (total white yellow white_tulips white_daisies yellow_tulips yellow_daisies : ℕ),
    total = white + yellow →
    white = 7 * total / 10 →
    yellow = total - white →
    white_tulips = white / 2 →
    white_daisies = white / 2 →
    yellow_daisies = 2 * yellow / 3 →
    yellow_tulips = yellow - yellow_daisies →
    (white_daisies + yellow_daisies) * 100 / total = 55) :=
by
  intros total white yellow white_tulips white_daisies yellow_tulips yellow_daisies h_total h_white h_yellow ht_wd hd_wd hd_yd ht_yt
  sorry

end bouquet_daisies_percentage_l219_219533


namespace find_y_intercept_l219_219025

theorem find_y_intercept (m : ℝ) (x_intercept: ℝ × ℝ) : (x_intercept.snd = 0) → (x_intercept = (-4, 0)) → m = 3 → (0, m * 4 - m * (-4)) = (0, 12) :=
by
  sorry

end find_y_intercept_l219_219025


namespace find_x_l219_219270

theorem find_x (x : ℝ) (i : ℂ) (h : i * i = -1) (h1 : (1 - i) * (Complex.ofReal x + i) = 1 + i) : x = 0 :=
by sorry

end find_x_l219_219270


namespace initial_tree_height_l219_219365

-- Definition of the problem conditions as Lean definitions.
def quadruple (x : ℕ) : ℕ := 4 * x

-- Given conditions of the problem
def final_height : ℕ := 256
def height_increase_each_year (initial_height : ℕ) : Prop :=
  quadruple (quadruple (quadruple (quadruple initial_height))) = final_height

-- The proof statement that we need to prove
theorem initial_tree_height 
  (initial_height : ℕ)
  (h : height_increase_each_year initial_height)
  : initial_height = 1 := sorry

end initial_tree_height_l219_219365


namespace tamara_is_68_inch_l219_219756

-- Defining the conditions
variables (K T : ℕ)

-- Condition 1: Tamara's height in terms of Kim's height
def tamara_height := T = 3 * K - 4

-- Condition 2: Combined height of Tamara and Kim
def combined_height := T + K = 92

-- Statement to prove: Tamara's height is 68 inches
theorem tamara_is_68_inch (h1 : tamara_height T K) (h2 : combined_height T K) : T = 68 :=
by
  sorry

end tamara_is_68_inch_l219_219756


namespace greatest_integer_with_gcf_5_l219_219490

theorem greatest_integer_with_gcf_5 :
  ∃ n, n < 200 ∧ gcd n 30 = 5 ∧ ∀ m, m < 200 ∧ gcd m 30 = 5 → m ≤ n :=
by
  sorry

end greatest_integer_with_gcf_5_l219_219490


namespace equilateral_triangle_dot_product_l219_219298

noncomputable def dot_product_sum (a b c : ℝ) := 
  a * b + b * c + c * a

theorem equilateral_triangle_dot_product 
  (A B C : ℝ) (a b c : ℝ)
  (h1 : A = 1)
  (h2 : B = 1)
  (h3 : C = 1)
  (h4 : a = 1)
  (h5 : b = 1)
  (h6 : c = 1) :
  dot_product_sum a b c = 1 / 2 :=
by 
  sorry

end equilateral_triangle_dot_product_l219_219298


namespace volume_pyramid_correct_l219_219080

noncomputable def volume_of_regular_triangular_pyramid 
  (R : ℝ) (β : ℝ) (a : ℝ) : ℝ :=
  (a^3 * (Real.tan β)) / 24

theorem volume_pyramid_correct 
  (R : ℝ) (β : ℝ) (a : ℝ) : 
  volume_of_regular_triangular_pyramid R β a = (a^3 * (Real.tan β)) / 24 :=
sorry

end volume_pyramid_correct_l219_219080


namespace abc_def_intersection_l219_219600

theorem abc_def_intersection (A B C D : ℝ × ℝ)
  (hA : A = (0, 0)) (hB : B = (2, 3)) (hC : C = (5, 4)) (hD : D = (6, 1)) :
  ∃ (p q r s : ℤ), p + q + r + s = 20 ∧ 
  (p : ℚ) / q = ((5 : ℚ) + (6 : ℚ)) / 2 ∧ 
  (r : ℚ) / s = ((4 : ℚ) + (1 : ℚ)) / 2 :=
sorry

end abc_def_intersection_l219_219600


namespace binom_18_10_l219_219692

theorem binom_18_10 :
  Nat.choose 18 10 = 43758 :=
by
  have binom_16_7 : Nat.choose 16 7 = 11440 := sorry
  have binom_16_9 : Nat.choose 16 9 = 11440 := by rw [Nat.choose_symm, binom_16_7]

  have pascal_16_8 : Nat.choose 16 8 = Nat.choose 15 8 + Nat.choose 15 7 := by rw Nat.choose_succ_succ
  have pascal_15_8 : Nat.choose 15 8 = sorry
  have pascal_17_9 : Nat.choose 17 9 = 11440 + pascal_15_8 := by rw [←binom_16_9, pascal_16_8]

  have pascal_17_10 : Nat.choose 17 10 = Nat.choose 16 10 + binom_16_9 := by rw Nat.choose_succ_succ
  have pascal_16_10 : Nat.choose 16 10 = sorry
  have pascal_17_10_final : Nat.choose 17 10 = 19448 := by exact pascal_16_10 + 11440

  show Nat.choose 18 10 = 43758 := by rw [Nat.choose_succ_succ, pascal_17_9, pascal_17_10_final]

end binom_18_10_l219_219692


namespace marble_prob_l219_219190

theorem marble_prob (a c x y p q : ℕ) (h1 : 2 * a + c = 36) 
    (h2 : (x / a) * (x / a) * (y / c) = 1 / 3) 
    (h3 : (a - x) / a * (a - x) / a * (c - y) / c = p / q) 
    (hpq_rel_prime : Nat.gcd p q = 1) : p + q = 65 := by
  sorry

end marble_prob_l219_219190


namespace product_of_consecutive_even_numbers_divisible_by_8_l219_219171

theorem product_of_consecutive_even_numbers_divisible_by_8 (n : ℤ) : 8 ∣ (2 * n * (2 * n + 2)) :=
by sorry

end product_of_consecutive_even_numbers_divisible_by_8_l219_219171


namespace find_f_sqrt2_l219_219950

noncomputable def f : ℝ → ℝ := sorry

axiom f_domain : ∀ x, x > 0 → (∃ y, f y = x ∨ y = x)

axiom f_multiplicative : ∀ x y : ℝ, x > 0 → y > 0 → f (x * y) = f x + f y
axiom f_at_8 : f 8 = 3

-- Define the problem statement
theorem find_f_sqrt2 : f (Real.sqrt 2) = 1 / 2 := sorry

end find_f_sqrt2_l219_219950


namespace meal_cost_l219_219549

theorem meal_cost (M : ℝ) (h1 : 3 * M + 15 = 45) : M = 10 :=
by
  sorry

end meal_cost_l219_219549


namespace fraction_subtraction_l219_219455

theorem fraction_subtraction (x : ℚ) : x - (1/5 : ℚ) = (3/5 : ℚ) → x = (4/5 : ℚ) :=
by
  sorry

end fraction_subtraction_l219_219455


namespace largest_n_for_quadratic_neg_l219_219564

theorem largest_n_for_quadratic_neg (n : ℤ) : n^2 - 11 * n + 24 < 0 → n ≤ 7 :=
begin
  sorry
end

end largest_n_for_quadratic_neg_l219_219564


namespace minimize_sum_of_squares_l219_219548

noncomputable def sum_of_squares (x : ℝ) : ℝ := x^2 + (18 - x)^2

theorem minimize_sum_of_squares : ∃ x : ℝ, x = 9 ∧ (18 - x) = 9 ∧ ∀ y : ℝ, sum_of_squares y ≥ sum_of_squares 9 :=
by
  sorry

end minimize_sum_of_squares_l219_219548


namespace find_fraction_l219_219110

theorem find_fraction
  (F : ℚ) (m : ℕ) 
  (h1 : F^m * (1 / 4)^2 = 1 / 10^4)
  (h2 : m = 4) : 
  F = 1 / 5 :=
by
  sorry

end find_fraction_l219_219110


namespace eq_of_plane_contains_points_l219_219696

noncomputable def plane_eq (p q r : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ × ℝ :=
  let ⟨px, py, pz⟩ := p
  let ⟨qx, qy, qz⟩ := q
  let ⟨rx, ry, rz⟩ := r
  -- Vector pq
  let pq := (qx - px, qy - py, qz - pz)
  let ⟨pqx, pqy, pqz⟩ := pq
  -- Vector pr
  let pr := (rx - px, ry - py, rz - pz)
  let ⟨prx, pry, prz⟩ := pr
  -- Normal vector via cross product
  let norm := (pqy * prz - pqz * pry, pqz * prx - pqx * prz, pqx * pry - pqy * prx)
  let ⟨nx, ny, nz⟩ := norm
  -- Use normalized normal vector (1, 2, -2)
  (1, 2, -2, -(1 * px + 2 * py + -2 * pz))

theorem eq_of_plane_contains_points : 
  plane_eq (-2, 5, -3) (2, 5, -1) (4, 3, -2) = (1, 2, -2, -14) :=
by
  sorry

end eq_of_plane_contains_points_l219_219696


namespace sum_of_cubes_eq_five_l219_219874

noncomputable def root_polynomial (a b c : ℂ) : Prop :=
  (a + b + c = 2) ∧ (a*b + b*c + c*a = 3) ∧ (a*b*c = 5)

theorem sum_of_cubes_eq_five (a b c : ℂ) (h : root_polynomial a b c) :
  a^3 + b^3 + c^3 = 5 :=
sorry

end sum_of_cubes_eq_five_l219_219874


namespace possible_values_of_quadratic_l219_219335

theorem possible_values_of_quadratic (x : ℝ) (h : x^2 - 5 * x + 4 < 0) : 10 < x^2 + 4 * x + 5 ∧ x^2 + 4 * x + 5 < 37 :=
by
  sorry

end possible_values_of_quadratic_l219_219335


namespace insects_ratio_l219_219209

theorem insects_ratio (total_insects : ℕ) (geckos : ℕ) (gecko_insects : ℕ) (lizards : ℕ)
  (H1 : geckos * gecko_insects + lizards * ((total_insects - geckos * gecko_insects) / lizards) = total_insects)
  (H2 : total_insects = 66)
  (H3 : geckos = 5)
  (H4 : gecko_insects = 6)
  (H5 : lizards = 3) :
  (total_insects - geckos * gecko_insects) / lizards / gecko_insects = 2 :=
by
  sorry

end insects_ratio_l219_219209


namespace find_pure_imaginary_solutions_l219_219193

noncomputable def poly_eq_zero (x : ℂ) : Prop :=
  x^4 - 6 * x^3 + 13 * x^2 - 42 * x - 72 = 0

noncomputable def is_imaginary (x : ℂ) : Prop :=
  x.im ≠ 0 ∧ x.re = 0

theorem find_pure_imaginary_solutions :
  ∀ x : ℂ, poly_eq_zero x ∧ is_imaginary x ↔ (x = Complex.I * Real.sqrt 7 ∨ x = -Complex.I * Real.sqrt 7) :=
by sorry

end find_pure_imaginary_solutions_l219_219193


namespace part_a_part_b_l219_219449

variable {ℝ : Type} [LinearOrderedField ℝ]

noncomputable def collinear (A B C : Point ℝ) : Prop :=
∃ (a b c : ℝ), a * A.x + b * A.y = c ∧ a * B.x + b * B.y = c ∧ a * C.x + b * C.y = c

structure Point (ℝ : Type) [LinearOrderedField ℝ] :=
(x y : ℝ)

variables
  (A B P Q R L T K S : Point ℝ)
  (l a b : set (Point ℝ))
  (hA : A ∈ l)
  (hB : B ∈ l)
  (hP : P ∈ l)
  (hA_B : A ≠ B)
  (hB_P : B ≠ P)
  (hA_P : A ≠ P)
  (ha_per : ∀ x, x ∈ a ↔ ∃ y, x = ⟨0, y⟩)
  (hb_per : ∀ x, x ∈ b ↔ ∃ y, x = ⟨1, y⟩)
  (hPQ : Q ∈ a ∧ ¬Q ∈ l ∧ Q ∈ line_through P Q)
  (hPR : R ∈ b ∧ ¬R ∈ l ∧ R ∈ line_through P R)
  (hL : L ∈ line_through A T ∧ L ∈ line_through B Q ∧ line_through A Q = a ∧ line_through B R = b)
  (hT : T ∈ line_through A T ∧ T ∈ line_through A T)
  (hS : S ∈ line_through B S ∧ S ∈ line_through B Q ∧ line_through A R = a ∧ line_through B Q = b)
  (hK : K ∈ line_through A R ∧ K ∈ line_through B S ∧ K ∈ line_through A R)

theorem part_a : collinear ℝ P T S := sorry

theorem part_b : collinear ℝ P K L := sorry

end part_a_part_b_l219_219449


namespace solution1_solution2_solution3_l219_219011

noncomputable def problem1 : Real :=
3.5 * 101

noncomputable def problem2 : Real :=
11 * 5.9 - 5.9

noncomputable def problem3 : Real :=
88 - 17.5 - 12.5

theorem solution1 : problem1 = 353.5 :=
by
  sorry

theorem solution2 : problem2 = 59 :=
by
  sorry

theorem solution3 : problem3 = 58 :=
by
  sorry

end solution1_solution2_solution3_l219_219011


namespace isosceles_triangle_l219_219429

theorem isosceles_triangle 
  (α β γ : ℝ) 
  (a b : ℝ) 
  (h_sum : a + b = (Real.tan (γ / 2)) * (a * (Real.tan α) + b * (Real.tan β)))
  (h_sum_angles : α + β + γ = π) 
  (zero_lt_γ : 0 < γ ∧ γ < π) 
  (zero_lt_α : 0 < α ∧ α < π / 2) 
  (zero_lt_β : 0 < β ∧ β < π / 2) : 
  α = β := 
sorry

end isosceles_triangle_l219_219429


namespace graveling_cost_is_correct_l219_219514

noncomputable def graveling_cost (lawn_length lawn_breadth road_width cost_per_sqm : ℝ) : ℝ :=
  let road1_area := road_width * lawn_breadth
  let road2_area := road_width * lawn_length
  let intersection_area := road_width * road_width
  let total_area := road1_area + road2_area - intersection_area
  total_area * cost_per_sqm

theorem graveling_cost_is_correct :
  graveling_cost 80 60 10 2 = 2600 := by
  sorry

end graveling_cost_is_correct_l219_219514


namespace find_second_sum_l219_219345

theorem find_second_sum (S : ℝ) (x : ℝ) (h : S = 2704 ∧ 24 * x / 100 = 15 * (S - x) / 100) : (S - x) = 1664 := 
  sorry

end find_second_sum_l219_219345


namespace integer_values_satisfying_square_root_condition_l219_219179

theorem integer_values_satisfying_square_root_condition :
  ∃ (s : Finset ℤ), s.card = 6 ∧ ∀ x ∈ s, 4 < Real.sqrt (3 * x) ∧ Real.sqrt (3 * x) < 6 := sorry

end integer_values_satisfying_square_root_condition_l219_219179


namespace cost_percentage_l219_219327

-- Define the original and new costs
def original_cost (t b : ℝ) : ℝ := t * b^4
def new_cost (t b : ℝ) : ℝ := t * (2 * b)^4

-- Define the theorem to prove the percentage relationship
theorem cost_percentage (t b : ℝ) (C R : ℝ) (h1 : C = original_cost t b) (h2 : R = new_cost t b) :
  (R / C) * 100 = 1600 :=
by sorry

end cost_percentage_l219_219327


namespace f_neg_one_value_l219_219277

theorem f_neg_one_value (f : ℝ → ℝ) (b : ℝ) (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_def : ∀ x : ℝ, 0 ≤ x → f x = 2^x + 2 * x + b) :
  f (-1) = -3 := by
sorry

end f_neg_one_value_l219_219277


namespace lcm_18_20_l219_219960

theorem lcm_18_20 : Nat.lcm 18 20 = 180 := by
  sorry

end lcm_18_20_l219_219960


namespace football_game_attendance_l219_219065

theorem football_game_attendance :
  ∃ y : ℕ, (∃ x : ℕ, x + y = 280 ∧ 60 * x + 25 * y = 14000) ∧ y = 80 :=
by
  sorry

end football_game_attendance_l219_219065


namespace distance_between_intersections_l219_219221

def ellipse_eq (x y : ℝ) : Prop := (x^2) / 9 + (y^2) / 25 = 1

def is_focus_of_ellipse (fx fy : ℝ) : Prop := (fx = 0 ∧ (fy = 4 ∨ fy = -4))

def parabola_eq (x y : ℝ) : Prop := y = x^2 / 8 + 2

theorem distance_between_intersections :
  let d := 12 * Real.sqrt 2 / 5
  ∃ x1 x2 y1 y2 : ℝ, 
    ellipse_eq x1 y1 ∧ 
    parabola_eq x1 y1 ∧
    ellipse_eq x2 y2 ∧
    parabola_eq x2 y2 ∧ 
    (x2 - x1)^2 + (y2 - y1)^2 = d^2 :=
by
  sorry

end distance_between_intersections_l219_219221


namespace gift_combinations_l219_219529

theorem gift_combinations (wrapping_paper_count ribbon_count card_count : ℕ)
  (restricted_wrapping : ℕ)
  (restricted_ribbon : ℕ)
  (total_combinations := wrapping_paper_count * ribbon_count * card_count)
  (invalid_combinations := card_count)
  (valid_combinations := total_combinations - invalid_combinations) :
  wrapping_paper_count = 10 →
  ribbon_count = 4 →
  card_count = 5 →
  restricted_wrapping = 10 →
  restricted_ribbon = 1 →
  valid_combinations = 195 :=
by
  intros
  sorry

end gift_combinations_l219_219529


namespace find_a_l219_219420

theorem find_a (a x : ℝ) 
  (h : x^2 + 3 * x + a = (x + 1) * (x + 2)) : 
  a = 2 :=
sorry

end find_a_l219_219420


namespace probability_three_pairs_six_dice_correct_l219_219808

noncomputable def probability_three_pairs_six_dice : ℚ :=
  let total_outcomes := 46656
  let successful_outcomes := 1800
  successful_outcomes / total_outcomes

theorem probability_three_pairs_six_dice_correct :
  probability_three_pairs_six_dice = 25 / 648 :=
by
  sorry

end probability_three_pairs_six_dice_correct_l219_219808


namespace sum_of_n_and_k_l219_219020

theorem sum_of_n_and_k (n k : ℕ) 
  (h1 : (n.choose k) * 3 = (n.choose (k + 1)))
  (h2 : (n.choose (k + 1)) * 2 = (n.choose (k + 2))) :
  n + k = 13 :=
by
  sorry

end sum_of_n_and_k_l219_219020


namespace problem1_problem2_l219_219350

-- Problem (1) Lean Statement
theorem problem1 (c a b : ℝ) (hc : c > a) (ha : a > b) (hb : b > 0) : 
  a / (c - a) > b / (c - b) :=
sorry

-- Problem (2) Lean Statement
theorem problem2 (x : ℝ) (hx : x > 2) : 
  ∃ (xmin : ℝ), xmin = 6 ∧ (x = 6 → (x + 16 / (x - 2)) = 10) :=
sorry

end problem1_problem2_l219_219350


namespace correct_option_a_l219_219909

theorem correct_option_a (x y a b : ℝ) : 3 * x - 2 * x = x :=
by sorry

end correct_option_a_l219_219909


namespace kitchen_length_l219_219869

-- Define the conditions
def tile_area : ℕ := 6
def kitchen_width : ℕ := 48
def number_of_tiles : ℕ := 96

-- The total area is the number of tiles times the area of each tile
def total_area : ℕ := number_of_tiles * tile_area

-- Statement to prove the length of the kitchen
theorem kitchen_length : (total_area / kitchen_width) = 12 :=
by
  sorry

end kitchen_length_l219_219869


namespace cost_of_fencing_correct_l219_219049

noncomputable def cost_of_fencing (d : ℝ) (r : ℝ) : ℝ :=
  Real.pi * d * r

theorem cost_of_fencing_correct : cost_of_fencing 30 5 = 471 :=
by
  sorry

end cost_of_fencing_correct_l219_219049


namespace total_value_of_coins_l219_219794

variable (numCoins : ℕ) (coinsValue : ℕ) 

theorem total_value_of_coins : 
  numCoins = 15 → 
  (∀ n: ℕ, n = 5 → coinsValue = 12) → 
  ∃ totalValue : ℕ, totalValue = 36 :=
  by
    sorry

end total_value_of_coins_l219_219794


namespace total_legs_on_farm_l219_219901

-- Define the number of each type of animal
def num_ducks : Nat := 6
def num_dogs : Nat := 5
def num_spiders : Nat := 3
def num_three_legged_dogs : Nat := 1

-- Define the number of legs for each type of animal
def legs_per_duck : Nat := 2
def legs_per_dog : Nat := 4
def legs_per_spider : Nat := 8
def legs_per_three_legged_dog : Nat := 3

-- Calculate the total number of legs
def total_duck_legs : Nat := num_ducks * legs_per_duck
def total_dog_legs : Nat := (num_dogs * legs_per_dog) - (num_three_legged_dogs * (legs_per_dog - legs_per_three_legged_dog))
def total_spider_legs : Nat := num_spiders * legs_per_spider

-- The total number of legs on the farm
def total_animal_legs : Nat := total_duck_legs + total_dog_legs + total_spider_legs

-- State the theorem to be proved
theorem total_legs_on_farm : total_animal_legs = 55 :=
by
  -- Assuming conditions and computing as per them
  sorry

end total_legs_on_farm_l219_219901


namespace no_triangle_with_heights_1_2_3_l219_219584

open Real

theorem no_triangle_with_heights_1_2_3 :
  ¬(∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
     ∃ (k : ℝ), k > 0 ∧ 
       a * k = 1 ∧ b * (k / 2) = 2 ∧ c * (k / 3) = 3 ∧
       (a + b > c) ∧ (a + c > b) ∧ (b + c > a)) :=
by 
  sorry

end no_triangle_with_heights_1_2_3_l219_219584


namespace original_weight_l219_219063

namespace MarbleProblem

def remainingWeightAfterCuts (w : ℝ) : ℝ :=
  w * 0.70 * 0.70 * 0.85

theorem original_weight (w : ℝ) : remainingWeightAfterCuts w = 124.95 → w = 299.94 :=
by
  intros h
  sorry

end MarbleProblem

end original_weight_l219_219063


namespace order_of_expressions_l219_219594

theorem order_of_expressions (k : ℕ) (hk : k > 4) : (k + 2) < (2 * k) ∧ (2 * k) < (k^2) ∧ (k^2) < (2^k) := by
  sorry

end order_of_expressions_l219_219594


namespace general_formula_for_an_l219_219976

-- Definitions for the first few terms of the sequence
def a1 : ℚ := 1 / 7
def a2 : ℚ := 3 / 77
def a3 : ℚ := 5 / 777

-- The sequence definition as per the identified pattern
def a_n (n : ℕ) : ℚ := (18 * n - 9) / (7 * (10^n - 1))

-- The theorem to establish that the sequence definition for general n holds given the initial terms 
theorem general_formula_for_an {n : ℕ} :
  (n = 1 → a_n n = a1) ∧
  (n = 2 → a_n n = a2) ∧ 
  (n = 3 → a_n n = a3) ∧ 
  (∀ n > 3, a_n n = (18 * n - 9) / (7 * (10^n - 1))) := 
by
  sorry

end general_formula_for_an_l219_219976


namespace exists_composite_power_sum_l219_219318

def is_composite (n : ℕ) : Prop := ∃ p q : ℕ, 1 < p ∧ 1 < q ∧ n = p * q 

theorem exists_composite_power_sum (a : ℕ) (h1 : 1 < a) (h2 : a ≤ 100) : 
  ∃ n, (n > 0) ∧ (n ≤ 6) ∧ is_composite (a ^ (2 ^ n) + 1) :=
by
  sorry

end exists_composite_power_sum_l219_219318


namespace compute_binom_product_l219_219238

/-- Definition of binomial coefficient -/
def binomial (n k : ℕ) : ℕ := n.choose k

/-- The main theorem to prove -/
theorem compute_binom_product : binomial 10 3 * binomial 8 3 = 6720 :=
by
  sorry

end compute_binom_product_l219_219238


namespace eighth_binomial_term_l219_219377

theorem eighth_binomial_term :
  let n := 10
  let a := 2 * x
  let b := 1
  let k := 7
  (Nat.choose n k) * (a ^ k) * (b ^ (n - k)) = 960 * (x ^ 3) := by
  sorry

end eighth_binomial_term_l219_219377


namespace initial_apples_count_l219_219481

theorem initial_apples_count (a b : ℕ) (h₁ : b = 13) (h₂ : b = a + 5) : a = 8 :=
by
  sorry

end initial_apples_count_l219_219481


namespace computer_price_in_2016_l219_219046

def price (p₀ : ℕ) (r : ℚ) (n : ℕ) : ℚ := p₀ * (r ^ (n / 4))

theorem computer_price_in_2016 :
  price 8100 (2/3 : ℚ) 16 = 1600 :=
by
  sorry

end computer_price_in_2016_l219_219046


namespace decomposition_x_pqr_l219_219513

-- Definitions of vectors x, p, q, r
def x : ℝ := sorry
def p : ℝ := sorry
def q : ℝ := sorry
def r : ℝ := sorry

-- The linear combination we want to prove
theorem decomposition_x_pqr : 
  (x = -1 • p + 4 • q + 3 • r) :=
sorry

end decomposition_x_pqr_l219_219513


namespace minimize_norm_l219_219971

noncomputable def vector_magnitude (v : ℝ × ℝ) : ℝ := real.sqrt (v.1 ^ 2 + v.2 ^ 2)
noncomputable def dot_product (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2

-- Given conditions
variables (a b : ℝ × ℝ)
axiom cond1 : vector_magnitude a = 2
axiom cond2 : vector_magnitude b = 1
axiom cond3 : dot_product a b = 2 * 1 * real.cos (real.pi / 3) -- cos(60°) = 1/2

-- Goal: finding the value of x that minimizes the expression
theorem minimize_norm (x : ℝ) :
  ∃ x, ∀ y, vector_magnitude (a.1 - x * b.1, a.2 - x * b.2) ≤ vector_magnitude (a.1 - y * b.1, a.2 - y * b.2) := 
sorry

end minimize_norm_l219_219971


namespace second_train_speed_l219_219037

variable (t v : ℝ)

-- Defining the first condition: 20t = vt + 55
def condition1 : Prop := 20 * t = v * t + 55

-- Defining the second condition: 20t + vt = 495
def condition2 : Prop := 20 * t + v * t = 495

-- Prove that the speed of the second train is 16 km/hr under given conditions
theorem second_train_speed : ∃ t : ℝ, condition1 t 16 ∧ condition2 t 16 := sorry

end second_train_speed_l219_219037


namespace no_solution_range_of_a_l219_219392

noncomputable def range_of_a : Set ℝ := {a | ∀ x : ℝ, ¬(abs (x - 1) + abs (x - 2) ≤ a^2 + a + 1)}

theorem no_solution_range_of_a :
  range_of_a = {a | -1 < a ∧ a < 0} :=
by
  sorry

end no_solution_range_of_a_l219_219392


namespace eq1_eq2_eq3_eq4_l219_219693

/-
  First, let's define each problem and then state the equivalency of the solutions.
  We will assume the real number type for the domain of x.
-/

-- Assume x is a real number
variable (x : ℝ)

theorem eq1 (x : ℝ) : (x - 3)^2 = 4 -> (x = 5 ∨ x = 1) := sorry

theorem eq2 (x : ℝ) : x^2 - 5 * x + 1 = 0 -> (x = (5 - Real.sqrt 21) / 2 ∨ x = (5 + Real.sqrt 21) / 2) := sorry

theorem eq3 (x : ℝ) : x * (3 * x - 2) = 2 * (3 * x - 2) -> (x = 2 / 3 ∨ x = 2) := sorry

theorem eq4 (x : ℝ) : (x + 1)^2 = 4 * (1 - x)^2 -> (x = 1 / 3 ∨ x = 3) := sorry

end eq1_eq2_eq3_eq4_l219_219693


namespace simple_interest_rate_l219_219069

/-- 
  Given conditions:
  1. Time period T is 10 years.
  2. Simple interest SI is 7/5 of the principal amount P.
  Prove that the rate percent per annum R for which the simple interest is 7/5 of the principal amount in 10 years is 14%.
-/
theorem simple_interest_rate (P : ℝ) (T : ℝ) (SI : ℝ) (R : ℝ) (hT : T = 10) (hSI : SI = (7 / 5) * P) : 
  (SI = (P * R * T) / 100) → R = 14 := 
by 
  sorry

end simple_interest_rate_l219_219069


namespace pet_store_cages_l219_219061

theorem pet_store_cages 
  (snakes parrots rabbits snake_cage_capacity parrot_cage_capacity rabbit_cage_capacity : ℕ)
  (h_snakes : snakes = 4) 
  (h_parrots : parrots = 6) 
  (h_rabbits : rabbits = 8) 
  (h_snake_cage_capacity : snake_cage_capacity = 2) 
  (h_parrot_cage_capacity : parrot_cage_capacity = 3) 
  (h_rabbit_cage_capacity : rabbit_cage_capacity = 4) 
  : (snakes / snake_cage_capacity) + (parrots / parrot_cage_capacity) + (rabbits / rabbit_cage_capacity) = 6 := 
by 
  sorry

end pet_store_cages_l219_219061


namespace binomial_product_l219_219242

open Nat

theorem binomial_product :
  (Nat.choose 10 3) * (Nat.choose 8 3) = 6720 :=
by
  sorry

end binomial_product_l219_219242


namespace largest_integer_n_l219_219561

theorem largest_integer_n (n : ℤ) :
  (n^2 - 11 * n + 24 < 0) → n ≤ 7 :=
by
  sorry

end largest_integer_n_l219_219561


namespace perimeter_of_plot_l219_219895

theorem perimeter_of_plot
  (width : ℝ) 
  (cost_per_meter : ℝ)
  (total_cost : ℝ)
  (h1 : cost_per_meter = 6.5)
  (h2 : total_cost = 1170)
  (h3 : total_cost = (2 * (width + (width + 10))) * cost_per_meter) 
  :
  (2 * ((width + 10) + width)) = 180 :=
by
  sorry

end perimeter_of_plot_l219_219895


namespace product_of_fractions_l219_219507

theorem product_of_fractions :
  (1/3 : ℚ) * (2/5) * (3/7) * (4/8) = 1/35 := 
by 
  sorry

end product_of_fractions_l219_219507


namespace best_model_is_A_l219_219854

-- Definitions of the models and their R^2 values
def ModelA_R_squared : ℝ := 0.95
def ModelB_R_squared : ℝ := 0.81
def ModelC_R_squared : ℝ := 0.50
def ModelD_R_squared : ℝ := 0.32

-- Definition stating that the best fitting model is the one with the highest R^2 value
def best_fitting_model (R_squared_A R_squared_B R_squared_C R_squared_D: ℝ) : Prop :=
  R_squared_A > R_squared_B ∧ R_squared_A > R_squared_C ∧ R_squared_A > R_squared_D

-- Proof statement
theorem best_model_is_A : best_fitting_model ModelA_R_squared ModelB_R_squared ModelC_R_squared ModelD_R_squared :=
by
  -- Skipping the proof logic
  sorry

end best_model_is_A_l219_219854


namespace candy_cases_total_l219_219164

theorem candy_cases_total
  (choco_cases lolli_cases : ℕ)
  (h1 : choco_cases = 25)
  (h2 : lolli_cases = 55) : 
  (choco_cases + lolli_cases) = 80 := by
-- The proof is omitted as requested.
sorry

end candy_cases_total_l219_219164


namespace alice_probability_same_color_l219_219353

def total_ways_to_draw : ℕ := 
  Nat.choose 9 3 * Nat.choose 6 3 * Nat.choose 3 3

def favorable_outcomes_for_alice : ℕ := 
  3 * Nat.choose 6 3 * Nat.choose 3 3

def probability_alice_same_color : ℚ := 
  favorable_outcomes_for_alice / total_ways_to_draw

theorem alice_probability_same_color : probability_alice_same_color = 1 / 28 := 
by
  -- Proof is omitted as per instructions
  sorry

end alice_probability_same_color_l219_219353


namespace continuous_function_identity_l219_219700

theorem continuous_function_identity (f : ℝ → ℝ)
  (h_cont : Continuous f)
  (h_func_eq : ∀ x y : ℝ, 2 * f (x + y) = f x * f y)
  (h_f1 : f 1 = 10) :
  ∀ x : ℝ, f x = 2 * 5^x :=
by
  sorry

end continuous_function_identity_l219_219700


namespace tan_two_beta_l219_219833

variables {α β : Real}

theorem tan_two_beta (h1 : Real.tan (α + β) = 1) (h2 : Real.tan (α - β) = 7) : Real.tan (2 * β) = -3 / 4 :=
by
  sorry

end tan_two_beta_l219_219833


namespace volume_of_polyhedron_l219_219940

theorem volume_of_polyhedron (s : ℝ) : 
  let base_area := (3 * Real.sqrt 3 / 2) * s^2
  let height := s
  let volume := (1 / 3) * base_area * height
  volume = (Real.sqrt 3 / 2) * s^3 :=
by
  let base_area := (3 * Real.sqrt 3 / 2) * s^2
  let height := s
  let volume := (1 / 3) * base_area * height
  show volume = (Real.sqrt 3 / 2) * s^3
  sorry

end volume_of_polyhedron_l219_219940


namespace total_fruit_punch_eq_21_l219_219161

def orange_punch : ℝ := 4.5
def cherry_punch := 2 * orange_punch
def apple_juice := cherry_punch - 1.5

theorem total_fruit_punch_eq_21 : orange_punch + cherry_punch + apple_juice = 21 := by 
  -- This is where the proof would go
  sorry

end total_fruit_punch_eq_21_l219_219161


namespace parrot_consumption_l219_219615

theorem parrot_consumption :
  ∀ (parakeet_daily : ℕ) (finch_daily : ℕ) (num_parakeets : ℕ) (num_parrots : ℕ) (num_finches : ℕ) (weekly_birdseed : ℕ),
    parakeet_daily = 2 →
    finch_daily = parakeet_daily / 2 →
    num_parakeets = 3 →
    num_parrots = 2 →
    num_finches = 4 →
    weekly_birdseed = 266 →
    14 = (weekly_birdseed - ((num_parakeets * parakeet_daily + num_finches * finch_daily) * 7)) / num_parrots / 7 :=
by
  intros parakeet_daily finch_daily num_parakeets num_parrots num_finches weekly_birdseed
  intros hp1 hp2 hp3 hp4 hp5 hp6
  sorry

end parrot_consumption_l219_219615


namespace yellow_balls_count_l219_219585

-- Definition of problem conditions
def initial_red_balls : ℕ := 16
def initial_blue_balls : ℕ := 2 * initial_red_balls
def red_balls_lost : ℕ := 6
def green_balls_given_away : ℕ := 7  -- This is not used in the calculations
def yellow_balls_bought : ℕ := 3 * red_balls_lost
def final_total_balls : ℕ := 74

-- Defining the total balls after all transactions
def remaining_red_balls : ℕ := initial_red_balls - red_balls_lost
def total_accounted_balls : ℕ := remaining_red_balls + initial_blue_balls + yellow_balls_bought

-- Lean statement to prove
theorem yellow_balls_count : yellow_balls_bought = 18 :=
by
  sorry

end yellow_balls_count_l219_219585


namespace robinson_crusoe_sees_multiple_colors_l219_219877

def chameleons_multiple_colors (r b v : ℕ) : Prop :=
  let d1 := (r - b) % 3
  let d2 := (b - v) % 3
  let d3 := (r - v) % 3
  -- Given initial counts and rules.
  (r = 155) ∧ (b = 49) ∧ (v = 96) ∧
  -- Translate specific steps and conditions into properties
  (d1 = 1 % 3) ∧ (d2 = 1 % 3) ∧ (d3 = 2 % 3)

noncomputable def will_see_multiple_colors : Prop :=
  chameleons_multiple_colors 155 49 96 →
  ∃ (r b v : ℕ), r + b + v = 300 ∧
  ((r % 3 = 0 ∧ b % 3 ≠ 0 ∧ v % 3 ≠ 0) ∨
   (r % 3 ≠ 0 ∧ b % 3 = 0 ∧ v % 3 ≠ 0) ∨
   (r % 3 ≠ 0 ∧ b % 3 ≠ 0 ∧ v % 3 = 0))

theorem robinson_crusoe_sees_multiple_colors : will_see_multiple_colors :=
sorry

end robinson_crusoe_sees_multiple_colors_l219_219877


namespace find_f_of_2013_l219_219836

theorem find_f_of_2013 (a α b β : ℝ) (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = a * Real.sin (Real.pi * x + α) + b * Real.cos (Real.pi * x + β)) (h4 : f 4 = 3) :
  f 2013 = -3 := 
sorry

end find_f_of_2013_l219_219836


namespace workshop_workers_transfer_l219_219540

theorem workshop_workers_transfer (w d t : ℕ) (h_w : 63 ≤ w) (h_d : d ≤ 31) 
(h_prod : 1994 = 31 * w + t * (t + 1) / 2) : 
(d = 28 ∧ t = 10) ∨ (d = 30 ∧ t = 21) := sorry

end workshop_workers_transfer_l219_219540


namespace average_writing_speed_time_to_write_10000_words_l219_219228

-- Definitions based on the problem conditions
def total_words : ℕ := 60000
def total_hours : ℝ := 90.5
def writing_speed : ℝ := 663
def words_to_write : ℕ := 10000
def writing_time : ℝ := 15.08

-- Proposition that the average writing speed is 663 words per hour
theorem average_writing_speed :
  (total_words : ℝ) / total_hours = writing_speed :=
sorry

-- Proposition that the time to write 10,000 words at the given average speed is 15.08 hours
theorem time_to_write_10000_words :
  (words_to_write : ℝ) / writing_speed = writing_time :=
sorry

end average_writing_speed_time_to_write_10000_words_l219_219228


namespace relationship_between_a_and_b_l219_219767

theorem relationship_between_a_and_b (a b : ℝ) (h₀ : a ≠ 0) (max_point : ∃ x, (x = 0 ∨ x = 1/3) ∧ (∀ y, (y = 0 ∨ y = 1/3) → (3 * a * y^2 + 2 * b * y) = 0)) : a + 2 * b = 0 :=
sorry

end relationship_between_a_and_b_l219_219767


namespace max_sum_abc_divisible_by_13_l219_219851

theorem max_sum_abc_divisible_by_13 :
  ∃ (A B C : ℕ), A ≤ 9 ∧ B ≤ 9 ∧ C ≤ 9 ∧ 13 ∣ (2000 + 100 * A + 10 * B + C) ∧ (A + B + C = 26) :=
by
  sorry

end max_sum_abc_divisible_by_13_l219_219851


namespace unique_symmetric_solutions_l219_219924

theorem unique_symmetric_solutions (a b α β : ℝ) (h_mul : α * β = a) (h_add : α + β = b) :
  ∀ (x y : ℝ), x * y = a ∧ x + y = b → (x = α ∧ y = β) ∨ (x = β ∧ y = α) :=
by
  sorry

end unique_symmetric_solutions_l219_219924


namespace smallest_b_for_quadratic_inequality_l219_219263

theorem smallest_b_for_quadratic_inequality : 
  ∃ b : ℝ, (b^2 - 16 * b + 63 ≤ 0) ∧ ∀ b' : ℝ, (b'^2 - 16 * b' + 63 ≤ 0) → b ≤ b' := sorry

end smallest_b_for_quadratic_inequality_l219_219263


namespace number_of_remaining_red_points_l219_219795

/-- 
Given a grid where the distance between any two adjacent points in a row or column is 1,
and any green point can turn points within a distance of no more than 1 into green every second.
Initial state of the grid is given. Determine the number of red points after 4 seconds.
-/
def remaining_red_points_after_4_seconds (initial_state : List (List Bool)) : Nat := 
41 -- assume this is the computed number after applying the infection rule for 4 seconds

theorem number_of_remaining_red_points (initial_state : List (List Bool)) :
  remaining_red_points_after_4_seconds initial_state = 41 := 
sorry

end number_of_remaining_red_points_l219_219795


namespace Kayla_picked_40_l219_219740

-- Definitions based on conditions transformed into Lean statements
variable (K : ℕ) -- Number of apples Kylie picked
variable (total_apples : ℕ) (fraction : ℚ)

-- Given conditions
def condition1 : Prop := total_apples = 200
def condition2 : Prop := fraction = 1 / 4
def condition3 : Prop := (K + fraction * K : ℚ) = total_apples

-- Prove that Kayla picked 40 apples
theorem Kayla_picked_40 : (fraction * K : ℕ) = 40 :=
by
  -- Transform integer conditions into real ones to work with the equation
  have int_to_rat: (K : ℚ) = K := by norm_num
  rw [int_to_rat, condition2, condition3]
  sorry

end Kayla_picked_40_l219_219740


namespace solve_exp_eq_l219_219078

theorem solve_exp_eq (x : ℝ) (h : Real.sqrt ((1 + Real.sqrt 2)^x) + Real.sqrt ((1 - Real.sqrt 2)^x) = 2) : 
  x = 0 := 
sorry

end solve_exp_eq_l219_219078


namespace solve_for_x_l219_219108

theorem solve_for_x (x : ℝ) (y : ℝ) (h1 : y = 2) (h2 : y = 1 / (4 * x + 2)) : x = -3/8 :=
by
  -- The proof will go here
  sorry

end solve_for_x_l219_219108


namespace range_of_k_l219_219822

open BigOperators

theorem range_of_k
  {f : ℝ → ℝ}
  (k : ℝ)
  (h : ∀ x : ℝ, f x = 32 * x - (k + 1) * 3^x + 2)
  (H : ∀ x : ℝ, f x > 0) :
  k < 1 /2 := 
sorry

end range_of_k_l219_219822


namespace shortest_side_of_similar_triangle_l219_219363

def Triangle (a b c : ℤ) : Prop := a^2 + b^2 = c^2
def SimilarTriangles (a b c a' b' c' : ℤ) : Prop := ∃ k : ℤ, k > 0 ∧ a' = k * a ∧ b' = k * b ∧ c' = k * c 

theorem shortest_side_of_similar_triangle (a b c a' b' c' : ℤ)
  (h₀ : Triangle 15 b 17)
  (h₁ : SimilarTriangles 15 b 17 a' b' c')
  (h₂ : c' = 51) : a' = 24 :=
by
  sorry

end shortest_side_of_similar_triangle_l219_219363


namespace sum_of_digits_of_N_l219_219131

-- Define N
def N := 10^3 + 10^4 + 10^5 + 10^6 + 10^7 + 10^8 + 10^9

-- Define function to calculate sum of digits
def sum_of_digits(n: Nat) : Nat :=
  n.digits 10 |>.sum

-- Theorem statement
theorem sum_of_digits_of_N : sum_of_digits N = 7 :=
  sorry

end sum_of_digits_of_N_l219_219131


namespace trapezoid_diagonals_l219_219757

theorem trapezoid_diagonals (AD BC : ℝ) (angle_DAB angle_BCD : ℝ)
  (hAD : AD = 8) (hBC : BC = 6) (h_angle_DAB : angle_DAB = 90)
  (h_angle_BCD : angle_BCD = 120) :
  ∃ AC BD : ℝ, AC = 4 * Real.sqrt 3 ∧ BD = 2 * Real.sqrt 19 :=
by
  sorry

end trapezoid_diagonals_l219_219757


namespace ab_value_l219_219735

theorem ab_value (a b c : ℤ) (h1 : a^2 = 16) (h2 : 2 * a * b = -40) : a * b = -20 := 
sorry

end ab_value_l219_219735


namespace exists_distinct_numbers_divisible_by_3_l219_219075

-- Define the problem in Lean with the given conditions and goal.
theorem exists_distinct_numbers_divisible_by_3 : 
  ∃ a b c d : ℕ, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  a % 3 = 0 ∧ b % 3 = 0 ∧ c % 3 = 0 ∧ d % 3 = 0 ∧
  (a + b + c) % d = 0 ∧ (a + b + d) % c = 0 ∧ (a + c + d) % b = 0 ∧ (b + c + d) % a = 0 :=
by
  sorry

end exists_distinct_numbers_divisible_by_3_l219_219075


namespace total_seeds_l219_219611

-- Define the conditions given in the problem
def morningMikeTomato := 50
def morningMikePepper := 30

def morningTedTomato := 2 * morningMikeTomato
def morningTedPepper := morningMikePepper / 2

def morningSarahTomato := morningMikeTomato + 30
def morningSarahPepper := morningMikePepper + 30

def afternoonMikeTomato := 60
def afternoonMikePepper := 40

def afternoonTedTomato := afternoonMikeTomato - 20
def afternoonTedPepper := afternoonMikePepper

def afternoonSarahTomato := morningSarahTomato + 20
def afternoonSarahPepper := morningSarahPepper + 10

-- Prove that the total number of seeds planted is 685
theorem total_seeds (total: Nat) : 
    total = (
        (morningMikeTomato + afternoonMikeTomato) + 
        (morningTedTomato + afternoonTedTomato) + 
        (morningSarahTomato + afternoonSarahTomato) +
        (morningMikePepper + afternoonMikePepper) + 
        (morningTedPepper + afternoonTedPepper) + 
        (morningSarahPepper + afternoonSarahPepper)
    ) := 
    by 
        have tomato_seeds := (
            morningMikeTomato + afternoonMikeTomato +
            morningTedTomato + afternoonTedTomato + 
            morningSarahTomato + afternoonSarahTomato
        )
        have pepper_seeds := (
            morningMikePepper + afternoonMikePepper +
            morningTedPepper + afternoonTedPepper + 
            morningSarahPepper + afternoonSarahPepper
        )
        have total_seeds := tomato_seeds + pepper_seeds
        sorry

end total_seeds_l219_219611


namespace find_f_of_1_div_8_l219_219632

noncomputable def f (x : ℝ) (a : ℝ) := (a^2 + a - 5) * Real.logb a x

theorem find_f_of_1_div_8 (a : ℝ) (hx1 : x = 1 / 8) (h1 : a > 0) (h2 : a ≠ 1) (h3 : a^2 + a - 5 = 1) :
  f x a = -3 :=
by
  sorry

end find_f_of_1_div_8_l219_219632


namespace minimum_abs_a_l219_219765

-- Given conditions as definitions
def has_integer_coeffs (a b c : ℤ) : Prop := true
def has_roots_in_range (a b c : ℤ) (x1 x2 : ℚ) : Prop :=
  x1 ≠ x2 ∧ 0 < x1 ∧ x1 < 1 ∧ 0 < x2 ∧ x2 < 1 ∧
  (a : ℚ) * x1^2 + (b : ℚ) * x1 + (c : ℚ) = 0 ∧
  (a : ℚ) * x2^2 + (b : ℚ) * x2 + (c : ℚ) = 0

-- Main statement (abstractly mentioning existence of x1, x2 such that they fulfill the polynomial conditions)
theorem minimum_abs_a (a b c : ℤ) (x1 x2 : ℚ) :
  has_integer_coeffs a b c →
  has_roots_in_range a b c x1 x2 →
  |a| ≥ 5 :=
by
  intros _ _
  sorry

end minimum_abs_a_l219_219765


namespace Jurassic_Zoo_Total_l219_219016

theorem Jurassic_Zoo_Total
  (C : ℕ) (A : ℕ)
  (h1 : C = 161)
  (h2 : 8 * A + 4 * C = 964) :
  A + C = 201 := by
  sorry

end Jurassic_Zoo_Total_l219_219016


namespace radius_of_circular_film_l219_219136

theorem radius_of_circular_film (r_canister h_canister t_film R: ℝ) 
  (V: ℝ) (h1: r_canister = 5) (h2: h_canister = 10) 
  (h3: t_film = 0.2) (h4: V = 250 * Real.pi): R = 25 * Real.sqrt 2 :=
by
  sorry

end radius_of_circular_film_l219_219136


namespace train_time_to_cross_tree_l219_219352

-- Definitions based on conditions
def length_of_train := 1200 -- in meters
def time_to_pass_platform := 150 -- in seconds
def length_of_platform := 300 -- in meters
def total_distance := length_of_train + length_of_platform
def speed_of_train := total_distance / time_to_pass_platform
def time_to_cross_tree := length_of_train / speed_of_train

-- Theorem stating the main question
theorem train_time_to_cross_tree : time_to_cross_tree = 120 := by
  sorry

end train_time_to_cross_tree_l219_219352


namespace even_iff_a_zero_max_value_f_l219_219284

noncomputable def f (x a : ℝ) : ℝ := -x^2 + |x - a| + a + 1

theorem even_iff_a_zero (a : ℝ) : (∀ x, f x a = f (-x) a) ↔ a = 0 :=
by {
  -- Proof is omitted
  sorry
}

theorem max_value_f (a : ℝ) : 
  ∃ max_val : ℝ, 
    ( 
      (-1/2 < a ∧ a ≤ 0 ∧ max_val = 5/4) ∨ 
      (0 < a ∧ a < 1/2 ∧ max_val = 5/4 + 2*a) ∨ 
      ((a ≤ -1/2 ∨ a ≥ 1/2) ∧ max_val = -a^2 + a + 1)
    ) :=
by {
  -- Proof is omitted
  sorry
}

end even_iff_a_zero_max_value_f_l219_219284


namespace train_length_proof_l219_219915

def convert_kmph_to_mps (speed_kmph : ℕ) : ℕ :=
  speed_kmph * 5 / 18

theorem train_length_proof (speed_kmph : ℕ) (platform_length_m : ℕ) (crossing_time_s : ℕ) (speed_mps : ℕ) (distance_covered_m : ℕ) (train_length_m : ℕ) :
  speed_kmph = 72 →
  platform_length_m = 270 →
  crossing_time_s = 26 →
  speed_mps = convert_kmph_to_mps speed_kmph →
  distance_covered_m = speed_mps * crossing_time_s →
  train_length_m = distance_covered_m - platform_length_m →
  train_length_m = 250 :=
by
  intros h_speed h_platform h_time h_conv h_dist h_train_length
  sorry

end train_length_proof_l219_219915


namespace symmetric_point_condition_l219_219396

theorem symmetric_point_condition (a b : ℝ) (l : ℝ → ℝ → Prop) 
  (H_line: ∀ x y, l x y ↔ x + y + 1 = 0)
  (H_symmetric: l a b ∧ l (2*(-a-1) + a) (2*(-b-1) + b))
  : a + b = -1 :=
by 
  sorry

end symmetric_point_condition_l219_219396


namespace cannot_be_2009_l219_219343

theorem cannot_be_2009 (a b c : ℕ) (h : b * 1234^2 + c * 1234 + a = c * 1234^2 + a * 1234 + b) : (b * 1^2 + c * 1 + a ≠ 2009) :=
by
  sorry

end cannot_be_2009_l219_219343


namespace min_dot_product_l219_219716

variable {α : Type}
variables {a b : α}

noncomputable def dot (x y : α) : ℝ := sorry

axiom condition (a b : α) : abs (3 * dot a b) ≤ 4

theorem min_dot_product : dot a b = -4 / 3 :=
by
  sorry

end min_dot_product_l219_219716


namespace necessarily_positive_expression_l219_219453

theorem necessarily_positive_expression
  (a b c : ℝ)
  (ha : 0 < a ∧ a < 2)
  (hb : -2 < b ∧ b < 0)
  (hc : 0 < c ∧ c < 3) :
  0 < b + 3 * b^2 := 
sorry

end necessarily_positive_expression_l219_219453


namespace simplify_expr1_simplify_expr2_l219_219323

theorem simplify_expr1 : (-4)^2023 * (-0.25)^2024 = -0.25 :=
by 
  sorry

theorem simplify_expr2 : 23 * (-4 / 11) + (-5 / 11) * 23 - 23 * (2 / 11) = -23 :=
by 
  sorry

end simplify_expr1_simplify_expr2_l219_219323


namespace mary_total_spent_l219_219607

def store1_shirt : ℝ := 13.04
def store1_jacket : ℝ := 12.27
def store2_shoes : ℝ := 44.15
def store2_dress : ℝ := 25.50
def hat_price : ℝ := 9.99
def discount : ℝ := 0.10
def store4_handbag : ℝ := 30.93
def store4_scarf : ℝ := 7.42
def sunglasses_price : ℝ := 20.75
def sales_tax : ℝ := 0.05

def store1_total : ℝ := store1_shirt + store1_jacket
def store2_total : ℝ := store2_shoes + store2_dress
def store3_total : ℝ := 
  let hat_cost := hat_price * 2
  let discount_amt := hat_cost * discount
  hat_cost - discount_amt
def store4_total : ℝ := store4_handbag + store4_scarf
def store5_total : ℝ := 
  let tax := sunglasses_price * sales_tax
  sunglasses_price + tax

def total_spent : ℝ := store1_total + store2_total + store3_total + store4_total + store5_total

theorem mary_total_spent : total_spent = 173.08 := sorry

end mary_total_spent_l219_219607


namespace cos_difference_simplification_l219_219148

theorem cos_difference_simplification :
  let x := Real.cos (20 * Real.pi / 180)
  let y := Real.cos (40 * Real.pi / 180)
  x - y = -1 / (2 * Real.sqrt 5) :=
sorry

end cos_difference_simplification_l219_219148


namespace intersection_eq_l219_219409

variable {R : Type} [linear_ordered_field R]

def setA (x : R) : Prop := x^2 - 2 * x > 0
def setB (x : R) : Prop := (x + 1) / (x - 1) ≤ 0

theorem intersection_eq : {x : R | setA x} ∩ {x : R | setB x} = {x : R | -1 ≤ x ∧ x < 0} := by
  sorry

end intersection_eq_l219_219409


namespace red_peaches_count_l219_219769

-- Definitions for the conditions
def yellow_peaches : ℕ := 11
def extra_red_peaches : ℕ := 8

-- The proof statement that the number of red peaches is 19
theorem red_peaches_count : (yellow_peaches + extra_red_peaches = 19) :=
by
  sorry

end red_peaches_count_l219_219769


namespace alice_paid_percentage_l219_219633

theorem alice_paid_percentage (SRP P : ℝ) (h1 : P = 0.60 * SRP) (h2 : P_alice = 0.60 * P) :
  (P_alice / SRP) * 100 = 36 := by
sorry

end alice_paid_percentage_l219_219633


namespace ellipse_focal_point_l219_219403

theorem ellipse_focal_point (m : ℝ) (m_pos : m > 0)
  (h : ∃ f : ℝ × ℝ, f = (1, 0) ∧ ∀ x y : ℝ, (x^2 / 4) + (y^2 / m^2) = 1 → 
    (x - 1)^2 + y^2 = (x^2 / 4) + (y^2 / m^2)) :
  m = Real.sqrt 3 := 
sorry

end ellipse_focal_point_l219_219403


namespace new_weekly_income_l219_219738

-- Define the conditions
def original_income : ℝ := 60
def raise_percentage : ℝ := 0.20

-- Define the question and the expected answer
theorem new_weekly_income : original_income * (1 + raise_percentage) = 72 := 
by
  sorry

end new_weekly_income_l219_219738


namespace find_sequence_l219_219603

noncomputable def sequence_satisfies (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = (1 / 2) * (a n + 1 / (a n))

theorem find_sequence (a : ℕ → ℝ) (S : ℕ → ℝ)
    (h_pos : ∀ n, 0 < a n)
    (h_S : sequence_satisfies a S) :
    ∀ n, a n = Real.sqrt n - Real.sqrt (n - 1) :=
sorry

end find_sequence_l219_219603


namespace exactly_two_statements_true_l219_219051

noncomputable def f : ℝ → ℝ := sorry -- Definition of f satisfying the conditions

-- Conditions
axiom functional_eq (x : ℝ) : f (x + 3/2) + f x = 0
axiom odd_function (x : ℝ) : f (- x - 3/4) = - f (x - 3/4)

-- Proof statement
theorem exactly_two_statements_true : 
  (¬(∀ (T : ℝ), T > 0 → (∀ (x : ℝ), f (x + T) = f x) → T = 3/2) ∧
   (∀ (x : ℝ), f (-x - 3/4) = - f (x - 3/4)) ∧
   (¬(∀ (x : ℝ), f x = f (-x)))) :=
sorry

end exactly_two_statements_true_l219_219051


namespace initial_weight_of_alloy_is_16_l219_219368

variable (Z C : ℝ)
variable (h1 : Z / C = 5 / 3)
variable (h2 : (Z + 8) / C = 3)
variable (A : ℝ := Z + C)

theorem initial_weight_of_alloy_is_16 (h1 : Z / C = 5 / 3) (h2 : (Z + 8) / C = 3) : A = 16 := by
  sorry

end initial_weight_of_alloy_is_16_l219_219368


namespace positive_root_condition_negative_root_condition_zero_root_condition_l219_219558

variable (a b c : ℝ)

-- Condition for a positive root
theorem positive_root_condition : 
  ((a > 0 ∧ b > c) ∨ (a < 0 ∧ b < c)) ↔ (∃ x : ℝ, x > 0 ∧ a * x = b - c) :=
sorry

-- Condition for a negative root
theorem negative_root_condition : 
  ((a > 0 ∧ b < c) ∨ (a < 0 ∧ b > c)) ↔ (∃ x : ℝ, x < 0 ∧ a * x = b - c) :=
sorry

-- Condition for a root equal to zero
theorem zero_root_condition : 
  (a ≠ 0 ∧ b = c) ↔ (∃ x : ℝ, x = 0 ∧ a * x = b - c) :=
sorry

end positive_root_condition_negative_root_condition_zero_root_condition_l219_219558


namespace number_of_ways_to_assign_students_l219_219820

open Combinatorics

noncomputable def num_ways_to_assign_students : ℕ :=
  let choose_students := Nat.choose 4 2
  let choose_university := 4
  let assign_remaining_students := 3 * 2
  choose_students * choose_university * assign_remaining_students

theorem number_of_ways_to_assign_students :
  num_ways_to_assign_students = 144 :=
by
  unfold num_ways_to_assign_students
  norm_num
  sorry

end number_of_ways_to_assign_students_l219_219820


namespace contrapositive_of_inequality_l219_219465

variable {a b c : ℝ}

theorem contrapositive_of_inequality (h : a + c ≤ b + c) : a ≤ b :=
sorry

end contrapositive_of_inequality_l219_219465


namespace mike_changed_64_tires_l219_219610

def total_tires_mike_changed (motorcycles : ℕ) (cars : ℕ) (tires_per_motorcycle : ℕ) (tires_per_car : ℕ) : ℕ :=
  motorcycles * tires_per_motorcycle + cars * tires_per_car

theorem mike_changed_64_tires :
  total_tires_mike_changed 12 10 2 4 = 64 :=
by
  sorry

end mike_changed_64_tires_l219_219610


namespace sector_area_l219_219892

theorem sector_area (θ : ℝ) (r : ℝ) (hθ : θ = (2 * Real.pi) / 3) (hr : r = Real.sqrt 3) : 
    (1/2 * r^2 * θ) = Real.pi :=
by
  sorry

end sector_area_l219_219892


namespace seth_spent_more_l219_219321

def cost_ice_cream (cartons : ℕ) (price : ℕ) := cartons * price
def cost_yogurt (cartons : ℕ) (price : ℕ) := cartons * price
def amount_spent (cost_ice : ℕ) (cost_yog : ℕ) := cost_ice - cost_yog

theorem seth_spent_more :
  amount_spent (cost_ice_cream 20 6) (cost_yogurt 2 1) = 118 := by
  sorry

end seth_spent_more_l219_219321


namespace ferris_wheel_rides_l219_219211

theorem ferris_wheel_rides :
  let people_per_20_minutes := 70
  let operation_duration_hours := 6
  let minutes_per_hour := 60
  let operation_duration_minutes := operation_duration_hours * minutes_per_hour
  let times_per_hour := minutes_per_hour / 20
  let total_people_per_hour := times_per_hour * people_per_20_minutes
  let total_people := total_people_per_hour * operation_duration_hours
  total_people = 1260 :=
by
  let people_per_20_minutes := 70
  let operation_duration_hours := 6
  let minutes_per_hour := 60
  let operation_duration_minutes := operation_duration_hours * minutes_per_hour
  let times_per_hour := minutes_per_hour / 20
  let total_people_per_hour := times_per_hour * people_per_20_minutes
  let total_people := total_people_per_hour * operation_duration_hours
  have : total_people = 1260 := by sorry
  exact this

end ferris_wheel_rides_l219_219211


namespace problem_statement_l219_219146

theorem problem_statement (n m N k : ℕ)
  (h : (n^2 + 1)^(2^k) * (44 * n^3 + 11 * n^2 + 10 * n + 2) = N^m) :
  m = 1 :=
sorry

end problem_statement_l219_219146


namespace range_of_x_plus_y_l219_219830

theorem range_of_x_plus_y (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x^2 + 2 * x * y + 4 * y^2 = 1) : 0 < x + y ∧ x + y < 1 :=
by
  sorry

end range_of_x_plus_y_l219_219830


namespace max_digit_sum_in_24_hour_format_l219_219928

theorem max_digit_sum_in_24_hour_format : 
  ∃ t : ℕ × ℕ, (0 ≤ t.fst ∧ t.fst < 24 ∧ 0 ≤ t.snd ∧ t.snd < 60 ∧ (t.fst / 10 + t.fst % 10 + t.snd / 10 + t.snd % 10 = 24)) :=
sorry

end max_digit_sum_in_24_hour_format_l219_219928


namespace gwen_points_per_bag_l219_219413

theorem gwen_points_per_bag : 
  ∀ (total_bags recycled_bags total_points_per_bag points_per_bag : ℕ),
  total_bags = 4 → 
  recycled_bags = total_bags - 2 →
  total_points_per_bag = 16 →
  points_per_bag = (total_points_per_bag / total_bags) →
  points_per_bag = 4 :=
by
  intros
  sorry

end gwen_points_per_bag_l219_219413


namespace find_ordered_triple_l219_219308

theorem find_ordered_triple
  (a b c : ℝ)
  (h1 : a > 2)
  (h2 : b > 2)
  (h3 : c > 2)
  (h4 : (a + 3) ^ 2 / (b + c - 3) + (b + 5) ^ 2 / (c + a - 5) + (c + 7) ^ 2 / (a + b - 7) = 48) :
  (a, b, c) = (7, 5, 3) :=
by {
  sorry
}

end find_ordered_triple_l219_219308


namespace length_segment_AE_l219_219008

open Real

noncomputable def AB := 4
noncomputable def radius := 2
def AC := AB
def BC := AB

def D := {
  x : ℝ, 
  y : ℝ,
  condition : y = sqrt(3) * x
}

def E := {
  x : ℝ, 
  y : ℝ,
  condition : y = sqrt(3) * (BC - x)
}

theorem length_segment_AE :
  let AE := BC - 2 * sqrt(3) in
  AE = 4 - 2 * sqrt(3) :=
by 
  sorry

end length_segment_AE_l219_219008


namespace chosen_number_is_30_l219_219537

theorem chosen_number_is_30 (x : ℤ) 
  (h1 : 8 * x - 138 = 102) : x = 30 := 
sorry

end chosen_number_is_30_l219_219537


namespace system_of_equations_inconsistent_l219_219380

theorem system_of_equations_inconsistent :
  ¬∃ (x1 x2 x3 x4 x5 : ℝ), 
    (x1 + 2 * x2 - x3 + 3 * x4 - x5 = 0) ∧ 
    (2 * x1 - x2 + 3 * x3 + x4 - x5 = -1) ∧
    (x1 - x2 + x3 + 2 * x4 = 2) ∧
    (4 * x1 + 3 * x3 + 6 * x4 - 2 * x5 = 5) := 
sorry

end system_of_equations_inconsistent_l219_219380


namespace find_a_given_inequality_1_find_a_given_inequality_2_l219_219282

-- Definitions given as conditions
def f (x a : ℝ) := a * (x - 1) / (x - 2)

-- (1) Given condition and required proof of a == 1
theorem find_a_given_inequality_1 (a : ℝ) :
  (∀ (x : ℝ), 2 < x ∧ x < 3 → f x a > 2) → a = 1 :=
by
  intros h
  sorry

-- (2) Given condition and required proof of a < 2 * sqrt 2 - 3
theorem find_a_given_inequality_2 (a : ℝ) :
  (∀ (x : ℝ), 2 < x → f x a < x - 3) → a < 2 * Real.sqrt 2 - 3 :=
by
  intros h
  sorry

end find_a_given_inequality_1_find_a_given_inequality_2_l219_219282


namespace prime_of_the_form_4x4_plus_1_l219_219781

theorem prime_of_the_form_4x4_plus_1 (x : ℤ) (p : ℤ) (h : 4 * x ^ 4 + 1 = p) (hp : Prime p) : p = 5 :=
sorry

end prime_of_the_form_4x4_plus_1_l219_219781


namespace arithmetic_sequence_property_l219_219827

variable {a : ℕ → ℝ} -- Define the arithmetic sequence
variable {S : ℕ → ℝ} -- Define the sum sequence
variable {d : ℝ} -- Define the common difference
variable {a1 : ℝ} -- Define the first term

-- Suppose the sum of the first 17 terms equals 306
axiom h1 : S 17 = 306
-- Suppose the sum of the first n terms of an arithmetic sequence formula
axiom sum_formula : ∀ n, S n = n * a1 + (n * (n - 1) / 2) * d
-- Suppose the relation between the first term, common difference and sum of the first 17 terms
axiom relation : a1 + 8 * d = 18 

theorem arithmetic_sequence_property : a 7 - (a 3) / 3 = 12 := 
by sorry

end arithmetic_sequence_property_l219_219827


namespace find_m_l219_219821

def vector (α : Type) := α × α

noncomputable def dot_product {α} [Add α] [Mul α] (a b : vector α) : α :=
a.1 * b.1 + a.2 * b.2

theorem find_m (m : ℝ) (a : vector ℝ) (b : vector ℝ) (h₁ : a = (1, 2)) (h₂ : b = (m, 1)) (h₃ : dot_product a b = 0) : 
m = -2 :=
by
  sorry

end find_m_l219_219821


namespace function_in_second_quadrant_l219_219999

theorem function_in_second_quadrant (k : ℝ) : (∀ x₁ x₂ : ℝ, x₁ < 0 → x₂ < 0 → x₁ < x₂ → (k / x₁ < k / x₂)) → (∀ x : ℝ, x < 0 → (k > 0)) :=
sorry

end function_in_second_quadrant_l219_219999


namespace ganpat_paint_time_l219_219981

theorem ganpat_paint_time (H_rate G_rate : ℝ) (together_time H_time : ℝ) (h₁ : H_time = 3)
  (h₂ : together_time = 2) (h₃ : H_rate = 1 / H_time) (h₄ : G_rate = 1 / G_time)
  (h₅ : 1/H_time + 1/G_rate = 1/together_time) : G_time = 3 := 
by 
  sorry

end ganpat_paint_time_l219_219981


namespace cream_ratio_l219_219305

theorem cream_ratio (john_coffee_initial jane_coffee_initial : ℕ)
  (john_drank john_added_cream jane_added_cream jane_drank : ℕ) :
  john_coffee_initial = 20 →
  jane_coffee_initial = 20 →
  john_drank = 3 →
  john_added_cream = 4 →
  jane_added_cream = 3 →
  jane_drank = 5 →
  john_added_cream / (jane_added_cream * 18 / (23 * 1)) = (46 / 27) := 
by
  intros
  sorry

end cream_ratio_l219_219305


namespace time_saved_by_increasing_speed_l219_219608

theorem time_saved_by_increasing_speed (d v1 v2 : ℕ) (h_v1 : v1 = 60) (h_v2 : v2 = 50) (h_d : d = 1200) : 
    d / v2 - d / v1 = 4 := 
by
  rw [h_v1, h_v2, h_d]
  have h1 : 1200 / 60 = 20 := by norm_num
  have h2 : 1200 / 50 = 24 := by norm_num
  rw [h1, h2]
  norm_num
  done

end time_saved_by_increasing_speed_l219_219608


namespace fraction_to_decimal_l219_219945

theorem fraction_to_decimal : (7 / 16 : ℚ) = 0.4375 := by
  sorry

end fraction_to_decimal_l219_219945


namespace cube_root_eval_l219_219955

noncomputable def cube_root_nested (N : ℝ) : ℝ := (N * (N * (N * (N)))) ^ (1/81)

theorem cube_root_eval (N : ℝ) (h : N > 1) : 
  cube_root_nested N = N ^ (40 / 81) := 
sorry

end cube_root_eval_l219_219955


namespace solution_l219_219432

variable (x : ℝ)
variable (friend_contribution : ℝ) (james_payment : ℝ)

def adoption_fee_problem : Prop :=
  friend_contribution = 0.25 * x ∧
  james_payment = 0.75 * x ∧
  james_payment = 150 →
  x = 200

  theorem solution : adoption_fee_problem x friend_contribution james_payment :=
  by
  unfold adoption_fee_problem
  intros
  sorry

end solution_l219_219432


namespace cost_of_art_book_l219_219366

theorem cost_of_art_book
  (total_cost m_c s_c : ℕ)
  (m_b s_b a_b : ℕ)
  (hm : m_c = 3)
  (hs : s_c = 3)
  (ht : total_cost = 30)
  (hm_books : m_b = 2)
  (hs_books : s_b = 6)
  (ha_books : a_b = 3)
  : ∃ (a_c : ℕ), a_c = 2 := 
by
  sorry

end cost_of_art_book_l219_219366


namespace cyclic_proportion_l219_219317

variable {A B C p q r : ℝ}

theorem cyclic_proportion (h1 : A / B = p) (h2 : B / C = q) (h3 : C / A = r) :
  ∃ x y z, A = x ∧ B = y ∧ C = z ∧ x / y = p ∧ y / z = q ∧ z / x = r ∧
  x = (p^2 * q / r)^(1/3:ℝ) ∧ y = (q^2 * r / p)^(1/3:ℝ) ∧ z = (r^2 * p / q)^(1/3:ℝ) :=
by sorry

end cyclic_proportion_l219_219317


namespace triangle_base_is_8_l219_219582

/- Problem Statement:
We have a square with a perimeter of 48 and a triangle with a height of 36.
We need to prove that if both the square and the triangle have the same area, then the base of the triangle (x) is 8.
-/

theorem triangle_base_is_8
  (square_perimeter : ℝ)
  (triangle_height : ℝ)
  (same_area : ℝ) :
  square_perimeter = 48 →
  triangle_height = 36 →
  same_area = (square_perimeter / 4) ^ 2 →
  same_area = (1 / 2) * x * triangle_height →
  x = 8 :=
by
  sorry

end triangle_base_is_8_l219_219582


namespace A_share_of_profit_l219_219212

section InvestmentProfit

variables (capitalA capitalB : ℕ) -- initial capitals
variables (withdrawA advanceB : ℕ) -- changes after 8 months
variables (profit : ℕ) -- total profit

def investment_months (initial : ℕ) (final : ℕ) (first_period : ℕ) (second_period : ℕ) : ℕ :=
  initial * first_period + final * second_period

def ratio (a b : ℕ) : ℚ := (a : ℚ) / (b : ℚ)

def A_share (total_profit : ℕ) (ratioA ratioB : ℚ) : ℚ :=
  (ratioA / (ratioA + ratioB)) * total_profit

theorem A_share_of_profit :
  let capitalA := 3000
  let capitalB := 4000
  let withdrawA := 1000
  let advanceB := 1000
  let profit := 756
  let A_investment_months := investment_months capitalA (capitalA - withdrawA) 8 4
  let B_investment_months := investment_months capitalB (capitalB + advanceB) 8 4
  let ratioA := ratio A_investment_months B_investment_months
  let ratioB := ratio B_investment_months A_investment_months
  A_share profit ratioA ratioB = 288 := sorry

end InvestmentProfit

end A_share_of_profit_l219_219212


namespace garden_width_l219_219062

theorem garden_width :
  ∃ w l : ℝ, (2 * l + 2 * w = 60) ∧ (l * w = 200) ∧ (l = 2 * w) ∧ (w = 10) :=
by
  sorry

end garden_width_l219_219062


namespace greatest_integer_gcf_l219_219493

theorem greatest_integer_gcf (x : ℕ) : x < 200 ∧ (gcd x 30 = 5) → x = 185 :=
by sorry

end greatest_integer_gcf_l219_219493


namespace x_is_one_if_pure_imaginary_l219_219579

theorem x_is_one_if_pure_imaginary
  (x : ℝ)
  (h1 : x^2 - 1 = 0)
  (h2 : x^2 + 3 * x + 2 ≠ 0) :
  x = 1 :=
sorry

end x_is_one_if_pure_imaginary_l219_219579


namespace sampleCandy_l219_219990

-- Define the percentage of customers who are caught
def P_caught := 0.22

-- Define the percentage of customers who sample candy and are not caught
def P_notCaught_of_Sample (x : ℝ) : ℝ := 0.08 * x

-- Using the condition that the sum of caught and not caught equals the total percentage
theorem sampleCandy (x : ℝ) (h₁ : P_caught = 0.22) (h₂ : P_notCaught_of_Sample x + P_caught = x) :
  x = 23.91 / 100 :=
by
  sorry

end sampleCandy_l219_219990


namespace translation_of_exponential_l219_219773

noncomputable def translated_function (a : ℝ × ℝ) (f : ℝ → ℝ) : ℝ → ℝ :=
  λ x => f (x - a.1) + a.2

theorem translation_of_exponential :
  translated_function (2, 3) (λ x => Real.exp x) = λ x => Real.exp (x - 2) + 3 :=
by
  sorry

end translation_of_exponential_l219_219773


namespace speedster_convertibles_l219_219914

theorem speedster_convertibles 
  (T : ℕ) 
  (h1 : T > 0)
  (h2 : 30 = (2/3 : ℚ) * T)
  (h3 : ∀ n, n = (1/3 : ℚ) * T → ∃ m, m = (4/5 : ℚ) * n) :
  ∃ m, m = 12 := 
sorry

end speedster_convertibles_l219_219914


namespace minimum_type_A_tickets_value_of_m_l219_219612

theorem minimum_type_A_tickets (x : ℕ) (h1 : x + (500 - x) = 500) (h2 : x ≥ 3 * (500 - x)) : x = 375 := by
  sorry

theorem value_of_m (m : ℕ) (h : 500 * (1 + (m + 10) / 100) * (m + 20) = 56000) : m = 50 := by
  sorry

end minimum_type_A_tickets_value_of_m_l219_219612


namespace geometric_sequence_11th_term_l219_219466

theorem geometric_sequence_11th_term (a r : ℕ) :
    a * r^4 = 3 →
    a * r^7 = 24 →
    a * r^10 = 192 := by
    sorry

end geometric_sequence_11th_term_l219_219466


namespace max_intersections_intersections_ge_n_special_case_l219_219183

variable {n m : ℕ}

-- Conditions: n points on a circumference, m and n are positive integers, relatively prime, 6 ≤ 2m < n
def valid_conditions (n m : ℕ) : Prop := Nat.gcd m n = 1 ∧ 6 ≤ 2 * m ∧ 2 * m < n

-- Maximum intersections I = (m-1)n
theorem max_intersections (h : valid_conditions n m) : ∃ I, I = (m - 1) * n :=
by
  sorry

-- Prove I ≥ n
theorem intersections_ge_n (h : valid_conditions n m) : ∃ I, I ≥ n :=
by
  sorry

-- Special case: m = 3 and n is even
theorem special_case (h : valid_conditions n 3) (hn : Even n) : ∃ I, I = n :=
by
  sorry

end max_intersections_intersections_ge_n_special_case_l219_219183


namespace increasing_interval_of_f_l219_219952

noncomputable def f (x : ℝ) : ℝ := Real.sin (x + Real.pi / 4)

theorem increasing_interval_of_f :
  ∀ x, x ∈ Set.Icc (-3 * Real.pi / 4) (Real.pi / 4) → MonotoneOn f (Set.Icc (-3 * Real.pi / 4) (Real.pi / 4)) :=
by
  sorry

end increasing_interval_of_f_l219_219952


namespace combined_seq_20th_term_l219_219175

def arithmetic_seq (a : ℕ) (d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d
def geometric_seq (g : ℕ) (r : ℕ) (n : ℕ) : ℕ := g * r^(n - 1)

theorem combined_seq_20th_term :
  let a := 3
  let d := 4
  let g := 2
  let r := 2
  let n := 20
  arithmetic_seq a d n + geometric_seq g r n = 1048655 :=
by 
  sorry

end combined_seq_20th_term_l219_219175


namespace largest_integer_n_neg_l219_219563

theorem largest_integer_n_neg (n : ℤ) : (n < 8 ∧ 3 < n) ∧ (n^2 - 11 * n + 24 < 0) → n ≤ 7 := by
  sorry

end largest_integer_n_neg_l219_219563


namespace sum_lengths_AMC_l219_219542

theorem sum_lengths_AMC : 
  let length_A := 2 * (Real.sqrt 2) + 2
  let length_M := 3 + 3 + 2 * (Real.sqrt 2)
  let length_C := 3 + 3 + 2
  length_A + length_M + length_C = 13 + 4 * (Real.sqrt 2)
  := by
  sorry

end sum_lengths_AMC_l219_219542


namespace maximize_x2y5_l219_219595

theorem maximize_x2y5 (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 50) : 
  x = 100 / 7 ∧ y = 250 / 7 :=
sorry

end maximize_x2y5_l219_219595


namespace arithmetic_sequence_sum_l219_219307

theorem arithmetic_sequence_sum (S : ℕ → ℤ) (a_1 d : ℤ) 
  (h1: S 3 = (3 * a_1) + (3 * (2 * d) / 2))
  (h2: S 7 = (7 * a_1) + (7 * (6 * d) / 2)) :
  S 5 = (5 * a_1) + (5 * (4 * d) / 2) := by
  sorry

end arithmetic_sequence_sum_l219_219307


namespace greatest_valid_number_l219_219498

-- Define the conditions
def is_valid_number (n : ℕ) : Prop :=
  n < 200 ∧ Nat.gcd n 30 = 5

-- Formulate the proof problem
theorem greatest_valid_number : ∃ n, is_valid_number n ∧ (∀ m, is_valid_number m → m ≤ n) ∧ n = 185 := 
by
  sorry

end greatest_valid_number_l219_219498


namespace solve_equation_l219_219556

theorem solve_equation :
  ∀ x : ℝ, 
    (8 / (Real.sqrt (x - 10) - 10) + 
     2 / (Real.sqrt (x - 10) - 5) + 
     9 / (Real.sqrt (x - 10) + 5) + 
     16 / (Real.sqrt (x - 10) + 10) = 0)
    ↔ 
    x = 1841 / 121 ∨ x = 190 / 9 :=
by
  sorry

end solve_equation_l219_219556


namespace inequality_neg_reciprocal_l219_219567

theorem inequality_neg_reciprocal (a b : ℝ) (h1 : a < b) (h2 : b < 0) : 
  - (1 / a) < - (1 / b) :=
sorry

end inequality_neg_reciprocal_l219_219567


namespace beach_relaxing_people_l219_219186

def row1_original := 24
def row1_got_up := 3

def row2_original := 20
def row2_got_up := 5

def row3_original := 18

def total_left_relaxing (r1o r1u r2o r2u r3o : Nat) : Nat :=
  r1o + r2o + r3o - (r1u + r2u)

theorem beach_relaxing_people : total_left_relaxing row1_original row1_got_up row2_original row2_got_up row3_original = 54 :=
by
  sorry

end beach_relaxing_people_l219_219186


namespace correct_average_is_40_point_3_l219_219890

noncomputable def incorrect_average : ℝ := 40.2
noncomputable def incorrect_total_sum : ℝ := incorrect_average * 10
noncomputable def incorrect_first_number_adjustment : ℝ := 17
noncomputable def incorrect_second_number_actual : ℝ := 31
noncomputable def incorrect_second_number_provided : ℝ := 13
noncomputable def correct_total_sum : ℝ := incorrect_total_sum - incorrect_first_number_adjustment + (incorrect_second_number_actual - incorrect_second_number_provided)
noncomputable def number_of_values : ℝ := 10

theorem correct_average_is_40_point_3 :
  correct_total_sum / number_of_values = 40.3 :=
by
  sorry

end correct_average_is_40_point_3_l219_219890


namespace total_decorations_l219_219948

theorem total_decorations 
  (skulls : ℕ) (broomsticks : ℕ) (spiderwebs : ℕ) (pumpkins : ℕ) 
  (cauldron : ℕ) (budget_decorations : ℕ) (left_decorations : ℕ)
  (h_skulls : skulls = 12)
  (h_broomsticks : broomsticks = 4)
  (h_spiderwebs : spiderwebs = 12)
  (h_pumpkins : pumpkins = 2 * spiderwebs)
  (h_cauldron : cauldron = 1)
  (h_budget_decorations : budget_decorations = 20)
  (h_left_decorations : left_decorations = 10) : 
  skulls + broomsticks + spiderwebs + pumpkins + cauldron + budget_decorations + left_decorations = 83 := 
by 
  sorry

end total_decorations_l219_219948


namespace tangent_line_through_origin_max_value_on_interval_min_value_on_interval_l219_219572
noncomputable def f (x : ℝ) : ℝ := x^2 / Real.exp x

theorem tangent_line_through_origin (x y : ℝ) :
  (∃ a : ℝ, (x, y) = (a, f a) ∧ (0, 0) = (0, 0) ∧ y - f a = ((2 * a - a^2) / Real.exp a) * (x - a)) →
  y = x / Real.exp 1 :=
sorry

theorem max_value_on_interval : ∃ (x : ℝ), x = 9 / Real.exp 3 :=
  sorry

theorem min_value_on_interval : ∃ (x : ℝ), x = 0 :=
  sorry

end tangent_line_through_origin_max_value_on_interval_min_value_on_interval_l219_219572


namespace eleven_y_minus_x_l219_219347

theorem eleven_y_minus_x (x y : ℤ) (h1 : x = 7 * y + 3) (h2 : 2 * x = 18 * y + 2) : 11 * y - x = 1 := by
  sorry

end eleven_y_minus_x_l219_219347


namespace isosceles_right_triangle_contains_probability_l219_219997

noncomputable def isosceles_right_triangle_probability : ℝ :=
  let leg_length := 2
  let triangle_area := (leg_length * leg_length) / 2
  let distance_radius := 1
  let quarter_circle_area := (Real.pi * (distance_radius * distance_radius)) / 4
  quarter_circle_area / triangle_area

theorem isosceles_right_triangle_contains_probability :
  isosceles_right_triangle_probability = (Real.pi / 8) :=
by
  sorry

end isosceles_right_triangle_contains_probability_l219_219997


namespace property_value_at_beginning_l219_219792

theorem property_value_at_beginning 
  (r : ℝ) (v3 : ℝ) (V : ℝ) (rate : ℝ) (years : ℕ) 
  (h_rate : rate = 6.25 / 100) 
  (h_years : years = 3) 
  (h_v3 : v3 = 21093) 
  (h_r : r = 1 - rate) 
  (h_V : V * r ^ years = v3) 
  : V = 25656.25 :=
by
  sorry

end property_value_at_beginning_l219_219792


namespace increasing_on_positive_reals_l219_219329

noncomputable def f (x : ℝ) : ℝ := Real.exp x + Real.exp (-x)

theorem increasing_on_positive_reals : ∀ x y : ℝ, 0 < x → 0 < y → x < y → f x < f y := by
  sorry

end increasing_on_positive_reals_l219_219329


namespace triangle_angles_sum_l219_219538

theorem triangle_angles_sum (x : ℝ) (h : 40 + 3 * x + (x + 10) = 180) : x = 32.5 := by
  sorry

end triangle_angles_sum_l219_219538


namespace greatest_int_less_than_200_gcd_30_is_5_l219_219503

theorem greatest_int_less_than_200_gcd_30_is_5 : ∃ n, n < 200 ∧ gcd n 30 = 5 ∧ ∀ m, m < 200 ∧ gcd m 30 = 5 → m ≤ n :=
by
  sorry

end greatest_int_less_than_200_gcd_30_is_5_l219_219503


namespace triangle_shape_max_value_expression_l219_219863

-- Part (1): Prove that the triangle is either right or isosceles
theorem triangle_shape (A B C a b c : ℝ)
(hSides : a = 1 / sin B ∧ b = 1 / sin A)
(hTrig: sin (A - B) * cos C = cos B * sin (A - C)) : 
    (A = π / 2) ∨ (B = C) := 
sorry

-- Part (2): Prove the maximum value of the expression:
theorem max_value_expression (A B : ℝ)
(hAcuteness: A < π / 2 ∧ B < π / 2)
(hSide: a = 1 / sin B):
    ∃ (max_val : ℝ), max_val = 25 / 16 :=
sorry

end triangle_shape_max_value_expression_l219_219863


namespace sum_c_d_eq_30_l219_219544

noncomputable def c_d_sum : ℕ :=
  let c : ℕ := 28
  let d : ℕ := 2
  c + d

theorem sum_c_d_eq_30 : c_d_sum = 30 :=
by {
  sorry
}

end sum_c_d_eq_30_l219_219544


namespace expand_product_l219_219552

theorem expand_product (x : ℝ) (hx : x ≠ 0) : 
  (3 / 7) * (7 / x^3 - 14 * x^4) = 3 / x^3 - 6 * x^4 :=
by
  sorry

end expand_product_l219_219552


namespace verify_statements_l219_219742

theorem verify_statements (S : Set ℝ) (m l : ℝ) (hS : ∀ x, x ∈ S → x^2 ∈ S) :
  (m = 1 → S = {1}) ∧
  (m = -1/2 → (1/4 ≤ l ∧ l ≤ 1)) ∧
  (l = 1/2 → -Real.sqrt 2 / 2 ≤ m ∧ m ≤ 0) ∧
  (l = 1 → -1 ≤ m ∧ m ≤ 1) :=
  sorry

end verify_statements_l219_219742


namespace original_cost_price_l219_219764

theorem original_cost_price (C : ℝ) (h : C + 0.15 * C + 0.05 * C + 0.10 * C = 6400) : C = 4923 :=
by
  sorry

end original_cost_price_l219_219764


namespace find_f_2023_l219_219930

noncomputable def f : ℤ → ℤ := sorry

theorem find_f_2023 (h1 : ∀ x : ℤ, f (x+2) + f x = 3) (h2 : f 1 = 0) : f 2023 = 3 := sorry

end find_f_2023_l219_219930


namespace area_under_cos_l219_219017

theorem area_under_cos :
  ∫ x in (0 : ℝ)..(3 * Real.pi / 2), |Real.cos x| = 3 :=
by
  sorry

end area_under_cos_l219_219017


namespace repeating_decimal_product_l219_219077

noncomputable def x : ℚ := 1 / 33
noncomputable def y : ℚ := 1 / 3

theorem repeating_decimal_product :
  (x * y) = 1 / 99 :=
by
  -- Definitions of x and y
  sorry

end repeating_decimal_product_l219_219077


namespace meaningful_fraction_l219_219644

theorem meaningful_fraction (x : ℝ) : (∃ (f : ℝ), f = 2 / x) ↔ x ≠ 0 :=
by
  sorry

end meaningful_fraction_l219_219644


namespace square_angle_l219_219118

theorem square_angle (PQ QR : ℝ) (x : ℝ) (PQR_is_square : true)
  (angle_sum_of_triangle : ∀ a b c : ℝ, a + b + c = 180)
  (right_angle : ∀ a, a = 90) :
  x = 45 :=
by
  -- We start with the properties of the square (implicitly given by the conditions)
  -- Now use the conditions and provided values to conclude the proof
  sorry

end square_angle_l219_219118


namespace candy_days_l219_219816

theorem candy_days (neighbor_candy older_sister_candy candy_per_day : ℝ) 
  (h1 : neighbor_candy = 11.0) 
  (h2 : older_sister_candy = 5.0) 
  (h3 : candy_per_day = 8.0) : 
  ((neighbor_candy + older_sister_candy) / candy_per_day) = 2.0 := 
by 
  sorry

end candy_days_l219_219816


namespace largest_integer_n_l219_219560

theorem largest_integer_n (n : ℤ) :
  (n^2 - 11 * n + 24 < 0) → n ≤ 7 :=
by
  sorry

end largest_integer_n_l219_219560


namespace sqrt_of_16_l219_219027

theorem sqrt_of_16 (x : ℝ) (hx : x^2 = 16) : x = 4 ∨ x = -4 := 
by
  sorry

end sqrt_of_16_l219_219027


namespace work_completion_time_l219_219346

-- Define the constants for work rates and times
def W : ℚ := 1
def P_rate : ℚ := W / 20
def Q_rate : ℚ := W / 12
def initial_days : ℚ := 4

-- Define the amount of work done by P in the initial 4 days
def work_done_initial : ℚ := initial_days * P_rate

-- Define the remaining work after initial 4 days
def remaining_work : ℚ := W - work_done_initial

-- Define the combined work rate of P and Q
def combined_rate : ℚ := P_rate + Q_rate

-- Define the time taken to complete the remaining work
def remaining_days : ℚ := remaining_work / combined_rate

-- Define the total time taken to complete the work
def total_days : ℚ := initial_days + remaining_days

-- The theorem to prove
theorem work_completion_time :
  total_days = 10 := 
by
  -- these term can be the calculation steps
  sorry

end work_completion_time_l219_219346


namespace lottery_numbers_bound_l219_219156

theorem lottery_numbers_bound (s : ℕ) (k : ℕ) (num_tickets : ℕ) (num_numbers : ℕ) (nums_per_ticket : ℕ)
  (h_tickets : num_tickets = 100) (h_numbers : num_numbers = 90) (h_nums_per_ticket : nums_per_ticket = 5)
  (h_s : s = num_tickets) (h_k : k = 49) :
  ∃ n : ℕ, n ≤ 10 :=
by
  sorry

end lottery_numbers_bound_l219_219156


namespace points_difference_l219_219688

theorem points_difference :
  let points_td := 7
  let points_epc := 1
  let points_fg := 3
  
  let touchdowns_BG := 6
  let epc_BG := 4
  let fg_BG := 2
  
  let touchdowns_CF := 8
  let epc_CF := 6
  let fg_CF := 3
  
  let total_BG := touchdowns_BG * points_td + epc_BG * points_epc + fg_BG * points_fg
  let total_CF := touchdowns_CF * points_td + epc_CF * points_epc + fg_CF * points_fg
  
  total_CF - total_BG = 19 := by
  sorry

end points_difference_l219_219688


namespace probability_at_least_one_male_and_one_female_l219_219850

theorem probability_at_least_one_male_and_one_female 
  (M F: ℕ) (M_eq : M = 5) (F_eq : F = 2) :
  let total_choices := Nat.choose (M + F) 3,
      event1 := Nat.choose M 1 * Nat.choose F 2,
      event2 := Nat.choose M 2 * Nat.choose F 1,
      desired_event := event1 + event2 in
  (desired_event : ℚ) / total_choices = 5 / 7 :=
by sorry

end probability_at_least_one_male_and_one_female_l219_219850


namespace cost_of_headphones_l219_219071

-- Define the constants for the problem
def bus_ticket_cost : ℕ := 11
def drinks_and_snacks_cost : ℕ := 3
def wifi_cost_per_hour : ℕ := 2
def trip_hours : ℕ := 3
def earnings_per_hour : ℕ := 12
def total_earnings := earnings_per_hour * trip_hours
def total_expenses_without_headphones := bus_ticket_cost + drinks_and_snacks_cost + (wifi_cost_per_hour * trip_hours)

-- Prove the cost of headphones, H, is $16 
theorem cost_of_headphones : total_earnings = total_expenses_without_headphones + 16 := by
  -- setup the goal
  sorry

end cost_of_headphones_l219_219071


namespace fraction_of_white_roses_l219_219543

open Nat

def rows : ℕ := 10
def roses_per_row : ℕ := 20
def total_roses : ℕ := rows * roses_per_row
def red_roses : ℕ := total_roses / 2
def pink_roses : ℕ := 40
def white_roses : ℕ := total_roses - red_roses - pink_roses
def remaining_roses : ℕ := white_roses + pink_roses
def fraction_white_roses : ℚ := white_roses / remaining_roses

theorem fraction_of_white_roses :
  fraction_white_roses = 3 / 5 :=
by
  sorry

end fraction_of_white_roses_l219_219543


namespace range_of_m_l219_219823

variable (m : ℝ)

def p : Prop := (m^2 - 4 > 0) ∧ (m > 0)
def q : Prop := 16 * (m - 2)^2 - 16 < 0

theorem range_of_m :
  (p m ∨ q m) ∧ ¬(p m ∧ q m) → (1 < m ∧ m ≤ 2) ∨ (3 ≤ m) :=
by
  intro h
  sorry

end range_of_m_l219_219823


namespace geometric_increasing_condition_l219_219128

structure GeometricSequence (a₁ q : ℝ) (a : ℕ → ℝ) :=
  (rec_rel : ∀ n : ℕ, a (n + 1) = a n * q)

def is_increasing (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

theorem geometric_increasing_condition (a₁ q : ℝ) (a : ℕ → ℝ) (h : GeometricSequence a₁ q a) :
  ¬ (q > 1 ↔ is_increasing a) := sorry

end geometric_increasing_condition_l219_219128


namespace trigonometric_identity_l219_219082

theorem trigonometric_identity (x : ℝ) (h : Real.tan x = 2) : 
  (Real.cos x + Real.sin x) / (3 * Real.cos x - Real.sin x) = 3 := 
by
  sorry

end trigonometric_identity_l219_219082


namespace fruit_seller_price_l219_219528

theorem fruit_seller_price 
  (CP SP SP_profit : ℝ)
  (h1 : SP = CP * 0.88)
  (h2 : SP_profit = CP * 1.20)
  (h3 : SP_profit = 21.818181818181817) :
  SP = 16 := 
by 
  sorry

end fruit_seller_price_l219_219528


namespace min_value_of_a_and_b_l219_219269

theorem min_value_of_a_and_b (a b : ℝ) (h : a ^ 2 + 2 * b ^ 2 = 6) : 
  ∃ (m : ℝ), (∀ (x y : ℝ), x ^ 2 + 2 * y ^ 2 = 6 → x + y ≥ m) ∧ (a + b = m) :=
sorry

end min_value_of_a_and_b_l219_219269


namespace possible_permutations_100_l219_219733

def tasty_permutations (n : ℕ) : ℕ := sorry

theorem possible_permutations_100 :
  2^100 ≤ tasty_permutations 100 ∧ tasty_permutations 100 ≤ 4^100 :=
sorry

end possible_permutations_100_l219_219733


namespace remainder_n_l219_219634

-- Definitions for the conditions
/-- m is a positive integer leaving a remainder of 2 when divided by 6 -/
def m (m : ℕ) : Prop := m % 6 = 2

/-- The remainder when m - n is divided by 6 is 5 -/
def mn_remainder (m n : ℕ) : Prop := (m - n) % 6 = 5

-- Theorem statement
theorem remainder_n (m n : ℕ) (h1 : m % 6 = 2) (h2 : (m - n) % 6 = 5) (h3 : m > n) :
  n % 6 = 4 :=
by
  sorry

end remainder_n_l219_219634


namespace max_volume_range_of_a_x1_x2_inequality_l219_219117

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

noncomputable def g (a x : ℝ) : ℝ := (Real.exp (a * x^2) - Real.exp 1 * x + a * x^2 - 1) / x

theorem max_volume (x : ℝ) (hx : 1 < x) :
  ∃ V : ℝ, V = (Real.pi / 3) * ((Real.log x)^2 / x) ∧ V = (4 * Real.pi / (3 * (Real.exp 2)^2)) :=
sorry

theorem range_of_a (x1 x2 a : ℝ) (hx1 : 1 < x1) (hx2 : 1 < x2) (hx12 : x1 < x2)
  (h_eq : ∀ x > 1, f x = g a x) :
  0 < a ∧ a < (1/2) * (Real.exp 1) :=
sorry

theorem x1_x2_inequality (x1 x2 a : ℝ) (hx1 : 1 < x1) (hx2 : 1 < x2) (hx12 : x1 < x2)
  (h_eq : ∀ x > 1, f x = g a x) :
  x1^2 + x2^2 > 2 / Real.exp 1 :=
sorry

end max_volume_range_of_a_x1_x2_inequality_l219_219117


namespace distance_between_points_l219_219040

open Real

theorem distance_between_points :
  let P := (1, 3)
  let Q := (-5, 7)
  dist P Q = 2 * sqrt 13 :=
by
  let P := (1, 3)
  let Q := (-5, 7)
  sorry

end distance_between_points_l219_219040


namespace bird_families_flew_away_l219_219911

def initial_families : ℕ := 41
def left_families : ℕ := 14

theorem bird_families_flew_away :
  initial_families - left_families = 27 :=
by
  -- This is a placeholder for the proof
  sorry

end bird_families_flew_away_l219_219911


namespace original_ratio_l219_219898

theorem original_ratio (x y : ℕ) (h1 : x = y + 5) (h2 : (x - 5) / (y - 5) = 5 / 4) : x / y = 6 / 5 :=
by sorry

end original_ratio_l219_219898


namespace product_of_numbers_l219_219637

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 30) (h2 : x - y = 10) : x * y = 200 :=
sorry

end product_of_numbers_l219_219637


namespace min_max_values_f_l219_219469

noncomputable def f (x : ℝ) : ℝ :=
  Real.cos x + (x + 1) * Real.sin x + 1

theorem min_max_values_f :
  ∃ (a b : ℝ), a = -3 * Real.pi / 2 ∧ b = Real.pi / 2 + 2 ∧ 
                ∀ x ∈ Set.Icc 0 (2 * Real.pi), f x ≥ a ∧ f x ≤ b :=
by
  sorry

end min_max_values_f_l219_219469


namespace cost_option1_eq_cost_option2_eq_option1_final_cost_eq_option2_final_cost_eq_option1_more_cost_effective_optimal_cost_eq_l219_219925

section price_calculations

variables {x : ℕ} (hx : x > 20)

-- Definitions based on the problem statement.
def suit_price : ℕ := 400
def tie_price : ℕ := 80

def option1_cost (x : ℕ) : ℕ :=
  20 * suit_price + tie_price * (x - 20)

def option2_cost (x : ℕ) : ℕ :=
  (20 * suit_price + tie_price * x) * 9 / 10

def option1_final_cost := option1_cost 30
def option2_final_cost := option2_cost 30

def optimal_cost : ℕ := 20 * suit_price + tie_price * 10 * 9 / 10

-- Proof obligations
theorem cost_option1_eq : option1_cost x = 80 * x + 6400 :=
by sorry

theorem cost_option2_eq : option2_cost x = 72 * x + 7200 :=
by sorry

theorem option1_final_cost_eq : option1_final_cost = 8800 :=
by sorry

theorem option2_final_cost_eq : option2_final_cost = 9360 :=
by sorry

theorem option1_more_cost_effective : option1_final_cost < option2_final_cost :=
by sorry

theorem optimal_cost_eq : optimal_cost = 8720 :=
by sorry

end price_calculations

end cost_option1_eq_cost_option2_eq_option1_final_cost_eq_option2_final_cost_eq_option1_more_cost_effective_optimal_cost_eq_l219_219925


namespace infinite_consecutive_pairs_l219_219038

-- Define the relation
def related (x y : ℕ) : Prop :=
  ∀ d ∈ (Nat.digits 10 (x + y)), d = 0 ∨ d = 1

-- Define sets A and B
variable (A B : Set ℕ)

-- Define the conditions
axiom cond1 : ∀ a ∈ A, ∀ b ∈ B, related a b
axiom cond2 : ∀ c, (∀ a ∈ A, related c a) → c ∈ B
axiom cond3 : ∀ c, (∀ b ∈ B, related c b) → c ∈ A

-- Prove that one of the sets contains infinitely many pairs of consecutive numbers
theorem infinite_consecutive_pairs :
  (∃ a ∈ A, ∀ n : ℕ, a + n ∈ A ∧ a + n + 1 ∈ A) ∨ (∃ b ∈ B, ∀ n : ℕ, b + n ∈ B ∧ b + n + 1 ∈ B) :=
sorry

end infinite_consecutive_pairs_l219_219038


namespace nancy_total_spending_l219_219880

theorem nancy_total_spending :
  let this_month_games := 9
  let this_month_price := 5
  let last_month_games := 8
  let last_month_price := 4
  let next_month_games := 7
  let next_month_price := 6
  let total_cost := (this_month_games * this_month_price) +
                    (last_month_games * last_month_price) +
                    (next_month_games * next_month_price)
  total_cost = 119 :=
by
  sorry

end nancy_total_spending_l219_219880


namespace intersection_of_A_and_B_l219_219569

def A : Set ℝ := {x | -1 < x ∧ x < 2}
def B : Set ℝ := {x | 1 < x ∧ x < 3}

theorem intersection_of_A_and_B : A ∩ B = {x | 1 < x ∧ x < 2} := by
  sorry

end intersection_of_A_and_B_l219_219569


namespace num_positive_integers_l219_219265

theorem num_positive_integers (N : ℕ) (h : N > 3) : (∃ (k : ℕ) (h_div : 48 % k = 0), k = N - 3) → (∃ (c : ℕ), c = 8) := sorry

end num_positive_integers_l219_219265


namespace no_possible_values_for_n_l219_219180

theorem no_possible_values_for_n (n a : ℤ) (h : n > 1) (d : ℤ := 3) (Sn : ℤ := 180) :
  ∃ n > 1, ∃ k : ℤ, a = k^2 ∧ Sn = n / 2 * (2 * a + (n - 1) * d) :=
sorry

end no_possible_values_for_n_l219_219180


namespace max_m_minus_n_l219_219708

theorem max_m_minus_n (m n : ℝ) (h : (m + 1)^2 + (n + 1)^2 = 4) : m - n ≤ 2 * Real.sqrt 2 :=
by {
  -- Here is where the proof would take place.
  sorry
}

end max_m_minus_n_l219_219708


namespace suff_but_not_necc_l219_219274

def p (x : ℝ) : Prop := x = 2
def q (x : ℝ) : Prop := (x - 2) * (x + 3) = 0

theorem suff_but_not_necc (x : ℝ) : (p x → q x) ∧ ¬(q x → p x) :=
by
  sorry

end suff_but_not_necc_l219_219274


namespace three_squares_sum_l219_219452

theorem three_squares_sum (n : ℤ) (h : n > 5) : 
  3 * (n - 1)^2 + 32 = (n - 5)^2 + (n - 1)^2 + (n + 3)^2 :=
by sorry

end three_squares_sum_l219_219452


namespace corn_acres_l219_219670

theorem corn_acres (total_acres : ℕ) (ratio_beans : ℕ) (ratio_wheat : ℕ) (ratio_corn : ℕ) (total_ratio : ℕ)
  (h_total : total_acres = 1034)
  (h_ratio_beans : ratio_beans = 5) 
  (h_ratio_wheat : ratio_wheat = 2) 
  (h_ratio_corn : ratio_corn = 4) 
  (h_total_ratio : total_ratio = ratio_beans + ratio_wheat + ratio_corn) :
  let acres_per_part := total_acres / total_ratio in
  total_acres / total_ratio * ratio_corn = 376 := 
by 
  sorry

end corn_acres_l219_219670


namespace right_triangle_area_inscribed_3_4_l219_219557

theorem right_triangle_area_inscribed_3_4 (r1 r2: ℝ) (h1 : r1 = 3) (h2 : r2 = 4) : 
  ∃ (S: ℝ), S = 150 :=
by
  sorry

end right_triangle_area_inscribed_3_4_l219_219557


namespace john_february_bill_l219_219664

-- Define the conditions as constants
def base_cost : ℝ := 25
def cost_per_text : ℝ := 0.1 -- 10 cents
def cost_per_over_minute : ℝ := 0.1 -- 10 cents
def texts_sent : ℝ := 200
def hours_talked : ℝ := 51
def included_hours : ℝ := 50
def minutes_per_hour : ℝ := 60

-- Total cost computation
def total_cost : ℝ :=
  base_cost +
  (texts_sent * cost_per_text) +
  ((hours_talked - included_hours) * minutes_per_hour * cost_per_over_minute)

-- Proof statement
theorem john_february_bill : total_cost = 51 := by
  -- Proof omitted
  sorry

end john_february_bill_l219_219664


namespace rotate_image_eq_A_l219_219511

def image_A : Type := sorry -- Image data for option (A)
def original_image : Type := sorry -- Original image data

def rotate_90_clockwise (img : Type) : Type := sorry -- Function to rotate image 90 degrees clockwise

theorem rotate_image_eq_A :
  rotate_90_clockwise original_image = image_A :=
sorry

end rotate_image_eq_A_l219_219511


namespace solution_set_for_inequality_l219_219973

def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x^2 + (a - b) * x + 1

theorem solution_set_for_inequality (a b : ℝ) (h1 : 2*a + 4 = -(a-1)) :
  ∀ x : ℝ, (f x a b > f b a b) ↔ ((x ∈ Set.Icc (-2 : ℝ) (2 : ℝ)) ∧ ((x < -1 ∨ 1 < x))) :=
by
  sorry

end solution_set_for_inequality_l219_219973


namespace total_gallons_of_seed_l219_219237

-- Condition (1): The area of the football field is 8000 square meters.
def area_football_field : ℝ := 8000

-- Condition (2): Each square meter needs 4 times as much seed as fertilizer.
def seed_to_fertilizer_ratio : ℝ := 4

-- Condition (3): Carson uses 240 gallons of seed and fertilizer combined for every 2000 square meters.
def combined_usage_per_2000sqm : ℝ := 240
def area_unit : ℝ := 2000

-- Target: Prove that the total gallons of seed Carson uses for the entire field is 768 gallons.
theorem total_gallons_of_seed : seed_to_fertilizer_ratio * area_football_field / area_unit / (seed_to_fertilizer_ratio + 1) * combined_usage_per_2000sqm * (area_football_field / area_unit) = 768 :=
sorry

end total_gallons_of_seed_l219_219237


namespace original_price_of_shirts_l219_219137

theorem original_price_of_shirts 
  (sale_price : ℝ) 
  (fraction_of_original : ℝ) 
  (original_price : ℝ) 
  (h1 : sale_price = 6) 
  (h2 : fraction_of_original = 0.25) 
  (h3 : sale_price = fraction_of_original * original_price) 
  : original_price = 24 := 
by 
  sorry

end original_price_of_shirts_l219_219137


namespace expression_equals_five_l219_219104

theorem expression_equals_five (a : ℝ) (h : 2 * a^2 - 3 * a + 4 = 5) : 7 + 6 * a - 4 * a^2 = 5 :=
by
  sorry

end expression_equals_five_l219_219104


namespace find_paycheck_l219_219613

variable (P : ℝ) -- P represents the paycheck amount

def initial_balance : ℝ := 800
def rent_payment : ℝ := 450
def electricity_bill : ℝ := 117
def internet_bill : ℝ := 100
def phone_bill : ℝ := 70
def final_balance : ℝ := 1563

theorem find_paycheck :
  initial_balance - rent_payment + P - (electricity_bill + internet_bill) - phone_bill = final_balance → 
    P = 1563 :=
by
  sorry

end find_paycheck_l219_219613


namespace correct_student_mark_l219_219626

theorem correct_student_mark :
  ∀ (total_marks total_correct_marks incorrect_mark correct_average students : ℝ)
  (h1 : total_marks = students * 100)
  (h2 : incorrect_mark = 60)
  (h3 : correct_average = 95)
  (h4 : total_correct_marks = students * correct_average),
  total_marks - incorrect_mark + (total_correct_marks - (total_marks - incorrect_mark)) = 10 :=
by
  intros total_marks total_correct_marks incorrect_mark correct_average students h1 h2 h3 h4
  sorry

end correct_student_mark_l219_219626


namespace triangle_area_l219_219725

theorem triangle_area (a b : ℝ) (C : ℝ) (h1 : a = 2) (h2 : b = 3) (h3 : C = π / 3) : 
  (1/2 * a * b * Real.sin C) = (3 * Real.sqrt 3 / 2) :=
by
  sorry

end triangle_area_l219_219725


namespace matrix_determinant_l219_219977

theorem matrix_determinant :
  let x := 1
  let y := -1
  det ![![x, -3], ![y, 2]] = -1 := 
by
  -- define x and y
  let x := 1
  let y := -1
  -- define the matrix
  have matrix := ![![x, -3], ![y, 2]]
  -- calculate the determinant
  calc
    det matrix = x * 2 - (-3) * y : by simp [matrix, det]
           ... = 1 * 2 - (-3) * (-1) : by simp [x, y]
           ... = 2 - 3 : by norm_num
           ... = -1 : by norm_num

end matrix_determinant_l219_219977


namespace books_assigned_total_l219_219445

-- Definitions for the conditions.
def Mcgregor_books := 34
def Floyd_books := 32
def remaining_books := 23

-- The total number of books assigned.
def total_books := Mcgregor_books + Floyd_books + remaining_books

-- The theorem that needs to be proven.
theorem books_assigned_total : total_books = 89 :=
by
  sorry

end books_assigned_total_l219_219445


namespace vectors_coplanar_l219_219232

def vector3 := ℝ × ℝ × ℝ

def scalar_triple_product (a b c : vector3) : ℝ :=
  match a, b, c with
  | (a1, a2, a3), (b1, b2, b3), (c1, c2, c3) =>
    a1 * (b2 * c3 - b3 * c2) - a2 * (b1 * c3 - b3 * c1) + a3 * (b1 * c2 - b2 * c1)

theorem vectors_coplanar : scalar_triple_product (-3, 3, 3) (-4, 7, 6) (3, 0, -1) = 0 :=
by
  sorry

end vectors_coplanar_l219_219232


namespace no_solutions_l219_219619

theorem no_solutions
  (x y z : ℤ)
  (h : x^2 + y^2 = 4 * z - 1) : False :=
sorry

end no_solutions_l219_219619


namespace marion_paperclips_correct_l219_219604

def yun_initial_paperclips := 30
def yun_remaining_paperclips (x : ℕ) : ℕ := (2 * x) / 5
def marion_paperclips (x y : ℕ) : ℕ := (4 * (yun_remaining_paperclips x)) / 3 + y
def y := 7

theorem marion_paperclips_correct : marion_paperclips yun_initial_paperclips y = 23 := by
  sorry

end marion_paperclips_correct_l219_219604


namespace largest_angle_of_pentagon_l219_219624

theorem largest_angle_of_pentagon (a d : ℝ) (h1 : a = 100) (h2 : d = 2) :
  let angle1 := a
  let angle2 := a + d
  let angle3 := a + 2 * d
  let angle4 := a + 3 * d
  let angle5 := a + 4 * d
  angle1 + angle2 + angle3 + angle4 + angle5 = 540 ∧ angle5 = 116 :=
by
  sorry

end largest_angle_of_pentagon_l219_219624


namespace part1_part2_l219_219081

open Real

-- Condition: tan(alpha) = 3
variable {α : ℝ} (h : tan α = 3)

-- Proof of first part
theorem part1 : (4 * sin α - 2 * cos α) / (5 * cos α + 3 * sin α) = 5 / 7 :=
by
  sorry

-- Proof of second part
theorem part2 : 1 - 4 * sin α * cos α + 2 * cos α ^ 2 = 0 :=
by
  sorry

end part1_part2_l219_219081


namespace cannot_achieve_1970_minuses_l219_219425

theorem cannot_achieve_1970_minuses :
  ∃ (x y : ℕ), x ≤ 100 ∧ y ≤ 100 ∧ (x - 50) * (y - 50) = 1515 → false :=
by
  sorry

end cannot_achieve_1970_minuses_l219_219425


namespace problem_conditions_l219_219280

theorem problem_conditions (a b c x : ℝ) :
  (∀ x, ax^2 + bx + c ≥ 0 ↔ (x ≤ -3 ∨ x ≥ 4)) →
  (a > 0) ∧
  (∀ x, bx + c > 0 → x > -12 = false) ∧
  (∀ x, cx^2 - bx + a < 0 ↔ (x < -1/4 ∨ x > 1/3)) ∧
  (a + b + c ≤ 0) :=
by
  sorry

end problem_conditions_l219_219280


namespace trader_profit_percentage_l219_219917

-- Definitions for the conditions
def trader_buys_weight (indicated_weight: ℝ) : ℝ :=
  1.10 * indicated_weight

def trader_claimed_weight_to_customer (actual_weight: ℝ) : ℝ :=
  1.30 * actual_weight

-- Main theorem statement
theorem trader_profit_percentage (indicated_weight: ℝ) (actual_weight: ℝ) (claimed_weight: ℝ) :
  trader_buys_weight 1000 = 1100 →
  trader_claimed_weight_to_customer actual_weight = claimed_weight →
  claimed_weight = 1000 →
  (1000 - actual_weight) / actual_weight * 100 = 30 :=
by
  intros h1 h2 h3
  sorry

end trader_profit_percentage_l219_219917


namespace student_count_l219_219165

theorem student_count 
  (initial_avg_height : ℚ)
  (incorrect_height : ℚ)
  (actual_height : ℚ)
  (actual_avg_height : ℚ)
  (n : ℕ)
  (h1 : initial_avg_height = 175)
  (h2 : incorrect_height = 151)
  (h3 : actual_height = 136)
  (h4 : actual_avg_height = 174.5)
  (h5 : n > 0) : n = 30 :=
by
  sorry

end student_count_l219_219165


namespace min_x_prime_factorization_sum_eq_31_l219_219437

theorem min_x_prime_factorization_sum_eq_31
    (x y a b c d : ℕ)
    (hx : x > 0)
    (hy : y > 0)
    (h : 7 * x^5 = 11 * y^13)
    (hx_prime_fact : ∃ a c b d : ℕ, x = a^c * b^d) :
    a + b + c + d = 31 :=
by
 sorry
 
end min_x_prime_factorization_sum_eq_31_l219_219437


namespace sqrt_221_range_l219_219391

theorem sqrt_221_range : 14 < Real.sqrt 221 ∧ Real.sqrt 221 < 15 := by
  sorry

end sqrt_221_range_l219_219391


namespace total_spent_at_music_store_l219_219073

-- Defining the costs
def clarinet_cost : ℝ := 130.30
def song_book_cost : ℝ := 11.24

-- The main theorem to prove
theorem total_spent_at_music_store : clarinet_cost + song_book_cost = 141.54 :=
by
  sorry

end total_spent_at_music_store_l219_219073


namespace percentage_decrease_in_price_l219_219635

theorem percentage_decrease_in_price (original_price new_price decrease percentage : ℝ) :
  original_price = 1300 → new_price = 988 →
  decrease = original_price - new_price →
  percentage = (decrease / original_price) * 100 →
  percentage = 24 := by
  sorry

end percentage_decrease_in_price_l219_219635


namespace competition_participants_l219_219993

theorem competition_participants (N : ℕ)
  (h1 : (1 / 12) * N = 18) :
  N = 216 := 
by
  sorry

end competition_participants_l219_219993


namespace range_of_a_l219_219045

namespace ProofProblem

theorem range_of_a (a : ℝ) (H1 : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → ∃ y : ℝ, y = a * x + 2 * a + 1 ∧ y > 0 ∧ y < 0) : 
  -1 < a ∧ a < -1/3 := 
sorry

end ProofProblem

end range_of_a_l219_219045


namespace number_of_persons_in_second_group_l219_219351

-- Definitions based on conditions
def total_man_hours_first_group : ℕ := 42 * 12 * 5

def total_man_hours_second_group (X : ℕ) : ℕ := X * 14 * 6

-- Theorem stating that the number of persons in the second group is 30, given the conditions
theorem number_of_persons_in_second_group (X : ℕ) : 
  total_man_hours_first_group = total_man_hours_second_group X → X = 30 :=
by
  sorry

end number_of_persons_in_second_group_l219_219351


namespace jane_book_pages_l219_219303

theorem jane_book_pages (x : ℝ) :
  (x - (1 / 4 * x + 10) - (1 / 5 * (x - (1 / 4 * x + 10)) + 20) - (1 / 2 * (x - (1 / 4 * x + 10) - (1 / 5 * (x - (1 / 4 * x + 10)) + 20)) + 25) = 75) → x = 380 :=
by
  sorry

end jane_book_pages_l219_219303


namespace puppy_food_cost_l219_219444

theorem puppy_food_cost :
  let puppy_cost : ℕ := 10
  let days_in_week : ℕ := 7
  let total_number_of_weeks : ℕ := 3
  let cups_per_day : ℚ := 1 / 3
  let cups_per_bag : ℚ := 3.5
  let cost_per_bag : ℕ := 2
  let total_days := total_number_of_weeks * days_in_week
  let total_cups := total_days * cups_per_day
  let total_bags := total_cups / cups_per_bag
  let food_cost := total_bags * cost_per_bag
  let total_cost := puppy_cost + food_cost
  total_cost = 14 := by
  sorry

end puppy_food_cost_l219_219444


namespace tan_2alpha_of_sin_cos_ratio_l219_219846

theorem tan_2alpha_of_sin_cos_ratio (α : ℝ) (h : (sin α + cos α) / (sin α - cos α) = 1/2) : 
  tan (2 * α) = 3 / 4 := 
by
  sorry

end tan_2alpha_of_sin_cos_ratio_l219_219846


namespace corn_acres_l219_219669

theorem corn_acres (total_acres : ℕ) (ratio_beans : ℕ) (ratio_wheat : ℕ) (ratio_corn : ℕ) (total_ratio : ℕ)
  (h_total : total_acres = 1034)
  (h_ratio_beans : ratio_beans = 5) 
  (h_ratio_wheat : ratio_wheat = 2) 
  (h_ratio_corn : ratio_corn = 4) 
  (h_total_ratio : total_ratio = ratio_beans + ratio_wheat + ratio_corn) :
  let acres_per_part := total_acres / total_ratio in
  total_acres / total_ratio * ratio_corn = 376 := 
by 
  sorry

end corn_acres_l219_219669


namespace find_c_value_l219_219657

theorem find_c_value (a b : ℝ) (h1 : 12 = (6 / 100) * a) (h2 : 6 = (12 / 100) * b) : b / a = 0.25 :=
by
  sorry

end find_c_value_l219_219657


namespace car_bus_initial_speed_l219_219522

theorem car_bus_initial_speed {d : ℝ} {t : ℝ} {s_c : ℝ} {s_b : ℝ}
    (h1 : t = 4) 
    (h2 : s_c = s_b + 8) 
    (h3 : d = 384)
    (h4 : ∀ t, 0 ≤ t → t ≤ 2 → d = s_c * t + s_b * t) 
    (h5 : ∀ t, 2 < t → t ≤ 4 → d = (s_c - 10) * (t - 2) + s_b * (t - 2)) 
    : s_b = 46.5 ∧ s_c = 54.5 := 
by 
    sorry

end car_bus_initial_speed_l219_219522


namespace regular_polygon_sides_l219_219807

theorem regular_polygon_sides (h : ∀ n : ℕ, (120 * n) = 180 * (n - 2)) : 6 = 6 :=
by
  sorry

end regular_polygon_sides_l219_219807


namespace geom_seq_expression_l219_219266

theorem geom_seq_expression (a : ℕ → ℝ) (q : ℝ) (h1 : a 1 + a 3 = 10) (h2 : a 2 + a 4 = 5) :
  ∀ n, a n = 2 ^ (4 - n) :=
by
  -- sorry is used to skip the proof
  sorry

end geom_seq_expression_l219_219266


namespace defect_rate_probability_l219_219382

theorem defect_rate_probability (p : ℝ) (n : ℕ) (ε : ℝ) (q : ℝ) : 
  p = 0.02 →
  n = 800 →
  ε = 0.01 →
  q = 1 - p →
  1 - (p * q) / (n * ε^2) = 0.755 :=
by
  intro hp hn he hq
  rw [hp, hn, he, hq]
  -- Calculation steps can be verified here
  sorry

end defect_rate_probability_l219_219382


namespace max_points_per_player_l219_219426

theorem max_points_per_player
  (num_players : ℕ)
  (total_points : ℕ)
  (min_points_per_player : ℕ)
  (extra_points : ℕ)
  (scores_by_two_or_three : Prop)
  (fouls : Prop) :
  num_players = 12 →
  total_points = 100 →
  min_points_per_player = 8 →
  scores_by_two_or_three →
  fouls →
  extra_points = (total_points - num_players * min_points_per_player) →
  q = min_points_per_player + extra_points →
  q = 12 :=
by
  intros
  sorry

end max_points_per_player_l219_219426


namespace find_mn_l219_219974

theorem find_mn
  (AB BC : ℝ) -- Lengths of AB and BC
  (m n : ℝ)   -- Coefficients of the quadratic equation
  (h_perimeter : 2 * (AB + BC) = 12)
  (h_area : AB * BC = 5)
  (h_roots_sum : AB + BC = -m)
  (h_roots_product : AB * BC = n) :
  m * n = -30 :=
by
  sorry

end find_mn_l219_219974


namespace n_mult_n_plus_1_eq_square_l219_219261

theorem n_mult_n_plus_1_eq_square (n : ℤ) : (∃ k : ℤ, n * (n + 1) = k^2) ↔ (n = 0 ∨ n = -1) := 
by sorry

end n_mult_n_plus_1_eq_square_l219_219261


namespace matrix_N_cross_product_l219_219074

theorem matrix_N_cross_product (v : ℝ^3) :
  let N := λ (v : ℝ^3), ![![0, -4, -1], ![4, 0, -3], ![1, 3, 0]]
  in N.mul_vec v = vector_cross_product ![3, -1, 4] v :=
by
  sorry

end matrix_N_cross_product_l219_219074


namespace arithmetic_square_root_of_9_is_3_l219_219461

-- Define the arithmetic square root property
def is_arithmetic_square_root (x : ℝ) (n : ℝ) : Prop :=
  x * x = n ∧ x ≥ 0

-- The main theorem: The arithmetic square root of 9 is 3
theorem arithmetic_square_root_of_9_is_3 : 
  is_arithmetic_square_root 3 9 :=
by
  -- This is where the proof would go, but since only the statement is required:
  sorry

end arithmetic_square_root_of_9_is_3_l219_219461


namespace medium_ceiling_lights_count_l219_219299

theorem medium_ceiling_lights_count (S M L : ℕ) 
  (h1 : L = 2 * M) 
  (h2 : S = M + 10) 
  (h_bulbs : S + 2 * M + 3 * L = 118) : M = 12 :=
by
  -- Proof omitted
  sorry

end medium_ceiling_lights_count_l219_219299


namespace find_x_l219_219072

noncomputable def arithmetic_sequence (x : ℝ) : Prop := 
  (x + 1) - (1/3) = 4 * x - (x + 1)

theorem find_x :
  ∃ x : ℝ, arithmetic_sequence x ∧ x = 5 / 6 :=
by
  use 5 / 6
  unfold arithmetic_sequence
  sorry

end find_x_l219_219072


namespace find_a8_a12_sum_l219_219969

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem find_a8_a12_sum
  (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a) 
  (h1 : a 2 + a 6 = 3) 
  (h2 : a 6 + a 10 = 12) : 
  a 8 + a 12 = 24 :=
sorry

end find_a8_a12_sum_l219_219969


namespace sequence_sum_l219_219966

theorem sequence_sum (S : ℕ → ℕ) (h : ∀ n, S n = n^2 + 2 * n) : S 6 - S 2 = 40 :=
by
  sorry

end sequence_sum_l219_219966


namespace normal_vector_proof_l219_219983

-- Define the 3D vector type
structure Vector3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a specific normal vector n
def n : Vector3D := ⟨1, -2, 2⟩

-- Define the vector v we need to prove is a normal vector of the same plane
def v : Vector3D := ⟨2, -4, 4⟩

-- Define the statement (without the proof)
theorem normal_vector_proof : v = ⟨2 * n.x, 2 * n.y, 2 * n.z⟩ :=
by
  sorry

end normal_vector_proof_l219_219983


namespace charity_race_finished_racers_l219_219033

theorem charity_race_finished_racers :
  let initial_racers := 50
  let joined_after_20_minutes := 30
  let doubled_after_30_minutes := 2
  let dropped_racers := 30
  let total_racers_after_20_minutes := initial_racers + joined_after_20_minutes
  let total_racers_after_50_minutes := total_racers_after_20_minutes * doubled_after_30_minutes
  let finished_racers := total_racers_after_50_minutes - dropped_racers
  finished_racers = 130 := by
    sorry

end charity_race_finished_racers_l219_219033


namespace P_gt_neg1_l219_219134

noncomputable def X : MeasureTheory.Measure ℝ := sorry

axiom normal_dist (X : MeasureTheory.Measure ℝ) : True := sorry

variable {p : ℝ}

axiom P_gt_1 (hX : X) : prob {ω | ω > 1} = p := sorry

theorem P_gt_neg1 (hX : X) : prob {ω | ω > -1} = 1 - p := sorry

end P_gt_neg1_l219_219134


namespace pens_difference_proof_l219_219310

variables (A B M N X Y : ℕ)

-- Initial number of pens for Alex and Jane
def Alex_initial (A : ℕ) := A
def Jane_initial (B : ℕ) := B

-- Weekly multiplication factors for Alex and Jane
def Alex_weekly_growth (X : ℕ) := X
def Jane_weekly_growth (Y : ℕ) := Y

-- Number of pens after 4 weeks
def Alex_after_4_weeks (A X : ℕ) := A * X^4
def Jane_after_4_weeks (B Y : ℕ) := B * Y^4

-- Proving the difference in the number of pens
theorem pens_difference_proof (hM : M = A * X^4) (hN : N = B * Y^4) :
  M - N = (A * X^4) - (B * Y^4) :=
by sorry

end pens_difference_proof_l219_219310


namespace square_of_number_l219_219262

theorem square_of_number (x : ℝ) (h : 2 * x = x / 5 + 9) : x^2 = 25 := 
sorry

end square_of_number_l219_219262


namespace sqrt_of_9_fact_over_84_eq_24_sqrt_15_l219_219388

theorem sqrt_of_9_fact_over_84_eq_24_sqrt_15 :
  Real.sqrt (9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1 / (2^2 * 3 * 7)) = 24 * Real.sqrt 15 :=
by
  sorry

end sqrt_of_9_fact_over_84_eq_24_sqrt_15_l219_219388


namespace min_value_b_l219_219276

noncomputable def f (x a : ℝ) := 3 * x^2 - 4 * a * x
noncomputable def g (x a b : ℝ) := 2 * a^2 * Real.log x - b
noncomputable def f' (x a : ℝ) := 6 * x - 4 * a
noncomputable def g' (x a : ℝ) := 2 * a^2 / x

theorem min_value_b (a : ℝ) (h_a : a > 0) :
  ∃ (b : ℝ), ∃ (x₀ : ℝ), 
  (f x₀ a = g x₀ a b ∧ f' x₀ a = g' x₀ a) ∧ 
  ∀ (b' : ℝ), (∀ (x' : ℝ), (f x' a = g x' a b' ∧ f' x' a = g' x' a) → b' ≥ -1 / Real.exp 2) := 
sorry

end min_value_b_l219_219276


namespace sum_of_abs_arithmetic_sequence_l219_219182

theorem sum_of_abs_arithmetic_sequence {a_n : ℕ → ℤ} {S_n : ℕ → ℤ} 
  (hS3 : S_n 3 = 21) (hS9 : S_n 9 = 9) :
  ∃ (T_n : ℕ → ℤ), 
    (∀ (n : ℕ), n ≤ 5 → T_n n = -n^2 + 10 * n) ∧
    (∀ (n : ℕ), n ≥ 6 → T_n n = n^2 - 10 * n + 50) :=
sorry

end sum_of_abs_arithmetic_sequence_l219_219182


namespace prime_sum_divisible_l219_219005

theorem prime_sum_divisible (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (h : q = p + 2) :
  (p ^ q + q ^ p) % (p + q) = 0 :=
by
  sorry

end prime_sum_divisible_l219_219005


namespace problem_1_problem_2_problem_3_problem_4_problem_5_l219_219260

theorem problem_1 : 286 = 200 + 80 + 6 := sorry
theorem problem_2 : 7560 = 7000 + 500 + 60 := sorry
theorem problem_3 : 2048 = 2000 + 40 + 8 := sorry
theorem problem_4 : 8009 = 8000 + 9 := sorry
theorem problem_5 : 3070 = 3000 + 70 := sorry

end problem_1_problem_2_problem_3_problem_4_problem_5_l219_219260


namespace point_in_first_or_third_quadrant_l219_219718

-- Definitions based on conditions
variables {x y : ℝ}

-- The proof statement
theorem point_in_first_or_third_quadrant (h : x * y > 0) : 
  (0 < x ∧ 0 < y) ∨ (x < 0 ∧ y < 0) :=
  sorry

end point_in_first_or_third_quadrant_l219_219718


namespace length_squared_of_segment_CD_is_196_l219_219747

theorem length_squared_of_segment_CD_is_196 :
  ∃ (C D : ℝ × ℝ), 
    (C.2 = 3 * C.1 ^ 2 + 6 * C.1 - 2) ∧
    (D.2 = 3 * (2 - C.1) ^ 2 + 6 * (2 - C.1) - 2) ∧
    (1 : ℝ) = (C.1 + D.1) / 2 ∧
    (0 : ℝ) = (C.2 + D.2) / 2 ∧
    ((C.1 - D.1) ^ 2 + (C.2 - D.2) ^ 2 = 196) :=
by
  -- The proof would go here
  sorry

end length_squared_of_segment_CD_is_196_l219_219747


namespace flour_per_special_crust_l219_219120

-- Definitions of daily pie crusts and flour usage for standard crusts
def daily_pie_crusts := 50
def flour_per_standard_crust := 1 / 10
def total_daily_flour := daily_pie_crusts * flour_per_standard_crust

-- Definitions for special pie crusts today
def special_pie_crusts := 25
def total_special_flour := total_daily_flour / special_pie_crusts

-- Problem statement in Lean
theorem flour_per_special_crust :
  total_special_flour = 1 / 5 := by
  sorry

end flour_per_special_crust_l219_219120


namespace henry_added_water_l219_219414

theorem henry_added_water (initial_fraction full_capacity final_fraction : ℝ) (h_initial_fraction : initial_fraction = 3/4) (h_full_capacity : full_capacity = 56) (h_final_fraction : final_fraction = 7/8) :
  final_fraction * full_capacity - initial_fraction * full_capacity = 7 :=
by
  sorry

end henry_added_water_l219_219414


namespace probability_of_bayonet_base_on_third_try_is_7_over_120_l219_219568

noncomputable def probability_picking_bayonet_base_bulb_on_third_try : ℚ :=
  (3 / 10) * (2 / 9) * (7 / 8)

/-- Given a box containing 3 screw base bulbs and 7 bayonet base bulbs, all with the
same shape and power and placed with their bases down. An electrician takes one bulb
at a time without returning it. The probability that he gets a bayonet base bulb on his
third try is 7/120. -/
theorem probability_of_bayonet_base_on_third_try_is_7_over_120 :
  probability_picking_bayonet_base_bulb_on_third_try = 7 / 120 :=
by 
  sorry

end probability_of_bayonet_base_on_third_try_is_7_over_120_l219_219568


namespace prove_parabola_points_l219_219715

open Real

noncomputable def parabola_equation (x y : ℝ) : Prop := x^2 = 4 * y

noncomputable def dist_to_focus (x y focus_x focus_y : ℝ) : ℝ :=
  (sqrt ((x - focus_x)^2 + (y - focus_y)^2))

theorem prove_parabola_points :
  ∀ (x1 y1 x2 y2 : ℝ),
  parabola_equation x1 y1 →
  parabola_equation x2 y2 →
  dist_to_focus x1 y1 0 1 - dist_to_focus x2 y2 0 1 = 2 →
  (y1 + x1^2 - y2 - x2^2 = 10) :=
by
  intros x1 y1 x2 y2 h₁ h₂ h₃
  sorry

end prove_parabola_points_l219_219715


namespace downloaded_data_l219_219744

/-- 
  Mason is trying to download a 880 MB game to his phone. After downloading some amount, his Internet
  connection slows to 3 MB/minute. It will take him 190 more minutes to download the game. Prove that 
  Mason has downloaded 310 MB before his connection slowed down. 
-/
theorem downloaded_data (total_size : ℕ) (speed : ℕ) (time_remaining : ℕ) (remaining_data : ℕ) (downloaded : ℕ) :
  total_size = 880 ∧
  speed = 3 ∧
  time_remaining = 190 ∧
  remaining_data = speed * time_remaining ∧
  downloaded = total_size - remaining_data →
  downloaded = 310 := 
by 
  sorry

end downloaded_data_l219_219744


namespace find_a_l219_219411

variable (U : Set ℝ) (A : Set ℝ) (a : ℝ)

theorem find_a (hU_def : U = {2, 3, a^2 - a - 1})
               (hA_def : A = {2, 3})
               (h_compl : U \ A = {1}) :
  a = -1 ∨ a = 2 := 
sorry

end find_a_l219_219411


namespace number_of_space_diagonals_l219_219927

theorem number_of_space_diagonals
  (V E F T Q : ℕ)
  (hV : V = 30)
  (hE : E = 70)
  (hF : F = 42)
  (hT : T = 30)
  (hQ : Q = 12):
  (V * (V - 1) / 2 - E - 2 * Q) = 341 :=
by
  sorry

end number_of_space_diagonals_l219_219927


namespace alice_winning_strategy_l219_219683

theorem alice_winning_strategy (N : ℕ) (hN : N > 0) : 
  (∃! n : ℕ, N = n * n) ↔ (∀ (k : ℕ), ∃ (m : ℕ), m ≠ k ∧ (m ∣ k ∨ k ∣ m)) :=
sorry

end alice_winning_strategy_l219_219683


namespace gray_region_area_l219_219860

theorem gray_region_area 
  (r : ℝ) 
  (h1 : ∀ r : ℝ, (3 * r) - r = 3) 
  (h2 : r = 1.5) 
  (inner_circle_area : ℝ := π * r * r) 
  (outer_circle_area : ℝ := π * (3 * r) * (3 * r)) : 
  outer_circle_area - inner_circle_area = 18 * π := 
by
  sorry

end gray_region_area_l219_219860


namespace line_transformation_l219_219593

theorem line_transformation (a b : ℝ)
  (h1 : ∀ x y : ℝ, a * x + y - 7 = 0)
  (A : Matrix (Fin 2) (Fin 2) ℝ) (hA : A = ![![3, 0], ![-1, b]])
  (h2 : ∀ x' y' : ℝ, 9 * x' + y' - 91 = 0) :
  (a = 2) ∧ (b = 13) :=
by
  sorry

end line_transformation_l219_219593


namespace B_to_A_ratio_l219_219786

-- Define the conditions
def timeA : ℝ := 18
def combinedWorkRate : ℝ := 1 / 6

-- Define the ratios
def ratioOfTimes (timeB : ℝ) : ℝ := timeB / timeA

-- Prove the ratio of times given the conditions
theorem B_to_A_ratio :
  (∃ (timeB : ℝ), (1 / timeA + 1 / timeB = combinedWorkRate) ∧ ratioOfTimes timeB = 1 / 2) :=
sorry

end B_to_A_ratio_l219_219786


namespace greatest_int_less_than_200_gcd_30_is_5_l219_219504

theorem greatest_int_less_than_200_gcd_30_is_5 : ∃ n, n < 200 ∧ gcd n 30 = 5 ∧ ∀ m, m < 200 ∧ gcd m 30 = 5 → m ≤ n :=
by
  sorry

end greatest_int_less_than_200_gcd_30_is_5_l219_219504


namespace expand_product_l219_219554

-- We need to state the problem as a theorem
theorem expand_product (y : ℝ) (hy : y ≠ 0) : (3 / 7) * (7 / y + 14 * y^3) = 3 / y + 6 * y^3 :=
by
  sorry -- Skipping the proof

end expand_product_l219_219554


namespace cos_diff_l219_219151

theorem cos_diff (x y : ℝ) (hx : x = Real.cos (20 * Real.pi / 180)) (hy : y = Real.cos (40 * Real.pi / 180)) :
  x - y = 1 / 2 :=
by sorry

end cos_diff_l219_219151


namespace problem_l219_219748

open BigOperators

variables {p q : ℝ} {n : ℕ}

theorem problem 
  (h : p + q = 1) : 
  ∑ r in Finset.range (n / 2 + 1), (-1 : ℝ) ^ r * (Nat.choose (n - r) r) * p^r * q^r = (p ^ (n + 1) - q ^ (n + 1)) / (p - q) :=
by
  sorry

end problem_l219_219748


namespace fraction_to_decimal_l219_219942

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 :=
by
  sorry

end fraction_to_decimal_l219_219942


namespace sufficient_but_not_necessary_l219_219709

theorem sufficient_but_not_necessary (x y : ℝ) (h₁ : x = 2) (h₂ : y = -1) :
    (x + y - 1 = 0) ∧ ¬ ∀ x y, (x + y - 1 = 0) → (x = 2 ∧ y = -1) :=
  by
  sorry

end sufficient_but_not_necessary_l219_219709


namespace negation_proposition_of_cube_of_odd_is_odd_l219_219332

def odd (n : ℤ) := ∃ k : ℤ, n = 2 * k + 1

theorem negation_proposition_of_cube_of_odd_is_odd :
  (¬ ∀ n : ℤ, odd n → odd (n^3)) ↔ (∃ n : ℤ, odd n ∧ ¬ odd (n^3)) :=
by
  sorry

end negation_proposition_of_cube_of_odd_is_odd_l219_219332


namespace distance_in_interval_l219_219472

open Set Real

def distance_to_town (d : ℝ) : Prop :=
d < 8 ∧ 7 < d ∧ 6 < d

theorem distance_in_interval (d : ℝ) : distance_to_town d → d ∈ Ioo 7 8 :=
by
  intro h
  have d_in_Ioo_8 := h.left
  have d_in_Ioo_7 := h.right.left
  have d_in_Ioo_6 := h.right.right
  /- The specific steps for combining inequalities aren't needed for the final proof. -/
  sorry

end distance_in_interval_l219_219472


namespace box_distribution_l219_219776

theorem box_distribution (A P S : ℕ) (h : A + P + S = 22) : A ≥ 8 ∨ P ≥ 8 ∨ S ≥ 8 := 
by 
-- The next step is to use proof by contradiction, assuming the opposite.
sorry

end box_distribution_l219_219776


namespace twenty_fourth_digit_sum_l219_219648

theorem twenty_fourth_digit_sum (a b : ℚ) (h₁ : a = 1/7) (h₂ : b = 1/9) : 
  (Nat.digits 10 (Rat.mkPnat (a + b - (a + b).floor)).numerator.digits.full 24) = 8 :=
by
  sorry

end twenty_fourth_digit_sum_l219_219648


namespace fraction_to_decimal_l219_219944

theorem fraction_to_decimal : (7 / 16 : ℚ) = 0.4375 := by
  sorry

end fraction_to_decimal_l219_219944


namespace tagged_fish_ratio_l219_219115

theorem tagged_fish_ratio (tagged_first_catch : ℕ) (total_second_catch : ℕ) (tagged_second_catch : ℕ) 
  (approx_total_fish : ℕ) (h1 : tagged_first_catch = 60) 
  (h2 : total_second_catch = 50) 
  (h3 : tagged_second_catch = 2) 
  (h4 : approx_total_fish = 1500) :
  tagged_second_catch / total_second_catch = 1 / 25 := by
  sorry

end tagged_fish_ratio_l219_219115


namespace find_a_and_b_l219_219724

theorem find_a_and_b (a b : ℚ) :
  ((∃ x y : ℚ, 3 * x - y = 7 ∧ a * x + y = b) ∧
   (∃ x y : ℚ, x + b * y = a ∧ 2 * x + y = 8)) →
  a = -7/5 ∧ b = -11/5 :=
by sorry

end find_a_and_b_l219_219724


namespace fraction_of_science_liking_students_l219_219991

open Real

theorem fraction_of_science_liking_students (total_students math_fraction english_fraction no_fav_students math_students english_students fav_students remaining_students science_students fraction_science) :
  total_students = 30 ∧
  math_fraction = 1/5 ∧
  english_fraction = 1/3 ∧
  no_fav_students = 12 ∧
  math_students = total_students * math_fraction ∧
  english_students = total_students * english_fraction ∧
  fav_students = total_students - no_fav_students ∧
  remaining_students = fav_students - (math_students + english_students) ∧
  science_students = remaining_students ∧
  fraction_science = science_students / remaining_students →
  fraction_science = 1 :=
by
  sorry

end fraction_of_science_liking_students_l219_219991


namespace course_selection_l219_219681

noncomputable def number_of_ways (nA nB : ℕ) : ℕ :=
  (Nat.choose nA 2) * (Nat.choose nB 1) + (Nat.choose nA 1) * (Nat.choose nB 2)

theorem course_selection :
  (number_of_ways 3 4) = 30 :=
by
  sorry

end course_selection_l219_219681


namespace composite_fraction_l219_219009

theorem composite_fraction (x : ℤ) (hx : x = 5^25) : 
  ∃ a b : ℤ, a > 1 ∧ b > 1 ∧ a * b = x^4 + x^3 + x^2 + x + 1 :=
by sorry

end composite_fraction_l219_219009


namespace inequality_nonnegative_reals_l219_219705

theorem inequality_nonnegative_reals (a b c : ℝ) (h_a : 0 ≤ a) (h_b : 0 ≤ b) (h_c : 0 ≤ c) :
  |(c * a - a * b)| + |(a * b - b * c)| + |(b * c - c * a)| ≤ |(b^2 - c^2)| + |(c^2 - a^2)| + |(a^2 - b^2)| :=
by
  sorry

end inequality_nonnegative_reals_l219_219705


namespace first_caller_to_win_all_prizes_is_900_l219_219651

-- Define the conditions: frequencies of win types
def every_25th_caller_wins_music_player (n : ℕ) : Prop := n % 25 = 0
def every_36th_caller_wins_concert_tickets (n : ℕ) : Prop := n % 36 = 0
def every_45th_caller_wins_backstage_passes (n : ℕ) : Prop := n % 45 = 0

-- Formalize the problem to prove
theorem first_caller_to_win_all_prizes_is_900 :
  ∃ n : ℕ, every_25th_caller_wins_music_player n ∧
           every_36th_caller_wins_concert_tickets n ∧
           every_45th_caller_wins_backstage_passes n ∧
           n = 900 :=
by {
  sorry
}

end first_caller_to_win_all_prizes_is_900_l219_219651


namespace geom_seq_min_value_l219_219431

open Real

/-- 
Theorem: For a geometric sequence {a_n} where a_n > 0 and a_7 = √2/2, 
the minimum value of 1/a_3 + 2/a_11 is 4.
-/
theorem geom_seq_min_value (a : ℕ → ℝ) (a_pos : ∀ n, 0 < a n) (h7 : a 7 = (sqrt 2) / 2) :
  (1 / (a 3) + 2 / (a 11) >= 4) :=
sorry

end geom_seq_min_value_l219_219431


namespace problem_statement_l219_219831

variable (x y : ℝ)

theorem problem_statement
  (h1 : 4 * x + y = 9)
  (h2 : x + 4 * y = 16) :
  18 * x^2 + 20 * x * y + 18 * y^2 = 337 :=
sorry

end problem_statement_l219_219831


namespace perfect_square_solution_l219_219811

theorem perfect_square_solution (m n : ℕ) (p : ℕ) [hp : Fact (Nat.Prime p)] :
  (∃ k : ℕ, (5 ^ m + 2 ^ n * p) / (5 ^ m - 2 ^ n * p) = k ^ 2)
  ↔ (m = 1 ∧ n = 1 ∧ p = 2 ∨ m = 3 ∧ n = 2 ∧ p = 3 ∨ m = 2 ∧ n = 2 ∧ p = 5) :=
by
  sorry

end perfect_square_solution_l219_219811


namespace vector_satisfy_condition_l219_219249

structure Point where
  x : ℝ
  y : ℝ

structure Line where
  parametrize : ℝ → Point

def l : Line :=
  { parametrize := λ t => {x := 1 + 4 * t, y := 4 + 3 * t} }

def m : Line :=
  { parametrize := λ s => {x := -5 + 4 * s, y := 6 + 3 * s} }

def A (t : ℝ) : Point := l.parametrize t
def B (s : ℝ) : Point := m.parametrize s

-- The specific point for A and B are not used directly in the further proof statement.

def v : Point := { x := -6, y := 8 }

theorem vector_satisfy_condition :
  ∃ v1 v2 : ℝ, (v1 * -6) + (v2 * 8) = 2 ∧ (v1 = -6 ∧ v2 = 8) :=
sorry

end vector_satisfy_condition_l219_219249


namespace identify_infected_person_in_4_tests_l219_219416

theorem identify_infected_person_in_4_tests :
  (∀ (group : Fin 16 → Bool), ∃ infected : Fin 16, group infected = ff) →
  ∃ (tests_needed : ℕ), tests_needed = 4 :=
by sorry

end identify_infected_person_in_4_tests_l219_219416


namespace Vanya_433_sum_l219_219195

theorem Vanya_433_sum : 
  ∃ (A B : ℕ), 
  A + B = 91 
  ∧ (3 * A + 7 * B = 433) 
  ∧ (∃ (subsetA subsetB : Finset ℕ),
      (∀ x ∈ subsetA, x ∈ Finset.range (13 + 1))
      ∧ (∀ x ∈ subsetB, x ∈ Finset.range (13 + 1))
      ∧ subsetA ∩ subsetB = ∅
      ∧ subsetA ∪ subsetB = Finset.range (13 + 1)
      ∧ subsetA.card = 5
      ∧ subsetA.sum id = A
      ∧ subsetB.sum id = B) :=
by
  sorry

end Vanya_433_sum_l219_219195


namespace different_types_of_players_l219_219729

theorem different_types_of_players :
  ∀ (cricket hockey football softball : ℕ) (total_players : ℕ),
    cricket = 12 → hockey = 17 → football = 11 → softball = 10 → total_players = 50 →
    cricket + hockey + football + softball = total_players → 
    4 = 4 :=
by
  intros
  rfl

end different_types_of_players_l219_219729


namespace minimum_time_reach_distance_minimum_l219_219364

/-- Given a right triangle with legs of length 1 meter, and two bugs starting crawling from the vertices
with speeds 5 cm/s and 10 cm/s respectively, prove that the minimum time after the start of their movement 
for the distance between the bugs to reach its minimum is 4 seconds. -/
theorem minimum_time_reach_distance_minimum (l : ℝ) (v_A v_B : ℝ) (h_l : l = 1) (h_vA : v_A = 5 / 100) (h_vB : v_B = 10 / 100) :
  ∃ t_min : ℝ, t_min = 4 := by
  -- Proof is omitted
  sorry

end minimum_time_reach_distance_minimum_l219_219364


namespace range_of_m_l219_219394

noncomputable def A (x : ℝ) : Prop := |x - 2| ≤ 4
noncomputable def B (x : ℝ) (m : ℝ) : Prop := (x - 1 - m) * (x - 1 + m) ≤ 0 ∧ m > 0

theorem range_of_m (m : ℝ) :
  (∀ x, (¬A x) → (¬B x m)) ∧ (∃ x, (¬B x m) ∧ ¬(¬A x)) → m ≥ 5 :=
sorry

end range_of_m_l219_219394


namespace weight_of_one_pencil_l219_219679

theorem weight_of_one_pencil (total_weight : ℝ) (num_pencils : ℕ) (H : total_weight = 141.5) (H' : num_pencils = 5) : (total_weight / num_pencils) = 28.3 :=
by sorry

end weight_of_one_pencil_l219_219679


namespace ratio_of_speeds_l219_219680

theorem ratio_of_speeds (v_A v_B : ℝ) (d_A d_B t : ℝ) (h1 : d_A = 100) (h2 : d_B = 50) (h3 : v_A = d_A / t) (h4 : v_B = d_B / t) : 
  v_A / v_B = 2 := 
by sorry

end ratio_of_speeds_l219_219680


namespace Q_over_P_l219_219167

theorem Q_over_P :
  (∀ (x : ℝ), x ≠ -6 ∧ x ≠ 0 ∧ x ≠ 6 → 
    (P / (x + 6) + Q / (x^2 - 6*x) = (x^2 - 3*x + 12) / (x^3 + x^2 - 24*x))) →
  Q / P = 5 / 3 :=
by
  sorry

end Q_over_P_l219_219167


namespace max_intersections_circle_pentagon_l219_219043

theorem max_intersections_circle_pentagon : 
  ∃ (circle : Set Point) (pentagon : List (Set Point)),
    (∀ (side : Set Point), side ∈ pentagon → ∃ p1 p2 : Point, p1 ∈ circle ∧ p2 ∈ circle ∧ p1 ≠ p2) ∧
    pentagon.length = 5 →
    (∃ n : ℕ, n = 10) :=
by
  sorry

end max_intersections_circle_pentagon_l219_219043


namespace sqrt_expression_equals_l219_219801

theorem sqrt_expression_equals : Real.sqrt (5^2 * 7^4) = 245 :=
by
  sorry

end sqrt_expression_equals_l219_219801


namespace charts_per_associate_professor_l219_219687

theorem charts_per_associate_professor (A B C : ℕ) 
  (h1 : A + B = 6) 
  (h2 : 2 * A + B = 10) 
  (h3 : C * A + 2 * B = 8) : 
  C = 1 :=
by
  sorry

end charts_per_associate_professor_l219_219687


namespace sushi_downstream_distance_l219_219752

variable (sushi_speed : ℕ)
variable (stream_speed : ℕ := 12)
variable (upstream_distance : ℕ := 27)
variable (upstream_time : ℕ := 9)
variable (downstream_time : ℕ := 9)

theorem sushi_downstream_distance (h : upstream_distance = (sushi_speed - stream_speed) * upstream_time) : 
  ∃ (D_d : ℕ), D_d = (sushi_speed + stream_speed) * downstream_time ∧ D_d = 243 :=
by {
  -- We assume the given condition for upstream_distance
  sorry
}

end sushi_downstream_distance_l219_219752


namespace tangent_line_perpendicular_l219_219986

theorem tangent_line_perpendicular (m : ℝ) :
  (∀ x : ℝ, y = 2 * x^2) →
  (∀ x : ℝ, (4 * x - y + m = 0) ∧ (x + 4 * y - 8 = 0) → 
  (16 + 8 * m = 0)) →
  m = -2 :=
by
  sorry

end tangent_line_perpendicular_l219_219986


namespace smallest_n_for_sum_is_24_l219_219436

theorem smallest_n_for_sum_is_24 :
  ∃ (n : ℕ), (0 < n) ∧ 
    (∃ (k : ℤ), (1 : ℚ) / 3 + (1 : ℚ) / 4 + (1 : ℚ) / 8 + (1 : ℚ) / n = k) ∧
    ∀ (m : ℕ), ((0 < m) ∧ 
                (∃ (k' : ℤ), (1 : ℚ) / 3 + (1 : ℚ) / 4 + (1 : ℚ) / 8 + (1 : ℚ) / m = k') → n ≤ m) := sorry

end smallest_n_for_sum_is_24_l219_219436


namespace tanC_over_tanA_plus_tanC_over_tanB_l219_219995

theorem tanC_over_tanA_plus_tanC_over_tanB {a b c : ℝ} (A B C : ℝ) (h : a / b + b / a = 6 * Real.cos C) (acute_triangle : A > 0 ∧ A < Real.pi / 2 ∧ B > 0 ∧ B < Real.pi / 2 ∧ C > 0 ∧ C < Real.pi / 2) :
  (Real.tan C / Real.tan A) + (Real.tan C / Real.tan B) = 4 :=
sorry -- Proof not required

end tanC_over_tanA_plus_tanC_over_tanB_l219_219995


namespace correct_speed_l219_219745

noncomputable def distance (t : ℝ) := 50 * (t + 5 / 60)
noncomputable def distance2 (t : ℝ) := 70 * (t - 5 / 60)

theorem correct_speed : 
  ∃ r : ℝ, 
    (∀ t : ℝ, distance t = distance2 t → r = 55) := 
by
  sorry

end correct_speed_l219_219745


namespace probability_exact_four_out_of_twelve_dice_is_approx_0_089_l219_219035

noncomputable def dice_probability_exact_four_six : ℝ :=
  let p := (1/6 : ℝ)
  let q := (5/6 : ℝ)
  (Nat.choose 12 4) * (p ^ 4) * (q ^ 8)

theorem probability_exact_four_out_of_twelve_dice_is_approx_0_089 :
  abs (dice_probability_exact_four_six - 0.089) < 0.001 :=
sorry

end probability_exact_four_out_of_twelve_dice_is_approx_0_089_l219_219035


namespace teams_B_and_C_worked_together_days_l219_219889

def workload_project_B := 5/4
def time_team_A_project_A := 20
def time_team_B_project_A := 24
def time_team_C_project_A := 30

def equation1 (x y : ℕ) : Prop := 
  3 * x + 5 * y = 60

def equation2 (x y : ℕ) : Prop := 
  9 * x + 5 * y = 150

theorem teams_B_and_C_worked_together_days (x : ℕ) (y : ℕ) :
  equation1 x y ∧ equation2 x y → x = 15 := 
by 
  sorry

end teams_B_and_C_worked_together_days_l219_219889


namespace total_points_first_half_l219_219730

noncomputable def raiders_wildcats_scores := 
  ∃ (a b d r : ℕ),
    (a = b + 1) ∧
    (a * (1 + r + r^2 + r^3) = 4 * b + 6 * d + 2) ∧
    (a + a * r ≤ 100) ∧
    (b + b + d ≤ 100)

theorem total_points_first_half : 
  raiders_wildcats_scores → 
  ∃ (total : ℕ), total = 25 :=
by
  sorry

end total_points_first_half_l219_219730


namespace fish_speed_in_still_water_l219_219527

theorem fish_speed_in_still_water (u d : ℕ) (v : ℕ) : 
  u = 35 → d = 55 → 2 * v = u + d → v = 45 := 
by 
  intros h1 h2 h3
  rw [h1, h2] at h3
  linarith

end fish_speed_in_still_water_l219_219527


namespace acres_used_for_corn_l219_219668

-- Define the conditions in the problem:
def total_land : ℕ := 1034
def ratio_beans : ℕ := 5
def ratio_wheat : ℕ := 2
def ratio_corn : ℕ := 4
def total_ratio : ℕ := ratio_beans + ratio_wheat + ratio_corn

-- Proof problem statement: Prove the number of acres used for corn is 376 acres
theorem acres_used_for_corn : total_land * ratio_corn / total_ratio = 376 := by
  -- Proof goes here
  sorry

end acres_used_for_corn_l219_219668


namespace sqrt_10_integer_decimal_partition_l219_219094

theorem sqrt_10_integer_decimal_partition:
  let a := Int.floor (Real.sqrt 10)
  let b := Real.sqrt 10 - a
  (Real.sqrt 10 + a) * b = 1 :=
by
  sorry

end sqrt_10_integer_decimal_partition_l219_219094


namespace total_students_in_class_l219_219338

def number_of_girls := 9
def number_of_boys := 16
def total_students := number_of_girls + number_of_boys

theorem total_students_in_class : total_students = 25 :=
by
  -- The proof will go here
  sorry

end total_students_in_class_l219_219338


namespace evaluate_expression_at_zero_l219_219699

theorem evaluate_expression_at_zero :
  ∀ x : ℝ, (x ≠ -1) ∧ (x ≠ 3) →
  ( (3 * x^2 - 2 * x + 1) / ((x + 1) * (x - 3)) - (5 + 2 * x) / ((x + 1) * (x - 3)) ) = 2 :=
by
  sorry

end evaluate_expression_at_zero_l219_219699


namespace count_nonempty_valid_subsets_eq_l219_219601

open Finset

namespace ProofProblem

def T : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

def S (A : Finset ℕ) : ℕ := A.sum id

def valid_subset (A : Finset ℕ) : Prop :=
  A ≠ ∅ ∧ 3 ∣ S A ∧ ¬ 5 ∣ S A

noncomputable def count_valid_subsets (T : Finset ℕ) : ℕ :=
  (powerset T).filter valid_subset).card

theorem count_nonempty_valid_subsets_eq : count_valid_subsets T = 70 :=
  sorry

end ProofProblem

end count_nonempty_valid_subsets_eq_l219_219601


namespace escalator_length_l219_219369

theorem escalator_length
  (escalator_speed : ℝ)
  (person_speed : ℝ)
  (time_taken : ℝ)
  (combined_speed := escalator_speed + person_speed)
  (distance := combined_speed * time_taken) :
  escalator_speed = 10 → person_speed = 4 → time_taken = 8 → distance = 112 := by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end escalator_length_l219_219369


namespace collapsing_fraction_l219_219316

-- Define the total number of homes on Gotham St as a variable.
variable (T : ℕ)

/-- Fraction of homes on Gotham Street that are termite-ridden. -/
def fraction_termite_ridden (T : ℕ) : ℚ := 1 / 3

/-- Fraction of homes on Gotham Street that are termite-ridden but not collapsing. -/
def fraction_termite_not_collapsing (T : ℕ) : ℚ := 1 / 10

/-- Fraction of termite-ridden homes that are collapsing. -/
theorem collapsing_fraction :
  (fraction_termite_ridden T - fraction_termite_not_collapsing T) = 7 / 30 :=
by
  sorry

end collapsing_fraction_l219_219316


namespace function_satisfies_conditions_l219_219079

theorem function_satisfies_conditions :
  (∃ f : ℤ × ℤ → ℝ,
    (∀ x y z : ℤ, f (x, y) * f (y, z) * f (z, x) = 1) ∧
    (∀ x : ℤ, f (x + 1, x) = 2) ∧
    (∀ x y : ℤ, f (x, y) = 2 ^ (x - y))) :=
by
  sorry

end function_satisfies_conditions_l219_219079


namespace car_travel_first_hour_l219_219026

-- Define the conditions as variables and the ultimate equality to be proved
theorem car_travel_first_hour (x : ℕ) (h : 12 * x + 132 = 612) : x = 40 :=
by
  -- Proof will be completed here
  sorry

end car_travel_first_hour_l219_219026


namespace ordered_concrete_weight_l219_219357

def weight_of_materials : ℝ := 0.83
def weight_of_bricks : ℝ := 0.17
def weight_of_stone : ℝ := 0.5

theorem ordered_concrete_weight :
  weight_of_materials - (weight_of_bricks + weight_of_stone) = 0.16 := by
  sorry

end ordered_concrete_weight_l219_219357


namespace sequence_form_l219_219871

theorem sequence_form (c : ℕ) (a : ℕ → ℕ) :
  (∀ n : ℕ, 0 < n →
    (∃! i : ℕ, 0 < i ∧ a i ≤ a (n + 1) + c)) ↔
  (∀ n : ℕ, 0 < n → a n = n + (c + 1)) :=
by
  sorry

end sequence_form_l219_219871


namespace inequality_proof_l219_219273

theorem inequality_proof (x y z : ℝ) (h1 : x ≥ y) (h2 : y ≥ z) (h3 : z > 0) :
  (x^2 * y / (y + z) + y^2 * z / (z + x) + z^2 * x / (x + y) ≥ 1 / 2 * (x^2 + y^2 + z^2)) :=
by sorry

end inequality_proof_l219_219273


namespace gcd_lcm_mul_l219_219145

theorem gcd_lcm_mul (a b : ℤ) : (Int.gcd a b) * (Int.lcm a b) = a * b := by
  sorry

end gcd_lcm_mul_l219_219145


namespace razorback_tshirt_money_l219_219163

noncomputable def money_made_from_texas_tech_game (tshirt_price : ℕ) (total_sold : ℕ) (arkansas_sold : ℕ) : ℕ :=
  tshirt_price * (total_sold - arkansas_sold)

theorem razorback_tshirt_money :
  money_made_from_texas_tech_game 78 186 172 = 1092 := by
  sorry

end razorback_tshirt_money_l219_219163


namespace range_of_values_for_a_l219_219421

theorem range_of_values_for_a (a : ℝ) :
  (∀ x : ℝ, (x + 2) / 3 - x / 2 > 1 → 2 * (x - a) ≤ 0) → a ≥ -2 :=
by
  intro h
  sorry

end range_of_values_for_a_l219_219421


namespace total_percent_decrease_is_19_l219_219661

noncomputable def original_value : ℝ := 100
noncomputable def first_year_decrease : ℝ := 0.10
noncomputable def second_year_decrease : ℝ := 0.10
noncomputable def value_after_first_year : ℝ := original_value * (1 - first_year_decrease)
noncomputable def value_after_second_year : ℝ := value_after_first_year * (1 - second_year_decrease)
noncomputable def total_decrease_in_dollars : ℝ := original_value - value_after_second_year
noncomputable def total_percent_decrease : ℝ := (total_decrease_in_dollars / original_value) * 100

theorem total_percent_decrease_is_19 :
  total_percent_decrease = 19 := by
  sorry

end total_percent_decrease_is_19_l219_219661


namespace xiaolin_distance_l219_219200

theorem xiaolin_distance (speed : ℕ) (time : ℕ) (distance : ℕ)
    (h1 : speed = 80) (h2 : time = 28) : distance = 2240 :=
by
  have h3 : distance = time * speed := by sorry
  rw [h1, h2] at h3
  exact h3

end xiaolin_distance_l219_219200


namespace garden_furniture_costs_l219_219216

theorem garden_furniture_costs (B T U : ℝ)
    (h1 : T + B + U = 765)
    (h2 : T = 2 * B)
    (h3 : U = 3 * B) :
    B = 127.5 ∧ T = 255 ∧ U = 382.5 :=
by
  sorry

end garden_furniture_costs_l219_219216


namespace inequality_not_always_true_l219_219271

theorem inequality_not_always_true (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  ¬(∀ a > 0, ∀ b > 0, (2 / ((1 / a) + (1 / b)) ≥ Real.sqrt (a * b))) :=
sorry

end inequality_not_always_true_l219_219271


namespace c_finish_work_in_6_days_l219_219987

theorem c_finish_work_in_6_days (a b c : ℝ) (ha : a = 1/36) (hb : b = 1/18) (habc : a + b + c = 1/4) : c = 1/6 :=
by
  sorry

end c_finish_work_in_6_days_l219_219987


namespace incorrect_statement_B_l219_219342

theorem incorrect_statement_B (A B C : ℝ) (hAB : A * B < 0) (hBC : B * C < 0) : ¬ ∀ (x y : ℝ), x * y + A * x + B * y + C = 0 → (x < 0 ∧ y < 0) :=
by
  sorry

end incorrect_statement_B_l219_219342


namespace number_of_smaller_cubes_in_larger_cube_l219_219935

-- Defining the conditions
def volume_large_cube : ℝ := 125
def volume_small_cube : ℝ := 1
def surface_area_difference : ℝ := 600

-- Translating the question into a math proof problem
theorem number_of_smaller_cubes_in_larger_cube : 
  ∃ n : ℕ, n * 6 - 6 * (volume_large_cube^(1/3) ^ 2) = surface_area_difference :=
by
  sorry

end number_of_smaller_cubes_in_larger_cube_l219_219935


namespace museum_ticket_cost_l219_219208

theorem museum_ticket_cost (students teachers : ℕ) (student_ticket_cost teacher_ticket_cost : ℕ) 
  (h_students : students = 12) (h_teachers : teachers = 4) 
  (h_student_ticket_cost : student_ticket_cost = 1) (h_teacher_ticket_cost : teacher_ticket_cost = 3) :
  students * student_ticket_cost + teachers * teacher_ticket_cost = 24 :=
by
  rw [h_students, h_teachers, h_student_ticket_cost, h_teacher_ticket_cost]
  exact (12 * 1 + 4 * 3) 
-- This produces 24

end museum_ticket_cost_l219_219208


namespace prob_not_same_city_l219_219194

def prob_A_city_A : ℝ := 0.6
def prob_B_city_A : ℝ := 0.3

theorem prob_not_same_city :
  (prob_A_city_A * (1 - prob_B_city_A) + (1 - prob_A_city_A) * prob_B_city_A) = 0.54 :=
by 
  -- This is just a placeholder to indicate that the proof is skipped
  sorry

end prob_not_same_city_l219_219194


namespace prod_sum_reciprocal_bounds_l219_219617

-- Define the product of the sum of three positive numbers and the sum of their reciprocals.
theorem prod_sum_reciprocal_bounds (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  9 ≤ (a + b + c) * (1 / a + 1 / b + 1 / c) :=
by
  sorry

end prod_sum_reciprocal_bounds_l219_219617


namespace not_hexagonal_pyramid_l219_219535

-- Definition of the pyramid with slant height, base radius, and height
structure Pyramid where
  r : ℝ  -- Side length of the base equilateral triangle
  h : ℝ  -- Height of the pyramid
  l : ℝ  -- Slant height (lateral edge)
  hypo : h^2 + (r / 2)^2 = l^2

-- The theorem to prove a pyramid with all edges equal cannot be hexagonal
theorem not_hexagonal_pyramid (p : Pyramid) : p.l ≠ p.r :=
sorry

end not_hexagonal_pyramid_l219_219535


namespace paintings_left_correct_l219_219791

def initial_paintings := 98
def paintings_gotten_rid_of := 3

theorem paintings_left_correct :
  initial_paintings - paintings_gotten_rid_of = 95 :=
by
  sorry

end paintings_left_correct_l219_219791


namespace fraction_square_eq_decimal_l219_219197

theorem fraction_square_eq_decimal :
  ∃ (x : ℚ), x^2 = 0.04000000000000001 ∧ x = 1 / 5 :=
by
  sorry

end fraction_square_eq_decimal_l219_219197


namespace amount_of_money_l219_219848

variable (x : ℝ)

-- Conditions
def condition1 : Prop := x < 2000
def condition2 : Prop := 4 * x > 2000
def condition3 : Prop := 4 * x - 2000 = 2000 - x

theorem amount_of_money (h1 : condition1 x) (h2 : condition2 x) (h3 : condition3 x) : x = 800 :=
by
  sorry

end amount_of_money_l219_219848


namespace landscape_length_l219_219919

theorem landscape_length (b l : ℕ) (playground_area : ℕ) (total_area : ℕ) 
  (h1 : l = 4 * b) (h2 : playground_area = 1200) (h3 : total_area = 3 * playground_area) (h4 : total_area = l * b) :
  l = 120 := 
by 
  sorry

end landscape_length_l219_219919


namespace correct_reasoning_methods_l219_219512

-- Definitions based on conditions
def reasoning_1 : String := "Inductive reasoning"
def reasoning_2 : String := "Deductive reasoning"
def reasoning_3 : String := "Analogical reasoning"

-- Proposition stating that the correct answer is D
theorem correct_reasoning_methods :
  (reasoning_1 = "Inductive reasoning") ∧
  (reasoning_2 = "Deductive reasoning") ∧
  (reasoning_3 = "Analogical reasoning") ↔
  (choice = "D") :=
by sorry

end correct_reasoning_methods_l219_219512


namespace sqrt_expression_equals_l219_219800

theorem sqrt_expression_equals : Real.sqrt (5^2 * 7^4) = 245 :=
by
  sorry

end sqrt_expression_equals_l219_219800


namespace greatest_integer_gcf_l219_219496

theorem greatest_integer_gcf (x : ℕ) : x < 200 ∧ (gcd x 30 = 5) → x = 185 :=
by sorry

end greatest_integer_gcf_l219_219496


namespace negation_exists_cube_positive_l219_219473

theorem negation_exists_cube_positive :
  ¬ (∃ x : ℝ, x^3 > 0) ↔ ∀ x : ℝ, x^3 ≤ 0 := by
  sorry

end negation_exists_cube_positive_l219_219473


namespace arithmetic_progression_terms_l219_219022

theorem arithmetic_progression_terms
  (n : ℕ) (a d : ℝ)
  (hn_odd : n % 2 = 1)
  (sum_odd_terms : n / 2 * (2 * a + (n / 2 - 1) * d) = 30)
  (sum_even_terms : (n / 2 - 1) * (2 * (a + d) + (n / 2 - 2) * d) = 36)
  (sum_all_terms : n / 2 * (2 * a + (n - 1) * d) = 66)
  (last_first_diff : (n - 1) * d = 12) :
  n = 9 := sorry

end arithmetic_progression_terms_l219_219022


namespace A_or_B_not_A_or_C_A_and_C_A_and_B_or_C_l219_219938

-- Definitions of events
def A : Prop := sorry -- event that the part is of the first grade
def B : Prop := sorry -- event that the part is of the second grade
def C : Prop := sorry -- event that the part is of the third grade

-- Mathematically equivalent proof problems
theorem A_or_B : A ∨ B ↔ (A ∨ B) :=
by sorry

theorem not_A_or_C : ¬(A ∨ C) ↔ B :=
by sorry

theorem A_and_C : (A ∧ C) ↔ false :=
by sorry

theorem A_and_B_or_C : ((A ∧ B) ∨ C) ↔ C :=
by sorry

end A_or_B_not_A_or_C_A_and_C_A_and_B_or_C_l219_219938


namespace longer_bus_ride_l219_219881

theorem longer_bus_ride :
  let oscar := 0.75
  let charlie := 0.25
  oscar - charlie = 0.50 :=
by
  sorry

end longer_bus_ride_l219_219881


namespace solve_equation_l219_219325

-- Define the equation and the conditions
def problem_equation (x : ℝ) : Prop :=
  (3 * x + 6) / (x^2 + 5 * x - 6) = (3 - x) / (x - 2)

def valid_solution (x : ℝ) : Prop :=
  x ≠ 2 ∧ x ≠ 1 ∧ x ≠ -6

-- State the theorem that solutions x = 3 and x = -4 solve the problem under the conditions
theorem solve_equation : ∀ x : ℝ, valid_solution x → (x = 3 ∨ x = -4 ∧ problem_equation x) :=
by
  sorry

end solve_equation_l219_219325


namespace smallest_divisor_l219_219812

-- Define the given number and the subtracting number
def original_num : ℕ := 378461
def subtract_num : ℕ := 5

-- Define the resulting number after subtraction
def resulting_num : ℕ := original_num - subtract_num

-- Theorem stating that 47307 is the smallest divisor greater than 5 of 378456
theorem smallest_divisor : ∃ d: ℕ, d > 5 ∧ d ∣ resulting_num ∧ ∀ x: ℕ, x > 5 → x ∣ resulting_num → d ≤ x := 
sorry

end smallest_divisor_l219_219812


namespace inequality_square_l219_219105

theorem inequality_square (a b : ℝ) (h : a > b ∧ b > 0) : a^2 > b^2 :=
by
  sorry

end inequality_square_l219_219105


namespace train_cross_time_l219_219865

noncomputable def speed_km_per_hr_to_m_per_s (speed : ℝ) : ℝ :=
  (speed * 1000) / 3600

noncomputable def time_to_cross (length : ℝ) (speed : ℝ) : ℝ :=
  length / speed

theorem train_cross_time (length : ℝ) (speed_km_per_hr : ℝ) :
  length = 100 → speed_km_per_hr = 144 → time_to_cross length (speed_km_per_hr_to_m_per_s speed_km_per_hr) = 2.5 :=
by
  intros length_eq speed_eq
  rw [length_eq, speed_eq]
  simp [speed_km_per_hr_to_m_per_s, time_to_cross]
  norm_num
  sorry

end train_cross_time_l219_219865


namespace smallest_prime_with_prime_digit_sum_l219_219815

def is_prime (n : ℕ) : Prop := ¬ ∃ m, m ∣ n ∧ 1 < m ∧ m < n

def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem smallest_prime_with_prime_digit_sum :
  ∃ p, is_prime p ∧ is_prime (digit_sum p) ∧ 10 < digit_sum p ∧ p = 29 :=
by
  sorry

end smallest_prime_with_prime_digit_sum_l219_219815


namespace smallest_n_solution_unique_l219_219918

theorem smallest_n_solution_unique (a b c d : ℤ) (h : a^2 + b^2 + c^2 = 4 * d^2) : 
  a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0 :=
sorry

end smallest_n_solution_unique_l219_219918


namespace part1_part2_l219_219133

-- Define the function f
def f (x a : ℝ) : ℝ := abs (x - a) + 2 * x

-- (1) Given a = -1, prove that the inequality f(x, -1) ≤ 0 implies x ≤ -1/3
theorem part1 (x : ℝ) : (f x (-1) ≤ 0) ↔ (x ≤ -1/3) :=
by
  sorry

-- (2) Given f(x) ≥ 0 for all x ≥ -1, prove that the range for a is a ≤ -3 or a ≥ 1
theorem part2 (a : ℝ) : (∀ x, x ≥ -1 → f x a ≥ 0) ↔ (a ≤ -3 ∨ a ≥ 1) :=
by
  sorry

end part1_part2_l219_219133


namespace printer_time_ratio_l219_219912

theorem printer_time_ratio
  (X_time : ℝ) (Y_time : ℝ) (Z_time : ℝ)
  (hX : X_time = 15)
  (hY : Y_time = 10)
  (hZ : Z_time = 20) :
  (X_time / (Y_time * Z_time / (Y_time + Z_time))) = 9 / 4 :=
by
  sorry

end printer_time_ratio_l219_219912


namespace nina_jerome_age_ratio_l219_219590

variable (N J L : ℕ)

theorem nina_jerome_age_ratio (h1 : L = N - 4) (h2 : L + N + J = 36) (h3 : L = 6) : N / J = 1 / 2 := by
  sorry

end nina_jerome_age_ratio_l219_219590


namespace algorithm_output_l219_219281

theorem algorithm_output (x y: Int) (h_x: x = -5) (h_y: y = 15) : 
  let x := if x < 0 then y + 3 else x;
  x - y = 3 ∧ x + y = 33 :=
by
  sorry

end algorithm_output_l219_219281


namespace f_2008th_derivative_at_0_l219_219598

noncomputable def f (x : ℝ) : ℝ := (Real.sin (x / 4))^6 + (Real.cos (x / 4))^6

theorem f_2008th_derivative_at_0 : (deriv^[2008] f) 0 = 3 / 8 :=
sorry

end f_2008th_derivative_at_0_l219_219598


namespace lcm_of_18_and_20_l219_219959

theorem lcm_of_18_and_20 : Nat.lcm 18 20 = 180 := by
  sorry

end lcm_of_18_and_20_l219_219959


namespace negative_integer_example_l219_219541

def is_negative_integer (n : ℤ) := n < 0

theorem negative_integer_example : is_negative_integer (-2) :=
by
  -- Proof will go here
  sorry

end negative_integer_example_l219_219541


namespace not_even_or_odd_l219_219951

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x

theorem not_even_or_odd : ¬(∀ x : ℝ, f (-x) = f x) ∧ ¬(∀ x : ℝ, f (-x) = -f x) := by
  sorry

end not_even_or_odd_l219_219951


namespace number_of_animal_books_l219_219376

variable (A : ℕ)

theorem number_of_animal_books (h1 : 6 * 6 + 3 * 6 + A * 6 = 102) : A = 8 :=
sorry

end number_of_animal_books_l219_219376


namespace no_valid_partition_exists_l219_219591

namespace MathProof

-- Define the set of positive integers
def N := {n : ℕ // n > 0}

-- Define non-empty sets A, B, C which are disjoint and partition N
def valid_partition (A B C : N → Prop) : Prop :=
  (∃ a, A a) ∧ (∃ b, B b) ∧ (∃ c, C c) ∧
  (∀ n, A n → ¬ B n ∧ ¬ C n) ∧
  (∀ n, B n → ¬ A n ∧ ¬ C n) ∧
  (∀ n, C n → ¬ A n ∧ ¬ B n) ∧
  (∀ n, A n ∨ B n ∨ C n)

-- Define the conditions in the problem
def condition_1 (A B C : N → Prop) : Prop :=
  ∀ a b, A a → B b → C ⟨a.val + b.val + 1, by linarith [a.prop, b.prop]⟩

def condition_2 (A B C : N → Prop) : Prop :=
  ∀ b c, B b → C c → A ⟨b.val + c.val + 1, by linarith [b.prop, c.prop]⟩

def condition_3 (A B C : N → Prop) : Prop :=
  ∀ c a, C c → A a → B ⟨c.val + a.val + 1, by linarith [c.prop, a.prop]⟩

-- State the problem that no valid partition exists
theorem no_valid_partition_exists :
  ¬ ∃ (A B C : N → Prop), valid_partition A B C ∧
    condition_1 A B C ∧
    condition_2 A B C ∧
    condition_3 A B C :=
by
  sorry

end MathProof

end no_valid_partition_exists_l219_219591


namespace parallel_lines_condition_l219_219349

theorem parallel_lines_condition (a : ℝ) (l : ℝ) :
  (∀ (x y : ℝ), ax + 3*y + 3 = 0 → x + (a - 2)*y + l = 0 → a = -1) ∧ (a = -1 → ∀ (x y : ℝ), (ax + 3*y + 3 = 0 ↔ x + (a - 2)*y + l = 0)) :=
sorry

end parallel_lines_condition_l219_219349


namespace rachel_reading_homework_l219_219144

theorem rachel_reading_homework (math_hw : ℕ) (additional_reading_hw : ℕ) (total_reading_hw : ℕ) 
  (h1 : math_hw = 8) (h2 : additional_reading_hw = 6) (h3 : total_reading_hw = math_hw + additional_reading_hw) :
  total_reading_hw = 14 :=
sorry

end rachel_reading_homework_l219_219144


namespace cos_diff_l219_219152

theorem cos_diff (x y : ℝ) (hx : x = Real.cos (20 * Real.pi / 180)) (hy : y = Real.cos (40 * Real.pi / 180)) :
  x - y = 1 / 2 :=
by sorry

end cos_diff_l219_219152


namespace find_y_l219_219288

theorem find_y (x y : ℝ) (h₁ : x = 51) (h₂ : x^3 * y - 2 * x^2 * y + x * y = 51000) : y = 2 / 5 := by
  sorry

end find_y_l219_219288


namespace total_cost_l219_219443

-- Definitions corresponding to the conditions
def puppy_cost : ℝ := 10
def daily_food_consumption : ℝ := 1 / 3
def food_bag_content : ℝ := 3.5
def food_bag_cost : ℝ := 2
def days_in_week : ℝ := 7
def weeks_of_food : ℝ := 3

-- Statement of the problem
theorem total_cost :
  let 
    days := weeks_of_food * days_in_week,
    total_food_needed := days * daily_food_consumption,
    bags_needed := total_food_needed / food_bag_content,
    food_cost := bags_needed * food_bag_cost
  in
  puppy_cost + food_cost = 14 := 
by
  sorry

end total_cost_l219_219443


namespace funnel_paper_area_l219_219526

theorem funnel_paper_area
  (slant_height : ℝ)
  (base_circumference : ℝ)
  (h1 : slant_height = 6)
  (h2 : base_circumference = 6 * Real.pi):
  (1 / 2) * base_circumference * slant_height = 18 * Real.pi :=
by
  sorry

end funnel_paper_area_l219_219526


namespace condition_sufficient_but_not_necessary_l219_219464

theorem condition_sufficient_but_not_necessary (x y : ℝ) :
  (|x| + |y| ≤ 1 → x^2 + y^2 ≤ 1) ∧ (x^2 + y^2 ≤ 1 → ¬ (|x| + |y| ≤ 1)) :=
sorry

end condition_sufficient_but_not_necessary_l219_219464


namespace fraction_to_decimal_l219_219943

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 :=
by
  sorry

end fraction_to_decimal_l219_219943


namespace intersecting_lines_l219_219761

theorem intersecting_lines (a b c d : ℝ) (h₁ : a ≠ b) (h₂ : ∃ x y : ℝ, y = a*x + a ∧ y = b*x + b ∧ y = c*x + d) : c = d :=
sorry

end intersecting_lines_l219_219761


namespace parabola_condition_l219_219834

/-- Given the point (3,0) lies on the parabola y = 2x^2 + (k + 2)x - k,
    prove that k = -12. -/
theorem parabola_condition (k : ℝ) (h : 0 = 2 * 3^2 + (k + 2) * 3 - k) : k = -12 :=
by 
  sorry

end parabola_condition_l219_219834


namespace average_after_adding_ten_l219_219653

theorem average_after_adding_ten (avg initial_sum new_mean : ℕ) (n : ℕ) (h1 : n = 15) (h2 : avg = 40) (h3 : initial_sum = n * avg) (h4 : new_mean = (initial_sum + n * 10) / n) : new_mean = 50 := 
by
  sorry

end average_after_adding_ten_l219_219653


namespace solve_inequality_l219_219386

theorem solve_inequality {a b : ℝ} (h : -2 * a + 1 < -2 * b + 1) : a > b :=
by
  sorry

end solve_inequality_l219_219386


namespace clock_angle_at_8_20_is_130_degrees_l219_219198

/--
A clock has 12 hours, and each hour represents 30 degrees.
The minute hand moves 6 degrees per minute.
The hour hand moves 0.5 degrees per minute from its current hour position.
Prove that the smaller angle between the hour and minute hands at 8:20 p.m. is 130 degrees.
-/
theorem clock_angle_at_8_20_is_130_degrees
    (hours_per_clock : ℝ := 12)
    (degrees_per_hour : ℝ := 360 / hours_per_clock)
    (minutes_per_hour : ℝ := 60)
    (degrees_per_minute : ℝ := 360 / minutes_per_hour)
    (hour_slider_per_minute : ℝ := degrees_per_hour / minutes_per_hour)
    (minute_hand_at_20 : ℝ := 20 * degrees_per_minute)
    (hour_hand_at_8: ℝ := 8 * degrees_per_hour)
    (hour_hand_move_in_20_minutes : ℝ := 20 * hour_slider_per_minute)
    (hour_hand_at_8_20 : ℝ := hour_hand_at_8 + hour_hand_move_in_20_minutes) :
  |hour_hand_at_8_20 - minute_hand_at_20| = 130 :=
by
  sorry

end clock_angle_at_8_20_is_130_degrees_l219_219198


namespace divisibility_of_n_l219_219052

theorem divisibility_of_n (P : Polynomial ℤ) (k n : ℕ)
  (hk : k % 2 = 0)
  (h_odd_coeffs : ∀ i, i ≤ k → i % 2 = 1)
  (h_div : ∃ Q : Polynomial ℤ, (X + 1)^n - 1 = (P * Q)) :
  n % (k + 1) = 0 :=
sorry

end divisibility_of_n_l219_219052


namespace card_average_2023_l219_219215

theorem card_average_2023 (n : ℕ) (h_pos : 0 < n) (h_avg : (2 * n + 1) / 3 = 2023) : n = 3034 := by
  sorry

end card_average_2023_l219_219215


namespace man_work_days_l219_219532

variable (W : ℝ) -- Denoting the amount of work by W

-- Defining the work rate variables
variables (M Wm B : ℝ)

-- Conditions from the problem:
-- Combined work rate of man, woman, and boy together completes the work in 3 days
axiom combined_work_rate : M + Wm + B = W / 3
-- Woman completes the work alone in 18 days
axiom woman_work_rate : Wm = W / 18
-- Boy completes the work alone in 9 days
axiom boy_work_rate : B = W / 9

-- The goal is to prove the man takes 6 days to complete the work alone
theorem man_work_days : (W / M) = 6 :=
by
  sorry

end man_work_days_l219_219532


namespace walking_time_proof_l219_219442

-- Define the conditions from the problem
def bus_ride : ℕ := 75
def train_ride : ℕ := 360
def total_trip_time : ℕ := 480

-- Define the walking time as variable
variable (W : ℕ)

-- State the theorem as a Lean statement
theorem walking_time_proof :
  bus_ride + W + 2 * W + train_ride = total_trip_time → W = 15 :=
by
  intros h
  sorry

end walking_time_proof_l219_219442


namespace shadow_of_cube_l219_219682

theorem shadow_of_cube (x : ℝ) (h_edge : ∀ c : ℝ, c = 2) (h_shadow_area : ∀ a : ℝ, a = 200 + 4) :
  ⌊1000 * x⌋ = 12280 :=
by
  sorry

end shadow_of_cube_l219_219682


namespace greatest_integer_with_gcd_30_eq_5_l219_219485

theorem greatest_integer_with_gcd_30_eq_5 :
  ∃ n : ℕ, n < 200 ∧ gcd n 30 = 5 ∧ ∀ m : ℕ, m < 200 ∧ gcd m 30 = 5 → m ≤ n :=
begin
  let n := 195,
  use n,
  split,
  { sorry }, -- Proof that n < 200
  split,
  { sorry }, -- Proof that gcd n 30 = 5
  { sorry }  -- Proof that n is the greatest integer satisfying the conditions
end

end greatest_integer_with_gcd_30_eq_5_l219_219485


namespace last_three_digits_of_8_pow_108_l219_219702

theorem last_three_digits_of_8_pow_108 :
  (8^108 % 1000) = 38 := 
sorry

end last_three_digits_of_8_pow_108_l219_219702


namespace find_m_plus_n_l219_219941

noncomputable def vertices := [(10, 45), (10, 114), (28, 153), (28, 84)]

def midpoint (p1 p2 : ℤ × ℤ) : ℤ × ℤ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

def center := midpoint (10, 45) (28, 153)

def slope_through_origin (c : ℤ × ℤ) : ℚ :=
  (c.2 : ℚ) / c.1

theorem find_m_plus_n : (m n : ℕ) (h : m + n = 118) 
  (h_rel_prime : Int.gcd m n = 1) (slope := slope_through_origin center) :
  slope = m / n :=
  sorry

end find_m_plus_n_l219_219941


namespace left_vertex_of_ellipse_l219_219828

theorem left_vertex_of_ellipse : 
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧ (∀ (x y : ℝ), (x^2 / a^2) + (y^2 / b^2) = 1) ∧
  (∀ (x y : ℝ), x^2 + y^2 - 6*x + 8 = 0 ∧ x = a - 5) ∧
  2 * b = 8 → left_vertex = (-5, 0) :=
sorry

end left_vertex_of_ellipse_l219_219828


namespace stratified_sampling_second_year_students_l219_219355

theorem stratified_sampling_second_year_students 
  (total_athletes : ℕ) 
  (first_year_students : ℕ) 
  (sample_size : ℕ) 
  (second_year_students_in_sample : ℕ)
  (h1 : total_athletes = 98) 
  (h2 : first_year_students = 56) 
  (h3 : sample_size = 28)
  (h4 : second_year_students_in_sample = (42 * sample_size) / total_athletes) :
  second_year_students_in_sample = 4 := 
sorry

end stratified_sampling_second_year_students_l219_219355


namespace find_x4_y4_l219_219419

theorem find_x4_y4 (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 15) : x^4 + y^4 = 175 := by
  sorry

end find_x4_y4_l219_219419


namespace find_other_number_l219_219168

theorem find_other_number (lcm_ab : Nat) (gcd_ab : Nat) (a b : Nat) 
  (hlcm : Nat.lcm a b = lcm_ab) 
  (hgcd : Nat.gcd a b = gcd_ab) 
  (ha : a = 210) 
  (hlcm_ab : lcm_ab = 2310) 
  (hgcd_ab : gcd_ab = 55) 
  : b = 605 := 
by 
  sorry

end find_other_number_l219_219168


namespace range_of_m_l219_219278

def A (x : ℝ) : Prop := 1/2 < x ∧ x < 1

def B (x : ℝ) (m : ℝ) : Prop := x^2 + 2 * x + 1 - m ≤ 0

theorem range_of_m (m : ℝ) : (∀ x : ℝ, A x → B x m) → 4 ≤ m := by
  sorry

end range_of_m_l219_219278


namespace find_constant_l219_219849

theorem find_constant
  {x : ℕ} (f : ℕ → ℕ)
  (h1 : ∀ x, f x = x^2 + 2*x + c)
  (h2 : f 2 = 12) :
  c = 4 :=
by sorry

end find_constant_l219_219849


namespace simplify_fraction_l219_219618

theorem simplify_fraction (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) : 
  1 / x - 1 / (x - 1) = -1 / (x * (x - 1)) :=
by
  sorry

end simplify_fraction_l219_219618


namespace problem_solution_l219_219570

theorem problem_solution (x : ℕ) (h : 3^x + 3^x + 3^x + 3^x = 972) : (x + 2) * (x - 2) = 5 :=
by
  sorry

end problem_solution_l219_219570


namespace original_number_of_movies_l219_219344

theorem original_number_of_movies (x : ℕ) (dvd blu_ray : ℕ)
  (h1 : dvd = 17 * x)
  (h2 : blu_ray = 4 * x)
  (h3 : 17 * x / (4 * x - 4) = 9 / 2) :
  dvd + blu_ray = 378 := by
  sorry

end original_number_of_movies_l219_219344


namespace binomial_product_l219_219245

def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_product : binomial 10 3 * binomial 8 3 = 6720 := 
by 
  sorry

end binomial_product_l219_219245


namespace intersection_A_complement_B_l219_219407

open Set

noncomputable def A : Set ℝ := {2, 3, 4, 5, 6}
noncomputable def B : Set ℝ := {x | x^2 - 8 * x + 12 >= 0}
noncomputable def complement_B : Set ℝ := {x | 2 < x ∧ x < 6}

theorem intersection_A_complement_B :
  A ∩ complement_B = {3, 4, 5} :=
sorry

end intersection_A_complement_B_l219_219407


namespace cos_difference_simplification_l219_219147

theorem cos_difference_simplification :
  let x := Real.cos (20 * Real.pi / 180)
  let y := Real.cos (40 * Real.pi / 180)
  x - y = -1 / (2 * Real.sqrt 5) :=
sorry

end cos_difference_simplification_l219_219147


namespace integer_mod_105_l219_219623

theorem integer_mod_105 (x : ℤ) :
  (4 + x ≡ 2 * 2 [ZMOD 3^3]) →
  (6 + x ≡ 3 * 3 [ZMOD 5^3]) →
  (8 + x ≡ 5 * 5 [ZMOD 7^3]) →
  x % 105 = 3 :=
by
  sorry

end integer_mod_105_l219_219623


namespace ammonium_bromide_total_weight_l219_219044

noncomputable def nitrogen_weight : ℝ := 14.01
noncomputable def hydrogen_weight : ℝ := 1.01
noncomputable def bromine_weight : ℝ := 79.90
noncomputable def ammonium_bromide_weight : ℝ := nitrogen_weight + 4 * hydrogen_weight + bromine_weight
noncomputable def moles : ℝ := 5
noncomputable def total_weight : ℝ := moles * ammonium_bromide_weight

theorem ammonium_bromide_total_weight :
  total_weight = 489.75 :=
by
  -- The proof is omitted.
  sorry

end ammonium_bromide_total_weight_l219_219044


namespace quadratic_real_roots_range_l219_219979

theorem quadratic_real_roots_range (m : ℝ) : 
(m - 1) * x^2 - 2 * x + 1 = 0 → (m ≤ 2 ∧ m ≠ 1) :=
begin
  sorry
end

end quadratic_real_roots_range_l219_219979


namespace homework_problems_l219_219797

theorem homework_problems (p t : ℕ) (h1 : p >= 10) (h2 : pt = (2 * p + 2) * (t + 1)) : p * t = 60 :=
by
  sorry

end homework_problems_l219_219797


namespace frac_eq_l219_219417

theorem frac_eq (x : ℝ) (h : 3 - 9 / x + 6 / x^2 = 0) : 2 / x = 1 ∨ 2 / x = 2 := 
by 
  sorry

end frac_eq_l219_219417


namespace cos_sin_eq_l219_219399

theorem cos_sin_eq (x : ℝ) (h : Real.cos x - 3 * Real.sin x = 2) :
  (Real.sin x + 3 * Real.cos x = (2 * Real.sqrt 6 - 3) / 5) ∨
  (Real.sin x + 3 * Real.cos x = -(2 * Real.sqrt 6 + 3) / 5) := 
by
  sorry

end cos_sin_eq_l219_219399


namespace find_ab_l219_219897

theorem find_ab (a b : ℤ) :
  (∀ x : ℤ, x^3 + a * x^2 + b * x + 5 % (x - 1) = 7) ∧ (∀ x : ℤ, x^3 + a * x^2 + b * x + 5 % (x + 1) = 9) →
  (a, b) = (3, -2) := 
by
  sorry

end find_ab_l219_219897


namespace sqrt_of_sixteen_l219_219030

theorem sqrt_of_sixteen (x : ℝ) (h : x^2 = 16) : x = 4 ∨ x = -4 := 
sorry

end sqrt_of_sixteen_l219_219030


namespace find_b_l219_219348

theorem find_b
  (a b c d : ℝ)
  (h₁ : -a + b - c + d = 0)
  (h₂ : a + b + c + d = 0)
  (h₃ : d = 2) :
  b = -2 := 
by 
  sorry

end find_b_l219_219348


namespace ball_distribution_l219_219320

theorem ball_distribution : 
  let ways := (∃ x y z : ℕ, x + y + z = 20 ∧ x ≥ 1 ∧ y ≥ 2 ∧ z ≥ 3) in 
  ways = (∑ i in (Finset.range (20 - 1 + 1)).image ((· + 1 : ℕ → ℕ) ∘ Finset.filter (λ n => n ≥ 2).card), 1) :=
begin
  sorry
end

end ball_distribution_l219_219320


namespace train_speed_l219_219630

theorem train_speed (distance_AB : ℕ) (start_time_A : ℕ) (start_time_B : ℕ) (meet_time : ℕ) (speed_B : ℕ) (time_travel_A : ℕ) (time_travel_B : ℕ)
  (total_distance : ℕ) (distance_B_covered : ℕ) (speed_A : ℕ)
  (h1 : distance_AB = 330)
  (h2 : start_time_A = 8)
  (h3 : start_time_B = 9)
  (h4 : meet_time = 11)
  (h5 : speed_B = 75)
  (h6 : time_travel_A = meet_time - start_time_A)
  (h7 : time_travel_B = meet_time - start_time_B)
  (h8 : distance_B_covered = time_travel_B * speed_B)
  (h9 : total_distance = distance_AB)
  (h10 : total_distance = time_travel_A * speed_A + distance_B_covered):
  speed_A = 60 := 
by
  sorry

end train_speed_l219_219630


namespace rectangle_area_integer_length_width_l219_219362

theorem rectangle_area_integer_length_width (l w : ℕ) (h1 : w = l / 2) (h2 : 2 * l + 2 * w = 200) :
  l * w = 2178 :=
by
  sorry

end rectangle_area_integer_length_width_l219_219362


namespace div_by_27_l219_219004

theorem div_by_27 (n : ℕ) : 27 ∣ (10^n + 18 * n - 1) :=
sorry

end div_by_27_l219_219004


namespace rosa_total_pages_called_l219_219841

variable (P_last P_this : ℝ)

theorem rosa_total_pages_called (h1 : P_last = 10.2) (h2 : P_this = 8.6) : P_last + P_this = 18.8 :=
by sorry

end rosa_total_pages_called_l219_219841


namespace nontrivial_solution_exists_l219_219439

theorem nontrivial_solution_exists 
  (a b : ℤ) 
  (h_square_a : ∀ k : ℤ, a ≠ k^2) 
  (h_square_b : ∀ k : ℤ, b ≠ k^2) 
  (h_nontrivial : ∃ (x y z w : ℤ), x^2 - a * y^2 - b * z^2 + a * b * w^2 = 0 ∧ (x, y, z, w) ≠ (0, 0, 0, 0)) : 
  ∃ (x y z : ℤ), x^2 - a * y^2 - b * z^2 = 0 ∧ (x, y, z) ≠ (0, 0, 0) :=
by
  sorry

end nontrivial_solution_exists_l219_219439


namespace max_min_f_l219_219021

noncomputable def f (x : ℝ) : ℝ :=
  if 6 ≤ x ∧ x ≤ 8 then
    (Real.sqrt (8 * x - x^2) - Real.sqrt (114 * x - x^2 - 48))
  else
    0

theorem max_min_f :
  ∀ x, 6 ≤ x ∧ x ≤ 8 → f x ≤ 2 * Real.sqrt 3 ∧ 0 ≤ f x :=
by
  intros
  sorry

end max_min_f_l219_219021


namespace satisify_absolute_value_inequality_l219_219415

theorem satisify_absolute_value_inequality :
  ∃ (t : Finset ℤ), t.card = 2 ∧ ∀ y ∈ t, |7 * y + 4| ≤ 10 :=
by
  sorry

end satisify_absolute_value_inequality_l219_219415


namespace common_ratio_of_geometric_seq_l219_219932

theorem common_ratio_of_geometric_seq (a b c d : ℤ) (h1 : a = 25)
    (h2 : b = -50) (h3 : c = 100) (h4 : d = -200)
    (h_geo_1 : b = a * -2)
    (h_geo_2 : c = b * -2)
    (h_geo_3 : d = c * -2) : 
    let r := (-2 : ℤ) in r = -2 := 
by 
  sorry

end common_ratio_of_geometric_seq_l219_219932


namespace analytical_expression_range_of_t_l219_219087

noncomputable def f (x : ℝ) : ℝ := x^2 - 3 * x

theorem analytical_expression (x : ℝ) :
  (f (x + 1) - f x = 2 * x - 2) ∧ (f 1 = -2) :=
by
  sorry

theorem range_of_t (t : ℝ) :
  (∀ x : ℝ, f x > 0 ∧ f (x + t) < 0 → x = 1) ↔ (-2 <= t ∧ t < -1) :=
by
  sorry

end analytical_expression_range_of_t_l219_219087


namespace total_pairs_of_jeans_purchased_l219_219566

-- Definitions based on the problem conditions
def price_fox : ℝ := 15
def price_pony : ℝ := 18
def discount_save : ℝ := 8.64
def pairs_fox : ℕ := 3
def pairs_pony : ℕ := 2
def sum_discount_rate : ℝ := 0.22
def discount_rate_pony : ℝ := 0.13999999999999993

-- Lean 4 statement to prove the total number of pairs of jeans purchased
theorem total_pairs_of_jeans_purchased :
  pairs_fox + pairs_pony = 5 :=
by
  sorry

end total_pairs_of_jeans_purchased_l219_219566


namespace total_fruit_weight_l219_219483

-- Definitions for the conditions
def mario_ounces : ℕ := 8
def lydia_ounces : ℕ := 24
def nicolai_pounds : ℕ := 6
def ounces_per_pound : ℕ := 16

-- Theorem statement
theorem total_fruit_weight : 
  ((mario_ounces / ounces_per_pound : ℚ) + 
   (lydia_ounces / ounces_per_pound : ℚ) + 
   (nicolai_pounds : ℚ)) = 8 := 
sorry

end total_fruit_weight_l219_219483


namespace least_number_added_1789_l219_219042

def least_number_added_to_divisible (n d : ℕ) : ℕ := d - (n % d)

theorem least_number_added_1789 :
  least_number_added_to_divisible 1789 (Nat.lcm (Nat.lcm 5 6) (Nat.lcm 4 3)) = 11 :=
by
  -- Step definitions
  have lcm_5_6 := Nat.lcm 5 6
  have lcm_4_3 := Nat.lcm 4 3
  have lcm_total := Nat.lcm lcm_5_6 lcm_4_3
  -- Computation of the final result
  have remainder := 1789 % lcm_total
  have required_add := lcm_total - remainder
  -- Conclusion based on the computed values
  sorry

end least_number_added_1789_l219_219042


namespace calculate_large_exponent_l219_219235

theorem calculate_large_exponent : (1307 * 1307)^3 = 4984209203082045649 :=
by {
   sorry
}

end calculate_large_exponent_l219_219235


namespace prove_op_eq_l219_219252

-- Define the new operation ⊕
def op (x y : ℝ) := x^3 - 2*y + x

-- State that for any k, k ⊕ (k ⊕ k) = -k^3 + 3k
theorem prove_op_eq (k : ℝ) : op k (op k k) = -k^3 + 3*k :=
by 
  sorry

end prove_op_eq_l219_219252


namespace calculate_gf3_l219_219438

def f (x : ℕ) : ℕ := x^3 - 1
def g (x : ℕ) : ℕ := 3 * x^2 + x + 2

theorem calculate_gf3 : g (f 3) = 2056 := by
  sorry

end calculate_gf3_l219_219438


namespace sum_1026_is_2008_l219_219839

def sequence_sum (n : ℕ) : ℕ :=
  if n = 0 then 0
  else
    let groups_sum : ℕ := (n * n)
    let extra_2s := (2008 - groups_sum) / 2
    (n * (n + 1)) / 2 + extra_2s

theorem sum_1026_is_2008 : sequence_sum 1026 = 2008 :=
  sorry

end sum_1026_is_2008_l219_219839


namespace greatest_integer_with_gcd_30_eq_5_l219_219487

theorem greatest_integer_with_gcd_30_eq_5 :
  ∃ n : ℕ, n < 200 ∧ gcd n 30 = 5 ∧ ∀ m : ℕ, m < 200 ∧ gcd m 30 = 5 → m ≤ n :=
begin
  let n := 195,
  use n,
  split,
  { sorry }, -- Proof that n < 200
  split,
  { sorry }, -- Proof that gcd n 30 = 5
  { sorry }  -- Proof that n is the greatest integer satisfying the conditions
end

end greatest_integer_with_gcd_30_eq_5_l219_219487


namespace polynomial_difference_square_l219_219440

theorem polynomial_difference_square (a : Fin 11 → ℝ) (x : ℝ) (sqrt2 : ℝ)
  (h_eq : (sqrt2 - x)^10 = a 0 + a 1 * x + a 2 * x^2 + a 3 * x^3 + a 4 * x^4 + a 5 * x^5 + 
          a 6 * x^6 + a 7 * x^7 + a 8 * x^8 + a 9 * x^9 + a 10 * x^10) : 
  ((a 0 + a 2 + a 4 + a 6 + a 8 + a 10)^2 - (a 1 + a 3 + a 5 + a 7 + a 9)^2 = 1) :=
by
  sorry

end polynomial_difference_square_l219_219440


namespace complex_number_solution_l219_219922

theorem complex_number_solution (z : ℂ) (i : ℂ) (h : i * z = 1) : z = -i :=
by sorry

end complex_number_solution_l219_219922


namespace integer_root_count_l219_219287

theorem integer_root_count (b : ℝ) :
  (∃ r s : ℤ, r + s = b ∧ r * s = 8 * b) ↔
  b = -9 ∨ b = 0 ∨ b = 9 :=
sorry

end integer_root_count_l219_219287


namespace initial_birds_179_l219_219622

theorem initial_birds_179 (B : ℕ) (h1 : B + 38 = 217) : B = 179 :=
sorry

end initial_birds_179_l219_219622


namespace find_numbers_l219_219638

theorem find_numbers (A B: ℕ) (h1: A + B = 581) (h2: (Nat.lcm A B) / (Nat.gcd A B) = 240) : 
  (A = 560 ∧ B = 21) ∨ (A = 21 ∧ B = 560) :=
by
  sorry

end find_numbers_l219_219638


namespace sum_of_squares_l219_219172

theorem sum_of_squares (x y : ℝ) (h1 : x * y = 120) (h2 : x + y = 23) : x^2 + y^2 = 289 :=
sorry

end sum_of_squares_l219_219172


namespace largest_possible_b_l219_219337

theorem largest_possible_b (a b c : ℕ) (h1 : 1 < c) (h2 : c < b) (h3 : b < a) (h4 : a * b * c = 360) : b ≤ 12 :=
by
  sorry

end largest_possible_b_l219_219337


namespace total_students_l219_219234

theorem total_students (T : ℝ) (h1 : 0.3 * T =  0.7 * T - 616) : T = 880 :=
by sorry

end total_students_l219_219234


namespace Tom_green_marbles_l219_219007

-- Define the given variables
def Sara_green_marbles : Nat := 3
def Total_green_marbles : Nat := 7

-- The statement to be proven
theorem Tom_green_marbles : (Total_green_marbles - Sara_green_marbles) = 4 := by
  sorry

end Tom_green_marbles_l219_219007


namespace volleyballs_basketballs_difference_l219_219224

variable (V B : ℕ)

theorem volleyballs_basketballs_difference :
  (V + B = 14) →
  (4 * V + 5 * B = 60) →
  V - B = 6 :=
by
  intros h1 h2
  sorry

end volleyballs_basketballs_difference_l219_219224


namespace chlorine_moles_l219_219962

theorem chlorine_moles (methane_used chlorine_used chloromethane_formed : ℕ)
  (h_combined_methane : methane_used = 3)
  (h_formed_chloromethane : chloromethane_formed = 3)
  (balanced_eq : methane_used = chloromethane_formed) :
  chlorine_used = 3 :=
by
  have h : chlorine_used = methane_used := by sorry
  rw [h_combined_methane] at h
  exact h

end chlorine_moles_l219_219962


namespace regular_polygon_sides_l219_219697

theorem regular_polygon_sides (N : ℕ) (h : ∀ θ, θ = 140 → N * (180 -θ) = 360) : N = 9 :=
by
  sorry

end regular_polygon_sides_l219_219697


namespace lcm_18_20_l219_219961

theorem lcm_18_20 : Nat.lcm 18 20 = 180 := by
  sorry

end lcm_18_20_l219_219961


namespace min_value_of_reciprocal_sum_l219_219970

theorem min_value_of_reciprocal_sum (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 * x + y = 1) :
  ∃ z, (z = 3 + 2 * Real.sqrt 2) ∧ (∀ z', (z' = 1 / x + 1 / y) → z ≤ z') :=
sorry

end min_value_of_reciprocal_sum_l219_219970


namespace book_width_l219_219116

noncomputable def golden_ratio : Real := (1 + Real.sqrt 5) / 2

theorem book_width (length : Real) (width : Real) 
(h1 : length = 20) 
(h2 : width / length = golden_ratio) : 
width = 12.36 := 
by 
  sorry

end book_width_l219_219116


namespace acres_used_for_corn_l219_219674

theorem acres_used_for_corn (total_acres : ℕ) (beans_ratio : ℕ) (wheat_ratio : ℕ) (corn_ratio : ℕ) :
  total_acres = 1034 → beans_ratio = 5 → wheat_ratio = 2 → corn_ratio = 4 →
  let total_parts := beans_ratio + wheat_ratio + corn_ratio in
  let acres_per_part := total_acres / total_parts in
  let corn_acres := acres_per_part * corn_ratio in
  corn_acres = 376 :=
by
  intros
  let total_parts := beans_ratio + wheat_ratio + corn_ratio
  let acres_per_part := total_acres / total_parts
  let corn_acres := acres_per_part * corn_ratio
  show corn_acres = 376
  sorry

end acres_used_for_corn_l219_219674


namespace min_max_values_l219_219468

noncomputable def f (x : ℝ) : ℝ := cos x + (x + 1) * sin x + 1

theorem min_max_values : 
  ∃ (min_val max_val : ℝ), 
  min_val = -3 * Real.pi / 2 ∧ 
  max_val = Real.pi / 2 + 2 ∧ 
  (∀ x ∈ Icc (0 : ℝ) (2 * Real.pi), f x ≥ min_val) ∧ 
  (∀ x ∈ Icc (0 : ℝ) (2 * Real.pi), f x ≤ max_val) ∧ 
  (∃ x ∈ Icc (0 : ℝ) (2 * Real.pi), f x = min_val) ∧ 
  (∃ x ∈ Icc (0 : ℝ) (2 * Real.pi), f x = max_val) := 
by
  sorry

end min_max_values_l219_219468


namespace difference_between_max_and_min_34_l219_219463

theorem difference_between_max_and_min_34 
  (A B C D E: ℕ) 
  (h_avg: (A + B + C + D + E) / 5 = 50) 
  (h_max: E ≤ 58) 
  (h_distinct: A < B ∧ B < C ∧ C < D ∧ D < E) 
: E - A = 34 := 
sorry

end difference_between_max_and_min_34_l219_219463


namespace original_selling_price_l219_219885

theorem original_selling_price (P : ℝ) (d1 d2 d3 t : ℝ) (final_price : ℝ) :
  d1 = 0.32 → -- first discount
  d2 = 0.10 → -- loyalty discount
  d3 = 0.05 → -- holiday discount
  t = 0.15 → -- state tax
  final_price = 650 → 
  1.15 * P * (1 - d1) * (1 - d2) * (1 - d3) = final_price →
  P = 722.57 :=
sorry

end original_selling_price_l219_219885


namespace distance_between_x_intercepts_l219_219220

-- Definitions for the conditions
def line_eq (m : ℝ) (x1 y1 : ℝ) : ℝ → ℝ := λ x, m * (x - x1) + y1

def x_intercept (m : ℝ) (x1 y1 : ℝ) : ℝ := classical.some (exists_x_intercept m x1 y1)

lemma exists_x_intercept (m x1 y1 : ℝ) : ∃ x : ℝ, line_eq m x1 y1 x = 0 :=
begin
  use (y1 / m + x1),
  simp [line_eq]
end

-- Main theorem
theorem distance_between_x_intercepts : 
  let l1 := line_eq 4 8 20,
      l2 := line_eq (-3) 8 20 in
  dist (x_intercept 4 8 20) (x_intercept (-3) 8 20) = 35 / 3 :=
by {
  -- Define equations of the lines
  let line1 := (λ x, 4 * (x - 8) + 20),
  let line2 := (λ x, -3 * (x - 8) + 20),

  -- Compute x-intercepts
  let x_int1 := classical.some (exists_x_intercept 4 8 20),
  let x_int2 := classical.some (exists_x_intercept (-3) 8 20),

  -- The derivation of x-intercepts and the distance calculation should be skipped for now
  sorry
}

end distance_between_x_intercepts_l219_219220


namespace find_x_y_sum_l219_219389

theorem find_x_y_sum :
  ∃ (x y : ℕ), 0 < x ∧ 0 < y ∧ (∃ (a b : ℕ), 360 * x = a^2 ∧ 360 * y = b^4) ∧ x + y = 2260 :=
by {
  sorry
}

end find_x_y_sum_l219_219389


namespace empty_set_negation_l219_219896

open Set

theorem empty_set_negation (α : Type) : ¬ (∀ s : Set α, ∅ ⊆ s) ↔ (∃ s : Set α, ¬(∅ ⊆ s)) :=
by
  sorry

end empty_set_negation_l219_219896


namespace adam_spent_on_ferris_wheel_l219_219068

theorem adam_spent_on_ferris_wheel (t_initial t_left t_price : ℕ) (h1 : t_initial = 13)
  (h2 : t_left = 4) (h3 : t_price = 9) : t_initial - t_left = 9 ∧ (t_initial - t_left) * t_price = 81 := 
by
  sorry

end adam_spent_on_ferris_wheel_l219_219068


namespace arithmetic_sqrt_9_l219_219459

theorem arithmetic_sqrt_9 :
  (∃ y : ℝ, y * y = 9 ∧ y ≥ 0) → (∃ y : ℝ, y = 3) :=
by
  sorry

end arithmetic_sqrt_9_l219_219459


namespace total_subjects_l219_219446

theorem total_subjects (subjects_monica subjects_marius subjects_millie : ℕ)
  (h1 : subjects_monica = 10)
  (h2 : subjects_marius = subjects_monica + 4)
  (h3 : subjects_millie = subjects_marius + 3) :
  subjects_monica + subjects_marius + subjects_millie = 41 :=
by
  sorry

end total_subjects_l219_219446


namespace solve_abs_eqn_l219_219751

theorem solve_abs_eqn (y : ℝ) : (|y - 4| + 3 * y = 11) ↔ (y = 3.5) := by
  sorry

end solve_abs_eqn_l219_219751


namespace parametric_to_cartesian_l219_219375

theorem parametric_to_cartesian (θ : ℝ) (x y : ℝ) :
  (x = 1 + 2 * Real.cos θ) →
  (y = 2 * Real.sin θ) →
  (x - 1) ^ 2 + y ^ 2 = 4 :=
by 
  sorry

end parametric_to_cartesian_l219_219375


namespace factorize_x4_plus_16_l219_219385

theorem factorize_x4_plus_16 :
  ∀ x : ℝ, (x^4 + 16) = (x^2 - 2 * x + 2) * (x^2 + 2 * x + 2) :=
by
  intro x
  sorry

end factorize_x4_plus_16_l219_219385


namespace necessary_but_not_sufficient_l219_219477

-- Define the necessary conditions
variables {a b c d : ℝ}

-- State the main theorem
theorem necessary_but_not_sufficient (h₁ : a > b) (h₂ : c > d) : (a + c > b + d) :=
by
  -- Placeholder for the proof (insufficient as per the context problem statement)
  sorry

end necessary_but_not_sufficient_l219_219477


namespace sufficient_not_necessary_l219_219921

theorem sufficient_not_necessary (x : ℝ) : (x - 1 > 0) → (x^2 - 1 > 0) ∧ ¬((x^2 - 1 > 0) → (x - 1 > 0)) :=
by
  sorry

end sufficient_not_necessary_l219_219921


namespace apples_in_box_ratio_mixed_fruits_to_total_l219_219641

variable (total_fruits : Nat) (oranges : Nat) (peaches : Nat) (apples : Nat) (mixed_fruits : Nat)
variable (one_fourth_of_box_contains_oranges : oranges = total_fruits / 4)
variable (half_as_many_peaches_as_oranges : peaches = oranges / 2)
variable (five_times_as_many_apples_as_peaches : apples = 5 * peaches)
variable (mixed_fruits_double_peaches : mixed_fruits = 2 * peaches)
variable (total_fruits_56 : total_fruits = 56)

theorem apples_in_box : apples = 35 := by
  sorry

theorem ratio_mixed_fruits_to_total : mixed_fruits / total_fruits = 1 / 4 := by
  sorry

end apples_in_box_ratio_mixed_fruits_to_total_l219_219641


namespace total_seats_l219_219367

theorem total_seats (F : ℕ) 
  (h1 : 305 = 4 * F + 2) 
  (h2 : 310 = 4 * F + 2) : 
  310 + F = 387 :=
by
  sorry

end total_seats_l219_219367


namespace solve_total_rainfall_l219_219853

def rainfall_2010 : ℝ := 50.0
def increase_2011 : ℝ := 3.0
def increase_2012 : ℝ := 4.0

def monthly_rainfall_2011 : ℝ := rainfall_2010 + increase_2011
def monthly_rainfall_2012 : ℝ := monthly_rainfall_2011 + increase_2012

def total_rainfall_2011 : ℝ := monthly_rainfall_2011 * 12
def total_rainfall_2012 : ℝ := monthly_rainfall_2012 * 12

def total_rainfall_2011_2012 : ℝ := total_rainfall_2011 + total_rainfall_2012

theorem solve_total_rainfall :
  total_rainfall_2011_2012 = 1320.0 :=
sorry

end solve_total_rainfall_l219_219853


namespace shorter_piece_is_20_l219_219520

def shorter_piece_length (total_length : ℕ) (ratio : ℚ) (shorter_piece : ℕ) : Prop :=
    shorter_piece * 7 = 2 * (total_length - shorter_piece)

theorem shorter_piece_is_20 : ∀ (total_length : ℕ) (shorter_piece : ℕ), 
    total_length = 90 ∧
    shorter_piece_length total_length (2/7 : ℚ) shorter_piece ->
    shorter_piece = 20 :=
by
  intro total_length shorter_piece
  intro h
  have h_total_length : total_length = 90 := h.1
  have h_equation : shorter_piece_length total_length (2/7 : ℚ) shorter_piece := h.2
  sorry

end shorter_piece_is_20_l219_219520


namespace part_a_part_b_l219_219923

-- Part (a)
theorem part_a (n : ℕ) (a b : ℝ) : 
  a^(n+1) + b^(n+1) = (a + b) * (a^n + b^n) - a * b * (a^(n - 1) + b^(n - 1)) :=
by sorry

-- Part (b)
theorem part_b {a b : ℝ} (h1 : a + b = 1) (h2: a * b = -1) : 
  a^10 + b^10 = 123 :=
by sorry

end part_a_part_b_l219_219923


namespace exists_sequence_a_l219_219701

-- Define the sequence and properties
def sequence_a (a : ℕ → ℤ) : Prop :=
  a 1 = 1 ∧
  a 2 = 2 ∧
  a 18 = 2019 ∧
  ∀ k, 3 ≤ k → k ≤ 18 → ∃ i j, 1 ≤ i → i < j → j < k → a k = a i + a j

-- The main theorem statement
theorem exists_sequence_a : ∃ (a : ℕ → ℤ), sequence_a a := 
sorry

end exists_sequence_a_l219_219701


namespace concert_ratio_l219_219019

theorem concert_ratio (a c : ℕ) (h1 : 30 * a + 15 * c = 2250) (h2 : a ≥ 1) (h3 : c ≥ 1) :
  a = 50 ∧ c = 50 ∧ a = c := 
sorry

end concert_ratio_l219_219019


namespace sum_of_digits_of_x_squared_l219_219676

theorem sum_of_digits_of_x_squared (p q r : ℕ) (x : ℕ) 
  (h1 : r ≤ 400)
  (h2 : 7 * q = 17 * p)
  (h3 : x = p * r^3 + p * r^2 + q * r + q)
  (h4 : ∃ (a b c : ℕ), x^2 = a * r^6 + b * r^5 + c * r^4 + 0 * r^3 + c * r^2 + b * r + a) :
  (∑ i in (x^2).digits r, id) = 400 :=
by
  sorry

end sum_of_digits_of_x_squared_l219_219676


namespace average_rate_of_reduction_l219_219523

theorem average_rate_of_reduction
  (original_price final_price : ℝ)
  (h1 : original_price = 200)
  (h2 : final_price = 128)
  : ∃ (x : ℝ), 0 ≤ x ∧ x < 1 ∧ 200 * (1 - x) * (1 - x) = 128 :=
by
  sorry

end average_rate_of_reduction_l219_219523


namespace total_cost_is_correct_l219_219772

-- Define the number of total tickets and the number of children's tickets
def total_tickets : ℕ := 21
def children_tickets : ℕ := 16
def adult_tickets : ℕ := total_tickets - children_tickets

-- Define the cost of tickets for adults and children
def cost_per_adult_ticket : ℝ := 5.50
def cost_per_child_ticket : ℝ := 3.50

-- Define the total cost spent
def total_cost_spent : ℝ :=
  (adult_tickets * cost_per_adult_ticket) + (children_tickets * cost_per_child_ticket)

-- Prove that the total amount spent on tickets is $83.50
theorem total_cost_is_correct : total_cost_spent = 83.50 := by
  sorry

end total_cost_is_correct_l219_219772


namespace diamonds_G20_l219_219695

def diamonds_in_figure (n : ℕ) : ℕ :=
if n = 1 then 1 else 4 * n^2 + 4 * n - 7

theorem diamonds_G20 : diamonds_in_figure 20 = 1673 :=
by sorry

end diamonds_G20_l219_219695


namespace find_p_q_sum_l219_219390

-- Define the conditions
def p (q : ℤ) : ℤ := q + 20

theorem find_p_q_sum (p q : ℤ) (hp : p * q = 1764) (hq : p - q = 20) :
  p + q = 86 :=
  sorry

end find_p_q_sum_l219_219390


namespace factorize_polynomial_l219_219259

theorem factorize_polynomial (x y : ℝ) : x^3 - 2 * x^2 * y + x * y^2 = x * (x - y)^2 := 
by 
  sorry

end factorize_polynomial_l219_219259


namespace total_questions_l219_219225

theorem total_questions (S C I : ℕ) (h1 : S = 73) (h2 : C = 91) (h3 : S = C - 2 * I) : C + I = 100 :=
sorry

end total_questions_l219_219225


namespace ratio_a6_b6_l219_219967

-- Definitions for sequences and sums
variable {α : Type*} [LinearOrderedField α] 
variable (a b : ℕ → α) 
variable (S T : ℕ → α)

-- Main theorem stating the problem
theorem ratio_a6_b6 (h : ∀ n, S n / T n = (2 * n - 5) / (4 * n + 3)) :
    a 6 / b 6 = 17 / 47 :=
sorry

end ratio_a6_b6_l219_219967


namespace solve_xy_eq_yx_l219_219012

theorem solve_xy_eq_yx (x y : ℕ) (hxy : x ≠ y) : x^y = y^x ↔ ((x = 2 ∧ y = 4) ∨ (x = 4 ∧ y = 2)) :=
by
  sorry

end solve_xy_eq_yx_l219_219012


namespace f1_min_max_f2_min_max_l219_219703

-- Define the first function and assert its max and min values
def f1 (x : ℝ) : ℝ := x^3 + 2 * x

theorem f1_min_max : ∀ x ∈ Set.Icc (-1 : ℝ) 1,
  (∃ x_min x_max, x_min = -1 ∧ x_max = 1 ∧ f1 x_min = -3 ∧ f1 x_max = 3) := by
  sorry

-- Define the second function and assert its max and min values
def f2 (x : ℝ) : ℝ := (x - 1) * (x - 2)^2

theorem f2_min_max : ∀ x ∈ Set.Icc (0 : ℝ) 3,
  (∃ x_min x_max, x_min = 0 ∧ x_max = 3 ∧ (f2 x_min = -4) ∧ f2 x_max = 2) := by
  sorry

end f1_min_max_f2_min_max_l219_219703


namespace original_price_of_sarees_l219_219899

theorem original_price_of_sarees 
  (P : ℝ) 
  (h1 : 0.72 * P = 144) : 
  P = 200 := 
sorry

end original_price_of_sarees_l219_219899


namespace rectangle_fitting_condition_l219_219003

variables {a b c d : ℝ}

theorem rectangle_fitting_condition
  (h1: a < c ∧ c ≤ d ∧ d < b)
  (h2: a * b < c * d) :
  (b^2 - a^2)^2 ≤ (b*c - a*d)^2 + (b*d - a*c)^2 :=
sorry

end rectangle_fitting_condition_l219_219003


namespace triangle_inscribed_and_arcs_l219_219205

theorem triangle_inscribed_and_arcs
  (PQ QR PR : ℝ) (X Y Z : ℝ)
  (QY XZ QX YZ PX RY : ℝ)
  (H1 : PQ = 26)
  (H2 : QR = 28) 
  (H3 : PR = 27)
  (H4 : QY = XZ)
  (H5 : QX = YZ)
  (H6 : PX = RY)
  (H7 : RY = PX + 1)
  (H8 : XZ = QX + 1)
  (H9 : QY = YZ + 2) :
  QX = 29 / 2 :=
by
  sorry

end triangle_inscribed_and_arcs_l219_219205


namespace greatest_integer_with_gcf_5_l219_219489

theorem greatest_integer_with_gcf_5 :
  ∃ n, n < 200 ∧ gcd n 30 = 5 ∧ ∀ m, m < 200 ∧ gcd m 30 = 5 → m ≤ n :=
by
  sorry

end greatest_integer_with_gcf_5_l219_219489


namespace quadratic_has_real_roots_l219_219978

theorem quadratic_has_real_roots (m : ℝ) : (∃ x : ℝ, (m - 1) * x^2 - 2 * x + 1 = 0) ↔ (m ≤ 2 ∧ m ≠ 1) := 
by 
  sorry

end quadratic_has_real_roots_l219_219978


namespace sum_series_eq_one_third_l219_219810

theorem sum_series_eq_one_third :
  ∑' n : ℕ, (if h : n > 0 then (2^n / (1 + 2^n + 2^(n + 1) + 2^(2 * n + 1))) else 0) = 1 / 3 :=
by
  sorry

end sum_series_eq_one_third_l219_219810


namespace proof_problem_l219_219408

open Set

variable {R : Set ℝ} (A B : Set ℝ) (complement_B : Set ℝ)

-- Defining set A
def setA : Set ℝ := { x | 1 < x ∧ x < 3 }

-- Defining set B based on the given functional relationship
def setB : Set ℝ := { x | 2 < x } 

-- Defining the complement of set B (in the universal set R)
def complementB : Set ℝ := { x | x ≤ 2 }

-- The intersection we need to prove is equivalent to the given answer
def intersection_result : Set ℝ := { x | 1 < x ∧ x ≤ 2 }

-- The theorem statement (no proof)
theorem proof_problem : setA ∩ complementB = intersection_result := 
by
  sorry

end proof_problem_l219_219408


namespace walt_total_invested_l219_219141

-- Given Conditions
def invested_at_seven : ℝ := 5500
def total_interest : ℝ := 970
def interest_rate_seven : ℝ := 0.07
def interest_rate_nine : ℝ := 0.09

-- Define the total amount invested
noncomputable def total_invested : ℝ := 12000

-- Prove the total amount invested
theorem walt_total_invested :
  interest_rate_seven * invested_at_seven + interest_rate_nine * (total_invested - invested_at_seven) = total_interest :=
by
  -- The proof goes here
  sorry

end walt_total_invested_l219_219141


namespace f_positive_when_a_1_f_negative_solution_sets_l219_219964

section

variable (f : ℝ → ℝ) (a x : ℝ)

def f_def := f x = (x - a) * (x - 2)

-- (Ⅰ) Problem statement
theorem f_positive_when_a_1 : (∀ x, f_def f 1 x → f x > 0 ↔ (x < 1) ∨ (x > 2)) :=
by sorry

-- (Ⅱ) Problem statement
theorem f_negative_solution_sets (a : ℝ) : 
  (∀ x, f_def f a x ∧ a = 2 → False) ∧ 
  (∀ x, f_def f a x ∧ a > 2 → 2 < x ∧ x < a) ∧ 
  (∀ x, f_def f a x ∧ a < 2 → a < x ∧ x < 2) :=
by sorry

end

end f_positive_when_a_1_f_negative_solution_sets_l219_219964


namespace types_of_problems_l219_219371

def bill_problems : ℕ := 20
def ryan_problems : ℕ := 2 * bill_problems
def frank_problems : ℕ := 3 * ryan_problems
def problems_per_type : ℕ := 30

theorem types_of_problems : (frank_problems / problems_per_type) = 4 := by
  sorry

end types_of_problems_l219_219371


namespace simple_interest_rate_l219_219907

theorem simple_interest_rate 
  (SI : ℝ) (P : ℝ) (T : ℝ) (SI_eq : SI = 260)
  (P_eq : P = 910) (T_eq : T = 4)
  (H : SI = P * R * T / 100) : 
  R = 26000 / 3640 := 
by
  sorry

end simple_interest_rate_l219_219907


namespace ab_value_l219_219736

theorem ab_value (a b c : ℤ) (h1 : a^2 = 16) (h2 : 2 * a * b = -40) : a * b = -20 := 
sorry

end ab_value_l219_219736


namespace wage_difference_seven_l219_219050

-- Define the parameters and conditions
variables (P Q h : ℝ)

-- Given conditions
def condition1 : Prop := P = 1.5 * Q
def condition2 : Prop := P * h = 420
def condition3 : Prop := Q * (h + 10) = 420

-- Theorem to be proved
theorem wage_difference_seven (h : ℝ) (P Q : ℝ) 
  (h_condition1 : condition1 P Q)
  (h_condition2 : condition2 P h)
  (h_condition3 : condition3 Q h) :
  (P - Q) = 7 :=
  sorry

end wage_difference_seven_l219_219050


namespace log_base_0_6_5_lt_point_6_pow_5_lt_5_pow_0_6_l219_219256

theorem log_base_0_6_5_lt_point_6_pow_5_lt_5_pow_0_6 
  (h1 : 5^0.6 > 1)
  (h2 : 0 < 0.6^5 ∧ 0.6^5 < 1)
  (h3 : Real.logb 0.6 5 < 0) :
  Real.logb 0.6 5 < 0.6^5 ∧ 0.6^5 < 5^0.6 :=
sorry

end log_base_0_6_5_lt_point_6_pow_5_lt_5_pow_0_6_l219_219256


namespace quadratic_roots_condition_l219_219267

theorem quadratic_roots_condition (a : ℝ) :
  (∃ α : ℝ, 5 * α = -(a - 4) ∧ 4 * α^2 = a - 5) ↔ (a = 7 ∨ a = 5) :=
by
  sorry

end quadratic_roots_condition_l219_219267


namespace min_digits_fraction_l219_219506

def minDigitsToRightOfDecimal (n : ℕ) : ℕ :=
  -- This represents the minimum number of digits needed to express n / (2^15 * 5^7)
  -- as a decimal.
  -- The actual function body is hypothetical and not implemented here.
  15

theorem min_digits_fraction :
  minDigitsToRightOfDecimal 987654321 = 15 :=
by
  sorry

end min_digits_fraction_l219_219506


namespace gcd_of_sum_and_fraction_l219_219291

theorem gcd_of_sum_and_fraction (p : ℕ) (a b : ℕ) (hp : Nat.Prime p) (hodd : p % 2 = 1)
  (hcoprime : Nat.gcd a b = 1) : Nat.gcd (a + b) ((a^p + b^p) / (a + b)) = p := 
sorry

end gcd_of_sum_and_fraction_l219_219291


namespace value_of_f_at_2_l219_219847

def f (x : ℝ) : ℝ := x^2 + 2*x - 3

theorem value_of_f_at_2 : f 2 = 5 :=
by
  -- proof steps would go here
  sorry

end value_of_f_at_2_l219_219847


namespace age_difference_l219_219726

theorem age_difference (A B : ℕ) (h1 : B = 39) (h2 : A + 10 = 2 * (B - 10)) : A - B = 9 := by
  sorry

end age_difference_l219_219726


namespace number_of_new_bottle_caps_l219_219251

def threw_away := 6
def total_bottle_caps_now := 60
def found_more_bottle_caps := 44

theorem number_of_new_bottle_caps (N : ℕ) (h1 : N = threw_away + found_more_bottle_caps) : N = 50 :=
sorry

end number_of_new_bottle_caps_l219_219251


namespace arithmetic_sequence_l219_219098

theorem arithmetic_sequence (n : ℕ) (a : ℕ → ℤ) (h : ∀ n, a n = 3 * n + 1) : 
  ∀ n, a (n + 1) - a n = 3 := by
  sorry

end arithmetic_sequence_l219_219098


namespace find_number_l219_219783

-- Define the main problem statement
theorem find_number (x : ℝ) (h : 0.50 * x = 0.80 * 150 + 80) : x = 400 := by
  sorry

end find_number_l219_219783


namespace arithmetic_sequence_S30_l219_219181

variable {α : Type*} [OrderedAddCommGroup α]

-- Definitions from the conditions
def arithmetic_sum (n : ℕ) : α :=
  sorry -- Placeholder for the sequence sum definition

axiom S10 : arithmetic_sum 10 = 20
axiom S20 : arithmetic_sum 20 = 15

-- The theorem to prove
theorem arithmetic_sequence_S30 : arithmetic_sum 30 = -15 :=
  sorry -- Proof will be completed here

end arithmetic_sequence_S30_l219_219181


namespace alissa_presents_l219_219550

def ethan_presents : ℝ := 31.0
def difference : ℝ := 22.0

theorem alissa_presents : ethan_presents - difference = 9.0 := by sorry

end alissa_presents_l219_219550


namespace fit_seven_rectangles_l219_219312

theorem fit_seven_rectangles (s : ℝ) (a : ℝ) : (s > 0) → (a > 0) → (14 * a ^ 2 ≤ s ^ 2 ∧ 2 * a ≤ s) → 
  (∃ (rectangles : Fin 7 → (ℝ × ℝ)), ∀ i, rectangles i = (a, 2 * a) ∧
   ∀ i j, i ≠ j → rectangles i ≠ rectangles j) :=
sorry

end fit_seven_rectangles_l219_219312


namespace original_price_of_coat_l219_219024

theorem original_price_of_coat (P : ℝ) (h : 0.70 * P = 350) : P = 500 :=
sorry

end original_price_of_coat_l219_219024


namespace infinitenat_not_sum_square_prime_l219_219143

theorem infinitenat_not_sum_square_prime : ∀ k : ℕ, ¬ ∃ (n : ℕ) (p : ℕ), Prime p ∧ (3 * k + 2) ^ 2 = n ^ 2 + p :=
by
  intro k
  sorry

end infinitenat_not_sum_square_prime_l219_219143


namespace find_x_y_l219_219627

theorem find_x_y 
  (x y : ℝ) 
  (h1 : (15 + 30 + x + y) / 4 = 25) 
  (h2 : x = y + 10) :
  x = 32.5 ∧ y = 22.5 := 
by 
  sorry

end find_x_y_l219_219627


namespace M_subset_N_l219_219706

def M : Set ℕ := {x | ∃ a : ℕ, x = a^2 + 2 * a + 2}
def N : Set ℕ := {y | ∃ b : ℕ, y = b^2 - 4 * b + 5}

theorem M_subset_N : M ⊆ N := 
by 
  sorry

end M_subset_N_l219_219706


namespace subset_iff_a_values_l219_219575

theorem subset_iff_a_values (a : ℝ) :
  let P := { x : ℝ | x^2 = 1 }
  let Q := { x : ℝ | a * x = 1 }
  Q ⊆ P ↔ a = 0 ∨ a = 1 ∨ a = -1 :=
by sorry

end subset_iff_a_values_l219_219575


namespace solve_for_x_l219_219109

theorem solve_for_x (x : ℝ) (y : ℝ) (h1 : y = 2) (h2 : y = 1 / (4 * x + 2)) : x = -3/8 :=
by
  -- The proof will go here
  sorry

end solve_for_x_l219_219109


namespace rectangle_area_l219_219524

noncomputable def circle_radius := 8
noncomputable def rect_ratio : ℕ × ℕ := (3, 1)
noncomputable def rect_area (width length : ℕ) : ℕ := width * length

theorem rectangle_area (width length : ℕ) 
  (h1 : 2 * circle_radius = width) 
  (h2 : rect_ratio.1 * width = length) : 
  rect_area width length = 768 := 
sorry

end rectangle_area_l219_219524


namespace find_other_discount_l219_219328

def other_discount (list_price final_price : ℝ) (first_discount : ℝ) (second_discount : ℝ) : Prop :=
  let price_after_first_discount := list_price - (first_discount / 100) * list_price
  final_price = price_after_first_discount - (second_discount / 100) * price_after_first_discount

theorem find_other_discount : 
  other_discount 70 59.22 10 6 :=
by
  sorry

end find_other_discount_l219_219328


namespace parabola_hyperbola_focus_l219_219113

theorem parabola_hyperbola_focus (p : ℝ) (h : p > 0) :
  (∀ x y : ℝ, (y ^ 2 = 2 * p * x) ∧ (x ^ 2 / 4 - y ^ 2 / 5 = 1) → p = 6) :=
by
  sorry

end parabola_hyperbola_focus_l219_219113


namespace speed_of_stream_l219_219675

theorem speed_of_stream 
  (b s : ℝ) 
  (h1 : 78 = (b + s) * 2) 
  (h2 : 50 = (b - s) * 2) 
  : s = 7 := 
sorry

end speed_of_stream_l219_219675


namespace binomial_product_l219_219244

def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_product : binomial 10 3 * binomial 8 3 = 6720 := 
by 
  sorry

end binomial_product_l219_219244


namespace greatest_integer_gcf_l219_219495

theorem greatest_integer_gcf (x : ℕ) : x < 200 ∧ (gcd x 30 = 5) → x = 185 :=
by sorry

end greatest_integer_gcf_l219_219495


namespace total_cost_of_shoes_l219_219314

theorem total_cost_of_shoes 
  (cost_first_pair : ℕ)
  (percentage_increase : ℕ)
  (price_first : cost_first_pair = 22)
  (percentage_increase_eq : percentage_increase = 50) :
  let additional_cost := (percentage_increase * cost_first_pair) / 100
  let cost_second_pair := cost_first_pair + additional_cost
  let total_cost := cost_first_pair + cost_second_pair
  in total_cost = 55 :=
by
  sorry

end total_cost_of_shoes_l219_219314


namespace range_of_k_l219_219837

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := x^(-k^2 + k + 2)

theorem range_of_k (k : ℝ) : (∃ k, (f 2 k < f 3 k)) ↔ (-1 < k) ∧ (k < 2) :=
by
  sorry

end range_of_k_l219_219837


namespace geoboard_quadrilaterals_l219_219374

-- Definitions of the quadrilaterals as required by the conditions of the problem.
def quadrilateral_area (quad : Type) : ℝ := sorry
def quadrilateral_perimeter (quad : Type) : ℝ := sorry

-- Declaration of Quadrilateral I and II on a geoboard.
def quadrilateral_i : Type := sorry
def quadrilateral_ii : Type := sorry

-- The proof problem statement.
theorem geoboard_quadrilaterals :
  quadrilateral_area quadrilateral_i = quadrilateral_area quadrilateral_ii ∧
  quadrilateral_perimeter quadrilateral_i < quadrilateral_perimeter quadrilateral_ii := by
  sorry

end geoboard_quadrilaterals_l219_219374


namespace liar_and_truth_tellers_l219_219770

-- Define the characters and their nature (truth-teller or liar)
inductive Character : Type
| Kikimora
| Leshy
| Vodyanoy

def always_truthful (c : Character) : Prop := sorry
def always_lying (c : Character) : Prop := sorry

axiom kikimora_statement : always_lying Character.Kikimora
axiom leshy_statement : ∃ l₁ l₂ : Character, l₁ ≠ l₂ ∧ always_lying l₁ ∧ always_lying l₂
axiom vodyanoy_statement : true -- Vodyanoy's silence

-- Proof that Kikimora and Vodyanoy are liars and Leshy is truthful
theorem liar_and_truth_tellers :
  always_lying Character.Kikimora ∧
  always_lying Character.Vodyanoy ∧
  always_truthful Character.Leshy := sorry

end liar_and_truth_tellers_l219_219770


namespace program1_values_program2_values_l219_219196

theorem program1_values :
  ∃ (a b c : ℤ), a = 3 ∧ b = -5 ∧ c = 8 ∧
  a = b ∧ b = c ∧
  a = -5 ∧ b = 8 ∧ c = 8 :=
by sorry

theorem program2_values :
  ∃ (a b c : ℤ), a = 3 ∧ b = -5 ∧ c = 8 ∧
  a = b ∧ b = c ∧ c = a ∧
  a = -5 ∧ b = 8 ∧ c = -5 :=
by sorry

end program1_values_program2_values_l219_219196


namespace cannot_be_sum_of_consecutive_nat_iff_power_of_two_l219_219509

theorem cannot_be_sum_of_consecutive_nat_iff_power_of_two (n : ℕ) : 
  (∀ a b : ℕ, n ≠ (b - a + 1) * (a + b) / 2) ↔ (∃ k : ℕ, n = 2 ^ k) := by
  sorry

end cannot_be_sum_of_consecutive_nat_iff_power_of_two_l219_219509


namespace fruit_punch_total_l219_219158

section fruit_punch
variable (orange_punch : ℝ) (cherry_punch : ℝ) (apple_juice : ℝ) (total_punch : ℝ)

axiom h1 : orange_punch = 4.5
axiom h2 : cherry_punch = 2 * orange_punch
axiom h3 : apple_juice = cherry_punch - 1.5
axiom h4 : total_punch = orange_punch + cherry_punch + apple_juice

theorem fruit_punch_total : total_punch = 21 := sorry

end fruit_punch

end fruit_punch_total_l219_219158


namespace find_y_when_x_is_4_l219_219640

def inverse_proportional (x y : ℝ) : Prop :=
  ∃ C : ℝ, x * y = C

theorem find_y_when_x_is_4 :
  ∀ x y : ℝ,
  inverse_proportional x y →
  (x + y = 20) →
  (x - y = 4) →
  (∃ y, y = 24 ∧ x = 4) :=
by
  sorry

end find_y_when_x_is_4_l219_219640


namespace total_fruit_punch_eq_21_l219_219160

def orange_punch : ℝ := 4.5
def cherry_punch := 2 * orange_punch
def apple_juice := cherry_punch - 1.5

theorem total_fruit_punch_eq_21 : orange_punch + cherry_punch + apple_juice = 21 := by 
  -- This is where the proof would go
  sorry

end total_fruit_punch_eq_21_l219_219160


namespace james_ate_eight_slices_l219_219302

-- Define the conditions
def num_pizzas := 2
def slices_per_pizza := 6
def fraction_james_ate := 2 / 3
def total_slices := num_pizzas * slices_per_pizza

-- Define the statement to prove
theorem james_ate_eight_slices : fraction_james_ate * total_slices = 8 :=
by
  sorry

end james_ate_eight_slices_l219_219302


namespace expected_pairs_of_adjacent_face_cards_is_44_over_17_l219_219018
noncomputable def expected_adjacent_face_card_pairs : ℚ :=
  12 * (11 / 51)

theorem expected_pairs_of_adjacent_face_cards_is_44_over_17 :
  expected_adjacent_face_card_pairs = 44 / 17 :=
by
  sorry

end expected_pairs_of_adjacent_face_cards_is_44_over_17_l219_219018


namespace smallest_positive_period_of_f_extreme_values_of_f_on_interval_l219_219412

noncomputable def f (x : ℝ) : ℝ :=
  let a : ℝ × ℝ := (2 * Real.cos x, Real.sqrt 3 * Real.cos x)
  let b : ℝ × ℝ := (Real.cos x, 2 * Real.sin x)
  a.1 * b.1 + a.2 * b.2

theorem smallest_positive_period_of_f :
  ∃ p > 0, ∀ x : ℝ, f (x + p) = f x ∧ p = Real.pi := sorry

theorem extreme_values_of_f_on_interval :
  ∃ max_val min_val, (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≤ max_val) ∧
                     (∀ x ∈ Set.Icc 0 (Real.pi / 2), min_val ≤ f x) ∧
                     max_val = 3 ∧ min_val = 0 := sorry

end smallest_positive_period_of_f_extreme_values_of_f_on_interval_l219_219412


namespace probability_of_no_adjacent_standing_is_123_over_1024_l219_219326

def total_outcomes : ℕ := 2 ^ 10

 -- Define the recursive sequence a_n
def a : ℕ → ℕ 
| 0 => 1
| 1 => 1
| n + 2 => a (n + 1) + a n

lemma a_10_val : a 10 = 123 := by
  sorry

def probability_no_adjacent_standing (n : ℕ): ℚ :=
  a n / total_outcomes

theorem probability_of_no_adjacent_standing_is_123_over_1024 :
  probability_no_adjacent_standing 10 = 123 / 1024 := by
  rw [probability_no_adjacent_standing, total_outcomes, a_10_val]
  norm_num

end probability_of_no_adjacent_standing_is_123_over_1024_l219_219326


namespace count_integers_l219_219578

theorem count_integers (n : ℤ) (h : -11 ≤ n ∧ n ≤ 11) : ∃ (s : Finset ℤ), s.card = 7 ∧ ∀ x ∈ s, (x - 1) * (x + 3) * (x + 7) < 0 :=
by
  sorry

end count_integers_l219_219578


namespace tylenol_tablet_mg_l219_219433

/-- James takes 2 Tylenol tablets every 6 hours and consumes 3000 mg a day.
    Prove the mg of each Tylenol tablet. -/
theorem tylenol_tablet_mg (t : ℕ) (h1 : t = 2) (h2 : 24 / 6 = 4) (h3 : 3000 / (4 * t) = 375) : t * (4 * t) = 3000 :=
by
  sorry

end tylenol_tablet_mg_l219_219433


namespace required_force_l219_219378

theorem required_force (m : ℝ) (g : ℝ) (T : ℝ) (F : ℝ) 
    (h1 : m = 3)
    (h2 : g = 10)
    (h3 : T = m * g)
    (h4 : F = 4 * T) : F = 120 := by
  sorry

end required_force_l219_219378


namespace integer_values_satisfying_square_root_condition_l219_219178

theorem integer_values_satisfying_square_root_condition :
  ∃ (s : Finset ℤ), s.card = 6 ∧ ∀ x ∈ s, 4 < Real.sqrt (3 * x) ∧ Real.sqrt (3 * x) < 6 := sorry

end integer_values_satisfying_square_root_condition_l219_219178


namespace greatest_int_less_than_200_gcd_30_is_5_l219_219501

theorem greatest_int_less_than_200_gcd_30_is_5 : ∃ n, n < 200 ∧ gcd n 30 = 5 ∧ ∀ m, m < 200 ∧ gcd m 30 = 5 → m ≤ n :=
by
  sorry

end greatest_int_less_than_200_gcd_30_is_5_l219_219501


namespace qy_length_l219_219873

theorem qy_length (Q : Type*) (C : Type*) (X Y Z : Q) (QX QZ QY : ℝ) 
  (h1 : 5 = QX)
  (h2 : QZ = 2 * (QY - QX))
  (PQ_theorem : QX * QY = QZ^2) :
  QY = 10 :=
by
  sorry

end qy_length_l219_219873


namespace articles_for_z_men_l219_219984

-- The necessary conditions and given values
def articles_produced (men hours days : ℕ) := men * hours * days

theorem articles_for_z_men (x z : ℕ) (H : articles_produced x x x = x^2) :
  articles_produced z z z = z^3 / x := by
  sorry

end articles_for_z_men_l219_219984


namespace divisor_of_sum_of_four_consecutive_integers_l219_219188

theorem divisor_of_sum_of_four_consecutive_integers (n : ℤ) :
  2 ∣ ((n - 1) + n + (n + 1) + (n + 2)) :=
by sorry

end divisor_of_sum_of_four_consecutive_integers_l219_219188


namespace line_always_passes_fixed_point_l219_219746

theorem line_always_passes_fixed_point : ∀ (m : ℝ), (m-1)*(-2) - 1 + (2*m-1) = 0 :=
by
  intro m
  -- Calculations can be done here to prove the theorem straightforwardly.
  sorry

end line_always_passes_fixed_point_l219_219746


namespace positive_difference_of_solutions_l219_219905

theorem positive_difference_of_solutions : 
    (∀ x : ℝ, |x + 3| = 15 → (x = 12 ∨ x = -18)) → 
    (abs (12 - (-18)) = 30) :=
begin
  intros,
  sorry
end

end positive_difference_of_solutions_l219_219905


namespace total_decorations_l219_219949

theorem total_decorations 
  (skulls : ℕ) (broomsticks : ℕ) (spiderwebs : ℕ) (pumpkins : ℕ) 
  (cauldron : ℕ) (budget_decorations : ℕ) (left_decorations : ℕ)
  (h_skulls : skulls = 12)
  (h_broomsticks : broomsticks = 4)
  (h_spiderwebs : spiderwebs = 12)
  (h_pumpkins : pumpkins = 2 * spiderwebs)
  (h_cauldron : cauldron = 1)
  (h_budget_decorations : budget_decorations = 20)
  (h_left_decorations : left_decorations = 10) : 
  skulls + broomsticks + spiderwebs + pumpkins + cauldron + budget_decorations + left_decorations = 83 := 
by 
  sorry

end total_decorations_l219_219949


namespace find_base_l219_219835

noncomputable def f (a x : ℝ) := 1 + (Real.log x) / (Real.log a)

theorem find_base (a : ℝ) (hinv_pass : (∀ y : ℝ, (∀ x : ℝ, f a x = y → x = 4 → y = 3))) : a = 2 :=
by
  sorry

end find_base_l219_219835


namespace six_digit_squares_l219_219253

theorem six_digit_squares :
    ∃ n m : ℕ, 100000 ≤ n ∧ n ≤ 999999 ∧ 100 ≤ m ∧ m ≤ 999 ∧ n = m^2 ∧ (n = 390625 ∨ n = 141376) :=
by
  sorry

end six_digit_squares_l219_219253


namespace sqrt_computation_l219_219799

open Real

theorem sqrt_computation : sqrt ((5: ℝ)^2 * (7: ℝ)^4) = 245 :=
by
  -- Proof here
  sorry

end sqrt_computation_l219_219799


namespace probability_circle_or_square_l219_219614

theorem probability_circle_or_square (total_figures : ℕ)
    (num_circles : ℕ) (num_squares : ℕ) (num_triangles : ℕ)
    (total_figures_eq : total_figures = 10)
    (num_circles_eq : num_circles = 3)
    (num_squares_eq : num_squares = 4)
    (num_triangles_eq : num_triangles = 3) :
    (num_circles + num_squares) / total_figures = 7 / 10 :=
by sorry

end probability_circle_or_square_l219_219614


namespace similar_triangles_perimeter_ratio_l219_219173

theorem similar_triangles_perimeter_ratio
  (a₁ a₂ s₁ s₂ : ℝ)
  (h₁ : a₁ / a₂ = 1 / 4)
  (h₂ : s₁ / s₂ = 1 / 2) :
  (s₁ / s₂ = 1 / 2) :=
by {
  sorry
}

end similar_triangles_perimeter_ratio_l219_219173


namespace pilot_fish_speed_is_30_l219_219587

-- Define the initial conditions
def keanu_speed : ℝ := 20
def shark_initial_speed : ℝ := keanu_speed
def shark_speed_increase_factor : ℝ := 2
def pilot_fish_speed_increase_factor : ℝ := 0.5

-- Calculating final speeds
def shark_final_speed : ℝ := shark_initial_speed * shark_speed_increase_factor
def shark_speed_increase : ℝ := shark_final_speed - shark_initial_speed
def pilot_fish_speed_increase : ℝ := shark_speed_increase * pilot_fish_speed_increase_factor
def pilot_fish_final_speed : ℝ := keanu_speed + pilot_fish_speed_increase

-- The statement to prove
theorem pilot_fish_speed_is_30 : pilot_fish_final_speed = 30 := by
  sorry

end pilot_fish_speed_is_30_l219_219587


namespace each_son_can_make_l219_219059

noncomputable def land_profit
    (total_land : ℕ) -- measured in hectares
    (num_sons : ℕ)
    (profit_per_section : ℕ) -- profit in dollars per 750 m^2 per 3 months
    (hectare_to_m2 : ℕ) -- conversion factor from hectares to square meters
    (section_area : ℕ) -- 750 m^2
    (periods_per_year : ℕ) : ℕ :=
  let each_son's_share := total_land * hectare_to_m2 / num_sons in
  let num_sections := each_son's_share / section_area in
  num_sections * profit_per_section * periods_per_year

theorem each_son_can_make
    (total_land : ℕ)
    (num_sons : ℕ)
    (profit_per_section : ℕ)
    (hectare_to_m2 : ℕ)
    (section_area : ℕ)
    (periods_per_year : ℕ) :
  total_land = 3 ∧
  num_sons = 8 ∧
  profit_per_section = 500 ∧
  hectare_to_m2 = 10000 ∧
  section_area = 750 ∧
  periods_per_year = 4 →
  land_profit total_land num_sons profit_per_section hectare_to_m2 section_area periods_per_year = 10000 :=
by
  intros h
  cases h
  sorry

end each_son_can_make_l219_219059


namespace simplify_cos_difference_l219_219149

noncomputable def cos (x : ℝ) : ℝ := real.cos x

def c := cos (20 * real.pi / 180)  -- cos(20°)
def d := cos (40 * real.pi / 180)  -- cos(40°)

theorem simplify_cos_difference :
  c - d =
  -- The expression below is placeholder; real expression involves radicals and squares
  sorry :=
by
  let c := cos (20 * real.pi / 180)
  let d := cos (40 * real.pi / 180)
  have h1 : d = 2 * c^2 - 1 := sorry
  let sqrt3 : ℝ := real.sqrt 3
  have h2 : c = (1 / 2) * d + (sqrt3 / 2) * real.sqrt (1 - d^2) := sorry
  sorry

end simplify_cos_difference_l219_219149


namespace failed_both_l219_219996

-- Defining the conditions based on the problem statement
def failed_hindi : ℝ := 0.34
def failed_english : ℝ := 0.44
def passed_both : ℝ := 0.44

-- Defining a proposition to represent the problem and its solution
theorem failed_both (x : ℝ) (h1 : x = failed_hindi + failed_english - (1 - passed_both)) : 
  x = 0.22 :=
by
  sorry

end failed_both_l219_219996


namespace fourth_vertex_parallelogram_coordinates_l219_219293

def fourth_vertex_of_parallelogram (A B C : ℝ × ℝ) :=
  ∃ D : ℝ × ℝ, (D = (11, 4) ∨ D = (-1, 12) ∨ D = (3, -12))

theorem fourth_vertex_parallelogram_coordinates :
  fourth_vertex_of_parallelogram (1, 0) (5, 8) (7, -4) :=
by
  sorry

end fourth_vertex_parallelogram_coordinates_l219_219293


namespace tamara_is_68_inch_l219_219755

-- Defining the conditions
variables (K T : ℕ)

-- Condition 1: Tamara's height in terms of Kim's height
def tamara_height := T = 3 * K - 4

-- Condition 2: Combined height of Tamara and Kim
def combined_height := T + K = 92

-- Statement to prove: Tamara's height is 68 inches
theorem tamara_is_68_inch (h1 : tamara_height T K) (h2 : combined_height T K) : T = 68 :=
by
  sorry

end tamara_is_68_inch_l219_219755


namespace birds_on_fence_l219_219155

theorem birds_on_fence (B : ℕ) : ∃ B, (∃ S, S = 6 ∧ S = (B + 3) + 1) → B = 2 :=
by
  sorry

end birds_on_fence_l219_219155


namespace katy_books_l219_219586

theorem katy_books (x : ℕ) (h : x + 2 * x + (2 * x - 3) = 37) : x = 8 :=
by
  sorry

end katy_books_l219_219586


namespace greatest_integer_with_gcd_30_eq_5_l219_219486

theorem greatest_integer_with_gcd_30_eq_5 :
  ∃ n : ℕ, n < 200 ∧ gcd n 30 = 5 ∧ ∀ m : ℕ, m < 200 ∧ gcd m 30 = 5 → m ≤ n :=
begin
  let n := 195,
  use n,
  split,
  { sorry }, -- Proof that n < 200
  split,
  { sorry }, -- Proof that gcd n 30 = 5
  { sorry }  -- Proof that n is the greatest integer satisfying the conditions
end

end greatest_integer_with_gcd_30_eq_5_l219_219486


namespace first_prize_probability_any_prize_probability_l219_219076

open ProbabilityTheory Classical

-- Assume we have 6 balls: 3 red labeled A, B, C and 3 white labeled by any white identity f'(x0) = 0.

def balls : Finset (String × Bool) := 
  { ("A", true), ("B", true), ("C", true), ("f'(x_0)=0", false), ("f'(x_0)=0", false), ("f'(x_0)=0", false) }

def draw (s : Finset (String × Bool)) : Finset (Finset (String × Bool)) :=
  s.powerset.filter (λ x, x.card = 2)

-- Define the probability definitions for the first and any prize case
def prob_first_prize (s : Finset (String × Bool)) : ℚ :=
  ((draw s).filter (λ x, x.filter (λ y, y.2 = true).card = 2).card : ℚ) / (draw s).card

def prob_any_prize (s : Finset (String × Bool)) : ℚ :=
  1 - (((draw s).filter (λ x, x.filter (λ y, y.2 = false).card = 2).card : ℚ) / (draw s).card)

-- Theorems
theorem first_prize_probability : prob_first_prize balls = 1 / 5 := 
by
  sorry

theorem any_prize_probability : prob_any_prize balls = 4 / 5 := 
by
  sorry

end first_prize_probability_any_prize_probability_l219_219076


namespace twelve_sided_figure_area_is_13_cm2_l219_219639

def twelve_sided_figure_area_cm2 : ℝ :=
  let unit_square := 1
  let full_squares := 9
  let triangle_pairs := 4
  full_squares * unit_square + triangle_pairs * unit_square

theorem twelve_sided_figure_area_is_13_cm2 :
  twelve_sided_figure_area_cm2 = 13 := 
by
  sorry

end twelve_sided_figure_area_is_13_cm2_l219_219639


namespace quadratic_inequality_l219_219279

noncomputable def quadratic_solution_set (a b c : ℝ) (x : ℝ) : Prop :=
ax^2 + bx + c ≥ 0

theorem quadratic_inequality (a b c : ℝ) (ha : a > 0) :
  (∀ x, quadratic_solution_set a b c x) = (x ≤ -3 ∨ x ≥ 4) →
  (∀ x, -12 * x^2 + x + 1 > 0) = (x < -1/4 ∨ x > 1/3) :=
by
  intros h1 h2
  sorry

end quadratic_inequality_l219_219279


namespace slope_of_line_6x_minus_4y_eq_16_l219_219963

noncomputable def slope_of_line (a b c : ℝ) : ℝ :=
  if b ≠ 0 then -a / b else 0

theorem slope_of_line_6x_minus_4y_eq_16 :
  slope_of_line 6 (-4) (-16) = 3 / 2 :=
by
  -- skipping the proof
  sorry

end slope_of_line_6x_minus_4y_eq_16_l219_219963
