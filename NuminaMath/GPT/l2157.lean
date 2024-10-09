import Mathlib

namespace pair_with_gcf_20_l2157_215704

theorem pair_with_gcf_20 (a b : ℕ) (h1 : a = 20) (h2 : b = 40) : Nat.gcd a b = 20 := by
  rw [h1, h2]
  sorry

end pair_with_gcf_20_l2157_215704


namespace total_bouncy_balls_l2157_215745

def red_packs := 4
def yellow_packs := 8
def green_packs := 4
def balls_per_pack := 10

theorem total_bouncy_balls:
  (red_packs * balls_per_pack + yellow_packs * balls_per_pack + green_packs * balls_per_pack) = 160 :=
by 
  sorry

end total_bouncy_balls_l2157_215745


namespace black_squares_in_45th_row_l2157_215761

-- Definitions based on the conditions
def number_of_squares_in_row (n : ℕ) : ℕ := 2 * n + 1

def number_of_black_squares (total_squares : ℕ) : ℕ := (total_squares - 1) / 2

-- The theorem statement
theorem black_squares_in_45th_row : number_of_black_squares (number_of_squares_in_row 45) = 45 :=
by sorry

end black_squares_in_45th_row_l2157_215761


namespace rate_for_gravelling_roads_l2157_215707

variable (length breadth width cost : ℕ)
variable (rate per_square_meter : ℕ)

def total_area_parallel_length : ℕ := length * width
def total_area_parallel_breadth : ℕ := (breadth * width) - (width * width)
def total_area : ℕ := total_area_parallel_length length width + total_area_parallel_breadth breadth width

def rate_per_square_meter := cost / total_area length breadth width

theorem rate_for_gravelling_roads :
  (length = 70) →
  (breadth = 30) →
  (width = 5) →
  (cost = 1900) →
  rate_per_square_meter length breadth width cost = 4 := by
  intros; exact sorry

end rate_for_gravelling_roads_l2157_215707


namespace combinations_with_common_subjects_l2157_215751

-- Conditions and known facts
def subjects : Finset String := {"politics", "history", "geography", "physics", "chemistry", "biology", "technology"}
def personA_must_choose : Finset String := {"physics", "politics"}
def personB_cannot_choose : String := "technology"
def total_combinations : Nat := Nat.choose 7 3
def valid_combinations : Nat := Nat.choose 5 1 * Nat.choose 6 3
def non_common_subject_combinations : Nat := 4 + 4

-- We need to prove this statement
theorem combinations_with_common_subjects : valid_combinations - non_common_subject_combinations = 92 := by
  sorry

end combinations_with_common_subjects_l2157_215751


namespace eval_expression_l2157_215727

theorem eval_expression (a b c : ℕ) (h₀ : a = 3) (h₁ : b = 2) (h₂ : c = 1) : 
  (a^3 + b^2 + c)^2 - (a^3 + b^2 - c)^2 = 124 :=
by
  sorry

end eval_expression_l2157_215727


namespace zhiqiang_series_l2157_215765

theorem zhiqiang_series (a b : ℝ) (n : ℕ) (n_pos : 0 < n) (h : a * b = 1) (h₀ : b ≠ 1):
  (1 + a^n) / (1 + b^n) = ((1 + a) / (1 + b)) ^ n :=
by
  sorry

end zhiqiang_series_l2157_215765


namespace fraction_simplification_l2157_215747

theorem fraction_simplification :
  (1^2 + 1) * (2^2 + 1) * (3^2 + 1) / ((2^2 - 1) * (3^2 - 1) * (4^2 - 1)) = 5 / 18 :=
by
  sorry

end fraction_simplification_l2157_215747


namespace soldiers_count_l2157_215718

-- Statements of conditions and proofs
theorem soldiers_count (n : ℕ) (s : ℕ) :
  (n * n + 30 = s) →
  ((n + 1) * (n + 1) - 50 = s) →
  s = 1975 :=
by
  intros h1 h2
  -- We know from h1 and h2 that there should be a unique solution for s and n that satisfies both
  -- conditions. Our goal is to show that s must be 1975.

  -- Initialize the proof structure
  sorry

end soldiers_count_l2157_215718


namespace hua_luogeng_optimal_selection_method_uses_golden_ratio_l2157_215739

-- Define the conditions
def optimal_selection_method (method: String) : Prop :=
  method = "associated with Hua Luogeng"

def options : List String :=
  ["Golden ratio", "Mean", "Mode", "Median"]

-- Define the proof problem
theorem hua_luogeng_optimal_selection_method_uses_golden_ratio :
  (∀ method, optimal_selection_method method → method ∈ options) → ("Golden ratio" ∈ options) :=
by
  sorry -- Placeholder for the proof


end hua_luogeng_optimal_selection_method_uses_golden_ratio_l2157_215739


namespace dodecagon_diagonals_l2157_215780

/--
The formula for the number of diagonals in a convex n-gon is given by (n * (n - 3)) / 2.
-/
def number_of_diagonals (n : Nat) : Nat := (n * (n - 3)) / 2

/--
A dodecagon has 12 sides.
-/
def dodecagon_sides : Nat := 12

/--
The number of diagonals in a convex dodecagon is 54.
-/
theorem dodecagon_diagonals : number_of_diagonals dodecagon_sides = 54 := by
  sorry

end dodecagon_diagonals_l2157_215780


namespace correct_figure_is_D_l2157_215730

def option_A : Prop := sorry -- placeholder for option A as a diagram representation
def option_B : Prop := sorry -- placeholder for option B as a diagram representation
def option_C : Prop := sorry -- placeholder for option C as a diagram representation
def option_D : Prop := sorry -- placeholder for option D as a diagram representation
def equilateral_triangle (figure : Prop) : Prop := sorry -- placeholder for the condition representing an equilateral triangle in the oblique projection method

theorem correct_figure_is_D : equilateral_triangle option_D := 
sorry

end correct_figure_is_D_l2157_215730


namespace mean_of_elements_increased_by_2_l2157_215723

noncomputable def calculate_mean_after_increase (m : ℝ) (median_value : ℝ) (increase_value : ℝ) : ℝ :=
  let set := [m, m + 2, m + 4, m + 7, m + 11, m + 13]
  let increased_set := set.map (λ x => x + increase_value)
  increased_set.sum / increased_set.length

theorem mean_of_elements_increased_by_2 (m : ℝ) (h : (m + 4 + m + 7) / 2 = 10) :
  calculate_mean_after_increase m 10 2 = 38 / 3 :=
by 
  sorry

end mean_of_elements_increased_by_2_l2157_215723


namespace solve_x_squared_eq_four_l2157_215736

theorem solve_x_squared_eq_four (x : ℝ) (h : x^2 = 4) : x = 2 ∨ x = -2 := 
by sorry

end solve_x_squared_eq_four_l2157_215736


namespace barry_pretzels_l2157_215799

theorem barry_pretzels (A S B : ℕ) (h1 : A = 3 * S) (h2 : S = B / 2) (h3 : A = 18) : B = 12 :=
  by
  sorry

end barry_pretzels_l2157_215799


namespace Joey_age_l2157_215787

-- Define the basic data
def ages : List ℕ := [4, 6, 8, 10, 12]

-- Define the conditions
def cinema_ages (x y : ℕ) : Prop := x + y = 18
def soccer_ages (x y : ℕ) : Prop := x < 11 ∧ y < 11
def stays_home (x : ℕ) : Prop := x = 6

-- The goal is to prove Joey's age
theorem Joey_age : ∃ j, j ∈ ages ∧ stays_home 6 ∧ (∀ x y, cinema_ages x y → x ≠ j ∧ y ≠ j) ∧ 
(∃ x y, soccer_ages x y ∧ x ≠ 6 ∧ y ≠ 6) ∧ j = 8 := by
  sorry

end Joey_age_l2157_215787


namespace cosine_expression_rewrite_l2157_215753

theorem cosine_expression_rewrite (x : ℝ) :
  ∃ a b c d : ℕ, 
    a * (Real.cos (b * x) * Real.cos (c * x) * Real.cos (d * x)) = 
    Real.cos (2 * x) + Real.cos (6 * x) + Real.cos (14 * x) + Real.cos (18 * x) 
    ∧ a + b + c + d = 22 := sorry

end cosine_expression_rewrite_l2157_215753


namespace solution_set_inequality_l2157_215702

noncomputable def f (x : ℝ) := Real.exp (2 * x) - 1
noncomputable def g (x : ℝ) := Real.log (x + 1)

theorem solution_set_inequality :
  {x : ℝ | f (g x) - g (f x) ≤ 1} = Set.Icc (-1 : ℝ) 1 :=
sorry

end solution_set_inequality_l2157_215702


namespace possible_k_values_l2157_215770

variables (p q r s k : ℂ) (h1 : p ≠ 0) (h2 : q ≠ 0) (h3 : r ≠ 0) (h4 : s ≠ 0)
          (h5 : p * q = r * s)
          (h6 : p * k ^ 3 + q * k ^ 2 + r * k + s = 0)
          (h7 : q * k ^ 3 + r * k ^ 2 + s * k + p = 0)

noncomputable def roots_of_unity := {k : ℂ | k ^ 4 = 1}

theorem possible_k_values : k ∈ roots_of_unity :=
by {
  sorry
}

end possible_k_values_l2157_215770


namespace range_of_a_l2157_215713

noncomputable def f (a x : ℝ) : ℝ := x^3 + 3 * a * x^2 + 3 * (a + 2) * x + 1

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, deriv (f a) x ≥ 0) ↔ -1 ≤ a ∧ a ≤ 2 :=
sorry

end range_of_a_l2157_215713


namespace hexagon_angle_U_l2157_215771

theorem hexagon_angle_U 
  (F I U G E R : ℝ)
  (h1 : F = I) 
  (h2 : I = U)
  (h3 : G + E = 180)
  (h4 : R + U = 180)
  (h5 : F + I + G + U + R + E = 720) :
  U = 120 := by
  sorry

end hexagon_angle_U_l2157_215771


namespace awareness_survey_sampling_l2157_215792

theorem awareness_survey_sampling
  (students : Set ℝ) -- assumption that defines the set of students
  (grades : Set ℝ) -- assumption that defines the set of grades
  (awareness : ℝ → ℝ) -- assumption defining the awareness function
  (significant_differences : ∀ g1 g2 : ℝ, g1 ≠ g2 → awareness g1 ≠ awareness g2) -- significant differences in awareness among grades
  (first_grade_students : Set ℝ) -- assumption defining the set of first grade students
  (second_grade_students : Set ℝ) -- assumption defining the set of second grade students
  (third_grade_students : Set ℝ) -- assumption defining the set of third grade students
  (students_from_grades : students = first_grade_students ∪ second_grade_students ∪ third_grade_students) -- assumption that the students are from first, second, and third grades
  (representative_method : (simple_random_sampling → False) ∧ (systematic_sampling_method → False))
  : stratified_sampling_method := 
sorry

end awareness_survey_sampling_l2157_215792


namespace number_of_ways_to_read_BANANA_l2157_215715

/-- 
In a 3x3 grid, there are 84 different ways to read the word BANANA 
by moving from one cell to another cell with which it shares an edge,
and cells may be visited more than once.
-/
theorem number_of_ways_to_read_BANANA (grid : Matrix (Fin 3) (Fin 3) Char) (word : String := "BANANA") : 
  ∃! n : ℕ, n = 84 :=
by
  sorry

end number_of_ways_to_read_BANANA_l2157_215715


namespace inequality_solution_l2157_215766

theorem inequality_solution (x : ℝ) :
  (7 / 36 + (abs (2 * x - (1 / 6)))^2 < 5 / 12) ↔
  (x ∈ Set.Ioo ((1 / 12 - (Real.sqrt 2 / 6))) ((1 / 12 + (Real.sqrt 2 / 6)))) :=
by
  sorry

end inequality_solution_l2157_215766


namespace angle_C_in_triangle_l2157_215769

theorem angle_C_in_triangle (A B C : ℝ) (h1 : A + B = 80) (h2 : A + B + C = 180) : C = 100 := 
by 
  sorry

end angle_C_in_triangle_l2157_215769


namespace eval_expression_l2157_215798

theorem eval_expression : 68 + (156 / 12) + (11 * 19) - 250 - (450 / 9) = -10 := 
by
  sorry

end eval_expression_l2157_215798


namespace carB_highest_avg_speed_l2157_215790

-- Define the distances and times for each car
def distanceA : ℕ := 715
def timeA : ℕ := 11
def distanceB : ℕ := 820
def timeB : ℕ := 12
def distanceC : ℕ := 950
def timeC : ℕ := 14

-- Define the average speeds
def avgSpeedA : ℚ := distanceA / timeA
def avgSpeedB : ℚ := distanceB / timeB
def avgSpeedC : ℚ := distanceC / timeC

theorem carB_highest_avg_speed : avgSpeedB > avgSpeedA ∧ avgSpeedB > avgSpeedC :=
by
  -- Proof will be filled in here
  sorry

end carB_highest_avg_speed_l2157_215790


namespace restaurant_hamburgers_l2157_215726

-- Define the conditions
def hamburgers_served : ℕ := 3
def hamburgers_left_over : ℕ := 6

-- Define the total hamburgers made
def hamburgers_made : ℕ := hamburgers_served + hamburgers_left_over

-- State and prove the theorem
theorem restaurant_hamburgers : hamburgers_made = 9 := by
  sorry

end restaurant_hamburgers_l2157_215726


namespace f_1987_is_3_l2157_215797

noncomputable def f : ℕ → ℕ :=
sorry

axiom f_is_defined : ∀ x : ℕ, f x ≠ 0
axiom f_initial : f 1 = 3
axiom f_functional_equation : ∀ (a b : ℕ), f (a + b) = f a + f b - 2 * f (a * b) + 1

theorem f_1987_is_3 : f 1987 = 3 :=
by
  -- Here we would provide the mathematical proof
  sorry

end f_1987_is_3_l2157_215797


namespace geometric_series_first_term_l2157_215768

theorem geometric_series_first_term
  (a r : ℝ)
  (h1 : a / (1 - r) = 30)
  (h2 : a^2 / (1 - r^2) = 90)
  (hrange : |r| < 1) :
  a = 60 / 11 :=
by 
  sorry

end geometric_series_first_term_l2157_215768


namespace min_dwarfs_l2157_215735

theorem min_dwarfs (chairs : Fin 30 → Prop) 
  (h1 : ∀ i : Fin 30, chairs i ∨ chairs ((i + 1) % 30) ∨ chairs ((i + 2) % 30)) :
  ∃ S : Finset (Fin 30), (S.card = 10) ∧ (∀ i : Fin 30, i ∈ S) :=
sorry

end min_dwarfs_l2157_215735


namespace delivery_time_is_40_minutes_l2157_215740

-- Define the conditions
def total_pizzas : Nat := 12
def two_pizza_stops : Nat := 2
def pizzas_per_stop_with_two_pizzas : Nat := 2
def time_per_stop_minutes : Nat := 4

-- Define the number of pizzas covered by stops with two pizzas
def pizzas_covered_by_two_pizza_stops : Nat := two_pizza_stops * pizzas_per_stop_with_two_pizzas

-- Define the number of single pizza stops
def single_pizza_stops : Nat := total_pizzas - pizzas_covered_by_two_pizza_stops

-- Define the total number of stops
def total_stops : Nat := two_pizza_stops + single_pizza_stops

-- Total time to deliver all pizzas
def total_delivery_time_minutes : Nat := total_stops * time_per_stop_minutes

theorem delivery_time_is_40_minutes : total_delivery_time_minutes = 40 := by
  sorry

end delivery_time_is_40_minutes_l2157_215740


namespace natural_number_pairs_lcm_gcd_l2157_215762

theorem natural_number_pairs_lcm_gcd (a b : ℕ) (h1 : lcm a b * gcd a b = a * b)
  (h2 : lcm a b - gcd a b = (a * b) / 5) : 
  (a = 4 ∧ b = 20) ∨ (a = 20 ∧ b = 4) :=
  sorry

end natural_number_pairs_lcm_gcd_l2157_215762


namespace total_books_proof_l2157_215783

-- Define the number of books Lily finished last month.
def books_last_month : ℕ := 4

-- Define the number of books Lily wants to finish this month.
def books_this_month : ℕ := books_last_month * 2

-- Define the total number of books Lily will finish in two months.
def total_books_two_months : ℕ := books_last_month + books_this_month

-- Theorem to prove the total number of books Lily will finish in two months is 12.
theorem total_books_proof : total_books_two_months = 12 := by
  -- Here would be the proof steps.
  sorry

end total_books_proof_l2157_215783


namespace tyler_bought_10_erasers_l2157_215709

/--
Given that Tyler initially has $100, buys 8 scissors for $5 each, buys some erasers for $4 each,
and has $20 remaining after these purchases, prove that he bought 10 erasers.
-/
theorem tyler_bought_10_erasers : ∀ (initial_money scissors_cost erasers_cost remaining_money : ℕ), 
  initial_money = 100 →
  scissors_cost = 5 →
  erasers_cost = 4 →
  remaining_money = 20 →
  ∃ (scissors_count erasers_count : ℕ),
    scissors_count = 8 ∧ 
    initial_money - scissors_count * scissors_cost - erasers_count * erasers_cost = remaining_money ∧ 
    erasers_count = 10 :=
by
  intros
  sorry

end tyler_bought_10_erasers_l2157_215709


namespace polynomial_equivalence_l2157_215774

def polynomial_expression (x : ℝ) : ℝ :=
  (3 * x ^ 2 + 2 * x - 5) * (x - 2) - (x - 2) * (x ^ 2 - 5 * x + 28) + (4 * x - 7) * (x - 2) * (x + 4)

theorem polynomial_equivalence (x : ℝ) : 
  polynomial_expression x = 6 * x ^ 3 + 4 * x ^ 2 - 93 * x + 122 :=
by {
  sorry
}

end polynomial_equivalence_l2157_215774


namespace students_with_both_l2157_215725

/-- There are 28 students in a class -/
def total_students : ℕ := 28

/-- Number of students with a cat -/
def students_with_cat : ℕ := 17

/-- Number of students with a dog -/
def students_with_dog : ℕ := 10

/-- Number of students with neither a cat nor a dog -/
def students_with_neither : ℕ := 5

/-- Number of students having both a cat and a dog -/
theorem students_with_both :
  students_with_cat + students_with_dog - (total_students - students_with_neither) = 4 :=
sorry

end students_with_both_l2157_215725


namespace ratio_perimeter_to_breadth_l2157_215789

-- Definitions of the conditions
def area_of_rectangle (length breadth : ℝ) := length * breadth
def perimeter_of_rectangle (length breadth : ℝ) := 2 * (length + breadth)

-- The problem statement: prove the ratio of perimeter to breadth
theorem ratio_perimeter_to_breadth (L B : ℝ) (hL : L = 18) (hA : area_of_rectangle L B = 216) :
  (perimeter_of_rectangle L B) / B = 5 :=
by 
  -- Given definitions and conditions, we skip the proof.
  sorry

end ratio_perimeter_to_breadth_l2157_215789


namespace brooke_total_jumping_jacks_l2157_215731

def sj1 : Nat := 20
def sj2 : Nat := 36
def sj3 : Nat := 40
def sj4 : Nat := 50
def Brooke_jumping_jacks : Nat := 3 * (sj1 + sj2 + sj3 + sj4)

theorem brooke_total_jumping_jacks : Brooke_jumping_jacks = 438 := by
  sorry

end brooke_total_jumping_jacks_l2157_215731


namespace mutually_exclusive_events_l2157_215732

/-- A group consists of 3 boys and 2 girls. Two students are to be randomly selected to participate in a speech competition. -/
def num_boys : ℕ := 3
def num_girls : ℕ := 2
def total_selected : ℕ := 2

/-- Possible events under consideration:
  A*: Exactly one boy is selected or exactly two girls are selected -/
def is_boy (s : ℕ) (boys : ℕ) : Prop := s ≤ boys 
def is_girl (s : ℕ) (girls : ℕ) : Prop := s ≤ girls
def one_boy_selected (selected : ℕ) (boys : ℕ) := selected = 1 ∧ is_boy selected boys
def two_girls_selected (selected : ℕ) (girls : ℕ) := selected = 2 ∧ is_girl selected girls

theorem mutually_exclusive_events 
  (selected_boy : ℕ) (selected_girl : ℕ) :
  one_boy_selected selected_boy num_boys ∧ selected_boy + selected_girl = total_selected 
  ∧ two_girls_selected selected_girl num_girls 
  → (one_boy_selected selected_boy num_boys ∨ two_girls_selected selected_girl num_girls) :=
by
  sorry

end mutually_exclusive_events_l2157_215732


namespace percentage_answered_first_correctly_l2157_215773

variable (A B C D : ℝ)

-- Conditions translated to Lean
variable (hB : B = 0.65)
variable (hC : C = 0.20)
variable (hD : D = 0.60)

-- Statement to prove
theorem percentage_answered_first_correctly (hI : A + B - D = 1 - C) : A = 0.75 := by
  -- import conditions
  rw [hB, hC, hD] at hI
  -- solve the equation
  sorry

end percentage_answered_first_correctly_l2157_215773


namespace length_of_train_is_135_l2157_215794

noncomputable def length_of_train (v : ℝ) (t : ℝ) : ℝ :=
  ((v * 1000) / 3600) * t

theorem length_of_train_is_135 :
  length_of_train 140 3.4711508793582233 = 135 :=
sorry

end length_of_train_is_135_l2157_215794


namespace find_f_of_13_l2157_215703

def f : ℤ → ℤ := sorry  -- We define f as a function from integers to integers

theorem find_f_of_13 : 
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x k : ℤ, f (x + 4 * k) = f x) ∧ 
  (f (-1) = 2) → 
  f 13 = -2 := 
by 
  sorry

end find_f_of_13_l2157_215703


namespace cricket_team_members_l2157_215756

theorem cricket_team_members (n : ℕ)
    (captain_age : ℕ) (wicket_keeper_age : ℕ) (average_age : ℕ)
    (remaining_average_age : ℕ) (total_age : ℕ) (remaining_players : ℕ) :
    captain_age = 27 →
    wicket_keeper_age = captain_age + 3 →
    average_age = 24 →
    remaining_average_age = average_age - 1 →
    total_age = average_age * n →
    remaining_players = n - 2 →
    total_age = captain_age + wicket_keeper_age + remaining_average_age * remaining_players →
    n = 11 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end cricket_team_members_l2157_215756


namespace profit_percent_is_25_l2157_215750

-- Define the cost price (CP) and selling price (SP) based on the given ratio.
def CP (x : ℝ) := 4 * x
def SP (x : ℝ) := 5 * x

-- Calculate the profit percent based on the given conditions.
noncomputable def profitPercent (x : ℝ) := ((SP x - CP x) / CP x) * 100

-- Prove that the profit percent is 25% given the ratio of CP to SP is 4:5.
theorem profit_percent_is_25 (x : ℝ) : profitPercent x = 25 := by
  sorry

end profit_percent_is_25_l2157_215750


namespace exp_sum_is_neg_one_l2157_215724

noncomputable def sumExpExpressions : ℂ :=
  (Complex.exp (Real.pi * Complex.I / 7) +
   Complex.exp (2 * Real.pi * Complex.I / 7) +
   Complex.exp (3 * Real.pi * Complex.I / 7) +
   Complex.exp (4 * Real.pi * Complex.I / 7) +
   Complex.exp (5 * Real.pi * Complex.I / 7) +
   Complex.exp (6 * Real.pi * Complex.I / 7) +
   Complex.exp (2 * Real.pi * Complex.I / 9) +
   Complex.exp (4 * Real.pi * Complex.I / 9) +
   Complex.exp (6 * Real.pi * Complex.I / 9) +
   Complex.exp (8 * Real.pi * Complex.I / 9) +
   Complex.exp (10 * Real.pi * Complex.I / 9) +
   Complex.exp (12 * Real.pi * Complex.I / 9) +
   Complex.exp (14 * Real.pi * Complex.I / 9) +
   Complex.exp (16 * Real.pi * Complex.I / 9))

theorem exp_sum_is_neg_one : sumExpExpressions = -1 := by
  sorry

end exp_sum_is_neg_one_l2157_215724


namespace tens_digit_17_pow_1993_l2157_215721

theorem tens_digit_17_pow_1993 :
  (17 ^ 1993) % 100 / 10 = 3 := by
  sorry

end tens_digit_17_pow_1993_l2157_215721


namespace runners_meet_opposite_dir_l2157_215700

theorem runners_meet_opposite_dir 
  {S x y : ℝ}
  (h1 : S / x + 5 = S / y)
  (h2 : S / (x - y) = 30) :
  S / (x + y) = 6 := 
sorry

end runners_meet_opposite_dir_l2157_215700


namespace fraction_values_l2157_215763

theorem fraction_values (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : 2 * x^2 + 2 * y^2 = 5 * x * y) :
  ∃ k ∈ ({3, -3} : Set ℝ), (x + y) / (x - y) = k :=
by
  sorry

end fraction_values_l2157_215763


namespace find_p_l2157_215733

theorem find_p 
  (p q x y : ℤ)
  (h1 : p * x + q * y = 8)
  (h2 : 3 * x - q * y = 38)
  (hx : x = 2)
  (hy : y = -4) : 
  p = 20 := 
by 
  subst hx
  subst hy
  sorry

end find_p_l2157_215733


namespace sufficient_condition_l2157_215796

theorem sufficient_condition (a : ℝ) (h : a > 0) : a^2 + a ≥ 0 :=
sorry

end sufficient_condition_l2157_215796


namespace totalInitialAmount_l2157_215729

variable (a j t k x : ℝ)

-- Given conditions
def initialToyAmount : Prop :=
  t = 48

def kimRedistribution : Prop :=
  k = 4 * x - 144

def amyRedistribution : Prop :=
  (a = 3 * x) ∧ (j = 2 * x) ∧ (t = 2 * x)

def janRedistribution : Prop :=
  (a = 3 * x) ∧ (t = 4 * x)

def toyRedistribution : Prop :=
  (a = 6 * x) ∧ (j = -6 * x) ∧ (t = 48) 

def toyFinalAmount : Prop :=
  t = 48

-- Proof Problem
theorem totalInitialAmount
  (h1 : initialToyAmount t)
  (h2 : kimRedistribution k x)
  (h3 : amyRedistribution a j t x)
  (h4 : janRedistribution a t x)
  (h5 : toyRedistribution a j t x)
  (h6 : toyFinalAmount t) :
  a + j + t + k = 192 :=
sorry

end totalInitialAmount_l2157_215729


namespace simplified_fraction_l2157_215711

noncomputable def simplify_and_rationalize (a b c d e f : ℝ) : ℝ :=
  (Real.sqrt a / Real.sqrt b) * (Real.sqrt c / Real.sqrt d) * (Real.sqrt e / Real.sqrt f)

theorem simplified_fraction :
  simplify_and_rationalize 3 7 5 9 6 8 = Real.sqrt 35 / 14 :=
by
  sorry

end simplified_fraction_l2157_215711


namespace minimum_bailing_rate_is_seven_l2157_215764

noncomputable def minimum_bailing_rate (shore_distance : ℝ) (paddling_speed : ℝ) 
                                       (water_intake_rate : ℝ) (max_capacity : ℝ) : ℝ := 
  let time_to_shore := shore_distance / paddling_speed
  let intake_total := water_intake_rate * time_to_shore
  let required_rate := (intake_total - max_capacity) / time_to_shore
  required_rate

theorem minimum_bailing_rate_is_seven 
  (shore_distance : ℝ) (paddling_speed : ℝ) (water_intake_rate : ℝ) (max_capacity : ℝ) :
  shore_distance = 2 →
  paddling_speed = 3 →
  water_intake_rate = 8 →
  max_capacity = 40 →
  minimum_bailing_rate shore_distance paddling_speed water_intake_rate max_capacity = 7 :=
by 
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end minimum_bailing_rate_is_seven_l2157_215764


namespace chromium_percentage_l2157_215716

theorem chromium_percentage (c1 c2 : ℝ) (w1 w2 : ℝ) (percentage1 percentage2 : ℝ) : 
  percentage1 = 0.1 → 
  percentage2 = 0.08 → 
  w1 = 15 → 
  w2 = 35 → 
  (c1 = percentage1 * w1) → 
  (c2 = percentage2 * w2) → 
  (c1 + c2 = 4.3) → 
  ((w1 + w2) = 50) →
  ((c1 + c2) / (w1 + w2) * 100 = 8.6) := 
by 
  sorry

end chromium_percentage_l2157_215716


namespace find_books_second_purchase_profit_l2157_215775

-- For part (1)
theorem find_books (x y : ℕ) (h₁ : 12 * x + 10 * y = 1200) (h₂ : 3 * x + 2 * y = 270) :
  x = 50 ∧ y = 60 :=
by 
  sorry

-- For part (2)
theorem second_purchase_profit (m : ℕ) (h₃ : 50 * (m - 12) + 2 * 60 * (12 - 10) ≥ 340) :
  m ≥ 14 :=
by 
  sorry

end find_books_second_purchase_profit_l2157_215775


namespace more_oranges_than_apples_l2157_215714

-- Definitions based on conditions
def apples : ℕ := 14
def oranges : ℕ := 2 * 12  -- 2 dozen oranges

-- Statement to prove
theorem more_oranges_than_apples : oranges - apples = 10 := by
  sorry

end more_oranges_than_apples_l2157_215714


namespace geometric_sequence_sum_l2157_215701

/-- 
In a geometric sequence of real numbers, the sum of the first 2 terms is 15,
and the sum of the first 6 terms is 195. Prove that the sum of the first 4 terms is 82.
-/
theorem geometric_sequence_sum :
  ∃ (a r : ℝ), (a + a * r = 15) ∧ (a * (1 - r^6) / (1 - r) = 195) ∧ (a * (1 + r + r^2 + r^3) = 82) :=
by
  sorry

end geometric_sequence_sum_l2157_215701


namespace cos_R_in_triangle_PQR_l2157_215741

theorem cos_R_in_triangle_PQR
  (P Q R : ℝ) (hP : P = 90) (hQ : Real.sin Q = 3/5)
  (h_sum : P + Q + R = 180) (h_PQ_comp : P + Q = 90) :
  Real.cos R = 3 / 5 := 
sorry

end cos_R_in_triangle_PQR_l2157_215741


namespace min_marked_price_l2157_215708

theorem min_marked_price 
  (x : ℝ) 
  (sets : ℝ) 
  (cost_per_set : ℝ) 
  (discount : ℝ) 
  (desired_profit : ℝ) 
  (purchase_cost : ℝ) 
  (total_revenue : ℝ) 
  (cost : ℝ)
  (h1 : sets = 40)
  (h2 : cost_per_set = 80)
  (h3 : discount = 0.9)
  (h4 : desired_profit = 4000)
  (h5 : cost = sets * cost_per_set)
  (h6 : total_revenue = sets * (discount * x))
  (h7 : total_revenue - cost ≥ desired_profit) : x ≥ 200 := by
  sorry

end min_marked_price_l2157_215708


namespace percentage_problem_l2157_215785

theorem percentage_problem (N : ℕ) (P : ℕ) (h1 : N = 25) (h2 : N = (P * N / 100) + 21) : P = 16 :=
sorry

end percentage_problem_l2157_215785


namespace jake_fewer_peaches_l2157_215754

theorem jake_fewer_peaches (steven_peaches : ℕ) (jake_peaches : ℕ) (h1 : steven_peaches = 19) (h2 : jake_peaches = 7) : steven_peaches - jake_peaches = 12 :=
sorry

end jake_fewer_peaches_l2157_215754


namespace catherine_friends_count_l2157_215720

/-
Definition and conditions:
- An equal number of pencils and pens, totaling 60 each.
- Gave away 8 pens and 6 pencils to each friend.
- Left with 22 pens and pencils.
Proof:
- The number of friends she gave pens and pencils to equals 7.
-/
theorem catherine_friends_count :
  ∀ (pencils pens friends : ℕ),
  pens = 60 →
  pencils = 60 →
  (pens + pencils) - friends * (8 + 6) = 22 →
  friends = 7 :=
sorry

end catherine_friends_count_l2157_215720


namespace initial_performers_count_l2157_215759

theorem initial_performers_count (n : ℕ)
    (h1 : ∃ rows, 8 * rows = n)
    (h2 : ∃ (m : ℕ), n + 16 = m ∧ ∃ s, s * s = m)
    (h3 : ∃ (k : ℕ), n + 1 = k ∧ ∃ t, t * t = k) : 
    n = 48 := 
sorry

end initial_performers_count_l2157_215759


namespace sum_placed_on_SI_l2157_215734

theorem sum_placed_on_SI :
  let P₁ := 4000
  let r₁ := 0.10
  let t₁ := 2
  let CI := P₁ * ((1 + r₁)^t₁ - 1)

  let SI := (1 / 2 * CI : ℝ)
  let r₂ := 0.08
  let t₂ := 3
  let P₂ := SI / (r₂ * t₂)

  P₂ = 1750 :=
by
  sorry

end sum_placed_on_SI_l2157_215734


namespace depth_of_grass_sheet_l2157_215706

-- Given conditions
def playground_area : ℝ := 5900
def grass_cost_per_cubic_meter : ℝ := 2.80
def total_cost : ℝ := 165.2

-- Variable to solve for
variable (d : ℝ)

-- Theorem statement
theorem depth_of_grass_sheet
  (h : total_cost = (playground_area * d) * grass_cost_per_cubic_meter) :
  d = 0.01 :=
by
  sorry

end depth_of_grass_sheet_l2157_215706


namespace product_in_A_l2157_215760

def A : Set ℤ := { z | ∃ a b : ℤ, z = a^2 + 4 * a * b + b^2 }

theorem product_in_A (x y : ℤ) (hx : x ∈ A) (hy : y ∈ A) : x * y ∈ A := 
by
  sorry

end product_in_A_l2157_215760


namespace pies_baked_l2157_215788

theorem pies_baked (days : ℕ) (eddie_rate : ℕ) (sister_rate : ℕ) (mother_rate : ℕ)
  (H1 : eddie_rate = 3) (H2 : sister_rate = 6) (H3 : mother_rate = 8) (days_eq : days = 7) :
  eddie_rate * days + sister_rate * days + mother_rate * days = 119 :=
by
  sorry

end pies_baked_l2157_215788


namespace locus_of_intersection_l2157_215793

-- Define the conditions
def line_e (m_e x y : ℝ) : Prop := y = m_e * (x - 1) + 1
def line_f (m_f x y : ℝ) : Prop := y = m_f * (x + 1) + 1
def slope_diff_cond (m_e m_f : ℝ) : Prop := (m_e - m_f = 2 ∨ m_f - m_e = 2)
def not_at_points (x y : ℝ) : Prop := (x, y) ≠ (1, 1) ∧ (x, y) ≠ (-1, 1)

-- Define the proof problem
theorem locus_of_intersection (x y m_e m_f : ℝ) :
  line_e m_e x y → line_f m_f x y → slope_diff_cond m_e m_f → not_at_points x y →
  (y = x^2 ∨ y = 2 - x^2) :=
by
  intros he hf h_diff h_not_at
  sorry

end locus_of_intersection_l2157_215793


namespace damage_in_dollars_l2157_215767

noncomputable def euros_to_dollars (euros : ℝ) : ℝ := euros * (1 / 0.9)

theorem damage_in_dollars :
  euros_to_dollars 45000000 = 49995000 :=
by
  -- This is where the proof would go
  sorry

end damage_in_dollars_l2157_215767


namespace grocery_store_total_bottles_l2157_215728

def total_bottles (regular_soda : Nat) (diet_soda : Nat) : Nat :=
  regular_soda + diet_soda

theorem grocery_store_total_bottles :
 (total_bottles 9 8 = 17) :=
 by
   sorry

end grocery_store_total_bottles_l2157_215728


namespace find_x_l2157_215705

theorem find_x (x y : ℤ) (hx : x > y) (hy : y > 0) (h : x + y + x * y = 143) : x = 15 :=
by sorry

end find_x_l2157_215705


namespace solve_for_a_l2157_215757

theorem solve_for_a (a x : ℤ) (h : x + 2 * a = -3) (hx : x = 1) : a = -2 := by
  sorry

end solve_for_a_l2157_215757


namespace arithmetic_sequence_second_term_l2157_215737

theorem arithmetic_sequence_second_term (S₃: ℕ) (a₁: ℕ) (h1: S₃ = 9) (h2: a₁ = 1) : 
∃ d a₂, 3 * a₁ + 3 * d = S₃ ∧ a₂ = a₁ + d ∧ a₂ = 3 :=
by
  sorry

end arithmetic_sequence_second_term_l2157_215737


namespace N_is_85714_l2157_215781

theorem N_is_85714 (N : ℕ) (hN : 10000 ≤ N ∧ N < 100000) 
  (P : ℕ := 200000 + N) 
  (Q : ℕ := 10 * N + 2) 
  (hQ_eq_3P : Q = 3 * P) 
  : N = 85714 := 
by 
  sorry

end N_is_85714_l2157_215781


namespace smallest_number_l2157_215712

theorem smallest_number (n : ℕ) :
  (n % 3 = 1) ∧
  (n % 5 = 3) ∧
  (n % 6 = 4) →
  n = 28 :=
sorry

end smallest_number_l2157_215712


namespace contestant_final_score_l2157_215719

theorem contestant_final_score 
    (content_score : ℕ)
    (delivery_score : ℕ)
    (weight_content : ℕ)
    (weight_delivery : ℕ)
    (h1 : content_score = 90)
    (h2 : delivery_score = 85)
    (h3 : weight_content = 6)
    (h4 : weight_delivery = 4) : 
    (content_score * weight_content + delivery_score * weight_delivery) / (weight_content + weight_delivery) = 88 := 
sorry

end contestant_final_score_l2157_215719


namespace compare_neg5_neg7_l2157_215722

theorem compare_neg5_neg7 : -5 > -7 := 
by
  sorry

end compare_neg5_neg7_l2157_215722


namespace sum_of_digits_of_triangular_number_2010_l2157_215755

noncomputable def triangular_number (n : ℕ) : ℕ :=
  n * (n + 1) / 2

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_of_digits_of_triangular_number_2010 (N : ℕ)
  (h₁ : triangular_number N = 2010) :
  sum_of_digits N = 9 :=
sorry

end sum_of_digits_of_triangular_number_2010_l2157_215755


namespace milk_water_ratio_l2157_215758

theorem milk_water_ratio
  (vessel1_milk_ratio : ℚ)
  (vessel1_water_ratio : ℚ)
  (vessel2_milk_ratio : ℚ)
  (vessel2_water_ratio : ℚ)
  (equal_mixture_units  : ℚ)
  (h1 : vessel1_milk_ratio / vessel1_water_ratio = 4 / 1)
  (h2 : vessel2_milk_ratio / vessel2_water_ratio = 7 / 3)
  :
  (vessel1_milk_ratio + vessel2_milk_ratio) / 
  (vessel1_water_ratio + vessel2_water_ratio) = 11 / 4 :=
by
  sorry

end milk_water_ratio_l2157_215758


namespace anticipated_sedans_l2157_215784

theorem anticipated_sedans (sales_sports_cars sedans_ratio sports_ratio sports_forecast : ℕ) 
  (h_ratio : sports_ratio = 5) (h_sedans_ratio : sedans_ratio = 8) (h_sports_forecast : sports_forecast = 35)
  (h_eq : sales_sports_cars = sports_ratio * sports_forecast) :
  sales_sports_cars * 8 / 5 = 56 :=
by
  sorry

end anticipated_sedans_l2157_215784


namespace min_value_expr_l2157_215738

theorem min_value_expr (x : ℝ) (hx : x > 0) : 4 * x + 1 / x^2 ≥ 5 :=
by
  sorry

end min_value_expr_l2157_215738


namespace theatre_fraction_l2157_215776

noncomputable def fraction_theatre_took_elective_last_year (T P Th M : ℕ) : Prop :=
  (P = 1 / 2 * T) ∧
  (Th + M = T - P) ∧
  (1 / 3 * P + M = 2 / 3 * T) ∧
  (Th = 1 / 6 * T)

theorem theatre_fraction (T P Th M : ℕ) :
  fraction_theatre_took_elective_last_year T P Th M →
  Th / T = 1 / 6 :=
by
  intro h
  cases h
  sorry

end theatre_fraction_l2157_215776


namespace parabola_vertex_example_l2157_215746

-- Definitions based on conditions
def parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c
def vertex (a : ℝ) (x : ℝ) : ℝ := a * (x - 1)^2 + 3

-- Conditions given in the problem
def condition1 (a b c : ℝ) : Prop := parabola a b c 2 = 5
def condition2 (a : ℝ) : Prop := vertex a 1 = 3

-- Goal statement to be proved
theorem parabola_vertex_example : ∃ (a b c : ℝ), 
  condition1 a b c ∧ condition2 a ∧ a - b + c = 11 :=
by
  sorry

end parabola_vertex_example_l2157_215746


namespace solve_for_x_l2157_215742

theorem solve_for_x : ∀ (x : ℝ), (4 * x^2 - 3 * x + 2) / (x + 2) = 4 * x - 5 → x = 2 :=
by
  intros x h
  sorry

end solve_for_x_l2157_215742


namespace select_team_with_smaller_variance_l2157_215752

theorem select_team_with_smaller_variance 
    (variance_A variance_B : ℝ)
    (hA : variance_A = 1.5)
    (hB : variance_B = 2.8)
    : variance_A < variance_B → "Team A" = "Team A" :=
by
  intros h
  sorry

end select_team_with_smaller_variance_l2157_215752


namespace fraction_of_journey_asleep_l2157_215749

theorem fraction_of_journey_asleep (x y : ℝ) (hx : x > 0) (hy : y = x / 3) :
  y / x = 1 / 3 :=
by
  sorry

end fraction_of_journey_asleep_l2157_215749


namespace total_seashells_l2157_215779

theorem total_seashells (a b : Nat) (h1 : a = 5) (h2 : b = 7) : 
  let total_first_two_days := a + b
  let third_day := 2 * total_first_two_days
  let total := total_first_two_days + third_day
  total = 36 := 
by
  sorry

end total_seashells_l2157_215779


namespace expression_equivalence_l2157_215748

-- Define the initial expression
def expr (w : ℝ) : ℝ := 3 * w + 4 - 2 * w^2 - 5 * w - 6 + w^2 + 7 * w + 8 - 3 * w^2

-- Define the simplified expression
def simplified_expr (w : ℝ) : ℝ := 5 * w - 4 * w^2 + 6

-- Theorem stating the equivalence
theorem expression_equivalence (w : ℝ) : expr w = simplified_expr w :=
by
  -- we would normally simplify and prove here, but we state the theorem and skip the proof for now.
  sorry

end expression_equivalence_l2157_215748


namespace parabola_focus_distance_l2157_215743

theorem parabola_focus_distance (p : ℝ) (h : 2 * p = 8) : p = 4 :=
  by
  sorry

end parabola_focus_distance_l2157_215743


namespace max_a4a7_value_l2157_215795

variable {a : ℕ → ℝ}
variable {d : ℝ}

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop := ∀ n m : ℕ, a (n + 1) = a n + d

-- Given conditions
def given_conditions (a : ℕ → ℝ) (d : ℝ) : Prop := 
  arithmetic_sequence a d ∧ a 5 = 4 -- a6 = 4 so we use index 5 since Lean is 0-indexed

-- Define the product a4 * a7
def a4a7_product (a : ℕ → ℝ) (d : ℝ) : ℝ := (a 5 - 2 * d) * (a 5 + d)

-- The maximum value of a4 * a7
def max_a4a7 (a : ℕ → ℝ) (d : ℝ) : ℝ := 18

-- The proof problem statement
theorem max_a4a7_value (a : ℕ → ℝ) (d : ℝ) :
  given_conditions a d → a4a7_product a d = max_a4a7 a d :=
by
  sorry

end max_a4a7_value_l2157_215795


namespace pipe_b_fills_tank_7_times_faster_l2157_215782

theorem pipe_b_fills_tank_7_times_faster 
  (time_A : ℝ) 
  (time_B : ℝ)
  (combined_time : ℝ) 
  (hA : time_A = 30)
  (h_combined : combined_time = 3.75) 
  (hB : time_B = time_A / 7) :
  time_B =  30 / 7 :=
by
  sorry

end pipe_b_fills_tank_7_times_faster_l2157_215782


namespace min_square_sum_l2157_215710

theorem min_square_sum (a b : ℝ) (h : a + b = 3) : a^2 + b^2 ≥ 9 / 2 :=
by 
  sorry

end min_square_sum_l2157_215710


namespace Tracy_sold_paintings_l2157_215772

theorem Tracy_sold_paintings (num_people : ℕ) (group1_customers : ℕ) (group1_paintings : ℕ)
    (group2_customers : ℕ) (group2_paintings : ℕ) (group3_customers : ℕ) (group3_paintings : ℕ) 
    (total_paintings : ℕ) :
    num_people = 20 →
    group1_customers = 4 →
    group1_paintings = 2 →
    group2_customers = 12 →
    group2_paintings = 1 →
    group3_customers = 4 →
    group3_paintings = 4 →
    total_paintings = (group1_customers * group1_paintings) + (group2_customers * group2_paintings) + 
                      (group3_customers * group3_paintings) →
    total_paintings = 36 :=
by
  intros 
  -- including this to ensure the lean code passes syntax checks
  sorry

end Tracy_sold_paintings_l2157_215772


namespace a_wins_by_200_meters_l2157_215786

-- Define the conditions
def race_distance : ℕ := 600
def speed_ratio_a_to_b := 5 / 4
def head_start_a : ℕ := 100

-- Define the proof statement
theorem a_wins_by_200_meters (x : ℝ) (ha_speed : ℝ := 5 * x) (hb_speed : ℝ := 4 * x)
  (ha_distance_to_win : ℝ := race_distance - head_start_a) :
  (ha_distance_to_win / ha_speed) = (400 / hb_speed) → 
  600 - (400) = 200 :=
by
  -- For now, skip the proof, focus on the statement.
  sorry

end a_wins_by_200_meters_l2157_215786


namespace sin_2pi_minus_alpha_l2157_215778

noncomputable def alpha_condition (α : ℝ) : Prop :=
  (3 * Real.pi / 2 < α) ∧ (α < 2 * Real.pi) ∧ (Real.cos (Real.pi + α) = -1 / 2)

theorem sin_2pi_minus_alpha (α : ℝ) (h : alpha_condition α) : Real.sin (2 * Real.pi - α) = Real.sqrt 3 / 2 :=
by
  sorry

end sin_2pi_minus_alpha_l2157_215778


namespace goldfish_count_equal_in_6_months_l2157_215744

def initial_goldfish_brent : ℕ := 3
def initial_goldfish_gretel : ℕ := 243

def goldfish_brent (n : ℕ) : ℕ := initial_goldfish_brent * 4^n
def goldfish_gretel (n : ℕ) : ℕ := initial_goldfish_gretel * 3^n

theorem goldfish_count_equal_in_6_months : 
  (∃ n : ℕ, goldfish_brent n = goldfish_gretel n) ↔ n = 6 :=
by
  sorry

end goldfish_count_equal_in_6_months_l2157_215744


namespace system_of_equations_m_value_l2157_215717

theorem system_of_equations_m_value {x y m : ℝ} 
  (h1 : 2 * x + y = 4)
  (h2 : x + 2 * y = m)
  (h3 : x + y = 1) : m = -1 := 
sorry

end system_of_equations_m_value_l2157_215717


namespace xyz_value_l2157_215791

theorem xyz_value (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x + 1/y = 5) (h2 : y + 1/z = 3) (h3 : z + 1/x = 2) :
  x * y * z = 10 + 3 * Real.sqrt 11 :=
by
  sorry

end xyz_value_l2157_215791


namespace scientific_notation_21600_l2157_215777

theorem scientific_notation_21600 : ∃ (a : ℝ) (n : ℤ), 21600 = a * 10^n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ a = 2.16 ∧ n = 4 :=
by
  sorry

end scientific_notation_21600_l2157_215777
