import Mathlib

namespace find_original_number_l868_86838

theorem find_original_number : 
  ∃ x : ℝ, 3 * (2 * x + 5) = 135 ∧ x = 20 := by
  sorry

end find_original_number_l868_86838


namespace track_length_proof_l868_86804

/-- The length of the circular track -/
def track_length : ℝ := 330

/-- The distance Pamela runs before the first meeting -/
def pamela_first_meeting : ℝ := 120

/-- The additional distance Jane runs between the first and second meeting -/
def jane_additional : ℝ := 210

/-- Proves that the track length is correct given the meeting conditions -/
theorem track_length_proof :
  ∃ (pamela_speed jane_speed : ℝ),
    pamela_speed > 0 ∧ jane_speed > 0 ∧
    pamela_first_meeting / (track_length - pamela_first_meeting) = pamela_speed / jane_speed ∧
    (track_length - pamela_first_meeting + jane_additional) / (pamela_first_meeting + track_length - jane_additional) = jane_speed / pamela_speed :=
by sorry


end track_length_proof_l868_86804


namespace system_of_inequalities_l868_86828

theorem system_of_inequalities (x : ℝ) : 
  (x - 1 < 3 ∧ x + 1 ≥ (1 + 2*x) / 3) ↔ -2 ≤ x ∧ x < 4 := by
  sorry

end system_of_inequalities_l868_86828


namespace furniture_cost_l868_86893

/-- Prove that the cost of the furniture is $400, given the conditions of Emma's spending. -/
theorem furniture_cost (initial_amount : ℝ) (remaining_amount : ℝ) 
  (h1 : initial_amount = 2000)
  (h2 : remaining_amount = 400)
  (h3 : ∃ (furniture_cost : ℝ), remaining_amount = (1/4) * (initial_amount - furniture_cost)) :
  ∃ (furniture_cost : ℝ), furniture_cost = 400 := by
  sorry

end furniture_cost_l868_86893


namespace min_one_by_one_required_l868_86839

/-- Represents a square on the grid -/
inductive Square
  | one : Square  -- 1x1 square
  | two : Square  -- 2x2 square
  | three : Square -- 3x3 square

/-- The size of the grid -/
def gridSize : Nat := 23

/-- Represents a cell on the grid -/
structure Cell where
  row : Fin gridSize
  col : Fin gridSize

/-- A covering of the grid -/
def Covering := List (Square × Cell)

/-- Checks if a covering is valid (covers all cells except one) -/
def isValidCovering (c : Covering) : Prop := sorry

/-- Checks if a covering uses only 2x2 and 3x3 squares -/
def usesOnlyTwoAndThree (c : Covering) : Prop := sorry

theorem min_one_by_one_required :
  ¬∃ (c : Covering), isValidCovering c ∧ usesOnlyTwoAndThree c :=
sorry

end min_one_by_one_required_l868_86839


namespace tallest_player_height_l868_86858

/-- Given a basketball team where the tallest player is 9.5 inches taller than
    the shortest player, and the shortest player is 68.25 inches tall,
    prove that the tallest player is 77.75 inches tall. -/
theorem tallest_player_height :
  let shortest_player_height : ℝ := 68.25
  let height_difference : ℝ := 9.5
  let tallest_player_height : ℝ := shortest_player_height + height_difference
  tallest_player_height = 77.75 := by sorry

end tallest_player_height_l868_86858


namespace nth_equation_pattern_l868_86823

theorem nth_equation_pattern (n : ℕ) (h : n > 0) :
  9 * (n - 1) + n = 10 * (n - 1) + 1 :=
by sorry

end nth_equation_pattern_l868_86823


namespace log_has_zero_in_open_interval_l868_86814

theorem log_has_zero_in_open_interval :
  ∃ x, 0 < x ∧ x < 2 ∧ Real.log x = 0 := by sorry

end log_has_zero_in_open_interval_l868_86814


namespace expression_factorization_l868_86867

theorem expression_factorization (x : ℝ) : 3 * x^2 + 12 * x + 12 = 3 * (x + 2)^2 := by
  sorry

end expression_factorization_l868_86867


namespace lawns_mowed_count_l868_86800

def shoe_cost : ℕ := 95
def saving_months : ℕ := 3
def monthly_allowance : ℕ := 5
def lawn_mowing_charge : ℕ := 15
def driveway_shoveling_charge : ℕ := 7
def change_after_purchase : ℕ := 15
def driveways_shoveled : ℕ := 5

def total_money : ℕ := shoe_cost + change_after_purchase
def allowance_savings : ℕ := saving_months * monthly_allowance
def shoveling_earnings : ℕ := driveways_shoveled * driveway_shoveling_charge
def mowing_earnings : ℕ := total_money - allowance_savings - shoveling_earnings

theorem lawns_mowed_count : mowing_earnings / lawn_mowing_charge = 4 := by
  sorry

end lawns_mowed_count_l868_86800


namespace smallest_m_satisfying_conditions_l868_86895

theorem smallest_m_satisfying_conditions : ∃ m : ℕ,
  (100 ≤ m ∧ m < 1000) ∧  -- m is a three-digit number
  (∃ k : ℤ, m + 7 = 9 * k) ∧  -- m + 7 is divisible by 9
  (∃ l : ℤ, m - 9 = 7 * l) ∧  -- m - 9 is divisible by 7
  (∀ n : ℕ, (100 ≤ n ∧ n < 1000 ∧
    (∃ p : ℤ, n + 7 = 9 * p) ∧
    (∃ q : ℤ, n - 9 = 7 * q)) → m ≤ n) ∧
  m = 128 :=
by sorry

end smallest_m_satisfying_conditions_l868_86895


namespace rectangle_area_l868_86811

theorem rectangle_area (a b : ℝ) (h1 : (a + b)^2 = 16) (h2 : (a - b)^2 = 4) : a * b = 3 := by
  sorry

end rectangle_area_l868_86811


namespace arman_sister_age_ratio_l868_86883

/-- Given Arman and his sister's ages at different points in time, prove the ratio of their current ages -/
theorem arman_sister_age_ratio :
  ∀ (sister_age_4_years_ago : ℕ) (arman_age_4_years_future : ℕ),
    sister_age_4_years_ago = 2 →
    arman_age_4_years_future = 40 →
    (arman_age_4_years_future - 4) / (sister_age_4_years_ago + 4) = 6 :=
by
  sorry


end arman_sister_age_ratio_l868_86883


namespace ratio_inequality_not_always_true_l868_86872

theorem ratio_inequality_not_always_true :
  ¬ (∀ (a b c d : ℝ), (a / b = c / d) → (a > b → c > d)) := by
  sorry

end ratio_inequality_not_always_true_l868_86872


namespace penultimate_digit_of_power_of_three_is_even_l868_86824

/-- The second to last digit of a natural number -/
def penultimate_digit (n : ℕ) : ℕ :=
  (n / 10) % 10

/-- Predicate for even numbers -/
def is_even (n : ℕ) : Prop :=
  ∃ k, n = 2 * k

theorem penultimate_digit_of_power_of_three_is_even (n : ℕ) (h : n ≥ 3) :
  is_even (penultimate_digit (3^n)) :=
sorry

end penultimate_digit_of_power_of_three_is_even_l868_86824


namespace sin_two_theta_plus_pi_sixth_l868_86825

theorem sin_two_theta_plus_pi_sixth (θ : Real) 
  (h : 7 * Real.sqrt 3 * Real.sin θ = 1 + 7 * Real.cos θ) : 
  Real.sin (2 * θ + π / 6) = 97 / 98 := by
  sorry

end sin_two_theta_plus_pi_sixth_l868_86825


namespace arithmetic_sequence_sum_squared_l868_86873

theorem arithmetic_sequence_sum_squared (a₁ : ℕ) (d : ℕ) (n : ℕ) :
  let seq := List.range n |>.map (fun i => a₁ + i * d)
  3 * (seq.sum)^2 = 1520832 :=
by
  sorry

end arithmetic_sequence_sum_squared_l868_86873


namespace count_integers_in_range_l868_86863

theorem count_integers_in_range : ∃ (S : Finset ℤ), 
  (∀ n : ℤ, n ∈ S ↔ -12 * Real.sqrt Real.pi ≤ (n : ℝ)^2 ∧ (n : ℝ)^2 ≤ 15 * Real.pi) ∧ 
  Finset.card S = 13 := by
  sorry

end count_integers_in_range_l868_86863


namespace lemoine_point_minimizes_distance_sum_l868_86861

/-- Given a triangle ABC, this theorem proves that the sum of squares of distances
    from any point to the sides of the triangle is minimized when the distances
    are proportional to the sides, and this point is the Lemoine point. -/
theorem lemoine_point_minimizes_distance_sum (a b c : ℝ) (S_ABC : ℝ) :
  let f (x y z : ℝ) := x^2 + y^2 + z^2
  ∀ (x y z : ℝ), a * x + b * y + c * z = 2 * S_ABC →
  f x y z ≥ f ((2 * S_ABC * a) / (a^2 + b^2 + c^2))
              ((2 * S_ABC * b) / (a^2 + b^2 + c^2))
              ((2 * S_ABC * c) / (a^2 + b^2 + c^2)) := by
  sorry

end lemoine_point_minimizes_distance_sum_l868_86861


namespace arithmetic_geometric_mean_square_sum_l868_86898

theorem arithmetic_geometric_mean_square_sum (x y : ℝ) 
  (h_arithmetic : (x + y) / 2 = 20)
  (h_geometric : Real.sqrt (x * y) = Real.sqrt 125) :
  x^2 + y^2 = 1350 := by
sorry

end arithmetic_geometric_mean_square_sum_l868_86898


namespace inequality_solution_l868_86837

def solution_set (a : ℝ) : Set ℝ :=
  {x | x^2 + (1 - a) * x - a < 0}

theorem inequality_solution (a : ℝ) :
  solution_set a = 
    if a > -1 then
      {x | -1 < x ∧ x < a}
    else if a < -1 then
      {x | a < x ∧ x < -1}
    else
      ∅ :=
by sorry

end inequality_solution_l868_86837


namespace purple_pants_count_l868_86884

/-- Represents the number of shirts Teairra has -/
def total_shirts : ℕ := 5

/-- Represents the number of pants Teairra has -/
def total_pants : ℕ := 24

/-- Represents the number of plaid shirts Teairra has -/
def plaid_shirts : ℕ := 3

/-- Represents the number of items that are neither plaid nor purple -/
def neither_plaid_nor_purple : ℕ := 21

/-- Represents the number of purple pants Teairra has -/
def purple_pants : ℕ := total_pants - (neither_plaid_nor_purple - (total_shirts - plaid_shirts))

theorem purple_pants_count : purple_pants = 5 := by
  sorry

end purple_pants_count_l868_86884


namespace line_intersects_circle_l868_86805

-- Define the line equation
def line_equation (a x y : ℝ) : Prop := a * x - y - a + 3 = 0

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y - 4 = 0

-- Theorem stating that the line intersects the circle for any real a
theorem line_intersects_circle (a : ℝ) : 
  ∃ x y : ℝ, line_equation a x y ∧ circle_equation x y := by
  sorry

end line_intersects_circle_l868_86805


namespace banana_kiwi_equivalence_l868_86885

-- Define the cost relationship between fruits
def cost_relation (banana pear kiwi : ℕ) : Prop :=
  4 * banana = 3 * pear ∧ 9 * pear = 6 * kiwi

-- Theorem statement
theorem banana_kiwi_equivalence :
  ∀ (banana pear kiwi : ℕ), cost_relation banana pear kiwi → 24 * banana = 12 * kiwi :=
by
  sorry

end banana_kiwi_equivalence_l868_86885


namespace adam_final_score_l868_86803

/-- Calculates the final score in a trivia game based on the given conditions --/
def calculate_final_score (
  science_correct : ℕ)
  (history_correct : ℕ)
  (sports_correct : ℕ)
  (literature_correct : ℕ)
  (science_points : ℕ)
  (history_points : ℕ)
  (sports_points : ℕ)
  (literature_points : ℕ)
  (history_multiplier : ℕ)
  (literature_penalty : ℕ) : ℕ :=
  science_correct * science_points +
  history_correct * history_points * history_multiplier +
  sports_correct * sports_points +
  literature_correct * (literature_points - literature_penalty)

/-- Theorem stating that Adam's final score is 99 points --/
theorem adam_final_score :
  calculate_final_score 5 3 1 1 10 5 15 7 2 3 = 99 := by
  sorry

#eval calculate_final_score 5 3 1 1 10 5 15 7 2 3

end adam_final_score_l868_86803


namespace popsicle_stick_sum_l868_86853

/-- The sum of popsicle sticks owned by two people -/
theorem popsicle_stick_sum (gino_sticks : ℕ) (your_sticks : ℕ) 
  (h1 : gino_sticks = 63) (h2 : your_sticks = 50) : 
  gino_sticks + your_sticks = 113 := by
  sorry

end popsicle_stick_sum_l868_86853


namespace field_fencing_l868_86852

/-- A rectangular field with one side of 20 feet and an area of 600 square feet
    requires 80 feet of fencing for the other three sides. -/
theorem field_fencing (length width : ℝ) : 
  length = 20 →
  length * width = 600 →
  length + 2 * width = 80 :=
by sorry

end field_fencing_l868_86852


namespace marathon_runners_finished_l868_86801

theorem marathon_runners_finished (total : ℕ) (difference : ℕ) (finished : ℕ) : 
  total = 1250 → 
  difference = 124 → 
  total = finished + (finished + difference) → 
  finished = 563 := by
sorry

end marathon_runners_finished_l868_86801


namespace y_derivative_is_zero_l868_86844

noncomputable def y (x : ℝ) : ℝ :=
  5 * x - Real.log (1 + Real.sqrt (1 - Real.exp (10 * x))) - Real.exp (-5 * x) * Real.arcsin (Real.exp (5 * x))

theorem y_derivative_is_zero :
  ∀ x : ℝ, deriv y x = 0 :=
by sorry

end y_derivative_is_zero_l868_86844


namespace cards_distribution_l868_86835

theorem cards_distribution (total_cards : Nat) (num_people : Nat) 
  (h1 : total_cards = 72) 
  (h2 : num_people = 10) : 
  (num_people - (total_cards % num_people)) = 8 := by
  sorry

end cards_distribution_l868_86835


namespace perpendicular_lines_product_sum_zero_l868_86874

/-- Two lines in the plane -/
structure Line where
  A : ℝ
  B : ℝ
  C : ℝ

/-- Perpendicularity of two lines -/
def perpendicular (l₁ l₂ : Line) : Prop :=
  l₁.A * l₂.A + l₁.B * l₂.B = 0

/-- Theorem: If two lines are perpendicular, then the sum of the products of their coefficients is zero -/
theorem perpendicular_lines_product_sum_zero (l₁ l₂ : Line) :
  perpendicular l₁ l₂ → l₁.A * l₂.A + l₁.B * l₂.B = 0 := by
  sorry

end perpendicular_lines_product_sum_zero_l868_86874


namespace f_formula_f_min_f_max_l868_86865

/-- A quadratic function satisfying certain conditions -/
def f (x : ℝ) : ℝ := sorry

/-- The conditions for the quadratic function -/
axiom f_quad : ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c ∧ a ≠ 0
axiom f_zero : f 0 = 1
axiom f_diff (x : ℝ) : f (x + 1) - f x = 2 * x

/-- Theorem: The quadratic function f(x) is x^2 - x + 1 -/
theorem f_formula : ∀ x, f x = x^2 - x + 1 := sorry

/-- Theorem: The minimum value of f(x) on [-1, 1] is 3/4 -/
theorem f_min : Set.Icc (-1 : ℝ) 1 ⊆ f ⁻¹' (Set.Icc (3/4 : ℝ) (f (-1))) := sorry

/-- Theorem: The maximum value of f(x) on [-1, 1] is 3 -/
theorem f_max : ∀ x ∈ Set.Icc (-1 : ℝ) 1, f x ≤ 3 := sorry

end f_formula_f_min_f_max_l868_86865


namespace wendy_furniture_time_l868_86876

/-- Given the number of chairs, tables, and total time spent, 
    calculate the time spent on each piece of furniture. -/
def time_per_piece (chairs : ℕ) (tables : ℕ) (total_time : ℕ) : ℚ :=
  total_time / (chairs + tables)

/-- Theorem: For Wendy's furniture assembly, 
    the time spent on each piece is 6 minutes. -/
theorem wendy_furniture_time :
  let chairs := 4
  let tables := 4
  let total_time := 48
  time_per_piece chairs tables total_time = 6 := by
  sorry

end wendy_furniture_time_l868_86876


namespace quadratic_equation_solution_l868_86860

theorem quadratic_equation_solution :
  let f (x : ℝ) := x^2 - (2/3)*x - 1
  ∃ (x₁ x₂ : ℝ), 
    f x₁ = 0 ∧ 
    f x₂ = 0 ∧ 
    x₁ = (Real.sqrt 10)/3 + 1/3 ∧ 
    x₂ = -(Real.sqrt 10)/3 + 1/3 := by
  sorry

end quadratic_equation_solution_l868_86860


namespace archies_backyard_sod_l868_86827

/-- The area of sod needed for Archie's backyard -/
def sod_area (backyard_length backyard_width shed_length shed_width : ℕ) : ℕ :=
  backyard_length * backyard_width - shed_length * shed_width

/-- Theorem stating the correct amount of sod needed for Archie's backyard -/
theorem archies_backyard_sod : sod_area 20 13 3 5 = 245 := by
  sorry

end archies_backyard_sod_l868_86827


namespace regular_polygon_sides_l868_86859

theorem regular_polygon_sides : ∃ (n : ℕ), n > 2 ∧ (2 * n - n * (n - 3) / 2 = 0) ↔ n = 7 := by
  sorry

end regular_polygon_sides_l868_86859


namespace compound_molecular_weight_l868_86855

/-- The atomic weight of Barium in g/mol -/
def atomic_weight_Ba : ℝ := 137.33

/-- The atomic weight of Sulphur in g/mol -/
def atomic_weight_S : ℝ := 32.07

/-- The atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The number of Barium atoms in the compound -/
def num_Ba : ℕ := 1

/-- The number of Sulphur atoms in the compound -/
def num_S : ℕ := 1

/-- The number of Oxygen atoms in the compound -/
def num_O : ℕ := 4

/-- The molecular weight of the compound in g/mol -/
def molecular_weight : ℝ := 
  num_Ba * atomic_weight_Ba + num_S * atomic_weight_S + num_O * atomic_weight_O

theorem compound_molecular_weight : 
  molecular_weight = 233.40 :=
by sorry

end compound_molecular_weight_l868_86855


namespace scientific_notation_exponent_is_integer_l868_86845

theorem scientific_notation_exponent_is_integer (x : ℝ) (A : ℝ) (N : ℝ) :
  x > 10 →
  x = A * 10^N →
  1 ≤ A →
  A < 10 →
  ∃ n : ℤ, N = n := by
  sorry

end scientific_notation_exponent_is_integer_l868_86845


namespace min_value_problem_l868_86897

theorem min_value_problem (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 2) :
  (1/3 : ℝ) * x^3 + y^2 + z ≥ 13/12 :=
sorry

end min_value_problem_l868_86897


namespace floor_length_percentage_l868_86880

/-- Proves that for a rectangular floor with length 20 meters, if the total cost to paint the floor
    at 3 currency units per square meter is 400 currency units, then the length is 200% more than
    the breadth. -/
theorem floor_length_percentage (breadth : ℝ) (percentage : ℝ) : 
  breadth > 0 →
  percentage > 0 →
  20 = breadth * (1 + percentage / 100) →
  400 = 3 * (20 * breadth) →
  percentage = 200 := by
sorry

end floor_length_percentage_l868_86880


namespace max_sphere_ratio_l868_86829

/-- Represents the configuration of spheres within two cones as described in the problem -/
structure SpheresInCones where
  r : ℝ  -- radius of the first two identical spheres
  x : ℝ  -- radius of the third sphere
  R : ℝ  -- radius of the base of the cones
  h : ℝ  -- height of each cone
  s : ℝ  -- slant height of each cone

/-- The conditions given in the problem -/
def problem_conditions (config : SpheresInCones) : Prop :=
  config.r > 0 ∧
  config.x > 0 ∧
  config.R > 0 ∧
  config.h > 0 ∧
  config.s > 0 ∧
  config.h = config.s / 2 ∧
  config.R = 3 * config.r

/-- The theorem stating the maximum ratio of the third sphere's radius to the first sphere's radius -/
theorem max_sphere_ratio (config : SpheresInCones) 
  (h : problem_conditions config) :
  ∃ (t : ℝ), t = config.x / config.r ∧ 
             t ≤ (7 - Real.sqrt 22) / 3 ∧
             ∀ (t' : ℝ), t' = config.x / config.r → t' ≤ t :=
sorry

end max_sphere_ratio_l868_86829


namespace mod_equivalence_solution_l868_86808

theorem mod_equivalence_solution : ∃ (n : ℕ), n ≤ 9 ∧ n ≡ -2023 [ZMOD 10] ∧ n = 7 := by
  sorry

end mod_equivalence_solution_l868_86808


namespace parabola_equation_l868_86856

/-- A parabola with directrix x = -7 has the standard equation y² = 28x -/
theorem parabola_equation (p : ℝ × ℝ → Prop) : 
  (∀ x y, p (x, y) ↔ x = -7) →  -- directrix equation
  (∃ k, ∀ x y, p (x, y) ↔ y^2 = 4 * k * x ∧ k > 0) →  -- general form of parabola equation
  (∀ x y, p (x, y) ↔ y^2 = 28 * x) :=  -- standard equation to be proved
by sorry

end parabola_equation_l868_86856


namespace smallest_angle_sum_l868_86816

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angle_A : ℝ
  angle_B : ℝ
  angle_C : ℝ
  sum_angles : angle_A + angle_B + angle_C = 180
  scalene : a ≠ b ∧ b ≠ c ∧ a ≠ c

-- Define the problem conditions
def problem_triangle (t : Triangle) : Prop :=
  (t.angle_A = 45 ∨ t.angle_B = 45 ∨ t.angle_C = 45) ∧
  (180 - t.angle_A = 135 ∨ 180 - t.angle_B = 135 ∨ 180 - t.angle_C = 135)

-- Theorem statement
theorem smallest_angle_sum (t : Triangle) (h : problem_triangle t) :
  ∃ x y, x ≤ t.angle_A ∧ x ≤ t.angle_B ∧ x ≤ t.angle_C ∧
         y ≤ t.angle_A ∧ y ≤ t.angle_B ∧ y ≤ t.angle_C ∧
         x + y = 90 :=
sorry

end smallest_angle_sum_l868_86816


namespace equation_solution_l868_86891

theorem equation_solution (x : ℝ) : 
  (2*x - 3) / (x + 4) = (3*x + 1) / (2*x - 5) ↔ 
  x = (29 + Real.sqrt 797) / 2 ∨ x = (29 - Real.sqrt 797) / 2 :=
by sorry

end equation_solution_l868_86891


namespace every_real_has_cube_root_real_number_line_bijection_correct_statements_l868_86846

-- Statement 1: Every real number has a cube root
theorem every_real_has_cube_root : ∀ x : ℝ, ∃ y : ℝ, y^3 = x := by sorry

-- Statement 2: Bijection between real numbers and points on a number line
theorem real_number_line_bijection : ∃ f : ℝ → ℝ, Function.Bijective f := by sorry

-- Main theorem combining both statements
theorem correct_statements :
  (∀ x : ℝ, ∃ y : ℝ, y^3 = x) ∧ (∃ f : ℝ → ℝ, Function.Bijective f) := by sorry

end every_real_has_cube_root_real_number_line_bijection_correct_statements_l868_86846


namespace hyperbola_sum_l868_86899

/-- Given a hyperbola with center (-2, 0), one focus at (-2 + √41, 0), and one vertex at (-7, 0),
    prove that h + k + a + b = 7, where (h, k) is the center, a is the distance from the center
    to a vertex, and b is the length of the conjugate axis. -/
theorem hyperbola_sum (h k a b c : ℝ) : 
  h = -2 ∧ 
  k = 0 ∧ 
  (h + Real.sqrt 41 - h)^2 = c^2 ∧
  (h - 5 - h)^2 = a^2 ∧
  c^2 = a^2 + b^2 →
  h + k + a + b = 7 := by sorry

end hyperbola_sum_l868_86899


namespace part_one_part_two_l868_86868

-- Define the sets A and B
def A : Set ℝ := {x | x - 2 ≥ 0}
def B (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < a + 1}

-- Define the complement of B in ℝ
def CompB (a : ℝ) : Set ℝ := {x | x ≤ a - 1 ∨ x ≥ a + 1}

-- Part I
theorem part_one : A ∩ (CompB 2) = {x | x ≥ 3} := by sorry

-- Part II
theorem part_two : ∀ a : ℝ, B a ⊆ A → a ≥ 3 := by sorry

end part_one_part_two_l868_86868


namespace difference_girls_boys_l868_86836

/-- The number of male students in village A -/
def male_A : ℕ := 204

/-- The number of female students in village A -/
def female_A : ℕ := 468

/-- The number of male students in village B -/
def male_B : ℕ := 334

/-- The number of female students in village B -/
def female_B : ℕ := 516

/-- The number of male students in village C -/
def male_C : ℕ := 427

/-- The number of female students in village C -/
def female_C : ℕ := 458

/-- The number of male students in village D -/
def male_D : ℕ := 549

/-- The number of female students in village D -/
def female_D : ℕ := 239

/-- The total number of male students in all villages -/
def total_males : ℕ := male_A + male_B + male_C + male_D

/-- The total number of female students in all villages -/
def total_females : ℕ := female_A + female_B + female_C + female_D

/-- Theorem: The difference between the total number of girls and boys in the town is 167 -/
theorem difference_girls_boys : total_females - total_males = 167 := by
  sorry

end difference_girls_boys_l868_86836


namespace inverse_inequality_for_negative_numbers_l868_86869

theorem inverse_inequality_for_negative_numbers (a b : ℝ) (h1 : a < b) (h2 : b < 0) :
  ¬(1 / a < 1 / b) :=
by sorry

end inverse_inequality_for_negative_numbers_l868_86869


namespace fifty_third_number_is_61_l868_86840

def adjustedSequence (n : ℕ) : ℕ :=
  n + (n - 1) / 4

theorem fifty_third_number_is_61 :
  adjustedSequence 53 = 61 := by
  sorry

end fifty_third_number_is_61_l868_86840


namespace sqrt_81_equals_3_to_m_l868_86892

theorem sqrt_81_equals_3_to_m (m : ℝ) : (81 : ℝ)^(1/2) = 3^m → m = 2 := by
  sorry

end sqrt_81_equals_3_to_m_l868_86892


namespace arithmetic_sqrt_of_nine_l868_86894

-- Define the arithmetic square root function
noncomputable def arithmeticSqrt (x : ℝ) : ℝ := 
  Real.sqrt x

-- State the theorem
theorem arithmetic_sqrt_of_nine : arithmeticSqrt 9 = 3 := by
  sorry

end arithmetic_sqrt_of_nine_l868_86894


namespace expression_evaluation_l868_86843

theorem expression_evaluation (a b c : ℝ) (ha : a = 12) (hb : b = 14) (hc : c = 18) :
  (a^2 * (1/b - 1/c) + b^2 * (1/c - 1/a) + c^2 * (1/a - 1/b)) /
  (a * (1/b - 1/c) + b * (1/c - 1/a) + c * (1/a - 1/b)) = a + b + c :=
by sorry

end expression_evaluation_l868_86843


namespace system_unique_solution_l868_86875

theorem system_unique_solution (a b c : ℝ) : 
  (a^2 + 3*a + 1 = (b + c) / 2) ∧ 
  (b^2 + 3*b + 1 = (a + c) / 2) ∧ 
  (c^2 + 3*c + 1 = (a + b) / 2) → 
  (a = -1 ∧ b = -1 ∧ c = -1) := by
  sorry

end system_unique_solution_l868_86875


namespace complex_cube_sum_l868_86833

theorem complex_cube_sum (w z : ℂ) (h1 : Complex.abs (w + z) = 2) (h2 : Complex.abs (w^2 + z^2) = 8) :
  Complex.abs (w^3 + z^3) = 20 := by
  sorry

end complex_cube_sum_l868_86833


namespace isosceles_triangle_perimeter_l868_86864

/-- An isosceles triangle with sides a, b, and c, where two sides are equal. -/
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  isosceles : (a = b ∧ a ≠ c) ∨ (a = c ∧ a ≠ b) ∨ (b = c ∧ b ≠ a)

/-- The perimeter of a triangle. -/
def perimeter (t : IsoscelesTriangle) : ℝ := t.a + t.b + t.c

/-- Theorem: The perimeter of an isosceles triangle with one side equal to 4
    and another side equal to 6 is either 14 or 16. -/
theorem isosceles_triangle_perimeter :
  ∀ t : IsoscelesTriangle, ((t.a = 4 ∨ t.b = 4 ∨ t.c = 4) ∧ (t.a = 6 ∨ t.b = 6 ∨ t.c = 6)) →
  (perimeter t = 14 ∨ perimeter t = 16) :=
by sorry

end isosceles_triangle_perimeter_l868_86864


namespace road_trip_distance_l868_86834

/-- Proves that given the conditions of the road trip, the first day's distance is 200 miles -/
theorem road_trip_distance (total_distance : ℝ) (day1 : ℝ) :
  total_distance = 525 →
  total_distance = day1 + (3/4 * day1) + (1/2 * (day1 + (3/4 * day1))) →
  day1 = 200 := by
  sorry

end road_trip_distance_l868_86834


namespace f_monotone_increasing_when_a_eq_1_f_range_of_a_l868_86815

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := 4 * a * x^3 + 3 * |a - 1| * x^2 + 2 * a * x - a

-- Theorem 1: Monotonicity when a = 1
theorem f_monotone_increasing_when_a_eq_1 :
  ∀ x y : ℝ, x < y → (f 1 x) < (f 1 y) :=
sorry

-- Theorem 2: Range of a
theorem f_range_of_a :
  {a : ℝ | ∀ x ∈ Set.Icc 0 1, |f a x| ≤ f a 1} = Set.Ici (-3/4) :=
sorry

end f_monotone_increasing_when_a_eq_1_f_range_of_a_l868_86815


namespace a_2009_equals_7_l868_86854

/-- Defines the array structure as described in the problem -/
def array_element (n i : ℕ) : ℚ :=
  if i ≤ n then i / (n + 1 - i) else 0

/-- Defines the index of the last element in the nth array -/
def last_index (n : ℕ) : ℕ :=
  n * (n + 1) / 2

/-- The main theorem stating that the 2009th element of the array is 7 -/
theorem a_2009_equals_7 : array_element 63 56 = 7 := by sorry

end a_2009_equals_7_l868_86854


namespace multiplication_addition_equality_l868_86847

theorem multiplication_addition_equality : 3.5 * 0.3 + 1.2 * 0.4 = 1.53 := by
  sorry

end multiplication_addition_equality_l868_86847


namespace expression_evaluation_l868_86877

theorem expression_evaluation : ((-2)^2)^(1^(0^2)) + 3^(0^(1^2)) = 5 := by
  sorry

end expression_evaluation_l868_86877


namespace isosceles_triangle_angle_l868_86862

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the property of an isosceles triangle
def isIsosceles (t : Triangle) : Prop :=
  dist t.A t.B = dist t.A t.C

-- Define the angle measure in degrees
def angleMeasure (t : Triangle) (vertex : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem isosceles_triangle_angle (t : Triangle) :
  isIsosceles t →
  angleMeasure t t.B = 55 →
  angleMeasure t t.A = 70 := by
  sorry


end isosceles_triangle_angle_l868_86862


namespace f_decreasing_interval_l868_86881

-- Define the derivative of f
def f' (x : ℝ) : ℝ := x^2 - 4*x + 3

-- State the theorem
theorem f_decreasing_interval :
  ∀ f : ℝ → ℝ, (∀ x, deriv f x = f' x) →
  ∀ x ∈ Set.Ioo 0 2, deriv (fun y ↦ f (y + 1)) x < 0 :=
by sorry

end f_decreasing_interval_l868_86881


namespace largest_divisor_of_f_l868_86810

def f (n : ℕ) : ℕ := (2 * n + 7) * 3^n + 9

theorem largest_divisor_of_f :
  ∀ m : ℕ, (∀ n : ℕ, m ∣ f n) → m ≤ 36 ∧
  ∀ n : ℕ, 36 ∣ f n :=
sorry

end largest_divisor_of_f_l868_86810


namespace wallpaper_three_layers_l868_86890

/-- Given wallpaper covering conditions, prove the area covered by three layers -/
theorem wallpaper_three_layers
  (total_area : ℝ)
  (wall_area : ℝ)
  (two_layer_area : ℝ)
  (h1 : total_area = 300)
  (h2 : wall_area = 180)
  (h3 : two_layer_area = 30)
  : ∃ (three_layer_area : ℝ),
    three_layer_area = total_area - (wall_area - two_layer_area + two_layer_area) ∧
    three_layer_area = 120 :=
by sorry

end wallpaper_three_layers_l868_86890


namespace f_inequality_l868_86857

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1|

-- Define the set M
def M : Set ℝ := {x | x < -1 ∨ x > 1}

-- State the theorem
theorem f_inequality (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) : f (a * b) > f a - f (-b) := by
  sorry

end f_inequality_l868_86857


namespace log_five_twelve_l868_86878

theorem log_five_twelve (a b : ℝ) (h1 : Real.log 2 = a * Real.log 10) (h2 : Real.log 3 = b * Real.log 10) :
  Real.log 12 / Real.log 5 = (2 * a + b) / (1 - a) := by
  sorry

end log_five_twelve_l868_86878


namespace age_puzzle_l868_86817

theorem age_puzzle (A : ℕ) (x : ℕ) (h1 : A = 24) (h2 : x = 3) :
  4 * (A + x) - 4 * (A - 3) = A := by
  sorry

end age_puzzle_l868_86817


namespace polynomial_remainder_l868_86832

theorem polynomial_remainder (Q : ℝ → ℝ) (h1 : Q 20 = 80) (h2 : Q 100 = 20) :
  ∃ R : ℝ → ℝ, ∀ x, Q x = (x - 20) * (x - 100) * R x + (-3/4 * x + 95) := by
sorry

end polynomial_remainder_l868_86832


namespace darnel_distance_difference_l868_86841

theorem darnel_distance_difference :
  let sprint_distance : ℚ := 875 / 1000
  let jog_distance : ℚ := 75 / 100
  sprint_distance - jog_distance = 125 / 1000 :=
by sorry

end darnel_distance_difference_l868_86841


namespace two_red_balls_probability_l868_86879

/-- Represents a bag of colored balls -/
structure Bag where
  red : Nat
  white : Nat

/-- Calculates the probability of drawing a red ball from a given bag -/
def redProbability (bag : Bag) : Rat :=
  bag.red / (bag.red + bag.white)

/-- Theorem: The probability of drawing two red balls, one from each bag, is 1/9 -/
theorem two_red_balls_probability
  (bagA : Bag)
  (bagB : Bag)
  (hA : bagA = { red := 4, white := 2 })
  (hB : bagB = { red := 1, white := 5 }) :
  redProbability bagA * redProbability bagB = 1 / 9 := by
  sorry


end two_red_balls_probability_l868_86879


namespace pet_store_dogs_l868_86822

theorem pet_store_dogs (dogs : ℕ) : 
  dogs + (dogs / 2) + (2 * dogs) + (3 * dogs) = 39 → dogs = 6 := by
  sorry

end pet_store_dogs_l868_86822


namespace mike_debt_proof_l868_86821

/-- Calculates the final amount owed after compound interest is applied -/
def final_amount (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Proves that the final amount owed is approximately $530.604 -/
theorem mike_debt_proof (ε : ℝ) (h_ε : ε > 0) :
  ∃ (result : ℝ), 
    final_amount 500 0.02 3 = result ∧ 
    abs (result - 530.604) < ε :=
by
  sorry

#eval final_amount 500 0.02 3

end mike_debt_proof_l868_86821


namespace missing_sale_proof_l868_86802

/-- Calculates the missing sale amount to achieve a desired average -/
def calculate_missing_sale (sales : List ℕ) (required_sale : ℕ) (desired_average : ℕ) : ℕ :=
  let total_sales := desired_average * (sales.length + 2)
  let known_sales := sales.sum + required_sale
  total_sales - known_sales

theorem missing_sale_proof (sales : List ℕ) (required_sale : ℕ) (desired_average : ℕ) 
  (h1 : sales = [6335, 6927, 6855, 7230])
  (h2 : required_sale = 5091)
  (h3 : desired_average = 6500) :
  calculate_missing_sale sales required_sale desired_average = 6562 := by
  sorry

#eval calculate_missing_sale [6335, 6927, 6855, 7230] 5091 6500

end missing_sale_proof_l868_86802


namespace women_to_men_ratio_l868_86819

/-- Given an event with guests, prove the ratio of women to men --/
theorem women_to_men_ratio 
  (total_guests : ℕ) 
  (num_men : ℕ) 
  (num_children_after : ℕ) 
  (h1 : total_guests = 80) 
  (h2 : num_men = 40) 
  (h3 : num_children_after = 30) :
  (total_guests - num_men - (num_children_after - 10)) / num_men = 1 / 2 :=
by
  sorry

end women_to_men_ratio_l868_86819


namespace strawberry_calculation_l868_86848

/-- Converts kilograms and grams to total grams -/
def to_grams (kg : ℕ) (g : ℕ) : ℕ := kg * 1000 + g

/-- Calculates remaining strawberries in grams -/
def remaining_strawberries (total_kg : ℕ) (total_g : ℕ) (given_kg : ℕ) (given_g : ℕ) : ℕ :=
  to_grams total_kg total_g - to_grams given_kg given_g

theorem strawberry_calculation :
  remaining_strawberries 3 300 1 900 = 1400 := by
  sorry

end strawberry_calculation_l868_86848


namespace distribute_5_4_l868_86826

/-- The number of ways to distribute n distinct objects into k indistinguishable containers,
    allowing empty containers. -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 5 distinct objects into 4 indistinguishable containers,
    allowing empty containers, is 51. -/
theorem distribute_5_4 : distribute 5 4 = 51 := by sorry

end distribute_5_4_l868_86826


namespace hyperbola_equation_l868_86818

-- Define the hyperbola C
def hyperbola_C (x y a b : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1 ∧ a > 0 ∧ b > 0

-- Define the asymptote
def asymptote (x y : ℝ) : Prop :=
  y = (Real.sqrt 5 / 2) * x

-- Define the ellipse
def ellipse (x y : ℝ) : Prop :=
  x^2 / 12 + y^2 / 3 = 1

-- Theorem statement
theorem hyperbola_equation :
  ∀ x y a b : ℝ,
  hyperbola_C x y a b →
  (∃ x₀ y₀, asymptote x₀ y₀) →
  (∃ x₁ y₁, ellipse x₁ y₁ ∧ 
    (x₁ - x)^2 + (y₁ - y)^2 = (x₁ + x)^2 + (y₁ + y)^2) →
  x^2 / 4 - y^2 / 5 = 1 :=
by sorry

end hyperbola_equation_l868_86818


namespace total_time_is_383_l868_86807

def total_time (mac_download : ℕ) (ny_audio_glitch : ℕ) (ny_video_glitch : ℕ) 
  (berlin_audio_glitch : ℕ) (berlin_video_glitch : ℕ) (tokyo_audio_glitch : ℕ) 
  (tokyo_video_glitch : ℕ) (sydney_audio_glitch : ℕ) : ℕ :=
  let windows_download := 3 * mac_download
  let ny_glitch_time := 2 * ny_audio_glitch + ny_video_glitch
  let ny_total := ny_glitch_time + 3 * ny_glitch_time
  let berlin_glitch_time := 3 * berlin_audio_glitch + 2 * berlin_video_glitch
  let berlin_total := berlin_glitch_time + 2 * berlin_glitch_time
  let tokyo_glitch_time := tokyo_audio_glitch + 2 * tokyo_video_glitch
  let tokyo_total := tokyo_glitch_time + 4 * tokyo_glitch_time
  let sydney_glitch_time := 2 * sydney_audio_glitch
  let sydney_total := sydney_glitch_time + 5 * sydney_glitch_time
  mac_download + windows_download + ny_total + berlin_total + tokyo_total + sydney_total

theorem total_time_is_383 : 
  total_time 10 6 8 4 5 7 9 6 = 383 := by
  sorry

end total_time_is_383_l868_86807


namespace instrument_probability_l868_86806

theorem instrument_probability (total : ℕ) (at_least_one : ℕ) (two_or_more : ℕ) :
  total = 800 →
  at_least_one = total / 5 →
  two_or_more = 32 →
  (at_least_one - two_or_more : ℚ) / total = 1 / 6.25 := by
  sorry

end instrument_probability_l868_86806


namespace intersection_and_union_when_m_is_3_intersection_empty_iff_l868_86887

-- Define sets A and B
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | m - 1 ≤ x ∧ x ≤ 2 * m + 1}

-- Part 1
theorem intersection_and_union_when_m_is_3 :
  (A ∩ B 3 = {x | 2 ≤ x ∧ x ≤ 5}) ∧
  ((Set.univ \ A) ∪ B 3 = {x | x < -2 ∨ x ≥ 2}) := by sorry

-- Part 2
theorem intersection_empty_iff :
  ∀ m : ℝ, A ∩ B m = ∅ ↔ m < -3/2 ∨ m > 6 := by sorry

end intersection_and_union_when_m_is_3_intersection_empty_iff_l868_86887


namespace frequency_of_defectives_example_l868_86870

/-- Given a sample of parts, calculate the frequency of defective parts -/
def frequency_of_defectives (total : ℕ) (defective : ℕ) : ℚ :=
  defective / total

/-- Theorem stating that for a sample of 500 parts with 8 defective parts, 
    the frequency of defective parts is 0.016 -/
theorem frequency_of_defectives_example : 
  frequency_of_defectives 500 8 = 16 / 1000 := by
  sorry

end frequency_of_defectives_example_l868_86870


namespace subtract_repeating_third_from_four_l868_86850

/-- The repeating decimal 0.3̅ -/
def repeating_third : ℚ := 1/3

/-- Proof that 4 - 0.3̅ = 11/3 -/
theorem subtract_repeating_third_from_four :
  4 - repeating_third = 11/3 := by sorry

end subtract_repeating_third_from_four_l868_86850


namespace geometric_sequence_a2_a4_condition_l868_86830

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

def is_monotonically_increasing (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, n < m → a n < a m

theorem geometric_sequence_a2_a4_condition (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (is_monotonically_increasing a → a 2 < a 4) ∧
  ¬(a 2 < a 4 → is_monotonically_increasing a) :=
by sorry

end geometric_sequence_a2_a4_condition_l868_86830


namespace sum_of_ages_l868_86849

/-- Given Bob's and Carol's ages, prove that their sum is 66 years. -/
theorem sum_of_ages (bob_age carol_age : ℕ) : 
  carol_age = 3 * bob_age + 2 →
  carol_age = 50 →
  bob_age = 16 →
  bob_age + carol_age = 66 := by
sorry

end sum_of_ages_l868_86849


namespace tan_sum_pi_fractions_l868_86882

theorem tan_sum_pi_fractions : Real.tan (π / 12) + Real.tan (5 * π / 12) = 4 := by
  sorry

end tan_sum_pi_fractions_l868_86882


namespace rays_number_l868_86896

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def sum_of_digits (n : ℕ) : ℕ := (n / 10) + (n % 10)

def reverse_digits (n : ℕ) : ℕ := 10 * (n % 10) + (n / 10)

theorem rays_number :
  ∃ n : ℕ,
    is_two_digit n ∧
    n > 4 * (sum_of_digits n) + 3 ∧
    n + 18 = reverse_digits n ∧
    n = 35 := by
  sorry

end rays_number_l868_86896


namespace inequality_range_l868_86820

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, x > 0 → x + 4 / x ≥ a) ↔ a ∈ Set.Iic 4 := by
  sorry

end inequality_range_l868_86820


namespace equation_solutions_l868_86866

theorem equation_solutions :
  (∃ x : ℝ, 9.9 + x = -18 ∧ x = -27.9) ∧
  (∃ x : ℝ, x - 8.8 = -8.8 ∧ x = 0) ∧
  (∃ x : ℚ, -3/4 + x = -1/4 ∧ x = 1/2) := by
  sorry

end equation_solutions_l868_86866


namespace first_part_interest_rate_l868_86831

/-- Proves that given the specified conditions, the interest rate of the first part is 3% -/
theorem first_part_interest_rate 
  (total_investment : ℝ) 
  (first_part : ℝ) 
  (second_part_rate : ℝ) 
  (total_interest : ℝ) : 
  total_investment = 4000 →
  first_part = 2800 →
  second_part_rate = 0.05 →
  total_interest = 144 →
  (first_part * (3 / 100) + (total_investment - first_part) * second_part_rate = total_interest) :=
by
  sorry

#check first_part_interest_rate

end first_part_interest_rate_l868_86831


namespace sandwiches_problem_l868_86842

theorem sandwiches_problem (S : ℚ) :
  (S > 0) →
  (3/4 * S - 1/8 * S - 1/4 * S - 5 = 4) →
  S = 24 := by
  sorry

end sandwiches_problem_l868_86842


namespace trigonometric_equation_solution_l868_86809

theorem trigonometric_equation_solution (x : ℝ) : 
  (∃ (n : ℤ), x = Real.pi / 2 * (2 * ↑n + 1)) ∨ 
  (∃ (k : ℤ), x = Real.pi / 18 * (4 * ↑k + 1)) ↔ 
  Real.sin (3 * x) + Real.sin (5 * x) = 2 * (Real.cos (2 * x))^2 - 2 * (Real.sin (3 * x))^2 := by
sorry

end trigonometric_equation_solution_l868_86809


namespace initial_workers_count_l868_86851

/-- The initial number of workers that can complete a job in 25 days, 
    where adding 10 more workers allows the job to be completed in 15 days -/
def initial_workers : ℕ :=
  sorry

/-- The total amount of work to be done -/
def total_work : ℝ :=
  sorry

theorem initial_workers_count : initial_workers = 15 := by
  sorry

end initial_workers_count_l868_86851


namespace triangle_properties_l868_86889

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  -- Triangle ABC exists
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧
  A + B + C = π →
  -- Given condition
  2 * a = Real.sqrt 3 * c * Real.sin A - a * Real.cos C →
  -- Part 1: Prove C = 2π/3
  C = 2 * π / 3 ∧
  -- Part 2: Prove maximum area is √3/4 when c = √3
  (c = Real.sqrt 3 →
    ∀ (a' b' : ℝ), 
      0 < a' ∧ 0 < b' ∧
      2 * a' = Real.sqrt 3 * c * Real.sin A - a' * Real.cos C →
      1/2 * a' * b' * Real.sin C ≤ Real.sqrt 3 / 4) :=
by sorry

end triangle_properties_l868_86889


namespace max_product_constrained_l868_86812

theorem max_product_constrained (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 3 * a + 2 * b = 1) :
  a * b ≤ 1 / 24 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 3 * a₀ + 2 * b₀ = 1 ∧ a₀ * b₀ = 1 / 24 :=
sorry

end max_product_constrained_l868_86812


namespace max_a_value_l868_86871

/-- The quadratic polynomial p(x) -/
def p (a : ℝ) (x : ℝ) : ℝ := (a - 1) * x^2 - (a - 1) * x + 2022

/-- The theorem stating the maximum value of a -/
theorem max_a_value : 
  (∀ x ∈ Set.Icc 0 1, -2022 ≤ p a x ∧ p a x ≤ 2022) → 
  a ≤ 16177 :=
by sorry

end max_a_value_l868_86871


namespace beluga_breath_interval_proof_l868_86886

/-- The average time (in minutes) between a bottle-nosed dolphin's air breaths -/
def dolphin_breath_interval : ℝ := 3

/-- The number of minutes in a 24-hour period -/
def minutes_per_day : ℝ := 24 * 60

/-- The ratio of dolphin breaths to beluga whale breaths in a 24-hour period -/
def breath_ratio : ℝ := 2.5

/-- The average time (in minutes) between a beluga whale's air breaths -/
def beluga_breath_interval : ℝ := 7.5

theorem beluga_breath_interval_proof :
  (minutes_per_day / dolphin_breath_interval) = breath_ratio * (minutes_per_day / beluga_breath_interval) :=
by sorry

end beluga_breath_interval_proof_l868_86886


namespace two_books_cost_l868_86888

/-- The cost of two books, where one is sold at a loss and the other at a gain --/
theorem two_books_cost (C₁ C₂ : ℝ) (h1 : C₁ = 274.1666666666667) 
  (h2 : C₁ * 0.85 = C₂ * 1.19) : 
  abs (C₁ + C₂ - 470) < 0.01 := by
  sorry

end two_books_cost_l868_86888


namespace order_of_trig_functions_l868_86813

theorem order_of_trig_functions : 
  let a := Real.sin (5 * Real.pi / 7)
  let b := Real.cos (2 * Real.pi / 7)
  let c := Real.tan (2 * Real.pi / 7)
  b < a ∧ a < c := by
  sorry

end order_of_trig_functions_l868_86813
