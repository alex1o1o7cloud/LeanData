import Mathlib

namespace one_sixth_percent_of_180_l39_3904

theorem one_sixth_percent_of_180 : (1 / 6 : ℚ) / 100 * 180 = 0.3 := by
  sorry

end one_sixth_percent_of_180_l39_3904


namespace sum_interior_angles_formula_l39_3932

/-- The sum of interior angles of a polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℝ :=
  (n - 2) * 180

/-- Theorem: For a polygon with n sides (where n ≥ 3), 
    the sum of interior angles is (n-2) * 180° -/
theorem sum_interior_angles_formula (n : ℕ) (h : n ≥ 3) : 
  sum_interior_angles n = (n - 2) * 180 := by
  sorry

#check sum_interior_angles_formula

end sum_interior_angles_formula_l39_3932


namespace max_single_player_salary_l39_3945

/-- Represents the number of players in a team -/
def num_players : ℕ := 18

/-- Represents the minimum salary for each player in dollars -/
def min_salary : ℕ := 20000

/-- Represents the maximum total salary for the team in dollars -/
def max_total_salary : ℕ := 800000

/-- Theorem stating the maximum possible salary for a single player -/
theorem max_single_player_salary :
  ∃ (max_salary : ℕ),
    max_salary = 460000 ∧
    max_salary + (num_players - 1) * min_salary = max_total_salary ∧
    ∀ (salary : ℕ),
      salary + (num_players - 1) * min_salary ≤ max_total_salary →
      salary ≤ max_salary :=
by sorry

end max_single_player_salary_l39_3945


namespace total_distance_driven_l39_3903

/-- The total distance driven by Renaldo and Ernesto -/
def total_distance (renaldo_distance : ℝ) (ernesto_distance : ℝ) : ℝ :=
  renaldo_distance + ernesto_distance

/-- Theorem stating the total distance driven by Renaldo and Ernesto -/
theorem total_distance_driven :
  let renaldo_distance : ℝ := 15
  let ernesto_distance : ℝ := (1/3 * renaldo_distance) + 7
  total_distance renaldo_distance ernesto_distance = 27 := by
  sorry

end total_distance_driven_l39_3903


namespace extra_workers_for_clay_soil_l39_3935

/-- Represents the digging problem with different soil types and worker requirements -/
structure DiggingProblem where
  sandy_workers : ℕ
  sandy_hours : ℕ
  clay_time_factor : ℕ
  new_hours : ℕ

/-- Calculates the number of extra workers needed for the clay soil digging task -/
def extra_workers_needed (p : DiggingProblem) : ℕ :=
  let sandy_man_hours := p.sandy_workers * p.sandy_hours
  let clay_man_hours := sandy_man_hours * p.clay_time_factor
  let total_workers_needed := clay_man_hours / p.new_hours
  total_workers_needed - p.sandy_workers

/-- Theorem stating that given the problem conditions, 75 extra workers are needed -/
theorem extra_workers_for_clay_soil : 
  let p : DiggingProblem := {
    sandy_workers := 45,
    sandy_hours := 8,
    clay_time_factor := 2,
    new_hours := 6
  }
  extra_workers_needed p = 75 := by sorry

end extra_workers_for_clay_soil_l39_3935


namespace top_is_nine_l39_3921

/-- Represents a valid labeling of the figure -/
structure Labeling where
  labels : Fin 9 → Fin 9
  bijective : Function.Bijective labels
  equal_sums : ∃ (s : ℕ), 
    (labels 0 + labels 1 + labels 3 + labels 4 = s) ∧
    (labels 1 + labels 2 + labels 4 + labels 5 = s) ∧
    (labels 0 + labels 3 + labels 6 = s) ∧
    (labels 1 + labels 4 + labels 7 = s) ∧
    (labels 2 + labels 5 + labels 8 = s) ∧
    (labels 3 + labels 4 + labels 5 = s)

/-- The theorem stating that the top number is always 9 in a valid labeling -/
theorem top_is_nine (l : Labeling) : l.labels 0 = 9 := by
  sorry

end top_is_nine_l39_3921


namespace boys_pass_percentage_l39_3991

/-- Proves that 28% of boys passed the examination given the problem conditions -/
theorem boys_pass_percentage (total_candidates : ℕ) (girls : ℕ) (girls_pass_rate : ℚ) (total_fail_rate : ℚ) :
  total_candidates = 2000 →
  girls = 900 →
  girls_pass_rate = 32 / 100 →
  total_fail_rate = 702 / 1000 →
  let boys := total_candidates - girls
  let total_pass := total_candidates * (1 - total_fail_rate)
  let girls_pass := girls * girls_pass_rate
  let boys_pass := total_pass - girls_pass
  (boys_pass / boys : ℚ) = 28 / 100 := by sorry

end boys_pass_percentage_l39_3991


namespace symmetry_about_xOy_plane_l39_3997

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The xOy plane in 3D space -/
def xOyPlane : Set Point3D := {p : Point3D | p.z = 0}

/-- Symmetry about the xOy plane -/
def symmetricAboutXOY (p : Point3D) : Point3D :=
  ⟨p.x, p.y, -p.z⟩

theorem symmetry_about_xOy_plane (p : Point3D) :
  symmetricAboutXOY p = ⟨p.x, p.y, -p.z⟩ := by
  sorry

#check symmetry_about_xOy_plane

end symmetry_about_xOy_plane_l39_3997


namespace tree_climbing_average_height_l39_3969

theorem tree_climbing_average_height : 
  let first_tree_height : ℝ := 1000
  let second_tree_height : ℝ := first_tree_height / 2
  let third_tree_height : ℝ := first_tree_height / 2
  let fourth_tree_height : ℝ := first_tree_height + 200
  let total_height : ℝ := first_tree_height + second_tree_height + third_tree_height + fourth_tree_height
  let num_trees : ℝ := 4
  (total_height / num_trees) = 800 := by sorry

end tree_climbing_average_height_l39_3969


namespace number_line_relations_l39_3995

theorem number_line_relations (a b : ℝ) (h1 : 1/2 < a) (h2 : a < 1) (h3 : 1/2 < b) (h4 : b < 1) :
  (1 < a + b ∧ a + b < 2) ∧ (a - b < 0) ∧ (1/4 < a * b ∧ a * b < 1) := by
  sorry

end number_line_relations_l39_3995


namespace ranch_cows_l39_3923

/-- The number of cows owned by We the People -/
def wtp_cows : ℕ := 17

/-- The number of cows owned by Happy Good Healthy Family -/
def hghf_cows : ℕ := 3 * wtp_cows + 2

/-- The total number of cows in the ranch -/
def total_cows : ℕ := wtp_cows + hghf_cows

theorem ranch_cows : total_cows = 70 := by
  sorry

end ranch_cows_l39_3923


namespace series_convergence_l39_3983

/-- The infinite sum of the given series converges to 2 -/
theorem series_convergence : 
  ∑' k : ℕ, (8 : ℝ)^k / ((4 : ℝ)^k - (3 : ℝ)^k) / ((4 : ℝ)^(k+1) - (3 : ℝ)^(k+1)) = 2 := by
  sorry

end series_convergence_l39_3983


namespace find_number_l39_3928

theorem find_number : ∃ x : ℤ, x - 29 + 64 = 76 ∧ x = 41 := by
  sorry

end find_number_l39_3928


namespace complex_square_one_plus_i_l39_3938

theorem complex_square_one_plus_i : (1 + Complex.I) ^ 2 = 2 * Complex.I := by sorry

end complex_square_one_plus_i_l39_3938


namespace book_cost_l39_3996

theorem book_cost (n₅ n₃ : ℕ) : 
  (n₅ + n₃ > 10) → 
  (n₅ + n₃ < 20) → 
  (5 * n₅ = 3 * n₃) → 
  (5 * n₅ = 30) := by
sorry

end book_cost_l39_3996


namespace replacement_paint_intensity_l39_3917

/-- Proves that the intensity of replacement paint is 25% given the original paint intensity,
    new paint intensity after mixing, and the fraction of original paint replaced. -/
theorem replacement_paint_intensity
  (original_intensity : ℝ)
  (new_intensity : ℝ)
  (replaced_fraction : ℝ)
  (h_original : original_intensity = 50)
  (h_new : new_intensity = 40)
  (h_replaced : replaced_fraction = 0.4)
  : (1 - replaced_fraction) * original_intensity + replaced_fraction * 25 = new_intensity :=
by sorry


end replacement_paint_intensity_l39_3917


namespace lines_are_parallel_l39_3987

/-- Two lines are parallel if they have the same slope -/
def parallel (m₁ b₁ m₂ b₂ : ℝ) : Prop := m₁ = m₂

/-- The first line: y = -2x + 1 -/
def line1 (x : ℝ) : ℝ := -2 * x + 1

/-- The second line: y = -2x + 3 -/
def line2 (x : ℝ) : ℝ := -2 * x + 3

/-- Theorem: The line y = -2x + 1 is parallel to the line y = -2x + 3 -/
theorem lines_are_parallel : parallel (-2) 1 (-2) 3 := by sorry

end lines_are_parallel_l39_3987


namespace tournament_probability_l39_3951

/-- The number of teams in the tournament -/
def num_teams : ℕ := 35

/-- The number of games each team plays -/
def games_per_team : ℕ := num_teams - 1

/-- The total number of games in the tournament -/
def total_games : ℕ := (num_teams * games_per_team) / 2

/-- The probability of a team winning a single game -/
def win_probability : ℚ := 1 / 2

/-- The number of possible outcomes in the tournament -/
def total_outcomes : ℕ := 2^total_games

/-- The number of ways to assign unique victory counts to all teams -/
def unique_victory_assignments : ℕ := num_teams.factorial

theorem tournament_probability : 
  (unique_victory_assignments : ℚ) / total_outcomes = (num_teams.factorial : ℚ) / 2^595 :=
sorry

end tournament_probability_l39_3951


namespace max_value_of_sum_l39_3966

theorem max_value_of_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = 1) :
  (1 / (a + 9 * b) + 1 / (9 * a + b)) ≤ 5 / 24 ∧
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ * b₀ = 1 ∧
    1 / (a₀ + 9 * b₀) + 1 / (9 * a₀ + b₀) = 5 / 24 :=
by sorry

end max_value_of_sum_l39_3966


namespace subset_properties_l39_3933

-- Define set A
def A : Set ℝ := {x | 0 < x ∧ x < 2}

-- Define the property that B is a subset of A
def is_subset_of_A (B : Set ℝ) : Prop := B ⊆ A

-- Theorem statement
theorem subset_properties (B : Set ℝ) (h : A ∩ B = B) :
  is_subset_of_A ∅ ∧
  is_subset_of_A {1} ∧
  is_subset_of_A A ∧
  ¬is_subset_of_A {x : ℝ | 0 ≤ x ∧ x ≤ 2} :=
sorry

end subset_properties_l39_3933


namespace hemisphere_cylinder_surface_area_l39_3964

/-- Given a hemisphere with base area 144π and a cylindrical extension of height 10 units
    with the same radius as the hemisphere, the total surface area of the combined object is 672π. -/
theorem hemisphere_cylinder_surface_area (r : ℝ) (h : ℝ) :
  r^2 * Real.pi = 144 * Real.pi →
  h = 10 →
  2 * Real.pi * r^2 + 2 * Real.pi * r * h + Real.pi * r^2 = 672 * Real.pi :=
by sorry

end hemisphere_cylinder_surface_area_l39_3964


namespace triangle_theorem_l39_3999

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the given conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a = 2 ∧
  (Real.cos t.C) * t.a * t.b = 1 ∧
  1/2 * t.a * t.b * (Real.sin t.C) = 1/2 ∧
  (Real.sin t.A) * (Real.cos t.A) = Real.sqrt 3 / 4

-- State the theorem
theorem triangle_theorem (t : Triangle) (h : triangle_conditions t) :
  t.C = Real.pi / 4 ∧ (t.c = Real.sqrt 6 ∨ t.c = 2 * Real.sqrt 6 / 3) :=
sorry

end triangle_theorem_l39_3999


namespace largest_n_for_trig_inequality_l39_3924

theorem largest_n_for_trig_inequality : 
  (∃ (n : ℕ), n > 0 ∧ ∀ (x : ℝ), (Real.sin x)^n + (Real.cos x)^n ≥ 1 / (n^2 : ℝ)) ∧ 
  (∀ (n : ℕ), n > 10 → ∃ (x : ℝ), (Real.sin x)^n + (Real.cos x)^n < 1 / (n^2 : ℝ)) :=
sorry

end largest_n_for_trig_inequality_l39_3924


namespace geometric_sequence_complex_l39_3968

def z₁ (a : ℝ) : ℂ := a + Complex.I
def z₂ (a : ℝ) : ℂ := 2*a + 2*Complex.I
def z₃ (a : ℝ) : ℂ := 3*a + 4*Complex.I

theorem geometric_sequence_complex (a : ℝ) :
  (Complex.abs (z₂ a))^2 = (Complex.abs (z₁ a)) * (Complex.abs (z₃ a)) → a = 0 := by
  sorry

end geometric_sequence_complex_l39_3968


namespace binary_arithmetic_equality_l39_3946

-- Define a function to convert binary to decimal
def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

-- Define the binary numbers
def b1101 : List Bool := [true, false, true, true]
def b1111 : List Bool := [true, true, true, true]
def b1001 : List Bool := [true, false, false, true]
def b10 : List Bool := [false, true]
def b1010 : List Bool := [false, true, false, true]

-- State the theorem
theorem binary_arithmetic_equality :
  (binary_to_decimal b1101 + binary_to_decimal b1111) -
  (binary_to_decimal b1001 * binary_to_decimal b10) =
  binary_to_decimal b1010 := by
  sorry


end binary_arithmetic_equality_l39_3946


namespace problem_statement_l39_3957

theorem problem_statement (a b : ℝ) : 
  |a - 2| + (b + 1/2)^2 = 0 → a^2022 * b^2023 = -1/2 := by sorry

end problem_statement_l39_3957


namespace value_range_of_function_l39_3901

theorem value_range_of_function : 
  ∀ (y : ℝ), (∃ (x : ℝ), y = (x^2 - 1) / (x^2 + 1)) ↔ -1 ≤ y ∧ y < 1 := by
  sorry

end value_range_of_function_l39_3901


namespace total_raised_l39_3950

/-- The amount of money raised by a local business for charity -/
def charity_fundraiser (num_tickets : ℕ) (ticket_price : ℚ) (donation1 : ℚ) (num_donation1 : ℕ) (donation2 : ℚ) : ℚ :=
  num_tickets * ticket_price + num_donation1 * donation1 + donation2

/-- Theorem stating the total amount raised for charity -/
theorem total_raised :
  charity_fundraiser 25 2 15 2 20 = 100 :=
by sorry

end total_raised_l39_3950


namespace shirt_boxes_per_roll_l39_3985

-- Define the variables
def xl_boxes_per_roll : ℕ := 3
def shirt_boxes_to_wrap : ℕ := 20
def xl_boxes_to_wrap : ℕ := 12
def cost_per_roll : ℚ := 4
def total_cost : ℚ := 32

-- Define the theorem
theorem shirt_boxes_per_roll :
  ∃ (s : ℕ), 
    s * ((total_cost / cost_per_roll) - (xl_boxes_to_wrap / xl_boxes_per_roll)) = shirt_boxes_to_wrap ∧ 
    s = 5 := by
  sorry

end shirt_boxes_per_roll_l39_3985


namespace mayoral_election_vote_ratio_l39_3992

theorem mayoral_election_vote_ratio :
  let votes_Z : ℕ := 25000
  let votes_X : ℕ := 22500
  let votes_Y : ℕ := 2 * votes_X / 3
  let fewer_votes : ℕ := votes_Z - votes_Y
  (fewer_votes : ℚ) / votes_Z = 2 / 5 := by
  sorry

end mayoral_election_vote_ratio_l39_3992


namespace integer_solution_exists_l39_3963

theorem integer_solution_exists (n : ℤ) : ∃ (a b c : ℤ), 
  a ≠ 0 ∧ 
  (a = b + c ∨ b = a + c ∨ c = a + b) ∧
  a * n + b = c :=
by sorry

end integer_solution_exists_l39_3963


namespace rational_product_sum_implies_negative_l39_3988

theorem rational_product_sum_implies_negative (a b : ℚ) 
  (h1 : a * b > 0) (h2 : a + b < 0) : a < 0 := by
  sorry

end rational_product_sum_implies_negative_l39_3988


namespace cubic_sum_theorem_l39_3967

theorem cubic_sum_theorem (x y : ℝ) 
  (h1 : y + 3 = (x - 3)^2)
  (h2 : x + 3 = (y - 3)^2)
  (h3 : x ≠ y) : 
  x^3 + y^3 = 217 := by
sorry

end cubic_sum_theorem_l39_3967


namespace parallel_planes_from_perpendicular_lines_l39_3941

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)

-- State the theorem
theorem parallel_planes_from_perpendicular_lines 
  (m n : Line) (α β : Plane) :
  parallel m n → 
  perpendicular_line_plane m α → 
  perpendicular_line_plane n β → 
  parallel_planes α β :=
sorry

end parallel_planes_from_perpendicular_lines_l39_3941


namespace largest_base6_4digit_in_base10_l39_3936

def largest_base6_4digit : ℕ := 5 * 6^3 + 5 * 6^2 + 5 * 6^1 + 5 * 6^0

theorem largest_base6_4digit_in_base10 : 
  largest_base6_4digit = 1295 := by sorry

end largest_base6_4digit_in_base10_l39_3936


namespace bus_rows_count_l39_3958

theorem bus_rows_count (total_capacity : ℕ) (row_capacity : ℕ) (h1 : total_capacity = 80) (h2 : row_capacity = 4) :
  total_capacity / row_capacity = 20 :=
by sorry

end bus_rows_count_l39_3958


namespace arithmetic_mean_problem_l39_3925

/-- Given two real numbers p and q with arithmetic mean 10, and a third real number r
    such that r - p = 30, prove that the arithmetic mean of q and r is 25. -/
theorem arithmetic_mean_problem (p q r : ℝ) 
  (h1 : (p + q) / 2 = 10) 
  (h2 : r - p = 30) : 
  (q + r) / 2 = 25 := by
  sorry

end arithmetic_mean_problem_l39_3925


namespace new_line_properties_new_line_equation_correct_l39_3934

/-- Given two lines in the plane -/
def line1 (x y : ℝ) : Prop := 3 * x + 2 * y - 5 = 0
def line2 (x y : ℝ) : Prop := 2 * x + 3 * y - 5 = 0

/-- The intersection point of the two lines -/
def intersection : ℝ × ℝ := (1, 1)

/-- The new line passing through the intersection point and having y-intercept -5 -/
def new_line (x y : ℝ) : Prop := 6 * x - y - 5 = 0

/-- Theorem stating that the new line passes through the intersection point and has y-intercept -5 -/
theorem new_line_properties :
  (line1 intersection.1 intersection.2) ∧
  (line2 intersection.1 intersection.2) ∧
  (new_line intersection.1 intersection.2) ∧
  (new_line 0 (-5)) :=
sorry

/-- Main theorem proving that the new line equation is correct -/
theorem new_line_equation_correct (x y : ℝ) :
  (line1 x y ∧ line2 x y) →
  (∃ t : ℝ, new_line (x + t * (intersection.1 - x)) (y + t * (intersection.2 - y))) :=
sorry

end new_line_properties_new_line_equation_correct_l39_3934


namespace sandys_change_l39_3947

/-- Calculates Sandy's change after shopping for toys -/
theorem sandys_change 
  (football_price : ℝ)
  (baseball_price : ℝ)
  (basketball_price : ℝ)
  (football_count : ℕ)
  (baseball_count : ℕ)
  (basketball_count : ℕ)
  (pounds_paid : ℝ)
  (euros_paid : ℝ)
  (h1 : football_price = 9.14)
  (h2 : baseball_price = 6.81)
  (h3 : basketball_price = 7.95)
  (h4 : football_count = 3)
  (h5 : baseball_count = 2)
  (h6 : basketball_count = 4)
  (h7 : pounds_paid = 50)
  (h8 : euros_paid = 20) :
  let pounds_spent := football_price * football_count + baseball_price * baseball_count
  let euros_spent := basketball_price * basketball_count
  let pounds_change := pounds_paid - pounds_spent
  let euros_change := max (euros_paid - euros_spent) 0
  (pounds_change = 8.96 ∧ euros_change = 0) :=
by sorry

end sandys_change_l39_3947


namespace triangle_sides_l39_3986

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (3 * Real.pi + x) * Real.cos (Real.pi - x) + (Real.cos (Real.pi / 2 + x))^2

theorem triangle_sides (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < Real.pi →
  f A = 3/2 →
  a = 2 →
  b + c = 4 →
  b = 2 ∧ c = 2 := by sorry

end triangle_sides_l39_3986


namespace sufficient_not_necessary_l39_3978

theorem sufficient_not_necessary (x y : ℝ) :
  (x ≥ 2 ∧ y ≥ 2 → x^2 + y^2 ≥ 4) ∧
  ∃ x y, x^2 + y^2 ≥ 4 ∧ ¬(x ≥ 2 ∧ y ≥ 2) := by
  sorry

end sufficient_not_necessary_l39_3978


namespace jack_heavier_than_sam_l39_3952

theorem jack_heavier_than_sam (total_weight jack_weight : ℕ) 
  (h1 : total_weight = 96)
  (h2 : jack_weight = 52) :
  jack_weight - (total_weight - jack_weight) = 8 :=
by sorry

end jack_heavier_than_sam_l39_3952


namespace cubic_root_sum_l39_3919

theorem cubic_root_sum (p q r : ℝ) : 
  p^3 - 6*p^2 + 11*p - 6 = 0 →
  q^3 - 6*q^2 + 11*q - 6 = 0 →
  r^3 - 6*r^2 + 11*r - 6 = 0 →
  p * q / r + p * r / q + q * r / p = 49 / 6 := by
  sorry

end cubic_root_sum_l39_3919


namespace number_puzzle_l39_3918

theorem number_puzzle : 
  ∃ x : ℝ, (x / 5 + 4 = x / 4 - 4) ∧ (x = 160) := by
  sorry

end number_puzzle_l39_3918


namespace quadratic_symmetry_l39_3971

/-- A quadratic function with the given properties -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_symmetry (a b c : ℝ) :
  f a b c (-2) = -13/2 ∧
  f a b c (-1) = -4 ∧
  f a b c 0 = -5/2 ∧
  f a b c 1 = -2 ∧
  f a b c 2 = -5/2 →
  f a b c 3 = -4 := by
  sorry

end quadratic_symmetry_l39_3971


namespace cos_150_degrees_l39_3944

theorem cos_150_degrees : Real.cos (150 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end cos_150_degrees_l39_3944


namespace existence_of_special_integers_l39_3960

theorem existence_of_special_integers : ∃ (a b : ℕ+), 
  (¬ (7 ∣ (a.val * b.val * (a.val + b.val)))) ∧ 
  ((7^7 : ℕ) ∣ ((a.val + b.val)^7 - a.val^7 - b.val^7)) := by
  sorry

end existence_of_special_integers_l39_3960


namespace apples_at_first_store_l39_3959

def first_store_price : ℝ := 3
def second_store_price : ℝ := 4
def second_store_apples : ℝ := 10
def savings_per_apple : ℝ := 0.1

theorem apples_at_first_store :
  let second_store_price_per_apple := second_store_price / second_store_apples
  let first_store_price_per_apple := second_store_price_per_apple + savings_per_apple
  first_store_price / first_store_price_per_apple = 6 := by sorry

end apples_at_first_store_l39_3959


namespace inverse_mod_53_l39_3990

theorem inverse_mod_53 (h : (17⁻¹ : ZMod 53) = 11) : (36⁻¹ : ZMod 53) = 42 := by
  sorry

end inverse_mod_53_l39_3990


namespace min_value_sum_l39_3981

theorem min_value_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : (a + 3)⁻¹ + (b + 3)⁻¹ = (1 : ℝ) / 4) :
  ∀ x y : ℝ, x > 0 → y > 0 → (x + 3)⁻¹ + (y + 3)⁻¹ = (1 : ℝ) / 4 → 
  a + 3 * b ≤ x + 3 * y ∧ a + 3 * b = 4 + 8 * Real.sqrt 3 :=
sorry

end min_value_sum_l39_3981


namespace parabola_ellipse_tangency_l39_3989

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := x^2 = 2*p*y ∧ p > 0

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/6 + y^2/4 = 1

-- Define the latus rectum of the parabola
def latus_rectum (p : ℝ) (y : ℝ) : Prop := y = -p/2

-- Theorem statement
theorem parabola_ellipse_tangency :
  ∃ (p : ℝ), ∃ (x y : ℝ),
    parabola p x y ∧
    ellipse x y ∧
    latus_rectum p y ∧
    p = 4 :=
sorry

end parabola_ellipse_tangency_l39_3989


namespace jakes_desired_rate_l39_3939

/-- Jake's hourly rate for planting flowers -/
def jakes_hourly_rate (total_charge : ℚ) (hours_worked : ℚ) : ℚ :=
  total_charge / hours_worked

/-- Theorem: Jake's hourly rate for planting flowers is $22.50 -/
theorem jakes_desired_rate :
  jakes_hourly_rate 45 2 = 22.5 := by
  sorry

end jakes_desired_rate_l39_3939


namespace simplify_expression_l39_3912

theorem simplify_expression (b : ℝ) : (1:ℝ)*(2*b)*(3*b^2)*(4*b^3)*(5*b^4)*(6*b^5) = 720 * b^15 := by
  sorry

end simplify_expression_l39_3912


namespace octal_addition_sum_l39_3911

/-- Given an octal addition 3XY₈ + 52₈ = 4X3₈, prove that X + Y = 1 in base 10 -/
theorem octal_addition_sum (X Y : ℕ) : 
  (3 * 8^2 + X * 8 + Y) + (5 * 8 + 2) = 4 * 8^2 + X * 8 + 3 → X + Y = 1 := by
  sorry

end octal_addition_sum_l39_3911


namespace naomi_saw_58_wheels_l39_3954

/-- The number of regular bikes at the park -/
def regular_bikes : ℕ := 7

/-- The number of children's bikes at the park -/
def children_bikes : ℕ := 11

/-- The number of wheels on a regular bike -/
def regular_bike_wheels : ℕ := 2

/-- The number of wheels on a children's bike -/
def children_bike_wheels : ℕ := 4

/-- The total number of wheels Naomi saw at the park -/
def total_wheels : ℕ := regular_bikes * regular_bike_wheels + children_bikes * children_bike_wheels

theorem naomi_saw_58_wheels : total_wheels = 58 := by
  sorry

end naomi_saw_58_wheels_l39_3954


namespace number_puzzle_l39_3970

theorem number_puzzle : ∃ x : ℝ, x = 280 ∧ x / 5 + 7 = x / 4 - 7 := by
  sorry

end number_puzzle_l39_3970


namespace triangle_problem_l39_3962

theorem triangle_problem (A B C : Real) (a b c : Real) (D : Real × Real) :
  (2 * a^2 * Real.sin B * Real.sin C = Real.sqrt 3 * (a^2 + b^2 - c^2) * Real.sin A) →
  (a = 1) →
  (b = 2) →
  (D = ((A + B) / 2, 0)) →  -- Assuming A and B are coordinates on the x-axis
  (C = Real.pi / 3) ∧
  (Real.sqrt ((C - D.1)^2 + D.2^2) = Real.sqrt 7 / 2) := by
  sorry

end triangle_problem_l39_3962


namespace roots_geometric_sequence_range_l39_3942

theorem roots_geometric_sequence_range (a b : ℝ) (q : ℝ) : 
  (∃ x₁ x₂ x₃ x₄ : ℝ, 
    (∀ x : ℝ, (x^2 - a*x + 1)*(x^2 - b*x + 1) = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄) ∧ 
    (∃ r : ℝ, x₂ = x₁ * r ∧ x₃ = x₂ * r ∧ x₄ = x₃ * r) ∧
    q = r ∧ 
    1/3 ≤ q ∧ q ≤ 2) →
  4 ≤ a*b ∧ a*b ≤ 112/9 := by
sorry

end roots_geometric_sequence_range_l39_3942


namespace min_time_to_target_l39_3906

/-- Represents the number of steps to the right per minute -/
def right_steps : ℕ := 47

/-- Represents the number of steps to the left per minute -/
def left_steps : ℕ := 37

/-- Represents the target position (one step to the right) -/
def target : ℤ := 1

/-- Theorem stating the minimum time to reach the target position -/
theorem min_time_to_target :
  ∃ (x y : ℕ), 
    right_steps * x - left_steps * y = target ∧
    (∀ (a b : ℕ), right_steps * a - left_steps * b = target → x + y ≤ a + b) ∧
    x + y = 59 := by
  sorry

end min_time_to_target_l39_3906


namespace diver_B_depth_l39_3908

/-- The depth of diver A in meters -/
def depth_A : ℝ := -55

/-- The vertical distance between diver B and diver A in meters -/
def distance_B_above_A : ℝ := 5

/-- The depth of diver B in meters -/
def depth_B : ℝ := depth_A + distance_B_above_A

theorem diver_B_depth : depth_B = -50 := by
  sorry

end diver_B_depth_l39_3908


namespace skee_ball_tickets_value_l39_3948

/-- The number of tickets Luke won playing 'whack a mole' -/
def whack_a_mole_tickets : ℕ := 2

/-- The cost of one candy in tickets -/
def candy_cost : ℕ := 3

/-- The number of candies Luke could buy -/
def candies_bought : ℕ := 5

/-- The number of tickets Luke won playing 'skee ball' -/
def skee_ball_tickets : ℕ := candies_bought * candy_cost - whack_a_mole_tickets

theorem skee_ball_tickets_value : skee_ball_tickets = 13 := by
  sorry

end skee_ball_tickets_value_l39_3948


namespace g_triple_equality_l39_3984

noncomputable def g (x : ℝ) : ℝ :=
  if x ≤ 0 then -x else 3 * x - 50

theorem g_triple_equality (a : ℝ) :
  a < 0 → (g (g (g 15)) = g (g (g a)) ↔ a = -55 / 3) := by
  sorry

end g_triple_equality_l39_3984


namespace supermarket_spending_l39_3909

theorem supermarket_spending (total : ℚ) 
  (h1 : total = 120) 
  (h2 : ∃ (fruits meat bakery candy : ℚ), 
    fruits + meat + bakery + candy = total ∧
    fruits = (1/2) * total ∧
    meat = (1/3) * total ∧
    bakery = (1/10) * total) : 
  ∃ (candy : ℚ), candy = 8 := by
sorry

end supermarket_spending_l39_3909


namespace scarves_per_yarn_l39_3955

/-- Given the total number of scarves and yarns, calculate the number of scarves per yarn -/
theorem scarves_per_yarn (total_scarves total_yarns : ℕ) 
  (h1 : total_scarves = 36)
  (h2 : total_yarns = 12) :
  total_scarves / total_yarns = 3 := by
  sorry

#eval 36 / 12  -- This should output 3

end scarves_per_yarn_l39_3955


namespace perpendicular_lines_l39_3922

theorem perpendicular_lines (x y : ℝ) : 
  let angle1 : ℝ := 50 + x - y
  let angle2 : ℝ := angle1 - (10 + 2*x - 2*y)
  angle1 + angle2 = 90 := by
sorry

end perpendicular_lines_l39_3922


namespace inequality_range_l39_3982

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 - 2*x + 5 ≥ a^2 - 3*a) ↔ a ∈ Set.Icc (-1) 4 :=
sorry

end inequality_range_l39_3982


namespace greenleaf_academy_history_class_l39_3937

/-- The number of students in the history class at Greenleaf Academy -/
def history_class_size : ℕ := by sorry

theorem greenleaf_academy_history_class :
  let total_students : ℕ := 70
  let both_subjects : ℕ := 10
  let geography_only : ℕ := 16  -- Derived from the solution, but mathematically necessary
  let history_only : ℕ := total_students - both_subjects - geography_only
  let history_class_size : ℕ := history_only + both_subjects
  let geography_class_size : ℕ := geography_only + both_subjects
  (total_students = geography_only + history_only + both_subjects) ∧
  (history_class_size = 2 * geography_class_size) →
  history_class_size = 52 :=
by sorry

end greenleaf_academy_history_class_l39_3937


namespace sin_150_minus_alpha_l39_3910

theorem sin_150_minus_alpha (α : Real) (h : α = 240 * Real.pi / 180) :
  Real.sin (150 * Real.pi / 180 - α) = -1 := by sorry

end sin_150_minus_alpha_l39_3910


namespace eldoria_license_plates_l39_3953

/-- The number of possible uppercase letters in a license plate. -/
def num_letters : ℕ := 26

/-- The number of possible digits in a license plate. -/
def num_digits : ℕ := 10

/-- The number of letter positions in a license plate. -/
def letter_positions : ℕ := 3

/-- The number of digit positions in a license plate. -/
def digit_positions : ℕ := 4

/-- The total number of possible license plates in Eldoria. -/
def total_license_plates : ℕ := num_letters ^ letter_positions * num_digits ^ digit_positions

theorem eldoria_license_plates :
  total_license_plates = 175760000 := by
  sorry

end eldoria_license_plates_l39_3953


namespace triangle_intersection_ratio_l39_3916

/-- Given a triangle XYZ, this theorem proves that if point P is on XY with XP:PY = 4:1,
    point Q is on YZ with YQ:QZ = 4:1, and lines PQ and XZ intersect at R,
    then PQ:QR = 4:1. -/
theorem triangle_intersection_ratio (X Y Z P Q R : ℝ × ℝ) : 
  (∃ t : ℝ, P = (1 - t) • X + t • Y ∧ t = 1/5) →  -- P is on XY with XP:PY = 4:1
  (∃ s : ℝ, Q = (1 - s) • Y + s • Z ∧ s = 4/5) →  -- Q is on YZ with YQ:QZ = 4:1
  (∃ u v : ℝ, R = (1 - u) • X + u • Z ∧ R = (1 - v) • P + v • Q) →  -- R is intersection of XZ and PQ
  ∃ k : ℝ, k • (Q - P) = R - Q ∧ k = 1/4 :=  -- PQ:QR = 4:1
by sorry

end triangle_intersection_ratio_l39_3916


namespace square_sum_zero_implies_all_zero_l39_3920

theorem square_sum_zero_implies_all_zero (a b c : ℝ) : 
  a^2 + b^2 + c^2 = 0 → a = 0 ∧ b = 0 ∧ c = 0 := by
  sorry

end square_sum_zero_implies_all_zero_l39_3920


namespace floor_painting_overlap_l39_3974

theorem floor_painting_overlap (red green blue : ℝ) 
  (h_red : red = 0.75) 
  (h_green : green = 0.7) 
  (h_blue : blue = 0.65) : 
  1 - (1 - red + 1 - green + 1 - blue) ≥ 0.1 := by sorry

end floor_painting_overlap_l39_3974


namespace purely_imaginary_complex_l39_3905

theorem purely_imaginary_complex (m : ℝ) : 
  let z : ℂ := (m + Complex.I) / (1 + Complex.I)
  (∃ (y : ℝ), z = Complex.I * y) ↔ m = -1 := by
  sorry

end purely_imaginary_complex_l39_3905


namespace continuity_at_two_l39_3993

/-- The function f(x) = -2x^2 - 5 is continuous at x₀ = 2 -/
theorem continuity_at_two :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x - 2| < δ → |(-2 * x^2 - 5) - (-2 * 2^2 - 5)| < ε :=
by sorry

end continuity_at_two_l39_3993


namespace nabla_problem_l39_3961

-- Define the ∇ operation
def nabla (a b : ℕ) : ℕ := 3 + b^a

-- Theorem statement
theorem nabla_problem : nabla (nabla 2 3) 4 = 16777219 := by
  sorry

end nabla_problem_l39_3961


namespace arithmetic_sequence_middle_term_l39_3913

theorem arithmetic_sequence_middle_term 
  (a : Fin 5 → ℝ)  -- a is a function from Fin 5 to ℝ, representing the 5 terms
  (h1 : a 0 = -8)  -- first term is -8
  (h2 : a 4 = 10)  -- last term is 10
  (h3 : ∀ i : Fin 4, a (i + 1) - a i = a 1 - a 0)  -- arithmetic sequence condition
  : a 2 = 1 := by
sorry

end arithmetic_sequence_middle_term_l39_3913


namespace stratified_sampling_sophomores_l39_3931

theorem stratified_sampling_sophomores 
  (total_students : ℕ) 
  (sophomores : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_students = 1000)
  (h2 : sophomores = 320)
  (h3 : sample_size = 50) : 
  (sophomores * sample_size) / total_students = 16 := by
  sorry

end stratified_sampling_sophomores_l39_3931


namespace max_elements_sum_to_target_l39_3998

/-- The sequence of consecutive odd numbers from 1 to 101 -/
def oddSequence : List Nat := List.range 51 |>.map (fun n => 2 * n + 1)

/-- The sum of selected numbers should be 2013 -/
def targetSum : Nat := 2013

/-- The maximum number of elements that can be selected -/
def maxElements : Nat := 43

theorem max_elements_sum_to_target :
  ∃ (selected : List Nat),
    selected.length = maxElements ∧
    selected.all (· ∈ oddSequence) ∧
    selected.sum = targetSum ∧
    ∀ (other : List Nat),
      other.all (· ∈ oddSequence) →
      other.sum = targetSum →
      other.length ≤ maxElements :=
sorry

end max_elements_sum_to_target_l39_3998


namespace complex_square_on_negative_y_axis_l39_3979

/-- A complex number z is on the negative y-axis if its real part is 0 and its imaginary part is negative -/
def on_negative_y_axis (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im < 0

/-- The main theorem -/
theorem complex_square_on_negative_y_axis (a : ℝ) :
  on_negative_y_axis ((a + Complex.I) ^ 2) → a = -1 := by
  sorry

end complex_square_on_negative_y_axis_l39_3979


namespace special_ellipse_d_value_l39_3940

/-- An ellipse in the first quadrant tangent to both axes with foci at (5,10) and (d,10) --/
structure Ellipse where
  d : ℝ
  tangent_x : Bool
  tangent_y : Bool
  first_quadrant : Bool
  focus1 : ℝ × ℝ
  focus2 : ℝ × ℝ

/-- The d value for the special ellipse described in the problem --/
def special_ellipse_d : ℝ := 20

/-- Theorem stating that the d value for the special ellipse is 20 --/
theorem special_ellipse_d_value (e : Ellipse) 
  (h_tangent_x : e.tangent_x = true)
  (h_tangent_y : e.tangent_y = true)
  (h_first_quadrant : e.first_quadrant = true)
  (h_focus1 : e.focus1 = (5, 10))
  (h_focus2 : e.focus2 = (e.d, 10)) :
  e.d = special_ellipse_d := by
  sorry

end special_ellipse_d_value_l39_3940


namespace union_of_A_and_B_l39_3915

def A : Set ℕ := {1, 2}
def B : Set ℕ := {x | 2^x = 8}

theorem union_of_A_and_B : A ∪ B = {1, 2, 3} := by sorry

end union_of_A_and_B_l39_3915


namespace exists_non_integer_root_l39_3976

theorem exists_non_integer_root (a b c : ℤ) : ∃ n : ℕ+, ¬ ∃ m : ℤ, (m : ℝ)^2 = (n : ℝ)^3 + (a : ℝ) * (n : ℝ)^2 + (b : ℝ) * (n : ℝ) + (c : ℝ) := by
  sorry

end exists_non_integer_root_l39_3976


namespace line_arrangement_result_l39_3929

/-- The number of ways to arrange 3 boys and 3 girls in a line with two girls together -/
def line_arrangement (num_boys num_girls : ℕ) : ℕ :=
  -- Define the function here
  sorry

/-- Theorem stating that the number of arrangements is 432 -/
theorem line_arrangement_result : line_arrangement 3 3 = 432 := by
  sorry

end line_arrangement_result_l39_3929


namespace bigger_part_is_38_l39_3907

theorem bigger_part_is_38 (x y : ℕ) (h1 : x + y = 56) (h2 : 10 * x + 22 * y = 780) :
  max x y = 38 := by
sorry

end bigger_part_is_38_l39_3907


namespace green_balls_count_l39_3956

theorem green_balls_count (total : ℕ) (white yellow red purple : ℕ) (prob : ℚ) :
  total = 60 ∧
  white = 22 ∧
  yellow = 17 ∧
  red = 3 ∧
  purple = 1 ∧
  prob = 95 / 100 ∧
  (total - red - purple : ℚ) / total = prob →
  total - white - yellow - red - purple = 17 := by
  sorry

end green_balls_count_l39_3956


namespace max_value_trig_expression_l39_3980

theorem max_value_trig_expression (a b φ : ℝ) :
  (∀ θ : ℝ, a * Real.cos θ + b * Real.sin (θ + φ) ≤ Real.sqrt (a^2 + 2*a*b * Real.sin φ + b^2)) ∧
  (∃ θ : ℝ, a * Real.cos θ + b * Real.sin (θ + φ) = Real.sqrt (a^2 + 2*a*b * Real.sin φ + b^2)) :=
by sorry

end max_value_trig_expression_l39_3980


namespace inequality_proof_l39_3975

theorem inequality_proof (x y : ℝ) (hx : x ≠ -1) (hy : y ≠ -1) (hxy : x * y = 1) :
  ((2 + x) / (1 + x))^2 + ((2 + y) / (1 + y))^2 ≥ 9/2 := by
  sorry

end inequality_proof_l39_3975


namespace alex_silk_distribution_l39_3977

/-- The amount of silk each friend receives when Alex distributes his remaining silk -/
def silk_per_friend (total_silk : ℕ) (silk_per_dress : ℕ) (num_dresses : ℕ) (num_friends : ℕ) : ℕ :=
  (total_silk - silk_per_dress * num_dresses) / num_friends

/-- Theorem stating that each friend receives 20 meters of silk -/
theorem alex_silk_distribution :
  silk_per_friend 600 5 100 5 = 20 := by
  sorry

end alex_silk_distribution_l39_3977


namespace rachel_initial_lives_l39_3902

/-- Rachel's initial number of lives -/
def initial_lives : ℕ := 10

/-- The number of lives Rachel lost -/
def lives_lost : ℕ := 4

/-- The number of lives Rachel gained -/
def lives_gained : ℕ := 26

/-- The final number of lives Rachel had -/
def final_lives : ℕ := 32

/-- Theorem stating that Rachel's initial number of lives was 10 -/
theorem rachel_initial_lives :
  initial_lives = 10 ∧
  final_lives = initial_lives - lives_lost + lives_gained :=
sorry

end rachel_initial_lives_l39_3902


namespace vector_inequalities_l39_3965

theorem vector_inequalities (a b c m n p : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 1) 
  (h2 : m^2 + n^2 + p^2 = 1) : 
  (|a*m + b*n + c*p| ≤ 1) ∧ 
  (a*b*c ≠ 0 → m^4/a^2 + n^4/b^2 + p^4/c^2 ≥ 1) := by
  sorry

end vector_inequalities_l39_3965


namespace function_properties_l39_3927

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.log x - m * x + m

theorem function_properties (m : ℝ) :
  (∀ x > 0, f m x ≤ 0) →
  m = 1 ∧ ∀ a b, 0 < a → a < b → (f m b - f m a) / (b - a) < 1 / (a * (a + 1)) :=
by sorry

end function_properties_l39_3927


namespace abs_half_minus_three_eighths_i_l39_3994

theorem abs_half_minus_three_eighths_i : Complex.abs (1/2 - 3/8 * Complex.I) = 5/8 := by
  sorry

end abs_half_minus_three_eighths_i_l39_3994


namespace merchant_markup_percentage_l39_3900

/-- Proves that if a merchant marks up goods by x%, then offers a 10% discount,
    and makes a 57.5% profit, the value of x is 75%. -/
theorem merchant_markup_percentage
  (cost_price : ℝ)
  (markup_percentage : ℝ)
  (discount_percentage : ℝ)
  (profit_percentage : ℝ)
  (h1 : discount_percentage = 10)
  (h2 : profit_percentage = 57.5)
  (h3 : cost_price > 0)
  : markup_percentage = 75 :=
by sorry

end merchant_markup_percentage_l39_3900


namespace boat_speed_in_still_water_l39_3930

/-- The speed of a boat in still water, given its speeds with and against a stream -/
theorem boat_speed_in_still_water (along_stream : ℝ) (against_stream : ℝ) 
    (h1 : along_stream = 21) 
    (h2 : against_stream = 9) : 
    (along_stream + against_stream) / 2 = 15 := by
  sorry

end boat_speed_in_still_water_l39_3930


namespace range_of_a_l39_3973

theorem range_of_a (x a : ℝ) : 
  (∀ x, (1 / (x - 2) ≥ 1 → |x - a| < 1) ∧ 
   ∃ x, (|x - a| < 1 ∧ 1 / (x - 2) < 1)) →
  a ∈ Set.Ioo 2 3 := by sorry

end range_of_a_l39_3973


namespace goose_price_after_increases_l39_3926

-- Define the initial prices
def initial_goose_price : ℝ := 0.8
def initial_wine_price : ℝ := 0.4

-- Define the price increase factor
def price_increase_factor : ℝ := 1.2

theorem goose_price_after_increases (goose_price : ℝ) (wine_price : ℝ) :
  goose_price = initial_goose_price ∧ 
  wine_price = initial_wine_price ∧
  goose_price + wine_price = 1 ∧
  goose_price + 0.5 * wine_price = 1 →
  goose_price * price_increase_factor * price_increase_factor < 1 := by
  sorry

#check goose_price_after_increases

end goose_price_after_increases_l39_3926


namespace union_of_M_and_N_l39_3949

def M : Set ℝ := {x | x^2 < x}
def N : Set ℝ := {x | x^2 + 2*x - 3 < 0}

theorem union_of_M_and_N : M ∪ N = Set.Ioo (-3) 1 := by
  sorry

end union_of_M_and_N_l39_3949


namespace infiniteSum_equals_power_l39_3972

/-- Number of paths from (0,0) to (k,n) satisfying the given conditions -/
def C (k n : ℕ) : ℕ := sorry

/-- The sum of C_{100j+19,17} for j from 0 to infinity -/
def infiniteSum : ℕ := sorry

/-- Theorem stating that the infinite sum equals 100^17 -/
theorem infiniteSum_equals_power : infiniteSum = 100^17 := by sorry

end infiniteSum_equals_power_l39_3972


namespace arithmetic_square_root_is_function_l39_3914

theorem arithmetic_square_root_is_function : 
  ∀ (x : ℝ), x > 0 → ∃! (y : ℝ), y > 0 ∧ y^2 = x :=
by sorry

end arithmetic_square_root_is_function_l39_3914


namespace outfits_count_l39_3943

/-- The number of outfits that can be made with given numbers of shirts, pants, and hats -/
def number_of_outfits (shirts : ℕ) (pants : ℕ) (hats : ℕ) : ℕ :=
  shirts * pants * hats

/-- Theorem stating that the number of outfits with 4 shirts, 5 pants, and 3 hats is 60 -/
theorem outfits_count :
  number_of_outfits 4 5 3 = 60 := by
  sorry

end outfits_count_l39_3943
