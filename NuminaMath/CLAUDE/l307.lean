import Mathlib

namespace next_larger_perfect_square_l307_30734

theorem next_larger_perfect_square (x : ℕ) (h : ∃ k : ℕ, x = k^2) :
  ∃ n : ℕ, n > x ∧ (∃ m : ℕ, n = m^2) ∧ n = x + 4 * (x.sqrt) + 4 := by
  sorry

end next_larger_perfect_square_l307_30734


namespace haley_small_gardens_l307_30794

def total_seeds : ℕ := 56
def big_garden_seeds : ℕ := 35
def seeds_per_small_garden : ℕ := 3

def small_gardens : ℕ := (total_seeds - big_garden_seeds) / seeds_per_small_garden

theorem haley_small_gardens : small_gardens = 7 := by
  sorry

end haley_small_gardens_l307_30794


namespace three_in_A_not_in_B_l307_30736

-- Define the universal set U
def U : Finset Nat := {1, 2, 3, 4, 5}

-- Define the complement of A in U
def complement_A : Finset Nat := {2, 4}

-- Define the complement of B in U
def complement_B : Finset Nat := {3, 4}

-- Define set A
def A : Finset Nat := U \ complement_A

-- Define set B
def B : Finset Nat := U \ complement_B

-- Theorem statement
theorem three_in_A_not_in_B : 3 ∈ A ∧ 3 ∉ B := by
  sorry

end three_in_A_not_in_B_l307_30736


namespace unique_solution_l307_30709

/-- Represents the guesses made by the three friends --/
def friends_guesses : List Nat := [16, 19, 25]

/-- Represents the errors in the guesses --/
def guess_errors : List Nat := [2, 4, 5]

/-- Checks if a number satisfies all constraints --/
def satisfies_constraints (x : Nat) : Prop :=
  ∃ (perm : List Nat), perm.Perm guess_errors ∧
    (friends_guesses.zip perm).all (fun (guess, error) => 
      (guess + error = x) ∨ (guess - error = x))

/-- The theorem stating that 21 is the only number satisfying all constraints --/
theorem unique_solution : 
  satisfies_constraints 21 ∧ ∀ x : Nat, satisfies_constraints x → x = 21 := by
  sorry


end unique_solution_l307_30709


namespace tan_value_from_sin_plus_cos_l307_30798

theorem tan_value_from_sin_plus_cos (α : Real) 
  (h1 : α ∈ Set.Ioo 0 Real.pi) 
  (h2 : Real.sin α + Real.cos α = 1/5) : 
  Real.tan α = -4/3 := by
  sorry

end tan_value_from_sin_plus_cos_l307_30798


namespace a_9_value_l307_30730

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem a_9_value (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 1 + a 11) / 2 = 15 →
  a 1 + a 2 + a 3 = 9 →
  a 9 = 24 := by
sorry

end a_9_value_l307_30730


namespace min_value_sum_reciprocals_l307_30710

theorem min_value_sum_reciprocals (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : x^2 + y^2 + z^2 = 1) : 
  (1 / (x^2 + y^2) + 1 / (x^2 + z^2) + 1 / (y^2 + z^2)) ≥ 9/2 := by
sorry

end min_value_sum_reciprocals_l307_30710


namespace rachels_to_christines_ratio_l307_30703

def strawberries_per_pie : ℕ := 3
def christines_strawberries : ℕ := 10
def total_pies : ℕ := 10

theorem rachels_to_christines_ratio :
  let total_strawberries := strawberries_per_pie * total_pies
  let rachels_strawberries := total_strawberries - christines_strawberries
  (rachels_strawberries : ℚ) / christines_strawberries = 2 := by sorry

end rachels_to_christines_ratio_l307_30703


namespace max_k_for_tangent_line_l307_30755

/-- The maximum value of k for which the line y = kx - 2 has at least one point 
    where a line tangent to the circle x^2 + y^2 = 1 can be drawn -/
theorem max_k_for_tangent_line : 
  ∃ (k : ℝ), ∀ (k' : ℝ), 
    (∃ (x y : ℝ), y = k' * x - 2 ∧ 
      ∃ (m : ℝ), (y - m * x)^2 = (1 + m^2) * (1 - x^2)) → 
    k' ≤ k ∧ 
    k = Real.sqrt 3 := by sorry

end max_k_for_tangent_line_l307_30755


namespace beetle_speed_l307_30732

/-- Given an ant's average speed and a beetle that walks 10% less distance in the same time,
    prove that the beetle's speed is 1.8 km/h. -/
theorem beetle_speed (ant_distance : ℝ) (time : ℝ) (beetle_percentage : ℝ) :
  ant_distance = 1000 →
  time = 30 →
  beetle_percentage = 0.9 →
  let beetle_distance := ant_distance * beetle_percentage
  let beetle_speed_mpm := beetle_distance / time
  let beetle_speed_kmh := beetle_speed_mpm * 2 * 0.001
  beetle_speed_kmh = 1.8 := by
sorry

end beetle_speed_l307_30732


namespace annual_growth_rate_l307_30733

theorem annual_growth_rate (initial : ℝ) (final : ℝ) (years : ℕ) (x : ℝ) 
  (h1 : initial = 1000000)
  (h2 : final = 1690000)
  (h3 : years = 2)
  (h4 : x > 0)
  (h5 : (1 + x)^years = final / initial) :
  x = 0.3 := by
sorry

end annual_growth_rate_l307_30733


namespace consecutive_even_numbers_sum_l307_30760

theorem consecutive_even_numbers_sum (a b c d : ℤ) : 
  (∀ n : ℤ, a = 2*n ∧ b = 2*n + 2 ∧ c = 2*n + 4 ∧ d = 2*n + 6) →  -- Consecutive even numbers
  (a + b + c + d = 140) →                                        -- Sum condition
  (d = 38) :=                                                    -- Conclusion (largest number)
by
  sorry

end consecutive_even_numbers_sum_l307_30760


namespace parabola_point_slope_l307_30791

/-- A point on a parabola with given properties -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  on_parabola : y^2 = 4*x
  distance_to_focus : Real.sqrt ((x - 1)^2 + y^2) = 5

/-- The theorem stating the absolute value of the slope -/
theorem parabola_point_slope (P : ParabolaPoint) : 
  |((P.y - 0) / (P.x - 1))| = 4/3 := by
  sorry

end parabola_point_slope_l307_30791


namespace car_distance_l307_30767

theorem car_distance (x : ℝ) (h : 12 * x = 10 * (x + 2)) : 12 * x = 120 := by
  sorry

end car_distance_l307_30767


namespace ellipse_equation_circle_diameter_property_l307_30750

-- Define the ellipse C
def ellipse (a b x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the conditions
structure EllipseConditions (a b c : ℝ) :=
  (a_gt_b : a > b)
  (b_gt_zero : b > 0)
  (perimeter : 2*c + 2*a = 6)
  (focal_distance : 2*c*b = a*b)
  (pythagoras : a^2 = b^2 + c^2)

-- Theorem for part 1
theorem ellipse_equation (a b c : ℝ) (h : EllipseConditions a b c) :
  a = 2 ∧ b = Real.sqrt 3 ∧ c = 1 :=
sorry

-- Theorem for part 2
theorem circle_diameter_property (m : ℝ) :
  let a := 2
  let b := Real.sqrt 3
  ∀ x₀ y₀ : ℝ, 
    ellipse a b x₀ y₀ → 
    x₀ ≠ 2 → 
    x₀ ≠ -2 → 
    (m - 2) * (x₀ - 2) + (y₀^2 / (x₀ + 2)) * (m + 2) = 0 →
    m = 14 :=
sorry

end ellipse_equation_circle_diameter_property_l307_30750


namespace coefficient_of_x_cubed_in_binomial_expansion_l307_30754

theorem coefficient_of_x_cubed_in_binomial_expansion (a₀ a₁ a₂ a₃ a₄ a₅ : ℚ) :
  (∀ x : ℚ, (x - 1)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₃ = 10 := by
  sorry

end coefficient_of_x_cubed_in_binomial_expansion_l307_30754


namespace curve_crosses_at_2_3_l307_30712

/-- A curve defined by x = t^2 - 4 and y = t^3 - 6t + 3 for all real t -/
def curve (t : ℝ) : ℝ × ℝ :=
  (t^2 - 4, t^3 - 6*t + 3)

/-- The point where the curve crosses itself -/
def crossing_point : ℝ × ℝ := (2, 3)

/-- Theorem stating that the curve crosses itself at (2, 3) -/
theorem curve_crosses_at_2_3 :
  ∃ (a b : ℝ), a ≠ b ∧ curve a = curve b ∧ curve a = crossing_point :=
sorry

end curve_crosses_at_2_3_l307_30712


namespace range_of_m_for_inequality_l307_30763

-- Define the function f
def f (x : ℝ) : ℝ := |x - 2|

-- State the theorem
theorem range_of_m_for_inequality (h : ∀ x ≤ 5, f x ≤ 3) :
  {m : ℝ | ∀ x, f x + (x + 5) ≥ m} = Set.Iic 5 := by
  sorry

end range_of_m_for_inequality_l307_30763


namespace exam_candidates_l307_30740

/-- Given an examination where the average marks obtained is 40 and the total marks are 2000,
    prove that the number of candidates who took the examination is 50. -/
theorem exam_candidates (average_marks : ℕ) (total_marks : ℕ) (h1 : average_marks = 40) (h2 : total_marks = 2000) :
  total_marks / average_marks = 50 := by
  sorry

end exam_candidates_l307_30740


namespace triangle_area_l307_30796

theorem triangle_area (A B C : ℝ) (a b c : ℝ) :
  A = 5 * π / 6 → b = 2 → c = 4 →
  (1 / 2) * b * c * Real.sin A = 2 := by
sorry

end triangle_area_l307_30796


namespace area_ratio_of_squares_l307_30789

/-- Given three square regions I, II, and III, where the perimeter of region I is 16 units
    and the perimeter of region II is 32 units, the ratio of the area of region II
    to the area of region III is 1/4. -/
theorem area_ratio_of_squares (side_length_I side_length_II side_length_III : ℝ)
    (h1 : side_length_I * 4 = 16)
    (h2 : side_length_II * 4 = 32)
    (h3 : side_length_III = 2 * side_length_II) :
    (side_length_II ^ 2) / (side_length_III ^ 2) = 1 / 4 := by
  sorry

end area_ratio_of_squares_l307_30789


namespace quadrilateral_with_perpendicular_bisecting_diagonals_not_necessarily_square_l307_30765

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : Point)

-- Define properties of a quadrilateral
def has_perpendicular_diagonals (q : Quadrilateral) : Prop := sorry
def has_bisecting_diagonals (q : Quadrilateral) : Prop := sorry
def is_square (q : Quadrilateral) : Prop := sorry

-- Theorem statement
theorem quadrilateral_with_perpendicular_bisecting_diagonals_not_necessarily_square :
  ∃ q : Quadrilateral, 
    has_perpendicular_diagonals q ∧ 
    has_bisecting_diagonals q ∧ 
    ¬ is_square q :=
by sorry

end quadrilateral_with_perpendicular_bisecting_diagonals_not_necessarily_square_l307_30765


namespace xyz_value_l307_30720

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 16)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 4) :
  x * y * z = 4 := by
  sorry

end xyz_value_l307_30720


namespace cube_face_sum_l307_30780

/-- Represents the six face values of a cube -/
structure CubeFaces where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  d : ℕ+
  e : ℕ+
  f : ℕ+

/-- Calculates the sum of vertex labels for a given set of cube faces -/
def vertexLabelSum (faces : CubeFaces) : ℕ :=
  faces.a * faces.b * faces.c +
  faces.a * faces.e * faces.c +
  faces.a * faces.b * faces.f +
  faces.a * faces.e * faces.f +
  faces.d * faces.b * faces.c +
  faces.d * faces.e * faces.c +
  faces.d * faces.b * faces.f +
  faces.d * faces.e * faces.f

/-- Theorem stating the sum of face values given the conditions -/
theorem cube_face_sum (faces : CubeFaces)
  (h1 : vertexLabelSum faces = 2002)
  (h2 : faces.a + faces.d = 22) :
  faces.a + faces.b + faces.c + faces.d + faces.e + faces.f = 42 := by
  sorry


end cube_face_sum_l307_30780


namespace specific_conference_games_l307_30715

/-- Calculates the number of games in a sports conference season -/
def conference_games (total_teams : ℕ) (division_size : ℕ) (intra_division_games : ℕ) : ℕ :=
  let teams_per_division := total_teams / 2
  let inter_division_games := division_size
  let games_per_team := (division_size - 1) * intra_division_games + inter_division_games
  (games_per_team * total_teams) / 2

/-- Theorem stating the number of games in the specific conference setup -/
theorem specific_conference_games :
  conference_games 16 8 2 = 176 := by
  sorry

end specific_conference_games_l307_30715


namespace average_score_five_students_l307_30749

theorem average_score_five_students
  (initial_students : Nat)
  (initial_average : ℝ)
  (fifth_student_score : ℝ)
  (h1 : initial_students = 4)
  (h2 : initial_average = 85)
  (h3 : fifth_student_score = 90) :
  (initial_students * initial_average + fifth_student_score) / (initial_students + 1) = 86 :=
by sorry

end average_score_five_students_l307_30749


namespace invitation_methods_l307_30761

def total_teachers : ℕ := 10
def invited_teachers : ℕ := 6

theorem invitation_methods (total : ℕ) (invited : ℕ) : 
  total = total_teachers → invited = invited_teachers →
  (Nat.choose total invited) - (Nat.choose (total - 2) (invited - 2)) = 140 := by
  sorry

end invitation_methods_l307_30761


namespace nested_sqrt_equality_l307_30742

theorem nested_sqrt_equality : Real.sqrt (64 * Real.sqrt (32 * Real.sqrt 16)) = 16 * Real.sqrt 2 := by
  sorry

end nested_sqrt_equality_l307_30742


namespace prob_sum_six_is_five_thirty_sixths_l307_30701

/-- The number of possible outcomes when rolling two dice -/
def total_outcomes : ℕ := 6 * 6

/-- The number of ways to get a sum of 6 when rolling two dice -/
def favorable_outcomes : ℕ := 5

/-- The probability of getting a sum of 6 when rolling two fair dice -/
def prob_sum_six : ℚ := favorable_outcomes / total_outcomes

theorem prob_sum_six_is_five_thirty_sixths :
  prob_sum_six = 5 / 36 := by sorry

end prob_sum_six_is_five_thirty_sixths_l307_30701


namespace sphere_radius_tangent_to_truncated_cone_l307_30777

/-- The radius of a sphere tangent to a truncated cone -/
theorem sphere_radius_tangent_to_truncated_cone 
  (r_bottom r_top h : ℝ) 
  (h_positive : 0 < h) 
  (r_bottom_positive : 0 < r_bottom) 
  (r_top_positive : 0 < r_top) 
  (r_bottom_gt_r_top : r_top < r_bottom) 
  (h_truncated_cone : r_bottom = 24 ∧ r_top = 6 ∧ h = 20) :
  let r := (17 * Real.sqrt 2) / 2
  r = (Real.sqrt ((h^2 + (r_bottom - r_top)^2)) / 2) :=
by sorry

end sphere_radius_tangent_to_truncated_cone_l307_30777


namespace total_games_played_l307_30748

-- Define the structure for the team's season statistics
structure SeasonStats where
  first100WinPercentage : ℝ
  homeWinPercentageAfter100 : ℝ
  awayWinPercentageAfter100 : ℝ
  overallWinPercentage : ℝ
  consecutiveWinStreak : ℕ

-- Define the theorem
theorem total_games_played (stats : SeasonStats) 
  (h1 : stats.first100WinPercentage = 0.85)
  (h2 : stats.homeWinPercentageAfter100 = 0.60)
  (h3 : stats.awayWinPercentageAfter100 = 0.45)
  (h4 : stats.overallWinPercentage = 0.70)
  (h5 : stats.consecutiveWinStreak = 15) :
  ∃ (totalGames : ℕ), totalGames = 186 ∧ 
  (∃ (remainingGames : ℕ), 
    remainingGames % 2 = 0 ∧
    totalGames = 100 + remainingGames ∧
    (85 + (stats.homeWinPercentageAfter100 + stats.awayWinPercentageAfter100) / 2 * remainingGames) / totalGames = stats.overallWinPercentage) :=
by
  sorry

end total_games_played_l307_30748


namespace g_zero_at_neg_one_l307_30793

-- Define the function g
def g (x s : ℝ) : ℝ := 3 * x^5 + 2 * x^4 - x^3 + x^2 - 4 * x + s

-- Theorem statement
theorem g_zero_at_neg_one (s : ℝ) : g (-1) s = 0 ↔ s = -5 := by
  sorry

end g_zero_at_neg_one_l307_30793


namespace polynomial_division_proof_l307_30799

theorem polynomial_division_proof (x : ℚ) : 
  (3 * x^3 + 3 * x^2 - x - 2/3) * (3 * x + 5) + (-2/3) = 
  9 * x^4 + 18 * x^3 + 8 * x^2 - 7 * x + 4 := by
  sorry

end polynomial_division_proof_l307_30799


namespace problem_part1_problem_part2_l307_30759

/-- Custom multiplication operation -/
def customMult (a b : ℤ) : ℤ := a^2 - b + a * b

/-- Theorem for the first part of the problem -/
theorem problem_part1 : customMult 2 (-5) = -1 := by sorry

/-- Theorem for the second part of the problem -/
theorem problem_part2 : customMult (-2) (customMult 2 (-3)) = 1 := by sorry

end problem_part1_problem_part2_l307_30759


namespace isosceles_right_triangle_condition_l307_30757

/-- If the ratios of sine and cosine of angles to their opposite sides are equal in a triangle, then it is an isosceles right triangle. -/
theorem isosceles_right_triangle_condition (A B C : Real) (a b c : Real) :
  (A + B + C = Real.pi) →
  (a > 0) →
  (b > 0) →
  (c > 0) →
  ((Real.sin A) / a = (Real.cos B) / b) →
  ((Real.sin A) / a = (Real.cos C) / c) →
  ((Real.cos B) / b = (Real.cos C) / c) →
  (A = Real.pi / 2 ∧ B = Real.pi / 4 ∧ C = Real.pi / 4) :=
by sorry

end isosceles_right_triangle_condition_l307_30757


namespace crayon_selection_theorem_l307_30787

def total_crayons : ℕ := 15
def karls_selection : ℕ := 3
def friends_selection : ℕ := 4

def selection_ways : ℕ := Nat.choose total_crayons karls_selection * 
                           Nat.choose (total_crayons - karls_selection) friends_selection

theorem crayon_selection_theorem : 
  selection_ways = 225225 := by sorry

end crayon_selection_theorem_l307_30787


namespace digit_sum_puzzle_l307_30766

def is_valid_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def are_different (a b c d e f : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
  d ≠ e ∧ d ≠ f ∧
  e ≠ f

theorem digit_sum_puzzle (a b c d e f : ℕ) :
  is_valid_digit a ∧ is_valid_digit b ∧ is_valid_digit c ∧
  is_valid_digit d ∧ is_valid_digit e ∧ is_valid_digit f ∧
  are_different a b c d e f ∧
  100 * a + 10 * b + c +
  100 * d + 10 * e + a +
  100 * f + 10 * a + b = 1111 →
  a + b + c + d + e + f = 24 := by
sorry

end digit_sum_puzzle_l307_30766


namespace card_value_decrease_l307_30707

theorem card_value_decrease (x : ℝ) :
  (1 - x/100) * (1 - x/100) = 0.64 → x = 20 := by
  sorry

end card_value_decrease_l307_30707


namespace gcd_consecutive_b_terms_l307_30747

def b (n : ℕ) : ℕ := n.factorial + 2 * n

theorem gcd_consecutive_b_terms (n : ℕ) (hn : n ≥ 1) : Nat.gcd (b n) (b (n + 1)) = 2 := by
  sorry

end gcd_consecutive_b_terms_l307_30747


namespace cost_per_page_is_five_cents_l307_30792

def manuscript_copies : ℕ := 10
def binding_cost : ℚ := 5
def pages_per_manuscript : ℕ := 400
def total_cost : ℚ := 250

theorem cost_per_page_is_five_cents :
  let total_binding_cost : ℚ := manuscript_copies * binding_cost
  let copying_cost : ℚ := total_cost - total_binding_cost
  let total_pages : ℕ := manuscript_copies * pages_per_manuscript
  let cost_per_page : ℚ := copying_cost / total_pages
  cost_per_page = 5 / 100 := by sorry

end cost_per_page_is_five_cents_l307_30792


namespace trajectory_of_complex_point_l307_30726

theorem trajectory_of_complex_point (z : ℂ) (h : Complex.abs z ≤ 1) :
  ∃ (P : ℝ × ℝ), P.1^2 + P.2^2 ≤ 1 :=
by sorry

end trajectory_of_complex_point_l307_30726


namespace arrangement_count_l307_30713

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem arrangement_count :
  let total_people : ℕ := 5
  let total_arrangements := factorial total_people
  let arrangements_with_A_first := factorial (total_people - 1)
  let arrangements_with_B_last := factorial (total_people - 1)
  let arrangements_with_A_first_and_B_last := factorial (total_people - 2)
  total_arrangements - arrangements_with_A_first - arrangements_with_B_last + arrangements_with_A_first_and_B_last = 78 :=
by sorry

end arrangement_count_l307_30713


namespace sixteen_divisors_problem_l307_30776

theorem sixteen_divisors_problem (n : ℕ+) : 
  (∃ (d : Fin 16 → ℕ+), 
    (∀ i : Fin 16, d i ∣ n) ∧ 
    (∀ i j : Fin 16, i < j → d i < d j) ∧
    (d 0 = 1) ∧ 
    (d 15 = n) ∧ 
    (d 5 = 18) ∧ 
    (d 8 - d 7 = 17) ∧
    (∀ m : ℕ+, m ∣ n → ∃ i : Fin 16, d i = m)) →
  n = 1998 ∨ n = 3834 := by
sorry

end sixteen_divisors_problem_l307_30776


namespace lcm_hcf_problem_l307_30704

theorem lcm_hcf_problem (a b : ℕ+) (h1 : a = 8) (h2 : Nat.lcm a b = 24) (h3 : Nat.gcd a b = 4) : b = 12 := by
  sorry

end lcm_hcf_problem_l307_30704


namespace leila_weekly_earnings_l307_30762

/-- Represents the earnings of a vlogger over a week -/
def weekly_earnings (daily_viewers : ℕ) (earnings_per_view : ℚ) : ℚ :=
  daily_viewers * earnings_per_view * 7

/-- Proves that Leila earns $350 per week given the conditions -/
theorem leila_weekly_earnings : 
  let voltaire_viewers : ℕ := 50
  let leila_viewers : ℕ := 2 * voltaire_viewers
  let earnings_per_view : ℚ := 1/2
  weekly_earnings leila_viewers earnings_per_view = 350 := by
sorry

end leila_weekly_earnings_l307_30762


namespace equivalent_discount_l307_30778

theorem equivalent_discount (p : ℝ) (k : ℝ) : 
  (1 - k) * p = (1 - 0.05) * (1 - 0.10) * (1 - 0.15) * p ↔ k = 0.27325 := by
  sorry

end equivalent_discount_l307_30778


namespace student_congress_sample_size_l307_30785

/-- Represents a school with classes and students -/
structure School where
  num_classes : Nat
  students_per_class : Nat
  selected_students : Nat

/-- Defines the sample size for a given school -/
def sample_size (s : School) : Nat :=
  s.selected_students

/-- Theorem: The sample size for a school with 40 classes of 50 students each,
    and 150 selected students, is 150 -/
theorem student_congress_sample_size :
  let s : School := { num_classes := 40, students_per_class := 50, selected_students := 150 }
  sample_size s = 150 := by
  sorry


end student_congress_sample_size_l307_30785


namespace prime_factors_of_1998_l307_30769

theorem prime_factors_of_1998 (a b c : ℕ) : 
  Prime a ∧ Prime b ∧ Prime c ∧ 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a < b ∧ b < c ∧
  a * b * c = 1998 →
  (b + c)^a = 1600 := by
  sorry

end prime_factors_of_1998_l307_30769


namespace product_of_reciprocals_plus_one_ge_nine_l307_30744

theorem product_of_reciprocals_plus_one_ge_nine (a b : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) : 
  (1/a + 1) * (1/b + 1) ≥ 9 := by
  sorry

end product_of_reciprocals_plus_one_ge_nine_l307_30744


namespace absolute_value_inequality_l307_30786

theorem absolute_value_inequality (x : ℝ) : |x - 3| < 5 ↔ -2 < x ∧ x < 8 := by
  sorry

end absolute_value_inequality_l307_30786


namespace complex_division_problem_l307_30772

theorem complex_division_problem : (2 - I) / (2 + I) = 3/5 - 4/5 * I :=
by sorry

end complex_division_problem_l307_30772


namespace spherical_to_rectangular_conversion_l307_30708

/-- Conversion from spherical coordinates to rectangular coordinates -/
theorem spherical_to_rectangular_conversion 
  (ρ θ φ : Real) 
  (hρ : ρ = 8) 
  (hθ : θ = 5 * Real.pi / 4) 
  (hφ : φ = Real.pi / 4) : 
  (ρ * Real.sin φ * Real.cos θ, 
   ρ * Real.sin φ * Real.sin θ, 
   ρ * Real.cos φ) = (-4, -4, 4 * Real.sqrt 2) := by
sorry

end spherical_to_rectangular_conversion_l307_30708


namespace one_third_equals_six_l307_30771

theorem one_third_equals_six (x : ℝ) : (1 / 3 : ℝ) * x = 6 → x = 18 := by
  sorry

end one_third_equals_six_l307_30771


namespace max_residents_top_floor_l307_30781

/-- Represents the number of people living on a floor --/
def residents (floor : ℕ) : ℕ := floor

/-- The number of floors in the building --/
def num_floors : ℕ := 10

/-- Theorem: The floor with the most residents is the top floor --/
theorem max_residents_top_floor :
  ∀ k : ℕ, k ≤ num_floors → residents k ≤ residents num_floors :=
by
  sorry

#check max_residents_top_floor

end max_residents_top_floor_l307_30781


namespace task_probability_l307_30752

/-- The probability that task 1 is completed on time -/
def prob_task1 : ℚ := 5/8

/-- The probability that task 2 is completed on time -/
def prob_task2 : ℚ := 3/5

/-- The probability that task 1 is completed on time but task 2 is not -/
def prob_task1_not_task2 : ℚ := prob_task1 * (1 - prob_task2)

theorem task_probability : prob_task1_not_task2 = 1/4 := by
  sorry

end task_probability_l307_30752


namespace price_reduction_percentage_l307_30782

theorem price_reduction_percentage (original_price reduction : ℝ) : 
  original_price = 500 → reduction = 200 → (reduction / original_price) * 100 = 40 := by
  sorry

end price_reduction_percentage_l307_30782


namespace marbles_ratio_l307_30764

/-- Proves that the ratio of marbles given to Savanna to Miriam's current marbles is 3:1 -/
theorem marbles_ratio (initial : ℕ) (current : ℕ) (brother : ℕ) (sister : ℕ) 
  (h1 : initial = 300)
  (h2 : current = 30)
  (h3 : brother = 60)
  (h4 : sister = 2 * brother) : 
  (initial - current - brother - sister) / current = 3 := by
  sorry

end marbles_ratio_l307_30764


namespace complex_inequality_implies_real_range_l307_30745

theorem complex_inequality_implies_real_range (a : ℝ) :
  let z : ℂ := 3 + a * I
  (Complex.abs (z - 2) < 2) → (-Real.sqrt 3 < a ∧ a < Real.sqrt 3) := by
  sorry

end complex_inequality_implies_real_range_l307_30745


namespace vector_AB_equals_2_2_l307_30706

def point := ℝ × ℝ

def A : point := (1, 0)
def B : point := (3, 2)

def vector_AB (p q : point) : ℝ × ℝ :=
  (q.1 - p.1, q.2 - p.2)

theorem vector_AB_equals_2_2 :
  vector_AB A B = (2, 2) := by sorry

end vector_AB_equals_2_2_l307_30706


namespace line_tangent_to_ellipse_l307_30727

/-- A line is tangent to an ellipse if and only if it intersects the ellipse at exactly one point. -/
axiom tangent_iff_single_intersection {m : ℝ} :
  (∃! x y : ℝ, y = m * x + 2 ∧ 3 * x^2 + 6 * y^2 = 6) ↔
  (∃ x y : ℝ, y = m * x + 2 ∧ 3 * x^2 + 6 * y^2 = 6 ∧
    ∀ x' y' : ℝ, y' = m * x' + 2 ∧ 3 * x'^2 + 6 * y'^2 = 6 → x' = x ∧ y' = y)

/-- The theorem stating that if the line y = mx + 2 is tangent to the ellipse 3x^2 + 6y^2 = 6,
    then m^2 = 3/2. -/
theorem line_tangent_to_ellipse (m : ℝ) :
  (∃! x y : ℝ, y = m * x + 2 ∧ 3 * x^2 + 6 * y^2 = 6) → m^2 = 3/2 := by
  sorry

end line_tangent_to_ellipse_l307_30727


namespace cos_five_pi_sixth_plus_alpha_l307_30770

theorem cos_five_pi_sixth_plus_alpha (α : ℝ) (h : Real.sin (π / 3 + α) = 1 / 3) :
  Real.cos (5 * π / 6 + α) = -1 / 3 := by
  sorry

end cos_five_pi_sixth_plus_alpha_l307_30770


namespace log_equality_implies_ratio_one_l307_30739

theorem log_equality_implies_ratio_one (p q : ℝ) 
  (hp : p > 0) (hq : q > 0)
  (h : Real.log p / Real.log 4 = Real.log q / Real.log 6 ∧ 
       Real.log q / Real.log 6 = Real.log (p * q) / Real.log 8) : 
  q / p = 1 := by
sorry

end log_equality_implies_ratio_one_l307_30739


namespace sufficient_not_necessary_condition_l307_30743

theorem sufficient_not_necessary_condition (a : ℝ) :
  (a < 1 → ∃ x : ℝ, x^2 - 2*x + a = 0) ∧
  ¬(∃ x : ℝ, x^2 - 2*x + a = 0 → a < 1) :=
by sorry

end sufficient_not_necessary_condition_l307_30743


namespace marble_selection_ways_l307_30788

def total_marbles : ℕ := 15
def special_colors : ℕ := 3
def marbles_per_special_color : ℕ := 2
def marbles_to_choose : ℕ := 6
def special_marbles_to_choose : ℕ := 2

def remaining_marbles : ℕ := total_marbles - special_colors * marbles_per_special_color
def remaining_marbles_to_choose : ℕ := marbles_to_choose - special_marbles_to_choose

theorem marble_selection_ways :
  (special_colors * (marbles_per_special_color.choose special_marbles_to_choose)) *
  (remaining_marbles.choose remaining_marbles_to_choose) = 1485 := by
  sorry

end marble_selection_ways_l307_30788


namespace max_value_of_vector_sum_l307_30719

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Given unit vectors a and b satisfying |3a + 4b| = |4a - 3b|, and a vector c with |c| = 2,
    the maximum value of |a + b - c| is √2 + 2. -/
theorem max_value_of_vector_sum (a b c : E) 
    (ha : ‖a‖ = 1) 
    (hb : ‖b‖ = 1) 
    (hab : ‖3 • a + 4 • b‖ = ‖4 • a - 3 • b‖) 
    (hc : ‖c‖ = 2) : 
  (‖a + b - c‖ : ℝ) ≤ Real.sqrt 2 + 2 := by
  sorry

end max_value_of_vector_sum_l307_30719


namespace brenda_skittles_l307_30735

/-- Calculates the total number of Skittles Brenda has after buying more. -/
def total_skittles (initial : ℕ) (bought : ℕ) : ℕ :=
  initial + bought

/-- Theorem stating that Brenda ends up with 15 Skittles. -/
theorem brenda_skittles : total_skittles 7 8 = 15 := by
  sorry

end brenda_skittles_l307_30735


namespace quadratic_discriminant_value_l307_30700

/-- The discriminant of a quadratic equation ax^2 + bx + c = 0 -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

theorem quadratic_discriminant_value (a : ℝ) :
  discriminant 1 (-3) (-2*a) = 1 → a = -1 := by
  sorry

end quadratic_discriminant_value_l307_30700


namespace square_pentagon_exterior_angle_l307_30711

/-- The exterior angle formed by a square and a regular pentagon sharing a common side --/
def exteriorAngle (n : ℕ) : ℚ :=
  360 - (180 * (n - 2) / n) - 90

/-- Theorem: The exterior angle BAC formed by a square and a regular pentagon sharing a common side AD is 162° --/
theorem square_pentagon_exterior_angle :
  exteriorAngle 5 = 162 := by
  sorry

end square_pentagon_exterior_angle_l307_30711


namespace gummy_bear_problem_l307_30797

/-- Calculates the number of gummy bear candies left to be shared with others. -/
def candies_left_to_share (initial_candies : ℕ) (siblings : ℕ) (candies_per_sibling : ℕ) (candies_to_eat : ℕ) : ℕ :=
  let remaining_after_siblings := initial_candies - siblings * candies_per_sibling
  let remaining_after_friend := remaining_after_siblings / 2
  remaining_after_friend - candies_to_eat

/-- Theorem stating that given the problem conditions, 19 candies are left to be shared. -/
theorem gummy_bear_problem :
  candies_left_to_share 100 3 10 16 = 19 := by
  sorry

end gummy_bear_problem_l307_30797


namespace female_officers_count_l307_30725

theorem female_officers_count (total_on_duty : ℕ) (male_percentage : ℚ) (female_on_duty_percentage : ℚ) :
  total_on_duty = 500 →
  male_percentage = 60 / 100 →
  female_on_duty_percentage = 10 / 100 →
  (female_on_duty_percentage * (total_female_officers : ℕ) : ℚ) = ((1 - male_percentage) * total_on_duty : ℚ) →
  total_female_officers = 2000 := by
  sorry

end female_officers_count_l307_30725


namespace range_of_a_l307_30751

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x > 2 → a ≤ x + 1 / (x - 2)) → 
  ∃ s : ℝ, s = 4 ∧ ∀ y : ℝ, (∀ x : ℝ, x > 2 → y ≤ x + 1 / (x - 2)) → y ≤ s :=
sorry

end range_of_a_l307_30751


namespace perpendicular_slope_l307_30716

/-- Given a line with equation 5x - 2y = 10, 
    the slope of a perpendicular line is -2/5 -/
theorem perpendicular_slope (x y : ℝ) : 
  (5 * x - 2 * y = 10) → 
  (∃ m : ℝ, m = -2/5 ∧ m * (5/2) = -1) :=
sorry

end perpendicular_slope_l307_30716


namespace todds_snow_cone_business_l307_30714

/-- Todd's snow-cone business problem -/
theorem todds_snow_cone_business 
  (borrowed : ℝ) 
  (repay : ℝ) 
  (ingredients_cost : ℝ) 
  (num_sold : ℕ) 
  (price_per_cone : ℝ) 
  (h1 : borrowed = 100)
  (h2 : repay = 110)
  (h3 : ingredients_cost = 75)
  (h4 : num_sold = 200)
  (h5 : price_per_cone = 0.75)
  : borrowed - ingredients_cost + (num_sold : ℝ) * price_per_cone - repay = 65 :=
by
  sorry


end todds_snow_cone_business_l307_30714


namespace part_one_part_two_l307_30774

-- Define the function f
def f (a x : ℝ) : ℝ := |x - a|

-- Part I
theorem part_one (x : ℝ) : 
  let a := 2
  (x ≤ -1/2 ∨ x ≥ 7/2) ↔ f a x ≥ 4 - |x - 1| :=
sorry

-- Part II
theorem part_two (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  (∀ x, f 1 x ≤ 1 ↔ 0 ≤ x ∧ x ≤ 2) →
  1/m + 1/(2*n) = 1 →
  m + 2*n ≥ 4 :=
sorry

end part_one_part_two_l307_30774


namespace max_triangle_area_max_triangle_area_is_156_l307_30756

/-- The maximum area of the triangle formed by the intersections of three lines in a coordinate plane. -/
theorem max_triangle_area : ℝ :=
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (8, 0)
  let C : ℝ × ℝ := (15, 0)
  let ℓ_A := {(x, y) : ℝ × ℝ | y = 2 * x}
  let ℓ_B := {(x, y) : ℝ × ℝ | x = 8}
  let ℓ_C := {(x, y) : ℝ × ℝ | y = -2 * (x - 15)}
  156

/-- The maximum area of the triangle is 156. -/
theorem max_triangle_area_is_156 : max_triangle_area = 156 := by
  sorry

end max_triangle_area_max_triangle_area_is_156_l307_30756


namespace negation_of_quadratic_equation_l307_30718

theorem negation_of_quadratic_equation :
  (¬ ∀ x : ℝ, x^2 + 2*x - 1 = 0) ↔ (∃ x : ℝ, x^2 + 2*x - 1 ≠ 0) := by
  sorry

end negation_of_quadratic_equation_l307_30718


namespace predicted_distance_is_4km_l307_30722

/-- Represents the cycling challenge scenario -/
structure CyclingChallenge where
  t : ℝ  -- Time taken to cycle first 1 km
  d : ℝ  -- Predicted distance for remaining time

/-- The cycling challenge satisfies the given conditions -/
def valid_challenge (c : CyclingChallenge) : Prop :=
  c.d = (60 - c.t) / c.t ∧  -- First prediction
  c.d = 384 / (c.t + 36)    -- Second prediction after cycling 15 km in 36 minutes

/-- The predicted distance is 4 km -/
theorem predicted_distance_is_4km (c : CyclingChallenge) 
  (h : valid_challenge c) : c.d = 4 := by
  sorry

#check predicted_distance_is_4km

end predicted_distance_is_4km_l307_30722


namespace greatest_common_multiple_15_20_under_150_l307_30795

def is_common_multiple (n m k : ℕ) : Prop := k % n = 0 ∧ k % m = 0

theorem greatest_common_multiple_15_20_under_150 : 
  (∀ k : ℕ, k < 150 → is_common_multiple 15 20 k → k ≤ 120) ∧ 
  is_common_multiple 15 20 120 ∧ 
  120 < 150 :=
sorry

end greatest_common_multiple_15_20_under_150_l307_30795


namespace line_relationships_l307_30790

/-- Two lines in the plane -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Defines when two lines are parallel -/
def parallel (l1 l2 : Line2D) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Defines when two lines coincide -/
def coincide (l1 l2 : Line2D) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ l1.a = k * l2.a ∧ l1.b = k * l2.b ∧ l1.c = k * l2.c

/-- Defines when two lines are perpendicular -/
def perpendicular (l1 l2 : Line2D) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- The main theorem about the relationships between two specific lines -/
theorem line_relationships (m : ℝ) :
  let l1 : Line2D := ⟨m + 3, 4, 3*m - 5⟩
  let l2 : Line2D := ⟨2, m + 5, -8⟩
  (parallel l1 l2 ↔ m = -7) ∧
  (coincide l1 l2 ↔ m = -1) ∧
  (perpendicular l1 l2 ↔ m = -13/3) := by
  sorry

end line_relationships_l307_30790


namespace cruise_group_selection_l307_30753

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem cruise_group_selection :
  choose 9 4 = 126 := by
  sorry

end cruise_group_selection_l307_30753


namespace square_side_length_equal_area_l307_30731

/-- The side length of a square with the same area as a rectangle with length 18 and width 8 is 12 -/
theorem square_side_length_equal_area (length width : ℝ) (x : ℝ) :
  length = 18 →
  width = 8 →
  x ^ 2 = length * width →
  x = 12 := by
  sorry

end square_side_length_equal_area_l307_30731


namespace z_in_second_quadrant_l307_30746

/-- The complex number z defined as (3 + i)i -/
def z : ℂ := (3 + Complex.I) * Complex.I

/-- Predicate to check if a complex number is in the second quadrant -/
def is_in_second_quadrant (w : ℂ) : Prop :=
  w.re < 0 ∧ w.im > 0

/-- Theorem stating that z is in the second quadrant -/
theorem z_in_second_quadrant : is_in_second_quadrant z := by
  sorry

end z_in_second_quadrant_l307_30746


namespace fraction_sum_theorem_l307_30717

theorem fraction_sum_theorem (x y : ℚ) (h : y ≠ 0) : 
  x / y = 2 / 3 → (x + y) / y = 5 / 3 := by
  sorry

end fraction_sum_theorem_l307_30717


namespace existence_of_x_l307_30773

/-- A sequence of nonnegative integers satisfying the given conditions -/
def ValidSequence (a : ℕ → ℕ) : Prop :=
  ∀ i j : ℕ, i ≥ 1 → j ≥ 1 → i + j ≤ 1997 →
    a i + a j ≤ a (i + j) ∧ a (i + j) ≤ a i + a j + 1

/-- The theorem to be proved -/
theorem existence_of_x (a : ℕ → ℕ) (h : ValidSequence a) :
  ∃ x : ℝ, ∀ n : ℕ, 1 ≤ n → n ≤ 1997 → a n = ⌊n * x⌋ := by
  sorry

end existence_of_x_l307_30773


namespace power_one_plus_power_five_quotient_l307_30737

theorem power_one_plus_power_five_quotient (n : ℕ) :
  (1 : ℕ)^345 + 5^10 / 5^7 = 126 := by
  sorry

end power_one_plus_power_five_quotient_l307_30737


namespace fathers_age_ratio_l307_30705

theorem fathers_age_ratio (R : ℕ) : 
  let F := 4 * R
  let father_age_after_8 := F + 8
  let ronit_age_after_8 := R + 8
  let father_age_after_16 := F + 16
  let ronit_age_after_16 := R + 16
  (∃ M : ℕ, father_age_after_8 = M * ronit_age_after_8) ∧ 
  (father_age_after_16 = 2 * ronit_age_after_16) →
  (father_age_after_8 : ℚ) / ronit_age_after_8 = 5 / 2 :=
by sorry

end fathers_age_ratio_l307_30705


namespace smaller_number_in_ratio_l307_30723

theorem smaller_number_in_ratio (x y a b c : ℝ) 
  (h_pos_x : x > 0) 
  (h_pos_y : y > 0) 
  (h_ratio : x / y = a / b) 
  (h_a_lt_b : 0 < a ∧ a < b) 
  (h_sum : x + y = c) : 
  min x y = a * c / (a + b) := by
  sorry

end smaller_number_in_ratio_l307_30723


namespace total_weight_calculation_l307_30783

/-- The molecular weight of a compound in grams per mole -/
def molecular_weight : ℝ := 72

/-- The number of moles of the compound -/
def number_of_moles : ℝ := 4

/-- The total weight of the compound in grams -/
def total_weight : ℝ := molecular_weight * number_of_moles

theorem total_weight_calculation : total_weight = 288 := by
  sorry

end total_weight_calculation_l307_30783


namespace bens_old_car_cost_l307_30702

/-- The cost of Ben's old car in dollars -/
def old_car_cost : ℝ := 1900

/-- The cost of Ben's new car in dollars -/
def new_car_cost : ℝ := 3800

/-- The amount Ben received from selling his old car in dollars -/
def old_car_sale : ℝ := 1800

/-- The amount Ben still owes on his new car in dollars -/
def remaining_debt : ℝ := 2000

/-- Theorem stating that the cost of Ben's old car was $1900 -/
theorem bens_old_car_cost :
  old_car_cost = 1900 ∧
  new_car_cost = 2 * old_car_cost ∧
  new_car_cost = old_car_sale + remaining_debt :=
by sorry

end bens_old_car_cost_l307_30702


namespace mean_equality_implies_y_value_l307_30768

theorem mean_equality_implies_y_value :
  let mean1 := (6 + 9 + 18) / 3
  let mean2 := (12 + y) / 2
  mean1 = mean2 → y = 10 := by
sorry

end mean_equality_implies_y_value_l307_30768


namespace min_value_quadratic_min_value_attainable_l307_30758

theorem min_value_quadratic (a b : ℝ) : a^2 + a*b + b^2 - a - 2*b ≥ -1 := by sorry

theorem min_value_attainable : ∃ (a b : ℝ), a^2 + a*b + b^2 - a - 2*b = -1 := by sorry

end min_value_quadratic_min_value_attainable_l307_30758


namespace parallel_lines_alternate_angles_l307_30729

/-- Two lines in a plane -/
structure Line :=
  (slope : ℝ)
  (intercept : ℝ)

/-- Angle between two lines -/
def angle (l1 l2 : Line) : ℝ := sorry

/-- Predicate for parallel lines -/
def parallel (l1 l2 : Line) : Prop := l1.slope = l2.slope

/-- Predicate for a line intersecting two other lines -/
def intersects (l : Line) (l1 l2 : Line) : Prop := sorry

/-- Predicate for alternate interior angles -/
def alternate_interior_angles (l : Line) (l1 l2 : Line) (α β : ℝ) : Prop := sorry

/-- Theorem: If two parallel lines are intersected by a third line, 
    then the alternate interior angles are equal -/
theorem parallel_lines_alternate_angles 
  (l1 l2 l : Line) (α β : ℝ) : 
  parallel l1 l2 → 
  intersects l l1 l2 → 
  alternate_interior_angles l l1 l2 α β →
  α = β := by sorry

end parallel_lines_alternate_angles_l307_30729


namespace z_max_plus_z_min_l307_30779

theorem z_max_plus_z_min (x y z : ℝ) 
  (h1 : x^2 + y^2 + z^2 = 3) 
  (h2 : x + 2*y - 2*z = 4) : 
  ∃ (z_max z_min : ℝ), 
    (∀ z' : ℝ, (x^2 + y^2 + z'^2 = 3 ∧ x + 2*y - 2*z' = 4) → z' ≤ z_max ∧ z' ≥ z_min) ∧
    z_max + z_min = -4 :=
sorry

end z_max_plus_z_min_l307_30779


namespace womens_doubles_handshakes_l307_30738

/-- The number of handshakes in a women's doubles tennis tournament --/
theorem womens_doubles_handshakes :
  let num_teams : ℕ := 4
  let team_size : ℕ := 2
  let total_players : ℕ := num_teams * team_size
  let handshakes_per_player : ℕ := total_players - team_size
  total_players * handshakes_per_player / 2 = 24 := by
  sorry

end womens_doubles_handshakes_l307_30738


namespace solve_for_y_l307_30784

theorem solve_for_y (x y : ℝ) (h1 : x^(3*y) = 8) (h2 : x = 2) : y = 1 := by
  sorry

end solve_for_y_l307_30784


namespace selection_methods_count_l307_30741

-- Define the total number of students
def total_students : ℕ := 5

-- Define the number of students needed for each role
def translation_students : ℕ := 2
def transportation_students : ℕ := 1
def protocol_students : ℕ := 1

-- Define the total number of students to be selected
def selected_students : ℕ := translation_students + transportation_students + protocol_students

-- The theorem to be proved
theorem selection_methods_count : 
  (Nat.choose total_students translation_students) * 
  (Nat.choose (total_students - translation_students) transportation_students) * 
  (Nat.choose (total_students - translation_students - transportation_students) protocol_students) = 60 := by
  sorry


end selection_methods_count_l307_30741


namespace consecutive_product_divisibility_l307_30721

theorem consecutive_product_divisibility (k : ℤ) :
  let n := k * (k + 1) * (k + 2)
  (∃ m : ℤ, n = 7 * m) →
  (∃ m : ℤ, n = 6 * m) ∧
  (∃ m : ℤ, n = 14 * m) ∧
  (∃ m : ℤ, n = 21 * m) ∧
  (∃ m : ℤ, n = 42 * m) ∧
  ¬(∀ k : ℤ, ∃ m : ℤ, k * (k + 1) * (k + 2) = 28 * m) :=
by sorry

end consecutive_product_divisibility_l307_30721


namespace xy_reciprocal_problem_l307_30724

theorem xy_reciprocal_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : x * y = 1) (h2 : x / y = 36) : y = 1/6 := by
  sorry

end xy_reciprocal_problem_l307_30724


namespace exclusive_or_implies_possible_p_true_q_false_l307_30775

theorem exclusive_or_implies_possible_p_true_q_false (P Q : Prop) 
  (h1 : P ∨ Q) (h2 : ¬(P ∧ Q)) : 
  ∃ (p q : Prop), p = P ∧ q = Q ∧ p = true ∧ q = false :=
sorry

end exclusive_or_implies_possible_p_true_q_false_l307_30775


namespace price_restoration_l307_30728

theorem price_restoration (original_price : ℝ) (reduced_price : ℝ) (h : reduced_price = 0.8 * original_price) :
  reduced_price * 1.25 = original_price := by
  sorry

end price_restoration_l307_30728
