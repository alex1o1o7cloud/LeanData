import Mathlib

namespace largest_integer_solution_l1415_141597

theorem largest_integer_solution : ∃ x : ℤ, (x ≤ 10) ∧ (∀ y : ℤ, (y > 10 → (y / 4 + 5 / 6 < 7 / 2) = false)) :=
sorry

end largest_integer_solution_l1415_141597


namespace fraction_problem_l1415_141596

-- Define the fractions involved in the problem
def frac1 := 18 / 45
def frac2 := 3 / 8
def frac3 := 1 / 9

-- Define the expected result
def expected_result := 49 / 360

-- The proof statement
theorem fraction_problem : frac1 - frac2 + frac3 = expected_result := by
  sorry

end fraction_problem_l1415_141596


namespace problem1_problem2_problem3_problem4_l1415_141517

-- Problem 1
theorem problem1 : (-10 + (-5) - (-18)) = 3 := 
by
  sorry

-- Problem 2
theorem problem2 : (-80 * (-(4 / 5)) / (abs 16)) = -4 := 
by 
  sorry

-- Problem 3
theorem problem3 : ((1/2 - 5/9 + 5/6 - 7/12) * (-36)) = -7 := 
by 
  sorry

-- Problem 4
theorem problem4 : (- 3^2 * (-1/3)^2 +(-2)^2 / (- (2/3))^3) = -29 / 27 :=
by 
  sorry

end problem1_problem2_problem3_problem4_l1415_141517


namespace max_four_color_rectangles_l1415_141512

def color := Fin 4
def grid := Fin 100 × Fin 100
def colored_grid := grid → color

def count_four_color_rectangles (g : colored_grid) : ℕ := sorry

theorem max_four_color_rectangles (g : colored_grid) :
  count_four_color_rectangles g ≤ 9375000 := sorry

end max_four_color_rectangles_l1415_141512


namespace wall_width_is_4_l1415_141550

structure Wall where
  width : ℝ
  height : ℝ
  length : ℝ
  volume : ℝ

theorem wall_width_is_4 (h_eq_6w : ∀ (wall : Wall), wall.height = 6 * wall.width)
                        (l_eq_7h : ∀ (wall : Wall), wall.length = 7 * wall.height)
                        (volume_16128 : ∀ (wall : Wall), wall.volume = 16128) :
  ∃ (wall : Wall), wall.width = 4 :=
by
  sorry

end wall_width_is_4_l1415_141550


namespace area_difference_l1415_141579

theorem area_difference (radius1 radius2 : ℝ) (pi : ℝ) (h1 : radius1 = 15) (h2 : radius2 = 14 / 2) :
  pi * radius1 ^ 2 - pi * radius2 ^ 2 = 176 * pi :=
by 
  sorry

end area_difference_l1415_141579


namespace find_t_l1415_141572

theorem find_t (t : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = |x - t| + |5 - x|) (h2 : ∃ x, f x = 3) : t = 2 ∨ t = 8 :=
by
  sorry

end find_t_l1415_141572


namespace sum_of_reciprocals_of_squares_l1415_141565

theorem sum_of_reciprocals_of_squares (a b : ℕ) (h : a * b = 3) :
  (1 / (a : ℚ)^2) + (1 / (b : ℚ)^2) = 10 / 9 :=
sorry

end sum_of_reciprocals_of_squares_l1415_141565


namespace find_y_l1415_141527

-- Definitions for the given conditions
def angle_sum_triangle (A B C : ℝ) : Prop :=
  A + B + C = 180

def right_triangle (A B : ℝ) : Prop :=
  A + B = 90

-- The main theorem to prove
theorem find_y 
  (angle_ABC : ℝ)
  (angle_BAC : ℝ)
  (angle_DCE : ℝ)
  (h1 : angle_ABC = 70)
  (h2 : angle_BAC = 50)
  (h3 : right_triangle angle_DCE 30)
  : 30 = 30 :=
sorry

end find_y_l1415_141527


namespace sequence_property_l1415_141537

theorem sequence_property (k : ℝ) (h_k : 0 < k) (x : ℕ → ℝ)
  (h₀ : x 0 = 1)
  (h₁ : x 1 = 1 + k)
  (rec1 : ∀ n, x (2*n + 1) - x (2*n) = x (2*n) - x (2*n - 1))
  (rec2 : ∀ n, x (2*n) / x (2*n - 1) = x (2*n - 1) / x (2*n - 2)) :
  ∃ N, ∀ n ≥ N, x n > 1994 :=
by
  sorry

end sequence_property_l1415_141537


namespace coins_in_stack_l1415_141504

-- Define the thickness of each coin type
def penny_thickness : ℝ := 1.55
def nickel_thickness : ℝ := 1.95
def dime_thickness : ℝ := 1.35
def quarter_thickness : ℝ := 1.75

-- Define the total stack height
def total_stack_height : ℝ := 15

-- The statement to prove
theorem coins_in_stack (pennies nickels dimes quarters : ℕ) :
  pennies * penny_thickness + nickels * nickel_thickness + 
  dimes * dime_thickness + quarters * quarter_thickness = total_stack_height →
  pennies + nickels + dimes + quarters = 9 :=
sorry

end coins_in_stack_l1415_141504


namespace A_share_in_profit_l1415_141586

-- Given conditions:
def A_investment : ℕ := 6300
def B_investment : ℕ := 4200
def C_investment : ℕ := 10500
def total_profit : ℕ := 12600

-- The statement we need to prove:
theorem A_share_in_profit :
  (3 / 10) * total_profit = 3780 := by
  sorry

end A_share_in_profit_l1415_141586


namespace max_value_m_l1415_141536

noncomputable def quadratic_function (a b c : ℝ) (x : ℝ) := a * x^2 + b * x + c

theorem max_value_m (a b c : ℝ) (h1 : a ≠ 0)
  (h2 : ∀ x : ℝ, quadratic_function a b c (x-4) = quadratic_function a b c (2-x))
  (h3 : ∀ x : ℝ, 0 < x ∧ x < 2 → quadratic_function a b c x ≤ ( (x+1)/2 )^2)
  (h4 : ∀ x : ℝ, quadratic_function a b c x ≥ 0)
  (h_min : ∃ x : ℝ, quadratic_function a b c x = 0) :
  ∃ (m : ℝ), m > 1 ∧ (∃ t : ℝ, ∀ x : ℝ, 1 ≤ x ∧ x ≤ m → quadratic_function a b c (x+t) ≤ x) ∧ m = 9 := 
sorry

end max_value_m_l1415_141536


namespace correct_match_results_l1415_141534

-- Define the teams in the league
inductive Team
| Scotland : Team
| England  : Team
| Wales    : Team
| Ireland  : Team

-- Define a match result for a pair of teams
structure MatchResult where
  team1 : Team
  team2 : Team
  goals1 : ℕ
  goals2 : ℕ

def scotland_vs_england : MatchResult := {
  team1 := Team.Scotland,
  team2 := Team.England,
  goals1 := 3,
  goals2 := 0
}

-- All possible match results
def england_vs_ireland : MatchResult := {
  team1 := Team.England,
  team2 := Team.Ireland,
  goals1 := 1,
  goals2 := 0
}

def wales_vs_england : MatchResult := {
  team1 := Team.Wales,
  team2 := Team.England,
  goals1 := 1,
  goals2 := 1
}

def wales_vs_ireland : MatchResult := {
  team1 := Team.Wales,
  team2 := Team.Ireland,
  goals1 := 2,
  goals2 := 1
}

def scotland_vs_ireland : MatchResult := {
  team1 := Team.Scotland,
  team2 := Team.Ireland,
  goals1 := 2,
  goals2 := 0
}

theorem correct_match_results : 
  (england_vs_ireland.goals1 = 1 ∧ england_vs_ireland.goals2 = 0) ∧
  (wales_vs_england.goals1 = 1 ∧ wales_vs_england.goals2 = 1) ∧
  (scotland_vs_england.goals1 = 3 ∧ scotland_vs_england.goals2 = 0) ∧
  (wales_vs_ireland.goals1 = 2 ∧ wales_vs_ireland.goals2 = 1) ∧
  (scotland_vs_ireland.goals1 = 2 ∧ scotland_vs_ireland.goals2 = 0) :=
by 
  sorry

end correct_match_results_l1415_141534


namespace solve_inequality_l1415_141507

theorem solve_inequality (x : ℝ) :
  (x * (x + 2) / (x - 3) < 0) ↔ (x < -2 ∨ (0 < x ∧ x < 3)) :=
sorry

end solve_inequality_l1415_141507


namespace kim_average_increase_l1415_141551

noncomputable def avg (scores : List ℚ) : ℚ :=
  (scores.sum) / (scores.length)

theorem kim_average_increase :
  let scores_initial := [85, 89, 90, 92]  -- Initial scores
  let score_fifth := 95  -- Fifth score
  let original_average := avg scores_initial
  let new_average := avg (scores_initial ++ [score_fifth])
  new_average - original_average = 1.2 := by
  let scores_initial : List ℚ := [85, 89, 90, 92]
  let score_fifth : ℚ := 95
  let original_average : ℚ := avg scores_initial
  let new_average : ℚ := avg (scores_initial ++ [score_fifth])
  have : new_average - original_average = 1.2 := sorry
  exact this

end kim_average_increase_l1415_141551


namespace movie_theater_attendance_l1415_141526

theorem movie_theater_attendance : 
  let total_seats := 750
  let empty_seats := 218
  let people := total_seats - empty_seats
  people = 532 :=
by
  sorry

end movie_theater_attendance_l1415_141526


namespace total_dolls_combined_l1415_141530

-- Define the number of dolls for Vera
def vera_dolls : ℕ := 20

-- Define the relationship that Sophie has twice as many dolls as Vera
def sophie_dolls : ℕ := 2 * vera_dolls

-- Define the relationship that Aida has twice as many dolls as Sophie
def aida_dolls : ℕ := 2 * sophie_dolls

-- The statement to prove that the total number of dolls is 140
theorem total_dolls_combined : aida_dolls + sophie_dolls + vera_dolls = 140 :=
by
  sorry

end total_dolls_combined_l1415_141530


namespace smallest_abc_sum_l1415_141587

theorem smallest_abc_sum : 
  ∃ (a b c : ℕ), (a * c + 2 * b * c + a + 2 * b = c^2 + c + 6) ∧ (∀ (a' b' c' : ℕ), (a' * c' + 2 * b' * c' + a' + 2 * b' = c'^2 + c' + 6) → (a' + b' + c' ≥ a + b + c)) → (a, b, c) = (2, 1, 1) := 
by
  sorry

end smallest_abc_sum_l1415_141587


namespace pause_point_l1415_141541

-- Definitions
def total_movie_length := 60 -- In minutes
def remaining_time := 30 -- In minutes

-- Theorem stating the pause point in the movie
theorem pause_point : total_movie_length - remaining_time = 30 := by
  -- This is the original solution in mathematical terms, omitted in lean statement.
  -- total_movie_length - remaining_time = 60 - 30 = 30
  sorry

end pause_point_l1415_141541


namespace servings_per_day_l1415_141533

-- Definitions based on the given problem conditions
def serving_size : ℚ := 0.5
def container_size : ℚ := 32 - 2 -- 1 quart is 32 ounces and the jar is 2 ounces less
def days_last : ℕ := 20

-- The theorem statement to prove
theorem servings_per_day (h1 : serving_size = 0.5) (h2 : container_size = 30) (h3 : days_last = 20) :
  (container_size / days_last) / serving_size = 3 :=
by
  sorry

end servings_per_day_l1415_141533


namespace find_a_l1415_141557

theorem find_a (a x1 x2 : ℝ)
  (h1: 4 * x1 ^ 2 - 4 * (a + 2) * x1 + a ^ 2 + 11 = 0)
  (h2: 4 * x2 ^ 2 - 4 * (a + 2) * x2 + a ^ 2 + 11 = 0)
  (h3: x1 - x2 = 3) : a = 4 := sorry

end find_a_l1415_141557


namespace greatest_value_exprD_l1415_141548

-- Conditions
def a : ℚ := 2
def b : ℚ := 5

-- Expressions
def exprA := a / b
def exprB := b / a
def exprC := a - b
def exprD := b - a
def exprE := (1/2 : ℚ) * a

-- Proof problem statement
theorem greatest_value_exprD : exprD = 3 ∧ exprD > exprA ∧ exprD > exprB ∧ exprD > exprC ∧ exprD > exprE := sorry

end greatest_value_exprD_l1415_141548


namespace cookies_in_fridge_l1415_141529

theorem cookies_in_fridge (total_baked : ℕ) (cookies_Tim : ℕ) (cookies_Mike : ℕ) (cookies_Sarah : ℕ) (cookies_Anna : ℕ)
  (h_total_baked : total_baked = 1024)
  (h_cookies_Tim : cookies_Tim = 48)
  (h_cookies_Mike : cookies_Mike = 58)
  (h_cookies_Sarah : cookies_Sarah = 78)
  (h_cookies_Anna : cookies_Anna = (2 * (cookies_Tim + cookies_Mike)) - (cookies_Sarah / 2)) :
  total_baked - (cookies_Tim + cookies_Mike + cookies_Sarah + cookies_Anna) = 667 := by
sorry

end cookies_in_fridge_l1415_141529


namespace volume_in_cubic_yards_l1415_141590

theorem volume_in_cubic_yards (V : ℝ) (conversion_factor : ℝ) (hV : V = 216) (hcf : conversion_factor = 27) :
  V / conversion_factor = 8 := by
  sorry

end volume_in_cubic_yards_l1415_141590


namespace calculation1_calculation2_calculation3_calculation4_l1415_141554

theorem calculation1 : 72 * 54 + 28 * 54 = 5400 := 
by sorry

theorem calculation2 : 60 * 25 * 8 = 12000 := 
by sorry

theorem calculation3 : 2790 / (250 * 12 - 2910) = 31 := 
by sorry

theorem calculation4 : (100 - 1456 / 26) * 78 = 3432 := 
by sorry

end calculation1_calculation2_calculation3_calculation4_l1415_141554


namespace readers_of_science_fiction_l1415_141563

variable (Total S L B : Nat)

theorem readers_of_science_fiction 
  (h1 : Total = 400) 
  (h2 : L = 230) 
  (h3 : B = 80) 
  (h4 : Total = S + L - B) : 
  S = 250 := 
by
  sorry

end readers_of_science_fiction_l1415_141563


namespace compare_fractions_l1415_141546

theorem compare_fractions : (6/29 : ℚ) < (8/25 : ℚ) ∧ (8/25 : ℚ) < (11/31 : ℚ):=
by
  have h1 : (6/29 : ℚ) < (8/25 : ℚ) := sorry
  have h2 : (8/25 : ℚ) < (11/31 : ℚ) := sorry
  exact ⟨h1, h2⟩

end compare_fractions_l1415_141546


namespace stock_exchange_total_l1415_141532

theorem stock_exchange_total (L H : ℕ) 
  (h1 : H = 1080) 
  (h2 : H = 6 * L / 5) : 
  (L + H = 1980) :=
by {
  -- L and H are given as natural numbers
  -- h1: H = 1080
  -- h2: H = 1.20L -> H = 6L/5 as Lean does not handle floating point well directly in integers.
  sorry
}

end stock_exchange_total_l1415_141532


namespace circumscribed_sphere_radius_l1415_141582

theorem circumscribed_sphere_radius (a b R : ℝ) (ha : a > 0) (hb : b > 0) :
  R = b^2 / (2 * (Real.sqrt (b^2 - a^2))) :=
sorry

end circumscribed_sphere_radius_l1415_141582


namespace x_varies_as_sin_squared_l1415_141553

variable {k j z : ℝ}
variable (x y : ℝ)

-- condition: x is proportional to y^2
def proportional_xy_square (x y : ℝ) (k : ℝ) : Prop :=
  x = k * y ^ 2

-- condition: y is proportional to sin(z)
def proportional_y_sin (y : ℝ) (j z : ℝ) : Prop :=
  y = j * Real.sin z

-- statement to prove: x is proportional to (sin(z))^2
theorem x_varies_as_sin_squared (k j z : ℝ) (x y : ℝ)
  (h1 : proportional_xy_square x y k)
  (h2 : proportional_y_sin y j z) :
  ∃ m, x = m * (Real.sin z) ^ 2 :=
by
  sorry

end x_varies_as_sin_squared_l1415_141553


namespace hyperbola_equation_l1415_141562

theorem hyperbola_equation (a b k : ℝ) (p : ℝ × ℝ) (h_asymptotes : b = 3 * a)
  (h_hyperbola_passes_point : p = (2, -3 * Real.sqrt 3)) (h_hyperbola : ∀ x y, x^2 - (y^2 / (3 * a)^2) = k) :
  ∃ k, k = 1 :=
by
  -- Given the point p and asymptotes, we should prove k = 1.
  sorry

end hyperbola_equation_l1415_141562


namespace sector_area_l1415_141524

theorem sector_area (n : ℝ) (r : ℝ) (h₁ : n = 120) (h₂ : r = 4) : 
  (n * Real.pi * r^2 / 360) = (16 * Real.pi / 3) :=
by 
  sorry

end sector_area_l1415_141524


namespace C_completes_work_in_4_days_l1415_141516

theorem C_completes_work_in_4_days
  (A_days : ℕ)
  (B_efficiency : ℕ → ℕ)
  (C_efficiency : ℕ → ℕ)
  (hA : A_days = 12)
  (hB : ∀ {x}, B_efficiency x = x * 3 / 2)
  (hC : ∀ {x}, C_efficiency x = x * 2) :
  (1 / (1 / (C_efficiency (B_efficiency A_days)))) = 4 := by
  sorry

end C_completes_work_in_4_days_l1415_141516


namespace hypotenuse_of_45_45_90_triangle_l1415_141531

theorem hypotenuse_of_45_45_90_triangle (a : ℝ) (h : ℝ) 
  (ha : a = 15) 
  (angle_opposite_leg : ℝ) 
  (h_angle : angle_opposite_leg = 45) 
  (right_triangle : ∃ θ : ℝ, θ = 90) : 
  h = 15 * Real.sqrt 2 := 
sorry

end hypotenuse_of_45_45_90_triangle_l1415_141531


namespace cannot_form_equilateral_triangle_from_spliced_isosceles_right_triangles_l1415_141567

/- Definitions -/
def is_isosceles_right_triangle (triangle : Type) (a b c : ℝ) (angleA angleB angleC : ℝ) : Prop :=
  -- A triangle is isosceles right triangle if it has two equal angles of 45 degrees and a right angle of 90 degrees
  a = b ∧ angleA = 45 ∧ angleB = 45 ∧ angleC = 90

/- Main Problem Statement -/
theorem cannot_form_equilateral_triangle_from_spliced_isosceles_right_triangles
  (T1 T2 : Type) (a1 b1 c1 a2 b2 c2 : ℝ) 
  (angleA1 angleB1 angleC1 angleA2 angleB2 angleC2 : ℝ) :
  is_isosceles_right_triangle T1 a1 b1 c1 angleA1 angleB1 angleC1 →
  is_isosceles_right_triangle T2 a2 b2 c2 angleA2 angleB2 angleC2 →
  ¬ (∃ (a b c : ℝ), a = b ∧ b = c ∧ a = c ∧ (a + b + c = 180)) :=
by
  intros hT1 hT2
  intro h
  sorry

end cannot_form_equilateral_triangle_from_spliced_isosceles_right_triangles_l1415_141567


namespace tailor_trim_length_l1415_141521

theorem tailor_trim_length (x : ℕ) : 
  (18 - x) * 15 = 120 → x = 10 := 
by
  sorry

end tailor_trim_length_l1415_141521


namespace ratio_of_radii_l1415_141503

theorem ratio_of_radii (r R : ℝ) (hR : R > 0) (hr : r > 0)
  (h : π * R^2 - π * r^2 = 4 * (π * r^2)) : r / R = 1 / Real.sqrt 5 :=
by
  sorry

end ratio_of_radii_l1415_141503


namespace range_of_abs_function_l1415_141560

theorem range_of_abs_function : ∀ (y : ℝ), (∃ (x : ℝ), y = |x + 5| - |x - 3|) ↔ y ∈ Set.Icc (-8) 8 :=
by
  sorry

end range_of_abs_function_l1415_141560


namespace convert_quadratic_l1415_141522

theorem convert_quadratic (x : ℝ) :
  (1 + 3 * x) * (x - 3) = 2 * x ^ 2 + 1 ↔ x ^ 2 - 8 * x - 4 = 0 := 
by sorry

end convert_quadratic_l1415_141522


namespace john_younger_than_mark_l1415_141598

variable (Mark_age John_age Parents_age : ℕ)
variable (h_mark : Mark_age = 18)
variable (h_parents_age_relation : Parents_age = 5 * John_age)
variable (h_parents_when_mark_born : Parents_age = 22 + Mark_age)

theorem john_younger_than_mark : Mark_age - John_age = 10 :=
by
  -- We state the theorem and leave the proof as sorry
  sorry

end john_younger_than_mark_l1415_141598


namespace inequality_proof_l1415_141547

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (a^2 / b + b^2 / c + c^2 / a) ≥ 3 * (a^3 + b^3 + c^3) / (a^2 + b^2 + c^2) := 
sorry

end inequality_proof_l1415_141547


namespace arithmetic_sequence_sum_l1415_141520

variable (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℕ)

def S₁₀ (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℕ) : ℕ :=
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀

theorem arithmetic_sequence_sum (h : S₁₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ = 120) :
  a₁ + a₁₀ = 24 :=
by
  sorry

end arithmetic_sequence_sum_l1415_141520


namespace intersection_eq_singleton_l1415_141509

def A : Set ℝ := {-1, 0, 1}
def B : Set ℝ := {x | 0 < x ∧ x < 2}

theorem intersection_eq_singleton :
  A ∩ B = {1} :=
sorry

end intersection_eq_singleton_l1415_141509


namespace mean_score_of_all_students_l1415_141595

-- Define the conditions as given in the problem
variables (M A : ℝ) (m a : ℝ)
  (hM : M = 90)
  (hA : A = 75)
  (hRatio : m / a = 2 / 5)

-- State the theorem which proves that the mean score of all students is 79
theorem mean_score_of_all_students (hM : M = 90) (hA : A = 75) (hRatio : m / a = 2 / 5) : 
  (36 * a + 75 * a) / ((2 / 5) * a + a) = 79 := 
by
  sorry -- Proof is omitted

end mean_score_of_all_students_l1415_141595


namespace hoursWorkedPerDay_l1415_141556

-- Define the conditions
def widgetsPerHour := 20
def daysPerWeek := 5
def totalWidgetsPerWeek := 800

-- Theorem statement
theorem hoursWorkedPerDay : (totalWidgetsPerWeek / widgetsPerHour) / daysPerWeek = 8 := 
  sorry

end hoursWorkedPerDay_l1415_141556


namespace simplify_and_evaluate_expression_l1415_141592

theorem simplify_and_evaluate_expression (m : ℝ) (h : m = Real.sqrt 3 + 1) :
  (1 - 1 / m) / ((m ^ 2 - 2 * m + 1) / m) = Real.sqrt 3 / 3 :=
by
  sorry

end simplify_and_evaluate_expression_l1415_141592


namespace initial_fraction_of_larger_jar_l1415_141589

theorem initial_fraction_of_larger_jar (S L W : ℝ) 
  (h1 : W = 1/6 * S) 
  (h2 : W = 1/3 * L) : 
  W / L = 1 / 3 := 
by 
  sorry

end initial_fraction_of_larger_jar_l1415_141589


namespace simplify_fraction_l1415_141528

variable {a b c : ℝ}

theorem simplify_fraction (h : a + b + c ≠ 0) :
  (a^2 + 3*a*b + b^2 - c^2) / (a^2 + 3*a*c + c^2 - b^2) = (a + b - c) / (a - b + c) := 
by
  sorry

end simplify_fraction_l1415_141528


namespace probability_no_shaded_rectangle_l1415_141549

theorem probability_no_shaded_rectangle :
  let n := (1002 * 1001) / 2
  let m := 501 * 501
  (1 - (m / n) = 500 / 1001) := sorry

end probability_no_shaded_rectangle_l1415_141549


namespace function_increasing_l1415_141519

noncomputable def is_monotonically_increasing (f : ℝ → ℝ) :=
  ∀ (x1 x2 : ℝ), x1 < x2 → f x1 < f x2

theorem function_increasing {f : ℝ → ℝ}
  (H : ∀ (x1 x2 : ℝ), x1 < x2 → f x1 < f x2) :
  is_monotonically_increasing f :=
by
  sorry

end function_increasing_l1415_141519


namespace arithmetic_sequence_inequality_l1415_141576

variable {α : Type*} [OrderedRing α]

theorem arithmetic_sequence_inequality 
  (a : ℕ → α) (d : α) 
  (h_arith_seq : ∀ n, a (n + 1) = a n + d)
  (h_pos : ∀ n, a n > 0)
  (h_d_ne_zero : d ≠ 0) : 
  a 0 * a 7 < a 3 * a 4 := 
by
  sorry

end arithmetic_sequence_inequality_l1415_141576


namespace oven_capacity_correct_l1415_141501

-- Definitions for the conditions
def dough_time := 30 -- minutes
def bake_time := 30 -- minutes
def pizzas_per_batch := 3
def total_time := 5 * 60 -- minutes (5 hours)
def total_pizzas := 12

-- Calculation of the number of batches
def batches_needed := total_pizzas / pizzas_per_batch

-- Calculation of the time for making dough
def dough_preparation_time := batches_needed * dough_time

-- Calculation of the remaining time for baking
def remaining_baking_time := total_time - dough_preparation_time

-- Calculation of the number of 30-minute baking intervals
def baking_intervals := remaining_baking_time / bake_time

-- Calculation of the capacity of the oven
def oven_capacity := total_pizzas / baking_intervals

theorem oven_capacity_correct : oven_capacity = 2 := by
  sorry

end oven_capacity_correct_l1415_141501


namespace min_value_of_x4_y3_z2_l1415_141506

noncomputable def min_value_x4_y3_z2 (x y z : ℝ) : ℝ :=
  x^4 * y^3 * z^2

theorem min_value_of_x4_y3_z2 (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : 1/x + 1/y + 1/z = 9) : 
  min_value_x4_y3_z2 x y z = 1 / 3456 :=
by
  sorry

end min_value_of_x4_y3_z2_l1415_141506


namespace spinner_points_east_l1415_141505

-- Definitions for the conditions
def initial_direction := "north"

-- Clockwise and counterclockwise movements as improper fractions
def clockwise_move := (7 : ℚ) / 2
def counterclockwise_move := (17 : ℚ) / 4

-- Compute the net movement (negative means counterclockwise)
def net_movement := clockwise_move - counterclockwise_move

-- Translate net movement into a final direction (using modulo arithmetic with 1 revolution = 360 degrees equivalent)
def final_position : ℚ := (net_movement + 1) % 1

-- The goal is to prove that the final direction is east (which corresponds to 1/4 revolution)
theorem spinner_points_east :
  final_position = (1 / 4 : ℚ) :=
by
  sorry

end spinner_points_east_l1415_141505


namespace ratio_of_ages_l1415_141543

theorem ratio_of_ages (x m : ℕ) 
  (mother_current_age : ℕ := 41) 
  (daughter_current_age : ℕ := 23) 
  (age_diff : ℕ := mother_current_age - daughter_current_age) 
  (eq : (mother_current_age - x) = m * (daughter_current_age - x)) : 
  (41 - x) / (23 - x) = m :=
by
  -- Proof not required
  sorry

end ratio_of_ages_l1415_141543


namespace choose_team_captains_l1415_141574

open Nat

def binom (n k : ℕ) : ℕ :=
  if k > n then 0 else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem choose_team_captains :
  let total_members := 15
  let shortlisted := 5
  let regular := total_members - shortlisted
  binom total_members 4 - binom regular 4 = 1155 :=
by
  sorry

end choose_team_captains_l1415_141574


namespace part1_part2_l1415_141591

-- Definitions for the sets A and B
def A : Set ℝ := {x | x^2 - 3 * x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + 2 * (a - 1) * x + (a^2 - 5) = 0}

-- Proof problem (1): A ∩ B = {2} implies a = -5 or a = 1
theorem part1 (a : ℝ) (h : A ∩ B a = {2}) : a = -5 ∨ a = 1 := 
sorry

-- Proof problem (2): A ∪ B = A implies a > 3
theorem part2 (a : ℝ) (h : A ∪ B a = A) : 3 < a :=
sorry

end part1_part2_l1415_141591


namespace compare_values_l1415_141568

variable (f : ℝ → ℝ)
variable (hf_even : ∀ x, f x = f (-x))
variable (hf_decreasing : ∀ x y, 0 ≤ x → x ≤ y → f y ≤ f x)

noncomputable def a : ℝ := f 1
noncomputable def b : ℝ := f (Real.log 3 / Real.log 0.5)
noncomputable def c : ℝ := f ((Real.log 3 / Real.log 2) - 1)

theorem compare_values (h_log1 : Real.log 3 / Real.log 0.5 < -1) 
                       (h_log2 : 0 < (Real.log 3 / Real.log 2) - 1 ∧ (Real.log 3 / Real.log 2) - 1 < 1) : 
  b < a ∧ a < c :=
by
  sorry

end compare_values_l1415_141568


namespace scientific_notation_40_9_billion_l1415_141514

theorem scientific_notation_40_9_billion :
  (40.9 * 10^9) = 4.09 * 10^9 :=
by
  sorry

end scientific_notation_40_9_billion_l1415_141514


namespace single_discount_equivalence_l1415_141559

variable (p : ℝ) (d1 d2 d3 : ℝ)

def apply_discount (price discount : ℝ) : ℝ :=
  price * (1 - discount)

def apply_multiple_discounts (price : ℝ) (discounts : List ℝ) : ℝ :=
  discounts.foldl apply_discount price

theorem single_discount_equivalence :
  p = 1200 →
  d1 = 0.15 →
  d2 = 0.10 →
  d3 = 0.05 →
  let final_price_multiple := apply_multiple_discounts p [d1, d2, d3]
  let single_discount := (p - final_price_multiple) / p
  single_discount = 0.27325 :=
by
  intros h1 h2 h3 h4
  let final_price_multiple := apply_multiple_discounts p [d1, d2, d3]
  let single_discount := (p - final_price_multiple) / p
  sorry

end single_discount_equivalence_l1415_141559


namespace find_middle_and_oldest_sons_l1415_141561

-- Defining the conditions
def youngest_age : ℕ := 2
def father_age : ℕ := 33
def father_age_in_12_years : ℕ := father_age + 12
def youngest_age_in_12_years : ℕ := youngest_age + 12

-- Lean theorem statement to find the ages of the middle and oldest sons
theorem find_middle_and_oldest_sons (y z : ℕ) (h1 : father_age_in_12_years = (youngest_age_in_12_years + 12 + y + 12 + z + 12)) :
  y = 3 ∧ z = 4 :=
sorry

end find_middle_and_oldest_sons_l1415_141561


namespace intersection_eq_two_l1415_141518

def A : Set ℤ := {1, 2, 3}
def B : Set ℤ := {-2, 2}

theorem intersection_eq_two : A ∩ B = {2} := by
  sorry

end intersection_eq_two_l1415_141518


namespace smallest_5_digit_number_divisible_by_and_factor_of_l1415_141502

def lcm (a b : ℕ) : ℕ := a * b / Nat.gcd a b

def is_divisible_by (x y : ℕ) : Prop := ∃ k : ℕ, x = y * k

def is_factor_of (x y : ℕ) : Prop := is_divisible_by y x

def is_5_digit_number (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

theorem smallest_5_digit_number_divisible_by_and_factor_of :
  ∃ n : ℕ,
    is_5_digit_number n ∧
    is_divisible_by n 32 ∧
    is_divisible_by n 45 ∧
    is_divisible_by n 54 ∧
    is_factor_of n 30 ∧
    (∀ m : ℕ, is_5_digit_number m → is_divisible_by m 32 → is_divisible_by m 45 → is_divisible_by m 54 → is_factor_of m 30 → n ≤ m) :=
sorry

end smallest_5_digit_number_divisible_by_and_factor_of_l1415_141502


namespace candidate_B_valid_votes_l1415_141542

theorem candidate_B_valid_votes:
  let eligible_voters := 12000
  let abstained_percent := 0.1
  let invalid_votes_percent := 0.2
  let votes_for_C_percent := 0.05
  let A_less_B_percent := 0.2
  let total_voted := (1 - abstained_percent) * eligible_voters
  let valid_votes := (1 - invalid_votes_percent) * total_voted
  let votes_for_C := votes_for_C_percent * valid_votes
  (∃ Vb, valid_votes = (1 - A_less_B_percent) * Vb + Vb + votes_for_C 
         ∧ Vb = 4560) :=
sorry

end candidate_B_valid_votes_l1415_141542


namespace roots_of_quadratic_l1415_141575

theorem roots_of_quadratic (x : ℝ) : 3 * (x - 3) = (x - 3) ^ 2 → x = 3 ∨ x = 6 :=
by
  intro h
  sorry

end roots_of_quadratic_l1415_141575


namespace exists_special_cubic_polynomial_l1415_141594

theorem exists_special_cubic_polynomial :
  ∃ P : Polynomial ℝ, 
    Polynomial.degree P = 3 ∧ 
    (∀ x : ℝ, Polynomial.IsRoot P x → x > 0) ∧
    (∀ x : ℝ, Polynomial.IsRoot (Polynomial.derivative P) x → x < 0) ∧
    (∃ x y : ℝ, Polynomial.IsRoot P x ∧ Polynomial.IsRoot (Polynomial.derivative P) y ∧ x ≠ y) :=
by
  sorry

end exists_special_cubic_polynomial_l1415_141594


namespace snow_shoveling_l1415_141508

noncomputable def volume_of_snow_shoveled (length1 length2 width depth1 depth2 : ℝ) : ℝ :=
  (length1 * width * depth1) + (length2 * width * depth2)

theorem snow_shoveling :
  volume_of_snow_shoveled 15 15 4 1 (1 / 2) = 90 :=
by
  sorry

end snow_shoveling_l1415_141508


namespace samuel_faster_than_sarah_l1415_141569

-- Definitions based on the conditions
def time_samuel : ℝ := 30
def time_sarah : ℝ := 1.3 * 60

-- The theorem to prove that Samuel finished his homework 48 minutes faster than Sarah
theorem samuel_faster_than_sarah : (time_sarah - time_samuel) = 48 := by
  sorry

end samuel_faster_than_sarah_l1415_141569


namespace tank_capacity_l1415_141578

theorem tank_capacity :
  ∃ T : ℝ, (5/8) * T + 12 = (11/16) * T ∧ T = 192 :=
sorry

end tank_capacity_l1415_141578


namespace calculate_expr_l1415_141545

theorem calculate_expr : (125 : ℝ)^(2/3) * 2 = 50 := sorry

end calculate_expr_l1415_141545


namespace sum_units_digits_3a_l1415_141500

theorem sum_units_digits_3a (a : ℕ) (h_pos : 0 < a) (h_units : (2 * a) % 10 = 4) : 
  ((3 * (a % 10) = (6 : ℕ) ∨ (3 * (a % 10) = (21 : ℕ))) → 6 + 1 = 7) := 
by
  sorry

end sum_units_digits_3a_l1415_141500


namespace center_of_circle_in_second_quadrant_l1415_141539

theorem center_of_circle_in_second_quadrant (a : ℝ) (h : a > 12) :
  ∃ x y : ℝ, x^2 + y^2 + a * x - 2 * a * y + a^2 + 3 * a = 0 ∧ (-a / 2, a).2 > 0 ∧ (-a / 2, a).1 < 0 :=
by
  sorry

end center_of_circle_in_second_quadrant_l1415_141539


namespace joshua_final_bottle_caps_l1415_141552

def initial_bottle_caps : ℕ := 150
def bought_bottle_caps : ℕ := 23
def given_away_bottle_caps : ℕ := 37

theorem joshua_final_bottle_caps : (initial_bottle_caps + bought_bottle_caps - given_away_bottle_caps) = 136 := by
  sorry

end joshua_final_bottle_caps_l1415_141552


namespace find_b_in_triangle_l1415_141515

theorem find_b_in_triangle (a B C A b : ℝ)
  (ha : a = Real.sqrt 3)
  (hB : Real.sin B = 1 / 2)
  (hC : C = Real.pi / 6)
  (hA : A = 2 * Real.pi / 3) :
  b = 1 :=
by
  -- proof omitted
  sorry

end find_b_in_triangle_l1415_141515


namespace range_of_m_l1415_141564

variable (a b c m y1 y2 y3 : Real)

-- Given points and the parabola equation
def on_parabola (x y a b c : Real) : Prop := y = a * x^2 + b * x + c

-- Conditions
variable (hP : on_parabola (-2) y1 a b c)
variable (hQ : on_parabola 4 y2 a b c)
variable (hM : on_parabola m y3 a b c)
variable (h_vertex : 2 * a * m + b = 0)
variable (h_y_order : y3 ≥ y2 ∧ y2 > y1)

-- Theorem to prove m > 1
theorem range_of_m : m > 1 :=
sorry

end range_of_m_l1415_141564


namespace polynomial_expansion_proof_l1415_141544

variable (z : ℤ)

-- Define the polynomials p and q
noncomputable def p (z : ℤ) : ℤ := 3 * z^2 - 4 * z + 1
noncomputable def q (z : ℤ) : ℤ := 2 * z^3 + 3 * z^2 - 5 * z + 2

-- Define the expanded polynomial
noncomputable def expanded (z : ℤ) : ℤ :=
  6 * z^5 + z^4 - 25 * z^3 + 29 * z^2 - 13 * z + 2

-- The goal is to prove the equivalence of (p * q) == expanded 
theorem polynomial_expansion_proof :
  (p z) * (q z) = expanded z :=
by
  sorry

end polynomial_expansion_proof_l1415_141544


namespace scientific_notation_32000000_l1415_141535

def scientific_notation (n : ℕ) : String := sorry

theorem scientific_notation_32000000 :
  scientific_notation 32000000 = "3.2 × 10^7" :=
sorry

end scientific_notation_32000000_l1415_141535


namespace three_times_first_number_minus_second_value_l1415_141577

theorem three_times_first_number_minus_second_value (x y : ℕ) 
  (h1 : x + y = 48) 
  (h2 : y = 17) : 
  3 * x - y = 76 := 
by 
  sorry

end three_times_first_number_minus_second_value_l1415_141577


namespace table_area_l1415_141558

theorem table_area (A : ℝ) (runner_total : ℝ) (cover_percentage : ℝ) (double_layer : ℝ) (triple_layer : ℝ) :
  runner_total = 208 ∧
  cover_percentage = 0.80 ∧
  double_layer = 24 ∧
  triple_layer = 22 →
  A = 260 :=
by
  sorry

end table_area_l1415_141558


namespace range_of_a_l1415_141570

noncomputable def f (x : ℝ) := (Real.log x) / x
noncomputable def g (x a : ℝ) := -Real.exp 1 * x^2 + a * x

theorem range_of_a (a : ℝ) : (∀ x1 : ℝ, ∃ x2 ∈ Set.Icc (1/3) 2, f x1 ≤ g x2 a) → 2 ≤ a :=
sorry

end range_of_a_l1415_141570


namespace quad_inequality_necessary_but_not_sufficient_l1415_141555

def quad_inequality (x : ℝ) : Prop := x^2 - x - 6 > 0
def less_than_negative_five (x : ℝ) : Prop := x < -5

theorem quad_inequality_necessary_but_not_sufficient :
  (∀ x : ℝ, less_than_negative_five x → quad_inequality x) ∧ 
  (∃ x : ℝ, quad_inequality x ∧ ¬ less_than_negative_five x) :=
by
  sorry

end quad_inequality_necessary_but_not_sufficient_l1415_141555


namespace juice_cans_count_l1415_141584

theorem juice_cans_count :
  let original_price := 12 
  let discount := 2 
  let tub_sale_price := original_price - discount 
  let tub_quantity := 2 
  let ice_cream_total := tub_quantity * tub_sale_price 
  let total_payment := 24 
  let juice_cost_per_5cans := 2 
  let remaining_amount := total_payment - ice_cream_total 
  let sets_of_juice_cans := remaining_amount / juice_cost_per_5cans 
  let cans_per_set := 5 
  2 * cans_per_set = 10 :=
by
  sorry

end juice_cans_count_l1415_141584


namespace percent_area_contained_l1415_141513

-- Define the conditions as Lean definitions
def side_length_square (s : ℝ) : ℝ := s
def width_rectangle (s : ℝ) : ℝ := 2 * s
def length_rectangle (s : ℝ) : ℝ := 3 * (width_rectangle s)

-- Define areas based on definitions
def area_square (s : ℝ) : ℝ := (side_length_square s) ^ 2
def area_rectangle (s : ℝ) : ℝ := (length_rectangle s) * (width_rectangle s)

-- The main theorem stating the percentage of the rectangle's area contained within the square
theorem percent_area_contained (s : ℝ) (h : s ≠ 0) :
  (area_square s / area_rectangle s) * 100 = 8.33 := by
  sorry

end percent_area_contained_l1415_141513


namespace arithmetic_sequence_diff_l1415_141523

theorem arithmetic_sequence_diff (a : ℕ → ℤ) (d : ℤ) 
  (h1 : a 7 = a 3 + 4 * d) :
  a 2008 - a 2000 = 8 * d :=
by
  sorry

end arithmetic_sequence_diff_l1415_141523


namespace total_fish_l1415_141525

def LillyFish : ℕ := 10
def RosyFish : ℕ := 8
def MaxFish : ℕ := 15

theorem total_fish : LillyFish + RosyFish + MaxFish = 33 := by
  sorry

end total_fish_l1415_141525


namespace family_of_sets_properties_l1415_141593

variable {X : Type}
variable {t n k : ℕ}
variable (A : Fin t → Set X)
variable (card : Set X → ℕ)
variable (h_card : ∀ (i j : Fin t), i ≠ j → card (A i ∩ A j) = k)

theorem family_of_sets_properties :
  (k = 0 → t ≤ n+1) ∧ (k ≠ 0 → t ≤ n) :=
by
  sorry

end family_of_sets_properties_l1415_141593


namespace points_on_line_y1_gt_y2_l1415_141599

theorem points_on_line_y1_gt_y2 (y1 y2 : ℝ) : 
    (∀ x y, y = -x + 3 → 
    ((x = -4 → y = y1) ∧ (x = 2 → y = y2))) → 
    y1 > y2 :=
by
  sorry

end points_on_line_y1_gt_y2_l1415_141599


namespace find_missing_number_l1415_141511

-- Define the given numbers as a list
def given_numbers : List ℕ := [3, 11, 7, 9, 15, 13, 8, 19, 17, 21, 14]

-- Define the arithmetic mean condition
def arithmetic_mean (xs : List ℕ) (mean : ℕ) : Prop :=
  (xs.sum + mean) / xs.length.succ = 12

-- Define the proof problem
theorem find_missing_number (x : ℕ) (h : arithmetic_mean given_numbers x) : x = 7 := 
sorry

end find_missing_number_l1415_141511


namespace polar_to_rectangular_coordinates_l1415_141538

noncomputable def rectangular_coordinates_from_polar (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

theorem polar_to_rectangular_coordinates :
  rectangular_coordinates_from_polar 12 (5 * Real.pi / 4) = (-6 * Real.sqrt 2, -6 * Real.sqrt 2) :=
  sorry

end polar_to_rectangular_coordinates_l1415_141538


namespace find_c_l1415_141566

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_c (c : ℝ) (h1 : f 1 = 1) (h2 : ∀ x y : ℝ, f (x + y) = f x + f y + 8 * x * y - c) (h3 : f 7 = 163) :
  c = 2 / 3 :=
sorry

end find_c_l1415_141566


namespace equal_sundays_tuesdays_l1415_141583

theorem equal_sundays_tuesdays (days_in_month : ℕ) (week_days : ℕ) (extra_days : ℕ) :
  days_in_month = 30 → week_days = 7 → extra_days = 2 → 
  ∃ n, n = 3 ∧ ∀ start_day : ℕ, start_day = 3 ∨ start_day = 4 ∨ start_day = 5 :=
by sorry

end equal_sundays_tuesdays_l1415_141583


namespace Louisa_average_speed_l1415_141510

theorem Louisa_average_speed : 
  ∀ (v : ℝ), (∀ v, (160 / v) + 3 = (280 / v)) → v = 40 :=
by
  intros v h
  sorry

end Louisa_average_speed_l1415_141510


namespace alice_coins_percentage_l1415_141588

theorem alice_coins_percentage :
  let penny := 1
  let dime := 10
  let quarter := 25
  let half_dollar := 50
  let total_cents := penny + dime + quarter + half_dollar
  (total_cents / 100) * 100 = 86 :=
by
  sorry

end alice_coins_percentage_l1415_141588


namespace minimum_value_l1415_141573

theorem minimum_value (a_n : ℕ → ℤ) (h : ∀ n, a_n n = n^2 - 8 * n + 5) : ∃ n, a_n n = -11 :=
by
  sorry

end minimum_value_l1415_141573


namespace minimize_prod_time_l1415_141571

noncomputable def shortest_production_time
  (items : ℕ) 
  (workers : ℕ) 
  (shaping_time : ℕ) 
  (firing_time : ℕ) : ℕ := by
  sorry

-- The main theorem statement
theorem minimize_prod_time
  (items : ℕ := 75)
  (workers : ℕ := 13)
  (shaping_time : ℕ := 15)
  (drying_time : ℕ := 10)
  (firing_time : ℕ := 30)
  (optimal_time : ℕ := 325) :
  shortest_production_time items workers shaping_time firing_time = optimal_time := by
  sorry

end minimize_prod_time_l1415_141571


namespace simplify_expression_l1415_141580

variable (a b : ℤ)

theorem simplify_expression : 
  (50 * a + 130 * b) + (21 * a + 64 * b) - (30 * a + 115 * b) - 2 * (10 * a - 25 * b) = 21 * a + 129 * b := 
by
  sorry

end simplify_expression_l1415_141580


namespace integer_not_natural_l1415_141540

theorem integer_not_natural (n : ℕ) (a : ℝ) (b : ℝ) (x y z : ℝ) 
  (h₁ : x = (1 + a) ^ n) 
  (h₂ : y = (1 - a) ^ n) 
  (h₃ : z = a): 
  ∃ k : ℤ, (x - y) / z = ↑k ∧ (k < 0 ∨ k ≠ 0) :=
by 
  sorry

end integer_not_natural_l1415_141540


namespace problem_1_problem_2_l1415_141581

noncomputable def f (x : ℝ) : ℝ := |2 * x + 1| + |3 * x - 2|

theorem problem_1 {a b : ℝ} (h : ∀ x, f x ≤ 5 → -4 * a / 5 ≤ x ∧ x ≤ 3 * b / 5) : 
  a = 1 ∧ b = 2 :=
sorry

theorem problem_2 {a b m : ℝ} (h1 : a = 1) (h2 : b = 2) (h3 : ∀ x, |x - a| + |x + b| ≥ m^2 - 3 * m + 5) :
  ∃ m, m = 2 :=
sorry

end problem_1_problem_2_l1415_141581


namespace length_of_bridge_l1415_141585

theorem length_of_bridge 
  (train_length : ℝ)
  (train_speed_kmh : ℝ)
  (time_to_pass_bridge : ℝ) 
  (train_length_eq : train_length = 400)
  (train_speed_kmh_eq : train_speed_kmh = 60) 
  (time_to_pass_bridge_eq : time_to_pass_bridge = 72)
  : ∃ (bridge_length : ℝ), bridge_length = 800.24 := 
by
  sorry

end length_of_bridge_l1415_141585
