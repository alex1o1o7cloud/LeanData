import Mathlib

namespace symmetry_implies_values_l1838_183837

/-- Two points are symmetric with respect to the y-axis if their x-coordinates are opposite and y-coordinates are equal -/
def symmetric_wrt_y_axis (p1 p2 : ℝ × ℝ) : Prop :=
  p1.1 = -p2.1 ∧ p1.2 = p2.2

theorem symmetry_implies_values :
  ∀ (a b : ℝ), symmetric_wrt_y_axis (a, 1) (5, b) → a = -5 ∧ b = 1 := by
  sorry

end symmetry_implies_values_l1838_183837


namespace sum_of_first_5n_integers_l1838_183855

theorem sum_of_first_5n_integers (n : ℕ) : 
  (3 * n * (3 * n + 1)) / 2 = (n * (n + 1)) / 2 + 210 → 
  (5 * n * (5 * n + 1)) / 2 = 630 := by
  sorry

end sum_of_first_5n_integers_l1838_183855


namespace range_of_m_l1838_183816

def A : Set ℝ := {y | ∃ x ∈ Set.Icc (3/4 : ℝ) 2, y = x^2 - (3/2)*x + 1}

def B (m : ℝ) : Set ℝ := {x | x + m^2 ≥ 1}

theorem range_of_m :
  {m : ℝ | A ⊆ B m} = Set.Iic (-3/4) ∪ Set.Ici (3/4) :=
sorry

end range_of_m_l1838_183816


namespace first_discount_percentage_l1838_183897

/-- Proves that the first discount is 15% given the initial price, final price, and second discount rate -/
theorem first_discount_percentage (initial_price final_price : ℝ) (second_discount : ℝ) :
  initial_price = 400 →
  final_price = 323 →
  second_discount = 0.05 →
  ∃ (first_discount : ℝ),
    first_discount = 0.15 ∧
    final_price = initial_price * (1 - first_discount) * (1 - second_discount) :=
by sorry

end first_discount_percentage_l1838_183897


namespace late_train_speed_l1838_183826

/-- Proves that given a journey of 15 km, if a train traveling at 100 kmph reaches the destination on time,
    and a train traveling at speed v kmph reaches the destination 15 minutes late, then v = 37.5 kmph. -/
theorem late_train_speed (journey_length : ℝ) (on_time_speed : ℝ) (late_time_diff : ℝ) (v : ℝ) :
  journey_length = 15 →
  on_time_speed = 100 →
  late_time_diff = 0.25 →
  journey_length / on_time_speed + late_time_diff = journey_length / v →
  v = 37.5 := by
  sorry

#check late_train_speed

end late_train_speed_l1838_183826


namespace shaded_area_theorem_l1838_183830

/-- The fraction of shaded area in each subdivision -/
def shaded_fraction : ℚ := 7 / 16

/-- The ratio of area of each subdivision to the whole square -/
def subdivision_ratio : ℚ := 1 / 16

/-- The total shaded fraction of the square -/
def total_shaded_fraction : ℚ := 7 / 15

theorem shaded_area_theorem :
  (shaded_fraction * (1 - subdivision_ratio)⁻¹ : ℚ) = total_shaded_fraction := by
  sorry

end shaded_area_theorem_l1838_183830


namespace digit_sum_congruence_part1_digit_sum_congruence_part2_l1838_183860

/-- S_r(n) is the sum of the digits of n in base r -/
def S_r (r : ℕ) (n : ℕ) : ℕ :=
  sorry

theorem digit_sum_congruence_part1 :
  ∀ r : ℕ, r > 2 → ∃ p : ℕ, Nat.Prime p ∧ ∀ n : ℕ, n > 0 → S_r r n ≡ n [MOD p] :=
sorry

theorem digit_sum_congruence_part2 :
  ∀ r : ℕ, r > 1 → ∀ p : ℕ, Nat.Prime p →
  ∃ f : ℕ → ℕ, Function.Injective f ∧ ∀ k : ℕ, S_r r (f k) ≡ f k [MOD p] :=
sorry

end digit_sum_congruence_part1_digit_sum_congruence_part2_l1838_183860


namespace initial_bacteria_count_l1838_183875

def bacteria_growth (initial_count : ℕ) (time : ℕ) : ℕ :=
  initial_count * 4^(time / 30)

theorem initial_bacteria_count :
  ∃ (initial_count : ℕ),
    bacteria_growth initial_count 360 = 262144 ∧
    initial_count = 1 :=
by sorry

end initial_bacteria_count_l1838_183875


namespace unicorns_total_games_l1838_183877

theorem unicorns_total_games : 
  ∀ (initial_games initial_wins district_wins district_losses : ℕ),
    initial_wins = initial_games / 2 →
    district_wins = 8 →
    district_losses = 3 →
    (initial_wins + district_wins) * 100 = 55 * (initial_games + district_wins + district_losses) →
    initial_games + district_wins + district_losses = 50 := by
  sorry

end unicorns_total_games_l1838_183877


namespace total_dividend_is_825_l1838_183865

/-- Represents the investment scenario with two types of shares --/
structure Investment where
  total_amount : ℕ
  type_a_face_value : ℕ
  type_b_face_value : ℕ
  type_a_premium : ℚ
  type_b_discount : ℚ
  type_a_dividend_rate : ℚ
  type_b_dividend_rate : ℚ

/-- Calculates the total dividend received from the investment --/
def calculate_total_dividend (inv : Investment) : ℚ :=
  sorry

/-- Theorem stating that the total dividend received is 825 --/
theorem total_dividend_is_825 :
  let inv : Investment := {
    total_amount := 14400,
    type_a_face_value := 100,
    type_b_face_value := 100,
    type_a_premium := 1/5,
    type_b_discount := 1/10,
    type_a_dividend_rate := 7/100,
    type_b_dividend_rate := 1/20
  }
  calculate_total_dividend inv = 825 := by sorry

end total_dividend_is_825_l1838_183865


namespace angel_score_is_11_l1838_183839

-- Define the scores for each player
def beth_score : ℕ := 12
def jan_score : ℕ := 10
def judy_score : ℕ := 8

-- Define the total score of the first team
def first_team_score : ℕ := beth_score + jan_score

-- Define the difference between the first and second team scores
def score_difference : ℕ := 3

-- Define Angel's score as a variable
def angel_score : ℕ := sorry

-- Theorem to prove
theorem angel_score_is_11 :
  angel_score = 11 :=
by
  sorry

end angel_score_is_11_l1838_183839


namespace graph_horizontal_shift_l1838_183841

-- Define a generic function g
variable (g : ℝ → ℝ)

-- Define a point (x, y) on the graph of y = g(x)
variable (x y : ℝ)

-- Define the horizontal shift
def h : ℝ := 3

-- Theorem statement
theorem graph_horizontal_shift :
  y = g x ↔ y = g (x - h) :=
sorry

end graph_horizontal_shift_l1838_183841


namespace natural_number_pairs_l1838_183891

theorem natural_number_pairs : ∀ (a b : ℕ+), 
  (∃ (k l : ℕ+), (a + 1 : ℕ) = k * b ∧ (b + 1 : ℕ) = l * a) →
  ((a = 1 ∧ b = 1) ∨ 
   (a = 1 ∧ b = 2) ∨ 
   (a = 2 ∧ b = 3) ∨ 
   (a = 2 ∧ b = 1) ∨ 
   (a = 3 ∧ b = 2)) :=
by sorry


end natural_number_pairs_l1838_183891


namespace problem_1_problem_2_l1838_183893

-- Problem 1
theorem problem_1 : Real.sqrt 3 * Real.sqrt 6 - (Real.sqrt (1/2) - Real.sqrt 8) = (9 * Real.sqrt 2) / 2 := by sorry

-- Problem 2
theorem problem_2 (x : ℝ) (h : x = Real.sqrt 5) : 
  (1 + 1/x) / ((x^2 + x) / x) = Real.sqrt 5 / 5 := by sorry

end problem_1_problem_2_l1838_183893


namespace inequality_solution_set_l1838_183849

theorem inequality_solution_set (x : ℝ) : (x^2 + x) / (2*x - 1) ≤ 1 ↔ x < 1/2 := by
  sorry

end inequality_solution_set_l1838_183849


namespace imon_entanglement_reduction_l1838_183883

/-- Represents a graph of imons and their entanglements -/
structure ImonGraph where
  vertices : Set ℕ
  edges : Set (ℕ × ℕ)

/-- Operation 1: Remove a vertex with odd degree -/
def removeOddDegreeVertex (G : ImonGraph) (v : ℕ) : ImonGraph :=
  sorry

/-- Operation 2: Duplicate the graph and connect each vertex to its duplicate -/
def duplicateGraph (G : ImonGraph) : ImonGraph :=
  sorry

/-- Predicate to check if a graph has no edges -/
def hasNoEdges (G : ImonGraph) : Prop :=
  G.edges = ∅

/-- Main theorem: There exists a sequence of operations to reduce any ImonGraph to one with no edges -/
theorem imon_entanglement_reduction (G : ImonGraph) :
  ∃ (seq : List (ImonGraph → ImonGraph)), hasNoEdges (seq.foldl (λ g f => f g) G) :=
  sorry

end imon_entanglement_reduction_l1838_183883


namespace problem_solution_l1838_183851

theorem problem_solution (x y : ℝ) (hx : x = 3) (hy : y = 4) :
  (x^5 + 3*y^3) / 8 = 54.375 := by
  sorry

end problem_solution_l1838_183851


namespace train_length_l1838_183894

theorem train_length (platform_time : ℝ) (pole_time : ℝ) (platform_length : ℝ)
  (h1 : platform_time = 39)
  (h2 : pole_time = 18)
  (h3 : platform_length = 350) :
  ∃ (train_length : ℝ), train_length = 300 ∧
    train_length / pole_time = (train_length + platform_length) / platform_time :=
by sorry

end train_length_l1838_183894


namespace linear_function_not_in_first_quadrant_implies_negative_k_and_b_l1838_183850

/-- A linear function that does not pass through the first quadrant -/
structure LinearFunctionNotInFirstQuadrant where
  k : ℝ
  b : ℝ
  not_in_first_quadrant : ∀ x y : ℝ, y = k * x + b → ¬(x > 0 ∧ y > 0)

/-- Theorem: If a linear function y = kx + b does not pass through the first quadrant, then k < 0 and b < 0 -/
theorem linear_function_not_in_first_quadrant_implies_negative_k_and_b 
  (f : LinearFunctionNotInFirstQuadrant) : f.k < 0 ∧ f.b < 0 := by
  sorry


end linear_function_not_in_first_quadrant_implies_negative_k_and_b_l1838_183850


namespace road_repair_theorem_l1838_183800

/-- The number of persons in the first group -/
def first_group : ℕ := 39

/-- The number of days for the first group to complete the work -/
def days_first : ℕ := 24

/-- The number of hours per day for the first group -/
def hours_first : ℕ := 5

/-- The number of days for the second group to complete the work -/
def days_second : ℕ := 26

/-- The number of hours per day for the second group -/
def hours_second : ℕ := 6

/-- The total man-hours required to complete the work -/
def total_man_hours : ℕ := first_group * days_first * hours_first

/-- The number of persons in the second group -/
def second_group : ℕ := total_man_hours / (days_second * hours_second)

theorem road_repair_theorem : second_group = 30 := by
  sorry

end road_repair_theorem_l1838_183800


namespace square_area_ratio_l1838_183862

theorem square_area_ratio (side_C side_D : ℝ) (h1 : side_C = 48) (h2 : side_D = 60) :
  (side_C ^ 2) / (side_D ^ 2) = 16 / 25 := by
  sorry

end square_area_ratio_l1838_183862


namespace geometric_sequence_first_term_l1838_183821

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a : ℕ → ℚ) : Prop :=
  ∃ r : ℚ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_first_term
  (a : ℕ → ℚ)
  (h_geometric : IsGeometricSequence a)
  (h_term2 : a 2 = 12)
  (h_term3 : a 3 = 24)
  (h_term6 : a 6 = 384) :
  a 1 = 3/4 := by
sorry

end geometric_sequence_first_term_l1838_183821


namespace first_train_length_l1838_183842

/-- Given two trains with specific speeds, lengths, and crossing time, prove the length of the first train. -/
theorem first_train_length
  (v1 : ℝ) -- Speed of first train
  (v2 : ℝ) -- Speed of second train
  (l2 : ℝ) -- Length of second train
  (d : ℝ)  -- Distance between trains
  (t : ℝ)  -- Time for second train to cross first train
  (h1 : v1 = 10)
  (h2 : v2 = 15)
  (h3 : l2 = 150)
  (h4 : d = 50)
  (h5 : t = 60) :
  ∃ l1 : ℝ, l1 = 100 ∧ l1 + l2 + d = (v2 - v1) * t := by
  sorry

#check first_train_length

end first_train_length_l1838_183842


namespace cube_volume_in_pyramid_l1838_183804

/-- A pyramid with a square base and equilateral triangle lateral faces -/
structure Pyramid :=
  (base_side : ℝ)
  (has_square_base : base_side > 0)
  (has_equilateral_lateral_faces : True)

/-- A cube placed inside the pyramid -/
structure InsideCube :=
  (side_length : ℝ)
  (base_on_pyramid_base : True)
  (top_vertices_touch_midpoints : True)

/-- The theorem stating the volume of the cube inside the pyramid -/
theorem cube_volume_in_pyramid (p : Pyramid) (c : InsideCube) 
  (h1 : p.base_side = 2) : c.side_length ^ 3 = 3 * Real.sqrt 6 / 4 := by
  sorry

#check cube_volume_in_pyramid

end cube_volume_in_pyramid_l1838_183804


namespace geometric_sequence_property_l1838_183828

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Theorem: In a geometric sequence, if a₅ = 2, then a₁ * a₉ = 4 -/
theorem geometric_sequence_property (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a) (h_a5 : a 5 = 2) : a 1 * a 9 = 4 := by
  sorry

end geometric_sequence_property_l1838_183828


namespace complex_equation_solution_l1838_183899

theorem complex_equation_solution (z : ℂ) : z = Complex.I * (2 - z) → z = 1 + Complex.I := by
  sorry

end complex_equation_solution_l1838_183899


namespace course_selection_combinations_l1838_183814

theorem course_selection_combinations :
  let total_courses : ℕ := 7
  let required_courses : ℕ := 2
  let math_courses : ℕ := 2
  let program_size : ℕ := 5
  let remaining_courses : ℕ := total_courses - required_courses
  let remaining_selections : ℕ := program_size - required_courses

  (Nat.choose remaining_courses remaining_selections) -
  (Nat.choose (remaining_courses - math_courses) remaining_selections) = 9 :=
by sorry

end course_selection_combinations_l1838_183814


namespace baseball_league_games_played_l1838_183887

/-- Represents a baseball league with the given parameters -/
structure BaseballLeague where
  num_teams : ℕ
  games_per_week : ℕ
  season_length_months : ℕ
  
/-- Calculates the total number of games played in a season -/
def total_games_played (league : BaseballLeague) : ℕ :=
  (league.num_teams * league.games_per_week * league.season_length_months * 4) / 2

/-- Theorem stating the total number of games played in the given league configuration -/
theorem baseball_league_games_played :
  ∃ (league : BaseballLeague),
    league.num_teams = 10 ∧
    league.games_per_week = 5 ∧
    league.season_length_months = 6 ∧
    total_games_played league = 600 := by
  sorry

end baseball_league_games_played_l1838_183887


namespace cos_equality_proof_l1838_183836

theorem cos_equality_proof (n : ℤ) : 
  0 ≤ n ∧ n ≤ 360 → (Real.cos (n * π / 180) = Real.cos (123 * π / 180) ↔ n = 123 ∨ n = 237) := by
  sorry

end cos_equality_proof_l1838_183836


namespace specific_quadrilateral_perimeter_l1838_183819

/-- A convex quadrilateral with a point inside it -/
structure ConvexQuadrilateral where
  W : ℝ × ℝ
  X : ℝ × ℝ
  Y : ℝ × ℝ
  Z : ℝ × ℝ
  Q : ℝ × ℝ
  area : ℝ
  wq : ℝ
  xq : ℝ
  yq : ℝ
  zq : ℝ
  convex : Bool
  inside : Bool

/-- The perimeter of a quadrilateral -/
def perimeter (quad : ConvexQuadrilateral) : ℝ :=
  sorry

/-- Theorem stating the perimeter of the specific quadrilateral -/
theorem specific_quadrilateral_perimeter :
  ∀ (quad : ConvexQuadrilateral),
    quad.area = 2500 ∧
    quad.wq = 30 ∧
    quad.xq = 40 ∧
    quad.yq = 35 ∧
    quad.zq = 50 ∧
    quad.convex = true ∧
    quad.inside = true →
    perimeter quad = 155 + 10 * Real.sqrt 34 + 5 * Real.sqrt 113 :=
  sorry

end specific_quadrilateral_perimeter_l1838_183819


namespace value_of_x_l1838_183874

theorem value_of_x : ∀ (x a b c : ℤ),
  x = a + 7 →
  a = b + 12 →
  b = c + 25 →
  c = 95 →
  x = 139 := by
  sorry

end value_of_x_l1838_183874


namespace least_denominator_for_0711_l1838_183854

theorem least_denominator_for_0711 : 
  ∃ (m : ℕ+), (711 : ℚ)/1000 ≤ m/45 ∧ m/45 < (712 : ℚ)/1000 ∧ 
  ∀ (n : ℕ+) (k : ℕ+), n < 45 → ¬((711 : ℚ)/1000 ≤ k/n ∧ k/n < (712 : ℚ)/1000) :=
by sorry

end least_denominator_for_0711_l1838_183854


namespace green_shirts_count_l1838_183820

-- Define the total number of shirts
def total_shirts : ℕ := 23

-- Define the number of blue shirts
def blue_shirts : ℕ := 6

-- Theorem: The number of green shirts is 17
theorem green_shirts_count : total_shirts - blue_shirts = 17 := by
  sorry

end green_shirts_count_l1838_183820


namespace max_marks_calculation_l1838_183853

/-- The maximum marks in an exam where:
  - The passing mark is 35% of the maximum marks
  - A student got 185 marks
  - The student failed by 25 marks
-/
theorem max_marks_calculation : ∃ (M : ℝ), 
  (0.35 * M = 185 + 25) ∧ 
  (M = 600) := by
  sorry

end max_marks_calculation_l1838_183853


namespace square_of_sum_equals_81_l1838_183856

theorem square_of_sum_equals_81 (x : ℝ) (h : Real.sqrt (x + 3) = 3) : 
  (x + 3)^2 = 81 := by
  sorry

end square_of_sum_equals_81_l1838_183856


namespace esperanza_savings_l1838_183861

theorem esperanza_savings :
  let rent : ℕ := 600
  let food_cost : ℕ := (3 * rent) / 5
  let mortgage : ℕ := 3 * food_cost
  let gross_salary : ℕ := 4840
  let expenses : ℕ := rent + food_cost + mortgage
  let pre_tax_savings : ℕ := gross_salary - expenses
  let taxes : ℕ := (2 * pre_tax_savings) / 5
  let savings : ℕ := pre_tax_savings - taxes
  savings = 1680 := by sorry

end esperanza_savings_l1838_183861


namespace quadratic_inequality_solution_set_l1838_183879

theorem quadratic_inequality_solution_set :
  {x : ℝ | 2*x + 3 - x^2 > 0} = {x : ℝ | -1 < x ∧ x < 3} := by
  sorry

end quadratic_inequality_solution_set_l1838_183879


namespace zoo_elephants_l1838_183843

theorem zoo_elephants (giraffes : ℕ) (penguins : ℕ) (total : ℕ) (elephants : ℕ) : 
  giraffes = 5 →
  penguins = 2 * giraffes →
  penguins = (20 : ℚ) / 100 * total →
  elephants = (4 : ℚ) / 100 * total →
  elephants = 2 :=
by sorry

end zoo_elephants_l1838_183843


namespace cordelia_hair_dyeing_l1838_183873

/-- Cordelia's hair dyeing problem -/
theorem cordelia_hair_dyeing (total_time bleach_time dye_time : ℝ) : 
  total_time = 9 ∧ dye_time = 2 * bleach_time ∧ total_time = bleach_time + dye_time → 
  bleach_time = 3 := by
  sorry

end cordelia_hair_dyeing_l1838_183873


namespace triangle_side_length_l1838_183829

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  a^2 - c^2 = 2*b →
  Real.sin B = 4 * Real.cos A * Real.sin C →
  b = 4 := by
  sorry

end triangle_side_length_l1838_183829


namespace trajectory_of_moving_circle_center_l1838_183888

/-- A moving circle passes through a fixed point F(2, 0) and is tangent to the line x = -2.
    The trajectory of its center C is a parabola. -/
theorem trajectory_of_moving_circle_center (C : ℝ × ℝ) : 
  (∃ (r : ℝ), (C.1 - 2)^2 + C.2^2 = r^2 ∧ (C.1 + 2)^2 + C.2^2 = r^2) →
  C.2^2 = 8 * C.1 := by
  sorry


end trajectory_of_moving_circle_center_l1838_183888


namespace solve_for_x_l1838_183823

theorem solve_for_x : ∃ x : ℤ, x + 1315 + 9211 - 1569 = 11901 ∧ x = 2944 := by
  sorry

end solve_for_x_l1838_183823


namespace f_of_4_equals_82_l1838_183818

-- Define a monotonic function f
def monotonic_function (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y ∨ f y ≤ f x

-- State the theorem
theorem f_of_4_equals_82
  (f : ℝ → ℝ)
  (h_monotonic : monotonic_function f)
  (h_condition : ∀ x : ℝ, f (f x - 3^x) = 4) :
  f 4 = 82 := by
  sorry

end f_of_4_equals_82_l1838_183818


namespace negation_equivalence_l1838_183834

def exactly_one_even (a b c : ℕ) : Prop :=
  (Even a ∧ Odd b ∧ Odd c) ∨ (Odd a ∧ Even b ∧ Odd c) ∨ (Odd a ∧ Odd b ∧ Even c)

def negation_statement (a b c : ℕ) : Prop :=
  (Even a ∧ Even b) ∨ (Even a ∧ Even c) ∨ (Even b ∧ Even c) ∨ (Odd a ∧ Odd b ∧ Odd c)

theorem negation_equivalence (a b c : ℕ) :
  ¬(exactly_one_even a b c) ↔ negation_statement a b c :=
sorry

end negation_equivalence_l1838_183834


namespace combination_equality_implies_five_l1838_183878

theorem combination_equality_implies_five (n : ℕ+) : 
  Nat.choose n 2 = Nat.choose (n - 1) 2 + Nat.choose (n - 1) 3 → n = 5 := by
  sorry

end combination_equality_implies_five_l1838_183878


namespace parabola_focus_and_directrix_l1838_183812

/-- Represents a parabola in the form y² = -4px where p is the focal length -/
structure Parabola where
  p : ℝ

/-- The focus of a parabola -/
def Parabola.focus (par : Parabola) : ℝ × ℝ := (-par.p, 0)

/-- The x-coordinate of the directrix of a parabola -/
def Parabola.directrix (par : Parabola) : ℝ := par.p

theorem parabola_focus_and_directrix (par : Parabola) 
  (h : par.p = 2) : 
  (par.focus = (-2, 0)) ∧ (par.directrix = 2) := by
  sorry

#check parabola_focus_and_directrix

end parabola_focus_and_directrix_l1838_183812


namespace least_cans_required_l1838_183802

theorem least_cans_required (maaza pepsi sprite cola fanta : ℕ) 
  (h_maaza : maaza = 200)
  (h_pepsi : pepsi = 288)
  (h_sprite : sprite = 736)
  (h_cola : cola = 450)
  (h_fanta : fanta = 625) :
  let gcd := Nat.gcd (Nat.gcd (Nat.gcd (Nat.gcd maaza pepsi) sprite) cola) fanta
  gcd = 1 ∧ maaza / gcd + pepsi / gcd + sprite / gcd + cola / gcd + fanta / gcd = 2299 :=
by sorry

end least_cans_required_l1838_183802


namespace expression_evaluation_l1838_183872

theorem expression_evaluation (x : ℝ) : 
  x * (x * (x * (3 - x) - 5) + 12) + 2 = -x^4 + 3*x^3 - 5*x^2 + 12*x + 2 := by
  sorry

end expression_evaluation_l1838_183872


namespace greatest_integer_less_than_neg_nineteen_fifths_l1838_183833

theorem greatest_integer_less_than_neg_nineteen_fifths :
  Int.floor (-19 / 5 : ℚ) = -4 := by sorry

end greatest_integer_less_than_neg_nineteen_fifths_l1838_183833


namespace boat_speed_l1838_183844

/-- The speed of a boat in still water, given its downstream and upstream speeds -/
theorem boat_speed (downstream upstream : ℝ) (h1 : downstream = 15) (h2 : upstream = 7) :
  (downstream + upstream) / 2 = 11 :=
by
  sorry

#check boat_speed

end boat_speed_l1838_183844


namespace countable_planar_graph_coloring_l1838_183884

-- Define a type for colors
inductive Color
| blue
| red
| green

-- Define a type for graphs
structure Graph (α : Type) where
  vertices : Set α
  edges : Set (α × α)

-- Define what it means for a graph to be planar
def isPlanar {α : Type} (G : Graph α) : Prop := sorry

-- Define what it means for a graph to be countable
def isCountable {α : Type} (G : Graph α) : Prop := sorry

-- Define what it means for a cycle to be odd
def isOddCycle {α : Type} (G : Graph α) (cycle : List α) : Prop := sorry

-- Define what it means for a coloring to be valid (no odd monochromatic cycles)
def isValidColoring {α : Type} (G : Graph α) (coloring : α → Color) : Prop :=
  ∀ cycle, isOddCycle G cycle → ∃ v ∈ cycle, ∃ w ∈ cycle, coloring v ≠ coloring w

-- The main theorem
theorem countable_planar_graph_coloring 
  {α : Type} (G : Graph α) 
  (h_planar : isPlanar G) 
  (h_countable : isCountable G) 
  (h_finite : ∀ (H : Graph α), isPlanar H → (Finite H.vertices) → 
    ∃ coloring : α → Color, isValidColoring H coloring) :
  ∃ coloring : α → Color, isValidColoring G coloring := by
  sorry

end countable_planar_graph_coloring_l1838_183884


namespace fraction_simplification_l1838_183824

/-- The number of quarters Sarah has -/
def total_quarters : ℕ := 30

/-- The number of states that joined the union from 1790 to 1799 -/
def states_1790_1799 : ℕ := 8

/-- The fraction of Sarah's quarters representing states that joined from 1790 to 1799 -/
def fraction_1790_1799 : ℚ := states_1790_1799 / total_quarters

theorem fraction_simplification :
  fraction_1790_1799 = 4 / 15 := by sorry

end fraction_simplification_l1838_183824


namespace two_corners_are_diagonal_endpoints_l1838_183866

/-- A structure representing a checkered rectangle divided into dominoes with diagonals -/
structure CheckeredRectangle where
  rows : ℕ
  cols : ℕ
  dominoes : List (Nat × Nat × Nat × Nat)
  diagonals : List (Nat × Nat × Nat × Nat)

/-- Predicate to check if a point is a corner of the rectangle -/
def is_corner (r : CheckeredRectangle) (x y : ℕ) : Prop :=
  (x = 0 ∨ x = r.cols - 1) ∧ (y = 0 ∨ y = r.rows - 1)

/-- Predicate to check if a point is an endpoint of any diagonal -/
def is_diagonal_endpoint (r : CheckeredRectangle) (x y : ℕ) : Prop :=
  ∃ (x1 y1 x2 y2 : ℕ), (x1, y1, x2, y2) ∈ r.diagonals ∧ ((x = x1 ∧ y = y1) ∨ (x = x2 ∧ y = y2))

/-- The main theorem stating that exactly two corners are diagonal endpoints -/
theorem two_corners_are_diagonal_endpoints (r : CheckeredRectangle) 
  (h1 : ∀ (x1 y1 x2 y2 : ℕ), (x1, y1, x2, y2) ∈ r.dominoes → 
    ((x2 = x1 + 1 ∧ y2 = y1) ∨ (x2 = x1 ∧ y2 = y1 + 1)))
  (h2 : ∀ (x1 y1 x2 y2 : ℕ), (x1, y1, x2, y2) ∈ r.diagonals → 
    ∃ (x3 y3 x4 y4 : ℕ), (x3, y3, x4, y4) ∈ r.dominoes ∧ 
    ((x1 = x3 ∧ y1 = y3 ∧ x2 = x4 ∧ y2 = y4) ∨ (x1 = x4 ∧ y1 = y4 ∧ x2 = x3 ∧ y2 = y3)))
  (h3 : ∀ (x1 y1 x2 y2 x3 y3 x4 y4 : ℕ), 
    (x1, y1, x2, y2) ∈ r.diagonals → (x3, y3, x4, y4) ∈ r.diagonals → 
    (x1 ≠ x3 ∨ y1 ≠ y3) ∧ (x1 ≠ x4 ∨ y1 ≠ y4) ∧ (x2 ≠ x3 ∨ y2 ≠ y3) ∧ (x2 ≠ x4 ∨ y2 ≠ y4)) :
  ∃! (c1 c2 : ℕ × ℕ), 
    c1 ≠ c2 ∧ 
    is_corner r c1.1 c1.2 ∧ 
    is_corner r c2.1 c2.2 ∧ 
    is_diagonal_endpoint r c1.1 c1.2 ∧ 
    is_diagonal_endpoint r c2.1 c2.2 ∧ 
    (∀ (x y : ℕ), is_corner r x y → (x, y) ≠ c1 → (x, y) ≠ c2 → ¬is_diagonal_endpoint r x y) :=
sorry

end two_corners_are_diagonal_endpoints_l1838_183866


namespace fraction_decomposition_l1838_183871

theorem fraction_decomposition : 
  ∃ (A B : ℚ), A = -12/11 ∧ B = 113/11 ∧
  ∀ (x : ℚ), x ≠ 1 ∧ x ≠ -8/3 →
  (7*x - 19) / (3*x^2 + 5*x - 8) = A / (x - 1) + B / (3*x + 8) := by
  sorry

end fraction_decomposition_l1838_183871


namespace ellipse_sum_coordinates_and_axes_specific_ellipse_sum_l1838_183889

/-- Definition of an ellipse with given center and axis lengths -/
structure Ellipse where
  center : ℝ × ℝ
  semi_major_axis : ℝ
  semi_minor_axis : ℝ

/-- Theorem: Sum of center coordinates and axis lengths for a specific ellipse -/
theorem ellipse_sum_coordinates_and_axes (e : Ellipse) 
  (h1 : e.center = (-3, 4)) 
  (h2 : e.semi_major_axis = 7) 
  (h3 : e.semi_minor_axis = 2) : 
  e.center.1 + e.center.2 + e.semi_major_axis + e.semi_minor_axis = 10 := by
  sorry

/-- Main theorem to be proved -/
theorem specific_ellipse_sum : 
  ∃ (e : Ellipse), e.center = (-3, 4) ∧ e.semi_major_axis = 7 ∧ e.semi_minor_axis = 2 ∧
  e.center.1 + e.center.2 + e.semi_major_axis + e.semi_minor_axis = 10 := by
  sorry

end ellipse_sum_coordinates_and_axes_specific_ellipse_sum_l1838_183889


namespace tangent_point_on_reciprocal_curve_l1838_183896

/-- Prove that the point of tangency on y = 1/x, where the tangent line passes through (0,2), is (1,1) -/
theorem tangent_point_on_reciprocal_curve :
  ∀ m n : ℝ,
  (n = 1 / m) →                         -- Point (m,n) is on the curve y = 1/x
  (2 - n) / m = -1 / (m^2) →            -- Tangent line passes through (0,2) with slope -1/m^2
  (m = 1 ∧ n = 1) :=                    -- The point of tangency is (1,1)
by sorry

end tangent_point_on_reciprocal_curve_l1838_183896


namespace track_circumference_l1838_183845

/-- Represents the circular track and the movement of A and B -/
structure CircularTrack where
  /-- The circumference of the track in yards -/
  circumference : ℝ
  /-- The distance B has traveled when they first meet -/
  first_meeting : ℝ
  /-- The distance A is shy of completing a full lap at the second meeting -/
  second_meeting : ℝ

/-- Theorem stating that under the given conditions, the track's circumference is 720 yards -/
theorem track_circumference (track : CircularTrack) 
  (h1 : track.first_meeting = 150)
  (h2 : track.second_meeting = 90)
  (h3 : track.circumference > 0) :
  track.circumference = 720 := by
  sorry

#check track_circumference

end track_circumference_l1838_183845


namespace vertical_shift_equation_line_shift_theorem_l1838_183898

/-- Represents a linear function of the form y = mx + b -/
structure LinearFunction where
  slope : ℝ
  intercept : ℝ

/-- Applies a vertical shift to a linear function -/
def verticalShift (f : LinearFunction) (shift : ℝ) : LinearFunction :=
  { slope := f.slope, intercept := f.intercept + shift }

theorem vertical_shift_equation (m : ℝ) (shift : ℝ) :
  let original := LinearFunction.mk m 0
  let shifted := verticalShift original shift
  shifted = LinearFunction.mk m shift := by sorry

/-- The main theorem proving that shifting y = -5x upwards by 2 units results in y = -5x + 2 -/
theorem line_shift_theorem :
  let original := LinearFunction.mk (-5) 0
  let shifted := verticalShift original 2
  shifted = LinearFunction.mk (-5) 2 := by sorry

end vertical_shift_equation_line_shift_theorem_l1838_183898


namespace inequality_system_solution_exists_l1838_183848

theorem inequality_system_solution_exists : ∃ (x y z t : ℝ), 
  (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0 ∨ t ≠ 0) ∧
  (abs x < abs (y - z + t)) ∧
  (abs y < abs (x - z + t)) ∧
  (abs z < abs (x - y + t)) ∧
  (abs t < abs (x - y + z)) := by
  sorry

end inequality_system_solution_exists_l1838_183848


namespace tonya_large_lemonade_sales_l1838_183806

/-- Represents the sales data for Tonya's lemonade stand. -/
structure LemonadeSales where
  small_price : ℕ
  medium_price : ℕ
  large_price : ℕ
  total_revenue : ℕ
  small_revenue : ℕ
  medium_revenue : ℕ

/-- Calculates the number of large lemonade cups sold. -/
def large_cups_sold (sales : LemonadeSales) : ℕ :=
  (sales.total_revenue - sales.small_revenue - sales.medium_revenue) / sales.large_price

/-- Theorem stating that Tonya sold 5 cups of large lemonade. -/
theorem tonya_large_lemonade_sales :
  let sales : LemonadeSales := {
    small_price := 1,
    medium_price := 2,
    large_price := 3,
    total_revenue := 50,
    small_revenue := 11,
    medium_revenue := 24
  }
  large_cups_sold sales = 5 := by sorry

end tonya_large_lemonade_sales_l1838_183806


namespace total_goals_is_sixteen_l1838_183895

def bruce_goals : ℕ := 4
def michael_goals_multiplier : ℕ := 3

def total_goals : ℕ := bruce_goals + michael_goals_multiplier * bruce_goals

theorem total_goals_is_sixteen : total_goals = 16 := by
  sorry

end total_goals_is_sixteen_l1838_183895


namespace total_cost_calculation_l1838_183822

def dog_cost : ℕ := 60
def cat_cost : ℕ := 40
def num_dogs : ℕ := 20
def num_cats : ℕ := 60

theorem total_cost_calculation : 
  dog_cost * num_dogs + cat_cost * num_cats = 3600 := by
  sorry

end total_cost_calculation_l1838_183822


namespace inequality_sum_l1838_183840

theorem inequality_sum (a b c d : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a > b) (h4 : c > d) : 
  a + c > b + d := by
  sorry

end inequality_sum_l1838_183840


namespace steps_correct_l1838_183801

/-- The number of steps Xiao Gang takes from his house to his school -/
def steps : ℕ := 2000

/-- The distance from Xiao Gang's house to his school in meters -/
def distance : ℝ := 900

/-- Xiao Gang's step length in meters -/
def step_length : ℝ := 0.45

/-- Theorem stating that the number of steps multiplied by the step length equals the distance -/
theorem steps_correct : (steps : ℝ) * step_length = distance := by sorry

end steps_correct_l1838_183801


namespace bobs_weight_l1838_183838

/-- Given two people, Jim and Bob, prove Bob's weight under specific conditions. -/
theorem bobs_weight (jim_weight bob_weight : ℝ) : 
  (jim_weight + bob_weight = 200) →
  (bob_weight + jim_weight = bob_weight / 3) →
  bob_weight = 120 := by
  sorry

end bobs_weight_l1838_183838


namespace cone_hemisphere_relation_cone_base_radius_is_10_5_l1838_183825

/-- Represents a cone with a hemisphere resting on its base --/
structure ConeWithHemisphere where
  cone_height : ℝ
  hemisphere_radius : ℝ
  cone_base_radius : ℝ

/-- Checks if the configuration is valid --/
def is_valid_configuration (c : ConeWithHemisphere) : Prop :=
  c.cone_height > 0 ∧ c.hemisphere_radius > 0 ∧ c.cone_base_radius > c.hemisphere_radius

/-- Theorem stating the relationship between cone dimensions and hemisphere --/
theorem cone_hemisphere_relation (c : ConeWithHemisphere) 
  (h_valid : is_valid_configuration c)
  (h_height : c.cone_height = 9)
  (h_radius : c.hemisphere_radius = 3) :
  c.cone_base_radius = 10.5 := by
  sorry

/-- Main theorem proving the base radius of the cone --/
theorem cone_base_radius_is_10_5 :
  ∃ c : ConeWithHemisphere, 
    is_valid_configuration c ∧ 
    c.cone_height = 9 ∧ 
    c.hemisphere_radius = 3 ∧ 
    c.cone_base_radius = 10.5 := by
  sorry

end cone_hemisphere_relation_cone_base_radius_is_10_5_l1838_183825


namespace units_digit_problem_l1838_183868

theorem units_digit_problem : ∃ x : ℕ, 
  (x > 0) ∧ 
  (x % 10 = 3) ∧ 
  ((35^87 + x^53) % 10 = 8) := by
  sorry

end units_digit_problem_l1838_183868


namespace ariels_age_ariels_current_age_l1838_183809

theorem ariels_age (birth_year : Nat) (fencing_start_year : Nat) (years_fencing : Nat) : Nat :=
  let current_year := fencing_start_year + years_fencing
  let age := current_year - birth_year
  age

theorem ariels_current_age : 
  ariels_age 1992 2006 16 = 30 := by
  sorry

end ariels_age_ariels_current_age_l1838_183809


namespace function_identity_l1838_183813

def IsNonDegenerateTriangle (a b c : ℕ+) : Prop :=
  a.val + b.val > c.val ∧ b.val + c.val > a.val ∧ c.val + a.val > b.val

theorem function_identity (f : ℕ+ → ℕ+) 
  (h : ∀ (a b : ℕ+), IsNonDegenerateTriangle a (f b) (f (b + f a - 1))) :
  ∀ (a : ℕ+), f a = a := by
  sorry

end function_identity_l1838_183813


namespace total_distance_is_963_l1838_183869

/-- The total combined distance of objects thrown by Bill, Ted, and Alice -/
def total_distance (ted_sticks ted_rocks : ℕ) 
  (bill_stick_dist bill_rock_dist : ℝ) : ℝ :=
  let bill_sticks := ted_sticks - 6
  let alice_sticks := ted_sticks / 2
  let bill_rocks := ted_rocks / 2
  let alice_rocks := bill_rocks * 3
  let ted_stick_dist := bill_stick_dist * 1.5
  let alice_stick_dist := bill_stick_dist * 2
  let ted_rock_dist := bill_rock_dist * 1.25
  let alice_rock_dist := bill_rock_dist * 3
  (bill_sticks : ℝ) * bill_stick_dist +
  (ted_sticks : ℝ) * ted_stick_dist +
  (alice_sticks : ℝ) * alice_stick_dist +
  (bill_rocks : ℝ) * bill_rock_dist +
  (ted_rocks : ℝ) * ted_rock_dist +
  (alice_rocks : ℝ) * alice_rock_dist

/-- Theorem stating the total distance is 963 meters given the problem conditions -/
theorem total_distance_is_963 :
  total_distance 12 18 8 6 = 963 := by
  sorry

end total_distance_is_963_l1838_183869


namespace math_competition_correct_answers_l1838_183870

theorem math_competition_correct_answers 
  (total_questions : Nat) 
  (correct_score : Nat) 
  (incorrect_penalty : Nat) 
  (xiao_ming_score : Nat) 
  (xiao_hong_score : Nat) 
  (xiao_hua_score : Nat) :
  total_questions = 10 →
  correct_score = 10 →
  incorrect_penalty = 3 →
  xiao_ming_score = 87 →
  xiao_hong_score = 74 →
  xiao_hua_score = 9 →
  (total_questions - (total_questions * correct_score - xiao_ming_score) / (correct_score + incorrect_penalty)) +
  (total_questions - (total_questions * correct_score - xiao_hong_score) / (correct_score + incorrect_penalty)) +
  (total_questions - (total_questions * correct_score - xiao_hua_score) / (correct_score + incorrect_penalty)) = 20 :=
by sorry

end math_competition_correct_answers_l1838_183870


namespace light_bulb_resistance_l1838_183817

theorem light_bulb_resistance (U I R : ℝ) (hU : U = 220) (hI : I ≤ 0.11) (hOhm : I = U / R) : R ≥ 2000 := by
  sorry

end light_bulb_resistance_l1838_183817


namespace circle_tangent_to_directrix_l1838_183857

-- Define the parabola
def parabola (x y : ℝ) : Prop := y = (1/4) * x^2

-- Define the point A
def point_A : ℝ × ℝ := (0, 1)

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the property that the circle passes through point A
def passes_through_A (c : Circle) : Prop :=
  let (x, y) := c.center
  (x - point_A.1)^2 + (y - point_A.2)^2 = c.radius^2

-- Define the property that the circle's center lies on the parabola
def center_on_parabola (c : Circle) : Prop :=
  let (x, y) := c.center
  parabola x y

-- Define the property that the circle is tangent to line l
def tangent_to_l (c : Circle) (l : ℝ → ℝ) : Prop :=
  let (x, y) := c.center
  (y - l x)^2 = c.radius^2

-- State the theorem
theorem circle_tangent_to_directrix :
  ∀ c : Circle,
  passes_through_A c →
  center_on_parabola c →
  ∃ l : ℝ → ℝ, (∀ x, l x = -1) ∧ tangent_to_l c l :=
sorry

end circle_tangent_to_directrix_l1838_183857


namespace x_squared_gt_1_sufficient_not_necessary_for_reciprocal_lt_1_l1838_183835

theorem x_squared_gt_1_sufficient_not_necessary_for_reciprocal_lt_1 :
  (∀ x : ℝ, x^2 > 1 → 1/x < 1) ∧
  (∃ x : ℝ, 1/x < 1 ∧ ¬(x^2 > 1)) := by
  sorry

end x_squared_gt_1_sufficient_not_necessary_for_reciprocal_lt_1_l1838_183835


namespace base_conversion_1234_to_base_4_l1838_183852

theorem base_conversion_1234_to_base_4 :
  (3 * 4^4 + 4 * 4^3 + 1 * 4^2 + 0 * 4^1 + 2 * 4^0) = 1234 := by
  sorry

end base_conversion_1234_to_base_4_l1838_183852


namespace tax_threshold_value_l1838_183827

def tax_calculation (X : ℝ) (I : ℝ) : ℝ := 0.12 * X + 0.20 * (I - X)

theorem tax_threshold_value :
  ∃ (X : ℝ), 
    X = 40000 ∧
    tax_calculation X 56000 = 8000 := by
  sorry

end tax_threshold_value_l1838_183827


namespace quadratic_real_roots_condition_l1838_183885

/-- Defines the quadratic equation kx^2 - x - 1 = 0 -/
def quadratic_equation (k x : ℝ) : Prop :=
  k * x^2 - x - 1 = 0

/-- Defines when a quadratic equation has real roots -/
def has_real_roots (k : ℝ) : Prop :=
  ∃ x : ℝ, quadratic_equation k x

/-- Theorem stating the condition for the quadratic equation to have real roots -/
theorem quadratic_real_roots_condition (k : ℝ) :
  has_real_roots k ↔ k ≥ -1/4 ∧ k ≠ 0 :=
sorry

end quadratic_real_roots_condition_l1838_183885


namespace find_divisor_l1838_183881

theorem find_divisor (dividend quotient remainder : ℕ) 
  (h1 : dividend = 12401)
  (h2 : quotient = 76)
  (h3 : remainder = 13)
  (h4 : dividend = quotient * (dividend / quotient) + remainder) : 
  dividend / quotient = 163 := by
sorry

end find_divisor_l1838_183881


namespace unique_a_for_set_equality_l1838_183846

theorem unique_a_for_set_equality : ∃! a : ℝ, ({1, 3, a^2} ∪ {1, a+2} : Set ℝ) = {1, 3, a^2} := by
  sorry

end unique_a_for_set_equality_l1838_183846


namespace fraction_addition_l1838_183815

theorem fraction_addition (a : ℝ) (ha : a ≠ 0) : 3 / a + 2 / a = 5 / a := by
  sorry

end fraction_addition_l1838_183815


namespace square_sum_geq_product_l1838_183876

theorem square_sum_geq_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a + b + c ≥ a * b * c) : a^2 + b^2 + c^2 ≥ a * b * c := by
  sorry

end square_sum_geq_product_l1838_183876


namespace cake_slices_proof_l1838_183864

/-- The number of calories in each slice of cake -/
def calories_per_cake_slice : ℕ := 347

/-- The number of brownies in a pan -/
def brownies_per_pan : ℕ := 6

/-- The number of calories in each brownie -/
def calories_per_brownie : ℕ := 375

/-- The difference in calories between the cake and the pan of brownies -/
def calorie_difference : ℕ := 526

/-- The number of slices in the cake -/
def cake_slices : ℕ := 8

theorem cake_slices_proof :
  cake_slices * calories_per_cake_slice = 
  brownies_per_pan * calories_per_brownie + calorie_difference := by
  sorry

end cake_slices_proof_l1838_183864


namespace problem_statement_l1838_183892

theorem problem_statement (x y : ℚ) : 
  x = 3/4 → y = 4/3 → (3/5 : ℚ) * x^5 * y^8 = 897/1000 := by
  sorry

end problem_statement_l1838_183892


namespace new_average_production_l1838_183882

theorem new_average_production (n : ℕ) (old_average : ℝ) (today_production : ℝ) 
  (h1 : n = 5)
  (h2 : old_average = 60)
  (h3 : today_production = 90) :
  let total_production := n * old_average
  let new_total_production := total_production + today_production
  let new_days := n + 1
  (new_total_production / new_days : ℝ) = 65 := by sorry

end new_average_production_l1838_183882


namespace fraction_equals_eight_over_twentyseven_l1838_183867

def numerator : ℕ := 1*2*4 + 2*4*8 + 3*6*12 + 4*8*16
def denominator : ℕ := 1*3*9 + 2*6*18 + 3*9*27 + 4*12*36

theorem fraction_equals_eight_over_twentyseven :
  (numerator : ℚ) / denominator = 8 / 27 := by
  sorry

end fraction_equals_eight_over_twentyseven_l1838_183867


namespace equilateral_side_length_l1838_183880

/-- Given a diagram with an equilateral triangle and a right-angled triangle,
    where the right-angled triangle has a side length of 6 and both triangles
    have a 45-degree angle, the side length y of the equilateral triangle is 6√2. -/
theorem equilateral_side_length (y : ℝ) : y = 6 * Real.sqrt 2 := by sorry

end equilateral_side_length_l1838_183880


namespace cylinder_volume_l1838_183807

theorem cylinder_volume (r_cylinder r_cone h_cylinder h_cone v_cone : ℝ) :
  r_cylinder / r_cone = 2 / 3 →
  h_cylinder / h_cone = 4 / 3 →
  v_cone = 5.4 →
  (π * r_cylinder^2 * h_cylinder) = 3.2 :=
by sorry

end cylinder_volume_l1838_183807


namespace hotdog_eating_record_l1838_183863

/-- The hotdog eating record problem -/
theorem hotdog_eating_record 
  (total_time : ℕ) 
  (halfway_time : ℕ) 
  (halfway_hotdogs : ℕ) 
  (required_rate : ℕ) 
  (h1 : total_time = 10) 
  (h2 : halfway_time = total_time / 2) 
  (h3 : halfway_hotdogs = 20) 
  (h4 : required_rate = 11) : 
  halfway_hotdogs + required_rate * (total_time - halfway_time) = 75 := by
sorry


end hotdog_eating_record_l1838_183863


namespace total_people_waiting_l1838_183805

/-- The number of people waiting at each entrance -/
def people_per_entrance : ℕ := 283

/-- The number of entrances -/
def num_entrances : ℕ := 5

/-- The total number of people waiting to get in -/
def total_people : ℕ := people_per_entrance * num_entrances

theorem total_people_waiting :
  total_people = 1415 := by sorry

end total_people_waiting_l1838_183805


namespace opera_selection_probability_l1838_183810

theorem opera_selection_probability :
  let total_operas : ℕ := 5
  let distinguished_operas : ℕ := 2
  let selection_size : ℕ := 2

  let total_combinations : ℕ := Nat.choose total_operas selection_size
  let favorable_combinations : ℕ := distinguished_operas * (total_operas - distinguished_operas)

  (favorable_combinations : ℚ) / total_combinations = 3 / 5 :=
by sorry

end opera_selection_probability_l1838_183810


namespace pharmacist_weights_exist_l1838_183858

theorem pharmacist_weights_exist : ∃ (a b c : ℝ), 
  0 < a ∧ a < 90 ∧
  0 < b ∧ b < 90 ∧
  0 < c ∧ c < 90 ∧
  a + b = 100 ∧
  a + c = 101 ∧
  b + c = 102 := by
sorry

end pharmacist_weights_exist_l1838_183858


namespace ratio_of_shares_l1838_183832

theorem ratio_of_shares (total amount_c : ℕ) (h1 : total = 2000) (h2 : amount_c = 1600) :
  (total - amount_c) / amount_c = 1 / 4 := by
  sorry

end ratio_of_shares_l1838_183832


namespace area_of_triangle_A_l1838_183803

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a parallelogram -/
structure Parallelogram :=
  (A B C D : Point)

/-- Represents the folded state of the parallelogram -/
structure FoldedParallelogram :=
  (original : Parallelogram)
  (A' : Point)
  (K : Point)

/-- The area of a parallelogram -/
def parallelogramArea (p : Parallelogram) : ℝ := 27

/-- The ratio of BK to KC -/
def BK_KC_ratio : ℝ × ℝ := (3, 2)

/-- The area of triangle A'KC -/
def triangleA'KC_area (fp : FoldedParallelogram) : ℝ := sorry

theorem area_of_triangle_A'KC 
  (p : Parallelogram)
  (fp : FoldedParallelogram)
  (h1 : fp.original = p)
  (h2 : parallelogramArea p = 27)
  (h3 : BK_KC_ratio = (3, 2)) :
  triangleA'KC_area fp = 3.6 :=
sorry

end area_of_triangle_A_l1838_183803


namespace not_parabola_l1838_183847

/-- The equation x^2 + ky^2 = 1 cannot represent a parabola for any real k -/
theorem not_parabola (k : ℝ) : 
  ¬ ∃ (a b c d e : ℝ), ∀ (x y : ℝ), 
    x^2 + k*y^2 = 1 ↔ a*x^2 + b*x*y + c*y^2 + d*x + e*y = 0 ∧ b^2 = 4*a*c := by
  sorry

end not_parabola_l1838_183847


namespace round_to_scientific_notation_l1838_183831

/-- Rounds a real number to a specified number of significant figures -/
def roundToSigFigs (x : ℝ) (n : ℕ) : ℝ := sorry

/-- Converts a real number to scientific notation (a * 10^b form) -/
def toScientificNotation (x : ℝ) : ℝ × ℤ := sorry

theorem round_to_scientific_notation :
  let x := -29800000
  let sigFigs := 3
  let (a, b) := toScientificNotation (roundToSigFigs x sigFigs)
  a = -2.98 ∧ b = 7 := by sorry

end round_to_scientific_notation_l1838_183831


namespace triangle_geometric_sequence_l1838_183886

/-- In a triangle ABC where sides a, b, c form a geometric sequence and satisfy a² - c² = ac - bc, 
    the ratio (b * sin B) / c is equal to √3/2. -/
theorem triangle_geometric_sequence (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  (b ^ 2 = a * c) →  -- geometric sequence condition
  (a ^ 2 - c ^ 2 = a * c - b * c) →
  (a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos A) →  -- cosine rule
  (b * Real.sin B = a * Real.sin A) →  -- sine rule
  (b * Real.sin B) / c = Real.sqrt 3 / 2 := by
sorry

end triangle_geometric_sequence_l1838_183886


namespace min_n_for_60n_divisible_by_4_and_8_l1838_183859

theorem min_n_for_60n_divisible_by_4_and_8 :
  ∃ (n : ℕ), n > 0 ∧ 4 ∣ 60 * n ∧ 8 ∣ 60 * n ∧
  ∀ (m : ℕ), m > 0 → 4 ∣ 60 * m → 8 ∣ 60 * m → n ≤ m :=
by sorry

end min_n_for_60n_divisible_by_4_and_8_l1838_183859


namespace chameleons_changed_count_l1838_183811

/-- Represents the number of chameleons that changed color --/
def chameleons_changed (total : ℕ) (blue_factor : ℕ) (red_factor : ℕ) : ℕ :=
  let initial_blue := blue_factor * (total / (blue_factor + 1))
  total - initial_blue - (total - initial_blue) / red_factor

/-- Theorem stating that 80 chameleons changed color under the given conditions --/
theorem chameleons_changed_count :
  chameleons_changed 140 5 3 = 80 := by
  sorry

end chameleons_changed_count_l1838_183811


namespace least_three_digit_multiple_l1838_183890

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem least_three_digit_multiple : 
  (∀ k : ℕ, is_three_digit k ∧ 3 ∣ k ∧ 7 ∣ k ∧ 11 ∣ k → 231 ≤ k) ∧ 
  is_three_digit 231 ∧ 3 ∣ 231 ∧ 7 ∣ 231 ∧ 11 ∣ 231 :=
sorry

end least_three_digit_multiple_l1838_183890


namespace most_cost_effective_plan_optimal_plan_is_valid_l1838_183808

/-- Represents the rental plan for buses -/
structure RentalPlan where
  large_buses : ℕ
  small_buses : ℕ

/-- Calculates the total number of seats for a given rental plan -/
def total_seats (plan : RentalPlan) : ℕ :=
  plan.large_buses * 45 + plan.small_buses * 30

/-- Calculates the total cost for a given rental plan -/
def total_cost (plan : RentalPlan) : ℕ :=
  plan.large_buses * 400 + plan.small_buses * 300

/-- Checks if a rental plan is valid according to the given conditions -/
def is_valid_plan (plan : RentalPlan) : Prop :=
  total_seats plan ≥ 240 ∧ 
  plan.large_buses + plan.small_buses ≤ 6 ∧
  total_cost plan ≤ 2300

/-- The theorem stating that the most cost-effective valid plan is 4 large buses and 2 small buses -/
theorem most_cost_effective_plan :
  ∀ (plan : RentalPlan),
    is_valid_plan plan →
    total_cost plan ≥ total_cost { large_buses := 4, small_buses := 2 } :=
by sorry

/-- The theorem stating that the plan with 4 large buses and 2 small buses is valid -/
theorem optimal_plan_is_valid :
  is_valid_plan { large_buses := 4, small_buses := 2 } :=
by sorry

end most_cost_effective_plan_optimal_plan_is_valid_l1838_183808
