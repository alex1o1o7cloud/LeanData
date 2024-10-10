import Mathlib

namespace original_number_proof_l1221_122152

theorem original_number_proof :
  ∃ (n : ℕ), n = 3830 ∧ (∃ (k : ℕ), n - 5 = 15 * k) ∧
  (∀ (m : ℕ), m < 5 → ¬(∃ (j : ℕ), n - m = 15 * j)) :=
by sorry

end original_number_proof_l1221_122152


namespace intersection_of_A_and_B_l1221_122185

-- Define the sets A and B
def A : Set (ℝ × ℝ) := {p | p.2 = p.1}
def B : Set (ℝ × ℝ) := {p | p.2 = p.1^2}

-- Define the intersection set
def intersection_set : Set (ℝ × ℝ) := {(0, 0), (1, 1)}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = intersection_set := by
  sorry

end intersection_of_A_and_B_l1221_122185


namespace regular_polygon_with_108_degree_interior_angles_l1221_122145

theorem regular_polygon_with_108_degree_interior_angles (n : ℕ) : 
  (n ≥ 3) →  -- ensuring it's a valid polygon
  (((n - 2) * 180) / n = 108) →  -- interior angle formula
  (n = 5) :=
by
  sorry

end regular_polygon_with_108_degree_interior_angles_l1221_122145


namespace probability_one_pair_l1221_122102

def total_gloves : ℕ := 10
def pairs_of_gloves : ℕ := 5
def gloves_picked : ℕ := 4

def total_ways : ℕ := Nat.choose total_gloves gloves_picked

def ways_one_pair : ℕ := 
  Nat.choose pairs_of_gloves 1 * Nat.choose 2 2 * Nat.choose (total_gloves - 2) (gloves_picked - 2)

theorem probability_one_pair :
  (ways_one_pair : ℚ) / total_ways = 1 / 7 := by sorry

end probability_one_pair_l1221_122102


namespace f_plus_3_abs_l1221_122111

noncomputable def f (x : ℝ) : ℝ :=
  if -3 ≤ x ∧ x ≤ 0 then -2 - x
  else if 0 < x ∧ x ≤ 2 then Real.sqrt (4 - (x - 2)^2) - 2
  else if 2 < x ∧ x ≤ 3 then 2 * (x - 2)
  else 0  -- undefined for other x values

theorem f_plus_3_abs (x : ℝ) (hx : -3 ≤ x ∧ x ≤ 3) : 
  |f x + 3| = f x + 3 :=
by sorry

end f_plus_3_abs_l1221_122111


namespace calculation_proof_l1221_122120

theorem calculation_proof : 
  (3/5 : ℚ) * 200 + (456/1000 : ℚ) * 875 + (7/8 : ℚ) * 320 - 
  ((5575/10000 : ℚ) * 1280 + (1/3 : ℚ) * 960) = -2349/10 := by
  sorry

end calculation_proof_l1221_122120


namespace parallel_to_y_axis_fourth_quadrant_integer_a_l1221_122139

-- Define point A
def A (a : ℝ) : ℝ × ℝ := (3*a - 9, 2*a - 10)

-- Define point B
def B : ℝ × ℝ := (4, 5)

-- Theorem 1
theorem parallel_to_y_axis (a : ℝ) : 
  (A a).1 = B.1 → a = 13/3 := by sorry

-- Theorem 2
theorem fourth_quadrant_integer_a : 
  ∃ (a : ℤ), (A a).1 > 0 ∧ (A a).2 < 0 → A a = (3, -2) := by sorry

end parallel_to_y_axis_fourth_quadrant_integer_a_l1221_122139


namespace tangent_line_problem_l1221_122180

-- Define the curve
def curve (x a b : ℝ) : ℝ := x^3 + a*x + b

-- Define the tangent line
def tangent_line (x k : ℝ) : ℝ := k*x + 1

-- Define the derivative of the curve
def curve_derivative (x a : ℝ) : ℝ := 3*x^2 + a

theorem tangent_line_problem (a b k : ℝ) : 
  curve 1 a b = 2 →
  tangent_line 1 k = 2 →
  curve_derivative 1 a = k →
  b - a = 5 := by
  sorry


end tangent_line_problem_l1221_122180


namespace quadratic_always_nonnegative_l1221_122143

theorem quadratic_always_nonnegative (a : ℝ) : 
  (∀ x : ℝ, x^2 + 2*a*x + 1 ≥ 0) ↔ a ∈ Set.Icc (-1 : ℝ) 1 := by
sorry

end quadratic_always_nonnegative_l1221_122143


namespace complex_equation_solution_l1221_122157

theorem complex_equation_solution (z : ℂ) (i : ℂ) (h : i^2 = -1) :
  (1 - i)^2 / z = 1 + i → z = -1 - i :=
by
  sorry

end complex_equation_solution_l1221_122157


namespace line_passes_through_fixed_point_l1221_122179

/-- Parabola C: y² = 4x -/
def parabola_C (x y : ℝ) : Prop := y^2 = 4*x

/-- Line l: x = my + 4 -/
def line_l (m : ℝ) (x y : ℝ) : Prop := x = m*y + 4

/-- Point on parabola C -/
structure PointOnC where
  x : ℝ
  y : ℝ
  on_C : parabola_C x y

/-- Points M and N are perpendicular from origin -/
def perpendicular_from_origin (M N : PointOnC) : Prop :=
  M.x * N.x + M.y * N.y = 0

theorem line_passes_through_fixed_point (m : ℝ) 
  (M N : PointOnC) (h_distinct : M ≠ N) 
  (h_on_l : line_l m M.x M.y ∧ line_l m N.x N.y)
  (h_perp : perpendicular_from_origin M N) :
  line_l m 4 0 := by sorry

end line_passes_through_fixed_point_l1221_122179


namespace all_six_lines_tangent_l1221_122131

/-- A line in a plane -/
structure Line :=
  (id : ℕ)

/-- A circle in a plane -/
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

/-- Predicate to check if a line is tangent to a circle -/
def is_tangent (l : Line) (c : Circle) : Prop :=
  sorry

/-- A set of six lines in a plane -/
def six_lines : Finset Line :=
  sorry

/-- Condition: For any three lines, there exists a fourth line such that all four are tangent to some circle -/
def four_line_tangent_condition (lines : Finset Line) : Prop :=
  ∀ (l1 l2 l3 : Line), l1 ∈ lines → l2 ∈ lines → l3 ∈ lines →
    ∃ (l4 : Line) (c : Circle), l4 ∈ lines ∧
      is_tangent l1 c ∧ is_tangent l2 c ∧ is_tangent l3 c ∧ is_tangent l4 c

/-- Theorem: If the four_line_tangent_condition holds for six lines, then all six lines are tangent to the same circle -/
theorem all_six_lines_tangent (h : four_line_tangent_condition six_lines) :
  ∃ (c : Circle), ∀ (l : Line), l ∈ six_lines → is_tangent l c :=
sorry

end all_six_lines_tangent_l1221_122131


namespace distributive_property_negative_three_l1221_122127

theorem distributive_property_negative_three (a b : ℝ) : -3 * (-a - b) = 3 * a + 3 * b := by
  sorry

end distributive_property_negative_three_l1221_122127


namespace playground_children_count_l1221_122138

theorem playground_children_count : 
  ∀ (girls boys : ℕ), 
  girls = 28 → 
  boys = 35 → 
  girls + boys = 63 := by
sorry

end playground_children_count_l1221_122138


namespace pudding_cost_pudding_cost_is_two_l1221_122100

/-- The cost of each cup of pudding, given the conditions of Jane's purchase -/
theorem pudding_cost (num_ice_cream : ℕ) (num_pudding : ℕ) (ice_cream_price : ℕ) (extra_spent : ℕ) : ℕ :=
  let total_ice_cream := num_ice_cream * ice_cream_price
  let pudding_cost := (total_ice_cream - extra_spent) / num_pudding
  pudding_cost

/-- Proof that each cup of pudding costs $2 -/
theorem pudding_cost_is_two :
  pudding_cost 15 5 5 65 = 2 := by
  sorry

end pudding_cost_pudding_cost_is_two_l1221_122100


namespace largest_package_size_l1221_122150

theorem largest_package_size (ming catherine alex : ℕ) 
  (h_ming : ming = 36) 
  (h_catherine : catherine = 60) 
  (h_alex : alex = 48) : 
  Nat.gcd ming (Nat.gcd catherine alex) = 12 := by
  sorry

end largest_package_size_l1221_122150


namespace states_fraction_1790_1799_l1221_122159

theorem states_fraction_1790_1799 (total_states : ℕ) (states_1790_1799 : ℕ) :
  total_states = 30 →
  states_1790_1799 = 9 →
  (states_1790_1799 : ℚ) / total_states = 3 / 10 := by
  sorry

end states_fraction_1790_1799_l1221_122159


namespace min_pool_cost_l1221_122118

/-- Minimum cost for constructing a rectangular open-top water pool --/
theorem min_pool_cost (volume : ℝ) (depth : ℝ) (bottom_cost : ℝ) (wall_cost : ℝ) :
  volume = 8 →
  depth = 2 →
  bottom_cost = 120 →
  wall_cost = 80 →
  ∃ (cost : ℝ), cost = 1760 ∧ 
    ∀ (length width : ℝ),
      length > 0 →
      width > 0 →
      length * width * depth = volume →
      bottom_cost * length * width + wall_cost * (2 * length + 2 * width) * depth ≥ cost :=
by sorry

end min_pool_cost_l1221_122118


namespace complex_equation_solution_l1221_122117

theorem complex_equation_solution : ∃ (z : ℂ), 3 - 2 * Complex.I * z = 7 + 4 * Complex.I * z ∧ z = (2 * Complex.I) / 3 := by
  sorry

end complex_equation_solution_l1221_122117


namespace right_angle_points_iff_h_squared_leq_ac_l1221_122164

/-- An isosceles trapezoid -/
structure IsoscelesTrapezoid where
  a : ℝ  -- length of one base
  c : ℝ  -- length of the other base
  h : ℝ  -- altitude
  a_pos : 0 < a  -- a is positive
  c_pos : 0 < c  -- c is positive
  h_pos : 0 < h  -- h is positive

/-- The existence of points P on the axis of symmetry where both legs subtend right angles -/
def exists_right_angle_points (t : IsoscelesTrapezoid) : Prop :=
  ∃ P : ℝ × ℝ, P.1 = 0 ∧ 0 ≤ P.2 ∧ P.2 ≤ t.h ∧ 
    (P.2^2 + (t.a/2)^2 = (P.2^2 + ((t.a - t.c)/2)^2 + t.h^2)/4) ∧
    ((t.h - P.2)^2 + (t.c/2)^2 = (P.2^2 + ((t.a - t.c)/2)^2 + t.h^2)/4)

/-- The main theorem -/
theorem right_angle_points_iff_h_squared_leq_ac (t : IsoscelesTrapezoid) :
  exists_right_angle_points t ↔ t.h^2 ≤ t.a * t.c :=
sorry

end right_angle_points_iff_h_squared_leq_ac_l1221_122164


namespace solution_set_range_of_m_l1221_122116

-- Define the function f
def f (x : ℝ) : ℝ := |2*x - 1| + |x + 1|

-- Theorem for part (Ⅰ)
theorem solution_set (x : ℝ) : f x ≤ 2 ↔ 0 ≤ x ∧ x ≤ 2/3 := by sorry

-- Theorem for part (Ⅱ)
theorem range_of_m (m : ℝ) : 
  (∀ x, ∃ a ∈ Set.Icc (-2) 1, f x ≥ f a + m) → m ≤ 0 := by sorry

end solution_set_range_of_m_l1221_122116


namespace bus_profit_properties_l1221_122198

/-- Represents the daily profit of a bus given the number of passengers -/
def daily_profit (x : ℕ) : ℤ :=
  2 * x - 600

theorem bus_profit_properties :
  let min_passengers_no_loss := 300
  let profit_500_passengers := daily_profit 500
  let relationship (x : ℕ) := daily_profit x = 2 * x - 600
  (∀ x : ℕ, x ≥ min_passengers_no_loss → daily_profit x ≥ 0) ∧
  (profit_500_passengers = 400) ∧
  (∀ x : ℕ, relationship x) :=
by sorry

end bus_profit_properties_l1221_122198


namespace triangle_theorem_l1221_122182

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_theorem (t : Triangle) 
  (h1 : t.b^2 + t.c^2 - Real.sqrt 2 * t.b * t.c = t.a^2)
  (h2 : t.c / t.b = 2 * Real.sqrt 2) : 
  t.A = π/4 ∧ Real.tan t.B = 1/3 := by
  sorry


end triangle_theorem_l1221_122182


namespace julio_twice_james_age_l1221_122123

/-- 
Given:
- Julio is currently 36 years old
- James is currently 11 years old

Prove that in 14 years, Julio's age will be twice James's age
-/
theorem julio_twice_james_age (julio_age : ℕ) (james_age : ℕ) (years : ℕ) : 
  julio_age = 36 → james_age = 11 → years = 14 → 
  julio_age + years = 2 * (james_age + years) := by
  sorry

end julio_twice_james_age_l1221_122123


namespace teachers_class_size_l1221_122114

/-- The number of students in Teacher Yang's class -/
def num_students : ℕ := 28

theorem teachers_class_size :
  (num_students / 2 : ℕ) +     -- Half in math competition
  (num_students / 4 : ℕ) +     -- Quarter in music group
  (num_students / 7 : ℕ) +     -- One-seventh in reading room
  3 =                          -- Remaining three watching TV
  num_students :=              -- Equals total number of students
by sorry

end teachers_class_size_l1221_122114


namespace select_students_l1221_122141

theorem select_students (n m : ℕ) (h1 : n = 10) (h2 : m = 3) : 
  Nat.choose n m = 120 := by
  sorry

end select_students_l1221_122141


namespace peach_count_l1221_122174

/-- Calculates the total number of peaches after picking -/
def total_peaches (initial : ℕ) (picked : ℕ) : ℕ :=
  initial + picked

/-- Theorem: The total number of peaches is the sum of initial and picked peaches -/
theorem peach_count (initial picked : ℕ) :
  total_peaches initial picked = initial + picked := by
  sorry

end peach_count_l1221_122174


namespace jenny_wedding_budget_l1221_122175

/-- Calculates the total catering budget for a wedding --/
def totalCateringBudget (totalGuests : ℕ) (steakMultiplier : ℕ) (steakCost chickenCost : ℚ) : ℚ :=
  let chickenGuests := totalGuests / (steakMultiplier + 1)
  let steakGuests := totalGuests - chickenGuests
  steakGuests * steakCost + chickenGuests * chickenCost

/-- Proves that the total catering budget for Jenny's wedding is $1860 --/
theorem jenny_wedding_budget :
  totalCateringBudget 80 3 25 18 = 1860 := by
  sorry

end jenny_wedding_budget_l1221_122175


namespace quadratic_two_distinct_roots_l1221_122163

theorem quadratic_two_distinct_roots : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  x₁^2 - 4*x₁ + 2 = 0 ∧ x₂^2 - 4*x₂ + 2 = 0 := by
  sorry

end quadratic_two_distinct_roots_l1221_122163


namespace parabola_directrix_l1221_122135

/-- The parabola equation -/
def parabola_eq (x y : ℝ) : Prop := x = (y^2 - 8*y - 20) / 16

/-- The directrix equation -/
def directrix_eq (x : ℝ) : Prop := x = -6.25

/-- Theorem stating that the given directrix equation is correct for the parabola -/
theorem parabola_directrix : 
  ∀ x y : ℝ, parabola_eq x y → (∃ x_d : ℝ, directrix_eq x_d ∧ 
    -- Additional conditions about the relationship between the point (x,y) on the parabola
    -- and its distance to the directrix would be specified here
    True) :=
by sorry

end parabola_directrix_l1221_122135


namespace bicycle_shop_inventory_l1221_122110

/-- Represents the bicycle shop inventory problem --/
theorem bicycle_shop_inventory
  (initial_stock : ℕ)
  (weekly_addition : ℕ)
  (weeks : ℕ)
  (final_stock : ℕ)
  (h1 : initial_stock = 51)
  (h2 : weekly_addition = 3)
  (h3 : weeks = 4)
  (h4 : final_stock = 45) :
  initial_stock + weekly_addition * weeks - final_stock = 18 :=
by sorry

end bicycle_shop_inventory_l1221_122110


namespace skyler_song_composition_l1221_122158

/-- Represents the success levels of songs --/
inductive SuccessLevel
  | ExtremelySuccessful
  | Successful
  | ModeratelySuccessful
  | LessSuccessful
  | Unreleased

/-- Represents Skyler's song composition --/
structure SongComposition where
  hitSongs : Nat
  top100Songs : Nat
  unreleasedSongs : Nat
  duetsTop20 : Nat
  duetsBelow200 : Nat
  soundtracksExtremely : Nat
  soundtracksModerate : Nat
  soundtracksLukewarm : Nat
  internationalGlobal : Nat
  internationalRegional : Nat
  internationalOverlooked : Nat

/-- Calculates the total number of songs --/
def totalSongs (composition : SongComposition) : Nat :=
  composition.hitSongs + composition.top100Songs + composition.unreleasedSongs +
  composition.duetsTop20 + composition.duetsBelow200 +
  composition.soundtracksExtremely + composition.soundtracksModerate + composition.soundtracksLukewarm +
  composition.internationalGlobal + composition.internationalRegional + composition.internationalOverlooked

/-- Calculates the number of songs for each success level --/
def songsBySuccessLevel (composition : SongComposition) : SuccessLevel → Nat
  | SuccessLevel.ExtremelySuccessful => composition.hitSongs + composition.internationalGlobal
  | SuccessLevel.Successful => composition.top100Songs + composition.duetsTop20 + composition.soundtracksExtremely
  | SuccessLevel.ModeratelySuccessful => composition.soundtracksModerate + composition.internationalRegional
  | SuccessLevel.LessSuccessful => composition.soundtracksLukewarm + composition.internationalOverlooked + composition.duetsBelow200
  | SuccessLevel.Unreleased => composition.unreleasedSongs

/-- Theorem stating the total number of songs and their success level breakdown --/
theorem skyler_song_composition :
  ∃ (composition : SongComposition),
    composition.hitSongs = 25 ∧
    composition.top100Songs = composition.hitSongs + 10 ∧
    composition.unreleasedSongs = composition.hitSongs - 5 ∧
    composition.duetsTop20 = 6 ∧
    composition.duetsBelow200 = 6 ∧
    composition.soundtracksExtremely = 3 ∧
    composition.soundtracksModerate = 8 ∧
    composition.soundtracksLukewarm = 7 ∧
    composition.internationalGlobal = 1 ∧
    composition.internationalRegional = 7 ∧
    composition.internationalOverlooked = 14 ∧
    totalSongs composition = 132 ∧
    songsBySuccessLevel composition SuccessLevel.ExtremelySuccessful = 26 ∧
    songsBySuccessLevel composition SuccessLevel.Successful = 44 ∧
    songsBySuccessLevel composition SuccessLevel.ModeratelySuccessful = 15 ∧
    songsBySuccessLevel composition SuccessLevel.LessSuccessful = 27 ∧
    songsBySuccessLevel composition SuccessLevel.Unreleased = 20 := by
  sorry

end skyler_song_composition_l1221_122158


namespace problem_statement_l1221_122148

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |x + 2|

-- Define the set T
def T : Set ℝ := {a | -Real.sqrt 3 < a ∧ a < Real.sqrt 3}

-- Theorem statement
theorem problem_statement :
  (∀ x : ℝ, f x > a^2) →
  (∀ m n : ℝ, m ∈ T → n ∈ T → Real.sqrt 3 * |m + n| < |m * n + 3|) :=
by sorry

end problem_statement_l1221_122148


namespace flag_problem_l1221_122188

theorem flag_problem (x : ℝ) : 
  (8 * 5 : ℝ) + (x * 7) + (5 * 5) = 15 * 9 → x = 10 := by
  sorry

end flag_problem_l1221_122188


namespace volleyball_match_probability_l1221_122140

/-- The probability of winning a single set for class 6 of senior year two -/
def win_prob : ℚ := 2/3

/-- The number of sets needed to win the match -/
def sets_to_win : ℕ := 3

/-- The probability of class 6 of senior year two winning by 3:0 -/
def prob_win_3_0 : ℚ := win_prob^sets_to_win

theorem volleyball_match_probability :
  prob_win_3_0 = 8/27 :=
sorry

end volleyball_match_probability_l1221_122140


namespace fraction_subtraction_theorem_l1221_122151

theorem fraction_subtraction_theorem : 
  (3 + 6 + 9) / (2 + 4 + 6) - (2 + 4 + 6) / (3 + 6 + 9) = 5 / 6 := by
  sorry

end fraction_subtraction_theorem_l1221_122151


namespace expression_evaluation_l1221_122166

theorem expression_evaluation :
  let x : ℚ := -3
  let numerator := 5 + x * (5 + x) - 5^2
  let denominator := x - 5 + x^2
  numerator / denominator = -26 := by
sorry

end expression_evaluation_l1221_122166


namespace extremum_condition_increasing_interval_two_roots_condition_l1221_122147

/-- The function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 + a * x^2 + 6 * x

/-- The derivative of f(x) -/
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := x^2 + 2 * a * x + 6

theorem extremum_condition (a : ℝ) : f' a 3 = 0 := by sorry

theorem increasing_interval (a : ℝ) :
  (∀ m : ℝ, (∀ x ∈ Set.Ioo m (m + 2), f' a x > 0) ↔ m ∈ Set.Iic 0 ∪ Set.Ici 3) := by sorry

theorem two_roots_condition (a : ℝ) :
  (∀ m : ℝ, (∃ x y : ℝ, x ∈ Set.Icc 1 3 ∧ y ∈ Set.Icc 1 3 ∧ x ≠ y ∧ f a x + m = 0 ∧ f a y + m = 0) ↔
  m ∈ Set.Ioo (-14/3) (-9/2)) := by sorry

end extremum_condition_increasing_interval_two_roots_condition_l1221_122147


namespace quadratic_max_value_l1221_122115

theorem quadratic_max_value :
  ∃ (M : ℝ), M = 34 ∧ ∀ (q : ℝ), -3 * q^2 + 18 * q + 7 ≤ M :=
by sorry

end quadratic_max_value_l1221_122115


namespace total_students_in_clubs_l1221_122186

def math_club_size : ℕ := 15
def science_club_size : ℕ := 10
def art_club_size : ℕ := 12
def math_science_overlap : ℕ := 5

theorem total_students_in_clubs : 
  math_club_size + science_club_size + art_club_size - math_science_overlap = 32 := by
  sorry

end total_students_in_clubs_l1221_122186


namespace prob_A_wins_one_round_prob_at_least_one_wins_l1221_122165

/-- Probability of A winning exactly one round in a two-round competition -/
theorem prob_A_wins_one_round 
  (p_A1 : ℚ) -- Probability of A winning first round
  (p_A2 : ℚ) -- Probability of A winning second round
  (h_p_A1 : p_A1 = 4/5)
  (h_p_A2 : p_A2 = 2/3) :
  p_A1 * (1 - p_A2) + (1 - p_A1) * p_A2 = 2/5 := by sorry

/-- Probability of at least one of A and B winning a two-round competition -/
theorem prob_at_least_one_wins
  (p_A1 p_A2 p_B1 p_B2 : ℚ) -- Probabilities of A and B winning each round
  (h_p_A1 : p_A1 = 4/5)
  (h_p_A2 : p_A2 = 2/3)
  (h_p_B1 : p_B1 = 3/5)
  (h_p_B2 : p_B2 = 3/4) :
  1 - (1 - p_A1 * p_A2) * (1 - p_B1 * p_B2) = 223/300 := by sorry

end prob_A_wins_one_round_prob_at_least_one_wins_l1221_122165


namespace intersection_implies_m_greater_than_one_l1221_122137

/-- Given a parabola y = x^2 - x + 2 and a line y = x + m, if they intersect at two points, then m > 1 -/
theorem intersection_implies_m_greater_than_one :
  ∀ m : ℝ, (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    (x₁^2 - x₁ + 2 = x₁ + m) ∧ 
    (x₂^2 - x₂ + 2 = x₂ + m)) →
  m > 1 := by
sorry

end intersection_implies_m_greater_than_one_l1221_122137


namespace common_tangent_parabola_log_l1221_122155

theorem common_tangent_parabola_log (a : ℝ) : 
  (∃ x₁ x₂ y : ℝ, 
    y = a * x₁^2 ∧ 
    y = Real.log x₂ ∧ 
    2 * a * x₁ = 2 ∧ 
    1 / x₂ = 2) → 
  a = 1 / Real.log (2 * Real.exp 1) := by
sorry

end common_tangent_parabola_log_l1221_122155


namespace milk_for_six_cookies_l1221_122173

/-- The number of cookies that can be baked with 1 gallon of milk -/
def cookies_per_gallon : ℕ := 24

/-- The number of quarts in a gallon -/
def quarts_per_gallon : ℕ := 4

/-- The number of cookies we want to bake -/
def target_cookies : ℕ := 6

/-- Calculate the amount of milk in quarts needed to bake a given number of cookies -/
def milk_needed (cookies : ℕ) : ℚ :=
  (cookies : ℚ) * (quarts_per_gallon : ℚ) / (cookies_per_gallon : ℚ)

/-- Theorem: The amount of milk needed to bake 6 cookies is 1 quart -/
theorem milk_for_six_cookies :
  milk_needed target_cookies = 1 := by
  sorry

end milk_for_six_cookies_l1221_122173


namespace percent_of_x_is_y_l1221_122133

theorem percent_of_x_is_y (x y : ℝ) (h : 0.6 * (x - y) = 0.2 * (x + y)) : y = 0.5 * x := by
  sorry

end percent_of_x_is_y_l1221_122133


namespace games_played_together_count_l1221_122160

/-- The number of players in the league -/
def totalPlayers : ℕ := 12

/-- The number of players in each game -/
def playersPerGame : ℕ := 6

/-- Function to calculate the number of games two specific players play together -/
def gamesPlayedTogether : ℕ := sorry

theorem games_played_together_count :
  gamesPlayedTogether = 210 := by sorry

end games_played_together_count_l1221_122160


namespace min_dot_product_of_tangents_l1221_122172

-- Define the circles
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 4
def circle_M (x y θ : ℝ) : Prop := (x - 5 * Real.cos θ)^2 + (y - 5 * Real.sin θ)^2 = 1

-- Define a point on circle M
def point_on_M (P : ℝ × ℝ) (θ : ℝ) : Prop := circle_M P.1 P.2 θ

-- Define tangent lines from P to circle O
def tangent_to_O (P E : ℝ × ℝ) : Prop := 
  circle_O E.1 E.2 ∧ (P.1 - E.1) * E.1 + (P.2 - E.2) * E.2 = 0

-- Statement of the theorem
theorem min_dot_product_of_tangents :
  ∀ (P : ℝ × ℝ) (θ : ℝ),
  point_on_M P θ →
  ∃ (E F : ℝ × ℝ),
  tangent_to_O P E ∧ tangent_to_O P F →
  (∀ (E' F' : ℝ × ℝ), tangent_to_O P E' ∧ tangent_to_O P F' →
    ((P.1 - E.1) * (P.1 - F.1) + (P.2 - E.2) * (P.2 - F.2)) ≤
    ((P.1 - E'.1) * (P.1 - F'.1) + (P.2 - E'.2) * (P.2 - F'.2))) ∧
  ((P.1 - E.1) * (P.1 - F.1) + (P.2 - E.2) * (P.2 - F.2)) = 6 :=
sorry

end min_dot_product_of_tangents_l1221_122172


namespace quadratic_root_zero_l1221_122126

theorem quadratic_root_zero (m : ℝ) : 
  (∃ x, (m - 1) * x^2 + x + m^2 - 1 = 0) ∧ 
  ((m - 1) * 0^2 + 0 + m^2 - 1 = 0) → 
  m = -1 := by
sorry

end quadratic_root_zero_l1221_122126


namespace arrangements_theorem_l1221_122177

/-- The number of people in the row -/
def n : ℕ := 6

/-- The number of different arrangements where both person A and person B
    are on the same side of person C -/
def arrangements_same_side (n : ℕ) : ℕ := 2 * (n - 1).factorial

theorem arrangements_theorem :
  arrangements_same_side n = 480 :=
sorry

end arrangements_theorem_l1221_122177


namespace james_payment_l1221_122167

/-- The cost of cable program for James and his roommate -/
def cable_cost (first_100_cost : ℕ) (total_channels : ℕ) : ℕ :=
  if total_channels ≤ 100 then
    first_100_cost
  else
    first_100_cost + (first_100_cost / 2)

/-- James' share of the cable cost -/
def james_share (total_cost : ℕ) : ℕ := total_cost / 2

theorem james_payment (first_100_cost : ℕ) (total_channels : ℕ) :
  first_100_cost = 100 →
  total_channels = 200 →
  james_share (cable_cost first_100_cost total_channels) = 75 := by
  sorry

#eval james_share (cable_cost 100 200)

end james_payment_l1221_122167


namespace elephant_donkey_weight_l1221_122129

/-- Calculates the combined weight of an elephant and a donkey in pounds -/
theorem elephant_donkey_weight (elephant_tons : ℝ) (donkey_percent_less : ℝ) : 
  elephant_tons = 3 ∧ donkey_percent_less = 90 →
  elephant_tons * 2000 + (elephant_tons * 2000 * (1 - donkey_percent_less / 100)) = 6600 := by
  sorry

end elephant_donkey_weight_l1221_122129


namespace equal_roots_quadratic_l1221_122122

theorem equal_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, x^2 - 2 * Real.sqrt 3 * x + k = 0 ∧ 
   ∀ y : ℝ, y^2 - 2 * Real.sqrt 3 * y + k = 0 → y = x) → 
  k = 3 := by
  sorry

end equal_roots_quadratic_l1221_122122


namespace highest_a_divisible_by_8_l1221_122107

def is_divisible_by_8 (n : ℕ) : Prop := n % 8 = 0

def last_three_digits (n : ℕ) : ℕ := n % 1000

theorem highest_a_divisible_by_8 :
  ∀ a : ℕ, a ≤ 9 →
    (is_divisible_by_8 (365 * 100 + a * 10 + 16) ↔ a ≤ 8) ∧
    (∀ b : ℕ, b > 8 ∧ b ≤ 9 → ¬ is_divisible_by_8 (365 * 100 + b * 10 + 16)) :=
sorry

end highest_a_divisible_by_8_l1221_122107


namespace x_is_negative_l1221_122132

theorem x_is_negative (x y : ℝ) (h1 : y ≠ 0) (h2 : y > 0) (h3 : x / y < -3) : x < 0 := by
  sorry

end x_is_negative_l1221_122132


namespace triangle_side_length_l1221_122189

-- Define the triangle PQR
structure Triangle :=
  (P Q R : ℝ × ℝ)

-- Define the properties of the triangle
def is_valid_triangle (t : Triangle) (pq pr pn : ℝ) : Prop :=
  let (px, py) := t.P
  let (qx, qy) := t.Q
  let (rx, ry) := t.R
  let (nx, ny) := ((qx + rx) / 2, (qy + ry) / 2)  -- N is midpoint of QR
  (px - qx)^2 + (py - qy)^2 = pq^2 ∧  -- PQ = 6
  (px - rx)^2 + (py - ry)^2 = pr^2 ∧  -- PR = 10
  (px - nx)^2 + (py - ny)^2 = pn^2    -- PN = 5

-- Theorem statement
theorem triangle_side_length (t : Triangle) :
  is_valid_triangle t 6 10 5 →
  let (qx, qy) := t.Q
  let (rx, ry) := t.R
  (qx - rx)^2 + (qy - ry)^2 = 4 * 43 :=
by sorry

end triangle_side_length_l1221_122189


namespace student_transfer_fraction_l1221_122154

theorem student_transfer_fraction (initial_students new_students final_students : ℕ) : 
  initial_students = 160 →
  new_students = 20 →
  final_students = 120 →
  (initial_students + new_students - final_students) / (initial_students + new_students) = 1 / 3 := by
sorry

end student_transfer_fraction_l1221_122154


namespace geometric_sequence_second_term_l1221_122113

/-- Given a geometric sequence where the fifth term is 48 and the sixth term is 72,
    prove that the second term of the sequence is 1152/81. -/
theorem geometric_sequence_second_term
  (a : ℚ) -- First term of the sequence
  (r : ℚ) -- Common ratio of the sequence
  (h1 : a * r^4 = 48) -- Fifth term is 48
  (h2 : a * r^5 = 72) -- Sixth term is 72
  : a * r = 1152 / 81 := by
sorry

end geometric_sequence_second_term_l1221_122113


namespace angle_in_second_quadrant_l1221_122181

-- Define the quadrant type
inductive Quadrant
  | first
  | second
  | third
  | fourth

-- Define the function to determine the quadrant of an angle
def angle_quadrant (θ : Real) : Quadrant :=
  sorry

-- Theorem statement
theorem angle_in_second_quadrant (θ : Real) 
  (h1 : Real.sin θ > Real.cos θ) 
  (h2 : Real.tan θ < 0) : 
  angle_quadrant θ = Quadrant.second :=
sorry

end angle_in_second_quadrant_l1221_122181


namespace expression_evaluation_l1221_122128

theorem expression_evaluation :
  let x : ℚ := 3
  let f (y : ℚ) := (y + 3) / (y - 2)
  3 * (f (f x) + 3) / (f (f x) - 2) = 27 / 4 := by
  sorry

end expression_evaluation_l1221_122128


namespace hyperbola_range_l1221_122142

/-- A function that represents the equation of a hyperbola -/
def hyperbola_equation (m : ℝ) (x y : ℝ) : Prop :=
  x^2 / (m + 2) - y^2 / (2*m - 1) = 1

/-- The theorem stating the range of m for which the equation represents a hyperbola -/
theorem hyperbola_range (m : ℝ) :
  (∀ x y, ∃ (h : hyperbola_equation m x y), True) ↔ m < -2 ∨ m > 1/2 := by
  sorry

end hyperbola_range_l1221_122142


namespace total_games_in_season_l1221_122184

/-- The number of games played in a hockey league season -/
def hockey_league_games (n : ℕ) (games_per_pair : ℕ) : ℕ :=
  games_per_pair * (n * (n - 1) / 2)

/-- Theorem: In a league with 12 teams, where each team plays 4 games with each other team,
    the total number of games played is 264 -/
theorem total_games_in_season :
  hockey_league_games 12 4 = 264 := by
  sorry

end total_games_in_season_l1221_122184


namespace snail_track_time_equivalence_l1221_122171

theorem snail_track_time_equivalence (clockwise_time : Real) (counterclockwise_time : Real) : 
  clockwise_time = 1.5 → counterclockwise_time = 90 → clockwise_time * 60 = counterclockwise_time :=
by
  sorry

end snail_track_time_equivalence_l1221_122171


namespace union_of_sets_l1221_122134

theorem union_of_sets : let A : Set ℕ := {2, 3}
                        let B : Set ℕ := {1, 2}
                        A ∪ B = {1, 2, 3} := by
  sorry

end union_of_sets_l1221_122134


namespace no_real_solution_l1221_122104

theorem no_real_solution :
  ¬∃ (x y : ℝ), 4 * x^2 + 9 * y^2 - 16 * x - 36 * y + 64 = 0 := by
sorry

end no_real_solution_l1221_122104


namespace triangle_inequality_l1221_122130

/-- Given a triangle ABC with area t, perimeter k, and circumradius R, 
    prove that 4tR ≤ (k/3)³ -/
theorem triangle_inequality (t k R : ℝ) (h_positive : t > 0 ∧ k > 0 ∧ R > 0) :
  4 * t * R ≤ (k / 3) ^ 3 := by
  sorry

end triangle_inequality_l1221_122130


namespace polynomial_factorization_l1221_122196

theorem polynomial_factorization (x : ℝ) :
  x^2 - 6*x + 9 - 64*x^4 = (-8*x^2 + x - 3)*(8*x^2 + x - 3) := by
  sorry

end polynomial_factorization_l1221_122196


namespace jenny_sold_192_packs_l1221_122193

-- Define the number of boxes sold
def boxes_sold : Float := 24.0

-- Define the number of packs per box
def packs_per_box : Float := 8.0

-- Define the total number of packs sold
def total_packs : Float := boxes_sold * packs_per_box

-- Theorem statement
theorem jenny_sold_192_packs : total_packs = 192.0 := by
  sorry

end jenny_sold_192_packs_l1221_122193


namespace remainder_equivalence_l1221_122183

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

/-- Theorem: The remainder when dividing a number by 3 or 9 is the same as
    the remainder when dividing the sum of its digits by 3 or 9 -/
theorem remainder_equivalence (n : ℕ) :
  (n % 3 = sum_of_digits n % 3) ∧ (n % 9 = sum_of_digits n % 9) :=
sorry

end remainder_equivalence_l1221_122183


namespace product_of_reciprocal_minus_one_geq_eight_l1221_122149

theorem product_of_reciprocal_minus_one_geq_eight (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 1) :
  (1 / a - 1) * (1 / b - 1) * (1 / c - 1) ≥ 8 := by
  sorry

end product_of_reciprocal_minus_one_geq_eight_l1221_122149


namespace sum_and_one_known_l1221_122161

theorem sum_and_one_known (x y : ℤ) : x + y = -26 ∧ x = 11 → y = -37 := by
  sorry

end sum_and_one_known_l1221_122161


namespace lizzie_wins_iff_composite_l1221_122168

/-- The game state represents the numbers on the blackboard -/
def GameState := List ℚ

/-- A move in the game is selecting a subset of numbers and replacing them with their average -/
def Move := List ℕ

/-- Represents whether a number is composite -/
def IsComposite (n : ℕ) : Prop := ∃ k, 1 < k ∧ k < n ∧ k ∣ n

/-- Applies a move to the game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  sorry

/-- Checks if all numbers in the game state are equal -/
def allEqual (state : GameState) : Prop :=
  sorry

/-- Represents a winning strategy for Lizzie -/
def WinningStrategy (n : ℕ) : Prop :=
  ∀ initialState : GameState, 
    initialState.length = n →
    ∃ moveSequence : List Move, 
      allEqual (moveSequence.foldl applyMove initialState)

theorem lizzie_wins_iff_composite (n : ℕ) (h : n ≥ 3) :
  WinningStrategy n ↔ IsComposite n :=
sorry

end lizzie_wins_iff_composite_l1221_122168


namespace fewer_buses_on_river_road_l1221_122119

theorem fewer_buses_on_river_road (num_cars : ℕ) (num_buses : ℕ) : 
  num_cars = 60 →
  num_buses < num_cars →
  num_buses * 3 = num_cars →
  num_cars - num_buses = 40 := by
  sorry

end fewer_buses_on_river_road_l1221_122119


namespace function_property_l1221_122108

/-- Given a function f(x) = ax^5 + bx^3 + 2 where f(2) = 7, prove that f(-2) = -3 -/
theorem function_property (a b : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ a * x^5 + b * x^3 + 2
  (f 2 = 7) → (f (-2) = -3) := by
sorry

end function_property_l1221_122108


namespace ice_cream_sundae_combinations_l1221_122197

/-- The number of different kinds of ice cream. -/
def n : ℕ := 8

/-- The number of scoops in a sundae. -/
def k : ℕ := 2

/-- The number of unique two-scoop sundaes with different ice cream flavors. -/
def different_flavors : ℕ := n.choose k

/-- The number of unique two-scoop sundaes with identical ice cream flavors. -/
def identical_flavors : ℕ := n

/-- The total number of unique two-scoop sundaes. -/
def total_sundaes : ℕ := different_flavors + identical_flavors

theorem ice_cream_sundae_combinations :
  total_sundaes = 36 := by
  sorry

end ice_cream_sundae_combinations_l1221_122197


namespace geometric_sequence_sum_l1221_122191

theorem geometric_sequence_sum (a r : ℝ) (h1 : a * (1 - r^1000) / (1 - r) = 1024) 
  (h2 : a * (1 - r^2000) / (1 - r) = 2040) : 
  a * (1 - r^3000) / (1 - r) = 3048 := by
  sorry

end geometric_sequence_sum_l1221_122191


namespace mangoes_rate_per_kg_l1221_122144

/-- Given Bruce's purchase of grapes and mangoes, prove the rate per kg for mangoes. -/
theorem mangoes_rate_per_kg 
  (grapes_quantity : ℕ) 
  (grapes_rate : ℕ) 
  (mangoes_quantity : ℕ) 
  (total_paid : ℕ) 
  (h1 : grapes_quantity = 8)
  (h2 : grapes_rate = 70)
  (h3 : mangoes_quantity = 11)
  (h4 : total_paid = 1165)
  (h5 : total_paid = grapes_quantity * grapes_rate + mangoes_quantity * (total_paid - grapes_quantity * grapes_rate) / mangoes_quantity) : 
  (total_paid - grapes_quantity * grapes_rate) / mangoes_quantity = 55 := by
  sorry

end mangoes_rate_per_kg_l1221_122144


namespace almond_walnut_ratio_is_five_to_two_l1221_122109

/-- Represents a mixture of nuts with almonds and walnuts. -/
structure NutMixture where
  total_weight : ℝ
  almond_weight : ℝ
  almond_parts : ℝ
  walnut_parts : ℝ

/-- The ratio of almonds to walnuts in the mixture. -/
def almond_walnut_ratio (mix : NutMixture) : ℝ × ℝ :=
  (mix.almond_parts, mix.walnut_parts)

theorem almond_walnut_ratio_is_five_to_two
  (mix : NutMixture)
  (h1 : mix.total_weight = 350)
  (h2 : mix.almond_weight = 250)
  (h3 : mix.walnut_parts = 2)
  (h4 : mix.almond_parts * mix.walnut_parts = mix.almond_weight * mix.walnut_parts) :
  almond_walnut_ratio mix = (5, 2) := by
  sorry

end almond_walnut_ratio_is_five_to_two_l1221_122109


namespace sum_of_integers_30_to_50_l1221_122199

def sum_of_integers (a b : ℕ) : ℕ := (b - a + 1) * (a + b) / 2

def count_even_integers (a b : ℕ) : ℕ := (b - a) / 2 + 1

theorem sum_of_integers_30_to_50 (x y : ℕ) :
  x = sum_of_integers 30 50 →
  y = count_even_integers 30 50 →
  x + y = 851 →
  x = 840 := by sorry

end sum_of_integers_30_to_50_l1221_122199


namespace scientific_notation_of_15000_l1221_122136

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a positive real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_of_15000 :
  toScientificNotation 15000 = ScientificNotation.mk 1.5 4 (by norm_num) :=
sorry

end scientific_notation_of_15000_l1221_122136


namespace car_sales_profit_percentage_l1221_122162

/-- Represents the sale of a car with its selling price and profit/loss percentage -/
structure CarSale where
  selling_price : ℝ
  profit_percentage : ℝ

/-- Calculates the overall profit percentage for a list of car sales -/
def overall_profit_percentage (sales : List CarSale) : ℝ :=
  sorry

/-- The main theorem stating the overall profit percentage for the given car sales -/
theorem car_sales_profit_percentage : 
  let sales := [
    CarSale.mk 404415 15,
    CarSale.mk 404415 (-15),
    CarSale.mk 550000 10
  ]
  abs (overall_profit_percentage sales - 2.36) < 0.01 := by
  sorry

end car_sales_profit_percentage_l1221_122162


namespace affordable_housing_theorem_l1221_122112

/-- Represents the affordable housing investment and construction scenario -/
structure AffordableHousing where
  investment_2011 : ℝ
  area_2011 : ℝ
  total_investment : ℝ
  growth_rate : ℝ

/-- The affordable housing scenario satisfies the given conditions -/
def valid_scenario (ah : AffordableHousing) : Prop :=
  ah.investment_2011 = 200 ∧
  ah.area_2011 = 0.08 ∧
  ah.total_investment = 950 ∧
  ah.investment_2011 * (1 + ah.growth_rate + (1 + ah.growth_rate)^2) = ah.total_investment

/-- The growth rate is 50% and the total area built is 38 million square meters -/
theorem affordable_housing_theorem (ah : AffordableHousing) 
  (h : valid_scenario ah) : 
  ah.growth_rate = 0.5 ∧ 
  ah.total_investment / (ah.investment_2011 / ah.area_2011) = 38 := by
  sorry


end affordable_housing_theorem_l1221_122112


namespace farm_entrance_fee_l1221_122125

theorem farm_entrance_fee (num_students : ℕ) (num_adults : ℕ) (student_fee : ℕ) (total_cost : ℕ) :
  num_students = 35 →
  num_adults = 4 →
  student_fee = 5 →
  total_cost = 199 →
  ∃ (adult_fee : ℕ), 
    adult_fee = 6 ∧ 
    num_students * student_fee + num_adults * adult_fee = total_cost :=
by sorry

end farm_entrance_fee_l1221_122125


namespace sum_even_number_of_even_is_even_sum_even_number_of_odd_is_even_sum_odd_number_of_even_is_even_sum_odd_number_of_odd_is_odd_l1221_122156

-- Define what it means for a number to be even
def IsEven (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

-- Define what it means for a number to be odd
def IsOdd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

-- Define a function that sums a list of integers
def SumList (list : List ℤ) : ℤ := list.foldl (· + ·) 0

-- Theorem 1: Sum of an even number of even integers is even
theorem sum_even_number_of_even_is_even (n : ℕ) (list : List ℤ) 
  (h1 : list.length = 2 * n) 
  (h2 : ∀ x ∈ list, IsEven x) : 
  IsEven (SumList list) := by sorry

-- Theorem 2: Sum of an even number of odd integers is even
theorem sum_even_number_of_odd_is_even (n : ℕ) (list : List ℤ) 
  (h1 : list.length = 2 * n) 
  (h2 : ∀ x ∈ list, IsOdd x) : 
  IsEven (SumList list) := by sorry

-- Theorem 3: Sum of an odd number of even integers is even
theorem sum_odd_number_of_even_is_even (n : ℕ) (list : List ℤ) 
  (h1 : list.length = 2 * n + 1) 
  (h2 : ∀ x ∈ list, IsEven x) : 
  IsEven (SumList list) := by sorry

-- Theorem 4: Sum of an odd number of odd integers is odd
theorem sum_odd_number_of_odd_is_odd (n : ℕ) (list : List ℤ) 
  (h1 : list.length = 2 * n + 1) 
  (h2 : ∀ x ∈ list, IsOdd x) : 
  IsOdd (SumList list) := by sorry

end sum_even_number_of_even_is_even_sum_even_number_of_odd_is_even_sum_odd_number_of_even_is_even_sum_odd_number_of_odd_is_odd_l1221_122156


namespace tangent_line_at_one_monotonic_increase_intervals_l1221_122192

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - x + 3

-- Theorem for the tangent line equation
theorem tangent_line_at_one :
  ∃ (a b c : ℝ), a ≠ 0 ∧ 
  (∀ x y : ℝ, y = f x → (x = 1 → a*x + b*y + c = 0)) ∧
  a = 2 ∧ b = -1 ∧ c = 1 := by sorry

-- Theorem for the intervals of monotonic increase
theorem monotonic_increase_intervals :
  ∃ (a : ℝ), a > 0 ∧
  (∀ x : ℝ, (x < -a ∨ x > a) → (∀ y : ℝ, x < y → f x < f y)) ∧
  a = Real.sqrt 3 / 3 := by sorry

end tangent_line_at_one_monotonic_increase_intervals_l1221_122192


namespace probability_yellow_ball_l1221_122187

theorem probability_yellow_ball (total_balls : ℕ) (yellow_balls : ℕ) 
  (h1 : total_balls = 8) (h2 : yellow_balls = 5) :
  (yellow_balls : ℚ) / total_balls = 5 / 8 := by
  sorry

end probability_yellow_ball_l1221_122187


namespace floor_of_negative_decimal_l1221_122176

theorem floor_of_negative_decimal (x : ℝ) : x = -3.7 → ⌊x⌋ = -4 := by
  sorry

end floor_of_negative_decimal_l1221_122176


namespace day_of_week_theorem_l1221_122169

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a year -/
structure Year where
  number : ℕ

/-- Returns the day of the week for a given day number in a year -/
def dayOfWeek (y : Year) (day : ℕ) : DayOfWeek := sorry

/-- Checks if a year is a leap year -/
def isLeapYear (y : Year) : Bool := sorry

/-- The main theorem to prove -/
theorem day_of_week_theorem (N : Year) :
  dayOfWeek N 250 = DayOfWeek.Wednesday →
  dayOfWeek (Year.mk (N.number + 1)) 150 = DayOfWeek.Wednesday →
  dayOfWeek (Year.mk (N.number - 1)) 50 = DayOfWeek.Monday := by
  sorry

end day_of_week_theorem_l1221_122169


namespace f_recursion_l1221_122153

/-- A function that computes the sum of binomial coefficients (n choose i) where k divides (n-2i) -/
def f (k : ℕ) (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).sum (λ i => if k ∣ (n - 2*i) then Nat.choose n i else 0)

/-- Theorem stating the recursion relation for f_n -/
theorem f_recursion (k : ℕ) (n : ℕ) (h : k > 1) (h_odd : Odd k) :
  (f k n)^2 = (Finset.range (n + 1)).sum (λ i => Nat.choose n i * f k i * f k (n - i)) := by
  sorry

end f_recursion_l1221_122153


namespace largest_integer_solution_l1221_122101

theorem largest_integer_solution : ∃ (x : ℤ), x ≤ 20 ∧ |x - 3| = 15 ∧ ∀ (y : ℤ), y ≤ 20 ∧ |y - 3| = 15 → y ≤ x :=
by
  -- Proof goes here
  sorry

end largest_integer_solution_l1221_122101


namespace gaochun_temperature_difference_l1221_122190

def temperature_difference (low high : Int) : Int :=
  high - low

theorem gaochun_temperature_difference :
  let low : Int := -2
  let high : Int := 9
  temperature_difference low high = 11 := by
  sorry

end gaochun_temperature_difference_l1221_122190


namespace min_distance_between_curve_and_line_l1221_122124

theorem min_distance_between_curve_and_line :
  ∀ (a b c d : ℝ),
  (Real.log b + 1 + a - 3 * b = 0) →
  (2 * d - c + Real.sqrt 5 = 0) →
  (∃ (m : ℝ), ∀ (a' b' c' d' : ℝ),
    (Real.log b' + 1 + a' - 3 * b' = 0) →
    (2 * d' - c' + Real.sqrt 5 = 0) →
    (a - c)^2 + (b - d)^2 ≤ (a' - c')^2 + (b' - d')^2) →
  m = 4/5 := by
sorry

end min_distance_between_curve_and_line_l1221_122124


namespace decagon_equilateral_triangles_l1221_122105

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- An equilateral triangle -/
structure EquilateralTriangle where
  vertices : Fin 3 → ℝ × ℝ

/-- Count of distinct equilateral triangles in a regular polygon -/
def countDistinctEquilateralTriangles (n : ℕ) (p : RegularPolygon n) : ℕ :=
  sorry

/-- Theorem: In a ten-sided regular polygon, there are 82 distinct equilateral triangles
    with at least two vertices from the set of polygon vertices -/
theorem decagon_equilateral_triangles :
  ∀ (p : RegularPolygon 10), countDistinctEquilateralTriangles 10 p = 82 :=
by sorry

end decagon_equilateral_triangles_l1221_122105


namespace min_side_length_l1221_122121

theorem min_side_length (PQ PR SR SQ : ℝ) (h1 : PQ = 7) (h2 : PR = 15) (h3 : SR = 10) (h4 : SQ = 25) :
  ∀ QR : ℝ, (QR > PR - PQ ∧ QR > SQ - SR) → QR ≥ 15 := by
  sorry

end min_side_length_l1221_122121


namespace selling_price_is_correct_l1221_122103

/-- Calculates the selling price per copy of a program given the production cost,
    advertisement revenue, number of copies to be sold, and desired profit. -/
def calculate_selling_price (production_cost : ℚ) (ad_revenue : ℚ) (copies : ℕ) (desired_profit : ℚ) : ℚ :=
  (desired_profit + (production_cost * copies) - ad_revenue) / copies

theorem selling_price_is_correct : 
  let production_cost : ℚ := 70/100
  let ad_revenue : ℚ := 15000
  let copies : ℕ := 35000
  let desired_profit : ℚ := 8000
  calculate_selling_price production_cost ad_revenue copies desired_profit = 1/2 := by
  sorry

end selling_price_is_correct_l1221_122103


namespace bill_selling_price_l1221_122195

theorem bill_selling_price (P : ℝ) 
  (h1 : P + 0.1 * P = 1.1 * P)  -- Original selling price
  (h2 : 0.9 * P + 0.3 * (0.9 * P) = 1.17 * P)  -- New selling price with 30% profit
  (h3 : 1.17 * P = 1.1 * P + 42)  -- Equation relating the two selling prices
  : 1.1 * P = 660 := by
  sorry

end bill_selling_price_l1221_122195


namespace representative_selection_cases_l1221_122106

def number_of_female_students : ℕ := 4
def number_of_male_students : ℕ := 6

theorem representative_selection_cases :
  (number_of_female_students * number_of_male_students) = 24 :=
by sorry

end representative_selection_cases_l1221_122106


namespace product_plus_245_divisible_by_5_l1221_122178

theorem product_plus_245_divisible_by_5 : ∃ k : ℤ, (1250 * 1625 * 1830 * 2075 + 245 : ℤ) = 5 * k := by
  sorry

end product_plus_245_divisible_by_5_l1221_122178


namespace hyperbola_tangent_coincidence_l1221_122194

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 9 - y^2 / 4 = 1

-- Define the curve
def curve (a x y : ℝ) : Prop := y = a * x^2 + 1/3

-- Define the asymptotes of the hyperbola
def asymptotes (x y : ℝ) : Prop := y = 2/3 * x ∨ y = -2/3 * x

-- Define the condition that asymptotes coincide with tangents
def coincide_with_tangents (a : ℝ) : Prop :=
  ∀ x y : ℝ, asymptotes x y → ∃ t : ℝ, curve a t y ∧ 
  (∀ s : ℝ, s ≠ t → curve a s (a * s^2 + 1/3) → (a * s^2 + 1/3 - y) * (s - t) > 0)

-- Theorem statement
theorem hyperbola_tangent_coincidence :
  ∀ a : ℝ, coincide_with_tangents a → a = 1/3 :=
sorry

end hyperbola_tangent_coincidence_l1221_122194


namespace sqrt_two_thirds_less_than_half_l1221_122146

theorem sqrt_two_thirds_less_than_half : (Real.sqrt 2) / 3 < 1 / 2 := by
  sorry

end sqrt_two_thirds_less_than_half_l1221_122146


namespace function_transformation_l1221_122170

theorem function_transformation (f : ℝ → ℝ) : 
  (∀ x, f (2 * x - 1) = x^2 - x) → 
  (∀ x, f x = (1/4) * (x^2 - 1)) := by
sorry

end function_transformation_l1221_122170
