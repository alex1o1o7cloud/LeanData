import Mathlib

namespace minimal_value_of_f_l3118_311868

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem minimal_value_of_f :
  ∃ (x_min : ℝ), f x_min = Real.exp (-1) ∧ ∀ (x : ℝ), f x ≥ Real.exp (-1) :=
sorry

end minimal_value_of_f_l3118_311868


namespace wheel_speed_is_seven_l3118_311829

noncomputable def wheel_speed (circumference : Real) (r : Real) : Prop :=
  let miles_per_rotation := circumference / 5280
  let t := miles_per_rotation / r
  let new_t := t - 1 / (3 * 3600)
  (r + 3) * new_t = miles_per_rotation

theorem wheel_speed_is_seven :
  ∀ (r : Real),
    wheel_speed 15 r →
    r = 7 :=
by
  sorry

end wheel_speed_is_seven_l3118_311829


namespace f_contraction_implies_a_bound_l3118_311821

open Real

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * log x - x^2 + 1

-- State the theorem
theorem f_contraction_implies_a_bound
  (a : ℝ)
  (h_a_neg : a < 0)
  (h_contraction : ∀ (x₁ x₂ : ℝ), x₁ > 0 → x₂ > 0 →
    |f a x₁ - f a x₂| ≥ |x₁ - x₂|) :
  a ≤ -1/8 := by
  sorry

end f_contraction_implies_a_bound_l3118_311821


namespace odd_function_negative_domain_l3118_311817

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_negative_domain
  (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_positive : ∀ x > 0, f x = -x + 2) :
  ∀ x < 0, f x = -x - 2 := by
sorry

end odd_function_negative_domain_l3118_311817


namespace lamp_configurations_l3118_311830

/-- Represents the number of reachable configurations for n lamps -/
def reachableConfigurations (n : ℕ) : ℕ :=
  if n % 3 = 0 then 2^(n-2) else 2^n

/-- Theorem stating the number of reachable configurations for n cyclically connected lamps -/
theorem lamp_configurations (n : ℕ) (h : n > 2) :
  reachableConfigurations n = if n % 3 = 0 then 2^(n-2) else 2^n :=
by sorry

end lamp_configurations_l3118_311830


namespace function_identity_l3118_311814

theorem function_identity (f : ℝ → ℝ) 
  (h₁ : f 0 = 1)
  (h₂ : ∀ x y : ℝ, f (x * y + 1) = f x * f y - f y - x + 2) :
  ∀ x : ℝ, f x = x + 1 := by
  sorry

end function_identity_l3118_311814


namespace gcd_1021_2729_l3118_311820

theorem gcd_1021_2729 : Nat.gcd 1021 2729 = 1 := by
  sorry

end gcd_1021_2729_l3118_311820


namespace box_surface_area_l3118_311818

/-- Proves that the surface area of a rectangular box is 975 given specific conditions -/
theorem box_surface_area (a b c : ℝ) 
  (edge_sum : 4 * a + 4 * b + 4 * c = 160)
  (diagonal : Real.sqrt (a^2 + b^2 + c^2) = 25)
  (volume : a * b * c = 600) :
  2 * (a * b + b * c + c * a) = 975 := by
sorry

end box_surface_area_l3118_311818


namespace complex_product_real_l3118_311806

theorem complex_product_real (a : ℝ) :
  (Complex.I : ℂ) * (Complex.I : ℂ) = -1 →
  (((a : ℂ) - 2 * Complex.I) * (3 + Complex.I)).im = 0 ↔ a = 6 :=
sorry

end complex_product_real_l3118_311806


namespace regular_polygon_sides_l3118_311846

/-- A regular polygon with exterior angles measuring 18 degrees has 20 sides. -/
theorem regular_polygon_sides (n : ℕ) : n > 0 → (360 : ℝ) / n = 18 → n = 20 := by
  sorry

end regular_polygon_sides_l3118_311846


namespace average_player_minutes_is_two_l3118_311858

/-- Represents the highlight film about Patricia's basketball team. -/
structure HighlightFilm where
  /-- Footage duration for each player in seconds -/
  point_guard : ℕ
  shooting_guard : ℕ
  small_forward : ℕ
  power_forward : ℕ
  center : ℕ
  /-- Additional content durations in seconds -/
  game_footage : ℕ
  interviews : ℕ
  opening_closing : ℕ
  /-- Pause duration between segments in seconds -/
  pause_duration : ℕ

/-- Calculates the average number of minutes attributed to each player's footage -/
def averagePlayerMinutes (film : HighlightFilm) : ℚ :=
  let total_player_footage := film.point_guard + film.shooting_guard + film.small_forward + 
                              film.power_forward + film.center
  let total_additional_content := film.game_footage + film.interviews + film.opening_closing
  let total_pause_time := film.pause_duration * 8
  let total_film_time := total_player_footage + total_additional_content + total_pause_time
  (total_player_footage : ℚ) / (5 * 60)

/-- Theorem stating that the average number of minutes attributed to each player's footage is 2 minutes -/
theorem average_player_minutes_is_two (film : HighlightFilm) 
  (h1 : film.point_guard = 130)
  (h2 : film.shooting_guard = 145)
  (h3 : film.small_forward = 85)
  (h4 : film.power_forward = 60)
  (h5 : film.center = 180)
  (h6 : film.game_footage = 120)
  (h7 : film.interviews = 90)
  (h8 : film.opening_closing = 30)
  (h9 : film.pause_duration = 15) :
  averagePlayerMinutes film = 2 := by
  sorry

end average_player_minutes_is_two_l3118_311858


namespace largest_divisor_of_2n3_minus_2n_l3118_311844

theorem largest_divisor_of_2n3_minus_2n (n : ℤ) : 
  (∃ (k : ℤ), 2 * n^3 - 2 * n = 12 * k) ∧ 
  (∀ (m : ℤ), m > 12 → ∃ (l : ℤ), 2 * l^3 - 2 * l ≠ m * (2 * l^3 - 2 * l) / m) :=
sorry

end largest_divisor_of_2n3_minus_2n_l3118_311844


namespace participation_difference_l3118_311889

def participants_2018 : ℕ := 150

def participants_2019 : ℕ := 2 * participants_2018 + 20

def participants_2020 : ℕ := participants_2019 / 2 - 40

def participants_2021 : ℕ := 30 + (participants_2018 - participants_2020)

theorem participation_difference : participants_2019 - participants_2020 = 200 := by
  sorry

end participation_difference_l3118_311889


namespace cos_double_angle_problem_l3118_311828

theorem cos_double_angle_problem (α : Real) : 
  (π/2 < α ∧ α < π) →  -- α is in the second quadrant
  (Real.sin α + Real.cos α = Real.sqrt 3 / 3) → 
  Real.cos (2 * α) = -(Real.sqrt 5 / 3) := by
sorry

end cos_double_angle_problem_l3118_311828


namespace klinker_age_problem_l3118_311878

/-- The age difference between Mr. Klinker and his daughter remains constant -/
theorem klinker_age_problem (klinker_age : ℕ) (daughter_age : ℕ) (years : ℕ) :
  klinker_age = 47 →
  daughter_age = 13 →
  klinker_age + years = 3 * (daughter_age + years) →
  years = 4 :=
by
  sorry

end klinker_age_problem_l3118_311878


namespace grid_30_8_uses_598_toothpicks_l3118_311872

/-- Calculates the total number of toothpicks in a reinforced rectangular grid. -/
def toothpicks_in_grid (height : ℕ) (width : ℕ) : ℕ :=
  let internal_horizontal := (height + 1) * width
  let internal_vertical := (width + 1) * height
  let external_horizontal := 2 * width
  let external_vertical := 2 * (height + 2)
  internal_horizontal + internal_vertical + external_horizontal + external_vertical

/-- Theorem stating that a reinforced rectangular grid of 30x8 uses 598 toothpicks. -/
theorem grid_30_8_uses_598_toothpicks :
  toothpicks_in_grid 30 8 = 598 := by
  sorry

#eval toothpicks_in_grid 30 8

end grid_30_8_uses_598_toothpicks_l3118_311872


namespace min_value_sum_product_l3118_311824

theorem min_value_sum_product (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a + b + c + d) * (1 / (a + b) + 1 / (a + c) + 1 / (b + d) + 1 / (c + d)) ≥ 8 := by
  sorry

end min_value_sum_product_l3118_311824


namespace l_shaped_area_l3118_311834

/-- The area of an L-shaped region formed by subtracting three squares from a larger square -/
theorem l_shaped_area (a b c d : ℕ) (h1 : a = 7) (h2 : b = 2) (h3 : c = 2) (h4 : d = 3) : 
  a^2 - (b^2 + c^2 + d^2) = 32 := by
  sorry

end l_shaped_area_l3118_311834


namespace subtracted_amount_l3118_311860

theorem subtracted_amount (N : ℝ) (A : ℝ) : 
  N = 200 → 0.4 * N - A = 50 → A = 30 := by
  sorry

end subtracted_amount_l3118_311860


namespace touch_football_point_difference_l3118_311862

/-- The point difference between two teams in a touch football game -/
def point_difference (
  touchdown_points : ℕ)
  (extra_point_points : ℕ)
  (field_goal_points : ℕ)
  (team1_touchdowns : ℕ)
  (team1_extra_points : ℕ)
  (team1_field_goals : ℕ)
  (team2_touchdowns : ℕ)
  (team2_extra_points : ℕ)
  (team2_field_goals : ℕ) : ℕ :=
  (team2_touchdowns * touchdown_points +
   team2_extra_points * extra_point_points +
   team2_field_goals * field_goal_points) -
  (team1_touchdowns * touchdown_points +
   team1_extra_points * extra_point_points +
   team1_field_goals * field_goal_points)

theorem touch_football_point_difference :
  point_difference 7 1 3 6 4 2 8 6 3 = 19 := by
  sorry

end touch_football_point_difference_l3118_311862


namespace A_inter_B_a_upper_bound_a_sufficient_l3118_311890

-- Define the sets A, B, and C
def A : Set ℝ := {x | 1 < x ∧ x ≤ 5}
def B : Set ℝ := {x | (2*x - 1)/(x - 3) > 0}
def C (a : ℝ) : Set ℝ := {x | a + 1 ≤ x ∧ x ≤ 4*a - 3}

-- Theorem for A ∩ B
theorem A_inter_B : A ∩ B = {x : ℝ | 3 < x ∧ x ≤ 5} := by sorry

-- Theorem for the upper bound of a
theorem a_upper_bound (a : ℝ) (h : C a ∪ A = A) : a ≤ 2 := by sorry

-- Theorem for the sufficiency of a ≤ 2
theorem a_sufficient (a : ℝ) (h : a ≤ 2) : C a ∪ A = A := by sorry

end A_inter_B_a_upper_bound_a_sufficient_l3118_311890


namespace red_points_centroid_theorem_l3118_311807

/-- Represents a point in a 2D grid -/
structure GridPoint where
  x : Int
  y : Int

/-- Represents a line in a 2D grid -/
inductive GridLine
  | Horizontal (y : Int)
  | Vertical (x : Int)

/-- Definition of a grid -/
structure Grid where
  size : Nat
  horizontal_lines : List GridLine
  vertical_lines : List GridLine

/-- Definition of a triangle -/
structure Triangle where
  a : GridPoint
  b : GridPoint
  c : GridPoint

/-- Calculates the centroid of a triangle -/
def centroid (t : Triangle) : GridPoint :=
  { x := (t.a.x + t.b.x + t.c.x) / 3,
    y := (t.a.y + t.b.y + t.c.y) / 3 }

/-- Theorem statement -/
theorem red_points_centroid_theorem (m : Nat) (grid : Grid)
  (h1 : grid.size = 4 * m + 2)
  (h2 : grid.horizontal_lines.length = 2 * m + 1)
  (h3 : grid.vertical_lines.length = 2 * m + 1) :
  ∃ (A B C D E F : GridPoint),
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧
    B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧
    C ≠ D ∧ C ≠ E ∧ C ≠ F ∧
    D ≠ E ∧ D ≠ F ∧
    E ≠ F ∧
    centroid {a := A, b := B, c := C} = {x := 0, y := 0} ∧
    centroid {a := D, b := E, c := F} = {x := 0, y := 0} :=
  sorry

end red_points_centroid_theorem_l3118_311807


namespace root_analysis_uses_classification_and_discussion_l3118_311895

/-- A mathematical thinking method -/
inductive MathThinkingMethod
| Transformation
| Equation
| ClassificationAndDiscussion
| NumberAndShapeCombination

/-- A number category for root analysis -/
inductive NumberCategory
| Positive
| Zero
| Negative

/-- Represents the analysis of roots -/
structure RootAnalysis where
  categories : List NumberCategory
  method : MathThinkingMethod

/-- The specific root analysis we're considering -/
def squareAndCubeRootAnalysis : RootAnalysis :=
  { categories := [NumberCategory.Positive, NumberCategory.Zero, NumberCategory.Negative],
    method := MathThinkingMethod.ClassificationAndDiscussion }

/-- Theorem stating that the given root analysis uses classification and discussion thinking -/
theorem root_analysis_uses_classification_and_discussion :
  squareAndCubeRootAnalysis.method = MathThinkingMethod.ClassificationAndDiscussion :=
by sorry

end root_analysis_uses_classification_and_discussion_l3118_311895


namespace intersection_area_theorem_l3118_311876

/-- Rectangle represented by its width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- The intersection area of two rectangles -/
def intersection_area (r1 r2 : Rectangle) : ℝ := sorry

/-- Checks if two integers are relatively prime -/
def are_relatively_prime (m n : ℕ) : Prop := sorry

theorem intersection_area_theorem (abcd aecf : Rectangle) 
  (h1 : abcd.width = 11 ∧ abcd.height = 3)
  (h2 : aecf.width = 9 ∧ aecf.height = 7) :
  ∃ (m n : ℕ), 
    (intersection_area abcd aecf = m / n) ∧ 
    (are_relatively_prime m n) ∧ 
    (m + n = 109) := by sorry

end intersection_area_theorem_l3118_311876


namespace f_inequality_solution_set_l3118_311803

def f (x : ℝ) : ℝ := |2*x + 1| - |x - 2|

theorem f_inequality_solution_set :
  {x : ℝ | f x > 2} = {x : ℝ | x < -5 ∨ x > 1} := by sorry

end f_inequality_solution_set_l3118_311803


namespace equation_solution_l3118_311869

theorem equation_solution : ∃ x : ℝ, 3 * x - 6 = |(-25 * 2 + 50)| ∧ x = 2 := by
  sorry

end equation_solution_l3118_311869


namespace right_triangle_area_l3118_311896

-- Define the right triangle
def RightTriangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2

-- Define the incircle radius formula for a right triangle
def IncircleRadius (a b c r : ℝ) : Prop :=
  r = (a + b - c) / 2

-- Theorem statement
theorem right_triangle_area (a b c r : ℝ) :
  RightTriangle a b c →
  IncircleRadius a b c r →
  a = 3 →
  r = 3/8 →
  (1/2) * a * b = 21/16 :=
by sorry

end right_triangle_area_l3118_311896


namespace equation_solutions_l3118_311874

theorem equation_solutions : ∃! (s : Set ℝ), 
  (∀ x ∈ s, (x - 4)^4 + (x - 6)^4 = 16) ∧
  (s = {4, 6}) := by
  sorry

end equation_solutions_l3118_311874


namespace not_p_sufficient_not_necessary_for_not_q_l3118_311856

-- Define the sets p and q
def p : Set ℝ := {x | |2*x - 3| > 1}
def q : Set ℝ := {x | x^2 + x - 6 > 0}

-- Define what it means for one set to be a sufficient condition for another
def is_sufficient_condition (A B : Set ℝ) : Prop := B ⊆ A

-- Define what it means for one set to be a necessary condition for another
def is_necessary_condition (A B : Set ℝ) : Prop := A ⊆ B

-- State the theorem
theorem not_p_sufficient_not_necessary_for_not_q :
  is_sufficient_condition (Set.univ \ p) (Set.univ \ q) ∧
  ¬ is_necessary_condition (Set.univ \ p) (Set.univ \ q) :=
sorry

end not_p_sufficient_not_necessary_for_not_q_l3118_311856


namespace investment_problem_l3118_311842

/-- The solution to Susie Q's investment problem -/
theorem investment_problem (total_investment : ℝ) (interest_rate1 : ℝ) (interest_rate2 : ℝ) 
  (years : ℕ) (final_amount : ℝ) (investment1 : ℝ) :
  total_investment = 1500 →
  interest_rate1 = 0.04 →
  interest_rate2 = 0.06 →
  years = 2 →
  final_amount = 1700.02 →
  investment1 * (1 + interest_rate1) ^ years + 
    (total_investment - investment1) * (1 + interest_rate2) ^ years = final_amount →
  investment1 = 348.095 := by
    sorry

end investment_problem_l3118_311842


namespace smallest_ending_in_9_divisible_by_13_l3118_311885

theorem smallest_ending_in_9_divisible_by_13 : 
  ∃ (n : ℕ), n > 0 ∧ n % 10 = 9 ∧ n % 13 = 0 ∧ n = 69 ∧ 
  ∀ (m : ℕ), m > 0 → m % 10 = 9 → m % 13 = 0 → m ≥ n :=
by sorry

end smallest_ending_in_9_divisible_by_13_l3118_311885


namespace cost_of_500_sheets_l3118_311850

/-- The cost in dollars of a given number of sheets of paper. -/
def paper_cost (sheets : ℕ) : ℚ :=
  (sheets * 2 : ℚ) / 100

/-- Theorem stating that 500 sheets of paper cost $10.00. -/
theorem cost_of_500_sheets :
  paper_cost 500 = 10 := by sorry

end cost_of_500_sheets_l3118_311850


namespace repeating_block_11_13_l3118_311865

def decimal_expansion (n d : ℕ) : List ℕ :=
  sorry

def is_repeating_block (l : List ℕ) (block : List ℕ) : Prop :=
  sorry

theorem repeating_block_11_13 :
  ∃ (block : List ℕ),
    block.length = 6 ∧
    is_repeating_block (decimal_expansion 11 13) block ∧
    ∀ (smaller_block : List ℕ),
      smaller_block.length < 6 →
      ¬ is_repeating_block (decimal_expansion 11 13) smaller_block :=
by sorry

end repeating_block_11_13_l3118_311865


namespace card_selection_counts_l3118_311819

/-- Represents a standard deck of 52 playing cards -/
structure Deck :=
  (cards : Finset (Fin 4 × Fin 13))
  (card_count : cards.card = 52)

/-- Counts the number of ways to select 4 cards with different suits and ranks -/
def count_four_different (d : Deck) : ℕ := sorry

/-- Counts the number of ways to select 6 cards with all suits represented -/
def count_six_all_suits (d : Deck) : ℕ := sorry

/-- Theorem stating the correct counts for both selections -/
theorem card_selection_counts (d : Deck) : 
  count_four_different d = 17160 ∧ count_six_all_suits d = 8682544 := by sorry

end card_selection_counts_l3118_311819


namespace final_salary_ratio_l3118_311832

/-- Represents the sequence of salary adjustments throughout the year -/
def salary_adjustments : List (ℝ → ℝ) := [
  (· * 1.20),       -- 20% increase after 2 months
  (· * 0.90),       -- 10% decrease in 3rd month
  (· * 1.12),       -- 12% increase in 4th month
  (· * 0.92),       -- 8% decrease in 5th month
  (· * 1.12),       -- 12% increase in 6th month
  (· * 0.92),       -- 8% decrease in 7th month
  (· * 1.08),       -- 8% bonus in 8th month
  (· * 0.50),       -- 50% decrease due to financial crisis
  (· * 0.90),       -- 10% decrease in 9th month
  (· * 1.15),       -- 15% increase in 10th month
  (· * 0.90),       -- 10% decrease in 11th month
  (· * 1.50)        -- 50% increase in last month
]

/-- Applies a list of functions sequentially to an initial value -/
def apply_adjustments (adjustments : List (ℝ → ℝ)) (initial : ℝ) : ℝ :=
  adjustments.foldl (λ acc f => f acc) initial

/-- Theorem stating the final salary ratio after adjustments -/
theorem final_salary_ratio (S : ℝ) (hS : S > 0) :
  let initial_after_tax := 0.70 * S
  let final_salary := apply_adjustments salary_adjustments initial_after_tax
  ∃ ε > 0, abs (final_salary / initial_after_tax - 0.8657) < ε :=
sorry

end final_salary_ratio_l3118_311832


namespace monotonic_range_a_l3118_311882

def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x

def is_monotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  (∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x ≤ f y) ∨
  (∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y ≤ f x)

theorem monotonic_range_a :
  ∀ a : ℝ, is_monotonic (f a) (-2) 3 ↔ a ≤ -27 ∨ 0 ≤ a :=
by sorry

end monotonic_range_a_l3118_311882


namespace y_at_64_l3118_311804

-- Define the function y in terms of k and x
def y (k : ℝ) (x : ℝ) : ℝ := k * x^(1/3)

-- State the theorem
theorem y_at_64 (k : ℝ) :
  y k 8 = 4 → y k 64 = 8 := by
  sorry

end y_at_64_l3118_311804


namespace min_value_expression_l3118_311836

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (((x^2 + y^2) * (4*x^2 + y^2)).sqrt) / (x*y) ≥ 3 :=
sorry

end min_value_expression_l3118_311836


namespace complex_fraction_simplification_l3118_311899

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_fraction_simplification :
  (5 + 7 * i) / (3 - 4 * i + 2 * i^2) = -23/17 + (27/17) * i :=
by
  sorry

end complex_fraction_simplification_l3118_311899


namespace unique_real_solution_iff_a_in_range_l3118_311871

/-- The equation x^3 - ax^2 - 4ax + 4a^2 - 1 = 0 has exactly one real solution in x if and only if a ∈ (-∞, 3/4). -/
theorem unique_real_solution_iff_a_in_range (a : ℝ) : 
  (∃! x : ℝ, x^3 - a*x^2 - 4*a*x + 4*a^2 - 1 = 0) ↔ a < 3/4 :=
sorry

end unique_real_solution_iff_a_in_range_l3118_311871


namespace figure_to_square_possible_l3118_311898

/-- Represents a point on a grid --/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- Represents a triangle on a grid --/
structure GridTriangle where
  p1 : GridPoint
  p2 : GridPoint
  p3 : GridPoint

/-- Represents a figure on a grid --/
structure GridFigure where
  points : List GridPoint

/-- Function to calculate the area of a grid figure --/
def calculateArea (figure : GridFigure) : ℕ :=
  sorry

/-- Function to check if a list of triangles forms a square --/
def formsSquare (triangles : List GridTriangle) : Prop :=
  sorry

/-- The main theorem --/
theorem figure_to_square_possible (figure : GridFigure) : 
  ∃ (triangles : List GridTriangle), 
    triangles.length = 5 ∧ 
    (calculateArea figure = calculateArea (GridFigure.mk (triangles.bind (λ t => [t.p1, t.p2, t.p3]))) ∧
    formsSquare triangles) :=
  sorry

end figure_to_square_possible_l3118_311898


namespace room_width_proof_l3118_311897

/-- Given a rectangular room with specified dimensions and veranda, prove its width. -/
theorem room_width_proof (room_length : ℝ) (veranda_width : ℝ) (veranda_area : ℝ) : 
  room_length = 17 →
  veranda_width = 2 →
  veranda_area = 132 →
  ∃ room_width : ℝ,
    (room_length + 2 * veranda_width) * (room_width + 2 * veranda_width) - 
    (room_length * room_width) = veranda_area ∧
    room_width = 12 := by
  sorry

end room_width_proof_l3118_311897


namespace average_salary_calculation_l3118_311886

/-- Average salary calculation problem -/
theorem average_salary_calculation (n : ℕ) 
  (avg_all : ℕ) 
  (avg_feb_may : ℕ) 
  (salary_may : ℕ) 
  (salary_jan : ℕ) 
  (h1 : avg_all = 8000)
  (h2 : avg_feb_may = 8700)
  (h3 : salary_may = 6500)
  (h4 : salary_jan = 3700) :
  (salary_jan + (4 * avg_feb_may - salary_may)) / 4 = 8000 := by
  sorry

end average_salary_calculation_l3118_311886


namespace solve_system_of_equations_l3118_311863

theorem solve_system_of_equations :
  ∀ (x y m n : ℝ),
  (4 * x + 3 * y = m) →
  (6 * x - y = n) →
  ((m / 3 + n / 8 = 8) ∧ (m / 6 + n / 2 = 11)) →
  (x = 3 ∧ y = 2) :=
by
  sorry

end solve_system_of_equations_l3118_311863


namespace smallest_N_property_l3118_311859

/-- The smallest natural number N such that N × 999 consists entirely of the digit seven in its decimal representation -/
def smallest_N : ℕ := 778556334111889667445223

/-- Predicate to check if a natural number consists entirely of the digit seven -/
def all_sevens (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 7

theorem smallest_N_property :
  (all_sevens (smallest_N * 999)) ∧
  (∀ m : ℕ, m < smallest_N → ¬(all_sevens (m * 999))) := by
  sorry

end smallest_N_property_l3118_311859


namespace school_election_l3118_311892

theorem school_election (total_students : ℕ) : total_students = 2000 :=
  let voter_percentage : ℚ := 25 / 100
  let winner_vote_percentage : ℚ := 55 / 100
  let loser_vote_percentage : ℚ := 1 - winner_vote_percentage
  let vote_difference : ℕ := 50
  have h1 : (winner_vote_percentage * voter_percentage * total_students : ℚ) = 
            (loser_vote_percentage * voter_percentage * total_students + vote_difference : ℚ) := by sorry
  sorry

end school_election_l3118_311892


namespace greatest_integer_inequality_l3118_311831

theorem greatest_integer_inequality : ∀ x : ℤ, (5 : ℚ) / 8 > (x : ℚ) / 17 ↔ x ≤ 10 :=
by sorry

end greatest_integer_inequality_l3118_311831


namespace chalk_problem_l3118_311883

theorem chalk_problem (total_people : ℕ) (added_chalk : ℕ) (lost_chalk : ℕ) (final_per_person : ℚ) :
  total_people = 11 →
  added_chalk = 28 →
  lost_chalk = 4 →
  final_per_person = 5.5 →
  ∃ (original_chalk : ℕ), original_chalk = 37 ∧ 
    (↑original_chalk - ↑lost_chalk + ↑added_chalk : ℚ) = ↑total_people * final_per_person :=
by sorry

end chalk_problem_l3118_311883


namespace llama_breeding_problem_llama_breeding_solution_l3118_311853

theorem llama_breeding_problem (pregnant_llamas : ℕ) (twin_pregnancies : ℕ) 
  (traded_calves : ℕ) (new_adults : ℕ) (final_herd : ℕ) : ℕ :=
  let single_pregnancies := pregnant_llamas - twin_pregnancies
  let total_calves := twin_pregnancies * 2 + single_pregnancies * 1
  let remaining_calves := total_calves - traded_calves
  let pre_sale_herd := final_herd / (2/3)
  let pre_sale_adults := pre_sale_herd - remaining_calves - new_adults
  let original_adults := pre_sale_adults - new_adults
  total_calves

theorem llama_breeding_solution : 
  llama_breeding_problem 9 5 8 2 18 = 14 := by sorry

end llama_breeding_problem_llama_breeding_solution_l3118_311853


namespace ratio_w_y_is_15_4_l3118_311877

-- Define the ratios as fractions
def ratio_w_x : ℚ := 5 / 4
def ratio_y_z : ℚ := 5 / 3
def ratio_z_x : ℚ := 1 / 5

-- Theorem statement
theorem ratio_w_y_is_15_4 :
  let ratio_w_y := ratio_w_x / (ratio_y_z * ratio_z_x)
  ratio_w_y = 15 / 4 := by
  sorry

end ratio_w_y_is_15_4_l3118_311877


namespace students_only_english_l3118_311827

theorem students_only_english (total : ℕ) (both : ℕ) (german : ℕ) (h1 : total = 52) (h2 : both = 12) (h3 : german = 22) (h4 : total ≥ german) : total - german + both = 30 := by
  sorry

end students_only_english_l3118_311827


namespace pear_sales_l3118_311867

theorem pear_sales (morning_sales afternoon_sales total_sales : ℕ) :
  morning_sales = 120 →
  afternoon_sales = 2 * morning_sales →
  total_sales = morning_sales + afternoon_sales →
  total_sales = 360 →
  afternoon_sales = 240 := by
sorry

end pear_sales_l3118_311867


namespace correct_transformation_l3118_311884

theorem correct_transformation (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : 3 * a = 2 * b) :
  a / 2 = b / 3 := by
  sorry

end correct_transformation_l3118_311884


namespace blue_stamp_price_l3118_311852

/-- Given a collection of stamps and their prices, prove the price of blue stamps --/
theorem blue_stamp_price
  (red_count : ℕ)
  (blue_count : ℕ)
  (yellow_count : ℕ)
  (red_price : ℚ)
  (yellow_price : ℚ)
  (total_earnings : ℚ)
  (h1 : red_count = 20)
  (h2 : blue_count = 80)
  (h3 : yellow_count = 7)
  (h4 : red_price = 11/10)
  (h5 : yellow_price = 2)
  (h6 : total_earnings = 100) :
  (total_earnings - red_count * red_price - yellow_count * yellow_price) / blue_count = 4/5 :=
by sorry

end blue_stamp_price_l3118_311852


namespace pet_shop_dogs_count_l3118_311891

/-- Given a pet shop with dogs, cats, and bunnies in stock, this theorem proves
    the number of dogs based on the given ratio and total count of dogs and bunnies. -/
theorem pet_shop_dogs_count
  (ratio_dogs : ℕ)
  (ratio_cats : ℕ)
  (ratio_bunnies : ℕ)
  (total_dogs_and_bunnies : ℕ)
  (h_ratio : ratio_dogs = 3 ∧ ratio_cats = 5 ∧ ratio_bunnies = 9)
  (h_total : total_dogs_and_bunnies = 204) :
  (ratio_dogs * total_dogs_and_bunnies) / (ratio_dogs + ratio_bunnies) = 51 :=
by
  sorry


end pet_shop_dogs_count_l3118_311891


namespace fraction_problem_l3118_311881

theorem fraction_problem (f : ℚ) : f * 12 + 5 = 11 → f = 1/2 := by
  sorry

end fraction_problem_l3118_311881


namespace tabletennis_arrangements_eq_252_l3118_311825

/-- The number of ways to arrange 5 players from a team of 10, 
    where 3 specific players must occupy positions 1, 3, and 5, 
    and 2 players from the remaining 7 must occupy positions 2 and 4 -/
def tabletennis_arrangements (total_players : ℕ) (main_players : ℕ) 
    (players_to_send : ℕ) (remaining_players : ℕ) : ℕ := 
  Nat.factorial main_players * (remaining_players * (remaining_players - 1))

theorem tabletennis_arrangements_eq_252 : 
  tabletennis_arrangements 10 3 5 7 = 252 := by
  sorry

#eval tabletennis_arrangements 10 3 5 7

end tabletennis_arrangements_eq_252_l3118_311825


namespace units_digit_of_8421_to_1287_l3118_311888

theorem units_digit_of_8421_to_1287 : ∃ n : ℕ, 8421^1287 = 10 * n + 1 := by sorry

end units_digit_of_8421_to_1287_l3118_311888


namespace chess_tournament_games_l3118_311810

theorem chess_tournament_games (n : ℕ) 
  (h1 : n > 0) 
  (h2 : (10 * 9 * n) / 2 = 90) : n = 2 := by
  sorry

end chess_tournament_games_l3118_311810


namespace water_tank_fill_time_l3118_311822

/-- Represents the time (in hours) it takes to fill a water tank -/
def fill_time : ℝ → ℝ → ℝ → Prop :=
  λ T leak_empty_time leak_fill_time =>
    (1 / T - 1 / leak_empty_time = 1 / leak_fill_time) ∧
    (leak_fill_time = T + 1)

theorem water_tank_fill_time :
  ∃ (T : ℝ), T > 0 ∧ fill_time T 30 (T + 1) ∧ T = 5 := by
  sorry

end water_tank_fill_time_l3118_311822


namespace sum_with_radical_conjugate_l3118_311848

theorem sum_with_radical_conjugate : 
  let x : ℝ := 15 - Real.sqrt 5000
  let y : ℝ := 15 + Real.sqrt 5000
  x + y = 30 := by
sorry

end sum_with_radical_conjugate_l3118_311848


namespace jackson_flight_distance_l3118_311816

theorem jackson_flight_distance (beka_distance : ℕ) (difference : ℕ) (jackson_distance : ℕ) : 
  beka_distance = 873 → 
  difference = 310 → 
  beka_distance = jackson_distance + difference → 
  jackson_distance = 563 :=
by sorry

end jackson_flight_distance_l3118_311816


namespace f_of_x_plus_one_l3118_311841

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- State the theorem
theorem f_of_x_plus_one (x : ℝ) : f (x + 1) = x^2 + 2*x + 1 := by
  sorry

end f_of_x_plus_one_l3118_311841


namespace tetrahedralContactsFormula_l3118_311823

/-- The number of contact points in a tetrahedral stack of spheres -/
def tetrahedralContacts (n : ℕ) : ℕ := n^3 - n

/-- Theorem: The number of contact points in a tetrahedral stack of spheres
    with n spheres along each edge is n³ - n -/
theorem tetrahedralContactsFormula (n : ℕ) :
  tetrahedralContacts n = n^3 - n := by
  sorry

end tetrahedralContactsFormula_l3118_311823


namespace min_dot_product_ellipse_l3118_311857

/-- The minimum dot product of OP and FP for an ellipse -/
theorem min_dot_product_ellipse :
  ∀ (x y : ℝ), 
  x^2 / 9 + y^2 / 8 = 1 →
  ∃ (min : ℝ), 
  (∀ (x' y' : ℝ), x'^2 / 9 + y'^2 / 8 = 1 → 
    x'^2 + x' + y'^2 ≥ min) ∧
  min = 6 :=
by sorry

end min_dot_product_ellipse_l3118_311857


namespace two_sqrt_two_minus_three_is_negative_l3118_311805

theorem two_sqrt_two_minus_three_is_negative : 2 * Real.sqrt 2 - 3 < 0 := by
  sorry

end two_sqrt_two_minus_three_is_negative_l3118_311805


namespace sum_of_real_roots_l3118_311840

theorem sum_of_real_roots (x : ℝ) : 
  let f : ℝ → ℝ := fun x => x^4 - 8*x + 4
  ∃ (r₁ r₂ : ℝ), (f r₁ = 0 ∧ f r₂ = 0 ∧ (∀ r : ℝ, f r = 0 → r = r₁ ∨ r = r₂)) ∧ 
  r₁ + r₂ = -2 * Real.sqrt 2 :=
sorry

end sum_of_real_roots_l3118_311840


namespace sum_of_coefficients_l3118_311873

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x, (x + 2)^4 = a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a) →
  a₁ + a₂ + a₃ + a₄ = 65 := by
sorry

end sum_of_coefficients_l3118_311873


namespace repunit_primes_upper_bound_l3118_311812

def repunit (k : ℕ) : ℕ := (10^k - 1) / 9

def is_repunit_prime (n : ℕ) : Prop :=
  Nat.Prime n ∧ ∃ k, repunit k = n

theorem repunit_primes_upper_bound :
  (∃ (S : Finset ℕ), ∀ n ∈ S, is_repunit_prime n ∧ n < 10^29) →
  (∃ (S : Finset ℕ), ∀ n ∈ S, is_repunit_prime n ∧ n < 10^29 ∧ S.card ≤ 9) :=
sorry

end repunit_primes_upper_bound_l3118_311812


namespace sum_of_factorials_last_two_digits_l3118_311893

def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def lastTwoDigits (n : ℕ) : ℕ := n % 100

def isExcluded (n : ℕ) : Bool := n % 3 == 0 && n % 5 == 0

def sumOfFactorials : ℕ := 
  (List.range 100).foldl (λ acc n => 
    if !isExcluded (n + 1) then 
      (acc + lastTwoDigits (factorial (n + 1))) % 100 
    else 
      acc
  ) 0

theorem sum_of_factorials_last_two_digits : 
  sumOfFactorials = 13 := by sorry

end sum_of_factorials_last_two_digits_l3118_311893


namespace ln_geq_num_prime_factors_ln2_l3118_311855

/-- The number of prime factors of a positive integer -/
def num_prime_factors (n : ℕ+) : ℕ := sorry

/-- For any positive integer n, ln n ≥ p(n) ln 2, where p(n) is the number of prime factors of n -/
theorem ln_geq_num_prime_factors_ln2 (n : ℕ+) : Real.log n ≥ (num_prime_factors n : ℝ) * Real.log 2 := by
  sorry

end ln_geq_num_prime_factors_ln2_l3118_311855


namespace lambda_5_lower_bound_l3118_311875

/-- The ratio of the longest distance to the shortest distance for n points in a plane -/
def lambda (n : ℕ) : ℝ := sorry

/-- Theorem: For 5 points in a plane, the ratio of the longest distance to the shortest distance
    is greater than or equal to 2 sin 54° -/
theorem lambda_5_lower_bound : lambda 5 ≥ 2 * Real.sin (54 * π / 180) := by sorry

end lambda_5_lower_bound_l3118_311875


namespace greatest_b_value_l3118_311838

theorem greatest_b_value (b : ℝ) : 
  (∀ x : ℝ, x^2 - 12*x + 32 ≤ 0 → x ≤ 8) ∧ 
  (8^2 - 12*8 + 32 ≤ 0) := by
  sorry

end greatest_b_value_l3118_311838


namespace max_value_2a_plus_b_l3118_311847

theorem max_value_2a_plus_b (a b : ℝ) (h : 4 * a^2 + b^2 + a * b = 1) :
  2 * a + b ≤ 2 * Real.sqrt 10 / 5 :=
by sorry

end max_value_2a_plus_b_l3118_311847


namespace anniversary_count_l3118_311864

def founding_year : Nat := 1949
def current_year : Nat := 2015

theorem anniversary_count :
  current_year - founding_year = 66 := by sorry

end anniversary_count_l3118_311864


namespace high_school_students_l3118_311802

theorem high_school_students (high_school middle_school lower_school : ℕ) : 
  high_school = 4 * lower_school →
  high_school + lower_school = 7 * middle_school →
  middle_school = 300 →
  high_school = 1680 := by
sorry

end high_school_students_l3118_311802


namespace milk_level_lowered_l3118_311845

/-- Proves that removing 5250 gallons of milk from a 56ft by 25ft rectangular box
    lowers the milk level by 6 inches. -/
theorem milk_level_lowered (box_length box_width : ℝ)
                            (milk_volume_gallons : ℝ)
                            (cubic_feet_to_gallons : ℝ)
                            (inches_per_foot : ℝ) :
  box_length = 56 →
  box_width = 25 →
  milk_volume_gallons = 5250 →
  cubic_feet_to_gallons = 7.5 →
  inches_per_foot = 12 →
  (milk_volume_gallons / cubic_feet_to_gallons) /
  (box_length * box_width) * inches_per_foot = 6 :=
by sorry

end milk_level_lowered_l3118_311845


namespace ink_cartridge_cost_l3118_311861

/-- Given that 13 ink cartridges cost 182 dollars in total, 
    prove that the cost of one ink cartridge is 14 dollars. -/
theorem ink_cartridge_cost : ℕ → Prop :=
  fun x => (13 * x = 182) → (x = 14)

/-- Proof of the theorem -/
example : ink_cartridge_cost 14 := by
  sorry

end ink_cartridge_cost_l3118_311861


namespace one_true_proposition_l3118_311866

theorem one_true_proposition :
  (∃! i : Fin 4, 
    (i = 0 ∧ (∀ x y : ℝ, ¬(x = -y) → x + y ≠ 0)) ∨
    (i = 1 ∧ (∀ a b : ℝ, a^2 > b^2 → a > b)) ∨
    (i = 2 ∧ (∃ x : ℝ, x ≤ -3 ∧ x^2 - x - 6 ≤ 0)) ∨
    (i = 3 ∧ (∀ a b : ℝ, Irrational a ∧ Irrational b → Irrational (a^b)))) :=
sorry

end one_true_proposition_l3118_311866


namespace quadratic_no_real_roots_l3118_311854

theorem quadratic_no_real_roots 
  (p q a b c : ℝ) 
  (hp_pos : p > 0) (hq_pos : q > 0) (ha_pos : a > 0) (hb_pos : b > 0) (hc_pos : c > 0)
  (hp_neq_q : p ≠ q)
  (h_geom : a^2 = p * q)
  (h_arith : b = (2*p + q)/3 ∧ c = (p + 2*q)/3) :
  (2*a)^2 - 4*b*c < 0 :=
sorry

end quadratic_no_real_roots_l3118_311854


namespace periodic_function_value_l3118_311809

/-- A function satisfying the given conditions -/
def periodic_function (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f (x + 2) * f x = 1) ∧ f 2 = 2

/-- Theorem stating the value of f(2016) given the conditions -/
theorem periodic_function_value (f : ℝ → ℝ) (h : periodic_function f) : f 2016 = 1/2 := by
  sorry

end periodic_function_value_l3118_311809


namespace all_balls_are_red_l3118_311849

/-- 
Given a bag of 12 balls that are either red or blue, 
prove that if the probability of drawing two red balls simultaneously is 1/10, 
then all 12 balls must be red.
-/
theorem all_balls_are_red (total_balls : ℕ) (red_balls : ℕ) 
  (h1 : total_balls = 12)
  (h2 : red_balls ≤ total_balls)
  (h3 : (red_balls : ℚ) / total_balls * (red_balls - 1) / (total_balls - 1) = 1 / 10) :
  red_balls = total_balls :=
sorry

end all_balls_are_red_l3118_311849


namespace distance_to_axes_l3118_311815

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the distance functions
def distToXAxis (p : Point2D) : ℝ := |p.y|
def distToYAxis (p : Point2D) : ℝ := |p.x|

-- State the theorem
theorem distance_to_axes (Q : Point2D) (hx : Q.x = -6) (hy : Q.y = 5) :
  distToXAxis Q = 5 ∧ distToYAxis Q = 6 := by
  sorry


end distance_to_axes_l3118_311815


namespace melanie_dimes_l3118_311801

theorem melanie_dimes (initial : ℕ) (from_dad : ℕ) (total : ℕ) : 
  initial = 7 → from_dad = 8 → total = 19 → total - (initial + from_dad) = 4 := by
sorry

end melanie_dimes_l3118_311801


namespace root_in_interval_l3118_311808

def f (x : ℝ) := x^3 - 2*x - 1

theorem root_in_interval :
  f 1 < 0 →
  f 2 > 0 →
  f (3/2) < 0 →
  ∃ x : ℝ, 3/2 < x ∧ x < 2 ∧ f x = 0 :=
by sorry

end root_in_interval_l3118_311808


namespace parabola_shift_theorem_l3118_311843

/-- Represents a parabola in the form y = ax² + bx + c --/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally and vertically --/
def shift_parabola (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a
    b := -2 * p.a * h + p.b
    c := p.a * h^2 - p.b * h + p.c - v }

theorem parabola_shift_theorem (x : ℝ) :
  let original := Parabola.mk 2 0 0  -- y = 2x²
  let shifted := shift_parabola original 3 1  -- Shift 3 right, 1 down
  shifted.a * x^2 + shifted.b * x + shifted.c = 2 * (x - 3)^2 - 1 := by
  sorry

#check parabola_shift_theorem

end parabola_shift_theorem_l3118_311843


namespace prob_sum_five_is_one_ninth_l3118_311835

/-- The number of faces on each die -/
def numFaces : ℕ := 6

/-- The set of possible outcomes when rolling two dice -/
def outcomes : Finset (ℕ × ℕ) :=
  Finset.product (Finset.range numFaces) (Finset.range numFaces)

/-- The set of outcomes where the sum of the dice is 5 -/
def sumFiveOutcomes : Finset (ℕ × ℕ) :=
  outcomes.filter (fun p => p.1 + p.2 + 2 = 5)

/-- The probability of the sum of two fair dice being 5 -/
theorem prob_sum_five_is_one_ninth :
  (sumFiveOutcomes.card : ℚ) / outcomes.card = 1 / 9 := by
  sorry


end prob_sum_five_is_one_ninth_l3118_311835


namespace overall_gain_percentage_is_10_51_l3118_311851

/-- Represents a transaction with quantity, buy price, and sell price -/
structure Transaction where
  quantity : ℕ
  buyPrice : ℚ
  sellPrice : ℚ

/-- Calculates the profit or loss for a single transaction -/
def transactionProfit (t : Transaction) : ℚ :=
  t.quantity * (t.sellPrice - t.buyPrice)

/-- Calculates the cost for a single transaction -/
def transactionCost (t : Transaction) : ℚ :=
  t.quantity * t.buyPrice

/-- Calculates the overall gain percentage for a list of transactions -/
def overallGainPercentage (transactions : List Transaction) : ℚ :=
  let totalProfit := (transactions.map transactionProfit).sum
  let totalCost := (transactions.map transactionCost).sum
  totalProfit / totalCost * 100

/-- The main theorem stating that the overall gain percentage for the given transactions is 10.51% -/
theorem overall_gain_percentage_is_10_51 :
  let transactions := [
    ⟨10, 8, 10⟩,
    ⟨7, 15, 18⟩,
    ⟨5, 22, 20⟩
  ]
  overallGainPercentage transactions = 10.51 := by
  sorry

end overall_gain_percentage_is_10_51_l3118_311851


namespace rectangle_area_l3118_311800

/-- Proves that a rectangle with a perimeter of 176 inches and a length 8 inches more than its width has an area of 1920 square inches. -/
theorem rectangle_area (w : ℝ) (l : ℝ) : 
  (2 * l + 2 * w = 176) →  -- Perimeter condition
  (l = w + 8) →            -- Length-width relation
  (l * w = 1920)           -- Area to be proved
  := by sorry

end rectangle_area_l3118_311800


namespace intersection_distance_implies_a_value_l3118_311880

-- Define the curve C
def curve_C (a : ℝ) (x y : ℝ) : Prop := y^2 = 2*a*x ∧ a > 0

-- Define the line l
def line_l (x y : ℝ) : Prop := y = x - 2

-- Define the intersection points
def intersection_points (a : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧ curve_C a x y ∧ line_l x y}

-- Theorem statement
theorem intersection_distance_implies_a_value (a : ℝ) :
  (∃ (A B : ℝ × ℝ), A ∈ intersection_points a ∧ B ∈ intersection_points a ∧ 
    A ≠ B ∧ Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 10) →
  a = 1 :=
sorry

end intersection_distance_implies_a_value_l3118_311880


namespace possible_values_of_a_l3118_311811

def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | a*x - 2 = 0}

theorem possible_values_of_a :
  ∀ a : ℝ, (A ∪ B a = A) ↔ a ∈ ({0, 1, 2} : Set ℝ) :=
by sorry

end possible_values_of_a_l3118_311811


namespace charity_event_equation_l3118_311879

theorem charity_event_equation :
  ∀ x : ℕ,
  (x + (12 - x) = 12) →  -- Total number of banknotes is 12
  (x ≤ 12) →             -- Ensure x doesn't exceed total banknotes
  (x + 5 * (12 - x) = 48) -- The equation correctly represents the problem
  :=
by
  sorry

end charity_event_equation_l3118_311879


namespace image_property_l3118_311826

class StarOperation (T : Type) where
  star : T → T → T

variable {T : Type} [StarOperation T]

def image (a : T) : Set T :=
  {c | ∃ b, c = StarOperation.star a b}

theorem image_property (a : T) (c : T) (h : c ∈ image a) :
  StarOperation.star a c = c := by
  sorry

end image_property_l3118_311826


namespace first_character_lines_l3118_311894

/-- Represents the number of lines for each character in Jerry's skit script. -/
structure ScriptLines where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Defines the conditions for Jerry's skit script. -/
def valid_script (s : ScriptLines) : Prop :=
  s.first = s.second + 8 ∧
  s.third = 2 ∧
  s.second = 3 * s.third + 6

/-- Theorem stating that the first character has 20 lines in a valid script. -/
theorem first_character_lines (s : ScriptLines) (h : valid_script s) : s.first = 20 := by
  sorry

end first_character_lines_l3118_311894


namespace triangle_properties_l3118_311839

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- State the theorem
theorem triangle_properties (t : Triangle) :
  (Real.cos t.C + (Real.cos t.A - Real.sqrt 3 * Real.sin t.A) * Real.cos t.B = 0) →
  (t.a + t.c = 1) →
  (t.B = π / 3 ∧ 1 / 2 ≤ t.b ∧ t.b < 1) := by
  sorry

end triangle_properties_l3118_311839


namespace expression_simplification_l3118_311837

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 2 + 1) :
  (((x^2 - 3) / (x + 2) - x + 2) / ((x^2 - 4) / (x^2 + 4*x + 4))) = Real.sqrt 2 + 1 := by
  sorry

end expression_simplification_l3118_311837


namespace quadratic_condition_l3118_311887

/-- Represents a quadratic equation in the form ax^2 + bx + c = 0 --/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a given equation is quadratic --/
def isQuadratic (eq : QuadraticEquation) : Prop :=
  eq.a ≠ 0

/-- The equation mx^2 + 3x - 4 = 3x^2 rearranged to standard form --/
def equationOfInterest (m : ℝ) : QuadraticEquation :=
  ⟨m - 3, 3, -4⟩

/-- Theorem stating that for the equation to be quadratic, m must not equal 3 --/
theorem quadratic_condition (m : ℝ) :
  isQuadratic (equationOfInterest m) ↔ m ≠ 3 := by sorry

end quadratic_condition_l3118_311887


namespace quadratic_equation_coefficients_l3118_311833

/-- 
Given a quadratic equation 6x² - 1 = 3x, when converted to the general form ax² + bx + c = 0,
the coefficient of x² (a) is 6 and the coefficient of x (b) is -3.
-/
theorem quadratic_equation_coefficients :
  ∃ (a b c : ℝ), 
    (∀ x, 6 * x^2 - 1 = 3 * x ↔ a * x^2 + b * x + c = 0) ∧
    a = 6 ∧ 
    b = -3 := by
  sorry

end quadratic_equation_coefficients_l3118_311833


namespace sum_of_roots_l3118_311813

theorem sum_of_roots (k c d : ℝ) (y₁ y₂ : ℝ) : 
  y₁ ≠ y₂ →
  5 * y₁^2 - k * y₁ - c = 0 →
  5 * y₂^2 - k * y₂ - c = 0 →
  5 * y₁^2 - k * y₁ = d →
  5 * y₂^2 - k * y₂ = d →
  d ≠ c →
  y₁ + y₂ = k / 5 := by
sorry

end sum_of_roots_l3118_311813


namespace angle_ratio_not_implies_right_triangle_l3118_311870

/-- Triangle ABC with angles A, B, and C -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  sum_angles : A + B + C = 180
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C

/-- The condition that angles are in the ratio 3:4:5 -/
def angle_ratio (t : Triangle) : Prop :=
  ∃ (x : ℝ), t.A = 3*x ∧ t.B = 4*x ∧ t.C = 5*x

/-- A triangle is right if one of its angles is 90 degrees -/
def is_right_triangle (t : Triangle) : Prop :=
  t.A = 90 ∨ t.B = 90 ∨ t.C = 90

/-- The main theorem: a triangle with angles in ratio 3:4:5 is not necessarily right -/
theorem angle_ratio_not_implies_right_triangle :
  ∃ (t : Triangle), angle_ratio t ∧ ¬is_right_triangle t :=
sorry

end angle_ratio_not_implies_right_triangle_l3118_311870
