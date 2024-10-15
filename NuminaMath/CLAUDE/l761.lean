import Mathlib

namespace NUMINAMATH_CALUDE_polynomial_factorization_l761_76163

theorem polynomial_factorization (x : ℝ) : 
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 9) = 
  (x^2 + 6*x + 3) * (x^2 + 6*x + 12) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l761_76163


namespace NUMINAMATH_CALUDE_student_count_l761_76165

/-- Given a student's position from both ends of a line, calculate the total number of students -/
theorem student_count (right_rank left_rank : ℕ) (h1 : right_rank = 13) (h2 : left_rank = 8) :
  right_rank + left_rank - 1 = 20 := by
  sorry

end NUMINAMATH_CALUDE_student_count_l761_76165


namespace NUMINAMATH_CALUDE_average_assembly_rate_l761_76176

/-- Represents the car assembly problem with the given conditions -/
def CarAssemblyProblem (x : ℝ) : Prop :=
  let original_plan := 21
  let assembled_before_order := 6
  let additional_order := 5
  let increased_rate := x + 2
  (original_plan / x) - (assembled_before_order / x) - 
    ((original_plan - assembled_before_order + additional_order) / increased_rate) = 1

/-- Theorem stating that the average daily assembly rate after the additional order is 5 cars per day -/
theorem average_assembly_rate : ∃ x : ℝ, CarAssemblyProblem x ∧ x + 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_average_assembly_rate_l761_76176


namespace NUMINAMATH_CALUDE_questionnaire_responses_l761_76108

theorem questionnaire_responses (response_rate : ℝ) (min_questionnaires : ℝ) : 
  response_rate = 0.62 → min_questionnaires = 483.87 → 
  ⌊(⌈min_questionnaires⌉ : ℝ) * response_rate⌋ = 300 := by
sorry

end NUMINAMATH_CALUDE_questionnaire_responses_l761_76108


namespace NUMINAMATH_CALUDE_deposit_growth_condition_l761_76174

theorem deposit_growth_condition 
  (X r s : ℝ) 
  (h_X_pos : X > 0) 
  (h_s_bound : s < 20) :
  X * (1 + r / 100) * (1 - s / 100) > X ↔ r > 100 * s / (100 - s) := by
  sorry

end NUMINAMATH_CALUDE_deposit_growth_condition_l761_76174


namespace NUMINAMATH_CALUDE_contrapositive_real_roots_l761_76179

theorem contrapositive_real_roots (a : ℝ) : 
  (∀ x : ℝ, x^2 + x - a ≠ 0) → a < 0 := by
  sorry

end NUMINAMATH_CALUDE_contrapositive_real_roots_l761_76179


namespace NUMINAMATH_CALUDE_joans_kittens_l761_76193

theorem joans_kittens (given_away : ℕ) (remaining : ℕ) (original : ℕ) : 
  given_away = 2 → remaining = 6 → original = given_away + remaining :=
by
  sorry

end NUMINAMATH_CALUDE_joans_kittens_l761_76193


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l761_76117

theorem completing_square_equivalence :
  ∀ x : ℝ, x^2 - 4*x - 1 = 0 ↔ (x - 2)^2 = 5 := by
sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l761_76117


namespace NUMINAMATH_CALUDE_function_decreasing_condition_l761_76188

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 4 * (a + 1) * x - 3

-- State the theorem
theorem function_decreasing_condition (a : ℝ) :
  (∀ x ∈ Set.Ici 2, ∀ y ∈ Set.Ici 2, x < y → f a x > f a y) ↔ a ≤ -1/2 := by
  sorry

end NUMINAMATH_CALUDE_function_decreasing_condition_l761_76188


namespace NUMINAMATH_CALUDE_special_arrangement_count_l761_76162

/-- The number of ways to arrange n people in a row --/
def linearArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to choose k items from n items --/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of ways to arrange 5 volunteers and 2 elderly people in a row,
    with the elderly people next to each other but not at the ends --/
def specialArrangement : ℕ :=
  choose 5 2 * linearArrangements 4 * 2

theorem special_arrangement_count : specialArrangement = 960 := by
  sorry

end NUMINAMATH_CALUDE_special_arrangement_count_l761_76162


namespace NUMINAMATH_CALUDE_linda_remaining_candies_l761_76178

-- Define the initial number of candies Linda has
def initial_candies : ℝ := 34.0

-- Define the number of candies Linda gave away
def candies_given : ℝ := 28.0

-- Define the number of candies Linda has left
def remaining_candies : ℝ := initial_candies - candies_given

-- Theorem statement
theorem linda_remaining_candies :
  remaining_candies = 6.0 := by sorry

end NUMINAMATH_CALUDE_linda_remaining_candies_l761_76178


namespace NUMINAMATH_CALUDE_triangle_angle_A_l761_76146

/-- Given a triangle ABC where C = π/3, b = √6, and c = 3, prove that A = 5π/12 -/
theorem triangle_angle_A (A B C : ℝ) (a b c : ℝ) : 
  C = π/3 → b = Real.sqrt 6 → c = 3 → 
  0 < a ∧ 0 < b ∧ 0 < c →
  A + B + C = π →
  a / (Real.sin A) = b / (Real.sin B) →
  a / (Real.sin A) = c / (Real.sin C) →
  A = 5*π/12 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_A_l761_76146


namespace NUMINAMATH_CALUDE_multiple_solutions_and_no_solution_for_2891_l761_76118

def equation (x y n : ℤ) : Prop := x^3 - 3*x*y^2 + y^3 = n

theorem multiple_solutions_and_no_solution_for_2891 :
  (∀ n : ℤ, (∃ x y : ℤ, equation x y n) → 
    (∃ a b c d : ℤ, equation a b n ∧ equation c d n ∧ 
      (a, b) ≠ (x, y) ∧ (c, d) ≠ (x, y) ∧ (a, b) ≠ (c, d))) ∧
  (¬ ∃ x y : ℤ, equation x y 2891) :=
by sorry

end NUMINAMATH_CALUDE_multiple_solutions_and_no_solution_for_2891_l761_76118


namespace NUMINAMATH_CALUDE_toms_profit_l761_76132

/-- Calculates Tom's profit from lawn mowing and weed pulling -/
def calculate_profit (lawns_mowed : ℕ) (price_per_lawn : ℕ) (gas_expense : ℕ) (weed_pulling_income : ℕ) : ℕ :=
  lawns_mowed * price_per_lawn + weed_pulling_income - gas_expense

/-- Theorem: Tom's profit last month was $29 -/
theorem toms_profit :
  calculate_profit 3 12 17 10 = 29 := by
  sorry

end NUMINAMATH_CALUDE_toms_profit_l761_76132


namespace NUMINAMATH_CALUDE_N_smallest_with_digit_sum_2021_sum_of_digits_N_plus_2021_l761_76175

/-- The smallest positive integer whose digits sum to 2021 -/
def N : ℕ := sorry

/-- Function to calculate the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem stating the property of N -/
theorem N_smallest_with_digit_sum_2021 :
  (∀ m : ℕ, m < N → sum_of_digits m ≠ 2021) ∧
  sum_of_digits N = 2021 := by sorry

/-- Main theorem to prove -/
theorem sum_of_digits_N_plus_2021 :
  sum_of_digits (N + 2021) = 10 := by sorry

end NUMINAMATH_CALUDE_N_smallest_with_digit_sum_2021_sum_of_digits_N_plus_2021_l761_76175


namespace NUMINAMATH_CALUDE_geometric_sum_equals_5592404_l761_76125

/-- The sum of a geometric series with 11 terms, first term 4, and common ratio 4 -/
def geometricSum : ℕ :=
  4 * (1 - 4^11) / (1 - 4)

/-- Theorem stating that the geometric sum is equal to 5592404 -/
theorem geometric_sum_equals_5592404 : geometricSum = 5592404 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sum_equals_5592404_l761_76125


namespace NUMINAMATH_CALUDE_matrix_power_2023_l761_76154

def A : Matrix (Fin 2) (Fin 2) ℤ := !![1, 0; 3, 1]

theorem matrix_power_2023 : 
  A ^ 2023 = !![1, 0; 6069, 1] := by sorry

end NUMINAMATH_CALUDE_matrix_power_2023_l761_76154


namespace NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l761_76161

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 
  (n ≥ 100 ∧ n < 1000) ∧ 
  n % 17 = 0 ∧ 
  (∀ m : ℕ, (m ≥ 100 ∧ m < 1000) ∧ m % 17 = 0 → n ≤ m) ∧
  n = 102 := by
  sorry

end NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l761_76161


namespace NUMINAMATH_CALUDE_t_leq_s_l761_76105

theorem t_leq_s (a b : ℝ) (t s : ℝ) (ht : t = a + 2*b) (hs : s = a + b^2 + 1) : t ≤ s := by
  sorry

end NUMINAMATH_CALUDE_t_leq_s_l761_76105


namespace NUMINAMATH_CALUDE_cos_pi_third_minus_alpha_l761_76191

theorem cos_pi_third_minus_alpha (α : ℝ) (h : Real.sin (π / 6 + α) = 2 / 3) :
  Real.cos (π / 3 - α) = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_cos_pi_third_minus_alpha_l761_76191


namespace NUMINAMATH_CALUDE_double_age_in_two_years_l761_76159

/-- Calculates the number of years until a man's age is twice his son's age -/
def years_until_double_age (man_age_difference : ℕ) (son_current_age : ℕ) : ℕ :=
  let man_current_age := son_current_age + man_age_difference
  2 * son_current_age + 2 - man_current_age

theorem double_age_in_two_years 
  (man_age_difference : ℕ) 
  (son_current_age : ℕ) 
  (h1 : man_age_difference = 25) 
  (h2 : son_current_age = 23) : 
  years_until_double_age man_age_difference son_current_age = 2 := by
sorry

#eval years_until_double_age 25 23

end NUMINAMATH_CALUDE_double_age_in_two_years_l761_76159


namespace NUMINAMATH_CALUDE_only_cylinder_quadrilateral_l761_76102

-- Define the types of geometric solids
inductive GeometricSolid
  | Cone
  | Sphere
  | Cylinder

-- Define the possible shapes of plane sections
inductive PlaneSection
  | Circle
  | Ellipse
  | Parabola
  | Triangle
  | Quadrilateral

-- Function to determine possible plane sections for each solid
def possibleSections (solid : GeometricSolid) : Set PlaneSection :=
  match solid with
  | GeometricSolid.Cone => {PlaneSection.Circle, PlaneSection.Ellipse, PlaneSection.Parabola, PlaneSection.Triangle}
  | GeometricSolid.Sphere => {PlaneSection.Circle}
  | GeometricSolid.Cylinder => {PlaneSection.Circle, PlaneSection.Ellipse, PlaneSection.Quadrilateral}

-- Theorem stating that only a cylinder can produce a quadrilateral section
theorem only_cylinder_quadrilateral :
  ∀ (solid : GeometricSolid),
    PlaneSection.Quadrilateral ∈ possibleSections solid ↔ solid = GeometricSolid.Cylinder :=
by sorry

end NUMINAMATH_CALUDE_only_cylinder_quadrilateral_l761_76102


namespace NUMINAMATH_CALUDE_gcd_lcm_product_24_36_l761_76157

theorem gcd_lcm_product_24_36 : Nat.gcd 24 36 * Nat.lcm 24 36 = 864 := by sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_24_36_l761_76157


namespace NUMINAMATH_CALUDE_total_distance_walked_l761_76198

def first_part : ℝ := 0.75
def second_part : ℝ := 0.25

theorem total_distance_walked : first_part + second_part = 1 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_walked_l761_76198


namespace NUMINAMATH_CALUDE_M_enumeration_l761_76187

def M : Set ℕ := {a | a > 0 ∧ ∃ k : ℤ, 4 / (1 - a) = k}

theorem M_enumeration : M = {2, 3, 5} := by sorry

end NUMINAMATH_CALUDE_M_enumeration_l761_76187


namespace NUMINAMATH_CALUDE_time_to_draw_picture_l761_76180

/-- Proves that the time to draw each picture is 2 hours -/
theorem time_to_draw_picture (num_pictures : ℕ) (coloring_ratio : ℚ) (total_time : ℚ) :
  num_pictures = 10 →
  coloring_ratio = 7/10 →
  total_time = 34 →
  ∃ (draw_time : ℚ), draw_time = 2 ∧ num_pictures * draw_time * (1 + coloring_ratio) = total_time :=
by sorry

end NUMINAMATH_CALUDE_time_to_draw_picture_l761_76180


namespace NUMINAMATH_CALUDE_geometric_sequence_a4_l761_76114

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- State the theorem
theorem geometric_sequence_a4 (a : ℕ → ℝ) :
  is_geometric_sequence a → a 2 = 4 → a 6 = 16 → a 4 = 8 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_a4_l761_76114


namespace NUMINAMATH_CALUDE_solve_sales_problem_l761_76126

def sales_problem (sales1 sales2 sales4 sales5 desired_average : ℕ) : Prop :=
  let total_months : ℕ := 5
  let known_sales : ℕ := sales1 + sales2 + sales4 + sales5
  let total_desired : ℕ := desired_average * total_months
  let sales3 : ℕ := total_desired - known_sales
  sales3 = 7570 ∧ 
  (sales1 + sales2 + sales3 + sales4 + sales5) / total_months = desired_average

theorem solve_sales_problem : 
  sales_problem 5420 5660 6350 6500 6300 := by
  sorry

end NUMINAMATH_CALUDE_solve_sales_problem_l761_76126


namespace NUMINAMATH_CALUDE_num_unique_labelings_eq_30_l761_76186

/-- A cube is a three-dimensional object with 6 faces. -/
structure Cube :=
  (faces : Fin 6 → ℕ)

/-- A labeling of a cube is valid if it uses the numbers 1 to 6 exactly once each. -/
def is_valid_labeling (c : Cube) : Prop :=
  (∀ n : ℕ, n ∈ Finset.range 6 → n + 1 ∈ Finset.image c.faces Finset.univ) ∧
  (∀ f₁ f₂ : Fin 6, f₁ ≠ f₂ → c.faces f₁ ≠ c.faces f₂)

/-- Two labelings are equivalent up to rotation if they can be transformed into each other by rotating the cube. -/
def equivalent_up_to_rotation (c₁ c₂ : Cube) : Prop :=
  ∃ (perm : Equiv.Perm (Fin 6)), ∀ (f : Fin 6), c₁.faces f = c₂.faces (perm f)

/-- The number of unique labelings of a cube up to rotation -/
def num_unique_labelings : ℕ := sorry

theorem num_unique_labelings_eq_30 : num_unique_labelings = 30 := by
  sorry

end NUMINAMATH_CALUDE_num_unique_labelings_eq_30_l761_76186


namespace NUMINAMATH_CALUDE_man_speed_in_still_water_l761_76147

/-- Represents the speed of a swimmer in still water and the speed of the stream -/
structure SwimmingSpeed where
  manSpeed : ℝ  -- Speed of the man in still water
  streamSpeed : ℝ  -- Speed of the stream

/-- Calculates the effective speed for downstream swimming -/
def downstreamSpeed (s : SwimmingSpeed) : ℝ := s.manSpeed + s.streamSpeed

/-- Calculates the effective speed for upstream swimming -/
def upstreamSpeed (s : SwimmingSpeed) : ℝ := s.manSpeed - s.streamSpeed

/-- Theorem stating that given the conditions, the man's speed in still water is 5 km/h -/
theorem man_speed_in_still_water :
  ∀ s : SwimmingSpeed,
    (downstreamSpeed s * 4 = 24) →  -- Downstream condition
    (upstreamSpeed s * 5 = 20) →    -- Upstream condition
    s.manSpeed = 5 := by
  sorry

end NUMINAMATH_CALUDE_man_speed_in_still_water_l761_76147


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l761_76172

/-- Given a hyperbola with equation x²/9 - y²/b² = 1 and foci at (-5,0) and (5,0),
    prove that its asymptotes have the equation 4x ± 3y = 0 -/
theorem hyperbola_asymptotes (b : ℝ) (h1 : b > 0) (h2 : 9 + b^2 = 25) :
  ∃ (k : ℝ), k > 0 ∧ 
  (∀ (x y : ℝ), (x^2 / 9 - y^2 / b^2 = 1) → 
   ((4*x + 3*y = 0) ∨ (4*x - 3*y = 0))) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l761_76172


namespace NUMINAMATH_CALUDE_team_a_more_uniform_l761_76110

/-- Represents a team of girls in a duet --/
structure Team where
  members : Fin 6 → ℝ
  variance : ℝ

/-- The problem setup --/
def problem_setup (team_a team_b : Team) : Prop :=
  team_a.variance = 1.2 ∧ team_b.variance = 2.0

/-- Definition of more uniform heights --/
def more_uniform (team1 team2 : Team) : Prop :=
  team1.variance < team2.variance

/-- The main theorem --/
theorem team_a_more_uniform (team_a team_b : Team) 
  (h : problem_setup team_a team_b) : 
  more_uniform team_a team_b := by
  sorry

#check team_a_more_uniform

end NUMINAMATH_CALUDE_team_a_more_uniform_l761_76110


namespace NUMINAMATH_CALUDE_quadratic_equation_game_l761_76106

/-- Represents a strategy for playing the quadratic equation game -/
def Strategy := Nat → Nat → ℝ

/-- Represents the outcome of a game given two strategies -/
def GameOutcome (n : Nat) (s1 s2 : Strategy) : Nat := sorry

/-- The maximum number of equations without roots that Player 1 can guarantee -/
def MaxRootlessEquations (n : Nat) : Nat := (n + 1) / 2

theorem quadratic_equation_game (n : Nat) (h : Odd n) :
  ∃ (s1 : Strategy), ∀ (s2 : Strategy), GameOutcome n s1 s2 ≥ MaxRootlessEquations n :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_game_l761_76106


namespace NUMINAMATH_CALUDE_train_crossing_time_l761_76104

/-- Calculates the time taken for two trains to cross each other. -/
theorem train_crossing_time (length1 length2 speed1 speed2 initial_distance : ℝ) 
  (h1 : length1 = 135.5)
  (h2 : length2 = 167.2)
  (h3 : speed1 = 55)
  (h4 : speed2 = 43)
  (h5 : initial_distance = 250) :
  ∃ (time : ℝ), (abs (time - 20.3) < 0.1) ∧ 
  (time = (length1 + length2 + initial_distance) / ((speed1 + speed2) * (5/18))) :=
sorry

end NUMINAMATH_CALUDE_train_crossing_time_l761_76104


namespace NUMINAMATH_CALUDE_diamond_equation_solution_l761_76169

/-- Custom binary operation ◇ -/
def diamond (a b : ℚ) : ℚ := a * b + 3 * b - 2 * a

/-- Theorem stating that if 4 ◇ y = 50, then y = 58/7 -/
theorem diamond_equation_solution :
  ∀ y : ℚ, diamond 4 y = 50 → y = 58 / 7 := by
  sorry

end NUMINAMATH_CALUDE_diamond_equation_solution_l761_76169


namespace NUMINAMATH_CALUDE_vector_inequality_l761_76113

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Given two non-zero vectors a and b satisfying |a + b| = |b|, prove |2b| > |a + 2b| -/
theorem vector_inequality (a b : E) (ha : a ≠ 0) (hb : b ≠ 0) (h : ‖a + b‖ = ‖b‖) :
  ‖(2 : ℝ) • b‖ > ‖a + (2 : ℝ) • b‖ := by sorry

end NUMINAMATH_CALUDE_vector_inequality_l761_76113


namespace NUMINAMATH_CALUDE_dandelion_color_change_l761_76181

/-- The number of dandelions that turned white in the first two days -/
def dandelions_turned_white_first_two_days : ℕ := 25

/-- The number of dandelions that will turn white on the fourth day -/
def dandelions_turn_white_fourth_day : ℕ := 9

/-- The total number of dandelions that have turned or will turn white over the four-day period -/
def total_white_dandelions : ℕ := dandelions_turned_white_first_two_days + dandelions_turn_white_fourth_day

theorem dandelion_color_change :
  total_white_dandelions = 34 := by sorry

end NUMINAMATH_CALUDE_dandelion_color_change_l761_76181


namespace NUMINAMATH_CALUDE_arcsin_sqrt2_over_2_l761_76155

theorem arcsin_sqrt2_over_2 : Real.arcsin (Real.sqrt 2 / 2) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_sqrt2_over_2_l761_76155


namespace NUMINAMATH_CALUDE_diamond_neg_one_six_l761_76170

-- Define the diamond operation
def diamond (a b : ℤ) : ℤ := a * b^2 - b + 1

-- Theorem statement
theorem diamond_neg_one_six : diamond (-1) 6 = -41 := by
  sorry

end NUMINAMATH_CALUDE_diamond_neg_one_six_l761_76170


namespace NUMINAMATH_CALUDE_min_value_expression_l761_76185

theorem min_value_expression (a θ : ℝ) : 
  (a - 2 * Real.cos θ)^2 + (a - 5 * Real.sqrt 2 - 2 * Real.sin θ)^2 ≥ 9 ∧
  ∃ a θ : ℝ, (a - 2 * Real.cos θ)^2 + (a - 5 * Real.sqrt 2 - 2 * Real.sin θ)^2 = 9 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l761_76185


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l761_76148

/-- Two vectors in R² are parallel if their components are proportional -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

/-- Given vectors a and b, prove that if they are parallel, then x = 6 -/
theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (4, 2)
  let b : ℝ × ℝ := (x, 3)
  are_parallel a b → x = 6 :=
by
  sorry


end NUMINAMATH_CALUDE_parallel_vectors_x_value_l761_76148


namespace NUMINAMATH_CALUDE_point_on_line_l761_76138

/-- Given that point A (3, a) lies on the line 2x + y - 7 = 0, prove that a = 1 -/
theorem point_on_line (a : ℝ) : 2 * 3 + a - 7 = 0 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_l761_76138


namespace NUMINAMATH_CALUDE_triangle_side_cube_l761_76120

/-- Given a triangle ABC with positive integer side lengths a, b, and c, 
    where gcd(a,b,c) = 1 and ∠A = 3∠B, at least one of a, b, and c is a cube. -/
theorem triangle_side_cube (a b c : ℕ+) (angleA angleB : ℝ) : 
  (a.val.gcd (b.val.gcd c.val) = 1) →
  (angleA = 3 * angleB) →
  (∃ (x : ℕ+), x^3 = a ∨ x^3 = b ∨ x^3 = c) := by
sorry

end NUMINAMATH_CALUDE_triangle_side_cube_l761_76120


namespace NUMINAMATH_CALUDE_stratified_sampling_proportionality_l761_76197

/-- Represents the number of students selected in stratified sampling -/
structure StratifiedSample where
  total : ℕ
  first_year : ℕ
  second_year : ℕ
  selected_first : ℕ
  selected_second : ℕ

/-- Checks if the stratified sample maintains proportionality -/
def is_proportional (s : StratifiedSample) : Prop :=
  s.selected_first * s.second_year = s.selected_second * s.first_year

theorem stratified_sampling_proportionality :
  ∀ s : StratifiedSample,
    s.total = 70 →
    s.first_year = 30 →
    s.second_year = 40 →
    s.selected_first = 6 →
    s.selected_second = 8 →
    is_proportional s :=
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_proportionality_l761_76197


namespace NUMINAMATH_CALUDE_rectangle_area_l761_76183

/-- A rectangle with perimeter 40 and length twice its width has area 800/9 -/
theorem rectangle_area (w : ℝ) (h1 : w > 0) (h2 : 6 * w = 40) : w * (2 * w) = 800 / 9 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l761_76183


namespace NUMINAMATH_CALUDE_barbara_shopping_cost_l761_76167

/-- The amount Barbara spent on goods other than tuna and water -/
def other_goods_cost (tuna_packs : ℕ) (tuna_price : ℚ) (water_bottles : ℕ) (water_price : ℚ) (total_paid : ℚ) : ℚ :=
  total_paid - (tuna_packs : ℚ) * tuna_price - (water_bottles : ℚ) * water_price

/-- Theorem stating that Barbara spent $40 on goods other than tuna and water -/
theorem barbara_shopping_cost :
  other_goods_cost 5 2 4 (3/2) 56 = 40 := by
  sorry

end NUMINAMATH_CALUDE_barbara_shopping_cost_l761_76167


namespace NUMINAMATH_CALUDE_range_of_a_l761_76156

/-- The function f(x) = x^2 - 2x --/
def f (x : ℝ) : ℝ := x^2 - 2*x

/-- The function g(x) = ax + 2, where a > 0 --/
def g (a : ℝ) (x : ℝ) : ℝ := a*x + 2

/-- The closed interval [-1, 2] --/
def I : Set ℝ := Set.Icc (-1) 2

theorem range_of_a :
  ∀ a : ℝ, (a > 0 ∧
    (∀ x₁ ∈ I, ∃ x₀ ∈ I, g a x₁ = f x₀)) ↔
    (a ∈ Set.Ioo 0 (1/2)) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l761_76156


namespace NUMINAMATH_CALUDE_smallest_number_of_editors_l761_76141

/-- The total number of people at the conference -/
def total : ℕ := 90

/-- The number of writers at the conference -/
def writers : ℕ := 45

/-- The number of people who are both writers and editors -/
def both : ℕ := 6

/-- The number of people who are neither writers nor editors -/
def neither : ℕ := 2 * both

/-- The number of editors at the conference -/
def editors : ℕ := total - writers - neither + both

theorem smallest_number_of_editors : editors = 39 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_of_editors_l761_76141


namespace NUMINAMATH_CALUDE_opposite_numbers_solution_l761_76192

theorem opposite_numbers_solution (x : ℚ) : (2 * x - 3 = -(1 - 4 * x)) → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_opposite_numbers_solution_l761_76192


namespace NUMINAMATH_CALUDE_intersection_M_N_l761_76152

-- Define the sets M and N
def M : Set ℝ := {x | |x - 1| > 1}
def N : Set ℝ := {x | x^2 - 3*x ≤ 0}

-- State the theorem
theorem intersection_M_N : M ∩ N = {x | 2 < x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l761_76152


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l761_76145

/-- A quadratic function satisfying certain conditions -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ),
    a ≠ 0 ∧
    (∀ x, f x = a * x^2 + b * x + c) ∧
    f (-1) = 0 ∧
    (∀ x, x ≤ f x ∧ f x ≤ (x^2 + 1) / 2) ∧
    {x : ℝ | |f x| < 1} = Set.Ioo (-1) 3

/-- The theorem stating the properties of the quadratic function -/
theorem quadratic_function_properties (f : ℝ → ℝ) (h : QuadraticFunction f) :
  (∀ x, f x = (1/4) * (x + 1)^2) ∧
  (∃ a : ℝ, (a > 0 ∧ a < 1/2) ∨ (a < 0 ∧ a > -1/2)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l761_76145


namespace NUMINAMATH_CALUDE_unique_digit_divisibility_l761_76133

def is_divisible (n m : ℕ) : Prop := ∃ k : ℕ, n = m * k

theorem unique_digit_divisibility :
  ∃! B : ℕ,
    B < 10 ∧
    let number := 658274 * 10 + B
    is_divisible number 2 ∧
    is_divisible number 4 ∧
    is_divisible number 5 ∧
    is_divisible number 7 ∧
    is_divisible number 8 :=
by sorry

end NUMINAMATH_CALUDE_unique_digit_divisibility_l761_76133


namespace NUMINAMATH_CALUDE_parabola_properties_l761_76119

-- Define the parabola and its properties
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  m : ℝ
  h_a_neg : a < 0
  h_m_bounds : 1 < m ∧ m < 2
  h_passes_through : a * (-1)^2 + b * (-1) + c = 0 ∧ a * m^2 + b * m + c = 0

-- Theorem statements
theorem parabola_properties (p : Parabola) :
  (p.b > 0) ∧
  (∀ x₁ x₂ y₁ y₂ : ℝ, 
    x₁ < x₂ → x₁ + x₂ > 1 → 
    p.a * x₁^2 + p.b * x₁ + p.c = y₁ → 
    p.a * x₂^2 + p.b * x₂ + p.c = y₂ → 
    y₁ > y₂) ∧
  (p.a ≤ -1 → ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ p.a * x₁^2 + p.b * x₁ + p.c = 1 ∧ p.a * x₂^2 + p.b * x₂ + p.c = 1) :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l761_76119


namespace NUMINAMATH_CALUDE_complex_magnitude_equation_l761_76127

theorem complex_magnitude_equation (t : ℝ) (h : t > 0) :
  Complex.abs (8 + 2 * t * Complex.I) = 12 → t = 2 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_equation_l761_76127


namespace NUMINAMATH_CALUDE_melanie_plums_l761_76168

/-- The number of plums Melanie has after giving some away -/
def plums_remaining (initial : ℕ) (given_away : ℕ) : ℕ :=
  initial - given_away

/-- Theorem: Melanie has 4 plums after initially picking 7 and giving 3 away -/
theorem melanie_plums : plums_remaining 7 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_melanie_plums_l761_76168


namespace NUMINAMATH_CALUDE_fraction_evaluation_l761_76112

theorem fraction_evaluation (x y : ℝ) (hx : x = 4) (hy : y = 5) :
  ((1 / y^2) / (1 / x^2))^2 = 256 / 625 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l761_76112


namespace NUMINAMATH_CALUDE_lampshire_parade_group_size_l761_76177

theorem lampshire_parade_group_size (n : ℕ) : 
  (∃ k : ℕ, n = 30 * k) →
  (30 * n) % 31 = 7 →
  (30 * n) % 17 = 0 →
  30 * n < 1500 →
  (∀ m : ℕ, 
    (∃ j : ℕ, m = 30 * j) →
    (30 * m) % 31 = 7 →
    (30 * m) % 17 = 0 →
    30 * m < 1500 →
    30 * m ≤ 30 * n) →
  30 * n = 1020 :=
by sorry

end NUMINAMATH_CALUDE_lampshire_parade_group_size_l761_76177


namespace NUMINAMATH_CALUDE_sampling_survey_more_appropriate_for_city_air_quality_l761_76195

-- Define the city and survey types
def City : Type := Unit
def ComprehensiveSurvey : Type := Unit
def SamplingSurvey : Type := Unit

-- Define the properties of the city and surveys
def has_vast_area (c : City) : Prop := sorry
def has_varying_conditions (c : City) : Prop := sorry
def is_comprehensive (s : ComprehensiveSurvey) : Prop := sorry
def is_strategically_sampled (s : SamplingSurvey) : Prop := sorry

-- Define the concept of feasibility and appropriateness
def is_feasible (c : City) (s : ComprehensiveSurvey) : Prop := sorry
def is_appropriate (c : City) (s : SamplingSurvey) : Prop := sorry

-- Theorem stating that sampling survey is more appropriate for air quality testing in a city
theorem sampling_survey_more_appropriate_for_city_air_quality 
  (c : City) (comp_survey : ComprehensiveSurvey) (samp_survey : SamplingSurvey) :
  has_vast_area c →
  has_varying_conditions c →
  is_comprehensive comp_survey →
  is_strategically_sampled samp_survey →
  ¬(is_feasible c comp_survey) →
  is_appropriate c samp_survey :=
by sorry

end NUMINAMATH_CALUDE_sampling_survey_more_appropriate_for_city_air_quality_l761_76195


namespace NUMINAMATH_CALUDE_unique_solution_proof_l761_76129

/-- The positive value of m for which the quadratic equation 4x^2 + mx + 4 = 0 has exactly one real solution -/
def unique_solution_m : ℝ := 8

/-- The quadratic equation 4x^2 + mx + 4 = 0 -/
def quadratic_equation (m : ℝ) (x : ℝ) : Prop :=
  4 * x^2 + m * x + 4 = 0

/-- The discriminant of the quadratic equation -/
def discriminant (m : ℝ) : ℝ :=
  m^2 - 4 * 4 * 4

theorem unique_solution_proof :
  unique_solution_m > 0 ∧
  discriminant unique_solution_m = 0 ∧
  ∀ m : ℝ, m > 0 → discriminant m = 0 → m = unique_solution_m :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_proof_l761_76129


namespace NUMINAMATH_CALUDE_katherines_bananas_l761_76153

/-- Given Katherine's fruit inventory, calculate the number of bananas -/
theorem katherines_bananas (apples pears bananas total : ℕ) : 
  apples = 4 →
  pears = 3 * apples →
  total = apples + pears + bananas →
  total = 21 →
  bananas = 5 := by
sorry

end NUMINAMATH_CALUDE_katherines_bananas_l761_76153


namespace NUMINAMATH_CALUDE_unique_triangle_number_three_identical_digits_l761_76135

/-- The sum of the first n positive integers -/
def triangle_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Checks if a number is a three-digit number composed of identical digits -/
def is_three_identical_digits (n : ℕ) : Prop :=
  ∃ (d : ℕ), d > 0 ∧ d < 10 ∧ n = d * 111

theorem unique_triangle_number_three_identical_digits :
  ∃! (n : ℕ), n > 0 ∧ is_three_identical_digits (triangle_number n) :=
sorry

end NUMINAMATH_CALUDE_unique_triangle_number_three_identical_digits_l761_76135


namespace NUMINAMATH_CALUDE_smallest_base_for_perfect_fourth_power_l761_76151

/-- Given that 5n is a positive integer represented as 777 in base b,
    and n is a perfect fourth power, prove that the smallest positive
    integer b satisfying these conditions is 41. -/
theorem smallest_base_for_perfect_fourth_power (n : ℕ) (b : ℕ) : 
  (5 * n : ℕ) > 0 ∧ 
  (5 * n = 7 * b^2 + 7 * b + 7) ∧
  (∃ (x : ℕ), n = x^4) →
  (∀ (b' : ℕ), b' ≥ 1 ∧ 
    (∃ (n' : ℕ), (5 * n' : ℕ) > 0 ∧ 
      (5 * n' = 7 * b'^2 + 7 * b' + 7) ∧
      (∃ (x : ℕ), n' = x^4)) →
    b' ≥ b) ∧
  b = 41 :=
by sorry

end NUMINAMATH_CALUDE_smallest_base_for_perfect_fourth_power_l761_76151


namespace NUMINAMATH_CALUDE_max_points_difference_between_adjacent_teams_l761_76139

/-- Represents a football league with the given properties -/
structure FootballLeague where
  num_teams : Nat
  points_for_win : Nat
  points_for_draw : Nat
  points_for_loss : Nat

/-- Calculates the maximum points a team can achieve in the league -/
def max_points (league : FootballLeague) : Nat :=
  (league.num_teams - 1) * 2 * league.points_for_win

/-- Calculates the minimum points a team can achieve in the league -/
def min_points (league : FootballLeague) : Nat :=
  (league.num_teams - 1) * 2 * league.points_for_draw

/-- Theorem stating the maximum points difference between adjacent teams -/
theorem max_points_difference_between_adjacent_teams 
  (league : FootballLeague) 
  (h1 : league.num_teams = 12)
  (h2 : league.points_for_win = 2)
  (h3 : league.points_for_draw = 1)
  (h4 : league.points_for_loss = 0) :
  max_points league - min_points league = 24 := by
  sorry


end NUMINAMATH_CALUDE_max_points_difference_between_adjacent_teams_l761_76139


namespace NUMINAMATH_CALUDE_deck_size_l761_76182

theorem deck_size (r b : ℕ) : 
  (r : ℚ) / (r + b) = 1/4 →
  (r : ℚ) / (r + b + 6) = 1/6 →
  r + b = 12 := by
sorry

end NUMINAMATH_CALUDE_deck_size_l761_76182


namespace NUMINAMATH_CALUDE_alice_stops_l761_76124

/-- Represents the coefficients of a quadratic equation ax² + bx + c = 0 -/
structure QuadraticCoeffs where
  a : ℤ
  b : ℤ
  c : ℤ

/-- Transformation rule for the quadratic coefficients -/
def transform (q : QuadraticCoeffs) : QuadraticCoeffs :=
  { a := q.b + q.c
  , b := q.c + q.a
  , c := q.a + q.b }

/-- Sequence of quadratic coefficients after n transformations -/
def coeff_seq (q₀ : QuadraticCoeffs) : ℕ → QuadraticCoeffs
  | 0 => q₀
  | n + 1 => transform (coeff_seq q₀ n)

/-- Predicate to check if a quadratic equation has real roots -/
def has_real_roots (q : QuadraticCoeffs) : Prop :=
  q.b ^ 2 ≥ 4 * q.a * q.c

/-- Main theorem: Alice will stop after a finite number of moves -/
theorem alice_stops (q₀ : QuadraticCoeffs)
  (h₁ : (q₀.a + q₀.c) * q₀.b > 0) :
  ∃ k : ℕ, ¬(has_real_roots (coeff_seq q₀ k)) := by
  sorry

end NUMINAMATH_CALUDE_alice_stops_l761_76124


namespace NUMINAMATH_CALUDE_second_number_proof_l761_76122

theorem second_number_proof (x : ℕ) : 
  (∃ k : ℕ, 60 = 18 * k + 6) →
  (∃ m : ℕ, x = 18 * m + 10) →
  (∀ d : ℕ, d > 18 → (d ∣ 60 ∧ d ∣ x) → False) →
  x > 60 →
  x = 64 := by
sorry

end NUMINAMATH_CALUDE_second_number_proof_l761_76122


namespace NUMINAMATH_CALUDE_team_selection_ways_l761_76143

def boys : ℕ := 10
def girls : ℕ := 10
def team_size : ℕ := 8
def boys_in_team : ℕ := team_size / 2
def girls_in_team : ℕ := team_size / 2

theorem team_selection_ways : 
  (Nat.choose boys boys_in_team) * (Nat.choose girls girls_in_team) = 44100 := by
  sorry

end NUMINAMATH_CALUDE_team_selection_ways_l761_76143


namespace NUMINAMATH_CALUDE_chord_intersection_lengths_l761_76150

theorem chord_intersection_lengths (r : ℝ) (AK CH : ℝ) :
  r = 7 →
  AK = 3 →
  CH = 12 →
  let KB := 2 * r - AK
  ∃ (CK KH : ℝ),
    CK + KH = CH ∧
    AK * KB = CK * KH ∧
    AK = 3 ∧
    KB = 11 := by
  sorry

end NUMINAMATH_CALUDE_chord_intersection_lengths_l761_76150


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l761_76121

theorem consecutive_integers_sum (x : ℤ) : 
  x * (x + 1) * (x + 2) = 384 → x + (x + 1) + (x + 2) = 24 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l761_76121


namespace NUMINAMATH_CALUDE_journey_distance_l761_76115

theorem journey_distance (train_fraction : ℚ) (bus_fraction : ℚ) (walk_distance : ℝ) :
  train_fraction = 3/5 →
  bus_fraction = 7/20 →
  walk_distance = 6.5 →
  1 - (train_fraction + bus_fraction) = walk_distance / 130 →
  130 = (walk_distance * 20 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_journey_distance_l761_76115


namespace NUMINAMATH_CALUDE_january_oil_bill_l761_76130

theorem january_oil_bill (january february : ℝ) 
  (h1 : february / january = 5 / 4)
  (h2 : (february + 30) / january = 3 / 2) :
  january = 120 := by
sorry

end NUMINAMATH_CALUDE_january_oil_bill_l761_76130


namespace NUMINAMATH_CALUDE_zero_real_necessary_not_sufficient_for_purely_imaginary_l761_76100

/-- A complex number is purely imaginary if its real part is zero -/
def isPurelyImaginary (z : ℂ) : Prop := z.re = 0

theorem zero_real_necessary_not_sufficient_for_purely_imaginary :
  ∃ (a b : ℝ), (isPurelyImaginary (Complex.mk a b) → a = 0) ∧
                ¬(a = 0 → isPurelyImaginary (Complex.mk a b)) :=
by sorry

end NUMINAMATH_CALUDE_zero_real_necessary_not_sufficient_for_purely_imaginary_l761_76100


namespace NUMINAMATH_CALUDE_y_value_theorem_l761_76131

theorem y_value_theorem (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (eq1 : x = 2 + 1 / y) (eq2 : y = 2 + 1 / x) :
  y = 1 + Real.sqrt 2 ∨ y = 1 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_y_value_theorem_l761_76131


namespace NUMINAMATH_CALUDE_class_mood_distribution_l761_76103

theorem class_mood_distribution (total_children : Nat) (happy_children : Nat) (sad_children : Nat) (anxious_children : Nat)
  (total_boys : Nat) (total_girls : Nat) (happy_boys : Nat) (sad_girls : Nat) (anxious_girls : Nat)
  (h1 : total_children = 80)
  (h2 : happy_children = 25)
  (h3 : sad_children = 15)
  (h4 : anxious_children = 20)
  (h5 : total_boys = 35)
  (h6 : total_girls = 45)
  (h7 : happy_boys = 10)
  (h8 : sad_girls = 6)
  (h9 : anxious_girls = 12)
  (h10 : total_children = total_boys + total_girls) :
  (total_boys - (happy_boys + (sad_children - sad_girls) + (anxious_children - anxious_girls)) = 8) ∧
  (happy_children - happy_boys = 15) :=
by sorry

end NUMINAMATH_CALUDE_class_mood_distribution_l761_76103


namespace NUMINAMATH_CALUDE_three_quarters_difference_l761_76164

theorem three_quarters_difference (n : ℕ) (h : n = 76) : n - (3 * n / 4) = 19 := by
  sorry

end NUMINAMATH_CALUDE_three_quarters_difference_l761_76164


namespace NUMINAMATH_CALUDE_line_intersections_l761_76140

/-- The line equation 4y - 5x = 20 -/
def line_equation (x y : ℝ) : Prop := 4 * y - 5 * x = 20

/-- The x-axis intercept of the line -/
def x_intercept : ℝ × ℝ := (-4, 0)

/-- The y-axis intercept of the line -/
def y_intercept : ℝ × ℝ := (0, 5)

/-- Theorem stating that the line intersects the x-axis and y-axis at the given points -/
theorem line_intersections :
  (line_equation x_intercept.1 x_intercept.2) ∧
  (line_equation y_intercept.1 y_intercept.2) :=
by sorry

end NUMINAMATH_CALUDE_line_intersections_l761_76140


namespace NUMINAMATH_CALUDE_system_solution_value_l761_76173

theorem system_solution_value (a b x y : ℝ) : 
  x = 2 ∧ 
  y = 1 ∧ 
  a * x + b * y = 5 ∧ 
  b * x + a * y = 1 → 
  3 - a - b = 1 := by
sorry

end NUMINAMATH_CALUDE_system_solution_value_l761_76173


namespace NUMINAMATH_CALUDE_emily_sees_emerson_time_l761_76184

def emily_speed : ℝ := 15
def emerson_speed : ℝ := 9
def initial_distance : ℝ := 1
def final_distance : ℝ := 1

theorem emily_sees_emerson_time : 
  let relative_speed := emily_speed - emerson_speed
  let time_to_catch := initial_distance / relative_speed
  let time_to_lose_sight := final_distance / relative_speed
  let total_time := time_to_catch + time_to_lose_sight
  (total_time * 60) = 20 := by sorry

end NUMINAMATH_CALUDE_emily_sees_emerson_time_l761_76184


namespace NUMINAMATH_CALUDE_ferris_wheel_large_seats_undetermined_l761_76149

structure FerrisWheel where
  smallSeats : Nat
  smallSeatCapacity : Nat
  largeSeatCapacity : Nat
  peopleOnSmallSeats : Nat

theorem ferris_wheel_large_seats_undetermined (fw : FerrisWheel)
  (h1 : fw.smallSeats = 2)
  (h2 : fw.smallSeatCapacity = 14)
  (h3 : fw.largeSeatCapacity = 54)
  (h4 : fw.peopleOnSmallSeats = 28) :
  ∀ n : Nat, ∃ m : Nat, m ≠ n ∧ 
    (∃ totalSeats totalCapacity : Nat,
      totalSeats = fw.smallSeats + n ∧
      totalCapacity = fw.smallSeats * fw.smallSeatCapacity + m * fw.largeSeatCapacity) :=
sorry

end NUMINAMATH_CALUDE_ferris_wheel_large_seats_undetermined_l761_76149


namespace NUMINAMATH_CALUDE_processing_box_function_is_assignment_and_calculation_l761_76101

/-- Represents the possible functions of a processing box in an algorithm -/
inductive ProcessingBoxFunction
  | startIndicator
  | inputIndicator
  | assignmentAndCalculation
  | conditionJudgment

/-- The function of a processing box -/
def processingBoxFunction : ProcessingBoxFunction :=
  ProcessingBoxFunction.assignmentAndCalculation

/-- Theorem stating that the function of a processing box is assignment and calculation -/
theorem processing_box_function_is_assignment_and_calculation :
  processingBoxFunction = ProcessingBoxFunction.assignmentAndCalculation :=
by sorry

end NUMINAMATH_CALUDE_processing_box_function_is_assignment_and_calculation_l761_76101


namespace NUMINAMATH_CALUDE_intersection_line_l761_76189

/-- Definition of the first circle -/
def circle1 (x y : ℝ) : Prop :=
  (x + 5)^2 + (y - 3)^2 = 100

/-- Definition of the second circle -/
def circle2 (x y : ℝ) : Prop :=
  (x - 4)^2 + (y + 6)^2 = 121

/-- Theorem stating that the line passing through the intersection points of the two circles
    has the equation x - y = -17/9 -/
theorem intersection_line : ∃ (x y : ℝ), 
  circle1 x y ∧ circle2 x y ∧ (x - y = -17/9) := by
  sorry

end NUMINAMATH_CALUDE_intersection_line_l761_76189


namespace NUMINAMATH_CALUDE_factorization_of_2a_squared_minus_8_l761_76199

theorem factorization_of_2a_squared_minus_8 (a : ℝ) : 2 * a^2 - 8 = 2 * (a + 2) * (a - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_2a_squared_minus_8_l761_76199


namespace NUMINAMATH_CALUDE_cake_frosting_theorem_l761_76123

/-- Represents a person who can frost cakes -/
structure FrostingPerson where
  name : String
  frostingTime : ℕ

/-- Represents the cake frosting problem -/
structure CakeFrostingProblem where
  people : List FrostingPerson
  numCakes : ℕ
  passingTime : ℕ

/-- Calculates the minimum time to frost all cakes -/
def minFrostingTime (problem : CakeFrostingProblem) : ℕ :=
  sorry

theorem cake_frosting_theorem (problem : CakeFrostingProblem) :
  problem.people = [
    { name := "Ann", frostingTime := 8 },
    { name := "Bob", frostingTime := 6 },
    { name := "Carol", frostingTime := 10 }
  ] ∧
  problem.numCakes = 10 ∧
  problem.passingTime = 1
  →
  minFrostingTime problem = 116 := by
  sorry

end NUMINAMATH_CALUDE_cake_frosting_theorem_l761_76123


namespace NUMINAMATH_CALUDE_triangle_count_on_circle_l761_76109

theorem triangle_count_on_circle (n : ℕ) (h : n = 10) : 
  (n.choose 3) = 120 := by
  sorry

end NUMINAMATH_CALUDE_triangle_count_on_circle_l761_76109


namespace NUMINAMATH_CALUDE_circle_containment_l761_76160

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

/-- A point is inside a circle if its distance from the center is less than the radius -/
def is_inside (p : ℝ × ℝ) (c : Circle) : Prop :=
  Real.sqrt ((p.1 - c.center.1)^2 + (p.2 - c.center.2)^2) < c.radius

/-- A circle contains another circle's center if the center is inside the circle -/
def contains_center (c1 c2 : Circle) : Prop :=
  is_inside c2.center c1

theorem circle_containment (circles : Fin 6 → Circle) (O : ℝ × ℝ)
  (h : ∀ i : Fin 6, is_inside O (circles i)) :
  ∃ i j : Fin 6, i ≠ j ∧ contains_center (circles i) (circles j) := by
  sorry

end NUMINAMATH_CALUDE_circle_containment_l761_76160


namespace NUMINAMATH_CALUDE_ball_hitting_ground_time_l761_76190

/-- The time when the ball hits the ground, given the initial conditions and equation of motion -/
theorem ball_hitting_ground_time : ∃ t : ℝ, t > 0 ∧ -16 * t^2 + 32 * t + 180 = 0 ∧ t = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_ball_hitting_ground_time_l761_76190


namespace NUMINAMATH_CALUDE_ratio_max_min_sequence_diff_l761_76107

def geometric_sequence (n : ℕ) : ℚ :=
  (3/2) * (-1/2) ^ (n - 1)

def sum_n_terms (n : ℕ) : ℚ :=
  (3/2) * (1 - (-1/2)^n) / (1 + 1/2)

def sequence_diff (n : ℕ) : ℚ :=
  sum_n_terms n - 1 / sum_n_terms n

theorem ratio_max_min_sequence_diff :
  (∃ (m n : ℕ), m > 0 ∧ n > 0 ∧
    sequence_diff m / sequence_diff n = -10/7 ∧
    ∀ (k : ℕ), k > 0 → 
      sequence_diff m ≥ sequence_diff k ∧
      sequence_diff k ≥ sequence_diff n) :=
sorry

end NUMINAMATH_CALUDE_ratio_max_min_sequence_diff_l761_76107


namespace NUMINAMATH_CALUDE_cricket_average_proof_l761_76171

def average_runs (total_runs : ℕ) (innings : ℕ) : ℚ :=
  (total_runs : ℚ) / (innings : ℚ)

theorem cricket_average_proof 
  (initial_innings : ℕ) 
  (next_innings_runs : ℕ) 
  (average_increase : ℚ) :
  initial_innings = 10 →
  next_innings_runs = 74 →
  average_increase = 4 →
  ∃ (initial_total_runs : ℕ),
    average_runs (initial_total_runs + next_innings_runs) (initial_innings + 1) =
    average_runs initial_total_runs initial_innings + average_increase →
    average_runs initial_total_runs initial_innings = 30 := by
  sorry

end NUMINAMATH_CALUDE_cricket_average_proof_l761_76171


namespace NUMINAMATH_CALUDE_not_perfect_square_l761_76111

theorem not_perfect_square (n : ℕ+) : ¬ ∃ m : ℤ, (2551 * 543^n.val - 2008 * 7^n.val : ℤ) = m^2 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_l761_76111


namespace NUMINAMATH_CALUDE_earth_inhabitable_fraction_l761_76116

/-- The fraction of Earth's surface that humans can inhabit -/
def inhabitable_fraction : ℚ := 1/4

theorem earth_inhabitable_fraction :
  (earth_land_fraction : ℚ) = 1/3 →
  (habitable_land_fraction : ℚ) = 3/4 →
  inhabitable_fraction = earth_land_fraction * habitable_land_fraction :=
by sorry

end NUMINAMATH_CALUDE_earth_inhabitable_fraction_l761_76116


namespace NUMINAMATH_CALUDE_sqrt_expressions_l761_76166

theorem sqrt_expressions :
  (∀ x y z : ℝ, x = 8 ∧ y = 2 ∧ z = 18 → Real.sqrt x + Real.sqrt y - Real.sqrt z = 0) ∧
  (∀ a : ℝ, a = 3 → (Real.sqrt a - 2)^2 = 7 - 4 * Real.sqrt a) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expressions_l761_76166


namespace NUMINAMATH_CALUDE_division_with_special_remainder_l761_76142

theorem division_with_special_remainder :
  ∃! (n : ℕ), n > 0 ∧ 
    ∃ (k m : ℕ), 
      180 = n * k + m ∧ 
      4 * m = k ∧ 
      m < n ∧ 
      n = 11 := by
  sorry

end NUMINAMATH_CALUDE_division_with_special_remainder_l761_76142


namespace NUMINAMATH_CALUDE_violin_enjoyment_misreporting_l761_76194

/-- Represents the student population at Peculiar Academy -/
def TotalStudents : ℝ := 100

/-- Fraction of students who enjoy playing the violin -/
def EnjoyViolin : ℝ := 0.4

/-- Fraction of students who do not enjoy playing the violin -/
def DislikeViolin : ℝ := 0.6

/-- Fraction of violin-enjoying students who accurately state they enjoy it -/
def AccurateEnjoy : ℝ := 0.7

/-- Fraction of violin-enjoying students who falsely claim they do not enjoy it -/
def FalseDislike : ℝ := 0.3

/-- Fraction of violin-disliking students who correctly claim they dislike it -/
def AccurateDislike : ℝ := 0.8

/-- Fraction of violin-disliking students who mistakenly say they like it -/
def FalseLike : ℝ := 0.2

theorem violin_enjoyment_misreporting :
  let enjoy_but_say_dislike := EnjoyViolin * FalseDislike * TotalStudents
  let total_say_dislike := EnjoyViolin * FalseDislike * TotalStudents + DislikeViolin * AccurateDislike * TotalStudents
  enjoy_but_say_dislike / total_say_dislike = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_violin_enjoyment_misreporting_l761_76194


namespace NUMINAMATH_CALUDE_school_network_connections_l761_76134

/-- The number of connections in a network of switches where each switch connects to a fixed number of others -/
def connections (n : ℕ) (k : ℕ) : ℕ := n * k / 2

/-- Theorem: In a network of 30 switches, where each switch connects to exactly 4 others, there are 60 connections -/
theorem school_network_connections :
  connections 30 4 = 60 := by
  sorry

end NUMINAMATH_CALUDE_school_network_connections_l761_76134


namespace NUMINAMATH_CALUDE_tangent_line_equation_l761_76136

/-- The equation of the line passing through the tangency points of two tangent lines drawn from a point to a circle. -/
theorem tangent_line_equation (P : ℝ × ℝ) (r : ℝ) :
  P = (5, 3) →
  r = 3 →
  ∃ A B : ℝ × ℝ,
    (A.1^2 + A.2^2 = r^2) ∧
    (B.1^2 + B.2^2 = r^2) ∧
    ((A.1 - P.1)^2 + (A.2 - P.2)^2 = ((P.1)^2 + (P.2)^2 - r^2)) ∧
    ((B.1 - P.1)^2 + (B.2 - P.2)^2 = ((P.1)^2 + (P.2)^2 - r^2)) ∧
    (∀ x y : ℝ, 5*x + 3*y - 9 = 0 ↔ (x - A.1)*(B.2 - A.2) = (y - A.2)*(B.1 - A.1)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l761_76136


namespace NUMINAMATH_CALUDE_max_sum_squares_l761_76196

theorem max_sum_squares (m n : ℕ) : 
  m ∈ Finset.range 1982 →
  n ∈ Finset.range 1982 →
  ((n^2 : ℤ) - m*n - m^2)^2 = 1 →
  m^2 + n^2 ≤ 3524578 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_squares_l761_76196


namespace NUMINAMATH_CALUDE_interval_of_increase_l761_76144

def f (x : ℝ) := 2 * x^3 + 3 * x^2 - 12 * x + 1

theorem interval_of_increase (x : ℝ) :
  StrictMonoOn f (Set.Iio (-2) ∪ Set.Ioi 1) :=
sorry

end NUMINAMATH_CALUDE_interval_of_increase_l761_76144


namespace NUMINAMATH_CALUDE_area_ADC_approx_l761_76137

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the angle bisector AD
def angleBisector (t : Triangle) : ℝ × ℝ := sorry

-- Define the properties of the triangle
def isRightTriangle (t : Triangle) : Prop := sorry
def hasAngleBisector (t : Triangle) : Prop := sorry
def sideAB (t : Triangle) : ℝ := sorry
def sideBC (t : Triangle) : ℝ := sorry
def sideAC (t : Triangle) : ℝ := sorry

-- Define the area calculation function
def areaADC (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem area_ADC_approx (t : Triangle) 
  (h1 : isRightTriangle t)
  (h2 : hasAngleBisector t)
  (h3 : sideAB t = 80)
  (h4 : ∃ x, sideBC t = x ∧ sideAC t = 2*x - 10) :
  ∃ ε > 0, |areaADC t - 949| < ε :=
sorry

end NUMINAMATH_CALUDE_area_ADC_approx_l761_76137


namespace NUMINAMATH_CALUDE_fraction_evaluation_l761_76158

theorem fraction_evaluation : (1/4 - 1/6) / (1/3 - 1/4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l761_76158


namespace NUMINAMATH_CALUDE_charges_needed_equals_total_rooms_l761_76128

def battery_duration : ℕ := 10
def vacuum_time_per_room : ℕ := 8
def num_bedrooms : ℕ := 3
def num_kitchen : ℕ := 1
def num_living_room : ℕ := 1
def num_dining_room : ℕ := 1
def num_office : ℕ := 1
def num_bathrooms : ℕ := 2

def total_rooms : ℕ := num_bedrooms + num_kitchen + num_living_room + num_dining_room + num_office + num_bathrooms

theorem charges_needed_equals_total_rooms :
  battery_duration > vacuum_time_per_room ∧
  battery_duration < 2 * vacuum_time_per_room →
  total_rooms = total_rooms :=
by sorry

end NUMINAMATH_CALUDE_charges_needed_equals_total_rooms_l761_76128
