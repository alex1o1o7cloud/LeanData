import Mathlib

namespace bus_capacity_problem_l2195_219574

/-- Proves that given two buses with a capacity of 150 people each, where one bus is 70% full
    and the total number of people in both buses is 195, the percentage of capacity full
    for the other bus is 60%. -/
theorem bus_capacity_problem (bus_capacity : ℕ) (total_people : ℕ) (second_bus_percentage : ℚ) :
  bus_capacity = 150 →
  total_people = 195 →
  second_bus_percentage = 70/100 →
  ∃ (first_bus_percentage : ℚ),
    first_bus_percentage * bus_capacity + second_bus_percentage * bus_capacity = total_people ∧
    first_bus_percentage = 60/100 :=
by sorry

end bus_capacity_problem_l2195_219574


namespace unique_card_arrangement_l2195_219581

def CardPair := (Nat × Nat)

def is_valid_pair (p : CardPair) : Prop :=
  (p.1 ∣ p.2) ∨ (p.2 ∣ p.1)

def is_unique_arrangement (arr : List CardPair) : Prop :=
  arr.length = 5 ∧
  (∀ p ∈ arr, 1 ≤ p.1 ∧ p.1 ≤ 10 ∧ 1 ≤ p.2 ∧ p.2 ≤ 10) ∧
  (∀ p ∈ arr, is_valid_pair p) ∧
  (∀ n : Nat, 1 ≤ n ∧ n ≤ 10 → (arr.map Prod.fst ++ arr.map Prod.snd).count n = 1)

theorem unique_card_arrangement :
  ∃! arr : List CardPair, is_unique_arrangement arr :=
sorry

end unique_card_arrangement_l2195_219581


namespace towel_area_decrease_l2195_219502

/-- Represents the properties of a fabric material -/
structure Material where
  cotton_percent : Real
  polyester_percent : Real
  cotton_length_shrinkage : Real
  cotton_breadth_shrinkage : Real
  polyester_length_shrinkage : Real
  polyester_breadth_shrinkage : Real

/-- Calculates the area decrease percentage of a fabric after shrinkage -/
def calculate_area_decrease (m : Material) : Real :=
  let effective_length_shrinkage := 
    m.cotton_length_shrinkage * m.cotton_percent + m.polyester_length_shrinkage * m.polyester_percent
  let effective_breadth_shrinkage := 
    m.cotton_breadth_shrinkage * m.cotton_percent + m.polyester_breadth_shrinkage * m.polyester_percent
  1 - (1 - effective_length_shrinkage) * (1 - effective_breadth_shrinkage)

/-- The towel material properties -/
def towel : Material := {
  cotton_percent := 0.60
  polyester_percent := 0.40
  cotton_length_shrinkage := 0.35
  cotton_breadth_shrinkage := 0.45
  polyester_length_shrinkage := 0.25
  polyester_breadth_shrinkage := 0.30
}

/-- Theorem: The area decrease of the towel after bleaching is approximately 57.91% -/
theorem towel_area_decrease : 
  ∃ ε > 0, |calculate_area_decrease towel - 0.5791| < ε :=
by sorry

end towel_area_decrease_l2195_219502


namespace integral_x_squared_minus_x_l2195_219519

theorem integral_x_squared_minus_x : ∫ x in (0)..(2), (x^2 - x) = 2/3 := by
  sorry

end integral_x_squared_minus_x_l2195_219519


namespace min_value_x_plus_y_l2195_219554

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1 / x + 9 / y = 2) : 
  ∀ a b : ℝ, a > 0 → b > 0 → 1 / a + 9 / b = 2 → x + y ≤ a + b ∧ 
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 1 / x + 9 / y = 2 ∧ x + y = 8 :=
sorry

end min_value_x_plus_y_l2195_219554


namespace phoebe_age_proof_l2195_219517

/-- Phoebe's current age -/
def phoebe_age : ℕ := 10

/-- Raven's current age -/
def raven_age : ℕ := 55

theorem phoebe_age_proof :
  (raven_age + 5 = 4 * (phoebe_age + 5)) → phoebe_age = 10 := by
  sorry

end phoebe_age_proof_l2195_219517


namespace trajectory_equation_l2195_219592

-- Define the property for a point (x, y)
def satisfiesProperty (x y : ℝ) : Prop :=
  2 * (|x| + |y|) = x^2 + y^2

-- Theorem statement
theorem trajectory_equation :
  ∀ x y : ℝ, satisfiesProperty x y ↔ x^2 + y^2 = 2 * |x| + 2 * |y| :=
by sorry

end trajectory_equation_l2195_219592


namespace sum_of_series_l2195_219573

theorem sum_of_series (a₁ : ℝ) (r : ℝ) (n : ℕ) (d : ℝ) :
  let geometric_sum := a₁ / (1 - r)
  let arithmetic_sum := n * (2 * a₁ + (n - 1) * d) / 2
  geometric_sum + arithmetic_sum = 115 / 3 :=
by
  sorry

end sum_of_series_l2195_219573


namespace right_triangle_with_tangent_circle_l2195_219553

theorem right_triangle_with_tangent_circle (a b c r : ℕ) : 
  a^2 + b^2 = c^2 → -- right triangle
  Nat.gcd a (Nat.gcd b c) = 1 → -- side lengths have no common divisor greater than 1
  r = (a + b - c) / 2 → -- radius of circle tangent to hypotenuse
  r = 420 → -- given radius
  (a = 399 ∧ b = 40 ∧ c = 401) ∨ (a = 40 ∧ b = 399 ∧ c = 401) := by
sorry

end right_triangle_with_tangent_circle_l2195_219553


namespace skew_lines_planes_perpendicularity_l2195_219579

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (are_skew : Line → Line → Prop)
variable (parallel_plane_line : Plane → Line → Prop)
variable (perpendicular_line_line : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_plane_plane : Plane → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)

-- State the theorem
theorem skew_lines_planes_perpendicularity 
  (m n l : Line) (α β : Plane) :
  are_skew m n →
  parallel_plane_line α m →
  parallel_plane_line α n →
  perpendicular_line_line l m →
  perpendicular_line_line l n →
  parallel_line_plane l β →
  perpendicular_plane_plane α β ∧ perpendicular_line_plane l α :=
sorry

end skew_lines_planes_perpendicularity_l2195_219579


namespace sqrt_2_minus_x_real_range_l2195_219531

theorem sqrt_2_minus_x_real_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = 2 - x) ↔ x ≤ 2 := by sorry

end sqrt_2_minus_x_real_range_l2195_219531


namespace kaleb_remaining_chocolates_l2195_219565

def boxes_bought : ℕ := 14
def pieces_per_box : ℕ := 6
def boxes_given_away : ℕ := 5 + 2 + 3

def remaining_boxes : ℕ := boxes_bought - boxes_given_away
def remaining_pieces : ℕ := remaining_boxes * pieces_per_box

def eaten_pieces : ℕ := (remaining_pieces * 10) / 100

theorem kaleb_remaining_chocolates :
  remaining_pieces - eaten_pieces = 22 := by sorry

end kaleb_remaining_chocolates_l2195_219565


namespace intersection_of_A_and_B_l2195_219515

def A : Set ℝ := {0, 1, 2}
def B : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}

theorem intersection_of_A_and_B : A ∩ B = {0, 1} := by
  sorry

end intersection_of_A_and_B_l2195_219515


namespace perfect_square_a_value_of_a_l2195_219591

theorem perfect_square_a : ∃ n : ℕ, 1995^2 + 1995^2 * 1996^2 + 1996^2 = n^2 :=
by
  use 3982021
  sorry

theorem value_of_a : 1995^2 + 1995^2 * 1996^2 + 1996^2 = 3982021^2 :=
by sorry

end perfect_square_a_value_of_a_l2195_219591


namespace probability_total_more_than_seven_l2195_219539

/-- The number of faces on each die -/
def numFaces : Nat := 6

/-- The total number of possible outcomes when throwing two dice -/
def totalOutcomes : Nat := numFaces * numFaces

/-- The number of favorable outcomes (total > 7) -/
def favorableOutcomes : Nat := 14

/-- The probability of getting a total more than 7 -/
def probabilityTotalMoreThan7 : Rat := favorableOutcomes / totalOutcomes

theorem probability_total_more_than_seven :
  probabilityTotalMoreThan7 = 7 / 18 := by
  sorry

end probability_total_more_than_seven_l2195_219539


namespace max_integer_difference_l2195_219533

theorem max_integer_difference (x y : ℝ) (hx : 6 < x ∧ x < 10) (hy : 10 < y ∧ y < 17) :
  (⌊y⌋ : ℤ) - (⌈x⌉ : ℤ) ≤ 9 ∧ ∃ (x₀ y₀ : ℝ), 6 < x₀ ∧ x₀ < 10 ∧ 10 < y₀ ∧ y₀ < 17 ∧ (⌊y₀⌋ : ℤ) - (⌈x₀⌉ : ℤ) = 9 :=
by sorry

end max_integer_difference_l2195_219533


namespace binomial_coefficient_19_12_l2195_219536

theorem binomial_coefficient_19_12 
  (h1 : Nat.choose 20 13 = 77520)
  (h2 : Nat.choose 18 11 = 31824) : 
  Nat.choose 19 12 = 77520 - 31824 := by
  sorry

end binomial_coefficient_19_12_l2195_219536


namespace complement_of_A_l2195_219538

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x ≤ -3} ∪ {x | x ≥ 0}

-- Theorem statement
theorem complement_of_A : Set.compl A = Set.Ioo (-3) 0 := by sorry

end complement_of_A_l2195_219538


namespace min_value_theorem_l2195_219550

/-- The line equation ax + by - 2 = 0 --/
def line_equation (a b x y : ℝ) : Prop := a * x + b * y - 2 = 0

/-- The circle equation x^2 + y^2 - 2x - 2y = 2 --/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y = 2

/-- The line bisects the circumference of the circle --/
def line_bisects_circle (a b : ℝ) : Prop :=
  ∀ x y, line_equation a b x y → circle_equation x y →
    ∃ c d, c^2 + d^2 = 1 ∧ line_equation a b (1 + c) (1 + d)

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_bisect : line_bisects_circle a b) :
  (1 / (2 * a) + 1 / b) ≥ (3 + 2 * Real.sqrt 2) / 4 :=
sorry

end min_value_theorem_l2195_219550


namespace survey_result_l2195_219526

theorem survey_result (total : ℕ) (tv_dislike_percent : ℚ) (both_dislike_percent : ℚ) 
  (h1 : total = 1500)
  (h2 : tv_dislike_percent = 25 / 100)
  (h3 : both_dislike_percent = 20 / 100) :
  ↑⌊both_dislike_percent * (tv_dislike_percent * total)⌋ = 75 := by
  sorry

end survey_result_l2195_219526


namespace lisa_cleaning_time_proof_l2195_219588

/-- The time it takes Lisa to clean her room alone -/
def lisa_cleaning_time : ℝ := 8

/-- The time it takes Kay to clean her room alone -/
def kay_cleaning_time : ℝ := 12

/-- The time it takes Lisa and Kay to clean a room together -/
def combined_cleaning_time : ℝ := 4.8

theorem lisa_cleaning_time_proof :
  lisa_cleaning_time = 8 ∧
  (1 / lisa_cleaning_time + 1 / kay_cleaning_time = 1 / combined_cleaning_time) :=
sorry

end lisa_cleaning_time_proof_l2195_219588


namespace square_area_error_l2195_219511

theorem square_area_error (actual_side : ℝ) (h : actual_side > 0) :
  let measured_side := actual_side * 1.1
  let actual_area := actual_side ^ 2
  let calculated_area := measured_side ^ 2
  let area_error := (calculated_area - actual_area) / actual_area
  area_error = 0.21 := by
sorry

end square_area_error_l2195_219511


namespace polar_to_rect_transformation_l2195_219529

/-- Given a point (12, 5) in rectangular coordinates and (r, θ) in polar coordinates,
    prove that the point (r³, 3θ) in polar coordinates is (5600, -325) in rectangular coordinates. -/
theorem polar_to_rect_transformation (r θ : ℝ) :
  r * Real.cos θ = 12 →
  r * Real.sin θ = 5 →
  (r^3 * Real.cos (3*θ), r^3 * Real.sin (3*θ)) = (5600, -325) := by
  sorry

end polar_to_rect_transformation_l2195_219529


namespace probability_one_defective_part_l2195_219557

/-- The probability of drawing exactly one defective part from a box containing 5 parts,
    of which 2 are defective, when randomly selecting 2 parts. -/
theorem probability_one_defective_part : 
  let total_parts : ℕ := 5
  let defective_parts : ℕ := 2
  let drawn_parts : ℕ := 2
  let total_ways := Nat.choose total_parts drawn_parts
  let favorable_ways := Nat.choose defective_parts 1 * Nat.choose (total_parts - defective_parts) (drawn_parts - 1)
  (favorable_ways : ℚ) / total_ways = 3 / 5 := by
  sorry

end probability_one_defective_part_l2195_219557


namespace problem_solution_l2195_219544

theorem problem_solution : (-1)^2023 + |2 * Real.sqrt 2 - 3| + (8 : ℝ)^(1/3) = 4 - 2 * Real.sqrt 2 := by
  sorry

end problem_solution_l2195_219544


namespace direct_proportion_through_point_decreasing_l2195_219570

-- Define the direct proportion function
def direct_proportion (m : ℝ) (x : ℝ) : ℝ := m * x

-- Define the theorem
theorem direct_proportion_through_point_decreasing (m : ℝ) :
  (direct_proportion m m = 4) →
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → direct_proportion m x₁ > direct_proportion m x₂) →
  m = -2 := by
  sorry

end direct_proportion_through_point_decreasing_l2195_219570


namespace congruence_problem_l2195_219587

theorem congruence_problem (a b : ℤ) (h1 : a ≡ 27 [ZMOD 60]) (h2 : b ≡ 94 [ZMOD 60]) :
  ∃ n : ℤ, 150 ≤ n ∧ n ≤ 211 ∧ (a - b) ≡ n [ZMOD 60] ∧ n = 173 := by
  sorry

end congruence_problem_l2195_219587


namespace birthday_friends_count_l2195_219505

theorem birthday_friends_count : ∃ (n : ℕ), 
  (12 * (n + 2) = 16 * n) ∧ 
  (∀ m : ℕ, 12 * (m + 2) = 16 * m → m = n) :=
by sorry

end birthday_friends_count_l2195_219505


namespace quadratic_inequality_implies_a_greater_than_two_l2195_219508

theorem quadratic_inequality_implies_a_greater_than_two (a : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ x^2 - a*x + 1 < 0) → a > 2 := by
  sorry

end quadratic_inequality_implies_a_greater_than_two_l2195_219508


namespace boundary_length_of_modified_square_l2195_219559

-- Define the square's area
def square_area : ℝ := 256

-- Define the number of divisions per side
def divisions : ℕ := 4

-- Theorem statement
theorem boundary_length_of_modified_square :
  let side_length := Real.sqrt square_area
  let segment_length := side_length / divisions
  let arc_length := 2 * Real.pi * segment_length
  let straight_segments_length := 2 * divisions * segment_length
  abs ((arc_length + straight_segments_length) - 57.1) < 0.05 := by
sorry

end boundary_length_of_modified_square_l2195_219559


namespace remainder_theorem_l2195_219522

theorem remainder_theorem (r : ℤ) : (r^13 - r^5 + 1) % (r - 1) = 1 := by
  sorry

end remainder_theorem_l2195_219522


namespace decreasing_interval_l2195_219599

def f (x : ℝ) := x^2 - 6*x + 8

theorem decreasing_interval (a : ℝ) :
  (∀ x ∈ Set.Icc 1 a, ∀ y ∈ Set.Icc 1 a, x < y → f x > f y) ↔ 1 < a ∧ a ≤ 3 :=
by sorry

end decreasing_interval_l2195_219599


namespace base5_division_l2195_219597

-- Define a function to convert from base 5 to base 10
def base5ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

-- Define the dividend and divisor in base 5
def dividend : List Nat := [4, 0, 2, 3, 1]  -- 13204₅
def divisor : List Nat := [3, 2]  -- 23₅

-- Define the expected quotient and remainder in base 5
def expectedQuotient : List Nat := [1, 1, 3]  -- 311₅
def expectedRemainder : Nat := 1  -- 1₅

-- Theorem statement
theorem base5_division :
  let dividend10 := base5ToBase10 dividend
  let divisor10 := base5ToBase10 divisor
  let quotient10 := dividend10 / divisor10
  let remainder10 := dividend10 % divisor10
  base5ToBase10 expectedQuotient = quotient10 ∧
  expectedRemainder = remainder10 := by
  sorry


end base5_division_l2195_219597


namespace solve_bowling_problem_l2195_219524

def bowling_problem (gretchen_score mitzi_score average_score : ℕ) : Prop :=
  let total_score := average_score * 3
  let beth_score := total_score - gretchen_score - mitzi_score
  gretchen_score = 120 ∧ 
  mitzi_score = 113 ∧ 
  average_score = 106 →
  beth_score = 85

theorem solve_bowling_problem :
  ∃ (gretchen_score mitzi_score average_score : ℕ),
    bowling_problem gretchen_score mitzi_score average_score :=
by
  sorry

end solve_bowling_problem_l2195_219524


namespace square_difference_evaluation_l2195_219577

theorem square_difference_evaluation : 81^2 - (45 + 9)^2 = 3645 := by
  sorry

end square_difference_evaluation_l2195_219577


namespace ball_game_probabilities_l2195_219548

theorem ball_game_probabilities (total : ℕ) (p_white p_red p_yellow : ℚ) 
  (h_total : total = 6)
  (h_white : p_white = 1/2)
  (h_red : p_red = 1/3)
  (h_yellow : p_yellow = 1/6)
  (h_sum : p_white + p_red + p_yellow = 1) :
  ∃ (white red yellow : ℕ),
    white + red + yellow = total ∧
    (white : ℚ) / total = p_white ∧
    (red : ℚ) / total = p_red ∧
    (yellow : ℚ) / total = p_yellow ∧
    white = 3 ∧ red = 2 ∧ yellow = 1 := by
  sorry

end ball_game_probabilities_l2195_219548


namespace sphere_volume_l2195_219510

theorem sphere_volume (prism_length prism_width prism_height : ℝ) 
  (sphere_volume : ℝ → ℝ) (L : ℝ) :
  prism_length = 4 →
  prism_width = 2 →
  prism_height = 1 →
  (∀ r : ℝ, sphere_volume r = (4 / 3) * π * r^3) →
  (∃ r : ℝ, 4 * π * r^2 = 2 * (prism_length * prism_width + 
    prism_length * prism_height + prism_width * prism_height)) →
  (∃ r : ℝ, sphere_volume r = L * Real.sqrt 2 / Real.sqrt π) →
  L = 14 * Real.sqrt 14 / 3 :=
by sorry

end sphere_volume_l2195_219510


namespace expression_simplification_l2195_219546

theorem expression_simplification (a : ℝ) (h : a = Real.sqrt 3 + 1) :
  ((a^2 / (a - 2) - 1 / (a - 2)) / ((a^2 - 2*a + 1) / (a - 2))) = (3 + 2 * Real.sqrt 3) / 3 := by
  sorry

end expression_simplification_l2195_219546


namespace mrs_heine_treats_l2195_219584

/-- The number of treats Mrs. Heine needs to buy for her pets -/
def total_treats (num_dogs : ℕ) (num_cats : ℕ) (num_parrots : ℕ) 
                 (biscuits_per_dog : ℕ) (treats_per_cat : ℕ) (sticks_per_parrot : ℕ) : ℕ :=
  num_dogs * biscuits_per_dog + num_cats * treats_per_cat + num_parrots * sticks_per_parrot

/-- Theorem stating that Mrs. Heine needs to buy 11 treats in total -/
theorem mrs_heine_treats : total_treats 2 1 3 3 2 1 = 11 := by
  sorry

end mrs_heine_treats_l2195_219584


namespace mitzel_spending_l2195_219543

/-- Proves that Mitzel spent $14, given the conditions of the problem -/
theorem mitzel_spending (allowance : ℝ) (spent_percentage : ℝ) (remaining : ℝ) : 
  spent_percentage = 0.35 →
  remaining = 26 →
  (1 - spent_percentage) * allowance = remaining →
  spent_percentage * allowance = 14 := by
  sorry

end mitzel_spending_l2195_219543


namespace ladder_problem_l2195_219556

theorem ladder_problem (ladder_length height : ℝ) 
  (h1 : ladder_length = 13)
  (h2 : height = 12) :
  ∃ base_distance : ℝ, base_distance^2 + height^2 = ladder_length^2 ∧ base_distance = 5 := by
  sorry

end ladder_problem_l2195_219556


namespace exists_line_with_three_colors_l2195_219530

/-- A color type with four possible values -/
inductive Color
  | One
  | Two
  | Three
  | Four

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A coloring function that assigns a color to each point in the plane -/
def Coloring := Point → Color

/-- A predicate that checks if a coloring uses all four colors -/
def uses_all_colors (f : Coloring) : Prop :=
  (∃ p : Point, f p = Color.One) ∧
  (∃ p : Point, f p = Color.Two) ∧
  (∃ p : Point, f p = Color.Three) ∧
  (∃ p : Point, f p = Color.Four)

/-- A predicate that checks if a point is on a line -/
def on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- The main theorem -/
theorem exists_line_with_three_colors (f : Coloring) (h : uses_all_colors f) :
  ∃ l : Line, ∃ p₁ p₂ p₃ : Point,
    on_line p₁ l ∧ on_line p₂ l ∧ on_line p₃ l ∧
    f p₁ ≠ f p₂ ∧ f p₁ ≠ f p₃ ∧ f p₂ ≠ f p₃ :=
sorry

end exists_line_with_three_colors_l2195_219530


namespace baker_duration_l2195_219598

/-- Represents the number of weeks Steve bakes pies -/
def duration : ℕ := sorry

/-- Number of days per week Steve bakes apple pies -/
def apple_days : ℕ := 3

/-- Number of days per week Steve bakes cherry pies -/
def cherry_days : ℕ := 2

/-- Number of pies Steve bakes per day -/
def pies_per_day : ℕ := 12

/-- The difference in the number of apple pies and cherry pies -/
def pie_difference : ℕ := 12

theorem baker_duration :
  apple_days * pies_per_day * duration = cherry_days * pies_per_day * duration + pie_difference ∧
  duration = 1 := by sorry

end baker_duration_l2195_219598


namespace inverse_g_87_l2195_219590

def g (x : ℝ) : ℝ := 3 * x^3 + 6

theorem inverse_g_87 : g⁻¹ 87 = 3 := by sorry

end inverse_g_87_l2195_219590


namespace function_values_l2195_219537

noncomputable section

def f (x : ℝ) : ℝ := -1/x

theorem function_values (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (f a = -1/3 → a = 3) ∧
  (f (a * b) = 1/6 → b = -2) ∧
  (f c = Real.sin c / Real.cos c → Real.tan c = -1/c) :=
by sorry

end function_values_l2195_219537


namespace negation_to_original_proposition_l2195_219547

theorem negation_to_original_proposition :
  (¬ (∃ x : ℝ, x < 1 ∧ x^2 < 1)) ↔ (∀ x : ℝ, x < 1 → x^2 ≥ 1) :=
by sorry

end negation_to_original_proposition_l2195_219547


namespace right_triangle_sets_l2195_219561

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

theorem right_triangle_sets :
  is_pythagorean_triple 7 24 25 ∧
  is_pythagorean_triple 6 8 10 ∧
  is_pythagorean_triple 9 12 15 ∧
  ¬ is_pythagorean_triple 3 4 6 :=
sorry

end right_triangle_sets_l2195_219561


namespace rates_sum_of_squares_l2195_219566

/-- Given Ed and Sue's rollerblading, biking, and swimming rates and their total distances,
    prove that the sum of squares of the rates is 485. -/
theorem rates_sum_of_squares (r b s : ℕ) : 
  (2 * r + 3 * b + s = 80) →
  (4 * r + 2 * b + 3 * s = 98) →
  r^2 + b^2 + s^2 = 485 := by
  sorry

end rates_sum_of_squares_l2195_219566


namespace min_max_abs_x_squared_minus_2xy_is_zero_l2195_219594

open Real

theorem min_max_abs_x_squared_minus_2xy_is_zero :
  ∃ y : ℝ, ∀ z : ℝ, (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |x^2 - 2*x*y| ≤ z) →
    (∀ y' : ℝ, ∃ x : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ |x^2 - 2*x*y'| ≥ z) :=
by sorry

end min_max_abs_x_squared_minus_2xy_is_zero_l2195_219594


namespace fixed_point_parabola_l2195_219540

/-- The fixed point theorem for a parabola -/
theorem fixed_point_parabola 
  (p a b : ℝ) 
  (h1 : a ≠ 0) 
  (h2 : b ≠ 0) 
  (h3 : b^2 ≠ 2*p*a) :
  ∃ C : ℝ × ℝ, 
    ∀ (M M₁ M₂ : ℝ × ℝ),
      (M.2)^2 = 2*p*M.1 →  -- M is on the parabola
      (M₁.2)^2 = 2*p*M₁.1 →  -- M₁ is on the parabola
      (M₂.2)^2 = 2*p*M₂.1 →  -- M₂ is on the parabola
      M₁ ≠ M →
      M₂ ≠ M →
      M₁ ≠ M₂ →
      (∃ t : ℝ, M₁.2 - b = t * (M₁.1 - a)) →  -- M₁ is on line AM
      (∃ t : ℝ, M₂.2 = t * (M₂.1 + a)) →  -- M₂ is on line BM
      (∃ t : ℝ, M₂.2 - M₁.2 = t * (M₂.1 - M₁.1)) →  -- M₁M₂ is a line
      C = (a, 2*p*a/b) ∧ 
      ∃ t : ℝ, C.2 - M₁.2 = t * (C.1 - M₁.1)  -- C is on line M₁M₂
  := by sorry

end fixed_point_parabola_l2195_219540


namespace circle_origin_range_l2195_219568

theorem circle_origin_range (m : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 - 2*m*x + 2*m*y + 2*m^2 - 4 = 0 → x^2 + y^2 < 4) → 
  -Real.sqrt 2 < m ∧ m < Real.sqrt 2 :=
sorry

end circle_origin_range_l2195_219568


namespace no_good_integers_l2195_219500

theorem no_good_integers : 
  ¬∃ (n : ℕ), n ≥ 1 ∧ 
  (∀ (k : ℕ), k > 0 → 
    ((∀ i ∈ Finset.range 9, k % (n + i + 1) = 0) → k % (n + 10) = 0)) :=
by sorry

end no_good_integers_l2195_219500


namespace proportion_solution_l2195_219513

theorem proportion_solution (x : ℚ) : (3 : ℚ) / 12 = x / 16 → x = 4 := by
  sorry

end proportion_solution_l2195_219513


namespace unique_solution_l2195_219580

-- Define the digits as natural numbers
def A : ℕ := sorry
def B : ℕ := sorry
def d : ℕ := sorry
def I : ℕ := sorry

-- Define the conditions
axiom digit_constraint : A < 10 ∧ B < 10 ∧ d < 10 ∧ I < 10
axiom equation : 58 * (100 * A + 10 * B + A) = 1000 * I + 100 * d + 10 * B + A

-- State the theorem
theorem unique_solution : d = 4 := by sorry

end unique_solution_l2195_219580


namespace quadratic_function_range_l2195_219575

/-- A quadratic function passing through (1,0) and (0,1) with vertex in second quadrant -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0
  point_zero_one : 1 = c
  point_one_zero : 0 = a + b + c
  vertex_second_quadrant : b < 0 ∧ a < 0

/-- The range of a - b + c for the given quadratic function -/
theorem quadratic_function_range (f : QuadraticFunction) : 
  0 < f.a - f.b + f.c ∧ f.a - f.b + f.c < 2 := by
  sorry

end quadratic_function_range_l2195_219575


namespace plan_comparison_l2195_219506

def suit_price : ℝ := 500
def tie_price : ℝ := 80
def num_suits : ℕ := 20

def plan1_cost (x : ℝ) : ℝ := 8400 + 80 * x
def plan2_cost (x : ℝ) : ℝ := 9000 + 72 * x

theorem plan_comparison (x : ℝ) (h : x > 20) :
  plan1_cost x ≤ plan2_cost x ↔ x ≤ 75 := by sorry

end plan_comparison_l2195_219506


namespace student_age_problem_l2195_219576

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The problem statement -/
theorem student_age_problem :
  ∃! n : ℕ, 1900 ≤ n ∧ n < 1960 ∧ (1960 - n = sum_of_digits n) := by sorry

end student_age_problem_l2195_219576


namespace divisibility_property_l2195_219514

theorem divisibility_property (A B n : ℕ) (hn : n = 7 ∨ n = 11 ∨ n = 13) 
  (h : n ∣ (B - A)) : n ∣ (1000 * A + B) := by
  sorry

end divisibility_property_l2195_219514


namespace prob_at_least_one_black_is_four_fifths_l2195_219583

/-- The number of balls in the bag -/
def total_balls : ℕ := 6

/-- The number of black balls in the bag -/
def black_balls : ℕ := 3

/-- The number of balls drawn -/
def drawn_balls : ℕ := 2

/-- The probability of drawing at least one black ball when two balls are randomly drawn -/
def prob_at_least_one_black : ℚ := 4/5

theorem prob_at_least_one_black_is_four_fifths :
  prob_at_least_one_black = 4/5 := by
  sorry

end prob_at_least_one_black_is_four_fifths_l2195_219583


namespace P_and_S_not_fourth_l2195_219521

-- Define the set of runners
inductive Runner : Type
| P | Q | R | S | T | U

-- Define the relation "finishes before"
def finishes_before (a b : Runner) : Prop := sorry

-- Define the conditions
axiom P_beats_R : finishes_before Runner.P Runner.R
axiom P_beats_S : finishes_before Runner.P Runner.S
axiom Q_beats_S : finishes_before Runner.Q Runner.S
axiom Q_before_U : finishes_before Runner.Q Runner.U
axiom U_before_P : finishes_before Runner.U Runner.P
axiom T_before_U : finishes_before Runner.T Runner.U
axiom T_before_Q : finishes_before Runner.T Runner.Q

-- Define what it means to finish fourth
def finishes_fourth (r : Runner) : Prop := 
  ∃ a b c : Runner, 
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    a ≠ r ∧ b ≠ r ∧ c ≠ r ∧
    finishes_before a r ∧ 
    finishes_before b r ∧ 
    finishes_before c r ∧
    (∀ x : Runner, x ≠ r → x ≠ a → x ≠ b → x ≠ c → finishes_before r x)

-- Theorem to prove
theorem P_and_S_not_fourth : 
  ¬(finishes_fourth Runner.P) ∧ ¬(finishes_fourth Runner.S) :=
sorry

end P_and_S_not_fourth_l2195_219521


namespace frustum_volume_ratio_l2195_219509

/-- Given a frustum with base area ratio 1:9, prove the volume ratio of parts divided by midsection is 7:19 -/
theorem frustum_volume_ratio (A₁ A₂ V₁ V₂ : ℝ) (h_area_ratio : A₁ / A₂ = 1 / 9) :
  V₁ / V₂ = 7 / 19 := by
  sorry

end frustum_volume_ratio_l2195_219509


namespace rice_distribution_l2195_219532

theorem rice_distribution (total_weight : ℚ) (num_containers : ℕ) (ounces_per_pound : ℕ) :
  total_weight = 33 / 4 →
  num_containers = 4 →
  ounces_per_pound = 16 →
  (total_weight * ounces_per_pound) / num_containers = 15 := by
  sorry

end rice_distribution_l2195_219532


namespace rainbow_population_proof_l2195_219551

/-- The number of settlements in Solar Valley -/
def num_settlements : ℕ := 10

/-- The population of Zhovtnevo -/
def zhovtnevo_population : ℕ := 1000

/-- The amount by which Zhovtnevo's population exceeds the average -/
def excess_population : ℕ := 90

/-- The population of Rainbow settlement -/
def rainbow_population : ℕ := 900

theorem rainbow_population_proof :
  rainbow_population = 
    (num_settlements * zhovtnevo_population - num_settlements * excess_population) / (num_settlements - 1) :=
by sorry

end rainbow_population_proof_l2195_219551


namespace min_buses_for_field_trip_l2195_219552

def min_buses (total_students : ℕ) (bus_cap_1 bus_cap_2 : ℕ) (min_bus_2 : ℕ) : ℕ :=
  let x := ((total_students - bus_cap_2 * min_bus_2 + bus_cap_1 - 1) / bus_cap_1 : ℕ)
  x + min_bus_2

theorem min_buses_for_field_trip :
  min_buses 530 45 35 3 = 13 :=
sorry

end min_buses_for_field_trip_l2195_219552


namespace cats_needed_to_reach_goal_l2195_219578

theorem cats_needed_to_reach_goal (current_cats goal_cats : ℕ) : 
  current_cats = 11 → goal_cats = 43 → goal_cats - current_cats = 32 := by
sorry

end cats_needed_to_reach_goal_l2195_219578


namespace expression_change_l2195_219501

/-- The change in the expression x^3 - 3x + 1 when x changes by a -/
def expressionChange (x a : ℝ) : ℝ := 
  (x + a)^3 - 3*(x + a) + 1 - (x^3 - 3*x + 1)

theorem expression_change (x a : ℝ) (h : a > 0) : 
  expressionChange x a = 3*a*x^2 + 3*a^2*x + a^3 - 3*a ∧
  expressionChange x (-a) = -3*a*x^2 + 3*a^2*x - a^3 + 3*a := by
  sorry

end expression_change_l2195_219501


namespace trigonometric_identities_l2195_219558

theorem trigonometric_identities : 
  (2 * Real.sin (75 * π / 180) * Real.cos (75 * π / 180) = 1/2) ∧ 
  (Real.sin (45 * π / 180) * Real.cos (15 * π / 180) - 
   Real.cos (45 * π / 180) * Real.sin (15 * π / 180) = 1/2) := by
  sorry

end trigonometric_identities_l2195_219558


namespace tie_in_may_l2195_219586

structure Player where
  january : ℕ
  february : ℕ
  march : ℕ
  april : ℕ
  may : ℕ

def johnson : Player := ⟨2, 12, 20, 15, 9⟩
def martinez : Player := ⟨5, 9, 15, 20, 9⟩

def cumulative_score (p : Player) (month : ℕ) : ℕ :=
  match month with
  | 1 => p.january
  | 2 => p.january + p.february
  | 3 => p.january + p.february + p.march
  | 4 => p.january + p.february + p.march + p.april
  | 5 => p.january + p.february + p.march + p.april + p.may
  | _ => 0

def first_tie_month : ℕ :=
  [1, 2, 3, 4, 5].find? (λ m => cumulative_score johnson m = cumulative_score martinez m)
    |>.getD 0

theorem tie_in_may :
  first_tie_month = 5 := by sorry

end tie_in_may_l2195_219586


namespace odd_function_implies_m_eq_neg_one_l2195_219589

/-- A function f is odd if f(-x) = -f(x) for all x in its domain -/
def IsOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The logarithm function with base a -/
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

/-- The given function f -/
noncomputable def f (a m : ℝ) (x : ℝ) : ℝ := log a ((1 - m*x) / (x - 1))

theorem odd_function_implies_m_eq_neg_one (a m : ℝ) 
    (h1 : a > 0) (h2 : a ≠ 1) (h3 : IsOddFunction (f a m)) : m = -1 := by
  sorry

end odd_function_implies_m_eq_neg_one_l2195_219589


namespace max_axes_of_symmetry_is_six_l2195_219512

/-- A line segment in a plane -/
structure LineSegment where
  -- Define properties of a line segment here
  -- For simplicity, we'll just use a placeholder
  id : Nat

/-- A configuration of three line segments in a plane -/
structure ThreeSegmentConfiguration where
  segments : Fin 3 → LineSegment

/-- An axis of symmetry for a configuration of line segments -/
structure AxisOfSymmetry where
  -- Define properties of an axis of symmetry here
  -- For simplicity, we'll just use a placeholder
  id : Nat

/-- The set of axes of symmetry for a given configuration -/
def axesOfSymmetry (config : ThreeSegmentConfiguration) : Set AxisOfSymmetry :=
  sorry

/-- The maximum number of axes of symmetry for any configuration of three line segments -/
def maxAxesOfSymmetry : Nat :=
  sorry

theorem max_axes_of_symmetry_is_six :
  maxAxesOfSymmetry = 6 :=
sorry

end max_axes_of_symmetry_is_six_l2195_219512


namespace linear_equation_solution_l2195_219507

theorem linear_equation_solution :
  let x : ℝ := -4
  let y : ℝ := 2
  x + 3 * y = 2 := by sorry

end linear_equation_solution_l2195_219507


namespace sequence_product_l2195_219520

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = r * b n

/-- The main theorem -/
theorem sequence_product (a b : ℕ → ℝ) :
  arithmetic_sequence a →
  geometric_sequence b →
  (∀ n : ℕ, a n ≠ 0) →
  (2 * a 3 - a 7 ^ 2 + 2 * a n = 0) →
  b 7 = a 7 →
  b 6 * b 8 = 16 := by
sorry

end sequence_product_l2195_219520


namespace min_value_expression_min_value_achieved_l2195_219541

theorem min_value_expression (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) :
  8 * p^4 + 18 * q^4 + 50 * r^4 + 1 / (8 * p * q * r) ≥ 6 :=
by sorry

theorem min_value_achieved (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) :
  ∃ (p₀ q₀ r₀ : ℝ), p₀ > 0 ∧ q₀ > 0 ∧ r₀ > 0 ∧
    8 * p₀^4 + 18 * q₀^4 + 50 * r₀^4 + 1 / (8 * p₀ * q₀ * r₀) = 6 :=
by sorry

end min_value_expression_min_value_achieved_l2195_219541


namespace solution_set_when_a_is_negative_four_range_of_a_for_inequality_l2195_219534

-- Define the function f
def f (x a : ℝ) : ℝ := |2*x + a| + |x - 2|

-- Theorem for part (1)
theorem solution_set_when_a_is_negative_four :
  {x : ℝ | f x (-4) ≥ 6} = {x : ℝ | x ≤ 0 ∨ x ≥ 4} := by sorry

-- Theorem for part (2)
theorem range_of_a_for_inequality :
  {a : ℝ | ∀ x, f x a ≥ 3*a^2 - |2 - x|} = {a : ℝ | -1 ≤ a ∧ a ≤ 4/3} := by sorry

end solution_set_when_a_is_negative_four_range_of_a_for_inequality_l2195_219534


namespace cosine_like_properties_l2195_219528

-- Define the cosine-like function
def cosine_like (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) + f (x - y) = 2 * f x * f y

-- State the theorem
theorem cosine_like_properties (f : ℝ → ℝ) 
  (h1 : cosine_like f) 
  (h2 : f 1 = 5/4)
  (h3 : ∀ t : ℝ, t ≠ 0 → f t > 1) :
  (f 0 = 1 ∧ f 2 = 17/8) ∧ 
  (∀ x : ℝ, f (-x) = f x) ∧
  (∀ x₁ x₂ : ℚ, |x₁| < |x₂| → f x₁ < f x₂) := by
  sorry


end cosine_like_properties_l2195_219528


namespace decorative_object_height_correct_l2195_219542

/-- Represents a circular fountain with water jets -/
structure Fountain where
  diameter : ℝ
  max_height : ℝ
  max_height_distance : ℝ
  decorative_object_height : ℝ

/-- Properties of the specific fountain described in the problem -/
def problem_fountain : Fountain where
  diameter := 20
  max_height := 8
  max_height_distance := 2
  decorative_object_height := 7.5

/-- Theorem stating that the decorative object height is correct for the given fountain parameters -/
theorem decorative_object_height_correct (f : Fountain) 
  (h1 : f.diameter = 20)
  (h2 : f.max_height = 8)
  (h3 : f.max_height_distance = 2) :
  f.decorative_object_height = 7.5 := by
  sorry

end decorative_object_height_correct_l2195_219542


namespace multiply_three_neg_two_l2195_219535

theorem multiply_three_neg_two : 3 * (-2) = -6 := by
  sorry

end multiply_three_neg_two_l2195_219535


namespace right_triangle_area_l2195_219516

theorem right_triangle_area (a b : ℝ) (ha : a = 5) (hb : b = 12) : 
  (1/2 : ℝ) * a * b = 30 := by
  sorry

end right_triangle_area_l2195_219516


namespace all_statements_imply_negation_l2195_219596

theorem all_statements_imply_negation (p q r : Prop) : 
  -- Statement 1
  ((p ∧ q ∧ r) → (¬p ∨ ¬q ∨ r)) ∧
  -- Statement 2
  ((p ∧ ¬q ∧ r) → (¬p ∨ ¬q ∨ r)) ∧
  -- Statement 3
  ((¬p ∧ q ∧ ¬r) → (¬p ∨ ¬q ∨ r)) ∧
  -- Statement 4
  ((¬p ∧ q ∧ ¬r) → (¬p ∨ ¬q ∨ r)) := by
  sorry

#check all_statements_imply_negation

end all_statements_imply_negation_l2195_219596


namespace problem_statement_l2195_219504

theorem problem_statement (a b : ℝ) 
  (h1 : a + 1 / (a + 1) = b + 1 / (b - 1) - 2)
  (h2 : a - b + 2 ≠ 0) : 
  a * b - a + b = 2 := by
  sorry

end problem_statement_l2195_219504


namespace sum_of_coefficients_l2195_219560

theorem sum_of_coefficients (a b x y : ℝ) : 
  (x = 3 ∧ y = -2) → 
  (a * x + b * y = 2 ∧ b * x + a * y = -3) → 
  a + b = -1 := by
sorry

end sum_of_coefficients_l2195_219560


namespace equivalent_form_proof_l2195_219518

theorem equivalent_form_proof (x y : ℝ) 
  (hx1 : x ≠ 0) (hx2 : x ≠ 3) (hy1 : y ≠ 0) (hy2 : y ≠ 7) 
  (h : (5 / x) + (4 / y) = (1 / 3)) : 
  x = (15 * y) / (y - 12) := by
sorry

end equivalent_form_proof_l2195_219518


namespace root_exists_in_interval_l2195_219562

def f (x : ℝ) := x^3 + x - 1

theorem root_exists_in_interval :
  Continuous f ∧ f 0 < 0 ∧ f 1 > 0 →
  ∃ x : ℝ, x ∈ Set.Ioo 0 1 ∧ f x = 0 := by
  sorry

end root_exists_in_interval_l2195_219562


namespace distribute_five_into_three_l2195_219569

/-- The number of ways to distribute n distinct objects into k distinct non-empty groups --/
def distribute (n k : ℕ) : ℕ := sorry

/-- The theorem stating that there are 150 ways to distribute 5 distinct objects into 3 distinct non-empty groups --/
theorem distribute_five_into_three : distribute 5 3 = 150 := by sorry

end distribute_five_into_three_l2195_219569


namespace toy_phone_price_l2195_219525

theorem toy_phone_price (bert_phones : ℕ) (tory_guns : ℕ) (gun_price : ℕ) (extra_earnings : ℕ) :
  bert_phones = 8 →
  tory_guns = 7 →
  gun_price = 20 →
  extra_earnings = 4 →
  (tory_guns * gun_price + extra_earnings) / bert_phones = 18 :=
by sorry

end toy_phone_price_l2195_219525


namespace inscribed_square_area_l2195_219593

/-- An ellipse with semi-major axis 2√2 and semi-minor axis 2√2 -/
def Ellipse (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 8 = 1

/-- A square inscribed in the ellipse with sides parallel to the axes -/
def InscribedSquare (s : ℝ) : Prop :=
  ∃ (x y : ℝ), Ellipse x y ∧ s = 2 * x ∧ s = 2 * y

/-- The area of the inscribed square is 32/3 -/
theorem inscribed_square_area :
  ∃ (s : ℝ), InscribedSquare s ∧ s^2 = 32/3 := by sorry

end inscribed_square_area_l2195_219593


namespace quadratic_two_distinct_roots_l2195_219549

theorem quadratic_two_distinct_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 4*x + m - 1 = 0 ∧ y^2 - 4*y + m - 1 = 0) → m < 5 :=
by sorry

end quadratic_two_distinct_roots_l2195_219549


namespace count_divisible_integers_l2195_219567

theorem count_divisible_integers : 
  ∃ (S : Finset Nat), 
    (∀ n ∈ S, n > 0 ∧ (8 * n) % ((n * (n + 1)) / 2) = 0) ∧ 
    (∀ n : Nat, n > 0 → (8 * n) % ((n * (n + 1)) / 2) = 0 → n ∈ S) ∧ 
    Finset.card S = 4 := by
  sorry

end count_divisible_integers_l2195_219567


namespace night_shift_guards_l2195_219564

/-- Represents the number of guards hired for a night shift -/
def num_guards (total_hours middle_guard_hours first_guard_hours last_guard_hours : ℕ) : ℕ :=
  let middle_guards := (total_hours - first_guard_hours - last_guard_hours) / middle_guard_hours
  1 + middle_guards + 1

/-- Theorem stating the number of guards hired for the night shift -/
theorem night_shift_guards : 
  num_guards 9 2 3 2 = 4 := by
  sorry

end night_shift_guards_l2195_219564


namespace painting_price_increase_l2195_219523

theorem painting_price_increase (P : ℝ) (X : ℝ) : 
  (P * (1 + X / 100) * (1 - 0.25) = P * 0.9) → X = 20 := by
  sorry

end painting_price_increase_l2195_219523


namespace equality_equivalence_l2195_219572

theorem equality_equivalence (a b c d : ℝ) : 
  (a - b)^2 + (c - d)^2 = 0 ↔ (a = b ∧ c = d) := by sorry

end equality_equivalence_l2195_219572


namespace exists_z_satisfying_conditions_l2195_219571

-- Define the complex function g
def g (z : ℂ) : ℂ := z^2 + 2*Complex.I*z + 2

-- State the theorem
theorem exists_z_satisfying_conditions : 
  ∃ z : ℂ, Complex.im z > 0 ∧ 
    (∃ a b : ℤ, g z = ↑a + ↑b * Complex.I ∧ 
      abs a ≤ 5 ∧ abs b ≤ 5) :=
sorry

end exists_z_satisfying_conditions_l2195_219571


namespace johns_book_expense_l2195_219585

def earnings : ℕ := 10 * 26

theorem johns_book_expense (money_left : ℕ) (book_expense : ℕ) : 
  money_left = 160 → 
  earnings = money_left + 2 * book_expense → 
  book_expense = 50 :=
by sorry

end johns_book_expense_l2195_219585


namespace derivative_of_f_composite_l2195_219582

-- Define the function f
def f (x : ℝ) : ℝ := x^3

-- State the theorem
theorem derivative_of_f_composite (a b : ℝ) :
  deriv (fun x => f (a - b*x)) = fun x => -3*b*(a - b*x)^2 := by sorry

end derivative_of_f_composite_l2195_219582


namespace line_not_in_second_quadrant_l2195_219555

/-- A line in 2D space defined by the equation x - y - 1 = 0 -/
def line (x y : ℝ) : Prop := x - y - 1 = 0

/-- The second quadrant of a 2D coordinate system -/
def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- Theorem stating that the line x - y - 1 = 0 does not pass through the second quadrant -/
theorem line_not_in_second_quadrant :
  ∀ x y : ℝ, line x y → ¬(second_quadrant x y) :=
sorry

end line_not_in_second_quadrant_l2195_219555


namespace systematic_sample_property_fourth_student_number_l2195_219527

/-- Represents a systematic sample from a population --/
structure SystematicSample where
  population_size : ℕ
  sample_size : ℕ
  interval : ℕ
  start : ℕ

/-- Calculates the nth element in a systematic sample --/
def nth_element (s : SystematicSample) (n : ℕ) : ℕ :=
  ((s.start + (n - 1) * s.interval - 1) % s.population_size) + 1

/-- Theorem stating the properties of the given systematic sample --/
theorem systematic_sample_property (s : SystematicSample) : 
  s.population_size = 54 ∧ 
  s.sample_size = 4 ∧ 
  s.start = 2 ∧ 
  nth_element s 2 = 28 ∧ 
  nth_element s 3 = 41 →
  nth_element s 4 = 1 := by
  sorry

/-- Main theorem to prove --/
theorem fourth_student_number : 
  ∃ (s : SystematicSample), 
    s.population_size = 54 ∧ 
    s.sample_size = 4 ∧ 
    s.start = 2 ∧ 
    nth_element s 2 = 28 ∧ 
    nth_element s 3 = 41 ∧
    nth_element s 4 = 1 := by
  sorry

end systematic_sample_property_fourth_student_number_l2195_219527


namespace average_income_P_R_l2195_219563

def average_income (x y : ℕ) : ℚ := (x + y) / 2

theorem average_income_P_R (P Q R : ℕ) : 
  average_income P Q = 5050 →
  average_income Q R = 6250 →
  P = 4000 →
  average_income P R = 5200 := by
sorry

end average_income_P_R_l2195_219563


namespace surface_area_of_solid_with_square_views_l2195_219545

/-- A solid with three square views -/
structure Solid where
  /-- The side length of the square views -/
  side_length : ℝ
  /-- The three views are squares -/
  square_views : Prop

/-- The surface area of a solid -/
def surface_area (s : Solid) : ℝ := sorry

/-- Theorem: The surface area of a solid with three square views of side length 2 is 24 -/
theorem surface_area_of_solid_with_square_views (s : Solid) 
  (h1 : s.side_length = 2) 
  (h2 : s.square_views) : 
  surface_area s = 24 := by sorry

end surface_area_of_solid_with_square_views_l2195_219545


namespace impossible_to_tile_rectangle_with_all_tetrominoes_l2195_219595

/-- Represents the different types of tetrominoes -/
inductive Tetromino
  | I
  | Square
  | Z
  | T
  | L

/-- Represents a color in a checkerboard pattern -/
inductive Color
  | Black
  | White

/-- Represents the coverage of squares by a tetromino on a checkerboard -/
structure TetrominoCoverage where
  black : Nat
  white : Nat

/-- The number of squares covered by each tetromino -/
def tetromino_size : Nat := 4

/-- The coverage of squares by each type of tetromino on a checkerboard -/
def tetromino_coverage (t : Tetromino) : TetrominoCoverage :=
  match t with
  | Tetromino.I => ⟨2, 2⟩
  | Tetromino.Square => ⟨2, 2⟩
  | Tetromino.Z => ⟨2, 2⟩
  | Tetromino.L => ⟨2, 2⟩
  | Tetromino.T => ⟨3, 1⟩  -- or ⟨1, 3⟩, doesn't matter for the proof

/-- Theorem stating that it's impossible to tile a rectangle with one of each tetromino type -/
theorem impossible_to_tile_rectangle_with_all_tetrominoes :
  ¬ ∃ (w h : Nat), w * h = 5 * tetromino_size ∧
    (∃ (c : Color), 
      (List.sum (List.map (λ t => (tetromino_coverage t).black) [Tetromino.I, Tetromino.Square, Tetromino.Z, Tetromino.T, Tetromino.L]) = w * h / 2) ∨
      (List.sum (List.map (λ t => (tetromino_coverage t).white) [Tetromino.I, Tetromino.Square, Tetromino.Z, Tetromino.T, Tetromino.L]) = w * h / 2)) :=
by sorry


end impossible_to_tile_rectangle_with_all_tetrominoes_l2195_219595


namespace two_fifths_divided_by_one_fifth_l2195_219503

theorem two_fifths_divided_by_one_fifth : (2 : ℚ) / 5 / ((1 : ℚ) / 5) = 2 := by sorry

end two_fifths_divided_by_one_fifth_l2195_219503
