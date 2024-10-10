import Mathlib

namespace letter_count_cycle_exists_l303_30371

/-- Represents the number of letters in the Russian word for a number -/
def russianWordLength (n : ℕ) : ℕ := sorry

/-- Generates the sequence of letter counts -/
def letterCountSequence (start : ℕ) : ℕ → ℕ
  | 0 => start
  | n + 1 => russianWordLength (letterCountSequence start n)

/-- Checks if a sequence has entered a cycle -/
def hasCycle (seq : ℕ → ℕ) (start : ℕ) (length : ℕ) : Prop :=
  ∃ k : ℕ, ∀ i, seq (k + i) = seq (k + i + length)

theorem letter_count_cycle_exists (start : ℕ) :
  ∃ k length : ℕ, hasCycle (letterCountSequence start) k length :=
sorry

end letter_count_cycle_exists_l303_30371


namespace circle_equation_line_equation_l303_30370

/-- A circle C passing through (2,-1), tangent to x+y=1, with center on y=-2x -/
structure CircleC where
  center : ℝ × ℝ
  radius : ℝ
  passes_through : center.1^2 + (center.2 + 1)^2 = radius^2
  tangent_to_line : |center.1 + center.2 - 1| / Real.sqrt 2 = radius
  center_on_line : center.2 = -2 * center.1

/-- A line passing through the origin and cutting a chord of length 2 on CircleC -/
structure LineL (c : CircleC) where
  slope : ℝ
  passes_origin : True
  cuts_chord : (2 * c.radius / Real.sqrt (1 + slope^2))^2 = 4

theorem circle_equation (c : CircleC) :
  ∀ x y : ℝ, (x - 1)^2 + (y + 2)^2 = 2 ↔ 
    (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2 :=
sorry

theorem line_equation (c : CircleC) (l : LineL c) :
  (l.slope = 0 ∧ ∀ x y : ℝ, y = l.slope * x ↔ x = 0) ∨
  (l.slope = -3/4 ∧ ∀ x y : ℝ, y = l.slope * x ↔ y = -3/4 * x) :=
sorry

end circle_equation_line_equation_l303_30370


namespace probability_of_two_pairs_and_one_different_l303_30327

-- Define the number of sides on each die
def numSides : ℕ := 10

-- Define the number of dice rolled
def numDice : ℕ := 5

-- Define the total number of possible outcomes
def totalOutcomes : ℕ := numSides ^ numDice

-- Define the number of ways to choose 2 distinct numbers for pairs
def waysToChoosePairs : ℕ := Nat.choose numSides 2

-- Define the number of choices for the fifth die
def choicesForFifthDie : ℕ := numSides - 2

-- Define the number of ways to arrange the digits
def arrangements : ℕ := Nat.factorial numDice / (2 * 2 * Nat.factorial 1)

-- Define the number of successful outcomes
def successfulOutcomes : ℕ := waysToChoosePairs * choicesForFifthDie * arrangements

-- The theorem to prove
theorem probability_of_two_pairs_and_one_different : 
  (successfulOutcomes : ℚ) / totalOutcomes = 108 / 1000 := by
  sorry


end probability_of_two_pairs_and_one_different_l303_30327


namespace intersection_point_a_value_l303_30304

-- Define the three lines
def line1 (a x y : ℝ) : Prop := a * x + 2 * y + 8 = 0
def line2 (x y : ℝ) : Prop := 4 * x + 3 * y = 10
def line3 (x y : ℝ) : Prop := 2 * x - y = 10

-- Theorem statement
theorem intersection_point_a_value :
  ∃! (a : ℝ), ∃! (x y : ℝ), line1 a x y ∧ line2 x y ∧ line3 x y → a = -1 := by
  sorry

end intersection_point_a_value_l303_30304


namespace cos_seven_pi_fourths_l303_30314

theorem cos_seven_pi_fourths : Real.cos (7 * π / 4) = 1 / Real.sqrt 2 := by
  sorry

end cos_seven_pi_fourths_l303_30314


namespace equation_solution_l303_30325

theorem equation_solution : ∃ x : ℤ, 121 * x = 75625 ∧ x = 625 := by
  sorry

end equation_solution_l303_30325


namespace condition_necessary_not_sufficient_l303_30366

theorem condition_necessary_not_sufficient :
  (∀ x : ℝ, (1 < x ∧ x < 3) → (1 < x ∧ x < 4)) ∧
  ¬(∀ x : ℝ, (1 < x ∧ x < 4) → (1 < x ∧ x < 3)) :=
by sorry

end condition_necessary_not_sufficient_l303_30366


namespace house_rooms_l303_30348

theorem house_rooms (outlets_per_room : ℕ) (total_outlets : ℕ) (h1 : outlets_per_room = 6) (h2 : total_outlets = 42) :
  total_outlets / outlets_per_room = 7 := by
  sorry

end house_rooms_l303_30348


namespace min_fraction_sum_l303_30316

def Digits : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

theorem min_fraction_sum :
  ∃ (W X Y Z : ℕ), W ∈ Digits ∧ X ∈ Digits ∧ Y ∈ Digits ∧ Z ∈ Digits ∧
  W ≠ X ∧ W ≠ Y ∧ W ≠ Z ∧ X ≠ Y ∧ X ≠ Z ∧ Y ≠ Z ∧
  ∀ (W' X' Y' Z' : ℕ), W' ∈ Digits → X' ∈ Digits → Y' ∈ Digits → Z' ∈ Digits →
  W' ≠ X' → W' ≠ Y' → W' ≠ Z' → X' ≠ Y' → X' ≠ Z' → Y' ≠ Z' →
  (W : ℚ) / X + (Y : ℚ) / Z ≤ (W' : ℚ) / X' + (Y' : ℚ) / Z' ∧
  (W : ℚ) / X + (Y : ℚ) / Z = 15 / 56 :=
by sorry

end min_fraction_sum_l303_30316


namespace intersection_k_range_l303_30399

-- Define the lines l₁ and l₂
def l₁ (x y k : ℝ) : Prop := y = 2 * x - 5 * k + 7
def l₂ (x y : ℝ) : Prop := y = -1/2 * x + 2

-- Define the intersection point
def intersection (x y k : ℝ) : Prop := l₁ x y k ∧ l₂ x y

-- Define the first quadrant condition
def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

-- Theorem statement
theorem intersection_k_range :
  ∀ k : ℝ, (∃ x y : ℝ, intersection x y k ∧ first_quadrant x y) ↔ (1 < k ∧ k < 3) :=
sorry

end intersection_k_range_l303_30399


namespace new_shoes_duration_l303_30397

/-- Given information about shoe costs and durability, prove the duration of new shoes. -/
theorem new_shoes_duration (used_repair_cost : ℝ) (used_duration : ℝ) (new_cost : ℝ) (cost_increase_percentage : ℝ) :
  used_repair_cost = 13.50 →
  used_duration = 1 →
  new_cost = 32.00 →
  cost_increase_percentage = 0.1852 →
  let new_duration := new_cost / (used_repair_cost * (1 + cost_increase_percentage))
  new_duration = 2 := by
  sorry

end new_shoes_duration_l303_30397


namespace one_third_of_one_fourth_l303_30324

theorem one_third_of_one_fourth (n : ℝ) : (3 / 10 : ℝ) * n = 64.8 → (1 / 3 : ℝ) * (1 / 4 : ℝ) * n = 18 := by
  sorry

end one_third_of_one_fourth_l303_30324


namespace problem_1_l303_30378

theorem problem_1 : -53 + 21 - (-79) - 37 = 10 := by sorry

end problem_1_l303_30378


namespace marathon_distance_theorem_l303_30300

/-- Represents the distance of a marathon in miles and yards -/
structure MarathonDistance :=
  (miles : ℕ)
  (yards : ℕ)

/-- Converts a MarathonDistance to total yards -/
def marathonToYards (d : MarathonDistance) : ℕ :=
  d.miles * 1760 + d.yards

/-- The standard marathon distance -/
def standardMarathon : MarathonDistance :=
  { miles := 26, yards := 395 }

/-- Converts total yards to miles and remaining yards -/
def yardsToMilesAndYards (totalYards : ℕ) : MarathonDistance :=
  { miles := totalYards / 1760,
    yards := totalYards % 1760 }

theorem marathon_distance_theorem :
  let totalYards := 15 * marathonToYards standardMarathon
  let result := yardsToMilesAndYards totalYards
  result.yards = 645 := by sorry

end marathon_distance_theorem_l303_30300


namespace a_equals_3_necessary_not_sufficient_l303_30388

/-- Two lines are parallel if their slopes are equal -/
def are_parallel (m1 m2 : ℝ) : Prop := m1 = m2

/-- The slope of the line (a^2 - 2a)x + y = 0 -/
def slope1 (a : ℝ) : ℝ := -(a^2 - 2*a)

/-- The slope of the line 3x + y + 1 = 0 -/
def slope2 : ℝ := -3

/-- The lines (a^2 - 2a)x + y = 0 and 3x + y + 1 = 0 are parallel -/
def lines_are_parallel (a : ℝ) : Prop := are_parallel (slope1 a) slope2

theorem a_equals_3_necessary_not_sufficient :
  (∀ a : ℝ, lines_are_parallel a → a = 3) ∧
  ¬(∀ a : ℝ, a = 3 → lines_are_parallel a) :=
by sorry

end a_equals_3_necessary_not_sufficient_l303_30388


namespace unique_solution_x2024_y3_3y_l303_30353

theorem unique_solution_x2024_y3_3y :
  ∀ x y : ℤ, x^2024 + y^3 = 3*y ↔ x = 0 ∧ y = 0 := by
  sorry

end unique_solution_x2024_y3_3y_l303_30353


namespace distinct_cubes_count_l303_30334

/-- The number of rotational symmetries of a cube -/
def cube_rotational_symmetries : ℕ := 24

/-- The number of unit cubes used to form the 2x2x2 cube -/
def num_unit_cubes : ℕ := 8

/-- The number of distinct 2x2x2 cubes that can be formed -/
def distinct_cubes : ℕ := Nat.factorial num_unit_cubes / cube_rotational_symmetries

theorem distinct_cubes_count :
  distinct_cubes = 1680 := by sorry

end distinct_cubes_count_l303_30334


namespace point_transformation_theorem_l303_30351

/-- Rotate a point (x, y) counterclockwise by 90° around (h, k) -/
def rotate90 (x y h k : ℝ) : ℝ × ℝ :=
  (h - (y - k), k + (x - h))

/-- Reflect a point (x, y) about the line y = x -/
def reflectAboutYeqX (x y : ℝ) : ℝ × ℝ :=
  (y, x)

theorem point_transformation_theorem (c d : ℝ) :
  let (x₁, y₁) := rotate90 c d 3 2
  let (x₂, y₂) := reflectAboutYeqX x₁ y₁
  (x₂ = 1 ∧ y₂ = -4) → d - c = -9 := by
  sorry

end point_transformation_theorem_l303_30351


namespace smallest_M_for_inequality_l303_30357

open Real

/-- The smallest real number M such that |∑ab(a²-b²)| ≤ M(∑a²)² holds for all real a, b, c -/
theorem smallest_M_for_inequality : 
  ∃ (M : ℝ), M = (9 * Real.sqrt 2) / 32 ∧ 
  (∀ (a b c : ℝ), |a*b*(a^2 - b^2) + b*c*(b^2 - c^2) + c*a*(c^2 - a^2)| ≤ M * (a^2 + b^2 + c^2)^2) ∧
  (∀ (M' : ℝ), (∀ (a b c : ℝ), |a*b*(a^2 - b^2) + b*c*(b^2 - c^2) + c*a*(c^2 - a^2)| ≤ M' * (a^2 + b^2 + c^2)^2) → M ≤ M') :=
sorry


end smallest_M_for_inequality_l303_30357


namespace negative_64_to_four_thirds_equals_256_l303_30376

theorem negative_64_to_four_thirds_equals_256 : (-64 : ℝ) ^ (4/3) = 256 := by
  sorry

end negative_64_to_four_thirds_equals_256_l303_30376


namespace outdoor_temp_correction_l303_30333

/-- Represents a thermometer with a linear error --/
structure Thermometer where
  /-- The slope of the linear relationship between actual and measured temperature --/
  k : ℝ
  /-- The y-intercept of the linear relationship between actual and measured temperature --/
  b : ℝ

/-- Calculates the actual temperature given a thermometer reading --/
def actualTemp (t : Thermometer) (reading : ℝ) : ℝ :=
  t.k * reading + t.b

theorem outdoor_temp_correction (t : Thermometer) 
  (h1 : actualTemp t (-11) = -7)
  (h2 : actualTemp t 32 = 36)
  (h3 : t.k = 1) -- This comes from solving the system of equations in the solution
  (h4 : t.b = -4) -- This comes from solving the system of equations in the solution
  : actualTemp t 22 = 18 := by
  sorry

end outdoor_temp_correction_l303_30333


namespace quadratic_equation_roots_l303_30330

theorem quadratic_equation_roots (k : ℝ) (θ : ℝ) : 
  (∃ x y : ℝ, x = Real.sin θ ∧ y = Real.cos θ ∧ 
    8 * x^2 + 6 * k * x + 2 * k + 1 = 0 ∧
    8 * y^2 + 6 * k * y + 2 * k + 1 = 0) →
  k = -10/9 := by
sorry

end quadratic_equation_roots_l303_30330


namespace toy_piles_l303_30372

theorem toy_piles (total : ℕ) (small : ℕ) (large : ℕ) : 
  total = 120 → 
  large = 2 * small → 
  total = small + large → 
  large = 80 := by
sorry

end toy_piles_l303_30372


namespace mike_lawn_money_l303_30384

/-- The amount of money Mike made mowing lawns -/
def lawn_money : ℝ := sorry

/-- The amount of money Mike made weed eating -/
def weed_eating_money : ℝ := 26

/-- The number of weeks the money lasted -/
def weeks : ℕ := 8

/-- The amount Mike spent per week -/
def weekly_spending : ℝ := 5

theorem mike_lawn_money :
  lawn_money = 14 :=
by
  have total_spent : ℝ := weekly_spending * weeks
  have total_money : ℝ := lawn_money + weed_eating_money
  have h1 : total_money = total_spent := by sorry
  sorry

end mike_lawn_money_l303_30384


namespace tutor_schedule_lcm_l303_30339

theorem tutor_schedule_lcm : Nat.lcm (Nat.lcm (Nat.lcm 4 5) 6) 8 = 120 := by
  sorry

end tutor_schedule_lcm_l303_30339


namespace draining_time_is_independent_variable_l303_30346

/-- Represents the water volume in the reservoir --/
def water_volume (t : ℝ) : ℝ := 50 - 2 * t

theorem draining_time_is_independent_variable :
  ∀ t₁ t₂ : ℝ, t₁ ≠ t₂ → water_volume t₁ ≠ water_volume t₂ :=
by sorry

end draining_time_is_independent_variable_l303_30346


namespace prime_composite_inequality_l303_30396

theorem prime_composite_inequality (n : ℕ) : 
  (Nat.Prime (2 * n - 1) → 
    ∀ (a : Fin n → ℕ), Function.Injective a → 
      ∃ i j : Fin n, (a i + a j : ℚ) / (Nat.gcd (a i) (a j)) ≥ 2 * n - 1) ∧
  (¬Nat.Prime (2 * n - 1) → 
    ∃ (a : Fin n → ℕ), Function.Injective a ∧
      ∀ i j : Fin n, (a i + a j : ℚ) / (Nat.gcd (a i) (a j)) < 2 * n - 1) :=
by sorry

end prime_composite_inequality_l303_30396


namespace store_rooms_problem_l303_30349

theorem store_rooms_problem (x : ℕ) : 
  (∃ (total_guests : ℕ), 
    total_guests = 7 * x + 7 ∧ 
    total_guests = 9 * (x - 1)) → 
  x = 8 := by
  sorry

end store_rooms_problem_l303_30349


namespace problem_statement_l303_30321

theorem problem_statement (x y : ℤ) (hx : x = 12) (hy : y = 7) :
  (x - y)^2 * (x + y)^2 = 9025 := by
sorry

end problem_statement_l303_30321


namespace quadratic_equation_solution_l303_30395

theorem quadratic_equation_solution :
  ∃! x : ℚ, x > 0 ∧ 3 * x^2 + 11 * x - 20 = 0 :=
by
  -- The unique positive solution is x = 4/3
  use 4/3
  sorry

end quadratic_equation_solution_l303_30395


namespace invalid_external_diagonals_l303_30398

theorem invalid_external_diagonals : ¬ ∃ (a b c : ℝ), 
  (a > 0 ∧ b > 0 ∧ c > 0) ∧
  (8 = Real.sqrt (a^2 + b^2) ∨ 8 = Real.sqrt (b^2 + c^2) ∨ 8 = Real.sqrt (a^2 + c^2)) ∧
  (15 = Real.sqrt (a^2 + b^2) ∨ 15 = Real.sqrt (b^2 + c^2) ∨ 15 = Real.sqrt (a^2 + c^2)) ∧
  (18 = Real.sqrt (a^2 + b^2) ∨ 18 = Real.sqrt (b^2 + c^2) ∨ 18 = Real.sqrt (a^2 + c^2)) :=
by sorry

end invalid_external_diagonals_l303_30398


namespace min_m_value_l303_30317

-- Define the points A and B
def A (m : ℝ) : ℝ × ℝ := (1, m)
def B (x : ℝ) : ℝ × ℝ := (-1, 1 - |x|)

-- Define symmetry with respect to the origin
def symmetric_wrt_origin (p q : ℝ × ℝ) : Prop :=
  p.1 = -q.1 ∧ p.2 = -q.2

-- State the theorem
theorem min_m_value (m x : ℝ) 
  (h : symmetric_wrt_origin (A m) (B x)) : 
  ∀ k, m ≤ k → -1 ≤ k :=
sorry

end min_m_value_l303_30317


namespace tissue_with_mitotic_and_meiotic_cells_is_gonad_l303_30301

structure Cell where
  chromosomeCount : ℕ

structure Tissue where
  cells : Set Cell

def isSomaticCell (c : Cell) : Prop := sorry

def isGermCell (c : Cell) (sc : Cell) : Prop :=
  isSomaticCell sc ∧ c.chromosomeCount = sc.chromosomeCount / 2

def containsMitoticCells (t : Tissue) : Prop :=
  ∃ c ∈ t.cells, isSomaticCell c

def containsMeioticCells (t : Tissue) : Prop :=
  ∃ c sc, c ∈ t.cells ∧ isGermCell c sc

def isGonad (t : Tissue) : Prop :=
  containsMitoticCells t ∧ containsMeioticCells t

theorem tissue_with_mitotic_and_meiotic_cells_is_gonad (t : Tissue) :
  containsMitoticCells t → containsMeioticCells t → isGonad t :=
by sorry

end tissue_with_mitotic_and_meiotic_cells_is_gonad_l303_30301


namespace tangent_y_intercept_l303_30355

-- Define the circles
def circle1_center : ℝ × ℝ := (3, 0)
def circle1_radius : ℝ := 3
def circle2_center : ℝ × ℝ := (7, 0)
def circle2_radius : ℝ := 1

-- Define the tangent line (implicitly)
def tangent_line : Set (ℝ × ℝ) := sorry

-- Condition that the tangent line touches both circles in the first quadrant
axiom tangent_touches_circles :
  ∃ (p q : ℝ × ℝ),
    p.1 > 0 ∧ p.2 > 0 ∧
    q.1 > 0 ∧ q.2 > 0 ∧
    p ∈ tangent_line ∧
    q ∈ tangent_line ∧
    (p.1 - circle1_center.1)^2 + (p.2 - circle1_center.2)^2 = circle1_radius^2 ∧
    (q.1 - circle2_center.1)^2 + (q.2 - circle2_center.2)^2 = circle2_radius^2

-- Theorem statement
theorem tangent_y_intercept :
  ∃ (y : ℝ), y = 9 ∧ (0, y) ∈ tangent_line :=
sorry

end tangent_y_intercept_l303_30355


namespace stratified_sample_female_result_l303_30318

/-- Represents the number of female athletes to be selected in a stratified sample -/
def stratified_sample_female (total_male : ℕ) (total_female : ℕ) (sample_size : ℕ) : ℕ :=
  (total_female * sample_size) / (total_male + total_female)

/-- Theorem: In a stratified sampling of 28 people from a population of 98 athletes 
    (56 male and 42 female), the number of female athletes that should be selected is 12 -/
theorem stratified_sample_female_result : 
  stratified_sample_female 56 42 28 = 12 := by
  sorry

end stratified_sample_female_result_l303_30318


namespace odd_function_and_monotone_increasing_l303_30312

/-- An odd function f(x) = x^2 + mx -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x

/-- f is an odd function -/
def is_odd (m : ℝ) : Prop := ∀ x, f m (-x) = -(f m x)

/-- f is monotonically increasing on an interval -/
def is_monotone_increasing (m : ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f m x < f m y

theorem odd_function_and_monotone_increasing :
  ∃ m, is_odd m ∧ 
  ∃ a, 1 < a ∧ a ≤ 3 ∧ 
  is_monotone_increasing m (-1) (a - 2) ∧
  m = 2 := by sorry

end odd_function_and_monotone_increasing_l303_30312


namespace x_value_proof_l303_30373

theorem x_value_proof (x : ℝ) (h : 9 / x^2 = x / 25) : x = (225 : ℝ)^(1/3) := by
  sorry

end x_value_proof_l303_30373


namespace total_pages_read_l303_30369

-- Define reading speeds for each genre and focus level
def novel_speed : Fin 3 → ℕ
| 0 => 21  -- low focus
| 1 => 25  -- medium focus
| 2 => 30  -- high focus
| _ => 0

def graphic_novel_speed : Fin 3 → ℕ
| 0 => 30  -- low focus
| 1 => 36  -- medium focus
| 2 => 42  -- high focus
| _ => 0

def comic_book_speed : Fin 3 → ℕ
| 0 => 45  -- low focus
| 1 => 54  -- medium focus
| 2 => 60  -- high focus
| _ => 0

def non_fiction_speed : Fin 3 → ℕ
| 0 => 18  -- low focus
| 1 => 22  -- medium focus
| 2 => 28  -- high focus
| _ => 0

def biography_speed : Fin 3 → ℕ
| 0 => 20  -- low focus
| 1 => 24  -- medium focus
| 2 => 29  -- high focus
| _ => 0

-- Define time allocations for each hour
def hour1_allocation : List (ℕ × ℕ × ℕ) := [
  (20, 2, 0),  -- 20 minutes, high focus, novel
  (10, 0, 1),  -- 10 minutes, low focus, graphic novel
  (15, 1, 3),  -- 15 minutes, medium focus, non-fiction
  (15, 0, 4)   -- 15 minutes, low focus, biography
]

def hour2_allocation : List (ℕ × ℕ × ℕ) := [
  (25, 1, 2),  -- 25 minutes, medium focus, comic book
  (15, 2, 1),  -- 15 minutes, high focus, graphic novel
  (20, 0, 0)   -- 20 minutes, low focus, novel
]

def hour3_allocation : List (ℕ × ℕ × ℕ) := [
  (10, 2, 3),  -- 10 minutes, high focus, non-fiction
  (20, 1, 4),  -- 20 minutes, medium focus, biography
  (30, 0, 2)   -- 30 minutes, low focus, comic book
]

-- Function to calculate pages read for a given time, focus, and genre
def pages_read (time : ℕ) (focus : Fin 3) (genre : Fin 5) : ℚ :=
  let speed := match genre with
    | 0 => novel_speed focus
    | 1 => graphic_novel_speed focus
    | 2 => comic_book_speed focus
    | 3 => non_fiction_speed focus
    | 4 => biography_speed focus
    | _ => 0
  (time : ℚ) / 60 * speed

-- Function to calculate total pages read for a list of allocations
def total_pages (allocations : List (ℕ × ℕ × ℕ)) : ℚ :=
  allocations.foldl (fun acc (time, focus, genre) => acc + pages_read time ⟨focus, by sorry⟩ ⟨genre, by sorry⟩) 0

-- Theorem stating the total pages read
theorem total_pages_read :
  ⌊total_pages hour1_allocation + total_pages hour2_allocation + total_pages hour3_allocation⌋ = 100 := by
  sorry


end total_pages_read_l303_30369


namespace additional_investment_rate_l303_30356

theorem additional_investment_rate
  (initial_investment : ℝ)
  (initial_rate : ℝ)
  (additional_investment : ℝ)
  (total_rate : ℝ)
  (h1 : initial_investment = 2800)
  (h2 : initial_rate = 0.05)
  (h3 : additional_investment = 1400)
  (h4 : total_rate = 0.06)
  : (initial_investment * initial_rate + additional_investment * (112 / 1400)) / (initial_investment + additional_investment) = total_rate :=
by sorry

end additional_investment_rate_l303_30356


namespace dealer_pricing_theorem_l303_30335

/-- A dealer's pricing strategy -/
structure DealerPricing where
  cash_discount : ℝ
  profit_percentage : ℝ
  articles_sold : ℕ
  articles_cost_price : ℕ

/-- Calculate the listing percentage above cost price -/
def listing_percentage (d : DealerPricing) : ℝ :=
  -- Define the calculation here
  sorry

/-- Theorem: Under specific conditions, the listing percentage is 60% -/
theorem dealer_pricing_theorem (d : DealerPricing) 
  (h1 : d.cash_discount = 0.15)
  (h2 : d.profit_percentage = 0.36)
  (h3 : d.articles_sold = 25)
  (h4 : d.articles_cost_price = 20) :
  listing_percentage d = 0.60 := by
  sorry

end dealer_pricing_theorem_l303_30335


namespace expression_value_proof_l303_30303

theorem expression_value_proof (a b c k : ℤ) 
  (ha : a = 30) (hb : b = 25) (hc : c = 4) (hk : k = 3) : 
  (a - (b - k * c)) - ((a - k * b) - c) = 66 := by
  sorry

end expression_value_proof_l303_30303


namespace four_bottles_cost_l303_30338

/-- The cost of a certain number of bottles of mineral water -/
def cost (bottles : ℕ) : ℚ :=
  if bottles = 3 then 3/2 else (3/2) * (bottles : ℚ) / 3

/-- Theorem: The cost of 4 bottles of mineral water is 2, given that 3 bottles cost 1.50 -/
theorem four_bottles_cost : cost 4 = 2 := by
  sorry

end four_bottles_cost_l303_30338


namespace probability_not_adjacent_l303_30323

theorem probability_not_adjacent (n : ℕ) : 
  n = 5 → (36 : ℚ) / (120 : ℚ) = (3 : ℚ) / (10 : ℚ) := by sorry

end probability_not_adjacent_l303_30323


namespace base7_sum_equality_l303_30306

-- Define a type for base 7 digits
def Base7Digit := { n : Nat // n > 0 ∧ n < 7 }

-- Function to convert a three-digit base 7 number to natural number
def base7ToNat (a b c : Base7Digit) : Nat :=
  49 * a.val + 7 * b.val + c.val

-- Statement of the theorem
theorem base7_sum_equality 
  (A B C : Base7Digit) 
  (hDistinct : A ≠ B ∧ B ≠ C ∧ A ≠ C) 
  (hSum : base7ToNat A B C + base7ToNat B C A + base7ToNat C A B = 
          343 * A.val + 49 * A.val + 7 * A.val) : 
  B.val + C.val = 6 := by
sorry

end base7_sum_equality_l303_30306


namespace five_pairs_l303_30367

/-- The number of pairs of natural numbers (a, b) satisfying the given conditions -/
def count_pairs : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ => 
    let (a, b) := p
    a ≥ b ∧ (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 6)
    (Finset.product (Finset.range 100) (Finset.range 100))).card

/-- Theorem stating that there are exactly 5 pairs of natural numbers satisfying the conditions -/
theorem five_pairs : count_pairs = 5 := by
  sorry

end five_pairs_l303_30367


namespace sqrt_360000_equals_600_l303_30341

theorem sqrt_360000_equals_600 : Real.sqrt 360000 = 600 := by
  sorry

end sqrt_360000_equals_600_l303_30341


namespace f_monotonicity_and_min_value_l303_30383

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x - 2 / x - a * (Real.log x - 1 / x^2)

-- Define the derivative of f
def f_deriv (a : ℝ) (x : ℝ) : ℝ := (x^2 + 2) * (x - a) / x^3

-- Define the minimum value function g
def g (a : ℝ) : ℝ := a - a * Real.log a - 1 / a

-- Theorem statement
theorem f_monotonicity_and_min_value (a : ℝ) (h : a > 0) :
  (∀ x ∈ Set.Ioo 0 a, f_deriv a x < 0) ∧
  (∀ x ∈ Set.Ioi a, f_deriv a x > 0) ∧
  g a < 1 := by sorry

end

end f_monotonicity_and_min_value_l303_30383


namespace johns_allowance_l303_30313

/-- Calculates the amount of allowance John received given his initial amount, spending, and final amount -/
def calculate_allowance (initial : ℕ) (spent : ℕ) (final : ℕ) : ℕ :=
  final - (initial - spent)

/-- Proves that John's allowance was 26 dollars given the problem conditions -/
theorem johns_allowance :
  let initial := 5
  let spent := 2
  let final := 29
  calculate_allowance initial spent final = 26 := by
  sorry

end johns_allowance_l303_30313


namespace bridget_profit_is_42_l303_30391

/-- Calculates Bridget's profit from bread sales --/
def bridget_profit (total_loaves : ℕ) (morning_price afternoon_price late_afternoon_price cost_per_loaf : ℚ) : ℚ :=
  let morning_sold := total_loaves / 3
  let morning_revenue := morning_sold * morning_price
  
  let afternoon_remaining := total_loaves - morning_sold
  let afternoon_sold := afternoon_remaining / 2
  let afternoon_revenue := afternoon_sold * afternoon_price
  
  let late_afternoon_remaining := afternoon_remaining - afternoon_sold
  let late_afternoon_sold := late_afternoon_remaining / 4
  let late_afternoon_revenue := late_afternoon_sold * late_afternoon_price
  
  let evening_remaining := late_afternoon_remaining - late_afternoon_sold
  let evening_price := late_afternoon_price / 2
  let evening_revenue := evening_remaining * evening_price
  
  let total_revenue := morning_revenue + afternoon_revenue + late_afternoon_revenue + evening_revenue
  let total_cost := total_loaves * cost_per_loaf
  
  total_revenue - total_cost

/-- Theorem stating Bridget's profit is $42 --/
theorem bridget_profit_is_42 :
  bridget_profit 60 3 (3/2) 1 1 = 42 := by
  sorry


end bridget_profit_is_42_l303_30391


namespace triangle_perimeter_l303_30363

theorem triangle_perimeter : 
  ∀ (a b c : ℝ), 
    a = 10 ∧ b = 6 ∧ c = 7 → 
    a + b > c ∧ a + c > b ∧ b + c > a → 
    a + b + c = 23 := by
  sorry

end triangle_perimeter_l303_30363


namespace six_digit_scrambled_divisibility_l303_30365

theorem six_digit_scrambled_divisibility (a b c : Nat) 
  (ha : a ∈ Finset.range 10) 
  (hb : b ∈ Finset.range 10) 
  (hc : c ∈ Finset.range 10) 
  (hpos : 100000 * a + 10000 * b + 1000 * c + 100 * a + 10 * c + b > 0) :
  let Z := 100000 * a + 10000 * b + 1000 * c + 100 * a + 10 * c + b
  ∃ k : Nat, Z = 101 * k := by
  sorry

end six_digit_scrambled_divisibility_l303_30365


namespace square_field_area_l303_30374

/-- Given a square field with two 1-meter wide gates, where the cost of drawing barbed wire
    is 1.10 per meter and the total cost is 732.6, prove that the area of the field is 27889 sq m. -/
theorem square_field_area (side : ℝ) (gate_width : ℝ) (wire_cost_per_meter : ℝ) (total_cost : ℝ)
  (h1 : gate_width = 1)
  (h2 : wire_cost_per_meter = 1.1)
  (h3 : total_cost = 732.6)
  (h4 : wire_cost_per_meter * (4 * side - 2 * gate_width) = total_cost) :
  side^2 = 27889 := by
  sorry

end square_field_area_l303_30374


namespace equation1_solution_equation2_no_solution_l303_30315

-- Define the equations
def equation1 (x : ℝ) : Prop := (x - 3) / (x - 2) - 1 = 3 / x
def equation2 (x : ℝ) : Prop := (x + 1) / (x - 1) - 4 / (x^2 - 1) = 1

-- Theorem for equation 1
theorem equation1_solution :
  (∃! x : ℝ, equation1 x) ∧ equation1 (3/2) :=
sorry

-- Theorem for equation 2
theorem equation2_no_solution :
  ¬∃ x : ℝ, equation2 x :=
sorry

end equation1_solution_equation2_no_solution_l303_30315


namespace catch_up_solution_l303_30386

/-- Represents the problem of a car catching up to a truck -/
def CatchUpProblem (truckSpeed carInitialSpeed carSpeedIncrease distance : ℝ) : Prop :=
  ∃ (t : ℝ),
    t > 0 ∧
    (carInitialSpeed * t + carSpeedIncrease * t * (t - 1) / 2) = (truckSpeed * t + distance)

/-- The solution to the catch-up problem -/
theorem catch_up_solution :
  CatchUpProblem 40 50 5 135 →
  ∃ (t : ℝ), t = 6 ∧ CatchUpProblem 40 50 5 135 := by sorry

#check catch_up_solution

end catch_up_solution_l303_30386


namespace hockey_league_games_l303_30354

theorem hockey_league_games (n : ℕ) (total_games : ℕ) (h1 : n = 16) (h2 : total_games = 1200) :
  ∃ x : ℕ, x * n * (n - 1) / 2 = total_games ∧ x = 10 := by
  sorry

end hockey_league_games_l303_30354


namespace tangent_line_proof_l303_30352

def circle_center : ℝ × ℝ := (6, 3)
def circle_radius : ℝ := 2
def point_p : ℝ × ℝ := (10, 0)

def is_on_circle (p : ℝ × ℝ) : Prop :=
  (p.1 - circle_center.1)^2 + (p.2 - circle_center.2)^2 = circle_radius^2

def is_on_line (p : ℝ × ℝ) : Prop :=
  4 * p.1 - 3 * p.2 = 19

theorem tangent_line_proof :
  ∃ (q : ℝ × ℝ),
    is_on_circle q ∧
    is_on_line q ∧
    is_on_line point_p ∧
    ∀ (r : ℝ × ℝ), is_on_circle r ∧ is_on_line r → r = q :=
  sorry

end tangent_line_proof_l303_30352


namespace inequality_system_solution_l303_30375

theorem inequality_system_solution (a : ℝ) : 
  (∀ x : ℝ, (x > 5 ∧ x > a) ↔ x > 5) → a ≤ 5 := by
  sorry

end inequality_system_solution_l303_30375


namespace train_length_l303_30332

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 60 → time = 15 → ∃ (length : ℝ), abs (length - 250.05) < 0.01 := by
  sorry

#check train_length

end train_length_l303_30332


namespace triangle_cos_2C_l303_30379

theorem triangle_cos_2C (a b : ℝ) (S : ℝ) (C : ℝ) :
  a = 8 →
  b = 5 →
  S = 12 →
  S = (1/2) * a * b * Real.sin C →
  Real.cos (2 * C) = 7/25 := by
sorry

end triangle_cos_2C_l303_30379


namespace min_distance_to_origin_l303_30311

/-- Circle A with equation x^2 + y^2 = 1 -/
def circle_A : Set (ℝ × ℝ) :=
  {p | p.1^2 + p.2^2 = 1}

/-- Circle B with equation (x-3)^2 + (y+4)^2 = 10 -/
def circle_B : Set (ℝ × ℝ) :=
  {p | (p.1 - 3)^2 + (p.2 + 4)^2 = 10}

/-- The point P satisfies the condition that its distances to the tangent points on circles A and B are equal -/
def point_P : Set (ℝ × ℝ) :=
  {p | ∃ d e : ℝ × ℝ, d ∈ circle_A ∧ e ∈ circle_B ∧ 
       (p.1 - d.1)^2 + (p.2 - d.2)^2 = (p.1 - e.1)^2 + (p.2 - e.2)^2}

/-- The minimum distance from point P to the origin is 8/5 -/
theorem min_distance_to_origin : 
  ∀ p ∈ point_P, (∀ q ∈ point_P, p.1^2 + p.2^2 ≤ q.1^2 + q.2^2) → 
  p.1^2 + p.2^2 = (8/5)^2 := by
sorry

end min_distance_to_origin_l303_30311


namespace value_of_r_l303_30377

theorem value_of_r (k r : ℝ) (h1 : 5 = k * 3^r) (h2 : 45 = k * 9^r) : r = 2 := by
  sorry

end value_of_r_l303_30377


namespace ball_max_height_l303_30322

def h (t : ℝ) : ℝ := -20 * t^2 + 100 * t + 36

theorem ball_max_height :
  ∃ (max : ℝ), max = 161 ∧ ∀ (t : ℝ), h t ≤ max :=
sorry

end ball_max_height_l303_30322


namespace min_value_on_circle_l303_30361

theorem min_value_on_circle (a b : ℝ) (h : a^2 + b^2 - 4*a + 3 = 0) :
  2 ≤ Real.sqrt (a^2 + b^2) + 1 ∧ 
  ∃ (a₀ b₀ : ℝ), a₀^2 + b₀^2 - 4*a₀ + 3 = 0 ∧ Real.sqrt (a₀^2 + b₀^2) + 1 = 2 := by
  sorry

end min_value_on_circle_l303_30361


namespace root_conditions_imply_a_range_l303_30364

theorem root_conditions_imply_a_range (a : ℝ) :
  (∃ (x₁ x₂ : ℝ), x₁ > 1 ∧ x₂ < 1 ∧ 
   x₁^2 + a*x₁ + a^2 - a - 2 = 0 ∧
   x₂^2 + a*x₂ + a^2 - a - 2 = 0) →
  -1 < a ∧ a < 1 := by
sorry

end root_conditions_imply_a_range_l303_30364


namespace parabola_intersection_l303_30331

theorem parabola_intersection :
  let f (x : ℝ) := 3 * x^2 - 9 * x - 15
  let g (x : ℝ) := x^2 - 5 * x + 7
  ∀ x y : ℝ, f x = g x ∧ y = f x ↔ 
    (x = 1 + 2 * Real.sqrt 3 ∧ y = 19 - 6 * Real.sqrt 3) ∨
    (x = 1 - 2 * Real.sqrt 3 ∧ y = 19 + 6 * Real.sqrt 3) :=
by sorry

end parabola_intersection_l303_30331


namespace complex_number_theorem_l303_30382

theorem complex_number_theorem (w z : ℂ) 
  (h1 : Complex.abs (w + z) = 2) 
  (h2 : Complex.abs (w^2 + z^2) = 8) : 
  Complex.abs (w^4 + z^4) = 56 := by
sorry

end complex_number_theorem_l303_30382


namespace knight_moves_correct_l303_30307

/-- The least number of moves for a knight to travel from one corner to the diagonally opposite corner on an n×n chessboard. -/
def knight_moves (n : ℕ) : ℕ := 2 * ((n + 1) / 3)

/-- Theorem: For an n×n chessboard where n ≥ 4, the least number of moves for a knight to travel
    from one corner to the diagonally opposite corner is equal to 2 ⌊(n+1)/3⌋. -/
theorem knight_moves_correct (n : ℕ) (h : n ≥ 4) :
  knight_moves n = 2 * ((n + 1) / 3) :=
by sorry

end knight_moves_correct_l303_30307


namespace find_other_number_l303_30328

theorem find_other_number (x y : ℤ) : 
  (3 * x + 4 * y = 151) → 
  ((x = 19 ∨ y = 19) → (x = 25 ∨ y = 25)) := by
sorry

end find_other_number_l303_30328


namespace age_difference_brother_cousin_l303_30368

/-- Proves that the age difference between Lexie's brother and cousin is 5 years -/
theorem age_difference_brother_cousin : 
  ∀ (lexie_age brother_age sister_age uncle_age grandma_age cousin_age : ℕ),
  lexie_age = 8 →
  grandma_age = 68 →
  lexie_age = brother_age + 6 →
  sister_age = 2 * lexie_age →
  uncle_age + 12 = grandma_age →
  cousin_age = brother_age + 5 →
  cousin_age - brother_age = 5 :=
by
  sorry

end age_difference_brother_cousin_l303_30368


namespace correct_calculation_l303_30310

theorem correct_calculation (a : ℝ) : 2 * a^3 * 3 * a^5 = 6 * a^8 := by
  sorry

end correct_calculation_l303_30310


namespace difference_of_squares_2023_2022_l303_30385

theorem difference_of_squares_2023_2022 : 2023^2 - 2022^2 = 4045 := by
  sorry

end difference_of_squares_2023_2022_l303_30385


namespace even_quadratic_function_sum_l303_30342

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem even_quadratic_function_sum (a b : ℝ) :
  let f := fun x => a * x^2 + b * x
  IsEven f ∧ (∀ x ∈ Set.Icc (a - 1) (2 * a), f x ∈ Set.range f) →
  a + b = 1/3 := by
  sorry

end even_quadratic_function_sum_l303_30342


namespace sum_abcd_equals_negative_twenty_thirds_l303_30390

theorem sum_abcd_equals_negative_twenty_thirds 
  (a b c d : ℚ) 
  (h : a + 2 = b + 4 ∧ b + 4 = c + 6 ∧ c + 6 = d + 8 ∧ d + 8 = a + b + c + d + 10) : 
  a + b + c + d = -20/3 := by
sorry

end sum_abcd_equals_negative_twenty_thirds_l303_30390


namespace incident_ray_equation_l303_30393

-- Define the points A and B
def A : ℝ × ℝ := (-2, 3)
def B : ℝ × ℝ := (5, 7)

-- Define the reflection of B across the x-axis
def B_reflected : ℝ × ℝ := (B.1, -B.2)

-- Define the line equation
def line_equation (x y : ℝ) : Prop :=
  10 * x + 7 * y - 1 = 0

-- Theorem statement
theorem incident_ray_equation :
  line_equation A.1 A.2 ∧
  line_equation B_reflected.1 B_reflected.2 :=
sorry

end incident_ray_equation_l303_30393


namespace math_club_pair_sequences_l303_30347

/-- The number of students in the Math Club -/
def num_students : ℕ := 12

/-- The number of sessions per week -/
def sessions_per_week : ℕ := 3

/-- The number of students selected per session -/
def students_per_session : ℕ := 2

/-- The number of different pair sequences that can be selected in one week -/
def pair_sequences_per_week : ℕ := (num_students * (num_students - 1)) ^ sessions_per_week

theorem math_club_pair_sequences :
  pair_sequences_per_week = 2299968 :=
sorry

end math_club_pair_sequences_l303_30347


namespace sum_of_squares_and_products_l303_30389

theorem sum_of_squares_and_products (a b c : ℝ) 
  (nonneg_a : a ≥ 0) (nonneg_b : b ≥ 0) (nonneg_c : c ≥ 0)
  (sum_of_squares : a^2 + b^2 + c^2 = 52)
  (sum_of_products : a*b + b*c + c*a = 24) : 
  a + b + c = 10 := by
sorry

end sum_of_squares_and_products_l303_30389


namespace price_reduction_percentage_l303_30392

theorem price_reduction_percentage (initial_price final_price : ℝ) 
  (h1 : initial_price = 25)
  (h2 : final_price = 16)
  (h3 : initial_price > 0)
  (h4 : final_price > 0)
  (h5 : final_price < initial_price) :
  ∃ (x : ℝ), 
    (x > 0) ∧ 
    (x < 1) ∧ 
    (initial_price * (1 - x)^2 = final_price) ∧
    (x = 1 - (4/5)) := by
  sorry

end price_reduction_percentage_l303_30392


namespace acrobats_count_correct_unique_solution_l303_30320

/-- Represents the number of acrobats in the circus show -/
def num_acrobats : ℕ := 10

/-- Represents the number of elephants in the circus show -/
def num_elephants : ℕ := 5

/-- The total number of legs observed in the circus show -/
def total_legs : ℕ := 40

/-- The total number of heads observed in the circus show -/
def total_heads : ℕ := 15

/-- Theorem stating that the number of acrobats is correct given the conditions -/
theorem acrobats_count_correct :
  (2 * num_acrobats + 4 * num_elephants = total_legs) ∧
  (num_acrobats + num_elephants = total_heads) :=
by sorry

/-- Theorem proving the uniqueness of the solution -/
theorem unique_solution (a e : ℕ) :
  (2 * a + 4 * e = total_legs) →
  (a + e = total_heads) →
  a = num_acrobats ∧ e = num_elephants :=
by sorry

end acrobats_count_correct_unique_solution_l303_30320


namespace book_selection_l303_30343

theorem book_selection (picture_books : ℕ) (sci_fi_books : ℕ) (total_selection : ℕ) : 
  picture_books = 4 → sci_fi_books = 2 → total_selection = 4 →
  (Nat.choose (picture_books + sci_fi_books) total_selection - 
   Nat.choose picture_books total_selection) = 14 :=
by sorry

end book_selection_l303_30343


namespace greatest_two_digit_multiple_of_17_l303_30381

theorem greatest_two_digit_multiple_of_17 : 
  ∀ n : ℕ, n < 100 → n % 17 = 0 → n ≤ 85 :=
by
  sorry

end greatest_two_digit_multiple_of_17_l303_30381


namespace unknown_number_proof_l303_30359

theorem unknown_number_proof : ∃ x : ℝ, x + 5 * 12 / (180 / 3) = 61 ∧ x = 60 := by
  sorry

end unknown_number_proof_l303_30359


namespace line_intersects_ellipse_l303_30329

/-- The line equation y = kx + 1 - 2k -/
def line (k x : ℝ) : ℝ := k * x + 1 - 2 * k

/-- The ellipse equation x²/9 + y²/4 = 1 -/
def ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 4 = 1

/-- The point P(2,1) is inside the ellipse -/
def point_inside_ellipse : Prop := 2^2 / 9 + 1^2 / 4 < 1

theorem line_intersects_ellipse :
  ∀ k : ℝ, ∃ x y : ℝ, line k x = y ∧ ellipse x y :=
sorry

end line_intersects_ellipse_l303_30329


namespace trey_bracelet_sales_l303_30394

/-- The average number of bracelets Trey needs to sell each day -/
def average_bracelets_per_day (total_cost : ℕ) (num_days : ℕ) (bracelet_price : ℕ) : ℚ :=
  (total_cost : ℚ) / (num_days : ℚ)

/-- Theorem stating that Trey needs to sell 8 bracelets per day on average -/
theorem trey_bracelet_sales :
  average_bracelets_per_day 112 14 1 = 8 := by
  sorry

end trey_bracelet_sales_l303_30394


namespace bill_vote_change_l303_30350

theorem bill_vote_change (total_voters : ℕ) (first_for first_against : ℕ) 
  (second_for second_against : ℕ) : 
  total_voters = 400 →
  first_for + first_against = total_voters →
  first_against > first_for →
  second_for + second_against = total_voters →
  second_for > second_against →
  (second_for - second_against) = 2 * (first_against - first_for) →
  second_for = (12 * first_against) / 11 →
  second_for - first_for = 60 := by
sorry

end bill_vote_change_l303_30350


namespace max_tiles_on_floor_l303_30345

/-- Represents a rectangular shape with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Calculates the maximum number of tiles that can fit in one dimension -/
def fitInDimension (floorSize tileSize : ℕ) : ℕ :=
  floorSize / tileSize

/-- Calculates the number of tiles that can fit on the floor for a given orientation -/
def tilesForOrientation (floor tile : Rectangle) : ℕ :=
  (fitInDimension floor.width tile.width) * (fitInDimension floor.height tile.height)

/-- Theorem: The maximum number of 50x40 tiles on a 120x150 floor is 9 -/
theorem max_tiles_on_floor :
  let floor : Rectangle := ⟨120, 150⟩
  let tile : Rectangle := ⟨50, 40⟩
  let orientation1 := tilesForOrientation floor tile
  let orientation2 := tilesForOrientation floor ⟨tile.height, tile.width⟩
  max orientation1 orientation2 = 9 := by
  sorry

end max_tiles_on_floor_l303_30345


namespace middle_integer_is_five_l303_30358

/-- A function that checks if a number is a one-digit positive integer -/
def isOneDigitPositive (n : ℕ) : Prop := 0 < n ∧ n < 10

/-- A function that checks if a number is odd -/
def isOdd (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k + 1

/-- The main theorem -/
theorem middle_integer_is_five :
  ∀ n : ℕ,
  isOneDigitPositive n ∧
  isOdd n ∧
  isOneDigitPositive (n - 2) ∧
  isOdd (n - 2) ∧
  isOneDigitPositive (n + 2) ∧
  isOdd (n + 2) ∧
  ((n - 2) + n + (n + 2)) = ((n - 2) * n * (n + 2)) / 8
  →
  n = 5 :=
by sorry

end middle_integer_is_five_l303_30358


namespace inequality_proof_l303_30344

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) :
  a * (b^2 + c^2) + b * (c^2 + a^2) ≥ 4 * a * b * c :=
by sorry

end inequality_proof_l303_30344


namespace inequality_proof_l303_30360

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (Real.arctan ((a * d - b * c) / (a * c + b * d)))^2 ≥ 2 * (1 - (a * c + b * d) / Real.sqrt ((a^2 + b^2) * (c^2 + d^2))) := by
  sorry

end inequality_proof_l303_30360


namespace charity_fundraising_l303_30302

theorem charity_fundraising (donation_percentage : ℚ) (num_organizations : ℕ) (amount_per_org : ℚ) :
  donation_percentage = 80 / 100 →
  num_organizations = 8 →
  amount_per_org = 250 →
  (num_organizations : ℚ) * amount_per_org / donation_percentage = 2500 :=
by sorry

end charity_fundraising_l303_30302


namespace simplify_and_evaluate_l303_30326

theorem simplify_and_evaluate (a b : ℤ) (h1 : a = -2) (h2 : b = 1) :
  ((a - 2*b)^2 - (a + 3*b)*(a - 2*b)) / b = 20 := by
  sorry

end simplify_and_evaluate_l303_30326


namespace min_squares_for_symmetric_x_l303_30305

/-- Represents a position on the 4x4 grid -/
structure Position :=
  (row : Fin 4)
  (col : Fin 4)

/-- Represents the state of the 4x4 grid -/
def Grid := Fin 4 → Fin 4 → Bool

/-- The initial grid with squares at (1,3) and (2,4) shaded -/
def initialGrid : Grid :=
  fun r c => (r = 0 ∧ c = 2) ∨ (r = 1 ∧ c = 3)

/-- Checks if a grid has both vertical and horizontal symmetry -/
def isSymmetric (g : Grid) : Prop :=
  (∀ r c, g r c = g r (3 - c)) ∧  -- Vertical symmetry
  (∀ r c, g r c = g (3 - r) c)    -- Horizontal symmetry

/-- Checks if a grid forms an 'X' shape -/
def formsX (g : Grid) : Prop :=
  (∀ r, g r r = true) ∧ 
  (∀ r, g r (3 - r) = true) ∧
  (∀ r c, r ≠ c ∧ r ≠ (3 - c) → g r c = false)

/-- The main theorem stating that 4 additional squares are needed -/
theorem min_squares_for_symmetric_x : 
  ∃ (finalGrid : Grid),
    (∀ r c, initialGrid r c → finalGrid r c) ∧
    isSymmetric finalGrid ∧
    formsX finalGrid ∧
    (∀ (g : Grid), 
      (∀ r c, initialGrid r c → g r c) → 
      isSymmetric g → 
      formsX g → 
      (∃ (newSquares : List Position),
        newSquares.length = 4 ∧
        (∀ p ∈ newSquares, g p.row p.col ∧ ¬initialGrid p.row p.col))) :=
sorry

end min_squares_for_symmetric_x_l303_30305


namespace fruit_shop_apples_l303_30336

/-- Given the ratio of mangoes : oranges : apples and the number of mangoes,
    calculate the number of apples -/
theorem fruit_shop_apples (ratio_mangoes ratio_oranges ratio_apples num_mangoes : ℕ) 
    (h_ratio : ratio_mangoes = 10 ∧ ratio_oranges = 2 ∧ ratio_apples = 3)
    (h_mangoes : num_mangoes = 120) :
    (num_mangoes / ratio_mangoes) * ratio_apples = 36 := by
  sorry

#check fruit_shop_apples

end fruit_shop_apples_l303_30336


namespace ellipse_m_range_l303_30337

/-- The equation of an ellipse in terms of m -/
def is_ellipse (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (m + 2) - y^2 / (m + 1) = 1

/-- The range of m for which the equation represents an ellipse -/
def m_range (m : ℝ) : Prop :=
  (m > -2 ∧ m < -3/2) ∨ (m > -3/2 ∧ m < -1)

theorem ellipse_m_range :
  ∀ m : ℝ, is_ellipse m → m_range m :=
sorry

end ellipse_m_range_l303_30337


namespace number_of_workers_l303_30319

/-- Proves that the number of men working on the jobs is 3 --/
theorem number_of_workers (time_per_job : ℝ) (num_jobs : ℕ) (hourly_rate : ℝ) (total_earned : ℝ) : ℕ :=
  by
  -- Assume the given conditions
  have h1 : time_per_job = 1 := by sorry
  have h2 : num_jobs = 5 := by sorry
  have h3 : hourly_rate = 10 := by sorry
  have h4 : total_earned = 150 := by sorry

  -- Define the number of workers
  let num_workers : ℕ := 3

  -- Prove that num_workers satisfies the conditions
  have h5 : (↑num_workers : ℝ) * num_jobs * hourly_rate = total_earned := by sorry

  -- Return the number of workers
  exact num_workers

end number_of_workers_l303_30319


namespace spontaneous_reaction_l303_30362

theorem spontaneous_reaction (ΔH ΔS : ℝ) (h1 : ΔH = -98.2) (h2 : ΔS = 70.5 / 1000) :
  ∀ T : ℝ, T ≥ 0 → ΔH - T * ΔS < 0 := by
sorry

end spontaneous_reaction_l303_30362


namespace paint_cost_per_kg_paint_cost_is_60_l303_30309

/-- The cost of paint per kilogram, given the coverage and the cost to paint a cube. -/
theorem paint_cost_per_kg (coverage : Real) (cube_side : Real) (total_cost : Real) : Real :=
  let surface_area := 6 * cube_side * cube_side
  let paint_needed := surface_area / coverage
  total_cost / paint_needed

/-- Proof that the cost of paint per kilogram is $60. -/
theorem paint_cost_is_60 :
  paint_cost_per_kg 20 10 1800 = 60 := by
  sorry

end paint_cost_per_kg_paint_cost_is_60_l303_30309


namespace goldbach_conjecture_false_l303_30308

/-- Goldbach's conjecture: Every even number greater than 2 can be expressed as the sum of two odd prime numbers -/
def goldbach_conjecture : Prop :=
  ∀ n : ℕ, n > 2 → Even n → ∃ p q : ℕ, Prime p ∧ Prime q ∧ Odd p ∧ Odd q ∧ n = p + q

/-- Theorem stating that Goldbach's conjecture is false -/
theorem goldbach_conjecture_false : ¬goldbach_conjecture := by
  sorry

/-- Lemma: 4 is a counterexample to Goldbach's conjecture -/
lemma four_is_counterexample : 
  ¬(∃ p q : ℕ, Prime p ∧ Prime q ∧ Odd p ∧ Odd q ∧ 4 = p + q) := by
  sorry

end goldbach_conjecture_false_l303_30308


namespace jessica_watermelons_l303_30387

/-- The number of watermelons Jessica has left -/
def watermelons_left (initial : ℕ) (eaten : ℕ) : ℕ :=
  initial - eaten

/-- Proof that Jessica has 8 watermelons left -/
theorem jessica_watermelons : watermelons_left 35 27 = 8 := by
  sorry

end jessica_watermelons_l303_30387


namespace initial_ratio_proof_l303_30340

theorem initial_ratio_proof (initial_boarders : ℕ) (new_boarders : ℕ) :
  initial_boarders = 560 →
  new_boarders = 80 →
  ∃ (initial_day_scholars : ℕ),
    (initial_boarders : ℚ) / initial_day_scholars = 7 / 16 ∧
    (initial_boarders + new_boarders : ℚ) / initial_day_scholars = 1 / 2 :=
by sorry

end initial_ratio_proof_l303_30340


namespace sufficient_not_necessary_l303_30380

theorem sufficient_not_necessary (x : ℝ) : 
  (x = 1 → x^2 - 3*x + 2 = 0) ∧ 
  (∃ y : ℝ, y ≠ 1 ∧ y^2 - 3*y + 2 = 0) :=
by sorry

end sufficient_not_necessary_l303_30380
