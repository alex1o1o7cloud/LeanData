import Mathlib

namespace solve_triangle_equation_l962_96233

-- Define the ∆ operation
def triangle (A B : ℕ) : ℕ := 2 * A + B

-- Theorem statement
theorem solve_triangle_equation : 
  ∃ x : ℕ, triangle (triangle 3 2) x = 20 ∧ x = 4 := by
sorry

end solve_triangle_equation_l962_96233


namespace cosine_equality_proof_l962_96220

theorem cosine_equality_proof : ∃! (n : ℤ), 0 ≤ n ∧ n ≤ 360 ∧ Real.cos (n * π / 180) = Real.cos (1234 * π / 180) ∧ n = 154 := by
  sorry

end cosine_equality_proof_l962_96220


namespace sum_of_squared_coefficients_l962_96293

/-- The expression to be simplified -/
def expression (x : ℝ) : ℝ := 3 * (x^3 - 2*x^2 + 3) - 5 * (x^4 - 4*x^2 + 2)

/-- The coefficients of the fully simplified expression -/
def coefficients : List ℝ := [-5, 3, 14, -1]

/-- Theorem: The sum of the squares of the coefficients of the fully simplified expression is 231 -/
theorem sum_of_squared_coefficients :
  (coefficients.map (λ c => c^2)).sum = 231 := by
  sorry

end sum_of_squared_coefficients_l962_96293


namespace distance_from_center_l962_96209

-- Define the circle and points
def Circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 50}

-- Define the conditions
def Conditions (A B C : ℝ × ℝ) : Prop :=
  A ∈ Circle ∧ 
  C ∈ Circle ∧ 
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 36 ∧  -- AB = 6
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = 4 ∧   -- BC = 2
  (A.1 - B.1) * (C.1 - B.1) + (A.2 - B.2) * (C.2 - B.2) = 0  -- Angle ABC is right

-- State the theorem
theorem distance_from_center (A B C : ℝ × ℝ) 
  (h : Conditions A B C) : B.1^2 + B.2^2 = 26 := by
  sorry

end distance_from_center_l962_96209


namespace change5_is_census_change5_most_suitable_for_census_l962_96263

/-- Represents a survey method -/
inductive SurveyMethod
  | Sample
  | Census

/-- Represents a survey target -/
structure SurveyTarget where
  name : String
  method : SurveyMethod

/-- Definition of a census -/
def isCensus (target : SurveyTarget) : Prop :=
  target.method = SurveyMethod.Census

/-- The "Chang'e 5" probe components survey -/
def change5Survey : SurveyTarget :=
  { name := "All components of the Chang'e 5 probe"
    method := SurveyMethod.Census }

/-- Theorem: The "Chang'e 5" probe components survey is a census -/
theorem change5_is_census : isCensus change5Survey := by
  sorry

/-- Theorem: The "Chang'e 5" probe components survey is the most suitable for a census -/
theorem change5_most_suitable_for_census (other : SurveyTarget) :
    isCensus other → other = change5Survey := by
  sorry

end change5_is_census_change5_most_suitable_for_census_l962_96263


namespace last_round_win_ratio_l962_96241

/-- Represents the number of matches in a kickboxing competition --/
structure KickboxingCompetition where
  firstTwoRoundsMatches : ℕ  -- Total matches in first two rounds
  lastRoundMatches : ℕ      -- Total matches in last round
  totalWins : ℕ             -- Total matches won by Brendan

/-- Theorem stating the ratio of matches won in the last round --/
theorem last_round_win_ratio (comp : KickboxingCompetition)
  (h1 : comp.firstTwoRoundsMatches = 12)
  (h2 : comp.lastRoundMatches = 4)
  (h3 : comp.totalWins = 14) :
  (comp.totalWins - comp.firstTwoRoundsMatches) * 2 = comp.lastRoundMatches := by
  sorry

#check last_round_win_ratio

end last_round_win_ratio_l962_96241


namespace quadratic_root_transformation_l962_96280

theorem quadratic_root_transformation (a b c r s : ℝ) : 
  (∀ x, a * x^2 + b * x + c = 0 ↔ x = r ∨ x = s) →
  (∀ y, y^2 - b * y + 4 * a * c = 0 ↔ y = 2 * a * r + b ∨ y = 2 * a * s + b) :=
by sorry

end quadratic_root_transformation_l962_96280


namespace simplest_quadratic_radical_l962_96265

/-- A quadratic radical is in its simplest form if it has no fractions inside 
    the radical and no coefficients outside the radical. -/
def is_simplest_quadratic_radical (x : ℝ) : Prop :=
  ∃ (n : ℕ), x = Real.sqrt n ∧ n ≠ 0 ∧ n ≠ 1 ∧ ∀ (m : ℕ), m * m ≤ n → m = 1

theorem simplest_quadratic_radical :
  is_simplest_quadratic_radical (Real.sqrt 5) ∧
  ¬is_simplest_quadratic_radical (Real.sqrt 0.2) ∧
  ¬is_simplest_quadratic_radical (Real.sqrt (1/2)) ∧
  ¬is_simplest_quadratic_radical (Real.sqrt 12) :=
by sorry

end simplest_quadratic_radical_l962_96265


namespace binomial_150_150_l962_96225

theorem binomial_150_150 : Nat.choose 150 150 = 1 := by
  sorry

end binomial_150_150_l962_96225


namespace max_third_term_geometric_progression_l962_96246

/-- Given an arithmetic progression of three terms starting with 5, 
    where adding 5 to the second term and 30 to the third term creates a geometric progression, 
    the maximum possible value for the third term of the resulting geometric progression is 45. -/
theorem max_third_term_geometric_progression : 
  ∀ (d : ℝ), 
  let a₁ : ℝ := 5
  let a₂ : ℝ := 5 + d
  let a₃ : ℝ := 5 + 2*d
  let g₁ : ℝ := a₁
  let g₂ : ℝ := a₂ + 5
  let g₃ : ℝ := a₃ + 30
  (g₂^2 = g₁ * g₃) →
  g₃ ≤ 45 :=
by sorry

end max_third_term_geometric_progression_l962_96246


namespace carpet_shampooing_time_l962_96201

theorem carpet_shampooing_time 
  (jason_rate : ℝ) 
  (tom_rate : ℝ) 
  (h1 : jason_rate = 1 / 3) 
  (h2 : tom_rate = 1 / 6) : 
  1 / (jason_rate + tom_rate) = 2 := by
  sorry

end carpet_shampooing_time_l962_96201


namespace sum_of_roots_quadratic_l962_96215

/-- The sum of roots of a quadratic equation x^2 + (m-1)x + (m+n) = 0 is 1 - m -/
theorem sum_of_roots_quadratic (m n : ℝ) (hm : m ≠ 1) (hn : n ≠ -m) :
  let f : ℝ → ℝ := λ x => x^2 + (m-1)*x + (m+n)
  (∃ r s : ℝ, f r = 0 ∧ f s = 0) → r + s = 1 - m :=
by sorry

end sum_of_roots_quadratic_l962_96215


namespace subset_pairs_count_for_six_elements_l962_96277

-- Define a function that counts the number of valid subset pairs
def countValidSubsetPairs (n : ℕ) : ℕ :=
  if n = 0 then 1 else 3 * countValidSubsetPairs (n - 1) - 1

-- Theorem statement
theorem subset_pairs_count_for_six_elements :
  countValidSubsetPairs 6 = 365 := by
  sorry

end subset_pairs_count_for_six_elements_l962_96277


namespace janet_fertilizer_spread_rate_l962_96296

theorem janet_fertilizer_spread_rate 
  (horses : ℕ) 
  (fertilizer_per_horse : ℚ) 
  (acres : ℕ) 
  (fertilizer_per_acre : ℚ) 
  (days : ℕ) 
  (h1 : horses = 80)
  (h2 : fertilizer_per_horse = 5)
  (h3 : acres = 20)
  (h4 : fertilizer_per_acre = 400)
  (h5 : days = 25)
  : (acres : ℚ) / days = 0.8 := by
  sorry

end janet_fertilizer_spread_rate_l962_96296


namespace min_value_bn_Sn_l962_96212

def a (n : ℕ) : ℕ := n * (n + 1)

def S (n : ℕ) : ℚ := 1 - 1 / (n + 1)

def b (n : ℕ) : ℤ := n - 8

theorem min_value_bn_Sn :
  (∀ n : ℕ, (b n : ℚ) * S n ≥ -4) ∧
  (∃ n : ℕ, (b n : ℚ) * S n = -4) :=
sorry

end min_value_bn_Sn_l962_96212


namespace a_plus_b_equals_five_l962_96297

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 > 0}
def B (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b ≤ 0}

-- State the theorem
theorem a_plus_b_equals_five (a b : ℝ) :
  (A ∪ B a b = Set.univ) →
  (A ∩ B a b = {x | 3 < x ∧ x ≤ 4}) →
  a + b = 5 := by sorry

end a_plus_b_equals_five_l962_96297


namespace inequality_solution_set_sum_of_coordinates_l962_96239

-- Problem 1
theorem inequality_solution_set (x : ℝ) :
  x + |2*x - 1| < 3 ↔ -2 < x ∧ x < 4/3 := by sorry

-- Problem 2
theorem sum_of_coordinates (x y z : ℝ) 
  (h1 : x^2 + y^2 + z^2 = 1) 
  (h2 : x + 2*y + 3*z = Real.sqrt 14) : 
  x + y + z = 3 * Real.sqrt 14 / 7 := by sorry

end inequality_solution_set_sum_of_coordinates_l962_96239


namespace rachel_books_total_l962_96204

/-- The number of books Rachel has in total -/
def total_books (mystery_shelves picture_shelves scifi_shelves bio_shelves books_per_shelf : ℕ) : ℕ :=
  (mystery_shelves + picture_shelves + scifi_shelves + bio_shelves) * books_per_shelf

/-- Theorem stating that Rachel has 135 books in total -/
theorem rachel_books_total :
  total_books 6 2 3 4 9 = 135 := by
  sorry

end rachel_books_total_l962_96204


namespace ribbon_length_difference_l962_96259

/-- Represents the dimensions of a box in centimeters -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the ribbon length for the first method -/
def ribbonLengthMethod1 (box : BoxDimensions) : ℝ :=
  2 * box.length + 2 * box.width + 4 * box.height + 24

/-- Calculates the ribbon length for the second method -/
def ribbonLengthMethod2 (box : BoxDimensions) : ℝ :=
  2 * box.length + 4 * box.width + 2 * box.height + 24

/-- Theorem stating that the difference in ribbon lengths equals one side of the box -/
theorem ribbon_length_difference (box : BoxDimensions) 
    (h1 : box.length = 22) 
    (h2 : box.width = 22) 
    (h3 : box.height = 11) : 
  ribbonLengthMethod2 box - ribbonLengthMethod1 box = box.length := by
  sorry

end ribbon_length_difference_l962_96259


namespace triangle_theorem_l962_96271

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Main theorem about the triangle -/
theorem triangle_theorem (t : Triangle) 
  (h1 : t.a * Real.sin (2 * t.B) = Real.sqrt 3 * t.b * Real.sin t.A)
  (h2 : Real.cos t.A = 1 / 3) :
  t.B = π / 6 ∧ Real.sin t.C = (2 * Real.sqrt 6 + 1) / 6 := by
  sorry


end triangle_theorem_l962_96271


namespace triangle_exists_iff_altitudes_condition_l962_96279

/-- A triangle with altitudes m_a, m_b, and m_c exists if and only if
    1/m_a + 1/m_b > 1/m_c and 1/m_b + 1/m_c > 1/m_a and 1/m_c + 1/m_a > 1/m_b -/
theorem triangle_exists_iff_altitudes_condition
  (m_a m_b m_c : ℝ) (h_pos_a : m_a > 0) (h_pos_b : m_b > 0) (h_pos_c : m_c > 0) :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    (a * m_a = b * m_b) ∧ (b * m_b = c * m_c) ∧ (c * m_c = a * m_a) ↔
  (1 / m_a + 1 / m_b > 1 / m_c) ∧
  (1 / m_b + 1 / m_c > 1 / m_a) ∧
  (1 / m_c + 1 / m_a > 1 / m_b) :=
by sorry

end triangle_exists_iff_altitudes_condition_l962_96279


namespace cynthia_water_balloons_l962_96269

/-- The number of water balloons each person has -/
structure WaterBalloons where
  janice : ℕ
  randy : ℕ
  cynthia : ℕ

/-- The conditions of the water balloon distribution -/
def water_balloon_conditions (wb : WaterBalloons) : Prop :=
  wb.janice = 6 ∧
  wb.randy = wb.janice / 2 ∧
  wb.cynthia = 4 * wb.randy

theorem cynthia_water_balloons (wb : WaterBalloons) 
  (h : water_balloon_conditions wb) : wb.cynthia = 12 := by
  sorry

#check cynthia_water_balloons

end cynthia_water_balloons_l962_96269


namespace valid_assignment_l962_96295

/-- Represents the squares in the grid -/
inductive Square
| One | Nine | A | B | C | D | E | F | G

/-- Represents the direction of arrows -/
inductive Direction
| Right | RightUp | Up | Down

/-- Define the arrow directions for each square -/
def arrowDirection (s : Square) : Option Direction :=
  match s with
  | Square.One => some Direction.Right
  | Square.B => some Direction.RightUp
  | Square.E => some Direction.Right
  | Square.C => some Direction.Right
  | Square.D => some Direction.Up
  | Square.A => some Direction.Down
  | Square.G => some Direction.Right
  | Square.F => some Direction.Right
  | Square.Nine => none

/-- Define the next square based on the current square and arrow direction -/
def nextSquare (s : Square) : Option Square :=
  match s, arrowDirection s with
  | Square.One, some Direction.Right => some Square.B
  | Square.B, some Direction.RightUp => some Square.E
  | Square.E, some Direction.Right => some Square.C
  | Square.C, some Direction.Right => some Square.D
  | Square.D, some Direction.Up => some Square.A
  | Square.A, some Direction.Down => some Square.G
  | Square.G, some Direction.Right => some Square.F
  | Square.F, some Direction.Right => some Square.Nine
  | _, _ => none

/-- The assignment of numbers to squares -/
def assignment : Square → Nat
| Square.One => 1
| Square.Nine => 9
| Square.A => 6
| Square.B => 2
| Square.C => 4
| Square.D => 5
| Square.E => 3
| Square.F => 8
| Square.G => 7

/-- Theorem stating that the assignment is valid -/
theorem valid_assignment :
  (∀ s : Square, s ≠ Square.One ∧ s ≠ Square.Nine →
    2 ≤ assignment s ∧ assignment s ≤ 8) ∧
  (∀ s : Square, s ≠ Square.Nine →
    ∃ next : Square, nextSquare s = some next ∧
    assignment next = assignment s + 1) :=
sorry

end valid_assignment_l962_96295


namespace cube_root_of_negative_eight_l962_96286

theorem cube_root_of_negative_eight : 
  ∃ x : ℝ, x^3 = -8 ∧ x = -2 := by sorry

end cube_root_of_negative_eight_l962_96286


namespace parabola_param_valid_l962_96229

/-- A parameterization of the curve y = x^2 -/
def parabola_param (t : ℝ) : ℝ × ℝ := (t, t^2)

/-- The curve y = x^2 -/
def parabola (x : ℝ) : ℝ := x^2

theorem parabola_param_valid :
  ∀ (x : ℝ), ∃ (t : ℝ), parabola_param t = (x, parabola x) :=
sorry

end parabola_param_valid_l962_96229


namespace double_root_values_l962_96290

def polynomial (b₃ b₂ b₁ : ℤ) (x : ℤ) : ℤ := x^4 + b₃ * x^3 + b₂ * x^2 + b₁ * x + 50

def is_double_root (p : ℤ → ℤ) (s : ℤ) : Prop :=
  p s = 0 ∧ (∃ q : ℤ → ℤ, ∀ x, p x = (x - s)^2 * q x)

theorem double_root_values (b₃ b₂ b₁ : ℤ) (s : ℤ) :
  is_double_root (polynomial b₃ b₂ b₁) s → s ∈ ({-5, -2, -1, 1, 2, 5} : Set ℤ) :=
by sorry

end double_root_values_l962_96290


namespace total_games_is_105_l962_96289

/-- The number of teams in the league -/
def num_teams : ℕ := 15

/-- The total number of games played in the league -/
def total_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem stating that the total number of games played is 105 -/
theorem total_games_is_105 : total_games num_teams = 105 := by
  sorry

end total_games_is_105_l962_96289


namespace rationalization_factor_l962_96258

theorem rationalization_factor (a b : ℝ) :
  (Real.sqrt a - Real.sqrt b) * (Real.sqrt a + Real.sqrt b) = a - b :=
by sorry

end rationalization_factor_l962_96258


namespace spherical_shell_surface_area_l962_96224

/-- The surface area of a spherical shell formed by two hemispheres -/
theorem spherical_shell_surface_area 
  (r : ℝ) -- radius of the inner hemisphere
  (h1 : r > 0) -- radius is positive
  (h2 : r^2 * π = 200 * π) -- base area of inner hemisphere is 200π
  : 2 * π * ((r + 1)^2 - r^2) = 2 * π + 40 * Real.sqrt 2 * π :=
by sorry

end spherical_shell_surface_area_l962_96224


namespace dried_fruit_percentage_l962_96275

/-- Represents the composition of a trail mix -/
structure TrailMix where
  nuts : ℚ
  dried_fruit : ℚ
  chocolate_chips : ℚ
  composition_sum : nuts + dried_fruit + chocolate_chips = 1

/-- The combined mixture of two trail mixes -/
def combined_mixture (mix1 mix2 : TrailMix) : TrailMix :=
  { nuts := (mix1.nuts + mix2.nuts) / 2,
    dried_fruit := (mix1.dried_fruit + mix2.dried_fruit) / 2,
    chocolate_chips := (mix1.chocolate_chips + mix2.chocolate_chips) / 2,
    composition_sum := by sorry }

theorem dried_fruit_percentage
  (sue_mix : TrailMix)
  (jane_mix : TrailMix)
  (h_sue_nuts : sue_mix.nuts = 0.3)
  (h_sue_dried_fruit : sue_mix.dried_fruit = 0.7)
  (h_jane_nuts : jane_mix.nuts = 0.6)
  (h_jane_chocolate : jane_mix.chocolate_chips = 0.4)
  (h_combined_nuts : (combined_mixture sue_mix jane_mix).nuts = 0.45) :
  (combined_mixture sue_mix jane_mix).dried_fruit = 0.35 := by
  sorry

end dried_fruit_percentage_l962_96275


namespace initial_money_proof_l962_96299

/-- The amount of money Mrs. Hilt had initially -/
def initial_money : ℕ := 15

/-- The cost of the pencil in cents -/
def pencil_cost : ℕ := 11

/-- The amount of money left after buying the pencil -/
def money_left : ℕ := 4

/-- Theorem stating that the initial money equals the sum of the pencil cost and money left -/
theorem initial_money_proof : initial_money = pencil_cost + money_left := by
  sorry

end initial_money_proof_l962_96299


namespace problem_solution_l962_96243

/-- Given constants a, b, and c satisfying the specified conditions, prove that a + 2b + 3c = 74 -/
theorem problem_solution (a b c : ℝ) 
  (h1 : ∀ x, (x - a) * (x - b) / (x - c) ≤ 0 ↔ (x < -6 ∨ |x - 30| ≤ 2))
  (h2 : a < b) : 
  a + 2 * b + 3 * c = 74 := by
sorry

end problem_solution_l962_96243


namespace fraction_inequality_l962_96255

theorem fraction_inequality (a b : ℝ) (h : a > b) : a / 4 > b / 4 := by
  sorry

end fraction_inequality_l962_96255


namespace hyperbola_equation_l962_96200

theorem hyperbola_equation (a b c : ℝ) : 
  (2 * c = 10) →  -- focal length is 10
  (b / a = 2) →   -- slope of asymptote is 2
  (a^2 + b^2 = c^2) →  -- relation between a, b, and c
  (a^2 = 5 ∧ b^2 = 20) := by
sorry

end hyperbola_equation_l962_96200


namespace M_subset_N_l962_96202

-- Define the sets M and N
def M : Set ℝ := {α | ∃ k : ℤ, α = k * 90 ∨ α = k * 180 + 45}
def N : Set ℝ := {α | ∃ k : ℤ, α = k * 45}

-- Theorem statement
theorem M_subset_N : M ⊆ N := by
  sorry

end M_subset_N_l962_96202


namespace A_D_mutually_exclusive_not_complementary_l962_96231

-- Define the sample space for a die throw
def Ω : Set Nat := {1, 2, 3, 4, 5, 6}

-- Define the events
def A : Set Nat := {n ∈ Ω | n % 2 = 1}
def B : Set Nat := {n ∈ Ω | n % 2 = 0}
def C : Set Nat := {n ∈ Ω | n % 2 = 0}
def D : Set Nat := {2, 4}

-- Define mutually exclusive
def mutually_exclusive (X Y : Set Nat) : Prop := X ∩ Y = ∅

-- Define complementary
def complementary (X Y : Set Nat) : Prop := X ∪ Y = Ω

-- Theorem to prove
theorem A_D_mutually_exclusive_not_complementary :
  mutually_exclusive A D ∧ ¬complementary A D :=
by sorry

end A_D_mutually_exclusive_not_complementary_l962_96231


namespace original_number_l962_96285

theorem original_number (y : ℚ) : (1 - (1 / y) = 5 / 4) → y = -4 := by
  sorry

end original_number_l962_96285


namespace power_of_power_five_l962_96205

theorem power_of_power_five : (5^2)^4 = 390625 := by
  sorry

end power_of_power_five_l962_96205


namespace total_oranges_count_l962_96228

/-- The number of oranges picked by Joan -/
def joan_oranges : ℕ := 37

/-- The number of oranges picked by Sara -/
def sara_oranges : ℕ := 10

/-- The total number of oranges picked -/
def total_oranges : ℕ := joan_oranges + sara_oranges

theorem total_oranges_count : total_oranges = 47 := by
  sorry

end total_oranges_count_l962_96228


namespace tailors_hourly_rate_l962_96287

theorem tailors_hourly_rate (num_shirts : ℕ) (num_pants : ℕ) (shirt_time : ℚ) (total_cost : ℚ) :
  num_shirts = 10 →
  num_pants = 12 →
  shirt_time = 3/2 →
  total_cost = 1530 →
  (num_shirts * shirt_time + num_pants * (2 * shirt_time)) * (total_cost / (num_shirts * shirt_time + num_pants * (2 * shirt_time))) = 30 := by
  sorry

end tailors_hourly_rate_l962_96287


namespace union_of_specific_sets_l962_96218

theorem union_of_specific_sets :
  let A : Set ℕ := {2, 5, 6}
  let B : Set ℕ := {3, 5}
  A ∪ B = {2, 3, 5, 6} := by
sorry

end union_of_specific_sets_l962_96218


namespace quadratic_distinct_roots_l962_96292

/-- 
Given a quadratic equation kx^2 - 6x + 9 = 0, this theorem states that
for the equation to have two distinct real roots, k must be less than 1
and not equal to 0.
-/
theorem quadratic_distinct_roots (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ k * x^2 - 6*x + 9 = 0 ∧ k * y^2 - 6*y + 9 = 0) ↔ 
  (k < 1 ∧ k ≠ 0) :=
sorry

end quadratic_distinct_roots_l962_96292


namespace floor_sqrt_27_squared_l962_96273

theorem floor_sqrt_27_squared : ⌊Real.sqrt 27⌋^2 = 25 := by
  sorry

end floor_sqrt_27_squared_l962_96273


namespace characterize_g_l962_96207

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- Define the properties of g
def is_valid_g (g : ℝ → ℝ) : Prop :=
  ∀ x, f (g x) = 9 * x^2 - 12 * x + 4

-- Theorem statement
theorem characterize_g :
  ∀ g : ℝ → ℝ, is_valid_g g ↔ (∀ x, g x = 3 * x - 2 ∨ g x = -3 * x + 2) :=
by sorry

end characterize_g_l962_96207


namespace common_root_exists_polynomial_common_root_l962_96235

def coefficients : Finset Int := {-7, 4, -3, 6}

theorem common_root_exists (a b c d : Int) 
  (h : {a, b, c, d} = coefficients) : 
  (a : ℝ) + b + c + d = 0 := by sorry

theorem polynomial_common_root (a b c d : Int) 
  (h : {a, b, c, d} = coefficients) :
  ∃ (x : ℝ), a * x^3 + b * x^2 + c * x + d = 0 := by sorry

end common_root_exists_polynomial_common_root_l962_96235


namespace train_length_l962_96232

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 72 → time = 12 → speed * time * (1000 / 3600) = 240 := by
  sorry

#check train_length

end train_length_l962_96232


namespace prob_same_color_is_34_100_l962_96237

/-- Represents an urn with balls of different colors -/
structure Urn :=
  (blue : ℕ)
  (red : ℕ)
  (green : ℕ)

/-- Calculates the total number of balls in an urn -/
def Urn.total (u : Urn) : ℕ := u.blue + u.red + u.green

/-- Calculates the probability of drawing a ball of a specific color from an urn -/
def prob_color (u : Urn) (color : ℕ) : ℚ :=
  color / u.total

/-- Calculates the probability of drawing balls of the same color from two urns -/
def prob_same_color (u1 u2 : Urn) : ℚ :=
  prob_color u1 u1.blue * prob_color u2 u2.blue +
  prob_color u1 u1.red * prob_color u2 u2.red +
  prob_color u1 u1.green * prob_color u2 u2.green

/-- The main theorem stating that the probability of drawing balls of the same color
    from the given urns is 0.34 -/
theorem prob_same_color_is_34_100 :
  let u1 : Urn := ⟨2, 3, 5⟩
  let u2 : Urn := ⟨4, 2, 4⟩
  prob_same_color u1 u2 = 34/100 := by
  sorry

end prob_same_color_is_34_100_l962_96237


namespace not_first_class_probability_l962_96266

theorem not_first_class_probability 
  (P_A P_B P_C : ℝ) 
  (h_A : P_A = 0.65) 
  (h_B : P_B = 0.2) 
  (h_C : P_C = 0.1) :
  1 - P_A = 0.35 := by
  sorry

end not_first_class_probability_l962_96266


namespace remaining_work_time_for_x_l962_96288

-- Define the work rates and work durations
def x_rate : ℚ := 1 / 30
def y_rate : ℚ := 1 / 15
def z_rate : ℚ := 1 / 20
def y_work_days : ℕ := 10
def z_work_days : ℕ := 5

-- Define the theorem
theorem remaining_work_time_for_x :
  let total_work : ℚ := 1
  let work_done_by_y : ℚ := y_rate * y_work_days
  let work_done_by_z : ℚ := z_rate * z_work_days
  let remaining_work : ℚ := total_work - (work_done_by_y + work_done_by_z)
  remaining_work / x_rate = 5 / 2 := by sorry

end remaining_work_time_for_x_l962_96288


namespace math_books_count_l962_96203

theorem math_books_count (total_books : ℕ) (math_cost history_cost total_price : ℚ)
  (h1 : total_books = 80)
  (h2 : math_cost = 4)
  (h3 : history_cost = 5)
  (h4 : total_price = 390) :
  ∃ (math_books : ℕ),
    math_books * math_cost + (total_books - math_books) * history_cost = total_price ∧
    math_books = 10 := by
  sorry

end math_books_count_l962_96203


namespace sqrt_sum_quotient_l962_96247

theorem sqrt_sum_quotient : (Real.sqrt 112 + Real.sqrt 567) / Real.sqrt 175 = 13 / 5 := by
  sorry

end sqrt_sum_quotient_l962_96247


namespace polynomial_subtraction_l962_96227

theorem polynomial_subtraction (x : ℝ) :
  (2 * x^6 + 3 * x^5 + x^4 - x^2 + 15) - (x^6 + x^5 - 2 * x^4 + x^3 + 5) =
  x^6 + 2 * x^5 + 3 * x^4 - x^3 + x^2 + 10 := by
  sorry

end polynomial_subtraction_l962_96227


namespace sum_condition_iff_divisible_l962_96267

/-- An arithmetic progression with first term a and common difference d. -/
structure ArithmeticProgression (α : Type*) [Ring α] where
  a : α
  d : α

/-- The nth term of an arithmetic progression. -/
def ArithmeticProgression.nthTerm {α : Type*} [Ring α] (ap : ArithmeticProgression α) (n : ℕ) : α :=
  ap.a + n • ap.d

/-- Condition for the sum of two terms to be another term in the progression. -/
def SumCondition {α : Type*} [Ring α] (ap : ArithmeticProgression α) : Prop :=
  ∀ n k : ℕ, ∃ p : ℕ, ap.nthTerm n + ap.nthTerm k = ap.nthTerm p

/-- Theorem: The sum condition holds if and only if the first term is divisible by the common difference. -/
theorem sum_condition_iff_divisible {α : Type*} [CommRing α] (ap : ArithmeticProgression α) :
    SumCondition ap ↔ ∃ m : α, ap.a = m * ap.d :=
  sorry

end sum_condition_iff_divisible_l962_96267


namespace inequality_proof_l962_96230

theorem inequality_proof (a b c : ℕ+) (h : c ≥ b) :
  (a ^ b.val) * ((a + b) ^ c.val) > (c ^ b.val) * (a ^ c.val) := by
  sorry

end inequality_proof_l962_96230


namespace quadratic_roots_for_negative_k_l962_96257

theorem quadratic_roots_for_negative_k (k : ℝ) (h : k < 0) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + x₁ + k - 1 = 0 ∧ x₂^2 + x₂ + k - 1 = 0 :=
by
  sorry

end quadratic_roots_for_negative_k_l962_96257


namespace tic_tac_toe_4x4_carl_wins_l962_96298

/-- Represents a 4x4 tic-tac-toe board --/
def Board := Fin 4 → Fin 4 → Option Bool

/-- Represents a winning line on the board --/
structure WinningLine :=
  (positions : List (Fin 4 × Fin 4))
  (is_valid : positions.length = 4)

/-- All possible winning lines on a 4x4 board --/
def winningLines : List WinningLine := sorry

/-- Checks if a given board configuration is valid --/
def isValidBoard (b : Board) : Prop := sorry

/-- Checks if Carl wins with exactly 4 O's --/
def carlWinsWithFourO (b : Board) : Prop := sorry

/-- The number of ways Carl can win with exactly 4 O's --/
def numWaysToWin : ℕ := sorry

theorem tic_tac_toe_4x4_carl_wins :
  numWaysToWin = 4950 := by sorry

end tic_tac_toe_4x4_carl_wins_l962_96298


namespace square_not_partitionable_into_10deg_isosceles_triangles_l962_96268

-- Define a square
def Square : Type := Unit

-- Define an isosceles triangle with a 10° vertex angle
def IsoscelesTriangle10Deg : Type := Unit

-- Define a partition of a square
def Partition (s : Square) : Type := List IsoscelesTriangle10Deg

-- Theorem statement
theorem square_not_partitionable_into_10deg_isosceles_triangles :
  ¬∃ (s : Square) (p : Partition s), p.length > 0 := by
  sorry

end square_not_partitionable_into_10deg_isosceles_triangles_l962_96268


namespace chicken_burger_price_proof_l962_96234

/-- The cost of a chicken burger in won -/
def chicken_burger_cost : ℕ := 3350

/-- The cost of a bulgogi burger in won -/
def bulgogi_burger_cost : ℕ := chicken_burger_cost + 300

/-- The total cost of three bulgogi burgers and three chicken burgers in won -/
def total_cost : ℕ := 21000

theorem chicken_burger_price_proof :
  chicken_burger_cost = 3350 ∧
  bulgogi_burger_cost = chicken_burger_cost + 300 ∧
  3 * chicken_burger_cost + 3 * bulgogi_burger_cost = total_cost :=
by sorry

end chicken_burger_price_proof_l962_96234


namespace arithmetic_sequence_sum_relation_l962_96282

/-- An arithmetic sequence is a sequence where the difference between consecutive terms is constant. -/
structure ArithmeticSequence where
  -- The sequence itself
  a : ℕ → ℝ
  -- The constant difference between consecutive terms
  d : ℝ
  -- The property that defines an arithmetic sequence
  is_arithmetic : ∀ n, a (n + 1) = a n + d

/-- The sum of the first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) : ℕ → ℝ
  | 0 => 0
  | n + 1 => S seq n + seq.a (n + 1)

/-- The main theorem about the relationship between S_3n, S_2n, and S_n -/
theorem arithmetic_sequence_sum_relation (seq : ArithmeticSequence) :
  ∀ n, S seq (3 * n) = 3 * (S seq (2 * n) - S seq n) := by
  sorry


end arithmetic_sequence_sum_relation_l962_96282


namespace ball_peak_time_l962_96245

/-- Given a ball thrown upwards, this theorem proves that with an initial velocity of 1.25 m/s, 
    it takes 6.25 seconds to reach its peak height. -/
theorem ball_peak_time (v : ℝ) (t : ℝ) :
  v = 1.25 → t = 4 * v^2 → t = 6.25 := by
  sorry

end ball_peak_time_l962_96245


namespace floor_expression_equals_eight_l962_96294

theorem floor_expression_equals_eight :
  ⌊(3005^3 : ℝ) / (3003 * 3004) - (3003^3 : ℝ) / (3004 * 3005)⌋ = 8 := by
  sorry

end floor_expression_equals_eight_l962_96294


namespace movie_ticket_revenue_l962_96210

/-- Calculates the total revenue from movie ticket sales --/
theorem movie_ticket_revenue
  (adult_price : ℕ)
  (child_price : ℕ)
  (total_tickets : ℕ)
  (child_tickets : ℕ)
  (h1 : adult_price = 7)
  (h2 : child_price = 4)
  (h3 : total_tickets = 900)
  (h4 : child_tickets = 400) :
  adult_price * (total_tickets - child_tickets) + child_price * child_tickets = 5100 :=
by sorry

end movie_ticket_revenue_l962_96210


namespace arccos_one_half_eq_pi_third_l962_96264

theorem arccos_one_half_eq_pi_third : Real.arccos (1/2) = π/3 := by
  sorry

end arccos_one_half_eq_pi_third_l962_96264


namespace min_value_expression_l962_96242

theorem min_value_expression (x y : ℝ) : 
  x^2 - 2*x*y + y^2 + 2*y + 1 ≥ 0 ∧ 
  ∃ (a b : ℝ), a^2 - 2*a*b + b^2 + 2*b + 1 = 0 := by
sorry

end min_value_expression_l962_96242


namespace calculator_game_sum_l962_96249

/-- Represents the state of the three calculators -/
structure CalculatorState where
  first : ℕ
  second : ℕ
  third : ℤ

/-- Applies the operations to the calculator state -/
def applyOperations (state : CalculatorState) : CalculatorState :=
  { first := state.first ^ 2,
    second := state.second ^ 3,
    third := -state.third }

/-- Applies the operations n times to the initial state -/
def nOperations (n : ℕ) : CalculatorState :=
  match n with
  | 0 => { first := 2, second := 1, third := -2 }
  | n + 1 => applyOperations (nOperations n)

theorem calculator_game_sum (N : ℕ) :
  ∃ (n : ℕ), n > 0 ∧ 
    let finalState := nOperations 50
    finalState.first = N ∧ 
    finalState.second = 1 ∧ 
    finalState.third = -2 ∧
    (finalState.first : ℤ) + finalState.second + finalState.third = N - 1 :=
  sorry

end calculator_game_sum_l962_96249


namespace grass_seed_price_five_pound_bag_price_l962_96252

/-- Represents the price of a bag of grass seed -/
structure BagPrice where
  weight : ℕ
  price : ℚ

/-- Represents the customer's purchase -/
structure Purchase where
  bags5lb : ℕ
  bags10lb : ℕ
  bags25lb : ℕ

def total_weight (p : Purchase) : ℕ :=
  5 * p.bags5lb + 10 * p.bags10lb + 25 * p.bags25lb

def total_cost (p : Purchase) (price5lb : ℚ) : ℚ :=
  price5lb * p.bags5lb + 20.42 * p.bags10lb + 32.25 * p.bags25lb

def is_valid_purchase (p : Purchase) : Prop :=
  65 ≤ total_weight p ∧ total_weight p ≤ 80

theorem grass_seed_price (price5lb : ℚ) : Prop :=
  ∃ (p : Purchase),
    is_valid_purchase p ∧
    total_cost p price5lb = 98.77 ∧
    ∀ (q : Purchase), is_valid_purchase q → total_cost q price5lb ≥ 98.77 →
    price5lb = 2.02

/-- The main theorem stating that the price of the 5-pound bag is $2.02 -/
theorem five_pound_bag_price : ∃ (price5lb : ℚ), grass_seed_price price5lb :=
  sorry

end grass_seed_price_five_pound_bag_price_l962_96252


namespace boat_journey_distance_l962_96240

/-- Calculates the total distance covered by a man rowing a boat in a river with varying currents. -/
theorem boat_journey_distance
  (man_speed : ℝ)
  (current1_speed : ℝ)
  (current1_time : ℝ)
  (current2_speed : ℝ)
  (current2_time : ℝ)
  (h1 : man_speed = 15)
  (h2 : current1_speed = 2.5)
  (h3 : current1_time = 2)
  (h4 : current2_speed = 3)
  (h5 : current2_time = 1.5) :
  (man_speed + current1_speed) * current1_time +
  (man_speed - current2_speed) * current2_time = 53 := by
sorry


end boat_journey_distance_l962_96240


namespace min_value_sum_fractions_l962_96238

theorem min_value_sum_fractions (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b + 1) / c + (a + c + 1) / b + (b + c + 1) / a ≥ 9 ∧
  ∃ (a₀ b₀ c₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ 0 < c₀ ∧
    (a₀ + b₀ + 1) / c₀ + (a₀ + c₀ + 1) / b₀ + (b₀ + c₀ + 1) / a₀ = 9 :=
by sorry

end min_value_sum_fractions_l962_96238


namespace parallel_vectors_l962_96206

theorem parallel_vectors (x : ℝ) : 
  (∃ k : ℝ, k ≠ 0 ∧ k • (1, x) = (x - 1, 2)) → x = 1 ∨ x = 2 :=
sorry

end parallel_vectors_l962_96206


namespace hyperbola_equation_l962_96244

/-- Given a hyperbola with the equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    left and right foci F₁ and F₂, and a point P(3,4) on its asymptote,
    prove that if |PF₁ + PF₂| = |F₁F₂|, then the equation of the hyperbola is x²/9 - y²/16 = 1 -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (F₁ F₂ : ℝ × ℝ) (hF : F₁.1 < F₂.1)
  (P : ℝ × ℝ) (hP : P = (3, 4))
  (h_asymptote : ∃ (k : ℝ), P.2 = k * P.1 ∧ k^2 * a^2 = b^2)
  (h_foci : |P - F₁ + (P - F₂)| = |F₂ - F₁|) :
  ∀ (x y : ℝ), x^2 / 9 - y^2 / 16 = 1 ↔ x^2 / a^2 - y^2 / b^2 = 1 :=
by sorry

end hyperbola_equation_l962_96244


namespace constant_t_equation_l962_96248

theorem constant_t_equation : ∃! t : ℝ, 
  ∀ x : ℝ, (2*x^2 - 3*x + 4)*(5*x^2 + t*x + 9) = 10*x^4 - t^2*x^3 + 23*x^2 - 27*x + 36 ∧ t = -5 := by
  sorry

end constant_t_equation_l962_96248


namespace square_root_divided_by_18_l962_96254

theorem square_root_divided_by_18 : Real.sqrt 5184 / 18 = 4 := by sorry

end square_root_divided_by_18_l962_96254


namespace fraction_difference_equals_one_l962_96270

theorem fraction_difference_equals_one (x y : ℝ) (h : x * y = x - y) (h_nonzero : x * y ≠ 0) :
  1 / y - 1 / x = 1 := by
  sorry

end fraction_difference_equals_one_l962_96270


namespace jacket_markup_percentage_l962_96213

theorem jacket_markup_percentage 
  (purchase_price : ℝ)
  (selling_price : ℝ)
  (markup_percentage : ℝ)
  (discount_rate : ℝ)
  (gross_profit : ℝ)
  (h1 : purchase_price = 56)
  (h2 : selling_price = purchase_price + markup_percentage * selling_price)
  (h3 : discount_rate = 0.2)
  (h4 : gross_profit = 8)
  (h5 : gross_profit = (1 - discount_rate) * selling_price - purchase_price) :
  markup_percentage = 0.3 := by
sorry

end jacket_markup_percentage_l962_96213


namespace fraction_sum_equals_decimal_l962_96291

theorem fraction_sum_equals_decimal : 
  (4 : ℚ) / 100 - 8 / 10 + 3 / 1000 + 2 / 10000 = -0.7568 := by
  sorry

end fraction_sum_equals_decimal_l962_96291


namespace equilateral_triangle_vertex_l962_96276

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if three points form an equilateral triangle -/
def isEquilateral (a b c : Point) : Prop :=
  (a.x - b.x)^2 + (a.y - b.y)^2 = (b.x - c.x)^2 + (b.y - c.y)^2 ∧
  (b.x - c.x)^2 + (b.y - c.y)^2 = (c.x - a.x)^2 + (c.y - a.y)^2

/-- Checks if a point is on the altitude from another point to a line segment -/
def isOnAltitude (a d : Point) (b c : Point) : Prop :=
  (d.x - b.x) * (c.x - b.x) + (d.y - b.y) * (c.y - b.y) = 0 ∧
  (a.x - d.x) * (c.x - b.x) + (a.y - d.y) * (c.y - b.y) = 0

theorem equilateral_triangle_vertex (a b d : Point) : 
  a = Point.mk 10 4 →
  b = Point.mk 1 (-5) →
  d = Point.mk 0 (-2) →
  ∃ c : Point, 
    isEquilateral a b c ∧ 
    isOnAltitude a d b c ∧ 
    c = Point.mk (-1) 1 := by
  sorry

end equilateral_triangle_vertex_l962_96276


namespace volume_between_spheres_l962_96262

theorem volume_between_spheres (π : ℝ) (h : π > 0) :
  let volume_sphere (r : ℝ) := (4 / 3) * π * r^3
  (volume_sphere 10 - volume_sphere 4) = (3744 / 3) * π :=
by
  sorry

end volume_between_spheres_l962_96262


namespace intersection_sum_l962_96278

/-- Given two functions f and g where
    f(x) = -|x-a| + b
    g(x) = |x-c| + d
    that intersect at points (2,5) and (8,3),
    prove that a + c = 10 -/
theorem intersection_sum (a b c d : ℝ) :
  (∀ x, -|x - a| + b = |x - c| + d ↔ (x = 2 ∧ -|x - a| + b = 5) ∨ (x = 8 ∧ -|x - a| + b = 3)) →
  a + c = 10 :=
sorry

end intersection_sum_l962_96278


namespace f_is_quadratic_l962_96214

/-- Definition of a quadratic function -/
def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function y = 2x² -/
def f (x : ℝ) : ℝ := 2 * x^2

/-- Theorem: f is a quadratic function -/
theorem f_is_quadratic : is_quadratic f := by
  sorry

end f_is_quadratic_l962_96214


namespace rectangle_dimensions_l962_96226

/-- A rectangle with perimeter 40 and area 96 has dimensions (12, 8) or (8, 12) -/
theorem rectangle_dimensions : 
  ∀ a b : ℝ, 
  (2 * a + 2 * b = 40) →  -- perimeter condition
  (a * b = 96) →          -- area condition
  ((a = 12 ∧ b = 8) ∨ (a = 8 ∧ b = 12)) :=
by sorry

end rectangle_dimensions_l962_96226


namespace emily_marbles_l962_96250

theorem emily_marbles (E : ℕ) : 
  (3 * E - (3 * E / 2 + 1) = 8) → E = 6 := by
  sorry

end emily_marbles_l962_96250


namespace local_max_value_l962_96261

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x

-- State the theorem
theorem local_max_value (a : ℝ) :
  (∃ (x : ℝ), x = 2 ∧ IsLocalMin (f a) x) →
  (∃ (y : ℝ), IsLocalMax (f a) y ∧ f a y = 16) :=
by sorry

end local_max_value_l962_96261


namespace largest_integer_l962_96283

theorem largest_integer (a b c d : ℤ) 
  (sum1 : a + b + c = 163)
  (sum2 : a + b + d = 178)
  (sum3 : a + c + d = 184)
  (sum4 : b + c + d = 194) :
  max a (max b (max c d)) = 77 := by
  sorry

end largest_integer_l962_96283


namespace cost_for_23_days_l962_96284

/-- Calculates the total cost of staying in a student youth hostel for a given number of days. -/
def hostelCost (days : ℕ) : ℚ :=
  let firstWeekRate : ℚ := 18
  let additionalWeekRate : ℚ := 13
  let firstWeekDays : ℕ := min days 7
  let additionalDays : ℕ := days - firstWeekDays
  firstWeekRate * firstWeekDays + additionalWeekRate * additionalDays

/-- Proves that the cost of staying for 23 days in the student youth hostel is $334.00. -/
theorem cost_for_23_days : hostelCost 23 = 334 := by
  sorry

end cost_for_23_days_l962_96284


namespace eighteen_to_binary_l962_96216

def decimal_to_binary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 2) ((m % 2) :: acc)
    aux n []

def binary_to_decimal (l : List ℕ) : ℕ :=
  l.foldl (fun acc d => 2 * acc + d) 0

theorem eighteen_to_binary :
  decimal_to_binary 18 = [1, 0, 0, 1, 0] ∧
  binary_to_decimal [1, 0, 0, 1, 0] = 18 :=
sorry

end eighteen_to_binary_l962_96216


namespace mary_bike_rental_cost_l962_96221

/-- Calculates the total cost of bike rental given the fixed fee, hourly rate, and duration. -/
def bikeRentalCost (fixedFee : ℕ) (hourlyRate : ℕ) (duration : ℕ) : ℕ :=
  fixedFee + hourlyRate * duration

/-- Theorem stating that the bike rental cost for Mary is $80 -/
theorem mary_bike_rental_cost :
  bikeRentalCost 17 7 9 = 80 := by
  sorry

end mary_bike_rental_cost_l962_96221


namespace product_and_multiple_l962_96236

theorem product_and_multiple : ∃ (ε : ℝ) (x : ℝ), 
  (ε > 0 ∧ ε < 1) ∧ 
  (abs (198 * 2 - 400) < ε) ∧ 
  (2 * x = 56) ∧ 
  (9 * x = 252) := by
  sorry

end product_and_multiple_l962_96236


namespace smallest_sum_of_a_and_b_l962_96253

theorem smallest_sum_of_a_and_b (a b : ℝ) : 
  a > 0 → b > 0 → 
  ((3 * a)^2 ≥ 16 * b) → 
  ((4 * b)^2 ≥ 12 * a) → 
  a + b ≥ 70/3 :=
by
  sorry

end smallest_sum_of_a_and_b_l962_96253


namespace isosceles_triangle_base_length_l962_96274

/-- An isosceles triangle with two sides of length 8 cm and perimeter 25 cm has a base of length 9 cm. -/
theorem isosceles_triangle_base_length 
  (a b c : ℝ) 
  (h_isosceles : a = b) 
  (h_congruent_sides : a = 8) 
  (h_perimeter : a + b + c = 25) : 
  c = 9 := by
sorry

end isosceles_triangle_base_length_l962_96274


namespace product_sequence_l962_96217

theorem product_sequence (seq : List ℕ) : 
  (∀ i, i + 3 < seq.length → seq[i]! * seq[i+1]! * seq[i+2]! * seq[i+3]! = 120) →
  (∃ i j k, i < j ∧ j < k ∧ k < seq.length ∧ seq[i]! = 2 ∧ seq[j]! = 4 ∧ seq[k]! = 3) →
  (∃ x, x ∈ seq ∧ x = 5) :=
by sorry

end product_sequence_l962_96217


namespace calculation_result_l962_96256

/-- The smallest two-digit prime number -/
def smallest_two_digit_prime : ℕ := 11

/-- The largest one-digit prime number -/
def largest_one_digit_prime : ℕ := 7

/-- The smallest one-digit prime number -/
def smallest_one_digit_prime : ℕ := 2

/-- Theorem stating the result of the calculation -/
theorem calculation_result :
  smallest_two_digit_prime * (largest_one_digit_prime ^ 2) - smallest_one_digit_prime = 537 := by
  sorry


end calculation_result_l962_96256


namespace farmland_and_spray_theorem_l962_96281

/-- Represents the farmland areas and drone spraying capacities in two zones -/
structure FarmlandData where
  zone_a : ℝ  -- Farmland area in Zone A
  zone_b : ℝ  -- Farmland area in Zone B
  spray_a : ℝ  -- Average area sprayed per sortie in Zone A

/-- The conditions given in the problem -/
def problem_conditions (data : FarmlandData) : Prop :=
  data.zone_a = data.zone_b + 10000 ∧  -- Zone A has 10,000 mu more farmland
  0.8 * data.zone_a = data.zone_b ∧  -- 80% of Zone A equals all of Zone B (suitable area)
  (data.zone_b / data.spray_a) * 1.2 = data.zone_b / (data.spray_a - 50/3)  -- Drone sortie relationship

/-- The theorem to be proved -/
theorem farmland_and_spray_theorem (data : FarmlandData) :
  problem_conditions data →
  data.zone_a = 50000 ∧ data.zone_b = 40000 ∧ data.spray_a = 100 := by
  sorry

end farmland_and_spray_theorem_l962_96281


namespace quadratic_roots_sum_product_l962_96260

theorem quadratic_roots_sum_product (a b : ℝ) : 
  a^2 + a - 1 = 0 → b^2 + b - 1 = 0 → a ≠ b → ab + a + b = -2 := by
  sorry

end quadratic_roots_sum_product_l962_96260


namespace cubic_equation_root_l962_96251

theorem cubic_equation_root (a b : ℚ) : 
  (2 - 3 * Real.sqrt 3) ^ 3 + a * (2 - 3 * Real.sqrt 3) ^ 2 + b * (2 - 3 * Real.sqrt 3) - 37 = 0 →
  a = -55/23 := by
sorry

end cubic_equation_root_l962_96251


namespace simplify_and_evaluate_expression_l962_96208

theorem simplify_and_evaluate_expression (m : ℝ) (h : m = Real.sqrt 2 - 3) :
  (1 - 3 / (m + 3)) / (m / (m^2 + 6*m + 9)) = Real.sqrt 2 := by
  sorry

end simplify_and_evaluate_expression_l962_96208


namespace janna_weekly_sleep_l962_96223

/-- Represents the number of hours Janna sleeps in a week. -/
def weekly_sleep_hours (weekday_sleep : ℕ) (weekend_sleep : ℕ) : ℕ :=
  5 * weekday_sleep + 2 * weekend_sleep

/-- Proves that Janna sleeps 51 hours in a week. -/
theorem janna_weekly_sleep :
  weekly_sleep_hours 7 8 = 51 :=
by sorry

end janna_weekly_sleep_l962_96223


namespace arithmetic_sequence_fourth_term_l962_96222

/-- Given an arithmetic sequence with the first three terms x-y, x+y, and x/y,
    the fourth term is (10 - (2√13)/3) / (√13 - 1) -/
theorem arithmetic_sequence_fourth_term (x y : ℝ) (h : x ≠ 0) :
  let a₁ : ℝ := x - y
  let a₂ : ℝ := x + y
  let a₃ : ℝ := x / y
  let d : ℝ := a₂ - a₁
  let a₄ : ℝ := a₃ + d
  a₄ = (10 - (2 * Real.sqrt 13) / 3) / (Real.sqrt 13 - 1) := by
sorry

end arithmetic_sequence_fourth_term_l962_96222


namespace regular_hexagon_perimeter_l962_96272

/-- The perimeter of a regular hexagon with radius √3 is 6√3 -/
theorem regular_hexagon_perimeter (r : ℝ) (h : r = Real.sqrt 3) : 
  6 * r = 6 * Real.sqrt 3 := by
  sorry

end regular_hexagon_perimeter_l962_96272


namespace chess_tournament_theorem_l962_96219

/-- A chess tournament with the given conditions -/
structure ChessTournament where
  num_players : ℕ
  games_per_player : ℕ
  losses_per_player : ℕ
  no_ties : Bool

/-- Calculates the total number of games in the tournament -/
def total_games (t : ChessTournament) : ℕ :=
  t.num_players * (t.num_players - 1) / 2

/-- Calculates the number of wins for each player -/
def wins_per_player (t : ChessTournament) : ℕ :=
  t.games_per_player - t.losses_per_player

/-- The theorem to be proved -/
theorem chess_tournament_theorem (t : ChessTournament) 
  (h1 : t.num_players = 200)
  (h2 : t.games_per_player = 199)
  (h3 : t.losses_per_player = 30)
  (h4 : t.no_ties = true) :
  total_games t = 19900 ∧ wins_per_player t = 169 := by
  sorry


end chess_tournament_theorem_l962_96219


namespace order_of_abc_l962_96211

noncomputable def a : ℝ := 2 + Real.sqrt 3
noncomputable def b : ℝ := 1 + Real.sqrt 6
noncomputable def c : ℝ := Real.sqrt 2 + Real.sqrt 5

theorem order_of_abc : a > c ∧ c > b := by sorry

end order_of_abc_l962_96211
