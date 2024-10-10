import Mathlib

namespace rectangle_length_equal_square_side_l202_20257

/-- The length of a rectangle with width 3 cm and area equal to a 3 cm square -/
theorem rectangle_length_equal_square_side : 
  ∀ (length : ℝ), 
  (3 : ℝ) * length = (3 : ℝ) * (3 : ℝ) → 
  length = (3 : ℝ) := by
  sorry

end rectangle_length_equal_square_side_l202_20257


namespace office_age_problem_l202_20238

theorem office_age_problem (total_persons : Nat) (avg_age_all : Nat) (group1_size : Nat) 
  (group1_avg_age : Nat) (group2_size : Nat) (group2_avg_age : Nat) :
  total_persons = 19 →
  avg_age_all = 15 →
  group1_size = 5 →
  group1_avg_age = 14 →
  group2_size = 9 →
  group2_avg_age = 16 →
  (total_persons * avg_age_all) - (group1_size * group1_avg_age + group2_size * group2_avg_age) = 71 := by
  sorry

#check office_age_problem

end office_age_problem_l202_20238


namespace bracket_removal_equality_l202_20243

theorem bracket_removal_equality (a b c : ℝ) : a - 2*(b - c) = a - 2*b + 2*c := by
  sorry

end bracket_removal_equality_l202_20243


namespace arithmetic_sequence_sum_l202_20262

-- Define the arithmetic sequence
def arithmeticSequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Define the theorem
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (d : ℝ) :
  arithmeticSequence a d →
  d > 0 →
  a 1 + a 2 + a 3 = 15 →
  a 1 * a 2 * a 3 = 80 →
  a 11 + a 12 + a 13 = 105 := by
  sorry


end arithmetic_sequence_sum_l202_20262


namespace salem_poem_words_l202_20294

/-- Calculate the total number of words in a poem given the number of stanzas, lines per stanza, and words per line -/
def totalWords (stanzas : ℕ) (linesPerStanza : ℕ) (wordsPerLine : ℕ) : ℕ :=
  stanzas * linesPerStanza * wordsPerLine

/-- Theorem: The total number of words in Salem's poem is 1600 -/
theorem salem_poem_words :
  totalWords 20 10 8 = 1600 := by
  sorry

end salem_poem_words_l202_20294


namespace a_works_friday_50th_week_l202_20223

/-- Represents the days of the week -/
inductive Day
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents the people working night shifts -/
inductive Person
  | A
  | B
  | C
  | D
  | E
  | F

/-- Returns the next day in the week -/
def nextDay (d : Day) : Day :=
  match d with
  | Day.Sunday => Day.Monday
  | Day.Monday => Day.Tuesday
  | Day.Tuesday => Day.Wednesday
  | Day.Wednesday => Day.Thursday
  | Day.Thursday => Day.Friday
  | Day.Friday => Day.Saturday
  | Day.Saturday => Day.Sunday

/-- Returns the next person in the rotation -/
def nextPerson (p : Person) : Person :=
  match p with
  | Person.A => Person.B
  | Person.B => Person.C
  | Person.C => Person.D
  | Person.D => Person.E
  | Person.E => Person.F
  | Person.F => Person.A

/-- Returns the person working on a given day number -/
def personOnDay (dayNumber : Nat) : Person :=
  match dayNumber % 6 with
  | 0 => Person.F
  | 1 => Person.A
  | 2 => Person.B
  | 3 => Person.C
  | 4 => Person.D
  | 5 => Person.E
  | _ => Person.A  -- This case should never occur

/-- Returns the day of the week for a given day number -/
def dayOfWeek (dayNumber : Nat) : Day :=
  match dayNumber % 7 with
  | 0 => Day.Saturday
  | 1 => Day.Sunday
  | 2 => Day.Monday
  | 3 => Day.Tuesday
  | 4 => Day.Wednesday
  | 5 => Day.Thursday
  | 6 => Day.Friday
  | _ => Day.Sunday  -- This case should never occur

theorem a_works_friday_50th_week :
  personOnDay (50 * 7 - 2) = Person.A ∧ dayOfWeek (50 * 7 - 2) = Day.Friday :=
by sorry

end a_works_friday_50th_week_l202_20223


namespace circle_chords_l202_20234

theorem circle_chords (n : ℕ) (h : n = 10) : 
  (n.choose 2 : ℕ) = 45 := by
  sorry

end circle_chords_l202_20234


namespace quadratic_inequality_implies_m_bound_l202_20281

theorem quadratic_inequality_implies_m_bound (m : ℝ) :
  (∃ x : ℝ, x^2 - 2*x + m ≤ 0) → m ≤ 1 := by
  sorry

end quadratic_inequality_implies_m_bound_l202_20281


namespace intersection_of_A_and_B_l202_20273

def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}
def B : Set ℝ := {x : ℝ | -3 < x ∧ x ≤ 1}

theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | -1 < x ∧ x ≤ 1} := by sorry

end intersection_of_A_and_B_l202_20273


namespace trombone_players_count_l202_20260

/-- Represents the Oprah Winfrey High School marching band -/
structure MarchingBand where
  trumpet_weight : ℕ := 5
  clarinet_weight : ℕ := 5
  trombone_weight : ℕ := 10
  tuba_weight : ℕ := 20
  drum_weight : ℕ := 15
  trumpet_count : ℕ := 6
  clarinet_count : ℕ := 9
  tuba_count : ℕ := 3
  drum_count : ℕ := 2
  total_weight : ℕ := 245

/-- Calculates the number of trombone players in the marching band -/
def trombone_players (band : MarchingBand) : ℕ :=
  let other_weight := band.trumpet_weight * band.trumpet_count +
                      band.clarinet_weight * band.clarinet_count +
                      band.tuba_weight * band.tuba_count +
                      band.drum_weight * band.drum_count
  let trombone_total_weight := band.total_weight - other_weight
  trombone_total_weight / band.trombone_weight

/-- Theorem stating that the number of trombone players is 8 -/
theorem trombone_players_count (band : MarchingBand) : trombone_players band = 8 := by
  sorry

end trombone_players_count_l202_20260


namespace frogs_in_pond_a_l202_20286

theorem frogs_in_pond_a (frogs_b : ℕ) : 
  frogs_b + 2 * frogs_b = 48 → 2 * frogs_b = 32 :=
by
  sorry

end frogs_in_pond_a_l202_20286


namespace expression_minimum_l202_20268

theorem expression_minimum (x : ℝ) (h : 1 < x ∧ x < 5) : 
  ∃ (y : ℝ), y = (x^2 - 4*x + 5) / (2*x - 6) ∧ 
  (∀ (z : ℝ), 1 < z ∧ z < 5 → (z^2 - 4*z + 5) / (2*z - 6) ≥ y) ∧
  y = 1 := by
  sorry

end expression_minimum_l202_20268


namespace smallest_y_for_perfect_square_l202_20206

def x : ℕ := 11 * 36 * 54

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem smallest_y_for_perfect_square : 
  (∃ y : ℕ, y > 0 ∧ is_perfect_square (x * y)) ∧ 
  (∀ z : ℕ, z > 0 ∧ z < 66 → ¬is_perfect_square (x * z)) ∧
  is_perfect_square (x * 66) :=
sorry

end smallest_y_for_perfect_square_l202_20206


namespace sarah_homework_problem_l202_20289

theorem sarah_homework_problem (math_pages : ℕ) (reading_pages : ℕ) (problems_per_page : ℕ) :
  math_pages = 4 →
  reading_pages = 6 →
  problems_per_page = 4 →
  (math_pages + reading_pages) * problems_per_page = 40 :=
by sorry

end sarah_homework_problem_l202_20289


namespace count_non_negative_numbers_l202_20213

theorem count_non_negative_numbers : 
  let numbers : List ℝ := [-8, 2.1, 1/9, 3, 0, -2.5, 10, -1]
  (numbers.filter (λ x => x ≥ 0)).length = 5 := by
sorry

end count_non_negative_numbers_l202_20213


namespace allocation_methods_count_l202_20271

def number_of_doctors : ℕ := 3
def number_of_nurses : ℕ := 6
def number_of_schools : ℕ := 3
def doctors_per_school : ℕ := 1
def nurses_per_school : ℕ := 2

theorem allocation_methods_count :
  (Nat.choose number_of_doctors doctors_per_school) *
  (Nat.choose number_of_nurses nurses_per_school) *
  (Nat.choose (number_of_doctors - doctors_per_school) doctors_per_school) *
  (Nat.choose (number_of_nurses - nurses_per_school) nurses_per_school) = 540 := by
  sorry

end allocation_methods_count_l202_20271


namespace muffin_price_theorem_l202_20208

/-- The price per muffin to raise the required amount -/
def price_per_muffin (total_amount : ℚ) (num_cases : ℕ) (packs_per_case : ℕ) (muffins_per_pack : ℕ) : ℚ :=
  total_amount / (num_cases * packs_per_case * muffins_per_pack)

/-- Theorem: The price per muffin to raise $120 by selling 5 cases of muffins, 
    where each case contains 3 packs and each pack contains 4 muffins, is $2 -/
theorem muffin_price_theorem :
  price_per_muffin 120 5 3 4 = 2 := by
  sorry

end muffin_price_theorem_l202_20208


namespace cone_lateral_surface_area_l202_20245

theorem cone_lateral_surface_area 
  (r : ℝ) (l : ℝ) (h1 : r = 3) (h2 : l = 10) : 
  π * r * l = 30 * π := by
  sorry

end cone_lateral_surface_area_l202_20245


namespace sharks_winning_percentage_l202_20226

theorem sharks_winning_percentage (N : ℕ) : 
  (∀ k : ℕ, k < N → (1 + k : ℚ) / (4 + k) < 9 / 10) ∧
  (1 + N : ℚ) / (4 + N) ≥ 9 / 10 →
  N = 26 :=
sorry

end sharks_winning_percentage_l202_20226


namespace min_value_expression_l202_20239

theorem min_value_expression (x y : ℝ) : (x*y)^2 + (x + 7)^2 + (2*y + 7)^2 ≥ 45 := by
  sorry

end min_value_expression_l202_20239


namespace adjacency_probability_correct_l202_20298

/-- The probability of A being adjacent to both B and C in a random lineup of 4 people --/
def adjacency_probability : ℚ := 1 / 6

/-- The total number of people in the group --/
def total_people : ℕ := 4

/-- The number of ways to arrange ABC as a unit with the fourth person --/
def favorable_arrangements : ℕ := 4

/-- The total number of possible arrangements of 4 people --/
def total_arrangements : ℕ := 24

theorem adjacency_probability_correct :
  adjacency_probability = (favorable_arrangements : ℚ) / total_arrangements := by
  sorry

end adjacency_probability_correct_l202_20298


namespace train_speed_problem_l202_20256

theorem train_speed_problem (x : ℝ) (V : ℝ) (h1 : x > 0) (h2 : V > 0) :
  (3 * x) / ((x / V) + ((2 * x) / 20)) = 25 →
  V = 50 := by
sorry

end train_speed_problem_l202_20256


namespace polynomial_division_remainder_l202_20201

theorem polynomial_division_remainder (k : ℚ) : 
  (∃! k, ∀ x, (3 * x^3 + k * x^2 + 5 * x - 8) % (3 * x + 4) = 10) ↔ k = 31/4 :=
by sorry

end polynomial_division_remainder_l202_20201


namespace nephews_count_l202_20221

/-- The number of nephews Alden and Vihaan have together -/
def total_nephews (alden_past : ℕ) (alden_ratio : ℕ) (vihaan_diff : ℕ) : ℕ :=
  let alden_current := alden_past * alden_ratio
  let vihaan_current := alden_current + vihaan_diff
  alden_current + vihaan_current

/-- Proof that Alden and Vihaan have 600 nephews together -/
theorem nephews_count : total_nephews 80 3 120 = 600 := by
  sorry

end nephews_count_l202_20221


namespace inequality_proof_l202_20247

theorem inequality_proof (a b c d e f : ℕ) 
  (h1 : (a : ℚ) / b > (c : ℚ) / d)
  (h2 : (c : ℚ) / d > (e : ℚ) / f)
  (h3 : a * f - b * e = 1) :
  d ≥ b + f := by
sorry

end inequality_proof_l202_20247


namespace tangent_line_and_a_range_l202_20249

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + x + 1

-- Define the derivative of f(x)
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + 1

theorem tangent_line_and_a_range (a : ℝ) :
  -- Condition: Tangent line at (1, f(1)) is parallel to 2x - y + 1 = 0
  (f_derivative a 1 = 2) →
  -- Condition: f(x) is decreasing on the interval [-2/3, -1/3]
  (∀ x ∈ Set.Icc (-2/3) (-1/3), f_derivative a x ≤ 0) →
  -- Conclusion 1: Equations of tangent lines passing through (0, 1)
  ((∃ x₀ y₀ : ℝ, f a x₀ = y₀ ∧ f_derivative a x₀ = (y₀ - 1) / x₀ ∧
    ((y₀ = 1 ∧ x₀ = 0) ∨ (y₀ = 11/8 ∧ x₀ = 1/2))) ∧
   (∀ x y : ℝ, (y = x + 1) ∨ (y = 3/4 * x + 1))) ∧
  -- Conclusion 2: Range of a
  (a ≥ 2) :=
by sorry

end tangent_line_and_a_range_l202_20249


namespace some_number_value_l202_20246

theorem some_number_value (x : ℝ) : 40 + x * 12 / (180 / 3) = 41 → x = 5 := by
  sorry

end some_number_value_l202_20246


namespace fish_added_calculation_james_added_eight_fish_l202_20219

theorem fish_added_calculation (initial_fish : ℕ) (fish_eaten_per_day : ℕ) 
  (days_before_adding : ℕ) (days_after_adding : ℕ) (final_fish : ℕ) : ℕ :=
  let total_days := days_before_adding + days_after_adding
  let total_fish_eaten := total_days * fish_eaten_per_day
  let expected_remaining := initial_fish - total_fish_eaten
  final_fish - expected_remaining
  
-- The main theorem
theorem james_added_eight_fish : 
  fish_added_calculation 60 2 14 7 26 = 8 := by
sorry

end fish_added_calculation_james_added_eight_fish_l202_20219


namespace euler_line_l202_20259

/-- The centroid of a triangle -/
def centroid (A B C : ℝ × ℝ) : ℝ × ℝ := sorry

/-- The orthocenter of a triangle -/
def orthocenter (A B C : ℝ × ℝ) : ℝ × ℝ := sorry

/-- The circumcenter of a triangle -/
def circumcenter (A B C : ℝ × ℝ) : ℝ × ℝ := sorry

/-- Three points are collinear -/
def collinear (P Q R : ℝ × ℝ) : Prop := sorry

theorem euler_line (A B C : ℝ × ℝ) : 
  collinear (centroid A B C) (orthocenter A B C) (circumcenter A B C) := by
  sorry

end euler_line_l202_20259


namespace fifth_runner_speed_doubling_l202_20285

-- Define the total time and individual runner times
variable (T : ℝ) -- Total time
variable (T1 T2 T3 T4 T5 : ℝ) -- Individual runner times

-- Define the conditions from the problem
axiom total_time : T1 + T2 + T3 + T4 + T5 = T
axiom first_runner : T1 / 2 + T2 + T3 + T4 + T5 = 0.95 * T
axiom second_runner : T1 + T2 / 2 + T3 + T4 + T5 = 0.9 * T
axiom third_runner : T1 + T2 + T3 / 2 + T4 + T5 = 0.88 * T
axiom fourth_runner : T1 + T2 + T3 + T4 / 2 + T5 = 0.85 * T

-- The theorem to prove
theorem fifth_runner_speed_doubling (h1 : T > 0) :
  T1 + T2 + T3 + T4 + T5 / 2 = 0.92 * T := by sorry

end fifth_runner_speed_doubling_l202_20285


namespace reciprocal_of_sum_l202_20287

theorem reciprocal_of_sum : (1 / (1/3 + 1/5) : ℚ) = 15/8 := by sorry

end reciprocal_of_sum_l202_20287


namespace square_sum_theorem_l202_20231

theorem square_sum_theorem (x y : ℝ) (h1 : x + y = -10) (h2 : x = 25 / y) : x^2 + y^2 = 50 := by
  sorry

end square_sum_theorem_l202_20231


namespace hyperbola_eccentricity_l202_20233

theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_intersect : ∃ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ∧ y = 2*x) :
  let e := Real.sqrt (1 + (b/a)^2)
  e > Real.sqrt 5 := by sorry

end hyperbola_eccentricity_l202_20233


namespace triangle_side_length_l202_20279

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the length function
def length (p q : ℝ × ℝ) : ℝ := sorry

-- Define the median function
def median (t : Triangle) (side : ℝ × ℝ) : ℝ := sorry

theorem triangle_side_length (t : Triangle) :
  -- Conditions
  (median t t.B = (1/3) * length t.B t.C) →
  (length t.A t.B = 3) →
  (length t.A t.C = 2) →
  -- Conclusion
  length t.B t.C = 3 * Real.sqrt 2 := by sorry

end triangle_side_length_l202_20279


namespace distinct_naturals_reciprocal_sum_l202_20282

theorem distinct_naturals_reciprocal_sum (x y z : ℕ) : 
  x ≠ y ∧ y ≠ z ∧ x ≠ z →  -- distinct
  0 < x ∧ 0 < y ∧ 0 < z →  -- natural numbers
  x < y ∧ y < z →  -- ascending order
  (∃ (n : ℕ), (1 : ℚ) / x + (1 : ℚ) / y + (1 : ℚ) / z = n) →  -- sum is a natural number
  x = 2 ∧ y = 3 ∧ z = 6 :=
by sorry

end distinct_naturals_reciprocal_sum_l202_20282


namespace unwashed_shirts_l202_20203

theorem unwashed_shirts 
  (short_sleeve : ℕ) 
  (long_sleeve : ℕ) 
  (washed : ℕ) 
  (h1 : short_sleeve = 9)
  (h2 : long_sleeve = 27)
  (h3 : washed = 20) : 
  short_sleeve + long_sleeve - washed = 16 := by
  sorry

end unwashed_shirts_l202_20203


namespace modulus_of_complex_fourth_power_l202_20267

theorem modulus_of_complex_fourth_power : 
  Complex.abs ((2 : ℂ) + (3 * Real.sqrt 2) * Complex.I) ^ 4 = 484 := by sorry

end modulus_of_complex_fourth_power_l202_20267


namespace fourth_power_representation_l202_20211

/-- For any base N ≥ 6, (N-1)^4 in base N can be represented as (N-4)5(N-4)1 -/
theorem fourth_power_representation (N : ℕ) (h : N ≥ 6) :
  ∃ (a b c d : ℕ), (N - 1)^4 = a * N^3 + b * N^2 + c * N + d ∧
                    a = N - 4 ∧
                    b = 5 ∧
                    c = N - 4 ∧
                    d = 1 :=
by sorry

end fourth_power_representation_l202_20211


namespace stick_markings_l202_20209

theorem stick_markings (stick_length : ℝ) (red_mark : ℝ) (blue_mark : ℝ) : 
  stick_length = 12 →
  red_mark = stick_length / 2 →
  blue_mark = red_mark / 2 →
  red_mark - blue_mark = 3 := by
sorry

end stick_markings_l202_20209


namespace invalid_votes_percentage_l202_20261

theorem invalid_votes_percentage
  (total_votes : ℕ)
  (candidate_a_percentage : ℚ)
  (candidate_a_votes : ℕ)
  (h1 : total_votes = 560000)
  (h2 : candidate_a_percentage = 80 / 100)
  (h3 : candidate_a_votes = 380800) :
  (total_votes - (candidate_a_votes / candidate_a_percentage)) / total_votes = 15 / 100 := by
sorry

end invalid_votes_percentage_l202_20261


namespace road_with_ten_trees_length_l202_20242

/-- The length of a road with trees planted at equal intervals -/
def road_length (num_trees : ℕ) (interval : ℝ) : ℝ :=
  (num_trees - 1 : ℝ) * interval

/-- Theorem: The length of a road with 10 trees planted at 10-meter intervals is 90 meters -/
theorem road_with_ten_trees_length :
  road_length 10 10 = 90 := by
  sorry

#eval road_length 10 10

end road_with_ten_trees_length_l202_20242


namespace even_number_property_l202_20232

/-- Sum of digits function -/
def sum_of_digits : ℕ → ℕ := sorry

/-- Theorem: If the sum of digits of N is 100 and the sum of digits of 5N is 50, then N is even -/
theorem even_number_property (N : ℕ) 
  (h1 : sum_of_digits N = 100) 
  (h2 : sum_of_digits (5 * N) = 50) : 
  Even N := by sorry

end even_number_property_l202_20232


namespace song_listeners_l202_20276

theorem song_listeners (total group_size : ℕ) (book_readers : ℕ) (both_listeners : ℕ) : 
  group_size = 100 → book_readers = 50 → both_listeners = 20 → 
  ∃ song_listeners : ℕ, song_listeners = 70 ∧ 
    group_size = book_readers + song_listeners - both_listeners :=
by sorry

end song_listeners_l202_20276


namespace standard_deviation_transformation_l202_20277

-- Define a sample data type
def SampleData := Fin 10 → ℝ

-- Define standard deviation for a sample
noncomputable def standardDeviation (data : SampleData) : ℝ := sorry

-- Define the transformation function
def transform (x : ℝ) : ℝ := 2 * x - 1

-- Main theorem
theorem standard_deviation_transformation (data : SampleData) :
  standardDeviation data = 8 →
  standardDeviation (fun i => transform (data i)) = 16 := by
  sorry

end standard_deviation_transformation_l202_20277


namespace hyperbola_symmetry_l202_20205

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola x² - y² = a² -/
def Hyperbola (a : ℝ) : Set Point :=
  {p : Point | p.x^2 - p.y^2 = a^2}

/-- Represents the line y = x - 2 -/
def SymmetryLine : Set Point :=
  {p : Point | p.y = p.x - 2}

/-- Represents the line 2x + 3y = 6 -/
def TangentLine : Set Point :=
  {p : Point | 2 * p.x + 3 * p.y = 6}

/-- Defines symmetry about a line -/
def SymmetricPoint (p : Point) : Point :=
  ⟨p.y + 2, p.x - 2⟩

/-- Defines the curve C₂ symmetric to C₁ about the symmetry line -/
def C₂ (a : ℝ) : Set Point :=
  {p : Point | SymmetricPoint p ∈ Hyperbola a}

/-- States that the tangent line is tangent to C₂ -/
def IsTangent (a : ℝ) : Prop :=
  ∃ p : Point, p ∈ C₂ a ∧ p ∈ TangentLine

theorem hyperbola_symmetry (a : ℝ) (h : a > 0) (h_tangent : IsTangent a) : 
  a = 8 * Real.sqrt 5 / 5 := by
  sorry

end hyperbola_symmetry_l202_20205


namespace range_of_x_l202_20210

theorem range_of_x (a b x : ℝ) (h_a : a ≠ 0) :
  (∀ a b, |a + b| + |a - b| ≥ |a| * (|x - 1| + |x - 2|)) →
  x ∈ Set.Icc (1/2 : ℝ) (5/2 : ℝ) := by
  sorry

end range_of_x_l202_20210


namespace conditional_probability_haze_wind_l202_20230

theorem conditional_probability_haze_wind (P_haze P_wind P_both : ℝ) 
  (h1 : P_haze = 0.25)
  (h2 : P_wind = 0.4)
  (h3 : P_both = 0.02) :
  P_both / P_haze = 0.08 :=
by sorry

end conditional_probability_haze_wind_l202_20230


namespace trigonometric_equation_solution_l202_20200

theorem trigonometric_equation_solution :
  ∀ t : ℝ, 
    (2 * (Real.cos (2 * t))^6 - (Real.cos (2 * t))^4 + 1.5 * (Real.sin (4 * t))^2 - 3 * (Real.sin (2 * t))^2 = 0) ↔ 
    (∃ k : ℤ, t = (Real.pi / 8) * (2 * ↑k + 1)) :=
by sorry

end trigonometric_equation_solution_l202_20200


namespace total_money_of_three_people_l202_20269

/-- Given three people A, B, and C with some money between them, prove that their total amount is 400. -/
theorem total_money_of_three_people (a b c : ℕ) : 
  a + c = 300 →
  b + c = 150 →
  c = 50 →
  a + b + c = 400 := by
sorry

end total_money_of_three_people_l202_20269


namespace least_with_four_prime_factors_l202_20255

/-- A function that returns the number of prime factors (counting multiplicity) of a positive integer -/
def num_prime_factors (n : ℕ+) : ℕ := sorry

/-- The property that both n and n+1 have exactly four prime factors -/
def has_four_prime_factors (n : ℕ+) : Prop :=
  num_prime_factors n = 4 ∧ num_prime_factors (n + 1) = 4

theorem least_with_four_prime_factors :
  ∀ n : ℕ+, n < 1155 → ¬(has_four_prime_factors n) ∧ has_four_prime_factors 1155 := by
  sorry

end least_with_four_prime_factors_l202_20255


namespace red_balls_count_l202_20248

theorem red_balls_count (total : ℕ) (red : ℕ) (h1 : total = 15) 
  (h2 : red ≤ total) 
  (h3 : (red * (red - 1)) / (total * (total - 1)) = 1 / 21) : 
  red = 4 := by
sorry

end red_balls_count_l202_20248


namespace flour_calculation_l202_20270

/-- The amount of flour originally called for in the recipe -/
def original_flour : ℝ := 7

/-- The extra amount of flour Mary added -/
def extra_flour : ℝ := 2

/-- The total amount of flour Mary used -/
def total_flour : ℝ := 9

/-- Theorem stating that the original amount of flour plus the extra amount equals the total amount -/
theorem flour_calculation : original_flour + extra_flour = total_flour := by
  sorry

end flour_calculation_l202_20270


namespace sugar_flour_ratio_l202_20215

theorem sugar_flour_ratio (flour baking_soda sugar : ℕ) : 
  (flour = 10 * baking_soda) →
  (flour = 8 * (baking_soda + 60)) →
  (sugar = 2000) →
  (sugar * 6 = flour * 5) :=
by sorry

end sugar_flour_ratio_l202_20215


namespace four_to_fourth_sum_l202_20244

theorem four_to_fourth_sum : (4 : ℕ) ^ 4 + (4 : ℕ) ^ 4 + (4 : ℕ) ^ 4 + (4 : ℕ) ^ 4 = (4 : ℕ) ^ 5 := by
  sorry

end four_to_fourth_sum_l202_20244


namespace roots_of_polynomial_l202_20258

theorem roots_of_polynomial (x : ℝ) : 
  x^3 - 3*x^2 - x + 3 = 0 ↔ x = 1 ∨ x = -1 ∨ x = 3 := by
  sorry

end roots_of_polynomial_l202_20258


namespace consecutive_integers_divisibility_l202_20204

theorem consecutive_integers_divisibility : ∃ (a b c : ℕ), 
  (a > 0) ∧ (b > 0) ∧ (c > 0) ∧  -- positive integers
  (b = a + 1) ∧ (c = b + 1) ∧    -- consecutive
  (a % 1 = 0) ∧                  -- a divisible by (b - a)^2
  (a % 4 = 0) ∧                  -- a divisible by (c - a)^2
  (b % 1 = 0) :=                 -- b divisible by (c - b)^2
by sorry

end consecutive_integers_divisibility_l202_20204


namespace unique_solution_quadratic_inequality_l202_20228

theorem unique_solution_quadratic_inequality (a : ℝ) :
  (∃! x : ℝ, 0 ≤ x^2 - a*x + a ∧ x^2 - a*x + a ≤ 1) → a = 2 := by
  sorry

end unique_solution_quadratic_inequality_l202_20228


namespace min_sum_squares_l202_20284

theorem min_sum_squares (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3*x*y*z = 8) :
  ∃ (m : ℝ), m = 3 ∧ (∀ (a b c : ℝ), a^3 + b^3 + c^3 - 3*a*b*c = 8 → a^2 + b^2 + c^2 ≥ m) :=
sorry

end min_sum_squares_l202_20284


namespace quotient_zeros_l202_20227

theorem quotient_zeros (x : ℚ) (h : x = 4227/1000) : 
  ¬ (∃ a b c : ℕ, x / 3 = (a : ℚ) + 1/10 + c/1000) :=
sorry

end quotient_zeros_l202_20227


namespace triangle_base_value_l202_20229

theorem triangle_base_value (square_perimeter : ℝ) (triangle_height : ℝ) (x : ℝ) :
  square_perimeter = 64 →
  triangle_height = 32 →
  (square_perimeter / 4) ^ 2 = (1 / 2) * x * triangle_height →
  x = 16 :=
by
  sorry

end triangle_base_value_l202_20229


namespace solve_exponential_equation_l202_20218

theorem solve_exponential_equation :
  ∃ t : ℝ, 4 * (4^t) + Real.sqrt (16 * (16^t)) = 32 ∧ t = 1 := by
  sorry

end solve_exponential_equation_l202_20218


namespace range_of_m_l202_20292

-- Define the functions f and g
def f (c : ℝ) (x : ℝ) : ℝ := x^2 - 4*x + c
def g (c m : ℝ) (x : ℝ) : ℝ := x * (f c x + m*x - 5)

-- State the theorem
theorem range_of_m (c : ℝ) :
  (∃! x, f c x = 0) →
  (∃ x₁ x₂, 2 < x₁ ∧ x₁ < x₂ ∧ x₂ < 3 ∧ g c m x₁ < g c m x₂ ∧ g c m x₂ > g c m x₁) →
  -1/3 < m ∧ m < 5/4 :=
by sorry

end range_of_m_l202_20292


namespace inequality_range_of_m_l202_20212

theorem inequality_range_of_m (m : ℝ) : 
  (∀ x : ℝ, (m^2 + 4*m - 5)*x^2 - 4*(m - 1)*x + 3 > 0) ↔ 
  (1 ≤ m ∧ m < 19) :=
sorry

end inequality_range_of_m_l202_20212


namespace cubes_not_touching_foil_l202_20220

/-- Represents the dimensions of a rectangular prism --/
structure PrismDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the number of cubes in a rectangular prism --/
def cubeCount (d : PrismDimensions) : ℕ :=
  d.length * d.width * d.height

/-- Theorem: The number of cubes not touching tin foil in the described prism is 128 --/
theorem cubes_not_touching_foil : ∃ (inner outer : PrismDimensions),
  -- The width of the foil-covered prism is 10 inches
  outer.width = 10 ∧
  -- The width of the inner figure is twice its length and height
  inner.width = 2 * inner.length ∧
  inner.width = 2 * inner.height ∧
  -- There is a 1-inch layer of cubes touching the foil on all sides
  outer.length = inner.length + 2 ∧
  outer.width = inner.width + 2 ∧
  outer.height = inner.height + 2 ∧
  -- The number of cubes not touching any tin foil is 128
  cubeCount inner = 128 := by
  sorry


end cubes_not_touching_foil_l202_20220


namespace factorization_1_factorization_2_simplification_and_evaluation_l202_20296

-- Question 1
theorem factorization_1 (m n : ℝ) : 
  m^2 * (n - 3) + 4 * (3 - n) = (n - 3) * (m + 2) * (m - 2) := by sorry

-- Question 2
theorem factorization_2 (p : ℝ) :
  (p - 3) * (p - 1) + 1 = (p - 2)^2 := by sorry

-- Question 3
theorem simplification_and_evaluation (x : ℝ) 
  (h : x^2 + x + 1/4 = 0) :
  ((2*x + 1) / (x + 1) + x - 1) / ((x + 2) / (x^2 + 2*x + 1)) = -1/4 := by sorry

end factorization_1_factorization_2_simplification_and_evaluation_l202_20296


namespace stock_price_increase_l202_20275

theorem stock_price_increase (initial_price : ℝ) (h : initial_price > 0) :
  let price_after_year1 := initial_price * 1.20
  let price_after_year2 := price_after_year1 * 0.75
  let price_after_year3 := initial_price * 1.035
  let increase_percentage := (price_after_year3 / price_after_year2 - 1) * 100
  increase_percentage = 15 := by
  sorry

end stock_price_increase_l202_20275


namespace melies_initial_money_l202_20283

/-- The amount of meat Méliès bought in kilograms -/
def meat_amount : ℝ := 2

/-- The cost of meat per kilogram in dollars -/
def meat_cost_per_kg : ℝ := 82

/-- The amount of money Méliès has left after paying for the meat in dollars -/
def money_left : ℝ := 16

/-- The initial amount of money in Méliès' wallet in dollars -/
def initial_money : ℝ := meat_amount * meat_cost_per_kg + money_left

theorem melies_initial_money :
  initial_money = 180 :=
sorry

end melies_initial_money_l202_20283


namespace smallest_consecutive_even_sum_162_l202_20217

theorem smallest_consecutive_even_sum_162 (n : ℤ) : 
  (∃ (a b c : ℤ), a = n ∧ b = n + 2 ∧ c = n + 4 ∧ a + b + c = 162) → n = 52 := by
sorry

end smallest_consecutive_even_sum_162_l202_20217


namespace carol_picked_29_carrots_l202_20252

/-- The number of carrots Carol picked -/
def carols_carrots (total_carrots good_carrots bad_carrots moms_carrots : ℕ) : ℕ :=
  total_carrots - moms_carrots

/-- Theorem stating that Carol picked 29 carrots -/
theorem carol_picked_29_carrots 
  (total_carrots : ℕ) 
  (good_carrots : ℕ) 
  (bad_carrots : ℕ) 
  (moms_carrots : ℕ) 
  (h1 : total_carrots = good_carrots + bad_carrots)
  (h2 : good_carrots = 38)
  (h3 : bad_carrots = 7)
  (h4 : moms_carrots = 16) :
  carols_carrots total_carrots good_carrots bad_carrots moms_carrots = 29 := by
  sorry

end carol_picked_29_carrots_l202_20252


namespace price_difference_l202_20299

theorem price_difference (P : ℝ) (h : P > 0) : 
  let P' := 1.25 * P
  (P' - P) / P' * 100 = 20 := by sorry

end price_difference_l202_20299


namespace ae_length_is_fifteen_l202_20290

/-- Represents a rectangle ABCD with a line EF dividing it into two equal areas -/
structure DividedRectangle where
  AB : ℝ
  AD : ℝ
  EB : ℝ
  EF : ℝ
  AE : ℝ
  area_AEFCD : ℝ
  area_EBCF : ℝ
  equal_areas : area_AEFCD = area_EBCF
  rectangle_area : AB * AD = area_AEFCD + area_EBCF

/-- The theorem stating that under given conditions, AE = 15 -/
theorem ae_length_is_fifteen (r : DividedRectangle)
  (h1 : r.EB = 40)
  (h2 : r.AD = 80)
  (h3 : r.EF = 30) :
  r.AE = 15 := by
  sorry

#check ae_length_is_fifteen

end ae_length_is_fifteen_l202_20290


namespace rectangle_ratio_in_square_config_l202_20266

-- Define the structure of our square-rectangle configuration
structure SquareRectConfig where
  inner_side : ℝ
  rect_short : ℝ
  rect_long : ℝ

-- State the theorem
theorem rectangle_ratio_in_square_config (config : SquareRectConfig) :
  -- The outer square's side is composed of the inner square's side and two short sides of rectangles
  config.inner_side + 2 * 2 * config.rect_short = 3 * config.inner_side →
  -- Two long sides and one short side of rectangles make up the outer square's side
  2 * config.rect_long + config.rect_short = 3 * config.inner_side →
  -- The ratio of long to short sides of the rectangle is 2.5
  config.rect_long / config.rect_short = 2.5 := by
  sorry

end rectangle_ratio_in_square_config_l202_20266


namespace circle_center_and_radius_l202_20295

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y - 4 = 0

-- State the theorem
theorem circle_center_and_radius :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (1, -2) ∧ 
    radius = 3 ∧
    ∀ (x y : ℝ), circle_equation x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
by sorry

end circle_center_and_radius_l202_20295


namespace wood_wasted_percentage_l202_20224

/-- The percentage of wood wasted when carving a cone from a sphere -/
theorem wood_wasted_percentage (sphere_radius cone_height cone_base_diameter : ℝ) :
  sphere_radius = 9 →
  cone_height = 9 →
  cone_base_diameter = 18 →
  let cone_base_radius := cone_base_diameter / 2
  let cone_volume := (1 / 3) * Real.pi * cone_base_radius^2 * cone_height
  let sphere_volume := (4 / 3) * Real.pi * sphere_radius^3
  let percentage_wasted := (cone_volume / sphere_volume) * 100
  percentage_wasted = 25 := by sorry

end wood_wasted_percentage_l202_20224


namespace half_of_1_01_l202_20241

theorem half_of_1_01 : (1.01 : ℝ) / 2 = 0.505 := by
  sorry

end half_of_1_01_l202_20241


namespace ellipse_eccentricity_l202_20235

/-- For an ellipse where the length of the major axis is twice its focal length, the eccentricity is 1/2. -/
theorem ellipse_eccentricity (a c : ℝ) (h : a = 2 * c) : c / a = 1 / 2 := by
  sorry

end ellipse_eccentricity_l202_20235


namespace total_profit_is_100_l202_20225

/-- Calculates the total profit given investments and A's profit share -/
def calculate_total_profit (a_investment : ℕ) (a_months : ℕ) (b_investment : ℕ) (b_months : ℕ) (a_profit_share : ℕ) : ℕ :=
  let a_investment_share := a_investment * a_months
  let b_investment_share := b_investment * b_months
  let total_investment_share := a_investment_share + b_investment_share
  let total_profit := a_profit_share * total_investment_share / a_investment_share
  total_profit

/-- Theorem stating that given the specified investments and A's profit share, the total profit is 100 -/
theorem total_profit_is_100 :
  calculate_total_profit 300 12 200 6 75 = 100 := by
  sorry

end total_profit_is_100_l202_20225


namespace liquid_x_percentage_in_solution_b_l202_20240

/-- Given two solutions A and B, where liquid X makes up 0.8% of solution A,
    and a mixture of 300g of A and 700g of B results in a solution with 1.5% of liquid X,
    prove that liquid X makes up 1.8% of solution B. -/
theorem liquid_x_percentage_in_solution_b : 
  let percent_x_in_a : ℝ := 0.008
  let mass_a : ℝ := 300
  let mass_b : ℝ := 700
  let percent_x_in_mixture : ℝ := 0.015
  let percent_x_in_b : ℝ := (percent_x_in_mixture * (mass_a + mass_b) - percent_x_in_a * mass_a) / mass_b
  percent_x_in_b = 0.018 := by
  sorry

end liquid_x_percentage_in_solution_b_l202_20240


namespace fruit_bowl_oranges_l202_20253

theorem fruit_bowl_oranges :
  let bananas : ℕ := 7
  let apples : ℕ := 2 * bananas
  let pears : ℕ := 4
  let grapes : ℕ := apples / 2
  let total_fruits : ℕ := 40
  let oranges : ℕ := total_fruits - (bananas + apples + pears + grapes)
  oranges = 8 := by sorry

end fruit_bowl_oranges_l202_20253


namespace expression_evaluation_l202_20272

theorem expression_evaluation : 
  (Real.sqrt 3) / (Real.cos (10 * π / 180)) - 1 / (Real.sin (170 * π / 180)) = -4 := by sorry

end expression_evaluation_l202_20272


namespace perpendicular_line_correct_l202_20202

/-- A line in polar coordinates passing through (2, 0) and perpendicular to the polar axis --/
def perpendicular_line (ρ θ : ℝ) : Prop :=
  ρ * Real.cos θ = 2

theorem perpendicular_line_correct :
  ∀ ρ θ : ℝ, perpendicular_line ρ θ ↔ 
    (ρ * Real.cos θ = 2 ∧ ρ * Real.sin θ = 0) ∨
    (ρ = 2 ∧ θ = 0) :=
by sorry

end perpendicular_line_correct_l202_20202


namespace pentagonal_pyramid_edges_and_faces_l202_20293

/-- A pentagonal pyramid is a polyhedron with a pentagonal base and triangular lateral faces. -/
structure PentagonalPyramid where
  base_edges : ℕ
  lateral_edges : ℕ
  lateral_faces : ℕ
  base_faces : ℕ

/-- The properties of a pentagonal pyramid. -/
def pentagonal_pyramid : PentagonalPyramid :=
  { base_edges := 5
  , lateral_edges := 5
  , lateral_faces := 5
  , base_faces := 1
  }

/-- The number of edges in a pentagonal pyramid. -/
def num_edges (p : PentagonalPyramid) : ℕ := p.base_edges + p.lateral_edges

/-- The number of faces in a pentagonal pyramid. -/
def num_faces (p : PentagonalPyramid) : ℕ := p.lateral_faces + p.base_faces

theorem pentagonal_pyramid_edges_and_faces :
  num_edges pentagonal_pyramid = 10 ∧ num_faces pentagonal_pyramid = 6 := by
  sorry

end pentagonal_pyramid_edges_and_faces_l202_20293


namespace unique_solution_is_two_l202_20291

def cyclic_index (i n : ℕ) : ℕ := i % n + 1

theorem unique_solution_is_two (x : ℕ → ℕ) (n : ℕ) (hn : n = 20) :
  (∀ i, x i > 0) →
  (∀ i, x (cyclic_index (i + 2) n)^2 = Nat.lcm (x (cyclic_index (i + 1) n)) (x (cyclic_index i n)) + 
                                       Nat.lcm (x (cyclic_index i n)) (x (cyclic_index (i - 1) n))) →
  (∀ i, x i = 2) :=
by sorry

end unique_solution_is_two_l202_20291


namespace sharp_composition_50_l202_20254

-- Define the # operation
def sharp (N : ℝ) : ℝ := 0.5 * N + 1

-- Theorem statement
theorem sharp_composition_50 : sharp (sharp (sharp 50)) = 8 := by
  sorry

end sharp_composition_50_l202_20254


namespace andrena_christel_doll_difference_l202_20236

/-- Proves that Andrena has 2 more dolls than Christel after gift exchanges -/
theorem andrena_christel_doll_difference :
  -- Initial conditions
  ∀ (debelyn_initial christel_initial andrena_initial : ℕ),
  debelyn_initial = 20 →
  christel_initial = 24 →
  -- Gift exchanges
  ∀ (debelyn_to_andrena christel_to_andrena : ℕ),
  debelyn_to_andrena = 2 →
  christel_to_andrena = 5 →
  -- Final condition
  andrena_initial + debelyn_to_andrena + christel_to_andrena =
    debelyn_initial - debelyn_to_andrena + 3 →
  -- Conclusion
  (andrena_initial + debelyn_to_andrena + christel_to_andrena) -
    (christel_initial - christel_to_andrena) = 2 :=
by sorry

end andrena_christel_doll_difference_l202_20236


namespace theater_capacity_filled_l202_20288

theorem theater_capacity_filled (seats : ℕ) (ticket_price : ℕ) (performances : ℕ) (total_revenue : ℕ) :
  seats = 400 →
  ticket_price = 30 →
  performances = 3 →
  total_revenue = 28800 →
  (total_revenue / ticket_price) / (seats * performances) * 100 = 80 := by
  sorry

end theater_capacity_filled_l202_20288


namespace smaller_equals_larger_l202_20251

/-- A circle with an inscribed rectangle and a smaller rectangle -/
structure InscribedRectangles where
  /-- The radius of the circle -/
  r : ℝ
  /-- Half-width of the larger rectangle -/
  a : ℝ
  /-- Half-height of the larger rectangle -/
  b : ℝ
  /-- Proportion of the smaller rectangle's side to the larger rectangle's side -/
  x : ℝ
  /-- The larger rectangle is inscribed in the circle -/
  inscribed : r^2 = a^2 + b^2
  /-- The smaller rectangle has two vertices on the circle -/
  smaller_on_circle : r^2 = (a*x)^2 + (b*x)^2
  /-- The smaller rectangle's side coincides with the larger rectangle's side -/
  coincide : 0 < x ∧ x ≤ 1

/-- The area of the smaller rectangle is equal to the area of the larger rectangle -/
theorem smaller_equals_larger (ir : InscribedRectangles) : 
  (ir.a * ir.x) * (ir.b * ir.x) = ir.a * ir.b := by
  sorry

end smaller_equals_larger_l202_20251


namespace inequality_two_integer_solutions_l202_20216

def has_exactly_two_integer_solutions (a : ℝ) : Prop :=
  ∃ x y : ℤ, x ≠ y ∧
    (x : ℝ)^2 - (a + 1) * (x : ℝ) + a < 0 ∧
    (y : ℝ)^2 - (a + 1) * (y : ℝ) + a < 0 ∧
    ∀ z : ℤ, z ≠ x → z ≠ y → (z : ℝ)^2 - (a + 1) * (z : ℝ) + a ≥ 0

theorem inequality_two_integer_solutions :
  {a : ℝ | has_exactly_two_integer_solutions a} = {a : ℝ | (3 < a ∧ a ≤ 4) ∨ (-2 ≤ a ∧ a < -1)} :=
by sorry

end inequality_two_integer_solutions_l202_20216


namespace cakes_distribution_l202_20297

theorem cakes_distribution (total_cakes : ℕ) (friends : ℕ) (cakes_per_friend : ℕ) :
  total_cakes = 30 →
  friends = 2 →
  cakes_per_friend = total_cakes / friends →
  cakes_per_friend = 15 := by
sorry

end cakes_distribution_l202_20297


namespace sawz_logging_total_cost_l202_20265

/-- The total cost of trees for Sawz Logging Co. -/
theorem sawz_logging_total_cost :
  let total_trees : ℕ := 850
  let douglas_fir_trees : ℕ := 350
  let ponderosa_pine_trees : ℕ := total_trees - douglas_fir_trees
  let douglas_fir_cost : ℕ := 300
  let ponderosa_pine_cost : ℕ := 225
  let total_cost : ℕ := douglas_fir_trees * douglas_fir_cost + ponderosa_pine_trees * ponderosa_pine_cost
  total_cost = 217500 := by
  sorry

#check sawz_logging_total_cost

end sawz_logging_total_cost_l202_20265


namespace inequality_solution_range_l202_20264

theorem inequality_solution_range (a b x : ℝ) : 
  (a > 0 ∧ b > 0) → 
  (∀ a b, a > 0 → b > 0 → x^2 + 2*x < a/b + 16*b/a) ↔ 
  (-4 < x ∧ x < 2) := by
sorry

end inequality_solution_range_l202_20264


namespace simplify_fraction_l202_20263

theorem simplify_fraction : 5 * (21 / 8) * (32 / -63) = -20 / 3 := by sorry

end simplify_fraction_l202_20263


namespace total_flowers_ratio_l202_20222

/-- Represents the number of pots in the garden -/
def num_pots : ℕ := 350

/-- Represents the ratio of flowers to total items in a pot -/
def flower_ratio : ℚ := 3 / 5

/-- Represents the number of flowers in a single pot -/
def flowers_per_pot (total_items : ℕ) : ℚ := flower_ratio * total_items

theorem total_flowers_ratio (total_items_per_pot : ℕ) :
  (num_pots : ℚ) * flowers_per_pot total_items_per_pot = 
  flower_ratio * ((num_pots : ℚ) * total_items_per_pot) := by sorry

end total_flowers_ratio_l202_20222


namespace equation_solutions_l202_20207

-- Define the equations as functions
def eqnA (x : ℝ) := (3*x + 1)^2 = 0
def eqnB (x : ℝ) := |2*x + 1| - 6 = 0
def eqnC (x : ℝ) := Real.sqrt (5 - x) + 3 = 0
def eqnD (x : ℝ) := Real.sqrt (4*x + 9) - 7 = 0
def eqnE (x : ℝ) := |5*x - 3| + 2 = -1

-- Define the existence of solutions
def has_solution (f : ℝ → Prop) := ∃ x, f x

-- Theorem statement
theorem equation_solutions :
  (has_solution eqnA) ∧
  (has_solution eqnB) ∧
  (¬ has_solution eqnC) ∧
  (has_solution eqnD) ∧
  (¬ has_solution eqnE) :=
sorry

end equation_solutions_l202_20207


namespace min_reciprocal_sum_l202_20237

theorem min_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 3 * b = 1) :
  (1 / a + 1 / b) ≥ 4 + 2 * Real.sqrt 3 :=
sorry

end min_reciprocal_sum_l202_20237


namespace russia_canada_size_comparison_l202_20274

theorem russia_canada_size_comparison 
  (us canada russia : ℝ) 
  (h1 : canada = 1.5 * us) 
  (h2 : russia = 2 * us) : 
  (russia - canada) / canada = 1 / 3 := by
sorry

end russia_canada_size_comparison_l202_20274


namespace min_pours_to_half_l202_20278

def water_remaining (n : ℕ) : ℝ := (0.9 : ℝ) ^ n

theorem min_pours_to_half : 
  (∀ k < 7, water_remaining k ≥ 0.5) ∧ 
  (water_remaining 7 < 0.5) := by
sorry

end min_pours_to_half_l202_20278


namespace tan_beta_value_l202_20214

theorem tan_beta_value (α β : Real) 
  (h1 : Real.tan α = 1/3) 
  (h2 : Real.tan (α + β) = 1/2) : 
  Real.tan β = 1/7 := by
sorry

end tan_beta_value_l202_20214


namespace expression_simplification_l202_20280

theorem expression_simplification (x y z : ℝ) : 
  (x + (y - z)) - ((x + z) - y) = 2 * y - 2 * z := by
  sorry

end expression_simplification_l202_20280


namespace H_range_l202_20250

def H (x : ℝ) : ℝ := |x + 2| - |x - 3|

theorem H_range : ∀ y : ℝ, (∃ x : ℝ, H x = y) ↔ -5 ≤ y ∧ y ≤ 5 := by
  sorry

end H_range_l202_20250
