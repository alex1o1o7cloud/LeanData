import Mathlib

namespace NUMINAMATH_CALUDE_triangle_and_function_properties_l3323_332395

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    and vectors m and n that are parallel, prove the angle B and properties of function f. -/
theorem triangle_and_function_properties
  (a b c : ℝ)
  (A B C : ℝ)
  (m : ℝ × ℝ)
  (n : ℝ × ℝ)
  (ω : ℝ)
  (h_triangle : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π)
  (h_sides : a > 0 ∧ b > 0 ∧ c > 0)
  (h_m : m = (b, 2*a - c))
  (h_n : n = (Real.cos B, Real.cos C))
  (h_parallel : ∃ (k : ℝ), m = k • n)
  (h_ω : ω > 0)
  (f : ℝ → ℝ)
  (h_f : f = λ x => Real.cos (ω * x - π/6) + Real.sin (ω * x))
  (h_period : ∀ x, f (x + π) = f x) :
  (B = π/3) ∧
  (∃ x₀ ∈ Set.Icc 0 (π/2), ∀ x ∈ Set.Icc 0 (π/2), f x ≤ f x₀ ∧ f x₀ = Real.sqrt 3) ∧
  (∃ x₁ ∈ Set.Icc 0 (π/2), ∀ x ∈ Set.Icc 0 (π/2), f x₁ ≤ f x ∧ f x₁ = -Real.sqrt 3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_and_function_properties_l3323_332395


namespace NUMINAMATH_CALUDE_spring_properties_l3323_332321

-- Define the spring's properties
def initial_length : ℝ := 18
def extension_rate : ℝ := 2

-- Define the relationship between mass and length
def spring_length (mass : ℝ) : ℝ := initial_length + extension_rate * mass

theorem spring_properties :
  (spring_length 4 = 26) ∧
  (∀ x y, x < y → spring_length x < spring_length y) ∧
  (∀ x, spring_length x = 2 * x + 18) ∧
  (spring_length 12 = 42) := by
  sorry

end NUMINAMATH_CALUDE_spring_properties_l3323_332321


namespace NUMINAMATH_CALUDE_original_number_proof_l3323_332379

theorem original_number_proof : ∃ n : ℕ, n + 1 = 30 ∧ n < 30 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l3323_332379


namespace NUMINAMATH_CALUDE_golden_man_poem_analysis_correct_l3323_332351

/-- Represents a poem --/
structure Poem where
  content : String
  deriving Repr

/-- Represents the analysis of a poem --/
structure PoemAnalysis where
  sentimentality_reasons : List String
  artistic_techniques : List String
  deriving Repr

/-- Function to analyze a poem --/
def analyze_poem (p : Poem) : PoemAnalysis :=
  { sentimentality_reasons := ["humiliating mission", "decline of homeland", "aging"],
    artistic_techniques := ["using scenery to express emotions"] }

/-- The poem in question --/
def golden_man_poem : Poem :=
  { content := "Recalling the divine capital, a bustling place, where I once roamed..." }

/-- Theorem stating that the analysis of the golden_man_poem is correct --/
theorem golden_man_poem_analysis_correct :
  analyze_poem golden_man_poem =
    { sentimentality_reasons := ["humiliating mission", "decline of homeland", "aging"],
      artistic_techniques := ["using scenery to express emotions"] } := by
  sorry


end NUMINAMATH_CALUDE_golden_man_poem_analysis_correct_l3323_332351


namespace NUMINAMATH_CALUDE_total_coins_last_month_l3323_332385

/-- The number of coins Mathilde had at the start of this month -/
def mathilde_this_month : ℕ := 100

/-- The number of coins Salah had at the start of this month -/
def salah_this_month : ℕ := 100

/-- The percentage increase in Mathilde's coins from last month to this month -/
def mathilde_increase : ℚ := 25/100

/-- The percentage decrease in Salah's coins from last month to this month -/
def salah_decrease : ℚ := 20/100

/-- Theorem stating that the total number of coins Mathilde and Salah had at the start of last month was 205 -/
theorem total_coins_last_month : 
  ∃ (mathilde_last_month salah_last_month : ℕ),
    (mathilde_this_month : ℚ) = mathilde_last_month * (1 + mathilde_increase) ∧
    (salah_this_month : ℚ) = salah_last_month * (1 - salah_decrease) ∧
    mathilde_last_month + salah_last_month = 205 := by
  sorry

end NUMINAMATH_CALUDE_total_coins_last_month_l3323_332385


namespace NUMINAMATH_CALUDE_angle_expression_value_l3323_332377

theorem angle_expression_value (α : Real) (h : Real.tan α = 2) :
  (2 * Real.sin α - Real.cos α) / (Real.sin α + 2 * Real.cos α) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_angle_expression_value_l3323_332377


namespace NUMINAMATH_CALUDE_quadratic_radical_combination_l3323_332307

theorem quadratic_radical_combination (x : ℝ) : 
  (∃ y : ℝ, y^2 = 2*x - 1 ∧ y = Real.sqrt 3) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_radical_combination_l3323_332307


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3323_332398

theorem inequality_solution_set (a b : ℝ) : 
  (∀ x : ℝ, ax - b > 0 ↔ x < 1/3) → 
  (∀ x : ℝ, (a - b) * x - (a + b) > 0 ↔ x < 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3323_332398


namespace NUMINAMATH_CALUDE_dans_apples_l3323_332340

theorem dans_apples (benny_apples total_apples : ℕ) 
  (h1 : benny_apples = 2)
  (h2 : total_apples = 11) :
  total_apples - benny_apples = 9 :=
by sorry

end NUMINAMATH_CALUDE_dans_apples_l3323_332340


namespace NUMINAMATH_CALUDE_total_balls_count_l3323_332388

/-- The number of soccer balls -/
def soccer_balls : ℕ := 20

/-- The number of basketballs -/
def basketballs : ℕ := soccer_balls + 5

/-- The number of tennis balls -/
def tennis_balls : ℕ := 2 * soccer_balls

/-- The number of baseballs -/
def baseballs : ℕ := soccer_balls + 10

/-- The number of volleyballs -/
def volleyballs : ℕ := 30

/-- The total number of balls -/
def total_balls : ℕ := soccer_balls + basketballs + tennis_balls + baseballs + volleyballs

theorem total_balls_count : total_balls = 145 := by sorry

end NUMINAMATH_CALUDE_total_balls_count_l3323_332388


namespace NUMINAMATH_CALUDE_probability_at_least_two_like_gender_intention_relationship_l3323_332386

-- Define the survey data
def total_students : ℕ := 200
def male_like : ℕ := 60
def male_dislike : ℕ := 40
def female_like : ℕ := 80
def female_dislike : ℕ := 20

-- Define the probability of liking employment
def p_like : ℚ := (male_like + female_like) / total_students

-- Define the chi-square statistic
def chi_square : ℚ :=
  let n := total_students
  let a := male_like
  let b := male_dislike
  let c := female_like
  let d := female_dislike
  n * (a * d - b * c)^2 / ((a + b) * (c + d) * (a + c) * (b + d))

-- Define the critical value for α = 0.005
def critical_value : ℚ := 7879 / 1000

-- Theorem for the probability of selecting at least 2 out of 3 students who like employment
theorem probability_at_least_two_like :
  (3 * p_like^2 * (1 - p_like) + p_like^3) = 98 / 125 := by sorry

-- Theorem for the relationship between gender and intention to work in the patent agency
theorem gender_intention_relationship :
  chi_square > critical_value := by sorry

end NUMINAMATH_CALUDE_probability_at_least_two_like_gender_intention_relationship_l3323_332386


namespace NUMINAMATH_CALUDE_smallest_square_count_is_minimal_l3323_332324

/-- The smallest positive integer n such that n * (1² + 2² + 3²) is a perfect square,
    where n represents the number of squares of each size (1x1, 2x2, 3x3) needed to form a larger square. -/
def smallest_square_count : ℕ := 14

/-- Predicate to check if a number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k * k

/-- Theorem stating that smallest_square_count is the smallest positive integer satisfying the conditions -/
theorem smallest_square_count_is_minimal :
  (is_perfect_square (smallest_square_count * (1 * 1 + 2 * 2 + 3 * 3))) ∧
  (∀ m : ℕ, m > 0 ∧ m < smallest_square_count →
    ¬(is_perfect_square (m * (1 * 1 + 2 * 2 + 3 * 3)))) :=
sorry

end NUMINAMATH_CALUDE_smallest_square_count_is_minimal_l3323_332324


namespace NUMINAMATH_CALUDE_rationalize_and_minimize_sum_l3323_332310

theorem rationalize_and_minimize_sum : ∃ (A B C D : ℕ),
  (D > 0) ∧
  (∀ (p : ℕ), Prime p → ¬(p^2 ∣ B)) ∧
  ((A : ℝ) * Real.sqrt B + C) / D = (Real.sqrt 32) / (Real.sqrt 16 - Real.sqrt 2) ∧
  (∀ (A' B' C' D' : ℕ),
    (D' > 0) →
    (∀ (p : ℕ), Prime p → ¬(p^2 ∣ B')) →
    ((A' : ℝ) * Real.sqrt B' + C') / D' = (Real.sqrt 32) / (Real.sqrt 16 - Real.sqrt 2) →
    A + B + C + D ≤ A' + B' + C' + D') ∧
  A + B + C + D = 21 :=
by sorry

end NUMINAMATH_CALUDE_rationalize_and_minimize_sum_l3323_332310


namespace NUMINAMATH_CALUDE_parabola_and_tangent_circle_l3323_332394

noncomputable section

-- Define the parabola C
def parabola_C (x y : ℝ) : Prop := y^2 = 8*x

-- Define the directrix l
def directrix_l (x : ℝ) : Prop := x = -2

-- Define point P on the directrix
def point_P (t : ℝ) : ℝ × ℝ := (-2, 3*t - 1/t)

-- Define point Q on the y-axis
def point_Q (t : ℝ) : ℝ × ℝ := (0, 2*t)

-- Define the circle M
def circle_M (x y : ℝ) : Prop := (x-2)^2 + y^2 = 4

-- Define a line through two points
def line_through (p q : ℝ × ℝ) (x y : ℝ) : Prop :=
  (y - p.2) * (q.1 - p.1) = (x - p.1) * (q.2 - p.2)

-- Main theorem
theorem parabola_and_tangent_circle (t : ℝ) (ht : t ≠ 0) :
  (∀ x y, parabola_C x y ↔ y^2 = 8*x) ∧
  (∀ x y, line_through (point_P t) (point_Q t) x y →
    ∃ x0 y0, circle_M x0 y0 ∧
      ((x - x0)^2 + (y - y0)^2 = 4 ∧
       ((x - x0) * (x - x0) + (y - y0) * (y - y0) = 4))) :=
sorry

end NUMINAMATH_CALUDE_parabola_and_tangent_circle_l3323_332394


namespace NUMINAMATH_CALUDE_homework_duration_equation_l3323_332390

-- Define the variables
variable (a b x : ℝ)

-- Define the conditions
variable (h1 : a > 0)  -- Initial average daily homework duration is positive
variable (h2 : b > 0)  -- Current average weekly homework duration is positive
variable (h3 : 0 < x ∧ x < 1)  -- Rate of decrease is between 0 and 1

-- Theorem statement
theorem homework_duration_equation : a * (1 - x)^2 = b := by
  sorry

end NUMINAMATH_CALUDE_homework_duration_equation_l3323_332390


namespace NUMINAMATH_CALUDE_initial_geese_count_l3323_332305

/-- Given that 28 geese flew away and 23 geese remain in a field,
    prove that there were initially 51 geese in the field. -/
theorem initial_geese_count (flew_away : ℕ) (remaining : ℕ) : 
  flew_away = 28 → remaining = 23 → flew_away + remaining = 51 := by
  sorry

end NUMINAMATH_CALUDE_initial_geese_count_l3323_332305


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l3323_332397

theorem expression_simplification_and_evaluation :
  ∀ x : ℝ, x ≠ -1 →
  (2 * x / (x + 1) - (2 * x + 6) / (x^2 - 1) / ((x + 3) / (x^2 - 2 * x + 1))) = 2 / (x + 1) ∧
  (2 / (0 + 1) = 2) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l3323_332397


namespace NUMINAMATH_CALUDE_quadratic_root_range_l3323_332364

theorem quadratic_root_range (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ < -1 ∧ x₂ > 1 ∧ 
   x₁^2 + (m-1)*x₁ + m^2 - 2 = 0 ∧ 
   x₂^2 + (m-1)*x₂ + m^2 - 2 = 0) → 
  0 < m ∧ m < 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_range_l3323_332364


namespace NUMINAMATH_CALUDE_certain_number_equation_l3323_332382

theorem certain_number_equation : ∃ x : ℝ, 
  (5 * x - (2 * 1.4) / 1.3 = 4) ∧ 
  (abs (x - 1.23076923077) < 0.00000000001) := by
  sorry

end NUMINAMATH_CALUDE_certain_number_equation_l3323_332382


namespace NUMINAMATH_CALUDE_total_voters_l3323_332358

/-- Given information about voters in three districts, prove the total number of voters. -/
theorem total_voters (d1 d2 d3 : ℕ) : 
  d1 = 322 →
  d3 = 2 * d1 →
  d2 = d3 - 19 →
  d1 + d2 + d3 = 1591 := by
sorry

end NUMINAMATH_CALUDE_total_voters_l3323_332358


namespace NUMINAMATH_CALUDE_multiplicative_inverse_203_mod_397_l3323_332369

theorem multiplicative_inverse_203_mod_397 : ∃ x : ℤ, 0 ≤ x ∧ x < 397 ∧ (203 * x) % 397 = 1 :=
by
  use 309
  sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_203_mod_397_l3323_332369


namespace NUMINAMATH_CALUDE_range_of_a_l3323_332383

-- Define the propositions p and q
def p (x : ℝ) : Prop := x > 1 ∨ x < -3
def q (x a : ℝ) : Prop := x > a

-- Define the sufficient but not necessary condition
def sufficient_not_necessary (a : ℝ) : Prop :=
  (∀ x, q x a → p x) ∧ (∃ x, p x ∧ ¬q x a)

-- Theorem statement
theorem range_of_a (a : ℝ) :
  sufficient_not_necessary a → a ∈ Set.Ici 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3323_332383


namespace NUMINAMATH_CALUDE_max_reverse_digit_diff_l3323_332327

/-- Given two two-digit positive integers with the same digits in reverse order and
    their positive difference less than 60, the maximum difference is 54 -/
theorem max_reverse_digit_diff :
  ∀ q r : ℕ,
  10 ≤ q ∧ q < 100 →  -- q is a two-digit number
  10 ≤ r ∧ r < 100 →  -- r is a two-digit number
  ∃ a b : ℕ,
    0 ≤ a ∧ a ≤ 9 ∧   -- a is a digit
    0 ≤ b ∧ b ≤ 9 ∧   -- b is a digit
    q = 10 * a + b ∧  -- q's representation
    r = 10 * b + a ∧  -- r's representation
    (q > r → q - r < 60) ∧  -- positive difference less than 60
    (r > q → r - q < 60) →
  (∀ q' r' : ℕ,
    (∃ a' b' : ℕ,
      0 ≤ a' ∧ a' ≤ 9 ∧
      0 ≤ b' ∧ b' ≤ 9 ∧
      q' = 10 * a' + b' ∧
      r' = 10 * b' + a' ∧
      (q' > r' → q' - r' < 60) ∧
      (r' > q' → r' - q' < 60)) →
    q' - r' ≤ 54) ∧
  ∃ q₀ r₀ : ℕ, q₀ - r₀ = 54 ∧
    (∃ a₀ b₀ : ℕ,
      0 ≤ a₀ ∧ a₀ ≤ 9 ∧
      0 ≤ b₀ ∧ b₀ ≤ 9 ∧
      q₀ = 10 * a₀ + b₀ ∧
      r₀ = 10 * b₀ + a₀ ∧
      q₀ - r₀ < 60) :=
by sorry

end NUMINAMATH_CALUDE_max_reverse_digit_diff_l3323_332327


namespace NUMINAMATH_CALUDE_garden_perimeter_l3323_332306

/-- A rectangular garden with equal length and breadth of 150 meters has a perimeter of 600 meters. -/
theorem garden_perimeter :
  ∀ (length breadth : ℝ),
  length > 0 →
  breadth > 0 →
  (length = 150 ∧ breadth = 150) →
  4 * length = 600 := by
sorry

end NUMINAMATH_CALUDE_garden_perimeter_l3323_332306


namespace NUMINAMATH_CALUDE_multiply_powers_of_a_l3323_332319

theorem multiply_powers_of_a (a : ℝ) : -2 * a^3 * (3 * a^2) = -6 * a^5 := by
  sorry

end NUMINAMATH_CALUDE_multiply_powers_of_a_l3323_332319


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3323_332316

-- Define set A
def A : Set ℝ := {x | 2 < x ∧ x < 4}

-- Define set B
def B : Set ℝ := {x | (x - 1) * (x - 3) < 0}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {x | 2 < x ∧ x < 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3323_332316


namespace NUMINAMATH_CALUDE_square_b_minus_d_l3323_332332

theorem square_b_minus_d (a b c d : ℤ) 
  (eq1 : a - b - c + d = 18) 
  (eq2 : a + b - c - d = 6) : 
  (b - d)^2 = 36 := by
sorry

end NUMINAMATH_CALUDE_square_b_minus_d_l3323_332332


namespace NUMINAMATH_CALUDE_room_observation_ratio_l3323_332391

-- Define the room dimensions
def room_length : ℝ := 40
def room_width : ℝ := 40

-- Define the area observed by both guards
def area_observed_by_both : ℝ := 400

-- Define the total area of the room
def total_area : ℝ := room_length * room_width

-- Theorem to prove
theorem room_observation_ratio :
  total_area / area_observed_by_both = 4 := by
  sorry


end NUMINAMATH_CALUDE_room_observation_ratio_l3323_332391


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3323_332360

def A : Set ℝ := {x | |x| < 1}
def B : Set ℝ := {x | x^2 - 2*x ≤ 0}

theorem intersection_of_A_and_B : A ∩ B = {x | 0 ≤ x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3323_332360


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3323_332333

theorem quadratic_equation_solution : 
  ∀ x : ℝ, x^2 = 2024*x ↔ x = 0 ∨ x = 2024 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3323_332333


namespace NUMINAMATH_CALUDE_prob_second_good_is_five_ninths_l3323_332339

/-- Represents the number of good transistors initially in the box -/
def initial_good : ℕ := 6

/-- Represents the number of bad transistors initially in the box -/
def initial_bad : ℕ := 4

/-- Represents the total number of transistors initially in the box -/
def initial_total : ℕ := initial_good + initial_bad

/-- Represents the probability of selecting a good transistor as the second one,
    given that the first one selected was good -/
def prob_second_good : ℚ := (initial_good - 1) / (initial_total - 1)

theorem prob_second_good_is_five_ninths :
  prob_second_good = 5 / 9 := by sorry

end NUMINAMATH_CALUDE_prob_second_good_is_five_ninths_l3323_332339


namespace NUMINAMATH_CALUDE_elliptical_track_distance_l3323_332378

/-- Represents the properties of an elliptical track with two objects moving on it. -/
structure EllipticalTrack where
  /-- Half the circumference of the track in yards -/
  half_circumference : ℝ
  /-- Distance traveled by object B at first meeting in yards -/
  first_meeting_distance : ℝ
  /-- Distance object A is short of completing a lap at second meeting in yards -/
  second_meeting_shortfall : ℝ

/-- Theorem stating the total distance around the track given specific conditions -/
theorem elliptical_track_distance 
  (track : EllipticalTrack)
  (h1 : track.first_meeting_distance = 150)
  (h2 : track.second_meeting_shortfall = 90)
  (h3 : (track.first_meeting_distance) / (track.half_circumference - track.first_meeting_distance) = 
        (track.half_circumference + track.second_meeting_shortfall) / 
        (2 * track.half_circumference - track.second_meeting_shortfall)) :
  2 * track.half_circumference = 720 := by
  sorry


end NUMINAMATH_CALUDE_elliptical_track_distance_l3323_332378


namespace NUMINAMATH_CALUDE_grid_squares_count_l3323_332334

/-- Represents a square grid --/
structure Grid :=
  (size : Nat)

/-- Counts the number of squares of a given size in the grid --/
def countSquares (g : Grid) (squareSize : Nat) : Nat :=
  (g.size + 1 - squareSize) * (g.size + 1 - squareSize)

/-- Calculates the total number of squares in the grid --/
def totalSquares (g : Grid) : Nat :=
  (countSquares g 1) + (countSquares g 2) + (countSquares g 3) + (countSquares g 4)

/-- Theorem stating that the total number of squares in a 5x5 grid is 54 --/
theorem grid_squares_count :
  let g : Grid := ⟨5⟩
  totalSquares g = 54 := by
  sorry

end NUMINAMATH_CALUDE_grid_squares_count_l3323_332334


namespace NUMINAMATH_CALUDE_range_of_m_l3323_332393

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∃ x y : ℝ, x + y - m = 0 ∧ (x - 1)^2 + y^2 = 1

def q (m : ℝ) : Prop := ∃ x : ℝ, m * x^2 - 2 * x + 1 = 0

-- Define the theorem
theorem range_of_m :
  ∀ m : ℝ, (p m ∨ q m) ∧ ¬(¬(q m)) → m ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l3323_332393


namespace NUMINAMATH_CALUDE_remainder_of_product_div_12_l3323_332384

theorem remainder_of_product_div_12 : (1125 * 1127 * 1129) % 12 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_product_div_12_l3323_332384


namespace NUMINAMATH_CALUDE_max_working_no_family_singing_is_50_l3323_332366

/-- Represents the properties of people in the village -/
structure Village where
  total : ℕ
  notWorking : ℕ
  withFamilies : ℕ
  singingInShower : ℕ

/-- Calculates the maximum number of people who are working, don't have families, and sing in the shower -/
def maxWorkingNoFamilySinging (v : Village) : ℕ :=
  min (v.total - v.notWorking) (min (v.total - v.withFamilies) v.singingInShower)

/-- Theorem stating the maximum number of people who are working, don't have families, and sing in the shower -/
theorem max_working_no_family_singing_is_50 (v : Village) 
  (h1 : v.total = 100)
  (h2 : v.notWorking = 50)
  (h3 : v.withFamilies = 25)
  (h4 : v.singingInShower = 75) :
  maxWorkingNoFamilySinging v = 50 := by
  sorry

#eval maxWorkingNoFamilySinging ⟨100, 50, 25, 75⟩

end NUMINAMATH_CALUDE_max_working_no_family_singing_is_50_l3323_332366


namespace NUMINAMATH_CALUDE_count_special_quadrilaterals_l3323_332380

/-- A quadrilateral with specific properties -/
structure SpecialQuadrilateral where
  ab : ℕ+
  bc : ℕ+
  cd : ℕ+
  ad : ℕ+
  right_angle_b : True  -- Represents the right angle at B
  right_angle_c : True  -- Represents the right angle at C
  ab_eq_two : ab = 2
  cd_eq_ad : cd = ad

/-- The perimeter of a SpecialQuadrilateral -/
def perimeter (q : SpecialQuadrilateral) : ℕ :=
  q.ab + q.bc + q.cd + q.ad

/-- The theorem statement -/
theorem count_special_quadrilaterals :
  (∃ (s : Finset ℕ), s.card = 31 ∧
    (∀ p ∈ s, p < 2015 ∧ ∃ q : SpecialQuadrilateral, perimeter q = p) ∧
    (∀ p < 2015, (∃ q : SpecialQuadrilateral, perimeter q = p) → p ∈ s)) :=
sorry

end NUMINAMATH_CALUDE_count_special_quadrilaterals_l3323_332380


namespace NUMINAMATH_CALUDE_range_of_m_l3323_332375

theorem range_of_m (m : ℝ) : 
  (¬ (∃ m : ℝ, m + 1 ≤ 0) ∨ ¬ (∀ x : ℝ, x^2 + m*x + 1 > 0)) → 
  (m ≤ -2 ∨ m > -1) := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l3323_332375


namespace NUMINAMATH_CALUDE_vendor_throw_away_percent_l3323_332365

-- Define the initial number of apples (100 for simplicity)
def initial_apples : ℝ := 100

-- Define the percentage of apples sold on the first day
def first_day_sale_percent : ℝ := 30

-- Define the percentage of apples sold on the second day
def second_day_sale_percent : ℝ := 50

-- Define the total percentage of apples thrown away
def total_thrown_away_percent : ℝ := 42

-- Define the percentage of remaining apples thrown away on the first day
def first_day_throw_away_percent : ℝ := 20

theorem vendor_throw_away_percent :
  let remaining_after_first_sale := initial_apples * (1 - first_day_sale_percent / 100)
  let remaining_after_first_throw := remaining_after_first_sale * (1 - first_day_throw_away_percent / 100)
  let sold_second_day := remaining_after_first_throw * (second_day_sale_percent / 100)
  let thrown_away_second_day := remaining_after_first_throw - sold_second_day
  let total_thrown_away := (remaining_after_first_sale - remaining_after_first_throw) + thrown_away_second_day
  total_thrown_away = initial_apples * (total_thrown_away_percent / 100) :=
by sorry

end NUMINAMATH_CALUDE_vendor_throw_away_percent_l3323_332365


namespace NUMINAMATH_CALUDE_travel_fraction_proof_l3323_332348

def initial_amount : ℚ := 750
def clothes_fraction : ℚ := 1/3
def food_fraction : ℚ := 1/5
def final_amount : ℚ := 300

theorem travel_fraction_proof :
  let remaining_after_clothes := initial_amount * (1 - clothes_fraction)
  let remaining_after_food := remaining_after_clothes * (1 - food_fraction)
  let spent_on_travel := remaining_after_food - final_amount
  spent_on_travel / remaining_after_food = 1/4 := by sorry

end NUMINAMATH_CALUDE_travel_fraction_proof_l3323_332348


namespace NUMINAMATH_CALUDE_part_one_part_two_l3323_332320

/-- Condition p: (x - a)(x - 3a) < 0 -/
def p (x a : ℝ) : Prop := (x - a) * (x - 3 * a) < 0

/-- Condition q: (x - 3) / (x - 2) ≤ 0 -/
def q (x : ℝ) : Prop := (x - 3) / (x - 2) ≤ 0

/-- Part 1: When a = 1 and p ∧ q is true, then 2 < x < 3 -/
theorem part_one (x : ℝ) (h1 : p x 1) (h2 : q x) : 2 < x ∧ x < 3 := by
  sorry

/-- Part 2: When p is necessary but not sufficient for q, and a > 0, then 1 ≤ a ≤ 2 -/
theorem part_two (a : ℝ) (h1 : a > 0) 
  (h2 : ∀ x, q x → p x a) 
  (h3 : ∃ x, p x a ∧ ¬q x) : 1 ≤ a ∧ a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3323_332320


namespace NUMINAMATH_CALUDE_fifth_term_zero_l3323_332352

/-- An arithmetic sequence {a_n} -/
def arithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n - d

/-- The sequence is decreasing -/
def decreasingSequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) < a n

theorem fifth_term_zero
  (a : ℕ → ℝ)
  (h_arith : arithmeticSequence a)
  (h_decr : decreasingSequence a)
  (h_eq : a 1 ^ 2 = a 9 ^ 2) :
  a 5 = 0 :=
sorry

end NUMINAMATH_CALUDE_fifth_term_zero_l3323_332352


namespace NUMINAMATH_CALUDE_simplify_expression_l3323_332301

theorem simplify_expression :
  ∃ (d e f : ℕ+), 
    (∀ (p : ℕ), Prime p → ¬(p^2 ∣ f.val)) ∧
    (((Real.sqrt 3 - 1) ^ (2 - Real.sqrt 2)) / ((Real.sqrt 3 + 1) ^ (2 + Real.sqrt 2)) = d.val - e.val * Real.sqrt f.val) ∧
    (d.val = 14 ∧ e.val = 8 ∧ f.val = 3) :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l3323_332301


namespace NUMINAMATH_CALUDE_cube_sum_solutions_l3323_332361

def is_cube_sum (a b c : ℕ+) : Prop :=
  ∃ n : ℕ+, 2^(Nat.factorial a.val) + 2^(Nat.factorial b.val) + 2^(Nat.factorial c.val) = n^3

theorem cube_sum_solutions :
  ∀ a b c : ℕ+, is_cube_sum a b c ↔ 
    ((a, b, c) = (1, 1, 2) ∨ (a, b, c) = (1, 2, 1) ∨ (a, b, c) = (2, 1, 1)) :=
by sorry

end NUMINAMATH_CALUDE_cube_sum_solutions_l3323_332361


namespace NUMINAMATH_CALUDE_student_claim_incorrect_l3323_332345

theorem student_claim_incorrect (m n : ℤ) (hn : 0 < n ∧ n ≤ 100) :
  ¬ (167 * n ≤ 1000 * m ∧ 1000 * m < 168 * n) := by
  sorry

end NUMINAMATH_CALUDE_student_claim_incorrect_l3323_332345


namespace NUMINAMATH_CALUDE_sum_of_sequences_l3323_332341

theorem sum_of_sequences : (2+12+22+32+42) + (10+20+30+40+50) = 260 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_sequences_l3323_332341


namespace NUMINAMATH_CALUDE_f_derivative_at_zero_l3323_332389

noncomputable def f (x : ℝ) : ℝ := x * (x + 1) * (x + 2) * (x + 3) * (x + 4) * (x + 5) + 6

theorem f_derivative_at_zero : 
  deriv f 0 = 120 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_zero_l3323_332389


namespace NUMINAMATH_CALUDE_books_on_shelf_l3323_332371

theorem books_on_shelf (initial_books : ℕ) (added_books : ℕ) (removed_books : ℕ) 
  (h1 : initial_books = 38) 
  (h2 : added_books = 10) 
  (h3 : removed_books = 5) : 
  initial_books + added_books - removed_books = 43 := by
  sorry

end NUMINAMATH_CALUDE_books_on_shelf_l3323_332371


namespace NUMINAMATH_CALUDE_dice_probability_l3323_332312

def num_dice : ℕ := 5
def num_faces : ℕ := 12
def num_divisible_by_three : ℕ := 4  -- 3, 6, 9, 12 are divisible by 3

def prob_divisible_by_three : ℚ := num_divisible_by_three / num_faces
def prob_not_divisible_by_three : ℚ := 1 - prob_divisible_by_three

def exactly_three_divisible_probability : ℚ :=
  (Nat.choose num_dice 3 : ℚ) * 
  (prob_divisible_by_three ^ 3) * 
  (prob_not_divisible_by_three ^ 2)

theorem dice_probability : 
  exactly_three_divisible_probability = 40 / 243 := by
  sorry

end NUMINAMATH_CALUDE_dice_probability_l3323_332312


namespace NUMINAMATH_CALUDE_cos_178_plus_theta_l3323_332374

theorem cos_178_plus_theta (θ : ℝ) (h : Real.sin (88 * π / 180 + θ) = 2/3) :
  Real.cos (178 * π / 180 + θ) = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_cos_178_plus_theta_l3323_332374


namespace NUMINAMATH_CALUDE_product_sum_relation_l3323_332311

theorem product_sum_relation (a b : ℝ) : 
  a * b = 2 * (a + b) + 1 → b = 7 → b - a = 4 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_relation_l3323_332311


namespace NUMINAMATH_CALUDE_unique_zero_composition_implies_m_bound_l3323_332387

/-- Given a function f(x) = x^2 + 2x + m where m is a real number,
    if f(f(x)) has exactly one zero, then 0 < m < 1 -/
theorem unique_zero_composition_implies_m_bound 
  (m : ℝ) 
  (f : ℝ → ℝ)
  (h1 : ∀ x, f x = x^2 + 2*x + m)
  (h2 : ∃! x, f (f x) = 0) :
  0 < m ∧ m < 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_zero_composition_implies_m_bound_l3323_332387


namespace NUMINAMATH_CALUDE_polygon_diagonals_l3323_332359

/-- The number of diagonals in a polygon with n sides -/
def diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem polygon_diagonals (m n : ℕ) : 
  m + n = 33 →
  diagonals m + diagonals n = 243 →
  max m n = 21 →
  diagonals (max m n) = 189 := by
sorry

end NUMINAMATH_CALUDE_polygon_diagonals_l3323_332359


namespace NUMINAMATH_CALUDE_extremum_implies_a_eq_neg_two_l3323_332325

/-- The function f(x) = a ln x + x^2 has an extremum at x = 1 -/
def has_extremum_at_one (a : ℝ) : Prop :=
  let f := fun (x : ℝ) => a * Real.log x + x^2
  ∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), x ≠ 1 ∧ |x - 1| < ε → f x ≤ f 1 ∨ f x ≥ f 1

/-- If f(x) = a ln x + x^2 has an extremum at x = 1, then a = -2 -/
theorem extremum_implies_a_eq_neg_two (a : ℝ) :
  has_extremum_at_one a → a = -2 :=
by sorry

end NUMINAMATH_CALUDE_extremum_implies_a_eq_neg_two_l3323_332325


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3323_332338

/-- The eccentricity of a hyperbola with the given properties is √3 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  ∃ (c x y n : ℝ),
  -- Hyperbola equation
  x^2 / a^2 - y^2 / b^2 = 1 ∧
  -- M is on the hyperbola
  c^2 / a^2 - (b^2 / a)^2 / b^2 = 1 ∧
  -- F is a focus
  c^2 = a^2 + b^2 ∧
  -- M is center of circle
  (x - c)^2 + (y - n)^2 = (b^2 / a)^2 ∧
  -- Circle tangent to x-axis at F
  n = b^2 / a ∧
  -- Circle intersects y-axis
  c^2 + n^2 = (2 * n)^2 ∧
  -- MPQ is equilateral
  c^2 = 3 * n^2 →
  -- Eccentricity is √3
  c / a = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3323_332338


namespace NUMINAMATH_CALUDE_cara_seating_arrangements_l3323_332357

/-- The number of people at the table -/
def total_people : ℕ := 8

/-- The number of people Cara can choose from for her other neighbor -/
def available_neighbors : ℕ := total_people - 2

/-- The number of different possible pairs of people Cara could be sitting between -/
def seating_arrangements : ℕ := available_neighbors

theorem cara_seating_arrangements :
  seating_arrangements = 6 :=
sorry

end NUMINAMATH_CALUDE_cara_seating_arrangements_l3323_332357


namespace NUMINAMATH_CALUDE_max_value_expression_l3323_332326

theorem max_value_expression (n : ℕ) (h : n = 15000) :
  let factorization := 2^3 * 3 * 5^4
  ∃ (x y : ℕ), 
    (2*x - y = 0 ∨ 3*x - y = 0) ∧ 
    (x ∣ n) ∧
    ∀ (x' y' : ℕ), (2*x' - y' = 0 ∨ 3*x' - y' = 0) ∧ (x' ∣ n) → 
      2*x + 3*y ≥ 2*x' + 3*y' ∧
      2*x + 3*y = 60000 := by
  sorry

end NUMINAMATH_CALUDE_max_value_expression_l3323_332326


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3323_332322

theorem quadratic_equation_solution :
  ∃! (x : ℝ), x > 0 ∧ 5 * x^2 + 8 * x - 24 = 0 ∧ x = 6/5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3323_332322


namespace NUMINAMATH_CALUDE_geometric_sequence_from_arithmetic_l3323_332314

/-- Given a geometric sequence {b_n} where b_1 = 3, and whose 7th, 10th, and 15th terms
    form consecutive terms of an arithmetic sequence with non-zero common difference,
    prove that the general form of b_n is 3 * (5/3)^(n-1). -/
theorem geometric_sequence_from_arithmetic (b : ℕ → ℚ) (d : ℚ) :
  b 1 = 3 →
  d ≠ 0 →
  (∃ a : ℚ, b 7 = a + 6 * d ∧ b 10 = a + 9 * d ∧ b 15 = a + 14 * d) →
  (∀ n : ℕ, b n = 3 * (5/3)^(n-1)) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_from_arithmetic_l3323_332314


namespace NUMINAMATH_CALUDE_least_period_is_36_l3323_332315

-- Define the property that f must satisfy
def satisfies_condition (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 6) + f (x - 6) = f x

-- Define what it means for a function to have a period
def has_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x : ℝ, f (x + p) = f x

-- Define the least positive period
def is_least_positive_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  p > 0 ∧ has_period f p ∧ ∀ q : ℝ, 0 < q ∧ q < p → ¬(has_period f q)

-- The main theorem
theorem least_period_is_36 (f : ℝ → ℝ) (h : satisfies_condition f) :
  is_least_positive_period f 36 := by
  sorry

end NUMINAMATH_CALUDE_least_period_is_36_l3323_332315


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3323_332362

/-- The line ax + y + a + 1 = 0 always passes through the point (-1, -1) for all values of a. -/
theorem line_passes_through_fixed_point (a : ℝ) : a * (-1) + (-1) + a + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3323_332362


namespace NUMINAMATH_CALUDE_lcm_of_25_35_50_l3323_332323

theorem lcm_of_25_35_50 : Nat.lcm 25 (Nat.lcm 35 50) = 350 := by sorry

end NUMINAMATH_CALUDE_lcm_of_25_35_50_l3323_332323


namespace NUMINAMATH_CALUDE_modulo_graph_intercepts_l3323_332303

theorem modulo_graph_intercepts (x₀ y₀ : ℕ) : 
  x₀ < 29 → y₀ < 29 → 
  (5 * x₀ ≡ 3 [ZMOD 29]) → 
  (2 * y₀ ≡ 26 [ZMOD 29]) → 
  x₀ + y₀ = 31 := by sorry

end NUMINAMATH_CALUDE_modulo_graph_intercepts_l3323_332303


namespace NUMINAMATH_CALUDE_sector_area_l3323_332347

/-- The area of a sector with radius 2 and central angle π/4 is π/2 -/
theorem sector_area (r : ℝ) (α : ℝ) (S : ℝ) : 
  r = 2 → α = π / 4 → S = (1 / 2) * r^2 * α → S = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l3323_332347


namespace NUMINAMATH_CALUDE_base_n_problem_l3323_332344

theorem base_n_problem (n : ℕ+) (d : ℕ) (h1 : d < 10) 
  (h2 : 4 * n ^ 2 + 2 * n + d = 347)
  (h3 : 4 * n ^ 2 + 2 * n + 9 = 1 * 7 ^ 3 + 2 * 7 ^ 2 + d * 7 + 2) :
  n + d = 11 := by
sorry

end NUMINAMATH_CALUDE_base_n_problem_l3323_332344


namespace NUMINAMATH_CALUDE_parallelogram_properties_l3323_332381

/-- Represents a parallelogram with given dimensions -/
structure Parallelogram where
  base : ℝ
  height : ℝ
  total_side : ℝ

/-- Calculate the area of a parallelogram -/
def area (p : Parallelogram) : ℝ := p.base * p.height

/-- Calculate the slant height of a parallelogram -/
def slant_height (p : Parallelogram) : ℝ := p.total_side - p.height

theorem parallelogram_properties (p : Parallelogram) 
  (h_base : p.base = 20)
  (h_height : p.height = 6)
  (h_total_side : p.total_side = 9) :
  area p = 120 ∧ slant_height p = 3 := by
  sorry


end NUMINAMATH_CALUDE_parallelogram_properties_l3323_332381


namespace NUMINAMATH_CALUDE_not_product_of_two_primes_l3323_332317

theorem not_product_of_two_primes (n : ℕ) (h : n ≥ 2) :
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a > 1 ∧ b > 1 ∧ c > 1 ∧
  (a * b * c ∣ 2^(4*n + 2) + 1) :=
sorry

end NUMINAMATH_CALUDE_not_product_of_two_primes_l3323_332317


namespace NUMINAMATH_CALUDE_prob_at_least_6_heads_value_l3323_332399

/-- The probability of getting at least 6 heads when flipping a fair coin 8 times -/
def prob_at_least_6_heads : ℚ :=
  (Nat.choose 8 6 + Nat.choose 8 7 + Nat.choose 8 8) / 2^8

/-- Theorem stating that the probability of getting at least 6 heads
    when flipping a fair coin 8 times is 37/256 -/
theorem prob_at_least_6_heads_value :
  prob_at_least_6_heads = 37 / 256 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_6_heads_value_l3323_332399


namespace NUMINAMATH_CALUDE_sqrt_three_squared_plus_one_l3323_332367

theorem sqrt_three_squared_plus_one : (Real.sqrt 3)^2 + 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_squared_plus_one_l3323_332367


namespace NUMINAMATH_CALUDE_room_width_l3323_332373

/-- Given a rectangular room with length 18 m and unknown width, surrounded by a 2 m wide veranda on all sides, 
    if the area of the veranda is 136 m², then the width of the room is 12 m. -/
theorem room_width (w : ℝ) : 
  w > 0 →  -- Ensure width is positive
  (22 * (w + 4) - 18 * w = 136) →  -- Area of veranda equation
  w = 12 := by
sorry

end NUMINAMATH_CALUDE_room_width_l3323_332373


namespace NUMINAMATH_CALUDE_cuboid_specific_surface_area_l3323_332356

/-- The surface area of a cuboid with given dimensions -/
def cuboid_surface_area (length breadth height : ℝ) : ℝ :=
  2 * (length * height + length * breadth + breadth * height)

/-- Theorem: The surface area of a cuboid with length 12 cm, breadth 6 cm, and height 10 cm is 504 cm² -/
theorem cuboid_specific_surface_area :
  cuboid_surface_area 12 6 10 = 504 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_specific_surface_area_l3323_332356


namespace NUMINAMATH_CALUDE_line_equation_l3323_332396

/-- Given a line y = kx + b passing through points (-1, 0) and (0, 3),
    prove that its equation is y = 3x + 3 -/
theorem line_equation (k b : ℝ) : 
  (k * (-1) + b = 0) → (b = 3) → (k = 3 ∧ b = 3) := by
  sorry

end NUMINAMATH_CALUDE_line_equation_l3323_332396


namespace NUMINAMATH_CALUDE_smallest_divisible_by_18_20_25_l3323_332342

theorem smallest_divisible_by_18_20_25 : ∃ n : ℕ+, n = 900 ∧ 
  (∀ m : ℕ+, m < n → (¬(18 ∣ m) ∨ ¬(20 ∣ m) ∨ ¬(25 ∣ m))) ∧
  18 ∣ n ∧ 20 ∣ n ∧ 25 ∣ n := by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_18_20_25_l3323_332342


namespace NUMINAMATH_CALUDE_shaded_area_semicircles_l3323_332353

/-- The area of the shaded region formed by semicircles in a given pattern -/
theorem shaded_area_semicircles (diameter : Real) (pattern_length_feet : Real) : 
  diameter = 3 →
  pattern_length_feet = 1.5 →
  (pattern_length_feet * 12 / diameter) * (π * (diameter / 2)^2 / 2) = 13.5 * π := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_semicircles_l3323_332353


namespace NUMINAMATH_CALUDE_weight_of_A_l3323_332309

/-- Prove that given the conditions, the weight of A is 79 kg -/
theorem weight_of_A (a b c d e : ℝ) : 
  (a + b + c) / 3 = 84 →
  (a + b + c + d) / 4 = 80 →
  e = d + 7 →
  (b + c + d + e) / 4 = 79 →
  a = 79 := by
sorry

end NUMINAMATH_CALUDE_weight_of_A_l3323_332309


namespace NUMINAMATH_CALUDE_five_sided_polygon_angle_sum_l3323_332363

theorem five_sided_polygon_angle_sum 
  (A B C x y : ℝ) 
  (h1 : A = 28)
  (h2 : B = 74)
  (h3 : C = 26)
  (h4 : A + B + (360 - x) + 90 + (116 - y) = 540) :
  x + y = 128 := by
  sorry

end NUMINAMATH_CALUDE_five_sided_polygon_angle_sum_l3323_332363


namespace NUMINAMATH_CALUDE_percentage_problem_l3323_332331

theorem percentage_problem (n : ℝ) (h : 1.2 * n = 6000) : 0.2 * n = 1000 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3323_332331


namespace NUMINAMATH_CALUDE_total_milk_poured_l3323_332370

/-- Represents a bottle with a certain capacity -/
structure Bottle where
  capacity : ℝ

/-- Represents the amount of milk poured into a bottle -/
def pour (b : Bottle) (fraction : ℝ) : ℝ := b.capacity * fraction

theorem total_milk_poured (bottle1 bottle2 : Bottle) 
  (h1 : bottle1.capacity = 4)
  (h2 : bottle2.capacity = 8)
  (h3 : pour bottle2 (5.333333333333333 / bottle2.capacity) = 5.333333333333333) :
  pour bottle1 (5.333333333333333 / bottle2.capacity) + 
  pour bottle2 (5.333333333333333 / bottle2.capacity) = 8 := by
sorry

end NUMINAMATH_CALUDE_total_milk_poured_l3323_332370


namespace NUMINAMATH_CALUDE_situps_problem_l3323_332302

def total_situps (diana_rate : ℕ) (diana_total : ℕ) (hani_extra : ℕ) : ℕ :=
  let diana_time := diana_total / diana_rate
  let hani_rate := diana_rate + hani_extra
  let hani_total := hani_rate * diana_time
  diana_total + hani_total

theorem situps_problem :
  total_situps 4 40 3 = 110 :=
by sorry

end NUMINAMATH_CALUDE_situps_problem_l3323_332302


namespace NUMINAMATH_CALUDE_inequality_proof_l3323_332318

theorem inequality_proof (x y : ℝ) (p q : ℕ) 
  (hx : x > 0) (hy : y > 0) (hxy : x ≠ y) :
  (x^(-(p:ℝ)/q) - y^((p:ℝ)/q) * x^(-(2*p:ℝ)/q)) / 
  (x^((1-2*p:ℝ)/q) - y^((1:ℝ)/q) * x^(-(2*p:ℝ)/q)) > 
  p * (x*y)^((p-1:ℝ)/(2*q)) :=
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3323_332318


namespace NUMINAMATH_CALUDE_product_greater_than_sum_l3323_332368

theorem product_greater_than_sum (a b : ℝ) (ha : a ≥ 2) (hb : b > 2) : a * b > a + b := by
  sorry

end NUMINAMATH_CALUDE_product_greater_than_sum_l3323_332368


namespace NUMINAMATH_CALUDE_solution_pair_l3323_332336

theorem solution_pair : ∃ (x y : ℝ), 
  (2 * x + 3 * y = (7 - x) + (7 - y)) ∧ 
  (x - 2 * y = (x - 3) + (y - 3)) ∧ 
  x = 2 ∧ y = 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_pair_l3323_332336


namespace NUMINAMATH_CALUDE_percentage_problem_l3323_332335

theorem percentage_problem (x : ℝ) : 200 = 4 * x → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3323_332335


namespace NUMINAMATH_CALUDE_min_value_exponential_sum_l3323_332376

theorem min_value_exponential_sum (x y : ℝ) (h : x + 2 * y = 1) :
  2^x + 4^y ≥ 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_exponential_sum_l3323_332376


namespace NUMINAMATH_CALUDE_smallest_quadratic_coefficient_l3323_332330

theorem smallest_quadratic_coefficient (a : ℕ) : a ≥ 5 ↔ 
  ∃ (b c : ℤ) (x₁ x₂ : ℝ), 
    (0 < x₁ ∧ x₁ < 1) ∧ 
    (0 < x₂ ∧ x₂ < 1) ∧ 
    (x₁ ≠ x₂) ∧
    (∀ x, a * x^2 + b * x + c = a * (x - x₁) * (x - x₂)) ∧
    a > 0 ∧
    (∀ a' < a, ¬∃ (b' c' : ℤ) (y₁ y₂ : ℝ), 
      (0 < y₁ ∧ y₁ < 1) ∧ 
      (0 < y₂ ∧ y₂ < 1) ∧ 
      (y₁ ≠ y₂) ∧
      (∀ x, a' * x^2 + b' * x + c' = a' * (x - y₁) * (x - y₂)) ∧
      a' > 0) := by
  sorry

end NUMINAMATH_CALUDE_smallest_quadratic_coefficient_l3323_332330


namespace NUMINAMATH_CALUDE_reading_time_per_day_l3323_332392

-- Define the given conditions
def num_books : ℕ := 3
def num_days : ℕ := 10
def reading_rate : ℕ := 100 -- words per hour
def book1_words : ℕ := 200
def book2_words : ℕ := 400
def book3_words : ℕ := 300

-- Define the theorem
theorem reading_time_per_day :
  let total_words := book1_words + book2_words + book3_words
  let total_hours := total_words / reading_rate
  let total_minutes := total_hours * 60
  total_minutes / num_days = 54 := by
sorry


end NUMINAMATH_CALUDE_reading_time_per_day_l3323_332392


namespace NUMINAMATH_CALUDE_two_special_integers_under_million_l3323_332300

theorem two_special_integers_under_million : 
  ∃! (S : Finset Nat), 
    (∀ n ∈ S, n < 1000000 ∧ 
      ∃ a b : Nat, n = 2 * a^2 ∧ n = 3 * b^3) ∧ 
    S.card = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_special_integers_under_million_l3323_332300


namespace NUMINAMATH_CALUDE_range_of_even_quadratic_function_l3323_332343

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 2

-- State the theorem
theorem range_of_even_quadratic_function
  (a b : ℝ)
  (h_even : ∀ x, f a b x = f a b (-x))
  (h_domain : Set.Icc (1 + a) 2 = Set.Icc (-2) 2) :
  Set.range (f a b) = Set.Icc (-10) 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_even_quadratic_function_l3323_332343


namespace NUMINAMATH_CALUDE_one_third_percent_of_180_l3323_332372

-- Define the percentage as a fraction
def one_third_percent : ℚ := 1 / 3 / 100

-- Define the value we're calculating the percentage of
def base_value : ℚ := 180

-- Theorem statement
theorem one_third_percent_of_180 : one_third_percent * base_value = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_one_third_percent_of_180_l3323_332372


namespace NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l3323_332346

theorem least_positive_integer_with_remainders : ∃! x : ℕ, 
  x > 0 ∧
  x % 4 = 1 ∧
  x % 5 = 2 ∧
  x % 6 = 3 ∧
  ∀ y : ℕ, y > 0 ∧ y % 4 = 1 ∧ y % 5 = 2 ∧ y % 6 = 3 → x ≤ y :=
by
  use 17
  sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l3323_332346


namespace NUMINAMATH_CALUDE_problem_solution_l3323_332350

theorem problem_solution (m n c d a : ℝ) 
  (h1 : m + n = 0)  -- m and n are opposite numbers
  (h2 : c * d = 1)  -- c and d are reciprocals
  (h3 : a = ⌊Real.sqrt 5⌋) -- a is the integer part of √5
  : Real.sqrt (c * d) + 2 * (m + n) - a = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3323_332350


namespace NUMINAMATH_CALUDE_circle_trajectory_l3323_332349

-- Define the circle equation as a function of m, x, and y
def circle_equation (m x y : ℝ) : Prop :=
  x^2 + y^2 - (4*m + 2)*x - 2*m*y + 4*m^2 + 4*m + 1 = 0

-- Define the trajectory equation
def trajectory_equation (x y : ℝ) : Prop :=
  x - 2*y - 1 = 0

-- State the theorem
theorem circle_trajectory :
  ∀ m x y : ℝ, x ≠ 1 →
  (∃ m, circle_equation m x y) ↔ trajectory_equation x y :=
sorry

end NUMINAMATH_CALUDE_circle_trajectory_l3323_332349


namespace NUMINAMATH_CALUDE_twice_largest_two_digit_is_190_l3323_332313

def largest_two_digit (a b c : Nat) : Nat :=
  max (10 * max a (max b c) + min (max a b) (max b c))
      (10 * min (max a b) (max b c) + max a (max b c))

theorem twice_largest_two_digit_is_190 :
  largest_two_digit 3 5 9 * 2 = 190 := by
  sorry

end NUMINAMATH_CALUDE_twice_largest_two_digit_is_190_l3323_332313


namespace NUMINAMATH_CALUDE_parallel_implies_t_half_magnitude_when_t_one_l3323_332337

-- Define the vectors a and b as functions of t
def a (t : ℝ) : Fin 2 → ℝ := ![2 - t, 3]
def b (t : ℝ) : Fin 2 → ℝ := ![t, 1]

-- Theorem 1: If a and b are parallel, then t = 1/2
theorem parallel_implies_t_half :
  ∀ t : ℝ, (∃ k : ℝ, a t = k • b t) → t = 1/2 := by sorry

-- Theorem 2: When t = 1, |a - 4b| = √10
theorem magnitude_when_t_one :
  ‖(a 1) - 4 • (b 1)‖ = Real.sqrt 10 := by sorry

end NUMINAMATH_CALUDE_parallel_implies_t_half_magnitude_when_t_one_l3323_332337


namespace NUMINAMATH_CALUDE_coin_problem_l3323_332308

theorem coin_problem (total_coins : ℕ) (total_value : ℚ) 
  (h_total_coins : total_coins = 336)
  (h_total_value : total_value = 71)
  : ∃ (coins_20p coins_25p : ℕ),
    coins_20p + coins_25p = total_coins ∧
    (20 : ℚ)/100 * coins_20p + (25 : ℚ)/100 * coins_25p = total_value ∧
    coins_20p = 260 := by
  sorry

end NUMINAMATH_CALUDE_coin_problem_l3323_332308


namespace NUMINAMATH_CALUDE_equivalence_condition_l3323_332328

theorem equivalence_condition (a b : ℝ) (h : a * b > 0) :
  (a > b) ↔ (1 / a < 1 / b) :=
by sorry

end NUMINAMATH_CALUDE_equivalence_condition_l3323_332328


namespace NUMINAMATH_CALUDE_loot_box_cost_l3323_332329

/-- Proves that the cost of each loot box is $5 given the specified conditions -/
theorem loot_box_cost (avg_value : ℝ) (total_spent : ℝ) (total_loss : ℝ) :
  avg_value = 3.5 →
  total_spent = 40 →
  total_loss = 12 →
  ∃ (cost : ℝ), cost = 5 ∧ cost * (total_spent - total_loss) / total_spent = avg_value :=
by
  sorry


end NUMINAMATH_CALUDE_loot_box_cost_l3323_332329


namespace NUMINAMATH_CALUDE_cody_money_calculation_l3323_332355

def final_money (initial : ℝ) (birthday_gift : ℝ) (game_cost : ℝ) (clothes_percentage : ℝ) (late_gift : ℝ) : ℝ :=
  let after_birthday := initial + birthday_gift
  let after_game := after_birthday - game_cost
  let clothes_cost := clothes_percentage * after_game
  let after_clothes := after_game - clothes_cost
  after_clothes + late_gift

theorem cody_money_calculation :
  final_money 45 9 19 0.4 4.5 = 25.5 := by
  sorry

end NUMINAMATH_CALUDE_cody_money_calculation_l3323_332355


namespace NUMINAMATH_CALUDE_r_squared_equals_one_for_linear_plot_l3323_332354

/-- A scatter plot where all points lie on a straight line -/
structure LinearScatterPlot where
  /-- The slope of the line on which all points lie -/
  slope : ℝ
  /-- All points in the scatter plot lie on a straight line -/
  all_points_on_line : Bool

/-- The coefficient of determination (R²) for a scatter plot -/
def r_squared (plot : LinearScatterPlot) : ℝ :=
  sorry

/-- Theorem: If all points in a scatter plot lie on a straight line with a slope of 2,
    then R² equals 1 -/
theorem r_squared_equals_one_for_linear_plot (plot : LinearScatterPlot)
    (h1 : plot.slope = 2)
    (h2 : plot.all_points_on_line = true) :
    r_squared plot = 1 := by
  sorry

end NUMINAMATH_CALUDE_r_squared_equals_one_for_linear_plot_l3323_332354


namespace NUMINAMATH_CALUDE_cheapest_for_second_caterer_l3323_332304

-- Define the pricing functions for both caterers
def first_caterer (x : ℕ) : ℕ := 150 + 18 * x

def second_caterer (x : ℕ) : ℕ :=
  if x ≤ 30 then 250 + 15 * x
  else 400 + 10 * x

-- Define a function to compare the prices
def second_cheaper (x : ℕ) : Prop :=
  second_caterer x < first_caterer x

-- Theorem statement
theorem cheapest_for_second_caterer :
  ∀ n : ℕ, n < 32 → ¬(second_cheaper n) ∧ second_cheaper 32 :=
sorry

end NUMINAMATH_CALUDE_cheapest_for_second_caterer_l3323_332304
