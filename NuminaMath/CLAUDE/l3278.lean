import Mathlib

namespace NUMINAMATH_CALUDE_negation_of_divisible_by_two_is_even_l3278_327855

theorem negation_of_divisible_by_two_is_even :
  ¬(∀ n : ℤ, 2 ∣ n → Even n) ↔ ∃ n : ℤ, 2 ∣ n ∧ ¬Even n :=
by sorry

end NUMINAMATH_CALUDE_negation_of_divisible_by_two_is_even_l3278_327855


namespace NUMINAMATH_CALUDE_relationship_between_abc_l3278_327884

theorem relationship_between_abc (x y a b c : ℝ) 
  (ha : a = x + y) 
  (hb : b = x * y) 
  (hc : c = x^2 + y^2) : 
  a^2 = c + 2*b := by
sorry

end NUMINAMATH_CALUDE_relationship_between_abc_l3278_327884


namespace NUMINAMATH_CALUDE_cubic_root_between_integers_l3278_327850

theorem cubic_root_between_integers : ∃ (A B : ℤ), 
  B = A + 1 ∧ 
  ∃ (x : ℝ), A < x ∧ x < B ∧ x^3 + 5*x^2 - 3*x + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_between_integers_l3278_327850


namespace NUMINAMATH_CALUDE_gumball_difference_l3278_327883

def carl_gumballs : ℕ := 16
def lewis_gumballs : ℕ := 12
def amy_gumballs : ℕ := 20

theorem gumball_difference (x y : ℕ) :
  (18 * 5 ≤ carl_gumballs + lewis_gumballs + amy_gumballs + x + y) ∧
  (carl_gumballs + lewis_gumballs + amy_gumballs + x + y ≤ 27 * 5) →
  (87 : ℕ) - (42 : ℕ) = 45 := by
  sorry

end NUMINAMATH_CALUDE_gumball_difference_l3278_327883


namespace NUMINAMATH_CALUDE_sum_of_A_and_B_l3278_327826

/-- Represents a 3x3 grid with numbers 1, 2, and 3 -/
def Grid := Fin 3 → Fin 3 → Fin 3

/-- Check if a row contains 1, 2, and 3 -/
def valid_row (g : Grid) (row : Fin 3) : Prop :=
  ∀ n : Fin 3, ∃ col : Fin 3, g row col = n

/-- Check if a column contains 1, 2, and 3 -/
def valid_column (g : Grid) (col : Fin 3) : Prop :=
  ∀ n : Fin 3, ∃ row : Fin 3, g row col = n

/-- Check if the main diagonal contains 1, 2, and 3 -/
def valid_main_diagonal (g : Grid) : Prop :=
  ∀ n : Fin 3, ∃ i : Fin 3, g i i = n

/-- Check if the anti-diagonal contains 1, 2, and 3 -/
def valid_anti_diagonal (g : Grid) : Prop :=
  ∀ n : Fin 3, ∃ i : Fin 3, g i (2 - i) = n

/-- A grid is valid if all rows, columns, and diagonals contain 1, 2, and 3 -/
def valid_grid (g : Grid) : Prop :=
  (∀ row : Fin 3, valid_row g row) ∧
  (∀ col : Fin 3, valid_column g col) ∧
  valid_main_diagonal g ∧
  valid_anti_diagonal g

theorem sum_of_A_and_B (g : Grid) (h : valid_grid g) 
  (h1 : g 0 0 = 2) (h2 : g 1 2 = 3) : 
  g 1 1 + g 2 0 = 3 := by
  sorry


end NUMINAMATH_CALUDE_sum_of_A_and_B_l3278_327826


namespace NUMINAMATH_CALUDE_eva_marks_total_l3278_327869

/-- Represents Eva's marks in a single semester -/
structure SemesterMarks where
  maths : ℕ
  arts : ℕ
  science : ℕ

/-- Calculates the total marks for a semester -/
def totalMarks (s : SemesterMarks) : ℕ :=
  s.maths + s.arts + s.science

/-- Represents Eva's marks for the entire year -/
structure YearMarks where
  first : SemesterMarks
  second : SemesterMarks

/-- Calculates the total marks for the year -/
def yearTotal (y : YearMarks) : ℕ :=
  totalMarks y.first + totalMarks y.second

theorem eva_marks_total (eva : YearMarks) 
  (h1 : eva.first.maths = eva.second.maths + 10)
  (h2 : eva.first.arts = eva.second.arts - 15)
  (h3 : eva.first.science = eva.second.science - eva.second.science / 3)
  (h4 : eva.second.maths = 80)
  (h5 : eva.second.arts = 90)
  (h6 : eva.second.science = 90) :
  yearTotal eva = 485 := by
  sorry

end NUMINAMATH_CALUDE_eva_marks_total_l3278_327869


namespace NUMINAMATH_CALUDE_solve_for_T_l3278_327856

theorem solve_for_T : ∃ T : ℚ, (1/3 : ℚ) * (1/6 : ℚ) * T = (1/4 : ℚ) * (1/8 : ℚ) * 120 ∧ T = 67.5 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_T_l3278_327856


namespace NUMINAMATH_CALUDE_value_of_a_l3278_327817

theorem value_of_a (a b d : ℤ) 
  (h1 : a + b = d) 
  (h2 : b + d = 7) 
  (h3 : d = 4) : 
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_value_of_a_l3278_327817


namespace NUMINAMATH_CALUDE_no_ruler_for_quadratic_sum_l3278_327898

-- Define the type of monotonic functions on [0, 10]
def MonotonicOn (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x ∧ x ≤ y ∧ y ≤ 10 → f x ≤ f y

-- State the theorem
theorem no_ruler_for_quadratic_sum :
  ¬ ∃ (f g h : ℝ → ℝ),
    (MonotonicOn f) ∧ (MonotonicOn g) ∧ (MonotonicOn h) ∧
    (∀ x y, 0 ≤ x ∧ x ≤ 10 ∧ 0 ≤ y ∧ y ≤ 10 →
      f x + g y = h (x^2 + x*y + y^2)) :=
by sorry

end NUMINAMATH_CALUDE_no_ruler_for_quadratic_sum_l3278_327898


namespace NUMINAMATH_CALUDE_jacks_tire_slashing_l3278_327887

theorem jacks_tire_slashing (tire_cost window_cost total_cost : ℕ) 
  (h1 : tire_cost = 250)
  (h2 : window_cost = 700)
  (h3 : total_cost = 1450) :
  ∃ (num_tires : ℕ), num_tires * tire_cost + window_cost = total_cost ∧ num_tires = 3 := by
  sorry

end NUMINAMATH_CALUDE_jacks_tire_slashing_l3278_327887


namespace NUMINAMATH_CALUDE_michaels_trophy_increase_l3278_327828

theorem michaels_trophy_increase :
  let michael_current : ℕ := 30
  let jack_future : ℕ := 10 * michael_current
  let total_future : ℕ := 430
  let michael_increase : ℕ := total_future - (michael_current + jack_future)
  michael_increase = 100 := by
sorry

end NUMINAMATH_CALUDE_michaels_trophy_increase_l3278_327828


namespace NUMINAMATH_CALUDE_sin_cos_sum_equals_sqrt2_over_2_l3278_327841

theorem sin_cos_sum_equals_sqrt2_over_2 :
  Real.sin (347 * π / 180) * Real.cos (148 * π / 180) +
  Real.sin (77 * π / 180) * Real.cos (58 * π / 180) =
  Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_equals_sqrt2_over_2_l3278_327841


namespace NUMINAMATH_CALUDE_ceiling_minus_x_equals_half_l3278_327892

theorem ceiling_minus_x_equals_half (x : ℝ) (h : x - ⌊x⌋ = 0.5) : ⌈x⌉ - x = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_minus_x_equals_half_l3278_327892


namespace NUMINAMATH_CALUDE_graph_transformation_l3278_327831

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the reflection operation about x = 1
def reflect (f : ℝ → ℝ) : ℝ → ℝ := λ x => f (2 - x)

-- Define the left shift operation
def shift_left (f : ℝ → ℝ) : ℝ → ℝ := λ x => f (x + 1)

-- Theorem statement
theorem graph_transformation (f : ℝ → ℝ) :
  shift_left (reflect f) = λ x => f (1 - x) := by sorry

end NUMINAMATH_CALUDE_graph_transformation_l3278_327831


namespace NUMINAMATH_CALUDE_tan_alpha_minus_beta_equals_one_l3278_327851

theorem tan_alpha_minus_beta_equals_one (α β : ℝ) 
  (h : (3 / (2 + Real.sin (2 * α))) + (2021 / (2 + Real.sin β)) = 2024) : 
  Real.tan (α - β) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_minus_beta_equals_one_l3278_327851


namespace NUMINAMATH_CALUDE_family_movie_night_l3278_327878

/-- Proves the number of adults in a family given ticket prices and payment information -/
theorem family_movie_night (regular_price : ℕ) (child_discount : ℕ) (total_payment : ℕ) (change : ℕ) (num_children : ℕ) : 
  regular_price = 9 →
  child_discount = 2 →
  total_payment = 40 →
  change = 1 →
  num_children = 3 →
  (total_payment - change - num_children * (regular_price - child_discount)) / regular_price = 2 := by
sorry

end NUMINAMATH_CALUDE_family_movie_night_l3278_327878


namespace NUMINAMATH_CALUDE_koala_fiber_consumption_l3278_327849

/-- The amount of fiber a koala absorbs as a percentage of what it eats -/
def koala_absorption_rate : ℝ := 0.30

/-- The amount of fiber absorbed by the koala in one day (in ounces) -/
def fiber_absorbed : ℝ := 12

/-- Theorem: If a koala absorbs 30% of the fiber it eats and it absorbed 12 ounces of fiber in one day,
    then the total amount of fiber the koala ate that day was 40 ounces. -/
theorem koala_fiber_consumption :
  fiber_absorbed = koala_absorption_rate * 40 := by
  sorry

end NUMINAMATH_CALUDE_koala_fiber_consumption_l3278_327849


namespace NUMINAMATH_CALUDE_union_covers_reals_l3278_327867

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - x - 2 > 0}
def B (a : ℝ) : Set ℝ := {x | |x - a| < 3}

-- State the theorem
theorem union_covers_reals (a : ℝ) : 
  (A ∪ B a = Set.univ) → a ∈ Set.Ioo (-1) 2 := by sorry

end NUMINAMATH_CALUDE_union_covers_reals_l3278_327867


namespace NUMINAMATH_CALUDE_middle_school_soccer_league_l3278_327825

theorem middle_school_soccer_league (n : ℕ) : n = 9 :=
  by
  have total_games : n * (n - 1) / 2 = 36 := by sorry
  have min_games_per_team : n - 1 ≥ 8 := by sorry
  sorry

#check middle_school_soccer_league

end NUMINAMATH_CALUDE_middle_school_soccer_league_l3278_327825


namespace NUMINAMATH_CALUDE_probability_second_red_given_first_red_for_given_numbers_l3278_327827

/-- Represents the probability of drawing a red ball on the second draw,
    given that the first ball drawn was red. -/
def probability_second_red_given_first_red (total : ℕ) (red : ℕ) (white : ℕ) : ℚ :=
  if total = red + white ∧ red > 0 ∧ white ≥ 0 then
    (red - 1) / (total - 1)
  else
    0

theorem probability_second_red_given_first_red_for_given_numbers :
  probability_second_red_given_first_red 10 6 4 = 5/9 := by
  sorry

end NUMINAMATH_CALUDE_probability_second_red_given_first_red_for_given_numbers_l3278_327827


namespace NUMINAMATH_CALUDE_P3_is_one_fourth_P4_is_three_fourths_l3278_327857

/-- The probability that the center of a circle is in the interior of the convex hull of n points
    selected independently with uniform distribution on the circle. -/
def P (n : ℕ) : ℝ := sorry

/-- Theorem: The probability P3 is 1/4 -/
theorem P3_is_one_fourth : P 3 = 1/4 := by sorry

/-- Theorem: The probability P4 is 3/4 -/
theorem P4_is_three_fourths : P 4 = 3/4 := by sorry

end NUMINAMATH_CALUDE_P3_is_one_fourth_P4_is_three_fourths_l3278_327857


namespace NUMINAMATH_CALUDE_highland_baseball_club_members_l3278_327819

/-- The cost of a pair of socks in dollars -/
def sockCost : ℕ := 6

/-- The additional cost of a T-shirt compared to a pair of socks in dollars -/
def tShirtAdditionalCost : ℕ := 7

/-- The total expenditure for all members in dollars -/
def totalExpenditure : ℕ := 5112

/-- Calculates the number of members in the Highland Baseball Club -/
def calculateMembers (sockCost tShirtAdditionalCost totalExpenditure : ℕ) : ℕ :=
  let tShirtCost := sockCost + tShirtAdditionalCost
  let capCost := sockCost
  let costPerMember := (sockCost + tShirtCost) + (sockCost + tShirtCost + capCost)
  totalExpenditure / costPerMember

theorem highland_baseball_club_members :
  calculateMembers sockCost tShirtAdditionalCost totalExpenditure = 116 := by
  sorry

end NUMINAMATH_CALUDE_highland_baseball_club_members_l3278_327819


namespace NUMINAMATH_CALUDE_problem_statement_l3278_327816

theorem problem_statement (x : ℝ) (h : x + 1/x = 7) : 
  (x - 3)^2 + 49/(x - 3)^2 = 23 :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l3278_327816


namespace NUMINAMATH_CALUDE_middle_digit_is_six_l3278_327815

/-- Represents a three-digit number in a given base -/
structure ThreeDigitNumber (base : ℕ) where
  hundreds : ℕ
  tens : ℕ
  ones : ℕ
  valid_digits : hundreds < base ∧ tens < base ∧ ones < base

/-- Converts a ThreeDigitNumber to its numerical value -/
def to_nat {base : ℕ} (n : ThreeDigitNumber base) : ℕ :=
  n.hundreds * base^2 + n.tens * base + n.ones

/-- Theorem: For a number M that is a three-digit number in base 8 and
    has its digits reversed in base 10, the middle digit of M in base 8 is 6 -/
theorem middle_digit_is_six :
  ∀ (M_base8 : ThreeDigitNumber 8) (M_base10 : ThreeDigitNumber 10),
    to_nat M_base8 = to_nat M_base10 →
    M_base8.hundreds = M_base10.ones →
    M_base8.tens = M_base10.tens →
    M_base8.ones = M_base10.hundreds →
    M_base8.tens = 6 := by
  sorry

end NUMINAMATH_CALUDE_middle_digit_is_six_l3278_327815


namespace NUMINAMATH_CALUDE_unique_divisor_problem_l3278_327876

theorem unique_divisor_problem (dividend : Nat) (divisor : Nat) : 
  dividend = 12128316 →
  divisor * 7 < 1000 →
  divisor * 7 ≥ 100 →
  dividend % divisor = 0 →
  (∀ d : Nat, d ≠ divisor → 
    (d * 7 < 1000 ∧ d * 7 ≥ 100 ∧ dividend % d = 0) → False) →
  divisor = 124 := by
sorry

end NUMINAMATH_CALUDE_unique_divisor_problem_l3278_327876


namespace NUMINAMATH_CALUDE_corresponding_angles_random_l3278_327830

-- Define the concept of an event
def Event : Type := Unit

-- Define the concept of a random event
def RandomEvent (e : Event) : Prop := sorry

-- Define the given events
def sunRisesWest : Event := sorry
def triangleAngleSum : Event := sorry
def correspondingAngles : Event := sorry
def drawRedBall : Event := sorry

-- State the theorem
theorem corresponding_angles_random : RandomEvent correspondingAngles := by sorry

end NUMINAMATH_CALUDE_corresponding_angles_random_l3278_327830


namespace NUMINAMATH_CALUDE_pelicans_remaining_theorem_l3278_327865

/-- The number of Pelicans remaining in Shark Bite Cove after some moved to Pelican Bay -/
def pelicansRemaining (sharksInPelicanBay : ℕ) : ℕ :=
  let originalPelicans := sharksInPelicanBay / 2
  let pelicansMoved := originalPelicans / 3
  originalPelicans - pelicansMoved

/-- Theorem stating that given 60 sharks in Pelican Bay, 20 Pelicans remain in Shark Bite Cove -/
theorem pelicans_remaining_theorem :
  pelicansRemaining 60 = 20 := by
  sorry

#eval pelicansRemaining 60

end NUMINAMATH_CALUDE_pelicans_remaining_theorem_l3278_327865


namespace NUMINAMATH_CALUDE_platform_length_l3278_327879

/-- Given a train of length 900 meters that crosses a platform in 39 seconds
    and a signal pole in 18 seconds, prove that the length of the platform is 1050 meters. -/
theorem platform_length
  (train_length : ℝ)
  (time_cross_platform : ℝ)
  (time_cross_pole : ℝ)
  (h1 : train_length = 900)
  (h2 : time_cross_platform = 39)
  (h3 : time_cross_pole = 18) :
  let train_speed := train_length / time_cross_pole
  let platform_length := train_speed * time_cross_platform - train_length
  platform_length = 1050 := by sorry

end NUMINAMATH_CALUDE_platform_length_l3278_327879


namespace NUMINAMATH_CALUDE_each_score_is_individual_l3278_327864

/-- Represents a candidate's math score -/
structure MathScore where
  score : ℝ

/-- Represents the population of candidates -/
structure Population where
  candidates : Finset MathScore
  size_gt_100000 : candidates.card > 100000

/-- Represents a sample of candidates -/
structure Sample where
  scores : Finset MathScore
  size_eq_1000 : scores.card = 1000

/-- Theorem stating that each math score in the sample is an individual data point -/
theorem each_score_is_individual (pop : Population) (sample : Sample) 
  (h_sample : ∀ s ∈ sample.scores, s ∈ pop.candidates) :
  ∀ s ∈ sample.scores, ∃! i : MathScore, i = s :=
sorry

end NUMINAMATH_CALUDE_each_score_is_individual_l3278_327864


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3278_327870

/-- A hyperbola with given asymptotes and passing through a specific point -/
theorem hyperbola_equation (x y : ℝ) : 
  (∀ (k : ℝ), (2*x = 3*y ∨ 2*x = -3*y) → k*(2*x) = k*(3*y)) →  -- Asymptotes condition
  (4*(1:ℝ)^2 - 9*(2:ℝ)^2 = -32) →                              -- Point (1,2) satisfies the equation
  (4*x^2 - 9*y^2 = -32)                                         -- Resulting hyperbola equation
  := by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3278_327870


namespace NUMINAMATH_CALUDE_sqrt_50_between_consecutive_integers_product_l3278_327860

theorem sqrt_50_between_consecutive_integers_product : ∃ n : ℕ, 
  (n : ℝ) < Real.sqrt 50 ∧ Real.sqrt 50 < (n + 1 : ℝ) ∧ n * (n + 1) = 56 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_50_between_consecutive_integers_product_l3278_327860


namespace NUMINAMATH_CALUDE_parallel_segments_length_l3278_327818

/-- In a triangle with sides a, b, and c, if three segments parallel to the sides
    pass through one point and have equal length x, then x = (2abc) / (ab + ac + bc). -/
theorem parallel_segments_length (a b c x : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  x = (2 * a * b * c) / (a * b + a * c + b * c) := by
  sorry

#check parallel_segments_length

end NUMINAMATH_CALUDE_parallel_segments_length_l3278_327818


namespace NUMINAMATH_CALUDE_blue_candy_count_l3278_327859

theorem blue_candy_count (total : ℕ) (red : ℕ) (blue : ℕ) 
  (h1 : total = 3409)
  (h2 : red = 145)
  (h3 : blue = total - red) : blue = 3264 := by
  sorry

end NUMINAMATH_CALUDE_blue_candy_count_l3278_327859


namespace NUMINAMATH_CALUDE_vector_perpendicular_parallel_l3278_327810

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-3, 2)

def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, v = (t * w.1, t * w.2)

theorem vector_perpendicular_parallel :
  (∃ k : ℝ, perpendicular (k * a.1 + b.1, k * a.2 + b.2) (a.1 - 3 * b.1, a.2 - 3 * b.2) ∧ k = 19) ∧
  (∃ k : ℝ, parallel (k * a.1 + b.1, k * a.2 + b.2) (a.1 - 3 * b.1, a.2 - 3 * b.2) ∧ k = -1/3) :=
sorry

end NUMINAMATH_CALUDE_vector_perpendicular_parallel_l3278_327810


namespace NUMINAMATH_CALUDE_outfit_count_l3278_327880

def red_shirts : ℕ := 7
def green_shirts : ℕ := 7
def pants : ℕ := 8
def green_hats : ℕ := 10
def red_hats : ℕ := 10
def blue_hats : ℕ := 5

def total_outfits : ℕ := red_shirts * pants * (green_hats + blue_hats) + 
                          green_shirts * pants * (red_hats + blue_hats)

theorem outfit_count : total_outfits = 1680 := by
  sorry

end NUMINAMATH_CALUDE_outfit_count_l3278_327880


namespace NUMINAMATH_CALUDE_absolute_value_sum_zero_l3278_327877

theorem absolute_value_sum_zero (x y : ℝ) :
  |x - 6| + |y + 5| = 0 → x - y = 11 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_sum_zero_l3278_327877


namespace NUMINAMATH_CALUDE_zero_in_interval_l3278_327847

open Real

noncomputable def f (x : ℝ) : ℝ := 2 * log x + 2 * x^2 - 4 * x

theorem zero_in_interval :
  ∃ (c : ℝ), c ∈ Set.Ioo 1 (exp 1) ∧ f c = 0 :=
sorry

end NUMINAMATH_CALUDE_zero_in_interval_l3278_327847


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3278_327896

theorem quadratic_inequality_solution_set (a x : ℝ) :
  let inequality := a * x^2 + (a - 1) * x - 1 < 0
  (a = 0 → (inequality ↔ x > -1)) ∧
  (a > 0 → (inequality ↔ -1 < x ∧ x < 1/a)) ∧
  (-1 < a ∧ a < 0 → (inequality ↔ x < 1/a ∨ x > -1)) ∧
  (a = -1 → (inequality ↔ x ≠ -1)) ∧
  (a < -1 → (inequality ↔ x < -1 ∨ x > 1/a)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3278_327896


namespace NUMINAMATH_CALUDE_angle_B_value_side_b_value_l3278_327804

/-- Given an acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure AcuteTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2

/-- The sine law holds for the triangle -/
axiom sine_law (t : AcuteTriangle) : t.a / Real.sin t.A = t.b / Real.sin t.B

/-- The cosine law holds for the triangle -/
axiom cosine_law (t : AcuteTriangle) : t.b^2 = t.a^2 + t.c^2 - 2*t.a*t.c*Real.cos t.B

/-- The given condition a = 2b sin A -/
def condition (t : AcuteTriangle) : Prop := t.a = 2*t.b*Real.sin t.A

theorem angle_B_value (t : AcuteTriangle) (h : condition t) : t.B = π/6 := by sorry

theorem side_b_value (t : AcuteTriangle) (h1 : t.a = 3*Real.sqrt 3) (h2 : t.c = 5) (h3 : t.B = π/6) : 
  t.b = Real.sqrt 7 := by sorry

end NUMINAMATH_CALUDE_angle_B_value_side_b_value_l3278_327804


namespace NUMINAMATH_CALUDE_no_integer_solutions_l3278_327833

theorem no_integer_solutions :
  ¬ ∃ (m n : ℤ), m^2 - 11*m*n - 8*n^2 = 88 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l3278_327833


namespace NUMINAMATH_CALUDE_point_coordinates_l3278_327848

def fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

theorem point_coordinates :
  ∀ (x y : ℝ),
  fourth_quadrant x y →
  |y| = 12 →
  |x| = 4 →
  (x, y) = (4, -12) :=
by sorry

end NUMINAMATH_CALUDE_point_coordinates_l3278_327848


namespace NUMINAMATH_CALUDE_campaign_donation_percentage_l3278_327836

theorem campaign_donation_percentage :
  let max_donation : ℕ := 1200
  let max_donors : ℕ := 500
  let half_donation : ℕ := max_donation / 2
  let half_donors : ℕ := 3 * max_donors
  let total_raised : ℕ := 3750000
  let donation_sum : ℕ := max_donation * max_donors + half_donation * half_donors
  (donation_sum : ℚ) / total_raised * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_campaign_donation_percentage_l3278_327836


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l3278_327834

theorem quadratic_roots_sum (a b : ℝ) : 
  (∀ x, ax^2 + bx + 2 = 0 ↔ x = -1/2 ∨ x = 1/3) → 
  a + b = -14 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l3278_327834


namespace NUMINAMATH_CALUDE_parabola_focus_l3278_327899

/-- The focus of a parabola y = ax^2 is at (0, 1/(4a)) -/
theorem parabola_focus (a : ℝ) (h : a ≠ 0) :
  let f : ℝ × ℝ := (0, 1 / (4 * a))
  ∀ (x y : ℝ), y = a * x^2 → (x - f.1)^2 = 4 * (1 / (4 * a)) * (y - f.2) :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_l3278_327899


namespace NUMINAMATH_CALUDE_sqrt_minus_one_mod_prime_l3278_327854

theorem sqrt_minus_one_mod_prime (p : Nat) (h_prime : Prime p) (h_gt_two : p > 2) :
  (∃ x : Nat, x^2 ≡ -1 [ZMOD p]) ↔ ∃ k : Nat, p = 4*k + 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_minus_one_mod_prime_l3278_327854


namespace NUMINAMATH_CALUDE_eventual_habitable_fraction_l3278_327822

-- Define the fraction of earth's surface not covered by water
def landFraction : ℚ := 1 / 3

-- Define the fraction of exposed land initially inhabitable
def initialInhabitableFraction : ℚ := 1 / 3

-- Define the additional fraction of non-inhabitable land made viable by technology
def techAdvancementFraction : ℚ := 1 / 2

-- Theorem statement
theorem eventual_habitable_fraction :
  let initialHabitableLand := landFraction * initialInhabitableFraction
  let additionalHabitableLand := landFraction * (1 - initialInhabitableFraction) * techAdvancementFraction
  initialHabitableLand + additionalHabitableLand = 2 / 9 :=
by sorry

end NUMINAMATH_CALUDE_eventual_habitable_fraction_l3278_327822


namespace NUMINAMATH_CALUDE_expression_evaluation_l3278_327863

theorem expression_evaluation : (25 * 5 + 5^2) / (5^2 - 15) = 15 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3278_327863


namespace NUMINAMATH_CALUDE_binary_1011_is_11_decimal_124_is_octal_174_l3278_327807

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

def decimal_to_octal (n : Nat) : List Nat :=
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc
    else aux (m / 8) ((m % 8) :: acc)
  aux n []

theorem binary_1011_is_11 :
  binary_to_decimal [true, true, false, true] = 11 := by sorry

theorem decimal_124_is_octal_174 :
  decimal_to_octal 124 = [1, 7, 4] := by sorry

end NUMINAMATH_CALUDE_binary_1011_is_11_decimal_124_is_octal_174_l3278_327807


namespace NUMINAMATH_CALUDE_bookstore_sales_l3278_327844

theorem bookstore_sales (wednesday_sales : ℕ) (thursday_sales : ℕ) (friday_sales : ℕ) : 
  wednesday_sales = 15 →
  thursday_sales = 3 * wednesday_sales →
  friday_sales = thursday_sales / 5 →
  wednesday_sales + thursday_sales + friday_sales = 69 := by
sorry

end NUMINAMATH_CALUDE_bookstore_sales_l3278_327844


namespace NUMINAMATH_CALUDE_correct_mark_calculation_l3278_327886

/-- Proves that if a mark of 83 in a class of 26 pupils increases the class average by 0.5,
    then the correct mark should have been 70. -/
theorem correct_mark_calculation (total_marks : ℝ) (wrong_mark correct_mark : ℝ) : 
  (wrong_mark = 83) →
  (((total_marks + wrong_mark) / 26) = ((total_marks + correct_mark) / 26 + 0.5)) →
  (correct_mark = 70) := by
sorry

end NUMINAMATH_CALUDE_correct_mark_calculation_l3278_327886


namespace NUMINAMATH_CALUDE_pages_read_first_day_l3278_327820

theorem pages_read_first_day (total_pages : ℕ) (days : ℕ) (first_day_pages : ℕ) : 
  total_pages = 130 →
  days = 7 →
  total_pages = first_day_pages + (days - 1) * (2 * first_day_pages) →
  first_day_pages = 10 :=
by sorry

end NUMINAMATH_CALUDE_pages_read_first_day_l3278_327820


namespace NUMINAMATH_CALUDE_solution_set_l3278_327852

/-- An even function that is monotonically decreasing on [0,+∞) and f(1) = 0 -/
def f (x : ℝ) : ℝ := sorry

/-- f is an even function -/
axiom f_even : ∀ x, f x = f (-x)

/-- f is monotonically decreasing on [0,+∞) -/
axiom f_decreasing : ∀ x y, 0 ≤ x → x < y → f y < f x

/-- f(1) = 0 -/
axiom f_one_eq_zero : f 1 = 0

/-- The solution set of f(x-3) ≥ 0 is [2,4] -/
theorem solution_set : Set.Icc 2 4 = {x | f (x - 3) ≥ 0} := by sorry

end NUMINAMATH_CALUDE_solution_set_l3278_327852


namespace NUMINAMATH_CALUDE_circle_area_ratio_l3278_327835

theorem circle_area_ratio (s r : ℝ) (hs : s > 0) (hr : r > 0) (h : r = 0.4 * s) :
  (π * (r / 2)^2) / (π * (s / 2)^2) = 0.16 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l3278_327835


namespace NUMINAMATH_CALUDE_quadratic_solution_l3278_327832

theorem quadratic_solution : ∃ x : ℝ, x^2 - x - 1 = 0 ∧ (x = (1 + Real.sqrt 5) / 2 ∨ x = -(1 + Real.sqrt 5) / 2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_l3278_327832


namespace NUMINAMATH_CALUDE_certain_number_proof_l3278_327803

theorem certain_number_proof (given_division : 7125 / 1.25 = 5700) 
  (certain_number : ℝ) (certain_division : certain_number / 12.5 = 57) : 
  certain_number = 712.5 := by
sorry

end NUMINAMATH_CALUDE_certain_number_proof_l3278_327803


namespace NUMINAMATH_CALUDE_evaluate_expression_l3278_327838

theorem evaluate_expression : -(18 / 3 * 8 - 48 + 4 * 6) = -24 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3278_327838


namespace NUMINAMATH_CALUDE_base_difference_proof_l3278_327893

/-- Converts a number from base b to base 10 -/
def to_base_10 (digits : List ℕ) (b : ℕ) : ℕ :=
  digits.enum.foldr (fun (i, d) acc => acc + d * b^i) 0

/-- The digits of 543210 in base 8 -/
def num1_digits : List ℕ := [5, 4, 3, 2, 1, 0]

/-- The digits of 43210 in base 5 -/
def num2_digits : List ℕ := [4, 3, 2, 1, 0]

theorem base_difference_proof :
  to_base_10 num1_digits 8 - to_base_10 num2_digits 5 = 177966 := by
  sorry

end NUMINAMATH_CALUDE_base_difference_proof_l3278_327893


namespace NUMINAMATH_CALUDE_bouquet_cost_l3278_327872

/-- The cost of the bouquet given Michael's budget and other expenses --/
theorem bouquet_cost (michael_money : ℕ) (cake_cost : ℕ) (balloons_cost : ℕ) (extra_needed : ℕ) : 
  michael_money = 50 →
  cake_cost = 20 →
  balloons_cost = 5 →
  extra_needed = 11 →
  michael_money + extra_needed = cake_cost + balloons_cost + 36 := by
sorry

end NUMINAMATH_CALUDE_bouquet_cost_l3278_327872


namespace NUMINAMATH_CALUDE_cubic_polynomials_common_roots_l3278_327890

theorem cubic_polynomials_common_roots (a b : ℝ) :
  (∃ r s : ℝ, r ≠ s ∧
    r^3 + a*r^2 + 20*r + 10 = 0 ∧
    r^3 + b*r^2 + 17*r + 12 = 0 ∧
    s^3 + a*s^2 + 20*s + 10 = 0 ∧
    s^3 + b*s^2 + 17*s + 12 = 0) →
  a = 1 ∧ b = 0 :=
by sorry

end NUMINAMATH_CALUDE_cubic_polynomials_common_roots_l3278_327890


namespace NUMINAMATH_CALUDE_at_least_two_equal_books_l3278_327801

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem at_least_two_equal_books (books : Fin 4 → ℕ) 
  (h : ∀ i, books i / sum_of_digits (books i) = 13) : 
  ∃ i j, i ≠ j ∧ books i = books j := by sorry

end NUMINAMATH_CALUDE_at_least_two_equal_books_l3278_327801


namespace NUMINAMATH_CALUDE_quadratic_prime_square_l3278_327889

/-- A function that represents the given quadratic expression -/
def f (n : ℕ) : ℤ := 2 * n^2 - 5 * n - 33

/-- Predicate to check if a number is prime -/
def isPrime (p : ℕ) : Prop := Nat.Prime p

/-- The main theorem stating that 6 and 14 are the only natural numbers
    for which f(n) is the square of a prime number -/
theorem quadratic_prime_square : 
  ∀ n : ℕ, (∃ p : ℕ, isPrime p ∧ f n = p^2) ↔ n = 6 ∨ n = 14 :=
sorry

end NUMINAMATH_CALUDE_quadratic_prime_square_l3278_327889


namespace NUMINAMATH_CALUDE_inequality_solution_l3278_327824

theorem inequality_solution (x : ℝ) : 
  (2 / (x + 2) + 4 / (x + 8) ≥ 4 / 5) ↔ (x ≤ -8 ∨ (-2 ≤ x ∧ x ≤ 2)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3278_327824


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l3278_327885

theorem algebraic_expression_value : 
  let x : ℝ := -1
  3 * x^2 + 2 * x - 1 = 0 := by
sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l3278_327885


namespace NUMINAMATH_CALUDE_mudits_age_l3278_327829

/-- Mudit's present age satisfies the given condition -/
theorem mudits_age : ∃ (x : ℕ), (x + 16 = 3 * (x - 4)) ∧ (x = 14) := by
  sorry

end NUMINAMATH_CALUDE_mudits_age_l3278_327829


namespace NUMINAMATH_CALUDE_compound_interest_problem_l3278_327866

/-- Calculates the total amount after compound interest -/
def totalAmountAfterCompoundInterest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

theorem compound_interest_problem (compoundInterest : ℝ) (rate : ℝ) (time : ℕ) 
  (h1 : compoundInterest = 2828.80)
  (h2 : rate = 0.08)
  (h3 : time = 2) :
  ∃ (principal : ℝ), 
    totalAmountAfterCompoundInterest principal rate time - principal = compoundInterest ∧
    totalAmountAfterCompoundInterest principal rate time = 19828.80 := by
  sorry

#eval totalAmountAfterCompoundInterest 17000 0.08 2

end NUMINAMATH_CALUDE_compound_interest_problem_l3278_327866


namespace NUMINAMATH_CALUDE_system_solution_unique_l3278_327808

theorem system_solution_unique :
  ∃! (x y z : ℝ), x + y + z = 11 ∧ x^2 + 2*y^2 + 3*z^2 = 66 ∧ x = 6 ∧ y = 3 ∧ z = 2 := by
sorry

end NUMINAMATH_CALUDE_system_solution_unique_l3278_327808


namespace NUMINAMATH_CALUDE_movie_collection_size_l3278_327882

theorem movie_collection_size :
  ∀ (dvd blu : ℕ),
  dvd > 0 ∧ blu > 0 →
  dvd / blu = 17 / 4 →
  dvd / (blu - 4) = 9 / 2 →
  dvd + blu = 378 := by
sorry

end NUMINAMATH_CALUDE_movie_collection_size_l3278_327882


namespace NUMINAMATH_CALUDE_divisibility_relations_l3278_327858

theorem divisibility_relations (a b : ℤ) (ha : a ≥ 1) (hb : b ≥ 1) :
  (¬ ((a ∣ b^2) ↔ (a ∣ b))) ∧ ((a^2 ∣ b^2) ↔ (a ∣ b)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_relations_l3278_327858


namespace NUMINAMATH_CALUDE_product_of_roots_quadratic_l3278_327821

/-- Given a quadratic equation x^2 - 4x + 3 = 0 with roots x₁ and x₂, 
    the product of the roots x₁ * x₂ equals 3. -/
theorem product_of_roots_quadratic (x₁ x₂ : ℝ) : 
  x₁^2 - 4*x₁ + 3 = 0 → x₂^2 - 4*x₂ + 3 = 0 → x₁ * x₂ = 3 := by
  sorry


end NUMINAMATH_CALUDE_product_of_roots_quadratic_l3278_327821


namespace NUMINAMATH_CALUDE_roots_cube_equality_l3278_327874

/-- Given a polynomial P(x) = 3x² + 3mx + m² - 1 where m is a real number,
    and x₁, x₂ are the roots of P(x), prove that P(x₁³) = P(x₂³) -/
theorem roots_cube_equality (m : ℝ) (x₁ x₂ : ℝ) : 
  let P := fun x : ℝ => 3 * x^2 + 3 * m * x + m^2 - 1
  (P x₁ = 0 ∧ P x₂ = 0) → P (x₁^3) = P (x₂^3) := by
  sorry

end NUMINAMATH_CALUDE_roots_cube_equality_l3278_327874


namespace NUMINAMATH_CALUDE_mean_median_difference_l3278_327809

/-- Represents the frequency distribution of missed school days -/
def frequency_distribution : List (Nat × Nat) := [
  (0, 2), (1, 5), (2, 1), (3, 3), (4, 2), (5, 4), (6, 1), (7, 2)
]

/-- Total number of students -/
def total_students : Nat := 20

/-- Calculates the median number of days missed -/
def median (dist : List (Nat × Nat)) (total : Nat) : Rat :=
  sorry

/-- Calculates the mean number of days missed -/
def mean (dist : List (Nat × Nat)) (total : Nat) : Rat :=
  sorry

/-- The main theorem to prove -/
theorem mean_median_difference :
  mean frequency_distribution total_students - median frequency_distribution total_students = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_mean_median_difference_l3278_327809


namespace NUMINAMATH_CALUDE_greatest_common_factor_of_three_digit_palindromes_l3278_327881

def three_digit_palindrome (a b : ℕ) : ℕ := 100 * a + 10 * b + a

def is_valid_palindrome (n : ℕ) : Prop :=
  ∃ a b : ℕ, a ≥ 1 ∧ a ≤ 9 ∧ b ≥ 0 ∧ b ≤ 9 ∧ n = three_digit_palindrome a b

theorem greatest_common_factor_of_three_digit_palindromes :
  ∃ g : ℕ, g > 0 ∧
  (∀ n : ℕ, is_valid_palindrome n → g ∣ n) ∧
  (∀ d : ℕ, d > 0 → (∀ n : ℕ, is_valid_palindrome n → d ∣ n) → d ≤ g) ∧
  g = 1 := by sorry

end NUMINAMATH_CALUDE_greatest_common_factor_of_three_digit_palindromes_l3278_327881


namespace NUMINAMATH_CALUDE_chess_tournament_25_players_l3278_327842

/-- Calculate the number of games in a chess tournament -/
def chess_tournament_games (n : ℕ) : ℕ :=
  n * (n - 1)

/-- Theorem: In a chess tournament with 25 players, where each player plays twice against every other player, the total number of games is 1200 -/
theorem chess_tournament_25_players :
  2 * chess_tournament_games 25 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_25_players_l3278_327842


namespace NUMINAMATH_CALUDE_sphere_xz_intersection_radius_l3278_327894

/-- A sphere in 3D space -/
structure Sphere where
  center : ℝ × ℝ × ℝ
  radius : ℝ

/-- A circle in 3D space -/
structure Circle where
  center : ℝ × ℝ × ℝ
  radius : ℝ

/-- Theorem: The radius of the circle where the sphere intersects the xz-plane is 6 -/
theorem sphere_xz_intersection_radius : 
  ∀ (s : Sphere),
  ∃ (c1 c2 : Circle),
  c1.center = (3, 5, 0) ∧ c1.radius = 3 ∧  -- xy-plane intersection
  c2.center = (0, 5, -6) ∧                 -- xz-plane intersection
  (∃ (x y z : ℝ), s.center = (x, y, z)) →
  c2.radius = 6 := by
sorry


end NUMINAMATH_CALUDE_sphere_xz_intersection_radius_l3278_327894


namespace NUMINAMATH_CALUDE_cow_husk_consumption_l3278_327871

/-- Given that 50 cows eat 50 bags of husk in 50 days, prove that one cow will eat one bag of husk in 50 days -/
theorem cow_husk_consumption (cows bags days : ℕ) (h : cows = 50 ∧ bags = 50 ∧ days = 50) :
  (1 : ℕ) * bags * days = cows * (1 : ℕ) * days :=
by sorry

end NUMINAMATH_CALUDE_cow_husk_consumption_l3278_327871


namespace NUMINAMATH_CALUDE_factorization_equality_l3278_327875

theorem factorization_equality (a b : ℝ) : a * b^2 - 8 * a * b + 16 * a = a * (b - 4)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3278_327875


namespace NUMINAMATH_CALUDE_magnitude_v_l3278_327873

theorem magnitude_v (u v : ℂ) (h1 : u * v = 20 - 15 * I) (h2 : Complex.abs u = Real.sqrt 34) : 
  Complex.abs v = (25 * Real.sqrt 34) / 34 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_v_l3278_327873


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_two_l3278_327897

theorem reciprocal_of_negative_two :
  (1 : ℚ) / (-2 : ℚ) = -1/2 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_two_l3278_327897


namespace NUMINAMATH_CALUDE_fraction_expression_equality_l3278_327845

theorem fraction_expression_equality : (3/7 + 4/5) / (5/11 + 2/3) = 1419/1295 := by
  sorry

end NUMINAMATH_CALUDE_fraction_expression_equality_l3278_327845


namespace NUMINAMATH_CALUDE_min_c_value_l3278_327853

theorem min_c_value (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_order : a < b ∧ b < c)
  (h_unique : ∃! (x y : ℝ), 2 * x + y = 2003 ∧ y = |x - a| + |x - b| + |x - c|) :
  c ≥ 1002 ∧ ∃ (a' b' : ℕ), 0 < a' ∧ 0 < b' ∧ a' < b' ∧ b' < 1002 ∧
    ∃! (x y : ℝ), 2 * x + y = 2003 ∧ y = |x - a'| + |x - b'| + |x - 1002| :=
by sorry

end NUMINAMATH_CALUDE_min_c_value_l3278_327853


namespace NUMINAMATH_CALUDE_vector_parallel_value_l3278_327861

/-- Two vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem vector_parallel_value (x : ℝ) :
  let a : ℝ × ℝ := (2, x)
  let b : ℝ × ℝ := (6, 8)
  parallel a b → x = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_vector_parallel_value_l3278_327861


namespace NUMINAMATH_CALUDE_line_intersects_midpoint_of_segment_l3278_327806

/-- The value of c for which the line 2x + y = c intersects the midpoint of the line segment from (1, 4) to (7, 10) -/
theorem line_intersects_midpoint_of_segment (c : ℝ) : 
  (∃ (x y : ℝ), 2*x + y = c ∧ 
   x = (1 + 7) / 2 ∧ 
   y = (4 + 10) / 2) → 
  c = 15 := by
sorry


end NUMINAMATH_CALUDE_line_intersects_midpoint_of_segment_l3278_327806


namespace NUMINAMATH_CALUDE_matt_first_quarter_score_l3278_327888

/-- Calculates the total score in basketball given the number of 2-point and 3-point shots made. -/
def totalScore (twoPointShots threePointShots : ℕ) : ℕ :=
  2 * twoPointShots + 3 * threePointShots

/-- Proves that Matt's score in the first quarter is 14 points. -/
theorem matt_first_quarter_score :
  totalScore 4 2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_matt_first_quarter_score_l3278_327888


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l3278_327895

/-- Represents the side lengths of the nine squares in the rectangle -/
structure SquareSides where
  a1 : ℕ
  a2 : ℕ
  a3 : ℕ
  a4 : ℕ
  a5 : ℕ
  a6 : ℕ
  a7 : ℕ
  a8 : ℕ
  a9 : ℕ

/-- Checks if the given SquareSides satisfy the conditions of the problem -/
def isValidSquareSides (s : SquareSides) : Prop :=
  s.a1 = 2 ∧
  s.a1 + s.a2 = s.a3 ∧
  s.a1 + s.a3 = s.a4 ∧
  s.a3 + s.a4 = s.a5 ∧
  s.a4 + s.a5 = s.a6 ∧
  s.a2 + s.a3 + s.a5 = s.a7 ∧
  s.a2 + s.a7 = s.a8 ∧
  s.a1 + s.a4 + s.a6 = s.a9 ∧
  s.a6 + s.a9 = s.a7 + s.a8

/-- Represents the dimensions of the rectangle -/
structure RectangleDimensions where
  length : ℕ
  width : ℕ

/-- Checks if the given RectangleDimensions satisfy the conditions of the problem -/
def isValidRectangle (r : RectangleDimensions) : Prop :=
  r.length > r.width ∧
  Even r.length ∧
  Even r.width ∧
  r.length = r.width + 2

theorem rectangle_perimeter (s : SquareSides) (r : RectangleDimensions) :
  isValidSquareSides s → isValidRectangle r →
  r.length = s.a9 → r.width = s.a8 →
  2 * (r.length + r.width) = 68 := by
  sorry


end NUMINAMATH_CALUDE_rectangle_perimeter_l3278_327895


namespace NUMINAMATH_CALUDE_absolute_value_equals_self_implies_nonnegative_l3278_327840

theorem absolute_value_equals_self_implies_nonnegative (a : ℝ) : (|a| = a) → a ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equals_self_implies_nonnegative_l3278_327840


namespace NUMINAMATH_CALUDE_min_value_theorem_l3278_327839

theorem min_value_theorem (a b : ℝ) (h1 : a + b = 2) (h2 : b > 0) :
  (∀ x y : ℝ, x + y = 2 → y > 0 → (1 / (2 * |x|) + |x| / y) ≥ 3/4) ∧
  (∃ x y : ℝ, x + y = 2 ∧ y > 0 ∧ 1 / (2 * |x|) + |x| / y = 3/4) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3278_327839


namespace NUMINAMATH_CALUDE_magician_earnings_l3278_327802

/-- Calculates the money earned from selling magic card decks -/
def money_earned (price_per_deck : ℕ) (initial_decks : ℕ) (final_decks : ℕ) : ℕ :=
  (initial_decks - final_decks) * price_per_deck

/-- Proves that the magician earned 56 dollars -/
theorem magician_earnings :
  let price_per_deck : ℕ := 7
  let initial_decks : ℕ := 16
  let final_decks : ℕ := 8
  money_earned price_per_deck initial_decks final_decks = 56 := by
  sorry

end NUMINAMATH_CALUDE_magician_earnings_l3278_327802


namespace NUMINAMATH_CALUDE_number_classification_l3278_327805

-- Define the set of given numbers
def givenNumbers : Set ℝ := {-3, -1, 0, 20, 1/4, -6.5, 17/100, -8.5, 7, Real.pi, 16, -3.14}

-- Define the classification sets
def positiveNumbers : Set ℝ := {x | x > 0}
def integers : Set ℝ := {x | ∃ n : ℤ, x = n}
def fractions : Set ℝ := {x | ∃ a b : ℤ, b ≠ 0 ∧ x = a / b}
def positiveIntegers : Set ℝ := {x | ∃ n : ℕ, x = n ∧ n > 0}
def nonNegativeRationals : Set ℝ := {x | x ≥ 0 ∧ ∃ a b : ℤ, b ≠ 0 ∧ x = a / b}

-- Theorem statement
theorem number_classification :
  (givenNumbers ∩ positiveNumbers = {20, 1/4, 17/100, 7, 16, Real.pi}) ∧
  (givenNumbers ∩ integers = {-3, -1, 0, 20, 7, 16}) ∧
  (givenNumbers ∩ fractions = {1/4, -6.5, 17/100, -8.5, -3.14}) ∧
  (givenNumbers ∩ positiveIntegers = {20, 7, 16}) ∧
  (givenNumbers ∩ nonNegativeRationals = {0, 20, 1/4, 17/100, 7, 16}) := by
  sorry

end NUMINAMATH_CALUDE_number_classification_l3278_327805


namespace NUMINAMATH_CALUDE_simplified_expression_ratio_l3278_327837

theorem simplified_expression_ratio (k : ℤ) : 
  let simplified := (6 * k + 12) / 6
  let a : ℤ := 1
  let b : ℤ := 2
  simplified = a * k + b ∧ a / b = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_simplified_expression_ratio_l3278_327837


namespace NUMINAMATH_CALUDE_surjective_sum_iff_constant_l3278_327891

-- Define a surjective function over ℤ
def Surjective (g : ℤ → ℤ) : Prop :=
  ∀ y : ℤ, ∃ x : ℤ, g x = y

-- Define the property that f + g is surjective for all surjective g
def SurjectiveSum (f : ℤ → ℤ) : Prop :=
  ∀ g : ℤ → ℤ, Surjective g → Surjective (fun x ↦ f x + g x)

-- Define a constant function
def ConstantFunction (f : ℤ → ℤ) : Prop :=
  ∃ c : ℤ, ∀ x : ℤ, f x = c

-- Theorem statement
theorem surjective_sum_iff_constant (f : ℤ → ℤ) :
  SurjectiveSum f ↔ ConstantFunction f :=
sorry

end NUMINAMATH_CALUDE_surjective_sum_iff_constant_l3278_327891


namespace NUMINAMATH_CALUDE_range_of_cubic_sum_l3278_327823

theorem range_of_cubic_sum (a b : ℝ) (h : a^2 + b^2 = a + b) :
  0 ≤ a^3 + b^3 ∧ a^3 + b^3 ≤ 2 := by sorry

end NUMINAMATH_CALUDE_range_of_cubic_sum_l3278_327823


namespace NUMINAMATH_CALUDE_compare_large_exponents_l3278_327811

theorem compare_large_exponents : 1997^(1998^1999) > 1999^(1998^1997) := by
  sorry

end NUMINAMATH_CALUDE_compare_large_exponents_l3278_327811


namespace NUMINAMATH_CALUDE_largest_divisor_of_n_squared_divisible_by_18_l3278_327800

theorem largest_divisor_of_n_squared_divisible_by_18 (n : ℕ) (h1 : n > 0) (h2 : 18 ∣ n^2) :
  ∃ (d : ℕ), d = 6 ∧ d ∣ n ∧ ∀ (k : ℕ), k ∣ n → k ≤ d :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_n_squared_divisible_by_18_l3278_327800


namespace NUMINAMATH_CALUDE_sqrt2_minus1_power_representation_l3278_327868

theorem sqrt2_minus1_power_representation (n : ℤ) :
  ∃ (N : ℕ), (Real.sqrt 2 - 1) ^ n = Real.sqrt N - Real.sqrt (N - 1) :=
sorry

end NUMINAMATH_CALUDE_sqrt2_minus1_power_representation_l3278_327868


namespace NUMINAMATH_CALUDE_smallest_k_divisible_by_500_l3278_327846

theorem smallest_k_divisible_by_500 : 
  ∀ k : ℕ+, k.val < 3000 → ¬(500 ∣ (k.val * (k.val + 1) * (2 * k.val + 1) / 6)) ∧ 
  (500 ∣ (3000 * 3001 * 6001 / 6)) := by
  sorry

end NUMINAMATH_CALUDE_smallest_k_divisible_by_500_l3278_327846


namespace NUMINAMATH_CALUDE_three_digit_odd_count_l3278_327813

theorem three_digit_odd_count : 
  (Finset.filter 
    (fun n => n ≥ 100 ∧ n < 1000 ∧ n % 2 = 1) 
    (Finset.range 1000)).card = 450 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_odd_count_l3278_327813


namespace NUMINAMATH_CALUDE_perfect_squares_implications_l3278_327843

theorem perfect_squares_implications (n : ℕ+) 
  (h1 : ∃ a : ℕ, 3 * n + 1 = a^2) 
  (h2 : ∃ b : ℕ, 5 * n - 1 = b^2) :
  (∃ p q : ℕ, p > 1 ∧ q > 1 ∧ 7 * n + 13 = p * q) ∧ 
  (∃ x y : ℕ, 8 * (17 * n^2 + 3 * n) = x^2 + y^2) := by
sorry

end NUMINAMATH_CALUDE_perfect_squares_implications_l3278_327843


namespace NUMINAMATH_CALUDE_bb_tileable_iff_2b_divides_l3278_327812

/-- A rectangle is (b,b)-tileable if it can be covered by b×b square tiles --/
def is_bb_tileable (m n b : ℕ) : Prop :=
  ∃ (k l : ℕ), m = k * b ∧ n = l * b

/-- Main theorem: An m×n rectangle is (b,b)-tileable iff 2b divides both m and n --/
theorem bb_tileable_iff_2b_divides (m n b : ℕ) (hm : m > 0) (hn : n > 0) (hb : b > 0) :
  is_bb_tileable m n b ↔ (2 * b ∣ m) ∧ (2 * b ∣ n) :=
sorry

end NUMINAMATH_CALUDE_bb_tileable_iff_2b_divides_l3278_327812


namespace NUMINAMATH_CALUDE_line_perpendicular_parallel_implies_planes_perpendicular_l3278_327814

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (perpendicularPlanes : Plane → Plane → Prop)

-- State the theorem
theorem line_perpendicular_parallel_implies_planes_perpendicular
  (m : Line) (α β : Plane) :
  perpendicular m β → parallel m α → perpendicularPlanes α β := by
  sorry

end NUMINAMATH_CALUDE_line_perpendicular_parallel_implies_planes_perpendicular_l3278_327814


namespace NUMINAMATH_CALUDE_day_one_fish_count_l3278_327862

/-- The number of fish counted on day one -/
def day_one_count : ℕ := sorry

/-- The percentage of fish that are sharks -/
def shark_percentage : ℚ := 1/4

/-- The total number of sharks counted over two days -/
def total_sharks : ℕ := 15

theorem day_one_fish_count : 
  day_one_count = 15 :=
by
  have h1 : shark_percentage * (day_one_count + 3 * day_one_count) = total_sharks := sorry
  sorry


end NUMINAMATH_CALUDE_day_one_fish_count_l3278_327862
