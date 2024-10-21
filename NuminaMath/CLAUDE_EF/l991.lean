import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l991_99136

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin x ^ 2 + 2 * Real.sin x * Real.cos x - Real.sqrt 3

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  b : ℝ
  arithmetic_sequence : 2 * B = A + C
  angle_sum : A + B + C = Real.pi
  side_b : b = Real.sqrt 3
  max_at_A : ∀ x, f A ≥ f x

-- Define the area function for the triangle
noncomputable def area (t : Triangle) : ℝ :=
  (1/2) * t.b * Real.sin t.C

theorem triangle_properties (t : Triangle) :
  (∀ x, -2 ≤ f x ∧ f x ≤ 2) ∧  -- Range of f(x)
  (∀ x, f (x + Real.pi) = f x) ∧  -- Period of f(x)
  (area t = (3 + Real.sqrt 3) / 4) :=  -- Area of triangle ABC
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l991_99136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_rhombus_l991_99161

/-- The radius of a circle inscribed in a rhombus with given diagonals and perimeter -/
theorem inscribed_circle_radius_rhombus (d1 d2 p : ℝ) (h1 : d1 = 18) (h2 : d2 = 30) (h3 : p = 68) :
  (d1 * d2) / (4 * (p / 4)) = 135 / 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_rhombus_l991_99161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_105_degrees_l991_99157

theorem cos_105_degrees : Real.cos (105 * π / 180) = (Real.sqrt 2 - Real.sqrt 6) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_105_degrees_l991_99157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_formulas_l991_99123

/-- Represents a quadrilateral with side lengths a, b, c, d -/
structure Quadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- The semiperimeter of a quadrilateral -/
noncomputable def semiperimeter (q : Quadrilateral) : ℝ := (q.a + q.b + q.c + q.d) / 2

/-- The area of a tangential quadrilateral -/
noncomputable def tangential_area (q : Quadrilateral) (A C : ℝ) : ℝ :=
  Real.sqrt (q.a * q.b * q.c * q.d) * Real.sin ((A + C) / 2)

/-- The area of a convex quadrilateral -/
noncomputable def convex_area (q : Quadrilateral) (φ : ℝ) : ℝ :=
  let p := semiperimeter q
  Real.sqrt ((p - q.a) * (p - q.b) * (p - q.c) * (p - q.d) - q.a * q.b * q.c * q.d * (Real.cos φ)^2)

/-- Theorem: Area formulas for tangential and convex quadrilaterals -/
theorem quadrilateral_area_formulas (q : Quadrilateral) (A C φ : ℝ) :
  (∀ (S : ℝ), S = tangential_area q A C → S = Real.sqrt (q.a * q.b * q.c * q.d) * Real.sin ((A + C) / 2)) ∧
  (∀ (S : ℝ), S = convex_area q φ → 
    S = Real.sqrt ((semiperimeter q - q.a) * (semiperimeter q - q.b) * (semiperimeter q - q.c) * (semiperimeter q - q.d) - 
      q.a * q.b * q.c * q.d * (Real.cos φ)^2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_formulas_l991_99123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_with_perimeter_8_l991_99178

theorem triangle_area_with_perimeter_8 (a b c : ℕ) : 
  a + b + c = 8 → 
  a > 0 → b > 0 → c > 0 →
  a + b > c → b + c > a → c + a > b →
  (a : ℝ) * (b : ℝ) * Real.sqrt (1 - ((a^2 + b^2 - c^2) / (2*a*b))^2) / 2 = 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_with_perimeter_8_l991_99178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_and_values_l991_99181

noncomputable def f (a b x : ℝ) := a * Real.sin (2 * x - Real.pi / 3) + b

theorem axis_of_symmetry_and_values (a b : ℝ) :
  (∀ x : ℝ, ∃ k : ℤ, f a b x = f a b ((1/2) * ↑k * Real.pi + 5*Real.pi/12)) ∧
  (a > 0 ∧
   (∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi/2 → -2 ≤ f a b x ∧ f a b x ≤ Real.sqrt 3) ∧
   (∃ x₁ : ℝ, 0 ≤ x₁ ∧ x₁ ≤ Real.pi/2 ∧ f a b x₁ = -2) ∧
   (∃ x₂ : ℝ, 0 ≤ x₂ ∧ x₂ ≤ Real.pi/2 ∧ f a b x₂ = Real.sqrt 3)) →
  a = 2 ∧ b = Real.sqrt 3 - 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_and_values_l991_99181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_at_2_tangents_through_2_4_l991_99128

noncomputable section

/-- The curve function -/
def f (x : ℝ) : ℝ := (1/3) * x^3 + 4/3

/-- The derivative of the curve function -/
def f' (x : ℝ) : ℝ := x^2

/-- Tangent line equation at a point (x₀, f x₀) -/
def tangent_line (x₀ : ℝ) (x y : ℝ) : Prop :=
  y - f x₀ = f' x₀ * (x - x₀)

theorem tangent_at_2 :
  ∀ x y : ℝ, tangent_line 2 x y ↔ 4*x - y - 4 = 0 :=
by sorry

theorem tangents_through_2_4 :
  ∀ x₀ : ℝ, (tangent_line x₀ 2 4 ∧ x₀ ≠ 2) →
    (∀ x y : ℝ, tangent_line x₀ x y ↔ (4*x - y - 4 = 0 ∨ x - y + 2 = 0)) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_at_2_tangents_through_2_4_l991_99128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_walking_time_is_22_l991_99169

/-- Represents the scenario of a man walking and his wife driving. -/
structure CommuteScenario where
  early_arrival : ℕ  -- Minutes the man arrives early
  total_time : ℕ     -- Total time of walking and driving
  earlier_arrival : ℕ -- Minutes they arrive home earlier than usual

/-- Calculates the walking time given the commute scenario. -/
def walking_time (c : CommuteScenario) : ℕ :=
  (c.total_time - c.earlier_arrival) / 2

/-- Theorem stating that under the given conditions, the walking time is 22 minutes. -/
theorem walking_time_is_22 (c : CommuteScenario) 
  (h1 : c.early_arrival = 60)
  (h2 : c.total_time = 60)
  (h3 : c.earlier_arrival = 16) : 
  walking_time c = 22 := by
  sorry

#eval walking_time { early_arrival := 60, total_time := 60, earlier_arrival := 16 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_walking_time_is_22_l991_99169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_special_set_l991_99172

theorem existence_of_special_set : ∃ (S : Finset ℕ), 
  Finset.card S = 100 ∧ 
  (∀ (T : Finset ℕ), T ⊆ S → Finset.card T = 5 → 
    (T.prod (λ i => i)) % (T.sum (λ i => i)) = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_special_set_l991_99172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_special_lines_l991_99177

/-- A line in 2D space represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a line passes through a given point -/
def Line.passesThroughPoint (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y + l.c = 0

/-- Checks if a line has intercepts on the coordinate axes that are negative reciprocals of each other -/
def Line.hasNegativeReciprocalIntercepts (l : Line) : Prop :=
  (l.a ≠ 0 ∧ l.b ≠ 0) ∧ (l.c / l.a) * (l.c / l.b) = -1

/-- The set of lines passing through (-2, 4) with intercepts on the coordinate axes that are negative reciprocals of each other -/
def specialLines : Set Line :=
  {l : Line | l.passesThroughPoint (-2) 4 ∧ l.hasNegativeReciprocalIntercepts}

/-- There are exactly two special lines -/
theorem two_special_lines : (∃ (s : Finset Line), s.toSet = specialLines ∧ s.card = 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_special_lines_l991_99177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclists_circular_track_circumference_l991_99119

/-- The circumference of a circular track where two cyclists meet at the starting point -/
theorem cyclists_circular_track_circumference : ∃ (circumference : ℝ), circumference = 180 :=
  let speed1 : ℝ := 7  -- speed of first cyclist in m/s
  let speed2 : ℝ := 8  -- speed of second cyclist in m/s
  let time : ℝ := 12   -- time taken to meet at starting point in seconds
  let distance1 : ℝ := speed1 * time  -- distance covered by first cyclist
  let distance2 : ℝ := speed2 * time  -- distance covered by second cyclist
  let circumference : ℝ := distance1 + distance2  -- total distance covered
  by
    use circumference
    sorry  -- Placeholder for the actual proof


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclists_circular_track_circumference_l991_99119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_team_analysis_l991_99120

-- Define the given data
def team_wins_with_A : ℕ := 22
def total_games_with_A : ℕ := 30
def team_losses_without_A : ℕ := 12
def total_wins : ℕ := 30

-- Define the probabilities for player B
def prob_forward : ℚ := 1/5
def prob_center_forward : ℚ := 1/2
def prob_defender : ℚ := 1/5
def prob_goalkeeper : ℚ := 1/10

def prob_lose_forward : ℚ := 2/5
def prob_lose_center_forward : ℚ := 1/5
def prob_lose_defender : ℚ := 3/5
def prob_lose_goalkeeper : ℚ := 1/5

-- Define the chi-square critical value
def chi_square_critical : ℚ := 5024/1000

-- Theorem to prove
theorem team_analysis :
  -- Part 1: Prove the calculated values
  let b := total_games_with_A - team_wins_with_A
  let c := total_wins - team_wins_with_A
  let e := b + team_losses_without_A
  let f := c + team_losses_without_A
  let n := total_wins + e
  -- Part 2: Prove the chi-square test result
  let chi_square : ℚ := (n : ℚ) * ((team_wins_with_A * team_losses_without_A - b * c) : ℚ)^2 /
                    ((team_wins_with_A + b : ℚ) * (c + team_losses_without_A : ℚ) *
                     (team_wins_with_A + c : ℚ) * (b + team_losses_without_A : ℚ))
  -- Part 3: Prove the probability of team losing when player B participates
  let prob_team_lose_B := prob_forward * prob_lose_forward +
                          prob_center_forward * prob_lose_center_forward +
                          prob_defender * prob_lose_defender +
                          prob_goalkeeper * prob_lose_goalkeeper
  -- Part 4: Prove the probability of player B playing as forward given the team loses
  let prob_B_forward_given_lose := (prob_forward * prob_lose_forward) / prob_team_lose_B
  
  b = 8 ∧ c = 8 ∧ e = 20 ∧ f = 20 ∧ n = 50 ∧
  chi_square > chi_square_critical ∧
  prob_team_lose_B = 16/50 ∧
  prob_B_forward_given_lose = 1/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_team_analysis_l991_99120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_solutions_condition_product_of_solutions_l991_99164

/-- The function f(x) as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := |(x^2 - 10*x + 25) / (x - 5) - (x^2 - 3*x) / (3 - x)|

/-- The function p(x) as defined in the problem -/
def p (a : ℝ) : ℝ → ℝ := λ x ↦ a

/-- The main theorem stating the conditions for three solutions -/
theorem three_solutions_condition (a : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    |f x₁ - 5| = p a x₁ ∧ |f x₂ - 5| = p a x₂ ∧ |f x₃ - 5| = p a x₃ ∧
    ∀ x : ℝ, |f x - 5| = p a x → x = x₁ ∨ x = x₂ ∨ x = x₃) ↔
  a = 4 ∨ a = 5 :=
by sorry

/-- The product of the values of a that yield three solutions -/
theorem product_of_solutions : (4 : ℝ) * 5 = 20 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_solutions_condition_product_of_solutions_l991_99164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ending_number_divisible_by_three_l991_99195

theorem ending_number_divisible_by_three (start : ℕ) (sequence : List ℕ) : 
  (start > 100) →
  (sequence.length = 33) →
  (∀ n ∈ sequence, n % 3 = 0) →
  (∀ i, i < sequence.length - 1 → sequence[i + 1]! = sequence[i]! + 3) →
  (sequence[0]! = start) →
  (sequence[32]! = 198) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ending_number_divisible_by_three_l991_99195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_coefficient_l991_99191

noncomputable def coefficient_of_x_squared (f : ℝ → ℝ) : ℝ := sorry

theorem expansion_coefficient (a : ℝ) : 
  (coefficient_of_x_squared (λ x ↦ (x - a)^5) = 10) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_coefficient_l991_99191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_fountain_length_l991_99162

/-- Represents the work done to build a water fountain -/
structure WaterFountainWork where
  workers : ℕ
  days : ℕ
  length : ℝ

/-- The relationship between work and fountain length -/
def work_ratio (w1 w2 : WaterFountainWork) : Prop :=
  (w1.workers * w1.days * w2.length) = (w2.workers * w2.days * w1.length)

theorem water_fountain_length 
  (work1 work2 : WaterFountainWork) :
  work1.workers = 20 →
  work1.days = 6 →
  work2.workers = 35 →
  work2.days = 3 →
  work2.length = 49 →
  work_ratio work1 work2 →
  work1.length = 56 := by
  sorry

#check water_fountain_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_fountain_length_l991_99162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cost_at_15_floors_l991_99104

noncomputable def average_comprehensive_cost (x : ℝ) : ℝ :=
  560 + 48 * x + 10800 / x

theorem min_cost_at_15_floors :
  ∃ (x : ℝ), x ≥ 10 ∧ ∀ (y : ℝ), y ≥ 10 → average_comprehensive_cost x ≤ average_comprehensive_cost y :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cost_at_15_floors_l991_99104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_range_l991_99185

noncomputable def f (x : ℝ) : ℝ := Real.exp (1 + abs x) - 1 / (1 + x^4)

theorem f_inequality_range (x : ℝ) : 
  f (2 * x) < f (1 - x) ↔ x > -1 ∧ x < 1/3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_range_l991_99185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_equation_solution_l991_99167

theorem sqrt_equation_solution : ∃ x : ℝ, x ≥ 1 ∧ 
  Real.sqrt (x + 5 - 6 * Real.sqrt (x - 1)) + Real.sqrt (x + 12 - 8 * Real.sqrt (x - 1)) = 2 ∧ 
  abs (x - 13.25) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_equation_solution_l991_99167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l991_99188

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt x - 1 / x

-- Define the derivative of f
noncomputable def f' (x : ℝ) : ℝ := 1 / (Real.sqrt x) + 1 / (x^2)

-- State the theorem
theorem tangent_line_at_one :
  ∃ (m b : ℝ), (∀ x y : ℝ, y = m * x + b ↔ y - f 1 = (f' 1) * (x - 1)) ∧
                m * 1 + b = f 1 ∧
                m = 2 ∧
                b = -1 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l991_99188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_m_value_l991_99184

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {A B C D E F : ℝ} :
  (∀ x y : ℝ, A * x + B * y + C = 0 ↔ D * x + E * y + F = 0) → A / B = D / E

/-- Definition of line l₁ -/
def l₁ (m : ℝ) : ℝ → ℝ → Prop :=
  λ x y ↦ (m - 2) * x - 3 * y - 1 = 0

/-- Definition of line l₂ -/
def l₂ (m : ℝ) : ℝ → ℝ → Prop :=
  λ x y ↦ m * x + (m + 2) * y + 1 = 0

/-- Theorem stating that if l₁ is parallel to l₂, then m = -4 -/
theorem parallel_lines_m_value :
  ∀ m : ℝ, (∀ x y : ℝ, l₁ m x y ↔ l₂ m x y) → m = -4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_m_value_l991_99184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_denominator_l991_99121

theorem rationalize_denominator :
  (Real.sqrt 12 + (3 : Real) ^ (1/3)) / (Real.sqrt 3 + (2 : Real) ^ (1/3)) =
  (9 - (12 : Real) ^ (1/3) - (6 : Real) ^ (1/3)) / (3 - (4 : Real) ^ (1/3)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_denominator_l991_99121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_sum_l991_99102

-- Define the hexagon and its properties
structure Hexagon where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ
  F : ℝ × ℝ

-- Define helper functions
def is_convex (h : Hexagon) : Prop := sorry
def is_equilateral (h : Hexagon) : Prop := sorry
def angle_FAB (h : Hexagon) : ℝ := sorry
def parallel (A B C D : ℝ × ℝ) : Prop := sorry
def distinct_y_coordinates (h : Hexagon) (s : Set ℕ) : Prop := sorry

-- Define the conditions
def is_valid_hexagon (h : Hexagon) : Prop :=
  h.A = (0, 0) ∧
  ∃ b, h.B = (b, 1) ∧
  is_convex h ∧
  is_equilateral h ∧
  angle_FAB h = 150 ∧
  parallel h.A h.B h.D h.E ∧
  parallel h.B h.C h.E h.F ∧
  parallel h.C h.D h.F h.A ∧
  distinct_y_coordinates h {0, 1, 3, 5, 7, 9}

-- Define the area expression
def area_expression (h : Hexagon) : ℝ × ℕ → Prop
  | (m, n) => ∃ (area : ℝ), 
    area = m * Real.sqrt n ∧ 
    m > 0 ∧ 
    n > 0 ∧ 
    ∀ p : ℕ, Nat.Prime p → ¬(p^2 ∣ n)

-- The main theorem
theorem hexagon_area_sum (h : Hexagon) :
  is_valid_hexagon h →
  ∃ m n : ℕ, area_expression h (m, n) ∧ m + n = 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_sum_l991_99102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_football_preference_gender_relation_l991_99115

-- Define the total number of students
def total_students : ℕ := 100

-- Define the number of male and female students
def male_students : ℕ := 60
def female_students : ℕ := 40

-- Define the fraction of students who do not like football
def dislike_fraction : ℚ := 2/5

-- Define the ratio of female to male students who like football
def like_ratio : ℚ := 1/3

-- Define the chi-square threshold for 99.9% certainty
def chi_square_threshold : ℝ := 10.828

-- Define the fraction of students who like football in the class
def class_like_fraction : ℚ := 33/56

-- Theorem statement
theorem football_preference_gender_relation :
  -- Part 1: Chi-square statistic is greater than the threshold
  ∃ (a b c d : ℕ),
    a + b + c + d = total_students ∧
    a + c = male_students ∧
    b + d = female_students ∧
    c + d = (dislike_fraction * total_students).floor ∧
    b = (like_ratio * a).floor ∧
    (total_students : ℝ) * (a * d - b * c)^2 / ((a + b) * (c + d) * (a + c) * (b + d)) > chi_square_threshold ∧
  -- Part 2: Proportion of female students in the class
  ∃ (x y : ℚ),
    x + y = 1 ∧
    x * (3/8 : ℚ) + y * (3/4 : ℚ) = class_like_fraction ∧
    x = 3/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_football_preference_gender_relation_l991_99115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_l991_99141

/-- Given two vectors a and b in ℝ², if a is parallel to b, then their components satisfy a specific relation. -/
theorem parallel_vectors (a₁ a₂ b₁ b₂ : ℝ) :
  (a₁ = 2 ∧ a₂ = 6 ∧ b₁ = -1) →
  (∃ l : ℝ, b₂ = l ∧ a₁ * b₂ = a₂ * b₁) →
  b₂ = -3 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_l991_99141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_P_contains_positive_P_contains_negative_P_contains_odd_P_contains_even_minus_one_not_in_P_P_closed_under_addition_zero_in_P_two_not_in_P_l991_99133

-- Define the set P
def P : Set ℤ := sorry

-- Define the properties of P as theorems instead of axioms
theorem P_contains_positive : ∃ x : ℤ, x > 0 ∧ x ∈ P := sorry
theorem P_contains_negative : ∃ x : ℤ, x < 0 ∧ x ∈ P := sorry
theorem P_contains_odd : ∃ x : ℤ, Odd x ∧ x ∈ P := sorry
theorem P_contains_even : ∃ x : ℤ, Even x ∧ x ∈ P := sorry
theorem minus_one_not_in_P : -1 ∉ P := sorry
theorem P_closed_under_addition : ∀ x y : ℤ, x ∈ P → y ∈ P → (x + y) ∈ P := sorry

-- Theorem to prove
theorem zero_in_P_two_not_in_P : 0 ∈ P ∧ 2 ∉ P := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_P_contains_positive_P_contains_negative_P_contains_odd_P_contains_even_minus_one_not_in_P_P_closed_under_addition_zero_in_P_two_not_in_P_l991_99133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_circles_arrangement_six_circles_impossible_l991_99197

/-- A circle on a plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

/-- The origin point O --/
def O : ℝ × ℝ := (0, 0)

/-- Predicate to check if a ray from O intersects a circle --/
noncomputable def intersects_ray (c : Circle) : Prop :=
  ∃ (θ : ℝ), ∃ (t : ℝ), t > 0 ∧ 
    (t * Real.cos θ - c.center.1)^2 + (t * Real.sin θ - c.center.2)^2 = c.radius^2

/-- Predicate to check if O is not covered by a circle --/
def not_covers_O (c : Circle) : Prop :=
  c.center.1^2 + c.center.2^2 > c.radius^2

/-- Theorem for 7 circles --/
theorem seven_circles_arrangement :
  ∃ (circles : Fin 7 → Circle),
    (∀ i : Fin 7, not_covers_O (circles i)) ∧
    (∀ θ : ℝ, ∃ (i j k : Fin 7), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
      intersects_ray (circles i) ∧ intersects_ray (circles j) ∧ intersects_ray (circles k)) :=
sorry

/-- Theorem for 6 circles --/
theorem six_circles_impossible :
  ¬ ∃ (circles : Fin 6 → Circle),
    (∀ i : Fin 6, not_covers_O (circles i)) ∧
    (∀ θ : ℝ, ∃ (i j k : Fin 6), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
      intersects_ray (circles i) ∧ intersects_ray (circles j) ∧ intersects_ray (circles k)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_circles_arrangement_six_circles_impossible_l991_99197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l991_99109

noncomputable def f (x : ℝ) := Real.log (abs (Real.sin x))

theorem f_properties :
  (∀ x, f (x + π) = f x) ∧ 
  (∀ x, f (-x) = f x) ∧ 
  (∀ x y, 0 < x ∧ x < y ∧ y < π / 2 → f x < f y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l991_99109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_max_a_for_inequality_ln_inequality_l991_99103

open Real

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := x * log x
def g (a : ℝ) (x : ℝ) : ℝ := -x^2 + a*x - 3

-- Statement 1
theorem min_value_f (t : ℝ) (h : t > 0) :
  (∀ x ∈ Set.Icc t (t + 2), f x ≥ min (f t) (-1/exp 1)) ∧
  (t < 1/exp 1 → ∃ x ∈ Set.Icc t (t + 2), f x = -1/exp 1) ∧
  (t ≥ 1/exp 1 → ∀ x ∈ Set.Icc t (t + 2), f x ≥ f t) :=
by sorry

-- Statement 2
theorem max_a_for_inequality :
  ∃ a : ℝ, a = 4 ∧ ∀ x > 0, 2 * f x ≥ g a x ∧
  ∀ b > a, ∃ x > 0, 2 * f x < g b x :=
by sorry

-- Statement 3
theorem ln_inequality (x : ℝ) (h : x > 0) :
  log x > 1 / (exp x) - 2 / (exp 1 * x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_max_a_for_inequality_ln_inequality_l991_99103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_zero_f_monotone_increasing_l991_99140

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x + a) / (x^2 + 2)

theorem odd_function_implies_a_zero (a : ℝ) :
  (∀ x, f a (-x) = -(f a x)) → a = 0 := by
  sorry

theorem f_monotone_increasing :
  ∀ x y, 0 < x ∧ x < y ∧ y ≤ Real.sqrt 2 → f 0 x < f 0 y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_zero_f_monotone_increasing_l991_99140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_justify_buying_skates_l991_99183

/-- Calculates the number of visits required to justify buying skates --/
def visits_to_justify_buying (admission_cost : ℚ) (rental_cost : ℚ) (skates_cost : ℚ) (tax_rate : ℚ) (life_expectancy : ℕ) : ℕ :=
  let total_cost := skates_cost * (1 + tax_rate)
  let rental_savings_per_visit := rental_cost
  (total_cost / rental_savings_per_visit).ceil.toNat

theorem justify_buying_skates :
  let admission_cost : ℚ := 5
  let rental_cost : ℚ := 5/2
  let skates_cost : ℚ := 65
  let tax_rate : ℚ := 9/100
  let life_expectancy : ℕ := 2
  visits_to_justify_buying admission_cost rental_cost skates_cost tax_rate life_expectancy = 29 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_justify_buying_skates_l991_99183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_max_value_on_interval_min_value_on_interval_l991_99179

-- Define the function f(x) as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) + Real.sqrt 3 * Real.cos (2 * x)

-- Theorem for the smallest positive period
theorem smallest_positive_period :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T' ≥ T) ∧
  T = Real.pi :=
sorry

-- Theorem for the maximum value on [0, π/6]
theorem max_value_on_interval :
  ∃ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 6) ∧
  f x = 2 ∧
  ∀ (y : ℝ), y ∈ Set.Icc 0 (Real.pi / 6) → f y ≤ 2 :=
sorry

-- Theorem for the minimum value on [0, π/6]
theorem min_value_on_interval :
  ∃ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 6) ∧
  f x = Real.sqrt 3 ∧
  ∀ (y : ℝ), y ∈ Set.Icc 0 (Real.pi / 6) → f y ≥ Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_max_value_on_interval_min_value_on_interval_l991_99179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_interval_of_f_l991_99154

open Real

-- Define the function
noncomputable def f (x : ℝ) : ℝ := 2 * sin (π / 6 - 2 * x)

-- State the theorem
theorem increasing_interval_of_f :
  ∃ (a b : ℝ), a = π / 3 ∧ b = 5 * π / 6 ∧
  (∀ x ∈ Set.Icc 0 π, 
    (∀ y ∈ Set.Icc a b, x < y → f x < f y) ∧
    (∀ y ∈ Set.Icc 0 a, x < y → f x > f y) ∧
    (∀ y ∈ Set.Icc b π, x < y → f x > f y)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_interval_of_f_l991_99154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_to_fraction_l991_99198

/-- Given that 0.3̄45 is a repeating decimal where 45 repeats infinitely after 0.3,
    prove that it is equal to 83/110 as a fraction. -/
theorem repeating_decimal_to_fraction :
  ∃ (x : ℚ), (x = 3/10 + 45/990) ∧ (x = 83/110) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_to_fraction_l991_99198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_equilateral_triangle_area_l991_99180

/-- A square with side length 1 -/
structure UnitSquare where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  is_square : A = (0, 0) ∧ B = (1, 0) ∧ C = (1, 1) ∧ D = (0, 1)

/-- An equilateral triangle inscribed in a unit square -/
structure InscribedEquilateralTriangle (s : UnitSquare) where
  E : ℝ × ℝ
  F : ℝ × ℝ
  E_on_BC : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ E = (1, t)
  F_on_CD : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ F = (1 - t, 1)
  is_equilateral : (s.A.1 - E.1)^2 + (s.A.2 - E.2)^2 = 
                   (E.1 - F.1)^2 + (E.2 - F.2)^2 ∧
                   (s.A.1 - F.1)^2 + (s.A.2 - F.2)^2 = 
                   (E.1 - F.1)^2 + (E.2 - F.2)^2

/-- The area of the inscribed equilateral triangle -/
noncomputable def triangle_area (s : UnitSquare) (t : InscribedEquilateralTriangle s) : ℝ := 
  2 * Real.sqrt 3 - 3

/-- Theorem: The area of the inscribed equilateral triangle is 2√3 - 3 -/
theorem inscribed_equilateral_triangle_area 
  (s : UnitSquare) (t : InscribedEquilateralTriangle s) : 
  triangle_area s t = 2 * Real.sqrt 3 - 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_equilateral_triangle_area_l991_99180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_submarine_work_theorem_l991_99105

/-- The work done by a submarine pulling an air-filled bag underwater -/
noncomputable def submarine_work (V : ℝ) (h : ℝ) (p₀ : ℝ) (γ : ℝ) : ℝ :=
  p₀ * V * Real.log ((γ * h + p₀) / p₀)

/-- Theorem stating the work done by the submarine -/
theorem submarine_work_theorem (V : ℝ) (h : ℝ) (p₀ : ℝ) (γ : ℝ) 
  (hV : V > 0) (hh : h ≥ 0) (hp₀ : p₀ > 0) (hγ : γ > 0) :
  submarine_work V h p₀ γ = p₀ * V * Real.log ((γ * h + p₀) / p₀) := by
  -- Unfold the definition of submarine_work
  unfold submarine_work
  -- The equality holds by definition
  rfl

#check submarine_work_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_submarine_work_theorem_l991_99105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_for_smallest_a_l991_99150

/-- The smallest positive real number such that there exists a positive real number b
    where all roots of x^4 - ax^3 + ax^2 - bx + b = 0 are real -/
noncomputable def smallest_a : ℝ := 4

/-- The unique positive real number b corresponding to the smallest a -/
noncomputable def unique_b : ℝ := 1

/-- The quartic polynomial with coefficients determined by a and b -/
def quartic (x a b : ℝ) : ℝ := x^4 - a*x^3 + a*x^2 - b*x + b

/-- Extension of the quartic function to complex numbers -/
def quartic_complex (x : ℂ) (a b : ℝ) : ℂ := x^4 - a*x^3 + a*x^2 - b*x + b

theorem unique_solution_for_smallest_a :
  ∃ (a : ℝ), a > 0 ∧
  (∀ (a' : ℝ), a' > 0 → a' < a →
    ¬∃ (b : ℝ), b > 0 ∧ ∀ (x : ℂ), quartic_complex x a' b = 0 → x.im = 0) ∧
  ∃! (b : ℝ), b > 0 ∧ ∀ (x : ℂ), quartic_complex x a b = 0 → x.im = 0 ∧
  a = smallest_a ∧ b = unique_b :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_for_smallest_a_l991_99150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_7_value_l991_99192

/-- Given a real number x where x + 1/x = 3, S_m is defined as x^m + 1/x^m -/
noncomputable def S (x : ℝ) (m : ℕ) : ℝ := x^m + 1/x^m

/-- Theorem: If x is a real number such that x + 1/x = 3, then S_7 = 843 -/
theorem S_7_value (x : ℝ) (h : x + 1/x = 3) : S x 7 = 843 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_7_value_l991_99192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_coordinate_sum_l991_99165

/-- Triangle DEF in the Cartesian plane -/
structure Triangle where
  D : ℝ × ℝ
  E : ℝ × ℝ
  F : ℝ × ℝ

/-- Calculate the area of a triangle given its vertices -/
noncomputable def triangleArea (t : Triangle) : ℝ :=
  let (x1, y1) := t.D
  let (x2, y2) := t.E
  let (x3, y3) := t.F
  (1/2) * abs (x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))

/-- Calculate the slope of a line given two points -/
noncomputable def slopeLine (p1 p2 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  (y2 - y1) / (x2 - x1)

/-- Calculate the midpoint of a line segment -/
noncomputable def midpointLine (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  ((x1 + x2) / 2, (y1 + y2) / 2)

/-- The main theorem -/
theorem triangle_max_coordinate_sum (t : Triangle) :
  t.D = (10, 15) →
  t.E = (20, 18) →
  triangleArea t = 50 →
  slopeLine (midpointLine t.D t.E) t.F = -3 →
  ∃ (max : ℝ), max = 38.911 ∧ ∀ (r s : ℝ), t.F = (r, s) → r + s ≤ max := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_coordinate_sum_l991_99165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l991_99138

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- The distance between foci of an ellipse -/
noncomputable def focal_distance (e : Ellipse) : ℝ := Real.sqrt (e.a^2 - e.b^2)

/-- Theorem about the properties of a specific ellipse -/
theorem ellipse_properties (e : Ellipse) 
  (h_focal : focal_distance e = 2 * Real.sqrt 3)
  (h_sum : ∀ (x y : ℝ), x^2 / e.a^2 + y^2 / e.b^2 = 1 → 
    Real.sqrt ((x + focal_distance e / 2)^2 + y^2) + 
    Real.sqrt ((x - focal_distance e / 2)^2 + y^2) = 4)
  (h_major : e.a^2 + e.b^2 = 7) :
  (e.a = 2 ∧ e.b = Real.sqrt 3) ∧
  (∀ (t x₁ y₁ x₂ y₂ : ℝ), t ≠ 0 → x₁ ≠ x₂ → y₁ + y₂ ≠ 0 →
    x₁ = t * y₁ - 1 → x₂ = t * y₂ - 1 →
    x₁^2 / 4 + y₁^2 / 3 = 1 → x₂^2 / 4 + y₂^2 / 3 = 1 →
    ∃ (k : ℝ), k * (x₂ - x₁) = y₂ + y₁ ∧ k * (-4 - x₁) = -y₁) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l991_99138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_symmetry_l991_99168

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem function_symmetry (ω φ : ℝ) (h1 : ω > 0) (h2 : |φ| < π/2) 
  (h3 : ∀ x, f ω φ (x + π) = f ω φ x)  -- Minimum positive period is π
  (h4 : ∀ x, f ω φ (x + π/3) = -f ω φ (-x - π/3))  -- f(x + π/3) is odd
  : ∀ x, f ω φ (π/6 + x) = f ω φ (π/6 - x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_symmetry_l991_99168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jamie_works_three_hours_l991_99145

/-- Represents Jamie's flyer delivery job --/
structure FlyerJob where
  hourly_rate : ℚ
  days_per_week : ℕ
  total_weeks : ℕ
  total_earnings : ℚ

/-- Calculates the number of hours Jamie works each time she delivers flyers --/
def hours_per_delivery (job : FlyerJob) : ℚ :=
  job.total_earnings / (job.hourly_rate * job.days_per_week * job.total_weeks)

/-- Theorem stating that Jamie works 3 hours each time she delivers flyers --/
theorem jamie_works_three_hours (job : FlyerJob) 
  (h1 : job.hourly_rate = 10)
  (h2 : job.days_per_week = 2)
  (h3 : job.total_weeks = 6)
  (h4 : job.total_earnings = 360) :
  hours_per_delivery job = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jamie_works_three_hours_l991_99145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functions_not_equal_l991_99190

theorem functions_not_equal : ¬(∀ x : ℝ, Real.log (Real.exp x) = (10 : ℝ) ^ (Real.log x / Real.log 10)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functions_not_equal_l991_99190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_team_wins_l991_99108

/-- Calculates the number of games a basketball team must win to achieve a target win percentage -/
def games_to_win (total_games : ℕ) (games_played : ℕ) (games_won : ℕ) (target_percentage : ℚ) : ℕ :=
  let total_wins_needed := (total_games : ℚ) * target_percentage
  (total_wins_needed - games_won).ceil.toNat

theorem basketball_team_wins (total_games : ℕ) (games_played : ℕ) (games_won : ℕ) 
  (target_percentage : ℚ) (h1 : total_games = 100) (h2 : games_played = 60) 
  (h3 : games_won = 45) (h4 : target_percentage = 3/4) :
  games_to_win total_games games_played games_won target_percentage = 30 := by
  sorry

#eval games_to_win 100 60 45 (3/4)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_team_wins_l991_99108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_config_properties_l991_99112

/-- Represents a geometric configuration with a cone, an inscribed sphere, and the sphere's circumscribed cylinder -/
structure GeometricConfig where
  R : ℝ  -- radius of the sphere
  r : ℝ  -- radius of the base of the cone
  h : ℝ  -- height of the cone
  θ : ℝ  -- half apex angle of the cone

/-- Volume of the cone -/
noncomputable def cone_volume (config : GeometricConfig) : ℝ :=
  (1/3) * Real.pi * config.r^2 * config.h

/-- Volume of the cylinder -/
noncomputable def cylinder_volume (config : GeometricConfig) : ℝ :=
  2 * Real.pi * config.R^3

/-- The ratio of cone volume to cylinder volume -/
noncomputable def volume_ratio (config : GeometricConfig) : ℝ :=
  cone_volume config / cylinder_volume config

/-- Theorem stating the properties of the geometric configuration -/
theorem geometric_config_properties (config : GeometricConfig) 
  (h_positive : config.R > 0 ∧ config.r > 0 ∧ config.h > 0)
  (h_inscribed : config.h > 2 * config.R)
  (h_relation : config.R / (config.h - config.R) = config.r / Real.sqrt (config.r^2 + config.h^2))
  (h_angle : Real.sin config.θ = config.R / (config.h - config.R)) :
  (∀ c : GeometricConfig, cone_volume c ≠ cylinder_volume c) ∧
  (∃ min_ratio : ℝ, min_ratio = 4/3 ∧ 
    (∀ c : GeometricConfig, volume_ratio c ≥ min_ratio) ∧
    (∃ c : GeometricConfig, volume_ratio c = min_ratio ∧ Real.sin c.θ = 1/3)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_config_properties_l991_99112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_pyramid_volume_l991_99100

/-- Represents a pyramid with a rectangular base -/
structure RectangularPyramid where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular pyramid -/
noncomputable def volume (p : RectangularPyramid) : ℝ :=
  (1 / 3) * p.length * p.width * p.height

/-- Theorem stating the relationship between the original and new pyramid volumes -/
theorem new_pyramid_volume
  (original : RectangularPyramid)
  (h_volume : volume original = 40)
  : volume { length := 2 * original.length,
             width := 3 * original.width,
             height := 1.5 * original.height } = 360 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_pyramid_volume_l991_99100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_digits_characterization_l991_99175

def has_at_least_two_distinct_digits (n : ℕ) : Prop :=
  ∀ m : ℕ, m > 0 → (Finset.card (Finset.range 10 ∩ Finset.image (λ d => d % 10) ((m * n).digits 10).reverse.toFinset)) ≥ 2

theorem distinct_digits_characterization (n : ℕ) (hn : n ∈ Finset.range 2010 \ Finset.range 10) :
  has_at_least_two_distinct_digits n ↔
    (10 ∣ n) ∨
    (2 ∣ n ∧ ¬(5 ∣ n) ∧ Nat.factorization n 2 ≥ 4) ∨
    (5 ∣ n ∧ ¬(2 ∣ n) ∧ Nat.factorization n 5 ≥ 2) :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_digits_characterization_l991_99175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l991_99116

theorem trig_identity (θ α β : Real) 
  (h1 : Real.sin θ + Real.cos θ = 2 * Real.sin α) 
  (h2 : Real.sin (2 * θ) = 2 * Real.sin β ^ 2) : 
  Real.cos (2 * β) = 2 * Real.cos (2 * α) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l991_99116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l991_99142

-- Define the function g
def g : ℝ → ℝ := sorry

-- Define the derivative of g
def g' : ℝ → ℝ := sorry

-- Axioms based on the given conditions
axiom g_derivative : ∀ x, HasDerivAt g (g' x) x

axiom g_ratio : ∀ x, g x / g (-x) = Real.exp (2 * x)

axiom g_inequality : ∀ x, x ≥ 0 → g' x > g x

-- Theorem statement
theorem range_of_m (m : ℝ) : 
  g (3 * m - 2) ≥ Real.exp (m - 3) * g (2 * m + 1) ↔ 
  m ≤ 1/5 ∨ m ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l991_99142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_x_given_complex_angle_l991_99158

theorem sin_x_given_complex_angle (x : ℝ) 
  (h1 : 0 < x) (h2 : x < Real.pi) 
  (h3 : Real.sin (x + Real.arccos (4/5)) = Real.sqrt 3/2) : 
  Real.sin x = (4*Real.sqrt 3 - 3)/10 ∨ Real.sin x = (4*Real.sqrt 3 + 3)/10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_x_given_complex_angle_l991_99158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l991_99174

-- Define the arithmetic sequence and its sum
noncomputable def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d
noncomputable def S (a₁ d : ℝ) (n : ℕ) : ℝ := (n * (2 * a₁ + (n - 1) * d)) / 2

-- State the theorem
theorem arithmetic_sequence_properties 
  (a₁ d : ℝ) -- First term and common difference
  (h : S a₁ d 6 > S a₁ d 7 ∧ S a₁ d 7 > S a₁ d 5) : -- Given condition
  d < 0 ∧ 
  S a₁ d 11 > 0 ∧ 
  S a₁ d 12 > 0 ∧ 
  |arithmetic_sequence a₁ d 6| > |arithmetic_sequence a₁ d 7| := 
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l991_99174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_ratio_l991_99151

/-- A regular octahedron with side length s -/
structure RegularOctahedron where
  s : ℝ
  s_pos : s > 0

/-- A regular tetrahedron formed by six vertices of the octahedron -/
structure RegularTetrahedron where
  s : ℝ
  s_pos : s > 0

/-- Surface area of a regular tetrahedron -/
noncomputable def surfaceAreaTetrahedron (t : RegularTetrahedron) : ℝ :=
  Real.sqrt 3 * t.s^2

/-- Surface area of a regular octahedron -/
noncomputable def surfaceAreaOctahedron (o : RegularOctahedron) : ℝ :=
  2 * Real.sqrt 3 * o.s^2

/-- Theorem stating the ratio of surface areas -/
theorem surface_area_ratio 
  (o : RegularOctahedron) 
  (t : RegularTetrahedron) 
  (h : t.s = o.s) : 
  surfaceAreaOctahedron o / surfaceAreaTetrahedron t = 2 := by
  sorry

#check surface_area_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_ratio_l991_99151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_perimeter_line_equation_l991_99124

-- Define the curve C₁ in polar coordinates
noncomputable def C₁ (θ : ℝ) : ℝ × ℝ := (2 * Real.cos θ, 2 * Real.sin θ)

-- Define the curve C₂ derived from C₁
noncomputable def C₂ (θ : ℝ) : ℝ × ℝ := (2 * Real.cos θ, Real.sin θ)

-- Define a line passing through the origin
def line_through_origin (m : ℝ) (x : ℝ) : ℝ := m * x

-- Define the perimeter of quadrilateral ABCD
noncomputable def perimeter (α : ℝ) : ℝ := 8 * Real.cos α + 4 * Real.sin α

theorem max_perimeter_line_equation :
  ∃ α : ℝ,
  (∀ β : ℝ, perimeter α ≥ perimeter β) ∧
  ∃ A : ℝ × ℝ,
    A.1 > 0 ∧ A.2 > 0 ∧
    A ∈ Set.range C₂ ∧
    A.2 = line_through_origin (1/4) A.1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_perimeter_line_equation_l991_99124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_group_frequency_l991_99143

theorem fourth_group_frequency 
  (total_data : ℕ)
  (num_groups : ℕ)
  (first_group_freq : ℕ)
  (second_fifth_sum : ℕ)
  (third_group_prop : ℚ)
  (h1 : total_data = 50)
  (h2 : num_groups = 5)
  (h3 : first_group_freq = 6)
  (h4 : second_fifth_sum = 20)
  (h5 : third_group_prop = 1/5) :
  total_data - first_group_freq - second_fifth_sum - (third_group_prop * ↑total_data).floor = 14 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_group_frequency_l991_99143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_inequality_l991_99147

-- Define factorial
def factorial (n : ℕ) : ℕ := Nat.factorial n

-- State the theorem
theorem factorial_inequality :
  (factorial 99)^(factorial 100) * (factorial 100)^(factorial 99) > factorial (factorial 100) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_inequality_l991_99147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_typical_discount_is_40_percent_l991_99144

/-- Represents the typical discount percentage applied to replica jerseys -/
def typical_discount : ℝ := sorry

/-- The list price of a replica jersey -/
def list_price : ℝ := 80

/-- The additional discount percentage during the summer sale -/
def summer_sale_discount : ℝ := 20

/-- The lowest possible sale price as a percentage of the list price -/
def lowest_sale_price_percent : ℝ := 40

/-- Theorem stating that the typical discount is 40% given the conditions -/
theorem typical_discount_is_40_percent :
  (100 - typical_discount - summer_sale_discount) * list_price / 100 = lowest_sale_price_percent * list_price / 100 →
  typical_discount = 40 := by
  sorry

/-- The range of the typical discount -/
def typical_discount_range : Set ℝ := {x | 40 ≤ x ∧ x < 60}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_typical_discount_is_40_percent_l991_99144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_convex_sequence_l991_99176

def convex_sequence (b : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, b (n + 1) = b n + b (n + 2)

theorem sum_of_convex_sequence
  (b : ℕ → ℤ)
  (h_convex : convex_sequence b)
  (h_b1 : b 0 = 1)
  (h_b2 : b 1 = -2) :
  (Finset.range 2014).sum b = 339 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_convex_sequence_l991_99176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l991_99194

noncomputable def f (x : ℝ) := Real.sin (4 * x) ^ 2

theorem f_properties :
  (∀ x, f x = f (-x)) ∧  -- f is even
  (∀ y, y > 0 ∧ (∀ x, f (x + y) = f x) → y ≥ π / 4) ∧  -- minimum positive period is at least π/4
  (∀ x, f (x + π / 4) = f x)  -- π/4 is indeed a period
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l991_99194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_rope_length_l991_99156

/-- Represents a rope with its length and knot loss --/
structure Rope where
  length : ℝ
  knotLoss : ℝ

/-- Represents the set of ropes Tony has --/
def tonyRopes : List Rope := [
  ⟨8, 1.2⟩,
  ⟨20, 1.5⟩,
  ⟨2, 1⟩,
  ⟨2, 1⟩,
  ⟨2, 1⟩,
  ⟨7, 0.8⟩,
  ⟨5, 1.2⟩,
  ⟨5, 1.2⟩
]

/-- The maximum number of knots Tony can tie --/
def maxKnots : ℕ := 3

/-- Function to calculate the length of a rope after tying knots --/
def tiedRopeLength (r : Rope) (knots : ℕ) : ℝ :=
  r.length - r.knotLoss * (knots : ℝ)

/-- Theorem stating that the maximum rope length Tony can achieve is 25.5 feet --/
theorem max_rope_length :
  ∃ (tiedRopes : List Rope),
    tiedRopes.length ≤ tonyRopes.length ∧
    (∀ r ∈ tiedRopes, r ∈ tonyRopes) ∧
    (tiedRopes.map (λ r => r.length)).sum -
      (tiedRopes.map (λ r => r.knotLoss)).sum * ((tiedRopes.length - 1) : ℝ) ≤ 25.5 ∧
    tiedRopes.length - 1 ≤ maxKnots := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_rope_length_l991_99156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parallel_no_minimum_implies_a_nonnegative_l991_99101

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x - a) / ((x + a)^2)

-- Statement for part (1)
theorem tangent_line_parallel (a : ℝ) :
  (∃ k, ∀ x, (deriv (f a)) 0 * x + f a 0 = 3 * x + k) → a = 1 ∨ a = -1 :=
sorry

-- Statement for part (2)
theorem no_minimum_implies_a_nonnegative (a : ℝ) :
  (∀ x₁, x₁ ≠ -a → ∃ x₂, x₂ ≠ -a ∧ f a x₂ < f a x₁) → a ≥ 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parallel_no_minimum_implies_a_nonnegative_l991_99101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clothing_factory_prefers_mode_l991_99113

/-- Represents the statistical measures of interest -/
inductive StatMeasure
| Mean
| Median
| Mode
| Variance

/-- Represents a clothing factory -/
structure ClothingFactory where
  name : String

/-- Represents a group of students -/
structure StudentGroup where
  size : Nat
  sample_size : Nat

/-- Represents the height measurements of students -/
def HeightData := List Float

/-- Defines the concept of usefulness for a statistical measure in clothing production -/
def is_more_useful_for_clothing (m1 m2 : StatMeasure) (f : ClothingFactory) (g : StudentGroup) (data : HeightData) : Prop :=
  sorry -- This definition would be implemented based on specific criteria

/-- Defines the concept of most useful statistical measure -/
def is_most_useful (m : StatMeasure) (f : ClothingFactory) (g : StudentGroup) (data : HeightData) : Prop :=
  ∀ other : StatMeasure, m ≠ other → is_more_useful_for_clothing m other f g data

/-- Axiom: The mode is the most useful measure for a clothing factory -/
axiom mode_most_useful (f : ClothingFactory) (g : StudentGroup) (data : HeightData) :
  g.size > 11000 → g.sample_size = 200 → is_most_useful StatMeasure.Mode f g data

/-- Theorem: The mode is the most useful statistical measure for the clothing factory -/
theorem clothing_factory_prefers_mode (f : ClothingFactory) (g : StudentGroup) (data : HeightData) :
  g.size > 11000 → g.sample_size = 200 → is_most_useful StatMeasure.Mode f g data :=
by
  exact mode_most_useful f g data


end NUMINAMATH_CALUDE_ERRORFEEDBACK_clothing_factory_prefers_mode_l991_99113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_identification_l991_99129

-- Define the function
noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + φ)

-- State the theorem
theorem function_identification (ω φ : ℝ) :
  ω > 0 ∧ 
  |φ| < Real.pi / 2 ∧
  f ω φ 0 = 1 ∧
  f ω φ (11 * Real.pi / 12) = 0 ∧
  11 * Real.pi / 12 < 2 * Real.pi / ω ∧
  2 * Real.pi / ω < 11 * Real.pi / 9 →
  ∀ x, f ω φ x = 2 * Real.sin (2 * x + Real.pi / 6) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_identification_l991_99129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identities_l991_99125

noncomputable def angle_alpha (a : ℝ) (h : a < 0) := Real.arctan (4 * a / (3 * a))

theorem trig_identities (a : ℝ) (h : a < 0) :
  let α := angle_alpha a h
  (Real.sin α = -4/5) ∧ (Real.tan (π - 2*α) = 24/7) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identities_l991_99125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_minus_point_nine_repeating_equals_zero_l991_99135

/-- The repeating decimal 0.999... -/
noncomputable def repeating_decimal_nine : ℝ := 
  ∑' n, 9 * (1/10)^(n+1)

/-- Theorem stating that 1 - 0.999... = 0 -/
theorem one_minus_point_nine_repeating_equals_zero : 
  1 - repeating_decimal_nine = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_minus_point_nine_repeating_equals_zero_l991_99135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_lily_half_coverage_l991_99166

/-- Represents the area of water lilies as a fraction of the lake's total area -/
def WaterLilyArea : ℕ → ℚ := sorry

/-- The number of days it takes for water lilies to cover the entire lake -/
def TotalDays : ℕ := 48

theorem water_lily_half_coverage :
  (∀ n : ℕ, WaterLilyArea (n + 1) = 2 * WaterLilyArea n) →  -- Area doubles every day
  WaterLilyArea TotalDays = 1 →                             -- Full coverage after TotalDays
  WaterLilyArea (TotalDays - 1) = 1 / 2 :=                  -- Half coverage one day before
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_lily_half_coverage_l991_99166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_phi_is_one_half_l991_99163

-- Define the dilation matrix D
def D (m : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![m, 0; 0, m]

-- Define the rotation matrix R
noncomputable def R (φ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![Real.cos φ, -Real.sin φ; Real.sin φ, Real.cos φ]

-- Theorem statement
theorem tan_phi_is_one_half
  (m : ℝ)
  (φ : ℝ)
  (h_m : m > 0)
  (h_matrix : R φ * D m = !![10, -5; 5, 10]) :
  Real.tan φ = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_phi_is_one_half_l991_99163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l991_99111

noncomputable def f (x : ℝ) : ℝ := 
  if x ≤ 0 then -x^2 else Real.log (x + 1) / Real.log 2

def g (a x : ℝ) : ℝ := a * x^2 - 2 * a * x + 1

theorem range_of_a (a : ℝ) :
  (∀ x₁ ∈ Set.Icc (-2) 9, ∃ x₂ ∈ Set.Icc (-2) 2, g a x₂ = f x₁) ↔
  a ∈ Set.Iic (-5/8) ∪ Set.Ici 5 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l991_99111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_Q_has_two_solutions_l991_99127

/-- The polynomial Q(x) as defined in the problem -/
noncomputable def Q (x : ℝ) : ℝ := 
  Real.cos x + 2 * Real.sin x - Real.cos (2 * x) - 2 * Real.sin (2 * x) + Real.cos (3 * x) + 2 * Real.sin (3 * x)

/-- The theorem stating that Q(x) = 0 has exactly two solutions in [0, 2π) -/
theorem Q_has_two_solutions : 
  ∃! (s : Finset ℝ), s.card = 2 ∧ (∀ x ∈ s, 0 ≤ x ∧ x < 2 * Real.pi ∧ Q x = 0) ∧ 
  (∀ x, 0 ≤ x → x < 2 * Real.pi → Q x = 0 → x ∈ s) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_Q_has_two_solutions_l991_99127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_k_l991_99186

theorem existence_of_k : ∃ k : ℕ, ∀ n : ℕ, ∃ (a b : ℕ), a * b = k * 2^n + 1 ∧ a > 1 ∧ b > 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_k_l991_99186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_two_zeros_a_in_range_l991_99196

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then Real.exp (x * Real.log 2) - a else 2*x - 1

theorem f_two_zeros_a_in_range (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ f a x = 0 ∧ f a y = 0) ∧
  (∀ z w v : ℝ, z ≠ w ∧ z ≠ v ∧ w ≠ v → ¬(f a z = 0 ∧ f a w = 0 ∧ f a v = 0)) →
  0 < a ∧ a ≤ 1 := by
  sorry

#check f_two_zeros_a_in_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_two_zeros_a_in_range_l991_99196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_and_increasing_l991_99114

noncomputable def f (x : ℝ) := |Real.sin (Real.pi + x)|

theorem f_is_even_and_increasing :
  (∀ x, f (-x) = f x) ∧
  (∀ x y, x ∈ Set.Icc 0 1 → y ∈ Set.Icc 0 1 → x ≤ y → f x ≤ f y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_and_increasing_l991_99114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_reciprocal_sum_l991_99160

/-- A plane intersecting the coordinate axes -/
structure IntersectingPlane where
  /-- x-intercept -/
  α : ℝ
  /-- y-intercept -/
  β : ℝ
  /-- z-intercept -/
  γ : ℝ
  /-- Condition that the plane is 2 units away from (1,1,1) -/
  distance_condition : 1 / α^2 + 1 / β^2 + 1 / γ^2 = 1
  /-- Condition that intercepts are distinct from (1,1,1) -/
  distinct_intercepts : α ≠ 1 ∧ β ≠ 1 ∧ γ ≠ 1

/-- The centroid of the triangle formed by the intercepts -/
noncomputable def centroid (plane : IntersectingPlane) : ℝ × ℝ × ℝ :=
  (plane.α / 3, plane.β / 3, plane.γ / 3)

/-- The main theorem -/
theorem centroid_reciprocal_sum (plane : IntersectingPlane) :
  let (p, q, r) := centroid plane
  1 / p^2 + 1 / q^2 + 1 / r^2 = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_reciprocal_sum_l991_99160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_possible_n_values_l991_99132

/-- Represents a cubic polynomial with specific properties -/
structure SpecialCubic where
  m : ℤ
  n : ℤ
  zero1 : ℝ
  zero2 : ℝ
  zero3 : ℤ
  h1 : zero1 > 0 ∧ zero2 > 0 ∧ zero3 > 0
  h2 : zero1 ≠ zero2 ∧ zero1 ≠ zero3 ∧ zero2 ≠ zero3
  h3 : zero3 = 2 * (zero1 + zero2)
  h4 : ∀ x : ℝ, x^3 - 2003*x^2 + m*x + n = (x - zero1) * (x - zero2) * (x - ↑zero3)

/-- The number of possible values for n is 160800 -/
theorem count_possible_n_values : 
  ∃ S : Finset ℤ, (∀ n ∈ S, ∃ m : ℤ, ∃ p : SpecialCubic, p.n = n) ∧ Finset.card S = 160800 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_possible_n_values_l991_99132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_teachers_with_no_conditions_l991_99148

def total_teachers : ℕ := 150
def high_blood_pressure : ℕ := 90
def heart_trouble : ℕ := 60
def diabetes : ℕ := 50
def high_blood_pressure_and_heart_trouble : ℕ := 30
def high_blood_pressure_and_diabetes : ℕ := 20
def heart_trouble_and_diabetes : ℕ := 10
def all_three_conditions : ℕ := 5

theorem teachers_with_no_conditions (ε : ℝ) :
  ε > 0 →
  ∃ (teachers_with_no_conditions : ℕ),
    |((teachers_with_no_conditions : ℝ) / total_teachers * 100) - 3.33| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_teachers_with_no_conditions_l991_99148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_squared_l991_99189

-- Define the expression
noncomputable def expression (x : ℝ) : ℝ := (1/x + 2) * (1 - x)^4

-- Theorem statement
theorem coefficient_of_x_squared :
  ∃ (f : ℝ → ℝ) (a b c d e : ℝ),
    (∀ x, x ≠ 0 → expression x = a / x + b + c * x + d * x^2 + e * x^3 + f x) ∧
    d = 8 := by
  sorry

#eval "Proof completed"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_squared_l991_99189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_bisector_sum_l991_99149

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Predicate that returns true if the line l bisects the area of triangle PQR -/
def area_bisector (P Q R : Point) (l : Line) : Prop :=
  sorry

/-- Given three points P, Q, R forming a triangle, and a line through Q that bisects
    the area of the triangle, the sum of the slope and y-intercept of this line is -20/3 -/
theorem triangle_bisector_sum (P Q R : Point) (l : Line) :
  P.x = 0 ∧ P.y = 10 ∧
  Q.x = 3 ∧ Q.y = 0 ∧
  R.x = 9 ∧ R.y = 0 ∧
  (l.slope * Q.x + l.intercept = Q.y) ∧
  (area_bisector P Q R l) →
  l.slope + l.intercept = -20/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_bisector_sum_l991_99149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_faulty_key_theorem_l991_99126

/-- Represents a digit key on a keypad -/
inductive Digit : Type
  | zero | one | two | three | four | five | six | seven | eight | nine
  deriving BEq, Repr

/-- Represents the result of pressing a key -/
inductive KeyPress
  | success
  | failure
  deriving BEq, Repr

/-- Represents a sequence of key presses -/
def KeySequence := List KeyPress

/-- The intended sequence of digits -/
def IntendedSequence := List Digit

/-- The actually registered sequence of digits -/
def RegisteredSequence := List Digit

/-- Checks if a key press pattern matches the faulty key description -/
def isFaultyKeyPattern (pattern : KeySequence) : Prop :=
  pattern.length ≥ 5 ∧
  pattern.get? 0 = some KeyPress.failure ∧
  pattern.get? 2 = some KeyPress.failure ∧
  pattern.get? 4 = some KeyPress.failure ∧
  pattern.get? 1 = some KeyPress.success ∧
  pattern.get? 3 = some KeyPress.success

/-- The main theorem -/
theorem faulty_key_theorem 
  (intended : IntendedSequence) 
  (registered : RegisteredSequence) 
  (h1 : intended.length = 10)
  (h2 : registered.length = 7)
  (h3 : ∃ (d : Digit) (pattern : KeySequence), 
        isFaultyKeyPattern pattern ∧ 
        intended.count d = pattern.length) :
  ∃ (d : Digit), d = Digit.seven ∨ d = Digit.nine := by
  sorry

#check faulty_key_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_faulty_key_theorem_l991_99126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_quadrilateral_area_l991_99187

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  point : Point

/-- Rotates a line around its point by an angle α -/
def Line.rotate (l : Line) (α : ℝ) : Line := sorry

/-- The quadrilateral formed by the four rotating lines -/
def Quadrilateral (ℓA ℓB ℓC ℓD : Line) : Set Point := sorry

/-- The area of the quadrilateral -/
noncomputable def area (quad : Set Point) : ℝ := sorry

theorem max_quadrilateral_area :
  let A : Point := ⟨0, 0⟩
  let B : Point := ⟨8, 0⟩
  let C : Point := ⟨15, 0⟩
  let D : Point := ⟨20, 0⟩
  let ℓA : Line := ⟨1, A⟩
  let ℓB : Line := ⟨0, B⟩  -- Using 0 instead of ∞ for vertical line
  let ℓC : Line := ⟨-1, C⟩
  let ℓD : Line := ⟨2, D⟩
  ∀ α : ℝ, area (Quadrilateral (ℓA.rotate α) (ℓB.rotate α) (ℓC.rotate α) (ℓD.rotate α)) ≤ 110.5 := by
  sorry

#check max_quadrilateral_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_quadrilateral_area_l991_99187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_equation_l991_99171

/-- A line parallel to 3x + 4y + 12 = 0 that forms a triangle with the coordinate axes -/
structure ParallelLine where
  k : ℝ
  equation : ℝ → ℝ → Prop := fun x y ↦ 3 * x + 4 * y + k = 0

/-- The area of the triangle formed by a line and the coordinate axes -/
noncomputable def triangleArea (l : ParallelLine) : ℝ :=
  abs (l.k ^ 2) / 24

theorem parallel_line_equation (l : ParallelLine) (h : triangleArea l = 24) :
  l.equation = fun x y ↦ 3 * x + 4 * y + 24 = 0 ∨
  l.equation = fun x y ↦ 3 * x + 4 * y - 24 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_equation_l991_99171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_chord_length_l991_99110

/-- The line l: 3x - y - 6 = 0 -/
def line (x y : ℝ) : Prop := 3 * x - y - 6 = 0

/-- The circle: (x-1)^2 + (y-2)^2 = 5 -/
def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 5

/-- The length of the chord AB formed by the intersection of the line and the circle -/
noncomputable def chord_length : ℝ := Real.sqrt 10

theorem intersection_chord_length :
  ∃ (A B : ℝ × ℝ),
    line A.1 A.2 ∧ line B.1 B.2 ∧
    circle_eq A.1 A.2 ∧ circle_eq B.1 B.2 ∧
    A ≠ B ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = chord_length :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_chord_length_l991_99110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_third_quadrant_l991_99139

theorem angle_in_third_quadrant (θ : Real) 
  (h1 : Real.cos θ < 0) (h2 : Real.tan θ > 0) : 
  θ % (2 * Real.pi) ∈ Set.Ioo Real.pi (3 * Real.pi / 2) := by
  sorry

#check angle_in_third_quadrant

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_third_quadrant_l991_99139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_angle_C_l991_99193

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.c = 5 ∧ Real.cos t.B = 3/5 ∧ t.a * t.c * Real.cos t.B = -21

-- Theorem for the area of the triangle
theorem triangle_area (t : Triangle) (h : triangle_conditions t) : 
  (1/2) * t.a * t.c * Real.sin t.B = 14 := by
  sorry

-- Theorem for angle C
theorem angle_C (t : Triangle) (h : triangle_conditions t) :
  t.C = π/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_angle_C_l991_99193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_rope_length_l991_99155

noncomputable def rope_length (n : ℕ) : ℝ :=
  (1 / 4 : ℝ) ^ n

theorem final_rope_length :
  rope_length 100 = (1 / 4 : ℝ) ^ 100 := by
  rfl

#check final_rope_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_rope_length_l991_99155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_roll_length_l991_99152

/-- Represents the parameters of a paper roll -/
structure PaperRoll where
  core_diameter : ℝ
  paper_width : ℝ
  num_wraps : ℕ
  final_diameter : ℝ

/-- Calculates the total length of paper in a roll -/
noncomputable def total_paper_length (roll : PaperRoll) : ℝ :=
  let diameter_increase := (roll.final_diameter - roll.core_diameter) / roll.num_wraps
  let sum_of_diameters := (roll.num_wraps : ℝ) * roll.core_diameter + 
                          diameter_increase * (roll.num_wraps - 1) * roll.num_wraps / 2
  Real.pi * sum_of_diameters

/-- Theorem stating the total length of paper in the given roll -/
theorem paper_roll_length : 
  let roll := PaperRoll.mk 4 4 800 16
  total_paper_length roll = 79.94 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_roll_length_l991_99152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_PQ_is_minus_one_plus_two_i_l991_99182

def complex_to_point (z : ℂ) : ℝ × ℝ := (z.re, z.im)

def vector_between_points (p q : ℝ × ℝ) : ℝ × ℝ :=
  (q.1 - p.1, q.2 - p.2)

def vector_to_complex (v : ℝ × ℝ) : ℂ :=
  Complex.ofReal v.1 + Complex.I * Complex.ofReal v.2

theorem vector_PQ_is_minus_one_plus_two_i :
  let z1 : ℂ := 3 + Complex.I
  let z2 : ℂ := 2 + 3 * Complex.I
  let p := complex_to_point z1
  let q := complex_to_point z2
  let pq := vector_between_points p q
  vector_to_complex pq = -1 + 2 * Complex.I := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_PQ_is_minus_one_plus_two_i_l991_99182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_two_eq_two_fifths_l991_99153

noncomputable def f (t : ℝ) : ℝ := (t^2 - t) / (t^2 + 1)

theorem f_of_two_eq_two_fifths :
  (∀ x : ℝ, f (Real.tan x) = Real.sin x * Real.sin x - Real.sin x * Real.cos x) →
  f 2 = 2/5 := by
  intro h
  -- The proof steps would go here
  sorry

#check f_of_two_eq_two_fifths

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_two_eq_two_fifths_l991_99153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_apple_profit_l991_99130

/-- Calculates the total percentage profit for a shopkeeper selling apples -/
theorem shopkeeper_apple_profit 
  (total_apples : ℝ) 
  (first_portion : ℝ) 
  (second_portion : ℝ) 
  (profit_rate : ℝ) 
  (h1 : total_apples = 280)
  (h2 : first_portion = 0.4)
  (h3 : second_portion = 0.6)
  (h4 : first_portion + second_portion = 1)
  (h5 : profit_rate = 0.3)
  : (((1 + profit_rate) * total_apples - total_apples) / total_apples) * 100 = 30 := by
  sorry

-- Remove the #eval line as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_apple_profit_l991_99130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_steiner_ellipse_equation_l991_99159

/-- Represents a point in barycentric coordinates -/
structure BarycentricPoint where
  α : ℝ
  β : ℝ
  γ : ℝ
  sum_to_one : α + β + γ = 1

/-- Represents a triangle -/
structure Triangle where
  A : BarycentricPoint
  B : BarycentricPoint
  C : BarycentricPoint

/-- Represents the equation of an ellipse in barycentric coordinates -/
def is_steiner_ellipse (p : BarycentricPoint) : Prop :=
  p.β * p.γ + p.α * p.γ + p.α * p.β = 0

/-- Represents the circumscribed Steiner ellipse of a triangle -/
def circumscribed_steiner_ellipse (t : Triangle) : Set BarycentricPoint :=
  {p : BarycentricPoint | is_steiner_ellipse p}

/-- The theorem stating that the given equation represents the circumscribed Steiner ellipse -/
theorem steiner_ellipse_equation (t : Triangle) :
    ∀ p : BarycentricPoint, is_steiner_ellipse p ↔ p ∈ circumscribed_steiner_ellipse t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_steiner_ellipse_equation_l991_99159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l991_99107

open Set Real

def A : Set ℝ := {x | (3*x - 1) / (x - 2) ≤ 1}

def B (a : ℝ) : Set ℝ := {x | x^2 - (a + 2)*x + 2*a < 0}

theorem range_of_a :
  ∀ a : ℝ, (A ⊂ B a ∧ A ≠ B a) ↔ a ∈ Iio (-1/2) := by
  sorry

#check range_of_a

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l991_99107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_sum_l991_99146

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1
  ratio_condition : a 11 / a 10 < -1
  has_max_sum : ∃ N, ∀ n > N, (Finset.range n).sum a ≤ (Finset.range N).sum a

/-- The sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  (Finset.range n).sum seq.a

/-- The theorem stating that 20 is the smallest n for which the sum is positive -/
theorem smallest_positive_sum (seq : ArithmeticSequence) :
  (∀ k < 20, sum_n seq k ≤ 0) ∧ sum_n seq 20 > 0 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_sum_l991_99146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_condition_implies_right_angle_l991_99131

noncomputable section

open Real

/-- Triangle ABC with vectors BA and BC -/
structure Triangle :=
  (A B C : ℝ × ℝ)
  (BA : ℝ × ℝ := (B.1 - A.1, B.2 - A.2))
  (BC : ℝ × ℝ := (C.1 - B.1, C.2 - B.2))
  (AC : ℝ × ℝ := (C.1 - A.1, C.2 - A.2))

/-- The norm (length) of a 2D vector -/
def vecNorm (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

/-- The angle at vertex C of the triangle -/
noncomputable def angle_C (triangle : Triangle) : ℝ := sorry

/-- The condition that for any t ≠ 1, |BA - tBC| > |AC| -/
def condition (triangle : Triangle) : Prop :=
  ∀ t : ℝ, t ≠ 1 → vecNorm (triangle.BA.1 - t * triangle.BC.1, triangle.BA.2 - t * triangle.BC.2) > vecNorm triangle.AC

/-- Theorem: If the condition holds, then angle C is a right angle -/
theorem condition_implies_right_angle (triangle : Triangle) :
  condition triangle → angle_C triangle = π / 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_condition_implies_right_angle_l991_99131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_sine_implies_zero_cosine_l991_99199

open Real

theorem symmetric_sine_implies_zero_cosine (ω φ : ℝ) :
  (∀ x, 3 * sin (ω * (π/6 + x) + φ) = 3 * sin (ω * (π/6 - x) + φ)) →
  3 * cos (ω * π/6 + φ) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_sine_implies_zero_cosine_l991_99199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_l991_99170

theorem expression_simplification (x : ℝ) (h : 0 < x ∧ x ≤ 1) :
  (2 * x^2 / (9 + 18*x + 9*x^2))^(1/3) *
  ((x + 1) * (1 - x)^(1/3) / x)^(1/2) *
  (3 * (1 - x^2)^(1/2) / (2 * x * x^(1/2)))^(1/3) =
  ((1 - x) / (3 * x))^(1/3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_l991_99170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_line_theorem_l991_99117

/-- The projection of vector v onto vector u -/
noncomputable def proj (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := u.1 * v.1 + u.2 * v.2
  let norm_squared := u.1 * u.1 + u.2 * u.2
  (dot_product / norm_squared • u.1, dot_product / norm_squared • u.2)

/-- The theorem stating that vectors satisfying the projection condition lie on the specified line -/
theorem projection_line_theorem (v : ℝ × ℝ) :
  proj (7, 3) v = (-7/2, -3/2) →
  v.2 = -7/3 * v.1 - 29/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_line_theorem_l991_99117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_asymptote_of_f_l991_99173

noncomputable def f (x : ℝ) : ℝ := (3 * x^3 + 2 * x^2 + 5 * x + 4) / (2 * x + 3)

noncomputable def g (x : ℝ) : ℝ := (3/2) * x^2 - (1/4) * x + 49/16

/-- Theorem: The oblique asymptote of f(x) is g(x) -/
theorem oblique_asymptote_of_f : 
  ∀ ε > 0, ∃ M, ∀ x > M, |f x - g x| < ε := by
  sorry

#check oblique_asymptote_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_asymptote_of_f_l991_99173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotatedSemicircleAreaTheorem_l991_99106

/-- The area of a figure formed by rotating a semicircle of radius R about one of its ends by an angle of 30 degrees -/
noncomputable def rotatedSemicircleArea (R : ℝ) : ℝ := (Real.pi * R^2) / 3

/-- Theorem stating that the area of the rotated semicircle figure is (π * R^2) / 3 -/
theorem rotatedSemicircleAreaTheorem (R : ℝ) (h : R > 0) :
  rotatedSemicircleArea R = (Real.pi * R^2) / 3 := by
  sorry

#check rotatedSemicircleAreaTheorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotatedSemicircleAreaTheorem_l991_99106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tennis_tournament_participant_count_l991_99134

/-- Represents a valid number of participants in a tennis tournament where each participant
    faces every other participant exactly once in pair matches (2 vs 2). -/
def ValidParticipantCount (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 8 * k + 1

/-- Proves that for a tennis tournament with the given conditions,
    the number of participants must be of the form 8k + 1, where k is a natural number. -/
theorem tennis_tournament_participant_count (n : ℕ) :
  (∀ i j : Fin n, i ≠ j → ∃! m : Fin (n * (n - 1) / 4),
    (∃ p q : Fin n, p ≠ q ∧ p ≠ i ∧ p ≠ j ∧ q ≠ i ∧ q ≠ j ∧
      ({i, j, p, q} : Finset (Fin n)).card = 4)) →
  ValidParticipantCount n :=
by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tennis_tournament_participant_count_l991_99134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_zero_sum_88_l991_99118

theorem determinant_zero_sum_88 (a b : ℝ) (h1 : a ≠ b) (h2 : Matrix.det
  ![![1, 6, 16],
    ![4, a, b],
    ![4, b, a]] = 0) :
  a + b = 88 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_zero_sum_88_l991_99118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_polygon_area_l991_99137

-- Define the basic shapes
noncomputable def unit_square_area : ℝ := 1
noncomputable def right_triangle_area : ℝ := 1 / 2
noncomputable def rectangle_area : ℝ := 2

-- Define the polygons
noncomputable def polygon_A : ℝ := 3 * unit_square_area + 2 * right_triangle_area
noncomputable def polygon_B : ℝ := 2 * unit_square_area + 4 * right_triangle_area
noncomputable def polygon_C : ℝ := 4 * unit_square_area + 1 * rectangle_area
noncomputable def polygon_D : ℝ := 3 * rectangle_area
noncomputable def polygon_E : ℝ := 2 * unit_square_area + 2 * right_triangle_area + 2 * rectangle_area

-- Define the list of polygon areas
noncomputable def polygon_areas : List ℝ := [polygon_A, polygon_B, polygon_C, polygon_D, polygon_E]

-- Theorem statement
theorem largest_polygon_area :
  List.maximum polygon_areas = some 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_polygon_area_l991_99137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_radical_fixed_point_f_contractive_infinite_nested_radical_equals_43_l991_99122

/-- The infinite nested radical function f(x) = √(86 + 41x) -/
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (86 + 41 * x)

/-- The limit of the infinite nested radical -/
def nested_radical_limit : ℝ := 43

/-- Theorem stating that f(nested_radical_limit) = nested_radical_limit -/
theorem nested_radical_fixed_point :
  f nested_radical_limit = nested_radical_limit :=
by sorry

/-- Theorem stating that f is contractive for x > nested_radical_limit -/
theorem f_contractive (x : ℝ) (h : x > nested_radical_limit) :
  f x < x :=
by sorry

/-- The main theorem stating that the infinite nested radical equals 43 -/
theorem infinite_nested_radical_equals_43 :
  ∃ (s : ℕ → ℝ), (∀ n, s (n + 1) = f (s n)) ∧
                 (∀ ε > 0, ∃ N, ∀ n ≥ N, |s n - nested_radical_limit| < ε) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_radical_fixed_point_f_contractive_infinite_nested_radical_equals_43_l991_99122
