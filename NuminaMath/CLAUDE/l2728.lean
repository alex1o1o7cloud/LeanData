import Mathlib

namespace NUMINAMATH_CALUDE_integral_problem_l2728_272883

theorem integral_problem : ∫ x in (0)..(2 * Real.arctan (1/2)), (1 - Real.sin x) / (Real.cos x * (1 + Real.cos x)) = 2 * Real.log (3/2) - 1/2 := by
  sorry

end NUMINAMATH_CALUDE_integral_problem_l2728_272883


namespace NUMINAMATH_CALUDE_fourth_power_one_fourth_equals_decimal_l2728_272881

theorem fourth_power_one_fourth_equals_decimal : (1 / 4 : ℚ) ^ 4 = 390625 / 100000000 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_one_fourth_equals_decimal_l2728_272881


namespace NUMINAMATH_CALUDE_batsman_average_l2728_272824

theorem batsman_average (previous_total : ℕ) (previous_average : ℚ) : 
  previous_total = 10 * previous_average →
  (previous_total + 69) / 11 = previous_average + 1 →
  (previous_total + 69) / 11 = 59 := by
sorry

end NUMINAMATH_CALUDE_batsman_average_l2728_272824


namespace NUMINAMATH_CALUDE_tangent_line_touches_both_curves_l2728_272826

noncomputable def curve1 (x : ℝ) : ℝ := x^2 - Real.log x

noncomputable def curve2 (a x : ℝ) : ℝ := a * x^2 + (a + 2) * x + 1

noncomputable def tangent_line (x : ℝ) : ℝ := x

theorem tangent_line_touches_both_curves (a : ℝ) :
  (∀ x, x > 0 → curve1 x ≥ tangent_line x) ∧
  (curve1 1 = tangent_line 1) ∧
  (∀ x, curve2 a x ≥ tangent_line x) ∧
  (∃ x, curve2 a x = tangent_line x) →
  a = 1 := by sorry

end NUMINAMATH_CALUDE_tangent_line_touches_both_curves_l2728_272826


namespace NUMINAMATH_CALUDE_min_value_a_l2728_272890

theorem min_value_a (a : ℝ) (h : a > 0) : 
  (∀ x y : ℝ, x > 0 → y > 0 → 1/x + a/y ≥ 16/(x+y)) → a ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_a_l2728_272890


namespace NUMINAMATH_CALUDE_hcf_problem_l2728_272860

theorem hcf_problem (a b : ℕ+) (h1 : a * b = 62216) (h2 : Nat.lcm a b = 2828) :
  Nat.gcd a b = 22 := by
  sorry

end NUMINAMATH_CALUDE_hcf_problem_l2728_272860


namespace NUMINAMATH_CALUDE_problem_solution_l2728_272851

theorem problem_solution (a b : ℚ) 
  (eq1 : 2 * a + 5 * b = 47)
  (eq2 : 8 * a + 2 * b = 50) :
  3 * a + 3 * b = 73 / 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2728_272851


namespace NUMINAMATH_CALUDE_max_value_of_function_l2728_272836

/-- The function f(x) = -x - 9/x + 18 for x > 0 has a maximum value of 12 -/
theorem max_value_of_function (x : ℝ) (hx : x > 0) :
  ∃ (M : ℝ), M = 12 ∧ ∀ y, y > 0 → -y - 9/y + 18 ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_function_l2728_272836


namespace NUMINAMATH_CALUDE_rectangles_in_4x4_grid_l2728_272885

/-- The number of rectangles in a 4x4 grid -/
def num_rectangles_4x4 : ℕ := 36

/-- The number of ways to choose 2 items from 4 -/
def choose_2_from_4 : ℕ := 6

/-- Theorem: The number of rectangles in a 4x4 grid is 36 -/
theorem rectangles_in_4x4_grid :
  num_rectangles_4x4 = choose_2_from_4 * choose_2_from_4 :=
by sorry

end NUMINAMATH_CALUDE_rectangles_in_4x4_grid_l2728_272885


namespace NUMINAMATH_CALUDE_min_distance_circle_to_line_polar_to_cartesian_line_l2728_272872

/-- Given a line and a circle, find the minimum distance from a point on the circle to the line -/
theorem min_distance_circle_to_line :
  let line := {(x, y) : ℝ × ℝ | x + y = 1}
  let circle := {(x, y) : ℝ × ℝ | ∃ θ : ℝ, x = 2 * Real.cos θ ∧ y = -2 + 2 * Real.sin θ}
  ∃ d : ℝ, d = (3 * Real.sqrt 2) / 2 - 2 ∧
    ∀ p ∈ circle, ∀ q ∈ line, Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≥ d :=
by sorry

/-- The polar equation of the line can be converted to Cartesian form -/
theorem polar_to_cartesian_line :
  ∀ ρ θ : ℝ, ρ * Real.sin (θ + π/4) = Real.sqrt 2 / 2 →
  ∃ x y : ℝ, x + y = 1 ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ :=
by sorry

end NUMINAMATH_CALUDE_min_distance_circle_to_line_polar_to_cartesian_line_l2728_272872


namespace NUMINAMATH_CALUDE_unit_vectors_sum_squares_lower_bound_l2728_272878

theorem unit_vectors_sum_squares_lower_bound 
  (p q r : EuclideanSpace ℝ (Fin 3)) 
  (hp : ‖p‖ = 1) (hq : ‖q‖ = 1) (hr : ‖r‖ = 1) : 
  ‖p + q‖^2 + ‖p + r‖^2 + ‖q + r‖^2 ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_unit_vectors_sum_squares_lower_bound_l2728_272878


namespace NUMINAMATH_CALUDE_similar_polygons_ratio_l2728_272806

theorem similar_polygons_ratio (A₁ A₂ : ℝ) (s₁ s₂ : ℝ) :
  A₁ / A₂ = 9 / 4 →
  s₁ / s₂ = (A₁ / A₂).sqrt →
  s₁ / s₂ = 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_similar_polygons_ratio_l2728_272806


namespace NUMINAMATH_CALUDE_unique_k_for_coplanarity_l2728_272877

-- Define the vector space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]

-- Define the origin and points
variable (O A B C D : V)

-- Define the condition for coplanarity
def coplanar (A B C D : V) : Prop :=
  ∃ (a b c : ℝ), (D - A) = a • (B - A) + b • (C - A) + c • (A - A)

-- State the theorem
theorem unique_k_for_coplanarity :
  ∃! k : ℝ, ∀ (A B C D : V),
    (4 • (A - O) - 3 • (B - O) + 6 • (C - O) + k • (D - O) = 0) →
    coplanar A B C D :=
by sorry

end NUMINAMATH_CALUDE_unique_k_for_coplanarity_l2728_272877


namespace NUMINAMATH_CALUDE_min_value_theorem_l2728_272845

theorem min_value_theorem (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a ≤ b + c) (h2 : b + c ≤ 3 * a) (h3 : 3 * b^2 ≤ a * (a + c)) (h4 : a * (a + c) ≤ 5 * b^2) :
  -18/5 ≤ (b - 2*c) / a ∧ ∃ (a₀ b₀ c₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ 0 < c₀ ∧
    a₀ ≤ b₀ + c₀ ∧ b₀ + c₀ ≤ 3 * a₀ ∧ 3 * b₀^2 ≤ a₀ * (a₀ + c₀) ∧ a₀ * (a₀ + c₀) ≤ 5 * b₀^2 ∧
    (b₀ - 2*c₀) / a₀ = -18/5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2728_272845


namespace NUMINAMATH_CALUDE_smallest_natural_ending_2012_l2728_272858

theorem smallest_natural_ending_2012 : 
  ∃ (n : ℕ), n = 1716 ∧ 
  (∀ (m : ℕ), m < n → (m * 7) % 10000 ≠ 2012) ∧ 
  (n * 7) % 10000 = 2012 := by
sorry

end NUMINAMATH_CALUDE_smallest_natural_ending_2012_l2728_272858


namespace NUMINAMATH_CALUDE_min_value_theorem_l2728_272844

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_geom_mean : Real.sqrt 3 = Real.sqrt (3^a * 3^b)) : 
  ∀ x y, x > 0 → y > 0 → 1/x + 2/y ≥ 3 + 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2728_272844


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l2728_272835

theorem trigonometric_equation_solution (t : ℝ) : 
  2 * (Real.sin t)^4 * (Real.sin (2 * t) - 3) - 2 * (Real.sin t)^2 * (Real.sin (2 * t) - 3) - 1 = 0 ↔ 
  (∃ k : ℤ, t = π/4 * (4 * k + 1)) ∨ 
  (∃ n : ℤ, t = (-1)^n * (1/2 * Real.arcsin (1 - Real.sqrt 3)) + π/2 * n) :=
sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l2728_272835


namespace NUMINAMATH_CALUDE_average_of_remaining_results_l2728_272896

theorem average_of_remaining_results (average_40 : ℝ) (average_all : ℝ) :
  average_40 = 30 →
  average_all = 34.285714285714285 →
  (70 * average_all - 40 * average_40) / 30 = 40 := by
sorry

end NUMINAMATH_CALUDE_average_of_remaining_results_l2728_272896


namespace NUMINAMATH_CALUDE_batsman_average_after_12_innings_l2728_272840

/-- Represents a batsman's cricket performance -/
structure Batsman where
  innings : Nat
  totalRuns : Nat
  averageIncrease : Nat
  lastInningsScore : Nat

/-- Calculates the average score of a batsman -/
def averageScore (b : Batsman) : Rat :=
  b.totalRuns / b.innings

/-- Theorem: A batsman's average after 12 innings is 38, given the conditions -/
theorem batsman_average_after_12_innings 
  (b : Batsman)
  (h1 : b.innings = 12)
  (h2 : b.lastInningsScore = 60)
  (h3 : b.averageIncrease = 2)
  (h4 : averageScore b = averageScore { b with 
    innings := b.innings - 1,
    totalRuns := b.totalRuns - b.lastInningsScore 
  } + b.averageIncrease) :
  averageScore b = 38 := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_after_12_innings_l2728_272840


namespace NUMINAMATH_CALUDE_heart_club_probability_l2728_272830

/-- Represents a standard deck of 52 playing cards -/
structure Deck :=
  (cards : Finset (Nat × Nat))
  (size : cards.card = 52)

/-- Represents the suit of a card -/
inductive Suit
| Hearts | Clubs | Diamonds | Spades

/-- The probability of drawing a heart first and a club second from a standard deck -/
def prob_heart_then_club (d : Deck) : ℚ :=
  (13 : ℚ) / 204

/-- Theorem stating the probability of drawing a heart first and a club second -/
theorem heart_club_probability (d : Deck) :
  prob_heart_then_club d = 13 / 204 := by
  sorry

end NUMINAMATH_CALUDE_heart_club_probability_l2728_272830


namespace NUMINAMATH_CALUDE_a_range_l2728_272837

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 > -a * x - 1 ∧ a ≠ 0

def q (a : ℝ) : Prop := ∀ x y : ℝ, 
  x^2 + y^2 = a^2 → (x + 3)^2 + (y - 4)^2 > 4

-- Define the range of a
def range_a (a : ℝ) : Prop := (a > -3 ∧ a ≤ 0) ∨ (a ≥ 3 ∧ a < 4)

-- State the theorem
theorem a_range : 
  (∀ a : ℝ, (p a ∨ q a) ∧ ¬(p a ∧ q a)) → 
  (∀ a : ℝ, range_a a ↔ (p a ∨ q a) ∧ ¬(p a ∧ q a)) :=
sorry

end NUMINAMATH_CALUDE_a_range_l2728_272837


namespace NUMINAMATH_CALUDE_tangerines_highest_frequency_l2728_272859

/-- Represents the number of boxes for each fruit type -/
def num_boxes_tangerines : ℕ := 5
def num_boxes_apples : ℕ := 3
def num_boxes_pears : ℕ := 4

/-- Represents the number of fruits per box for each fruit type -/
def fruits_per_box_tangerines : ℕ := 30
def fruits_per_box_apples : ℕ := 20
def fruits_per_box_pears : ℕ := 15

/-- Represents the weight of each fruit in grams -/
def weight_tangerine : ℕ := 200
def weight_apple : ℕ := 450
def weight_pear : ℕ := 800

/-- Calculates the total number of fruits for each type -/
def total_tangerines : ℕ := num_boxes_tangerines * fruits_per_box_tangerines
def total_apples : ℕ := num_boxes_apples * fruits_per_box_apples
def total_pears : ℕ := num_boxes_pears * fruits_per_box_pears

/-- Theorem: Tangerines have the highest frequency -/
theorem tangerines_highest_frequency :
  total_tangerines > total_apples ∧ total_tangerines > total_pears :=
sorry

end NUMINAMATH_CALUDE_tangerines_highest_frequency_l2728_272859


namespace NUMINAMATH_CALUDE_impossible_distance_l2728_272815

/-- Two circles with no common points -/
structure DisjointCircles where
  r₁ : ℝ
  r₂ : ℝ
  d : ℝ
  h₁ : r₁ = 2
  h₂ : r₂ = 5
  h₃ : d < r₂ - r₁ ∨ d > r₂ + r₁

theorem impossible_distance (c : DisjointCircles) : c.d ≠ 5 := by
  sorry

end NUMINAMATH_CALUDE_impossible_distance_l2728_272815


namespace NUMINAMATH_CALUDE_set_union_problem_l2728_272875

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {-1, a}
def B (a b : ℝ) : Set ℝ := {2^a, b}

-- Theorem statement
theorem set_union_problem (a b : ℝ) : 
  A a ∩ B a b = {1} → A a ∪ B a b = {-1, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_set_union_problem_l2728_272875


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2728_272852

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (4 - 5 * x) = 8 → x = -12 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2728_272852


namespace NUMINAMATH_CALUDE_base_angle_measure_l2728_272887

-- Define an isosceles triangle
structure IsoscelesTriangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  sum_of_angles : angle1 + angle2 + angle3 = 180
  isosceles : angle2 = angle3

-- Theorem statement
theorem base_angle_measure (t : IsoscelesTriangle) (h : t.angle1 = 80 ∨ t.angle2 = 80) :
  t.angle2 = 50 ∨ t.angle2 = 80 := by
  sorry


end NUMINAMATH_CALUDE_base_angle_measure_l2728_272887


namespace NUMINAMATH_CALUDE_combinatorial_equation_l2728_272811

theorem combinatorial_equation (n : ℕ) : (Nat.choose (n + 1) (n - 1) = 28) → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_combinatorial_equation_l2728_272811


namespace NUMINAMATH_CALUDE_divisibility_by_27_l2728_272894

theorem divisibility_by_27 (x y z : ℤ) (h : (x - y) * (y - z) * (z - x) = x + y + z) :
  27 ∣ (x + y + z) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_27_l2728_272894


namespace NUMINAMATH_CALUDE_circle_equation_proof_l2728_272831

/-- Given a circle with center (1, 1) intersecting the line x + y = 4 to form a chord of length 2√3,
    prove that the equation of the circle is (x-1)² + (y-1)² = 5. -/
theorem circle_equation_proof (x y : ℝ) : 
  let center : ℝ × ℝ := (1, 1)
  let line_equation := x + y = 4
  let chord_length : ℝ := 2 * Real.sqrt 3
  true → (x - 1)^2 + (y - 1)^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_proof_l2728_272831


namespace NUMINAMATH_CALUDE_gcd_1729_1314_l2728_272868

theorem gcd_1729_1314 : Nat.gcd 1729 1314 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1729_1314_l2728_272868


namespace NUMINAMATH_CALUDE_cross_country_winning_scores_l2728_272825

/-- Represents a cross country race between two teams -/
structure CrossCountryRace where
  num_runners_per_team : ℕ
  total_runners : ℕ
  min_score : ℕ
  max_score : ℕ

/-- Calculates the number of different winning scores in a cross country race -/
def num_winning_scores (race : CrossCountryRace) : ℕ :=
  race.max_score - race.min_score + 1

/-- The specific cross country race described in the problem -/
def specific_race : CrossCountryRace :=
  { num_runners_per_team := 6
  , total_runners := 12
  , min_score := 21
  , max_score := 39 }

theorem cross_country_winning_scores :
  num_winning_scores specific_race = 19 := by
  sorry

end NUMINAMATH_CALUDE_cross_country_winning_scores_l2728_272825


namespace NUMINAMATH_CALUDE_inequality_proof_l2728_272876

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  1/x + 1/y + 1/z + 9/(x+y+z) ≥ 
  3 * ((1/(2*x+y) + 1/(x+2*y)) + (1/(2*y+z) + 1/(y+2*z)) + (1/(2*z+x) + 1/(x+2*z))) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2728_272876


namespace NUMINAMATH_CALUDE_fibonacci_closed_form_l2728_272898

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

theorem fibonacci_closed_form (a b : ℝ) (h1 : a + b = 1) (h2 : a * b = -1) (h3 : a > b) :
  ∀ n : ℕ, fibonacci n = (a^(n+1) - b^(n+1)) / Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_fibonacci_closed_form_l2728_272898


namespace NUMINAMATH_CALUDE_f_properties_l2728_272802

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x)^2 - 2 * Real.sqrt 3 * Real.sin x * Real.sin (x - Real.pi / 2)

def is_period (T : ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + T) = f x

theorem f_properties :
  ∃ (T : ℝ) (A B : ℝ) (a b c : ℝ),
    T > 0 ∧
    is_period T f ∧
    (∀ T' > 0, is_period T' f → T ≤ T') ∧
    f (A / 2) = 3 ∧
    (1 / 4 * (a^2 + c^2 - b^2) = 1 / 2 * a * c * Real.sin B) →
    (T = Real.pi ∧ b / a = Real.sqrt 6 / 3) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l2728_272802


namespace NUMINAMATH_CALUDE_hyperbola_real_axis_length_l2728_272873

-- Define the hyperbola and parabola
def hyperbola (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1 ∧ a > 0 ∧ b > 0
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the intersection points A and B
def intersectionPoints (a b : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), hyperbola a b x₁ y₁ ∧ parabola x₁ y₁ ∧
                       hyperbola a b x₂ y₂ ∧ parabola x₂ y₂ ∧
                       (x₁ ≠ x₂ ∨ y₁ ≠ y₂)

-- Define the common focus F
def commonFocus (a b : ℝ) : Prop :=
  ∃ (xf yf : ℝ), (xf = a ∧ yf = 0) ∧ (xf = 1 ∧ yf = 0)

-- Define that line AB passes through F
def lineABThroughF (a b : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ xf yf : ℝ),
    hyperbola a b x₁ y₁ ∧ parabola x₁ y₁ ∧
    hyperbola a b x₂ y₂ ∧ parabola x₂ y₂ ∧
    commonFocus a b ∧
    (y₂ - y₁) * (xf - x₁) = (yf - y₁) * (x₂ - x₁)

-- Theorem statement
theorem hyperbola_real_axis_length
  (a b : ℝ)
  (h_intersect : intersectionPoints a b)
  (h_focus : commonFocus a b)
  (h_line : lineABThroughF a b) :
  2 * a = 2 * Real.sqrt 2 - 2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_real_axis_length_l2728_272873


namespace NUMINAMATH_CALUDE_difference_between_point_eight_and_half_l2728_272869

theorem difference_between_point_eight_and_half : (0.8 : ℝ) - (1/2 : ℝ) = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_difference_between_point_eight_and_half_l2728_272869


namespace NUMINAMATH_CALUDE_current_speed_l2728_272882

theorem current_speed (speed_with_current speed_against_current : ℝ) 
  (h1 : speed_with_current = 15)
  (h2 : speed_against_current = 8.6) :
  ∃ (current_speed : ℝ), current_speed = 3.2 :=
by
  sorry

end NUMINAMATH_CALUDE_current_speed_l2728_272882


namespace NUMINAMATH_CALUDE_gcd_repeated_digit_numbers_l2728_272855

def repeated_digit_number (n : ℕ) : ℕ := 100001 * n

theorem gcd_repeated_digit_numbers :
  ∃ (d : ℕ), d > 0 ∧ ∀ (n : ℕ), 10000 ≤ n ∧ n < 100000 →
    d ∣ repeated_digit_number n ∧
    ∀ (m : ℕ), m > 0 ∧ (∀ (k : ℕ), 10000 ≤ k ∧ k < 100000 → m ∣ repeated_digit_number k) →
      m ∣ d :=
by sorry

end NUMINAMATH_CALUDE_gcd_repeated_digit_numbers_l2728_272855


namespace NUMINAMATH_CALUDE_matrix_crossout_theorem_l2728_272871

theorem matrix_crossout_theorem (M : Matrix (Fin 1000) (Fin 1000) Bool) :
  (∃ (rows : Finset (Fin 1000)), rows.card = 10 ∧
    ∀ j, ∃ i ∈ rows, M i j = true) ∨
  (∃ (cols : Finset (Fin 1000)), cols.card = 10 ∧
    ∀ i, ∃ j ∈ cols, M i j = false) :=
sorry

end NUMINAMATH_CALUDE_matrix_crossout_theorem_l2728_272871


namespace NUMINAMATH_CALUDE_jasons_treats_cost_l2728_272800

/-- Represents the quantity and price of a treat type -/
structure Treat where
  quantity : ℕ  -- quantity in dozens
  price : ℕ     -- price per dozen in dollars
  deriving Repr

/-- Calculates the total cost of treats -/
def totalCost (treats : List Treat) : ℕ :=
  treats.foldl (fun acc t => acc + t.quantity * t.price) 0

theorem jasons_treats_cost (cupcakes cookies brownies : Treat)
    (h1 : cupcakes = { quantity := 4, price := 10 })
    (h2 : cookies = { quantity := 3, price := 8 })
    (h3 : brownies = { quantity := 2, price := 12 }) :
    totalCost [cupcakes, cookies, brownies] = 88 := by
  sorry


end NUMINAMATH_CALUDE_jasons_treats_cost_l2728_272800


namespace NUMINAMATH_CALUDE_brandons_sales_l2728_272810

/-- Given that 2/5 of total sales were credit sales, 3/5 were cash sales,
    and cash sales amounted to $48, prove that the total sales were $80. -/
theorem brandons_sales (T : ℚ) 
  (h1 : (2 : ℚ) / 5 * T + (3 : ℚ) / 5 * T = T)  -- Total sales split into credit and cash
  (h2 : (3 : ℚ) / 5 * T = 48)                   -- Cash sales amount
  : T = 80 := by
  sorry

end NUMINAMATH_CALUDE_brandons_sales_l2728_272810


namespace NUMINAMATH_CALUDE_matrix_equation_proof_l2728_272834

theorem matrix_equation_proof :
  let A : Matrix (Fin 2) (Fin 2) ℚ := !![2, 5; -1, 4]
  let B : Matrix (Fin 2) (Fin 2) ℚ := !![3, -8; 4, -11]
  let P : Matrix (Fin 2) (Fin 2) ℚ := !![4/13, -31/13; 5/13, -42/13]
  P * A = B := by sorry

end NUMINAMATH_CALUDE_matrix_equation_proof_l2728_272834


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2728_272893

theorem sqrt_equation_solution (x : ℝ) : (5 - 1/x)^(1/4) = -3 → x = -1/76 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2728_272893


namespace NUMINAMATH_CALUDE_crossing_time_for_49_explorers_l2728_272819

/-- The minimum time required for all explorers to cross a river -/
def minimum_crossing_time (
  num_explorers : ℕ
  ) (boat_capacity : ℕ
  ) (crossing_time : ℕ
  ) : ℕ :=
  -- The actual calculation would go here
  45

/-- Theorem stating that for 49 explorers, a boat capacity of 7, and a crossing time of 3 minutes,
    the minimum time to cross is 45 minutes -/
theorem crossing_time_for_49_explorers :
  minimum_crossing_time 49 7 3 = 45 := by
  sorry

end NUMINAMATH_CALUDE_crossing_time_for_49_explorers_l2728_272819


namespace NUMINAMATH_CALUDE_interest_rate_is_ten_percent_l2728_272816

/-- Simple interest calculation -/
def simple_interest (principal time rate : ℝ) : ℝ :=
  principal * time * rate

/-- Given conditions -/
def principal : ℝ := 2500
def time : ℝ := 4
def interest : ℝ := 1000

/-- Theorem to prove -/
theorem interest_rate_is_ten_percent :
  ∃ (rate : ℝ), simple_interest principal time rate = interest ∧ rate = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_is_ten_percent_l2728_272816


namespace NUMINAMATH_CALUDE_fifteen_percent_of_x_is_ninety_l2728_272807

theorem fifteen_percent_of_x_is_ninety (x : ℝ) : (15 / 100) * x = 90 → x = 600 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_percent_of_x_is_ninety_l2728_272807


namespace NUMINAMATH_CALUDE_maci_school_supplies_cost_l2728_272813

/-- The cost of Maci's school supplies -/
def school_supplies_cost (blue_pen_price : ℚ) : ℚ :=
  let red_pen_price := 2 * blue_pen_price
  let pencil_price := red_pen_price / 2
  let notebook_price := 10 * blue_pen_price
  10 * blue_pen_price +  -- 10 blue pens
  15 * red_pen_price +   -- 15 red pens
  5 * pencil_price +     -- 5 pencils
  3 * notebook_price     -- 3 notebooks

/-- Theorem stating that the cost of Maci's school supplies is $7.50 -/
theorem maci_school_supplies_cost :
  school_supplies_cost (10 / 100) = 75 / 10 := by
  sorry

end NUMINAMATH_CALUDE_maci_school_supplies_cost_l2728_272813


namespace NUMINAMATH_CALUDE_half_coverage_days_l2728_272862

/-- Represents the number of days it takes for the lily pad patch to cover the entire lake -/
def full_coverage_days : ℕ := 58

/-- Represents the daily growth factor of the lily pad patch -/
def daily_growth_factor : ℕ := 2

/-- Theorem stating that the number of days to cover half the lake is one less than the full coverage days -/
theorem half_coverage_days : 
  ∃ (days : ℕ), days = full_coverage_days - 1 ∧ 
  (daily_growth_factor : ℚ) * ((1 : ℚ) / 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_half_coverage_days_l2728_272862


namespace NUMINAMATH_CALUDE_division_multiplication_result_l2728_272854

theorem division_multiplication_result : 
  let number : ℚ := 4
  let divisor : ℚ := 6
  let multiplier : ℚ := 12
  (number / divisor) * multiplier = 8 := by
sorry

end NUMINAMATH_CALUDE_division_multiplication_result_l2728_272854


namespace NUMINAMATH_CALUDE_cos_120_degrees_l2728_272833

theorem cos_120_degrees : Real.cos (120 * π / 180) = -1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_120_degrees_l2728_272833


namespace NUMINAMATH_CALUDE_function_always_negative_l2728_272891

theorem function_always_negative
  (f : ℝ → ℝ)
  (h_diff : Differentiable ℝ f)
  (h_ineq : ∀ x : ℝ, (2 - x) * f x + x * deriv f x < 0) :
  ∀ x : ℝ, f x < 0 :=
by sorry

end NUMINAMATH_CALUDE_function_always_negative_l2728_272891


namespace NUMINAMATH_CALUDE_cid_earnings_l2728_272889

/-- Represents the earnings from Cid's mechanic shop --/
def mechanic_earnings (oil_change_price : ℕ) (repair_price : ℕ) (car_wash_price : ℕ)
  (oil_changes : ℕ) (repairs : ℕ) (car_washes : ℕ) : ℕ :=
  oil_change_price * oil_changes + repair_price * repairs + car_wash_price * car_washes

/-- Theorem stating that Cid's earnings are $475 given the specific prices and services --/
theorem cid_earnings : 
  mechanic_earnings 20 30 5 5 10 15 = 475 := by
  sorry

end NUMINAMATH_CALUDE_cid_earnings_l2728_272889


namespace NUMINAMATH_CALUDE_equal_charge_at_60_minutes_l2728_272888

/-- United Telephone's base rate in dollars -/
def united_base : ℝ := 9

/-- United Telephone's per-minute charge in dollars -/
def united_per_minute : ℝ := 0.25

/-- Atlantic Call's base rate in dollars -/
def atlantic_base : ℝ := 12

/-- Atlantic Call's per-minute charge in dollars -/
def atlantic_per_minute : ℝ := 0.20

/-- The number of minutes at which both companies charge the same amount -/
def equal_charge_minutes : ℝ := 60

theorem equal_charge_at_60_minutes :
  united_base + united_per_minute * equal_charge_minutes =
  atlantic_base + atlantic_per_minute * equal_charge_minutes :=
by sorry

end NUMINAMATH_CALUDE_equal_charge_at_60_minutes_l2728_272888


namespace NUMINAMATH_CALUDE_exists_non_intersecting_line_l2728_272823

/-- Represents a domino on a grid -/
structure Domino where
  x1 : Nat
  y1 : Nat
  x2 : Nat
  y2 : Nat

/-- Represents a 6x6 grid covered by dominoes -/
structure DominoGrid where
  dominoes : List Domino
  domino_count : dominoes.length = 18
  covers_grid : ∀ x y, x < 6 ∧ y < 6 → ∃ d ∈ dominoes, 
    ((d.x1 = x ∧ d.y1 = y) ∨ (d.x2 = x ∧ d.y2 = y))
  valid_dominoes : ∀ d ∈ dominoes, 
    (d.x1 = d.x2 ∧ d.y2 = d.y1 + 1) ∨ (d.y1 = d.y2 ∧ d.x2 = d.x1 + 1)

/-- Main theorem: There exists a grid line not intersecting any domino -/
theorem exists_non_intersecting_line (grid : DominoGrid) :
  (∃ x : Nat, x < 5 ∧ ∀ d ∈ grid.dominoes, d.x1 ≠ x + 1 ∨ d.x2 ≠ x + 1) ∨
  (∃ y : Nat, y < 5 ∧ ∀ d ∈ grid.dominoes, d.y1 ≠ y + 1 ∨ d.y2 ≠ y + 1) :=
sorry

end NUMINAMATH_CALUDE_exists_non_intersecting_line_l2728_272823


namespace NUMINAMATH_CALUDE_circle_and_tangent_line_l2728_272865

-- Define the vertices of the triangle and point P
def A : ℝ × ℝ := (8, 0)
def B : ℝ × ℝ := (0, 6)
def O : ℝ × ℝ := (0, 0)
def P : ℝ × ℝ := (-1, 5)

-- Define the circumcircle equation
def is_circumcircle_equation (eq : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, eq x y ↔ (x - 4)^2 + (y - 3)^2 = 25

-- Define the tangent line equation
def is_tangent_line_equation (eq : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, eq x y ↔ (x = -1 ∨ 21*x - 20*y + 121 = 0)

-- Theorem statement
theorem circle_and_tangent_line (eq_circle eq_line : ℝ → ℝ → Prop) : 
  is_circumcircle_equation eq_circle ∧ is_tangent_line_equation eq_line :=
sorry

end NUMINAMATH_CALUDE_circle_and_tangent_line_l2728_272865


namespace NUMINAMATH_CALUDE_negation_of_inequality_l2728_272895

theorem negation_of_inequality (x : Real) : 
  (¬ ∀ x ∈ Set.Ioo 0 (π/2), x > Real.sin x) ↔ 
  (∃ x ∈ Set.Ioo 0 (π/2), x ≤ Real.sin x) := by
sorry

end NUMINAMATH_CALUDE_negation_of_inequality_l2728_272895


namespace NUMINAMATH_CALUDE_parabola_equation_l2728_272892

/-- Parabola with focus F and point M -/
structure Parabola where
  p : ℝ
  F : ℝ × ℝ
  M : ℝ × ℝ
  h_p_pos : p > 0
  h_F : F = (p/2, 0)
  h_M_on_C : M.2^2 = 2 * p * M.1
  h_MF_dist : Real.sqrt ((M.1 - F.1)^2 + (M.2 - F.2)^2) = 5

/-- Circle with diameter MF passing through (0,2) -/
def circle_passes_through (P : Parabola) : Prop :=
  let center := ((P.M.1 + P.F.1)/2, (P.M.2 + P.F.2)/2)
  Real.sqrt (center.1^2 + (center.2 - 2)^2) = Real.sqrt ((P.M.1 - P.F.1)^2 + (P.M.2 - P.F.2)^2) / 2

/-- Main theorem -/
theorem parabola_equation (P : Parabola) (h_circle : circle_passes_through P) :
  P.p = 2 ∨ P.p = 8 :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_l2728_272892


namespace NUMINAMATH_CALUDE_min_m_plus_n_l2728_272879

/-- The set T of real numbers satisfying the given condition -/
def T : Set ℝ := Set.Iic 1

/-- The theorem stating the minimum value of m + n -/
theorem min_m_plus_n (m n : ℝ) (h_m : m > 1) (h_n : n > 1)
  (h_exists : ∃ x₀ : ℝ, ∀ x t : ℝ, t ∈ T → |x - 1| - |x - 2| ≥ t)
  (h_log : ∀ t ∈ T, Real.log m / Real.log 3 * Real.log n / Real.log 3 ≥ t) :
  m + n ≥ 6 ∧ ∃ m₀ n₀ : ℝ, m₀ > 1 ∧ n₀ > 1 ∧ m₀ + n₀ = 6 ∧
    (∀ t ∈ T, Real.log m₀ / Real.log 3 * Real.log n₀ / Real.log 3 ≥ t) :=
sorry

end NUMINAMATH_CALUDE_min_m_plus_n_l2728_272879


namespace NUMINAMATH_CALUDE_tan_alpha_half_implies_expression_equals_two_l2728_272849

theorem tan_alpha_half_implies_expression_equals_two (α : Real) 
  (h : Real.tan α = 1 / 2) : 
  (2 * Real.sin α + Real.cos α) / (4 * Real.sin α - Real.cos α) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_half_implies_expression_equals_two_l2728_272849


namespace NUMINAMATH_CALUDE_arithmetic_progression_middle_term_l2728_272847

/-- If 2, b, and 10 form an arithmetic progression, then b = 6 -/
theorem arithmetic_progression_middle_term : 
  ∀ b : ℝ, (2 - b = b - 10) → b = 6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_middle_term_l2728_272847


namespace NUMINAMATH_CALUDE_some_employees_not_in_management_team_l2728_272832

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (Employee : U → Prop)
variable (ManagementTeam : U → Prop)
variable (CompletesTraining : U → Prop)

-- State the theorem
theorem some_employees_not_in_management_team
  (h1 : ∃ x, Employee x ∧ ¬CompletesTraining x)
  (h2 : ∀ x, ManagementTeam x → CompletesTraining x) :
  ∃ x, Employee x ∧ ¬ManagementTeam x :=
by sorry

end NUMINAMATH_CALUDE_some_employees_not_in_management_team_l2728_272832


namespace NUMINAMATH_CALUDE_points_order_l2728_272897

-- Define the line equation
def line_equation (x y b : ℝ) : Prop := y = 3 * x - b

-- Define the points
def point1 (y₁ b : ℝ) : Prop := line_equation (-3) y₁ b
def point2 (y₂ b : ℝ) : Prop := line_equation 1 y₂ b
def point3 (y₃ b : ℝ) : Prop := line_equation (-1) y₃ b

theorem points_order (y₁ y₂ y₃ b : ℝ) 
  (h1 : point1 y₁ b) (h2 : point2 y₂ b) (h3 : point3 y₃ b) : 
  y₁ < y₃ ∧ y₃ < y₂ := by
  sorry

end NUMINAMATH_CALUDE_points_order_l2728_272897


namespace NUMINAMATH_CALUDE_cistern_length_is_eight_l2728_272856

/-- Represents a rectangular cistern with water --/
structure Cistern where
  length : ℝ
  width : ℝ
  depth : ℝ
  wetSurfaceArea : ℝ

/-- Calculates the wet surface area of a cistern --/
def wetSurfaceArea (c : Cistern) : ℝ :=
  c.length * c.width + 2 * c.length * c.depth + 2 * c.width * c.depth

/-- Theorem stating that a cistern with given dimensions has a length of 8 meters --/
theorem cistern_length_is_eight (c : Cistern) 
    (h1 : c.width = 4)
    (h2 : c.depth = 1.25)
    (h3 : c.wetSurfaceArea = 62)
    (h4 : wetSurfaceArea c = c.wetSurfaceArea) : 
    c.length = 8 := by
  sorry


end NUMINAMATH_CALUDE_cistern_length_is_eight_l2728_272856


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_three_numbers_l2728_272886

theorem arithmetic_mean_of_three_numbers (a b c : ℝ) (h : a = 14 ∧ b = 22 ∧ c = 36) :
  (a + b + c) / 3 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_three_numbers_l2728_272886


namespace NUMINAMATH_CALUDE_tan_seventeen_pi_over_four_l2728_272827

theorem tan_seventeen_pi_over_four : Real.tan (17 * π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_seventeen_pi_over_four_l2728_272827


namespace NUMINAMATH_CALUDE_largest_non_expressible_not_expressible_83_largest_non_expressible_is_83_l2728_272829

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def is_expressible (n : ℕ) : Prop :=
  ∃ k c, k > 0 ∧ is_composite c ∧ n = 36 * k + c

theorem largest_non_expressible : ∀ n : ℕ, n > 83 → is_expressible n :=
  sorry

theorem not_expressible_83 : ¬ is_expressible 83 :=
  sorry

theorem largest_non_expressible_is_83 :
  (∀ n : ℕ, n > 83 → is_expressible n) ∧ ¬ is_expressible 83 :=
  sorry

end NUMINAMATH_CALUDE_largest_non_expressible_not_expressible_83_largest_non_expressible_is_83_l2728_272829


namespace NUMINAMATH_CALUDE_horners_first_step_l2728_272843

-- Define the polynomial coefficients
def a₅ : ℝ := 0.5
def a₄ : ℝ := 4
def a₃ : ℝ := 0
def a₂ : ℝ := -3
def a₁ : ℝ := 1
def a₀ : ℝ := -1

-- Define the polynomial
def f (x : ℝ) : ℝ := a₅ * x^5 + a₄ * x^4 + a₃ * x^3 + a₂ * x^2 + a₁ * x + a₀

-- Define the point at which to evaluate the polynomial
def x : ℝ := 3

-- State the theorem
theorem horners_first_step :
  a₅ * x + a₄ = 5.5 :=
sorry

end NUMINAMATH_CALUDE_horners_first_step_l2728_272843


namespace NUMINAMATH_CALUDE_nested_quadrilaterals_diagonal_inequality_l2728_272809

/-- A convex quadrilateral -/
structure ConvexQuadrilateral where
  /-- The perimeter of the quadrilateral -/
  perimeter : ℝ
  /-- The sum of the lengths of the diagonals -/
  diagonalSum : ℝ
  /-- The perimeter is positive -/
  perimeterPos : perimeter > 0
  /-- The sum of diagonal lengths is positive -/
  diagonalSumPos : diagonalSum > 0
  /-- The perimeter is greater than the sum of diagonal lengths -/
  perimeterGreaterThanDiagonals : perimeter > diagonalSum
  /-- The sum of diagonal lengths is greater than half the perimeter -/
  diagonalsGreaterThanHalfPerimeter : diagonalSum > perimeter / 2

/-- Two convex quadrilaterals where one is inside the other -/
structure NestedQuadrilaterals where
  outer : ConvexQuadrilateral
  inner : ConvexQuadrilateral
  /-- The perimeter of the inner quadrilateral is less than the outer -/
  innerSmallerPerimeter : inner.perimeter < outer.perimeter

theorem nested_quadrilaterals_diagonal_inequality (q : NestedQuadrilaterals) :
  q.inner.diagonalSum < 2 * q.outer.diagonalSum :=
sorry

end NUMINAMATH_CALUDE_nested_quadrilaterals_diagonal_inequality_l2728_272809


namespace NUMINAMATH_CALUDE_sunset_time_correct_l2728_272899

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat

/-- Represents a duration in hours and minutes -/
structure Duration where
  hours : Nat
  minutes : Nat

/-- Adds a duration to a time -/
def addDuration (t : Time) (d : Duration) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + d.hours * 60 + d.minutes
  { hours := totalMinutes / 60, minutes := totalMinutes % 60 }

theorem sunset_time_correct 
  (sunrise : Time)
  (daylight : Duration)
  (sunset : Time)
  (h1 : sunrise = { hours := 6, minutes := 45 })
  (h2 : daylight = { hours := 11, minutes := 12 })
  (h3 : sunset = { hours := 17, minutes := 57 }) :
  addDuration sunrise daylight = sunset :=
sorry

end NUMINAMATH_CALUDE_sunset_time_correct_l2728_272899


namespace NUMINAMATH_CALUDE_triangle_area_l2728_272805

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    where b = 1, c = √3, and ∠C = 2π/3, prove that its area is √3/4 -/
theorem triangle_area (a b c : ℝ) (A B C : ℝ) : 
  b = 1 → 
  c = Real.sqrt 3 → 
  C = 2 * Real.pi / 3 → 
  (1/2) * a * b * Real.sin C = Real.sqrt 3 / 4 := by
sorry


end NUMINAMATH_CALUDE_triangle_area_l2728_272805


namespace NUMINAMATH_CALUDE_election_winner_votes_l2728_272866

theorem election_winner_votes 
  (total_votes : ℕ) 
  (winner_percentage : ℚ) 
  (vote_difference : ℕ) 
  (h1 : winner_percentage = 62 / 100) 
  (h2 : vote_difference = 336) 
  (h3 : ↑total_votes * winner_percentage - ↑total_votes * (1 - winner_percentage) = vote_difference) :
  ↑total_votes * winner_percentage = 868 :=
sorry

end NUMINAMATH_CALUDE_election_winner_votes_l2728_272866


namespace NUMINAMATH_CALUDE_solution_set_f_max_value_f_range_of_m_l2728_272864

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 3| - |x + 5|

-- Theorem for the solution set of f(x) ≥ 2
theorem solution_set_f (x : ℝ) : f x ≥ 2 ↔ x ≤ -2 := by sorry

-- Theorem for the maximum value of f(x)
theorem max_value_f : ∃ M : ℝ, M = 8 ∧ ∀ x : ℝ, f x ≤ M := by sorry

-- Theorem for the range of m
theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, x^2 + 2*x + m ≤ 8) ↔ m ≤ 9 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_max_value_f_range_of_m_l2728_272864


namespace NUMINAMATH_CALUDE_train_length_l2728_272853

/-- Given a train crossing a bridge, calculate its length -/
theorem train_length (train_speed : ℝ) (crossing_time : ℝ) (bridge_length : ℝ) : 
  train_speed = 45 * (1000 / 3600) →
  crossing_time = 30 →
  bridge_length = 235 →
  (train_speed * crossing_time) - bridge_length = 140 := by
sorry

end NUMINAMATH_CALUDE_train_length_l2728_272853


namespace NUMINAMATH_CALUDE_average_bicycling_speed_l2728_272861

/-- Calculates the average bicycling speed given the conditions of the problem -/
theorem average_bicycling_speed (total_distance : ℝ) (bicycle_time : ℝ) (run_speed : ℝ) (total_time : ℝ) :
  total_distance = 20 →
  bicycle_time = 12 / 60 →
  run_speed = 8 →
  total_time = 117 / 60 →
  let run_time := total_time - bicycle_time
  let run_distance := run_speed * run_time
  let bicycle_distance := total_distance - run_distance
  bicycle_distance / bicycle_time = 30 := by
  sorry

#check average_bicycling_speed

end NUMINAMATH_CALUDE_average_bicycling_speed_l2728_272861


namespace NUMINAMATH_CALUDE_lesser_fraction_proof_l2728_272874

theorem lesser_fraction_proof (x y : ℚ) : 
  x + y = 11/12 → x * y = 1/6 → min x y = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_lesser_fraction_proof_l2728_272874


namespace NUMINAMATH_CALUDE_angle_in_fourth_quadrant_l2728_272846

-- Define the hyperbola equation
def hyperbola_equation (x y α : ℝ) : Prop :=
  x^2 * Real.sin α + y^2 * Real.cos α = 1

-- Define the property of hyperbola with foci on y-axis
def foci_on_y_axis (α : ℝ) : Prop :=
  ∃ (x y : ℝ), hyperbola_equation x y α ∧ Real.cos α > 0 ∧ Real.sin α < 0

-- Theorem statement
theorem angle_in_fourth_quadrant (α : ℝ) (h : foci_on_y_axis α) :
  α > -π/2 ∧ α < 0 :=
sorry

end NUMINAMATH_CALUDE_angle_in_fourth_quadrant_l2728_272846


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2728_272850

def M : Set ℝ := {y | ∃ x, y = 2^x}
def N : Set ℝ := {x | ∃ y, y = Real.sqrt (x - 1)}

theorem intersection_of_M_and_N :
  M ∩ N = {y | y ≥ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2728_272850


namespace NUMINAMATH_CALUDE_max_additional_plates_achievable_additional_plates_l2728_272838

/-- Represents the sets of symbols for car plates in Rivertown -/
structure CarPlateSymbols where
  firstLetters : Finset Char
  secondLetters : Finset Char
  digits : Finset Char

/-- Calculates the total number of possible car plates -/
def totalPlates (symbols : CarPlateSymbols) : ℕ :=
  symbols.firstLetters.card * symbols.secondLetters.card * symbols.digits.card

/-- The initial configuration of car plate symbols -/
def initialSymbols : CarPlateSymbols :=
  { firstLetters := {'A', 'B', 'G', 'H', 'T'},
    secondLetters := {'E', 'I', 'O', 'U'},
    digits := {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'} }

/-- Represents the addition of new symbols -/
structure NewSymbols where
  newLetters : ℕ
  newDigits : ℕ

/-- The new symbols to be added -/
def addedSymbols : NewSymbols :=
  { newLetters := 2,
    newDigits := 1 }

/-- Theorem: The maximum number of additional car plates after adding new symbols is 130 -/
theorem max_additional_plates :
  ∀ (newDistribution : CarPlateSymbols),
    (newDistribution.firstLetters.card + newDistribution.secondLetters.card = 
      initialSymbols.firstLetters.card + initialSymbols.secondLetters.card + addedSymbols.newLetters) →
    (newDistribution.digits.card = initialSymbols.digits.card + addedSymbols.newDigits) →
    totalPlates newDistribution - totalPlates initialSymbols ≤ 130 :=
by sorry

/-- Theorem: There exists a distribution that achieves 130 additional plates -/
theorem achievable_additional_plates :
  ∃ (newDistribution : CarPlateSymbols),
    (newDistribution.firstLetters.card + newDistribution.secondLetters.card = 
      initialSymbols.firstLetters.card + initialSymbols.secondLetters.card + addedSymbols.newLetters) ∧
    (newDistribution.digits.card = initialSymbols.digits.card + addedSymbols.newDigits) ∧
    totalPlates newDistribution - totalPlates initialSymbols = 130 :=
by sorry

end NUMINAMATH_CALUDE_max_additional_plates_achievable_additional_plates_l2728_272838


namespace NUMINAMATH_CALUDE_simplify_expression_l2728_272839

theorem simplify_expression (y : ℝ) : 3 * y - 7 * y^2 + 15 - (6 - 5 * y + 7 * y^2) = -14 * y^2 + 8 * y + 9 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2728_272839


namespace NUMINAMATH_CALUDE_max_rectangles_in_square_l2728_272818

/-- Given a square with side length 14 cm and rectangles of width 2 cm and length 8 cm,
    the maximum number of whole rectangles that can fit within the square is 12. -/
theorem max_rectangles_in_square : ∀ (square_side : ℝ) (rect_width : ℝ) (rect_length : ℝ),
  square_side = 14 →
  rect_width = 2 →
  rect_length = 8 →
  ⌊(square_side * square_side) / (rect_width * rect_length)⌋ = 12 := by
  sorry

end NUMINAMATH_CALUDE_max_rectangles_in_square_l2728_272818


namespace NUMINAMATH_CALUDE_decreasing_quadratic_implies_a_equals_negative_one_l2728_272822

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 1

-- State the theorem
theorem decreasing_quadratic_implies_a_equals_negative_one :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → y < 2 → f a x > f a y) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_decreasing_quadratic_implies_a_equals_negative_one_l2728_272822


namespace NUMINAMATH_CALUDE_second_attempt_score_l2728_272884

/-- Represents the number of points scored in each attempt -/
structure Attempts where
  first : ℕ
  second : ℕ
  third : ℕ

/-- The minimum and maximum possible points for a single dart throw -/
def min_points : ℕ := 3
def max_points : ℕ := 9

/-- The number of darts thrown in each attempt -/
def num_darts : ℕ := 8

theorem second_attempt_score (a : Attempts) : 
  (a.second = 2 * a.first) → 
  (a.third = 3 * a.first) → 
  (a.first ≥ num_darts * min_points) → 
  (a.third ≤ num_darts * max_points) → 
  a.second = 48 := by
  sorry

end NUMINAMATH_CALUDE_second_attempt_score_l2728_272884


namespace NUMINAMATH_CALUDE_positive_integer_division_l2728_272863

theorem positive_integer_division (a b : ℕ+) :
  (a * b^2 + b + 7) ∣ (a^2 * b + a + b) ↔
    ((a = 11 ∧ b = 1) ∨
     (a = 49 ∧ b = 1) ∨
     (∃ k : ℕ+, a = 7 * k^2 ∧ b = 7 * k)) := by
  sorry

end NUMINAMATH_CALUDE_positive_integer_division_l2728_272863


namespace NUMINAMATH_CALUDE_sum_of_combinations_l2728_272814

theorem sum_of_combinations : Nat.choose 10 3 + Nat.choose 10 4 = 330 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_combinations_l2728_272814


namespace NUMINAMATH_CALUDE_tan_one_iff_quarter_pi_plus_multiple_pi_l2728_272801

theorem tan_one_iff_quarter_pi_plus_multiple_pi (x : ℝ) : 
  Real.tan x = 1 ↔ ∃ k : ℤ, x = k * Real.pi + Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_tan_one_iff_quarter_pi_plus_multiple_pi_l2728_272801


namespace NUMINAMATH_CALUDE_assembly_line_theorem_l2728_272842

/-- Represents the number of tasks in the assembly line -/
def num_tasks : ℕ := 6

/-- Represents the number of freely arrangeable tasks -/
def num_free_tasks : ℕ := 5

/-- The number of ways to arrange the assembly line -/
def assembly_line_arrangements : ℕ := Nat.factorial num_free_tasks

/-- Theorem stating the number of ways to arrange the assembly line -/
theorem assembly_line_theorem : 
  assembly_line_arrangements = 120 := by sorry

end NUMINAMATH_CALUDE_assembly_line_theorem_l2728_272842


namespace NUMINAMATH_CALUDE_division_problem_l2728_272870

theorem division_problem : ∃ (q : ℕ), 
  220080 = (555 + 445) * q + 80 ∧ 
  q = 2 * (555 - 445) := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l2728_272870


namespace NUMINAMATH_CALUDE_polynomial_irreducibility_l2728_272821

theorem polynomial_irreducibility (n : ℕ) (hn : n > 1) :
  let f : Polynomial ℤ := X^n + 5 * X^(n-1) + 3
  Irreducible f := by sorry

end NUMINAMATH_CALUDE_polynomial_irreducibility_l2728_272821


namespace NUMINAMATH_CALUDE_xyz_inequality_l2728_272867

theorem xyz_inequality (x y z : ℝ) (h_nonneg : 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z) 
  (h_eq : x^2 + y^2 + z^2 + x*y*z = 4) : 
  x*y*z ≤ x*y + y*z + z*x ∧ x*y + y*z + z*x ≤ x*y*z + 2 := by
  sorry

end NUMINAMATH_CALUDE_xyz_inequality_l2728_272867


namespace NUMINAMATH_CALUDE_factor_expression_l2728_272820

theorem factor_expression (x : ℝ) : 3*x*(x-5) - 2*(x-5) + 4*x*(x-5) = (x-5)*(7*x-2) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l2728_272820


namespace NUMINAMATH_CALUDE_six_people_arrangement_l2728_272804

/-- The number of arrangements with A at the edge -/
def edge_arrangements : ℕ := 4 * 3 * 24

/-- The number of arrangements with A in the middle -/
def middle_arrangements : ℕ := 2 * 2 * 24

/-- The total number of valid arrangements -/
def total_arrangements : ℕ := edge_arrangements + middle_arrangements

theorem six_people_arrangement :
  total_arrangements = 384 :=
sorry

end NUMINAMATH_CALUDE_six_people_arrangement_l2728_272804


namespace NUMINAMATH_CALUDE_problem_statement_l2728_272828

variables (a b c : ℝ)

def f (x : ℝ) := a * x^2 + b * x + c
def g (x : ℝ) := a * x + b

theorem problem_statement :
  (∀ x : ℝ, abs x ≤ 1 → abs (f a b c x) ≤ 1) →
  (abs c ≤ 1 ∧ ∀ x : ℝ, abs x ≤ 1 → abs (g a b x) ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l2728_272828


namespace NUMINAMATH_CALUDE_quadratic_rewrite_l2728_272857

theorem quadratic_rewrite (a b c : ℤ) :
  (∀ x : ℝ, 16 * x^2 - 48 * x - 72 = (a * x + b)^2 + c) →
  a * b = -24 := by
sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_l2728_272857


namespace NUMINAMATH_CALUDE_cost_of_one_sandwich_and_juice_l2728_272812

/-- Given the cost of multiple items, calculate the cost of one item and one juice -/
theorem cost_of_one_sandwich_and_juice 
  (juice_cost : ℝ) 
  (juice_count : ℕ) 
  (sandwich_cost : ℝ) 
  (sandwich_count : ℕ) : 
  juice_cost / juice_count + sandwich_cost / sandwich_count = 5 :=
by
  sorry

#check cost_of_one_sandwich_and_juice 10 5 6 2

end NUMINAMATH_CALUDE_cost_of_one_sandwich_and_juice_l2728_272812


namespace NUMINAMATH_CALUDE_inverse_proposition_false_l2728_272841

theorem inverse_proposition_false : 
  ¬ (∀ a b c : ℝ, a > b → a * c^2 > b * c^2) := by
sorry

end NUMINAMATH_CALUDE_inverse_proposition_false_l2728_272841


namespace NUMINAMATH_CALUDE_product_mod_seventeen_l2728_272848

theorem product_mod_seventeen :
  (1234 * 1235 * 1236 * 1237 * 1238) % 17 = 9 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_seventeen_l2728_272848


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l2728_272817

/-- A regular polygon with an exterior angle of 18 degrees has 20 sides. -/
theorem regular_polygon_sides (n : ℕ) (exterior_angle : ℝ) : 
  exterior_angle = 18 → n * exterior_angle = 360 → n = 20 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l2728_272817


namespace NUMINAMATH_CALUDE_divided_triangle_properties_l2728_272803

structure DividedTriangle where
  u : ℝ
  v : ℝ
  x : ℝ
  y : ℝ
  z : ℝ
  S : ℝ
  u_pos : u > 0
  v_pos : v > 0
  x_pos : x > 0
  y_pos : y > 0
  z_pos : z > 0
  S_pos : S > 0

theorem divided_triangle_properties (t : DividedTriangle) :
  t.u * t.v = t.y * t.z ∧ t.S ≤ (t.x * t.z) / t.y := by
  sorry

end NUMINAMATH_CALUDE_divided_triangle_properties_l2728_272803


namespace NUMINAMATH_CALUDE_exponential_inequality_l2728_272808

theorem exponential_inequality (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (1.5 : ℝ) ^ a > (1.5 : ℝ) ^ b := by
  sorry

end NUMINAMATH_CALUDE_exponential_inequality_l2728_272808


namespace NUMINAMATH_CALUDE_line_hyperbola_intersection_l2728_272880

/-- Given a line y = ax + 1 and a hyperbola 3x^2 - y^2 = 1 that intersect at points A and B,
    if a circle with AB as its diameter passes through the origin,
    then a = 1 or a = -1 -/
theorem line_hyperbola_intersection (a : ℝ) : 
  (∃ A B : ℝ × ℝ, 
    (A.2 = a * A.1 + 1 ∧ 3 * A.1^2 - A.2^2 = 1) ∧ 
    (B.2 = a * B.1 + 1 ∧ 3 * B.1^2 - B.2^2 = 1) ∧ 
    A ≠ B ∧
    (A.1 * B.1 + A.2 * B.2 = 0)) →
  a = 1 ∨ a = -1 := by
sorry

end NUMINAMATH_CALUDE_line_hyperbola_intersection_l2728_272880
