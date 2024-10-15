import Mathlib

namespace NUMINAMATH_CALUDE_remainder_theorem_l3200_320000

theorem remainder_theorem (d : ℕ) (r : ℕ) : d > 1 →
  (1059 % d = r ∧ 1417 % d = r ∧ 2312 % d = r) →
  d - r = 15 := by
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3200_320000


namespace NUMINAMATH_CALUDE_line_chart_best_for_fever_temperature_l3200_320014

/- Define the types of charts -/
inductive ChartType
| Bar
| Line
| Pie

/- Define the properties of data we want to visualize -/
structure TemperatureData where
  showsQuantity : Bool
  showsChanges : Bool
  showsRelationship : Bool

/- Define the characteristics of fever temperature data -/
def feverTemperatureData : TemperatureData :=
  { showsQuantity := true
  , showsChanges := true
  , showsRelationship := false }

/- Define which chart types are suitable for different data properties -/
def suitableChartType (data : TemperatureData) : ChartType :=
  if data.showsChanges then ChartType.Line
  else if data.showsQuantity then ChartType.Bar
  else ChartType.Pie

/- Theorem: Line chart is the best for tracking fever temperature changes -/
theorem line_chart_best_for_fever_temperature : 
  suitableChartType feverTemperatureData = ChartType.Line := by
  sorry

end NUMINAMATH_CALUDE_line_chart_best_for_fever_temperature_l3200_320014


namespace NUMINAMATH_CALUDE_equilateral_triangle_centroid_sum_l3200_320058

/-- An equilateral triangle with side length 1 -/
structure EquilateralTriangle where
  side_length : ℝ
  is_equilateral : side_length = 1

/-- The centroid of a triangle -/
class Centroid (T : Type) where
  point : T → ℝ × ℝ

/-- The length of a segment from a vertex to the centroid -/
def vertex_to_centroid_length (t : EquilateralTriangle) : ℝ := sorry

theorem equilateral_triangle_centroid_sum 
  (t : EquilateralTriangle) 
  [Centroid EquilateralTriangle] : 
  3 * vertex_to_centroid_length t = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_centroid_sum_l3200_320058


namespace NUMINAMATH_CALUDE_club_membership_l3200_320009

theorem club_membership (total_members attendance : ℕ) 
  (h1 : total_members = 30)
  (h2 : attendance = 20)
  (h3 : ∃ (men women : ℕ), men + women = total_members ∧ men + women / 3 = attendance) :
  ∃ (men : ℕ), men = 15 ∧ 
    ∃ (women : ℕ), men + women = total_members ∧ men + women / 3 = attendance :=
by sorry

end NUMINAMATH_CALUDE_club_membership_l3200_320009


namespace NUMINAMATH_CALUDE_corn_purchase_proof_l3200_320057

/-- The cost of corn in cents per pound -/
def corn_cost : ℚ := 99

/-- The cost of beans in cents per pound -/
def bean_cost : ℚ := 45

/-- The total weight of corn and beans in pounds -/
def total_weight : ℚ := 24

/-- The total cost in cents -/
def total_cost : ℚ := 1809

/-- The number of pounds of corn bought -/
def corn_pounds : ℚ := 13.5

theorem corn_purchase_proof :
  ∃ (bean_pounds : ℚ),
    bean_pounds + corn_pounds = total_weight ∧
    bean_cost * bean_pounds + corn_cost * corn_pounds = total_cost :=
by sorry

end NUMINAMATH_CALUDE_corn_purchase_proof_l3200_320057


namespace NUMINAMATH_CALUDE_root_product_equals_negative_183_l3200_320056

-- Define the polynomial h
def h (y : ℝ) : ℝ := y^5 - y^3 + 2*y + 3

-- Define the polynomial p
def p (y : ℝ) : ℝ := y^2 - 3

-- State the theorem
theorem root_product_equals_negative_183 
  (y₁ y₂ y₃ y₄ y₅ : ℝ) 
  (h_roots : h y₁ = 0 ∧ h y₂ = 0 ∧ h y₃ = 0 ∧ h y₄ = 0 ∧ h y₅ = 0) :
  p y₁ * p y₂ * p y₃ * p y₄ * p y₅ = -183 :=
sorry

end NUMINAMATH_CALUDE_root_product_equals_negative_183_l3200_320056


namespace NUMINAMATH_CALUDE_expression_integer_iff_special_form_l3200_320039

def expression (n : ℤ) : ℝ :=
  (n + (n^2 + 1).sqrt)^(1/3) + (n - (n^2 + 1).sqrt)^(1/3)

theorem expression_integer_iff_special_form (n : ℤ) :
  ∃ (k : ℤ), k > 0 ∧ expression n = k ↔ ∃ (m : ℤ), m > 0 ∧ n = m * (m^2 + 3) / 2 :=
sorry

end NUMINAMATH_CALUDE_expression_integer_iff_special_form_l3200_320039


namespace NUMINAMATH_CALUDE_fraction_value_l3200_320021

theorem fraction_value (x y : ℝ) (h : 1 / x - 1 / y = 3) :
  (2 * x + 3 * x * y - 2 * y) / (x - 2 * x * y - y) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l3200_320021


namespace NUMINAMATH_CALUDE_exists_number_with_digit_sum_decrease_l3200_320026

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem stating the existence of a number satisfying the conditions -/
theorem exists_number_with_digit_sum_decrease : 
  ∃ (n : ℕ), 
    (∃ (m : ℕ), (11 * n = 10 * m)) ∧ 
    (sum_of_digits (11 * n / 10) = (9 * sum_of_digits n) / 10) := by sorry

end NUMINAMATH_CALUDE_exists_number_with_digit_sum_decrease_l3200_320026


namespace NUMINAMATH_CALUDE_vector_operation_proof_l3200_320034

def v1 : Fin 3 → ℝ := ![3, -2, 5]
def v2 : Fin 3 → ℝ := ![-1, 6, -3]

theorem vector_operation_proof :
  (2 : ℝ) • (v1 + v2) = ![4, 8, 4] := by sorry

end NUMINAMATH_CALUDE_vector_operation_proof_l3200_320034


namespace NUMINAMATH_CALUDE_no_solution_iff_a_geq_five_l3200_320098

theorem no_solution_iff_a_geq_five (a : ℝ) :
  (∀ x : ℝ, ¬(x ≤ 5 ∧ x > a)) ↔ a ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_iff_a_geq_five_l3200_320098


namespace NUMINAMATH_CALUDE_toy_factory_production_l3200_320096

/-- Represents the production constraints and goal for a toy factory --/
theorem toy_factory_production :
  ∃ (x y : ℕ),
    15 * x + 10 * y ≤ 450 ∧  -- Labor constraint
    20 * x + 5 * y ≤ 400 ∧   -- Raw material constraint
    80 * x + 45 * y = 2200   -- Total selling price
    := by sorry

end NUMINAMATH_CALUDE_toy_factory_production_l3200_320096


namespace NUMINAMATH_CALUDE_pete_triple_age_of_son_l3200_320046

def pete_age : ℕ := 35
def son_age : ℕ := 9
def years_until_triple : ℕ := 4

theorem pete_triple_age_of_son :
  pete_age + years_until_triple = 3 * (son_age + years_until_triple) :=
by sorry

end NUMINAMATH_CALUDE_pete_triple_age_of_son_l3200_320046


namespace NUMINAMATH_CALUDE_locus_of_equilateral_triangle_vertex_l3200_320089

-- Define the circle type
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a rotation function
def rotate (p : ℝ × ℝ) (center : ℝ × ℝ) (angle : ℝ) : ℝ × ℝ :=
  sorry

-- Define a function to check if a point is on a circle
def onCircle (p : ℝ × ℝ) (c : Circle) : Prop :=
  sorry

-- Define a function to check if a triangle is equilateral
def isEquilateral (a b c : ℝ × ℝ) : Prop :=
  sorry

-- Theorem statement
theorem locus_of_equilateral_triangle_vertex (C : Circle) (P : ℝ × ℝ) :
  let locusM := {M : ℝ × ℝ | ∃ K, onCircle K C ∧ isEquilateral P K M}
  let rotated_circle_1 := {p : ℝ × ℝ | ∃ q, onCircle q C ∧ p = rotate q P (π/3)}
  let rotated_circle_2 := {p : ℝ × ℝ | ∃ q, onCircle q C ∧ p = rotate q P (-π/3)}
  if P = C.center then
    locusM = {p : ℝ × ℝ | onCircle p C}
  else
    locusM = rotated_circle_1 ∪ rotated_circle_2 :=
by sorry


end NUMINAMATH_CALUDE_locus_of_equilateral_triangle_vertex_l3200_320089


namespace NUMINAMATH_CALUDE_mean_proportional_problem_l3200_320033

theorem mean_proportional_problem (x : ℝ) : 
  (156 : ℝ)^2 = x * 104 → x = 234 := by
  sorry

end NUMINAMATH_CALUDE_mean_proportional_problem_l3200_320033


namespace NUMINAMATH_CALUDE_line_circle_relationship_l3200_320051

theorem line_circle_relationship (k : ℝ) : 
  ∃ (x y : ℝ), (x - k*y + 1 = 0) ∧ (x^2 + y^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_line_circle_relationship_l3200_320051


namespace NUMINAMATH_CALUDE_sum_of_digits_congruence_l3200_320024

def S (n : ℕ+) : ℕ := sorry

theorem sum_of_digits_congruence (n : ℕ+) (h : S n = 29) : 
  S (n + 1) % 9 = 3 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_congruence_l3200_320024


namespace NUMINAMATH_CALUDE_smallest_period_of_sine_function_l3200_320005

/-- Given a function f(x) = √3 * sin(πx/k) whose adjacent maximum and minimum points
    lie on the circle x^2 + y^2 = k^2, prove that its smallest positive period is 4. -/
theorem smallest_period_of_sine_function (k : ℝ) (h : k > 0) :
  let f : ℝ → ℝ := λ x ↦ Real.sqrt 3 * Real.sin (π * x / k)
  (∃ x y : ℝ, x^2 + y^2 = k^2 ∧ 
    (f x = Real.sqrt 3 ∧ f ((x + k/2) % (2*k)) = -Real.sqrt 3)) →
  2 * k = 4 :=
by sorry

end NUMINAMATH_CALUDE_smallest_period_of_sine_function_l3200_320005


namespace NUMINAMATH_CALUDE_natural_solutions_3x_plus_4y_eq_12_l3200_320012

theorem natural_solutions_3x_plus_4y_eq_12 :
  {(x, y) : ℕ × ℕ | 3 * x + 4 * y = 12} = {(4, 0), (0, 3)} := by
  sorry

end NUMINAMATH_CALUDE_natural_solutions_3x_plus_4y_eq_12_l3200_320012


namespace NUMINAMATH_CALUDE_pipe_fill_time_l3200_320052

/-- Given a pipe and a tank with a leak, this theorem proves the time taken for the pipe
    to fill the tank alone, based on the time taken to fill with both pipe and leak,
    and the time taken for the leak to empty the tank. -/
theorem pipe_fill_time (fill_time_with_leak : ℝ) (leak_empty_time : ℝ) 
    (h1 : fill_time_with_leak = 18) 
    (h2 : leak_empty_time = 36) : 
    (1 : ℝ) / ((1 : ℝ) / fill_time_with_leak + (1 : ℝ) / leak_empty_time) = 12 := by
  sorry

end NUMINAMATH_CALUDE_pipe_fill_time_l3200_320052


namespace NUMINAMATH_CALUDE_diver_min_trips_l3200_320004

/-- The minimum number of trips required to carry all objects to the surface -/
def min_trips (capacity : ℕ) (total_objects : ℕ) : ℕ :=
  (total_objects + capacity - 1) / capacity

/-- Theorem: Given a diver who can carry 3 objects at a time and has found 17 objects,
    the minimum number of trips required to carry all objects to the surface is 6 -/
theorem diver_min_trips :
  min_trips 3 17 = 6 := by
  sorry

end NUMINAMATH_CALUDE_diver_min_trips_l3200_320004


namespace NUMINAMATH_CALUDE_rap_song_requests_l3200_320049

/-- Represents the number of song requests for different genres in a night --/
structure SongRequests where
  total : ℕ
  electropop : ℕ
  dance : ℕ
  rock : ℕ
  oldies : ℕ
  dj_choice : ℕ
  rap : ℕ

/-- Theorem stating the number of rap song requests given the conditions --/
theorem rap_song_requests (r : SongRequests) : r.rap = 2 :=
  by
  have h1 : r.total = 30 := by sorry
  have h2 : r.electropop = r.total / 2 := by sorry
  have h3 : r.dance = r.electropop / 3 := by sorry
  have h4 : r.rock = 5 := by sorry
  have h5 : r.oldies = r.rock - 3 := by sorry
  have h6 : r.dj_choice = r.oldies / 2 := by sorry
  have h7 : r.total = r.electropop + r.dance + r.rock + r.oldies + r.dj_choice + r.rap := by sorry
  sorry

#check rap_song_requests

end NUMINAMATH_CALUDE_rap_song_requests_l3200_320049


namespace NUMINAMATH_CALUDE_original_amount_l3200_320060

theorem original_amount (X : ℝ) : (0.1 * (0.5 * X) = 25) → X = 500 := by
  sorry

end NUMINAMATH_CALUDE_original_amount_l3200_320060


namespace NUMINAMATH_CALUDE_value_range_of_f_l3200_320042

def f (x : ℝ) : ℝ := x^2 - 4*x + 3

theorem value_range_of_f :
  ∀ y ∈ Set.Icc (-1 : ℝ) 3, ∃ x ∈ Set.Icc 0 3, f x = y ∧
  ∀ x ∈ Set.Icc 0 3, f x ∈ Set.Icc (-1 : ℝ) 3 :=
sorry

end NUMINAMATH_CALUDE_value_range_of_f_l3200_320042


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l3200_320067

/-- Represents a geometric sequence -/
structure GeometricSequence where
  a : ℕ → ℚ
  q : ℚ
  geom_prop : ∀ n : ℕ, a (n + 1) = a n * q

/-- Sum of the first n terms of a geometric sequence -/
def GeometricSequence.sum (g : GeometricSequence) (n : ℕ) : ℚ :=
  if g.q = 1 then n * g.a 1 else g.a 1 * (1 - g.q ^ n) / (1 - g.q)

theorem geometric_sequence_fourth_term 
  (g : GeometricSequence) 
  (h1 : g.a 1 - g.a 5 = -15/2) 
  (h2 : g.sum 4 = -5) : 
  g.a 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l3200_320067


namespace NUMINAMATH_CALUDE_AC_greater_than_CK_l3200_320059

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  AB = BC ∧ AC = 2 * Real.sqrt 7 ∧ AB = 8

-- Define point D as the foot of the height from B
def HeightFoot (A B C D : ℝ × ℝ) : Prop :=
  (B.1 - D.1) * (A.1 - C.1) + (B.2 - D.2) * (A.2 - C.2) = 0 ∧
  D.1 = (A.1 + C.1) / 2 ∧ D.2 = (A.2 + C.2) / 2

-- Define point K on BD
def PointK (B D K : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), t = 2/5 ∧ K = (t * B.1 + (1 - t) * D.1, t * B.2 + (1 - t) * D.2)

-- Main theorem
theorem AC_greater_than_CK (A B C D K : ℝ × ℝ) :
  Triangle A B C → HeightFoot A B C D → PointK B D K →
  Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2) > Real.sqrt ((C.1 - K.1)^2 + (C.2 - K.2)^2) := by
  sorry

end NUMINAMATH_CALUDE_AC_greater_than_CK_l3200_320059


namespace NUMINAMATH_CALUDE_correlation_significance_l3200_320072

-- Define r as a real number representing a correlation coefficient
variable (r : ℝ)

-- Define r_0.05 as the critical value for a 5% significance level
variable (r_0_05 : ℝ)

-- Define a function that represents the probability of an event
def event_probability (r : ℝ) (r_0_05 : ℝ) : Prop :=
  ∃ p : ℝ, p < 0.05 ∧ (|r| > r_0_05 ↔ p < 0.05)

-- Theorem stating the equivalence
theorem correlation_significance (r : ℝ) (r_0_05 : ℝ) :
  |r| > r_0_05 ↔ event_probability r r_0_05 :=
sorry

end NUMINAMATH_CALUDE_correlation_significance_l3200_320072


namespace NUMINAMATH_CALUDE_circle_equation_from_diameter_l3200_320070

/-- Given two points A and B as the endpoints of a circle's diameter, 
    prove that the equation of the circle is (x - 1)² + (y - 2)² = 25 -/
theorem circle_equation_from_diameter (A B : ℝ × ℝ) : 
  A = (-3, -1) → B = (5, 5) → 
  ∀ (x y : ℝ), (x - 1)^2 + (y - 2)^2 = 25 ↔ 
    ((x - (-3))^2 + (y - (-1))^2 = ((5 - (-3))^2 + (5 - (-1))^2) / 4 ∧ 
     (x - 5)^2 + (y - 5)^2 = ((5 - (-3))^2 + (5 - (-1))^2) / 4) :=
by sorry


end NUMINAMATH_CALUDE_circle_equation_from_diameter_l3200_320070


namespace NUMINAMATH_CALUDE_max_value_of_s_l3200_320019

theorem max_value_of_s (p q r s : ℝ) 
  (sum_eq : p + q + r + s = 12)
  (sum_prod_eq : p*q + p*r + p*s + q*r + q*s + r*s = 24) :
  s ≤ 3 + 3 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_s_l3200_320019


namespace NUMINAMATH_CALUDE_tan_alpha_value_l3200_320075

theorem tan_alpha_value (α : Real) : 
  (π/2 < α ∧ α < π) →  -- α is in the third quadrant
  (Real.tan (π/4 - α) = (2/3) * Real.tan (α + π)) → 
  Real.tan α = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l3200_320075


namespace NUMINAMATH_CALUDE_max_sqrt_sum_l3200_320031

theorem max_sqrt_sum (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 17) :
  Real.sqrt (x + 34) + Real.sqrt (17 - x) + Real.sqrt (2 * x) ≤ Real.sqrt 51 + Real.sqrt 34 :=
by sorry

end NUMINAMATH_CALUDE_max_sqrt_sum_l3200_320031


namespace NUMINAMATH_CALUDE_first_digit_change_largest_l3200_320055

def original : ℚ := 0.1234567

def change_digit (n : ℚ) (pos : ℕ) : ℚ :=
  n + (8 - (n * 10^pos % 10)) / 10^pos

theorem first_digit_change_largest :
  ∀ pos : ℕ, pos > 0 → change_digit original 0 ≥ change_digit original pos :=
by
  sorry

end NUMINAMATH_CALUDE_first_digit_change_largest_l3200_320055


namespace NUMINAMATH_CALUDE_employed_males_percentage_l3200_320085

/-- In a population where 60% are employed and 30% of the employed are females,
    the percentage of employed males in the total population is 42%. -/
theorem employed_males_percentage
  (total : ℕ) -- Total population
  (employed_ratio : ℚ) -- Ratio of employed people to total population
  (employed_females_ratio : ℚ) -- Ratio of employed females to employed people
  (h1 : employed_ratio = 60 / 100)
  (h2 : employed_females_ratio = 30 / 100)
  : (employed_ratio * (1 - employed_females_ratio)) * 100 = 42 := by
  sorry

end NUMINAMATH_CALUDE_employed_males_percentage_l3200_320085


namespace NUMINAMATH_CALUDE_parabola_focus_directrix_l3200_320028

/-- For a parabola y = ax^2 where a > 0, if the distance from the focus to the directrix is 2, then a = 1/4 -/
theorem parabola_focus_directrix (a : ℝ) (h1 : a > 0) : 
  (∃ (f d : ℝ), ∀ (x y : ℝ), y = a * x^2 ∧ |f - d| = 2) → a = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_directrix_l3200_320028


namespace NUMINAMATH_CALUDE_first_discount_percentage_l3200_320025

theorem first_discount_percentage
  (original_price : ℝ)
  (final_price : ℝ)
  (second_discount : ℝ)
  (h1 : original_price = 149.99999999999997)
  (h2 : final_price = 108)
  (h3 : second_discount = 0.2)
  : (original_price - (final_price / (1 - second_discount))) / original_price = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_first_discount_percentage_l3200_320025


namespace NUMINAMATH_CALUDE_total_kernels_needed_l3200_320080

/-- Represents a popcorn preference with its kernel-to-popcorn ratio -/
structure PopcornPreference where
  cups_wanted : ℚ
  kernels : ℚ
  cups_produced : ℚ

/-- Calculates the amount of kernels needed for a given preference -/
def kernels_needed (pref : PopcornPreference) : ℚ :=
  pref.kernels * (pref.cups_wanted / pref.cups_produced)

/-- The list of popcorn preferences for the movie night -/
def movie_night_preferences : List PopcornPreference := [
  ⟨3, 3, 6⟩,  -- Joanie
  ⟨4, 2, 4⟩,  -- Mitchell
  ⟨6, 4, 8⟩,  -- Miles and Davis
  ⟨3, 1, 3⟩   -- Cliff
]

/-- Theorem stating that the total amount of kernels needed is 7.5 tablespoons -/
theorem total_kernels_needed :
  (movie_night_preferences.map kernels_needed).sum = 15/2 := by
  sorry


end NUMINAMATH_CALUDE_total_kernels_needed_l3200_320080


namespace NUMINAMATH_CALUDE_bob_time_improvement_l3200_320065

/-- 
Given Bob's current mile time and his sister's mile time in seconds,
calculate the percentage improvement Bob needs to match his sister's time.
-/
theorem bob_time_improvement (bob_time sister_time : ℕ) :
  bob_time = 640 ∧ sister_time = 608 →
  (bob_time - sister_time : ℚ) / bob_time * 100 = 5 := by
sorry

end NUMINAMATH_CALUDE_bob_time_improvement_l3200_320065


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3200_320002

theorem complex_equation_solution (z : ℂ) (h : (1 + 2*I)*z = 4 + 3*I) : z = 2 - I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3200_320002


namespace NUMINAMATH_CALUDE_scores_mode_is_9_l3200_320063

def scores : List Nat := [7, 10, 9, 8, 7, 9, 9, 8]

def mode (l : List Nat) : Nat :=
  l.foldl (fun acc x => if l.count x > l.count acc then x else acc) 0

theorem scores_mode_is_9 : mode scores = 9 := by sorry

end NUMINAMATH_CALUDE_scores_mode_is_9_l3200_320063


namespace NUMINAMATH_CALUDE_four_monotonic_intervals_condition_l3200_320041

/-- A function f(x) defined by a quadratic expression inside an absolute value. -/
def f (m : ℝ) (x : ℝ) : ℝ := |m * x^2 - (2*m + 1) * x + (m + 2)|

/-- The property of having exactly four monotonic intervals. -/
def has_four_monotonic_intervals (g : ℝ → ℝ) : Prop := sorry

/-- The main theorem stating the conditions on m for f to have exactly four monotonic intervals. -/
theorem four_monotonic_intervals_condition (m : ℝ) :
  has_four_monotonic_intervals (f m) ↔ m < (1/4 : ℝ) ∧ m ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_four_monotonic_intervals_condition_l3200_320041


namespace NUMINAMATH_CALUDE_theater_construction_cost_ratio_l3200_320071

/-- Proves that the ratio of construction cost to land cost is 2:1 given the theater construction scenario --/
theorem theater_construction_cost_ratio :
  let cost_per_sqft : ℝ := 5
  let space_per_seat : ℝ := 12
  let num_seats : ℕ := 500
  let partner_share : ℝ := 0.4
  let tom_spent : ℝ := 54000

  let total_sqft : ℝ := space_per_seat * num_seats
  let land_cost : ℝ := total_sqft * cost_per_sqft
  let total_cost : ℝ := tom_spent / (1 - partner_share)
  let construction_cost : ℝ := total_cost - land_cost

  construction_cost / land_cost = 2 := by sorry

end NUMINAMATH_CALUDE_theater_construction_cost_ratio_l3200_320071


namespace NUMINAMATH_CALUDE_immediate_prepayment_better_l3200_320029

variable (S T r : ℝ)

-- S: initial loan balance
-- T: monthly payment amount
-- r: interest rate for the period

-- Assumption: All variables are positive and r is between 0 and 1
axiom S_pos : S > 0
axiom T_pos : T > 0
axiom r_pos : r > 0
axiom r_lt_one : r < 1

-- Define the final balance for immediate prepayment
def final_balance_immediate (S T r : ℝ) : ℝ :=
  S - 2*T + r*S - 0.5*r*T + (0.5*r*S)^2

-- Define the final balance for waiting until the end of the period
def final_balance_waiting (S T r : ℝ) : ℝ :=
  S - 2*T + r*S

-- Theorem: Immediate prepayment results in a lower final balance
theorem immediate_prepayment_better :
  final_balance_immediate S T r < final_balance_waiting S T r :=
sorry

end NUMINAMATH_CALUDE_immediate_prepayment_better_l3200_320029


namespace NUMINAMATH_CALUDE_sine_of_sum_angle_l3200_320069

theorem sine_of_sum_angle (θ : Real) :
  (∃ (x y : Real), x = -3 ∧ y = 4 ∧ 
   x = Real.cos θ * Real.sqrt (x^2 + y^2) ∧ 
   y = Real.sin θ * Real.sqrt (x^2 + y^2)) →
  Real.sin (θ + π/4) = Real.sqrt 2 / 10 := by
sorry

end NUMINAMATH_CALUDE_sine_of_sum_angle_l3200_320069


namespace NUMINAMATH_CALUDE_pears_amount_correct_l3200_320053

/-- The amount of peaches received by the store in kilograms. -/
def peaches : ℕ := 250

/-- The amount of pears received by the store in kilograms. -/
def pears : ℕ := 100

/-- Theorem stating that the amount of pears is correct given the conditions. -/
theorem pears_amount_correct : peaches = 2 * pears + 50 := by sorry

end NUMINAMATH_CALUDE_pears_amount_correct_l3200_320053


namespace NUMINAMATH_CALUDE_total_legs_of_three_spiders_l3200_320088

def human_legs : ℕ := 2

def spider1_legs : ℕ := 2 * (2 * human_legs)

def spider2_legs : ℕ := 3 * spider1_legs

def spider3_legs : ℕ := spider2_legs - 5

def total_spider_legs : ℕ := spider1_legs + spider2_legs + spider3_legs

theorem total_legs_of_three_spiders :
  total_spider_legs = 51 := by sorry

end NUMINAMATH_CALUDE_total_legs_of_three_spiders_l3200_320088


namespace NUMINAMATH_CALUDE_simplify_expression_l3200_320054

theorem simplify_expression (x : ℝ) : 7*x + 8 - 3*x + 14 = 4*x + 22 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3200_320054


namespace NUMINAMATH_CALUDE_complex_exponential_to_rectangular_l3200_320084

theorem complex_exponential_to_rectangular : Complex.exp (13 * Real.pi * Complex.I / 6) = Complex.mk (Real.sqrt 3 / 2) (1 / 2) := by sorry

end NUMINAMATH_CALUDE_complex_exponential_to_rectangular_l3200_320084


namespace NUMINAMATH_CALUDE_gasoline_spending_increase_l3200_320068

theorem gasoline_spending_increase (P Q : ℝ) (P_new Q_new : ℝ) : 
  P_new = 1.20 * P →
  Q_new = 0.90 * Q →
  P_new * Q_new = 1.08 * (P * Q) :=
sorry

end NUMINAMATH_CALUDE_gasoline_spending_increase_l3200_320068


namespace NUMINAMATH_CALUDE_max_value_inequality_l3200_320099

theorem max_value_inequality (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0)
  (h_sum : x^2 + y^2 + z^2 = 1) :
  2 * x * y * Real.sqrt 8 + 7 * y * z + 5 * x * z ≤ 23.0219 := by
  sorry

end NUMINAMATH_CALUDE_max_value_inequality_l3200_320099


namespace NUMINAMATH_CALUDE_simplify_fraction_l3200_320022

theorem simplify_fraction (a : ℝ) (ha : a > 0) :
  a^2 / (a^(1/2) * a^(2/3)) = a^(5/6) := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3200_320022


namespace NUMINAMATH_CALUDE_bill_difference_zero_l3200_320044

theorem bill_difference_zero (anna_tip : ℝ) (anna_percent : ℝ) 
  (ben_tip : ℝ) (ben_percent : ℝ) 
  (h1 : anna_tip = 5) 
  (h2 : anna_percent = 25 / 100)
  (h3 : ben_tip = 3)
  (h4 : ben_percent = 15 / 100)
  (h5 : anna_tip = anna_percent * anna_bill)
  (h6 : ben_tip = ben_percent * ben_bill) :
  anna_bill - ben_bill = 0 :=
by
  sorry

#check bill_difference_zero

end NUMINAMATH_CALUDE_bill_difference_zero_l3200_320044


namespace NUMINAMATH_CALUDE_unequal_grandchildren_probability_l3200_320086

theorem unequal_grandchildren_probability (n : ℕ) (p_male : ℝ) (p_female : ℝ) : 
  n = 12 →
  p_male = 0.6 →
  p_female = 0.4 →
  p_male + p_female = 1 →
  let p_equal := (n.choose (n / 2)) * (p_male ^ (n / 2)) * (p_female ^ (n / 2))
  1 - p_equal = 0.823 := by
sorry

end NUMINAMATH_CALUDE_unequal_grandchildren_probability_l3200_320086


namespace NUMINAMATH_CALUDE_definite_integral_equality_l3200_320032

theorem definite_integral_equality : 
  let a : Real := 0
  let b : Real := Real.arcsin (Real.sqrt (7/8))
  let f (x : Real) := (6 * Real.sin x ^ 2) / (4 + 3 * Real.cos (2 * x))
  ∫ x in a..b, f x = (Real.sqrt 7 * Real.pi) / 4 - Real.arctan (Real.sqrt 7) := by
  sorry

end NUMINAMATH_CALUDE_definite_integral_equality_l3200_320032


namespace NUMINAMATH_CALUDE_quadratic_polynomial_value_l3200_320095

/-- A quadratic polynomial -/
def QuadraticPolynomial (a b c : ℚ) : ℚ → ℚ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_polynomial_value (a b c : ℚ) :
  let p : ℚ → ℚ := QuadraticPolynomial a b c
  (∀ x : ℚ, (x - 1) * (x + 1) * (x - 8) ∣ p x ^ 3 - x) →
  p 13 = -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_value_l3200_320095


namespace NUMINAMATH_CALUDE_triangles_in_decagon_count_l3200_320007

/-- The number of triangles that can be formed from the vertices of a regular decagon -/
def trianglesInDecagon : ℕ := 120

/-- The number of vertices in a regular decagon -/
def decagonVertices : ℕ := 10

/-- The number of vertices needed to form a triangle -/
def triangleVertices : ℕ := 3

/-- Theorem: The number of triangles that can be formed by selecting 3 vertices
    from a 10-vertex polygon is equal to 120 -/
theorem triangles_in_decagon_count :
  Nat.choose decagonVertices triangleVertices = trianglesInDecagon := by
  sorry

end NUMINAMATH_CALUDE_triangles_in_decagon_count_l3200_320007


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l3200_320013

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Main theorem -/
theorem unique_solution_for_equation :
  ∀ (m n : ℕ), n * (n + 1) = 3^m + sum_of_digits n + 1182 → m = 0 ∧ n = 34 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l3200_320013


namespace NUMINAMATH_CALUDE_a_eq_one_sufficient_not_necessary_l3200_320001

def z (a : ℝ) : ℂ := Complex.mk (a^2 - 1) (a - 2)

theorem a_eq_one_sufficient_not_necessary :
  (∀ a : ℝ, a = 1 → z a = Complex.I * Complex.im (z a)) ∧
  (∃ a : ℝ, a ≠ 1 ∧ z a = Complex.I * Complex.im (z a)) :=
by sorry

end NUMINAMATH_CALUDE_a_eq_one_sufficient_not_necessary_l3200_320001


namespace NUMINAMATH_CALUDE_cyclists_speeds_l3200_320073

/-- Represents the scenario of two cyclists riding towards each other -/
structure CyclistsScenario where
  x : ℝ  -- Speed of the first cyclist in km/h
  y : ℝ  -- Speed of the second cyclist in km/h
  AB : ℝ  -- Distance between the two starting points in km

/-- Condition 1: If the first cyclist starts 1 hour earlier and the second one starts half an hour later,
    they meet 18 minutes earlier than normal -/
def condition1 (s : CyclistsScenario) : Prop :=
  (s.AB / (s.x + s.y) + 1 - 18/60) * s.x + (s.AB / (s.x + s.y) - 1/2 - 18/60) * s.y = s.AB

/-- Condition 2: If the first cyclist starts half an hour later and the second one starts 1 hour earlier,
    the meeting point moves by 11.2 km (11200 meters) -/
def condition2 (s : CyclistsScenario) : Prop :=
  (s.AB - 1.5 * s.y) / (s.x + s.y) * s.x + 11.2 = s.AB / (s.x + s.y) * s.x

/-- Theorem stating that given the conditions, the speeds of the cyclists are 16 km/h and 14 km/h -/
theorem cyclists_speeds (s : CyclistsScenario) 
  (h1 : condition1 s) (h2 : condition2 s) : s.x = 16 ∧ s.y = 14 := by
  sorry

end NUMINAMATH_CALUDE_cyclists_speeds_l3200_320073


namespace NUMINAMATH_CALUDE_circle_P_properties_l3200_320037

/-- Given a circle P with center (a, b) and radius R -/
theorem circle_P_properties (a b R : ℝ) :
  R^2 - b^2 = 2 →
  R^2 - a^2 = 3 →
  (∃ x y : ℝ, y^2 - x^2 = 1) ∧
  (|b - a| = 1 →
    ((∃ x y : ℝ, x^2 + (y - 1)^2 = 3) ∨
     (∃ x y : ℝ, x^2 + (y + 1)^2 = 3))) :=
by sorry

end NUMINAMATH_CALUDE_circle_P_properties_l3200_320037


namespace NUMINAMATH_CALUDE_max_bottles_from_c_and_d_l3200_320027

/-- Represents a shop selling recyclable bottles -/
structure Shop where
  price : ℕ
  available : ℕ

/-- Calculates the total cost of purchasing a given number of bottles from a shop -/
def totalCost (shop : Shop) (bottles : ℕ) : ℕ :=
  shop.price * bottles

theorem max_bottles_from_c_and_d (budget : ℕ) (shopA shopB shopC shopD : Shop) 
  (bottlesA bottlesB : ℕ) :
  budget = 600 ∧
  shopA = { price := 1, available := 200 } ∧
  shopB = { price := 2, available := 150 } ∧
  shopC = { price := 3, available := 100 } ∧
  shopD = { price := 5, available := 50 } ∧
  bottlesA = 150 ∧
  bottlesB = 180 ∧
  bottlesA ≤ shopA.available ∧
  bottlesB ≤ shopB.available →
  ∃ (bottlesC bottlesD : ℕ),
    bottlesC + bottlesD = 30 ∧
    bottlesC ≤ shopC.available ∧
    bottlesD ≤ shopD.available ∧
    totalCost shopA bottlesA + totalCost shopB bottlesB + totalCost shopC bottlesC + totalCost shopD bottlesD = budget ∧
    ∀ (newBottlesC newBottlesD : ℕ),
      newBottlesC ≤ shopC.available →
      newBottlesD ≤ shopD.available →
      totalCost shopA bottlesA + totalCost shopB bottlesB + totalCost shopC newBottlesC + totalCost shopD newBottlesD ≤ budget →
      newBottlesC + newBottlesD ≤ bottlesC + bottlesD :=
by sorry

end NUMINAMATH_CALUDE_max_bottles_from_c_and_d_l3200_320027


namespace NUMINAMATH_CALUDE_gcd_of_2_powers_l3200_320045

theorem gcd_of_2_powers : Nat.gcd (2^2018 - 1) (2^2029 - 1) = 2^11 - 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_2_powers_l3200_320045


namespace NUMINAMATH_CALUDE_class_age_difference_l3200_320093

theorem class_age_difference (n : ℕ) (T : ℕ) : 
  T = n * 40 →
  (T + 408) / (n + 12) = 36 →
  40 - (T + 408) / (n + 12) = 4 :=
by sorry

end NUMINAMATH_CALUDE_class_age_difference_l3200_320093


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l3200_320082

-- Define an even function f: ℝ → ℝ
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- Define an increasing function on (-∞, 0]
def increasing_on_neg (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y ∧ y ≤ 0 → f x ≤ f y

-- Theorem statement
theorem solution_set_of_inequality (f : ℝ → ℝ) 
  (h_even : even_function f) 
  (h_inc : increasing_on_neg f) :
  {x : ℝ | f (x - 1) ≥ f 1} = Set.Icc 0 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l3200_320082


namespace NUMINAMATH_CALUDE_speed_conversion_l3200_320038

/-- Proves that a speed of 0.8 km/h, when expressed as a fraction in m/s with numerator 8, has a denominator of 36 -/
theorem speed_conversion (speed_kmh : ℚ) (speed_ms_num : ℕ) : 
  speed_kmh = 0.8 → speed_ms_num = 8 → 
  ∃ (speed_ms_den : ℕ), 
    (speed_kmh * 1000 / 3600 = speed_ms_num / speed_ms_den) ∧ 
    speed_ms_den = 36 := by
  sorry

end NUMINAMATH_CALUDE_speed_conversion_l3200_320038


namespace NUMINAMATH_CALUDE_optimal_pricing_achieves_target_profit_l3200_320016

/-- Represents the pricing and sales model for a desk lamp in a shopping mall. -/
structure LampSalesModel where
  initial_purchase_price : ℝ
  initial_selling_price : ℝ
  initial_monthly_sales : ℝ
  price_sales_slope : ℝ
  target_monthly_profit : ℝ

/-- Calculates the monthly profit for a given selling price and number of lamps sold. -/
def monthly_profit (model : LampSalesModel) (selling_price : ℝ) (lamps_sold : ℝ) : ℝ :=
  (selling_price - model.initial_purchase_price) * lamps_sold

/-- Calculates the number of lamps sold based on the selling price. -/
def lamps_sold (model : LampSalesModel) (selling_price : ℝ) : ℝ :=
  model.initial_monthly_sales - model.price_sales_slope * (selling_price - model.initial_selling_price)

/-- Theorem stating that the optimal selling price and number of lamps achieve the target monthly profit. -/
theorem optimal_pricing_achieves_target_profit (model : LampSalesModel)
  (h_model : model = {
    initial_purchase_price := 30,
    initial_selling_price := 40,
    initial_monthly_sales := 600,
    price_sales_slope := 10,
    target_monthly_profit := 10000
  })
  (optimal_price : ℝ)
  (optimal_lamps : ℝ)
  (h_price : optimal_price = 50)
  (h_lamps : optimal_lamps = 500) :
  monthly_profit model optimal_price optimal_lamps = model.target_monthly_profit :=
sorry


end NUMINAMATH_CALUDE_optimal_pricing_achieves_target_profit_l3200_320016


namespace NUMINAMATH_CALUDE_set_intersection_conditions_l3200_320079

-- Define the sets A and B
def A : Set ℝ := {y | ∃ x, y = 2*x - 1 ∧ 0 < x ∧ x ≤ 1}
def B (a : ℝ) : Set ℝ := {x | (x - a) * (x - (a + 3)) < 0}

-- State the theorem
theorem set_intersection_conditions (a : ℝ) :
  (A ∩ B a = A ↔ a ∈ Set.Ioc (-2) (-1)) ∧
  (A ∩ B a ≠ ∅ ↔ a ∈ Set.Ioo (-4) 1) :=
sorry

end NUMINAMATH_CALUDE_set_intersection_conditions_l3200_320079


namespace NUMINAMATH_CALUDE_probability_of_shaded_triangle_l3200_320018

/-- Given a set of triangles where some are shaded, this theorem proves
    the probability of selecting a shaded triangle when each triangle
    has an equal probability of being selected. -/
theorem probability_of_shaded_triangle 
  (total_triangles : ℕ) 
  (shaded_triangles : ℕ) 
  (h1 : total_triangles = 8) 
  (h2 : shaded_triangles = 4) 
  (h3 : shaded_triangles ≤ total_triangles) :
  (shaded_triangles : ℚ) / total_triangles = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_shaded_triangle_l3200_320018


namespace NUMINAMATH_CALUDE_gcd_7654321_6789012_l3200_320066

theorem gcd_7654321_6789012 : Nat.gcd 7654321 6789012 = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_7654321_6789012_l3200_320066


namespace NUMINAMATH_CALUDE_otimes_inequality_iff_interval_l3200_320035

/-- Custom binary operation ⊗ on real numbers -/
def otimes (a b : ℝ) : ℝ := a * b + 2 * a + b

/-- Theorem stating the equivalence between the inequality and the interval -/
theorem otimes_inequality_iff_interval (x : ℝ) :
  otimes x (x - 2) < 0 ↔ -2 < x ∧ x < 1 :=
sorry

end NUMINAMATH_CALUDE_otimes_inequality_iff_interval_l3200_320035


namespace NUMINAMATH_CALUDE_three_male_students_probability_l3200_320030

theorem three_male_students_probability 
  (total_male : ℕ) 
  (total_female : ℕ) 
  (selection_size : ℕ) 
  (prob_at_least_one_female : ℚ) : 
  total_male = 4 → 
  total_female = 2 → 
  selection_size = 3 → 
  prob_at_least_one_female = 4/5 → 
  (1 : ℚ) - prob_at_least_one_female = 1/5 := by
sorry

end NUMINAMATH_CALUDE_three_male_students_probability_l3200_320030


namespace NUMINAMATH_CALUDE_intersecting_lines_slope_product_l3200_320011

/-- Given two lines in the xy-plane that intersect at a 30° angle, 
    where the slope of one line is 3 times the slope of the other, 
    the product of their slopes is 1. -/
theorem intersecting_lines_slope_product (m₁ m₂ : ℝ) : 
  m₂ = 3 * m₁ → 
  (|((m₂ - m₁) / (1 + m₁ * m₂))|) = Real.tan (30 * π / 180) → 
  m₁ * m₂ = 1 := by
sorry

end NUMINAMATH_CALUDE_intersecting_lines_slope_product_l3200_320011


namespace NUMINAMATH_CALUDE_action_figure_shelves_l3200_320090

/-- Given a room with action figures and shelves, calculate the number of shelves. -/
theorem action_figure_shelves 
  (total_figures : ℕ) 
  (figures_per_shelf : ℕ) 
  (h1 : total_figures = 120) 
  (h2 : figures_per_shelf = 15) 
  (h3 : figures_per_shelf > 0) : 
  total_figures / figures_per_shelf = 8 := by
  sorry

end NUMINAMATH_CALUDE_action_figure_shelves_l3200_320090


namespace NUMINAMATH_CALUDE_range_of_m_l3200_320020

-- Define propositions p and q
def p (m : ℝ) : Prop := (m - 2) / (m - 3) ≤ 2 / 3

def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 - 4*x + m^2 > 0

-- Define the range of m
def m_range (m : ℝ) : Prop := m < -2 ∨ (0 ≤ m ∧ m ≤ 2) ∨ m ≥ 3

-- Theorem statement
theorem range_of_m :
  ∀ m : ℝ, (p m ∨ q m) ∧ ¬(p m ∧ q m) → m_range m :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l3200_320020


namespace NUMINAMATH_CALUDE_mediant_inequality_l3200_320097

theorem mediant_inequality (a b p q r s : ℕ) 
  (h1 : q * r - p * s = 1) 
  (h2 : (p : ℚ) / q < (a : ℚ) / b) 
  (h3 : (a : ℚ) / b < (r : ℚ) / s) : 
  b ≥ q + s := by
  sorry

end NUMINAMATH_CALUDE_mediant_inequality_l3200_320097


namespace NUMINAMATH_CALUDE_nadia_playing_time_l3200_320043

/-- Represents the number of mistakes Nadia makes per 40 notes -/
def mistakes_per_40_notes : ℚ := 3

/-- Represents the number of notes Nadia can play per minute -/
def notes_per_minute : ℚ := 60

/-- Represents the total number of mistakes Nadia made -/
def total_mistakes : ℚ := 36

/-- Calculates the number of minutes Nadia played -/
def minutes_played : ℚ :=
  total_mistakes / (mistakes_per_40_notes * notes_per_minute / 40)

theorem nadia_playing_time :
  minutes_played = 8 := by sorry

end NUMINAMATH_CALUDE_nadia_playing_time_l3200_320043


namespace NUMINAMATH_CALUDE_smallest_sum_reciprocals_l3200_320081

theorem smallest_sum_reciprocals (x y : ℕ+) : 
  x ≠ y → 
  (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 10 → 
  ∀ a b : ℕ+, a ≠ b → (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 10 → 
  (x : ℕ) + y ≤ (a : ℕ) + b :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_reciprocals_l3200_320081


namespace NUMINAMATH_CALUDE_second_meeting_time_l3200_320003

/-- The time (in seconds) it takes for the racing magic to complete one round -/
def racing_magic_time : ℕ := 60

/-- The time (in seconds) it takes for the charging bull to complete one round -/
def charging_bull_time : ℕ := 90

/-- The time (in minutes) it takes for both objects to meet at the starting point for the second time -/
def meeting_time : ℕ := 3

/-- Theorem stating that the meeting time is correct given the individual round times -/
theorem second_meeting_time (racing_time : ℕ) (bull_time : ℕ) (meet_time : ℕ) 
  (h1 : racing_time = racing_magic_time)
  (h2 : bull_time = charging_bull_time)
  (h3 : meet_time = meeting_time) :
  Nat.lcm racing_time bull_time = meet_time * 60 := by
  sorry

end NUMINAMATH_CALUDE_second_meeting_time_l3200_320003


namespace NUMINAMATH_CALUDE_certain_negative_integer_l3200_320036

theorem certain_negative_integer (a b : ℤ) (x : ℤ) : 
  (-11 * a < 0) →
  (x < 0) →
  (x * b < 0) →
  ((-11 * a * x) * (x * b) + a * b = 89) →
  x = -1 :=
by sorry

end NUMINAMATH_CALUDE_certain_negative_integer_l3200_320036


namespace NUMINAMATH_CALUDE_intersection_point_x_coordinate_l3200_320008

theorem intersection_point_x_coordinate :
  ∀ (x y : ℝ),
  (y = 3 * x - 15) →
  (3 * x + y = 120) →
  (x = 22.5) := by
sorry

end NUMINAMATH_CALUDE_intersection_point_x_coordinate_l3200_320008


namespace NUMINAMATH_CALUDE_parallel_implies_a_eq_2_perpendicular_implies_a_eq_neg3_or_0_l3200_320061

-- Define the lines l₁ and l₂
def l₁ (a x y : ℝ) : Prop := a * x - y + 2 = 0
def l₂ (a x y : ℝ) : Prop := (a + 2) * x - a * y - 2 = 0

-- Define parallel and perpendicular relations
def parallel (a : ℝ) : Prop := ∀ x y, l₁ a x y ↔ l₂ a x y
def perpendicular (a : ℝ) : Prop := ∀ x₁ y₁ x₂ y₂, l₁ a x₁ y₁ → l₂ a x₂ y₂ → 
  (x₂ - x₁) * (y₂ - y₁) = 0

-- Theorem statements
theorem parallel_implies_a_eq_2 : ∀ a : ℝ, parallel a → a = 2 := by sorry

theorem perpendicular_implies_a_eq_neg3_or_0 : ∀ a : ℝ, perpendicular a → a = -3 ∨ a = 0 := by sorry

end NUMINAMATH_CALUDE_parallel_implies_a_eq_2_perpendicular_implies_a_eq_neg3_or_0_l3200_320061


namespace NUMINAMATH_CALUDE_basketball_games_left_to_play_l3200_320048

theorem basketball_games_left_to_play 
  (games_played : ℕ) 
  (win_percentage : ℚ) 
  (additional_losses : ℕ) 
  (final_win_percentage : ℚ) :
  games_played = 40 →
  win_percentage = 70 / 100 →
  additional_losses = 8 →
  final_win_percentage = 60 / 100 →
  ∃ (games_left : ℕ), games_left = 7 ∧ 
    (games_played * win_percentage).floor + (games_played + games_left - (games_played * win_percentage).floor - additional_losses) = 
    (final_win_percentage * (games_played + games_left)).floor :=
by sorry

end NUMINAMATH_CALUDE_basketball_games_left_to_play_l3200_320048


namespace NUMINAMATH_CALUDE_parallel_lines_corresponding_angles_not_always_supplementary_l3200_320076

-- Define the concept of parallel lines
def parallel (l1 l2 : Line) : Prop := sorry

-- Define the concept of corresponding angles
def corresponding_angles (a1 a2 : Angle) (l1 l2 : Line) : Prop := sorry

-- Define the concept of supplementary angles
def supplementary (a1 a2 : Angle) : Prop := sorry

-- The theorem to be proven
theorem parallel_lines_corresponding_angles_not_always_supplementary :
  ¬ ∀ (l1 l2 : Line) (a1 a2 : Angle), 
    parallel l1 l2 → corresponding_angles a1 a2 l1 l2 → supplementary a1 a2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_corresponding_angles_not_always_supplementary_l3200_320076


namespace NUMINAMATH_CALUDE_original_ball_count_original_ball_count_is_960_l3200_320094

-- Define the initial ratio of red to white balls
def initial_ratio : Rat := 19 / 13

-- Define the ratio after adding red balls
def ratio_after_red : Rat := 5 / 3

-- Define the ratio after adding white balls
def ratio_after_white : Rat := 13 / 11

-- Define the difference in added balls
def added_difference : ℕ := 80

-- Theorem statement
theorem original_ball_count : ℕ :=
  let initial_red : ℕ := 57
  let initial_white : ℕ := 39
  let final_red : ℕ := 65
  let final_white : ℕ := 55
  let portion_size : ℕ := added_difference / (final_white - initial_white - (final_red - initial_red))
  (initial_red + initial_white) * portion_size

-- Proof
theorem original_ball_count_is_960 : original_ball_count = 960 := by
  sorry

end NUMINAMATH_CALUDE_original_ball_count_original_ball_count_is_960_l3200_320094


namespace NUMINAMATH_CALUDE_mrs_anderson_pet_food_l3200_320050

/-- Calculates the total ounces of pet food bought by Mrs. Anderson -/
def total_pet_food_ounces (cat_food_bags : ℕ) (cat_food_weight : ℕ) 
  (dog_food_bags : ℕ) (dog_food_extra_weight : ℕ) (ounces_per_pound : ℕ) : ℕ :=
  let total_cat_food := cat_food_bags * cat_food_weight
  let total_dog_food := dog_food_bags * (cat_food_weight + dog_food_extra_weight)
  let total_pounds := total_cat_food + total_dog_food
  total_pounds * ounces_per_pound

/-- Theorem stating that Mrs. Anderson bought 256 ounces of pet food -/
theorem mrs_anderson_pet_food : 
  total_pet_food_ounces 2 3 2 2 16 = 256 := by
  sorry

end NUMINAMATH_CALUDE_mrs_anderson_pet_food_l3200_320050


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l3200_320078

theorem arithmetic_sequence_ratio (a : ℕ+ → ℝ) (q : ℝ) :
  q ≠ 1 →
  (∀ n : ℕ+, a (n + 2) + a (n + 1) = 2 * a n) →
  (∀ n : ℕ+, a (n + 1) = a n * q) →
  q = -2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l3200_320078


namespace NUMINAMATH_CALUDE_quadratic_inequality_solutions_l3200_320077

theorem quadratic_inequality_solutions (a : ℝ) :
  (a = -1 → {x : ℝ | a * x^2 + 5 * x + 6 > 0} = {x : ℝ | -1 < x ∧ x < 6}) ∧
  ({x : ℝ | a * x^2 + 5 * x + 6 > 0} = {x : ℝ | x < -3 ∨ x > -2} → a = 1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solutions_l3200_320077


namespace NUMINAMATH_CALUDE_cuboids_painted_l3200_320040

theorem cuboids_painted (total_faces : ℕ) (faces_per_cuboid : ℕ) : 
  total_faces = 60 → faces_per_cuboid = 6 → total_faces / faces_per_cuboid = 10 := by
  sorry

end NUMINAMATH_CALUDE_cuboids_painted_l3200_320040


namespace NUMINAMATH_CALUDE_max_production_theorem_l3200_320017

/-- Represents a clothing factory -/
structure Factory where
  production_per_month : ℕ
  top_time_ratio : ℕ
  pant_time_ratio : ℕ

/-- Calculates the maximum number of sets two factories can produce in a month -/
def max_production (factory_a factory_b : Factory) : ℕ :=
  sorry

/-- Theorem stating the maximum production of two specific factories -/
theorem max_production_theorem :
  let factory_a : Factory := ⟨2700, 2, 1⟩
  let factory_b : Factory := ⟨3600, 3, 2⟩
  max_production factory_a factory_b = 6700 := by
  sorry

end NUMINAMATH_CALUDE_max_production_theorem_l3200_320017


namespace NUMINAMATH_CALUDE_complementary_angles_l3200_320015

theorem complementary_angles (C D : ℝ) : 
  C + D = 90 →  -- C and D are complementary
  C = 5 * D →   -- C is 5 times D
  C = 75 :=     -- C is 75°
by sorry

end NUMINAMATH_CALUDE_complementary_angles_l3200_320015


namespace NUMINAMATH_CALUDE_scooter_repair_cost_l3200_320074

/-- Calculates the repair cost of a scooter given the conditions of the problem -/
def repair_cost (cost : ℝ) : ℝ :=
  0.1 * cost

/-- Calculates the selling price of a scooter given the conditions of the problem -/
def selling_price (cost : ℝ) : ℝ :=
  1.2 * cost

/-- Theorem stating the repair cost under the given conditions -/
theorem scooter_repair_cost :
  ∀ (cost : ℝ),
  cost > 0 →
  selling_price cost - cost = 1100 →
  repair_cost cost = 550 := by
sorry

end NUMINAMATH_CALUDE_scooter_repair_cost_l3200_320074


namespace NUMINAMATH_CALUDE_equation_solution_l3200_320062

theorem equation_solution (x : ℝ) : 
  (4 * x - 3 > 0) → 
  (Real.sqrt (4 * x - 3) + 12 / Real.sqrt (4 * x - 3) = 8) ↔ 
  (x = 7/4 ∨ x = 39/4) := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l3200_320062


namespace NUMINAMATH_CALUDE_power_product_equality_l3200_320047

theorem power_product_equality : 2^4 * 3^2 * 5^2 * 7^2 = 176400 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equality_l3200_320047


namespace NUMINAMATH_CALUDE_max_revenue_is_50_l3200_320006

def neighborhood_A_homes : ℕ := 10
def neighborhood_A_boxes_per_home : ℕ := 2
def neighborhood_B_homes : ℕ := 5
def neighborhood_B_boxes_per_home : ℕ := 5
def price_per_box : ℕ := 2

def revenue_A : ℕ := neighborhood_A_homes * neighborhood_A_boxes_per_home * price_per_box
def revenue_B : ℕ := neighborhood_B_homes * neighborhood_B_boxes_per_home * price_per_box

theorem max_revenue_is_50 : max revenue_A revenue_B = 50 := by
  sorry

end NUMINAMATH_CALUDE_max_revenue_is_50_l3200_320006


namespace NUMINAMATH_CALUDE_not_square_product_l3200_320023

theorem not_square_product (a : ℕ) : 
  (∀ n : ℕ, ¬∃ m : ℕ, n * (n + a) = m ^ 2) ↔ a = 1 ∨ a = 2 ∨ a = 4 := by
  sorry

end NUMINAMATH_CALUDE_not_square_product_l3200_320023


namespace NUMINAMATH_CALUDE_divisibility_implies_multiple_of_three_l3200_320064

theorem divisibility_implies_multiple_of_three (n : ℕ) :
  n ≥ 2 →
  (∃ k : ℕ, 2^n + 1 = k * n) →
  ∃ m : ℕ, n = 3 * m :=
sorry

end NUMINAMATH_CALUDE_divisibility_implies_multiple_of_three_l3200_320064


namespace NUMINAMATH_CALUDE_class_composition_l3200_320092

theorem class_composition (total_students : ℕ) (girls_ratio boys_ratio : ℕ) 
  (h1 : total_students = 56)
  (h2 : girls_ratio = 4)
  (h3 : boys_ratio = 3) :
  ∃ (girls boys : ℕ), 
    girls + boys = total_students ∧ 
    girls * boys_ratio = boys * girls_ratio ∧
    girls = 32 ∧ 
    boys = 24 :=
by sorry

end NUMINAMATH_CALUDE_class_composition_l3200_320092


namespace NUMINAMATH_CALUDE_train_crossing_time_l3200_320010

/-- Proves that a train with given length and speed takes a specific time to cross a fixed point -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 140 →
  train_speed_kmh = 31.5 →
  crossing_time = 16 →
  train_length / (train_speed_kmh * 1000 / 3600) = crossing_time :=
by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l3200_320010


namespace NUMINAMATH_CALUDE_ants_crushed_calculation_l3200_320091

/-- The number of ants crushed by a man's foot, given the original number of ants and the number of ants left alive -/
def antsCrushed (originalAnts : ℕ) (antsAlive : ℕ) : ℕ :=
  originalAnts - antsAlive

/-- Theorem stating that 60 ants were crushed when 102 ants were originally present and 42 ants remained alive -/
theorem ants_crushed_calculation :
  antsCrushed 102 42 = 60 := by
  sorry

end NUMINAMATH_CALUDE_ants_crushed_calculation_l3200_320091


namespace NUMINAMATH_CALUDE_aquarium_fish_count_l3200_320087

theorem aquarium_fish_count (total : ℕ) (blue orange green : ℕ) : 
  total = 80 ∧ 
  blue = total / 2 ∧ 
  orange = blue - 15 ∧ 
  total = blue + orange + green → 
  green = 15 := by
sorry

end NUMINAMATH_CALUDE_aquarium_fish_count_l3200_320087


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3200_320083

theorem quadratic_inequality_solution_set 
  (a b c : ℝ) 
  (h1 : a ≠ 0) 
  (h2 : Set.Ioo (-2 : ℝ) 3 = {x : ℝ | a * x^2 + b * x + c > 0}) : 
  Set.Ioo (-3 : ℝ) 2 = {x : ℝ | a * x^2 - b * x + c > 0} := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3200_320083
