import Mathlib

namespace shekar_biology_score_l913_91376

/-- Represents a student's scores in various subjects -/
structure StudentScores where
  mathematics : ℕ
  science : ℕ
  socialStudies : ℕ
  english : ℕ
  biology : ℕ

/-- Calculates the average score given a StudentScores instance -/
def calculateAverage (scores : StudentScores) : ℚ :=
  (scores.mathematics + scores.science + scores.socialStudies + scores.english + scores.biology) / 5

/-- Theorem: Given Shekar's scores and average, his biology score must be 95 -/
theorem shekar_biology_score :
  ∀ (scores : StudentScores),
    scores.mathematics = 76 →
    scores.science = 65 →
    scores.socialStudies = 82 →
    scores.english = 67 →
    calculateAverage scores = 77 →
    scores.biology = 95 := by
  sorry


end shekar_biology_score_l913_91376


namespace condition_A_implies_A_eq_pi_third_condition_D_implies_A_eq_pi_third_l913_91399

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Theorem for condition A
theorem condition_A_implies_A_eq_pi_third (t : Triangle) 
  (h1 : t.a = 7) (h2 : t.b = 8) (h3 : t.c = 5) : 
  t.A = π / 3 := by sorry

-- Theorem for condition D
theorem condition_D_implies_A_eq_pi_third (t : Triangle) 
  (h : 2 * Real.sin (t.B / 2 + t.C / 2) ^ 2 + Real.cos (2 * t.A) = 1) : 
  t.A = π / 3 := by sorry

end condition_A_implies_A_eq_pi_third_condition_D_implies_A_eq_pi_third_l913_91399


namespace expected_black_pairs_in_circular_arrangement_l913_91338

/-- The number of cards in the modified deck -/
def total_cards : ℕ := 60

/-- The number of black cards in the deck -/
def black_cards : ℕ := 30

/-- The number of red cards in the deck -/
def red_cards : ℕ := 30

/-- The expected number of pairs of adjacent black cards in a circular arrangement -/
def expected_black_pairs : ℚ := 870 / 59

theorem expected_black_pairs_in_circular_arrangement :
  let total := total_cards
  let black := black_cards
  let red := red_cards
  total = black + red →
  expected_black_pairs = (black * (black - 1) : ℚ) / (total - 1) := by
  sorry

end expected_black_pairs_in_circular_arrangement_l913_91338


namespace ratio_equality_l913_91344

theorem ratio_equality (x y z : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0) 
  (h_diff : x ≠ y ∧ y ≠ z ∧ x ≠ z)
  (h_eq : y / (x + z) = (x + 2*y) / (z + 2*y) ∧ (x + 2*y) / (z + 2*y) = x / (2*y)) :
  x / y = 3 := by
sorry

end ratio_equality_l913_91344


namespace chord_intercept_theorem_l913_91302

def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 4*y - 20 = 0

def line_equation (x y c : ℝ) : Prop :=
  5*x - 12*y + c = 0

def chord_length (c : ℝ) : ℝ := 8

theorem chord_intercept_theorem (c : ℝ) :
  (∃ (x₁ y₁ x₂ y₂ : ℝ),
    circle_equation x₁ y₁ ∧ circle_equation x₂ y₂ ∧
    line_equation x₁ y₁ c ∧ line_equation x₂ y₂ c ∧
    (x₂ - x₁)^2 + (y₂ - y₁)^2 = (chord_length c)^2) ↔
  c = 10 ∨ c = -68 :=
sorry

end chord_intercept_theorem_l913_91302


namespace max_value_a_l913_91362

theorem max_value_a (a b c d : ℕ+) 
  (h1 : a < 3 * b)
  (h2 : b < 4 * c)
  (h3 : c < 5 * d)
  (h4 : d < 80) :
  a ≤ 4724 ∧ ∃ (a' b' c' d' : ℕ+), 
    a' = 4724 ∧ 
    a' < 3 * b' ∧ 
    b' < 4 * c' ∧ 
    c' < 5 * d' ∧ 
    d' < 80 :=
sorry

end max_value_a_l913_91362


namespace multiply_99_105_l913_91312

theorem multiply_99_105 : 99 * 105 = 10395 := by
  sorry

end multiply_99_105_l913_91312


namespace max_value_problem_l913_91357

theorem max_value_problem (x y : ℝ) 
  (h1 : x - y ≥ 2) 
  (h2 : x + y ≤ 3) 
  (h3 : x ≥ 0) 
  (h4 : y ≥ 0) : 
  ∃ (z : ℝ), z = 6 ∧ ∀ (w : ℝ), w = 2*x - 3*y → w ≤ z :=
by sorry

end max_value_problem_l913_91357


namespace work_earnings_equation_l913_91304

theorem work_earnings_equation (t : ℝ) : 
  (t + 1) * (3 * t - 3) = (3 * t - 5) * (t + 2) + 2 → t = 5 := by
  sorry

end work_earnings_equation_l913_91304


namespace doughnuts_per_box_l913_91314

theorem doughnuts_per_box (total_doughnuts : ℕ) (num_boxes : ℕ) 
  (h1 : total_doughnuts = 48)
  (h2 : num_boxes = 4)
  (h3 : total_doughnuts % num_boxes = 0) : 
  total_doughnuts / num_boxes = 12 := by
sorry

end doughnuts_per_box_l913_91314


namespace small_rhombus_area_l913_91333

theorem small_rhombus_area (r : ℝ) (h : r = 10) : 
  let large_rhombus_diagonal := 2 * r
  let small_rhombus_side := large_rhombus_diagonal / 2
  small_rhombus_side ^ 2 = 100 := by sorry

end small_rhombus_area_l913_91333


namespace angle_measure_120_l913_91395

-- Define a triangle
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (hc : c > 0)
  (hA : 0 < A ∧ A < π)
  (hB : 0 < B ∧ B < π)
  (hC : 0 < C ∧ C < π)
  (hsum : A + B + C = π)

-- State the theorem
theorem angle_measure_120 (t : Triangle) (h : t.a^2 = t.b^2 + t.b*t.c + t.c^2) :
  t.A = 2*π/3 := by sorry

end angle_measure_120_l913_91395


namespace yellow_highlighters_count_l913_91345

theorem yellow_highlighters_count (pink : ℕ) (blue : ℕ) (total : ℕ) (yellow : ℕ) : 
  pink = 10 → blue = 8 → total = 33 → yellow = total - (pink + blue) → yellow = 15 := by
  sorry

end yellow_highlighters_count_l913_91345


namespace hexagon_ratio_is_two_l913_91310

/-- Represents a hexagon with specific properties -/
structure Hexagon where
  /-- Total number of unit squares in the hexagon -/
  total_squares : ℕ
  /-- Number of unit squares above the diagonal PQ -/
  squares_above_pq : ℕ
  /-- Base length of the triangle above PQ -/
  triangle_base : ℝ
  /-- Total length of XQ + QY -/
  xq_plus_qy : ℝ
  /-- Condition: The area above PQ is half of the total area -/
  area_condition : squares_above_pq + (triangle_base * triangle_base / 4) = (total_squares + triangle_base * triangle_base / 4) / 2

/-- Theorem: For a hexagon with the given properties, XQ/QY = 2 -/
theorem hexagon_ratio_is_two (h : Hexagon) (h_total : h.total_squares = 8) 
  (h_above : h.squares_above_pq = 3) (h_base : h.triangle_base = 4) (h_xq_qy : h.xq_plus_qy = 4) : 
  ∃ (xq qy : ℝ), xq + qy = h.xq_plus_qy ∧ xq / qy = 2 := by
  sorry

end hexagon_ratio_is_two_l913_91310


namespace component_reliability_l913_91358

/-- Represents the service life of an electronic component in years -/
def ServiceLife : Type := ℝ

/-- The probability that a single electronic component works normally for more than 9 years -/
def ProbSingleComponentWorksOver9Years : ℝ := 0.2

/-- The number of electronic components in parallel -/
def NumComponents : ℕ := 3

/-- The probability that the component (made up of 3 parallel electronic components) 
    can work normally for more than 9 years -/
def ProbComponentWorksOver9Years : ℝ :=
  1 - (1 - ProbSingleComponentWorksOver9Years) ^ NumComponents

theorem component_reliability :
  ProbComponentWorksOver9Years = 0.488 :=
sorry

end component_reliability_l913_91358


namespace georginas_parrots_l913_91352

/-- Represents a parrot with its phrases and learning rate -/
structure Parrot where
  name : String
  current_phrases : ℕ
  phrases_per_week : ℕ
  initial_phrases : ℕ

/-- Calculates the number of weekdays since a parrot was bought -/
def weekdays_since_bought (p : Parrot) : ℕ :=
  ((p.current_phrases - p.initial_phrases + p.phrases_per_week - 1) / p.phrases_per_week) * 5

/-- The main theorem about Georgina's parrots -/
theorem georginas_parrots :
  let polly : Parrot := { name := "Polly", current_phrases := 17, phrases_per_week := 2, initial_phrases := 3 }
  let pedro : Parrot := { name := "Pedro", current_phrases := 12, phrases_per_week := 3, initial_phrases := 0 }
  let penelope : Parrot := { name := "Penelope", current_phrases := 8, phrases_per_week := 1, initial_phrases := 0 }
  let pascal : Parrot := { name := "Pascal", current_phrases := 20, phrases_per_week := 4, initial_phrases := 1 }
  weekdays_since_bought polly = 35 ∧
  weekdays_since_bought pedro = 20 ∧
  weekdays_since_bought penelope = 40 ∧
  weekdays_since_bought pascal = 25 := by
  sorry

end georginas_parrots_l913_91352


namespace common_number_in_overlapping_sets_l913_91382

theorem common_number_in_overlapping_sets (numbers : List ℝ) : 
  numbers.length = 9 →
  (numbers.take 5).sum / 5 = 7 →
  (numbers.drop 4).sum / 5 = 10 →
  numbers.sum / 9 = 74 / 9 →
  ∃ x ∈ numbers.take 5 ∩ numbers.drop 4, x = 11 :=
by sorry

end common_number_in_overlapping_sets_l913_91382


namespace total_area_calculation_l913_91326

/-- Calculates the total area of rooms given initial dimensions and modifications --/
theorem total_area_calculation (initial_length initial_width increase : ℕ) : 
  let new_length : ℕ := initial_length + increase
  let new_width : ℕ := initial_width + increase
  let single_room_area : ℕ := new_length * new_width
  let total_area : ℕ := 4 * single_room_area + 2 * single_room_area
  (initial_length = 13 ∧ initial_width = 18 ∧ increase = 2) → total_area = 1800 := by
  sorry

#check total_area_calculation

end total_area_calculation_l913_91326


namespace tan_alpha_minus_pi_fourth_l913_91361

theorem tan_alpha_minus_pi_fourth (α β : Real) 
  (h1 : Real.tan (α + β) = 2)
  (h2 : Real.tan (β + Real.pi / 4) = 3) :
  Real.tan (α - Real.pi / 4) = -1 / 7 := by
sorry

end tan_alpha_minus_pi_fourth_l913_91361


namespace circle_radius_zero_circle_equation_implies_zero_radius_l913_91356

theorem circle_radius_zero (x y : ℝ) :
  x^2 + 8*x + y^2 - 4*y + 20 = 0 → (x + 4)^2 + (y - 2)^2 = 0 := by
  sorry

theorem circle_equation_implies_zero_radius :
  ∃ (x y : ℝ), x^2 + 8*x + y^2 - 4*y + 20 = 0 → 0 = 0 := by
  sorry

end circle_radius_zero_circle_equation_implies_zero_radius_l913_91356


namespace fraction_sum_simplification_l913_91388

theorem fraction_sum_simplification :
  3 / 840 + 37 / 120 = 131 / 420 := by
sorry

end fraction_sum_simplification_l913_91388


namespace room_population_lower_limit_l913_91340

theorem room_population_lower_limit :
  ∀ (P : ℕ),
  (P < 100) →
  ((3 : ℚ) / 8 * P = 36) →
  (∃ (n : ℕ), (5 : ℚ) / 12 * P = n) →
  P ≥ 96 :=
by
  sorry

end room_population_lower_limit_l913_91340


namespace brandon_job_applications_l913_91308

theorem brandon_job_applications (total_businesses : ℕ) 
  (h1 : total_businesses = 72) 
  (fired : ℕ) (h2 : fired = total_businesses / 2)
  (quit : ℕ) (h3 : quit = total_businesses / 3) : 
  total_businesses - (fired + quit) = 12 :=
by sorry

end brandon_job_applications_l913_91308


namespace arccos_cos_eleven_l913_91363

theorem arccos_cos_eleven : 
  Real.arccos (Real.cos 11) = 11 - 4 * Real.pi + 2 * Real.pi := by
  sorry

end arccos_cos_eleven_l913_91363


namespace tangent_parallel_points_l913_91328

-- Define the curve
def f (x : ℝ) : ℝ := x^3 + x - 2

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 + 1

-- Theorem statement
theorem tangent_parallel_points :
  ∀ x y : ℝ, (y = f x) ∧ (f' x = 4) ↔ (x = 1 ∧ y = 0) ∨ (x = -1 ∧ y = -4) := by
  sorry

end tangent_parallel_points_l913_91328


namespace tangent_line_equation_l913_91313

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - x^2

-- Define the point of tangency
def P : ℝ × ℝ := (2, 4)

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 - 2*x

-- Theorem statement
theorem tangent_line_equation :
  let slope := f' P.1
  let tangent_eq (x y : ℝ) := slope * (x - P.1) - (y - P.2)
  tangent_eq = λ x y => 8*x - y - 12 := by sorry

end tangent_line_equation_l913_91313


namespace orthocenter_ratio_l913_91323

-- Define the triangle XYZ
structure Triangle (X Y Z : ℝ × ℝ) : Prop where
  -- Add any necessary conditions for a valid triangle

-- Define the length of a side
def side_length (A B : ℝ × ℝ) : ℝ := sorry

-- Define the measure of an angle
def angle_measure (A B C : ℝ × ℝ) : ℝ := sorry

-- Define an altitude of a triangle
def altitude (A B C P : ℝ × ℝ) : Prop := sorry

-- Define the orthocenter of a triangle
def orthocenter (X Y Z H : ℝ × ℝ) : Prop := sorry

-- Main theorem
theorem orthocenter_ratio {X Y Z P H : ℝ × ℝ} :
  Triangle X Y Z →
  side_length Y Z = 5 →
  side_length X Z = 4 * Real.sqrt 2 →
  angle_measure X Z Y = π / 4 →
  altitude X Y Z P →
  orthocenter X Y Z H →
  (side_length X H) / (side_length H P) = 3 := by
  sorry

end orthocenter_ratio_l913_91323


namespace power_of_seven_l913_91318

theorem power_of_seven (k : ℕ) (h : 7^k = 2) : 7^(4*k + 2) = 784 := by
  sorry

end power_of_seven_l913_91318


namespace girls_entered_l913_91393

theorem girls_entered (initial_children final_children boys_left : ℕ) 
  (h1 : initial_children = 85)
  (h2 : boys_left = 31)
  (h3 : final_children = 78) :
  final_children - (initial_children - boys_left) = 24 :=
by
  sorry

end girls_entered_l913_91393


namespace flower_pots_theorem_l913_91321

/-- Represents the number of pots of each type of flower seedling --/
structure FlowerPots where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Checks if the given FlowerPots satisfies all conditions --/
def isValid (pots : FlowerPots) : Prop :=
  pots.a > 0 ∧ pots.b > 0 ∧ pots.c > 0 ∧
  pots.a + pots.b + pots.c = 16 ∧
  2 * pots.a + 4 * pots.b + 10 * pots.c = 50

/-- The theorem stating that the only valid numbers of pots for type A are 10 and 13 --/
theorem flower_pots_theorem :
  ∀ pots : FlowerPots, isValid pots → pots.a = 10 ∨ pots.a = 13 := by
  sorry

end flower_pots_theorem_l913_91321


namespace factorization_problem_fraction_simplification_l913_91392

-- Factorization problem
theorem factorization_problem (m : ℝ) : m^3 - 4*m^2 + 4*m = m*(m-2)^2 := by
  sorry

-- Fraction simplification problem
theorem fraction_simplification (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) :
  2 / (x^2 - 1) - 1 / (x - 1) = -1 / (x + 1) := by
  sorry

end factorization_problem_fraction_simplification_l913_91392


namespace infinitely_many_a_composite_sum_l913_91346

theorem infinitely_many_a_composite_sum : ∃ f : ℕ → ℕ, 
  (∀ k : ℕ, f k > f (k - 1)) ∧ 
  (∀ a : ℕ, ∃ m : ℕ, a = f m) ∧
  (∀ a : ℕ, a ∈ Set.range f → ∀ n : ℕ, ∃ x y : ℕ, x > 1 ∧ y > 1 ∧ n^4 + a = x * y) :=
by sorry

end infinitely_many_a_composite_sum_l913_91346


namespace sequence_sum_problem_l913_91397

theorem sequence_sum_problem (x₁ x₂ x₃ x₄ x₅ x₆ x₇ : ℝ) 
  (eq1 : x₁ + 4*x₂ + 9*x₃ + 16*x₄ + 25*x₅ + 36*x₆ + 49*x₇ = 5)
  (eq2 : 4*x₁ + 9*x₂ + 16*x₃ + 25*x₄ + 36*x₅ + 49*x₆ + 64*x₇ = 14)
  (eq3 : 9*x₁ + 16*x₂ + 25*x₃ + 36*x₄ + 49*x₅ + 64*x₆ + 81*x₇ = 30)
  (eq4 : 16*x₁ + 25*x₂ + 36*x₃ + 49*x₄ + 64*x₅ + 81*x₆ + 100*x₇ = 70) :
  25*x₁ + 36*x₂ + 49*x₃ + 64*x₄ + 81*x₅ + 100*x₆ + 121*x₇ = 130 := by
  sorry


end sequence_sum_problem_l913_91397


namespace new_room_size_l913_91360

theorem new_room_size (bedroom_size : ℝ) (bathroom_size : ℝ) 
  (h1 : bedroom_size = 309) 
  (h2 : bathroom_size = 150) : 
  2 * (bedroom_size + bathroom_size) = 918 := by
  sorry

end new_room_size_l913_91360


namespace unique_number_satisfying_conditions_l913_91369

theorem unique_number_satisfying_conditions : ∃! n : ℕ,
  35 < n ∧ n < 70 ∧ (n - 3) % 6 = 0 ∧ (n - 1) % 8 = 0 :=
by
  -- The proof goes here
  sorry

end unique_number_satisfying_conditions_l913_91369


namespace negative_five_plus_abs_negative_three_equals_negative_two_l913_91391

theorem negative_five_plus_abs_negative_three_equals_negative_two :
  -5 + |(-3)| = -2 := by
  sorry

end negative_five_plus_abs_negative_three_equals_negative_two_l913_91391


namespace largest_s_value_l913_91339

/-- The interior angle of a regular n-gon -/
def interior_angle (n : ℕ) : ℚ :=
  (n - 2 : ℚ) * 180 / n

/-- The largest possible value of s for regular polygons Q_1 (r-gon) and Q_2 (s-gon) -/
theorem largest_s_value (r s : ℕ) (hr : r ≥ s) (hs : s ≥ 3) 
  (h_ratio : interior_angle r / interior_angle s = 39 / 38) : 
  s ≤ 76 ∧ ∃ (r' : ℕ), r' ≥ 76 ∧ interior_angle r' / interior_angle 76 = 39 / 38 :=
sorry

end largest_s_value_l913_91339


namespace baking_scoops_calculation_l913_91370

/-- Calculates the total number of scoops needed for baking a cake --/
def total_scoops (flour_cups : ℚ) (sugar_cups : ℚ) (scoop_size : ℚ) : ℕ :=
  (flour_cups / scoop_size + sugar_cups / scoop_size).ceil.toNat

/-- Proves that given 3 cups of flour, 2 cups of sugar, and a 1/3 cup scoop, 
    the total number of scoops needed is 15 --/
theorem baking_scoops_calculation : 
  total_scoops 3 2 (1/3) = 15 := by sorry

end baking_scoops_calculation_l913_91370


namespace circle_configuration_l913_91329

-- Define the types of people
inductive PersonType
| Knight
| Liar
| Visitor

-- Define a person
structure Person where
  id : Fin 7
  type : PersonType

-- Define the circle of people
def Circle := Fin 7 → Person

-- Define a statement made by a pair of people
structure Statement where
  speaker1 : Fin 7
  speaker2 : Fin 7
  content : Nat
  category : PersonType

-- Define the function to check if a statement is true
def isStatementTrue (c : Circle) (s : Statement) : Prop :=
  (c s.speaker1).type = PersonType.Knight ∨
  (c s.speaker2).type = PersonType.Knight ∨
  ((c s.speaker1).type = PersonType.Visitor ∧ (c s.speaker2).type = PersonType.Visitor)

-- Define the list of statements
def statements : List Statement := [
  ⟨0, 1, 1, PersonType.Liar⟩,
  ⟨1, 2, 2, PersonType.Knight⟩,
  ⟨2, 3, 3, PersonType.Liar⟩,
  ⟨3, 4, 4, PersonType.Knight⟩,
  ⟨4, 5, 5, PersonType.Liar⟩,
  ⟨5, 6, 6, PersonType.Knight⟩,
  ⟨6, 0, 7, PersonType.Liar⟩
]

-- Define the theorem
theorem circle_configuration (c : Circle) :
  (∀ s ∈ statements, isStatementTrue c s ∨ ¬isStatementTrue c s) →
  (∃! (i j : Fin 7), i ≠ j ∧ 
    (c i).type = PersonType.Visitor ∧ 
    (c j).type = PersonType.Visitor ∧
    (∀ k : Fin 7, k ≠ i ∧ k ≠ j → (c k).type = PersonType.Liar)) :=
by
  sorry


end circle_configuration_l913_91329


namespace quadratic_roots_characterization_l913_91385

/-- The quadratic equation a² - 18a + 72 = 0 has solutions a = 6 and a = 12 -/
def quad_eq (a : ℝ) : Prop := a^2 - 18*a + 72 = 0

/-- The general form of the roots -/
def root_form (a x : ℝ) : Prop := x = a + Real.sqrt (18*(a-4)) ∨ x = a - Real.sqrt (18*(a-4))

/-- Condition for distinct positive roots -/
def distinct_positive_roots (a : ℝ) : Prop :=
  (4 < a ∧ a < 6) ∨ a > 12

/-- Condition for equal roots -/
def equal_roots (a : ℝ) : Prop :=
  (6 ≤ a ∧ a ≤ 12) ∨ a = 22

theorem quadratic_roots_characterization :
  ∀ a : ℝ, quad_eq a →
    (∃ x y : ℝ, x ≠ y ∧ root_form a x ∧ root_form a y ∧ x > 0 ∧ y > 0 ↔ distinct_positive_roots a) ∧
    (∃ x : ℝ, root_form a x ∧ x > 0 ↔ equal_roots a) :=
sorry

end quadratic_roots_characterization_l913_91385


namespace inequality_solution_l913_91381

-- Define the solution set for x^2 - ax - b < 0
def solution_set (a b : ℝ) : Set ℝ := {x | 2 < x ∧ x < 3}

-- Theorem statement
theorem inequality_solution :
  ∀ a b : ℝ, (solution_set a b = {x | 2 < x ∧ x < 3}) →
  (a = 5 ∧ b = -6) ∧
  ({x : ℝ | b * x^2 - a * x - 1 > 0} = {x : ℝ | -1/2 < x ∧ x < -1/3}) :=
by sorry

end inequality_solution_l913_91381


namespace no_base6_digit_divisible_by_7_l913_91303

/-- Converts a base-6 number to base-10 --/
def base6ToBase10 (d : ℕ) : ℕ := 3 * 6^3 + d * 6^2 + d * 6 + 6

/-- Represents a base-6 digit --/
def isBase6Digit (d : ℕ) : Prop := d ≥ 0 ∧ d ≤ 5

theorem no_base6_digit_divisible_by_7 : 
  ∀ d : ℕ, isBase6Digit d → ¬(base6ToBase10 d % 7 = 0) := by
  sorry

#check no_base6_digit_divisible_by_7

end no_base6_digit_divisible_by_7_l913_91303


namespace left_number_20th_row_l913_91368

/- Define the sequence of numbers in the array -/
def array_sequence (n : ℕ) : ℕ := n^2

/- Define the sum of numbers in the first n rows -/
def sum_of_rows (n : ℕ) : ℕ := n^2

/- Define the number on the far left of the nth row -/
def left_number (n : ℕ) : ℕ := sum_of_rows (n - 1) + 1

/- Theorem statement -/
theorem left_number_20th_row : left_number 20 = 362 := by
  sorry

end left_number_20th_row_l913_91368


namespace animal_shelter_dogs_l913_91365

theorem animal_shelter_dogs (initial_dogs : ℕ) (adoption_rate : ℚ) (returned_dogs : ℕ) : 
  initial_dogs = 80 → 
  adoption_rate = 2/5 →
  returned_dogs = 5 →
  initial_dogs - (initial_dogs * adoption_rate).floor + returned_dogs = 53 := by
sorry

end animal_shelter_dogs_l913_91365


namespace workout_schedule_l913_91331

theorem workout_schedule (x : ℝ) 
  (h1 : x > 0)  -- Workout duration is positive
  (h2 : x + (x - 2) + 2*x + 2*(x - 2) = 18) :  -- Total workout time is 18 hours
  x = 4 := by
sorry

end workout_schedule_l913_91331


namespace prob_select_all_leaders_in_district_l913_91330

/-- Represents a math club with a given number of students and leaders -/
structure MathClub where
  students : Nat
  leaders : Nat

/-- Calculates the probability of selecting all leaders in a given club -/
def prob_select_all_leaders (club : MathClub) : Rat :=
  (club.students - club.leaders).choose 1 / club.students.choose 4

/-- The list of math clubs in the school district -/
def math_clubs : List MathClub := [
  ⟨6, 3⟩,
  ⟨8, 3⟩,
  ⟨9, 3⟩,
  ⟨10, 3⟩
]

/-- The main theorem stating the probability of selecting all leaders -/
theorem prob_select_all_leaders_in_district : 
  (1 / 4 : Rat) * (math_clubs.map prob_select_all_leaders).sum = 37 / 420 := by
  sorry

end prob_select_all_leaders_in_district_l913_91330


namespace beth_crayons_l913_91380

/-- The number of crayons Beth has altogether -/
def total_crayons (packs : ℕ) (crayons_per_pack : ℕ) (extra_crayons : ℕ) : ℕ :=
  packs * crayons_per_pack + extra_crayons

/-- Theorem stating that Beth has 175 crayons in total -/
theorem beth_crayons : total_crayons 8 20 15 = 175 := by
  sorry

end beth_crayons_l913_91380


namespace meal_profit_and_purchase_theorem_l913_91366

/-- Represents the profit for meals A and B -/
structure MealProfit where
  a : ℝ
  b : ℝ

/-- Represents the purchase quantities for meals A and B -/
structure PurchaseQuantity where
  a : ℝ
  b : ℝ

/-- Conditions for the meal profit problem -/
def meal_profit_conditions (p : MealProfit) : Prop :=
  p.a + 2 * p.b = 35 ∧ 2 * p.a + 3 * p.b = 60

/-- Conditions for the meal purchase problem -/
def meal_purchase_conditions (q : PurchaseQuantity) : Prop :=
  q.a + q.b = 1000 ∧ q.a ≤ 3/2 * q.b

/-- The theorem to be proved -/
theorem meal_profit_and_purchase_theorem 
  (p : MealProfit) 
  (q : PurchaseQuantity) 
  (h1 : meal_profit_conditions p) 
  (h2 : meal_purchase_conditions q) :
  p.a = 15 ∧ 
  p.b = 10 ∧ 
  q.a = 600 ∧ 
  q.b = 400 ∧ 
  p.a * q.a + p.b * q.b = 13000 := by
  sorry

end meal_profit_and_purchase_theorem_l913_91366


namespace intersection_of_perpendicular_lines_l913_91317

/-- Given two lines in a plane, where one is perpendicular to the other and passes through a specific point, this theorem proves that their intersection point is as calculated. -/
theorem intersection_of_perpendicular_lines 
  (line1 : ℝ → ℝ)
  (line2 : ℝ → ℝ)
  (h1 : ∀ x, line1 x = -3 * x + 4)
  (h2 : ∀ x, line2 x = (1/3) * x - 1)
  (h3 : line2 3 = -2)
  (h4 : ∀ x y, line1 x = y → line2 x = y → x = 1.5 ∧ y = -0.5) :
  ∃ x y, line1 x = y ∧ line2 x = y ∧ x = 1.5 ∧ y = -0.5 := by
sorry


end intersection_of_perpendicular_lines_l913_91317


namespace cake_remaining_l913_91348

theorem cake_remaining (alex_portion jordan_portion remaining_portion : ℚ) : 
  alex_portion = 40 / 100 →
  jordan_portion = (1 - alex_portion) / 2 →
  remaining_portion = 1 - alex_portion - jordan_portion →
  remaining_portion = 30 / 100 := by
  sorry

end cake_remaining_l913_91348


namespace transform_equation_5x2_eq_6x_minus_8_l913_91307

/-- Represents a quadratic equation in general form ax² + bx + c = 0 --/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  h : a ≠ 0

/-- Transforms an equation of the form px² = qx + r into general quadratic form --/
def transform_to_general_form (p q r : ℝ) (hp : p ≠ 0) : QuadraticEquation :=
  { a := p
  , b := -q
  , c := r
  , h := hp }

theorem transform_equation_5x2_eq_6x_minus_8 :
  let eq := transform_to_general_form 5 6 (-8) (by norm_num)
  eq.a = 5 ∧ eq.b = -6 ∧ eq.c = 8 := by sorry

end transform_equation_5x2_eq_6x_minus_8_l913_91307


namespace line_equation_correct_l913_91341

/-- A line in the 2D plane represented by its slope and a point it passes through. -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- The equation of a line in slope-intercept form (y = mx + b). -/
def lineEquation (l : Line) : ℝ → ℝ := fun x => l.slope * x + (l.point.2 - l.slope * l.point.1)

theorem line_equation_correct (l : Line) : 
  l.slope = 3 ∧ l.point = (-2, 0) → lineEquation l = fun x => 3 * x + 6 := by
  sorry

end line_equation_correct_l913_91341


namespace complement_of_N_in_M_l913_91387

def M : Set Nat := {0, 1, 2, 3, 4, 5}
def N : Set Nat := {0, 2, 3}

theorem complement_of_N_in_M :
  M \ N = {1, 4, 5} := by sorry

end complement_of_N_in_M_l913_91387


namespace shipment_size_l913_91322

/-- The total number of novels in the shipment -/
def total_novels : ℕ := 300

/-- The fraction of novels displayed in the storefront -/
def display_fraction : ℚ := 30 / 100

/-- The number of novels in the storage room -/
def storage_novels : ℕ := 210

/-- Theorem stating that the total number of novels is 300 -/
theorem shipment_size :
  total_novels = 300 ∧
  display_fraction = 30 / 100 ∧
  storage_novels = 210 ∧
  (1 - display_fraction) * total_novels = storage_novels :=
by sorry

end shipment_size_l913_91322


namespace water_amount_in_new_recipe_l913_91389

/-- Represents the ratio of ingredients in a recipe -/
structure RecipeRatio where
  flour : ℚ
  water : ℚ
  sugar : ℚ

/-- The original recipe ratio -/
def original_ratio : RecipeRatio := ⟨7, 2, 1⟩

/-- The new recipe ratio -/
def new_ratio : RecipeRatio :=
  let flour_water_ratio := original_ratio.flour / original_ratio.water
  let flour_sugar_ratio := original_ratio.flour / original_ratio.sugar
  ⟨original_ratio.flour,
   original_ratio.flour / (2 * flour_water_ratio),
   original_ratio.flour / (flour_sugar_ratio / 2)⟩

/-- The amount of sugar in the new recipe -/
def sugar_amount : ℚ := 4

theorem water_amount_in_new_recipe :
  (sugar_amount * new_ratio.water / new_ratio.sugar) = 2 := by
  sorry

end water_amount_in_new_recipe_l913_91389


namespace area_of_triangle_BQW_l913_91342

/-- Given a rectangle ABCD with the following properties:
    - AZ = WC = 8 units
    - AB = 16 units
    - Area of trapezoid ZWCD is 160 square units
    - Q divides ZW in the ratio 1:3 starting from Z
    Prove that the area of triangle BQW is 16 square units. -/
theorem area_of_triangle_BQW (AZ WC AB : ℝ) (area_ZWCD : ℝ) (Q : ℝ) :
  AZ = 8 →
  WC = 8 →
  AB = 16 →
  area_ZWCD = 160 →
  Q = 2 →  -- This represents Q dividing ZW in 1:3 ratio
  (1/2 : ℝ) * AB * Q = 16 := by
  sorry

end area_of_triangle_BQW_l913_91342


namespace max_area_and_optimal_length_l913_91386

/-- Represents the dimensions and cost of a simple house. -/
structure SimpleHouse where
  x : ℝ  -- Length of front wall
  y : ℝ  -- Length of side wall
  h : ℝ  -- Height of walls
  colorSteelPrice : ℝ  -- Price per meter of color steel
  compositeSteelPrice : ℝ  -- Price per meter of composite steel
  roofPrice : ℝ  -- Price per square meter of roof material
  maxCost : ℝ  -- Maximum allowed cost

/-- Calculates the total material cost of the house. -/
def materialCost (h : SimpleHouse) : ℝ :=
  2 * h.x * h.colorSteelPrice * h.h + 
  2 * h.y * h.compositeSteelPrice * h.h + 
  h.x * h.y * h.roofPrice

/-- Calculates the area of the house. -/
def area (h : SimpleHouse) : ℝ := h.x * h.y

/-- Theorem stating the maximum area and optimal front wall length. -/
theorem max_area_and_optimal_length (h : SimpleHouse) 
    (h_height : h.h = 2.5)
    (h_colorSteel : h.colorSteelPrice = 450)
    (h_compositeSteel : h.compositeSteelPrice = 200)
    (h_roof : h.roofPrice = 200)
    (h_maxCost : h.maxCost = 32000)
    (h_cost_constraint : materialCost h ≤ h.maxCost) :
    ∃ (maxArea : ℝ) (optimalLength : ℝ),
      maxArea = 100 ∧
      optimalLength = 20 / 3 ∧
      ∀ (x y : ℝ), 
        x > 0 → y > 0 → 
        materialCost { h with x := x, y := y } ≤ h.maxCost →
        area { h with x := x, y := y } ≤ maxArea ∧
        (area { h with x := x, y := y } = maxArea → x = optimalLength) :=
  sorry

end max_area_and_optimal_length_l913_91386


namespace problem_solution_l913_91355

theorem problem_solution (m : ℤ) (a b c : ℝ) 
  (h1 : ∃! (x : ℤ), |2 * (x : ℝ) - m| ≤ 1 ∧ x = 2)
  (h2 : 4 * a^4 + 4 * b^4 + 4 * c^4 = m) :
  m = 4 ∧ (a^2 + b^2 + c^2 ≤ Real.sqrt 3 ∧ ∃ x y z, x^2 + y^2 + z^2 = Real.sqrt 3) :=
by sorry

end problem_solution_l913_91355


namespace range_of_a_for_surjective_f_l913_91390

/-- A piecewise function f dependent on parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (1 - 2*a)*x + 3*a else 2^(x - 1)

/-- The theorem stating the relationship between the range of f and the range of a -/
theorem range_of_a_for_surjective_f :
  (∀ a : ℝ, Set.range (f a) = Set.univ) ↔ (Set.Icc 0 (1/2) : Set ℝ) = {a : ℝ | 0 ≤ a ∧ a < 1/2} :=
sorry

end range_of_a_for_surjective_f_l913_91390


namespace max_value_of_s_l913_91327

theorem max_value_of_s (x y : ℝ) (h : 4 * x^2 - 5 * x * y + 4 * y^2 = 5) :
  ∃ (s_max : ℝ), s_max = (10 : ℝ) / 3 ∧ ∀ (s : ℝ), s = x^2 + y^2 → s ≤ s_max :=
sorry

end max_value_of_s_l913_91327


namespace divisibility_by_seven_l913_91301

theorem divisibility_by_seven (A a b : ℕ) : A = 100 * a + b →
  (7 ∣ A ↔ 7 ∣ (2 * a + b)) ∧ (7 ∣ A ↔ 7 ∣ (5 * a - b)) := by
  sorry

end divisibility_by_seven_l913_91301


namespace smallest_n_inequality_l913_91324

theorem smallest_n_inequality (x y z w : ℝ) : 
  ∃ (n : ℕ), n = 4 ∧ 
  (∀ (m : ℕ), m < n → ∃ (a b c d : ℝ), (a^2 + b^2 + c^2 + d^2)^2 > m*(a^4 + b^4 + c^4 + d^4)) ∧
  (x^2 + y^2 + z^2 + w^2)^2 ≤ n*(x^4 + y^4 + z^4 + w^4) :=
sorry

end smallest_n_inequality_l913_91324


namespace set_mean_given_median_l913_91354

theorem set_mean_given_median (n : ℝ) :
  (Finset.range 5).card = 5 →
  n + 8 = 14 →
  let s := {n, n + 6, n + 8, n + 10, n + 18}
  (Finset.filter (λ x => x ≤ n + 8) s).card = 3 →
  (Finset.sum s id) / 5 = 14.4 := by
sorry

end set_mean_given_median_l913_91354


namespace cube_sum_is_90_l913_91305

-- Define the type for the cube faces
def CubeFaces := Fin 6 → ℕ

-- Define the property of consecutive even numbers
def ConsecutiveEven (faces : CubeFaces) : Prop :=
  ∃ n : ℕ, ∀ i : Fin 6, faces i = 2 * (n + i.val)

-- Define the property of opposite face sums being equal
def OppositeFaceSumsEqual (faces : CubeFaces) : Prop :=
  ∃ s : ℕ, 
    faces 0 + faces 5 + 2 = s ∧
    faces 1 + faces 4 + 2 = s ∧
    faces 2 + faces 3 + 2 = s

-- Theorem statement
theorem cube_sum_is_90 (faces : CubeFaces) 
  (h1 : ConsecutiveEven faces) 
  (h2 : OppositeFaceSumsEqual faces) : 
  (faces 0 + faces 1 + faces 2 + faces 3 + faces 4 + faces 5 = 90) :=
sorry

end cube_sum_is_90_l913_91305


namespace fewer_servings_l913_91396

def total_ounces : ℕ := 64
def old_serving_size : ℕ := 8
def new_serving_size : ℕ := 16

theorem fewer_servings :
  (total_ounces / old_serving_size) - (total_ounces / new_serving_size) = 4 :=
by sorry

end fewer_servings_l913_91396


namespace window_width_calculation_l913_91334

/-- Represents the dimensions of a glass pane -/
structure Pane where
  height : ℝ
  width : ℝ

/-- Represents the dimensions of a window -/
structure Window where
  rows : ℕ
  columns : ℕ
  pane : Pane
  border_width : ℝ

/-- Calculates the width of a window given its specifications -/
def window_width (w : Window) : ℝ :=
  w.columns * w.pane.width + (w.columns + 1) * w.border_width

/-- Theorem stating the width of the window with given specifications -/
theorem window_width_calculation (x : ℝ) :
  let w : Window := {
    rows := 3,
    columns := 4,
    pane := { height := 4 * x, width := 3 * x },
    border_width := 3
  }
  window_width w = 12 * x + 15 := by sorry

end window_width_calculation_l913_91334


namespace arithmetic_sequence_average_l913_91325

/-- 
Given an arithmetic sequence with:
- First term a₁ = 10
- Last term aₙ = 160
- Common difference d = 10

Prove that the average (arithmetic mean) of this sequence is 85.
-/
theorem arithmetic_sequence_average : 
  let a₁ : ℕ := 10
  let aₙ : ℕ := 160
  let d : ℕ := 10
  let n : ℕ := (aₙ - a₁) / d + 1
  (a₁ + aₙ) / 2 = 85 := by
  sorry

end arithmetic_sequence_average_l913_91325


namespace two_people_available_l913_91316

-- Define the types for people and days
inductive Person : Type
| Anna : Person
| Bill : Person
| Carl : Person
| Dana : Person

inductive Day : Type
| Monday : Day
| Tuesday : Day
| Wednesday : Day
| Thursday : Day
| Friday : Day
| Saturday : Day

-- Define a function to represent availability
def isAvailable : Person → Day → Bool
| Person.Anna, Day.Monday => false
| Person.Anna, Day.Tuesday => true
| Person.Anna, Day.Wednesday => false
| Person.Anna, Day.Thursday => true
| Person.Anna, Day.Friday => true
| Person.Anna, Day.Saturday => false
| Person.Bill, Day.Monday => true
| Person.Bill, Day.Tuesday => false
| Person.Bill, Day.Wednesday => true
| Person.Bill, Day.Thursday => false
| Person.Bill, Day.Friday => false
| Person.Bill, Day.Saturday => true
| Person.Carl, Day.Monday => false
| Person.Carl, Day.Tuesday => false
| Person.Carl, Day.Wednesday => true
| Person.Carl, Day.Thursday => false
| Person.Carl, Day.Friday => false
| Person.Carl, Day.Saturday => true
| Person.Dana, Day.Monday => true
| Person.Dana, Day.Tuesday => true
| Person.Dana, Day.Wednesday => false
| Person.Dana, Day.Thursday => true
| Person.Dana, Day.Friday => true
| Person.Dana, Day.Saturday => false

-- Define a function to count available people for a given day
def countAvailable (d : Day) : Nat :=
  List.foldl (λ count p => if isAvailable p d then count + 1 else count) 0 [Person.Anna, Person.Bill, Person.Carl, Person.Dana]

-- Theorem: For each day, exactly 2 people can attend the meeting
theorem two_people_available (d : Day) : countAvailable d = 2 := by
  sorry

#eval [Day.Monday, Day.Tuesday, Day.Wednesday, Day.Thursday, Day.Friday, Day.Saturday].map countAvailable

end two_people_available_l913_91316


namespace P_intersect_Q_eq_P_l913_91375

def P : Set ℝ := {-1, 0, 1}
def Q : Set ℝ := {y | ∃ x, y = Real.cos x}

theorem P_intersect_Q_eq_P : P ∩ Q = P := by sorry

end P_intersect_Q_eq_P_l913_91375


namespace max_sum_of_factors_l913_91315

theorem max_sum_of_factors (A B C : ℕ) : 
  A ≠ B ∧ B ≠ C ∧ A ≠ C →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A * B * C = 2550 →
  A + B + C ≤ 98 := by
sorry

end max_sum_of_factors_l913_91315


namespace fixed_point_of_exponential_function_l913_91332

theorem fixed_point_of_exponential_function (a : ℝ) (h : a > 0) :
  let f : ℝ → ℝ := λ x ↦ a^(4-x) + 3
  f 4 = 4 := by
sorry

end fixed_point_of_exponential_function_l913_91332


namespace second_hand_movement_l913_91372

/-- Represents the number of seconds it takes for the second hand to move from one number to another on a clock face -/
def secondsBetweenNumbers (start finish : Nat) : Nat :=
  ((finish - start + 12) % 12) * 5

theorem second_hand_movement : secondsBetweenNumbers 5 9 ≠ 4 := by
  sorry

end second_hand_movement_l913_91372


namespace cylinder_height_relationship_l913_91320

theorem cylinder_height_relationship (r₁ h₁ r₂ h₂ : ℝ) :
  r₁ > 0 ∧ h₁ > 0 ∧ r₂ > 0 ∧ h₂ > 0 →
  r₂ = 1.2 * r₁ →
  π * r₁^2 * h₁ = π * r₂^2 * h₂ →
  h₁ = 1.44 * h₂ := by
sorry

end cylinder_height_relationship_l913_91320


namespace square_difference_l913_91398

theorem square_difference (a b : ℝ) (h1 : a + b = 8) (h2 : a - b = 4) : a^2 - b^2 = 32 := by
  sorry

end square_difference_l913_91398


namespace total_blue_marbles_l913_91349

/-- The total number of blue marbles owned by Jason, Tom, and Emily is 104. -/
theorem total_blue_marbles (jason_blue : ℕ) (tom_blue : ℕ) (emily_blue : ℕ)
  (h1 : jason_blue = 44)
  (h2 : tom_blue = 24)
  (h3 : emily_blue = 36) :
  jason_blue + tom_blue + emily_blue = 104 :=
by sorry

end total_blue_marbles_l913_91349


namespace dogs_count_l913_91353

/-- Represents the number of animals in a pet shop -/
structure PetShop where
  dogs : ℕ
  cats : ℕ
  bunnies : ℕ

/-- The ratio of dogs to cats to bunnies is 4 : 7 : 9 -/
def ratio_condition (shop : PetShop) : Prop :=
  ∃ (x : ℕ), shop.dogs = 4 * x ∧ shop.cats = 7 * x ∧ shop.bunnies = 9 * x

/-- The total number of dogs and bunnies is 364 -/
def total_condition (shop : PetShop) : Prop :=
  shop.dogs + shop.bunnies = 364

/-- Theorem stating that under the given conditions, there are 112 dogs -/
theorem dogs_count (shop : PetShop) 
  (h_ratio : ratio_condition shop) 
  (h_total : total_condition shop) : 
  shop.dogs = 112 := by
  sorry

end dogs_count_l913_91353


namespace unique_a_value_l913_91350

def A (a : ℝ) : Set ℝ := {1, 3, a^2}
def B (a : ℝ) : Set ℝ := {1, a+2}

theorem unique_a_value : ∀ a : ℝ, A a ∩ B a = B a → a = 2 := by
  sorry

end unique_a_value_l913_91350


namespace correct_num_persons_first_group_l913_91335

/-- The number of persons in the first group that can repair a road -/
def num_persons_first_group : ℕ := 39

/-- The number of days the first group works -/
def days_first_group : ℕ := 12

/-- The number of hours per day the first group works -/
def hours_per_day_first_group : ℕ := 5

/-- The number of persons in the second group -/
def num_persons_second_group : ℕ := 30

/-- The number of days the second group works -/
def days_second_group : ℕ := 13

/-- The number of hours per day the second group works -/
def hours_per_day_second_group : ℕ := 6

/-- Theorem stating that the number of persons in the first group is correct -/
theorem correct_num_persons_first_group :
  num_persons_first_group * days_first_group * hours_per_day_first_group =
  num_persons_second_group * days_second_group * hours_per_day_second_group :=
by sorry

end correct_num_persons_first_group_l913_91335


namespace min_distance_sum_l913_91359

/-- Given points A and B, and a point P on a circle, prove the minimum value of |PA|^2 + |PB|^2 -/
theorem min_distance_sum (A B P : ℝ × ℝ) : 
  A = (-2, 0) →
  B = (2, 0) →
  (P.1 - 3)^2 + (P.2 - 4)^2 = 4 →
  (P.1 + 2)^2 + P.2^2 + (P.1 - 2)^2 + P.2^2 ≥ 26 := by
  sorry

#check min_distance_sum

end min_distance_sum_l913_91359


namespace expression_evaluation_l913_91309

theorem expression_evaluation :
  let x : ℚ := -2
  let y : ℚ := 1/2
  (x + 2*y)^2 - (x + y)*(x - y) = -11/4 := by
sorry

end expression_evaluation_l913_91309


namespace multiple_p_values_exist_l913_91319

theorem multiple_p_values_exist : ∃ p₁ p₂ : ℝ, 
  0 < p₁ ∧ p₁ < 1 ∧ 
  0 < p₂ ∧ p₂ < 1 ∧ 
  p₁ ≠ p₂ ∧
  (Nat.choose 5 3 : ℝ) * p₁^3 * (1 - p₁)^2 = 144/625 ∧
  (Nat.choose 5 3 : ℝ) * p₂^3 * (1 - p₂)^2 = 144/625 :=
by sorry

end multiple_p_values_exist_l913_91319


namespace largest_three_digit_number_l913_91351

def digits : Finset Nat := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

def valid_equation (a b c d e f : Nat) : Prop :=
  a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧ e ∈ digits ∧ f ∈ digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
  d ≠ e ∧ d ≠ f ∧
  e ≠ f ∧
  a + 10 * b + c = 100 * d + 10 * e + f

theorem largest_three_digit_number :
  ∀ a b c d e f : Nat,
    valid_equation a b c d e f →
    100 * d + 10 * e + f ≤ 105 :=
by sorry

end largest_three_digit_number_l913_91351


namespace a_plus_b_values_l913_91384

theorem a_plus_b_values (a b : ℝ) : 
  (abs a = 1) → (b = -2) → ((a + b = -1) ∨ (a + b = -3)) :=
by sorry

end a_plus_b_values_l913_91384


namespace expressions_equality_l913_91300

theorem expressions_equality (a b c : ℝ) : a + b * c = (a + b) * (a + c) ↔ a + b + c = 1 := by
  sorry

end expressions_equality_l913_91300


namespace quadratic_equation_roots_properties_l913_91367

theorem quadratic_equation_roots_properties : ∃ (r₁ r₂ : ℝ),
  (r₁^2 - 6*r₁ + 8 = 0) ∧
  (r₂^2 - 6*r₂ + 8 = 0) ∧
  (r₁ ≠ r₂) ∧
  (|r₁ - r₂| = 2) ∧
  (|r₁| + |r₂| = 6) := by
sorry

end quadratic_equation_roots_properties_l913_91367


namespace tan_negative_405_degrees_l913_91311

theorem tan_negative_405_degrees : Real.tan ((-405 : ℝ) * π / 180) = -1 := by
  sorry

end tan_negative_405_degrees_l913_91311


namespace abs_4y_minus_6_not_positive_l913_91374

theorem abs_4y_minus_6_not_positive (y : ℚ) : ¬(|4*y - 6| > 0) ↔ y = 3/2 := by
  sorry

end abs_4y_minus_6_not_positive_l913_91374


namespace ellipse_equation_l913_91383

/-- Given an ellipse with standard form equation, prove its specific equation -/
theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  (∃ (x y : ℝ), x^2/a^2 + y^2/b^2 = 1) → -- standard form of ellipse
  (3^2 = a^2 - b^2) →                    -- condition for right focus at (3,0)
  (9/b^2 = 1) →                          -- condition for point (0,-3) on ellipse
  (∀ (x y : ℝ), x^2/18 + y^2/9 = 1 ↔ x^2/a^2 + y^2/b^2 = 1) :=
by sorry

end ellipse_equation_l913_91383


namespace min_sum_absolute_differences_l913_91371

theorem min_sum_absolute_differences (a : ℚ) : 
  ∃ (min : ℚ), min = 4 ∧ ∀ (x : ℚ), |x-1| + |x-2| + |x-3| + |x-4| ≥ min := by
  sorry

end min_sum_absolute_differences_l913_91371


namespace min_value_theorem_l913_91378

theorem min_value_theorem (a : ℝ) (h : a > 0) : 
  (∃ (x : ℝ), x = 3 / (2 * a) + 4 * a ∧ ∀ (y : ℝ), y = 3 / (2 * a) + 4 * a → x ≤ y) ∧ 
  (∃ (z : ℝ), z = 3 / (2 * a) + 4 * a ∧ z = 2 * Real.sqrt 6) :=
by sorry

end min_value_theorem_l913_91378


namespace janet_complaint_time_l913_91343

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The time Janet spends looking for her keys daily (in minutes) -/
def daily_key_search_time : ℕ := 8

/-- The total time Janet saves weekly by not losing her keys (in minutes) -/
def weekly_time_saved : ℕ := 77

/-- The time Janet spends complaining after finding her keys daily (in minutes) -/
def daily_complaint_time : ℕ := (weekly_time_saved - days_in_week * daily_key_search_time) / days_in_week

theorem janet_complaint_time :
  daily_complaint_time = 3 :=
sorry

end janet_complaint_time_l913_91343


namespace max_m_value_l913_91347

-- Define the function f(x) = x³ - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- State the theorem
theorem max_m_value :
  (∃ (m : ℝ), ∃ (x : ℝ), x ∈ Set.Icc 0 1 ∧ f x = m) →
  (∀ (m : ℝ), (∃ (x : ℝ), x ∈ Set.Icc 0 1 ∧ f x = m) → m ≤ 0) ∧
  (∃ (x : ℝ), x ∈ Set.Icc 0 1 ∧ f x = 0) :=
by sorry


end max_m_value_l913_91347


namespace borrowed_sheets_theorem_l913_91337

/-- Represents a collection of sheets with page numbers -/
structure Sheets :=
  (total_sheets : ℕ)
  (total_pages : ℕ)
  (borrowed_sheets : ℕ)

/-- Calculates the average page number of remaining sheets -/
def average_remaining_pages (s : Sheets) : ℚ :=
  let remaining_pages := s.total_pages - 2 * s.borrowed_sheets
  let sum_remaining := (s.total_pages * (s.total_pages + 1) / 2) -
                       (2 * s.borrowed_sheets * (2 * s.borrowed_sheets + 1) / 2)
  sum_remaining / remaining_pages

/-- The main theorem to prove -/
theorem borrowed_sheets_theorem (s : Sheets) 
  (h1 : s.total_sheets = 30)
  (h2 : s.total_pages = 60)
  (h3 : s.borrowed_sheets = 10) :
  average_remaining_pages s = 25 := by
  sorry

#eval average_remaining_pages ⟨30, 60, 10⟩

end borrowed_sheets_theorem_l913_91337


namespace polynomial_satisfies_conditions_l913_91394

theorem polynomial_satisfies_conditions :
  ∃ (p : ℝ → ℝ), 
    (∀ x, p x = x^2 + 1) ∧ 
    (p 3 = 10) ∧ 
    (∀ x y, p x * p y = p x + p y + p (x * y) - 3) := by
  sorry

end polynomial_satisfies_conditions_l913_91394


namespace complex_ratio_condition_l913_91336

theorem complex_ratio_condition (z : ℂ) :
  let x := z.re
  let y := z.im
  (((x + 5)^2 - y^2) / (2 * (x + 5) * y) = -3/4) ↔
  ((x + 2*y + 5) * (x - y/2 + 5) = 0 ∧ (x + 5) * y ≠ 0) :=
by sorry

end complex_ratio_condition_l913_91336


namespace original_number_proof_l913_91377

theorem original_number_proof (h1 : 213 * 16 = 3408) 
  (h2 : 1.6 * x = 34.080000000000005) : x = 21.3 := by
  sorry

end original_number_proof_l913_91377


namespace cats_left_after_sale_l913_91379

theorem cats_left_after_sale (siamese : ℕ) (house : ℕ) (sold : ℕ) : siamese = 38 → house = 25 → sold = 45 → siamese + house - sold = 18 := by
  sorry

end cats_left_after_sale_l913_91379


namespace line_parallel_to_plane_condition_l913_91364

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between planes
variable (plane_parallel : Plane → Plane → Prop)

-- Define the parallel relation between a line and a plane
variable (line_parallel_to_plane : Line → Plane → Prop)

-- Define the containment relation of a line in a plane
variable (line_in_plane : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_to_plane_condition 
  (a : Line) (α : Plane) :
  (∃ β : Plane, line_in_plane a β ∧ plane_parallel α β) →
  line_parallel_to_plane a α :=
by sorry

end line_parallel_to_plane_condition_l913_91364


namespace frustum_volume_l913_91373

/-- The volume of a frustum formed by cutting a square pyramid --/
theorem frustum_volume (base_edge : ℝ) (altitude : ℝ) (small_base_edge : ℝ) (small_altitude : ℝ)
  (h1 : base_edge = 10)
  (h2 : altitude = 10)
  (h3 : small_base_edge = 5)
  (h4 : small_altitude = 5) :
  (base_edge ^ 2 * altitude / 3) - (small_base_edge ^ 2 * small_altitude / 3) = 875 / 3 := by
  sorry

end frustum_volume_l913_91373


namespace point_on_negative_x_axis_l913_91306

/-- Given a point A with coordinates (a+1, a^2-4) that lies on the negative half of the x-axis,
    prove that its coordinates are (-1, 0). -/
theorem point_on_negative_x_axis (a : ℝ) : 
  (a + 1 < 0) ∧ (a^2 - 4 = 0) → (a + 1 = -1 ∧ a^2 - 4 = 0) :=
by sorry

end point_on_negative_x_axis_l913_91306
