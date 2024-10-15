import Mathlib

namespace NUMINAMATH_CALUDE_girls_with_tablets_l4021_402172

/-- Proves that the number of girls who brought tablets is 13 -/
theorem girls_with_tablets (total_boys : ℕ) (students_with_tablets : ℕ) (boys_with_tablets : ℕ)
  (h1 : total_boys = 20)
  (h2 : students_with_tablets = 24)
  (h3 : boys_with_tablets = 11) :
  students_with_tablets - boys_with_tablets = 13 := by
  sorry

end NUMINAMATH_CALUDE_girls_with_tablets_l4021_402172


namespace NUMINAMATH_CALUDE_sin_product_theorem_l4021_402123

theorem sin_product_theorem (x : ℝ) (h : Real.sin (5 * Real.pi / 2 - x) = 3 / 5) :
  Real.sin (x / 2) * Real.sin (5 * x / 2) = 86 / 125 := by
  sorry

end NUMINAMATH_CALUDE_sin_product_theorem_l4021_402123


namespace NUMINAMATH_CALUDE_min_value_sum_squared_ratios_l4021_402114

theorem min_value_sum_squared_ratios (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 1) :
  (x^2 / y) + (y^2 / z) + (z^2 / x) ≥ 3 ∧
  ((x^2 / y) + (y^2 / z) + (z^2 / x) = 3 ↔ x = 1 ∧ y = 1 ∧ z = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_squared_ratios_l4021_402114


namespace NUMINAMATH_CALUDE_fraction_equality_l4021_402136

theorem fraction_equality (a b c d : ℝ) 
  (h1 : a / b = 20)
  (h2 : c / b = 5)
  (h3 : c / d = 1 / 8) :
  a / d = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l4021_402136


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l4021_402158

theorem least_addition_for_divisibility (n m : ℕ) (h : n = 1057 ∧ m = 23) : 
  ∃ x : ℕ, x = 1 ∧ 
  (∀ y : ℕ, (n + y) % m = 0 → y ≥ x) ∧
  (n + x) % m = 0 :=
sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l4021_402158


namespace NUMINAMATH_CALUDE_logarithm_properties_l4021_402105

theorem logarithm_properties :
  (Real.log 2 / Real.log 10 + Real.log 5 / Real.log 10 = 1) ∧
  (Real.log 2 / Real.log 4 + 2^(Real.log 3 / Real.log 2 - 1) = 2) := by
  sorry

end NUMINAMATH_CALUDE_logarithm_properties_l4021_402105


namespace NUMINAMATH_CALUDE_dvd_total_count_l4021_402161

theorem dvd_total_count (store_dvds : ℕ) (online_dvds : ℕ) : 
  store_dvds = 8 → online_dvds = 2 → store_dvds + online_dvds = 10 := by
  sorry

end NUMINAMATH_CALUDE_dvd_total_count_l4021_402161


namespace NUMINAMATH_CALUDE_quiz_score_problem_l4021_402135

theorem quiz_score_problem (initial_students : ℕ) (dropped_students : ℕ) 
  (initial_average : ℚ) (new_average : ℚ) : 
  initial_students = 16 ∧ 
  dropped_students = 3 ∧ 
  initial_average = 62.5 ∧ 
  new_average = 62 →
  (initial_students * initial_average - 
   (initial_students - dropped_students) * new_average : ℚ) = 194 := by
  sorry

end NUMINAMATH_CALUDE_quiz_score_problem_l4021_402135


namespace NUMINAMATH_CALUDE_batch_size_is_84_l4021_402131

/-- The number of assignments in Mr. Wang's batch -/
def total_assignments : ℕ := 84

/-- The original grading rate (assignments per hour) -/
def original_rate : ℕ := 6

/-- The new grading rate (assignments per hour) -/
def new_rate : ℕ := 8

/-- The number of hours spent grading at the original rate -/
def hours_at_original_rate : ℕ := 2

/-- The number of hours saved compared to the initial plan -/
def hours_saved : ℕ := 3

/-- Theorem stating that the total number of assignments is 84 -/
theorem batch_size_is_84 :
  total_assignments = 84 ∧
  original_rate = 6 ∧
  new_rate = 8 ∧
  hours_at_original_rate = 2 ∧
  hours_saved = 3 ∧
  (total_assignments - original_rate * hours_at_original_rate) / new_rate + hours_at_original_rate + hours_saved = total_assignments / original_rate :=
by sorry

end NUMINAMATH_CALUDE_batch_size_is_84_l4021_402131


namespace NUMINAMATH_CALUDE_max_value_expression_l4021_402177

theorem max_value_expression (a b c d : ℝ) 
  (ha : -6.5 ≤ a ∧ a ≤ 6.5)
  (hb : -6.5 ≤ b ∧ b ≤ 6.5)
  (hc : -6.5 ≤ c ∧ c ≤ 6.5)
  (hd : -6.5 ≤ d ∧ d ≤ 6.5) :
  (a + 2*b + c + 2*d - a*b - b*c - c*d - d*a) ≤ 182 :=
sorry

end NUMINAMATH_CALUDE_max_value_expression_l4021_402177


namespace NUMINAMATH_CALUDE_total_earnings_l4021_402116

def price_per_kg : ℝ := 0.50
def rooster1_weight : ℝ := 30
def rooster2_weight : ℝ := 40

theorem total_earnings :
  price_per_kg * rooster1_weight + price_per_kg * rooster2_weight = 35 := by
sorry

end NUMINAMATH_CALUDE_total_earnings_l4021_402116


namespace NUMINAMATH_CALUDE_number_subtraction_problem_l4021_402122

theorem number_subtraction_problem :
  ∀ x : ℤ, x - 2 = 6 → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_number_subtraction_problem_l4021_402122


namespace NUMINAMATH_CALUDE_square_sum_equals_sixteen_l4021_402145

theorem square_sum_equals_sixteen (a b : ℝ) (h : a + b = 4) : a^2 + 2*a*b + b^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equals_sixteen_l4021_402145


namespace NUMINAMATH_CALUDE_computer_price_increase_l4021_402101

theorem computer_price_increase (b : ℝ) : 
  2 * b = 540 → (351 - b) / b * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_computer_price_increase_l4021_402101


namespace NUMINAMATH_CALUDE_rational_terms_not_adjacent_probability_l4021_402181

theorem rational_terms_not_adjacent_probability :
  let total_terms : ℕ := 9
  let rational_terms : ℕ := 3
  let irrational_terms : ℕ := 6
  let total_arrangements := Nat.factorial total_terms
  let favorable_arrangements := Nat.factorial irrational_terms * (Nat.factorial (irrational_terms + 1)).choose rational_terms
  (favorable_arrangements : ℚ) / total_arrangements = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_rational_terms_not_adjacent_probability_l4021_402181


namespace NUMINAMATH_CALUDE_mean_proportional_81_100_l4021_402148

theorem mean_proportional_81_100 : ∃ x : ℝ, x^2 = 81 * 100 ∧ x = 90 := by
  sorry

end NUMINAMATH_CALUDE_mean_proportional_81_100_l4021_402148


namespace NUMINAMATH_CALUDE_min_cost_plan_l4021_402102

/-- Represents the production plan for student desks and chairs -/
structure ProductionPlan where
  typeA : ℕ  -- Number of type A sets
  typeB : ℕ  -- Number of type B sets

/-- Calculates the total cost of a production plan -/
def totalCost (plan : ProductionPlan) : ℕ :=
  102 * plan.typeA + 124 * plan.typeB

/-- Checks if a production plan is valid according to the given constraints -/
def isValidPlan (plan : ProductionPlan) : Prop :=
  plan.typeA + plan.typeB = 500 ∧
  2 * plan.typeA + 3 * plan.typeB ≥ 1250 ∧
  5 * plan.typeA + 7 * plan.typeB ≤ 3020

/-- Theorem stating that the minimum total cost is achieved with 250 sets of each type -/
theorem min_cost_plan :
  ∀ (plan : ProductionPlan),
    isValidPlan plan →
    totalCost plan ≥ 56500 ∧
    (totalCost plan = 56500 ↔ plan.typeA = 250 ∧ plan.typeB = 250) := by
  sorry


end NUMINAMATH_CALUDE_min_cost_plan_l4021_402102


namespace NUMINAMATH_CALUDE_theme_park_youngest_child_age_l4021_402120

theorem theme_park_youngest_child_age (father_charge : ℝ) (age_cost : ℝ) (total_cost : ℝ) :
  father_charge = 6.5 →
  age_cost = 0.55 →
  total_cost = 15.95 →
  ∃ (twin_age : ℕ) (youngest_age : ℕ),
    youngest_age < twin_age ∧
    youngest_age + 4 * twin_age = 17 ∧
    (youngest_age = 1 ∨ youngest_age = 5) :=
by sorry

end NUMINAMATH_CALUDE_theme_park_youngest_child_age_l4021_402120


namespace NUMINAMATH_CALUDE_dormitory_to_city_distance_l4021_402128

theorem dormitory_to_city_distance : 
  ∀ (total_distance : ℝ),
    (1/5 : ℝ) * total_distance + (2/3 : ℝ) * total_distance + 4 = total_distance →
    total_distance = 30 := by
  sorry

end NUMINAMATH_CALUDE_dormitory_to_city_distance_l4021_402128


namespace NUMINAMATH_CALUDE_derivative_of_constant_cosine_l4021_402179

theorem derivative_of_constant_cosine (x : ℝ) : 
  deriv (λ _ : ℝ => Real.cos (π / 3)) x = 0 :=
by sorry

end NUMINAMATH_CALUDE_derivative_of_constant_cosine_l4021_402179


namespace NUMINAMATH_CALUDE_expression_values_l4021_402153

theorem expression_values (p q r s : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) :
  let expr := p / |p| + q / |q| + r / |r| + s / |s| + (p * q * r) / |p * q * r| + (p * r * s) / |p * r * s|
  expr = 6 ∨ expr = 2 ∨ expr = 0 ∨ expr = -2 ∨ expr = -6 := by
  sorry

end NUMINAMATH_CALUDE_expression_values_l4021_402153


namespace NUMINAMATH_CALUDE_no_valid_decagon_labeling_l4021_402187

/-- Represents a labeling of a regular decagon with center -/
def DecagonLabeling := Fin 11 → Fin 10

/-- The sum of digits on a line through the center of the decagon -/
def line_sum (l : DecagonLabeling) (i j : Fin 11) : ℕ :=
  l i + l j + l 10

/-- Checks if a labeling is valid according to the problem constraints -/
def is_valid_labeling (l : DecagonLabeling) : Prop :=
  (∀ i j : Fin 11, i ≠ j → l i ≠ l j) ∧
  (line_sum l 0 4 = line_sum l 1 5) ∧
  (line_sum l 0 4 = line_sum l 2 6) ∧
  (line_sum l 0 4 = line_sum l 3 7) ∧
  (line_sum l 0 4 = line_sum l 4 8)

theorem no_valid_decagon_labeling :
  ¬∃ l : DecagonLabeling, is_valid_labeling l :=
sorry

end NUMINAMATH_CALUDE_no_valid_decagon_labeling_l4021_402187


namespace NUMINAMATH_CALUDE_perimeter_of_quadrilateral_l4021_402112

-- Define the quadrilateral EFGH
structure Quadrilateral :=
  (E F G H : ℝ × ℝ)

-- Define the properties of the quadrilateral
def is_right_angled (q : Quadrilateral) : Prop :=
  (q.E.1 = q.F.1 ∧ q.F.2 = q.G.2) ∧ (q.G.1 = q.H.1 ∧ q.F.2 = q.G.2)

def side_lengths (q : Quadrilateral) : ℝ × ℝ × ℝ :=
  (15, 14, 7)

-- Theorem statement
theorem perimeter_of_quadrilateral (q : Quadrilateral) 
  (h1 : is_right_angled q) 
  (h2 : side_lengths q = (15, 14, 7)) : 
  ∃ (p : ℝ), p = 36 + 2 * Real.sqrt 65 ∧ 
  p = q.E.1 - q.F.1 + q.F.2 - q.G.2 + q.G.1 - q.H.1 + Real.sqrt ((q.E.1 - q.H.1)^2 + (q.E.2 - q.H.2)^2) :=
sorry

end NUMINAMATH_CALUDE_perimeter_of_quadrilateral_l4021_402112


namespace NUMINAMATH_CALUDE_will_initial_money_l4021_402132

-- Define the initial amount of money Will had
def initial_money : ℕ := sorry

-- Define the cost of the game
def game_cost : ℕ := 47

-- Define the number of toys bought
def num_toys : ℕ := 9

-- Define the cost of each toy
def toy_cost : ℕ := 4

-- Theorem to prove
theorem will_initial_money :
  initial_money = game_cost + num_toys * toy_cost :=
by sorry

end NUMINAMATH_CALUDE_will_initial_money_l4021_402132


namespace NUMINAMATH_CALUDE_square_sum_equality_l4021_402121

theorem square_sum_equality : (-2)^2 + 2^2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equality_l4021_402121


namespace NUMINAMATH_CALUDE_first_number_is_five_l4021_402183

/-- A sequence where each term is obtained by adding 9 to the previous term -/
def arithmeticSequence (a₁ : ℕ) : ℕ → ℕ
  | 0 => a₁
  | n + 1 => arithmeticSequence a₁ n + 9

/-- The property that 2012 is in the sequence -/
def contains2012 (a₁ : ℕ) : Prop :=
  ∃ n : ℕ, arithmeticSequence a₁ n = 2012

theorem first_number_is_five :
  ∃ a₁ : ℕ, a₁ < 10 ∧ contains2012 a₁ ∧ a₁ = 5 :=
by sorry

end NUMINAMATH_CALUDE_first_number_is_five_l4021_402183


namespace NUMINAMATH_CALUDE_trees_chopped_per_day_l4021_402170

/-- Represents the number of blocks of wood Ragnar gets per tree -/
def blocks_per_tree : ℕ := 3

/-- Represents the total number of blocks of wood Ragnar gets in 5 days -/
def total_blocks : ℕ := 30

/-- Represents the number of days Ragnar works -/
def days : ℕ := 5

/-- Theorem stating the number of trees Ragnar chops each day -/
theorem trees_chopped_per_day : 
  (total_blocks / days) / blocks_per_tree = 2 := by sorry

end NUMINAMATH_CALUDE_trees_chopped_per_day_l4021_402170


namespace NUMINAMATH_CALUDE_red_balls_count_l4021_402168

/-- Given a bag with 15 balls of red, yellow, and blue colors, 
    if the probability of drawing two non-red balls at the same time is 2/7, 
    then the number of red balls in the bag is 5. -/
theorem red_balls_count (total : ℕ) (red : ℕ) 
  (h_total : total = 15)
  (h_prob : (total - red : ℚ) / total * ((total - 1 - red) : ℚ) / (total - 1) = 2 / 7) :
  red = 5 := by
  sorry

end NUMINAMATH_CALUDE_red_balls_count_l4021_402168


namespace NUMINAMATH_CALUDE_book_arrangement_count_l4021_402142

/-- Number of ways to arrange books with specific conditions -/
def arrange_books (math_books : ℕ) (history_books : ℕ) : ℕ :=
  let total_books := math_books + history_books
  let middle_slots := total_books - 2
  let unrestricted_arrangements := Nat.factorial middle_slots
  let adjacent_arrangements := Nat.factorial (middle_slots - 1) * 2
  (math_books * (math_books - 1)) * (unrestricted_arrangements - adjacent_arrangements)

/-- Theorem stating the number of ways to arrange books under given conditions -/
theorem book_arrangement_count :
  arrange_books 4 6 = 362880 :=
by sorry

end NUMINAMATH_CALUDE_book_arrangement_count_l4021_402142


namespace NUMINAMATH_CALUDE_expression_equality_l4021_402197

theorem expression_equality : 
  Real.sqrt 8 - (2017 - Real.pi)^0 - 4^(-1 : ℤ) + (-1/2)^2 = 2 * Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l4021_402197


namespace NUMINAMATH_CALUDE_f_13_equals_223_l4021_402109

/-- Define the function f for natural numbers -/
def f (n : ℕ) : ℕ := n^2 + n + 41

/-- Theorem stating that f(13) equals 223 -/
theorem f_13_equals_223 : f 13 = 223 := by
  sorry

end NUMINAMATH_CALUDE_f_13_equals_223_l4021_402109


namespace NUMINAMATH_CALUDE_johns_calculation_l4021_402103

theorem johns_calculation (x : ℝ) : 
  Real.sqrt x - 20 = 15 → x^2 + 20 = 1500645 := by
  sorry

end NUMINAMATH_CALUDE_johns_calculation_l4021_402103


namespace NUMINAMATH_CALUDE_circle_tangent_to_line_intersection_chord_length_l4021_402106

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 3)^2 + (y - 1)^2 = 2

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := y = x

-- Define the intersecting line
def intersecting_line (x y : ℝ) (a : ℝ) : Prop := x - y + a = 0

-- Statement 1: Circle C is tangent to y = x
theorem circle_tangent_to_line : ∃ (x y : ℝ), circle_C x y ∧ tangent_line x y :=
sorry

-- Statement 2: Finding the value of a
theorem intersection_chord_length (a : ℝ) :
  (a ≠ 0) →
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
    intersecting_line x₁ y₁ a ∧ intersecting_line x₂ y₂ a ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 4) →
  (a = Real.sqrt 2 - 2 ∨ a = -Real.sqrt 2 - 2) :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_to_line_intersection_chord_length_l4021_402106


namespace NUMINAMATH_CALUDE_intersection_P_complement_M_l4021_402143

-- Define the universal set U as the set of integers
def U : Set Int := Set.univ

-- Define set M
def M : Set Int := {1, 2}

-- Define set P
def P : Set Int := {-2, -1, 0, 1, 2}

-- Theorem statement
theorem intersection_P_complement_M : 
  P ∩ (U \ M) = {-2, -1, 0} := by sorry

end NUMINAMATH_CALUDE_intersection_P_complement_M_l4021_402143


namespace NUMINAMATH_CALUDE_f_not_monotonic_l4021_402134

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 - 2 * m * x + 3

-- State that f is an even function
axiom f_even (m : ℝ) : ∀ x, f m x = f m (-x)

-- Define the derivative of f
def f_deriv (m : ℝ) (x : ℝ) : ℝ := 2 * (m - 1) * x - 2 * m

-- Theorem: f is not monotonic on (-∞, 3)
theorem f_not_monotonic (m : ℝ) : 
  ¬(∀ x y, x < y → x < 3 → y < 3 → f m x < f m y) ∧ 
  ¬(∀ x y, x < y → x < 3 → y < 3 → f m x > f m y) :=
sorry

end NUMINAMATH_CALUDE_f_not_monotonic_l4021_402134


namespace NUMINAMATH_CALUDE_inequality_proof_l4021_402146

theorem inequality_proof (x y z t : ℝ) 
  (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : t ≥ 0) 
  (h5 : x + y + z + t = 5) : 
  Real.sqrt (x^2 + y^2) + Real.sqrt (x^2 + 1) + Real.sqrt (z^2 + y^2) + 
  Real.sqrt (z^2 + t^2) + Real.sqrt (t^2 + 9) ≥ 10 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4021_402146


namespace NUMINAMATH_CALUDE_two_from_four_is_six_l4021_402176

/-- The number of ways to choose 2 items from a set of 4 distinct items -/
def choose_two_from_four : ℕ := Nat.choose 4 2

/-- Theorem stating that choosing 2 items from 4 distinct items results in 6 possibilities -/
theorem two_from_four_is_six : choose_two_from_four = 6 := by
  sorry

end NUMINAMATH_CALUDE_two_from_four_is_six_l4021_402176


namespace NUMINAMATH_CALUDE_cos_2alpha_plus_4pi_3_l4021_402193

theorem cos_2alpha_plus_4pi_3 (α : ℝ) (h : Real.sqrt 3 * Real.sin α * Real.cos α = 1 / 2) :
  Real.cos (2 * α + 4 * π / 3) = -7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_cos_2alpha_plus_4pi_3_l4021_402193


namespace NUMINAMATH_CALUDE_right_triangle_cone_properties_l4021_402171

/-- Given a right triangle with legs a and b, if rotating about leg a produces a cone
    with volume 500π cm³ and rotating about leg b produces a cone with volume 1800π cm³,
    then the hypotenuse length is √(a² + b²) and the surface area of the smaller cone
    is πb√(a² + b²). -/
theorem right_triangle_cone_properties (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (1/3 * π * b^2 * a = 500 * π) →
  (1/3 * π * a^2 * b = 1800 * π) →
  ∃ (hypotenuse surface_area : ℝ),
    hypotenuse = Real.sqrt (a^2 + b^2) ∧
    surface_area = π * min a b * Real.sqrt (a^2 + b^2) := by
  sorry

#check right_triangle_cone_properties

end NUMINAMATH_CALUDE_right_triangle_cone_properties_l4021_402171


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l4021_402184

theorem arithmetic_calculation : 3521 + 480 / 60 * 3 - 521 = 3024 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l4021_402184


namespace NUMINAMATH_CALUDE_chord_length_for_max_distance_l4021_402115

-- Define the circle
def Circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}

-- Define the chord AB
def Chord (A B : ℝ × ℝ) := A ∈ Circle ∧ B ∈ Circle

-- Define the semicircle ACB
def Semicircle (A B C : ℝ × ℝ) := 
  Chord A B ∧ 
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = (B.1 - A.1)^2 + (B.2 - A.2)^2 ∧
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = (B.1 - A.1)^2 + (B.2 - A.2)^2

-- Define the point C as the farthest point on semicircle ACB from O
def FarthestPoint (O A B C : ℝ × ℝ) := 
  Semicircle A B C ∧
  ∀ D, Semicircle A B D → (C.1 - O.1)^2 + (C.2 - O.2)^2 ≥ (D.1 - O.1)^2 + (D.2 - O.2)^2

-- Define OC perpendicular to AB
def Perpendicular (O A B C : ℝ × ℝ) := 
  (C.1 - O.1) * (B.1 - A.1) + (C.2 - O.2) * (B.2 - A.2) = 0

-- The main theorem
theorem chord_length_for_max_distance (O A B C : ℝ × ℝ) :
  O = (0, 0) →
  Chord A B →
  FarthestPoint O A B C →
  Perpendicular O A B C →
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 2 :=
sorry

end NUMINAMATH_CALUDE_chord_length_for_max_distance_l4021_402115


namespace NUMINAMATH_CALUDE_probability_specific_order_correct_l4021_402139

/-- Represents a standard deck of 52 cards -/
def standardDeck : ℕ := 52

/-- Represents the number of cards in each suit -/
def cardsPerSuit : ℕ := 13

/-- Represents the number of cards to be drawn -/
def cardsDrawn : ℕ := 4

/-- Calculates the probability of drawing one card from each suit in a specific order -/
def probabilitySpecificOrder : ℚ :=
  (cardsPerSuit : ℚ) / standardDeck *
  (cardsPerSuit : ℚ) / (standardDeck - 1) *
  (cardsPerSuit : ℚ) / (standardDeck - 2) *
  (cardsPerSuit : ℚ) / (standardDeck - 3)

/-- Theorem: The probability of drawing one card from each suit in a specific order is 2197/499800 -/
theorem probability_specific_order_correct :
  probabilitySpecificOrder = 2197 / 499800 := by
  sorry

end NUMINAMATH_CALUDE_probability_specific_order_correct_l4021_402139


namespace NUMINAMATH_CALUDE_apple_cost_calculation_l4021_402150

/-- If 4 dozen apples cost $31.20, then 5 dozen apples at the same rate will cost $39.00 -/
theorem apple_cost_calculation (cost_four_dozen : ℝ) (h : cost_four_dozen = 31.20) :
  let cost_per_dozen : ℝ := cost_four_dozen / 4
  let cost_five_dozen : ℝ := 5 * cost_per_dozen
  cost_five_dozen = 39.00 := by sorry

end NUMINAMATH_CALUDE_apple_cost_calculation_l4021_402150


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_equal_area_l4021_402119

/-- An isosceles trapezoid -/
structure IsoscelesTrapezoid where
  /-- The length of the diagonal -/
  diagonalLength : ℝ
  /-- The angle between the diagonals -/
  diagonalAngle : ℝ
  /-- The area of the trapezoid -/
  area : ℝ

/-- 
Theorem: If two isosceles trapezoids have equal diagonal lengths and equal angles between their diagonals, 
then their areas are equal.
-/
theorem isosceles_trapezoid_equal_area 
  (t1 t2 : IsoscelesTrapezoid) 
  (h1 : t1.diagonalLength = t2.diagonalLength) 
  (h2 : t1.diagonalAngle = t2.diagonalAngle) : 
  t1.area = t2.area :=
sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_equal_area_l4021_402119


namespace NUMINAMATH_CALUDE_solution_product_l4021_402124

theorem solution_product (p q : ℝ) : 
  (p - 6) * (2 * p + 8) + p^2 - 15 * p + 56 = 0 →
  (q - 6) * (2 * q + 8) + q^2 - 15 * q + 56 = 0 →
  p ≠ q →
  (p + 3) * (q + 3) = 92 / 3 := by
sorry

end NUMINAMATH_CALUDE_solution_product_l4021_402124


namespace NUMINAMATH_CALUDE_prop_two_prop_three_prop_one_false_prop_four_false_l4021_402151

-- Proposition ②
theorem prop_two (a b : ℝ) : a > |b| → a^2 > b^2 := by sorry

-- Proposition ③
theorem prop_three (a b : ℝ) : a > b → a^3 > b^3 := by sorry

-- Proposition ① is false
theorem prop_one_false : ∃ a b c : ℝ, a > b ∧ ¬(a*c^2 > b*c^2) := by sorry

-- Proposition ④ is false
theorem prop_four_false : ∃ a b : ℝ, |a| > b ∧ ¬(a^2 > b^2) := by sorry

end NUMINAMATH_CALUDE_prop_two_prop_three_prop_one_false_prop_four_false_l4021_402151


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l4021_402110

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |2*x + 8| = 4 - 3*x :=
by
  -- The unique solution is x = -4/5
  use -4/5
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l4021_402110


namespace NUMINAMATH_CALUDE_arccos_cos_fifteen_l4021_402175

theorem arccos_cos_fifteen (x : Real) (h : x = 15) :
  Real.arccos (Real.cos x) = x % (2 * Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_arccos_cos_fifteen_l4021_402175


namespace NUMINAMATH_CALUDE_ratio_equals_five_sixths_l4021_402118

theorem ratio_equals_five_sixths
  (a b c x y z : ℝ)
  (sum_squares_abc : a^2 + b^2 + c^2 = 25)
  (sum_squares_xyz : x^2 + y^2 + z^2 = 36)
  (dot_product : a*x + b*y + c*z = 30) :
  (a + b + c) / (x + y + z) = 5/6 := by
  sorry

#check ratio_equals_five_sixths

end NUMINAMATH_CALUDE_ratio_equals_five_sixths_l4021_402118


namespace NUMINAMATH_CALUDE_not_proportional_l4021_402137

theorem not_proportional (x y : ℝ) : 
  (3 * x - y = 7 ∨ 4 * x + 3 * y = 13) → 
  ¬(∃ (k₁ k₂ : ℝ), (y = k₁ * x ∨ x * y = k₂)) :=
by sorry

end NUMINAMATH_CALUDE_not_proportional_l4021_402137


namespace NUMINAMATH_CALUDE_triangulation_count_l4021_402174

/-- A triangulation of a non-self-intersecting n-gon using m interior vertices -/
structure Triangulation where
  n : ℕ  -- number of vertices in the n-gon
  m : ℕ  -- number of interior vertices
  N : ℕ  -- number of triangles in the triangulation
  h1 : N > 0  -- there is at least one triangle
  h2 : n ≥ 3  -- n-gon has at least 3 vertices

/-- The number of triangles in a triangulation of an n-gon with m interior vertices -/
theorem triangulation_count (T : Triangulation) : T.N = T.n + 2 * T.m - 2 := by
  sorry

end NUMINAMATH_CALUDE_triangulation_count_l4021_402174


namespace NUMINAMATH_CALUDE_probability_of_pink_flower_l4021_402189

-- Define the contents of the bags
def bag_A_red : ℕ := 6
def bag_A_pink : ℕ := 3
def bag_B_red : ℕ := 2
def bag_B_pink : ℕ := 7

-- Define the total number of flowers in each bag
def total_A : ℕ := bag_A_red + bag_A_pink
def total_B : ℕ := bag_B_red + bag_B_pink

-- Define the probability of choosing a pink flower from each bag
def prob_pink_A : ℚ := bag_A_pink / total_A
def prob_pink_B : ℚ := bag_B_pink / total_B

-- Theorem statement
theorem probability_of_pink_flower :
  (prob_pink_A + prob_pink_B) / 2 = 5 / 9 := by sorry

end NUMINAMATH_CALUDE_probability_of_pink_flower_l4021_402189


namespace NUMINAMATH_CALUDE_pool_cost_is_90000_l4021_402147

/-- The cost to fill a rectangular pool with bottled water -/
def pool_fill_cost (length width depth : ℝ) (liters_per_cubic_foot : ℝ) (cost_per_liter : ℝ) : ℝ :=
  length * width * depth * liters_per_cubic_foot * cost_per_liter

/-- Theorem: The cost to fill the specified pool is $90,000 -/
theorem pool_cost_is_90000 :
  pool_fill_cost 20 6 10 25 3 = 90000 := by
  sorry

#eval pool_fill_cost 20 6 10 25 3

end NUMINAMATH_CALUDE_pool_cost_is_90000_l4021_402147


namespace NUMINAMATH_CALUDE_function_properties_l4021_402167

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x > f y

def is_symmetric_about (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ x, f (c + x) = f (c - x)

theorem function_properties (f : ℝ → ℝ) 
  (h_even : is_even f)
  (h_shift : ∀ x, f (x + 1) = -f x)
  (h_incr : is_increasing_on f (-1) 0) :
  (∀ x, f (x + 2) = f x) ∧ 
  (is_symmetric_about f 1) ∧
  (is_decreasing_on f 0 1) ∧
  (f 2014 = f 0) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l4021_402167


namespace NUMINAMATH_CALUDE_volume_S_polynomial_bc_over_ad_value_l4021_402186

/-- A right rectangular prism with edge lengths 2, 4, and 5 -/
structure RectPrism where
  length : ℝ := 2
  width : ℝ := 4
  height : ℝ := 5

/-- The set of points within distance r of the prism -/
def S (B : RectPrism) (r : ℝ) : Set (ℝ × ℝ × ℝ) := sorry

/-- The volume of S(r) -/
noncomputable def volume_S (B : RectPrism) (r : ℝ) : ℝ := sorry

/-- The coefficients of the volume polynomial -/
structure VolumeCoeffs where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

theorem volume_S_polynomial (B : RectPrism) :
  ∃ coeffs : VolumeCoeffs,
    ∀ r : ℝ, volume_S B r = coeffs.a * r^3 + coeffs.b * r^2 + coeffs.c * r + coeffs.d :=
  sorry

theorem bc_over_ad_value (B : RectPrism) (coeffs : VolumeCoeffs)
  (h : ∀ r : ℝ, volume_S B r = coeffs.a * r^3 + coeffs.b * r^2 + coeffs.c * r + coeffs.d) :
  coeffs.b * coeffs.c / (coeffs.a * coeffs.d) = 15.675 :=
  sorry

end NUMINAMATH_CALUDE_volume_S_polynomial_bc_over_ad_value_l4021_402186


namespace NUMINAMATH_CALUDE_crafts_club_beads_l4021_402149

/-- The number of beads needed for a group of people making necklaces -/
def total_beads (num_members : ℕ) (necklaces_per_member : ℕ) (beads_per_necklace : ℕ) : ℕ :=
  num_members * necklaces_per_member * beads_per_necklace

theorem crafts_club_beads : 
  total_beads 9 2 50 = 900 := by
  sorry

end NUMINAMATH_CALUDE_crafts_club_beads_l4021_402149


namespace NUMINAMATH_CALUDE_square_field_perimeter_l4021_402154

theorem square_field_perimeter (a : ℝ) :
  (∃ s : ℝ, a = s^2) →  -- area is a square number
  (∃ P : ℝ, P = 36) →  -- perimeter is 36 feet
  (6 * a = 6 * (2 * 36 + 9)) →  -- given equation
  (2 * 36 = 72) :=  -- twice the perimeter is 72 feet
by
  sorry

end NUMINAMATH_CALUDE_square_field_perimeter_l4021_402154


namespace NUMINAMATH_CALUDE_polynomial_multiplication_l4021_402190

theorem polynomial_multiplication :
  ∀ x : ℝ, (7 * x^2 + 3 * x + 1) * (5 * x^3 + 2 * x + 6) = 
    35 * x^5 + 15 * x^4 + 19 * x^3 + 48 * x^2 + 20 * x + 6 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_multiplication_l4021_402190


namespace NUMINAMATH_CALUDE_cat_mouse_positions_after_360_moves_l4021_402195

/-- Represents the number of squares in the cat's path -/
def cat_squares : ℕ := 5

/-- Represents the number of segments in the mouse's path -/
def mouse_segments : ℕ := 10

/-- Represents the number of segments the mouse moves per turn -/
def mouse_move_rate : ℕ := 2

/-- Represents the total number of moves -/
def total_moves : ℕ := 360

/-- Calculates the cat's position after a given number of moves -/
def cat_position (moves : ℕ) : ℕ :=
  moves % cat_squares + 1

/-- Calculates the mouse's effective moves after accounting for skipped segments -/
def mouse_effective_moves (moves : ℕ) : ℕ :=
  (moves / mouse_segments) * (mouse_segments - 1) + (moves % mouse_segments)

/-- Calculates the mouse's position after a given number of effective moves -/
def mouse_position (effective_moves : ℕ) : ℕ :=
  (effective_moves * mouse_move_rate) % mouse_segments + 1

theorem cat_mouse_positions_after_360_moves :
  cat_position total_moves = 1 ∧ 
  mouse_position (mouse_effective_moves total_moves) = 4 := by
  sorry

end NUMINAMATH_CALUDE_cat_mouse_positions_after_360_moves_l4021_402195


namespace NUMINAMATH_CALUDE_exponential_decreasing_range_l4021_402173

/-- Given a function f(x) = a^x where a > 0 and a ≠ 1, 
    if f(m) < f(n) for all m > n, then 0 < a < 1 -/
theorem exponential_decreasing_range (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ m n : ℝ, m > n → a^m < a^n) → 0 < a ∧ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_exponential_decreasing_range_l4021_402173


namespace NUMINAMATH_CALUDE_x_intercept_is_six_l4021_402126

-- Define the line equation
def line_equation (x y : ℚ) : Prop := 4 * x - 3 * y = 24

-- Define x-intercept
def is_x_intercept (x : ℚ) : Prop := line_equation x 0

-- Theorem statement
theorem x_intercept_is_six : is_x_intercept 6 := by
  sorry

end NUMINAMATH_CALUDE_x_intercept_is_six_l4021_402126


namespace NUMINAMATH_CALUDE_trigonometric_identity_l4021_402166

theorem trigonometric_identity : 
  1 / Real.cos (70 * π / 180) - Real.sqrt 3 / Real.sin (70 * π / 180) = 2 / Real.cos (10 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l4021_402166


namespace NUMINAMATH_CALUDE_decimal_multiplication_meaning_l4021_402155

theorem decimal_multiplication_meaning (a b : ℝ) : 
  ¬ (∀ (a b : ℝ), ∃ (n : ℕ), a * b = n * (min a b)) :=
sorry

end NUMINAMATH_CALUDE_decimal_multiplication_meaning_l4021_402155


namespace NUMINAMATH_CALUDE_sin_five_alpha_identity_l4021_402140

theorem sin_five_alpha_identity (α : ℝ) : 
  16 * (Real.sin α)^5 - 20 * (Real.sin α)^3 + 5 * Real.sin α = Real.sin (5 * α) := by
  sorry

end NUMINAMATH_CALUDE_sin_five_alpha_identity_l4021_402140


namespace NUMINAMATH_CALUDE_first_tract_width_l4021_402199

/-- Given two rectangular tracts of land with specified dimensions and combined area,
    calculates the width of the first tract. -/
theorem first_tract_width (length1 : ℝ) (length2 width2 : ℝ) (combined_area : ℝ) : 
  length1 = 300 →
  length2 = 250 →
  width2 = 630 →
  combined_area = 307500 →
  combined_area = length1 * (combined_area - length2 * width2) / length1 + length2 * width2 →
  (combined_area - length2 * width2) / length1 = 500 := by
sorry

end NUMINAMATH_CALUDE_first_tract_width_l4021_402199


namespace NUMINAMATH_CALUDE_decimal_sum_to_fraction_l4021_402152

theorem decimal_sum_to_fraction :
  (0.2 : ℚ) + 0.03 + 0.004 + 0.0005 + 0.00006 + 0.000007 = 234567 / 1000000 := by
  sorry

end NUMINAMATH_CALUDE_decimal_sum_to_fraction_l4021_402152


namespace NUMINAMATH_CALUDE_franks_daily_work_hours_l4021_402169

/-- Given that Frank worked a total of 8.0 hours over 4.0 days, with equal time worked each day,
    prove that he worked 2.0 hours per day. -/
theorem franks_daily_work_hours (total_hours : ℝ) (total_days : ℝ) (hours_per_day : ℝ)
    (h1 : total_hours = 8.0)
    (h2 : total_days = 4.0)
    (h3 : hours_per_day * total_days = total_hours) :
    hours_per_day = 2.0 := by
  sorry

end NUMINAMATH_CALUDE_franks_daily_work_hours_l4021_402169


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l4021_402117

theorem equilateral_triangle_area_perimeter_ratio :
  let s : ℝ := 6
  let area : ℝ := (s^2 * Real.sqrt 3) / 4
  let perimeter : ℝ := 3 * s
  area / perimeter = Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l4021_402117


namespace NUMINAMATH_CALUDE_total_earnings_theorem_l4021_402185

/-- Represents the ticket tiers --/
inductive TicketTier
  | Standard
  | Premium
  | VIP

/-- Represents the ticket types for the second show --/
inductive TicketType
  | Regular
  | Student
  | Senior

/-- Ticket prices for the first show --/
def firstShowPrice (tier : TicketTier) : ℕ :=
  match tier with
  | TicketTier.Standard => 25
  | TicketTier.Premium => 40
  | TicketTier.VIP => 60

/-- Ticket quantities for the first show --/
def firstShowQuantity (tier : TicketTier) : ℕ :=
  match tier with
  | TicketTier.Standard => 120
  | TicketTier.Premium => 60
  | TicketTier.VIP => 20

/-- Discount rates for the second show --/
def discountRate (type : TicketType) : ℚ :=
  match type with
  | TicketType.Regular => 0
  | TicketType.Student => 0.1
  | TicketType.Senior => 0.15

/-- Ticket quantities for the second show --/
def secondShowQuantity (tier : TicketTier) (type : TicketType) : ℕ :=
  match tier, type with
  | TicketTier.Standard, TicketType.Student => 240
  | TicketTier.Standard, TicketType.Senior => 120
  | TicketTier.Premium, TicketType.Student => 120
  | TicketTier.Premium, TicketType.Senior => 60
  | TicketTier.VIP, TicketType.Student => 40
  | TicketTier.VIP, TicketType.Senior => 20
  | _, TicketType.Regular => 0

/-- Calculate the earnings from the first show --/
def firstShowEarnings : ℕ :=
  (firstShowQuantity TicketTier.Standard * firstShowPrice TicketTier.Standard) +
  (firstShowQuantity TicketTier.Premium * firstShowPrice TicketTier.Premium) +
  (firstShowQuantity TicketTier.VIP * firstShowPrice TicketTier.VIP)

/-- Calculate the discounted price for the second show --/
def secondShowPrice (tier : TicketTier) (type : TicketType) : ℚ :=
  (firstShowPrice tier : ℚ) * (1 - discountRate type)

/-- Calculate the earnings from the second show --/
def secondShowEarnings : ℚ :=
  (secondShowQuantity TicketTier.Standard TicketType.Student * secondShowPrice TicketTier.Standard TicketType.Student) +
  (secondShowQuantity TicketTier.Standard TicketType.Senior * secondShowPrice TicketTier.Standard TicketType.Senior) +
  (secondShowQuantity TicketTier.Premium TicketType.Student * secondShowPrice TicketTier.Premium TicketType.Student) +
  (secondShowQuantity TicketTier.Premium TicketType.Senior * secondShowPrice TicketTier.Premium TicketType.Senior) +
  (secondShowQuantity TicketTier.VIP TicketType.Student * secondShowPrice TicketTier.VIP TicketType.Student) +
  (secondShowQuantity TicketTier.VIP TicketType.Senior * secondShowPrice TicketTier.VIP TicketType.Senior)

/-- The main theorem stating that the total earnings from both shows equal $24,090 --/
theorem total_earnings_theorem :
  (firstShowEarnings : ℚ) + secondShowEarnings = 24090 := by
  sorry

end NUMINAMATH_CALUDE_total_earnings_theorem_l4021_402185


namespace NUMINAMATH_CALUDE_compressor_stations_theorem_l4021_402196

/-- Represents the configuration of three compressor stations -/
structure CompressorStations where
  x : ℝ  -- Distance between first and second stations
  y : ℝ  -- Distance between second and third stations
  z : ℝ  -- Distance between first and third stations
  a : ℝ  -- Additional parameter

/-- The conditions for the compressor stations configuration -/
def validConfiguration (s : CompressorStations) : Prop :=
  s.x > 0 ∧ s.y > 0 ∧ s.z > 0 ∧  -- Positive distances
  s.x + s.y = 2 * s.z ∧          -- Condition 1
  s.x + s.z = s.y + s.a ∧        -- Condition 2
  s.x + s.z = 75                 -- Condition 3

/-- The theorem stating the properties of the compressor stations configuration -/
theorem compressor_stations_theorem (s : CompressorStations) 
  (h : validConfiguration s) : 
  0 < s.a ∧ s.a < 100 ∧ 
  (s.a = 15 → s.x = 42 ∧ s.y = 24 ∧ s.z = 33) := by
  sorry

#check compressor_stations_theorem

end NUMINAMATH_CALUDE_compressor_stations_theorem_l4021_402196


namespace NUMINAMATH_CALUDE_remaining_note_denomination_l4021_402130

theorem remaining_note_denomination 
  (total_amount : ℕ)
  (total_notes : ℕ)
  (fifty_notes : ℕ)
  (h1 : total_amount = 10350)
  (h2 : total_notes = 72)
  (h3 : fifty_notes = 57) :
  (total_amount - 50 * fifty_notes) / (total_notes - fifty_notes) = 500 := by
  sorry

end NUMINAMATH_CALUDE_remaining_note_denomination_l4021_402130


namespace NUMINAMATH_CALUDE_matts_stair_climbing_rate_l4021_402104

theorem matts_stair_climbing_rate 
  (M : ℝ)  -- Matt's rate of climbing stairs in steps per minute
  (h1 : M > 0)  -- Matt's rate is positive
  (h2 : ∃ t : ℝ, t > 0 ∧ M * t = 220 ∧ (M + 5) * t = 275)  -- Condition when Matt reaches 220 steps and Tom reaches 275 steps
  : M = 20 := by
  sorry

end NUMINAMATH_CALUDE_matts_stair_climbing_rate_l4021_402104


namespace NUMINAMATH_CALUDE_new_R_value_l4021_402144

/-- A function that calculates R given g and S -/
def R (g : ℝ) (S : ℝ) : ℝ := g * S - 7

/-- The theorem stating the new value of R after S increases by 50% -/
theorem new_R_value (g : ℝ) (S : ℝ) (h1 : S = 5) (h2 : R g S = 8) :
  R g (S * 1.5) = 15.5 := by
  sorry


end NUMINAMATH_CALUDE_new_R_value_l4021_402144


namespace NUMINAMATH_CALUDE_terrell_weight_lifting_l4021_402108

/-- The number of times Terrell lifts the original weights -/
def original_lifts : ℕ := 16

/-- The weight of each original weight in pounds -/
def original_weight : ℕ := 25

/-- The weight of each new weight in pounds -/
def new_weight : ℕ := 10

/-- The number of weights Terrell lifts each time -/
def num_weights : ℕ := 2

/-- The number of times Terrell must lift the new weights to achieve the same total weight -/
def new_lifts : ℕ := (num_weights * original_lifts * original_weight) / (num_weights * new_weight)

theorem terrell_weight_lifting :
  new_lifts = 40 :=
sorry

end NUMINAMATH_CALUDE_terrell_weight_lifting_l4021_402108


namespace NUMINAMATH_CALUDE_scale_drawing_conversion_l4021_402125

/-- Proves that a 6.5 cm line segment in a scale drawing where 1 cm represents 250 meters
    is equivalent to 5332.125 feet, given that 1 meter equals approximately 3.281 feet. -/
theorem scale_drawing_conversion (scale : ℝ) (length : ℝ) (meter_to_feet : ℝ) :
  scale = 250 →
  length = 6.5 →
  meter_to_feet = 3.281 →
  length * scale * meter_to_feet = 5332.125 := by
sorry

end NUMINAMATH_CALUDE_scale_drawing_conversion_l4021_402125


namespace NUMINAMATH_CALUDE_kenny_book_purchase_l4021_402188

/-- Calculates the number of books Kenny can buy after mowing lawns and purchasing video games -/
def books_kenny_can_buy (lawn_price : ℕ) (video_game_price : ℕ) (book_price : ℕ) 
                        (lawns_mowed : ℕ) (video_games_to_buy : ℕ) : ℕ :=
  let total_earnings := lawn_price * lawns_mowed
  let video_games_cost := video_game_price * video_games_to_buy
  let remaining_money := total_earnings - video_games_cost
  remaining_money / book_price

/-- Theorem stating that Kenny can buy 60 books given the problem conditions -/
theorem kenny_book_purchase :
  books_kenny_can_buy 15 45 5 35 5 = 60 := by
  sorry

#eval books_kenny_can_buy 15 45 5 35 5

end NUMINAMATH_CALUDE_kenny_book_purchase_l4021_402188


namespace NUMINAMATH_CALUDE_a_6_equals_2_l4021_402191

/-- An arithmetic sequence with common difference 2 where a₁, a₃, and a₄ form a geometric sequence -/
def ArithGeomSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a (n + 1) = a n + 2) ∧ 
  (a 3)^2 = a 1 * a 4

theorem a_6_equals_2 (a : ℕ → ℝ) (h : ArithGeomSequence a) : a 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_a_6_equals_2_l4021_402191


namespace NUMINAMATH_CALUDE_value_of_y_l4021_402160

theorem value_of_y : ∃ y : ℚ, (3 * y) / 7 = 15 ∧ y = 35 := by sorry

end NUMINAMATH_CALUDE_value_of_y_l4021_402160


namespace NUMINAMATH_CALUDE_favorite_numbers_l4021_402159

def is_favorite_number (n : ℕ) : Prop :=
  100 < n ∧ n < 150 ∧
  n % 13 = 0 ∧
  n % 3 ≠ 0 ∧
  (n / 100 + (n / 10) % 10 + n % 10) % 4 = 0

theorem favorite_numbers :
  ∀ n : ℕ, is_favorite_number n ↔ n = 130 ∨ n = 143 :=
by sorry

end NUMINAMATH_CALUDE_favorite_numbers_l4021_402159


namespace NUMINAMATH_CALUDE_function_characterization_l4021_402127

def is_positive_integer (n : ℕ) : Prop := n > 0

def satisfies_equation (f : ℕ → ℕ) : Prop :=
  ∀ n, is_positive_integer n → f (f n) + f n = 2 * n + 6

theorem function_characterization (f : ℕ → ℕ) :
  (∀ n, is_positive_integer n → is_positive_integer (f n)) →
  satisfies_equation f →
  ∀ n, is_positive_integer n → f n = n + 2 :=
by sorry

end NUMINAMATH_CALUDE_function_characterization_l4021_402127


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l4021_402107

theorem sum_of_roots_quadratic (x₁ x₂ : ℝ) : 
  (x₁^2 - 2*x₁ - 8 = 0) → (x₂^2 - 2*x₂ - 8 = 0) → x₁ + x₂ = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l4021_402107


namespace NUMINAMATH_CALUDE_intersection_segment_length_l4021_402178

-- Define the curve C
def curve_C (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the line l
def line_l (x y : ℝ) : Prop := x - Real.sqrt 3 * y + Real.sqrt 3 = 0

-- Define the intersection points
def intersection_points (A B : ℝ × ℝ) : Prop :=
  curve_C A.1 A.2 ∧ line_l A.1 A.2 ∧
  curve_C B.1 B.2 ∧ line_l B.1 B.2 ∧
  A ≠ B

-- Theorem statement
theorem intersection_segment_length :
  ∀ A B : ℝ × ℝ, intersection_points A B →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 32 / 7 := by sorry

end NUMINAMATH_CALUDE_intersection_segment_length_l4021_402178


namespace NUMINAMATH_CALUDE_remainder_sum_of_three_l4021_402180

theorem remainder_sum_of_three (a b c : ℕ) : 
  a % 14 = 5 → b % 14 = 5 → c % 14 = 5 → (a + b + c) % 14 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_of_three_l4021_402180


namespace NUMINAMATH_CALUDE_probability_two_red_chips_l4021_402100

-- Define the number of red and white chips
def total_red : Nat := 4
def total_white : Nat := 2

-- Define the number of chips in each urn
def chips_per_urn : Nat := 3

-- Define a function to calculate the number of ways to distribute chips
def distribute_chips (r w : Nat) : Nat :=
  Nat.choose total_red r * Nat.choose total_white w

-- Define the probability of drawing a red chip from an urn
def prob_red (red_in_urn total_in_urn : Nat) : Rat :=
  red_in_urn / total_in_urn

-- Theorem statement
theorem probability_two_red_chips :
  -- Calculate the total number of ways to distribute chips
  let total_distributions : Nat :=
    distribute_chips 1 2 + distribute_chips 2 1 + distribute_chips 3 0
  
  -- Calculate the probability for each case
  let prob_case1 : Rat := (distribute_chips 1 2 : Rat) / total_distributions *
    prob_red 1 chips_per_urn * prob_red 3 chips_per_urn
  let prob_case2 : Rat := (distribute_chips 2 1 : Rat) / total_distributions *
    prob_red 2 chips_per_urn * prob_red 2 chips_per_urn
  let prob_case3 : Rat := (distribute_chips 3 0 : Rat) / total_distributions *
    prob_red 3 chips_per_urn * prob_red 1 chips_per_urn

  -- The total probability is the sum of all cases
  prob_case1 + prob_case2 + prob_case3 = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_red_chips_l4021_402100


namespace NUMINAMATH_CALUDE_altitude_polynomial_exists_l4021_402156

/-- Given a triangle whose side lengths are the roots of a cubic polynomial
    with rational coefficients, there exists a polynomial of sixth degree
    with rational coefficients whose roots are the altitudes of this triangle. -/
theorem altitude_polynomial_exists (a b c d : ℚ) (r₁ r₂ r₃ : ℝ) :
  (∀ x : ℝ, a * x^3 + b * x^2 + c * x + d = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃) →
  (r₁ > 0 ∧ r₂ > 0 ∧ r₃ > 0) →
  (r₁ + r₂ > r₃ ∧ r₂ + r₃ > r₁ ∧ r₃ + r₁ > r₂) →
  ∃ (p q s t u v w : ℚ), ∀ x : ℝ,
    p * x^6 + q * x^5 + s * x^4 + t * x^3 + u * x^2 + v * x + w = 0 ↔
    x = (2 * (r₁ * r₂ * r₃ * (r₁ + r₂ + r₃) * (r₁ + r₂ - r₃) * (r₂ + r₃ - r₁) * (r₃ + r₁ - r₂))^(1/2)) / (r₁ * (r₂ + r₃ - r₁))
    ∨ x = (2 * (r₁ * r₂ * r₃ * (r₁ + r₂ + r₃) * (r₁ + r₂ - r₃) * (r₂ + r₃ - r₁) * (r₃ + r₁ - r₂))^(1/2)) / (r₂ * (r₃ + r₁ - r₂))
    ∨ x = (2 * (r₁ * r₂ * r₃ * (r₁ + r₂ + r₃) * (r₁ + r₂ - r₃) * (r₂ + r₃ - r₁) * (r₃ + r₁ - r₂))^(1/2)) / (r₃ * (r₁ + r₂ - r₃)) :=
by
  sorry

end NUMINAMATH_CALUDE_altitude_polynomial_exists_l4021_402156


namespace NUMINAMATH_CALUDE_tangent_line_and_function_inequality_l4021_402165

open Real

theorem tangent_line_and_function_inequality (a b m : ℝ) : 
  (∀ x, x = -π/4 → (tan x = a*x + b + π/2)) →
  (∀ x, x ∈ Set.Icc 1 2 → m ≤ (exp x + b*x^2 + a) ∧ (exp x + b*x^2 + a) ≤ m^2 - 2) →
  (∃ m_max : ℝ, m_max = exp 1 + 1 ∧ m ≤ m_max) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_and_function_inequality_l4021_402165


namespace NUMINAMATH_CALUDE_x_intercepts_count_l4021_402138

theorem x_intercepts_count (x : ℝ) :
  (∃! x, (x - 5) * (x^2 + 6*x + 10) = 0) :=
sorry

end NUMINAMATH_CALUDE_x_intercepts_count_l4021_402138


namespace NUMINAMATH_CALUDE_board_sum_always_odd_l4021_402192

theorem board_sum_always_odd (n : ℕ) (h : n = 1966) :
  let initial_sum := n * (n + 1) / 2
  ∀ (operations : ℕ), ∃ (final_sum : ℤ),
    final_sum ≡ initial_sum [ZMOD 2] ∧ final_sum ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_board_sum_always_odd_l4021_402192


namespace NUMINAMATH_CALUDE_unique_quadratic_solution_l4021_402198

theorem unique_quadratic_solution (a : ℝ) (ha : a ≠ 0) :
  (∃! x, a * x^2 + 30 * x + 5 = 0) → 
  (∃ x, a * x^2 + 30 * x + 5 = 0 ∧ x = -1/3) :=
by sorry

end NUMINAMATH_CALUDE_unique_quadratic_solution_l4021_402198


namespace NUMINAMATH_CALUDE_correct_front_view_l4021_402133

def StackColumn := List Nat

def frontView (stacks : List StackColumn) : List Nat :=
  stacks.map (List.foldl max 0)

theorem correct_front_view (stacks : List StackColumn) :
  stacks = [[3, 5], [2, 6, 4], [1, 1, 3, 8], [5, 2]] →
  frontView stacks = [5, 6, 8, 5] := by
  sorry

end NUMINAMATH_CALUDE_correct_front_view_l4021_402133


namespace NUMINAMATH_CALUDE_even_increasing_inequality_l4021_402141

/-- A function f is even if f(-x) = f(x) for all x -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- A function f is increasing on [0, +∞) if f(x) ≤ f(y) for all 0 ≤ x ≤ y -/
def IncreasingOnNonnegatives (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x ∧ x ≤ y → f x ≤ f y

theorem even_increasing_inequality (f : ℝ → ℝ) (a : ℝ) 
    (heven : EvenFunction f) (hincr : IncreasingOnNonnegatives f) :
    f (-1) < f (a^2 - 2*a + 3) := by
  sorry

end NUMINAMATH_CALUDE_even_increasing_inequality_l4021_402141


namespace NUMINAMATH_CALUDE_quadratic_equation_equivalence_l4021_402111

/-- Given a quadratic equation (x + 3)² = x(3x - 1), prove it's equivalent to 2x² - 7x - 9 = 0 in general form -/
theorem quadratic_equation_equivalence (x : ℝ) : (x + 3)^2 = x * (3*x - 1) ↔ 2*x^2 - 7*x - 9 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_equivalence_l4021_402111


namespace NUMINAMATH_CALUDE_vector_parallel_tangent_l4021_402129

/-- Given points A, B, and C in a 2D Cartesian coordinate system,
    prove that vector AB equals (1, √3) and tan x equals √3 when AB is parallel to OC. -/
theorem vector_parallel_tangent (x : ℝ) : 
  let A : ℝ × ℝ := (-1, 0)
  let B : ℝ × ℝ := (0, Real.sqrt 3)
  let C : ℝ × ℝ := (Real.cos x, Real.sin x)
  let AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
  let OC : ℝ × ℝ := C
  AB.2 / AB.1 = OC.2 / OC.1 →
  AB = (1, Real.sqrt 3) ∧ Real.tan x = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_vector_parallel_tangent_l4021_402129


namespace NUMINAMATH_CALUDE_system_of_equations_solution_range_l4021_402194

theorem system_of_equations_solution_range (a x y : ℝ) : 
  x + 3*y = 3 - a →
  2*x + y = 1 + 3*a →
  x + y > 3*a + 4 →
  a < -3/2 := by
sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_range_l4021_402194


namespace NUMINAMATH_CALUDE_parallel_line_slope_l4021_402157

/-- The slope of any line parallel to 3x + 6y = -21 is -1/2 -/
theorem parallel_line_slope (a b c : ℝ) (h : 3 * a + 6 * b = -21) :
  ∃ (m : ℝ), m = -1/2 ∧ ∀ (x y : ℝ), 3 * x + 6 * y = -21 → y = m * x + c :=
sorry

end NUMINAMATH_CALUDE_parallel_line_slope_l4021_402157


namespace NUMINAMATH_CALUDE_inequality_solution_range_l4021_402162

theorem inequality_solution_range (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1/x + 2/y = 1) :
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ 1/x + 2/y = 1 ∧ x + y/2 < m^2 + 3*m) ↔ 
  (m < -4 ∨ m > 1) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l4021_402162


namespace NUMINAMATH_CALUDE_police_catch_thief_time_police_catch_thief_time_equals_two_l4021_402164

/-- Proves that the time taken for a police officer to catch a thief is 2 hours,
    given specific initial conditions. -/
theorem police_catch_thief_time (thief_speed : ℝ) (police_speed : ℝ) 
  (initial_distance : ℝ) (delay_time : ℝ) : ℝ :=
  by
  -- Define the conditions
  have h1 : thief_speed = 20 := by sorry
  have h2 : police_speed = 40 := by sorry
  have h3 : initial_distance = 60 := by sorry
  have h4 : delay_time = 1 := by sorry

  -- Calculate the distance covered by the thief during the delay
  let thief_distance := thief_speed * delay_time

  -- Calculate the remaining distance between the police and thief
  let remaining_distance := initial_distance - thief_distance

  -- Calculate the relative speed between police and thief
  let relative_speed := police_speed - thief_speed

  -- Calculate the time taken to catch the thief
  let catch_time := remaining_distance / relative_speed

  -- Prove that catch_time equals 2
  sorry

/-- The time taken for the police officer to catch the thief -/
def catch_time : ℝ := 2

-- Proof that the theorem result equals the defined catch_time
theorem police_catch_thief_time_equals_two :
  police_catch_thief_time 20 40 60 1 = catch_time := by sorry

end NUMINAMATH_CALUDE_police_catch_thief_time_police_catch_thief_time_equals_two_l4021_402164


namespace NUMINAMATH_CALUDE_girls_money_and_scarf_price_l4021_402113

-- Define variables
variable (x y s m v : ℝ)

-- Define the conditions
def conditions (x y s m v : ℝ) : Prop :=
  y + 40 < s ∧ s < y + 50 ∧
  x + 30 < s ∧ s ≤ x + 40 - m ∧ m < 10 ∧
  0.8 * s ≤ x + 20 ∧ 0.8 * s ≤ y + 30 ∧
  0.8 * s - 4 = y + 20 ∧
  y < 0.6 * s - 3 ∧ 0.6 * s - 3 < y + 10 ∧
  x - 10 < 0.6 * s - 3 ∧ 0.6 * s - 3 < x ∧
  x + y - 1.2 * s = v

-- Theorem statement
theorem girls_money_and_scarf_price (x y s m v : ℝ) 
  (h : conditions x y s m v) : 
  61 ≤ x ∧ x ≤ 69 ∧ 52 ≤ y ∧ y ≤ 60 ∧ 91 ≤ s ∧ s ≤ 106 := by
  sorry

end NUMINAMATH_CALUDE_girls_money_and_scarf_price_l4021_402113


namespace NUMINAMATH_CALUDE_geometric_sequence_formula_and_sum_l4021_402163

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_formula_and_sum 
  (a : ℕ → ℝ) (b : ℕ → ℝ) (S : ℕ → ℝ) :
  geometric_sequence a →
  a 1 + a 2 = 9 →
  a 2 + a 3 = 18 →
  (∀ n, b n = a n + 2 * n) →
  (∀ n, a n = 3 * 2^(n - 1)) ∧ 
  (∀ n, S n = 3 * 2^n + n * (n + 1) - 3) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_formula_and_sum_l4021_402163


namespace NUMINAMATH_CALUDE_end_zeros_imply_n_greater_than_seven_l4021_402182

theorem end_zeros_imply_n_greater_than_seven (m n : ℕ+) 
  (h1 : m > n) 
  (h2 : (22220038 ^ m.val - 22220038 ^ n.val) % (10 ^ 8) = 0) : 
  n > 7 := by
  sorry

end NUMINAMATH_CALUDE_end_zeros_imply_n_greater_than_seven_l4021_402182
