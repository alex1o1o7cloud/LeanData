import Mathlib

namespace square_reciprocal_sum_l792_79259

theorem square_reciprocal_sum (m : ℝ) (h : m + 1/m = 10) : m^2 + 1/m^2 + 4 = 102 := by
  sorry

end square_reciprocal_sum_l792_79259


namespace total_books_read_proof_l792_79224

/-- The number of books read by the entire student body in one year -/
def total_books_read (c s : ℕ) : ℕ :=
  48 * c * s

/-- Theorem stating that the total number of books read by the entire student body in one year
    is equal to 48 * c * s, where c is the number of classes and s is the number of students per class -/
theorem total_books_read_proof (c s : ℕ) :
  total_books_read c s = 48 * c * s :=
by sorry

end total_books_read_proof_l792_79224


namespace average_age_decrease_l792_79219

theorem average_age_decrease (N : ℕ) : 
  let original_avg : ℚ := 40
  let new_students : ℕ := 12
  let new_students_avg : ℚ := 34
  let total_original_age : ℚ := N * original_avg
  let total_new_age : ℚ := new_students * new_students_avg
  let new_total_students : ℕ := N + new_students
  let new_avg : ℚ := (total_original_age + total_new_age) / new_total_students
  original_avg - new_avg = 6 := by
sorry

end average_age_decrease_l792_79219


namespace lunch_cost_distribution_l792_79242

theorem lunch_cost_distribution (total_cost : ℕ) 
  (your_cost first_friend_extra second_friend_less third_friend_multiplier : ℕ) :
  total_cost = 100 ∧ 
  first_friend_extra = 15 ∧ 
  second_friend_less = 20 ∧ 
  third_friend_multiplier = 2 →
  ∃ (your_amount : ℕ),
    your_amount = 21 ∧
    your_amount + (your_amount + first_friend_extra) + 
    (your_amount - second_friend_less) + (your_amount * third_friend_multiplier) = total_cost :=
by sorry

end lunch_cost_distribution_l792_79242


namespace unique_pie_purchase_l792_79246

/-- Represents the number of pies bought by each classmate -/
structure PiePurchase where
  kostya : Nat
  volodya : Nat
  tolya : Nat

/-- Checks if a PiePurchase satisfies all the conditions of the problem -/
def isValidPurchase (p : PiePurchase) : Prop :=
  p.kostya + p.volodya + p.tolya = 13 ∧
  p.tolya = 2 * p.kostya ∧
  p.kostya < p.volodya ∧
  p.volodya < p.tolya

/-- The theorem stating that there is only one valid solution to the problem -/
theorem unique_pie_purchase :
  ∃! p : PiePurchase, isValidPurchase p ∧ p = ⟨3, 4, 6⟩ := by
  sorry

end unique_pie_purchase_l792_79246


namespace smallest_positive_angle_2002_l792_79215

theorem smallest_positive_angle_2002 : 
  ∃ (θ : ℝ), θ > 0 ∧ θ < 360 ∧ ∀ (k : ℤ), -2002 = θ + 360 * k → θ = 158 := by
  sorry

end smallest_positive_angle_2002_l792_79215


namespace tom_rare_cards_l792_79209

/-- The number of rare cards in Tom's deck -/
def rare_cards : ℕ := 19

/-- The number of uncommon cards in Tom's deck -/
def uncommon_cards : ℕ := 11

/-- The number of common cards in Tom's deck -/
def common_cards : ℕ := 30

/-- The cost of a rare card in dollars -/
def rare_cost : ℚ := 1

/-- The cost of an uncommon card in dollars -/
def uncommon_cost : ℚ := 1/2

/-- The cost of a common card in dollars -/
def common_cost : ℚ := 1/4

/-- The total cost of Tom's deck in dollars -/
def total_cost : ℚ := 32

theorem tom_rare_cards : 
  rare_cards * rare_cost + 
  uncommon_cards * uncommon_cost + 
  common_cards * common_cost = total_cost := by sorry

end tom_rare_cards_l792_79209


namespace quadratic_equation_result_l792_79249

theorem quadratic_equation_result : ∀ y : ℝ, 6 * y^2 + 7 = 2 * y + 12 → (12 * y - 4)^2 = 128 := by
  sorry

end quadratic_equation_result_l792_79249


namespace two_guests_mixed_probability_l792_79285

/-- The number of guests -/
def num_guests : ℕ := 3

/-- The number of pastry types -/
def num_pastry_types : ℕ := 3

/-- The total number of pastries -/
def total_pastries : ℕ := num_guests * num_pastry_types

/-- The number of pastries each guest receives -/
def pastries_per_guest : ℕ := num_pastry_types

/-- The probability of exactly two guests receiving one of each type of pastry -/
def probability_two_guests_mixed : ℚ := 27 / 280

theorem two_guests_mixed_probability :
  probability_two_guests_mixed = 27 / 280 := by
  sorry

end two_guests_mixed_probability_l792_79285


namespace hyperbola_condition_roots_or_hyperbola_condition_l792_79277

-- Define the conditions
def has_two_distinct_positive_roots (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ 
  x₁^2 + 2*m*x₁ + (m+2) = 0 ∧ x₂^2 + 2*m*x₂ + (m+2) = 0

def is_hyperbola_with_foci_on_y_axis (m : ℝ) : Prop :=
  ∀ x y : ℝ, x^2/(m+3) - y^2/(2*m-1) = 1 → 
  (m+3 < 0 ∧ 2*m-1 > 0)

-- Theorem statements
theorem hyperbola_condition (m : ℝ) : 
  is_hyperbola_with_foci_on_y_axis m → m < -3 :=
sorry

theorem roots_or_hyperbola_condition (m : ℝ) :
  (has_two_distinct_positive_roots m ∨ is_hyperbola_with_foci_on_y_axis m) ∧
  ¬(has_two_distinct_positive_roots m ∧ is_hyperbola_with_foci_on_y_axis m) →
  (-2 < m ∧ m < -1) ∨ m < -3 :=
sorry

end hyperbola_condition_roots_or_hyperbola_condition_l792_79277


namespace z_eighth_power_equals_one_l792_79294

theorem z_eighth_power_equals_one :
  let z : ℂ := (-Real.sqrt 3 - I) / 2
  z^8 = 1 := by sorry

end z_eighth_power_equals_one_l792_79294


namespace max_odd_integers_in_even_product_l792_79270

theorem max_odd_integers_in_even_product (integers : Finset ℕ) :
  integers.card = 6 ∧
  (∀ n ∈ integers, n > 0) ∧
  Even (integers.prod id) →
  (integers.filter Odd).card ≤ 5 ∧
  ∃ (subset : Finset ℕ),
    subset ⊆ integers ∧
    subset.card = 5 ∧
    ∀ n ∈ subset, Odd n :=
by sorry

end max_odd_integers_in_even_product_l792_79270


namespace supplement_statement_is_proposition_l792_79274

-- Define what a proposition is
def isPropositon (s : String) : Prop := ∃ (b : Bool), (b = true ∨ b = false)

-- Define the statement
def supplementStatement : String := "The supplements of the same angle are equal"

-- Theorem to prove
theorem supplement_statement_is_proposition : isPropositon supplementStatement := by
  sorry

end supplement_statement_is_proposition_l792_79274


namespace distinct_roots_condition_zero_root_condition_other_root_when_m_is_one_other_root_when_m_is_negative_one_l792_79281

-- Define the quadratic equation
def quadratic_equation (x m : ℝ) : Prop :=
  x^2 + 2*(m-1)*x + m^2 - 1 = 0

-- Part 1: Distinct real roots condition
theorem distinct_roots_condition (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ quadratic_equation x m ∧ quadratic_equation y m) ↔ m < 1 :=
sorry

-- Part 2: Zero root condition and other root
theorem zero_root_condition (m : ℝ) :
  quadratic_equation 0 m → (m = 1 ∨ m = -1) :=
sorry

theorem other_root_when_m_is_one :
  quadratic_equation 0 1 → quadratic_equation 0 1 :=
sorry

theorem other_root_when_m_is_negative_one :
  quadratic_equation 0 (-1) → quadratic_equation 4 (-1) :=
sorry

end distinct_roots_condition_zero_root_condition_other_root_when_m_is_one_other_root_when_m_is_negative_one_l792_79281


namespace orchard_theorem_l792_79240

/-- Represents an apple orchard with Fuji and Gala trees -/
structure Orchard where
  total : ℕ
  pure_fuji : ℕ
  pure_gala : ℕ
  cross_pollinated : ℕ

/-- The conditions of the orchard as described in the problem -/
def orchard_conditions (o : Orchard) : Prop :=
  o.cross_pollinated = o.total / 10 ∧
  o.pure_fuji = (o.total * 3) / 4 ∧
  o.pure_gala = 42 ∧
  o.total = o.pure_fuji + o.pure_gala + o.cross_pollinated

/-- The theorem stating that under the given conditions, 
    the number of pure Fuji plus cross-pollinated trees is 238 -/
theorem orchard_theorem (o : Orchard) 
  (h : orchard_conditions o) : o.pure_fuji + o.cross_pollinated = 238 := by
  sorry

end orchard_theorem_l792_79240


namespace monic_quadratic_unique_l792_79230

/-- A monic quadratic polynomial is a polynomial of the form x^2 + bx + c -/
def MonicQuadraticPolynomial (b c : ℝ) : ℝ → ℝ := fun x ↦ x^2 + b*x + c

theorem monic_quadratic_unique (b c : ℝ) :
  let g := MonicQuadraticPolynomial b c
  g 0 = 8 ∧ g 1 = 14 → b = 5 ∧ c = 8 := by sorry

end monic_quadratic_unique_l792_79230


namespace star_polygon_n_is_24_l792_79244

/-- Represents an n-pointed star polygon -/
structure StarPolygon where
  n : ℕ
  angle_A : ℝ
  angle_B : ℝ
  angles_congruent : True  -- Represents that A₁, A₂, ..., Aₙ are congruent and B₁, B₂, ..., Bₙ are congruent
  angle_difference : angle_B = angle_A + 15

/-- Theorem stating that in a star polygon with the given properties, n = 24 -/
theorem star_polygon_n_is_24 (star : StarPolygon) : star.n = 24 := by
  sorry

end star_polygon_n_is_24_l792_79244


namespace range_of_expression_l792_79200

theorem range_of_expression (x y z : ℝ) (h : x^2 + y^2 + z^2 = 4) :
  -(6 : ℝ) ≤ x + 2*y - 2*z ∧ x + 2*y - 2*z ≤ 6 := by
  sorry

end range_of_expression_l792_79200


namespace seventh_term_of_arithmetic_sequence_l792_79275

def arithmetic_sequence (a : ℚ) (d : ℚ) : ℕ → ℚ
  | 0 => a
  | n + 1 => arithmetic_sequence a d n + d

theorem seventh_term_of_arithmetic_sequence 
  (a d : ℚ) 
  (h1 : (arithmetic_sequence a d 0) + 
        (arithmetic_sequence a d 1) + 
        (arithmetic_sequence a d 2) + 
        (arithmetic_sequence a d 3) + 
        (arithmetic_sequence a d 4) = 20)
  (h2 : arithmetic_sequence a d 5 = 8) :
  arithmetic_sequence a d 6 = 28 / 3 :=
by
  sorry

end seventh_term_of_arithmetic_sequence_l792_79275


namespace max_value_cos_sin_expression_l792_79258

theorem max_value_cos_sin_expression (a b c : ℝ) :
  (∀ θ : ℝ, a * Real.cos θ + b * Real.sin θ + c ≤ Real.sqrt (a^2 + b^2) + c) ∧
  (∃ θ : ℝ, a * Real.cos θ + b * Real.sin θ + c = Real.sqrt (a^2 + b^2) + c) :=
by sorry

end max_value_cos_sin_expression_l792_79258


namespace projectile_max_height_l792_79291

/-- The height function of the projectile -/
def h (t : ℝ) : ℝ := -20 * t^2 + 80 * t + 36

/-- The maximum height reached by the projectile -/
def max_height : ℝ := 116

/-- Theorem stating that the maximum height reached by the projectile is 116 meters -/
theorem projectile_max_height :
  ∃ t : ℝ, h t = max_height ∧ ∀ s : ℝ, h s ≤ max_height :=
sorry

end projectile_max_height_l792_79291


namespace infinitely_many_primes_of_form_l792_79241

theorem infinitely_many_primes_of_form (m n : ℤ) : 
  ∃ (S : Set Nat), Set.Infinite S ∧ ∀ p ∈ S, Prime p ∧ ∃ m n : ℤ, p = m^2 + m*n + n^2 :=
sorry

end infinitely_many_primes_of_form_l792_79241


namespace apples_left_in_basket_l792_79237

/-- Given a basket of apples, calculate the number of apples left after Ricki and Samson remove some. -/
theorem apples_left_in_basket 
  (initial_apples : ℕ) 
  (ricki_removes : ℕ) 
  (h1 : initial_apples = 184) 
  (h2 : ricki_removes = 34) :
  initial_apples - (ricki_removes + 3 * ricki_removes) = 48 := by
  sorry

#check apples_left_in_basket

end apples_left_in_basket_l792_79237


namespace unique_positive_solution_l792_79288

theorem unique_positive_solution (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h1 : x₁ > 0) (h2 : x₂ > 0) (h3 : x₃ > 0) (h4 : x₄ > 0) (h5 : x₅ > 0)
  (eq1 : x₁ + x₂ = x₃^2)
  (eq2 : x₂ + x₃ = x₄^2)
  (eq3 : x₃ + x₄ = x₅^2)
  (eq4 : x₄ + x₅ = x₁^2)
  (eq5 : x₅ + x₁ = x₂^2) :
  x₁ = 2 ∧ x₂ = 2 ∧ x₃ = 2 ∧ x₄ = 2 ∧ x₅ = 2 := by
  sorry

#check unique_positive_solution

end unique_positive_solution_l792_79288


namespace point_on_line_l792_79253

/-- Given a line passing through points (2, 1) and (10, 5), 
    prove that the point (14, 7) lies on this line. -/
theorem point_on_line : ∀ (t : ℝ), 
  let p1 : ℝ × ℝ := (2, 1)
  let p2 : ℝ × ℝ := (10, 5)
  let p3 : ℝ × ℝ := (t, 7)
  -- Check if p3 is on the line through p1 and p2
  (p3.2 - p1.2) * (p2.1 - p1.1) = (p3.1 - p1.1) * (p2.2 - p1.2) →
  t = 14 :=
by
  sorry

#check point_on_line

end point_on_line_l792_79253


namespace parabola_focus_property_l792_79290

/-- Parabola with equation y^2 = 16x -/
def Parabola : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2^2 = 16 * p.1}

/-- Focus of the parabola -/
def F : ℝ × ℝ := (4, 0)

/-- Point on y-axis with |OA| = |OF| -/
def A : ℝ × ℝ := (0, 4) -- We choose the positive y-coordinate

/-- Intersection of directrix and x-axis -/
def B : ℝ × ℝ := (-4, 0)

/-- Vector from F to A -/
def FA : ℝ × ℝ := (A.1 - F.1, A.2 - F.2)

/-- Vector from A to B -/
def AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

/-- Dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem parabola_focus_property :
  F ∈ Parabola ∧
  A.1 = 0 ∧
  (A.1 - 0)^2 + (A.2 - 0)^2 = (F.1 - 0)^2 + (F.2 - 0)^2 ∧
  B.2 = 0 →
  dot_product FA AB = 0 := by
  sorry

end parabola_focus_property_l792_79290


namespace exists_region_with_min_area_l792_79252

/-- Represents a line segment in a unit square --/
structure Segment where
  length : ℝ
  parallel_to_side : Bool

/-- Represents a configuration of segments in a unit square --/
structure SquareConfiguration where
  segments : List Segment
  total_length : ℝ
  total_length_eq : total_length = (segments.map Segment.length).sum
  total_length_bound : total_length = 18
  segments_within_square : ∀ s ∈ segments, s.length ≤ 1

/-- Represents a region formed by the segments --/
structure Region where
  area : ℝ

/-- The theorem to be proved --/
theorem exists_region_with_min_area (config : SquareConfiguration) :
  ∃ (r : Region), r.area ≥ 1 / 100 := by
  sorry

end exists_region_with_min_area_l792_79252


namespace ages_sum_l792_79223

theorem ages_sum (a b c : ℕ+) : 
  a = b ∧ a > c ∧ a * b * c = 72 → a + b + c = 14 := by
  sorry

end ages_sum_l792_79223


namespace common_tangents_exist_l792_79204

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a line in a 2D plane -/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- Checks if a line is tangent to a circle -/
def isTangent (l : Line) (c : Circle) : Prop := sorry

/-- Theorem: Common tangents exist for two given circles -/
theorem common_tangents_exist (c1 c2 : Circle) 
  (h : c1.radius > c2.radius) : 
  ∃ (l : Line), isTangent l c1 ∧ isTangent l c2 := by
  sorry

/-- Function to construct common tangents -/
noncomputable def construct_common_tangents (c1 c2 : Circle) 
  (h : c1.radius > c2.radius) : 
  List Line := sorry

end common_tangents_exist_l792_79204


namespace solution_set_f_geq_8_range_of_a_when_solution_exists_l792_79251

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |x + 3|

-- Theorem for the solution set of f(x) ≥ 8
theorem solution_set_f_geq_8 :
  {x : ℝ | f x ≥ 8} = {x : ℝ | x ≤ -5 ∨ x ≥ 3} := by sorry

-- Theorem for the range of a when the solution set of f(x) < a^2 - 3a is not empty
theorem range_of_a_when_solution_exists (a : ℝ) :
  (∃ x, f x < a^2 - 3*a) → (a < -1 ∨ a > 4) := by sorry

end solution_set_f_geq_8_range_of_a_when_solution_exists_l792_79251


namespace part_I_part_II_l792_79226

-- Define sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 3}
def B (a b : ℝ) : Set ℝ := {x | x^2 - a*x + b < 0}

-- Theorem for part I
theorem part_I : ∀ a b : ℝ, A = B a b → a = 2 ∧ b = -3 := by sorry

-- Theorem for part II
theorem part_II : ∀ a : ℝ, (A ∩ B a 3) ⊇ B a 3 → -2 * Real.sqrt 3 ≤ a ∧ a ≤ 4 := by sorry

end part_I_part_II_l792_79226


namespace expression_evaluation_l792_79254

theorem expression_evaluation : 3^(0^(2^11)) + ((3^0)^2)^11 = 2 := by
  sorry

end expression_evaluation_l792_79254


namespace inequality_solution_difference_l792_79229

theorem inequality_solution_difference : ∃ (M m : ℝ),
  (∀ x, 4 * x * (x - 5) ≤ 375 → x ≤ M ∧ m ≤ x) ∧
  (4 * M * (M - 5) ≤ 375) ∧
  (4 * m * (m - 5) ≤ 375) ∧
  (M - m = 20) := by
  sorry

end inequality_solution_difference_l792_79229


namespace pure_imaginary_complex_number_l792_79299

theorem pure_imaginary_complex_number (m : ℝ) : 
  (∃ z : ℂ, z = (m^2 - 4 : ℝ) + (m + 2 : ℝ) * I ∧ z.re = 0 ∧ m + 2 ≠ 0) → m = 2 := by
  sorry

end pure_imaginary_complex_number_l792_79299


namespace exactly_three_true_l792_79213

theorem exactly_three_true : 
  (∀ x > 0, x > Real.sin x) ∧ 
  ((∀ x, x - Real.sin x = 0 → x = 0) ↔ (∀ x, x ≠ 0 → x - Real.sin x ≠ 0)) ∧ 
  (∀ p q : Prop, (p ∧ q → p ∨ q) ∧ ¬(p ∨ q → p ∧ q)) ∧ 
  ¬(¬(∀ x : ℝ, x - Real.log x > 0) ↔ (∃ x : ℝ, x - Real.log x < 0)) := by
  sorry

end exactly_three_true_l792_79213


namespace even_function_decreasing_interval_l792_79201

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 + m * x + 3

-- Define the property of being an even function
def isEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Define the decreasing interval
def decreasingInterval (f : ℝ → ℝ) : Set ℝ := {x | ∀ y, x ≤ y → f y ≤ f x}

-- State the theorem
theorem even_function_decreasing_interval :
  ∀ m : ℝ, isEven (f m) → decreasingInterval (f m) = Set.Ici 0 :=
sorry

end even_function_decreasing_interval_l792_79201


namespace yoongi_has_fewest_l792_79220

/-- Represents the number of apples each person has -/
structure AppleCount where
  yoongi : Nat
  jungkook : Nat
  yuna : Nat

/-- Defines the given apple counts -/
def given_apples : AppleCount :=
  { yoongi := 4
  , jungkook := 9
  , yuna := 5 }

/-- Theorem: Yoongi has the fewest apples -/
theorem yoongi_has_fewest (a : AppleCount := given_apples) :
  a.yoongi < a.jungkook ∧ a.yoongi < a.yuna :=
by sorry

end yoongi_has_fewest_l792_79220


namespace shooting_probabilities_l792_79235

/-- Represents the probabilities of hitting different rings in a shooting training session -/
structure ShootingProbabilities where
  ring10 : ℝ
  ring9 : ℝ
  ring8 : ℝ
  ring7 : ℝ
  sum_to_one : ring10 + ring9 + ring8 + ring7 < 1
  non_negative : ring10 ≥ 0 ∧ ring9 ≥ 0 ∧ ring8 ≥ 0 ∧ ring7 ≥ 0

/-- The probability of hitting either the 10 or 9 ring -/
def prob_10_or_9 (p : ShootingProbabilities) : ℝ := p.ring10 + p.ring9

/-- The probability of scoring less than 7 rings -/
def prob_less_than_7 (p : ShootingProbabilities) : ℝ := 1 - (p.ring10 + p.ring9 + p.ring8 + p.ring7)

theorem shooting_probabilities (p : ShootingProbabilities) 
  (h1 : p.ring10 = 0.21) 
  (h2 : p.ring9 = 0.23) 
  (h3 : p.ring8 = 0.25) 
  (h4 : p.ring7 = 0.28) : 
  prob_10_or_9 p = 0.44 ∧ prob_less_than_7 p = 0.03 := by
  sorry

#eval prob_10_or_9 ⟨0.21, 0.23, 0.25, 0.28, by norm_num, by norm_num⟩
#eval prob_less_than_7 ⟨0.21, 0.23, 0.25, 0.28, by norm_num, by norm_num⟩

end shooting_probabilities_l792_79235


namespace evaluate_nested_square_roots_l792_79203

theorem evaluate_nested_square_roots : 
  Real.sqrt (64 * Real.sqrt (32 * Real.sqrt (4^3))) = 64 := by
  sorry

end evaluate_nested_square_roots_l792_79203


namespace exponential_multiplication_specific_exponential_multiplication_l792_79273

theorem exponential_multiplication (n : ℕ) : (10 : ℝ) ^ n * (10 : ℝ) ^ n = (10 : ℝ) ^ (2 * n) := by
  sorry

-- The specific case for n = 1000
theorem specific_exponential_multiplication : (10 : ℝ) ^ 1000 * (10 : ℝ) ^ 1000 = (10 : ℝ) ^ 2000 := by
  sorry

end exponential_multiplication_specific_exponential_multiplication_l792_79273


namespace museum_entrance_cost_l792_79221

theorem museum_entrance_cost (num_students : ℕ) (num_teachers : ℕ) (ticket_price : ℕ) : 
  num_students = 20 → num_teachers = 3 → ticket_price = 5 → 
  (num_students + num_teachers) * ticket_price = 115 := by
  sorry

end museum_entrance_cost_l792_79221


namespace factorial_ratio_l792_79227

theorem factorial_ratio : Nat.factorial 30 / Nat.factorial 28 = 870 := by
  sorry

end factorial_ratio_l792_79227


namespace subset_implies_a_less_than_neg_two_l792_79287

-- Define the sets A and B
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def B (a : ℝ) : Set ℝ := {x | x > a}

-- State the theorem
theorem subset_implies_a_less_than_neg_two (a : ℝ) : A ⊆ B a → a < -2 := by
  sorry

end subset_implies_a_less_than_neg_two_l792_79287


namespace twenty_four_is_eighty_percent_of_thirty_l792_79218

theorem twenty_four_is_eighty_percent_of_thirty : 
  ∃ x : ℝ, 24 = 0.8 * x ∧ x = 30 := by
sorry

end twenty_four_is_eighty_percent_of_thirty_l792_79218


namespace easter_egg_arrangement_l792_79286

theorem easter_egg_arrangement (yellow_eggs : Nat) (blue_eggs : Nat) 
  (min_eggs_per_basket : Nat) (min_baskets : Nat) :
  yellow_eggs = 30 →
  blue_eggs = 42 →
  min_eggs_per_basket = 6 →
  min_baskets = 3 →
  ∃ (eggs_per_basket : Nat),
    eggs_per_basket ≥ min_eggs_per_basket ∧
    eggs_per_basket ∣ yellow_eggs ∧
    eggs_per_basket ∣ blue_eggs ∧
    yellow_eggs / eggs_per_basket ≥ min_baskets ∧
    blue_eggs / eggs_per_basket ≥ min_baskets ∧
    ∀ (n : Nat),
      n > eggs_per_basket →
      ¬(n ≥ min_eggs_per_basket ∧
        n ∣ yellow_eggs ∧
        n ∣ blue_eggs ∧
        yellow_eggs / n ≥ min_baskets ∧
        blue_eggs / n ≥ min_baskets) :=
by
  sorry

end easter_egg_arrangement_l792_79286


namespace prob_at_least_seven_stay_value_l792_79295

def num_friends : ℕ := 8
def num_unsure : ℕ := 5
def num_certain : ℕ := 3
def prob_unsure_stay : ℚ := 3/7

def prob_at_least_seven_stay : ℚ :=
  Nat.choose num_unsure 3 * (prob_unsure_stay ^ 3) * ((1 - prob_unsure_stay) ^ 2) +
  prob_unsure_stay ^ num_unsure

theorem prob_at_least_seven_stay_value :
  prob_at_least_seven_stay = 4563/16807 := by sorry

end prob_at_least_seven_stay_value_l792_79295


namespace tinas_money_left_l792_79289

/-- Calculates the amount of money Tina has left after saving and spending --/
theorem tinas_money_left (june_savings july_savings august_savings : ℕ) 
  (book_expense shoe_expense : ℕ) : 
  june_savings = 27 →
  july_savings = 14 →
  august_savings = 21 →
  book_expense = 5 →
  shoe_expense = 17 →
  (june_savings + july_savings + august_savings) - (book_expense + shoe_expense) = 40 := by
sorry


end tinas_money_left_l792_79289


namespace inequality_proof_l792_79279

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  1 / a^2 + 1 / b^2 + a * b ≥ 2 * Real.sqrt 2 := by sorry

end inequality_proof_l792_79279


namespace mn_positive_necessary_not_sufficient_l792_79261

def not_in_second_quadrant (m n : ℝ) : Prop :=
  (m / n > 0) ∧ (1 / n < 0)

theorem mn_positive_necessary_not_sufficient (m n : ℝ) :
  (not_in_second_quadrant m n → m * n > 0) ∧
  ¬(m * n > 0 → not_in_second_quadrant m n) :=
sorry

end mn_positive_necessary_not_sufficient_l792_79261


namespace root_transformation_l792_79298

/-- Given that a, b, and c are the roots of x^3 - 4x + 6 = 0,
    prove that a - 3, b - 3, and c - 3 are the roots of x^3 + 9x^2 + 23x + 21 = 0 -/
theorem root_transformation (a b c : ℂ) : 
  (a^3 - 4*a + 6 = 0) ∧ (b^3 - 4*b + 6 = 0) ∧ (c^3 - 4*c + 6 = 0) →
  ((a - 3)^3 + 9*(a - 3)^2 + 23*(a - 3) + 21 = 0) ∧
  ((b - 3)^3 + 9*(b - 3)^2 + 23*(b - 3) + 21 = 0) ∧
  ((c - 3)^3 + 9*(c - 3)^2 + 23*(c - 3) + 21 = 0) :=
by sorry

end root_transformation_l792_79298


namespace washing_machine_capacity_l792_79210

theorem washing_machine_capacity (pounds_per_machine : ℕ) (num_machines : ℕ) 
  (h1 : pounds_per_machine = 28) 
  (h2 : num_machines = 8) : 
  pounds_per_machine * num_machines = 224 := by
  sorry

end washing_machine_capacity_l792_79210


namespace purple_cars_count_l792_79282

theorem purple_cars_count (total : ℕ) (blue red orange yellow purple green : ℕ) : 
  total = 1423 →
  blue = 2 * red →
  red = 3 * orange →
  yellow = orange / 2 →
  yellow = 3 * purple →
  green = 5 * purple →
  blue ≥ 200 →
  red ≥ 50 →
  total = blue + red + orange + yellow + purple + green →
  purple = 20 := by
  sorry

#check purple_cars_count

end purple_cars_count_l792_79282


namespace no_primes_from_200_l792_79239

def change_one_digit (n : ℕ) : Set ℕ :=
  {m : ℕ | ∃ (i : Fin 3), ∃ (d : Fin 10), 
    m = n + d * (10 ^ i.val) - (n / (10 ^ i.val) % 10) * (10 ^ i.val)}

theorem no_primes_from_200 :
  ∀ n ∈ change_one_digit 200, ¬ Nat.Prime n :=
sorry

end no_primes_from_200_l792_79239


namespace total_money_is_84_l792_79212

/-- Represents the money redistribution process among three people. -/
def redistribute (j a t : ℚ) : Prop :=
  ∃ (j₁ a₁ t₁ j₂ a₂ t₂ j₃ a₃ t₃ : ℚ),
    -- Step 1: Jan's redistribution
    j₁ + a₁ + t₁ = j + a + t ∧
    a₁ = 2 * a ∧
    t₁ = 2 * t ∧
    -- Step 2: Toy's redistribution
    j₂ + a₂ + t₂ = j₁ + a₁ + t₁ ∧
    j₂ = 2 * j₁ ∧
    a₂ = 2 * a₁ ∧
    -- Step 3: Amy's redistribution
    j₃ + a₃ + t₃ = j₂ + a₂ + t₂ ∧
    j₃ = 2 * j₂ ∧
    t₃ = 2 * t₂

/-- The main theorem stating the total amount of money. -/
theorem total_money_is_84 :
  ∀ j a t : ℚ, t = 48 → redistribute j a t → j + a + t = 84 :=
by sorry

end total_money_is_84_l792_79212


namespace x_power_y_value_l792_79263

theorem x_power_y_value (x y : ℝ) (h : |x + 1/2| + (y - 3)^2 = 0) : x^y = -1/8 := by
  sorry

end x_power_y_value_l792_79263


namespace proper_subset_condition_l792_79205

def M : Set ℝ := {x : ℝ | 2 * x^2 - 3 * x - 2 = 0}

def N (a : ℝ) : Set ℝ := {x : ℝ | a * x = 1}

theorem proper_subset_condition (a : ℝ) :
  N a ⊂ M → a = 0 ∨ a = -2 ∨ a = 1/2 := by sorry

end proper_subset_condition_l792_79205


namespace inequality_proof_l792_79247

theorem inequality_proof (x y : ℝ) (h : x^8 + y^8 ≤ 2) :
  x^2 * y^2 + |x^2 - y^2| ≤ π/2 := by
  sorry

end inequality_proof_l792_79247


namespace prices_and_schemes_l792_79264

def soccer_ball_price : ℕ := 60
def basketball_price : ℕ := 80

def initial_purchase_cost : ℕ := 1600
def initial_soccer_balls : ℕ := 8
def initial_basketballs : ℕ := 14

def total_balls : ℕ := 50
def min_budget : ℕ := 3200
def max_budget : ℕ := 3240

theorem prices_and_schemes :
  (initial_soccer_balls * soccer_ball_price + initial_basketballs * basketball_price = initial_purchase_cost) ∧
  (basketball_price = soccer_ball_price + 20) ∧
  (∀ y : ℕ, y ≤ total_balls →
    (y * soccer_ball_price + (total_balls - y) * basketball_price ≥ min_budget ∧
     y * soccer_ball_price + (total_balls - y) * basketball_price ≤ max_budget)
    ↔ (y = 38 ∨ y = 39 ∨ y = 40)) :=
by sorry

end prices_and_schemes_l792_79264


namespace red_ants_count_l792_79232

theorem red_ants_count (total : ℕ) (black : ℕ) (red : ℕ) : 
  total = 900 → black = 487 → total = red + black → red = 413 := by
  sorry

end red_ants_count_l792_79232


namespace raul_money_left_l792_79269

/-- Calculates the money left after buying comics -/
def money_left (initial_money : ℕ) (num_comics : ℕ) (cost_per_comic : ℕ) : ℕ :=
  initial_money - (num_comics * cost_per_comic)

/-- Proves that Raul's remaining money is correct -/
theorem raul_money_left :
  money_left 87 8 4 = 55 := by
  sorry

end raul_money_left_l792_79269


namespace enhanced_square_triangle_count_l792_79238

/-- A square with diagonals, midpoint connections, and additional bisections -/
structure EnhancedSquare where
  /-- The original square -/
  square : Set (ℝ × ℝ)
  /-- The diagonals of the square -/
  diagonals : Set (Set (ℝ × ℝ))
  /-- The segments connecting midpoints of opposite sides -/
  midpoint_connections : Set (Set (ℝ × ℝ))
  /-- The additional bisections of midpoint connections -/
  bisections : Set (Set (ℝ × ℝ))

/-- A triangle in the enhanced square -/
structure Triangle where
  vertices : Fin 3 → (ℝ × ℝ)

/-- Count the number of triangles in the enhanced square -/
def countTriangles (es : EnhancedSquare) : ℕ :=
  sorry

/-- The main theorem: The number of triangles in the enhanced square is 28 -/
theorem enhanced_square_triangle_count (es : EnhancedSquare) : 
  countTriangles es = 28 := by
  sorry

end enhanced_square_triangle_count_l792_79238


namespace cricketer_bowling_runs_cricketer_last_match_runs_l792_79257

theorem cricketer_bowling_runs (initial_average : ℝ) (initial_wickets : ℕ) 
  (last_match_wickets : ℕ) (average_decrease : ℝ) : ℝ :=
  let final_average := initial_average - average_decrease
  let total_wickets := initial_wickets + last_match_wickets
  let initial_runs := initial_average * initial_wickets
  let final_runs := final_average * total_wickets
  final_runs - initial_runs

theorem cricketer_last_match_runs : 
  cricketer_bowling_runs 12.4 85 5 0.4 = 26 := by sorry

end cricketer_bowling_runs_cricketer_last_match_runs_l792_79257


namespace golden_ratio_expressions_l792_79293

theorem golden_ratio_expressions (θ : Real) (h : θ = 18 * π / 180) :
  let φ := (Real.sqrt 5 - 1) / 4
  φ = Real.sin θ ∧
  φ = Real.cos (10 * π / 180) * Real.cos (82 * π / 180) + Real.sin (10 * π / 180) * Real.sin (82 * π / 180) ∧
  φ = Real.sin (173 * π / 180) * Real.cos (11 * π / 180) - Real.sin (83 * π / 180) * Real.cos (101 * π / 180) ∧
  φ = Real.sqrt ((1 - Real.sin (54 * π / 180)) / 2) :=
by sorry

end golden_ratio_expressions_l792_79293


namespace final_shape_independent_of_initial_fold_l792_79207

/-- Represents a square sheet of paper -/
structure Square :=
  (side : ℝ)
  (side_positive : side > 0)

/-- Represents the folded state of the paper -/
inductive FoldedState
  | Unfolded
  | FoldedOnce
  | FoldedTwice
  | FoldedThrice

/-- Represents the initial fold direction -/
inductive FoldDirection
  | MN
  | AB

/-- Represents the final shape after unfolding -/
structure FinalShape :=
  (shape : Set (ℝ × ℝ))

/-- Function to fold the paper -/
def fold (s : Square) (state : FoldedState) : FoldedState :=
  match state with
  | FoldedState.Unfolded => FoldedState.FoldedOnce
  | FoldedState.FoldedOnce => FoldedState.FoldedTwice
  | FoldedState.FoldedTwice => FoldedState.FoldedThrice
  | FoldedState.FoldedThrice => FoldedState.FoldedThrice

/-- Function to cut and unfold the paper -/
def cutAndUnfold (s : Square) (state : FoldedState) (dir : FoldDirection) : FinalShape :=
  sorry

/-- Theorem stating that the final shape is independent of initial fold direction -/
theorem final_shape_independent_of_initial_fold (s : Square) :
  ∀ (dir1 dir2 : FoldDirection),
    cutAndUnfold s (fold s (fold s (fold s FoldedState.Unfolded))) dir1 =
    cutAndUnfold s (fold s (fold s (fold s FoldedState.Unfolded))) dir2 :=
  sorry

end final_shape_independent_of_initial_fold_l792_79207


namespace point_coordinates_l792_79278

/-- If point A(a, a-2) lies on the x-axis, then the coordinates of point B(a+2, a-1) are (4, 1) -/
theorem point_coordinates (a : ℝ) :
  (a = 2) → (a + 2 = 4 ∧ a - 1 = 1) := by sorry

end point_coordinates_l792_79278


namespace strawberry_basket_price_is_9_l792_79222

/-- Represents the harvest and sales information for Nathan's garden --/
structure GardenSales where
  strawberry_plants : ℕ
  tomato_plants : ℕ
  strawberries_per_plant : ℕ
  tomatoes_per_plant : ℕ
  fruits_per_basket : ℕ
  tomato_basket_price : ℕ
  total_revenue : ℕ

/-- Calculates the price of a basket of strawberries --/
def strawberry_basket_price (g : GardenSales) : ℚ :=
  let total_strawberries := g.strawberry_plants * g.strawberries_per_plant
  let total_tomatoes := g.tomato_plants * g.tomatoes_per_plant
  let strawberry_baskets := total_strawberries / g.fruits_per_basket
  let tomato_baskets := total_tomatoes / g.fruits_per_basket
  let tomato_revenue := tomato_baskets * g.tomato_basket_price
  let strawberry_revenue := g.total_revenue - tomato_revenue
  strawberry_revenue / strawberry_baskets

/-- Theorem stating that the price of a basket of strawberries is 9 --/
theorem strawberry_basket_price_is_9 (g : GardenSales) 
  (h1 : g.strawberry_plants = 5)
  (h2 : g.tomato_plants = 7)
  (h3 : g.strawberries_per_plant = 14)
  (h4 : g.tomatoes_per_plant = 16)
  (h5 : g.fruits_per_basket = 7)
  (h6 : g.tomato_basket_price = 6)
  (h7 : g.total_revenue = 186) :
  strawberry_basket_price g = 9 := by
  sorry

end strawberry_basket_price_is_9_l792_79222


namespace ratio_to_eight_l792_79268

theorem ratio_to_eight : ∃ x : ℚ, (5 : ℚ) / 1 = x / 8 ∧ x = 40 := by sorry

end ratio_to_eight_l792_79268


namespace fraction_equation_solution_l792_79225

theorem fraction_equation_solution (n : ℚ) : 
  (1 : ℚ) / (n + 2) + 3 / (n + 2) + n / (n + 2) = 4 → n = -4/3 := by
  sorry

end fraction_equation_solution_l792_79225


namespace zero_point_of_f_l792_79245

-- Define the function
def f (x : ℝ) : ℝ := (x + 1)^2

-- State the theorem
theorem zero_point_of_f : 
  ∃ (x : ℝ), f x = 0 ∧ x = -1 :=
sorry

end zero_point_of_f_l792_79245


namespace jingyuetan_park_probability_l792_79297

theorem jingyuetan_park_probability (total_envelopes : ℕ) (jingyuetan_tickets : ℕ) 
  (changying_tickets : ℕ) (h1 : total_envelopes = 5) (h2 : jingyuetan_tickets = 3) 
  (h3 : changying_tickets = 2) (h4 : total_envelopes = jingyuetan_tickets + changying_tickets) :
  (jingyuetan_tickets : ℚ) / total_envelopes = 3 / 5 := by
sorry

end jingyuetan_park_probability_l792_79297


namespace BaBr2_molecular_weight_l792_79267

/-- The atomic weight of barium in g/mol -/
def atomic_weight_Ba : ℝ := 137.33

/-- The atomic weight of bromine in g/mol -/
def atomic_weight_Br : ℝ := 79.90

/-- The molecular weight of BaBr2 in g/mol -/
def molecular_weight_BaBr2 : ℝ := atomic_weight_Ba + 2 * atomic_weight_Br

/-- Theorem stating that the molecular weight of BaBr2 is 297.13 g/mol -/
theorem BaBr2_molecular_weight : 
  molecular_weight_BaBr2 = 297.13 := by sorry

end BaBr2_molecular_weight_l792_79267


namespace solution_difference_l792_79208

theorem solution_difference (r s : ℝ) : 
  ((6 * r - 18) / (r^2 + 4 * r - 21) = r + 3) →
  ((6 * s - 18) / (s^2 + 4 * s - 21) = s + 3) →
  r ≠ s →
  r > s →
  r - s = 12 := by sorry

end solution_difference_l792_79208


namespace sum_of_coordinates_for_symmetric_points_l792_79228

-- Define the points P and Q
def P (x : ℝ) : ℝ × ℝ := (x, -3)
def Q (y : ℝ) : ℝ × ℝ := (4, y)

-- Define the property of being symmetric with respect to the origin
def symmetric_about_origin (p q : ℝ × ℝ) : Prop :=
  p.1 = -q.1 ∧ p.2 = -q.2

-- Theorem statement
theorem sum_of_coordinates_for_symmetric_points (x y : ℝ) :
  symmetric_about_origin (P x) (Q y) → x + y = -1 := by
  sorry

end sum_of_coordinates_for_symmetric_points_l792_79228


namespace capital_ratio_specific_case_l792_79255

/-- Given a total loss and Pyarelal's loss, calculate the ratio of Ashok's capital to Pyarelal's capital -/
def capital_ratio (total_loss : ℕ) (pyarelal_loss : ℕ) : ℕ × ℕ :=
  let ashok_loss := total_loss - pyarelal_loss
  (ashok_loss, pyarelal_loss)

/-- Theorem stating that given the specific losses, the capital ratio is 67:603 -/
theorem capital_ratio_specific_case :
  capital_ratio 670 603 = (67, 603) := by
  sorry

end capital_ratio_specific_case_l792_79255


namespace equilateral_triangle_side_length_l792_79217

-- Define the cubic equation
def cubic_equation (x : ℝ) : Prop := x^3 - 9*x^2 + 10*x + 5 = 0

-- Define the property of distinct roots
def distinct_roots (a b c : ℝ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c

-- Theorem statement
theorem equilateral_triangle_side_length
  (a b c : ℝ)
  (ha : cubic_equation a)
  (hb : cubic_equation b)
  (hc : cubic_equation c)
  (hdistinct : distinct_roots a b c) :
  ∃ (side_length : ℝ), side_length = 2 * Real.sqrt 17 ∧
  side_length^2 = (a - b)^2 + (b - c)^2 + (c - a)^2 :=
sorry

end equilateral_triangle_side_length_l792_79217


namespace third_house_price_l792_79260

/-- Brian's commission rate as a decimal -/
def commission_rate : ℚ := 0.02

/-- Selling price of the first house -/
def house1_price : ℚ := 157000

/-- Selling price of the second house -/
def house2_price : ℚ := 499000

/-- Total commission Brian earned from all three sales -/
def total_commission : ℚ := 15620

/-- The selling price of the third house -/
def house3_price : ℚ := (total_commission - (house1_price * commission_rate + house2_price * commission_rate)) / commission_rate

theorem third_house_price :
  house3_price = 125000 :=
by sorry

end third_house_price_l792_79260


namespace composite_n_pow_2016_plus_4_l792_79250

theorem composite_n_pow_2016_plus_4 (n : ℕ) (h : n > 1) :
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ n^2016 + 4 = a * b :=
by
  sorry

end composite_n_pow_2016_plus_4_l792_79250


namespace complex_number_problem_l792_79233

/-- Given a complex number z = b - 2i where b is real, and z / (2 - i) is real,
    prove that z = 4 - 2i and for (z + ai)² to be in the fourth quadrant, -2 < a < 2 -/
theorem complex_number_problem (b : ℝ) (z : ℂ) (h1 : z = b - 2*I) 
  (h2 : ∃ (r : ℝ), z / (2 - I) = r) :
  z = 4 - 2*I ∧ 
  ∀ (a : ℝ), (z + a*I)^2 ∈ {w : ℂ | w.re > 0 ∧ w.im < 0} → -2 < a ∧ a < 2 := by
  sorry

end complex_number_problem_l792_79233


namespace conjugate_sum_product_l792_79265

theorem conjugate_sum_product (c d : ℝ) :
  (c + Real.sqrt d + (c - Real.sqrt d) = -8) →
  ((c + Real.sqrt d) * (c - Real.sqrt d) = 4) →
  c + d = 8 := by
sorry

end conjugate_sum_product_l792_79265


namespace quadratic_equation_roots_solution_satisfies_conditions_l792_79211

theorem quadratic_equation_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ m * x^2 + 2 * x = 1 ∧ m * y^2 + 2 * y = 1) ↔ 
  (m > -1 ∧ m ≠ 0) :=
by sorry

theorem solution_satisfies_conditions : 
  1 > -1 ∧ 1 ≠ 0 :=
by sorry

end quadratic_equation_roots_solution_satisfies_conditions_l792_79211


namespace water_remaining_after_14_pourings_fourteen_pourings_is_minimum_l792_79280

/-- Calculates the fraction of water remaining after n pourings -/
def waterRemaining (n : ℕ) : ℚ :=
  2 / (n + 2 : ℚ)

/-- Theorem: After 14 pourings, exactly 1/8 of the original water remains -/
theorem water_remaining_after_14_pourings :
  waterRemaining 14 = 1/8 := by
  sorry

/-- Theorem: 14 is the smallest number of pourings that leaves exactly 1/8 of the original water -/
theorem fourteen_pourings_is_minimum :
  ∀ k : ℕ, k < 14 → waterRemaining k > 1/8 := by
  sorry

end water_remaining_after_14_pourings_fourteen_pourings_is_minimum_l792_79280


namespace product_derivative_at_zero_l792_79202

/-- Given differentiable real functions f, g, h, prove that (fgh)'(0) = 16 -/
theorem product_derivative_at_zero
  (f g h : ℝ → ℝ)
  (hf : Differentiable ℝ f)
  (hg : Differentiable ℝ g)
  (hh : Differentiable ℝ h)
  (hf0 : f 0 = 1)
  (hg0 : g 0 = 2)
  (hh0 : h 0 = 3)
  (hgh : deriv (g * h) 0 = 4)
  (hhf : deriv (h * f) 0 = 5)
  (hfg : deriv (f * g) 0 = 6) :
  deriv (f * g * h) 0 = 16 := by
sorry

end product_derivative_at_zero_l792_79202


namespace jake_bitcoin_theorem_l792_79296

def jake_bitcoin_problem (initial_fortune : ℕ) (first_donation : ℕ) (second_donation : ℕ) : ℕ :=
  let after_first_donation := initial_fortune - first_donation
  let after_giving_to_brother := after_first_donation / 2
  let after_tripling := after_giving_to_brother * 3
  after_tripling - second_donation

theorem jake_bitcoin_theorem :
  jake_bitcoin_problem 80 20 10 = 80 := by
  sorry

end jake_bitcoin_theorem_l792_79296


namespace brocard_and_steiner_coordinates_l792_79292

/-- Given a triangle with side lengths a, b, and c, this theorem states the trilinear coordinates
    of vertex A1 of the Brocard triangle and the Steiner point. -/
theorem brocard_and_steiner_coordinates (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) :
  ∃ (k₁ k₂ : ℝ),
    k₁ > 0 ∧ k₂ > 0 ∧
    (k₁ * (a * b * c), k₁ * c^3, k₁ * b^3) = (1, 1, 1) ∧
    (k₂ / (a * (b^2 - c^2)), k₂ / (b * (c^2 - a^2)), k₂ / (c * (a^2 - b^2))) = (1, 1, 1) :=
by sorry

end brocard_and_steiner_coordinates_l792_79292


namespace complex_on_imaginary_axis_l792_79266

theorem complex_on_imaginary_axis (a : ℝ) : 
  (Complex.I * ((2 * a + Complex.I) * (1 + Complex.I))).re = 0 → a = 1/2 := by
  sorry

end complex_on_imaginary_axis_l792_79266


namespace temperature_difference_l792_79214

theorem temperature_difference (M L : ℝ) (N : ℝ) : 
  (M = L + N) →
  (abs ((M - 7) - (L + 5)) = 4) →
  (∃ N₁ N₂ : ℝ, (N = N₁ ∨ N = N₂) ∧ N₁ * N₂ = 128) :=
by sorry

end temperature_difference_l792_79214


namespace inequality_proof_l792_79216

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (2*a+b)*(2*a-b)*(a-c) + (2*b+c)*(2*b-c)*(b-a) + (2*c+a)*(2*c-a)*(c-b) ≥ 0 :=
by sorry

end inequality_proof_l792_79216


namespace nabla_example_l792_79276

-- Define the ∇ operation
def nabla (a b c d : ℝ) : ℝ := a * c + b * d

-- Theorem statement
theorem nabla_example : nabla 3 1 4 2 = 14 := by
  sorry

end nabla_example_l792_79276


namespace three_bus_interval_l792_79206

/-- Given a circular bus route with two buses operating at an interval of 21 minutes,
    this theorem proves that when three buses operate on the same route at the same speed,
    the new interval between consecutive buses is 14 minutes. -/
theorem three_bus_interval (interval_two_buses : ℕ) (h : interval_two_buses = 21) :
  let total_time := 2 * interval_two_buses
  let interval_three_buses := total_time / 3
  interval_three_buses = 14 := by
sorry

end three_bus_interval_l792_79206


namespace softball_team_size_l792_79234

theorem softball_team_size :
  ∀ (men women : ℕ),
  women = men + 6 →
  (men : ℚ) / (women : ℚ) = 6 / 10 →
  men + women = 24 :=
by
  sorry

end softball_team_size_l792_79234


namespace wheat_mixture_profit_percentage_wheat_profit_approximately_30_percent_l792_79284

/-- Calculates the profit percentage for a wheat mixture sale --/
theorem wheat_mixture_profit_percentage 
  (wheat1_weight : ℝ) (wheat1_price : ℝ) 
  (wheat2_weight : ℝ) (wheat2_price : ℝ) 
  (selling_price : ℝ) : ℝ :=
  let total_cost := wheat1_weight * wheat1_price + wheat2_weight * wheat2_price
  let total_weight := wheat1_weight + wheat2_weight
  let selling_total := total_weight * selling_price
  let profit := selling_total - total_cost
  let profit_percentage := (profit / total_cost) * 100
  profit_percentage

/-- Proves that the profit percentage is approximately 30% --/
theorem wheat_profit_approximately_30_percent : 
  abs (wheat_mixture_profit_percentage 30 11.50 20 14.25 16.38 - 30) < 0.1 := by
  sorry

end wheat_mixture_profit_percentage_wheat_profit_approximately_30_percent_l792_79284


namespace arithmetic_sequence_problem_l792_79271

/-- An arithmetic sequence with its sum sequence -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The arithmetic sequence
  S : ℕ → ℝ  -- The sum sequence
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n : ℕ, S n = (n : ℝ) * (a 1 + a n) / 2

/-- Theorem: In an arithmetic sequence, if a₈ = 20 and S₇ = 56, then a₁₂ = 32 -/
theorem arithmetic_sequence_problem (seq : ArithmeticSequence) 
    (h₁ : seq.a 8 = 20)
    (h₂ : seq.S 7 = 56) :
  seq.a 12 = 32 := by
  sorry

end arithmetic_sequence_problem_l792_79271


namespace kyles_profit_is_99_l792_79236

/-- The amount of money Kyle will make by selling all his remaining baked goods -/
def kyles_profit (initial_cookies initial_brownies : ℕ) 
                 (kyle_eats_cookies kyle_eats_brownies : ℕ) 
                 (mom_eats_cookies mom_eats_brownies : ℕ) 
                 (cookie_price brownie_price : ℚ) : ℚ :=
  let remaining_cookies := initial_cookies - kyle_eats_cookies - mom_eats_cookies
  let remaining_brownies := initial_brownies - kyle_eats_brownies - mom_eats_brownies
  remaining_cookies * cookie_price + remaining_brownies * brownie_price

/-- Theorem stating that Kyle will make $99 by selling all his remaining baked goods -/
theorem kyles_profit_is_99 : 
  kyles_profit 60 32 2 2 1 2 1 (3/2) = 99 := by
  sorry

end kyles_profit_is_99_l792_79236


namespace cards_eaten_ratio_l792_79248

theorem cards_eaten_ratio (initial_cards new_cards remaining_cards : ℕ) :
  initial_cards = 84 →
  new_cards = 8 →
  remaining_cards = 46 →
  (initial_cards + new_cards - remaining_cards) * 2 = initial_cards + new_cards :=
by
  sorry

end cards_eaten_ratio_l792_79248


namespace power_of_three_equality_l792_79262

theorem power_of_three_equality : (3^5)^6 = 3^12 * 3^18 := by sorry

end power_of_three_equality_l792_79262


namespace two_distinct_roots_l792_79256

theorem two_distinct_roots
  (a b c d : ℝ)
  (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (f_no_roots : ∀ x : ℝ, x^2 + b*x + a ≠ 0)
  (g_condition1 : a^2 + c*a + d = b)
  (g_condition2 : b^2 + c*b + d = a) :
  ∃ x y : ℝ, x ≠ y ∧ a*x^2 + b*x + c = 0 ∧ a*y^2 + b*y + c = 0 :=
by sorry

end two_distinct_roots_l792_79256


namespace privateer_overtakes_at_6_08_pm_l792_79283

/-- Represents the chase scenario between a privateer and a merchantman -/
structure ChaseScenario where
  initial_distance : ℝ
  privateer_initial_speed : ℝ
  merchantman_speed : ℝ
  time_before_damage : ℝ
  new_speed_ratio_privateer : ℝ
  new_speed_ratio_merchantman : ℝ

/-- Calculates the time when the privateer overtakes the merchantman -/
def overtake_time (scenario : ChaseScenario) : ℝ :=
  sorry

/-- Theorem stating that given the specific chase scenario, the privateer overtakes the merchantman at 6:08 p.m. -/
theorem privateer_overtakes_at_6_08_pm (scenario : ChaseScenario) 
  (h1 : scenario.initial_distance = 12)
  (h2 : scenario.privateer_initial_speed = 10)
  (h3 : scenario.merchantman_speed = 7)
  (h4 : scenario.time_before_damage = 3)
  (h5 : scenario.new_speed_ratio_privateer = 13)
  (h6 : scenario.new_speed_ratio_merchantman = 12) :
  overtake_time scenario = 8.1333333333 :=
  sorry

#eval 10 + 8.1333333333  -- Should output approximately 18.1333333333, representing 6:08 p.m.

end privateer_overtakes_at_6_08_pm_l792_79283


namespace society_member_sum_or_double_l792_79231

theorem society_member_sum_or_double {n : ℕ} (hn : n = 1978) :
  ∀ (f : Fin n → Fin 6),
  ∃ (i : Fin 6) (a b c : Fin n),
    f a = i ∧ f b = i ∧ f c = i ∧
    (a.val + 1 = b.val + c.val + 2 ∨ a.val + 1 = 2 * (b.val + 1)) := by
  sorry


end society_member_sum_or_double_l792_79231


namespace ship_cannot_escape_illumination_l792_79243

/-- Represents a lighthouse with a rotating beam -/
structure Lighthouse where
  beam_length : ℝ
  beam_velocity : ℝ

/-- Represents a ship moving towards the lighthouse -/
structure Ship where
  speed : ℝ
  initial_distance : ℝ

/-- Theorem: A ship cannot reach the lighthouse without being illuminated -/
theorem ship_cannot_escape_illumination (L : Lighthouse) (S : Ship) 
  (h1 : S.speed ≤ L.beam_velocity / 8)
  (h2 : S.initial_distance = L.beam_length) : 
  ∃ (t : ℝ), t > 0 ∧ S.initial_distance - S.speed * t > 0 ∧ 
  2 * π * L.beam_length / L.beam_velocity ≥ t :=
sorry

end ship_cannot_escape_illumination_l792_79243


namespace binomial_expansion_x_squared_term_l792_79272

theorem binomial_expansion_x_squared_term (x : ℝ) (n : ℕ) :
  (∃ r : ℕ, r ≤ n ∧ (n.choose r) * x^((5*r)/2 - n) = x^2) → n = 8 := by
  sorry

end binomial_expansion_x_squared_term_l792_79272
