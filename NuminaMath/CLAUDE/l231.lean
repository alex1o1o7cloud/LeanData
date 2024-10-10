import Mathlib

namespace f_minimum_and_inequality_l231_23127

noncomputable def f (x : ℝ) := x * Real.log x

theorem f_minimum_and_inequality :
  (∃ (x : ℝ), x > 0 ∧ f x = -1 / Real.exp 1) ∧
  (∀ (x : ℝ), x > 0 → Real.log x > 1 / Real.exp x - 2 / (Real.exp 1 * x)) := by
  sorry

end f_minimum_and_inequality_l231_23127


namespace blender_sunday_price_l231_23151

/-- The Sunday price of a blender after applying discounts -/
theorem blender_sunday_price (original_price : ℝ) (regular_discount : ℝ) (sunday_discount : ℝ) :
  original_price = 250 →
  regular_discount = 0.60 →
  sunday_discount = 0.25 →
  original_price * (1 - regular_discount) * (1 - sunday_discount) = 75 := by
sorry

end blender_sunday_price_l231_23151


namespace f_properties_l231_23130

def f (x : ℝ) := x^3 + 2*x^2 - 4*x + 5

theorem f_properties :
  (f (-2) = 13) ∧
  (HasDerivAt f 0 (-2)) ∧
  (∀ x ∈ Set.Icc (-3) 0, f x ≤ 13) ∧
  (∃ x ∈ Set.Icc (-3) 0, f x = 13) ∧
  (∀ x ∈ Set.Icc (-3) 0, f x ≥ 5) ∧
  (∃ x ∈ Set.Icc (-3) 0, f x = 5) :=
by sorry

end f_properties_l231_23130


namespace chord_ratio_implies_slope_l231_23140

theorem chord_ratio_implies_slope (k : ℝ) (h1 : k > 0) : 
  let l := {(x, y) : ℝ × ℝ | y = k * x}
  let C1 := {(x, y) : ℝ × ℝ | (x - 1)^2 + y^2 = 1}
  let C2 := {(x, y) : ℝ × ℝ | (x - 3)^2 + y^2 = 1}
  let chord1 := {p : ℝ × ℝ | p ∈ l ∩ C1}
  let chord2 := {p : ℝ × ℝ | p ∈ l ∩ C2}
  (∃ (p q : ℝ × ℝ), p ∈ chord1 ∧ q ∈ chord1 ∧ p ≠ q) →
  (∃ (r s : ℝ × ℝ), r ∈ chord2 ∧ s ∈ chord2 ∧ r ≠ s) →
  (∃ (p q r s : ℝ × ℝ), p ∈ chord1 ∧ q ∈ chord1 ∧ r ∈ chord2 ∧ s ∈ chord2 ∧
    dist p q / dist r s = 3) →
  k = 1/3 :=
by sorry


end chord_ratio_implies_slope_l231_23140


namespace square_circle_radius_l231_23173

theorem square_circle_radius (r : ℝ) (h : r > 0) :
  4 * r * Real.sqrt 2 = π * r^2 → r = 4 * Real.sqrt 2 / π := by
  sorry

end square_circle_radius_l231_23173


namespace solve_for_k_l231_23159

-- Define the functions f and g
def f (x : ℝ) : ℝ := 3 * x^3 - 2 * x + 4
def g (k : ℝ) (x : ℝ) : ℝ := 2 * x^3 + x^2 - 5 * x - k

-- State the theorem
theorem solve_for_k : ∃ k : ℝ, f 3 - g k 3 = 14 ∧ k = -17 := by sorry

end solve_for_k_l231_23159


namespace problem_solution_l231_23152

theorem problem_solution :
  (∀ x : ℝ, x^2 - 5*x + 4 < 0 ∧ (x - 2)*(x - 5) < 0 ↔ 2 < x ∧ x < 4) ∧
  (∀ a : ℝ, a > 0 ∧ 
    (∀ x : ℝ, (x - 2)*(x - 5) < 0 → x^2 - 5*a*x + 4*a^2 < 0) ∧ 
    (∃ x : ℝ, x^2 - 5*a*x + 4*a^2 < 0 ∧ (x - 2)*(x - 5) ≥ 0) ↔
    5/4 ≤ a ∧ a ≤ 2) :=
by sorry

end problem_solution_l231_23152


namespace trick_deck_cost_l231_23195

theorem trick_deck_cost (tom_decks : ℕ) (friend_decks : ℕ) (total_spent : ℕ) 
  (h1 : tom_decks = 3)
  (h2 : friend_decks = 5)
  (h3 : total_spent = 64) :
  (total_spent : ℚ) / (tom_decks + friend_decks : ℚ) = 8 := by
  sorry

end trick_deck_cost_l231_23195


namespace problem_statement_l231_23124

theorem problem_statement (x : ℝ) (h : x + 3 = 10) : 5 * x + 15 = 50 := by
  sorry

end problem_statement_l231_23124


namespace teds_overall_correct_percentage_l231_23105

theorem teds_overall_correct_percentage
  (t : ℝ) -- total number of problems
  (h_t_pos : t > 0) -- ensure t is positive
  (independent_solving : ℝ := 0.4 * t) -- 40% of problems solved independently
  (collaborative_solving : ℝ := 0.6 * t) -- 60% of problems solved collaboratively
  (ned_independent_correct : ℝ := 0.7 * independent_solving) -- Ned's correct answers for independent solving
  (ned_overall_correct : ℝ := 0.82 * t) -- Ned's overall correct answers
  (ted_independent_correct : ℝ := 0.85 * independent_solving) -- Ted's correct answers for independent solving
  : (ted_independent_correct + (ned_overall_correct - ned_independent_correct)) / t = 0.88 := by
  sorry

end teds_overall_correct_percentage_l231_23105


namespace opposite_of_one_half_l231_23144

theorem opposite_of_one_half : 
  (-(1/2) : ℚ) = (-1/2 : ℚ) := by sorry

end opposite_of_one_half_l231_23144


namespace tangent_slope_circle_l231_23112

/-- Slope of the tangent line to a circle -/
theorem tangent_slope_circle (center_x center_y tangent_x tangent_y : ℝ) :
  center_x = 2 →
  center_y = 3 →
  tangent_x = 7 →
  tangent_y = 8 →
  (tangent_y - center_y) / (tangent_x - center_x) = 1 →
  -(((tangent_y - center_y) / (tangent_x - center_x))⁻¹) = -1 :=
by sorry

end tangent_slope_circle_l231_23112


namespace fewer_children_than_adults_l231_23184

theorem fewer_children_than_adults : 
  ∀ (children seniors : ℕ),
  58 + children + seniors = 127 →
  seniors = 2 * children →
  58 - children = 35 := by
sorry

end fewer_children_than_adults_l231_23184


namespace three_integer_pairs_satisfy_equation_l231_23113

theorem three_integer_pairs_satisfy_equation : 
  ∃! (s : Finset (ℤ × ℤ)), 
    (∀ (x y : ℤ), (x, y) ∈ s ↔ x^3 + y^2 = 2*y + 1) ∧ 
    s.card = 3 := by
  sorry

end three_integer_pairs_satisfy_equation_l231_23113


namespace solve_equation_l231_23157

theorem solve_equation : ∃ x : ℚ, 5 * (x - 9) = 7 * (3 - 3 * x) + 10 ∧ x = 38 / 13 := by
  sorry

end solve_equation_l231_23157


namespace max_value_theorem_l231_23128

theorem max_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x : ℝ, 2 * (a - x) * (x + Real.sqrt (x^2 + 3 * b^2)) ≤ a^2 + 3 * b^2) ∧
  (∃ x : ℝ, 2 * (a - x) * (x + Real.sqrt (x^2 + 3 * b^2)) = a^2 + 3 * b^2) :=
by sorry

end max_value_theorem_l231_23128


namespace root_sum_theorem_l231_23162

theorem root_sum_theorem (a b : ℝ) : 
  a^2 - 5*a + 6 = 0 → 
  b^2 - 5*b + 6 = 0 → 
  a^3 + a^4*b^2 + a^2*b^4 + b^3 + a*b^3 + b*a^3 = 683 := by
  sorry

end root_sum_theorem_l231_23162


namespace max_m_value_l231_23129

theorem max_m_value (a b m : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : ∀ a b, a > 0 → b > 0 → m / (3 * a + b) - 3 / a - 1 / b ≤ 0) : 
  m ≤ 16 :=
sorry

end max_m_value_l231_23129


namespace quadratic_equation_roots_range_l231_23171

theorem quadratic_equation_roots_range (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   k * x₁^2 - Real.sqrt (2 * k + 1) * x₁ + 1 = 0 ∧
   k * x₂^2 - Real.sqrt (2 * k + 1) * x₂ + 1 = 0) ∧
  k ≠ 0 →
  -1/2 ≤ k ∧ k < 1/2 ∧ k ≠ 0 :=
by sorry

end quadratic_equation_roots_range_l231_23171


namespace square_difference_equation_l231_23139

theorem square_difference_equation : 9^2 - 8^2 = 17 := by
  sorry

end square_difference_equation_l231_23139


namespace tangent_line_to_ellipse_l231_23196

/-- The ellipse defined by x²/4 + y²/m = 1 -/
def is_ellipse (x y m : ℝ) : Prop :=
  x^2 / 4 + y^2 / m = 1

/-- The line y = mx + 2 -/
def is_line (x y m : ℝ) : Prop :=
  y = m * x + 2

/-- The line is tangent to the ellipse -/
def is_tangent (m : ℝ) : Prop :=
  ∃ (x y : ℝ), is_ellipse x y m ∧ is_line x y m ∧
  ∀ (x' y' : ℝ), is_ellipse x' y' m → is_line x' y' m → (x = x' ∧ y = y')

theorem tangent_line_to_ellipse (m : ℝ) :
  is_tangent m → m = 1 := by
  sorry

end tangent_line_to_ellipse_l231_23196


namespace first_reduction_percentage_l231_23165

theorem first_reduction_percentage (x : ℝ) : 
  (1 - 0.7) * (1 - x / 100) = 1 - 0.775 → x = 25 := by sorry

end first_reduction_percentage_l231_23165


namespace student_marks_average_l231_23189

theorem student_marks_average (P C M : ℕ) (h : P + C + M = P + 150) :
  (C + M) / 2 = 75 := by
  sorry

end student_marks_average_l231_23189


namespace two_digit_fraction_problem_l231_23169

theorem two_digit_fraction_problem :
  ∃ (A B : ℕ), 
    (10 ≤ A ∧ A ≤ 99) ∧ 
    (10 ≤ B ∧ B ≤ 99) ∧ 
    (A - 5 : ℚ) / A + 4 / B = 1 ∧
    (∀ A' : ℕ, (10 ≤ A' ∧ A' ≤ 99) → (A' - 5 : ℚ) / A' + 4 / B = 1 → A ≤ A') ∧
    (∀ B' : ℕ, (10 ≤ B' ∧ B' ≤ 99) → (A - 5 : ℚ) / A + 4 / B' = 1 → B' ≤ B) ∧
    A = 15 ∧ B = 76 :=
by sorry

end two_digit_fraction_problem_l231_23169


namespace min_shift_for_symmetry_l231_23133

theorem min_shift_for_symmetry (m : ℝ) : 
  m > 0 ∧ 
  (∀ x : ℝ, 2 * Real.sin (x + m + π/3) = 2 * Real.sin (-x + m + π/3)) →
  m ≥ π/6 :=
by sorry

end min_shift_for_symmetry_l231_23133


namespace sum_of_specific_common_multiples_l231_23149

theorem sum_of_specific_common_multiples (a b : ℕ) (h : Nat.lcm a b = 21) :
  (9 * 21) + (10 * 21) + (11 * 21) = 630 := by
  sorry

end sum_of_specific_common_multiples_l231_23149


namespace perpendicular_segments_equal_length_l231_23103

-- Define a type for lines in a plane
def Line : Type := ℝ → ℝ → Prop

-- Define what it means for two lines to be parallel
def Parallel (l₁ l₂ : Line) : Prop := sorry

-- Define a perpendicular segment between two lines
def PerpendicularSegment (l₁ l₂ : Line) : Type := sorry

-- Define the length of a perpendicular segment
def Length (seg : PerpendicularSegment l₁ l₂) : ℝ := sorry

-- Theorem statement
theorem perpendicular_segments_equal_length 
  (l₁ l₂ : Line) (h : Parallel l₁ l₂) :
  ∀ (seg₁ seg₂ : PerpendicularSegment l₁ l₂), 
  Length seg₁ = Length seg₂ :=
sorry

end perpendicular_segments_equal_length_l231_23103


namespace choose_two_from_four_l231_23145

theorem choose_two_from_four (n : ℕ) (h : n = 4) : Nat.choose n 2 = 6 := by
  sorry

end choose_two_from_four_l231_23145


namespace inscribed_square_area_is_2275_l231_23183

/-- A right triangle with an inscribed square -/
structure InscribedSquareTriangle where
  /-- The length of side XY of the triangle -/
  xy : ℝ
  /-- The length of side ZQ of the triangle -/
  zq : ℝ
  /-- The side length of the inscribed square -/
  square_side : ℝ
  /-- Condition ensuring the square fits in the triangle -/
  square_fits : square_side ≤ min xy zq

/-- The area of the inscribed square in the triangle -/
def inscribed_square_area (t : InscribedSquareTriangle) : ℝ := t.square_side ^ 2

/-- The theorem stating the area of the inscribed square -/
theorem inscribed_square_area_is_2275 
  (t : InscribedSquareTriangle) 
  (h1 : t.xy = 35) 
  (h2 : t.zq = 65) : 
  inscribed_square_area t = 2275 := by
  sorry

end inscribed_square_area_is_2275_l231_23183


namespace pencil_carton_cost_pencil_carton_cost_proof_l231_23182

/-- The cost of a carton of pencils given the following conditions:
  1. Erasers cost 3 dollars per carton
  2. Total order is 100 cartons
  3. Total order cost is 360 dollars
  4. The order includes 20 cartons of pencils -/
theorem pencil_carton_cost : ℝ :=
  let eraser_cost : ℝ := 3
  let total_cartons : ℕ := 100
  let total_cost : ℝ := 360
  let pencil_cartons : ℕ := 20
  6

/-- Proof that the cost of a carton of pencils is 6 dollars -/
theorem pencil_carton_cost_proof :
  pencil_carton_cost = 6 := by sorry

end pencil_carton_cost_pencil_carton_cost_proof_l231_23182


namespace vector_parallel_condition_l231_23148

/-- Given two 2D vectors a and b, if a + b is parallel to a - b, 
    then the first component of a is -4/3. -/
theorem vector_parallel_condition (a b : ℝ × ℝ) :
  a.1 = m ∧ a.2 = 2 ∧ b = (2, -3) ∧ 
  (∃ (k : ℝ), k ≠ 0 ∧ (a + b) = k • (a - b)) →
  m = -4/3 := by sorry

end vector_parallel_condition_l231_23148


namespace rhombuses_in_5x5_grid_l231_23160

/-- Represents a grid of equilateral triangles -/
structure TriangleGrid where
  rows : Nat
  cols : Nat

/-- Counts the number of rhombuses in a triangle grid -/
def count_rhombuses (grid : TriangleGrid) : Nat :=
  sorry

/-- Theorem: In a 5x5 grid of equilateral triangles, there are 30 rhombuses -/
theorem rhombuses_in_5x5_grid :
  let grid : TriangleGrid := { rows := 5, cols := 5 }
  count_rhombuses grid = 30 := by
  sorry

end rhombuses_in_5x5_grid_l231_23160


namespace A_equals_set_l231_23136

def A : Set ℤ := {x | -1 < |x - 1| ∧ |x - 1| < 2}

theorem A_equals_set : A = {0, 1, 2} := by sorry

end A_equals_set_l231_23136


namespace gcd_power_two_minus_one_l231_23194

theorem gcd_power_two_minus_one : 
  Nat.gcd (2^2024 - 1) (2^2000 - 1) = 2^24 - 1 := by
  sorry

end gcd_power_two_minus_one_l231_23194


namespace math_team_selection_l231_23177

/-- The number of ways to select k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem math_team_selection :
  let girls := 5
  let boys := 5
  let girls_on_team := 3
  let boys_on_team := 2
  (choose girls girls_on_team) * (choose boys boys_on_team) = 100 := by
  sorry

end math_team_selection_l231_23177


namespace binomial_expansion_coefficient_l231_23179

theorem binomial_expansion_coefficient (a : ℝ) : 
  (Nat.choose 6 3 : ℝ) * a^3 * 2^3 = 5/2 → a = 1/4 := by
  sorry

end binomial_expansion_coefficient_l231_23179


namespace min_sum_positive_reals_l231_23192

theorem min_sum_positive_reals (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / (3 * b)) + (b / (6 * c)) + (c / (9 * a)) ≥ 3 * (1 / Real.rpow 162 (1/3)) :=
sorry

end min_sum_positive_reals_l231_23192


namespace smallest_sum_is_381_l231_23154

/-- A permutation of the digits 1 to 6 -/
def Digit6Perm := Fin 6 → Fin 6

/-- Checks if a permutation is valid (bijective) -/
def isValidPerm (p : Digit6Perm) : Prop :=
  Function.Bijective p

/-- Converts a permutation to two 3-digit numbers -/
def permToNumbers (p : Digit6Perm) : ℕ × ℕ :=
  ((p 0 + 1) * 100 + (p 1 + 1) * 10 + (p 2 + 1),
   (p 3 + 1) * 100 + (p 4 + 1) * 10 + (p 5 + 1))

/-- Sums the two numbers obtained from a permutation -/
def sumFromPerm (p : Digit6Perm) : ℕ :=
  let (n1, n2) := permToNumbers p
  n1 + n2

/-- The main theorem stating that 381 is the smallest possible sum -/
theorem smallest_sum_is_381 :
  ∀ p : Digit6Perm, isValidPerm p → sumFromPerm p ≥ 381 :=
sorry

end smallest_sum_is_381_l231_23154


namespace factorization_equality_l231_23197

theorem factorization_equality (x₁ x₂ : ℝ) :
  x₁^3 - 2*x₁^2*x₂ - x₁ + 2*x₂ = (x₁ - 1) * (x₁ + 1) * (x₁ - 2*x₂) := by
  sorry

end factorization_equality_l231_23197


namespace quadratic_root_implies_m_l231_23147

theorem quadratic_root_implies_m (m : ℝ) : 
  (∃ x : ℝ, x^2 - 3*x + m = 0 ∧ x = 1) → m = 2 := by
sorry

end quadratic_root_implies_m_l231_23147


namespace elevator_trips_l231_23122

def masses : List ℕ := [150, 62, 63, 66, 70, 75, 79, 84, 95, 96, 99]
def capacity : ℕ := 190

def is_valid_trip (trip : List ℕ) : Bool :=
  trip.sum ≤ capacity

def min_trips (masses : List ℕ) (capacity : ℕ) : ℕ :=
  sorry

theorem elevator_trips :
  min_trips masses capacity = 6 := by
  sorry

end elevator_trips_l231_23122


namespace prob_sum_5_is_one_ninth_l231_23150

/-- The number of sides on each die -/
def sides : ℕ := 6

/-- The total number of possible outcomes when rolling two dice -/
def total_outcomes : ℕ := sides * sides

/-- The number of favorable outcomes (sum of 5) -/
def favorable_outcomes : ℕ := 4

/-- The probability of rolling a sum of 5 with two dice -/
def prob_sum_5 : ℚ := favorable_outcomes / total_outcomes

theorem prob_sum_5_is_one_ninth :
  prob_sum_5 = 1 / 9 := by sorry

end prob_sum_5_is_one_ninth_l231_23150


namespace contrapositive_equivalence_l231_23156

/-- The proposition "If m > 0, then the equation x^2 + x - m = 0 has real roots" -/
def original_proposition (m : ℝ) : Prop :=
  m > 0 → ∃ x : ℝ, x^2 + x - m = 0

/-- The contrapositive of the original proposition -/
def contrapositive (m : ℝ) : Prop :=
  (¬∃ x : ℝ, x^2 + x - m = 0) → m ≤ 0

/-- Theorem stating that the contrapositive is equivalent to the expected form -/
theorem contrapositive_equivalence :
  ∀ m : ℝ, contrapositive m ↔ (¬∃ x : ℝ, x^2 + x - m = 0 → m ≤ 0) :=
by sorry

end contrapositive_equivalence_l231_23156


namespace wall_building_time_l231_23143

/-- Given that 18 persons can build a 140 m long wall in 42 days, 
    prove that 30 persons will take 18 days to complete a similar 100 m long wall. -/
theorem wall_building_time 
  (original_workers : ℕ) 
  (original_length : ℝ) 
  (original_days : ℕ) 
  (new_workers : ℕ) 
  (new_length : ℝ) 
  (h1 : original_workers = 18) 
  (h2 : original_length = 140) 
  (h3 : original_days = 42) 
  (h4 : new_workers = 30) 
  (h5 : new_length = 100) :
  (new_length / new_workers) / (original_length / original_workers) * original_days = 18 :=
by sorry

end wall_building_time_l231_23143


namespace consecutive_integers_median_l231_23119

theorem consecutive_integers_median (n : ℕ) (S : ℕ) (h1 : n = 81) (h2 : S = 9^5) :
  let median := S / n
  median = 729 := by
  sorry

end consecutive_integers_median_l231_23119


namespace square_perimeter_l231_23167

theorem square_perimeter (area : ℝ) (side : ℝ) (h1 : area = 625) (h2 : side * side = area) :
  4 * side = 100 := by
  sorry

end square_perimeter_l231_23167


namespace B_age_is_18_l231_23185

/-- Given three people A, B, and C with the following conditions:
  1. A is two years older than B
  2. B is twice as old as C
  3. The sum of their ages is 47
  Prove that B is 18 years old -/
theorem B_age_is_18 (A B C : ℕ) 
  (h1 : A = B + 2)
  (h2 : B = 2 * C)
  (h3 : A + B + C = 47) :
  B = 18 := by
  sorry

end B_age_is_18_l231_23185


namespace stacys_current_height_l231_23131

/-- Prove Stacy's current height given the conditions of the problem -/
theorem stacys_current_height 
  (S J M S' J' M' : ℕ) 
  (h1 : S = 50)
  (h2 : S' = J' + 6)
  (h3 : J' = J + 1)
  (h4 : M' = M + 2 * (J' - J))
  (h5 : S + J + M = 128)
  (h6 : S' + J' + M' = 140) :
  S' = 59 := by
  sorry

end stacys_current_height_l231_23131


namespace point_placement_result_l231_23158

theorem point_placement_result (x : ℕ) : ∃ x > 0, 9 * x - 8 = 82 := by
  sorry

#check point_placement_result

end point_placement_result_l231_23158


namespace factory_production_l231_23117

/-- Given a factory that produces a certain number of toys per week and workers
    that work a certain number of days per week, calculate the number of toys
    produced each day (rounded down). -/
def toysPerDay (toysPerWeek : ℕ) (daysWorked : ℕ) : ℕ :=
  toysPerWeek / daysWorked

/-- Theorem stating that for a factory producing 6400 toys per week with workers
    working 3 days a week, the number of toys produced each day is 2133. -/
theorem factory_production :
  toysPerDay 6400 3 = 2133 := by
  sorry

end factory_production_l231_23117


namespace jack_afternoon_emails_l231_23111

/-- The number of emails Jack received in the morning -/
def morning_emails : ℕ := 10

/-- The difference between morning and afternoon emails -/
def email_difference : ℕ := 7

/-- The number of emails Jack received in the afternoon -/
def afternoon_emails : ℕ := morning_emails - email_difference

theorem jack_afternoon_emails : afternoon_emails = 3 := by
  sorry

end jack_afternoon_emails_l231_23111


namespace complex_number_problem_l231_23153

-- Define the complex number z
variable (z : ℂ)

-- Define the conditions
def condition1 : Prop := (z + 3 + 4 * Complex.I).im = 0
def condition2 : Prop := (z / (1 - 2 * Complex.I)).im = 0
def condition3 (m : ℝ) : Prop := 
  let w := (z - m * Complex.I)^2
  w.re < 0 ∧ w.im > 0

-- State the theorem
theorem complex_number_problem (h1 : condition1 z) (h2 : condition2 z) :
  z = 2 - 4 * Complex.I ∧ 
  ∃ m₀ : ℝ, ∀ m : ℝ, condition3 z m ↔ m < m₀ ∧ m₀ = -6 :=
sorry

end complex_number_problem_l231_23153


namespace isosceles_triangles_perimeter_l231_23120

-- Define the points
variable (P Q R S : ℝ × ℝ)

-- Define the distances between points
def PQ : ℝ := sorry
def PR : ℝ := sorry
def QR : ℝ := sorry
def PS : ℝ := sorry
def SR : ℝ := sorry

-- Define x
def x : ℝ := sorry

-- State the theorem
theorem isosceles_triangles_perimeter (P Q R S : ℝ × ℝ) (PQ PR QR PS SR x : ℝ) :
  PQ = PR →                           -- Triangle PQR is isosceles
  PS = SR →                           -- Triangle PRS is isosceles
  PS = x →                            -- PS = x
  SR = x →                            -- SR = x
  PQ + QR + PR = 22 →                 -- Perimeter of Triangle PQR is 22
  PR + PS + SR = 22 →                 -- Perimeter of Triangle PRS is 22
  PQ + QR + SR + PS = 24 →            -- Perimeter of quadrilateral PQRS is 24
  x = 6 := by
sorry

end isosceles_triangles_perimeter_l231_23120


namespace rational_function_value_l231_23110

/-- A rational function with specific properties -/
def rational_function (p q : ℝ → ℝ) : Prop :=
  (∃ k : ℝ, ∀ x, p x = k * x) ∧  -- p is linear
  (∃ a b c : ℝ, ∀ x, q x = a * x^2 + b * x + c) ∧  -- q is quadratic
  p 0 / q 0 = 0 ∧  -- passes through (0,0)
  p 4 / q 4 = 2 ∧  -- passes through (4,2)
  q (-4) = 0 ∧  -- vertical asymptote at x = -4
  q 1 = 0  -- vertical asymptote at x = 1

theorem rational_function_value (p q : ℝ → ℝ) :
  rational_function p q → p (-1) / q (-1) = 8/3 := by
  sorry

end rational_function_value_l231_23110


namespace only_one_satisfies_property_l231_23181

theorem only_one_satisfies_property : ∃! (n : ℕ), 
  n > 0 ∧ 
  (∀ (a : ℤ), Odd a → (a^2 : ℤ) ≤ n → a ∣ n) :=
by sorry

end only_one_satisfies_property_l231_23181


namespace sin_alpha_plus_pi_fourth_l231_23116

theorem sin_alpha_plus_pi_fourth (α : Real) :
  (Complex.mk (Real.sin α - 3/5) (Real.cos α - 4/5)).re = 0 →
  Real.sin (α + Real.pi/4) = -Real.sqrt 2 / 10 := by
  sorry

end sin_alpha_plus_pi_fourth_l231_23116


namespace sin_sum_to_product_l231_23175

theorem sin_sum_to_product (x : ℝ) : 
  Real.sin (6 * x) + Real.sin (10 * x) = 2 * Real.sin (8 * x) * Real.cos (2 * x) := by
  sorry

end sin_sum_to_product_l231_23175


namespace pet_store_problem_l231_23132

/-- The number of ways to choose pets for Emily, John, and Lucy -/
def pet_store_combinations (num_puppies num_kittens num_rabbits : ℕ) : ℕ :=
  num_puppies * num_kittens * num_rabbits * 6

/-- Theorem: Given 20 puppies, 10 kittens, and 12 rabbits, there are 14400 ways for
    Emily, John, and Lucy to buy pets, ensuring they all get different types of pets. -/
theorem pet_store_problem : pet_store_combinations 20 10 12 = 14400 := by
  sorry

end pet_store_problem_l231_23132


namespace swing_slide_wait_time_difference_l231_23187

theorem swing_slide_wait_time_difference :
  let swingKids : ℕ := 6
  let slideKids : ℕ := 4 * swingKids
  let swingWaitTime1 : ℝ := 3.5 * 60  -- 3.5 minutes in seconds
  let slideWaitTime1 : ℝ := 45  -- 45 seconds
  let rounds : ℕ := 3
  
  let swingTotalWait : ℝ := swingKids * (swingWaitTime1 * (1 - 2^rounds) / (1 - 2))
  let slideTotalWait : ℝ := slideKids * (slideWaitTime1 * (1 - 2^rounds) / (1 - 2))
  
  swingTotalWait - slideTotalWait = 1260
  := by sorry

end swing_slide_wait_time_difference_l231_23187


namespace function_value_at_three_l231_23188

/-- Given a positive real number and a function satisfying certain conditions,
    prove that the function evaluated at 3 equals 1/3. -/
theorem function_value_at_three (x : ℝ) (f : ℝ → ℝ) 
    (h1 : x > 0)
    (h2 : x + 17 = 60 * f x)
    (h3 : x = 3) : 
  f 3 = 1/3 := by
  sorry

end function_value_at_three_l231_23188


namespace carries_babysitting_earnings_l231_23123

/-- Carrie's babysitting earnings problem -/
theorem carries_babysitting_earnings 
  (iphone_cost : ℕ) 
  (trade_in_value : ℕ) 
  (weeks_to_work : ℕ) 
  (h1 : iphone_cost = 800)
  (h2 : trade_in_value = 240)
  (h3 : weeks_to_work = 7) :
  (iphone_cost - trade_in_value) / weeks_to_work = 80 :=
by sorry

end carries_babysitting_earnings_l231_23123


namespace quadrilateral_area_l231_23178

-- Define the vertices of the quadrilateral
def v1 : ℝ × ℝ := (0, 0)
def v2 : ℝ × ℝ := (4, 3)
def v3 : ℝ × ℝ := (7, 0)
def v4 : ℝ × ℝ := (4, 4)

-- Define the quadrilateral as a list of vertices
def quadrilateral : List (ℝ × ℝ) := [v1, v2, v3, v4]

-- Function to calculate the area of a quadrilateral using its vertices
def quadrilateralArea (vertices : List (ℝ × ℝ)) : ℝ := sorry

-- Theorem stating that the area of the given quadrilateral is 3.5
theorem quadrilateral_area : quadrilateralArea quadrilateral = 3.5 := by sorry

end quadrilateral_area_l231_23178


namespace cylinder_inscribed_sphere_tangent_spheres_l231_23125

theorem cylinder_inscribed_sphere_tangent_spheres 
  (cylinder_radius : ℝ) 
  (cylinder_height : ℝ) 
  (large_sphere_radius : ℝ) 
  (small_sphere_radius : ℝ) :
  cylinder_radius = 15 →
  cylinder_height = 16 →
  large_sphere_radius = Real.sqrt (cylinder_radius^2 + (cylinder_height/2)^2) →
  large_sphere_radius = small_sphere_radius + Real.sqrt ((cylinder_height/2 + small_sphere_radius)^2 + (2*small_sphere_radius*Real.sqrt 3/3)^2) →
  small_sphere_radius = (15 * Real.sqrt 37 - 75) / 4 := by
  sorry

end cylinder_inscribed_sphere_tangent_spheres_l231_23125


namespace truth_values_of_p_and_q_l231_23138

theorem truth_values_of_p_and_q (hp_and_q : ¬(p ∧ q)) (hnot_p_or_q : ¬p ∨ q) :
  ¬p ∧ (q ∨ ¬q) :=
by sorry

end truth_values_of_p_and_q_l231_23138


namespace quadratic_inequality_solution_set_l231_23186

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 + 3*x - 4 < 0} = Set.Ioo (-4 : ℝ) 1 := by sorry

end quadratic_inequality_solution_set_l231_23186


namespace granger_spam_cans_l231_23176

/-- Represents the grocery items and their prices --/
structure GroceryItems where
  spam_price : ℕ
  peanut_butter_price : ℕ
  bread_price : ℕ

/-- Represents the quantities of items bought --/
structure Quantities where
  spam_cans : ℕ
  peanut_butter_jars : ℕ
  bread_loaves : ℕ

/-- Calculates the total cost of the groceries --/
def total_cost (items : GroceryItems) (quantities : Quantities) : ℕ :=
  items.spam_price * quantities.spam_cans +
  items.peanut_butter_price * quantities.peanut_butter_jars +
  items.bread_price * quantities.bread_loaves

/-- Theorem stating that Granger bought 4 cans of Spam --/
theorem granger_spam_cans :
  ∀ (items : GroceryItems) (quantities : Quantities),
    items.spam_price = 3 →
    items.peanut_butter_price = 5 →
    items.bread_price = 2 →
    quantities.peanut_butter_jars = 3 →
    quantities.bread_loaves = 4 →
    total_cost items quantities = 59 →
    quantities.spam_cans = 4 :=
by sorry

end granger_spam_cans_l231_23176


namespace food_expense_percentage_l231_23199

/-- Proves that the percentage of salary spent on food is 32% given the specified conditions --/
theorem food_expense_percentage
  (salary : ℝ)
  (medicine_percentage : ℝ)
  (savings_percentage : ℝ)
  (savings_amount : ℝ)
  (h1 : salary = 15000)
  (h2 : medicine_percentage = 20)
  (h3 : savings_percentage = 60)
  (h4 : savings_amount = 4320)
  (h5 : savings_amount = (salary - (medicine_percentage / 100) * salary - food_expense) * (savings_percentage / 100))
  : (food_expense / salary) * 100 = 32 := by
  sorry

end food_expense_percentage_l231_23199


namespace max_a_value_l231_23109

theorem max_a_value (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  (∀ a : ℝ, m^2 - a*m*n + 2*n^2 ≥ 0 → a ≤ 2*Real.sqrt 2) ∧
  ∃ a : ℝ, a = 2*Real.sqrt 2 ∧ m^2 - a*m*n + 2*n^2 ≥ 0 :=
sorry

end max_a_value_l231_23109


namespace augmented_matrix_sum_l231_23155

/-- Given a system of linear equations represented by an augmented matrix,
    prove that the sum of certain parameters equals 3. -/
theorem augmented_matrix_sum (m n : ℝ) : 
  (∃ (x y : ℝ), 2 * x = m ∧ n * x + y = 2 ∧ x = 1 ∧ y = 1) →
  m + n = 3 := by
  sorry

end augmented_matrix_sum_l231_23155


namespace two_propositions_correct_l231_23166

-- Define the original proposition
def original (x : ℝ) : Prop := x = 3 → x^2 - 7*x + 12 = 0

-- Define the converse proposition
def converse (x : ℝ) : Prop := x^2 - 7*x + 12 = 0 → x = 3

-- Define the inverse proposition
def inverse (x : ℝ) : Prop := x ≠ 3 → x^2 - 7*x + 12 ≠ 0

-- Define the contrapositive proposition
def contrapositive (x : ℝ) : Prop := x^2 - 7*x + 12 ≠ 0 → x ≠ 3

-- Theorem stating that exactly two propositions are correct
theorem two_propositions_correct :
  (∃! (n : ℕ), n = 2 ∧
    (∀ (x : ℝ), original x) ∧
    (∀ (x : ℝ), contrapositive x) ∧
    ¬(∀ (x : ℝ), converse x) ∧
    ¬(∀ (x : ℝ), inverse x)) :=
sorry

end two_propositions_correct_l231_23166


namespace sector_min_perimeter_l231_23118

theorem sector_min_perimeter (r l : ℝ) (h : r > 0) (h' : l > 0) : 
  (1/2 * l * r = 4) → (l + 2*r ≥ 8) := by
  sorry

end sector_min_perimeter_l231_23118


namespace specific_normal_distribution_mean_l231_23168

/-- A normal distribution with given properties -/
structure NormalDistribution where
  μ : ℝ  -- arithmetic mean
  σ : ℝ  -- standard deviation
  value_2sd_below : ℝ  -- value 2 standard deviations below the mean

/-- Theorem stating the properties of the specific normal distribution -/
theorem specific_normal_distribution_mean 
  (d : NormalDistribution) 
  (h1 : d.σ = 2.3)
  (h2 : d.value_2sd_below = 11.6)
  (h3 : d.value_2sd_below = d.μ - 2 * d.σ) : 
  d.μ = 16.2 := by
  sorry

end specific_normal_distribution_mean_l231_23168


namespace cu_cn2_formation_l231_23100

-- Define the chemical species
inductive Species
| HCN
| CuSO4
| CuCN2
| H2SO4

-- Define a type for chemical reactions
structure Reaction where
  reactants : List (Species × ℕ)
  products : List (Species × ℕ)

-- Define the balanced equation
def balancedEquation : Reaction :=
  { reactants := [(Species.HCN, 2), (Species.CuSO4, 1)]
  , products := [(Species.CuCN2, 1), (Species.H2SO4, 1)] }

-- Define the initial amounts of reactants
def initialHCN : ℕ := 2
def initialCuSO4 : ℕ := 1

-- Theorem statement
theorem cu_cn2_formation
  (reaction : Reaction)
  (hreaction : reaction = balancedEquation)
  (hHCN : initialHCN = 2)
  (hCuSO4 : initialCuSO4 = 1) :
  ∃ (amount : ℕ), amount = 1 ∧ 
  (Species.CuCN2, amount) ∈ reaction.products :=
sorry

end cu_cn2_formation_l231_23100


namespace green_shirt_pairs_l231_23174

theorem green_shirt_pairs 
  (total_students : ℕ) 
  (blue_shirts : ℕ) 
  (yellow_shirts : ℕ) 
  (green_shirts : ℕ) 
  (total_pairs : ℕ) 
  (blue_blue_pairs : ℕ) :
  total_students = 200 →
  blue_shirts = 70 →
  yellow_shirts = 80 →
  green_shirts = 50 →
  total_pairs = 100 →
  blue_blue_pairs = 30 →
  total_students = blue_shirts + yellow_shirts + green_shirts →
  ∃ (green_green_pairs : ℕ), green_green_pairs = 25 := by
  sorry

end green_shirt_pairs_l231_23174


namespace polyhedron_diagonals_l231_23146

/-- A polyhedron with the given properties -/
structure Polyhedron :=
  (num_vertices : ℕ)
  (edges_per_vertex : ℕ)

/-- The number of interior diagonals in a polyhedron -/
def interior_diagonals (p : Polyhedron) : ℕ :=
  (p.num_vertices * (p.num_vertices - 1 - p.edges_per_vertex)) / 2

/-- Theorem: A polyhedron with 15 vertices and 6 edges per vertex has 60 interior diagonals -/
theorem polyhedron_diagonals :
  ∀ (p : Polyhedron), p.num_vertices = 15 ∧ p.edges_per_vertex = 6 →
  interior_diagonals p = 60 :=
by sorry

end polyhedron_diagonals_l231_23146


namespace product_of_squared_fractions_l231_23126

theorem product_of_squared_fractions : (1/3 * 9)^2 * (1/27 * 81)^2 * (1/243 * 729)^2 = 729 := by
  sorry

end product_of_squared_fractions_l231_23126


namespace trapezoid_area_l231_23161

-- Define the rectangle ABCD
structure Rectangle :=
  (AB : ℝ)
  (AD : ℝ)

-- Define the circle
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

-- Define the trapezoid AFCB
structure Trapezoid :=
  (AB : ℝ)
  (FC : ℝ)
  (AD : ℝ)

-- Define the problem setup
def setup (rect : Rectangle) (circ : Circle) : Prop :=
  rect.AB = 32 ∧
  rect.AD = 40 ∧
  -- Circle is tangent to AB and AD
  circ.radius ≤ min rect.AB rect.AD ∧
  -- E is on BC and BE = 1
  1 ≤ rect.AB - circ.radius

-- Theorem statement
theorem trapezoid_area 
  (rect : Rectangle) 
  (circ : Circle) 
  (trap : Trapezoid) 
  (h : setup rect circ) :
  trap.AB = rect.AB ∧ 
  trap.AD = rect.AD ∧
  trap.FC = 27 → 
  (trap.AB + trap.FC) * trap.AD / 2 = 1180 :=
sorry

end trapezoid_area_l231_23161


namespace simplify_fraction_l231_23170

theorem simplify_fraction : 8 * (15 / 4) * (-45 / 50) = -12 / 25 := by
  sorry

end simplify_fraction_l231_23170


namespace base_prime_1260_l231_23141

/-- Base prime representation of a number -/
def BasePrimeRepresentation (n : ℕ) : List ℕ :=
  sorry

/-- Prime factorization of 1260 -/
def PrimeFactorization1260 : List (ℕ × ℕ) :=
  [(2, 2), (3, 2), (5, 1), (7, 1)]

/-- Theorem: The base prime representation of 1260 is [2, 2, 1, 2] -/
theorem base_prime_1260 : 
  BasePrimeRepresentation 1260 = [2, 2, 1, 2] := by
  sorry

end base_prime_1260_l231_23141


namespace replacement_paint_intensity_l231_23142

theorem replacement_paint_intensity
  (original_intensity : ℝ)
  (new_mixture_intensity : ℝ)
  (fraction_replaced : ℝ)
  (replacement_intensity : ℝ)
  (h1 : original_intensity = 50)
  (h2 : new_mixture_intensity = 40)
  (h3 : fraction_replaced = 1 / 3)
  (h4 : (1 - fraction_replaced) * original_intensity + fraction_replaced * replacement_intensity = new_mixture_intensity) :
  replacement_intensity = 20 := by
sorry

end replacement_paint_intensity_l231_23142


namespace problem_solution_l231_23102

/-- Represents the box of electronic products -/
structure Box where
  total : ℕ
  first_class : ℕ
  second_class : ℕ

/-- The probability of drawing a first-class product only on the third draw without replacement -/
def prob_first_class_third_draw (b : Box) : ℚ :=
  (b.second_class : ℚ) / b.total *
  ((b.second_class - 1) : ℚ) / (b.total - 1) *
  (b.first_class : ℚ) / (b.total - 2)

/-- The expected number of first-class products in n draws with replacement -/
def expected_first_class (b : Box) (n : ℕ) : ℚ :=
  (n : ℚ) * (b.first_class : ℚ) / b.total

/-- The box described in the problem -/
def problem_box : Box := { total := 5, first_class := 3, second_class := 2 }

theorem problem_solution :
  prob_first_class_third_draw problem_box = 1 / 10 ∧
  expected_first_class problem_box 10 = 6 := by
  sorry

#eval prob_first_class_third_draw problem_box
#eval expected_first_class problem_box 10

end problem_solution_l231_23102


namespace prime_factorization_sum_l231_23164

theorem prime_factorization_sum (a b c : ℕ+) : 
  2^(a : ℕ) * 3^(b : ℕ) * 5^(c : ℕ) = 36000 → 3*(a : ℕ) + 4*(b : ℕ) + 6*(c : ℕ) = 41 := by
  sorry

end prime_factorization_sum_l231_23164


namespace junior_rabbit_toys_l231_23101

def toys_per_rabbit (num_rabbits : ℕ) (monday_toys : ℕ) : ℕ :=
  let wednesday_toys := 2 * monday_toys
  let friday_toys := 4 * monday_toys
  let saturday_toys := wednesday_toys / 2
  let total_toys := monday_toys + wednesday_toys + friday_toys + saturday_toys
  total_toys / num_rabbits

theorem junior_rabbit_toys :
  toys_per_rabbit 16 6 = 3 := by
  sorry

end junior_rabbit_toys_l231_23101


namespace waitress_income_fraction_l231_23137

theorem waitress_income_fraction (S : ℚ) : 
  let first_week_salary := S
  let first_week_tips := (11 / 4) * S
  let second_week_salary := (5 / 4) * S
  let second_week_tips := (7 / 3) * second_week_salary
  let total_salary := first_week_salary + second_week_salary
  let total_tips := first_week_tips + second_week_tips
  let total_income := total_salary + total_tips
  (total_tips / total_income) = 68 / 95 := by
  sorry

end waitress_income_fraction_l231_23137


namespace quarter_circle_area_l231_23193

/-- The area of a quarter circle with radius 2 is equal to π -/
theorem quarter_circle_area : 
  let r : Real := 2
  let circle_area : Real := π * r^2
  let quarter_circle_area : Real := circle_area / 4
  quarter_circle_area = π := by
  sorry

end quarter_circle_area_l231_23193


namespace triangle_inequality_equality_condition_l231_23114

theorem triangle_inequality (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) 
  (h4 : x^2 + y^2 = z^2) : 
  x^2 * (y + z) + y^2 * (z + x) + z^2 * (x + y) ≤ (2 + 3 * Real.sqrt 2) * x * y * z :=
sorry

theorem equality_condition (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) 
  (h4 : x^2 + y^2 = z^2) : 
  x^2 * (y + z) + y^2 * (z + x) + z^2 * (x + y) = (2 + 3 * Real.sqrt 2) * x * y * z ↔ x = y :=
sorry

end triangle_inequality_equality_condition_l231_23114


namespace unique_n_satisfying_conditions_l231_23134

theorem unique_n_satisfying_conditions : ∃! n : ℤ,
  50 < n ∧ n < 120 ∧
  n % 8 = 0 ∧
  n % 9 = 5 ∧
  n % 7 = 3 ∧
  n = 104 := by
  sorry

end unique_n_satisfying_conditions_l231_23134


namespace largest_triangle_perimeter_l231_23172

/-- The largest perimeter of a triangle with two sides of 7 and 8 units, and the third side being an integer --/
theorem largest_triangle_perimeter : 
  ∀ x : ℤ, 
  (7 : ℝ) + 8 > x ∧ 
  (7 : ℝ) + x > 8 ∧ 
  8 + x > 7 →
  (∀ y : ℤ, 
    (7 : ℝ) + 8 > y ∧ 
    (7 : ℝ) + y > 8 ∧ 
    8 + y > 7 →
    7 + 8 + x ≥ 7 + 8 + y) →
  7 + 8 + x = 29 :=
by sorry

end largest_triangle_perimeter_l231_23172


namespace no_solution_iff_k_eq_18_div_5_l231_23115

/-- The equation has no solutions if and only if k = 18/5 -/
theorem no_solution_iff_k_eq_18_div_5 :
  let v1 : Fin 2 → ℝ := ![1, 3]
  let v2 : Fin 2 → ℝ := ![5, -9]
  let v3 : Fin 2 → ℝ := ![4, 0]
  let v4 : Fin 2 → ℝ := ![-2, k]
  (∀ t s : ℝ, v1 + t • v2 ≠ v3 + s • v4) ↔ k = 18/5 := by
sorry

end no_solution_iff_k_eq_18_div_5_l231_23115


namespace polynomial_division_remainder_l231_23108

theorem polynomial_division_remainder : ∃ (q r : Polynomial ℝ), 
  (X^3 + 2*X^2 - 3 : Polynomial ℝ) = (X^2 + 2) * q + r ∧ 
  r.degree < (X^2 + 2).degree ∧
  r = -2*X - 7 :=
sorry

end polynomial_division_remainder_l231_23108


namespace coupons_used_proof_l231_23104

/-- Calculates the total number of coupons used in a store's promotion --/
def total_coupons_used (initial_stock : ℝ) (books_sold : ℝ) (coupons_per_book : ℝ) : ℝ :=
  (initial_stock - books_sold) * coupons_per_book

/-- Proves that the total number of coupons used is 80.0 --/
theorem coupons_used_proof (initial_stock : ℝ) (books_sold : ℝ) (coupons_per_book : ℝ)
  (h1 : initial_stock = 40.0)
  (h2 : books_sold = 20.0)
  (h3 : coupons_per_book = 4.0) :
  total_coupons_used initial_stock books_sold coupons_per_book = 80.0 := by
  sorry

end coupons_used_proof_l231_23104


namespace add_45_minutes_to_10_20_l231_23163

/-- Represents a time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  valid : hours < 24 ∧ minutes < 60

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : Nat) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + m
  let newHours := (totalMinutes / 60) % 24
  let newMinutes := totalMinutes % 60
  ⟨newHours, newMinutes, sorry⟩

theorem add_45_minutes_to_10_20 :
  addMinutes ⟨10, 20, sorry⟩ 45 = ⟨11, 5, sorry⟩ := by
  sorry

end add_45_minutes_to_10_20_l231_23163


namespace john_finishes_at_305_l231_23191

/-- Represents time in minutes since midnight -/
def Time := ℕ

/-- Converts hours and minutes to Time -/
def toTime (hours minutes : ℕ) : Time :=
  hours * 60 + minutes

/-- The time John starts working -/
def startTime : Time := toTime 9 0

/-- The time John finishes the fourth task -/
def fourthTaskEndTime : Time := toTime 13 0

/-- The number of tasks John completes -/
def totalTasks : ℕ := 6

/-- The number of tasks completed before the first break -/
def tasksBeforeBreak : ℕ := 1

/-- The duration of each break in minutes -/
def breakDuration : ℕ := 10

/-- Calculates the time John finishes all tasks -/
noncomputable def calculateEndTime : Time := sorry

theorem john_finishes_at_305 :
  calculateEndTime = toTime 15 5 := by sorry

end john_finishes_at_305_l231_23191


namespace team_total_catch_l231_23180

/-- Represents the number of days in the fishing competition -/
def competition_days : ℕ := 5

/-- Represents Jackson's daily catch -/
def jackson_daily_catch : ℕ := 6

/-- Represents Jonah's daily catch -/
def jonah_daily_catch : ℕ := 4

/-- Represents George's daily catch -/
def george_daily_catch : ℕ := 8

/-- Theorem stating the total catch of the team during the competition -/
theorem team_total_catch : 
  competition_days * (jackson_daily_catch + jonah_daily_catch + george_daily_catch) = 90 := by
  sorry

end team_total_catch_l231_23180


namespace conversion_theorem_l231_23190

-- Define conversion rates
def meters_per_km : ℝ := 1000
def minutes_per_hour : ℝ := 60

-- Problem 1: Convert 70 kilometers and 50 meters to kilometers
def problem1 (km : ℝ) (m : ℝ) : Prop :=
  km + m / meters_per_km = 70.05

-- Problem 2: Convert 3.6 hours to hours and minutes
def problem2 (h : ℝ) : Prop :=
  ∃ (whole_hours : ℕ) (minutes : ℕ),
    h = whole_hours + (minutes : ℝ) / minutes_per_hour ∧
    whole_hours = 3 ∧
    minutes = 36

theorem conversion_theorem :
  problem1 70 50 ∧ problem2 3.6 := by sorry

end conversion_theorem_l231_23190


namespace sum_of_three_circles_l231_23135

-- Define the values for triangles and circles
variable (triangle : ℝ)
variable (circle : ℝ)

-- Define the conditions
axiom condition1 : 3 * triangle + 2 * circle = 21
axiom condition2 : 2 * triangle + 3 * circle = 19

-- Theorem to prove
theorem sum_of_three_circles : 3 * circle = 9 := by
  sorry

end sum_of_three_circles_l231_23135


namespace fraction_simplification_l231_23107

theorem fraction_simplification (a b m : ℝ) (hb : b ≠ 0) (hm : m ≠ 0) :
  (a * m) / (b * m) = a / b :=
by sorry

end fraction_simplification_l231_23107


namespace evaluate_expression_l231_23106

theorem evaluate_expression (x y z : ℚ) (hx : x = 1/4) (hy : y = 1/3) (hz : z = -6) :
  x^2 * y^3 * z^2 = 1/12 := by
  sorry

end evaluate_expression_l231_23106


namespace factors_of_1728_l231_23198

def number_of_factors (n : ℕ) : ℕ := (Nat.divisors n).card

theorem factors_of_1728 : number_of_factors 1728 = 28 := by
  sorry

end factors_of_1728_l231_23198


namespace incorrect_addition_theorem_l231_23121

/-- Represents a 6-digit number as a list of digits -/
def SixDigitNumber := List Nat

/-- Checks if a number is a valid 6-digit number -/
def isValidSixDigitNumber (n : SixDigitNumber) : Prop :=
  n.length = 6 ∧ n.all (λ d => d < 10)

/-- Converts a 6-digit number to its integer value -/
def toInt (n : SixDigitNumber) : Nat :=
  n.foldl (λ acc d => acc * 10 + d) 0

/-- Replaces all occurrences of one digit with another in a number -/
def replaceDigit (n : SixDigitNumber) (d e : Nat) : SixDigitNumber :=
  n.map (λ x => if x = d then e else x)

theorem incorrect_addition_theorem :
  ∃ (A B : SixDigitNumber) (d e : Nat),
    isValidSixDigitNumber A ∧
    isValidSixDigitNumber B ∧
    d < 10 ∧
    e < 10 ∧
    toInt A + toInt B ≠ 1061835 ∧
    toInt (replaceDigit A d e) + toInt (replaceDigit B d e) = 1061835 ∧
    d + e = 1 :=
  sorry

end incorrect_addition_theorem_l231_23121
