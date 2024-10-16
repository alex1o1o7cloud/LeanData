import Mathlib

namespace NUMINAMATH_CALUDE_x_plus_p_equals_2p_plus_3_l417_41771

theorem x_plus_p_equals_2p_plus_3 (x p : ℝ) (h1 : |x - 3| = p) (h2 : x > 3) :
  x + p = 2*p + 3 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_p_equals_2p_plus_3_l417_41771


namespace NUMINAMATH_CALUDE_cosine_identity_l417_41765

theorem cosine_identity (x : ℝ) (h1 : x ∈ Set.Ioo 0 π) (h2 : Real.cos (x - π/6) = -Real.sqrt 3 / 3) :
  Real.cos (x - π/3) = (-3 + Real.sqrt 6) / 6 := by
  sorry

end NUMINAMATH_CALUDE_cosine_identity_l417_41765


namespace NUMINAMATH_CALUDE_q_join_time_l417_41726

/-- Represents the number of months after which Q joined the business --/
def x : ℕ := sorry

/-- P's initial investment --/
def p_investment : ℕ := 4000

/-- Q's investment --/
def q_investment : ℕ := 9000

/-- Total number of months in a year --/
def total_months : ℕ := 12

/-- Ratio of P's profit share to Q's profit share --/
def profit_ratio : ℚ := 2 / 3

theorem q_join_time :
  (p_investment * total_months) / (q_investment * (total_months - x)) = profit_ratio →
  x = 4 := by sorry

end NUMINAMATH_CALUDE_q_join_time_l417_41726


namespace NUMINAMATH_CALUDE_odometer_problem_l417_41768

theorem odometer_problem (a b c : ℕ) : 
  a < 10 → b < 10 → c < 10 →  -- Digits are less than 10
  a ≠ b → b ≠ c → a ≠ c →     -- Digits are distinct
  a + b + c ≤ 9 →             -- Sum of digits is at most 9
  (100 * c + 10 * b + a) - (100 * a + 10 * b + c) % 60 = 0 → -- Difference divisible by 60
  a^2 + b^2 + c^2 = 35 :=
by sorry

end NUMINAMATH_CALUDE_odometer_problem_l417_41768


namespace NUMINAMATH_CALUDE_certain_number_problem_l417_41736

theorem certain_number_problem (x y : ℝ) (hx : x = 4) (hy : y + y * x = 48) : y = 9.6 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l417_41736


namespace NUMINAMATH_CALUDE_jessica_quarters_l417_41702

/-- The number of quarters Jessica has after receiving quarters from her sister and friend. -/
def total_quarters (initial : ℕ) (from_sister : ℕ) (from_friend : ℕ) : ℕ :=
  initial + from_sister + from_friend

/-- Theorem stating that Jessica's total quarters is 16 given the initial amount and gifts. -/
theorem jessica_quarters : total_quarters 8 3 5 = 16 := by
  sorry

end NUMINAMATH_CALUDE_jessica_quarters_l417_41702


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l417_41721

theorem polynomial_evaluation :
  ∀ y : ℝ, y > 0 → y^2 - 3*y - 9 = 0 → y^3 - 3*y^2 - 9*y + 7 = 7 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l417_41721


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l417_41774

theorem repeating_decimal_sum : 
  (1 : ℚ) / 9 + (2 : ℚ) / 99 + (2 : ℚ) / 333 = (503 : ℚ) / 3663 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l417_41774


namespace NUMINAMATH_CALUDE_sixteen_triangles_l417_41787

/-- A right triangle with integer leg lengths and hypotenuse b + 3 --/
structure RightTriangle where
  a : ℕ
  b : ℕ
  hyp_eq : a^2 + b^2 = (b + 3)^2
  b_bound : b < 200

/-- The count of right triangles satisfying the given conditions --/
def count_triangles : ℕ := sorry

/-- Theorem stating that there are exactly 16 right triangles satisfying the conditions --/
theorem sixteen_triangles : count_triangles = 16 := by sorry

end NUMINAMATH_CALUDE_sixteen_triangles_l417_41787


namespace NUMINAMATH_CALUDE_min_value_on_circle_equality_condition_l417_41747

theorem min_value_on_circle (x y : ℝ) : 
  x^2 + y^2 = 2 → (1 / (1 + x^2) + 1 / (1 + y^2)) ≥ 1 :=
by sorry

theorem equality_condition (x y : ℝ) :
  x^2 + y^2 = 2 → 
  (1 / (1 + x^2) + 1 / (1 + y^2) = 1 ↔ x^2 = 1 ∧ y^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_on_circle_equality_condition_l417_41747


namespace NUMINAMATH_CALUDE_arithmetic_has_three_term_correlation_geometric_has_three_term_correlation_l417_41794

def has_three_term_correlation (a : ℕ → ℝ) : Prop :=
  ∃ A B : ℝ, A * B ≠ 0 ∧ ∀ n : ℕ, a (n + 2) = A * a (n + 1) + B * a n

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem arithmetic_has_three_term_correlation :
  ∀ a : ℕ → ℝ, arithmetic_sequence a → has_three_term_correlation a :=
sorry

theorem geometric_has_three_term_correlation :
  ∀ a : ℕ → ℝ, geometric_sequence a → has_three_term_correlation a :=
sorry

end NUMINAMATH_CALUDE_arithmetic_has_three_term_correlation_geometric_has_three_term_correlation_l417_41794


namespace NUMINAMATH_CALUDE_min_value_of_m_l417_41703

theorem min_value_of_m (a b c m : ℝ) (h1 : a > b) (h2 : b > c) (h3 : m > 0)
  (h4 : ∀ a b c, a > b ∧ b > c → 1 / (a - b) + m / (b - c) ≥ 9 / (a - c)) :
  m ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_m_l417_41703


namespace NUMINAMATH_CALUDE_point_on_x_axis_l417_41777

theorem point_on_x_axis (m : ℝ) : (3, m) ∈ {p : ℝ × ℝ | p.2 = 0} → m = 0 := by
  sorry

end NUMINAMATH_CALUDE_point_on_x_axis_l417_41777


namespace NUMINAMATH_CALUDE_complex_exp_conversion_l417_41720

theorem complex_exp_conversion (z : ℂ) :
  z = Real.sqrt 2 * Complex.exp (13 * Real.pi * Complex.I / 4) →
  z = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_exp_conversion_l417_41720


namespace NUMINAMATH_CALUDE_unique_arrangement_l417_41705

/-- Represents a 4x4 grid with letters A, B, and C --/
def Grid := Fin 4 → Fin 4 → Char

/-- Checks if a given grid satisfies the arrangement conditions --/
def valid_arrangement (g : Grid) : Prop :=
  -- A is in the upper left corner
  g 0 0 = 'A' ∧
  -- Each row contains one of each letter
  (∀ i : Fin 4, ∃ j₁ j₂ j₃ : Fin 4, g i j₁ = 'A' ∧ g i j₂ = 'B' ∧ g i j₃ = 'C') ∧
  -- Each column contains one of each letter
  (∀ j : Fin 4, ∃ i₁ i₂ i₃ : Fin 4, g i₁ j = 'A' ∧ g i₂ j = 'B' ∧ g i₃ j = 'C') ∧
  -- Main diagonal (top-left to bottom-right) contains one of each letter
  (∃ i₁ i₂ i₃ : Fin 4, g i₁ i₁ = 'A' ∧ g i₂ i₂ = 'B' ∧ g i₃ i₃ = 'C') ∧
  -- Anti-diagonal (top-right to bottom-left) contains one of each letter
  (∃ i₁ i₂ i₃ : Fin 4, g i₁ (3 - i₁) = 'A' ∧ g i₂ (3 - i₂) = 'B' ∧ g i₃ (3 - i₃) = 'C')

/-- The main theorem stating there is only one valid arrangement --/
theorem unique_arrangement : ∃! g : Grid, valid_arrangement g :=
  sorry

end NUMINAMATH_CALUDE_unique_arrangement_l417_41705


namespace NUMINAMATH_CALUDE_triangle_properties_l417_41701

/-- Given a triangle ABC with circumradius R and satisfying the given equation,
    prove that angle C is π/3 and the maximum area is 3√3/2 -/
theorem triangle_properties (A B C : Real) (a b c : Real) (R : Real) :
  R = Real.sqrt 2 →
  2 * Real.sqrt 2 * (Real.sin A ^ 2 - Real.sin C ^ 2) = (a - b) * Real.sin B →
  (C = Real.pi / 3 ∧ 
   ∃ (S : Real), S = 3 * Real.sqrt 3 / 2 ∧ 
   ∀ (S' : Real), S' = 1/2 * a * b * Real.sin C → S' ≤ S) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l417_41701


namespace NUMINAMATH_CALUDE_total_money_l417_41725

theorem total_money (a b : ℝ) (h1 : (4/15) * a = (2/5) * b) (h2 : b = 484) :
  a + b = 1210 := by
  sorry

end NUMINAMATH_CALUDE_total_money_l417_41725


namespace NUMINAMATH_CALUDE_function_properties_l417_41744

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sin x ^ 2 + a * Real.sin x * Real.cos x - Real.cos x ^ 2

theorem function_properties :
  ∃ (a : ℝ), 
    f a (π / 4) = 1 ∧ 
    (∀ x : ℝ, f a x = f a (x + π)) ∧
    (∀ x : ℝ, f a x ≥ -Real.sqrt 2) ∧
    (∃ x : ℝ, f a x = -Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l417_41744


namespace NUMINAMATH_CALUDE_expression_range_l417_41741

theorem expression_range (x y : ℝ) (h : x^2 + (y - 2)^2 ≤ 1) :
  1 ≤ (x + Real.sqrt 3 * y) / Real.sqrt (x^2 + y^2) ∧
  (x + Real.sqrt 3 * y) / Real.sqrt (x^2 + y^2) ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_expression_range_l417_41741


namespace NUMINAMATH_CALUDE_sandy_sums_l417_41782

theorem sandy_sums (total_sums : ℕ) (correct_marks : ℕ) (incorrect_marks : ℕ) (total_marks : ℤ) :
  total_sums = 40 →
  correct_marks = 4 →
  incorrect_marks = 3 →
  total_marks = 72 →
  ∃ (correct_sums : ℕ),
    correct_sums ≤ total_sums ∧
    (correct_marks * correct_sums : ℤ) - 
    (incorrect_marks * (total_sums - correct_sums) : ℤ) = total_marks ∧
    correct_sums = 27 :=
by sorry

end NUMINAMATH_CALUDE_sandy_sums_l417_41782


namespace NUMINAMATH_CALUDE_count_non_similar_triangles_l417_41760

/-- A regular decagon with all diagonals drawn -/
structure RegularDecagonWithDiagonals where
  /-- The number of vertices in a decagon -/
  num_vertices : ℕ
  /-- The central angle of a regular decagon -/
  central_angle : ℝ
  /-- The internal angle of a regular decagon -/
  internal_angle : ℝ
  /-- The smallest angle increment between diagonals -/
  diagonal_angle_increment : ℝ
  /-- Assertion that the number of vertices is 10 -/
  vertices_eq : num_vertices = 10
  /-- Assertion that the central angle is 36° -/
  central_angle_eq : central_angle = 36
  /-- Assertion that the internal angle is 144° -/
  internal_angle_eq : internal_angle = 144
  /-- Assertion that the diagonal angle increment is 18° -/
  diagonal_angle_increment_eq : diagonal_angle_increment = 18

/-- The number of pairwise non-similar triangles in a regular decagon with all diagonals drawn -/
def num_non_similar_triangles (d : RegularDecagonWithDiagonals) : ℕ := 8

/-- Theorem stating that the number of pairwise non-similar triangles in a regular decagon with all diagonals drawn is 8 -/
theorem count_non_similar_triangles (d : RegularDecagonWithDiagonals) :
  num_non_similar_triangles d = 8 := by
  sorry

end NUMINAMATH_CALUDE_count_non_similar_triangles_l417_41760


namespace NUMINAMATH_CALUDE_midpoint_coordinate_product_l417_41795

/-- The product of the coordinates of the midpoint of a line segment with endpoints (4, -1) and (-2, 7) is 3. -/
theorem midpoint_coordinate_product : 
  let x1 : ℝ := 4
  let y1 : ℝ := -1
  let x2 : ℝ := -2
  let y2 : ℝ := 7
  let midpoint_x : ℝ := (x1 + x2) / 2
  let midpoint_y : ℝ := (y1 + y2) / 2
  midpoint_x * midpoint_y = 3 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_product_l417_41795


namespace NUMINAMATH_CALUDE_candidate_vote_percentage_l417_41797

/-- Proves that given a total of 8000 votes and a loss margin of 4000 votes,
    the percentage of votes received by the losing candidate is 25%. -/
theorem candidate_vote_percentage
  (total_votes : ℕ)
  (loss_margin : ℕ)
  (h_total : total_votes = 8000)
  (h_margin : loss_margin = 4000) :
  (total_votes - loss_margin) / total_votes * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_candidate_vote_percentage_l417_41797


namespace NUMINAMATH_CALUDE_airplane_speed_proof_l417_41710

/-- Proves that the speed of one airplane is 400 mph given the conditions of the problem -/
theorem airplane_speed_proof (v : ℝ) : 
  v > 0 →  -- Assuming positive speed for the first airplane
  (2.5 * v + 2.5 * 250 = 1625) →  -- Condition from the problem
  v = 400 := by
sorry

end NUMINAMATH_CALUDE_airplane_speed_proof_l417_41710


namespace NUMINAMATH_CALUDE_johns_fee_value_l417_41724

/-- The one-time sitting fee for John's Photo World -/
def johns_fee : ℝ := sorry

/-- The price per sheet at John's Photo World -/
def johns_price_per_sheet : ℝ := 2.75

/-- The price per sheet at Sam's Picture Emporium -/
def sams_price_per_sheet : ℝ := 1.50

/-- The one-time sitting fee for Sam's Picture Emporium -/
def sams_fee : ℝ := 140

/-- The number of sheets being compared -/
def num_sheets : ℝ := 12

theorem johns_fee_value : johns_fee = 125 :=
  by
    have h : johns_price_per_sheet * num_sheets + johns_fee = sams_price_per_sheet * num_sheets + sams_fee :=
      sorry
    sorry

#check johns_fee_value

end NUMINAMATH_CALUDE_johns_fee_value_l417_41724


namespace NUMINAMATH_CALUDE_x_can_be_negative_one_l417_41751

theorem x_can_be_negative_one : ∃ (x : ℝ), x = -1 ∧ x^2 ∈ ({0, 1, x} : Set ℝ) := by
  sorry

end NUMINAMATH_CALUDE_x_can_be_negative_one_l417_41751


namespace NUMINAMATH_CALUDE_ihsan_children_l417_41722

/-- The number of children each person has (except great-great-grandchildren) -/
def n : ℕ := 7

/-- The total number of people in the family, including Ihsan -/
def total_people : ℕ := 2801

/-- Theorem stating that n satisfies the conditions of the problem -/
theorem ihsan_children :
  n + n^2 + n^3 + n^4 + 1 = total_people :=
by sorry

end NUMINAMATH_CALUDE_ihsan_children_l417_41722


namespace NUMINAMATH_CALUDE_expression_evaluation_l417_41799

theorem expression_evaluation :
  let x : ℝ := 3
  let expr := (1 / (x^2 - 2*x) - 1 / (x^2 - 4*x + 4)) / (2 / (x^2 - 2*x))
  expr = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l417_41799


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l417_41714

/-- The quadratic function f(x) = x^2 + 2x + m -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + 2*x + m

/-- f has a root -/
def has_root (m : ℝ) : Prop := ∃ x, f m x = 0

/-- m < 1 is sufficient but not necessary for f to have a root -/
theorem sufficient_not_necessary :
  (∀ m, m < 1 → has_root m) ∧ 
  (∃ m, ¬(m < 1) ∧ has_root m) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l417_41714


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l417_41735

theorem polynomial_division_remainder : ∃ q : Polynomial ℤ, 
  3 * X^3 - 4 * X^2 - 23 * X + 60 = (X - 3) * q + 36 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l417_41735


namespace NUMINAMATH_CALUDE_diamond_ratio_equals_three_fifths_l417_41784

-- Define the ♢ operation
def diamond (n m : ℕ) : ℕ := n^2 * m^3

-- Theorem statement
theorem diamond_ratio_equals_three_fifths :
  (diamond 5 3) / (diamond 3 5 : ℚ) = 3/5 := by sorry

end NUMINAMATH_CALUDE_diamond_ratio_equals_three_fifths_l417_41784


namespace NUMINAMATH_CALUDE_largest_unexpressible_l417_41742

/-- The set An for a given n -/
def An (n : ℕ) : Set ℕ :=
  {x | ∃ k : ℕ, k < n ∧ x = 2^n - 2^k}

/-- The property of being expressible as a sum of elements from An -/
def isExpressible (n : ℕ) (m : ℕ) : Prop :=
  ∃ (s : Multiset ℕ), (∀ x ∈ s, x ∈ An n) ∧ (s.sum = m)

/-- The main theorem -/
theorem largest_unexpressible (n : ℕ) (h : n ≥ 2) :
  ∀ m : ℕ, m > (n - 2) * 2^n + 1 → isExpressible n m :=
sorry

end NUMINAMATH_CALUDE_largest_unexpressible_l417_41742


namespace NUMINAMATH_CALUDE_howard_window_washing_earnings_l417_41750

theorem howard_window_washing_earnings
  (initial_amount : ℝ)
  (final_amount : ℝ)
  (cleaning_expenses : ℝ)
  (h1 : initial_amount = 26)
  (h2 : final_amount = 52)
  (h3 : final_amount = initial_amount + earnings - cleaning_expenses) :
  earnings = 26 + cleaning_expenses :=
by sorry

end NUMINAMATH_CALUDE_howard_window_washing_earnings_l417_41750


namespace NUMINAMATH_CALUDE_gcd_factorial_eight_ten_l417_41733

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem gcd_factorial_eight_ten : 
  Nat.gcd (factorial 8) (factorial 10) = factorial 8 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_eight_ten_l417_41733


namespace NUMINAMATH_CALUDE_simplify_polynomial_l417_41766

theorem simplify_polynomial (x : ℝ) : (x - 1)^4 + 4*(x - 1)^3 + 6*(x - 1)^2 + 4*(x - 1) = x^4 - 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_l417_41766


namespace NUMINAMATH_CALUDE_four_person_apartments_count_l417_41789

/-- Represents the number of 4-person apartments in each building -/
def four_person_apartments : ℕ := sorry

/-- The number of identical buildings in the complex -/
def num_buildings : ℕ := 4

/-- The number of studio apartments in each building -/
def studio_apartments : ℕ := 10

/-- The number of 2-person apartments in each building -/
def two_person_apartments : ℕ := 20

/-- The occupancy rate of the apartment complex -/
def occupancy_rate : ℚ := 3/4

/-- The total number of people living in the apartment complex -/
def total_occupants : ℕ := 210

/-- Theorem stating that the number of 4-person apartments in each building is 5 -/
theorem four_person_apartments_count : four_person_apartments = 5 :=
  by sorry

end NUMINAMATH_CALUDE_four_person_apartments_count_l417_41789


namespace NUMINAMATH_CALUDE_arithmetic_progression_common_difference_l417_41791

/-- Proves that in an arithmetic progression with first term 2, last term 62, and 31 terms, the common difference is 2. -/
theorem arithmetic_progression_common_difference 
  (first_term : ℕ) 
  (last_term : ℕ) 
  (num_terms : ℕ) 
  (h1 : first_term = 2) 
  (h2 : last_term = 62) 
  (h3 : num_terms = 31) : 
  (last_term - first_term) / (num_terms - 1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_common_difference_l417_41791


namespace NUMINAMATH_CALUDE_count_divisible_numbers_l417_41713

theorem count_divisible_numbers (n : ℕ) : 
  (Finset.filter (fun k => (k^2 - 1) % 291 = 0) (Finset.range (291000 + 1))).card = 4000 :=
sorry

end NUMINAMATH_CALUDE_count_divisible_numbers_l417_41713


namespace NUMINAMATH_CALUDE_sqrt_300_simplified_l417_41754

theorem sqrt_300_simplified : Real.sqrt 300 = 10 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_300_simplified_l417_41754


namespace NUMINAMATH_CALUDE_problem_solution_l417_41779

noncomputable section

variables (a b c x : ℝ)

def f (x : ℝ) (b : ℝ) : ℝ := |x + b^2| - |-x + 1|
def g (x a b c : ℝ) : ℝ := |x + a^2 + c^2| + |x - 2*b^2|

theorem problem_solution (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a*b + b*c + a*c = 1) :
  (∀ x, f x 1 ≥ 1 ↔ x ∈ Set.Ici (1/2)) ∧
  (∀ x, f x b ≤ g x a b c) := by sorry

end NUMINAMATH_CALUDE_problem_solution_l417_41779


namespace NUMINAMATH_CALUDE_fenced_area_blocks_l417_41709

def total_blocks : ℕ := 344
def building_blocks : ℕ := 80
def farmhouse_blocks : ℕ := 123
def remaining_blocks : ℕ := 84

theorem fenced_area_blocks :
  total_blocks - building_blocks - farmhouse_blocks - remaining_blocks = 57 := by
  sorry

end NUMINAMATH_CALUDE_fenced_area_blocks_l417_41709


namespace NUMINAMATH_CALUDE_chess_tournament_participants_l417_41757

theorem chess_tournament_participants (n : ℕ) : 
  (n * (n - 1)) / 2 = 231 → n = 22 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_participants_l417_41757


namespace NUMINAMATH_CALUDE_remaining_water_bottles_samiras_remaining_bottles_l417_41740

/-- Calculates the number of water bottles remaining after a soccer game --/
theorem remaining_water_bottles (initial_bottles : ℕ) (players : ℕ) 
  (bottles_first_break : ℕ) (bottles_end_game : ℕ) : ℕ :=
  let bottles_after_first_break := initial_bottles - players * bottles_first_break
  let final_remaining_bottles := bottles_after_first_break - players * bottles_end_game
  final_remaining_bottles

/-- Proves that given the specific conditions of Samira's soccer game, 
    15 water bottles remain --/
theorem samiras_remaining_bottles : 
  remaining_water_bottles (4 * 12) 11 2 1 = 15 := by
  sorry

end NUMINAMATH_CALUDE_remaining_water_bottles_samiras_remaining_bottles_l417_41740


namespace NUMINAMATH_CALUDE_factorial_sum_equals_35906_l417_41786

def factorial : ℕ → ℕ
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem factorial_sum_equals_35906 : 
  7 * factorial 7 + 5 * factorial 5 + 3 * factorial 3 + 2 * (factorial 2)^2 = 35906 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_equals_35906_l417_41786


namespace NUMINAMATH_CALUDE_sqrt_expression_equals_six_l417_41781

theorem sqrt_expression_equals_six :
  (Real.sqrt 3 - 1)^2 + Real.sqrt 12 + (1/2)⁻¹ = 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equals_six_l417_41781


namespace NUMINAMATH_CALUDE_remaining_budget_for_public_spaces_l417_41732

/-- Proof of remaining budget for public spaces -/
theorem remaining_budget_for_public_spaces 
  (total_budget : ℝ) 
  (education_budget : ℝ) 
  (h1 : total_budget = 32000000)
  (h2 : education_budget = 12000000) :
  total_budget - (total_budget / 2 + education_budget) = 4000000 := by
  sorry

end NUMINAMATH_CALUDE_remaining_budget_for_public_spaces_l417_41732


namespace NUMINAMATH_CALUDE_power_zero_eq_one_l417_41723

theorem power_zero_eq_one (n : ℤ) (h : n ≠ 0) : n^0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_zero_eq_one_l417_41723


namespace NUMINAMATH_CALUDE_average_b_c_l417_41773

theorem average_b_c (a b c : ℝ) 
  (h1 : (a + b) / 2 = 35) 
  (h2 : c - a = 90) : 
  (b + c) / 2 = 80 := by
sorry

end NUMINAMATH_CALUDE_average_b_c_l417_41773


namespace NUMINAMATH_CALUDE_consecutive_squares_difference_l417_41792

theorem consecutive_squares_difference (t : ℕ) : 
  (t + 1)^2 - t^2 = 191 → t^2 = 9025 :=
by
  sorry

end NUMINAMATH_CALUDE_consecutive_squares_difference_l417_41792


namespace NUMINAMATH_CALUDE_quadrilateral_sum_l417_41763

/-- Represents a point in 2D space -/
structure Point where
  x : ℤ
  y : ℤ

/-- Represents a quadrilateral PQRS -/
structure Quadrilateral where
  P : Point
  Q : Point
  R : Point
  S : Point

def area (q : Quadrilateral) : ℚ :=
  sorry  -- Area calculation implementation

theorem quadrilateral_sum (a b : ℤ) :
  a > b ∧ b > 0 →
  let q := Quadrilateral.mk
    (Point.mk (2*a) (2*b))
    (Point.mk (2*b) (2*a))
    (Point.mk (-2*a) (-2*b))
    (Point.mk (-2*b) (-2*a))
  area q = 32 →
  a + b = 4 := by
    sorry

end NUMINAMATH_CALUDE_quadrilateral_sum_l417_41763


namespace NUMINAMATH_CALUDE_sunflower_majority_on_wednesday_sunflower_proportion_increases_l417_41712

/-- Represents the amount of sunflower seeds on a given day -/
def sunflower_seeds (day : ℕ) : ℝ :=
  3 * (1 - (0.8 ^ day))

/-- Represents the total amount of seeds on any day after adding new seeds -/
def total_seeds : ℝ := 2

/-- Theorem stating that Wednesday (day 3) is the first day when sunflower seeds exceed half of total seeds -/
theorem sunflower_majority_on_wednesday :
  (∀ d < 3, sunflower_seeds d ≤ total_seeds / 2) ∧
  (sunflower_seeds 3 > total_seeds / 2) :=
by sorry

/-- Helper theorem: The proportion of sunflower seeds increases each day -/
theorem sunflower_proportion_increases (d : ℕ) :
  sunflower_seeds d < sunflower_seeds (d + 1) :=
by sorry

end NUMINAMATH_CALUDE_sunflower_majority_on_wednesday_sunflower_proportion_increases_l417_41712


namespace NUMINAMATH_CALUDE_remaining_game_price_l417_41728

def total_games : ℕ := 346
def expensive_games : ℕ := 80
def expensive_price : ℕ := 12
def mid_price : ℕ := 7
def total_spent : ℕ := 2290

theorem remaining_game_price :
  let remaining_games := total_games - expensive_games
  let mid_games := remaining_games / 2
  let cheap_games := remaining_games - mid_games
  let spent_on_expensive := expensive_games * expensive_price
  let spent_on_mid := mid_games * mid_price
  let spent_on_cheap := total_spent - spent_on_expensive - spent_on_mid
  spent_on_cheap / cheap_games = 3 := by
sorry

end NUMINAMATH_CALUDE_remaining_game_price_l417_41728


namespace NUMINAMATH_CALUDE_P_in_quadrant_III_l417_41738

-- Define the point P
def P : ℝ × ℝ := (-3, -4)

-- Define the quadrants
def in_quadrant_I (p : ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 > 0
def in_quadrant_II (p : ℝ × ℝ) : Prop := p.1 < 0 ∧ p.2 > 0
def in_quadrant_III (p : ℝ × ℝ) : Prop := p.1 < 0 ∧ p.2 < 0
def in_quadrant_IV (p : ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 < 0

-- Theorem: P lies in Quadrant III
theorem P_in_quadrant_III : in_quadrant_III P := by sorry

end NUMINAMATH_CALUDE_P_in_quadrant_III_l417_41738


namespace NUMINAMATH_CALUDE_expired_yogurt_percentage_l417_41759

theorem expired_yogurt_percentage (total_packs : ℕ) (cost_per_pack : ℚ) (refund_amount : ℚ) :
  total_packs = 80 →
  cost_per_pack = 12 →
  refund_amount = 384 →
  (refund_amount / cost_per_pack) / total_packs * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_expired_yogurt_percentage_l417_41759


namespace NUMINAMATH_CALUDE_odd_side_length_l417_41775

/-- A triangle with two known sides and an odd third side -/
structure OddTriangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  h1 : side1 = 2
  h2 : side2 = 5
  h3 : ∃ k : ℕ, side3 = 2 * k + 1

/-- The triangle inequality theorem -/
axiom triangle_inequality (t : OddTriangle) : 
  t.side1 + t.side2 > t.side3 ∧ 
  t.side1 + t.side3 > t.side2 ∧ 
  t.side2 + t.side3 > t.side1

theorem odd_side_length (t : OddTriangle) : t.side3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_odd_side_length_l417_41775


namespace NUMINAMATH_CALUDE_largest_number_of_circles_l417_41739

/-- Given a convex quadrilateral BCDE in the plane where lines EB and DC intersect at A,
    this theorem proves that the largest number of nonoverlapping circles that can lie in
    BCDE and are tangent to both BE and CD is 5, given the specified conditions. -/
theorem largest_number_of_circles
  (AB : ℝ) (AC : ℝ) (AD : ℝ) (AE : ℝ) (cos_BAC : ℝ)
  (h_AB : AB = 2)
  (h_AC : AC = 5)
  (h_AD : AD = 200)
  (h_AE : AE = 500)
  (h_cos_BAC : cos_BAC = 7/9)
  : ℕ :=
5

#check largest_number_of_circles

end NUMINAMATH_CALUDE_largest_number_of_circles_l417_41739


namespace NUMINAMATH_CALUDE_system_of_equations_sum_l417_41790

theorem system_of_equations_sum (a b c x y z : ℝ) 
  (eq1 : 17 * x + b * y + c * z = 0)
  (eq2 : a * x + 29 * y + c * z = 0)
  (eq3 : a * x + b * y + 53 * z = 0)
  (ha : a ≠ 17)
  (hx : x ≠ 0) : 
  a / (a - 17) + b / (b - 29) + c / (c - 53) = 1 := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_sum_l417_41790


namespace NUMINAMATH_CALUDE_find_Y_l417_41718

theorem find_Y : ∃ Y : ℝ, (100 + Y / 90) * 90 = 9020 ∧ Y = 20 := by
  sorry

end NUMINAMATH_CALUDE_find_Y_l417_41718


namespace NUMINAMATH_CALUDE_inequality_proof_l417_41717

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : b > c) : 
  (1 / (a - b)) + (1 / (b - c)) + (1 / (c - a)) > 0 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l417_41717


namespace NUMINAMATH_CALUDE_nearest_fraction_sum_l417_41706

theorem nearest_fraction_sum : ∃ (x : ℕ), 
  (2007 : ℝ) / 2999 + (8001 : ℝ) / x + (2001 : ℝ) / 3999 = 3.0035428163476343 ∧ 
  x = 4362 := by
sorry

end NUMINAMATH_CALUDE_nearest_fraction_sum_l417_41706


namespace NUMINAMATH_CALUDE_perimeter_stones_count_l417_41762

/-- Given a square arrangement of stones with 5 stones on each side,
    the number of stones on the perimeter is 16. -/
theorem perimeter_stones_count (side_length : ℕ) (h : side_length = 5) :
  4 * side_length - 4 = 16 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_stones_count_l417_41762


namespace NUMINAMATH_CALUDE_mary_turnips_count_l417_41708

/-- The number of turnips grown by Sally -/
def sally_turnips : ℕ := 113

/-- The total number of turnips grown by Sally and Mary -/
def total_turnips : ℕ := 242

/-- The number of turnips grown by Mary -/
def mary_turnips : ℕ := total_turnips - sally_turnips

theorem mary_turnips_count : mary_turnips = 129 := by
  sorry

end NUMINAMATH_CALUDE_mary_turnips_count_l417_41708


namespace NUMINAMATH_CALUDE_vectors_orthogonal_l417_41719

-- Define the vectors
def v1 : Fin 2 → ℝ := ![3, 7]
def v2 (x : ℝ) : Fin 2 → ℝ := ![x, -4]

-- Define orthogonality condition
def isOrthogonal (u v : Fin 2 → ℝ) : Prop :=
  (u 0) * (v 0) + (u 1) * (v 1) = 0

-- State the theorem
theorem vectors_orthogonal :
  isOrthogonal v1 (v2 (28/3)) := by sorry

end NUMINAMATH_CALUDE_vectors_orthogonal_l417_41719


namespace NUMINAMATH_CALUDE_min_value_of_function_l417_41704

theorem min_value_of_function (a : ℝ) (h : a > 1) :
  ∃ (min_val : ℝ), min_val = 3 + 2 * Real.sqrt 2 ∧
  ∀ x > 1, x + x^2 / (x - 1) ≥ min_val ∧
  ∃ y > 1, y + y^2 / (y - 1) = min_val :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_function_l417_41704


namespace NUMINAMATH_CALUDE_intersection_point_l417_41776

-- Define the three lines
def line1 (x y : ℝ) : Prop := y = 2 * x - 1
def line2 (x y : ℝ) : Prop := y = -3 * x + 4
def line3 (x y m : ℝ) : Prop := y = 4 * x + m

-- Theorem statement
theorem intersection_point (m : ℝ) : 
  (∃ x y : ℝ, line1 x y ∧ line2 x y ∧ line3 x y m) → m = -3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_l417_41776


namespace NUMINAMATH_CALUDE_area_of_triangle_PQR_l417_41727

/-- Given two lines intersecting at P(2,5) with slopes 3 and 1 respectively,
    and Q and R as the intersections of these lines with the x-axis,
    prove that the area of triangle PQR is 25/3 -/
theorem area_of_triangle_PQR (P Q R : ℝ × ℝ) : 
  P = (2, 5) →
  (∃ m₁ m₂ : ℝ, m₁ = 3 ∧ m₂ = 1 ∧ 
    (∀ x y : ℝ, y - 5 = m₁ * (x - 2) ∨ y - 5 = m₂ * (x - 2))) →
  Q.2 = 0 ∧ R.2 = 0 →
  (∃ x₁ x₂ : ℝ, Q = (x₁, 0) ∧ R = (x₂, 0) ∧ 
    (5 - 0) = 3 * (2 - x₁) ∧ (5 - 0) = 1 * (2 - x₂)) →
  (1/2 : ℝ) * |Q.1 - R.1| * 5 = 25/3 :=
by sorry

end NUMINAMATH_CALUDE_area_of_triangle_PQR_l417_41727


namespace NUMINAMATH_CALUDE_unique_number_with_properties_l417_41793

/-- Given a natural number n, returns the sum of its digits. -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Given a two-digit number n, returns the number formed by reversing its digits. -/
def reverse_digits (n : ℕ) : ℕ := sorry

/-- Predicate that checks if a number is two-digit. -/
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

theorem unique_number_with_properties : 
  ∃! n : ℕ, is_two_digit n ∧ 
    n = 4 * (sum_of_digits n) + 3 ∧ 
    n + 18 = reverse_digits n := by sorry

end NUMINAMATH_CALUDE_unique_number_with_properties_l417_41793


namespace NUMINAMATH_CALUDE_rob_unique_cards_rob_doubles_ratio_jess_doubles_ratio_alex_doubles_ratio_l417_41778

-- Define the friends
structure Friend where
  name : String
  total_cards : ℕ
  doubles : ℕ

-- Define the problem setup
def rob : Friend := { name := "Rob", total_cards := 24, doubles := 8 }
def jess : Friend := { name := "Jess", total_cards := 0, doubles := 40 } -- total_cards unknown
def alex : Friend := { name := "Alex", total_cards := 0, doubles := 0 } -- both unknown

-- Theorem: Rob has 16 unique cards
theorem rob_unique_cards :
  rob.total_cards - rob.doubles = 16 :=
by
  sorry

-- Conditions from the problem
theorem rob_doubles_ratio :
  3 * rob.doubles = rob.total_cards :=
by
  sorry

theorem jess_doubles_ratio :
  jess.doubles = 5 * rob.doubles :=
by
  sorry

theorem alex_doubles_ratio (alex_total : ℕ) :
  4 * alex.doubles = alex_total :=
by
  sorry

end NUMINAMATH_CALUDE_rob_unique_cards_rob_doubles_ratio_jess_doubles_ratio_alex_doubles_ratio_l417_41778


namespace NUMINAMATH_CALUDE_hearty_beads_count_l417_41785

/-- The number of packages of blue beads Hearty bought -/
def blue_packages : ℕ := 3

/-- The number of packages of red beads Hearty bought -/
def red_packages : ℕ := 5

/-- The number of beads in each package -/
def beads_per_package : ℕ := 40

/-- The total number of beads Hearty has -/
def total_beads : ℕ := blue_packages * beads_per_package + red_packages * beads_per_package

theorem hearty_beads_count : total_beads = 320 := by
  sorry

end NUMINAMATH_CALUDE_hearty_beads_count_l417_41785


namespace NUMINAMATH_CALUDE_just_passed_theorem_l417_41783

/-- Represents the three subjects in the examination -/
inductive Subject
| Mathematics
| English
| Science

/-- Represents the grading divisions -/
inductive Division
| First
| Second
| JustPassed

/-- The total number of students who took the examination -/
def total_students : ℕ := 500

/-- The percentage of students who got each division in each subject -/
def division_percentage (s : Subject) (d : Division) : ℚ :=
  match s, d with
  | Subject.Mathematics, Division.First => 35/100
  | Subject.Mathematics, Division.Second => 48/100
  | Subject.English, Division.First => 25/100
  | Subject.English, Division.Second => 60/100
  | Subject.Science, Division.First => 40/100
  | Subject.Science, Division.Second => 45/100
  | _, Division.JustPassed => 0  -- This will be calculated

/-- The number of students who just passed in each subject -/
def just_passed_count (s : Subject) : ℕ :=
  match s with
  | Subject.Mathematics => 85
  | Subject.English => 75
  | Subject.Science => 75

/-- Theorem stating the number of students who just passed in each subject -/
theorem just_passed_theorem :
  (∀ s : Subject, just_passed_count s = 
    (1 - (division_percentage s Division.First + division_percentage s Division.Second)) * total_students) :=
by sorry

end NUMINAMATH_CALUDE_just_passed_theorem_l417_41783


namespace NUMINAMATH_CALUDE_ratio_c_d_equals_two_thirds_l417_41730

theorem ratio_c_d_equals_two_thirds
  (x y c d : ℝ)
  (h1 : 8 * x - 5 * y = c)
  (h2 : 10 * y - 12 * x = d)
  (hx : x ≠ 0)
  (hy : y ≠ 0)
  (hd : d ≠ 0) :
  c / d = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_ratio_c_d_equals_two_thirds_l417_41730


namespace NUMINAMATH_CALUDE_gcd_lcm_product_48_75_l417_41767

theorem gcd_lcm_product_48_75 : Nat.gcd 48 75 * Nat.lcm 48 75 = 3600 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_48_75_l417_41767


namespace NUMINAMATH_CALUDE_translation_preserves_shape_and_size_l417_41734

-- Define a geometric figure
structure GeometricFigure where
  -- We don't need to specify the exact properties of a geometric figure for this statement
  dummy : Unit

-- Define a translation
def Translation := ℝ → ℝ → ℝ → ℝ

-- Define the concept of shape preservation
def PreservesShape (f : Translation) (fig : GeometricFigure) : Prop :=
  -- The exact definition is not provided in the problem, so we leave it abstract
  sorry

-- Define the concept of size preservation
def PreservesSize (f : Translation) (fig : GeometricFigure) : Prop :=
  -- The exact definition is not provided in the problem, so we leave it abstract
  sorry

-- Theorem statement
theorem translation_preserves_shape_and_size (f : Translation) (fig : GeometricFigure) :
  PreservesShape f fig ∧ PreservesSize f fig :=
by
  sorry

end NUMINAMATH_CALUDE_translation_preserves_shape_and_size_l417_41734


namespace NUMINAMATH_CALUDE_os_overhead_calculation_l417_41798

/-- The cost per millisecond of computer time -/
def cost_per_ms : ℚ := 23 / 1000

/-- The cost for mounting a data tape -/
def tape_cost : ℚ := 535 / 100

/-- The duration of the program run in seconds -/
def run_duration : ℚ := 3 / 2

/-- The total cost for one run of the program -/
def total_cost : ℚ := 4092 / 100

/-- The operating-system overhead cost -/
def os_overhead : ℚ := 107 / 100

theorem os_overhead_calculation :
  os_overhead = total_cost - (run_duration * 1000 * cost_per_ms + tape_cost) := by
  sorry

end NUMINAMATH_CALUDE_os_overhead_calculation_l417_41798


namespace NUMINAMATH_CALUDE_root_equation_problem_l417_41743

theorem root_equation_problem (a b m p r : ℝ) : 
  (a^2 - m*a + 4 = 0) →
  (b^2 - m*b + 4 = 0) →
  ((a - 1/b)^2 - p*(a - 1/b) + r = 0) →
  ((b - 1/a)^2 - p*(b - 1/a) + r = 0) →
  r = 9/4 := by sorry

end NUMINAMATH_CALUDE_root_equation_problem_l417_41743


namespace NUMINAMATH_CALUDE_probability_at_least_one_from_A_l417_41745

/-- Represents the number of factories in each district -/
structure DistrictFactories where
  A : ℕ
  B : ℕ
  C : ℕ

/-- Represents the number of factories sampled from each district -/
structure SampledFactories where
  A : ℕ
  B : ℕ
  C : ℕ

/-- Calculates the probability of selecting at least one factory from district A 
    when randomly choosing 2 out of 7 stratified sampled factories -/
def probabilityAtLeastOneFromA (df : DistrictFactories) (sf : SampledFactories) : ℚ :=
  sorry

/-- Theorem stating the probability of selecting at least one factory from district A 
    is 11/21 given the specific conditions -/
theorem probability_at_least_one_from_A : 
  let df : DistrictFactories := { A := 18, B := 27, C := 18 }
  let sf : SampledFactories := { A := 2, B := 3, C := 2 }
  probabilityAtLeastOneFromA df sf = 11/21 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_from_A_l417_41745


namespace NUMINAMATH_CALUDE_gcd_condition_iff_prime_representation_l417_41700

theorem gcd_condition_iff_prime_representation (x y : ℕ) : 
  (∀ n : ℕ, Nat.gcd (n * (Nat.factorial x - x * y - x - y + 2) + 2) 
                    (n * (Nat.factorial x - x * y - x - y + 3) + 3) > 1) ↔
  (∃ q : ℕ, Prime q ∧ q > 3 ∧ x = q - 1 ∧ y = (Nat.factorial (q - 1) - (q - 1)) / q) :=
by sorry

end NUMINAMATH_CALUDE_gcd_condition_iff_prime_representation_l417_41700


namespace NUMINAMATH_CALUDE_range_of_m_l417_41772

-- Define a monotonically decreasing function
def MonoDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

-- State the theorem
theorem range_of_m (f : ℝ → ℝ) (m : ℝ) 
  (h1 : MonoDecreasing f) (h2 : f (2 * m) > f (1 + m)) : 
  m < 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l417_41772


namespace NUMINAMATH_CALUDE_dance_result_l417_41753

/-- Represents a sequence of dance steps, where positive numbers are forward steps
    and negative numbers are backward steps. -/
def dance_sequence : List Int := [-5, 10, -2, 2 * 2]

/-- Calculates the final position after performing a sequence of dance steps. -/
def final_position (steps : List Int) : Int :=
  steps.sum

/-- Proves that the given dance sequence results in a final position 7 steps forward. -/
theorem dance_result :
  final_position dance_sequence = 7 := by
  sorry

end NUMINAMATH_CALUDE_dance_result_l417_41753


namespace NUMINAMATH_CALUDE_hydroton_rainfall_l417_41756

/-- The total rainfall in Hydroton from 2019 to 2021 -/
def total_rainfall (r2019 r2020 r2021 : ℝ) : ℝ :=
  12 * (r2019 + r2020 + r2021)

/-- Theorem: The total rainfall in Hydroton from 2019 to 2021 is 1884 mm -/
theorem hydroton_rainfall : 
  let r2019 : ℝ := 50
  let r2020 : ℝ := r2019 + 5
  let r2021 : ℝ := r2020 - 3
  total_rainfall r2019 r2020 r2021 = 1884 :=
by
  sorry


end NUMINAMATH_CALUDE_hydroton_rainfall_l417_41756


namespace NUMINAMATH_CALUDE_prob_defective_second_draw_specific_l417_41752

/-- Probability of drawing a defective item on the second draw -/
def prob_defective_second_draw (total : ℕ) (defective : ℕ) (good : ℕ) : ℚ :=
  if total > 0 ∧ good > 0 then
    defective / (total - 1 : ℚ)
  else
    0

theorem prob_defective_second_draw_specific :
  prob_defective_second_draw 10 3 7 = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_prob_defective_second_draw_specific_l417_41752


namespace NUMINAMATH_CALUDE_largest_n_satisfying_conditions_l417_41715

theorem largest_n_satisfying_conditions : ∃ (n : ℕ), n = 50 ∧ 
  (∀ m : ℕ, n^2 = (m+1)^3 - m^3 → m ≤ 50) ∧
  (∃ k : ℕ, 2*n + 99 = k^2) ∧
  (∀ n' : ℕ, n' > n → 
    (¬∃ m : ℕ, n'^2 = (m+1)^3 - m^3) ∨ 
    (¬∃ k : ℕ, 2*n' + 99 = k^2)) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_satisfying_conditions_l417_41715


namespace NUMINAMATH_CALUDE_carl_practice_hours_l417_41770

/-- The number of weeks Carl practices -/
def total_weeks : ℕ := 8

/-- The required average hours per week -/
def required_average : ℕ := 15

/-- The hours practiced in the first 7 weeks -/
def first_seven_weeks : List ℕ := [14, 16, 12, 18, 15, 13, 17]

/-- The sum of hours practiced in the first 7 weeks -/
def sum_first_seven : ℕ := first_seven_weeks.sum

/-- The number of hours Carl must practice in the 8th week -/
def hours_eighth_week : ℕ := 15

theorem carl_practice_hours :
  (sum_first_seven + hours_eighth_week) / total_weeks = required_average :=
sorry

end NUMINAMATH_CALUDE_carl_practice_hours_l417_41770


namespace NUMINAMATH_CALUDE_zero_neither_positive_nor_negative_l417_41748

theorem zero_neither_positive_nor_negative :
  ¬(0 > 0) ∧ ¬(0 < 0) := by
  sorry

end NUMINAMATH_CALUDE_zero_neither_positive_nor_negative_l417_41748


namespace NUMINAMATH_CALUDE_raines_change_l417_41716

/-- Calculates the change Raine receives after purchasing items with a discount --/
theorem raines_change (bracelet_price necklace_price mug_price : ℚ)
  (bracelet_qty necklace_qty mug_qty : ℕ)
  (discount_rate : ℚ)
  (payment : ℚ) :
  bracelet_price = 15 →
  necklace_price = 10 →
  mug_price = 20 →
  bracelet_qty = 3 →
  necklace_qty = 2 →
  mug_qty = 1 →
  discount_rate = 1/10 →
  payment = 100 →
  let total_cost := bracelet_price * bracelet_qty + necklace_price * necklace_qty + mug_price * mug_qty
  let discounted_cost := total_cost * (1 - discount_rate)
  payment - discounted_cost = 23.5 := by sorry

end NUMINAMATH_CALUDE_raines_change_l417_41716


namespace NUMINAMATH_CALUDE_hyperbola_m_range_l417_41755

-- Define the hyperbola equation
def is_hyperbola (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (4 - m) - y^2 / (2 + m) = 1

-- Theorem statement
theorem hyperbola_m_range (m : ℝ) :
  is_hyperbola m → m > -2 ∧ m < 4 :=
by
  sorry

end NUMINAMATH_CALUDE_hyperbola_m_range_l417_41755


namespace NUMINAMATH_CALUDE_probability_point_in_circle_l417_41780

theorem probability_point_in_circle (s : ℝ) (r : ℝ) (h_s : s = 6) (h_r : r = 1.5) :
  (π * r^2) / (s^2) = π / 16 := by
  sorry

end NUMINAMATH_CALUDE_probability_point_in_circle_l417_41780


namespace NUMINAMATH_CALUDE_rabbit_fur_genetics_l417_41758

/-- Represents the phase of meiotic division --/
inductive MeioticPhase
  | LateFirst
  | LateSecond

/-- Represents the fur length gene --/
inductive FurGene
  | Long
  | Short

/-- Represents a rabbit's genotype for fur length --/
structure RabbitGenotype where
  allele1 : FurGene
  allele2 : FurGene

/-- Represents the genetic characteristics of rabbit fur --/
structure RabbitFurGenetics where
  totalGenes : ℕ
  genesPerOocyte : ℕ
  nucleotideTypes : ℕ
  separationPhase : MeioticPhase

def isHeterozygous (genotype : RabbitGenotype) : Prop :=
  genotype.allele1 ≠ genotype.allele2

def maxShortFurOocytes (genetics : RabbitFurGenetics) (genotype : RabbitGenotype) : ℕ :=
  genetics.totalGenes / genetics.genesPerOocyte

theorem rabbit_fur_genetics 
  (genetics : RabbitFurGenetics) 
  (genotype : RabbitGenotype) :
  isHeterozygous genotype →
  genetics.totalGenes = 20 →
  genetics.genesPerOocyte = 4 →
  genetics.nucleotideTypes = 4 →
  genetics.separationPhase = MeioticPhase.LateFirst →
  maxShortFurOocytes genetics genotype = 5 :=
by sorry

end NUMINAMATH_CALUDE_rabbit_fur_genetics_l417_41758


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l417_41729

noncomputable section

/-- Triangle ABC with given properties -/
structure TriangleABC where
  -- Sides of the triangle
  a : ℝ
  b : ℝ
  c : ℝ
  -- Angles of the triangle
  A : ℝ
  B : ℝ
  C : ℝ
  -- Given conditions
  a_eq : a = Real.sqrt 5
  b_eq : b = 3
  sin_C_eq : Real.sin C = 2 * Real.sin A
  -- Triangle inequality and angle sum
  triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b
  angle_sum : A + B + C = π

theorem triangle_abc_properties (t : TriangleABC) : 
  t.c = 2 * Real.sqrt 5 ∧ 
  Real.sin (2 * t.A - π/4) = Real.sqrt 2 / 10 := by
  sorry

end

end NUMINAMATH_CALUDE_triangle_abc_properties_l417_41729


namespace NUMINAMATH_CALUDE_digit_sum_theorem_l417_41737

/-- Concatenate two digits to form a two-digit number -/
def concatenate (a b : Nat) : Nat := 10 * a + b

/-- Check if a number is prime -/
def isPrime (n : Nat) : Prop := n > 1 ∧ ∀ d : Nat, d > 1 → d < n → ¬(n % d = 0)

theorem digit_sum_theorem (p q r : Nat) : 
  p < 10 → q < 10 → r < 10 →
  p ≠ q → p ≠ r → q ≠ r →
  isPrime (concatenate p q) →
  isPrime (concatenate p r) →
  isPrime (concatenate q r) →
  concatenate p q ≠ concatenate p r →
  concatenate p q ≠ concatenate q r →
  concatenate p r ≠ concatenate q r →
  (concatenate p q) * (concatenate p r) = 221 →
  p + q + r = 11 := by
sorry

end NUMINAMATH_CALUDE_digit_sum_theorem_l417_41737


namespace NUMINAMATH_CALUDE_min_distance_C1_to_C2_sum_distances_PA_PB_l417_41796

-- Define the circle C1
def C1 (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the line C2
def C2 (x y : ℝ) : Prop := y = x + 2

-- Define the ellipse C1'
def C1' (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define point P
def P : ℝ × ℝ := (-1, 1)

-- Theorem for the minimum distance
theorem min_distance_C1_to_C2 :
  ∃ (d : ℝ), d = Real.sqrt 2 - 1 ∧
  (∀ (x y : ℝ), C1 x y → ∀ (x' y' : ℝ), C2 x' y' →
    Real.sqrt ((x - x')^2 + (y - y')^2) ≥ d) ∧
  (∃ (x y : ℝ), C1 x y ∧ ∃ (x' y' : ℝ), C2 x' y' ∧
    Real.sqrt ((x - x')^2 + (y - y')^2) = d) :=
sorry

-- Theorem for the sum of distances
theorem sum_distances_PA_PB :
  ∃ (A B : ℝ × ℝ), C1' A.1 A.2 ∧ C1' B.1 B.2 ∧
  C2 A.1 A.2 ∧ C2 B.1 B.2 ∧
  Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2) +
  Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2) = 12 * Real.sqrt 2 / 7 :=
sorry

end NUMINAMATH_CALUDE_min_distance_C1_to_C2_sum_distances_PA_PB_l417_41796


namespace NUMINAMATH_CALUDE_line_intersections_l417_41761

/-- The line equation 2y + 5x = 10 -/
def line_equation (x y : ℝ) : Prop := 2 * y + 5 * x = 10

/-- X-axis intersection point -/
def x_intersection : ℝ × ℝ := (2, 0)

/-- Y-axis intersection point -/
def y_intersection : ℝ × ℝ := (0, 5)

/-- Theorem stating that the line intersects the x-axis and y-axis at the given points -/
theorem line_intersections :
  (line_equation x_intersection.1 x_intersection.2) ∧
  (line_equation y_intersection.1 y_intersection.2) ∧
  (x_intersection.2 = 0) ∧
  (y_intersection.1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_line_intersections_l417_41761


namespace NUMINAMATH_CALUDE_inequality_proof_l417_41749

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  2 * Real.sqrt (b * c + c * a + a * b) ≤ Real.sqrt 3 * (((b + c) * (c + a) * (a + b)) ^ (1/3)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l417_41749


namespace NUMINAMATH_CALUDE_max_value_implies_a_l417_41707

/-- The function f(x) = -x^2 + 2ax + 1 - a has a maximum value of 2 in the interval [0, 1] -/
def f (a : ℝ) (x : ℝ) : ℝ := -x^2 + 2*a*x + 1 - a

/-- The maximum value of f(x) in the interval [0, 1] is 2 -/
def max_value (a : ℝ) : Prop := ∀ x, x ∈ Set.Icc 0 1 → f a x ≤ 2

/-- The theorem stating that if f(x) has a maximum value of 2 in [0, 1], then a = -1 or a = 2 -/
theorem max_value_implies_a (a : ℝ) : max_value a → (a = -1 ∨ a = 2) := by sorry

end NUMINAMATH_CALUDE_max_value_implies_a_l417_41707


namespace NUMINAMATH_CALUDE_function_minimum_value_equality_condition_l417_41746

theorem function_minimum_value (x : ℝ) (h : x > 0) : x^2 + 2/x ≥ 3 := by sorry

theorem equality_condition (x : ℝ) (h : x > 0) : x^2 + 2/x = 3 ↔ x = 1 := by sorry

end NUMINAMATH_CALUDE_function_minimum_value_equality_condition_l417_41746


namespace NUMINAMATH_CALUDE_inverse_matrices_solution_l417_41788

theorem inverse_matrices_solution :
  ∀ (a b : ℚ),
  let A : Matrix (Fin 2) (Fin 2) ℚ := ![![a, 3], ![2, 5]]
  let B : Matrix (Fin 2) (Fin 2) ℚ := ![![b, -1/5], ![1/2, 1/10]]
  let I : Matrix (Fin 2) (Fin 2) ℚ := ![![1, 0], ![0, 1]]
  A * B = I → a = 3/2 ∧ b = -5/4 := by sorry

end NUMINAMATH_CALUDE_inverse_matrices_solution_l417_41788


namespace NUMINAMATH_CALUDE_min_sum_squared_distances_l417_41764

open Real
open InnerProductSpace

theorem min_sum_squared_distances (a b c : EuclideanSpace ℝ (Fin 2)) 
  (ha : ‖a‖^2 = 4)
  (hb : ‖b‖^2 = 1)
  (hc : ‖c‖^2 = 9) :
  ∃ (min : ℝ), min = 2 ∧ 
    ∀ (x y z : EuclideanSpace ℝ (Fin 2)), 
      ‖x‖^2 = 4 → ‖y‖^2 = 1 → ‖z‖^2 = 9 →
      ‖x - y‖^2 + ‖x - z‖^2 + ‖y - z‖^2 ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squared_distances_l417_41764


namespace NUMINAMATH_CALUDE_anna_sandwiches_l417_41769

theorem anna_sandwiches (slices_per_sandwich : ℕ) (current_slices : ℕ) (additional_slices : ℕ) :
  slices_per_sandwich = 3 →
  current_slices = 31 →
  additional_slices = 119 →
  (current_slices + additional_slices) / slices_per_sandwich = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_anna_sandwiches_l417_41769


namespace NUMINAMATH_CALUDE_batsman_new_average_l417_41711

/-- Represents a batsman's score statistics -/
structure BatsmanStats where
  initialAverage : ℝ
  inningsPlayed : ℕ
  newInningScore : ℝ
  averageIncrease : ℝ

/-- Theorem: Given a batsman's stats, prove that his new average is 40 runs -/
theorem batsman_new_average (stats : BatsmanStats)
  (h1 : stats.inningsPlayed = 10)
  (h2 : stats.newInningScore = 90)
  (h3 : stats.averageIncrease = 5)
  : stats.initialAverage + stats.averageIncrease = 40 := by
  sorry

#check batsman_new_average

end NUMINAMATH_CALUDE_batsman_new_average_l417_41711


namespace NUMINAMATH_CALUDE_f_properties_l417_41731

-- Define a real-valued function f and its derivative f'
variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

-- Define the properties of f
axiom f_odd : ∀ x, f (-x) = -f x
axiom f_periodic : ∀ x, f (x + 2) = f x
axiom f_differentiable : ∀ x, deriv f x = f' x

-- Theorem statement
theorem f_properties :
  (∀ x, f' (-x) = f' x) ∧ 
  (∀ x, f (x + 1) = -f (1 - x)) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l417_41731
