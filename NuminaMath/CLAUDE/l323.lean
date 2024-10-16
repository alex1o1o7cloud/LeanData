import Mathlib

namespace NUMINAMATH_CALUDE_isosceles_triangle_third_side_l323_32388

/-- An isosceles triangle with side lengths 4 and 8 has its third side equal to 8 -/
theorem isosceles_triangle_third_side : ∀ (a b c : ℝ),
  a = 4 ∧ b = 8 ∧ (a = b ∨ b = c ∨ a = c) →  -- isosceles condition
  (a + b > c ∧ b + c > a ∧ a + c > b) →      -- triangle inequality
  c = 8 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_third_side_l323_32388


namespace NUMINAMATH_CALUDE_rowers_who_voted_l323_32303

theorem rowers_who_voted (num_coaches : ℕ) (votes_per_rower : ℕ) (votes_per_coach : ℕ) : 
  num_coaches = 36 → votes_per_rower = 3 → votes_per_coach = 5 → 
  (num_coaches * votes_per_coach) / votes_per_rower = 60 := by
  sorry

end NUMINAMATH_CALUDE_rowers_who_voted_l323_32303


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l323_32318

/-- Given a line L1 with equation x - 2y - 2 = 0, prove that the line L2 with equation x - 2y - 1 = 0
    passes through the point (1, 0) and is parallel to L1. -/
theorem parallel_line_through_point (x y : ℝ) : 
  (x - 2*y - 1 = 0) ↔ 
  ((x = 1 ∧ y = 0) ∨ 
   ∃ (t : ℝ), x = 1 + t ∧ y = t/2) :=
sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l323_32318


namespace NUMINAMATH_CALUDE_lucas_football_scores_l323_32345

def first_ten_games : List Nat := [5, 2, 6, 3, 10, 1, 3, 3, 4, 2]

def total_first_ten : Nat := first_ten_games.sum

theorem lucas_football_scores :
  ∃ (game11 game12 : Nat),
    game11 < 10 ∧
    game12 < 10 ∧
    (total_first_ten + game11) % 11 = 0 ∧
    (total_first_ten + game11 + game12) % 12 = 0 ∧
    game11 * game12 = 20 := by
  sorry

end NUMINAMATH_CALUDE_lucas_football_scores_l323_32345


namespace NUMINAMATH_CALUDE_pairwise_product_signs_l323_32336

theorem pairwise_product_signs (a b c : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  let products := [a * b, b * c, c * a]
  (products.filter (· > 0)).length = 1 ∨ (products.filter (· > 0)).length = 3 :=
sorry

end NUMINAMATH_CALUDE_pairwise_product_signs_l323_32336


namespace NUMINAMATH_CALUDE_c_possible_values_l323_32374

/-- Represents a string of base-ten digits -/
def DigitString : Type := List Nat

/-- Represents the number of valid splits of a digit string -/
def c (m : Nat) (S : DigitString) : Nat :=
  sorry

/-- Theorem stating the possible values of c(S) -/
theorem c_possible_values (m : Nat) (S : DigitString) :
  m > 1 → ∃ n : Nat, c m S = 0 ∨ c m S = 2^n := by
  sorry

end NUMINAMATH_CALUDE_c_possible_values_l323_32374


namespace NUMINAMATH_CALUDE_tangent_line_to_cubic_curve_l323_32362

theorem tangent_line_to_cubic_curve (k : ℝ) : 
  (∃ x y : ℝ, y = x^3 ∧ y = k*x + 2 ∧ (3 * x^2 = k)) → k = 3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_to_cubic_curve_l323_32362


namespace NUMINAMATH_CALUDE_power_product_simplification_l323_32355

theorem power_product_simplification (x y : ℝ) :
  (x^3 * y^2)^2 * (x / y^3) = x^7 * y := by sorry

end NUMINAMATH_CALUDE_power_product_simplification_l323_32355


namespace NUMINAMATH_CALUDE_max_pons_l323_32313

/-- Represents the purchase of items --/
structure Purchase where
  packs : ℕ
  pins : ℕ
  pons : ℕ

/-- The cost of a purchase --/
def cost (p : Purchase) : ℕ :=
  p.packs + 3 * p.pins + 7 * p.pons

/-- A purchase is valid if it meets the problem conditions --/
def isValid (p : Purchase) : Prop :=
  p.packs ≥ 2 ∧ p.pins ≥ 2 ∧ p.pons ≥ 2 ∧ cost p = 75

/-- The theorem stating that 9 is the maximum number of pons in a valid purchase --/
theorem max_pons :
  ∀ p : Purchase, isValid p → p.pons ≤ 9 ∧ ∃ q : Purchase, isValid q ∧ q.pons = 9 :=
sorry

end NUMINAMATH_CALUDE_max_pons_l323_32313


namespace NUMINAMATH_CALUDE_constant_term_expansion_l323_32375

theorem constant_term_expansion (x : ℝ) (x_neq_0 : x ≠ 0) :
  ∃ (c : ℝ), (x - 1/x)^4 = c + (terms_with_x : ℝ) ∧ c = 6 :=
sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l323_32375


namespace NUMINAMATH_CALUDE_consecutive_product_sum_l323_32335

theorem consecutive_product_sum : ∃ (a b x y z : ℕ), 
  (a + 1 = b) ∧ 
  (x + 1 = y) ∧ 
  (y + 1 = z) ∧
  (a * b = 1320) ∧ 
  (x * y * z = 1320) ∧ 
  (a + b + x + y + z = 106) := by
sorry

end NUMINAMATH_CALUDE_consecutive_product_sum_l323_32335


namespace NUMINAMATH_CALUDE_largest_divisor_of_n_l323_32346

theorem largest_divisor_of_n (n : ℕ) (h1 : n > 0) (h2 : 36 ∣ n^3) : 
  ∀ d : ℕ, d ∣ n → d ≤ 6 :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_n_l323_32346


namespace NUMINAMATH_CALUDE_conference_handshakes_l323_32372

/-- The number of unique handshakes in a conference -/
def unique_handshakes (n : ℕ) (k : ℕ) : ℕ :=
  (n * k) / 2

/-- Theorem: In a conference of 12 people where each person shakes hands with 6 others,
    there are 36 unique handshakes -/
theorem conference_handshakes :
  unique_handshakes 12 6 = 36 := by
  sorry

end NUMINAMATH_CALUDE_conference_handshakes_l323_32372


namespace NUMINAMATH_CALUDE_paper_cutting_equations_l323_32342

/-- Represents the paper cutting scenario in a seventh-grade class. -/
theorem paper_cutting_equations (x y : ℕ) : 
  (x + y = 12 ∧ 6 * x = 3 * (4 * y)) ↔ 
  (x = number_of_sheets_for_stars ∧ 
   y = number_of_sheets_for_flowers ∧ 
   total_sheets_used = 12 ∧ 
   stars_per_sheet = 6 ∧ 
   flowers_per_sheet = 4 ∧ 
   total_stars = 3 * total_flowers) :=
sorry

end NUMINAMATH_CALUDE_paper_cutting_equations_l323_32342


namespace NUMINAMATH_CALUDE_water_tank_capacity_l323_32376

theorem water_tank_capacity (x : ℚ) : 
  (1 / 3 : ℚ) * x + 16 = x → x = 24 := by
  sorry

end NUMINAMATH_CALUDE_water_tank_capacity_l323_32376


namespace NUMINAMATH_CALUDE_complex_determinant_solution_l323_32360

/-- Definition of the determinant operation -/
def det (a b c d : ℂ) : ℂ := a * d - b * c

/-- Theorem stating that z = 2 - i satisfies the given condition -/
theorem complex_determinant_solution :
  ∃ z : ℂ, det z (1 + 2*I) (1 - I) (1 + I) = 0 ∧ z = 2 - I :=
by sorry

end NUMINAMATH_CALUDE_complex_determinant_solution_l323_32360


namespace NUMINAMATH_CALUDE_mod_product_253_649_l323_32348

theorem mod_product_253_649 (n : ℕ) : 
  253 * 649 ≡ n [ZMOD 100] → 0 ≤ n → n < 100 → n = 97 := by
  sorry

end NUMINAMATH_CALUDE_mod_product_253_649_l323_32348


namespace NUMINAMATH_CALUDE_fraction_equals_zero_l323_32315

theorem fraction_equals_zero (x : ℝ) : (x - 2) / (x + 2) = 0 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_zero_l323_32315


namespace NUMINAMATH_CALUDE_novelist_writing_speed_l323_32365

/-- Calculates the average writing speed given total words, total hours, and break hours. -/
def average_writing_speed (total_words : ℕ) (total_hours : ℕ) (break_hours : ℕ) : ℚ :=
  total_words / (total_hours - break_hours)

/-- Proves that the average writing speed is 625 words per hour for the given conditions. -/
theorem novelist_writing_speed :
  average_writing_speed 50000 100 20 = 625 := by
  sorry

end NUMINAMATH_CALUDE_novelist_writing_speed_l323_32365


namespace NUMINAMATH_CALUDE_square_sum_proof_l323_32359

theorem square_sum_proof (a b : ℝ) (h1 : a - b = 6) (h2 : a * b = 32) : a^2 + b^2 = 100 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_proof_l323_32359


namespace NUMINAMATH_CALUDE_percentage_relation_l323_32311

theorem percentage_relation (x : ℝ) (h : 0.4 * x = 160) : 0.5 * x = 200 := by
  sorry

end NUMINAMATH_CALUDE_percentage_relation_l323_32311


namespace NUMINAMATH_CALUDE_mountain_trail_length_l323_32322

/-- Represents the hike on the Mountain Trail -/
structure MountainTrail where
  -- Daily distances hiked
  day1 : ℝ
  day2 : ℝ
  day3 : ℝ
  day4 : ℝ
  day5 : ℝ
  -- Conditions
  first_two_days : day1 + day2 = 30
  second_third_avg : (day2 + day3) / 2 = 16
  last_three_days : day3 + day4 + day5 = 45
  first_fourth_days : day1 + day4 = 32

/-- The theorem stating the total length of the Mountain Trail -/
theorem mountain_trail_length (hike : MountainTrail) : 
  hike.day1 + hike.day2 + hike.day3 + hike.day4 + hike.day5 = 107 := by
  sorry


end NUMINAMATH_CALUDE_mountain_trail_length_l323_32322


namespace NUMINAMATH_CALUDE_product_different_from_hundred_l323_32393

theorem product_different_from_hundred : ∃! (x y : ℚ), 
  ((x = 10 ∧ y = 10) ∨ 
   (x = 20 ∧ y = -5) ∨ 
   (x = -4 ∧ y = -25) ∨ 
   (x = 50 ∧ y = 2) ∨ 
   (x = 5/2 ∧ y = 40)) ∧ 
  x * y ≠ 100 := by
  sorry

end NUMINAMATH_CALUDE_product_different_from_hundred_l323_32393


namespace NUMINAMATH_CALUDE_ball_distribution_exists_l323_32319

theorem ball_distribution_exists : ∃ N : ℕ+, 
  81 < N ∧ N ≤ 200 ∧ 
  (0.12 * N.val : ℚ).num % (0.12 * N.val : ℚ).den = 0 ∧
  (0.20 * N.val : ℚ).num % (0.20 * N.val : ℚ).den = 0 ∧
  (2 * N.val) % 3 = 0 ∧
  N.val - (0.12 * N.val + 0.20 * N.val + 2 * N.val / 3 : ℚ).floor > 0 :=
sorry

end NUMINAMATH_CALUDE_ball_distribution_exists_l323_32319


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l323_32364

/-- Two vectors in ℝ² are perpendicular if their dot product is zero -/
def perpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

/-- Given vectors a = (1, 2) and b = (x, 1), if they are perpendicular, then x = -2 -/
theorem perpendicular_vectors (x : ℝ) :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (x, 1)
  perpendicular a b → x = -2 := by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l323_32364


namespace NUMINAMATH_CALUDE_trigonometric_inequality_l323_32389

theorem trigonometric_inequality (x y z : ℝ) 
  (hx : 0 < x ∧ x < π/2) 
  (hy : 0 < y ∧ y < π/2) 
  (hz : 0 < z ∧ z < π/2) : 
  π/2 + 2 * Real.sin x * Real.cos y + 2 * Real.sin y * Real.cos z > 
  Real.sin (2*x) + Real.sin (2*y) + Real.sin (2*z) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_inequality_l323_32389


namespace NUMINAMATH_CALUDE_rectangular_prism_parallel_edges_l323_32324

-- Define a rectangular prism
structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ
  length_pos : length > 0
  width_pos : width > 0
  height_pos : height > 0
  unequal_faces : length ≠ width ∧ width ≠ height ∧ height ≠ length

-- Define a function to count parallel edge pairs
def count_parallel_edge_pairs (prism : RectangularPrism) : ℕ :=
  12

-- Theorem statement
theorem rectangular_prism_parallel_edges (prism : RectangularPrism) :
  count_parallel_edge_pairs prism = 12 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_parallel_edges_l323_32324


namespace NUMINAMATH_CALUDE_prize_orders_count_l323_32323

/-- Represents a tournament with n players -/
structure Tournament (n : ℕ) where
  players : Fin n

/-- The playoff structure for a 6-player tournament -/
def playoff_structure (t : Tournament 6) : Prop :=
  ∃ (order : Fin 6 → Fin 6),
    -- Each player gets a unique position
    Function.Injective order ∧
    -- The structure follows the described playoff

    -- #6 vs #5, loser gets 6th place
    (order 5 = 6 ∨ order 4 = 6) ∧
    
    -- Winner of #6 vs #5 plays against #4
    (order 5 ≠ 6 → order 5 < order 3) ∧
    (order 4 ≠ 6 → order 4 < order 3) ∧
    
    -- Subsequent matches
    order 3 < order 2 ∧
    order 2 < order 1 ∧
    order 1 < order 0

/-- The number of possible prize orders in a 6-player tournament with the given playoff structure -/
def num_prize_orders (t : Tournament 6) : ℕ := 32

/-- Theorem stating that the number of possible prize orders is 32 -/
theorem prize_orders_count (t : Tournament 6) :
  playoff_structure t → num_prize_orders t = 32 := by
  sorry


end NUMINAMATH_CALUDE_prize_orders_count_l323_32323


namespace NUMINAMATH_CALUDE_prob_sum_three_two_dice_l323_32301

/-- The number of faces on each die -/
def num_faces : ℕ := 6

/-- The total number of possible outcomes when rolling two dice -/
def total_outcomes : ℕ := num_faces * num_faces

/-- The number of ways to roll a sum of 3 with two dice -/
def favorable_outcomes : ℕ := 2

/-- The probability of an event occurring -/
def probability (favorable : ℕ) (total : ℕ) : ℚ := favorable / total

/-- Theorem: The probability of rolling a sum of 3 with two fair dice is 1/18 -/
theorem prob_sum_three_two_dice : 
  probability favorable_outcomes total_outcomes = 1 / 18 := by sorry

end NUMINAMATH_CALUDE_prob_sum_three_two_dice_l323_32301


namespace NUMINAMATH_CALUDE_second_polygon_sides_l323_32352

/-- 
Given two regular polygons with the same perimeter, where the first has 50 sides 
and a side length three times as long as the second, prove that the second polygon has 150 sides.
-/
theorem second_polygon_sides (s : ℝ) (n : ℕ) : 
  s > 0 → 
  50 * (3 * s) = n * s → 
  n = 150 := by
  sorry

end NUMINAMATH_CALUDE_second_polygon_sides_l323_32352


namespace NUMINAMATH_CALUDE_exactly_two_valid_numbers_l323_32325

def is_valid_number (n : Nat) : Prop :=
  let digits := n.digits 10
  digits.length = 6 ∧
  digits.toFinset = {1, 2, 3, 4, 5, 6} ∧
  (n / 10000) % 2 = 0 ∧
  (n / 1000) % 3 = 0 ∧
  (n / 100) % 4 = 0 ∧
  (n / 10) % 5 = 0 ∧
  n % 6 = 0

theorem exactly_two_valid_numbers : 
  ∃! (s : Finset Nat), s.card = 2 ∧ ∀ n ∈ s, is_valid_number n :=
sorry

end NUMINAMATH_CALUDE_exactly_two_valid_numbers_l323_32325


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l323_32337

theorem diophantine_equation_solution (x y z : ℤ) :
  x^2 + y^2 + z^2 = 2*x*y*z → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l323_32337


namespace NUMINAMATH_CALUDE_tan_80_in_terms_of_cos_100_l323_32387

theorem tan_80_in_terms_of_cos_100 (m : ℝ) (h : Real.cos (100 * π / 180) = m) :
  Real.tan (80 * π / 180) = Real.sqrt (1 - m^2) / (-m) := by
  sorry

end NUMINAMATH_CALUDE_tan_80_in_terms_of_cos_100_l323_32387


namespace NUMINAMATH_CALUDE_hyperbola_m_range_l323_32331

-- Define the hyperbola equation
def hyperbola_equation (x y m : ℝ) : Prop :=
  x^2 / (m^2 - 4) - y^2 / (m + 1) = 1

-- Define the condition that foci are on y-axis
def foci_on_y_axis (m : ℝ) : Prop :=
  -(m + 1) > 0 ∧ 4 - m^2 > 0

-- Theorem statement
theorem hyperbola_m_range :
  ∀ m : ℝ, (∃ x y : ℝ, hyperbola_equation x y m) ∧ foci_on_y_axis m → 
  m > -2 ∧ m < -1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_m_range_l323_32331


namespace NUMINAMATH_CALUDE_chess_team_boys_l323_32358

theorem chess_team_boys (total : ℕ) (attendees : ℕ) 
  (junior_girls : ℕ) (senior_girls : ℕ) (boys : ℕ) :
  total = 32 →
  attendees = 18 →
  junior_girls + senior_girls + boys = total →
  (junior_girls / 3 + senior_girls / 2 + boys : ℚ) = attendees →
  boys = 4 :=
by sorry

end NUMINAMATH_CALUDE_chess_team_boys_l323_32358


namespace NUMINAMATH_CALUDE_hyperbola_midpoint_l323_32399

def hyperbola (x y : ℝ) : Prop := x^2 - y^2/9 = 1

def is_midpoint (x₀ y₀ x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₀ = (x₁ + x₂) / 2 ∧ y₀ = (y₁ + y₂) / 2

theorem hyperbola_midpoint :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    hyperbola x₁ y₁ ∧
    hyperbola x₂ y₂ ∧
    is_midpoint (-1) (-4) x₁ y₁ x₂ y₂ ∧
    ¬∃ (x₁' y₁' x₂' y₂' : ℝ),
      hyperbola x₁' y₁' ∧
      hyperbola x₂' y₂' ∧
      (is_midpoint 1 1 x₁' y₁' x₂' y₂' ∨
       is_midpoint (-1) 2 x₁' y₁' x₂' y₂' ∨
       is_midpoint 1 3 x₁' y₁' x₂' y₂') :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_midpoint_l323_32399


namespace NUMINAMATH_CALUDE_min_distance_squared_l323_32329

def is_geometric_progression (x y z : ℝ) : Prop :=
  ∃ r : ℝ, y = x * r ∧ z = y * r

def is_arithmetic_progression (a b c : ℝ) : Prop :=
  b - a = c - b

theorem min_distance_squared (x y z : ℝ) :
  is_geometric_progression x y z →
  is_arithmetic_progression (x * y) (y * z) (x * z) →
  z ≥ 1 →
  x ≠ y →
  y ≠ z →
  x ≠ z →
  (∀ x' y' z' : ℝ, 
    is_geometric_progression x' y' z' →
    is_arithmetic_progression (x' * y') (y' * z') (x' * z') →
    z' ≥ 1 →
    x' ≠ y' →
    y' ≠ z' →
    x' ≠ z' →
    (x - 1)^2 + (y - 1)^2 + (z - 1)^2 ≤ (x' - 1)^2 + (y' - 1)^2 + (z' - 1)^2) →
  (x - 1)^2 + (y - 1)^2 + (z - 1)^2 = 18 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_squared_l323_32329


namespace NUMINAMATH_CALUDE_point_on_line_l323_32347

/-- A point represented by its x and y coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

theorem point_on_line : 
  let p1 : Point := ⟨0, 4⟩
  let p2 : Point := ⟨-6, 1⟩
  let p3 : Point := ⟨6, 7⟩
  collinear p1 p2 p3 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_l323_32347


namespace NUMINAMATH_CALUDE_pets_problem_l323_32396

theorem pets_problem (total_students : ℕ) 
  (students_with_dogs : ℕ) 
  (students_with_cats : ℕ) 
  (students_with_other_pets : ℕ) 
  (students_no_pets : ℕ) 
  (only_dogs : ℕ) 
  (only_cats : ℕ) 
  (only_other_pets : ℕ) 
  (dogs_and_cats : ℕ) 
  (dogs_and_other : ℕ) 
  (cats_and_other : ℕ) :
  total_students = 40 →
  students_with_dogs = 20 →
  students_with_cats = total_students / 4 →
  students_with_other_pets = 10 →
  students_no_pets = 5 →
  only_dogs = 15 →
  only_cats = 4 →
  only_other_pets = 5 →
  total_students = only_dogs + only_cats + only_other_pets + 
    dogs_and_cats + dogs_and_other + cats_and_other + 
    students_no_pets + (students_with_dogs + students_with_cats + 
    students_with_other_pets - (only_dogs + only_cats + only_other_pets + 
    dogs_and_cats + dogs_and_other + cats_and_other)) →
  students_with_dogs + students_with_cats + students_with_other_pets - 
    (only_dogs + only_cats + only_other_pets + 
    dogs_and_cats + dogs_and_other + cats_and_other) = 0 :=
by sorry

end NUMINAMATH_CALUDE_pets_problem_l323_32396


namespace NUMINAMATH_CALUDE_divisibility_statements_l323_32385

theorem divisibility_statements : 
  (∃ n : ℤ, 25 = 5 * n) ∧ 
  (∃ n : ℤ, 209 = 19 * n) ∧ 
  ¬(∃ n : ℤ, 63 = 19 * n) ∧
  (∃ n : ℤ, 140 = 7 * n) ∧
  (∃ n : ℤ, 90 = 30 * n) ∧
  (∃ n : ℤ, 34 = 17 * n) ∧
  (∃ n : ℤ, 68 = 17 * n) := by
  sorry

#check divisibility_statements

end NUMINAMATH_CALUDE_divisibility_statements_l323_32385


namespace NUMINAMATH_CALUDE_y_intercept_of_line_l323_32343

/-- The y-intercept of the line 3x - 5y = 15 is -3 -/
theorem y_intercept_of_line (x y : ℝ) : 3 * x - 5 * y = 15 → x = 0 → y = -3 := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_of_line_l323_32343


namespace NUMINAMATH_CALUDE_snooker_ticket_difference_l323_32350

theorem snooker_ticket_difference :
  ∀ (vip_price general_price : ℚ) 
    (total_tickets : ℕ) 
    (total_cost : ℚ) 
    (vip_count general_count : ℕ),
  vip_price = 45 →
  general_price = 20 →
  total_tickets = 320 →
  total_cost = 7500 →
  vip_count + general_count = total_tickets →
  vip_price * vip_count + general_price * general_count = total_cost →
  general_count - vip_count = 232 := by
sorry

end NUMINAMATH_CALUDE_snooker_ticket_difference_l323_32350


namespace NUMINAMATH_CALUDE_tangent_line_at_origin_minimum_value_condition_function_inequality_condition_l323_32328

open Real

/-- The function f(x) with parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + 2 * a * x

theorem tangent_line_at_origin (h : ℝ → ℝ := fun x ↦ Real.exp x + 2 * x) :
  ∃ (m b : ℝ), m = 3 ∧ b = 1 ∧ ∀ x y, y = h x → m * x - y + b = 0 := by sorry

theorem minimum_value_condition (a : ℝ) :
  (∀ x ≥ 1, f a x ≥ 0) ∧ (∃ x ≥ 1, f a x = 0) → a = -Real.exp 1 / 2 := by sorry

theorem function_inequality_condition (a : ℝ) :
  (∀ x ≥ 0, f a x ≥ Real.exp (-x)) ↔ a ≥ -1 := by sorry

end NUMINAMATH_CALUDE_tangent_line_at_origin_minimum_value_condition_function_inequality_condition_l323_32328


namespace NUMINAMATH_CALUDE_existence_of_special_binary_number_l323_32371

/-- Represents a binary number as a list of booleans -/
def BinaryNumber := List Bool

/-- Generates all n-digit binary numbers -/
def allNDigitBinaryNumbers (n : Nat) : List BinaryNumber :=
  sorry

/-- Checks if a binary number is a substring of another binary number -/
def isSubstring (sub target : BinaryNumber) : Bool :=
  sorry

/-- Checks if all n-digit binary numbers are substrings of T -/
def allNDigitNumbersAreSubstrings (T : BinaryNumber) (n : Nat) : Prop :=
  ∀ sub, sub ∈ allNDigitBinaryNumbers n → isSubstring sub T

/-- Checks if all n-digit substrings of T are distinct -/
def allNDigitSubstringsAreDistinct (T : BinaryNumber) (n : Nat) : Prop :=
  sorry

theorem existence_of_special_binary_number (n : Nat) :
  ∃ T : BinaryNumber,
    T.length = 2^n + (n - 1) ∧
    allNDigitNumbersAreSubstrings T n ∧
    allNDigitSubstringsAreDistinct T n :=
  sorry

end NUMINAMATH_CALUDE_existence_of_special_binary_number_l323_32371


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l323_32391

def A : Set ℝ := {x | -2 < x ∧ x < 2}
def B : Set ℝ := {x | 1 < x ∧ x < 3}

theorem intersection_of_A_and_B : A ∩ B = {x | 1 < x ∧ x < 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l323_32391


namespace NUMINAMATH_CALUDE_exponent_division_l323_32316

theorem exponent_division (a : ℝ) (h : a ≠ 0) : a^6 / a^2 = a^4 := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l323_32316


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l323_32370

theorem sqrt_equation_solution (y : ℝ) : 
  Real.sqrt (2 + Real.sqrt (3 * y - 4)) = Real.sqrt 9 → y = 53 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l323_32370


namespace NUMINAMATH_CALUDE_tetrahedron_volume_l323_32326

/-- The volume of a tetrahedron with an inscribed sphere -/
theorem tetrahedron_volume (S₁ S₂ S₃ S₄ r : ℝ) (h₁ : 0 < S₁) (h₂ : 0 < S₂) (h₃ : 0 < S₃) (h₄ : 0 < S₄) (hr : 0 < r) :
  ∃ V : ℝ, V = (1 / 3) * (S₁ + S₂ + S₃ + S₄) * r ∧ V > 0 :=
by sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_l323_32326


namespace NUMINAMATH_CALUDE_games_needed_for_512_players_l323_32308

/-- Represents a single-elimination tournament -/
structure SingleEliminationTournament where
  initial_players : ℕ
  games_played : ℕ

/-- Calculates the number of games needed to declare a champion -/
def games_needed (tournament : SingleEliminationTournament) : ℕ :=
  tournament.initial_players - 1

/-- Theorem: In a single-elimination tournament with 512 initial players,
    511 games are needed to declare a champion -/
theorem games_needed_for_512_players :
  ∀ (tournament : SingleEliminationTournament),
    tournament.initial_players = 512 →
    games_needed tournament = 511 := by
  sorry

#check games_needed_for_512_players

end NUMINAMATH_CALUDE_games_needed_for_512_players_l323_32308


namespace NUMINAMATH_CALUDE_revenue_change_after_price_and_sales_change_l323_32312

theorem revenue_change_after_price_and_sales_change
  (initial_price initial_sales : ℝ)
  (price_increase : ℝ)
  (sales_decrease : ℝ)
  (h1 : price_increase = 0.3)
  (h2 : sales_decrease = 0.2)
  : (((initial_price * (1 + price_increase)) * (initial_sales * (1 - sales_decrease)) - initial_price * initial_sales) / (initial_price * initial_sales)) * 100 = 4 := by
sorry

end NUMINAMATH_CALUDE_revenue_change_after_price_and_sales_change_l323_32312


namespace NUMINAMATH_CALUDE_purple_nails_count_l323_32381

/-- The number of nails painted purple -/
def purple_nails : ℕ := sorry

/-- The number of nails painted blue -/
def blue_nails : ℕ := 8

/-- The number of nails painted striped -/
def striped_nails : ℕ := sorry

/-- The total number of nails -/
def total_nails : ℕ := 20

/-- The difference in percentage points between blue and striped nails -/
def percentage_difference : ℝ := 10

theorem purple_nails_count : purple_nails = 6 := by
  have h1 : purple_nails + blue_nails + striped_nails = total_nails := sorry
  have h2 : (blue_nails : ℝ) / total_nails * 100 - (striped_nails : ℝ) / total_nails * 100 = percentage_difference := sorry
  sorry

end NUMINAMATH_CALUDE_purple_nails_count_l323_32381


namespace NUMINAMATH_CALUDE_simplify_expression_l323_32340

theorem simplify_expression (y : ℝ) : 4*y + 9*y^2 + 8 - (3 - 4*y - 9*y^2) = 18*y^2 + 8*y + 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l323_32340


namespace NUMINAMATH_CALUDE_product_remainder_mod_five_l323_32397

theorem product_remainder_mod_five : (1236 * 7483 * 53) % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_mod_five_l323_32397


namespace NUMINAMATH_CALUDE_unique_quadratic_solution_l323_32366

theorem unique_quadratic_solution (a : ℝ) (h1 : a ≠ 0) 
  (h2 : ∃! x, a * x^2 + 36 * x + 12 = 0) : 
  ∃ x, a * x^2 + 36 * x + 12 = 0 ∧ x = -2/3 := by
sorry

end NUMINAMATH_CALUDE_unique_quadratic_solution_l323_32366


namespace NUMINAMATH_CALUDE_line_with_definite_slope_line_through_two_points_l323_32320

-- Statement B
theorem line_with_definite_slope (m : ℝ) :
  ∃ (k : ℝ), ∀ (x y : ℝ), m * x + y - 2 = 0 → y = k * x + (2 : ℝ) := by
  sorry

-- Statement D
theorem line_through_two_points (x₁ y₁ x₂ y₂ : ℝ) (h : x₁ ≠ x₂) :
  ∀ (x y : ℝ), y - y₁ = ((y₂ - y₁) / (x₂ - x₁)) * (x - x₁) ↔
    ∃ (m b : ℝ), y = m * x + b ∧ y₁ = m * x₁ + b ∧ y₂ = m * x₂ + b := by
  sorry

end NUMINAMATH_CALUDE_line_with_definite_slope_line_through_two_points_l323_32320


namespace NUMINAMATH_CALUDE_equal_interior_angles_decagon_l323_32392

/-- The measure of an interior angle in a regular decagon -/
def regular_decagon_angle : ℝ := 144

/-- A decagon is a polygon with 10 sides -/
def decagon_sides : ℕ := 10

/-- Theorem: In a decagon where all interior angles are equal, each interior angle measures 144° -/
theorem equal_interior_angles_decagon : 
  ∀ (angles : Fin decagon_sides → ℝ), 
    (∀ (i j : Fin decagon_sides), angles i = angles j) →
    (∀ (i : Fin decagon_sides), angles i = regular_decagon_angle) :=
by sorry

end NUMINAMATH_CALUDE_equal_interior_angles_decagon_l323_32392


namespace NUMINAMATH_CALUDE_at_least_one_first_class_l323_32367

def total_parts : ℕ := 20
def first_class_parts : ℕ := 16
def second_class_parts : ℕ := 4
def selections : ℕ := 3

theorem at_least_one_first_class :
  (Nat.choose first_class_parts 1 * Nat.choose second_class_parts 2) +
  (Nat.choose first_class_parts 2 * Nat.choose second_class_parts 1) +
  (Nat.choose first_class_parts 3) =
  Nat.choose total_parts selections - Nat.choose second_class_parts selections :=
sorry

end NUMINAMATH_CALUDE_at_least_one_first_class_l323_32367


namespace NUMINAMATH_CALUDE_max_value_of_roots_squared_sum_l323_32333

theorem max_value_of_roots_squared_sum (k : ℝ) (x₁ x₂ : ℝ) : 
  x₁^2 - (k-2)*x₁ + (k^2 + 3*k + 5) = 0 →
  x₂^2 - (k-2)*x₂ + (k^2 + 3*k + 5) = 0 →
  x₁ ≠ x₂ →
  ∃ (max : ℝ), max = 18 ∧ x₁^2 + x₂^2 ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_roots_squared_sum_l323_32333


namespace NUMINAMATH_CALUDE_dihedral_angle_is_45_degrees_l323_32338

-- Define the regular triangular prism
structure RegularTriangularPrism :=
  (A B C A₁ B₁ C₁ : EuclideanSpace ℝ (Fin 3))

-- Define points D and E on the lateral edges
def D (prism : RegularTriangularPrism) : EuclideanSpace ℝ (Fin 3) := sorry
def E (prism : RegularTriangularPrism) : EuclideanSpace ℝ (Fin 3) := sorry

-- Define the conditions
def conditions (prism : RegularTriangularPrism) : Prop :=
  (dist (E prism) (prism.C) = dist prism.B prism.C) ∧
  (dist (E prism) (prism.C) = 2 * dist (D prism) prism.B)

-- Define the dihedral angle between ADE and ABC
def dihedralAngle (prism : RegularTriangularPrism) : ℝ := sorry

-- State the theorem
theorem dihedral_angle_is_45_degrees (prism : RegularTriangularPrism) :
  conditions prism → dihedralAngle prism = 45 * π / 180 := by sorry

end NUMINAMATH_CALUDE_dihedral_angle_is_45_degrees_l323_32338


namespace NUMINAMATH_CALUDE_total_selections_l323_32334

/-- Represents a hexagonal arrangement of circles -/
structure HexCircleArrangement :=
  (total_circles : ℕ)
  (side_length : ℕ)

/-- Calculates the number of ways to select three consecutive circles in one direction -/
def consecutive_selections (n : ℕ) : ℕ :=
  if n < 3 then 0 else n - 2

/-- Calculates the number of ways to select three consecutive circles in a diagonal direction -/
def diagonal_selections (n : ℕ) : ℕ :=
  if n < 3 then 0 else (n - 2) * (n - 1) / 2

/-- The main theorem stating the total number of ways to select three consecutive circles -/
theorem total_selections (h : HexCircleArrangement) 
  (h_total : h.total_circles = 33) 
  (h_side : h.side_length = 7) : 
  consecutive_selections h.side_length + 2 * diagonal_selections h.side_length = 57 := by
  sorry


end NUMINAMATH_CALUDE_total_selections_l323_32334


namespace NUMINAMATH_CALUDE_min_n_plus_d_l323_32384

/-- An arithmetic sequence with positive integer terms -/
structure ArithmeticSequence where
  a : ℕ → ℕ
  d : ℕ
  first_term : a 1 = 1949
  nth_term : ∃ n : ℕ, a n = 2009
  is_arithmetic : ∀ n : ℕ, a (n + 1) = a n + d

/-- The minimum value of n + d for the given arithmetic sequence -/
theorem min_n_plus_d (seq : ArithmeticSequence) : 
  ∃ n d : ℕ, seq.d = d ∧ (∃ k, seq.a k = 2009) ∧ 
  (∀ m e : ℕ, seq.d = e ∧ (∃ j, seq.a j = 2009) → n + d ≤ m + e) ∧
  n + d = 17 := by
  sorry

end NUMINAMATH_CALUDE_min_n_plus_d_l323_32384


namespace NUMINAMATH_CALUDE_cube_volume_l323_32306

theorem cube_volume (s : ℝ) : 
  (s + 2) * (s - 2) * s = s^3 - 12 → s^3 = 27 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_l323_32306


namespace NUMINAMATH_CALUDE_tangential_quadrilateral_properties_l323_32300

/-- A circumscribed and tangent quadrilateral with sides a, b, c, d, area t, 
    inscribed circle radius r, and circumscribed circle radius R. -/
structure TangentialQuadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  t : ℝ
  r : ℝ
  R : ℝ
  h_positive : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < t ∧ 0 < r ∧ 0 < R

theorem tangential_quadrilateral_properties (q : TangentialQuadrilateral) :
  q.t = Real.sqrt (q.a * q.b * q.c * q.d) ∧ q.r * Real.sqrt 2 ≤ q.R := by
  sorry

end NUMINAMATH_CALUDE_tangential_quadrilateral_properties_l323_32300


namespace NUMINAMATH_CALUDE_average_of_three_numbers_l323_32369

theorem average_of_three_numbers (y : ℝ) : (15 + 18 + y) / 3 = 21 → y = 30 := by
  sorry

end NUMINAMATH_CALUDE_average_of_three_numbers_l323_32369


namespace NUMINAMATH_CALUDE_weight_problem_l323_32353

theorem weight_problem (a b c : ℝ) 
  (h1 : (a + b + c) / 3 = 45)
  (h2 : (a + b) / 2 = 42)
  (h3 : (b + c) / 2 = 43) :
  b = 35 := by
sorry

end NUMINAMATH_CALUDE_weight_problem_l323_32353


namespace NUMINAMATH_CALUDE_toy_factory_wage_calculation_l323_32398

/-- Toy factory production and wage calculation -/
theorem toy_factory_wage_calculation 
  (planned_weekly_production : ℕ)
  (average_daily_production : ℕ)
  (deviations : List ℤ)
  (base_wage_per_toy : ℕ)
  (bonus_per_extra_toy : ℕ)
  (deduction_per_missing_toy : ℕ)
  (h1 : planned_weekly_production = 700)
  (h2 : average_daily_production = 100)
  (h3 : deviations = [5, -2, -4, 13, -6, 6, -3])
  (h4 : base_wage_per_toy = 20)
  (h5 : bonus_per_extra_toy = 5)
  (h6 : deduction_per_missing_toy = 4)
  : (planned_weekly_production + deviations.sum) * base_wage_per_toy + 
    (deviations.sum * (base_wage_per_toy + bonus_per_extra_toy)) = 14225 := by
  sorry

end NUMINAMATH_CALUDE_toy_factory_wage_calculation_l323_32398


namespace NUMINAMATH_CALUDE_cone_height_l323_32377

/-- Given a cone with slant height 10 and base radius 5, its height is 5√3 -/
theorem cone_height (l r h : ℝ) (hl : l = 10) (hr : r = 5) 
  (h_def : h = Real.sqrt (l^2 - r^2)) : h = 5 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_cone_height_l323_32377


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l323_32332

/-- A rhombus with given diagonal lengths has a specific perimeter. -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 12) (h2 : d2 = 16) :
  let side := Real.sqrt ((d1/2)^2 + (d2/2)^2)
  4 * side = 40 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l323_32332


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l323_32383

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 3 * x + 2

-- Define the solution set condition
def solution_set (a b : ℝ) : Set ℝ := {x : ℝ | x < 1 ∨ x > b}

-- Define the inequality function
def g (c : ℝ) (x : ℝ) : ℝ := x^2 - (c + 2) * x + 2 * x

-- Theorem statement
theorem quadratic_inequality_solution :
  ∀ (a b : ℝ), (∀ x, f a x > 0 ↔ x ∈ solution_set a b) →
    (a = 1 ∧ b = 2) ∧
    (∀ c : ℝ,
      (c > 0 → {x | g c x < 0} = Set.Ioo 0 c) ∧
      (c = 0 → {x | g c x < 0} = ∅) ∧
      (c < 0 → {x | g c x < 0} = Set.Ioo c 0)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l323_32383


namespace NUMINAMATH_CALUDE_string_longest_piece_fraction_l323_32386

theorem string_longest_piece_fraction (L : ℝ) (x : ℝ) (h1 : x > 0) : 
  x + 2*x + 4*x + 8*x = L → 8*x / L = 8/15 := by
  sorry

end NUMINAMATH_CALUDE_string_longest_piece_fraction_l323_32386


namespace NUMINAMATH_CALUDE_water_fee_calculation_l323_32357

-- Define the water fee structure
structure WaterFee where
  a : ℝ  -- rate for usage ≤ 6m³
  b : ℝ  -- rate for usage > 6m³

-- Define the water usage data
structure WaterUsage where
  usage : ℝ
  fee : ℝ

-- Theorem statement
theorem water_fee_calculation (wf : WaterFee) (march : WaterUsage) (april : WaterUsage)
  (h1 : march.usage = 5 ∧ march.fee = 7.5)
  (h2 : april.usage = 9 ∧ april.fee = 27)
  (h3 : march.usage ≤ 6 ∧ april.usage > 6) :
  (∀ x > 6, wf.a * 6 + wf.b * (x - 6) = 6 * x - 27) ∧
  (∃ x > 6, wf.a * 6 + wf.b * (x - 6) = 39 ∧ x = 11) := by
  sorry

end NUMINAMATH_CALUDE_water_fee_calculation_l323_32357


namespace NUMINAMATH_CALUDE_family_age_problem_l323_32351

theorem family_age_problem (family_size : ℕ) (current_average_age : ℝ) (youngest_age : ℝ) :
  family_size = 5 →
  current_average_age = 20 →
  youngest_age = 10 →
  let total_age : ℝ := family_size * current_average_age
  let other_members_age : ℝ := total_age - youngest_age
  let age_reduction : ℝ := (family_size - 1) * youngest_age
  let total_age_at_birth : ℝ := other_members_age - age_reduction
  let average_age_at_birth : ℝ := total_age_at_birth / family_size
  average_age_at_birth = 10 := by
sorry

end NUMINAMATH_CALUDE_family_age_problem_l323_32351


namespace NUMINAMATH_CALUDE_books_added_to_bin_l323_32373

theorem books_added_to_bin (initial_books : ℕ) (books_sold : ℕ) (final_books : ℕ)
  (h1 : initial_books = 4)
  (h2 : books_sold = 3)
  (h3 : final_books = 11)
  (h4 : initial_books ≥ books_sold) :
  final_books - (initial_books - books_sold) = 10 := by
  sorry

end NUMINAMATH_CALUDE_books_added_to_bin_l323_32373


namespace NUMINAMATH_CALUDE_sweater_shirt_price_difference_l323_32382

theorem sweater_shirt_price_difference : 
  let shirt_total : ℚ := 400
  let shirt_count : ℕ := 25
  let sweater_total : ℚ := 1500
  let sweater_count : ℕ := 75
  let shirt_avg : ℚ := shirt_total / shirt_count
  let sweater_avg : ℚ := sweater_total / sweater_count
  sweater_avg - shirt_avg = 4 := by
sorry

end NUMINAMATH_CALUDE_sweater_shirt_price_difference_l323_32382


namespace NUMINAMATH_CALUDE_power_equation_l323_32321

theorem power_equation (x a b : ℝ) (h1 : x^a = 2) (h2 : x^b = 3) : x^(3*a + 2*b) = 72 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_l323_32321


namespace NUMINAMATH_CALUDE_animus_tower_beavers_l323_32390

/-- The number of beavers hired for the Animus Tower project -/
def num_beavers : ℕ := 862 - 544

/-- The total number of workers hired for the Animus Tower project -/
def total_workers : ℕ := 862

/-- The number of spiders hired for the Animus Tower project -/
def num_spiders : ℕ := 544

theorem animus_tower_beavers : num_beavers = 318 := by
  sorry

end NUMINAMATH_CALUDE_animus_tower_beavers_l323_32390


namespace NUMINAMATH_CALUDE_lemonade_stand_cost_l323_32304

theorem lemonade_stand_cost (net_profit babysitting_income : ℝ)
  (gross_revenue num_lemonades : ℕ)
  (lemon_cost sugar_cost ice_cost : ℝ)
  (bulk_discount sales_tax sunhat_cost : ℝ) :
  net_profit = 44 →
  babysitting_income = 31 →
  gross_revenue = 47 →
  num_lemonades = 50 →
  lemon_cost = 0.20 →
  sugar_cost = 0.15 →
  ice_cost = 0.05 →
  bulk_discount = 0.10 →
  sales_tax = 0.05 →
  sunhat_cost = 10 →
  ∃ (total_cost : ℝ),
    total_cost = (num_lemonades * (lemon_cost + sugar_cost + ice_cost) -
      num_lemonades * (lemon_cost + sugar_cost) * bulk_discount +
      gross_revenue * sales_tax + sunhat_cost) ∧
    total_cost = 30.60 :=
by sorry

end NUMINAMATH_CALUDE_lemonade_stand_cost_l323_32304


namespace NUMINAMATH_CALUDE_rectangular_field_fencing_l323_32394

theorem rectangular_field_fencing (area : ℝ) (fencing : ℝ) :
  area = 680 ∧ fencing = 74 →
  ∃ (length width : ℝ),
    length > 0 ∧ width > 0 ∧
    area = length * width ∧
    fencing = 2 * width + length ∧
    length = 40 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_fencing_l323_32394


namespace NUMINAMATH_CALUDE_union_A_B_l323_32380

def A : Set ℤ := {0, 1}

def B : Set ℤ := {x | (x + 2) * (x - 1) < 0}

theorem union_A_B : A ∪ B = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_union_A_B_l323_32380


namespace NUMINAMATH_CALUDE_cone_height_ratio_l323_32349

theorem cone_height_ratio (base_circumference : Real) (original_height : Real) (new_volume : Real) :
  base_circumference = 24 * Real.pi →
  original_height = 36 →
  new_volume = 432 * Real.pi →
  ∃ (new_height : Real),
    (1 / 3) * Real.pi * ((base_circumference / (2 * Real.pi)) ^ 2) * new_height = new_volume ∧
    new_height / original_height = 1 / 4 :=
by sorry

end NUMINAMATH_CALUDE_cone_height_ratio_l323_32349


namespace NUMINAMATH_CALUDE_symmetry_of_point_wrt_y_axis_l323_32317

/-- A point in a 2D Cartesian coordinate system -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- The y-axis symmetry operation on a point -/
def yAxisSymmetry (p : Point2D) : Point2D :=
  { x := -p.x, y := p.y }

theorem symmetry_of_point_wrt_y_axis :
  let P : Point2D := { x := 2, y := 1 }
  yAxisSymmetry P = { x := -2, y := 1 } := by
  sorry

end NUMINAMATH_CALUDE_symmetry_of_point_wrt_y_axis_l323_32317


namespace NUMINAMATH_CALUDE_eleventhDrawnNumber_l323_32305

/-- Systematic sampling function -/
def systematicSample (totalParticipants : ℕ) (sampleSize : ℕ) (firstDrawn : ℕ) (n : ℕ) : ℕ :=
  firstDrawn + (n - 1) * (totalParticipants / sampleSize)

/-- Theorem: 11th number drawn in the systematic sampling -/
theorem eleventhDrawnNumber (totalParticipants : ℕ) (sampleSize : ℕ) (firstDrawn : ℕ) :
  totalParticipants = 1000 →
  sampleSize = 50 →
  firstDrawn = 15 →
  systematicSample totalParticipants sampleSize firstDrawn 11 = 215 := by
  sorry

#check eleventhDrawnNumber

end NUMINAMATH_CALUDE_eleventhDrawnNumber_l323_32305


namespace NUMINAMATH_CALUDE_polygon_sides_l323_32341

theorem polygon_sides (S : ℝ) (h : S = 1080) :
  ∃ n : ℕ, n > 2 ∧ (n - 2) * 180 = S ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l323_32341


namespace NUMINAMATH_CALUDE_weak_arithmetic_progression_of_three_weak_arithmetic_progression_in_large_subset_l323_32354

-- Definition of weak arithmetic progression
def is_weak_arithmetic_progression (a : Fin M → ℝ) : Prop :=
  ∃ x : Fin (M + 1) → ℝ, ∀ i : Fin M, x i ≤ a i ∧ a i < x (i + 1)

-- Part (a)
theorem weak_arithmetic_progression_of_three (a₁ a₂ a₃ : ℝ) (h : a₁ < a₂ ∧ a₂ < a₃) :
  is_weak_arithmetic_progression (fun i => [a₁, a₂, a₃].get i) :=
sorry

-- Part (b)
theorem weak_arithmetic_progression_in_large_subset :
  ∀ (S : Finset (Fin 1000)),
    S.card ≥ 730 →
    ∃ (a : Fin 10 → Fin 1000), (∀ i, a i ∈ S) ∧ is_weak_arithmetic_progression (fun i => (a i : ℝ)) :=
sorry

end NUMINAMATH_CALUDE_weak_arithmetic_progression_of_three_weak_arithmetic_progression_in_large_subset_l323_32354


namespace NUMINAMATH_CALUDE_special_triples_characterization_l323_32327

/-- A quadratic function with integer coefficients -/
def QuadraticFunction (a b c : ℤ) : ℤ → ℤ := fun x ↦ a * x^2 + b * x + c

/-- The property that for any integer n, there exists an integer m such that f(m) = f(n)f(n+1) -/
def HasSpecialProperty (f : ℤ → ℤ) : Prop :=
  ∀ n : ℤ, ∃ m : ℤ, f m = f n * f (n + 1)

/-- The set of all integer triples (a, b, c) satisfying the special property -/
def SpecialTriples : Set (ℤ × ℤ × ℤ) :=
  {abc | let (a, b, c) := abc
         a ≠ 0 ∧ HasSpecialProperty (QuadraticFunction a b c)}

/-- The characterization of the special triples -/
def CharacterizedTriples : Set (ℤ × ℤ × ℤ) :=
  {abc | let (a, b, c) := abc
         (a = 1) ∨
         (∃ k l : ℤ, k > 0 ∧ a = k^2 ∧ b = 2*k*l ∧ c = l^2 ∧
          (k ∣ (l^2 - l) ∨ k ∣ (l^2 + l)))}

theorem special_triples_characterization :
  SpecialTriples = CharacterizedTriples :=
sorry


end NUMINAMATH_CALUDE_special_triples_characterization_l323_32327


namespace NUMINAMATH_CALUDE_power_equality_l323_32361

theorem power_equality : (16 : ℕ) ^ 5 = (4 : ℕ) ^ 10 := by sorry

end NUMINAMATH_CALUDE_power_equality_l323_32361


namespace NUMINAMATH_CALUDE_proportion_equality_l323_32309

theorem proportion_equality : (5 / 34) / (7 / 48) = (120 / 1547) / (1 / 13) := by
  sorry

end NUMINAMATH_CALUDE_proportion_equality_l323_32309


namespace NUMINAMATH_CALUDE_defective_units_shipped_for_sale_l323_32310

/-- 
Given that 4% of units produced are defective and 0.16% of units produced
are defective units shipped for sale, prove that 4% of defective units
are shipped for sale.
-/
theorem defective_units_shipped_for_sale 
  (total_units : ℝ) 
  (defective_rate : ℝ) 
  (defective_shipped_rate : ℝ) 
  (h1 : defective_rate = 0.04) 
  (h2 : defective_shipped_rate = 0.0016) : 
  defective_shipped_rate / defective_rate = 0.04 := by
  sorry

end NUMINAMATH_CALUDE_defective_units_shipped_for_sale_l323_32310


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_difference_absolute_value_equation_solution_difference_proof_l323_32379

theorem absolute_value_equation_solution_difference : ℝ → Prop :=
  fun d => ∃ x y : ℝ,
    (|x - 3| = 15 ∧ |y - 3| = 15) ∧
    x ≠ y ∧
    d = |x - y| ∧
    d = 30

-- The proof is omitted
theorem absolute_value_equation_solution_difference_proof :
  ∃ d : ℝ, absolute_value_equation_solution_difference d :=
by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_difference_absolute_value_equation_solution_difference_proof_l323_32379


namespace NUMINAMATH_CALUDE_difference_value_l323_32307

-- Define the variables x and y
variable (x y : ℝ)

-- Define the conditions
def sum_condition : Prop := x + y = 500
def ratio_condition : Prop := x / y = 0.8

-- Define the theorem
theorem difference_value (h1 : sum_condition x y) (h2 : ratio_condition x y) :
  ∃ ε > 0, |y - x - 55.56| < ε :=
sorry

end NUMINAMATH_CALUDE_difference_value_l323_32307


namespace NUMINAMATH_CALUDE_transfer_schemes_count_l323_32344

/-- The number of torchbearers and segments in the relay --/
def n : ℕ := 6

/-- The set of possible first torchbearers --/
inductive FirstTorchbearer
| A
| B
| C

/-- The set of possible last torchbearers --/
inductive LastTorchbearer
| A
| B

/-- A function to calculate the number of transfer schemes --/
def countTransferSchemes : ℕ :=
  let firstChoices := 3  -- A, B, or C
  let lastChoices := 2   -- A or B
  let middleArrangements := Nat.factorial (n - 2)
  firstChoices * lastChoices * middleArrangements

/-- Theorem stating that the number of transfer schemes is 96 --/
theorem transfer_schemes_count :
  countTransferSchemes = 96 := by
  sorry

end NUMINAMATH_CALUDE_transfer_schemes_count_l323_32344


namespace NUMINAMATH_CALUDE_tan_225_degrees_equals_one_l323_32356

theorem tan_225_degrees_equals_one : Real.tan (225 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_225_degrees_equals_one_l323_32356


namespace NUMINAMATH_CALUDE_calculation_proof_l323_32363

theorem calculation_proof :
  (2 * (-5) + 2^3 - 3 + (1/2 : ℚ) = -15/2) ∧
  (-3^2 * (-1/3)^2 + (3/4 - 1/6 + 3/8) * (-24) = -24) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l323_32363


namespace NUMINAMATH_CALUDE_simplify_expression_solve_fractional_equation_l323_32302

-- Problem 1
theorem simplify_expression (x : ℝ) (hx : x ≠ 0) :
  (12 * x^4 + 6 * x^2) / (3 * x) - (-2 * x)^2 * (x + 1) = 2 * x - 4 * x^2 := by
  sorry

-- Problem 2
theorem solve_fractional_equation :
  ∃ x : ℝ, x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1 ∧ 5 / (x^2 + x) - 1 / (x^2 - x) = 0 ∧ x = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_solve_fractional_equation_l323_32302


namespace NUMINAMATH_CALUDE_expression_simplification_l323_32368

theorem expression_simplification :
  3 + Real.sqrt 3 + 1 / (3 + Real.sqrt 3) + 1 / (Real.sqrt 3 - 3) = 3 + (2 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l323_32368


namespace NUMINAMATH_CALUDE_smallest_n_for_candy_purchase_l323_32314

/-- The smallest positive integer n such that 15n is divisible by 10, 18, and 25 is 30 -/
theorem smallest_n_for_candy_purchase : 
  (∀ k : ℕ+, k < 30 → ¬(10 ∣ 15*k ∧ 18 ∣ 15*k ∧ 25 ∣ 15*k)) ∧ 
  (10 ∣ 15*30 ∧ 18 ∣ 15*30 ∧ 25 ∣ 15*30) := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_candy_purchase_l323_32314


namespace NUMINAMATH_CALUDE_power_of_fraction_l323_32339

theorem power_of_fraction (x y : ℝ) : 
  (-(1/3) * x^2 * y)^3 = -(x^6 * y^3) / 27 := by
  sorry

end NUMINAMATH_CALUDE_power_of_fraction_l323_32339


namespace NUMINAMATH_CALUDE_comparison_theorem_l323_32330

theorem comparison_theorem (n : ℕ) (h : n ≥ 4) : 3 * 2^(n - 1) > n^2 + 3 := by
  sorry

end NUMINAMATH_CALUDE_comparison_theorem_l323_32330


namespace NUMINAMATH_CALUDE_first_1000_decimals_are_zero_l323_32378

theorem first_1000_decimals_are_zero (a : ℕ) (n : ℕ) 
    (ha : a = 35 ∨ a = 37) (hn : n = 1999 ∨ n = 2000) :
  ∃ (k : ℕ), (6 + Real.sqrt a)^n = k + (1 / 10^1000) * (Real.sqrt a) := by
  sorry

end NUMINAMATH_CALUDE_first_1000_decimals_are_zero_l323_32378


namespace NUMINAMATH_CALUDE_fraction_relation_l323_32395

theorem fraction_relation (a b c : ℚ) 
  (h1 : a / b = 2) 
  (h2 : b / c = 4 / 3) : 
  c / a = 3 / 8 := by
sorry

end NUMINAMATH_CALUDE_fraction_relation_l323_32395
