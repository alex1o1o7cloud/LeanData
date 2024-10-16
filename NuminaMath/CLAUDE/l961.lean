import Mathlib

namespace NUMINAMATH_CALUDE_machine_production_time_difference_l961_96175

/-- Given two machines X and Y that produce widgets, this theorem proves
    that machine X takes 2 days longer than machine Y to produce W widgets. -/
theorem machine_production_time_difference
  (W : ℝ) -- W represents the number of widgets
  (h1 : (W / 6 + W / 4) * 3 = 5 * W / 4) -- Combined production in 3 days
  (h2 : W / 6 * 18 = 3 * W) -- Machine X production in 18 days
  : (W / (W / 6)) - (W / (W / 4)) = 2 :=
sorry

end NUMINAMATH_CALUDE_machine_production_time_difference_l961_96175


namespace NUMINAMATH_CALUDE_circle_center_radius_sum_l961_96179

/-- Given a circle D with equation x² - 4y - 36 = -y² + 14x + 4,
    prove that its center (c, d) and radius s satisfy c + d + s = 9 + √93 -/
theorem circle_center_radius_sum (x y c d s : ℝ) : 
  (∀ x y, x^2 - 4*y - 36 = -y^2 + 14*x + 4) →
  (c = 7 ∧ d = 2) →
  (∀ x y, (x - c)^2 + (y - d)^2 = s^2) →
  c + d + s = 9 + Real.sqrt 93 := by
sorry

end NUMINAMATH_CALUDE_circle_center_radius_sum_l961_96179


namespace NUMINAMATH_CALUDE_triangle_problem_l961_96158

theorem triangle_problem (A B C a b c : ℝ) : 
  a ≠ b →
  c = Real.sqrt 7 →
  b * Real.sin B - a * Real.sin A = Real.sqrt 3 * a * Real.cos A - Real.sqrt 3 * b * Real.cos B →
  (1/2) * a * b * Real.sin C = (3 * Real.sqrt 3) / 2 →
  C = π / 3 ∧ ((a = 2 ∧ b = 3) ∨ (a = 3 ∧ b = 2)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l961_96158


namespace NUMINAMATH_CALUDE_percentage_difference_l961_96126

theorem percentage_difference (x y : ℝ) (h : x = 4 * y) :
  (x - y) / x * 100 = 75 :=
by sorry

end NUMINAMATH_CALUDE_percentage_difference_l961_96126


namespace NUMINAMATH_CALUDE_min_time_less_than_3_9_l961_96167

/-- The walking speed of each person in km/h -/
def walking_speed : ℝ := 6

/-- The speed of the motorcycle in km/h -/
def motorcycle_speed : ℝ := 90

/-- The total distance to be covered in km -/
def total_distance : ℝ := 135

/-- The maximum number of people the motorcycle can carry -/
def max_motorcycle_capacity : ℕ := 2

/-- The number of people travelling -/
def num_people : ℕ := 3

/-- The minimum time required for all people to reach the destination -/
noncomputable def min_time : ℝ := 
  (23 * total_distance) / (9 * motorcycle_speed)

theorem min_time_less_than_3_9 : min_time < 3.9 := by
  sorry

end NUMINAMATH_CALUDE_min_time_less_than_3_9_l961_96167


namespace NUMINAMATH_CALUDE_equation_solution_l961_96151

theorem equation_solution : ∃ x : ℝ, 35 - (23 - (15 - x)) = 12 * 2 / (1 / 2) ∧ x = -21 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l961_96151


namespace NUMINAMATH_CALUDE_power_sum_equality_l961_96122

theorem power_sum_equality : (-1)^43 + 2^(2^3 + 5^2 - 7^2) = -(65535 / 65536) := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equality_l961_96122


namespace NUMINAMATH_CALUDE_gwen_birthday_money_l961_96182

/-- Given that Gwen received 14 dollars for her birthday and spent 8 dollars,
    prove that she has 6 dollars left. -/
theorem gwen_birthday_money (received : ℕ) (spent : ℕ) (left : ℕ) : 
  received = 14 → spent = 8 → left = received - spent → left = 6 := by
  sorry

end NUMINAMATH_CALUDE_gwen_birthday_money_l961_96182


namespace NUMINAMATH_CALUDE_mirella_orange_books_l961_96170

/-- The number of pages in each purple book -/
def purple_pages : ℕ := 230

/-- The number of pages in each orange book -/
def orange_pages : ℕ := 510

/-- The number of purple books Mirella read -/
def purple_books_read : ℕ := 5

/-- The difference between orange and purple pages Mirella read -/
def page_difference : ℕ := 890

/-- The number of orange books Mirella read -/
def orange_books_read : ℕ := 4

theorem mirella_orange_books :
  orange_books_read * orange_pages = 
  purple_books_read * purple_pages + page_difference := by
  sorry

end NUMINAMATH_CALUDE_mirella_orange_books_l961_96170


namespace NUMINAMATH_CALUDE_complex_product_equality_l961_96123

theorem complex_product_equality : (3 + 4*Complex.I) * (2 - 3*Complex.I) * (1 + 2*Complex.I) = 20 + 35*Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_product_equality_l961_96123


namespace NUMINAMATH_CALUDE_airplane_distance_theorem_l961_96109

/-- Calculates the distance traveled by an airplane given its speed and time. -/
def airplane_distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Proves that an airplane flying for 38 hours at 30 miles per hour travels 1140 miles. -/
theorem airplane_distance_theorem :
  let speed : ℝ := 30
  let time : ℝ := 38
  airplane_distance speed time = 1140 := by
  sorry

end NUMINAMATH_CALUDE_airplane_distance_theorem_l961_96109


namespace NUMINAMATH_CALUDE_function_not_in_third_quadrant_l961_96186

theorem function_not_in_third_quadrant 
  (a b : ℝ) (ha : 0 < a) (ha' : a < 1) (hb : 0 < b) (hb' : b < 1) :
  ∀ x : ℝ, x < 0 → a^x + b - 1 > 0 := by
  sorry

end NUMINAMATH_CALUDE_function_not_in_third_quadrant_l961_96186


namespace NUMINAMATH_CALUDE_non_congruent_squares_on_6x6_grid_l961_96162

/-- A square on a lattice grid --/
structure LatticeSquare where
  side_length : ℕ
  is_diagonal : Bool

/-- The size of the grid --/
def grid_size : ℕ := 6

/-- Counts the number of squares with a given side length that fit on the grid --/
def count_squares (s : ℕ) : ℕ :=
  (grid_size - s + 1) ^ 2

/-- Counts all non-congruent squares on the grid --/
def total_non_congruent_squares : ℕ :=
  (List.range 5).map (λ i => count_squares (i + 1)) |> List.sum

/-- The main theorem stating the number of non-congruent squares on a 6x6 grid --/
theorem non_congruent_squares_on_6x6_grid :
  total_non_congruent_squares = 110 := by
  sorry


end NUMINAMATH_CALUDE_non_congruent_squares_on_6x6_grid_l961_96162


namespace NUMINAMATH_CALUDE_librarian_took_two_books_l961_96114

/-- The number of books the librarian took -/
def librarian_took (total_books : ℕ) (shelves_needed : ℕ) (books_per_shelf : ℕ) : ℕ :=
  total_books - (shelves_needed * books_per_shelf)

/-- Theorem stating that the librarian took 2 books -/
theorem librarian_took_two_books :
  librarian_took 14 4 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_librarian_took_two_books_l961_96114


namespace NUMINAMATH_CALUDE_no_special_two_digit_primes_l961_96119

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → n % d ≠ 0

def reverse_digits (n : ℕ) : ℕ :=
  let tens := n / 10
  let ones := n % 10
  ones * 10 + tens

def digit_sum (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

theorem no_special_two_digit_primes :
  ∀ n : ℕ, 10 ≤ n → n < 100 →
    is_prime n ∧ is_prime (reverse_digits n) ∧ is_prime (digit_sum n) → False :=
sorry

end NUMINAMATH_CALUDE_no_special_two_digit_primes_l961_96119


namespace NUMINAMATH_CALUDE_area_of_similar_rectangle_l961_96134

-- Define the properties of rectangle R1
def side_R1 : ℝ := 3
def area_R1 : ℝ := 18

-- Define the diagonal of rectangle R2
def diagonal_R2 : ℝ := 20

-- Theorem statement
theorem area_of_similar_rectangle (side_R1 area_R1 diagonal_R2 : ℝ) 
  (h1 : side_R1 > 0)
  (h2 : area_R1 > 0)
  (h3 : diagonal_R2 > 0) :
  let other_side_R1 := area_R1 / side_R1
  let ratio := other_side_R1 / side_R1
  let side_R2 := (diagonal_R2^2 / (1 + ratio^2))^(1/2)
  side_R2 * (ratio * side_R2) = 160 := by
sorry

end NUMINAMATH_CALUDE_area_of_similar_rectangle_l961_96134


namespace NUMINAMATH_CALUDE_calculation_result_l961_96199

theorem calculation_result : (377 / 13) / 29 * (1 / 4) / 2 = 0.125 := by
  sorry

end NUMINAMATH_CALUDE_calculation_result_l961_96199


namespace NUMINAMATH_CALUDE_tower_has_four_levels_l961_96146

/-- Calculates the number of levels in a tower given the number of steps per level,
    blocks per step, and total blocks climbed. -/
def tower_levels (steps_per_level : ℕ) (blocks_per_step : ℕ) (total_blocks : ℕ) : ℕ :=
  (total_blocks / blocks_per_step) / steps_per_level

/-- Theorem stating that a tower with 8 steps per level, 3 blocks per step,
    and 96 total blocks climbed has 4 levels. -/
theorem tower_has_four_levels :
  tower_levels 8 3 96 = 4 := by
  sorry

end NUMINAMATH_CALUDE_tower_has_four_levels_l961_96146


namespace NUMINAMATH_CALUDE_hyperbola_line_intersection_fixed_point_l961_96101

/-- Represents a hyperbola with center at origin -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  e : ℝ

/-- Represents a line in slope-intercept form -/
structure Line where
  k : ℝ
  m : ℝ

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

def Hyperbola.standard_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / h.a^2 - y^2 / h.b^2 = 1

def Line.equation (l : Line) (x y : ℝ) : Prop :=
  y = l.k * x + l.m

def circle_diameter_passes_through (A B D : Point) : Prop :=
  (A.y / (A.x + D.x)) * (B.y / (B.x + D.x)) = -1

theorem hyperbola_line_intersection_fixed_point
  (h : Hyperbola)
  (l : Line)
  (A B D : Point) :
  h.a = 2 →
  h.b = 1 →
  h.e = Real.sqrt 5 / 2 →
  Hyperbola.standard_equation h A.x A.y →
  Hyperbola.standard_equation h B.x B.y →
  Line.equation l A.x A.y →
  Line.equation l B.x B.y →
  D.x = -2 →
  D.y = 0 →
  A ≠ D →
  B ≠ D →
  circle_diameter_passes_through A B D →
  ∃ P : Point, P.x = -10/3 ∧ P.y = 0 ∧ Line.equation l P.x P.y :=
sorry

end NUMINAMATH_CALUDE_hyperbola_line_intersection_fixed_point_l961_96101


namespace NUMINAMATH_CALUDE_max_cut_length_30x30_l961_96166

/-- Represents a square board with side length and number of parts it's cut into -/
structure Board :=
  (side : ℕ)
  (parts : ℕ)

/-- Calculates the maximum total length of cuts for a given board -/
def max_cut_length (b : Board) : ℕ :=
  sorry

/-- Theorem stating the maximum cut length for a 30x30 board cut into 225 parts -/
theorem max_cut_length_30x30 :
  let b : Board := ⟨30, 225⟩
  max_cut_length b = 1065 := by
  sorry

end NUMINAMATH_CALUDE_max_cut_length_30x30_l961_96166


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l961_96180

/-- Given a geometric sequence {a_n} where a₄ + a₆ = 3, prove that a₅(a₃ + 2a₅ + a₇) = 9 -/
theorem geometric_sequence_property (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- a_n is a geometric sequence with common ratio q
  a 4 + a 6 = 3 →               -- given condition
  a 5 * (a 3 + 2 * a 5 + a 7) = 9 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l961_96180


namespace NUMINAMATH_CALUDE_modulus_of_complex_l961_96177

theorem modulus_of_complex (i : ℂ) : i * i = -1 → Complex.abs (2 * i - 5 / (2 - i)) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_l961_96177


namespace NUMINAMATH_CALUDE_circle_and_line_properties_l961_96161

-- Define the circle C
def circle_C (x y b : ℝ) : Prop := (x + 2)^2 + (y - b)^2 = 3

-- Define the line l
def line_l (x y m : ℝ) : Prop := y = x + m

-- Define the point that the circle passes through
def point_on_circle (b : ℝ) : Prop := circle_C (-2 + Real.sqrt 2) 0 b

-- Define the tangency condition
def is_tangent (m b : ℝ) : Prop :=
  (|(-2) - 1 + m| / Real.sqrt 2) = Real.sqrt 3

-- Define the perpendicular condition
def is_perpendicular (m : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂ : ℝ,
    circle_C x₁ y₁ 1 ∧ circle_C x₂ y₂ 1 ∧
    line_l x₁ y₁ m ∧ line_l x₂ y₂ m ∧
    x₁ * x₂ + y₁ * y₂ = 0

theorem circle_and_line_properties :
  (∃ b : ℝ, b > 0 ∧ point_on_circle b) ∧
  (∃ m : ℝ, is_tangent m 1) ∧
  (∃ m : ℝ, is_perpendicular m) :=
sorry

end NUMINAMATH_CALUDE_circle_and_line_properties_l961_96161


namespace NUMINAMATH_CALUDE_reflection_across_x_axis_l961_96128

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- The original point in the standard coordinate system -/
def original_point : ℝ × ℝ := (-2, 3)

theorem reflection_across_x_axis :
  reflect_x original_point = (-2, -3) := by sorry

end NUMINAMATH_CALUDE_reflection_across_x_axis_l961_96128


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l961_96190

theorem sum_of_squares_of_roots (a b c : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c
  let x₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  a = 2 ∧ b = -6 ∧ c = 4 → x₁^2 + x₂^2 = 5 :=
by sorry


end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l961_96190


namespace NUMINAMATH_CALUDE_allocation_schemes_count_l961_96192

/-- The number of ways to divide 6 volunteers into 4 groups and assign them to venues -/
def allocationSchemes : ℕ :=
  let n := 6  -- number of volunteers
  let k := 4  -- number of groups/venues
  let g₂ := 2  -- number of groups with 2 people
  let g₁ := 2  -- number of groups with 1 person
  540

/-- Theorem stating that the number of allocation schemes is 540 -/
theorem allocation_schemes_count : allocationSchemes = 540 := by
  sorry

end NUMINAMATH_CALUDE_allocation_schemes_count_l961_96192


namespace NUMINAMATH_CALUDE_square_sum_geq_product_sum_l961_96141

theorem square_sum_geq_product_sum (a b : ℝ) : a^2 + b^2 ≥ a*b + a + b - 1 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_geq_product_sum_l961_96141


namespace NUMINAMATH_CALUDE_unique_n_exists_l961_96136

theorem unique_n_exists : ∃! n : ℤ,
  50 ≤ n ∧ n ≤ 200 ∧
  8 ∣ n ∧
  n % 6 = 4 ∧
  n % 7 = 3 ∧
  n = 136 := by
sorry

end NUMINAMATH_CALUDE_unique_n_exists_l961_96136


namespace NUMINAMATH_CALUDE_range_sin_plus_cos_range_sin_plus_cos_minus_sin_2x_l961_96142

-- Part 1
theorem range_sin_plus_cos :
  Set.range (fun x : ℝ => Real.sin x + Real.cos x) = Set.Icc (-Real.sqrt 2) (Real.sqrt 2) := by
sorry

-- Part 2
theorem range_sin_plus_cos_minus_sin_2x :
  Set.range (fun x : ℝ => Real.sin x + Real.cos x - Real.sin (2 * x)) = Set.Icc (-1 - Real.sqrt 2) (5/4) := by
sorry

end NUMINAMATH_CALUDE_range_sin_plus_cos_range_sin_plus_cos_minus_sin_2x_l961_96142


namespace NUMINAMATH_CALUDE_min_value_of_squares_l961_96130

theorem min_value_of_squares (a b t : ℝ) (h : 2 * a + b = 2 * t) :
  ∃ (min : ℝ), min = (4 * t^2) / 5 ∧ ∀ (x y : ℝ), 2 * x + y = 2 * t → x^2 + y^2 ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_of_squares_l961_96130


namespace NUMINAMATH_CALUDE_log_inequality_l961_96154

theorem log_inequality (x : ℝ) (h : x > 0) : Real.log (1 + x) < x / (2 + x) := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l961_96154


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l961_96197

theorem quadratic_roots_relation (b c : ℚ) : 
  (∃ r s : ℚ, 4 * r^2 - 7 * r - 10 = 0 ∧ 4 * s^2 - 7 * s - 10 = 0 ∧
   ∀ x : ℚ, x^2 + b * x + c = 0 ↔ (x = r + 3 ∨ x = s + 3)) →
  c = 47 / 4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l961_96197


namespace NUMINAMATH_CALUDE_overlap_area_of_circles_l961_96127

/-- The area of overlap between two circles with given properties -/
theorem overlap_area_of_circles (r : ℝ) (overlap_percentage : ℝ) : 
  r = 10 ∧ overlap_percentage = 0.25 → 
  2 * (25 * Real.pi - 50) = 
    2 * ((overlap_percentage * 2 * Real.pi * r^2 / 4) - (r^2 / 2)) := by
  sorry

end NUMINAMATH_CALUDE_overlap_area_of_circles_l961_96127


namespace NUMINAMATH_CALUDE_inequality_proof_l961_96111

theorem inequality_proof (a b : ℝ) (h1 : a < 1) (h2 : b < 1) (h3 : a + b ≥ 1/3) :
  (1 - a) * (1 - b) ≤ 25/36 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l961_96111


namespace NUMINAMATH_CALUDE_unique_solution_exists_l961_96171

theorem unique_solution_exists : ∃! (a b : ℝ), 2 * a + b = 7 ∧ a - b = 2 := by sorry

end NUMINAMATH_CALUDE_unique_solution_exists_l961_96171


namespace NUMINAMATH_CALUDE_no_real_solutions_l961_96113

theorem no_real_solutions : ¬∃ (x : ℝ), x + 48 / (x - 3) = -1 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l961_96113


namespace NUMINAMATH_CALUDE_b_share_is_302_l961_96163

/-- Given a division of money among five people A, B, C, D, and E, 
    prove that B's share is 302 rupees. -/
theorem b_share_is_302 
  (total : ℕ) 
  (share_a share_b share_c share_d share_e : ℕ) 
  (h_total : total = 1540)
  (h_a : share_a = share_b + 40)
  (h_c : share_c = share_a + 30)
  (h_d : share_d = share_b - 50)
  (h_e : share_e = share_d + 20)
  (h_sum : share_a + share_b + share_c + share_d + share_e = total) : 
  share_b = 302 := by
  sorry


end NUMINAMATH_CALUDE_b_share_is_302_l961_96163


namespace NUMINAMATH_CALUDE_binomial_12_10_l961_96178

theorem binomial_12_10 : Nat.choose 12 10 = 66 := by
  sorry

end NUMINAMATH_CALUDE_binomial_12_10_l961_96178


namespace NUMINAMATH_CALUDE_expected_black_pairs_in_circular_deal_l961_96137

/-- Represents a standard deck of 52 cards -/
def StandardDeck : ℕ := 52

/-- Number of black cards in a standard deck -/
def BlackCards : ℕ := 26

/-- Expected number of pairs of adjacent black cards in a circular deal -/
def ExpectedBlackPairs : ℚ := 650 / 51

theorem expected_black_pairs_in_circular_deal :
  let total_cards := StandardDeck
  let black_cards := BlackCards
  let prob_next_black : ℚ := (black_cards - 1) / (total_cards - 1)
  black_cards * prob_next_black = ExpectedBlackPairs :=
sorry

end NUMINAMATH_CALUDE_expected_black_pairs_in_circular_deal_l961_96137


namespace NUMINAMATH_CALUDE_subtracted_value_l961_96150

theorem subtracted_value (x y : ℤ) (h1 : x = 122) (h2 : 2 * x - y = 106) : y = 138 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_value_l961_96150


namespace NUMINAMATH_CALUDE_lcm_of_54_and_16_l961_96105

theorem lcm_of_54_and_16 : Nat.lcm 54 16 = 48 :=
by
  have h1 : Nat.gcd 54 16 = 18 := by sorry
  sorry

end NUMINAMATH_CALUDE_lcm_of_54_and_16_l961_96105


namespace NUMINAMATH_CALUDE_smallest_first_term_of_arithmetic_sequence_l961_96118

def arithmetic_sequence (c₁ : ℚ) (d : ℚ) : ℕ → ℚ
  | 0 => c₁
  | n+1 => arithmetic_sequence c₁ d n + d

def sum_of_terms (c₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (2 * c₁ + (n - 1 : ℚ) * d) / 2

theorem smallest_first_term_of_arithmetic_sequence :
  ∃ (c₁ d : ℚ),
    (c₁ ≥ 1/3) ∧
    (∃ (S₃ S₇ : ℕ),
      sum_of_terms c₁ d 3 = S₃ ∧
      sum_of_terms c₁ d 7 = S₇) ∧
    (∀ (c₁' d' : ℚ),
      (c₁' ≥ 1/3) →
      (∃ (S₃' S₇' : ℕ),
        sum_of_terms c₁' d' 3 = S₃' ∧
        sum_of_terms c₁' d' 7 = S₇') →
      c₁' ≥ c₁) ∧
    c₁ = 5/14 :=
sorry

end NUMINAMATH_CALUDE_smallest_first_term_of_arithmetic_sequence_l961_96118


namespace NUMINAMATH_CALUDE_probability_four_ones_twelve_dice_l961_96102

theorem probability_four_ones_twelve_dice :
  let n : ℕ := 12  -- total number of dice
  let k : ℕ := 4   -- number of dice showing 1
  let p : ℚ := 1/6 -- probability of rolling a 1 on a single die
  
  let probability := (n.choose k) * (p ^ k) * ((1 - p) ^ (n - k))
  
  probability = 495 * (5^8 : ℚ) / (6^12 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_probability_four_ones_twelve_dice_l961_96102


namespace NUMINAMATH_CALUDE_ambulance_ride_cost_ambulance_cost_proof_l961_96112

/-- Calculates the cost of an ambulance ride given a hospital bill breakdown -/
theorem ambulance_ride_cost (total_bill : ℝ) (medication_percentage : ℝ) 
  (overnight_percentage : ℝ) (food_cost : ℝ) : ℝ :=
  let medication_cost := medication_percentage * total_bill
  let remaining_after_medication := total_bill - medication_cost
  let overnight_cost := overnight_percentage * remaining_after_medication
  let ambulance_cost := total_bill - medication_cost - overnight_cost - food_cost
  ambulance_cost

/-- Proves that the ambulance ride cost is $1700 given specific bill details -/
theorem ambulance_cost_proof :
  ambulance_ride_cost 5000 0.5 0.25 175 = 1700 := by
  sorry

end NUMINAMATH_CALUDE_ambulance_ride_cost_ambulance_cost_proof_l961_96112


namespace NUMINAMATH_CALUDE_merchant_articles_l961_96121

/-- Represents the number of articles a merchant has -/
def N : ℕ := 20

/-- Represents the cost price of each article -/
def CP : ℝ := 1

/-- Represents the selling price of each article -/
def SP : ℝ := 1.25 * CP

theorem merchant_articles :
  (N * CP = 16 * SP) ∧ (SP = 1.25 * CP) → N = 20 := by
  sorry

end NUMINAMATH_CALUDE_merchant_articles_l961_96121


namespace NUMINAMATH_CALUDE_no_solution_iff_k_geq_two_l961_96183

theorem no_solution_iff_k_geq_two (k : ℝ) :
  (∀ x : ℝ, ¬(1 < x ∧ x ≤ 2 ∧ x > k)) ↔ k ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_iff_k_geq_two_l961_96183


namespace NUMINAMATH_CALUDE_revenue_decrease_percentage_l961_96156

def old_revenue : ℝ := 85.0
def new_revenue : ℝ := 48.0

theorem revenue_decrease_percentage :
  abs (((old_revenue - new_revenue) / old_revenue) * 100 - 43.53) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_revenue_decrease_percentage_l961_96156


namespace NUMINAMATH_CALUDE_brittany_second_test_score_l961_96149

/-- Proves that given the conditions of Brittany's test scores, her second test score must be 83. -/
theorem brittany_second_test_score
  (first_test_score : ℝ)
  (first_test_weight : ℝ)
  (second_test_weight : ℝ)
  (final_weighted_average : ℝ)
  (h1 : first_test_score = 78)
  (h2 : first_test_weight = 0.4)
  (h3 : second_test_weight = 0.6)
  (h4 : final_weighted_average = 81)
  (h5 : first_test_weight + second_test_weight = 1) :
  ∃ (second_test_score : ℝ),
    first_test_weight * first_test_score + second_test_weight * second_test_score = final_weighted_average ∧
    second_test_score = 83 :=
by sorry

end NUMINAMATH_CALUDE_brittany_second_test_score_l961_96149


namespace NUMINAMATH_CALUDE_green_hats_count_l961_96108

theorem green_hats_count (total_hats : ℕ) (blue_cost green_cost total_cost : ℕ) 
  (h1 : total_hats = 85)
  (h2 : blue_cost = 6)
  (h3 : green_cost = 7)
  (h4 : total_cost = 548) :
  ∃ (blue_hats green_hats : ℕ),
    blue_hats + green_hats = total_hats ∧
    blue_cost * blue_hats + green_cost * green_hats = total_cost ∧
    green_hats = 38 := by
sorry

end NUMINAMATH_CALUDE_green_hats_count_l961_96108


namespace NUMINAMATH_CALUDE_both_unsuccessful_correct_both_successful_correct_exactly_one_successful_correct_at_least_one_successful_correct_at_most_one_successful_correct_l961_96133

-- Define the propositions
variable (p q : Prop)

-- Define the shooting scenarios
def both_unsuccessful : Prop := ¬p ∧ ¬q
def both_successful : Prop := p ∧ q
def exactly_one_successful : Prop := (¬p ∧ q) ∨ (p ∧ ¬q)
def at_least_one_successful : Prop := p ∨ q
def at_most_one_successful : Prop := ¬(p ∧ q)

-- Theorem statements
theorem both_unsuccessful_correct (p q : Prop) : 
  both_unsuccessful p q ↔ ¬p ∧ ¬q := by sorry

theorem both_successful_correct (p q : Prop) : 
  both_successful p q ↔ p ∧ q := by sorry

theorem exactly_one_successful_correct (p q : Prop) : 
  exactly_one_successful p q ↔ (¬p ∧ q) ∨ (p ∧ ¬q) := by sorry

theorem at_least_one_successful_correct (p q : Prop) : 
  at_least_one_successful p q ↔ p ∨ q := by sorry

theorem at_most_one_successful_correct (p q : Prop) : 
  at_most_one_successful p q ↔ ¬(p ∧ q) := by sorry

end NUMINAMATH_CALUDE_both_unsuccessful_correct_both_successful_correct_exactly_one_successful_correct_at_least_one_successful_correct_at_most_one_successful_correct_l961_96133


namespace NUMINAMATH_CALUDE_yearly_increase_fraction_l961_96153

theorem yearly_increase_fraction (initial_value final_value : ℝ) (f : ℝ) 
    (h1 : initial_value = 51200)
    (h2 : final_value = 64800)
    (h3 : initial_value * (1 + f)^2 = final_value) :
  f = 0.125 := by
  sorry

end NUMINAMATH_CALUDE_yearly_increase_fraction_l961_96153


namespace NUMINAMATH_CALUDE_no_fourteen_consecutive_integers_exist_twentyone_consecutive_integers_l961_96160

/-- Defines a function that checks if a number is divisible by any prime in a given range -/
def divisible_by_prime_in_range (n : ℕ) (lower upper : ℕ) : Prop :=
  ∃ p, Prime p ∧ lower ≤ p ∧ p ≤ upper ∧ p ∣ n

/-- Theorem stating that there do not exist 14 consecutive positive integers
    each divisible by at least one prime p where 2 ≤ p ≤ 11 -/
theorem no_fourteen_consecutive_integers : ¬ ∃ start : ℕ, ∀ k : ℕ, k < 14 →
  divisible_by_prime_in_range (start + k) 2 11 := by sorry

/-- Theorem stating that there exist 21 consecutive positive integers
    each divisible by at least one prime p where 2 ≤ p ≤ 13 -/
theorem exist_twentyone_consecutive_integers : ∃ start : ℕ, ∀ k : ℕ, k < 21 →
  divisible_by_prime_in_range (start + k) 2 13 := by sorry

end NUMINAMATH_CALUDE_no_fourteen_consecutive_integers_exist_twentyone_consecutive_integers_l961_96160


namespace NUMINAMATH_CALUDE_max_triples_count_l961_96168

def N (n : ℕ) : ℕ := sorry

theorem max_triples_count (n : ℕ) (h : n ≥ 2) :
  N n = ⌊(2 * n : ℚ) / 3 + 1⌋ :=
by sorry

end NUMINAMATH_CALUDE_max_triples_count_l961_96168


namespace NUMINAMATH_CALUDE_new_year_markup_l961_96124

theorem new_year_markup (initial_markup : ℝ) (february_discount : ℝ) (final_profit : ℝ) :
  initial_markup = 0.2 →
  february_discount = 0.06 →
  final_profit = 0.41 →
  ∃ (new_year_markup : ℝ),
    (1 - february_discount) * (1 + new_year_markup) * (1 + initial_markup) = 1 + final_profit ∧
    new_year_markup = 0.5 :=
by sorry

end NUMINAMATH_CALUDE_new_year_markup_l961_96124


namespace NUMINAMATH_CALUDE_repeating_decimal_denominator_l961_96184

theorem repeating_decimal_denominator : ∃ (n : ℕ) (d : ℕ), d ≠ 0 ∧ (n / d : ℚ) = 0.6666666666666667 ∧ (∀ (n' : ℕ) (d' : ℕ), d' ≠ 0 → (n' / d' : ℚ) = (n / d : ℚ) → d' ≥ d) ∧ d = 3 :=
by sorry

end NUMINAMATH_CALUDE_repeating_decimal_denominator_l961_96184


namespace NUMINAMATH_CALUDE_eleven_percent_greater_than_80_l961_96174

theorem eleven_percent_greater_than_80 (x : ℝ) : 
  x = 80 * (1 + 11 / 100) → x = 88.8 := by
sorry

end NUMINAMATH_CALUDE_eleven_percent_greater_than_80_l961_96174


namespace NUMINAMATH_CALUDE_photo_arrangements_count_l961_96115

/-- Represents the number of students -/
def total_students : ℕ := 7

/-- Represents the number of students on each side of the tallest student -/
def students_per_side : ℕ := 3

/-- The number of possible arrangements of students for the photo -/
def num_arrangements : ℕ := Nat.choose (total_students - 1) students_per_side

/-- Theorem stating that the number of arrangements is correct -/
theorem photo_arrangements_count :
  num_arrangements = 20 :=
sorry

end NUMINAMATH_CALUDE_photo_arrangements_count_l961_96115


namespace NUMINAMATH_CALUDE_z_in_first_quadrant_l961_96164

def z : ℂ := (2 + Complex.I) * (1 + Complex.I)

theorem z_in_first_quadrant : 
  Complex.re z > 0 ∧ Complex.im z > 0 := by
  sorry

end NUMINAMATH_CALUDE_z_in_first_quadrant_l961_96164


namespace NUMINAMATH_CALUDE_sum_of_solutions_eq_18_l961_96129

theorem sum_of_solutions_eq_18 : 
  ∃ (S : Finset ℝ), (∀ x ∈ S, x^2 - 8*x + 21 = |x - 5| + 4) ∧ 
  (∀ x : ℝ, x^2 - 8*x + 21 = |x - 5| + 4 → x ∈ S) ∧
  (S.sum id = 18) :=
sorry

end NUMINAMATH_CALUDE_sum_of_solutions_eq_18_l961_96129


namespace NUMINAMATH_CALUDE_growth_rate_is_25_percent_l961_96181

/-- The average monthly growth rate of new 5G physical base stations -/
def average_growth_rate : ℝ := 0.25

/-- The number of new 5G physical base stations opened in January -/
def january_stations : ℕ := 1600

/-- The number of new 5G physical base stations opened in March -/
def march_stations : ℕ := 2500

/-- Theorem stating that the average monthly growth rate is 25% -/
theorem growth_rate_is_25_percent :
  january_stations * (1 + average_growth_rate)^2 = march_stations := by
  sorry

#check growth_rate_is_25_percent

end NUMINAMATH_CALUDE_growth_rate_is_25_percent_l961_96181


namespace NUMINAMATH_CALUDE_quadratic_equation_root_relation_l961_96147

theorem quadratic_equation_root_relation (m : ℝ) (hm : m ≠ 0) :
  (∃ x₁ x₂ : ℝ, x₁^2 - x₁ + m = 0 ∧ x₂^2 - x₂ + 3*m = 0 ∧ x₂ = 2*x₁) →
  m = -2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_relation_l961_96147


namespace NUMINAMATH_CALUDE_flower_bed_area_is_35_l961_96144

/-- The area of a rectangular flower bed -/
def flower_bed_area (width : ℝ) (length : ℝ) : ℝ := width * length

theorem flower_bed_area_is_35 :
  flower_bed_area 5 7 = 35 := by
  sorry

end NUMINAMATH_CALUDE_flower_bed_area_is_35_l961_96144


namespace NUMINAMATH_CALUDE_plane_equation_proof_l961_96155

/-- A plane in 3D space represented by its equation coefficients -/
structure Plane where
  A : ℤ
  B : ℤ
  C : ℤ
  D : ℤ
  A_pos : A > 0
  coeff_coprime : Nat.gcd (Nat.gcd (Int.natAbs A) (Int.natAbs B)) (Nat.gcd (Int.natAbs C) (Int.natAbs D)) = 1

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Check if a point lies on a plane -/
def point_on_plane (p : Point3D) (plane : Plane) : Prop :=
  plane.A * p.x + plane.B * p.y + plane.C * p.z + plane.D = 0

/-- Check if two planes are parallel -/
def planes_parallel (p1 p2 : Plane) : Prop :=
  ∃ (k : ℚ), k ≠ 0 ∧ p1.A = k * p2.A ∧ p1.B = k * p2.B ∧ p1.C = k * p2.C

theorem plane_equation_proof (given_plane : Plane) (point : Point3D) :
  given_plane.A = 4 ∧ given_plane.B = -2 ∧ given_plane.C = 6 ∧ given_plane.D = 14 →
  point.x = 2 ∧ point.y = -1 ∧ point.z = 3 →
  ∃ (result_plane : Plane),
    point_on_plane point result_plane ∧
    planes_parallel result_plane given_plane ∧
    result_plane.A = 2 ∧ result_plane.B = -1 ∧ result_plane.C = 3 ∧ result_plane.D = -14 :=
by sorry

end NUMINAMATH_CALUDE_plane_equation_proof_l961_96155


namespace NUMINAMATH_CALUDE_shawna_situps_wednesday_l961_96100

/-- Calculates the number of situps Shawna needs to do on Wednesday -/
def situps_needed_wednesday (daily_goal : ℕ) (monday_situps : ℕ) (tuesday_situps : ℕ) : ℕ :=
  daily_goal + (daily_goal - monday_situps) + (daily_goal - tuesday_situps)

/-- Theorem: Given Shawna's daily goal and her performance on Monday and Tuesday,
    she needs to do 59 situps on Wednesday to meet her goal and make up for missed situps -/
theorem shawna_situps_wednesday :
  situps_needed_wednesday 30 12 19 = 59 := by
  sorry

end NUMINAMATH_CALUDE_shawna_situps_wednesday_l961_96100


namespace NUMINAMATH_CALUDE_sum_of_xyz_l961_96165

theorem sum_of_xyz (x y z : ℝ) (h1 : y = 3 * x) (h2 : z = y + 5) : 
  x + y + z = 7 * x + 5 := by
sorry

end NUMINAMATH_CALUDE_sum_of_xyz_l961_96165


namespace NUMINAMATH_CALUDE_brothers_total_goals_l961_96120

/-- The total number of goals scored by Louie and his brother -/
def total_goals (louie_last_match : ℕ) (louie_previous : ℕ) (brother_seasons : ℕ) (games_per_season : ℕ) : ℕ :=
  let louie_total := louie_last_match + louie_previous
  let brother_per_game := 2 * louie_last_match
  let brother_total := brother_seasons * games_per_season * brother_per_game
  louie_total + brother_total

/-- Theorem stating the total number of goals scored by the brothers -/
theorem brothers_total_goals :
  total_goals 4 40 3 50 = 1244 := by
  sorry

end NUMINAMATH_CALUDE_brothers_total_goals_l961_96120


namespace NUMINAMATH_CALUDE_train_speed_l961_96196

/-- Proves that the current average speed of a train is 48 kmph given the specified conditions -/
theorem train_speed (distance : ℝ) : 
  (distance = (50 / 60) * 48) → 
  (distance = (40 / 60) * 60) → 
  48 = (60 * 40) / 50 := by
  sorry

#check train_speed

end NUMINAMATH_CALUDE_train_speed_l961_96196


namespace NUMINAMATH_CALUDE_rachels_father_age_rachels_father_age_at_25_l961_96172

/-- Rachel's age problem -/
theorem rachels_father_age (rachel_age : ℕ) (grandfather_age_multiplier : ℕ) 
  (father_age_difference : ℕ) (rachel_future_age : ℕ) : ℕ :=
  let grandfather_age := rachel_age * grandfather_age_multiplier
  let mother_age := grandfather_age / 2
  let father_age := mother_age + father_age_difference
  let years_passed := rachel_future_age - rachel_age
  father_age + years_passed

/-- Proof of Rachel's father's age when she is 25 -/
theorem rachels_father_age_at_25 : 
  rachels_father_age 12 7 5 25 = 60 := by
  sorry

end NUMINAMATH_CALUDE_rachels_father_age_rachels_father_age_at_25_l961_96172


namespace NUMINAMATH_CALUDE_expression_value_l961_96195

theorem expression_value (a b : ℝ) (h : a * b > 0) :
  (a / abs a) + (b / abs b) + ((a * b) / abs (a * b)) = 3 ∨
  (a / abs a) + (b / abs b) + ((a * b) / abs (a * b)) = -1 :=
by sorry

end NUMINAMATH_CALUDE_expression_value_l961_96195


namespace NUMINAMATH_CALUDE_ratatouille_cost_per_quart_l961_96103

/-- Calculates the cost per quart of ratatouille given the ingredients and their prices. -/
theorem ratatouille_cost_per_quart 
  (eggplant_zucchini_weight : ℝ)
  (eggplant_zucchini_price : ℝ)
  (tomato_weight : ℝ)
  (tomato_price : ℝ)
  (onion_weight : ℝ)
  (onion_price : ℝ)
  (basil_weight : ℝ)
  (basil_half_pound_price : ℝ)
  (total_quarts : ℝ)
  (h1 : eggplant_zucchini_weight = 9)
  (h2 : eggplant_zucchini_price = 2)
  (h3 : tomato_weight = 4)
  (h4 : tomato_price = 3.5)
  (h5 : onion_weight = 3)
  (h6 : onion_price = 1)
  (h7 : basil_weight = 1)
  (h8 : basil_half_pound_price = 2.5)
  (h9 : total_quarts = 4) :
  (eggplant_zucchini_weight * eggplant_zucchini_price + 
   tomato_weight * tomato_price + 
   onion_weight * onion_price + 
   basil_weight * basil_half_pound_price * 2) / total_quarts = 10 :=
by sorry

end NUMINAMATH_CALUDE_ratatouille_cost_per_quart_l961_96103


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l961_96176

/-- An isosceles trapezoid with specific dimensions and inscribed circles --/
structure IsoscelesTrapezoidWithCircles where
  -- The length of side AB
  ab : ℝ
  -- The length of sides BC and DA
  bc : ℝ
  -- The length of side CD
  cd : ℝ
  -- The radius of circles centered at A and B
  r_ab : ℝ
  -- The radius of circles centered at C and D
  r_cd : ℝ

/-- The theorem stating the radius of the inscribed circle tangent to all four circles --/
theorem inscribed_circle_radius (t : IsoscelesTrapezoidWithCircles)
  (h_ab : t.ab = 10)
  (h_bc : t.bc = 7)
  (h_cd : t.cd = 6)
  (h_r_ab : t.r_ab = 4)
  (h_r_cd : t.r_cd = 3) :
  ∃ r : ℝ, r = (-81 + 57 * Real.sqrt 5) / 23 ∧
    (∃ O : ℝ × ℝ, ∃ A B C D : ℝ × ℝ,
      -- O is the center of the inscribed circle
      -- A, B, C, D are the centers of the given circles
      -- The inscribed circle is tangent to all four given circles
      True) := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l961_96176


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l961_96139

theorem max_value_sqrt_sum (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) (hsum : a + b + c = 7) :
  Real.sqrt (3 * a + 2) + Real.sqrt (3 * b + 2) + Real.sqrt (3 * c + 2) ≤ Real.sqrt 69 := by
  sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l961_96139


namespace NUMINAMATH_CALUDE_non_acute_triangle_sides_count_l961_96189

/-- Given a triangle with two sides of lengths 20 and 19, this function returns the number of possible integer lengths for the third side that make the triangle not acute. -/
def count_non_acute_triangle_sides : ℕ :=
  let a : ℕ := 20
  let b : ℕ := 19
  let possible_sides := (Finset.range 37).filter (fun s => 
    (s > 1 ∧ s < 39) ∧  -- Triangle inequality
    ((s * s ≥ a * a + b * b) ∨  -- s is the longest side (obtuse or right triangle)
     (a * a ≥ s * s + b * b)))  -- a is the longest side (obtuse or right triangle)
  possible_sides.card

/-- Theorem stating that there are exactly 16 possible integer lengths for the third side of a triangle with sides 20 and 19 that make it not acute. -/
theorem non_acute_triangle_sides_count : count_non_acute_triangle_sides = 16 := by
  sorry

end NUMINAMATH_CALUDE_non_acute_triangle_sides_count_l961_96189


namespace NUMINAMATH_CALUDE_micah_ate_six_strawberries_l961_96159

/-- The number of strawberries in a dozen -/
def dozen : ℕ := 12

/-- The number of dozens of strawberries Micah picked -/
def dozens_picked : ℕ := 2

/-- The number of strawberries Micah saved for his mom -/
def saved_for_mom : ℕ := 18

/-- The total number of strawberries Micah picked -/
def total_picked : ℕ := dozens_picked * dozen

/-- The number of strawberries Micah ate -/
def eaten_by_micah : ℕ := total_picked - saved_for_mom

theorem micah_ate_six_strawberries : eaten_by_micah = 6 := by
  sorry

end NUMINAMATH_CALUDE_micah_ate_six_strawberries_l961_96159


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l961_96187

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | 1 ≤ x ∧ x < 4}
def B : Set ℝ := {x : ℝ | -2 ≤ x ∧ x < 2}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 1 ≤ x ∧ x < 2} :=
by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l961_96187


namespace NUMINAMATH_CALUDE_min_value_of_sequence_l961_96138

theorem min_value_of_sequence (a : ℕ → ℝ) :
  a 1 = 2 →
  (∀ n : ℕ, n ≥ 1 → a (n + 1) - a n = 2 * n) →
  (∀ n : ℕ, n ≥ 1 → a n / n ≥ 2) ∧ (∃ n : ℕ, n ≥ 1 ∧ a n / n = 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_sequence_l961_96138


namespace NUMINAMATH_CALUDE_container_volume_ratio_l961_96117

theorem container_volume_ratio : 
  ∀ (v1 v2 : ℚ), v1 > 0 → v2 > 0 →
  (5 / 6 : ℚ) * v1 = (3 / 4 : ℚ) * v2 →
  v1 / v2 = (9 / 10 : ℚ) := by
sorry

end NUMINAMATH_CALUDE_container_volume_ratio_l961_96117


namespace NUMINAMATH_CALUDE_geometric_mean_proof_l961_96145

theorem geometric_mean_proof (a b : ℝ) (hb : b ≠ 0) :
  Real.sqrt ((2 * (a^2 - a*b)) / (35*b) * (10*a) / (7*(a*b - b^2))) = 2*a / (7*b) := by
  sorry

end NUMINAMATH_CALUDE_geometric_mean_proof_l961_96145


namespace NUMINAMATH_CALUDE_minimum_time_for_assessment_l961_96194

/-- Represents the minimum time needed to assess students -/
def minimum_assessment_time (
  teacher1_problem_solving_time : ℕ)
  (teacher1_theory_time : ℕ)
  (teacher2_problem_solving_time : ℕ)
  (teacher2_theory_time : ℕ)
  (total_students : ℕ) : ℕ :=
  110

/-- Theorem stating the minimum time needed to assess 25 students
    given the specified conditions -/
theorem minimum_time_for_assessment :
  minimum_assessment_time 5 7 3 4 25 = 110 := by
  sorry

end NUMINAMATH_CALUDE_minimum_time_for_assessment_l961_96194


namespace NUMINAMATH_CALUDE_millet_dominant_on_wednesday_l961_96106

/-- Represents the state of the bird feeder on a given day -/
structure FeederState where
  day : ℕ
  millet : ℝ
  other : ℝ

/-- Calculates the next day's feeder state -/
def nextDay (state : FeederState) : FeederState :=
  { day := state.day + 1,
    millet := 0.8 * state.millet + 0.3,
    other := 0.5 * state.other + 0.7 }

/-- Checks if millet constitutes more than half of the seeds -/
def milletDominant (state : FeederState) : Prop :=
  state.millet > (state.millet + state.other) / 2

/-- Initial state of the feeder -/
def initialState : FeederState :=
  { day := 1, millet := 0.3, other := 0.7 }

/-- Theorem stating that millet becomes dominant on day 3 (Wednesday) -/
theorem millet_dominant_on_wednesday :
  let day3 := nextDay (nextDay initialState)
  milletDominant day3 ∧ ¬milletDominant (nextDay initialState) :=
sorry

end NUMINAMATH_CALUDE_millet_dominant_on_wednesday_l961_96106


namespace NUMINAMATH_CALUDE_simplify_expression_l961_96169

theorem simplify_expression (a : ℝ) : 3*a^2 - 2*a + 1 + (3*a - a^2 + 2) = 2*a^2 + a + 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l961_96169


namespace NUMINAMATH_CALUDE_carnation_dozen_cost_carnation_dozen_cost_proof_l961_96188

theorem carnation_dozen_cost (single_cost : ℚ) (teacher_dozens : ℕ) (friend_singles : ℕ) (total_spent : ℚ) : ℚ :=
  let dozen_cost := (total_spent - single_cost * friend_singles) / teacher_dozens
  dozen_cost

#check carnation_dozen_cost (1/2) 5 14 25 = 18/5

-- The proof is omitted
theorem carnation_dozen_cost_proof :
  carnation_dozen_cost (1/2) 5 14 25 = 18/5 := by sorry

end NUMINAMATH_CALUDE_carnation_dozen_cost_carnation_dozen_cost_proof_l961_96188


namespace NUMINAMATH_CALUDE_midpoint_sum_l961_96157

/-- Given points A (a, 6), B (-2, b), and P (2, 3) where P bisects AB, prove a + b = 6 -/
theorem midpoint_sum (a b : ℝ) : 
  (2 : ℝ) = (a + (-2)) / 2 → 
  (3 : ℝ) = (6 + b) / 2 → 
  a + b = 6 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_sum_l961_96157


namespace NUMINAMATH_CALUDE_digit_B_is_three_l961_96131

/-- Represents a digit from 1 to 7 -/
def Digit := Fin 7

/-- Represents the set of points A, B, C, D, E, F -/
structure Points where
  A : Digit
  B : Digit
  C : Digit
  D : Digit
  E : Digit
  F : Digit
  distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧
             B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧
             C ≠ D ∧ C ≠ E ∧ C ≠ F ∧
             D ≠ E ∧ D ≠ F ∧
             E ≠ F

/-- The sum of digits along each line -/
def lineSums (p : Points) : ℕ :=
  (p.A.val + p.B.val + p.C.val + 1) +
  (p.A.val + p.E.val + p.F.val + 1) +
  (p.C.val + p.D.val + p.E.val + 1) +
  (p.B.val + p.D.val + 1) +
  (p.B.val + p.F.val + 1)

theorem digit_B_is_three (p : Points) (h : lineSums p = 51) : p.B.val + 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_digit_B_is_three_l961_96131


namespace NUMINAMATH_CALUDE_cupcakes_per_box_l961_96107

theorem cupcakes_per_box 
  (total_baked : ℕ) 
  (left_at_home : ℕ) 
  (boxes_given : ℕ) 
  (h1 : total_baked = 53) 
  (h2 : left_at_home = 2) 
  (h3 : boxes_given = 17) :
  (total_baked - left_at_home) / boxes_given = 3 := by
sorry

end NUMINAMATH_CALUDE_cupcakes_per_box_l961_96107


namespace NUMINAMATH_CALUDE_triangle_segment_equality_l961_96135

theorem triangle_segment_equality (AB AC : ℝ) (m n : ℕ+) :
  AB = 33 →
  AC = 21 →
  ∃ (BC : ℝ), BC = m →
  ∃ (D E : ℝ × ℝ),
    D.1 + D.2 = AB ∧
    E.1 + E.2 = AC ∧
    D.1 = n ∧
    Real.sqrt ((D.1 - E.1)^2 + (D.2 - E.2)^2) = n ∧
    E.2 = n →
  n = 11 ∨ n = 21 := by
sorry


end NUMINAMATH_CALUDE_triangle_segment_equality_l961_96135


namespace NUMINAMATH_CALUDE_sin_alpha_equals_half_l961_96148

theorem sin_alpha_equals_half (α : Real) (h1 : α ∈ Set.Ioo 0 π) (h2 : Real.sin α = Real.cos (2 * α)) : 
  Real.sin α = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_sin_alpha_equals_half_l961_96148


namespace NUMINAMATH_CALUDE_cubic_root_sum_l961_96132

theorem cubic_root_sum (a b c : ℝ) : 
  (0 < a ∧ a < 1) ∧ (0 < b ∧ b < 1) ∧ (0 < c ∧ c < 1) →
  24 * a^3 - 36 * a^2 + 14 * a - 1 = 0 →
  24 * b^3 - 36 * b^2 + 14 * b - 1 = 0 →
  24 * c^3 - 36 * c^2 + 14 * c - 1 = 0 →
  1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c) = 158 / 73 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l961_96132


namespace NUMINAMATH_CALUDE_factorization_equality_l961_96116

theorem factorization_equality (a x : ℝ) : a * x^2 - 4 * a * x + 4 * a = a * (x - 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l961_96116


namespace NUMINAMATH_CALUDE_power_function_property_l961_96125

theorem power_function_property (a : ℝ) (f : ℝ → ℝ) : 
  (∀ x > 0, f x = x ^ a) → f 2 = Real.sqrt 2 → f 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_function_property_l961_96125


namespace NUMINAMATH_CALUDE_president_vice_president_selection_l961_96143

/-- The number of candidates for class president and vice president -/
def num_candidates : ℕ := 4

/-- The number of positions to be filled (president and vice president) -/
def num_positions : ℕ := 2

/-- Theorem: The number of ways to choose a president and a vice president from 4 candidates is 12 -/
theorem president_vice_president_selection :
  (num_candidates * (num_candidates - 1)) = 12 := by
  sorry

#check president_vice_president_selection

end NUMINAMATH_CALUDE_president_vice_president_selection_l961_96143


namespace NUMINAMATH_CALUDE_number_of_arrangements_l961_96104

/-- The number of foreign guests -/
def num_foreign_guests : ℕ := 4

/-- The number of security officers -/
def num_security_officers : ℕ := 2

/-- The total number of individuals -/
def total_individuals : ℕ := num_foreign_guests + num_security_officers

/-- The number of foreign guests that must be together -/
def num_guests_together : ℕ := 2

/-- The function to calculate the number of possible arrangements -/
def calculate_arrangements (n_foreign : ℕ) (n_security : ℕ) (n_together : ℕ) : ℕ :=
  sorry

/-- The theorem stating the number of possible arrangements -/
theorem number_of_arrangements :
  calculate_arrangements num_foreign_guests num_security_officers num_guests_together = 24 :=
sorry

end NUMINAMATH_CALUDE_number_of_arrangements_l961_96104


namespace NUMINAMATH_CALUDE_equal_probability_for_claudia_and_adela_l961_96198

/-- The probability that a single die roll is not a multiple of 3 -/
def p_not_multiple_of_3 : ℚ := 2/3

/-- The probability that a single die roll is a multiple of 3 -/
def p_multiple_of_3 : ℚ := 1/3

/-- The number of dice rolled -/
def n : ℕ := 2

theorem equal_probability_for_claudia_and_adela :
  p_not_multiple_of_3 ^ n = n * p_multiple_of_3 * p_not_multiple_of_3 ^ (n - 1) :=
sorry

end NUMINAMATH_CALUDE_equal_probability_for_claudia_and_adela_l961_96198


namespace NUMINAMATH_CALUDE_brownie_ratio_l961_96185

def total_brownies : ℕ := 15
def monday_brownies : ℕ := 5

def tuesday_brownies : ℕ := total_brownies - monday_brownies

theorem brownie_ratio :
  (tuesday_brownies : ℚ) / monday_brownies = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_brownie_ratio_l961_96185


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l961_96173

def arithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property (a : ℕ → ℝ) :
  arithmeticSequence a →
  a 4 + a 6 + a 8 + a 10 + a 12 = 120 →
  a 7 - (1/3) * a 5 = 16 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l961_96173


namespace NUMINAMATH_CALUDE_carol_wins_probability_l961_96140

/-- The probability of getting a six on a single die toss -/
def prob_six : ℚ := 1 / 6

/-- The probability of not getting a six on a single die toss -/
def prob_not_six : ℚ := 1 - prob_six

/-- The number of players before Carol in the sequence -/
def players_before_carol : ℕ := 2

/-- The total number of players in the sequence -/
def total_players : ℕ := 4

/-- The probability that Carol wins on her first turn in any cycle -/
def prob_carol_wins_first_turn : ℚ := prob_not_six ^ players_before_carol * prob_six

/-- The probability that no one wins in a full cycle -/
def prob_no_win_cycle : ℚ := prob_not_six ^ total_players

/-- Theorem: The probability that Carol is the first to toss a six is 25/91 -/
theorem carol_wins_probability :
  prob_carol_wins_first_turn / (1 - prob_no_win_cycle) = 25 / 91 := by
  sorry

end NUMINAMATH_CALUDE_carol_wins_probability_l961_96140


namespace NUMINAMATH_CALUDE_chords_and_triangles_10_points_l961_96191

/-- The number of chords formed by n points on a circumference -/
def num_chords (n : ℕ) : ℕ := n.choose 2

/-- The number of triangles formed by n points on a circumference -/
def num_triangles (n : ℕ) : ℕ := n.choose 3

/-- Theorem about chords and triangles formed by 10 points on a circumference -/
theorem chords_and_triangles_10_points :
  num_chords 10 = 45 ∧ num_triangles 10 = 120 := by
  sorry

end NUMINAMATH_CALUDE_chords_and_triangles_10_points_l961_96191


namespace NUMINAMATH_CALUDE_max_value_of_f_l961_96193

-- Define the function f(x) = x³ - 3x²
def f (x : ℝ) : ℝ := x^3 - 3*x^2

-- Theorem statement
theorem max_value_of_f :
  ∃ (c : ℝ), c ∈ Set.Icc (-2 : ℝ) 4 ∧
  (∀ x, x ∈ Set.Icc (-2 : ℝ) 4 → f x ≤ f c) ∧
  f c = 16 := by
sorry


end NUMINAMATH_CALUDE_max_value_of_f_l961_96193


namespace NUMINAMATH_CALUDE_fundraiser_theorem_l961_96110

def fundraiser (num_students : ℕ) (individual_cost : ℕ) (collective_cost : ℕ) 
                (day1_raised : ℕ) (day2_raised : ℕ) (day3_raised : ℕ) : ℕ :=
  let total_needed := num_students * individual_cost + collective_cost
  let first3days_raised := day1_raised + day2_raised + day3_raised
  let next4days_raised := first3days_raised / 2
  let total_raised := first3days_raised + next4days_raised
  let remaining := total_needed - total_raised
  remaining / num_students

theorem fundraiser_theorem : 
  fundraiser 6 450 3000 600 900 400 = 475 := by
  sorry

end NUMINAMATH_CALUDE_fundraiser_theorem_l961_96110


namespace NUMINAMATH_CALUDE_y_intercept_of_line_l961_96152

/-- The y-intercept of a line is the y-coordinate of the point where the line intersects the y-axis. -/
def y_intercept (f : ℝ → ℝ) : ℝ := f 0

/-- The line equation y = -3x + 5 -/
def line (x : ℝ) : ℝ := -3 * x + 5

theorem y_intercept_of_line :
  y_intercept line = 5 := by sorry

end NUMINAMATH_CALUDE_y_intercept_of_line_l961_96152
