import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_digits_square_n_l1321_132183

/-- The number formed by repeating the digit 7 eight times -/
def n : ℕ := 77777777

/-- Sum of digits function -/
def sum_of_digits (k : ℕ) : ℕ := sorry

/-- The theorem to be proved -/
theorem sum_of_digits_square_n : sum_of_digits (n^2) = 13 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_square_n_l1321_132183


namespace NUMINAMATH_CALUDE_snow_clearing_time_l1321_132142

/-- Calculates the number of hours required to clear snow given the total volume,
    initial shoveling capacity, and hourly decrease in capacity. -/
def snow_clearing_hours (total_volume : ℕ) (initial_capacity : ℕ) (hourly_decrease : ℕ) : ℕ :=
  -- Definition to be implemented
  sorry

theorem snow_clearing_time :
  snow_clearing_hours 216 25 2 = 21 := by
  sorry

end NUMINAMATH_CALUDE_snow_clearing_time_l1321_132142


namespace NUMINAMATH_CALUDE_convention_handshakes_l1321_132115

theorem convention_handshakes (num_companies : ℕ) (reps_per_company : ℕ) : 
  num_companies = 5 → 
  reps_per_company = 4 → 
  (num_companies * reps_per_company * (num_companies * reps_per_company - reps_per_company)) / 2 = 160 := by
sorry

end NUMINAMATH_CALUDE_convention_handshakes_l1321_132115


namespace NUMINAMATH_CALUDE_largest_m_for_quadratic_inequality_l1321_132107

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem largest_m_for_quadratic_inequality 
  (a b c : ℝ) 
  (ha : a ≠ 0)
  (h1 : ∀ x : ℝ, f a b c (x - 4) = f a b c (2 - x))
  (h2 : ∀ x : ℝ, f a b c x ≥ x)
  (h3 : ∀ x ∈ Set.Ioo 0 2, f a b c x ≤ ((x + 1) / 2)^2)
  (h4 : ∃ x : ℝ, ∀ y : ℝ, f a b c x ≤ f a b c y)
  (h5 : ∃ x : ℝ, f a b c x = 0) :
  (∃ m : ℝ, m > 1 ∧ 
    (∃ t : ℝ, ∀ x ∈ Set.Icc 1 m, f a b c (x + t) ≤ x) ∧
    (∀ m' : ℝ, m' > m → 
      ¬∃ t : ℝ, ∀ x ∈ Set.Icc 1 m', f a b c (x + t) ≤ x)) ∧
  (∀ m : ℝ, (∃ t : ℝ, ∀ x ∈ Set.Icc 1 m, f a b c (x + t) ≤ x) → m ≤ 9) :=
by sorry

end NUMINAMATH_CALUDE_largest_m_for_quadratic_inequality_l1321_132107


namespace NUMINAMATH_CALUDE_right_triangle_sets_l1321_132130

def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

theorem right_triangle_sets :
  (is_right_triangle 1 2 (Real.sqrt 5)) ∧
  (is_right_triangle (Real.sqrt 2) (Real.sqrt 2) 2) ∧
  (is_right_triangle 13 12 5) ∧
  ¬(is_right_triangle 1 3 (Real.sqrt 7)) := by sorry

end NUMINAMATH_CALUDE_right_triangle_sets_l1321_132130


namespace NUMINAMATH_CALUDE_chad_cracker_boxes_l1321_132139

/-- The number of crackers Chad uses per sandwich -/
def crackers_per_sandwich : ℕ := 2

/-- The number of sandwiches Chad eats per night -/
def sandwiches_per_night : ℕ := 5

/-- The number of sleeves in a box of crackers -/
def sleeves_per_box : ℕ := 4

/-- The number of crackers in each sleeve -/
def crackers_per_sleeve : ℕ := 28

/-- The number of nights the crackers will last -/
def nights_lasting : ℕ := 56

/-- Calculates the number of boxes of crackers Chad has -/
def boxes_of_crackers : ℕ :=
  (crackers_per_sandwich * sandwiches_per_night * nights_lasting) /
  (sleeves_per_box * crackers_per_sleeve)

theorem chad_cracker_boxes :
  boxes_of_crackers = 5 := by
  sorry

end NUMINAMATH_CALUDE_chad_cracker_boxes_l1321_132139


namespace NUMINAMATH_CALUDE_system_solution_l1321_132148

theorem system_solution (x y b : ℝ) : 
  (4 * x + y = b) → 
  (3 * x + 4 * y = 3 * b) → 
  (x = 3) → 
  (b = 39) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l1321_132148


namespace NUMINAMATH_CALUDE_back_seat_capacity_l1321_132157

def bus_capacity := 88
def left_side_seats := 15
def seat_capacity := 3

theorem back_seat_capacity : 
  ∀ (right_side_seats : ℕ) (back_seat_capacity : ℕ),
    right_side_seats = left_side_seats - 3 →
    bus_capacity = left_side_seats * seat_capacity + right_side_seats * seat_capacity + back_seat_capacity →
    back_seat_capacity = 7 := by
  sorry

end NUMINAMATH_CALUDE_back_seat_capacity_l1321_132157


namespace NUMINAMATH_CALUDE_tangent_line_and_inequality_l1321_132145

noncomputable section

-- Define the functions f and g
def f (x : ℝ) := x * Real.log x
def g (a : ℝ) (x : ℝ) := -x^2 + a*x - 3

-- State the theorem
theorem tangent_line_and_inequality (a : ℝ) :
  -- Part 1: Tangent line equation
  (∀ x : ℝ, HasDerivAt f (x - 1) 1) ∧
  -- Part 2: Inequality condition
  (∀ x : ℝ, x > 0 → 2 * f x ≥ g a x) ↔ a ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_and_inequality_l1321_132145


namespace NUMINAMATH_CALUDE_xyz_equals_seven_l1321_132132

theorem xyz_equals_seven (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 30)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 9) :
  x * y * z = 7 := by
sorry

end NUMINAMATH_CALUDE_xyz_equals_seven_l1321_132132


namespace NUMINAMATH_CALUDE_meaningful_range_of_fraction_l1321_132122

/-- The meaningful range of a fraction is the set of values for which the denominator is non-zero. -/
def meaningful_range (f : ℝ → ℝ) : Set ℝ :=
  {x | f x ≠ 0}

/-- The function representing the denominator of the fraction x / (x - 3). -/
def denominator (x : ℝ) : ℝ := x - 3

theorem meaningful_range_of_fraction :
    meaningful_range denominator = {x | x ≠ 3} := by
  sorry

end NUMINAMATH_CALUDE_meaningful_range_of_fraction_l1321_132122


namespace NUMINAMATH_CALUDE_unique_positive_solution_l1321_132198

theorem unique_positive_solution :
  ∃! (x : ℝ), x > 0 ∧ (x - 5) / 10 = 5 / (x - 10) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l1321_132198


namespace NUMINAMATH_CALUDE_trees_planted_by_fourth_grade_l1321_132197

theorem trees_planted_by_fourth_grade :
  ∀ (fifth_grade third_grade fourth_grade : ℕ),
    fifth_grade = 114 →
    fifth_grade = 2 * third_grade →
    fourth_grade = third_grade + 32 →
    fourth_grade = 89 := by sorry

end NUMINAMATH_CALUDE_trees_planted_by_fourth_grade_l1321_132197


namespace NUMINAMATH_CALUDE_company2_manager_percent_is_22_5_l1321_132109

/-- Represents the workforce composition of two companies before and after a merger -/
structure CompanyMerger where
  company1_manager_percent : Real
  company2_manager_percent : Real
  company2_engineer_percent : Real
  company2_support_percent : Real
  merged_manager_percent : Real
  merged_company1_percent : Real

/-- Theorem stating that given the conditions of the merger, the percentage of managers in Company 2 is 22.5% -/
theorem company2_manager_percent_is_22_5 (merger : CompanyMerger) 
  (h1 : merger.company1_manager_percent = 0.1)
  (h2 : merger.company2_engineer_percent = 0.1)
  (h3 : merger.company2_support_percent = 0.6)
  (h4 : merger.merged_manager_percent = 0.25)
  (h5 : merger.merged_company1_percent = 0.25) :
  merger.company2_manager_percent = 0.225 := by
  sorry

#check company2_manager_percent_is_22_5

end NUMINAMATH_CALUDE_company2_manager_percent_is_22_5_l1321_132109


namespace NUMINAMATH_CALUDE_coin_collection_value_johns_collection_value_l1321_132138

/-- Proves the value of a coin collection given certain conditions -/
theorem coin_collection_value
  (total_coins : ℕ)
  (sample_coins : ℕ)
  (sample_value : ℚ)
  (h1 : total_coins = 24)
  (h2 : sample_coins = 8)
  (h3 : sample_value = 20)
  : ℚ
:=
by
  -- The value of the entire collection
  sorry

/-- The main theorem stating the value of John's coin collection -/
theorem johns_collection_value : coin_collection_value 24 8 20 rfl rfl rfl = 60 := by sorry

end NUMINAMATH_CALUDE_coin_collection_value_johns_collection_value_l1321_132138


namespace NUMINAMATH_CALUDE_value_of_expression_l1321_132191

-- Define the function f
def f (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

-- State the theorem
theorem value_of_expression (a b c d : ℝ) :
  f a b c d (-2) = -3 → 10 * a - 5 * b + 3 * c - 2 * d = 12 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l1321_132191


namespace NUMINAMATH_CALUDE_empty_set_implies_m_zero_l1321_132136

theorem empty_set_implies_m_zero (m : ℝ) : (∀ x : ℝ, m * x ≠ 1) → m = 0 := by
  sorry

end NUMINAMATH_CALUDE_empty_set_implies_m_zero_l1321_132136


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_inequality_l1321_132146

theorem unique_solution_quadratic_inequality (b : ℝ) : 
  (∃! x : ℝ, |x^2 - 6*b*x + 5*b| ≤ 3) ↔ 
  (b = (5 + Real.sqrt 73) / 8 ∨ b = (5 - Real.sqrt 73) / 8) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_inequality_l1321_132146


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_sqrt_60_l1321_132121

theorem sqrt_sum_equals_sqrt_60 :
  Real.sqrt (25 - 10 * Real.sqrt 6) + Real.sqrt (25 + 10 * Real.sqrt 6) = Real.sqrt 60 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_sqrt_60_l1321_132121


namespace NUMINAMATH_CALUDE_unique_solution_l1321_132104

theorem unique_solution : 
  ∀ (a b : ℕ), 
    a > 1 → 
    b > 0 → 
    b ∣ (a - 1) → 
    (2 * a + 1) ∣ (5 * b - 3) → 
    a = 10 ∧ b = 9 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l1321_132104


namespace NUMINAMATH_CALUDE_fraction_of_fraction_of_fraction_main_theorem_l1321_132125

theorem fraction_of_fraction_of_fraction (a b c d : ℚ) :
  a * b * c * d = (a * b * c) * d := by sorry

theorem main_theorem : (1 / 2 : ℚ) * (1 / 3 : ℚ) * (1 / 6 : ℚ) * 72 = 2 := by sorry

end NUMINAMATH_CALUDE_fraction_of_fraction_of_fraction_main_theorem_l1321_132125


namespace NUMINAMATH_CALUDE_rope_cutting_problem_l1321_132108

theorem rope_cutting_problem :
  let total_length_feet : ℝ := 6
  let number_of_pieces : ℕ := 10
  let inches_per_foot : ℝ := 12
  let piece_length_inches : ℝ := total_length_feet * inches_per_foot / number_of_pieces
  piece_length_inches = 7.2 := by sorry

end NUMINAMATH_CALUDE_rope_cutting_problem_l1321_132108


namespace NUMINAMATH_CALUDE_song_size_calculation_l1321_132149

/-- Given a total number of songs and total memory space occupied,
    calculate the size of each song. -/
def song_size (total_songs : ℕ) (total_memory : ℕ) : ℚ :=
  total_memory / total_songs

theorem song_size_calculation :
  let morning_songs : ℕ := 10
  let later_songs : ℕ := 15
  let night_songs : ℕ := 3
  let total_songs : ℕ := morning_songs + later_songs + night_songs
  let total_memory : ℕ := 140
  song_size total_songs total_memory = 5 := by
  sorry

end NUMINAMATH_CALUDE_song_size_calculation_l1321_132149


namespace NUMINAMATH_CALUDE_max_value_sum_product_l1321_132192

theorem max_value_sum_product (a b c d : ℝ) : 
  a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 → a + b + c + d = 200 → 
  a * b + b * c + c * d + d * a ≤ 10000 := by
  sorry

end NUMINAMATH_CALUDE_max_value_sum_product_l1321_132192


namespace NUMINAMATH_CALUDE_sin_double_alpha_on_line_l1321_132101

/-- Given that a point P on the terminal side of angle α lies on the line y = 2x, prove that sin(2α) = 4/5 -/
theorem sin_double_alpha_on_line (α : Real) (P : ℝ × ℝ) : 
  (P.2 = 2 * P.1) → -- P lies on the line y = 2x
  (∃ r : ℝ, r > 0 ∧ P = (r * Real.cos α, r * Real.sin α)) → -- P is on the terminal side of angle α
  Real.sin (2 * α) = 4 / 5 := by sorry

end NUMINAMATH_CALUDE_sin_double_alpha_on_line_l1321_132101


namespace NUMINAMATH_CALUDE_modulus_of_z_l1321_132187

def z : ℂ := 3 + 4 * Complex.I

theorem modulus_of_z : Complex.abs z = 5 := by sorry

end NUMINAMATH_CALUDE_modulus_of_z_l1321_132187


namespace NUMINAMATH_CALUDE_expression_change_l1321_132100

theorem expression_change (x b : ℝ) (h : b > 0) :
  let f := fun t => t^3 - 2*t + 1
  (f (x + b) - f x = 3*b*x^2 + 3*b^2*x + b^3 - 2*b) ∧
  (f (x - b) - f x = -3*b*x^2 + 3*b^2*x - b^3 + 2*b) := by
  sorry

end NUMINAMATH_CALUDE_expression_change_l1321_132100


namespace NUMINAMATH_CALUDE_square_odd_implies_odd_l1321_132106

theorem square_odd_implies_odd (n : ℕ) : Odd (n^2) → Odd n := by
  sorry

end NUMINAMATH_CALUDE_square_odd_implies_odd_l1321_132106


namespace NUMINAMATH_CALUDE_tangent_and_locus_l1321_132114

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define point M
def point_M : ℝ × ℝ := (-1, -4)

-- Define point N
def point_N : ℝ × ℝ := (2, 0)

-- Define the tangent line equations
def tangent_line (x y : ℝ) : Prop := x = -1 ∨ 15*x - 8*y - 17 = 0

-- Define the locus of midpoint T
def locus_T (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1 ∧ 0 ≤ x ∧ x < 1/2

theorem tangent_and_locus :
  (∀ x y, circle_O x y → 
    (∃ x' y', tangent_line x' y' ∧ 
      (x' = point_M.1 ∧ y' = point_M.2))) ∧
  (∀ x y, locus_T x y ↔ 
    (∃ p q : ℝ × ℝ, 
      circle_O p.1 p.2 ∧ 
      circle_O q.1 q.2 ∧ 
      (q.2 - point_N.2) * (p.1 - point_N.1) = (q.1 - point_N.1) * (p.2 - point_N.2) ∧
      x = (p.1 + q.1) / 2 ∧ 
      y = (p.2 + q.2) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_and_locus_l1321_132114


namespace NUMINAMATH_CALUDE_abs_2x_plus_1_gt_3_l1321_132171

theorem abs_2x_plus_1_gt_3 (x : ℝ) : |2*x + 1| > 3 ↔ x > 1 ∨ x < -2 := by
  sorry

end NUMINAMATH_CALUDE_abs_2x_plus_1_gt_3_l1321_132171


namespace NUMINAMATH_CALUDE_franks_candy_bags_l1321_132179

theorem franks_candy_bags (pieces_per_bag : ℕ) (total_pieces : ℕ) (h1 : pieces_per_bag = 11) (h2 : total_pieces = 22) :
  total_pieces / pieces_per_bag = 2 :=
by sorry

end NUMINAMATH_CALUDE_franks_candy_bags_l1321_132179


namespace NUMINAMATH_CALUDE_x_n_perfect_square_iff_b_10_l1321_132105

def x_n (b n : ℕ) : ℕ :=
  let ones := (b^(2*n) - b^(n+1)) / (b - 1)
  let twos := 2 * (b^n - 1) / (b - 1)
  ones + twos + 5

theorem x_n_perfect_square_iff_b_10 (b : ℕ) (h : b > 5) :
  (∃ M : ℕ, ∀ n : ℕ, n > M → ∃ k : ℕ, x_n b n = k^2) ↔ b = 10 :=
sorry

end NUMINAMATH_CALUDE_x_n_perfect_square_iff_b_10_l1321_132105


namespace NUMINAMATH_CALUDE_parabola_hyperbola_equations_l1321_132165

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola with focus on the x-axis -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Represents a hyperbola with focus on the x-axis -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h : a > 0 ∧ b > 0

/-- The intersection point of the asymptotes -/
def intersectionPoint : Point := { x := 4, y := 8 }

/-- Check if a point satisfies the parabola equation -/
def satisfiesParabola (point : Point) (parabola : Parabola) : Prop :=
  point.y^2 = 2 * parabola.p * point.x

/-- Check if a point satisfies the hyperbola equation -/
def satisfiesHyperbola (point : Point) (hyperbola : Hyperbola) : Prop :=
  (point.x^2 / hyperbola.a^2) - (point.y^2 / hyperbola.b^2) = 1

/-- The main theorem -/
theorem parabola_hyperbola_equations :
  ∃ (parabola : Parabola) (hyperbola : Hyperbola),
    satisfiesParabola intersectionPoint parabola ∧
    satisfiesHyperbola intersectionPoint hyperbola ∧
    parabola.p = 8 ∧
    hyperbola.a^2 = 16/5 ∧
    hyperbola.b^2 = 64/5 := by
  sorry

end NUMINAMATH_CALUDE_parabola_hyperbola_equations_l1321_132165


namespace NUMINAMATH_CALUDE_apples_picked_total_l1321_132103

/-- The number of apples Benny picked -/
def benny_apples : ℕ := 2

/-- The number of apples Dan picked -/
def dan_apples : ℕ := 9

/-- The total number of apples picked -/
def total_apples : ℕ := benny_apples + dan_apples

theorem apples_picked_total :
  total_apples = 11 := by sorry

end NUMINAMATH_CALUDE_apples_picked_total_l1321_132103


namespace NUMINAMATH_CALUDE_min_value_quadratic_form_l1321_132186

theorem min_value_quadratic_form :
  ∀ x y : ℝ, x^2 + x*y + y^2 ≥ 0 ∧ (x^2 + x*y + y^2 = 0 ↔ x = 0 ∧ y = 0) :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_form_l1321_132186


namespace NUMINAMATH_CALUDE_darryl_break_even_l1321_132156

/-- Calculates the number of machines needed to break even given costs and selling price -/
def machines_to_break_even (parts_cost patent_cost selling_price : ℕ) : ℕ :=
  (parts_cost + patent_cost) / selling_price

/-- Theorem: Darryl needs to sell 45 machines to break even -/
theorem darryl_break_even :
  machines_to_break_even 3600 4500 180 = 45 := by
  sorry

end NUMINAMATH_CALUDE_darryl_break_even_l1321_132156


namespace NUMINAMATH_CALUDE_largest_reciprocal_l1321_132180

theorem largest_reciprocal (a b c d e : ℚ) : 
  a = 1/4 → b = 3/8 → c = 0 → d = -2 → e = 4 → 
  (1/a > 1/b ∧ 1/a > 1/d ∧ 1/a > 1/e) := by
  sorry

#check largest_reciprocal

end NUMINAMATH_CALUDE_largest_reciprocal_l1321_132180


namespace NUMINAMATH_CALUDE_solve_for_A_l1321_132127

/-- Given the equation 691-6A7=4 in base 10, prove that A = 8 -/
theorem solve_for_A : ∃ (A : ℕ), A < 10 ∧ 691 - (600 + A * 10 + 7) = 4 → A = 8 := by sorry

end NUMINAMATH_CALUDE_solve_for_A_l1321_132127


namespace NUMINAMATH_CALUDE_vector_magnitude_l1321_132111

/-- Given two vectors a and b in ℝ², prove that if a = (1, -1) and a + b = (3, 1), 
    then the magnitude of b is 2√2. -/
theorem vector_magnitude (a b : ℝ × ℝ) : 
  a = (1, -1) → a + b = (3, 1) → ‖b‖ = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_l1321_132111


namespace NUMINAMATH_CALUDE_problem_solution_l1321_132143

theorem problem_solution (x y : ℝ) (h1 : x^(2*y) = 81) (h2 : x = 9) : y = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1321_132143


namespace NUMINAMATH_CALUDE_cards_at_home_l1321_132159

def cards_in_hospital : ℕ := 403
def total_cards : ℕ := 690

theorem cards_at_home : total_cards - cards_in_hospital = 287 := by
  sorry

end NUMINAMATH_CALUDE_cards_at_home_l1321_132159


namespace NUMINAMATH_CALUDE_probability_four_blue_l1321_132184

/-- The number of blue marbles initially in the bag -/
def initial_blue : ℕ := 10

/-- The number of red marbles initially in the bag -/
def initial_red : ℕ := 5

/-- The total number of draws -/
def total_draws : ℕ := 10

/-- The number of blue marbles we want to draw -/
def target_blue : ℕ := 4

/-- The probability of drawing a blue marble, approximated as constant throughout the process -/
def p_blue : ℚ := 2/3

/-- The probability of drawing a red marble, approximated as constant throughout the process -/
def p_red : ℚ := 1/3

/-- The probability of drawing exactly 4 blue marbles out of 10 draws -/
theorem probability_four_blue : 
  (Nat.choose total_draws target_blue : ℚ) * p_blue^target_blue * p_red^(total_draws - target_blue) = 
  (210 * 16 : ℚ) / (81 * 729) := by sorry

end NUMINAMATH_CALUDE_probability_four_blue_l1321_132184


namespace NUMINAMATH_CALUDE_f_composition_negative_four_l1321_132181

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then Real.sqrt x else (1/2)^x

theorem f_composition_negative_four : f (f (-4)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_negative_four_l1321_132181


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l1321_132124

/-- Definition of the repeating decimal 0.3333... -/
def repeating_3 : ℚ := 1/3

/-- Definition of the repeating decimal 0.0404... -/
def repeating_04 : ℚ := 4/99

/-- Definition of the repeating decimal 0.005005... -/
def repeating_005 : ℚ := 5/999

/-- Theorem stating that the sum of the three repeating decimals equals 1135/2997 -/
theorem sum_of_repeating_decimals : 
  repeating_3 + repeating_04 + repeating_005 = 1135/2997 := by sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l1321_132124


namespace NUMINAMATH_CALUDE_prop_p_false_prop_q_true_l1321_132133

-- Define the curve C
def curve_C (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / (25 - k) + p.2^2 / (k - 9) = 1}

-- Define what it means for a curve to be an ellipse
def is_ellipse (S : Set (ℝ × ℝ)) : Prop :=
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ S = {p : ℝ × ℝ | p.1^2 / a^2 + p.2^2 / b^2 = 1}

-- Define what it means for a curve to be a hyperbola with foci on the x-axis
def is_hyperbola_x_axis (S : Set (ℝ × ℝ)) : Prop :=
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ S = {p : ℝ × ℝ | p.1^2 / a^2 - p.2^2 / b^2 = 1}

-- Theorem 1: Proposition p is false
theorem prop_p_false : ¬(∀ k : ℝ, 9 < k ∧ k < 25 → is_ellipse (curve_C k)) :=
  sorry

-- Theorem 2: Proposition q is true
theorem prop_q_true : ∀ k : ℝ, is_hyperbola_x_axis (curve_C k) → k < 9 :=
  sorry

end NUMINAMATH_CALUDE_prop_p_false_prop_q_true_l1321_132133


namespace NUMINAMATH_CALUDE_financial_equation_solution_l1321_132172

/-- Given a financial equation and some conditions, prove the value of p -/
theorem financial_equation_solution (q v : ℂ) (h1 : 3 * q - v = 5000) (h2 : q = 3) (h3 : v = 3 + 75 * Complex.I) :
  ∃ p : ℂ, p = 1667 + 25 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_financial_equation_solution_l1321_132172


namespace NUMINAMATH_CALUDE_train_passenger_count_l1321_132147

/-- Calculates the total number of passengers transported by a train between two stations -/
def total_passengers (num_round_trips : ℕ) (passengers_first_trip : ℕ) (passengers_return_trip : ℕ) : ℕ :=
  num_round_trips * (passengers_first_trip + passengers_return_trip)

/-- Proves that the total number of passengers transported is 640 given the specified conditions -/
theorem train_passenger_count :
  let num_round_trips : ℕ := 4
  let passengers_first_trip : ℕ := 100
  let passengers_return_trip : ℕ := 60
  total_passengers num_round_trips passengers_first_trip passengers_return_trip = 640 :=
by
  sorry


end NUMINAMATH_CALUDE_train_passenger_count_l1321_132147


namespace NUMINAMATH_CALUDE_cards_per_set_is_13_l1321_132134

/-- The number of trading cards in one set -/
def cards_per_set (initial_cards : ℕ) (sets_to_brother : ℕ) (sets_to_sister : ℕ) (sets_to_friend : ℕ) (total_cards_given : ℕ) : ℕ :=
  total_cards_given / (sets_to_brother + sets_to_sister + sets_to_friend)

/-- Proof that the number of trading cards in one set is 13 -/
theorem cards_per_set_is_13 :
  cards_per_set 365 8 5 2 195 = 13 := by
  sorry

end NUMINAMATH_CALUDE_cards_per_set_is_13_l1321_132134


namespace NUMINAMATH_CALUDE_acute_triangle_angle_sine_inequality_l1321_132155

theorem acute_triangle_angle_sine_inequality (A B C : Real) 
  (h1 : 0 < A ∧ A < π/2) 
  (h2 : 0 < B ∧ B < π/2) 
  (h3 : 0 < C ∧ C < π/2) 
  (h4 : A + B + C = π) 
  (h5 : A < B) 
  (h6 : B < C) : 
  Real.sin (2*A) > Real.sin (2*B) ∧ Real.sin (2*B) > Real.sin (2*C) := by
  sorry

end NUMINAMATH_CALUDE_acute_triangle_angle_sine_inequality_l1321_132155


namespace NUMINAMATH_CALUDE_initial_workers_l1321_132119

/-- Proves that the initial number of workers is 14, given the problem conditions --/
theorem initial_workers (total_toys : ℕ) (initial_days : ℕ) (added_workers : ℕ) (remaining_days : ℕ) :
  total_toys = 1400 →
  initial_days = 5 →
  added_workers = 14 →
  remaining_days = 2 →
  ∃ (initial_workers : ℕ),
    (initial_workers * initial_days + (initial_workers + added_workers) * remaining_days) * total_toys / 
    (initial_days + remaining_days) = total_toys ∧
    initial_workers = 14 := by
  sorry


end NUMINAMATH_CALUDE_initial_workers_l1321_132119


namespace NUMINAMATH_CALUDE_circle_center_radius_sum_l1321_132194

theorem circle_center_radius_sum :
  ∀ (a b r : ℝ),
  (∀ (x y : ℝ), x^2 - 16*x + y^2 + 6*y = 20 ↔ (x - a)^2 + (y - b)^2 = r^2) →
  a + b + r = 5 + Real.sqrt 93 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_radius_sum_l1321_132194


namespace NUMINAMATH_CALUDE_chess_team_girls_l1321_132162

theorem chess_team_girls (total : ℕ) (attended : ℕ) (boys : ℕ) (girls : ℕ) : 
  total = 30 →
  attended = 18 →
  total = boys + girls →
  attended = boys + girls / 3 →
  girls = 18 := by
sorry

end NUMINAMATH_CALUDE_chess_team_girls_l1321_132162


namespace NUMINAMATH_CALUDE_jones_elementary_population_l1321_132141

/-- The total number of students at Jones Elementary School -/
def total_students : ℕ := 150

/-- The number of boys at Jones Elementary School -/
def num_boys : ℕ := (60 * total_students) / 100

/-- Theorem stating that 90 students represent some percentage of the boys,
    and boys make up 60% of the total school population of 150 students -/
theorem jones_elementary_population :
  90 * total_students = 60 * num_boys :=
sorry

end NUMINAMATH_CALUDE_jones_elementary_population_l1321_132141


namespace NUMINAMATH_CALUDE_action_figure_sale_conditions_l1321_132110

/-- Represents the pricing of action figures and the financial situation of Lee. -/
structure ActionFigureSale where
  x : ℝ  -- Price of Type A
  y : ℝ  -- Price of Type B
  z : ℝ  -- Price of Type C
  w : ℝ  -- Price of Type D
  sneaker_cost : ℝ := 200
  initial_savings : ℝ := 35
  final_savings : ℝ := 55

/-- The main theorem stating the conditions for the action figure sale. -/
theorem action_figure_sale_conditions (sale : ActionFigureSale) :
  12 * sale.x + 8 * sale.y + 5 * sale.z + 10 * sale.w = sale.sneaker_cost - sale.initial_savings + sale.final_savings ∧
  sale.x / 4 = sale.y / 3 ∧
  sale.x / 4 = sale.z / 2 ∧
  sale.x / 4 = sale.w / 1 :=
by
  sorry

#check action_figure_sale_conditions

end NUMINAMATH_CALUDE_action_figure_sale_conditions_l1321_132110


namespace NUMINAMATH_CALUDE_quadrilateral_perimeter_l1321_132189

/-- A quadrilateral ABCD with specific properties -/
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)
  (perpendicular : (B.1 - A.1) * (C.1 - B.1) + (B.2 - A.2) * (C.2 - B.2) = 0)
  (parallel : (D.1 - C.1) * (B.2 - A.2) = (D.2 - C.2) * (B.1 - A.1))
  (AB_length : Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 7)
  (DC_length : Real.sqrt ((D.1 - C.1)^2 + (D.2 - C.2)^2) = 6)
  (BC_length : Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) = 10)

/-- The perimeter of the quadrilateral ABCD is 35.2 cm -/
theorem quadrilateral_perimeter (q : Quadrilateral) :
  Real.sqrt ((q.B.1 - q.A.1)^2 + (q.B.2 - q.A.2)^2) +
  Real.sqrt ((q.C.1 - q.B.1)^2 + (q.C.2 - q.B.2)^2) +
  Real.sqrt ((q.D.1 - q.C.1)^2 + (q.D.2 - q.C.2)^2) +
  Real.sqrt ((q.A.1 - q.D.1)^2 + (q.A.2 - q.D.2)^2) = 35.2 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_perimeter_l1321_132189


namespace NUMINAMATH_CALUDE_first_round_cookies_count_l1321_132169

/-- Represents the number of cookies sold in each round -/
structure CookieSales where
  first_round : ℕ
  second_round : ℕ

/-- Calculates the total number of cookies sold -/
def total_cookies (sales : CookieSales) : ℕ :=
  sales.first_round + sales.second_round

/-- Theorem: Given the total cookies sold and the number sold in the second round,
    we can determine the number sold in the first round -/
theorem first_round_cookies_count 
  (sales : CookieSales) 
  (h1 : sales.second_round = 27) 
  (h2 : total_cookies sales = 61) : 
  sales.first_round = 34 := by
  sorry

end NUMINAMATH_CALUDE_first_round_cookies_count_l1321_132169


namespace NUMINAMATH_CALUDE_inclination_angle_range_l1321_132193

-- Define the slope range
def slope_range : Set ℝ := {k : ℝ | -Real.sqrt 3 ≤ k ∧ k ≤ Real.sqrt 3 / 3}

-- Define the inclination angle range
def angle_range : Set ℝ := {α : ℝ | (0 ≤ α ∧ α ≤ Real.pi / 6) ∨ (2 * Real.pi / 3 ≤ α ∧ α < Real.pi)}

-- Theorem statement
theorem inclination_angle_range (k : ℝ) (α : ℝ) :
  k ∈ slope_range → α = Real.arctan k → α ∈ angle_range := by sorry

end NUMINAMATH_CALUDE_inclination_angle_range_l1321_132193


namespace NUMINAMATH_CALUDE_arithmetic_computation_l1321_132160

theorem arithmetic_computation : -12 * 5 - (-4 * -2) + (-15 * -3) / 3 = -53 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l1321_132160


namespace NUMINAMATH_CALUDE_sqrt_equality_condition_l1321_132188

theorem sqrt_equality_condition (a b c : ℝ) :
  Real.sqrt (4 * a^2 + 9 * b^2) = 2 * a + 3 * b + c ↔
  12 * a * b + 4 * a * c + 6 * b * c + c^2 = 0 ∧ 2 * a + 3 * b + c ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equality_condition_l1321_132188


namespace NUMINAMATH_CALUDE_parallel_segments_and_midpoint_l1321_132154

/-- Given four points on a Cartesian plane, if two line segments formed by these points are parallel,
    then we can determine the y-coordinate of one point and the midpoint of one segment. -/
theorem parallel_segments_and_midpoint
  (A B X Y : ℝ × ℝ)
  (hA : A = (-6, 2))
  (hB : B = (2, -6))
  (hX : X = (4, 16))
  (hY : Y = (20, k))
  (h_parallel : (B.2 - A.2) / (B.1 - A.1) = (Y.2 - X.2) / (Y.1 - X.1)) :
  k = 0 ∧ ((X.1 + Y.1) / 2, (X.2 + Y.2) / 2) = (12, 8) :=
by sorry

end NUMINAMATH_CALUDE_parallel_segments_and_midpoint_l1321_132154


namespace NUMINAMATH_CALUDE_quadratic_symmetry_l1321_132173

/-- A quadratic function with axis of symmetry at x = 6 and p(0) = -3 -/
def p (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_symmetry (a b c : ℝ) :
  (∀ x, p a b c (6 + x) = p a b c (6 - x)) →  -- axis of symmetry at x = 6
  p a b c 0 = -3 →                           -- p(0) = -3
  p a b c 12 = -3 :=                         -- p(12) = -3
by
  sorry

end NUMINAMATH_CALUDE_quadratic_symmetry_l1321_132173


namespace NUMINAMATH_CALUDE_candy_has_nine_pencils_l1321_132168

-- Define variables
def candy_pencils : ℕ := sorry
def caleb_pencils : ℕ := sorry
def calen_original_pencils : ℕ := sorry
def calen_final_pencils : ℕ := sorry

-- Define conditions
axiom caleb_pencils_def : caleb_pencils = 2 * candy_pencils - 3
axiom calen_original_pencils_def : calen_original_pencils = caleb_pencils + 5
axiom calen_final_pencils_def : calen_final_pencils = calen_original_pencils - 10
axiom calen_final_pencils_value : calen_final_pencils = 10

-- Theorem to prove
theorem candy_has_nine_pencils : candy_pencils = 9 := by sorry

end NUMINAMATH_CALUDE_candy_has_nine_pencils_l1321_132168


namespace NUMINAMATH_CALUDE_surface_area_of_specific_solid_l1321_132112

/-- A right prism with equilateral triangular bases -/
structure RightPrism where
  height : ℝ
  base_side : ℝ

/-- A solid formed by slicing off the top of the prism -/
structure SlicedSolid where
  prism : RightPrism

/-- The surface area of the sliced solid -/
def surface_area (solid : SlicedSolid) : ℝ :=
  sorry

/-- Theorem stating the surface area of the specific sliced solid -/
theorem surface_area_of_specific_solid :
  let prism := RightPrism.mk 20 10
  let solid := SlicedSolid.mk prism
  surface_area solid = 50 + (25 * Real.sqrt 3) / 4 + (5 * Real.sqrt 118.75) / 2 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_of_specific_solid_l1321_132112


namespace NUMINAMATH_CALUDE_monotone_cubic_implies_nonneg_a_l1321_132131

/-- A function f : ℝ → ℝ is monotonically increasing if for all x, y ∈ ℝ, x < y implies f(x) < f(y) -/
def MonotonicallyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

/-- The cubic function with parameter a -/
def f (a : ℝ) : ℝ → ℝ := λ x => x^3 + a*x

theorem monotone_cubic_implies_nonneg_a :
  ∀ a : ℝ, MonotonicallyIncreasing (f a) → a ≥ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_monotone_cubic_implies_nonneg_a_l1321_132131


namespace NUMINAMATH_CALUDE_rationalize_denominator_l1321_132129

theorem rationalize_denominator : 7 / Real.sqrt 175 = Real.sqrt 7 / 5 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l1321_132129


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l1321_132182

-- Define the inequality function
def f (x : ℝ) := (x - 2) * (x + 1)

-- State the theorem
theorem solution_set_of_inequality :
  {x : ℝ | f x < 0} = Set.Ioo (-1 : ℝ) 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l1321_132182


namespace NUMINAMATH_CALUDE_largest_solution_of_equation_l1321_132117

theorem largest_solution_of_equation (x : ℝ) :
  (x^2 - x - 72) / (x - 9) = 5 / (x + 4) →
  x ≤ -3 :=
by sorry

end NUMINAMATH_CALUDE_largest_solution_of_equation_l1321_132117


namespace NUMINAMATH_CALUDE_intersection_condition_minimum_condition_l1321_132175

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then x^3 - x^2 else a * x * Real.exp x

-- Theorem for the range of m
theorem intersection_condition (a : ℝ) (h : a > 0) :
  ∀ m : ℝ, (∃! x : ℝ, x ∈ Set.Ioo 0 2 ∧ f a x = m) ↔ (0 ≤ m ∧ m ≤ 4) ∨ m = -4/27 :=
sorry

-- Theorem for the range of a
theorem minimum_condition :
  ∀ a : ℝ, (∀ x : ℝ, f a x ≥ -a) ↔ a ≥ 4/27 :=
sorry

end NUMINAMATH_CALUDE_intersection_condition_minimum_condition_l1321_132175


namespace NUMINAMATH_CALUDE_warm_production_time_l1321_132153

/-- Represents the production time for flower pots -/
structure PotProduction where
  cold_time : ℕ  -- Time to produce a pot when machine is cold (in minutes)
  warm_time : ℕ  -- Time to produce a pot when machine is warm (in minutes)
  hour_length : ℕ  -- Length of a production hour (in minutes)
  extra_pots : ℕ  -- Additional pots produced in the last hour compared to the first

/-- Theorem stating the warm production time given the conditions -/
theorem warm_production_time (p : PotProduction) 
  (h1 : p.cold_time = 6)
  (h2 : p.hour_length = 60)
  (h3 : p.extra_pots = 2)
  (h4 : p.hour_length / p.cold_time + p.extra_pots = p.hour_length / p.warm_time) :
  p.warm_time = 5 := by
  sorry

#check warm_production_time

end NUMINAMATH_CALUDE_warm_production_time_l1321_132153


namespace NUMINAMATH_CALUDE_perfect_square_polynomial_l1321_132176

theorem perfect_square_polynomial (x a : ℝ) :
  (x + a) * (x + 2 * a) * (x + 3 * a) * (x + 4 * a) + a^4 = (x^2 + 5 * a * x + 5 * a^2)^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_polynomial_l1321_132176


namespace NUMINAMATH_CALUDE_cubic_equation_root_l1321_132151

theorem cubic_equation_root (a b : ℚ) :
  ((-2 : ℝ) - 5 * Real.sqrt 3) ^ 3 + a * ((-2 : ℝ) - 5 * Real.sqrt 3) ^ 2 + 
  b * ((-2 : ℝ) - 5 * Real.sqrt 3) + 49 = 0 →
  a = 235 / 71 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_root_l1321_132151


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1321_132190

-- Define set A
def A : Set ℝ := {x | |x + 3| + |x - 4| ≤ 9}

-- Define set B
def B : Set ℝ := {x | ∃ t > 0, x = 4*t + 1/t - 6}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {x | -2 ≤ x ∧ x ≤ 5} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1321_132190


namespace NUMINAMATH_CALUDE_sqrt_sum_inequality_l1321_132158

theorem sqrt_sum_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  Real.sqrt (a + 1/2) + Real.sqrt (b + 1/2) ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_inequality_l1321_132158


namespace NUMINAMATH_CALUDE_article_font_pages_l1321_132161

theorem article_font_pages (total_words : ℕ) (large_font_words : ℕ) (small_font_words : ℕ) (total_pages : ℕ) :
  total_words = 48000 →
  large_font_words = 1800 →
  small_font_words = 2400 →
  total_pages = 21 →
  ∃ (large_pages : ℕ) (small_pages : ℕ),
    large_pages + small_pages = total_pages ∧
    large_pages * large_font_words + small_pages * small_font_words = total_words ∧
    large_pages = 4 :=
by sorry

end NUMINAMATH_CALUDE_article_font_pages_l1321_132161


namespace NUMINAMATH_CALUDE_sum_equals_twelve_l1321_132185

theorem sum_equals_twelve 
  (a b c : ℕ) 
  (h : 28 * a + 30 * b + 31 * c = 365) : 
  a + b + c = 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_twelve_l1321_132185


namespace NUMINAMATH_CALUDE_yearly_fluid_intake_l1321_132196

def weekday_soda : ℕ := 5 * 12
def weekday_water : ℕ := 64
def weekday_juice : ℕ := 3 * 8
def weekday_sports : ℕ := 2 * 16

def weekend_soda : ℕ := 5 * 12
def weekend_water : ℕ := 64
def weekend_juice : ℕ := 3 * 8
def weekend_sports : ℕ := 1 * 16
def weekend_smoothie : ℕ := 32

def weekdays : ℕ := 260
def weekend_days : ℕ := 104
def holidays : ℕ := 1

def weekday_total : ℕ := weekday_soda + weekday_water + weekday_juice + weekday_sports
def weekend_total : ℕ := weekend_soda + weekend_water + weekend_juice + weekend_sports + weekend_smoothie

theorem yearly_fluid_intake :
  weekday_total * weekdays + weekend_total * (weekend_days + holidays) = 67380 := by
  sorry

end NUMINAMATH_CALUDE_yearly_fluid_intake_l1321_132196


namespace NUMINAMATH_CALUDE_next_four_valid_numbers_l1321_132135

/-- Represents a bag of milk with a unique number -/
structure BagOfMilk where
  number : Nat
  h_number : number ≤ 850

/-- Checks if a number is valid for bag selection -/
def isValidNumber (n : Nat) : Bool :=
  n ≥ 1 ∧ n ≤ 850

/-- Selects the next valid numbers from a given sequence -/
def selectNextValidNumbers (sequence : List Nat) (count : Nat) : List Nat :=
  sequence.filter isValidNumber |>.take count

theorem next_four_valid_numbers 
  (sequence : List Nat)
  (h_sequence : sequence = [614, 593, 379, 242, 203, 722, 104, 887, 088]) :
  selectNextValidNumbers (sequence.drop 4) 4 = [203, 722, 104, 088] := by
  sorry

#eval selectNextValidNumbers [614, 593, 379, 242, 203, 722, 104, 887, 088] 4

end NUMINAMATH_CALUDE_next_four_valid_numbers_l1321_132135


namespace NUMINAMATH_CALUDE_unique_triple_solution_l1321_132118

theorem unique_triple_solution : 
  ∃! (a b c : ℕ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    a + b * c = 2010 ∧ 
    b + c * a = 250 ∧
    a = 3 ∧ b = 223 ∧ c = 9 := by
  sorry

end NUMINAMATH_CALUDE_unique_triple_solution_l1321_132118


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l1321_132120

theorem partial_fraction_decomposition (x A B C : ℝ) :
  (x + 2) / (x^3 - 9*x^2 + 14*x + 24) = A / (x - 4) + B / (x - 3) + C / ((x + 2)^2) →
  A = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l1321_132120


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1321_132128

/-- Given that the solution set of ax^2 - 5x + b > 0 is {x | -3 < x < 2},
    prove that the solution set of bx^2 - 5x + a > 0 is {x | x < -1/3 or x > 1/2} -/
theorem quadratic_inequality_solution_set 
  (a b : ℝ) 
  (h : ∀ x : ℝ, ax^2 - 5*x + b > 0 ↔ -3 < x ∧ x < 2) :
  ∀ x : ℝ, b*x^2 - 5*x + a > 0 ↔ x < -1/3 ∨ x > 1/2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1321_132128


namespace NUMINAMATH_CALUDE_infinite_solutions_imply_d_equals_five_l1321_132178

theorem infinite_solutions_imply_d_equals_five :
  (∀ (d : ℝ), (∃ (S : Set ℝ), Set.Infinite S ∧ ∀ y ∈ S, 3 * (5 + d * y) = 15 * y + 15) → d = 5) :=
by sorry

end NUMINAMATH_CALUDE_infinite_solutions_imply_d_equals_five_l1321_132178


namespace NUMINAMATH_CALUDE_correct_sampling_methods_l1321_132144

-- Define the community structure
structure Community where
  high_income : Nat
  middle_income : Nat
  low_income : Nat

-- Define the student group
structure StudentGroup where
  total : Nat

-- Define sampling methods
inductive SamplingMethod
  | Stratified
  | SimpleRandom
  | Systematic

-- Define the function to determine the correct sampling method for the community survey
def community_sampling_method (c : Community) (sample_size : Nat) : SamplingMethod :=
  sorry

-- Define the function to determine the correct sampling method for the student survey
def student_sampling_method (s : StudentGroup) (sample_size : Nat) : SamplingMethod :=
  sorry

-- Theorem stating the correct sampling methods for both surveys
theorem correct_sampling_methods 
  (community : Community)
  (students : StudentGroup) :
  community_sampling_method {high_income := 100, middle_income := 210, low_income := 90} 100 = SamplingMethod.Stratified ∧
  student_sampling_method {total := 10} 3 = SamplingMethod.SimpleRandom :=
sorry

end NUMINAMATH_CALUDE_correct_sampling_methods_l1321_132144


namespace NUMINAMATH_CALUDE_grape_rate_calculation_l1321_132150

/-- The rate per kg for grapes -/
def grape_rate : ℝ := 68

/-- The amount of grapes purchased in kg -/
def grape_amount : ℝ := 7

/-- The rate per kg for mangoes -/
def mango_rate : ℝ := 48

/-- The amount of mangoes purchased in kg -/
def mango_amount : ℝ := 9

/-- The total amount paid -/
def total_paid : ℝ := 908

theorem grape_rate_calculation :
  grape_rate * grape_amount + mango_rate * mango_amount = total_paid :=
by sorry

end NUMINAMATH_CALUDE_grape_rate_calculation_l1321_132150


namespace NUMINAMATH_CALUDE_first_team_speed_calculation_l1321_132167

/-- The speed of the first team in miles per hour -/
def first_team_speed : ℝ := 20

/-- The speed of the second team in miles per hour -/
def second_team_speed : ℝ := 30

/-- The radio range in miles -/
def radio_range : ℝ := 125

/-- The time until radio contact is lost in hours -/
def time_until_lost_contact : ℝ := 2.5

theorem first_team_speed_calculation :
  first_team_speed = (radio_range / time_until_lost_contact) - second_team_speed := by
  sorry

#check first_team_speed_calculation

end NUMINAMATH_CALUDE_first_team_speed_calculation_l1321_132167


namespace NUMINAMATH_CALUDE_race_outcomes_l1321_132137

theorem race_outcomes (n : ℕ) (k : ℕ) (h : n = 6 ∧ k = 4) :
  Nat.factorial n / Nat.factorial (n - k) = 360 := by
  sorry

end NUMINAMATH_CALUDE_race_outcomes_l1321_132137


namespace NUMINAMATH_CALUDE_division_remainder_l1321_132113

theorem division_remainder (x : ℕ) (h : 23 / x = 7) : 23 % x = 2 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_l1321_132113


namespace NUMINAMATH_CALUDE_xiaoming_characters_proof_l1321_132152

theorem xiaoming_characters_proof : 
  ∀ (N : ℕ),
  (N / 2 - 50 : ℕ) + -- Day 1
  ((N / 2 + 50) / 2 - 20 : ℕ) + -- Day 2
  (((N / 4 + 45 : ℕ) / 2 + 10) : ℕ) + -- Day 3
  60 + -- Day 4
  40 = N → -- Remaining characters
  N = 700 := by
sorry

end NUMINAMATH_CALUDE_xiaoming_characters_proof_l1321_132152


namespace NUMINAMATH_CALUDE_spaceship_age_conversion_l1321_132199

/-- Converts an octal number represented as a list of digits to its decimal equivalent. -/
def octal_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (8 ^ i)) 0

/-- The octal representation of the spaceship's age -/
def spaceship_age_octal : List Nat := [3, 5, 1]

theorem spaceship_age_conversion :
  octal_to_decimal spaceship_age_octal = 233 := by
  sorry

end NUMINAMATH_CALUDE_spaceship_age_conversion_l1321_132199


namespace NUMINAMATH_CALUDE_mistaken_division_correction_l1321_132116

theorem mistaken_division_correction (N : ℕ) : 
  N % 23 = 17 ∧ N / 23 = 3 → (N / 32) + (N % 32) = 24 := by
sorry

end NUMINAMATH_CALUDE_mistaken_division_correction_l1321_132116


namespace NUMINAMATH_CALUDE_max_value_of_f_l1321_132126

def f (x : ℝ) : ℝ := -(x + 1)^2 + 5

theorem max_value_of_f :
  ∀ x : ℝ, f x ≤ 5 ∧ ∃ x₀ : ℝ, f x₀ = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_l1321_132126


namespace NUMINAMATH_CALUDE_intersection_point_equality_l1321_132102

-- Define the functions
def f (x : ℝ) : ℝ := 20 * x^3 + 19 * x^2
def g (x : ℝ) : ℝ := 20 * x^2 + 19 * x
def h (x : ℝ) : ℝ := 20 * x + 19

-- Theorem statement
theorem intersection_point_equality :
  ∀ x : ℝ, g x = h x → f x = g x :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_point_equality_l1321_132102


namespace NUMINAMATH_CALUDE_dinner_bill_proof_l1321_132166

theorem dinner_bill_proof (total_friends : Nat) (paying_friends : Nat) (extra_payment : Real) (total_bill : Real) : 
  total_friends = 10 →
  paying_friends = 9 →
  extra_payment = 3 →
  paying_friends * (total_bill / total_friends + extra_payment) = total_bill →
  total_bill = 270 := by
sorry

end NUMINAMATH_CALUDE_dinner_bill_proof_l1321_132166


namespace NUMINAMATH_CALUDE_lunas_budget_l1321_132123

/-- Luna's monthly budget problem -/
theorem lunas_budget (H F : ℝ) : 
  H + F = 240 →  -- Total budget for house rental and food
  H + F + 0.1 * F = 249 →  -- Total budget including phone bill
  F / H = 0.6  -- Food budget is 60% of house rental budget
:= by sorry

end NUMINAMATH_CALUDE_lunas_budget_l1321_132123


namespace NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l1321_132195

theorem condition_sufficient_not_necessary (a b : ℝ) :
  (∀ a b : ℝ, (abs a < 1 ∧ abs b < 1) → abs (1 - a * b) > abs (a - b)) ∧
  (∃ a b : ℝ, abs (1 - a * b) > abs (a - b) ∧ ¬(abs a < 1 ∧ abs b < 1)) :=
by sorry

end NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l1321_132195


namespace NUMINAMATH_CALUDE_saddle_value_l1321_132177

theorem saddle_value (total_value : ℝ) (horse_saddle_ratio : ℝ) :
  total_value = 100 →
  horse_saddle_ratio = 7 →
  ∃ (saddle_value : ℝ),
    saddle_value + horse_saddle_ratio * saddle_value = total_value ∧
    saddle_value = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_saddle_value_l1321_132177


namespace NUMINAMATH_CALUDE_mary_warm_hours_l1321_132163

/-- The number of sticks of wood produced by chopping up a chair. -/
def sticksPerChair : ℕ := 6

/-- The number of sticks of wood produced by chopping up a table. -/
def sticksPerTable : ℕ := 9

/-- The number of sticks of wood produced by chopping up a stool. -/
def sticksPerStool : ℕ := 2

/-- The number of sticks of wood Mary needs to burn per hour to stay warm. -/
def sticksPerHour : ℕ := 5

/-- The number of chairs Mary chops up. -/
def numChairs : ℕ := 18

/-- The number of tables Mary chops up. -/
def numTables : ℕ := 6

/-- The number of stools Mary chops up. -/
def numStools : ℕ := 4

/-- Theorem stating that Mary can keep warm for 34 hours with the firewood from the chopped furniture. -/
theorem mary_warm_hours : 
  (numChairs * sticksPerChair + numTables * sticksPerTable + numStools * sticksPerStool) / sticksPerHour = 34 := by
  sorry


end NUMINAMATH_CALUDE_mary_warm_hours_l1321_132163


namespace NUMINAMATH_CALUDE_interest_difference_relation_l1321_132140

/-- Represents the compound interest scenario -/
structure CompoundInterest where
  P : ℝ  -- Principal amount
  r : ℝ  -- Interest rate

/-- Calculate the difference in compound interest between year 2 and year 1 -/
def interestDifference (ci : CompoundInterest) : ℝ :=
  ci.P * ci.r^2

/-- The theorem stating the relationship between the original and tripled interest rate scenarios -/
theorem interest_difference_relation (ci : CompoundInterest) :
  interestDifference { P := ci.P, r := 3 * ci.r } = 360 →
  interestDifference ci = 40 :=
by
  sorry

#check interest_difference_relation

end NUMINAMATH_CALUDE_interest_difference_relation_l1321_132140


namespace NUMINAMATH_CALUDE_inverse_function_point_l1321_132170

open Real

theorem inverse_function_point (f : ℝ → ℝ) (h_inv : Function.Bijective f) :
  tan (π / 3) - f 2 = Real.sqrt 3 - 1 / 3 →
  Function.invFun f (1 / 3) - π / 2 = 2 - π / 2 := by
sorry

end NUMINAMATH_CALUDE_inverse_function_point_l1321_132170


namespace NUMINAMATH_CALUDE_max_equilateral_triangles_l1321_132164

/-- Represents a matchstick configuration --/
structure MatchstickConfig where
  num_matchsticks : ℕ
  connected_end_to_end : Bool

/-- Represents the number of equilateral triangles in a configuration --/
def num_equilateral_triangles (config : MatchstickConfig) : ℕ := sorry

/-- The theorem stating the maximum number of equilateral triangles --/
theorem max_equilateral_triangles (config : MatchstickConfig) 
  (h1 : config.num_matchsticks = 6) 
  (h2 : config.connected_end_to_end = true) : 
  ∃ (n : ℕ), n ≤ 4 ∧ ∀ (m : ℕ), num_equilateral_triangles config ≤ m → n ≤ m :=
sorry

end NUMINAMATH_CALUDE_max_equilateral_triangles_l1321_132164


namespace NUMINAMATH_CALUDE_smallest_three_digit_palindrome_non_palindromic_product_l1321_132174

/-- A function that checks if a number is a three-digit palindrome -/
def isThreeDigitPalindrome (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ (n / 100 = n % 10)

/-- A function that checks if a number is a five-digit palindrome -/
def isFiveDigitPalindrome (n : ℕ) : Prop :=
  10000 ≤ n ∧ n ≤ 99999 ∧ (n / 10000 = n % 10) ∧ ((n / 1000) % 10 = (n / 10) % 10)

/-- The main theorem stating that 707 is the smallest three-digit palindrome
    whose product with 103 is not a five-digit palindrome -/
theorem smallest_three_digit_palindrome_non_palindromic_product :
  (∀ n : ℕ, isThreeDigitPalindrome n ∧ n < 707 → isFiveDigitPalindrome (n * 103)) ∧
  isThreeDigitPalindrome 707 ∧
  ¬isFiveDigitPalindrome (707 * 103) :=
sorry

end NUMINAMATH_CALUDE_smallest_three_digit_palindrome_non_palindromic_product_l1321_132174
