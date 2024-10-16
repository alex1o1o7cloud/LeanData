import Mathlib

namespace NUMINAMATH_CALUDE_pentagon_perimeter_is_49_l1401_140102

/-- The perimeter of a pentagon with given side lengths -/
def pentagon_perimeter (x y z : ℝ) : ℝ :=
  3*x + 5*y + 6*z + 4*x + 7*y

/-- Theorem: The perimeter of the specified pentagon is 49 cm -/
theorem pentagon_perimeter_is_49 :
  pentagon_perimeter 1 2 3 = 49 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_perimeter_is_49_l1401_140102


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1401_140116

theorem negation_of_universal_proposition :
  (¬ ∀ (m : ℝ) (x : ℝ), m ∈ Set.Icc 0 1 → x + 1/x ≥ 2^m) ↔
  (∃ (m : ℝ) (x : ℝ), m ∈ Set.Icc 0 1 ∧ x + 1/x < 2^m) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1401_140116


namespace NUMINAMATH_CALUDE_tank_insulation_cost_l1401_140163

/-- Calculates the surface area of a rectangular prism -/
def surfaceArea (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)

/-- Calculates the cost of insulating a rectangular tank -/
def insulationCost (l w h costPerSqFt : ℝ) : ℝ :=
  surfaceArea l w h * costPerSqFt

theorem tank_insulation_cost :
  let l : ℝ := 6
  let w : ℝ := 3
  let h : ℝ := 2
  let costPerSqFt : ℝ := 20
  insulationCost l w h costPerSqFt = 1440 := by
  sorry

#eval insulationCost 6 3 2 20

end NUMINAMATH_CALUDE_tank_insulation_cost_l1401_140163


namespace NUMINAMATH_CALUDE_square_existence_l1401_140133

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line in 2D space (ax + by + c = 0)
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a square
structure Square where
  side1 : Line2D
  side2 : Line2D
  side3 : Line2D
  side4 : Line2D

-- Function to check if a point lies on a line
def pointOnLine (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Function to check if three points are collinear
def areCollinear (p1 p2 p3 : Point2D) : Prop :=
  (p2.y - p1.y) * (p3.x - p2.x) = (p3.y - p2.y) * (p2.x - p1.x)

-- Theorem statement
theorem square_existence
  (A B C D : Point2D)
  (h_not_collinear : ¬(areCollinear A B C ∨ areCollinear A B D ∨ areCollinear A C D ∨ areCollinear B C D)) :
  ∃ (s : Square),
    pointOnLine A s.side1 ∧
    pointOnLine B s.side2 ∧
    pointOnLine C s.side3 ∧
    pointOnLine D s.side4 :=
sorry

end NUMINAMATH_CALUDE_square_existence_l1401_140133


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l1401_140118

theorem partial_fraction_decomposition :
  ∃ (C D : ℚ), C = 17/7 ∧ D = 11/7 ∧
  ∀ (x : ℚ), x ≠ 5 ∧ x ≠ -2 →
    (4*x - 3) / (x^2 - 3*x - 10) = C / (x - 5) + D / (x + 2) :=
by sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l1401_140118


namespace NUMINAMATH_CALUDE_percentage_equality_l1401_140166

theorem percentage_equality (x y : ℝ) (h : (18 / 100) * x = (9 / 100) * y) :
  (12 / 100) * x = (6 / 100) * y := by
  sorry

end NUMINAMATH_CALUDE_percentage_equality_l1401_140166


namespace NUMINAMATH_CALUDE_prime_relation_l1401_140164

theorem prime_relation (p q : ℕ) : 
  Nat.Prime p ∧ 
  p = Nat.minFac (Nat.minFac 2) ∧ 
  q = 13 * p + 3 ∧ 
  Nat.Prime q → 
  q = 29 := by sorry

end NUMINAMATH_CALUDE_prime_relation_l1401_140164


namespace NUMINAMATH_CALUDE_tangent_line_equality_implies_m_equals_5_l1401_140158

open Real

theorem tangent_line_equality_implies_m_equals_5 
  (f g : ℝ → ℝ) 
  (hf : ∀ x > 0, f x = x^2 - m) 
  (hg : ∀ x > 0, g x = 6 * log x - 4 * x) 
  (h_common : ∃ x₀ > 0, f x₀ = g x₀) 
  (h_tangent : ∀ x₀ > 0, f x₀ = g x₀ → (deriv f) x₀ = (deriv g) x₀) :
  m = 5 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_equality_implies_m_equals_5_l1401_140158


namespace NUMINAMATH_CALUDE_rectangle_area_l1401_140138

/-- Given a rectangle with perimeter 28 cm and width 6 cm, prove its area is 48 square cm. -/
theorem rectangle_area (perimeter width : ℝ) (h1 : perimeter = 28) (h2 : width = 6) :
  let length := (perimeter - 2 * width) / 2
  width * length = 48 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_l1401_140138


namespace NUMINAMATH_CALUDE_all_statements_false_l1401_140152

def sharp (n : ℕ) : ℚ := 1 / (n + 1)

theorem all_statements_false :
  (sharp 4 + sharp 8 ≠ sharp 12) ∧
  (sharp 9 - sharp 3 ≠ sharp 6) ∧
  (sharp 5 * sharp 7 ≠ sharp 35) ∧
  (sharp 15 / sharp 3 ≠ sharp 5) := by
sorry

end NUMINAMATH_CALUDE_all_statements_false_l1401_140152


namespace NUMINAMATH_CALUDE_units_digit_of_fraction_l1401_140140

def numerator : ℕ := 25 * 26 * 27 * 28 * 29 * 30
def denominator : ℕ := 1250

theorem units_digit_of_fraction : (numerator / denominator) % 10 = 2 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_fraction_l1401_140140


namespace NUMINAMATH_CALUDE_certain_number_exists_l1401_140103

theorem certain_number_exists : ∃ x : ℝ, 0.35 * x - (1/3) * (0.35 * x) = 42 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_exists_l1401_140103


namespace NUMINAMATH_CALUDE_spoiled_apples_count_l1401_140176

def total_apples : ℕ := 7
def prob_at_least_one_spoiled : ℚ := 2857142857142857 / 10000000000000000

theorem spoiled_apples_count (S : ℕ) : 
  S < total_apples → 
  (1 : ℚ) - (↑(total_apples - S) / ↑total_apples) * (↑(total_apples - S - 1) / ↑(total_apples - 1)) = prob_at_least_one_spoiled → 
  S = 1 := by sorry

end NUMINAMATH_CALUDE_spoiled_apples_count_l1401_140176


namespace NUMINAMATH_CALUDE_complex_power_approximation_l1401_140136

/-- Prove that (3 * cos(30°) + 3i * sin(30°))^8 is approximately equal to -3281 - 3281i * √3 -/
theorem complex_power_approximation :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ 
  Complex.abs ((3 * Complex.cos (30 * π / 180) + 3 * Complex.I * Complex.sin (30 * π / 180))^8 - 
               (-3281 - 3281 * Complex.I * Real.sqrt 3)) < ε :=
by sorry

end NUMINAMATH_CALUDE_complex_power_approximation_l1401_140136


namespace NUMINAMATH_CALUDE_chord_slope_l1401_140141

theorem chord_slope (k : ℝ) : 
  (∃ (x y : ℝ), x^2 + (y-1)^2 = 3 ∧ y = k*x - 1) ∧ 
  (∃ (x1 y1 x2 y2 : ℝ), 
    x1^2 + (y1-1)^2 = 3 ∧ y1 = k*x1 - 1 ∧
    x2^2 + (y2-1)^2 = 3 ∧ y2 = k*x2 - 1 ∧
    (x1-x2)^2 + (y1-y2)^2 = 4) →
  k = 1 ∨ k = -1 :=
by sorry

end NUMINAMATH_CALUDE_chord_slope_l1401_140141


namespace NUMINAMATH_CALUDE_parabola_equation_l1401_140112

/-- Prove that for a parabola y² = 2px (p > 0) with focus F(p/2, 0), if there exists a point A 
on the parabola such that AF = 4 and a point B(0, 2) on the y-axis satisfying BA · BF = 0, 
then p = 4. -/
theorem parabola_equation (p : ℝ) (h_p : p > 0) : 
  ∃ (A : ℝ × ℝ), 
    (A.2)^2 = 2 * p * A.1 ∧  -- A is on the parabola
    (A.1 - p/2)^2 + (A.2)^2 = 16 ∧  -- AF = 4
    (A.1 * p/2 + A.2 * (-2)) = 0  -- BA · BF = 0
  → p = 4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_l1401_140112


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l1401_140194

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (2 * x^2 - 8 * x - 10 = 5 * x + 20) → 
  ∃ (y : ℝ), (2 * y^2 - 8 * y - 10 = 5 * y + 20) ∧ (x + y = 13/2) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l1401_140194


namespace NUMINAMATH_CALUDE_backpacking_cooks_l1401_140161

/-- The number of ways to choose k items from n items --/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of people in the group --/
def total_people : ℕ := 10

/-- The number of people willing to cook --/
def eligible_people : ℕ := total_people - 1

/-- The number of cooks needed --/
def cooks_needed : ℕ := 2

theorem backpacking_cooks : choose eligible_people cooks_needed = 36 := by
  sorry

end NUMINAMATH_CALUDE_backpacking_cooks_l1401_140161


namespace NUMINAMATH_CALUDE_gcd_problem_l1401_140135

theorem gcd_problem (h : Nat.Prime 97) : Nat.gcd (97^9 + 1) (97^9 + 97^2 + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l1401_140135


namespace NUMINAMATH_CALUDE_bobs_walking_rate_l1401_140108

/-- Proves that Bob's walking rate is 5 miles per hour given the problem conditions -/
theorem bobs_walking_rate
  (total_distance : ℝ)
  (yolanda_rate : ℝ)
  (bob_start_delay : ℝ)
  (bob_distance : ℝ)
  (h1 : total_distance = 60)
  (h2 : yolanda_rate = 5)
  (h3 : bob_start_delay = 1)
  (h4 : bob_distance = 30) :
  bob_distance / (total_distance / yolanda_rate - bob_start_delay) = 5 :=
by sorry

end NUMINAMATH_CALUDE_bobs_walking_rate_l1401_140108


namespace NUMINAMATH_CALUDE_mod_seven_equality_l1401_140187

theorem mod_seven_equality : (45^1234 - 25^1234) % 7 = 5 := by
  sorry

end NUMINAMATH_CALUDE_mod_seven_equality_l1401_140187


namespace NUMINAMATH_CALUDE_final_position_after_nine_swaps_l1401_140192

/-- Represents the position of a racer -/
inductive Position
| first
| second
| third

/-- Represents a racer in the race -/
structure Racer :=
  (position : Position)

/-- Swaps the position of a racer -/
def swap_position (r : Racer) : Racer :=
  match r.position with
  | Position.first => { position := Position.second }
  | Position.second => { position := Position.third }
  | Position.third => { position := Position.second }

/-- Applies n swaps to a racer's position -/
def apply_swaps (r : Racer) (n : Nat) : Racer :=
  match n with
  | 0 => r
  | n + 1 => swap_position (apply_swaps r n)

theorem final_position_after_nine_swaps (r : Racer) :
  r.position = Position.third →
  (apply_swaps r 9).position = Position.second :=
by
  sorry

#check final_position_after_nine_swaps

end NUMINAMATH_CALUDE_final_position_after_nine_swaps_l1401_140192


namespace NUMINAMATH_CALUDE_lottery_winnings_l1401_140159

theorem lottery_winnings (total_given : ℝ) (num_students : ℕ) (fraction : ℝ) 
  (h1 : total_given = 15525)
  (h2 : num_students = 100)
  (h3 : fraction = 1 / 1000) : 
  total_given / (num_students * fraction) = 155250 := by
  sorry

end NUMINAMATH_CALUDE_lottery_winnings_l1401_140159


namespace NUMINAMATH_CALUDE_power_27_mod_13_l1401_140142

theorem power_27_mod_13 : 27^482 ≡ 1 [ZMOD 13] := by sorry

end NUMINAMATH_CALUDE_power_27_mod_13_l1401_140142


namespace NUMINAMATH_CALUDE_scientific_notation_conversion_l1401_140170

theorem scientific_notation_conversion :
  (380180000000 : ℝ) = 3.8018 * (10 : ℝ)^11 :=
by sorry

end NUMINAMATH_CALUDE_scientific_notation_conversion_l1401_140170


namespace NUMINAMATH_CALUDE_mirella_purple_books_l1401_140188

/-- The number of pages in each purple book -/
def purple_book_pages : ℕ := 230

/-- The number of pages in each orange book -/
def orange_book_pages : ℕ := 510

/-- The number of orange books Mirella read -/
def orange_books_read : ℕ := 4

/-- The difference between orange and purple pages read -/
def page_difference : ℕ := 890

/-- The number of purple books Mirella read -/
def purple_books_read : ℕ := 5

theorem mirella_purple_books :
  orange_books_read * orange_book_pages - purple_books_read * purple_book_pages = page_difference :=
by sorry

end NUMINAMATH_CALUDE_mirella_purple_books_l1401_140188


namespace NUMINAMATH_CALUDE_triangle_side_length_l1401_140195

theorem triangle_side_length 
  (A B C : ℝ) 
  (a b c : ℝ) 
  (h_area : (1/2) * a * c * Real.sin B = Real.sqrt 3)
  (h_angle : B = π/3)
  (h_sides : a^2 + c^2 = 3*a*c) :
  b = 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1401_140195


namespace NUMINAMATH_CALUDE_fourth_month_sale_l1401_140171

def sales_4_months : List Int := [6335, 6927, 6855, 6562]
def sale_6th_month : Int := 5091
def average_sale : Int := 6500
def num_months : Int := 6

theorem fourth_month_sale :
  let total_sales := average_sale * num_months
  let sum_known_sales := (sales_4_months.sum + sale_6th_month)
  total_sales - sum_known_sales = 7230 := by sorry

end NUMINAMATH_CALUDE_fourth_month_sale_l1401_140171


namespace NUMINAMATH_CALUDE_probability_seven_chairs_probability_n_chairs_l1401_140146

/-- The probability of three knights being seated at a round table with empty chairs on either side of each knight. -/
def knight_seating_probability (n : ℕ) : ℚ :=
  if n = 7 then 1 / 35
  else if n ≥ 6 then (n - 4) * (n - 5) / ((n - 1) * (n - 2))
  else 0

/-- Theorem stating the probability for 7 chairs -/
theorem probability_seven_chairs :
  knight_seating_probability 7 = 1 / 35 := by sorry

/-- Theorem stating the probability for n chairs (n ≥ 6) -/
theorem probability_n_chairs (n : ℕ) (h : n ≥ 6) :
  knight_seating_probability n = (n - 4) * (n - 5) / ((n - 1) * (n - 2)) := by sorry

end NUMINAMATH_CALUDE_probability_seven_chairs_probability_n_chairs_l1401_140146


namespace NUMINAMATH_CALUDE_problem_solution_l1401_140127

theorem problem_solution (x y : ℝ) (h1 : x + 3*y = 6) (h2 : x*y = -12) : 
  x^2 + 9*y^2 = 108 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1401_140127


namespace NUMINAMATH_CALUDE_continuity_not_implies_differentiability_l1401_140180

-- Define a real-valued function
variable (f : ℝ → ℝ)

-- Define a point in the real line
variable (x₀ : ℝ)

-- Theorem statement
theorem continuity_not_implies_differentiability :
  ∃ f : ℝ → ℝ, ∃ x₀ : ℝ, ContinuousAt f x₀ ∧ ¬DifferentiableAt ℝ f x₀ := by
  sorry

end NUMINAMATH_CALUDE_continuity_not_implies_differentiability_l1401_140180


namespace NUMINAMATH_CALUDE_complex_number_problem_l1401_140100

theorem complex_number_problem (α β : ℂ) 
  (h1 : (α + β).re > 0)
  (h2 : (Complex.I * (α - 3 * β)).re > 0)
  (h3 : β = 4 + 3 * Complex.I) :
  α = 12 - 3 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_problem_l1401_140100


namespace NUMINAMATH_CALUDE_greg_total_distance_l1401_140197

/-- The total distance Greg travels given his individual trip distances -/
theorem greg_total_distance (d1 d2 d3 : ℝ) 
  (h1 : d1 = 30) -- Distance from workplace to farmer's market
  (h2 : d2 = 20) -- Distance from farmer's market to friend's house
  (h3 : d3 = 25) -- Distance from friend's house to home
  : d1 + d2 + d3 = 75 := by sorry

end NUMINAMATH_CALUDE_greg_total_distance_l1401_140197


namespace NUMINAMATH_CALUDE_find_B_l1401_140193

theorem find_B (A C B : ℤ) (h1 : A = 634) (h2 : A = C + 593) (h3 : B = C + 482) : B = 523 := by
  sorry

end NUMINAMATH_CALUDE_find_B_l1401_140193


namespace NUMINAMATH_CALUDE_rose_needs_more_l1401_140124

def paintbrush_cost : ℚ := 2.40
def paints_cost : ℚ := 9.20
def easel_cost : ℚ := 6.50
def rose_has : ℚ := 7.10

theorem rose_needs_more : 
  paintbrush_cost + paints_cost + easel_cost - rose_has = 11 :=
by sorry

end NUMINAMATH_CALUDE_rose_needs_more_l1401_140124


namespace NUMINAMATH_CALUDE_no_square_base_b_l1401_140172

theorem no_square_base_b : ¬ ∃ (b : ℤ), ∃ (n : ℤ), b^2 + 3*b + 1 = n^2 := by sorry

end NUMINAMATH_CALUDE_no_square_base_b_l1401_140172


namespace NUMINAMATH_CALUDE_tadpoles_kept_calculation_l1401_140182

/-- The number of tadpoles Trent kept, given the initial number and percentage released -/
def tadpoles_kept (x : ℝ) : ℝ :=
  x * (1 - 0.825)

/-- Theorem stating that the number of tadpoles kept is 0.175 * x -/
theorem tadpoles_kept_calculation (x : ℝ) :
  tadpoles_kept x = 0.175 * x := by
  sorry

end NUMINAMATH_CALUDE_tadpoles_kept_calculation_l1401_140182


namespace NUMINAMATH_CALUDE_bus_and_walking_problem_l1401_140145

/-- Proof of the bus and walking problem -/
theorem bus_and_walking_problem
  (total_distance : ℝ)
  (walking_speed : ℝ)
  (bus_speed : ℝ)
  (rest_time : ℝ)
  (h1 : total_distance = 21)
  (h2 : walking_speed = 4)
  (h3 : bus_speed = 60)
  (h4 : rest_time = 1/6) -- 10 minutes in hours
  : ∃ (x y : ℝ),
    x + y = total_distance ∧
    x / bus_speed + total_distance / bus_speed = rest_time + y / walking_speed ∧
    x = 19 ∧
    y = 2 := by
  sorry


end NUMINAMATH_CALUDE_bus_and_walking_problem_l1401_140145


namespace NUMINAMATH_CALUDE_banana_arrangement_count_l1401_140121

/-- The number of distinct arrangements of the letters in "BANANA" -/
def banana_arrangements : ℕ := 60

/-- The total number of letters in "BANANA" -/
def total_letters : ℕ := 6

/-- The number of 'A's in "BANANA" -/
def num_a : ℕ := 3

/-- The number of 'N's in "BANANA" -/
def num_n : ℕ := 2

/-- The number of 'B's in "BANANA" -/
def num_b : ℕ := 1

theorem banana_arrangement_count :
  banana_arrangements = Nat.factorial total_letters / (Nat.factorial num_a * Nat.factorial num_n) :=
by sorry

end NUMINAMATH_CALUDE_banana_arrangement_count_l1401_140121


namespace NUMINAMATH_CALUDE_problem_solution_l1401_140196

theorem problem_solution (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_prod : x * y * z = 1)
  (h_eq1 : x + 1 / z = 6)
  (h_eq2 : y + 1 / x = 15) :
  z + 1 / y = 23 / 89 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1401_140196


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1401_140154

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  h_d : d ≠ 0
  h_arith : ∀ n, a (n + 1) = a n + d

/-- The theorem statement -/
theorem arithmetic_sequence_problem (seq : ArithmeticSequence)
  (h1 : seq.a 1 + seq.a 3 = 8)
  (h2 : seq.a 4 ^ 2 = seq.a 2 * seq.a 9) :
  seq.a 5 = 13 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1401_140154


namespace NUMINAMATH_CALUDE_pascal_triangle_61_row_third_number_l1401_140130

theorem pascal_triangle_61_row_third_number : 
  let n : ℕ := 60  -- The row number (61 numbers means it's the 60th row, 0-indexed)
  let k : ℕ := 2   -- The position of the number we're interested in (3rd number, 0-indexed)
  Nat.choose n k = 1770 := by
sorry

end NUMINAMATH_CALUDE_pascal_triangle_61_row_third_number_l1401_140130


namespace NUMINAMATH_CALUDE_proportion_problem_l1401_140199

theorem proportion_problem (x y : ℚ) 
  (h1 : (3/4 : ℚ) / x = 5 / 7)
  (h2 : y / 19 = 11 / 3) :
  x = 21/20 ∧ y = 209/3 := by
  sorry

end NUMINAMATH_CALUDE_proportion_problem_l1401_140199


namespace NUMINAMATH_CALUDE_expression_evaluation_l1401_140147

theorem expression_evaluation : (-6)^6 / 6^4 + 4^5 - 7^2 * 2 = 890 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1401_140147


namespace NUMINAMATH_CALUDE_ternary_to_decimal_l1401_140101

theorem ternary_to_decimal :
  (1 * 3^2 + 2 * 3^1 + 1 * 3^0 : ℕ) = 16 := by sorry

end NUMINAMATH_CALUDE_ternary_to_decimal_l1401_140101


namespace NUMINAMATH_CALUDE_theresa_final_count_l1401_140185

/-- Represents the number of crayons each person has -/
structure CrayonCount where
  theresa : ℕ
  janice : ℕ
  nancy : ℕ
  mark : ℕ

/-- Represents the initial state and actions taken -/
def initial_state : CrayonCount := {
  theresa := 32,
  janice := 12,
  nancy := 0,
  mark := 0
}

/-- Janice shares half of her crayons with Nancy and gives 3 to Mark -/
def share_crayons (state : CrayonCount) : CrayonCount := {
  theresa := state.theresa,
  janice := state.janice - (state.janice / 2) - 3,
  nancy := state.nancy + (state.janice / 2),
  mark := state.mark + 3
}

/-- Nancy gives 8 crayons to Theresa -/
def give_to_theresa (state : CrayonCount) : CrayonCount := {
  theresa := state.theresa + 8,
  janice := state.janice,
  nancy := state.nancy - 8,
  mark := state.mark
}

/-- The final state after all actions -/
def final_state : CrayonCount := give_to_theresa (share_crayons initial_state)

theorem theresa_final_count : final_state.theresa = 40 := by
  sorry

end NUMINAMATH_CALUDE_theresa_final_count_l1401_140185


namespace NUMINAMATH_CALUDE_chocolate_candies_cost_l1401_140198

theorem chocolate_candies_cost (box_size : ℕ) (box_cost : ℚ) (total_candies : ℕ) : 
  box_size = 30 → box_cost = 7.5 → total_candies = 450 → 
  (total_candies / box_size : ℚ) * box_cost = 112.5 := by
sorry

end NUMINAMATH_CALUDE_chocolate_candies_cost_l1401_140198


namespace NUMINAMATH_CALUDE_jesse_blocks_left_l1401_140122

/-- The number of building blocks Jesse has left after constructing various structures --/
def blocks_left (initial : ℕ) (building : ℕ) (farmhouse : ℕ) (fence : ℕ) : ℕ :=
  initial - (building + farmhouse + fence)

/-- Theorem stating that Jesse has 84 blocks left --/
theorem jesse_blocks_left :
  blocks_left 344 80 123 57 = 84 := by
  sorry

end NUMINAMATH_CALUDE_jesse_blocks_left_l1401_140122


namespace NUMINAMATH_CALUDE_system1_solution_system2_solution_l1401_140178

-- System 1
theorem system1_solution :
  ∃ (x y : ℝ), y = x + 1 ∧ x + y = 5 ∧ x = 2 ∧ y = 3 := by sorry

-- System 2
theorem system2_solution :
  ∃ (x y : ℝ), x + 2*y = 9 ∧ 3*x - 2*y = -1 ∧ x = 2 ∧ y = 3.5 := by sorry

end NUMINAMATH_CALUDE_system1_solution_system2_solution_l1401_140178


namespace NUMINAMATH_CALUDE_opposite_sides_range_l1401_140120

def line_equation (x y a : ℝ) : ℝ := 3 * x - 2 * y + a

theorem opposite_sides_range (a : ℝ) : 
  (line_equation 3 1 a) * (line_equation (-4) 6 a) < 0 ↔ -7 < a ∧ a < 24 := by sorry

end NUMINAMATH_CALUDE_opposite_sides_range_l1401_140120


namespace NUMINAMATH_CALUDE_equation_solution_l1401_140148

theorem equation_solution : ∃ (x : ℚ), (3/4 : ℚ) + 1/x = 7/8 ∧ x = 8 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1401_140148


namespace NUMINAMATH_CALUDE_customers_in_other_countries_l1401_140126

theorem customers_in_other_countries 
  (total_customers : ℕ) 
  (us_customers : ℕ) 
  (h1 : total_customers = 7422) 
  (h2 : us_customers = 723) : 
  total_customers - us_customers = 6699 := by
  sorry

end NUMINAMATH_CALUDE_customers_in_other_countries_l1401_140126


namespace NUMINAMATH_CALUDE_midpoint_chain_l1401_140183

theorem midpoint_chain (A B C D E F G : ℝ) : 
  C = (A + B) / 2 →
  D = (A + C) / 2 →
  E = (A + D) / 2 →
  F = (A + E) / 2 →
  G = (A + F) / 2 →
  G - A = 5 →
  B - A = 160 := by
sorry

end NUMINAMATH_CALUDE_midpoint_chain_l1401_140183


namespace NUMINAMATH_CALUDE_magnitude_of_sum_l1401_140113

/-- Given real x, vectors a and b, with a parallel to b, 
    prove that the magnitude of their sum is √5 -/
theorem magnitude_of_sum (x : ℝ) (a b : ℝ × ℝ) :
  a = (x, 1) →
  b = (4, -2) →
  ∃ (k : ℝ), a = k • b →
  ‖a + b‖ = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_sum_l1401_140113


namespace NUMINAMATH_CALUDE_regular_polygon_144_degree_angles_has_10_sides_l1401_140137

/-- A regular polygon with interior angles of 144 degrees has 10 sides. -/
theorem regular_polygon_144_degree_angles_has_10_sides :
  ∀ n : ℕ,
  n > 2 →
  (180 * (n - 2) : ℝ) = 144 * n →
  n = 10 :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_144_degree_angles_has_10_sides_l1401_140137


namespace NUMINAMATH_CALUDE_shortest_distance_parabola_to_line_l1401_140156

/-- The shortest distance between a point on the parabola y = x^2 - 6x + 11 and the line y = 2x - 5 -/
theorem shortest_distance_parabola_to_line :
  let parabola := fun x : ℝ => x^2 - 6*x + 11
  let line := fun x : ℝ => 2*x - 5
  let distance := fun a : ℝ => |2*a - (a^2 - 6*a + 11) - 5| / Real.sqrt 5
  ∃ (min_dist : ℝ), min_dist = 16 * Real.sqrt 5 / 5 ∧
    ∀ a : ℝ, distance a ≥ min_dist :=
by sorry


end NUMINAMATH_CALUDE_shortest_distance_parabola_to_line_l1401_140156


namespace NUMINAMATH_CALUDE_percent_equality_l1401_140175

theorem percent_equality (x : ℝ) : (60 / 100 * 600 = 50 / 100 * x) → x = 720 := by
  sorry

end NUMINAMATH_CALUDE_percent_equality_l1401_140175


namespace NUMINAMATH_CALUDE_negation_of_proposition_l1401_140190

theorem negation_of_proposition (a b x : ℝ) :
  (¬(x ≥ a^2 + b^2 → x ≥ 2*a*b)) ↔ (x < a^2 + b^2 → x < 2*a*b) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l1401_140190


namespace NUMINAMATH_CALUDE_factor_of_polynomial_l1401_140123

theorem factor_of_polynomial (x : ℝ) : 
  ∃ (q : ℝ → ℝ), 4 * x^3 + 6 * x^2 + 11 * x - 6 = (x - 1/2) * q x := by
  sorry

end NUMINAMATH_CALUDE_factor_of_polynomial_l1401_140123


namespace NUMINAMATH_CALUDE_f_properties_l1401_140129

-- Define the function f
def f (a b x : ℝ) : ℝ := 3 * a * x^2 - 2 * (a + b) * x + b

-- State the theorem
theorem f_properties (a b : ℝ) (ha : a > 0) :
  -- Part I
  (b = 1/2 ∧ 
   ∃ x₁ x₂ : ℝ, 0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 1 ∧ 
   f a (1/2) x₁ = |x₁ - 1/2| ∧ 
   f a (1/2) x₂ = |x₂ - 1/2|) →
  a ≥ 1 ∧
  -- Part II
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |f a b 0| ≤ 2 ∧ |f a b 1| ≤ 2) →
  ∃ M : ℝ, M = 2 ∧ ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |f a b x| ≤ M :=
by
  sorry

end NUMINAMATH_CALUDE_f_properties_l1401_140129


namespace NUMINAMATH_CALUDE_f_odd_and_increasing_l1401_140143

def f (x : ℝ) : ℝ := x^3

theorem f_odd_and_increasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_f_odd_and_increasing_l1401_140143


namespace NUMINAMATH_CALUDE_regular_octagon_angles_l1401_140106

theorem regular_octagon_angles :
  ∀ (n : ℕ) (interior_angle exterior_angle : ℝ),
    n = 8 →
    interior_angle = (180 * (n - 2 : ℝ)) / n →
    exterior_angle = 180 - interior_angle →
    interior_angle = 135 ∧ exterior_angle = 45 := by
  sorry

end NUMINAMATH_CALUDE_regular_octagon_angles_l1401_140106


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l1401_140117

theorem triangle_angle_calculation (y : ℝ) : 
  y > 0 ∧ y < 60 ∧ 45 + 3 * y + y = 180 → y = 33.75 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l1401_140117


namespace NUMINAMATH_CALUDE_linear_function_max_min_sum_l1401_140157

theorem linear_function_max_min_sum (a : ℝ) : 
  (∀ x ∈ Set.Icc 0 1, a * x ≤ max (a * 0) (a * 1) ∧ min (a * 0) (a * 1) ≤ a * x) →
  max (a * 0) (a * 1) + min (a * 0) (a * 1) = 3 →
  a = 3 := by sorry

end NUMINAMATH_CALUDE_linear_function_max_min_sum_l1401_140157


namespace NUMINAMATH_CALUDE_digit_removal_theorem_l1401_140186

theorem digit_removal_theorem :
  (∀ (n : ℕ), n ≥ 2 → 
    (∃! (x : ℕ), x = 625 * 10^(n-2) ∧ 
      (∃ (m : ℕ), x = 6 * 10^n + m ∧ m = x / 25))) ∧
  (¬ ∃ (x : ℕ), ∃ (n : ℕ), ∃ (m : ℕ), 
    x = 6 * 10^n + m ∧ m = x / 35) :=
by sorry

end NUMINAMATH_CALUDE_digit_removal_theorem_l1401_140186


namespace NUMINAMATH_CALUDE_shaded_area_ratio_l1401_140153

theorem shaded_area_ratio : 
  ∀ (r₁ r₂ r₃ r₄ : ℝ), 
    r₁ = 1 → r₂ = 2 → r₃ = 3 → r₄ = 4 →
    (π * r₁^2 + π * r₃^2 - π * r₂^2) / (π * r₄^2) = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_ratio_l1401_140153


namespace NUMINAMATH_CALUDE_two_less_than_negative_one_l1401_140184

theorem two_less_than_negative_one : -1 - 2 = -3 := by
  sorry

end NUMINAMATH_CALUDE_two_less_than_negative_one_l1401_140184


namespace NUMINAMATH_CALUDE_phils_remaining_pages_l1401_140114

/-- Given an initial number of books, pages per book, and books lost,
    calculate the total number of pages remaining. -/
def remaining_pages (initial_books : ℕ) (pages_per_book : ℕ) (books_lost : ℕ) : ℕ :=
  (initial_books - books_lost) * pages_per_book

/-- Theorem stating that with 10 initial books, 100 pages per book,
    and 2 books lost, the remaining pages total 800. -/
theorem phils_remaining_pages :
  remaining_pages 10 100 2 = 800 := by
  sorry

end NUMINAMATH_CALUDE_phils_remaining_pages_l1401_140114


namespace NUMINAMATH_CALUDE_prob_end_two_tails_after_second_head_l1401_140168

/-- A fair coin flip can result in either heads or tails with equal probability -/
def FairCoin : Type := Bool

/-- The outcome of a sequence of coin flips -/
inductive FlipOutcome
| TwoHeads
| TwoTails
| Incomplete

/-- The state of the coin flipping process -/
structure FlipState :=
  (seenSecondHead : Bool)
  (lastFlip : Option Bool)
  (outcome : FlipOutcome)

/-- Simulates a single coin flip and updates the state -/
def flipCoin (state : FlipState) : FlipState := sorry

/-- Calculates the probability of ending with two tails after seeing the second head -/
def probEndTwoTailsAfterSecondHead : ℝ := sorry

/-- The main theorem to prove -/
theorem prob_end_two_tails_after_second_head :
  probEndTwoTailsAfterSecondHead = 1 / 24 := by sorry

end NUMINAMATH_CALUDE_prob_end_two_tails_after_second_head_l1401_140168


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1401_140151

def A : Set ℝ := {x : ℝ | x^2 + x = 0}
def B : Set ℝ := {x : ℝ | x^2 - x = 0}

theorem intersection_of_A_and_B : A ∩ B = {0} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1401_140151


namespace NUMINAMATH_CALUDE_book_words_per_page_l1401_140105

theorem book_words_per_page 
  (total_pages : ℕ)
  (words_per_page : ℕ)
  (max_words_per_page : ℕ)
  (total_words_mod : ℕ)
  (h1 : total_pages = 224)
  (h2 : words_per_page ≤ max_words_per_page)
  (h3 : max_words_per_page = 150)
  (h4 : (total_pages * words_per_page) % 253 = total_words_mod)
  (h5 : total_words_mod = 156) :
  words_per_page = 106 := by
sorry

end NUMINAMATH_CALUDE_book_words_per_page_l1401_140105


namespace NUMINAMATH_CALUDE_candy_mix_cost_per_pound_l1401_140125

/-- Proves that the desired cost per pound of mixed candy is $2.00 given the specified conditions --/
theorem candy_mix_cost_per_pound
  (total_weight : ℝ)
  (cost_A : ℝ)
  (cost_B : ℝ)
  (weight_A : ℝ)
  (h_total_weight : total_weight = 5)
  (h_cost_A : cost_A = 3.2)
  (h_cost_B : cost_B = 1.7)
  (h_weight_A : weight_A = 1)
  : (weight_A * cost_A + (total_weight - weight_A) * cost_B) / total_weight = 2 := by
  sorry

#check candy_mix_cost_per_pound

end NUMINAMATH_CALUDE_candy_mix_cost_per_pound_l1401_140125


namespace NUMINAMATH_CALUDE_gcd_150_450_l1401_140155

theorem gcd_150_450 : Nat.gcd 150 450 = 150 := by
  sorry

end NUMINAMATH_CALUDE_gcd_150_450_l1401_140155


namespace NUMINAMATH_CALUDE_min_value_of_trig_function_l1401_140115

open Real

theorem min_value_of_trig_function :
  ∃ (x : ℝ), ∀ (y : ℝ), 2 * sin (π / 3 - x) - cos (π / 6 + x) ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_trig_function_l1401_140115


namespace NUMINAMATH_CALUDE_distance_between_squares_l1401_140162

/-- Given a configuration of two squares where:
    - The smaller square has a perimeter of 8 cm
    - The larger square has an area of 49 cm²
    This theorem states that the distance between point A (top-right corner of the larger square)
    and point B (top-left corner of the smaller square) is approximately 10.3 cm. -/
theorem distance_between_squares (small_square_perimeter : ℝ) (large_square_area : ℝ)
    (h1 : small_square_perimeter = 8)
    (h2 : large_square_area = 49) :
    ∃ (distance : ℝ), abs (distance - Real.sqrt 106) < 0.1 ∧
    distance = Real.sqrt ((large_square_area.sqrt + small_square_perimeter / 4) ^ 2 +
    (large_square_area.sqrt - small_square_perimeter / 4) ^ 2) := by
  sorry

end NUMINAMATH_CALUDE_distance_between_squares_l1401_140162


namespace NUMINAMATH_CALUDE_division_problem_solution_l1401_140128

/-- Represents the division problem with given conditions -/
structure DivisionProblem where
  D : ℕ  -- dividend
  d : ℕ  -- divisor
  q : ℕ  -- quotient
  r : ℕ  -- remainder
  P : ℕ  -- prime number
  h1 : D = d * q + r
  h2 : r = 6
  h3 : d = 5 * q
  h4 : d = 3 * r + 2
  h5 : ∃ k : ℕ, D = P * k
  h6 : ∃ n : ℕ, q = n * n
  h7 : Nat.Prime P

theorem division_problem_solution (prob : DivisionProblem) : prob.D = 86 ∧ ∃ k : ℕ, prob.D = prob.P * k := by
  sorry

#check division_problem_solution

end NUMINAMATH_CALUDE_division_problem_solution_l1401_140128


namespace NUMINAMATH_CALUDE_number_of_gharials_l1401_140160

/-- Represents the number of flies eaten per day by one frog -/
def flies_per_frog : ℕ := 30

/-- Represents the number of frogs eaten per day by one fish -/
def frogs_per_fish : ℕ := 8

/-- Represents the number of fish eaten per day by one gharial -/
def fish_per_gharial : ℕ := 15

/-- Represents the total number of flies eaten per day in the swamp -/
def total_flies_eaten : ℕ := 32400

/-- Proves that the number of gharials in the swamp is 9 -/
theorem number_of_gharials : ℕ := by
  sorry

end NUMINAMATH_CALUDE_number_of_gharials_l1401_140160


namespace NUMINAMATH_CALUDE_butterflies_let_go_l1401_140173

theorem butterflies_let_go (original : ℕ) (left : ℕ) (h1 : original = 93) (h2 : left = 82) :
  original - left = 11 := by
  sorry

end NUMINAMATH_CALUDE_butterflies_let_go_l1401_140173


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_l1401_140111

theorem vector_sum_magnitude (a b : ℝ × ℝ) :
  ‖a‖ = 1 →
  b = (Real.sqrt 3, 1) →
  a • b = 0 →
  ‖2 • a + b‖ = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_magnitude_l1401_140111


namespace NUMINAMATH_CALUDE_sandy_marks_calculation_l1401_140189

theorem sandy_marks_calculation :
  ∀ (total_attempts : ℕ) (correct_attempts : ℕ) (marks_per_correct : ℕ) (marks_per_incorrect : ℕ),
    total_attempts = 30 →
    correct_attempts = 24 →
    marks_per_correct = 3 →
    marks_per_incorrect = 2 →
    (correct_attempts * marks_per_correct) - ((total_attempts - correct_attempts) * marks_per_incorrect) = 60 :=
by sorry

end NUMINAMATH_CALUDE_sandy_marks_calculation_l1401_140189


namespace NUMINAMATH_CALUDE_books_taken_out_on_tuesday_l1401_140107

/-- Prove that the number of books taken out on Tuesday is 120, given the initial and final number of books in the library and the changes on Wednesday and Thursday. -/
theorem books_taken_out_on_tuesday (initial_books : ℕ) (final_books : ℕ) (returned_wednesday : ℕ) (withdrawn_thursday : ℕ) 
  (h_initial : initial_books = 250)
  (h_final : final_books = 150)
  (h_wednesday : returned_wednesday = 35)
  (h_thursday : withdrawn_thursday = 15) :
  initial_books - final_books + returned_wednesday - withdrawn_thursday = 120 := by
  sorry

end NUMINAMATH_CALUDE_books_taken_out_on_tuesday_l1401_140107


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1401_140150

def arithmetic_sequence (a₁ d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

theorem arithmetic_sequence_problem (a₁ d : ℤ) :
  arithmetic_sequence a₁ d 5 = 10 ∧
  (arithmetic_sequence a₁ d 1 + arithmetic_sequence a₁ d 2 + arithmetic_sequence a₁ d 3 = 3) →
  a₁ = -2 ∧ d = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1401_140150


namespace NUMINAMATH_CALUDE_kevins_cards_l1401_140181

/-- Kevin's card problem -/
theorem kevins_cards (initial_cards found_cards : ℕ) 
  (h1 : initial_cards = 65)
  (h2 : found_cards = 539) :
  initial_cards + found_cards = 604 := by
  sorry

end NUMINAMATH_CALUDE_kevins_cards_l1401_140181


namespace NUMINAMATH_CALUDE_minimum_bottles_needed_l1401_140104

def small_bottle_capacity : ℚ := 45
def large_bottle_1_capacity : ℚ := 630
def large_bottle_2_capacity : ℚ := 850

theorem minimum_bottles_needed : 
  ∃ (n : ℕ), n * small_bottle_capacity ≥ large_bottle_1_capacity + large_bottle_2_capacity ∧
  ∀ (m : ℕ), m * small_bottle_capacity ≥ large_bottle_1_capacity + large_bottle_2_capacity → m ≥ n ∧
  n = 33 := by
  sorry

end NUMINAMATH_CALUDE_minimum_bottles_needed_l1401_140104


namespace NUMINAMATH_CALUDE_solve_inequality_when_a_is_5_range_of_a_for_always_positive_l1401_140174

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 6

-- Statement for part I
theorem solve_inequality_when_a_is_5 :
  {x : ℝ | f 5 x < 0} = {x : ℝ | -3 < x ∧ x < -2} := by sorry

-- Statement for part II
theorem range_of_a_for_always_positive :
  ∀ a : ℝ, (∀ x : ℝ, f a x > 0) ↔ a ∈ Set.Ioo (-2 * Real.sqrt 6) (2 * Real.sqrt 6) := by sorry

end NUMINAMATH_CALUDE_solve_inequality_when_a_is_5_range_of_a_for_always_positive_l1401_140174


namespace NUMINAMATH_CALUDE_binary_to_octal_conversion_l1401_140149

/-- Converts a binary number represented as a list of bits to a natural number. -/
def binary_to_natural (bits : List Bool) : ℕ :=
  bits.foldr (fun b n => 2 * n + if b then 1 else 0) 0

/-- Converts a natural number to its octal representation as a list of digits. -/
def natural_to_octal (n : ℕ) : List ℕ :=
  if n < 8 then [n]
  else (n % 8) :: natural_to_octal (n / 8)

/-- The binary representation of the number to be converted. -/
def binary_number : List Bool := [true, true, true, false, false]

/-- The expected octal representation. -/
def expected_octal : List ℕ := [4, 3]

theorem binary_to_octal_conversion :
  natural_to_octal (binary_to_natural binary_number) = expected_octal := by
  sorry

#eval binary_to_natural binary_number
#eval natural_to_octal (binary_to_natural binary_number)

end NUMINAMATH_CALUDE_binary_to_octal_conversion_l1401_140149


namespace NUMINAMATH_CALUDE_sum_of_positive_numbers_l1401_140109

theorem sum_of_positive_numbers (a b : ℝ) : 
  a > 0 → b > 0 → (a + b) / (a^2 + a*b + b^2) = 4/49 → a + b = 16 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_positive_numbers_l1401_140109


namespace NUMINAMATH_CALUDE_ring_weights_sum_to_total_l1401_140131

/-- The weight of the orange ring in ounces -/
def orange_weight : ℚ := 0.08333333333333333

/-- The weight of the purple ring in ounces -/
def purple_weight : ℚ := 0.3333333333333333

/-- The weight of the white ring in ounces -/
def white_weight : ℚ := 0.4166666666666667

/-- The total weight of all rings in ounces -/
def total_weight : ℚ := 0.8333333333333333

/-- Theorem stating that the sum of individual ring weights equals the total weight -/
theorem ring_weights_sum_to_total : 
  orange_weight + purple_weight + white_weight = total_weight := by
  sorry

end NUMINAMATH_CALUDE_ring_weights_sum_to_total_l1401_140131


namespace NUMINAMATH_CALUDE_fraction_product_result_l1401_140191

def fraction_product (n : ℕ) : ℚ :=
  if n < 6 then 1
  else (n : ℚ) / (n + 5) * fraction_product (n - 1)

theorem fraction_product_result : fraction_product 95 = 1 / 75287520 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_result_l1401_140191


namespace NUMINAMATH_CALUDE_specific_cone_measurements_l1401_140165

/-- Represents a cone formed from a circular sector -/
structure SectorCone where
  circle_radius : ℝ
  sector_angle : ℝ

/-- Calculate the volume of the cone divided by π -/
def volume_div_pi (cone : SectorCone) : ℝ :=
  sorry

/-- Calculate the lateral surface area of the cone divided by π -/
def lateral_area_div_pi (cone : SectorCone) : ℝ :=
  sorry

/-- Theorem stating the volume and lateral surface area for a specific cone -/
theorem specific_cone_measurements :
  let cone : SectorCone := { circle_radius := 16, sector_angle := 270 }
  volume_div_pi cone = 384 ∧ lateral_area_div_pi cone = 192 := by
  sorry

end NUMINAMATH_CALUDE_specific_cone_measurements_l1401_140165


namespace NUMINAMATH_CALUDE_remainder_of_factorial_sum_l1401_140119

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem remainder_of_factorial_sum (n : ℕ) (h : n ≥ 100) :
  (sum_factorials n) % 30 = (sum_factorials 4) % 30 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_factorial_sum_l1401_140119


namespace NUMINAMATH_CALUDE_total_beakers_l1401_140139

theorem total_beakers (copper_beakers : ℕ) (drops_per_test : ℕ) (total_drops : ℕ) (non_copper_tested : ℕ) :
  copper_beakers = 8 →
  drops_per_test = 3 →
  total_drops = 45 →
  non_copper_tested = 7 →
  copper_beakers + non_copper_tested = total_drops / drops_per_test :=
by sorry

end NUMINAMATH_CALUDE_total_beakers_l1401_140139


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1401_140177

theorem inequality_system_solution (x : ℝ) : 
  (2 + x > 7 - 4 * x) ∧ (x < (4 + x) / 2) → 1 < x ∧ x < 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1401_140177


namespace NUMINAMATH_CALUDE_polynomial_remainder_theorem_l1401_140144

theorem polynomial_remainder_theorem (x : ℝ) : 
  let f : ℝ → ℝ := λ x => 8*x^4 - 18*x^3 + 5*x^2 - 3*x - 30
  let g : ℝ → ℝ := λ x => 2*x - 4
  f 2 = -32 ∧ (∃ q : ℝ → ℝ, ∀ x, f x = g x * q x + f 2) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_remainder_theorem_l1401_140144


namespace NUMINAMATH_CALUDE_circle_equation_l1401_140169

/-- The standard equation of a circle with center (-2, 1) passing through (0, 1) -/
theorem circle_equation :
  let center : ℝ × ℝ := (-2, 1)
  let point_on_circle : ℝ × ℝ := (0, 1)
  ∀ (x y : ℝ),
    (x + 2)^2 + (y - 1)^2 = 4 ↔
    (x - center.1)^2 + (y - center.2)^2 = (point_on_circle.1 - center.1)^2 + (point_on_circle.2 - center.2)^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l1401_140169


namespace NUMINAMATH_CALUDE_floor_sqrt_120_l1401_140134

theorem floor_sqrt_120 : ⌊Real.sqrt 120⌋ = 10 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_120_l1401_140134


namespace NUMINAMATH_CALUDE_smallest_integer_divisible_by_18_with_sqrt_between_30_and_30_5_l1401_140110

theorem smallest_integer_divisible_by_18_with_sqrt_between_30_and_30_5 :
  ∃ n : ℕ+, (∀ m : ℕ+, m < n → ¬(18 ∣ m ∧ 30 < Real.sqrt m ∧ Real.sqrt m < 30.5)) ∧
            (18 ∣ n) ∧ (30 < Real.sqrt n) ∧ (Real.sqrt n < 30.5) ∧ n = 900 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_divisible_by_18_with_sqrt_between_30_and_30_5_l1401_140110


namespace NUMINAMATH_CALUDE_smallest_odd_six_digit_divisible_by_125_l1401_140179

def is_odd_digit (d : Nat) : Prop := d % 2 = 1 ∧ d < 10

def all_digits_odd (n : Nat) : Prop :=
  ∀ d, d ∈ n.digits 10 → is_odd_digit d

def is_six_digit (n : Nat) : Prop :=
  100000 ≤ n ∧ n ≤ 999999

theorem smallest_odd_six_digit_divisible_by_125 :
  ∀ n : Nat, is_six_digit n → all_digits_odd n → n % 125 = 0 →
  111375 ≤ n := by sorry

end NUMINAMATH_CALUDE_smallest_odd_six_digit_divisible_by_125_l1401_140179


namespace NUMINAMATH_CALUDE_anya_pancakes_l1401_140167

theorem anya_pancakes (x : ℝ) (x_pos : x > 0) : 
  let flipped := x * (2/3)
  let not_burnt := flipped * 0.6
  let not_dropped := not_burnt * 0.8
  not_dropped / x = 0.32 := by sorry

end NUMINAMATH_CALUDE_anya_pancakes_l1401_140167


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l1401_140132

theorem complex_magnitude_problem (z w : ℂ) 
  (h1 : Complex.abs z = 1)
  (h2 : Complex.abs w = 2)
  (h3 : Complex.arg z = Real.pi / 2)
  (h4 : Complex.abs (z + w) = 3) :
  Complex.abs (1 / z + 1 / w) = (Real.sqrt (5 + 2 * Real.sqrt 2)) / (2 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l1401_140132
