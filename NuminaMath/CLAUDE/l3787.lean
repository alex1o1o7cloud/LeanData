import Mathlib

namespace NUMINAMATH_CALUDE_inequality_proof_l3787_378705

theorem inequality_proof (a b c : ℝ) : (1/4) * a^2 + b^2 + c^2 ≥ a*b - a*c + 2*b*c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3787_378705


namespace NUMINAMATH_CALUDE_sum_of_powers_l3787_378755

theorem sum_of_powers : (-2)^4 + (-2)^(3/2) + (-2)^1 + 2^1 + 2^(3/2) + 2^4 = 32 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_l3787_378755


namespace NUMINAMATH_CALUDE_expression_evaluation_l3787_378765

theorem expression_evaluation (m n : ℚ) (hm : m = -1) (hn : n = 1/2) :
  (2 / (m - n) - 1 / (m + n)) / ((m * n + 3 * n^2) / (m^3 - m * n^2)) = -2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3787_378765


namespace NUMINAMATH_CALUDE_algebraic_expression_equals_one_l3787_378741

theorem algebraic_expression_equals_one
  (m n : ℝ)
  (hm : m ≠ 0)
  (hn : n ≠ 0)
  (h_diff : m - n = 1/2) :
  (m^2 - n^2) / (2*m^2 + 2*m*n) / (m - (2*m*n - n^2) / m) = 1 := by
sorry

end NUMINAMATH_CALUDE_algebraic_expression_equals_one_l3787_378741


namespace NUMINAMATH_CALUDE_complex_magnitude_theorem_l3787_378772

theorem complex_magnitude_theorem (a b : ℂ) (x : ℝ) :
  Complex.abs a = 3 →
  Complex.abs b = 5 →
  a * b = x - 6 * Complex.I →
  x = 3 * Real.sqrt 21 :=
by sorry

end NUMINAMATH_CALUDE_complex_magnitude_theorem_l3787_378772


namespace NUMINAMATH_CALUDE_move_right_result_l3787_378752

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Move a point horizontally in the Cartesian coordinate system -/
def moveHorizontal (p : Point) (units : ℝ) : Point :=
  { x := p.x + units, y := p.y }

/-- The initial point (-1, 2) -/
def initialPoint : Point :=
  { x := -1, y := 2 }

/-- The number of units to move right -/
def moveRightUnits : ℝ := 3

/-- The final point after moving -/
def finalPoint : Point := moveHorizontal initialPoint moveRightUnits

/-- Theorem: Moving the initial point 3 units to the right results in (2, 2) -/
theorem move_right_result :
  finalPoint = { x := 2, y := 2 } := by
  sorry

end NUMINAMATH_CALUDE_move_right_result_l3787_378752


namespace NUMINAMATH_CALUDE_cody_remaining_games_l3787_378763

theorem cody_remaining_games (initial_games given_away_games : ℕ) 
  (h1 : initial_games = 9)
  (h2 : given_away_games = 4) :
  initial_games - given_away_games = 5 := by
  sorry

end NUMINAMATH_CALUDE_cody_remaining_games_l3787_378763


namespace NUMINAMATH_CALUDE_revenue_change_l3787_378774

theorem revenue_change 
  (price_increase : ℝ) 
  (sales_decrease : ℝ) 
  (price_increase_percent : price_increase = 30) 
  (sales_decrease_percent : sales_decrease = 20) : 
  (1 + price_increase / 100) * (1 - sales_decrease / 100) - 1 = 0.04 := by
sorry

end NUMINAMATH_CALUDE_revenue_change_l3787_378774


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l3787_378750

-- Define a sequence
def Sequence := ℕ → ℝ

-- Define the property of being a geometric sequence
def IsGeometric (a : Sequence) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the given condition
def Condition (a : Sequence) : Prop :=
  ∀ n : ℕ, n > 1 → a n ^ 2 = a (n - 1) * a (n + 1)

-- State the theorem
theorem condition_necessary_not_sufficient :
  (∀ a : Sequence, IsGeometric a → Condition a) ∧
  (∃ a : Sequence, Condition a ∧ ¬IsGeometric a) := by
  sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l3787_378750


namespace NUMINAMATH_CALUDE_quadratic_roots_constraint_l3787_378718

/-- A quadratic function f(x) = x^2 + 2bx + c -/
def f (b c x : ℝ) : ℝ := x^2 + 2*b*x + c

/-- The equation f(x) + x + b = 0 -/
def g (b c x : ℝ) : ℝ := f b c x + x + b

theorem quadratic_roots_constraint (b c : ℝ) :
  f b c 1 = 0 ∧
  (∃ x₁ x₂, x₁ ∈ Set.Ioo (-3) (-2) ∧ x₂ ∈ Set.Ioo 0 1 ∧
    g b c x₁ = 0 ∧ g b c x₂ = 0) →
  b ∈ Set.Ioo (-5/2) (-1/2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_constraint_l3787_378718


namespace NUMINAMATH_CALUDE_negation_equivalence_l3787_378738

theorem negation_equivalence (a : ℝ) : (¬(a < 0)) ↔ (¬(a^2 > a)) := by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3787_378738


namespace NUMINAMATH_CALUDE_stratified_sampling_grade10_l3787_378732

theorem stratified_sampling_grade10 (total_students : ℕ) (grade10_students : ℕ) (sample_size : ℕ) :
  total_students = 700 →
  grade10_students = 300 →
  sample_size = 35 →
  (grade10_students * sample_size) / total_students = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_grade10_l3787_378732


namespace NUMINAMATH_CALUDE_prob_not_snowing_l3787_378773

theorem prob_not_snowing (p_snow : ℚ) (h : p_snow = 1/4) : 1 - p_snow = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_prob_not_snowing_l3787_378773


namespace NUMINAMATH_CALUDE_stock_fall_amount_l3787_378749

/-- Represents the daily change in stock value -/
structure StockChange where
  morning_rise : ℚ
  afternoon_fall : ℚ

/-- Models the stock behavior over time -/
def stock_value (initial_value : ℚ) (daily_change : StockChange) (days : ℕ) : ℚ :=
  initial_value + (daily_change.morning_rise - daily_change.afternoon_fall) * days

/-- Theorem stating the condition for the stock to reach a specific value -/
theorem stock_fall_amount (initial_value target_value : ℚ) (days : ℕ) :
  let morning_rise := 2
  ∀ afternoon_fall : ℚ,
    stock_value initial_value ⟨morning_rise, afternoon_fall⟩ (days - 1) < target_value ∧
    stock_value initial_value ⟨morning_rise, afternoon_fall⟩ days ≥ target_value →
    afternoon_fall = 98 / 99 :=
by sorry

end NUMINAMATH_CALUDE_stock_fall_amount_l3787_378749


namespace NUMINAMATH_CALUDE_negative_movement_l3787_378769

-- Define a type for direction
inductive Direction
| East
| West

-- Define a function to represent movement
def movement (distance : ℤ) : Direction :=
  if distance > 0 then Direction.East else Direction.West

-- Define the theorem
theorem negative_movement :
  (movement 30 = Direction.East) →
  (movement (-50) = Direction.West) :=
by
  sorry

end NUMINAMATH_CALUDE_negative_movement_l3787_378769


namespace NUMINAMATH_CALUDE_xy_and_expression_values_l3787_378748

theorem xy_and_expression_values (x y : ℝ) 
  (h1 : x - 2*y = 3) 
  (h2 : (x - 2)*(y + 1) = 2) : 
  x*y = 1 ∧ (x^2 - 2)*(2*y^2 - 1) = -9 := by
  sorry

end NUMINAMATH_CALUDE_xy_and_expression_values_l3787_378748


namespace NUMINAMATH_CALUDE_water_displacement_cubed_l3787_378762

/-- Given a cylindrical tank and a partially submerged cube, calculate the volume of water displaced cubed. -/
theorem water_displacement_cubed (tank_radius : ℝ) (cube_side : ℝ) (h : tank_radius = 3 ∧ cube_side = 6) : 
  let submerged_height := cube_side / 2
  let tank_diameter := 2 * tank_radius
  let inscribed_square_side := tank_diameter / Real.sqrt 2
  let intersection_area := inscribed_square_side ^ 2
  let displaced_volume := intersection_area * submerged_height
  displaced_volume ^ 3 = 157464 := by
  sorry

end NUMINAMATH_CALUDE_water_displacement_cubed_l3787_378762


namespace NUMINAMATH_CALUDE_raghu_investment_l3787_378716

theorem raghu_investment (raghu trishul vishal : ℝ) : 
  trishul = 0.9 * raghu →
  vishal = 1.1 * trishul →
  raghu + trishul + vishal = 6069 →
  raghu = 2100 := by
sorry

end NUMINAMATH_CALUDE_raghu_investment_l3787_378716


namespace NUMINAMATH_CALUDE_volume_of_rotated_region_l3787_378753

/-- The volume of the solid formed by rotating the region bounded by y = 2x - x^2 and y = 2x^2 - 4x around the x-axis. -/
theorem volume_of_rotated_region : ∃ V : ℝ,
  (∀ x y : ℝ, (y = 2*x - x^2 ∨ y = 2*x^2 - 4*x) → 
    V = π * ∫ x in (0)..(2), ((2*x^2 - 4*x)^2 - (2*x - x^2)^2)) ∧
  V = (16 * π) / 5 := by sorry

end NUMINAMATH_CALUDE_volume_of_rotated_region_l3787_378753


namespace NUMINAMATH_CALUDE_sum_of_squares_orthogonal_matrix_l3787_378760

theorem sum_of_squares_orthogonal_matrix (A : Matrix (Fin 3) (Fin 3) ℝ) 
  (h : A.transpose = A⁻¹) : 
  (A 0 0)^2 + (A 0 1)^2 + (A 0 2)^2 + 
  (A 1 0)^2 + (A 1 1)^2 + (A 1 2)^2 + 
  (A 2 0)^2 + (A 2 1)^2 + (A 2 2)^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_orthogonal_matrix_l3787_378760


namespace NUMINAMATH_CALUDE_divisibility_by_eleven_l3787_378706

theorem divisibility_by_eleven (n : ℕ) (h : Odd n) : ∃ k : ℤ, (10 : ℤ)^n + 1 = 11 * k := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_eleven_l3787_378706


namespace NUMINAMATH_CALUDE_thought_number_is_729_l3787_378795

/-- 
Given a three-digit number, if each of the numbers 109, 704, and 124 
matches it exactly in one digit place, then the number is 729.
-/
theorem thought_number_is_729 (x : ℕ) : 
  (100 ≤ x ∧ x < 1000) → 
  (∃! d : ℕ, d < 3 ∧ (x / 10^d % 10 = 109 / 10^d % 10)) →
  (∃! d : ℕ, d < 3 ∧ (x / 10^d % 10 = 704 / 10^d % 10)) →
  (∃! d : ℕ, d < 3 ∧ (x / 10^d % 10 = 124 / 10^d % 10)) →
  x = 729 := by
sorry


end NUMINAMATH_CALUDE_thought_number_is_729_l3787_378795


namespace NUMINAMATH_CALUDE_smallest_number_with_remainder_one_l3787_378731

theorem smallest_number_with_remainder_one : ∃ n : ℕ, 
  n > 0 ∧ 
  n % 9 = 0 ∧ 
  (∀ k : ℕ, 2 ≤ k ∧ k ≤ 8 → n % k = 1) ∧
  (∀ m : ℕ, 0 < m ∧ m < n → ¬(m % 9 = 0 ∧ (∀ k : ℕ, 2 ≤ k ∧ k ≤ 8 → m % k = 1))) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_remainder_one_l3787_378731


namespace NUMINAMATH_CALUDE_largest_inscribed_circle_radius_is_2_sqrt_8_l3787_378761

/-- A quadrilateral with given side lengths -/
structure Quadrilateral where
  ab : ℝ
  bc : ℝ
  cd : ℝ
  da : ℝ

/-- The radius of the largest inscribed circle in a quadrilateral -/
def largest_inscribed_circle_radius (q : Quadrilateral) : ℝ := sorry

/-- Theorem stating that the largest inscribed circle radius for the given quadrilateral is 2√8 -/
theorem largest_inscribed_circle_radius_is_2_sqrt_8 :
  let q : Quadrilateral := ⟨15, 10, 8, 13⟩
  largest_inscribed_circle_radius q = 2 * Real.sqrt 8 := by sorry

end NUMINAMATH_CALUDE_largest_inscribed_circle_radius_is_2_sqrt_8_l3787_378761


namespace NUMINAMATH_CALUDE_unique_magnitude_quadratic_l3787_378766

theorem unique_magnitude_quadratic : ∃! m : ℝ, ∀ z : ℂ, z^2 - 10*z + 50 = 0 → Complex.abs z = m := by
  sorry

end NUMINAMATH_CALUDE_unique_magnitude_quadratic_l3787_378766


namespace NUMINAMATH_CALUDE_f_has_max_and_min_iff_m_in_range_l3787_378719

/-- The function f with parameter m -/
def f (m : ℝ) (x : ℝ) : ℝ := x^3 + m*x^2 + (m + 6)*x + 1

/-- The derivative of f with respect to x -/
def f' (m : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*m*x + (m + 6)

/-- The discriminant of f' -/
def discriminant (m : ℝ) : ℝ := (2*m)^2 - 4*3*(m + 6)

theorem f_has_max_and_min_iff_m_in_range (m : ℝ) :
  (∃ (a b : ℝ), ∀ x, f m x ≤ f m a ∧ f m x ≥ f m b) ↔ 
  m < -3 ∨ m > 6 :=
sorry

end NUMINAMATH_CALUDE_f_has_max_and_min_iff_m_in_range_l3787_378719


namespace NUMINAMATH_CALUDE_right_triangle_area_rational_l3787_378736

/-- A right-angled triangle with integer coordinates -/
structure RightTriangle where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ

/-- The area of a right-angled triangle with integer coordinates -/
def area (t : RightTriangle) : ℚ :=
  (|t.a * t.d - t.b * t.c| : ℚ) / 2

/-- Theorem: The area of a right-angled triangle with integer coordinates is always rational -/
theorem right_triangle_area_rational (t : RightTriangle) : 
  ∃ (q : ℚ), area t = q :=
sorry

end NUMINAMATH_CALUDE_right_triangle_area_rational_l3787_378736


namespace NUMINAMATH_CALUDE_m_range_l3787_378784

theorem m_range (m : ℝ) : 
  (∃ x : ℝ, m * x^2 + 1 ≤ 0) ∧ 
  (∀ x : ℝ, x^2 + m * x + 1 > 0) → 
  m > -2 ∧ m < 0 :=
by sorry

end NUMINAMATH_CALUDE_m_range_l3787_378784


namespace NUMINAMATH_CALUDE_sequence_properties_l3787_378723

def sequence_a (n : ℕ) : ℕ := 2 * n - 1

def sequence_b (n : ℕ) : ℕ := 2^(n - 1)

def sum_sequence_a (n : ℕ) : ℕ := n^2

def sum_sequence_ab (n : ℕ) : ℕ := (2 * n - 3) * 2^n + 3

theorem sequence_properties :
  (∀ n, sum_sequence_a n = n^2) →
  sequence_b 2 = 2 →
  sequence_b 5 = 16 →
  (∀ n, sequence_a n = 2 * n - 1) ∧
  (∀ n, sequence_b n = 2^(n - 1)) ∧
  (∀ n, sum_sequence_ab n = (2 * n - 3) * 2^n + 3) := by
  sorry

end NUMINAMATH_CALUDE_sequence_properties_l3787_378723


namespace NUMINAMATH_CALUDE_solve_equation_l3787_378735

theorem solve_equation (m : ℝ) : (m - 4)^2 = (1/16)⁻¹ → m = 8 ∨ m = 0 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3787_378735


namespace NUMINAMATH_CALUDE_log_relation_l3787_378737

theorem log_relation (x k : ℝ) (h1 : Real.log 3 / Real.log 4 = x) (h2 : Real.log 64 / Real.log 2 = k * x) : k = 3 := by
  sorry

end NUMINAMATH_CALUDE_log_relation_l3787_378737


namespace NUMINAMATH_CALUDE_wrapping_paper_distribution_l3787_378704

theorem wrapping_paper_distribution (total : ℚ) (decoration : ℚ) (num_presents : ℕ) :
  total = 5/8 ∧ decoration = 1/24 ∧ num_presents = 4 →
  (total - decoration) / (num_presents - 1) = 7/36 := by
sorry

end NUMINAMATH_CALUDE_wrapping_paper_distribution_l3787_378704


namespace NUMINAMATH_CALUDE_gcd_values_for_special_m_l3787_378728

theorem gcd_values_for_special_m (m n : ℕ) (h : m + 6 = 9 * m) : 
  Nat.gcd m n = 3 ∨ Nat.gcd m n = 6 := by
sorry

end NUMINAMATH_CALUDE_gcd_values_for_special_m_l3787_378728


namespace NUMINAMATH_CALUDE_adultChildRatioIsTwo_l3787_378757

/-- Represents the ticket prices and attendance information for a show -/
structure ShowInfo where
  adultTicketPrice : ℚ
  childTicketPrice : ℚ
  totalReceipts : ℚ
  numAdults : ℕ

/-- Calculates the ratio of adults to children given show information -/
def adultChildRatio (info : ShowInfo) : ℚ :=
  let numChildren := (info.totalReceipts - info.adultTicketPrice * info.numAdults) / info.childTicketPrice
  info.numAdults / numChildren

/-- Theorem stating that the ratio of adults to children is 2:1 for the given show information -/
theorem adultChildRatioIsTwo (info : ShowInfo) 
    (h1 : info.adultTicketPrice = 11/2)
    (h2 : info.childTicketPrice = 5/2)
    (h3 : info.totalReceipts = 1026)
    (h4 : info.numAdults = 152) : 
  adultChildRatio info = 2 := by
  sorry

#eval adultChildRatio {
  adultTicketPrice := 11/2,
  childTicketPrice := 5/2,
  totalReceipts := 1026,
  numAdults := 152
}

end NUMINAMATH_CALUDE_adultChildRatioIsTwo_l3787_378757


namespace NUMINAMATH_CALUDE_camp_attendance_outside_county_attendance_l3787_378700

theorem camp_attendance (lawrence_camp : ℕ) (lawrence_home : ℕ) (lawrence_total : ℕ)
  (h1 : lawrence_camp = 610769)
  (h2 : lawrence_home = 590796)
  (h3 : lawrence_total = 1201565)
  (h4 : lawrence_total = lawrence_camp + lawrence_home) :
  lawrence_camp = lawrence_total - lawrence_home :=
by sorry

theorem outside_county_attendance (lawrence_camp : ℕ) (lawrence_home : ℕ) (lawrence_total : ℕ)
  (h1 : lawrence_camp = 610769)
  (h2 : lawrence_home = 590796)
  (h3 : lawrence_total = 1201565)
  (h4 : lawrence_total = lawrence_camp + lawrence_home) :
  0 = lawrence_camp - (lawrence_total - lawrence_home) :=
by sorry

end NUMINAMATH_CALUDE_camp_attendance_outside_county_attendance_l3787_378700


namespace NUMINAMATH_CALUDE_triangle_in_radius_l3787_378725

/-- Given a triangle with perimeter 36 cm and area 45 cm², prove that its in radius is 2.5 cm. -/
theorem triangle_in_radius (P : ℝ) (A : ℝ) (r : ℝ) 
  (h_perimeter : P = 36) 
  (h_area : A = 45) 
  (h_in_radius : A = r * (P / 2)) : r = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_in_radius_l3787_378725


namespace NUMINAMATH_CALUDE_quadratic_minimum_l3787_378780

/-- Given a quadratic function y = 2x^2 + px + q, 
    prove that q = 10 + p^2/8 when the minimum value of y is 10 -/
theorem quadratic_minimum (p : ℝ) :
  ∃ (q : ℝ), (∀ x : ℝ, 2 * x^2 + p * x + q ≥ 10) ∧
             (∃ x₀ : ℝ, 2 * x₀^2 + p * x₀ + q = 10) →
  q = 10 + p^2 / 8 :=
sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l3787_378780


namespace NUMINAMATH_CALUDE_solution_set_f_leq_6_range_of_a_l3787_378799

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 1| + |2*x - 3|

-- Theorem for the solution set of f(x) ≤ 6
theorem solution_set_f_leq_6 :
  {x : ℝ | f x ≤ 6} = {x : ℝ | -1 ≤ x ∧ x ≤ 2} := by sorry

-- Theorem for the range of a
theorem range_of_a :
  {a : ℝ | ∃ x, f x < |a - 1|} = {a : ℝ | a < -3 ∨ a > 5} := by sorry

end NUMINAMATH_CALUDE_solution_set_f_leq_6_range_of_a_l3787_378799


namespace NUMINAMATH_CALUDE_wire_ratio_l3787_378715

/-- Given a wire of length 21 cm cut into two pieces, where the shorter piece is 5.999999999999998 cm long,
    prove that the ratio of the shorter piece to the longer piece is 2:5. -/
theorem wire_ratio (total_length : ℝ) (shorter_length : ℝ) :
  total_length = 21 →
  shorter_length = 5.999999999999998 →
  let longer_length := total_length - shorter_length
  shorter_length / longer_length = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_wire_ratio_l3787_378715


namespace NUMINAMATH_CALUDE_arithmetic_progression_problem_l3787_378791

/-- An arithmetic progression with its sum sequence -/
structure ArithmeticProgression where
  a : ℕ → ℤ  -- The sequence
  S : ℕ → ℤ  -- The sum sequence
  is_arithmetic : ∀ n : ℕ, a (n + 2) - a (n + 1) = a (n + 1) - a n
  sum_formula : ∀ n : ℕ, S n = n * (a 1 + a n) / 2

/-- The main theorem -/
theorem arithmetic_progression_problem (seq : ArithmeticProgression) 
    (h1 : seq.a 1 + (seq.a 2)^2 = -3)
    (h2 : seq.S 5 = 10) :
  seq.a 9 = 20 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_problem_l3787_378791


namespace NUMINAMATH_CALUDE_probability_black_ball_l3787_378720

/-- The probability of drawing a black ball from a bag of colored balls. -/
theorem probability_black_ball (total : ℕ) (red : ℕ) (white : ℕ) (black : ℕ) :
  total = red + white + black →
  total = 6 →
  red = 1 →
  white = 2 →
  black = 3 →
  (black : ℚ) / total = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_probability_black_ball_l3787_378720


namespace NUMINAMATH_CALUDE_mono_increasing_and_even_shift_implies_l3787_378721

/-- A function that is monotonically increasing on [1,+∞) and f(x+1) is even -/
def MonoIncreasingAndEvenShift (f : ℝ → ℝ) : Prop :=
  (∀ x y, 1 ≤ x ∧ x ≤ y → f x ≤ f y) ∧
  (∀ x, f (x + 1) = f (-x + 1))

/-- Theorem: If f is monotonically increasing on [1,+∞) and f(x+1) is even, then f(-2) > f(2) -/
theorem mono_increasing_and_even_shift_implies (f : ℝ → ℝ) 
  (h : MonoIncreasingAndEvenShift f) : f (-2) > f 2 := by
  sorry

end NUMINAMATH_CALUDE_mono_increasing_and_even_shift_implies_l3787_378721


namespace NUMINAMATH_CALUDE_height_of_equilateral_triangle_l3787_378708

/-- An equilateral triangle with base 2 and an inscribed circle --/
structure EquilateralTriangleWithInscribedCircle where
  /-- The base of the triangle --/
  base : ℝ
  /-- The height of the triangle --/
  height : ℝ
  /-- The radius of the inscribed circle --/
  radius : ℝ
  /-- The base is 2 --/
  base_eq_two : base = 2
  /-- The radius is half the height --/
  radius_half_height : radius = height / 2

/-- The height of an equilateral triangle with base 2 and an inscribed circle is √3 --/
theorem height_of_equilateral_triangle
  (triangle : EquilateralTriangleWithInscribedCircle) :
  triangle.height = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_height_of_equilateral_triangle_l3787_378708


namespace NUMINAMATH_CALUDE_equation_solution_l3787_378739

theorem equation_solution (x : ℝ) : 
  (Real.cos (2 * x / 5) - Real.cos (2 * Real.pi / 15))^2 + 
  (Real.sin (2 * x / 3) - Real.sin (4 * Real.pi / 9))^2 = 0 ↔ 
  ∃ t : ℤ, x = 29 * Real.pi / 3 + 15 * Real.pi * (t : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3787_378739


namespace NUMINAMATH_CALUDE_smallest_number_proof_l3787_378729

theorem smallest_number_proof (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Three positive integers
  (a + b + c) / 3 = 30 →   -- Arithmetic mean is 30
  (a ≤ b ∧ b ≤ c) ∨ (a ≤ c ∧ c ≤ b) ∨ (b ≤ a ∧ a ≤ c) ∨ 
  (b ≤ c ∧ c ≤ a) ∨ (c ≤ a ∧ a ≤ b) ∨ (c ≤ b ∧ b ≤ a) →  -- Median condition
  b = 28 →                 -- Median is 28
  c = 34 →                 -- Largest number is 6 more than median
  a = 28                   -- Smallest number is 28
:= by sorry

end NUMINAMATH_CALUDE_smallest_number_proof_l3787_378729


namespace NUMINAMATH_CALUDE_incorrect_inequality_l3787_378794

theorem incorrect_inequality (a b : ℝ) (h : (1 / a) < (1 / b) ∧ (1 / b) < 0) : 
  ¬(a * b > b^2) := by
sorry

end NUMINAMATH_CALUDE_incorrect_inequality_l3787_378794


namespace NUMINAMATH_CALUDE_num_different_selections_eq_six_l3787_378712

/-- Represents the set of attractions -/
inductive Attraction : Type
  | A : Attraction
  | B : Attraction
  | C : Attraction

/-- Represents a selection of two attractions -/
def Selection := Finset Attraction

/-- The set of all possible selections -/
def all_selections : Finset Selection :=
  sorry

/-- Predicate to check if two selections are different -/
def different_selections (s1 s2 : Selection) : Prop :=
  s1 ≠ s2

/-- The number of ways two people can choose different selections -/
def num_different_selections : ℕ :=
  sorry

/-- Theorem: The number of ways two people can choose two out of three attractions,
    such that their choices are different, is equal to 6 -/
theorem num_different_selections_eq_six :
  num_different_selections = 6 :=
sorry

end NUMINAMATH_CALUDE_num_different_selections_eq_six_l3787_378712


namespace NUMINAMATH_CALUDE_inequality_proof_l3787_378740

theorem inequality_proof (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) 
  (hx : x = Real.sqrt (a^2 + b^2)) (hy : y = Real.sqrt (c^2 + d^2)) :
  x * y ≥ a * c + b * d := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3787_378740


namespace NUMINAMATH_CALUDE_arithmetic_sequence_remainder_l3787_378764

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

def sequence_sum (a₁ : ℤ) (aₙ : ℤ) (n : ℕ) : ℤ := n * (a₁ + aₙ) / 2

theorem arithmetic_sequence_remainder (a₁ : ℤ) (aₙ : ℤ) (d : ℤ) (n : ℕ) :
  a₁ = 3 ∧ aₙ = 347 ∧ d = 8 →
  (sequence_sum a₁ aₙ n) % 8 = 4 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_remainder_l3787_378764


namespace NUMINAMATH_CALUDE_u_converges_to_zero_l3787_378775

open Real

variable (f : ℝ → ℝ)
variable (u : ℕ → ℝ)

-- f is non-decreasing
axiom f_nondecreasing : ∀ x y, x ≤ y → f x ≤ f y

-- f(y) - f(x) < y - x for all real numbers x and y > x
axiom f_contractive : ∀ x y, x < y → f y - f x < y - x

-- Recurrence relation for u
axiom u_recurrence : ∀ n : ℕ, u (n + 2) = f (u (n + 1)) - f (u n)

-- Theorem to prove
theorem u_converges_to_zero : 
  (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |u n| < ε) :=
sorry

end NUMINAMATH_CALUDE_u_converges_to_zero_l3787_378775


namespace NUMINAMATH_CALUDE_cookie_division_l3787_378751

/-- The area of a cookie piece when a cookie with total area 81.12 cm² is divided equally among 6 friends -/
theorem cookie_division (total_area : ℝ) (num_friends : ℕ) 
  (h1 : total_area = 81.12)
  (h2 : num_friends = 6) :
  total_area / num_friends = 13.52 := by
  sorry

end NUMINAMATH_CALUDE_cookie_division_l3787_378751


namespace NUMINAMATH_CALUDE_number_problem_l3787_378724

theorem number_problem (x : ℤ) (h : x + 1015 = 3016) : x = 2001 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l3787_378724


namespace NUMINAMATH_CALUDE_fourth_root_of_y_squared_times_sqrt_y_l3787_378702

theorem fourth_root_of_y_squared_times_sqrt_y (y : ℝ) (h : y > 0) :
  (y^2 * y^(1/2))^(1/4) = y^(5/8) := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_of_y_squared_times_sqrt_y_l3787_378702


namespace NUMINAMATH_CALUDE_evaluate_expression_l3787_378744

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3787_378744


namespace NUMINAMATH_CALUDE_abc_sum_root_l3787_378713

theorem abc_sum_root (a b c : ℝ) 
  (h1 : b + c = 7) 
  (h2 : c + a = 8) 
  (h3 : a + b = 9) : 
  Real.sqrt (a * b * c * (a + b + c)) = 12 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_abc_sum_root_l3787_378713


namespace NUMINAMATH_CALUDE_tile_ratio_l3787_378742

/-- Given a square pattern with black and white tiles and a white border added, 
    calculate the ratio of black to white tiles. -/
theorem tile_ratio (initial_black initial_white border_width : ℕ) 
  (h1 : initial_black = 5)
  (h2 : initial_white = 20)
  (h3 : border_width = 1)
  (h4 : initial_black + initial_white = (initial_black + initial_white).sqrt ^ 2) :
  let total_side := (initial_black + initial_white).sqrt + 2 * border_width
  let total_tiles := total_side ^ 2
  let added_white := total_tiles - (initial_black + initial_white)
  let final_white := initial_white + added_white
  (initial_black : ℚ) / final_white = 5 / 44 := by sorry

end NUMINAMATH_CALUDE_tile_ratio_l3787_378742


namespace NUMINAMATH_CALUDE_percentage_correct_second_question_l3787_378785

/-- Given a class of students taking a test with two questions, this theorem proves
    the percentage of students who answered the second question correctly. -/
theorem percentage_correct_second_question
  (total : ℝ) -- Total number of students
  (first_correct : ℝ) -- Number of students who answered the first question correctly
  (both_correct : ℝ) -- Number of students who answered both questions correctly
  (neither_correct : ℝ) -- Number of students who answered neither question correctly
  (h1 : first_correct = 0.75 * total) -- 75% answered the first question correctly
  (h2 : both_correct = 0.25 * total) -- 25% answered both questions correctly
  (h3 : neither_correct = 0.2 * total) -- 20% answered neither question correctly
  : (total - neither_correct - (first_correct - both_correct)) / total = 0.3 := by
  sorry


end NUMINAMATH_CALUDE_percentage_correct_second_question_l3787_378785


namespace NUMINAMATH_CALUDE_movie_and_popcorn_expense_l3787_378727

/-- The fraction of allowance spent on movie ticket and popcorn -/
theorem movie_and_popcorn_expense (B : ℝ) (m p : ℝ) 
  (hm : m = (1/4) * (B - p)) 
  (hp : p = (1/10) * (B - m)) : 
  (m + p) / B = 4/13 := by
  sorry

end NUMINAMATH_CALUDE_movie_and_popcorn_expense_l3787_378727


namespace NUMINAMATH_CALUDE_first_half_speed_l3787_378792

theorem first_half_speed (total_distance : ℝ) (first_half_distance : ℝ) (second_half_speed : ℝ) (average_speed : ℝ) :
  total_distance = 20 →
  first_half_distance = 10 →
  second_half_speed = 10 →
  average_speed = 10.909090909090908 →
  (total_distance / (first_half_distance / (total_distance / average_speed - first_half_distance / second_half_speed) + first_half_distance / second_half_speed)) = 12 :=
by sorry

end NUMINAMATH_CALUDE_first_half_speed_l3787_378792


namespace NUMINAMATH_CALUDE_three_distinct_real_roots_l3787_378746

/-- The cubic function f(x) = x^3 - 3x - a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*x - a

/-- Theorem stating the condition for three distinct real roots -/
theorem three_distinct_real_roots (a : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f a x = 0 ∧ f a y = 0 ∧ f a z = 0) ↔ 
  -2 < a ∧ a < 2 :=
sorry

end NUMINAMATH_CALUDE_three_distinct_real_roots_l3787_378746


namespace NUMINAMATH_CALUDE_wolf_still_hungry_l3787_378714

/-- Represents the food quantity provided by a hare -/
def hare_food : ℝ := sorry

/-- Represents the food quantity provided by a pig -/
def pig_food : ℝ := sorry

/-- Represents the food quantity needed to satisfy the wolf's hunger -/
def wolf_satiety : ℝ := sorry

/-- The wolf is still hungry after eating 3 pigs and 7 hares -/
axiom hunger_condition : 3 * pig_food + 7 * hare_food < wolf_satiety

/-- The wolf has overeaten after consuming 7 pigs and 1 hare -/
axiom overeating_condition : 7 * pig_food + hare_food > wolf_satiety

/-- Theorem: The wolf will still be hungry after eating 11 hares -/
theorem wolf_still_hungry : 11 * hare_food < wolf_satiety := by
  sorry

end NUMINAMATH_CALUDE_wolf_still_hungry_l3787_378714


namespace NUMINAMATH_CALUDE_number_fraction_relation_l3787_378710

theorem number_fraction_relation (x : ℝ) (h : (2 / 5) * x = 20) : (1 / 3) * x = 50 / 3 := by
  sorry

end NUMINAMATH_CALUDE_number_fraction_relation_l3787_378710


namespace NUMINAMATH_CALUDE_pounds_per_pillow_is_two_l3787_378797

-- Define the constants from the problem
def feathers_per_pound : ℕ := 300
def total_feathers : ℕ := 3600
def number_of_pillows : ℕ := 6

-- Define the function to calculate pounds of feathers needed per pillow
def pounds_per_pillow : ℚ :=
  (total_feathers / feathers_per_pound) / number_of_pillows

-- Theorem to prove
theorem pounds_per_pillow_is_two : pounds_per_pillow = 2 := by
  sorry


end NUMINAMATH_CALUDE_pounds_per_pillow_is_two_l3787_378797


namespace NUMINAMATH_CALUDE_machine_b_time_for_150_copies_l3787_378703

/-- Given two machines A and B with the following properties:
    1. Machine A makes 100 copies in 20 minutes
    2. Machines A and B working simultaneously for 30 minutes produce 600 copies
    This theorem proves that it takes 10 minutes for Machine B to make 150 copies -/
theorem machine_b_time_for_150_copies 
  (rate_a : ℚ) -- rate of machine A in copies per minute
  (rate_b : ℚ) -- rate of machine B in copies per minute
  (h1 : rate_a = 100 / 20) -- condition 1
  (h2 : 30 * (rate_a + rate_b) = 600) -- condition 2
  : 150 / rate_b = 10 := by sorry

end NUMINAMATH_CALUDE_machine_b_time_for_150_copies_l3787_378703


namespace NUMINAMATH_CALUDE_money_division_theorem_l3787_378793

theorem money_division_theorem (p q r : ℕ) (total : ℚ) :
  p + q + r = 22 →  -- Ratio sum: 3 + 7 + 12 = 22
  (7 * total / 22 - 3 * total / 22 = 2800) →  -- Difference between q and p's shares
  (12 * total / 22 - 7 * total / 22 = 3500) :=  -- Difference between r and q's shares
by sorry

end NUMINAMATH_CALUDE_money_division_theorem_l3787_378793


namespace NUMINAMATH_CALUDE_ellipse_condition_l3787_378759

/-- Represents an ellipse equation with parameter m -/
def is_ellipse (m : ℝ) : Prop :=
  (5 - m > 0) ∧ (m + 3 > 0) ∧ (5 - m ≠ m + 3)

/-- The condition -3 < m < 5 -/
def condition (m : ℝ) : Prop :=
  -3 < m ∧ m < 5

theorem ellipse_condition (m : ℝ) :
  (is_ellipse m → condition m) ∧ 
  ¬(condition m → is_ellipse m) :=
sorry

end NUMINAMATH_CALUDE_ellipse_condition_l3787_378759


namespace NUMINAMATH_CALUDE_maisy_new_job_earnings_l3787_378768

/-- Represents Maisy's job options and calculates the difference in earnings -/
def earnings_difference (current_hours : ℕ) (current_wage : ℕ) (new_hours : ℕ) (new_wage : ℕ) (bonus : ℕ) : ℕ :=
  let current_earnings := current_hours * current_wage
  let new_earnings := new_hours * new_wage + bonus
  new_earnings - current_earnings

/-- Proves that Maisy will earn $15 more at her new job -/
theorem maisy_new_job_earnings :
  earnings_difference 8 10 4 15 35 = 15 := by
  sorry

end NUMINAMATH_CALUDE_maisy_new_job_earnings_l3787_378768


namespace NUMINAMATH_CALUDE_fractional_equation_root_l3787_378733

theorem fractional_equation_root (k : ℚ) : 
  (∃ x : ℚ, x ≠ 1 ∧ (2 * k) / (x - 1) - 3 / (1 - x) = 1) → k = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_root_l3787_378733


namespace NUMINAMATH_CALUDE_factor_expression_l3787_378788

theorem factor_expression (x : ℝ) : 63 * x + 45 = 9 * (7 * x + 5) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l3787_378788


namespace NUMINAMATH_CALUDE_final_ratio_theorem_l3787_378782

/-- Represents the final amount ratio between two players -/
structure FinalRatio where
  player1 : ℕ
  player2 : ℕ

/-- Represents a game with three players -/
structure Game where
  initialAmount : ℕ
  finalRatioAS : FinalRatio
  sGain : ℕ

theorem final_ratio_theorem (g : Game) 
  (h1 : g.initialAmount = 70)
  (h2 : g.finalRatioAS = FinalRatio.mk 1 2)
  (h3 : g.sGain = 50) :
  ∃ (finalRatioSB : FinalRatio), 
    finalRatioSB.player1 = 4 ∧ 
    finalRatioSB.player2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_final_ratio_theorem_l3787_378782


namespace NUMINAMATH_CALUDE_bankers_discount_example_l3787_378783

/-- Given a true discount and a sum due, calculate the banker's discount -/
def bankers_discount (true_discount : ℚ) (sum_due : ℚ) : ℚ :=
  (true_discount * sum_due) / (sum_due - true_discount)

/-- Theorem: The banker's discount is 78 given a true discount of 66 and a sum due of 429 -/
theorem bankers_discount_example : bankers_discount 66 429 = 78 := by
  sorry

end NUMINAMATH_CALUDE_bankers_discount_example_l3787_378783


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l3787_378798

theorem product_of_three_numbers (a b c : ℕ) : 
  a * b * c = 224 →
  a < b →
  b < c →
  2 * a = c →
  ∃ (x y z : ℕ), x * y * z = 224 ∧ 2 * x = z ∧ x < y ∧ y < z :=
by sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l3787_378798


namespace NUMINAMATH_CALUDE_price_ratio_theorem_l3787_378777

theorem price_ratio_theorem (cost_price : ℝ) (first_price second_price : ℝ) :
  first_price = cost_price * (1 + 1.4) ∧
  second_price = cost_price * (1 - 0.2) →
  second_price / first_price = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_price_ratio_theorem_l3787_378777


namespace NUMINAMATH_CALUDE_sequence_fourth_term_l3787_378730

theorem sequence_fourth_term 
  (a : ℕ → ℚ) 
  (S : ℕ → ℚ) 
  (h : ∀ n, S n = (n + 1 : ℚ) / (n + 2 : ℚ)) : 
  a 4 = 1 / 30 := by
  sorry

end NUMINAMATH_CALUDE_sequence_fourth_term_l3787_378730


namespace NUMINAMATH_CALUDE_chocolate_distribution_l3787_378701

def is_valid_distribution (n m : ℕ) : Prop :=
  n ≤ m ∨ (m < n ∧ m ∣ (n - m))

def possible_n_for_m_9 : Set ℕ :=
  {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 18}

theorem chocolate_distribution (n m : ℕ) :
  (m = 9 → n ∈ possible_n_for_m_9) ↔ is_valid_distribution n m :=
sorry

end NUMINAMATH_CALUDE_chocolate_distribution_l3787_378701


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3787_378787

theorem sufficient_not_necessary (a b : ℝ) :
  (((a - b) * a^2 < 0 → a < b) ∧
  (∃ a b : ℝ, a < b ∧ (a - b) * a^2 ≥ 0)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3787_378787


namespace NUMINAMATH_CALUDE_min_value_shifted_sine_l3787_378722

theorem min_value_shifted_sine (φ : ℝ) (h_φ : |φ| < π/2) :
  let f : ℝ → ℝ := λ x ↦ Real.sin (2*x - π/3)
  ∃ x₀ ∈ Set.Icc 0 (π/2), ∀ x ∈ Set.Icc 0 (π/2), f x₀ ≤ f x ∧ f x₀ = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_shifted_sine_l3787_378722


namespace NUMINAMATH_CALUDE_water_cup_pricing_equation_l3787_378781

/-- Represents the pricing of a Huashan brand water cup -/
def water_cup_pricing (x : ℝ) : Prop :=
  let first_discount := x - 5
  let second_discount := 0.8 * first_discount
  second_discount = 60

/-- The equation representing the water cup pricing after discounts -/
theorem water_cup_pricing_equation (x : ℝ) :
  water_cup_pricing x ↔ 0.8 * (x - 5) = 60 := by sorry

end NUMINAMATH_CALUDE_water_cup_pricing_equation_l3787_378781


namespace NUMINAMATH_CALUDE_problem_statement_l3787_378707

theorem problem_statement (a b x y : ℝ) 
  (h1 : a*x + b*y = 5)
  (h2 : a*x^2 + b*y^2 = 9)
  (h3 : a*x^3 + b*y^3 = 20)
  (h4 : a*x^4 + b*y^4 = 48)
  (h5 : x + y = -15)
  (h6 : x^2 + y^2 = 55) :
  a*x^5 + b*y^5 = -1065 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3787_378707


namespace NUMINAMATH_CALUDE_padma_valuable_cards_l3787_378789

theorem padma_valuable_cards (padma_initial : ℕ) (robert_initial : ℕ) (total_traded : ℕ) 
  (padma_received : ℕ) (robert_received : ℕ) (robert_traded : ℕ) 
  (h1 : padma_initial = 75)
  (h2 : robert_initial = 88)
  (h3 : total_traded = 35)
  (h4 : padma_received = 10)
  (h5 : robert_received = 15)
  (h6 : robert_traded = 8) :
  ∃ (padma_valuable : ℕ), 
    padma_valuable + robert_received = total_traded ∧ 
    padma_valuable = 20 :=
by sorry

end NUMINAMATH_CALUDE_padma_valuable_cards_l3787_378789


namespace NUMINAMATH_CALUDE_bells_toll_together_once_l3787_378776

def bell_intervals : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23]

def lcm_list (L : List ℕ) : ℕ :=
  L.foldl Nat.lcm 1

theorem bells_toll_together_once (intervals : List ℕ) (duration : ℕ) : 
  intervals = bell_intervals → duration = 60 * 60 → 
  (duration / (lcm_list intervals) + 1 : ℕ) = 1 :=
by sorry

end NUMINAMATH_CALUDE_bells_toll_together_once_l3787_378776


namespace NUMINAMATH_CALUDE_triangle_angle_not_all_greater_than_60_l3787_378771

theorem triangle_angle_not_all_greater_than_60 :
  ¬ ∃ (a b c : ℝ), 
    (0 < a ∧ 0 < b ∧ 0 < c) ∧  -- angles are positive
    (a + b + c = 180) ∧        -- sum of angles in a triangle is 180°
    (60 < a ∧ 60 < b ∧ 60 < c) -- all angles greater than 60°
  := by sorry

end NUMINAMATH_CALUDE_triangle_angle_not_all_greater_than_60_l3787_378771


namespace NUMINAMATH_CALUDE_greatest_integer_for_all_real_domain_l3787_378734

theorem greatest_integer_for_all_real_domain : 
  ∃ (b : ℤ), (∀ (x : ℝ), x^2 + b*x + 15 ≠ 0) ∧ 
  (∀ (c : ℤ), (∀ (x : ℝ), x^2 + c*x + 15 ≠ 0) → c ≤ b) ∧ 
  b = 7 := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_for_all_real_domain_l3787_378734


namespace NUMINAMATH_CALUDE_sqrt_difference_less_than_sqrt_of_difference_l3787_378796

theorem sqrt_difference_less_than_sqrt_of_difference 
  (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  Real.sqrt a - Real.sqrt b < Real.sqrt (a - b) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_less_than_sqrt_of_difference_l3787_378796


namespace NUMINAMATH_CALUDE_factorization_equality_l3787_378726

theorem factorization_equality (a b : ℝ) : 
  a^2 - b^2 + 4*a + 2*b + 3 = (a + b + 1)*(a - b + 3) := by
sorry

end NUMINAMATH_CALUDE_factorization_equality_l3787_378726


namespace NUMINAMATH_CALUDE_haley_marbles_l3787_378758

/-- The number of boys who love to play marbles -/
def num_marble_boys : ℕ := 13

/-- The number of marbles each boy receives -/
def marbles_per_boy : ℕ := 2

/-- The total number of marbles Haley has -/
def total_marbles : ℕ := num_marble_boys * marbles_per_boy

theorem haley_marbles : total_marbles = 26 := by
  sorry

end NUMINAMATH_CALUDE_haley_marbles_l3787_378758


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l3787_378767

theorem necessary_but_not_sufficient_condition (a : ℝ) : 
  (∀ x, -1 ≤ x ∧ x < 2 → x ≤ a) ∧ 
  (∃ x, x ≤ a ∧ (x < -1 ∨ x ≥ 2)) →
  a ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l3787_378767


namespace NUMINAMATH_CALUDE_system_solution_l3787_378711

theorem system_solution (a b c x y z : ℝ) 
  (eq1 : x / (a * b) + y / (b * c) + z / (a * c) = 3)
  (eq2 : x / a + y / b + z / c = a + b + c)
  (eq3 : c^2 * x + a^2 * y + b^2 * z = a * b * c * (a + b + c)) :
  x = a * b ∧ y = b * c ∧ z = a * c := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3787_378711


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_9_l3787_378743

theorem arithmetic_square_root_of_9 : Real.sqrt 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_9_l3787_378743


namespace NUMINAMATH_CALUDE_cubic_expression_property_l3787_378717

theorem cubic_expression_property (a b : ℝ) :
  a * (3^3) + b * 3 - 5 = 20 → a * ((-3)^3) + b * (-3) - 5 = -30 := by
  sorry

end NUMINAMATH_CALUDE_cubic_expression_property_l3787_378717


namespace NUMINAMATH_CALUDE_oocyte_characteristics_l3787_378747

/-- Represents the hair length trait in rabbits -/
inductive HairTrait
  | Long
  | Short

/-- Represents the phase of meiosis -/
inductive MeioticPhase
  | First
  | Second

/-- Represents a heterozygous rabbit with long hair trait dominant over short hair trait -/
structure HeterozygousRabbit where
  dominantTrait : HairTrait
  recessiveTrait : HairTrait
  totalGenes : ℕ
  genesPerOocyte : ℕ
  nucleotideTypes : ℕ
  allelesSeperationPhase : MeioticPhase

/-- Main theorem about the characteristics of oocytes in a heterozygous rabbit -/
theorem oocyte_characteristics (rabbit : HeterozygousRabbit)
  (h1 : rabbit.dominantTrait = HairTrait.Long)
  (h2 : rabbit.recessiveTrait = HairTrait.Short)
  (h3 : rabbit.totalGenes = 20)
  (h4 : rabbit.genesPerOocyte = 4)
  (h5 : rabbit.nucleotideTypes = 4)
  (h6 : rabbit.allelesSeperationPhase = MeioticPhase.First) :
  let maxShortHairOocytes := rabbit.totalGenes / rabbit.genesPerOocyte / 2
  maxShortHairOocytes = 5 ∧
  rabbit.nucleotideTypes = 4 ∧
  rabbit.allelesSeperationPhase = MeioticPhase.First :=
by sorry

end NUMINAMATH_CALUDE_oocyte_characteristics_l3787_378747


namespace NUMINAMATH_CALUDE_parabola_directrix_l3787_378786

/-- The equation of the directrix of the parabola y = 4x^2 -/
theorem parabola_directrix (x y : ℝ) :
  (y = 4 * x^2) →  -- Given parabola equation
  ∃ (d : ℝ), d = -1/16 ∧ (∀ (x₀ y₀ : ℝ), y₀ = 4 * x₀^2 → y₀ ≥ d) ∧
              (∀ ε > 0, ∃ (x₁ y₁ : ℝ), y₁ = 4 * x₁^2 ∧ y₁ < d + ε) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l3787_378786


namespace NUMINAMATH_CALUDE_divisors_of_36_l3787_378745

theorem divisors_of_36 : 
  ∃ (divs : List Nat), 
    (∀ d, d ∈ divs ↔ d ∣ 36) ∧ 
    divs.length = 9 ∧
    divs = [1, 2, 3, 4, 6, 9, 12, 18, 36] :=
by sorry

end NUMINAMATH_CALUDE_divisors_of_36_l3787_378745


namespace NUMINAMATH_CALUDE_remainder_twelve_remainder_107_is_least_unique_divisor_l3787_378756

def least_number : ℕ := 540

-- 540 leaves a remainder of 5 when divided by 12
theorem remainder_twelve : least_number % 12 = 5 := by sorry

-- 107 leaves a remainder of 5 when 540 is divided by it
theorem remainder_107 : least_number % 107 = 5 := by sorry

-- 540 is the least number that leaves a remainder of 5 when divided by some numbers
theorem is_least (n : ℕ) : n < least_number → ¬(∃ m : ℕ, m > 1 ∧ n % m = 5) := by sorry

-- 107 is the only number (other than 12) that leaves a remainder of 5 when 540 is divided by it
theorem unique_divisor (n : ℕ) : n ≠ 12 → n ≠ 107 → least_number % n ≠ 5 := by sorry

end NUMINAMATH_CALUDE_remainder_twelve_remainder_107_is_least_unique_divisor_l3787_378756


namespace NUMINAMATH_CALUDE_sharmila_average_earnings_l3787_378754

/-- Represents Sharmila's work schedule and earnings --/
structure WorkSchedule where
  job1_long_days : Nat -- Number of 10-hour days in job 1
  job1_short_days : Nat -- Number of 8-hour days in job 1
  job1_hourly_rate : ℚ -- Hourly rate for job 1
  job1_long_day_bonus : ℚ -- Bonus for 10-hour days in job 1
  job2_hours : Nat -- Hours worked in job 2
  job2_hourly_rate : ℚ -- Hourly rate for job 2
  job2_bonus : ℚ -- Bonus for job 2

/-- Calculates the average hourly earnings --/
def average_hourly_earnings (schedule : WorkSchedule) : ℚ :=
  let job1_hours := schedule.job1_long_days * 10 + schedule.job1_short_days * 8
  let job1_earnings := job1_hours * schedule.job1_hourly_rate + schedule.job1_long_days * schedule.job1_long_day_bonus
  let job2_earnings := schedule.job2_hours * schedule.job2_hourly_rate + schedule.job2_bonus
  let total_earnings := job1_earnings + job2_earnings
  let total_hours := job1_hours + schedule.job2_hours
  total_earnings / total_hours

/-- Sharmila's work schedule --/
def sharmila_schedule : WorkSchedule := {
  job1_long_days := 3
  job1_short_days := 2
  job1_hourly_rate := 15
  job1_long_day_bonus := 20
  job2_hours := 5
  job2_hourly_rate := 12
  job2_bonus := 10
}

/-- Theorem stating Sharmila's average hourly earnings --/
theorem sharmila_average_earnings :
  average_hourly_earnings sharmila_schedule = 16.08 := by
  sorry


end NUMINAMATH_CALUDE_sharmila_average_earnings_l3787_378754


namespace NUMINAMATH_CALUDE_ab_product_l3787_378778

theorem ab_product (a b : ℚ) (h1 : 10 * a = 20) (h2 : 6 * b = 20) : 120 * a * b = 800 := by
  sorry

end NUMINAMATH_CALUDE_ab_product_l3787_378778


namespace NUMINAMATH_CALUDE_theorem_60752_infinite_primes_4k_plus_1_l3787_378790

-- Theorem from problem 60752
theorem theorem_60752 (N : ℕ) (a : ℕ) (h : N = a^2 + 1) :
  ∃ p : ℕ, Prime p ∧ p ∣ N ∧ ∃ k : ℕ, p = 4 * k + 1 := sorry

theorem infinite_primes_4k_plus_1 :
  ∀ n : ℕ, ∃ p : ℕ, p > n ∧ Prime p ∧ ∃ k : ℕ, p = 4 * k + 1 := by sorry

end NUMINAMATH_CALUDE_theorem_60752_infinite_primes_4k_plus_1_l3787_378790


namespace NUMINAMATH_CALUDE_xiaoning_score_is_87_l3787_378770

/-- The maximum score for a student's semester physical education comprehensive score. -/
def max_score : ℝ := 100

/-- The weight of the midterm exam score in the comprehensive score calculation. -/
def midterm_weight : ℝ := 0.3

/-- The weight of the final exam score in the comprehensive score calculation. -/
def final_weight : ℝ := 0.7

/-- Xiaoning's midterm exam score as a percentage. -/
def xiaoning_midterm : ℝ := 80

/-- Xiaoning's final exam score as a percentage. -/
def xiaoning_final : ℝ := 90

/-- Calculates the comprehensive score based on midterm and final exam scores and their weights. -/
def comprehensive_score (midterm : ℝ) (final : ℝ) : ℝ :=
  midterm * midterm_weight + final * final_weight

/-- Theorem stating that Xiaoning's physical education comprehensive score is 87 points. -/
theorem xiaoning_score_is_87 :
  comprehensive_score xiaoning_midterm xiaoning_final = 87 := by
  sorry

end NUMINAMATH_CALUDE_xiaoning_score_is_87_l3787_378770


namespace NUMINAMATH_CALUDE_arithmetic_sequence_seventh_term_l3787_378779

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_seventh_term
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_sum1 : a 1 + a 2 = 4)
  (h_sum2 : a 2 + a 3 = 8) :
  a 7 = 13 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_seventh_term_l3787_378779


namespace NUMINAMATH_CALUDE_cookies_in_class_l3787_378709

/-- The number of cookies brought by Mona, Jasmine, Rachel, and Carlos -/
def totalCookies (mona jasmine rachel carlos : ℕ) : ℕ :=
  mona + jasmine + rachel + carlos

/-- Theorem stating the total number of cookies brought to class -/
theorem cookies_in_class :
  ∀ (mona jasmine rachel carlos : ℕ),
  mona = 20 →
  jasmine = mona - 5 →
  rachel = jasmine + 10 →
  carlos = rachel * 2 →
  totalCookies mona jasmine rachel carlos = 110 := by
  sorry

end NUMINAMATH_CALUDE_cookies_in_class_l3787_378709
