import Mathlib

namespace NUMINAMATH_CALUDE_digit_difference_in_base_d_l3989_398948

/-- Given two digits X and Y in base d > 8, if XY + XX = 182 in base d, then X - Y = d - 8 in base d -/
theorem digit_difference_in_base_d (d : ℕ) (X Y : Fin d) (h_d : d > 8) 
  (h_sum : d * X.val + Y.val + d * X.val + X.val = d^2 + 8*d + 2) : 
  X.val - Y.val = d - 8 := by
  sorry

end NUMINAMATH_CALUDE_digit_difference_in_base_d_l3989_398948


namespace NUMINAMATH_CALUDE_tom_program_duration_l3989_398982

/-- Represents the duration of a combined BS and Ph.D. program -/
structure ProgramDuration where
  bs : ℕ
  phd : ℕ

/-- Calculates the time taken to complete a program given the standard duration and a completion factor -/
def completionTime (d : ProgramDuration) (factor : ℚ) : ℚ :=
  factor * (d.bs + d.phd)

theorem tom_program_duration :
  let standard_duration : ProgramDuration := { bs := 3, phd := 5 }
  let completion_factor : ℚ := 3/4
  completionTime standard_duration completion_factor = 6 := by sorry

end NUMINAMATH_CALUDE_tom_program_duration_l3989_398982


namespace NUMINAMATH_CALUDE_remainder_1234567_div_123_l3989_398938

theorem remainder_1234567_div_123 : 1234567 % 123 = 129 := by
  sorry

end NUMINAMATH_CALUDE_remainder_1234567_div_123_l3989_398938


namespace NUMINAMATH_CALUDE_expansion_coefficient_zero_l3989_398988

-- Define the binomial coefficient function
def binomial (n k : ℕ) : ℕ := sorry

-- Define the coefficient function for the expansion of (1 - 1/x)(1+x)^5
def coefficient (r : ℤ) : ℚ :=
  if r = 2 then binomial 5 2 - binomial 5 3
  else if r = 1 then binomial 5 1 - binomial 5 2
  else if r = 0 then 1 - binomial 5 1
  else if r = -1 then -1
  else if r = 3 then binomial 5 3 - binomial 5 4
  else if r = 4 then binomial 5 4 - binomial 5 5
  else if r = 5 then binomial 5 5
  else 0

theorem expansion_coefficient_zero :
  ∃ (r : ℤ), r ∈ Set.Icc (-1 : ℤ) 5 ∧ coefficient r = 0 ∧ r = 2 :=
by sorry

end NUMINAMATH_CALUDE_expansion_coefficient_zero_l3989_398988


namespace NUMINAMATH_CALUDE_negative_two_cubed_equality_l3989_398976

theorem negative_two_cubed_equality : (-2)^3 = -2^3 := by
  sorry

end NUMINAMATH_CALUDE_negative_two_cubed_equality_l3989_398976


namespace NUMINAMATH_CALUDE_vector_difference_magnitude_l3989_398992

/-- Given vectors a and b in ℝ², prove that the magnitude of their difference is 5. -/
theorem vector_difference_magnitude (a b : ℝ × ℝ) : 
  a = (2, 1) → b = (-2, 4) → ‖a - b‖ = 5 := by sorry

end NUMINAMATH_CALUDE_vector_difference_magnitude_l3989_398992


namespace NUMINAMATH_CALUDE_bons_winning_probability_l3989_398984

/-- The probability of rolling a six -/
def prob_six : ℚ := 1/6

/-- The probability of not rolling a six -/
def prob_not_six : ℚ := 5/6

/-- The probability that B. Bons wins the game -/
def prob_bons_wins : ℚ := 5/11

/-- Theorem stating that the probability of B. Bons winning is 5/11 -/
theorem bons_winning_probability : 
  prob_bons_wins = prob_not_six * prob_six + prob_not_six * prob_not_six * prob_bons_wins :=
by sorry

end NUMINAMATH_CALUDE_bons_winning_probability_l3989_398984


namespace NUMINAMATH_CALUDE_triangle_properties_l3989_398911

/-- Triangle ABC with given side lengths and angle -/
structure Triangle where
  c : ℝ
  b : ℝ
  B : ℝ

/-- The possible values for angle C in the triangle -/
def possible_C (t : Triangle) : Set ℝ :=
  {60, 120}

/-- The possible areas of the triangle -/
def possible_areas (t : Triangle) : Set ℝ :=
  {Real.sqrt 3 / 2, Real.sqrt 3 / 4}

/-- Theorem stating the properties of the triangle -/
theorem triangle_properties (t : Triangle) 
  (h_c : t.c = Real.sqrt 3)
  (h_b : t.b = 1)
  (h_B : t.B = 30) :
  (∃ (C : ℝ), C ∈ possible_C t) ∧
  (∃ (area : ℝ), area ∈ possible_areas t) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3989_398911


namespace NUMINAMATH_CALUDE_correct_expression_proof_l3989_398971

theorem correct_expression_proof (x a b : ℝ) : 
  ((2*x - a) * (3*x + b) = 6*x^2 - 13*x + 6) →
  ((2*x + a) * (x + b) = 2*x^2 - x - 6) →
  (a = 3 ∧ b = -2 ∧ (2*x + a) * (3*x + b) = 6*x^2 + 5*x - 6) := by
  sorry

end NUMINAMATH_CALUDE_correct_expression_proof_l3989_398971


namespace NUMINAMATH_CALUDE_largest_divisor_of_15_less_than_15_l3989_398909

theorem largest_divisor_of_15_less_than_15 :
  ∃ n : ℕ, n ∣ 15 ∧ n ≠ 15 ∧ ∀ m : ℕ, m ∣ 15 ∧ m ≠ 15 → m ≤ n :=
by
  sorry

end NUMINAMATH_CALUDE_largest_divisor_of_15_less_than_15_l3989_398909


namespace NUMINAMATH_CALUDE_total_fencing_cost_l3989_398903

/-- Calculates the total fencing cost for an irregular shaped plot -/
theorem total_fencing_cost (square_area : ℝ) (rect_length rect_height : ℝ) (triangle_side : ℝ)
  (square_cost rect_cost triangle_cost : ℝ) (gate_cost : ℝ)
  (h_square_area : square_area = 289)
  (h_rect_length : rect_length = 45)
  (h_rect_height : rect_height = 15)
  (h_triangle_side : triangle_side = 20)
  (h_square_cost : square_cost = 55)
  (h_rect_cost : rect_cost = 65)
  (h_triangle_cost : triangle_cost = 70)
  (h_gate_cost : gate_cost = 750) :
  4 * Real.sqrt square_area * square_cost +
  (2 * rect_height + rect_length) * rect_cost +
  3 * triangle_side * triangle_cost +
  gate_cost = 13565 := by
  sorry


end NUMINAMATH_CALUDE_total_fencing_cost_l3989_398903


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3989_398979

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (a > 1 ∧ b > 1 → (a - 1) * (b - 1) > 0) ∧
  ¬(∀ a b : ℝ, (a - 1) * (b - 1) > 0 → a > 1 ∧ b > 1) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3989_398979


namespace NUMINAMATH_CALUDE_unique_charming_number_l3989_398910

theorem unique_charming_number :
  ∃! (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 10 * a + b = 2 * a + b^3 := by
  sorry

end NUMINAMATH_CALUDE_unique_charming_number_l3989_398910


namespace NUMINAMATH_CALUDE_simplified_expression_sum_l3989_398941

theorem simplified_expression_sum (d : ℝ) (a b c : ℤ) : 
  d ≠ 0 → 
  (15 * d + 16 + 17 * d^2) + (3 * d + 2) = a * d + b + c * d^2 → 
  a + b + c = 53 := by
sorry

end NUMINAMATH_CALUDE_simplified_expression_sum_l3989_398941


namespace NUMINAMATH_CALUDE_number_of_schools_l3989_398945

theorem number_of_schools (n : ℕ) : n = 22 :=
  -- Define the total number of students
  let total_students := 4 * n
  -- Define Alex's rank
  let alex_rank := 2 * n
  -- Define the ranks of Alex's teammates
  let jordan_rank := 45
  let kim_rank := 73
  let lee_rank := 98
  -- State the conditions
  have h1 : alex_rank < jordan_rank := by sorry
  have h2 : alex_rank < kim_rank := by sorry
  have h3 : alex_rank < lee_rank := by sorry
  have h4 : total_students = 2 * alex_rank - 1 := by sorry
  have h5 : alex_rank ≤ 49 := by sorry
  -- Prove that n = 22
  sorry

#check number_of_schools

end NUMINAMATH_CALUDE_number_of_schools_l3989_398945


namespace NUMINAMATH_CALUDE_new_average_weight_l3989_398956

def original_team_size : ℕ := 7
def original_average_weight : ℚ := 76
def new_player1_weight : ℚ := 110
def new_player2_weight : ℚ := 60

theorem new_average_weight :
  let original_total_weight := original_team_size * original_average_weight
  let new_total_weight := original_total_weight + new_player1_weight + new_player2_weight
  let new_team_size := original_team_size + 2
  new_total_weight / new_team_size = 78 := by
sorry

end NUMINAMATH_CALUDE_new_average_weight_l3989_398956


namespace NUMINAMATH_CALUDE_apple_sales_theorem_l3989_398907

/-- Represents the sales of apples over three days in a store. -/
structure AppleSales where
  day1 : ℝ  -- Sales on day 1 in kg
  day2 : ℝ  -- Sales on day 2 in kg
  day3 : ℝ  -- Sales on day 3 in kg

/-- The conditions of the apple sales problem. -/
def appleSalesProblem (s : AppleSales) : Prop :=
  s.day2 = s.day1 / 4 + 8 ∧
  s.day3 = s.day2 / 4 + 8 ∧
  s.day3 = 18

/-- The theorem stating that if the conditions are met, 
    the sales on the first day were 128 kg. -/
theorem apple_sales_theorem (s : AppleSales) :
  appleSalesProblem s → s.day1 = 128 := by
  sorry

#check apple_sales_theorem

end NUMINAMATH_CALUDE_apple_sales_theorem_l3989_398907


namespace NUMINAMATH_CALUDE_actual_distance_calculation_l3989_398933

/-- Given a map distance and scale, calculate the actual distance between two towns. -/
theorem actual_distance_calculation (map_distance : ℝ) (scale_distance : ℝ) (scale_miles : ℝ) : 
  map_distance = 20 → scale_distance = 0.5 → scale_miles = 10 → 
  (map_distance * scale_miles / scale_distance) = 400 := by
sorry

end NUMINAMATH_CALUDE_actual_distance_calculation_l3989_398933


namespace NUMINAMATH_CALUDE_john_drive_distance_l3989_398951

-- Define the constants
def speed : ℝ := 55
def time_before_lunch : ℝ := 2
def time_after_lunch : ℝ := 3

-- Define the total distance function
def total_distance (s t1 t2 : ℝ) : ℝ := s * (t1 + t2)

-- Theorem statement
theorem john_drive_distance :
  total_distance speed time_before_lunch time_after_lunch = 275 := by
  sorry

end NUMINAMATH_CALUDE_john_drive_distance_l3989_398951


namespace NUMINAMATH_CALUDE_trajectory_of_center_l3989_398901

-- Define the circles F1 and F2
def circle_F1 (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 1
def circle_F2 (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 25

-- Define the property of being externally tangent
def externally_tangent (C_x C_y R : ℝ) : Prop :=
  (C_x + 1)^2 + C_y^2 = (R + 1)^2

-- Define the property of being internally tangent
def internally_tangent (C_x C_y R : ℝ) : Prop :=
  (C_x - 1)^2 + C_y^2 = (5 - R)^2

-- Theorem stating the trajectory of the center C
theorem trajectory_of_center :
  ∀ C_x C_y R : ℝ,
  externally_tangent C_x C_y R →
  internally_tangent C_x C_y R →
  C_x^2 / 9 + C_y^2 / 8 = 1 :=
sorry

end NUMINAMATH_CALUDE_trajectory_of_center_l3989_398901


namespace NUMINAMATH_CALUDE_sixth_sum_is_189_l3989_398930

/-- A sequence and its partial sums satisfying the given condition -/
def SequenceWithSum (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → S n = 2 * a n - 3

/-- The sixth partial sum of the sequence is 189 -/
theorem sixth_sum_is_189 (a : ℕ → ℝ) (S : ℕ → ℝ) 
    (h : SequenceWithSum a S) : S 6 = 189 := by
  sorry

end NUMINAMATH_CALUDE_sixth_sum_is_189_l3989_398930


namespace NUMINAMATH_CALUDE_tangent_line_to_quartic_l3989_398931

/-- The value of b for which y = x^4 is tangent to y = 4x + b is -3 -/
theorem tangent_line_to_quartic (x : ℝ) : 
  ∃ (m n : ℝ), 
    n = m^4 ∧ 
    n = 4*m + (-3) ∧ 
    (4:ℝ) = 4*m^3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_to_quartic_l3989_398931


namespace NUMINAMATH_CALUDE_hex_lattice_equilateral_triangles_l3989_398947

/-- Represents a point in a 2D hexagonal lattice -/
structure LatticePoint where
  x : ℝ
  y : ℝ

/-- Represents the hexagonal lattice -/
def HexagonalLattice : Type := List LatticePoint

/-- Calculates the distance between two points -/
def distance (p1 p2 : LatticePoint) : ℝ := sorry

/-- Checks if three points form an equilateral triangle -/
def isEquilateralTriangle (p1 p2 p3 : LatticePoint) : Bool := sorry

/-- Counts the number of equilateral triangles in the lattice -/
def countEquilateralTriangles (lattice : HexagonalLattice) : Nat := sorry

/-- The hexagonal lattice with 7 points -/
def hexLattice : HexagonalLattice := sorry

theorem hex_lattice_equilateral_triangles :
  countEquilateralTriangles hexLattice = 6 := by sorry

end NUMINAMATH_CALUDE_hex_lattice_equilateral_triangles_l3989_398947


namespace NUMINAMATH_CALUDE_sum_between_nine_half_and_ten_l3989_398927

theorem sum_between_nine_half_and_ten : 
  let sum := (29/9 : ℚ) + (11/4 : ℚ) + (81/20 : ℚ)
  (9.5 : ℚ) < sum ∧ sum < (10 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_sum_between_nine_half_and_ten_l3989_398927


namespace NUMINAMATH_CALUDE_poly_properties_l3989_398954

/-- The polynomial under consideration -/
def p (x y : ℝ) : ℝ := 2*x*y - x^2*y + 3*x^3*y - 5

/-- The degree of a term in a polynomial of two variables -/
def term_degree (a b : ℕ) : ℕ := a + b

/-- The degree of the polynomial p -/
def poly_degree : ℕ := 4

/-- The number of terms in the polynomial p -/
def num_terms : ℕ := 4

theorem poly_properties :
  (∃ x y : ℝ, term_degree 3 1 = poly_degree ∧ p x y ≠ 0) ∧
  num_terms = 4 :=
sorry

end NUMINAMATH_CALUDE_poly_properties_l3989_398954


namespace NUMINAMATH_CALUDE_carson_total_stars_l3989_398968

/-- The number of gold stars Carson earned yesterday -/
def stars_yesterday : ℕ := 6

/-- The number of gold stars Carson earned today -/
def stars_today : ℕ := 9

/-- The total number of gold stars Carson earned -/
def total_stars : ℕ := stars_yesterday + stars_today

theorem carson_total_stars : total_stars = 15 := by
  sorry

end NUMINAMATH_CALUDE_carson_total_stars_l3989_398968


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3989_398925

def polynomial (x : ℝ) : ℝ := -3*(x^8 - 2*x^5 + x^3 - 6) + 5*(2*x^4 - 3*x + 1) - 2*(x^6 - 5)

theorem sum_of_coefficients : 
  (polynomial 1) = 26 := by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3989_398925


namespace NUMINAMATH_CALUDE_vip_seat_cost_l3989_398972

theorem vip_seat_cost (total_tickets : ℕ) (total_revenue : ℕ) (general_price : ℕ) (ticket_difference : ℕ) :
  total_tickets = 320 →
  total_revenue = 7500 →
  general_price = 20 →
  ticket_difference = 276 →
  ∃ vip_price : ℕ,
    vip_price = 70 ∧
    (total_tickets - ticket_difference) * general_price + ticket_difference * vip_price = total_revenue :=
by
  sorry

#check vip_seat_cost

end NUMINAMATH_CALUDE_vip_seat_cost_l3989_398972


namespace NUMINAMATH_CALUDE_coffee_cost_for_three_dozen_l3989_398996

/-- Calculates the cost of coffee for a given number of dozens of donuts -/
def coffee_cost (dozens : ℕ) : ℕ :=
  let donuts_per_dozen : ℕ := 12
  let coffee_per_donut : ℕ := 2
  let coffee_per_pot : ℕ := 12
  let cost_per_pot : ℕ := 3
  let total_donuts : ℕ := dozens * donuts_per_dozen
  let total_coffee : ℕ := total_donuts * coffee_per_donut
  let pots_needed : ℕ := (total_coffee + coffee_per_pot - 1) / coffee_per_pot
  pots_needed * cost_per_pot

theorem coffee_cost_for_three_dozen : coffee_cost 3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_coffee_cost_for_three_dozen_l3989_398996


namespace NUMINAMATH_CALUDE_trig_expression_evaluation_l3989_398990

theorem trig_expression_evaluation : 
  (Real.sqrt 3 * Real.tan (12 * π / 180) - 3) / 
  (Real.sin (12 * π / 180) * (4 * Real.cos (12 * π / 180) ^ 2 - 2)) = -2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_evaluation_l3989_398990


namespace NUMINAMATH_CALUDE_travel_ratio_l3989_398998

/-- The ratio of distances in a specific travel scenario -/
theorem travel_ratio (d x : ℝ) (h1 : 0 < d) (h2 : 0 < x) (h3 : x < d) :
  (d - x) / 1 = x / 1 + d / 7 → x / (d - x) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_travel_ratio_l3989_398998


namespace NUMINAMATH_CALUDE_money_distribution_l3989_398929

theorem money_distribution (A B C : ℤ) 
  (total : A + B + C = 300)
  (AC_sum : A + C = 200)
  (BC_sum : B + C = 350) :
  C = 250 := by
sorry

end NUMINAMATH_CALUDE_money_distribution_l3989_398929


namespace NUMINAMATH_CALUDE_smallest_sum_of_sequence_l3989_398999

theorem smallest_sum_of_sequence (P Q R S : ℤ) : 
  P > 0 → Q > 0 → R > 0 →  -- P, Q, R are positive integers
  (R - Q = Q - P) →  -- P, Q, R form an arithmetic sequence
  (R * R = Q * S) →  -- Q, R, S form a geometric sequence
  (R = (4 * Q) / 3) →  -- R/Q = 4/3
  (∀ P' Q' R' S' : ℤ, 
    P' > 0 → Q' > 0 → R' > 0 → 
    (R' - Q' = Q' - P') → 
    (R' * R' = Q' * S') → 
    (R' = (4 * Q') / 3) → 
    P + Q + R + S ≤ P' + Q' + R' + S') →
  P + Q + R + S = 171 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_sequence_l3989_398999


namespace NUMINAMATH_CALUDE_cube_less_than_triple_l3989_398980

theorem cube_less_than_triple : ∃! x : ℤ, x^3 < 3*x :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_cube_less_than_triple_l3989_398980


namespace NUMINAMATH_CALUDE_max_value_cubic_sum_l3989_398987

theorem max_value_cubic_sum (x y : ℝ) 
  (h_pos_x : 0 < x) (h_pos_y : 0 < y) 
  (h_x_bound : x ≤ 2) (h_y_bound : y ≤ 3) 
  (h_sum : x + y = 3) : 
  (∀ a b : ℝ, 0 < a → 0 < b → a ≤ 2 → b ≤ 3 → a + b = 3 → 4*a^3 + b^3 ≤ 4*x^3 + y^3) → 
  4*x^3 + y^3 = 33 := by
sorry

end NUMINAMATH_CALUDE_max_value_cubic_sum_l3989_398987


namespace NUMINAMATH_CALUDE_book_pages_count_l3989_398967

/-- Given a book with 24 chapters that Frank read in 6 days at a rate of 102 pages per day,
    prove that the total number of pages in the book is 612. -/
theorem book_pages_count (chapters : ℕ) (days : ℕ) (pages_per_day : ℕ) 
  (h1 : chapters = 24)
  (h2 : days = 6)
  (h3 : pages_per_day = 102) :
  chapters * (days * pages_per_day) / chapters = 612 :=
by sorry

end NUMINAMATH_CALUDE_book_pages_count_l3989_398967


namespace NUMINAMATH_CALUDE_min_value_expression_l3989_398900

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + 1 / (2 * y))^2 + (y + 1 / (2 * x))^2 ≥ 4 ∧
  ((x + 1 / (2 * y))^2 + (y + 1 / (2 * x))^2 = 4 ↔ x = y ∧ x = Real.sqrt 2 / 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3989_398900


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_second_quadrant_condition_l3989_398960

-- Define the complex number z as a function of m
def z (m : ℝ) : ℂ := Complex.mk (m^2 - m - 6) (m^2 - 11*m + 24)

-- Theorem for part 1
theorem pure_imaginary_condition (m : ℝ) :
  z m = Complex.I * Complex.im (z m) ↔ m = -2 :=
sorry

-- Theorem for part 2
theorem second_quadrant_condition (m : ℝ) :
  Complex.re (z m) < 0 ∧ Complex.im (z m) > 0 ↔ -2 < m ∧ m < 3 :=
sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_second_quadrant_condition_l3989_398960


namespace NUMINAMATH_CALUDE_rachel_earnings_calculation_l3989_398963

/-- Rachel's earnings in one hour -/
def rachel_earnings (base_wage : ℚ) (num_customers : ℕ) (tip_per_customer : ℚ) : ℚ :=
  base_wage + num_customers * tip_per_customer

/-- Theorem: Rachel's earnings in one hour -/
theorem rachel_earnings_calculation : 
  rachel_earnings 12 20 (5/4) = 37 := by
  sorry

end NUMINAMATH_CALUDE_rachel_earnings_calculation_l3989_398963


namespace NUMINAMATH_CALUDE_rectangular_box_width_is_correct_boxes_fit_in_wooden_box_l3989_398915

/-- The width of rectangular boxes that fit in a wooden box -/
def rectangular_box_width : ℝ :=
  let wooden_box_length : ℝ := 800  -- 8 m in cm
  let wooden_box_width : ℝ := 700   -- 7 m in cm
  let wooden_box_height : ℝ := 600  -- 6 m in cm
  let box_length : ℝ := 8
  let box_height : ℝ := 6
  let max_boxes : ℕ := 1000000
  7  -- Width of rectangular boxes in cm

theorem rectangular_box_width_is_correct : rectangular_box_width = 7 := by
  sorry

/-- The volume of the wooden box in cubic centimeters -/
def wooden_box_volume : ℝ :=
  800 * 700 * 600

/-- The volume of a single rectangular box in cubic centimeters -/
def single_box_volume (w : ℝ) : ℝ :=
  8 * w * 6

/-- The total volume of all rectangular boxes -/
def total_boxes_volume (w : ℝ) : ℝ :=
  1000000 * single_box_volume w

theorem boxes_fit_in_wooden_box :
  total_boxes_volume rectangular_box_width ≤ wooden_box_volume := by
  sorry

end NUMINAMATH_CALUDE_rectangular_box_width_is_correct_boxes_fit_in_wooden_box_l3989_398915


namespace NUMINAMATH_CALUDE_expression_simplification_l3989_398983

theorem expression_simplification (x y : ℝ) (n : Nat) (h1 : x > 0) (h2 : y > 0) (h3 : x ≠ y) (h4 : n = 2 ∨ n = 3 ∨ n = 4) :
  let r := (x^2 + y^2) / (2*x*y)
  (((Real.sqrt (r + 1) - Real.sqrt (r - 1))^n - (Real.sqrt (r + 1) + Real.sqrt (r - 1))^n) /
   ((Real.sqrt (r + 1) - Real.sqrt (r - 1))^n + (Real.sqrt (r + 1) + Real.sqrt (r - 1))^n)) =
  (y^n - x^n) / (y^n + x^n) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3989_398983


namespace NUMINAMATH_CALUDE_positive_cube_sum_inequality_l3989_398918

theorem positive_cube_sum_inequality (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a^3 + b^3 = 2) : 
  (a + b) * (a^5 + b^5) ≥ 4 ∧ a + b ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_positive_cube_sum_inequality_l3989_398918


namespace NUMINAMATH_CALUDE_midpoint_of_intersections_l3989_398912

/-- The line equation y = x - 3 -/
def line_eq (x y : ℝ) : Prop := y = x - 3

/-- The parabola equation y^2 = 2x -/
def parabola_eq (x y : ℝ) : Prop := y^2 = 2*x

/-- A point (x, y) is on both the line and the parabola -/
def intersection_point (x y : ℝ) : Prop := line_eq x y ∧ parabola_eq x y

/-- There exist two distinct intersection points -/
axiom two_intersections : ∃ (x₁ y₁ x₂ y₂ : ℝ), 
  x₁ ≠ x₂ ∧ intersection_point x₁ y₁ ∧ intersection_point x₂ y₂

theorem midpoint_of_intersections : 
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁ ≠ x₂ ∧ 
    intersection_point x₁ y₁ ∧ 
    intersection_point x₂ y₂ ∧ 
    ((x₁ + x₂) / 2 = 4 ∧ (y₁ + y₂) / 2 = 1) :=
sorry

end NUMINAMATH_CALUDE_midpoint_of_intersections_l3989_398912


namespace NUMINAMATH_CALUDE_prob_shortest_diagonal_21_sided_l3989_398946

/-- The number of sides in the regular polygon -/
def n : ℕ := 21

/-- The number of shortest diagonals in a regular n-sided polygon -/
def num_shortest_diagonals (n : ℕ) : ℕ := n / 2

/-- The total number of diagonals in a regular n-sided polygon -/
def total_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- The probability of randomly selecting one of the shortest diagonals
    from all the diagonals of a regular n-sided polygon -/
def prob_shortest_diagonal (n : ℕ) : ℚ :=
  (num_shortest_diagonals n : ℚ) / (total_diagonals n : ℚ)

theorem prob_shortest_diagonal_21_sided :
  prob_shortest_diagonal n = 10 / 189 := by
  sorry

end NUMINAMATH_CALUDE_prob_shortest_diagonal_21_sided_l3989_398946


namespace NUMINAMATH_CALUDE_balloon_count_is_22_l3989_398923

/-- The number of balloons each person brought to the park -/
structure BalloonCount where
  allan : ℕ
  jake : ℕ
  maria : ℕ
  tom_initial : ℕ
  tom_lost : ℕ

/-- The total number of balloons in the park -/
def total_balloons (bc : BalloonCount) : ℕ :=
  bc.allan + bc.jake + bc.maria + (bc.tom_initial - bc.tom_lost)

/-- Theorem: The total number of balloons in the park is 22 -/
theorem balloon_count_is_22 (bc : BalloonCount) 
    (h1 : bc.allan = 5)
    (h2 : bc.jake = 7)
    (h3 : bc.maria = 3)
    (h4 : bc.tom_initial = 9)
    (h5 : bc.tom_lost = 2) : 
  total_balloons bc = 22 := by
  sorry

end NUMINAMATH_CALUDE_balloon_count_is_22_l3989_398923


namespace NUMINAMATH_CALUDE_complex_multiplication_result_l3989_398969

theorem complex_multiplication_result : (1 + 2 * Complex.I) * (1 - Complex.I) = 3 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_result_l3989_398969


namespace NUMINAMATH_CALUDE_pascals_triangle_ratio_l3989_398962

theorem pascals_triangle_ratio (n : ℕ) (r : ℕ) : n = 84 →
  ∃ r, r + 2 ≤ n ∧
    (Nat.choose n r : ℚ) / (Nat.choose n (r + 1)) = 5 / 6 ∧
    (Nat.choose n (r + 1) : ℚ) / (Nat.choose n (r + 2)) = 6 / 7 :=
by
  sorry


end NUMINAMATH_CALUDE_pascals_triangle_ratio_l3989_398962


namespace NUMINAMATH_CALUDE_circle_center_l3989_398966

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- Represents a circle in polar coordinates -/
structure PolarCircle where
  equation : ℝ → ℝ → Prop

theorem circle_center (C : PolarCircle) :
  C.equation = (fun ρ θ ↦ ρ = 2 * Real.cos (θ + π/4)) →
  ∃ (center : PolarPoint), center.r = 1 ∧ center.θ = -π/4 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_l3989_398966


namespace NUMINAMATH_CALUDE_vacation_pictures_remaining_l3989_398922

def zoo_pictures : ℕ := 15
def museum_pictures : ℕ := 18
def deleted_pictures : ℕ := 31

theorem vacation_pictures_remaining :
  zoo_pictures + museum_pictures - deleted_pictures = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_vacation_pictures_remaining_l3989_398922


namespace NUMINAMATH_CALUDE_probability_two_red_shoes_l3989_398961

theorem probability_two_red_shoes :
  let total_shoes : ℕ := 9
  let red_shoes : ℕ := 5
  let green_shoes : ℕ := 4
  let draw_count : ℕ := 2
  
  total_shoes = red_shoes + green_shoes →
  (Nat.choose red_shoes draw_count : ℚ) / (Nat.choose total_shoes draw_count : ℚ) = 5 / 18 :=
by sorry

end NUMINAMATH_CALUDE_probability_two_red_shoes_l3989_398961


namespace NUMINAMATH_CALUDE_uncool_parents_count_l3989_398958

theorem uncool_parents_count (total : ℕ) (cool_dads : ℕ) (cool_moms : ℕ) (cool_both : ℕ) : 
  total = 30 → cool_dads = 12 → cool_moms = 15 → cool_both = 9 →
  total - (cool_dads + cool_moms - cool_both) = 12 :=
by sorry

end NUMINAMATH_CALUDE_uncool_parents_count_l3989_398958


namespace NUMINAMATH_CALUDE_line_intersection_parameter_range_l3989_398953

/-- Given two points A and B, and a line that intersects the line segment AB,
    this theorem proves the range of the parameter m in the line equation. -/
theorem line_intersection_parameter_range :
  let A : ℝ × ℝ := (-1, 2)
  let B : ℝ × ℝ := (2, -1)
  let line (m : ℝ) (x y : ℝ) := x - 2*y + m = 0
  ∀ m : ℝ, (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 
    line m ((1-t)*A.1 + t*B.1) ((1-t)*A.2 + t*B.2)) ↔ 
  -4 ≤ m ∧ m ≤ 5 := by
sorry

end NUMINAMATH_CALUDE_line_intersection_parameter_range_l3989_398953


namespace NUMINAMATH_CALUDE_a_approximation_l3989_398995

/-- For large x, the value of a that makes (a * x) / (0.5x - 406) closest to 3 is approximately 1.5 -/
theorem a_approximation (x : ℝ) (hx : x > 3000) :
  ∃ (a : ℝ), ∀ (ε : ℝ), ε > 0 → 
    ∃ (δ : ℝ), δ > 0 ∧ 
      ∀ (y : ℝ), y > x → 
        |((a * y) / (0.5 * y - 406) - 3)| < ε ∧ 
        |a - 1.5| < δ :=
sorry

end NUMINAMATH_CALUDE_a_approximation_l3989_398995


namespace NUMINAMATH_CALUDE_square_side_length_l3989_398950

theorem square_side_length (d : ℝ) (h : d = 2) :
  ∃ s : ℝ, s * s = 2 ∧ s * Real.sqrt 2 = d :=
by sorry

end NUMINAMATH_CALUDE_square_side_length_l3989_398950


namespace NUMINAMATH_CALUDE_ratio_problem_l3989_398991

/-- Given that a:b = 4:3 and a:c = 4:15, prove that b:c = 1:5 -/
theorem ratio_problem (a b c : ℚ) 
  (hab : a / b = 4 / 3) 
  (hac : a / c = 4 / 15) : 
  b / c = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_ratio_problem_l3989_398991


namespace NUMINAMATH_CALUDE_monotonic_f_implies_m_range_inequality_implies_a_range_l3989_398974

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x * Real.log x + m * x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := -x^2 + a * x - 3

theorem monotonic_f_implies_m_range (m : ℝ) :
  (∀ x₁ x₂, 1 < x₁ ∧ x₁ < x₂ → f m x₁ < f m x₂) → m ≤ -1 := by sorry

theorem inequality_implies_a_range (a : ℝ) :
  (∀ x, x > 0 → 2 * (f 0 x) ≥ g a x) → a ≤ 4 := by sorry

end NUMINAMATH_CALUDE_monotonic_f_implies_m_range_inequality_implies_a_range_l3989_398974


namespace NUMINAMATH_CALUDE_road_length_probability_l3989_398917

theorem road_length_probability : 
  ∀ (p_ab p_bc : ℝ),
    0 ≤ p_ab ∧ p_ab ≤ 1 →
    0 ≤ p_bc ∧ p_bc ≤ 1 →
    p_ab = 2/3 →
    p_bc = 1/2 →
    1 - (1 - p_ab) * (1 - p_bc) = 5/6 :=
by
  sorry

end NUMINAMATH_CALUDE_road_length_probability_l3989_398917


namespace NUMINAMATH_CALUDE_stacy_homework_problem_l3989_398957

/-- Represents the number of homework problems assigned by Stacy. -/
def homework_problems : ℕ → ℕ → ℕ → ℕ 
  | true_false, free_response, multiple_choice => 
    true_false + free_response + multiple_choice

theorem stacy_homework_problem :
  ∃ (true_false free_response multiple_choice : ℕ),
    true_false = 6 ∧
    free_response = true_false + 7 ∧
    multiple_choice = 2 * free_response ∧
    homework_problems true_false free_response multiple_choice = 45 :=
by
  sorry

#check stacy_homework_problem

end NUMINAMATH_CALUDE_stacy_homework_problem_l3989_398957


namespace NUMINAMATH_CALUDE_cubic_roots_sum_of_squares_reciprocals_l3989_398914

theorem cubic_roots_sum_of_squares_reciprocals (p q r : ℂ) : 
  p^3 - 15*p^2 + 26*p + 3 = 0 →
  q^3 - 15*q^2 + 26*q + 3 = 0 →
  r^3 - 15*r^2 + 26*r + 3 = 0 →
  p ≠ q → p ≠ r → q ≠ r →
  1/p^2 + 1/q^2 + 1/r^2 = 766/9 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_of_squares_reciprocals_l3989_398914


namespace NUMINAMATH_CALUDE_smallest_staircase_steps_l3989_398932

theorem smallest_staircase_steps (n : ℕ) : 
  n > 20 ∧ 
  n % 6 = 4 ∧ 
  n % 7 = 3 ∧ 
  (∀ m : ℕ, m > 20 ∧ m % 6 = 4 ∧ m % 7 = 3 → m ≥ n) → 
  n = 52 := by
sorry

end NUMINAMATH_CALUDE_smallest_staircase_steps_l3989_398932


namespace NUMINAMATH_CALUDE_board_number_remainder_l3989_398926

theorem board_number_remainder (n a b c d : ℕ) : 
  n = 102 * a + b ∧ 
  n = 103 * c + d ∧ 
  a + d = 20 ∧ 
  b < 102 →
  b = 20 := by
sorry

end NUMINAMATH_CALUDE_board_number_remainder_l3989_398926


namespace NUMINAMATH_CALUDE_square_binomial_expansion_l3989_398970

theorem square_binomial_expansion (x : ℝ) : (x - 2)^2 = x^2 - 4*x + 4 := by
  sorry

end NUMINAMATH_CALUDE_square_binomial_expansion_l3989_398970


namespace NUMINAMATH_CALUDE_problem_solution_l3989_398904

theorem problem_solution (x y n : ℝ) : 
  x = 3 → y = 1 → n = x - y^(x-y) → n = 2 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l3989_398904


namespace NUMINAMATH_CALUDE_transportation_problem_l3989_398939

/-- Transportation problem between warehouses and factories -/
theorem transportation_problem 
  (warehouse_a warehouse_b : ℕ)
  (factory_a factory_b : ℕ)
  (cost_a_to_a cost_a_to_b cost_b_to_a cost_b_to_b : ℕ)
  (total_cost : ℕ)
  (h1 : warehouse_a = 20)
  (h2 : warehouse_b = 6)
  (h3 : factory_a = 10)
  (h4 : factory_b = 16)
  (h5 : cost_a_to_a = 400)
  (h6 : cost_a_to_b = 800)
  (h7 : cost_b_to_a = 300)
  (h8 : cost_b_to_b = 500)
  (h9 : total_cost = 16000) :
  ∃ (x y : ℕ),
    x + (warehouse_b - y) = factory_a ∧
    (warehouse_a - x) + y = factory_b ∧
    cost_a_to_a * x + cost_a_to_b * (warehouse_a - x) + 
    cost_b_to_a * (warehouse_b - y) + cost_b_to_b * y = total_cost ∧
    x = 5 ∧ y = 1 :=
by sorry

end NUMINAMATH_CALUDE_transportation_problem_l3989_398939


namespace NUMINAMATH_CALUDE_cover_room_with_tiles_l3989_398936

/-- The width of the room -/
def room_width : ℝ := 8

/-- The length of the room -/
def room_length : ℝ := 12

/-- The width of a tile -/
def tile_width : ℝ := 1.5

/-- The length of a tile -/
def tile_length : ℝ := 2

/-- The number of tiles needed to cover the room -/
def tiles_needed : ℕ := 32

theorem cover_room_with_tiles :
  (room_width * room_length) / (tile_width * tile_length) = tiles_needed := by
  sorry

end NUMINAMATH_CALUDE_cover_room_with_tiles_l3989_398936


namespace NUMINAMATH_CALUDE_brittany_brooke_money_ratio_l3989_398997

/-- Given the following conditions about money possession:
  - Alison has half as much money as Brittany
  - Brooke has twice as much money as Kent
  - Kent has $1,000
  - Alison has $4,000
Prove that Brittany has 4 times as much money as Brooke -/
theorem brittany_brooke_money_ratio :
  ∀ (alison brittany brooke kent : ℝ),
  alison = brittany / 2 →
  brooke = 2 * kent →
  kent = 1000 →
  alison = 4000 →
  brittany = 4 * brooke :=
by sorry

end NUMINAMATH_CALUDE_brittany_brooke_money_ratio_l3989_398997


namespace NUMINAMATH_CALUDE_total_amount_is_468_l3989_398934

/-- Calculates the total amount paid including service charge -/
def totalAmountPaid (originalAmount : ℝ) (serviceChargeRate : ℝ) : ℝ :=
  originalAmount * (1 + serviceChargeRate)

/-- Proves that the total amount paid is 468 given the conditions -/
theorem total_amount_is_468 :
  let originalAmount : ℝ := 450
  let serviceChargeRate : ℝ := 0.04
  totalAmountPaid originalAmount serviceChargeRate = 468 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_is_468_l3989_398934


namespace NUMINAMATH_CALUDE_sum_of_H_and_J_l3989_398937

theorem sum_of_H_and_J : ∃ (H J K L : ℕ),
  H ∈ ({1, 2, 5, 6} : Set ℕ) ∧
  J ∈ ({1, 2, 5, 6} : Set ℕ) ∧
  K ∈ ({1, 2, 5, 6} : Set ℕ) ∧
  L ∈ ({1, 2, 5, 6} : Set ℕ) ∧
  H ≠ J ∧ H ≠ K ∧ H ≠ L ∧ J ≠ K ∧ J ≠ L ∧ K ≠ L ∧
  (H : ℚ) / J - (K : ℚ) / L = 5 / 6 →
  H + J = 7 :=
sorry

end NUMINAMATH_CALUDE_sum_of_H_and_J_l3989_398937


namespace NUMINAMATH_CALUDE_students_not_playing_sports_l3989_398973

theorem students_not_playing_sports (total_students football_players cricket_players both_players : ℕ) 
  (h1 : total_students = 420)
  (h2 : football_players = 325)
  (h3 : cricket_players = 175)
  (h4 : both_players = 130)
  (h5 : both_players ≤ football_players)
  (h6 : both_players ≤ cricket_players)
  (h7 : football_players ≤ total_students)
  (h8 : cricket_players ≤ total_students) :
  total_students - (football_players + cricket_players - both_players) = 50 := by
sorry

end NUMINAMATH_CALUDE_students_not_playing_sports_l3989_398973


namespace NUMINAMATH_CALUDE_album_jumps_l3989_398964

/-- Calculates the number of jumps a person can do while listening to an album --/
theorem album_jumps (jumps_per_second : ℝ) (num_songs : ℕ) (song_length_minutes : ℝ) : 
  jumps_per_second = 1 →
  num_songs = 10 →
  song_length_minutes = 3.5 →
  (jumps_per_second * num_songs * song_length_minutes * 60 : ℝ) = 2100 :=
by
  sorry

end NUMINAMATH_CALUDE_album_jumps_l3989_398964


namespace NUMINAMATH_CALUDE_fraction_difference_l3989_398916

theorem fraction_difference : (7 : ℚ) / 4 - (2 : ℚ) / 3 = (13 : ℚ) / 12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_difference_l3989_398916


namespace NUMINAMATH_CALUDE_product_of_fractions_l3989_398986

theorem product_of_fractions : (1 : ℚ) / 5 * (3 : ℚ) / 7 = (3 : ℚ) / 35 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l3989_398986


namespace NUMINAMATH_CALUDE_quadratic_roots_l3989_398921

theorem quadratic_roots : 
  let f : ℝ → ℝ := λ x ↦ x^2 - 3*x
  ∃ x₁ x₂ : ℝ, x₁ = 0 ∧ x₂ = 3 ∧ 
    (∀ x : ℝ, f x = 0 ↔ x = x₁ ∨ x = x₂) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_l3989_398921


namespace NUMINAMATH_CALUDE_david_did_58_pushups_l3989_398977

/-- The number of push-ups David did -/
def davids_pushups (zachary_pushups : ℕ) (difference : ℕ) : ℕ :=
  zachary_pushups + difference

theorem david_did_58_pushups :
  davids_pushups 19 39 = 58 := by
  sorry

end NUMINAMATH_CALUDE_david_did_58_pushups_l3989_398977


namespace NUMINAMATH_CALUDE_max_plain_cookies_is_20_l3989_398920

/-- Represents the number of cookies with a specific ingredient -/
structure CookieCount where
  total : ℕ
  chocolate : ℕ
  nuts : ℕ
  raisins : ℕ
  sprinkles : ℕ

/-- The conditions of the cookie problem -/
def cookieProblem : CookieCount where
  total := 60
  chocolate := 20
  nuts := 30
  raisins := 40
  sprinkles := 15

/-- The maximum number of cookies without any of the specified ingredients -/
def maxPlainCookies (c : CookieCount) : ℕ :=
  c.total - max c.chocolate (max c.nuts (max c.raisins c.sprinkles))

/-- Theorem stating the maximum number of plain cookies in the given problem -/
theorem max_plain_cookies_is_20 :
  maxPlainCookies cookieProblem = 20 := by
  sorry

end NUMINAMATH_CALUDE_max_plain_cookies_is_20_l3989_398920


namespace NUMINAMATH_CALUDE_two_digit_primes_with_rearranged_digits_and_square_difference_l3989_398913

def is_two_digit_prime (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ Nat.Prime n

def digits_rearranged (a b : ℕ) : Prop :=
  (a / 10 = b % 10) ∧ (a % 10 = b / 10)

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

theorem two_digit_primes_with_rearranged_digits_and_square_difference :
  ∀ a b : ℕ,
    is_two_digit_prime a ∧
    is_two_digit_prime b ∧
    digits_rearranged a b ∧
    is_perfect_square (a - b) →
    (a = 73 ∧ b = 37) ∨ (a = 37 ∧ b = 73) :=
by sorry

end NUMINAMATH_CALUDE_two_digit_primes_with_rearranged_digits_and_square_difference_l3989_398913


namespace NUMINAMATH_CALUDE_iris_count_after_rose_addition_l3989_398989

/-- Given a garden with an initial ratio of irises to roses of 3:7,
    and an initial count of 35 roses, prove that after adding 30 roses,
    the number of irises that maintains the ratio is 27. -/
theorem iris_count_after_rose_addition 
  (initial_roses : ℕ) 
  (added_roses : ℕ) 
  (iris_ratio : ℕ) 
  (rose_ratio : ℕ) : 
  initial_roses = 35 →
  added_roses = 30 →
  iris_ratio = 3 →
  rose_ratio = 7 →
  (∃ (total_irises : ℕ), 
    total_irises * rose_ratio = (initial_roses + added_roses) * iris_ratio ∧
    total_irises = 27) :=
by sorry

end NUMINAMATH_CALUDE_iris_count_after_rose_addition_l3989_398989


namespace NUMINAMATH_CALUDE_digit_symmetrical_equation_l3989_398942

theorem digit_symmetrical_equation (a b : ℤ) (h : 2 ≤ a + b ∧ a + b ≤ 9) :
  (10*a + b) * (100*b + 10*(a + b) + a) = (100*a + 10*(a + b) + b) * (10*b + a) := by
  sorry

end NUMINAMATH_CALUDE_digit_symmetrical_equation_l3989_398942


namespace NUMINAMATH_CALUDE_jennifers_money_l3989_398959

theorem jennifers_money (initial_amount : ℚ) : 
  initial_amount > 0 →
  initial_amount - (initial_amount / 5 + initial_amount / 6 + initial_amount / 2) = 12 →
  initial_amount = 90 := by
sorry

end NUMINAMATH_CALUDE_jennifers_money_l3989_398959


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l3989_398994

-- Define the vectors
def a : Fin 2 → ℝ := ![2, 3]
def b (x : ℝ) : Fin 2 → ℝ := ![x, 1]

-- Define perpendicularity condition
def perpendicular (u v : Fin 2 → ℝ) : Prop :=
  u 0 * v 0 + u 1 * v 1 = 0

theorem perpendicular_vectors_x_value :
  ∀ x : ℝ, perpendicular a (b x) → x = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l3989_398994


namespace NUMINAMATH_CALUDE_equation_solution_set_l3989_398928

theorem equation_solution_set : 
  {x : ℝ | x > 0 ∧ x^(Real.log x / Real.log 10) = x^4 / 1000} = {10, 1000} := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_set_l3989_398928


namespace NUMINAMATH_CALUDE_y_completion_time_l3989_398906

/-- Represents the time it takes for worker y to complete the job alone -/
def y_time : ℝ := 12

/-- Represents the time it takes for workers x and y to complete the job together -/
def xy_time : ℝ := 20

/-- Represents the number of days x worked alone before y joined -/
def x_solo_days : ℝ := 4

/-- Represents the total number of days the job took to complete -/
def total_days : ℝ := 10

/-- Represents the portion of work completed in one day -/
def work_unit : ℝ := 1

theorem y_completion_time :
  (x_solo_days * (work_unit / xy_time)) +
  ((total_days - x_solo_days) * (work_unit / xy_time + work_unit / y_time)) = work_unit :=
sorry

end NUMINAMATH_CALUDE_y_completion_time_l3989_398906


namespace NUMINAMATH_CALUDE_prob_at_least_two_of_six_l3989_398902

/-- The number of questions randomly guessed -/
def n : ℕ := 6

/-- The number of choices for each question -/
def k : ℕ := 5

/-- The probability of getting a single question correct -/
def p : ℚ := 1 / k

/-- The probability of getting a single question incorrect -/
def q : ℚ := 1 - p

/-- The probability of getting at least two questions correct out of n questions -/
def prob_at_least_two (n : ℕ) (p : ℚ) : ℚ :=
  1 - (q ^ n + n * p * q ^ (n - 1))

theorem prob_at_least_two_of_six :
  prob_at_least_two n p = 5385 / 15625 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_two_of_six_l3989_398902


namespace NUMINAMATH_CALUDE_income_expenditure_ratio_l3989_398949

/-- Given income and savings, calculate the ratio of income to expenditure --/
theorem income_expenditure_ratio 
  (income : ℕ) 
  (savings : ℕ) 
  (expenditure : ℕ) 
  (h1 : income = 16000) 
  (h2 : savings = 3200) 
  (h3 : savings = income - expenditure) : 
  (income : ℚ) / expenditure = 5 / 4 := by
sorry

end NUMINAMATH_CALUDE_income_expenditure_ratio_l3989_398949


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_expression_l3989_398944

theorem simplify_and_evaluate_expression (x y : ℚ) 
  (hx : x = -1) (hy : y = 1/5) : 
  2 * (x^2 * y - 2 * x * y) - 3 * (x^2 * y - 3 * x * y) + x^2 * y = -1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_expression_l3989_398944


namespace NUMINAMATH_CALUDE_unique_number_with_specific_remainders_l3989_398993

theorem unique_number_with_specific_remainders :
  ∃! n : ℕ, ∀ k ∈ Finset.range 11, n % (k + 2) = k + 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_with_specific_remainders_l3989_398993


namespace NUMINAMATH_CALUDE_line_passes_first_third_quadrants_l3989_398985

/-- A line y = kx passes through the first and third quadrants if and only if k > 0 -/
theorem line_passes_first_third_quadrants (k : ℝ) (hk : k ≠ 0) :
  (∀ x y : ℝ, y = k * x → ((x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0))) ↔ k > 0 :=
sorry

end NUMINAMATH_CALUDE_line_passes_first_third_quadrants_l3989_398985


namespace NUMINAMATH_CALUDE_binomial_expansion_example_l3989_398908

theorem binomial_expansion_example : 7^3 + 3*(7^2)*2 + 3*7*(2^2) + 2^3 = (7 + 2)^3 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_example_l3989_398908


namespace NUMINAMATH_CALUDE_trailer_homes_added_l3989_398965

/-- Represents the number of trailer homes added 5 years ago -/
def added_homes : ℕ := sorry

/-- The initial number of trailer homes -/
def initial_homes : ℕ := 30

/-- The initial average age of trailer homes in years -/
def initial_avg_age : ℕ := 15

/-- The age of added homes when they were added, in years -/
def added_homes_age : ℕ := 3

/-- The number of years that have passed since new homes were added -/
def years_passed : ℕ := 5

/-- The current average age of all trailer homes in years -/
def current_avg_age : ℕ := 17

theorem trailer_homes_added :
  (initial_homes * (initial_avg_age + years_passed) + added_homes * (added_homes_age + years_passed)) /
  (initial_homes + added_homes) = current_avg_age →
  added_homes = 10 := by sorry

end NUMINAMATH_CALUDE_trailer_homes_added_l3989_398965


namespace NUMINAMATH_CALUDE_student_calculation_error_l3989_398955

theorem student_calculation_error (x : ℤ) : 
  (x + 5) - (x - (-5)) = 10 :=
sorry

end NUMINAMATH_CALUDE_student_calculation_error_l3989_398955


namespace NUMINAMATH_CALUDE_cubic_root_sum_cubes_l3989_398935

theorem cubic_root_sum_cubes (a b c : ℝ) : 
  (a^3 - 3*a^2 + 4*a - 5 = 0) → 
  (b^3 - 3*b^2 + 4*b - 5 = 0) → 
  (c^3 - 3*c^2 + 4*c - 5 = 0) → 
  a^3 + b^3 + c^3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_cubes_l3989_398935


namespace NUMINAMATH_CALUDE_equation_result_l3989_398981

theorem equation_result : 300 * 2 + (12 + 4) * 1 / 8 = 602 := by
  sorry

end NUMINAMATH_CALUDE_equation_result_l3989_398981


namespace NUMINAMATH_CALUDE_max_daily_revenue_l3989_398905

-- Define the sales price function
def P (t : ℕ) : ℝ :=
  if 0 < t ∧ t < 25 then t + 20
  else if 25 ≤ t ∧ t ≤ 30 then -t + 70
  else 0

-- Define the daily sales volume function
def Q (t : ℕ) : ℝ := -t + 40

-- Define the daily sales revenue function
def R (t : ℕ) : ℝ := P t * Q t

-- Theorem statement
theorem max_daily_revenue :
  (∃ t : ℕ, 1 ≤ t ∧ t ≤ 30 ∧ R t = 1125) ∧
  (∀ t : ℕ, 1 ≤ t ∧ t ≤ 30 → R t ≤ 1125) ∧
  (R 25 = 1125) :=
sorry

end NUMINAMATH_CALUDE_max_daily_revenue_l3989_398905


namespace NUMINAMATH_CALUDE_rita_dress_count_l3989_398940

def initial_money : ℕ := 400
def remaining_money : ℕ := 139
def pants_count : ℕ := 3
def jackets_count : ℕ := 4
def dress_price : ℕ := 20
def pants_price : ℕ := 12
def jacket_price : ℕ := 30
def transportation_cost : ℕ := 5

theorem rita_dress_count :
  let total_spent := initial_money - remaining_money
  let pants_jackets_cost := pants_count * pants_price + jackets_count * jacket_price
  let dress_total_cost := total_spent - pants_jackets_cost - transportation_cost
  dress_total_cost / dress_price = 5 := by sorry

end NUMINAMATH_CALUDE_rita_dress_count_l3989_398940


namespace NUMINAMATH_CALUDE_polynomial_primes_theorem_l3989_398975

def is_valid_polynomial (Q : ℤ → ℤ) : Prop :=
  ∃ (a b c : ℤ), ∀ x, Q x = a * x^2 + b * x + c

def satisfies_condition (Q : ℤ → ℤ) : Prop :=
  ∃ (p₁ p₂ p₃ : ℕ), 
    Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ 
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₂ ≠ p₃ ∧
    |Q p₁| = 11 ∧ |Q p₂| = 11 ∧ |Q p₃| = 11

def is_solution (Q : ℤ → ℤ) : Prop :=
  (∀ x, Q x = 11) ∨
  (∀ x, Q x = x^2 - 13*x + 11) ∨
  (∀ x, Q x = 2*x^2 - 32*x + 67) ∨
  (∀ x, Q x = 11*x^2 - 77*x + 121)

theorem polynomial_primes_theorem :
  ∀ Q : ℤ → ℤ, is_valid_polynomial Q → satisfies_condition Q → is_solution Q :=
sorry

end NUMINAMATH_CALUDE_polynomial_primes_theorem_l3989_398975


namespace NUMINAMATH_CALUDE_lorry_weight_is_1800_l3989_398924

/-- The total weight of a fully loaded lorry -/
def lorry_weight (empty_weight : ℕ) (apple_bags : ℕ) (apple_weight : ℕ) 
  (orange_bags : ℕ) (orange_weight : ℕ) (watermelon_crates : ℕ) (watermelon_weight : ℕ)
  (firewood_bundles : ℕ) (firewood_weight : ℕ) : ℕ :=
  empty_weight + 
  apple_bags * apple_weight + 
  orange_bags * orange_weight + 
  watermelon_crates * watermelon_weight + 
  firewood_bundles * firewood_weight

/-- Theorem stating the total weight of the fully loaded lorry is 1800 pounds -/
theorem lorry_weight_is_1800 : 
  lorry_weight 500 10 55 5 45 3 125 2 75 = 1800 := by
  sorry

#eval lorry_weight 500 10 55 5 45 3 125 2 75

end NUMINAMATH_CALUDE_lorry_weight_is_1800_l3989_398924


namespace NUMINAMATH_CALUDE_girls_to_boys_ratio_l3989_398952

def physics_students : ℕ := 200
def biology_students : ℕ := physics_students / 2
def boys_in_biology : ℕ := 25

def girls_in_biology : ℕ := biology_students - boys_in_biology

theorem girls_to_boys_ratio :
  girls_in_biology / boys_in_biology = 3 := by sorry

end NUMINAMATH_CALUDE_girls_to_boys_ratio_l3989_398952


namespace NUMINAMATH_CALUDE_sqrt_two_minus_one_zero_minus_three_inv_l3989_398919

theorem sqrt_two_minus_one_zero_minus_three_inv :
  (Real.sqrt 2 - 1) ^ 0 - 3⁻¹ = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_minus_one_zero_minus_three_inv_l3989_398919


namespace NUMINAMATH_CALUDE_valid_arrangement_probability_l3989_398943

/-- Represents the color of a bead -/
inductive BeadColor
  | Green
  | Yellow
  | Purple

/-- Represents an arrangement of beads -/
def BeadArrangement := List BeadColor

/-- Checks if an arrangement is valid according to the given conditions -/
def isValidArrangement (arr : BeadArrangement) : Bool :=
  sorry

/-- Counts the number of valid arrangements -/
def countValidArrangements (green yellow purple : Nat) : Nat :=
  sorry

/-- Calculates the total number of possible arrangements -/
def totalArrangements (green yellow purple : Nat) : Nat :=
  sorry

/-- Theorem stating the probability of a valid arrangement -/
theorem valid_arrangement_probability :
  let green := 4
  let yellow := 3
  let purple := 2
  (countValidArrangements green yellow purple : Rat) / (totalArrangements green yellow purple) = 7 / 315 :=
sorry

end NUMINAMATH_CALUDE_valid_arrangement_probability_l3989_398943


namespace NUMINAMATH_CALUDE_johns_annual_profit_l3989_398978

/-- Calculates the annual profit from subletting an apartment --/
def annual_profit (num_subletters : ℕ) (subletter_rent : ℕ) (apartment_rent : ℕ) : ℕ :=
  (num_subletters * subletter_rent * 12) - (apartment_rent * 12)

/-- Theorem: John's annual profit from subletting his apartment is $3600 --/
theorem johns_annual_profit :
  annual_profit 3 400 900 = 3600 := by
  sorry

end NUMINAMATH_CALUDE_johns_annual_profit_l3989_398978
