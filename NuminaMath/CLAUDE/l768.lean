import Mathlib

namespace NUMINAMATH_CALUDE_perfect_squares_condition_l768_76866

theorem perfect_squares_condition (n : ℤ) : 
  (∃ a : ℤ, 4 * n + 1 = a ^ 2) ∧ (∃ b : ℤ, 9 * n + 1 = b ^ 2) → n = 0 :=
by sorry

end NUMINAMATH_CALUDE_perfect_squares_condition_l768_76866


namespace NUMINAMATH_CALUDE_total_balls_l768_76863

/-- Given the number of basketballs, volleyballs, and soccer balls in a school,
    prove that the total number of balls is 94. -/
theorem total_balls (b v s : ℕ) : 
  b = 32 →
  b = v + 5 →
  b = s - 3 →
  b + v + s = 94 := by
  sorry

end NUMINAMATH_CALUDE_total_balls_l768_76863


namespace NUMINAMATH_CALUDE_petya_wins_l768_76867

/-- Represents a 7x7 game board --/
def GameBoard := Fin 7 → Fin 7 → Option (Fin 7)

/-- Checks if a move is valid on the given board --/
def is_valid_move (board : GameBoard) (row col : Fin 7) (digit : Fin 7) : Prop :=
  (∀ i : Fin 7, board i col ≠ some digit) ∧
  (∀ j : Fin 7, board row j ≠ some digit)

/-- Represents a player's strategy --/
def Strategy := GameBoard → Option (Fin 7 × Fin 7 × Fin 7)

/-- Defines a winning strategy for the first player --/
def winning_strategy (s : Strategy) : Prop :=
  ∀ (board : GameBoard),
    (∃ row col digit, is_valid_move board row col digit) →
    ∃ row col digit, s board = some (row, col, digit) ∧ is_valid_move board row col digit

theorem petya_wins : ∃ s : Strategy, winning_strategy s :=
  sorry

end NUMINAMATH_CALUDE_petya_wins_l768_76867


namespace NUMINAMATH_CALUDE_ratio_of_segments_on_line_l768_76888

/-- Given four points P, Q, R, S on a line in that order, with given distances between them,
    prove that the ratio of PR to QS is 7/12. -/
theorem ratio_of_segments_on_line (P Q R S : ℝ) (h_order : P < Q ∧ Q < R ∧ R < S)
    (h_PQ : Q - P = 4) (h_QR : R - Q = 10) (h_PS : S - P = 28) :
    (R - P) / (S - Q) = 7 / 12 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_segments_on_line_l768_76888


namespace NUMINAMATH_CALUDE_phd_team_combinations_setup_correct_l768_76883

def total_engineers : ℕ := 8
def phd_engineers : ℕ := 3
def ms_bs_engineers : ℕ := 5
def team_size : ℕ := 3

-- Function to calculate combinations
def choose (n k : ℕ) : ℕ := Nat.choose n k

-- Theorem statement
theorem phd_team_combinations : 
  choose phd_engineers 1 * choose ms_bs_engineers 2 + 
  choose phd_engineers 2 * choose ms_bs_engineers 1 + 
  choose phd_engineers 3 = 46 := by
  sorry

-- Additional theorem to ensure the setup is correct
theorem setup_correct : 
  total_engineers = phd_engineers + ms_bs_engineers ∧ 
  team_size ≤ total_engineers := by
  sorry

end NUMINAMATH_CALUDE_phd_team_combinations_setup_correct_l768_76883


namespace NUMINAMATH_CALUDE_dallas_pears_count_l768_76890

/-- The number of bags of pears Dallas picked -/
def dallas_pears : ℕ := 9

/-- The number of bags of apples Dallas picked -/
def dallas_apples : ℕ := 14

/-- The total number of bags Austin picked -/
def austin_total : ℕ := 24

theorem dallas_pears_count :
  dallas_pears = 9 ∧
  dallas_apples = 14 ∧
  austin_total = 24 ∧
  austin_total = (dallas_apples + 6) + (dallas_pears - 5) :=
by sorry

end NUMINAMATH_CALUDE_dallas_pears_count_l768_76890


namespace NUMINAMATH_CALUDE_decagon_diagonals_from_vertex_l768_76857

/-- The number of diagonals from a vertex in a regular decagon -/
def diagonals_from_vertex (n : ℕ) : ℕ := n - 3

/-- A regular decagon has 10 sides -/
def decagon_sides : ℕ := 10

/-- Theorem: The number of diagonals from a vertex in a regular decagon is 7 -/
theorem decagon_diagonals_from_vertex :
  diagonals_from_vertex decagon_sides = 7 := by
  sorry

end NUMINAMATH_CALUDE_decagon_diagonals_from_vertex_l768_76857


namespace NUMINAMATH_CALUDE_rectangle_width_decrease_l768_76814

/-- Proves that if the length of a rectangle is increased by 30% and the width is adjusted to keep the area constant, then the width is decreased by 23.08% -/
theorem rectangle_width_decrease (L W : ℝ) (L' W' : ℝ) :
  L' = 1.3 * L →  -- Length is increased by 30%
  L * W = L' * W' →  -- Area remains constant
  (W - W') / W = 0.2308 :=  -- Width decrease percentage
by sorry

end NUMINAMATH_CALUDE_rectangle_width_decrease_l768_76814


namespace NUMINAMATH_CALUDE_coefficient_x_cubed_expansion_l768_76872

/-- The coefficient of x^3 in the expansion of (1+2x^2)(1+x)^4 is 12 -/
theorem coefficient_x_cubed_expansion : ∃ (p : Polynomial ℝ),
  p = (1 + 2 * X^2) * (1 + X)^4 ∧ p.coeff 3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_cubed_expansion_l768_76872


namespace NUMINAMATH_CALUDE_cos_90_degrees_eq_zero_l768_76893

theorem cos_90_degrees_eq_zero : Real.cos (π / 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_90_degrees_eq_zero_l768_76893


namespace NUMINAMATH_CALUDE_polygon_area_is_12_l768_76849

-- Define the polygon vertices
def polygon_vertices : List (ℝ × ℝ) := [(0,0), (4,0), (4,4), (2,4), (2,2), (0,2)]

-- Define the function to calculate the area of a polygon given its vertices
def polygon_area (vertices : List (ℝ × ℝ)) : ℝ := sorry

-- Theorem stating that the area of the given polygon is 12 square units
theorem polygon_area_is_12 : polygon_area polygon_vertices = 12 := by sorry

end NUMINAMATH_CALUDE_polygon_area_is_12_l768_76849


namespace NUMINAMATH_CALUDE_job_completion_time_B_l768_76880

theorem job_completion_time_B (r_A r_B r_C : ℝ) : 
  (r_A + r_B = 1 / 3) →
  (r_B + r_C = 2 / 7) →
  (r_A + r_C = 1 / 4) →
  (1 / r_B = 168 / 31) :=
by sorry

end NUMINAMATH_CALUDE_job_completion_time_B_l768_76880


namespace NUMINAMATH_CALUDE_scout_saturday_hours_scout_saturday_hours_is_four_l768_76898

/-- Scout's delivery earnings over a weekend --/
theorem scout_saturday_hours : ℕ :=
  let base_pay : ℕ := 10  -- Base pay per hour in dollars
  let tip_per_customer : ℕ := 5  -- Tip per customer in dollars
  let saturday_customers : ℕ := 5  -- Number of customers on Saturday
  let sunday_hours : ℕ := 5  -- Hours worked on Sunday
  let sunday_customers : ℕ := 8  -- Number of customers on Sunday
  let total_earnings : ℕ := 155  -- Total earnings for the weekend in dollars

  let saturday_hours : ℕ := 
    (total_earnings - 
     (base_pay * sunday_hours + tip_per_customer * sunday_customers + 
      tip_per_customer * saturday_customers)) / base_pay

  saturday_hours

/-- Proof that Scout worked 4 hours on Saturday --/
theorem scout_saturday_hours_is_four : scout_saturday_hours = 4 := by
  sorry

end NUMINAMATH_CALUDE_scout_saturday_hours_scout_saturday_hours_is_four_l768_76898


namespace NUMINAMATH_CALUDE_product_abcd_is_zero_l768_76802

theorem product_abcd_is_zero 
  (a b c d : ℤ) 
  (eq1 : 2*a + 3*b + 5*c + 7*d = 34)
  (eq2 : 3*(d + c) = b)
  (eq3 : 3*b + c = a)
  (eq4 : c - 1 = d) :
  a * b * c * d = 0 :=
by sorry

end NUMINAMATH_CALUDE_product_abcd_is_zero_l768_76802


namespace NUMINAMATH_CALUDE_probability_red_ball_in_bag_l768_76846

/-- The probability of drawing a red ball from a bag -/
def probability_red_ball (total_balls : ℕ) (red_balls : ℕ) : ℚ :=
  red_balls / total_balls

/-- Theorem: The probability of drawing a red ball from an opaque bag with 5 balls, 2 of which are red, is 2/5 -/
theorem probability_red_ball_in_bag : probability_red_ball 5 2 = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_red_ball_in_bag_l768_76846


namespace NUMINAMATH_CALUDE_existence_of_x0_l768_76830

theorem existence_of_x0 (a b : ℝ) : ∃ x0 : ℝ, x0 ∈ Set.Icc 1 9 ∧ |a * x0 + b + 9 / x0| ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_x0_l768_76830


namespace NUMINAMATH_CALUDE_remainder_4523_div_32_l768_76819

theorem remainder_4523_div_32 : 4523 % 32 = 11 := by
  sorry

end NUMINAMATH_CALUDE_remainder_4523_div_32_l768_76819


namespace NUMINAMATH_CALUDE_wire_length_equals_49_l768_76801

/-- The total length of a wire cut into two pieces forming a square and a regular octagon -/
def wire_length (square_side : ℝ) : ℝ :=
  4 * square_side

theorem wire_length_equals_49 (square_side : ℝ) (h1 : square_side = 7) :
  let octagon_side := (3 * wire_length square_side) / (8 * 4)
  let square_area := square_side ^ 2
  let octagon_area := 2 * (1 + Real.sqrt 2) * octagon_side ^ 2
  square_area = octagon_area →
  wire_length square_side = 49 := by sorry

end NUMINAMATH_CALUDE_wire_length_equals_49_l768_76801


namespace NUMINAMATH_CALUDE_probability_two_red_balls_l768_76859

/-- The probability of picking two red balls from a bag containing 3 red, 4 blue, and 4 green balls. -/
theorem probability_two_red_balls (total_balls : ℕ) (red_balls : ℕ) (blue_balls : ℕ) (green_balls : ℕ)
  (h_total : total_balls = red_balls + blue_balls + green_balls)
  (h_red : red_balls = 3)
  (h_blue : blue_balls = 4)
  (h_green : green_balls = 4) :
  (red_balls : ℚ) / total_balls * ((red_balls - 1) : ℚ) / (total_balls - 1) = 3 / 55 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_red_balls_l768_76859


namespace NUMINAMATH_CALUDE_perpendicular_lines_k_values_l768_76869

theorem perpendicular_lines_k_values (k : ℝ) :
  let l1 : ℝ → ℝ → ℝ := λ x y => (k - 3) * x + (k + 4) * y + 1
  let l2 : ℝ → ℝ → ℝ := λ x y => (k + 1) * x + 2 * (k - 3) * y + 3
  (∀ x y, l1 x y = 0 → l2 x y = 0 → (k - 3) * (k + 1) + 2 * (k + 4) * (k - 3) = 0) →
  k = 3 ∨ k = -3 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_k_values_l768_76869


namespace NUMINAMATH_CALUDE_sin_seven_pi_sixths_l768_76847

theorem sin_seven_pi_sixths : Real.sin (7 * π / 6) = -1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_seven_pi_sixths_l768_76847


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_relation_l768_76845

def geometric_sequence (n : ℕ) : ℝ := 2^(n-1)

def sum_geometric_sequence (n : ℕ) : ℝ := 2^n - 1

theorem geometric_sequence_sum_relation (n : ℕ) :
  sum_geometric_sequence n = 2 * geometric_sequence n - 1 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_relation_l768_76845


namespace NUMINAMATH_CALUDE_problem_solution_l768_76860

-- Define the function f
def f (x m : ℝ) : ℝ := |x - 1| + |x + 3| - m

-- Define the theorem
theorem problem_solution :
  (∃ m : ℝ, ∀ x : ℝ, f x m < 5 ↔ -4 < x ∧ x < 2) →
  (∀ a b c : ℝ, a^2 + b^2/4 + c^2/9 = 1 → a + b + c ≤ Real.sqrt 14) :=
by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l768_76860


namespace NUMINAMATH_CALUDE_exp_13pi_over_3_rectangular_form_l768_76864

open Complex

theorem exp_13pi_over_3_rectangular_form :
  exp (13 * π * I / 3) = (1 / 2 : ℂ) + (I * (Real.sqrt 3 / 2)) := by
  sorry

end NUMINAMATH_CALUDE_exp_13pi_over_3_rectangular_form_l768_76864


namespace NUMINAMATH_CALUDE_reflection_of_circle_center_l768_76879

/-- Reflects a point about the line y = -x -/
def reflect_about_y_neg_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, -p.1)

theorem reflection_of_circle_center :
  let original_center : ℝ × ℝ := (9, -4)
  let reflected_center := reflect_about_y_neg_x original_center
  reflected_center = (4, -9) := by sorry

end NUMINAMATH_CALUDE_reflection_of_circle_center_l768_76879


namespace NUMINAMATH_CALUDE_point_above_with_distance_l768_76812

/-- Given two points P(3, a) and Q(3, 4) in a Cartesian coordinate system,
    if P is above Q and the distance between P and Q is 3,
    then the y-coordinate of P (which is a) equals 7. -/
theorem point_above_with_distance (a : ℝ) :
  a > 4 →  -- P is above Q
  (3 - 3)^2 + (a - 4)^2 = 3^2 →  -- Distance formula
  a = 7 := by
sorry

end NUMINAMATH_CALUDE_point_above_with_distance_l768_76812


namespace NUMINAMATH_CALUDE_fraction_equality_l768_76865

theorem fraction_equality (w z : ℝ) (h : (1/w + 1/z)/(1/w - 1/z) = 1008) :
  (w + z)/(w - z) = 1008 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l768_76865


namespace NUMINAMATH_CALUDE_chocolate_bar_cost_is_two_l768_76835

/-- The cost of a chocolate bar given Frank's purchase information -/
def chocolate_bar_cost (num_bars : ℕ) (num_chips : ℕ) (chips_cost : ℕ) (total_paid : ℕ) (change : ℕ) : ℕ :=
  (total_paid - change - num_chips * chips_cost) / num_bars

/-- Theorem stating that the cost of each chocolate bar is $2 -/
theorem chocolate_bar_cost_is_two :
  chocolate_bar_cost 5 2 3 20 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bar_cost_is_two_l768_76835


namespace NUMINAMATH_CALUDE_sin_tan_inequality_l768_76881

theorem sin_tan_inequality (α : Real) (h1 : 0 < α) (h2 : α < Real.pi / 2) :
  2 * Real.sin α + Real.tan α > 3 * α := by
  sorry

end NUMINAMATH_CALUDE_sin_tan_inequality_l768_76881


namespace NUMINAMATH_CALUDE_tangent_sum_l768_76892

theorem tangent_sum (x y : ℝ) 
  (h1 : (Real.sin x / Real.cos y) + (Real.sin y / Real.cos x) = 1)
  (h2 : (Real.cos x / Real.sin y) + (Real.cos y / Real.sin x) = 6) :
  (Real.tan x / Real.tan y) + (Real.tan y / Real.tan x) = 124/13 := by
  sorry

end NUMINAMATH_CALUDE_tangent_sum_l768_76892


namespace NUMINAMATH_CALUDE_value_of_s_l768_76895

theorem value_of_s (a b c w s p : ℕ) 
  (h1 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ w ≠ 0 ∧ s ≠ 0 ∧ p ≠ 0)
  (h2 : a ≠ b ∧ a ≠ c ∧ a ≠ w ∧ a ≠ s ∧ a ≠ p)
  (h3 : b ≠ c ∧ b ≠ w ∧ b ≠ s ∧ b ≠ p)
  (h4 : c ≠ w ∧ c ≠ s ∧ c ≠ p)
  (h5 : w ≠ s ∧ w ≠ p)
  (h6 : s ≠ p)
  (eq1 : a + b = w)
  (eq2 : w + c = s)
  (eq3 : s + a = p)
  (eq4 : b + c + p = 16) : 
  s = 8 := by
sorry

end NUMINAMATH_CALUDE_value_of_s_l768_76895


namespace NUMINAMATH_CALUDE_pages_to_read_tonight_l768_76836

theorem pages_to_read_tonight (total_pages : ℕ) (first_night : ℕ) : 
  total_pages = 100 → 
  first_night = 15 → 
  (total_pages - (first_night + 2 * first_night + (2 * first_night + 5))) = 20 := by
  sorry

end NUMINAMATH_CALUDE_pages_to_read_tonight_l768_76836


namespace NUMINAMATH_CALUDE_range_of_x_l768_76853

theorem range_of_x (x y : ℝ) (h : x - 6 * Real.sqrt y - 4 * Real.sqrt (x - y) + 12 = 0) :
  14 - 2 * Real.sqrt 13 ≤ x ∧ x ≤ 14 + 2 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_l768_76853


namespace NUMINAMATH_CALUDE_unique_occurrence_l768_76858

-- Define the sequence type
def IntegerSequence := ℕ → ℤ

-- Define the property of having infinitely many positive and negative elements
def HasInfinitelyManyPositiveAndNegative (a : IntegerSequence) : Prop :=
  (∀ N : ℕ, ∃ n > N, a n > 0) ∧ (∀ N : ℕ, ∃ n > N, a n < 0)

-- Define the property of distinct remainders
def HasDistinctRemainders (a : IntegerSequence) : Prop :=
  ∀ n : ℕ, ∀ i j : ℕ, i < n → j < n → i ≠ j → a i % n ≠ a j % n

-- The main theorem
theorem unique_occurrence (a : IntegerSequence) 
  (h1 : HasInfinitelyManyPositiveAndNegative a)
  (h2 : HasDistinctRemainders a)
  (k : ℤ) : 
  ∃! n : ℕ, a n = k :=
sorry

end NUMINAMATH_CALUDE_unique_occurrence_l768_76858


namespace NUMINAMATH_CALUDE_operations_result_l768_76856

-- Define operation S
def S (a b : ℤ) : ℤ := 4*a + 6*b

-- Define operation T
def T (a b : ℤ) : ℤ := 2*a - 3*b

-- Theorem statement
theorem operations_result : T (S 8 3) 4 = 88 := by
  sorry

end NUMINAMATH_CALUDE_operations_result_l768_76856


namespace NUMINAMATH_CALUDE_phi_value_is_65_degrees_l768_76878

-- Define the condition that φ is an acute angle
def is_acute_angle (φ : Real) : Prop := 0 < φ ∧ φ < Real.pi / 2

-- State the theorem
theorem phi_value_is_65_degrees :
  ∀ φ : Real,
  is_acute_angle φ →
  Real.sqrt 2 * Real.cos (20 * Real.pi / 180) = Real.sin φ - Real.cos φ →
  φ = 65 * Real.pi / 180 := by
sorry

end NUMINAMATH_CALUDE_phi_value_is_65_degrees_l768_76878


namespace NUMINAMATH_CALUDE_matrix_multiplication_result_l768_76884

theorem matrix_multiplication_result : 
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![3, -2; 5, 0]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![0, 6; -1, 2]
  A * B = !![2, 14; 0, 30] := by sorry

end NUMINAMATH_CALUDE_matrix_multiplication_result_l768_76884


namespace NUMINAMATH_CALUDE_quadratic_sum_of_b_and_c_l768_76862

/-- Given a quadratic expression x^2 - 24x + 50, prove that when written in the form (x+b)^2 + c, b + c = -106 -/
theorem quadratic_sum_of_b_and_c : ∃ b c : ℝ, 
  (∀ x : ℝ, x^2 - 24*x + 50 = (x + b)^2 + c) ∧ 
  (b + c = -106) := by
sorry

end NUMINAMATH_CALUDE_quadratic_sum_of_b_and_c_l768_76862


namespace NUMINAMATH_CALUDE_bid_probabilities_theorem_l768_76825

/-- Represents the probability of winning a bid for a project -/
structure BidProbability where
  value : ℝ
  is_probability : 0 ≤ value ∧ value ≤ 1

/-- Represents the probabilities of winning bids for three projects -/
structure ProjectProbabilities where
  a : BidProbability
  b : BidProbability
  c : BidProbability
  a_gt_b : a.value > b.value
  c_eq_quarter : c.value = 1/4

/-- The main theorem stating the properties of the bid probabilities -/
theorem bid_probabilities_theorem (p : ProjectProbabilities) : 
  p.a.value * p.b.value * p.c.value = 1/24 ∧
  1 - (1 - p.a.value) * (1 - p.b.value) * (1 - p.c.value) = 3/4 →
  p.a.value = 1/2 ∧ p.b.value = 1/3 ∧
  p.a.value * p.b.value * (1 - p.c.value) + 
  p.a.value * (1 - p.b.value) * p.c.value + 
  (1 - p.a.value) * p.b.value * p.c.value = 5/24 := by
  sorry

end NUMINAMATH_CALUDE_bid_probabilities_theorem_l768_76825


namespace NUMINAMATH_CALUDE_evaluate_expression_l768_76804

theorem evaluate_expression (x y : ℕ) (h1 : x = 3) (h2 : y = 4) :
  5 * x^y + 6 * y^x = 789 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l768_76804


namespace NUMINAMATH_CALUDE_proposition_analysis_l768_76886

theorem proposition_analysis :
  let converse := ∀ a b c : ℝ, a * c^2 > b * c^2 → a > b
  let negation := ∃ a b c : ℝ, a > b ∧ a * c^2 ≤ b * c^2
  let contrapositive := ∀ a b c : ℝ, a * c^2 ≤ b * c^2 → a ≤ b
  (converse ∧ negation ∧ ¬contrapositive) ∨
  (converse ∧ ¬negation ∧ contrapositive) ∨
  (¬converse ∧ negation ∧ contrapositive) :=
by sorry

#check proposition_analysis

end NUMINAMATH_CALUDE_proposition_analysis_l768_76886


namespace NUMINAMATH_CALUDE_number_operation_result_l768_76848

theorem number_operation_result : ∃ (x : ℝ), x = 295 ∧ (x / 5 + 6 = 65) := by sorry

end NUMINAMATH_CALUDE_number_operation_result_l768_76848


namespace NUMINAMATH_CALUDE_count_nonzero_monomials_l768_76809

/-- The number of monomials with non-zero coefficients in the expansion of (x+y+z)^2028 + (x-y-z)^2028 -/
def num_nonzero_monomials : ℕ := 1030225

/-- The exponent in the given expression -/
def exponent : ℕ := 2028

theorem count_nonzero_monomials :
  num_nonzero_monomials = (exponent / 2 + 1)^2 := by sorry

end NUMINAMATH_CALUDE_count_nonzero_monomials_l768_76809


namespace NUMINAMATH_CALUDE_one_root_l768_76829

/-- A quadratic function f(x) = x^2 + bx + c with discriminant 2020 -/
def f (b c : ℝ) (x : ℝ) : ℝ := x^2 + b*x + c

/-- The discriminant of f(x) = 0 is 2020 -/
axiom discriminant_is_2020 (b c : ℝ) : b^2 - 4*c = 2020

/-- The equation f(x - 2020) + f(x) = 0 has exactly one root -/
theorem one_root (b c : ℝ) : ∃! x, f b c (x - 2020) + f b c x = 0 :=
sorry

end NUMINAMATH_CALUDE_one_root_l768_76829


namespace NUMINAMATH_CALUDE_smallest_square_cover_l768_76816

/-- The width of the rectangle -/
def rectangle_width : ℕ := 3

/-- The height of the rectangle -/
def rectangle_height : ℕ := 4

/-- The area of a single rectangle -/
def rectangle_area : ℕ := rectangle_width * rectangle_height

/-- The side length of the smallest square that can be covered exactly by the rectangles -/
def square_side : ℕ := lcm rectangle_width rectangle_height

/-- The area of the square -/
def square_area : ℕ := square_side * square_side

/-- The number of rectangles needed to cover the square -/
def num_rectangles : ℕ := square_area / rectangle_area

theorem smallest_square_cover :
  num_rectangles = 12 ∧
  ∀ n : ℕ, n < square_side → ¬(n * n % rectangle_area = 0) :=
sorry

end NUMINAMATH_CALUDE_smallest_square_cover_l768_76816


namespace NUMINAMATH_CALUDE_fraction_division_proof_l768_76899

theorem fraction_division_proof : (5 / 4) / (8 / 15) = 75 / 32 := by
  sorry

end NUMINAMATH_CALUDE_fraction_division_proof_l768_76899


namespace NUMINAMATH_CALUDE_circular_film_diameter_l768_76875

theorem circular_film_diameter 
  (volume : ℝ) 
  (thickness : ℝ) 
  (π : ℝ) 
  (h1 : volume = 576) 
  (h2 : thickness = 0.2) 
  (h3 : π = Real.pi) : 
  let radius := Real.sqrt (volume / (thickness * π))
  2 * radius = 2 * Real.sqrt (2880 / π) :=
sorry

end NUMINAMATH_CALUDE_circular_film_diameter_l768_76875


namespace NUMINAMATH_CALUDE_steve_orange_count_l768_76827

/-- The number of oranges each person has -/
structure OrangeCount where
  marcie : ℝ
  brian : ℝ
  shawn : ℝ
  steve : ℝ

/-- The conditions of the orange distribution problem -/
def orange_problem (o : OrangeCount) : Prop :=
  o.marcie = 12 ∧
  o.brian = o.marcie ∧
  o.shawn = (o.marcie + o.brian) * 1.075 ∧
  o.steve = 3 * (o.marcie + o.brian + o.shawn)

/-- The theorem stating Steve's orange count -/
theorem steve_orange_count (o : OrangeCount) (h : orange_problem o) : o.steve = 149.4 := by
  sorry

end NUMINAMATH_CALUDE_steve_orange_count_l768_76827


namespace NUMINAMATH_CALUDE_apple_difference_l768_76807

theorem apple_difference (jackie_apples adam_apples : ℕ) 
  (h1 : jackie_apples = 10) (h2 : adam_apples = 8) : 
  jackie_apples - adam_apples = 2 := by
  sorry

end NUMINAMATH_CALUDE_apple_difference_l768_76807


namespace NUMINAMATH_CALUDE_pilot_miles_flown_l768_76896

theorem pilot_miles_flown (tuesday_miles : ℕ) (thursday_miles : ℕ) (total_weeks : ℕ) (total_miles : ℕ) : 
  thursday_miles = 1475 → 
  total_weeks = 3 → 
  total_miles = 7827 → 
  total_miles = total_weeks * (tuesday_miles + thursday_miles) → 
  tuesday_miles = 1134 := by
sorry

end NUMINAMATH_CALUDE_pilot_miles_flown_l768_76896


namespace NUMINAMATH_CALUDE_polynomial_fixed_point_l768_76877

theorem polynomial_fixed_point (P : ℤ → ℤ) (h_poly : ∃ (coeffs : List ℤ), ∀ x, P x = (coeffs.map (λ (c : ℤ) (i : ℕ) => c * x ^ i)).sum) :
  P 1 = 2013 → P 2013 = 1 → ∃ k : ℤ, P k = k → k = 1007 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_fixed_point_l768_76877


namespace NUMINAMATH_CALUDE_arabella_dance_steps_l768_76811

/-- Arabella's dance step learning problem -/
theorem arabella_dance_steps (T₁ T₂ T₃ : ℚ) 
  (h1 : T₁ = 30)
  (h2 : T₃ = T₁ + T₂)
  (h3 : T₁ + T₂ + T₃ = 90) :
  T₂ / T₁ = 1/2 := by
  sorry

#check arabella_dance_steps

end NUMINAMATH_CALUDE_arabella_dance_steps_l768_76811


namespace NUMINAMATH_CALUDE_triangle_side_and_area_l768_76838

theorem triangle_side_and_area 
  (a b c : ℝ) 
  (A : ℝ) 
  (h1 : a = Real.sqrt 7)
  (h2 : c = 3)
  (h3 : A = π / 3) :
  (b = 1 ∨ b = 2) ∧
  ((b = 1 → (1/2 * b * c * Real.sin A = (3 * Real.sqrt 3) / 4)) ∧
   (b = 2 → (1/2 * b * c * Real.sin A = (3 * Real.sqrt 3) / 2))) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_and_area_l768_76838


namespace NUMINAMATH_CALUDE_circle_equation_shortest_chord_line_l768_76832

-- Define the circle
def circle_center : ℝ × ℝ := (1, -2)
def tangent_line (x y : ℝ) : Prop := x + y - 1 = 0

-- Define point P
def point_P : ℝ × ℝ := (2, -2.5)

-- Function to check if a point is inside the circle
def is_inside_circle (p : ℝ × ℝ) : Prop := sorry

-- Theorem for the standard equation of the circle
theorem circle_equation (x y : ℝ) : 
  is_inside_circle point_P →
  ((x - circle_center.1)^2 + (y - circle_center.2)^2 = 2) ↔ 
  (∃ (p : ℝ × ℝ), p.1 = x ∧ p.2 = y ∧ (is_inside_circle p ∨ tangent_line p.1 p.2)) :=
sorry

-- Theorem for the equation of the line containing the shortest chord
theorem shortest_chord_line (x y : ℝ) :
  is_inside_circle point_P →
  (4*x - 2*y - 13 = 0) ↔ 
  (∃ (p q : ℝ × ℝ), 
    p ≠ q ∧ 
    is_inside_circle p ∧ 
    is_inside_circle q ∧ 
    p.1 = x ∧ p.2 = y ∧
    (q.1 - point_P.1) * (p.2 - point_P.2) = (q.2 - point_P.2) * (p.1 - point_P.1) ∧
    ∀ (r s : ℝ × ℝ), 
      r ≠ s → 
      is_inside_circle r → 
      is_inside_circle s → 
      (r.1 - point_P.1) * (s.2 - point_P.2) = (r.2 - point_P.2) * (s.1 - point_P.1) →
      (p.1 - q.1)^2 + (p.2 - q.2)^2 ≤ (r.1 - s.1)^2 + (r.2 - s.2)^2) :=
sorry

end NUMINAMATH_CALUDE_circle_equation_shortest_chord_line_l768_76832


namespace NUMINAMATH_CALUDE_cole_fence_cost_is_225_l768_76891

/-- Calculates the total cost for Cole's fence installation given the backyard dimensions,
    fencing costs, and neighbor contributions. -/
def cole_fence_cost (side_length : ℝ) (back_length : ℝ) (side_cost : ℝ) (back_cost : ℝ)
                    (back_neighbor_contribution : ℝ) (left_neighbor_contribution : ℝ)
                    (installation_fee : ℝ) : ℝ :=
  let total_fencing_cost := 2 * side_length * side_cost + back_length * back_cost
  let neighbor_contributions := back_neighbor_contribution + left_neighbor_contribution
  total_fencing_cost - neighbor_contributions + installation_fee

theorem cole_fence_cost_is_225 :
  cole_fence_cost 15 30 4 5 75 20 50 = 225 := by
  sorry

end NUMINAMATH_CALUDE_cole_fence_cost_is_225_l768_76891


namespace NUMINAMATH_CALUDE_round_robin_tournament_l768_76855

theorem round_robin_tournament (x : ℕ) : x > 0 → (x * (x - 1)) / 2 = 15 → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_round_robin_tournament_l768_76855


namespace NUMINAMATH_CALUDE_base_7_addition_l768_76887

/-- Addition in base 7 --/
def add_base_7 (a b : Nat) : Nat :=
  sorry

/-- Conversion from base 10 to base 7 --/
def to_base_7 (n : Nat) : Nat :=
  sorry

/-- Conversion from base 7 to base 10 --/
def from_base_7 (n : Nat) : Nat :=
  sorry

theorem base_7_addition :
  add_base_7 (from_base_7 25) (from_base_7 54) = from_base_7 112 :=
by sorry

end NUMINAMATH_CALUDE_base_7_addition_l768_76887


namespace NUMINAMATH_CALUDE_probability_one_red_two_blue_l768_76800

/-- The probability of selecting one red marble and two blue marbles from a bag -/
theorem probability_one_red_two_blue (total_marbles : ℕ) (red_marbles : ℕ) (blue_marbles : ℕ) 
  (h1 : total_marbles = red_marbles + blue_marbles)
  (h2 : red_marbles = 10)
  (h3 : blue_marbles = 6) : 
  (red_marbles * blue_marbles * (blue_marbles - 1) + 
   blue_marbles * red_marbles * (blue_marbles - 1) + 
   blue_marbles * (blue_marbles - 1) * red_marbles) / 
  (total_marbles * (total_marbles - 1) * (total_marbles - 2)) = 15 / 56 := by
  sorry

end NUMINAMATH_CALUDE_probability_one_red_two_blue_l768_76800


namespace NUMINAMATH_CALUDE_company_fund_distribution_l768_76842

/-- Represents the company fund distribution problem --/
theorem company_fund_distribution (n : ℕ) (initial_fund : ℕ) : 
  (75 * n = initial_fund + 15) →  -- Planned distribution
  (60 * n + 210 = initial_fund) →  -- Actual distribution
  initial_fund = 1110 := by
  sorry

end NUMINAMATH_CALUDE_company_fund_distribution_l768_76842


namespace NUMINAMATH_CALUDE_no_solution_implies_a_leq_two_l768_76824

theorem no_solution_implies_a_leq_two (a : ℝ) : 
  (∀ x : ℝ, ¬(x > 1 ∧ x < a - 1)) → a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_implies_a_leq_two_l768_76824


namespace NUMINAMATH_CALUDE_gcf_360_150_l768_76823

theorem gcf_360_150 : Nat.gcd 360 150 = 30 := by
  sorry

end NUMINAMATH_CALUDE_gcf_360_150_l768_76823


namespace NUMINAMATH_CALUDE_parallel_line_difference_l768_76808

/-- Given two points (-1, q) and (-3, r) on a line parallel to y = (3/2)x + 1, 
    prove that r - q = -3 -/
theorem parallel_line_difference (q r : ℝ) : 
  (∃ (m b : ℝ), m = 3/2 ∧ 
    (∀ (x y : ℝ), y = m * x + b ↔ (x = -1 ∧ y = q) ∨ (x = -3 ∧ y = r))) →
  r - q = -3 :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_difference_l768_76808


namespace NUMINAMATH_CALUDE_greatest_integer_with_gcd_six_gcd_192_18_is_six_less_than_200_exists_no_greater_main_result_l768_76894

theorem greatest_integer_with_gcd_six (n : ℕ) : n < 200 ∧ Nat.gcd n 18 = 6 → n ≤ 192 :=
by sorry

theorem gcd_192_18_is_six : Nat.gcd 192 18 = 6 :=
by sorry

theorem less_than_200 : 192 < 200 :=
by sorry

theorem exists_no_greater : ¬∃ m : ℕ, 192 < m ∧ m < 200 ∧ Nat.gcd m 18 = 6 :=
by sorry

theorem main_result : 
  (∃ n : ℕ, n < 200 ∧ Nat.gcd n 18 = 6) ∧ 
  (∀ n : ℕ, n < 200 ∧ Nat.gcd n 18 = 6 → n ≤ 192) ∧
  (Nat.gcd 192 18 = 6) ∧
  (192 < 200) :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_with_gcd_six_gcd_192_18_is_six_less_than_200_exists_no_greater_main_result_l768_76894


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l768_76844

theorem completing_square_equivalence (x : ℝ) : 
  x^2 - 2*x = 2 ↔ (x - 1)^2 = 3 := by sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l768_76844


namespace NUMINAMATH_CALUDE_chicken_count_l768_76826

/-- The number of chickens in the coop -/
def coop_chickens : ℕ := 14

/-- The number of chickens in the run -/
def run_chickens : ℕ := 2 * coop_chickens

/-- The total number of chickens in the coop and run -/
def total_coop_run : ℕ := coop_chickens + run_chickens

/-- The number of free-ranging chickens -/
def free_range_chickens : ℕ := 105

theorem chicken_count : 
  (2 : ℚ) / 5 = (total_coop_run : ℚ) / free_range_chickens := by
  sorry

end NUMINAMATH_CALUDE_chicken_count_l768_76826


namespace NUMINAMATH_CALUDE_initial_customer_count_l768_76840

/-- Represents the number of customers at different times -/
structure CustomerCount where
  initial : ℕ
  after_first_hour : ℕ
  after_second_hour : ℕ

/-- Calculates the number of customers after the first hour -/
def first_hour_change (c : CustomerCount) : ℕ := c.initial + 7 - 4

/-- Calculates the number of customers after the second hour -/
def second_hour_change (c : CustomerCount) : ℕ := c.after_first_hour + 3 - 9

/-- The main theorem stating the initial number of customers -/
theorem initial_customer_count : ∃ (c : CustomerCount), 
  c.initial = 15 ∧ 
  c.after_first_hour = first_hour_change c ∧
  c.after_second_hour = second_hour_change c ∧
  c.after_second_hour = 12 := by
  sorry

end NUMINAMATH_CALUDE_initial_customer_count_l768_76840


namespace NUMINAMATH_CALUDE_complex_equation_solution_l768_76876

theorem complex_equation_solution (z : ℂ) :
  (3 - 4 * Complex.I) * z = 5 → z = 3/5 + 4/5 * Complex.I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l768_76876


namespace NUMINAMATH_CALUDE_sin_cos_value_l768_76889

def determinant (a b c d : ℝ) : ℝ := a * d - b * c

theorem sin_cos_value (θ : ℝ) :
  determinant (Real.sin θ) 2 (Real.cos θ) 3 = 0 →
  2 * (Real.sin θ)^2 + (Real.sin θ) * (Real.cos θ) = 14/13 :=
by
  sorry

end NUMINAMATH_CALUDE_sin_cos_value_l768_76889


namespace NUMINAMATH_CALUDE_rectangle_triangle_configuration_l768_76861

theorem rectangle_triangle_configuration (AB AD : ℝ) (h1 : AB = 8) (h2 : AD = 10) : ∃ (DE : ℝ),
  let ABCD_area := AB * AD
  let DCE_area := ABCD_area / 2
  let DC := AD
  let CE := 2 * DCE_area / DC
  DE^2 = DC^2 + CE^2 ∧ DE = 2 * Real.sqrt 41 := by sorry

end NUMINAMATH_CALUDE_rectangle_triangle_configuration_l768_76861


namespace NUMINAMATH_CALUDE_cubic_expression_evaluation_l768_76871

theorem cubic_expression_evaluation : 
  3000^3 - 2999 * 3000^2 - 2999^2 * 3000 + 2999^3 = 26991001 := by
sorry

end NUMINAMATH_CALUDE_cubic_expression_evaluation_l768_76871


namespace NUMINAMATH_CALUDE_star_properties_l768_76850

/-- The star operation -/
def star (a b : ℝ) : ℝ := a + b + a * b

/-- The prime operation -/
noncomputable def prime (a : ℝ) : ℝ := -a / (a + 1)

theorem star_properties (a b : ℝ) (ha : a ≠ -1) (hb : b ≠ -1) :
  (prime (prime a) = a) ∧
  (prime (star a b) = star (prime a) (prime b)) ∧
  (prime (star (prime a) b) = star (prime a) (prime b)) ∧
  (prime (star (prime a) (prime b)) = star a b) :=
by sorry

end NUMINAMATH_CALUDE_star_properties_l768_76850


namespace NUMINAMATH_CALUDE_professor_newtons_students_l768_76806

theorem professor_newtons_students (N M : ℕ) : 
  N % 4 = 2 →
  N % 5 = 1 →
  N = M + 15 →
  M < 15 →
  N = 26 ∧ M = 11 := by
sorry

end NUMINAMATH_CALUDE_professor_newtons_students_l768_76806


namespace NUMINAMATH_CALUDE_fixed_point_theorem_l768_76822

/-- The parabola y^2 = 2px where p > 0 -/
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

/-- The moving line y = kx + b where k ≠ 0 and b ≠ 0 -/
def movingLine (k b x y : ℝ) : Prop := y = k*x + b ∧ k ≠ 0 ∧ b ≠ 0

/-- The slopes of OA and OB multiply to √3 -/
def slopeProduct (k₁ k₂ : ℝ) : Prop := k₁ * k₂ = Real.sqrt 3

/-- The theorem stating that the line always passes through a fixed point -/
theorem fixed_point_theorem (p k b : ℝ) :
  (∃ x₁ y₁ x₂ y₂ k₁ k₂,
    parabola p x₁ y₁ ∧ parabola p x₂ y₂ ∧
    movingLine k b x₁ y₁ ∧ movingLine k b x₂ y₂ ∧
    slopeProduct k₁ k₂) →
  movingLine k b (-2 * Real.sqrt 3 * p / 3) 0 :=
sorry

end NUMINAMATH_CALUDE_fixed_point_theorem_l768_76822


namespace NUMINAMATH_CALUDE_find_n_l768_76813

theorem find_n : ∃ n : ℤ, 5^2 - 7 = 3^3 + n ∧ n = -9 := by sorry

end NUMINAMATH_CALUDE_find_n_l768_76813


namespace NUMINAMATH_CALUDE_simplify_fraction_l768_76841

theorem simplify_fraction : 9 * (12 / 7) * (-35 / 36) = -15 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l768_76841


namespace NUMINAMATH_CALUDE_plant_branches_l768_76897

theorem plant_branches (x : ℕ) (h : 1 + x + x^2 = 31) : x = 5 := by
  sorry

end NUMINAMATH_CALUDE_plant_branches_l768_76897


namespace NUMINAMATH_CALUDE_unique_polynomial_solution_l768_76870

/-- A polynomial P(x) that satisfies P(P(x)) = (x^2 + x + 1) P(x) -/
def P (x : ℝ) : ℝ := x^2 + x

/-- Theorem stating that P(x) = x^2 + x is the unique nonconstant polynomial solution 
    to the equation P(P(x)) = (x^2 + x + 1) P(x) -/
theorem unique_polynomial_solution :
  (∀ x, P (P x) = (x^2 + x + 1) * P x) ∧
  (∀ Q : ℝ → ℝ, (∀ x, Q (Q x) = (x^2 + x + 1) * Q x) → 
    (∃ a b c, ∀ x, Q x = a * x^2 + b * x + c) →
    (∃ x y, Q x ≠ Q y) →
    (∀ x, Q x = P x)) :=
by sorry


end NUMINAMATH_CALUDE_unique_polynomial_solution_l768_76870


namespace NUMINAMATH_CALUDE_unique_digit_multiple_6_and_9_l768_76805

def is_multiple_of_6_and_9 (n : ℕ) : Prop :=
  n % 6 = 0 ∧ n % 9 = 0

def five_digit_number (d : ℕ) : ℕ :=
  74820 + d

theorem unique_digit_multiple_6_and_9 :
  ∃! d : ℕ, d < 10 ∧ is_multiple_of_6_and_9 (five_digit_number d) :=
by sorry

end NUMINAMATH_CALUDE_unique_digit_multiple_6_and_9_l768_76805


namespace NUMINAMATH_CALUDE_bowling_team_average_weight_l768_76837

theorem bowling_team_average_weight 
  (initial_players : ℕ) 
  (initial_average : ℝ) 
  (new_player1_weight : ℝ) 
  (new_player2_weight : ℝ) 
  (h1 : initial_players = 7) 
  (h2 : initial_average = 94) 
  (h3 : new_player1_weight = 110) 
  (h4 : new_player2_weight = 60) :
  let total_weight := initial_players * initial_average + new_player1_weight + new_player2_weight
  let new_players := initial_players + 2
  (total_weight / new_players : ℝ) = 92 := by
sorry

end NUMINAMATH_CALUDE_bowling_team_average_weight_l768_76837


namespace NUMINAMATH_CALUDE_sand_pile_base_area_l768_76834

/-- Given a rectangular compartment of sand and a conical pile, this theorem proves
    that the base area of the pile is 81/2 square meters. -/
theorem sand_pile_base_area
  (length width height : ℝ)
  (pile_height : ℝ)
  (h_length : length = 6)
  (h_width : width = 1.5)
  (h_height : height = 3)
  (h_pile_height : pile_height = 2)
  (h_volume_conservation : length * width * height = (1/3) * Real.pi * (pile_base_area / Real.pi) * pile_height)
  : pile_base_area = 81/2 := by
  sorry

end NUMINAMATH_CALUDE_sand_pile_base_area_l768_76834


namespace NUMINAMATH_CALUDE_pattern_B_cannot_fold_into_tetrahedron_l768_76868

-- Define the structure of a pattern
structure Pattern :=
  (squares : ℕ)
  (foldLines : ℕ)

-- Define the properties of a regular tetrahedron
structure RegularTetrahedron :=
  (faces : ℕ)
  (edges : ℕ)
  (vertices : ℕ)
  (edgesPerVertex : ℕ)

-- Define the folding function (noncomputable as it's conceptual)
noncomputable def canFoldIntoTetrahedron (p : Pattern) : Prop := sorry

-- Define the specific patterns
def patternA : Pattern := ⟨4, 3⟩
def patternB : Pattern := ⟨4, 3⟩
def patternC : Pattern := ⟨4, 3⟩
def patternD : Pattern := ⟨4, 3⟩

-- Define the properties of a regular tetrahedron
def tetrahedron : RegularTetrahedron := ⟨4, 6, 4, 3⟩

-- State the theorem
theorem pattern_B_cannot_fold_into_tetrahedron :
  ¬(canFoldIntoTetrahedron patternB) :=
sorry

end NUMINAMATH_CALUDE_pattern_B_cannot_fold_into_tetrahedron_l768_76868


namespace NUMINAMATH_CALUDE_june_election_win_l768_76854

/-- The minimum percentage of boys required for June to win the election -/
def min_boys_percentage : ℝ :=
  -- We'll define this later in the proof
  sorry

theorem june_election_win (total_students : ℕ) (boys_vote_percentage : ℝ) (girls_vote_percentage : ℝ) 
  (h_total : total_students = 200)
  (h_boys_vote : boys_vote_percentage = 67.5)
  (h_girls_vote : girls_vote_percentage = 25)
  (h_win_threshold : ∀ x : ℝ, x > 50 → x ≥ (total_students : ℝ) / 2 + 0.5) :
  ∃ ε > 0, abs (min_boys_percentage - 60) < ε ∧ 
  ∀ boys_percentage : ℝ, boys_percentage ≥ min_boys_percentage →
    (boys_percentage * boys_vote_percentage + (100 - boys_percentage) * girls_vote_percentage) / 100 > 50 :=
by sorry

end NUMINAMATH_CALUDE_june_election_win_l768_76854


namespace NUMINAMATH_CALUDE_system_solution_l768_76852

theorem system_solution (x y z : ℝ) 
  (eq1 : x + 3*y = 20)
  (eq2 : x + y + z = 25)
  (eq3 : x - z = 5) :
  x = 14 ∧ y = 2 ∧ z = 9 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l768_76852


namespace NUMINAMATH_CALUDE_john_biking_distance_l768_76818

/-- Converts a base-7 number to base-10 --/
def base7ToBase10 (d₃ d₂ d₁ d₀ : ℕ) : ℕ :=
  d₃ * 7^3 + d₂ * 7^2 + d₁ * 7^1 + d₀ * 7^0

/-- The problem statement --/
theorem john_biking_distance :
  base7ToBase10 3 9 5 6 = 1511 := by
  sorry

end NUMINAMATH_CALUDE_john_biking_distance_l768_76818


namespace NUMINAMATH_CALUDE_race_result_l768_76833

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ
  (speed_pos : 0 < speed)

/-- The race setup -/
structure Race where
  anton : Runner
  seryozha : Runner
  tolya : Runner
  (different_speeds : anton.speed ≠ seryozha.speed ∧ seryozha.speed ≠ tolya.speed ∧ anton.speed ≠ tolya.speed)

/-- The race conditions -/
def race_conditions (r : Race) : Prop :=
  let t_anton := 100 / r.anton.speed
  let d_seryozha := r.seryozha.speed * t_anton
  let t_seryozha := 100 / r.seryozha.speed
  let d_tolya := r.tolya.speed * t_seryozha
  d_seryozha = 90 ∧ d_tolya = 90

theorem race_result (r : Race) (h : race_conditions r) :
  r.tolya.speed * (100 / r.anton.speed) = 81 := by
  sorry

#check race_result

end NUMINAMATH_CALUDE_race_result_l768_76833


namespace NUMINAMATH_CALUDE_carol_owns_twice_as_many_as_cathy_l768_76839

/-- Represents the number of cars owned by each person -/
structure CarOwnership where
  cathy : ℕ
  lindsey : ℕ
  susan : ℕ
  carol : ℕ

/-- The conditions of the car ownership problem -/
def carProblemConditions (o : CarOwnership) : Prop :=
  o.lindsey = o.cathy + 4 ∧
  o.susan = o.carol - 2 ∧
  o.carol = 2 * o.cathy ∧
  o.cathy + o.lindsey + o.susan + o.carol = 32 ∧
  o.cathy = 5

theorem carol_owns_twice_as_many_as_cathy (o : CarOwnership) 
  (h : carProblemConditions o) : o.carol = 2 * o.cathy := by
  sorry

#check carol_owns_twice_as_many_as_cathy

end NUMINAMATH_CALUDE_carol_owns_twice_as_many_as_cathy_l768_76839


namespace NUMINAMATH_CALUDE_min_slope_tangent_line_l768_76843

/-- The function f(x) = x^3 + 3x^2 + 6x - 10 -/
def f (x : ℝ) : ℝ := x^3 + 3*x^2 + 6*x - 10

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3*x^2 + 6*x + 6

theorem min_slope_tangent_line :
  ∃ (m : ℝ), m = 3 ∧ ∀ (x : ℝ), f' x ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_slope_tangent_line_l768_76843


namespace NUMINAMATH_CALUDE_odd_function_condition_l768_76817

/-- A function f is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The function f(x) = 2x^3 + ax^2 + b - 1 -/
def f (a b : ℝ) (x : ℝ) : ℝ :=
  2 * x^3 + a * x^2 + b - 1

/-- If f(x) = 2x^3 + ax^2 + b - 1 is an odd function, then a - b = -1 -/
theorem odd_function_condition (a b : ℝ) :
  IsOdd (f a b) → a - b = -1 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_condition_l768_76817


namespace NUMINAMATH_CALUDE_chickens_and_rabbits_equation_l768_76820

/-- Represents the number of chickens in the cage -/
def chickens : ℕ := sorry

/-- Represents the number of rabbits in the cage -/
def rabbits : ℕ := sorry

/-- The total number of heads in the cage -/
def total_heads : ℕ := 16

/-- The total number of feet in the cage -/
def total_feet : ℕ := 44

/-- Theorem stating that the system of equations correctly represents the problem -/
theorem chickens_and_rabbits_equation :
  (chickens + rabbits = total_heads) ∧
  (2 * chickens + 4 * rabbits = total_feet) :=
sorry

end NUMINAMATH_CALUDE_chickens_and_rabbits_equation_l768_76820


namespace NUMINAMATH_CALUDE_unique_integer_solution_l768_76831

theorem unique_integer_solution (x y z : ℤ) :
  x^2 + y^2 + z^2 = x^2 * y^2 → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_solution_l768_76831


namespace NUMINAMATH_CALUDE_triangle_inequality_l768_76821

theorem triangle_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) :
  (a + b) * Real.sqrt (a * b) + (a + c) * Real.sqrt (a * c) + (b + c) * Real.sqrt (b * c) ≥ (a + b + c)^2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l768_76821


namespace NUMINAMATH_CALUDE_unique_k_for_integer_roots_l768_76885

/-- The polynomial f(x) parameterized by k -/
def f (k : ℤ) (x : ℝ) : ℝ := x^3 - (k-3)*x^2 - 11*x + (4*k-8)

/-- A root of f is an x such that f(x) = 0 -/
def is_root (k : ℤ) (x : ℝ) : Prop := f k x = 0

/-- All roots of f are integers -/
def all_roots_integer (k : ℤ) : Prop :=
  ∀ x : ℝ, is_root k x → ∃ n : ℤ, x = n

/-- The main theorem: k = 5 is the only integer for which all roots of f are integers -/
theorem unique_k_for_integer_roots :
  ∃! k : ℤ, all_roots_integer k ∧ k = 5 := by sorry

end NUMINAMATH_CALUDE_unique_k_for_integer_roots_l768_76885


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_fraction_reciprocal_of_negative_one_thirteenth_l768_76873

theorem reciprocal_of_negative_fraction (a b : ℤ) (hb : b ≠ 0) :
  ((-1 : ℚ) / (a : ℚ) / (b : ℚ))⁻¹ = -((b : ℚ) / (a : ℚ)) :=
by sorry

theorem reciprocal_of_negative_one_thirteenth :
  ((-1 : ℚ) / 13)⁻¹ = -13 :=
by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_fraction_reciprocal_of_negative_one_thirteenth_l768_76873


namespace NUMINAMATH_CALUDE_cricket_team_throwers_l768_76815

theorem cricket_team_throwers 
  (total_players : ℕ) 
  (total_right_handed : ℕ) 
  (throwers : ℕ) 
  (left_handed : ℕ) 
  (right_handed : ℕ) :
  total_players = 55 →
  total_right_handed = 49 →
  throwers + left_handed + right_handed = total_players →
  throwers + right_handed = total_right_handed →
  left_handed = (total_players - throwers) / 3 →
  throwers = 37 := by
  sorry

#check cricket_team_throwers

end NUMINAMATH_CALUDE_cricket_team_throwers_l768_76815


namespace NUMINAMATH_CALUDE_ceiling_minus_y_l768_76803

theorem ceiling_minus_y (x : ℝ) : 
  let y := 2 * x
  let f := y - ⌊y⌋
  (⌈y⌉ - ⌊y⌋ = 1) → (0 < f ∧ f < 1) → (⌈y⌉ - y = 1 - f) :=
by sorry

end NUMINAMATH_CALUDE_ceiling_minus_y_l768_76803


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l768_76810

theorem polynomial_division_remainder :
  let f (x : ℝ) := x^6 - 2*x^5 + x^4 - x^2 + 3*x - 1
  let g (x : ℝ) := (x^2 - 1)*(x + 2)
  let r (x : ℝ) := 7/3*x^2 + x - 7/3
  ∃ q : ℝ → ℝ, ∀ x, f x = g x * q x + r x :=
by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l768_76810


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l768_76851

theorem geometric_sequence_sum (a : ℚ) (r : ℚ) (n : ℕ) (h1 : a = 1/4) (h2 : r = 1/4) (h3 : n = 6) :
  a * (1 - r^n) / (1 - r) = 1365/4096 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l768_76851


namespace NUMINAMATH_CALUDE_pick_school_supply_l768_76882

/-- The number of pencils in the pencil case -/
def num_pencils : ℕ := 2

/-- The number of erasers in the pencil case -/
def num_erasers : ℕ := 4

/-- The total number of school supplies in the pencil case -/
def total_supplies : ℕ := num_pencils + num_erasers

/-- Theorem stating that the number of ways to pick up a school supply is 6 -/
theorem pick_school_supply : total_supplies = 6 := by
  sorry

end NUMINAMATH_CALUDE_pick_school_supply_l768_76882


namespace NUMINAMATH_CALUDE_min_lines_to_cover_plane_l768_76828

-- Define the circle on a plane
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

-- Define a line on a plane
structure Line :=
  (a b c : ℝ)

-- Define a reflection of a point with respect to a line
def reflect (p : ℝ × ℝ) (l : Line) : ℝ × ℝ := sorry

-- Define a function to check if a point is covered by a circle
def is_covered (p : ℝ × ℝ) (c : Circle) : Prop := sorry

-- Define a function to perform a finite sequence of reflections
def reflect_sequence (c : Circle) (lines : List Line) : Circle := sorry

-- Theorem statement
theorem min_lines_to_cover_plane (c : Circle) :
  ∃ (lines : List Line),
    (lines.length = 3) ∧
    (∀ (p : ℝ × ℝ), ∃ (seq : List Line),
      (∀ (l : Line), l ∈ seq → l ∈ lines) ∧
      is_covered p (reflect_sequence c seq)) ∧
    (∀ (lines' : List Line),
      lines'.length < 3 →
      ∃ (p : ℝ × ℝ), ∀ (seq : List Line),
        (∀ (l : Line), l ∈ seq → l ∈ lines') →
        ¬is_covered p (reflect_sequence c seq)) :=
sorry

end NUMINAMATH_CALUDE_min_lines_to_cover_plane_l768_76828


namespace NUMINAMATH_CALUDE_inequality_proof_l768_76874

theorem inequality_proof (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) :
  (1 + y * z) / (1 + x^2) + (1 + z * x) / (1 + y^2) + (1 + x * y) / (1 + z^2) ≥ 2 ∧
  ((1 + y * z) / (1 + x^2) + (1 + z * x) / (1 + y^2) + (1 + x * y) / (1 + z^2) = 2 ↔ x = 0 ∧ y = 0 ∧ z = 0) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l768_76874
