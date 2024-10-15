import Mathlib

namespace NUMINAMATH_CALUDE_stability_comparison_l4006_400622

-- Define the Student type
structure Student where
  name : String
  average_score : ℝ
  variance : ℝ

-- Define the concept of stability
def more_stable (a b : Student) : Prop :=
  a.variance < b.variance

-- Theorem statement
theorem stability_comparison (A B : Student) 
  (h1 : A.average_score = B.average_score)
  (h2 : A.variance > B.variance) : 
  more_stable B A := by
  sorry

end NUMINAMATH_CALUDE_stability_comparison_l4006_400622


namespace NUMINAMATH_CALUDE_man_money_calculation_l4006_400609

/-- Calculates the total amount of money given the number of Rs. 50 and Rs. 500 notes -/
def totalAmount (fiftyNotes : ℕ) (fiveHundredNotes : ℕ) : ℕ :=
  50 * fiftyNotes + 500 * fiveHundredNotes

theorem man_money_calculation (totalNotes : ℕ) (fiftyNotes : ℕ) 
  (h1 : totalNotes = 126)
  (h2 : fiftyNotes = 117)
  (h3 : totalNotes = fiftyNotes + (totalNotes - fiftyNotes)) :
  totalAmount fiftyNotes (totalNotes - fiftyNotes) = 10350 := by
  sorry

#eval totalAmount 117 9

end NUMINAMATH_CALUDE_man_money_calculation_l4006_400609


namespace NUMINAMATH_CALUDE_greatest_k_value_l4006_400623

theorem greatest_k_value (k : ℝ) : 
  (∃ x y : ℝ, x^2 + k*x + 8 = 0 ∧ y^2 + k*y + 8 = 0 ∧ |x - y| = Real.sqrt 73) →
  k ≤ Real.sqrt 105 :=
sorry

end NUMINAMATH_CALUDE_greatest_k_value_l4006_400623


namespace NUMINAMATH_CALUDE_x_value_l4006_400654

theorem x_value : ∃ x : ℝ, x = 12 * (1 + 0.2) ∧ x = 14.4 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l4006_400654


namespace NUMINAMATH_CALUDE_unique_solution_condition_l4006_400670

theorem unique_solution_condition (p q : ℝ) :
  (∃! x : ℝ, 4 * x - 7 + p = q * x + 2) ↔ q ≠ 4 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l4006_400670


namespace NUMINAMATH_CALUDE_hilton_marbles_l4006_400674

/-- Calculates the final number of marbles Hilton has --/
def final_marbles (initial : ℕ) (found : ℕ) (lost : ℕ) : ℕ :=
  initial + found - lost + 2 * lost

/-- Proves that Hilton ends up with 42 marbles given the initial conditions --/
theorem hilton_marbles : final_marbles 26 6 10 = 42 := by
  sorry

end NUMINAMATH_CALUDE_hilton_marbles_l4006_400674


namespace NUMINAMATH_CALUDE_exists_expression_for_100_l4006_400633

/-- A type representing arithmetic expressions using only the number 7 --/
inductive Expr
  | seven : Expr
  | add : Expr → Expr → Expr
  | sub : Expr → Expr → Expr
  | mul : Expr → Expr → Expr
  | div : Expr → Expr → Expr

/-- Evaluate an expression to a rational number --/
def eval : Expr → ℚ
  | Expr.seven => 7
  | Expr.add e₁ e₂ => eval e₁ + eval e₂
  | Expr.sub e₁ e₂ => eval e₁ - eval e₂
  | Expr.mul e₁ e₂ => eval e₁ * eval e₂
  | Expr.div e₁ e₂ => eval e₁ / eval e₂

/-- Count the number of sevens in an expression --/
def countSevens : Expr → ℕ
  | Expr.seven => 1
  | Expr.add e₁ e₂ => countSevens e₁ + countSevens e₂
  | Expr.sub e₁ e₂ => countSevens e₁ + countSevens e₂
  | Expr.mul e₁ e₂ => countSevens e₁ + countSevens e₂
  | Expr.div e₁ e₂ => countSevens e₁ + countSevens e₂

/-- There exists an expression using fewer than 10 sevens that evaluates to 100 --/
theorem exists_expression_for_100 : ∃ e : Expr, eval e = 100 ∧ countSevens e < 10 := by
  sorry

end NUMINAMATH_CALUDE_exists_expression_for_100_l4006_400633


namespace NUMINAMATH_CALUDE_quadratic_sum_of_squares_l4006_400698

theorem quadratic_sum_of_squares (a b c : ℝ) (ha : a ≠ b) (hb : b ≠ c) (hc : c ≠ a) :
  (∃ (x y z : ℝ),
    (x^2 + a*x + b = 0 ∧ y^2 + b*y + c = 0 ∧ x = y) ∧
    (y^2 + b*y + c = 0 ∧ z^2 + c*z + a = 0 ∧ y = z) ∧
    (z^2 + c*z + a = 0 ∧ x^2 + a*x + b = 0 ∧ z = x)) →
  a^2 + b^2 + c^2 = 6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_sum_of_squares_l4006_400698


namespace NUMINAMATH_CALUDE_asha_win_probability_l4006_400681

theorem asha_win_probability (lose_prob tie_prob : ℚ) 
  (lose_eq : lose_prob = 3 / 7)
  (tie_eq : tie_prob = 1 / 7)
  (total_prob : lose_prob + tie_prob + (1 - lose_prob - tie_prob) = 1) :
  1 - lose_prob - tie_prob = 3 / 7 := by
sorry

end NUMINAMATH_CALUDE_asha_win_probability_l4006_400681


namespace NUMINAMATH_CALUDE_simplified_expression_value_l4006_400641

theorem simplified_expression_value (a b : ℚ) 
  (h1 : a = -1) 
  (h2 : b = 1/4) : 
  (a + 2*b)^2 + (a + 2*b)*(a - 2*b) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplified_expression_value_l4006_400641


namespace NUMINAMATH_CALUDE_fare_ratio_proof_l4006_400697

theorem fare_ratio_proof (passenger_ratio : ℚ) (total_amount : ℕ) (second_class_amount : ℕ) :
  passenger_ratio = 1 / 50 →
  total_amount = 1325 →
  second_class_amount = 1250 →
  ∃ (first_class_fare second_class_fare : ℕ),
    first_class_fare / second_class_fare = 3 :=
by sorry

end NUMINAMATH_CALUDE_fare_ratio_proof_l4006_400697


namespace NUMINAMATH_CALUDE_traditionalist_fraction_l4006_400680

theorem traditionalist_fraction (num_provinces : ℕ) (num_traditionalists_per_province : ℚ) (num_progressives : ℚ) :
  num_provinces = 15 →
  num_traditionalists_per_province = num_progressives / 20 →
  (num_provinces : ℚ) * num_traditionalists_per_province / ((num_provinces : ℚ) * num_traditionalists_per_province + num_progressives) = 3 / 7 := by
  sorry

end NUMINAMATH_CALUDE_traditionalist_fraction_l4006_400680


namespace NUMINAMATH_CALUDE_square_area_from_perimeter_l4006_400651

theorem square_area_from_perimeter (perimeter : ℝ) (area : ℝ) : 
  perimeter = 48 → area = (perimeter / 4)^2 → area = 144 := by
  sorry

end NUMINAMATH_CALUDE_square_area_from_perimeter_l4006_400651


namespace NUMINAMATH_CALUDE_tan_beta_value_l4006_400667

theorem tan_beta_value (α β : Real) 
  (h1 : Real.tan α = 1/3) 
  (h2 : Real.tan (α + β) = 1/2) : 
  Real.tan β = 1/7 := by sorry

end NUMINAMATH_CALUDE_tan_beta_value_l4006_400667


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l4006_400613

theorem complex_fraction_equality : (1 - Complex.I) / (1 + Complex.I) = -Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l4006_400613


namespace NUMINAMATH_CALUDE_m_plus_n_values_l4006_400645

theorem m_plus_n_values (m n : ℤ) (hm : |m| = 4) (hn : |n| = 5) (hn_neg : n < 0) :
  m + n = -1 ∨ m + n = -9 := by
  sorry

end NUMINAMATH_CALUDE_m_plus_n_values_l4006_400645


namespace NUMINAMATH_CALUDE_ordering_abc_l4006_400692

theorem ordering_abc (a b c : ℝ) : 
  a = 7/9 → b = 0.7 * Real.exp 0.1 → c = Real.cos (2/3) → c > a ∧ a > b :=
sorry

end NUMINAMATH_CALUDE_ordering_abc_l4006_400692


namespace NUMINAMATH_CALUDE_max_sqrt_sum_l4006_400615

theorem max_sqrt_sum (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hsum : x + y = 18) :
  ∃ (d : ℝ), d = 6 ∧ ∀ (a b : ℝ), a ≥ 0 → b ≥ 0 → a + b = 18 → Real.sqrt a + Real.sqrt b ≤ d :=
by sorry

end NUMINAMATH_CALUDE_max_sqrt_sum_l4006_400615


namespace NUMINAMATH_CALUDE_square_on_hypotenuse_side_length_l4006_400642

/-- Given a right triangle PQR with leg PR = 9 and leg PQ = 12, 
    prove that a square with one side along the hypotenuse and 
    one vertex each on legs PR and PQ has a side length of 5 5/7 -/
theorem square_on_hypotenuse_side_length 
  (P Q R : ℝ × ℝ) 
  (right_angle : (P.1 - Q.1) * (P.1 - R.1) + (P.2 - Q.2) * (P.2 - R.2) = 0)
  (leg_PR : Real.sqrt ((P.1 - R.1)^2 + (P.2 - R.2)^2) = 9)
  (leg_PQ : Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 12)
  (S : ℝ × ℝ) 
  (T : ℝ × ℝ) 
  (square_side_on_hypotenuse : ∃ U : ℝ × ℝ, 
    (S.1 - T.1) * (Q.1 - R.1) + (S.2 - T.2) * (Q.2 - R.2) = 0 ∧
    Real.sqrt ((S.1 - T.1)^2 + (S.2 - T.2)^2) = 
    Real.sqrt ((S.1 - U.1)^2 + (S.2 - U.2)^2) ∧
    Real.sqrt ((T.1 - U.1)^2 + (T.2 - U.2)^2) = 
    Real.sqrt ((S.1 - T.1)^2 + (S.2 - T.2)^2))
  (S_on_PR : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ S = (P.1 + t * (R.1 - P.1), P.2 + t * (R.2 - P.2)))
  (T_on_PQ : ∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ T = (P.1 + s * (Q.1 - P.1), P.2 + s * (Q.2 - P.2))) :
  Real.sqrt ((S.1 - T.1)^2 + (S.2 - T.2)^2) = 5 + 5/7 := by
  sorry


end NUMINAMATH_CALUDE_square_on_hypotenuse_side_length_l4006_400642


namespace NUMINAMATH_CALUDE_square_rectangle_area_relation_l4006_400669

theorem square_rectangle_area_relation :
  let square_side : ℝ → ℝ := λ x => x - 5
  let rect_length : ℝ → ℝ := λ x => x - 4
  let rect_width : ℝ → ℝ := λ x => x + 5
  let square_area : ℝ → ℝ := λ x => (square_side x) ^ 2
  let rect_area : ℝ → ℝ := λ x => (rect_length x) * (rect_width x)
  ∃ x₁ x₂ : ℝ, x₁ > 5 ∧ x₂ > 5 ∧
    2 * x₁^2 - 31 * x₁ + 95 = 0 ∧
    2 * x₂^2 - 31 * x₂ + 95 = 0 ∧
    3 * (square_area x₁) = rect_area x₁ ∧
    3 * (square_area x₂) = rect_area x₂ ∧
    x₁ + x₂ = 31/2 :=
by
  sorry

end NUMINAMATH_CALUDE_square_rectangle_area_relation_l4006_400669


namespace NUMINAMATH_CALUDE_min_value_theorem_min_value_achieved_l4006_400659

theorem min_value_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : 1/a + 2/b + 3/c = 2) :
  a + 2*b + 3*c ≥ 18 :=
by sorry

theorem min_value_achieved (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : 1/a + 2/b + 3/c = 2) :
  (a + 2*b + 3*c = 18) ↔ (a = 3 ∧ b = 3 ∧ c = 3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_min_value_achieved_l4006_400659


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_one_l4006_400606

theorem fraction_zero_implies_x_negative_one (x : ℝ) :
  (1 - x^2) / (x - 1) = 0 ∧ x ≠ 1 → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_one_l4006_400606


namespace NUMINAMATH_CALUDE_tenth_term_of_sequence_l4006_400648

def geometric_sequence (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * r^(n - 1)

theorem tenth_term_of_sequence (a : ℚ) (r : ℚ) (h : a = 4 ∧ r = 1) :
  geometric_sequence a r 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_of_sequence_l4006_400648


namespace NUMINAMATH_CALUDE_coordinate_points_existence_l4006_400611

theorem coordinate_points_existence :
  ∃ (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ),
    (a₁ - b₁ = 4 ∧ a₁^2 + b₁^2 = 30 ∧ a₁ * b₁ = c₁) ∧
    (a₂ - b₂ = 4 ∧ a₂^2 + b₂^2 = 30 ∧ a₂ * b₂ = c₂) ∧
    a₁ = 2 + Real.sqrt 11 ∧
    b₁ = -2 + Real.sqrt 11 ∧
    c₁ = -15 + 4 * Real.sqrt 11 ∧
    a₂ = 2 - Real.sqrt 11 ∧
    b₂ = -2 - Real.sqrt 11 ∧
    c₂ = -15 - 4 * Real.sqrt 11 :=
by sorry

end NUMINAMATH_CALUDE_coordinate_points_existence_l4006_400611


namespace NUMINAMATH_CALUDE_b_not_played_e_l4006_400685

/-- Represents a soccer team in the tournament -/
inductive Team : Type
| A | B | C | D | E | F

/-- Represents the number of matches played by each team -/
def matches_played : Team → Nat
| Team.A => 5
| Team.B => 4
| Team.C => 3
| Team.D => 2
| Team.E => 1
| Team.F => 0  -- Inferred from the problem

/-- Predicate to check if two teams have played against each other -/
def has_played_against : Team → Team → Prop := sorry

/-- The theorem stating that team B has not played against team E -/
theorem b_not_played_e : ¬(has_played_against Team.B Team.E) := by
  sorry

end NUMINAMATH_CALUDE_b_not_played_e_l4006_400685


namespace NUMINAMATH_CALUDE_regular_milk_consumption_l4006_400679

def total_milk : ℝ := 0.6
def soy_milk : ℝ := 0.1

theorem regular_milk_consumption : total_milk - soy_milk = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_regular_milk_consumption_l4006_400679


namespace NUMINAMATH_CALUDE_simplify_expression_l4006_400683

theorem simplify_expression (b : ℝ) (h : b ≠ -2/3) :
  3 - 2 / (2 + b / (1 + b)) = 3 - (1 + b) / (2 + 3*b) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l4006_400683


namespace NUMINAMATH_CALUDE_marie_stamps_l4006_400630

theorem marie_stamps (notebooks : ℕ) (stamps_per_notebook : ℕ) (binders : ℕ) (stamps_per_binder : ℕ) (stamps_given_away : ℕ) :
  notebooks = 4 →
  stamps_per_notebook = 20 →
  binders = 2 →
  stamps_per_binder = 50 →
  stamps_given_away = 135 →
  (notebooks * stamps_per_notebook + binders * stamps_per_binder - stamps_given_away : ℚ) / (notebooks * stamps_per_notebook + binders * stamps_per_binder) = 1 / 4 :=
by sorry

end NUMINAMATH_CALUDE_marie_stamps_l4006_400630


namespace NUMINAMATH_CALUDE_max_red_balls_l4006_400686

/-- Given a pile of red and white balls, with the total number not exceeding 50,
    and the number of red balls being three times the number of white balls,
    prove that the maximum number of red balls is 36. -/
theorem max_red_balls (r w : ℕ) : 
  r + w ≤ 50 →  -- Total number of balls not exceeding 50
  r = 3 * w →   -- Number of red balls is three times the number of white balls
  r ≤ 36        -- Maximum number of red balls is 36
  := by sorry

end NUMINAMATH_CALUDE_max_red_balls_l4006_400686


namespace NUMINAMATH_CALUDE_smallest_fraction_between_l4006_400608

theorem smallest_fraction_between (p q : ℕ+) : 
  (3 : ℚ) / 5 < (p : ℚ) / q ∧ 
  (p : ℚ) / q < (5 : ℚ) / 8 ∧ 
  (∀ p' q' : ℕ+, (3 : ℚ) / 5 < (p' : ℚ) / q' ∧ (p' : ℚ) / q' < (5 : ℚ) / 8 → q ≤ q') →
  q - p = 5 := by
sorry

end NUMINAMATH_CALUDE_smallest_fraction_between_l4006_400608


namespace NUMINAMATH_CALUDE_total_age_l4006_400691

def kate_age : ℕ := 19
def maggie_age : ℕ := 17
def sue_age : ℕ := 12

theorem total_age : kate_age + maggie_age + sue_age = 48 := by
  sorry

end NUMINAMATH_CALUDE_total_age_l4006_400691


namespace NUMINAMATH_CALUDE_exponent_multiplication_l4006_400632

theorem exponent_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l4006_400632


namespace NUMINAMATH_CALUDE_max_value_complex_expression_l4006_400656

theorem max_value_complex_expression (z : ℂ) (h : Complex.abs z = 1) :
  Complex.abs (z^3 + z^2 - 5*z + 3) ≤ 128 * Real.sqrt 3 / 27 := by
  sorry

end NUMINAMATH_CALUDE_max_value_complex_expression_l4006_400656


namespace NUMINAMATH_CALUDE_gas_station_candy_boxes_l4006_400601

/-- The number of boxes of chocolate candy sold by a gas station -/
def chocolate_boxes : ℕ := 9 - (5 + 2)

/-- The total number of boxes sold -/
def total_boxes : ℕ := 9

/-- The number of boxes of sugar candy sold -/
def sugar_boxes : ℕ := 5

/-- The number of boxes of gum sold -/
def gum_boxes : ℕ := 2

theorem gas_station_candy_boxes :
  chocolate_boxes = 2 ∧
  chocolate_boxes + sugar_boxes + gum_boxes = total_boxes :=
sorry

end NUMINAMATH_CALUDE_gas_station_candy_boxes_l4006_400601


namespace NUMINAMATH_CALUDE_total_consumption_is_7700_l4006_400639

/-- Fuel consumption rates --/
def highway_rate : ℝ := 3
def city_rate : ℝ := 5

/-- Miles driven each day --/
def day1_highway : ℝ := 200
def day1_city : ℝ := 300
def day2_highway : ℝ := 300
def day2_city : ℝ := 500
def day3_highway : ℝ := 150
def day3_city : ℝ := 350

/-- Total gas consumption calculation --/
def total_consumption : ℝ :=
  (day1_highway * highway_rate + day1_city * city_rate) +
  (day2_highway * highway_rate + day2_city * city_rate) +
  (day3_highway * highway_rate + day3_city * city_rate)

/-- Theorem stating that the total gas consumption is 7700 gallons --/
theorem total_consumption_is_7700 : total_consumption = 7700 := by
  sorry

end NUMINAMATH_CALUDE_total_consumption_is_7700_l4006_400639


namespace NUMINAMATH_CALUDE_triangle_area_from_lines_l4006_400661

/-- The area of the triangle formed by the intersection of three lines -/
theorem triangle_area_from_lines (f g h : ℝ → ℝ) :
  (f = fun x ↦ x + 2) →
  (g = fun x ↦ -3*x + 9) →
  (h = fun _ ↦ 2) →
  let p₁ := (0, 2)
  let p₂ := (7/3, 2)
  let p₃ := (7/4, 15/4)
  let base := p₂.1 - p₁.1
  let height := p₃.2 - 2
  1/2 * base * height = 49/24 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_from_lines_l4006_400661


namespace NUMINAMATH_CALUDE_segment_length_in_triangle_l4006_400603

/-- Given a triangle with sides a, b, c, and three lines parallel to the sides
    intersecting at one point, with segments of length x cut off by the sides,
    prove that x = abc / (ab + bc + ac) -/
theorem segment_length_in_triangle (a b c x : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  x = (a * b * c) / (a * b + b * c + a * c) := by
  sorry

end NUMINAMATH_CALUDE_segment_length_in_triangle_l4006_400603


namespace NUMINAMATH_CALUDE_imaginary_part_of_product_l4006_400694

theorem imaginary_part_of_product : Complex.im ((3 - Complex.I) * (2 + Complex.I)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_product_l4006_400694


namespace NUMINAMATH_CALUDE_board_length_problem_l4006_400627

/-- The length of a board before the final cut, given initial length, first cut, and final adjustment cut. -/
def board_length_before_final_cut (initial_length first_cut final_cut : ℕ) : ℕ :=
  initial_length - first_cut + final_cut

/-- Theorem stating that the length of the boards before the final 7 cm cut was 125 cm. -/
theorem board_length_problem :
  board_length_before_final_cut 143 25 7 = 125 := by
  sorry

end NUMINAMATH_CALUDE_board_length_problem_l4006_400627


namespace NUMINAMATH_CALUDE_exists_greater_or_equal_scores_64_exists_greater_or_equal_scores_49_l4006_400657

/-- Represents a student's scores on three problems -/
structure StudentScores where
  problem1 : Nat
  problem2 : Nat
  problem3 : Nat
  h1 : problem1 ≤ 7
  h2 : problem2 ≤ 7
  h3 : problem3 ≤ 7

/-- Checks if one student's scores are greater than or equal to another's -/
def scoresGreaterOrEqual (a b : StudentScores) : Prop :=
  a.problem1 ≥ b.problem1 ∧ a.problem2 ≥ b.problem2 ∧ a.problem3 ≥ b.problem3

/-- Main theorem for part (a) -/
theorem exists_greater_or_equal_scores_64 :
  ∀ (students : Fin 64 → StudentScores),
  ∃ (i j : Fin 64), i ≠ j ∧ scoresGreaterOrEqual (students i) (students j) := by
  sorry

/-- Main theorem for part (b) -/
theorem exists_greater_or_equal_scores_49 :
  ∀ (students : Fin 49 → StudentScores),
  ∃ (i j : Fin 49), i ≠ j ∧ scoresGreaterOrEqual (students i) (students j) := by
  sorry

end NUMINAMATH_CALUDE_exists_greater_or_equal_scores_64_exists_greater_or_equal_scores_49_l4006_400657


namespace NUMINAMATH_CALUDE_sergey_age_l4006_400660

/-- Calculates the number of full years given a person's age components --/
def fullYears (years months weeks days hours : ℕ) : ℕ :=
  years + (months / 12) + ((weeks * 7 + days) / 365)

/-- Theorem stating that given the specific age components, the result is 39 full years --/
theorem sergey_age : fullYears 36 36 36 36 36 = 39 := by
  sorry

end NUMINAMATH_CALUDE_sergey_age_l4006_400660


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l4006_400675

/-- An arithmetic sequence with sum S_n for the first n terms -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  S : ℕ → ℝ  -- Sum function

/-- The common difference of an arithmetic sequence given specific conditions -/
theorem arithmetic_sequence_common_difference 
  (seq : ArithmeticSequence)
  (h1 : seq.a 4 + seq.a 5 = 24)
  (h2 : seq.S 6 = 48) :
  seq.d = 4 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l4006_400675


namespace NUMINAMATH_CALUDE_tangent_parallel_points_l4006_400625

-- Define the curve
def f (x : ℝ) : ℝ := x^3 + x - 2

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 + 1

-- Theorem statement
theorem tangent_parallel_points :
  ∀ x y : ℝ, f x = y ∧ f' x = 4 ↔ (x = -1 ∧ y = -4) ∨ (x = 1 ∧ y = 0) := by
  sorry

end NUMINAMATH_CALUDE_tangent_parallel_points_l4006_400625


namespace NUMINAMATH_CALUDE_wand_price_l4006_400673

theorem wand_price (purchase_price : ℝ) (original_price : ℝ) : 
  purchase_price = 4 → 
  purchase_price = (1/8) * original_price → 
  original_price = 32 := by
sorry

end NUMINAMATH_CALUDE_wand_price_l4006_400673


namespace NUMINAMATH_CALUDE_fraction_problem_l4006_400638

theorem fraction_problem (F : ℚ) (m : ℕ) : 
  F = 1/5 ∧ m = 4 → (F^m) * (1/4)^2 = 1/((10:ℚ)^4) :=
by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l4006_400638


namespace NUMINAMATH_CALUDE_undeveloped_land_area_l4006_400628

theorem undeveloped_land_area (total_area : ℝ) (num_sections : ℕ) 
  (h1 : total_area = 7305)
  (h2 : num_sections = 3) :
  total_area / num_sections = 2435 := by
  sorry

end NUMINAMATH_CALUDE_undeveloped_land_area_l4006_400628


namespace NUMINAMATH_CALUDE_coin_combination_theorem_l4006_400690

/-- Represents the number of coins of each denomination -/
structure CoinCounts where
  fiveCent : ℕ
  tenCent : ℕ
  twentyFiveCent : ℕ

/-- Calculates the number of different values obtainable from a given set of coins -/
def differentValues (coins : CoinCounts) : ℕ := sorry

theorem coin_combination_theorem (coins : CoinCounts) :
  coins.fiveCent + coins.tenCent + coins.twentyFiveCent = 15 →
  differentValues coins = 23 →
  coins.twentyFiveCent = 3 := by sorry

end NUMINAMATH_CALUDE_coin_combination_theorem_l4006_400690


namespace NUMINAMATH_CALUDE_ad_greater_than_bc_l4006_400666

theorem ad_greater_than_bc (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (sum_eq : a + d = b + c)
  (abs_ineq : |a - d| < |b - c|) : 
  a * d > b * c := by
sorry

end NUMINAMATH_CALUDE_ad_greater_than_bc_l4006_400666


namespace NUMINAMATH_CALUDE_concert_songs_count_l4006_400602

/-- Calculates the number of songs in a concert given the total duration,
    intermission time, regular song duration, and special song duration. -/
def number_of_songs (total_duration intermission_time regular_song_duration special_song_duration : ℕ) : ℕ :=
  let singing_time := total_duration - intermission_time
  let regular_songs_time := singing_time - special_song_duration
  (regular_songs_time / regular_song_duration) + 1

/-- Theorem stating that the number of songs in the given concert is 13. -/
theorem concert_songs_count :
  number_of_songs 80 10 5 10 = 13 := by
  sorry

end NUMINAMATH_CALUDE_concert_songs_count_l4006_400602


namespace NUMINAMATH_CALUDE_cylinder_cone_volume_ratio_l4006_400699

theorem cylinder_cone_volume_ratio (h r_cylinder r_cone : ℝ) 
  (h_positive : h > 0)
  (r_cylinder_positive : r_cylinder > 0)
  (r_cone_positive : r_cone > 0)
  (cross_section_equal : h * (2 * r_cylinder) = h * r_cone) :
  (π * r_cylinder^2 * h) / ((1/3) * π * r_cone^2 * h) = 3/4 := by
sorry

end NUMINAMATH_CALUDE_cylinder_cone_volume_ratio_l4006_400699


namespace NUMINAMATH_CALUDE_joe_ball_choices_l4006_400614

/-- The number of balls in the bin -/
def num_balls : ℕ := 18

/-- The number of times a ball is chosen -/
def num_choices : ℕ := 4

/-- The number of different possible lists -/
def num_lists : ℕ := num_balls ^ num_choices

theorem joe_ball_choices :
  num_lists = 104976 := by
  sorry

end NUMINAMATH_CALUDE_joe_ball_choices_l4006_400614


namespace NUMINAMATH_CALUDE_compare_sqrt_l4006_400637

theorem compare_sqrt : 2 * Real.sqrt 11 < 3 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_compare_sqrt_l4006_400637


namespace NUMINAMATH_CALUDE_polynomial_factor_coefficient_l4006_400635

theorem polynomial_factor_coefficient (a b : ℤ) : 
  (∃ (c d : ℤ), ∀ (x : ℝ), 
    a * x^4 + b * x^3 + 40 * x^2 - 20 * x + 8 = (2 * x^2 - 3 * x + 2) * (c * x^2 + d * x + 4)) →
  a = 112 ∧ b = -152 := by
sorry

end NUMINAMATH_CALUDE_polynomial_factor_coefficient_l4006_400635


namespace NUMINAMATH_CALUDE_optimal_AD_length_l4006_400650

/-- Triangle ABC with given side lengths -/
structure Triangle :=
  (AB : ℝ)
  (AC : ℝ)
  (BC : ℝ)

/-- Point D on AB -/
def D (t : Triangle) := ℝ

/-- Expected value of EF -/
noncomputable def expectedEF (t : Triangle) (d : D t) : ℝ := sorry

/-- Theorem statement -/
theorem optimal_AD_length (t : Triangle) 
  (h1 : t.AB = 14) 
  (h2 : t.AC = 13) 
  (h3 : t.BC = 15) : 
  ∃ (d : D t), 
    (∀ (d' : D t), expectedEF t d ≥ expectedEF t d') ∧ 
    d = Real.sqrt 70 :=
sorry

end NUMINAMATH_CALUDE_optimal_AD_length_l4006_400650


namespace NUMINAMATH_CALUDE_combination_sum_equals_seven_l4006_400684

theorem combination_sum_equals_seven (n : ℕ) 
  (h1 : 0 ≤ 5 - n ∧ 5 - n ≤ n) 
  (h2 : 0 ≤ 10 - n ∧ 10 - n ≤ n + 1) : 
  Nat.choose n (5 - n) + Nat.choose (n + 1) (10 - n) = 7 := by
  sorry

end NUMINAMATH_CALUDE_combination_sum_equals_seven_l4006_400684


namespace NUMINAMATH_CALUDE_back_wheel_circumference_l4006_400658

/-- Given a cart with front and back wheels, this theorem proves the circumference of the back wheel
    based on the given conditions. -/
theorem back_wheel_circumference
  (front_circumference : ℝ)
  (distance : ℝ)
  (revolution_difference : ℕ)
  (h1 : front_circumference = 30)
  (h2 : distance = 1650)
  (h3 : revolution_difference = 5) :
  ∃ (back_circumference : ℝ),
    back_circumference * (distance / front_circumference - revolution_difference) = distance ∧
    back_circumference = 33 :=
by sorry

end NUMINAMATH_CALUDE_back_wheel_circumference_l4006_400658


namespace NUMINAMATH_CALUDE_school_selection_probability_l4006_400655

theorem school_selection_probability :
  let total_schools : ℕ := 4
  let schools_to_select : ℕ := 2
  let total_combinations : ℕ := (total_schools.choose schools_to_select)
  let favorable_outcomes : ℕ := ((total_schools - 1).choose (schools_to_select - 1))
  favorable_outcomes / total_combinations = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_school_selection_probability_l4006_400655


namespace NUMINAMATH_CALUDE_conic_section_classification_l4006_400665

/-- The equation y^4 - 9x^4 = 3y^2 - 4 represents the union of two hyperbolas -/
theorem conic_section_classification (x y : ℝ) :
  (y^4 - 9*x^4 = 3*y^2 - 4) ↔
  ((y^2 - 3*x^2 = 5/2) ∨ (y^2 - 3*x^2 = 1)) :=
sorry

end NUMINAMATH_CALUDE_conic_section_classification_l4006_400665


namespace NUMINAMATH_CALUDE_first_stack_height_is_5_l4006_400689

/-- The height of the first stack of blocks -/
def first_stack_height : ℕ := sorry

/-- The height of the second stack of blocks -/
def second_stack_height : ℕ := first_stack_height + 2

/-- The height of the third stack of blocks -/
def third_stack_height : ℕ := second_stack_height - 5

/-- The height of the fourth stack of blocks -/
def fourth_stack_height : ℕ := third_stack_height + 5

/-- The total number of blocks used -/
def total_blocks : ℕ := 21

theorem first_stack_height_is_5 : 
  first_stack_height = 5 ∧ 
  first_stack_height + second_stack_height + third_stack_height + fourth_stack_height = total_blocks :=
sorry

end NUMINAMATH_CALUDE_first_stack_height_is_5_l4006_400689


namespace NUMINAMATH_CALUDE_three_correct_propositions_l4006_400619

/-- Represents the type of events --/
inductive EventType
  | Certain
  | Impossible
  | Random

/-- Represents a proposition about an event --/
structure Proposition where
  statement : String
  eventType : EventType

/-- Checks if a proposition is correct --/
def isCorrectProposition (p : Proposition) : Bool :=
  match p.statement, p.eventType with
  | "Placing all three balls into two boxes, there must be one box containing more than one ball", EventType.Certain => true
  | "For some real number x, it can make x^2 < 0", EventType.Impossible => true
  | "It will rain in Guangzhou tomorrow", EventType.Certain => false
  | "Out of 100 light bulbs, there are 5 defective ones. Taking out 5 bulbs and all 5 are defective", EventType.Random => true
  | _, _ => false

/-- The list of propositions --/
def propositions : List Proposition := [
  ⟨"Placing all three balls into two boxes, there must be one box containing more than one ball", EventType.Certain⟩,
  ⟨"For some real number x, it can make x^2 < 0", EventType.Impossible⟩,
  ⟨"It will rain in Guangzhou tomorrow", EventType.Certain⟩,
  ⟨"Out of 100 light bulbs, there are 5 defective ones. Taking out 5 bulbs and all 5 are defective", EventType.Random⟩
]

theorem three_correct_propositions :
  (propositions.filter isCorrectProposition).length = 3 := by
  sorry

end NUMINAMATH_CALUDE_three_correct_propositions_l4006_400619


namespace NUMINAMATH_CALUDE_candidate_count_l4006_400682

theorem candidate_count (total : ℕ) : 
  (total * 6 / 100 : ℕ) + 84 = total * 7 / 100 → total = 8400 := by
  sorry

end NUMINAMATH_CALUDE_candidate_count_l4006_400682


namespace NUMINAMATH_CALUDE_head_start_time_l4006_400653

/-- Proves that given a runner completes a 1000-meter race in 190 seconds,
    the time equivalent to a 50-meter head start is 9.5 seconds. -/
theorem head_start_time (race_distance : ℝ) (race_time : ℝ) (head_start_distance : ℝ) : 
  race_distance = 1000 →
  race_time = 190 →
  head_start_distance = 50 →
  (head_start_distance / (race_distance / race_time)) = 9.5 := by
  sorry

end NUMINAMATH_CALUDE_head_start_time_l4006_400653


namespace NUMINAMATH_CALUDE_floor_ceil_sum_l4006_400695

theorem floor_ceil_sum : ⌊(1.99 : ℝ)⌋ + ⌈(3.02 : ℝ)⌉ = 5 := by
  sorry

end NUMINAMATH_CALUDE_floor_ceil_sum_l4006_400695


namespace NUMINAMATH_CALUDE_sample_size_is_ten_l4006_400605

/-- A structure representing a quality inspection scenario -/
structure QualityInspection where
  total_products : ℕ
  selected_products : ℕ

/-- Definition of sample size for a quality inspection -/
def sample_size (qi : QualityInspection) : ℕ := qi.selected_products

/-- Theorem stating that for the given scenario, the sample size is 10 -/
theorem sample_size_is_ten (qi : QualityInspection) 
  (h1 : qi.total_products = 80) 
  (h2 : qi.selected_products = 10) : 
  sample_size qi = 10 := by
  sorry

end NUMINAMATH_CALUDE_sample_size_is_ten_l4006_400605


namespace NUMINAMATH_CALUDE_sundae_probability_l4006_400621

def ice_cream_flavors : ℕ := 3
def syrup_types : ℕ := 2
def topping_options : ℕ := 3

def total_combinations : ℕ := ice_cream_flavors * syrup_types * topping_options

def specific_combination : ℕ := 1

theorem sundae_probability :
  (specific_combination : ℚ) / total_combinations = 1 / 18 := by sorry

end NUMINAMATH_CALUDE_sundae_probability_l4006_400621


namespace NUMINAMATH_CALUDE_area_enclosed_by_line_and_parabola_l4006_400688

/-- The area of the region enclosed by y = (a/6)x and y = x^2, 
    where a is the constant term in (x + 2/x)^n and 
    the sum of coefficients in the expansion is 81 -/
theorem area_enclosed_by_line_and_parabola (n : ℕ) (a : ℝ) : 
  (3 : ℝ)^n = 81 →
  (∃ k, (Nat.choose 4 2) * 2^2 = k ∧ k = a) →
  (∫ x in (0)..(a/4), (a/6 * x - x^2)) = 32/3 := by
sorry

end NUMINAMATH_CALUDE_area_enclosed_by_line_and_parabola_l4006_400688


namespace NUMINAMATH_CALUDE_sqrt_expression_simplification_l4006_400636

theorem sqrt_expression_simplification :
  Real.sqrt 24 - 3 * Real.sqrt (1/6) + Real.sqrt 6 = (5 * Real.sqrt 6) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_simplification_l4006_400636


namespace NUMINAMATH_CALUDE_min_length_of_rectangle_l4006_400663

theorem min_length_of_rectangle (a : ℝ) (h : a > 0) :
  ∀ x y : ℝ, x > 0 → y > 0 → x * y = a^2 → min x y ≥ a :=
by sorry

end NUMINAMATH_CALUDE_min_length_of_rectangle_l4006_400663


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l4006_400668

/-- Given a line L1: 2x + y - 3 = 0 and a point P(0, 1), 
    prove that the line L2: 2x + y - 1 = 0 passes through P and is parallel to L1. -/
theorem parallel_line_through_point (x y : ℝ) : 
  let L1 := {(x, y) | 2 * x + y - 3 = 0}
  let P := (0, 1)
  let L2 := {(x, y) | 2 * x + y - 1 = 0}
  (P ∈ L2) ∧ (∀ (x1 y1 x2 y2 : ℝ), (x1, y1) ∈ L1 → (x2, y2) ∈ L1 → 
    (x2 - x1) * 1 = (y2 - y1) * 2 ↔ 
    (x2 - x1) * 1 = (y2 - y1) * 2) := by
  sorry


end NUMINAMATH_CALUDE_parallel_line_through_point_l4006_400668


namespace NUMINAMATH_CALUDE_floor_sqrt_ten_l4006_400647

theorem floor_sqrt_ten : ⌊Real.sqrt 10⌋ = 3 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_ten_l4006_400647


namespace NUMINAMATH_CALUDE_work_completion_time_l4006_400644

-- Define the work completion time for B
def b_time : ℝ := 8

-- Define the work completion time for A and B together
def ab_time : ℝ := 4.444444444444445

-- Define the work completion time for A
def a_time : ℝ := 10

-- Theorem statement
theorem work_completion_time :
  b_time = 8 ∧ ab_time = 4.444444444444445 →
  a_time = 10 :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l4006_400644


namespace NUMINAMATH_CALUDE_natalie_portion_ratio_l4006_400620

def total_amount : ℝ := 10000

def third_person_amount : ℝ := 2000

def second_person_percentage : ℝ := 0.6

theorem natalie_portion_ratio (first_person_amount : ℝ) 
  (h1 : third_person_amount = total_amount - first_person_amount - second_person_percentage * (total_amount - first_person_amount))
  (h2 : first_person_amount > 0)
  (h3 : first_person_amount < total_amount) :
  first_person_amount / total_amount = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_natalie_portion_ratio_l4006_400620


namespace NUMINAMATH_CALUDE_tangent_line_to_parabola_l4006_400631

/-- A line is tangent to a parabola if and only if the discriminant of the resulting quadratic equation is zero -/
axiom tangent_condition (a b c : ℝ) : 
  b^2 - 4*a*c = 0 ↔ ∃ (x y : ℝ), y^2 = 12*x ∧ y = 3*x + c

/-- If the line y = 3x + d is tangent to the parabola y^2 = 12x, then d = 1 -/
theorem tangent_line_to_parabola (d : ℝ) : 
  (∃ (x y : ℝ), y^2 = 12*x ∧ y = 3*x + d) → d = 1 := by
  sorry

#check tangent_line_to_parabola

end NUMINAMATH_CALUDE_tangent_line_to_parabola_l4006_400631


namespace NUMINAMATH_CALUDE_routes_on_grid_l4006_400616

/-- The number of routes from A to B on a 3x3 grid -/
def num_routes : ℕ := 20

/-- The size of the grid -/
def grid_size : ℕ := 3

/-- The number of right moves required -/
def right_moves : ℕ := 3

/-- The number of down moves required -/
def down_moves : ℕ := 3

/-- The total number of moves required -/
def total_moves : ℕ := right_moves + down_moves

theorem routes_on_grid : 
  num_routes = (Nat.choose total_moves right_moves) := by
  sorry

end NUMINAMATH_CALUDE_routes_on_grid_l4006_400616


namespace NUMINAMATH_CALUDE_triangle_existence_l4006_400677

-- Define the basic types and structures
structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the triangle
structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define the function to check if a point is on a line
def isPointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Define the function to check if a line cuts off equal segments from a point on two other lines
def cutsEqualSegments (l : Line) (A : Point) (AB AC : Line) : Prop :=
  ∃ (P Q : Point), isPointOnLine P AB ∧ isPointOnLine Q AC ∧
    isPointOnLine P l ∧ isPointOnLine Q l ∧
    (P.x - A.x)^2 + (P.y - A.y)^2 = (Q.x - A.x)^2 + (Q.y - A.y)^2

-- State the theorem
theorem triangle_existence 
  (A O : Point) 
  (l : Line) 
  (h_euler : isPointOnLine O l)
  (h_equal_segments : ∃ (AB AC : Line), cutsEqualSegments l A AB AC) :
  ∃ (T : Triangle), T.A = A ∧ 
    isPointOnLine T.B l ∧ isPointOnLine T.C l ∧
    (T.B.x - O.x)^2 + (T.B.y - O.y)^2 = (T.C.x - O.x)^2 + (T.C.y - O.y)^2 :=
sorry

end NUMINAMATH_CALUDE_triangle_existence_l4006_400677


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l4006_400626

theorem hyperbola_eccentricity (a b c : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 ∧ 
                x = a^2 / c ∧ 
                y = -a * b / c ∧
                ∃ (x_f1 y_f1 : ℝ), (x_f1 = -c ∧ y_f1 = 0) ∧
                                   (x + x_f1) / 2 = a^2 / c ∧
                                   (y + y_f1) / 2 = -a * b / c) →
  c / a = Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l4006_400626


namespace NUMINAMATH_CALUDE_simplify_expression_l4006_400612

theorem simplify_expression (m : ℝ) (h : m > 0) : 
  (Real.sqrt m * 3 * m) / ((6 * m) ^ 5) = 1 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l4006_400612


namespace NUMINAMATH_CALUDE_curve_c_properties_l4006_400618

/-- Curve C defined by mx^2 - ny^2 = 1 -/
structure CurveC (m n : ℝ) where
  equation : ∀ (x y : ℝ), m * x^2 - n * y^2 = 1

/-- Definition of a hyperbola -/
def is_hyperbola (C : CurveC m n) : Prop := sorry

/-- Definition of an ellipse with foci on the x-axis -/
def is_ellipse_x_foci (C : CurveC m n) : Prop := sorry

/-- Definition of a circle -/
def is_circle (C : CurveC m n) : Prop := sorry

/-- Definition of two straight lines -/
def is_two_straight_lines (C : CurveC m n) : Prop := sorry

theorem curve_c_properties (m n : ℝ) (C : CurveC m n) :
  (m * n > 0 → is_hyperbola C) ∧
  (m > 0 ∧ m + n < 0 → is_ellipse_x_foci C) ∧
  (¬(m > 0 ∧ n < 0 → ¬is_circle C)) ∧
  (m > 0 ∧ n = 0 → is_two_straight_lines C) := by
  sorry

end NUMINAMATH_CALUDE_curve_c_properties_l4006_400618


namespace NUMINAMATH_CALUDE_number_to_billions_l4006_400604

/-- Converts a number to billions -/
def to_billions (n : ℕ) : ℚ :=
  (n : ℚ) / 1000000000

theorem number_to_billions :
  to_billions 640080000 = 0.64008 := by sorry

end NUMINAMATH_CALUDE_number_to_billions_l4006_400604


namespace NUMINAMATH_CALUDE_angle_sum_in_special_figure_l4006_400676

theorem angle_sum_in_special_figure (A B C x y : ℝ) : 
  A = 34 → B = 80 → C = 30 →
  (A + B + (360 - x) + 90 + (120 - y) = 720) →
  x + y = 36 := by sorry

end NUMINAMATH_CALUDE_angle_sum_in_special_figure_l4006_400676


namespace NUMINAMATH_CALUDE_jennifer_cookie_sales_l4006_400643

theorem jennifer_cookie_sales (kim_sales : ℕ) (jennifer_extra : ℕ) 
  (h1 : kim_sales = 54)
  (h2 : jennifer_extra = 17) : 
  kim_sales + jennifer_extra = 71 := by
  sorry

end NUMINAMATH_CALUDE_jennifer_cookie_sales_l4006_400643


namespace NUMINAMATH_CALUDE_jessica_jelly_bean_guess_l4006_400629

/-- Represents the number of jelly beans of each color in a bag -/
structure JellyBeanBag where
  red : ℕ
  black : ℕ
  green : ℕ
  purple : ℕ
  yellow : ℕ
  white : ℕ

/-- Calculates the number of bags needed to fill the fishbowl -/
def bagsNeeded (bag : JellyBeanBag) (guessRedWhite : ℕ) : ℕ :=
  guessRedWhite / (bag.red + bag.white)

theorem jessica_jelly_bean_guess 
  (bag : JellyBeanBag)
  (guessRedWhite : ℕ)
  (h1 : bag.red = 24)
  (h2 : bag.black = 13)
  (h3 : bag.green = 36)
  (h4 : bag.purple = 28)
  (h5 : bag.yellow = 32)
  (h6 : bag.white = 18)
  (h7 : guessRedWhite = 126) :
  bagsNeeded bag guessRedWhite = 3 := by
  sorry

end NUMINAMATH_CALUDE_jessica_jelly_bean_guess_l4006_400629


namespace NUMINAMATH_CALUDE_target_probabilities_l4006_400664

def prob_hit : ℝ := 0.8
def total_shots : ℕ := 4

theorem target_probabilities :
  let prob_miss := 1 - prob_hit
  (1 - prob_miss ^ total_shots = 0.9984) ∧
  (prob_hit ^ 3 * prob_miss * total_shots + prob_hit ^ total_shots = 0.8192) ∧
  (prob_miss ^ total_shots + total_shots * prob_hit * prob_miss ^ 3 = 0.2576) := by
  sorry

#check target_probabilities

end NUMINAMATH_CALUDE_target_probabilities_l4006_400664


namespace NUMINAMATH_CALUDE_two_distinct_roots_range_l4006_400607

theorem two_distinct_roots_range (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 
    x^4 - 2*a*x^2 - x + a^2 - a = 0 ∧ 
    y^4 - 2*a*y^2 - y + a^2 - a = 0 ∧
    (∀ z : ℝ, z^4 - 2*a*z^2 - z + a^2 - a = 0 → z = x ∨ z = y)) →
  a > -1/4 ∧ a < 3/4 :=
sorry

end NUMINAMATH_CALUDE_two_distinct_roots_range_l4006_400607


namespace NUMINAMATH_CALUDE_movie_duration_l4006_400662

theorem movie_duration (flight_duration : ℕ) (tv_time : ℕ) (sleep_time : ℕ) (remaining_time : ℕ) (num_movies : ℕ) :
  flight_duration = 600 →
  tv_time = 75 →
  sleep_time = 270 →
  remaining_time = 45 →
  num_movies = 2 →
  ∃ (movie_duration : ℕ),
    flight_duration = tv_time + sleep_time + num_movies * movie_duration + remaining_time ∧
    movie_duration = 105 := by
  sorry

end NUMINAMATH_CALUDE_movie_duration_l4006_400662


namespace NUMINAMATH_CALUDE_solution_satisfies_equations_l4006_400672

theorem solution_satisfies_equations :
  ∃ (x y : ℝ), 3 * x - 8 * y = 2 ∧ 4 * y - x = 6 ∧ x = 14 ∧ y = 5 := by
  sorry

end NUMINAMATH_CALUDE_solution_satisfies_equations_l4006_400672


namespace NUMINAMATH_CALUDE_sqrt_six_times_sqrt_two_equals_two_sqrt_three_l4006_400617

theorem sqrt_six_times_sqrt_two_equals_two_sqrt_three :
  Real.sqrt 6 * Real.sqrt 2 = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_six_times_sqrt_two_equals_two_sqrt_three_l4006_400617


namespace NUMINAMATH_CALUDE_jose_bottle_caps_l4006_400600

theorem jose_bottle_caps (initial : ℕ) (received : ℕ) (total : ℕ) : 
  initial = 7 → received = 2 → total = initial + received → total = 9 := by
  sorry

end NUMINAMATH_CALUDE_jose_bottle_caps_l4006_400600


namespace NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_squared_l4006_400646

theorem greatest_divisor_four_consecutive_integers_squared (n : ℕ) :
  ∃ (k : ℕ), k = 144 ∧ (∀ m : ℕ, m > k → ¬(m ∣ (n * (n + 1) * (n + 2) * (n + 3))^2)) ∧
  (k ∣ (n * (n + 1) * (n + 2) * (n + 3))^2) := by
  sorry

end NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_squared_l4006_400646


namespace NUMINAMATH_CALUDE_license_plate_count_l4006_400634

/-- The number of possible letters in each position of the license plate -/
def num_letters : ℕ := 26

/-- The number of odd digits available for the first position -/
def num_odd_digits : ℕ := 5

/-- The number of even digits available for the second position -/
def num_even_digits : ℕ := 5

/-- The number of digits that are multiples of 3 available for the third position -/
def num_multiples_of_3 : ℕ := 4

/-- The total number of license plates satisfying the given conditions -/
def total_license_plates : ℕ := num_letters ^ 3 * num_odd_digits * num_even_digits * num_multiples_of_3

theorem license_plate_count : total_license_plates = 878800 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l4006_400634


namespace NUMINAMATH_CALUDE_polygon_interior_exterior_angle_sum_l4006_400671

theorem polygon_interior_exterior_angle_sum (n : ℕ) : 
  (n ≥ 3) → 
  ((n - 2) * 180 = 2 * 360) → 
  n = 6 := by
  sorry

end NUMINAMATH_CALUDE_polygon_interior_exterior_angle_sum_l4006_400671


namespace NUMINAMATH_CALUDE_symmetry_and_periodicity_l4006_400610

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the derivative of f
variable (f' : ℝ → ℝ)

-- Assume f' is the derivative of f
axiom is_derivative : ∀ x, HasDerivAt f (f' x) x

-- Define even function property
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

-- State the theorem
theorem symmetry_and_periodicity 
  (h1 : is_even (fun x ↦ f (x - 1)))
  (h2 : is_even (fun x ↦ f (x - 2))) :
  (∀ x, f (-x - 2) = f x) ∧ 
  (∀ x, f (x + 2) = f x) ∧
  (∀ x, f' (-x + 4) = f' x) :=
sorry

end NUMINAMATH_CALUDE_symmetry_and_periodicity_l4006_400610


namespace NUMINAMATH_CALUDE_probability_two_white_balls_l4006_400696

/-- The probability of drawing two white balls sequentially without replacement from a box containing 7 white balls and 8 black balls is 1/5. -/
theorem probability_two_white_balls (total_balls : ℕ) (white_balls : ℕ) (black_balls : ℕ) :
  total_balls = white_balls + black_balls →
  white_balls = 7 →
  black_balls = 8 →
  (white_balls / total_balls) * ((white_balls - 1) / (total_balls - 1)) = 1 / 5 :=
by sorry

end NUMINAMATH_CALUDE_probability_two_white_balls_l4006_400696


namespace NUMINAMATH_CALUDE_maria_chocolate_chip_cookies_l4006_400678

/-- Calculates the number of chocolate chip cookies Maria had -/
def chocolateChipCookies (cookiesPerBag : ℕ) (oatmealCookies : ℕ) (numBags : ℕ) : ℕ :=
  cookiesPerBag * numBags - oatmealCookies

/-- Proves that Maria had 5 chocolate chip cookies -/
theorem maria_chocolate_chip_cookies :
  chocolateChipCookies 8 19 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_maria_chocolate_chip_cookies_l4006_400678


namespace NUMINAMATH_CALUDE_third_term_is_64_l4006_400652

/-- A geometric sequence with positive integer terms -/
structure GeometricSequence where
  terms : ℕ → ℕ
  first_term : terms 1 = 4
  is_geometric : ∀ n : ℕ, n > 0 → ∃ r : ℚ, terms (n + 1) = (terms n : ℚ) * r

/-- The theorem stating that for a geometric sequence with first term 4 and fourth term 256, the third term is 64 -/
theorem third_term_is_64 (seq : GeometricSequence) (h : seq.terms 4 = 256) : seq.terms 3 = 64 := by
  sorry

end NUMINAMATH_CALUDE_third_term_is_64_l4006_400652


namespace NUMINAMATH_CALUDE_consecutive_numbers_sum_l4006_400640

theorem consecutive_numbers_sum (a : ℕ) (h : a + (a + 1) + (a + 2) + (a + 3) + (a + 4) = 60) :
  a + 4 = 14 := by sorry

end NUMINAMATH_CALUDE_consecutive_numbers_sum_l4006_400640


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l4006_400693

theorem absolute_value_inequality (x : ℝ) : |x| > 2 ↔ x > 2 ∨ x < -2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l4006_400693


namespace NUMINAMATH_CALUDE_modular_arithmetic_problem_l4006_400649

theorem modular_arithmetic_problem (m : ℕ) : 
  m < 41 ∧ (5 * m) % 41 = 1 → (3^m % 41)^2 % 41 - 3 % 41 = 6 % 41 := by
  sorry

end NUMINAMATH_CALUDE_modular_arithmetic_problem_l4006_400649


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l4006_400624

/-- A geometric sequence with a_3 = 4 and a_6 = 1/2 has a common ratio of 1/2. -/
theorem geometric_sequence_common_ratio : ∀ (a : ℕ → ℝ), 
  (∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q) →  -- Geometric sequence definition
  a 3 = 4 →                                  -- Condition 1
  a 6 = 1 / 2 →                              -- Condition 2
  (∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q ∧ q = 1 / 2) := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l4006_400624


namespace NUMINAMATH_CALUDE_inlet_fill_rate_l4006_400687

/-- Given a tank with the following properties:
  * Capacity: 12960 liters
  * Time to empty with leak alone: 9 hours
  * Time to empty with leak and inlet: 12 hours
  Prove that the rate at which the inlet pipe fills water is 2520 liters per hour. -/
theorem inlet_fill_rate 
  (tank_capacity : ℝ) 
  (empty_time_leak : ℝ) 
  (empty_time_leak_and_inlet : ℝ) 
  (h1 : tank_capacity = 12960)
  (h2 : empty_time_leak = 9)
  (h3 : empty_time_leak_and_inlet = 12) :
  (tank_capacity / empty_time_leak) + (tank_capacity / empty_time_leak_and_inlet) = 2520 :=
by sorry

end NUMINAMATH_CALUDE_inlet_fill_rate_l4006_400687
