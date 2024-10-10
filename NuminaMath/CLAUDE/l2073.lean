import Mathlib

namespace tangent_line_equation_l2073_207317

def parabola (x : ℝ) : ℝ := x^2 + x + 1

theorem tangent_line_equation :
  let f := parabola
  let x₀ : ℝ := 0
  let y₀ : ℝ := f x₀
  let m := (deriv f) x₀
  ∀ x y, y - y₀ = m * (x - x₀) ↔ x - y + 1 = 0 := by sorry

end tangent_line_equation_l2073_207317


namespace harmonious_point_in_third_quadrant_l2073_207327

/-- A point (x, y) is harmonious if 3x = 2y + 5 -/
def IsHarmonious (x y : ℝ) : Prop := 3 * x = 2 * y + 5

/-- The x-coordinate of point M -/
def Mx (m : ℝ) : ℝ := m - 1

/-- The y-coordinate of point M -/
def My (m : ℝ) : ℝ := 3 * m + 2

theorem harmonious_point_in_third_quadrant :
  ∀ m : ℝ, IsHarmonious (Mx m) (My m) → Mx m < 0 ∧ My m < 0 := by
  sorry

end harmonious_point_in_third_quadrant_l2073_207327


namespace binary_ternary_equality_l2073_207371

theorem binary_ternary_equality (a b : ℕ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : b ≤ 1) (h4 : a ≤ 2) (h5 : 9 + 2*b = 9*a + 2) : 2*a + b = 3 := by
  sorry

end binary_ternary_equality_l2073_207371


namespace length_of_ac_l2073_207389

/-- Given 5 consecutive points on a straight line, prove that the length of ac is 11 -/
theorem length_of_ac (a b c d e : Real) : 
  (b - a) = 5 →
  (c - b) = 3 * (d - c) →
  (e - d) = 7 →
  (e - a) = 20 →
  (c - a) = 11 :=
by sorry

end length_of_ac_l2073_207389


namespace smallest_x_multiple_of_61_l2073_207346

theorem smallest_x_multiple_of_61 :
  ∃ (x : ℕ), x > 0 ∧ 
  (∀ (y : ℕ), y > 0 → y < x → ¬(61 ∣ (3*y)^2 + 3*58*3*y + 58^2)) ∧ 
  (61 ∣ (3*x)^2 + 3*58*3*x + 58^2) :=
by
  -- The proof goes here
  sorry

end smallest_x_multiple_of_61_l2073_207346


namespace quadratic_equation_roots_specific_quadratic_roots_l2073_207342

theorem quadratic_equation_roots (a b c : ℝ) (h : a ≠ 0) : 
  let discriminant := b^2 - 4*a*c
  discriminant > 0 → ∃ x y : ℝ, x ≠ y ∧ a*x^2 + b*x + c = 0 ∧ a*y^2 + b*y + c = 0 :=
by
  sorry

theorem specific_quadratic_roots : 
  ∃ x y : ℝ, x ≠ y ∧ x^2 - 2*x - 6 = 0 ∧ y^2 - 2*y - 6 = 0 :=
by
  sorry

end quadratic_equation_roots_specific_quadratic_roots_l2073_207342


namespace tangent_line_equation_l2073_207309

/-- The equation of the tangent line to y = xe^(2x-1) at (1, e) is 3ex - y - 2e = 0 -/
theorem tangent_line_equation (x y : ℝ) : 
  (y = x * Real.exp (2 * x - 1)) → -- Given curve equation
  (3 * Real.exp 1 * x - y - 2 * Real.exp 1 = 0) ↔ -- Tangent line equation
  (y - Real.exp 1 = 3 * Real.exp 1 * (x - 1)) -- Point-slope form at (1, e)
  := by sorry

end tangent_line_equation_l2073_207309


namespace simplify_sqrt_expression_simplify_algebraic_expression_simplify_complex_sqrt_expression_simplify_difference_of_squares_l2073_207352

-- (1)
theorem simplify_sqrt_expression : 
  Real.sqrt 18 - Real.sqrt 8 + Real.sqrt (1/8) = (5 * Real.sqrt 2) / 4 := by sorry

-- (2)
theorem simplify_algebraic_expression : 
  (1 + Real.sqrt 3) * (2 - Real.sqrt 3) = Real.sqrt 3 - 1 := by sorry

-- (3)
theorem simplify_complex_sqrt_expression : 
  (Real.sqrt 15 + Real.sqrt 60) / Real.sqrt 3 - 3 * Real.sqrt 5 = -5 * Real.sqrt 5 := by sorry

-- (4)
theorem simplify_difference_of_squares : 
  (Real.sqrt 7 + Real.sqrt 3) * (Real.sqrt 7 - Real.sqrt 3) - Real.sqrt 36 = -2 := by sorry

end simplify_sqrt_expression_simplify_algebraic_expression_simplify_complex_sqrt_expression_simplify_difference_of_squares_l2073_207352


namespace triangle_shape_l2073_207328

/-- Given a triangle ABC, prove that it is a right isosceles triangle under certain conditions. -/
theorem triangle_shape (a b c : ℝ) (A B C : ℝ) :
  (Real.log a - Real.log c = Real.log (Real.sin B)) →
  (Real.log (Real.sin B) = -Real.log (Real.sqrt 2)) →
  (0 < B) →
  (B < π / 2) →
  (A + B + C = π) →
  (a * Real.sin C = b * Real.sin A) →
  (b * Real.sin C = c * Real.sin B) →
  (c * Real.sin A = a * Real.sin B) →
  (A = π / 4 ∧ B = π / 4 ∧ C = π / 2) :=
by sorry

end triangle_shape_l2073_207328


namespace john_earnings_is_80_l2073_207306

/-- Calculates the amount of money John makes repairing cars --/
def john_earnings (total_cars : ℕ) (standard_repair_time : ℕ) (longer_repair_percentage : ℚ) (hourly_rate : ℚ) : ℚ :=
  let standard_cars := 3
  let longer_cars := total_cars - standard_cars
  let standard_time := standard_cars * standard_repair_time
  let longer_time := longer_cars * (standard_repair_time * (1 + longer_repair_percentage))
  let total_time := standard_time + longer_time
  let total_hours := total_time / 60
  total_hours * hourly_rate

/-- Theorem stating that John makes $80 repairing cars --/
theorem john_earnings_is_80 :
  john_earnings 5 40 (1/2) 20 = 80 := by
  sorry

end john_earnings_is_80_l2073_207306


namespace rectangular_solid_diagonal_angles_l2073_207314

/-- In a rectangular solid, if one of its diagonals forms angles α, β, and γ 
    with the three edges emanating from one of its vertices, 
    then cos²α + cos²β + cos²γ = 1 -/
theorem rectangular_solid_diagonal_angles (α β γ : Real) 
  (hα : α = angle_between_diagonal_and_edge1)
  (hβ : β = angle_between_diagonal_and_edge2)
  (hγ : γ = angle_between_diagonal_and_edge3)
  (h_rectangular_solid : is_rectangular_solid) :
  Real.cos α ^ 2 + Real.cos β ^ 2 + Real.cos γ ^ 2 = 1 :=
by sorry

end rectangular_solid_diagonal_angles_l2073_207314


namespace dividend_divisor_problem_l2073_207305

theorem dividend_divisor_problem (a b : ℕ+) : 
  (a : ℚ) / b = (b : ℚ) + (a : ℚ) / 10 → a = 5 ∧ b = 2 :=
by sorry

end dividend_divisor_problem_l2073_207305


namespace total_elixir_ways_l2073_207381

/-- The number of ways to prepare magical dust -/
def total_magical_dust_ways : ℕ := 4

/-- The number of elixirs made from fairy dust -/
def fairy_dust_elixirs : ℕ := 3

/-- The number of elixirs made from elf dust -/
def elf_dust_elixirs : ℕ := 4

/-- The number of ways to prepare fairy dust -/
def fairy_dust_ways : ℕ := 2

/-- The number of ways to prepare elf dust -/
def elf_dust_ways : ℕ := 2

/-- Theorem: The total number of ways to prepare all the elixirs is 14 -/
theorem total_elixir_ways : 
  fairy_dust_ways * fairy_dust_elixirs + elf_dust_ways * elf_dust_elixirs = 14 :=
by sorry

end total_elixir_ways_l2073_207381


namespace max_rect_box_length_l2073_207394

-- Define the dimensions of the wooden box in centimeters
def wooden_box_length : ℝ := 800
def wooden_box_width : ℝ := 700
def wooden_box_height : ℝ := 600

-- Define the dimensions of the rectangular box in centimeters
def rect_box_width : ℝ := 7
def rect_box_height : ℝ := 6

-- Define the maximum number of rectangular boxes
def max_boxes : ℕ := 2000000

-- Theorem statement
theorem max_rect_box_length :
  ∀ x : ℝ,
  x > 0 →
  (x * rect_box_width * rect_box_height * max_boxes : ℝ) ≤ wooden_box_length * wooden_box_width * wooden_box_height →
  x ≤ 4 := by
sorry


end max_rect_box_length_l2073_207394


namespace eugene_initial_pencils_l2073_207379

/-- The number of pencils Eugene initially had -/
def initial_pencils : ℕ := sorry

/-- The number of pencils Eugene gave to Joyce -/
def pencils_given : ℕ := 6

/-- The number of pencils Eugene has left -/
def pencils_left : ℕ := 45

/-- Theorem: Eugene initially had 51 pencils -/
theorem eugene_initial_pencils :
  initial_pencils = pencils_given + pencils_left ∧ initial_pencils = 51 := by
  sorry

end eugene_initial_pencils_l2073_207379


namespace f_one_zero_implies_a_gt_one_l2073_207332

/-- A function f(x) = 2ax^2 - x - 1 with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := 2*a*x^2 - x - 1

/-- The theorem stating that if f has exactly one zero in (0,1), then a > 1 -/
theorem f_one_zero_implies_a_gt_one (a : ℝ) :
  (∃! x, x ∈ (Set.Ioo 0 1) ∧ f a x = 0) → a > 1 := by
  sorry

#check f_one_zero_implies_a_gt_one

end f_one_zero_implies_a_gt_one_l2073_207332


namespace no_condition_satisfies_equations_l2073_207388

theorem no_condition_satisfies_equations (a b c : ℤ) : 
  a + b + c = 3 →
  (∀ (condition : Prop), 
    (condition = (a = b ∧ b = c ∧ c = 1) ∨
     condition = (a = b - 1 ∧ b = c - 1) ∨
     condition = (a = b ∧ b = c) ∨
     condition = (a > c ∧ c = b - 1)) →
    ¬(condition → a*(a-b)^3 + b*(b-c)^3 + c*(c-a)^3 = 3)) :=
by sorry

end no_condition_satisfies_equations_l2073_207388


namespace last_three_digits_of_7_to_103_l2073_207344

theorem last_three_digits_of_7_to_103 : 7^103 ≡ 614 [ZMOD 1000] := by
  sorry

end last_three_digits_of_7_to_103_l2073_207344


namespace sum_a_d_l2073_207310

theorem sum_a_d (a b c d : ℤ) 
  (h1 : a + b = 5) 
  (h2 : b + c = 6) 
  (h3 : c + d = 3) : 
  a + d = -1 := by
sorry

end sum_a_d_l2073_207310


namespace lcm_of_coprime_product_l2073_207300

theorem lcm_of_coprime_product (a b : ℕ+) (h_coprime : Nat.Coprime a b) (h_product : a * b = 117) :
  Nat.lcm a b = 117 := by
  sorry

end lcm_of_coprime_product_l2073_207300


namespace percent_division_multiplication_equality_l2073_207374

theorem percent_division_multiplication_equality : 
  (30 / 100 : ℚ) / (1 + 2 / 5) * (1 / 3 + 1 / 7) = 5 / 49 := by sorry

end percent_division_multiplication_equality_l2073_207374


namespace equal_area_rectangles_l2073_207390

/-- Given two rectangles with equal area, where one rectangle measures 12 inches by 15 inches
    and the other has a width of 30 inches, the length of the second rectangle is 6 inches. -/
theorem equal_area_rectangles (carol_length carol_width jordan_width : ℝ) 
  (h1 : carol_length = 12)
  (h2 : carol_width = 15)
  (h3 : jordan_width = 30)
  (h4 : carol_length * carol_width = jordan_width * (carol_length * carol_width / jordan_width)) :
  carol_length * carol_width / jordan_width = 6 :=
by sorry

end equal_area_rectangles_l2073_207390


namespace cost_of_paving_floor_l2073_207331

/-- The cost of paving a rectangular floor given its dimensions and rate per square meter. -/
theorem cost_of_paving_floor (length width rate : ℝ) : 
  length = 5.5 → width = 4 → rate = 800 → length * width * rate = 17600 := by
  sorry

end cost_of_paving_floor_l2073_207331


namespace range_of_m_l2073_207370

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 4*x - 2

-- Define the theorem
theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Icc 0 m, f x ∈ Set.Icc (-6) (-2)) ∧
  (∀ y ∈ Set.Icc (-6) (-2), ∃ x ∈ Set.Icc 0 m, f x = y) →
  m ∈ Set.Icc 2 4 :=
sorry

end range_of_m_l2073_207370


namespace parabola_intersection_l2073_207349

theorem parabola_intersection (m : ℝ) : 
  (m > 0) →
  (∃! x : ℝ, -1 < x ∧ x < 4 ∧ -x^2 + 4*x - 2 + m = 0) →
  (2 ≤ m ∧ m < 7) :=
by sorry

end parabola_intersection_l2073_207349


namespace fixed_point_on_line_l2073_207338

/-- The line equation (m-1)x + (2m-1)y = m-5 always passes through the point (9, -4) for any real m -/
theorem fixed_point_on_line (m : ℝ) : (m - 1) * 9 + (2 * m - 1) * (-4) = m - 5 := by
  sorry

end fixed_point_on_line_l2073_207338


namespace total_book_pages_l2073_207308

def book_pages : ℕ → ℕ
| 1 => 25
| 2 => 2 * book_pages 1
| 3 => 2 * book_pages 2
| 4 => 10
| _ => 0

def pages_written : ℕ := book_pages 1 + book_pages 2 + book_pages 3 + book_pages 4

def remaining_pages : ℕ := 315

theorem total_book_pages : pages_written + remaining_pages = 500 := by
  sorry

end total_book_pages_l2073_207308


namespace fruit_juice_volume_l2073_207302

/-- Proves that the volume of fruit juice in Carrie's punch is 40 oz -/
theorem fruit_juice_volume (total_punch : ℕ) (mountain_dew : ℕ) (ice : ℕ) :
  total_punch = 140 ∧ mountain_dew = 72 ∧ ice = 28 →
  ∃ (fruit_juice : ℕ), total_punch = mountain_dew + ice + fruit_juice ∧ fruit_juice = 40 := by
sorry

end fruit_juice_volume_l2073_207302


namespace line_equation_theorem_l2073_207380

-- Define the line l
structure Line where
  slope : ℝ
  passesThrough : ℝ × ℝ
  xIntercept : ℝ
  yIntercept : ℝ

-- Define the conditions
def lineConditions (l : Line) : Prop :=
  l.passesThrough = (2, 3) ∧
  l.slope = Real.tan (2 * Real.pi / 3) ∧
  l.xIntercept + l.yIntercept = 0

-- Define the possible equations of the line
def lineEquation (l : Line) (x y : ℝ) : Prop :=
  (3 * x - 2 * y = 0) ∨ (x - y + 1 = 0)

-- The theorem to prove
theorem line_equation_theorem (l : Line) :
  lineConditions l → ∀ x y, lineEquation l x y :=
sorry

end line_equation_theorem_l2073_207380


namespace inverse_variation_problem_l2073_207373

/-- Given that x varies inversely as square of y, prove that x = 1/9 when y = 6,
    given that y = 2 when x = 1 -/
theorem inverse_variation_problem (x y : ℝ) (k : ℝ) (h1 : x = k / y^2) 
    (h2 : 1 = k / 2^2) : 
  y = 6 → x = 1/9 := by
sorry

end inverse_variation_problem_l2073_207373


namespace binomial_coefficient_19_13_l2073_207301

theorem binomial_coefficient_19_13 
  (h1 : (20 : ℕ).choose 13 = 77520)
  (h2 : (20 : ℕ).choose 14 = 38760)
  (h3 : (18 : ℕ).choose 13 = 18564) :
  (19 : ℕ).choose 13 = 37128 := by
  sorry

end binomial_coefficient_19_13_l2073_207301


namespace fraction_evaluation_l2073_207313

theorem fraction_evaluation : (1 - 1/4) / (1 - 1/3) = 9/8 := by
  sorry

end fraction_evaluation_l2073_207313


namespace largest_rational_root_quadratic_l2073_207330

theorem largest_rational_root_quadratic (a b c : ℕ+) 
  (ha : a ≤ 100) (hb : b ≤ 100) (hc : c ≤ 100) :
  let roots := {x : ℚ | a * x^2 + b * x + c = 0}
  ∃ (max_root : ℚ), max_root ∈ roots ∧ 
    ∀ (r : ℚ), r ∈ roots → r ≤ max_root ∧
    max_root = -1 / 99 := by
  sorry

end largest_rational_root_quadratic_l2073_207330


namespace line_slope_intercept_sum_l2073_207312

/-- Given a line with slope -5 passing through (4, 2), prove that m + b = 17 in y = mx + b -/
theorem line_slope_intercept_sum (m b : ℝ) : 
  m = -5 → 
  2 = m * 4 + b → 
  m + b = 17 := by sorry

end line_slope_intercept_sum_l2073_207312


namespace tangent_roots_expression_value_l2073_207392

theorem tangent_roots_expression_value (α β : Real) : 
  (∃ x y : Real, x^2 - 4*x - 2 = 0 ∧ y^2 - 4*y - 2 = 0 ∧ x ≠ y ∧ Real.tan α = x ∧ Real.tan β = y) →
  (Real.cos (α + β))^2 + 2*(Real.sin (α + β))*(Real.cos (α + β)) - 2*(Real.sin (α + β))^2 = 1/25 := by
  sorry

end tangent_roots_expression_value_l2073_207392


namespace bug_total_distance_l2073_207343

def bug_path : List ℤ := [4, -3, 6, 2]

def distance (a b : ℤ) : ℕ := (a - b).natAbs

theorem bug_total_distance :
  (List.zip bug_path bug_path.tail).foldl (λ acc (a, b) => acc + distance a b) 0 = 20 :=
by sorry

end bug_total_distance_l2073_207343


namespace intersection_of_three_lines_l2073_207336

/-- Given three lines that intersect at the same point, prove the value of k -/
theorem intersection_of_three_lines (x y k : ℚ) :
  (y = 4 * x - 1) ∧
  (y = -3 * x + 9) ∧
  (y = 2 * x + k) →
  k = 13 / 7 := by
  sorry

end intersection_of_three_lines_l2073_207336


namespace probability_no_adjacent_standing_l2073_207383

/-- The number of valid arrangements for n people in a circle where no two adjacent people stand -/
def validArrangements : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | n + 2 => validArrangements (n + 1) + validArrangements n

/-- The total number of possible outcomes when n people flip coins -/
def totalOutcomes (n : ℕ) : ℕ := 2^n

theorem probability_no_adjacent_standing (n : ℕ) : 
  n = 8 → (validArrangements n : ℚ) / totalOutcomes n = 47 / 256 := by
  sorry

end probability_no_adjacent_standing_l2073_207383


namespace root_condition_implies_k_value_l2073_207351

theorem root_condition_implies_k_value (a b c k : ℝ) :
  (∃ x₁ x₂ : ℝ, 
    (x₁^2 - (b+1)*x₁) / ((a+1)*x₁ - c) = (k-2)/(k+2) ∧
    (x₂^2 - (b+1)*x₂) / ((a+1)*x₂ - c) = (k-2)/(k+2) ∧
    x₁ = -x₂ ∧ x₁ ≠ 0) →
  k = (-2*(b-a))/(b+a+2) :=
by sorry

end root_condition_implies_k_value_l2073_207351


namespace sqrt_50_between_consecutive_integers_l2073_207339

theorem sqrt_50_between_consecutive_integers : 
  ∃ n : ℕ, n > 0 ∧ n < Real.sqrt 50 ∧ Real.sqrt 50 < n + 1 ∧ n * (n + 1) = 56 := by
  sorry

end sqrt_50_between_consecutive_integers_l2073_207339


namespace balloon_count_l2073_207350

theorem balloon_count (friend_balloons : ℕ) (difference : ℕ) : 
  friend_balloons = 5 → difference = 2 → friend_balloons + difference = 7 :=
by sorry

end balloon_count_l2073_207350


namespace average_income_q_and_r_l2073_207333

/-- Given the average monthly incomes of P and Q, P and R, and P's income,
    prove that the average monthly income of Q and R is 6250. -/
theorem average_income_q_and_r (p q r : ℕ) : 
  (p + q) / 2 = 5050 →
  (p + r) / 2 = 5200 →
  p = 4000 →
  (q + r) / 2 = 6250 := by
sorry

end average_income_q_and_r_l2073_207333


namespace min_sum_of_log_arithmetic_sequence_l2073_207325

theorem min_sum_of_log_arithmetic_sequence (x y : ℝ) 
  (hx : x > 1) (hy : y > 1) 
  (h_seq : (Real.log x + Real.log y) / 2 = 2) : 
  (∀ a b : ℝ, a > 1 → b > 1 → (Real.log a + Real.log b) / 2 = 2 → x + y ≤ a + b) ∧ 
  ∃ a b : ℝ, a > 1 ∧ b > 1 ∧ (Real.log a + Real.log b) / 2 = 2 ∧ a + b = 200 :=
by sorry

end min_sum_of_log_arithmetic_sequence_l2073_207325


namespace train_length_l2073_207320

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 30 → time = 9 → ∃ length : ℝ, abs (length - 74.97) < 0.01 := by
  sorry

end train_length_l2073_207320


namespace distance_to_FA_l2073_207376

/-- RegularHexagon represents a regular hexagon with a point inside -/
structure RegularHexagon where
  -- Point inside the hexagon
  P : Point
  -- Distances from P to each side
  dist_AB : ℝ
  dist_BC : ℝ
  dist_CD : ℝ
  dist_DE : ℝ
  dist_EF : ℝ
  dist_FA : ℝ

/-- Theorem stating the distance from P to FA in the given hexagon -/
theorem distance_to_FA (h : RegularHexagon)
  (h_AB : h.dist_AB = 1)
  (h_BC : h.dist_BC = 2)
  (h_CD : h.dist_CD = 5)
  (h_DE : h.dist_DE = 7)
  (h_EF : h.dist_EF = 6)
  : h.dist_FA = 3 := by
  sorry

end distance_to_FA_l2073_207376


namespace circle_x_intersection_l2073_207340

theorem circle_x_intersection (x : ℝ) : 
  let center_x := (-2 + 6) / 2
  let center_y := (1 + 9) / 2
  let radius := Real.sqrt (((-2 - center_x)^2 + (1 - center_y)^2) : ℝ)
  (x - center_x)^2 + (0 - center_y)^2 = radius^2 →
  x = 2 + Real.sqrt 7 ∨ x = 2 - Real.sqrt 7 :=
by sorry

end circle_x_intersection_l2073_207340


namespace balloon_arrangements_l2073_207324

-- Define the word length and repeated letter counts
def word_length : ℕ := 7
def l_count : ℕ := 2
def o_count : ℕ := 2

-- Theorem statement
theorem balloon_arrangements : 
  (Nat.factorial word_length) / (Nat.factorial l_count * Nat.factorial o_count) = 1260 :=
by sorry

end balloon_arrangements_l2073_207324


namespace missing_number_proof_l2073_207372

theorem missing_number_proof : ∃ x : ℤ, |7 - 8 * (3 - x)| - |5 - 11| = 73 ∧ x = 12 := by
  sorry

end missing_number_proof_l2073_207372


namespace hyperbola_properties_l2073_207375

/-- Represents a hyperbola in standard form -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  m : ℝ  -- slope
  c : ℝ  -- y-intercept

def distance_to_asymptote (h : Hyperbola) (l : Line) : ℝ := sorry

def standard_equation (h : Hyperbola) : Prop :=
  h.a = 1 ∧ h.b = 2

def slope_ratio (h : Hyperbola) (l : Line) : ℝ := sorry

def fixed_point_exists (h : Hyperbola) : Prop :=
  ∃ (G : Point), G.x = 1 ∧ G.y = 0 ∧
  ∀ (l : Line), slope_ratio h l = -1/3 →
  ∃ (H : Point), (H.x - G.x)^2 + (H.y - G.y)^2 = 1

theorem hyperbola_properties (h : Hyperbola) 
  (asymptote : Line)
  (h_asymptote : asymptote.m = 2 ∧ asymptote.c = 0)
  (h_distance : distance_to_asymptote h asymptote = 2) :
  standard_equation h ∧ fixed_point_exists h := by sorry

end hyperbola_properties_l2073_207375


namespace system_solution_l2073_207384

theorem system_solution (x y k : ℝ) 
  (eq1 : x - y = k + 2)
  (eq2 : x + 3*y = k)
  (eq3 : x + y = 2) :
  k = 1 := by sorry

end system_solution_l2073_207384


namespace trees_in_yard_l2073_207364

theorem trees_in_yard (yard_length : ℕ) (tree_distance : ℕ) (h1 : yard_length = 434) (h2 : tree_distance = 14) :
  (yard_length / tree_distance) + 1 = 32 := by
  sorry

end trees_in_yard_l2073_207364


namespace average_running_time_l2073_207307

theorem average_running_time (sixth_grade_time seventh_grade_time eighth_grade_time : ℝ)
  (sixth_to_eighth_ratio sixth_to_seventh_ratio : ℝ) :
  sixth_grade_time = 10 →
  seventh_grade_time = 18 →
  eighth_grade_time = 14 →
  sixth_to_eighth_ratio = 3 →
  sixth_to_seventh_ratio = 3/2 →
  let e := 1  -- Assuming 1 eighth grader for simplicity
  let sixth_count := e * sixth_to_eighth_ratio
  let seventh_count := sixth_count / sixth_to_seventh_ratio
  let eighth_count := e
  let total_time := sixth_grade_time * sixth_count + 
                    seventh_grade_time * seventh_count + 
                    eighth_grade_time * eighth_count
  let total_students := sixth_count + seventh_count + eighth_count
  total_time / total_students = 40/3 := by
  sorry

end average_running_time_l2073_207307


namespace solution_of_equation_l2073_207382

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the equation
def equation (x : ℂ) : Prop := 2 + i * x = -2 - 2 * i * x

-- State the theorem
theorem solution_of_equation :
  ∃ (x : ℂ), equation x ∧ x = (4 * i) / 3 :=
by sorry

end solution_of_equation_l2073_207382


namespace illegal_parking_percentage_l2073_207367

theorem illegal_parking_percentage
  (total_cars : ℝ)
  (towed_percentage : ℝ)
  (not_towed_percentage : ℝ)
  (h1 : towed_percentage = 0.02)
  (h2 : not_towed_percentage = 0.80)
  (h3 : total_cars > 0) :
  let towed_cars := towed_percentage * total_cars
  let illegally_parked_cars := towed_cars / (1 - not_towed_percentage)
  illegally_parked_cars / total_cars = 0.10 := by
sorry

end illegal_parking_percentage_l2073_207367


namespace arrangement_theorem_l2073_207337

def num_girls : ℕ := 3
def num_boys : ℕ := 5

def arrangements_girls_together : ℕ := 4320
def arrangements_girls_separate : ℕ := 14400

theorem arrangement_theorem :
  (num_girls = 3 ∧ num_boys = 5) →
  (arrangements_girls_together = 4320 ∧ arrangements_girls_separate = 14400) :=
by
  sorry

end arrangement_theorem_l2073_207337


namespace modified_rectangle_areas_l2073_207365

/-- Represents a rectangle with integer dimensions -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℕ := r.length * r.width

/-- The original rectangle -/
def original : Rectangle := { length := 7, width := 5 }

/-- Theorem stating the relationship between the two modified rectangles -/
theorem modified_rectangle_areas :
  ∃ (r1 r2 : Rectangle),
    (r1.length = original.length ∧ r1.width + 2 = original.width ∧ area r1 = 21) →
    (r2.width = original.width ∧ r2.length + 2 = original.length) →
    area r2 = 25 := by
  sorry

end modified_rectangle_areas_l2073_207365


namespace emily_scores_mean_l2073_207378

def emily_scores : List ℕ := [84, 90, 93, 85, 91, 87]

theorem emily_scores_mean : 
  (emily_scores.sum : ℚ) / emily_scores.length = 530 / 6 := by
sorry

end emily_scores_mean_l2073_207378


namespace new_students_count_l2073_207386

/-- Represents the problem of calculating the number of new students joining a school --/
theorem new_students_count (initial_avg_age initial_count new_students_avg_age final_avg_age final_count : ℕ) : 
  initial_avg_age = 48 →
  new_students_avg_age = 32 →
  final_avg_age = 44 →
  final_count = 160 →
  ∃ new_students : ℕ,
    new_students = 40 ∧
    final_count = initial_count + new_students ∧
    final_avg_age * final_count = initial_avg_age * initial_count + new_students_avg_age * new_students :=
by sorry

end new_students_count_l2073_207386


namespace fourth_cubed_decimal_l2073_207359

theorem fourth_cubed_decimal : (1/4)^3 = 0.015625 := by
  sorry

end fourth_cubed_decimal_l2073_207359


namespace lucy_liam_family_theorem_l2073_207348

/-- Represents a family with siblings -/
structure Family where
  girls : Nat
  boys : Nat

/-- Calculates the number of sisters and brothers for a sibling in the family -/
def sibling_count (f : Family) : Nat × Nat :=
  (f.girls, f.boys - 1)

/-- The main theorem about Lucy and Liam's family -/
theorem lucy_liam_family_theorem : 
  ∀ (f : Family), 
  f.girls = 5 → f.boys = 7 → 
  let (s, b) := sibling_count f
  s * b = 25 := by
  sorry

#check lucy_liam_family_theorem

end lucy_liam_family_theorem_l2073_207348


namespace product_sum_relation_l2073_207321

theorem product_sum_relation (a b : ℝ) : 
  a * b = 2 * (a + b) + 11 → b = 7 → b - a = 2 := by
  sorry

end product_sum_relation_l2073_207321


namespace tan_P_is_two_one_l2073_207322

/-- Represents a right triangle PQR with altitude QS --/
structure RightTrianglePQR where
  -- Side lengths
  PQ : ℕ
  QR : ℕ
  PR : ℕ
  PS : ℕ
  -- PR = 3^5
  h_PR : PR = 3^5
  -- PS = 3^3
  h_PS : PS = 3^3
  -- Right angle at Q
  h_right_angle : PQ^2 + QR^2 = PR^2
  -- Altitude property
  h_altitude : PQ * PS = PR * QS

/-- The main theorem --/
theorem tan_P_is_two_one (t : RightTrianglePQR) : 
  (t.QR : ℚ) / t.PQ = 2 / 1 := by
  sorry

end tan_P_is_two_one_l2073_207322


namespace sufficient_not_necessary_condition_l2073_207356

theorem sufficient_not_necessary_condition (x : ℝ) :
  (x < -1 → 2*x^2 + x - 1 > 0) ∧ 
  ¬(2*x^2 + x - 1 > 0 → x < -1) := by
  sorry

end sufficient_not_necessary_condition_l2073_207356


namespace trader_gain_percentage_l2073_207385

theorem trader_gain_percentage : 
  ∀ (cost_per_pen : ℝ), cost_per_pen > 0 →
  (19 * cost_per_pen) / (95 * cost_per_pen) * 100 = 20 := by
  sorry

end trader_gain_percentage_l2073_207385


namespace equation1_solution_equation2_no_solution_l2073_207319

-- Define the equations
def equation1 (x : ℝ) : Prop := (3 / (x + 1)) = (2 / (x - 1))
def equation2 (x : ℝ) : Prop := (2 * x + 9) / (3 * x - 9) = (4 * x - 7) / (x - 3) + 2

-- Theorem for equation 1
theorem equation1_solution : 
  ∃! x : ℝ, equation1 x ∧ x ≠ -1 ∧ x ≠ 1 := by sorry

-- Theorem for equation 2
theorem equation2_no_solution : 
  ∀ x : ℝ, ¬(equation2 x ∧ x ≠ 3) := by sorry

end equation1_solution_equation2_no_solution_l2073_207319


namespace runner_time_difference_l2073_207369

theorem runner_time_difference 
  (x y : ℝ) 
  (h1 : y - x / 2 = 12) 
  (h2 : x - y / 2 = 36) : 
  2 * y - 2 * x = -16 := by
sorry

end runner_time_difference_l2073_207369


namespace correct_algebraic_operation_l2073_207377

variable (x y : ℝ)

theorem correct_algebraic_operation : y * x - 3 * x * y = -2 * x * y := by
  sorry

end correct_algebraic_operation_l2073_207377


namespace kannon_fruit_consumption_l2073_207345

/-- Represents the number of fruits Kannon ate last night -/
structure LastNightFruits where
  apples : ℕ
  bananas : ℕ
  oranges : ℕ
  strawberries : ℕ
  kiwis : ℕ

/-- Represents the number of fruits Kannon will eat today -/
structure TodayFruits where
  apples : ℕ
  bananas : ℕ
  oranges : ℕ
  strawberries : ℕ
  kiwis : ℕ

/-- Calculates the total number of fruits eaten over two days -/
def totalFruits (last : LastNightFruits) (today : TodayFruits) : ℕ :=
  last.apples + last.bananas + last.oranges + last.strawberries + last.kiwis +
  today.apples + today.bananas + today.oranges + today.strawberries + today.kiwis

/-- Theorem stating that the total number of fruits eaten is 54 -/
theorem kannon_fruit_consumption :
  ∀ (last : LastNightFruits) (today : TodayFruits),
  last.apples = 3 ∧ last.bananas = 1 ∧ last.oranges = 4 ∧ last.strawberries = 2 ∧ last.kiwis = 3 →
  today.apples = last.apples + 4 →
  today.bananas = 10 * last.bananas →
  today.oranges = 2 * today.apples →
  today.strawberries = (3 * last.oranges) / 2 →
  today.kiwis = today.bananas - 3 →
  totalFruits last today = 54 := by
  sorry


end kannon_fruit_consumption_l2073_207345


namespace stratified_sample_female_count_l2073_207395

/-- Represents the number of female athletes in a stratified sample -/
def female_athletes_in_sample (total_athletes : ℕ) (female_athletes : ℕ) (sample_size : ℕ) : ℕ :=
  (female_athletes * sample_size) / total_athletes

theorem stratified_sample_female_count :
  female_athletes_in_sample 98 42 28 = 12 := by
  sorry

#eval female_athletes_in_sample 98 42 28

end stratified_sample_female_count_l2073_207395


namespace kelly_carrots_l2073_207397

/-- The number of carrots Kelly pulled out from the first bed -/
def carrots_first_bed (total_carrots second_bed third_bed : ℕ) : ℕ :=
  total_carrots - second_bed - third_bed

/-- Theorem stating the number of carrots Kelly pulled out from the first bed -/
theorem kelly_carrots :
  carrots_first_bed (39 * 6) 101 78 = 55 := by
  sorry

end kelly_carrots_l2073_207397


namespace twenty_fifth_term_of_sequence_l2073_207398

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

theorem twenty_fifth_term_of_sequence : 
  let a₁ := 2
  let a₂ := 5
  let d := a₂ - a₁
  arithmetic_sequence a₁ d 25 = 74 := by
sorry

end twenty_fifth_term_of_sequence_l2073_207398


namespace min_product_constrained_l2073_207353

theorem min_product_constrained (x y z : ℝ) : 
  x > 0 → y > 0 → z > 0 → 
  x + y + z = 1 → 
  x ≤ 3*y ∧ x ≤ 3*z ∧ y ≤ 3*x ∧ y ≤ 3*z ∧ z ≤ 3*x ∧ z ≤ 3*y → 
  x * y * z ≥ 3/125 := by
sorry

end min_product_constrained_l2073_207353


namespace distance_between_vertices_l2073_207329

-- Define the equation of the graph
def graph_equation (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + y^2) + |y + 2| = 4

-- Define the parabolas
def parabola1 (x y : ℝ) : Prop :=
  graph_equation x y ∧ y ≥ -2

def parabola2 (x y : ℝ) : Prop :=
  graph_equation x y ∧ y < -2

-- Define the vertices
def vertex1 : ℝ × ℝ := (0, 1)
def vertex2 : ℝ × ℝ := (0, -3)

-- Theorem statement
theorem distance_between_vertices :
  ∃ (v1 v2 : ℝ × ℝ),
    (∀ x y, parabola1 x y → (x, y) = v1 ∨ y > v1.2) ∧
    (∀ x y, parabola2 x y → (x, y) = v2 ∨ y < v2.2) ∧
    ‖v1 - v2‖ = 4 :=
  sorry

end distance_between_vertices_l2073_207329


namespace distance_to_focus_l2073_207387

def parabola (x y : ℝ) : Prop := y^2 = 4*x

def focus (x y : ℝ) : Prop := x = 1 ∧ y = 0

theorem distance_to_focus (x y : ℝ) (h1 : parabola x y) (h2 : x = 3) : 
  ∃ (fx fy : ℝ), focus fx fy ∧ Real.sqrt ((x - fx)^2 + (y - fy)^2) = 4 :=
sorry

end distance_to_focus_l2073_207387


namespace toothpick_pattern_sum_l2073_207316

def arithmeticSequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

def arithmeticSum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem toothpick_pattern_sum :
  arithmeticSum 6 5 150 = 56775 := by sorry

end toothpick_pattern_sum_l2073_207316


namespace smallest_shift_l2073_207355

-- Define the function g with the given property
def g_periodic (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (x - 15) = g x

-- Define the property for the shifted function
def shifted_function_equal (g : ℝ → ℝ) (b : ℝ) : Prop :=
  ∀ x : ℝ, g ((x - b) / 3) = g (x / 3)

-- Theorem statement
theorem smallest_shift (g : ℝ → ℝ) :
  g_periodic g →
  (∃ b : ℝ, b > 0 ∧ shifted_function_equal g b ∧
    ∀ b' : ℝ, b' > 0 ∧ shifted_function_equal g b' → b ≤ b') →
  (∃ b : ℝ, b = 45 ∧ shifted_function_equal g b) :=
sorry

end smallest_shift_l2073_207355


namespace expression_value_l2073_207393

theorem expression_value (x y z w : ℝ) 
  (h1 : 4 * x * z + y * w = 3) 
  (h2 : x * w + y * z = 6) : 
  (2 * x + y) * (2 * z + w) = 15 := by
  sorry

end expression_value_l2073_207393


namespace distinct_intersection_points_l2073_207360

/-- A line in a plane -/
structure Line :=
  (id : ℕ)

/-- A point where at least two lines intersect -/
structure IntersectionPoint :=
  (lines : Finset Line)

/-- The set of all lines in the plane -/
def all_lines : Finset Line := sorry

/-- The set of all intersection points -/
def intersection_points : Finset IntersectionPoint := sorry

theorem distinct_intersection_points :
  (∀ l ∈ all_lines, ∀ l' ∈ all_lines, l ≠ l' → l.id ≠ l'.id) →  -- lines are distinct
  (Finset.card all_lines = 5) →  -- there are five lines
  (∀ p ∈ intersection_points, Finset.card p.lines ≥ 2) →  -- each intersection point has at least two lines
  (∀ p ∈ intersection_points, Finset.card p.lines ≤ 3) →  -- no more than three lines intersect at a point
  Finset.card intersection_points = 10 :=  -- there are 10 distinct intersection points
by sorry

end distinct_intersection_points_l2073_207360


namespace cos_double_angle_special_case_l2073_207399

/-- Given a vector a = (cos α, 1/2) with magnitude √2/2, prove that cos 2α = -1/2 -/
theorem cos_double_angle_special_case (α : ℝ) (a : ℝ × ℝ) :
  a = (Real.cos α, (1 : ℝ) / 2) →
  Real.sqrt ((a.1 ^ 2) + (a.2 ^ 2)) = Real.sqrt 2 / 2 →
  Real.cos (2 * α) = -(1 : ℝ) / 2 := by
  sorry

end cos_double_angle_special_case_l2073_207399


namespace sequence_a_correct_l2073_207391

def sequence_a (n : ℕ) : ℚ :=
  (2 * 3^n) / (3^n - 1)

theorem sequence_a_correct (n : ℕ) : 
  n ≥ 1 → 
  sequence_a (n + 1) = (3^(n + 1) * sequence_a n) / (sequence_a n + 3^(n + 1)) ∧
  sequence_a 1 = 3 := by
sorry

end sequence_a_correct_l2073_207391


namespace problem_statement_l2073_207323

theorem problem_statement (x y : ℝ) (h1 : x + y = 5) (h2 : x * y = 3) :
  x + (x^3 / y^2) + (y^3 / x^2) + y = 174 := by
  sorry

end problem_statement_l2073_207323


namespace max_interval_length_l2073_207362

theorem max_interval_length (a b : ℝ) (h1 : a < 0) 
  (h2 : ∀ x ∈ Set.Ioo a b, (3 * x^2 + a) * (2 * x + b) ≥ 0) : 
  (b - a) ≤ 1/3 :=
sorry

end max_interval_length_l2073_207362


namespace trig_expression_equals_three_halves_l2073_207363

theorem trig_expression_equals_three_halves :
  (Real.sin (30 * π / 180) - 1) ^ 0 - Real.sqrt 2 * Real.sin (45 * π / 180) +
  Real.tan (60 * π / 180) * Real.cos (30 * π / 180) = 3 / 2 := by
  sorry

end trig_expression_equals_three_halves_l2073_207363


namespace statement_1_statement_4_l2073_207354

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (line_parallel_plane : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- Define the axioms
variable (m n : Line)
variable (α β : Plane)
variable (h_different_lines : m ≠ n)
variable (h_non_coincident_planes : α ≠ β)

-- Statement 1
theorem statement_1 : 
  perpendicular m α → perpendicular n α → parallel m n :=
sorry

-- Statement 4
theorem statement_4 :
  perpendicular m α → line_parallel_plane m β → plane_perpendicular α β :=
sorry

end statement_1_statement_4_l2073_207354


namespace smallest_k_proof_l2073_207357

def is_perfect_cube (m : ℤ) : Prop := ∃ n : ℤ, m = n^3

def smallest_k : ℕ := 60

theorem smallest_k_proof : 
  (∀ k : ℕ, k < smallest_k → ¬ is_perfect_cube (2^4 * 3^2 * 5^5 * k)) ∧ 
  is_perfect_cube (2^4 * 3^2 * 5^5 * smallest_k) :=
sorry

end smallest_k_proof_l2073_207357


namespace increasing_sequence_condition_l2073_207326

def sequence_term (n : ℕ) (b : ℝ) : ℝ := n^2 + b*n

theorem increasing_sequence_condition (b : ℝ) : 
  (∀ n : ℕ, sequence_term (n + 1) b > sequence_term n b) → b > -3 :=
by
  sorry

#check increasing_sequence_condition

end increasing_sequence_condition_l2073_207326


namespace cube_root_sum_equation_l2073_207335

theorem cube_root_sum_equation (y : ℝ) (hy : y > 0) 
  (h : Real.rpow (2 - y^3) (1/3) + Real.rpow (2 + y^3) (1/3) = 2) : 
  y^6 = 116/27 := by sorry

end cube_root_sum_equation_l2073_207335


namespace quadratic_one_solution_quadratic_one_solution_positive_l2073_207304

/-- The positive value of m for which the quadratic equation 9x^2 + mx + 36 = 0 has exactly one solution -/
theorem quadratic_one_solution (m : ℝ) : 
  (∃! x : ℝ, 9 * x^2 + m * x + 36 = 0) → m = 36 ∨ m = -36 :=
by sorry

/-- The positive value of m for which the quadratic equation 9x^2 + mx + 36 = 0 has exactly one solution is 36 -/
theorem quadratic_one_solution_positive (m : ℝ) : 
  (∃! x : ℝ, 9 * x^2 + m * x + 36 = 0) ∧ m > 0 → m = 36 :=
by sorry

end quadratic_one_solution_quadratic_one_solution_positive_l2073_207304


namespace condition_necessary_not_sufficient_l2073_207347

/-- Defines whether an equation represents an ellipse -/
def IsEllipse (m : ℝ) : Prop :=
  (m - 1 > 0) ∧ (3 - m > 0) ∧ (m - 1 ≠ 3 - m)

/-- The condition on m -/
def Condition (m : ℝ) : Prop :=
  1 < m ∧ m < 3

/-- Theorem stating that the condition is necessary but not sufficient -/
theorem condition_necessary_not_sufficient :
  (∀ m : ℝ, IsEllipse m → Condition m) ∧
  (∃ m : ℝ, Condition m ∧ ¬IsEllipse m) :=
sorry

end condition_necessary_not_sufficient_l2073_207347


namespace expression_evaluation_l2073_207366

theorem expression_evaluation :
  let a : ℚ := -3/2
  let expr := 1 + (1 - a) / a / ((a^2 - 1) / (a^2 + 2*a))
  expr = 2 := by sorry

end expression_evaluation_l2073_207366


namespace product_inequality_l2073_207311

theorem product_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum : a + b + c + 2 = a * b * c) :
  (a + 1) * (b + 1) * (c + 1) ≥ 27 ∧
  ((a + 1) * (b + 1) * (c + 1) = 27 ↔ a = 2 ∧ b = 2 ∧ c = 2) := by
  sorry

end product_inequality_l2073_207311


namespace prob_white_given_red_is_two_ninths_l2073_207361

/-- The number of red balls in the box -/
def num_red : ℕ := 3

/-- The number of white balls in the box -/
def num_white : ℕ := 2

/-- The number of black balls in the box -/
def num_black : ℕ := 5

/-- The total number of balls in the box -/
def total_balls : ℕ := num_red + num_white + num_black

/-- The probability of picking a white ball on the second draw given that the first ball picked is red -/
def prob_white_given_red : ℚ := num_white / (total_balls - 1)

theorem prob_white_given_red_is_two_ninths :
  prob_white_given_red = 2 / 9 := by
  sorry

end prob_white_given_red_is_two_ninths_l2073_207361


namespace parallel_line_distance_l2073_207358

/-- Represents a circle intersected by three equally spaced parallel lines -/
structure CircleWithParallelLines where
  /-- The radius of the circle -/
  radius : ℝ
  /-- The distance between two adjacent parallel lines -/
  line_distance : ℝ
  /-- The length of the first chord -/
  chord1 : ℝ
  /-- The length of the second chord -/
  chord2 : ℝ
  /-- The length of the third chord -/
  chord3 : ℝ
  /-- The first chord has length 30 -/
  chord1_eq : chord1 = 30
  /-- The second chord has length 40 -/
  chord2_eq : chord2 = 40
  /-- The third chord has length 30 -/
  chord3_eq : chord3 = 30

/-- Theorem: The distance between two adjacent parallel lines is 2√30 -/
theorem parallel_line_distance (c : CircleWithParallelLines) :
  c.line_distance = 2 * Real.sqrt 30 := by
  sorry

end parallel_line_distance_l2073_207358


namespace field_trip_passengers_l2073_207315

/-- The number of passengers a single bus can transport -/
def passengers_per_bus : ℕ := 48

/-- The number of buses needed for the field trip -/
def buses_needed : ℕ := 26

/-- The total number of passengers (students and teachers) going on the field trip -/
def total_passengers : ℕ := passengers_per_bus * buses_needed

theorem field_trip_passengers :
  total_passengers = 1248 :=
sorry

end field_trip_passengers_l2073_207315


namespace x_equation_result_l2073_207303

theorem x_equation_result (x : ℝ) (h : x + 1/x = Real.sqrt 3) :
  x^7 - 3*x^5 + x^2 = -5*x + 4*Real.sqrt 3 := by
  sorry

end x_equation_result_l2073_207303


namespace non_trivial_solution_exists_l2073_207334

theorem non_trivial_solution_exists (a b c : ℤ) (p : ℕ) (hp : Nat.Prime p) :
  ∃ x y z : ℤ, (x, y, z) ≠ (0, 0, 0) ∧ (a * x^2 + b * y^2 + c * z^2) % p = 0 := by
  sorry

end non_trivial_solution_exists_l2073_207334


namespace negation_of_existence_negation_of_proposition_l2073_207318

theorem negation_of_existence (p : ℝ → Prop) :
  (¬∃ x, p x) ↔ (∀ x, ¬p x) :=
by sorry

theorem negation_of_proposition :
  (¬∃ x : ℝ, x^2 - x + 1 < 0) ↔ (∀ x : ℝ, x^2 - x + 1 ≥ 0) :=
by sorry

end negation_of_existence_negation_of_proposition_l2073_207318


namespace coin_flip_sequences_l2073_207341

/-- The number of flips performed -/
def num_flips : ℕ := 10

/-- The number of possible outcomes for each flip -/
def outcomes_per_flip : ℕ := 2

/-- The total number of distinct sequences possible -/
def total_sequences : ℕ := outcomes_per_flip ^ num_flips

theorem coin_flip_sequences :
  total_sequences = 1024 := by
  sorry

end coin_flip_sequences_l2073_207341


namespace sum_plus_count_theorem_l2073_207368

def sum_of_integers (a b : ℕ) : ℕ := ((b - a + 1) * (a + b)) / 2

def count_even_integers (a b : ℕ) : ℕ := ((b - a) / 2) + 1

theorem sum_plus_count_theorem : 
  sum_of_integers 50 70 + count_even_integers 50 70 = 1271 := by
  sorry

end sum_plus_count_theorem_l2073_207368


namespace prob_standard_bulb_l2073_207396

/-- Probability of selecting a light bulb from the first factory -/
def p_factory1 : ℝ := 0.2

/-- Probability of selecting a light bulb from the second factory -/
def p_factory2 : ℝ := 0.3

/-- Probability of selecting a light bulb from the third factory -/
def p_factory3 : ℝ := 0.5

/-- Probability of producing a defective light bulb in the first factory -/
def q1 : ℝ := 0.01

/-- Probability of producing a defective light bulb in the second factory -/
def q2 : ℝ := 0.005

/-- Probability of producing a defective light bulb in the third factory -/
def q3 : ℝ := 0.006

/-- Theorem: The probability of randomly selecting a standard (non-defective) light bulb -/
theorem prob_standard_bulb : 
  p_factory1 * (1 - q1) + p_factory2 * (1 - q2) + p_factory3 * (1 - q3) = 0.9935 := by
  sorry

end prob_standard_bulb_l2073_207396
