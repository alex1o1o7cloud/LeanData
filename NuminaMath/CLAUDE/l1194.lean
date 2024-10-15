import Mathlib

namespace NUMINAMATH_CALUDE_frog_ratio_l1194_119483

/-- Given two ponds A and B with frogs, prove the ratio of frogs in A to B -/
theorem frog_ratio (total : ℕ) (pond_a : ℕ) (h1 : total = 48) (h2 : pond_a = 32) :
  (pond_a : ℚ) / ((total - pond_a) : ℚ) = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_frog_ratio_l1194_119483


namespace NUMINAMATH_CALUDE_square_sum_range_l1194_119461

theorem square_sum_range (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hsum : x + y = 1) :
  1/2 ≤ x^2 + y^2 ∧ x^2 + y^2 ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_range_l1194_119461


namespace NUMINAMATH_CALUDE_solve_linear_equation_l1194_119445

theorem solve_linear_equation (x : ℝ) (h : 7 - x = 12) : x = -5 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l1194_119445


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l1194_119446

/-- Given two vectors a and b in ℝ², if a is parallel to b and a = (1, -2) and b = (x, 1), then x = -1/2 -/
theorem parallel_vectors_x_value (a b : ℝ × ℝ) (x : ℝ) 
  (h1 : a = (1, -2))
  (h2 : b = (x, 1))
  (h_parallel : ∃ (k : ℝ), a = k • b) :
  x = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l1194_119446


namespace NUMINAMATH_CALUDE_quadratic_inequality_roots_l1194_119402

theorem quadratic_inequality_roots (k : ℝ) : 
  (∀ x : ℝ, -x^2 + k*x + 4 < 0 ↔ x < 2 ∨ x > 3) → k = 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_roots_l1194_119402


namespace NUMINAMATH_CALUDE_x_plus_2y_equals_10_l1194_119429

theorem x_plus_2y_equals_10 (x y : ℝ) (hx : x = 4) (hy : y = 3) : x + 2*y = 10 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_2y_equals_10_l1194_119429


namespace NUMINAMATH_CALUDE_tiffany_monday_bags_l1194_119437

/-- The number of bags Tiffany had on Monday -/
def monday_bags : ℕ := sorry

/-- The number of bags Tiffany found on Tuesday -/
def tuesday_bags : ℕ := 3

/-- The number of bags Tiffany found on Wednesday -/
def wednesday_bags : ℕ := 7

/-- The total number of bags Tiffany had -/
def total_bags : ℕ := 20

/-- Theorem stating that Tiffany had 10 bags on Monday -/
theorem tiffany_monday_bags : 
  monday_bags + tuesday_bags + wednesday_bags = total_bags ∧ monday_bags = 10 := by
  sorry

end NUMINAMATH_CALUDE_tiffany_monday_bags_l1194_119437


namespace NUMINAMATH_CALUDE_triangle_theorem_l1194_119400

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  angle_sum : A + B + C = π
  law_of_sines : a / Real.sin A = b / Real.sin B
  law_of_cosines : a^2 = b^2 + c^2 - 2*b*c*Real.cos A

/-- The main theorem -/
theorem triangle_theorem (t : Triangle) : 
  (2 * t.b = t.c + 2 * t.a * Real.cos t.C) → 
  (t.A = π / 3 ∧ 
   (1/2 * t.b * t.c * Real.sin t.A = 4 * Real.sqrt 3 / 3 ∧ t.a = 3 → 
    t.a + t.b + t.c = 8)) := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l1194_119400


namespace NUMINAMATH_CALUDE_true_propositions_l1194_119490

-- Define proposition p
def p : Prop := ∀ x y : ℝ, x > y → -x < -y

-- Define proposition q
def q : Prop := ∀ x y : ℝ, x < y → x^2 > y^2

-- Define the four compound propositions
def prop1 : Prop := p ∧ q
def prop2 : Prop := p ∨ q
def prop3 : Prop := p ∧ (¬q)
def prop4 : Prop := (¬p) ∨ q

-- Theorem stating which propositions are true
theorem true_propositions : prop2 ∧ prop3 ∧ ¬prop1 ∧ ¬prop4 := by
  sorry

end NUMINAMATH_CALUDE_true_propositions_l1194_119490


namespace NUMINAMATH_CALUDE_cubic_root_sum_l1194_119479

theorem cubic_root_sum (r s t : ℝ) : 
  r^3 - 15*r^2 + 13*r - 8 = 0 →
  s^3 - 15*s^2 + 13*s - 8 = 0 →
  t^3 - 15*t^2 + 13*t - 8 = 0 →
  r / (1/r + s*t) + s / (1/s + r*t) + t / (1/t + r*s) = 199/9 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l1194_119479


namespace NUMINAMATH_CALUDE_consecutive_integers_product_sum_l1194_119435

theorem consecutive_integers_product_sum (n : ℕ) : 
  n > 0 ∧ n * (n + 1) * (n + 2) * (n + 3) = 358800 → 
  n + (n + 1) + (n + 2) + (n + 3) = 98 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_sum_l1194_119435


namespace NUMINAMATH_CALUDE_gcd_consecutive_terms_unbounded_l1194_119470

def a (n : ℕ) : ℤ := n.factorial - n

theorem gcd_consecutive_terms_unbounded :
  ∀ M : ℕ, ∃ n : ℕ, Int.gcd (a n) (a (n + 1)) > M :=
sorry

end NUMINAMATH_CALUDE_gcd_consecutive_terms_unbounded_l1194_119470


namespace NUMINAMATH_CALUDE_equal_playing_time_l1194_119413

theorem equal_playing_time (total_players : ℕ) (players_on_field : ℕ) (match_duration : ℕ) :
  total_players = 10 →
  players_on_field = 8 →
  match_duration = 45 →
  (players_on_field * match_duration) % total_players = 0 →
  (players_on_field * match_duration) / total_players = 36 := by
  sorry

end NUMINAMATH_CALUDE_equal_playing_time_l1194_119413


namespace NUMINAMATH_CALUDE_point_transformation_identity_l1194_119456

def rotateZ90 (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (-y, x, z)

def reflectXY (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (x, y, -z)

def rotateX90 (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (x, -z, y)

def reflectYZ (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (-x, y, z)

theorem point_transformation_identity :
  let initial_point : ℝ × ℝ × ℝ := (2, 2, 2)
  let transformed_point := reflectYZ (rotateX90 (reflectXY (rotateZ90 initial_point)))
  transformed_point = initial_point := by
  sorry

end NUMINAMATH_CALUDE_point_transformation_identity_l1194_119456


namespace NUMINAMATH_CALUDE_boxes_per_crate_l1194_119407

theorem boxes_per_crate 
  (total_crates : ℕ) 
  (original_machines_per_box : ℕ) 
  (machines_removed_per_box : ℕ) 
  (total_machines_removed : ℕ) 
  (h1 : total_crates = 10)
  (h2 : original_machines_per_box = 4)
  (h3 : machines_removed_per_box = 1)
  (h4 : total_machines_removed = 60) :
  total_machines_removed / total_crates = 6 := by
sorry

end NUMINAMATH_CALUDE_boxes_per_crate_l1194_119407


namespace NUMINAMATH_CALUDE_sqrt_neg_four_squared_l1194_119431

theorem sqrt_neg_four_squared : Real.sqrt ((-4)^2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_neg_four_squared_l1194_119431


namespace NUMINAMATH_CALUDE_parallel_lines_equal_angles_with_plane_l1194_119415

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation for lines
variable (parallel : Line → Line → Prop)

-- Define the relation for a line forming an angle with a plane
variable (angle_with_plane : Line → Plane → ℝ)

-- State the theorem
theorem parallel_lines_equal_angles_with_plane
  (m n : Line) (α : Plane) :
  (parallel m n → angle_with_plane m α = angle_with_plane n α) ∧
  ¬(angle_with_plane m α = angle_with_plane n α → parallel m n) :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_equal_angles_with_plane_l1194_119415


namespace NUMINAMATH_CALUDE_park_birds_difference_l1194_119454

/-- The number of geese and ducks remaining at a park after some changes. -/
theorem park_birds_difference (initial_ducks : ℕ) (geese_leave : ℕ) : 
  let initial_geese := 2 * initial_ducks - 10
  let final_ducks := initial_ducks + 4
  let final_geese := initial_geese - (15 - 5)
  final_geese - final_ducks = 1 :=
by sorry

end NUMINAMATH_CALUDE_park_birds_difference_l1194_119454


namespace NUMINAMATH_CALUDE_complex_magnitude_difference_zero_l1194_119401

theorem complex_magnitude_difference_zero : Complex.abs (3 - 5*Complex.I) - Complex.abs (3 + 5*Complex.I) = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_difference_zero_l1194_119401


namespace NUMINAMATH_CALUDE_most_likely_red_balls_l1194_119452

theorem most_likely_red_balls
  (total_balls : ℕ)
  (red_probability : ℚ)
  (h_total : total_balls = 20)
  (h_prob : red_probability = 1/5) :
  (red_probability * total_balls : ℚ) = 4 := by
sorry

end NUMINAMATH_CALUDE_most_likely_red_balls_l1194_119452


namespace NUMINAMATH_CALUDE_max_value_cos_sin_l1194_119458

theorem max_value_cos_sin (x : ℝ) : 2 * Real.cos x + 3 * Real.sin x ≤ Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_max_value_cos_sin_l1194_119458


namespace NUMINAMATH_CALUDE_mathborough_rainfall_2004_l1194_119417

/-- The total rainfall in Mathborough for the year 2004 -/
def total_rainfall_2004 (avg_rainfall_2003 : ℕ) (rainfall_increase : ℕ) : ℕ := 
  let avg_rainfall_2004 := avg_rainfall_2003 + rainfall_increase
  let feb_rainfall := avg_rainfall_2004 * 29
  let other_months_rainfall := avg_rainfall_2004 * 30 * 11
  feb_rainfall + other_months_rainfall

/-- Theorem stating the total rainfall in Mathborough for 2004 -/
theorem mathborough_rainfall_2004 : 
  total_rainfall_2004 50 3 = 19027 := by
sorry

end NUMINAMATH_CALUDE_mathborough_rainfall_2004_l1194_119417


namespace NUMINAMATH_CALUDE_expression_equals_seven_l1194_119438

theorem expression_equals_seven (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (sum_zero : a + b + c = 0) (sum_prod_nonzero : a * b + a * c + b * c ≠ 0) :
  (a^7 + b^7 + c^7) / (a * b * c * (a * b + a * c + b * c)) = 7 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_seven_l1194_119438


namespace NUMINAMATH_CALUDE_no_solution_lcm_gcd_equation_l1194_119441

theorem no_solution_lcm_gcd_equation : 
  ∀ n : ℕ+, n.lcm 200 ≠ n.gcd 200 + 1000 := by
sorry

end NUMINAMATH_CALUDE_no_solution_lcm_gcd_equation_l1194_119441


namespace NUMINAMATH_CALUDE_butterfly_failure_rate_l1194_119474

theorem butterfly_failure_rate 
  (total_caterpillars : ℕ) 
  (butterfly_price : ℚ) 
  (total_revenue : ℚ) : 
  total_caterpillars = 40 →
  butterfly_price = 3 →
  total_revenue = 72 →
  (total_caterpillars - (total_revenue / butterfly_price)) / total_caterpillars * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_butterfly_failure_rate_l1194_119474


namespace NUMINAMATH_CALUDE_express_delivery_growth_rate_l1194_119404

theorem express_delivery_growth_rate 
  (initial_revenue : ℝ)
  (final_revenue : ℝ)
  (years : ℕ)
  (h1 : initial_revenue = 400)
  (h2 : final_revenue = 576)
  (h3 : years = 2) :
  ∃ (growth_rate : ℝ), 
    growth_rate = 0.2 ∧ 
    initial_revenue * (1 + growth_rate) ^ years = final_revenue :=
sorry

end NUMINAMATH_CALUDE_express_delivery_growth_rate_l1194_119404


namespace NUMINAMATH_CALUDE_part_one_part_two_l1194_119465

-- Define the function f
def f (b : ℝ) (x : ℝ) : ℝ := |2 * x + b|

-- Part I
theorem part_one (b : ℝ) : 
  (∀ x, f b x ≤ 3 ↔ -1 ≤ x ∧ x ≤ 2) → b = -1 := by sorry

-- Part II
theorem part_two :
  ∃ m : ℝ, ∀ x : ℝ, f (-1) (x + 3) + f (-1) (x + 1) ≥ m ∧ 
  ¬∃ m' : ℝ, (m' < m ∧ ∀ x : ℝ, f (-1) (x + 3) + f (-1) (x + 1) ≥ m') := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1194_119465


namespace NUMINAMATH_CALUDE_fraction_equality_l1194_119467

theorem fraction_equality (a b c d : ℚ) 
  (h1 : a / b = 2 / 3) 
  (h2 : c / b = 1 / 5) 
  (h3 : c / d = 7 / 15) : 
  a * b / (c * d) = 140 / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1194_119467


namespace NUMINAMATH_CALUDE_triangular_number_200_l1194_119466

/-- Triangular number sequence -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The 200th triangular number is 20100 -/
theorem triangular_number_200 : triangular_number 200 = 20100 := by
  sorry

end NUMINAMATH_CALUDE_triangular_number_200_l1194_119466


namespace NUMINAMATH_CALUDE_masha_wins_l1194_119495

/-- Represents a pile of candies -/
structure Pile :=
  (size : ℕ)

/-- Represents the game state -/
structure GameState :=
  (piles : List Pile)

/-- Calculates the number of moves required to split a pile into single candies -/
def movesForPile (p : Pile) : ℕ :=
  p.size - 1

/-- Calculates the total number of moves for all piles -/
def totalMoves (gs : GameState) : ℕ :=
  gs.piles.map movesForPile |>.sum

/-- Determines if the first player wins given a game state -/
def firstPlayerWins (gs : GameState) : Prop :=
  Odd (totalMoves gs)

/-- Theorem: Masha (first player) wins the candy splitting game -/
theorem masha_wins :
  let initialState : GameState := ⟨[⟨10⟩, ⟨20⟩, ⟨30⟩]⟩
  firstPlayerWins initialState := by
  sorry


end NUMINAMATH_CALUDE_masha_wins_l1194_119495


namespace NUMINAMATH_CALUDE_cnc_machine_profit_l1194_119408

/-- Represents the profit function for a CNC machine -/
def profit_function (x : ℕ+) : ℤ := -2 * x.val ^ 2 + 40 * x.val - 98

/-- Represents when the machine starts generating profit -/
def profit_start : ℕ+ := 3

/-- Represents the year of maximum average annual profit -/
def max_avg_profit_year : ℕ+ := 7

/-- Represents the year of maximum total profit -/
def max_total_profit_year : ℕ+ := 10

/-- Theorem stating the properties of the CNC machine profit -/
theorem cnc_machine_profit :
  (∀ x : ℕ+, profit_function x = -2 * x.val ^ 2 + 40 * x.val - 98) ∧
  (∀ x : ℕ+, x < profit_start → profit_function x ≤ 0) ∧
  (∀ x : ℕ+, x ≥ profit_start → profit_function x > 0) ∧
  (∀ x : ℕ+, x ≠ max_avg_profit_year → 
    (profit_function x : ℚ) / x.val ≤ (profit_function max_avg_profit_year : ℚ) / max_avg_profit_year.val) ∧
  (∀ x : ℕ+, profit_function x ≤ profit_function max_total_profit_year) :=
by sorry

end NUMINAMATH_CALUDE_cnc_machine_profit_l1194_119408


namespace NUMINAMATH_CALUDE_allowance_spent_on_games_l1194_119468

theorem allowance_spent_on_games (total : ℝ) (books_frac snacks_frac music_frac : ℝ) : 
  total = 50 ∧ 
  books_frac = 1/4 ∧ 
  snacks_frac = 1/5 ∧ 
  music_frac = 2/5 → 
  total - (books_frac * total + snacks_frac * total + music_frac * total) = 7.5 := by
sorry

end NUMINAMATH_CALUDE_allowance_spent_on_games_l1194_119468


namespace NUMINAMATH_CALUDE_find_divisor_l1194_119403

theorem find_divisor (dividend quotient remainder divisor : ℕ) : 
  dividend = 15968 →
  quotient = 89 →
  remainder = 37 →
  dividend = divisor * quotient + remainder →
  divisor = 179 := by
sorry

end NUMINAMATH_CALUDE_find_divisor_l1194_119403


namespace NUMINAMATH_CALUDE_isosceles_top_angle_l1194_119447

-- Define an isosceles triangle
def IsIsosceles (a b c : ℝ) : Prop :=
  a = b ∨ b = c ∨ a = c

-- Define the sum of angles in a triangle
axiom angle_sum : ∀ (x y z : ℝ), x + y + z = 180

-- Theorem statement
theorem isosceles_top_angle (a b c : ℝ) 
  (h1 : IsIsosceles a b c) (h2 : a = 40 ∨ b = 40 ∨ c = 40) : 
  a = 40 ∨ b = 40 ∨ c = 40 ∨ a = 100 ∨ b = 100 ∨ c = 100 := by
  sorry


end NUMINAMATH_CALUDE_isosceles_top_angle_l1194_119447


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_and_cube_root_l1194_119472

theorem min_value_sum_reciprocals_and_cube_root (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) 
  (sum_eq_3 : x + y + z = 3) : 
  1/x + 1/y + 1/z + (x*y*z)^(1/3 : ℝ) ≥ 4 ∧ 
  (1/x + 1/y + 1/z + (x*y*z)^(1/3 : ℝ) = 4 ↔ x = 1 ∧ y = 1 ∧ z = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_and_cube_root_l1194_119472


namespace NUMINAMATH_CALUDE_circle_ratio_proof_l1194_119414

theorem circle_ratio_proof (b a c : ℝ) (h1 : b > 0) (h2 : a > 0) (h3 : c > 0)
  (h4 : b^2 - c^2 = 2 * a^2) (h5 : c = 1.5 * a) :
  a / b = 2 / Real.sqrt 17 := by
sorry

end NUMINAMATH_CALUDE_circle_ratio_proof_l1194_119414


namespace NUMINAMATH_CALUDE_triangle_area_l1194_119421

/-- The area of a triangle with side lengths √29, √13, and √34 is 19/2 -/
theorem triangle_area (a b c : ℝ) (ha : a = Real.sqrt 29) (hb : b = Real.sqrt 13) (hc : c = Real.sqrt 34) :
  (1/2) * b * c * Real.sqrt (1 - ((a^2 + b^2 - c^2) / (2*b*c))^2) = 19/2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1194_119421


namespace NUMINAMATH_CALUDE_bowling_team_size_l1194_119411

theorem bowling_team_size (original_avg : ℝ) (new_player1_weight : ℝ) (new_player2_weight : ℝ) (new_avg : ℝ) :
  original_avg = 121 →
  new_player1_weight = 110 →
  new_player2_weight = 60 →
  new_avg = 113 →
  ∃ n : ℕ, n > 0 ∧ 
    (n * original_avg + new_player1_weight + new_player2_weight) / (n + 2) = new_avg ∧
    n = 7 :=
by sorry

end NUMINAMATH_CALUDE_bowling_team_size_l1194_119411


namespace NUMINAMATH_CALUDE_power_division_equality_l1194_119406

theorem power_division_equality : (3^2)^4 / 3^2 = 729 := by
  sorry

end NUMINAMATH_CALUDE_power_division_equality_l1194_119406


namespace NUMINAMATH_CALUDE_retirement_total_is_70_l1194_119419

/-- Represents the retirement eligibility rule for a company -/
structure RetirementRule :=
  (hire_year : ℕ)
  (hire_age : ℕ)
  (retirement_year : ℕ)

/-- Calculates the required total of age and years of employment for retirement -/
def retirement_total (rule : RetirementRule) : ℕ :=
  (rule.retirement_year - rule.hire_year) + rule.hire_age + (rule.retirement_year - rule.hire_year)

/-- Theorem stating the required total for retirement is 70 -/
theorem retirement_total_is_70 (rule : RetirementRule) 
  (h1 : rule.hire_year = 1986)
  (h2 : rule.hire_age = 30)
  (h3 : rule.retirement_year = 2006) :
  retirement_total rule = 70 := by
  sorry

end NUMINAMATH_CALUDE_retirement_total_is_70_l1194_119419


namespace NUMINAMATH_CALUDE_base_seven_43210_equals_10738_l1194_119444

def base_seven_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

theorem base_seven_43210_equals_10738 :
  base_seven_to_decimal [0, 1, 2, 3, 4] = 10738 := by
  sorry

end NUMINAMATH_CALUDE_base_seven_43210_equals_10738_l1194_119444


namespace NUMINAMATH_CALUDE_candy_count_correct_l1194_119469

/-- Represents the number of pieces in each box of chocolates -/
def chocolate_boxes : List Nat := [500, 350, 700, 400, 450, 600]

/-- Represents the number of pieces in each box of lollipops -/
def lollipop_boxes : List Nat := [200, 300, 250, 350]

/-- Represents the number of pieces in each box of gummy bears -/
def gummy_bear_boxes : List Nat := [500, 550, 400, 600, 450]

/-- The total number of candy pieces in all boxes -/
def total_candies : Nat :=
  chocolate_boxes.sum + lollipop_boxes.sum + gummy_bear_boxes.sum

theorem candy_count_correct : total_candies = 6600 := by
  sorry

end NUMINAMATH_CALUDE_candy_count_correct_l1194_119469


namespace NUMINAMATH_CALUDE_composite_function_properties_l1194_119486

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
variable (h_even : ∀ x, f x = f (-x))
variable (h_domain : ∀ x, x ∈ Set.univ)
variable (h_increasing : ∀ x y, 2 < x ∧ x < y ∧ y < 6 → f x < f y)

-- Define the composite function g
def g (x : ℝ) := f (2 - x)

-- Theorem statement
theorem composite_function_properties :
  (∀ x y, 4 < x ∧ x < y ∧ y < 8 → g x < g y) ∧
  (∀ x, g x = g (4 - x)) :=
sorry

end NUMINAMATH_CALUDE_composite_function_properties_l1194_119486


namespace NUMINAMATH_CALUDE_daily_production_is_2170_l1194_119424

/-- The number of toys produced per week -/
def weekly_production : ℕ := 4340

/-- The number of working days per week -/
def working_days : ℕ := 2

/-- The number of toys produced each day -/
def daily_production : ℕ := weekly_production / working_days

/-- Theorem stating that the daily production is 2170 toys -/
theorem daily_production_is_2170 : daily_production = 2170 := by
  sorry

end NUMINAMATH_CALUDE_daily_production_is_2170_l1194_119424


namespace NUMINAMATH_CALUDE_item_cost_calculation_l1194_119434

theorem item_cost_calculation (total_items : ℕ) (total_cost : ℕ) : 
  total_items = 15 → total_cost = 30 → (total_cost / total_items : ℚ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_item_cost_calculation_l1194_119434


namespace NUMINAMATH_CALUDE_complex_magnitude_l1194_119487

theorem complex_magnitude (z : ℂ) (h : Complex.I * z = 1) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l1194_119487


namespace NUMINAMATH_CALUDE_inverse_function_root_uniqueness_l1194_119440

theorem inverse_function_root_uniqueness 
  (f : ℝ → ℝ) (h_inv : Function.Injective f) :
  ∀ m : ℝ, ∃! x : ℝ, f x = m ∨ ∀ y : ℝ, f y ≠ m :=
by sorry

end NUMINAMATH_CALUDE_inverse_function_root_uniqueness_l1194_119440


namespace NUMINAMATH_CALUDE_cat_dog_ratio_l1194_119422

def kennel (num_dogs : ℕ) (num_cats : ℕ) : Prop :=
  num_cats = num_dogs - 6 ∧ num_dogs = 18

theorem cat_dog_ratio (num_dogs num_cats : ℕ) :
  kennel num_dogs num_cats →
  (num_cats : ℚ) / (num_dogs : ℚ) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cat_dog_ratio_l1194_119422


namespace NUMINAMATH_CALUDE_five_digit_divisible_count_l1194_119493

theorem five_digit_divisible_count : 
  let lcm := Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 9))
  let lower_bound := ((10000 + lcm - 1) / lcm) * lcm
  let upper_bound := (99999 / lcm) * lcm
  (upper_bound - lower_bound) / lcm + 1 = 179 := by
  sorry

end NUMINAMATH_CALUDE_five_digit_divisible_count_l1194_119493


namespace NUMINAMATH_CALUDE_days_without_visits_l1194_119473

def days_in_year : ℕ := 366

def visit_period_1 : ℕ := 6
def visit_period_2 : ℕ := 8
def visit_period_3 : ℕ := 10

def days_with_visits (period : ℕ) : ℕ := days_in_year / period

def lcm_two (a b : ℕ) : ℕ := Nat.lcm a b
def lcm_three (a b c : ℕ) : ℕ := Nat.lcm a (Nat.lcm b c)

def days_with_two_visits (period1 period2 : ℕ) : ℕ := days_in_year / (lcm_two period1 period2)

def days_with_three_visits (period1 period2 period3 : ℕ) : ℕ := days_in_year / (lcm_three period1 period2 period3)

theorem days_without_visits :
  days_in_year - 
  ((days_with_visits visit_period_1 + days_with_visits visit_period_2 + days_with_visits visit_period_3) -
   (days_with_two_visits visit_period_1 visit_period_2 + 
    days_with_two_visits visit_period_1 visit_period_3 + 
    days_with_two_visits visit_period_2 visit_period_3) +
   days_with_three_visits visit_period_1 visit_period_2 visit_period_3) = 257 :=
by sorry

end NUMINAMATH_CALUDE_days_without_visits_l1194_119473


namespace NUMINAMATH_CALUDE_sixth_term_is_36_l1194_119478

/-- The sequence of squares of natural numbers from 1 to 7 -/
def square_sequence : Fin 7 → ℕ := fun n => (n + 1)^2

/-- The 6th term of the square sequence is 36 -/
theorem sixth_term_is_36 : square_sequence 5 = 36 := by
  sorry

end NUMINAMATH_CALUDE_sixth_term_is_36_l1194_119478


namespace NUMINAMATH_CALUDE_line_vector_to_slope_intercept_l1194_119443

/-- Given a line in vector form, prove its slope-intercept form and find (m, b) -/
theorem line_vector_to_slope_intercept :
  ∀ (x y : ℝ), 
  (2 : ℝ) * (x - 3) + (-1 : ℝ) * (y + 4) = 0 →
  y = 2 * x - 10 ∧ (2, -10) = (2, -10) := by
  sorry

end NUMINAMATH_CALUDE_line_vector_to_slope_intercept_l1194_119443


namespace NUMINAMATH_CALUDE_factorization_proofs_l1194_119455

theorem factorization_proofs (x y : ℝ) :
  (x^2*y - 2*x*y + x*y^2 = x*y*(x - 2 + y)) ∧
  (x^2 - 3*x + 2 = (x - 1)*(x - 2)) ∧
  (4*x^4 - 64 = 4*(x^2 + 4)*(x + 2)*(x - 2)) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proofs_l1194_119455


namespace NUMINAMATH_CALUDE_range_of_m_l1194_119460

theorem range_of_m (x y m : ℝ) : 
  (2 * x - y = 5 * m) →
  (3 * x + 4 * y = 2 * m) →
  (x + y ≤ 5) →
  (2 * x + 7 * y < 18) →
  (-6 < m ∧ m ≤ 5) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l1194_119460


namespace NUMINAMATH_CALUDE_cos_x_plus_2y_equals_one_l1194_119426

theorem cos_x_plus_2y_equals_one 
  (x y : ℝ) 
  (h_x : x ∈ Set.Icc (-π/4) (π/4))
  (h_y : y ∈ Set.Icc (-π/4) (π/4))
  (h_eq1 : ∃ a : ℝ, x^3 + Real.sin x - 2*a = 0)
  (h_eq2 : ∃ a : ℝ, 4*y^3 + (1/2) * Real.sin (2*y) + a = 0) :
  Real.cos (x + 2*y) = 1 :=
by sorry

end NUMINAMATH_CALUDE_cos_x_plus_2y_equals_one_l1194_119426


namespace NUMINAMATH_CALUDE_total_cost_is_43_l1194_119448

-- Define the prices
def sandwich_price : ℚ := 4
def soda_price : ℚ := 3

-- Define the discount threshold and rate
def discount_threshold : ℚ := 50
def discount_rate : ℚ := 0.1

-- Define the quantities
def num_sandwiches : ℕ := 7
def num_sodas : ℕ := 5

-- Calculate the total cost before discount
def total_cost : ℚ := sandwich_price * num_sandwiches + soda_price * num_sodas

-- Function to apply discount if applicable
def apply_discount (cost : ℚ) : ℚ :=
  if cost > discount_threshold then cost * (1 - discount_rate) else cost

-- Theorem to prove
theorem total_cost_is_43 : apply_discount total_cost = 43 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_43_l1194_119448


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1194_119418

/-- 
Given an arithmetic sequence with n + 2 terms, where the first term is x and the last term is y,
prove that the common difference is (y - x) / (n + 1).
-/
theorem arithmetic_sequence_common_difference 
  (n : ℕ) (x y : ℝ) : 
  let d := (y - x) / (n + 1)
  ∀ (a : Fin (n + 2) → ℝ), 
    (a 0 = x) → 
    (a (Fin.last (n + 1)) = y) → 
    (∀ i : Fin (n + 1), a i.succ - a i = d) → 
    d = (y - x) / (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1194_119418


namespace NUMINAMATH_CALUDE_number_problem_l1194_119492

theorem number_problem (N : ℝ) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N = 20 → (40/100 : ℝ) * N = 240 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1194_119492


namespace NUMINAMATH_CALUDE_brian_tape_problem_l1194_119430

/-- The amount of tape needed for a rectangular box -/
def tape_needed (length width : ℕ) : ℕ := length + 2 * width

/-- The total amount of tape needed for multiple boxes of the same size -/
def total_tape_for_boxes (length width count : ℕ) : ℕ :=
  count * tape_needed length width

/-- The problem statement -/
theorem brian_tape_problem :
  let tape_for_small_boxes := total_tape_for_boxes 30 15 5
  let tape_for_large_boxes := total_tape_for_boxes 40 40 2
  tape_for_small_boxes + tape_for_large_boxes = 540 := by
sorry


end NUMINAMATH_CALUDE_brian_tape_problem_l1194_119430


namespace NUMINAMATH_CALUDE_book_sale_problem_l1194_119488

/-- Proves that the total cost of two books is 600, given the specified conditions --/
theorem book_sale_problem (cost_loss : ℝ) (selling_price : ℝ) :
  cost_loss = 350 →
  selling_price = cost_loss * (1 - 0.15) →
  ∃ (cost_gain : ℝ), 
    selling_price = cost_gain * (1 + 0.19) ∧
    cost_loss + cost_gain = 600 := by
  sorry

end NUMINAMATH_CALUDE_book_sale_problem_l1194_119488


namespace NUMINAMATH_CALUDE_problem_statement_l1194_119497

theorem problem_statement (a b : ℝ) : 
  (a^2 + 4*a + 6) * (2*b^2 - 4*b + 7) ≤ 10 → a + 2*b = 0 :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l1194_119497


namespace NUMINAMATH_CALUDE_total_adding_schemes_l1194_119436

/-- Represents the number of available raw materials -/
def total_materials : ℕ := 5

/-- Represents the number of materials to be added sequentially -/
def materials_to_add : ℕ := 2

/-- Represents the number of ways to add material A first -/
def ways_with_A_first : ℕ := 3

/-- Represents the number of ways to add material B first -/
def ways_with_B_first : ℕ := 6

/-- Represents the number of ways to add materials without A or B -/
def ways_without_A_or_B : ℕ := 6

/-- Theorem stating the total number of different adding schemes -/
theorem total_adding_schemes :
  ways_with_A_first + ways_with_B_first + ways_without_A_or_B = 15 :=
by sorry

end NUMINAMATH_CALUDE_total_adding_schemes_l1194_119436


namespace NUMINAMATH_CALUDE_jellybean_probability_l1194_119425

theorem jellybean_probability : 
  let total_jellybeans : ℕ := 12
  let red_jellybeans : ℕ := 5
  let blue_jellybeans : ℕ := 3
  let white_jellybeans : ℕ := 4
  let picked_jellybeans : ℕ := 3
  
  total_jellybeans = red_jellybeans + blue_jellybeans + white_jellybeans →
  
  (Nat.choose blue_jellybeans 2 * Nat.choose white_jellybeans 1) / 
  Nat.choose total_jellybeans picked_jellybeans = 3 / 55 := by
  sorry

end NUMINAMATH_CALUDE_jellybean_probability_l1194_119425


namespace NUMINAMATH_CALUDE_min_value_expression_l1194_119457

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_constraint : a + b + c = 13) : 
  (a^2 + b^3 + c^4 + 2019) / (10*b + 123*c + 26) ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1194_119457


namespace NUMINAMATH_CALUDE_sqrt_27_simplification_l1194_119439

theorem sqrt_27_simplification : Real.sqrt 27 = 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_27_simplification_l1194_119439


namespace NUMINAMATH_CALUDE_negation_of_implication_l1194_119420

theorem negation_of_implication (a b c : ℝ) :
  ¬(a + b + c = 1 → a^2 + b^2 + c^2 ≤ 1/9) ↔ (a + b + c ≠ 1 → a^2 + b^2 + c^2 > 1/9) := by
sorry

end NUMINAMATH_CALUDE_negation_of_implication_l1194_119420


namespace NUMINAMATH_CALUDE_min_squared_distance_to_origin_l1194_119410

theorem min_squared_distance_to_origin (x y : ℝ) : 
  (x + 5)^2 + (y - 12)^2 = 14^2 → 
  ∃ (min : ℝ), (∀ (a b : ℝ), (a + 5)^2 + (b - 12)^2 = 14^2 → x^2 + y^2 ≤ a^2 + b^2) ∧ min = 1 :=
sorry

end NUMINAMATH_CALUDE_min_squared_distance_to_origin_l1194_119410


namespace NUMINAMATH_CALUDE_a_is_zero_l1194_119412

/-- If a and b are natural numbers such that for every natural number n, 
    2^n * a + b is a perfect square, then a = 0. -/
theorem a_is_zero (a b : ℕ) 
    (h : ∀ n : ℕ, ∃ k : ℕ, 2^n * a + b = k^2) : 
  a = 0 := by
  sorry

end NUMINAMATH_CALUDE_a_is_zero_l1194_119412


namespace NUMINAMATH_CALUDE_candy_cost_l1194_119485

theorem candy_cost (total_cents : ℕ) (num_gumdrops : ℕ) (h1 : total_cents = 224) (h2 : num_gumdrops = 28) :
  total_cents / num_gumdrops = 8 := by
  sorry

end NUMINAMATH_CALUDE_candy_cost_l1194_119485


namespace NUMINAMATH_CALUDE_total_production_theorem_l1194_119451

def week1_production : ℕ := 320
def week2_production : ℕ := 400
def week3_production : ℕ := 300

def total_3_weeks : ℕ := week1_production + week2_production + week3_production
def average_3_weeks : ℕ := total_3_weeks / 3
def total_4_weeks : ℕ := total_3_weeks + average_3_weeks

theorem total_production_theorem : total_4_weeks = 1360 := by
  sorry

end NUMINAMATH_CALUDE_total_production_theorem_l1194_119451


namespace NUMINAMATH_CALUDE_hormone_related_phenomena_l1194_119496

-- Define the set of all phenomena
def Phenomena : Set String :=
  {"Fruit ripening", "Leaves turning yellow", "Fruit shedding", "CO2 fixation",
   "Topping cotton plants", "Absorption of mineral elements"}

-- Define the set of phenomena related to plant hormones
def HormoneRelatedPhenomena : Set String :=
  {"Fruit ripening", "Fruit shedding", "Topping cotton plants"}

-- Define a predicate for phenomena related to plant hormones
def isHormoneRelated (p : String) : Prop :=
  p ∈ HormoneRelatedPhenomena

-- Theorem statement
theorem hormone_related_phenomena :
  ∀ p ∈ Phenomena, isHormoneRelated p ↔
    (p = "Fruit ripening" ∨ p = "Fruit shedding" ∨ p = "Topping cotton plants") :=
by sorry

end NUMINAMATH_CALUDE_hormone_related_phenomena_l1194_119496


namespace NUMINAMATH_CALUDE_jeff_scores_mean_l1194_119498

theorem jeff_scores_mean : 
  let scores : List ℕ := [89, 92, 88, 95, 91, 93]
  (scores.sum : ℚ) / scores.length = 548 / 6 := by sorry

end NUMINAMATH_CALUDE_jeff_scores_mean_l1194_119498


namespace NUMINAMATH_CALUDE_sqrt_equality_implies_one_and_six_l1194_119462

theorem sqrt_equality_implies_one_and_six (a b : ℕ) (ha : a > 0) (hb : b > 0) (hlt : a < b) :
  (Real.sqrt (1 + Real.sqrt (40 + 24 * Real.sqrt 5)) = Real.sqrt a + Real.sqrt b) →
  a = 1 ∧ b = 6 := by sorry

end NUMINAMATH_CALUDE_sqrt_equality_implies_one_and_six_l1194_119462


namespace NUMINAMATH_CALUDE_johnny_earnings_l1194_119423

/-- Represents Johnny's daily work schedule and earnings --/
structure DailyWork where
  hours1 : ℕ
  rate1 : ℕ
  hours2 : ℕ
  rate2 : ℕ
  hours3 : ℕ
  rate3 : ℕ

/-- Calculates the total earnings for a given number of days --/
def totalEarnings (work : DailyWork) (days : ℕ) : ℕ :=
  days * (work.hours1 * work.rate1 + work.hours2 * work.rate2 + work.hours3 * work.rate3)

/-- Johnny's work schedule --/
def johnnysWork : DailyWork :=
  { hours1 := 3
  , rate1 := 7
  , hours2 := 2
  , rate2 := 10
  , hours3 := 4
  , rate3 := 12 }

theorem johnny_earnings :
  totalEarnings johnnysWork 5 = 445 := by
  sorry

end NUMINAMATH_CALUDE_johnny_earnings_l1194_119423


namespace NUMINAMATH_CALUDE_work_rate_proof_l1194_119405

/-- Given work rates for individuals and pairs, prove the work rate for a specific pair -/
theorem work_rate_proof 
  (c_rate : ℚ)
  (bc_rate : ℚ)
  (ca_rate : ℚ)
  (h1 : c_rate = 1 / 24)
  (h2 : bc_rate = 1 / 3)
  (h3 : ca_rate = 1 / 4) :
  ∃ (ab_rate : ℚ), ab_rate = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_work_rate_proof_l1194_119405


namespace NUMINAMATH_CALUDE_num_buses_is_ten_l1194_119449

-- Define the given conditions
def total_people : ℕ := 342
def num_vans : ℕ := 9
def people_per_van : ℕ := 8
def people_per_bus : ℕ := 27

-- Define the function to calculate the number of buses
def calculate_buses : ℕ :=
  (total_people - num_vans * people_per_van) / people_per_bus

-- Theorem statement
theorem num_buses_is_ten : calculate_buses = 10 := by
  sorry

end NUMINAMATH_CALUDE_num_buses_is_ten_l1194_119449


namespace NUMINAMATH_CALUDE_fifth_term_of_sequence_l1194_119450

def geometric_sequence (a : ℕ → ℝ) (x : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * (3 * x)

theorem fifth_term_of_sequence (a : ℕ → ℝ) (x : ℝ) :
  geometric_sequence a x →
  a 0 = 3 →
  a 1 = 9 * x →
  a 2 = 27 * x^2 →
  a 3 = 81 * x^3 →
  a 4 = 243 * x^4 := by
sorry

end NUMINAMATH_CALUDE_fifth_term_of_sequence_l1194_119450


namespace NUMINAMATH_CALUDE_correct_factorization_l1194_119433

theorem correct_factorization (x : ℝ) : x^2 - 0.01 = (x + 0.1) * (x - 0.1) := by
  sorry

end NUMINAMATH_CALUDE_correct_factorization_l1194_119433


namespace NUMINAMATH_CALUDE_jane_stopped_babysitting_16_years_ago_l1194_119491

/-- Represents a person with their current age and the age they started babysitting -/
structure Babysitter where
  current_age : ℕ
  start_age : ℕ

/-- Represents a person who was babysat -/
structure BabysatPerson where
  current_age : ℕ

def Babysitter.max_babysat_age (b : Babysitter) : ℕ := b.current_age / 2

def years_since_stopped_babysitting (b : Babysitter) (p : BabysatPerson) : ℕ :=
  b.current_age - p.current_age

theorem jane_stopped_babysitting_16_years_ago
  (jane : Babysitter)
  (oldest_babysat : BabysatPerson)
  (h1 : jane.current_age = 32)
  (h2 : jane.start_age = 16)
  (h3 : oldest_babysat.current_age = 24)
  (h4 : oldest_babysat.current_age ≤ jane.max_babysat_age) :
  years_since_stopped_babysitting jane oldest_babysat = 16 := by
  sorry

#check jane_stopped_babysitting_16_years_ago

end NUMINAMATH_CALUDE_jane_stopped_babysitting_16_years_ago_l1194_119491


namespace NUMINAMATH_CALUDE_largest_integer_less_than_100_remainder_5_mod_8_l1194_119471

theorem largest_integer_less_than_100_remainder_5_mod_8 :
  ∃ n : ℕ, n < 100 ∧ n % 8 = 5 ∧ ∀ m : ℕ, m < 100 ∧ m % 8 = 5 → m ≤ n :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_largest_integer_less_than_100_remainder_5_mod_8_l1194_119471


namespace NUMINAMATH_CALUDE_animal_feet_count_animal_feet_theorem_l1194_119432

theorem animal_feet_count (total_heads : Nat) (hen_count : Nat) : Nat :=
  let cow_count := total_heads - hen_count
  let hen_feet := hen_count * 2
  let cow_feet := cow_count * 4
  hen_feet + cow_feet

theorem animal_feet_theorem (total_heads : Nat) (hen_count : Nat) 
  (h1 : total_heads = 44) (h2 : hen_count = 18) : 
  animal_feet_count total_heads hen_count = 140 := by
  sorry

end NUMINAMATH_CALUDE_animal_feet_count_animal_feet_theorem_l1194_119432


namespace NUMINAMATH_CALUDE_equilateral_triangle_on_parabola_l1194_119427

/-- Given two points A(1, 0) and B(b, 0), if there exists a point C on the parabola y^2 = 4x
    such that triangle ABC is equilateral, then b = 5 or b = -1/3 -/
theorem equilateral_triangle_on_parabola (b : ℝ) :
  (∃ (x y : ℝ), y^2 = 4*x ∧ 
    ((x - 1)^2 + y^2 = (x - b)^2 + y^2) ∧
    ((x - 1)^2 + y^2 = (b - 1)^2) ∧
    ((x - b)^2 + y^2 = (b - 1)^2)) →
  b = 5 ∨ b = -1/3 :=
by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_on_parabola_l1194_119427


namespace NUMINAMATH_CALUDE_max_value_constrained_l1194_119481

theorem max_value_constrained (x y : ℝ) (h : x^2 + y^2 = 4) :
  ∃ (max : ℝ), max = 14 ∧ ∀ (x' y' : ℝ), x'^2 + y'^2 = 4 → x'^2 + 6*y' + 2 ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_constrained_l1194_119481


namespace NUMINAMATH_CALUDE_sally_peach_cost_l1194_119499

-- Define the given amounts
def total_spent : ℚ := 23.86
def cherry_cost : ℚ := 11.54

-- Define the amount spent on peaches after coupon
def peach_cost : ℚ := total_spent - cherry_cost

-- Theorem to prove
theorem sally_peach_cost : peach_cost = 12.32 := by
  sorry

end NUMINAMATH_CALUDE_sally_peach_cost_l1194_119499


namespace NUMINAMATH_CALUDE_real_part_of_i_times_one_minus_i_l1194_119453

theorem real_part_of_i_times_one_minus_i (i : ℂ) : 
  i * i = -1 → Complex.re (i * (1 - i)) = 1 := by sorry

end NUMINAMATH_CALUDE_real_part_of_i_times_one_minus_i_l1194_119453


namespace NUMINAMATH_CALUDE_mod_congruence_solution_l1194_119442

theorem mod_congruence_solution : ∃! (n : ℤ), 0 ≤ n ∧ n ≤ 10 ∧ n ≡ -2154 [ZMOD 7] ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_mod_congruence_solution_l1194_119442


namespace NUMINAMATH_CALUDE_side_length_relationship_l1194_119482

/-- Side length of an inscribed regular n-gon in a circle with radius r -/
def a (n : ℕ) (r : ℝ) : ℝ := sorry

/-- Side length of a circumscribed regular n-gon around a circle with radius r -/
def A (n : ℕ) (r : ℝ) : ℝ := sorry

/-- Theorem stating the relationship between side lengths of regular polygons -/
theorem side_length_relationship (n : ℕ) (r : ℝ) (h : 0 < r) :
  1 / A (2 * n) r = 1 / A n r + 1 / a n r := by sorry

end NUMINAMATH_CALUDE_side_length_relationship_l1194_119482


namespace NUMINAMATH_CALUDE_alice_bob_difference_zero_l1194_119416

/-- Represents the vacation expenses problem -/
def vacation_expenses (alice_paid bob_paid charlie_paid : ℝ) (a b : ℝ) : Prop :=
  let total_paid := alice_paid + bob_paid + charlie_paid
  let equal_share := total_paid / 3
  -- Alice's balance after giving 'a' to Charlie
  (alice_paid - a = equal_share) ∧
  -- Bob's balance after giving 'b' to Charlie
  (bob_paid - b = equal_share) ∧
  -- Charlie's balance after receiving 'a' from Alice and 'b' from Bob
  (charlie_paid + a + b = equal_share)

/-- Theorem stating that the difference between what Alice and Bob give to Charlie is zero -/
theorem alice_bob_difference_zero 
  (alice_paid bob_paid charlie_paid : ℝ) 
  (h_alice : alice_paid = 180) 
  (h_bob : bob_paid = 240) 
  (h_charlie : charlie_paid = 120) :
  ∃ a b : ℝ, vacation_expenses alice_paid bob_paid charlie_paid a b ∧ a - b = 0 :=
sorry

end NUMINAMATH_CALUDE_alice_bob_difference_zero_l1194_119416


namespace NUMINAMATH_CALUDE_fourth_roots_of_unity_solution_l1194_119476

theorem fourth_roots_of_unity_solution (a b c d k : ℂ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (eq1 : a * k^3 + b * k^2 + c * k + d = 0)
  (eq2 : b * k^3 + c * k^2 + d * k + a = 0) :
  k = 1 ∨ k = -1 ∨ k = I ∨ k = -I :=
sorry

end NUMINAMATH_CALUDE_fourth_roots_of_unity_solution_l1194_119476


namespace NUMINAMATH_CALUDE_symmetric_sequence_sum_is_n_squared_l1194_119463

/-- The sum of the symmetric sequence 1+2+3+...+(n-1)+n+(n+1)+n+...+3+2+1 -/
def symmetricSequenceSum (n : ℕ) : ℕ :=
  (List.range n).sum + n + (n + 1) + (List.range n).sum

/-- Theorem: The sum of the symmetric sequence is equal to n^2 -/
theorem symmetric_sequence_sum_is_n_squared (n : ℕ) :
  symmetricSequenceSum n = n^2 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_sequence_sum_is_n_squared_l1194_119463


namespace NUMINAMATH_CALUDE_sin_negative_seventeen_pi_thirds_l1194_119464

theorem sin_negative_seventeen_pi_thirds :
  Real.sin (-17 * π / 3) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_seventeen_pi_thirds_l1194_119464


namespace NUMINAMATH_CALUDE_function_properties_l1194_119494

noncomputable def f (a b x : ℝ) : ℝ := (a * x) / (Real.exp x + 1) + b * Real.exp (-x)

theorem function_properties (a b k : ℝ) :
  (f a b 0 = 1) →
  (HasDerivAt (f a b) (-1/2) 0) →
  (∀ x ≠ 0, f a b x > x / (Real.exp x - 1) + k * Real.exp (-x)) →
  (a = 1 ∧ b = 1 ∧ k ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l1194_119494


namespace NUMINAMATH_CALUDE_cubic_odd_and_increasing_l1194_119477

-- Define the function f(x) = x³
def f (x : ℝ) : ℝ := x^3

-- State the theorem
theorem cubic_odd_and_increasing :
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x y, 0 < x → x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_cubic_odd_and_increasing_l1194_119477


namespace NUMINAMATH_CALUDE_incoming_students_l1194_119489

theorem incoming_students (n : ℕ) : n < 600 ∧ n % 26 = 25 ∧ n % 24 = 15 → n = 519 :=
by sorry

end NUMINAMATH_CALUDE_incoming_students_l1194_119489


namespace NUMINAMATH_CALUDE_remainder_of_3_pow_20_mod_5_l1194_119459

theorem remainder_of_3_pow_20_mod_5 : 3^20 % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_3_pow_20_mod_5_l1194_119459


namespace NUMINAMATH_CALUDE_correct_arrangement_count_l1194_119409

/-- The number of ways to arrange 3 girls and 5 boys in a row -/
def arrangement_count (n_girls : ℕ) (n_boys : ℕ) : ℕ × ℕ × ℕ :=
  let total := n_girls + n_boys
  let adjacent := (Nat.factorial n_girls) * (Nat.factorial (total - n_girls + 1))
  let not_adjacent := (Nat.factorial n_boys) * (Nat.choose (n_boys + 1) n_girls) * (Nat.factorial n_girls)
  let boys_fixed := Nat.choose total n_girls
  (adjacent, not_adjacent, boys_fixed)

/-- Theorem stating the correct number of arrangements for 3 girls and 5 boys -/
theorem correct_arrangement_count :
  arrangement_count 3 5 = (4320, 14400, 336) := by
  sorry

end NUMINAMATH_CALUDE_correct_arrangement_count_l1194_119409


namespace NUMINAMATH_CALUDE_sum_of_digits_power_of_two_l1194_119475

-- Define s(n) as the sum of digits of n
def sum_of_digits (n : ℕ) : ℕ := sorry

-- Define the property that n - s(n) is divisible by 9
def divisible_by_nine (n : ℕ) : Prop :=
  ∃ k : ℕ, n - sum_of_digits n = 9 * k

-- State the theorem
theorem sum_of_digits_power_of_two :
  (∀ n : ℕ, divisible_by_nine n) →
  2^2009 % 9 = 5 →
  sum_of_digits (sum_of_digits (sum_of_digits (2^2009))) < 9 →
  sum_of_digits (sum_of_digits (sum_of_digits (2^2009))) = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_power_of_two_l1194_119475


namespace NUMINAMATH_CALUDE_sara_initial_peaches_l1194_119484

/-- The number of peaches Sara picked at the orchard -/
def peaches_picked : ℕ := 37

/-- The total number of peaches Sara has now -/
def total_peaches_now : ℕ := 61

/-- The initial number of peaches Sara had -/
def initial_peaches : ℕ := total_peaches_now - peaches_picked

theorem sara_initial_peaches : initial_peaches = 24 := by
  sorry

end NUMINAMATH_CALUDE_sara_initial_peaches_l1194_119484


namespace NUMINAMATH_CALUDE_cubic_equation_integer_solutions_l1194_119480

theorem cubic_equation_integer_solutions :
  (∀ k : ℤ, ∃! n : ℕ, ∀ x : ℤ, x^3 - 24*x + k = 0 → (x.natAbs ≤ n)) ∧
  (∃! x : ℤ, x^3 + 24*x - 2016 = 0) ∧
  (12^3 + 24*12 - 2016 = 0) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_integer_solutions_l1194_119480


namespace NUMINAMATH_CALUDE_train_speed_calculation_l1194_119428

theorem train_speed_calculation (train_length : ℝ) (crossing_time : ℝ) : 
  train_length = 120 ∧ crossing_time = 8 → 
  ∃ (speed : ℝ), speed = 54 ∧ 
  (2 * train_length) / crossing_time * 3.6 = 2 * speed := by
sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l1194_119428
