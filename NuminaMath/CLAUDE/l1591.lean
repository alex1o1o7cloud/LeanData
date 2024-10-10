import Mathlib

namespace sin_50_plus_sqrt3_tan_10_equals_1_l1591_159147

theorem sin_50_plus_sqrt3_tan_10_equals_1 :
  Real.sin (50 * π / 180) * (1 + Real.sqrt 3 * Real.tan (10 * π / 180)) = 1 := by
  sorry

end sin_50_plus_sqrt3_tan_10_equals_1_l1591_159147


namespace sum_of_fractions_l1591_159199

/-- The sum of specific fractions is equal to -2/15 -/
theorem sum_of_fractions :
  (1 : ℚ) / 3 + 1 / 2 + (-5) / 6 + 1 / 5 + 1 / 4 + (-9) / 20 + (-2) / 15 = -2 / 15 := by
  sorry

end sum_of_fractions_l1591_159199


namespace complex_modulus_problem_l1591_159143

theorem complex_modulus_problem (z : ℂ) : z * (1 + Complex.I) = 1 - Complex.I → Complex.abs z = 1 := by
  sorry

end complex_modulus_problem_l1591_159143


namespace arcsin_sqrt3_div2_l1591_159153

theorem arcsin_sqrt3_div2 : Real.arcsin (Real.sqrt 3 / 2) = π / 3 := by
  sorry

end arcsin_sqrt3_div2_l1591_159153


namespace probability_blue_red_white_l1591_159170

/-- Represents the number of marbles of each color in the bag -/
structure MarbleCounts where
  blue : ℕ
  red : ℕ
  white : ℕ

/-- Calculates the probability of drawing a specific sequence of marbles -/
def probability_of_sequence (counts : MarbleCounts) : ℚ :=
  let total := counts.blue + counts.red + counts.white
  (counts.blue : ℚ) / total *
  (counts.red : ℚ) / (total - 1) *
  (counts.white : ℚ) / (total - 2)

/-- The main theorem stating the probability of drawing blue, red, then white -/
theorem probability_blue_red_white :
  probability_of_sequence ⟨4, 3, 6⟩ = 6 / 143 := by
  sorry

end probability_blue_red_white_l1591_159170


namespace common_chord_length_is_10_l1591_159180

/-- The equation of the first circle -/
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 10*x - 10*y = 0

/-- The equation of the second circle -/
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 6*x + 2*y - 40 = 0

/-- The length of the common chord of two intersecting circles -/
def common_chord_length : ℝ := 10

/-- Theorem: The length of the common chord of the given intersecting circles is 10 -/
theorem common_chord_length_is_10 :
  ∃ (A B : ℝ × ℝ),
    circle1 A.1 A.2 ∧ circle1 B.1 B.2 ∧
    circle2 A.1 A.2 ∧ circle2 B.1 B.2 ∧
    A ≠ B ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = common_chord_length :=
by sorry

end common_chord_length_is_10_l1591_159180


namespace h_zero_iff_b_eq_neg_seven_fifths_l1591_159139

def h (x : ℝ) := 5*x + 7

theorem h_zero_iff_b_eq_neg_seven_fifths :
  ∀ b : ℝ, h b = 0 ↔ b = -7/5 := by sorry

end h_zero_iff_b_eq_neg_seven_fifths_l1591_159139


namespace ellipse_and_point_theorem_l1591_159190

-- Define the ellipse M
structure Ellipse :=
  (a b : ℝ)
  (center_x center_y : ℝ)
  (eccentricity : ℝ)

-- Define the line l
structure Line :=
  (m : ℝ)
  (b : ℝ)

-- Define a point
structure Point :=
  (x y : ℝ)

-- Define the problem
theorem ellipse_and_point_theorem 
  (M : Ellipse)
  (l : Line)
  (N : ℝ → Point) :
  M.center_x = 0 ∧ 
  M.center_y = 0 ∧ 
  M.a = 2 ∧ 
  M.eccentricity = 1/2 ∧
  l.m ≠ 0 ∧
  (∀ t, N t = Point.mk t 0) →
  (M.a^2 * M.b^2 = 12 ∧ 
   (∀ t, (0 < t ∧ t < 1/4) ↔ 
     ∃ A B : Point, 
       A.x^2 / 4 + A.y^2 / 3 = 1 ∧
       B.x^2 / 4 + B.y^2 / 3 = 1 ∧
       A.x = l.m * A.y + l.b ∧
       B.x = l.m * B.y + l.b ∧
       ((A.x - t)^2 + A.y^2 = (B.x - t)^2 + B.y^2) ∧
       ((A.x - (N t).x) * (B.y - A.y) = (A.y - (N t).y) * (B.x - A.x)))) :=
by sorry

end ellipse_and_point_theorem_l1591_159190


namespace angle_bisector_construction_with_two_sided_ruler_l1591_159186

/-- A point in a plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A line in a plane -/
structure Line :=
  (a : ℝ) (b : ℝ) (c : ℝ)

/-- An angle formed by two lines -/
structure Angle :=
  (vertex : Point)
  (line1 : Line)
  (line2 : Line)

/-- A two-sided ruler -/
structure TwoSidedRuler :=
  (length : ℝ)

/-- Definition of an angle bisector -/
def is_angle_bisector (a : Angle) (l : Line) : Prop :=
  sorry

/-- Definition of an inaccessible point -/
def is_inaccessible (p : Point) : Prop :=
  sorry

/-- Main theorem: It is possible to construct the bisector of an angle with an inaccessible vertex using only a two-sided ruler -/
theorem angle_bisector_construction_with_two_sided_ruler 
  (a : Angle) (r : TwoSidedRuler) (h : is_inaccessible a.vertex) : 
  ∃ (l : Line), is_angle_bisector a l :=
sorry

end angle_bisector_construction_with_two_sided_ruler_l1591_159186


namespace function_characterization_l1591_159160

theorem function_characterization (f : ℕ → ℕ) : 
  (∀ m n : ℕ, 2 * f (m * n) ≥ f (m^2 + n^2) - f m^2 - f n^2 ∧ 
               f (m^2 + n^2) - f m^2 - f n^2 ≥ 2 * f m * f n) → 
  (∀ n : ℕ, f n = n^2) := by
sorry

end function_characterization_l1591_159160


namespace symmetric_points_l1591_159162

/-- Given a line ax + by + c = 0 and two points (x₁, y₁) and (x₂, y₂), 
    this function returns true if the points are symmetric with respect to the line -/
def are_symmetric (a b c : ℝ) (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  -- The midpoint of the two points lies on the line
  a * ((x₁ + x₂) / 2) + b * ((y₁ + y₂) / 2) + c = 0 ∧
  -- The line connecting the two points is perpendicular to the given line
  (y₂ - y₁) * a = -(x₂ - x₁) * b

/-- Theorem stating that (-5, -4) is symmetric to (3, 4) with respect to the line x + y + 1 = 0 -/
theorem symmetric_points : are_symmetric 1 1 1 3 4 (-5) (-4) := by
  sorry

end symmetric_points_l1591_159162


namespace ratio_calculations_l1591_159141

theorem ratio_calculations (A B C : ℚ) 
  (h : A / B = 3 / 2 ∧ B / C = 2 / 6) : 
  (4 * A + 3 * B) / (5 * C - 2 * A) = 5 / 8 ∧ 
  (A + C) / (2 * B + A) = 9 / 5 := by
  sorry

end ratio_calculations_l1591_159141


namespace abc_inequality_l1591_159100

/-- Given a = √2, b = √7 - √3, and c = √6 - √2, prove that a > c > b -/
theorem abc_inequality :
  let a : ℝ := Real.sqrt 2
  let b : ℝ := Real.sqrt 7 - Real.sqrt 3
  let c : ℝ := Real.sqrt 6 - Real.sqrt 2
  a > c ∧ c > b := by sorry

end abc_inequality_l1591_159100


namespace non_juniors_playing_instruments_l1591_159175

theorem non_juniors_playing_instruments (total_students : ℕ) 
  (junior_play_percent : ℚ) (non_junior_not_play_percent : ℚ) 
  (total_not_play_percent : ℚ) :
  total_students = 600 →
  junior_play_percent = 30 / 100 →
  non_junior_not_play_percent = 35 / 100 →
  total_not_play_percent = 40 / 100 →
  ∃ (non_juniors_playing : ℕ), non_juniors_playing = 334 :=
by sorry

end non_juniors_playing_instruments_l1591_159175


namespace solve_equation_l1591_159119

theorem solve_equation : ∃ x : ℚ, 5 * (x - 9) = 6 * (3 - 3 * x) + 6 ∧ x = 3 := by
  sorry

end solve_equation_l1591_159119


namespace mean_median_difference_l1591_159126

/-- Represents the score distribution on a math test -/
structure ScoreDistribution where
  score65 : ℝ
  score75 : ℝ
  score85 : ℝ
  score92 : ℝ
  score98 : ℝ
  sum_to_one : score65 + score75 + score85 + score92 + score98 = 1

/-- Calculates the mean score given a score distribution -/
def mean_score (sd : ScoreDistribution) : ℝ :=
  65 * sd.score65 + 75 * sd.score75 + 85 * sd.score85 + 92 * sd.score92 + 98 * sd.score98

/-- Determines the median score given a score distribution -/
noncomputable def median_score (sd : ScoreDistribution) : ℝ :=
  if sd.score65 + sd.score75 > 0.5 then 75
  else if sd.score65 + sd.score75 + sd.score85 > 0.5 then 85
  else if sd.score65 + sd.score75 + sd.score85 + sd.score92 > 0.5 then 92
  else 98

/-- Theorem stating that the absolute difference between mean and median is 1.05 -/
theorem mean_median_difference (sd : ScoreDistribution) 
  (h1 : sd.score65 = 0.15)
  (h2 : sd.score75 = 0.20)
  (h3 : sd.score85 = 0.30)
  (h4 : sd.score92 = 0.10) :
  |mean_score sd - median_score sd| = 1.05 := by
  sorry


end mean_median_difference_l1591_159126


namespace product_sum_multiple_l1591_159116

theorem product_sum_multiple (a b m : ℤ) : 
  b = 9 → b - a = 5 → a * b = m * (a + b) + 10 → m = 2 := by
  sorry

end product_sum_multiple_l1591_159116


namespace unique_prime_polynomial_l1591_159111

theorem unique_prime_polynomial : ∃! (n : ℕ), n > 0 ∧ Nat.Prime (n^3 - 9*n^2 + 27*n - 28) :=
sorry

end unique_prime_polynomial_l1591_159111


namespace lcm_of_ratio_and_hcf_l1591_159192

theorem lcm_of_ratio_and_hcf (a b : ℕ+) : 
  (a : ℚ) / b = 3 / 4 → 
  Nat.gcd a b = 8 → 
  Nat.lcm a b = 96 := by
sorry

end lcm_of_ratio_and_hcf_l1591_159192


namespace train_passing_jogger_train_passes_jogger_in_36_seconds_l1591_159165

/-- Time for a train to pass a jogger --/
theorem train_passing_jogger (jogger_speed : Real) (train_speed : Real) 
  (train_length : Real) (initial_distance : Real) : Real :=
  let jogger_speed_ms := jogger_speed * 1000 / 3600
  let train_speed_ms := train_speed * 1000 / 3600
  let relative_speed := train_speed_ms - jogger_speed_ms
  let total_distance := initial_distance + train_length
  total_distance / relative_speed

/-- The train passes the jogger in 36 seconds --/
theorem train_passes_jogger_in_36_seconds : 
  train_passing_jogger 9 45 120 240 = 36 := by
  sorry

end train_passing_jogger_train_passes_jogger_in_36_seconds_l1591_159165


namespace parallel_condition_l1591_159156

-- Define the lines l1 and l2
def l1 (a x y : ℝ) : Prop := a * x + y = 1
def l2 (a x y : ℝ) : Prop := 9 * x + a * y = 1

-- Define parallel lines
def parallel (a : ℝ) : Prop := ∀ x y, l1 a x y ↔ l2 a x y

-- Theorem statement
theorem parallel_condition (a : ℝ) :
  (a + 3 = 0 → parallel a) ∧ ¬(parallel a → a + 3 = 0) :=
sorry

end parallel_condition_l1591_159156


namespace sum_of_coefficients_l1591_159125

theorem sum_of_coefficients (x y : ℝ) : (2*x + 3*y)^12 = 244140625 := by sorry

end sum_of_coefficients_l1591_159125


namespace pond_width_l1591_159115

/-- The width of a rectangular pond, given its length, depth, and volume -/
theorem pond_width (length : ℝ) (depth : ℝ) (volume : ℝ) : 
  length = 28 → depth = 5 → volume = 1400 → volume = length * depth * 10 := by
  sorry

end pond_width_l1591_159115


namespace solve_equation_l1591_159195

theorem solve_equation (x : ℝ) : (2 * x + 7) / 7 = 13 → x = 42 := by
  sorry

end solve_equation_l1591_159195


namespace simplify_expression_l1591_159108

theorem simplify_expression (y : ℝ) : (3*y)^3 - (2*y)*(y^2) + y^4 = 25*y^3 + y^4 := by
  sorry

end simplify_expression_l1591_159108


namespace cookfire_logs_after_three_hours_l1591_159107

/-- Calculates the number of logs left in a cookfire after a given number of hours. -/
def logs_left (initial_logs : ℕ) (burn_rate : ℕ) (add_rate : ℕ) (hours : ℕ) : ℕ :=
  initial_logs + add_rate * hours - burn_rate * hours

/-- Theorem: After 3 hours, a cookfire that starts with 6 logs, burns 3 logs per hour, 
    and receives 2 logs at the end of each hour will have 3 logs left. -/
theorem cookfire_logs_after_three_hours :
  logs_left 6 3 2 3 = 3 := by
  sorry

end cookfire_logs_after_three_hours_l1591_159107


namespace function_equation_l1591_159151

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem function_equation (h : ∀ x : ℝ, 2 * f x - f (-x) = 3 * x + 1) : 
  f = fun x ↦ x + 1 := by
sorry

end function_equation_l1591_159151


namespace snail_distance_is_31_l1591_159169

def snail_path : List ℤ := [3, -5, 10, 2]

def distance (a b : ℤ) : ℕ := Int.natAbs (b - a)

def total_distance (path : List ℤ) : ℕ :=
  match path with
  | [] => 0
  | [_] => 0
  | x :: y :: rest => distance x y + total_distance (y :: rest)

theorem snail_distance_is_31 : total_distance snail_path = 31 := by
  sorry

end snail_distance_is_31_l1591_159169


namespace equal_numbers_product_l1591_159164

theorem equal_numbers_product (a b c d : ℝ) : 
  (a + b + c + d) / 4 = 20 →
  a = 12 →
  b = 22 →
  c = d →
  c * d = 529 := by
sorry

end equal_numbers_product_l1591_159164


namespace correct_experimental_procedure_l1591_159171

-- Define the type for experimental procedures
inductive ExperimentalProcedure
| MicroorganismIsolation
| WineFermentation
| CellObservation
| EcoliCounting

-- Define the properties of each procedure
def requiresLight (p : ExperimentalProcedure) : Prop :=
  match p with
  | ExperimentalProcedure.MicroorganismIsolation => false
  | _ => true

def requiresOpenBottle (p : ExperimentalProcedure) : Prop :=
  match p with
  | ExperimentalProcedure.WineFermentation => false
  | _ => true

def adjustAperture (p : ExperimentalProcedure) : Prop :=
  match p with
  | ExperimentalProcedure.CellObservation => false
  | _ => true

def ensureDilution (p : ExperimentalProcedure) : Prop :=
  match p with
  | ExperimentalProcedure.EcoliCounting => true
  | _ => false

-- Theorem stating that E. coli counting is the correct procedure
theorem correct_experimental_procedure :
  ∀ p : ExperimentalProcedure,
    (p = ExperimentalProcedure.EcoliCounting) ↔
    (¬requiresLight p ∧ ¬requiresOpenBottle p ∧ ¬adjustAperture p ∧ ensureDilution p) :=
by sorry

end correct_experimental_procedure_l1591_159171


namespace power_function_inequality_l1591_159163

/-- A power function that passes through the point (2,√2) -/
def f (x : ℝ) : ℝ := x^(1/2)

/-- Theorem stating the inequality for any two points on the graph of f -/
theorem power_function_inequality (x₁ x₂ : ℝ) (h1 : 0 < x₁) (h2 : x₁ < x₂) :
  x₂ * f x₁ > x₁ * f x₂ := by
  sorry

end power_function_inequality_l1591_159163


namespace complex_square_ratio_real_l1591_159131

theorem complex_square_ratio_real (z₁ z₂ : ℂ) (hz₁ : z₁ ≠ 0) (hz₂ : z₂ ≠ 0) 
  (h : Complex.abs (z₁ + z₂) = Complex.abs (z₁ - z₂)) : 
  ∃ (r : ℝ), (z₁ / z₂)^2 = r := by
sorry

end complex_square_ratio_real_l1591_159131


namespace product_sum_relation_l1591_159176

theorem product_sum_relation (a b : ℤ) : 
  (a * b = 2 * (a + b) + 11) → (b = 7) → (b - a = 2) := by
  sorry

end product_sum_relation_l1591_159176


namespace negative_number_identification_l1591_159103

theorem negative_number_identification (a b c d : ℝ) : 
  a = -6 ∧ b = 0 ∧ c = 0.2 ∧ d = 3 →
  (a < 0 ∧ b ≥ 0 ∧ c > 0 ∧ d > 0) := by sorry

end negative_number_identification_l1591_159103


namespace smallest_square_with_five_interior_points_l1591_159157

/-- A lattice point in 2D space -/
def LatticePoint := ℤ × ℤ

/-- The number of interior lattice points in a square with side length s -/
def interiorLatticePoints (s : ℕ) : ℕ := (s - 1) ^ 2

/-- The smallest square side length with exactly 5 interior lattice points -/
def smallestSquareSide : ℕ := 4

theorem smallest_square_with_five_interior_points :
  (∀ n < smallestSquareSide, interiorLatticePoints n ≠ 5) ∧
  interiorLatticePoints smallestSquareSide = 5 := by
  sorry

end smallest_square_with_five_interior_points_l1591_159157


namespace ab_neq_zero_sufficient_not_necessary_for_a_neq_zero_l1591_159101

theorem ab_neq_zero_sufficient_not_necessary_for_a_neq_zero :
  (∀ a b : ℝ, ab ≠ 0 → a ≠ 0) ∧
  (∃ a b : ℝ, a ≠ 0 ∧ ab = 0) :=
by sorry

end ab_neq_zero_sufficient_not_necessary_for_a_neq_zero_l1591_159101


namespace woman_work_days_l1591_159138

-- Define the work rates
def man_rate : ℚ := 1 / 6
def boy_rate : ℚ := 1 / 12
def combined_rate : ℚ := 1 / 3

-- Define the woman's work rate
def woman_rate : ℚ := combined_rate - man_rate - boy_rate

-- Theorem to prove
theorem woman_work_days : (1 : ℚ) / woman_rate = 12 := by
  sorry


end woman_work_days_l1591_159138


namespace range_of_f_l1591_159109

def f (x : ℕ) : ℤ := x^2 - 2*x

def domain : Set ℕ := {0, 1, 2, 3}

theorem range_of_f : 
  {y | ∃ x ∈ domain, f x = y} = {-1, 0, 3} := by sorry

end range_of_f_l1591_159109


namespace fraction_zero_solution_l1591_159168

theorem fraction_zero_solution (x : ℝ) : 
  (x^2 - 4) / (x + 2) = 0 ∧ x + 2 ≠ 0 → x = 2 := by
  sorry

end fraction_zero_solution_l1591_159168


namespace unicorn_flowers_theorem_l1591_159114

/-- The number of flowers that bloom per unicorn step -/
def flowers_per_step (num_unicorns : ℕ) (total_distance : ℕ) (step_length : ℕ) (total_flowers : ℕ) : ℚ :=
  total_flowers / (num_unicorns * (total_distance * 1000 / step_length))

/-- Theorem: Given the conditions, 4 flowers bloom per unicorn step -/
theorem unicorn_flowers_theorem (num_unicorns : ℕ) (total_distance : ℕ) (step_length : ℕ) (total_flowers : ℕ)
  (h1 : num_unicorns = 6)
  (h2 : total_distance = 9)
  (h3 : step_length = 3)
  (h4 : total_flowers = 72000) :
  flowers_per_step num_unicorns total_distance step_length total_flowers = 4 := by
  sorry

end unicorn_flowers_theorem_l1591_159114


namespace two_y_squared_over_x_is_fraction_l1591_159128

/-- A fraction is an expression with a variable in the denominator -/
def is_fraction (numerator denominator : ℚ) : Prop :=
  ∃ (x : ℚ), denominator = x

/-- The expression 2y^2/x is a fraction -/
theorem two_y_squared_over_x_is_fraction (x y : ℚ) :
  is_fraction (2 * y^2) x :=
sorry

end two_y_squared_over_x_is_fraction_l1591_159128


namespace price_reduction_theorem_l1591_159198

/-- Represents the mall's sales and profit model -/
structure MallSales where
  initial_sales : ℕ  -- Initial daily sales
  initial_profit : ℕ  -- Initial profit per item in yuan
  sales_increase_rate : ℕ  -- Additional items sold per yuan of price reduction
  price_reduction : ℕ  -- Price reduction in yuan

/-- Calculates the daily profit given a MallSales structure -/
def daily_profit (m : MallSales) : ℕ :=
  let new_sales := m.initial_sales + m.sales_increase_rate * m.price_reduction
  let new_profit_per_item := m.initial_profit - m.price_reduction
  new_sales * new_profit_per_item

/-- Theorem stating that a price reduction of 20 yuan results in a daily profit of 2100 yuan -/
theorem price_reduction_theorem (m : MallSales) 
  (h1 : m.initial_sales = 30)
  (h2 : m.initial_profit = 50)
  (h3 : m.sales_increase_rate = 2)
  (h4 : m.price_reduction = 20) :
  daily_profit m = 2100 := by
  sorry

#eval daily_profit { initial_sales := 30, initial_profit := 50, sales_increase_rate := 2, price_reduction := 20 }

end price_reduction_theorem_l1591_159198


namespace expression_simplification_l1591_159137

theorem expression_simplification :
  (∀ a : ℝ, 2 * (a - 1) - (2 * a - 3) + 3 = 4) ∧
  (∀ x : ℝ, 3 * x^2 - (7 * x - (4 * x - 3) - 2 * x^2) = x^2 - 3 * x + 3) := by
  sorry

end expression_simplification_l1591_159137


namespace line_segment_endpoint_l1591_159193

/-- Given a line segment from (4, 3) to (x, 9) with length 15 and x < 0, prove x = 4 - √189 -/
theorem line_segment_endpoint (x : ℝ) : 
  x < 0 → 
  ((x - 4)^2 + (9 - 3)^2 : ℝ) = 15^2 → 
  x = 4 - Real.sqrt 189 := by
sorry

end line_segment_endpoint_l1591_159193


namespace inequality_range_l1591_159173

theorem inequality_range :
  {a : ℝ | ∀ x : ℝ, a * (4 - Real.sin x)^4 - 3 + (Real.cos x)^2 + a > 0} = {a : ℝ | a > 3/82} := by
  sorry

end inequality_range_l1591_159173


namespace star_value_for_specific_conditions_l1591_159118

-- Define the operation * for non-zero integers
def star (a b : ℤ) : ℚ := (1 : ℚ) / a + (1 : ℚ) / b

-- Theorem statement
theorem star_value_for_specific_conditions 
  (a b : ℤ) 
  (h1 : a ≠ 0) 
  (h2 : b ≠ 0) 
  (h3 : a + b = 15) 
  (h4 : a * b = 36) : 
  star a b = 5 / 12 := by
  sorry

end star_value_for_specific_conditions_l1591_159118


namespace point_three_units_from_negative_four_l1591_159136

theorem point_three_units_from_negative_four (x : ℝ) : 
  (x = -4 - 3 ∨ x = -4 + 3) ↔ |x - (-4)| = 3 :=
by sorry

end point_three_units_from_negative_four_l1591_159136


namespace cheese_fries_cost_is_eight_l1591_159134

/-- Represents the cost of items and money brought by Jim and his cousin --/
structure RestaurantScenario where
  cheeseburger_cost : ℚ
  milkshake_cost : ℚ
  jim_money : ℚ
  cousin_money : ℚ
  spent_percentage : ℚ

/-- Calculates the cost of cheese fries given a RestaurantScenario --/
def cheese_fries_cost (scenario : RestaurantScenario) : ℚ :=
  let total_money := scenario.jim_money + scenario.cousin_money
  let total_spent := scenario.spent_percentage * total_money
  let burger_shake_cost := 2 * (scenario.cheeseburger_cost + scenario.milkshake_cost)
  total_spent - burger_shake_cost

/-- Theorem stating that the cost of cheese fries is 8 given the specific scenario --/
theorem cheese_fries_cost_is_eight :
  let scenario := {
    cheeseburger_cost := 3,
    milkshake_cost := 5,
    jim_money := 20,
    cousin_money := 10,
    spent_percentage := 4/5
  }
  cheese_fries_cost scenario = 8 := by
  sorry


end cheese_fries_cost_is_eight_l1591_159134


namespace arrangement_count_l1591_159194

def committee_size : ℕ := 12
def num_men : ℕ := 3
def num_women : ℕ := 9

theorem arrangement_count :
  (committee_size.choose num_men) = 220 := by
  sorry

end arrangement_count_l1591_159194


namespace matrix_inverse_proof_l1591_159123

def A : Matrix (Fin 2) (Fin 2) ℝ := !![5, -3; -2, 1]

theorem matrix_inverse_proof :
  A⁻¹ = !![(-1), (-3); (-2), (-5)] := by
  sorry

end matrix_inverse_proof_l1591_159123


namespace bridge_length_l1591_159189

/-- The length of a bridge given train parameters and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : 
  train_length = 140 ∧ train_speed_kmh = 45 ∧ crossing_time = 30 →
  ∃ (bridge_length : ℝ), bridge_length = 235 := by
sorry

end bridge_length_l1591_159189


namespace sum_of_coefficients_l1591_159184

/-- A function f satisfying the given conditions -/
def f (x : ℝ) : ℝ := sorry

/-- The coefficients of f in its quadratic form -/
def a : ℝ := sorry
def b : ℝ := sorry
def c : ℝ := sorry

/-- The main theorem stating that a + b + c = 50 -/
theorem sum_of_coefficients :
  (∀ x, f (x + 5) = 5 * x^2 + 9 * x + 6) →
  (∀ x, f x = a * x^2 + b * x + c) →
  a + b + c = 50 := by sorry

end sum_of_coefficients_l1591_159184


namespace original_number_proof_l1591_159159

theorem original_number_proof (x y : ℝ) : 
  x = 19 ∧ 8 * x + 3 * y = 203 → x + y = 36 := by
  sorry

end original_number_proof_l1591_159159


namespace max_min_on_interval_l1591_159188

def f (x : ℝ) := x^3 - 3*x^2 + 5

theorem max_min_on_interval :
  ∃ (min max : ℝ),
    (∀ x ∈ Set.Icc 1 3, f x ≥ min ∧ f x ≤ max) ∧
    (∃ x₁ ∈ Set.Icc 1 3, f x₁ = min) ∧
    (∃ x₂ ∈ Set.Icc 1 3, f x₂ = max) ∧
    min = 1 ∧ max = 5 := by
  sorry

end max_min_on_interval_l1591_159188


namespace people_who_left_line_l1591_159127

theorem people_who_left_line (initial : ℕ) (joined : ℕ) (final : ℕ) : 
  initial = 7 → joined = 8 → final = 11 → initial - (initial - final + joined) = 4 := by
  sorry

end people_who_left_line_l1591_159127


namespace triangle_inradius_l1591_159102

/-- Given a triangle with perimeter 60 cm and area 75 cm², its inradius is 2.5 cm. -/
theorem triangle_inradius (perimeter : ℝ) (area : ℝ) (inradius : ℝ) :
  perimeter = 60 ∧ area = 75 → inradius = 2.5 := by
  sorry

#check triangle_inradius

end triangle_inradius_l1591_159102


namespace max_probability_dice_difference_l1591_159105

def roll_dice : Finset (ℕ × ℕ) := Finset.product (Finset.range 6) (Finset.range 6)

def difference (roll : ℕ × ℕ) : ℤ := (roll.1 : ℤ) - (roll.2 : ℤ)

def target_differences : Finset ℤ := {-2, -1, 0, 1, 2}

def favorable_outcomes : Finset (ℕ × ℕ) :=
  roll_dice.filter (λ roll => difference roll ∈ target_differences)

theorem max_probability_dice_difference :
  (favorable_outcomes.card : ℚ) / roll_dice.card = 1 / 6 := by sorry

end max_probability_dice_difference_l1591_159105


namespace arithmetic_equation_l1591_159150

theorem arithmetic_equation : 12 - 11 + (9 * 8) + 7 - (6 * 5) + 4 - 3 = 51 := by
  sorry

end arithmetic_equation_l1591_159150


namespace kids_wearing_socks_l1591_159135

theorem kids_wearing_socks (total : ℕ) (wearing_shoes : ℕ) (wearing_both : ℕ) (barefoot : ℕ) :
  total = 22 →
  wearing_shoes = 8 →
  wearing_both = 6 →
  barefoot = 8 →
  total - barefoot - (wearing_shoes - wearing_both) = 12 :=
by sorry

end kids_wearing_socks_l1591_159135


namespace complex_fraction_equality_l1591_159172

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem complex_fraction_equality : i / (2 + i) = (1 + 2*i) / 5 := by
  sorry

end complex_fraction_equality_l1591_159172


namespace cube_painting_l1591_159187

theorem cube_painting (m : ℚ) : 
  m > 0 → 
  let n : ℚ := 12 / m
  6 * (n - 2)^2 = 12 * (n - 2) → 
  m = 3 := by
sorry

end cube_painting_l1591_159187


namespace bales_in_barn_l1591_159124

/-- The number of bales in the barn after stacking more bales -/
def total_bales (initial : ℕ) (stacked : ℕ) : ℕ := initial + stacked

/-- Theorem: Given 22 initial bales and 67 newly stacked bales, the total is 89 bales -/
theorem bales_in_barn : total_bales 22 67 = 89 := by
  sorry

end bales_in_barn_l1591_159124


namespace solutions_for_14_solutions_for_0_1_solutions_for_neg_0_0544_l1591_159120

-- Define the equation
def f (x : ℝ) := (x - 1) * (2 * x - 3) * (3 * x - 4) * (6 * x - 5)

-- Theorem for a = 14
theorem solutions_for_14 :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 14 ∧ f x₂ = 14 ∧
  ∀ x : ℝ, f x = 14 → x = x₁ ∨ x = x₂ :=
sorry

-- Theorem for a = 0.1
theorem solutions_for_0_1 :
  ∃ x₁ x₂ x₃ x₄ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
  f x₁ = 0.1 ∧ f x₂ = 0.1 ∧ f x₃ = 0.1 ∧ f x₄ = 0.1 ∧
  ∀ x : ℝ, f x = 0.1 → x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄ :=
sorry

-- Theorem for a = -0.0544
theorem solutions_for_neg_0_0544 :
  ∃ x₁ x₂ x₃ x₄ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
  f x₁ = -0.0544 ∧ f x₂ = -0.0544 ∧ f x₃ = -0.0544 ∧ f x₄ = -0.0544 ∧
  ∀ x : ℝ, f x = -0.0544 → x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄ :=
sorry

end solutions_for_14_solutions_for_0_1_solutions_for_neg_0_0544_l1591_159120


namespace min_max_area_14_sided_lattice_polygon_l1591_159179

/-- A lattice point in 2D space -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- A lattice polygon is a polygon with all vertices at lattice points -/
structure LatticePolygon where
  vertices : List LatticePoint
  is_convex : Bool
  is_closed : Bool

/-- A lattice parallelogram is a parallelogram with all vertices at lattice points -/
structure LatticeParallelogram where
  vertices : List LatticePoint
  is_parallelogram : Bool

def area (p : LatticeParallelogram) : ℚ :=
  sorry

def can_be_divided_into_parallelograms (poly : LatticePolygon) (parallelograms : List LatticeParallelogram) : Prop :=
  sorry

theorem min_max_area_14_sided_lattice_polygon :
  ∀ (poly : LatticePolygon) (parallelograms : List LatticeParallelogram),
    poly.vertices.length = 14 →
    poly.is_convex →
    can_be_divided_into_parallelograms poly parallelograms →
    (∃ (C : ℚ), ∀ (p : LatticeParallelogram), p ∈ parallelograms → area p ≤ C) →
    (∀ (C : ℚ), (∀ (p : LatticeParallelogram), p ∈ parallelograms → area p ≤ C) → C ≥ 5) :=
  sorry

end min_max_area_14_sided_lattice_polygon_l1591_159179


namespace fraction_simplification_l1591_159183

theorem fraction_simplification :
  (5 : ℝ) / (Real.sqrt 48 + 3 * Real.sqrt 75 + 5 * Real.sqrt 27) = (5 * Real.sqrt 3) / 102 := by
  sorry

end fraction_simplification_l1591_159183


namespace josh_age_at_marriage_l1591_159191

/-- Proves that Josh's age at marriage was 22 given the conditions of the problem -/
theorem josh_age_at_marriage :
  ∀ (josh_age_at_marriage : ℕ),
    (josh_age_at_marriage + 30 + (28 + 30) = 5 * josh_age_at_marriage) →
    josh_age_at_marriage = 22 :=
by
  sorry

end josh_age_at_marriage_l1591_159191


namespace trig_simplification_l1591_159145

theorem trig_simplification :
  (Real.cos (5 * π / 180))^2 - (Real.sin (5 * π / 180))^2 =
  2 * Real.sin (40 * π / 180) * Real.cos (40 * π / 180) := by
  sorry

end trig_simplification_l1591_159145


namespace sn_double_angle_l1591_159132

-- Define the cosine and sine functions
noncomputable def cs : Real → Real := Real.cos
noncomputable def sn : Real → Real := Real.sin

-- Define the theorem
theorem sn_double_angle (α : Real) 
  (h1 : cs (α + Real.pi/2) = 3/5) 
  (h2 : -Real.pi/2 < α ∧ α < Real.pi/2) : 
  sn (2*α) = -24/25 := by
  sorry

end sn_double_angle_l1591_159132


namespace point_translation_coordinates_equal_l1591_159112

theorem point_translation_coordinates_equal (m : ℝ) : 
  let A : ℝ × ℝ := (m, 2)
  let B : ℝ × ℝ := (m + 1, 5)
  (B.1 = B.2) → m = 4 := by
  sorry

end point_translation_coordinates_equal_l1591_159112


namespace trains_crossing_time_l1591_159155

/-- Time for two trains to cross each other -/
theorem trains_crossing_time
  (length1 : ℝ) (length2 : ℝ) (speed1 : ℝ) (speed2 : ℝ)
  (h1 : length1 = 140)
  (h2 : length2 = 160)
  (h3 : speed1 = 60)
  (h4 : speed2 = 48)
  : (length1 + length2) / (speed1 + speed2) * (1000 / 3600) = 10 := by
  sorry

end trains_crossing_time_l1591_159155


namespace worker_savings_proof_l1591_159174

/-- Represents the fraction of take-home pay saved each month -/
def savings_fraction : ℚ := 2 / 5

theorem worker_savings_proof (monthly_pay : ℝ) (h1 : monthly_pay > 0) :
  let yearly_savings := 12 * savings_fraction * monthly_pay
  let monthly_not_saved := (1 - savings_fraction) * monthly_pay
  yearly_savings = 8 * monthly_not_saved :=
by sorry

end worker_savings_proof_l1591_159174


namespace passing_percentage_l1591_159142

def total_marks : ℕ := 400
def student_marks : ℕ := 100
def failing_margin : ℕ := 40

theorem passing_percentage :
  (student_marks + failing_margin) * 100 / total_marks = 35 := by
sorry

end passing_percentage_l1591_159142


namespace find_x_l1591_159130

theorem find_x : ∃ x : ℝ, 3 * x = (26 - x) + 14 ∧ x = 10 := by sorry

end find_x_l1591_159130


namespace min_ticket_cost_l1591_159148

theorem min_ticket_cost (total_tickets : ℕ) (price_low price_high : ℕ) 
  (h_total : total_tickets = 140)
  (h_price_low : price_low = 6)
  (h_price_high : price_high = 10)
  (h_constraint : ∀ x : ℕ, x ≤ total_tickets → total_tickets - x ≥ 2 * x → x ≤ 46) :
  ∃ (low_count high_count : ℕ),
    low_count + high_count = total_tickets ∧
    high_count ≥ 2 * low_count ∧
    low_count = 46 ∧
    high_count = 94 ∧
    low_count * price_low + high_count * price_high = 1216 ∧
    (∀ (a b : ℕ), a + b = total_tickets → b ≥ 2 * a → 
      a * price_low + b * price_high ≥ 1216) :=
by sorry

end min_ticket_cost_l1591_159148


namespace three_digit_number_rearrangement_l1591_159140

theorem three_digit_number_rearrangement (a b c : ℕ) : 
  a < 10 → b < 10 → c < 10 → a ≠ 0 →
  (100 * a + 10 * b + c) + 
  (100 * a + 10 * c + b) + 
  (100 * b + 10 * c + a) + 
  (100 * b + 10 * a + c) + 
  (100 * c + 10 * a + b) + 
  (100 * c + 10 * b + a) = 4422 →
  a + b + c ≥ 18 →
  100 * a + 10 * b + c = 785 :=
by sorry

end three_digit_number_rearrangement_l1591_159140


namespace golden_ratio_geometric_sequence_l1591_159161

theorem golden_ratio_geometric_sequence : 
  let x : ℝ := (1 + Real.sqrt 5) / 2
  let int_part := ⌊x⌋
  let frac_part := x - int_part
  (frac_part * x = int_part * int_part) ∧ (int_part * x = x * x) := by
  sorry

end golden_ratio_geometric_sequence_l1591_159161


namespace min_value_theorem_l1591_159178

theorem min_value_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 12) :
  2 * x + 3 * y + 6 * z ≥ 18 * Real.rpow 2 (1/3) ∧
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ * y₀ * z₀ = 12 ∧
  2 * x₀ + 3 * y₀ + 6 * z₀ = 18 * Real.rpow 2 (1/3) := by
sorry

end min_value_theorem_l1591_159178


namespace area_invariant_under_opposite_vertex_translation_l1591_159104

/-- Represents a vector in 2D space -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a quadrilateral in 2D space -/
structure Quadrilateral where
  A : Point2D
  B : Point2D
  C : Point2D
  D : Point2D

/-- Calculates the area of a quadrilateral -/
def area (q : Quadrilateral) : ℝ :=
  sorry

/-- Moves a point by a given vector -/
def movePoint (p : Point2D) (v : Vector2D) : Point2D :=
  { x := p.x + v.x, y := p.y + v.y }

/-- Theorem: The area of a quadrilateral remains unchanged when two opposite vertices
    are moved by the same vector -/
theorem area_invariant_under_opposite_vertex_translation (q : Quadrilateral) (v : Vector2D) :
  let q' := { q with
    A := movePoint q.A v,
    C := movePoint q.C v
  }
  area q = area q' :=
sorry

end area_invariant_under_opposite_vertex_translation_l1591_159104


namespace parking_lot_wheels_l1591_159110

/-- The number of wheels on a car -/
def car_wheels : ℕ := 4

/-- The number of wheels on a bike -/
def bike_wheels : ℕ := 2

/-- The number of cars in the parking lot -/
def num_cars : ℕ := 10

/-- The number of bikes in the parking lot -/
def num_bikes : ℕ := 2

/-- The total number of wheels in the parking lot -/
def total_wheels : ℕ := num_cars * car_wheels + num_bikes * bike_wheels

theorem parking_lot_wheels : total_wheels = 44 := by
  sorry

end parking_lot_wheels_l1591_159110


namespace smallest_four_digit_multiple_l1591_159144

theorem smallest_four_digit_multiple : ∃ n : ℕ, 
  (n ≥ 1000 ∧ n < 10000) ∧ 
  3 ∣ n ∧ 4 ∣ n ∧ 5 ∣ n ∧
  (∀ m : ℕ, m ≥ 1000 ∧ m < 10000 ∧ 3 ∣ m ∧ 4 ∣ m ∧ 5 ∣ m → m ≥ n) ∧
  n = 1020 :=
by sorry

end smallest_four_digit_multiple_l1591_159144


namespace flying_blade_diameter_l1591_159185

theorem flying_blade_diameter (d : ℝ) (n : ℤ) : 
  d = 0.0000009 → d = 9 * 10^n → n = -7 := by sorry

end flying_blade_diameter_l1591_159185


namespace martin_trip_distance_is_1185_l1591_159121

/-- Calculates the total distance traveled during Martin's business trip --/
def martin_trip_distance : ℝ :=
  let segment1 := 70 * 3
  let segment2 := 80 * 4
  let segment3 := 65 * 3
  let segment4 := 50 * 2
  let segment5 := 90 * 4
  segment1 + segment2 + segment3 + segment4 + segment5

/-- Theorem stating that Martin's total trip distance is 1185 km --/
theorem martin_trip_distance_is_1185 :
  martin_trip_distance = 1185 := by sorry

end martin_trip_distance_is_1185_l1591_159121


namespace all_students_same_classroom_l1591_159146

/-- The probability that all three students choose the same classroom when randomly selecting between two classrooms with equal probability. -/
theorem all_students_same_classroom (num_classrooms : ℕ) (num_students : ℕ) : 
  num_classrooms = 2 → num_students = 3 → (1 : ℚ) / 4 = 
    (1 : ℚ) / num_classrooms^num_students + (1 : ℚ) / num_classrooms^num_students :=
by sorry

end all_students_same_classroom_l1591_159146


namespace lyras_initial_budget_l1591_159154

-- Define the cost of the chicken bucket
def chicken_cost : ℕ := 12

-- Define the cost per pound of beef
def beef_cost_per_pound : ℕ := 3

-- Define the number of pounds of beef bought
def beef_pounds : ℕ := 5

-- Define the amount left in the budget
def amount_left : ℕ := 53

-- Theorem to prove
theorem lyras_initial_budget :
  chicken_cost + beef_cost_per_pound * beef_pounds + amount_left = 80 := by
  sorry

end lyras_initial_budget_l1591_159154


namespace chlorine_atomic_weight_l1591_159152

/-- The atomic weight of hydrogen in g/mol -/
def hydrogen_weight : ℝ := 1

/-- The atomic weight of oxygen in g/mol -/
def oxygen_weight : ℝ := 16

/-- The total molecular weight of the compound in g/mol -/
def total_weight : ℝ := 68

/-- The atomic weight of chlorine in g/mol -/
def chlorine_weight : ℝ := total_weight - hydrogen_weight - 2 * oxygen_weight

theorem chlorine_atomic_weight : chlorine_weight = 35 := by
  sorry

end chlorine_atomic_weight_l1591_159152


namespace systematic_sampling_l1591_159158

theorem systematic_sampling (total_students : Nat) (sample_size : Nat) (included_number : Nat) (group_number : Nat) : 
  total_students = 50 →
  sample_size = 10 →
  included_number = 46 →
  group_number = 7 →
  (included_number - (3 * (total_students / sample_size))) = 31 := by
sorry

end systematic_sampling_l1591_159158


namespace circle_equation_l1591_159197

/-- The standard equation of a circle with given center and radius -/
theorem circle_equation (x y : ℝ) : 
  (∃ (C : ℝ × ℝ) (r : ℝ), C = (1, -2) ∧ r = 3) →
  ((x - 1)^2 + (y + 2)^2 = 9) :=
by sorry

end circle_equation_l1591_159197


namespace cubic_equation_solutions_no_solution_for_2009_l1591_159196

theorem cubic_equation_solutions (n : ℕ+) :
  (∃ (x y : ℤ), x^3 - 3*x*y^2 + y^3 = n) →
  (∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℤ),
    (x₁^3 - 3*x₁*y₁^2 + y₁^3 = n) ∧
    (x₂^3 - 3*x₂*y₂^2 + y₂^3 = n) ∧
    (x₃^3 - 3*x₃*y₃^2 + y₃^3 = n) ∧
    (x₁, y₁) ≠ (x₂, y₂) ∧ (x₁, y₁) ≠ (x₃, y₃) ∧ (x₂, y₂) ≠ (x₃, y₃)) :=
by sorry

theorem no_solution_for_2009 :
  ¬ ∃ (x y : ℤ), x^3 - 3*x*y^2 + y^3 = 2009 :=
by sorry

end cubic_equation_solutions_no_solution_for_2009_l1591_159196


namespace new_person_weight_l1591_159177

def initial_group_size : ℕ := 4
def average_weight_increase : ℝ := 3
def replaced_person_weight : ℝ := 70

theorem new_person_weight :
  let total_weight_increase : ℝ := initial_group_size * average_weight_increase
  let new_person_weight : ℝ := replaced_person_weight + total_weight_increase
  new_person_weight = 82 :=
by sorry

end new_person_weight_l1591_159177


namespace max_baggies_count_l1591_159122

def cookies_per_bag : ℕ := 3
def chocolate_chip_cookies : ℕ := 2
def oatmeal_cookies : ℕ := 16

theorem max_baggies_count : 
  (chocolate_chip_cookies + oatmeal_cookies) / cookies_per_bag = 6 := by
  sorry

end max_baggies_count_l1591_159122


namespace geometric_difference_sequence_properties_l1591_159182

/-- A geometric difference sequence -/
def GeometricDifferenceSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, (a (n + 2) / a (n + 1)) - (a (n + 1) / a n) = d

theorem geometric_difference_sequence_properties
  (a : ℕ → ℚ)
  (h_gds : GeometricDifferenceSequence a)
  (h_a1 : a 1 = 1)
  (h_a2 : a 2 = 1)
  (h_a3 : a 3 = 3) :
  a 5 = 105 ∧ a 31 / a 29 = 3363 := by
  sorry

end geometric_difference_sequence_properties_l1591_159182


namespace perpendicular_bisector_b_value_l1591_159166

/-- The line x + y = b is a perpendicular bisector of the line segment from (2, 5) to (8, 11) -/
def is_perpendicular_bisector (b : ℝ) : Prop :=
  let midpoint := ((2 + 8) / 2, (5 + 11) / 2)
  midpoint.1 + midpoint.2 = b

/-- The value of b for which x + y = b is a perpendicular bisector of the line segment from (2, 5) to (8, 11) is 13 -/
theorem perpendicular_bisector_b_value :
  ∃ b : ℝ, is_perpendicular_bisector b ∧ b = 13 := by
  sorry

end perpendicular_bisector_b_value_l1591_159166


namespace certain_number_sum_l1591_159117

theorem certain_number_sum : ∃ x : ℝ, x = 5.46 - 3.97 ∧ x + 5.46 = 6.95 := by
  sorry

end certain_number_sum_l1591_159117


namespace basketball_team_callbacks_l1591_159113

theorem basketball_team_callbacks (girls boys cut : ℕ) 
  (h1 : girls = 17)
  (h2 : boys = 32)
  (h3 : cut = 39) :
  girls + boys - cut = 10 := by
sorry

end basketball_team_callbacks_l1591_159113


namespace axis_of_symmetry_cosine_l1591_159181

theorem axis_of_symmetry_cosine (x : ℝ) : 
  ∀ k : ℤ, (2 * x + π / 3 = k * π) ↔ x = -π / 6 := by sorry

end axis_of_symmetry_cosine_l1591_159181


namespace sum_of_r_values_l1591_159149

/-- Given two quadratic equations with a common real root, prove the sum of possible values of r -/
theorem sum_of_r_values (r : ℝ) : 
  (∃ x : ℝ, x^2 + (r-1)*x + 6 = 0 ∧ x^2 + (2*r+1)*x + 22 = 0) → 
  (∃ r1 r2 : ℝ, (r = r1 ∨ r = r2) ∧ r1 + r2 = 12/5) :=
by sorry

end sum_of_r_values_l1591_159149


namespace equilateral_triangle_area_perimeter_ratio_l1591_159106

/-- The ratio of the area to the perimeter of an equilateral triangle with side length 6 -/
theorem equilateral_triangle_area_perimeter_ratio : 
  let side_length : ℝ := 6
  let area : ℝ := side_length^2 * Real.sqrt 3 / 4
  let perimeter : ℝ := 3 * side_length
  area / perimeter = Real.sqrt 3 / 2 := by
  sorry

end equilateral_triangle_area_perimeter_ratio_l1591_159106


namespace inverse_g_at_124_l1591_159133

noncomputable def g (x : ℝ) : ℝ := 5 * x^3 - 4 * x + 1

theorem inverse_g_at_124 : g⁻¹ 124 = 3 := by sorry

end inverse_g_at_124_l1591_159133


namespace paco_initial_salty_cookies_l1591_159167

/-- Represents the number of cookies Paco has --/
structure CookieCount where
  salty : ℕ
  sweet : ℕ

/-- The problem of determining Paco's initial salty cookie count --/
theorem paco_initial_salty_cookies 
  (initial : CookieCount) 
  (eaten : CookieCount) 
  (final : CookieCount) : 
  (initial.sweet = 17) →
  (eaten.sweet = 14) →
  (eaten.salty = 9) →
  (final.salty = 17) →
  (initial.salty = final.salty + eaten.salty) →
  (initial.salty = 26) := by
sorry


end paco_initial_salty_cookies_l1591_159167


namespace system_solution_relation_l1591_159129

theorem system_solution_relation (a₁ a₂ c₁ c₂ : ℝ) :
  (2 * a₁ + 3 = c₁ ∧ 2 * a₂ + 3 = c₂) →
  (-1 * a₁ + (-3) = a₁ - c₁ ∧ -1 * a₂ + (-3) = a₂ - c₂) :=
by sorry

end system_solution_relation_l1591_159129
