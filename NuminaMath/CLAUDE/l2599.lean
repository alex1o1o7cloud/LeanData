import Mathlib

namespace smallest_n_for_candy_l2599_259962

theorem smallest_n_for_candy (n : ℕ) : 
  (∃ m : ℕ, m > 0 ∧ 25 * m % 10 = 0 ∧ 25 * m % 16 = 0 ∧ 25 * m % 18 = 0) →
  (25 * n % 10 = 0 ∧ 25 * n % 16 = 0 ∧ 25 * n % 18 = 0) →
  n ≥ 29 :=
sorry

end smallest_n_for_candy_l2599_259962


namespace equation_solutions_l2599_259956

-- Define the function representing the left side of the equation
def f (x : ℝ) : ℝ := (18 * x - 1) ^ (1/3) - (10 * x + 1) ^ (1/3) - 3 * x ^ (1/3)

-- Define the set of solutions
def solutions : Set ℝ := {0, -5/8317, -60/1614}

-- Theorem statement
theorem equation_solutions :
  ∀ x : ℝ, f x = 0 ↔ x ∈ solutions :=
sorry

end equation_solutions_l2599_259956


namespace shoes_sold_this_week_l2599_259988

def monthly_goal : ℕ := 80
def sold_last_week : ℕ := 27
def still_needed : ℕ := 41

theorem shoes_sold_this_week :
  monthly_goal - sold_last_week - still_needed = 12 := by
  sorry

end shoes_sold_this_week_l2599_259988


namespace circle_area_special_condition_l2599_259992

/-- For a circle where three times the reciprocal of its circumference equals half its diameter, 
    the area of the circle is 3/2. -/
theorem circle_area_special_condition (r : ℝ) (h : 3 * (1 / (2 * π * r)) = 1/2 * (2 * r)) : 
  π * r^2 = 3/2 := by
  sorry

end circle_area_special_condition_l2599_259992


namespace slope_intercept_product_specific_line_l2599_259993

/-- A line in a coordinate plane. -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- The product of a line's slope and y-intercept. -/
def slopeInterceptProduct (l : Line) : ℝ := l.slope * l.yIntercept

/-- Theorem: For a line with y-intercept -2 and slope 3, the product of its slope and y-intercept is -6. -/
theorem slope_intercept_product_specific_line :
  ∃ l : Line, l.yIntercept = -2 ∧ l.slope = 3 ∧ slopeInterceptProduct l = -6 := by
  sorry

end slope_intercept_product_specific_line_l2599_259993


namespace decimal_sum_theorem_l2599_259923

theorem decimal_sum_theorem : 0.03 + 0.004 + 0.009 + 0.0001 = 0.0431 := by
  sorry

end decimal_sum_theorem_l2599_259923


namespace hyperbola_equation_from_ellipse_foci_l2599_259914

/-- Given an ellipse and a hyperbola E, if the hyperbola has the foci of the ellipse as its vertices,
    then the equation of the hyperbola E can be determined. -/
theorem hyperbola_equation_from_ellipse_foci (x y : ℝ) :
  (x^2 / 10 + y^2 / 5 = 1) →  -- Equation of the ellipse
  (∃ k : ℝ, 3*x + 4*y = k) →  -- Asymptote equation of hyperbola E
  (∃ a b : ℝ, a^2 = 5 ∧ x^2 / 10 + y^2 / 5 = 1 → (x = a ∨ x = -a) ∧ y = 0) →  -- Foci of ellipse as vertices of hyperbola
  (x^2 / 5 - 16*y^2 / 45 = 1)  -- Equation of hyperbola E
:= by sorry

end hyperbola_equation_from_ellipse_foci_l2599_259914


namespace proposition_is_false_l2599_259953

theorem proposition_is_false : ¬(∀ x : ℝ, x ≠ 1 → x^2 - 1 ≠ 0) := by sorry

end proposition_is_false_l2599_259953


namespace complex_equation_sum_l2599_259968

theorem complex_equation_sum (a b : ℝ) : 
  (a - Complex.I) * Complex.I = -b + 2 * Complex.I → a + b = 1 := by
  sorry

end complex_equation_sum_l2599_259968


namespace g_properties_and_range_l2599_259906

def f (x : ℝ) : ℝ := x^2 - 3*x + 2

def g (x : ℝ) : ℝ := |x|^2 - 3*|x| + 2

theorem g_properties_and_range :
  (∀ x : ℝ, g (-x) = g x) ∧
  (∀ x : ℝ, x ≥ 0 → g x = f x) ∧
  ({m : ℝ | g m > 2} = {m : ℝ | m < -3 ∨ m > 3}) := by
  sorry

end g_properties_and_range_l2599_259906


namespace factorization_problem_l2599_259938

theorem factorization_problem (x y : ℝ) : (x - y)^2 - 2*(x - y) + 1 = (x - y - 1)^2 := by
  sorry

end factorization_problem_l2599_259938


namespace sector_central_angle_l2599_259991

/-- Given a circular sector with radius 8 cm and area 4 cm², 
    prove that its central angle measures 1/8 radians. -/
theorem sector_central_angle (radius : ℝ) (area : ℝ) (angle : ℝ) :
  radius = 8 →
  area = 4 →
  area = 1/2 * angle * radius^2 →
  angle = 1/8 := by
  sorry

end sector_central_angle_l2599_259991


namespace fraction_simplification_l2599_259904

theorem fraction_simplification (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (a^(2*b) * b^a) / (b^(2*a) * a^b) = (a/b)^b := by sorry

end fraction_simplification_l2599_259904


namespace arithmetic_sequence_problem_l2599_259982

/-- An arithmetic sequence with its sum sequence -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  S : ℕ → ℤ  -- The sum sequence
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_property : ∀ n, S n = (n * (a 1 + a n)) / 2

/-- Theorem: For an arithmetic sequence satisfying given conditions, a₈ = -26 -/
theorem arithmetic_sequence_problem (seq : ArithmeticSequence)
    (h1 : seq.S 6 = 8 * seq.S 3)
    (h2 : seq.a 3 - seq.a 5 = 8) :
  seq.a 8 = -26 := by
  sorry


end arithmetic_sequence_problem_l2599_259982


namespace vector_collinearity_l2599_259932

def a : ℝ × ℝ := (3, 1)
def b (x : ℝ) : ℝ × ℝ := (x, -1)

theorem vector_collinearity (x : ℝ) :
  (∃ (k : ℝ), a - b x = k • b x) → x = -3 := by
  sorry

end vector_collinearity_l2599_259932


namespace simplify_fraction_l2599_259964

theorem simplify_fraction : (4^5 + 4^3) / (4^4 - 4^2 + 2) = 544 / 121 := by
  sorry

end simplify_fraction_l2599_259964


namespace min_dividing_segment_length_l2599_259950

/-- A trapezoid with midsegment length 4 and a line parallel to the bases dividing its area in half -/
structure DividedTrapezoid where
  /-- The length of the midsegment -/
  midsegment_length : ℝ
  /-- The length of the lower base -/
  lower_base : ℝ
  /-- The length of the upper base -/
  upper_base : ℝ
  /-- The height of the trapezoid -/
  height : ℝ
  /-- The length of the segment created by the dividing line -/
  dividing_segment : ℝ
  /-- The midsegment length is 4 -/
  midsegment_eq : midsegment_length = 4
  /-- The dividing line splits the area in half -/
  area_split : (lower_base + dividing_segment) * (height / 2) = (upper_base + dividing_segment) * (height / 2)
  /-- The sum of the bases is twice the midsegment length -/
  bases_sum : lower_base + upper_base = 2 * midsegment_length

/-- The minimum possible length of the dividing segment is 4 -/
theorem min_dividing_segment_length (t : DividedTrapezoid) : 
  ∃ (min_length : ℝ), min_length = 4 ∧ ∀ (x : ℝ), t.dividing_segment ≥ min_length :=
by sorry

end min_dividing_segment_length_l2599_259950


namespace fraction_equation_solution_l2599_259996

theorem fraction_equation_solution : 
  ∃ x : ℝ, (1 / (x - 1) = 2 / (1 - x) + 1) ∧ (x = 4) := by
  sorry

end fraction_equation_solution_l2599_259996


namespace imaginary_part_of_z_l2599_259969

theorem imaginary_part_of_z (z : ℂ) (h : (z - Complex.I) * (1 + 2 * Complex.I) = Complex.I ^ 3) :
  z.im = 4 / 5 := by
  sorry

end imaginary_part_of_z_l2599_259969


namespace solution_implies_k_value_l2599_259922

theorem solution_implies_k_value (k : ℝ) :
  (∃ x y : ℝ, k * x + y = 5 ∧ x = 2 ∧ y = 1) → k = 2 := by
  sorry

end solution_implies_k_value_l2599_259922


namespace product_of_numbers_l2599_259902

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 24) (h2 : x^2 + y^2 = 404) : x * y = 86 := by
  sorry

end product_of_numbers_l2599_259902


namespace x_needs_seven_days_l2599_259957

/-- The number of days x needs to finish the remaining work -/
def remaining_days_for_x (x_days y_days y_worked_days : ℕ) : ℚ :=
  (y_days - y_worked_days) * x_days / y_days

/-- Theorem stating that x needs 7 days to finish the remaining work -/
theorem x_needs_seven_days (x_days y_days y_worked_days : ℕ) 
  (hx : x_days = 21)
  (hy : y_days = 15)
  (hw : y_worked_days = 10) :
  remaining_days_for_x x_days y_days y_worked_days = 7 := by
  sorry

#eval remaining_days_for_x 21 15 10

end x_needs_seven_days_l2599_259957


namespace clothing_price_difference_l2599_259900

theorem clothing_price_difference :
  ∀ (x y : ℝ),
    9 * x + 10 * y = 1810 →
    11 * x + 8 * y = 1790 →
    x - y = -10 :=
by sorry

end clothing_price_difference_l2599_259900


namespace pig_count_l2599_259965

/-- Given a group of pigs and hens, if the total number of legs is 22 more than twice 
    the total number of heads, then the number of pigs is 11. -/
theorem pig_count (pigs hens : ℕ) : 
  4 * pigs + 2 * hens = 2 * (pigs + hens) + 22 → pigs = 11 := by
  sorry

end pig_count_l2599_259965


namespace square_side_length_l2599_259944

theorem square_side_length (diagonal : ℝ) (h : diagonal = Real.sqrt 2) : 
  ∃ (side : ℝ), side * side + side * side = diagonal * diagonal ∧ side = 1 := by
  sorry

end square_side_length_l2599_259944


namespace percentage_increase_l2599_259930

theorem percentage_increase (original : ℝ) (new : ℝ) (increase_percent : ℝ) 
  (h1 : original = 50)
  (h2 : new = 80)
  (h3 : increase_percent = 60) :
  (new - original) / original * 100 = increase_percent :=
by sorry

end percentage_increase_l2599_259930


namespace factor_expression_l2599_259933

theorem factor_expression (y : ℝ) : 84 * y^13 + 210 * y^26 = 42 * y^13 * (2 + 5 * y^13) := by
  sorry

end factor_expression_l2599_259933


namespace euler_children_mean_age_l2599_259999

def euler_children_ages : List ℕ := [7, 7, 7, 12, 12, 14, 15]

theorem euler_children_mean_age :
  (euler_children_ages.sum : ℚ) / euler_children_ages.length = 74 / 7 := by
  sorry

end euler_children_mean_age_l2599_259999


namespace delta_max_ratio_l2599_259995

def charlie_day1_score : ℕ := 200
def charlie_day1_attempted : ℕ := 400
def charlie_day2_score : ℕ := 160
def charlie_day2_attempted : ℕ := 200
def total_points_attempted : ℕ := 600

def charlie_day1_ratio : ℚ := charlie_day1_score / charlie_day1_attempted
def charlie_day2_ratio : ℚ := charlie_day2_score / charlie_day2_attempted
def charlie_total_ratio : ℚ := (charlie_day1_score + charlie_day2_score) / total_points_attempted

theorem delta_max_ratio (delta_day1_score delta_day1_attempted delta_day2_score delta_day2_attempted : ℕ) :
  delta_day1_attempted + delta_day2_attempted = total_points_attempted →
  delta_day1_attempted ≠ charlie_day1_attempted →
  delta_day1_score > 0 →
  delta_day2_score > 0 →
  (delta_day1_score : ℚ) / delta_day1_attempted < charlie_day1_ratio →
  (delta_day2_score : ℚ) / delta_day2_attempted < charlie_day2_ratio →
  (delta_day1_score + delta_day2_score : ℚ) / total_points_attempted ≤ 479 / 600 :=
by sorry

end delta_max_ratio_l2599_259995


namespace sum_of_five_cubes_l2599_259963

theorem sum_of_five_cubes (n : ℤ) : ∃ a b c d e : ℤ, n = a^3 + b^3 + c^3 + d^3 + e^3 := by
  sorry

end sum_of_five_cubes_l2599_259963


namespace product_of_solutions_abs_y_eq_3_abs_y_minus_2_l2599_259986

theorem product_of_solutions_abs_y_eq_3_abs_y_minus_2 :
  ∃ (y₁ y₂ : ℝ), (|y₁| = 3*(|y₁| - 2)) ∧ (|y₂| = 3*(|y₂| - 2)) ∧ (y₁ ≠ y₂) ∧ (y₁ * y₂ = -9) :=
by sorry

end product_of_solutions_abs_y_eq_3_abs_y_minus_2_l2599_259986


namespace marco_run_time_l2599_259983

-- Define the track and run parameters
def total_laps : ℕ := 6
def track_length : ℝ := 450
def first_segment : ℝ := 150
def second_segment : ℝ := 300
def speed_first : ℝ := 5
def speed_second : ℝ := 4

-- Define the theorem
theorem marco_run_time :
  let time_first := first_segment / speed_first
  let time_second := second_segment / speed_second
  let time_per_lap := time_first + time_second
  let total_time := total_laps * time_per_lap
  total_time = 630 := by sorry

end marco_run_time_l2599_259983


namespace lucas_100_mod_9_l2599_259972

/-- Lucas sequence -/
def lucas : ℕ → ℕ
  | 0 => 1
  | 1 => 3
  | n + 2 => lucas n + lucas (n + 1)

/-- Lucas sequence modulo 9 has a period of 12 -/
axiom lucas_mod_9_period (n : ℕ) : lucas (n + 12) % 9 = lucas n % 9

/-- The 4th term of Lucas sequence modulo 9 is 7 -/
axiom lucas_4_mod_9 : lucas 3 % 9 = 7

theorem lucas_100_mod_9 : lucas 99 % 9 = 7 := by
  sorry

end lucas_100_mod_9_l2599_259972


namespace ball_speed_equality_time_ball_speed_equality_time_specific_l2599_259936

/-- The time when a ball's average speed equals its instantaneous speed after being dropped from a height and experiencing a perfectly elastic collision. -/
theorem ball_speed_equality_time
  (h : ℝ)  -- Initial height
  (g : ℝ)  -- Acceleration due to gravity
  (h_pos : h > 0)
  (g_pos : g > 0)
  : ∃ (t : ℝ), t > 0 ∧ t = Real.sqrt (2 * h / g + 8 * h / g) :=
by
  sorry

/-- The specific case where h = 45 m and g = 10 m/s² -/
theorem ball_speed_equality_time_specific :
  ∃ (t : ℝ), t > 0 ∧ t = Real.sqrt 18 :=
by
  sorry

end ball_speed_equality_time_ball_speed_equality_time_specific_l2599_259936


namespace Bob_is_shortest_l2599_259910

-- Define a type for the friends
inductive Friend
| Amy
| Bob
| Carla
| Dan
| Eric

-- Define a relation for "taller than"
def taller_than : Friend → Friend → Prop :=
  sorry

-- State the theorem
theorem Bob_is_shortest (h1 : taller_than Friend.Amy Friend.Carla)
                        (h2 : taller_than Friend.Eric Friend.Dan)
                        (h3 : taller_than Friend.Dan Friend.Bob)
                        (h4 : taller_than Friend.Carla Friend.Eric) :
  ∀ f : Friend, f ≠ Friend.Bob → taller_than f Friend.Bob :=
by sorry

end Bob_is_shortest_l2599_259910


namespace greatest_common_piece_length_l2599_259976

theorem greatest_common_piece_length : Nat.gcd 28 (Nat.gcd 42 70) = 14 := by
  sorry

end greatest_common_piece_length_l2599_259976


namespace solution_set_of_inequality_l2599_259909

-- Define the inequality function
def f (x : ℝ) : ℝ := |x - 2| * (x - 1)

-- State the theorem
theorem solution_set_of_inequality :
  {x : ℝ | f x < 2} = Set.Ioi (-Real.pi) ∩ Set.Iio 3 := by sorry

end solution_set_of_inequality_l2599_259909


namespace sum_abcd_equals_neg_ten_thirds_l2599_259954

theorem sum_abcd_equals_neg_ten_thirds 
  (a b c d : ℚ) 
  (h : a + 2 = b + 3 ∧ b + 3 = c + 4 ∧ c + 4 = d + 5 ∧ d + 5 = a + b + c + d + 6) : 
  a + b + c + d = -10/3 := by
sorry

end sum_abcd_equals_neg_ten_thirds_l2599_259954


namespace douglas_county_z_votes_l2599_259958

theorem douglas_county_z_votes
  (total_percent : ℝ)
  (county_x_percent : ℝ)
  (county_y_percent : ℝ)
  (county_x_voters : ℝ)
  (county_y_voters : ℝ)
  (county_z_voters : ℝ)
  (h1 : total_percent = 63)
  (h2 : county_x_percent = 74)
  (h3 : county_y_percent = 67)
  (h4 : county_x_voters = 3 * county_z_voters)
  (h5 : county_y_voters = 2 * county_z_voters)
  : ∃ county_z_percent : ℝ,
    county_z_percent = 22 ∧
    total_percent / 100 * (county_x_voters + county_y_voters + county_z_voters) =
    county_x_percent / 100 * county_x_voters +
    county_y_percent / 100 * county_y_voters +
    county_z_percent / 100 * county_z_voters :=
by sorry

end douglas_county_z_votes_l2599_259958


namespace trajectory_of_point_M_l2599_259947

/-- The trajectory of point M given the specified conditions -/
theorem trajectory_of_point_M :
  ∀ (x y : ℝ),
    x ≠ -1 →
    x ≠ 1 →
    y ≠ 0 →
    (y / (x + 1)) / (y / (x - 1)) = 2 →
    x = -3 :=
by sorry

end trajectory_of_point_M_l2599_259947


namespace range_of_a_when_B_subset_A_l2599_259974

/-- The set A -/
def A : Set ℝ := {x | x^2 + 4*x = 0}

/-- The set B parameterized by a -/
def B (a : ℝ) : Set ℝ := {x | x^2 + 2*(a+1)*x + a^2 - 1 = 0}

/-- The range of a -/
def range_a : Set ℝ := {a | a = 1 ∨ a ≤ -1}

/-- Theorem stating the range of a when B is a subset of A -/
theorem range_of_a_when_B_subset_A :
  ∀ a : ℝ, B a ⊆ A → a ∈ range_a :=
sorry

end range_of_a_when_B_subset_A_l2599_259974


namespace dollar_equality_l2599_259919

/-- Custom operation definition -/
def dollar (a b : ℝ) : ℝ := (a - b)^2

/-- Theorem statement -/
theorem dollar_equality (x y z : ℝ) : dollar ((x - y + z)^2) ((y - x - z)^2) = 0 := by
  sorry

end dollar_equality_l2599_259919


namespace quadratic_roots_l2599_259912

theorem quadratic_roots (a c : ℝ) (h1 : a ≠ 0) :
  let f : ℝ → ℝ := fun x ↦ a * x^2 - 2*a*x + c
  (f (-1) = 0) →
  (∃ x₁ x₂ : ℝ, x₁ = -1 ∧ x₂ = 3 ∧
    ∀ x : ℝ, (a * x^2 - 2*a*x + c = 0) ↔ (x = x₁ ∨ x = x₂)) :=
by sorry

end quadratic_roots_l2599_259912


namespace no_common_points_l2599_259973

theorem no_common_points (m : ℝ) : 
  (∀ x y : ℝ, x + m^2 * y + 6 = 0 ∧ (m - 2) * x + 3 * m * y + 2 * m = 0 → False) ↔ 
  (m = 0 ∨ m = -1) :=
sorry

end no_common_points_l2599_259973


namespace pure_imaginary_complex_fraction_l2599_259967

theorem pure_imaginary_complex_fraction (a : ℝ) : 
  (Complex.I : ℂ) ^ 2 = -1 →
  (∃ b : ℝ, (a + Complex.I) / (1 + 2 * Complex.I) = b * Complex.I) →
  a = -2 := by
  sorry

end pure_imaginary_complex_fraction_l2599_259967


namespace equation_proof_l2599_259935

theorem equation_proof : (12 : ℕ)^2 * 6^4 / 432 = 432 := by
  sorry

end equation_proof_l2599_259935


namespace inequality_solution_minimum_value_exists_minimum_l2599_259966

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1|

-- Define the function g
def g (x : ℝ) : ℝ := f (-x) + f (x + 5)

-- Theorem for part (I)
theorem inequality_solution (x : ℝ) : f x > 2 ↔ x > 3 ∨ x < -1 := by sorry

-- Theorem for part (II)
theorem minimum_value : ∀ x, g x ≥ 3 := by sorry

-- Theorem to show that 3 is indeed the minimum value
theorem exists_minimum : ∃ x, g x = 3 := by sorry

end inequality_solution_minimum_value_exists_minimum_l2599_259966


namespace exponent_equality_l2599_259952

theorem exponent_equality (y x : ℕ) (h1 : 16 ^ y = 4 ^ x) (h2 : y = 7) : x = 14 := by
  sorry

end exponent_equality_l2599_259952


namespace complex_number_in_first_quadrant_l2599_259941

theorem complex_number_in_first_quadrant : 
  let z : ℂ := (-2 + 3 * Complex.I) / Complex.I
  (z.re > 0) ∧ (z.im > 0) := by sorry

end complex_number_in_first_quadrant_l2599_259941


namespace alex_and_father_ages_l2599_259970

/-- Proves that given the conditions about Alex and his father's ages, 
    Alex is 9 years old and his father is 23 years old. -/
theorem alex_and_father_ages :
  ∀ (alex_age father_age : ℕ),
    father_age = 2 * alex_age + 5 →
    alex_age - 6 = alex_age / 3 →
    alex_age = 9 ∧ father_age = 23 := by
  sorry

end alex_and_father_ages_l2599_259970


namespace problem_solution_l2599_259918

noncomputable def f (t : ℝ) (x : ℝ) : ℝ := Real.exp x - t * (x + 1)

noncomputable def g (t : ℝ) (x : ℝ) : ℝ := f t x + t / Real.exp x

theorem problem_solution :
  (∀ t : ℝ, (∀ x : ℝ, x > 0 → f t x ≥ 0) → t ≤ 1) ∧
  (∀ t : ℝ, t ≤ -1 →
    (∀ m : ℝ, (∀ x₁ x₂ y₁ y₂ : ℝ, x₁ ≠ x₂ →
      y₁ = g t x₁ → y₂ = g t x₂ → (y₂ - y₁) / (x₂ - x₁) > m) → m < 3)) ∧
  (∀ n : ℕ, n > 0 →
    Real.log (1 + n) < (Finset.sum (Finset.range n) (λ i => 1 / (i + 1 : ℝ))) ∧
    (Finset.sum (Finset.range n) (λ i => 1 / (i + 1 : ℝ))) ≤ 1 + Real.log n) :=
by sorry

end problem_solution_l2599_259918


namespace complement_U_P_l2599_259903

-- Define the set U
def U : Set ℝ := {x | x^2 - 2*x < 3}

-- Define the set P
def P : Set ℝ := {x | -1 < x ∧ x ≤ 2}

-- Theorem statement
theorem complement_U_P : 
  (U \ P) = {x | 2 < x ∧ x < 3} := by sorry

end complement_U_P_l2599_259903


namespace colten_chicken_count_l2599_259905

/-- Represents the number of chickens each person has. -/
structure ChickenCount where
  colten : ℕ
  skylar : ℕ
  quentin : ℕ

/-- The conditions of the chicken problem. -/
def ChickenProblem (c : ChickenCount) : Prop :=
  c.colten + c.skylar + c.quentin = 383 ∧
  c.quentin = 25 + 2 * c.skylar ∧
  c.skylar = 3 * c.colten - 4

theorem colten_chicken_count :
  ∀ c : ChickenCount, ChickenProblem c → c.colten = 37 := by
  sorry

end colten_chicken_count_l2599_259905


namespace parabola_x_intercepts_l2599_259997

theorem parabola_x_intercepts :
  ∃! x : ℝ, ∃ y : ℝ, x = -3 * y^2 + 2 * y + 4 ∧ y = 0 :=
by sorry

end parabola_x_intercepts_l2599_259997


namespace munchausen_theorem_l2599_259971

/-- Represents a polynomial of degree n with n natural roots -/
structure PolynomialWithNaturalRoots (n : ℕ) where
  coeff_a : ℕ
  coeff_b : ℕ
  has_n_natural_roots : Bool

/-- Represents a configuration of lines in a plane -/
structure LineConfiguration where
  num_lines : ℕ
  num_intersections : ℕ

/-- 
Given a polynomial of degree n with n natural roots, 
there exists a configuration of lines in the plane 
where the number of lines equals the coefficient of x^(n-1) 
and the number of intersections equals the coefficient of x^(n-2)
-/
theorem munchausen_theorem {n : ℕ} (p : PolynomialWithNaturalRoots n) 
  (h : p.has_n_natural_roots = true) :
  ∃ (lc : LineConfiguration), lc.num_lines = p.coeff_a ∧ lc.num_intersections = p.coeff_b :=
sorry

end munchausen_theorem_l2599_259971


namespace abc_product_l2599_259908

theorem abc_product (a b c : ℕ) : 
  Prime a ∧ Prime b ∧ Prime c ∧
  (∃ k : ℕ, b + 8 = k * a) ∧
  (∃ m n : ℕ, b^2 - 1 = m * a ∧ b^2 - 1 = n * c) ∧
  b + c = a^2 - 1 →
  a * b * c = 2009 := by
sorry

end abc_product_l2599_259908


namespace derivative_inequality_implies_function_inequality_l2599_259942

theorem derivative_inequality_implies_function_inequality
  (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, deriv f x < 2 * f x) : 
  Real.exp 2 * f 0 > f 1 := by
  sorry

end derivative_inequality_implies_function_inequality_l2599_259942


namespace power_multiplication_l2599_259989

theorem power_multiplication (x : ℝ) : x^2 * x^3 = x^5 := by
  sorry

end power_multiplication_l2599_259989


namespace min_value_of_f_l2599_259943

def f (x : ℝ) : ℝ := x^3 - 3*x + 9

theorem min_value_of_f :
  ∃ (m : ℝ), (∀ (x : ℝ), f x ≥ m) ∧ (∃ (x : ℝ), f x = m) ∧ m = 7 := by
  sorry

end min_value_of_f_l2599_259943


namespace shoe_box_problem_l2599_259917

theorem shoe_box_problem (num_pairs : ℕ) (prob_match : ℚ) :
  num_pairs = 9 →
  prob_match = 1 / 17 →
  (num_pairs * 2 : ℕ) = 18 :=
by
  sorry

end shoe_box_problem_l2599_259917


namespace arrangement_theorem_l2599_259920

/-- The number of ways to arrange n boys and m girls in a row with girls standing together -/
def arrange_girls_together (n m : ℕ) : ℕ := sorry

/-- The number of ways to arrange n boys and m girls in a row with no two boys next to each other -/
def arrange_boys_apart (n m : ℕ) : ℕ := sorry

theorem arrangement_theorem :
  (arrange_girls_together 3 4 = 576) ∧ 
  (arrange_boys_apart 3 4 = 1440) := by sorry

end arrangement_theorem_l2599_259920


namespace f_odd_and_increasing_l2599_259960

-- Define the function f(x) = 2x
def f (x : ℝ) : ℝ := 2 * x

-- Theorem stating that f is both odd and increasing
theorem f_odd_and_increasing :
  (∀ x, f (-x) = -f x) ∧ (∀ x y, x < y → f x < f y) := by sorry

end f_odd_and_increasing_l2599_259960


namespace unique_a_value_l2599_259931

def A (a : ℝ) : Set ℝ := {a + 2, 2 * a^2 + a}

theorem unique_a_value : ∃! a : ℝ, 3 ∈ A a ∧ a = 3/2 := by
  sorry

end unique_a_value_l2599_259931


namespace geometric_sequence_sum_l2599_259934

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  (∀ n, a n > 0) →
  a 1 = 3 →
  a 1 + a 2 + a 3 = 21 →
  a 3 + a 4 + a 5 = 84 := by
  sorry

end geometric_sequence_sum_l2599_259934


namespace rectangle_longer_side_length_l2599_259978

/-- Given a circle of radius 6 cm tangent to three sides of a rectangle,
    and the area of the rectangle is three times the area of the circle,
    prove that the length of the longer side of the rectangle is 4.5π cm. -/
theorem rectangle_longer_side_length (r : ℝ) (circle_area rectangle_area : ℝ) 
  (h1 : r = 6)
  (h2 : circle_area = Real.pi * r^2)
  (h3 : rectangle_area = 3 * circle_area)
  (h4 : ∃ (shorter_side longer_side : ℝ), 
        shorter_side = 2 * (2 * r) ∧ 
        rectangle_area = shorter_side * longer_side) :
  ∃ (longer_side : ℝ), longer_side = 4.5 * Real.pi := by
  sorry

end rectangle_longer_side_length_l2599_259978


namespace arrangement_counts_l2599_259927

/-- The number of boys in the row -/
def num_boys : ℕ := 4

/-- The number of girls in the row -/
def num_girls : ℕ := 2

/-- The total number of students in the row -/
def total_students : ℕ := num_boys + num_girls

/-- The number of arrangements where Boy A does not stand at the head or the tail of the row -/
def arrangements_A_not_ends : ℕ := 480

/-- The number of arrangements where the two girls must stand next to each other -/
def arrangements_girls_together : ℕ := 240

/-- The number of arrangements where Students A, B, and C are not next to each other -/
def arrangements_ABC_not_adjacent : ℕ := 144

/-- The number of arrangements where A does not stand at the head, and B does not stand at the tail -/
def arrangements_A_not_head_B_not_tail : ℕ := 504

theorem arrangement_counts :
  arrangements_A_not_ends = 480 ∧
  arrangements_girls_together = 240 ∧
  arrangements_ABC_not_adjacent = 144 ∧
  arrangements_A_not_head_B_not_tail = 504 := by sorry

end arrangement_counts_l2599_259927


namespace car_distance_theorem_l2599_259949

/-- Represents the car's driving characteristics and total driving time -/
structure CarDriving where
  speed : ℕ              -- Speed in miles per hour
  drive_time : ℕ         -- Continuous driving time in hours
  cool_time : ℕ          -- Cooling time in hours
  total_time : ℕ         -- Total available time in hours

/-- Calculates the total distance a car can travel given its driving characteristics -/
def total_distance (car : CarDriving) : ℕ :=
  sorry

/-- Theorem stating that a car with given characteristics can travel 88 miles in 13 hours -/
theorem car_distance_theorem :
  let car := CarDriving.mk 8 5 1 13
  total_distance car = 88 := by
  sorry

end car_distance_theorem_l2599_259949


namespace coffee_maker_capacity_l2599_259926

/-- Represents a cylindrical coffee maker -/
structure CoffeeMaker where
  capacity : ℝ
  remaining : ℝ
  emptyPercentage : ℝ

/-- Theorem: A coffee maker with 30 cups remaining when 75% empty has a total capacity of 120 cups -/
theorem coffee_maker_capacity (cm : CoffeeMaker) 
  (h1 : cm.remaining = 30)
  (h2 : cm.emptyPercentage = 0.75)
  : cm.capacity = 120 := by
  sorry

end coffee_maker_capacity_l2599_259926


namespace furniture_purchase_proof_l2599_259911

/-- Calculates the number of furniture pieces purchased given the total payment, reimbursement, and cost per piece. -/
def furniture_pieces (total_payment : ℕ) (reimbursement : ℕ) (cost_per_piece : ℕ) : ℕ :=
  (total_payment - reimbursement) / cost_per_piece

/-- Proves that given the specific values in the problem, the number of furniture pieces is 150. -/
theorem furniture_purchase_proof :
  furniture_pieces 20700 600 134 = 150 := by
  sorry

#eval furniture_pieces 20700 600 134

end furniture_purchase_proof_l2599_259911


namespace sum_of_squares_2005_squared_l2599_259987

theorem sum_of_squares_2005_squared :
  ∃ (a b c d e f g h : ℤ),
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧ f ≠ 0 ∧ g ≠ 0 ∧ h ≠ 0 ∧
    2005^2 = a^2 + b^2 ∧
    2005^2 = c^2 + d^2 ∧
    2005^2 = e^2 + f^2 ∧
    2005^2 = g^2 + h^2 ∧
    (a, b) ≠ (c, d) ∧ (a, b) ≠ (e, f) ∧ (a, b) ≠ (g, h) ∧
    (c, d) ≠ (e, f) ∧ (c, d) ≠ (g, h) ∧
    (e, f) ≠ (g, h) :=
by sorry

end sum_of_squares_2005_squared_l2599_259987


namespace bus_dispatch_interval_l2599_259924

/-- Represents the speed of a vehicle or person -/
structure Speed : Type :=
  (value : ℝ)

/-- Represents a time interval -/
structure TimeInterval : Type :=
  (minutes : ℝ)

/-- Represents a distance -/
structure Distance : Type :=
  (value : ℝ)

/-- The speed of Xiao Nan -/
def xiao_nan_speed : Speed := ⟨1⟩

/-- The speed of Xiao Yu -/
def xiao_yu_speed : Speed := ⟨3 * xiao_nan_speed.value⟩

/-- The time interval at which Xiao Nan encounters buses -/
def xiao_nan_encounter_interval : TimeInterval := ⟨10⟩

/-- The time interval at which Xiao Yu encounters buses -/
def xiao_yu_encounter_interval : TimeInterval := ⟨5⟩

/-- The speed of the bus -/
def bus_speed : Speed := ⟨5 * xiao_nan_speed.value⟩

/-- The distance between two consecutive buses -/
def bus_distance (s : Speed) (t : TimeInterval) : Distance :=
  ⟨s.value * t.minutes⟩

/-- The theorem stating that the interval between bus dispatches is 8 minutes -/
theorem bus_dispatch_interval :
  ∃ (t : TimeInterval),
    t.minutes = 8 ∧
    bus_distance bus_speed t = bus_distance (Speed.mk (bus_speed.value - xiao_nan_speed.value)) xiao_nan_encounter_interval ∧
    bus_distance bus_speed t = bus_distance (Speed.mk (bus_speed.value + xiao_yu_speed.value)) xiao_yu_encounter_interval :=
sorry

end bus_dispatch_interval_l2599_259924


namespace abs_eq_sum_l2599_259979

theorem abs_eq_sum (x : ℝ) : (|x - 5| = 23) → (∃ y : ℝ, |y - 5| = 23 ∧ x + y = 10) := by
  sorry

end abs_eq_sum_l2599_259979


namespace polynomial_identity_sum_of_squares_l2599_259980

theorem polynomial_identity_sum_of_squares : 
  ∀ (p q r s t u : ℤ), 
  (∀ x : ℝ, 1728 * x^3 + 64 = (p * x^2 + q * x + r) * (s * x^2 + t * x + u)) →
  p^2 + q^2 + r^2 + s^2 + t^2 + u^2 = 23456 := by
  sorry

end polynomial_identity_sum_of_squares_l2599_259980


namespace fraction_simplification_l2599_259921

theorem fraction_simplification : (270 / 18) * (7 / 210) * (9 / 4) = 9 / 8 := by
  sorry

end fraction_simplification_l2599_259921


namespace negation_of_existential_proposition_l2599_259955

theorem negation_of_existential_proposition :
  (¬ ∃ x : ℝ, x^2 + 2*x - 3 ≥ 0) ↔ (∀ x : ℝ, x^2 + 2*x - 3 < 0) :=
by sorry

end negation_of_existential_proposition_l2599_259955


namespace circle_and_triangle_properties_l2599_259929

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop :=
  (k - 1) * x - 2 * y + 5 - 3 * k = 0

-- Define the fixed point P
def point_P : ℝ × ℝ := (3, 1)

-- Define point A
def point_A : ℝ × ℝ := (4, 0)

-- Define the line on which the center of circle C lies
def center_line (x y : ℝ) : Prop :=
  x - 2 * y + 1 = 0

-- Define circle C
def circle_C (x y : ℝ) : Prop :=
  x^2 + y^2 - 14*x - 8*y + 40 = 0

-- Define point Q
def point_Q : ℝ × ℝ := (11, 7)

theorem circle_and_triangle_properties :
  ∀ k : ℝ,
  (∀ x y : ℝ, line_l k x y → (x = point_P.1 ∧ y = point_P.2)) →
  circle_C point_A.1 point_A.2 →
  circle_C point_P.1 point_P.2 →
  (∃ x y : ℝ, center_line x y ∧ (x - point_P.1)^2 + (y - point_P.2)^2 = (x - point_A.1)^2 + (y - point_A.2)^2) →
  (point_Q.1 - point_P.1)^2 + (point_Q.2 - point_P.2)^2 = 4 * ((point_P.1 - 7)^2 + (point_P.2 - 4)^2) →
  ∃ m : ℝ, (m = 5 ∨ m = 65/3) ∧
    ((point_P.1 - 0)^2 + (point_P.2 - m)^2 + (point_Q.1 - 0)^2 + (point_Q.2 - m)^2 =
     (point_Q.1 - point_P.1)^2 + (point_Q.2 - point_P.2)^2) :=
by sorry

end circle_and_triangle_properties_l2599_259929


namespace distance_between_complex_points_l2599_259985

theorem distance_between_complex_points :
  let z₁ : ℂ := 3 - 4*I
  let z₂ : ℂ := -2 - 3*I
  Complex.abs (z₁ - z₂) = Real.sqrt 26 := by
  sorry

end distance_between_complex_points_l2599_259985


namespace arrangement_count_correct_l2599_259937

/-- The number of ways to arrange students from 5 grades visiting 5 museums,
    with exactly 2 grades visiting the Jia Science Museum -/
def arrangement_count : ℕ :=
  Nat.choose 5 2 * (4^3)

/-- Theorem stating that the number of arrangements is correct -/
theorem arrangement_count_correct :
  arrangement_count = Nat.choose 5 2 * (4^3) := by
  sorry

end arrangement_count_correct_l2599_259937


namespace evaluate_expression_l2599_259913

theorem evaluate_expression : -1^2010 + (-1)^2011 + 1^2012 - 1^2013 = -2 := by
  sorry

end evaluate_expression_l2599_259913


namespace power_of_product_of_ten_l2599_259915

theorem power_of_product_of_ten : (2 * 10^3)^3 = 8 * 10^9 := by sorry

end power_of_product_of_ten_l2599_259915


namespace odd_function_property_l2599_259981

-- Define an odd function on an interval
def odd_function_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  (∀ x ∈ Set.Icc a b, f (-x) = -f x) ∧ (a ≤ b)

-- Theorem statement
theorem odd_function_property (f : ℝ → ℝ) (a b : ℝ) 
  (h : odd_function_on_interval f a b) : f (2 * (a + b)) = 0 := by
  sorry

end odd_function_property_l2599_259981


namespace simplify_and_evaluate_l2599_259959

theorem simplify_and_evaluate (a : ℤ) (h : a = -1) : 
  (a + 3)^2 + (3 + a) * (3 - a) = 12 := by
  sorry

end simplify_and_evaluate_l2599_259959


namespace sea_glass_collection_l2599_259939

/-- Sea glass collection problem -/
theorem sea_glass_collection (blanche_green blanche_red rose_red dorothy_total : ℕ)
  (h1 : blanche_green = 12)
  (h2 : blanche_red = 3)
  (h3 : rose_red = 9)
  (h4 : dorothy_total = 57) :
  ∃ (rose_blue : ℕ),
    dorothy_total = 2 * (blanche_red + rose_red) + 3 * rose_blue ∧
    rose_blue = 11 := by
  sorry

end sea_glass_collection_l2599_259939


namespace ball_probabilities_l2599_259907

-- Define the number of balls in each can
def can_A : Fin 3 → ℕ
| 0 => 5  -- red balls
| 1 => 2  -- white balls
| 2 => 3  -- black balls

def can_B : Fin 3 → ℕ
| 0 => 4  -- red balls
| 1 => 3  -- white balls
| 2 => 3  -- black balls

-- Define the probability of drawing a ball of each color from can A
def prob_A (i : Fin 3) : ℚ :=
  (can_A i : ℚ) / (can_A 0 + can_A 1 + can_A 2 : ℚ)

-- Define the probability of drawing a red ball from can B after moving a ball from A
def prob_B_red (i : Fin 3) : ℚ :=
  (can_B 0 + (if i = 0 then 1 else 0) : ℚ) / 
  ((can_B 0 + can_B 1 + can_B 2 + 1) : ℚ)

theorem ball_probabilities :
  (prob_B_red 0 = 5/11) ∧ 
  (prob_A 2 * prob_B_red 2 = 6/55) ∧
  (prob_A 0 * prob_B_red 0 / (prob_A 0 * prob_B_red 0 + prob_A 1 * prob_B_red 1 + prob_A 2 * prob_B_red 2) = 5/9) := by
  sorry


end ball_probabilities_l2599_259907


namespace king_middle_school_teachers_l2599_259940

/-- Represents a school with students and teachers -/
structure School where
  num_students : ℕ
  classes_per_student : ℕ
  classes_per_teacher : ℕ
  students_per_class : ℕ

/-- Calculates the number of teachers in a school -/
def num_teachers (s : School) : ℕ :=
  let total_classes := s.num_students * s.classes_per_student
  let unique_classes := (total_classes + s.students_per_class - 1) / s.students_per_class
  (unique_classes + s.classes_per_teacher - 1) / s.classes_per_teacher

/-- King Middle School -/
def king_middle_school : School :=
  { num_students := 1200
  , classes_per_student := 6
  , classes_per_teacher := 5
  , students_per_class := 35 }

theorem king_middle_school_teachers :
  num_teachers king_middle_school = 42 := by
  sorry

end king_middle_school_teachers_l2599_259940


namespace sweater_wool_correct_l2599_259948

/-- The number of balls of wool used for a sweater -/
def sweater_wool : ℕ := 4

/-- The number of scarves Aaron makes -/
def aaron_scarves : ℕ := 10

/-- The number of sweaters Aaron makes -/
def aaron_sweaters : ℕ := 5

/-- The number of sweaters Enid makes -/
def enid_sweaters : ℕ := 8

/-- The number of balls of wool used for a scarf -/
def scarf_wool : ℕ := 3

/-- The total number of balls of wool used -/
def total_wool : ℕ := 82

theorem sweater_wool_correct : 
  aaron_scarves * scarf_wool + (aaron_sweaters + enid_sweaters) * sweater_wool = total_wool := by
  sorry

end sweater_wool_correct_l2599_259948


namespace percentage_increase_lines_l2599_259977

theorem percentage_increase_lines (initial : ℕ) (final : ℕ) (increase_percent : ℚ) : 
  initial = 500 →
  final = 800 →
  increase_percent = 60 →
  (final - initial : ℚ) / initial * 100 = increase_percent :=
by sorry

end percentage_increase_lines_l2599_259977


namespace tangent_circle_equation_l2599_259961

/-- A circle with center on the y-axis, radius 1, and tangent to y = 2 -/
structure TangentCircle where
  center : ℝ × ℝ
  radius : ℝ
  center_on_y_axis : center.1 = 0
  radius_is_one : radius = 1
  tangent_to_y_2 : ∃ (x : ℝ), (center.1 - x)^2 + (center.2 - 2)^2 = radius^2

/-- The equation of a TangentCircle is x^2 + (y-3)^2 = 1 or x^2 + (y-1)^2 = 1 -/
theorem tangent_circle_equation (c : TangentCircle) :
  ∃ (y₀ : ℝ), y₀ = 1 ∨ y₀ = 3 ∧ ∀ (x y : ℝ), (x, y) ∈ {p : ℝ × ℝ | (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2} ↔ x^2 + (y - y₀)^2 = 1 :=
sorry

end tangent_circle_equation_l2599_259961


namespace constant_product_of_reciprocal_inputs_l2599_259994

theorem constant_product_of_reciprocal_inputs (a b : ℝ) (h : a * b ≠ 2) :
  let f : ℝ → ℝ := λ x => (b * x + 1) / (2 * x + a)
  ∃ k : ℝ, ∀ x : ℝ, x ≠ 0 → f x * f (1 / x) = k → k = 1 / 4 := by
  sorry

end constant_product_of_reciprocal_inputs_l2599_259994


namespace four_digit_sum_divisible_by_nine_l2599_259998

theorem four_digit_sum_divisible_by_nine 
  (a b c d e f g h i j : Nat) 
  (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧ a ≠ j ∧
              b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧ b ≠ j ∧
              c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧ c ≠ j ∧
              d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧ d ≠ j ∧
              e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧ e ≠ j ∧
              f ≠ g ∧ f ≠ h ∧ f ≠ i ∧ f ≠ j ∧
              g ≠ h ∧ g ≠ i ∧ g ≠ j ∧
              h ≠ i ∧ h ≠ j ∧
              i ≠ j)
  (digits : a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧ f < 10 ∧ g < 10 ∧ h < 10 ∧ i < 10 ∧ j < 10)
  (sum_equality : 100 * a + 10 * b + c + 100 * d + 10 * e + f = 1000 * g + 100 * h + 10 * i + j) :
  (1000 * g + 100 * h + 10 * i + j) % 9 = 0 := by
  sorry


end four_digit_sum_divisible_by_nine_l2599_259998


namespace football_team_progress_l2599_259925

/-- 
Given a football team's yard changes, calculate their net progress.
-/
theorem football_team_progress 
  (loss : ℤ) 
  (gain : ℤ) 
  (h1 : loss = -5)
  (h2 : gain = 9) : 
  loss + gain = 4 := by
sorry

end football_team_progress_l2599_259925


namespace circle_radius_through_ROV_l2599_259946

-- Define the pentagon LOVER
structure Pentagon :=
  (L O V E R : ℝ × ℝ)

-- Define properties of the pentagon
def is_convex (p : Pentagon) : Prop := sorry

def is_rectangle (A B C D : ℝ × ℝ) : Prop := sorry

def distance (A B : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem circle_radius_through_ROV (LOVER : Pentagon) :
  is_convex LOVER →
  is_rectangle LOVER.L LOVER.O LOVER.V LOVER.E →
  distance LOVER.O LOVER.V = 20 →
  distance LOVER.L LOVER.O = 23 →
  distance LOVER.V LOVER.E = 23 →
  distance LOVER.R LOVER.E = 23 →
  distance LOVER.R LOVER.L = 23 →
  ∃ (center : ℝ × ℝ), 
    distance center LOVER.R = 23 ∧
    distance center LOVER.O = 23 ∧
    distance center LOVER.V = 23 :=
by
  sorry


end circle_radius_through_ROV_l2599_259946


namespace adi_change_l2599_259916

/-- Calculate the change Adi will receive after purchasing items and paying with a $20 bill. -/
theorem adi_change (pencil_cost notebook_cost colored_pencils_cost paid : ℚ) : 
  pencil_cost = 35/100 →
  notebook_cost = 3/2 →
  colored_pencils_cost = 11/4 →
  paid = 20 →
  paid - (pencil_cost + notebook_cost + colored_pencils_cost) = 77/5 := by
  sorry

#eval (20 : ℚ) - (35/100 + 3/2 + 11/4)

end adi_change_l2599_259916


namespace oil_depth_theorem_l2599_259945

/-- Represents a horizontal cylindrical oil tank -/
structure OilTank where
  length : ℝ
  diameter : ℝ

/-- Calculates the depth of oil in the tank given the surface area -/
def oilDepth (tank : OilTank) (surfaceArea : ℝ) : Set ℝ :=
  { h : ℝ | ∃ (c : ℝ), 
    c = surfaceArea / tank.length ∧
    c = 2 * Real.sqrt (tank.diameter * h - h^2) ∧
    0 < h ∧ h < tank.diameter }

/-- The main theorem about oil depth in a cylindrical tank -/
theorem oil_depth_theorem (tank : OilTank) (surfaceArea : ℝ) :
  tank.length = 12 →
  tank.diameter = 8 →
  surfaceArea = 60 →
  oilDepth tank surfaceArea = {4 - Real.sqrt 39 / 2, 4 + Real.sqrt 39 / 2} := by
  sorry

end oil_depth_theorem_l2599_259945


namespace reciprocal_comparison_l2599_259901

theorem reciprocal_comparison :
  let numbers : List ℚ := [1/3, 1/2, 1, 2, 3]
  ∀ x ∈ numbers, x < (1 / x) ↔ (x = 1/3 ∨ x = 1/2) := by
  sorry

end reciprocal_comparison_l2599_259901


namespace faye_crayons_l2599_259928

/-- The number of rows of crayons and pencils -/
def num_rows : ℕ := 16

/-- The number of crayons in each row -/
def crayons_per_row : ℕ := 6

/-- The total number of crayons -/
def total_crayons : ℕ := num_rows * crayons_per_row

theorem faye_crayons : total_crayons = 96 := by
  sorry

end faye_crayons_l2599_259928


namespace seven_balls_four_boxes_l2599_259984

/-- The number of ways to distribute n indistinguishable balls into k distinguishable boxes,
    with each box containing at least one ball. -/
def distribute_balls (n k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 220 ways to distribute 7 indistinguishable balls into 4 distinguishable boxes,
    with each box containing at least one ball. -/
theorem seven_balls_four_boxes : distribute_balls 7 4 = 220 := by
  sorry

end seven_balls_four_boxes_l2599_259984


namespace range_of_a_l2599_259975

theorem range_of_a (a : ℝ) : 
  Real.sqrt (a^3 + 2*a^2) = -a * Real.sqrt (a + 2) → 
  -2 ≤ a ∧ a ≤ 0 := by sorry

end range_of_a_l2599_259975


namespace train_speed_l2599_259990

/-- The speed of a train given specific conditions -/
theorem train_speed (t_pole : ℝ) (t_stationary : ℝ) (l_stationary : ℝ) :
  t_pole = 10 →
  t_stationary = 30 →
  l_stationary = 600 →
  ∃ v : ℝ, v * t_pole = v * t_stationary - l_stationary ∧ v * 3.6 = 108 := by
  sorry

end train_speed_l2599_259990


namespace not_all_angles_greater_than_60_l2599_259951

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angle_sum : a + b + c = 180

-- Theorem statement
theorem not_all_angles_greater_than_60 (t : Triangle) : 
  ¬(t.a > 60 ∧ t.b > 60 ∧ t.c > 60) :=
by sorry

end not_all_angles_greater_than_60_l2599_259951
