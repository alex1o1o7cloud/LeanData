import Mathlib

namespace point_in_fourth_quadrant_l2858_285847

/-- A point is in the fourth quadrant if its x-coordinate is positive and its y-coordinate is negative -/
def is_in_fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

/-- The point (3, -2) is in the fourth quadrant of the Cartesian coordinate system -/
theorem point_in_fourth_quadrant : is_in_fourth_quadrant 3 (-2) := by
  sorry

end point_in_fourth_quadrant_l2858_285847


namespace angle_ABC_measure_l2858_285893

/-- A regular octagon with a square constructed outward on one side -/
structure RegularOctagonWithSquare where
  /-- The vertices of the octagon -/
  vertices : Fin 8 → ℝ × ℝ
  /-- The square constructed outward on one side -/
  square : Fin 4 → ℝ × ℝ
  /-- The octagon is regular -/
  regular : ∀ i j : Fin 8, dist (vertices i) (vertices ((i + 1) % 8)) = dist (vertices j) (vertices ((j + 1) % 8))
  /-- The square is connected to the octagon -/
  square_connected : ∃ i : Fin 8, square 0 = vertices i ∧ square 1 = vertices ((i + 1) % 8)

/-- Point B where two diagonals intersect inside the octagon -/
def intersection_point (o : RegularOctagonWithSquare) : ℝ × ℝ := sorry

/-- Angle ABC in the octagon -/
def angle_ABC (o : RegularOctagonWithSquare) : ℝ := sorry

/-- Theorem: The measure of angle ABC is 22.5° -/
theorem angle_ABC_measure (o : RegularOctagonWithSquare) : angle_ABC o = 22.5 := by sorry

end angle_ABC_measure_l2858_285893


namespace smallest_value_theorem_l2858_285836

theorem smallest_value_theorem (n : ℕ+) : 
  (n : ℝ) / 2 + 18 / (n : ℝ) ≥ 6 ∧ 
  ((6 : ℕ+) : ℝ) / 2 + 18 / ((6 : ℕ+) : ℝ) = 6 := by
  sorry

end smallest_value_theorem_l2858_285836


namespace marks_leftover_amount_marks_leftover_is_980_l2858_285816

/-- Calculates the amount Mark has leftover each week after his raise and new expenses -/
theorem marks_leftover_amount (old_wage : ℝ) (raise_percentage : ℝ) 
  (hours_per_day : ℝ) (days_per_week : ℝ) (old_bills : ℝ) (trainer_cost : ℝ) : ℝ :=
  let new_wage := old_wage * (1 + raise_percentage / 100)
  let weekly_hours := hours_per_day * days_per_week
  let weekly_earnings := new_wage * weekly_hours
  let weekly_expenses := old_bills + trainer_cost
  weekly_earnings - weekly_expenses

/-- Proves that Mark has $980 leftover each week after his raise and new expenses -/
theorem marks_leftover_is_980 : 
  marks_leftover_amount 40 5 8 5 600 100 = 980 := by
  sorry

end marks_leftover_amount_marks_leftover_is_980_l2858_285816


namespace problem_statement_l2858_285827

theorem problem_statement (a b : ℝ) (h : a + b = 3) : 2*a^2 + 4*a*b + 2*b^2 - 4 = 14 := by
  sorry

end problem_statement_l2858_285827


namespace value_of_expression_l2858_285895

theorem value_of_expression (x y : ℝ) (hx : x = 2) (hy : y = 1) : 2 * x - 3 * y = 1 := by
  sorry

end value_of_expression_l2858_285895


namespace distinct_lines_count_l2858_285891

/-- Represents a 4-by-4 grid of lattice points -/
def Grid := Fin 4 × Fin 4

/-- A line in the grid is defined by two distinct points it passes through -/
def Line := { pair : Grid × Grid // pair.1 ≠ pair.2 }

/-- Counts the number of distinct lines passing through at least two points in the grid -/
def countDistinctLines : Nat :=
  sorry

/-- The main theorem stating that the number of distinct lines is 84 -/
theorem distinct_lines_count : countDistinctLines = 84 :=
  sorry

end distinct_lines_count_l2858_285891


namespace triangle_segment_proof_l2858_285882

theorem triangle_segment_proof (a b c h x : ℝ) : 
  a = 40 ∧ b = 75 ∧ c = 100 ∧ 
  a^2 = x^2 + h^2 ∧
  b^2 = (c - x)^2 + h^2 →
  c - x = 70.125 := by
sorry

end triangle_segment_proof_l2858_285882


namespace household_electricity_most_suitable_l2858_285858

/-- Represents an investigation option --/
inductive InvestigationOption
  | ProductPopularity
  | TVViewershipRatings
  | AmmunitionExplosivePower
  | HouseholdElectricityConsumption

/-- Defines what makes an investigation suitable for a census method --/
def suitableForCensus (option : InvestigationOption) : Prop :=
  match option with
  | InvestigationOption.HouseholdElectricityConsumption => True
  | _ => False

/-- Theorem stating that investigating household electricity consumption is most suitable for census --/
theorem household_electricity_most_suitable :
    ∀ option : InvestigationOption,
      suitableForCensus option →
      option = InvestigationOption.HouseholdElectricityConsumption :=
by
  sorry

/-- Definition of a census method --/
def censusMethod (population : Type) (examine : population → Prop) : Prop :=
  ∀ subject : population, examine subject

#check household_electricity_most_suitable

end household_electricity_most_suitable_l2858_285858


namespace right_triangle_area_l2858_285883

/-- The area of a right triangle with legs of 30 inches and 45 inches is 675 square inches. -/
theorem right_triangle_area (a b : ℝ) (h1 : a = 30) (h2 : b = 45) : 
  (1/2) * a * b = 675 := by
  sorry

end right_triangle_area_l2858_285883


namespace last_two_average_l2858_285878

theorem last_two_average (list : List ℝ) : 
  list.length = 7 →
  (list.sum / 7 : ℝ) = 63 →
  ((list.take 3).sum / 3 : ℝ) = 58 →
  ((list.drop 3).take 2).sum / 2 = 70 →
  ((list.drop 5).sum / 2 : ℝ) = 63.5 := by
  sorry

end last_two_average_l2858_285878


namespace parallelogram_secant_minimum_sum_l2858_285861

/-- Given a parallelogram ABCD with side lengths a and b, and a secant through
    vertex B intersecting extensions of sides DA and DC at points P and Q
    respectively, the sum of segments PA and CQ is minimized when PA = CQ = √(ab). -/
theorem parallelogram_secant_minimum_sum (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) :
  let f : ℝ → ℝ := λ x => x + (a * b) / x
  ∃ (x : ℝ), x > 0 ∧ f x = Real.sqrt (a * b) + Real.sqrt (a * b) ∧
    ∀ (y : ℝ), y > 0 → f y ≥ f x :=
  sorry

end parallelogram_secant_minimum_sum_l2858_285861


namespace lemonade_pitchers_sum_l2858_285868

theorem lemonade_pitchers_sum : 
  let first_intermission : ℚ := 0.25
  let second_intermission : ℚ := 0.4166666666666667
  let third_intermission : ℚ := 0.25
  first_intermission + second_intermission + third_intermission = 0.9166666666666667 := by
sorry

end lemonade_pitchers_sum_l2858_285868


namespace complement_M_inter_N_eq_one_two_l2858_285826

def U : Finset ℕ := {1, 2, 3, 4, 5}
def M : Finset ℕ := {3, 4, 5}
def N : Finset ℕ := {1, 2, 5}

theorem complement_M_inter_N_eq_one_two :
  (U \ M) ∩ N = {1, 2} := by sorry

end complement_M_inter_N_eq_one_two_l2858_285826


namespace sum_of_fractions_equals_ten_l2858_285807

theorem sum_of_fractions_equals_ten : 
  (1 / 10 : ℚ) + (2 / 10 : ℚ) + (3 / 10 : ℚ) + (4 / 10 : ℚ) + (5 / 10 : ℚ) + 
  (6 / 10 : ℚ) + (7 / 10 : ℚ) + (8 / 10 : ℚ) + (9 / 10 : ℚ) + (55 / 10 : ℚ) = 10 := by
  sorry

end sum_of_fractions_equals_ten_l2858_285807


namespace multiple_of_number_l2858_285801

theorem multiple_of_number : ∃ m : ℕ, m < 4 ∧ 7 * 5 - 15 > m * 5 := by
  sorry

end multiple_of_number_l2858_285801


namespace smallest_marble_count_l2858_285854

def is_valid_marble_count (n : ℕ) : Prop :=
  n > 1 ∧ n % 5 = 2 ∧ n % 7 = 2 ∧ n % 9 = 2

theorem smallest_marble_count :
  ∃ (n : ℕ), is_valid_marble_count n ∧
  ∀ (m : ℕ), is_valid_marble_count m → n ≤ m :=
by
  use 317
  sorry

end smallest_marble_count_l2858_285854


namespace sample_product_l2858_285842

/-- Given a sample of five numbers (7, 8, 9, x, y) with an average of 8 
    and a standard deviation of √2, prove that xy = 60 -/
theorem sample_product (x y : ℝ) : 
  (7 + 8 + 9 + x + y) / 5 = 8 → 
  Real.sqrt (((7 - 8)^2 + (8 - 8)^2 + (9 - 8)^2 + (x - 8)^2 + (y - 8)^2) / 5) = Real.sqrt 2 →
  x * y = 60 := by
  sorry

end sample_product_l2858_285842


namespace people_who_left_gym_l2858_285843

theorem people_who_left_gym (initial_people : ℕ) (people_came_in : ℕ) (current_people : ℕ)
  (h1 : initial_people = 16)
  (h2 : people_came_in = 5)
  (h3 : current_people = 19) :
  initial_people + people_came_in - current_people = 2 := by
sorry

end people_who_left_gym_l2858_285843


namespace smallest_x_satisfying_equation_l2858_285885

theorem smallest_x_satisfying_equation : 
  ∃ (x : ℝ), x > 0 ∧ x = 131/11 ∧ 
  (∀ y : ℝ, y > 0 → ⌊y^2⌋ - y * ⌊y⌋ = 10 → x ≤ y) ∧
  (⌊x^2⌋ - x * ⌊x⌋ = 10) := by
  sorry

end smallest_x_satisfying_equation_l2858_285885


namespace product_of_D_coordinates_l2858_285850

-- Define the points
def C : ℝ × ℝ := (6, -1)
def N : ℝ × ℝ := (4, 3)

-- Define D as a variable point
variable (D : ℝ × ℝ)

-- Define the midpoint condition
def is_midpoint (M A B : ℝ × ℝ) : Prop :=
  M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2

-- State the theorem
theorem product_of_D_coordinates :
  is_midpoint N C D → D.1 * D.2 = 14 := by sorry

end product_of_D_coordinates_l2858_285850


namespace min_value_trig_expression_l2858_285898

theorem min_value_trig_expression (α β : ℝ) :
  (3 * Real.cos α + 4 * Real.sin β - 7)^2 + (3 * Real.sin α + 4 * Real.cos β - 12)^2 ≥ 36 ∧
  ∃ α₀ β₀ : ℝ, (3 * Real.cos α₀ + 4 * Real.sin β₀ - 7)^2 + (3 * Real.sin α₀ + 4 * Real.cos β₀ - 12)^2 = 36 :=
by sorry

end min_value_trig_expression_l2858_285898


namespace fraction_ordering_l2858_285887

theorem fraction_ordering : (8 : ℚ) / 24 < 6 / 17 ∧ 6 / 17 < 10 / 27 := by
  sorry

end fraction_ordering_l2858_285887


namespace largest_number_with_given_hcf_and_lcm_factors_l2858_285864

theorem largest_number_with_given_hcf_and_lcm_factors (a b : ℕ+) 
  (h_hcf : Nat.gcd a b = 52)
  (h_lcm : Nat.lcm a b = 52 * 11 * 12) :
  max a b = 624 := by
  sorry

end largest_number_with_given_hcf_and_lcm_factors_l2858_285864


namespace unique_pythagorean_triple_l2858_285810

/-- A function to check if a triple of natural numbers is a Pythagorean triple -/
def isPythagoreanTriple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

/-- The theorem stating that (5, 12, 13) is the only Pythagorean triple among the given options -/
theorem unique_pythagorean_triple :
  ¬ isPythagoreanTriple 3 4 5 ∧
  ¬ isPythagoreanTriple 1 1 2 ∧
  isPythagoreanTriple 5 12 13 ∧
  ¬ isPythagoreanTriple 1 3 2 :=
by sorry

end unique_pythagorean_triple_l2858_285810


namespace larger_number_ratio_l2858_285822

theorem larger_number_ratio (m v : ℝ) (h1 : m < v) (h2 : v - m/4 = 5*(3*m/4)) : v = 4*m := by
  sorry

end larger_number_ratio_l2858_285822


namespace wrong_mark_calculation_l2858_285873

theorem wrong_mark_calculation (correct_mark : ℕ) (num_pupils : ℕ) :
  correct_mark = 45 →
  num_pupils = 44 →
  ∃ (wrong_mark : ℕ),
    (wrong_mark - correct_mark : ℚ) / num_pupils = 1/2 ∧
    wrong_mark = 67 :=
by sorry

end wrong_mark_calculation_l2858_285873


namespace exactly_one_fail_probability_l2858_285880

/-- The probability that exactly one item fails the inspection when one item is taken from each of two types of products with pass rates of 0.90 and 0.95 respectively is 0.14. -/
theorem exactly_one_fail_probability (pass_rate1 pass_rate2 : ℝ) 
  (h1 : pass_rate1 = 0.90) (h2 : pass_rate2 = 0.95) : 
  pass_rate1 * (1 - pass_rate2) + (1 - pass_rate1) * pass_rate2 = 0.14 := by
  sorry

end exactly_one_fail_probability_l2858_285880


namespace prob_all_fives_four_dice_l2858_285872

-- Define a standard six-sided die
def standard_die := Finset.range 6

-- Define the probability of getting a specific number on a standard die
def prob_specific_number (die : Finset Nat) : ℚ :=
  1 / die.card

-- Define the number of dice
def num_dice : Nat := 4

-- Define the desired outcome (all fives)
def all_fives (n : Nat) : Bool := n = 5

-- Theorem: The probability of getting all fives on four standard six-sided dice is 1/1296
theorem prob_all_fives_four_dice : 
  (prob_specific_number standard_die) ^ num_dice = 1 / 1296 :=
sorry

end prob_all_fives_four_dice_l2858_285872


namespace simple_interest_calculation_l2858_285859

/-- Calculate the total amount owed after one year with simple interest -/
theorem simple_interest_calculation 
  (principal : ℝ) 
  (rate : ℝ) 
  (time : ℝ) :
  principal = 35 →
  rate = 0.05 →
  time = 1 →
  principal + principal * rate * time = 36.75 := by
  sorry


end simple_interest_calculation_l2858_285859


namespace jeff_bought_two_stars_l2858_285845

/-- The number of ninja throwing stars Jeff bought from Chad -/
def stars_bought_by_jeff (eric_stars chad_stars jeff_stars total_stars : ℕ) : ℕ :=
  chad_stars - (total_stars - eric_stars - jeff_stars)

theorem jeff_bought_two_stars :
  let eric_stars : ℕ := 4
  let chad_stars : ℕ := 2 * eric_stars
  let jeff_stars : ℕ := 6
  let total_stars : ℕ := 16
  stars_bought_by_jeff eric_stars chad_stars jeff_stars total_stars = 2 := by
sorry

end jeff_bought_two_stars_l2858_285845


namespace quadratic_one_solution_l2858_285830

theorem quadratic_one_solution (q : ℝ) : 
  (q ≠ 0 ∧ ∃! x : ℝ, q * x^2 - 16 * x + 9 = 0) ↔ q = 64/9 := by sorry

end quadratic_one_solution_l2858_285830


namespace root_sum_square_l2858_285857

theorem root_sum_square (α β : ℝ) : 
  α^2 + 2*α - 2021 = 0 → 
  β^2 + 2*β - 2021 = 0 → 
  α^2 + 3*α + β = 2019 :=
by
  sorry

end root_sum_square_l2858_285857


namespace common_chord_of_intersecting_circles_l2858_285838

-- Define the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 2*y - 8 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 10*y - 24 = 0

-- Define the common chord
def common_chord (x y : ℝ) : Prop := x - 2*y + 4 = 0

-- Theorem statement
theorem common_chord_of_intersecting_circles :
  ∀ x y : ℝ, (circle1 x y ∧ circle2 x y) → common_chord x y :=
by sorry

end common_chord_of_intersecting_circles_l2858_285838


namespace max_value_d_l2858_285800

theorem max_value_d (a b c d : ℝ) 
  (sum_eq : a + b + c + d = 10)
  (sum_prod_eq : a*b + a*c + a*d + b*c + b*d + c*d = 17) :
  d ≤ (5 + Real.sqrt 123) / 2 ∧ 
  ∃ (a' b' c' : ℝ), a' + b' + c' + (5 + Real.sqrt 123) / 2 = 10 ∧ 
    a'*b' + a'*c' + a'*((5 + Real.sqrt 123) / 2) + b'*c' + 
    b'*((5 + Real.sqrt 123) / 2) + c'*((5 + Real.sqrt 123) / 2) = 17 :=
by sorry

end max_value_d_l2858_285800


namespace total_flowers_l2858_285899

theorem total_flowers (roses : ℕ) (lilies : ℕ) (tulips : ℕ) : 
  roses = 34 →
  lilies = roses + 13 →
  tulips = lilies - 23 →
  roses + lilies + tulips = 105 := by
sorry

end total_flowers_l2858_285899


namespace nitin_rank_l2858_285856

theorem nitin_rank (total_students : ℕ) (rank_from_last : ℕ) (rank_from_first : ℕ) : 
  total_students = 58 → rank_from_last = 34 → rank_from_first = 25 :=
by sorry

end nitin_rank_l2858_285856


namespace min_value_xyz_l2858_285879

theorem min_value_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 18) :
  x^2 + 4*x*y + y^2 + 3*z^2 ≥ 63 ∧ ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x * y * z = 18 ∧ x^2 + 4*x*y + y^2 + 3*z^2 = 63 :=
by sorry

end min_value_xyz_l2858_285879


namespace hyperbola_asymptotes_l2858_285892

/-- Given a hyperbola C with equation (x^2 / a^2) - (y^2 / b^2) = 1, where a > 0, b > 0, 
    and eccentricity √10, its asymptotes are y = ±3x -/
theorem hyperbola_asymptotes (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let C := {(x, y) : ℝ × ℝ | x^2 / a^2 - y^2 / b^2 = 1}
  let e := Real.sqrt 10  -- eccentricity
  (∀ p ∈ C, (p.1^2 / a^2 - p.2^2 / b^2 = 1)) →
  (e^2 = (a^2 + b^2) / a^2) →
  (∃ k : ℝ, k = b / a ∧ 
    (∀ (x y : ℝ), (x, y) ∈ C → (y = k * x ∨ y = -k * x)) ∧
    k = 3) :=
by sorry

end hyperbola_asymptotes_l2858_285892


namespace circle_value_l2858_285808

theorem circle_value (circle triangle : ℕ) 
  (eq1 : circle + circle + circle + circle = triangle + triangle + circle)
  (eq2 : triangle = 63) : 
  circle = 42 := by
  sorry

end circle_value_l2858_285808


namespace cistern_fill_time_l2858_285840

theorem cistern_fill_time (fill_time : ℝ) (empty_time : ℝ) (h1 : fill_time = 10) (h2 : empty_time = 12) :
  let net_fill_rate := 1 / fill_time - 1 / empty_time
  1 / net_fill_rate = 60 := by sorry

end cistern_fill_time_l2858_285840


namespace exam_average_marks_l2858_285824

theorem exam_average_marks (total_boys : ℕ) (total_avg : ℚ) (passed_avg : ℚ) (passed_boys : ℕ) :
  total_boys = 120 →
  total_avg = 35 →
  passed_avg = 39 →
  passed_boys = 100 →
  let failed_boys := total_boys - passed_boys
  let total_marks := total_avg * total_boys
  let passed_marks := passed_avg * passed_boys
  let failed_marks := total_marks - passed_marks
  (failed_marks / failed_boys : ℚ) = 15 := by
  sorry

end exam_average_marks_l2858_285824


namespace sum_interior_angles_theorem_l2858_285849

/-- The sum of interior angles of an n-gon -/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- Theorem: The sum of the interior angles of any n-gon is (n-2) * 180° -/
theorem sum_interior_angles_theorem (n : ℕ) (h : n ≥ 3) :
  sum_interior_angles n = (n - 2) * 180 := by
  sorry

end sum_interior_angles_theorem_l2858_285849


namespace inequality_implication_l2858_285888

theorem inequality_implication (a b : ℝ) (h : a > b) : b - a < 0 := by
  sorry

end inequality_implication_l2858_285888


namespace probability_differ_by_three_l2858_285805

/-- A type representing the possible outcomes of rolling a standard 6-sided die -/
inductive DieRoll : Type
  | one : DieRoll
  | two : DieRoll
  | three : DieRoll
  | four : DieRoll
  | five : DieRoll
  | six : DieRoll

/-- The total number of possible outcomes when rolling a die twice -/
def totalOutcomes : ℕ := 36

/-- A function that returns true if two die rolls differ by 3 -/
def differByThree (roll1 roll2 : DieRoll) : Prop :=
  match roll1, roll2 with
  | DieRoll.one, DieRoll.four => True
  | DieRoll.two, DieRoll.five => True
  | DieRoll.three, DieRoll.six => True
  | DieRoll.four, DieRoll.one => True
  | DieRoll.five, DieRoll.two => True
  | DieRoll.six, DieRoll.three => True
  | _, _ => False

/-- The number of favorable outcomes (pairs of rolls that differ by 3) -/
def favorableOutcomes : ℕ := 6

/-- The main theorem: the probability of rolling two numbers that differ by 3 is 1/6 -/
theorem probability_differ_by_three :
  (favorableOutcomes : ℚ) / totalOutcomes = 1 / 6 := by
  sorry


end probability_differ_by_three_l2858_285805


namespace ellipse_k_range_l2858_285828

/-- An ellipse with equation x^2 + ky^2 = 2 and foci on the y-axis -/
structure EllipseOnYAxis where
  k : ℝ
  is_ellipse : k > 0
  foci_on_y_axis : k < 1

/-- The range of k for an ellipse with equation x^2 + ky^2 = 2 and foci on the y-axis is (0, 1) -/
theorem ellipse_k_range (e : EllipseOnYAxis) : 0 < e.k ∧ e.k < 1 := by
  sorry

end ellipse_k_range_l2858_285828


namespace cars_in_ten_hours_l2858_285886

-- Define the time interval between cars (in minutes)
def time_interval : ℕ := 20

-- Define the total duration (in hours)
def total_duration : ℕ := 10

-- Define the function to calculate the number of cars
def num_cars (interval : ℕ) (duration : ℕ) : ℕ :=
  (duration * 60) / interval

-- Theorem to prove
theorem cars_in_ten_hours :
  num_cars time_interval total_duration = 30 := by
  sorry

end cars_in_ten_hours_l2858_285886


namespace complex_function_equality_l2858_285829

-- Define the complex function f
def f : ℂ → ℂ := fun z ↦ 2 * (1 - z) - Complex.I

-- State the theorem
theorem complex_function_equality :
  (1 + Complex.I) * f (1 - Complex.I) = -1 + Complex.I :=
by
  sorry

end complex_function_equality_l2858_285829


namespace geometric_series_sum_l2858_285897

theorem geometric_series_sum : ∀ (a r : ℚ), 
  a = 1 → r = 1/3 → abs r < 1 → 
  (∑' n, a * r^n) = 3/2 := by sorry

end geometric_series_sum_l2858_285897


namespace cube_sum_equals_110_l2858_285860

theorem cube_sum_equals_110 (x : ℝ) (h : x^3 + 1/x^3 = 110) : x + 1/x = 5 := by
  sorry

end cube_sum_equals_110_l2858_285860


namespace garden_walkway_area_l2858_285853

/-- Calculates the area of walkways in a garden with vegetable beds -/
theorem garden_walkway_area (rows : Nat) (cols : Nat) (bed_width : Nat) (bed_height : Nat) (walkway_width : Nat) : 
  rows = 4 → cols = 3 → bed_width = 8 → bed_height = 3 → walkway_width = 2 →
  (rows * cols * bed_width * bed_height + 
   (rows + 1) * walkway_width * (cols * bed_width + (cols + 1) * walkway_width) + 
   rows * (cols + 1) * walkway_width * bed_height) - 
  (rows * cols * bed_width * bed_height) = 416 := by
sorry

end garden_walkway_area_l2858_285853


namespace otimes_result_l2858_285851

/-- Definition of the ⊗ operation -/
def otimes (a b : ℚ) (x y : ℚ) : ℚ := a^2 * x + b * y - 3

/-- Theorem stating that 2 ⊗ (-6) = 7 given 1 ⊗ (-3) = 2 -/
theorem otimes_result (a b : ℚ) (h : otimes a b 1 (-3) = 2) : otimes a b 2 (-6) = 7 := by
  sorry

end otimes_result_l2858_285851


namespace triangle_side_length_l2858_285802

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) :
  (0 < a ∧ 0 < b ∧ 0 < c) →  -- Positive side lengths
  (0 < A ∧ A < π) →  -- Valid angle measures
  (0 < B ∧ B < π) →
  (0 < C ∧ C < π) →
  (A + B + C = π) →  -- Sum of angles in a triangle
  (c * Real.cos B = 12) →  -- Given condition
  (b * Real.sin C = 5) →  -- Given condition
  (a / Real.sin A = b / Real.sin B) →  -- Sine rule
  (b / Real.sin B = c / Real.sin C) →  -- Sine rule
  c = 13 := by
sorry


end triangle_side_length_l2858_285802


namespace lunch_change_calculation_l2858_285863

/-- Calculates the change received when buying lunch items --/
theorem lunch_change_calculation (hamburger_cost onion_rings_cost smoothie_cost amount_paid : ℕ) :
  hamburger_cost = 4 →
  onion_rings_cost = 2 →
  smoothie_cost = 3 →
  amount_paid = 20 →
  amount_paid - (hamburger_cost + onion_rings_cost + smoothie_cost) = 11 := by
  sorry

end lunch_change_calculation_l2858_285863


namespace remaining_amount_after_expenses_l2858_285841

def bonus : ℚ := 1496
def kitchen_fraction : ℚ := 1 / 22
def holiday_fraction : ℚ := 1 / 4
def gift_fraction : ℚ := 1 / 8

theorem remaining_amount_after_expenses : 
  bonus - (bonus * kitchen_fraction + bonus * holiday_fraction + bonus * gift_fraction) = 867 := by
  sorry

end remaining_amount_after_expenses_l2858_285841


namespace plot_length_l2858_285817

/-- The length of a rectangular plot given specific conditions -/
theorem plot_length (breadth : ℝ) (length : ℝ) (cost_per_meter : ℝ) (total_cost : ℝ) :
  length = breadth + 32 →
  cost_per_meter = 26.50 →
  total_cost = 5300 →
  2 * (length + breadth) * cost_per_meter = total_cost →
  length = 66 := by
sorry

end plot_length_l2858_285817


namespace power_calculation_l2858_285820

theorem power_calculation : 3000 * (3000^3000)^2 = 3000^6001 := by
  sorry

end power_calculation_l2858_285820


namespace sum_of_roots_l2858_285839

theorem sum_of_roots (h b x₁ x₂ : ℝ) (hx : x₁ ≠ x₂) 
  (eq₁ : 4 * x₁^2 - h * x₁ = b) (eq₂ : 4 * x₂^2 - h * x₂ = b) : 
  x₁ + x₂ = h / 4 := by
sorry

end sum_of_roots_l2858_285839


namespace right_triangle_circumcenter_angles_l2858_285809

theorem right_triangle_circumcenter_angles (α : Real) (h1 : α = 25 * π / 180) :
  let β := π / 2 - α
  let θ₁ := 2 * α
  let θ₂ := 2 * β
  θ₁ = 50 * π / 180 ∧ θ₂ = 130 * π / 180 := by
  sorry

end right_triangle_circumcenter_angles_l2858_285809


namespace tan_half_sum_angles_l2858_285871

theorem tan_half_sum_angles (x y : ℝ) 
  (h1 : Real.cos x + Real.cos y = 3/5)
  (h2 : Real.sin x + Real.sin y = 8/17) : 
  Real.tan ((x + y)/2) = 40/51 := by
  sorry

end tan_half_sum_angles_l2858_285871


namespace hermione_badges_l2858_285825

theorem hermione_badges (total luna celestia : ℕ) (h1 : total = 83) (h2 : luna = 17) (h3 : celestia = 52) :
  total - luna - celestia = 14 := by
  sorry

end hermione_badges_l2858_285825


namespace smallest_n_equality_l2858_285862

def C (n : ℕ) : ℚ := 512 * (1 - (1/4)^n) / (1 - 1/4)

def D (n : ℕ) : ℚ := 3072 * (1 - (1/(-3))^n) / (1 + 1/3)

theorem smallest_n_equality :
  ∃ (n : ℕ), n ≥ 1 ∧ C n = D n ∧ ∀ (m : ℕ), m ≥ 1 ∧ m < n → C m ≠ D m :=
sorry

end smallest_n_equality_l2858_285862


namespace toy_problem_solution_l2858_285812

/-- Represents the toy purchase and sale problem -/
structure ToyProblem where
  first_purchase_cost : ℝ
  second_purchase_cost : ℝ
  cost_increase_rate : ℝ
  quantity_decrease : ℕ
  min_profit : ℝ

/-- Calculates the cost per item for the first purchase -/
def first_item_cost (p : ToyProblem) : ℝ :=
  50

/-- Calculates the minimum selling price to achieve the desired profit -/
def min_selling_price (p : ToyProblem) : ℝ :=
  70

/-- Theorem stating the correctness of the calculated values -/
theorem toy_problem_solution (p : ToyProblem)
  (h1 : p.first_purchase_cost = 3000)
  (h2 : p.second_purchase_cost = 3000)
  (h3 : p.cost_increase_rate = 0.2)
  (h4 : p.quantity_decrease = 10)
  (h5 : p.min_profit = 1700) :
  first_item_cost p = 50 ∧
  min_selling_price p = 70 ∧
  (min_selling_price p * (p.first_purchase_cost / first_item_cost p +
    p.second_purchase_cost / (first_item_cost p * (1 + p.cost_increase_rate))) -
    (p.first_purchase_cost + p.second_purchase_cost) ≥ p.min_profit) :=
  sorry

end toy_problem_solution_l2858_285812


namespace not_prime_if_perfect_square_l2858_285811

theorem not_prime_if_perfect_square (n : ℕ) (h1 : n > 0) 
  (h2 : ∃ k : ℕ, n * (n + 2013) = k^2) : ¬ Prime n := by
  sorry

end not_prime_if_perfect_square_l2858_285811


namespace line_parallel_to_x_axis_l2858_285835

/-- 
A line through two points (x₁, y₁) and (x₂, y₂) is parallel to the x-axis 
if and only if y₁ = y₂.
-/
def parallel_to_x_axis (x₁ y₁ x₂ y₂ : ℝ) : Prop := y₁ = y₂

/-- 
The value of k for which the line through the points (3, 2k+1) and (8, 4k-5) 
is parallel to the x-axis.
-/
theorem line_parallel_to_x_axis (k : ℝ) : 
  parallel_to_x_axis 3 (2*k+1) 8 (4*k-5) ↔ k = 3 := by
  sorry

#check line_parallel_to_x_axis

end line_parallel_to_x_axis_l2858_285835


namespace a_lt_neg_four_sufficient_not_necessary_l2858_285813

-- Define the function f(x) = ax + 3
def f (a : ℝ) (x : ℝ) : ℝ := a * x + 3

-- Define the interval [-1, 1]
def interval : Set ℝ := Set.Icc (-1) 1

-- Define what it means for f to have a zero in the interval
def has_zero_in_interval (a : ℝ) : Prop :=
  ∃ x ∈ interval, f a x = 0

-- State the theorem
theorem a_lt_neg_four_sufficient_not_necessary :
  (∀ a : ℝ, a < -4 → has_zero_in_interval a) ∧
  ¬(∀ a : ℝ, has_zero_in_interval a → a < -4) :=
sorry

end a_lt_neg_four_sufficient_not_necessary_l2858_285813


namespace spelling_bee_contest_l2858_285806

theorem spelling_bee_contest (initial_students : ℕ) : 
  (initial_students : ℚ) * (1 - 0.66) * (1 - 3/4) = 30 →
  initial_students = 120 := by
sorry

end spelling_bee_contest_l2858_285806


namespace sum_of_powers_equals_99_l2858_285876

theorem sum_of_powers_equals_99 :
  3^4 + (-3)^3 + (-3)^2 + (-3)^1 + 3^1 + 3^2 + 3^3 = 99 := by
  sorry

end sum_of_powers_equals_99_l2858_285876


namespace ellipse_and_chord_properties_l2858_285874

-- Define the ellipse C₁
def ellipse (x y : ℝ) : Prop := x^2 / 6 + y^2 / 2 = 1

-- Define the line l₂
def line (x y : ℝ) : Prop := y = Real.sqrt 3 * (x - 2)

-- Define the intersection points
def intersection_points (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧ line x₁ y₁ ∧ line x₂ y₂

theorem ellipse_and_chord_properties :
  -- The ellipse equation is correct
  (∀ x y, ellipse x y ↔ x^2 / 6 + y^2 / 2 = 1) ∧
  -- The chord length is correct
  (∀ x₁ y₁ x₂ y₂, intersection_points x₁ y₁ x₂ y₂ →
    Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) = 4 * Real.sqrt 6 / 5) :=
by sorry

end ellipse_and_chord_properties_l2858_285874


namespace triangle_area_l2858_285803

def a : Fin 2 → ℝ := ![3, 2]
def b : Fin 2 → ℝ := ![1, 5]

theorem triangle_area : 
  (1/2 : ℝ) * |Matrix.det !![a 0, a 1; b 0, b 1]| = (13/2 : ℝ) := by sorry

end triangle_area_l2858_285803


namespace derivative_f_at_zero_l2858_285846

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ :=
  if x ≠ 0 then Real.exp (x * Real.sin (5 * x)) - 1 else 0

-- State the theorem
theorem derivative_f_at_zero :
  deriv f 0 = 0 := by
  sorry

end

end derivative_f_at_zero_l2858_285846


namespace sqrt_meaningful_range_l2858_285875

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = 1 - x) ↔ x ≤ 1 := by
sorry

end sqrt_meaningful_range_l2858_285875


namespace odd_number_divisibility_l2858_285819

theorem odd_number_divisibility (a : ℤ) (h : ∃ n : ℤ, a = 2*n + 1) :
  ∃ k : ℤ, a^4 + 9*(9 - 2*a^2) = 16*k := by
sorry

end odd_number_divisibility_l2858_285819


namespace journey_problem_l2858_285823

theorem journey_problem (total_distance : ℝ) (days : ℕ) (q : ℝ) :
  total_distance = 378 ∧ days = 6 ∧ q = 1/2 →
  ∃ a : ℝ, a * (1 - q^days) / (1 - q) = total_distance ∧ a * q^(days - 1) = 6 := by
  sorry

end journey_problem_l2858_285823


namespace polynomial_value_l2858_285821

theorem polynomial_value (x : ℝ) (h : x^2 + 2*x - 2 = 0) : 4 - 2*x - x^2 = 2 := by
  sorry

end polynomial_value_l2858_285821


namespace unique_m_value_l2858_285804

theorem unique_m_value (a b c m : ℤ) 
  (h1 : 0 ≤ m ∧ m ≤ 26)
  (h2 : (a + b + c) % 27 = m)
  (h3 : ((a - b) * (b - c) * (c - a)) % 27 = m) : 
  m = 0 := by
sorry

end unique_m_value_l2858_285804


namespace even_periodic_function_monotonicity_l2858_285867

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def monotone_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x ≤ y → f x ≤ f y

theorem even_periodic_function_monotonicity
  (f : ℝ → ℝ) (h_even : is_even f) (h_period : has_period f 2) :
  (monotone_on f (Set.Icc 0 1)) ↔ (∀ x ∈ Set.Icc 3 4, ∀ y ∈ Set.Icc 3 4, x ≤ y → f y ≤ f x) :=
sorry

end even_periodic_function_monotonicity_l2858_285867


namespace partnership_profit_calculation_l2858_285896

/-- Calculates the total profit of a partnership given the investments and one partner's share --/
def calculate_total_profit (invest_a invest_b invest_c c_share : ℕ) : ℕ :=
  let total_parts := invest_a + invest_b + invest_c
  let c_parts := invest_c
  let profit_per_part := c_share / c_parts
  profit_per_part * total_parts

/-- Theorem stating that given the specific investments and C's share, the total profit is 90000 --/
theorem partnership_profit_calculation :
  calculate_total_profit 30000 45000 50000 36000 = 90000 := by
  sorry

#eval calculate_total_profit 30000 45000 50000 36000

end partnership_profit_calculation_l2858_285896


namespace ellipse_foci_distance_l2858_285848

/-- The distance between the foci of an ellipse given by 4x^2 - 16x + y^2 + 10y + 5 = 0 is 6√3 -/
theorem ellipse_foci_distance :
  ∃ (h k a b : ℝ),
    (∀ x y : ℝ, 4*x^2 - 16*x + y^2 + 10*y + 5 = 0 ↔ (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1) →
    a > b →
    2 * Real.sqrt (a^2 - b^2) = 6 * Real.sqrt 3 :=
by sorry

end ellipse_foci_distance_l2858_285848


namespace distance_between_cities_l2858_285889

/-- The distance between City A and City B in miles -/
def distance : ℝ := 427.5

/-- The travel time from City A to City B in hours -/
def time_A_to_B : ℝ := 6

/-- The travel time from City B to City A in hours -/
def time_B_to_A : ℝ := 4.5

/-- The time saved on each trip in hours -/
def time_saved : ℝ := 0.5

/-- The average speed for the round trip if time were saved, in miles per hour -/
def average_speed : ℝ := 90

theorem distance_between_cities :
  distance = 427.5 ∧
  (2 * distance) / (time_A_to_B + time_B_to_A - 2 * time_saved) = average_speed :=
sorry

end distance_between_cities_l2858_285889


namespace technicians_count_l2858_285884

/-- Proves the number of technicians in a workshop with given salary conditions -/
theorem technicians_count (total_workers : ℕ) (avg_salary : ℕ) (tech_salary : ℕ) (rest_salary : ℕ) :
  total_workers = 14 ∧ 
  avg_salary = 8000 ∧ 
  tech_salary = 10000 ∧ 
  rest_salary = 6000 → 
  ∃ (tech_count : ℕ),
    tech_count = 7 ∧ 
    tech_count ≤ total_workers ∧
    tech_count * tech_salary + (total_workers - tech_count) * rest_salary = total_workers * avg_salary :=
by sorry

end technicians_count_l2858_285884


namespace money_distribution_l2858_285852

/-- Given three people A, B, and C with money, prove that B and C together have 320 Rs. -/
theorem money_distribution (A B C : ℕ) 
  (total : A + B + C = 500)
  (ac_sum : A + C = 200)
  (c_amount : C = 20) : 
  B + C = 320 := by
  sorry

end money_distribution_l2858_285852


namespace golden_ratio_expression_l2858_285890

theorem golden_ratio_expression (R : ℝ) (h1 : R^2 + R - 1 = 0) (h2 : R > 0) :
  R^(R^(R^2 + 1/R) + 1/R) + 1/R = 2 := by
  sorry

end golden_ratio_expression_l2858_285890


namespace birthday_box_crayons_l2858_285881

/-- The number of crayons Paul gave away -/
def crayons_given : ℕ := 571

/-- The number of crayons Paul lost -/
def crayons_lost : ℕ := 161

/-- The difference between crayons given away and lost -/
def crayons_difference : ℕ := 410

/-- Theorem: The number of crayons in Paul's birthday box is 732 -/
theorem birthday_box_crayons :
  crayons_given + crayons_lost = 732 ∧
  crayons_given - crayons_lost = crayons_difference :=
by sorry

end birthday_box_crayons_l2858_285881


namespace parallel_lines_slope_l2858_285833

theorem parallel_lines_slope (a : ℝ) : 
  (∃ (b c : ℝ), (∀ x y : ℝ, y = (a^2 - a) * x + 2 ↔ y = 6 * x + 3)) → 
  (a = -2 ∨ a = 3) := by
  sorry

end parallel_lines_slope_l2858_285833


namespace distance_after_walk_l2858_285844

/-- The distance from the starting point after walking 5 miles east, 
    turning 45 degrees north, and walking 7 miles. -/
theorem distance_after_walk (east_distance : ℝ) (angle : ℝ) (final_distance : ℝ) 
  (h1 : east_distance = 5)
  (h2 : angle = 45)
  (h3 : final_distance = 7) : 
  Real.sqrt (74 + 35 * Real.sqrt 2) = 
    Real.sqrt ((east_distance + final_distance * Real.sqrt 2 / 2) ^ 2 + 
               (final_distance * Real.sqrt 2 / 2) ^ 2) :=
by sorry

end distance_after_walk_l2858_285844


namespace multiply_after_subtract_l2858_285870

theorem multiply_after_subtract (n : ℝ) (x : ℝ) : n = 12 → 4 * n - 3 = (n - 7) * x → x = 9 := by
  sorry

end multiply_after_subtract_l2858_285870


namespace square_modification_l2858_285832

theorem square_modification (x : ℝ) : 
  x > 0 →
  x^2 = (x - 2) * (1.2 * x) →
  x = 12 :=
by sorry

end square_modification_l2858_285832


namespace square_of_binomial_condition_l2858_285831

theorem square_of_binomial_condition (a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, 9*x^2 - 18*x + a = (3*x + b)^2) → a = 9 := by
  sorry

end square_of_binomial_condition_l2858_285831


namespace combined_molecular_weight_l2858_285818

/-- Atomic weight of Carbon in g/mol -/
def carbon_weight : Float := 12.01

/-- Atomic weight of Hydrogen in g/mol -/
def hydrogen_weight : Float := 1.008

/-- Atomic weight of Oxygen in g/mol -/
def oxygen_weight : Float := 16.00

/-- Molecular weight of Butanoic acid (C4H8O2) in g/mol -/
def butanoic_weight : Float :=
  4 * carbon_weight + 8 * hydrogen_weight + 2 * oxygen_weight

/-- Molecular weight of Propanoic acid (C3H6O2) in g/mol -/
def propanoic_weight : Float :=
  3 * carbon_weight + 6 * hydrogen_weight + 2 * oxygen_weight

/-- Number of moles of Butanoic acid in the mixture -/
def butanoic_moles : Float := 9

/-- Number of moles of Propanoic acid in the mixture -/
def propanoic_moles : Float := 5

/-- Theorem: The combined molecular weight of the mixture is 1163.326 grams -/
theorem combined_molecular_weight :
  butanoic_moles * butanoic_weight + propanoic_moles * propanoic_weight = 1163.326 := by
  sorry

end combined_molecular_weight_l2858_285818


namespace intersection_line_is_correct_l2858_285814

/-- The canonical equations of a line that is the intersection of two planes. -/
def is_intersection_line (p₁ p₂ : ℝ → ℝ → ℝ → Prop) (l : ℝ → ℝ → ℝ → Prop) : Prop :=
  ∀ x y z, l x y z ↔ (p₁ x y z ∧ p₂ x y z)

/-- The first plane equation -/
def plane1 (x y z : ℝ) : Prop := 6 * x - 7 * y - z - 2 = 0

/-- The second plane equation -/
def plane2 (x y z : ℝ) : Prop := x + 7 * y - 4 * z - 5 = 0

/-- The canonical equations of the line -/
def line (x y z : ℝ) : Prop := (x - 1) / 35 = (y - 4/7) / 23 ∧ (x - 1) / 35 = z / 49

theorem intersection_line_is_correct :
  is_intersection_line plane1 plane2 line := by sorry

end intersection_line_is_correct_l2858_285814


namespace articles_sold_at_cost_price_l2858_285837

theorem articles_sold_at_cost_price :
  ∀ (X : ℕ) (C S : ℝ),
  X * C = 32 * S →
  S = C * (1 + 0.5625) →
  X = 50 :=
by
  sorry

end articles_sold_at_cost_price_l2858_285837


namespace triangle_max_area_l2858_285877

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C,
    if (3+b)(sin A - sin B) = (c-b)sin C and a = 3,
    then the maximum area of triangle ABC is 9√3/4 -/
theorem triangle_max_area (a b c A B C : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧
  (3 + b) * (Real.sin A - Real.sin B) = (c - b) * Real.sin C ∧
  a = 3 →
  (∃ (S : ℝ), S = (1/2) * b * c * Real.sin A ∧ 
    ∀ (S' : ℝ), S' = (1/2) * b * c * Real.sin A → S' ≤ S) ∧
  (1/2) * b * c * Real.sin A ≤ (9 * Real.sqrt 3) / 4 :=
by sorry


end triangle_max_area_l2858_285877


namespace exactly_three_rainy_days_probability_l2858_285834

/-- The probability of exactly k successes in n independent trials with probability p of success in each trial -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

/-- The number of days in the period -/
def num_days : ℕ := 4

/-- The number of rainy days we're interested in -/
def num_rainy_days : ℕ := 3

/-- The probability of rain on any given day -/
def rain_probability : ℝ := 0.5

theorem exactly_three_rainy_days_probability :
  binomial_probability num_days num_rainy_days rain_probability = 0.25 := by
  sorry

end exactly_three_rainy_days_probability_l2858_285834


namespace percentage_difference_l2858_285865

theorem percentage_difference : (0.60 * 50) - (0.45 * 30) = 16.5 := by
  sorry

end percentage_difference_l2858_285865


namespace fifth_color_count_l2858_285894

/-- Represents the number of marbles of each color in a box -/
structure MarbleCount where
  red : ℕ
  green : ℕ
  yellow : ℕ
  blue : ℕ
  fifth : ℕ

/-- Defines the properties of the marble counts as given in the problem -/
def valid_marble_count (m : MarbleCount) : Prop :=
  m.red = 25 ∧
  m.green = 3 * m.red ∧
  m.yellow = m.green / 5 ∧
  m.blue = 2 * m.yellow ∧
  m.fifth = (m.red + m.blue) + (m.red + m.blue) / 2 ∧
  m.red + m.green + m.yellow + m.blue + m.fifth = 4 * m.green

theorem fifth_color_count (m : MarbleCount) (h : valid_marble_count m) : m.fifth = 155 := by
  sorry

end fifth_color_count_l2858_285894


namespace system_solution_l2858_285869

theorem system_solution : ∃ (x y : ℝ), x + 3 * y = 7 ∧ y = 2 * x ∧ x = 1 ∧ y = 2 := by
  sorry

end system_solution_l2858_285869


namespace S_max_l2858_285866

/-- The general term of the sequence -/
def a (n : ℕ) : ℤ := 26 - 2 * n

/-- The sum of the first n terms of the sequence -/
def S (n : ℕ) : ℤ := n * (26 - n)

/-- The theorem stating that S is maximized when n is 12 or 13 -/
theorem S_max : ∀ k : ℕ, S k ≤ max (S 12) (S 13) :=
sorry

end S_max_l2858_285866


namespace fraction_inequality_implies_inequality_l2858_285855

theorem fraction_inequality_implies_inequality (a b c : ℝ) :
  c ≠ 0 → (a / c^2 < b / c^2) → a < b := by
  sorry

end fraction_inequality_implies_inequality_l2858_285855


namespace abs_gt_one_iff_square_minus_one_gt_zero_l2858_285815

theorem abs_gt_one_iff_square_minus_one_gt_zero :
  ∀ x : ℝ, |x| > 1 ↔ x^2 - 1 > 0 := by sorry

end abs_gt_one_iff_square_minus_one_gt_zero_l2858_285815
