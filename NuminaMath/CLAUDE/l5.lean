import Mathlib

namespace ln_cube_relation_l5_544

theorem ln_cube_relation (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (Real.log x > Real.log y → x^3 > y^3) ∧
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a^3 > b^3 ∧ ¬(Real.log a > Real.log b) :=
sorry

end ln_cube_relation_l5_544


namespace total_amount_after_four_years_l5_524

/-- Jo's annual earnings in USD -/
def annual_earnings : ℕ := 3^5 - 3^4 + 3^3 - 3^2 + 3

/-- Annual investment return in USD -/
def investment_return : ℕ := 2^5 - 2^4 + 2^3 - 2^2 + 2

/-- Number of years -/
def years : ℕ := 4

/-- Theorem stating the total amount after four years -/
theorem total_amount_after_four_years : 
  (annual_earnings + investment_return) * years = 820 := by
  sorry

end total_amount_after_four_years_l5_524


namespace ceiling_product_equation_l5_598

theorem ceiling_product_equation : ∃ x : ℝ, ⌈x⌉ * x = 210 ∧ x = 14 := by
  sorry

end ceiling_product_equation_l5_598


namespace function_from_derivative_and_point_l5_561

open Real

theorem function_from_derivative_and_point (f : ℝ → ℝ) 
  (h1 : ∀ x, HasDerivAt f (4 * x^3) x) 
  (h2 : f 1 = -1) : 
  ∀ x, f x = x^4 - 2 := by
  sorry

end function_from_derivative_and_point_l5_561


namespace hulk_jump_exceeds_1500_l5_590

def hulk_jump (n : ℕ) : ℝ := 2 * (3 : ℝ) ^ (n - 1)

theorem hulk_jump_exceeds_1500 :
  ∀ k < 8, hulk_jump k ≤ 1500 ∧ hulk_jump 8 > 1500 := by sorry

end hulk_jump_exceeds_1500_l5_590


namespace number_with_special_average_l5_583

theorem number_with_special_average (x : ℝ) (h1 : x ≠ 0) :
  (x + x^2) / 2 = 5 * x → x = 9 := by
sorry

end number_with_special_average_l5_583


namespace infinitely_many_coprimes_in_arithmetic_sequence_l5_530

theorem infinitely_many_coprimes_in_arithmetic_sequence 
  (a b m : ℕ+) (h : Nat.Coprime a b) :
  ∃ (s : Set ℕ), Set.Infinite s ∧ ∀ k ∈ s, Nat.Coprime (a + k * b) m :=
sorry

end infinitely_many_coprimes_in_arithmetic_sequence_l5_530


namespace point_on_linear_graph_l5_578

theorem point_on_linear_graph (a : ℝ) : (1 : ℝ) = 3 * a + 4 → a = -1 := by
  sorry

end point_on_linear_graph_l5_578


namespace total_coins_last_month_l5_509

/-- The number of coins Mathilde had at the start of this month -/
def mathilde_this_month : ℕ := 100

/-- The number of coins Salah had at the start of this month -/
def salah_this_month : ℕ := 100

/-- The percentage increase in Mathilde's coins from last month to this month -/
def mathilde_increase_percent : ℚ := 25/100

/-- The percentage decrease in Salah's coins from last month to this month -/
def salah_decrease_percent : ℚ := 20/100

/-- The number of coins Mathilde had at the start of last month -/
def mathilde_last_month : ℚ := mathilde_this_month / (1 + mathilde_increase_percent)

/-- The number of coins Salah had at the start of last month -/
def salah_last_month : ℚ := salah_this_month / (1 - salah_decrease_percent)

theorem total_coins_last_month :
  mathilde_last_month + salah_last_month = 205 := by sorry

end total_coins_last_month_l5_509


namespace system_solution_l5_562

theorem system_solution (x y : ℝ) 
  (eq1 : 4 * x - y = 2) 
  (eq2 : 3 * x - 2 * y = -1) : 
  x - y = -1 := by
sorry

end system_solution_l5_562


namespace dante_remaining_coconuts_l5_591

/-- Paolo's number of coconuts -/
def paolo_coconuts : ℕ := 14

/-- Dante's initial number of coconuts in terms of Paolo's -/
def dante_initial_coconuts : ℕ := 3 * paolo_coconuts

/-- Number of coconuts Dante sold -/
def dante_sold_coconuts : ℕ := 10

/-- Theorem: Dante has 32 coconuts left after selling -/
theorem dante_remaining_coconuts : 
  dante_initial_coconuts - dante_sold_coconuts = 32 := by sorry

end dante_remaining_coconuts_l5_591


namespace parabola_sum_l5_517

/-- Represents a parabola of the form y = ax^2 + c -/
structure Parabola where
  a : ℝ
  c : ℝ

/-- The area of the kite formed by the intersections of the parabolas with the axes -/
def kite_area (p1 p2 : Parabola) : ℝ := 12

/-- The parabolas intersect the coordinate axes at exactly four points -/
def intersect_at_four_points (p1 p2 : Parabola) : Prop := sorry

theorem parabola_sum (p1 p2 : Parabola) 
  (h1 : p1.c = -2 ∧ p2.c = 4) 
  (h2 : intersect_at_four_points p1 p2) 
  (h3 : kite_area p1 p2 = 12) : 
  p1.a + p2.a = 1.5 := by sorry

end parabola_sum_l5_517


namespace negation_of_universal_proposition_l5_581

theorem negation_of_universal_proposition 
  (f : ℝ → ℝ) (m : ℝ) : 
  (¬ ∀ x, f x ≥ m) ↔ (∃ x, f x < m) := by sorry

end negation_of_universal_proposition_l5_581


namespace karen_picked_up_three_cases_l5_541

/-- The number of boxes of Tagalongs Karen sold -/
def boxes_sold : ℕ := 36

/-- The number of boxes in each case -/
def boxes_per_case : ℕ := 12

/-- The number of cases Karen picked up -/
def cases_picked_up : ℕ := boxes_sold / boxes_per_case

theorem karen_picked_up_three_cases : cases_picked_up = 3 := by
  sorry

end karen_picked_up_three_cases_l5_541


namespace distance_in_15_minutes_l5_538

/-- Given a constant speed calculated from driving 80 miles in 2 hours, 
    prove that the distance traveled in 15 minutes is 10 miles. -/
theorem distance_in_15_minutes (total_distance : ℝ) (total_time : ℝ) 
  (travel_time : ℝ) (h1 : total_distance = 80) (h2 : total_time = 2) 
  (h3 : travel_time = 15 / 60) : 
  (total_distance / total_time) * travel_time = 10 := by
  sorry

end distance_in_15_minutes_l5_538


namespace circular_path_diameter_increase_l5_580

theorem circular_path_diameter_increase 
  (original_rounds : ℕ) 
  (original_time : ℝ) 
  (new_time : ℝ) 
  (original_rounds_pos : original_rounds > 0)
  (original_time_pos : original_time > 0)
  (new_time_pos : new_time > 0)
  (h_original : original_rounds = 8)
  (h_original_time : original_time = 40)
  (h_new_time : new_time = 50) :
  let original_single_round_time := original_time / original_rounds
  let diameter_increase_factor := new_time / original_single_round_time
  diameter_increase_factor = 10 := by
  sorry

end circular_path_diameter_increase_l5_580


namespace total_stuffed_animals_l5_547

theorem total_stuffed_animals (mckenna kenley tenly : ℕ) : 
  mckenna = 34 → 
  kenley = 2 * mckenna → 
  tenly = kenley + 5 → 
  mckenna + kenley + tenly = 175 :=
by
  sorry

end total_stuffed_animals_l5_547


namespace negation_true_l5_504

theorem negation_true : 
  ¬(∀ a : ℝ, a ≤ 3 → a^2 < 9) ↔ True :=
by sorry

end negation_true_l5_504


namespace circle_equation_l5_548

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a line
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the conditions
def passes_through (c : Circle) (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 = c.radius^2

def is_tangent_to (c : Circle) (l : Line) : Prop :=
  let (cx, cy) := c.center
  |l.a * cx + l.b * cy + l.c| / Real.sqrt (l.a^2 + l.b^2) = c.radius

def center_on_line (c : Circle) (l : Line) : Prop :=
  let (cx, cy) := c.center
  l.a * cx + l.b * cy + l.c = 0

-- Theorem statement
theorem circle_equation (c : Circle) :
  passes_through c (0, -1) ∧
  is_tangent_to c { a := 1, b := 1, c := -1 } ∧
  center_on_line c { a := 2, b := 1, c := 0 } →
  ((∀ x y, (x - 1)^2 + (y + 2)^2 = 2 ↔ passes_through c (x, y)) ∨
   (∀ x y, (x - 1/9)^2 + (y + 2/9)^2 = 50/81 ↔ passes_through c (x, y))) :=
by sorry

end circle_equation_l5_548


namespace range_of_b_l5_577

def solution_set (b : ℝ) : Set ℝ := {x : ℝ | |3*x - b| < 4}

theorem range_of_b :
  (∃ b : ℝ, solution_set b = {1, 2, 3}) →
  (∀ b : ℝ, solution_set b = {1, 2, 3} → b ∈ Set.Ioo 5 7) ∧
  (∀ b : ℝ, b ∈ Set.Ioo 5 7 → solution_set b = {1, 2, 3}) :=
by sorry

end range_of_b_l5_577


namespace speed_from_x_to_y_l5_552

/-- Proves that given two towns and specific travel conditions, the speed from x to y is 60 km/hr -/
theorem speed_from_x_to_y (D : ℝ) (V : ℝ) (h : D > 0) : 
  (2 * D) / (D / V + D / 36) = 45 → V = 60 := by
  sorry

end speed_from_x_to_y_l5_552


namespace runs_by_running_percentage_l5_555

def total_runs : ℕ := 120
def num_boundaries : ℕ := 3
def num_sixes : ℕ := 8
def runs_per_boundary : ℕ := 4
def runs_per_six : ℕ := 6

theorem runs_by_running_percentage : 
  (total_runs - (num_boundaries * runs_per_boundary + num_sixes * runs_per_six)) / total_runs * 100 = 50 := by
  sorry

end runs_by_running_percentage_l5_555


namespace land_area_calculation_l5_594

/-- The total area of 9 square-shaped plots of land, each measuring 6 meters in length and width, is 324 square meters. -/
theorem land_area_calculation (num_plots : ℕ) (side_length : ℝ) : 
  num_plots = 9 → side_length = 6 → num_plots * (side_length * side_length) = 324 := by
  sorry

end land_area_calculation_l5_594


namespace pizza_toppings_theorem_l5_516

/-- Represents the number of distinct toppings on a pizza slice -/
def toppings_on_slice (n k : ℕ+) (t : Fin (2 * k)) : ℕ :=
  sorry

/-- The minimum number of distinct toppings on any slice -/
def min_toppings (n k : ℕ+) : ℕ :=
  sorry

/-- The maximum number of distinct toppings on any slice -/
def max_toppings (n k : ℕ+) : ℕ :=
  sorry

/-- Theorem stating that the sum of minimum and maximum toppings equals the total number of toppings -/
theorem pizza_toppings_theorem (n k : ℕ+) :
    min_toppings n k + max_toppings n k = n :=
  sorry

end pizza_toppings_theorem_l5_516


namespace triangle_tangent_half_angles_sum_l5_511

theorem triangle_tangent_half_angles_sum (A B C : ℝ) 
  (h : A + B + C = π) : 
  Real.tan (A/2) * Real.tan (B/2) + Real.tan (B/2) * Real.tan (C/2) + Real.tan (C/2) * Real.tan (A/2) = 1 := by
  sorry

end triangle_tangent_half_angles_sum_l5_511


namespace equation_2x_minus_y_eq_2_is_linear_l5_519

/-- A linear equation in two variables is of the form ax + by + c = 0, where a and b are not both zero -/
def is_linear_equation (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧ ∀ x y, f x y = a * x + b * y + c

/-- The function representing the equation 2x - y = 2 -/
def f (x y : ℝ) : ℝ := 2 * x - y - 2

theorem equation_2x_minus_y_eq_2_is_linear : is_linear_equation f :=
sorry

end equation_2x_minus_y_eq_2_is_linear_l5_519


namespace set_equality_implies_sum_l5_593

theorem set_equality_implies_sum (a b : ℝ) : 
  ({0, b, b/a} : Set ℝ) = {1, a, a+b} → a + 2*b = 1 := by
  sorry

end set_equality_implies_sum_l5_593


namespace variance_mean_preserved_l5_558

def initial_set : List Int := [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]

def mean (xs : List Int) : ℚ := (xs.sum : ℚ) / xs.length

def variance (xs : List Int) : ℚ :=
  let m := mean xs
  (xs.map (fun x => ((x : ℚ) - m) ^ 2)).sum / xs.length

def replace_4_with_neg1_and_5 (xs : List Int) : List Int :=
  xs.filter (· ≠ 4) ++ [-1, 5]

def replace_neg4_with_1_and_neg5 (xs : List Int) : List Int :=
  xs.filter (· ≠ -4) ++ [1, -5]

theorem variance_mean_preserved :
  (mean initial_set = mean (replace_4_with_neg1_and_5 initial_set) ∧
   variance initial_set = variance (replace_4_with_neg1_and_5 initial_set)) ∨
  (mean initial_set = mean (replace_neg4_with_1_and_neg5 initial_set) ∧
   variance initial_set = variance (replace_neg4_with_1_and_neg5 initial_set)) :=
by sorry

end variance_mean_preserved_l5_558


namespace no_rectangle_from_five_distinct_squares_l5_576

/-- A configuration of five squares with side lengths q₁, q₂, q₃, q₄, q₅ -/
structure FiveSquares where
  q₁ : ℝ
  q₂ : ℝ
  q₃ : ℝ
  q₄ : ℝ
  q₅ : ℝ
  h₁ : 0 < q₁
  h₂ : q₁ < q₂
  h₃ : q₂ < q₃
  h₄ : q₃ < q₄
  h₅ : q₄ < q₅

/-- Predicate to check if the five squares can form a rectangle -/
def CanFormRectangle (s : FiveSquares) : Prop :=
  ∃ (w h : ℝ), w > 0 ∧ h > 0 ∧ w * h = s.q₁^2 + s.q₂^2 + s.q₃^2 + s.q₄^2 + s.q₅^2

/-- Theorem stating that it's impossible to form a rectangle with five squares of distinct sizes -/
theorem no_rectangle_from_five_distinct_squares :
  ¬∃ (s : FiveSquares), CanFormRectangle s := by
  sorry

end no_rectangle_from_five_distinct_squares_l5_576


namespace arcsin_arccos_equation_solution_l5_545

theorem arcsin_arccos_equation_solution :
  ∃ x : ℝ, x = -1 / Real.sqrt 7 ∧ Real.arcsin (3 * x) - Real.arccos (2 * x) = π / 6 :=
by sorry

end arcsin_arccos_equation_solution_l5_545


namespace arrangement_count_l5_506

def number_of_people : Nat := 6
def number_of_special_people : Nat := 3

theorem arrangement_count : 
  (number_of_people : Nat) = 6 →
  (number_of_special_people : Nat) = 3 →
  (∃ (arrangement_count : Nat), arrangement_count = 144 ∧
    arrangement_count = (number_of_people - number_of_special_people).factorial * 
                        (number_of_people - number_of_special_people + 1).choose number_of_special_people) :=
by
  sorry

end arrangement_count_l5_506


namespace puppies_sold_l5_599

theorem puppies_sold (initial_puppies : ℕ) (puppies_per_cage : ℕ) (cages_used : ℕ) : 
  initial_puppies = 13 → puppies_per_cage = 2 → cages_used = 3 →
  initial_puppies - (puppies_per_cage * cages_used) = 7 := by
  sorry

end puppies_sold_l5_599


namespace jane_drawing_paper_l5_563

/-- The number of old, brown sheets of drawing paper Jane has. -/
def brown_sheets : ℕ := 28

/-- The number of old, yellow sheets of drawing paper Jane has. -/
def yellow_sheets : ℕ := 27

/-- The total number of sheets of drawing paper Jane has. -/
def total_sheets : ℕ := brown_sheets + yellow_sheets

theorem jane_drawing_paper :
  total_sheets = 55 := by sorry

end jane_drawing_paper_l5_563


namespace jim_journey_remaining_distance_l5_527

/-- Calculates the remaining distance to drive given the total journey distance and the distance already driven. -/
def remaining_distance (total_distance driven_distance : ℕ) : ℕ :=
  total_distance - driven_distance

/-- Theorem stating that for a 1200-mile journey with 923 miles driven, the remaining distance is 277 miles. -/
theorem jim_journey_remaining_distance :
  remaining_distance 1200 923 = 277 := by
  sorry

end jim_journey_remaining_distance_l5_527


namespace sufficient_but_not_necessary_l5_584

theorem sufficient_but_not_necessary (a : ℝ) :
  (∀ a ≥ 0, ∃ x : ℝ, a * x^2 + x + 1 ≥ 0) ∧
  (∃ a < 0, ∃ x : ℝ, a * x^2 + x + 1 ≥ 0) :=
by sorry

end sufficient_but_not_necessary_l5_584


namespace min_value_ratio_l5_520

theorem min_value_ratio (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (∃ (x y : ℝ), x > y ∧ y > 0 ∧ 
    2*x + y + 1/(x-y) + 4/(x+2*y) < 2*a + b + 1/(a-b) + 4/(a+2*b)) ∨
  a/b = 4 := by
sorry

end min_value_ratio_l5_520


namespace base_2_representation_of_236_l5_539

theorem base_2_representation_of_236 :
  ∃ (a : List Bool),
    a.length = 9 ∧
    a = [true, true, true, false, true, false, true, false, false] ∧
    (a.foldr (λ (b : Bool) (acc : Nat) => 2 * acc + if b then 1 else 0) 0) = 236 :=
by sorry

end base_2_representation_of_236_l5_539


namespace unique_prime_with_prime_sums_l5_526

theorem unique_prime_with_prime_sums : ∃! p : ℕ, 
  Nat.Prime p ∧ Nat.Prime (p + 28) ∧ Nat.Prime (p + 56) ∧ p = 3 := by
  sorry

end unique_prime_with_prime_sums_l5_526


namespace subtraction_problem_l5_543

theorem subtraction_problem : 2000000000000 - 1111111111111 = 888888888889 := by
  sorry

end subtraction_problem_l5_543


namespace smallest_a1_l5_579

/-- Given a sequence of positive real numbers where aₙ = 8aₙ₋₁ - n² for all n > 1,
    the smallest possible value of a₁ is 2/7 -/
theorem smallest_a1 (a : ℕ → ℝ) (h_pos : ∀ n, a n > 0) 
    (h_rec : ∀ n > 1, a n = 8 * a (n - 1) - n^2) :
  ∀ a₁ > 0, (∀ n > 1, a n = 8 * a (n - 1) - n^2) → a₁ ≥ 2/7 :=
by sorry

end smallest_a1_l5_579


namespace repeating_decimal_sum_l5_523

/-- Represents a repeating decimal with a single digit repeating -/
def RepeatingDecimal (n : ℕ) : ℚ := n / 9

/-- The sum of specific repeating decimals -/
theorem repeating_decimal_sum :
  RepeatingDecimal 6 + RepeatingDecimal 2 - RepeatingDecimal 4 = 4 / 9 := by
  sorry

end repeating_decimal_sum_l5_523


namespace earthwork_transport_theorem_prove_earthwork_transport_l5_559

/-- Represents the capacity of earthwork transport vehicles -/
structure VehicleCapacity where
  large : ℕ
  small : ℕ

/-- Represents a dispatch plan for earthwork transport vehicles -/
structure DispatchPlan where
  large : ℕ
  small : ℕ

/-- Theorem stating the correct vehicle capacities and possible dispatch plans -/
theorem earthwork_transport_theorem 
  (capacity : VehicleCapacity)
  (plans : List DispatchPlan) : Prop :=
  -- Conditions
  (3 * capacity.large + 4 * capacity.small = 44) ∧
  (4 * capacity.large + 6 * capacity.small = 62) ∧
  (∀ plan ∈ plans, 
    plan.large + plan.small = 12 ∧
    plan.small ≥ 4 ∧
    plan.large * capacity.large + plan.small * capacity.small ≥ 78) ∧
  -- Conclusions
  (capacity.large = 8 ∧ capacity.small = 5) ∧
  (plans = [
    DispatchPlan.mk 8 4,
    DispatchPlan.mk 7 5,
    DispatchPlan.mk 6 6
  ])

/-- Proof of the earthwork transport theorem -/
theorem prove_earthwork_transport : 
  ∃ (capacity : VehicleCapacity) (plans : List DispatchPlan),
    earthwork_transport_theorem capacity plans := by
  sorry

end earthwork_transport_theorem_prove_earthwork_transport_l5_559


namespace line_slope_points_l5_529

/-- Given m > 0 and three points on a line with slope m^2, prove m = √3 --/
theorem line_slope_points (m : ℝ) 
  (h_pos : m > 0)
  (h_line : ∃ (k b : ℝ), k = m^2 ∧ 
    3 = k * m + b ∧ 
    m = k * 1 + b ∧ 
    m^2 = k * 2 + b) : 
  m = Real.sqrt 3 := by
sorry

end line_slope_points_l5_529


namespace athlete_shots_l5_567

theorem athlete_shots (x y z : ℕ) : 
  x > 0 → y > 0 → z > 0 →  -- Each point value scored at least once
  x + y + z > 11 →         -- More than 11 shots
  8*x + 9*y + 10*z = 100 → -- Total score is 100
  x = 9                    -- Number of 8-point shots is 9
  := by sorry

end athlete_shots_l5_567


namespace unique_solution_xyz_l5_505

theorem unique_solution_xyz : 
  ∀ x y z : ℕ+, 
    (x : ℤ) + (y : ℤ)^2 + (z : ℤ)^3 = (x : ℤ) * (y : ℤ) * (z : ℤ) → 
    z = Nat.gcd x y → 
    (x = 5 ∧ y = 1 ∧ z = 1) :=
by sorry

end unique_solution_xyz_l5_505


namespace min_points_to_guarantee_highest_score_eighteen_points_achievable_smallest_points_to_guarantee_highest_score_l5_521

/-- Represents the possible points for a single race -/
inductive RacePoints
  | first : RacePoints
  | second : RacePoints
  | third : RacePoints
  | fourth : RacePoints

/-- Converts RacePoints to their numerical value -/
def pointValue (p : RacePoints) : Nat :=
  match p with
  | .first => 7
  | .second => 4
  | .third => 2
  | .fourth => 1

/-- Calculates the total points for a sequence of three races -/
def totalPoints (r1 r2 r3 : RacePoints) : Nat :=
  pointValue r1 + pointValue r2 + pointValue r3

/-- Theorem stating that 18 points is the minimum to guarantee the highest score -/
theorem min_points_to_guarantee_highest_score :
  ∀ (r1 r2 r3 : RacePoints),
    totalPoints r1 r2 r3 ≥ 18 →
    ∀ (s1 s2 s3 : RacePoints),
      totalPoints r1 r2 r3 > totalPoints s1 s2 s3 ∨
      (r1 = s1 ∧ r2 = s2 ∧ r3 = s3) :=
by sorry

/-- Theorem stating that 18 points is achievable -/
theorem eighteen_points_achievable :
  ∃ (r1 r2 r3 : RacePoints), totalPoints r1 r2 r3 = 18 :=
by sorry

/-- Main theorem combining the above results -/
theorem smallest_points_to_guarantee_highest_score :
  (∃ (r1 r2 r3 : RacePoints), totalPoints r1 r2 r3 = 18) ∧
  (∀ (r1 r2 r3 : RacePoints),
    totalPoints r1 r2 r3 ≥ 18 →
    ∀ (s1 s2 s3 : RacePoints),
      totalPoints r1 r2 r3 > totalPoints s1 s2 s3 ∨
      (r1 = s1 ∧ r2 = s2 ∧ r3 = s3)) ∧
  (∀ n : Nat, n < 18 →
    ∃ (s1 s2 s3 r1 r2 r3 : RacePoints),
      totalPoints s1 s2 s3 = n ∧
      totalPoints r1 r2 r3 > n) :=
by sorry

end min_points_to_guarantee_highest_score_eighteen_points_achievable_smallest_points_to_guarantee_highest_score_l5_521


namespace rectangular_prism_volume_l5_531

theorem rectangular_prism_volume 
  (a b c : ℝ) 
  (h1 : a * b = 10) 
  (h2 : b * c = 15) 
  (h3 : c * a = 18) : 
  a * b * c = 30 * Real.sqrt 3 := by
sorry

end rectangular_prism_volume_l5_531


namespace parabola_translation_l5_592

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := x^2 + 1

-- Define the translated parabola
def translated_parabola (x : ℝ) : ℝ := (x - 3)^2 - 1

-- Theorem statement
theorem parabola_translation :
  ∀ x y : ℝ, y = original_parabola (x - 3) - 2 ↔ y = translated_parabola x :=
by sorry

end parabola_translation_l5_592


namespace satisfactory_grades_fraction_l5_582

theorem satisfactory_grades_fraction :
  let grades := [3, 7, 4, 2, 4]  -- A, B, C, D, E+F
  let satisfactory := 4  -- Number of satisfactory grade categories (A, B, C, D)
  let total_students := grades.sum
  let satisfactory_students := (grades.take satisfactory).sum
  (satisfactory_students : ℚ) / total_students = 2 / 3 := by
  sorry

end satisfactory_grades_fraction_l5_582


namespace line_circle_intersection_l5_533

/-- Given a line mx + y + √3 = 0 intersecting a circle (x+1)² + y² = 2 with a chord length of 2,
    prove that m = √3/3 -/
theorem line_circle_intersection (m : ℝ) : 
  (∃ x y : ℝ, mx + y + Real.sqrt 3 = 0 ∧ (x + 1)^2 + y^2 = 2) → 
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    mx₁ + y₁ + Real.sqrt 3 = 0 ∧ (x₁ + 1)^2 + y₁^2 = 2 ∧
    mx₂ + y₂ + Real.sqrt 3 = 0 ∧ (x₂ + 1)^2 + y₂^2 = 2 ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 4) →
  m = Real.sqrt 3 / 3 := by
sorry

end line_circle_intersection_l5_533


namespace parabola_intersection_length_l5_569

/-- Represents a point on a parabola -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ

/-- Theorem: For a parabola x² = 2py (p > 0 and constant), if a line with slope 1
    passing through the focus intersects the parabola at points A and B,
    then the length of AB is 4p. -/
theorem parabola_intersection_length
  (p : ℝ)
  (hp : p > 0)
  (A B : ParabolaPoint)
  (h_parabola_A : A.x^2 = 2*p*A.y)
  (h_parabola_B : B.x^2 = 2*p*B.y)
  (h_line : B.y - A.y = B.x - A.x)
  (h_focus : ∃ (f : ℝ), A.y = A.x + f ∧ B.y = B.x + f ∧ f = p/2) :
  Real.sqrt ((B.x - A.x)^2 + (B.y - A.y)^2) = 4*p :=
sorry

end parabola_intersection_length_l5_569


namespace count_scalene_triangles_l5_514

def is_valid_scalene_triangle (a b c : ℕ) : Prop :=
  a < b ∧ b < c ∧ a + b > c ∧ a + c > b ∧ b + c > a ∧ a + b + c < 16

theorem count_scalene_triangles :
  ∃! (triangles : Finset (ℕ × ℕ × ℕ)),
    triangles.card = 6 ∧
    ∀ (t : ℕ × ℕ × ℕ), t ∈ triangles ↔ is_valid_scalene_triangle t.1 t.2.1 t.2.2 :=
by sorry

end count_scalene_triangles_l5_514


namespace sector_central_angle_l5_570

/-- Given a sector with perimeter 8 and area 4, its central angle is 2 radians -/
theorem sector_central_angle (l r : ℝ) (h1 : 2 * r + l = 8) (h2 : (1 / 2) * l * r = 4) :
  l / r = 2 := by
sorry

end sector_central_angle_l5_570


namespace tangent_line_perpendicular_range_of_m_l5_565

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := ((x + a) * Real.log x) / (x + 1)

-- Define the derivative of f(x)
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ :=
  ((Real.log x + (x + a) / x) * (x + 1) - (x + a) * Real.log x) / ((x + 1)^2)

-- Theorem for part (I)
theorem tangent_line_perpendicular (a : ℝ) :
  f_derivative a 1 = 1/2 → a = 0 :=
by sorry

-- Theorem for part (II)
theorem range_of_m (m : ℝ) :
  (∀ x ≥ 1, f 0 x ≤ m * (x - 1)) ↔ m ≥ 1/2 :=
by sorry

end tangent_line_perpendicular_range_of_m_l5_565


namespace complex_fraction_simplification_l5_542

def i : ℂ := Complex.I

theorem complex_fraction_simplification :
  (2 - 3 * i) / (1 + 4 * i) = -10/17 - 11/17 * i :=
by
  sorry

end complex_fraction_simplification_l5_542


namespace new_students_count_l5_550

theorem new_students_count (n : ℕ) : n < 600 → n % 28 = 27 → n % 26 = 20 → n = 615 :=
by
  sorry

end new_students_count_l5_550


namespace arithmetic_sequence_sum_l5_522

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

theorem arithmetic_sequence_sum (a : ℕ → ℚ) :
  arithmetic_sequence a → a 5 = 21 → a 4 + a 5 + a 6 = 63 := by
sorry

end arithmetic_sequence_sum_l5_522


namespace line_properties_l5_549

/-- A line in 2D space represented by ax + by + c = 0 -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A vector in 2D space -/
structure Vector2D where
  x : ℝ
  y : ℝ

def isDirectionalVector (v : Vector2D) (l : Line2D) : Prop :=
  v.y / v.x = -l.a / l.b

def hasEqualIntercepts (l : Line2D) : Prop :=
  -l.c / l.a = -l.c / l.b

def passesThrough (l : Line2D) (p : Point2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

def yIntercept (m : ℝ) (b : ℝ) : ℝ := b

theorem line_properties :
  let l1 : Line2D := ⟨2, 1, 3⟩
  let l2 : Line2D := ⟨1, 1, -6⟩
  let v : Vector2D := ⟨1, -2⟩
  let p : Point2D := ⟨2, 4⟩
  isDirectionalVector v l1 ∧
  hasEqualIntercepts l2 ∧
  passesThrough l2 p ∧
  yIntercept 3 (-2) = -2 := by sorry

end line_properties_l5_549


namespace equality_of_squared_terms_l5_501

theorem equality_of_squared_terms (a b : ℝ) : 7 * a^2 * b - 7 * b * a^2 = 0 := by
  sorry

end equality_of_squared_terms_l5_501


namespace choose_15_4_l5_571

theorem choose_15_4 : Nat.choose 15 4 = 1365 := by
  sorry

end choose_15_4_l5_571


namespace triangle_side_count_l5_508

theorem triangle_side_count : ∃! n : ℕ, 
  n = (Finset.filter (fun x : ℕ => 
    x > 0 ∧ x + 5 > 8 ∧ 8 + 5 > x
  ) (Finset.range 100)).card ∧ n = 9 := by
sorry

end triangle_side_count_l5_508


namespace system_solution_l5_586

theorem system_solution :
  ∃ (x y z : ℚ),
    (4 * x - 6 * y + 2 * z = -14) ∧
    (8 * x + 3 * y - z = -15) ∧
    (3 * x + z = 7) ∧
    (x = 100 / 33) ∧
    (y = 146 / 33) ∧
    (z = 29 / 11) :=
by sorry

end system_solution_l5_586


namespace unicorn_stitches_unicorn_stitches_proof_l5_535

/-- Proves that the number of stitches required to embroider a unicorn is 180 --/
theorem unicorn_stitches : ℕ → Prop :=
  fun (unicorn_stitches : ℕ) =>
    let stitches_per_minute : ℕ := 4
    let flower_stitches : ℕ := 60
    let godzilla_stitches : ℕ := 800
    let total_flowers : ℕ := 50
    let total_unicorns : ℕ := 3
    let total_minutes : ℕ := 1085
    let total_stitches : ℕ := total_minutes * stitches_per_minute
    let flower_and_godzilla_stitches : ℕ := total_flowers * flower_stitches + godzilla_stitches
    let remaining_stitches : ℕ := total_stitches - flower_and_godzilla_stitches
    remaining_stitches = total_unicorns * unicorn_stitches → unicorn_stitches = 180

/-- Proof of the theorem --/
theorem unicorn_stitches_proof : unicorn_stitches 180 :=
  by sorry

end unicorn_stitches_unicorn_stitches_proof_l5_535


namespace int_endomorphisms_characterization_l5_560

/-- An endomorphism of the additive group of integers -/
def IntEndomorphism : Type := ℤ → ℤ

/-- The homomorphism property for integer endomorphisms -/
def IsHomomorphism (φ : IntEndomorphism) : Prop :=
  ∀ a b : ℤ, φ (a + b) = φ a + φ b

/-- The set of all endomorphisms of the additive group of integers -/
def IntEndomorphisms : Set IntEndomorphism :=
  {φ : IntEndomorphism | IsHomomorphism φ}

/-- A linear function with integer coefficient -/
def LinearIntFunction (d : ℤ) : IntEndomorphism :=
  fun x => d * x

theorem int_endomorphisms_characterization :
  ∀ φ : IntEndomorphism, φ ∈ IntEndomorphisms ↔ ∃ d : ℤ, φ = LinearIntFunction d :=
by sorry

end int_endomorphisms_characterization_l5_560


namespace percentage_runs_from_running_approx_l5_500

def total_runs : ℕ := 120
def num_boundaries : ℕ := 5
def num_sixes : ℕ := 5
def runs_per_boundary : ℕ := 4
def runs_per_six : ℕ := 6

def runs_from_boundaries : ℕ := num_boundaries * runs_per_boundary
def runs_from_sixes : ℕ := num_sixes * runs_per_six
def runs_without_running : ℕ := runs_from_boundaries + runs_from_sixes
def runs_from_running : ℕ := total_runs - runs_without_running

theorem percentage_runs_from_running_approx (ε : ℚ) (h : ε > 0) :
  ∃ (p : ℚ), abs (p - (runs_from_running : ℚ) / (total_runs : ℚ) * 100) < ε ∧ 
             abs (p - 58.33) < ε :=
sorry

end percentage_runs_from_running_approx_l5_500


namespace ellipse_problem_l5_540

-- Define the ellipse C
def ellipse (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the line l
def line (x y k : ℝ) : Prop := y = k * (x - 1)

-- Define the theorem
theorem ellipse_problem (a b c k : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : k > 0) :
  -- Condition 2: C passes through Q(√2, 1)
  ellipse (Real.sqrt 2) 1 a b →
  -- Condition 3: Right focus at F(√2, 0)
  c = Real.sqrt 2 →
  a^2 - b^2 = c^2 →
  -- Condition 6: CN = MD (implicitly used in the solution)
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    ellipse x₁ y₁ a b ∧ 
    ellipse x₂ y₂ a b ∧
    line x₁ y₁ k ∧ 
    line x₂ y₂ k ∧
    x₂ - 1 = -x₁ ∧ 
    y₂ = -k - y₁) →
  -- Conclusion I: Equation of ellipse C
  (a = 2 ∧ b = Real.sqrt 2) ∧
  -- Conclusion II: Value of k and length of MN
  (k = Real.sqrt 2 / 2 ∧ 
   ∃ x₁ x₂ : ℝ, 
     ellipse x₁ (k * (x₁ - 1)) 2 (Real.sqrt 2) ∧
     ellipse x₂ (k * (x₂ - 1)) 2 (Real.sqrt 2) ∧
     Real.sqrt ((x₂ - x₁)^2 + (k * (x₂ - x₁))^2) = Real.sqrt 42 / 2) :=
by sorry

end ellipse_problem_l5_540


namespace abs_minus_one_lt_two_iff_product_lt_zero_l5_573

theorem abs_minus_one_lt_two_iff_product_lt_zero (x : ℝ) :
  |x - 1| < 2 ↔ (x + 1) * (x - 3) < 0 := by sorry

end abs_minus_one_lt_two_iff_product_lt_zero_l5_573


namespace tetrahedron_edge_length_l5_507

/-- A regular tetrahedron with one vertex on the axis of a cylinder and the other three vertices on the lateral surface of the cylinder. -/
structure TetrahedronInCylinder where
  R : ℝ  -- Radius of the cylinder's base
  edge_length : ℝ  -- Edge length of the tetrahedron

/-- The edge length of the tetrahedron is either R√3 or (R√11)/3. -/
theorem tetrahedron_edge_length (t : TetrahedronInCylinder) :
  t.edge_length = t.R * Real.sqrt 3 ∨ t.edge_length = t.R * Real.sqrt 11 / 3 := by
  sorry

end tetrahedron_edge_length_l5_507


namespace frood_game_theorem_l5_585

/-- Score for dropping n froods -/
def drop_score (n : ℕ) : ℕ := n * (n + 1)

/-- Score for eating n froods -/
def eat_score (n : ℕ) : ℕ := 8 * n

/-- The least number of froods for which dropping them earns more points than eating them -/
def least_frood_number : ℕ := 8

theorem frood_game_theorem :
  least_frood_number = 8 ∧
  ∀ n : ℕ, n < least_frood_number → drop_score n ≤ eat_score n ∧
  drop_score least_frood_number > eat_score least_frood_number :=
by sorry

end frood_game_theorem_l5_585


namespace player_percentage_of_team_points_l5_503

def three_point_goals : ℕ := 5
def two_point_goals : ℕ := 10
def team_total_points : ℕ := 70

def player_points : ℕ := three_point_goals * 3 + two_point_goals * 2

theorem player_percentage_of_team_points :
  (player_points : ℚ) / team_total_points * 100 = 50 := by
  sorry

end player_percentage_of_team_points_l5_503


namespace function_shift_l5_537

noncomputable def f (x : ℝ) (φ : ℝ) := Real.sin (1/2 * x + φ)

theorem function_shift (φ : ℝ) (h1 : |φ| < Real.pi/2) 
  (h2 : ∀ x, f x φ = f (Real.pi/3 - x) φ) :
  ∀ x, f (x + Real.pi/3) φ = Real.cos (1/2 * x) := by
sorry

end function_shift_l5_537


namespace owen_sleep_hours_l5_534

theorem owen_sleep_hours (total_hours work_hours chore_hours sleep_hours : ℕ) :
  total_hours = 24 ∧ work_hours = 6 ∧ chore_hours = 7 ∧ 
  sleep_hours = total_hours - (work_hours + chore_hours) →
  sleep_hours = 11 := by
  sorry

end owen_sleep_hours_l5_534


namespace cricketer_average_score_l5_596

theorem cricketer_average_score 
  (total_matches : Nat) 
  (matches_with_known_average : Nat) 
  (known_average : ℝ) 
  (total_average : ℝ) 
  (h1 : total_matches = 5)
  (h2 : matches_with_known_average = 3)
  (h3 : known_average = 10)
  (h4 : total_average = 22) :
  let remaining_matches := total_matches - matches_with_known_average
  let remaining_average := (total_matches * total_average - matches_with_known_average * known_average) / remaining_matches
  remaining_average = 40 := by
sorry

end cricketer_average_score_l5_596


namespace nested_sum_equals_geometric_sum_l5_551

def nested_sum : ℕ → ℕ
  | 0 => 5
  | n + 1 => 5 * (1 + nested_sum n)

theorem nested_sum_equals_geometric_sum : nested_sum 11 = 305175780 := by
  sorry

end nested_sum_equals_geometric_sum_l5_551


namespace drain_time_to_half_l5_566

/-- Represents the remaining water volume in the pool after draining for a given time. -/
def remaining_water (t : ℝ) : ℝ := 300 - 25 * t

/-- Proves that it takes 6 hours to drain the pool from 300 m³ to 150 m³. -/
theorem drain_time_to_half : ∃ t : ℝ, t = 6 ∧ remaining_water t = 150 := by
  sorry

end drain_time_to_half_l5_566


namespace herman_feeding_months_l5_536

/-- The number of months Herman feeds the birds -/
def feeding_months (cups_per_day : ℚ) (total_cups : ℚ) (days_per_month : ℚ) : ℚ :=
  (total_cups / cups_per_day) / days_per_month

theorem herman_feeding_months :
  feeding_months 1 90 30 = 3 := by
  sorry

end herman_feeding_months_l5_536


namespace opposite_unit_vector_l5_515

def a : ℝ × ℝ := (4, 3)

theorem opposite_unit_vector (a : ℝ × ℝ) :
  let magnitude := Real.sqrt (a.1^2 + a.2^2)
  let opposite_unit := (-a.1 / magnitude, -a.2 / magnitude)
  opposite_unit = (-4/5, -3/5) ∧
  opposite_unit.1^2 + opposite_unit.2^2 = 1 ∧
  a.1 * opposite_unit.1 + a.2 * opposite_unit.2 < 0 :=
by sorry

end opposite_unit_vector_l5_515


namespace jessica_bank_balance_l5_572

theorem jessica_bank_balance (B : ℝ) 
  (h1 : B * (2/5) = 200)  -- Condition: 2/5 of initial balance equals $200
  (h2 : B > 200)          -- Implicit condition: initial balance is greater than withdrawal
  : B - 200 + (B - 200) / 2 = 450 := by
  sorry

end jessica_bank_balance_l5_572


namespace f_extrema_l5_518

open Real

noncomputable def a (x : ℝ) : ℝ × ℝ := (cos (3*x/2), sin (3*x/2))
noncomputable def b (x : ℝ) : ℝ × ℝ := (cos (x/2), -sin (x/2))

noncomputable def f (x : ℝ) : ℝ := 
  (a x).1 * (b x).1 + (a x).2 * (b x).2 - 
  Real.sqrt ((a x).1 + (b x).1)^2 + ((a x).2 + (b x).2)^2

theorem f_extrema :
  ∀ x ∈ Set.Icc (-π/3) (π/4),
    (∀ y ∈ Set.Icc (-π/3) (π/4), f y ≤ -1) ∧
    (∃ y ∈ Set.Icc (-π/3) (π/4), f y = -1) ∧
    (∀ y ∈ Set.Icc (-π/3) (π/4), f y ≥ -Real.sqrt 2) ∧
    (∃ y ∈ Set.Icc (-π/3) (π/4), f y = -Real.sqrt 2) :=
by sorry

end f_extrema_l5_518


namespace candidates_per_state_l5_564

theorem candidates_per_state : 
  ∀ x : ℝ,
  (x * 0.07 = x * 0.06 + 79) →
  x = 7900 := by
sorry

end candidates_per_state_l5_564


namespace quadratic_vertex_form_sum_l5_597

/-- The quadratic function f(x) = 4x^2 - 8x + 6 -/
def f (x : ℝ) : ℝ := 4 * x^2 - 8 * x + 6

/-- The vertex form of the quadratic function -/
def vertex_form (a h k : ℝ) (x : ℝ) : ℝ := a * (x - h)^2 + k

theorem quadratic_vertex_form_sum :
  ∃ (a h k : ℝ), (∀ x, f x = vertex_form a h k x) ∧ (a + h + k = 7) := by
sorry

end quadratic_vertex_form_sum_l5_597


namespace two_from_three_permutations_l5_510

/-- The number of permutations of k items chosen from n items. -/
def permutations (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial (n - k))

/-- Theorem: There are 6 ways to choose and line up 2 people from a group of 3. -/
theorem two_from_three_permutations :
  permutations 3 2 = 6 := by
  sorry

end two_from_three_permutations_l5_510


namespace quadratic_equation_properties_l5_557

theorem quadratic_equation_properties (a b : ℝ) (h1 : a > 0) 
  (h2 : ∃! x y : ℝ, x ≠ y ∧ x^2 + a*x + b = 0 ∧ y^2 + a*y + b = 0) :
  (a^2 - b^2 ≤ 4) ∧ 
  (a^2 + 1/b ≥ 4) ∧
  (∀ c x₁ x₂ : ℝ, (x₁^2 + a*x₁ + b < c ∧ x₂^2 + a*x₂ + b < c ∧ |x₁ - x₂| = 4) → c = 4) :=
by sorry

end quadratic_equation_properties_l5_557


namespace stacy_bought_two_packs_l5_525

/-- The number of sheets per pack of printer paper -/
def sheets_per_pack : ℕ := 240

/-- The number of sheets used per day -/
def sheets_per_day : ℕ := 80

/-- The number of days the paper lasts -/
def days_lasted : ℕ := 6

/-- The number of packs of printer paper Stacy bought -/
def packs_bought : ℕ := (sheets_per_day * days_lasted) / sheets_per_pack

theorem stacy_bought_two_packs : packs_bought = 2 := by
  sorry

end stacy_bought_two_packs_l5_525


namespace root_product_theorem_l5_595

theorem root_product_theorem (a b : ℂ) : 
  a ≠ b →
  a^4 + a^3 - 1 = 0 →
  b^4 + b^3 - 1 = 0 →
  (a*b)^6 + (a*b)^4 + (a*b)^3 - (a*b)^2 - 1 = 0 :=
by sorry

end root_product_theorem_l5_595


namespace isabel_paper_problem_l5_532

theorem isabel_paper_problem (total : ℕ) (used : ℕ) (remaining : ℕ) : 
  total = 900 → used = 156 → remaining = total - used → remaining = 744 := by
  sorry

end isabel_paper_problem_l5_532


namespace nested_inverse_expression_l5_575

theorem nested_inverse_expression : ((((3+2)⁻¹ - 1)⁻¹ - 1)⁻¹ - 1 : ℚ) = -13/9 := by
  sorry

end nested_inverse_expression_l5_575


namespace onion_chopping_difference_l5_528

/-- Represents the rate of chopping onions in terms of number of onions and time in minutes -/
structure ChoppingRate where
  onions : ℕ
  minutes : ℕ

/-- Calculates the number of onions chopped in a given time based on a chopping rate -/
def chop_onions (rate : ChoppingRate) (time : ℕ) : ℕ :=
  (rate.onions * time) / rate.minutes

theorem onion_chopping_difference :
  let brittney_rate : ChoppingRate := ⟨15, 5⟩
  let carl_rate : ChoppingRate := ⟨20, 5⟩
  let time : ℕ := 30
  chop_onions carl_rate time - chop_onions brittney_rate time = 30 := by
  sorry

end onion_chopping_difference_l5_528


namespace largest_negative_integer_congruence_l5_546

theorem largest_negative_integer_congruence :
  ∃ (x : ℤ), x = -6 ∧
  (34 * x + 6) % 20 = 2 % 20 ∧
  ∀ (y : ℤ), y < 0 → (34 * y + 6) % 20 = 2 % 20 → y ≤ x :=
by sorry

end largest_negative_integer_congruence_l5_546


namespace xiaoMingCarbonEmissions_l5_553

/-- The carbon dioxide emissions formula for household tap water usage -/
def carbonEmissions (x : ℝ) : ℝ := 0.9 * x

/-- Xiao Ming's household tap water usage in tons -/
def xiaoMingWaterUsage : ℝ := 10

theorem xiaoMingCarbonEmissions :
  carbonEmissions xiaoMingWaterUsage = 9 := by
  sorry

end xiaoMingCarbonEmissions_l5_553


namespace surface_area_difference_l5_574

theorem surface_area_difference (large_cube_volume : ℝ) (small_cube_volume : ℝ) (num_small_cubes : ℕ) :
  large_cube_volume = 64 →
  small_cube_volume = 1 →
  num_small_cubes = 64 →
  (num_small_cubes : ℝ) * (6 * small_cube_volume ^ (2/3)) - (6 * large_cube_volume ^ (2/3)) = 288 := by
  sorry

end surface_area_difference_l5_574


namespace solve_for_c_l5_554

theorem solve_for_c (y : ℝ) (h1 : y > 0) : 
  ∃ c : ℝ, (7 * y) / 20 + (c * y) / 10 = 0.6499999999999999 * y ∧ c = 3 := by
  sorry

end solve_for_c_l5_554


namespace right_triangle_shortest_leg_l5_589

theorem right_triangle_shortest_leg : ∃ (a b : ℕ),
  a < b ∧ a^2 + b^2 = 65^2 ∧ ∀ (x y : ℕ), x < y ∧ x^2 + y^2 = 65^2 → a ≤ x :=
by sorry

end right_triangle_shortest_leg_l5_589


namespace complex_number_magnitude_squared_l5_588

theorem complex_number_magnitude_squared (z : ℂ) (h : z^2 + Complex.abs z ^ 2 = 8 - 2*I) : 
  Complex.abs z ^ 2 = 17/4 := by sorry

end complex_number_magnitude_squared_l5_588


namespace perfect_square_trinomial_l5_513

theorem perfect_square_trinomial (b : ℝ) : 
  (∃ k : ℝ, ∀ x : ℝ, x^2 + 8*x + b = (x + k)^2) → b = 16 := by
  sorry

end perfect_square_trinomial_l5_513


namespace work_completion_time_l5_512

/-- The number of days it takes for person A to complete the work alone -/
def days_A : ℝ := 15

/-- The fraction of work completed by A and B together in 5 days -/
def work_completed : ℝ := 0.5

/-- The number of days A and B work together -/
def days_together : ℝ := 5

/-- The number of days it takes for person B to complete the work alone -/
def days_B : ℝ := 30

theorem work_completion_time :
  (1 / days_A + 1 / days_B) * days_together = work_completed := by
  sorry

end work_completion_time_l5_512


namespace caroline_lassi_production_caroline_lassi_production_proof_l5_587

/-- Given that Caroline can make 7 lassis from 3 mangoes, 
    prove that she can make 35 lassis from 15 mangoes. -/
theorem caroline_lassi_production : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun mangoes_small lassis_small mangoes_large lassis_large =>
    mangoes_small = 3 ∧ 
    lassis_small = 7 ∧ 
    mangoes_large = 15 ∧
    lassis_large = 35 ∧
    (mangoes_large * lassis_small = mangoes_small * lassis_large) →
    lassis_large = (mangoes_large * lassis_small) / mangoes_small

theorem caroline_lassi_production_proof : 
  caroline_lassi_production 3 7 15 35 := by
  sorry

end caroline_lassi_production_caroline_lassi_production_proof_l5_587


namespace largest_valid_three_digit_number_l5_568

/-- Represents a three-digit number as a tuple of its digits -/
def ThreeDigitNumber := (Nat × Nat × Nat)

/-- Converts a ThreeDigitNumber to its numerical value -/
def toNumber (n : ThreeDigitNumber) : Nat :=
  100 * n.1 + 10 * n.2.1 + n.2.2

/-- Calculates the sum of digits of a ThreeDigitNumber -/
def digitSum (n : ThreeDigitNumber) : Nat :=
  n.1 + n.2.1 + n.2.2

/-- Checks if a ThreeDigitNumber satisfies all conditions -/
def isValid (n : ThreeDigitNumber) : Prop :=
  n.1 ≠ 0 ∧  -- Ensures it's a three-digit number
  n.1 = n.2.2 ∧  -- First digit matches third digit
  n.1 ≠ n.2.1 ∧  -- First digit doesn't match second digit
  (toNumber n) % (digitSum n) = 0  -- Number is divisible by sum of its digits

theorem largest_valid_three_digit_number :
  ∀ n : ThreeDigitNumber, isValid n → toNumber n ≤ 828 :=
by sorry

end largest_valid_three_digit_number_l5_568


namespace polynomial_equation_sum_l5_556

theorem polynomial_equation_sum (a b : ℤ) : 
  (∀ x : ℝ, 2 * x^3 - a * x^2 - 5 * x + 5 = (2 * x^2 + a * x - 1) * (x - b) + 3) → 
  a + b = 4 := by
  sorry

end polynomial_equation_sum_l5_556


namespace total_students_l5_502

theorem total_students (absent_percentage : ℝ) (present_students : ℕ) 
  (h1 : absent_percentage = 14) 
  (h2 : present_students = 86) : 
  ↑present_students / (1 - absent_percentage / 100) = 100 := by
  sorry

end total_students_l5_502
