import Mathlib

namespace NUMINAMATH_CALUDE_dal_gain_is_104_l2716_271667

/-- Calculates the total gain from selling a mixture of dals -/
def calculate_dal_gain (dal_a_kg : ℝ) (dal_a_rate : ℝ) (dal_b_kg : ℝ) (dal_b_rate : ℝ)
                       (dal_c_kg : ℝ) (dal_c_rate : ℝ) (dal_d_kg : ℝ) (dal_d_rate : ℝ)
                       (mixture_rate : ℝ) : ℝ :=
  let total_cost := dal_a_kg * dal_a_rate + dal_b_kg * dal_b_rate +
                    dal_c_kg * dal_c_rate + dal_d_kg * dal_d_rate
  let total_weight := dal_a_kg + dal_b_kg + dal_c_kg + dal_d_kg
  let total_revenue := total_weight * mixture_rate
  total_revenue - total_cost

theorem dal_gain_is_104 :
  calculate_dal_gain 15 14.5 10 13 12 16 8 18 17.5 = 104 := by
  sorry

end NUMINAMATH_CALUDE_dal_gain_is_104_l2716_271667


namespace NUMINAMATH_CALUDE_five_student_committee_from_eight_l2716_271616

theorem five_student_committee_from_eight (n k : ℕ) : n = 8 → k = 5 → Nat.choose n k = 56 := by
  sorry

end NUMINAMATH_CALUDE_five_student_committee_from_eight_l2716_271616


namespace NUMINAMATH_CALUDE_ob_value_l2716_271635

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the points
variable (O F₁ F₂ A B : ℝ × ℝ)

-- State the conditions
variable (h1 : O = (0, 0))
variable (h2 : ellipse (F₁.1) (F₁.2))
variable (h3 : ellipse (F₂.1) (F₂.2))
variable (h4 : F₁.1 < 0 ∧ F₂.1 > 0)
variable (h5 : ellipse A.1 A.2)
variable (h6 : (A.1 - F₂.1) * (F₂.1 - F₁.1) + (A.2 - F₂.2) * (F₂.2 - F₁.2) = 0)
variable (h7 : B.1 = 0)
variable (h8 : ∃ t : ℝ, B = t • F₁ + (1 - t) • A)

-- State the theorem
theorem ob_value : abs B.2 = 3/4 := by sorry

end NUMINAMATH_CALUDE_ob_value_l2716_271635


namespace NUMINAMATH_CALUDE_dans_remaining_money_dans_remaining_money_proof_l2716_271622

/-- Calculates the remaining money after purchases and tax --/
theorem dans_remaining_money (initial_amount : ℚ) 
  (candy_price : ℚ) (candy_count : ℕ) 
  (gum_price : ℚ) (soda_price : ℚ) (tax_rate : ℚ) : ℚ :=
  let total_before_tax := candy_price * candy_count + gum_price + soda_price
  let total_tax := total_before_tax * tax_rate
  let total_cost := total_before_tax + total_tax
  initial_amount - total_cost

/-- Proves that Dan's remaining money is $40.98 --/
theorem dans_remaining_money_proof :
  dans_remaining_money 50 1.75 3 0.85 2.25 0.08 = 40.98 := by
  sorry

end NUMINAMATH_CALUDE_dans_remaining_money_dans_remaining_money_proof_l2716_271622


namespace NUMINAMATH_CALUDE_derivative_of_y_at_2_l2716_271632

-- Define the function y = 3x
def y (x : ℝ) : ℝ := 3 * x

-- State the theorem
theorem derivative_of_y_at_2 :
  deriv y 2 = 3 := by sorry

end NUMINAMATH_CALUDE_derivative_of_y_at_2_l2716_271632


namespace NUMINAMATH_CALUDE_sum_two_smallest_trite_numbers_l2716_271642

def is_trite (n : ℕ+) : Prop :=
  ∃ (d : Fin 12 → ℕ+),
    (∀ i j, i < j → d i < d j) ∧
    d 0 = 1 ∧
    d 11 = n ∧
    (∀ k, k ∣ n ↔ ∃ i, d i = k) ∧
    5 + (d 5) * (d 5 + d 3) = (d 6) * (d 3)

theorem sum_two_smallest_trite_numbers : 
  ∃ (a b : ℕ+), is_trite a ∧ is_trite b ∧ 
  (∀ n : ℕ+, is_trite n → a ≤ n) ∧
  (∀ n : ℕ+, is_trite n ∧ n ≠ a → b ≤ n) ∧
  a + b = 151127 :=
sorry

end NUMINAMATH_CALUDE_sum_two_smallest_trite_numbers_l2716_271642


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_one_l2716_271656

theorem fraction_zero_implies_x_equals_one (x : ℝ) :
  x ≠ -1 →
  (x^2 - 1) / (x + 1) = 0 →
  x = 1 := by
sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_one_l2716_271656


namespace NUMINAMATH_CALUDE_oblique_view_isosceles_implies_right_trapezoid_l2716_271657

/-- A plane figure. -/
structure PlaneFigure where
  -- Add necessary fields here
  mk :: -- Constructor

/-- An oblique view of a plane figure. -/
structure ObliqueView where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Represents that a figure is an isosceles trapezoid. -/
def is_isosceles_trapezoid (f : ObliqueView) : Prop :=
  sorry

/-- Represents the base angle of a figure. -/
def base_angle (f : ObliqueView) : ℝ :=
  sorry

/-- Represents that a figure is a right trapezoid. -/
def is_right_trapezoid (f : PlaneFigure) : Prop :=
  sorry

/-- 
If the oblique view of a plane figure is an isosceles trapezoid 
with a base angle of 45°, then the original figure is a right trapezoid.
-/
theorem oblique_view_isosceles_implies_right_trapezoid 
  (f : PlaneFigure) (v : ObliqueView) :
  is_isosceles_trapezoid v → base_angle v = 45 → is_right_trapezoid f :=
by
  sorry

end NUMINAMATH_CALUDE_oblique_view_isosceles_implies_right_trapezoid_l2716_271657


namespace NUMINAMATH_CALUDE_smallest_solution_of_equation_l2716_271690

theorem smallest_solution_of_equation :
  ∃ (x : ℝ), x = -3 ∧ 
  (3 * x) / (x + 3) + (3 * x^2 - 18) / x = 9 ∧
  (∀ y : ℝ, (3 * y) / (y + 3) + (3 * y^2 - 18) / y = 9 → y ≥ x) := by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_of_equation_l2716_271690


namespace NUMINAMATH_CALUDE_problem_solution_l2716_271670

def A : Set ℝ := {x | (2*x - 2)/(x + 1) < 1}

def B (a : ℝ) : Set ℝ := {x | x^2 + x + a - a^2 < 0}

theorem problem_solution :
  (∀ x, x ∈ (B 1 ∪ (Set.univ \ A)) ↔ (x < 0 ∨ x ≥ 3)) ∧
  (∀ a, A = B a ↔ a ∈ Set.Iic (-3) ∪ Set.Ici 4) := by sorry

end NUMINAMATH_CALUDE_problem_solution_l2716_271670


namespace NUMINAMATH_CALUDE_sum_of_a_values_for_single_solution_l2716_271646

theorem sum_of_a_values_for_single_solution (a : ℝ) : 
  let equation := fun (x : ℝ) => 2 * x^2 + a * x + 6 * x + 7
  let discriminant := (a + 6)^2 - 4 * 2 * 7
  let sum_of_a_values := -(12 : ℝ)
  (∃ (a₁ a₂ : ℝ), 
    (∀ x, equation x = 0 → discriminant = 0) ∧ 
    (a₁ ≠ a₂) ∧ 
    (a₁ + a₂ = sum_of_a_values)) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_a_values_for_single_solution_l2716_271646


namespace NUMINAMATH_CALUDE_tim_works_six_days_l2716_271645

/-- Represents Tim's work schedule and earnings --/
structure TimsWork where
  tasks_per_day : ℕ
  pay_per_task : ℚ
  weekly_earnings : ℚ

/-- Calculates the number of days Tim works per week --/
def days_worked (w : TimsWork) : ℚ :=
  w.weekly_earnings / (w.tasks_per_day * w.pay_per_task)

/-- Theorem stating that Tim works 6 days a week --/
theorem tim_works_six_days (w : TimsWork) 
  (h1 : w.tasks_per_day = 100)
  (h2 : w.pay_per_task = 6/5) -- $1.2 represented as a fraction
  (h3 : w.weekly_earnings = 720) :
  days_worked w = 6 := by
  sorry

#eval days_worked { tasks_per_day := 100, pay_per_task := 6/5, weekly_earnings := 720 }

end NUMINAMATH_CALUDE_tim_works_six_days_l2716_271645


namespace NUMINAMATH_CALUDE_min_throws_for_three_occurrences_l2716_271680

/-- The number of sides on each die -/
def sides : ℕ := 6

/-- The number of dice thrown each time -/
def num_dice : ℕ := 4

/-- The minimum possible sum when rolling the dice -/
def min_sum : ℕ := num_dice

/-- The maximum possible sum when rolling the dice -/
def max_sum : ℕ := num_dice * sides

/-- The number of possible different sums -/
def num_sums : ℕ := max_sum - min_sum + 1

/-- The minimum number of throws required -/
def min_throws : ℕ := num_sums * 2 + 1

theorem min_throws_for_three_occurrences :
  min_throws = 43 :=
sorry

end NUMINAMATH_CALUDE_min_throws_for_three_occurrences_l2716_271680


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_implies_divisibility_l2716_271605

theorem gcd_lcm_sum_implies_divisibility (a b : ℤ) 
  (h : Nat.gcd a.natAbs b.natAbs + Nat.lcm a.natAbs b.natAbs = a.natAbs + b.natAbs) : 
  a ∣ b ∨ b ∣ a := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_implies_divisibility_l2716_271605


namespace NUMINAMATH_CALUDE_green_ball_probability_l2716_271687

-- Define the containers and their contents
def container_A : Nat × Nat := (3, 7)  -- (red, green)
def container_B : Nat × Nat := (5, 5)
def container_C : Nat × Nat := (5, 5)

-- Define the probability of selecting each container
def container_prob : Rat := 1/3

-- Define the probability of selecting a green ball from each container
def green_prob_A : Rat := container_prob * (container_A.2 / (container_A.1 + container_A.2))
def green_prob_B : Rat := container_prob * (container_B.2 / (container_B.1 + container_B.2))
def green_prob_C : Rat := container_prob * (container_C.2 / (container_C.1 + container_C.2))

-- Theorem: The probability of selecting a green ball is 17/30
theorem green_ball_probability : 
  green_prob_A + green_prob_B + green_prob_C = 17/30 := by
  sorry

end NUMINAMATH_CALUDE_green_ball_probability_l2716_271687


namespace NUMINAMATH_CALUDE_min_shots_to_hit_ship_l2716_271613

/-- Represents a point on the game board -/
structure Point where
  x : Fin 10
  y : Fin 10

/-- Represents a ship on the game board -/
inductive Ship
  | Horizontal : Fin 10 → Fin 7 → Ship
  | Vertical : Fin 7 → Fin 10 → Ship

/-- Checks if a point is on a ship -/
def pointOnShip (p : Point) (s : Ship) : Prop :=
  match s with
  | Ship.Horizontal row col => p.y = row ∧ col ≤ p.x ∧ p.x < col + 4
  | Ship.Vertical row col => p.x = col ∧ row ≤ p.y ∧ p.y < row + 4

/-- The theorem to be proved -/
theorem min_shots_to_hit_ship :
  ∃ (shots : Finset Point),
    shots.card = 14 ∧
    ∀ (s : Ship), ∃ (p : Point), p ∈ shots ∧ pointOnShip p s ∧
    ∀ (shots' : Finset Point),
      shots'.card < 14 →
      ∃ (s : Ship), ∀ (p : Point), p ∈ shots' → ¬pointOnShip p s :=
by sorry

end NUMINAMATH_CALUDE_min_shots_to_hit_ship_l2716_271613


namespace NUMINAMATH_CALUDE_tan_x_eq_2_implies_expression_l2716_271682

theorem tan_x_eq_2_implies_expression (x : ℝ) (h : Real.tan x = 2) :
  (6 * Real.sin (2 * x) + 2 * Real.cos (2 * x)) / (Real.cos (2 * x) - 3 * Real.sin (2 * x)) = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_tan_x_eq_2_implies_expression_l2716_271682


namespace NUMINAMATH_CALUDE_city_parking_fee_l2716_271621

def weekly_salary : ℚ := 450
def federal_tax_rate : ℚ := 1/3
def state_tax_rate : ℚ := 8/100
def health_insurance : ℚ := 50
def life_insurance : ℚ := 20
def final_paycheck : ℚ := 184

theorem city_parking_fee :
  let after_federal := weekly_salary * (1 - federal_tax_rate)
  let after_state := after_federal * (1 - state_tax_rate)
  let after_insurance := after_state - health_insurance - life_insurance
  after_insurance - final_paycheck = 22 := by sorry

end NUMINAMATH_CALUDE_city_parking_fee_l2716_271621


namespace NUMINAMATH_CALUDE_range_of_a_l2716_271677

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 + a*x + 1 ≥ 0) → -2 ≤ a ∧ a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2716_271677


namespace NUMINAMATH_CALUDE_positive_expressions_l2716_271691

theorem positive_expressions (a b c : ℝ) 
  (ha : 0 < a ∧ a < 2) 
  (hb : -2 < b ∧ b < 0) 
  (hc : 0 < c ∧ c < 3) : 
  0 < b + b^2 ∧ 0 < b + 3*b^2 := by
  sorry

end NUMINAMATH_CALUDE_positive_expressions_l2716_271691


namespace NUMINAMATH_CALUDE_staircase_perimeter_l2716_271624

/-- Represents a staircase-shaped region -/
structure StaircaseRegion where
  congruent_sides : ℕ
  rectangle_length : ℝ
  area : ℝ

/-- The perimeter of a staircase-shaped region -/
def perimeter (region : StaircaseRegion) : ℝ :=
  sorry

/-- Theorem stating the perimeter of the specific staircase region -/
theorem staircase_perimeter : 
  ∀ (region : StaircaseRegion), 
    region.congruent_sides = 12 ∧ 
    region.rectangle_length = 12 ∧ 
    region.area = 85 → 
    perimeter region = 41 :=
by sorry

end NUMINAMATH_CALUDE_staircase_perimeter_l2716_271624


namespace NUMINAMATH_CALUDE_sum_of_cubes_special_case_l2716_271675

theorem sum_of_cubes_special_case (x y : ℝ) (h1 : x + y = 1) (h2 : x * y = 1) : 
  x^3 + y^3 = -2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_cubes_special_case_l2716_271675


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_special_set_l2716_271611

theorem arithmetic_mean_of_special_set (n : ℕ) (h : n > 1) :
  let s : Finset ℕ := Finset.range n
  let f : ℕ → ℝ := λ i => if i = 0 then 1/n + 2/n^2 else 1
  (s.sum f) / n = 1 + 2/n^2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_special_set_l2716_271611


namespace NUMINAMATH_CALUDE_log_equation_solution_l2716_271678

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  (Real.log x / Real.log 2) + 1 / (Real.log (x + 1) / Real.log 2) = 1 ↔ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l2716_271678


namespace NUMINAMATH_CALUDE_karen_wrong_answers_l2716_271623

/-- Represents the number of wrong answers for each person -/
structure TestResults where
  karen : ℕ
  leo : ℕ
  morgan : ℕ
  nora : ℕ

/-- The conditions of the problem -/
def satisfiesConditions (r : TestResults) : Prop :=
  r.karen + r.leo = r.morgan + r.nora ∧
  r.karen + r.nora = r.leo + r.morgan + 3 ∧
  r.morgan = 6

theorem karen_wrong_answers (r : TestResults) (h : satisfiesConditions r) : r.karen = 6 := by
  sorry

#check karen_wrong_answers

end NUMINAMATH_CALUDE_karen_wrong_answers_l2716_271623


namespace NUMINAMATH_CALUDE_age_sum_proof_l2716_271631

theorem age_sum_proof (A B C : ℕ) 
  (h1 : A = B + C + 16) 
  (h2 : A^2 = (B + C)^2 + 1632) : 
  A + B + C = 102 := by
sorry

end NUMINAMATH_CALUDE_age_sum_proof_l2716_271631


namespace NUMINAMATH_CALUDE_cattle_train_speed_calculation_l2716_271658

/-- The speed of the cattle train in miles per hour -/
def cattle_train_speed : ℝ := 93.33333333333333

/-- The speed of the diesel train in miles per hour -/
def diesel_train_speed (x : ℝ) : ℝ := x - 33

/-- The time difference between the trains' departures in hours -/
def time_difference : ℝ := 6

/-- The travel time of the diesel train in hours -/
def diesel_travel_time : ℝ := 12

/-- The total distance between the trains after the diesel train's travel -/
def total_distance : ℝ := 1284

theorem cattle_train_speed_calculation :
  time_difference * cattle_train_speed +
  diesel_travel_time * cattle_train_speed +
  diesel_travel_time * (diesel_train_speed cattle_train_speed) = total_distance := by
  sorry

end NUMINAMATH_CALUDE_cattle_train_speed_calculation_l2716_271658


namespace NUMINAMATH_CALUDE_probability_equals_three_fourteenths_l2716_271684

-- Define the number of red and blue marbles
def red_marbles : ℕ := 15
def blue_marbles : ℕ := 10

-- Define the total number of marbles
def total_marbles : ℕ := red_marbles + blue_marbles

-- Define the number of marbles to be selected
def selected_marbles : ℕ := 4

-- Define the function to calculate combinations
def combination (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- Define the probability of selecting two red and two blue marbles
def probability_two_red_two_blue : ℚ :=
  (6 * combination red_marbles 2 * combination blue_marbles 2) / (combination total_marbles selected_marbles)

-- Theorem statement
theorem probability_equals_three_fourteenths : 
  probability_two_red_two_blue = 3 / 14 := by sorry

end NUMINAMATH_CALUDE_probability_equals_three_fourteenths_l2716_271684


namespace NUMINAMATH_CALUDE_line_symmetry_l2716_271606

-- Define the original line
def original_line (x y : ℝ) : Prop := 2 * x - y + 4 = 0

-- Define the axis of symmetry
def axis_of_symmetry (x : ℝ) : Prop := x = 1

-- Define the symmetric line
def symmetric_line (x y : ℝ) : Prop := 2 * x + y - 8 = 0

-- Theorem statement
theorem line_symmetry :
  ∀ (x y : ℝ),
  (∃ (x₀ y₀ : ℝ), original_line x₀ y₀ ∧ axis_of_symmetry ((x + x₀) / 2)) →
  symmetric_line x y :=
sorry

end NUMINAMATH_CALUDE_line_symmetry_l2716_271606


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2716_271641

theorem sufficient_not_necessary (a : ℝ) : 
  (∀ x : ℝ, x > 1 → x < 1) ∧ 
  (∃ y : ℝ, y < 1 ∧ ¬(y > 1)) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2716_271641


namespace NUMINAMATH_CALUDE_prism_diagonals_l2716_271617

/-- Checks if three numbers can be the lengths of external diagonals of a right regular prism -/
def valid_diagonals (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a^2 + b^2 > c^2 ∧
  b^2 + c^2 > a^2 ∧
  a^2 + c^2 > b^2

theorem prism_diagonals :
  ¬(valid_diagonals 6 8 11) ∧
  (valid_diagonals 6 8 10) ∧
  (valid_diagonals 6 10 11) ∧
  (valid_diagonals 8 10 11) ∧
  (valid_diagonals 8 11 12) :=
by sorry

end NUMINAMATH_CALUDE_prism_diagonals_l2716_271617


namespace NUMINAMATH_CALUDE_snooker_tournament_ticket_difference_l2716_271626

theorem snooker_tournament_ticket_difference :
  ∀ (vip_tickets general_tickets : ℕ),
    vip_tickets + general_tickets = 320 →
    40 * vip_tickets + 10 * general_tickets = 7500 →
    general_tickets - vip_tickets = 34 := by
  sorry

end NUMINAMATH_CALUDE_snooker_tournament_ticket_difference_l2716_271626


namespace NUMINAMATH_CALUDE_cos_2alpha_minus_3pi_over_5_l2716_271674

theorem cos_2alpha_minus_3pi_over_5 (α : Real) 
  (h : Real.sin (α + π/5) = Real.sqrt 7 / 3) : 
  Real.cos (2*α - 3*π/5) = 5/9 := by
sorry

end NUMINAMATH_CALUDE_cos_2alpha_minus_3pi_over_5_l2716_271674


namespace NUMINAMATH_CALUDE_expected_occurrences_is_two_l2716_271630

/-- The probability of event A occurring -/
def prob_A : ℝ := 0.2

/-- The probability of event B occurring -/
def prob_B : ℝ := 0.4

/-- The number of trials -/
def num_trials : ℕ := 25

/-- The expected number of trials where both events occur simultaneously -/
def expected_occurrences : ℝ := num_trials * (prob_A * prob_B)

/-- Theorem stating that the expected number of occurrences is 2 -/
theorem expected_occurrences_is_two : expected_occurrences = 2 := by
  sorry

end NUMINAMATH_CALUDE_expected_occurrences_is_two_l2716_271630


namespace NUMINAMATH_CALUDE_product_xy_on_line_k_l2716_271604

/-- A line passing through the origin with slope 1/4 -/
def line_k (x y : ℝ) : Prop := y = (1/4) * x

theorem product_xy_on_line_k :
  ∀ x y : ℝ,
  line_k x 8 → line_k 20 y →
  x * y = 160 := by
sorry

end NUMINAMATH_CALUDE_product_xy_on_line_k_l2716_271604


namespace NUMINAMATH_CALUDE_person_peach_count_l2716_271600

theorem person_peach_count (jake_peaches jake_apples person_apples person_peaches : ℕ) : 
  jake_peaches + 6 = person_peaches →
  jake_apples = person_apples + 8 →
  person_apples = 16 →
  person_peaches = person_apples + 1 →
  person_peaches = 17 := by
sorry

end NUMINAMATH_CALUDE_person_peach_count_l2716_271600


namespace NUMINAMATH_CALUDE_regular_polygon_perimeter_l2716_271695

/-- A regular polygon with side length 7 and exterior angle 90 degrees has a perimeter of 28 units. -/
theorem regular_polygon_perimeter (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ) :
  n > 0 ∧ 
  side_length = 7 ∧ 
  exterior_angle = 90 ∧ 
  exterior_angle = 360 / n →
  n * side_length = 28 :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_perimeter_l2716_271695


namespace NUMINAMATH_CALUDE_road_construction_equation_l2716_271696

theorem road_construction_equation (x : ℝ) : 
  x > 0 →
  (9 : ℝ) / x - 12 / (x + 1) = (1 : ℝ) / 2 ↔
  (9 / x = 12 / (x + 1) + 1 / 2 ∧
   9 = x * (12 / (x + 1) + 1 / 2) ∧
   12 = (x + 1) * (9 / x - 1 / 2)) :=
by sorry


end NUMINAMATH_CALUDE_road_construction_equation_l2716_271696


namespace NUMINAMATH_CALUDE_tan_product_from_cosine_sum_l2716_271665

theorem tan_product_from_cosine_sum (α β : ℝ) 
  (h : 3 * Real.cos (2 * α + β) + 5 * Real.cos β = 0) : 
  Real.tan (α + β) * Real.tan α = -4 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_from_cosine_sum_l2716_271665


namespace NUMINAMATH_CALUDE_count_valid_integers_l2716_271693

/-- The set of available digits -/
def available_digits : Finset ℕ := {1, 4, 7}

/-- The count of each digit in the available set -/
def digit_count : ℕ → ℕ
  | 1 => 2
  | 4 => 3
  | 7 => 1
  | _ => 0

/-- A valid three-digit integer formed from the available digits -/
structure ValidInteger where
  hundreds : ℕ
  tens : ℕ
  ones : ℕ
  hundreds_in_set : hundreds ∈ available_digits
  tens_in_set : tens ∈ available_digits
  ones_in_set : ones ∈ available_digits
  valid_count : ∀ d ∈ available_digits,
    (if hundreds = d then 1 else 0) +
    (if tens = d then 1 else 0) +
    (if ones = d then 1 else 0) ≤ digit_count d

/-- The set of all valid three-digit integers -/
def valid_integers : Finset ValidInteger := sorry

theorem count_valid_integers :
  Finset.card valid_integers = 31 := by sorry

end NUMINAMATH_CALUDE_count_valid_integers_l2716_271693


namespace NUMINAMATH_CALUDE_waiter_income_fraction_l2716_271610

theorem waiter_income_fraction (salary tips income : ℚ) : 
  income = salary + tips → 
  tips = (5 : ℚ) / 3 * salary → 
  tips / income = (5 : ℚ) / 8 := by
sorry

end NUMINAMATH_CALUDE_waiter_income_fraction_l2716_271610


namespace NUMINAMATH_CALUDE_log_27_3_l2716_271649

theorem log_27_3 : Real.log 3 / Real.log 27 = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_log_27_3_l2716_271649


namespace NUMINAMATH_CALUDE_inequality_proof_l2716_271609

theorem inequality_proof (a b : Real) (ha : 0 < a ∧ a < Real.pi / 2) (hb : 0 < b ∧ b < Real.pi / 2) :
  (5 / Real.cos a ^ 2) + (5 / (Real.sin a ^ 2 * Real.sin b ^ 2 * Real.cos b ^ 2)) ≥ 27 * Real.cos a + 36 * Real.sin a := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2716_271609


namespace NUMINAMATH_CALUDE_work_completion_days_l2716_271673

/-- Represents the work scenario with initial workers and additional workers joining later. -/
structure WorkScenario where
  initial_workers : ℕ
  additional_workers : ℕ
  days_saved : ℕ

/-- Calculates the original number of days required to complete the work. -/
def original_days (scenario : WorkScenario) : ℕ :=
  2 * scenario.days_saved

/-- Theorem stating that for the given scenario, the original number of days is 6. -/
theorem work_completion_days (scenario : WorkScenario) 
  (h1 : scenario.initial_workers = 10)
  (h2 : scenario.additional_workers = 10)
  (h3 : scenario.days_saved = 3) :
  original_days scenario = 6 := by
  sorry

#eval original_days { initial_workers := 10, additional_workers := 10, days_saved := 3 }

end NUMINAMATH_CALUDE_work_completion_days_l2716_271673


namespace NUMINAMATH_CALUDE_smallest_product_l2716_271629

def digits : List Nat := [5, 6, 7, 8]

def is_valid_arrangement (a b c d : Nat) : Prop :=
  a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

def product (a b c d : Nat) : Nat := (10 * a + b) * (10 * c + d)

theorem smallest_product :
  ∀ a b c d : Nat, is_valid_arrangement a b c d →
    product a b c d ≥ 3876 :=
by sorry

end NUMINAMATH_CALUDE_smallest_product_l2716_271629


namespace NUMINAMATH_CALUDE_jessica_attended_two_games_l2716_271602

/-- The number of soccer games Jessica attended -/
def games_attended (total_games missed_games : ℕ) : ℕ :=
  total_games - missed_games

/-- Proof that Jessica attended 2 games -/
theorem jessica_attended_two_games :
  games_attended 6 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_jessica_attended_two_games_l2716_271602


namespace NUMINAMATH_CALUDE_larger_number_proof_l2716_271685

theorem larger_number_proof (a b : ℕ+) : 
  (Nat.gcd a b = 20) →
  (Nat.lcm a b = 3640) →
  (13 ∣ Nat.lcm a b) →
  (14 ∣ Nat.lcm a b) →
  max a b = 280 := by
sorry

end NUMINAMATH_CALUDE_larger_number_proof_l2716_271685


namespace NUMINAMATH_CALUDE_minimum_travel_time_l2716_271654

/-- The minimum time for a person to travel from point A to point B -/
theorem minimum_travel_time (BC : ℝ) (angle_BAC : ℝ) (swimming_speed : ℝ) 
  (h1 : BC = 30)
  (h2 : angle_BAC = 15 * π / 180)
  (h3 : swimming_speed = 3) :
  ∃ t : ℝ, t = 20 ∧ 
  ∀ t' : ℝ, t' ≥ t ∧ 
  ∃ d : ℝ, t' = d / (swimming_speed * Real.sqrt 2) + Real.sqrt (d^2 - BC^2) / swimming_speed :=
by sorry

end NUMINAMATH_CALUDE_minimum_travel_time_l2716_271654


namespace NUMINAMATH_CALUDE_min_value_product_quotient_l2716_271676

theorem min_value_product_quotient (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  ((x^2 + 4*x + 1) * (y^2 + 4*y + 1) * (z^2 + 4*z + 1)) / (x*y*z) ≥ 216 ∧
  ((1^2 + 4*1 + 1) * (1^2 + 4*1 + 1) * (1^2 + 4*1 + 1)) / (1*1*1) = 216 := by
  sorry

end NUMINAMATH_CALUDE_min_value_product_quotient_l2716_271676


namespace NUMINAMATH_CALUDE_spencer_total_distance_l2716_271660

/-- The total distance Spencer walked on Saturday -/
def total_distance (house_to_library library_to_post_office post_office_to_house : ℝ) : ℝ :=
  house_to_library + library_to_post_office + post_office_to_house

/-- Theorem stating that Spencer walked 0.8 mile in total -/
theorem spencer_total_distance :
  total_distance 0.3 0.1 0.4 = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_spencer_total_distance_l2716_271660


namespace NUMINAMATH_CALUDE_total_miles_is_35_l2716_271652

def andrew_daily_miles : ℕ := 2
def peter_extra_miles : ℕ := 3
def days : ℕ := 5

def total_miles : ℕ := 
  (andrew_daily_miles * days) + ((andrew_daily_miles + peter_extra_miles) * days)

theorem total_miles_is_35 : total_miles = 35 := by
  sorry

end NUMINAMATH_CALUDE_total_miles_is_35_l2716_271652


namespace NUMINAMATH_CALUDE_inverse_f_123_l2716_271686

noncomputable def f (x : ℝ) : ℝ := 3 * x^3 + 6

theorem inverse_f_123 : f⁻¹ 123 = (39 : ℝ)^(1/3) := by sorry

end NUMINAMATH_CALUDE_inverse_f_123_l2716_271686


namespace NUMINAMATH_CALUDE_train_length_calculation_l2716_271625

/-- Calculates the length of a train given its speed, time to cross a bridge, and the bridge length. -/
theorem train_length_calculation (train_speed : ℝ) (crossing_time : ℝ) (bridge_length : ℝ) :
  train_speed = 80 * (1000 / 3600) →
  crossing_time = 10.889128869690424 →
  bridge_length = 142 →
  ∃ (train_length : ℝ), abs (train_length - 100.222) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l2716_271625


namespace NUMINAMATH_CALUDE_grape_problem_l2716_271688

theorem grape_problem (x : ℕ) : x > 100 ∧ 
                                x % 3 = 1 ∧ 
                                x % 5 = 2 ∧ 
                                x % 7 = 4 → 
                                x ≤ 172 :=
by sorry

end NUMINAMATH_CALUDE_grape_problem_l2716_271688


namespace NUMINAMATH_CALUDE_track_distance_proof_l2716_271640

/-- The distance Albert needs to run in total, in meters. -/
def total_distance : ℝ := 99

/-- The number of laps Albert has already completed. -/
def completed_laps : ℕ := 6

/-- The number of additional laps Albert will run. -/
def additional_laps : ℕ := 5

/-- The distance around the track, in meters. -/
def track_distance : ℝ := 9

theorem track_distance_proof : 
  (completed_laps + additional_laps : ℝ) * track_distance = total_distance := by
  sorry

end NUMINAMATH_CALUDE_track_distance_proof_l2716_271640


namespace NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_18_l2716_271650

theorem smallest_four_digit_multiple_of_18 :
  ∀ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ n % 18 = 0 → n ≥ 1008 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_18_l2716_271650


namespace NUMINAMATH_CALUDE_acute_triangle_in_right_triangle_l2716_271666

/-- A triangle represented by its vertices -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Predicate to check if a triangle is acute-angled -/
def IsAcuteAngled (t : Triangle) : Prop := sorry

/-- Function to calculate the area of a triangle -/
def TriangleArea (t : Triangle) : ℝ := sorry

/-- Predicate to check if a triangle is right-angled -/
def IsRightAngled (t : Triangle) : Prop := sorry

/-- Predicate to check if one triangle contains another -/
def Contains (t1 t2 : Triangle) : Prop := sorry

theorem acute_triangle_in_right_triangle :
  ∀ (t : Triangle), IsAcuteAngled t → TriangleArea t = 1 →
  ∃ (r : Triangle), IsRightAngled r ∧ TriangleArea r = Real.sqrt 3 ∧ Contains r t := by
  sorry

end NUMINAMATH_CALUDE_acute_triangle_in_right_triangle_l2716_271666


namespace NUMINAMATH_CALUDE_holly_chocolate_milk_l2716_271659

/-- Holly's chocolate milk consumption throughout the day -/
def chocolate_milk_problem (initial_consumption breakfast_consumption lunch_consumption dinner_consumption new_container_size : ℕ) : Prop :=
  let remaining_milk := new_container_size - (lunch_consumption + dinner_consumption)
  remaining_milk = 48

/-- Theorem stating Holly ends the day with 48 ounces of chocolate milk -/
theorem holly_chocolate_milk :
  chocolate_milk_problem 8 8 8 8 64 := by
  sorry

end NUMINAMATH_CALUDE_holly_chocolate_milk_l2716_271659


namespace NUMINAMATH_CALUDE_fir_trees_count_l2716_271612

/-- Represents the statements made by each child --/
inductive Statement
  | Anya : Statement
  | Borya : Statement
  | Vera : Statement
  | Gena : Statement

/-- Represents the gender of each child --/
inductive Gender
  | Boy : Gender
  | Girl : Gender

/-- Associates each child with their gender --/
def childGender : Statement → Gender
  | Statement.Anya => Gender.Girl
  | Statement.Borya => Gender.Boy
  | Statement.Vera => Gender.Girl
  | Statement.Gena => Gender.Boy

/-- Checks if a given number satisfies a child's statement --/
def satisfiesStatement (n : ℕ) : Statement → Bool
  | Statement.Anya => n = 15
  | Statement.Borya => n % 11 = 0
  | Statement.Vera => n < 25
  | Statement.Gena => n % 22 = 0

/-- Theorem: The number of fir trees is 11 --/
theorem fir_trees_count : 
  ∃ (n : ℕ) (t₁ t₂ : Statement), 
    n = 11 ∧ 
    childGender t₁ ≠ childGender t₂ ∧
    satisfiesStatement n t₁ ∧ 
    satisfiesStatement n t₂ ∧
    (∀ t : Statement, t ≠ t₁ → t ≠ t₂ → ¬satisfiesStatement n t) :=
  sorry

end NUMINAMATH_CALUDE_fir_trees_count_l2716_271612


namespace NUMINAMATH_CALUDE_kelly_baking_powder_l2716_271653

/-- The amount of baking powder Kelly has today in boxes -/
def today_amount : ℝ := 0.3

/-- The difference in baking powder between yesterday and today in boxes -/
def difference : ℝ := 0.1

/-- The amount of baking powder Kelly had yesterday in boxes -/
def yesterday_amount : ℝ := today_amount + difference

theorem kelly_baking_powder : yesterday_amount = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_kelly_baking_powder_l2716_271653


namespace NUMINAMATH_CALUDE_expression_calculation_l2716_271697

theorem expression_calculation : (75 * 2024 - 25 * 2024) / 2 = 50600 := by
  sorry

end NUMINAMATH_CALUDE_expression_calculation_l2716_271697


namespace NUMINAMATH_CALUDE_solution_to_equation_l2716_271692

theorem solution_to_equation : ∃ x : ℤ, (2010 + x)^2 = x^2 ∧ x = -1005 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_equation_l2716_271692


namespace NUMINAMATH_CALUDE_complement_of_A_union_B_l2716_271618

def U : Set ℕ := {0,1,2,3,4,5}
def A : Set ℕ := {1,2}
def B : Set ℕ := {x ∈ U | x^2 - 5*x + 4 < 0}

theorem complement_of_A_union_B (x : ℕ) : 
  x ∈ (U \ (A ∪ B)) ↔ x ∈ ({0,4,5} : Set ℕ) := by
  sorry

end NUMINAMATH_CALUDE_complement_of_A_union_B_l2716_271618


namespace NUMINAMATH_CALUDE_order_divides_exponent_l2716_271689

theorem order_divides_exponent (x m d p : ℕ) (hp : Prime p) : 
  (∀ k : ℕ, k > 0 ∧ k < d → x^k % p ≠ 1) →  -- d is the order of x modulo p
  x^d % p = 1 →                             -- definition of order
  x^m % p = 1 →                             -- given condition
  d ∣ m :=                                  -- conclusion: d divides m
sorry

end NUMINAMATH_CALUDE_order_divides_exponent_l2716_271689


namespace NUMINAMATH_CALUDE_factory_B_cheaper_for_200_copies_l2716_271694

/-- Cost calculation for Factory A -/
def cost_A (x : ℝ) : ℝ := 4.8 * x + 500

/-- Cost calculation for Factory B -/
def cost_B (x : ℝ) : ℝ := 6 * x + 200

/-- Theorem stating that Factory B has lower cost for 200 copies -/
theorem factory_B_cheaper_for_200_copies :
  cost_B 200 < cost_A 200 := by
  sorry

end NUMINAMATH_CALUDE_factory_B_cheaper_for_200_copies_l2716_271694


namespace NUMINAMATH_CALUDE_complementary_angles_difference_l2716_271627

theorem complementary_angles_difference (a b : ℝ) : 
  a + b = 90 → -- angles are complementary
  a / b = 5 / 4 → -- ratio of angles is 5:4
  |a - b| = 10 := by sorry

end NUMINAMATH_CALUDE_complementary_angles_difference_l2716_271627


namespace NUMINAMATH_CALUDE_total_fishes_caught_l2716_271639

/-- The number of fishes caught by Hazel and her father in Lake Erie -/
theorem total_fishes_caught (hazel_fishes : Nat) (father_fishes : Nat)
  (h1 : hazel_fishes = 48)
  (h2 : father_fishes = 46) :
  hazel_fishes + father_fishes = 94 := by
  sorry

end NUMINAMATH_CALUDE_total_fishes_caught_l2716_271639


namespace NUMINAMATH_CALUDE_inequalities_proof_l2716_271648

theorem inequalities_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a * b * c ≤ 1/9 ∧ 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (a * b * c)) :=
by sorry

end NUMINAMATH_CALUDE_inequalities_proof_l2716_271648


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2716_271601

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Theorem statement
theorem geometric_sequence_property (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (a 4)^2 - 34 * (a 4) + 64 = 0 →
  (a 8)^2 - 34 * (a 8) + 64 = 0 →
  a 6 = 8 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l2716_271601


namespace NUMINAMATH_CALUDE_total_rowing_campers_l2716_271607

def morning_rowing : ℕ := 13
def afternoon_rowing : ℕ := 21

theorem total_rowing_campers :
  morning_rowing + afternoon_rowing = 34 := by
  sorry

end NUMINAMATH_CALUDE_total_rowing_campers_l2716_271607


namespace NUMINAMATH_CALUDE_single_elimination_tournament_games_l2716_271681

/-- Calculates the number of games needed in a single-elimination tournament. -/
def gamesNeeded (n : ℕ) : ℕ := n - 1

/-- Theorem: In a single-elimination tournament with 512 players, 511 games are needed to crown the champion. -/
theorem single_elimination_tournament_games :
  gamesNeeded 512 = 511 := by
  sorry

end NUMINAMATH_CALUDE_single_elimination_tournament_games_l2716_271681


namespace NUMINAMATH_CALUDE_intersection_solution_set_l2716_271668

def solution_set (f : ℝ → ℝ) : Set ℝ := {x | f x < 0}

def A : Set ℝ := solution_set (λ x => x^2 - 2*x - 3)
def B : Set ℝ := solution_set (λ x => x^2 + x - 6)

theorem intersection_solution_set (a b : ℝ) :
  solution_set (λ x => x^2 + a*x + b) = A ∩ B → a + b = -3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_solution_set_l2716_271668


namespace NUMINAMATH_CALUDE_simplify_expression_l2716_271664

theorem simplify_expression :
  (((Real.sqrt 5 - 2) ^ (Real.sqrt 3 - 2)) / ((Real.sqrt 5 + 2) ^ (Real.sqrt 3 + 2))) = 41 + 20 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2716_271664


namespace NUMINAMATH_CALUDE_helga_extra_hours_thursday_l2716_271603

/-- Represents Helga's work schedule and article production --/
structure HelgaWorkSchedule where
  articles_per_30min : ℕ
  normal_hours_per_day : ℕ
  normal_days_per_week : ℕ
  articles_this_week : ℕ
  extra_hours_friday : ℕ

/-- Calculates the number of extra hours Helga worked on Thursday --/
def extra_hours_thursday (schedule : HelgaWorkSchedule) : ℕ :=
  sorry

/-- Theorem stating that given Helga's work schedule, she worked 2 extra hours on Thursday --/
theorem helga_extra_hours_thursday 
  (schedule : HelgaWorkSchedule)
  (h1 : schedule.articles_per_30min = 5)
  (h2 : schedule.normal_hours_per_day = 4)
  (h3 : schedule.normal_days_per_week = 5)
  (h4 : schedule.articles_this_week = 250)
  (h5 : schedule.extra_hours_friday = 3) :
  extra_hours_thursday schedule = 2 :=
sorry

end NUMINAMATH_CALUDE_helga_extra_hours_thursday_l2716_271603


namespace NUMINAMATH_CALUDE_min_value_of_z_l2716_271633

/-- The objective function to be minimized -/
def z (x y : ℝ) : ℝ := 2*x + 5*y

/-- The feasible region defined by the given constraints -/
def feasible_region (x y : ℝ) : Prop :=
  x - y + 2 ≥ 0 ∧ 2*x + 3*y - 6 ≥ 0 ∧ 3*x + 2*y - 9 ≤ 0

/-- Theorem stating that the minimum value of z in the feasible region is 6 -/
theorem min_value_of_z : 
  ∀ x y : ℝ, feasible_region x y → z x y ≥ 6 ∧ ∃ x₀ y₀ : ℝ, feasible_region x₀ y₀ ∧ z x₀ y₀ = 6 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_z_l2716_271633


namespace NUMINAMATH_CALUDE_inequality_proof_l2716_271644

theorem inequality_proof (a b c : ℝ) (h1 : b > a) (h2 : a > 0) :
  a^2 < b^2 ∧ a*b < b^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2716_271644


namespace NUMINAMATH_CALUDE_product_sum_in_base_l2716_271636

/-- Converts a number from base b to base 10 -/
def to_base_10 (n : ℕ) (b : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base b -/
def from_base_10 (n : ℕ) (b : ℕ) : ℕ := sorry

/-- Checks if a number is expressed in a given base -/
def is_in_base (n : ℕ) (b : ℕ) : Prop := sorry

theorem product_sum_in_base (b : ℕ) :
  (b > 1) →
  (is_in_base 13 b) →
  (is_in_base 14 b) →
  (is_in_base 17 b) →
  (is_in_base 5167 b) →
  (to_base_10 13 b * to_base_10 14 b * to_base_10 17 b = to_base_10 5167 b) →
  (from_base_10 (to_base_10 13 b + to_base_10 14 b + to_base_10 17 b) 7 = 50) :=
by sorry

end NUMINAMATH_CALUDE_product_sum_in_base_l2716_271636


namespace NUMINAMATH_CALUDE_shaded_square_covers_all_rows_l2716_271661

def shaded_sequence : ℕ → ℕ
  | 0 => 1
  | n + 1 => shaded_sequence n + (n + 2)

def covers_all_remainders (n : ℕ) : Prop :=
  ∀ k : Fin 12, ∃ m : ℕ, m ≤ n ∧ shaded_sequence m % 12 = k

theorem shaded_square_covers_all_rows :
  covers_all_remainders 11 ∧ shaded_sequence 11 = 144 ∧
  ∀ k < 11, ¬covers_all_remainders k :=
sorry

end NUMINAMATH_CALUDE_shaded_square_covers_all_rows_l2716_271661


namespace NUMINAMATH_CALUDE_range_of_f_l2716_271655

def f (x : ℝ) : ℝ := 4 * (x - 1)^2 - 1

theorem range_of_f : 
  ∀ y ∈ Set.Icc (-1 : ℝ) 15, ∃ x ∈ Set.Ico (-1 : ℝ) 2, f x = y ∧
  ∀ x ∈ Set.Ico (-1 : ℝ) 2, f x ∈ Set.Icc (-1 : ℝ) 15 :=
by sorry

end NUMINAMATH_CALUDE_range_of_f_l2716_271655


namespace NUMINAMATH_CALUDE_empty_solution_implies_a_geq_half_l2716_271671

theorem empty_solution_implies_a_geq_half (a : ℝ) :
  (∀ x : ℝ, a * x^2 + x + a ≥ 0) → a ≥ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_empty_solution_implies_a_geq_half_l2716_271671


namespace NUMINAMATH_CALUDE_handball_tournament_impossibility_l2716_271634

structure Tournament :=
  (teams : ℕ)
  (points_for_win : ℕ)
  (points_for_draw : ℕ)
  (points_for_loss : ℕ)

def total_games (t : Tournament) : ℕ :=
  t.teams * (t.teams - 1) / 2

def total_points (t : Tournament) : ℕ :=
  total_games t * (t.points_for_win + t.points_for_loss)

theorem handball_tournament_impossibility 
  (t : Tournament)
  (h1 : t.teams = 14)
  (h2 : t.points_for_win = 2)
  (h3 : t.points_for_draw = 1)
  (h4 : t.points_for_loss = 0)
  (h5 : ∀ (i j : ℕ), i ≠ j → i < t.teams → j < t.teams → 
       ∃ (pi pj : ℕ), pi ≠ pj ∧ pi ≤ total_points t ∧ pj ≤ total_points t) :
  ¬(∃ (top bottom : Finset ℕ), 
    top.card = 3 ∧ 
    bottom.card = 3 ∧ 
    (∀ i ∈ top, ∀ j ∈ bottom, 
      ∃ (pi pj : ℕ), pi > pj ∧ 
      pi ≤ total_points t ∧ 
      pj ≤ total_points t)) :=
sorry

end NUMINAMATH_CALUDE_handball_tournament_impossibility_l2716_271634


namespace NUMINAMATH_CALUDE_min_bushes_for_zucchinis_l2716_271651

/-- The number of containers of blueberries yielded by one bush -/
def containers_per_bush : ℕ := 11

/-- The number of containers of blueberries needed to trade for one zucchini -/
def containers_per_zucchini : ℕ := 3

/-- The number of zucchinis Natalie wants to obtain -/
def target_zucchinis : ℕ := 60

/-- The function to calculate the number of bushes needed for a given number of zucchinis -/
def bushes_needed (zucchinis : ℕ) : ℕ :=
  (zucchinis * containers_per_zucchini + containers_per_bush - 1) / containers_per_bush

theorem min_bushes_for_zucchinis :
  bushes_needed target_zucchinis = 17 := by sorry

end NUMINAMATH_CALUDE_min_bushes_for_zucchinis_l2716_271651


namespace NUMINAMATH_CALUDE_age_solution_l2716_271637

/-- The ages of Petya and Anya satisfy the given conditions -/
def age_relationship (petya_age anya_age : ℕ) : Prop :=
  (petya_age = 3 * anya_age) ∧ (petya_age - anya_age = 8)

/-- Theorem stating that Petya is 12 years old and Anya is 4 years old -/
theorem age_solution : ∃ (petya_age anya_age : ℕ), 
  age_relationship petya_age anya_age ∧ petya_age = 12 ∧ anya_age = 4 := by
  sorry

end NUMINAMATH_CALUDE_age_solution_l2716_271637


namespace NUMINAMATH_CALUDE_inequality_range_l2716_271615

theorem inequality_range (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 4) :
  (∀ m : ℝ, (1 / x + 4 / y ≥ m) ↔ m ≤ 9 / 4) := by
  sorry

end NUMINAMATH_CALUDE_inequality_range_l2716_271615


namespace NUMINAMATH_CALUDE_limit_sine_cosine_ratio_l2716_271614

theorem limit_sine_cosine_ratio : 
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 
    0 < |x| ∧ |x| < δ → 
    |(1 + Real.sin (2*x) - Real.cos (2*x)) / (1 - Real.sin (2*x) - Real.cos (2*x)) + 1| < ε :=
by sorry

end NUMINAMATH_CALUDE_limit_sine_cosine_ratio_l2716_271614


namespace NUMINAMATH_CALUDE_initial_books_in_bin_l2716_271643

theorem initial_books_in_bin (initial_books sold_books added_books final_books : ℕ) :
  sold_books = 3 →
  added_books = 10 →
  final_books = 11 →
  initial_books - sold_books + added_books = final_books →
  initial_books = 4 := by
  sorry

end NUMINAMATH_CALUDE_initial_books_in_bin_l2716_271643


namespace NUMINAMATH_CALUDE_pats_running_speed_l2716_271638

/-- Proves that given a 20-mile course, where a person bicycles at 30 mph for 12 minutes
and then runs the rest of the distance, taking a total of 117 minutes to complete the course,
the person's average running speed is 8 mph. -/
theorem pats_running_speed (total_distance : ℝ) (bicycle_speed : ℝ) (bicycle_time : ℝ) (total_time : ℝ)
  (h1 : total_distance = 20)
  (h2 : bicycle_speed = 30)
  (h3 : bicycle_time = 12 / 60)
  (h4 : total_time = 117 / 60) :
  let bicycle_distance := bicycle_speed * bicycle_time
  let run_distance := total_distance - bicycle_distance
  let run_time := total_time - bicycle_time
  run_distance / run_time = 8 := by sorry

end NUMINAMATH_CALUDE_pats_running_speed_l2716_271638


namespace NUMINAMATH_CALUDE_selection_schemes_count_l2716_271672

/-- The number of people to choose from -/
def total_people : ℕ := 6

/-- The number of pavilions to visit -/
def pavilions : ℕ := 4

/-- The number of people who cannot visit a specific pavilion -/
def restricted_people : ℕ := 2

/-- Calculates the number of ways to select people for pavilions with restrictions -/
def selection_schemes : ℕ :=
  Nat.descFactorial total_people pavilions - 
  restricted_people * Nat.descFactorial (total_people - 1) (pavilions - 1)

/-- The theorem stating the number of selection schemes -/
theorem selection_schemes_count : selection_schemes = 240 := by sorry

end NUMINAMATH_CALUDE_selection_schemes_count_l2716_271672


namespace NUMINAMATH_CALUDE_constant_function_theorem_l2716_271647

def IsNonZeroInteger (x : ℚ) : Prop := ∃ (n : ℤ), n ≠ 0 ∧ x = n

theorem constant_function_theorem (f : ℚ → ℚ) 
  (h : ∀ x y, IsNonZeroInteger x → IsNonZeroInteger y → 
    f ((x + y) / 3) = (f x + f y) / 2) :
  ∃ c, ∀ x, IsNonZeroInteger x → f x = c := by
sorry

end NUMINAMATH_CALUDE_constant_function_theorem_l2716_271647


namespace NUMINAMATH_CALUDE_probability_of_marked_items_l2716_271620

theorem probability_of_marked_items 
  (N M n m : ℕ) 
  (h1 : M ≤ N) 
  (h2 : n ≤ N) 
  (h3 : m ≤ n) 
  (h4 : m ≤ M) :
  (Nat.choose M m * Nat.choose (N - M) (n - m)) / Nat.choose N n = 
  (Nat.choose M m * Nat.choose (N - M) (n - m)) / Nat.choose N n :=
by sorry

end NUMINAMATH_CALUDE_probability_of_marked_items_l2716_271620


namespace NUMINAMATH_CALUDE_cookies_calculation_l2716_271698

/-- The number of people receiving cookies -/
def num_people : ℝ := 6.0

/-- The number of cookies each person should receive -/
def cookies_per_person : ℝ := 24.0

/-- The total number of cookies needed -/
def total_cookies : ℝ := num_people * cookies_per_person

theorem cookies_calculation : total_cookies = 144.0 := by
  sorry

end NUMINAMATH_CALUDE_cookies_calculation_l2716_271698


namespace NUMINAMATH_CALUDE_equation_solutions_l2716_271662

/-- A parabola that intersects the x-axis at (-1, 0) and (3, 0) -/
structure Parabola where
  m : ℝ
  n : ℝ
  intersect_neg_one : (-1 - m)^2 + n = 0
  intersect_three : (3 - m)^2 + n = 0

/-- The equation to solve -/
def equation (p : Parabola) (x : ℝ) : Prop :=
  (x - 1)^2 + p.m^2 = 2 * p.m * (x - 1) - p.n

/-- The theorem stating the solutions to the equation -/
theorem equation_solutions (p : Parabola) :
  ∃ (x₁ x₂ : ℝ), x₁ = 0 ∧ x₂ = 4 ∧ equation p x₁ ∧ equation p x₂ :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l2716_271662


namespace NUMINAMATH_CALUDE_power_of_product_l2716_271628

theorem power_of_product (a b : ℝ) : (2 * a * b^2)^3 = 8 * a^3 * b^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l2716_271628


namespace NUMINAMATH_CALUDE_number_of_girls_in_field_trip_l2716_271699

/-- The number of girls in a field trip given the number of students in each van and the total number of boys -/
theorem number_of_girls_in_field_trip (van1 van2 van3 van4 van5 total_boys : ℕ) 
  (h1 : van1 = 24)
  (h2 : van2 = 30)
  (h3 : van3 = 20)
  (h4 : van4 = 36)
  (h5 : van5 = 29)
  (h6 : total_boys = 64) :
  van1 + van2 + van3 + van4 + van5 - total_boys = 75 := by
  sorry

end NUMINAMATH_CALUDE_number_of_girls_in_field_trip_l2716_271699


namespace NUMINAMATH_CALUDE_neither_directly_nor_inversely_proportional_l2716_271669

-- Define what it means for y to be directly proportional to x
def is_directly_proportional (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x

-- Define what it means for y to be inversely proportional to x
def is_inversely_proportional (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, x ≠ 0 → f x = k / x

-- Define the two equations
def eq_A (x y : ℝ) : Prop := x^2 + x*y = 0
def eq_D (x y : ℝ) : Prop := 4*x + y^2 = 7

-- Theorem statement
theorem neither_directly_nor_inversely_proportional :
  (¬ ∃ f : ℝ → ℝ, (∀ x y : ℝ, eq_A x y ↔ y = f x) ∧ 
    (is_directly_proportional f ∨ is_inversely_proportional f)) ∧
  (¬ ∃ g : ℝ → ℝ, (∀ x y : ℝ, eq_D x y ↔ y = g x) ∧ 
    (is_directly_proportional g ∨ is_inversely_proportional g)) :=
by sorry

end NUMINAMATH_CALUDE_neither_directly_nor_inversely_proportional_l2716_271669


namespace NUMINAMATH_CALUDE_third_term_is_negative_45_l2716_271608

/-- A geometric sequence with common ratio -3 and sum of first 2 terms equal to 10 -/
structure GeometricSequence where
  a₁ : ℝ
  ratio : ℝ
  sum_first_two : ℝ
  ratio_eq : ratio = -3
  sum_eq : a₁ + a₁ * ratio = sum_first_two
  sum_first_two_eq : sum_first_two = 10

/-- The third term of the geometric sequence -/
def third_term (seq : GeometricSequence) : ℝ :=
  seq.a₁ * seq.ratio^2

theorem third_term_is_negative_45 (seq : GeometricSequence) :
  third_term seq = -45 := by
  sorry

#check third_term_is_negative_45

end NUMINAMATH_CALUDE_third_term_is_negative_45_l2716_271608


namespace NUMINAMATH_CALUDE_distinct_collections_count_l2716_271663

def word : String := "COMPUTATIONS"

def vowels : Finset Char := {'O', 'U', 'A', 'I'}
def consonants : Multiset Char := {'C', 'M', 'P', 'T', 'T', 'S', 'N'}

def vowel_count : Nat := 4
def consonant_count : Nat := 11

def selected_vowels : Nat := 3
def selected_consonants : Nat := 4

theorem distinct_collections_count :
  (Nat.choose vowel_count selected_vowels) *
  (Nat.choose (consonant_count - 1) selected_consonants +
   Nat.choose (consonant_count - 1) (selected_consonants - 1) +
   Nat.choose (consonant_count - 1) (selected_consonants - 2)) = 200 := by
  sorry

end NUMINAMATH_CALUDE_distinct_collections_count_l2716_271663


namespace NUMINAMATH_CALUDE_concatenated_number_irrational_l2716_271679

/-- The number formed by concatenating the digits of 3^k for k = 1, 2, ... -/
def concatenated_number : ℝ :=
  sorry

/-- Theorem stating that the concatenated number is irrational -/
theorem concatenated_number_irrational : Irrational concatenated_number :=
sorry

end NUMINAMATH_CALUDE_concatenated_number_irrational_l2716_271679


namespace NUMINAMATH_CALUDE_two_numbers_with_specific_means_l2716_271619

theorem two_numbers_with_specific_means :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  Real.sqrt (a * b) = Real.sqrt 5 ∧
  (a + b) / 2 = 5 ∧
  a = 5 + 2 * Real.sqrt 5 ∧
  b = 5 - 2 * Real.sqrt 5 := by
    sorry

end NUMINAMATH_CALUDE_two_numbers_with_specific_means_l2716_271619


namespace NUMINAMATH_CALUDE_inequalities_always_true_l2716_271683

theorem inequalities_always_true (x : ℝ) : 
  (x^2 + 6*x + 10 > 0) ∧ (-x^2 + x - 2 < 0) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_always_true_l2716_271683
