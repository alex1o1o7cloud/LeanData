import Mathlib

namespace bread_slices_remaining_l2118_211828

theorem bread_slices_remaining (total_slices : ℕ) (breakfast_fraction : ℚ) (lunch_slices : ℕ) : 
  total_slices = 12 →
  breakfast_fraction = 1 / 3 →
  lunch_slices = 2 →
  total_slices - (breakfast_fraction * total_slices).num - lunch_slices = 6 := by
sorry

end bread_slices_remaining_l2118_211828


namespace person_a_work_time_l2118_211811

theorem person_a_work_time (person_b_time : ℝ) (combined_work_rate : ℝ) (combined_work_time : ℝ) :
  person_b_time = 45 →
  combined_work_rate = 2/9 →
  combined_work_time = 4 →
  ∃ person_a_time : ℝ,
    person_a_time = 30 ∧
    combined_work_rate = combined_work_time * (1 / person_a_time + 1 / person_b_time) :=
by sorry

end person_a_work_time_l2118_211811


namespace circle_center_radius_sum_l2118_211832

/-- Given a circle C' with equation x^2 - 4y - 15 = -y^2 + 12x + 27,
    prove that its center (p, q) and radius s satisfy p + q + s = 8 + √82 -/
theorem circle_center_radius_sum (x y p q s : ℝ) : 
  (∀ x y, x^2 - 4*y - 15 = -y^2 + 12*x + 27) →
  (∀ x y, (x - p)^2 + (y - q)^2 = s^2) →
  p + q + s = 8 + Real.sqrt 82 := by
  sorry

end circle_center_radius_sum_l2118_211832


namespace family_age_difference_l2118_211802

/-- Represents a family with changing composition over time -/
structure Family where
  initialSize : ℕ
  initialAvgAge : ℕ
  timePassed : ℕ
  currentSize : ℕ
  currentAvgAge : ℕ
  youngestChildAge : ℕ

/-- The age difference between the two youngest children in the family -/
def ageDifference (f : Family) : ℕ := sorry

theorem family_age_difference (f : Family)
  (h1 : f.initialSize = 4)
  (h2 : f.initialAvgAge = 24)
  (h3 : f.timePassed = 10)
  (h4 : f.currentSize = 6)
  (h5 : f.currentAvgAge = 24)
  (h6 : f.youngestChildAge = 3) :
  ageDifference f = 2 := by sorry

end family_age_difference_l2118_211802


namespace peaches_picked_l2118_211833

theorem peaches_picked (initial_peaches final_peaches : ℕ) 
  (h1 : initial_peaches = 34)
  (h2 : final_peaches = 86) :
  final_peaches - initial_peaches = 52 := by
  sorry

end peaches_picked_l2118_211833


namespace problem_statement_l2118_211857

theorem problem_statement (a b c : ℝ) 
  (h1 : a + 2*b + 3*c = 12)
  (h2 : a^2 + b^2 + c^2 = a*b + a*c + b*c) :
  a + b^2 + c^3 = 14 := by
  sorry

end problem_statement_l2118_211857


namespace total_glasses_at_restaurant_l2118_211871

/-- Represents the number of glasses in a small box -/
def small_box : ℕ := 12

/-- Represents the number of glasses in a large box -/
def large_box : ℕ := 16

/-- Represents the difference in the number of large boxes compared to small boxes -/
def box_difference : ℕ := 16

/-- Represents the average number of glasses per box -/
def average_glasses : ℕ := 15

theorem total_glasses_at_restaurant :
  ∃ (small_boxes large_boxes : ℕ),
    large_boxes = small_boxes + box_difference ∧
    (small_box * small_boxes + large_box * large_boxes) / (small_boxes + large_boxes) = average_glasses ∧
    small_box * small_boxes + large_box * large_boxes = 480 :=
sorry

end total_glasses_at_restaurant_l2118_211871


namespace cuboid_4x3x3_two_sided_cubes_l2118_211826

/-- Represents a cuboid with given dimensions -/
structure Cuboid where
  width : ℕ
  length : ℕ
  height : ℕ

/-- Represents a cube with given side length -/
structure Cube where
  side : ℕ

/-- Function to calculate the number of cubes with paint on exactly two sides -/
def cubesWithTwoPaintedSides (c : Cuboid) (numCubes : ℕ) : ℕ :=
  sorry

/-- Theorem stating that a 4x3x3 cuboid cut into 36 equal-sized cubes has 16 cubes with paint on exactly two sides -/
theorem cuboid_4x3x3_two_sided_cubes :
  let c : Cuboid := { width := 4, length := 3, height := 3 }
  cubesWithTwoPaintedSides c 36 = 16 := by
  sorry

end cuboid_4x3x3_two_sided_cubes_l2118_211826


namespace six_possible_values_for_A_l2118_211800

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents the sum operation in the problem -/
def SumOperation (A B X Y : Digit) : Prop :=
  (A.val * 1000000 + B.val * 1000 + A.val) + 
  (B.val * 1000000 + A.val * 1000 + B.val) = 
  (X.val * 10000000 + X.val * 1000000 + X.val * 10000 + Y.val * 1000 + X.val * 100 + X.val)

/-- The main theorem stating that there are exactly 6 possible values for A -/
theorem six_possible_values_for_A :
  ∃! (s : Finset Digit), 
    (∀ A ∈ s, ∃ (B X Y : Digit), A ≠ B ∧ A ≠ X ∧ A ≠ Y ∧ B ≠ X ∧ B ≠ Y ∧ X ≠ Y ∧ SumOperation A B X Y) ∧
    s.card = 6 := by
  sorry

end six_possible_values_for_A_l2118_211800


namespace smallest_n_with_five_pairs_l2118_211890

/-- The function f(n) returns the number of distinct ordered pairs of positive integers (a, b) such that a² + b² = n -/
def f (n : ℕ) : ℕ := (Finset.filter (fun p : ℕ × ℕ => p.1 * p.1 + p.2 * p.2 = n ∧ p.1 > 0 ∧ p.2 > 0) (Finset.product (Finset.range n) (Finset.range n))).card

/-- 125 is the smallest positive integer n for which f(n) = 5 -/
theorem smallest_n_with_five_pairs : (∀ m : ℕ, m > 0 ∧ m < 125 → f m ≠ 5) ∧ f 125 = 5 := by
  sorry

end smallest_n_with_five_pairs_l2118_211890


namespace actual_car_mass_l2118_211873

/-- The mass of a scaled object given the original mass and scale factor. -/
def scaled_mass (original_mass : ℝ) (scale_factor : ℝ) : ℝ :=
  original_mass * scale_factor^3

/-- The mass of the actual car body is 1024 kg. -/
theorem actual_car_mass (model_mass : ℝ) (scale_factor : ℝ) 
  (h1 : model_mass = 2)
  (h2 : scale_factor = 8) :
  scaled_mass model_mass scale_factor = 1024 := by
  sorry

end actual_car_mass_l2118_211873


namespace stratified_sampling_total_employees_l2118_211893

/-- Given a stratified sampling of employees from four companies, 
    prove the total number of employees across all companies. -/
theorem stratified_sampling_total_employees 
  (total_A : ℕ) 
  (selected_A selected_B selected_C selected_D : ℕ) 
  (h1 : total_A = 96)
  (h2 : selected_A = 12)
  (h3 : selected_B = 21)
  (h4 : selected_C = 25)
  (h5 : selected_D = 43) :
  (total_A * (selected_A + selected_B + selected_C + selected_D)) / selected_A = 808 := by
  sorry

#check stratified_sampling_total_employees

end stratified_sampling_total_employees_l2118_211893


namespace arithmetic_expressions_l2118_211807

-- Define the number of arithmetic expressions
def f (n : ℕ) : ℚ :=
  (7/10) * 12^n - (1/5) * (-3)^n

-- State the theorem
theorem arithmetic_expressions (n : ℕ) :
  f n = (7/10) * 12^n - (1/5) * (-3)^n ∧
  f 1 = 9 ∧
  f 2 = 99 :=
by sorry

end arithmetic_expressions_l2118_211807


namespace today_is_wednesday_l2118_211821

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

def prevDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Saturday
  | DayOfWeek.Monday => DayOfWeek.Sunday
  | DayOfWeek.Tuesday => DayOfWeek.Monday
  | DayOfWeek.Wednesday => DayOfWeek.Tuesday
  | DayOfWeek.Thursday => DayOfWeek.Wednesday
  | DayOfWeek.Friday => DayOfWeek.Thursday
  | DayOfWeek.Saturday => DayOfWeek.Friday

def dayAfterTomorrow (d : DayOfWeek) : DayOfWeek := nextDay (nextDay d)

def distanceToSunday (d : DayOfWeek) : Nat :=
  match d with
  | DayOfWeek.Sunday => 0
  | DayOfWeek.Monday => 6
  | DayOfWeek.Tuesday => 5
  | DayOfWeek.Wednesday => 4
  | DayOfWeek.Thursday => 3
  | DayOfWeek.Friday => 2
  | DayOfWeek.Saturday => 1

theorem today_is_wednesday :
  ∃ (today : DayOfWeek),
    (dayAfterTomorrow today = prevDay today) ∧
    (distanceToSunday today = distanceToSunday (prevDay (nextDay today))) ∧
    (today = DayOfWeek.Wednesday) := by
  sorry


end today_is_wednesday_l2118_211821


namespace sandy_shopping_money_l2118_211897

theorem sandy_shopping_money (total : ℝ) (spent_percentage : ℝ) (left : ℝ) : 
  total = 320 →
  spent_percentage = 30 →
  left = total * (1 - spent_percentage / 100) →
  left = 224 :=
by sorry

end sandy_shopping_money_l2118_211897


namespace journey_length_l2118_211845

theorem journey_length :
  ∀ (total : ℝ),
  (total / 4 : ℝ) + 30 + (total / 3 : ℝ) = total →
  total = 72 := by
sorry

end journey_length_l2118_211845


namespace horner_method_v4_l2118_211899

def horner_polynomial (x : ℝ) : ℝ := x^6 - 12*x^5 + 60*x^4 - 160*x^3 + 240*x^2 - 192*x + 64

def horner_v4 (x : ℝ) : ℝ :=
  let v0 := 1
  let v1 := x * v0 - 12
  let v2 := x * v1 + 60
  let v3 := x * v2 - 160
  x * v3 + 240

theorem horner_method_v4 :
  horner_v4 2 = 80 :=
by sorry

end horner_method_v4_l2118_211899


namespace sum_of_17th_roots_minus_one_l2118_211891

theorem sum_of_17th_roots_minus_one (ω : ℂ) : 
  ω^17 = 1 → ω ≠ 1 → ω + ω^2 + ω^3 + ω^4 + ω^5 + ω^6 + ω^7 + ω^8 + ω^9 + ω^10 + ω^11 + ω^12 + ω^13 + ω^14 + ω^15 + ω^16 = -1 := by
sorry

end sum_of_17th_roots_minus_one_l2118_211891


namespace parallel_line_properties_l2118_211867

/-- A line parallel to y = -2x passing through (1, 2) has b = 4 and intersects x-axis at (2, 0) -/
theorem parallel_line_properties (k b : ℝ) :
  (k = -2) →  -- Parallel to y = -2x
  (2 = k * 1 + b) →  -- Passes through (1, 2)
  (b = 4 ∧ ∃ x : ℝ, k * x + b = 0 ∧ x = 2) := by
  sorry

end parallel_line_properties_l2118_211867


namespace smallest_number_divisible_l2118_211866

theorem smallest_number_divisible (n : ℕ) : 
  (∀ m : ℕ, m < n → ¬(∃ k₁ k₂ k₃ k₄ k₅ : ℕ, 
    m - 5 = 27 * k₁ ∧ 
    m - 5 = 36 * k₂ ∧ 
    m - 5 = 44 * k₃ ∧ 
    m - 5 = 52 * k₄ ∧ 
    m - 5 = 65 * k₅)) →
  (∃ k₁ k₂ k₃ k₄ k₅ : ℕ, 
    n - 5 = 27 * k₁ ∧ 
    n - 5 = 36 * k₂ ∧ 
    n - 5 = 44 * k₃ ∧ 
    n - 5 = 52 * k₄ ∧ 
    n - 5 = 65 * k₅) →
  n = 386105 :=
by sorry

end smallest_number_divisible_l2118_211866


namespace inequality_solution_set_l2118_211883

theorem inequality_solution_set (a : ℝ) : 
  (∀ x : ℝ, 3*x - |(-2)*x + 1| ≥ a ↔ x ∈ Set.Ici 2) → a = 3 :=
sorry

end inequality_solution_set_l2118_211883


namespace malcolm_brushing_time_l2118_211889

/-- The number of days Malcolm brushes his teeth -/
def days : ℕ := 30

/-- The number of times Malcolm brushes his teeth per day -/
def brushings_per_day : ℕ := 3

/-- The total time Malcolm spends brushing his teeth in hours -/
def total_hours : ℕ := 3

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

theorem malcolm_brushing_time :
  (total_hours * minutes_per_hour) / (days * brushings_per_day) = 2 := by
  sorry

end malcolm_brushing_time_l2118_211889


namespace limit_rational_function_l2118_211892

/-- The limit of (2x^2 - x - 1) / (x^3 + 2x^2 - x - 2) as x approaches 1 is 1/2 -/
theorem limit_rational_function : 
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 
    0 < |x - 1| ∧ |x - 1| < δ → 
    |(2*x^2 - x - 1) / (x^3 + 2*x^2 - x - 2) - 1/2| < ε := by
  sorry

end limit_rational_function_l2118_211892


namespace shoe_cost_calculation_l2118_211836

def initial_savings : ℕ := 30
def earnings_per_lawn : ℕ := 5
def lawns_per_weekend : ℕ := 3
def weekends_needed : ℕ := 6

def total_earnings : ℕ := earnings_per_lawn * lawns_per_weekend * weekends_needed

theorem shoe_cost_calculation :
  initial_savings + total_earnings = 120 := by sorry

end shoe_cost_calculation_l2118_211836


namespace planes_parallel_transitivity_l2118_211810

-- Define non-coincident planes
variable (α β γ : Plane)
variable (h_distinct : α ≠ β ∧ β ≠ γ ∧ α ≠ γ)

-- Define parallel relation
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem planes_parallel_transitivity 
  (h1 : parallel α γ) 
  (h2 : parallel β γ) : 
  parallel α β :=
sorry

end planes_parallel_transitivity_l2118_211810


namespace S_equation_holds_iff_specific_pairs_l2118_211881

/-- Given real numbers x, y, z with x + y + z = 0, S_r is defined as x^r + y^r + z^r -/
def S (r : ℕ+) (x y z : ℝ) : ℝ := x^(r:ℕ) + y^(r:ℕ) + z^(r:ℕ)

/-- The theorem states that for positive integers m and n, 
    the equation S_{m+n}/(m+n) = (S_m/m) * (S_n/n) holds if and only if 
    (m, n) is one of the pairs (2, 3), (3, 2), (2, 5), or (5, 2) -/
theorem S_equation_holds_iff_specific_pairs (x y z : ℝ) (h : x + y + z = 0) :
  ∀ m n : ℕ+, 
    (S (m + n) x y z) / (m + n : ℝ) = (S m x y z) / (m : ℝ) * (S n x y z) / (n : ℝ) ↔ 
    ((m = 2 ∧ n = 3) ∨ (m = 3 ∧ n = 2) ∨ (m = 2 ∧ n = 5) ∨ (m = 5 ∧ n = 2)) :=
by sorry

end S_equation_holds_iff_specific_pairs_l2118_211881


namespace smallest_inverse_mod_735_l2118_211872

theorem smallest_inverse_mod_735 : 
  ∀ n : ℕ, n > 2 → (∃ m : ℕ, n * m ≡ 1 [MOD 735]) → n ≥ 4 :=
by sorry

end smallest_inverse_mod_735_l2118_211872


namespace min_distance_to_line_l2118_211819

theorem min_distance_to_line (x y : ℝ) (h : 2 * x + y + 5 = 0) :
  ∃ (min_val : ℝ), min_val = Real.sqrt 5 ∧
  ∀ (a b : ℝ), 2 * a + b + 5 = 0 → Real.sqrt (a^2 + b^2) ≥ min_val :=
sorry

end min_distance_to_line_l2118_211819


namespace monkey_multiplication_l2118_211878

/-- The number of spirit monkeys created from one hair -/
def spiritsPerHair : ℕ := 3

/-- The number of new spirit monkeys created by each existing spirit monkey per second -/
def splitRate : ℕ := 3

/-- The number of hairs the Monkey King pulls out -/
def numHairs : ℕ := 10

/-- The number of seconds that pass -/
def timeElapsed : ℕ := 5

/-- The total number of monkeys after the given time -/
def totalMonkeys : ℕ := numHairs * spiritsPerHair * splitRate ^ timeElapsed + 1

theorem monkey_multiplication (spiritsPerHair splitRate numHairs timeElapsed : ℕ) :
  totalMonkeys = 7290 :=
sorry

end monkey_multiplication_l2118_211878


namespace river_width_proof_l2118_211854

/-- The width of a river where two men start from opposite banks, meet 340 meters from one bank
    on their forward journey, and 170 meters from the other bank on their backward journey. -/
def river_width : ℝ := 340

theorem river_width_proof (forward_meeting : ℝ) (backward_meeting : ℝ) 
  (h1 : forward_meeting = 340)
  (h2 : backward_meeting = 170)
  (h3 : forward_meeting + (river_width - forward_meeting) = river_width)
  (h4 : backward_meeting + (river_width - backward_meeting) = river_width) :
  river_width = 340 := by
  sorry

end river_width_proof_l2118_211854


namespace employee_age_problem_l2118_211864

theorem employee_age_problem (total_employees : Nat) 
  (group1_count : Nat) (group1_avg_age : Nat)
  (group2_count : Nat) (group2_avg_age : Nat)
  (group3_count : Nat) (group3_avg_age : Nat)
  (avg_age_29 : Nat) :
  total_employees = 30 →
  group1_count = 10 →
  group1_avg_age = 24 →
  group2_count = 12 →
  group2_avg_age = 30 →
  group3_count = 7 →
  group3_avg_age = 35 →
  avg_age_29 = 29 →
  ∃ (age_30th : Nat), age_30th = 25 := by
sorry


end employee_age_problem_l2118_211864


namespace complex_number_location_l2118_211877

theorem complex_number_location : ∃ (z : ℂ), z = 2 / (1 - Complex.I) - 2 ∧ 
  (z.re < 0 ∧ z.im > 0) := by
  sorry

end complex_number_location_l2118_211877


namespace max_candy_remainder_l2118_211876

theorem max_candy_remainder (x : ℕ) : 
  ∃ (q r : ℕ), x = 9 * q + r ∧ r < 9 ∧ r ≤ 8 :=
by sorry

end max_candy_remainder_l2118_211876


namespace quadratic_minimum_l2118_211804

/-- Given a quadratic function y = x^2 - px + q where the minimum value of y is 1,
    prove that q = 1 + (p^2 / 4) -/
theorem quadratic_minimum (p q : ℝ) : 
  (∀ x, x^2 - p*x + q ≥ 1) ∧ (∃ x, x^2 - p*x + q = 1) → 
  q = 1 + p^2 / 4 := by
sorry

end quadratic_minimum_l2118_211804


namespace train_speed_calculation_l2118_211809

/-- Proves that under given conditions, the train's speed is 45 km/hr -/
theorem train_speed_calculation (jogger_speed : ℝ) (initial_distance : ℝ) 
  (train_length : ℝ) (passing_time : ℝ) :
  jogger_speed = 9 →
  initial_distance = 200 →
  train_length = 210 →
  passing_time = 41 →
  ∃ (train_speed : ℝ), train_speed = 45 := by
  sorry

end train_speed_calculation_l2118_211809


namespace rally_speaking_orders_l2118_211859

theorem rally_speaking_orders (n : ℕ) : 
  2 * (n.factorial) * (n.factorial) = 72 → n = 3 :=
by sorry

end rally_speaking_orders_l2118_211859


namespace first_road_workers_approx_30_man_hours_proportional_to_length_l2118_211815

/-- Represents the details of a road construction project -/
structure RoadProject where
  length : ℝ  -- Road length in km
  workers : ℝ  -- Number of workers
  days : ℝ    -- Number of days worked
  hoursPerDay : ℝ  -- Hours worked per day

/-- Calculates the total man-hours for a road project -/
def manHours (project : RoadProject) : ℝ :=
  project.workers * project.days * project.hoursPerDay

/-- The first road project -/
def road1 : RoadProject := {
  length := 1,
  workers := 30,  -- This is what we're trying to prove
  days := 12,
  hoursPerDay := 8
}

/-- The second road project -/
def road2 : RoadProject := {
  length := 2,
  workers := 20,
  days := 20.571428571428573,
  hoursPerDay := 14
}

/-- Theorem stating that the number of workers on the first road is approximately 30 -/
theorem first_road_workers_approx_30 :
  ∃ ε > 0, ε < 1 ∧ |road1.workers - 30| < ε :=
by sorry

/-- Theorem showing the relationship between man-hours and road length -/
theorem man_hours_proportional_to_length :
  2 * manHours road1 = manHours road2 :=
by sorry

end first_road_workers_approx_30_man_hours_proportional_to_length_l2118_211815


namespace t_range_l2118_211823

-- Define the propositions p and q as functions of t
def p (t : ℝ) : Prop := ∀ x, x^2 + 2*x + 2*t - 4 ≠ 0

def q (t : ℝ) : Prop := 2 < t ∧ t < 3

-- Define the main theorem
theorem t_range (t : ℝ) : 
  (p t ∨ q t) ∧ ¬(p t ∧ q t) → (2 < t ∧ t ≤ 5/2) ∨ t ≥ 3 := by
  sorry

end t_range_l2118_211823


namespace inequality_proof_l2118_211822

theorem inequality_proof (a : ℝ) (h : a ≠ 2) :
  1 / (a^2 - 4*a + 4) > 2 / (a^3 - 8) := by
  sorry

end inequality_proof_l2118_211822


namespace sector_angle_l2118_211895

/-- Given a circular sector with arc length and area both equal to 3,
    prove that the central angle in radians is 3/2. -/
theorem sector_angle (r : ℝ) (θ : ℝ) 
  (arc_length : θ * r = 3)
  (area : 1/2 * θ * r^2 = 3) :
  θ = 3/2 := by
  sorry

end sector_angle_l2118_211895


namespace range_of_x_l2118_211869

-- Define the determinant operation
def det (a b c d : ℝ) : ℝ := a * d - b * c

-- Define the theorem
theorem range_of_x (x : ℝ) : 
  det x 3 (-x) x < det 2 0 1 2 → -4 < x ∧ x < 1 := by
  sorry

end range_of_x_l2118_211869


namespace conditional_statements_requirement_l2118_211896

-- Define a type for the problems
inductive Problem
| AbsoluteValue
| CubeVolume
| PiecewiseFunction

-- Define a function to check if a problem requires conditional statements
def requiresConditionalStatements (p : Problem) : Prop :=
  match p with
  | Problem.AbsoluteValue => true
  | Problem.CubeVolume => false
  | Problem.PiecewiseFunction => true

-- Theorem statement
theorem conditional_statements_requirement :
  (requiresConditionalStatements Problem.AbsoluteValue ∧
   requiresConditionalStatements Problem.PiecewiseFunction) ∧
  ¬requiresConditionalStatements Problem.CubeVolume := by
  sorry


end conditional_statements_requirement_l2118_211896


namespace cube_cut_possible_4_5_cube_cut_impossible_4_7_l2118_211841

/-- Represents a cut of a cube using four planes -/
structure CubeCut where
  planes : Fin 4 → Plane

/-- The maximum distance between any two points in a part resulting from a cube cut -/
def max_distance (cut : CubeCut) : ℝ := sorry

theorem cube_cut_possible_4_5 :
  ∃ (cut : CubeCut), max_distance cut < 4/5 := by sorry

theorem cube_cut_impossible_4_7 :
  ¬ ∃ (cut : CubeCut), max_distance cut < 4/7 := by sorry

end cube_cut_possible_4_5_cube_cut_impossible_4_7_l2118_211841


namespace refrigerator_installation_cost_l2118_211849

def refrigerator_problem (purchase_price : ℝ) (discount_rate : ℝ) 
  (transport_cost : ℝ) (selling_price : ℝ) : Prop :=
  let labelled_price := purchase_price / (1 - discount_rate)
  let profit_rate := 0.1
  let total_cost := labelled_price + transport_cost + 
    (selling_price - labelled_price * (1 + profit_rate))
  total_cost - purchase_price - transport_cost = 287.5

theorem refrigerator_installation_cost :
  refrigerator_problem 13500 0.2 125 18975 :=
sorry

end refrigerator_installation_cost_l2118_211849


namespace ken_payment_l2118_211808

/-- The price of steak per pound -/
def price_per_pound : ℕ := 7

/-- The number of pounds of steak Ken bought -/
def pounds_bought : ℕ := 2

/-- The amount of change Ken received -/
def change_received : ℕ := 6

/-- The amount Ken paid -/
def amount_paid : ℕ := 20

/-- Proof that Ken paid $20 given the conditions -/
theorem ken_payment :
  amount_paid = price_per_pound * pounds_bought + change_received :=
by sorry

end ken_payment_l2118_211808


namespace swim_meet_transportation_l2118_211848

/-- Represents the swim meet transportation problem -/
theorem swim_meet_transportation (num_cars : ℕ) (people_per_car : ℕ) (people_per_van : ℕ)
  (max_car_capacity : ℕ) (max_van_capacity : ℕ) (additional_capacity : ℕ) :
  num_cars = 2 →
  people_per_car = 5 →
  people_per_van = 3 →
  max_car_capacity = 6 →
  max_van_capacity = 8 →
  additional_capacity = 17 →
  ∃ (num_vans : ℕ), 
    num_vans = 3 ∧
    (num_cars * max_car_capacity + num_vans * max_van_capacity) - 
    (num_cars * people_per_car + num_vans * people_per_van) = additional_capacity :=
by sorry

end swim_meet_transportation_l2118_211848


namespace wolf_catches_hare_in_problem_l2118_211843

/-- Represents the chase scenario between a wolf and a hare -/
structure ChaseScenario where
  initial_distance : ℝ
  hiding_spot_distance : ℝ
  wolf_speed : ℝ
  hare_speed : ℝ

/-- Determines if the wolf catches the hare in the given chase scenario -/
def wolf_catches_hare (scenario : ChaseScenario) : Prop :=
  let relative_speed := scenario.wolf_speed - scenario.hare_speed
  let chase_distance := scenario.hiding_spot_distance - scenario.initial_distance
  let chase_time := chase_distance / relative_speed
  scenario.hare_speed * chase_time ≤ scenario.hiding_spot_distance

/-- The specific chase scenario from the problem -/
def problem_scenario : ChaseScenario :=
  { initial_distance := 30
    hiding_spot_distance := 333
    wolf_speed := 600
    hare_speed := 550 }

/-- Theorem stating that the wolf catches the hare in the problem scenario -/
theorem wolf_catches_hare_in_problem : wolf_catches_hare problem_scenario := by
  sorry

end wolf_catches_hare_in_problem_l2118_211843


namespace angle_sum_equality_l2118_211879

-- Define the points in 2D space
def A : ℝ × ℝ := (0, 1)
def B : ℝ × ℝ := (0, 0)
def C : ℝ × ℝ := (1, 0)
def D : ℝ × ℝ := (2, 0)
def E : ℝ × ℝ := (3, 0)
def F : ℝ × ℝ := (3, 1)

-- Define the angles
def angle_FBE : ℝ := sorry
def angle_FCE : ℝ := sorry
def angle_FDE : ℝ := sorry

-- Theorem statement
theorem angle_sum_equality : angle_FBE + angle_FCE = angle_FDE := by sorry

end angle_sum_equality_l2118_211879


namespace simplify_and_evaluate_l2118_211830

theorem simplify_and_evaluate : 
  ∀ x : ℝ, (3*x^2 + 8*x - 6) - (2*x^2 + 4*x - 15) = x^2 + 4*x + 9 ∧ 
  (let x : ℝ := 3; (3*x^2 + 8*x - 6) - (2*x^2 + 4*x - 15) = 30) :=
by
  sorry

#check simplify_and_evaluate

end simplify_and_evaluate_l2118_211830


namespace exists_steps_for_1001_free_ends_l2118_211852

/-- Represents the number of free ends after k steps of construction -/
def free_ends (k : ℕ) : ℕ := 4 * k + 1

/-- Theorem stating that there exists a number of steps that results in 1001 free ends -/
theorem exists_steps_for_1001_free_ends : ∃ k : ℕ, free_ends k = 1001 := by
  sorry

end exists_steps_for_1001_free_ends_l2118_211852


namespace expression_simplification_and_evaluation_l2118_211850

theorem expression_simplification_and_evaluation (x : ℝ) (h1 : x ≠ -1) (h2 : x ≠ 1) :
  (x^2 - x) / (x^2 + 2*x + 1) / ((x - 1) / (x + 1)) = x / (x + 1) ∧
  (3^2 - 3) / (3^2 + 2*3 + 1) / ((3 - 1) / (3 + 1)) = 3 / 4 :=
by sorry

end expression_simplification_and_evaluation_l2118_211850


namespace problem_solution_l2118_211875

theorem problem_solution :
  ∀ (a b c : ℕ),
  ({a, b, c} : Set ℕ) = {0, 1, 2} →
  (((a ≠ 2) ∧ (b ≠ 2) ∧ (c = 0)) ∨
   ((a = 2) ∧ (b ≠ 2) ∧ (c ≠ 0)) ∨
   ((a ≠ 2) ∧ (b = 2) ∧ (c ≠ 0))) →
  10 * a + 2 * b + c = 21 := by
sorry

end problem_solution_l2118_211875


namespace task_completion_probability_l2118_211820

theorem task_completion_probability (p1 p2 : ℚ) (h1 : p1 = 5/8) (h2 : p2 = 3/5) :
  p1 * (1 - p2) = 1/4 := by sorry

end task_completion_probability_l2118_211820


namespace cakes_per_friend_l2118_211853

def total_cakes : ℕ := 8
def num_friends : ℕ := 4

theorem cakes_per_friend :
  total_cakes / num_friends = 2 :=
by sorry

end cakes_per_friend_l2118_211853


namespace even_quadratic_function_l2118_211861

def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem even_quadratic_function (a b c : ℝ) :
  (∀ x, (2 * a - 3 ≤ x ∧ x ≤ 1) → f a b c x = f a b c (-x)) →
  a = 1 ∧ b = 0 ∧ ∃ c : ℝ, True :=
by sorry

end even_quadratic_function_l2118_211861


namespace sum_of_reciprocals_range_l2118_211814

/-- A cubic function with three distinct real roots -/
structure CubicFunction where
  m : ℝ
  n : ℝ
  p : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  has_three_roots : a ≠ b ∧ b ≠ c ∧ a ≠ c
  is_root_a : a^3 + m*a^2 + n*a + p = 0
  is_root_b : b^3 + m*b^2 + n*b + p = 0
  is_root_c : c^3 + m*c^2 + n*c + p = 0
  neg_one_eq_two : ((-1)^3 + m*(-1)^2 + n*(-1) + p) = (2^3 + m*2^2 + n*2 + p)
  one_eq_four : (1^3 + m*1^2 + n*1 + p) = (4^3 + m*4^2 + n*4 + p)
  neg_one_neg : ((-1)^3 + m*(-1)^2 + n*(-1) + p) < 0
  one_pos : (1^3 + m*1^2 + n*1 + p) > 0

/-- The main theorem stating the range of the sum of reciprocals of roots -/
theorem sum_of_reciprocals_range (f : CubicFunction) :
  -(3/4) < (1/f.a + 1/f.b + 1/f.c) ∧ (1/f.a + 1/f.b + 1/f.c) < -(3/14) := by
  sorry

end sum_of_reciprocals_range_l2118_211814


namespace column_compression_strength_l2118_211827

theorem column_compression_strength (T H L : ℚ) : 
  T = 3 → H = 6 → L = (15 * T^5) / H^3 → L = 55 / 13 := by sorry

end column_compression_strength_l2118_211827


namespace lollipop_bouquets_l2118_211825

theorem lollipop_bouquets (cherry orange raspberry lemon candycane chocolate : ℕ) 
  (h1 : cherry = 4)
  (h2 : orange = 6)
  (h3 : raspberry = 8)
  (h4 : lemon = 10)
  (h5 : candycane = 12)
  (h6 : chocolate = 14) :
  Nat.gcd cherry (Nat.gcd orange (Nat.gcd raspberry (Nat.gcd lemon (Nat.gcd candycane chocolate)))) = 2 := by
  sorry

end lollipop_bouquets_l2118_211825


namespace wedding_attendance_percentage_l2118_211816

def expected_attendees : ℕ := 220
def actual_attendees : ℕ := 209

theorem wedding_attendance_percentage :
  (expected_attendees - actual_attendees : ℚ) / expected_attendees * 100 = 5 := by
  sorry

end wedding_attendance_percentage_l2118_211816


namespace max_value_f_in_interval_l2118_211894

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 2

-- State the theorem
theorem max_value_f_in_interval :
  ∃ (c : ℝ), c ∈ Set.Icc (-1) 1 ∧
  ∀ (x : ℝ), x ∈ Set.Icc (-1) 1 → f x ≤ f c ∧ f c = 2 :=
by sorry

end max_value_f_in_interval_l2118_211894


namespace passing_percentage_is_40_l2118_211801

/-- The maximum marks possible in the exam -/
def max_marks : ℕ := 550

/-- The marks obtained by the student -/
def obtained_marks : ℕ := 200

/-- The number of marks by which the student failed -/
def fail_margin : ℕ := 20

/-- The passing percentage for the exam -/
def passing_percentage : ℚ :=
  (obtained_marks + fail_margin : ℚ) / max_marks * 100

theorem passing_percentage_is_40 :
  passing_percentage = 40 := by sorry

end passing_percentage_is_40_l2118_211801


namespace greatest_good_set_size_l2118_211847

/-- A set S of positive integers is "good" if there exists a coloring of positive integers
    with k colors such that no element from S can be written as the sum of two distinct
    positive integers having the same color. -/
def IsGood (S : Set ℕ) (k : ℕ) : Prop :=
  ∃ (c : ℕ → Fin k), ∀ s ∈ S, ∀ x y : ℕ, x < y → x + y = s → c x ≠ c y

/-- The set S defined as {a+1, a+2, ..., a+t} for some positive integer a -/
def S (a t : ℕ) : Set ℕ := {n : ℕ | a + 1 ≤ n ∧ n ≤ a + t}

theorem greatest_good_set_size (k : ℕ) (h : k > 1) :
  (∃ t : ℕ, ∀ a : ℕ, a > 0 → IsGood (S a t) k ∧
    ∀ t' : ℕ, t' > t → ∃ a : ℕ, a > 0 ∧ ¬IsGood (S a t') k) ∧
  (∀ t : ℕ, (∀ a : ℕ, a > 0 → IsGood (S a t) k) → t ≤ 2 * k - 2) :=
sorry

end greatest_good_set_size_l2118_211847


namespace couple_stock_purchase_l2118_211862

/-- Calculates the number of shares a couple can buy given their savings plan and stock price --/
def shares_to_buy (wife_weekly_savings : ℕ) (husband_monthly_savings : ℕ) (months : ℕ) (stock_price : ℕ) : ℕ :=
  let wife_monthly_savings := wife_weekly_savings * 4
  let total_monthly_savings := wife_monthly_savings + husband_monthly_savings
  let total_savings := total_monthly_savings * months
  let investment := total_savings / 2
  investment / stock_price

/-- Theorem stating that the couple can buy 25 shares given their specific savings plan --/
theorem couple_stock_purchase :
  shares_to_buy 100 225 4 50 = 25 := by
  sorry

end couple_stock_purchase_l2118_211862


namespace problem_l2118_211837

noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem problem (a b : ℝ) (h : f (a * b) = 1) : f (a^2) + f (b^2) = 2 := by
  sorry

end problem_l2118_211837


namespace basket_problem_l2118_211898

theorem basket_problem (total : ℕ) (apples : ℕ) (oranges : ℕ) (both : ℕ) 
  (h1 : total = 15)
  (h2 : apples = 10)
  (h3 : oranges = 8)
  (h4 : both = 5) :
  total - (apples + oranges - both) = 2 := by
  sorry

end basket_problem_l2118_211898


namespace solve_exponential_equation_l2118_211839

theorem solve_exponential_equation (x y z : ℕ) :
  (3 : ℝ)^x * (4 : ℝ)^y / (2 : ℝ)^z = 59049 ∧ x - y + 2*z = 10 → x = 10 := by
  sorry

end solve_exponential_equation_l2118_211839


namespace complex_functional_equation_l2118_211856

theorem complex_functional_equation 
  (f : ℂ → ℂ) 
  (h : ∀ z : ℂ, f z + z * f (1 - z) = 1 + z) : 
  ∀ w : ℂ, f w = 1 := by
sorry

end complex_functional_equation_l2118_211856


namespace polynomial_division_remainder_l2118_211874

theorem polynomial_division_remainder :
  ∃ q : Polynomial ℤ, 
    3 * X^4 + 8 * X^3 - 29 * X^2 - 17 * X + 34 = 
    (X^2 + 5 * X - 3) * q + (79 * X - 11) ∧ 
    (79 * X - 11).degree < (X^2 + 5 * X - 3).degree :=
by sorry

end polynomial_division_remainder_l2118_211874


namespace complex_second_quadrant_l2118_211817

theorem complex_second_quadrant (a : ℝ) : 
  let z : ℂ := (a + 3*Complex.I)/Complex.I + a
  (z.re < 0 ∧ z.im > 0) → a = -4 := by
  sorry

end complex_second_quadrant_l2118_211817


namespace sixth_term_is_twelve_l2118_211806

/-- An arithmetic sequence with its first term and sum of first three terms specified -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  a₁ : a 1 = 2
  S₃ : (a 1) + (a 2) + (a 3) = 12
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1

/-- The 6th term of the arithmetic sequence is 12 -/
theorem sixth_term_is_twelve (seq : ArithmeticSequence) : seq.a 6 = 12 := by
  sorry

end sixth_term_is_twelve_l2118_211806


namespace total_money_calculation_l2118_211844

-- Define the proportions
def prop1 : ℚ := 1/2
def prop2 : ℚ := 1/3
def prop3 : ℚ := 3/4

-- Define the value of the second part
def second_part : ℝ := 164.6315789473684

-- Theorem statement
theorem total_money_calculation (total : ℝ) :
  (total * (prop2 / (prop1 + prop2 + prop3)) = second_part) →
  total = 65.1578947368421 := by
sorry

end total_money_calculation_l2118_211844


namespace infinitely_many_a_for_positive_integer_l2118_211846

theorem infinitely_many_a_for_positive_integer (n : ℕ) :
  ∃ (f : ℕ → ℤ), Function.Injective f ∧
  ∀ (k : ℕ), (n^6 + 3 * (f k) : ℤ) > 0 :=
sorry

end infinitely_many_a_for_positive_integer_l2118_211846


namespace train_length_l2118_211868

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 52 → time = 9 → ∃ length : ℝ, abs (length - 129.96) < 0.01 := by
  sorry

#check train_length

end train_length_l2118_211868


namespace tv_sets_in_shop_a_l2118_211870

/-- The number of electronic shops in the Naza market -/
def num_shops : ℕ := 5

/-- The average number of TV sets in each shop -/
def average_tv_sets : ℕ := 48

/-- The number of TV sets in shop b -/
def tv_sets_b : ℕ := 30

/-- The number of TV sets in shop c -/
def tv_sets_c : ℕ := 60

/-- The number of TV sets in shop d -/
def tv_sets_d : ℕ := 80

/-- The number of TV sets in shop e -/
def tv_sets_e : ℕ := 50

/-- Theorem: Given the conditions, shop a must have 20 TV sets -/
theorem tv_sets_in_shop_a : 
  (num_shops * average_tv_sets) - (tv_sets_b + tv_sets_c + tv_sets_d + tv_sets_e) = 20 := by
  sorry

end tv_sets_in_shop_a_l2118_211870


namespace total_cost_usd_l2118_211838

/-- Calculate the total cost of items with discounts and tax -/
def calculate_total_cost (shirt_price : ℚ) (shoe_price_diff : ℚ) (dress_price : ℚ)
  (shoe_discount : ℚ) (dress_discount : ℚ) (sales_tax : ℚ) (exchange_rate : ℚ) : ℚ :=
  let shoe_price := shirt_price + shoe_price_diff
  let discounted_shoe_price := shoe_price * (1 - shoe_discount)
  let discounted_dress_price := dress_price * (1 - dress_discount)
  let subtotal := 2 * shirt_price + discounted_shoe_price + discounted_dress_price
  let bag_price := subtotal / 2
  let total_before_tax := subtotal + bag_price
  let tax_amount := total_before_tax * sales_tax
  let total_with_tax := total_before_tax + tax_amount
  total_with_tax * exchange_rate

/-- Theorem stating the total cost in USD -/
theorem total_cost_usd :
  calculate_total_cost 12 5 25 (1/10) (1/20) (7/100) (118/100) = 11942/100 :=
by sorry

end total_cost_usd_l2118_211838


namespace line_equation_proof_l2118_211834

-- Define the parabola C
def parabola_C (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line l
def line_l (x y : ℝ) : Prop := y = 2*x - 3

-- Define the midpoint of AB
def midpoint_AB : ℝ × ℝ := (2, 1)

-- Theorem statement
theorem line_equation_proof :
  ∀ (A B : ℝ × ℝ),
  parabola_C A.1 A.2 →
  parabola_C B.1 B.2 →
  (A.1 + B.1) / 2 = midpoint_AB.1 →
  (A.2 + B.2) / 2 = midpoint_AB.2 →
  line_l A.1 A.2 ∧ line_l B.1 B.2 :=
by sorry


end line_equation_proof_l2118_211834


namespace ordered_pairs_satisfying_equations_l2118_211855

theorem ordered_pairs_satisfying_equations :
  ∀ (x y : ℝ), x^2 * y = 3 ∧ x + x*y = 4 ↔ (x = 1 ∧ y = 3) ∨ (x = 3 ∧ y = 1/3) := by
  sorry

end ordered_pairs_satisfying_equations_l2118_211855


namespace elsa_marbles_l2118_211882

/-- The number of marbles Elsa started with -/
def initial_marbles : ℕ := sorry

/-- The number of marbles Elsa lost at breakfast -/
def lost_at_breakfast : ℕ := 3

/-- The number of marbles Elsa gave to Susie at lunch -/
def given_to_susie : ℕ := 5

/-- The number of new marbles Elsa's mom bought -/
def new_marbles : ℕ := 12

/-- The number of marbles Elsa had at the end of the day -/
def final_marbles : ℕ := 54

theorem elsa_marbles :
  initial_marbles = 40 :=
by
  have h1 : initial_marbles - lost_at_breakfast - given_to_susie + new_marbles + 2 * given_to_susie = final_marbles :=
    sorry
  sorry

end elsa_marbles_l2118_211882


namespace reach_target_probability_l2118_211813

-- Define the number of lily pads
def num_pads : ℕ := 16

-- Define the starting pad
def start_pad : ℕ := 2

-- Define the target pad
def target_pad : ℕ := 14

-- Define the predator pads
def predator_pads : List ℕ := [4, 9]

-- Define the movement probabilities
def move_prob : ℚ := 1/3
def skip_one_prob : ℚ := 1/3
def skip_two_prob : ℚ := 1/3

-- Function to calculate the probability of reaching the target pad
def reach_target_prob : ℚ := sorry

-- Theorem stating the probability of reaching the target pad
theorem reach_target_probability :
  reach_target_prob = 1/729 :=
sorry

end reach_target_probability_l2118_211813


namespace train_speed_calculation_l2118_211888

/-- Given two trains running on parallel rails in the same direction, this theorem
    calculates the speed of the second train based on the given conditions. -/
theorem train_speed_calculation
  (length1 : ℝ) (length2 : ℝ) (speed1 : ℝ) (crossing_time : ℝ)
  (h1 : length1 = 200) -- Length of first train in meters
  (h2 : length2 = 180) -- Length of second train in meters
  (h3 : speed1 = 45) -- Speed of first train in km/h
  (h4 : crossing_time = 273.6) -- Time to cross in seconds
  : ∃ (speed2 : ℝ), speed2 = 40 ∧ 
    (speed1 - speed2) * (crossing_time / 3600) = (length1 + length2) / 1000 :=
by sorry

end train_speed_calculation_l2118_211888


namespace binomial_coefficient_divisibility_equivalence_l2118_211880

def is_prime (p : ℕ) : Prop := Nat.Prime p

theorem binomial_coefficient_divisibility_equivalence 
  (p : ℕ) (n : ℕ) 
  (h1 : is_prime p) 
  (h2 : is_prime (11 * 39 * p)) : 
  (∃ k : ℕ, k ≤ n ∧ p ∣ Nat.choose n k) ↔ 
  (∃ s q : ℕ, n = p^s * q - 1 ∧ s ≥ 0 ∧ 0 < q ∧ q < p) :=
sorry

end binomial_coefficient_divisibility_equivalence_l2118_211880


namespace smallest_positive_integer_satisfying_congruences_l2118_211842

theorem smallest_positive_integer_satisfying_congruences :
  ∃! x : ℕ+, 
    (45 * x.val + 9) % 25 = 3 ∧
    (2 * x.val) % 5 = 3 ∧
    ∀ y : ℕ+, 
      ((45 * y.val + 9) % 25 = 3 ∧ (2 * y.val) % 5 = 3) → x ≤ y :=
by
  -- Proof goes here
  sorry

end smallest_positive_integer_satisfying_congruences_l2118_211842


namespace problem_1_problem_2_l2118_211818

-- Problem 1
theorem problem_1 : Real.sqrt 5 ^ 2 + |(-3)| - (Real.pi + Real.sqrt 3) ^ 0 = 7 := by sorry

-- Problem 2
theorem problem_2 : 
  Set.Ioo (-1 : ℝ) 2 = {x : ℝ | 5 * x - 10 ≤ 0 ∧ x + 3 > -2 * x} := by sorry

end problem_1_problem_2_l2118_211818


namespace zlatoust_miass_distance_l2118_211885

/-- The distance between Zlatoust and Miass -/
def distance : ℝ := sorry

/-- The speed of GAZ truck -/
def speed_gaz : ℝ := sorry

/-- The speed of MAZ truck -/
def speed_maz : ℝ := sorry

/-- The speed of KAMAZ truck -/
def speed_kamaz : ℝ := sorry

theorem zlatoust_miass_distance :
  (distance + 18) / speed_kamaz = (distance - 18) / speed_maz ∧
  (distance + 25) / speed_kamaz = (distance - 25) / speed_gaz ∧
  (distance + 8) / speed_maz = (distance - 8) / speed_gaz →
  distance = 60 := by sorry

end zlatoust_miass_distance_l2118_211885


namespace files_remaining_l2118_211865

theorem files_remaining (music_files video_files deleted_files : ℕ) 
  (h1 : music_files = 27)
  (h2 : video_files = 42)
  (h3 : deleted_files = 11) :
  music_files + video_files - deleted_files = 58 := by
  sorry

end files_remaining_l2118_211865


namespace sufficient_not_necessary_condition_l2118_211805

theorem sufficient_not_necessary_condition (x y : ℝ) :
  (x + y > 2 → max x y > 1) ∧
  ¬(max x y > 1 → x + y > 2) :=
by sorry

end sufficient_not_necessary_condition_l2118_211805


namespace unbroken_seashells_l2118_211863

/-- Given that Mike found a total of 6 seashells and 4 of them were broken,
    prove that the number of unbroken seashells is 2. -/
theorem unbroken_seashells (total : ℕ) (broken : ℕ) (h1 : total = 6) (h2 : broken = 4) :
  total - broken = 2 := by
  sorry

end unbroken_seashells_l2118_211863


namespace joeys_lawn_mowing_l2118_211835

theorem joeys_lawn_mowing (
  sneaker_cost : ℕ)
  (lawn_earnings : ℕ)
  (figure_price : ℕ)
  (figure_count : ℕ)
  (job_hours : ℕ)
  (hourly_rate : ℕ)
  (h1 : sneaker_cost = 92)
  (h2 : lawn_earnings = 8)
  (h3 : figure_price = 9)
  (h4 : figure_count = 2)
  (h5 : job_hours = 10)
  (h6 : hourly_rate = 5)
  : (sneaker_cost - (figure_price * figure_count + job_hours * hourly_rate)) / lawn_earnings = 3 := by
  sorry

end joeys_lawn_mowing_l2118_211835


namespace cost_of_one_juice_and_sandwich_janice_purchase_l2118_211824

/-- Given the cost of multiple juices and sandwiches, calculate the cost of one juice and one sandwich. -/
theorem cost_of_one_juice_and_sandwich 
  (total_juice_cost : ℝ) 
  (juice_quantity : ℕ) 
  (total_sandwich_cost : ℝ) 
  (sandwich_quantity : ℕ) : 
  total_juice_cost / juice_quantity + total_sandwich_cost / sandwich_quantity = 5 :=
by
  sorry

/-- Specific instance of the theorem with given values -/
theorem janice_purchase : 
  (10 : ℝ) / 5 + (6 : ℝ) / 2 = 5 :=
by
  sorry

end cost_of_one_juice_and_sandwich_janice_purchase_l2118_211824


namespace bernardo_silvia_game_l2118_211884

theorem bernardo_silvia_game (M : ℕ) : 
  (M ≤ 1999) →
  (32 * M + 1600 < 2000) →
  (32 * M + 1700 ≥ 2000) →
  (∀ N : ℕ, N < M → (32 * N + 1600 < 2000 → 32 * N + 1700 < 2000)) →
  (M = 10 ∧ (M / 10 + M % 10 = 1)) := by
  sorry

end bernardo_silvia_game_l2118_211884


namespace triangle_problem_l2118_211812

theorem triangle_problem (A B C : Real) (a b c : Real) (D : Real × Real) :
  -- Given conditions
  0 < A ∧ A < π/2 ∧
  0 < B ∧ B < π/2 ∧
  0 < C ∧ C < π/2 ∧
  A + B + C = π ∧
  a = 3 ∧
  b = Real.sqrt 13 ∧
  a * Real.sin (2 * B) = b * Real.sin A ∧
  -- Definition of point D
  D = ((1/3) * (Real.cos A, Real.sin A) + (2/3) * (Real.cos C, Real.sin C)) →
  -- Conclusions
  B = π/3 ∧
  Real.sqrt ((D.1 - Real.cos B)^2 + (D.2 - Real.sin B)^2) = (2 * Real.sqrt 19) / 3 :=
by sorry

end triangle_problem_l2118_211812


namespace no_solutions_exist_l2118_211829

theorem no_solutions_exist : ¬∃ (x y z : ℝ), (x + y = 3) ∧ (x * y - z^2 = 4) := by
  sorry

end no_solutions_exist_l2118_211829


namespace inequality_solution_set_l2118_211886

theorem inequality_solution_set (m n : ℝ) (h : m > n) :
  {x : ℝ | (n - m) * x > 0} = {x : ℝ | x < 0} := by sorry

end inequality_solution_set_l2118_211886


namespace jack_deer_hunting_l2118_211803

/-- The number of times Jack goes hunting per month -/
def hunts_per_month : ℕ := 6

/-- The duration of the hunting season in months -/
def hunting_season_months : ℕ := 3

/-- The number of deer Jack catches per hunting trip -/
def deer_per_hunt : ℕ := 2

/-- The weight of each deer in pounds -/
def deer_weight : ℕ := 600

/-- The fraction of deer weight Jack keeps -/
def kept_fraction : ℚ := 1 / 2

/-- The total amount of deer Jack keeps in pounds -/
def deer_kept : ℕ := 10800

theorem jack_deer_hunting :
  hunts_per_month * hunting_season_months * deer_per_hunt * deer_weight * kept_fraction = deer_kept := by
  sorry

end jack_deer_hunting_l2118_211803


namespace x_plus_y_equals_four_l2118_211858

/-- Geometric configuration with segments AB and A'B' --/
structure GeometricConfiguration where
  AB : ℝ
  APB : ℝ
  P_distance_from_D : ℝ
  total_distance : ℝ

/-- Theorem stating that x + y = 4 in the given geometric configuration --/
theorem x_plus_y_equals_four (config : GeometricConfiguration) 
  (h1 : config.AB = 6)
  (h2 : config.APB = 10)
  (h3 : config.P_distance_from_D = 2)
  (h4 : config.total_distance = 12) :
  let D := config.AB / 2
  let D' := config.APB / 2
  let x := config.P_distance_from_D
  let y := config.total_distance - (D + x + D')
  x + y = 4 := by
  sorry


end x_plus_y_equals_four_l2118_211858


namespace pure_imaginary_ratio_l2118_211831

theorem pure_imaginary_ratio (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) 
  (h3 : ∃ (y : ℝ), (3 - 8*I) * (a + b*I) = y*I) : 
  a / b = -8 / 3 := by
sorry

end pure_imaginary_ratio_l2118_211831


namespace inscribed_squares_product_l2118_211851

theorem inscribed_squares_product (a b : ℝ) : 
  (∃ small_square large_square : ℝ → ℝ → Prop,
    (∀ x y, small_square x y → x^2 + y^2 ≤ 9) ∧
    (∀ x y, large_square x y → x^2 + y^2 ≤ 16) ∧
    (∀ x y, small_square x y → ∃ u v, large_square u v ∧ 
      ((x = u ∧ y ∈ [0, 4]) ∨ (x ∈ [0, 4] ∧ y = v) ∨ 
       (x = -u ∧ y ∈ [0, 4]) ∨ (x ∈ [0, 4] ∧ y = -v))) ∧
    (a + b = 4) ∧
    (a^2 + b^2 = 18)) →
  a * b = -1 :=
by sorry

end inscribed_squares_product_l2118_211851


namespace three_factors_for_cash_preference_l2118_211840

/-- Represents an economic factor influencing payment preference --/
structure EconomicFactor where
  name : String
  description : String

/-- Represents a large retail chain --/
structure RetailChain where
  name : String
  prefersCash : Bool

/-- Determines if an economic factor contributes to cash preference --/
def contributesToCashPreference (factor : EconomicFactor) (chain : RetailChain) : Prop :=
  factor.description ≠ "" ∧ chain.prefersCash

/-- The main theorem stating that there are at least three distinct economic factors
    contributing to cash preference for large retail chains --/
theorem three_factors_for_cash_preference :
  ∃ (f1 f2 f3 : EconomicFactor) (chain : RetailChain),
    f1 ≠ f2 ∧ f1 ≠ f3 ∧ f2 ≠ f3 ∧
    contributesToCashPreference f1 chain ∧
    contributesToCashPreference f2 chain ∧
    contributesToCashPreference f3 chain :=
  sorry

end three_factors_for_cash_preference_l2118_211840


namespace division_remainder_proof_l2118_211887

theorem division_remainder_proof (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) : 
  dividend = 690 →
  divisor = 36 →
  quotient = 19 →
  dividend = divisor * quotient + remainder →
  remainder = 6 := by
sorry

end division_remainder_proof_l2118_211887


namespace bert_spent_nine_at_dry_cleaners_l2118_211860

/-- Represents Bert's spending problem --/
def BertSpending (initial_amount : ℚ) (dry_cleaner_amount : ℚ) : Prop :=
  let hardware_store := initial_amount / 4
  let after_hardware := initial_amount - hardware_store
  let after_dry_cleaner := after_hardware - dry_cleaner_amount
  let grocery_store := after_dry_cleaner / 2
  let final_amount := after_dry_cleaner - grocery_store
  (initial_amount = 44) ∧ (final_amount = 12)

/-- Theorem stating that Bert spent $9 at the dry cleaners --/
theorem bert_spent_nine_at_dry_cleaners :
  ∃ (dry_cleaner_amount : ℚ), BertSpending 44 dry_cleaner_amount ∧ dry_cleaner_amount = 9 := by
  sorry

end bert_spent_nine_at_dry_cleaners_l2118_211860
