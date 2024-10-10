import Mathlib

namespace fib_150_mod_5_l2567_256771

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

-- Define the property we want to prove
theorem fib_150_mod_5 : fib 150 % 5 = 0 := by
  sorry

end fib_150_mod_5_l2567_256771


namespace divisible_by_five_not_ending_in_five_l2567_256745

theorem divisible_by_five_not_ending_in_five : ∃ n : ℕ, 5 ∣ n ∧ n % 10 ≠ 5 := by
  sorry

end divisible_by_five_not_ending_in_five_l2567_256745


namespace percent_composition_l2567_256756

theorem percent_composition (y : ℝ) : (18 / 100) * y = (30 / 100) * ((60 / 100) * y) := by
  sorry

end percent_composition_l2567_256756


namespace toothpicks_for_1002_base_l2567_256777

/-- Calculates the number of toothpicks required for a large equilateral triangle
    constructed with rows of small equilateral triangles. -/
def toothpicks_count (base_triangles : ℕ) : ℕ :=
  let total_triangles := base_triangles * (base_triangles + 1) / 2
  let total_sides := 3 * total_triangles
  let boundary_sides := 3 * base_triangles
  (total_sides - boundary_sides) / 2 + boundary_sides

/-- Theorem stating that for a large equilateral triangle with 1002 small triangles
    in its base, the total number of toothpicks required is 752253. -/
theorem toothpicks_for_1002_base : toothpicks_count 1002 = 752253 := by
  sorry

end toothpicks_for_1002_base_l2567_256777


namespace triangle_transformation_exists_l2567_256737

-- Define a point in the 2D plane
structure Point :=
  (x : Int) (y : Int)

-- Define a triangle as a set of three points
structure Triangle :=
  (a : Point) (b : Point) (c : Point)

-- Define the 90° counterclockwise rotation transformation
def rotate90 (center : Point) (p : Point) : Point :=
  let dx := p.x - center.x
  let dy := p.y - center.y
  Point.mk (center.x - dy) (center.y + dx)

-- Define the initial and target triangles
def initialTriangle : Triangle :=
  Triangle.mk (Point.mk 0 0) (Point.mk 1 0) (Point.mk 0 1)

def targetTriangle : Triangle :=
  Triangle.mk (Point.mk 0 0) (Point.mk 1 0) (Point.mk 1 1)

-- Theorem statement
theorem triangle_transformation_exists :
  ∃ (rotationCenter : Point),
    rotate90 rotationCenter initialTriangle.a = targetTriangle.a ∧
    rotate90 rotationCenter initialTriangle.b = targetTriangle.b ∧
    rotate90 rotationCenter initialTriangle.c = targetTriangle.c :=
by sorry

end triangle_transformation_exists_l2567_256737


namespace train_length_l2567_256730

/-- The length of a train given its speed and time to cross a bridge -/
theorem train_length (speed : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) :
  speed = 10 →
  bridge_length = 250 →
  crossing_time = 34.997200223982084 →
  ∃ train_length : ℝ, 
    train_length + bridge_length = speed * crossing_time ∧ 
    abs (train_length - 99.97) < 0.01 := by
  sorry

end train_length_l2567_256730


namespace average_after_adding_constant_specific_average_problem_l2567_256780

theorem average_after_adding_constant (n : ℕ) (original_avg : ℚ) (added_const : ℚ) :
  n > 0 →
  let new_avg := original_avg + added_const
  new_avg = (n * original_avg + n * added_const) / n := by
  sorry

theorem specific_average_problem :
  let n : ℕ := 15
  let original_avg : ℚ := 40
  let added_const : ℚ := 10
  let new_avg := original_avg + added_const
  new_avg = 50 := by
  sorry

end average_after_adding_constant_specific_average_problem_l2567_256780


namespace simplify_and_evaluate_expression_l2567_256762

theorem simplify_and_evaluate_expression :
  let a : ℝ := 3 - Real.sqrt 2
  let expression := (((a^2 - 1) / (a - 3) - a - 1) / ((a + 1) / (a^2 - 6*a + 9)))
  expression = -2 * Real.sqrt 2 := by
  sorry

end simplify_and_evaluate_expression_l2567_256762


namespace total_money_raised_is_correct_l2567_256750

/-- Represents the total amount of money raised in a two-month period of telethons --/
def total_money_raised : ℝ :=
  let friday_rate_first_12h := 4000
  let friday_rate_last_14h := friday_rate_first_12h * 1.1
  let saturday_rate_first_12h := 5000
  let saturday_rate_last_14h := saturday_rate_first_12h * 1.2
  let sunday_initial_rate := saturday_rate_first_12h * 0.85
  let sunday_rate_5_percent_increase := sunday_initial_rate * 1.05
  let sunday_rate_30_percent_increase := sunday_initial_rate * 1.3
  let sunday_rate_10_percent_decrease := sunday_initial_rate * 0.9
  let sunday_rate_20_percent_increase := sunday_initial_rate * 1.2
  let sunday_rate_25_percent_decrease := sunday_initial_rate * 0.75

  let friday_total := friday_rate_first_12h * 12 + friday_rate_last_14h * 14
  let saturday_total := saturday_rate_first_12h * 12 + saturday_rate_last_14h * 14
  let sunday_total := sunday_initial_rate * 10 + sunday_rate_5_percent_increase * 2 +
                      sunday_rate_30_percent_increase * 4 + sunday_rate_10_percent_decrease * 2 +
                      sunday_rate_20_percent_increase * 1 + sunday_rate_25_percent_decrease * 7

  let weekend_total := friday_total + saturday_total + sunday_total
  weekend_total * 8

/-- The theorem states that the total money raised in the two-month period is $2,849,500 --/
theorem total_money_raised_is_correct : total_money_raised = 2849500 := by
  sorry

end total_money_raised_is_correct_l2567_256750


namespace intersection_when_a_is_two_complement_union_when_a_is_two_union_equals_B_iff_l2567_256722

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < 2 * a + 3}
def B : Set ℝ := {x | -2 ≤ x ∧ x ≤ 4}

-- Theorem 1: When a = 2, A ∩ B = {x | 1 < x ≤ 4}
theorem intersection_when_a_is_two :
  A 2 ∩ B = {x : ℝ | 1 < x ∧ x ≤ 4} := by sorry

-- Theorem 2: When a = 2, (Uᶜ A) ∪ (Uᶜ B) = {x | x ≤ 1 or x > 4}
theorem complement_union_when_a_is_two :
  (Set.univ \ A 2) ∪ (Set.univ \ B) = {x : ℝ | x ≤ 1 ∨ x > 4} := by sorry

-- Theorem 3: A ∪ B = B if and only if a ≤ -4 or -1 ≤ a ≤ 1/2
theorem union_equals_B_iff :
  ∀ a : ℝ, A a ∪ B = B ↔ a ≤ -4 ∨ (-1 ≤ a ∧ a ≤ 1/2) := by sorry

end intersection_when_a_is_two_complement_union_when_a_is_two_union_equals_B_iff_l2567_256722


namespace no_rational_roots_l2567_256707

theorem no_rational_roots (p q : ℤ) 
  (hp : p % 3 = 2) 
  (hq : q % 3 = 2) : 
  ¬ ∃ (r : ℚ), r^2 + p * r + q = 0 := by
sorry

end no_rational_roots_l2567_256707


namespace path_construction_cost_l2567_256753

/-- Given a rectangular grass field with surrounding path, calculate the total cost of constructing the path -/
theorem path_construction_cost 
  (field_length : ℝ) 
  (field_width : ℝ) 
  (long_side_path_width : ℝ) 
  (short_side1_path_width : ℝ) 
  (short_side2_path_width : ℝ) 
  (long_side_cost_per_sqm : ℝ) 
  (short_side1_cost_per_sqm : ℝ) 
  (short_side2_cost_per_sqm : ℝ) 
  (h1 : field_length = 75) 
  (h2 : field_width = 55) 
  (h3 : long_side_path_width = 2.5) 
  (h4 : short_side1_path_width = 3) 
  (h5 : short_side2_path_width = 4) 
  (h6 : long_side_cost_per_sqm = 7) 
  (h7 : short_side1_cost_per_sqm = 9) 
  (h8 : short_side2_cost_per_sqm = 12) :
  let long_sides_area := 2 * field_length * long_side_path_width
  let short_side1_area := field_width * short_side1_path_width
  let short_side2_area := field_width * short_side2_path_width
  let long_sides_cost := long_sides_area * long_side_cost_per_sqm
  let short_side1_cost := short_side1_area * short_side1_cost_per_sqm
  let short_side2_cost := short_side2_area * short_side2_cost_per_sqm
  let total_cost := long_sides_cost + short_side1_cost + short_side2_cost
  total_cost = 6750 := by sorry


end path_construction_cost_l2567_256753


namespace at_least_one_greater_than_seventeen_tenths_l2567_256711

theorem at_least_one_greater_than_seventeen_tenths
  (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum : a + b + c = a * b * c) :
  max a (max b c) > 17/10 := by
sorry

end at_least_one_greater_than_seventeen_tenths_l2567_256711


namespace money_split_l2567_256728

theorem money_split (donna_share : ℚ) (donna_amount : ℕ) (total : ℕ) : 
  donna_share = 5 / 17 →
  donna_amount = 35 →
  donna_share * total = donna_amount →
  total = 119 := by
sorry

end money_split_l2567_256728


namespace hexagonal_gcd_bound_hexagonal_gcd_achieves_bound_l2567_256736

def H (n : ℕ+) : ℕ := 2 * n.val ^ 2 - n.val

theorem hexagonal_gcd_bound (n : ℕ+) : Nat.gcd (3 * H n) (n.val + 1) ≤ 12 :=
sorry

theorem hexagonal_gcd_achieves_bound : ∃ n : ℕ+, Nat.gcd (3 * H n) (n.val + 1) = 12 :=
sorry

end hexagonal_gcd_bound_hexagonal_gcd_achieves_bound_l2567_256736


namespace oil_measurement_l2567_256790

/-- The total amount of oil in a measuring cup after adding more -/
theorem oil_measurement (initial : ℚ) (additional : ℚ) : 
  initial = 0.16666666666666666 →
  additional = 0.6666666666666666 →
  initial + additional = 0.8333333333333333 := by
  sorry

end oil_measurement_l2567_256790


namespace function_inequality_l2567_256721

-- Define the interval (3,7)
def openInterval : Set ℝ := {x : ℝ | 3 < x ∧ x < 7}

-- Define the theorem
theorem function_inequality
  (f g : ℝ → ℝ)
  (h_diff_f : DifferentiableOn ℝ f openInterval)
  (h_diff_g : DifferentiableOn ℝ g openInterval)
  (h_deriv : ∀ x ∈ openInterval, deriv f x < deriv g x) :
  ∀ x ∈ openInterval, f x + g 3 < g x + f 3 :=
by sorry

end function_inequality_l2567_256721


namespace marbles_on_desk_l2567_256746

theorem marbles_on_desk (desk_marbles : ℕ) : desk_marbles + 6 = 8 → desk_marbles = 2 := by
  sorry

end marbles_on_desk_l2567_256746


namespace expression_evaluation_l2567_256714

theorem expression_evaluation :
  let x : ℚ := -1/3
  (2*x - 1)^2 - (3*x + 1)*(3*x - 1) + 5*x*(x - 1) = 5 := by
  sorry

end expression_evaluation_l2567_256714


namespace ellipse_intersection_theorem_l2567_256723

/-- Ellipse C with equation (x^2 / 4) + (y^2 / m) = 1 -/
def ellipse_C (m : ℝ) (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / m = 1

/-- Point P on the x-axis -/
def point_P : ℝ × ℝ := (-1, 0)

/-- Line l passing through point P -/
def line_l (k : ℝ) (x y : ℝ) : Prop :=
  y = k * (x + 1)

/-- Condition for circle with AB as diameter passing through origin -/
def circle_condition (A B : ℝ × ℝ) : Prop :=
  A.1 * B.1 + A.2 * B.2 = 0

/-- Main theorem: If there exists a line l intersecting ellipse C at points A and B
    such that the circle with AB as diameter passes through the origin,
    then m is in the range (0, 4/3] -/
theorem ellipse_intersection_theorem (m : ℝ) :
  (m > 0) →
  (∃ (k : ℝ) (A B : ℝ × ℝ),
    ellipse_C m A.1 A.2 ∧
    ellipse_C m B.1 B.2 ∧
    line_l k A.1 A.2 ∧
    line_l k B.1 B.2 ∧
    circle_condition A B) →
  0 < m ∧ m ≤ 4/3 :=
sorry

end ellipse_intersection_theorem_l2567_256723


namespace inequality_solution_range_l2567_256734

theorem inequality_solution_range (m : ℝ) :
  (∀ x : ℝ, (m + 1) * x^2 - 2 * (m - 1) * x + 3 * (m - 1) < 0) ↔ m < -1 :=
by sorry

end inequality_solution_range_l2567_256734


namespace cosine_of_angle_between_vectors_l2567_256747

/-- Given vectors a and b in ℝ², if a + b = (5, -10) and a - b = (3, 6),
    then the cosine of the angle between a and b is 2√13/13. -/
theorem cosine_of_angle_between_vectors (a b : ℝ × ℝ) 
  (h1 : a + b = (5, -10)) 
  (h2 : a - b = (3, 6)) : 
  (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)) = 2 * Real.sqrt 13 / 13 := by
  sorry

end cosine_of_angle_between_vectors_l2567_256747


namespace sum_of_exponents_15_factorial_l2567_256700

def largest_perfect_cube_divisor (n : ℕ) : ℕ := sorry

def cube_root (n : ℕ) : ℕ := sorry

def sum_of_prime_exponents (n : ℕ) : ℕ := sorry

theorem sum_of_exponents_15_factorial : 
  sum_of_prime_exponents (cube_root (largest_perfect_cube_divisor (Nat.factorial 15))) = 6 := by sorry

end sum_of_exponents_15_factorial_l2567_256700


namespace alice_walk_distance_l2567_256763

theorem alice_walk_distance (grass_miles : ℝ) : 
  (∀ (day : Fin 5), grass_miles > 0) →  -- Alice walks a positive distance through grass each weekday
  (∀ (day : Fin 5), 12 > 0) →  -- Alice walks 12 miles through forest each weekday
  (5 * grass_miles + 5 * 12 = 110) →  -- Total weekly distance is 110 miles
  grass_miles = 10 := by
  sorry

end alice_walk_distance_l2567_256763


namespace allison_win_probability_l2567_256705

-- Define the faces of each cube
def allison_cube : Finset Nat := {4, 4, 4, 4, 4, 4}
def charlie_cube : Finset Nat := {1, 1, 2, 3, 4, 5}
def dani_cube : Finset Nat := {3, 3, 3, 3, 5, 5}

-- Define the probability of rolling each number for each person
def prob_roll (cube : Finset Nat) (n : Nat) : Rat :=
  (cube.filter (· = n)).card / cube.card

-- Define the event of Allison winning
def allison_wins (c : Nat) (d : Nat) : Prop :=
  4 > c ∧ 4 > d

-- Theorem statement
theorem allison_win_probability :
  (prob_roll charlie_cube 1 + prob_roll charlie_cube 2) *
  (prob_roll dani_cube 3) = 1/3 := by
  sorry

end allison_win_probability_l2567_256705


namespace circle_m_range_l2567_256769

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle in 2D space -/
structure Circle where
  center : Point
  radius : ℝ

/-- Checks if a point is outside a circle -/
def isOutside (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 > c.radius^2

/-- The circle equation in the form x^2 + y^2 - 2x + 1 - m = 0 -/
def circleEquation (m : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 1 - m = 0

theorem circle_m_range :
  ∀ m : ℝ,
  (∃ c : Circle, 
    (∀ x y : ℝ, circleEquation m x y ↔ (x - c.center.x)^2 + (y - c.center.y)^2 = c.radius^2) ∧
    isOutside ⟨1, 1⟩ c) →
  0 < m ∧ m < 1 :=
sorry

end circle_m_range_l2567_256769


namespace permutations_minus_combinations_l2567_256729

/-- The number of r-permutations from n elements -/
def permutations (n : ℕ) (r : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - r)

/-- The number of r-combinations from n elements -/
def combinations (n : ℕ) (r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

theorem permutations_minus_combinations : permutations 7 3 - combinations 6 4 = 195 := by
  sorry

end permutations_minus_combinations_l2567_256729


namespace cryptarithm_solution_l2567_256727

theorem cryptarithm_solution :
  ∃! (A B : ℕ), 
    A < 10 ∧ B < 10 ∧ A ≠ B ∧
    9 * (10 * A + B) = 100 * A + 10 * A + B ∧
    A = 2 ∧ B = 5 :=
by sorry

end cryptarithm_solution_l2567_256727


namespace factor_quadratic_l2567_256749

theorem factor_quadratic (a : ℝ) : a^2 - 2*a - 15 = (a + 3)*(a - 5) := by
  sorry

end factor_quadratic_l2567_256749


namespace integer_count_inequality_l2567_256781

theorem integer_count_inequality (x : ℤ) : 
  (Finset.filter (fun i => (i - 1)^2 ≤ 9) (Finset.range 7)).card = 7 := by
  sorry

end integer_count_inequality_l2567_256781


namespace endpoint_coordinate_sum_l2567_256701

/-- Given a line segment with one endpoint at (1, -3) and midpoint at (3, 5),
    the sum of the coordinates of the other endpoint is 18. -/
theorem endpoint_coordinate_sum : 
  ∀ (x y : ℝ), 
    (3 = (x + 1) / 2) →  -- Midpoint x-coordinate condition
    (5 = (y - 3) / 2) →  -- Midpoint y-coordinate condition
    x + y = 18 :=
by
  sorry

end endpoint_coordinate_sum_l2567_256701


namespace odd_function_value_at_half_l2567_256779

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_value_at_half
  (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_neg : ∀ x < 0, f x = 1 / (x + 1)) :
  f (1/2) = -2 := by
sorry

end odd_function_value_at_half_l2567_256779


namespace euler_property_division_l2567_256765

theorem euler_property_division (x : ℝ) : 
  (x > 0) →
  (1/2 * x - 3000 + 1/3 * x - 1000 + 1/4 * x + 1/5 * x + 600 = x) →
  (x = 12000 ∧ 
   1/2 * x - 3000 = 3000 ∧
   1/3 * x - 1000 = 3000 ∧
   1/4 * x = 3000 ∧
   1/5 * x + 600 = 3000) :=
by sorry

end euler_property_division_l2567_256765


namespace banana_milk_distribution_l2567_256760

/-- The amount of banana milk Hyeonju drinks in milliliters -/
def hyeonju_amount : ℕ := 1000

/-- The amount of banana milk Jinsol drinks in milliliters -/
def jinsol_amount : ℕ := hyeonju_amount + 200

/-- The amount of banana milk Changhyeok drinks in milliliters -/
def changhyeok_amount : ℕ := hyeonju_amount - 200

/-- The total amount of banana milk in milliliters -/
def total_amount : ℕ := 3000

theorem banana_milk_distribution :
  hyeonju_amount + jinsol_amount + changhyeok_amount = total_amount ∧
  jinsol_amount = hyeonju_amount + 200 ∧
  hyeonju_amount = changhyeok_amount + 200 := by
  sorry

end banana_milk_distribution_l2567_256760


namespace cliff_rock_ratio_l2567_256713

/-- Represents Cliff's rock collection -/
structure RockCollection where
  igneous : ℕ
  sedimentary : ℕ
  shinyIgneous : ℕ
  shinySedimentary : ℕ

/-- The properties of Cliff's rock collection -/
def cliffCollection : RockCollection where
  igneous := 90
  sedimentary := 180
  shinyIgneous := 30
  shinySedimentary := 36

theorem cliff_rock_ratio :
  let c := cliffCollection
  c.igneous + c.sedimentary = 270 ∧
  c.shinyIgneous = 30 ∧
  c.shinyIgneous = c.igneous / 3 ∧
  c.shinySedimentary = c.sedimentary / 5 →
  c.igneous / c.sedimentary = 1 / 2 := by
  sorry

#check cliff_rock_ratio

end cliff_rock_ratio_l2567_256713


namespace lenny_remaining_money_l2567_256786

-- Define the initial amount and expenses
def initial_amount : ℝ := 270
def console_price : ℝ := 149
def console_discount : ℝ := 0.15
def grocery_price : ℝ := 60
def grocery_discount : ℝ := 0.10
def lunch_price : ℝ := 30
def magazine_price : ℝ := 3.99

-- Define the function to calculate the remaining money
def remaining_money : ℝ :=
  initial_amount -
  (console_price * (1 - console_discount)) -
  (grocery_price * (1 - grocery_discount)) -
  lunch_price -
  magazine_price

-- Theorem to prove
theorem lenny_remaining_money :
  remaining_money = 55.36 := by sorry

end lenny_remaining_money_l2567_256786


namespace intersection_nonempty_implies_a_equals_four_l2567_256799

theorem intersection_nonempty_implies_a_equals_four :
  ∀ (a : ℝ), 
  let A : Set ℝ := {3, 4, 2*a - 3}
  let B : Set ℝ := {a}
  (A ∩ B).Nonempty → a = 4 := by
sorry

end intersection_nonempty_implies_a_equals_four_l2567_256799


namespace village_population_l2567_256706

theorem village_population (P : ℕ) : 
  (P : ℝ) * 0.9 * 0.85 = 2907 → P = 3801 := by
sorry

end village_population_l2567_256706


namespace dinner_bill_tip_percentage_l2567_256787

theorem dinner_bill_tip_percentage 
  (total_bill : ℝ)
  (num_friends : ℕ)
  (silas_payment : ℝ)
  (one_friend_payment : ℝ)
  (h1 : total_bill = 150)
  (h2 : num_friends = 6)
  (h3 : silas_payment = total_bill / 2)
  (h4 : one_friend_payment = 18)
  : (((one_friend_payment - (total_bill - silas_payment) / (num_friends - 1)) * (num_friends - 1)) / total_bill) * 100 = 10 := by
  sorry

end dinner_bill_tip_percentage_l2567_256787


namespace late_car_speed_l2567_256718

/-- Proves that given a journey of 70 km, if a car arrives on time with an average speed
    of 40 km/hr and arrives 15 minutes late with a slower speed, then the slower speed
    is 35 km/hr. -/
theorem late_car_speed (distance : ℝ) (on_time_speed : ℝ) (late_time : ℝ) :
  distance = 70 →
  on_time_speed = 40 →
  late_time = 0.25 →
  let on_time_duration := distance / on_time_speed
  let late_duration := on_time_duration + late_time
  let late_speed := distance / late_duration
  late_speed = 35 := by
  sorry

end late_car_speed_l2567_256718


namespace geometric_mean_a4_a8_l2567_256717

def geometric_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := a₁ * q^(n - 1)

theorem geometric_mean_a4_a8 :
  let a := geometric_sequence (1/8) 2
  (a 4 * a 8)^(1/2) = 4 := by sorry

end geometric_mean_a4_a8_l2567_256717


namespace natalia_clip_sales_l2567_256703

/-- The number of clips Natalia sold in April and May combined -/
def total_clips (april_sales : ℕ) (may_sales : ℕ) : ℕ :=
  april_sales + may_sales

/-- Theorem stating that given the conditions, Natalia sold 72 clips in total -/
theorem natalia_clip_sales : 
  ∀ (april_sales : ℕ) (may_sales : ℕ),
    april_sales = 48 →
    may_sales = april_sales / 2 →
    total_clips april_sales may_sales = 72 :=
by
  sorry

end natalia_clip_sales_l2567_256703


namespace quadratic_ratio_l2567_256775

/-- Given a quadratic polynomial x^2 + 1560x + 2400, prove that when written in the form (x + b)^2 + c, the ratio c/b equals -300 -/
theorem quadratic_ratio (x : ℝ) : 
  ∃ (b c : ℝ), (∀ x, x^2 + 1560*x + 2400 = (x + b)^2 + c) ∧ c/b = -300 := by
sorry

end quadratic_ratio_l2567_256775


namespace camp_total_boys_l2567_256788

structure Camp where
  totalBoys : ℕ
  schoolA : ℕ
  schoolB : ℕ
  schoolC : ℕ
  schoolAScience : ℕ
  schoolAMath : ℕ
  schoolBScience : ℕ
  schoolBEnglish : ℕ

def isValidCamp (c : Camp) : Prop :=
  c.schoolA + c.schoolB + c.schoolC = c.totalBoys ∧
  c.schoolA = c.totalBoys / 5 ∧
  c.schoolB = c.totalBoys / 4 ∧
  c.schoolC = c.totalBoys - c.schoolA - c.schoolB ∧
  c.schoolAScience = c.schoolA * 3 / 10 ∧
  c.schoolAMath = c.schoolA * 2 / 5 ∧
  c.schoolBScience = c.schoolB / 2 ∧
  c.schoolBEnglish = c.schoolB / 10 ∧
  c.schoolA - c.schoolAScience = 56 ∧
  c.schoolBEnglish = 35

theorem camp_total_boys (c : Camp) (h : isValidCamp c) : c.totalBoys = 400 := by
  sorry

end camp_total_boys_l2567_256788


namespace polygon_with_17_diagonals_has_8_sides_l2567_256761

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A regular polygon with 17 diagonals has 8 sides -/
theorem polygon_with_17_diagonals_has_8_sides :
  ∃ (n : ℕ), n > 2 ∧ num_diagonals n = 17 → n = 8 :=
by sorry

end polygon_with_17_diagonals_has_8_sides_l2567_256761


namespace renovation_sand_required_l2567_256773

/-- The amount of sand required for a renovation project -/
theorem renovation_sand_required (total_material dirt cement : ℝ) 
  (h_total : total_material = 0.67)
  (h_dirt : dirt = 0.33)
  (h_cement : cement = 0.17) :
  total_material - dirt - cement = 0.17 := by
  sorry

end renovation_sand_required_l2567_256773


namespace fahrenheit_to_celsius_l2567_256764

theorem fahrenheit_to_celsius (C F : ℝ) : C = (5 / 9) * (F - 32) → C = 40 → F = 104 := by
  sorry

end fahrenheit_to_celsius_l2567_256764


namespace complement_of_A_in_U_l2567_256798

def U : Set ℕ := {1,2,3,4,5,6}
def A : Set ℕ := {2,4,6}

theorem complement_of_A_in_U :
  (U \ A) = {1,3,5} := by sorry

end complement_of_A_in_U_l2567_256798


namespace divisibility_of_sum_of_squares_l2567_256733

theorem divisibility_of_sum_of_squares (p x y z : ℕ) : 
  Prime p → 
  0 < x → x < y → y < z → z < p → 
  x^3 % p = y^3 % p → y^3 % p = z^3 % p → 
  (x + y + z) ∣ (x^2 + y^2 + z^2) := by
sorry

end divisibility_of_sum_of_squares_l2567_256733


namespace matrix_product_is_zero_l2567_256742

def A (k a c d : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, k * d, -k * c],
    ![-k * d, 0, k * a],
    ![k * c, -k * a, 0]]

def B (d e f : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![d^2, d * e, d * f],
    ![d * e, e^2, e * f],
    ![d * f, e * f, f^2]]

theorem matrix_product_is_zero (k a c d e f : ℝ) :
  A k a c d * B d e f = 0 := by
  sorry

end matrix_product_is_zero_l2567_256742


namespace inequality_proof_l2567_256755

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a * b + b * c + c * a = 3) :
  (b * c / (1 + a^4)) + (c * a / (1 + b^4)) + (a * b / (1 + c^4)) ≥ 3/2 := by
  sorry

end inequality_proof_l2567_256755


namespace flash_catch_up_distance_l2567_256709

theorem flash_catch_up_distance
  (v a x y : ℝ) -- v: Ace's speed, a: Flash's acceleration, x: Flash's initial speed multiplier, y: initial distance behind
  (hx : x > 1)
  (ha : a > 0) :
  let d := y + x * v * (-(x - 1) * v + Real.sqrt ((x - 1)^2 * v^2 + 2 * a * y)) / a
  let t := (-(x - 1) * v + Real.sqrt ((x - 1)^2 * v^2 + 2 * a * y)) / a
  d = y + x * v * t + (1/2) * a * t^2 ∧
  d = v * t :=
by sorry

end flash_catch_up_distance_l2567_256709


namespace inequality_proof_l2567_256757

theorem inequality_proof (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : x + y + z = 1) : 
  (2 * x^2) / (y + z) + (2 * y^2) / (z + x) + (2 * z^2) / (x + y) ≥ 1 := by
  sorry

end inequality_proof_l2567_256757


namespace polynomial_evaluation_l2567_256759

theorem polynomial_evaluation (x : ℝ) (h1 : x > 0) (h2 : x^2 - 3*x - 9 = 0) :
  x^3 - 3*x^2 - 9*x + 5 = 5 := by
  sorry

end polynomial_evaluation_l2567_256759


namespace power_product_equality_l2567_256724

theorem power_product_equality (a b : ℝ) : (-a * b)^3 * (-3 * b)^2 = -9 * a^3 * b^5 := by
  sorry

end power_product_equality_l2567_256724


namespace line_parabola_intersection_count_l2567_256789

theorem line_parabola_intersection_count : 
  ∃! (s : Finset ℝ), 
    (∀ a ∈ s, ∃ x y : ℝ, 
      y = 2*x + a + 1 ∧ 
      y = x^2 + (a+1)^2 ∧ 
      ∀ x' : ℝ, x'^2 + (a+1)^2 ≥ x^2 + (a+1)^2) ∧
    s.card = 2 := by
  sorry

end line_parabola_intersection_count_l2567_256789


namespace count_minimally_intersecting_mod_1000_l2567_256712

def Universe : Finset Nat := {1,2,3,4,5,6,7,8}

def MinimallyIntersecting (D E F : Finset Nat) : Prop :=
  (D ∩ E).card = 1 ∧ (E ∩ F).card = 1 ∧ (F ∩ D).card = 1 ∧ (D ∩ E ∩ F).card = 0

def CountMinimallyIntersecting : Nat :=
  (Finset.powerset Universe).card.choose 3

theorem count_minimally_intersecting_mod_1000 :
  CountMinimallyIntersecting % 1000 = 64 := by sorry

end count_minimally_intersecting_mod_1000_l2567_256712


namespace average_of_xyz_is_one_l2567_256738

theorem average_of_xyz_is_one (x y z : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0)
  (h_diff : x ≠ y ∧ y ≠ z ∧ x ≠ z)
  (h_prod : x * y * z = 1)
  (h_sum : x + y + z = 1/x + 1/y + 1/z) :
  (x + y + z) / 3 = 1 := by
sorry

end average_of_xyz_is_one_l2567_256738


namespace hexagon_area_is_52_l2567_256720

-- Define the hexagon vertices
def hexagon_vertices : List (ℝ × ℝ) := [
  (0, 0), (2, 4), (5, 4), (7, 0), (5, -4), (2, -4)
]

-- Function to calculate the area of a trapezoid given its four vertices
def trapezoid_area (v1 v2 v3 v4 : ℝ × ℝ) : ℝ := sorry

-- Function to calculate the area of the hexagon
def hexagon_area (vertices : List (ℝ × ℝ)) : ℝ := sorry

-- Theorem stating that the area of the hexagon is 52 square units
theorem hexagon_area_is_52 : hexagon_area hexagon_vertices = 52 := by sorry

end hexagon_area_is_52_l2567_256720


namespace circus_kids_l2567_256725

theorem circus_kids (total_cost : ℕ) (kid_ticket_cost : ℕ) (num_adults : ℕ) : 
  total_cost = 50 →
  kid_ticket_cost = 5 →
  num_adults = 2 →
  ∃ (num_kids : ℕ), 
    (num_kids * kid_ticket_cost + num_adults * (2 * kid_ticket_cost) = total_cost) ∧
    num_kids = 2 := by
  sorry

end circus_kids_l2567_256725


namespace consecutive_integers_around_sqrt_seven_l2567_256704

theorem consecutive_integers_around_sqrt_seven (a b : ℤ) : 
  (b = a + 1) → (a < Real.sqrt 7) → (Real.sqrt 7 < b) → (a + b = 5) := by
  sorry

end consecutive_integers_around_sqrt_seven_l2567_256704


namespace cone_volume_from_half_sector_l2567_256776

/-- The volume of a right circular cone formed by rolling up a half-sector of a circle -/
theorem cone_volume_from_half_sector (r : ℝ) (h : r = 6) :
  let base_radius : ℝ := r / 2
  let height : ℝ := r * Real.sqrt 3 / 2
  (1 / 3) * Real.pi * base_radius^2 * height = 9 * Real.pi * Real.sqrt 3 := by
  sorry

end cone_volume_from_half_sector_l2567_256776


namespace min_value_expression_l2567_256772

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a^2 * b + b^2 * c + c^2 * a = 3) :
  let A := (Real.sqrt (a^6 + b^4 * c^6)) / b + 
           (Real.sqrt (b^6 + c^4 * a^6)) / c + 
           (Real.sqrt (c^6 + a^4 * b^6)) / a
  A ≥ 3 * Real.sqrt 2 :=
by sorry

end min_value_expression_l2567_256772


namespace employee_payments_l2567_256702

theorem employee_payments (total_payment : ℕ) (base_c : ℕ) (commission_c : ℕ) :
  total_payment = 2000 ∧
  base_c = 400 ∧
  commission_c = 100 →
  ∃ (payment_a payment_b payment_c : ℕ),
    payment_a = (3 * payment_b) / 2 ∧
    payment_c = base_c + commission_c ∧
    payment_a + payment_b + payment_c = total_payment ∧
    payment_a = 900 ∧
    payment_b = 600 ∧
    payment_c = 500 :=
by
  sorry


end employee_payments_l2567_256702


namespace ratio_problem_l2567_256731

theorem ratio_problem (a b c : ℚ) : 
  b / a = 4 → 
  b = 18 - 7 * a → 
  c = 2 * a - 6 → 
  a = 18 / 11 ∧ c = -30 / 11 := by
  sorry

end ratio_problem_l2567_256731


namespace common_divisors_of_36_and_60_l2567_256766

/-- The number of positive integers that are divisors of both 36 and 60 -/
def common_divisors_count : ℕ := 
  (Finset.filter (fun d => 36 % d = 0 ∧ 60 % d = 0) (Finset.range 61)).card

/-- Theorem stating that the number of common divisors of 36 and 60 is 6 -/
theorem common_divisors_of_36_and_60 : common_divisors_count = 6 := by
  sorry

end common_divisors_of_36_and_60_l2567_256766


namespace high_school_total_students_l2567_256797

/-- Represents a high school with three grades and stratified sampling -/
structure HighSchool where
  total_students : ℕ
  freshmen : ℕ
  sample_size : ℕ
  sampled_sophomores : ℕ
  sampled_seniors : ℕ

/-- The conditions of the problem -/
def problem_conditions (hs : HighSchool) : Prop :=
  hs.freshmen = 600 ∧
  hs.sample_size = 45 ∧
  hs.sampled_sophomores = 20 ∧
  hs.sampled_seniors = 10

/-- The theorem to prove -/
theorem high_school_total_students (hs : HighSchool) 
  (h : problem_conditions hs) : 
  hs.total_students = 1800 :=
sorry

end high_school_total_students_l2567_256797


namespace sum_of_x_and_y_l2567_256785

theorem sum_of_x_and_y (x y : ℚ) 
  (h1 : 2 / x + 3 / y = 4) 
  (h2 : 2 / x - 3 / y = -2) : 
  x + y = 3 := by
sorry

end sum_of_x_and_y_l2567_256785


namespace triangle_with_ratio_1_2_3_is_right_triangle_l2567_256754

-- Define a triangle type
structure Triangle :=
  (angle1 : ℝ)
  (angle2 : ℝ)
  (angle3 : ℝ)

-- Define the properties of a valid triangle
def is_valid_triangle (t : Triangle) : Prop :=
  t.angle1 > 0 ∧ t.angle2 > 0 ∧ t.angle3 > 0 ∧ t.angle1 + t.angle2 + t.angle3 = 180

-- Define a right triangle
def is_right_triangle (t : Triangle) : Prop :=
  t.angle1 = 90 ∨ t.angle2 = 90 ∨ t.angle3 = 90

-- Define a triangle with angles in the ratio 1:2:3
def triangle_with_ratio_1_2_3 (t : Triangle) : Prop :=
  ∃ k : ℝ, k > 0 ∧ t.angle1 = k ∧ t.angle2 = 2*k ∧ t.angle3 = 3*k

-- Theorem statement
theorem triangle_with_ratio_1_2_3_is_right_triangle (t : Triangle) :
  is_valid_triangle t → triangle_with_ratio_1_2_3 t → is_right_triangle t :=
by sorry

end triangle_with_ratio_1_2_3_is_right_triangle_l2567_256754


namespace consecutive_integers_product_divisibility_l2567_256794

theorem consecutive_integers_product_divisibility
  (m n : ℕ) (h : m < n) :
  ∀ (a : ℕ), ∃ (i j : ℕ), i ≠ j ∧ i < n ∧ j < n ∧ (mn ∣ (a + i) * (a + j)) :=
by sorry

end consecutive_integers_product_divisibility_l2567_256794


namespace farm_problem_l2567_256795

theorem farm_problem :
  ∃ (l g : ℕ), l > 0 ∧ g > 0 ∧ 30 * l + 32 * g = 1200 ∧ l > g :=
by sorry

end farm_problem_l2567_256795


namespace xiaoying_final_score_l2567_256791

/-- Calculates the weighted sum of scores given the scores and weights -/
def weightedSum (scores : List ℝ) (weights : List ℝ) : ℝ :=
  List.sum (List.zipWith (· * ·) scores weights)

/-- Xiaoying's speech competition scores -/
def speechScores : List ℝ := [86, 90, 80]

/-- Weights for each category in the speech competition -/
def categoryWeights : List ℝ := [0.5, 0.4, 0.1]

/-- Theorem stating that Xiaoying's final score is 87 -/
theorem xiaoying_final_score :
  weightedSum speechScores categoryWeights = 87 := by
  sorry

end xiaoying_final_score_l2567_256791


namespace quadratic_inequality_properties_l2567_256782

-- Define the quadratic function
def f (a b c x : ℝ) := a * x^2 + b * x + c

-- Define the solution set of the quadratic inequality
def solution_set (a b c : ℝ) := {x : ℝ | f a b c x < 0}

-- State the theorem
theorem quadratic_inequality_properties
  (a b c : ℝ)
  (h : solution_set a b c = {x : ℝ | x < -4 ∨ x > 3}) :
  (a + b + c > 0) ∧
  ({x : ℝ | (a * x - b) / (a * x - c) ≤ 0} = {x : ℝ | -12 < x ∧ x ≤ 1}) :=
by sorry

end quadratic_inequality_properties_l2567_256782


namespace term_2005_is_334th_l2567_256732

-- Define the arithmetic sequence
def arithmeticSequence (n : ℕ) : ℕ := 7 + 6 * (n - 1)

-- State the theorem
theorem term_2005_is_334th :
  arithmeticSequence 334 = 2005 := by
  sorry

end term_2005_is_334th_l2567_256732


namespace second_car_distance_l2567_256739

-- Define the initial distance between the cars
def initial_distance : ℝ := 150

-- Define the final distance between the cars
def final_distance : ℝ := 65

-- Theorem to prove
theorem second_car_distance :
  ∃ (x : ℝ), x ≥ 0 ∧ initial_distance - x = final_distance ∧ x = 85 :=
by sorry

end second_car_distance_l2567_256739


namespace expression_equals_25_l2567_256735

theorem expression_equals_25 (x y : ℝ) (h1 : x + y = 25) (h2 : x * y = 1) :
  x + y = 25 := by sorry

end expression_equals_25_l2567_256735


namespace farmer_chicken_sales_l2567_256796

def duck_price : ℕ := 10
def chicken_price : ℕ := 8
def ducks_sold : ℕ := 2

theorem farmer_chicken_sales : 
  ∃ (chickens_sold : ℕ),
    (duck_price * ducks_sold + chicken_price * chickens_sold) / 2 = 30 ∧
    chickens_sold = 5 := by
  sorry

end farmer_chicken_sales_l2567_256796


namespace parallel_segments_k_value_l2567_256768

/-- Given four points A(-3, 0), B(0, -3), X(0, 9), and Y(18, k) on a Cartesian plane,
    if segment AB is parallel to segment XY, then k = -9. -/
theorem parallel_segments_k_value (k : ℝ) : 
  let A : ℝ × ℝ := (-3, 0)
  let B : ℝ × ℝ := (0, -3)
  let X : ℝ × ℝ := (0, 9)
  let Y : ℝ × ℝ := (18, k)
  (B.2 - A.2) / (B.1 - A.1) = (Y.2 - X.2) / (Y.1 - X.1) →
  k = -9 := by
sorry

end parallel_segments_k_value_l2567_256768


namespace tablet_price_after_discounts_l2567_256726

theorem tablet_price_after_discounts (original_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) :
  original_price = 250 ∧ discount1 = 0.30 ∧ discount2 = 0.25 →
  original_price * (1 - discount1) * (1 - discount2) = 131.25 := by
  sorry

end tablet_price_after_discounts_l2567_256726


namespace min_value_of_f_l2567_256793

theorem min_value_of_f (a₁ a₂ a₃ a₄ : ℝ) 
  (pos₁ : 0 < a₁) (pos₂ : 0 < a₂) (pos₃ : 0 < a₃) (pos₄ : 0 < a₄)
  (sum_cond : a₁ + 2*a₂ + 3*a₃ + 4*a₄ ≤ 10)
  (lower_bound₁ : a₁ ≥ 1/8) (lower_bound₂ : a₂ ≥ 1/4)
  (lower_bound₃ : a₃ ≥ 1/2) (lower_bound₄ : a₄ ≥ 1) : 
  1/(1 + a₁) + 1/(1 + a₂^2) + 1/(1 + a₃^3) + 1/(1 + a₄^4) ≥ 2 := by
  sorry

end min_value_of_f_l2567_256793


namespace rice_price_calculation_l2567_256710

def initial_amount : ℝ := 500
def wheat_flour_price : ℝ := 25
def wheat_flour_quantity : ℕ := 3
def soda_price : ℝ := 150
def soda_quantity : ℕ := 1
def rice_quantity : ℕ := 2
def remaining_balance : ℝ := 235

theorem rice_price_calculation : 
  ∃ (rice_price : ℝ), 
    initial_amount - 
    (rice_price * rice_quantity + 
     wheat_flour_price * wheat_flour_quantity + 
     soda_price * soda_quantity) = remaining_balance ∧ 
    rice_price = 20 := by
  sorry

end rice_price_calculation_l2567_256710


namespace cubic_yards_to_cubic_feet_l2567_256743

-- Define the conversion factor from yards to feet
def yards_to_feet : ℝ := 3

-- Define the volume in cubic yards
def volume_cubic_yards : ℝ := 7

-- Theorem: 7 cubic yards are equal to 189 cubic feet
theorem cubic_yards_to_cubic_feet :
  (volume_cubic_yards * yards_to_feet ^ 3 : ℝ) = 189 :=
by sorry

end cubic_yards_to_cubic_feet_l2567_256743


namespace triangle_angle_measure_l2567_256783

theorem triangle_angle_measure (D E F : ℝ) :
  D + E + F = 180 →  -- Sum of angles in a triangle
  F = D + 40 →       -- Angle F is 40 degrees more than angle D
  E = 2 * D →        -- Angle E is twice the measure of angle D
  F = 75 :=          -- Measure of angle F is 75 degrees
by
  sorry

end triangle_angle_measure_l2567_256783


namespace value_of_x_l2567_256778

theorem value_of_x (x y z : ℝ) 
  (h1 : x = (1/2) * y) 
  (h2 : y = (1/4) * z) 
  (h3 : z = 80) : 
  x = 10 := by
sorry

end value_of_x_l2567_256778


namespace quadratic_roots_sum_l2567_256716

theorem quadratic_roots_sum (a b : ℝ) : 
  a^2 - 4*a - 1 = 0 → b^2 - 4*b - 1 = 0 → 2*a^2 + 3/b + 5*b = 22 := by
  sorry

end quadratic_roots_sum_l2567_256716


namespace sugar_used_in_two_minutes_l2567_256715

/-- Calculates the total sugar used in chocolate production over a given time period. -/
def sugarUsed (sugarPerBar : ℝ) (barsPerMinute : ℕ) (minutes : ℕ) : ℝ :=
  sugarPerBar * (barsPerMinute : ℝ) * (minutes : ℝ)

/-- Theorem stating that given the specified production parameters, 
    the total sugar used in two minutes is 108 grams. -/
theorem sugar_used_in_two_minutes :
  sugarUsed 1.5 36 2 = 108 := by
  sorry

end sugar_used_in_two_minutes_l2567_256715


namespace prism_height_to_base_ratio_l2567_256740

/-- 
For a regular quadrangular prism where a plane passes through the diagonal 
of the lower base and the opposite vertex of the upper base, forming a 
cross-section with angle α between its equal sides, the ratio of the prism's 
height to the side length of its base is (√(2 cos α)) / (2 sin(α/2)).
-/
theorem prism_height_to_base_ratio (α : Real) : 
  let h := Real.sqrt (2 * Real.cos α) / (2 * Real.sin (α / 2))
  let a := 1  -- Assuming unit side length for simplicity
  (h : Real) = (Real.sqrt (2 * Real.cos α)) / (2 * Real.sin (α / 2)) := by
  sorry

end prism_height_to_base_ratio_l2567_256740


namespace cube_piercing_theorem_l2567_256792

/-- Represents a brick with dimensions 2 × 2 × 1 -/
structure Brick :=
  (x : ℕ) (y : ℕ) (z : ℕ)

/-- Represents a cube constructed from bricks -/
structure Cube :=
  (size : ℕ)
  (bricks : List Brick)

/-- Represents a line perpendicular to a face of the cube -/
structure PerpLine :=
  (x : ℕ) (y : ℕ) (face : Nat)

/-- Function to check if a line intersects a brick -/
def intersects (l : PerpLine) (b : Brick) : Prop := sorry

/-- Theorem stating that there exists a line not intersecting any brick -/
theorem cube_piercing_theorem (c : Cube) 
  (h1 : c.size = 20) 
  (h2 : c.bricks.length = 2000) 
  (h3 : ∀ b ∈ c.bricks, b.x = 2 ∧ b.y = 2 ∧ b.z = 1) :
  ∃ l : PerpLine, ∀ b ∈ c.bricks, ¬(intersects l b) := by sorry

end cube_piercing_theorem_l2567_256792


namespace perpendicular_lines_l2567_256748

theorem perpendicular_lines (a : ℝ) : 
  (∃ (x y : ℝ), x + a * y - a = 0 ∧ a * x - (2 * a - 3) * y - 1 = 0) →
  ((-1 : ℝ) / a) * (a / (2 * a - 3)) = -1 →
  a = 0 ∨ a = 2 := by
sorry

end perpendicular_lines_l2567_256748


namespace stratified_sampling_middle_aged_l2567_256741

theorem stratified_sampling_middle_aged (total_teachers : ℕ) (middle_aged : ℕ) (sample_size : ℕ)
  (h1 : total_teachers = 480)
  (h2 : middle_aged = 160)
  (h3 : sample_size = 60) :
  (middle_aged : ℚ) / total_teachers * sample_size = 20 := by
  sorry

end stratified_sampling_middle_aged_l2567_256741


namespace starting_lineup_count_l2567_256719

def team_size : ℕ := 12
def center_capable : ℕ := 4

def starting_lineup_combinations : ℕ :=
  center_capable * (team_size - 1) * (team_size - 2) * (team_size - 3)

theorem starting_lineup_count :
  starting_lineup_combinations = 3960 := by
  sorry

end starting_lineup_count_l2567_256719


namespace pig_farm_fence_length_l2567_256752

theorem pig_farm_fence_length 
  (area : ℝ) 
  (short_side : ℝ) 
  (long_side : ℝ) :
  area = 1250 ∧ 
  long_side = 2 * short_side ∧ 
  area = long_side * short_side →
  short_side + short_side + long_side = 100 := by
sorry

end pig_farm_fence_length_l2567_256752


namespace smallest_binary_divisible_by_225_proof_l2567_256744

/-- A function that checks if a natural number only contains digits 0 and 1 in base 10 -/
def only_zero_one_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 0 ∨ d = 1

/-- The smallest natural number with only 0 and 1 digits divisible by 225 -/
def smallest_binary_divisible_by_225 : ℕ := 11111111100

theorem smallest_binary_divisible_by_225_proof :
  (smallest_binary_divisible_by_225 % 225 = 0) ∧
  only_zero_one_digits smallest_binary_divisible_by_225 ∧
  ∀ n : ℕ, n < smallest_binary_divisible_by_225 →
    ¬(n % 225 = 0 ∧ only_zero_one_digits n) :=
by sorry

end smallest_binary_divisible_by_225_proof_l2567_256744


namespace prob_C_is_one_fourth_l2567_256767

/-- A game spinner with four regions A, B, C, and D -/
structure Spinner :=
  (probA : ℚ)
  (probB : ℚ)
  (probC : ℚ)
  (probD : ℚ)

/-- The probability of all regions in a spinner sum to 1 -/
def valid_spinner (s : Spinner) : Prop :=
  s.probA + s.probB + s.probC + s.probD = 1

/-- Theorem: Given a valid spinner with probA = 1/4, probB = 1/3, and probD = 1/6, 
    the probability of region C is 1/4 -/
theorem prob_C_is_one_fourth (s : Spinner) 
  (h_valid : valid_spinner s)
  (h_probA : s.probA = 1/4)
  (h_probB : s.probB = 1/3)
  (h_probD : s.probD = 1/6) :
  s.probC = 1/4 := by
  sorry

end prob_C_is_one_fourth_l2567_256767


namespace chocolate_bar_count_l2567_256751

/-- The number of small boxes in the large box -/
def small_boxes : ℕ := 15

/-- The number of chocolate bars in each small box -/
def bars_per_box : ℕ := 25

/-- The total number of chocolate bars in the large box -/
def total_bars : ℕ := small_boxes * bars_per_box

theorem chocolate_bar_count : total_bars = 375 := by
  sorry

end chocolate_bar_count_l2567_256751


namespace train_passes_jogger_l2567_256784

/-- The time it takes for a train to pass a jogger given their speeds and initial positions -/
theorem train_passes_jogger (jogger_speed train_speed : ℝ) (train_length initial_distance : ℝ) : 
  jogger_speed = 9 * (1000 / 3600) →
  train_speed = 45 * (1000 / 3600) →
  train_length = 120 →
  initial_distance = 230 →
  (initial_distance + train_length) / (train_speed - jogger_speed) = 35 :=
by sorry

end train_passes_jogger_l2567_256784


namespace ryan_final_tokens_l2567_256708

def initial_tokens : ℕ := 36
def pacman_fraction : ℚ := 1/3
def candy_crush_fraction : ℚ := 1/4
def skiball_tokens : ℕ := 7
def parent_multiplier : ℕ := 7

theorem ryan_final_tokens :
  let pacman_tokens := (pacman_fraction * initial_tokens).floor
  let candy_crush_tokens := (candy_crush_fraction * initial_tokens).floor
  let total_spent := pacman_tokens + candy_crush_tokens + skiball_tokens
  let tokens_left := initial_tokens - total_spent
  let parent_bought := parent_multiplier * skiball_tokens
  tokens_left + parent_bought = 57 := by
  sorry

end ryan_final_tokens_l2567_256708


namespace remainder_theorem_l2567_256770

-- Define the polynomial q(x)
def q (x : ℝ) (D : ℝ) : ℝ := 2 * x^6 - 3 * x^4 + D * x^2 + 6

-- State the theorem
theorem remainder_theorem (D : ℝ) :
  q 2 D = 14 → q (-2) D = 158 := by
  sorry

end remainder_theorem_l2567_256770


namespace sqrt_x_plus_one_meaningful_l2567_256758

theorem sqrt_x_plus_one_meaningful (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x + 1) ↔ x ≥ -1 :=
sorry

end sqrt_x_plus_one_meaningful_l2567_256758


namespace product_difference_of_squares_l2567_256774

theorem product_difference_of_squares (m n : ℝ) : 
  m * n = ((m + n)/2)^2 - ((m - n)/2)^2 := by sorry

end product_difference_of_squares_l2567_256774
