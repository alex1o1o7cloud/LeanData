import Mathlib

namespace angle_C_is_30_l3138_313863

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)

-- Define the property that the sum of angles in a triangle is 180°
axiom triangle_angle_sum (t : Triangle) : t.A + t.B + t.C = 180

-- Theorem: If the sum of angles A and B in triangle ABC is 150°, then angle C is 30°
theorem angle_C_is_30 (t : Triangle) (h : t.A + t.B = 150) : t.C = 30 := by
  sorry


end angle_C_is_30_l3138_313863


namespace height_difference_petronas_empire_l3138_313811

/-- The height difference between two buildings -/
def height_difference (h1 h2 : ℝ) : ℝ := |h1 - h2|

/-- The Empire State Building is 443 m tall -/
def empire_state_height : ℝ := 443

/-- The Petronas Towers is 452 m tall -/
def petronas_towers_height : ℝ := 452

/-- Theorem: The height difference between the Petronas Towers and the Empire State Building is 9 meters -/
theorem height_difference_petronas_empire : 
  height_difference petronas_towers_height empire_state_height = 9 := by
  sorry

end height_difference_petronas_empire_l3138_313811


namespace quadratic_roots_range_l3138_313896

/-- Given a quadratic equation x^2 - 2mx + 4 = 0 where m is a real number,
    if both of its real roots are greater than 1, then m is in the range [2, 5/2). -/
theorem quadratic_roots_range (m : ℝ) : 
  (∀ x : ℝ, x^2 - 2*m*x + 4 = 0 → x > 1) → 
  m ∈ Set.Icc 2 (5/2) :=
by
  sorry

end quadratic_roots_range_l3138_313896


namespace womens_average_age_l3138_313883

theorem womens_average_age 
  (n : ℕ) 
  (initial_avg : ℝ) 
  (age1 age2 : ℕ) 
  (new_avg_increase : ℝ) :
  n = 8 →
  age1 = 20 →
  age2 = 28 →
  new_avg_increase = 2 →
  (n * initial_avg - (age1 + age2 : ℝ) + 2 * ((n * initial_avg + n * new_avg_increase - n * initial_avg + age1 + age2) / 2)) / n = initial_avg + new_avg_increase →
  ((n * initial_avg + n * new_avg_increase - n * initial_avg + age1 + age2) / 2) / 2 = 32 :=
by sorry

end womens_average_age_l3138_313883


namespace greatest_integer_in_odd_set_l3138_313875

def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

def is_consecutive_odd_set (s : Set ℤ) : Prop :=
  ∃ a b : ℤ, a ≤ b ∧ s = {n : ℤ | a ≤ n ∧ n ≤ b ∧ is_odd n ∧ ∀ m : ℤ, a ≤ m ∧ m < n → is_odd m}

def median (s : Set ℤ) : ℤ := sorry

theorem greatest_integer_in_odd_set (s : Set ℤ) :
  is_consecutive_odd_set s →
  155 ∈ s →
  median s = 167 →
  ∃ m : ℤ, m ∈ s ∧ ∀ n ∈ s, n ≤ m ∧ m = 179 :=
sorry

end greatest_integer_in_odd_set_l3138_313875


namespace antoinette_weight_l3138_313834

theorem antoinette_weight (rupert_weight : ℝ) : 
  let antoinette_weight := 2 * rupert_weight - 7
  (antoinette_weight + rupert_weight = 98) → antoinette_weight = 63 := by
sorry

end antoinette_weight_l3138_313834


namespace cans_ratio_theorem_l3138_313839

/-- Represents the number of cans collected by each person -/
structure CansCollected where
  solomon : ℕ
  juwan : ℕ
  levi : ℕ

/-- Represents a ratio between two numbers -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- The theorem to be proved -/
theorem cans_ratio_theorem (c : CansCollected) 
  (h1 : c.solomon = 66)
  (h2 : c.solomon + c.juwan + c.levi = 99)
  (h3 : c.levi = c.juwan / 2)
  : Ratio.mk 3 1 = Ratio.mk c.solomon c.juwan := by
  sorry

#check cans_ratio_theorem

end cans_ratio_theorem_l3138_313839


namespace yogurt_milk_calculation_l3138_313874

/-- The cost of milk per liter in dollars -/
def milk_cost : ℚ := 3/2

/-- The cost of fruit per kilogram in dollars -/
def fruit_cost : ℚ := 2

/-- The amount of fruit needed for one batch of yogurt in kilograms -/
def fruit_per_batch : ℚ := 3

/-- The total cost to produce three batches of yogurt in dollars -/
def total_cost_three_batches : ℚ := 63

/-- The number of liters of milk needed for one batch of yogurt -/
def milk_per_batch : ℚ := 10

theorem yogurt_milk_calculation :
  milk_per_batch * milk_cost * 3 + fruit_per_batch * fruit_cost * 3 = total_cost_three_batches :=
sorry

end yogurt_milk_calculation_l3138_313874


namespace smallest_dual_representation_l3138_313849

/-- Represents a number in a given base -/
def represent_in_base (n : ℕ) (base : ℕ) : List ℕ := sorry

/-- Converts a number from a given base to base 10 -/
def to_base_10 (digits : List ℕ) (base : ℕ) : ℕ := sorry

/-- Checks if a number can be represented as 13 in base c and 31 in base d -/
def is_valid_representation (n : ℕ) (c : ℕ) (d : ℕ) : Prop :=
  (represent_in_base n c = [1, 3]) ∧ (represent_in_base n d = [3, 1])

theorem smallest_dual_representation :
  ∃ (n : ℕ) (c : ℕ) (d : ℕ),
    c > 3 ∧ d > 3 ∧
    is_valid_representation n c d ∧
    (∀ (m : ℕ) (c' : ℕ) (d' : ℕ),
      c' > 3 → d' > 3 → is_valid_representation m c' d' → n ≤ m) ∧
    n = 13 := by sorry

#check smallest_dual_representation

end smallest_dual_representation_l3138_313849


namespace roots_product_l3138_313818

theorem roots_product (p q : ℝ) : 
  (p - 3) * (3 * p + 8) = p^2 - 17 * p + 56 →
  (q - 3) * (3 * q + 8) = q^2 - 17 * q + 56 →
  p ≠ q →
  (p + 2) * (q + 2) = -60 := by
sorry

end roots_product_l3138_313818


namespace total_sleep_time_in_week_l3138_313822

/-- The number of hours a cougar sleeps per night -/
def cougar_sleep_hours : ℕ := 4

/-- The additional hours a zebra sleeps compared to a cougar -/
def zebra_extra_sleep : ℕ := 2

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- Theorem: The total sleep time for a cougar and a zebra in one week is 70 hours -/
theorem total_sleep_time_in_week : 
  (cougar_sleep_hours * days_in_week) + 
  ((cougar_sleep_hours + zebra_extra_sleep) * days_in_week) = 70 := by
  sorry

end total_sleep_time_in_week_l3138_313822


namespace time_before_second_rewind_is_45_l3138_313868

/-- Represents the movie watching scenario with rewinds -/
structure MovieWatching where
  totalTime : ℕ
  initialWatchTime : ℕ
  firstRewindTime : ℕ
  secondRewindTime : ℕ
  finalWatchTime : ℕ

/-- Calculates the time watched before the second rewind -/
def timeBeforeSecondRewind (m : MovieWatching) : ℕ :=
  m.totalTime - (m.initialWatchTime + m.firstRewindTime + m.secondRewindTime + m.finalWatchTime)

/-- Theorem stating the time watched before the second rewind is 45 minutes -/
theorem time_before_second_rewind_is_45 (m : MovieWatching)
    (h1 : m.totalTime = 120)
    (h2 : m.initialWatchTime = 35)
    (h3 : m.firstRewindTime = 5)
    (h4 : m.secondRewindTime = 15)
    (h5 : m.finalWatchTime = 20) :
    timeBeforeSecondRewind m = 45 := by
  sorry

end time_before_second_rewind_is_45_l3138_313868


namespace base_10_678_to_base_7_l3138_313852

/-- Converts a base-10 integer to its base-7 representation -/
def toBase7 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base-7 to a base-10 integer -/
def fromBase7 (digits : List ℕ) : ℕ :=
  sorry

theorem base_10_678_to_base_7 :
  toBase7 678 = [1, 6, 5, 6] ∧ fromBase7 [1, 6, 5, 6] = 678 := by
  sorry

end base_10_678_to_base_7_l3138_313852


namespace unique_spicy_pair_l3138_313824

/-- A three-digit number is spicy if it equals the sum of the cubes of its digits. -/
def IsSpicy (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧ 
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  n = a^3 + b^3 + c^3

/-- 370 is the unique three-digit number n such that both n and n+1 are spicy. -/
theorem unique_spicy_pair : ∀ n : ℕ, (IsSpicy n ∧ IsSpicy (n + 1)) ↔ n = 370 := by
  sorry

end unique_spicy_pair_l3138_313824


namespace number_of_shortest_paths_is_54_l3138_313856

/-- Represents a point on the grid -/
structure GridPoint where
  x : ℕ
  y : ℕ

/-- Represents the grid configuration -/
structure Grid where
  squareSize : ℕ  -- Side length of each square in km
  refuelDistance : ℕ  -- Distance the car can travel before refueling in km

/-- Calculates the number of shortest paths between two points on the grid -/
def numberOfShortestPaths (g : Grid) (start finish : GridPoint) : ℕ :=
  sorry

/-- The specific grid configuration for the problem -/
def problemGrid : Grid :=
  { squareSize := 10
  , refuelDistance := 30 }

/-- The start point A -/
def pointA : GridPoint :=
  { x := 0, y := 0 }

/-- The end point B -/
def pointB : GridPoint :=
  { x := 6, y := 6 }  -- Assuming a 6x6 grid based on the problem description

theorem number_of_shortest_paths_is_54 :
  numberOfShortestPaths problemGrid pointA pointB = 54 :=
by sorry

end number_of_shortest_paths_is_54_l3138_313856


namespace ratio_problem_l3138_313850

theorem ratio_problem (x y : ℚ) (h : (3 * x - 2 * y) / (x + 2 * y) = 4 / 5) : 
  x / y = 18 / 11 := by sorry

end ratio_problem_l3138_313850


namespace product_abc_l3138_313800

theorem product_abc (a b c : ℕ+) (h : a * b^3 = 180) : a * b * c = 60 * c := by
  sorry

end product_abc_l3138_313800


namespace minutkin_bedtime_l3138_313829

/-- Represents the number of minutes in a day -/
def minutes_per_day : ℕ := 24 * 60

/-- Represents the time Minutkin winds his watch in the morning (8:30 AM) -/
def morning_wind_time : ℕ := 8 * 60 + 30

/-- Represents the number of full turns Minutkin makes in the morning -/
def morning_turns : ℕ := 9

/-- Represents the number of full turns Minutkin makes at night -/
def night_turns : ℕ := 11

/-- Represents the total number of turns in a day -/
def total_turns : ℕ := morning_turns + night_turns

/-- Theorem stating that Minutkin goes to bed at 9:42 PM -/
theorem minutkin_bedtime :
  ∃ (bedtime : ℕ),
    bedtime = (minutes_per_day + morning_wind_time - (morning_turns * minutes_per_day / total_turns)) % minutes_per_day ∧
    bedtime = 21 * 60 + 42 :=
by sorry

end minutkin_bedtime_l3138_313829


namespace max_min_constrained_optimization_l3138_313880

-- Define the constraint function
def constraint (x y : ℝ) : Prop :=
  Real.sqrt (x - 2) + Real.sqrt (y - 3) = 3

-- Define the objective function
def objective (x y : ℝ) : ℝ :=
  x - 2*y

-- Theorem statement
theorem max_min_constrained_optimization :
  ∃ (x_max y_max x_min y_min : ℝ),
    constraint x_max y_max ∧
    constraint x_min y_min ∧
    (∀ x y, constraint x y → objective x y ≤ objective x_max y_max) ∧
    (∀ x y, constraint x y → objective x_min y_min ≤ objective x y) ∧
    x_max = 11 ∧ y_max = 3 ∧
    x_min = 2 ∧ y_min = 12 ∧
    objective x_max y_max = 5 ∧
    objective x_min y_min = -22 :=
  sorry

end max_min_constrained_optimization_l3138_313880


namespace salad_dressing_weight_l3138_313858

/-- Calculates the total weight of a salad dressing mixture --/
theorem salad_dressing_weight (bowl_capacity : ℝ) (oil_fraction vinegar_fraction : ℝ)
  (oil_density vinegar_density : ℝ) :
  bowl_capacity = 150 ∧
  oil_fraction = 2/3 ∧
  vinegar_fraction = 1/3 ∧
  oil_density = 5 ∧
  vinegar_density = 4 →
  bowl_capacity * oil_fraction * oil_density +
  bowl_capacity * vinegar_fraction * vinegar_density = 700 := by
  sorry

end salad_dressing_weight_l3138_313858


namespace intersection_of_B_and_complement_of_A_l3138_313832

def A : Set ℝ := {x | x^2 ≤ 3}
def B : Set ℝ := {-2, -1, 1, 2}

theorem intersection_of_B_and_complement_of_A : B ∩ (Set.univ \ A) = {-2, 2} := by sorry

end intersection_of_B_and_complement_of_A_l3138_313832


namespace least_positive_t_for_geometric_progression_l3138_313861

open Real

theorem least_positive_t_for_geometric_progression :
  ∃ (t : ℝ) (α : ℝ),
    0 < α ∧ α < π / 2 ∧
    (∃ (r : ℝ),
      arcsin (sin α) * r = arcsin (sin (3 * α)) ∧
      arcsin (sin (3 * α)) * r = arcsin (sin (5 * α)) ∧
      arcsin (sin (5 * α)) * r = arcsin (sin (t * α))) ∧
    (∀ (t' : ℝ) (α' : ℝ),
      0 < α' ∧ α' < π / 2 ∧
      (∃ (r' : ℝ),
        arcsin (sin α') * r' = arcsin (sin (3 * α')) ∧
        arcsin (sin (3 * α')) * r' = arcsin (sin (5 * α')) ∧
        arcsin (sin (5 * α')) * r' = arcsin (sin (t' * α'))) →
      t ≤ t') ∧
    t = 9 + 4 * Real.sqrt 5 :=
by sorry

end least_positive_t_for_geometric_progression_l3138_313861


namespace f_increasing_on_interval_l3138_313886

def f (x : ℝ) : ℝ := |x - 1|

theorem f_increasing_on_interval : 
  ∀ x y : ℝ, x ∈ Set.Ici 1 → y ∈ Set.Ici 1 → x ≤ y → f x ≤ f y :=
by
  sorry

end f_increasing_on_interval_l3138_313886


namespace tim_payment_l3138_313802

/-- The total amount Tim paid for his and his cat's medical visits -/
def total_payment (doctor_visit_cost : ℝ) (doctor_insurance_coverage : ℝ) 
  (cat_visit_cost : ℝ) (cat_insurance_coverage : ℝ) : ℝ :=
  (doctor_visit_cost - doctor_visit_cost * doctor_insurance_coverage) +
  (cat_visit_cost - cat_insurance_coverage)

/-- Theorem stating that Tim paid $135 in total -/
theorem tim_payment : 
  total_payment 300 0.75 120 60 = 135 := by
  sorry

end tim_payment_l3138_313802


namespace total_books_proof_l3138_313846

/-- The total number of books on two bookshelves -/
def total_books : ℕ := 30

/-- The number of books moved from the first shelf to the second shelf -/
def books_moved : ℕ := 5

theorem total_books_proof :
  (∃ (initial_books_per_shelf : ℕ),
    initial_books_per_shelf * 2 = total_books ∧
    (initial_books_per_shelf + books_moved) = 2 * (initial_books_per_shelf - books_moved)) :=
by
  sorry

end total_books_proof_l3138_313846


namespace smallest_list_size_l3138_313867

theorem smallest_list_size (n a b : ℕ) (h1 : n = a + b) (h2 : 89 * n = 73 * a + 111 * b) : n ≥ 19 := by
  sorry

end smallest_list_size_l3138_313867


namespace number_divided_by_three_l3138_313817

theorem number_divided_by_three (x : ℝ) (h : x - 39 = 54) : x / 3 = 31 := by
  sorry

end number_divided_by_three_l3138_313817


namespace max_value_of_f_on_S_l3138_313891

/-- The set S of real numbers x where x^4 - 13x^2 + 36 ≤ 0 -/
def S : Set ℝ := {x : ℝ | x^4 - 13*x^2 + 36 ≤ 0}

/-- The function f(x) = x^3 - 3x -/
def f (x : ℝ) : ℝ := x^3 - 3*x

/-- Theorem stating that the maximum value of f(x) on S is 18 -/
theorem max_value_of_f_on_S : ∃ (m : ℝ), m = 18 ∧ ∀ (x : ℝ), x ∈ S → f x ≤ m :=
sorry

end max_value_of_f_on_S_l3138_313891


namespace sqrt_13_parts_sum_l3138_313869

theorem sqrt_13_parts_sum (a b : ℝ) : 
  (a = ⌊Real.sqrt 13⌋) → 
  (b = Real.sqrt 13 - ⌊Real.sqrt 13⌋) → 
  2 * a^2 + b - Real.sqrt 13 = 15 := by
sorry

end sqrt_13_parts_sum_l3138_313869


namespace graveyard_bone_ratio_l3138_313812

theorem graveyard_bone_ratio :
  let total_skeletons : ℕ := 20
  let adult_women_skeletons : ℕ := total_skeletons / 2
  let remaining_skeletons : ℕ := total_skeletons - adult_women_skeletons
  let adult_men_skeletons : ℕ := remaining_skeletons / 2
  let children_skeletons : ℕ := remaining_skeletons / 2
  let adult_woman_bones : ℕ := 20
  let adult_man_bones : ℕ := adult_woman_bones + 5
  let total_bones : ℕ := 375
  let child_bones : ℕ := (total_bones - (adult_women_skeletons * adult_woman_bones + adult_men_skeletons * adult_man_bones)) / children_skeletons
  (child_bones : ℚ) / (adult_woman_bones : ℚ) = 1 / 2 := by
sorry

end graveyard_bone_ratio_l3138_313812


namespace perpendicular_circle_radius_l3138_313872

/-- Given two perpendicular lines and a circle of radius R tangent to these lines,
    the radius of a circle that is tangent to the same lines and intersects
    the given circle at a right angle is R(2 ± √3). -/
theorem perpendicular_circle_radius (R : ℝ) (R_pos : R > 0) :
  ∃ x : ℝ, x > 0 ∧ 
  (x = R * (2 + Real.sqrt 3) ∨ x = R * (2 - Real.sqrt 3)) ∧
  (∃ (C C₁ : ℝ × ℝ),
    (C.1 = R ∧ C.2 = R) ∧  -- Center of the given circle
    (C₁.1 > 0 ∧ C₁.2 > 0) ∧  -- Center of the new circle in the first quadrant
    ((C₁.1 - C.1)^2 + (C₁.2 - C.2)^2 = (x + R)^2) ∧  -- Circles intersect at right angle
    (C₁.1 = x ∧ C₁.2 = x))  -- New circle is tangent to the perpendicular lines
:= by sorry

end perpendicular_circle_radius_l3138_313872


namespace ball_count_proof_l3138_313898

/-- 
Given a bag with m balls, including 6 red balls, 
if the probability of picking a red ball is 0.3, then m = 20.
-/
theorem ball_count_proof (m : ℕ) (h1 : m > 0) (h2 : 6 ≤ m) : 
  (6 : ℝ) / m = 0.3 → m = 20 := by
  sorry

end ball_count_proof_l3138_313898


namespace train_y_completion_time_l3138_313854

/-- Represents the time it takes for Train Y to complete the trip -/
def train_y_time (route_length : ℝ) (train_x_time : ℝ) (train_x_distance : ℝ) : ℝ :=
  4

/-- Theorem stating that Train Y takes 4 hours to complete the trip under the given conditions -/
theorem train_y_completion_time 
  (route_length : ℝ) 
  (train_x_time : ℝ) 
  (train_x_distance : ℝ)
  (h1 : route_length = 180)
  (h2 : train_x_time = 5)
  (h3 : train_x_distance = 80) :
  train_y_time route_length train_x_time train_x_distance = 4 := by
  sorry

#check train_y_completion_time

end train_y_completion_time_l3138_313854


namespace sum_of_coefficients_cubic_factorization_l3138_313808

theorem sum_of_coefficients_cubic_factorization :
  ∃ (p q r s t : ℤ), 
    (∀ y, 512 * y^3 + 27 = (p * y + q) * (r * y^2 + s * y + t)) ∧
    p + q + r + s + t = 60 := by
  sorry

end sum_of_coefficients_cubic_factorization_l3138_313808


namespace parabola_y_intercept_l3138_313873

/-- A parabola passing through two given points has a specific y-intercept -/
theorem parabola_y_intercept (b c : ℝ) : 
  (∀ x y, y = 2 * x^2 + b * x + c → 
    ((x = -2 ∧ y = -20) ∨ (x = 2 ∧ y = 24))) → 
  c = -6 := by sorry

end parabola_y_intercept_l3138_313873


namespace cos_13_cos_17_minus_sin_17_sin_13_l3138_313801

theorem cos_13_cos_17_minus_sin_17_sin_13 :
  Real.cos (13 * π / 180) * Real.cos (17 * π / 180) - 
  Real.sin (17 * π / 180) * Real.sin (13 * π / 180) = 
  Real.sqrt 3 / 2 := by
sorry

end cos_13_cos_17_minus_sin_17_sin_13_l3138_313801


namespace distinct_integer_roots_l3138_313862

theorem distinct_integer_roots (a : ℝ) : 
  (∃ x y : ℤ, x ≠ y ∧ x^2 + 2*a*x = 8*a ∧ y^2 + 2*a*y = 8*a) ↔ 
  (a = 4.5 ∨ a = 1 ∨ a = -9 ∨ a = -12.5) :=
sorry

end distinct_integer_roots_l3138_313862


namespace line_through_point_l3138_313833

/-- Given a line equation bx - (b+2)y = b - 3 passing through the point (3, -5), prove that b = -13/7 -/
theorem line_through_point (b : ℚ) : 
  (b * 3 - (b + 2) * (-5) = b - 3) → b = -13/7 := by
  sorry

end line_through_point_l3138_313833


namespace bill_division_l3138_313888

/-- Proves that when three people divide a 99-dollar bill evenly, each person pays 33 dollars. -/
theorem bill_division (total_bill : ℕ) (num_people : ℕ) (each_share : ℕ) :
  total_bill = 99 → num_people = 3 → each_share = total_bill / num_people → each_share = 33 := by
  sorry

#check bill_division

end bill_division_l3138_313888


namespace at_least_one_goes_probability_l3138_313887

def prob_at_least_one_goes (prob_A prob_B : ℚ) : Prop :=
  1 - (1 - prob_A) * (1 - prob_B) = 2/5

theorem at_least_one_goes_probability :
  prob_at_least_one_goes (1/4 : ℚ) (1/5 : ℚ) :=
by
  sorry

end at_least_one_goes_probability_l3138_313887


namespace r_bounds_for_area_range_l3138_313835

/-- The parabola function -/
def parabola (x : ℝ) : ℝ := x^2 + 2

/-- The line function -/
def line (r : ℝ) (x : ℝ) : ℝ := r - 1

/-- The intersection points of the parabola and the line -/
def intersection_points (r : ℝ) : Set ℝ := {x | parabola x = line r x}

/-- The area of the triangle formed by the vertex of the parabola and the intersection points -/
def triangle_area (r : ℝ) : ℝ := (r - 3)^(3/2)

/-- Theorem stating the relationship between r and the area of the triangle -/
theorem r_bounds_for_area_range :
  ∀ r : ℝ, (16 ≤ triangle_area r ∧ triangle_area r ≤ 128) ↔ (7 ≤ r ∧ r ≤ 19) :=
sorry

end r_bounds_for_area_range_l3138_313835


namespace max_value_of_product_l3138_313889

/-- The function f(x) = 6x^3 - ax^2 - 2bx + 2 -/
def f (a b x : ℝ) : ℝ := 6 * x^3 - a * x^2 - 2 * b * x + 2

/-- The derivative of f(x) -/
def f' (a b x : ℝ) : ℝ := 18 * x^2 - 2 * a * x - 2 * b

theorem max_value_of_product (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_extremum : f' a b 1 = 0) : 
  a * b ≤ (81 : ℝ) / 4 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ f' a₀ b₀ 1 = 0 ∧ a₀ * b₀ = (81 : ℝ) / 4 :=
sorry

end max_value_of_product_l3138_313889


namespace simple_annual_interest_rate_l3138_313840

/-- Simple annual interest rate calculation -/
theorem simple_annual_interest_rate 
  (monthly_interest : ℝ) 
  (investment_amount : ℝ) 
  (h1 : monthly_interest = 225)
  (h2 : investment_amount = 30000) : 
  (monthly_interest * 12) / investment_amount = 0.09 := by
sorry

end simple_annual_interest_rate_l3138_313840


namespace cupcake_net_profit_l3138_313890

/-- Calculates the net profit from selling cupcakes given the specified conditions -/
theorem cupcake_net_profit :
  let cupcake_cost : ℚ := 0.75
  let burnt_cupcakes : ℕ := 24
  let first_batch : ℕ := 24
  let second_batch : ℕ := 24
  let eaten_immediately : ℕ := 5
  let eaten_later : ℕ := 4
  let selling_price : ℚ := 2

  let total_cupcakes : ℕ := burnt_cupcakes + first_batch + second_batch
  let total_cost : ℚ := cupcake_cost * total_cupcakes
  let cupcakes_to_sell : ℕ := total_cupcakes - burnt_cupcakes - eaten_immediately - eaten_later
  let revenue : ℚ := selling_price * cupcakes_to_sell
  let net_profit : ℚ := revenue - total_cost

  net_profit = 72 := by sorry

end cupcake_net_profit_l3138_313890


namespace twelfth_term_of_specific_arithmetic_sequence_l3138_313820

/-- The nth term of an arithmetic sequence -/
def arithmeticSequenceTerm (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  a₁ + (n - 1 : ℚ) * d

theorem twelfth_term_of_specific_arithmetic_sequence :
  let a₁ : ℚ := 1
  let a₂ : ℚ := 3/2
  let d : ℚ := a₂ - a₁
  arithmeticSequenceTerm a₁ d 12 = 13/2 := by
  sorry

end twelfth_term_of_specific_arithmetic_sequence_l3138_313820


namespace min_X_value_l3138_313831

def F (X : ℤ) : List ℤ := [-4, -1, 0, 6, X]

def F_new (X : ℤ) : List ℤ := [2, 3, 0, 6, X]

def mean (l : List ℤ) : ℚ := (l.sum : ℚ) / l.length

theorem min_X_value : 
  ∀ X : ℤ, (mean (F_new X) ≥ 2 * mean (F X)) → X ≥ 9 ∧
  ∀ Y : ℤ, Y < 9 → mean (F_new Y) < 2 * mean (F Y) :=
sorry

end min_X_value_l3138_313831


namespace card_arrangement_exists_l3138_313838

/-- Represents a card with two sides, each containing a natural number -/
structure Card where
  side1 : Nat
  side2 : Nat

/-- Represents the set of n cards -/
def CardSet (n : Nat) := {cards : Finset Card // cards.card = n}

/-- Predicate to check if a set of cards satisfies the problem conditions -/
def ValidCardSet (n : Nat) (cards : CardSet n) : Prop :=
  (∀ i : Nat, i ∈ Finset.range n → (cards.val.filter (λ c => c.side1 = i + 1 ∨ c.side2 = i + 1)).card = 2) ∧
  (∀ c : Card, c ∈ cards.val → c.side1 ≤ n ∧ c.side2 ≤ n)

/-- Represents an arrangement of cards on the table -/
def Arrangement (n : Nat) := Fin n → Bool

/-- Predicate to check if an arrangement is valid (shows numbers 1 to n exactly once) -/
def ValidArrangement (n : Nat) (cards : CardSet n) (arr : Arrangement n) : Prop :=
  ∀ i : Fin n, ∃! c : Card, c ∈ cards.val ∧
    ((arr i = true ∧ c.side1 = i + 1) ∨ (arr i = false ∧ c.side2 = i + 1))

theorem card_arrangement_exists (n : Nat) (cards : CardSet n) 
  (h : ValidCardSet n cards) : ∃ arr : Arrangement n, ValidArrangement n cards arr := by
  sorry

end card_arrangement_exists_l3138_313838


namespace sailboat_problem_l3138_313857

theorem sailboat_problem (small_sail_size : ℝ) (small_sail_speed : ℝ) 
  (big_sail_speed : ℝ) (distance : ℝ) (time_difference : ℝ) :
  small_sail_size = 12 →
  small_sail_speed = 20 →
  big_sail_speed = 50 →
  distance = 200 →
  time_difference = 6 →
  distance / small_sail_speed - distance / big_sail_speed = time_difference →
  ∃ big_sail_size : ℝ, 
    big_sail_size = 30 ∧ 
    small_sail_speed / big_sail_speed = small_sail_size / big_sail_size :=
by sorry


end sailboat_problem_l3138_313857


namespace coeff_x20_Q_greater_than_P_l3138_313837

-- Define the two expressions
def P (x : ℝ) : ℝ := (1 - x^2 + x^3)^1000
def Q (x : ℝ) : ℝ := (1 + x^2 - x^3)^1000

-- Define a function to get the coefficient of x^20 in a polynomial
noncomputable def coeff_x20 (f : ℝ → ℝ) : ℝ := sorry

-- Theorem statement
theorem coeff_x20_Q_greater_than_P :
  coeff_x20 Q > coeff_x20 P := by sorry

end coeff_x20_Q_greater_than_P_l3138_313837


namespace power_sum_integer_l3138_313806

theorem power_sum_integer (x : ℝ) (h : ∃ (k : ℤ), x + 1/x = k) :
  ∀ (n : ℕ), n > 0 → ∃ (m : ℤ), x^n + 1/(x^n) = m :=
by sorry

end power_sum_integer_l3138_313806


namespace avery_egg_cartons_l3138_313809

/-- Calculates the number of full egg cartons given the number of chickens,
    eggs per chicken, and eggs per carton. -/
def full_egg_cartons (num_chickens : ℕ) (eggs_per_chicken : ℕ) (eggs_per_carton : ℕ) : ℕ :=
  (num_chickens * eggs_per_chicken) / eggs_per_carton

/-- Proves that Avery can fill 10 egg cartons with the given conditions. -/
theorem avery_egg_cartons :
  full_egg_cartons 20 6 12 = 10 := by
  sorry

#eval full_egg_cartons 20 6 12

end avery_egg_cartons_l3138_313809


namespace equation_solutions_l3138_313859

theorem equation_solutions (x : ℚ) :
  (x = 2/9 ∧ 81 * x^2 + 220 = 196 * x - 15) →
  (5/9 : ℚ)^2 * 81 + 220 = 196 * (5/9 : ℚ) - 15 := by
  sorry

end equation_solutions_l3138_313859


namespace f_greater_than_three_f_inequality_solution_range_l3138_313884

-- Define the function f
def f (x : ℝ) : ℝ := |x + 4| - |x - 1|

-- Theorem 1: f(x) > 3 iff x > 0
theorem f_greater_than_three (x : ℝ) : f x > 3 ↔ x > 0 := by sorry

-- Theorem 2: f(x) + 1 ≤ 4^a - 5×2^a has a solution iff a ∈ (-∞,0] ∪ [2,+∞)
theorem f_inequality_solution_range (a : ℝ) : 
  (∃ x, f x + 1 ≤ 4^a - 5*2^a) ↔ (a ≤ 0 ∨ a ≥ 2) := by sorry

end f_greater_than_three_f_inequality_solution_range_l3138_313884


namespace sequence_term_equals_three_l3138_313848

def a (n : ℝ) : ℝ := n^2 - 8*n + 15

theorem sequence_term_equals_three :
  ∃! (s : Set ℝ), s = {n : ℝ | a n = 3} ∧ s = {2, 6} :=
by sorry

end sequence_term_equals_three_l3138_313848


namespace sqrt_equation_proof_l3138_313882

theorem sqrt_equation_proof (y : ℝ) : 
  (Real.sqrt 1.21 / Real.sqrt y) + (Real.sqrt 1.00 / Real.sqrt 0.49) = 2.650793650793651 → 
  y = 0.81 := by
sorry

end sqrt_equation_proof_l3138_313882


namespace min_elements_special_relation_l3138_313897

/-- A relation on a set X satisfying the given properties -/
structure SpecialRelation (X : Type) where
  rel : X → X → Prop
  irreflexive : ∀ x, ¬(rel x x)
  trichotomous : ∀ x y, x ≠ y → (rel x y ∨ rel y x) ∧ ¬(rel x y ∧ rel y x)
  transitive_element : ∀ x y, rel x y → ∃ z, rel x z ∧ rel z y

/-- The minimum number of elements in a set with a SpecialRelation is 7 -/
theorem min_elements_special_relation :
  ∀ (X : Type) [Fintype X] (r : SpecialRelation X),
  Fintype.card X ≥ 7 ∧ (∀ (Y : Type) [Fintype Y], SpecialRelation Y → Fintype.card Y < 7 → False) :=
sorry

end min_elements_special_relation_l3138_313897


namespace books_bought_l3138_313843

theorem books_bought (initial_books final_books : ℕ) 
  (h1 : initial_books = 50)
  (h2 : final_books = 151) :
  final_books - initial_books = 101 := by
  sorry

end books_bought_l3138_313843


namespace soccer_match_players_l3138_313881

/-- The number of players in a soccer match -/
def num_players : ℕ := 11

/-- The total number of socks in the washing machine -/
def total_socks : ℕ := 22

/-- Each player wears exactly two socks -/
def socks_per_player : ℕ := 2

/-- Theorem: The number of players is 11 given the conditions -/
theorem soccer_match_players :
  num_players = total_socks / socks_per_player :=
by sorry

end soccer_match_players_l3138_313881


namespace tangent_line_b_value_l3138_313844

/-- The curve function -/
def f (x : ℝ) : ℝ := x^3 - 3*x^2

/-- The derivative of the curve function -/
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x

/-- The tangent line function -/
def g (x b : ℝ) : ℝ := -3*x + b

theorem tangent_line_b_value :
  ∀ b : ℝ, (∃ x : ℝ, f x = g x b ∧ f' x = -3) → b = 1 := by
  sorry

end tangent_line_b_value_l3138_313844


namespace f_decreasing_on_interval_l3138_313825

def f (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 3

theorem f_decreasing_on_interval :
  ∀ (x₁ x₂ : ℝ), x₁ < x₂ ∧ x₂ ≤ 1 → f x₁ > f x₂ := by
  sorry

end f_decreasing_on_interval_l3138_313825


namespace sin_75_degrees_l3138_313816

theorem sin_75_degrees : Real.sin (75 * π / 180) = (Real.sqrt 6 + Real.sqrt 2) / 4 := by
  sorry

end sin_75_degrees_l3138_313816


namespace unique_sum_of_three_squares_l3138_313847

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def sum_of_three_squares (a b c : ℕ) : Prop :=
  is_perfect_square a ∧ is_perfect_square b ∧ is_perfect_square c ∧ a + b + c = 100

def distinct_combinations (a b c : ℕ) : Prop :=
  sum_of_three_squares a b c ∧ 
  (a ≤ b ∧ b ≤ c)

theorem unique_sum_of_three_squares : 
  ∃! (abc : ℕ × ℕ × ℕ), distinct_combinations abc.1 abc.2.1 abc.2.2 :=
sorry

end unique_sum_of_three_squares_l3138_313847


namespace arithmetic_progression_relatively_prime_l3138_313807

theorem arithmetic_progression_relatively_prime :
  ∃ (a : ℕ → ℕ) (d : ℕ),
    (∀ n, 1 ≤ n → n ≤ 100 → a n > 0) ∧
    (∀ n m, 1 ≤ n → n < m → m ≤ 100 → a m > a n) ∧
    (∀ n, 1 < n → n ≤ 100 → a n - a (n-1) = d) ∧
    (∀ n m, 1 ≤ n → n < m → m ≤ 100 → Nat.gcd (a n) (a m) = 1) :=
by sorry

end arithmetic_progression_relatively_prime_l3138_313807


namespace negation_of_exists_leq_negation_of_proposition_l3138_313821

theorem negation_of_exists_leq (p : ℝ → Prop) : 
  (¬ ∃ x, p x) ↔ ∀ x, ¬(p x) := by sorry

theorem negation_of_proposition :
  (¬ ∃ x : ℝ, Real.exp x - x - 1 ≤ 0) ↔ (∀ x : ℝ, Real.exp x - x - 1 > 0) := by sorry

end negation_of_exists_leq_negation_of_proposition_l3138_313821


namespace number_addition_problem_l3138_313813

theorem number_addition_problem (N : ℝ) (X : ℝ) : 
  N = 180 → 
  N + X = (1/15) * N → 
  X = -168 := by sorry

end number_addition_problem_l3138_313813


namespace percent_to_decimal_five_percent_to_decimal_l3138_313841

theorem percent_to_decimal (p : ℚ) : p / 100 = p * (1 / 100) := by sorry

theorem five_percent_to_decimal : (5 : ℚ) / 100 = 0.05 := by sorry

end percent_to_decimal_five_percent_to_decimal_l3138_313841


namespace unique_valid_tournament_l3138_313860

/-- Represents the result of a chess game -/
inductive GameResult
  | Win
  | Draw
  | Loss

/-- Represents a player in the chess tournament -/
structure Player where
  id : Fin 5
  score : Rat

/-- Represents the result of a game between two players -/
structure GameOutcome where
  player1 : Fin 5
  player2 : Fin 5
  result : GameResult

/-- Represents the chess tournament -/
structure ChessTournament where
  players : Fin 5 → Player
  games : List GameOutcome

def ChessTournament.isValid (t : ChessTournament) : Prop :=
  -- Each player played exactly once with each other
  (t.games.length = 10) ∧
  -- First-place winner had no draws
  (¬ ∃ g ∈ t.games, g.player1 = 0 ∧ g.result = GameResult.Draw) ∧
  (¬ ∃ g ∈ t.games, g.player2 = 0 ∧ g.result = GameResult.Draw) ∧
  -- Second-place winner did not lose any game
  (¬ ∃ g ∈ t.games, g.player1 = 1 ∧ g.result = GameResult.Loss) ∧
  (¬ ∃ g ∈ t.games, g.player2 = 1 ∧ g.result = GameResult.Win) ∧
  -- Fourth-place player did not win any game
  (¬ ∃ g ∈ t.games, g.player1 = 3 ∧ g.result = GameResult.Win) ∧
  (¬ ∃ g ∈ t.games, g.player2 = 3 ∧ g.result = GameResult.Loss) ∧
  -- Scores of all participants were different
  (∀ i j : Fin 5, i ≠ j → (t.players i).score ≠ (t.players j).score)

/-- The unique valid tournament configuration -/
def uniqueTournament : ChessTournament := sorry

theorem unique_valid_tournament :
  ∀ t : ChessTournament, t.isValid → t = uniqueTournament := by sorry

end unique_valid_tournament_l3138_313860


namespace shaded_area_theorem_l3138_313804

def U : Set Nat := {0, 1, 2, 3, 4}
def A : Set Nat := {1, 2, 3}
def B : Set Nat := {2, 4}

theorem shaded_area_theorem :
  (U \ (A ∪ B)) ∪ (A ∩ B) = {0, 2} := by sorry

end shaded_area_theorem_l3138_313804


namespace complex_expression_simplification_l3138_313855

theorem complex_expression_simplification :
  let a : ℂ := 3 - I
  let b : ℂ := 2 + I
  let c : ℂ := -1 + 2 * I
  3 * a + 4 * b - 2 * c = 19 := by sorry

end complex_expression_simplification_l3138_313855


namespace days_to_finish_book_l3138_313851

def total_pages : ℕ := 285
def pages_per_day : ℕ := 23

theorem days_to_finish_book :
  ⌈(total_pages : ℝ) / pages_per_day⌉ = 13 := by sorry

end days_to_finish_book_l3138_313851


namespace half_dollar_percentage_l3138_313870

theorem half_dollar_percentage : 
  let nickel_count : ℕ := 80
  let half_dollar_count : ℕ := 40
  let nickel_value : ℕ := 5
  let half_dollar_value : ℕ := 50
  let total_value := nickel_count * nickel_value + half_dollar_count * half_dollar_value
  let half_dollar_total := half_dollar_count * half_dollar_value
  (half_dollar_total : ℚ) / total_value = 5 / 6 := by sorry

end half_dollar_percentage_l3138_313870


namespace quadratic_properties_l3138_313853

/-- Given a quadratic function y = (x - m)² - 2(x - m), where m is a constant -/
def f (x m : ℝ) : ℝ := (x - m)^2 - 2*(x - m)

theorem quadratic_properties (m : ℝ) :
  /- The x-intercepts are at x = m and x = m + 2 -/
  (∃ x, f x m = 0 ↔ x = m ∨ x = m + 2) ∧
  /- The vertex is at (m + 1, -1) -/
  (f (m + 1) m = -1 ∧ ∀ x, f x m ≥ -1) ∧
  /- When the graph is shifted 3 units left and 1 unit up to become y = x², m = 2 -/
  (∀ x, f (x + 3) m - 1 = x^2 → m = 2) :=
by sorry

end quadratic_properties_l3138_313853


namespace smallest_integer_solution_smallest_integer_solution_exists_smallest_integer_solution_is_zero_l3138_313823

theorem smallest_integer_solution (x : ℤ) : (7 - 5*x < 12) → x ≥ 0 :=
by
  sorry

theorem smallest_integer_solution_exists : ∃ x : ℤ, (7 - 5*x < 12) ∧ (∀ y : ℤ, (7 - 5*y < 12) → y ≥ x) :=
by
  sorry

theorem smallest_integer_solution_is_zero : 
  ∃ x : ℤ, x = 0 ∧ (7 - 5*x < 12) ∧ (∀ y : ℤ, (7 - 5*y < 12) → y ≥ x) :=
by
  sorry

end smallest_integer_solution_smallest_integer_solution_exists_smallest_integer_solution_is_zero_l3138_313823


namespace line_equation_through_point_with_angle_l3138_313814

/-- The equation of a line passing through (-1, 2) with a slope angle of 45° is x - y + 3 = 0 -/
theorem line_equation_through_point_with_angle (x y : ℝ) :
  (x + 1 = -1 ∧ y - 2 = 0) →  -- The line passes through (-1, 2)
  (Real.tan (45 * π / 180) = 1) →  -- The slope angle is 45°
  x - y + 3 = 0  -- The equation of the line
  := by sorry


end line_equation_through_point_with_angle_l3138_313814


namespace sector_max_area_l3138_313893

/-- Given a rope of length 20cm forming a sector, the area of the sector is maximized when the central angle is 2 radians. -/
theorem sector_max_area (r l α : ℝ) : 
  0 < r → r < 10 →
  l + 2 * r = 20 →
  l = r * α →
  ∀ r' l' α', 
    0 < r' → r' < 10 →
    l' + 2 * r' = 20 →
    l' = r' * α' →
    r * l ≥ r' * l' →
  α = 2 := by
sorry

end sector_max_area_l3138_313893


namespace angle_bac_equals_arcsin_four_fifths_l3138_313827

-- Define the triangle ABC and point O
structure Triangle :=
  (A B C O : ℝ × ℝ)

-- Define the distances OA, OB, OC
def distOA (t : Triangle) : ℝ := 15
def distOB (t : Triangle) : ℝ := 12
def distOC (t : Triangle) : ℝ := 20

-- Define the property that the feet of perpendiculars form an equilateral triangle
def perpendicularsFormEquilateralTriangle (t : Triangle) : Prop := sorry

-- Define the angle BAC
def angleBac (t : Triangle) : ℝ := sorry

-- State the theorem
theorem angle_bac_equals_arcsin_four_fifths (t : Triangle) :
  distOA t = 15 →
  distOB t = 12 →
  distOC t = 20 →
  perpendicularsFormEquilateralTriangle t →
  angleBac t = Real.arcsin (4/5) :=
by sorry

end angle_bac_equals_arcsin_four_fifths_l3138_313827


namespace zero_in_interval_l3138_313865

noncomputable def f (x : ℝ) : ℝ := 2^x - 6 - Real.log x

theorem zero_in_interval :
  ∃ c ∈ Set.Ioo 2 3, f c = 0 :=
by sorry

end zero_in_interval_l3138_313865


namespace volleyball_tournament_probabilities_l3138_313828

theorem volleyball_tournament_probabilities :
  -- Definition of probability of student team winning a match
  let p_student_win : ℝ := 1/2
  -- Definition of probability of teacher team winning a match
  let p_teacher_win : ℝ := 3/5
  -- Total number of teams
  let total_teams : ℕ := 21
  -- Number of student teams
  let student_teams : ℕ := 20
  -- Number of teams advancing directly to quarterfinals
  let direct_advance : ℕ := 5
  -- Number of teams selected by drawing
  let drawn_teams : ℕ := 2

  -- 1. Probability of a student team winning two consecutive matches
  (p_student_win * p_student_win = 1/4) ∧

  -- 2. Probability distribution of number of rounds teacher team participates
  (1 - p_teacher_win = 2/5) ∧
  (p_teacher_win * (1 - p_teacher_win) = 6/25) ∧
  (p_teacher_win * p_teacher_win = 9/25) ∧

  -- 3. Expectation of number of rounds teacher team participates
  (1 * (1 - p_teacher_win) + 2 * (p_teacher_win * (1 - p_teacher_win)) + 3 * (p_teacher_win * p_teacher_win) = 49/25) :=
by
  sorry

end volleyball_tournament_probabilities_l3138_313828


namespace imaginary_part_of_complex_product_l3138_313876

theorem imaginary_part_of_complex_product : Complex.im ((1 + Complex.I)^2 * (2 + Complex.I)) = 4 := by
  sorry

end imaginary_part_of_complex_product_l3138_313876


namespace two_tails_one_head_prob_l3138_313864

/-- Represents a biased coin with probabilities for heads and tails -/
structure BiasedCoin where
  probHeads : ℝ
  probTails : ℝ
  prob_sum : probHeads + probTails = 1

/-- Calculates the probability of getting exactly two tails followed by one head within 5 flips -/
def prob_two_tails_one_head (c : BiasedCoin) : ℝ :=
  3 * (c.probTails * c.probTails * c.probTails * c.probHeads)

/-- The main theorem to be proved -/
theorem two_tails_one_head_prob :
  let c : BiasedCoin := ⟨0.3, 0.7, by norm_num⟩
  prob_two_tails_one_head c = 0.3087 := by
  sorry


end two_tails_one_head_prob_l3138_313864


namespace two_numbers_difference_l3138_313878

theorem two_numbers_difference (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 24) :
  |x - y| = 12/5 := by
  sorry

end two_numbers_difference_l3138_313878


namespace polynomial_evaluation_l3138_313830

theorem polynomial_evaluation : 
  let x : ℝ := 6
  (3 * x^2 + 15 * x + 7) + (4 * x^3 + 8 * x^2 - 5 * x + 10) = 1337 := by
  sorry

end polynomial_evaluation_l3138_313830


namespace invertible_function_fixed_point_l3138_313895

/-- Given an invertible function f: ℝ → ℝ, if f(a) = 3 and f(3) = a, then a - 3 = 0 -/
theorem invertible_function_fixed_point 
  (f : ℝ → ℝ) (hf : Function.Bijective f) (a : ℝ) 
  (h1 : f a = 3) (h2 : f 3 = a) : a - 3 = 0 :=
sorry

end invertible_function_fixed_point_l3138_313895


namespace smallest_class_size_l3138_313836

theorem smallest_class_size (n : ℕ) (h1 : n > 50) 
  (h2 : ∃ x : ℕ, n = 3*x + 2*(x+1)) : n ≥ 52 :=
by
  sorry

end smallest_class_size_l3138_313836


namespace johns_initial_squat_weight_l3138_313871

/-- Calculates John's initial squat weight based on given conditions --/
theorem johns_initial_squat_weight :
  ∀ (initial_bench initial_deadlift new_total : ℝ),
  initial_bench = 400 →
  initial_deadlift = 800 →
  new_total = 1490 →
  ∃ (initial_squat : ℝ),
    initial_squat * 0.7 + initial_bench + (initial_deadlift - 200) = new_total ∧
    initial_squat = 700 :=
by
  sorry

end johns_initial_squat_weight_l3138_313871


namespace sum_marked_sides_ge_one_l3138_313803

/-- A rectangle within a unit square --/
structure Rectangle where
  width : ℝ
  height : ℝ
  markedSide : ℝ
  width_pos : 0 < width
  height_pos : 0 < height
  in_unit_square : width ≤ 1 ∧ height ≤ 1
  marked_side_valid : markedSide = width ∨ markedSide = height

/-- A partition of the unit square into rectangles --/
def UnitSquarePartition := List Rectangle

/-- The sum of the marked sides in a partition --/
def sumMarkedSides (partition : UnitSquarePartition) : ℝ :=
  partition.map (·.markedSide) |>.sum

/-- The total area of rectangles in a partition --/
def totalArea (partition : UnitSquarePartition) : ℝ :=
  partition.map (λ r => r.width * r.height) |>.sum

/-- Theorem: The sum of marked sides in any valid partition is at least 1 --/
theorem sum_marked_sides_ge_one (partition : UnitSquarePartition) 
  (h_valid : totalArea partition = 1) : 
  sumMarkedSides partition ≥ 1 := by
  sorry


end sum_marked_sides_ge_one_l3138_313803


namespace percentage_of_girls_who_want_to_be_doctors_l3138_313819

theorem percentage_of_girls_who_want_to_be_doctors
  (total_students : ℝ)
  (boys_ratio : ℝ)
  (boys_doctor_ratio : ℝ)
  (boys_doctor_all_doctor_ratio : ℝ)
  (h1 : boys_ratio = 3 / 5)
  (h2 : boys_doctor_ratio = 1 / 3)
  (h3 : boys_doctor_all_doctor_ratio = 2 / 5) :
  (((1 - boys_ratio) * total_students) / total_students - 
   ((1 - boys_doctor_all_doctor_ratio) * (boys_ratio * boys_doctor_ratio * total_students)) / 
   ((1 - boys_ratio) * total_students)) * 100 = 75 := by
sorry

end percentage_of_girls_who_want_to_be_doctors_l3138_313819


namespace intersection_when_m_3_m_value_for_given_intersection_l3138_313845

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -1 < x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x : ℝ | -x^2 + 2*x + m > 0}

-- Theorem 1: Intersection of A and B when m = 3
theorem intersection_when_m_3 :
  A ∩ B 3 = {x : ℝ | -1 < x ∧ x < 3} := by sorry

-- Theorem 2: Value of m when A ∩ B = {x | -1 < x < 4}
theorem m_value_for_given_intersection :
  ∃ m : ℝ, A ∩ B m = {x : ℝ | -1 < x ∧ x < 4} ∧ m = 8 := by sorry

end intersection_when_m_3_m_value_for_given_intersection_l3138_313845


namespace largest_integer_less_than_100_with_remainder_5_mod_9_l3138_313826

theorem largest_integer_less_than_100_with_remainder_5_mod_9 :
  ∀ n : ℕ, n < 100 → n % 9 = 5 → n ≤ 95 :=
by
  sorry

end largest_integer_less_than_100_with_remainder_5_mod_9_l3138_313826


namespace motel_total_rent_l3138_313866

/-- Represents the total rent charged by a motel on a Saturday night. -/
def total_rent (r40 r60 : ℕ) : ℕ := 40 * r40 + 60 * r60

/-- The condition that changing 10 rooms from $60 to $40 reduces the total rent by 50%. -/
def rent_reduction_condition (r40 r60 : ℕ) : Prop :=
  total_rent (r40 + 10) (r60 - 10) = (total_rent r40 r60) / 2

/-- The theorem stating that the total rent charged by the motel is $800. -/
theorem motel_total_rent :
  ∃ (r40 r60 : ℕ), rent_reduction_condition r40 r60 ∧ total_rent r40 r60 = 800 :=
sorry

end motel_total_rent_l3138_313866


namespace correct_quotient_proof_l3138_313879

theorem correct_quotient_proof (N : ℕ) (h1 : N % 21 = 0) (h2 : N / 12 = 70) : N / 21 = 40 := by
  sorry

end correct_quotient_proof_l3138_313879


namespace quadruple_equality_l3138_313815

theorem quadruple_equality (a b c d : ℝ) : 
  (∀ X : ℝ, X^2 + a*X + b = (X-a)*(X-c)) ∧
  (∀ X : ℝ, X^2 + c*X + d = (X-b)*(X-d)) →
  ((a = 1 ∧ b = 2 ∧ c = -2 ∧ d = 0) ∨ 
   (a = -1 ∧ b = -2 ∧ c = 2 ∧ d = 0)) :=
by sorry

end quadruple_equality_l3138_313815


namespace min_value_theorem_l3138_313899

theorem min_value_theorem (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hxy : x + y > z) (hyz : y + z > x) (hzx : z + x > y) :
  (x + y + z) * (1 / (x + y - z) + 1 / (y + z - x) + 1 / (z + x - y)) ≥ 9 := by
  sorry

end min_value_theorem_l3138_313899


namespace function_property_l3138_313877

/-- A function satisfying the given property -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, y^3 * f x = x^3 * f y

theorem function_property (f : ℝ → ℝ) (h1 : special_function f) (h2 : f 3 ≠ 0) :
  (f 20 - f 2) / f 3 = 296 := by
  sorry

end function_property_l3138_313877


namespace prob_four_sixes_eq_one_over_1296_l3138_313892

-- Define a fair six-sided die
def fair_six_sided_die : Finset ℕ := Finset.range 6

-- Define the probability of rolling a specific number on a fair six-sided die
def prob_single_roll (n : ℕ) : ℚ :=
  if n ∈ fair_six_sided_die then 1 / 6 else 0

-- Define the probability of rolling four sixes
def prob_four_sixes : ℚ := (prob_single_roll 6) ^ 4

-- Theorem statement
theorem prob_four_sixes_eq_one_over_1296 :
  prob_four_sixes = 1 / 1296 := by sorry

end prob_four_sixes_eq_one_over_1296_l3138_313892


namespace percentage_problem_l3138_313894

theorem percentage_problem : ∃ p : ℚ, p = 55/100 ∧ p * 40 = 4/5 * 25 + 2 := by
  sorry

end percentage_problem_l3138_313894


namespace smallest_angle_CBD_l3138_313810

theorem smallest_angle_CBD (ABC : ℝ) (ABD : ℝ) (CBD : ℝ) 
  (h1 : ABC = 40)
  (h2 : ABD = 15)
  (h3 : CBD = ABC - ABD) :
  CBD = 25 := by
sorry

end smallest_angle_CBD_l3138_313810


namespace min_value_condition_l3138_313885

open Set

variables {f : ℝ → ℝ} {a b : ℝ}

theorem min_value_condition (h_diff : Differentiable ℝ f) (h_cont : ContinuousOn f (Ioo a b)) :
  (∃ x₀ ∈ Ioo a b, deriv f x₀ = 0) →
  (∃ x_min ∈ Ioo a b, ∀ x ∈ Ioo a b, f x_min ≤ f x) ∧
  ¬ ((∃ x_min ∈ Ioo a b, ∀ x ∈ Ioo a b, f x_min ≤ f x) →
     (∃ x₀ ∈ Ioo a b, deriv f x₀ = 0)) :=
by sorry

end min_value_condition_l3138_313885


namespace five_dollar_neg_one_eq_zero_l3138_313805

-- Define the $ operation
def dollar_op (a b : ℤ) : ℤ := a * (b + 2) + a * b

-- Theorem statement
theorem five_dollar_neg_one_eq_zero : dollar_op 5 (-1) = 0 := by
  sorry

end five_dollar_neg_one_eq_zero_l3138_313805


namespace prime_factor_sum_squares_l3138_313842

theorem prime_factor_sum_squares (n : ℕ+) : 
  (∃ p q : ℕ, 
    Nat.Prime p ∧ 
    Nat.Prime q ∧ 
    p ∣ n ∧ 
    q ∣ n ∧ 
    (∀ r : ℕ, Nat.Prime r → r ∣ n → p ≤ r) ∧
    (∀ r : ℕ, Nat.Prime r → r ∣ n → r ≤ q) ∧
    p^2 + q^2 = n + 9) ↔ 
  n = 9 ∨ n = 20 := by
sorry


end prime_factor_sum_squares_l3138_313842
