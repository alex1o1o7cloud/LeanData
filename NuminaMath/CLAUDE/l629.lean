import Mathlib

namespace NUMINAMATH_CALUDE_alligator_journey_time_l629_62940

/-- The combined time of Paul's journey to the Nile Delta and back -/
def combined_journey_time (initial_time : ℕ) (additional_return_time : ℕ) : ℕ :=
  initial_time + (initial_time + additional_return_time)

/-- Theorem stating that the combined journey time is 10 hours -/
theorem alligator_journey_time : combined_journey_time 4 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_alligator_journey_time_l629_62940


namespace NUMINAMATH_CALUDE_cody_remaining_games_l629_62928

theorem cody_remaining_games (initial_games given_away_games : ℕ) 
  (h1 : initial_games = 9)
  (h2 : given_away_games = 4) :
  initial_games - given_away_games = 5 := by
  sorry

end NUMINAMATH_CALUDE_cody_remaining_games_l629_62928


namespace NUMINAMATH_CALUDE_equidistant_complex_function_l629_62990

theorem equidistant_complex_function (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ z : ℂ, Complex.abs ((a + Complex.I * b) * z - z) = Complex.abs ((a + Complex.I * b) * z)) →
  Complex.abs (a + Complex.I * b) = 8 →
  b^2 = 255/4 := by
sorry

end NUMINAMATH_CALUDE_equidistant_complex_function_l629_62990


namespace NUMINAMATH_CALUDE_strawberry_cake_cost_l629_62945

/-- Proves that the cost of each strawberry cake is $22 given the order details --/
theorem strawberry_cake_cost
  (num_chocolate : ℕ)
  (price_chocolate : ℕ)
  (num_strawberry : ℕ)
  (total_cost : ℕ)
  (h1 : num_chocolate = 3)
  (h2 : price_chocolate = 12)
  (h3 : num_strawberry = 6)
  (h4 : total_cost = 168)
  : (total_cost - num_chocolate * price_chocolate) / num_strawberry = 22 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_cake_cost_l629_62945


namespace NUMINAMATH_CALUDE_correct_number_of_arrangements_l629_62950

/-- The number of arrangements for 3 boys and 3 girls in a line, where students of the same gender are adjacent -/
def number_of_arrangements : ℕ := 72

/-- The number of boys -/
def num_boys : ℕ := 3

/-- The number of girls -/
def num_girls : ℕ := 3

/-- The total number of students -/
def total_students : ℕ := num_boys + num_girls

/-- Theorem stating that the number of arrangements is correct -/
theorem correct_number_of_arrangements :
  number_of_arrangements = (Nat.factorial num_boys) * (Nat.factorial num_girls) * 2 :=
by sorry

end NUMINAMATH_CALUDE_correct_number_of_arrangements_l629_62950


namespace NUMINAMATH_CALUDE_sum_of_square_roots_bound_l629_62913

theorem sum_of_square_roots_bound (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) 
  (sum_one : x + y + z = 1) : 
  Real.sqrt (7 * x + 3) + Real.sqrt (7 * y + 3) + Real.sqrt (7 * z + 3) ≤ 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_square_roots_bound_l629_62913


namespace NUMINAMATH_CALUDE_greatest_integer_for_all_real_domain_l629_62946

theorem greatest_integer_for_all_real_domain : 
  ∃ (b : ℤ), (∀ (x : ℝ), x^2 + b*x + 15 ≠ 0) ∧ 
  (∀ (c : ℤ), (∀ (x : ℝ), x^2 + c*x + 15 ≠ 0) → c ≤ b) ∧ 
  b = 7 := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_for_all_real_domain_l629_62946


namespace NUMINAMATH_CALUDE_aruns_weight_estimation_l629_62925

/-- Arun's weight estimation problem -/
theorem aruns_weight_estimation (W : ℝ) (L : ℝ) : 
  (L < W ∧ W < 72) →  -- Arun's estimation
  (60 < W ∧ W < 70) →  -- Brother's estimation
  (W ≤ 68) →  -- Mother's estimation
  (∃ (a b : ℝ), 60 < a ∧ a < b ∧ b ≤ 68 ∧ (a + b) / 2 = 67) →  -- Average condition
  L > 60 := by
sorry

end NUMINAMATH_CALUDE_aruns_weight_estimation_l629_62925


namespace NUMINAMATH_CALUDE_inequality_solutions_l629_62948

theorem inequality_solutions :
  (∀ x : ℝ, x^2 - 2*x - 1 > 0 ↔ (x > Real.sqrt 2 + 1 ∨ x < -Real.sqrt 2 + 1)) ∧
  (∀ x : ℝ, x ≠ 3 → ((2*x - 1) / (x - 3) ≥ 3 ↔ 3 < x ∧ x ≤ 8)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solutions_l629_62948


namespace NUMINAMATH_CALUDE_ellipse_equation_and_fixed_point_l629_62959

structure Ellipse where
  center : ℝ × ℝ
  a : ℝ
  b : ℝ

def pointOnEllipse (E : Ellipse) (p : ℝ × ℝ) : Prop :=
  (p.1 - E.center.1)^2 / E.a^2 + (p.2 - E.center.2)^2 / E.b^2 = 1

def Line (p q : ℝ × ℝ) := {r : ℝ × ℝ | ∃ t, r = (1 - t) • p + t • q}

theorem ellipse_equation_and_fixed_point 
  (E : Ellipse)
  (h_center : E.center = (0, 0))
  (h_A : pointOnEllipse E (0, -2))
  (h_B : pointOnEllipse E (3/2, -1)) :
  (E.a^2 = 3 ∧ E.b^2 = 4) ∧
  ∀ (P M N T H : ℝ × ℝ),
    P = (1, -2) →
    pointOnEllipse E M →
    pointOnEllipse E N →
    M ∈ Line P N →
    T.1 = M.1 ∧ T ∈ Line (0, -2) (3/2, -1) →
    H.1 - T.1 = T.1 - M.1 ∧ H.2 = T.2 →
    (0, -2) ∈ Line H N :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_and_fixed_point_l629_62959


namespace NUMINAMATH_CALUDE_circle_radius_with_perpendicular_chords_l629_62904

/-- Given a circle with two perpendicular chords intersecting at the center,
    if two parallel sides of the formed quadrilateral have length 2,
    then the radius of the circle is √2. -/
theorem circle_radius_with_perpendicular_chords 
  (O : ℝ × ℝ) -- Center of the circle
  (K L M N : ℝ × ℝ) -- Points on the circle
  (h1 : (K.1 - M.1) * (L.2 - N.2) = 0) -- KM ⊥ LN
  (h2 : (K.2 - L.2) = (M.2 - N.2)) -- KL ∥ MN
  (h3 : Real.sqrt ((K.1 - L.1)^2 + (K.2 - L.2)^2) = 2) -- KL = 2
  (h4 : Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2) = 2) -- MN = 2
  (h5 : O = (0, 0)) -- Center at origin
  (h6 : K.2 = 0 ∧ M.2 = 0) -- K and M on x-axis
  (h7 : L.1 = 0 ∧ N.1 = 0) -- L and N on y-axis
  : Real.sqrt (K.1^2 + N.2^2) = Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_circle_radius_with_perpendicular_chords_l629_62904


namespace NUMINAMATH_CALUDE_divisibility_problem_l629_62951

theorem divisibility_problem (n : ℕ) (h1 : n = 6268440) (h2 : n % 5 = 0) : n % 30 = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_problem_l629_62951


namespace NUMINAMATH_CALUDE_cafeteria_optimal_location_l629_62985

-- Define the offices and their employee counts
structure Office where
  location : ℝ × ℝ
  employees : ℕ

-- Define the triangle formed by the offices
def office_triangle (A B C : Office) : Prop :=
  A.location ≠ B.location ∧ B.location ≠ C.location ∧ C.location ≠ A.location

-- Define the total distance function
def total_distance (cafeteria : ℝ × ℝ) (A B C : Office) : ℝ :=
  A.employees * dist cafeteria A.location +
  B.employees * dist cafeteria B.location +
  C.employees * dist cafeteria C.location

-- State the theorem
theorem cafeteria_optimal_location (A B C : Office) 
  (h_triangle : office_triangle A B C)
  (h_employees : A.employees = 10 ∧ B.employees = 20 ∧ C.employees = 30) :
  ∀ cafeteria : ℝ × ℝ, total_distance C.location A B C ≤ total_distance cafeteria A B C :=
sorry

end NUMINAMATH_CALUDE_cafeteria_optimal_location_l629_62985


namespace NUMINAMATH_CALUDE_lines_are_parallel_l629_62949

/-- Two lines a₁x + b₁y + c₁ = 0 and a₂x + b₂y + c₂ = 0 are parallel if and only if a₁b₂ = a₂b₁ -/
def parallel (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : Prop :=
  a₁ * b₂ = a₂ * b₁ ∧ a₁ * c₂ ≠ a₂ * c₁

/-- The line x - 2y + 1 = 0 -/
def line1 : ℝ → ℝ → ℝ := λ x y => x - 2*y + 1

/-- The line 2x - 4y + 1 = 0 -/
def line2 : ℝ → ℝ → ℝ := λ x y => 2*x - 4*y + 1

theorem lines_are_parallel : parallel 1 (-2) 1 2 (-4) 1 :=
  sorry

end NUMINAMATH_CALUDE_lines_are_parallel_l629_62949


namespace NUMINAMATH_CALUDE_fraction_product_simplification_l629_62939

theorem fraction_product_simplification :
  (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) = 3 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_simplification_l629_62939


namespace NUMINAMATH_CALUDE_woman_lawyer_probability_l629_62975

/-- Represents a study group with given proportions of women and women lawyers -/
structure StudyGroup where
  totalMembers : ℕ
  womenPercentage : ℚ
  womenLawyerPercentage : ℚ

/-- Calculates the probability of selecting a woman lawyer at random from the study group -/
def probWomanLawyer (group : StudyGroup) : ℚ :=
  group.womenPercentage * group.womenLawyerPercentage

theorem woman_lawyer_probability (group : StudyGroup) 
  (h1 : group.womenPercentage = 9/10)
  (h2 : group.womenLawyerPercentage = 6/10) :
  probWomanLawyer group = 54/100 := by
  sorry

#eval probWomanLawyer { totalMembers := 100, womenPercentage := 9/10, womenLawyerPercentage := 6/10 }

end NUMINAMATH_CALUDE_woman_lawyer_probability_l629_62975


namespace NUMINAMATH_CALUDE_train_braking_problem_l629_62909

/-- The braking distance function for a train -/
def S (t : ℝ) : ℝ := 27 * t - 0.45 * t^2

/-- The derivative of the braking distance function -/
def S' (t : ℝ) : ℝ := 27 - 0.9 * t

theorem train_braking_problem :
  (∃ t : ℝ, S' t = 0 ∧ t = 30) ∧
  S 30 = 405 := by
  sorry

end NUMINAMATH_CALUDE_train_braking_problem_l629_62909


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l629_62960

def M : Set ℕ := {1, 2, 3, 5}
def N : Set ℕ := {2, 3, 4}

theorem intersection_of_M_and_N : M ∩ N = {2, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l629_62960


namespace NUMINAMATH_CALUDE_super_ball_distance_l629_62996

/-- The total distance traveled by a bouncing ball -/
def totalDistance (initialHeight : ℝ) (bounceRatio : ℝ) (numBounces : ℕ) : ℝ :=
  let descentDistances := List.range (numBounces + 1) |>.map (fun i => initialHeight * bounceRatio^i)
  let ascentDistances := List.range numBounces |>.map (fun i => initialHeight * bounceRatio^(i + 1))
  (descentDistances.sum + ascentDistances.sum)

/-- Theorem: The total distance traveled by a ball dropped from 20 meters, 
    bouncing 5/8 of its previous height each time, and hitting the ground 4 times, 
    is 73.442078125 meters. -/
theorem super_ball_distance :
  totalDistance 20 (5/8) 4 = 73.442078125 := by
  sorry


end NUMINAMATH_CALUDE_super_ball_distance_l629_62996


namespace NUMINAMATH_CALUDE_correct_oranges_to_remove_l629_62933

/-- Represents the fruit selection problem -/
structure FruitSelection where
  applePrice : ℚ  -- Price of each apple in cents
  orangePrice : ℚ  -- Price of each orange in cents
  totalFruits : ℕ  -- Total number of fruits initially selected
  initialAvgPrice : ℚ  -- Initial average price of all fruits
  desiredAvgPrice : ℚ  -- Desired average price after removing oranges

/-- Calculates the number of oranges to remove -/
def orangesToRemove (fs : FruitSelection) : ℕ :=
  sorry

/-- Theorem stating the correct number of oranges to remove -/
theorem correct_oranges_to_remove (fs : FruitSelection) 
  (h1 : fs.applePrice = 40/100)
  (h2 : fs.orangePrice = 60/100)
  (h3 : fs.totalFruits = 20)
  (h4 : fs.initialAvgPrice = 56/100)
  (h5 : fs.desiredAvgPrice = 52/100) :
  orangesToRemove fs = 10 := by sorry

end NUMINAMATH_CALUDE_correct_oranges_to_remove_l629_62933


namespace NUMINAMATH_CALUDE_nate_total_distance_l629_62901

/-- The length of the football field in meters -/
def field_length : ℕ := 168

/-- The distance Nate ran in the first part, equal to four times the field length -/
def first_distance : ℕ := 4 * field_length

/-- The additional distance Nate ran in meters -/
def additional_distance : ℕ := 500

/-- The total distance Nate ran -/
def total_distance : ℕ := first_distance + additional_distance

/-- Theorem stating that the total distance Nate ran is 1172 meters -/
theorem nate_total_distance : total_distance = 1172 := by
  sorry

end NUMINAMATH_CALUDE_nate_total_distance_l629_62901


namespace NUMINAMATH_CALUDE_wage_calculation_l629_62900

/-- The original daily wage of a worker -/
def original_wage : ℝ := 242.83

/-- The new total weekly salary after increases -/
def new_total_salary : ℝ := 1457

/-- The list of wage increase percentages for each day of the work week -/
def wage_increases : List ℝ := [0.2, 0.3, 0.4, 0.5, 0.6]

theorem wage_calculation (W : ℝ) :
  (W * (List.sum (List.map (λ x => 1 + x) wage_increases))) = new_total_salary →
  W = original_wage :=
by sorry

end NUMINAMATH_CALUDE_wage_calculation_l629_62900


namespace NUMINAMATH_CALUDE_cat_and_mouse_positions_l629_62957

def cat_cycle_length : ℕ := 4
def mouse_cycle_length : ℕ := 8
def total_moves : ℕ := 247

theorem cat_and_mouse_positions :
  (total_moves % cat_cycle_length = 3) ∧
  (total_moves % mouse_cycle_length = 7) := by
  sorry

end NUMINAMATH_CALUDE_cat_and_mouse_positions_l629_62957


namespace NUMINAMATH_CALUDE_sphere_radius_from_shadows_l629_62922

/-- Given a sphere and a stick under parallel sun rays, prove the radius of the sphere -/
theorem sphere_radius_from_shadows
  (shadow_sphere : ℝ)  -- Length of the sphere's shadow
  (height_stick : ℝ)   -- Height of the stick
  (shadow_stick : ℝ)   -- Length of the stick's shadow
  (h_shadow_sphere : shadow_sphere = 20)
  (h_height_stick : height_stick = 1)
  (h_shadow_stick : shadow_stick = 4)
  : ∃ (radius : ℝ), radius = 5 ∧ (radius / shadow_sphere = height_stick / shadow_stick) :=
sorry

end NUMINAMATH_CALUDE_sphere_radius_from_shadows_l629_62922


namespace NUMINAMATH_CALUDE_rhombus_area_l629_62955

theorem rhombus_area (d₁ d₂ : ℝ) : 
  d₁^2 - 6*d₁ + 8 = 0 → 
  d₂^2 - 6*d₂ + 8 = 0 → 
  d₁ ≠ d₂ →
  (1/2) * d₁ * d₂ = 4 := by
  sorry


end NUMINAMATH_CALUDE_rhombus_area_l629_62955


namespace NUMINAMATH_CALUDE_savings_in_cents_l629_62974

-- Define the prices and quantities for each store
def store1_price : ℚ := 3
def store1_quantity : ℕ := 6
def store2_price : ℚ := 4
def store2_quantity : ℕ := 10

-- Define the price per apple for each store
def price_per_apple_store1 : ℚ := store1_price / store1_quantity
def price_per_apple_store2 : ℚ := store2_price / store2_quantity

-- Define the savings per apple in dollars
def savings_per_apple : ℚ := price_per_apple_store1 - price_per_apple_store2

-- Theorem to prove
theorem savings_in_cents : savings_per_apple * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_savings_in_cents_l629_62974


namespace NUMINAMATH_CALUDE_range_of_a_l629_62971

def sequence_a (a : ℝ) (n : ℕ) : ℝ := a * n^2 + n

theorem range_of_a (a : ℝ) :
  (∀ n, sequence_a a n < sequence_a a (n + 1)) ↔ a ≥ 0 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l629_62971


namespace NUMINAMATH_CALUDE_bankers_discount_example_l629_62988

/-- Given a true discount and a sum due, calculate the banker's discount -/
def bankers_discount (true_discount : ℚ) (sum_due : ℚ) : ℚ :=
  (true_discount * sum_due) / (sum_due - true_discount)

/-- Theorem: The banker's discount is 78 given a true discount of 66 and a sum due of 429 -/
theorem bankers_discount_example : bankers_discount 66 429 = 78 := by
  sorry

end NUMINAMATH_CALUDE_bankers_discount_example_l629_62988


namespace NUMINAMATH_CALUDE_stratified_sampling_size_l629_62953

theorem stratified_sampling_size 
  (high_school_students : ℕ) 
  (junior_high_students : ℕ) 
  (sampled_high_school : ℕ) 
  (h1 : high_school_students = 3500)
  (h2 : junior_high_students = 1500)
  (h3 : sampled_high_school = 70) :
  let total_students := high_school_students + junior_high_students
  let sampling_ratio := sampled_high_school / high_school_students
  let total_sample_size := total_students * sampling_ratio
  total_sample_size = 100 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_size_l629_62953


namespace NUMINAMATH_CALUDE_square_root_of_one_is_one_l629_62937

theorem square_root_of_one_is_one : Real.sqrt 1 = 1 := by sorry

end NUMINAMATH_CALUDE_square_root_of_one_is_one_l629_62937


namespace NUMINAMATH_CALUDE_f_3_equals_6_l629_62978

def f : ℕ → ℕ
  | 0 => 1
  | n + 1 => (n + 1) * f n

theorem f_3_equals_6 : f 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_f_3_equals_6_l629_62978


namespace NUMINAMATH_CALUDE_distance_circle_center_to_line_l629_62918

/-- The distance from the center of a circle to a line --/
theorem distance_circle_center_to_line :
  let circle_eq (x y : ℝ) := x^2 + y^2 + 2*x + 2*y - 2 = 0
  let line_eq (x y : ℝ) := x - y + 2 = 0
  let center : ℝ × ℝ := (-1, -1)
  ∃ (d : ℝ), d = Real.sqrt 2 ∧
    d = (|center.1 - center.2 + 2|) / Real.sqrt (1^2 + (-1)^2) ∧
    ∀ (x y : ℝ), circle_eq x y → (x + 1)^2 + (y + 1)^2 = 4 :=
by
  sorry


end NUMINAMATH_CALUDE_distance_circle_center_to_line_l629_62918


namespace NUMINAMATH_CALUDE_sqrt_fraction_equality_l629_62926

theorem sqrt_fraction_equality : Real.sqrt (9/4) - Real.sqrt (4/9) + 1/6 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fraction_equality_l629_62926


namespace NUMINAMATH_CALUDE_mckenna_work_hours_l629_62995

-- Define the start and end times of Mckenna's work day
def start_time : ℕ := 8
def office_end_time : ℕ := 11
def conference_end_time : ℕ := 13
def work_end_time : ℕ := conference_end_time + 2

-- Define the duration of each part of Mckenna's work day
def office_duration : ℕ := office_end_time - start_time
def conference_duration : ℕ := conference_end_time - office_end_time
def after_conference_duration : ℕ := 2

-- Theorem to prove
theorem mckenna_work_hours :
  office_duration + conference_duration + after_conference_duration = 7 := by
  sorry


end NUMINAMATH_CALUDE_mckenna_work_hours_l629_62995


namespace NUMINAMATH_CALUDE_special_prime_sum_of_squares_l629_62970

theorem special_prime_sum_of_squares (n : ℕ) : 
  (∃ a b : ℤ, n = a^2 + b^2 ∧ Int.gcd a b = 1) →
  (∀ p : ℕ, Nat.Prime p → p ≤ Nat.sqrt n → ∃ k : ℤ, k * p = a * b) →
  n = 5 ∨ n = 13 := by sorry

end NUMINAMATH_CALUDE_special_prime_sum_of_squares_l629_62970


namespace NUMINAMATH_CALUDE_base_with_five_digits_l629_62931

theorem base_with_five_digits : ∃! b : ℕ+, b ≥ 2 ∧ b ^ 4 ≤ 500 ∧ 500 < b ^ 5 := by sorry

end NUMINAMATH_CALUDE_base_with_five_digits_l629_62931


namespace NUMINAMATH_CALUDE_negative_movement_l629_62930

-- Define a type for direction
inductive Direction
| East
| West

-- Define a function to represent movement
def movement (distance : ℤ) : Direction :=
  if distance > 0 then Direction.East else Direction.West

-- Define the theorem
theorem negative_movement :
  (movement 30 = Direction.East) →
  (movement (-50) = Direction.West) :=
by
  sorry

end NUMINAMATH_CALUDE_negative_movement_l629_62930


namespace NUMINAMATH_CALUDE_min_value_abc_l629_62920

theorem min_value_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : 2*a + 4*b + 7*c ≤ 2*a*b*c) : 
  a + b + c ≥ 15/2 := by
  sorry


end NUMINAMATH_CALUDE_min_value_abc_l629_62920


namespace NUMINAMATH_CALUDE_carla_candy_bags_l629_62924

/-- Calculates the number of bags bought given the original price, discount percentage, and total amount spent -/
def bags_bought (original_price : ℚ) (discount_percentage : ℚ) (total_spent : ℚ) : ℚ :=
  total_spent / (original_price * (1 - discount_percentage))

/-- Proves that Carla bought 2 bags of candy -/
theorem carla_candy_bags : 
  let original_price : ℚ := 6
  let discount_percentage : ℚ := 3/4
  let total_spent : ℚ := 3
  bags_bought original_price discount_percentage total_spent = 2 := by
sorry

#eval bags_bought 6 (3/4) 3

end NUMINAMATH_CALUDE_carla_candy_bags_l629_62924


namespace NUMINAMATH_CALUDE_lattice_points_count_is_35_l629_62977

/-- The number of lattice points in the region bounded by the x-axis, 
    the line x=4, and the parabola y=x^2 -/
def lattice_points_count : ℕ :=
  (Finset.range 5).sum (λ x => x^2 + 1)

/-- The theorem stating that the number of lattice points in the specified region is 35 -/
theorem lattice_points_count_is_35 : lattice_points_count = 35 := by
  sorry

end NUMINAMATH_CALUDE_lattice_points_count_is_35_l629_62977


namespace NUMINAMATH_CALUDE_books_given_to_sandy_l629_62984

/-- Given that Benny initially had 24 books, Tim has 33 books, and the total number of books
    among Benny, Tim, and Sandy is 47, prove that Benny gave Sandy 10 books. -/
theorem books_given_to_sandy (benny_initial : ℕ) (tim : ℕ) (total : ℕ)
    (h1 : benny_initial = 24)
    (h2 : tim = 33)
    (h3 : total = 47)
    : benny_initial - (total - tim) = 10 := by
  sorry

end NUMINAMATH_CALUDE_books_given_to_sandy_l629_62984


namespace NUMINAMATH_CALUDE_simplify_nested_roots_l629_62947

theorem simplify_nested_roots (b : ℝ) (hb : b > 0) :
  (((b^16)^(1/8))^(1/4))^3 * (((b^16)^(1/4))^(1/8))^3 = b^3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_nested_roots_l629_62947


namespace NUMINAMATH_CALUDE_max_sum_on_circle_l629_62983

theorem max_sum_on_circle (x y : ℤ) (h : x^2 + y^2 = 36) : x + y ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_on_circle_l629_62983


namespace NUMINAMATH_CALUDE_hippopotamus_bottle_caps_l629_62972

/-- The number of bottle caps eaten by the hippopotamus -/
def bottleCapsEaten (initialCaps : ℕ) (remainingCaps : ℕ) : ℕ :=
  initialCaps - remainingCaps

theorem hippopotamus_bottle_caps : bottleCapsEaten 34 26 = 8 := by
  sorry

end NUMINAMATH_CALUDE_hippopotamus_bottle_caps_l629_62972


namespace NUMINAMATH_CALUDE_largest_integer_before_zero_l629_62903

noncomputable def f (x : ℝ) := Real.log x + 2 * x - 6

theorem largest_integer_before_zero (x₀ : ℝ) (h : f x₀ = 0) :
  ∃ k : ℤ, k = 2 ∧ k ≤ ⌊x₀⌋ ∧ ∀ m : ℤ, m > k → m > ⌊x₀⌋ :=
sorry

end NUMINAMATH_CALUDE_largest_integer_before_zero_l629_62903


namespace NUMINAMATH_CALUDE_combined_savings_difference_l629_62938

/-- The cost of a single window -/
def window_cost : ℕ := 100

/-- The number of windows purchased to get one free -/
def windows_for_free : ℕ := 4

/-- The number of windows Dave needs -/
def dave_windows : ℕ := 7

/-- The number of windows Doug needs -/
def doug_windows : ℕ := 8

/-- Calculate the cost of windows with the promotion -/
def cost_with_promotion (n : ℕ) : ℕ :=
  ((n + windows_for_free - 1) / windows_for_free) * windows_for_free * window_cost

/-- Calculate the savings for a given number of windows -/
def savings (n : ℕ) : ℕ :=
  n * window_cost - cost_with_promotion n

/-- The main theorem: combined savings minus individual savings equals $100 -/
theorem combined_savings_difference : 
  savings (dave_windows + doug_windows) - (savings dave_windows + savings doug_windows) = 100 := by
  sorry

end NUMINAMATH_CALUDE_combined_savings_difference_l629_62938


namespace NUMINAMATH_CALUDE_part1_part2_l629_62923

-- Part 1
def f (m x : ℝ) : ℝ := x^2 - (m + 2) * x + 3

def has_max_min_in_range (m : ℝ) : Prop :=
  ∃ (M N : ℝ), (∀ x ∈ Set.Icc 1 2, f m x ≤ M ∧ f m x ≥ N) ∧ M - N ≤ 2

theorem part1 (m : ℝ) : has_max_min_in_range m → m ∈ Set.Icc (-1) 3 := by sorry

-- Part 2
def has_solution_in_range (m : ℝ) : Prop :=
  ∃ x ∈ Set.Icc 0 2, x^2 - (m + 2) * x + 3 = -(2 * m + 1) * x + 2

theorem part2 (m : ℝ) : has_solution_in_range m → m ∈ Set.Iic (-1) := by sorry

end NUMINAMATH_CALUDE_part1_part2_l629_62923


namespace NUMINAMATH_CALUDE_unique_magnitude_quadratic_l629_62967

theorem unique_magnitude_quadratic : ∃! m : ℝ, ∀ z : ℂ, z^2 - 10*z + 50 = 0 → Complex.abs z = m := by
  sorry

end NUMINAMATH_CALUDE_unique_magnitude_quadratic_l629_62967


namespace NUMINAMATH_CALUDE_x_squared_minus_two_is_quadratic_l629_62912

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x² - 2 = 0 is a quadratic equation -/
theorem x_squared_minus_two_is_quadratic :
  is_quadratic_equation (λ x : ℝ ↦ x^2 - 2) :=
by
  sorry


end NUMINAMATH_CALUDE_x_squared_minus_two_is_quadratic_l629_62912


namespace NUMINAMATH_CALUDE_locus_of_Q_l629_62964

-- Define the polar coordinate system
structure PolarCoord where
  ρ : ℝ
  θ : ℝ

-- Define the circle C
def circle_C (p : PolarCoord) : Prop :=
  p.ρ = 2

-- Define the line l
def line_l (p : PolarCoord) : Prop :=
  p.ρ * (Real.cos p.θ + Real.sin p.θ) = 2

-- Define the relationship between points O, P, Q, and R
def point_relationship (P Q R : PolarCoord) : Prop :=
  Q.ρ * P.ρ = R.ρ^2

-- Theorem statement
theorem locus_of_Q (P Q R : PolarCoord) :
  circle_C R →
  line_l P →
  point_relationship P Q R →
  Q.ρ = 2 * (Real.cos Q.θ + Real.sin Q.θ) ∧ Q.ρ ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_locus_of_Q_l629_62964


namespace NUMINAMATH_CALUDE_circumcircle_equation_l629_62934

/-- Given the vertices of triangle ABC: A(4,4), B(5,3), and C(1,1),
    prove that the equation of the circumcircle is x^2 + y^2 - 6x - 4y + 8 = 0 -/
theorem circumcircle_equation (A B C : ℝ × ℝ) :
  A = (4, 4) → B = (5, 3) → C = (1, 1) →
  ∃ D E F : ℝ, ∀ x y : ℝ,
    (x^2 + y^2 + D*x + E*y + F = 0 ↔
     ((x - 4)^2 + (y - 4)^2 = 0 ∨
      (x - 5)^2 + (y - 3)^2 = 0 ∨
      (x - 1)^2 + (y - 1)^2 = 0)) →
    D = -6 ∧ E = -4 ∧ F = 8 := by
  sorry


end NUMINAMATH_CALUDE_circumcircle_equation_l629_62934


namespace NUMINAMATH_CALUDE_no_solution_in_A_l629_62935

-- Define the set A
def A : Set ℕ :=
  {n : ℕ | ∃ k : ℤ, |n * Real.sqrt 2022 - 1/3 - ↑k| ≤ 1/2022}

-- State the theorem
theorem no_solution_in_A :
  ∀ x y z : ℕ, x ∈ A → y ∈ A → z ∈ A → 20 * x + 21 * y ≠ 22 * z :=
by sorry

end NUMINAMATH_CALUDE_no_solution_in_A_l629_62935


namespace NUMINAMATH_CALUDE_eggs_in_box_l629_62916

/-- Given an initial count of eggs and a number of whole eggs added, 
    calculate the total number of whole eggs, ignoring fractional parts. -/
def total_whole_eggs (initial : ℕ) (added : ℕ) : ℕ :=
  initial + added

/-- Theorem stating that with 7 initial eggs and 3 added whole eggs, 
    the total number of whole eggs is 10. -/
theorem eggs_in_box : total_whole_eggs 7 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_eggs_in_box_l629_62916


namespace NUMINAMATH_CALUDE_rectangular_toilet_area_l629_62911

theorem rectangular_toilet_area :
  let length : ℝ := 5
  let width : ℝ := 17 / 20
  let area := length * width
  area = 4.25 := by sorry

end NUMINAMATH_CALUDE_rectangular_toilet_area_l629_62911


namespace NUMINAMATH_CALUDE_probability_shaded_isosceles_triangle_l629_62987

/-- Represents a game board shaped like an isosceles triangle -/
structure GameBoard where
  regions : ℕ
  shaded_regions : ℕ

/-- Calculates the probability of landing in a shaded region -/
def probability_shaded (board : GameBoard) : ℚ :=
  board.shaded_regions / board.regions

theorem probability_shaded_isosceles_triangle :
  ∀ (board : GameBoard),
    board.regions = 7 →
    board.shaded_regions = 3 →
    probability_shaded board = 3 / 7 := by
  sorry

#eval probability_shaded { regions := 7, shaded_regions := 3 }

end NUMINAMATH_CALUDE_probability_shaded_isosceles_triangle_l629_62987


namespace NUMINAMATH_CALUDE_expression_evaluation_l629_62966

theorem expression_evaluation (m n : ℚ) (hm : m = -1) (hn : n = 1/2) :
  (2 / (m - n) - 1 / (m + n)) / ((m * n + 3 * n^2) / (m^3 - m * n^2)) = -2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l629_62966


namespace NUMINAMATH_CALUDE_sum_of_squares_roots_l629_62997

theorem sum_of_squares_roots (a : ℝ) : 
  (∃ x y : ℝ, x^2 + a*x + 2*a = 0 ∧ y^2 + a*y + 2*a = 0 ∧ x^2 + y^2 = 21) ↔ a = -3 :=
sorry

end NUMINAMATH_CALUDE_sum_of_squares_roots_l629_62997


namespace NUMINAMATH_CALUDE_area_FDBG_is_155_l629_62921

/-- Triangle ABC with given properties -/
structure Triangle :=
  (A B C : ℝ × ℝ)
  (AB : Real)
  (AC : Real)
  (area : Real)
  (h_AB : AB = 60)
  (h_AC : AC = 15)
  (h_area : area = 180)

/-- Point D on AB -/
def D (t : Triangle) : ℝ × ℝ :=
  sorry

/-- Point E on AC -/
def E (t : Triangle) : ℝ × ℝ :=
  sorry

/-- Point F on DE and angle bisector of BAC -/
def F (t : Triangle) : ℝ × ℝ :=
  sorry

/-- Point G on BC and angle bisector of BAC -/
def G (t : Triangle) : ℝ × ℝ :=
  sorry

/-- Length of AD -/
def AD (t : Triangle) : Real :=
  20

/-- Length of DB -/
def DB (t : Triangle) : Real :=
  40

/-- Length of AE -/
def AE (t : Triangle) : Real :=
  5

/-- Length of EC -/
def EC (t : Triangle) : Real :=
  10

/-- Area of quadrilateral FDBG -/
def area_FDBG (t : Triangle) : Real :=
  sorry

/-- Main theorem: Area of FDBG is 155 -/
theorem area_FDBG_is_155 (t : Triangle) :
  area_FDBG t = 155 := by
  sorry

end NUMINAMATH_CALUDE_area_FDBG_is_155_l629_62921


namespace NUMINAMATH_CALUDE_factor_probability_l629_62979

/-- The number of factors of m -/
def d (m : ℕ) : ℕ := (Nat.divisors m).card

/-- The probability of selecting a factor of m from 1 to m -/
def prob_factor (m : ℕ) : ℚ := (d m : ℚ) / m

theorem factor_probability (m : ℕ) (p : ℕ) (h : prob_factor m = p / 39) : p = 4 := by
  sorry

end NUMINAMATH_CALUDE_factor_probability_l629_62979


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l629_62968

theorem necessary_but_not_sufficient_condition (a : ℝ) : 
  (∀ x, -1 ≤ x ∧ x < 2 → x ≤ a) ∧ 
  (∃ x, x ≤ a ∧ (x < -1 ∨ x ≥ 2)) →
  a ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l629_62968


namespace NUMINAMATH_CALUDE_eighth_grade_class_problem_l629_62952

theorem eighth_grade_class_problem (total students_math students_foreign : ℕ) 
  (h_total : total = 93)
  (h_math : students_math = 70)
  (h_foreign : students_foreign = 54) :
  students_math - (total - (students_math + students_foreign - total)) = 39 := by
  sorry

end NUMINAMATH_CALUDE_eighth_grade_class_problem_l629_62952


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l629_62910

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property 
  (a : ℕ → ℝ) 
  (h_arithmetic : is_arithmetic_sequence a) 
  (h_sum : a 3 + a 4 + a 5 + a 6 + a 7 = 45) : 
  a 5 = 9 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l629_62910


namespace NUMINAMATH_CALUDE_hyperbola_circle_eccentricity_l629_62919

/-- Given a hyperbola with equation x^2 - ny^2 = 1 and eccentricity 2, 
    prove that the eccentricity of the circle x^2 + ny^2 = 1 is √6/3 -/
theorem hyperbola_circle_eccentricity (m n : ℝ) (hm : m > 0) (hn : n > 0) 
  (h_hyperbola_ecc : (m⁻¹ + n⁻¹) / m⁻¹ = 4) :
  Real.sqrt ((n⁻¹ - m⁻¹) / n⁻¹) = Real.sqrt 6 / 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_circle_eccentricity_l629_62919


namespace NUMINAMATH_CALUDE_correct_arrangement_plans_group_composition_l629_62905

/-- The number of ways to divide 2 teachers and 4 students into two groups -/
def arrangement_plans (num_teachers : ℕ) (num_students : ℕ) : ℕ :=
  if num_teachers = 2 ∧ num_students = 4 then
    12
  else
    0

/-- Theorem stating that the number of arrangement plans is 12 -/
theorem correct_arrangement_plans :
  arrangement_plans 2 4 = 12 := by
  sorry

/-- Theorem stating that each group must have 1 teacher and 2 students -/
theorem group_composition (num_teachers : ℕ) (num_students : ℕ) :
  arrangement_plans num_teachers num_students ≠ 0 →
  num_teachers = 2 ∧ num_students = 4 := by
  sorry

end NUMINAMATH_CALUDE_correct_arrangement_plans_group_composition_l629_62905


namespace NUMINAMATH_CALUDE_sequence_difference_l629_62915

theorem sequence_difference (a : ℕ → ℤ) (h : ∀ n, a (n + 1) - a n - n = 0) : 
  a 2017 - a 2016 = 2016 := by
sorry

end NUMINAMATH_CALUDE_sequence_difference_l629_62915


namespace NUMINAMATH_CALUDE_cos_theta_value_l629_62994

theorem cos_theta_value (θ : Real) 
  (h1 : 10 * Real.tan θ = 4 * Real.cos θ) 
  (h2 : 0 < θ) 
  (h3 : θ < Real.pi) : 
  Real.cos θ = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_theta_value_l629_62994


namespace NUMINAMATH_CALUDE_min_value_trig_expression_l629_62927

theorem min_value_trig_expression (θ φ : ℝ) :
  (3 * Real.cos θ + 4 * Real.sin φ - 10)^2 + (3 * Real.sin θ + 4 * Real.cos φ - 20)^2 ≥ 549 - 140 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_l629_62927


namespace NUMINAMATH_CALUDE_sin_two_alpha_value_l629_62965

theorem sin_two_alpha_value (α : Real) 
  (h1 : α ∈ Set.Ioo (π/2) π) 
  (h2 : 2 * Real.cos (2*α) = Real.sin (π/4 - α)) : 
  Real.sin (2*α) = -7/8 := by
  sorry

end NUMINAMATH_CALUDE_sin_two_alpha_value_l629_62965


namespace NUMINAMATH_CALUDE_sin_15_mul_sin_75_eq_quarter_l629_62962

theorem sin_15_mul_sin_75_eq_quarter : Real.sin (15 * π / 180) * Real.sin (75 * π / 180) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_15_mul_sin_75_eq_quarter_l629_62962


namespace NUMINAMATH_CALUDE_initial_average_calculation_l629_62980

/-- Calculates the initial average daily production given the number of days,
    today's production, and the new average. -/
def initial_average (n : ℕ) (today_production : ℕ) (new_average : ℕ) : ℚ :=
  ((n + 1 : ℕ) * new_average - today_production) / n

theorem initial_average_calculation :
  initial_average 12 115 55 = 50 := by sorry

end NUMINAMATH_CALUDE_initial_average_calculation_l629_62980


namespace NUMINAMATH_CALUDE_function_inequality_l629_62941

open Set

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x < 0, 2 * f x + x * deriv f x > x^2) :
  {x : ℝ | (x + 2017)^2 * f (x + 2017) - 4 * f (-2) > 0} = Iio (-2019) := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l629_62941


namespace NUMINAMATH_CALUDE_mango_rate_calculation_l629_62917

def grape_weight : ℝ := 7
def grape_rate : ℝ := 68
def mango_weight : ℝ := 9
def total_paid : ℝ := 908

theorem mango_rate_calculation :
  (total_paid - grape_weight * grape_rate) / mango_weight = 48 := by
  sorry

end NUMINAMATH_CALUDE_mango_rate_calculation_l629_62917


namespace NUMINAMATH_CALUDE_negative_one_greater_than_negative_two_l629_62942

theorem negative_one_greater_than_negative_two : -1 > -2 := by
  sorry

end NUMINAMATH_CALUDE_negative_one_greater_than_negative_two_l629_62942


namespace NUMINAMATH_CALUDE_rosemary_pots_correct_l629_62907

/-- The number of pots of rosemary Annie planted -/
def rosemary_pots : ℕ := 
  let basil_pots : ℕ := 3
  let thyme_pots : ℕ := 6
  let basil_leaves_per_pot : ℕ := 4
  let rosemary_leaves_per_pot : ℕ := 18
  let thyme_leaves_per_pot : ℕ := 30
  let total_leaves : ℕ := 354
  9

theorem rosemary_pots_correct : 
  let basil_pots : ℕ := 3
  let thyme_pots : ℕ := 6
  let basil_leaves_per_pot : ℕ := 4
  let rosemary_leaves_per_pot : ℕ := 18
  let thyme_leaves_per_pot : ℕ := 30
  let total_leaves : ℕ := 354
  rosemary_pots * rosemary_leaves_per_pot + 
  basil_pots * basil_leaves_per_pot + 
  thyme_pots * thyme_leaves_per_pot = total_leaves :=
by sorry

end NUMINAMATH_CALUDE_rosemary_pots_correct_l629_62907


namespace NUMINAMATH_CALUDE_complex_arithmetic_evaluation_l629_62963

theorem complex_arithmetic_evaluation :
  1234562 - ((12 * 3 * (2 + 7))^2 / 6) + 18 = 1217084 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_evaluation_l629_62963


namespace NUMINAMATH_CALUDE_basketball_prices_and_discounts_l629_62998

/-- Represents the prices and quantities of basketballs --/
structure BasketballPrices where
  price_a : ℝ
  price_b : ℝ
  quantity_a : ℝ
  quantity_b : ℝ

/-- Represents the two discount options --/
inductive DiscountOption
  | Option1
  | Option2

/-- The main theorem about basketball prices and discount options --/
theorem basketball_prices_and_discounts 
  (prices : BasketballPrices)
  (x : ℝ)
  (h1 : prices.price_a = prices.price_b + 40)
  (h2 : 1200 / prices.price_a = 600 / prices.price_b)
  (h3 : x ≥ 5) :
  prices.price_a = 80 ∧ 
  prices.price_b = 40 ∧
  (∀ y : ℝ, y > 20 → 
    (0.9 * (80 * 15 + 40 * y) < 80 * 15 + 40 * (y - 15 / 3))) ∧
  (∀ y : ℝ, y < 20 → 
    (0.9 * (80 * 15 + 40 * y) > 80 * 15 + 40 * (y - 15 / 3))) ∧
  (0.9 * (80 * 15 + 40 * 20) = 80 * 15 + 40 * (20 - 15 / 3)) := by
  sorry

#check basketball_prices_and_discounts

end NUMINAMATH_CALUDE_basketball_prices_and_discounts_l629_62998


namespace NUMINAMATH_CALUDE_negative_abs_negative_three_l629_62982

theorem negative_abs_negative_three : -|-3| = -3 := by
  sorry

end NUMINAMATH_CALUDE_negative_abs_negative_three_l629_62982


namespace NUMINAMATH_CALUDE_m_range_l629_62989

theorem m_range (m : ℝ) : 
  (∃ x : ℝ, m * x^2 + 1 ≤ 0) ∧ 
  (∀ x : ℝ, x^2 + m * x + 1 > 0) → 
  m > -2 ∧ m < 0 :=
by sorry

end NUMINAMATH_CALUDE_m_range_l629_62989


namespace NUMINAMATH_CALUDE_max_min_quadratic_form_l629_62943

theorem max_min_quadratic_form (x y : ℝ) (h : x^2 + 4*y^2 = 4) :
  ∃ (max min : ℝ), max = 6 ∧ min = 2 ∧
  (∀ z, z = x^2 + 2*x*y + 4*y^2 → z ≤ max ∧ z ≥ min) :=
by sorry

end NUMINAMATH_CALUDE_max_min_quadratic_form_l629_62943


namespace NUMINAMATH_CALUDE_joans_sandwiches_l629_62908

/-- Given the conditions for Joan's sandwich making, prove the number of grilled cheese sandwiches. -/
theorem joans_sandwiches (total_cheese : ℕ) (ham_sandwiches : ℕ) (cheese_per_ham : ℕ) (cheese_per_grilled : ℕ)
  (h_total : total_cheese = 50)
  (h_ham : ham_sandwiches = 10)
  (h_cheese_ham : cheese_per_ham = 2)
  (h_cheese_grilled : cheese_per_grilled = 3) :
  (total_cheese - ham_sandwiches * cheese_per_ham) / cheese_per_grilled = 10 := by
  sorry

#eval (50 - 10 * 2) / 3  -- Expected output: 10

end NUMINAMATH_CALUDE_joans_sandwiches_l629_62908


namespace NUMINAMATH_CALUDE_egyptian_fraction_sum_l629_62961

theorem egyptian_fraction_sum : ∃! (b₂ b₃ b₄ b₅ : ℤ),
  (3 : ℚ) / 5 = (b₂ : ℚ) / 2 + (b₃ : ℚ) / 6 + (b₄ : ℚ) / 24 + (b₅ : ℚ) / 120 ∧
  (0 ≤ b₂ ∧ b₂ < 2) ∧
  (0 ≤ b₃ ∧ b₃ < 3) ∧
  (0 ≤ b₄ ∧ b₄ < 4) ∧
  (0 ≤ b₅ ∧ b₅ < 5) ∧
  b₂ + b₃ + b₄ + b₅ = 5 := by
sorry

end NUMINAMATH_CALUDE_egyptian_fraction_sum_l629_62961


namespace NUMINAMATH_CALUDE_problem_solution_l629_62993

theorem problem_solution : (3 - Real.pi) ^ 0 - 3 ^ (-1 : ℤ) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l629_62993


namespace NUMINAMATH_CALUDE_octal_addition_l629_62902

/-- Converts a base-10 integer to its octal representation -/
def to_octal (n : ℕ) : ℕ := sorry

/-- Converts an octal representation to base-10 integer -/
def from_octal (n : ℕ) : ℕ := sorry

theorem octal_addition : to_octal (from_octal 321 + from_octal 127) = 450 := by sorry

end NUMINAMATH_CALUDE_octal_addition_l629_62902


namespace NUMINAMATH_CALUDE_min_students_in_class_l629_62954

theorem min_students_in_class (boys girls : ℕ) : 
  boys > 0 → 
  girls > 0 → 
  (3 * boys) % 4 = 0 → 
  (3 * boys) / 4 = girls / 2 → 
  5 ≤ boys + girls :=
sorry

end NUMINAMATH_CALUDE_min_students_in_class_l629_62954


namespace NUMINAMATH_CALUDE_cone_volume_l629_62958

/-- The volume of a cone with given slant height and height --/
theorem cone_volume (slant_height height : ℝ) (h1 : slant_height = 15) (h2 : height = 9) :
  (1 / 3 : ℝ) * Real.pi * (slant_height ^ 2 - height ^ 2) * height = 432 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_l629_62958


namespace NUMINAMATH_CALUDE_sum_of_sixth_powers_l629_62956

theorem sum_of_sixth_powers (ζ₁ ζ₂ ζ₃ : ℂ) 
  (h1 : ζ₁ + ζ₂ + ζ₃ = 2)
  (h2 : ζ₁^2 + ζ₂^2 + ζ₃^2 = 5)
  (h3 : ζ₁^3 + ζ₂^3 + ζ₃^3 = 14) :
  ζ₁^6 + ζ₂^6 + ζ₃^6 = 128.75 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_sixth_powers_l629_62956


namespace NUMINAMATH_CALUDE_monotonic_unique_zero_l629_62932

/-- A function f is monotonic on (a, b) -/
def Monotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → (f x < f y ∨ f y < f x)

/-- f has exactly one zero in [a, b] -/
def HasUniqueZero (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃! x, a ≤ x ∧ x ≤ b ∧ f x = 0

theorem monotonic_unique_zero (f : ℝ → ℝ) (a b : ℝ) 
  (h1 : Monotonic f a b) (h2 : f a * f b < 0) :
  HasUniqueZero f a b :=
sorry

end NUMINAMATH_CALUDE_monotonic_unique_zero_l629_62932


namespace NUMINAMATH_CALUDE_greatest_c_for_quadratic_range_l629_62969

theorem greatest_c_for_quadratic_range (c : ℤ) : 
  (∀ x : ℝ, x^2 + c*x + 18 ≠ -6) ↔ c ≤ 9 :=
sorry

end NUMINAMATH_CALUDE_greatest_c_for_quadratic_range_l629_62969


namespace NUMINAMATH_CALUDE_plane_speed_problem_l629_62981

theorem plane_speed_problem (speed1 : ℝ) (time : ℝ) (total_distance : ℝ) (speed2 : ℝ) : 
  speed1 = 75 →
  time = 4.84848484848 →
  total_distance = 800 →
  (speed1 + speed2) * time = total_distance →
  speed2 = 90 := by
sorry

end NUMINAMATH_CALUDE_plane_speed_problem_l629_62981


namespace NUMINAMATH_CALUDE_subset_intersection_iff_a_range_l629_62944

/-- Given non-empty sets A and B, prove that A is a subset of (A ∩ B) if and only if 1 ≤ a ≤ 9 -/
theorem subset_intersection_iff_a_range (a : ℝ) :
  let A : Set ℝ := {x | 2 * a + 1 ≤ x ∧ x ≤ 3 * a - 5}
  let B : Set ℝ := {x | 3 ≤ x ∧ x ≤ 22}
  A.Nonempty ∧ B.Nonempty →
  (A ⊆ A ∩ B) ↔ (1 ≤ a ∧ a ≤ 9) :=
by sorry

end NUMINAMATH_CALUDE_subset_intersection_iff_a_range_l629_62944


namespace NUMINAMATH_CALUDE_reaction_enthalpy_positive_l629_62973

-- Define the reaction type
structure Reaction where
  ΔH : ℝ
  ΔS : ℝ

-- Define room temperature as a constant
def roomTemp : ℝ := 298.15  -- in Kelvin

-- Define the spontaneity condition
def isSpontaneous (r : Reaction) (T : ℝ) : Prop :=
  r.ΔH - T * r.ΔS < 0

-- Define the non-spontaneity condition
def isNonSpontaneous (r : Reaction) (T : ℝ) : Prop :=
  r.ΔH - T * r.ΔS > 0

-- Theorem statement
theorem reaction_enthalpy_positive
  (r : Reaction)
  (h1 : isNonSpontaneous r roomTemp)
  (h2 : r.ΔS > 0) :
  r.ΔH > 0 := by
  sorry

end NUMINAMATH_CALUDE_reaction_enthalpy_positive_l629_62973


namespace NUMINAMATH_CALUDE_point_above_line_l629_62914

/-- A point is above a line if its y-coordinate is greater than the y-coordinate of the point on the line with the same x-coordinate. -/
def IsAboveLine (x y : ℝ) (a b c : ℝ) : Prop :=
  y > (a * x + c) / b

/-- The theorem states that for a point P(-2, t) to be above the line 2x - 3y + 6 = 0, t must be greater than 2/3. -/
theorem point_above_line (t : ℝ) :
  IsAboveLine (-2) t 2 (-3) 6 ↔ t > 2/3 := by
  sorry

#check point_above_line

end NUMINAMATH_CALUDE_point_above_line_l629_62914


namespace NUMINAMATH_CALUDE_soccer_camp_ratio_l629_62992

/-- Proves the ratio of kids going to soccer camp in the morning to the total number of kids going to soccer camp -/
theorem soccer_camp_ratio (total_kids : ℕ) (soccer_kids : ℕ) (afternoon_kids : ℕ) 
  (h1 : total_kids = 2000)
  (h2 : soccer_kids = total_kids / 2)
  (h3 : afternoon_kids = 750) :
  (soccer_kids - afternoon_kids) / soccer_kids = 1 / 4 := by
  sorry

#check soccer_camp_ratio

end NUMINAMATH_CALUDE_soccer_camp_ratio_l629_62992


namespace NUMINAMATH_CALUDE_maisy_new_job_earnings_l629_62929

/-- Represents Maisy's job options and calculates the difference in earnings -/
def earnings_difference (current_hours : ℕ) (current_wage : ℕ) (new_hours : ℕ) (new_wage : ℕ) (bonus : ℕ) : ℕ :=
  let current_earnings := current_hours * current_wage
  let new_earnings := new_hours * new_wage + bonus
  new_earnings - current_earnings

/-- Proves that Maisy will earn $15 more at her new job -/
theorem maisy_new_job_earnings :
  earnings_difference 8 10 4 15 35 = 15 := by
  sorry

end NUMINAMATH_CALUDE_maisy_new_job_earnings_l629_62929


namespace NUMINAMATH_CALUDE_padma_valuable_cards_l629_62991

theorem padma_valuable_cards (padma_initial : ℕ) (robert_initial : ℕ) (total_traded : ℕ) 
  (padma_received : ℕ) (robert_received : ℕ) (robert_traded : ℕ) 
  (h1 : padma_initial = 75)
  (h2 : robert_initial = 88)
  (h3 : total_traded = 35)
  (h4 : padma_received = 10)
  (h5 : robert_received = 15)
  (h6 : robert_traded = 8) :
  ∃ (padma_valuable : ℕ), 
    padma_valuable + robert_received = total_traded ∧ 
    padma_valuable = 20 :=
by sorry

end NUMINAMATH_CALUDE_padma_valuable_cards_l629_62991


namespace NUMINAMATH_CALUDE_point_in_third_quadrant_l629_62936

theorem point_in_third_quadrant :
  let A : ℝ × ℝ := (Real.sin (2014 * π / 180), Real.cos (2014 * π / 180))
  A.1 < 0 ∧ A.2 < 0 :=
by sorry

end NUMINAMATH_CALUDE_point_in_third_quadrant_l629_62936


namespace NUMINAMATH_CALUDE_triangle_tangent_inequality_l629_62906

/-- Given a triangle ABC with sides a, b, c, and points A₁, A₂, B₁, B₂, C₁, C₂ defined by lines
    parallel to the opposite sides and tangent to the incircle, prove the inequality. -/
theorem triangle_tangent_inequality 
  (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (htriangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (AA₁ AA₂ BB₁ BB₂ CC₁ CC₂ : ℝ)
  (hAA₁ : AA₁ = b * (b + c - a) / (a + b + c))
  (hAA₂ : AA₂ = c * (b + c - a) / (a + b + c))
  (hBB₁ : BB₁ = c * (c + a - b) / (a + b + c))
  (hBB₂ : BB₂ = a * (c + a - b) / (a + b + c))
  (hCC₁ : CC₁ = a * (a + b - c) / (a + b + c))
  (hCC₂ : CC₂ = b * (a + b - c) / (a + b + c)) :
  AA₁ * AA₂ + BB₁ * BB₂ + CC₁ * CC₂ ≥ (1 / 9) * (a^2 + b^2 + c^2) :=
sorry

end NUMINAMATH_CALUDE_triangle_tangent_inequality_l629_62906


namespace NUMINAMATH_CALUDE_milk_consumption_l629_62986

theorem milk_consumption (total_monitors : ℕ) (monitors_per_group : ℕ) (students_per_group : ℕ)
  (girl_percentage : ℚ) (boy_milk : ℕ) (girl_milk : ℕ) :
  total_monitors = 8 →
  monitors_per_group = 2 →
  students_per_group = 15 →
  girl_percentage = 2/5 →
  boy_milk = 1 →
  girl_milk = 2 →
  (total_monitors / monitors_per_group * students_per_group *
    ((1 - girl_percentage) * boy_milk + girl_percentage * girl_milk) : ℚ) = 84 :=
by sorry

end NUMINAMATH_CALUDE_milk_consumption_l629_62986


namespace NUMINAMATH_CALUDE_circle_tangents_k_range_l629_62999

-- Define the circle C
def C (k : ℝ) (x y : ℝ) : Prop := x^2 + y^2 + 2*k*x + 2*y + k^2 = 0

-- Define the point P
def P : ℝ × ℝ := (1, -1)

-- Define the condition for two tangents
def has_two_tangents (k : ℝ) : Prop := ∃ (t1 t2 : ℝ × ℝ), t1 ≠ t2 ∧ C k t1.1 t1.2 ∧ C k t2.1 t2.2

-- Theorem statement
theorem circle_tangents_k_range :
  ∀ k : ℝ, has_two_tangents k → (k > 0 ∨ k < -2) :=
by sorry

end NUMINAMATH_CALUDE_circle_tangents_k_range_l629_62999


namespace NUMINAMATH_CALUDE_polynomial_root_k_value_l629_62976

theorem polynomial_root_k_value :
  ∀ k : ℚ, (3 : ℚ)^4 + k * (3 : ℚ)^2 - 26 = 0 → k = -55/9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_root_k_value_l629_62976
