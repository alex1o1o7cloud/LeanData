import Mathlib

namespace x_squared_plus_y_squared_l709_70997

theorem x_squared_plus_y_squared (x y : ℝ) : 
  x * y = 8 → x^2 * y + x * y^2 + x + y = 80 → x^2 + y^2 = 5104 / 81 := by
sorry

end x_squared_plus_y_squared_l709_70997


namespace system_solution_l709_70964

theorem system_solution :
  let x : ℚ := -7/3
  let y : ℚ := -1/9
  (4 * x - 3 * y = -9) ∧ (5 * x + 6 * y = -3) := by sorry

end system_solution_l709_70964


namespace log_equality_l709_70982

theorem log_equality (x k : ℝ) :
  (Real.log 3 / Real.log 4 = x) →
  (Real.log 9 / Real.log 2 = k * x) →
  k = 4 := by
sorry

end log_equality_l709_70982


namespace complex_power_four_l709_70957

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_power_four (h : i^2 = -1) : (1 - i)^4 = -4 := by
  sorry

end complex_power_four_l709_70957


namespace minimum_correct_problems_l709_70992

def total_problems : ℕ := 25
def attempted_problems : ℕ := 21
def unanswered_problems : ℕ := total_problems - attempted_problems
def correct_points : ℕ := 7
def incorrect_points : ℤ := -1
def unanswered_points : ℕ := 2
def minimum_score : ℕ := 120

def score (correct : ℕ) : ℤ :=
  (correct * correct_points : ℤ) + 
  ((attempted_problems - correct) * incorrect_points) + 
  (unanswered_problems * unanswered_points)

theorem minimum_correct_problems : 
  ∀ x : ℕ, x ≥ 17 ↔ score x ≥ minimum_score :=
by sorry

end minimum_correct_problems_l709_70992


namespace cost_function_property_l709_70923

/-- A function representing the cost with respect to some parameter b -/
def cost_function (f : ℝ → ℝ) : Prop :=
  ∀ b : ℝ, f (2 * b) = 16 * f b

/-- Theorem stating that if doubling the input results in a cost that is 1600% of the original,
    then f(2b) = 16f(b) for any value of b -/
theorem cost_function_property (f : ℝ → ℝ) (h : ∀ b : ℝ, f (2 * b) = 16 * f b) :
  cost_function f := by sorry

end cost_function_property_l709_70923


namespace circle_distance_to_line_l709_70962

/-- Given a circle (x-a)^2 + (y-a)^2 = 8 and the shortest distance from any point on the circle
    to the line y = -x is √2, prove that a = ±3 -/
theorem circle_distance_to_line (a : ℝ) :
  (∀ x y : ℝ, (x - a)^2 + (y - a)^2 = 8 →
    (∃ d : ℝ, d = Real.sqrt 2 ∧
      ∀ d' : ℝ, d' ≥ 0 → (x + y) / Real.sqrt 2 ≤ d')) →
  a = 3 ∨ a = -3 := by
  sorry

end circle_distance_to_line_l709_70962


namespace simplify_expression_l709_70961

theorem simplify_expression : (27 * (10 ^ 12)) / (9 * (10 ^ 5)) = 30000000 := by
  sorry

end simplify_expression_l709_70961


namespace sales_discount_effect_l709_70934

theorem sales_discount_effect (P N : ℝ) (h_positive : P > 0 ∧ N > 0) :
  let D : ℝ := 10  -- Discount percentage
  let new_price : ℝ := P * (1 - D / 100)
  let new_quantity : ℝ := N * 1.20
  let original_income : ℝ := P * N
  let new_income : ℝ := new_price * new_quantity
  (new_quantity = N * 1.20) ∧ (new_income = original_income * 1.08) :=
by sorry

end sales_discount_effect_l709_70934


namespace range_of_a_for_quadratic_inequality_l709_70943

theorem range_of_a_for_quadratic_inequality 
  (h : ∃ x ∈ Set.Icc 1 2, x^2 + a*x - 2 > 0) :
  a ∈ Set.Ioi (-1) :=
sorry

end range_of_a_for_quadratic_inequality_l709_70943


namespace rational_function_property_l709_70990

theorem rational_function_property (f : ℚ → ℝ) 
  (h : ∀ r s : ℚ, ∃ n : ℤ, f (r + s) - f r - f s = n) :
  ∃ (q : ℕ+) (p : ℤ), |f (1 / q) - p| ≤ 1 / 2012 := by
  sorry

end rational_function_property_l709_70990


namespace square_diff_and_product_l709_70914

theorem square_diff_and_product (a b : ℝ) 
  (sum_eq : a + b = 10) 
  (diff_eq : a - b = 4) 
  (sum_squares_eq : a^2 + b^2 = 58) : 
  a^2 - b^2 = 40 ∧ a * b = 21 := by
  sorry

end square_diff_and_product_l709_70914


namespace tv_show_episodes_l709_70944

/-- Given a TV show with the following properties:
  - There were 9 seasons before a new season was announced
  - The last (10th) season has 4 more episodes than the others
  - Each episode is 0.5 hours long
  - It takes 112 hours to watch all episodes after the last season finishes
  This theorem proves that each season (except the last) has 22 episodes. -/
theorem tv_show_episodes :
  let seasons_before : ℕ := 9
  let extra_episodes_last_season : ℕ := 4
  let episode_length : ℚ := 1/2
  let total_watch_time : ℕ := 112
  let episodes_per_season : ℕ := (2 * total_watch_time - 2 * extra_episodes_last_season) / (2 * seasons_before + 2)
  episodes_per_season = 22 := by
sorry

end tv_show_episodes_l709_70944


namespace guests_per_table_l709_70924

theorem guests_per_table (tables : ℝ) (total_guests : ℕ) 
  (h1 : tables = 252.0) 
  (h2 : total_guests = 1008) : 
  (total_guests : ℝ) / tables = 4 := by
  sorry

end guests_per_table_l709_70924


namespace aunt_may_milk_problem_l709_70988

/-- Aunt May's milk problem -/
theorem aunt_may_milk_problem (morning_milk : ℕ) (evening_milk : ℕ) (sold_milk : ℕ) (leftover_milk : ℕ)
  (h1 : morning_milk = 365)
  (h2 : evening_milk = 380)
  (h3 : sold_milk = 612)
  (h4 : leftover_milk = 15) :
  morning_milk + evening_milk + leftover_milk - sold_milk = 148 :=
by sorry

end aunt_may_milk_problem_l709_70988


namespace fraction_multiplication_l709_70983

theorem fraction_multiplication : ((1 / 4 : ℚ) * (1 / 8 : ℚ)) * 4 = 1 / 8 := by
  sorry

end fraction_multiplication_l709_70983


namespace unique_triangle_l709_70999

/-- A triangle with integer side lengths a, b, c, where a ≤ b ≤ c -/
structure IntegerTriangle where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  a_le_b : a ≤ b
  b_le_c : b ≤ c

/-- The set of all integer triangles with perimeter 10 satisfying the triangle inequality -/
def ValidTriangles : Set IntegerTriangle :=
  {t : IntegerTriangle | t.a + t.b + t.c = 10 ∧ t.a + t.b > t.c}

theorem unique_triangle : ∃! t : IntegerTriangle, t ∈ ValidTriangles :=
sorry

end unique_triangle_l709_70999


namespace rhombus_sides_not_equal_l709_70907

theorem rhombus_sides_not_equal (d1 d2 p : ℝ) (h1 : d1 = 30) (h2 : d2 = 18) (h3 : p = 80) :
  ¬ (4 * Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2) = p) :=
by sorry

end rhombus_sides_not_equal_l709_70907


namespace square_area_ratio_l709_70969

/-- The ratio of the areas of two squares, where one has a side length 5 times the other, is 1/25. -/
theorem square_area_ratio (y : ℝ) (h : y > 0) : 
  (y^2) / ((5*y)^2) = 1 / 25 := by sorry

end square_area_ratio_l709_70969


namespace committee_age_difference_l709_70920

theorem committee_age_difference (n : ℕ) (A : ℝ) (O N : ℝ) : 
  n = 20 → 
  n * A = n * A + O - N → 
  O - N = 160 :=
by
  sorry

end committee_age_difference_l709_70920


namespace oil_leaked_before_work_l709_70937

def total_oil_leaked : ℕ := 11687
def oil_leaked_during_work : ℕ := 5165

theorem oil_leaked_before_work (total : ℕ) (during_work : ℕ) 
  (h1 : total = total_oil_leaked) 
  (h2 : during_work = oil_leaked_during_work) : 
  total - during_work = 6522 := by
  sorry

end oil_leaked_before_work_l709_70937


namespace integral_inequality_l709_70993

open MeasureTheory

theorem integral_inequality 
  (f g : ℝ → ℝ) 
  (hf_pos : ∀ x, 0 ≤ f x) 
  (hg_pos : ∀ x, 0 ≤ g x)
  (hf_cont : Continuous f) 
  (hg_cont : Continuous g)
  (hf_incr : MonotoneOn f (Set.Icc 0 1))
  (hg_decr : AntitoneOn g (Set.Icc 0 1)) :
  ∫ x in (Set.Icc 0 1), f x * g x ≤ ∫ x in (Set.Icc 0 1), f x * g (1 - x) :=
sorry

end integral_inequality_l709_70993


namespace fraction_equality_l709_70959

theorem fraction_equality (a b : ℚ) 
  (h1 : a = 1/2) 
  (h2 : b = 2/3) : 
  (6*a + 18*b) / (12*a + 6*b) = 3/2 := by
sorry

end fraction_equality_l709_70959


namespace function_interval_theorem_l709_70958

theorem function_interval_theorem (a b : Real) :
  let f := fun x => -1/2 * x^2 + 13/2
  (∀ x ∈ Set.Icc a b, f x ≥ 2*a) ∧
  (∀ x ∈ Set.Icc a b, f x ≤ 2*b) ∧
  (∃ x ∈ Set.Icc a b, f x = 2*a) ∧
  (∃ x ∈ Set.Icc a b, f x = 2*b) →
  ((a = 1 ∧ b = 3) ∨ (a = -2 - Real.sqrt 17 ∧ b = 13/4)) := by
  sorry

#check function_interval_theorem

end function_interval_theorem_l709_70958


namespace monday_temperature_l709_70918

theorem monday_temperature
  (avg_mon_to_thu : (mon + tue + wed + thu) / 4 = 48)
  (avg_tue_to_fri : (tue + wed + thu + 36) / 4 = 46)
  : mon = 44 := by
  sorry

end monday_temperature_l709_70918


namespace number_division_problem_l709_70938

theorem number_division_problem (x : ℝ) : x / 5 = 100 + x / 6 → x = 3000 := by
  sorry

end number_division_problem_l709_70938


namespace point_set_classification_l709_70940

-- Define the type for 2D points
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the distance squared between two points
def distanceSquared (p q : Point2D) : ℝ :=
  (p.x - q.x)^2 + (p.y - q.y)^2

-- Define the equation
def satisfiesEquation (X : Point2D) (A : List Point2D) (k : List ℝ) (c : ℝ) : Prop :=
  (List.zip k A).foldl (λ sum (kᵢ, Aᵢ) => sum + kᵢ * distanceSquared Aᵢ X) 0 = c

-- State the theorem
theorem point_set_classification 
  (A : List Point2D) (k : List ℝ) (c : ℝ) 
  (h_length : A.length = k.length) :
  (k.sum ≠ 0 → 
    (∃ center : Point2D, ∃ radius : ℝ, 
      ∀ X, satisfiesEquation X A k c ↔ distanceSquared center X = radius^2) ∨
    (∀ X, ¬satisfiesEquation X A k c)) ∧
  (k.sum = 0 → 
    (∃ a b d : ℝ, ∀ X, satisfiesEquation X A k c ↔ a * X.x + b * X.y = d) ∨
    (∀ X, ¬satisfiesEquation X A k c)) :=
sorry

end point_set_classification_l709_70940


namespace combination_problem_l709_70960

theorem combination_problem (m : ℕ) : 
  (1 : ℚ) / (Nat.choose 5 m) - (1 : ℚ) / (Nat.choose 6 m) = (7 : ℚ) / (10 * Nat.choose 7 m) →
  Nat.choose 21 m = 210 := by
  sorry

end combination_problem_l709_70960


namespace ratio_is_three_halves_l709_70950

/-- Represents a rectangular parallelepiped with dimensions a, b, and c -/
structure RectangularParallelepiped (α : Type*) [LinearOrderedField α] where
  a : α
  b : α
  c : α

/-- The ratio of the sum of squares of sides of triangle KLM to the square of the parallelepiped's diagonal -/
def triangle_to_diagonal_ratio {α : Type*} [LinearOrderedField α] (p : RectangularParallelepiped α) : α :=
  (3 : α) / 2

/-- Theorem stating that the ratio is always 3/2 for any rectangular parallelepiped -/
theorem ratio_is_three_halves {α : Type*} [LinearOrderedField α] (p : RectangularParallelepiped α) :
  triangle_to_diagonal_ratio p = (3 : α) / 2 := by
  sorry

#check ratio_is_three_halves

end ratio_is_three_halves_l709_70950


namespace divisor_problem_l709_70967

theorem divisor_problem (D : ℚ) : 
  (1280 + 720) / 125 = 7392 / D → D = 462 := by
  sorry

end divisor_problem_l709_70967


namespace necessary_not_sufficient_condition_l709_70991

theorem necessary_not_sufficient_condition (a b : ℝ) : 
  (∀ a b : ℝ, a < b → a < b + 1) ∧ 
  (∃ a b : ℝ, a < b + 1 ∧ ¬(a < b)) := by
  sorry

end necessary_not_sufficient_condition_l709_70991


namespace larger_number_of_sum_and_difference_l709_70946

theorem larger_number_of_sum_and_difference (x y : ℝ) : 
  x + y = 40 → x - y = 4 → max x y = 22 := by
  sorry

end larger_number_of_sum_and_difference_l709_70946


namespace unique_divisor_of_18_l709_70975

def divides (m n : ℕ) : Prop := ∃ k, n = m * k

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem unique_divisor_of_18 : ∃! a : ℕ, 
  divides 3 a ∧ 
  divides a 18 ∧ 
  Even (sum_of_digits a) := by sorry

end unique_divisor_of_18_l709_70975


namespace find_a_l709_70927

-- Define the universal set U
def U (a : ℝ) : Set ℝ := {2, 4, 3 - a^2}

-- Define set P
def P (a : ℝ) : Set ℝ := {2, a^2 - a + 2}

-- Define the complement of P with respect to U
def complement_P (a : ℝ) : Set ℝ := {-1}

-- Theorem statement
theorem find_a : ∃ a : ℝ, (U a = P a ∪ complement_P a) ∧ (a = 2) := by
  sorry

end find_a_l709_70927


namespace unique_solution_quadratic_l709_70987

/-- Given a quadratic equation ax^2 + 10x + c = 0 with exactly one solution,
    where a + c = 12 and a < c, prove that a = 6 - √11 and c = 6 + √11 -/
theorem unique_solution_quadratic (a c : ℝ) : 
  (∃! x, a * x^2 + 10 * x + c = 0) → 
  a + c = 12 → 
  a < c → 
  (a = 6 - Real.sqrt 11 ∧ c = 6 + Real.sqrt 11) := by
  sorry

end unique_solution_quadratic_l709_70987


namespace power_five_mod_seven_l709_70954

theorem power_five_mod_seven : 5^207 % 7 = 6 := by
  sorry

end power_five_mod_seven_l709_70954


namespace average_age_increase_l709_70945

theorem average_age_increase (num_students : ℕ) (student_avg_age : ℝ) (teacher_age : ℝ) :
  num_students = 19 →
  student_avg_age = 20 →
  teacher_age = 40 →
  ((num_students : ℝ) * student_avg_age + teacher_age) / ((num_students : ℝ) + 1) - student_avg_age = 1 :=
by sorry

end average_age_increase_l709_70945


namespace swimming_problem_l709_70968

/-- Proves that Jamir swims 20 more meters per day than Sarah given the conditions of the swimming problem. -/
theorem swimming_problem (julien sarah jamir : ℕ) : 
  julien = 50 →  -- Julien swims 50 meters per day
  sarah = 2 * julien →  -- Sarah swims twice the distance Julien swims
  jamir > sarah →  -- Jamir swims some more meters per day than Sarah
  7 * (julien + sarah + jamir) = 1890 →  -- Combined distance for the week
  jamir - sarah = 20 := by  -- Jamir swims 20 more meters per day than Sarah
sorry

end swimming_problem_l709_70968


namespace marbles_remaining_l709_70926

/-- Calculates the number of marbles remaining in a store after sales. -/
theorem marbles_remaining 
  (initial_marbles : ℕ) 
  (num_customers : ℕ) 
  (marbles_per_customer : ℕ) 
  (h1 : initial_marbles = 400)
  (h2 : num_customers = 20)
  (h3 : marbles_per_customer = 15) : 
  initial_marbles - num_customers * marbles_per_customer = 100 := by
sorry

end marbles_remaining_l709_70926


namespace planes_distance_zero_l709_70963

-- Define the planes
def plane1 (x y z : ℝ) : Prop := x + 2*y - z = 3
def plane2 (x y z : ℝ) : Prop := 2*x + 4*y - 2*z = 6

-- Define the distance function between two planes
noncomputable def distance_between_planes (p1 p2 : ℝ → ℝ → ℝ → Prop) : ℝ := sorry

-- Theorem statement
theorem planes_distance_zero :
  distance_between_planes plane1 plane2 = 0 := by sorry

end planes_distance_zero_l709_70963


namespace company_c_cheapest_l709_70974

-- Define the pricing structures for each company
def company_a_cost (miles : ℝ) : ℝ :=
  2.10 + 0.40 * (miles * 5 - 1)

def company_b_cost (miles : ℝ) : ℝ :=
  3.00 + 0.50 * (miles * 4 - 1)

def company_c_cost (miles : ℝ) : ℝ :=
  1.50 * miles + 2.00

-- Theorem statement
theorem company_c_cheapest :
  let journey_length : ℝ := 8
  company_c_cost journey_length < company_a_cost journey_length ∧
  company_c_cost journey_length < company_b_cost journey_length :=
by sorry

end company_c_cheapest_l709_70974


namespace abs_two_over_one_plus_i_l709_70900

theorem abs_two_over_one_plus_i : Complex.abs (2 / (1 + Complex.I)) = Real.sqrt 2 := by
  sorry

end abs_two_over_one_plus_i_l709_70900


namespace xyz_sign_sum_l709_70933

theorem xyz_sign_sum (x y z : ℝ) (h : x * y * z / |x * y * z| = 1) :
  |x| / x + y / |y| + |z| / z = -1 ∨ |x| / x + y / |y| + |z| / z = 3 :=
by sorry

end xyz_sign_sum_l709_70933


namespace quadratic_equation_sum_l709_70901

theorem quadratic_equation_sum (x q t : ℝ) : 
  (9 * x^2 - 36 * x - 81 = 0) →
  ((x + q)^2 = t) →
  (9 * (x + q)^2 = 9 * t) →
  (q + t = 11) := by
sorry

end quadratic_equation_sum_l709_70901


namespace farmer_tomatoes_l709_70912

def initial_tomatoes (picked_yesterday : ℕ) (picked_today : ℕ) (remaining : ℕ) : ℕ :=
  picked_yesterday + picked_today + remaining

theorem farmer_tomatoes : initial_tomatoes 134 30 7 = 171 := by
  sorry

end farmer_tomatoes_l709_70912


namespace linear_function_intersection_l709_70952

theorem linear_function_intersection (k : ℝ) : 
  (∃ x : ℝ, k * x + 2 = 0 ∧ abs x = 4) → k = 1/2 ∨ k = -1/2 := by
  sorry

end linear_function_intersection_l709_70952


namespace train_length_and_speed_l709_70935

/-- Proves the length and speed of a train given its crossing times over two platforms. -/
theorem train_length_and_speed 
  (train_length : ℝ) 
  (train_speed : ℝ) 
  (platform1_length : ℝ) 
  (platform2_length : ℝ) 
  (time1 : ℝ) 
  (time2 : ℝ) 
  (h1 : platform1_length = 90) 
  (h2 : platform2_length = 120) 
  (h3 : time1 = 12) 
  (h4 : time2 = 15) 
  (h5 : train_speed * time1 = train_length + platform1_length) 
  (h6 : train_speed * time2 = train_length + platform2_length) : 
  train_length = 30 ∧ train_speed = 10 := by
  sorry

#check train_length_and_speed

end train_length_and_speed_l709_70935


namespace ages_solution_l709_70902

/-- Represents the ages of Rahul and Deepak -/
structure Ages where
  rahul : ℕ
  deepak : ℕ

/-- The conditions of the problem -/
def satisfies_conditions (ages : Ages) : Prop :=
  -- The ratio between Rahul and Deepak's age is 4:3
  4 * ages.deepak = 3 * ages.rahul ∧
  -- In 6 years, Rahul will be 26 years old
  ages.rahul + 6 = 26 ∧
  -- In 6 years, Deepak's age will be equal to half the sum of Rahul's present and future ages
  ages.deepak + 6 = (ages.rahul + (ages.rahul + 6)) / 2 ∧
  -- Five years after that, the sum of their ages will be 59
  (ages.rahul + 11) + (ages.deepak + 11) = 59

/-- The theorem to prove -/
theorem ages_solution :
  ∃ (ages : Ages), satisfies_conditions ages ∧ ages.rahul = 20 ∧ ages.deepak = 17 := by
  sorry

end ages_solution_l709_70902


namespace train_speed_l709_70915

/-- Given a train of length 360 meters passing a bridge of length 140 meters in 36 seconds,
    prove that its speed is 50 km/h. -/
theorem train_speed (train_length : ℝ) (bridge_length : ℝ) (time : ℝ) :
  train_length = 360 →
  bridge_length = 140 →
  time = 36 →
  (train_length + bridge_length) / time * 3.6 = 50 := by
  sorry

end train_speed_l709_70915


namespace garden_length_is_140_l709_70913

/-- Represents a rectangular garden -/
structure RectangularGarden where
  length : ℝ
  breadth : ℝ

/-- Calculates the perimeter of a rectangular garden -/
def perimeter (g : RectangularGarden) : ℝ :=
  2 * (g.length + g.breadth)

/-- Theorem: A rectangular garden with perimeter 480 m and breadth 100 m has length 140 m -/
theorem garden_length_is_140
  (g : RectangularGarden)
  (h1 : perimeter g = 480)
  (h2 : g.breadth = 100) :
  g.length = 140 := by
  sorry

end garden_length_is_140_l709_70913


namespace ceiling_sum_equals_56_l709_70909

theorem ceiling_sum_equals_56 :
  ⌈Real.sqrt 3⌉ + ⌈Real.sqrt 30⌉^2 + ⌈Real.sqrt 300⌉ = 56 := by
  sorry

end ceiling_sum_equals_56_l709_70909


namespace perfect_square_trinomial_condition_l709_70932

/-- A trinomial ax^2 + bx + c is a perfect square if there exist real numbers p and q
    such that ax^2 + bx + c = (px + q)^2 for all x. -/
def is_perfect_square_trinomial (a b c : ℝ) : Prop :=
  ∃ p q : ℝ, ∀ x, a * x^2 + b * x + c = (p * x + q)^2

/-- If 4x^2 + 2kx + 9 is a perfect square trinomial, then k = ±6. -/
theorem perfect_square_trinomial_condition (k : ℝ) :
  is_perfect_square_trinomial 4 (2 * k) 9 → k = 6 ∨ k = -6 :=
by
  sorry


end perfect_square_trinomial_condition_l709_70932


namespace geometric_sequence_condition_l709_70951

/-- A sequence of real numbers -/
def Sequence := ℕ → ℝ

/-- Condition for three consecutive terms of a sequence to form a geometric sequence -/
def IsGeometric (a : Sequence) (n : ℕ) : Prop :=
  ∃ r : ℝ, a (n + 1) = a n * r ∧ a (n + 2) = a (n + 1) * r

/-- The condition a_{n+1}^2 = a_n * a_{n+2} -/
def SquareMiddleCondition (a : Sequence) (n : ℕ) : Prop :=
  a (n + 1) ^ 2 = a n * a (n + 2)

theorem geometric_sequence_condition (a : Sequence) :
  (∀ n : ℕ, IsGeometric a n → SquareMiddleCondition a n) ∧
  ¬(∀ n : ℕ, SquareMiddleCondition a n → IsGeometric a n) :=
sorry

end geometric_sequence_condition_l709_70951


namespace union_of_M_and_N_l709_70981

def M : Set ℝ := {x : ℝ | -3 < x ∧ x ≤ 5}
def N : Set ℝ := {x : ℝ | x > 3}

theorem union_of_M_and_N : M ∪ N = {x : ℝ | x > -3} := by sorry

end union_of_M_and_N_l709_70981


namespace quadratic_form_sum_l709_70939

theorem quadratic_form_sum (x : ℝ) : ∃ (d e : ℝ), 
  (∀ x, x^2 - 24*x + 50 = (x + d)^2 + e) ∧ d + e = -106 := by
  sorry

end quadratic_form_sum_l709_70939


namespace negation_of_universal_proposition_l709_70917

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 ≤ 1) ↔ (∃ x : ℝ, x^2 > 1) := by
  sorry

end negation_of_universal_proposition_l709_70917


namespace quadratic_equation_roots_l709_70971

theorem quadratic_equation_roots (m : ℝ) : 
  let f : ℝ → ℝ := λ x => x^2 - 6*m*x + 9*m^2 - 4
  ∃ x₁ x₂ : ℝ, x₁ > x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ = 2*x₂ → m = 2 := by
  sorry

end quadratic_equation_roots_l709_70971


namespace plane_contains_points_and_normalized_l709_70998

def point1 : ℝ × ℝ × ℝ := (2, -1, 3)
def point2 : ℝ × ℝ × ℝ := (0, 3, 1)
def point3 : ℝ × ℝ × ℝ := (-1, 2, 4)

def plane_equation (x y z : ℝ) := 5*x + 2*y + 3*z - 17

theorem plane_contains_points_and_normalized :
  (plane_equation point1.1 point1.2.1 point1.2.2 = 0) ∧
  (plane_equation point2.1 point2.2.1 point2.2.2 = 0) ∧
  (plane_equation point3.1 point3.2.1 point3.2.2 = 0) ∧
  (5 > 0) ∧
  (Nat.gcd (Nat.gcd (Nat.gcd 5 2) 3) 17 = 1) := by
  sorry

end plane_contains_points_and_normalized_l709_70998


namespace lily_hydrangea_plants_l709_70941

/-- Prove that Lily buys 1 hydrangea plant per year -/
theorem lily_hydrangea_plants (start_year end_year : ℕ) (plant_cost total_spent : ℚ) : 
  start_year = 1989 →
  end_year = 2021 →
  plant_cost = 20 →
  total_spent = 640 →
  (total_spent / (end_year - start_year : ℚ)) / plant_cost = 1 := by
  sorry

end lily_hydrangea_plants_l709_70941


namespace parallelogram_roots_l709_70984

/-- The polynomial equation with parameter b -/
def P (b : ℝ) (z : ℂ) : ℂ :=
  z^4 - 8*z^3 + 15*b*z^2 - 5*(3*b^2 + 4*b - 4)*z + 9

/-- Condition for roots to form a parallelogram -/
def forms_parallelogram (b : ℝ) : Prop :=
  ∃ (w₁ w₂ : ℂ), (P b w₁ = 0) ∧ (P b (-w₁) = 0) ∧ (P b w₂ = 0) ∧ (P b (-w₂) = 0)

/-- The main theorem stating the values of b for which the roots form a parallelogram -/
theorem parallelogram_roots :
  ∀ b : ℝ, forms_parallelogram b ↔ (b = 2/3 ∨ b = -2) :=
sorry

end parallelogram_roots_l709_70984


namespace exists_x_less_than_zero_l709_70925

theorem exists_x_less_than_zero : ∃ x : ℝ, x^2 - 4*x + 3 < 0 := by
  sorry

end exists_x_less_than_zero_l709_70925


namespace smallest_number_l709_70919

theorem smallest_number (S : Set ℚ) (h : S = {-3, -1, 0, 1}) : 
  ∃ (m : ℚ), m ∈ S ∧ ∀ (x : ℚ), x ∈ S → m ≤ x ∧ m = -3 :=
by sorry

end smallest_number_l709_70919


namespace radical_simplification_l709_70928

theorem radical_simplification (x : ℝ) (h : 4 < x ∧ x < 7) :
  (((x - 4) ^ 4) ^ (1/4 : ℝ)) + (((x - 7) ^ 4) ^ (1/4 : ℝ)) = 3 := by
  sorry

end radical_simplification_l709_70928


namespace recurring_decimal_division_l709_70910

def repeating_decimal_to_fraction (a b c : ℕ) : ℚ :=
  (a * 1000 + b * 100 + c * 10 + a * 1 + b * (1/10) + c * (1/100)) / 999

theorem recurring_decimal_division (a b c d e f : ℕ) :
  (repeating_decimal_to_fraction a b c) / (1 + repeating_decimal_to_fraction d e f) = 714 / 419 :=
by
  sorry

end recurring_decimal_division_l709_70910


namespace apples_per_person_is_two_l709_70930

/-- Calculates the number of pounds of apples each person gets in a family -/
def applesPerPerson (originalPrice : ℚ) (priceIncrease : ℚ) (totalCost : ℚ) (familySize : ℕ) : ℚ :=
  let newPrice := originalPrice * (1 + priceIncrease)
  let costPerPerson := totalCost / familySize
  costPerPerson / newPrice

/-- Theorem stating that under the given conditions, each person gets 2 pounds of apples -/
theorem apples_per_person_is_two :
  applesPerPerson (8/5) (1/4) 16 4 = 2 := by
  sorry

end apples_per_person_is_two_l709_70930


namespace seans_apples_l709_70942

/-- Sean's apple problem -/
theorem seans_apples (initial_apples final_apples susans_apples : ℕ) :
  final_apples = initial_apples + susans_apples →
  susans_apples = 8 →
  final_apples = 17 →
  initial_apples = 9 := by
  sorry

end seans_apples_l709_70942


namespace square_diagonals_equal_l709_70980

-- Define the basic shapes
class Rectangle where
  diagonals_equal : Bool

class Square extends Rectangle

-- Define the properties
axiom rectangle_diagonals_equal : ∀ (r : Rectangle), r.diagonals_equal = true

-- Theorem to prove
theorem square_diagonals_equal (s : Square) : s.diagonals_equal = true := by
  sorry

end square_diagonals_equal_l709_70980


namespace ahmed_goats_l709_70911

/-- Given information about goats owned by Adam, Andrew, and Ahmed -/
theorem ahmed_goats (adam : ℕ) (andrew : ℕ) (ahmed : ℕ) : 
  adam = 7 →
  andrew = 2 * adam + 5 →
  ahmed = andrew - 6 →
  ahmed = 13 := by sorry

end ahmed_goats_l709_70911


namespace geometric_sum_problem_l709_70973

/-- Sum of the first n terms of a geometric sequence -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The problem statement -/
theorem geometric_sum_problem :
  let a : ℚ := 1/3
  let r : ℚ := 1/4
  let n : ℕ := 7
  geometric_sum a r n = 16383/12288 := by
sorry

end geometric_sum_problem_l709_70973


namespace wolverine_workout_hours_l709_70906

def workout_hours (rayman junior wolverine : ℕ) : Prop :=
  rayman = junior / 2 ∧ 
  wolverine = 2 * (rayman + junior) ∧ 
  rayman = 10

theorem wolverine_workout_hours :
  ∀ rayman junior wolverine : ℕ,
  workout_hours rayman junior wolverine →
  wolverine = 60 := by
sorry

end wolverine_workout_hours_l709_70906


namespace inscribed_sphere_theorem_l709_70965

/-- A truncated triangular pyramid with an inscribed sphere -/
structure TruncatedPyramid where
  /-- Height of the pyramid -/
  h : ℝ
  /-- Radius of the circle described around the first base -/
  R₁ : ℝ
  /-- Radius of the circle described around the second base -/
  R₂ : ℝ
  /-- Distance between the center of the first base circle and the point where the sphere touches it -/
  O₁T₁ : ℝ
  /-- Distance between the center of the second base circle and the point where the sphere touches it -/
  O₂T₂ : ℝ
  /-- All lengths are positive -/
  h_pos : 0 < h
  R₁_pos : 0 < R₁
  R₂_pos : 0 < R₂
  O₁T₁_pos : 0 < O₁T₁
  O₂T₂_pos : 0 < O₂T₂
  /-- The sphere touches the bases inside the circles -/
  O₁T₁_le_R₁ : O₁T₁ ≤ R₁
  O₂T₂_le_R₂ : O₂T₂ ≤ R₂

/-- The main theorem about the inscribed sphere in a truncated triangular pyramid -/
theorem inscribed_sphere_theorem (p : TruncatedPyramid) :
    p.R₁ * p.R₂ * p.h^2 = (p.R₁^2 - p.O₁T₁^2) * (p.R₂^2 - p.O₂T₂^2) := by
  sorry

end inscribed_sphere_theorem_l709_70965


namespace first_race_length_l709_70985

/-- Represents the length of the first race in meters -/
def L : ℝ := sorry

/-- Theorem stating that the length of the first race is 100 meters -/
theorem first_race_length : L = 100 := by
  -- Define the relationships between runners based on the given conditions
  let A_finish := L
  let B_finish := L - 10
  let C_finish := L - 13
  
  -- Define the relationship in the second race
  let B_second_race := 180
  let C_second_race := 174  -- 180 - 6
  
  -- The ratio of B's performance to C's performance should be consistent across races
  have ratio_equality : (B_finish / C_finish) = (B_second_race / C_second_race) := by sorry
  
  -- Use the ratio equality to solve for L
  sorry

end first_race_length_l709_70985


namespace quadratic_max_l709_70970

/-- The function f(x) = -2x^2 + 8x - 6 achieves its maximum value when x = 2 -/
theorem quadratic_max (x : ℝ) : 
  ∀ y : ℝ, -2 * x^2 + 8 * x - 6 ≥ -2 * y^2 + 8 * y - 6 ↔ x = 2 :=
by sorry

end quadratic_max_l709_70970


namespace domain_of_composite_function_l709_70903

-- Define the function f with domain [-1, 2]
def f : Set ℝ := Set.Icc (-1) 2

-- Define the function g(x) = f(2x-1)
def g (x : ℝ) : ℝ := 2 * x - 1

-- Theorem statement
theorem domain_of_composite_function :
  {x : ℝ | g x ∈ f} = Set.Icc 0 (3/2) := by sorry

end domain_of_composite_function_l709_70903


namespace intersection_of_A_and_B_l709_70989

-- Define the sets A and B
def A : Set (ℝ × ℝ) := {p : ℝ × ℝ | 2 * p.1 - p.2 = 0}
def B : Set (ℝ × ℝ) := {p : ℝ × ℝ | 3 * p.1 + p.2 = 0}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {(0, 0)} := by sorry

end intersection_of_A_and_B_l709_70989


namespace distribution_count_7_l709_70922

/-- The number of ways to distribute n distinct objects into 3 distinct containers
    labeled 1, 2, and 3, such that each container has at least as many objects as its label -/
def distribution_count (n : ℕ) : ℕ :=
  let ways_221 := (n.choose 2) * ((n - 2).choose 2)
  let ways_133 := (n.choose 1) * ((n - 1).choose 3)
  let ways_124 := (n.choose 1) * ((n - 1).choose 2)
  ways_221 + ways_133 + ways_124

/-- Theorem stating that there are 455 ways to distribute 7 distinct objects
    into 3 distinct containers with the given constraints -/
theorem distribution_count_7 : distribution_count 7 = 455 := by
  sorry

end distribution_count_7_l709_70922


namespace highest_class_strength_l709_70996

theorem highest_class_strength (total : ℕ) (g1 g2 g3 : ℕ) : 
  total = 333 →
  g1 + g2 + g3 = total →
  5 * g1 = 3 * g2 →
  11 * g2 = 7 * g3 →
  g1 ≤ g2 ∧ g2 ≤ g3 →
  g3 = 165 := by
sorry

end highest_class_strength_l709_70996


namespace heejin_drinks_most_l709_70904

-- Define the drinking habits
def dongguk_frequency : ℕ := 5
def dongguk_amount : ℝ := 0.2

def yoonji_frequency : ℕ := 6
def yoonji_amount : ℝ := 0.3

def heejin_frequency : ℕ := 4
def heejin_amount : ℝ := 0.5  -- 500 ml = 0.5 L

-- Calculate total daily water intake for each person
def dongguk_total : ℝ := dongguk_frequency * dongguk_amount
def yoonji_total : ℝ := yoonji_frequency * yoonji_amount
def heejin_total : ℝ := heejin_frequency * heejin_amount

-- Theorem stating Heejin drinks the most water
theorem heejin_drinks_most : 
  heejin_total > dongguk_total ∧ heejin_total > yoonji_total :=
by sorry

end heejin_drinks_most_l709_70904


namespace trig_identity_l709_70948

theorem trig_identity (α : ℝ) : 
  (Real.cos (π / 2 - α / 4) - Real.sin (π / 2 - α / 4) * Real.tan (α / 8)) / 
  (Real.sin (7 * π / 2 - α / 4) + Real.sin (α / 4 - 3 * π) * Real.tan (α / 8)) = 
  -Real.tan (α / 8) := by sorry

end trig_identity_l709_70948


namespace interest_rate_problem_l709_70953

/-- Given a sum P at simple interest rate R for 3 years, if increasing the rate by 1%
    results in Rs. 78 more interest, then P = 2600. -/
theorem interest_rate_problem (P R : ℝ) (h : P * (R + 1) * 3 / 100 - P * R * 3 / 100 = 78) :
  P = 2600 := by
  sorry

end interest_rate_problem_l709_70953


namespace eighth_group_selection_l709_70956

/-- Systematic sampling from a population -/
def systematicSampling (totalPopulation : ℕ) (numGroups : ℕ) (firstGroupSelection : ℕ) (targetGroup : ℕ) : ℕ :=
  (targetGroup - 1) * (totalPopulation / numGroups) + firstGroupSelection

/-- Theorem: In a systematic sampling of 30 groups from 480 students, 
    if the selected number from the first group is 5, 
    then the selected number from the eighth group is 117. -/
theorem eighth_group_selection :
  systematicSampling 480 30 5 8 = 117 := by
  sorry

end eighth_group_selection_l709_70956


namespace unicorn_rope_problem_l709_70995

theorem unicorn_rope_problem (tower_radius : ℝ) (rope_length : ℝ) (rope_end_distance : ℝ)
  (a b c : ℕ) (h_radius : tower_radius = 10)
  (h_rope_length : rope_length = 25)
  (h_rope_end : rope_end_distance = 5)
  (h_c_prime : Nat.Prime c)
  (h_rope_touch : (a : ℝ) - Real.sqrt b = c * (rope_length - Real.sqrt ((tower_radius + rope_end_distance) ^ 2 + 5 ^ 2))) :
  a + b + c = 136 := by
sorry

end unicorn_rope_problem_l709_70995


namespace jane_ribbons_theorem_l709_70955

/-- The number of dresses Jane sews per day in the first week -/
def dresses_per_day_week1 : ℕ := 2

/-- The number of days Jane sews in the first week -/
def days_week1 : ℕ := 7

/-- The number of dresses Jane sews per day in the second period -/
def dresses_per_day_week2 : ℕ := 3

/-- The number of days Jane sews in the second period -/
def days_week2 : ℕ := 2

/-- The number of ribbons Jane adds to each dress -/
def ribbons_per_dress : ℕ := 2

/-- The total number of ribbons Jane uses -/
def total_ribbons : ℕ := 40

theorem jane_ribbons_theorem : 
  (dresses_per_day_week1 * days_week1 + dresses_per_day_week2 * days_week2) * ribbons_per_dress = total_ribbons := by
  sorry

end jane_ribbons_theorem_l709_70955


namespace average_reading_time_emery_serena_l709_70994

/-- The average reading time for two people, given one person's reading speed and time -/
def averageReadingTime (fasterReaderTime : ℕ) (speedRatio : ℕ) : ℚ :=
  (fasterReaderTime + fasterReaderTime * speedRatio) / 2

/-- Theorem: The average reading time for Emery and Serena is 60 days -/
theorem average_reading_time_emery_serena :
  averageReadingTime 20 5 = 60 := by
  sorry

end average_reading_time_emery_serena_l709_70994


namespace incorrect_statement_C_l709_70936

theorem incorrect_statement_C : ¬ (∀ (p q : Prop), (¬(p ∧ q)) → (¬p ∧ ¬q)) := by
  sorry

end incorrect_statement_C_l709_70936


namespace one_minus_repeating_third_eq_two_thirds_l709_70908

/-- The repeating decimal 0.333... -/
def repeating_third : ℚ := 1/3

/-- Theorem stating that 1 minus the repeating decimal 0.333... equals 2/3 -/
theorem one_minus_repeating_third_eq_two_thirds :
  1 - repeating_third = 2/3 := by sorry

end one_minus_repeating_third_eq_two_thirds_l709_70908


namespace problem_solution_l709_70949

theorem problem_solution (f : ℝ → ℝ) (m : ℝ) (a b c : ℝ) : 
  (∀ x, f x = m - |x - 2|) →
  ({x | f (x + 2) ≥ 0} = Set.Icc (-1) 1) →
  (1/a + 1/(2*b) + 1/(3*c) = m) →
  (m = 1 ∧ a + 2*b + 3*c ≥ 9) := by
  sorry

end problem_solution_l709_70949


namespace permutation_remainder_cardinality_l709_70979

theorem permutation_remainder_cardinality 
  (a : Fin 100 → Fin 100) 
  (h_perm : Function.Bijective a) :
  let b : Fin 100 → ℕ := fun i => (Finset.range i.succ).sum (fun j => (a j).val + 1)
  let r : Fin 100 → Fin 100 := fun i => (b i) % 100
  Finset.card (Finset.image r (Finset.univ : Finset (Fin 100))) ≥ 11 :=
by
  sorry

end permutation_remainder_cardinality_l709_70979


namespace erasers_bought_l709_70905

theorem erasers_bought (initial_erasers final_erasers : ℝ) (h1 : initial_erasers = 95.0) (h2 : final_erasers = 137) : 
  final_erasers - initial_erasers = 42 := by
  sorry

end erasers_bought_l709_70905


namespace no_real_roots_quadratic_l709_70931

theorem no_real_roots_quadratic (k : ℝ) : 
  (∀ x : ℝ, x^2 + 2*x + k ≠ 0) ↔ k > 1 := by
  sorry

end no_real_roots_quadratic_l709_70931


namespace soap_brand_ratio_l709_70966

theorem soap_brand_ratio (total : ℕ) (neither : ℕ) (only_a : ℕ) (both : ℕ) 
  (h1 : total = 300)
  (h2 : neither = 80)
  (h3 : only_a = 60)
  (h4 : both = 40) :
  (total - neither - only_a - both) / both = 3 := by
  sorry

end soap_brand_ratio_l709_70966


namespace total_molecular_weight_l709_70947

/-- Atomic weight of Aluminium in g/mol -/
def Al : ℝ := 26.98

/-- Atomic weight of Oxygen in g/mol -/
def O : ℝ := 16.00

/-- Atomic weight of Hydrogen in g/mol -/
def H : ℝ := 1.01

/-- Atomic weight of Sodium in g/mol -/
def Na : ℝ := 22.99

/-- Atomic weight of Chlorine in g/mol -/
def Cl : ℝ := 35.45

/-- Atomic weight of Calcium in g/mol -/
def Ca : ℝ := 40.08

/-- Atomic weight of Carbon in g/mol -/
def C : ℝ := 12.01

/-- Molecular weight of Aluminium hydroxide in g/mol -/
def Al_OH_3 : ℝ := Al + 3 * O + 3 * H

/-- Molecular weight of Sodium chloride in g/mol -/
def NaCl : ℝ := Na + Cl

/-- Molecular weight of Calcium carbonate in g/mol -/
def CaCO_3 : ℝ := Ca + C + 3 * O

/-- Total molecular weight of the given compounds in grams -/
def total_weight : ℝ := 4 * Al_OH_3 + 2 * NaCl + 3 * CaCO_3

theorem total_molecular_weight : total_weight = 729.19 := by
  sorry

end total_molecular_weight_l709_70947


namespace equation_solution_l709_70916

theorem equation_solution :
  ∀ x : ℚ, 7 * (4 * x + 3) - 3 = -3 * (2 - 5 * x) + 5 * x / 2 ↔ x = -16 / 7 :=
by sorry

end equation_solution_l709_70916


namespace inequality_proof_l709_70976

theorem inequality_proof (x : ℝ) : (2*x - 1)/3 + 1 ≤ 0 → x ≤ -1 := by
  sorry

end inequality_proof_l709_70976


namespace complex_number_properties_l709_70972

theorem complex_number_properties (z : ℂ) (h : z - 2*I = z*I + 4) : 
  Complex.abs z = Real.sqrt 10 ∧ ((z - 1) / 3) ^ 2023 = -I := by
  sorry

end complex_number_properties_l709_70972


namespace problem_statement_l709_70986

theorem problem_statement (a b : ℚ) (h1 : a = 1/2) (h2 : b = 1/3) : 
  (a - b) / (1/78) = 13 := by sorry

end problem_statement_l709_70986


namespace hexagon_minus_triangle_area_l709_70921

/-- The area of a hexagon with side length 2 and height 4, minus the area of an inscribed equilateral triangle with side length 4 -/
theorem hexagon_minus_triangle_area : 
  let hexagon_side : ℝ := 2
  let hexagon_height : ℝ := 4
  let triangle_side : ℝ := 4
  let hexagon_area : ℝ := 6 * (1/2 * hexagon_side * hexagon_height)
  let triangle_area : ℝ := (Real.sqrt 3 / 4) * triangle_side^2
  hexagon_area - triangle_area = 24 - 4 * Real.sqrt 3 :=
by sorry

end hexagon_minus_triangle_area_l709_70921


namespace fibonacci_pythagorean_hypotenuse_l709_70929

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

theorem fibonacci_pythagorean_hypotenuse (k : ℕ) (h : k ≥ 2) :
  fibonacci (2 * k + 1) = fibonacci k ^ 2 + fibonacci (k + 1) ^ 2 := by
  sorry

end fibonacci_pythagorean_hypotenuse_l709_70929


namespace absolute_value_equation_l709_70977

theorem absolute_value_equation (a : ℝ) : |a - 1| = 2 → a = 3 ∨ a = -1 := by
  sorry

end absolute_value_equation_l709_70977


namespace bathroom_kitchen_bulbs_l709_70978

theorem bathroom_kitchen_bulbs 
  (total_packs : ℕ) 
  (bulbs_per_pack : ℕ) 
  (bedroom_bulbs : ℕ) 
  (basement_bulbs : ℕ) 
  (h1 : total_packs = 6) 
  (h2 : bulbs_per_pack = 2) 
  (h3 : bedroom_bulbs = 2) 
  (h4 : basement_bulbs = 4) :
  total_packs * bulbs_per_pack - (bedroom_bulbs + basement_bulbs + basement_bulbs / 2) = 4 := by
  sorry

end bathroom_kitchen_bulbs_l709_70978
