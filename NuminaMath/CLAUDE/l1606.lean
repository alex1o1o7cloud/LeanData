import Mathlib

namespace backyard_fence_problem_l1606_160619

theorem backyard_fence_problem (back_length : ℝ) (fence_cost_per_foot : ℝ) 
  (owner_back_fraction : ℝ) (owner_left_fraction : ℝ) (owner_total_cost : ℝ) :
  back_length = 18 →
  fence_cost_per_foot = 3 →
  owner_back_fraction = 1/2 →
  owner_left_fraction = 2/3 →
  owner_total_cost = 72 →
  ∃ side_length : ℝ,
    side_length * fence_cost_per_foot * owner_left_fraction + 
    side_length * fence_cost_per_foot +
    back_length * fence_cost_per_foot * owner_back_fraction = owner_total_cost ∧
    side_length = 9 := by
  sorry

end backyard_fence_problem_l1606_160619


namespace lcm_48_180_l1606_160653

theorem lcm_48_180 : Nat.lcm 48 180 = 720 := by
  sorry

end lcm_48_180_l1606_160653


namespace area_of_S_l1606_160657

/-- A regular octagon in the complex plane -/
structure RegularOctagon where
  center : ℂ
  side_distance : ℝ
  parallel_to_real_axis : Prop

/-- The region outside the octagon -/
def R (octagon : RegularOctagon) : Set ℂ :=
  sorry

/-- The set S defined by the inversion of R -/
def S (octagon : RegularOctagon) : Set ℂ :=
  {w | ∃ z ∈ R octagon, w = 1 / z}

/-- The area of a set in the complex plane -/
noncomputable def area (s : Set ℂ) : ℝ :=
  sorry

theorem area_of_S (octagon : RegularOctagon) 
    (h1 : octagon.center = 0)
    (h2 : octagon.side_distance = 1.5)
    (h3 : octagon.parallel_to_real_axis) :
  area (S octagon) = 9 * Real.pi + 4.5 * Real.sqrt 2 * Real.pi :=
sorry

end area_of_S_l1606_160657


namespace least_possible_difference_l1606_160631

theorem least_possible_difference (x y z : ℤ) : 
  x < y → y < z → 
  y - x > 9 → 
  Even x → Odd y → Odd z → 
  (∀ w, w = z - x → w ≥ 13) ∧ ∃ w, w = z - x ∧ w = 13 :=
by sorry

end least_possible_difference_l1606_160631


namespace fourth_customer_new_item_probability_l1606_160675

/-- The number of menu items --/
def menu_items : ℕ := 5

/-- The number of customers --/
def customers : ℕ := 4

/-- The probability that the 4th customer orders a previously unordered item --/
def probability : ℚ := 32 / 125

theorem fourth_customer_new_item_probability :
  (menu_items ^ (customers - 1) * (menu_items - (customers - 1))) /
  (menu_items ^ customers) = probability := by
  sorry

end fourth_customer_new_item_probability_l1606_160675


namespace triangle_similarity_from_arithmetic_sides_l1606_160670

/-- Two triangles with sides in arithmetic progression and one equal angle are similar -/
theorem triangle_similarity_from_arithmetic_sides (a b c a₁ b₁ c₁ : ℝ) 
  (angleCAB angleCBA angleABC angleC₁A₁B₁ angleC₁B₁A₁ angleA₁B₁C₁ : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < a₁ ∧ 0 < b₁ ∧ 0 < c₁ →
  b - a = c - b →
  b₁ - a₁ = c₁ - b₁ →
  angleCAB + angleCBA + angleABC = π →
  angleC₁A₁B₁ + angleC₁B₁A₁ + angleA₁B₁C₁ = π →
  angleCAB = angleC₁A₁B₁ →
  ∃ (k : ℝ), k > 0 ∧ a = k * a₁ ∧ b = k * b₁ ∧ c = k * c₁ :=
by sorry

end triangle_similarity_from_arithmetic_sides_l1606_160670


namespace acceleration_at_two_seconds_l1606_160681

-- Define the distance function
def s (t : ℝ) : ℝ := 2 * t^3 - 5 * t^2 + 2

-- Define the velocity function as the derivative of the distance function
def v (t : ℝ) : ℝ := 6 * t^2 - 10 * t

-- Define the acceleration function as the derivative of the velocity function
def a (t : ℝ) : ℝ := 12 * t - 10

-- Theorem: The acceleration at t = 2 seconds is 14 m/s²
theorem acceleration_at_two_seconds : a 2 = 14 := by sorry

end acceleration_at_two_seconds_l1606_160681


namespace division_problem_l1606_160615

theorem division_problem : ∃ (d r : ℕ), d > 0 ∧ 1270 = 74 * d + r ∧ r < d := by
  -- The proof goes here
  sorry

end division_problem_l1606_160615


namespace difference_in_tickets_l1606_160624

def tickets_for_toys : ℕ := 31
def tickets_for_clothes : ℕ := 14

theorem difference_in_tickets : tickets_for_toys - tickets_for_clothes = 17 := by
  sorry

end difference_in_tickets_l1606_160624


namespace mrs_hilt_friends_l1606_160641

/-- The number of friends Mrs. Hilt met who were carrying pears -/
def friends_with_pears : ℕ := 9

/-- The number of friends Mrs. Hilt met who were carrying oranges -/
def friends_with_oranges : ℕ := 6

/-- The total number of friends Mrs. Hilt met -/
def total_friends : ℕ := friends_with_pears + friends_with_oranges

theorem mrs_hilt_friends :
  total_friends = 15 :=
by sorry

end mrs_hilt_friends_l1606_160641


namespace max_product_distances_area_triangle_45_slope_l1606_160643

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse (a b : ℝ) where
  h₁ : a > 0
  h₂ : b > 0
  h₃ : a > b

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Define the foci of an ellipse -/
def foci (e : Ellipse a b) : Point × Point :=
  sorry

/-- Define the endpoints of the minor axis of an ellipse -/
def minorAxisEndpoints (e : Ellipse a b) : Point × Point :=
  sorry

/-- Calculate the perimeter of a quadrilateral given its vertices -/
def perimeter (p₁ p₂ p₃ p₄ : Point) : ℝ :=
  sorry

/-- Calculate the distance between two points -/
def distance (p₁ p₂ : Point) : ℝ :=
  sorry

/-- Calculate the area of a triangle given its vertices -/
def triangleArea (p₁ p₂ p₃ : Point) : ℝ :=
  sorry

/-- Theorem about the maximum product of distances from foci to points on the ellipse -/
theorem max_product_distances (e : Ellipse a b) (F₁ F₂ A B : Point) :
  let (F₁', F₂') := foci e
  let (M, N) := minorAxisEndpoints e
  perimeter F₁ F₂ M N = 4 →
  F₁ = F₁' →
  distance A B = 4/3 →
  (∃ (l : Point → Prop), l F₁ ∧ l A ∧ l B) →
  (∀ A' B' : Point, (∃ (l : Point → Prop), l F₁ ∧ l A' ∧ l B') →
    distance A' F₂ * distance B' F₂ ≤ 16/9) :=
  sorry

/-- Theorem about the area of the triangle when the line has a 45-degree slope -/
theorem area_triangle_45_slope (e : Ellipse a b) (F₁ F₂ A B : Point) :
  let (F₁', F₂') := foci e
  let (M, N) := minorAxisEndpoints e
  perimeter F₁ F₂ M N = 4 →
  F₁ = F₁' →
  distance A B = 4/3 →
  (∃ (l : Point → Prop), l F₁ ∧ l A ∧ l B ∧ ∀ p q : Point, l p ∧ l q → (p.y - q.y) = (p.x - q.x)) →
  triangleArea A B F₂ = 2/3 :=
  sorry

end max_product_distances_area_triangle_45_slope_l1606_160643


namespace transport_speed_problem_l1606_160617

/-- Proves that given two transports traveling in opposite directions, with one traveling at 60 mph,
    if they are 348 miles apart after 2.71875 hours, then the speed of the second transport is 68 mph. -/
theorem transport_speed_problem (speed_a speed_b : ℝ) (time : ℝ) (distance : ℝ) : 
  speed_a = 60 →
  time = 2.71875 →
  distance = 348 →
  (speed_a + speed_b) * time = distance →
  speed_b = 68 := by
  sorry

#check transport_speed_problem

end transport_speed_problem_l1606_160617


namespace croissant_distribution_l1606_160614

theorem croissant_distribution (total : Nat) (neighbors : Nat) (h1 : total = 59) (h2 : neighbors = 8) :
  total - (neighbors * (total / neighbors)) = 3 := by
  sorry

end croissant_distribution_l1606_160614


namespace square_circle_area_ratio_l1606_160698

theorem square_circle_area_ratio (s r : ℝ) (h : s > 0) (k : r > 0) (eq : 4 * s = 4 * Real.pi * r) :
  s^2 / (Real.pi * r^2) = Real.pi := by
  sorry

end square_circle_area_ratio_l1606_160698


namespace expression_evaluation_l1606_160632

theorem expression_evaluation : (2023 - 1910 + 5)^2 / 121 = 114 + 70 / 121 := by
  sorry

end expression_evaluation_l1606_160632


namespace number_problem_l1606_160660

theorem number_problem (n : ℝ) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * n = 15 → (40/100 : ℝ) * n = 180 :=
by
  sorry

end number_problem_l1606_160660


namespace max_value_theorem_l1606_160677

theorem max_value_theorem (x y k : ℝ) (hx : x > 0) (hy : y > 0) (hk : k > 0) :
  (∀ a b : ℝ, a > 0 → b > 0 → (k * a + b)^2 / (a^2 + b^2) ≤ (k * x + y)^2 / (x^2 + y^2)) →
  (k * x + y)^2 / (x^2 + y^2) = 2 :=
sorry

end max_value_theorem_l1606_160677


namespace not_all_electric_implies_some_not_electric_l1606_160693

-- Define the set of all cars in the parking lot
variable (Car : Type)
variable (parking_lot : Set Car)

-- Define a predicate for electric cars
variable (is_electric : Car → Prop)

-- Define the theorem
theorem not_all_electric_implies_some_not_electric
  (h : ¬ ∀ (c : Car), c ∈ parking_lot → is_electric c) :
  ∃ (c : Car), c ∈ parking_lot ∧ ¬ is_electric c :=
by
  sorry

end not_all_electric_implies_some_not_electric_l1606_160693


namespace solution_replacement_fraction_l1606_160680

theorem solution_replacement_fraction 
  (initial_concentration : Real)
  (replacement_concentration : Real)
  (final_concentration : Real)
  (h1 : initial_concentration = 0.40)
  (h2 : replacement_concentration = 0.25)
  (h3 : final_concentration = 0.35)
  : ∃ x : Real, x = 1/3 ∧ 
    final_concentration * 1 = 
    (initial_concentration * (1 - x)) + (replacement_concentration * x) := by
  sorry

end solution_replacement_fraction_l1606_160680


namespace cost_of_five_cds_l1606_160673

/-- The cost of a certain number of identical CDs -/
def cost_of_cds (n : ℕ) : ℚ :=
  28 * (n / 2 : ℚ)

/-- Theorem stating that the cost of five CDs is 70 dollars -/
theorem cost_of_five_cds : cost_of_cds 5 = 70 := by
  sorry

end cost_of_five_cds_l1606_160673


namespace smallest_non_factor_product_l1606_160692

theorem smallest_non_factor_product (a b : ℕ+) : 
  a ≠ b →
  a ∣ 48 →
  b ∣ 48 →
  ¬(a * b ∣ 48) →
  (∀ (c d : ℕ+), c ≠ d → c ∣ 48 → d ∣ 48 → ¬(c * d ∣ 48) → a * b ≤ c * d) →
  a * b = 18 := by
sorry

end smallest_non_factor_product_l1606_160692


namespace key_lime_requirement_l1606_160696

/-- The number of tablespoons in one cup -/
def tablespoons_per_cup : ℕ := 16

/-- The original amount of key lime juice in cups -/
def original_juice_cups : ℚ := 1/4

/-- The multiplication factor for the juice amount -/
def juice_multiplier : ℕ := 3

/-- The minimum amount of juice (in tablespoons) that a key lime can yield -/
def min_juice_per_lime : ℕ := 1

/-- The maximum amount of juice (in tablespoons) that a key lime can yield -/
def max_juice_per_lime : ℕ := 2

/-- The number of key limes needed to ensure enough juice for the recipe -/
def key_limes_needed : ℕ := 12

theorem key_lime_requirement :
  key_limes_needed * min_juice_per_lime ≥ 
  juice_multiplier * (original_juice_cups * tablespoons_per_cup) ∧
  key_limes_needed * max_juice_per_lime ≥
  juice_multiplier * (original_juice_cups * tablespoons_per_cup) ∧
  ∀ n : ℕ, n < key_limes_needed →
    n * min_juice_per_lime < juice_multiplier * (original_juice_cups * tablespoons_per_cup) :=
by sorry

end key_lime_requirement_l1606_160696


namespace regular_octagon_interior_angle_l1606_160667

/-- The number of sides in an octagon -/
def octagon_sides : ℕ := 8

/-- The measure of each interior angle in a regular octagon -/
def octagon_interior_angle : ℝ := 135

/-- Theorem: Each interior angle of a regular octagon measures 135 degrees -/
theorem regular_octagon_interior_angle :
  (180 * (octagon_sides - 2 : ℝ)) / octagon_sides = octagon_interior_angle :=
sorry

end regular_octagon_interior_angle_l1606_160667


namespace kim_money_l1606_160672

/-- Given that Kim has 40% more money than Sal, Sal has 20% less money than Phil,
    and Sal and Phil have a combined total of $1.80, prove that Kim has $1.12. -/
theorem kim_money (sal phil kim : ℝ) 
  (h1 : kim = sal * 1.4)
  (h2 : sal = phil * 0.8)
  (h3 : sal + phil = 1.8) : 
  kim = 1.12 := by
sorry

end kim_money_l1606_160672


namespace cistern_filling_time_l1606_160601

-- Define the filling rates of pipes p and q
def fill_rate_p : ℚ := 1 / 10
def fill_rate_q : ℚ := 1 / 15

-- Define the time both pipes are open together
def initial_time : ℚ := 4

-- Define the total capacity of the cistern
def total_capacity : ℚ := 1

-- Theorem statement
theorem cistern_filling_time :
  let filled_initially := (fill_rate_p + fill_rate_q) * initial_time
  let remaining_to_fill := total_capacity - filled_initially
  let remaining_time := remaining_to_fill / fill_rate_q
  remaining_time = 5 := by sorry

end cistern_filling_time_l1606_160601


namespace contrapositive_equivalence_l1606_160652

theorem contrapositive_equivalence (x : ℝ) : 
  (x^2 = 1 → x = 1 ∨ x = -1) ↔ (x ≠ 1 ∧ x ≠ -1 → x^2 ≠ 1) :=
sorry

end contrapositive_equivalence_l1606_160652


namespace pet_ownership_percentage_l1606_160682

theorem pet_ownership_percentage (total_students : ℕ) (cat_owners : ℕ) (dog_owners : ℕ) (both_owners : ℕ)
  (h1 : total_students = 500)
  (h2 : cat_owners = 150)
  (h3 : dog_owners = 100)
  (h4 : both_owners = 40) :
  (cat_owners + dog_owners - both_owners) / total_students = 42 / 100 := by
  sorry

end pet_ownership_percentage_l1606_160682


namespace three_right_angles_implies_rectangle_l1606_160637

/-- A quadrilateral is a polygon with four sides and four vertices. -/
structure Quadrilateral where
  vertices : Fin 4 → ℝ × ℝ

/-- An angle is right if it measures 90 degrees or π/2 radians. -/
def is_right_angle (a b c : ℝ × ℝ) : Prop := sorry

/-- A quadrilateral is a rectangle if all its angles are right angles. -/
def is_rectangle (q : Quadrilateral) : Prop :=
  ∀ i : Fin 4, is_right_angle (q.vertices i) (q.vertices (i + 1)) (q.vertices (i + 2))

/-- If a quadrilateral has three right angles, then it is a rectangle. -/
theorem three_right_angles_implies_rectangle (q : Quadrilateral) :
  (∃ i j k : Fin 4, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    is_right_angle (q.vertices i) (q.vertices (i + 1)) (q.vertices (i + 2)) ∧
    is_right_angle (q.vertices j) (q.vertices (j + 1)) (q.vertices (j + 2)) ∧
    is_right_angle (q.vertices k) (q.vertices (k + 1)) (q.vertices (k + 2)))
  → is_rectangle q :=
sorry

end three_right_angles_implies_rectangle_l1606_160637


namespace point_movement_power_l1606_160656

/-- 
Given a point (-1, 1) in the Cartesian coordinate system,
if it is moved up 1 unit and then left 2 units to reach a point (x, y),
then x^y = 9.
-/
theorem point_movement_power (x y : ℝ) : 
  ((-1 : ℝ) + -2 = x) → ((1 : ℝ) + 1 = y) → x^y = 9 := by
  sorry

end point_movement_power_l1606_160656


namespace partition_into_three_exists_partition_into_four_not_exists_l1606_160685

-- Define a partition of positive integers into three sets
def PartitionIntoThree : (ℕ → Fin 3) → Prop :=
  λ f => ∀ n, n > 0 → ∃ i, f n = i

-- Define a partition of positive integers into four sets
def PartitionIntoFour : (ℕ → Fin 4) → Prop :=
  λ f => ∀ n, n > 0 → ∃ i, f n = i

-- Statement 1
theorem partition_into_three_exists :
  ∃ f : ℕ → Fin 3, PartitionIntoThree f ∧
    ∀ n ≥ 15, ∀ i : Fin 3,
      ∃ a b : ℕ, a ≠ b ∧ f a = i ∧ f b = i ∧ a + b = n :=
sorry

-- Statement 2
theorem partition_into_four_not_exists :
  ∀ f : ℕ → Fin 4, PartitionIntoFour f →
    ∃ n ≥ 15, ∃ i : Fin 4,
      ∀ a b : ℕ, a ≠ b → f a = i → f b = i → a + b ≠ n :=
sorry

end partition_into_three_exists_partition_into_four_not_exists_l1606_160685


namespace units_digit_of_product_units_digit_of_27_times_68_l1606_160626

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The product of two natural numbers has the same units digit as the product of their units digits -/
theorem units_digit_of_product (a b : ℕ) :
  unitsDigit (a * b) = unitsDigit (unitsDigit a * unitsDigit b) := by sorry

theorem units_digit_of_27_times_68 :
  unitsDigit (27 * 68) = 6 := by sorry

end units_digit_of_product_units_digit_of_27_times_68_l1606_160626


namespace parabola_directrix_parameter_l1606_160659

/-- 
For a parabola y = ax^2 with directrix y = 1, the value of a is -1/4.
-/
theorem parabola_directrix_parameter (a : ℝ) : 
  (∀ x y : ℝ, y = a * x^2) →  -- Condition 1: Parabola equation
  (∃ y : ℝ, y = 1 ∧ ∀ x : ℝ, y = 1 → (x, y) ∉ {(x, y) | y = a * x^2}) →  -- Condition 2: Directrix equation
  a = -1/4 := by
sorry

end parabola_directrix_parameter_l1606_160659


namespace woogle_threshold_l1606_160606

/-- The score for dropping n woogles -/
def drop_score (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The score for eating n woogles -/
def eat_score (n : ℕ) : ℕ := 15 * n

/-- 30 is the smallest positive integer for which dropping woogles scores more than eating them -/
theorem woogle_threshold : ∀ k : ℕ, k < 30 → drop_score k ≤ eat_score k ∧ drop_score 30 > eat_score 30 := by
  sorry

end woogle_threshold_l1606_160606


namespace parabola_ratio_l1606_160634

/-- Given a parabola y = ax² + bx + c passing through points (-1, 1) and (3, 1),
    prove that a/b = -2 -/
theorem parabola_ratio (a b c : ℝ) : 
  (∀ x, a * x^2 + b * x + c = 1 → x = -1 ∨ x = 3) →
  a / b = -2 := by
  sorry

end parabola_ratio_l1606_160634


namespace algebraic_expression_value_l1606_160622

theorem algebraic_expression_value (a b : ℝ) (h : a - 2*b = -1) :
  1 - 2*a + 4*b = 3 := by sorry

end algebraic_expression_value_l1606_160622


namespace smallest_five_digit_divisible_by_3_and_4_l1606_160644

theorem smallest_five_digit_divisible_by_3_and_4 : ∃ n : ℕ, 
  (n ≥ 10000 ∧ n < 100000) ∧ 
  n % 3 = 0 ∧ 
  n % 4 = 0 ∧
  (∀ m : ℕ, m ≥ 10000 ∧ m < 100000 ∧ m % 3 = 0 ∧ m % 4 = 0 → m ≥ n) ∧
  n = 10008 := by
sorry

end smallest_five_digit_divisible_by_3_and_4_l1606_160644


namespace intersection_equality_l1606_160636

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.log (1 - x) ∧ 1 - x > 0}
def B (m : ℝ) : Set ℝ := {y | ∃ x, y = -x^2 + 2*x + m}

-- State the theorem
theorem intersection_equality (m : ℝ) : A ∩ B m = A ↔ m ≥ 0 := by sorry

end intersection_equality_l1606_160636


namespace quadratic_root_ratio_l1606_160620

theorem quadratic_root_ratio (k : ℝ) : 
  (∃ x y : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ x / y = 3 ∧ 
   x^2 + 10*x + k = 0 ∧ y^2 + 10*y + k = 0) →
  k = 18.75 := by
sorry

end quadratic_root_ratio_l1606_160620


namespace range_of_f_domain1_range_of_f_domain2_l1606_160607

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 2*x + 9

-- Define the domains
def domain1 : Set ℝ := {x | 3 < x ∧ x ≤ 8}
def domain2 : Set ℝ := {x | -3 < x ∧ x ≤ 2}

-- State the theorems
theorem range_of_f_domain1 :
  f '' domain1 = Set.Ioc 12 57 := by sorry

theorem range_of_f_domain2 :
  f '' domain2 = Set.Ico 8 24 := by sorry

end range_of_f_domain1_range_of_f_domain2_l1606_160607


namespace marble_247_is_white_l1606_160662

/-- Represents the color of a marble -/
inductive MarbleColor
| Gray
| White
| Black

/-- Returns the color of the nth marble in the sequence -/
def marbleColor (n : ℕ) : MarbleColor :=
  match n % 12 with
  | 0 | 1 | 2 | 3 => MarbleColor.Gray
  | 4 | 5 | 6 | 7 | 8 => MarbleColor.White
  | _ => MarbleColor.Black

/-- Theorem stating that the 247th marble is white -/
theorem marble_247_is_white : marbleColor 247 = MarbleColor.White := by
  sorry


end marble_247_is_white_l1606_160662


namespace seed_cost_calculation_l1606_160602

def seed_cost_2lb : ℝ := 44.68
def seed_amount : ℝ := 6

theorem seed_cost_calculation : 
  seed_amount * (seed_cost_2lb / 2) = 134.04 := by
  sorry

end seed_cost_calculation_l1606_160602


namespace zeros_not_adjacent_probability_l1606_160609

/-- The number of ones in the arrangement -/
def num_ones : ℕ := 4

/-- The number of zeros in the arrangement -/
def num_zeros : ℕ := 2

/-- The total number of spaces where zeros can be placed -/
def total_spaces : ℕ := num_ones + 1

/-- The probability that two zeros are not adjacent when randomly arranged with four ones -/
theorem zeros_not_adjacent_probability : 
  (Nat.choose total_spaces num_zeros : ℚ) / 
  (Nat.choose total_spaces 1 + Nat.choose total_spaces num_zeros : ℚ) = 2/3 := by
  sorry

end zeros_not_adjacent_probability_l1606_160609


namespace tailor_cuts_difference_l1606_160665

theorem tailor_cuts_difference : 
  (7/8 + 11/12) - (5/6 + 3/4) = 5/24 := by
  sorry

end tailor_cuts_difference_l1606_160665


namespace max_value_on_sphere_l1606_160650

theorem max_value_on_sphere (x y z : ℝ) (h : x^2 + y^2 + 4*z^2 = 1) :
  ∃ (max : ℝ), max = Real.sqrt 6 ∧ ∀ (a b c : ℝ), a^2 + b^2 + 4*c^2 = 1 → a + b + 4*c ≤ max :=
by sorry

end max_value_on_sphere_l1606_160650


namespace fair_coin_same_side_four_times_l1606_160610

theorem fair_coin_same_side_four_times (p : ℝ) :
  (p = 1 / 2) →                        -- The coin is fair (equal probability for each side)
  (p ^ 4 : ℝ) = 1 / 16 := by            -- Probability of same side 4 times is 1/16
sorry


end fair_coin_same_side_four_times_l1606_160610


namespace only_valid_rectangles_l1606_160697

/-- A rectangle that can be divided into 13 equal squares -/
structure Rectangle13Squares where
  width : ℕ
  height : ℕ
  is_valid : width * height = 13

/-- The set of all valid rectangles that can be divided into 13 equal squares -/
def valid_rectangles : Set Rectangle13Squares :=
  {r : Rectangle13Squares | r.width = 1 ∧ r.height = 13 ∨ r.width = 13 ∧ r.height = 1}

/-- Theorem stating that the only valid rectangles are 1x13 or 13x1 -/
theorem only_valid_rectangles :
  ∀ r : Rectangle13Squares, r ∈ valid_rectangles :=
by
  sorry

end only_valid_rectangles_l1606_160697


namespace excluded_students_average_mark_l1606_160629

/-- Proves that given a class of 35 students with an average mark of 80,
    if 5 students are excluded and the remaining students have an average mark of 90,
    then the average mark of the excluded students is 20. -/
theorem excluded_students_average_mark
  (total_students : ℕ)
  (class_average : ℚ)
  (remaining_students : ℕ)
  (remaining_average : ℚ)
  (h1 : total_students = 35)
  (h2 : class_average = 80)
  (h3 : remaining_students = 30)
  (h4 : remaining_average = 90) :
  let excluded_students := total_students - remaining_students
  let excluded_average := (total_students * class_average - remaining_students * remaining_average) / excluded_students
  excluded_average = 20 := by
  sorry

#check excluded_students_average_mark

end excluded_students_average_mark_l1606_160629


namespace product_zero_from_sum_and_cube_sum_l1606_160651

theorem product_zero_from_sum_and_cube_sum (a b : ℝ) 
  (h1 : a + b = 5) 
  (h2 : a^3 + b^3 = 125) : 
  a * b = 0 := by
  sorry

end product_zero_from_sum_and_cube_sum_l1606_160651


namespace complement_union_theorem_l1606_160642

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 2}
def B : Set Nat := {2, 3, 4}

theorem complement_union_theorem : 
  (U \ A) ∪ B = {2, 3, 4, 5} := by sorry

end complement_union_theorem_l1606_160642


namespace percentage_of_democrat_voters_prove_percentage_of_democrat_voters_l1606_160645

theorem percentage_of_democrat_voters : ℝ → ℝ → Prop :=
  fun d r =>
    d + r = 100 →
    0.7 * d + 0.2 * r = 50 →
    d = 60

-- Proof
theorem prove_percentage_of_democrat_voters :
  ∃ d r : ℝ, percentage_of_democrat_voters d r :=
by
  sorry

end percentage_of_democrat_voters_prove_percentage_of_democrat_voters_l1606_160645


namespace total_value_calculation_l1606_160658

/-- Calculates the total value of coins and paper money with a certificate bonus --/
def totalValue (goldWorth silverWorth bronzeWorth titaniumWorth : ℝ)
                (banknoteWorth couponWorth voucherWorth : ℝ)
                (goldCount silverCount bronzeCount titaniumCount : ℕ)
                (banknoteCount couponCount voucherCount : ℕ)
                (certificateBonus : ℝ) : ℝ :=
  let goldValue := goldWorth * goldCount
  let silverValue := silverWorth * silverCount
  let bronzeValue := bronzeWorth * bronzeCount
  let titaniumValue := titaniumWorth * titaniumCount
  let banknoteValue := banknoteWorth * banknoteCount
  let couponValue := couponWorth * couponCount
  let voucherValue := voucherWorth * voucherCount
  let baseTotal := goldValue + silverValue + bronzeValue + titaniumValue +
                   banknoteValue + couponValue + voucherValue
  let bonusAmount := certificateBonus * (goldValue + silverValue)
  baseTotal + bonusAmount

theorem total_value_calculation :
  totalValue 80 45 25 10 50 10 20 7 9 12 5 3 6 4 0.05 = 1653.25 := by
  sorry

end total_value_calculation_l1606_160658


namespace age_ratio_problem_l1606_160690

theorem age_ratio_problem (ann_age : ℕ) (x : ℚ) : 
  ann_age = 6 →
  (ann_age + 10) + (x * ann_age + 10) = 38 →
  x * ann_age / ann_age = 2 := by
  sorry

end age_ratio_problem_l1606_160690


namespace cube_inequality_l1606_160666

theorem cube_inequality (a b : ℝ) (h : a > b) : a^3 > b^3 := by
  sorry

end cube_inequality_l1606_160666


namespace states_fraction_proof_l1606_160648

theorem states_fraction_proof (total_states : ℕ) (decade_states : ℕ) :
  total_states = 22 →
  decade_states = 8 →
  (decade_states : ℚ) / total_states = 4 / 11 := by
sorry

end states_fraction_proof_l1606_160648


namespace binomial_coefficient_problem_l1606_160600

theorem binomial_coefficient_problem (a : ℝ) : 
  (Finset.range 11).sum (λ k => Nat.choose 10 k * a^(10 - k) * (if k = 3 then 1 else 0)) = 15 → 
  a = 1/2 := by
sorry

end binomial_coefficient_problem_l1606_160600


namespace three_numbers_sum_l1606_160638

theorem three_numbers_sum (a b c : ℝ) 
  (sum_ab : a + b = 37)
  (sum_bc : b + c = 58)
  (sum_ca : c + a = 72) :
  a + b + c - 10 = 73.5 := by sorry

end three_numbers_sum_l1606_160638


namespace geometric_sequence_proof_l1606_160612

theorem geometric_sequence_proof :
  let a : ℚ := 3
  let r : ℚ := 8 / 27
  let sequence : ℕ → ℚ := λ n => a * r ^ (n - 1)
  (sequence 1 = 3) ∧ 
  (sequence 2 = 8 / 9) ∧ 
  (sequence 3 = 32 / 81) :=
by
  sorry

#check geometric_sequence_proof

end geometric_sequence_proof_l1606_160612


namespace taobao_villages_growth_l1606_160603

/-- 
Given an arithmetic sequence with first term 1311 and common difference 1000,
prove that the 8th term of this sequence is 8311.
-/
theorem taobao_villages_growth (a₁ : ℕ) (d : ℕ) (n : ℕ) :
  a₁ = 1311 → d = 1000 → n = 8 →
  a₁ + (n - 1) * d = 8311 :=
by sorry

end taobao_villages_growth_l1606_160603


namespace pigeon_hole_theorem_l1606_160668

/-- The number of pigeons -/
def num_pigeons : ℕ := 160

/-- The function that determines which hole a pigeon flies to -/
def pigeon_hole (i n : ℕ) : ℕ := i^2 % n

/-- Predicate to check if all pigeons fly to unique holes -/
def all_unique_holes (n : ℕ) : Prop :=
  ∀ i j, i ≤ num_pigeons → j ≤ num_pigeons → i ≠ j → pigeon_hole i n ≠ pigeon_hole j n

/-- The minimum number of holes needed -/
def min_holes : ℕ := 326

theorem pigeon_hole_theorem :
  (∀ k, k < min_holes → ¬(all_unique_holes k)) ∧ all_unique_holes min_holes :=
by sorry

end pigeon_hole_theorem_l1606_160668


namespace least_number_divisible_by_five_primes_l1606_160604

theorem least_number_divisible_by_five_primes : ℕ := by
  -- Define the property of being divisible by five different primes
  let divisible_by_five_primes (n : ℕ) :=
    ∃ (p₁ p₂ p₃ p₄ p₅ : ℕ),
      Nat.Prime p₁ ∧ Nat.Prime p₂ ∧ Nat.Prime p₃ ∧ Nat.Prime p₄ ∧ Nat.Prime p₅ ∧
      p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₁ ≠ p₅ ∧
      p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₂ ≠ p₅ ∧
      p₃ ≠ p₄ ∧ p₃ ≠ p₅ ∧
      p₄ ≠ p₅ ∧
      n % p₁ = 0 ∧ n % p₂ = 0 ∧ n % p₃ = 0 ∧ n % p₄ = 0 ∧ n % p₅ = 0

  -- State that 2310 is divisible by five different primes
  have h1 : divisible_by_five_primes 2310 := by sorry

  -- State that 2310 is the least such number
  have h2 : ∀ m : ℕ, m < 2310 → ¬(divisible_by_five_primes m) := by sorry

  -- Conclude that 2310 is the answer
  exact 2310

end least_number_divisible_by_five_primes_l1606_160604


namespace solve_equation_l1606_160627

theorem solve_equation (a b c : ℤ) (x : ℝ) (h : 5 / (a^2 + b * Real.log x) = c) :
  x = 10 ^ ((5 / c - a^2) / b) :=
by sorry

end solve_equation_l1606_160627


namespace sarah_skateboard_speed_l1606_160613

/-- Given the following conditions:
1. Pete walks backwards three times faster than Susan walks forwards.
2. Tracy does one-handed cartwheels twice as fast as Susan walks forwards.
3. Mike swims eight times faster than Tracy does cartwheels.
4. Pete can walk on his hands at only one quarter of the speed that Tracy can do cartwheels.
5. Pete can ride his bike five times faster than Mike swims.
6. Pete walks on his hands at 2 miles per hour.
7. Patty can row three times faster than Pete walks backwards.
8. Sarah can skateboard six times faster than Patty rows.

Prove that Sarah can skateboard at 216 miles per hour. -/
theorem sarah_skateboard_speed :
  ∀ (pete_backward_speed pete_hand_speed pete_bike_speed susan_speed tracy_speed
     mike_speed patty_speed sarah_speed : ℝ),
  pete_backward_speed = 3 * susan_speed →
  tracy_speed = 2 * susan_speed →
  mike_speed = 8 * tracy_speed →
  pete_hand_speed = 1/4 * tracy_speed →
  pete_bike_speed = 5 * mike_speed →
  pete_hand_speed = 2 →
  patty_speed = 3 * pete_backward_speed →
  sarah_speed = 6 * patty_speed →
  sarah_speed = 216 := by
  sorry

end sarah_skateboard_speed_l1606_160613


namespace opposite_of_one_half_l1606_160684

theorem opposite_of_one_half : 
  (1 / 2 : ℚ) + (-1 / 2 : ℚ) = 0 := by sorry

end opposite_of_one_half_l1606_160684


namespace sum_abc_is_zero_l1606_160611

theorem sum_abc_is_zero (a b c : ℝ) 
  (h1 : (a + b) / c = (b + c) / a) 
  (h2 : (b + c) / a = (a + c) / b) 
  (h3 : b ≠ c) : 
  a + b + c = 0 := by
sorry

end sum_abc_is_zero_l1606_160611


namespace x_value_proof_l1606_160671

theorem x_value_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x^4 / y = 2) (h2 : y^3 / z = 6) (h3 : z^2 / x = 8) :
  x = (18432 : ℝ)^(1/23) := by
sorry

end x_value_proof_l1606_160671


namespace find_k_l1606_160678

theorem find_k (k : ℝ) (h1 : k ≠ 0) : 
  (∀ x : ℝ, (x^2 - k) * (x + k) = x^3 + k * (x^2 - x - 2)) → k = 2 := by
  sorry

end find_k_l1606_160678


namespace geometric_number_difference_l1606_160664

/-- A geometric number is a 3-digit number with distinct digits forming a geometric sequence,
    and the middle digit is odd. -/
def IsGeometricNumber (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧
  ∃ (a b c : ℕ),
    n = 100 * a + 10 * b + c ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    Odd b ∧
    b * b = a * c

theorem geometric_number_difference :
  ∃ (min max : ℕ),
    IsGeometricNumber min ∧
    IsGeometricNumber max ∧
    (∀ n, IsGeometricNumber n → min ≤ n ∧ n ≤ max) ∧
    max - min = 220 := by
  sorry

end geometric_number_difference_l1606_160664


namespace matrix_product_is_zero_l1606_160623

open Matrix

/-- Given two 3x3 matrices A and B, prove that their product is the zero matrix --/
theorem matrix_product_is_zero (a b c : ℝ) :
  let A : Matrix (Fin 3) (Fin 3) ℝ := !![0, 2*c, -2*b; -2*c, 0, 2*a; 2*b, -2*a, 0]
  let B : Matrix (Fin 3) (Fin 3) ℝ := !![2*a^2, a^2+b^2, a^2+c^2; a^2+b^2, 2*b^2, b^2+c^2; a^2+c^2, b^2+c^2, 2*c^2]
  A * B = 0 := by
  sorry

#check matrix_product_is_zero

end matrix_product_is_zero_l1606_160623


namespace max_digits_product_5_and_3_l1606_160649

theorem max_digits_product_5_and_3 : 
  ∀ a b : ℕ, 
  10000 ≤ a ∧ a ≤ 99999 → 
  100 ≤ b ∧ b ≤ 999 → 
  a * b < 100000000 := by
sorry

end max_digits_product_5_and_3_l1606_160649


namespace no_solution_arcsin_arccos_squared_l1606_160691

theorem no_solution_arcsin_arccos_squared (x : ℝ) : 
  (Real.arcsin x + Real.arccos x = π / 2) → (Real.arcsin x)^2 + (Real.arccos x)^2 ≠ 1 := by
  sorry

end no_solution_arcsin_arccos_squared_l1606_160691


namespace x_to_twenty_l1606_160688

theorem x_to_twenty (x : ℝ) (h : x + 1/x = Real.sqrt 5) : x^20 = 16163 := by
  sorry

end x_to_twenty_l1606_160688


namespace intersection_of_P_and_Q_l1606_160661

def P : Set ℝ := {1, 2}
def Q : Set ℝ := {x | |x| < 2}

theorem intersection_of_P_and_Q : P ∩ Q = {1} := by sorry

end intersection_of_P_and_Q_l1606_160661


namespace QR_distance_l1606_160618

-- Define the right triangle DEF
structure RightTriangle where
  DE : ℝ
  EF : ℝ
  DF : ℝ
  is_right_triangle : DE^2 + EF^2 = DF^2

-- Define the circles
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the problem setup
def problem_setup (t : RightTriangle) (Q R : Circle) : Prop :=
  t.DE = 9 ∧ t.EF = 12 ∧ t.DF = 15 ∧
  Q.center.2 = t.EF ∧ 
  R.center.1 = 0 ∧
  Q.radius = t.DE ∧
  R.radius = t.EF

-- Theorem statement
theorem QR_distance (t : RightTriangle) (Q R : Circle) 
  (h : problem_setup t Q R) : 
  Real.sqrt ((Q.center.1 - R.center.1)^2 + (Q.center.2 - R.center.2)^2) = 15 :=
sorry

end QR_distance_l1606_160618


namespace complex_fraction_real_l1606_160686

theorem complex_fraction_real (t : ℝ) : 
  (Complex.I * (2 * t + Complex.I) / (1 - 2 * Complex.I)).im = 0 → t = -1/4 := by
  sorry

end complex_fraction_real_l1606_160686


namespace rectangle_perimeter_is_46_l1606_160663

/-- A rectangle dissection puzzle with seven squares -/
structure RectangleDissection where
  b₁ : ℕ
  b₂ : ℕ
  b₃ : ℕ
  b₄ : ℕ
  b₅ : ℕ
  b₆ : ℕ
  b₇ : ℕ
  rel₁ : b₁ + b₂ = b₃
  rel₂ : b₁ + b₃ = b₄
  rel₃ : b₃ + b₄ = b₅
  rel₄ : b₄ + b₅ = b₆
  rel₅ : b₂ + b₅ = b₇
  b₁_eq_one : b₁ = 1
  b₂_eq_two : b₂ = 2

/-- The perimeter of the rectangle in the dissection puzzle -/
def perimeter (r : RectangleDissection) : ℕ :=
  2 * (r.b₆ + r.b₇)

/-- Theorem stating that the perimeter of the rectangle is 46 -/
theorem rectangle_perimeter_is_46 (r : RectangleDissection) : perimeter r = 46 := by
  sorry

end rectangle_perimeter_is_46_l1606_160663


namespace perpendicular_condition_l1606_160647

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between planes and between a line and a plane
variable (perp_planes : Plane → Plane → Prop)
variable (perp_line_plane : Line → Plane → Prop)

-- Define the relation of a line being within a plane
variable (line_in_plane : Line → Plane → Prop)

theorem perpendicular_condition (α β : Plane) (m : Line) 
  (h1 : α ≠ β) 
  (h2 : line_in_plane m α) : 
  (∀ m, line_in_plane m α → perp_line_plane m β → perp_planes α β) ∧ 
  (∃ m, line_in_plane m α ∧ perp_planes α β ∧ ¬perp_line_plane m β) := by
  sorry

end perpendicular_condition_l1606_160647


namespace point_in_first_quadrant_l1606_160676

theorem point_in_first_quadrant (α : Real) : 
  α ∈ Set.Icc 0 (2 * Real.pi) →
  (Real.sin α - Real.cos α > 0 ∧ Real.tan α > 0) ↔ 
  (α ∈ Set.Ioo (Real.pi / 4) (Real.pi / 2) ∪ Set.Ioo Real.pi (5 * Real.pi / 4)) := by
sorry

end point_in_first_quadrant_l1606_160676


namespace insertion_methods_l1606_160639

theorem insertion_methods (n : ℕ) (k : ℕ) : n = 5 ∧ k = 2 → (n + 1) * (n + 2) = 42 := by
  sorry

end insertion_methods_l1606_160639


namespace power_equation_l1606_160683

theorem power_equation (x y z : ℕ) : 
  3^x * 4^y = z → x - y = 9 → x = 9 → z = 19683 := by
  sorry

end power_equation_l1606_160683


namespace circle_center_coordinates_l1606_160669

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in the form ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y = l.c

/-- Check if a circle with center p is tangent to two parallel lines -/
def circleTangentToParallelLines (p : Point) (l1 l2 : Line) : Prop :=
  l1.a = l2.a ∧ l1.b = l2.b ∧ 
  abs (l1.a * p.x + l1.b * p.y - l1.c) = abs (l2.a * p.x + l2.b * p.y - l2.c)

theorem circle_center_coordinates : 
  ∃ (p : Point),
    circleTangentToParallelLines p (Line.mk 3 4 40) (Line.mk 3 4 (-20)) ∧
    pointOnLine p (Line.mk 1 (-2) 0) ∧
    p.x = 2 ∧ p.y = 1 := by
  sorry

end circle_center_coordinates_l1606_160669


namespace min_value_and_nonexistence_l1606_160655

theorem min_value_and_nonexistence (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 4 * b = (a * b) ^ (3/2)) :
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → a' + 4 * b' = (a' * b') ^ (3/2) → a' ^ 2 + 16 * b' ^ 2 ≥ 32) ∧
  ¬∃ a' b' : ℝ, a' > 0 ∧ b' > 0 ∧ a' + 4 * b' = (a' * b') ^ (3/2) ∧ a' + 3 * b' = 6 :=
by sorry

end min_value_and_nonexistence_l1606_160655


namespace train_length_calculation_l1606_160605

/-- Given a train passing a bridge, calculate its length -/
theorem train_length_calculation (train_speed : ℝ) (bridge_length : ℝ) (time_to_pass : ℝ) : 
  train_speed = 45 * 1000 / 3600 →
  bridge_length = 140 →
  time_to_pass = 52 →
  (train_speed * time_to_pass - bridge_length) = 510 := by
  sorry

#check train_length_calculation

end train_length_calculation_l1606_160605


namespace negation_of_existence_negation_of_inequality_negation_of_proposition_l1606_160694

theorem negation_of_existence (p : ℝ → Prop) : 
  (¬∃ x, p x) ↔ (∀ x, ¬p x) :=
by sorry

theorem negation_of_inequality : 
  (¬(x + 2 ≤ 0)) ↔ (x + 2 > 0) :=
by sorry

theorem negation_of_proposition : 
  (¬∃ x : ℝ, x + 2 ≤ 0) ↔ (∀ x : ℝ, x + 2 > 0) :=
by sorry

end negation_of_existence_negation_of_inequality_negation_of_proposition_l1606_160694


namespace square_difference_65_35_l1606_160621

theorem square_difference_65_35 : 65^2 - 35^2 = 3000 := by sorry

end square_difference_65_35_l1606_160621


namespace at_most_two_sides_equal_longest_diagonal_l1606_160699

/-- A convex polygon -/
structure ConvexPolygon where
  -- We don't need to define the full structure, just declare it exists
  mk :: 

/-- The longest diagonal of a convex polygon -/
def longest_diagonal (p : ConvexPolygon) : ℝ := sorry

/-- A side of a convex polygon -/
def side (p : ConvexPolygon) : ℝ := sorry

/-- The number of sides in a convex polygon that are equal to the longest diagonal -/
def num_sides_equal_to_longest_diagonal (p : ConvexPolygon) : ℕ := sorry

/-- Theorem: At most two sides of a convex polygon can be equal to its longest diagonal -/
theorem at_most_two_sides_equal_longest_diagonal (p : ConvexPolygon) :
  num_sides_equal_to_longest_diagonal p ≤ 2 := by sorry

end at_most_two_sides_equal_longest_diagonal_l1606_160699


namespace coin_distribution_theorem_l1606_160635

-- Define the number of cans
def num_cans : ℕ := 2015

-- Define the three initial configurations
def config_a (j : ℕ) : ℤ := 0
def config_b (j : ℕ) : ℤ := j
def config_c (j : ℕ) : ℤ := 2016 - j

-- Define the property that needs to be proven for each configuration
def has_solution (d : ℕ → ℤ) : Prop :=
  ∃ X : ℤ, ∀ j : ℕ, 1 ≤ j ∧ j ≤ num_cans → X ≡ d j [ZMOD j]

-- Theorem statement
theorem coin_distribution_theorem :
  has_solution config_a ∧ has_solution config_b ∧ has_solution config_c :=
sorry

end coin_distribution_theorem_l1606_160635


namespace monotonicity_of_g_minimum_a_for_negative_f_l1606_160695

noncomputable section

def f (a x : ℝ) : ℝ := x * Real.log (x + 1) + (1/2 - a) * x + 2 - a

def g (a x : ℝ) : ℝ := f a x + Real.log (x + 1) + (1/2) * x

theorem monotonicity_of_g (a : ℝ) :
  (a ≤ 2 → StrictMono (g a)) ∧
  (a > 2 → StrictAntiOn (g a) (Set.Ioo 0 (Real.exp (a - 2) - 1)) ∧
           StrictMono (g a ∘ (λ x => x + Real.exp (a - 2) - 1))) :=
sorry

theorem minimum_a_for_negative_f :
  (∃ (a : ℤ), ∃ (x : ℝ), x ≥ 0 ∧ f a x < 0) ∧
  (∀ (a : ℤ), a < 3 → ∀ (x : ℝ), x ≥ 0 → f a x ≥ 0) :=
sorry

end monotonicity_of_g_minimum_a_for_negative_f_l1606_160695


namespace right_rectangular_prism_x_value_l1606_160633

theorem right_rectangular_prism_x_value 
  (x : ℝ) 
  (h1 : x > 0) -- Ensure x is positive for valid logarithms
  (edge1 : ℝ := Real.log x / Real.log 5)
  (edge2 : ℝ := Real.log x / Real.log 6)
  (edge3 : ℝ := Real.log x / Real.log 10)
  (surface_area : ℝ := 2 * (edge1 * edge2 + edge1 * edge3 + edge2 * edge3))
  (volume : ℝ := edge1 * edge2 * edge3)
  (h2 : surface_area = 3 * volume) :
  x = 300^(2/3) :=
by sorry

end right_rectangular_prism_x_value_l1606_160633


namespace square_area_from_adjacent_points_l1606_160640

/-- The area of a square given two adjacent points on a Cartesian plane -/
theorem square_area_from_adjacent_points (x₁ y₁ x₂ y₂ : ℝ) :
  x₁ = 1 ∧ y₁ = 2 ∧ x₂ = 4 ∧ y₂ = 6 →
  (∃ (area : ℝ), area = 25 ∧ 
    area = ((x₂ - x₁)^2 + (y₂ - y₁)^2)) :=
by sorry


end square_area_from_adjacent_points_l1606_160640


namespace max_ab_value_l1606_160674

theorem max_ab_value (a b : ℝ) (h1 : 1 ≤ a - b) (h2 : a - b ≤ 2) (h3 : 3 ≤ a + b) (h4 : a + b ≤ 4) :
  ∃ (m : ℝ), m = 15/4 ∧ ab ≤ m ∧ ∃ (a' b' : ℝ), 1 ≤ a' - b' ∧ a' - b' ≤ 2 ∧ 3 ≤ a' + b' ∧ a' + b' ≤ 4 ∧ a' * b' = m :=
sorry

end max_ab_value_l1606_160674


namespace inverse_proportion_ordering_l1606_160679

/-- Represents a point on the inverse proportion function -/
structure InversePoint where
  x : ℝ
  y : ℝ
  k : ℝ
  h : y = k / x

/-- The theorem statement -/
theorem inverse_proportion_ordering
  (p₁ : InversePoint)
  (p₂ : InversePoint)
  (p₃ : InversePoint)
  (h₁ : p₁.x = -1)
  (h₂ : p₂.x = 2)
  (h₃ : p₃.x = 3)
  (hk : p₁.k = p₂.k ∧ p₂.k = p₃.k ∧ p₁.k < 0) :
  p₁.y > p₃.y ∧ p₃.y > p₂.y :=
by sorry

end inverse_proportion_ordering_l1606_160679


namespace student_arrangement_counts_l1606_160689

/-- The number of ways to arrange 5 male and 2 female students in a row --/
def arrange_students (n_male : ℕ) (n_female : ℕ) : ℕ → ℕ → ℕ
| 1 => λ _ => 1400  -- females must be next to each other
| 2 => λ _ => 3600  -- females must not be next to each other
| 3 => λ _ => 3720  -- specific placement restrictions for females
| _ => λ _ => 0     -- undefined for other cases

/-- Theorem stating the correct number of arrangements for each scenario --/
theorem student_arrangement_counts :
  let n_male := 5
  let n_female := 2
  (arrange_students n_male n_female 1 0 = 1400) ∧
  (arrange_students n_male n_female 2 0 = 3600) ∧
  (arrange_students n_male n_female 3 0 = 3720) :=
by sorry


end student_arrangement_counts_l1606_160689


namespace total_cds_on_shelf_l1606_160625

/-- The number of CDs that a single rack can hold -/
def cds_per_rack : ℕ := 8

/-- The number of racks that can fit on a shelf -/
def racks_per_shelf : ℕ := 4

/-- Theorem: The total number of CDs that can fit on a shelf is 32 -/
theorem total_cds_on_shelf : cds_per_rack * racks_per_shelf = 32 := by
  sorry

end total_cds_on_shelf_l1606_160625


namespace inequality_proof_l1606_160646

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x / Real.sqrt (y + z)) + (y / Real.sqrt (z + x)) + (z / Real.sqrt (x + y)) ≥ Real.sqrt ((3 / 2) * (x + y + z)) :=
by sorry

end inequality_proof_l1606_160646


namespace power_of_power_l1606_160616

theorem power_of_power (x : ℝ) : (x^3)^2 = x^6 := by
  sorry

end power_of_power_l1606_160616


namespace find_n_l1606_160687

def quarter_value : ℕ := 25
def nickel_value : ℕ := 5

theorem find_n : ∃ n : ℕ, 
  15 * quarter_value + 20 * nickel_value = 10 * quarter_value + n * nickel_value ∧ n = 45 := by
  sorry

end find_n_l1606_160687


namespace workshop_average_salary_l1606_160608

/-- Given a workshop with workers, prove that the average salary of all workers is 8000 Rs. -/
theorem workshop_average_salary
  (total_workers : ℕ)
  (technicians : ℕ)
  (tech_salary : ℕ)
  (non_tech_salary : ℕ)
  (h1 : total_workers = 21)
  (h2 : technicians = 7)
  (h3 : tech_salary = 12000)
  (h4 : non_tech_salary = 6000) :
  (technicians * tech_salary + (total_workers - technicians) * non_tech_salary) / total_workers = 8000 := by
  sorry

end workshop_average_salary_l1606_160608


namespace quadratic_expression_k_value_l1606_160630

theorem quadratic_expression_k_value :
  ∀ a h k : ℝ, (∀ x : ℝ, x^2 - 8*x = a*(x - h)^2 + k) → k = -16 := by
  sorry

end quadratic_expression_k_value_l1606_160630


namespace alcohol_mixture_proof_l1606_160628

/-- Proves that adding 2.4 liters of pure alcohol to a 6-liter solution
    that is 30% alcohol results in a 50% alcohol solution -/
theorem alcohol_mixture_proof :
  let initial_volume : ℝ := 6
  let initial_concentration : ℝ := 0.3
  let final_concentration : ℝ := 0.5
  let added_alcohol : ℝ := 2.4

  let initial_alcohol : ℝ := initial_volume * initial_concentration
  let final_volume : ℝ := initial_volume + added_alcohol
  let final_alcohol : ℝ := initial_alcohol + added_alcohol

  final_alcohol / final_volume = final_concentration :=
by
  sorry


end alcohol_mixture_proof_l1606_160628


namespace f_properties_l1606_160654

noncomputable def f (x : ℝ) : ℝ := Real.cos (Real.pi / 2 - x) * Real.cos x + Real.sqrt 3 * Real.sin x ^ 2

theorem f_properties :
  -- Smallest positive period is π
  (∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧ 
    (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T')) ∧
  -- Monotonically decreasing in [5π/12 + kπ, 11π/12 + kπ]
  (∀ (k : ℤ), StrictMonoOn f (Set.Icc (5 * Real.pi / 12 + k * Real.pi) (11 * Real.pi / 12 + k * Real.pi))) ∧
  -- Minimum and maximum values on [π/6, π/2]
  (∃ (x_min x_max : ℝ), x_min ∈ Set.Icc (Real.pi / 6) (Real.pi / 2) ∧
                        x_max ∈ Set.Icc (Real.pi / 6) (Real.pi / 2) ∧
                        (∀ (x : ℝ), x ∈ Set.Icc (Real.pi / 6) (Real.pi / 2) → 
                          f x_min ≤ f x ∧ f x ≤ f x_max) ∧
                        f x_min = Real.sqrt 3 / 2 ∧
                        f x_max = Real.sqrt 3 / 2 + 1) :=
by sorry

end f_properties_l1606_160654
