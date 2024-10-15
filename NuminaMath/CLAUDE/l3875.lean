import Mathlib

namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l3875_387566

theorem ellipse_eccentricity (k : ℝ) :
  (∃ x y : ℝ, x^2 / 9 + y^2 / (4 + k) = 1) →
  (∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c / a = 4/5 ∧
    ((a^2 = 9 ∧ b^2 = 4 + k) ∨ (a^2 = 4 + k ∧ b^2 = 9)) ∧
    c^2 = a^2 - b^2) →
  k = -19/25 ∨ k = 21 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l3875_387566


namespace NUMINAMATH_CALUDE_naomi_bike_count_l3875_387517

theorem naomi_bike_count (total_wheels : ℕ) (childrens_bikes : ℕ) (regular_bike_wheels : ℕ) (childrens_bike_wheels : ℕ) : 
  total_wheels = 58 →
  childrens_bikes = 11 →
  regular_bike_wheels = 2 →
  childrens_bike_wheels = 4 →
  ∃ (regular_bikes : ℕ), regular_bikes = 7 ∧ 
    total_wheels = regular_bikes * regular_bike_wheels + childrens_bikes * childrens_bike_wheels :=
by
  sorry

end NUMINAMATH_CALUDE_naomi_bike_count_l3875_387517


namespace NUMINAMATH_CALUDE_tony_weightlifting_ratio_l3875_387575

/-- Given Tony's weightlifting capabilities, prove the ratio of his military press to curl weight. -/
theorem tony_weightlifting_ratio :
  ∀ (curl_weight military_press_weight squat_weight : ℝ),
    curl_weight = 90 →
    squat_weight = 5 * military_press_weight →
    squat_weight = 900 →
    military_press_weight / curl_weight = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_tony_weightlifting_ratio_l3875_387575


namespace NUMINAMATH_CALUDE_x_fourth_plus_y_fourth_l3875_387543

theorem x_fourth_plus_y_fourth (x y : ℝ) 
  (h1 : x - y = 2) 
  (h2 : x * y = 48) : 
  x^4 + y^4 = 5392 := by
sorry

end NUMINAMATH_CALUDE_x_fourth_plus_y_fourth_l3875_387543


namespace NUMINAMATH_CALUDE_javier_ate_five_meat_ravioli_l3875_387519

/-- Represents the weight of each type of ravioli in ounces -/
structure RavioliWeights where
  meat : Float
  pumpkin : Float
  cheese : Float

/-- Represents the number of each type of ravioli eaten by Javier -/
structure JavierMeal where
  meat : Nat
  pumpkin : Nat
  cheese : Nat

/-- Calculates the total weight of Javier's meal -/
def mealWeight (weights : RavioliWeights) (meal : JavierMeal) : Float :=
  weights.meat * meal.meat.toFloat + weights.pumpkin * meal.pumpkin.toFloat + weights.cheese * meal.cheese.toFloat

/-- Theorem: Given the conditions, Javier ate 5 meat ravioli -/
theorem javier_ate_five_meat_ravioli (weights : RavioliWeights) (meal : JavierMeal) : 
  weights.meat = 1.5 ∧ 
  weights.pumpkin = 1.25 ∧ 
  weights.cheese = 1 ∧ 
  meal.pumpkin = 2 ∧ 
  meal.cheese = 4 ∧ 
  mealWeight weights meal = 15 → 
  meal.meat = 5 := by
  sorry


end NUMINAMATH_CALUDE_javier_ate_five_meat_ravioli_l3875_387519


namespace NUMINAMATH_CALUDE_books_distribution_l3875_387567

/-- Number of ways to distribute books among students -/
def distribute_books (n_books : ℕ) (n_students : ℕ) : ℕ :=
  -- Definition goes here
  sorry

/-- Theorem: Distributing 5 books among 3 students results in 90 different methods -/
theorem books_distribution :
  distribute_books 5 3 = 90 :=
by
  sorry

end NUMINAMATH_CALUDE_books_distribution_l3875_387567


namespace NUMINAMATH_CALUDE_smallest_sum_of_sequence_l3875_387521

theorem smallest_sum_of_sequence (A B C D : ℤ) : 
  A > 0 → B > 0 → C > 0 →  -- A, B, C are positive integers
  (∃ r : ℤ, C - B = B - A) →  -- A, B, C form an arithmetic sequence
  (∃ q : ℚ, C = B * q ∧ D = C * q) →  -- B, C, D form a geometric sequence
  C = (7 * B) / 4 →  -- C/B = 7/4
  (∀ A' B' C' D' : ℤ, 
    A' > 0 → B' > 0 → C' > 0 → 
    (∃ r : ℤ, C' - B' = B' - A') → 
    (∃ q : ℚ, C' = B' * q ∧ D' = C' * q) → 
    C' = (7 * B') / 4 → 
    A + B + C + D ≤ A' + B' + C' + D') →
  A + B + C + D = 97 := by
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_sequence_l3875_387521


namespace NUMINAMATH_CALUDE_hike_length_is_83_l3875_387515

/-- Represents the length of a 5-day hike satisfying specific conditions -/
def HikeLength (a b c d e : ℝ) : Prop :=
  a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ e ≥ 0 ∧  -- Non-negative distances
  a + b = 36 ∧                              -- First two days
  (b + c + d) / 3 = 15 ∧                    -- Average of days 2, 3, 4
  c + d + e = 45 ∧                          -- Last three days
  a + c + e = 38                            -- Days 1, 3, 5

/-- The theorem stating that the total hike length is 83 miles -/
theorem hike_length_is_83 {a b c d e : ℝ} (h : HikeLength a b c d e) :
  a + b + c + d + e = 83 := by
  sorry


end NUMINAMATH_CALUDE_hike_length_is_83_l3875_387515


namespace NUMINAMATH_CALUDE_brown_leaves_percentage_l3875_387594

/-- Given a collection of leaves with known percentages of green and yellow leaves,
    calculate the percentage of brown leaves. -/
theorem brown_leaves_percentage
  (total_leaves : ℕ)
  (green_percentage : ℚ)
  (yellow_count : ℕ)
  (h1 : total_leaves = 25)
  (h2 : green_percentage = 1/5)
  (h3 : yellow_count = 15) :
  (total_leaves : ℚ) - green_percentage * total_leaves - yellow_count = 1/5 * total_leaves :=
sorry

end NUMINAMATH_CALUDE_brown_leaves_percentage_l3875_387594


namespace NUMINAMATH_CALUDE_successful_table_filling_l3875_387581

theorem successful_table_filling :
  ∃ (t : Fin 6 → Fin 3 → Bool),
    ∀ (r1 r2 : Fin 6) (c1 c2 : Fin 3),
      r1 ≠ r2 → c1 ≠ c2 →
        (t r1 c1 = t r1 c2 ∧ t r1 c1 = t r2 c1 ∧ t r1 c1 = t r2 c2) = False :=
by sorry

end NUMINAMATH_CALUDE_successful_table_filling_l3875_387581


namespace NUMINAMATH_CALUDE_corner_removed_cube_edge_count_l3875_387587

/-- Represents a cube with a given side length -/
structure Cube :=
  (sideLength : ℝ)

/-- Represents the solid formed by removing smaller cubes from the corners of a larger cube -/
structure CornerRemovedCube :=
  (originalCube : Cube)
  (removedCubeSize : ℝ)

/-- Calculates the number of edges in the solid formed by removing smaller cubes from the corners of a larger cube -/
def edgeCount (c : CornerRemovedCube) : ℕ :=
  12 * 2  -- Each original edge is split into two

/-- Theorem stating that removing cubes of side length 2 from each corner of a cube with side length 4 results in a solid with 24 edges -/
theorem corner_removed_cube_edge_count :
  let originalCube : Cube := ⟨4⟩
  let cornerRemovedCube : CornerRemovedCube := ⟨originalCube, 2⟩
  edgeCount cornerRemovedCube = 24 :=
by sorry

end NUMINAMATH_CALUDE_corner_removed_cube_edge_count_l3875_387587


namespace NUMINAMATH_CALUDE_units_digit_of_expression_l3875_387558

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The units digit of 8 * 19 * 1983 - 8^3 is 4 -/
theorem units_digit_of_expression : unitsDigit (8 * 19 * 1983 - 8^3) = 4 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_expression_l3875_387558


namespace NUMINAMATH_CALUDE_abs_x_minus_3_minus_sqrt_x_minus_4_squared_l3875_387585

theorem abs_x_minus_3_minus_sqrt_x_minus_4_squared (x : ℝ) (h : x < 3) :
  |x - 3 - Real.sqrt ((x - 4)^2)| = 7 - 2*x := by sorry

end NUMINAMATH_CALUDE_abs_x_minus_3_minus_sqrt_x_minus_4_squared_l3875_387585


namespace NUMINAMATH_CALUDE_sqrt_equality_implies_relation_l3875_387564

theorem sqrt_equality_implies_relation (a b c : ℕ+) :
  (a.val ^ 2 : ℝ) - (b.val : ℝ) / (c.val : ℝ) ≥ 0 →
  Real.sqrt ((a.val ^ 2 : ℝ) - (b.val : ℝ) / (c.val : ℝ)) = a.val - Real.sqrt ((b.val : ℝ) / (c.val : ℝ)) →
  b = a ^ 2 * c := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equality_implies_relation_l3875_387564


namespace NUMINAMATH_CALUDE_inverse_of_A_l3875_387511

def A : Matrix (Fin 2) (Fin 2) ℚ := !![4, 5; -2, 9]

theorem inverse_of_A :
  let A_inv : Matrix (Fin 2) (Fin 2) ℚ := !![9/46, -5/46; 1/23, 2/23]
  A * A_inv = 1 ∧ A_inv * A = 1 :=
by sorry

end NUMINAMATH_CALUDE_inverse_of_A_l3875_387511


namespace NUMINAMATH_CALUDE_continued_fraction_evaluation_l3875_387574

theorem continued_fraction_evaluation :
  2 + (3 / (4 + (5/6))) = 76/29 := by
  sorry

end NUMINAMATH_CALUDE_continued_fraction_evaluation_l3875_387574


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_intersection_l3875_387593

-- Define the ellipse and hyperbola
def ellipse (m : ℝ) (x y : ℝ) : Prop := x^2 / 10 + y^2 / m = 1
def hyperbola (b : ℝ) (x y : ℝ) : Prop := x^2 - y^2 / b = 1

-- Define the intersection point
def intersection_point (x y : ℝ) : Prop := x = Real.sqrt 10 / 3

-- Define that the ellipse and hyperbola have the same foci
def same_foci (m b : ℝ) : Prop := 10 - m = 1 + b

-- Theorem statement
theorem ellipse_hyperbola_intersection (m b : ℝ) :
  (∃ y, ellipse m (Real.sqrt 10 / 3) y ∧ 
        hyperbola b (Real.sqrt 10 / 3) y ∧
        intersection_point (Real.sqrt 10 / 3) y) →
  same_foci m b →
  m = 1 ∧ b = 8 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_intersection_l3875_387593


namespace NUMINAMATH_CALUDE_kaydence_age_l3875_387572

/-- Kaydence's family ages problem -/
theorem kaydence_age (total_age father_age mother_age brother_age sister_age kaydence_age : ℕ) :
  total_age = 200 ∧
  father_age = 60 ∧
  mother_age = father_age - 2 ∧
  brother_age = father_age / 2 ∧
  sister_age = 40 ∧
  total_age = father_age + mother_age + brother_age + sister_age + kaydence_age →
  kaydence_age = 12 := by
  sorry

end NUMINAMATH_CALUDE_kaydence_age_l3875_387572


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l3875_387506

-- Problem 1
theorem problem_1 : (-2)^3 / (-2)^2 * (1/2)^0 = -2 := by sorry

-- Problem 2
theorem problem_2 : 199 * 201 + 1 = 40000 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l3875_387506


namespace NUMINAMATH_CALUDE_dave_winfield_home_runs_l3875_387599

theorem dave_winfield_home_runs : ∃ (x : ℕ), 
  (755 = 2 * x - 175) ∧ x = 465 := by sorry

end NUMINAMATH_CALUDE_dave_winfield_home_runs_l3875_387599


namespace NUMINAMATH_CALUDE_surface_area_ratio_of_spheres_l3875_387504

theorem surface_area_ratio_of_spheres (r₁ r₂ : ℝ) (h : r₁ / r₂ = 1 / 2) :
  (4 * Real.pi * r₁^2) / (4 * Real.pi * r₂^2) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_ratio_of_spheres_l3875_387504


namespace NUMINAMATH_CALUDE_largest_remainder_209_l3875_387516

theorem largest_remainder_209 :
  (∀ n : ℕ, n < 120 → ∃ k r : ℕ, 209 = n * k + r ∧ r < n ∧ r ≤ 104) ∧
  (∃ n : ℕ, n < 120 ∧ ∃ k : ℕ, 209 = n * k + 104) ∧
  (∀ n : ℕ, n < 90 → ∃ k r : ℕ, 209 = n * k + r ∧ r < n ∧ r ≤ 69) ∧
  (∃ n : ℕ, n < 90 ∧ ∃ k : ℕ, 209 = n * k + 69) :=
by sorry

end NUMINAMATH_CALUDE_largest_remainder_209_l3875_387516


namespace NUMINAMATH_CALUDE_largest_number_is_nineteen_l3875_387518

theorem largest_number_is_nineteen 
  (a b c : ℕ) 
  (sum_ab : a + b = 16)
  (sum_ac : a + c = 20)
  (sum_bc : b + c = 23) :
  max a (max b c) = 19 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_is_nineteen_l3875_387518


namespace NUMINAMATH_CALUDE_perimeter_is_22_l3875_387571

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 14 - y^2 / 11 = 1

-- Define the focus F₁
def F₁ : ℝ × ℝ := sorry

-- Define the line l
def l : Set (ℝ × ℝ) := sorry

-- Define points P and Q
def P : ℝ × ℝ := sorry
def Q : ℝ × ℝ := sorry

-- State that P and Q are on the hyperbola
axiom P_on_hyperbola : hyperbola P.1 P.2
axiom Q_on_hyperbola : hyperbola Q.1 Q.2

-- State that P and Q are on line l
axiom P_on_l : P ∈ l
axiom Q_on_l : Q ∈ l

-- State that line l passes through the origin
axiom l_through_origin : (0, 0) ∈ l

-- State that PF₁ · QF₁ = 0
axiom PF₁_perp_QF₁ : (P.1 - F₁.1, P.2 - F₁.2) • (Q.1 - F₁.1, Q.2 - F₁.2) = 0

-- Define the perimeter of triangle PF₁Q
def perimeter_PF₁Q : ℝ := sorry

-- Theorem to prove
theorem perimeter_is_22 : perimeter_PF₁Q = 22 := sorry

end NUMINAMATH_CALUDE_perimeter_is_22_l3875_387571


namespace NUMINAMATH_CALUDE_sqrt_combination_l3875_387509

theorem sqrt_combination (t : ℝ) : 
  (∃ k : ℝ, k * Real.sqrt 12 = Real.sqrt (2 * t - 1)) → 
  Real.sqrt 12 = 2 * Real.sqrt 3 → 
  t = 2 := by
sorry

end NUMINAMATH_CALUDE_sqrt_combination_l3875_387509


namespace NUMINAMATH_CALUDE_movie_production_cost_ratio_l3875_387590

/-- Proves that the ratio of equipment rental cost to the combined cost of food and actors is 2:1 --/
theorem movie_production_cost_ratio :
  let actor_cost : ℕ := 1200
  let num_people : ℕ := 50
  let food_cost_per_person : ℕ := 3
  let total_food_cost : ℕ := num_people * food_cost_per_person
  let combined_cost : ℕ := actor_cost + total_food_cost
  let selling_price : ℕ := 10000
  let profit : ℕ := 5950
  let total_cost : ℕ := selling_price - profit
  let equipment_cost : ℕ := total_cost - combined_cost
  equipment_cost / combined_cost = 2 := by sorry

end NUMINAMATH_CALUDE_movie_production_cost_ratio_l3875_387590


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3875_387562

-- Define the sets M and N
def M : Set ℝ := {x | 2 * x - 3 < 1}
def N : Set ℝ := {x | -1 < x ∧ x < 3}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {x : ℝ | -1 < x ∧ x < 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3875_387562


namespace NUMINAMATH_CALUDE_quadratic_sum_of_constants_l3875_387512

-- Define the quadratic function
def f (x : ℝ) : ℝ := 4*x^2 - 16*x - 64

-- Define the completed square form
def g (x a b c : ℝ) : ℝ := a*(x+b)^2 + c

-- Theorem statement
theorem quadratic_sum_of_constants :
  ∃ (a b c : ℝ), (∀ x, f x = g x a b c) ∧ (a + b + c = -78) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_of_constants_l3875_387512


namespace NUMINAMATH_CALUDE_B_complete_work_in_40_days_l3875_387520

/-- The number of days it takes A to complete the work alone -/
def A_days : ℝ := 45

/-- The number of days A and B work together -/
def together_days : ℝ := 9

/-- The number of days B works alone after A leaves -/
def B_alone_days : ℝ := 23

/-- The number of days it takes B to complete the work alone -/
def B_days : ℝ := 40

/-- Theorem stating that given the conditions, B can complete the work alone in 40 days -/
theorem B_complete_work_in_40_days :
  (together_days * (1 / A_days + 1 / B_days)) + (B_alone_days * (1 / B_days)) = 1 :=
by sorry

end NUMINAMATH_CALUDE_B_complete_work_in_40_days_l3875_387520


namespace NUMINAMATH_CALUDE_tomato_picking_second_week_l3875_387591

/-- Represents the number of tomatoes picked in each week -/
structure TomatoPicking where
  initial : ℕ
  first_week : ℕ
  second_week : ℕ
  third_week : ℕ
  remaining : ℕ

/-- Checks if the tomato picking satisfies the given conditions -/
def is_valid_picking (p : TomatoPicking) : Prop :=
  p.initial = 100 ∧
  p.first_week = p.initial / 4 ∧
  p.third_week = 2 * p.second_week ∧
  p.remaining = 15 ∧
  p.first_week + p.second_week + p.third_week + p.remaining = p.initial

theorem tomato_picking_second_week :
  ∀ p : TomatoPicking, is_valid_picking p → p.second_week = 20 := by
  sorry

end NUMINAMATH_CALUDE_tomato_picking_second_week_l3875_387591


namespace NUMINAMATH_CALUDE_exam_marks_proof_l3875_387528

theorem exam_marks_proof (T : ℝ) 
  (h1 : 0.3 * T + 50 = 199.99999999999997) 
  (passing_mark : ℝ := 199.99999999999997) 
  (second_candidate_score : ℝ := 0.45 * T) : 
  second_candidate_score - passing_mark = 25 := by
sorry

end NUMINAMATH_CALUDE_exam_marks_proof_l3875_387528


namespace NUMINAMATH_CALUDE_total_vehicles_in_yard_l3875_387546

theorem total_vehicles_in_yard (num_trucks : ℕ) (num_tanks : ℕ) : 
  num_trucks = 20 → 
  num_tanks = 5 * num_trucks → 
  num_tanks + num_trucks = 120 :=
by
  sorry

end NUMINAMATH_CALUDE_total_vehicles_in_yard_l3875_387546


namespace NUMINAMATH_CALUDE_donuts_left_for_coworkers_l3875_387525

def total_donuts : ℕ := 30
def gluten_free_donuts : ℕ := 12
def regular_donuts : ℕ := total_donuts - gluten_free_donuts

def gluten_free_eaten_driving : ℕ := 1
def regular_eaten_driving : ℕ := 0

def gluten_free_afternoon_snack : ℕ := 2
def regular_afternoon_snack : ℕ := 4

theorem donuts_left_for_coworkers :
  total_donuts - 
  (gluten_free_eaten_driving + regular_eaten_driving + 
   gluten_free_afternoon_snack + regular_afternoon_snack) = 23 := by
  sorry

end NUMINAMATH_CALUDE_donuts_left_for_coworkers_l3875_387525


namespace NUMINAMATH_CALUDE_fraction_of_product_l3875_387551

theorem fraction_of_product (x : ℚ) : x * (1/2 * 2/5 * 5100) = 765.0000000000001 → x = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_product_l3875_387551


namespace NUMINAMATH_CALUDE_acute_angle_condition_l3875_387508

/-- Given vectors a and b in ℝ², prove that x > -3 is a necessary but not sufficient condition
    for the angle between a and b to be acute -/
theorem acute_angle_condition (a b : ℝ × ℝ) (x : ℝ) 
    (ha : a = (2, 3)) (hb : b = (x, 2)) : 
    (∃ (y : ℝ), y > -3 ∧ y ≠ x ∧ 
      ((a.1 * b.1 + a.2 * b.2 > 0) ∧ 
       ¬(∃ (k : ℝ), k ≠ 0 ∧ a.1 = k * b.1 ∧ a.2 = k * b.2))) ∧
    (x > -3 → 
      (a.1 * b.1 + a.2 * b.2 > 0) ∧ 
      ¬(∃ (k : ℝ), k ≠ 0 ∧ a.1 = k * b.1 ∧ a.2 = k * b.2)) :=
by sorry

end NUMINAMATH_CALUDE_acute_angle_condition_l3875_387508


namespace NUMINAMATH_CALUDE_min_value_sum_l3875_387522

theorem min_value_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / (3 * b) + b / (6 * c) + c / (9 * a) + (a^2 * b) / (18 * b * c)) ≥ 4/9 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_l3875_387522


namespace NUMINAMATH_CALUDE_stump_pulling_force_l3875_387514

/-- The force required to pull a stump varies inversely with the lever length -/
def inverse_variation (force length : ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ force * length = k

theorem stump_pulling_force 
  (force_10 length_10 force_25 length_25 : ℝ)
  (h1 : force_10 = 180)
  (h2 : length_10 = 10)
  (h3 : length_25 = 25)
  (h4 : inverse_variation force_10 length_10)
  (h5 : inverse_variation force_25 length_25)
  : force_25 = 72 := by
sorry

end NUMINAMATH_CALUDE_stump_pulling_force_l3875_387514


namespace NUMINAMATH_CALUDE_point_quadrant_l3875_387549

/-- If a point A(a,b) is in the first quadrant, then the point B(a,-b) is in the fourth quadrant. -/
theorem point_quadrant (a b : ℝ) (h : a > 0 ∧ b > 0) : a > 0 ∧ -b < 0 := by
  sorry

end NUMINAMATH_CALUDE_point_quadrant_l3875_387549


namespace NUMINAMATH_CALUDE_function_properties_l3875_387523

-- Define the function f(x) = ax³ + bx - 1
def f (a b x : ℝ) : ℝ := a * x^3 + b * x - 1

-- Define the derivative of f
def f_derivative (a b x : ℝ) : ℝ := 3 * a * x^2 + b

theorem function_properties :
  ∃ (a b : ℝ),
    (f a b 1 = -3) ∧
    (f_derivative a b 1 = 0) ∧
    (a = 1) ∧
    (b = -3) ∧
    (∀ x ∈ Set.Icc (-2) 3, f a b x ≤ 17) ∧
    (∃ x ∈ Set.Icc (-2) 3, f a b x = 17) ∧
    (∀ x ∈ Set.Icc (-2) 3, f a b x ≥ -3) ∧
    (∃ x ∈ Set.Icc (-2) 3, f a b x = -3) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l3875_387523


namespace NUMINAMATH_CALUDE_percentage_problem_l3875_387542

theorem percentage_problem (P : ℝ) : P = 35 ↔ (P / 100) * 180 - (1 / 3) * (P / 100) * 180 = 42 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3875_387542


namespace NUMINAMATH_CALUDE_at_least_one_travels_to_beijing_l3875_387559

theorem at_least_one_travels_to_beijing 
  (prob_A : ℝ) 
  (prob_B : ℝ) 
  (h1 : prob_A = 1/3) 
  (h2 : prob_B = 1/4) 
  (h3 : 0 ≤ prob_A ∧ prob_A ≤ 1) 
  (h4 : 0 ≤ prob_B ∧ prob_B ≤ 1) : 
  1 - (1 - prob_A) * (1 - prob_B) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_at_least_one_travels_to_beijing_l3875_387559


namespace NUMINAMATH_CALUDE_power_of_power_l3875_387584

theorem power_of_power : (2^3)^3 = 512 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l3875_387584


namespace NUMINAMATH_CALUDE_root_of_two_equations_l3875_387568

theorem root_of_two_equations (p q r s t k : ℂ) 
  (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) (ht : t ≠ 0)
  (eq1 : p * k^5 + q * k^4 + r * k^3 + s * k^2 + t * k + p = 0)
  (eq2 : q * k^5 + r * k^4 + s * k^3 + t * k^2 + p * k + q = 0) :
  k = 1 ∨ k = Complex.exp (Complex.I * π / 3) ∨ 
  k = Complex.exp (-Complex.I * π / 3) ∨ k = -1 ∨ 
  k = Complex.exp (2 * Complex.I * π / 3) ∨ 
  k = Complex.exp (-2 * Complex.I * π / 3) := by
  sorry

end NUMINAMATH_CALUDE_root_of_two_equations_l3875_387568


namespace NUMINAMATH_CALUDE_range_of_f_l3875_387563

def f (x : ℝ) : ℝ := -x^2 + 2*x - 3

theorem range_of_f :
  ∃ (a b : ℝ), a = -3 ∧ b = -2 ∧
  (∀ x ∈ Set.Icc 0 2, a ≤ f x ∧ f x ≤ b) ∧
  (∀ y ∈ Set.Icc a b, ∃ x ∈ Set.Icc 0 2, f x = y) :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l3875_387563


namespace NUMINAMATH_CALUDE_opposite_reciprocal_abs_l3875_387545

theorem opposite_reciprocal_abs (x : ℚ) (h : x = -4/3) : 
  (-x = 4/3) ∧ (x⁻¹ = -3/4) ∧ (|x| = 4/3) := by
  sorry

end NUMINAMATH_CALUDE_opposite_reciprocal_abs_l3875_387545


namespace NUMINAMATH_CALUDE_number_problem_l3875_387576

theorem number_problem (x : ℚ) : x - (3/5) * x = 56 → x = 140 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l3875_387576


namespace NUMINAMATH_CALUDE_car_speed_calculation_l3875_387580

/-- Represents the speed of a car during a journey -/
structure CarJourney where
  first_speed : ℝ  -- Speed for the first 160 km
  second_speed : ℝ  -- Speed for the next 160 km
  average_speed : ℝ  -- Average speed for the entire 320 km

/-- Theorem stating the speed of the car during the next 160 km -/
theorem car_speed_calculation (journey : CarJourney) 
  (h1 : journey.first_speed = 70)
  (h2 : journey.average_speed = 74.67) : 
  journey.second_speed = 80 := by
  sorry

#check car_speed_calculation

end NUMINAMATH_CALUDE_car_speed_calculation_l3875_387580


namespace NUMINAMATH_CALUDE_divisor_problem_l3875_387583

theorem divisor_problem (original : Nat) (subtracted : Nat) (remaining : Nat) :
  original = 165826 →
  subtracted = 2 →
  remaining = original - subtracted →
  (∃ (d : Nat), d > 1 ∧ remaining % d = 0 ∧ ∀ (k : Nat), k > d → remaining % k ≠ 0) →
  (∃ (d : Nat), d = 2 ∧ remaining % d = 0 ∧ ∀ (k : Nat), k > d → remaining % k ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_divisor_problem_l3875_387583


namespace NUMINAMATH_CALUDE_sum_of_cubes_of_roots_l3875_387586

theorem sum_of_cubes_of_roots (a b c : ℝ) : 
  (a^3 - 2*a^2 + 2*a - 3 = 0) →
  (b^3 - 2*b^2 + 2*b - 3 = 0) →
  (c^3 - 2*c^2 + 2*c - 3 = 0) →
  a^3 + b^3 + c^3 = 5 := by
sorry

end NUMINAMATH_CALUDE_sum_of_cubes_of_roots_l3875_387586


namespace NUMINAMATH_CALUDE_point_P_satisfies_conditions_l3875_387532

-- Define the curve C
def C (x : ℝ) : ℝ := x^3 - 10*x + 3

-- Define the derivative of C
def C' (x : ℝ) : ℝ := 3*x^2 - 10

theorem point_P_satisfies_conditions : 
  let x₀ : ℝ := -2
  let y₀ : ℝ := 15
  (x₀ < 0) ∧ 
  (C x₀ = y₀) ∧ 
  (C' x₀ = 2) := by sorry

end NUMINAMATH_CALUDE_point_P_satisfies_conditions_l3875_387532


namespace NUMINAMATH_CALUDE_janet_time_saved_l3875_387548

/-- The number of minutes Janet spends looking for keys daily -/
def keys_time : ℕ := 8

/-- The number of minutes Janet spends complaining after finding keys daily -/
def complain_time : ℕ := 3

/-- The number of minutes Janet spends searching for phone daily -/
def phone_time : ℕ := 5

/-- The number of minutes Janet spends looking for wallet daily -/
def wallet_time : ℕ := 4

/-- The number of minutes Janet spends trying to remember sunglasses location daily -/
def sunglasses_time : ℕ := 2

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- Theorem: Janet will save 154 minutes per week by stopping all these activities -/
theorem janet_time_saved :
  (keys_time + complain_time + phone_time + wallet_time + sunglasses_time) * days_in_week = 154 := by
  sorry

end NUMINAMATH_CALUDE_janet_time_saved_l3875_387548


namespace NUMINAMATH_CALUDE_total_albums_l3875_387561

theorem total_albums (adele bridget katrina miriam : ℕ) : 
  adele = 30 →
  bridget = adele - 15 →
  katrina = 6 * bridget →
  miriam = 5 * katrina →
  adele + bridget + katrina + miriam = 585 :=
by
  sorry

end NUMINAMATH_CALUDE_total_albums_l3875_387561


namespace NUMINAMATH_CALUDE_product_of_two_greatest_unattainable_scores_l3875_387526

/-- A score is attainable if it can be expressed as a non-negative integer combination of 19, 9, and 8. -/
def IsAttainable (n : ℕ) : Prop :=
  ∃ a b c : ℕ, n = 19 * a + 9 * b + 8 * c

/-- The set of all attainable scores. -/
def AttainableScores : Set ℕ :=
  {n : ℕ | IsAttainable n}

/-- The set of all unattainable scores. -/
def UnattainableScores : Set ℕ :=
  {n : ℕ | ¬IsAttainable n}

/-- The two greatest unattainable scores. -/
def TwoGreatestUnattainableScores : Fin 2 → ℕ :=
  fun i => if i = 0 then 39 else 31

theorem product_of_two_greatest_unattainable_scores :
  (TwoGreatestUnattainableScores 0) * (TwoGreatestUnattainableScores 1) = 1209 ∧
  (∀ n : ℕ, n ∈ UnattainableScores → n ≤ (TwoGreatestUnattainableScores 0)) ∧
  (∀ n : ℕ, n ∈ UnattainableScores ∧ n ≠ (TwoGreatestUnattainableScores 0) → n ≤ (TwoGreatestUnattainableScores 1)) :=
by sorry

end NUMINAMATH_CALUDE_product_of_two_greatest_unattainable_scores_l3875_387526


namespace NUMINAMATH_CALUDE_audiobook_listening_time_l3875_387529

/-- Calculates the average daily listening time for audiobooks -/
def average_daily_listening_time (num_audiobooks : ℕ) (audiobook_length : ℕ) (total_days : ℕ) : ℚ :=
  (num_audiobooks * audiobook_length : ℚ) / total_days

/-- Proves that the average daily listening time is 2 hours given the specific conditions -/
theorem audiobook_listening_time :
  let num_audiobooks : ℕ := 6
  let audiobook_length : ℕ := 30
  let total_days : ℕ := 90
  average_daily_listening_time num_audiobooks audiobook_length total_days = 2 := by
  sorry

end NUMINAMATH_CALUDE_audiobook_listening_time_l3875_387529


namespace NUMINAMATH_CALUDE_union_equality_implies_a_equals_two_l3875_387500

def A (a : ℝ) : Set ℝ := {1, 3, a^2}
def B (a : ℝ) : Set ℝ := {1, 2+a}

theorem union_equality_implies_a_equals_two :
  ∀ a : ℝ, A a ∪ B a = A a → a = 2 :=
by sorry

end NUMINAMATH_CALUDE_union_equality_implies_a_equals_two_l3875_387500


namespace NUMINAMATH_CALUDE_sum_of_integers_l3875_387592

theorem sum_of_integers (p q r s t : ℤ) : 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧
  q ≠ r ∧ q ≠ s ∧ q ≠ t ∧
  r ≠ s ∧ r ≠ t ∧
  s ≠ t ∧
  (7 - p) * (7 - q) * (7 - r) * (7 - s) * (7 - t) = -120 →
  p + q + r + s + t = 22 := by
sorry

end NUMINAMATH_CALUDE_sum_of_integers_l3875_387592


namespace NUMINAMATH_CALUDE_twenty_sixth_term_is_79_l3875_387589

/-- An arithmetic sequence with first term 4 and common difference 3 -/
def arithmetic_sequence (n : ℕ) : ℕ :=
  4 + 3 * (n - 1)

/-- The 26th term of the arithmetic sequence is 79 -/
theorem twenty_sixth_term_is_79 : arithmetic_sequence 26 = 79 := by
  sorry

end NUMINAMATH_CALUDE_twenty_sixth_term_is_79_l3875_387589


namespace NUMINAMATH_CALUDE_y_divisibility_l3875_387531

def y : ℕ := 48 + 72 + 144 + 192 + 336 + 384 + 3072

theorem y_divisibility :
  (∃ k : ℕ, y = 4 * k) ∧
  (∃ k : ℕ, y = 8 * k) ∧
  (∃ k : ℕ, y = 24 * k) ∧
  ¬(∃ k : ℕ, y = 16 * k) := by
  sorry

end NUMINAMATH_CALUDE_y_divisibility_l3875_387531


namespace NUMINAMATH_CALUDE_count_integers_satisfying_inequality_l3875_387535

theorem count_integers_satisfying_inequality : 
  ∃ (S : Finset ℕ), 
    (∀ n ∈ S, 2000 < Real.sqrt (n * (n - 1)) ∧ Real.sqrt (n * (n - 1)) < 2005) ∧
    (∀ n : ℕ, 2000 < Real.sqrt (n * (n - 1)) ∧ Real.sqrt (n * (n - 1)) < 2005 → n ∈ S) ∧
    Finset.card S = 5 :=
by sorry

end NUMINAMATH_CALUDE_count_integers_satisfying_inequality_l3875_387535


namespace NUMINAMATH_CALUDE_fraction_subtraction_l3875_387501

theorem fraction_subtraction : (3 + 6 + 9) / (2 + 5 + 7) - (2 + 5 + 7) / (3 + 6 + 9) = 32 / 63 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l3875_387501


namespace NUMINAMATH_CALUDE_sphere_hemisphere_volume_ratio_l3875_387556

theorem sphere_hemisphere_volume_ratio (r : ℝ) (h : r > 0) :
  (4 / 3 * Real.pi * r^3) / (1 / 2 * 4 / 3 * Real.pi * (r / 3)^3) = 54 := by
  sorry

end NUMINAMATH_CALUDE_sphere_hemisphere_volume_ratio_l3875_387556


namespace NUMINAMATH_CALUDE_inequalities_hold_l3875_387513

theorem inequalities_hold (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (b + 1) / (a + 1) > b / a ∧ a + 1 / b > b + 1 / a := by
  sorry

end NUMINAMATH_CALUDE_inequalities_hold_l3875_387513


namespace NUMINAMATH_CALUDE_cole_drive_time_to_work_l3875_387557

/-- Proves that given the conditions of Cole's round trip, it took him 210 minutes to drive to work. -/
theorem cole_drive_time_to_work (speed_to_work : ℝ) (speed_to_home : ℝ) (total_time : ℝ) :
  speed_to_work = 75 →
  speed_to_home = 105 →
  total_time = 6 →
  (total_time * speed_to_work * speed_to_home) / (speed_to_work + speed_to_home) * (60 / speed_to_work) = 210 := by
  sorry

#check cole_drive_time_to_work

end NUMINAMATH_CALUDE_cole_drive_time_to_work_l3875_387557


namespace NUMINAMATH_CALUDE_trig_expression_simplification_l3875_387510

theorem trig_expression_simplification :
  (Real.sin (15 * π / 180) + Real.sin (25 * π / 180) + Real.sin (35 * π / 180) + Real.sin (45 * π / 180) +
   Real.sin (55 * π / 180) + Real.sin (65 * π / 180) + Real.sin (75 * π / 180) + Real.sin (85 * π / 180)) /
  (Real.cos (10 * π / 180) * Real.cos (15 * π / 180) * Real.cos (25 * π / 180)) = 8 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_simplification_l3875_387510


namespace NUMINAMATH_CALUDE_intersection_point_B_coords_l3875_387524

/-- Two circles with centers on the line y = 1 - x, intersecting at points A and B -/
structure IntersectingCircles where
  O₁ : ℝ × ℝ
  O₂ : ℝ × ℝ
  A : ℝ × ℝ
  B : ℝ × ℝ
  centers_on_line : ∀ c, c = O₁ ∨ c = O₂ → c.2 = 1 - c.1
  A_coords : A = (-7, 9)
  intersection : A ≠ B

/-- The theorem stating that point B has coordinates (-8, 8) -/
theorem intersection_point_B_coords (circles : IntersectingCircles) : circles.B = (-8, 8) := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_B_coords_l3875_387524


namespace NUMINAMATH_CALUDE_min_sum_of_even_factors_l3875_387555

theorem min_sum_of_even_factors (a b : ℤ) : 
  Even a → Even b → a * b = 144 → (∀ x y : ℤ, Even x → Even y → x * y = 144 → a + b ≤ x + y) → a + b = -74 :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_even_factors_l3875_387555


namespace NUMINAMATH_CALUDE_will_initial_candy_l3875_387544

/-- The amount of candy Will gave to Haley -/
def candy_given : ℕ := 6

/-- The amount of candy Will had left after giving some to Haley -/
def candy_left : ℕ := 9

/-- The initial amount of candy Will had -/
def initial_candy : ℕ := candy_given + candy_left

theorem will_initial_candy : initial_candy = 15 := by
  sorry

end NUMINAMATH_CALUDE_will_initial_candy_l3875_387544


namespace NUMINAMATH_CALUDE_rectangle_to_square_l3875_387598

theorem rectangle_to_square (k : ℕ) (h1 : k > 5) :
  (∃ n : ℕ, k * (k - 5) = n^2) → k * (k - 5) = 6^2 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_to_square_l3875_387598


namespace NUMINAMATH_CALUDE_final_single_stone_piles_l3875_387540

/-- Represents the state of the game -/
structure GameState where
  piles : List Nat
  deriving Repr

/-- Initial game state -/
def initialState : GameState :=
  { piles := List.range 10 |>.map (· + 1) }

/-- Combines two piles and adds 2 stones -/
def combinePiles (state : GameState) (i j : Nat) : GameState :=
  sorry

/-- Splits a pile into two after removing 2 stones -/
def splitPile (state : GameState) (i : Nat) (split : Nat) : GameState :=
  sorry

/-- Checks if the game has ended -/
def isGameOver (state : GameState) : Bool :=
  sorry

/-- Counts the number of piles with one stone -/
def countSingleStonePiles (state : GameState) : Nat :=
  sorry

/-- The main theorem to prove -/
theorem final_single_stone_piles (finalState : GameState) :
  isGameOver finalState → countSingleStonePiles finalState = 23 := by
  sorry

end NUMINAMATH_CALUDE_final_single_stone_piles_l3875_387540


namespace NUMINAMATH_CALUDE_initial_average_height_l3875_387547

theorem initial_average_height (n : ℕ) (wrong_height actual_height : ℝ) (actual_average : ℝ) :
  n = 35 ∧
  wrong_height = 166 ∧
  actual_height = 106 ∧
  actual_average = 181 →
  (n * actual_average + (wrong_height - actual_height)) / n = 182 + 5 / 7 :=
by sorry

end NUMINAMATH_CALUDE_initial_average_height_l3875_387547


namespace NUMINAMATH_CALUDE_g_has_two_zeros_l3875_387505

noncomputable def f (x : ℝ) : ℝ := (x - Real.sin x) / Real.exp x

noncomputable def g (x : ℝ) : ℝ := f x - 1 / (2 * Real.exp 2)

theorem g_has_two_zeros :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ g x₁ = 0 ∧ g x₂ = 0 ∧
  ∀ (x : ℝ), g x = 0 → x = x₁ ∨ x = x₂ :=
sorry

end NUMINAMATH_CALUDE_g_has_two_zeros_l3875_387505


namespace NUMINAMATH_CALUDE_quadratic_function_property_l3875_387538

/-- Given a quadratic function f(x) = ax² + bx + c, 
    if f(1) - f(-1) = -6, then b = -3 -/
theorem quadratic_function_property (a b c : ℝ) : 
  let f := fun (x : ℝ) => a * x^2 + b * x + c
  (f 1 - f (-1) = -6) → b = -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l3875_387538


namespace NUMINAMATH_CALUDE_maria_travel_fraction_l3875_387536

theorem maria_travel_fraction (total_distance : ℝ) (remaining_distance : ℝ) 
  (first_stop_fraction : ℝ) :
  total_distance = 480 →
  remaining_distance = 180 →
  remaining_distance = (1 - first_stop_fraction) * total_distance * (3/4) →
  first_stop_fraction = 1/2 := by
sorry

end NUMINAMATH_CALUDE_maria_travel_fraction_l3875_387536


namespace NUMINAMATH_CALUDE_angle_measure_l3875_387596

theorem angle_measure (x : Real) : 
  (0.4 * (180 - x) = 90 - x) → x = 30 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_l3875_387596


namespace NUMINAMATH_CALUDE_initial_volume_of_solution_l3875_387530

/-- Given a solution with initial volume V, prove that V = 40 liters -/
theorem initial_volume_of_solution (V : ℝ) : 
  (0.05 * V + 3.5 = 0.11 * (V + 10)) → V = 40 := by sorry

end NUMINAMATH_CALUDE_initial_volume_of_solution_l3875_387530


namespace NUMINAMATH_CALUDE_ursulas_salads_l3875_387539

theorem ursulas_salads :
  ∀ (hot_dog_price salad_price : ℚ)
    (num_hot_dogs : ℕ)
    (initial_money change : ℚ),
  hot_dog_price = 3/2 →
  salad_price = 5/2 →
  num_hot_dogs = 5 →
  initial_money = 20 →
  change = 5 →
  ∃ (num_salads : ℕ),
    num_salads = 3 ∧
    initial_money - change = num_hot_dogs * hot_dog_price + num_salads * salad_price :=
by sorry

end NUMINAMATH_CALUDE_ursulas_salads_l3875_387539


namespace NUMINAMATH_CALUDE_expansion_equality_l3875_387550

theorem expansion_equality (x : ℝ) : (x + 6) * (x - 1) = x^2 + 5*x - 6 := by sorry

end NUMINAMATH_CALUDE_expansion_equality_l3875_387550


namespace NUMINAMATH_CALUDE_tax_rate_65_percent_l3875_387537

/-- Given a tax rate as a percentage, calculate the equivalent dollar amount per $100.00 -/
def tax_rate_to_dollars (percent : ℝ) : ℝ :=
  percent

theorem tax_rate_65_percent : tax_rate_to_dollars 65 = 65 := by
  sorry

end NUMINAMATH_CALUDE_tax_rate_65_percent_l3875_387537


namespace NUMINAMATH_CALUDE_min_value_of_f_l3875_387573

-- Define the function
def f (x : ℝ) : ℝ := 4 * x^2 * (x - 2)

-- State the theorem
theorem min_value_of_f :
  ∃ (m : ℝ), (∀ x ∈ Set.Icc (-2 : ℝ) 2, m ≤ f x) ∧ (∃ x ∈ Set.Icc (-2 : ℝ) 2, f x = m) ∧ m = -64 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3875_387573


namespace NUMINAMATH_CALUDE_sin_pi_minus_alpha_l3875_387554

theorem sin_pi_minus_alpha (α : Real) :
  (∃ (x y : Real), x = -4 ∧ y = 3 ∧ x^2 + y^2 = 25 ∧ 
   (∃ (r : Real), r > 0 ∧ x = r * (Real.cos α) ∧ y = r * (Real.sin α))) →
  Real.sin (π - α) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_pi_minus_alpha_l3875_387554


namespace NUMINAMATH_CALUDE_scientific_notation_of_small_decimal_l3875_387553

theorem scientific_notation_of_small_decimal (x : ℝ) :
  x = 0.000815 →
  ∃ (a : ℝ) (n : ℤ), x = a * 10^n ∧ 1 ≤ a ∧ a < 10 ∧ n = -4 ∧ a = 8.15 :=
by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_small_decimal_l3875_387553


namespace NUMINAMATH_CALUDE_largest_of_five_consecutive_composites_l3875_387552

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n ≤ 99

theorem largest_of_five_consecutive_composites (a b c d e : ℕ) :
  is_two_digit a ∧ is_two_digit b ∧ is_two_digit c ∧ is_two_digit d ∧ is_two_digit e →
  a < 40 ∧ b < 40 ∧ c < 40 ∧ d < 40 ∧ e < 40 →
  b = a + 1 ∧ c = b + 1 ∧ d = c + 1 ∧ e = d + 1 →
  ¬(is_prime a) ∧ ¬(is_prime b) ∧ ¬(is_prime c) ∧ ¬(is_prime d) ∧ ¬(is_prime e) →
  e = 36 :=
sorry

end NUMINAMATH_CALUDE_largest_of_five_consecutive_composites_l3875_387552


namespace NUMINAMATH_CALUDE_solve_for_y_l3875_387579

theorem solve_for_y (x y p : ℝ) (h : p = (5 * x * y) / (x - y)) : 
  y = (p * x) / (5 * x + p) := by
sorry

end NUMINAMATH_CALUDE_solve_for_y_l3875_387579


namespace NUMINAMATH_CALUDE_mode_most_effective_l3875_387570

/-- Represents different statistical measures -/
inductive StatisticalMeasure
  | Variance
  | Mean
  | Median
  | Mode

/-- Represents a shoe model -/
structure ShoeModel where
  id : Nat
  sales : Nat

/-- Represents a shoe store -/
structure ShoeStore where
  models : List ShoeModel
  
/-- Determines the most effective statistical measure for increasing sales -/
def mostEffectiveMeasure (store : ShoeStore) : StatisticalMeasure :=
  StatisticalMeasure.Mode

/-- Theorem: The mode is the most effective statistical measure for increasing sales -/
theorem mode_most_effective (store : ShoeStore) :
  mostEffectiveMeasure store = StatisticalMeasure.Mode :=
by sorry

end NUMINAMATH_CALUDE_mode_most_effective_l3875_387570


namespace NUMINAMATH_CALUDE_product_inequality_l3875_387597

theorem product_inequality (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_eq : a + d = b + c) 
  (abs_ineq : |a - d| < |b - c|) : 
  a * d < b * c := by
sorry

end NUMINAMATH_CALUDE_product_inequality_l3875_387597


namespace NUMINAMATH_CALUDE_expand_product_l3875_387595

theorem expand_product (y : ℝ) (h : y ≠ 0) :
  (3 / 7) * ((7 / y) + 8 * y^2 - 3 * y) = 3 / y + (24 * y^2 - 9 * y) / 7 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l3875_387595


namespace NUMINAMATH_CALUDE_greatest_multiple_of_30_l3875_387527

/-- A function that checks if a list of digits represents a valid arrangement
    according to the problem conditions -/
def is_valid_arrangement (digits : List Nat) : Prop :=
  digits.length = 6 ∧
  digits.toFinset = {1, 3, 4, 6, 8, 9} ∧
  (digits.foldl (fun acc d => acc * 10 + d) 0) % 30 = 0

/-- The claim that 986310 is the greatest possible number satisfying the conditions -/
theorem greatest_multiple_of_30 :
  ∀ (digits : List Nat),
    is_valid_arrangement digits →
    (digits.foldl (fun acc d => acc * 10 + d) 0) ≤ 986310 :=
by sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_30_l3875_387527


namespace NUMINAMATH_CALUDE_book_sales_total_l3875_387533

/-- Calculates the total amount received from book sales given the number of books and their prices -/
def totalAmountReceived (fictionBooks nonFictionBooks childrensBooks : ℕ)
                        (fictionPrice nonFictionPrice childrensPrice : ℚ)
                        (fictionSoldRatio nonFictionSoldRatio childrensSoldRatio : ℚ) : ℚ :=
  (fictionBooks : ℚ) * fictionSoldRatio * fictionPrice +
  (nonFictionBooks : ℚ) * nonFictionSoldRatio * nonFictionPrice +
  (childrensBooks : ℚ) * childrensSoldRatio * childrensPrice

/-- The total amount received from book sales is $799 -/
theorem book_sales_total : 
  totalAmountReceived 60 84 42 5 7 3 (3/4) (5/6) (2/3) = 799 := by
  sorry

end NUMINAMATH_CALUDE_book_sales_total_l3875_387533


namespace NUMINAMATH_CALUDE_five_balls_four_boxes_l3875_387541

/-- The number of ways to distribute distinguishable balls into distinguishable boxes -/
def distribute_balls (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  num_boxes ^ num_balls

/-- Theorem: There are 1024 ways to put 5 distinguishable balls into 4 distinguishable boxes -/
theorem five_balls_four_boxes :
  distribute_balls 5 4 = 1024 := by
  sorry

end NUMINAMATH_CALUDE_five_balls_four_boxes_l3875_387541


namespace NUMINAMATH_CALUDE_apple_orchard_composition_l3875_387502

/-- Represents the composition of an apple orchard -/
structure Orchard where
  total : ℕ
  pure_fuji : ℕ
  pure_gala : ℕ
  cross_pollinated : ℕ

/-- The number of pure Gala trees in an orchard with given conditions -/
def pure_gala_count (o : Orchard) : Prop :=
  o.cross_pollinated = o.total / 10 ∧
  o.pure_fuji = 3 * o.total / 4 ∧
  o.pure_fuji + o.cross_pollinated = 204 ∧
  o.pure_gala = 36

theorem apple_orchard_composition :
  ∃ (o : Orchard), pure_gala_count o :=
sorry

end NUMINAMATH_CALUDE_apple_orchard_composition_l3875_387502


namespace NUMINAMATH_CALUDE_derivative_product_polynomial_l3875_387578

theorem derivative_product_polynomial (x : ℝ) :
  let f : ℝ → ℝ := λ x => (2*x^2 + 3)*(3*x - 1)
  let f' : ℝ → ℝ := λ x => 18*x^2 - 4*x + 9
  HasDerivAt f (f' x) x := by sorry

end NUMINAMATH_CALUDE_derivative_product_polynomial_l3875_387578


namespace NUMINAMATH_CALUDE_probability_of_drawing_balls_l3875_387577

def total_balls : ℕ := 15
def black_balls : ℕ := 10
def white_balls : ℕ := 5
def drawn_balls : ℕ := 5
def drawn_black : ℕ := 3
def drawn_white : ℕ := 2

theorem probability_of_drawing_balls : 
  (Nat.choose black_balls drawn_black * Nat.choose white_balls drawn_white) / 
  Nat.choose total_balls drawn_balls = 400 / 1001 := by
sorry

end NUMINAMATH_CALUDE_probability_of_drawing_balls_l3875_387577


namespace NUMINAMATH_CALUDE_log_equation_holds_l3875_387503

theorem log_equation_holds (x : ℝ) (h1 : x > 0) (h2 : x ≠ 1) :
  (Real.log x / Real.log 2) * (Real.log 7 / Real.log x) = Real.log 49 / Real.log 2 :=
by sorry

end NUMINAMATH_CALUDE_log_equation_holds_l3875_387503


namespace NUMINAMATH_CALUDE_jonessa_pay_l3875_387534

theorem jonessa_pay (tax_rate : ℝ) (take_home_pay : ℝ) (total_pay : ℝ) : 
  tax_rate = 0.1 →
  take_home_pay = 450 →
  take_home_pay = total_pay * (1 - tax_rate) →
  total_pay = 500 := by
sorry

end NUMINAMATH_CALUDE_jonessa_pay_l3875_387534


namespace NUMINAMATH_CALUDE_e_4i_in_third_quadrant_l3875_387569

-- Define the complex exponential function
noncomputable def cexp (z : ℂ) : ℂ := Real.exp z.re * (Real.cos z.im + Complex.I * Real.sin z.im)

-- Define Euler's formula
axiom eulers_formula (x : ℝ) : cexp (Complex.I * x) = Complex.cos x + Complex.I * Complex.sin x

-- Define the third quadrant
def third_quadrant (z : ℂ) : Prop := z.re < 0 ∧ z.im < 0

-- Theorem statement
theorem e_4i_in_third_quadrant :
  third_quadrant (cexp (4 * Complex.I)) :=
sorry

end NUMINAMATH_CALUDE_e_4i_in_third_quadrant_l3875_387569


namespace NUMINAMATH_CALUDE_janet_stuffies_l3875_387565

theorem janet_stuffies (total : ℕ) (kept_fraction : ℚ) (given_fraction : ℚ) : 
  total = 60 →
  kept_fraction = 1/3 →
  given_fraction = 1/4 →
  (total - kept_fraction * total) * given_fraction = 10 := by
sorry

end NUMINAMATH_CALUDE_janet_stuffies_l3875_387565


namespace NUMINAMATH_CALUDE_red_green_peaches_count_l3875_387560

/-- The number of red peaches in the basket -/
def red_peaches : ℕ := 6

/-- The number of green peaches in the basket -/
def green_peaches : ℕ := 16

/-- The total number of red and green peaches in the basket -/
def total_red_green : ℕ := red_peaches + green_peaches

theorem red_green_peaches_count : total_red_green = 22 := by
  sorry

end NUMINAMATH_CALUDE_red_green_peaches_count_l3875_387560


namespace NUMINAMATH_CALUDE_weight_difference_e_d_l3875_387588

/-- Given the weights of individuals A, B, C, D, and E, prove that E weighs 3 kg more than D. -/
theorem weight_difference_e_d (w_a w_b w_c w_d w_e : ℝ) : 
  w_a = 81 →
  (w_a + w_b + w_c) / 3 = 70 →
  (w_a + w_b + w_c + w_d) / 4 = 70 →
  (w_b + w_c + w_d + w_e) / 4 = 68 →
  w_e > w_d →
  w_e - w_d = 3 := by
  sorry

end NUMINAMATH_CALUDE_weight_difference_e_d_l3875_387588


namespace NUMINAMATH_CALUDE_mike_spent_500_on_plants_l3875_387507

/-- The amount Mike spent on plants for himself -/
def mike_spent_on_plants : ℕ → ℕ → ℕ → ℕ → ℕ → ℕ
  | total_rose_bushes, rose_bush_price, friend_rose_bushes, num_aloes, aloe_price =>
    let self_rose_bushes := total_rose_bushes - friend_rose_bushes
    let rose_bush_cost := self_rose_bushes * rose_bush_price
    let aloe_cost := num_aloes * aloe_price
    rose_bush_cost + aloe_cost

theorem mike_spent_500_on_plants :
  mike_spent_on_plants 6 75 2 2 100 = 500 := by
  sorry

end NUMINAMATH_CALUDE_mike_spent_500_on_plants_l3875_387507


namespace NUMINAMATH_CALUDE_polynomial_ascending_powers_l3875_387582

theorem polynomial_ascending_powers (x : ℝ) :
  x^2 - 2 - 5*x^4 + 3*x^3 = -2 + x^2 + 3*x^3 - 5*x^4 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_ascending_powers_l3875_387582
