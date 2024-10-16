import Mathlib

namespace NUMINAMATH_CALUDE_justin_tim_games_l1039_103998

theorem justin_tim_games (n : ℕ) (k : ℕ) (total_players : ℕ) (justin tim : Fin total_players) :
  n = 12 →
  k = 6 →
  total_players = n →
  justin ≠ tim →
  (Nat.choose n k : ℚ) * k / n = 210 :=
sorry

end NUMINAMATH_CALUDE_justin_tim_games_l1039_103998


namespace NUMINAMATH_CALUDE_range_of_c_l1039_103983

-- Define the sets corresponding to propositions p and q
def p (c : ℝ) : Set ℝ := {x | 1 - c < x ∧ x < 1 + c ∧ c > 0}
def q : Set ℝ := {x | (x - 3)^2 < 16}

-- Define the property that p is a sufficient but not necessary condition for q
def sufficient_not_necessary (c : ℝ) : Prop :=
  p c ⊂ q ∧ p c ≠ q

-- State the theorem
theorem range_of_c :
  ∀ c : ℝ, sufficient_not_necessary c ↔ 0 < c ∧ c ≤ 6 := by sorry

end NUMINAMATH_CALUDE_range_of_c_l1039_103983


namespace NUMINAMATH_CALUDE_scale_length_l1039_103929

/-- A scale is divided into equal parts -/
structure Scale :=
  (parts : ℕ)
  (part_length : ℕ)
  (total_length : ℕ)

/-- Convert inches to feet -/
def inches_to_feet (inches : ℕ) : ℚ :=
  inches / 12

/-- Theorem: A scale with 4 parts, each 24 inches long, is 8 feet long -/
theorem scale_length (s : Scale) (h1 : s.parts = 4) (h2 : s.part_length = 24) :
  inches_to_feet s.total_length = 8 := by
  sorry

#check scale_length

end NUMINAMATH_CALUDE_scale_length_l1039_103929


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1039_103918

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℕ, x^3 > x^2) ↔ (∃ x : ℕ, x^3 ≤ x^2) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1039_103918


namespace NUMINAMATH_CALUDE_select_three_from_five_eq_ten_distribute_five_to_three_eq_onefifty_l1039_103901

def select_three_from_five : ℕ := Nat.choose 5 3

def distribute_five_to_three : ℕ :=
  let scenario1 := Nat.choose 5 3 * Nat.factorial 3
  let scenario2 := Nat.choose 5 1 * Nat.choose 4 2 * Nat.factorial 3 / 2
  scenario1 + scenario2

theorem select_three_from_five_eq_ten :
  select_three_from_five = 10 := by sorry

theorem distribute_five_to_three_eq_onefifty :
  distribute_five_to_three = 150 := by sorry

end NUMINAMATH_CALUDE_select_three_from_five_eq_ten_distribute_five_to_three_eq_onefifty_l1039_103901


namespace NUMINAMATH_CALUDE_ratio_a_to_c_l1039_103948

theorem ratio_a_to_c (a b c d : ℚ) 
  (hab : a / b = 5 / 4)
  (hcd : c / d = 2 / 1)
  (hdb : d / b = 1 / 8) :
  a / c = 5 / 1 := by
sorry

end NUMINAMATH_CALUDE_ratio_a_to_c_l1039_103948


namespace NUMINAMATH_CALUDE_laundry_synchronization_l1039_103949

def ronald_cycle : ℕ := 6
def tim_cycle : ℕ := 9
def laura_cycle : ℕ := 12
def dani_cycle : ℕ := 15
def laura_birthday : ℕ := 35

theorem laundry_synchronization (ronald_cycle tim_cycle laura_cycle dani_cycle laura_birthday : ℕ) 
  (h1 : ronald_cycle = 6)
  (h2 : tim_cycle = 9)
  (h3 : laura_cycle = 12)
  (h4 : dani_cycle = 15)
  (h5 : laura_birthday = 35) :
  ∃ (next_sync : ℕ), next_sync - laura_birthday = 145 ∧ 
  next_sync % ronald_cycle = 0 ∧
  next_sync % tim_cycle = 0 ∧
  next_sync % laura_cycle = 0 ∧
  next_sync % dani_cycle = 0 :=
by sorry

end NUMINAMATH_CALUDE_laundry_synchronization_l1039_103949


namespace NUMINAMATH_CALUDE_cos_sin_225_degrees_l1039_103942

theorem cos_sin_225_degrees :
  Real.cos (225 * Real.pi / 180) = -Real.sqrt 2 / 2 ∧
  Real.sin (225 * Real.pi / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_sin_225_degrees_l1039_103942


namespace NUMINAMATH_CALUDE_inseparable_triangles_exist_l1039_103961

-- Define a Point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a Triangle in 3D space
structure Triangle3D where
  a : Point3D
  b : Point3D
  c : Point3D

-- Define a function to check if two triangles can be separated by a plane
def canBeSeparated (t1 t2 : Triangle3D) : Prop :=
  ∃ (a b c d : ℝ), ∀ (p : Point3D),
    (p = t1.a ∨ p = t1.b ∨ p = t1.c) →
      a * p.x + b * p.y + c * p.z + d > 0 ∧
    (p = t2.a ∨ p = t2.b ∨ p = t2.c) →
      a * p.x + b * p.y + c * p.z + d < 0

-- Theorem statement
theorem inseparable_triangles_exist (points : Fin 6 → Point3D) :
  ∃ (t1 t2 : Triangle3D),
    (∀ i j k : Fin 6, i ≠ j ∧ j ≠ k ∧ i ≠ k →
      (t1 = Triangle3D.mk (points i) (points j) (points k) ∨
       t2 = Triangle3D.mk (points i) (points j) (points k))) ∧
    ¬(canBeSeparated t1 t2) :=
sorry

end NUMINAMATH_CALUDE_inseparable_triangles_exist_l1039_103961


namespace NUMINAMATH_CALUDE_problem_solution_l1039_103965

theorem problem_solution (a b c : ℝ) (h1 : a < b) (h2 : b < 0) (h3 : c > 0) :
  (a * c < b * c) ∧ (a + b + c < b + c) ∧ (c / a > 1) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1039_103965


namespace NUMINAMATH_CALUDE_constant_term_expansion_l1039_103932

/-- The constant term in the expansion of (x - 2/x^2)^9 is -672 -/
theorem constant_term_expansion : 
  let f : ℝ → ℝ := fun x ↦ (x - 2 / x^2)^9
  ∃ c : ℝ, ∀ x : ℝ, x ≠ 0 → f x = c + x * (f x - c) / x ∧ c = -672 :=
sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l1039_103932


namespace NUMINAMATH_CALUDE_marble_fraction_after_tripling_l1039_103925

theorem marble_fraction_after_tripling (total : ℚ) (h : total > 0) :
  let blue := (2 / 3) * total
  let red := total - blue
  let new_red := 3 * red
  let new_total := blue + new_red
  new_red / new_total = 3 / 5 := by
sorry

end NUMINAMATH_CALUDE_marble_fraction_after_tripling_l1039_103925


namespace NUMINAMATH_CALUDE_self_square_root_numbers_l1039_103903

theorem self_square_root_numbers : {x : ℝ | x ≥ 0 ∧ x = Real.sqrt x} = {0, 1} := by sorry

end NUMINAMATH_CALUDE_self_square_root_numbers_l1039_103903


namespace NUMINAMATH_CALUDE_symmetric_points_m_value_l1039_103997

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Symmetry about the origin -/
def symmetricAboutOrigin (p q : Point) : Prop :=
  q.x = -p.x ∧ q.y = -p.y

theorem symmetric_points_m_value :
  let p : Point := ⟨2, -1⟩
  let q : Point := ⟨-2, m⟩
  symmetricAboutOrigin p q → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_m_value_l1039_103997


namespace NUMINAMATH_CALUDE_jose_initial_caps_l1039_103913

/-- The number of bottle caps Jose started with -/
def initial_caps : ℕ := sorry

/-- The number of bottle caps Jose received from Rebecca -/
def received_caps : ℕ := 2

/-- The total number of bottle caps Jose ended up with -/
def total_caps : ℕ := 9

/-- Theorem stating that Jose started with 7 bottle caps -/
theorem jose_initial_caps : initial_caps = 7 := by
  sorry

end NUMINAMATH_CALUDE_jose_initial_caps_l1039_103913


namespace NUMINAMATH_CALUDE_binary_11_equals_3_l1039_103962

/-- Converts a binary number represented as a list of bits (least significant bit first) to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : Nat :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 3 -/
def binary_three : List Bool := [true, true]

/-- Theorem stating that the binary number 11 (base 2) is equal to 3 (base 10) -/
theorem binary_11_equals_3 : binary_to_decimal binary_three = 3 := by
  sorry

end NUMINAMATH_CALUDE_binary_11_equals_3_l1039_103962


namespace NUMINAMATH_CALUDE_equation_solution_l1039_103966

theorem equation_solution : ∃ x : ℝ, 45 - (28 - (37 - (15 - x))) = 55 ∧ x = 16 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1039_103966


namespace NUMINAMATH_CALUDE_modified_circle_radius_l1039_103988

/-- Given a circle with radius r, prove that if its modified area and circumference
    sum to 180π, then r satisfies the equation r² + 2r - 90 = 0 -/
theorem modified_circle_radius (r : ℝ) : 
  (2 * Real.pi * r^2) + (4 * Real.pi * r) = 180 * Real.pi → 
  r^2 + 2*r - 90 = 0 := by
  sorry


end NUMINAMATH_CALUDE_modified_circle_radius_l1039_103988


namespace NUMINAMATH_CALUDE_max_triangle_area_l1039_103992

/-- The trajectory of point M -/
def trajectory (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

/-- The line l passing through F₂(1,0) -/
def line_l (m : ℝ) (x y : ℝ) : Prop :=
  x = m * y + 1

/-- The area of triangle F₁AB -/
def triangle_area (y₁ y₂ : ℝ) : ℝ :=
  |y₁ - y₂|

/-- The theorem stating the maximum area of triangle F₁AB -/
theorem max_triangle_area :
  ∃ (max_area : ℝ), max_area = 3 ∧
  ∀ (m : ℝ) (x₁ y₁ x₂ y₂ : ℝ),
    trajectory x₁ y₁ →
    trajectory x₂ y₂ →
    line_l m x₁ y₁ →
    line_l m x₂ y₂ →
    x₁ ≠ x₂ →
    triangle_area y₁ y₂ ≤ max_area :=
by sorry

end NUMINAMATH_CALUDE_max_triangle_area_l1039_103992


namespace NUMINAMATH_CALUDE_role_assignment_combinations_l1039_103978

def number_of_friends : ℕ := 6

theorem role_assignment_combinations (maria_is_cook : Bool) 
  (h1 : maria_is_cook = true) 
  (h2 : number_of_friends = 6) : 
  (Nat.choose (number_of_friends - 1) 1) * (Nat.choose (number_of_friends - 2) 2) = 30 := by
  sorry

end NUMINAMATH_CALUDE_role_assignment_combinations_l1039_103978


namespace NUMINAMATH_CALUDE_consecutive_even_numbers_largest_l1039_103936

theorem consecutive_even_numbers_largest (n : ℕ) : 
  (∀ k : ℕ, k < 7 → ∃ m : ℕ, n + 2*k = 2*m) →  -- 7 consecutive even numbers
  (n + 12 = 3 * n) →                           -- largest is 3 times the smallest
  (n + 12 = 18) :=                             -- largest number is 18
by sorry

end NUMINAMATH_CALUDE_consecutive_even_numbers_largest_l1039_103936


namespace NUMINAMATH_CALUDE_intercept_sum_l1039_103941

/-- A line is described by the equation y + 3 = -3(x - 5) -/
def line_equation (x y : ℝ) : Prop := y + 3 = -3 * (x - 5)

/-- The x-intercept of the line -/
def x_intercept : ℝ := 4

/-- The y-intercept of the line -/
def y_intercept : ℝ := 12

/-- The sum of x-intercept and y-intercept is 16 -/
theorem intercept_sum : x_intercept + y_intercept = 16 := by sorry

end NUMINAMATH_CALUDE_intercept_sum_l1039_103941


namespace NUMINAMATH_CALUDE_sqrt_expression_equality_l1039_103959

theorem sqrt_expression_equality : 
  Real.sqrt 8 - 2 * Real.sqrt 18 + Real.sqrt 24 = -4 * Real.sqrt 2 + 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equality_l1039_103959


namespace NUMINAMATH_CALUDE_orangeade_ratio_l1039_103919

/-- Proves that the ratio of water to orange juice on the first day is 1:1 --/
theorem orangeade_ratio (orange_juice water : ℝ) : 
  orange_juice > 0 → water > 0 →
  (orange_juice + water) * 0.6 = (orange_juice + 2 * water) * 0.4 →
  water / orange_juice = 1 := by
  sorry

end NUMINAMATH_CALUDE_orangeade_ratio_l1039_103919


namespace NUMINAMATH_CALUDE_complex_product_real_imag_parts_l1039_103906

theorem complex_product_real_imag_parts : 
  let Z : ℂ := (1 + Complex.I) * (2 - Complex.I)
  let m : ℝ := Z.re
  let n : ℝ := Z.im
  m * n = 3 := by sorry

end NUMINAMATH_CALUDE_complex_product_real_imag_parts_l1039_103906


namespace NUMINAMATH_CALUDE_bob_journey_distance_l1039_103934

/-- Calculates the total distance traveled given two journey segments -/
def totalDistance (speed1 speed2 time1 time2 : ℝ) : ℝ :=
  speed1 * time1 + speed2 * time2

/-- Proves that Bob's journey results in a total distance of 180 miles -/
theorem bob_journey_distance :
  let speed1 : ℝ := 60
  let speed2 : ℝ := 45
  let time1 : ℝ := 1.5
  let time2 : ℝ := 2
  totalDistance speed1 speed2 time1 time2 = 180 := by
  sorry

#eval totalDistance 60 45 1.5 2

end NUMINAMATH_CALUDE_bob_journey_distance_l1039_103934


namespace NUMINAMATH_CALUDE_fort_men_count_l1039_103956

/-- Represents the initial number of men in the fort -/
def initial_men : ℕ := 150

/-- Represents the number of days the initial provision would last -/
def initial_days : ℕ := 45

/-- Represents the number of days after which some men left -/
def days_before_leaving : ℕ := 10

/-- Represents the number of men who left the fort -/
def men_who_left : ℕ := 25

/-- Represents the number of days the remaining food lasted -/
def remaining_days : ℕ := 42

/-- Theorem stating that given the conditions, the initial number of men in the fort was 150 -/
theorem fort_men_count :
  initial_men * (initial_days - days_before_leaving) = 
  (initial_men - men_who_left) * remaining_days :=
by sorry

end NUMINAMATH_CALUDE_fort_men_count_l1039_103956


namespace NUMINAMATH_CALUDE_photo_shoot_count_l1039_103979

/-- The number of photos taken during a photo shoot, given initial conditions and final count --/
theorem photo_shoot_count (initial : ℕ) (deleted_first : ℕ) (added_first : ℕ)
  (deleted_friend1 : ℕ) (added_friend1 : ℕ)
  (deleted_friend2 : ℕ) (added_friend2 : ℕ)
  (added_friend3 : ℕ)
  (deleted_last : ℕ) (final : ℕ) :
  initial = 63 →
  deleted_first = 7 →
  added_first = 15 →
  deleted_friend1 = 3 →
  added_friend1 = 5 →
  deleted_friend2 = 1 →
  added_friend2 = 4 →
  added_friend3 = 6 →
  deleted_last = 2 →
  final = 112 →
  ∃ x : ℕ, x = 32 ∧
    final = initial - deleted_first + added_first + x - deleted_friend1 + added_friend1 - deleted_friend2 + added_friend2 + added_friend3 - deleted_last :=
by sorry

end NUMINAMATH_CALUDE_photo_shoot_count_l1039_103979


namespace NUMINAMATH_CALUDE_gwens_birthday_money_l1039_103902

/-- The amount of money Gwen received from her mom -/
def money_from_mom : ℕ := sorry

/-- The amount of money Gwen received from her dad -/
def money_from_dad : ℕ := 5

/-- The amount of money Gwen spent -/
def money_spent : ℕ := 4

/-- The difference between money from mom and dad after spending -/
def difference_after_spending : ℕ := 2

theorem gwens_birthday_money : 
  money_from_mom = 6 ∧
  money_from_mom + money_from_dad - money_spent = 
  money_from_dad + difference_after_spending :=
by sorry

end NUMINAMATH_CALUDE_gwens_birthday_money_l1039_103902


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l1039_103951

theorem arithmetic_mean_problem (x : ℝ) : 
  ((x + 8) + 15 + (2 * x) + 13 + (2 * x + 4) + (3 * x + 5)) / 6 = 30 → x = 13.5 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l1039_103951


namespace NUMINAMATH_CALUDE_percentage_less_relation_l1039_103970

/-- Given three real numbers A, B, and C, where A is 35% less than C,
    and B is 10.76923076923077% less than A, prove that B is
    approximately 42% less than C. -/
theorem percentage_less_relation (A B C : ℝ) 
  (h1 : A = 0.65 * C)  -- A is 35% less than C
  (h2 : B = 0.8923076923076923 * A)  -- B is 10.76923076923077% less than A
  : ∃ (ε : ℝ), abs (B - 0.58 * C) < ε ∧ ε < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_percentage_less_relation_l1039_103970


namespace NUMINAMATH_CALUDE_square_root_sum_equality_l1039_103928

theorem square_root_sum_equality : 
  Real.sqrt (16 - 8 * Real.sqrt 3) + 2 * Real.sqrt (16 + 8 * Real.sqrt 3) = 10 + 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_square_root_sum_equality_l1039_103928


namespace NUMINAMATH_CALUDE_min_club_members_l1039_103944

theorem min_club_members : ∃ (n : ℕ), n > 0 ∧ (∃ (m : ℕ), m > 0 ∧ 2/5 < m/n ∧ m/n < 1/2) ∧ 
  (∀ (k : ℕ), 0 < k ∧ k < n → ¬∃ (j : ℕ), j > 0 ∧ 2/5 < j/k ∧ j/k < 1/2) ∧ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_min_club_members_l1039_103944


namespace NUMINAMATH_CALUDE_genuine_purses_and_handbags_l1039_103900

theorem genuine_purses_and_handbags (total_purses : ℕ) (total_handbags : ℕ)
  (h_purses : total_purses = 26)
  (h_handbags : total_handbags = 24)
  (fake_purses : ℕ → ℕ)
  (fake_handbags : ℕ → ℕ)
  (h_fake_purses : fake_purses total_purses = total_purses / 2)
  (h_fake_handbags : fake_handbags total_handbags = total_handbags / 4) :
  total_purses - fake_purses total_purses + total_handbags - fake_handbags total_handbags = 31 := by
  sorry

end NUMINAMATH_CALUDE_genuine_purses_and_handbags_l1039_103900


namespace NUMINAMATH_CALUDE_ellipse_sine_intersections_l1039_103931

/-- An ellipse with center (h, k) and semi-major and semi-minor axes a and b -/
structure Ellipse where
  h : ℝ
  k : ℝ
  a : ℝ
  b : ℝ

/-- The equation of an ellipse -/
def ellipse_eq (e : Ellipse) (x y : ℝ) : Prop :=
  (x - e.h)^2 / e.a^2 + (y - e.k)^2 / e.b^2 = 1

/-- The graph of y = sin x -/
def sine_graph (x y : ℝ) : Prop :=
  y = Real.sin x

/-- A point (x, y) is an intersection point if it satisfies both equations -/
def is_intersection_point (e : Ellipse) (x y : ℝ) : Prop :=
  ellipse_eq e x y ∧ sine_graph x y

/-- The theorem stating that there exists an ellipse with more than 8 intersection points -/
theorem ellipse_sine_intersections :
  ∃ (e : Ellipse), ∃ (points : Finset (ℝ × ℝ)),
    (∀ (p : ℝ × ℝ), p ∈ points → is_intersection_point e p.1 p.2) ∧
    points.card > 8 :=
  sorry

end NUMINAMATH_CALUDE_ellipse_sine_intersections_l1039_103931


namespace NUMINAMATH_CALUDE_pencil_cartons_theorem_l1039_103968

/-- Represents the purchase of school supplies -/
structure SchoolSupplies where
  pencil_boxes_per_carton : ℕ
  pencil_box_cost : ℕ
  marker_cartons : ℕ
  marker_boxes_per_carton : ℕ
  marker_carton_cost : ℕ
  total_spent : ℕ

/-- Calculates the number of pencil cartons bought -/
def pencil_cartons_bought (s : SchoolSupplies) : ℕ :=
  (s.total_spent - s.marker_cartons * s.marker_carton_cost) / (s.pencil_boxes_per_carton * s.pencil_box_cost)

/-- Theorem stating the number of pencil cartons bought -/
theorem pencil_cartons_theorem (s : SchoolSupplies) 
  (h1 : s.pencil_boxes_per_carton = 10)
  (h2 : s.pencil_box_cost = 2)
  (h3 : s.marker_cartons = 10)
  (h4 : s.marker_boxes_per_carton = 5)
  (h5 : s.marker_carton_cost = 4)
  (h6 : s.total_spent = 600) :
  pencil_cartons_bought s = 10 := by
  sorry

end NUMINAMATH_CALUDE_pencil_cartons_theorem_l1039_103968


namespace NUMINAMATH_CALUDE_line_through_point_l1039_103916

theorem line_through_point (k : ℚ) : 
  (2 * k * 3 - 5 = -4 * (-4)) → k = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_l1039_103916


namespace NUMINAMATH_CALUDE_intersection_A_B_l1039_103967

-- Define set A
def A : Set ℝ := {x | x - 1 < 2}

-- Define set B
def B : Set ℝ := {y | ∃ x ∈ A, y = 2^x}

-- Theorem statement
theorem intersection_A_B : A ∩ B = Set.Ioo 0 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l1039_103967


namespace NUMINAMATH_CALUDE_shortest_distance_from_start_l1039_103963

-- Define the walker's movements
def north_distance : ℝ := 15
def west_distance : ℝ := 8
def south_distance : ℝ := 10
def east_distance : ℝ := 1

-- Calculate net distances
def net_north : ℝ := north_distance - south_distance
def net_west : ℝ := west_distance - east_distance

-- Theorem statement
theorem shortest_distance_from_start :
  Real.sqrt (net_north ^ 2 + net_west ^ 2) = Real.sqrt 74 := by
  sorry

#check shortest_distance_from_start

end NUMINAMATH_CALUDE_shortest_distance_from_start_l1039_103963


namespace NUMINAMATH_CALUDE_greatest_distance_between_circle_centers_l1039_103922

/-- The greatest possible distance between the centers of two circles in a rectangle -/
theorem greatest_distance_between_circle_centers
  (rectangle_width : ℝ)
  (rectangle_height : ℝ)
  (circle_diameter : ℝ)
  (h_width : rectangle_width = 15)
  (h_height : rectangle_height = 10)
  (h_diameter : circle_diameter = 5)
  (h_nonneg_width : 0 ≤ rectangle_width)
  (h_nonneg_height : 0 ≤ rectangle_height)
  (h_nonneg_diameter : 0 ≤ circle_diameter)
  (h_fit : circle_diameter ≤ min rectangle_width rectangle_height) :
  ∃ (d : ℝ), d = 5 * Real.sqrt 5 ∧
    ∀ (d' : ℝ), d' ≤ d ∧
      ∃ (x₁ y₁ x₂ y₂ : ℝ),
        0 ≤ x₁ ∧ x₁ ≤ rectangle_width ∧
        0 ≤ y₁ ∧ y₁ ≤ rectangle_height ∧
        0 ≤ x₂ ∧ x₂ ≤ rectangle_width ∧
        0 ≤ y₂ ∧ y₂ ≤ rectangle_height ∧
        circle_diameter / 2 ≤ x₁ ∧ x₁ ≤ rectangle_width - circle_diameter / 2 ∧
        circle_diameter / 2 ≤ y₁ ∧ y₁ ≤ rectangle_height - circle_diameter / 2 ∧
        circle_diameter / 2 ≤ x₂ ∧ x₂ ≤ rectangle_width - circle_diameter / 2 ∧
        circle_diameter / 2 ≤ y₂ ∧ y₂ ≤ rectangle_height - circle_diameter / 2 ∧
        d' = Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) :=
by sorry

end NUMINAMATH_CALUDE_greatest_distance_between_circle_centers_l1039_103922


namespace NUMINAMATH_CALUDE_work_duration_l1039_103940

/-- Given workers A and B with their individual work rates and the time B takes to finish after A leaves,
    prove that A and B worked together for 2 days. -/
theorem work_duration (a_rate b_rate : ℚ) (b_finish_time : ℚ) : 
  a_rate = 1/4 →
  b_rate = 1/10 →
  b_finish_time = 3 →
  ∃ (x : ℚ), x = 2 ∧ (a_rate + b_rate) * x + b_rate * b_finish_time = 1 := by
  sorry

#eval (1/4 : ℚ) + (1/10 : ℚ)  -- Combined work rate
#eval ((1/4 : ℚ) + (1/10 : ℚ)) * 2 + (1/10 : ℚ) * 3  -- Total work done

end NUMINAMATH_CALUDE_work_duration_l1039_103940


namespace NUMINAMATH_CALUDE_solve_textbook_problems_l1039_103933

/-- The number of days it takes to solve all problems -/
def solve_duration (total_problems : ℕ) (problems_left_day3 : ℕ) : ℕ :=
  let problems_solved_day3 := total_problems - problems_left_day3
  let z := problems_solved_day3 / 3
  let daily_problems := List.range 7 |>.map (fun i => z + 1 - i)
  daily_problems.length

/-- Theorem stating that it takes 7 days to solve all problems under given conditions -/
theorem solve_textbook_problems :
  solve_duration 91 46 = 7 := by
  sorry

end NUMINAMATH_CALUDE_solve_textbook_problems_l1039_103933


namespace NUMINAMATH_CALUDE_hockey_league_games_l1039_103977

/-- The number of games played in a hockey league --/
def total_games (n : ℕ) (games_per_pair : ℕ) : ℕ :=
  n * (n - 1) / 2 * games_per_pair

/-- Theorem: In a league with 12 teams, where each team plays 4 games against every other team, 
    the total number of games played is 264. --/
theorem hockey_league_games :
  total_games 12 4 = 264 := by
  sorry

end NUMINAMATH_CALUDE_hockey_league_games_l1039_103977


namespace NUMINAMATH_CALUDE_angle_A_is_60_degrees_perimeter_range_l1039_103909

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given equation
def given_equation (t : Triangle) : Prop :=
  t.a * Real.cos t.C + Real.sqrt 3 * t.a * Real.sin t.C - t.b - t.c = 0

-- Theorem 1: Prove that A = 60°
theorem angle_A_is_60_degrees (t : Triangle) (h : given_equation t) : t.A = Real.pi / 3 := by
  sorry

-- Theorem 2: Prove the range of perimeters when a = 7
theorem perimeter_range (t : Triangle) (h1 : given_equation t) (h2 : t.a = 7) :
  14 < t.a + t.b + t.c ∧ t.a + t.b + t.c ≤ 21 := by
  sorry

end NUMINAMATH_CALUDE_angle_A_is_60_degrees_perimeter_range_l1039_103909


namespace NUMINAMATH_CALUDE_rational_function_simplification_l1039_103971

theorem rational_function_simplification (x : ℝ) (h : x ≠ -1) :
  (x^3 + 4*x^2 + 5*x + 2) / (x + 1) = x^2 + 3*x + 2 := by
  sorry

end NUMINAMATH_CALUDE_rational_function_simplification_l1039_103971


namespace NUMINAMATH_CALUDE_min_value_S_n_l1039_103908

/-- The sum of the first n terms of the sequence -/
def S_n (n : ℕ+) : ℤ := n^2 - 12*n

/-- The minimum value of S_n for positive integers n -/
def min_S_n : ℤ := -36

theorem min_value_S_n :
  ∀ n : ℕ+, S_n n ≥ min_S_n ∧ ∃ m : ℕ+, S_n m = min_S_n :=
sorry

end NUMINAMATH_CALUDE_min_value_S_n_l1039_103908


namespace NUMINAMATH_CALUDE_linear_function_through_two_points_l1039_103981

/-- A linear function passing through two points -/
def LinearFunction (k b : ℝ) : ℝ → ℝ := fun x ↦ k * x + b

theorem linear_function_through_two_points :
  ∀ k b : ℝ,
  LinearFunction k b 3 = 4 →
  LinearFunction k b 4 = 5 →
  ∀ x : ℝ, LinearFunction k b x = x + 1 := by
sorry

end NUMINAMATH_CALUDE_linear_function_through_two_points_l1039_103981


namespace NUMINAMATH_CALUDE_largest_number_l1039_103911

def a : ℚ := 8.23455
def b : ℚ := 8 + 234 / 1000 + 5 / 9000
def c : ℚ := 8 + 23 / 100 + 45 / 9900
def d : ℚ := 8 + 2 / 10 + 345 / 999
def e : ℚ := 8 + 2345 / 9999

theorem largest_number : b > a ∧ b > c ∧ b > d ∧ b > e := by
  sorry

end NUMINAMATH_CALUDE_largest_number_l1039_103911


namespace NUMINAMATH_CALUDE_power_division_equality_l1039_103915

theorem power_division_equality : (3^18 : ℕ) / (27^3 : ℕ) = 19683 := by
  sorry

end NUMINAMATH_CALUDE_power_division_equality_l1039_103915


namespace NUMINAMATH_CALUDE_squats_on_fourth_day_l1039_103986

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def squats_on_day (initial_squats : ℕ) (day : ℕ) : ℕ :=
  match day with
  | 0 => initial_squats
  | n + 1 => squats_on_day initial_squats n + factorial n

theorem squats_on_fourth_day (initial_squats : ℕ) :
  initial_squats = 30 → squats_on_day initial_squats 3 = 39 := by
  sorry

end NUMINAMATH_CALUDE_squats_on_fourth_day_l1039_103986


namespace NUMINAMATH_CALUDE_football_game_attendance_l1039_103990

theorem football_game_attendance (saturday_attendance : ℕ) 
  (expected_total : ℕ) : saturday_attendance = 80 →
  expected_total = 350 →
  (saturday_attendance + 
   (saturday_attendance - 20) + 
   (saturday_attendance - 20 + 50) + 
   (saturday_attendance + (saturday_attendance - 20))) - 
  expected_total = 40 := by
  sorry

end NUMINAMATH_CALUDE_football_game_attendance_l1039_103990


namespace NUMINAMATH_CALUDE_music_movements_duration_l1039_103985

theorem music_movements_duration 
  (a b c : ℝ) 
  (h_order : a ≤ b ∧ b ≤ c) 
  (h_total : a + b + c = 60) 
  (h_max : c ≤ a + b) 
  (h_diff : b - a ≥ 3 ∧ c - b ≥ 3) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) : 
  3 ≤ a ∧ a ≤ 17 := by
sorry

end NUMINAMATH_CALUDE_music_movements_duration_l1039_103985


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1039_103953

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * a * x + 4

-- State the theorem
theorem quadratic_inequality (a x₁ x₂ : ℝ) 
  (ha : 0 < a ∧ a < 3) 
  (hx : x₁ < x₂) 
  (hsum : x₁ + x₂ = 0) : 
  f a x₁ < f a x₂ := by
  sorry


end NUMINAMATH_CALUDE_quadratic_inequality_l1039_103953


namespace NUMINAMATH_CALUDE_pen_notebook_difference_l1039_103957

theorem pen_notebook_difference (notebooks pens : ℕ) : 
  notebooks = 30 →
  notebooks + pens = 110 →
  pens > notebooks →
  pens - notebooks = 50 := by
sorry

end NUMINAMATH_CALUDE_pen_notebook_difference_l1039_103957


namespace NUMINAMATH_CALUDE_fifteenth_even_multiple_of_3_l1039_103955

/-- The nth positive even integer that is a multiple of 3 -/
def evenMultipleOf3 (n : ℕ) : ℕ := 6 * n

/-- The 15th positive even integer that is a multiple of 3 is 90 -/
theorem fifteenth_even_multiple_of_3 : evenMultipleOf3 15 = 90 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_even_multiple_of_3_l1039_103955


namespace NUMINAMATH_CALUDE_coefficient_x_cubed_in_expansion_l1039_103964

theorem coefficient_x_cubed_in_expansion :
  let n : ℕ := 20
  let k : ℕ := 3
  let a : ℤ := 2
  let b : ℤ := -3
  (n.choose k) * a^k * b^(n-k) = -1174898049840 :=
by sorry

end NUMINAMATH_CALUDE_coefficient_x_cubed_in_expansion_l1039_103964


namespace NUMINAMATH_CALUDE_initial_snowflakes_count_l1039_103984

/-- Calculates the initial number of snowflakes given the rate of snowfall and total snowflakes after one hour -/
def initial_snowflakes (rate : ℕ) (interval : ℕ) (total_after_hour : ℕ) : ℕ :=
  total_after_hour - (60 / interval) * rate

/-- Theorem: The initial number of snowflakes is 10 -/
theorem initial_snowflakes_count : initial_snowflakes 4 5 58 = 10 := by
  sorry

end NUMINAMATH_CALUDE_initial_snowflakes_count_l1039_103984


namespace NUMINAMATH_CALUDE_polynomial_equation_solution_l1039_103939

theorem polynomial_equation_solution (p : ℝ → ℝ) :
  (∀ x : ℝ, p (5 * x)^2 - 3 = p (5 * x^2 + 1)) →
  (p = λ _ ↦ (1 + Real.sqrt 13) / 2) ∨ (p = λ _ ↦ (1 - Real.sqrt 13) / 2) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_equation_solution_l1039_103939


namespace NUMINAMATH_CALUDE_triangle_construction_l1039_103950

-- Define the basic structures
structure Point where
  x : ℝ
  y : ℝ

structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define properties
def isAcute (t : Triangle) : Prop := sorry

def onCircumcircle (p : Point) (t : Triangle) : Prop := sorry

def isAltitude (l : Point → Point → Prop) (t : Triangle) : Prop := sorry

def isAngleBisector (l : Point → Point → Prop) (t : Triangle) : Prop := sorry

def intersectsCircumcircle (l : Point → Point → Prop) (t : Triangle) : Point := sorry

-- Main theorem
theorem triangle_construction (t' : Triangle) (h_acute : isAcute t') :
  ∃ t : Triangle,
    (∀ p : Point, onCircumcircle p t' ↔ onCircumcircle p t) ∧
    isAcute t ∧
    (∀ l, isAltitude l t' → onCircumcircle (intersectsCircumcircle l t') t) ∧
    (∀ l, isAngleBisector l t → onCircumcircle (intersectsCircumcircle l t) t) :=
sorry

end NUMINAMATH_CALUDE_triangle_construction_l1039_103950


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1039_103938

theorem sum_of_coefficients (d : ℝ) (a b c : ℤ) : 
  d ≠ 0 → 
  (10 : ℝ) * d + 15 + 16 * d^2 + 4 * d + 3 = (a : ℝ) * d + b + (c : ℝ) * d^2 → 
  a + b + c = 48 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1039_103938


namespace NUMINAMATH_CALUDE_ticket_price_is_three_l1039_103910

/-- Represents an amusement park's weekly operations and revenue -/
structure AmusementPark where
  regularDays : Nat
  regularVisitors : Nat
  specialDay1Visitors : Nat
  specialDay2Visitors : Nat
  weeklyRevenue : Nat

/-- Calculates the ticket price given the park's weekly data -/
def calculateTicketPrice (park : AmusementPark) : Rat :=
  park.weeklyRevenue / (park.regularDays * park.regularVisitors + park.specialDay1Visitors + park.specialDay2Visitors)

/-- Theorem stating that the ticket price is $3 given the specific conditions -/
theorem ticket_price_is_three (park : AmusementPark) 
  (h1 : park.regularDays = 5)
  (h2 : park.regularVisitors = 100)
  (h3 : park.specialDay1Visitors = 200)
  (h4 : park.specialDay2Visitors = 300)
  (h5 : park.weeklyRevenue = 3000) :
  calculateTicketPrice park = 3 := by
  sorry

#eval calculateTicketPrice { 
  regularDays := 5, 
  regularVisitors := 100, 
  specialDay1Visitors := 200, 
  specialDay2Visitors := 300, 
  weeklyRevenue := 3000 
}

end NUMINAMATH_CALUDE_ticket_price_is_three_l1039_103910


namespace NUMINAMATH_CALUDE_cooking_and_weaving_count_l1039_103947

/-- Represents the number of people in various curriculum combinations -/
structure CurriculumParticipation where
  yoga : ℕ
  cooking : ℕ
  weaving : ℕ
  cookingOnly : ℕ
  cookingAndYoga : ℕ
  allCurriculums : ℕ

/-- Theorem stating the number of people studying both cooking and weaving -/
theorem cooking_and_weaving_count (cp : CurriculumParticipation)
  (h1 : cp.yoga = 35)
  (h2 : cp.cooking = 20)
  (h3 : cp.weaving = 15)
  (h4 : cp.cookingOnly = 7)
  (h5 : cp.cookingAndYoga = 5)
  (h6 : cp.allCurriculums = 3) :
  cp.cooking - cp.cookingOnly - cp.cookingAndYoga + cp.allCurriculums = 5 := by
  sorry


end NUMINAMATH_CALUDE_cooking_and_weaving_count_l1039_103947


namespace NUMINAMATH_CALUDE_solution_set_f_geq_6_min_value_f_min_value_a_plus_2b_l1039_103976

-- Define the function f(x)
def f (x : ℝ) : ℝ := |2*x - 1| + |2*x + 3|

-- Theorem for the solution set of f(x) ≥ 6
theorem solution_set_f_geq_6 :
  {x : ℝ | f x ≥ 6} = {x : ℝ | x ≥ 1 ∨ x ≤ -2} :=
sorry

-- Theorem for the minimum value of f(x)
theorem min_value_f :
  ∃ (m : ℝ), m = 4 ∧ ∀ x, f x ≥ m :=
sorry

-- Theorem for the minimum value of a + 2b
theorem min_value_a_plus_2b :
  ∀ (a b : ℝ), a > 0 → b > 0 → 2*a*b + a + 2*b = 4 →
  a + 2*b ≥ 2*Real.sqrt 5 - 2 :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_geq_6_min_value_f_min_value_a_plus_2b_l1039_103976


namespace NUMINAMATH_CALUDE_rectangle_area_perimeter_relation_l1039_103996

theorem rectangle_area_perimeter_relation :
  ∀ (a b : ℕ), 
    a ≠ b →                  -- non-square condition
    a > 0 →                  -- positive dimension
    b > 0 →                  -- positive dimension
    a * b = 2 * (2 * a + 2 * b) →  -- area equals twice perimeter
    2 * (a + b) = 36 :=      -- perimeter is 36
by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_perimeter_relation_l1039_103996


namespace NUMINAMATH_CALUDE_constant_expression_theorem_l1039_103975

theorem constant_expression_theorem (x y m n : ℝ) :
  (∀ y, (x + y) * (x - 2*y) - m*y*(n*x - y) = 25) →
  m = 2 ∧ n = -1/2 ∧ (x = 5 ∨ x = -5) := by sorry

end NUMINAMATH_CALUDE_constant_expression_theorem_l1039_103975


namespace NUMINAMATH_CALUDE_drink_expense_l1039_103994

def initial_amount : ℝ := 9
def final_amount : ℝ := 6
def additional_expense : ℝ := 1.25

theorem drink_expense : 
  initial_amount - final_amount - additional_expense = 1.75 := by
  sorry

end NUMINAMATH_CALUDE_drink_expense_l1039_103994


namespace NUMINAMATH_CALUDE_lose_condition_win_condition_rattle_count_l1039_103987

/-- The number of rattles Twalley has -/
def t : ℕ := 7

/-- The number of rattles Tweerley has -/
def r : ℕ := 5

/-- If Twalley loses the bet, he will have the same number of rattles as Tweerley -/
theorem lose_condition : t - 1 = r + 1 := by sorry

/-- If Twalley wins the bet, he will have twice as many rattles as Tweerley -/
theorem win_condition : t + 1 = 2 * (r - 1) := by sorry

/-- Prove that given the conditions of the bet, Twalley must have 7 rattles and Tweerley must have 5 rattles -/
theorem rattle_count : t = 7 ∧ r = 5 := by sorry

end NUMINAMATH_CALUDE_lose_condition_win_condition_rattle_count_l1039_103987


namespace NUMINAMATH_CALUDE_fraction_to_decimal_decimal_representation_l1039_103989

theorem fraction_to_decimal : (47 : ℚ) / (2^3 * 5^4) = (94 : ℚ) / 10000 := by
  sorry

theorem decimal_representation : (94 : ℚ) / 10000 = 0.0094 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_decimal_representation_l1039_103989


namespace NUMINAMATH_CALUDE_complement_M_in_U_l1039_103999

-- Define the universal set U
def U : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

-- Define the set M
def M : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}

-- Theorem statement
theorem complement_M_in_U : 
  (U \ M) = {x | 1 < x ∧ x ≤ 3} :=
sorry

end NUMINAMATH_CALUDE_complement_M_in_U_l1039_103999


namespace NUMINAMATH_CALUDE_malcolm_primes_l1039_103927

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

theorem malcolm_primes (n : ℕ) :
  n > 0 ∧ is_prime n ∧ is_prime (2 * n - 1) ∧ is_prime (4 * n - 1) → n = 2 ∨ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_malcolm_primes_l1039_103927


namespace NUMINAMATH_CALUDE_scientific_notation_of_3500000_l1039_103945

theorem scientific_notation_of_3500000 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 3500000 = a * (10 : ℝ) ^ n ∧ a = 3.5 ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_3500000_l1039_103945


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l1039_103958

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  (3 * X^5 + 15 * X^4 - 42 * X^3 - 60 * X^2 + 48 * X - 47) = 
  (X^3 + 7 * X^2 + 5 * X - 5) * q + (3 * X - 47) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l1039_103958


namespace NUMINAMATH_CALUDE_min_treasures_is_15_l1039_103974

/-- Represents the number of palm trees with signs -/
def total_trees : ℕ := 30

/-- Represents the number of signs saying "Exactly under 15 signs a treasure is buried." -/
def signs_15 : ℕ := 15

/-- Represents the number of signs saying "Exactly under 8 signs a treasure is buried." -/
def signs_8 : ℕ := 8

/-- Represents the number of signs saying "Exactly under 4 signs a treasure is buried." -/
def signs_4 : ℕ := 4

/-- Represents the number of signs saying "Exactly under 3 signs a treasure is buried." -/
def signs_3 : ℕ := 3

/-- Predicate to check if a sign is truthful given the number of treasures -/
def is_truthful (sign_value : ℕ) (num_treasures : ℕ) : Prop :=
  sign_value ≠ num_treasures

/-- Theorem stating that the minimum number of signs under which treasures can be buried is 15 -/
theorem min_treasures_is_15 :
  ∃ (n : ℕ),
    n = 15 ∧
    (∀ m : ℕ, m < n →
      ¬(is_truthful signs_15 m ∧
        is_truthful signs_8 m ∧
        is_truthful signs_4 m ∧
        is_truthful signs_3 m)) :=
by
  sorry

end NUMINAMATH_CALUDE_min_treasures_is_15_l1039_103974


namespace NUMINAMATH_CALUDE_expected_total_score_l1039_103982

/-- The number of students participating in the contest -/
def num_students : ℕ := 10

/-- The number of shooting opportunities for each student -/
def shots_per_student : ℕ := 2

/-- The probability of scoring a goal -/
def goal_probability : ℝ := 0.6

/-- The scoring system -/
def score (goals : ℕ) : ℝ :=
  match goals with
  | 0 => 0
  | 1 => 5
  | _ => 10

/-- The expected score for a single student -/
def expected_score_per_student : ℝ :=
  (score 0) * (1 - goal_probability)^2 +
  (score 1) * 2 * goal_probability * (1 - goal_probability) +
  (score 2) * goal_probability^2

/-- Theorem: The expected total score for all students is 60 -/
theorem expected_total_score :
  num_students * expected_score_per_student = 60 := by
  sorry

end NUMINAMATH_CALUDE_expected_total_score_l1039_103982


namespace NUMINAMATH_CALUDE_closest_multiple_of_18_to_2021_l1039_103921

def closest_multiple (n m : ℕ) : ℕ := 
  let q := n / m
  if n % m ≤ m / 2 then m * q else m * (q + 1)

theorem closest_multiple_of_18_to_2021 : 
  closest_multiple 2021 18 = 2016 := by
  sorry

end NUMINAMATH_CALUDE_closest_multiple_of_18_to_2021_l1039_103921


namespace NUMINAMATH_CALUDE_complement_A_inter_B_l1039_103924

def U : Set Int := {-1, 0, 1, 2, 3, 4}
def A : Set Int := {1, 2, 3, 4}
def B : Set Int := {0, 2}

theorem complement_A_inter_B : (U \ A) ∩ B = {0} := by sorry

end NUMINAMATH_CALUDE_complement_A_inter_B_l1039_103924


namespace NUMINAMATH_CALUDE_chris_age_l1039_103917

theorem chris_age (a b c : ℕ) : 
  (a + b + c) / 3 = 9 →  -- The average of their ages is 9
  c - 4 = a →            -- Four years ago, Chris was Amy's current age
  b + 3 = 2 * (a + 3) / 3 →  -- In 3 years, Ben's age will be 2/3 of Amy's age
  c = 13 :=               -- Chris's current age is 13
by sorry

end NUMINAMATH_CALUDE_chris_age_l1039_103917


namespace NUMINAMATH_CALUDE_school_pupils_l1039_103912

theorem school_pupils (girls : ℕ) (boys : ℕ) (h1 : girls = 692) (h2 : girls = boys + 458) :
  girls + boys = 926 := by
  sorry

end NUMINAMATH_CALUDE_school_pupils_l1039_103912


namespace NUMINAMATH_CALUDE_infinitely_many_primes_divide_fib_l1039_103904

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

-- State the theorem
theorem infinitely_many_primes_divide_fib : 
  ∃ (S : Set ℕ), (Set.Infinite S) ∧ (∀ p ∈ S, Prime p ∧ (fib (p - 1) % p = 0)) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_primes_divide_fib_l1039_103904


namespace NUMINAMATH_CALUDE_problems_left_to_grade_l1039_103914

/-- Calculates the number of problems left to grade for a teacher grading worksheets from three subjects. -/
theorem problems_left_to_grade
  (math_problems_per_sheet : ℕ)
  (science_problems_per_sheet : ℕ)
  (english_problems_per_sheet : ℕ)
  (total_math_sheets : ℕ)
  (total_science_sheets : ℕ)
  (total_english_sheets : ℕ)
  (graded_math_sheets : ℕ)
  (graded_science_sheets : ℕ)
  (graded_english_sheets : ℕ)
  (h_math : math_problems_per_sheet = 5)
  (h_science : science_problems_per_sheet = 3)
  (h_english : english_problems_per_sheet = 7)
  (h_total_math : total_math_sheets = 10)
  (h_total_science : total_science_sheets = 15)
  (h_total_english : total_english_sheets = 12)
  (h_graded_math : graded_math_sheets = 6)
  (h_graded_science : graded_science_sheets = 10)
  (h_graded_english : graded_english_sheets = 5) :
  (total_math_sheets * math_problems_per_sheet - graded_math_sheets * math_problems_per_sheet) +
  (total_science_sheets * science_problems_per_sheet - graded_science_sheets * science_problems_per_sheet) +
  (total_english_sheets * english_problems_per_sheet - graded_english_sheets * english_problems_per_sheet) = 84 :=
by sorry

end NUMINAMATH_CALUDE_problems_left_to_grade_l1039_103914


namespace NUMINAMATH_CALUDE_min_a3_and_a2b2_l1039_103980

theorem min_a3_and_a2b2 (a₁ a₂ a₃ b₁ b₂ b₃ : ℝ) 
  (h_pos : a₁ > 0 ∧ a₂ > 0 ∧ a₃ > 0 ∧ b₁ > 0 ∧ b₂ > 0 ∧ b₃ > 0)
  (h_arith : a₂ = a₁ + b₁ ∧ a₃ = a₁ + 2*b₁)
  (h_geom : b₂ = b₁ * a₁ ∧ b₃ = b₁ * a₁^2)
  (h_equal : a₃ = b₃) :
  (∀ a₁' a₂' a₃' b₁' b₂' b₃', 
    a₁' > 0 ∧ a₂' > 0 ∧ a₃' > 0 ∧ b₁' > 0 ∧ b₂' > 0 ∧ b₃' > 0 →
    a₂' = a₁' + b₁' ∧ a₃' = a₁' + 2*b₁' →
    b₂' = b₁' * a₁' ∧ b₃' = b₁' * a₁'^2 →
    a₃' = b₃' →
    a₃' ≥ 3 * Real.sqrt 6 / 2) ∧
  (a₃ = 3 * Real.sqrt 6 / 2 → a₂ * b₂ = 15 * Real.sqrt 6 / 8) :=
by sorry

end NUMINAMATH_CALUDE_min_a3_and_a2b2_l1039_103980


namespace NUMINAMATH_CALUDE_intersection_volume_zero_l1039_103946

/-- The region defined by |x| + |y| + |z| ≤ 2 -/
def Region1 (x y z : ℝ) : Prop :=
  abs x + abs y + abs z ≤ 2

/-- The region defined by |x| + |y| + |z-2| ≤ 1 -/
def Region2 (x y z : ℝ) : Prop :=
  abs x + abs y + abs (z - 2) ≤ 1

/-- The volume of a region in ℝ³ -/
noncomputable def volume (S : Set (ℝ × ℝ × ℝ)) : ℝ := sorry

/-- The intersection of Region1 and Region2 -/
def IntersectionRegion : Set (ℝ × ℝ × ℝ) :=
  {p | Region1 p.1 p.2.1 p.2.2 ∧ Region2 p.1 p.2.1 p.2.2}

theorem intersection_volume_zero :
  volume IntersectionRegion = 0 := by sorry

end NUMINAMATH_CALUDE_intersection_volume_zero_l1039_103946


namespace NUMINAMATH_CALUDE_quadratic_root_zero_l1039_103954

/-- Given a quadratic equation (a+3)x^2 - 4x + a^2 - 9 = 0 with 0 as a root and a + 3 ≠ 0, prove that a = 3 -/
theorem quadratic_root_zero (a : ℝ) : 
  ((a + 3) * 0^2 - 4 * 0 + a^2 - 9 = 0) → 
  (a + 3 ≠ 0) → 
  (a = 3) := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_zero_l1039_103954


namespace NUMINAMATH_CALUDE_high_five_problem_l1039_103993

theorem high_five_problem (n : ℕ) (h : n > 0) :
  (∀ (person : Fin n), (person.val < n → 2 * 2021 = n - 1)) →
  (n = 4043 ∧ Nat.choose n 3 = 11024538580) := by
  sorry

end NUMINAMATH_CALUDE_high_five_problem_l1039_103993


namespace NUMINAMATH_CALUDE_inequality_proof_l1039_103920

theorem inequality_proof (x y : ℝ) (hx : x > 2) (hy : y > 2) :
  (x^2 - x) / (y^2 + y) + (y^2 - y) / (x^2 + x) > 2/3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1039_103920


namespace NUMINAMATH_CALUDE_inverse_sum_mod_31_l1039_103991

theorem inverse_sum_mod_31 :
  ∃ (a b : ℤ), a ≡ 25 [ZMOD 31] ∧ b ≡ 5 [ZMOD 31] ∧ (a + b) ≡ 30 [ZMOD 31] := by
  sorry

end NUMINAMATH_CALUDE_inverse_sum_mod_31_l1039_103991


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l1039_103907

theorem pure_imaginary_complex_number (m : ℝ) : 
  (m^2 - 3*m : ℂ) + (m^2 - 5*m + 6 : ℂ)*Complex.I = Complex.I * ((m^2 - 5*m + 6 : ℂ)) → m = 0 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l1039_103907


namespace NUMINAMATH_CALUDE_nonzero_sum_zero_power_equality_l1039_103969

theorem nonzero_sum_zero_power_equality (a b c : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (sum_zero : a + b + c = 0)
  (power_equality : a^4 + b^4 + c^4 = a^6 + b^6 + c^6) :
  a^2 + b^2 + c^2 = 3/2 := by
sorry

end NUMINAMATH_CALUDE_nonzero_sum_zero_power_equality_l1039_103969


namespace NUMINAMATH_CALUDE_paula_candy_problem_l1039_103995

theorem paula_candy_problem (initial_candies : ℕ) (num_friends : ℕ) (candies_per_friend : ℕ)
  (h1 : initial_candies = 20)
  (h2 : num_friends = 6)
  (h3 : candies_per_friend = 4) :
  num_friends * candies_per_friend - initial_candies = 4 := by
  sorry

end NUMINAMATH_CALUDE_paula_candy_problem_l1039_103995


namespace NUMINAMATH_CALUDE_prove_earnings_l1039_103905

/-- Gondor's earnings from repairing devices -/
def earnings_problem : Prop :=
  let phone_repair_fee : ℕ := 10
  let laptop_repair_fee : ℕ := 20
  let monday_phones : ℕ := 3
  let tuesday_phones : ℕ := 5
  let wednesday_laptops : ℕ := 2
  let thursday_laptops : ℕ := 4
  let total_earnings : ℕ := 
    phone_repair_fee * (monday_phones + tuesday_phones) +
    laptop_repair_fee * (wednesday_laptops + thursday_laptops)
  total_earnings = 200

theorem prove_earnings : earnings_problem := by
  sorry

end NUMINAMATH_CALUDE_prove_earnings_l1039_103905


namespace NUMINAMATH_CALUDE_code_number_correspondence_exists_l1039_103973

-- Define the set of codes
def Codes : Type := Fin 5 → Fin 3 → Char

-- Define the set of numbers
def Numbers : Type := Fin 5 → Nat

-- Define the given codes
def given_codes : Codes := λ i j ↦ 
  match i, j with
  | 0, 0 => 'R' | 0, 1 => 'W' | 0, 2 => 'Q'
  | 1, 0 => 'S' | 1, 1 => 'X' | 1, 2 => 'W'
  | 2, 0 => 'P' | 2, 1 => 'S' | 2, 2 => 'T'
  | 3, 0 => 'X' | 3, 1 => 'N' | 3, 2 => 'Y'
  | 4, 0 => 'N' | 4, 1 => 'X' | 4, 2 => 'Y'
  | _, _ => 'A' -- Default case, should never be reached

-- Define the given and solution numbers
def given_and_solution_numbers : Numbers := λ i ↦
  match i with
  | 0 => 286
  | 1 => 540
  | 2 => 793
  | 3 => 948
  | 4 => 450

-- Define a bijection type between Codes and Numbers
def CodeNumberBijection := {f : Codes → Numbers // Function.Bijective f}

theorem code_number_correspondence_exists : ∃ (f : CodeNumberBijection), 
  ∀ (i : Fin 5), f.val given_codes i = given_and_solution_numbers i :=
sorry

end NUMINAMATH_CALUDE_code_number_correspondence_exists_l1039_103973


namespace NUMINAMATH_CALUDE_sector_max_area_l1039_103923

/-- Given a sector with circumference 8, its area is at most 4 -/
theorem sector_max_area :
  ∀ (r l : ℝ), r > 0 → l > 0 → 2 * r + l = 8 →
  (1 / 2) * l * r ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_sector_max_area_l1039_103923


namespace NUMINAMATH_CALUDE_investment_average_rate_l1039_103943

/-- Proves that for a $6000 investment split between 3% and 5.5% interest rates
    with equal annual returns, the average interest rate is 3.88% -/
theorem investment_average_rate (total : ℝ) (rate1 rate2 : ℝ) :
  total = 6000 →
  rate1 = 0.03 →
  rate2 = 0.055 →
  ∃ (x : ℝ), 
    x > 0 ∧ 
    x < total ∧
    rate1 * (total - x) = rate2 * x →
  (rate1 * (total - x) + rate2 * x) / total = 0.0388 :=
by sorry


end NUMINAMATH_CALUDE_investment_average_rate_l1039_103943


namespace NUMINAMATH_CALUDE_hexadecagon_area_theorem_l1039_103960

/-- A hexadecagon inscribed in a square with specific properties -/
structure InscribedHexadecagon where
  /-- The perimeter of the square in which the hexadecagon is inscribed -/
  square_perimeter : ℝ
  /-- The property that every side of the square is trisected twice equally -/
  trisected_twice : Prop

/-- The area of the inscribed hexadecagon -/
def hexadecagon_area (h : InscribedHexadecagon) : ℝ := sorry

/-- Theorem stating the area of the inscribed hexadecagon with given properties -/
theorem hexadecagon_area_theorem (h : InscribedHexadecagon) 
  (h_perimeter : h.square_perimeter = 160) : hexadecagon_area h = 1344 := by sorry

end NUMINAMATH_CALUDE_hexadecagon_area_theorem_l1039_103960


namespace NUMINAMATH_CALUDE_dog_food_per_dog_l1039_103972

/-- The amount of dog food two dogs eat together per day -/
def total_food : ℝ := 0.25

/-- The number of dogs -/
def num_dogs : ℕ := 2

theorem dog_food_per_dog :
  ∀ (food_per_dog : ℝ),
  (food_per_dog * num_dogs = total_food) →
  (food_per_dog = 0.125) := by
sorry

end NUMINAMATH_CALUDE_dog_food_per_dog_l1039_103972


namespace NUMINAMATH_CALUDE_adult_tickets_bought_l1039_103952

/-- Proves the number of adult tickets bought given ticket prices and total information -/
theorem adult_tickets_bought (adult_price child_price : ℚ) (total_tickets : ℕ) (total_cost : ℚ) 
  (h1 : adult_price = 5.5)
  (h2 : child_price = 3.5)
  (h3 : total_tickets = 21)
  (h4 : total_cost = 83.5) :
  ∃ (adult_tickets : ℕ), 
    adult_tickets * adult_price + (total_tickets - adult_tickets) * child_price = total_cost ∧ 
    adult_tickets = 5 := by
  sorry

end NUMINAMATH_CALUDE_adult_tickets_bought_l1039_103952


namespace NUMINAMATH_CALUDE_first_phase_revenue_calculation_l1039_103930

/-- Represents a two-phase sales scenario -/
structure SalesScenario where
  total_purchase : ℝ
  first_markup : ℝ
  second_markup : ℝ
  total_revenue_increase : ℝ

/-- Calculates the revenue from the first phase of sales -/
def first_phase_revenue (s : SalesScenario) : ℝ :=
  sorry

/-- Theorem stating the first phase revenue for the given scenario -/
theorem first_phase_revenue_calculation (s : SalesScenario) 
  (h1 : s.total_purchase = 180000)
  (h2 : s.first_markup = 0.25)
  (h3 : s.second_markup = 0.16)
  (h4 : s.total_revenue_increase = 0.20) :
  first_phase_revenue s = 100000 :=
sorry

end NUMINAMATH_CALUDE_first_phase_revenue_calculation_l1039_103930


namespace NUMINAMATH_CALUDE_dessert_preference_l1039_103926

theorem dessert_preference (total : ℕ) (apple : ℕ) (chocolate : ℕ) (neither : ℕ) :
  total = 40 →
  apple = 18 →
  chocolate = 15 →
  neither = 12 →
  ∃ (both : ℕ), both = 5 ∧ total = apple + chocolate - both + neither :=
by
  sorry

end NUMINAMATH_CALUDE_dessert_preference_l1039_103926


namespace NUMINAMATH_CALUDE_ball_box_theorem_l1039_103935

/-- Represents the state of boxes after a number of steps -/
def BoxState := List Nat

/-- Converts a natural number to its septenary (base 7) representation -/
def toSeptenary (n : Nat) : List Nat :=
  sorry

/-- Simulates the ball-placing process for a given number of steps -/
def simulateSteps (steps : Nat) : BoxState :=
  sorry

/-- Counts the number of non-zero elements in a list -/
def countNonZero (l : List Nat) : Nat :=
  sorry

/-- Sums all elements in a list -/
def sumList (l : List Nat) : Nat :=
  sorry

theorem ball_box_theorem (steps : Nat := 3456) :
  let septenaryRep := toSeptenary steps
  let finalState := simulateSteps steps
  countNonZero finalState = countNonZero septenaryRep ∧
  sumList finalState = sumList septenaryRep :=
by sorry

end NUMINAMATH_CALUDE_ball_box_theorem_l1039_103935


namespace NUMINAMATH_CALUDE_car_count_l1039_103937

/-- The total number of cars in a rectangular arrangement -/
def total_cars (front_to_back : ℕ) (left_to_right : ℕ) : ℕ :=
  front_to_back * left_to_right

/-- Theorem stating the total number of cars given the position of red cars -/
theorem car_count (red_from_front red_from_left red_from_back red_from_right : ℕ) 
    (h1 : red_from_front + red_from_back = 25)
    (h2 : red_from_left + red_from_right = 35) :
    total_cars (red_from_front + red_from_back - 1) (red_from_left + red_from_right - 1) = 816 := by
  sorry

#eval total_cars 24 34  -- Should output 816

end NUMINAMATH_CALUDE_car_count_l1039_103937
