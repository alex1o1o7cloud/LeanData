import Mathlib

namespace NUMINAMATH_CALUDE_triangle_base_height_proof_l3239_323988

theorem triangle_base_height_proof :
  ∀ (base height : ℝ),
    base = height - 4 →
    (1/2) * base * height = 96 →
    base = 12 ∧ height = 16 := by
  sorry

end NUMINAMATH_CALUDE_triangle_base_height_proof_l3239_323988


namespace NUMINAMATH_CALUDE_ratio_problem_l3239_323933

theorem ratio_problem (a b : ℤ) : 
  (a : ℚ) / b = 1 / 4 → 
  (a + 6 : ℚ) / b = 1 / 2 → 
  b = 24 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l3239_323933


namespace NUMINAMATH_CALUDE_inequality_proof_l3239_323942

theorem inequality_proof (a b c d : ℝ) 
  (nonneg_a : 0 ≤ a) (nonneg_b : 0 ≤ b) (nonneg_c : 0 ≤ c) (nonneg_d : 0 ≤ d)
  (sum_one : a + b + c + d = 1) :
  a * b * c + b * c * d + c * a * d + d * a * b ≤ 1 / 27 + (176 / 27) * a * b * c * d :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3239_323942


namespace NUMINAMATH_CALUDE_unique_solution_linear_system_l3239_323984

theorem unique_solution_linear_system
  (a b c d : ℝ)
  (h : a * d - c * b ≠ 0) :
  ∀ x y : ℝ, a * x + b * y = 0 ∧ c * x + d * y = 0 → x = 0 ∧ y = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_linear_system_l3239_323984


namespace NUMINAMATH_CALUDE_non_monotonic_implies_a_gt_two_thirds_l3239_323923

/-- A function f is non-monotonic on an interval (a, b) if there exist
    x₁, x₂, x₃ in (a, b) such that x₁ < x₂ < x₃ and either
    f(x₁) < f(x₂) and f(x₂) > f(x₃), or f(x₁) > f(x₂) and f(x₂) < f(x₃) -/
def NonMonotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x₁ x₂ x₃, a < x₁ ∧ x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < b ∧
  ((f x₁ < f x₂ ∧ f x₂ > f x₃) ∨ (f x₁ > f x₂ ∧ f x₂ < f x₃))

theorem non_monotonic_implies_a_gt_two_thirds :
  ∀ a : ℝ, a > 0 →
  NonMonotonic (fun x ↦ (1/3) * a * x^3 - x^2) 0 3 →
  a > 2/3 := by
  sorry

end NUMINAMATH_CALUDE_non_monotonic_implies_a_gt_two_thirds_l3239_323923


namespace NUMINAMATH_CALUDE_train_length_l3239_323972

/-- The length of a train given its speed and time to cross an overbridge -/
theorem train_length (speed : ℝ) (time : ℝ) (bridge_length : ℝ) :
  speed = 36 * 1000 / 3600 →
  time = 70 →
  bridge_length = 100 →
  speed * time - bridge_length = 600 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l3239_323972


namespace NUMINAMATH_CALUDE_ada_original_seat_l3239_323915

-- Define the type for seats
inductive Seat
| one | two | three | four | five | six

-- Define the type for friends
inductive Friend
| ada | bea | ceci | dee | edie | fana

-- Define the initial seating arrangement
def initial_seating : Friend → Seat := sorry

-- Define the movement function
def move (s : Seat) (n : Int) : Seat := sorry

-- Define the final seating arrangement after movements
def final_seating : Friend → Seat := sorry

-- Theorem to prove
theorem ada_original_seat :
  (∀ f : Friend, f ≠ Friend.ada → final_seating f ≠ initial_seating f) →
  (final_seating Friend.ada = Seat.one ∨ final_seating Friend.ada = Seat.six) →
  initial_seating Friend.ada = Seat.two :=
sorry

end NUMINAMATH_CALUDE_ada_original_seat_l3239_323915


namespace NUMINAMATH_CALUDE_arrangements_count_l3239_323971

/-- The number of arrangements of 3 boys and 2 girls in a row with girls at both ends -/
def arrangements_with_girls_at_ends : ℕ :=
  let num_boys : ℕ := 3
  let num_girls : ℕ := 2
  let girl_arrangements : ℕ := 2  -- A_2^2
  let boy_arrangements : ℕ := 6  -- A_3^3
  girl_arrangements * boy_arrangements

/-- Theorem stating that the number of arrangements is 12 -/
theorem arrangements_count : arrangements_with_girls_at_ends = 12 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_count_l3239_323971


namespace NUMINAMATH_CALUDE_distribute_6_3_max2_l3239_323928

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute (n : ℕ) (k : ℕ) (max_per_box : ℕ) : ℕ := sorry

/-- The number of ways to distribute 6 distinguishable balls into 3 indistinguishable boxes
    with at most 2 balls per box -/
theorem distribute_6_3_max2 : distribute 6 3 2 = 100 := by sorry

end NUMINAMATH_CALUDE_distribute_6_3_max2_l3239_323928


namespace NUMINAMATH_CALUDE_focus_to_asymptote_distance_l3239_323924

/-- Represents a hyperbola with equation x²/(3a) - y²/a = 1 where a > 0 -/
structure Hyperbola (a : ℝ) where
  a_pos : a > 0

/-- A focus of the hyperbola -/
def focus (h : Hyperbola a) : ℝ × ℝ := sorry

/-- An asymptote of the hyperbola -/
def asymptote (h : Hyperbola a) : ℝ → ℝ := sorry

/-- The distance from a point to a line -/
def distance_point_to_line (p : ℝ × ℝ) (f : ℝ → ℝ) : ℝ := sorry

/-- Theorem: The distance from a focus to an asymptote of the hyperbola is √a -/
theorem focus_to_asymptote_distance (h : Hyperbola a) : 
  distance_point_to_line (focus h) (asymptote h) = Real.sqrt a := by sorry

end NUMINAMATH_CALUDE_focus_to_asymptote_distance_l3239_323924


namespace NUMINAMATH_CALUDE_smallest_stairs_fifty_three_satisfies_stairs_solution_l3239_323947

theorem smallest_stairs (n : ℕ) : 
  (n > 20 ∧ n % 6 = 5 ∧ n % 7 = 4) → n ≥ 53 :=
by sorry

theorem fifty_three_satisfies : 
  53 > 20 ∧ 53 % 6 = 5 ∧ 53 % 7 = 4 :=
by sorry

theorem stairs_solution : 
  ∃ (n : ℕ), n = 53 ∧ n > 20 ∧ n % 6 = 5 ∧ n % 7 = 4 ∧
  ∀ (m : ℕ), (m > 20 ∧ m % 6 = 5 ∧ m % 7 = 4) → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_stairs_fifty_three_satisfies_stairs_solution_l3239_323947


namespace NUMINAMATH_CALUDE_probability_of_identical_cubes_l3239_323981

/-- Represents the three possible colors for painting cube faces -/
inductive Color
  | Red
  | Blue
  | Green

/-- Represents a cube with six colored faces -/
def Cube := Fin 6 → Color

/-- The number of ways to paint a single cube -/
def waysToColorCube : ℕ := 729

/-- The total number of ways to paint three cubes -/
def totalWaysToPaintThreeCubes : ℕ := 387420489

/-- The number of ways to paint three cubes so they are rotationally identical -/
def waysToColorIdenticalCubes : ℕ := 633

/-- Checks if two cubes are rotationally identical -/
def areRotationallyIdentical (c1 c2 : Cube) : Prop := sorry

/-- The probability of three independently painted cubes being rotationally identical -/
theorem probability_of_identical_cubes :
  (waysToColorIdenticalCubes : ℚ) / totalWaysToPaintThreeCubes = 211 / 129140163 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_identical_cubes_l3239_323981


namespace NUMINAMATH_CALUDE_complex_arithmetic_calculation_l3239_323910

theorem complex_arithmetic_calculation : 
  3 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 2400 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_calculation_l3239_323910


namespace NUMINAMATH_CALUDE_cube_root_eight_times_sixth_root_sixtyfour_equals_four_l3239_323905

theorem cube_root_eight_times_sixth_root_sixtyfour_equals_four :
  (8 : ℝ) ^ (1/3) * (64 : ℝ) ^ (1/6) = 4 := by sorry

end NUMINAMATH_CALUDE_cube_root_eight_times_sixth_root_sixtyfour_equals_four_l3239_323905


namespace NUMINAMATH_CALUDE_hari_contribution_is_9000_l3239_323912

/-- Represents the business partnership between Praveen and Hari -/
structure Partnership where
  praveen_investment : ℕ
  praveen_months : ℕ
  hari_months : ℕ
  profit_ratio_praveen : ℕ
  profit_ratio_hari : ℕ

/-- Calculates Hari's contribution to the capital -/
def hari_contribution (p : Partnership) : ℕ :=
  (p.praveen_investment * p.praveen_months * p.profit_ratio_hari) / (p.hari_months * p.profit_ratio_praveen)

/-- Theorem stating that Hari's contribution is 9000 given the specified conditions -/
theorem hari_contribution_is_9000 :
  let p : Partnership := {
    praveen_investment := 3500,
    praveen_months := 12,
    hari_months := 7,
    profit_ratio_praveen := 2,
    profit_ratio_hari := 3
  }
  hari_contribution p = 9000 := by sorry

end NUMINAMATH_CALUDE_hari_contribution_is_9000_l3239_323912


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3239_323970

theorem min_value_sum_reciprocals (m n : ℝ) 
  (h1 : 2 * m + n = 2) 
  (h2 : m > 0) 
  (h3 : n > 0) : 
  (∀ x y : ℝ, x > 0 → y > 0 → 2 * x + y = 2 → 1 / m + 2 / n ≤ 1 / x + 2 / y) ∧ 
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 2 * x + y = 2 ∧ 1 / x + 2 / y = 4) := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3239_323970


namespace NUMINAMATH_CALUDE_restaurant_bill_split_l3239_323993

def bill : ℚ := 314.16
def payment_per_person : ℚ := 34.91
def total_payment : ℚ := 314.19

theorem restaurant_bill_split :
  ∃ (n : ℕ), n > 0 ∧ 
  (n : ℚ) * payment_per_person ≥ bill ∧
  (n : ℚ) * payment_per_person < bill + 1 ∧
  n * payment_per_person = total_payment ∧
  n = 8 := by sorry

end NUMINAMATH_CALUDE_restaurant_bill_split_l3239_323993


namespace NUMINAMATH_CALUDE_walking_problem_l3239_323954

/-- Proves that given the conditions of the walking problem, the speed of the second man is 3 km/h -/
theorem walking_problem (distance : ℝ) (speed_first : ℝ) (time_diff : ℝ) :
  distance = 6 →
  speed_first = 4 →
  time_diff = 0.5 →
  let time_first := distance / speed_first
  let time_second := time_first + time_diff
  let speed_second := distance / time_second
  speed_second = 3 := by
  sorry

end NUMINAMATH_CALUDE_walking_problem_l3239_323954


namespace NUMINAMATH_CALUDE_watch_price_equation_l3239_323936

/-- The original cost price of a watch satisfies the equation relating its discounted price and taxed price with profit. -/
theorem watch_price_equation (C : ℝ) : C > 0 → 0.855 * C + 540 = 1.2096 * C := by
  sorry

end NUMINAMATH_CALUDE_watch_price_equation_l3239_323936


namespace NUMINAMATH_CALUDE_unique_k_value_l3239_323961

-- Define the quadratic equation
def quadratic (k : ℝ) (x : ℝ) : ℝ := 3 * x^2 - (k + 2) * x + 6

-- Define the condition for real roots
def has_real_roots (k : ℝ) : Prop :=
  (k + 2)^2 - 4 * 3 * 6 ≥ 0

-- Define the condition that 3 is a root
def three_is_root (k : ℝ) : Prop :=
  quadratic k 3 = 0

-- The main theorem
theorem unique_k_value :
  ∃! k : ℝ, has_real_roots k ∧ three_is_root k :=
sorry

end NUMINAMATH_CALUDE_unique_k_value_l3239_323961


namespace NUMINAMATH_CALUDE_sum_of_fourth_and_fifth_terms_l3239_323966

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_of_fourth_and_fifth_terms
  (a : ℕ → ℕ)
  (h_arithmetic : arithmetic_sequence a)
  (h_first : a 1 = 3)
  (h_second : a 2 = 10)
  (h_third : a 3 = 17)
  (h_sixth : a 6 = 38) :
  a 4 + a 5 = 55 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_fourth_and_fifth_terms_l3239_323966


namespace NUMINAMATH_CALUDE_photo_lineup_arrangements_l3239_323921

/-- The number of boys in the lineup -/
def num_boys : ℕ := 4

/-- The number of girls in the lineup -/
def num_girls : ℕ := 3

/-- The total number of people in the lineup -/
def total_people : ℕ := num_boys + num_girls

/-- The number of arrangements when Boy A must stand at either end -/
def arrangements_boy_a_at_end : ℕ := 1440

/-- The number of arrangements when Girl B cannot stand to the left of Girl C -/
def arrangements_girl_b_not_left_of_c : ℕ := 2520

/-- The number of arrangements when Girl B does not stand at either end, and Girl C does not stand in the middle -/
def arrangements_girl_b_not_end_c_not_middle : ℕ := 3120

theorem photo_lineup_arrangements :
  (arrangements_boy_a_at_end = 1440) ∧
  (arrangements_girl_b_not_left_of_c = 2520) ∧
  (arrangements_girl_b_not_end_c_not_middle = 3120) := by
  sorry

end NUMINAMATH_CALUDE_photo_lineup_arrangements_l3239_323921


namespace NUMINAMATH_CALUDE_students_passing_both_tests_l3239_323958

theorem students_passing_both_tests (total : ℕ) (long_jump : ℕ) (shot_put : ℕ) (failed_both : ℕ) :
  total = 50 →
  long_jump = 40 →
  shot_put = 31 →
  failed_both = 4 →
  ∃ x : ℕ, x = 25 ∧ total = (long_jump - x) + (shot_put - x) + x + failed_both :=
by sorry

end NUMINAMATH_CALUDE_students_passing_both_tests_l3239_323958


namespace NUMINAMATH_CALUDE_max_employees_l3239_323996

theorem max_employees (x : ℝ) (h : x > 4) : 
  ∃ (n : ℕ), n = ⌊2 * (x / (2 * x - 8))⌋ ∧ 
  ∀ (m : ℕ), (∀ (i j : ℕ), i < m → j < m → i ≠ j → 
    ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 8 ∧ t + x / 60 ≤ 8 ∧
    ∃ (ti tj : ℝ), 0 ≤ ti ∧ ti ≤ 8 ∧ 0 ≤ tj ∧ tj ≤ 8 ∧
    (t ≤ ti ∧ ti < t + x / 60) ∧ (t ≤ tj ∧ tj < t + x / 60)) →
  m ≤ n :=
sorry

end NUMINAMATH_CALUDE_max_employees_l3239_323996


namespace NUMINAMATH_CALUDE_other_intersection_point_l3239_323941

def f (x k : ℝ) : ℝ := 3 * (x - 4)^2 + k

theorem other_intersection_point (k : ℝ) :
  f 2 k = 0 → f 6 k = 0 := by
  sorry

end NUMINAMATH_CALUDE_other_intersection_point_l3239_323941


namespace NUMINAMATH_CALUDE_range_of_a_l3239_323973

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (0 < x ∧ x < 4) → (x^2 - 2*x + 1 - a^2 < 0)) →
  (a > 0) →
  (a ≥ 3) := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l3239_323973


namespace NUMINAMATH_CALUDE_tan_ratio_difference_l3239_323979

theorem tan_ratio_difference (x y : ℝ) 
  (h1 : (Real.sin x / Real.cos y) - (Real.sin y / Real.cos x) = 2)
  (h2 : (Real.cos x / Real.sin y) - (Real.cos y / Real.sin x) = 3) :
  (Real.tan x / Real.tan y) - (Real.tan y / Real.tan x) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_ratio_difference_l3239_323979


namespace NUMINAMATH_CALUDE_lawn_mowing_total_l3239_323998

theorem lawn_mowing_total (spring_mows summer_mows : ℕ) 
  (h1 : spring_mows = 6) 
  (h2 : summer_mows = 5) : 
  spring_mows + summer_mows = 11 := by
  sorry

end NUMINAMATH_CALUDE_lawn_mowing_total_l3239_323998


namespace NUMINAMATH_CALUDE_largest_number_l3239_323914

theorem largest_number (a b c d : ℝ) : 
  a = 1 → b = 0 → c = |-2| → d = -3 → 
  max a (max b (max c d)) = |-2| := by
sorry

end NUMINAMATH_CALUDE_largest_number_l3239_323914


namespace NUMINAMATH_CALUDE_side_face_area_is_288_l3239_323982

/-- Represents a rectangular box with length, width, and height. -/
structure RectangularBox where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular box. -/
def volume (box : RectangularBox) : ℝ :=
  box.length * box.width * box.height

/-- Calculates the area of the front face of a rectangular box. -/
def frontFaceArea (box : RectangularBox) : ℝ :=
  box.length * box.width

/-- Calculates the area of the top face of a rectangular box. -/
def topFaceArea (box : RectangularBox) : ℝ :=
  box.length * box.height

/-- Calculates the area of the side face of a rectangular box. -/
def sideFaceArea (box : RectangularBox) : ℝ :=
  box.width * box.height

/-- Theorem stating that given the conditions, the area of the side face is 288. -/
theorem side_face_area_is_288 (box : RectangularBox) 
  (h1 : frontFaceArea box = (1/2) * topFaceArea box)
  (h2 : topFaceArea box = (3/2) * sideFaceArea box)
  (h3 : volume box = 5184) :
  sideFaceArea box = 288 := by
  sorry

end NUMINAMATH_CALUDE_side_face_area_is_288_l3239_323982


namespace NUMINAMATH_CALUDE_union_M_N_complement_M_U_l3239_323927

-- Define the universe set U
def U : Set Nat := {1, 2, 3, 4, 5, 6}

-- Define set M
def M : Set Nat := {2, 3, 4}

-- Define set N
def N : Set Nat := {4, 5}

-- Theorem for the union of M and N
theorem union_M_N : M ∪ N = {2, 3, 4, 5} := by sorry

-- Theorem for the complement of M with respect to U
theorem complement_M_U : (U \ M) = {1, 5, 6} := by sorry

end NUMINAMATH_CALUDE_union_M_N_complement_M_U_l3239_323927


namespace NUMINAMATH_CALUDE_roy_blue_pens_l3239_323903

/-- The number of blue pens Roy has -/
def blue_pens : ℕ := 2

/-- The number of black pens Roy has -/
def black_pens : ℕ := 2 * blue_pens

/-- The number of red pens Roy has -/
def red_pens : ℕ := 2 * black_pens - 2

/-- The total number of pens Roy has -/
def total_pens : ℕ := 12

theorem roy_blue_pens :
  blue_pens = 2 ∧
  black_pens = 2 * blue_pens ∧
  red_pens = 2 * black_pens - 2 ∧
  total_pens = blue_pens + black_pens + red_pens ∧
  total_pens = 12 := by
  sorry

end NUMINAMATH_CALUDE_roy_blue_pens_l3239_323903


namespace NUMINAMATH_CALUDE_number_division_theorem_l3239_323976

theorem number_division_theorem (x : ℚ) : 
  x / 6 = 1 / 10 → x / (3 / 25) = 5 := by
  sorry

end NUMINAMATH_CALUDE_number_division_theorem_l3239_323976


namespace NUMINAMATH_CALUDE_banana_cantaloupe_cost_l3239_323911

/-- Represents the cost of fruits in dollars -/
structure FruitCosts where
  apples : ℝ
  bananas : ℝ
  cantaloupe : ℝ
  dates : ℝ
  cherries : ℝ

/-- The conditions of the fruit purchase problem -/
def fruitProblemConditions (c : FruitCosts) : Prop :=
  c.apples + c.bananas + c.cantaloupe + c.dates + c.cherries = 30 ∧
  c.dates = 3 * c.apples ∧
  c.cantaloupe = c.apples - c.bananas ∧
  c.cherries = c.apples + c.bananas

/-- The theorem stating that under the given conditions, 
    the cost of bananas and cantaloupe is $5 -/
theorem banana_cantaloupe_cost (c : FruitCosts) 
  (h : fruitProblemConditions c) : 
  c.bananas + c.cantaloupe = 5 := by
  sorry

end NUMINAMATH_CALUDE_banana_cantaloupe_cost_l3239_323911


namespace NUMINAMATH_CALUDE_r_value_when_n_is_3_l3239_323985

theorem r_value_when_n_is_3 :
  let n : ℕ := 3
  let s : ℕ := 2^n + 1
  let r : ℕ := 3^s - s + 2
  r = 19676 := by sorry

end NUMINAMATH_CALUDE_r_value_when_n_is_3_l3239_323985


namespace NUMINAMATH_CALUDE_triangle_height_l3239_323968

theorem triangle_height (a b c : ℝ) (h_sum : a + c = 11) 
  (h_angle : Real.cos (π / 3) = 1 / 2) 
  (h_radius : (a * b * Real.sin (π / 3)) / (a + b + c) = 2 / Real.sqrt 3) 
  (h_longer : a > c) : 
  c * Real.sin (π / 3) = 4 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_height_l3239_323968


namespace NUMINAMATH_CALUDE_no_solutions_abs_equation_l3239_323962

theorem no_solutions_abs_equation : ¬∃ y : ℝ, |y - 2| = |y - 1| + |y - 4| := by
  sorry

end NUMINAMATH_CALUDE_no_solutions_abs_equation_l3239_323962


namespace NUMINAMATH_CALUDE_second_company_daily_rate_l3239_323992

/-- The daily rate of Sunshine Car Rentals -/
def sunshine_daily_rate : ℝ := 17.99

/-- The per-mile rate of Sunshine Car Rentals -/
def sunshine_mile_rate : ℝ := 0.18

/-- The per-mile rate of the second car rental company -/
def second_mile_rate : ℝ := 0.16

/-- The number of miles driven -/
def miles_driven : ℝ := 48.0

/-- The daily rate of the second car rental company -/
def second_daily_rate : ℝ := 18.95

theorem second_company_daily_rate :
  sunshine_daily_rate + sunshine_mile_rate * miles_driven =
  second_daily_rate + second_mile_rate * miles_driven :=
by sorry

end NUMINAMATH_CALUDE_second_company_daily_rate_l3239_323992


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3239_323945

/-- Given a hyperbola and a circle satisfying certain conditions, prove that the eccentricity of the hyperbola is 2 -/
theorem hyperbola_eccentricity (a b c : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →  -- Hyperbola equation
  ((c - a)^2 = c^2 / 16) →  -- Circle passes through right focus F(c, 0)
  (∃ k : ℝ, ∀ x y : ℝ, 
    (x - a)^2 + y^2 = c^2 / 16 →  -- Circle equation
    (y = k * x ∨ y = -k * x) →  -- Asymptote equations
    ∃ m : ℝ, m * k = -1 ∧ 
      ∃ x₀ y₀ : ℝ, (x₀ - a)^2 + y₀^2 = c^2 / 16 ∧ 
        y₀ - 0 = m * (x₀ - c)) →  -- Tangent line perpendicular to asymptote
  c / a = 2  -- Eccentricity is 2
:= by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3239_323945


namespace NUMINAMATH_CALUDE_years_until_arun_36_l3239_323937

/-- Proves the number of years that will pass before Arun's age is 36 years -/
theorem years_until_arun_36 (arun_age deepak_age : ℕ) (future_arun_age : ℕ) : 
  arun_age / deepak_age = 5 / 7 →
  deepak_age = 42 →
  future_arun_age = 36 →
  future_arun_age - arun_age = 6 := by
  sorry

end NUMINAMATH_CALUDE_years_until_arun_36_l3239_323937


namespace NUMINAMATH_CALUDE_nested_series_sum_l3239_323999

def nested_series (n : ℕ) : ℕ :=
  match n with
  | 0 => 2
  | k + 1 => 2 * (1 + nested_series k)

theorem nested_series_sum : nested_series 5 = 126 := by
  sorry

end NUMINAMATH_CALUDE_nested_series_sum_l3239_323999


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l3239_323925

theorem absolute_value_inequality (a b c : ℝ) :
  |a + c| < b → |a| < |b| - |c| := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l3239_323925


namespace NUMINAMATH_CALUDE_gcd_of_three_numbers_l3239_323913

theorem gcd_of_three_numbers : Nat.gcd 8885 (Nat.gcd 4514 5246) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_three_numbers_l3239_323913


namespace NUMINAMATH_CALUDE_baker_cakes_problem_l3239_323980

theorem baker_cakes_problem (pastries_made : ℕ) (pastries_sold : ℕ) (cakes_sold : ℕ) :
  pastries_made = 114 →
  pastries_sold = 154 →
  cakes_sold = 78 →
  pastries_sold = cakes_sold + 76 →
  cakes_sold = 78 :=
by sorry

end NUMINAMATH_CALUDE_baker_cakes_problem_l3239_323980


namespace NUMINAMATH_CALUDE_function_always_negative_iff_a_in_range_l3239_323978

theorem function_always_negative_iff_a_in_range :
  ∀ (a : ℝ), (∀ x : ℝ, a * x^2 + a * x - 1 < 0) ↔ -4 < a ∧ a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_function_always_negative_iff_a_in_range_l3239_323978


namespace NUMINAMATH_CALUDE_bridge_length_l3239_323967

/-- The length of a bridge given train characteristics and crossing time -/
theorem bridge_length (train_length : Real) (train_speed_kmh : Real) (crossing_time_s : Real) :
  train_length = 130 ∧ 
  train_speed_kmh = 45 ∧ 
  crossing_time_s = 30 →
  ∃ (bridge_length : Real),
    bridge_length = 245 ∧
    bridge_length + train_length = (train_speed_kmh * 1000 / 3600) * crossing_time_s :=
by sorry

end NUMINAMATH_CALUDE_bridge_length_l3239_323967


namespace NUMINAMATH_CALUDE_monkey_banana_distribution_l3239_323920

/-- Calculates the number of bananas each monkey receives when a family of monkeys divides a collection of bananas equally -/
def bananas_per_monkey (num_monkeys : ℕ) (num_piles_type1 : ℕ) (hands_per_pile_type1 : ℕ) (bananas_per_hand_type1 : ℕ)
                       (num_piles_type2 : ℕ) (hands_per_pile_type2 : ℕ) (bananas_per_hand_type2 : ℕ) : ℕ :=
  let total_bananas := num_piles_type1 * hands_per_pile_type1 * bananas_per_hand_type1 +
                       num_piles_type2 * hands_per_pile_type2 * bananas_per_hand_type2
  total_bananas / num_monkeys

/-- Theorem stating that under the given conditions, each monkey receives 99 bananas -/
theorem monkey_banana_distribution :
  bananas_per_monkey 12 6 9 14 4 12 9 = 99 := by
  sorry

end NUMINAMATH_CALUDE_monkey_banana_distribution_l3239_323920


namespace NUMINAMATH_CALUDE_base_with_six_digits_for_256_l3239_323950

theorem base_with_six_digits_for_256 :
  ∃! (b : ℕ), b > 0 ∧ b^5 ≤ 256 ∧ 256 < b^6 :=
by
  sorry

end NUMINAMATH_CALUDE_base_with_six_digits_for_256_l3239_323950


namespace NUMINAMATH_CALUDE_symmetry_implies_values_l3239_323930

/-- Two points are symmetric with respect to the y-axis if their x-coordinates are negatives of each other and their y-coordinates are equal. -/
def symmetric_y_axis (p q : ℝ × ℝ) : Prop :=
  p.1 = -q.1 ∧ p.2 = q.2

/-- The theorem states that if point P with coordinates (a-b, 2a+b) is symmetric to point Q (3, 2) with respect to the y-axis, then a = -1/3 and b = 8/3. -/
theorem symmetry_implies_values (a b : ℝ) :
  symmetric_y_axis (a - b, 2 * a + b) (3, 2) →
  a = -1/3 ∧ b = 8/3 := by
sorry

end NUMINAMATH_CALUDE_symmetry_implies_values_l3239_323930


namespace NUMINAMATH_CALUDE_min_disks_for_given_files_l3239_323990

/-- Represents the minimum number of disks needed to store files --/
def min_disks (total_files : ℕ) (disk_space : ℚ) 
  (files_1_2MB : ℕ) (files_0_9MB : ℕ) (files_0_5MB : ℕ) : ℕ :=
  sorry

/-- The main theorem stating the minimum number of disks needed --/
theorem min_disks_for_given_files : 
  min_disks 40 2 5 15 20 = 16 := by sorry

end NUMINAMATH_CALUDE_min_disks_for_given_files_l3239_323990


namespace NUMINAMATH_CALUDE_fourth_term_is_sixty_l3239_323994

/-- Represents a stratified sample drawn from an arithmetic sequence of questionnaires. -/
structure StratifiedSample where
  total_questionnaires : ℕ
  sample_size : ℕ
  second_term : ℕ
  h_total : total_questionnaires = 1000
  h_sample : sample_size = 150
  h_second : second_term = 30

/-- The number of questionnaires drawn from the fourth term of the sequence. -/
def fourth_term (s : StratifiedSample) : ℕ := 60

/-- Theorem stating that the fourth term of the stratified sample is 60. -/
theorem fourth_term_is_sixty (s : StratifiedSample) : fourth_term s = 60 := by
  sorry

end NUMINAMATH_CALUDE_fourth_term_is_sixty_l3239_323994


namespace NUMINAMATH_CALUDE_average_speed_calculation_l3239_323935

theorem average_speed_calculation (total_distance : ℝ) (first_half_distance : ℝ) (second_half_distance : ℝ) 
  (first_half_speed : ℝ) (second_half_speed : ℝ) 
  (h1 : total_distance = 50)
  (h2 : first_half_distance = 25)
  (h3 : second_half_distance = 25)
  (h4 : first_half_speed = 60)
  (h5 : second_half_speed = 30)
  (h6 : total_distance = first_half_distance + second_half_distance) :
  (total_distance) / ((first_half_distance / first_half_speed) + (second_half_distance / second_half_speed)) = 40 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l3239_323935


namespace NUMINAMATH_CALUDE_order_of_sqrt_differences_l3239_323901

theorem order_of_sqrt_differences :
  let m : ℝ := Real.sqrt 6 - Real.sqrt 5
  let n : ℝ := Real.sqrt 7 - Real.sqrt 6
  let p : ℝ := Real.sqrt 8 - Real.sqrt 7
  m > n ∧ n > p :=
by sorry

end NUMINAMATH_CALUDE_order_of_sqrt_differences_l3239_323901


namespace NUMINAMATH_CALUDE_trouser_discount_proof_l3239_323934

/-- The final percent decrease in price for a trouser with given original price and discount -/
def final_percent_decrease (original_price discount_percent : ℝ) : ℝ :=
  discount_percent

theorem trouser_discount_proof (original_price discount_percent : ℝ) 
  (h1 : original_price = 100)
  (h2 : discount_percent = 30) :
  final_percent_decrease original_price discount_percent = 30 := by
  sorry

#eval final_percent_decrease 100 30

end NUMINAMATH_CALUDE_trouser_discount_proof_l3239_323934


namespace NUMINAMATH_CALUDE_school_supplies_cost_l3239_323977

/-- The cost of all pencils and pens given their individual prices and quantities -/
def total_cost (pencil_price pen_price : ℚ) (num_pencils num_pens : ℕ) : ℚ :=
  pencil_price * num_pencils + pen_price * num_pens

/-- Theorem stating the total cost of 38 pencils at $2.50 each and 56 pens at $3.50 each is $291.00 -/
theorem school_supplies_cost :
  total_cost (5/2) (7/2) 38 56 = 291 := by
  sorry

end NUMINAMATH_CALUDE_school_supplies_cost_l3239_323977


namespace NUMINAMATH_CALUDE_problem_solution_l3239_323957

theorem problem_solution (x z : ℚ) : 
  x = 103 → x^3*z - 3*x^2*z + 2*x*z = 208170 → z = 5/265 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3239_323957


namespace NUMINAMATH_CALUDE_cubic_expression_zero_l3239_323926

theorem cubic_expression_zero (x : ℝ) (h : x^2 + 3*x - 3 = 0) : 
  x^3 + 2*x^2 - 6*x + 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_expression_zero_l3239_323926


namespace NUMINAMATH_CALUDE_power_function_value_l3239_323986

/-- Given a power function f and a point on its graph, prove the value of f at -2 -/
theorem power_function_value (f : ℝ → ℝ) (α : ℝ) :
  (∀ x, f x = x ^ α) →  -- f is a power function
  f (Real.sqrt 2 / 2) = Real.sqrt 2 / 4 →  -- given point lies on the graph
  f (-2) = -8 :=
by sorry

end NUMINAMATH_CALUDE_power_function_value_l3239_323986


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3239_323964

theorem polynomial_factorization (a : ℝ) : a^3 + 2*a^2 + a = a*(a+1)^2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3239_323964


namespace NUMINAMATH_CALUDE_third_month_sale_proof_l3239_323916

/-- Calculates the sale in the third month given the sales for other months and the average --/
def third_month_sale (first_month : ℕ) (second_month : ℕ) (fourth_month : ℕ) (fifth_month : ℕ) (sixth_month : ℕ) (average : ℕ) : ℕ :=
  6 * average - (first_month + second_month + fourth_month + fifth_month + sixth_month)

theorem third_month_sale_proof :
  third_month_sale 5266 5744 6122 6588 4916 5750 = 5864 := by
  sorry

end NUMINAMATH_CALUDE_third_month_sale_proof_l3239_323916


namespace NUMINAMATH_CALUDE_multiples_count_theorem_l3239_323960

def count_multiples (n : ℕ) (d : ℕ) : ℕ :=
  (n / d : ℕ)

def count_multiples_of_2_or_3_not_4_or_5 (upper_bound : ℕ) : ℕ :=
  count_multiples upper_bound 2 + count_multiples upper_bound 3 -
  count_multiples upper_bound 6 - count_multiples upper_bound 4 -
  count_multiples upper_bound 5 + count_multiples upper_bound 20

theorem multiples_count_theorem (upper_bound : ℕ) :
  upper_bound = 200 →
  count_multiples_of_2_or_3_not_4_or_5 upper_bound = 53 := by
  sorry

end NUMINAMATH_CALUDE_multiples_count_theorem_l3239_323960


namespace NUMINAMATH_CALUDE_nanjing_visitors_scientific_notation_l3239_323959

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem nanjing_visitors_scientific_notation :
  toScientificNotation 44300000 = ScientificNotation.mk 4.43 7 sorry := by
  sorry

end NUMINAMATH_CALUDE_nanjing_visitors_scientific_notation_l3239_323959


namespace NUMINAMATH_CALUDE_simplify_expression_l3239_323989

theorem simplify_expression (x : ℝ) : 3*x + 6*x + 9*x + 12*x + 15*x + 18 = 45*x + 18 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3239_323989


namespace NUMINAMATH_CALUDE_coupon1_best_l3239_323931

def coupon1_discount (x : ℝ) : ℝ := 0.15 * x

def coupon2_discount (x : ℝ) : ℝ := 30

def coupon3_discount (x : ℝ) : ℝ := 0.25 * (x - 150)

theorem coupon1_best (x : ℝ) (h1 : x > 100) : 
  (coupon1_discount x > coupon2_discount x ∧ coupon1_discount x > coupon3_discount x) ↔ 
  (200 < x ∧ x < 375) := by sorry

end NUMINAMATH_CALUDE_coupon1_best_l3239_323931


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3239_323951

def A : Set ℝ := {x | x + 1/2 ≥ 3/2}
def B : Set ℝ := {x | x^2 + x < 6}

theorem intersection_of_A_and_B : 
  A ∩ B = {x : ℝ | (-3 < x ∧ x ≤ -2) ∨ (1 ≤ x ∧ x < 2)} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3239_323951


namespace NUMINAMATH_CALUDE_lcm_gcd_sum_implies_divisibility_l3239_323929

theorem lcm_gcd_sum_implies_divisibility (m n : ℕ) :
  Nat.lcm m n + Nat.gcd m n = m + n → m ∣ n ∨ n ∣ m := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_sum_implies_divisibility_l3239_323929


namespace NUMINAMATH_CALUDE_shirt_cost_to_marked_price_ratio_l3239_323946

/-- Given a shop with shirts on sale, this theorem proves the ratio of cost to marked price. -/
theorem shirt_cost_to_marked_price_ratio :
  ∀ (marked_price : ℝ), marked_price > 0 →
  let discount_rate : ℝ := 0.25
  let selling_price : ℝ := marked_price * (1 - discount_rate)
  let cost_rate : ℝ := 0.60
  let cost_price : ℝ := selling_price * cost_rate
  cost_price / marked_price = 0.45 := by
sorry


end NUMINAMATH_CALUDE_shirt_cost_to_marked_price_ratio_l3239_323946


namespace NUMINAMATH_CALUDE_wedding_guests_l3239_323965

theorem wedding_guests (total : ℕ) 
  (h1 : (83 : ℚ) / 100 * total + (9 : ℚ) / 100 * total + 16 = total) : 
  total = 200 := by
  sorry

end NUMINAMATH_CALUDE_wedding_guests_l3239_323965


namespace NUMINAMATH_CALUDE_square_root_inequalities_l3239_323917

theorem square_root_inequalities : 
  (∃ (x y : ℝ), x = Real.sqrt 7 ∧ y = Real.sqrt 3 ∧ x + y ≠ Real.sqrt 10) ∧
  (Real.sqrt 3 * Real.sqrt 5 = Real.sqrt 15) ∧
  (Real.sqrt 6 / Real.sqrt 3 = Real.sqrt 2) ∧
  ((-Real.sqrt 3)^2 = 3) := by
  sorry


end NUMINAMATH_CALUDE_square_root_inequalities_l3239_323917


namespace NUMINAMATH_CALUDE_st_plus_tu_equals_ten_l3239_323955

/-- Represents a polygon PQRSTU -/
structure Polygon where
  area : ℝ
  pq : ℝ
  qr : ℝ
  up : ℝ
  st : ℝ
  tu : ℝ

/-- Theorem stating the sum of ST and TU in the given polygon -/
theorem st_plus_tu_equals_ten (poly : Polygon) 
  (h_area : poly.area = 64)
  (h_pq : poly.pq = 10)
  (h_qr : poly.qr = 10)
  (h_up : poly.up = 6) :
  poly.st + poly.tu = 10 := by
  sorry

end NUMINAMATH_CALUDE_st_plus_tu_equals_ten_l3239_323955


namespace NUMINAMATH_CALUDE_complex_distance_bounds_l3239_323969

theorem complex_distance_bounds (z : ℂ) (h : Complex.abs (z + 2 - 2*I) = 1) :
  (∃ (w : ℂ), Complex.abs (z - 2 - 2*I) = 3 ∧ 
    ∀ (u : ℂ), Complex.abs (u + 2 - 2*I) = 1 → Complex.abs (u - 2 - 2*I) ≥ 3) ∧
  (∃ (w : ℂ), Complex.abs (z - 2 - 2*I) = 5 ∧ 
    ∀ (u : ℂ), Complex.abs (u + 2 - 2*I) = 1 → Complex.abs (u - 2 - 2*I) ≤ 5) :=
sorry

end NUMINAMATH_CALUDE_complex_distance_bounds_l3239_323969


namespace NUMINAMATH_CALUDE_unique_scenario_l3239_323906

/-- Represents the type of islander -/
inductive IslanderType
  | Knight
  | Liar

/-- Represents the possible responses to the question -/
inductive Response
  | Yes
  | No

/-- Represents the scenario of two islanders -/
structure IslandScenario where
  askedIslander : IslanderType
  otherIslander : IslanderType
  response : Response

/-- Determines if a given scenario is consistent with the rules of knights and liars -/
def isConsistentScenario (scenario : IslandScenario) : Prop :=
  match scenario.askedIslander, scenario.response with
  | IslanderType.Knight, Response.Yes => scenario.askedIslander = IslanderType.Knight ∨ scenario.otherIslander = IslanderType.Knight
  | IslanderType.Knight, Response.No => scenario.askedIslander ≠ IslanderType.Knight ∧ scenario.otherIslander ≠ IslanderType.Knight
  | IslanderType.Liar, Response.Yes => scenario.askedIslander ≠ IslanderType.Knight ∧ scenario.otherIslander ≠ IslanderType.Knight
  | IslanderType.Liar, Response.No => scenario.askedIslander = IslanderType.Knight ∨ scenario.otherIslander = IslanderType.Knight

/-- Determines if a given scenario provides definitive information about both islanders -/
def providesDefinitiveInfo (scenario : IslandScenario) : Prop :=
  isConsistentScenario scenario ∧
  ∀ (altScenario : IslandScenario),
    isConsistentScenario altScenario →
    scenario.askedIslander = altScenario.askedIslander ∧
    scenario.otherIslander = altScenario.otherIslander

/-- The main theorem: The only scenario that satisfies all conditions is when the asked islander is a liar and the other is a knight -/
theorem unique_scenario :
  ∃! (scenario : IslandScenario),
    isConsistentScenario scenario ∧
    providesDefinitiveInfo scenario ∧
    scenario.askedIslander = IslanderType.Liar ∧
    scenario.otherIslander = IslanderType.Knight :=
  sorry

end NUMINAMATH_CALUDE_unique_scenario_l3239_323906


namespace NUMINAMATH_CALUDE_sum_of_squares_unique_value_l3239_323991

theorem sum_of_squares_unique_value 
  (p q r : ℕ+) 
  (sum_eq : p + q + r = 30) 
  (gcd_sum_eq : Nat.gcd p.val q.val + Nat.gcd q.val r.val + Nat.gcd r.val p.val = 10) : 
  p^2 + q^2 + r^2 = 584 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_unique_value_l3239_323991


namespace NUMINAMATH_CALUDE_card_game_proof_l3239_323918

def deck_size : ℕ := 60
def hand_size : ℕ := 12

theorem card_game_proof :
  let combinations := Nat.choose deck_size hand_size
  ∃ (B : ℕ), 
    B = 7 ∧ 
    combinations = 17 * 10^10 + B * 10^9 + B * 10^7 + 5 * 10^6 + 2 * 10^5 + 9 * 10^4 + 8 * 10 + B ∧
    combinations % 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_card_game_proof_l3239_323918


namespace NUMINAMATH_CALUDE_probability_club_heart_king_l3239_323963

theorem probability_club_heart_king (total_cards : ℕ) (clubs : ℕ) (hearts : ℕ) (kings : ℕ) :
  total_cards = 52 →
  clubs = 13 →
  hearts = 13 →
  kings = 4 →
  (clubs / total_cards) * (hearts / (total_cards - 1)) * (kings / (total_cards - 2)) = 13 / 2550 := by
  sorry

end NUMINAMATH_CALUDE_probability_club_heart_king_l3239_323963


namespace NUMINAMATH_CALUDE_smallest_d_inequality_l3239_323997

theorem smallest_d_inequality (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) :
  Real.sqrt (x * y * z) + (1/3) * |x^2 - y^2 + z^2| ≥ (x + y + z) / 3 ∧
  ∀ d : ℝ, d > 0 → d < 1/3 → ∃ a b c : ℝ, a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧
    Real.sqrt (a * b * c) + d * |a^2 - b^2 + c^2| < (a + b + c) / 3 :=
sorry

end NUMINAMATH_CALUDE_smallest_d_inequality_l3239_323997


namespace NUMINAMATH_CALUDE_pearl_string_value_l3239_323902

/-- Represents a string of pearls with a given middle pearl value and decreasing rates on each side. -/
structure PearlString where
  middleValue : ℕ
  decreaseRate1 : ℕ
  decreaseRate2 : ℕ

/-- Calculates the total value of the pearl string. -/
def totalValue (ps : PearlString) : ℕ :=
  ps.middleValue + 16 * ps.middleValue - 16 * 17 * ps.decreaseRate1 / 2 +
  16 * ps.middleValue - 16 * 17 * ps.decreaseRate2 / 2

/-- Calculates the value of the fourth pearl from the middle on the more expensive side. -/
def fourthPearlValue (ps : PearlString) : ℕ :=
  ps.middleValue - 4 * min ps.decreaseRate1 ps.decreaseRate2

/-- The main theorem stating the conditions and the result to be proven. -/
theorem pearl_string_value (ps : PearlString) :
  ps.decreaseRate1 = 3000 →
  ps.decreaseRate2 = 4500 →
  totalValue ps = 25 * fourthPearlValue ps →
  ps.middleValue = 90000 := by
  sorry

end NUMINAMATH_CALUDE_pearl_string_value_l3239_323902


namespace NUMINAMATH_CALUDE_parabola_shift_l3239_323939

/-- The original parabola function -/
def f (x : ℝ) : ℝ := x^2 - 4*x + 3

/-- The shifted parabola function -/
def g (x : ℝ) : ℝ := (x + 1)^2 - 4*(x + 1) + 3 + 2

/-- Theorem stating that the shifted parabola is equivalent to x^2 - 2x + 2 -/
theorem parabola_shift :
  ∀ x : ℝ, g x = x^2 - 2*x + 2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_shift_l3239_323939


namespace NUMINAMATH_CALUDE_ratio_sum_theorem_l3239_323995

theorem ratio_sum_theorem (a b c d : ℝ) : 
  b = 2 * a ∧ c = 4 * a ∧ d = 5 * a ∧ 
  a^2 + b^2 + c^2 + d^2 = 2460 →
  abs ((a + b + c + d) - 87.744) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_ratio_sum_theorem_l3239_323995


namespace NUMINAMATH_CALUDE_invalid_set_l3239_323983

/-- A set of three positive real numbers representing the lengths of external diagonals of a right regular prism. -/
structure ExternalDiagonals where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : a > 0
  b_pos : b > 0
  c_pos : c > 0

/-- The condition for a valid set of external diagonals. -/
def is_valid (d : ExternalDiagonals) : Prop :=
  d.a^2 + d.b^2 > d.c^2 ∧ 
  d.b^2 + d.c^2 > d.a^2 ∧ 
  d.c^2 + d.a^2 > d.b^2

/-- The set {5,7,9} is not a valid set of external diagonals for a right regular prism. -/
theorem invalid_set : ¬ is_valid ⟨5, 7, 9, by norm_num, by norm_num, by norm_num⟩ := by
  sorry

end NUMINAMATH_CALUDE_invalid_set_l3239_323983


namespace NUMINAMATH_CALUDE_jessie_final_position_l3239_323949

/-- The number of steps Jessie takes in total -/
def total_steps : ℕ := 6

/-- The final position Jessie reaches -/
def final_position : ℕ := 24

/-- The number of steps to reach point x -/
def steps_to_x : ℕ := 4

/-- The number of steps from x to z -/
def steps_x_to_z : ℕ := 1

/-- The number of steps from z to y -/
def steps_z_to_y : ℕ := 1

/-- The length of each step -/
def step_length : ℚ := final_position / total_steps

/-- The position of point x -/
def x : ℚ := step_length * steps_to_x

/-- The position of point z -/
def z : ℚ := x + step_length * steps_x_to_z

/-- The position of point y -/
def y : ℚ := z + step_length * steps_z_to_y

theorem jessie_final_position : y = 24 := by
  sorry

end NUMINAMATH_CALUDE_jessie_final_position_l3239_323949


namespace NUMINAMATH_CALUDE_soap_amount_is_fifteen_l3239_323987

/-- Represents the recipe for bubble mix -/
structure BubbleMixRecipe where
  soap_per_cup : ℚ  -- tablespoons of soap per cup of water
  ounces_per_cup : ℚ  -- ounces in a cup of water

/-- Represents a container for bubble mix -/
structure BubbleMixContainer where
  capacity : ℚ  -- capacity in ounces

/-- Calculates the amount of soap needed for a given container and recipe -/
def soap_needed (recipe : BubbleMixRecipe) (container : BubbleMixContainer) : ℚ :=
  (container.capacity / recipe.ounces_per_cup) * recipe.soap_per_cup

/-- Theorem: The amount of soap needed for the given recipe and container is 15 tablespoons -/
theorem soap_amount_is_fifteen (recipe : BubbleMixRecipe) (container : BubbleMixContainer) 
    (h1 : recipe.soap_per_cup = 3)
    (h2 : recipe.ounces_per_cup = 8)
    (h3 : container.capacity = 40) :
    soap_needed recipe container = 15 := by
  sorry

end NUMINAMATH_CALUDE_soap_amount_is_fifteen_l3239_323987


namespace NUMINAMATH_CALUDE_abc_sum_bound_l3239_323953

theorem abc_sum_bound (a b c : ℝ) (h1 : a + b + c = 1) (h2 : c = -1) :
  (∀ x, x ≤ a * b + a * c + b * c → x ≤ -1) ∧
  (∀ ε > 0, ∃ a b, a + b = 2 ∧ a * b + a * c + b * c > -1 - ε) :=
sorry

end NUMINAMATH_CALUDE_abc_sum_bound_l3239_323953


namespace NUMINAMATH_CALUDE_geometry_relations_l3239_323932

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Line → Prop)
variable (perp_line_plane : Line → Plane → Prop)
variable (perp_plane : Plane → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (parallel_line : Line → Line → Prop)

-- Define the lines and planes
variable (l m : Line) (α β γ : Plane)

-- State the theorem
theorem geometry_relations :
  (perp l m ∧ perp_line_plane l α ∧ perp_line_plane m β → perp_plane α β) ∧
  (parallel α β ∧ parallel β γ → parallel α γ) ∧
  (perp_line_plane l α ∧ parallel α β → perp_line_plane l β) ∧
  (perp_line_plane l α ∧ perp_line_plane m α → parallel_line l m) :=
by sorry

end NUMINAMATH_CALUDE_geometry_relations_l3239_323932


namespace NUMINAMATH_CALUDE_factor_polynomial_l3239_323907

theorem factor_polynomial (x : ℝ) : 46 * x^3 - 115 * x^7 = -23 * x^3 * (5 * x^4 - 2) := by
  sorry

end NUMINAMATH_CALUDE_factor_polynomial_l3239_323907


namespace NUMINAMATH_CALUDE_quadratic_coefficient_determination_l3239_323948

/-- A quadratic function f(x) = ax^2 + bx + c -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_coefficient_determination
  (a b c : ℝ)
  (f : ℝ → ℝ)
  (h_f : f = QuadraticFunction a b c)
  (h_point : f 0 = 3)
  (h_vertex : ∃ (k : ℝ), f 2 = -1 ∧ ∀ x, f x ≥ f 2) :
  a = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_determination_l3239_323948


namespace NUMINAMATH_CALUDE_roots_relation_l3239_323956

theorem roots_relation (n r : ℝ) (c d : ℝ) : 
  (c^2 - n*c + 3 = 0) → 
  (d^2 - n*d + 3 = 0) → 
  ((c + 1/d)^2 - r*(c + 1/d) + s = 0) → 
  ((d + 1/c)^2 - r*(d + 1/c) + s = 0) → 
  s = 16/3 := by sorry

end NUMINAMATH_CALUDE_roots_relation_l3239_323956


namespace NUMINAMATH_CALUDE_fruit_shop_apples_l3239_323940

theorem fruit_shop_apples (total : ℕ) 
  (h1 : (3 : ℚ) / 10 * total + (4 : ℚ) / 10 * total = 140) : total = 200 := by
  sorry

end NUMINAMATH_CALUDE_fruit_shop_apples_l3239_323940


namespace NUMINAMATH_CALUDE_multiplicative_inverse_modulo_l3239_323919

def A : Nat := 123456
def B : Nat := 142857
def M : Nat := 1000000

theorem multiplicative_inverse_modulo :
  (892857 * (A * B)) % M = 1 := by sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_modulo_l3239_323919


namespace NUMINAMATH_CALUDE_intersection_A_B_l3239_323908

def A : Set ℤ := {1, 2, 3}
def B : Set ℤ := {x | x^2 - 4 ≤ 0}

theorem intersection_A_B : A ∩ B = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3239_323908


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l3239_323952

-- Problem 1
theorem problem_1 : (Real.sqrt 12 - Real.sqrt 6) / Real.sqrt 3 + 2 / Real.sqrt 2 = 2 := by sorry

-- Problem 2
theorem problem_2 : (2 + Real.sqrt 3) * (2 - Real.sqrt 3) + (2 - Real.sqrt 3)^2 = 8 - 4 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l3239_323952


namespace NUMINAMATH_CALUDE_community_service_arrangements_l3239_323904

def volunteers : ℕ := 8
def service_days : ℕ := 5

def arrangements (n m : ℕ) : ℕ := sorry

theorem community_service_arrangements :
  let total_arrangements := 
    (arrangements 2 1 * arrangements 6 4 * arrangements 5 5) + 
    (arrangements 2 2 * arrangements 6 3 * arrangements 4 2)
  total_arrangements = 5040 := by sorry

end NUMINAMATH_CALUDE_community_service_arrangements_l3239_323904


namespace NUMINAMATH_CALUDE_tank_filling_time_l3239_323938

/-- Given a tap that can fill a tank in 16 hours, and 3 additional similar taps opened after half the tank is filled, prove that the total time taken to fill the tank completely is 10 hours. -/
theorem tank_filling_time (fill_time : ℝ) (additional_taps : ℕ) : 
  fill_time = 16 → additional_taps = 3 → 
  (fill_time / 2) + (fill_time / (2 * (additional_taps + 1))) = 10 :=
by sorry

end NUMINAMATH_CALUDE_tank_filling_time_l3239_323938


namespace NUMINAMATH_CALUDE_inequalities_given_m_gt_neg_one_l3239_323944

theorem inequalities_given_m_gt_neg_one (m : ℝ) (h : m > -1) :
  (4*m > -4) ∧ (-5*m < -5) ∧ (m+1 > 0) ∧ (1-m < 2) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_given_m_gt_neg_one_l3239_323944


namespace NUMINAMATH_CALUDE_distance_between_points_l3239_323900

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (2, 3)
  let p2 : ℝ × ℝ := (7, -2)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = 5 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_distance_between_points_l3239_323900


namespace NUMINAMATH_CALUDE_triangle_angle_A_l3239_323974

theorem triangle_angle_A (A : Real) (h : 4 * Real.pi * Real.sin A - 3 * Real.arccos (-1/2) = 0) :
  A = Real.pi / 6 ∨ A = 5 * Real.pi / 6 :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_A_l3239_323974


namespace NUMINAMATH_CALUDE_power_of_product_l3239_323922

theorem power_of_product (a : ℝ) : (2 * a) ^ 3 = 8 * a ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l3239_323922


namespace NUMINAMATH_CALUDE_rectangle_folding_l3239_323975

theorem rectangle_folding (a b : ℝ) : 
  a = 5 ∧ 
  0 < b ∧ 
  b < 4 ∧ 
  (a - b)^2 + b^2 = 6 → 
  b = Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_rectangle_folding_l3239_323975


namespace NUMINAMATH_CALUDE_phone_plan_fee_proof_l3239_323943

/-- The monthly fee for the first plan -/
def first_plan_fee : ℝ := 22

/-- The per-minute rate for the first plan -/
def first_plan_rate : ℝ := 0.13

/-- The per-minute rate for the second plan -/
def second_plan_rate : ℝ := 0.18

/-- The number of minutes at which the plans cost the same -/
def equal_cost_minutes : ℝ := 280

/-- The monthly fee for the second plan -/
def second_plan_fee : ℝ := 8

theorem phone_plan_fee_proof :
  first_plan_fee + first_plan_rate * equal_cost_minutes =
  second_plan_fee + second_plan_rate * equal_cost_minutes :=
by sorry

end NUMINAMATH_CALUDE_phone_plan_fee_proof_l3239_323943


namespace NUMINAMATH_CALUDE_expand_expression_l3239_323909

theorem expand_expression (x y z : ℝ) : 
  (2*x - 3) * (4*y + 5 - 2*z) = 8*x*y + 10*x - 4*x*z - 12*y + 6*z - 15 := by
sorry

end NUMINAMATH_CALUDE_expand_expression_l3239_323909
