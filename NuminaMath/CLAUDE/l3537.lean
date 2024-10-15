import Mathlib

namespace NUMINAMATH_CALUDE_factorization_yx_squared_minus_y_l3537_353754

theorem factorization_yx_squared_minus_y (x y : ℝ) : y * x^2 - y = y * (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_yx_squared_minus_y_l3537_353754


namespace NUMINAMATH_CALUDE_smallest_positive_multiple_of_45_l3537_353789

theorem smallest_positive_multiple_of_45 :
  ∀ n : ℕ, n > 0 → 45 ∣ n → n ≥ 45 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_multiple_of_45_l3537_353789


namespace NUMINAMATH_CALUDE_arithmetic_expression_result_l3537_353772

theorem arithmetic_expression_result : 5 + 12 / 3 - 4 * 2 + 3^2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_result_l3537_353772


namespace NUMINAMATH_CALUDE_sequence_divisibility_l3537_353722

/-- A sequence of 2007 elements, each either 2 or 3 -/
def Sequence := Fin 2007 → Fin 2

/-- The property that all elements of a sequence are divisible by 5 -/
def AllDivisibleBy5 (x : Fin 2007 → ℤ) : Prop :=
  ∀ i, x i % 5 = 0

/-- The main theorem -/
theorem sequence_divisibility (a : Sequence) (x : Fin 2007 → ℤ)
    (h : ∀ i : Fin 2007, (a i.val + 2 : Fin 2) * x i + x ((i + 2) % 2007) ≡ 0 [ZMOD 5]) :
    AllDivisibleBy5 x := by
  sorry

end NUMINAMATH_CALUDE_sequence_divisibility_l3537_353722


namespace NUMINAMATH_CALUDE_cube_root_27_fourth_root_81_sixth_root_64_eq_18_l3537_353745

theorem cube_root_27_fourth_root_81_sixth_root_64_eq_18 :
  (27 : ℝ) ^ (1/3) * (81 : ℝ) ^ (1/4) * (64 : ℝ) ^ (1/6) = 18 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_27_fourth_root_81_sixth_root_64_eq_18_l3537_353745


namespace NUMINAMATH_CALUDE_mask_count_l3537_353727

theorem mask_count (num_boxes : ℕ) (capacity : ℕ) (lacking : ℕ) (total_masks : ℕ) : 
  num_boxes = 18 → 
  capacity = 15 → 
  lacking = 3 → 
  total_masks = num_boxes * (capacity - lacking) → 
  total_masks = 216 := by
sorry

end NUMINAMATH_CALUDE_mask_count_l3537_353727


namespace NUMINAMATH_CALUDE_average_miles_per_year_approx_2000_l3537_353717

/-- Calculates the approximate average miles rowed per year -/
def approximateAverageMilesPerYear (currentAge : ℕ) (ageReceived : ℕ) (totalMiles : ℕ) : ℕ :=
  let yearsRowing := currentAge - ageReceived
  let exactAverage := totalMiles / yearsRowing
  -- Round to the nearest thousand
  (exactAverage + 500) / 1000 * 1000

/-- Theorem stating that the average miles rowed per year is approximately 2000 -/
theorem average_miles_per_year_approx_2000 :
  approximateAverageMilesPerYear 63 50 25048 = 2000 := by
  sorry

#eval approximateAverageMilesPerYear 63 50 25048

end NUMINAMATH_CALUDE_average_miles_per_year_approx_2000_l3537_353717


namespace NUMINAMATH_CALUDE_problem_statement_l3537_353769

theorem problem_statement (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_not_all_equal : ¬(a = b ∧ b = c)) : 
  ((a - b)^2 + (b - c)^2 + (c - a)^2 ≠ 0) ∧ 
  (a > b ∨ a < b ∨ a = b) ∧
  (∃ (x y z : ℝ), x ≠ z ∧ y ≠ z ∧ x ≠ y) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l3537_353769


namespace NUMINAMATH_CALUDE_square_perimeter_after_scaling_l3537_353750

theorem square_perimeter_after_scaling (a : ℝ) (h : a > 0) : 
  let s := Real.sqrt a
  let new_s := 3 * s
  a = 4 → 4 * new_s = 24 := by
sorry

end NUMINAMATH_CALUDE_square_perimeter_after_scaling_l3537_353750


namespace NUMINAMATH_CALUDE_f_pi_sixth_value_l3537_353738

/-- Given a function f(x) = 2sin(ωx + φ) where for all x, f(π/3 + x) = f(-x),
    prove that f(π/6) is either -2 or 2. -/
theorem f_pi_sixth_value (ω φ : ℝ) :
  let f : ℝ → ℝ := λ x ↦ 2 * Real.sin (ω * x + φ)
  (∀ x, f (π / 3 + x) = f (-x)) →
  f (π / 6) = -2 ∨ f (π / 6) = 2 :=
by sorry

end NUMINAMATH_CALUDE_f_pi_sixth_value_l3537_353738


namespace NUMINAMATH_CALUDE_log_sqrt10_1000sqrt10_l3537_353707

theorem log_sqrt10_1000sqrt10 : Real.log (1000 * Real.sqrt 10) / Real.log (Real.sqrt 10) = 7 := by
  sorry

end NUMINAMATH_CALUDE_log_sqrt10_1000sqrt10_l3537_353707


namespace NUMINAMATH_CALUDE_jersey_shoe_cost_ratio_l3537_353725

/-- Given the information about Jeff's purchase of shoes and jerseys,
    prove that the ratio of the cost of one jersey to one pair of shoes is 1:4 -/
theorem jersey_shoe_cost_ratio :
  ∀ (total_cost shoe_cost : ℕ) (shoe_pairs jersey_count : ℕ),
    total_cost = 560 →
    shoe_cost = 480 →
    shoe_pairs = 6 →
    jersey_count = 4 →
    (total_cost - shoe_cost) / jersey_count / (shoe_cost / shoe_pairs) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_jersey_shoe_cost_ratio_l3537_353725


namespace NUMINAMATH_CALUDE_star_properties_l3537_353729

noncomputable def star (x y : ℝ) : ℝ := Real.log (10^x + 10^y) / Real.log 10

theorem star_properties :
  (∀ a b : ℝ, star a b = star b a) ∧
  (∀ a b c : ℝ, star (star a b) c = star a (star b c)) ∧
  (∀ a b c : ℝ, star a b + c = star (a + c) (b + c)) ∧
  (∃ a b c : ℝ, star a b * c ≠ star (a * c) (b * c)) :=
by sorry

end NUMINAMATH_CALUDE_star_properties_l3537_353729


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l3537_353775

theorem diophantine_equation_solutions (p q : ℕ) (hp : Prime p) (hq : Prime q) (hpq : p ≠ q) :
  let solutions := {(a, b) : ℕ × ℕ | (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / (p * q)}
  solutions = {
    (1 + p*q, p^2*q^2 + p*q),
    (p*(q + 1), p*q*(q + 1)),
    (q*(p + 1), p*q*(p + 1)),
    (2*p*q, 2*p*q),
    (p^2*q*(p + q), q^2 + p*q),
    (q^2 + p*q, p^2 + p*q),
    (p*q*(p + 1), q*(p + 1)),
    (p*q*(q + 1), p*(q + 1)),
    (p^2*q^2 + p*q, 1 + p*q)
  } := by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l3537_353775


namespace NUMINAMATH_CALUDE_limit_log_div_power_l3537_353768

open Real

-- Define the function f(x) = ln(x) / x^α
noncomputable def f (α : ℝ) (x : ℝ) : ℝ := (log x) / (x ^ α)

-- State the theorem
theorem limit_log_div_power (α : ℝ) (h₁ : α > 0) :
  ∀ ε > 0, ∃ N, ∀ x ≥ N, x > 0 → |f α x - 0| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_log_div_power_l3537_353768


namespace NUMINAMATH_CALUDE_log_lower_bound_l3537_353781

/-- The number of distinct prime factors of a positive integer -/
def num_distinct_prime_factors (n : ℕ+) : ℕ :=
  (Nat.factors n).toFinset.card

/-- For any positive integer n, log(n) ≥ k * log(2), where k is the number of distinct prime factors of n -/
theorem log_lower_bound (n : ℕ+) :
  Real.log n ≥ (num_distinct_prime_factors n : ℝ) * Real.log 2 := by
  sorry


end NUMINAMATH_CALUDE_log_lower_bound_l3537_353781


namespace NUMINAMATH_CALUDE_max_value_of_sum_l3537_353739

theorem max_value_of_sum (a b c d : ℝ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_sum : a / b + b / c + c / d + d / a = 4) (h_prod : a * c = b * d) :
  ∃ (max : ℝ), max = -12 ∧ ∀ (a' b' c' d' : ℝ),
    a' / b' + b' / c' + c' / d' + d' / a' = 4 → a' * c' = b' * d' →
    a' / c' + b' / d' + c' / a' + d' / b' ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_sum_l3537_353739


namespace NUMINAMATH_CALUDE_min_value_of_f_l3537_353795

/-- The function f(x) = 2x^3 - 6x^2 + a -/
def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 6 * x^2 + a

theorem min_value_of_f (a : ℝ) :
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f a x = 3) ∧
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f a x ≤ 3) →
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f a x = -37) ∧
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f a x ≥ -37) := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3537_353795


namespace NUMINAMATH_CALUDE_julie_reading_problem_l3537_353782

/-- The number of pages in Julie's book -/
def total_pages : ℕ := 120

/-- The number of pages Julie read yesterday -/
def pages_yesterday : ℕ := 12

/-- The number of pages Julie read today -/
def pages_today : ℕ := 2 * pages_yesterday

/-- The number of pages remaining after Julie read yesterday and today -/
def remaining_pages : ℕ := total_pages - (pages_yesterday + pages_today)

theorem julie_reading_problem :
  (pages_yesterday = 12) ∧
  (total_pages = 120) ∧
  (pages_today = 2 * pages_yesterday) ∧
  (remaining_pages / 2 = 42) :=
by sorry

end NUMINAMATH_CALUDE_julie_reading_problem_l3537_353782


namespace NUMINAMATH_CALUDE_circle_radius_existence_l3537_353714

/-- Representation of a circle -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Representation of a point -/
def Point := ℝ × ℝ

/-- Distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Check if a point is on a circle -/
def isOnCircle (p : Point) (c : Circle) : Prop := sorry

/-- Check if two circles intersect at two points -/
def circlesIntersect (c1 c2 : Circle) : Prop := sorry

/-- Check if a point is outside a circle -/
def isOutside (p : Point) (c : Circle) : Prop := sorry

/-- Check if a circle is the circumcircle of a triangle -/
def isCircumcircle (c : Circle) (p1 p2 p3 : Point) : Prop := sorry

theorem circle_radius_existence :
  ∃! r : ℝ, r > 0 ∧
  ∃ (C1 C2 : Circle) (O X Y Z : Point),
    C1.radius = r ∧
    C1.center = O ∧
    isOnCircle O C2 ∧
    circlesIntersect C1 C2 ∧
    isOnCircle X C1 ∧ isOnCircle X C2 ∧
    isOnCircle Y C1 ∧ isOnCircle Y C2 ∧
    isOnCircle Z C2 ∧
    isOutside Z C1 ∧
    distance X Z = 15 ∧
    distance O Z = 13 ∧
    distance Y Z = 9 ∧
    isCircumcircle C2 X O Z ∧
    isCircumcircle C2 O Y Z :=
sorry

end NUMINAMATH_CALUDE_circle_radius_existence_l3537_353714


namespace NUMINAMATH_CALUDE_definite_integral_evaluation_l3537_353702

theorem definite_integral_evaluation : ∫ x in (1:ℝ)..2, (3 * x^2 - 1) = 6 := by
  sorry

end NUMINAMATH_CALUDE_definite_integral_evaluation_l3537_353702


namespace NUMINAMATH_CALUDE_smallest_n_divisible_by_1000_l3537_353757

theorem smallest_n_divisible_by_1000 : 
  ∃ n : ℕ, (∀ m : ℕ, m < n → ¬(1000 ∣ (m+1)*(m+2)*(m+3)*(m+4))) ∧ 
  (1000 ∣ (n+1)*(n+2)*(n+3)*(n+4)) ∧ n = 121 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_divisible_by_1000_l3537_353757


namespace NUMINAMATH_CALUDE_dormitory_to_city_distance_l3537_353749

theorem dormitory_to_city_distance : ∃ (D : ℝ), 
  (1/5 : ℝ) * D + (2/3 : ℝ) * D + 14 = D ∧ D = 105 := by
  sorry

end NUMINAMATH_CALUDE_dormitory_to_city_distance_l3537_353749


namespace NUMINAMATH_CALUDE_penthouse_units_l3537_353761

theorem penthouse_units (total_floors : ℕ) (regular_units : ℕ) (penthouse_floors : ℕ) (total_units : ℕ)
  (h1 : total_floors = 23)
  (h2 : regular_units = 12)
  (h3 : penthouse_floors = 2)
  (h4 : total_units = 256) :
  (total_units - (total_floors - penthouse_floors) * regular_units) / penthouse_floors = 2 := by
  sorry

end NUMINAMATH_CALUDE_penthouse_units_l3537_353761


namespace NUMINAMATH_CALUDE_undefined_fraction_l3537_353765

theorem undefined_fraction (b : ℝ) : 
  ¬ (∃ x : ℝ, x = (b - 2) / (b^2 - 9)) ↔ b = -3 ∨ b = 3 := by
sorry

end NUMINAMATH_CALUDE_undefined_fraction_l3537_353765


namespace NUMINAMATH_CALUDE_triangle_side_calculation_l3537_353709

theorem triangle_side_calculation (a b c : ℝ) (A B C : ℝ) :
  a = 4 →
  A = π / 6 →
  B = π / 3 →
  (a / Real.sin A = b / Real.sin B) →
  b = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_calculation_l3537_353709


namespace NUMINAMATH_CALUDE_max_turtles_on_board_l3537_353740

/-- Represents a rectangular board -/
structure Board :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a position on the board -/
structure Position :=
  (row : ℕ)
  (col : ℕ)

/-- Represents a turtle on the board -/
structure Turtle :=
  (position : Position)
  (last_move : Direction)

/-- Possible directions of movement -/
inductive Direction
  | Up
  | Down
  | Left
  | Right

/-- Defines a valid move for a turtle -/
def valid_move (b : Board) (t : Turtle) (new_pos : Position) : Prop :=
  (new_pos.row < b.rows) ∧
  (new_pos.col < b.cols) ∧
  ((t.last_move = Direction.Up ∨ t.last_move = Direction.Down) →
    (new_pos.row = t.position.row ∧ (new_pos.col = t.position.col + 1 ∨ new_pos.col = t.position.col - 1))) ∧
  ((t.last_move = Direction.Left ∨ t.last_move = Direction.Right) →
    (new_pos.col = t.position.col ∧ (new_pos.row = t.position.row + 1 ∨ new_pos.row = t.position.row - 1)))

/-- Defines a valid configuration of turtles on the board -/
def valid_configuration (b : Board) (turtles : List Turtle) : Prop :=
  ∀ t1 t2 : Turtle, t1 ∈ turtles → t2 ∈ turtles → t1 ≠ t2 →
    t1.position ≠ t2.position

/-- Theorem: The maximum number of turtles that can move indefinitely on a 101x99 board is 9800 -/
theorem max_turtles_on_board :
  ∀ (turtles : List Turtle),
    valid_configuration (Board.mk 101 99) turtles →
    (∀ (n : ℕ), ∃ (new_turtles : List Turtle),
      valid_configuration (Board.mk 101 99) new_turtles ∧
      turtles.length = new_turtles.length ∧
      ∀ (t : Turtle), t ∈ turtles →
        ∃ (new_t : Turtle), new_t ∈ new_turtles ∧
          valid_move (Board.mk 101 99) t new_t.position) →
    turtles.length ≤ 9800 :=
sorry

end NUMINAMATH_CALUDE_max_turtles_on_board_l3537_353740


namespace NUMINAMATH_CALUDE_convex_polygon_division_impossibility_l3537_353774

-- Define a polygon
def Polygon : Type := List (ℝ × ℝ)

-- Define a function to check if a polygon is convex
def isConvex (p : Polygon) : Prop := sorry

-- Define a function to check if a quadrilateral is non-convex
def isNonConvexQuadrilateral (q : Polygon) : Prop := sorry

-- Define a function to represent the division of a polygon into quadrilaterals
def divideIntoQuadrilaterals (p : Polygon) (qs : List Polygon) : Prop := sorry

theorem convex_polygon_division_impossibility (p : Polygon) (qs : List Polygon) :
  isConvex p → (∀ q ∈ qs, isNonConvexQuadrilateral q) → divideIntoQuadrilaterals p qs → False :=
by sorry

end NUMINAMATH_CALUDE_convex_polygon_division_impossibility_l3537_353774


namespace NUMINAMATH_CALUDE_star_theorems_l3537_353748

variable {S : Type*} [Inhabited S] [Nontrivial S]
variable (star : S → S → S)

axiom star_property : ∀ a b : S, star a (star b a) = b

theorem star_theorems :
  (∀ a b : S, star (star a (star b a)) (star a b) = a) ∧
  (∀ b : S, star b (star b b) = b) ∧
  (∀ a b : S, star (star a b) (star b (star a b)) = b) :=
by sorry

end NUMINAMATH_CALUDE_star_theorems_l3537_353748


namespace NUMINAMATH_CALUDE_CD_possible_values_l3537_353786

-- Define the points on the number line
def A : ℝ := -3
def B : ℝ := 6

-- Define the distances
def AC : ℝ := 8
def BD : ℝ := 2

-- Define the possible positions for C and D
def C1 : ℝ := A + AC
def C2 : ℝ := A - AC
def D1 : ℝ := B + BD
def D2 : ℝ := B - BD

-- Define the set of possible CD values
def CD_values : Set ℝ := {|C1 - D1|, |C1 - D2|, |C2 - D1|, |C2 - D2|}

-- Theorem statement
theorem CD_possible_values : CD_values = {3, 1, 19, 15} := by sorry

end NUMINAMATH_CALUDE_CD_possible_values_l3537_353786


namespace NUMINAMATH_CALUDE_youtube_time_is_17_minutes_l3537_353793

/-- The total time spent on YouTube per day -/
def total_youtube_time (num_videos : ℕ) (video_length : ℕ) (ad_time : ℕ) : ℕ :=
  num_videos * video_length + ad_time

/-- Theorem stating that the total time spent on YouTube is 17 minutes -/
theorem youtube_time_is_17_minutes :
  total_youtube_time 2 7 3 = 17 := by
  sorry

end NUMINAMATH_CALUDE_youtube_time_is_17_minutes_l3537_353793


namespace NUMINAMATH_CALUDE_encryption_assignment_exists_l3537_353747

/-- Represents a user on the platform -/
structure User :=
  (id : Nat)

/-- Represents an encryption key -/
structure EncryptionKey :=
  (id : Nat)

/-- Represents a messaging channel between two users -/
structure Channel :=
  (user1 : User)
  (user2 : User)
  (key : EncryptionKey)

/-- The total number of users on the platform -/
def totalUsers : Nat := 105

/-- The total number of available encryption keys -/
def totalKeys : Nat := 100

/-- A function that assigns an encryption key to a channel between two users -/
def assignKey : User → User → EncryptionKey := sorry

/-- Theorem stating that there exists a key assignment satisfying the required property -/
theorem encryption_assignment_exists :
  ∃ (assignKey : User → User → EncryptionKey),
    ∀ (u1 u2 u3 u4 : User),
      u1 ≠ u2 ∧ u1 ≠ u3 ∧ u1 ≠ u4 ∧ u2 ≠ u3 ∧ u2 ≠ u4 ∧ u3 ≠ u4 →
        ¬(assignKey u1 u2 = assignKey u1 u3 ∧
          assignKey u1 u2 = assignKey u1 u4 ∧
          assignKey u1 u2 = assignKey u2 u3 ∧
          assignKey u1 u2 = assignKey u2 u4 ∧
          assignKey u1 u2 = assignKey u3 u4) :=
by sorry

end NUMINAMATH_CALUDE_encryption_assignment_exists_l3537_353747


namespace NUMINAMATH_CALUDE_value_of_N_l3537_353730

theorem value_of_N : ∃ N : ℕ, (15 * N = 45 * 2003) ∧ (N = 6009) := by
  sorry

end NUMINAMATH_CALUDE_value_of_N_l3537_353730


namespace NUMINAMATH_CALUDE_quadratic_root_relation_l3537_353724

/-- 
Given a quadratic equation ax^2 + bx + c = 0, 
if the sum of its roots is twice their difference, 
then 3b^2 = 16ac 
-/
theorem quadratic_root_relation (a b c : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : a ≠ 0)
  (h₂ : a * x₁^2 + b * x₁ + c = 0)
  (h₃ : a * x₂^2 + b * x₂ + c = 0)
  (h₄ : x₁ + x₂ = 2 * (x₁ - x₂)) : 
  3 * b^2 = 16 * a * c := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_relation_l3537_353724


namespace NUMINAMATH_CALUDE_angle_C_is_60_degrees_angle_B_and_area_l3537_353783

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition in the problem -/
def given_condition (t : Triangle) : Prop :=
  ((t.a + t.b)^2 - t.c^2) / (3 * t.a * t.b) = 1

/-- Part 1 of the theorem -/
theorem angle_C_is_60_degrees (t : Triangle) (h : given_condition t) :
  t.C = Real.pi / 3 := by sorry

/-- Part 2 of the theorem -/
theorem angle_B_and_area (t : Triangle) 
  (h1 : t.c = Real.sqrt 3) 
  (h2 : t.b = Real.sqrt 2) 
  (h3 : t.C = Real.pi / 3) :
  t.B = Real.pi / 4 ∧ 
  (1/2 * t.b * t.c * Real.sin t.A) = (3 + Real.sqrt 3) / 4 := by sorry

end NUMINAMATH_CALUDE_angle_C_is_60_degrees_angle_B_and_area_l3537_353783


namespace NUMINAMATH_CALUDE_egg_packing_problem_l3537_353771

theorem egg_packing_problem (initial_eggs : Nat) (eggs_per_carton : Nat) (broken_eggs : Nat) :
  initial_eggs = 1000 →
  eggs_per_carton = 12 →
  broken_eggs < 12 →
  ∃ (filled_cartons : Nat), (initial_eggs - broken_eggs) = filled_cartons * eggs_per_carton →
  broken_eggs = 4 := by
  sorry

end NUMINAMATH_CALUDE_egg_packing_problem_l3537_353771


namespace NUMINAMATH_CALUDE_log_equality_l3537_353791

theorem log_equality (a b : ℝ) (h1 : a = Real.log 900 / Real.log 4) (h2 : b = Real.log 30 / Real.log 2) : a = b := by
  sorry

end NUMINAMATH_CALUDE_log_equality_l3537_353791


namespace NUMINAMATH_CALUDE_right_triangle_third_side_product_l3537_353703

theorem right_triangle_third_side_product (a b c : ℝ) : 
  (a = 6 ∧ b = 8 ∧ a^2 + b^2 = c^2) ∨ (a = 6 ∧ c = 8 ∧ a^2 + b^2 = c^2) →
  c * b = 20 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_product_l3537_353703


namespace NUMINAMATH_CALUDE_system_equation_sum_l3537_353723

theorem system_equation_sum (a b c x y z : ℝ) 
  (eq1 : 11 * x + b * y + c * z = 0)
  (eq2 : a * x + 19 * y + c * z = 0)
  (eq3 : a * x + b * y + 37 * z = 0)
  (ha : a ≠ 11)
  (hx : x ≠ 0) :
  a / (a - 11) + b / (b - 19) + c / (c - 37) = 1 := by
sorry

end NUMINAMATH_CALUDE_system_equation_sum_l3537_353723


namespace NUMINAMATH_CALUDE_total_students_l3537_353776

theorem total_students (boys girls : ℕ) (h1 : boys * 5 = girls * 6) (h2 : girls = 200) :
  boys + girls = 440 := by
  sorry

end NUMINAMATH_CALUDE_total_students_l3537_353776


namespace NUMINAMATH_CALUDE_perfect_square_polynomial_l3537_353728

theorem perfect_square_polynomial (x : ℤ) : 
  (∃ y : ℤ, x^4 + x^3 + x^2 + x + 1 = y^2) ↔ x = -1 ∨ x = 0 ∨ x = 3 :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_polynomial_l3537_353728


namespace NUMINAMATH_CALUDE_ellipse_m_range_l3537_353767

/-- The equation of an ellipse with foci on the x-axis -/
def ellipse_equation (x y m : ℝ) : Prop :=
  x^2 / (10 - m) + y^2 / (m - 2) = 1

/-- Conditions for the ellipse -/
def ellipse_conditions (m : ℝ) : Prop :=
  10 - m > 0 ∧ m - 2 > 0 ∧ 10 - m > m - 2

/-- The range of m for which the ellipse exists -/
theorem ellipse_m_range :
  ∀ m : ℝ, ellipse_conditions m ↔ 2 < m ∧ m < 6 :=
sorry

end NUMINAMATH_CALUDE_ellipse_m_range_l3537_353767


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3537_353788

theorem complex_equation_solution (z : ℂ) (h : (1 - Complex.I) * z = 1) : 
  z = (1 : ℂ) / 2 + Complex.I / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3537_353788


namespace NUMINAMATH_CALUDE_unique_valid_grid_l3537_353763

/-- Represents a 3x3 grid with letters A, B, and C -/
def Grid := Fin 3 → Fin 3 → Fin 3

/-- Checks if a row contains exactly one of each letter -/
def valid_row (g : Grid) (row : Fin 3) : Prop :=
  ∀ letter : Fin 3, ∃! col : Fin 3, g row col = letter

/-- Checks if a column contains exactly one of each letter -/
def valid_column (g : Grid) (col : Fin 3) : Prop :=
  ∀ letter : Fin 3, ∃! row : Fin 3, g row col = letter

/-- Checks if the primary diagonal contains exactly one of each letter -/
def valid_diagonal (g : Grid) : Prop :=
  ∀ letter : Fin 3, ∃! i : Fin 3, g i i = letter

/-- Checks if A is in the upper left corner -/
def a_in_corner (g : Grid) : Prop := g 0 0 = 0

/-- Checks if the grid is valid according to all conditions -/
def valid_grid (g : Grid) : Prop :=
  (∀ row : Fin 3, valid_row g row) ∧
  (∀ col : Fin 3, valid_column g col) ∧
  valid_diagonal g ∧
  a_in_corner g

/-- The main theorem: there is exactly one valid grid arrangement -/
theorem unique_valid_grid : ∃! g : Grid, valid_grid g :=
  sorry

end NUMINAMATH_CALUDE_unique_valid_grid_l3537_353763


namespace NUMINAMATH_CALUDE_solve_channels_problem_l3537_353751

def channels_problem (initial_channels : ℕ) 
                     (removed_channels : ℕ) 
                     (replaced_channels : ℕ) 
                     (sports_package_channels : ℕ) 
                     (supreme_sports_package_channels : ℕ) 
                     (final_channels : ℕ) : Prop :=
  let after_company_changes := initial_channels - removed_channels + replaced_channels
  let sports_packages_total := sports_package_channels + supreme_sports_package_channels
  let before_sports_packages := final_channels - sports_packages_total
  after_company_changes - before_sports_packages = 10

theorem solve_channels_problem : 
  channels_problem 150 20 12 8 7 147 := by
  sorry

end NUMINAMATH_CALUDE_solve_channels_problem_l3537_353751


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3537_353758

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | 0 < x ∧ x < 3}
def B : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x : ℝ | -1 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3537_353758


namespace NUMINAMATH_CALUDE_inscribed_circle_and_square_l3537_353734

theorem inscribed_circle_and_square (r : ℝ) (s : ℝ) : 
  -- Circle inscribed in a 3-4-5 right triangle
  r = 1 →
  -- Square concentric with circle and inside it
  s * Real.sqrt 2 = 2 →
  -- Side length of square is √2
  s = Real.sqrt 2 ∧
  -- Area between circle and square is π - 2
  π * r^2 - s^2 = π - 2 := by
sorry

end NUMINAMATH_CALUDE_inscribed_circle_and_square_l3537_353734


namespace NUMINAMATH_CALUDE_shaded_area_percentage_l3537_353764

/-- The size of the square grid -/
def gridSize : ℕ := 6

/-- The number of shaded squares -/
def shadedSquares : ℕ := 16

/-- The total number of squares in the grid -/
def totalSquares : ℕ := gridSize * gridSize

/-- The percentage of shaded area -/
def shadedPercentage : ℚ := (shadedSquares : ℚ) / (totalSquares : ℚ) * 100

theorem shaded_area_percentage :
  shadedPercentage = 4444 / 10000 := by sorry

end NUMINAMATH_CALUDE_shaded_area_percentage_l3537_353764


namespace NUMINAMATH_CALUDE_smallest_slope_for_tangent_circle_l3537_353773

def circle_u₁ (x y : ℝ) : Prop := x^2 + y^2 + 8*x - 18*y - 75 = 0
def circle_u₂ (x y : ℝ) : Prop := x^2 + y^2 - 8*x - 18*y + 135 = 0

def externally_tangent (x y r : ℝ) : Prop := (x - 4)^2 + (y - 9)^2 = (r + 2)^2
def internally_tangent (x y r : ℝ) : Prop := (x + 4)^2 + (y - 9)^2 = (10 - r)^2

def contains_center (b x y : ℝ) : Prop := y = b * x

theorem smallest_slope_for_tangent_circle :
  ∃ (n : ℝ), n > 0 ∧
    (∀ b : ℝ, b > 0 →
      (∃ x y r : ℝ, contains_center b x y ∧ externally_tangent x y r ∧ internally_tangent x y r) →
      b ≥ n) ∧
    n^2 = 61/24 :=
sorry

end NUMINAMATH_CALUDE_smallest_slope_for_tangent_circle_l3537_353773


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l3537_353712

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, where a > 0, b > 0, 
    and eccentricity e = √3, prove that its asymptotes are y = ±√2 x -/
theorem hyperbola_asymptotes 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (he : Real.sqrt ((a^2 + b^2) / a^2) = Real.sqrt 3) :
  ∃ (k : ℝ), k = Real.sqrt 2 ∧ 
  (∀ (x y : ℝ), (x^2 / a^2 - y^2 / b^2 = 1) → (y = k * x ∨ y = -k * x)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l3537_353712


namespace NUMINAMATH_CALUDE_leading_coefficient_of_p_l3537_353704

/-- The polynomial in question -/
def p (x : ℝ) : ℝ := 5*(x^5 - 2*x^4 + 3*x^2) - 8*(x^5 + x^3 - x) + 6*(3*x^5 - x^4 + 2)

/-- The leading coefficient of a polynomial -/
def leading_coefficient (p : ℝ → ℝ) : ℝ := 
  sorry -- Definition of leading coefficient

theorem leading_coefficient_of_p : leading_coefficient p = 15 := by
  sorry

end NUMINAMATH_CALUDE_leading_coefficient_of_p_l3537_353704


namespace NUMINAMATH_CALUDE_a_zero_iff_multiple_of_ten_sum_a_1_to_2005_l3537_353756

def a (n : ℕ+) : ℕ :=
  (7 * n.val) % 10

theorem a_zero_iff_multiple_of_ten (n : ℕ+) : a n = 0 ↔ 10 ∣ n.val := by
  sorry

theorem sum_a_1_to_2005 : (Finset.range 2005).sum (λ i => a ⟨i + 1, Nat.succ_pos i⟩) = 9025 := by
  sorry

end NUMINAMATH_CALUDE_a_zero_iff_multiple_of_ten_sum_a_1_to_2005_l3537_353756


namespace NUMINAMATH_CALUDE_inequality_proof_l3537_353777

theorem inequality_proof (a b c r : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hr : r > 1) : 
  (a / (b + c)) + (b / (c + a)) + (c / (a + b)) ≤ 
  (a^r / (b^r + c^r)) + (b^r / (c^r + a^r)) + (c^r / (a^r + b^r)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3537_353777


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_expression_l3537_353759

theorem imaginary_part_of_complex_expression : 
  let z : ℂ := 1 - Complex.I
  let expression : ℂ := z^2 + 2/z
  Complex.im expression = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_expression_l3537_353759


namespace NUMINAMATH_CALUDE_range_of_a_l3537_353708

/-- Given a line l: x + y + a = 0 and a point A(0, 2), if there exists a point M on line l 
    such that |MA|^2 + |MO|^2 = 10 (where O is the origin), then -2√2 - 1 ≤ a ≤ 2√2 - 1 -/
theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, x^2 + (-x-a)^2 + x^2 + (-x-a-2)^2 = 10) → 
  -2 * Real.sqrt 2 - 1 ≤ a ∧ a ≤ 2 * Real.sqrt 2 - 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3537_353708


namespace NUMINAMATH_CALUDE_money_left_after_distributions_and_donations_l3537_353770

def total_income : ℝ := 1200000

def children_share : ℝ := 0.2
def wife_share : ℝ := 0.3
def donation_rate : ℝ := 0.05
def num_children : ℕ := 3

theorem money_left_after_distributions_and_donations :
  let amount_to_children := children_share * total_income * num_children
  let amount_to_wife := wife_share * total_income
  let remaining_before_donation := total_income - (amount_to_children + amount_to_wife)
  let donation_amount := donation_rate * remaining_before_donation
  total_income - (amount_to_children + amount_to_wife + donation_amount) = 114000 := by
  sorry

end NUMINAMATH_CALUDE_money_left_after_distributions_and_donations_l3537_353770


namespace NUMINAMATH_CALUDE_walker_cyclist_speed_ratio_l3537_353797

/-- Given two people, a walker and a cyclist, prove that the walker is twice as slow as the cyclist
    when the cyclist's speed is three times the walker's speed. -/
theorem walker_cyclist_speed_ratio
  (S : ℝ) -- distance between home and lake
  (x : ℝ) -- walking speed
  (h1 : 0 < x) -- walking speed is positive
  (h2 : 0 < S) -- distance is positive
  (v : ℝ) -- cycling speed
  (h3 : v = 3 * x) -- cyclist speed is 3 times walker speed
  : (S / x) / (S / v) = 2 := by
  sorry

end NUMINAMATH_CALUDE_walker_cyclist_speed_ratio_l3537_353797


namespace NUMINAMATH_CALUDE_oldest_daughter_ages_l3537_353700

def is_valid_triple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a * b * c = 168

def has_ambiguous_sum (a b c : ℕ) : Prop :=
  ∃ (x y z : ℕ), is_valid_triple x y z ∧ 
  x + y + z = a + b + c ∧ (x ≠ a ∨ y ≠ b ∨ z ≠ c)

theorem oldest_daughter_ages :
  ∀ (a b c : ℕ), is_valid_triple a b c → has_ambiguous_sum a b c →
  (max a (max b c) = 12 ∨ max a (max b c) = 14 ∨ max a (max b c) = 21) :=
by sorry

end NUMINAMATH_CALUDE_oldest_daughter_ages_l3537_353700


namespace NUMINAMATH_CALUDE_complex_equation_difference_l3537_353721

theorem complex_equation_difference (a b : ℝ) :
  (a : ℂ) + b * Complex.I = (1 + 2 * Complex.I) * (3 - Complex.I) + (1 + Complex.I) / (1 - Complex.I) →
  a - b = -1 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_difference_l3537_353721


namespace NUMINAMATH_CALUDE_parabola_tangents_and_triangle_l3537_353726

/-- Parabola equation: 8y = x^2 + 16 -/
def parabola (x y : ℝ) : Prop := 8 * y = x^2 + 16

/-- Point M coordinates -/
def M : ℝ × ℝ := (3, 0)

/-- Tangent line equation -/
def tangent_line (m b : ℝ) (x y : ℝ) : Prop := y = m * x + b

/-- Points of tangency A and B -/
def A : ℝ × ℝ := (-2, 2.5)
def B : ℝ × ℝ := (8, 10)

/-- Main theorem -/
theorem parabola_tangents_and_triangle :
  ∃ (m₁ b₁ m₂ b₂ : ℝ),
    /- Tangent equations -/
    (∀ x y, tangent_line m₁ b₁ x y ↔ y = -1/2 * x + 1.5) ∧
    (∀ x y, tangent_line m₂ b₂ x y ↔ y = 2 * x - 6) ∧
    /- Angle between tangents -/
    (Real.arctan ((m₂ - m₁) / (1 + m₁ * m₂)) = Real.pi / 2) ∧
    /- Area of triangle ABM -/
    (1/2 * Real.sqrt ((M.1 - A.1)^2 + (M.2 - A.2)^2) *
     Real.sqrt ((M.1 - B.1)^2 + (M.2 - B.2)^2) = 125/4) := by
  sorry

end NUMINAMATH_CALUDE_parabola_tangents_and_triangle_l3537_353726


namespace NUMINAMATH_CALUDE_inequality_holds_l3537_353798

/-- The inequality holds for the given pairs of non-negative integers -/
theorem inequality_holds (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  ∀ k n : ℕ, 
    (1 + y^n / x^k ≥ (1 + y)^n / (1 + x)^k) ↔ 
    ((k = 0 ∧ n ≥ 0) ∨ 
     (k = 1 ∧ n = 0) ∨ 
     (k = 0 ∧ n = 0) ∨ 
     (k ≥ n - 1 ∧ n ≥ 1)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_holds_l3537_353798


namespace NUMINAMATH_CALUDE_gcd_of_repeated_digit_ints_l3537_353741

/-- Represents a four-digit positive integer -/
def FourDigitInt := {n : ℕ // 1000 ≤ n ∧ n < 10000}

/-- Constructs an eight-digit integer by repeating a four-digit integer -/
def repeatFourDigits (n : FourDigitInt) : ℕ :=
  10000 * n.val + n.val

/-- The set of all eight-digit integers formed by repeating a four-digit integer -/
def RepeatedDigitInts : Set ℕ :=
  {m | ∃ n : FourDigitInt, m = repeatFourDigits n}

/-- Theorem stating that 10001 is the greatest common divisor of all eight-digit integers
    formed by repeating a four-digit integer -/
theorem gcd_of_repeated_digit_ints :
  ∃ d : ℕ, d > 0 ∧ (∀ m ∈ RepeatedDigitInts, d ∣ m) ∧
  (∀ d' : ℕ, d' > 0 → (∀ m ∈ RepeatedDigitInts, d' ∣ m) → d' ≤ d) ∧
  d = 10001 := by
sorry

end NUMINAMATH_CALUDE_gcd_of_repeated_digit_ints_l3537_353741


namespace NUMINAMATH_CALUDE_sum_of_powers_l3537_353780

theorem sum_of_powers : -2^2003 + (-2)^2004 + 2^2005 - 2^2006 = -3 * 2^2003 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_l3537_353780


namespace NUMINAMATH_CALUDE_butter_theorem_l3537_353778

def butter_problem (total_butter : ℝ) (chocolate_chip : ℝ) (peanut_butter : ℝ) (sugar : ℝ) (oatmeal : ℝ) (spilled : ℝ) : Prop :=
  let used_butter := chocolate_chip * total_butter + peanut_butter * total_butter + sugar * total_butter + oatmeal * total_butter
  let remaining_before_spill := total_butter - used_butter
  let remaining_after_spill := remaining_before_spill - spilled
  remaining_after_spill = 0.375

theorem butter_theorem : 
  butter_problem 15 (2/5) (1/6) (1/8) (1/4) 0.5 := by
  sorry

end NUMINAMATH_CALUDE_butter_theorem_l3537_353778


namespace NUMINAMATH_CALUDE_intersection_when_a_is_one_range_of_a_when_union_is_real_l3537_353779

def A (a : ℝ) : Set ℝ := {x | a - 4 < x ∧ x < a + 4}
def B : Set ℝ := {x | x < -1 ∨ x > 5}

theorem intersection_when_a_is_one :
  A 1 ∩ B = {x | -3 < x ∧ x < -1} := by sorry

theorem range_of_a_when_union_is_real :
  (∃ a, A a ∪ B = Set.univ) ↔ ∃ a, 1 < a ∧ a < 3 := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_is_one_range_of_a_when_union_is_real_l3537_353779


namespace NUMINAMATH_CALUDE_initial_soldiers_count_l3537_353792

theorem initial_soldiers_count (provisions : ℝ) : ∃ (initial_soldiers : ℕ),
  provisions = initial_soldiers * 3 * 30 ∧
  provisions = (initial_soldiers + 528) * 2.5 * 25 ∧
  initial_soldiers = 1200 := by
sorry

end NUMINAMATH_CALUDE_initial_soldiers_count_l3537_353792


namespace NUMINAMATH_CALUDE_trajectory_and_symmetry_l3537_353794

-- Define the fixed circle F
def F (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1

-- Define the line L that the moving circle is tangent to
def L (x : ℝ) : Prop := x = -1

-- Define the trajectory C of the center P
def C (x y : ℝ) : Prop := y^2 = 8*x

-- Define symmetry about the line y = x - 1
def symmetric_about_line (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  (x₁ + x₂)/2 - ((y₁ + y₂)/2 + 1) = 0 ∧ y₁ + y₂ = x₁ + x₂ - 2

theorem trajectory_and_symmetry :
  (∀ x y, C x y ↔ ∃ r, (∀ xf yf, F xf yf → (x - xf)^2 + (y - yf)^2 = (r + 1)^2) ∧
                       (∀ xl, L xl → |x - xl| = r)) ∧
  ¬(∃ x₁ y₁ x₂ y₂, C x₁ y₁ ∧ C x₂ y₂ ∧ symmetric_about_line x₁ y₁ x₂ y₂) :=
sorry

end NUMINAMATH_CALUDE_trajectory_and_symmetry_l3537_353794


namespace NUMINAMATH_CALUDE_dandelion_seed_production_dandelion_seed_production_proof_l3537_353716

/-- Calculates the total number of seeds produced by dandelion plants in three months --/
theorem dandelion_seed_production : ℕ :=
  let initial_seeds := 50
  let germination_rate := 0.60
  let one_month_growth_rate := 0.80
  let two_month_growth_rate := 0.10
  let three_month_growth_rate := 0.10
  let one_month_seed_production := 60
  let two_month_seed_production := 40
  let three_month_seed_production := 20

  let germinated_seeds := (initial_seeds : ℚ) * germination_rate
  let one_month_plants := germinated_seeds * one_month_growth_rate
  let two_month_plants := germinated_seeds * two_month_growth_rate
  let three_month_plants := germinated_seeds * three_month_growth_rate

  let one_month_seeds := (one_month_plants * one_month_seed_production).floor
  let two_month_seeds := (two_month_plants * two_month_seed_production).floor
  let three_month_seeds := (three_month_plants * three_month_seed_production).floor

  let total_seeds := one_month_seeds + two_month_seeds + three_month_seeds

  1620

theorem dandelion_seed_production_proof : dandelion_seed_production = 1620 := by
  sorry

end NUMINAMATH_CALUDE_dandelion_seed_production_dandelion_seed_production_proof_l3537_353716


namespace NUMINAMATH_CALUDE_triangle_properties_l3537_353787

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.a * Real.sin (2 * t.B) = t.b * Real.sin t.A)
  (h2 : t.b = 3 * Real.sqrt 2)
  (h3 : (1/2) * t.a * t.c * Real.sin t.B = (3 * Real.sqrt 3) / 2) :
  t.B = π/3 ∧ t.a + t.b + t.c = 6 + 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3537_353787


namespace NUMINAMATH_CALUDE_machine_a_production_rate_l3537_353735

/-- The number of sprockets produced by each machine -/
def total_sprockets : ℕ := 880

/-- The additional time taken by Machine P compared to Machine Q -/
def time_difference : ℕ := 10

/-- The production rate of Machine Q relative to Machine A -/
def q_rate_relative_to_a : ℚ := 11/10

/-- The production rate of Machine A in sprockets per hour -/
def machine_a_rate : ℚ := 8

/-- The production rate of Machine Q in sprockets per hour -/
def machine_q_rate : ℚ := q_rate_relative_to_a * machine_a_rate

/-- The time taken by Machine Q to produce the total sprockets -/
def machine_q_time : ℚ := total_sprockets / machine_q_rate

/-- The time taken by Machine P to produce the total sprockets -/
def machine_p_time : ℚ := machine_q_time + time_difference

theorem machine_a_production_rate :
  (total_sprockets : ℚ) = machine_a_rate * machine_p_time ∧
  (total_sprockets : ℚ) = machine_q_rate * machine_q_time ∧
  machine_a_rate = 8 := by
  sorry

end NUMINAMATH_CALUDE_machine_a_production_rate_l3537_353735


namespace NUMINAMATH_CALUDE_sin_product_equality_l3537_353711

theorem sin_product_equality : 
  Real.sin (8 * π / 180) * Real.sin (40 * π / 180) * Real.sin (70 * π / 180) * Real.sin (82 * π / 180) = 3 * Real.sqrt 3 / 16 := by
  sorry

end NUMINAMATH_CALUDE_sin_product_equality_l3537_353711


namespace NUMINAMATH_CALUDE_pizza_slices_left_l3537_353755

theorem pizza_slices_left (total_slices : ℕ) (eaten_slices : ℕ) (h1 : total_slices = 32) (h2 : eaten_slices = 25) :
  total_slices - eaten_slices = 7 := by
  sorry

end NUMINAMATH_CALUDE_pizza_slices_left_l3537_353755


namespace NUMINAMATH_CALUDE_p_true_q_false_l3537_353785

-- Define the quadratic equation
def hasRealRoots (m : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + x - m = 0

-- Proposition p
theorem p_true : ∀ m : ℝ, m > 0 → hasRealRoots m :=
sorry

-- Converse of p (proposition q) is false
theorem q_false : ∃ m : ℝ, m ≥ -1 ∧ m ≤ 0 ∧ hasRealRoots m :=
sorry

end NUMINAMATH_CALUDE_p_true_q_false_l3537_353785


namespace NUMINAMATH_CALUDE_honey_ratio_proof_l3537_353766

/-- Given the conditions of James' honey production and jar requirements, 
    prove that the ratio of honey his friend is bringing jars for to the total honey produced is 1:2 -/
theorem honey_ratio_proof (hives : ℕ) (honey_per_hive : ℝ) (jar_capacity : ℝ) (james_jars : ℕ) 
  (h1 : hives = 5)
  (h2 : honey_per_hive = 20)
  (h3 : jar_capacity = 0.5)
  (h4 : james_jars = 100) :
  (↑james_jars : ℝ) / ((↑hives * honey_per_hive) / jar_capacity) = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_honey_ratio_proof_l3537_353766


namespace NUMINAMATH_CALUDE_infinite_series_sum_l3537_353753

/-- The sum of the infinite series ∑(k=1 to ∞) k^2 / 3^k is equal to 6 -/
theorem infinite_series_sum : ∑' k, (k^2 : ℝ) / 3^k = 6 := by sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l3537_353753


namespace NUMINAMATH_CALUDE_near_square_quotient_l3537_353743

/-- A natural number is a near-square if it is the product of two consecutive natural numbers. -/
def is_near_square (k : ℕ) : Prop := ∃ n : ℕ, k = n * (n + 1)

/-- Theorem stating that any near-square can be represented as the quotient of two near-squares. -/
theorem near_square_quotient (n : ℕ) : 
  is_near_square (n * (n + 1)) → 
  ∃ a b c : ℕ, 
    is_near_square a ∧ 
    is_near_square b ∧ 
    is_near_square c ∧ 
    n * (n + 1) = a / c ∧
    b = c * (n + 2) :=
sorry

end NUMINAMATH_CALUDE_near_square_quotient_l3537_353743


namespace NUMINAMATH_CALUDE_equal_area_rectangles_width_l3537_353701

/-- Given two rectangles of equal area, where one rectangle has dimensions 8 inches by 45 inches,
    and the other has a length of 15 inches, prove that the width of the second rectangle is 24 inches. -/
theorem equal_area_rectangles_width (area jordan_length jordan_width carol_length : ℝ)
    (h1 : area = jordan_length * jordan_width)
    (h2 : area = carol_length * (area / carol_length))
    (h3 : jordan_length = 8)
    (h4 : jordan_width = 45)
    (h5 : carol_length = 15) :
    area / carol_length = 24 := by
  sorry

end NUMINAMATH_CALUDE_equal_area_rectangles_width_l3537_353701


namespace NUMINAMATH_CALUDE_point_b_coordinates_l3537_353718

/-- Given a line segment AB parallel to the x-axis with length 3 and point A at coordinates (-1, 2),
    the coordinates of point B are either (-4, 2) or (2, 2). -/
theorem point_b_coordinates :
  ∀ (A B : ℝ × ℝ),
  A = (-1, 2) →
  norm (B.1 - A.1) = 3 →
  B.2 = A.2 →
  B = (-4, 2) ∨ B = (2, 2) :=
by sorry

end NUMINAMATH_CALUDE_point_b_coordinates_l3537_353718


namespace NUMINAMATH_CALUDE_unique_integer_divisible_by_16_cube_root_between_9_and_9_1_l3537_353705

theorem unique_integer_divisible_by_16_cube_root_between_9_and_9_1 :
  ∃! n : ℕ+, 
    (∃ k : ℕ, n = 16 * k) ∧ 
    9 < (n : ℝ) ^ (1/3) ∧ 
    (n : ℝ) ^ (1/3) < 9.1 ∧
    n = 736 := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_divisible_by_16_cube_root_between_9_and_9_1_l3537_353705


namespace NUMINAMATH_CALUDE_triangle_properties_l3537_353799

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Theorem statement
theorem triangle_properties (t : Triangle) : 
  -- 1. If A > B, then sin A > sin B
  (t.A > t.B → Real.sin t.A > Real.sin t.B) ∧ 
  -- 2. sin 2A = sin 2B does not necessarily imply isosceles
  ¬(Real.sin (2 * t.A) = Real.sin (2 * t.B) → t.a = t.b) ∧ 
  -- 3. a² + b² = c² does not necessarily imply isosceles
  ¬(t.a^2 + t.b^2 = t.c^2 → t.a = t.b) ∧ 
  -- 4. a² + b² > c² does not necessarily imply largest angle is obtuse
  ¬(t.a^2 + t.b^2 > t.c^2 → t.A > Real.pi/2 ∨ t.B > Real.pi/2 ∨ t.C > Real.pi/2) := by
sorry


end NUMINAMATH_CALUDE_triangle_properties_l3537_353799


namespace NUMINAMATH_CALUDE_cookie_ratio_l3537_353706

/-- Prove that the ratio of Chris's cookies to Kenny's cookies is 1:2 -/
theorem cookie_ratio (total : ℕ) (glenn : ℕ) (kenny : ℕ) (chris : ℕ)
  (h1 : total = 33)
  (h2 : glenn = 24)
  (h3 : glenn = 4 * kenny)
  (h4 : total = chris + kenny + glenn) :
  chris = kenny / 2 := by
  sorry

end NUMINAMATH_CALUDE_cookie_ratio_l3537_353706


namespace NUMINAMATH_CALUDE_max_increase_two_letters_l3537_353744

/-- Represents the sets of letters for each position in the license plate --/
structure LetterSets :=
  (first : Finset Char)
  (second : Finset Char)
  (third : Finset Char)

/-- Calculates the total number of possible license plates --/
def totalPlates (sets : LetterSets) : ℕ :=
  sets.first.card * sets.second.card * sets.third.card

/-- The initial configuration of letter sets --/
def initialSets : LetterSets :=
  { first := {'C', 'H', 'L', 'P', 'R'},
    second := {'A', 'I', 'O'},
    third := {'D', 'M', 'N', 'T'} }

/-- Theorem stating the maximum increase in license plates after adding two letters --/
theorem max_increase_two_letters :
  ∃ (newSets : LetterSets), 
    (newSets.first.card + newSets.second.card + newSets.third.card = 
     initialSets.first.card + initialSets.second.card + initialSets.third.card + 2) ∧
    (totalPlates newSets - totalPlates initialSets = 40) ∧
    ∀ (otherSets : LetterSets), 
      (otherSets.first.card + otherSets.second.card + otherSets.third.card = 
       initialSets.first.card + initialSets.second.card + initialSets.third.card + 2) →
      (totalPlates otherSets - totalPlates initialSets ≤ 40) :=
by sorry


end NUMINAMATH_CALUDE_max_increase_two_letters_l3537_353744


namespace NUMINAMATH_CALUDE_inequality_proof_l3537_353736

theorem inequality_proof (x y : ℝ) (hx : 0 < x) (hx1 : x < 1) (hy : 0 < y) (hy1 : y < 1) :
  (x^2 / (x + y)) + (y^2 / (1 - x)) + ((1 - x - y)^2 / (1 - y)) ≥ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3537_353736


namespace NUMINAMATH_CALUDE_v_1013_equals_5_l3537_353719

def g : ℕ → ℕ
| 1 => 5
| 2 => 3
| 3 => 4
| 4 => 1
| 5 => 2
| _ => 0  -- Default case for completeness

def v : ℕ → ℕ
| 0 => 3
| (n + 1) => g (v n)

theorem v_1013_equals_5 : v 1013 = 5 := by
  sorry

end NUMINAMATH_CALUDE_v_1013_equals_5_l3537_353719


namespace NUMINAMATH_CALUDE_price_reduction_achieves_target_profit_l3537_353731

/-- Represents the supermarket's mineral water sales scenario -/
structure MineralWaterSales where
  costPrice : ℝ
  initialSellingPrice : ℝ
  initialMonthlySales : ℝ
  salesIncrease : ℝ
  targetMonthlyProfit : ℝ

/-- Calculates the monthly profit given a price reduction -/
def monthlyProfit (s : MineralWaterSales) (priceReduction : ℝ) : ℝ :=
  let newPrice := s.initialSellingPrice - priceReduction
  let newSales := s.initialMonthlySales + s.salesIncrease * priceReduction
  (newPrice - s.costPrice) * newSales

/-- Theorem stating that a 7 yuan price reduction achieves the target monthly profit -/
theorem price_reduction_achieves_target_profit (s : MineralWaterSales) 
    (h1 : s.costPrice = 24)
    (h2 : s.initialSellingPrice = 36)
    (h3 : s.initialMonthlySales = 60)
    (h4 : s.salesIncrease = 10)
    (h5 : s.targetMonthlyProfit = 650) :
    monthlyProfit s 7 = s.targetMonthlyProfit := by
  sorry

end NUMINAMATH_CALUDE_price_reduction_achieves_target_profit_l3537_353731


namespace NUMINAMATH_CALUDE_inequalities_hold_l3537_353760

theorem inequalities_hold (a b c d : ℝ) 
  (h1 : a > b) (h2 : b > 0) (h3 : 0 > c) (h4 : c > d) : 
  (a - d > b - c) ∧ (a * d^2 > b * c^2) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_hold_l3537_353760


namespace NUMINAMATH_CALUDE_increasing_f_implies_a_in_range_l3537_353742

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 2 then x^3 - 2*a*x + 1 else (a-1)^x - 7

theorem increasing_f_implies_a_in_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) → 2 < a ∧ a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_increasing_f_implies_a_in_range_l3537_353742


namespace NUMINAMATH_CALUDE_cube_triangle_areas_sum_l3537_353746

/-- Represents a 3D point in space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a triangle in 3D space -/
structure Triangle3D where
  a : Point3D
  b : Point3D
  c : Point3D

/-- The vertices of a 2x2x2 cube -/
def cubeVertices : List Point3D := [
  ⟨0, 0, 0⟩, ⟨0, 0, 2⟩, ⟨0, 2, 0⟩, ⟨0, 2, 2⟩,
  ⟨2, 0, 0⟩, ⟨2, 0, 2⟩, ⟨2, 2, 0⟩, ⟨2, 2, 2⟩
]

/-- All possible triangles formed by the vertices of the cube -/
def cubeTriangles : List Triangle3D := sorry

/-- Calculates the area of a triangle in 3D space -/
def triangleArea (t : Triangle3D) : ℝ := sorry

/-- The sum of areas of all triangles formed by the cube vertices -/
def totalArea : ℝ := (cubeTriangles.map triangleArea).sum

/-- The theorem to be proved -/
theorem cube_triangle_areas_sum :
  ∃ (m n p : ℕ), totalArea = m + Real.sqrt n + Real.sqrt p ∧ m + n + p = 7728 := by
  sorry

end NUMINAMATH_CALUDE_cube_triangle_areas_sum_l3537_353746


namespace NUMINAMATH_CALUDE_saeyoung_money_conversion_l3537_353784

/-- The exchange rate from yuan to yen -/
def exchange_rate : ℝ := 17.25

/-- The value of Saeyoung's 1000 yuan bill -/
def bill_value : ℝ := 1000

/-- The value of Saeyoung's 10 yuan coin -/
def coin_value : ℝ := 10

/-- The total value of Saeyoung's Chinese money in yen -/
def total_yen : ℝ := (bill_value + coin_value) * exchange_rate

theorem saeyoung_money_conversion :
  total_yen = 17422.5 := by sorry

end NUMINAMATH_CALUDE_saeyoung_money_conversion_l3537_353784


namespace NUMINAMATH_CALUDE_expression_simplification_l3537_353713

theorem expression_simplification (y : ℝ) : 3*y + 4*y^2 + 2 - (7 - 3*y - 4*y^2) = 8*y^2 + 6*y - 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3537_353713


namespace NUMINAMATH_CALUDE_square_triangle_area_ratio_l3537_353737

/-- Given a square with side length s, where R is the midpoint of one side,
    S is the midpoint of a diagonal, and V is a vertex,
    prove that the area of triangle RSV is √2/16 of the square's area. -/
theorem square_triangle_area_ratio (s : ℝ) (h : s > 0) :
  let square_area := s^2
  let r_to_s := s / 2
  let s_to_v := s * Real.sqrt 2 / 2
  let r_to_v := s
  let triangle_height := s * Real.sqrt 2 / 4
  let triangle_area := 1 / 2 * r_to_s * triangle_height
  triangle_area / square_area = Real.sqrt 2 / 16 := by
sorry

end NUMINAMATH_CALUDE_square_triangle_area_ratio_l3537_353737


namespace NUMINAMATH_CALUDE_gcf_lcm_sum_of_numbers_l3537_353790

def numbers : List Nat := [16, 32, 48]

theorem gcf_lcm_sum_of_numbers (A B : Nat) 
  (h1 : A = Nat.gcd 16 (Nat.gcd 32 48))
  (h2 : B = Nat.lcm 16 (Nat.lcm 32 48)) : 
  A + B = 112 := by
  sorry

end NUMINAMATH_CALUDE_gcf_lcm_sum_of_numbers_l3537_353790


namespace NUMINAMATH_CALUDE_club_officers_count_l3537_353715

/-- Represents the number of ways to choose officers from a club with boys and girls --/
def chooseOfficers (boys girls : ℕ) : ℕ :=
  boys * girls * (boys - 1) + girls * boys * (girls - 1)

/-- Theorem stating the number of ways to choose officers in the given scenario --/
theorem club_officers_count :
  chooseOfficers 18 12 = 6048 := by
  sorry

end NUMINAMATH_CALUDE_club_officers_count_l3537_353715


namespace NUMINAMATH_CALUDE_daniel_gpa_probability_l3537_353733

structure GradeSystem where
  a_points : ℕ
  b_points : ℕ
  c_points : ℕ
  d_points : ℕ

structure SubjectGrades where
  math : ℕ
  history : ℕ
  english : ℕ
  science : ℕ

def gpa (gs : GradeSystem) (sg : SubjectGrades) : ℚ :=
  (sg.math + sg.history + sg.english + sg.science : ℚ) / 4

def english_prob_a : ℚ := 1/5
def english_prob_b : ℚ := 1/3
def english_prob_c : ℚ := 1 - english_prob_a - english_prob_b

def science_prob_a : ℚ := 1/3
def science_prob_b : ℚ := 1/2
def science_prob_c : ℚ := 1/6

theorem daniel_gpa_probability (gs : GradeSystem) 
  (h1 : gs.a_points = 4 ∧ gs.b_points = 3 ∧ gs.c_points = 2 ∧ gs.d_points = 1) :
  let prob_gpa_gte_3_25 := 
    english_prob_a * science_prob_a +
    english_prob_a * science_prob_b +
    english_prob_b * science_prob_a +
    english_prob_b * science_prob_b
  prob_gpa_gte_3_25 = 4/9 := by
  sorry

end NUMINAMATH_CALUDE_daniel_gpa_probability_l3537_353733


namespace NUMINAMATH_CALUDE_sum_of_max_min_g_l3537_353752

-- Define the function g(x)
def g (x : ℝ) : ℝ := |x - 1| + |x - 5| - |2*x - 10|

-- Define the domain of x
def domain (x : ℝ) : Prop := 1 ≤ x ∧ x ≤ 10

-- Theorem statement
theorem sum_of_max_min_g : 
  ∃ (max min : ℝ), (∀ x, domain x → g x ≤ max) ∧ 
                    (∀ x, domain x → min ≤ g x) ∧ 
                    (max + min = 13) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_max_min_g_l3537_353752


namespace NUMINAMATH_CALUDE_b_55_mod_56_l3537_353762

/-- b_n is the integer obtained by writing all integers from 1 to n from left to right -/
def b (n : ℕ) : ℕ := sorry

/-- The theorem states that b_55 mod 56 = 0 -/
theorem b_55_mod_56 : b 55 % 56 = 0 := by sorry

end NUMINAMATH_CALUDE_b_55_mod_56_l3537_353762


namespace NUMINAMATH_CALUDE_combined_value_of_a_and_b_l3537_353720

-- Define the conversion rate from paise to rupees
def paise_to_rupees (paise : ℚ) : ℚ := paise / 100

-- Define the value of a in rupees
def a : ℚ := (paise_to_rupees 95) / 0.005

-- Define the value of b in rupees
def b : ℚ := 3 * a - 50

-- Theorem statement
theorem combined_value_of_a_and_b : a + b = 710 := by sorry

end NUMINAMATH_CALUDE_combined_value_of_a_and_b_l3537_353720


namespace NUMINAMATH_CALUDE_value_of_two_minus_c_l3537_353710

theorem value_of_two_minus_c (c d : ℤ) 
  (h1 : 5 + c = 6 - d) 
  (h2 : 3 + d = 8 + c) : 
  2 - c = -1 := by
sorry

end NUMINAMATH_CALUDE_value_of_two_minus_c_l3537_353710


namespace NUMINAMATH_CALUDE_collatz_100th_term_l3537_353796

def collatz (n : ℕ) : ℕ :=
  if n % 2 = 0 then n / 2 else 3 * n + 1

def collatzSequence (n : ℕ) : ℕ → ℕ
  | 0 => 6
  | m + 1 => collatz (collatzSequence n m)

theorem collatz_100th_term :
  collatzSequence 100 99 = 4 := by sorry

end NUMINAMATH_CALUDE_collatz_100th_term_l3537_353796


namespace NUMINAMATH_CALUDE_integral_equals_pi_over_four_plus_e_minus_one_l3537_353732

theorem integral_equals_pi_over_four_plus_e_minus_one : 
  ∫ (x : ℝ) in (0)..(1), (Real.sqrt (1 - x^2) + Real.exp x) = π/4 + Real.exp 1 - 1 := by
  sorry

end NUMINAMATH_CALUDE_integral_equals_pi_over_four_plus_e_minus_one_l3537_353732
