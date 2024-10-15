import Mathlib

namespace NUMINAMATH_CALUDE_no_integer_solutions_l3555_355579

theorem no_integer_solutions : ¬ ∃ x : ℤ, ∃ k : ℤ, x^2 + x + 13 = 121 * k := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l3555_355579


namespace NUMINAMATH_CALUDE_arcade_spend_proof_l3555_355556

/-- Calculates the total amount spent at an arcade given the play time and cost per interval. -/
def arcade_spend (play_time_hours : ℕ) (cost_per_interval : ℚ) (interval_minutes : ℕ) : ℚ :=
  let total_minutes := play_time_hours * 60
  let num_intervals := total_minutes / interval_minutes
  num_intervals * cost_per_interval

/-- Proves that playing at an arcade for 3 hours at $0.50 per 6 minutes costs $15. -/
theorem arcade_spend_proof :
  arcade_spend 3 (1/2) 6 = 15 := by
  sorry

end NUMINAMATH_CALUDE_arcade_spend_proof_l3555_355556


namespace NUMINAMATH_CALUDE_cake_division_l3555_355571

theorem cake_division (total_cake : ℚ) (num_people : ℕ) :
  total_cake = 7/8 ∧ num_people = 4 →
  total_cake / num_people = 7/32 := by
sorry

end NUMINAMATH_CALUDE_cake_division_l3555_355571


namespace NUMINAMATH_CALUDE_cos_2α_in_second_quadrant_l3555_355597

theorem cos_2α_in_second_quadrant (α : Real) : 
  (π/2 < α ∧ α < π) →  -- α is in the second quadrant
  (Real.sin α + Real.cos α = Real.sqrt 3 / 3) → 
  Real.cos (2 * α) = -(Real.sqrt 5 / 3) := by
sorry

end NUMINAMATH_CALUDE_cos_2α_in_second_quadrant_l3555_355597


namespace NUMINAMATH_CALUDE_min_value_a_l3555_355509

theorem min_value_a (a : ℝ) : 
  (∀ x : ℝ, |x + a| - |x + 1| ≤ 2*a) ↔ a ≥ 1/3 :=
sorry

end NUMINAMATH_CALUDE_min_value_a_l3555_355509


namespace NUMINAMATH_CALUDE_angle_sum_when_product_is_four_l3555_355544

theorem angle_sum_when_product_is_four (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2) 
  (h3 : (1 + Real.tan α) * (1 + Real.tan β) = 4) : α + β = π * 3/4 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_when_product_is_four_l3555_355544


namespace NUMINAMATH_CALUDE_three_digit_swap_subtraction_l3555_355520

theorem three_digit_swap_subtraction (a b c : ℕ) : 
  a ≤ 9 → b ≤ 9 → c ≤ 9 → a ≠ 0 → a = c + 3 →
  (100 * a + 10 * b + c) - (100 * c + 10 * b + a) ≡ 7 [MOD 10] :=
by sorry

end NUMINAMATH_CALUDE_three_digit_swap_subtraction_l3555_355520


namespace NUMINAMATH_CALUDE_problem_2023_squared_minus_2024_times_2022_l3555_355569

theorem problem_2023_squared_minus_2024_times_2022 : 2023^2 - 2024 * 2022 = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_2023_squared_minus_2024_times_2022_l3555_355569


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l3555_355545

/-- Given a quadratic polynomial 7x^2 + 4x + 9, if α and β are the reciprocals of its roots,
    then their sum α + β equals -4/9 -/
theorem sum_of_reciprocals_of_roots (α β : ℝ) : 
  (∃ a b : ℝ, (7 * a^2 + 4 * a + 9 = 0) ∧ 
              (7 * b^2 + 4 * b + 9 = 0) ∧ 
              (α = 1 / a) ∧ 
              (β = 1 / b)) → 
  α + β = -4/9 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l3555_355545


namespace NUMINAMATH_CALUDE_consecutive_integers_sqrt_3_l3555_355504

theorem consecutive_integers_sqrt_3 (a b : ℤ) : 
  (b = a + 1) →  -- a and b are consecutive integers
  (↑a < Real.sqrt 3) →  -- a < sqrt(3)
  (Real.sqrt 3 < ↑b) →  -- sqrt(3) < b
  a + b = 3 := by sorry

end NUMINAMATH_CALUDE_consecutive_integers_sqrt_3_l3555_355504


namespace NUMINAMATH_CALUDE_arccos_sum_solution_l3555_355517

theorem arccos_sum_solution (x : ℝ) : 
  Real.arccos (2 * x) + Real.arccos (3 * x) = π / 2 → x = 1 / Real.sqrt 13 ∨ x = -1 / Real.sqrt 13 :=
by sorry

end NUMINAMATH_CALUDE_arccos_sum_solution_l3555_355517


namespace NUMINAMATH_CALUDE_train_distance_problem_l3555_355584

/-- Given two trains with specified lengths, speeds, and crossing time, 
    calculate the initial distance between them. -/
theorem train_distance_problem (length1 length2 speed1 speed2 crossing_time : ℝ) 
  (h1 : length1 = 100)
  (h2 : length2 = 150)
  (h3 : speed1 = 10)
  (h4 : speed2 = 15)
  (h5 : crossing_time = 60)
  (h6 : speed2 > speed1) : 
  (speed2 - speed1) * crossing_time = length1 + length2 + 50 := by
  sorry

end NUMINAMATH_CALUDE_train_distance_problem_l3555_355584


namespace NUMINAMATH_CALUDE_equidistant_point_on_x_axis_l3555_355561

theorem equidistant_point_on_x_axis :
  ∃ x : ℝ,
    (x^2 + 4*x + 4 = x^2 + 16) ∧
    (∀ y : ℝ, y ≠ x → (y^2 + 4*y + 4 ≠ y^2 + 16)) →
    x = 3 := by
  sorry

end NUMINAMATH_CALUDE_equidistant_point_on_x_axis_l3555_355561


namespace NUMINAMATH_CALUDE_largest_gcd_of_sum_1008_l3555_355581

theorem largest_gcd_of_sum_1008 :
  ∃ (a b : ℕ+), a + b = 1008 ∧ 
  ∀ (c d : ℕ+), c + d = 1008 → Nat.gcd a b ≥ Nat.gcd c d ∧
  Nat.gcd a b = 504 :=
by sorry

end NUMINAMATH_CALUDE_largest_gcd_of_sum_1008_l3555_355581


namespace NUMINAMATH_CALUDE_morse_high_school_students_l3555_355501

/-- The number of seniors at Morse High School -/
def num_seniors : ℕ := 300

/-- The percentage of seniors with cars -/
def senior_car_percentage : ℚ := 40 / 100

/-- The percentage of seniors with motorcycles -/
def senior_motorcycle_percentage : ℚ := 5 / 100

/-- The percentage of lower grade students with cars -/
def lower_car_percentage : ℚ := 10 / 100

/-- The percentage of lower grade students with motorcycles -/
def lower_motorcycle_percentage : ℚ := 3 / 100

/-- The percentage of all students with either a car or a motorcycle -/
def total_vehicle_percentage : ℚ := 20 / 100

/-- The number of students in the lower grades -/
def num_lower_grades : ℕ := 1071

theorem morse_high_school_students :
  ∃ (total_students : ℕ),
    (num_seniors + num_lower_grades = total_students) ∧
    (↑num_seniors * senior_car_percentage + 
     ↑num_seniors * senior_motorcycle_percentage +
     ↑num_lower_grades * lower_car_percentage + 
     ↑num_lower_grades * lower_motorcycle_percentage : ℚ) = 
    ↑total_students * total_vehicle_percentage :=
by sorry

end NUMINAMATH_CALUDE_morse_high_school_students_l3555_355501


namespace NUMINAMATH_CALUDE_conveyance_percentage_l3555_355543

def salary : ℝ := 5000
def food_percent : ℝ := 40
def rent_percent : ℝ := 20
def entertainment_percent : ℝ := 10
def savings : ℝ := 1000

theorem conveyance_percentage :
  let other_expenses := (food_percent + rent_percent + entertainment_percent) / 100 * salary
  let conveyance := salary - savings - other_expenses
  conveyance / salary * 100 = 10 := by sorry

end NUMINAMATH_CALUDE_conveyance_percentage_l3555_355543


namespace NUMINAMATH_CALUDE_distance_between_points_l3555_355518

theorem distance_between_points : Real.sqrt ((7 - (-5))^2 + (3 - (-2))^2) = 13 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l3555_355518


namespace NUMINAMATH_CALUDE_chocolate_division_l3555_355551

theorem chocolate_division (total : ℚ) (piles : ℕ) (h1 : total = 60 / 7) (h2 : piles = 5) :
  let pile_weight := total / piles
  let received := pile_weight
  let given_back := received / 2
  received - given_back = 6 / 7 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_division_l3555_355551


namespace NUMINAMATH_CALUDE_typing_difference_l3555_355585

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- Micah's typing speed in words per minute -/
def micah_speed : ℕ := 20

/-- Isaiah's typing speed in words per minute -/
def isaiah_speed : ℕ := 40

/-- The difference in words typed per hour between Isaiah and Micah -/
theorem typing_difference : 
  (isaiah_speed * minutes_per_hour) - (micah_speed * minutes_per_hour) = 1200 := by
  sorry

end NUMINAMATH_CALUDE_typing_difference_l3555_355585


namespace NUMINAMATH_CALUDE_machine_fill_time_l3555_355570

theorem machine_fill_time (time_A time_AB : ℝ) (time_A_pos : time_A > 0) (time_AB_pos : time_AB > 0) :
  time_A = 20 → time_AB = 12 → ∃ time_B : ℝ, time_B > 0 ∧ 1 / time_A + 1 / time_B = 1 / time_AB ∧ time_B = 30 :=
by sorry

end NUMINAMATH_CALUDE_machine_fill_time_l3555_355570


namespace NUMINAMATH_CALUDE_special_square_area_l3555_355515

/-- Square ABCD with points E on AD and F on BC, where BE = EF = FD = 20,
    AE = 2 * ED, and BF = 2 * FC -/
structure SpecialSquare where
  -- Define the side length of the square
  side : ℝ
  -- Define points E and F
  e : ℝ -- distance AE
  f : ℝ -- distance BF
  -- Conditions
  e_on_side : 0 < e ∧ e < side
  f_on_side : 0 < f ∧ f < side
  be_ef_fd : side - f + e = 20 -- BE + EF = 20
  ef_fd : e + side - f = 40 -- EF + FD = 40
  ae_twice_ed : e = 2 * (side - e)
  bf_twice_fc : f = 2 * (side - f)

/-- The area of the SpecialSquare is 720 -/
theorem special_square_area (sq : SpecialSquare) : sq.side ^ 2 = 720 :=
  sorry

end NUMINAMATH_CALUDE_special_square_area_l3555_355515


namespace NUMINAMATH_CALUDE_parabola_focus_distance_l3555_355588

/-- Theorem: For a parabola y^2 = 2px (p > 0) with vertex at origin, passing through (x₀, 2),
    if the distance from A to focus is 3 times the distance from origin to focus, then p = √2 -/
theorem parabola_focus_distance (p : ℝ) (x₀ : ℝ) (h_p_pos : p > 0) :
  (2 : ℝ)^2 = 2 * p * x₀ →  -- parabola passes through (x₀, 2)
  x₀ + p / 2 = 3 * (p / 2) →  -- |AF| = 3|OF|
  p = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_distance_l3555_355588


namespace NUMINAMATH_CALUDE_five_consecutive_integers_product_not_square_l3555_355537

theorem five_consecutive_integers_product_not_square (n : ℕ+) :
  ∃ m : ℕ, (n * (n + 1) * (n + 2) * (n + 3) * (n + 4) : ℕ) ≠ m^2 := by
  sorry

end NUMINAMATH_CALUDE_five_consecutive_integers_product_not_square_l3555_355537


namespace NUMINAMATH_CALUDE_keychain_thread_calculation_l3555_355582

theorem keychain_thread_calculation (class_friends : ℕ) (club_friends : ℕ) (total_thread : ℕ) : 
  class_friends = 6 →
  club_friends = class_friends / 2 →
  total_thread = 108 →
  total_thread / (class_friends + club_friends) = 12 :=
by sorry

end NUMINAMATH_CALUDE_keychain_thread_calculation_l3555_355582


namespace NUMINAMATH_CALUDE_potato_cooking_time_l3555_355512

/-- Given a chef cooking potatoes with the following conditions:
  - Total potatoes to cook is 15
  - Potatoes already cooked is 6
  - Time to cook the remaining potatoes is 72 minutes
  Prove that the time to cook one potato is 8 minutes. -/
theorem potato_cooking_time (total : Nat) (cooked : Nat) (remaining_time : Nat) :
  total = 15 → cooked = 6 → remaining_time = 72 → (remaining_time / (total - cooked) = 8) :=
by sorry

end NUMINAMATH_CALUDE_potato_cooking_time_l3555_355512


namespace NUMINAMATH_CALUDE_coordinates_wrt_x_axis_l3555_355527

/-- Represents a point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point across the x-axis -/
def reflectAcrossXAxis (p : Point) : Point :=
  ⟨p.x, -p.y⟩

/-- Theorem: The coordinates of point A(2,3) with respect to the x-axis are (2,-3) -/
theorem coordinates_wrt_x_axis :
  let A : Point := ⟨2, 3⟩
  reflectAcrossXAxis A = ⟨2, -3⟩ := by
  sorry

end NUMINAMATH_CALUDE_coordinates_wrt_x_axis_l3555_355527


namespace NUMINAMATH_CALUDE_opposite_of_three_l3555_355580

theorem opposite_of_three : -(3 : ℤ) = -3 := by sorry

end NUMINAMATH_CALUDE_opposite_of_three_l3555_355580


namespace NUMINAMATH_CALUDE_solution_set_f_range_of_m_l3555_355519

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 1|
def g (x m : ℝ) : ℝ := -|x + 3| + m

-- Theorem for part (1)
theorem solution_set_f (x : ℝ) : f x + x^2 - 1 > 0 ↔ x > 1 ∨ x < 0 := by sorry

-- Theorem for part (2)
theorem range_of_m (m : ℝ) : (∃ x, f x < g x m) ↔ m > 4 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_range_of_m_l3555_355519


namespace NUMINAMATH_CALUDE_cubic_roots_condition_l3555_355593

theorem cubic_roots_condition (a b c : ℝ) (α β γ : ℝ) : 
  (∀ x : ℝ, x^3 + a*x^2 + b*x + c = (x - α)*(x - β)*(x - γ)) →
  (∀ x : ℝ, x^3 + a^3*x^2 + b^3*x + c^3 = (x - α^3)*(x - β^3)*(x - γ^3)) →
  c = a*b ∧ b ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_cubic_roots_condition_l3555_355593


namespace NUMINAMATH_CALUDE_dvd_price_calculation_l3555_355514

theorem dvd_price_calculation (num_dvd : ℕ) (num_bluray : ℕ) (bluray_price : ℚ) (avg_price : ℚ) :
  num_dvd = 8 →
  num_bluray = 4 →
  bluray_price = 18 →
  avg_price = 14 →
  ∃ (dvd_price : ℚ),
    dvd_price * num_dvd + bluray_price * num_bluray = avg_price * (num_dvd + num_bluray) ∧
    dvd_price = 12 :=
by sorry

end NUMINAMATH_CALUDE_dvd_price_calculation_l3555_355514


namespace NUMINAMATH_CALUDE_triangle_properties_l3555_355540

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given triangle satisfies the condition b² + c² - a² = 2bc sin(B+C) -/
def satisfiesCondition (t : Triangle) : Prop :=
  t.b^2 + t.c^2 - t.a^2 = 2 * t.b * t.c * Real.sin (t.B + t.C)

/-- Theorem about the angle A and area of the triangle -/
theorem triangle_properties (t : Triangle) 
    (h1 : satisfiesCondition t) 
    (h2 : t.a = 2) 
    (h3 : t.B = π/3) : 
    t.A = π/4 ∧ 
    (1/2 * t.a * t.b * Real.sin t.C = (3 + Real.sqrt 3) / 2) := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l3555_355540


namespace NUMINAMATH_CALUDE_smallest_of_three_l3555_355589

theorem smallest_of_three : ∀ (a b c : ℕ), a = 10 ∧ b = 11 ∧ c = 12 → a < b ∧ a < c := by
  sorry

end NUMINAMATH_CALUDE_smallest_of_three_l3555_355589


namespace NUMINAMATH_CALUDE_impossibleSquareConstruction_l3555_355562

/-- Represents a square constructed on a chord of a unit circle -/
structure SquareOnChord where
  sideLength : ℝ
  twoVerticesOnChord : Bool
  twoVerticesOnCircumference : Bool

/-- Represents a chord of a unit circle -/
structure Chord where
  length : ℝ
  inUnitCircle : length > 0 ∧ length ≤ 2

theorem impossibleSquareConstruction (c : Chord) :
  ¬∃ (s1 s2 : SquareOnChord),
    s1.twoVerticesOnChord ∧
    s1.twoVerticesOnCircumference ∧
    s2.twoVerticesOnChord ∧
    s2.twoVerticesOnCircumference ∧
    s1.sideLength - s2.sideLength = 1 ∧
    s1.sideLength = c.length / Real.sqrt 2 ∧
    s2.sideLength = (c.length - Real.sqrt 2) / Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_impossibleSquareConstruction_l3555_355562


namespace NUMINAMATH_CALUDE_football_outcomes_count_l3555_355502

def FootballOutcome := Nat × Nat × Nat

def total_matches (outcome : FootballOutcome) : Nat :=
  outcome.1 + outcome.2.1 + outcome.2.2

def total_points (outcome : FootballOutcome) : Nat :=
  3 * outcome.1 + outcome.2.1

def is_valid_outcome (outcome : FootballOutcome) : Prop :=
  total_matches outcome = 14 ∧ total_points outcome = 19

theorem football_outcomes_count :
  ∃! n : Nat, ∃ outcomes : Finset FootballOutcome,
    outcomes.card = n ∧
    (∀ o : FootballOutcome, o ∈ outcomes ↔ is_valid_outcome o) ∧
    n = 4 := by sorry

end NUMINAMATH_CALUDE_football_outcomes_count_l3555_355502


namespace NUMINAMATH_CALUDE_diagonal_less_than_half_perimeter_l3555_355592

-- Define a quadrilateral with sides a, b, c, d and diagonal x
structure Quadrilateral :=
  (a b c d x : ℝ)
  (positive_sides : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d)
  (positive_diagonal : 0 < x)

-- Theorem: The diagonal is less than half the perimeter
theorem diagonal_less_than_half_perimeter (q : Quadrilateral) :
  q.x < (q.a + q.b + q.c + q.d) / 2 := by
  sorry

end NUMINAMATH_CALUDE_diagonal_less_than_half_perimeter_l3555_355592


namespace NUMINAMATH_CALUDE_sin_transformation_l3555_355578

theorem sin_transformation (x : ℝ) : 
  2 * Real.sin (x / 3 - π / 6) = 2 * Real.sin ((x - π / 2) / 3) := by sorry

end NUMINAMATH_CALUDE_sin_transformation_l3555_355578


namespace NUMINAMATH_CALUDE_field_completion_time_l3555_355586

theorem field_completion_time (team1_time team2_time initial_days joint_days : ℝ) : 
  team1_time = 12 →
  team2_time = 0.75 * team1_time →
  initial_days = 5 →
  (initial_days / team1_time) + joint_days * (1 / team1_time + 1 / team2_time) = 1 →
  joint_days = 3 := by
  sorry

end NUMINAMATH_CALUDE_field_completion_time_l3555_355586


namespace NUMINAMATH_CALUDE_solution_set_equality_l3555_355563

theorem solution_set_equality (a : ℝ) : 
  (∀ x, (a - 1) * x < a + 5 ↔ 2 * x < 4) → a = 7 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_equality_l3555_355563


namespace NUMINAMATH_CALUDE_base_difference_l3555_355596

/-- Converts a number from base b to base 10 -/
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * b ^ i) 0

/-- The problem statement -/
theorem base_difference : 
  let base_9_number := [1, 2, 3]  -- 321 in base 9, least significant digit first
  let base_6_number := [5, 6, 1]  -- 165 in base 6, least significant digit first
  (to_base_10 base_9_number 9) - (to_base_10 base_6_number 6) = 221 := by
  sorry


end NUMINAMATH_CALUDE_base_difference_l3555_355596


namespace NUMINAMATH_CALUDE_sin_shift_l3555_355583

theorem sin_shift (x : ℝ) : 
  Real.sin (2 * x + π / 3) = Real.sin (2 * (x + π / 6)) := by
  sorry

end NUMINAMATH_CALUDE_sin_shift_l3555_355583


namespace NUMINAMATH_CALUDE_equation_solutions_l3555_355595

theorem equation_solutions : 
  ∃ (x₁ x₂ : ℝ), 
    (x₁ > 0 ∧ x₂ > 0) ∧
    (∀ x : ℝ, x > 0 → 
      ((1/2) * (4*x^2 - 1) = (x^2 - 60*x - 20) * (x^2 + 30*x + 10)) ↔ 
      (x = x₁ ∨ x = x₂)) ∧
    x₁ = 30 + Real.sqrt 919 ∧
    x₂ = -15 + Real.sqrt 216 := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3555_355595


namespace NUMINAMATH_CALUDE_fixed_points_range_l3555_355552

/-- The function f(x) = x^2 + ax + 4 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 4

/-- A fixed point of f is a real number x such that f(x) = x -/
def is_fixed_point (a : ℝ) (x : ℝ) : Prop := f a x = x

/-- The proposition that f has exactly two different fixed points in [1,3] -/
def has_two_fixed_points (a : ℝ) : Prop :=
  ∃ (x y : ℝ), x ≠ y ∧ x ∈ Set.Icc 1 3 ∧ y ∈ Set.Icc 1 3 ∧
  is_fixed_point a x ∧ is_fixed_point a y ∧
  ∀ (z : ℝ), z ∈ Set.Icc 1 3 → is_fixed_point a z → (z = x ∨ z = y)

/-- The main theorem stating the range of a -/
theorem fixed_points_range :
  ∀ a : ℝ, has_two_fixed_points a ↔ a ∈ Set.Icc (-10/3) (-3) :=
sorry

end NUMINAMATH_CALUDE_fixed_points_range_l3555_355552


namespace NUMINAMATH_CALUDE_modulus_of_complex_number_l3555_355542

theorem modulus_of_complex_number (z : ℂ) (h : z = 3 + 4 * Complex.I) : Complex.abs z = 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_number_l3555_355542


namespace NUMINAMATH_CALUDE_max_pages_copied_l3555_355550

/-- The number of pages that can be copied given a budget and copying costs -/
def pages_copied (cost_per_4_pages : ℕ) (flat_fee : ℕ) (budget : ℕ) : ℕ :=
  ((budget - flat_fee) * 4) / cost_per_4_pages

/-- Theorem stating the maximum number of pages that can be copied under given conditions -/
theorem max_pages_copied : 
  pages_copied 7 100 3000 = 1657 := by
  sorry

end NUMINAMATH_CALUDE_max_pages_copied_l3555_355550


namespace NUMINAMATH_CALUDE_james_twitch_income_l3555_355572

/-- Calculates the monthly income from Twitch subscriptions given the subscriber counts, costs, and revenue percentages for each tier. -/
def monthly_twitch_income (tier1_subs tier2_subs tier3_subs : ℕ) 
                          (tier1_cost tier2_cost tier3_cost : ℚ) 
                          (tier1_percent tier2_percent tier3_percent : ℚ) : ℚ :=
  tier1_subs * tier1_cost * tier1_percent +
  tier2_subs * tier2_cost * tier2_percent +
  tier3_subs * tier3_cost * tier3_percent

/-- Proves that James' monthly income from Twitch subscriptions is $2065.41 given the specified conditions. -/
theorem james_twitch_income : 
  monthly_twitch_income 130 75 45 (499/100) (999/100) (2499/100) (70/100) (80/100) (90/100) = 206541/100 := by
  sorry

end NUMINAMATH_CALUDE_james_twitch_income_l3555_355572


namespace NUMINAMATH_CALUDE_factor_expression_l3555_355567

theorem factor_expression (y : ℝ) : 4 * y * (y + 2) + 6 * (y + 2) = (y + 2) * (2 * (2 * y + 3)) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l3555_355567


namespace NUMINAMATH_CALUDE_no_real_solution_l3555_355503

theorem no_real_solution :
  ¬∃ (x y z : ℝ), x^2 + 4*y*z + 2*z = 0 ∧ x + 2*x*y + 2*z^2 = 0 ∧ 2*x*z + y^2 + y + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solution_l3555_355503


namespace NUMINAMATH_CALUDE_binomial_26_6_l3555_355574

theorem binomial_26_6 (h1 : Nat.choose 24 4 = 10626)
                      (h2 : Nat.choose 24 5 = 42504)
                      (h3 : Nat.choose 24 6 = 53130) :
  Nat.choose 26 6 = 148764 := by
  sorry

end NUMINAMATH_CALUDE_binomial_26_6_l3555_355574


namespace NUMINAMATH_CALUDE_second_smallest_prime_perimeter_l3555_355590

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- A function that checks if three numbers can form a triangle -/
def isTriangle (a b c : ℕ) : Prop := a + b > c ∧ a + c > b ∧ b + c > a

/-- A function that checks if three numbers are distinct -/
def areDistinct (a b c : ℕ) : Prop := a ≠ b ∧ a ≠ c ∧ b ≠ c

/-- The main theorem stating that the second smallest perimeter of a scalene triangle
    with distinct prime sides and a prime perimeter is 29 -/
theorem second_smallest_prime_perimeter :
  ∃ (a b c : ℕ),
    isPrime a ∧ isPrime b ∧ isPrime c ∧
    areDistinct a b c ∧
    isTriangle a b c ∧
    isPrime (a + b + c) ∧
    (a + b + c = 29) ∧
    (∀ (x y z : ℕ),
      isPrime x ∧ isPrime y ∧ isPrime z ∧
      areDistinct x y z ∧
      isTriangle x y z ∧
      isPrime (x + y + z) ∧
      (x + y + z < 29) →
      (x + y + z = 23)) :=
by sorry

end NUMINAMATH_CALUDE_second_smallest_prime_perimeter_l3555_355590


namespace NUMINAMATH_CALUDE_equation_system_has_real_solution_l3555_355500

theorem equation_system_has_real_solution (x y : ℝ) 
  (h1 : 1 ≤ Real.sqrt x) (h2 : Real.sqrt x ≤ y) (h3 : y ≤ x^2) :
  ∃ (a b c : ℝ),
    a + b + c = (x + x^2 + x^4 + y + y^2 + y^4) / 2 ∧
    a * b + a * c + b * c = (x^3 + x^5 + x^6 + y^3 + y^5 + y^6) / 2 ∧
    a * b * c = (x^7 + y^7) / 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_system_has_real_solution_l3555_355500


namespace NUMINAMATH_CALUDE_apple_picking_ratio_l3555_355559

/-- Represents the number of apples picked in each hour -/
structure ApplePicking where
  first_hour : ℕ
  second_hour : ℕ
  third_hour : ℕ

/-- Calculates the total number of apples picked -/
def total_apples (a : ApplePicking) : ℕ :=
  a.first_hour + a.second_hour + a.third_hour

/-- Theorem: The ratio of apples picked in the second hour to the first hour is 2:1 -/
theorem apple_picking_ratio (a : ApplePicking) :
  a.first_hour = 66 →
  a.third_hour = a.first_hour / 3 →
  total_apples a = 220 →
  a.second_hour = 2 * a.first_hour :=
by
  sorry


end NUMINAMATH_CALUDE_apple_picking_ratio_l3555_355559


namespace NUMINAMATH_CALUDE_no_three_digit_odd_divisible_by_six_l3555_355558

theorem no_three_digit_odd_divisible_by_six : 
  ¬ ∃ n : ℕ, 
    (100 ≤ n ∧ n ≤ 999) ∧ 
    (∀ d, d ∈ n.digits 10 → d % 2 = 1 ∧ d > 4) ∧ 
    n % 6 = 0 :=
by sorry

end NUMINAMATH_CALUDE_no_three_digit_odd_divisible_by_six_l3555_355558


namespace NUMINAMATH_CALUDE_scientific_notation_216000_l3555_355526

theorem scientific_notation_216000 : 216000 = 2.16 * (10 ^ 5) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_216000_l3555_355526


namespace NUMINAMATH_CALUDE_ellipse_and_line_equation_l3555_355575

noncomputable section

-- Define the ellipse E
def ellipse (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the hyperbola
def hyperbola (m n : ℝ) (x y : ℝ) : Prop := x^2 / m - y^2 / n = 1

-- Define eccentricity
def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

theorem ellipse_and_line_equation 
  (a b m n e₁ e₂ : ℝ)
  (h₁ : a > b)
  (h₂ : b > 0)
  (h₃ : m > 0)
  (h₄ : n > 0)
  (h₅ : b = Real.sqrt 3)
  (h₆ : n / m = 3)
  (h₇ : e₁ * e₂ = 1)
  (h₈ : e₂ = Real.sqrt 4)
  (h₉ : e₁ = eccentricity a b)
  (P : ℝ × ℝ)
  (h₁₀ : P = (-1, 3/2))
  (S₁ S₂ : ℝ)
  (h₁₁ : S₁ = 6 * S₂) :
  (∀ x y, ellipse a b x y ↔ x^2 / 4 + y^2 / 3 = 1) ∧
  (∃ k, k = Real.sqrt 6 / 2 ∧ 
    (∀ x y, y = k * x + 1 ∨ y = -k * x + 1)) :=
by sorry

end

end NUMINAMATH_CALUDE_ellipse_and_line_equation_l3555_355575


namespace NUMINAMATH_CALUDE_mistaken_calculation_correction_l3555_355577

theorem mistaken_calculation_correction (x : ℤ) : 
  x - 15 + 27 = 41 → x - 27 + 15 = 17 := by
  sorry

end NUMINAMATH_CALUDE_mistaken_calculation_correction_l3555_355577


namespace NUMINAMATH_CALUDE_cosine_square_root_pi_eighths_l3555_355548

theorem cosine_square_root_pi_eighths :
  Real.sqrt ((3 - Real.cos (π / 8) ^ 2) * (3 - Real.cos (3 * π / 8) ^ 2)) = 3 * Real.sqrt 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_cosine_square_root_pi_eighths_l3555_355548


namespace NUMINAMATH_CALUDE_distance_to_focus_l3555_355507

/-- Given a parabola x = (1/2)y², prove that the distance from a point P(1, y) on the parabola to its focus F is 3/2 -/
theorem distance_to_focus (y : ℝ) (h : 1 = (1/2) * y^2) : 
  let p : ℝ × ℝ := (1, y)
  let f : ℝ × ℝ := ((1/4), 0)  -- Focus of the parabola x = (1/2)y²
  ‖p - f‖ = 3/2 := by sorry

end NUMINAMATH_CALUDE_distance_to_focus_l3555_355507


namespace NUMINAMATH_CALUDE_cubic_equation_c_value_l3555_355573

/-- Given a cubic equation with coefficients a, b, c, d, returns whether it has three distinct positive roots -/
def has_three_distinct_positive_roots (a b c d : ℝ) : Prop := sorry

/-- Given three real numbers, returns their sum of base-3 logarithms -/
def sum_of_log3 (x y z : ℝ) : ℝ := sorry

theorem cubic_equation_c_value (c d : ℝ) :
  has_three_distinct_positive_roots 4 (5 * c) (3 * d) c →
  ∃ (x y z : ℝ), sum_of_log3 x y z = 3 ∧ 
    4 * x^3 + 5 * c * x^2 + 3 * d * x + c = 0 ∧
    4 * y^3 + 5 * c * y^2 + 3 * d * y + c = 0 ∧
    4 * z^3 + 5 * c * z^2 + 3 * d * z + c = 0 →
  c = -108 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_c_value_l3555_355573


namespace NUMINAMATH_CALUDE_complex_inequality_nonexistence_l3555_355535

theorem complex_inequality_nonexistence : 
  ∀ (a b c : ℂ) (h : ℕ), a ≠ 0 → b ≠ 0 → c ≠ 0 → 
  ∃ (k l m : ℤ), (abs k + abs l + abs m ≥ 1996) ∧ 
  (Complex.abs (1 + k * a + l * b + m * c) ≤ 1 / h) := by
  sorry

end NUMINAMATH_CALUDE_complex_inequality_nonexistence_l3555_355535


namespace NUMINAMATH_CALUDE_small_rectangle_perimeter_l3555_355521

/-- Given a square with perimeter 160 units, divided into two congruent rectangles,
    with one of those rectangles further divided into three smaller congruent rectangles,
    the perimeter of one of the three smaller congruent rectangles is equal to 2 * (20 + 40/3) units. -/
theorem small_rectangle_perimeter (s : ℝ) (h1 : s > 0) (h2 : 4 * s = 160) :
  2 * (s / 2 + s / 6) = 2 * (20 + 40 / 3) :=
sorry

end NUMINAMATH_CALUDE_small_rectangle_perimeter_l3555_355521


namespace NUMINAMATH_CALUDE_large_planter_capacity_l3555_355541

/-- Proves that each large planter can hold 20 seeds given the problem conditions -/
theorem large_planter_capacity
  (total_seeds : ℕ)
  (num_large_planters : ℕ)
  (small_planter_capacity : ℕ)
  (num_small_planters : ℕ)
  (h1 : total_seeds = 200)
  (h2 : num_large_planters = 4)
  (h3 : small_planter_capacity = 4)
  (h4 : num_small_planters = 30)
  : (total_seeds - num_small_planters * small_planter_capacity) / num_large_planters = 20 := by
  sorry

end NUMINAMATH_CALUDE_large_planter_capacity_l3555_355541


namespace NUMINAMATH_CALUDE_max_value_of_f_range_of_k_l3555_355511

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x - 9 / x

theorem max_value_of_f :
  ∃ (x : ℝ), x > 0 ∧ f x = (1 : ℝ) / Real.exp 10 ∧ ∀ y > 0, f y ≤ f x :=
sorry

theorem range_of_k :
  ∀ k : ℝ,
  (∀ x : ℝ, x ≥ 1 → x^2 * ((Real.log x) / x - k / x) + 1 / (x + 1) ≥ 0) →
  (∀ x : ℝ, x ≥ 1 → k ≥ (1/2) * x^2 + (Real.exp 2 - 2) * x - Real.exp x - 7) →
  (Real.exp 2 - 9 ≤ k ∧ k ≤ 1/2) :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_range_of_k_l3555_355511


namespace NUMINAMATH_CALUDE_chess_tournament_games_l3555_355554

/-- The number of games played in a chess tournament -/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a chess tournament with 8 players, where each player plays every other player 
    exactly once, the total number of games played is 28. -/
theorem chess_tournament_games :
  num_games 8 = 28 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l3555_355554


namespace NUMINAMATH_CALUDE_cube_root_monotone_l3555_355522

theorem cube_root_monotone {a b : ℝ} (h : a > b) : a^(1/3) > b^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_monotone_l3555_355522


namespace NUMINAMATH_CALUDE_jean_initial_candy_l3555_355513

/-- The number of candy pieces Jean had initially -/
def initial_candy : ℕ := sorry

/-- The number of candy pieces Jean gave to her first friend -/
def first_friend : ℕ := 18

/-- The number of candy pieces Jean gave to her second friend -/
def second_friend : ℕ := 12

/-- The number of candy pieces Jean gave to her third friend -/
def third_friend : ℕ := 25

/-- The number of candy pieces Jean bought -/
def bought : ℕ := 10

/-- The number of candy pieces Jean ate -/
def ate : ℕ := 7

/-- The number of candy pieces Jean has left -/
def left : ℕ := 16

theorem jean_initial_candy : 
  initial_candy = first_friend + second_friend + third_friend + left + ate - bought :=
by sorry

end NUMINAMATH_CALUDE_jean_initial_candy_l3555_355513


namespace NUMINAMATH_CALUDE_range_of_a_l3555_355539

theorem range_of_a (a : ℝ) : 
  Real.sqrt ((2*a - 1)^2) = 1 - 2*a → a ≤ 1/2 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l3555_355539


namespace NUMINAMATH_CALUDE_range_of_a_l3555_355598

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0
def q (a : ℝ) : Prop := ∃ x₀ : ℝ, x₀^2 + (a-1)*x₀ + 1 < 0

-- Define the theorem
theorem range_of_a :
  ∀ a : ℝ, (p a ∨ q a) ∧ ¬(p a ∧ q a) →
  (a ∈ Set.Icc (-1) 1) ∨ (a > 3) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3555_355598


namespace NUMINAMATH_CALUDE_interval_condition_l3555_355553

theorem interval_condition (x : ℝ) : 
  (2 < 4 * x ∧ 4 * x < 3) ∧ (2 < 5 * x ∧ 5 * x < 3) ↔ 1/2 < x ∧ x < 3/5 :=
by sorry

end NUMINAMATH_CALUDE_interval_condition_l3555_355553


namespace NUMINAMATH_CALUDE_april_production_l3555_355532

/-- Calculates the production after n months given an initial production and monthly growth rate -/
def production_after_months (initial_production : ℕ) (growth_rate : ℝ) (months : ℕ) : ℝ :=
  initial_production * (1 + growth_rate) ^ months

/-- Proves that the production in April is 926,100 pencils given the initial conditions -/
theorem april_production :
  let initial_production := 800000
  let growth_rate := 0.05
  let months := 3
  ⌊production_after_months initial_production growth_rate months⌋ = 926100 := by
  sorry

end NUMINAMATH_CALUDE_april_production_l3555_355532


namespace NUMINAMATH_CALUDE_fraction_of_total_l3555_355536

theorem fraction_of_total (total : ℚ) (r_amount : ℚ) : 
  total = 9000 → r_amount = 3600 → r_amount / total = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_total_l3555_355536


namespace NUMINAMATH_CALUDE_line_circle_distance_theorem_l3555_355508

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a 2D plane -/
structure Line where
  point : ℝ × ℝ
  direction : ℝ × ℝ

/-- Distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- Length of tangent from a point to a circle -/
def tangentLength (p : ℝ × ℝ) (c : Circle) : ℝ := sorry

/-- Whether a line intersects a circle -/
def intersects (l : Line) (c : Circle) : Prop := sorry

/-- Whether a point is on a line -/
def onLine (p : ℝ × ℝ) (l : Line) : Prop := sorry

theorem line_circle_distance_theorem (c : Circle) (l : Line) :
  (¬ intersects l c) →
  (∀ (A B : ℝ × ℝ), onLine A l → onLine B l →
    (distance A B > |tangentLength A c - tangentLength B c| ∧
     distance A B < tangentLength A c + tangentLength B c)) ∧
  (∃ (A B : ℝ × ℝ), onLine A l → onLine B l →
    (distance A B ≤ |tangentLength A c - tangentLength B c| ∨
     distance A B ≥ tangentLength A c + tangentLength B c) →
    intersects l c) :=
by sorry

end NUMINAMATH_CALUDE_line_circle_distance_theorem_l3555_355508


namespace NUMINAMATH_CALUDE_emily_contribution_l3555_355538

/-- Proves that Emily needs to contribute 3 more euros to buy the pie -/
theorem emily_contribution (pie_cost : ℝ) (emily_usd : ℝ) (berengere_euro : ℝ) (exchange_rate : ℝ) :
  pie_cost = 15 →
  emily_usd = 10 →
  berengere_euro = 3 →
  exchange_rate = 1.1 →
  ∃ (emily_extra : ℝ), emily_extra = 3 ∧ 
    pie_cost = berengere_euro + (emily_usd / exchange_rate) + emily_extra :=
by sorry

end NUMINAMATH_CALUDE_emily_contribution_l3555_355538


namespace NUMINAMATH_CALUDE_geometric_arithmetic_ratio_l3555_355549

/-- Given a geometric sequence with positive terms where a₂, ½a₃, and a₁ form an arithmetic sequence,
    the ratio (a₄ + a₅)/(a₃ + a₄) equals (1 + √5)/2. -/
theorem geometric_arithmetic_ratio (a : ℕ → ℝ) (h_pos : ∀ n, a n > 0)
  (h_geom : ∃ q : ℝ, q > 0 ∧ ∀ n, a (n + 1) = q * a n)
  (h_arith : a 2 - a 1 = (1/2 : ℝ) * a 3 - a 2) :
  (a 4 + a 5) / (a 3 + a 4) = (1 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_ratio_l3555_355549


namespace NUMINAMATH_CALUDE_money_problem_l3555_355568

theorem money_problem (a b : ℚ) (h1 : 7 * a + b = 89) (h2 : 4 * a - b = 38) :
  a = 127 / 11 ∧ b = 90 / 11 := by
  sorry

end NUMINAMATH_CALUDE_money_problem_l3555_355568


namespace NUMINAMATH_CALUDE_area_of_region_is_5_25_l3555_355506

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The region defined by the given inequalities -/
def Region : Set Point :=
  {p : Point | p.y > 3 * p.x ∧ p.y > 5 - 2 * p.x ∧ p.y < 6}

/-- The area of the region -/
noncomputable def areaOfRegion : ℝ := sorry

/-- Theorem stating that the area of the region is 5.25 square units -/
theorem area_of_region_is_5_25 : areaOfRegion = 5.25 := by sorry

end NUMINAMATH_CALUDE_area_of_region_is_5_25_l3555_355506


namespace NUMINAMATH_CALUDE_gas_cost_proof_l3555_355566

/-- The total cost of gas for a trip to New York City -/
def total_cost : ℝ := 82.50

/-- The number of friends initially splitting the cost -/
def initial_friends : ℕ := 3

/-- The number of friends who joined later -/
def additional_friends : ℕ := 2

/-- The total number of friends after more joined -/
def total_friends : ℕ := initial_friends + additional_friends

/-- The amount by which each original friend's cost decreased -/
def cost_decrease : ℝ := 11

theorem gas_cost_proof :
  (total_cost / initial_friends) - (total_cost / total_friends) = cost_decrease :=
sorry

end NUMINAMATH_CALUDE_gas_cost_proof_l3555_355566


namespace NUMINAMATH_CALUDE_ball_hits_ground_at_calculated_time_l3555_355530

/-- The time when a ball hits the ground, given its height equation -/
def ball_ground_time : ℝ :=
  let initial_height : ℝ := 180
  let initial_velocity : ℝ := -32  -- negative because it's downward
  let release_delay : ℝ := 1
  let height (t : ℝ) : ℝ := -16 * (t - release_delay)^2 - 32 * (t - release_delay) + initial_height
  3.5

/-- Theorem stating that the ball hits the ground at the calculated time -/
theorem ball_hits_ground_at_calculated_time :
  let height (t : ℝ) : ℝ := -16 * (t - 1)^2 - 32 * (t - 1) + 180
  height ball_ground_time = 0 := by
  sorry

end NUMINAMATH_CALUDE_ball_hits_ground_at_calculated_time_l3555_355530


namespace NUMINAMATH_CALUDE_game_solvable_l3555_355564

/-- The game state, representing the positions of the red and blue beads -/
structure GameState where
  red : ℚ
  blue : ℚ

/-- The possible moves in the game -/
inductive Move
  | Red (k : ℤ)
  | Blue (k : ℤ)

/-- Apply a move to the game state -/
def applyMove (r : ℚ) (state : GameState) (move : Move) : GameState :=
  match move with
  | Move.Red k => 
      { red := state.blue + r^k * (state.red - state.blue),
        blue := state.blue }
  | Move.Blue k => 
      { red := state.red,
        blue := state.red + r^k * (state.blue - state.red) }

/-- A sequence of moves -/
def MoveSequence := List Move

/-- Apply a sequence of moves to the initial game state -/
def applyMoveSequence (r : ℚ) (moves : MoveSequence) : GameState :=
  moves.foldl (applyMove r) { red := 0, blue := 1 }

/-- The main theorem -/
theorem game_solvable (r : ℚ) : 
  (∃ (moves : MoveSequence), moves.length ≤ 2021 ∧ (applyMoveSequence r moves).red = 1) ↔ 
  (∃ (m : ℕ), m ≥ 1 ∧ m ≤ 1010 ∧ r = (m + 1) / m) :=
sorry

end NUMINAMATH_CALUDE_game_solvable_l3555_355564


namespace NUMINAMATH_CALUDE_pears_picked_total_l3555_355560

/-- The number of pears Alyssa picked -/
def alyssa_pears : ℕ := 42

/-- The number of pears Nancy picked -/
def nancy_pears : ℕ := 17

/-- The total number of pears picked -/
def total_pears : ℕ := alyssa_pears + nancy_pears

theorem pears_picked_total : total_pears = 59 := by
  sorry

end NUMINAMATH_CALUDE_pears_picked_total_l3555_355560


namespace NUMINAMATH_CALUDE_solution_x_squared_equals_three_l3555_355524

theorem solution_x_squared_equals_three :
  ∀ x : ℝ, x^2 = 3 ↔ x = Real.sqrt 3 ∨ x = -Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_solution_x_squared_equals_three_l3555_355524


namespace NUMINAMATH_CALUDE_binomial_expansion_example_l3555_355576

theorem binomial_expansion_example : 8^3 + 3*(8^2)*2 + 3*8*(2^2) + 2^3 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_example_l3555_355576


namespace NUMINAMATH_CALUDE_evening_ticket_price_l3555_355565

/-- The cost of an evening movie ticket --/
def evening_ticket_cost : ℝ := 10

/-- The cost of a large popcorn & drink combo --/
def combo_cost : ℝ := 10

/-- The discount rate for tickets during the special offer --/
def ticket_discount_rate : ℝ := 0.2

/-- The discount rate for food combos during the special offer --/
def combo_discount_rate : ℝ := 0.5

/-- The amount saved by going to the earlier movie --/
def savings : ℝ := 7

theorem evening_ticket_price :
  evening_ticket_cost = 10 ∧
  combo_cost = 10 ∧
  ticket_discount_rate = 0.2 ∧
  combo_discount_rate = 0.5 ∧
  savings = 7 →
  evening_ticket_cost + combo_cost - 
  (evening_ticket_cost * (1 - ticket_discount_rate) + combo_cost * (1 - combo_discount_rate)) = savings :=
by sorry

end NUMINAMATH_CALUDE_evening_ticket_price_l3555_355565


namespace NUMINAMATH_CALUDE_min_sum_squares_addends_of_18_l3555_355523

theorem min_sum_squares_addends_of_18 :
  ∀ x y : ℝ, x + y = 18 → x^2 + y^2 ≥ 2 * 9^2 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_addends_of_18_l3555_355523


namespace NUMINAMATH_CALUDE_exactly_one_greater_than_one_l3555_355557

theorem exactly_one_greater_than_one 
  (a b c : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c)
  (prod_one : a * b * c = 1)
  (sum_greater : a + b + c > 1/a + 1/b + 1/c) :
  (a > 1 ∧ b ≤ 1 ∧ c ≤ 1) ∨ 
  (a ≤ 1 ∧ b > 1 ∧ c ≤ 1) ∨ 
  (a ≤ 1 ∧ b ≤ 1 ∧ c > 1) :=
by sorry

end NUMINAMATH_CALUDE_exactly_one_greater_than_one_l3555_355557


namespace NUMINAMATH_CALUDE_consecutive_product_l3555_355505

theorem consecutive_product (t : ℤ) :
  let n : ℤ := t * (t + 1) - 1
  (n^2 - 1 : ℤ) = (t - 1) * t * (t + 1) * (t + 2) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_product_l3555_355505


namespace NUMINAMATH_CALUDE_stephanie_oranges_l3555_355525

/-- Represents the number of store visits -/
def store_visits : ℕ := 8

/-- Represents the total number of oranges bought -/
def total_oranges : ℕ := 16

/-- Represents the number of oranges bought per visit -/
def oranges_per_visit : ℕ := total_oranges / store_visits

/-- Theorem stating that Stephanie buys 2 oranges each time she goes to the store -/
theorem stephanie_oranges : oranges_per_visit = 2 := by
  sorry

end NUMINAMATH_CALUDE_stephanie_oranges_l3555_355525


namespace NUMINAMATH_CALUDE_chinese_coin_problem_l3555_355510

/-- Represents an arithmetic sequence of 5 terms -/
structure ArithmeticSequence :=
  (a : ℚ) -- First term
  (d : ℚ) -- Common difference

/-- Properties of the specific arithmetic sequence in the problem -/
def ProblemSequence (seq : ArithmeticSequence) : Prop :=
  -- Sum of all terms is 5
  seq.a - 2*seq.d + seq.a - seq.d + seq.a + seq.a + seq.d + seq.a + 2*seq.d = 5 ∧
  -- Sum of first two terms equals sum of last three terms
  seq.a - 2*seq.d + seq.a - seq.d = seq.a + seq.a + seq.d + seq.a + 2*seq.d

theorem chinese_coin_problem (seq : ArithmeticSequence) 
  (h : ProblemSequence seq) : seq.a - seq.d = 7/6 := by
  sorry

end NUMINAMATH_CALUDE_chinese_coin_problem_l3555_355510


namespace NUMINAMATH_CALUDE_sum_of_three_consecutive_cubes_divisible_by_nine_l3555_355534

theorem sum_of_three_consecutive_cubes_divisible_by_nine (n : ℤ) :
  ∃ k : ℤ, n^3 + (n+1)^3 + (n+2)^3 = 9 * k := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_consecutive_cubes_divisible_by_nine_l3555_355534


namespace NUMINAMATH_CALUDE_inequality_proof_l3555_355528

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : c < 0) : a / c < b / c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3555_355528


namespace NUMINAMATH_CALUDE_water_bottles_left_l3555_355591

theorem water_bottles_left (initial_bottles : ℕ) (bottles_drunk : ℕ) : 
  initial_bottles = 301 → bottles_drunk = 144 → initial_bottles - bottles_drunk = 157 := by
  sorry

end NUMINAMATH_CALUDE_water_bottles_left_l3555_355591


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3555_355529

theorem sqrt_equation_solution :
  ∃! z : ℚ, Real.sqrt (7 - 5 * z) = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3555_355529


namespace NUMINAMATH_CALUDE_arun_speed_l3555_355533

theorem arun_speed (arun_speed : ℝ) (anil_speed : ℝ) : 
  (30 / arun_speed = 30 / anil_speed + 2) →
  (30 / (2 * arun_speed) = 30 / anil_speed - 1) →
  arun_speed = Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_arun_speed_l3555_355533


namespace NUMINAMATH_CALUDE_arcade_change_machine_l3555_355547

theorem arcade_change_machine (total_value : ℕ) (one_dollar_bills : ℕ) : 
  total_value = 300 → one_dollar_bills = 175 → 
  ∃ (five_dollar_bills : ℕ), 
    one_dollar_bills + five_dollar_bills = 200 ∧ 
    one_dollar_bills + 5 * five_dollar_bills = total_value :=
by sorry

end NUMINAMATH_CALUDE_arcade_change_machine_l3555_355547


namespace NUMINAMATH_CALUDE_sum_of_products_l3555_355516

theorem sum_of_products (x y z : ℝ) 
  (sum_condition : x + y + z = 20) 
  (sum_squares_condition : x^2 + y^2 + z^2 = 200) : 
  x*y + x*z + y*z = 100 := by sorry

end NUMINAMATH_CALUDE_sum_of_products_l3555_355516


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3555_355555

/-- Given a geometric sequence {a_n} with common ratio q, if the sum of the first 3 terms is 7
    and the sum of the first 6 terms is 63, then q = 2. -/
theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = q * a n) →  -- a_n is a geometric sequence with common ratio q
  (a 1 + a 2 + a 3 = 7) →       -- Sum of first 3 terms is 7
  (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 = 63) →  -- Sum of first 6 terms is 63
  q = 2 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3555_355555


namespace NUMINAMATH_CALUDE_triangle_3_4_5_l3555_355587

/-- A function that checks if three numbers can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Theorem stating that the line segments 3, 4, and 5 can form a triangle -/
theorem triangle_3_4_5 : can_form_triangle 3 4 5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_3_4_5_l3555_355587


namespace NUMINAMATH_CALUDE_billy_score_problem_l3555_355594

/-- Billy's video game score problem -/
theorem billy_score_problem (old_score : ℕ) (rounds : ℕ) : 
  old_score = 725 → rounds = 363 → (old_score + 1) / rounds = 2 := by
  sorry

end NUMINAMATH_CALUDE_billy_score_problem_l3555_355594


namespace NUMINAMATH_CALUDE_perfect_square_factors_of_360_l3555_355531

def prime_factorization (n : ℕ) : List (ℕ × ℕ) := sorry

def is_perfect_square (n : ℕ) : Prop := sorry

def count_perfect_square_factors (n : ℕ) : ℕ := sorry

theorem perfect_square_factors_of_360 :
  let factorization := prime_factorization 360
  (factorization = [(2, 3), (3, 2), (5, 1)]) →
  count_perfect_square_factors 360 = 4 := by sorry

end NUMINAMATH_CALUDE_perfect_square_factors_of_360_l3555_355531


namespace NUMINAMATH_CALUDE_james_total_score_l3555_355599

-- Define the number of field goals and shots
def field_goals : ℕ := 13
def shots : ℕ := 20

-- Define the point values for field goals and shots
def field_goal_points : ℕ := 3
def shot_points : ℕ := 2

-- Define the total points scored
def total_points : ℕ := field_goals * field_goal_points + shots * shot_points

-- Theorem stating that the total points scored is 79
theorem james_total_score : total_points = 79 := by
  sorry

end NUMINAMATH_CALUDE_james_total_score_l3555_355599


namespace NUMINAMATH_CALUDE_donut_selection_problem_l3555_355546

/-- The number of ways to select n items from k types with at least one of each type -/
def selectWithMinimum (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose (n - k + k - 1) (k - 1)

/-- The problem statement -/
theorem donut_selection_problem :
  selectWithMinimum 6 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_donut_selection_problem_l3555_355546
