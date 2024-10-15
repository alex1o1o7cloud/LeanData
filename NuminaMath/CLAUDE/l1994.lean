import Mathlib

namespace NUMINAMATH_CALUDE_complex_sum_equality_l1994_199437

theorem complex_sum_equality :
  let B : ℂ := 3 - 2*I
  let Q : ℂ := 1 + 3*I
  let R : ℂ := -2 + 4*I
  let T : ℂ := 5 - 3*I
  B + Q + R + T = 7 + 2*I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_sum_equality_l1994_199437


namespace NUMINAMATH_CALUDE_amanda_marbles_l1994_199495

theorem amanda_marbles (katrina_marbles : ℕ) (amanda_marbles : ℕ) (mabel_marbles : ℕ) : 
  mabel_marbles = 5 * katrina_marbles →
  mabel_marbles = 85 →
  mabel_marbles = amanda_marbles + 63 →
  2 * katrina_marbles - amanda_marbles = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_amanda_marbles_l1994_199495


namespace NUMINAMATH_CALUDE_quadrilateral_bd_length_l1994_199407

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the length function
def length (p q : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem quadrilateral_bd_length (ABCD : Quadrilateral) : 
  length ABCD.A ABCD.B = 4 →
  length ABCD.B ABCD.C = 14 →
  length ABCD.C ABCD.D = 4 →
  length ABCD.D ABCD.A = 7 →
  ∃ (n : ℕ), length ABCD.B ABCD.D = n →
  length ABCD.B ABCD.D = 11 := by
sorry

end NUMINAMATH_CALUDE_quadrilateral_bd_length_l1994_199407


namespace NUMINAMATH_CALUDE_positive_root_condition_negative_root_condition_zero_root_condition_l1994_199442

-- Define the equation ax = b - c
def equation (a b c x : ℝ) : Prop := a * x = b - c

-- Theorem for positive root condition
theorem positive_root_condition (a b c : ℝ) :
  (∃ x > 0, equation a b c x) ↔ (a > 0 ∧ b > c) ∨ (a < 0 ∧ b < c) :=
sorry

-- Theorem for negative root condition
theorem negative_root_condition (a b c : ℝ) :
  (∃ x < 0, equation a b c x) ↔ (a > 0 ∧ b < c) ∨ (a < 0 ∧ b > c) :=
sorry

-- Theorem for zero root condition
theorem zero_root_condition (a b c : ℝ) :
  (∃ x, x = 0 ∧ equation a b c x) ↔ (a ≠ 0 ∧ b = c) :=
sorry

end NUMINAMATH_CALUDE_positive_root_condition_negative_root_condition_zero_root_condition_l1994_199442


namespace NUMINAMATH_CALUDE_school_average_gpa_l1994_199428

theorem school_average_gpa (gpa_6th : ℝ) (gpa_7th : ℝ) (gpa_8th : ℝ)
  (h1 : gpa_6th = 93)
  (h2 : gpa_7th = gpa_6th + 2)
  (h3 : gpa_8th = 91) :
  (gpa_6th + gpa_7th + gpa_8th) / 3 = 93 := by
  sorry

end NUMINAMATH_CALUDE_school_average_gpa_l1994_199428


namespace NUMINAMATH_CALUDE_officer_selection_count_l1994_199422

/-- Represents the number of ways to choose officers in a club --/
def choose_officers (total_members boys girls : ℕ) : ℕ :=
  girls * boys * (girls - 1)

/-- Theorem stating the number of ways to choose officers under given conditions --/
theorem officer_selection_count :
  let total_members : ℕ := 24
  let boys : ℕ := 12
  let girls : ℕ := 12
  choose_officers total_members boys girls = 1584 := by
  sorry

#eval choose_officers 24 12 12

end NUMINAMATH_CALUDE_officer_selection_count_l1994_199422


namespace NUMINAMATH_CALUDE_line_equation_proof_l1994_199478

/-- A line in the 2D plane represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a line passes through a point -/
def Line.passesThrough (l : Line) (p : Point) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def Line.isParallelTo (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem line_equation_proof (l : Line) (p : Point) (given_line : Line) :
  p.x = 2 ∧ p.y = -1 ∧
  given_line.a = 2 ∧ given_line.b = 3 ∧ given_line.c = -4 ∧
  l.passesThrough p ∧
  l.isParallelTo given_line →
  l.a = 2 ∧ l.b = 3 ∧ l.c = -1 :=
by sorry

end NUMINAMATH_CALUDE_line_equation_proof_l1994_199478


namespace NUMINAMATH_CALUDE_arithmetic_sequence_k_value_l1994_199403

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_k_value
  (a : ℕ → ℚ)
  (h_arith : ArithmeticSequence a)
  (h1 : a 4 + a 7 + a 10 = 17)
  (h2 : a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 + a 11 + a 12 + a 13 + a 14 = 77)
  (h3 : ∃ k : ℕ, a k = 13) :
  ∃ k : ℕ, a k = 13 ∧ k = 18 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_k_value_l1994_199403


namespace NUMINAMATH_CALUDE_cosine_sine_eighth_power_bounds_l1994_199466

theorem cosine_sine_eighth_power_bounds (x : ℝ) : 
  1/8 ≤ (Real.cos x)^8 + (Real.sin x)^8 ∧ (Real.cos x)^8 + (Real.sin x)^8 ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sine_eighth_power_bounds_l1994_199466


namespace NUMINAMATH_CALUDE_minimum_orange_chips_l1994_199486

theorem minimum_orange_chips 
  (purple green orange : ℕ) 
  (h1 : green ≥ purple / 3)
  (h2 : green ≤ orange / 4)
  (h3 : purple + green ≥ 75) :
  orange ≥ 76 := by
  sorry

end NUMINAMATH_CALUDE_minimum_orange_chips_l1994_199486


namespace NUMINAMATH_CALUDE_circular_saw_blade_distance_l1994_199452

/-- Given a circle with center (2, 3) and radius 8, and points A, B, and C on the circle
    such that ∠ABC is a right angle, AB = 8, and BC = 3, 
    prove that the square of the distance from B to the center of the circle is 41. -/
theorem circular_saw_blade_distance (A B C : ℝ × ℝ) : 
  let O : ℝ × ℝ := (2, 3)
  let r : ℝ := 8
  (A.1 - O.1)^2 + (A.2 - O.2)^2 = r^2 →  -- A is on the circle
  (B.1 - O.1)^2 + (B.2 - O.2)^2 = r^2 →  -- B is on the circle
  (C.1 - O.1)^2 + (C.2 - O.2)^2 = r^2 →  -- C is on the circle
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 8^2 →  -- AB = 8
  (B.1 - C.1)^2 + (B.2 - C.2)^2 = 3^2 →  -- BC = 3
  (A.1 - B.1) * (C.1 - B.1) + (A.2 - B.2) * (C.2 - B.2) = 0 →  -- ∠ABC is a right angle
  (B.1 - O.1)^2 + (B.2 - O.2)^2 = 41 := by
sorry

end NUMINAMATH_CALUDE_circular_saw_blade_distance_l1994_199452


namespace NUMINAMATH_CALUDE_largest_sulfuric_acid_percentage_l1994_199454

/-- Represents the largest integer percentage of sulfuric acid solution that can be achieved in the first vessel after transfer -/
def largest_integer_percentage : ℕ := 76

/-- Represents the initial volume of solution in the first vessel -/
def initial_volume_1 : ℚ := 4

/-- Represents the initial volume of solution in the second vessel -/
def initial_volume_2 : ℚ := 3

/-- Represents the initial concentration of sulfuric acid in the first vessel -/
def initial_concentration_1 : ℚ := 70 / 100

/-- Represents the initial concentration of sulfuric acid in the second vessel -/
def initial_concentration_2 : ℚ := 90 / 100

/-- Represents the capacity of each vessel -/
def vessel_capacity : ℚ := 6

theorem largest_sulfuric_acid_percentage :
  ∀ x : ℚ,
  0 ≤ x ∧ x ≤ initial_volume_2 →
  (initial_volume_1 * initial_concentration_1 + x * initial_concentration_2) / (initial_volume_1 + x) ≤ largest_integer_percentage / 100 ∧
  ∃ y : ℚ, 0 < y ∧ y ≤ initial_volume_2 ∧
  (initial_volume_1 * initial_concentration_1 + y * initial_concentration_2) / (initial_volume_1 + y) > (largest_integer_percentage - 1) / 100 ∧
  initial_volume_1 + y ≤ vessel_capacity :=
by sorry

#check largest_sulfuric_acid_percentage

end NUMINAMATH_CALUDE_largest_sulfuric_acid_percentage_l1994_199454


namespace NUMINAMATH_CALUDE_solution_set_when_m_2_solution_set_condition_l1994_199439

-- Define the function f
def f (x m : ℝ) : ℝ := |2*x - m| + 4*x

-- Part I
theorem solution_set_when_m_2 :
  {x : ℝ | f x 2 ≤ 1} = {x : ℝ | x ≤ -1/2} := by sorry

-- Part II
theorem solution_set_condition (m : ℝ) :
  {x : ℝ | f x m ≤ 2} = {x : ℝ | x ≤ -2} ↔ m = 6 ∨ m = -14 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_m_2_solution_set_condition_l1994_199439


namespace NUMINAMATH_CALUDE_complex_product_symmetric_imaginary_axis_l1994_199402

theorem complex_product_symmetric_imaginary_axis :
  ∀ (z₁ z₂ : ℂ),
  z₁ = 2 + Complex.I →
  Complex.re z₂ = -Complex.re z₁ →
  Complex.im z₂ = Complex.im z₁ →
  z₁ * z₂ = -5 := by
sorry

end NUMINAMATH_CALUDE_complex_product_symmetric_imaginary_axis_l1994_199402


namespace NUMINAMATH_CALUDE_workout_total_weight_l1994_199416

/-- Represents a weightlifting exercise with weight and repetitions -/
structure Exercise where
  weight : ℕ
  reps : ℕ

/-- Calculates the total weight lifted for an exercise -/
def totalWeight (e : Exercise) : ℕ := e.weight * e.reps

/-- Represents a workout session with three exercises -/
structure WorkoutSession where
  chest : Exercise
  back : Exercise
  legs : Exercise

/-- Calculates the grand total weight lifted in a workout session -/
def grandTotalWeight (w : WorkoutSession) : ℕ :=
  totalWeight w.chest + totalWeight w.back + totalWeight w.legs

/-- Theorem: The grand total weight lifted in the given workout session is 2200 pounds -/
theorem workout_total_weight :
  let workout : WorkoutSession := {
    chest := { weight := 90, reps := 8 },
    back := { weight := 70, reps := 10 },
    legs := { weight := 130, reps := 6 }
  }
  grandTotalWeight workout = 2200 := by sorry

end NUMINAMATH_CALUDE_workout_total_weight_l1994_199416


namespace NUMINAMATH_CALUDE_stream_speed_l1994_199417

theorem stream_speed (boat_speed : ℝ) (downstream_distance : ℝ) (upstream_distance : ℝ) :
  boat_speed = 18 →
  downstream_distance = 48 →
  upstream_distance = 32 →
  ∃ (time : ℝ), time > 0 ∧
    time * (boat_speed + 3.6) = downstream_distance ∧
    time * (boat_speed - 3.6) = upstream_distance :=
by sorry

end NUMINAMATH_CALUDE_stream_speed_l1994_199417


namespace NUMINAMATH_CALUDE_smallest_even_triangle_perimeter_l1994_199447

/-- A triangle with consecutive even integer side lengths -/
structure EvenTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  consecutive : b = a + 2 ∧ c = b + 2
  even_a : Even a

/-- The perimeter of an EvenTriangle -/
def perimeter (t : EvenTriangle) : ℕ := t.a + t.b + t.c

/-- The triangle inequality for an EvenTriangle -/
def satisfies_triangle_inequality (t : EvenTriangle) : Prop :=
  t.a + t.b > t.c ∧ t.a + t.c > t.b ∧ t.b + t.c > t.a

theorem smallest_even_triangle_perimeter :
  ∃ (t : EvenTriangle), satisfies_triangle_inequality t ∧
    ∀ (t' : EvenTriangle), satisfies_triangle_inequality t' → perimeter t ≤ perimeter t' ∧
    perimeter t = 12 := by
  sorry

end NUMINAMATH_CALUDE_smallest_even_triangle_perimeter_l1994_199447


namespace NUMINAMATH_CALUDE_chi_square_independence_hypothesis_l1994_199468

/-- Represents a χ² test of independence -/
structure ChiSquareTest where
  /-- The statistical hypothesis of the test -/
  hypothesis : Prop

/-- Represents events in a statistical context -/
structure Event

/-- Defines mutual independence for a list of events -/
def mutually_independent (events : List Event) : Prop :=
  sorry -- Definition of mutual independence

/-- The χ² test of independence assumes mutual independence of events -/
theorem chi_square_independence_hypothesis :
  ∀ (test : ChiSquareTest) (events : List Event),
    test.hypothesis ↔ mutually_independent events := by
  sorry

end NUMINAMATH_CALUDE_chi_square_independence_hypothesis_l1994_199468


namespace NUMINAMATH_CALUDE_problem_solution_l1994_199421

def f (m : ℝ) (x : ℝ) : ℝ := |x + 3| - m

theorem problem_solution (m : ℝ) (h_m : m > 0) 
  (h_solution_set : {x : ℝ | f m (x - 3) ≥ 0} = Set.Iic (-2) ∪ Set.Ici 2) :
  m = 2 ∧ 
  ∀ (x t : ℝ), f 2 x ≥ |2 * x - 1| - t^2 + (3/2) * t + 1 → 
    t ∈ Set.Iic (1/2) ∪ Set.Ici 1 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1994_199421


namespace NUMINAMATH_CALUDE_right_triangle_angle_calculation_l1994_199473

theorem right_triangle_angle_calculation (α β γ : ℝ) :
  α = 90 ∧ β = 63 ∧ α + β + γ = 180 → γ = 27 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_angle_calculation_l1994_199473


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l1994_199425

theorem least_addition_for_divisibility (n : ℕ) : 
  (1024 + n) % 25 = 0 ∧ ∀ m : ℕ, m < n → (1024 + m) % 25 ≠ 0 ↔ n = 1 :=
by sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l1994_199425


namespace NUMINAMATH_CALUDE_root_in_interval_l1994_199433

def f (x : ℝ) : ℝ := x^3 - 3*x + 1

theorem root_in_interval (k : ℕ) : 
  (∃ x : ℝ, x > k ∧ x < k + 1 ∧ f x = 0) → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_root_in_interval_l1994_199433


namespace NUMINAMATH_CALUDE_sqrt_360000_equals_600_l1994_199451

theorem sqrt_360000_equals_600 : Real.sqrt 360000 = 600 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_360000_equals_600_l1994_199451


namespace NUMINAMATH_CALUDE_overlap_area_theorem_l1994_199493

-- Define the square ABCD
def square_side : ℝ := 8

-- Define the rectangle WXYZ
def rect_length : ℝ := 12
def rect_width : ℝ := 8

-- Define the theorem
theorem overlap_area_theorem (shaded_area : ℝ) (AP : ℝ) :
  -- Conditions
  shaded_area = (rect_length * rect_width) / 2 →
  shaded_area = (square_side - AP) * square_side →
  -- Conclusion
  AP = 2 := by
  sorry

end NUMINAMATH_CALUDE_overlap_area_theorem_l1994_199493


namespace NUMINAMATH_CALUDE_odd_multiples_of_three_count_l1994_199405

theorem odd_multiples_of_three_count : 
  (Finset.filter (fun n => n % 2 = 1 ∧ n % 3 = 0) (Finset.range 1001)).card = 167 := by
  sorry

end NUMINAMATH_CALUDE_odd_multiples_of_three_count_l1994_199405


namespace NUMINAMATH_CALUDE_dress_costs_sum_l1994_199450

/-- The cost of dresses for four ladies -/
def dress_costs (pauline_cost ida_cost jean_cost patty_cost : ℕ) : Prop :=
  pauline_cost = 30 ∧
  jean_cost = pauline_cost - 10 ∧
  ida_cost = jean_cost + 30 ∧
  patty_cost = ida_cost + 10

/-- The total cost of all dresses -/
def total_cost (pauline_cost ida_cost jean_cost patty_cost : ℕ) : ℕ :=
  pauline_cost + ida_cost + jean_cost + patty_cost

/-- Theorem: The total cost of all dresses is $160 -/
theorem dress_costs_sum :
  ∀ (pauline_cost ida_cost jean_cost patty_cost : ℕ),
  dress_costs pauline_cost ida_cost jean_cost patty_cost →
  total_cost pauline_cost ida_cost jean_cost patty_cost = 160 := by
  sorry

end NUMINAMATH_CALUDE_dress_costs_sum_l1994_199450


namespace NUMINAMATH_CALUDE_car_speed_problem_l1994_199432

theorem car_speed_problem (total_distance : ℝ) (first_leg_distance : ℝ) (first_leg_speed : ℝ) (average_speed : ℝ) :
  total_distance = 320 →
  first_leg_distance = 160 →
  first_leg_speed = 75 →
  average_speed = 77.4193548387097 →
  let second_leg_distance := total_distance - first_leg_distance
  let total_time := total_distance / average_speed
  let first_leg_time := first_leg_distance / first_leg_speed
  let second_leg_time := total_time - first_leg_time
  let second_leg_speed := second_leg_distance / second_leg_time
  second_leg_speed = 80 := by
sorry

end NUMINAMATH_CALUDE_car_speed_problem_l1994_199432


namespace NUMINAMATH_CALUDE_bowl_weight_after_refill_l1994_199457

theorem bowl_weight_after_refill (empty_bowl_weight : ℕ) 
  (day1_food day2_food day3_food day4_food : ℕ) :
  let total_food := day1_food + day2_food + day3_food + day4_food
  empty_bowl_weight + total_food = 
    empty_bowl_weight + day1_food + day2_food + day3_food + day4_food :=
by sorry

end NUMINAMATH_CALUDE_bowl_weight_after_refill_l1994_199457


namespace NUMINAMATH_CALUDE_last_term_of_ap_l1994_199485

def arithmeticProgression (a : ℕ) (d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

theorem last_term_of_ap : 
  let a := 2  -- first term
  let d := 2  -- common difference
  let n := 31 -- number of terms
  arithmeticProgression a d n = 62 := by
  sorry

end NUMINAMATH_CALUDE_last_term_of_ap_l1994_199485


namespace NUMINAMATH_CALUDE_inscribed_square_probability_l1994_199434

theorem inscribed_square_probability (r : ℝ) (h : r > 0) :
  let circle_area := π * r^2
  let square_side := r * Real.sqrt 2
  let square_area := square_side^2
  square_area / circle_area = 2 / π := by sorry

end NUMINAMATH_CALUDE_inscribed_square_probability_l1994_199434


namespace NUMINAMATH_CALUDE_mirror_area_l1994_199471

/-- Calculates the area of a rectangular mirror inside a frame -/
theorem mirror_area (frame_width frame_height frame_thickness : ℕ) : 
  frame_width = 65 ∧ frame_height = 85 ∧ frame_thickness = 15 →
  (frame_width - 2 * frame_thickness) * (frame_height - 2 * frame_thickness) = 1925 := by
  sorry

end NUMINAMATH_CALUDE_mirror_area_l1994_199471


namespace NUMINAMATH_CALUDE_salary_comparison_l1994_199491

/-- Given salaries in ratio 1:2:3 and sum of B and C's salaries is 6000,
    prove C's salary is 200% more than A's -/
theorem salary_comparison (a b c : ℕ) : 
  a + b + c > 0 →
  b = 2 * a →
  c = 3 * a →
  b + c = 6000 →
  (c - a) * 100 / a = 200 := by
sorry

end NUMINAMATH_CALUDE_salary_comparison_l1994_199491


namespace NUMINAMATH_CALUDE_max_value_sum_fractions_l1994_199436

theorem max_value_sum_fractions (a b c : ℝ) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 2) :
  (∃ (x y z : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ x + y + z = 2 ∧ 
    (x * y) / (x + y) + (x * z) / (x + z) + (y * z) / (y + z) = 1) ∧
  (∀ (x y z : ℝ), x ≥ 0 → y ≥ 0 → z ≥ 0 → x + y + z = 2 → 
    (x * y) / (x + y) + (x * z) / (x + z) + (y * z) / (y + z) ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_max_value_sum_fractions_l1994_199436


namespace NUMINAMATH_CALUDE_triangle_side_length_l1994_199443

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) :
  b = 7 →
  c = 6 →
  Real.cos (B - C) = 31/32 →
  a = (Real.sqrt 299) / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1994_199443


namespace NUMINAMATH_CALUDE_x_squared_minus_y_squared_l1994_199480

theorem x_squared_minus_y_squared (x y : ℝ) 
  (sum_eq : x + y = 20) 
  (diff_eq : x - y = 4) : 
  x^2 - y^2 = 80 := by
sorry

end NUMINAMATH_CALUDE_x_squared_minus_y_squared_l1994_199480


namespace NUMINAMATH_CALUDE_light_ray_distance_l1994_199474

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/9 + y^2/5 = 1

-- Define the foci of the ellipse
def left_focus : ℝ × ℝ := sorry
def right_focus : ℝ × ℝ := sorry

-- Define the total distance traveled by the light ray
def total_distance (p q : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem light_ray_distance :
  ∀ p q : ℝ × ℝ,
  ellipse p.1 p.2 →
  ellipse q.1 q.2 →
  total_distance left_focus p + total_distance p right_focus +
  total_distance right_focus q + total_distance q left_focus = 12 :=
sorry

end NUMINAMATH_CALUDE_light_ray_distance_l1994_199474


namespace NUMINAMATH_CALUDE_dog_food_consumption_l1994_199481

/-- The amount of dog food two dogs eat together daily -/
def total_food : ℝ := 0.25

/-- Given two dogs that eat the same amount of food daily, 
    prove that each dog eats half of the total food -/
theorem dog_food_consumption (dog1_food dog2_food : ℝ) 
  (h1 : dog1_food = dog2_food) 
  (h2 : dog1_food + dog2_food = total_food) : 
  dog1_food = 0.125 := by sorry

end NUMINAMATH_CALUDE_dog_food_consumption_l1994_199481


namespace NUMINAMATH_CALUDE_zachary_pushups_l1994_199441

theorem zachary_pushups (zachary : ℕ) (david : ℕ) : 
  david = zachary + 58 → 
  zachary + david = 146 → 
  zachary = 44 := by
sorry

end NUMINAMATH_CALUDE_zachary_pushups_l1994_199441


namespace NUMINAMATH_CALUDE_boxwood_trim_charge_l1994_199426

/-- Calculates the total charge for trimming boxwoods with various shapes -/
def total_charge (basic_trim_cost sphere_cost pyramid_cost cube_cost : ℚ)
                 (total_boxwoods spheres pyramids cubes : ℕ) : ℚ :=
  basic_trim_cost * total_boxwoods +
  sphere_cost * spheres +
  pyramid_cost * pyramids +
  cube_cost * cubes

/-- Theorem stating the total charge for the given scenario -/
theorem boxwood_trim_charge :
  total_charge 5 15 20 25 30 4 3 2 = 320 := by
  sorry

end NUMINAMATH_CALUDE_boxwood_trim_charge_l1994_199426


namespace NUMINAMATH_CALUDE_man_speed_against_current_l1994_199489

/-- Given a man's speed with the current and the speed of the current, 
    calculate the man's speed against the current. -/
def speed_against_current (speed_with_current speed_of_current : ℝ) : ℝ :=
  speed_with_current - 2 * speed_of_current

/-- Theorem stating that for the given speeds, the man's speed against the current is 9.6 km/hr -/
theorem man_speed_against_current :
  speed_against_current 16 3.2 = 9.6 := by
  sorry

#eval speed_against_current 16 3.2

end NUMINAMATH_CALUDE_man_speed_against_current_l1994_199489


namespace NUMINAMATH_CALUDE_students_in_both_competitions_l1994_199431

/-- The number of students who participated in both competitions A and B -/
def students_in_both (total students_A students_B : ℕ) : ℕ :=
  students_A + students_B - total

theorem students_in_both_competitions 
  (total : ℕ) (students_A : ℕ) (students_B : ℕ)
  (h_total : total = 55)
  (h_A : students_A = 38)
  (h_B : students_B = 42)
  (h_all_participated : total ≤ students_A + students_B) :
  students_in_both total students_A students_B = 25 := by
  sorry

#eval students_in_both 55 38 42  -- Should output 25

end NUMINAMATH_CALUDE_students_in_both_competitions_l1994_199431


namespace NUMINAMATH_CALUDE_horners_method_operations_l1994_199404

/-- The polynomial f(x) = 5x^5 + 4x^4 + 3x^3 + 2x^2 + x + 1 -/
def f (x : ℝ) : ℝ := 5*x^5 + 4*x^4 + 3*x^3 + 2*x^2 + x + 1

/-- Horner's method representation of the polynomial -/
def horner_f (x : ℝ) : ℝ := ((((5*x + 4)*x + 3)*x + 2)*x + 1)*x + 1

/-- The number of multiplication operations in Horner's method for this polynomial -/
def mult_ops : ℕ := 5

/-- The number of addition operations in Horner's method for this polynomial -/
def add_ops : ℕ := 5

theorem horners_method_operations :
  f 5 = horner_f 5 ∧ mult_ops = 5 ∧ add_ops = 5 := by sorry

end NUMINAMATH_CALUDE_horners_method_operations_l1994_199404


namespace NUMINAMATH_CALUDE_de_plus_ef_sum_l1994_199424

/-- Represents a polygon ABCDEF with specific properties -/
structure Polygon where
  area : ℝ
  ab : ℝ
  bc : ℝ
  fa : ℝ
  de_parallel_ab : Prop
  df_horizontal : ℝ

/-- Theorem stating the sum of DE and EF in the given polygon -/
theorem de_plus_ef_sum (p : Polygon) 
  (h1 : p.area = 75)
  (h2 : p.ab = 7)
  (h3 : p.bc = 10)
  (h4 : p.fa = 6)
  (h5 : p.de_parallel_ab)
  (h6 : p.df_horizontal = 8) :
  ∃ (de ef : ℝ), de + ef = 8.25 := by
  sorry

end NUMINAMATH_CALUDE_de_plus_ef_sum_l1994_199424


namespace NUMINAMATH_CALUDE_grocery_cost_l1994_199440

/-- The cost of groceries problem -/
theorem grocery_cost (mango_price rice_price flour_price : ℝ) 
  (h1 : 10 * mango_price = 24 * rice_price)
  (h2 : flour_price = 2 * rice_price)
  (h3 : flour_price = 20.50) : 
  4 * mango_price + 3 * rice_price + 5 * flour_price = 231.65 := by
  sorry

end NUMINAMATH_CALUDE_grocery_cost_l1994_199440


namespace NUMINAMATH_CALUDE_recyclable_containers_l1994_199449

theorem recyclable_containers (total_guests : ℕ) (soda_cans : ℕ) (water_bottles : ℕ) (juice_bottles : ℕ)
  (h_guests : total_guests = 90)
  (h_soda : soda_cans = 50)
  (h_water : water_bottles = 50)
  (h_juice : juice_bottles = 50)
  (h_soda_drinkers : total_guests / 2 = 45)
  (h_water_drinkers : total_guests / 3 = 30)
  (h_juice_consumed : juice_bottles * 4 / 5 = 40) :
  45 + 30 + 40 = 115 := by
  sorry

#check recyclable_containers

end NUMINAMATH_CALUDE_recyclable_containers_l1994_199449


namespace NUMINAMATH_CALUDE_polyhedron_special_value_l1994_199482

/-- Represents a convex polyhedron with specific properties -/
structure ConvexPolyhedron where
  V : ℕ  -- Number of vertices
  E : ℕ  -- Number of edges
  F : ℕ  -- Number of faces
  T : ℕ  -- Number of triangular faces meeting at each vertex
  P : ℕ  -- Number of pentagonal faces meeting at each vertex
  euler_formula : V - E + F = 2
  face_count : F = 32
  vertex_face_relation : V * (P / 5 + T / 3 : ℚ) = 32

/-- Theorem stating the specific value of 100P + 10T + V for the given polyhedron -/
theorem polyhedron_special_value (poly : ConvexPolyhedron) : 
  100 * poly.P + 10 * poly.T + poly.V = 250 := by
  sorry


end NUMINAMATH_CALUDE_polyhedron_special_value_l1994_199482


namespace NUMINAMATH_CALUDE_f_2002_eq_zero_l1994_199418

-- Define the real-valued functions f and g
variable (f g : ℝ → ℝ)

-- Define the properties of f and g
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def is_odd (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

-- State the theorem
theorem f_2002_eq_zero
  (h1 : is_even f)
  (h2 : f 2 = 0)
  (h3 : is_odd g)
  (h4 : ∀ x, g x = f (x - 1)) :
  f 2002 = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_2002_eq_zero_l1994_199418


namespace NUMINAMATH_CALUDE_solution_set_f_gt_5_range_of_a_empty_solution_l1994_199429

-- Define the function f
def f (x : ℝ) : ℝ := |x - 2| + |2*x + 1|

-- Theorem for part I
theorem solution_set_f_gt_5 :
  {x : ℝ | f x > 5} = {x : ℝ | x < -4/3 ∨ x > 2} :=
sorry

-- Theorem for part II
theorem range_of_a_empty_solution :
  {a : ℝ | ∀ x, 1 / (f x - 4) ≠ a} = {a : ℝ | -2/3 < a ∧ a ≤ 0} :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_gt_5_range_of_a_empty_solution_l1994_199429


namespace NUMINAMATH_CALUDE_function_periodicity_l1994_199413

/-- A function satisfying the given functional equation is periodic with period 4 -/
theorem function_periodicity (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, f (x + 1) + f (x - 1) = Real.sqrt 2 * f x) : 
  ∀ x : ℝ, f (x + 4) = f x := by
  sorry

end NUMINAMATH_CALUDE_function_periodicity_l1994_199413


namespace NUMINAMATH_CALUDE_cubic_root_sum_squares_l1994_199458

theorem cubic_root_sum_squares (p q r : ℝ) (x : ℝ → ℝ) 
  (hx : ∀ t, x t = t^3 - p*t^2 + q*t - r) : 
  ∃ (r s t : ℝ), (x r = 0 ∧ x s = 0 ∧ x t = 0) ∧ 
  (r^2 + s^2 + t^2 = p^2 - 2*q) := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_squares_l1994_199458


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l1994_199423

/-- The line equation ax + (2a-1)y + a-3 = 0 passes through the point (5, -3) for all values of a. -/
theorem fixed_point_on_line (a : ℝ) : a * 5 + (2 * a - 1) * (-3) + a - 3 = 0 := by
  sorry

#check fixed_point_on_line

end NUMINAMATH_CALUDE_fixed_point_on_line_l1994_199423


namespace NUMINAMATH_CALUDE_max_n_is_26_l1994_199427

/-- The number of non-congruent trapezoids formed by four points out of n equally spaced points on a circle's circumference -/
def num_trapezoids (n : ℕ) : ℕ := sorry

/-- The maximum value of n such that the number of non-congruent trapezoids is no more than 2012 -/
def max_n : ℕ := sorry

theorem max_n_is_26 :
  (∀ n : ℕ, n > 0 → num_trapezoids n ≤ 2012) ∧
  (∀ m : ℕ, m > max_n → num_trapezoids m > 2012) ∧
  max_n = 26 := by sorry

end NUMINAMATH_CALUDE_max_n_is_26_l1994_199427


namespace NUMINAMATH_CALUDE_jason_initial_quarters_l1994_199464

/-- The number of quarters Jason's dad gave him -/
def quarters_from_dad : ℕ := 25

/-- The total number of quarters Jason has now -/
def total_quarters_now : ℕ := 74

/-- The number of quarters Jason had initially -/
def initial_quarters : ℕ := total_quarters_now - quarters_from_dad

theorem jason_initial_quarters : initial_quarters = 49 := by
  sorry

end NUMINAMATH_CALUDE_jason_initial_quarters_l1994_199464


namespace NUMINAMATH_CALUDE_solve_equation_l1994_199444

theorem solve_equation (x : ℝ) (n : ℝ) (expr : ℝ → ℝ) : 
  x = 1 → 
  n = 4 * x → 
  2 * x * expr x = 10 → 
  n = 4 := by sorry

end NUMINAMATH_CALUDE_solve_equation_l1994_199444


namespace NUMINAMATH_CALUDE_cab_driver_income_l1994_199492

theorem cab_driver_income (day1 day2 day3 day4 day5 : ℕ) 
  (h1 : day1 = 250)
  (h2 : day2 = 400)
  (h4 : day4 = 400)
  (h5 : day5 = 500)
  (h_avg : (day1 + day2 + day3 + day4 + day5) / 5 = 460) :
  day3 = 750 := by
  sorry

end NUMINAMATH_CALUDE_cab_driver_income_l1994_199492


namespace NUMINAMATH_CALUDE_complement_union_theorem_l1994_199498

open Set

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 4}
def N : Set ℕ := {1, 3, 5}

theorem complement_union_theorem : 
  (U \ M) ∪ (U \ N) = {2, 3, 4, 5} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l1994_199498


namespace NUMINAMATH_CALUDE_ceiling_sum_of_roots_l1994_199406

theorem ceiling_sum_of_roots : ⌈Real.sqrt 3⌉ + ⌈Real.sqrt 33⌉ + ⌈Real.sqrt 333⌉ = 27 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_sum_of_roots_l1994_199406


namespace NUMINAMATH_CALUDE_isosceles_triangle_exists_l1994_199469

/-- A regular polygon with 101 vertices -/
structure RegularPolygon101 where
  vertices : Fin 101 → ℝ × ℝ

/-- A selection of 51 vertices from a 101-regular polygon -/
structure Selection51 (polygon : RegularPolygon101) where
  selected : Fin 51 → Fin 101
  distinct : ∀ i j, i ≠ j → selected i ≠ selected j

/-- Three points form an isosceles triangle -/
def IsIsoscelesTriangle (a b c : ℝ × ℝ) : Prop :=
  (a.1 - b.1)^2 + (a.2 - b.2)^2 = (a.1 - c.1)^2 + (a.2 - c.2)^2 ∨
  (b.1 - a.1)^2 + (b.2 - a.2)^2 = (b.1 - c.1)^2 + (b.2 - c.2)^2 ∨
  (c.1 - a.1)^2 + (c.2 - a.2)^2 = (c.1 - b.1)^2 + (c.2 - b.2)^2

/-- Main theorem: Among any 51 vertices of the 101-regular polygon, 
    there are three that form an isosceles triangle -/
theorem isosceles_triangle_exists (polygon : RegularPolygon101) 
  (selection : Selection51 polygon) : 
  ∃ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 
    IsIsoscelesTriangle 
      (polygon.vertices (selection.selected i))
      (polygon.vertices (selection.selected j))
      (polygon.vertices (selection.selected k)) := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_exists_l1994_199469


namespace NUMINAMATH_CALUDE_lowest_sale_price_percentage_l1994_199455

theorem lowest_sale_price_percentage (list_price : ℝ) (regular_discount_max : ℝ) (additional_discount : ℝ) : 
  list_price = 80 →
  regular_discount_max = 0.5 →
  additional_discount = 0.2 →
  (list_price * (1 - regular_discount_max) - list_price * additional_discount) / list_price = 0.3 := by
sorry

end NUMINAMATH_CALUDE_lowest_sale_price_percentage_l1994_199455


namespace NUMINAMATH_CALUDE_sequence_squares_l1994_199412

theorem sequence_squares (n : ℕ) : 
  let a : ℕ → ℕ := λ k => k^2
  (a 1 = 1) ∧ (a 2 = 4) ∧ (a 3 = 9) ∧ (a 4 = 16) ∧ (a 5 = 25) := by
  sorry

end NUMINAMATH_CALUDE_sequence_squares_l1994_199412


namespace NUMINAMATH_CALUDE_neighborhood_total_l1994_199476

/-- Represents the number of households in different categories -/
structure Neighborhood where
  neither : ℕ
  both : ℕ
  with_car : ℕ
  bike_only : ℕ

/-- Calculates the total number of households in the neighborhood -/
def total_households (n : Neighborhood) : ℕ :=
  n.neither + (n.with_car - n.both) + n.bike_only + n.both

/-- Theorem stating that the total number of households is 90 -/
theorem neighborhood_total (n : Neighborhood) 
  (h1 : n.neither = 11)
  (h2 : n.both = 14)
  (h3 : n.with_car = 44)
  (h4 : n.bike_only = 35) : 
  total_households n = 90 := by
  sorry

#eval total_households { neither := 11, both := 14, with_car := 44, bike_only := 35 }

end NUMINAMATH_CALUDE_neighborhood_total_l1994_199476


namespace NUMINAMATH_CALUDE_statement_correctness_l1994_199430

theorem statement_correctness : 
  (∀ a b : ℝ, a > b ∧ b > 0 → a^2 > b^2) ∧
  (∃ a b : ℝ, a^2 > b^2 ∧ ¬(a > b ∧ b > 0)) ∧
  ¬(∀ a b : ℝ, (a > b ∧ b > 0 ↔ 1/a < 1/b)) ∧
  ¬(∀ a b : ℝ, (a > b ∧ b > 0 ↔ a^3 > b^3)) :=
by sorry

end NUMINAMATH_CALUDE_statement_correctness_l1994_199430


namespace NUMINAMATH_CALUDE_least_common_multiple_18_35_l1994_199408

theorem least_common_multiple_18_35 : Nat.lcm 18 35 = 630 := by
  sorry

end NUMINAMATH_CALUDE_least_common_multiple_18_35_l1994_199408


namespace NUMINAMATH_CALUDE_rectangle_forms_same_solid_l1994_199497

-- Define the shapes
inductive Shape
  | RightTriangle
  | Rectangle
  | RightTrapezoid
  | IsoscelesRightTriangle

-- Define a function that determines if a shape forms the same solid when rotated around any edge
def forms_same_solid (s : Shape) : Prop :=
  match s with
  | Shape.Rectangle => true
  | _ => false

-- Theorem statement
theorem rectangle_forms_same_solid :
  ∀ s : Shape, forms_same_solid s ↔ s = Shape.Rectangle :=
by sorry

end NUMINAMATH_CALUDE_rectangle_forms_same_solid_l1994_199497


namespace NUMINAMATH_CALUDE_smallest_integer_with_given_remainders_l1994_199470

theorem smallest_integer_with_given_remainders : ∃ (b : ℕ), 
  b > 0 ∧ 
  b % 3 = 0 ∧ 
  b % 4 = 2 ∧ 
  b % 5 = 3 ∧ 
  ∀ (n : ℕ), n > 0 ∧ n % 3 = 0 ∧ n % 4 = 2 ∧ n % 5 = 3 → b ≤ n :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_integer_with_given_remainders_l1994_199470


namespace NUMINAMATH_CALUDE_first_class_equipment_amount_l1994_199415

/-- Represents the amount of equipment -/
structure Equipment where
  higherClass : ℕ
  firstClass : ℕ

/-- The initial distribution of equipment at two sites -/
structure InitialDistribution where
  site1 : Equipment
  site2 : Equipment

/-- The final distribution of equipment after transfers -/
structure FinalDistribution where
  site1 : Equipment
  site2 : Equipment

/-- Transfers equipment between sites according to the problem description -/
def transfer (init : InitialDistribution) : FinalDistribution :=
  sorry

/-- The conditions of the problem -/
def problemConditions (init : InitialDistribution) (final : FinalDistribution) : Prop :=
  init.site1.firstClass = 0 ∧
  init.site2.higherClass = 0 ∧
  init.site1.higherClass < init.site2.firstClass ∧
  final = transfer init ∧
  final.site1.higherClass = final.site2.higherClass + 26 ∧
  final.site2.higherClass + final.site2.firstClass > 
    (init.site2.higherClass + init.site2.firstClass) * 21 / 20

theorem first_class_equipment_amount 
  (init : InitialDistribution) 
  (final : FinalDistribution) 
  (h : problemConditions init final) : 
  init.site2.firstClass = 60 :=
sorry

end NUMINAMATH_CALUDE_first_class_equipment_amount_l1994_199415


namespace NUMINAMATH_CALUDE_equation_solution_l1994_199488

theorem equation_solution (x : ℝ) (hx : x ≠ 0) :
  (6 * x)^5 = (18 * x)^4 → x = 27 / 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1994_199488


namespace NUMINAMATH_CALUDE_special_polynomial_n_is_two_l1994_199419

/-- A polynomial of degree 2n satisfying specific conditions -/
def SpecialPolynomial (p : ℝ → ℝ) (n : ℕ) : Prop :=
  (∀ k : ℕ, k ≤ n → p (2 * k) = 0) ∧
  (∀ k : ℕ, k < n → p (2 * k + 1) = 2) ∧
  (p (2 * n + 1) = -30)

/-- The theorem stating that n must be 2 for the given conditions -/
theorem special_polynomial_n_is_two :
  ∀ p : ℝ → ℝ, ∀ n : ℕ, SpecialPolynomial p n → n = 2 :=
by sorry

end NUMINAMATH_CALUDE_special_polynomial_n_is_two_l1994_199419


namespace NUMINAMATH_CALUDE_power_sum_negative_two_l1994_199461

theorem power_sum_negative_two : (-2)^2002 + (-2)^2003 = -2^2002 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_negative_two_l1994_199461


namespace NUMINAMATH_CALUDE_tan_2x_value_l1994_199400

/-- Given a function f(x) = sin x + cos x with f'(x) = 3f(x), prove that tan 2x = -4/3 -/
theorem tan_2x_value (f : ℝ → ℝ) (hf : ∀ x, f x = Real.sin x + Real.cos x)
  (hf' : ∀ x, deriv f x = 3 * f x) : 
  Real.tan (2 : ℝ) = -4/3 := by
sorry

end NUMINAMATH_CALUDE_tan_2x_value_l1994_199400


namespace NUMINAMATH_CALUDE_parking_lot_theorem_l1994_199463

/-- Represents a parking lot configuration --/
structure ParkingLot where
  grid : Fin 7 → Fin 7 → Bool
  gate : Fin 7 × Fin 7

/-- Checks if a car can exit from its position --/
def canExit (lot : ParkingLot) (pos : Fin 7 × Fin 7) : Bool :=
  sorry

/-- Counts the number of cars in the parking lot --/
def carCount (lot : ParkingLot) : Nat :=
  sorry

/-- Checks if all cars in the lot can exit --/
def allCarsCanExit (lot : ParkingLot) : Bool :=
  sorry

/-- The maximum number of cars that can be parked --/
def maxCars : Nat := 28

theorem parking_lot_theorem (lot : ParkingLot) :
  (allCarsCanExit lot) → (carCount lot ≤ maxCars) :=
  sorry

end NUMINAMATH_CALUDE_parking_lot_theorem_l1994_199463


namespace NUMINAMATH_CALUDE_second_group_average_age_l1994_199445

theorem second_group_average_age 
  (n₁ : ℕ) (n₂ : ℕ) (m₁ : ℝ) (m_combined : ℝ) :
  n₁ = 11 →
  n₂ = 7 →
  m₁ = 25 →
  m_combined = 32 →
  (n₁ * m₁ + n₂ * ((n₁ + n₂) * m_combined - n₁ * m₁) / n₂) / (n₁ + n₂) = m_combined →
  ((n₁ + n₂) * m_combined - n₁ * m₁) / n₂ = 43 := by
sorry

end NUMINAMATH_CALUDE_second_group_average_age_l1994_199445


namespace NUMINAMATH_CALUDE_sin_x_sin_2x_integral_l1994_199479

theorem sin_x_sin_2x_integral (x : ℝ) :
  deriv (λ x => (1/2) * Real.sin x - (1/6) * Real.sin (3*x)) x = Real.sin x * Real.sin (2*x) := by
  sorry

end NUMINAMATH_CALUDE_sin_x_sin_2x_integral_l1994_199479


namespace NUMINAMATH_CALUDE_john_notebooks_correct_l1994_199477

/-- The number of notebooks John bought -/
def notebooks : ℕ := 5

/-- The number of pages in each notebook -/
def pages_per_notebook : ℕ := 40

/-- The number of pages John uses per day -/
def pages_per_day : ℕ := 4

/-- The number of days the notebooks last -/
def days : ℕ := 50

/-- Theorem stating that the number of notebooks John bought is correct -/
theorem john_notebooks_correct : 
  notebooks * pages_per_notebook = pages_per_day * days := by
  sorry


end NUMINAMATH_CALUDE_john_notebooks_correct_l1994_199477


namespace NUMINAMATH_CALUDE_range_of_k_l1994_199496

theorem range_of_k (n : ℕ+) (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   2*n - 1 < x₁ ∧ x₁ ≤ 2*n + 1 ∧
   2*n - 1 < x₂ ∧ x₂ ≤ 2*n + 1 ∧
   |x₁ - 2*n| = k ∧ |x₂ - 2*n| = k) →
  0 < k ∧ k ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_k_l1994_199496


namespace NUMINAMATH_CALUDE_new_observations_sum_l1994_199494

theorem new_observations_sum (initial_count : ℕ) (initial_avg : ℚ) (new_count : ℕ) (new_avg : ℚ) :
  initial_count = 9 →
  initial_avg = 15 →
  new_count = 3 →
  new_avg = 13 →
  (initial_count * initial_avg + new_count * (3 * new_avg - initial_count * initial_avg)) / new_count = 21 :=
by sorry

end NUMINAMATH_CALUDE_new_observations_sum_l1994_199494


namespace NUMINAMATH_CALUDE_championship_winner_l1994_199472

-- Define the teams
inductive Team : Type
| A | B | C | D

-- Define the positions
inductive Position : Type
| First | Second | Third | Fourth

-- Define a prediction as a pair of (Team, Position)
def Prediction := Team × Position

-- Define the predictions made by each person
def WangPredictions : Prediction × Prediction := ((Team.D, Position.First), (Team.B, Position.Second))
def LiPredictions : Prediction × Prediction := ((Team.A, Position.Second), (Team.C, Position.Fourth))
def ZhangPredictions : Prediction × Prediction := ((Team.C, Position.Third), (Team.D, Position.Second))

-- Define a function to check if a prediction is correct
def isPredictionCorrect (prediction : Prediction) (result : Team → Position) : Prop :=
  result prediction.1 = prediction.2

-- Define the theorem
theorem championship_winner (result : Team → Position) : 
  (isPredictionCorrect WangPredictions.1 result ≠ isPredictionCorrect WangPredictions.2 result) ∧
  (isPredictionCorrect LiPredictions.1 result ≠ isPredictionCorrect LiPredictions.2 result) ∧
  (isPredictionCorrect ZhangPredictions.1 result ≠ isPredictionCorrect ZhangPredictions.2 result) →
  result Team.D = Position.First :=
by
  sorry

end NUMINAMATH_CALUDE_championship_winner_l1994_199472


namespace NUMINAMATH_CALUDE_max_product_digits_sum_23_l1994_199411

/-- The sum of digits of a positive integer -/
def sum_of_digits (n : ℕ+) : ℕ := sorry

/-- The product of digits of a positive integer -/
def product_of_digits (n : ℕ+) : ℕ := sorry

/-- Theorem: The maximum product of digits for a positive integer with digit sum 23 is 432 -/
theorem max_product_digits_sum_23 :
  ∀ n : ℕ+, sum_of_digits n = 23 → product_of_digits n ≤ 432 :=
sorry

end NUMINAMATH_CALUDE_max_product_digits_sum_23_l1994_199411


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_sum_squares_l1994_199410

theorem arithmetic_geometric_mean_sum_squares (x y : ℝ) 
  (h_arithmetic : (x + y) / 2 = 20) 
  (h_geometric : Real.sqrt (x * y) = Real.sqrt 80) : 
  x^2 + y^2 = 1440 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_sum_squares_l1994_199410


namespace NUMINAMATH_CALUDE_parallelograms_count_formula_l1994_199484

/-- The number of parallelograms formed by the intersection of two sets of parallel lines -/
def parallelograms_count (m n : ℕ) : ℕ :=
  Nat.choose m 2 * Nat.choose n 2

/-- Theorem stating that the number of parallelograms formed by the intersection
    of two sets of parallel lines is equal to C_m^2 * C_n^2 -/
theorem parallelograms_count_formula (m n : ℕ) :
  parallelograms_count m n = Nat.choose m 2 * Nat.choose n 2 := by
  sorry

end NUMINAMATH_CALUDE_parallelograms_count_formula_l1994_199484


namespace NUMINAMATH_CALUDE_calculator_to_protractor_equivalence_l1994_199460

/-- Exchange rates at a math conference -/
structure ExchangeRates where
  calculator_to_ruler : ℚ
  ruler_to_compass : ℚ
  compass_to_protractor : ℚ

/-- The exchange rates given in the problem -/
def conference_rates : ExchangeRates where
  calculator_to_ruler := 100
  ruler_to_compass := 3/1
  compass_to_protractor := 2/1

/-- Theorem stating the equivalence between calculators and protractors -/
theorem calculator_to_protractor_equivalence (rates : ExchangeRates) :
  rates.calculator_to_ruler * rates.ruler_to_compass * rates.compass_to_protractor = 600 → 
  rates = conference_rates :=
sorry

#check calculator_to_protractor_equivalence

end NUMINAMATH_CALUDE_calculator_to_protractor_equivalence_l1994_199460


namespace NUMINAMATH_CALUDE_early_bird_dinner_bill_l1994_199456

def early_bird_dinner (curtis_steak rob_steak curtis_side rob_side curtis_drink rob_drink : ℝ)
  (discount_rate tax_rate tip_rate : ℝ) : ℝ :=
  let discounted_curtis_steak := curtis_steak * discount_rate
  let discounted_rob_steak := rob_steak * discount_rate
  let curtis_total := discounted_curtis_steak + curtis_side + curtis_drink
  let rob_total := discounted_rob_steak + rob_side + rob_drink
  let combined_total := curtis_total + rob_total
  let tax := combined_total * tax_rate
  let tip := combined_total * tip_rate
  combined_total + tax + tip

theorem early_bird_dinner_bill : 
  early_bird_dinner 16 18 6 7 3 3.5 0.5 0.07 0.2 = 46.36 := by
  sorry

end NUMINAMATH_CALUDE_early_bird_dinner_bill_l1994_199456


namespace NUMINAMATH_CALUDE_quadratic_equation_general_form_l1994_199487

theorem quadratic_equation_general_form :
  ∀ x : ℝ, (4 * x = x^2 - 8) ↔ (x^2 - 4*x - 8 = 0) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_general_form_l1994_199487


namespace NUMINAMATH_CALUDE_unique_f_two_l1994_199435

def functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x * f y - f (x + y) = x * y

theorem unique_f_two (f : ℝ → ℝ) (h : functional_equation f) : f 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_f_two_l1994_199435


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l1994_199453

theorem quadratic_real_roots (m : ℝ) :
  (∃ x : ℝ, (m - 1) * x^2 + 4 * x - 1 = 0) ↔ (m ≥ -3 ∧ m ≠ 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l1994_199453


namespace NUMINAMATH_CALUDE_bottles_poured_is_four_l1994_199409

def cylinder_capacity : ℚ := 80

def initial_fullness : ℚ := 3/4

def final_fullness : ℚ := 4/5

def bottles_poured (capacity : ℚ) (initial : ℚ) (final : ℚ) : ℚ :=
  capacity * final - capacity * initial

theorem bottles_poured_is_four :
  bottles_poured cylinder_capacity initial_fullness final_fullness = 4 := by
  sorry

end NUMINAMATH_CALUDE_bottles_poured_is_four_l1994_199409


namespace NUMINAMATH_CALUDE_remainder_is_zero_l1994_199401

def divisors : List ℕ := [12, 15, 20, 54]
def least_number : ℕ := 540

theorem remainder_is_zero (n : ℕ) (h : n ∈ divisors) : 
  least_number % n = 0 := by sorry

end NUMINAMATH_CALUDE_remainder_is_zero_l1994_199401


namespace NUMINAMATH_CALUDE_shopping_lottery_largest_number_l1994_199499

/-- Represents the largest number in a systematic sample -/
def largest_sample_number (total : ℕ) (start : ℕ) (interval : ℕ) : ℕ :=
  start + interval * ((total - start) / interval)

/-- The problem statement as a theorem -/
theorem shopping_lottery_largest_number :
  let total := 160
  let start := 7
  let second := 23
  let interval := second - start
  largest_sample_number total start interval = 151 := by
  sorry

#eval largest_sample_number 160 7 16

end NUMINAMATH_CALUDE_shopping_lottery_largest_number_l1994_199499


namespace NUMINAMATH_CALUDE_missing_number_value_l1994_199467

theorem missing_number_value : 
  ∃ (x : ℚ), ((476 + 424) * 2 - x * 476 * 424 = 2704) ∧ (x = -1/223) := by
  sorry

end NUMINAMATH_CALUDE_missing_number_value_l1994_199467


namespace NUMINAMATH_CALUDE_probability_of_blue_ball_l1994_199459

-- Define the total number of balls
def total_balls : ℕ := 10

-- Define the number of blue balls
def blue_balls : ℕ := 6

-- Define the probability of drawing a blue ball
def prob_blue_ball : ℚ := blue_balls / total_balls

-- Theorem statement
theorem probability_of_blue_ball : prob_blue_ball = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_blue_ball_l1994_199459


namespace NUMINAMATH_CALUDE_convex_quadrilaterals_from_circle_points_l1994_199475

theorem convex_quadrilaterals_from_circle_points (n : ℕ) (h : n = 20) :
  Nat.choose n 4 = 4845 :=
sorry

end NUMINAMATH_CALUDE_convex_quadrilaterals_from_circle_points_l1994_199475


namespace NUMINAMATH_CALUDE_integer_equation_solution_l1994_199446

theorem integer_equation_solution (x y : ℤ) : 
  x^2 = y^2 + 2*y + 13 ↔ (x = 4 ∧ y = 1) ∨ (x = -4 ∧ y = 1) ∨ (x = 4 ∧ y = -3) ∨ (x = -4 ∧ y = -3) := by
  sorry

end NUMINAMATH_CALUDE_integer_equation_solution_l1994_199446


namespace NUMINAMATH_CALUDE_soccer_league_games_l1994_199462

theorem soccer_league_games (n : ℕ) (h : n = 14) : (n * (n - 1)) / 2 = 91 := by
  sorry

end NUMINAMATH_CALUDE_soccer_league_games_l1994_199462


namespace NUMINAMATH_CALUDE_hemisphere_surface_area_l1994_199438

theorem hemisphere_surface_area (r : ℝ) (h : r > 0) :
  π * r^2 = 256 * π → 2 * π * r^2 + π * r^2 = 768 * π := by
  sorry

end NUMINAMATH_CALUDE_hemisphere_surface_area_l1994_199438


namespace NUMINAMATH_CALUDE_lcm_504_630_980_l1994_199448

theorem lcm_504_630_980 : Nat.lcm (Nat.lcm 504 630) 980 = 17640 := by
  sorry

end NUMINAMATH_CALUDE_lcm_504_630_980_l1994_199448


namespace NUMINAMATH_CALUDE_total_seeds_planted_l1994_199465

/-- The number of tomato seeds planted by Mike and Ted -/
def total_seeds (mike_morning mike_afternoon ted_morning ted_afternoon : ℕ) : ℕ :=
  mike_morning + mike_afternoon + ted_morning + ted_afternoon

/-- Theorem stating the total number of tomato seeds planted by Mike and Ted -/
theorem total_seeds_planted : 
  ∀ (mike_morning mike_afternoon ted_morning ted_afternoon : ℕ),
  mike_morning = 50 →
  ted_morning = 2 * mike_morning →
  mike_afternoon = 60 →
  ted_afternoon = mike_afternoon - 20 →
  total_seeds mike_morning mike_afternoon ted_morning ted_afternoon = 250 :=
by
  sorry


end NUMINAMATH_CALUDE_total_seeds_planted_l1994_199465


namespace NUMINAMATH_CALUDE_sport_formulation_water_amount_l1994_199414

/-- Represents the ratios and amounts in a flavored drink formulation -/
structure DrinkFormulation where
  standard_ratio_flavoring : ℚ
  standard_ratio_corn_syrup : ℚ
  standard_ratio_water : ℚ
  sport_ratio_flavoring_corn_syrup_multiplier : ℚ
  sport_ratio_flavoring_water_multiplier : ℚ
  sport_corn_syrup_amount : ℚ

/-- Calculates the amount of water in the sport formulation -/
def water_amount (d : DrinkFormulation) : ℚ :=
  let sport_ratio_flavoring := d.standard_ratio_flavoring
  let sport_ratio_corn_syrup := d.standard_ratio_corn_syrup / d.sport_ratio_flavoring_corn_syrup_multiplier
  let sport_ratio_water := d.standard_ratio_water / d.sport_ratio_flavoring_water_multiplier
  let flavoring_amount := d.sport_corn_syrup_amount * (sport_ratio_flavoring / sport_ratio_corn_syrup)
  flavoring_amount * (sport_ratio_water / sport_ratio_flavoring)

theorem sport_formulation_water_amount 
  (d : DrinkFormulation)
  (h1 : d.standard_ratio_flavoring = 1)
  (h2 : d.standard_ratio_corn_syrup = 12)
  (h3 : d.standard_ratio_water = 30)
  (h4 : d.sport_ratio_flavoring_corn_syrup_multiplier = 3)
  (h5 : d.sport_ratio_flavoring_water_multiplier = 2)
  (h6 : d.sport_corn_syrup_amount = 5) :
  water_amount d = 75/4 := by
  sorry

end NUMINAMATH_CALUDE_sport_formulation_water_amount_l1994_199414


namespace NUMINAMATH_CALUDE_sandhill_football_club_members_l1994_199483

/-- Represents the Sandhill Football Club problem --/
theorem sandhill_football_club_members :
  let sock_cost : ℕ := 5
  let tshirt_cost : ℕ := sock_cost + 6
  let home_game_socks : ℕ := 1
  let home_game_tshirts : ℕ := 1
  let away_game_socks : ℕ := 2
  let away_game_tshirts : ℕ := 1
  let total_expenditure : ℕ := 4150
  let member_cost : ℕ := 
    (home_game_socks + away_game_socks) * sock_cost + 
    (home_game_tshirts + away_game_tshirts) * tshirt_cost
  let number_of_members : ℕ := total_expenditure / member_cost
  number_of_members = 112 :=
by
  sorry


end NUMINAMATH_CALUDE_sandhill_football_club_members_l1994_199483


namespace NUMINAMATH_CALUDE_orthogonal_vectors_x_value_l1994_199420

theorem orthogonal_vectors_x_value (x : ℝ) : 
  let a : Fin 2 → ℝ := ![2, 1]
  let b : Fin 2 → ℝ := ![x, -1]
  (∀ i, i < 2 → a i * b i = 0) → x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_orthogonal_vectors_x_value_l1994_199420


namespace NUMINAMATH_CALUDE_jeff_travel_distance_l1994_199490

/-- Calculates the total distance traveled given a list of speed-time pairs -/
def totalDistance (segments : List (Real × Real)) : Real :=
  segments.foldl (fun acc (speed, time) => acc + speed * time) 0

/-- Proves that Jeff's total travel distance is 820 miles -/
theorem jeff_travel_distance :
  let segments : List (Real × Real) := [
    (80, 3), (50, 2), (70, 1), (60, 2),
    (45, 3), (40, 2), (30, 2.5)
  ]
  totalDistance segments = 820 := by
  sorry

#eval totalDistance [(80, 3), (50, 2), (70, 1), (60, 2), (45, 3), (40, 2), (30, 2.5)]

end NUMINAMATH_CALUDE_jeff_travel_distance_l1994_199490
