import Mathlib

namespace NUMINAMATH_CALUDE_bake_sale_money_raised_l2549_254991

/-- Represents the number of items in a dozen -/
def dozenSize : Nat := 12

/-- Calculates the total number of items given the number of dozens -/
def totalItems (dozens : Nat) : Nat := dozens * dozenSize

/-- Represents the price of a cookie in cents -/
def cookiePrice : Nat := 100

/-- Represents the price of a brownie or blondie in cents -/
def browniePrice : Nat := 200

/-- Calculates the total money raised from the bake sale -/
def totalMoneyRaised : Nat :=
  let bettyChocolateChip := totalItems 4
  let bettyOatmealRaisin := totalItems 6
  let bettyBrownies := totalItems 2
  let paigeSugar := totalItems 6
  let paigeBlondies := totalItems 3
  let paigeCreamCheese := totalItems 5

  let totalCookies := bettyChocolateChip + bettyOatmealRaisin + paigeSugar
  let totalBrowniesBlondies := bettyBrownies + paigeBlondies + paigeCreamCheese

  totalCookies * cookiePrice + totalBrowniesBlondies * browniePrice

theorem bake_sale_money_raised :
  totalMoneyRaised = 43200 := by sorry

end NUMINAMATH_CALUDE_bake_sale_money_raised_l2549_254991


namespace NUMINAMATH_CALUDE_pencil_buyers_difference_l2549_254961

-- Define the cost of a pencil in cents
def pencil_cost : ℕ := 12

-- Define the total amount paid by seventh graders in cents
def seventh_grade_total : ℕ := 192

-- Define the total amount paid by sixth graders in cents
def sixth_grade_total : ℕ := 252

-- Define the number of sixth graders
def total_sixth_graders : ℕ := 35

-- Theorem statement
theorem pencil_buyers_difference : 
  (sixth_grade_total / pencil_cost) - (seventh_grade_total / pencil_cost) = 5 := by
  sorry

end NUMINAMATH_CALUDE_pencil_buyers_difference_l2549_254961


namespace NUMINAMATH_CALUDE_arithmetic_sequence_min_sum_l2549_254963

/-- An arithmetic sequence with a_4 = -14 and common difference d = 3 -/
def ArithmeticSequence (n : ℕ) : ℤ := 3*n - 26

/-- The sum of the first n terms of the arithmetic sequence -/
def SequenceSum (n : ℕ) : ℤ := n * (ArithmeticSequence 1 + ArithmeticSequence n) / 2

theorem arithmetic_sequence_min_sum :
  (∀ m : ℕ, SequenceSum m ≥ SequenceSum 8) ∧
  SequenceSum 8 = -100 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_min_sum_l2549_254963


namespace NUMINAMATH_CALUDE_parabola_axis_of_symmetry_l2549_254911

/-- The axis of symmetry of a parabola y=(x-h)^2 is x=h -/
theorem parabola_axis_of_symmetry (h : ℝ) :
  let f : ℝ → ℝ := fun x ↦ (x - h)^2
  ∃! a : ℝ, ∀ x y : ℝ, f x = f y → (x + y) / 2 = a :=
by sorry

end NUMINAMATH_CALUDE_parabola_axis_of_symmetry_l2549_254911


namespace NUMINAMATH_CALUDE_brody_calculator_theorem_l2549_254977

def calculator_problem (total_battery : ℝ) (used_fraction : ℝ) (exam_duration : ℝ) : Prop :=
  let remaining_before_exam := total_battery * (1 - used_fraction)
  let remaining_after_exam := remaining_before_exam - exam_duration
  remaining_after_exam = 13

theorem brody_calculator_theorem :
  calculator_problem 60 (3/4) 2 := by
  sorry

end NUMINAMATH_CALUDE_brody_calculator_theorem_l2549_254977


namespace NUMINAMATH_CALUDE_evaluate_expression_l2549_254960

theorem evaluate_expression (x : ℝ) (h : x = -3) :
  (3 + x * (3 + x) - 3^2 + x) / (x - 3 + x^2 - x) = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2549_254960


namespace NUMINAMATH_CALUDE_willey_farm_capital_l2549_254945

def total_land : ℕ := 4500
def corn_cost : ℕ := 42
def wheat_cost : ℕ := 35
def wheat_acres : ℕ := 3400

theorem willey_farm_capital :
  let corn_acres := total_land - wheat_acres
  let wheat_total_cost := wheat_cost * wheat_acres
  let corn_total_cost := corn_cost * corn_acres
  wheat_total_cost + corn_total_cost = 165200 := by sorry

end NUMINAMATH_CALUDE_willey_farm_capital_l2549_254945


namespace NUMINAMATH_CALUDE_aa_existence_l2549_254922

theorem aa_existence : ∃ aa : ℕ, 1 ≤ aa ∧ aa ≤ 9 ∧ (7 * aa^3) % 100 ≥ 10 ∧ (7 * aa^3) % 100 < 20 :=
by sorry

end NUMINAMATH_CALUDE_aa_existence_l2549_254922


namespace NUMINAMATH_CALUDE_closest_point_l2549_254999

def v (t : ℝ) : Fin 3 → ℝ := fun i => 
  match i with
  | 0 => 3 + 8*t
  | 1 => -1 + 2*t
  | 2 => -2 - 3*t

def a : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 1
  | 1 => 7
  | 2 => 1

def direction : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 8
  | 1 => 2
  | 2 => -3

theorem closest_point (t : ℝ) : 
  (∀ s : ℝ, ‖v t - a‖ ≤ ‖v s - a‖) ↔ t = -1/7 := by sorry

end NUMINAMATH_CALUDE_closest_point_l2549_254999


namespace NUMINAMATH_CALUDE_train_speed_l2549_254919

/-- The speed of a train given its length and time to cross a fixed point -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 300) (h2 : time = 10) :
  length / time = 30 := by
  sorry

#check train_speed

end NUMINAMATH_CALUDE_train_speed_l2549_254919


namespace NUMINAMATH_CALUDE_hexagon_angles_arithmetic_progression_l2549_254939

theorem hexagon_angles_arithmetic_progression :
  ∃ (a d : ℝ), 
    (∀ i : Fin 6, 0 ≤ a + i * d ∧ a + i * d ≤ 180) ∧ 
    (a + (a + d) + (a + 2*d) + (a + 3*d) + (a + 4*d) + (a + 5*d) = 720) ∧
    (∃ i : Fin 6, a + i * d = 120) := by
  sorry

end NUMINAMATH_CALUDE_hexagon_angles_arithmetic_progression_l2549_254939


namespace NUMINAMATH_CALUDE_divisible_by_10101010101_has_at_least_6_nonzero_digits_l2549_254944

/-- The number of non-zero digits in the decimal representation of a natural number -/
def num_nonzero_digits (n : ℕ) : ℕ := sorry

/-- Theorem: Any natural number divisible by 10101010101 has at least 6 non-zero digits -/
theorem divisible_by_10101010101_has_at_least_6_nonzero_digits (k : ℕ) :
  k % 10101010101 = 0 → num_nonzero_digits k ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_10101010101_has_at_least_6_nonzero_digits_l2549_254944


namespace NUMINAMATH_CALUDE_collinear_points_x_value_l2549_254938

/-- Given three collinear points A(-1, 2), B(2, 4), and C(x, 3), prove that x = 1/2 --/
theorem collinear_points_x_value :
  let A : ℝ × ℝ := (-1, 2)
  let B : ℝ × ℝ := (2, 4)
  let C : ℝ × ℝ := (x, 3)
  (∀ t : ℝ, ∃ u v : ℝ, u * (B.1 - A.1) + v * (C.1 - A.1) = t * (B.1 - A.1) ∧
                       u * (B.2 - A.2) + v * (C.2 - A.2) = t * (B.2 - A.2)) →
  x = 1/2 := by
sorry

end NUMINAMATH_CALUDE_collinear_points_x_value_l2549_254938


namespace NUMINAMATH_CALUDE_all_signs_flippable_l2549_254951

/-- Represents a grid of +1 and -1 values -/
def Grid (m n : ℕ) := Fin m → Fin n → Int

/-- Represents the allowed sign-changing patterns -/
inductive Pattern
| horizontal : Pattern
| vertical : Pattern

/-- Applies a pattern to a specific location in the grid -/
def applyPattern (g : Grid m n) (p : Pattern) (i j : ℕ) : Grid m n :=
  sorry

/-- Checks if all signs in the grid have been flipped -/
def allSignsFlipped (g₁ g₂ : Grid m n) : Prop :=
  sorry

/-- Main theorem: All signs can be flipped iff m and n are multiples of 4 -/
theorem all_signs_flippable (m n : ℕ) :
  (∃ (g : Grid m n), ∃ (operations : List (Pattern × ℕ × ℕ)),
    allSignsFlipped g (operations.foldl (λ acc (p, i, j) => applyPattern acc p i j) g))
  ↔ (∃ (k₁ k₂ : ℕ), m = 4 * k₁ ∧ n = 4 * k₂) :=
sorry

end NUMINAMATH_CALUDE_all_signs_flippable_l2549_254951


namespace NUMINAMATH_CALUDE_school_classrooms_l2549_254948

theorem school_classrooms 
  (total_students : ℕ) 
  (seats_per_bus : ℕ) 
  (buses_needed : ℕ) 
  (h1 : total_students = 58)
  (h2 : seats_per_bus = 2)
  (h3 : buses_needed = 29)
  (h4 : total_students = buses_needed * seats_per_bus)
  (h5 : ∃ (students_per_class : ℕ), total_students % students_per_class = 0) :
  ∃ (num_classrooms : ℕ), num_classrooms = 2 ∧ 
    total_students / num_classrooms = buses_needed := by
  sorry

end NUMINAMATH_CALUDE_school_classrooms_l2549_254948


namespace NUMINAMATH_CALUDE_orchid_rose_difference_l2549_254904

/-- The number of roses initially in the vase -/
def initial_roses : ℕ := 9

/-- The number of orchids initially in the vase -/
def initial_orchids : ℕ := 6

/-- The current number of roses in the vase -/
def current_roses : ℕ := 3

/-- The current number of orchids in the vase -/
def current_orchids : ℕ := 13

/-- Theorem stating the difference between the current number of orchids and roses -/
theorem orchid_rose_difference :
  current_orchids - current_roses = 10 := by sorry

end NUMINAMATH_CALUDE_orchid_rose_difference_l2549_254904


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2549_254949

theorem sum_of_roots_quadratic (a b : ℝ) : 
  (∀ x : ℝ, x^2 - (a+b)*x + a*b + 1 = 0 ↔ x = a ∨ x = b) → 
  a + b = a + b :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2549_254949


namespace NUMINAMATH_CALUDE_trig_expressions_l2549_254971

theorem trig_expressions (α : Real) (h : Real.tan α = 2) :
  (4 * Real.sin α - 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = 6/11 ∧
  (1/4) * Real.sin α ^ 2 + (1/3) * Real.sin α * Real.cos α + (1/2) * Real.cos α ^ 2 = 13/30 := by
  sorry

end NUMINAMATH_CALUDE_trig_expressions_l2549_254971


namespace NUMINAMATH_CALUDE_compute_alpha_l2549_254926

theorem compute_alpha (α β : ℂ) 
  (h1 : (α + β).re > 0)
  (h2 : (Complex.I * (2 * α - 3 * β)).re > 0)
  (h3 : β = 4 + 3 * Complex.I) :
  α = 6 + (3 / 2) * Complex.I := by sorry

end NUMINAMATH_CALUDE_compute_alpha_l2549_254926


namespace NUMINAMATH_CALUDE_cylindrical_containers_radius_l2549_254974

theorem cylindrical_containers_radius (h : ℝ) (r : ℝ) :
  h > 0 →
  (π * (8^2) * (4 * h) = π * r^2 * h) →
  r = 16 := by
sorry

end NUMINAMATH_CALUDE_cylindrical_containers_radius_l2549_254974


namespace NUMINAMATH_CALUDE_tom_apple_slices_l2549_254992

theorem tom_apple_slices (total_apples : ℕ) (slices_per_apple : ℕ) (slices_left : ℕ) :
  total_apples = 2 →
  slices_per_apple = 8 →
  slices_left = 5 →
  (∃ (slices_given : ℕ),
    slices_given + 2 * slices_left = total_apples * slices_per_apple ∧
    slices_given = (3 : ℚ) / 8 * (total_apples * slices_per_apple : ℚ)) :=
by sorry

end NUMINAMATH_CALUDE_tom_apple_slices_l2549_254992


namespace NUMINAMATH_CALUDE_unit_vector_xy_plane_l2549_254965

theorem unit_vector_xy_plane (u : ℝ × ℝ × ℝ) : 
  let (x, y, z) := u
  (x^2 + y^2 = 1 ∧ z = 0) →  -- u is a unit vector in the xy-plane
  (x + 3*y = Real.sqrt 30 / 2) →  -- angle with (1, 3, 0) is 30°
  (3*x - y = Real.sqrt 20 / 2) →  -- angle with (3, -1, 0) is 45°
  x = (3 * Real.sqrt 20 + Real.sqrt 30) / 20 :=
by sorry

end NUMINAMATH_CALUDE_unit_vector_xy_plane_l2549_254965


namespace NUMINAMATH_CALUDE_erics_chickens_l2549_254969

/-- The number of chickens on Eric's farm -/
def num_chickens : ℕ := 4

/-- The number of eggs each chicken lays per day -/
def eggs_per_chicken_per_day : ℕ := 3

/-- The number of days Eric collected eggs -/
def days_collected : ℕ := 3

/-- The total number of eggs collected -/
def total_eggs_collected : ℕ := 36

theorem erics_chickens :
  num_chickens * eggs_per_chicken_per_day * days_collected = total_eggs_collected :=
by sorry

end NUMINAMATH_CALUDE_erics_chickens_l2549_254969


namespace NUMINAMATH_CALUDE_sum_of_last_two_digits_9_20_plus_11_20_l2549_254935

theorem sum_of_last_two_digits_9_20_plus_11_20 :
  (9^20 + 11^20) % 100 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_last_two_digits_9_20_plus_11_20_l2549_254935


namespace NUMINAMATH_CALUDE_largest_integer_with_remainder_l2549_254918

theorem largest_integer_with_remainder : ∃ n : ℕ, n < 100 ∧ n % 7 = 4 ∧ ∀ m : ℕ, m < 100 ∧ m % 7 = 4 → m ≤ n :=
  by sorry

end NUMINAMATH_CALUDE_largest_integer_with_remainder_l2549_254918


namespace NUMINAMATH_CALUDE_circle_symmetry_l2549_254901

-- Define the original circle
def original_circle (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 2*y + 1 = 0

-- Define the line of symmetry
def symmetry_line (x y : ℝ) : Prop :=
  x + y - 1 = 0

-- Define the symmetric circle
def symmetric_circle (x y : ℝ) : Prop :=
  x^2 + y^2 = 1

-- Theorem statement
theorem circle_symmetry :
  ∀ (x y : ℝ), 
    (∃ (x₀ y₀ : ℝ), original_circle x₀ y₀ ∧ 
      symmetry_line ((x + x₀)/2) ((y + y₀)/2)) →
    symmetric_circle x y :=
sorry

end NUMINAMATH_CALUDE_circle_symmetry_l2549_254901


namespace NUMINAMATH_CALUDE_integer_triangle_exists_l2549_254968

/-- A triangle with integer side lengths forming an arithmetic progression and integer area -/
structure IntegerTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  area : ℕ
  arith_prog : b - a = c - b
  area_formula : area^2 = (a + b + c) * (b + c - a) * (a + c - b) * (a + b - c) / 16

/-- The existence of a specific integer triangle with sides 3, 4, 5 -/
theorem integer_triangle_exists : ∃ (t : IntegerTriangle), t.a = 3 ∧ t.b = 4 ∧ t.c = 5 := by
  sorry

end NUMINAMATH_CALUDE_integer_triangle_exists_l2549_254968


namespace NUMINAMATH_CALUDE_ceiling_of_negative_two_point_four_l2549_254909

theorem ceiling_of_negative_two_point_four :
  ⌈(-2.4 : ℝ)⌉ = -2 := by sorry

end NUMINAMATH_CALUDE_ceiling_of_negative_two_point_four_l2549_254909


namespace NUMINAMATH_CALUDE_student_failed_marks_l2549_254937

def total_marks : ℕ := 500
def passing_percentage : ℚ := 33 / 100
def student_marks : ℕ := 125

theorem student_failed_marks :
  (total_marks * passing_percentage).floor - student_marks = 40 :=
by sorry

end NUMINAMATH_CALUDE_student_failed_marks_l2549_254937


namespace NUMINAMATH_CALUDE_initial_peaches_l2549_254910

/-- Given a basket of peaches, prove that the initial number of peaches is 20 
    when 25 more are added to make a total of 45. -/
theorem initial_peaches (initial : ℕ) : initial + 25 = 45 → initial = 20 := by
  sorry

end NUMINAMATH_CALUDE_initial_peaches_l2549_254910


namespace NUMINAMATH_CALUDE_smallest_number_proof_l2549_254988

theorem smallest_number_proof (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  (a + b + c) / 3 = 24 →
  b = 23 →
  max a (max b c) = b + 4 →
  min a (min b c) = 22 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_proof_l2549_254988


namespace NUMINAMATH_CALUDE_problem_solution_l2549_254907

theorem problem_solution (x y : ℝ) (h1 : y > x) (h2 : x > 0) (h3 : x / y + y / x = 8) :
  (x + y) / (x - y) = -Real.sqrt (5 / 3) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2549_254907


namespace NUMINAMATH_CALUDE_inscribed_cube_volume_l2549_254902

/-- The volume of a cube inscribed in a sphere, which is itself inscribed in a larger cube -/
theorem inscribed_cube_volume (large_cube_edge : ℝ) (sphere_diameter : ℝ) (small_cube_diagonal : ℝ) :
  large_cube_edge = 12 →
  sphere_diameter = large_cube_edge →
  small_cube_diagonal = sphere_diameter / 2 →
  (small_cube_diagonal / Real.sqrt 3) ^ 3 = 24 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_cube_volume_l2549_254902


namespace NUMINAMATH_CALUDE_f_monotonicity_when_k_zero_f_nonnegative_condition_exponential_fraction_inequality_l2549_254980

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := Real.exp (2 * x) - 1 - 2 * x - k * x^2

theorem f_monotonicity_when_k_zero :
  ∀ x : ℝ, x > 0 → (deriv (f 0)) x > 0 ∧
  ∀ x : ℝ, x < 0 → (deriv (f 0)) x < 0 := by sorry

theorem f_nonnegative_condition :
  ∀ k : ℝ, (∀ x : ℝ, x ≥ 0 → f k x ≥ 0) ↔ k ≤ 2 := by sorry

theorem exponential_fraction_inequality :
  ∀ n : ℕ+, (Real.exp (2 * ↑n) - 1) / (Real.exp 2 - 1) ≥ (2 * ↑n^3 + ↑n) / 3 := by sorry

end NUMINAMATH_CALUDE_f_monotonicity_when_k_zero_f_nonnegative_condition_exponential_fraction_inequality_l2549_254980


namespace NUMINAMATH_CALUDE_sacks_per_day_l2549_254932

/-- Given a harvest of oranges that lasts for a certain number of days and produces a total number of sacks, this theorem proves the number of sacks harvested per day. -/
theorem sacks_per_day (total_sacks : ℕ) (harvest_days : ℕ) (h1 : total_sacks = 56) (h2 : harvest_days = 4) :
  total_sacks / harvest_days = 14 := by
  sorry

end NUMINAMATH_CALUDE_sacks_per_day_l2549_254932


namespace NUMINAMATH_CALUDE_overlap_area_and_perimeter_l2549_254943

/-- Given two strips of widths 1 and 2 overlapping at an angle of π/4 radians,
    the area of the overlap region is √2 and the perimeter is 4√3. -/
theorem overlap_area_and_perimeter :
  ∀ (strip1_width strip2_width overlap_angle : ℝ),
    strip1_width = 1 →
    strip2_width = 2 →
    overlap_angle = π / 4 →
    ∃ (area perimeter : ℝ),
      area = Real.sqrt 2 ∧
      perimeter = 4 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_overlap_area_and_perimeter_l2549_254943


namespace NUMINAMATH_CALUDE_victors_friend_bought_two_decks_l2549_254985

/-- The number of decks Victor's friend bought given the conditions of the problem -/
def victors_friend_decks (deck_cost : ℕ) (victors_decks : ℕ) (total_spent : ℕ) : ℕ :=
  (total_spent - deck_cost * victors_decks) / deck_cost

/-- Theorem stating that Victor's friend bought 2 decks -/
theorem victors_friend_bought_two_decks :
  victors_friend_decks 8 6 64 = 2 := by
  sorry

end NUMINAMATH_CALUDE_victors_friend_bought_two_decks_l2549_254985


namespace NUMINAMATH_CALUDE_f_increasing_when_a_zero_f_odd_when_a_one_f_domain_iff_a_lt_two_l2549_254986

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (4^x - 1) / (4^x + 1 - a * 2^x)

-- Theorem 1: When a = 0, f is increasing
theorem f_increasing_when_a_zero : 
  ∀ x y : ℝ, x < y → f 0 x < f 0 y :=
sorry

-- Theorem 2: When a = 1, f is odd
theorem f_odd_when_a_one :
  ∀ x : ℝ, f 1 (-x) = -(f 1 x) :=
sorry

-- Theorem 3: Domain of f is R iff a < 2
theorem f_domain_iff_a_lt_two :
  ∀ a : ℝ, (∀ x : ℝ, ∃ y : ℝ, f a x = y) ↔ a < 2 :=
sorry

end NUMINAMATH_CALUDE_f_increasing_when_a_zero_f_odd_when_a_one_f_domain_iff_a_lt_two_l2549_254986


namespace NUMINAMATH_CALUDE_sin_2x_minus_y_eq_neg_one_l2549_254950

theorem sin_2x_minus_y_eq_neg_one 
  (hx : x + Real.sin x * Real.cos x - 1 = 0)
  (hy : 2 * Real.cos y - 2 * y + Real.pi + 4 = 0) : 
  Real.sin (2 * x - y) = -1 := by
sorry

end NUMINAMATH_CALUDE_sin_2x_minus_y_eq_neg_one_l2549_254950


namespace NUMINAMATH_CALUDE_class_a_win_probability_class_b_score_expectation_l2549_254997

/-- Represents the result of a single event --/
inductive EventResult
| Win
| Lose

/-- Represents the outcome of the three events for a class --/
structure ClassOutcome :=
  (event1 : EventResult)
  (event2 : EventResult)
  (event3 : EventResult)

/-- Calculates the score for a given ClassOutcome --/
def score (outcome : ClassOutcome) : Int :=
  let e1 := match outcome.event1 with
    | EventResult.Win => 2
    | EventResult.Lose => -1
  let e2 := match outcome.event2 with
    | EventResult.Win => 2
    | EventResult.Lose => -1
  let e3 := match outcome.event3 with
    | EventResult.Win => 2
    | EventResult.Lose => -1
  e1 + e2 + e3

/-- Probabilities of Class A winning each event --/
def probA1 : Float := 0.4
def probA2 : Float := 0.5
def probA3 : Float := 0.8

/-- Theorem stating the probability of Class A winning the championship --/
theorem class_a_win_probability :
  let p := probA1 * probA2 * probA3 +
           (1 - probA1) * probA2 * probA3 +
           probA1 * (1 - probA2) * probA3 +
           probA1 * probA2 * (1 - probA3)
  p = 0.6 := by sorry

/-- Theorem stating the expectation of Class B's total score --/
theorem class_b_score_expectation :
  let p_neg3 := probA1 * probA2 * probA3
  let p_0 := (1 - probA1) * probA2 * probA3 + probA1 * (1 - probA2) * probA3 + probA1 * probA2 * (1 - probA3)
  let p_3 := (1 - probA1) * (1 - probA2) * probA3 + (1 - probA1) * probA2 * (1 - probA3) + probA1 * (1 - probA2) * (1 - probA3)
  let p_6 := (1 - probA1) * (1 - probA2) * (1 - probA3)
  let expectation := -3 * p_neg3 + 0 * p_0 + 3 * p_3 + 6 * p_6
  expectation = 0.9 := by sorry

end NUMINAMATH_CALUDE_class_a_win_probability_class_b_score_expectation_l2549_254997


namespace NUMINAMATH_CALUDE_problem_statement_l2549_254941

theorem problem_statement : 2 * ((7 + 5)^2 + (7^2 + 5^2)) = 436 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2549_254941


namespace NUMINAMATH_CALUDE_four_percent_of_y_is_sixteen_l2549_254916

theorem four_percent_of_y_is_sixteen (y : ℝ) (h1 : y > 0) (h2 : 0.04 * y = 16) : y = 400 := by
  sorry

end NUMINAMATH_CALUDE_four_percent_of_y_is_sixteen_l2549_254916


namespace NUMINAMATH_CALUDE_solution_set_l2549_254906

theorem solution_set (x y z : ℝ) : 
  x^2 = y^2 + z^2 ∧ 
  x^2024 = y^2024 + z^2024 ∧ 
  x^2025 = y^2025 + z^2025 →
  ((y = x ∧ z = 0) ∨ (y = -x ∧ z = 0) ∨ (y = 0 ∧ z = x) ∨ (y = 0 ∧ z = -x)) := by
sorry

end NUMINAMATH_CALUDE_solution_set_l2549_254906


namespace NUMINAMATH_CALUDE_train_length_train_length_approx_200m_l2549_254946

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmph : ℝ) (crossing_time_sec : ℝ) : ℝ :=
  let speed_mps := speed_kmph * 1000 / 3600
  speed_mps * crossing_time_sec

/-- Proof that a train traveling at 120 kmph crossing a pole in 6 seconds is approximately 200 meters long -/
theorem train_length_approx_200m :
  ∃ ε > 0, |train_length 120 6 - 200| < ε :=
sorry

end NUMINAMATH_CALUDE_train_length_train_length_approx_200m_l2549_254946


namespace NUMINAMATH_CALUDE_equation_solution_l2549_254912

theorem equation_solution (x : ℝ) : 
  (x / 6) / 3 = 6 / (x / 3) → x = 18 ∨ x = -18 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2549_254912


namespace NUMINAMATH_CALUDE_mateo_net_salary_proof_l2549_254993

/-- Calculate Mateo's net salary for a week with absences -/
def calculate_net_salary (regular_salary : ℝ) (absence_days : ℕ) : ℝ :=
  let absence_deduction := 
    if absence_days ≥ 1 then 0.01 * regular_salary else 0 +
    if absence_days ≥ 2 then 0.02 * regular_salary else 0 +
    if absence_days ≥ 3 then 0.03 * regular_salary else 0 +
    if absence_days ≥ 4 then 0.04 * regular_salary else 0
  let salary_after_absence := regular_salary - absence_deduction
  let income_tax := 0.07 * salary_after_absence
  salary_after_absence - income_tax

theorem mateo_net_salary_proof :
  let regular_salary : ℝ := 791
  let absence_days : ℕ := 4
  let net_salary := calculate_net_salary regular_salary absence_days
  ∃ ε > 0, |net_salary - 662.07| < ε ∧ ε < 0.01 :=
by sorry

end NUMINAMATH_CALUDE_mateo_net_salary_proof_l2549_254993


namespace NUMINAMATH_CALUDE_pizza_slice_price_l2549_254964

theorem pizza_slice_price 
  (whole_pizza_price : ℝ)
  (slices_sold : ℕ)
  (whole_pizzas_sold : ℕ)
  (total_revenue : ℝ)
  (h1 : whole_pizza_price = 15)
  (h2 : slices_sold = 24)
  (h3 : whole_pizzas_sold = 3)
  (h4 : total_revenue = 117) :
  ∃ (price_per_slice : ℝ), 
    price_per_slice * slices_sold + whole_pizza_price * whole_pizzas_sold = total_revenue ∧ 
    price_per_slice = 3 := by
  sorry

end NUMINAMATH_CALUDE_pizza_slice_price_l2549_254964


namespace NUMINAMATH_CALUDE_union_of_sets_l2549_254972

theorem union_of_sets : 
  let M : Set Nat := {1, 2, 5}
  let N : Set Nat := {1, 3, 5, 7}
  M ∪ N = {1, 2, 3, 5, 7} := by
sorry

end NUMINAMATH_CALUDE_union_of_sets_l2549_254972


namespace NUMINAMATH_CALUDE_selection_theorem_l2549_254913

/-- The number of ways to select 3 people from 11, with at least one of A or B selected and C not selected -/
def selection_ways (n : ℕ) (k : ℕ) (total : ℕ) : ℕ :=
  (2 * Nat.choose (total - 3) (k - 1)) + Nat.choose (total - 3) (k - 2)

/-- Theorem stating that the number of selection ways is 64 -/
theorem selection_theorem : selection_ways 3 3 11 = 64 := by
  sorry

end NUMINAMATH_CALUDE_selection_theorem_l2549_254913


namespace NUMINAMATH_CALUDE_tommy_pencil_case_erasers_l2549_254982

/-- Represents the contents of Tommy's pencil case -/
structure PencilCase where
  total_items : ℕ
  pencils : ℕ
  pens : ℕ
  erasers : ℕ

/-- Theorem stating the number of erasers in Tommy's pencil case -/
theorem tommy_pencil_case_erasers (pc : PencilCase)
    (h1 : pc.total_items = 13)
    (h2 : pc.pens = 2 * pc.pencils)
    (h3 : pc.pencils = 4)
    (h4 : pc.total_items = pc.pencils + pc.pens + pc.erasers) :
    pc.erasers = 1 := by
  sorry

end NUMINAMATH_CALUDE_tommy_pencil_case_erasers_l2549_254982


namespace NUMINAMATH_CALUDE_submerged_sphere_pressure_l2549_254975

/-- The total water pressure on a submerged sphere -/
theorem submerged_sphere_pressure
  (diameter : ℝ) (depth : ℝ) (ρ : ℝ) (g : ℝ) :
  diameter = 4 →
  depth = 3 →
  ρ > 0 →
  g > 0 →
  (∫ x in (-2 : ℝ)..2, 4 * π * ρ * g * (depth + x)) = 64 * π * ρ * g :=
by sorry

end NUMINAMATH_CALUDE_submerged_sphere_pressure_l2549_254975


namespace NUMINAMATH_CALUDE_num_parallelepipeds_is_29_l2549_254996

/-- A set of 4 points in 3D space -/
structure PointSet :=
  (points : Fin 4 → ℝ × ℝ × ℝ)
  (not_coplanar : ∀ (p : ℝ × ℝ × ℝ → ℝ), ¬(∀ i, p (points i) = 0))

/-- A parallelepiped formed by 4 vertices -/
structure Parallelepiped :=
  (vertices : Fin 8 → ℝ × ℝ × ℝ)

/-- The number of distinct parallelepipeds that can be formed from a set of 4 points -/
def num_parallelepipeds (ps : PointSet) : ℕ :=
  -- Definition here (not implemented)
  0

/-- Theorem: The number of distinct parallelepipeds formed by 4 non-coplanar points is 29 -/
theorem num_parallelepipeds_is_29 (ps : PointSet) : num_parallelepipeds ps = 29 := by
  sorry

end NUMINAMATH_CALUDE_num_parallelepipeds_is_29_l2549_254996


namespace NUMINAMATH_CALUDE_percentage_5_years_plus_is_30_percent_l2549_254903

/-- Represents the number of employees for each year group -/
def employee_distribution : List ℕ := [5, 5, 8, 3, 2, 2, 2, 1, 1, 1]

/-- Calculates the total number of employees -/
def total_employees : ℕ := employee_distribution.sum

/-- Calculates the number of employees who have worked for 5 years or more -/
def employees_5_years_plus : ℕ := (employee_distribution.drop 5).sum

/-- Theorem: The percentage of employees who have worked at the Gauss company for 5 years or more is 30% -/
theorem percentage_5_years_plus_is_30_percent :
  (employees_5_years_plus : ℚ) / total_employees * 100 = 30 := by sorry

end NUMINAMATH_CALUDE_percentage_5_years_plus_is_30_percent_l2549_254903


namespace NUMINAMATH_CALUDE_max_area_triangle_l2549_254959

noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.cos x + Real.sqrt 3 * (Real.cos x)^2

noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x + 2 * Real.pi / 3)

theorem max_area_triangle (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < Real.pi / 2 →
  0 < B ∧ B < Real.pi / 2 →
  0 < C ∧ C < Real.pi / 2 →
  A + B + C = Real.pi →
  a > 0 ∧ b > 0 ∧ c > 0 →
  g (A / 2) = 1 / 2 →
  a = 1 →
  a^2 = b^2 + c^2 - 2 * b * c * Real.cos A →
  (1 / 2 * b * c * Real.sin A) ≤ (2 + Real.sqrt 3) / 4 :=
by sorry

end NUMINAMATH_CALUDE_max_area_triangle_l2549_254959


namespace NUMINAMATH_CALUDE_cookie_difference_l2549_254933

/-- The number of sweet cookies Paco had -/
def total_sweet : ℕ := 40

/-- The number of salty cookies Paco had -/
def total_salty : ℕ := 25

/-- The number of sweet cookies Paco ate -/
def sweet_eaten : ℕ := total_sweet * 2

/-- The number of salty cookies Paco ate -/
noncomputable def salty_eaten : ℕ := (total_salty * 5) / 3

theorem cookie_difference :
  (salty_eaten : ℤ) - sweet_eaten = -38 := by sorry

end NUMINAMATH_CALUDE_cookie_difference_l2549_254933


namespace NUMINAMATH_CALUDE_product_of_terms_l2549_254927

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The roots of the quadratic equation 2x^2 + 5x + 1 = 0 -/
def roots_of_equation (x y : ℝ) : Prop :=
  2 * x^2 + 5 * x + 1 = 0 ∧ 2 * y^2 + 5 * y + 1 = 0

theorem product_of_terms (a : ℕ → ℝ) :
  geometric_sequence a →
  roots_of_equation (a 1) (a 10) →
  a 4 * a 7 = 1/2 := by sorry

end NUMINAMATH_CALUDE_product_of_terms_l2549_254927


namespace NUMINAMATH_CALUDE_unique_solution_l2549_254979

/-- Represents the denomination of a coin -/
inductive Coin : Type
  | One : Coin
  | Two : Coin
  | Five : Coin
  | Ten : Coin
  | Twenty : Coin

/-- The value of a coin in forints -/
def coin_value : Coin → Nat
  | Coin.One => 1
  | Coin.Two => 2
  | Coin.Five => 5
  | Coin.Ten => 10
  | Coin.Twenty => 20

/-- Represents the count of each coin type -/
structure CoinCount where
  one : Nat
  two : Nat
  five : Nat
  ten : Nat
  twenty : Nat

/-- The given coin count from the problem -/
def problem_coin_count : CoinCount :=
  { one := 3, two := 9, five := 5, ten := 6, twenty := 3 }

/-- Check if a number can be represented by the given coin count -/
def can_represent (n : Nat) (cc : CoinCount) : Prop :=
  ∃ (a b c d e : Nat),
    a ≤ cc.twenty ∧ b ≤ cc.ten ∧ c ≤ cc.five ∧ d ≤ cc.two ∧ e ≤ cc.one ∧
    n = a * 20 + b * 10 + c * 5 + d * 2 + e * 1

/-- The set of drawn numbers -/
def drawn_numbers : Finset Nat :=
  {34, 33, 29, 19, 18, 17, 16}

/-- The theorem to be proved -/
theorem unique_solution :
  (∀ n ∈ drawn_numbers, n ≤ 35) ∧
  (drawn_numbers.card = 7) ∧
  (∀ n ∈ drawn_numbers, can_represent n problem_coin_count) ∧
  (∀ s : Finset Nat, s ≠ drawn_numbers →
    s.card = 7 →
    (∀ n ∈ s, n ≤ 35) →
    (∀ n ∈ s, can_represent n problem_coin_count) →
    False) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_l2549_254979


namespace NUMINAMATH_CALUDE_got_percentage_is_fifty_percent_l2549_254981

/-- Represents the vote counts for three books -/
structure VoteCounts where
  got : ℕ  -- Game of Thrones
  twi : ℕ  -- Twilight
  aotd : ℕ  -- The Art of the Deal

/-- Calculates the altered vote counts after throwing away votes -/
def alterVotes (vc : VoteCounts) : VoteCounts :=
  { got := vc.got,
    twi := vc.twi / 2,
    aotd := vc.aotd - (vc.aotd * 4 / 5) }

/-- Calculates the percentage of votes for Game of Thrones after alteration -/
def gotPercentage (vc : VoteCounts) : ℚ :=
  let altered := alterVotes vc
  altered.got * 100 / (altered.got + altered.twi + altered.aotd)

/-- Theorem stating that for the given vote counts, the percentage of
    altered votes for Game of Thrones is 50% -/
theorem got_percentage_is_fifty_percent :
  let original := VoteCounts.mk 10 12 20
  gotPercentage original = 50 := by sorry

end NUMINAMATH_CALUDE_got_percentage_is_fifty_percent_l2549_254981


namespace NUMINAMATH_CALUDE_discount_percentage_calculation_l2549_254914

/-- Given an article with a marked price and cost price, calculate the discount percentage. -/
theorem discount_percentage_calculation
  (marked_price : ℝ)
  (cost_price : ℝ)
  (h1 : cost_price = 0.64 * marked_price)
  (h2 : (cost_price * 1.375 - cost_price) / cost_price = 0.375) :
  (marked_price - cost_price * 1.375) / marked_price = 0.12 := by
  sorry

#check discount_percentage_calculation

end NUMINAMATH_CALUDE_discount_percentage_calculation_l2549_254914


namespace NUMINAMATH_CALUDE_inequality_range_l2549_254967

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Ioo 0 1 → a * x^3 - x^2 + 4*x + 3 ≥ 0) → 
  a ≥ -6 := by
  sorry

end NUMINAMATH_CALUDE_inequality_range_l2549_254967


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l2549_254947

theorem polynomial_division_theorem (x : ℝ) : 
  x^6 + 8 = (x - 1) * (x^5 + x^4 + x^3 + x^2 + x + 1) + 9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l2549_254947


namespace NUMINAMATH_CALUDE_xyz_inequality_l2549_254942

theorem xyz_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h : x * y * z ≥ x * y + y * z + z * x) : x * y * z ≥ 3 * (x + y + z) := by
  sorry

end NUMINAMATH_CALUDE_xyz_inequality_l2549_254942


namespace NUMINAMATH_CALUDE_amy_bought_21_tickets_l2549_254970

/-- Calculates the number of tickets Amy bought at the fair -/
def tickets_bought (initial_tickets total_tickets : ℕ) : ℕ :=
  total_tickets - initial_tickets

/-- Proves that Amy bought 21 tickets at the fair -/
theorem amy_bought_21_tickets :
  tickets_bought 33 54 = 21 := by
  sorry

end NUMINAMATH_CALUDE_amy_bought_21_tickets_l2549_254970


namespace NUMINAMATH_CALUDE_total_donation_to_orphanages_l2549_254905

theorem total_donation_to_orphanages (donation1 donation2 donation3 : ℝ) 
  (h1 : donation1 = 175)
  (h2 : donation2 = 225)
  (h3 : donation3 = 250) :
  donation1 + donation2 + donation3 = 650 :=
by sorry

end NUMINAMATH_CALUDE_total_donation_to_orphanages_l2549_254905


namespace NUMINAMATH_CALUDE_fourth_month_sale_l2549_254908

def average_sale : ℝ := 2500
def month1_sale : ℝ := 2435
def month2_sale : ℝ := 2920
def month3_sale : ℝ := 2855
def month5_sale : ℝ := 2560
def month6_sale : ℝ := 1000

theorem fourth_month_sale (x : ℝ) : 
  (month1_sale + month2_sale + month3_sale + x + month5_sale + month6_sale) / 6 = average_sale →
  x = 3230 := by
sorry

end NUMINAMATH_CALUDE_fourth_month_sale_l2549_254908


namespace NUMINAMATH_CALUDE_subtraction_difference_l2549_254983

theorem subtraction_difference (x : ℝ) : (x - 2152) - (x - 1264) = 888 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_difference_l2549_254983


namespace NUMINAMATH_CALUDE_average_age_problem_l2549_254957

theorem average_age_problem (devin_age eden_age mom_age : ℕ) : 
  devin_age = 12 →
  eden_age = 2 * devin_age →
  mom_age = 2 * eden_age →
  (devin_age + eden_age + mom_age) / 3 = 28 := by
  sorry

end NUMINAMATH_CALUDE_average_age_problem_l2549_254957


namespace NUMINAMATH_CALUDE_rectangle_area_l2549_254962

theorem rectangle_area (length width : ℝ) (h1 : length = Real.sqrt 6) (h2 : width = Real.sqrt 3) :
  length * width = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l2549_254962


namespace NUMINAMATH_CALUDE_problem_solution_l2549_254953

def A : Set ℝ := {x | x^2 - 2*x - 15 > 0}
def B : Set ℝ := {x | x - 6 < 0}

theorem problem_solution :
  (∀ m : ℝ, m ∈ A ↔ (m < -3 ∨ m > 5)) ∧
  (∀ m : ℝ, (m ∈ A ∨ m ∈ B) ∧ (m ∈ A ∧ m ∈ B) ↔ (m < -3 ∨ (5 < m ∧ m < 6))) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2549_254953


namespace NUMINAMATH_CALUDE_system_solution_l2549_254994

theorem system_solution : ∃! (x y : ℝ), 
  (2 * x + Real.sqrt (2 * x + 3 * y) - 3 * y = 5) ∧ 
  (4 * x^2 + 2 * x + 3 * y - 9 * y^2 = 32) ∧ 
  (x = 17/4) ∧ 
  (y = 5/2) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2549_254994


namespace NUMINAMATH_CALUDE_remainder_property_l2549_254940

theorem remainder_property (x y u v : ℕ) (h1 : 0 < x) (h2 : 0 < y) 
  (h3 : x = u * y + v) (h4 : 0 ≤ v) (h5 : v < y) : 
  (x + 3 * u * y) % y = v := by
sorry

end NUMINAMATH_CALUDE_remainder_property_l2549_254940


namespace NUMINAMATH_CALUDE_expression_equals_zero_l2549_254920

theorem expression_equals_zero :
  (π - 2023) ^ (0 : ℝ) - |1 - Real.sqrt 2| + 2 * Real.cos (π / 4) - (1 / 2) ^ (-1 : ℝ) = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_zero_l2549_254920


namespace NUMINAMATH_CALUDE_cube_sum_is_90_l2549_254984

-- Define the type for cube face numbers
def CubeFaces := Fin 6 → ℝ

-- Define the property of consecutive numbers
def IsConsecutive (faces : CubeFaces) : Prop :=
  ∀ i j : Fin 6, i.val < j.val → faces j - faces i = j.val - i.val

-- Define the property of opposite faces summing to 30
def OppositeFacesSum30 (faces : CubeFaces) : Prop :=
  faces 0 + faces 5 = 30 ∧ faces 1 + faces 4 = 30 ∧ faces 2 + faces 3 = 30

-- Theorem statement
theorem cube_sum_is_90 (faces : CubeFaces) 
  (h1 : IsConsecutive faces) (h2 : OppositeFacesSum30 faces) : 
  (Finset.univ.sum faces) = 90 := by sorry

end NUMINAMATH_CALUDE_cube_sum_is_90_l2549_254984


namespace NUMINAMATH_CALUDE_fill_time_both_pipes_l2549_254923

-- Define the time it takes for Pipe A to fill the tank
def pipeA_time : ℝ := 12

-- Define the rate at which Pipe B fills the tank relative to Pipe A
def pipeB_rate_multiplier : ℝ := 3

-- Theorem stating the time it takes to fill the tank with both pipes open
theorem fill_time_both_pipes (pipeA_time : ℝ) (pipeB_rate_multiplier : ℝ) 
  (h1 : pipeA_time > 0) (h2 : pipeB_rate_multiplier > 0) :
  (1 / (1 / pipeA_time + pipeB_rate_multiplier / pipeA_time)) = 3 := by
  sorry

#check fill_time_both_pipes

end NUMINAMATH_CALUDE_fill_time_both_pipes_l2549_254923


namespace NUMINAMATH_CALUDE_committee_choices_theorem_l2549_254995

/-- The number of ways to choose a committee with constraints -/
def committee_choices (total : ℕ) (women : ℕ) (men : ℕ) (committee_size : ℕ) (min_women : ℕ) : ℕ :=
  (Nat.choose women min_women) * (Nat.choose (total - min_women) (committee_size - min_women))

/-- Theorem: The number of ways to choose a 5-person committee from a club of 12 people
    (7 women and 5 men), where the committee must include at least 2 women, is 2520 -/
theorem committee_choices_theorem :
  committee_choices 12 7 5 5 2 = 2520 := by
  sorry

#eval committee_choices 12 7 5 5 2

end NUMINAMATH_CALUDE_committee_choices_theorem_l2549_254995


namespace NUMINAMATH_CALUDE_no_root_in_interval_l2549_254989

-- Define the function f(x) = x^5 - 3x - 1
def f (x : ℝ) : ℝ := x^5 - 3*x - 1

-- State the theorem
theorem no_root_in_interval :
  (∀ x ∈ Set.Ioo 2 3, f x ≠ 0) ∧ Continuous f := by sorry

end NUMINAMATH_CALUDE_no_root_in_interval_l2549_254989


namespace NUMINAMATH_CALUDE_andy_weight_change_l2549_254915

/-- Calculates Andy's weight change over the year -/
theorem andy_weight_change (initial_weight : ℝ) (weight_gain : ℝ) (months : ℕ) : 
  initial_weight = 156 →
  weight_gain = 36 →
  months = 3 →
  initial_weight - (initial_weight + weight_gain) * (1 - 1/8)^months = 36 := by
  sorry

#check andy_weight_change

end NUMINAMATH_CALUDE_andy_weight_change_l2549_254915


namespace NUMINAMATH_CALUDE_ratio_equality_l2549_254924

theorem ratio_equality (x y z : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0)
  (h_diff : x ≠ y ∧ y ≠ z ∧ x ≠ z)
  (h_eq : y / (2*x - z) = (x + y) / (2*z) ∧ (x + y) / (2*z) = x / y) :
  x / y = 3 := by
sorry

end NUMINAMATH_CALUDE_ratio_equality_l2549_254924


namespace NUMINAMATH_CALUDE_square_root_fraction_simplification_l2549_254973

theorem square_root_fraction_simplification :
  (Real.sqrt (7^2 + 24^2)) / (Real.sqrt (49 + 16)) = 5 * Real.sqrt 65 / 13 := by
  sorry

end NUMINAMATH_CALUDE_square_root_fraction_simplification_l2549_254973


namespace NUMINAMATH_CALUDE_melanie_grew_more_turnips_l2549_254952

/-- The number of turnips Melanie grew -/
def melanie_turnips : ℕ := 139

/-- The number of turnips Benny grew -/
def benny_turnips : ℕ := 113

/-- The difference in turnips grown between Melanie and Benny -/
def turnip_difference : ℕ := melanie_turnips - benny_turnips

theorem melanie_grew_more_turnips : turnip_difference = 26 := by
  sorry

end NUMINAMATH_CALUDE_melanie_grew_more_turnips_l2549_254952


namespace NUMINAMATH_CALUDE_locus_of_circumscribed_rectangles_centers_l2549_254987

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a triangle in 2D space -/
structure Triangle where
  A : Point2D
  B : Point2D
  C : Point2D

/-- Represents a rectangle in 2D space -/
structure Rectangle where
  center : Point2D
  width : ℝ
  height : ℝ

/-- Represents a curvilinear triangle formed by semicircles -/
structure CurvilinearTriangle where
  midpoints : Triangle  -- Represents the triangle formed by midpoints of the original triangle

/-- Checks if a triangle is acute-angled -/
def isAcuteTriangle (t : Triangle) : Prop :=
  sorry  -- Definition of acute triangle

/-- Checks if a rectangle is circumscribed around a triangle -/
def isCircumscribed (r : Rectangle) (t : Triangle) : Prop :=
  sorry  -- Definition of circumscribed rectangle

/-- Computes the midpoints of a triangle -/
def midpoints (t : Triangle) : Triangle :=
  sorry  -- Computation of midpoints

/-- Checks if a point is on the locus (curvilinear triangle) -/
def isOnLocus (p : Point2D) (ct : CurvilinearTriangle) : Prop :=
  sorry  -- Definition of being on the locus

/-- The main theorem -/
theorem locus_of_circumscribed_rectangles_centers 
  (t : Triangle) (h : isAcuteTriangle t) :
  ∀ (r : Rectangle), isCircumscribed r t → 
    isOnLocus r.center (CurvilinearTriangle.mk (midpoints t)) :=
  sorry

end NUMINAMATH_CALUDE_locus_of_circumscribed_rectangles_centers_l2549_254987


namespace NUMINAMATH_CALUDE_train_length_l2549_254990

/-- The length of a train given its speed and time to cross an electric pole. -/
theorem train_length (speed_kmh : ℝ) (time_sec : ℝ) (length : ℝ) : 
  speed_kmh = 72 → time_sec = 15 → length = (speed_kmh * 1000 / 3600) * time_sec → length = 300 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l2549_254990


namespace NUMINAMATH_CALUDE_sqrt_49_times_sqrt_25_squared_l2549_254934

theorem sqrt_49_times_sqrt_25_squared : (Real.sqrt (49 * Real.sqrt 25))^2 = 245 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_49_times_sqrt_25_squared_l2549_254934


namespace NUMINAMATH_CALUDE_evaluate_expression_l2549_254978

theorem evaluate_expression (c d : ℝ) (h1 : c = 3) (h2 : d = 2) :
  (c^2 + d)^2 - (c^2 - d)^2 = 72 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2549_254978


namespace NUMINAMATH_CALUDE_camping_activities_count_l2549_254976

/-- The number of campers who went rowing and hiking in total, considering both morning and afternoon sessions -/
def total_rowing_and_hiking (total_campers : ℕ)
  (morning_rowing : ℕ) (morning_hiking : ℕ) (morning_swimming : ℕ)
  (afternoon_rowing : ℕ) (afternoon_hiking : ℕ) : ℕ :=
  morning_rowing + afternoon_rowing + morning_hiking + afternoon_hiking

/-- Theorem stating that the total number of campers who went rowing and hiking is 79 -/
theorem camping_activities_count
  (total_campers : ℕ)
  (morning_rowing : ℕ) (morning_hiking : ℕ) (morning_swimming : ℕ)
  (afternoon_rowing : ℕ) (afternoon_hiking : ℕ)
  (h1 : total_campers = 80)
  (h2 : morning_rowing = 41)
  (h3 : morning_hiking = 4)
  (h4 : morning_swimming = 15)
  (h5 : afternoon_rowing = 26)
  (h6 : afternoon_hiking = 8) :
  total_rowing_and_hiking total_campers morning_rowing morning_hiking morning_swimming afternoon_rowing afternoon_hiking = 79 := by
  sorry

#check camping_activities_count

end NUMINAMATH_CALUDE_camping_activities_count_l2549_254976


namespace NUMINAMATH_CALUDE_function_properties_l2549_254921

/-- Given a function f with parameter ω, prove properties about its graph -/
theorem function_properties (ω : ℝ) (h_ω_pos : ω > 0) :
  let f : ℝ → ℝ := λ x ↦ Real.sqrt 3 * Real.sin (ω * x) + 2 * (Real.sin (ω * x / 2))^2
  -- Assume the graph has exactly three symmetric centers on [0, π]
  (∃ (x₁ x₂ x₃ : ℝ), 0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ ≤ π ∧
    (∀ (y : ℝ), 0 ≤ y ∧ y ≤ π → (f y = f (2 * x₁ - y) ∨ f y = f (2 * x₂ - y) ∨ f y = f (2 * x₃ - y))) ∧
    (∀ (z : ℝ), 0 ≤ z ∧ z ≤ π → (z = x₁ ∨ z = x₂ ∨ z = x₃ ∨ f z ≠ f (2 * z - z)))) →
  -- Then prove:
  (13/6 ≤ ω ∧ ω < 19/6) ∧  -- 1. Range of ω
  (∃ (n : ℕ), n = 2 ∨ n = 3 ∧  -- 2. Number of axes of symmetry
    ∃ (x₁ x₂ x₃ : ℝ), 0 < x₁ ∧ x₁ < x₂ ∧ (n = 3 → x₂ < x₃) ∧ x₃ < π ∧
    (∀ (y : ℝ), 0 ≤ y ∧ y ≤ π → (f y = f (2 * x₁ - y) ∨ f y = f (2 * x₂ - y) ∨ (n = 3 → f y = f (2 * x₃ - y))))) ∧
  (∃ (x : ℝ), 0 < x ∧ x < π/4 ∧ f x = 3) ∧  -- 3. Maximum value on (0, π/4)
  (∀ (x y : ℝ), 0 < x ∧ x < y ∧ y < π/6 → f x < f y)  -- 4. Increasing on (0, π/6)
  := by sorry

end NUMINAMATH_CALUDE_function_properties_l2549_254921


namespace NUMINAMATH_CALUDE_sandwich_fraction_l2549_254936

theorem sandwich_fraction (total : ℝ) (ticket_fraction : ℝ) (book_fraction : ℝ) (leftover : ℝ) 
  (h1 : total = 90)
  (h2 : ticket_fraction = 1/6)
  (h3 : book_fraction = 1/2)
  (h4 : leftover = 12) :
  ∃ (sandwich_fraction : ℝ), 
    sandwich_fraction * total + ticket_fraction * total + book_fraction * total + leftover = total ∧ 
    sandwich_fraction = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_fraction_l2549_254936


namespace NUMINAMATH_CALUDE_system_solution_unique_l2549_254928

theorem system_solution_unique : 
  ∃! (x y : ℝ), 2 * x - y = 3 ∧ 3 * x + 2 * y = 8 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_system_solution_unique_l2549_254928


namespace NUMINAMATH_CALUDE_point_B_coordinates_l2549_254931

-- Define the coordinate system
def Point := ℝ × ℝ

-- Define point A
def A : Point := (2, 1)

-- Define the length of AB
def AB_length : ℝ := 4

-- Define a function to check if a point is on the line parallel to x-axis passing through A
def on_parallel_line (p : Point) : Prop :=
  p.2 = A.2

-- Define a function to check if the distance between two points is correct
def correct_distance (p : Point) : Prop :=
  (p.1 - A.1)^2 + (p.2 - A.2)^2 = AB_length^2

-- Theorem statement
theorem point_B_coordinates :
  ∃ (B : Point), on_parallel_line B ∧ correct_distance B ∧
  (B = (6, 1) ∨ B = (-2, 1)) :=
sorry

end NUMINAMATH_CALUDE_point_B_coordinates_l2549_254931


namespace NUMINAMATH_CALUDE_floor_area_less_than_ten_l2549_254954

/-- Represents a rectangular room -/
structure Room where
  length : ℝ
  width : ℝ
  height : ℝ

/-- The condition that the room's height is 3 meters -/
def height_is_three (r : Room) : Prop :=
  r.height = 3

/-- The condition that each wall's area is greater than the floor area -/
def walls_larger_than_floor (r : Room) : Prop :=
  r.length * r.height > r.length * r.width ∧ 
  r.width * r.height > r.length * r.width

/-- The theorem stating that under the given conditions, 
    the floor area must be less than 10 square meters -/
theorem floor_area_less_than_ten (r : Room) 
  (h1 : height_is_three r) 
  (h2 : walls_larger_than_floor r) : 
  r.length * r.width < 10 := by
  sorry


end NUMINAMATH_CALUDE_floor_area_less_than_ten_l2549_254954


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l2549_254956

/-- Given an arithmetic sequence {a_n} where n ∈ ℕ+, if a_n + a_{n+2} = 4n + 6,
    then a_n = 2n + 1 for all n ∈ ℕ+ -/
theorem arithmetic_sequence_general_term
  (a : ℕ+ → ℝ)  -- a is a function from positive naturals to reals
  (h : ∀ n : ℕ+, a n + a (n + 2) = 4 * n + 6) :  -- given condition
  ∀ n : ℕ+, a n = 2 * n + 1 :=  -- conclusion to prove
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l2549_254956


namespace NUMINAMATH_CALUDE_total_cost_after_discount_l2549_254917

def child_ticket_price : ℚ := 4.25
def adult_ticket_price : ℚ := child_ticket_price + 3.5
def senior_ticket_price : ℚ := adult_ticket_price - 1.75
def discount_per_5_tickets : ℚ := 3
def num_adult_tickets : ℕ := 2
def num_child_tickets : ℕ := 4
def num_senior_tickets : ℕ := 1

def total_ticket_cost : ℚ :=
  num_adult_tickets * adult_ticket_price +
  num_child_tickets * child_ticket_price +
  num_senior_tickets * senior_ticket_price

def total_tickets : ℕ := num_adult_tickets + num_child_tickets + num_senior_tickets

def discount_amount : ℚ := (total_tickets / 5 : ℚ) * discount_per_5_tickets

theorem total_cost_after_discount :
  total_ticket_cost - discount_amount = 35.5 := by sorry

end NUMINAMATH_CALUDE_total_cost_after_discount_l2549_254917


namespace NUMINAMATH_CALUDE_animals_to_shore_l2549_254966

theorem animals_to_shore (initial_sheep initial_cows initial_dogs : ℕ) 
  (drowned_sheep : ℕ) (h1 : initial_sheep = 20) (h2 : initial_cows = 10) 
  (h3 : initial_dogs = 14) (h4 : drowned_sheep = 3) 
  (h5 : 2 * drowned_sheep = initial_cows - (initial_cows - 2 * drowned_sheep)) :
  initial_sheep - drowned_sheep + (initial_cows - 2 * drowned_sheep) + initial_dogs = 35 := by
  sorry

end NUMINAMATH_CALUDE_animals_to_shore_l2549_254966


namespace NUMINAMATH_CALUDE_find_m_l2549_254955

-- Define the inequality
def inequality (x m : ℝ) : Prop := -1/2 * x^2 + 2*x > -m*x

-- Define the solution set
def solution_set (m : ℝ) : Set ℝ := {x : ℝ | 0 < x ∧ x < 2}

-- Theorem statement
theorem find_m : 
  ∀ m : ℝ, (∀ x : ℝ, inequality x m ↔ x ∈ solution_set m) → m = -1 :=
sorry

end NUMINAMATH_CALUDE_find_m_l2549_254955


namespace NUMINAMATH_CALUDE_art_fair_sales_l2549_254958

/-- The total number of paintings sold at Tracy's art fair booth -/
def total_paintings_sold (first_group : Nat) (second_group : Nat) (third_group : Nat)
  (first_group_purchase : Nat) (second_group_purchase : Nat) (third_group_purchase : Nat) : Nat :=
  first_group * first_group_purchase +
  second_group * second_group_purchase +
  third_group * third_group_purchase

/-- Theorem stating the total number of paintings sold at Tracy's art fair booth -/
theorem art_fair_sales :
  total_paintings_sold 4 12 4 2 1 4 = 36 := by
  sorry

end NUMINAMATH_CALUDE_art_fair_sales_l2549_254958


namespace NUMINAMATH_CALUDE_exists_common_tiling_l2549_254929

/-- Represents a domino type with integer dimensions -/
structure Domino where
  length : ℤ
  width : ℤ

/-- Checks if a rectangle can be tiled by a given domino type -/
def canTile (d : Domino) (rectLength rectWidth : ℤ) : Prop :=
  rectLength ≥ max 1 (2 * d.length) ∧ rectWidth % (2 * d.width) = 0

/-- Proves the existence of a rectangle that can be tiled by either of two domino types -/
theorem exists_common_tiling (d1 d2 : Domino) : 
  ∃ (rectLength rectWidth : ℤ), 
    canTile d1 rectLength rectWidth ∧ canTile d2 rectLength rectWidth :=
by
  sorry

end NUMINAMATH_CALUDE_exists_common_tiling_l2549_254929


namespace NUMINAMATH_CALUDE_right_triangle_arctan_sum_l2549_254998

theorem right_triangle_arctan_sum (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  a^2 + c^2 = b^2 →
  Real.arctan (a / (c + b)) + Real.arctan (c / (a + b)) = π / 4 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_arctan_sum_l2549_254998


namespace NUMINAMATH_CALUDE_range_of_cos_2x_plus_2sin_x_l2549_254900

open Real

theorem range_of_cos_2x_plus_2sin_x :
  ∀ x ∈ Set.Ioo 0 π,
  ∃ y ∈ Set.Icc 1 (3/2),
  y = cos (2*x) + 2 * sin x ∧
  ∀ z, z = cos (2*x) + 2 * sin x → z ∈ Set.Icc 1 (3/2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_cos_2x_plus_2sin_x_l2549_254900


namespace NUMINAMATH_CALUDE_congruence_systems_solutions_l2549_254925

theorem congruence_systems_solutions :
  (∃ x : ℤ, x % 7 = 3 ∧ (6 * x) % 8 = 10) ∧
  (∀ x : ℤ, x % 7 = 3 ∧ (6 * x) % 8 = 10 → x % 56 = 3 ∨ x % 56 = 31) ∧
  (¬ ∃ x : ℤ, (3 * x) % 10 = 1 ∧ (4 * x) % 15 = 7) :=
by sorry

end NUMINAMATH_CALUDE_congruence_systems_solutions_l2549_254925


namespace NUMINAMATH_CALUDE_area_ratio_hexagons_l2549_254930

/-- A point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A regular hexagon -/
structure RegularHexagon :=
  (center : Point)
  (sideLength : ℝ)

/-- An equilateral triangle -/
structure EquilateralTriangle :=
  (center : Point)
  (sideLength : ℝ)

/-- The hexagon ABCD -/
def hexagonABCD : RegularHexagon := sorry

/-- The equilateral triangles constructed on the sides of ABCD -/
def trianglesOnABCD : List EquilateralTriangle := sorry

/-- The hexagon EFGHIJ formed by the centers of the equilateral triangles -/
def hexagonEFGHIJ : RegularHexagon := sorry

/-- The area of a regular hexagon -/
def areaRegularHexagon (h : RegularHexagon) : ℝ := sorry

/-- Theorem: The ratio of the area of hexagon EFGHIJ to the area of hexagon ABCD is 4/3 -/
theorem area_ratio_hexagons :
  (areaRegularHexagon hexagonEFGHIJ) / (areaRegularHexagon hexagonABCD) = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_hexagons_l2549_254930
