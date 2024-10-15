import Mathlib

namespace NUMINAMATH_CALUDE_square_plate_nails_l1983_198318

/-- The number of nails on each side of the square plate -/
def nails_per_side : ℕ := 25

/-- The number of sides of a square -/
def sides_of_square : ℕ := 4

/-- The number of corners in a square -/
def corners_of_square : ℕ := 4

/-- The total number of nails used to fix the square plate -/
def total_nails : ℕ := nails_per_side * sides_of_square - corners_of_square

theorem square_plate_nails :
  total_nails = 96 := by sorry

end NUMINAMATH_CALUDE_square_plate_nails_l1983_198318


namespace NUMINAMATH_CALUDE_total_miles_driven_is_2225_l1983_198333

/-- A structure representing a car's weekly fuel consumption and mileage. -/
structure Car where
  gallons_consumed : ℝ
  average_mpg : ℝ

/-- Calculates the total miles driven by a car given its fuel consumption and average mpg. -/
def miles_driven (car : Car) : ℝ :=
  car.gallons_consumed * car.average_mpg

/-- Represents the family's two cars and their combined mileage. -/
structure FamilyCars where
  car1 : Car
  car2 : Car
  total_average_mpg : ℝ

/-- Theorem stating that under the given conditions, the total miles driven by both cars is 2225. -/
theorem total_miles_driven_is_2225 (family_cars : FamilyCars)
    (h1 : family_cars.car1.gallons_consumed = 25)
    (h2 : family_cars.car2.gallons_consumed = 35)
    (h3 : family_cars.car1.average_mpg = 40)
    (h4 : family_cars.total_average_mpg = 75) :
    miles_driven family_cars.car1 + miles_driven family_cars.car2 = 2225 := by
  sorry


end NUMINAMATH_CALUDE_total_miles_driven_is_2225_l1983_198333


namespace NUMINAMATH_CALUDE_sqrt_sum_equality_l1983_198369

theorem sqrt_sum_equality (x : ℝ) : 
  Real.sqrt (x^2 + 4*x + 4) + Real.sqrt (x^2 - 6*x + 9) = |x + 2| + |x - 3| := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equality_l1983_198369


namespace NUMINAMATH_CALUDE_card_sets_l1983_198320

def is_valid_card_set (a b c d : ℕ) : Prop :=
  0 < a ∧ a < b ∧ b < c ∧ c < d ∧ d ≤ 9 ∧
  (([a + b, a + c, a + d, b + c, b + d, c + d].filter (· = 9)).length = 2) ∧
  (([a + b, a + c, a + d, b + c, b + d, c + d].filter (· < 9)).length = 2) ∧
  (([a + b, a + c, a + d, b + c, b + d, c + d].filter (· > 9)).length = 2)

theorem card_sets :
  ∀ a b c d : ℕ,
    is_valid_card_set a b c d ↔
      (a = 1 ∧ b = 2 ∧ c = 7 ∧ d = 8) ∨
      (a = 1 ∧ b = 3 ∧ c = 6 ∧ d = 8) ∨
      (a = 1 ∧ b = 4 ∧ c = 5 ∧ d = 8) ∨
      (a = 2 ∧ b = 3 ∧ c = 6 ∧ d = 7) ∨
      (a = 2 ∧ b = 4 ∧ c = 5 ∧ d = 7) ∨
      (a = 3 ∧ b = 4 ∧ c = 5 ∧ d = 6) :=
by sorry

end NUMINAMATH_CALUDE_card_sets_l1983_198320


namespace NUMINAMATH_CALUDE_equation_solution_l1983_198311

theorem equation_solution (x : ℝ) : 
  x ≠ 2 → ((4*x^2 + 3*x + 2)/(x - 2) = 4*x + 5 ↔ x = -4) := by sorry

end NUMINAMATH_CALUDE_equation_solution_l1983_198311


namespace NUMINAMATH_CALUDE_complex_modulus_equality_l1983_198341

theorem complex_modulus_equality (m : ℝ) (h : m > 0) :
  Complex.abs (4 + m * Complex.I) = 4 * Real.sqrt 13 → m = 8 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_equality_l1983_198341


namespace NUMINAMATH_CALUDE_consecutive_lcm_inequality_l1983_198366

theorem consecutive_lcm_inequality : ∃ n : ℕ, 
  Nat.lcm (Nat.lcm n (n + 1)) (n + 2) > Nat.lcm (Nat.lcm (n + 3) (n + 4)) (n + 5) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_lcm_inequality_l1983_198366


namespace NUMINAMATH_CALUDE_existence_of_special_integer_l1983_198398

theorem existence_of_special_integer :
  ∃ (n : ℕ), n ≥ 2^2018 ∧
  ∀ (x y u v : ℕ), u > 1 → v > 1 → n ≠ x^u + y^v :=
by sorry

end NUMINAMATH_CALUDE_existence_of_special_integer_l1983_198398


namespace NUMINAMATH_CALUDE_apple_distribution_l1983_198305

theorem apple_distribution (total_apples : ℕ) (apples_per_classmate : ℕ) 
  (h1 : total_apples = 15) (h2 : apples_per_classmate = 5) : 
  total_apples / apples_per_classmate = 3 := by
  sorry

end NUMINAMATH_CALUDE_apple_distribution_l1983_198305


namespace NUMINAMATH_CALUDE_intersection_range_l1983_198390

theorem intersection_range (k₁ k₂ t p q m n : ℝ) : 
  k₁ > 0 → k₂ > 0 → 
  k₁ * 1 = k₂ / 1 →
  t ≠ 0 → t ≠ -2 →
  p = k₁ * t →
  q = k₁ * (t + 2) →
  m = k₂ / t →
  n = k₂ / (t + 2) →
  (p - m) * (q - n) < 0 ↔ (-3 < t ∧ t < -2) ∨ (0 < t ∧ t < 1) :=
by sorry

end NUMINAMATH_CALUDE_intersection_range_l1983_198390


namespace NUMINAMATH_CALUDE_rectangular_field_dimensions_l1983_198337

theorem rectangular_field_dimensions (m : ℕ) : 
  (3 * m + 10) * (m - 5) = 72 → m = 7 := by sorry

end NUMINAMATH_CALUDE_rectangular_field_dimensions_l1983_198337


namespace NUMINAMATH_CALUDE_min_omega_for_sine_symmetry_l1983_198352

theorem min_omega_for_sine_symmetry :
  ∀ ω : ℕ+,
  (∀ x : ℝ, ∃ y : ℝ, y = Real.sin (ω * x + π / 6)) →
  (∀ x : ℝ, Real.sin (ω * (π / 3 - x) + π / 6) = Real.sin (ω * x + π / 6)) →
  2 ≤ ω :=
by
  sorry

end NUMINAMATH_CALUDE_min_omega_for_sine_symmetry_l1983_198352


namespace NUMINAMATH_CALUDE_BC_length_is_580_l1983_198315

-- Define a structure for a quadrilateral
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define properties of the quadrilateral
def is_convex (q : Quadrilateral) : Prop := sorry

def has_integer_lengths (q : Quadrilateral) : Prop := sorry

def right_angle_at_B_and_D (q : Quadrilateral) : Prop := sorry

def AB_equals_BD (q : Quadrilateral) : Prop := sorry

def CD_equals_41 (q : Quadrilateral) : Prop := sorry

-- Define the length of BC
def BC_length (q : Quadrilateral) : ℝ := sorry

-- Theorem statement
theorem BC_length_is_580 (q : Quadrilateral) 
  (h_convex : is_convex q)
  (h_integer : has_integer_lengths q)
  (h_right_angles : right_angle_at_B_and_D q)
  (h_AB_BD : AB_equals_BD q)
  (h_CD_41 : CD_equals_41 q) :
  BC_length q = 580 := by sorry

end NUMINAMATH_CALUDE_BC_length_is_580_l1983_198315


namespace NUMINAMATH_CALUDE_equation_solution_l1983_198302

theorem equation_solution :
  ∃! x : ℚ, (x^2 + 2*x + 2) / (x + 2) = x + 3 :=
by
  use (-4/3)
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1983_198302


namespace NUMINAMATH_CALUDE_deal_or_no_deal_probability_l1983_198314

/-- The total number of boxes in the game --/
def total_boxes : ℕ := 26

/-- The number of boxes containing at least $250,000 --/
def high_value_boxes : ℕ := 6

/-- The number of boxes to eliminate --/
def boxes_to_eliminate : ℕ := 8

/-- The probability of selecting a high-value box after elimination --/
def probability_high_value : ℚ := 1 / 3

theorem deal_or_no_deal_probability :
  (high_value_boxes : ℚ) / (total_boxes - boxes_to_eliminate : ℚ) = probability_high_value :=
sorry

end NUMINAMATH_CALUDE_deal_or_no_deal_probability_l1983_198314


namespace NUMINAMATH_CALUDE_jack_additional_money_l1983_198327

/-- The amount of additional money Jack needs to buy socks and shoes -/
theorem jack_additional_money (sock_cost shoes_cost jack_money : ℚ)
  (h1 : sock_cost = 19)
  (h2 : shoes_cost = 92)
  (h3 : jack_money = 40) :
  sock_cost + shoes_cost - jack_money = 71 := by
  sorry

end NUMINAMATH_CALUDE_jack_additional_money_l1983_198327


namespace NUMINAMATH_CALUDE_quadratic_equation_problem_l1983_198360

theorem quadratic_equation_problem (m : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, x^2 + 2*m*x + m^2 - m + 2 = 0 ↔ x = x₁ ∨ x = x₂) →
  x₁ ≠ x₂ →
  x₁ + x₂ + x₁ * x₂ = 2 →
  m = 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_problem_l1983_198360


namespace NUMINAMATH_CALUDE_triangle_inequality_with_altitudes_l1983_198370

/-- Given a triangle with sides a > b and corresponding altitudes h_a and h_b,
    prove that a + h_a ≥ b + h_b with equality iff the angle between a and b is 90° -/
theorem triangle_inequality_with_altitudes (a b h_a h_b : ℝ) (S : ℝ) (γ : ℝ) :
  a > b →
  S = (1/2) * a * h_a →
  S = (1/2) * b * h_b →
  S = (1/2) * a * b * Real.sin γ →
  (a + h_a ≥ b + h_b) ∧ (a + h_a = b + h_b ↔ γ = Real.pi / 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_with_altitudes_l1983_198370


namespace NUMINAMATH_CALUDE_rational_sum_of_three_cubes_l1983_198350

theorem rational_sum_of_three_cubes (t : ℚ) : 
  ∃ (x y z : ℚ), t = x^3 + y^3 + z^3 := by
  sorry

end NUMINAMATH_CALUDE_rational_sum_of_three_cubes_l1983_198350


namespace NUMINAMATH_CALUDE_gcd_count_for_product_600_l1983_198384

theorem gcd_count_for_product_600 : 
  ∃ (S : Finset Nat), 
    (∀ d ∈ S, ∃ a b : Nat, 
      gcd a b = d ∧ Nat.lcm a b * d = 600) ∧
    (∀ d : Nat, (∃ a b : Nat, 
      gcd a b = d ∧ Nat.lcm a b * d = 600) → d ∈ S) ∧
    S.card = 14 := by
  sorry

end NUMINAMATH_CALUDE_gcd_count_for_product_600_l1983_198384


namespace NUMINAMATH_CALUDE_parallel_x_implies_parallel_y_implies_on_bisector_implies_l1983_198362

-- Define the coordinates of points A and B
def A (a : ℝ) : ℝ × ℝ := (a - 1, 2)
def B (b : ℝ) : ℝ × ℝ := (-3, b + 1)

-- Define the conditions
def parallel_to_x_axis (a b : ℝ) : Prop := (A a).2 = (B b).2
def parallel_to_y_axis (a b : ℝ) : Prop := (A a).1 = (B b).1
def on_bisector (a b : ℝ) : Prop := (A a).1 = (A a).2 ∧ (B b).1 = (B b).2

-- Theorem statements
theorem parallel_x_implies (a b : ℝ) : parallel_to_x_axis a b → a ≠ -2 ∧ b = 1 := by sorry

theorem parallel_y_implies (a b : ℝ) : parallel_to_y_axis a b → a = -2 ∧ b ≠ 1 := by sorry

theorem on_bisector_implies (a b : ℝ) : on_bisector a b → a = 3 ∧ b = -4 := by sorry

end NUMINAMATH_CALUDE_parallel_x_implies_parallel_y_implies_on_bisector_implies_l1983_198362


namespace NUMINAMATH_CALUDE_sum_of_roots_cubic_sum_of_roots_specific_cubic_l1983_198354

theorem sum_of_roots_cubic (a b c d : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^3 + b * x^2 + c * x + d
  (∃ x y z : ℝ, f x = 0 ∧ f y = 0 ∧ f z = 0) →
  x + y + z = -b / a :=
sorry

theorem sum_of_roots_specific_cubic :
  let f : ℝ → ℝ := λ x => 25 * x^3 - 50 * x^2 + 35 * x + 7
  (∃ x y z : ℝ, f x = 0 ∧ f y = 0 ∧ f z = 0) →
  x + y + z = 2 :=
sorry

end NUMINAMATH_CALUDE_sum_of_roots_cubic_sum_of_roots_specific_cubic_l1983_198354


namespace NUMINAMATH_CALUDE_square_inscribed_in_circle_sum_of_cube_and_reciprocal_polygon_diagonals_distance_between_points_l1983_198389

-- Problem G10.1
theorem square_inscribed_in_circle (d : ℝ) (A : ℝ) (h : d = 10) :
  A = (d^2) / 2 → A = 50 := by sorry

-- Problem G10.2
theorem sum_of_cube_and_reciprocal (a : ℝ) (S : ℝ) (h : a + 1/a = 2) :
  S = a^3 + 1/(a^3) → S = 2 := by sorry

-- Problem G10.3
theorem polygon_diagonals (n : ℕ) :
  n * (n - 3) / 2 = 14 → n = 7 := by sorry

-- Problem G10.4
theorem distance_between_points (d : ℝ) :
  d = Real.sqrt ((2 - (-1))^2 + (3 - 7)^2) → d = 5 := by sorry

end NUMINAMATH_CALUDE_square_inscribed_in_circle_sum_of_cube_and_reciprocal_polygon_diagonals_distance_between_points_l1983_198389


namespace NUMINAMATH_CALUDE_coin_toss_probability_l1983_198373

theorem coin_toss_probability : 
  let n : ℕ := 5
  let p_tail : ℚ := 1 / 2
  let p_all_tails : ℚ := p_tail ^ n
  let p_at_least_one_head : ℚ := 1 - p_all_tails
  p_at_least_one_head = 31 / 32 := by
sorry

end NUMINAMATH_CALUDE_coin_toss_probability_l1983_198373


namespace NUMINAMATH_CALUDE_product_sum_equality_l1983_198346

theorem product_sum_equality : 1520 * 1997 * 0.152 * 100 + 152^2 = 46161472 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_equality_l1983_198346


namespace NUMINAMATH_CALUDE_wednesday_water_intake_total_water_intake_correct_l1983_198356

/-- Represents the days of the week -/
inductive Day
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Water intake for a given day -/
def water_intake (d : Day) : ℕ :=
  match d with
  | Day.Monday => 9
  | Day.Tuesday => 8
  | Day.Wednesday => 9  -- This is what we want to prove
  | Day.Thursday => 9
  | Day.Friday => 8
  | Day.Saturday => 9
  | Day.Sunday => 8

/-- Total water intake for the week -/
def total_water_intake : ℕ := 60

/-- Theorem: The water intake on Wednesday is 9 liters -/
theorem wednesday_water_intake :
  water_intake Day.Wednesday = 9 :=
by
  sorry

/-- Theorem: The total water intake for the week is correct -/
theorem total_water_intake_correct :
  (water_intake Day.Monday) +
  (water_intake Day.Tuesday) +
  (water_intake Day.Wednesday) +
  (water_intake Day.Thursday) +
  (water_intake Day.Friday) +
  (water_intake Day.Saturday) +
  (water_intake Day.Sunday) = total_water_intake :=
by
  sorry

end NUMINAMATH_CALUDE_wednesday_water_intake_total_water_intake_correct_l1983_198356


namespace NUMINAMATH_CALUDE_exam_failure_rate_l1983_198348

/-- Examination results -/
structure ExamResults where
  total_candidates : ℕ
  num_girls : ℕ
  boys_math_pass_rate : ℚ
  boys_science_pass_rate : ℚ
  boys_lang_pass_rate : ℚ
  girls_math_pass_rate : ℚ
  girls_science_pass_rate : ℚ
  girls_lang_pass_rate : ℚ

/-- Calculate the failure rate given exam results -/
def calculate_failure_rate (results : ExamResults) : ℚ :=
  let num_boys := results.total_candidates - results.num_girls
  let boys_passing := min (results.boys_math_pass_rate * num_boys)
                          (min (results.boys_science_pass_rate * num_boys)
                               (results.boys_lang_pass_rate * num_boys))
  let girls_passing := min (results.girls_math_pass_rate * results.num_girls)
                           (min (results.girls_science_pass_rate * results.num_girls)
                                (results.girls_lang_pass_rate * results.num_girls))
  let total_passing := boys_passing + girls_passing
  let total_failing := results.total_candidates - total_passing
  total_failing / results.total_candidates

/-- The main theorem about the examination failure rate -/
theorem exam_failure_rate :
  let results := ExamResults.mk 2500 1100 (42/100) (39/100) (36/100) (35/100) (32/100) (40/100)
  calculate_failure_rate results = 6576/10000 := by
  sorry


end NUMINAMATH_CALUDE_exam_failure_rate_l1983_198348


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1983_198393

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x > 0 → x * Real.exp x > 0)) ↔ (∃ x₀ : ℝ, x₀ > 0 ∧ x₀ * Real.exp x₀ ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1983_198393


namespace NUMINAMATH_CALUDE_chocolate_milk_probability_l1983_198396

theorem chocolate_milk_probability : 
  let n : ℕ := 7  -- total number of days
  let k : ℕ := 5  -- number of days with chocolate milk
  let p : ℚ := 3/4  -- probability of bottling chocolate milk on any given day
  Nat.choose n k * p^k * (1-p)^(n-k) = 5103/16384 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_milk_probability_l1983_198396


namespace NUMINAMATH_CALUDE_quadratic_minimum_quadratic_minimum_attained_l1983_198301

theorem quadratic_minimum (x : ℝ) : 2 * x^2 + 16 * x + 40 ≥ 8 := by sorry

theorem quadratic_minimum_attained : ∃ x : ℝ, 2 * x^2 + 16 * x + 40 = 8 := by sorry

end NUMINAMATH_CALUDE_quadratic_minimum_quadratic_minimum_attained_l1983_198301


namespace NUMINAMATH_CALUDE_lcm_inequality_l1983_198347

theorem lcm_inequality (n : ℕ) (k : ℕ) (a : Fin k → ℕ) 
  (h1 : ∀ i : Fin k, n ≥ a i)
  (h2 : ∀ i j : Fin k, i < j → a i > a j)
  (h3 : ∀ i j : Fin k, Nat.lcm (a i) (a j) ≤ n) :
  ∀ i : Fin k, (i.val + 1) * a i ≤ n := by
  sorry

end NUMINAMATH_CALUDE_lcm_inequality_l1983_198347


namespace NUMINAMATH_CALUDE_fourth_selection_is_65_l1983_198303

/-- Systematic sampling function -/
def systematicSample (totalParts : ℕ) (sampleSize : ℕ) (firstSelection : ℕ) (selectionNumber : ℕ) : ℕ :=
  let samplingInterval := totalParts / sampleSize
  firstSelection + (selectionNumber - 1) * samplingInterval

/-- Theorem: In the given systematic sampling scenario, the fourth selection is part number 65 -/
theorem fourth_selection_is_65 :
  let totalParts := 200
  let sampleSize := 10
  let firstSelection := 5
  let fourthSelection := 4
  systematicSample totalParts sampleSize firstSelection fourthSelection = 65 := by
  sorry

#eval systematicSample 200 10 5 4  -- Should output 65

end NUMINAMATH_CALUDE_fourth_selection_is_65_l1983_198303


namespace NUMINAMATH_CALUDE_product_equality_l1983_198345

theorem product_equality : 1500 * 451 * 0.0451 * 25 = 7627537500 := by
  sorry

end NUMINAMATH_CALUDE_product_equality_l1983_198345


namespace NUMINAMATH_CALUDE_problem_polygon_area_l1983_198365

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Calculates the area of a rectangle -/
def rectangleArea (r : Rectangle) : ℕ := r.width * r.height

/-- Represents a polygon composed of rectangles -/
structure Polygon where
  rectangles : List Rectangle

/-- Calculates the total area of a polygon -/
def polygonArea (p : Polygon) : ℕ :=
  p.rectangles.map rectangleArea |>.sum

/-- The polygon in the problem -/
def problemPolygon : Polygon :=
  { rectangles := [
      { width := 2, height := 2 },  -- 2x2 square
      { width := 1, height := 2 },  -- 1x2 rectangle
      { width := 1, height := 2 }   -- 1x2 rectangle
    ] 
  }

theorem problem_polygon_area : polygonArea problemPolygon = 8 := by
  sorry

end NUMINAMATH_CALUDE_problem_polygon_area_l1983_198365


namespace NUMINAMATH_CALUDE_divisible_by_seven_last_digits_l1983_198316

theorem divisible_by_seven_last_digits :
  ∃ (S : Finset Nat), (∀ n : Nat, n % 10 ∈ S ↔ ∃ m : Nat, m % 7 = 0 ∧ m % 10 = n % 10) ∧ Finset.card S = 2 :=
by sorry

end NUMINAMATH_CALUDE_divisible_by_seven_last_digits_l1983_198316


namespace NUMINAMATH_CALUDE_monkey_ladder_min_steps_l1983_198338

/-- The minimum number of steps for the monkey's ladder. -/
def min_steps : ℕ := 26

/-- Represents the possible movements of the monkey. -/
inductive Movement
| up : Movement
| down : Movement

/-- The number of steps the monkey moves in each direction. -/
def step_count (m : Movement) : ℤ :=
  match m with
  | Movement.up => 18
  | Movement.down => -10

/-- A sequence of movements that allows the monkey to reach the top and return to the ground. -/
def valid_sequence : List Movement := 
  [Movement.up, Movement.down, Movement.up, Movement.down, Movement.down, Movement.up, 
   Movement.down, Movement.down, Movement.up, Movement.down, Movement.down, Movement.up, 
   Movement.down, Movement.down]

theorem monkey_ladder_min_steps :
  (∀ (seq : List Movement), 
    (seq.foldl (λ acc m => acc + step_count m) 0 = 0) →
    (seq.foldl (λ acc m => max acc (acc + step_count m)) 0 ≥ min_steps)) ∧
  (valid_sequence.foldl (λ acc m => acc + step_count m) 0 = 0) ∧
  (valid_sequence.foldl (λ acc m => max acc (acc + step_count m)) 0 = min_steps) := by
  sorry

#check monkey_ladder_min_steps

end NUMINAMATH_CALUDE_monkey_ladder_min_steps_l1983_198338


namespace NUMINAMATH_CALUDE_peggy_stickers_count_l1983_198395

/-- The number of folders Peggy buys -/
def num_folders : Nat := 3

/-- The number of sheets in each folder -/
def sheets_per_folder : Nat := 10

/-- The number of stickers on each sheet in the red folder -/
def red_stickers_per_sheet : Nat := 3

/-- The number of stickers on each sheet in the green folder -/
def green_stickers_per_sheet : Nat := 2

/-- The number of stickers on each sheet in the blue folder -/
def blue_stickers_per_sheet : Nat := 1

/-- The total number of stickers Peggy uses -/
def total_stickers : Nat := 
  sheets_per_folder * red_stickers_per_sheet +
  sheets_per_folder * green_stickers_per_sheet +
  sheets_per_folder * blue_stickers_per_sheet

theorem peggy_stickers_count : total_stickers = 60 := by
  sorry

end NUMINAMATH_CALUDE_peggy_stickers_count_l1983_198395


namespace NUMINAMATH_CALUDE_value_of_a_l1983_198359

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {1, 3, a}
def B (a : ℝ) : Set ℝ := {1, a^2 - a + 1}

-- State the theorem
theorem value_of_a : 
  ∀ a : ℝ, (A a ⊇ B a) → (a = -1 ∨ a = 2) := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l1983_198359


namespace NUMINAMATH_CALUDE_product_of_roots_plus_two_l1983_198313

theorem product_of_roots_plus_two (a b c : ℝ) : 
  a^3 - 15*a^2 + 25*a - 10 = 0 →
  b^3 - 15*b^2 + 25*b - 10 = 0 →
  c^3 - 15*c^2 + 25*c - 10 = 0 →
  (2+a)*(2+b)*(2+c) = 128 := by
sorry

end NUMINAMATH_CALUDE_product_of_roots_plus_two_l1983_198313


namespace NUMINAMATH_CALUDE_consignment_total_items_l1983_198304

/-- Represents the price and quantity of items in a consignment shop. -/
structure ConsignmentItems where
  camera_price : ℕ
  clock_price : ℕ
  pen_price : ℕ
  receiver_price : ℕ
  camera_quantity : ℕ

/-- Conditions for the consignment shop problem -/
def ConsignmentConditions (items : ConsignmentItems) : Prop :=
  -- Total value of all items is 240 rubles
  (3 * items.camera_quantity * items.pen_price + 
   items.camera_quantity * items.clock_price + 
   items.camera_quantity * items.receiver_price + 
   items.camera_quantity * items.camera_price = 240) ∧
  -- Sum of receiver and clock prices is 4 rubles more than sum of camera and pen prices
  (items.receiver_price + items.clock_price = items.camera_price + items.pen_price + 4) ∧
  -- Sum of clock and pen prices is 24 rubles less than sum of camera and receiver prices
  (items.clock_price + items.pen_price + 24 = items.camera_price + items.receiver_price) ∧
  -- Pen price is an integer not exceeding 6 rubles
  (items.pen_price ≤ 6) ∧
  -- Number of cameras equals camera price divided by 10
  (items.camera_quantity = items.camera_price / 10) ∧
  -- Number of clocks equals number of receivers and number of cameras
  (items.camera_quantity = items.camera_quantity) ∧
  -- Number of pens is three times the number of cameras
  (3 * items.camera_quantity = 3 * items.camera_quantity)

/-- The theorem stating that under the given conditions, the total number of items is 18 -/
theorem consignment_total_items (items : ConsignmentItems) 
  (h : ConsignmentConditions items) : 
  (6 * items.camera_quantity = 18) := by
  sorry


end NUMINAMATH_CALUDE_consignment_total_items_l1983_198304


namespace NUMINAMATH_CALUDE_sixteenth_row_seats_l1983_198343

/-- 
Represents the number of seats in a row of an auditorium where:
- The first row has 5 seats
- Each subsequent row increases by 2 seats
-/
def seats_in_row (n : ℕ) : ℕ := 2 * n + 3

/-- 
Theorem: The 16th row of the auditorium has 35 seats
-/
theorem sixteenth_row_seats : seats_in_row 16 = 35 := by
  sorry

end NUMINAMATH_CALUDE_sixteenth_row_seats_l1983_198343


namespace NUMINAMATH_CALUDE_line_passes_through_P_and_parallel_to_tangent_l1983_198351

-- Define the curve
def f (x : ℝ) : ℝ := 3 * x^2 - 4 * x + 2

-- Define the point P
def P : ℝ × ℝ := (-1, 2)

-- Define the point M
def M : ℝ × ℝ := (1, 1)

-- Define the slope of the tangent line at M
def m : ℝ := (6 : ℝ) * M.1 - 4

-- Define the equation of the line
def line_equation (x y : ℝ) : Prop := 2 * x - y + 4 = 0

-- Theorem statement
theorem line_passes_through_P_and_parallel_to_tangent :
  line_equation P.1 P.2 ∧
  (∀ x y : ℝ, line_equation x y → (y - P.2) = m * (x - P.1)) :=
sorry

end NUMINAMATH_CALUDE_line_passes_through_P_and_parallel_to_tangent_l1983_198351


namespace NUMINAMATH_CALUDE_equal_numbers_iff_odd_l1983_198383

/-- Represents a square table of numbers -/
def Table (n : ℕ) := Fin n → Fin n → ℕ

/-- Initial state of the table with ones on the diagonal and zeros elsewhere -/
def initialTable (n : ℕ) : Table n :=
  λ i j => if i = j then 1 else 0

/-- Represents a closed path of a rook on the table -/
def RookPath (n : ℕ) := List (Fin n × Fin n)

/-- Checks if a path is valid (closed and non-self-intersecting) -/
def isValidPath (n : ℕ) (path : RookPath n) : Prop := sorry

/-- Applies the transformation along a given path -/
def applyTransformation (n : ℕ) (table : Table n) (path : RookPath n) : Table n := sorry

/-- Checks if all numbers in the table are equal -/
def allEqual (n : ℕ) (table : Table n) : Prop := sorry

/-- Main theorem: It's possible to make all numbers equal if and only if n is odd -/
theorem equal_numbers_iff_odd (n : ℕ) :
  (∃ (transformations : List (RookPath n)), 
    (∀ path ∈ transformations, isValidPath n path) ∧ 
    allEqual n (transformations.foldl (applyTransformation n) (initialTable n))) 
  ↔ n % 2 = 1 := by sorry

end NUMINAMATH_CALUDE_equal_numbers_iff_odd_l1983_198383


namespace NUMINAMATH_CALUDE_pyramid_volume_l1983_198349

/-- The volume of a pyramid with a right triangular base of side length 2 and height 2 is 4/3 -/
theorem pyramid_volume (s h : ℝ) (hs : s = 2) (hh : h = 2) :
  (1 / 3 : ℝ) * (1 / 2 * s * s) * h = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_volume_l1983_198349


namespace NUMINAMATH_CALUDE_system_solution_l1983_198306

theorem system_solution (x y z : ℝ) : 
  x = 1 ∧ y = -1 ∧ z = -2 →
  (2 * x + y + z = -1) ∧
  (3 * y - z = -1) ∧
  (3 * x + 2 * y + 3 * z = -5) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1983_198306


namespace NUMINAMATH_CALUDE_expand_expression_l1983_198328

theorem expand_expression (x y : ℝ) : (x + 15) * (3 * y + 20) = 3 * x * y + 20 * x + 45 * y + 300 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l1983_198328


namespace NUMINAMATH_CALUDE_conference_session_duration_l1983_198368

/-- Given a conference duration and break time, calculate the session time in minutes. -/
def conference_session_time (hours minutes break_time : ℕ) : ℕ :=
  hours * 60 + minutes - break_time

/-- Theorem: A conference lasting 8 hours and 45 minutes with a 30-minute break has a session time of 495 minutes. -/
theorem conference_session_duration :
  conference_session_time 8 45 30 = 495 :=
by sorry

end NUMINAMATH_CALUDE_conference_session_duration_l1983_198368


namespace NUMINAMATH_CALUDE_zero_sufficient_for_perpendicular_zero_not_necessary_for_perpendicular_zero_sufficient_not_necessary_for_perpendicular_l1983_198385

/-- Line l1 with equation x + ay - a = 0 -/
def line1 (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + a * p.2 - a = 0}

/-- Line l2 with equation ax - (2a - 3)y - 1 = 0 -/
def line2 (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | a * p.1 - (2 * a - 3) * p.2 - 1 = 0}

/-- Two lines are perpendicular -/
def perpendicular (l1 l2 : Set (ℝ × ℝ)) : Prop :=
  ∃ (m1 m2 : ℝ), (∀ (p q : ℝ × ℝ), p ∈ l1 → q ∈ l1 → p ≠ q → (q.2 - p.2) = m1 * (q.1 - p.1)) ∧
                 (∀ (p q : ℝ × ℝ), p ∈ l2 → q ∈ l2 → p ≠ q → (q.2 - p.2) = m2 * (q.1 - p.1)) ∧
                 m1 * m2 = -1

/-- a=0 is a sufficient condition for perpendicularity -/
theorem zero_sufficient_for_perpendicular :
  perpendicular (line1 0) (line2 0) :=
sorry

/-- a=0 is not a necessary condition for perpendicularity -/
theorem zero_not_necessary_for_perpendicular :
  ∃ a : ℝ, a ≠ 0 ∧ perpendicular (line1 a) (line2 a) :=
sorry

/-- Main theorem: a=0 is sufficient but not necessary for perpendicularity -/
theorem zero_sufficient_not_necessary_for_perpendicular :
  (perpendicular (line1 0) (line2 0)) ∧
  (∃ a : ℝ, a ≠ 0 ∧ perpendicular (line1 a) (line2 a)) :=
sorry

end NUMINAMATH_CALUDE_zero_sufficient_for_perpendicular_zero_not_necessary_for_perpendicular_zero_sufficient_not_necessary_for_perpendicular_l1983_198385


namespace NUMINAMATH_CALUDE_rectangle_area_l1983_198392

theorem rectangle_area (x y : ℝ) 
  (h1 : (x + 3) * (y - 1) = x * y)
  (h2 : (x - 3) * (y + 2) = x * y)
  (h3 : (x + 4) * (y - 2) = x * y) :
  x * y = 36 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l1983_198392


namespace NUMINAMATH_CALUDE_rental_cost_equality_l1983_198397

/-- The daily rate for Safety Rent-a-Car in dollars -/
def safety_daily_rate : ℝ := 21.95

/-- The per-mile rate for Safety Rent-a-Car in dollars -/
def safety_mile_rate : ℝ := 0.19

/-- The daily rate for City Rentals in dollars -/
def city_daily_rate : ℝ := 18.95

/-- The per-mile rate for City Rentals in dollars -/
def city_mile_rate : ℝ := 0.21

/-- The mileage at which the cost is the same for both companies -/
def equal_cost_mileage : ℝ := 150

theorem rental_cost_equality :
  safety_daily_rate + safety_mile_rate * equal_cost_mileage =
  city_daily_rate + city_mile_rate * equal_cost_mileage := by
  sorry

#check rental_cost_equality

end NUMINAMATH_CALUDE_rental_cost_equality_l1983_198397


namespace NUMINAMATH_CALUDE_parabola_rectangle_problem_l1983_198382

/-- The parabola equation -/
def parabola_equation (k x y : ℝ) : Prop := y = k^2 - x^2

/-- Rectangle ABCD properties -/
structure Rectangle (k : ℝ) where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  parallel_to_axes : Prop
  A_on_x_axis : A.2 = 0
  D_on_x_axis : D.2 = 0
  V_midpoint_BC : (B.1 + C.1) / 2 = 0 ∧ (B.2 + C.2) / 2 = k^2

/-- Perimeter of the rectangle -/
def perimeter (rect : Rectangle k) : ℝ :=
  2 * (|rect.A.1 - rect.B.1| + |rect.A.2 - rect.B.2|)

/-- Main theorem -/
theorem parabola_rectangle_problem (k : ℝ) 
  (h_pos : k > 0)
  (rect : Rectangle k)
  (h_perimeter : perimeter rect = 48) :
  k = 4 := by sorry

end NUMINAMATH_CALUDE_parabola_rectangle_problem_l1983_198382


namespace NUMINAMATH_CALUDE_abc_fraction_equals_twelve_l1983_198344

theorem abc_fraction_equals_twelve
  (a b c m : ℝ)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hc : c ≠ 0)
  (hsum : a + b + c = m)
  (hsquare_sum : a^2 + b^2 + c^2 = m^2 / 2) :
  (a * (m - 2*a)^2 + b * (m - 2*b)^2 + c * (m - 2*c)^2) / (a * b * c) = 12 :=
by sorry

end NUMINAMATH_CALUDE_abc_fraction_equals_twelve_l1983_198344


namespace NUMINAMATH_CALUDE_revenue_maximizing_price_l1983_198364

/-- Revenue function for toy sales -/
def R (p : ℝ) : ℝ := p * (150 - 4 * p)

/-- The price that maximizes revenue is 18.75 -/
theorem revenue_maximizing_price :
  ∃ (p : ℝ), p ≤ 30 ∧ 
  (∀ (q : ℝ), q ≤ 30 → R q ≤ R p) ∧ 
  p = 18.75 := by
sorry

end NUMINAMATH_CALUDE_revenue_maximizing_price_l1983_198364


namespace NUMINAMATH_CALUDE_min_value_product_l1983_198322

theorem min_value_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : x/y + y/z + z/x + y/x + z/y + x/z = 8) :
  (x/y + y/z + z/x) * (y/x + z/y + x/z) ≥ 22 * Real.sqrt 11 - 57 :=
by sorry

end NUMINAMATH_CALUDE_min_value_product_l1983_198322


namespace NUMINAMATH_CALUDE_more_boys_probability_l1983_198319

-- Define the possible number of children
inductive ChildCount : Type
  | zero : ChildCount
  | one : ChildCount
  | two : ChildCount
  | three : ChildCount

-- Define the probability distribution for the number of children
def childCountProb : ChildCount → ℚ
  | ChildCount.zero => 1/15
  | ChildCount.one => 6/15
  | ChildCount.two => 6/15
  | ChildCount.three => 2/15

-- Define the probability of a child being a boy
def boyProb : ℚ := 1/2

-- Define the event of having more boys than girls
def moreBoysEvent : ChildCount → ℚ
  | ChildCount.zero => 0
  | ChildCount.one => 1/2
  | ChildCount.two => 1/4
  | ChildCount.three => 1/2

-- State the theorem
theorem more_boys_probability :
  (moreBoysEvent ChildCount.zero * childCountProb ChildCount.zero +
   moreBoysEvent ChildCount.one * childCountProb ChildCount.one +
   moreBoysEvent ChildCount.two * childCountProb ChildCount.two +
   moreBoysEvent ChildCount.three * childCountProb ChildCount.three) = 11/30 := by
  sorry

end NUMINAMATH_CALUDE_more_boys_probability_l1983_198319


namespace NUMINAMATH_CALUDE_dodge_trucks_count_l1983_198375

theorem dodge_trucks_count (ford dodge toyota vw : ℕ) 
  (h1 : ford = dodge / 3)
  (h2 : ford = 2 * toyota)
  (h3 : vw = toyota / 2)
  (h4 : vw = 5) : 
  dodge = 60 := by
  sorry

end NUMINAMATH_CALUDE_dodge_trucks_count_l1983_198375


namespace NUMINAMATH_CALUDE_oliver_candy_theorem_l1983_198321

/-- Oliver's Halloween candy problem -/
theorem oliver_candy_theorem (initial_candy : ℕ) (candy_given : ℕ) (remaining_candy : ℕ) :
  initial_candy = 78 →
  candy_given = 10 →
  remaining_candy = initial_candy - candy_given →
  remaining_candy = 68 :=
by
  sorry

end NUMINAMATH_CALUDE_oliver_candy_theorem_l1983_198321


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l1983_198372

theorem pure_imaginary_condition (a : ℝ) : 
  (∃ (b : ℝ), a^2 - 1 + (a + 1) * Complex.I = Complex.I * b) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l1983_198372


namespace NUMINAMATH_CALUDE_exists_grid_with_more_than_20_components_l1983_198388

/-- Represents a diagonal in a cell --/
inductive Diagonal
| TopLeft
| TopRight

/-- Represents the grid --/
def Grid := Matrix (Fin 8) (Fin 8) Diagonal

/-- A function that counts the number of connected components in a grid --/
def countComponents (g : Grid) : ℕ := sorry

/-- Theorem stating that there exists a grid configuration with more than 20 components --/
theorem exists_grid_with_more_than_20_components :
  ∃ (g : Grid), countComponents g > 20 :=
sorry

end NUMINAMATH_CALUDE_exists_grid_with_more_than_20_components_l1983_198388


namespace NUMINAMATH_CALUDE_problem_solution_l1983_198353

noncomputable def x : ℝ := Real.sqrt (19 - 8 * Real.sqrt 3)

theorem problem_solution : 
  (x^4 - 6*x^3 - 2*x^2 + 18*x + 23) / (x^2 - 8*x + 15) = 5 :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1983_198353


namespace NUMINAMATH_CALUDE_quadratic_properties_l1983_198386

/-- A quadratic equation with roots 1 and -1 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0
  root_one : a + b + c = 0
  root_neg_one : a - b + c = 0

theorem quadratic_properties (eq : QuadraticEquation) :
  eq.a + eq.b + eq.c = 0 ∧ eq.b = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_properties_l1983_198386


namespace NUMINAMATH_CALUDE_cubic_function_extrema_l1983_198308

/-- A cubic function with parameters a and b -/
def f (a b x : ℝ) : ℝ := a * x^3 + b * x^2 + x

/-- The derivative of f with respect to x -/
def f' (a b x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x + 1

theorem cubic_function_extrema (a b : ℝ) :
  (f' a b 1 = 0 ∧ f' a b 2 = 0) →
  (a = 1/6 ∧ b = -3/4) ∧
  (IsLocalMax (f a b) 1 ∧ IsLocalMin (f a b) 2) :=
sorry

end NUMINAMATH_CALUDE_cubic_function_extrema_l1983_198308


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_exponential_inequality_l1983_198371

theorem negation_of_existence (f : ℝ → ℝ) :
  (¬ ∃ x : ℝ, f x < 0) ↔ (∀ x : ℝ, f x ≥ 0) :=
by sorry

theorem negation_of_exponential_inequality :
  (¬ ∃ x : ℝ, Real.exp x - x - 1 < 0) ↔ (∀ x : ℝ, Real.exp x - x - 1 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_exponential_inequality_l1983_198371


namespace NUMINAMATH_CALUDE_rectangular_solid_surface_area_l1983_198380

/-- A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself. -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- The volume of a rectangular solid is the product of its length, width, and height. -/
def volume (l w h : ℕ) : ℕ := l * w * h

/-- The surface area of a rectangular solid is given by 2(lw + wh + hl). -/
def surfaceArea (l w h : ℕ) : ℕ := 2 * (l * w + w * h + h * l)

theorem rectangular_solid_surface_area 
  (l w h : ℕ) 
  (hl : isPrime l) 
  (hw : isPrime w) 
  (hh : isPrime h) 
  (hv : volume l w h = 437) : 
  surfaceArea l w h = 958 := by
  sorry

#check rectangular_solid_surface_area

end NUMINAMATH_CALUDE_rectangular_solid_surface_area_l1983_198380


namespace NUMINAMATH_CALUDE_horner_method_evaluation_l1983_198367

def horner_polynomial (x : ℝ) : ℝ := (((((3 * x - 4) * x + 6) * x - 2) * x - 5) * x - 2)

theorem horner_method_evaluation :
  horner_polynomial 5 = 7548 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_evaluation_l1983_198367


namespace NUMINAMATH_CALUDE_thumbtacks_total_l1983_198330

/-- Given 3 cans of thumbtacks, where 120 thumbtacks are used from each can
    and 30 thumbtacks remain in each can after use, prove that the total
    number of thumbtacks in the three full cans initially was 450. -/
theorem thumbtacks_total (cans : Nat) (used_per_can : Nat) (remaining_per_can : Nat)
    (h1 : cans = 3)
    (h2 : used_per_can = 120)
    (h3 : remaining_per_can = 30) :
    cans * (used_per_can + remaining_per_can) = 450 := by
  sorry

end NUMINAMATH_CALUDE_thumbtacks_total_l1983_198330


namespace NUMINAMATH_CALUDE_prohor_receives_all_money_l1983_198317

/-- Represents a person with their initial number of flatbreads -/
structure Person where
  name : String
  flatbreads : ℕ

/-- Represents the situation with the woodcutters and hunter -/
structure WoodcutterSituation where
  ivan : Person
  prohor : Person
  hunter : Person
  total_flatbreads : ℕ
  total_people : ℕ
  hunter_payment : ℕ

/-- Calculates the fair compensation for a person based on shared flatbreads -/
def fair_compensation (situation : WoodcutterSituation) (person : Person) : ℕ :=
  let shared_flatbreads := person.flatbreads - (situation.total_flatbreads / situation.total_people)
  shared_flatbreads * (situation.hunter_payment / situation.total_flatbreads)

/-- Theorem stating that Prohor should receive all the money -/
theorem prohor_receives_all_money (situation : WoodcutterSituation) : 
  situation.ivan.flatbreads = 4 →
  situation.prohor.flatbreads = 8 →
  situation.total_flatbreads = 12 →
  situation.total_people = 3 →
  situation.hunter_payment = 60 →
  fair_compensation situation situation.prohor = situation.hunter_payment :=
sorry

end NUMINAMATH_CALUDE_prohor_receives_all_money_l1983_198317


namespace NUMINAMATH_CALUDE_min_disks_required_l1983_198335

def total_files : ℕ := 40
def disk_capacity : ℚ := 2

def file_sizes : List ℚ := List.replicate 8 0.9 ++ List.replicate 20 0.6 ++ List.replicate 12 0.5

def is_valid_disk_assignment (assignment : List (List ℚ)) : Prop :=
  assignment.all (λ disk => disk.sum ≤ disk_capacity) ∧
  assignment.join.length = total_files ∧
  assignment.join.toFinset = file_sizes.toFinset

theorem min_disks_required :
  ∃ (assignment : List (List ℚ)),
    is_valid_disk_assignment assignment ∧
    assignment.length = 15 ∧
    ∀ (other_assignment : List (List ℚ)),
      is_valid_disk_assignment other_assignment →
      other_assignment.length ≥ 15 :=
by sorry

end NUMINAMATH_CALUDE_min_disks_required_l1983_198335


namespace NUMINAMATH_CALUDE_pool_capacity_l1983_198339

/-- Represents the capacity of a pool and properties of a pump -/
structure Pool :=
  (capacity : ℝ)
  (pumpRate : ℝ)
  (pumpTime : ℝ)
  (remainingWater : ℝ)

/-- Theorem stating the capacity of the pool given the conditions -/
theorem pool_capacity 
  (p : Pool)
  (h1 : p.pumpRate = 2/3)
  (h2 : p.pumpTime = 7.5)
  (h3 : p.pumpTime * 8 = 0.15 * 60)
  (h4 : p.remainingWater = 25)
  (h5 : p.capacity * (1 - p.pumpRate * (0.15 * 60 / p.pumpTime)) = p.remainingWater) :
  p.capacity = 125 := by
  sorry

end NUMINAMATH_CALUDE_pool_capacity_l1983_198339


namespace NUMINAMATH_CALUDE_smallest_delicious_integer_l1983_198331

/-- An integer is delicious if there exist several consecutive integers, starting from it, that add up to 2020. -/
def Delicious (n : ℤ) : Prop :=
  ∃ k : ℕ+, (Finset.range k).sum (fun i => n + i) = 2020

/-- The smallest delicious integer less than -2020 is -2021. -/
theorem smallest_delicious_integer :
  (∀ n < -2020, Delicious n → n ≥ -2021) ∧ Delicious (-2021) :=
sorry

end NUMINAMATH_CALUDE_smallest_delicious_integer_l1983_198331


namespace NUMINAMATH_CALUDE_problem_statement_l1983_198358

theorem problem_statement (x : ℝ) (h : x = 4) : 5 * x + 7 = 27 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1983_198358


namespace NUMINAMATH_CALUDE_jessie_weight_loss_l1983_198355

/-- Jessie's weight loss calculation -/
theorem jessie_weight_loss 
  (weight_before : ℝ) 
  (weight_after : ℝ) 
  (h1 : weight_before = 192) 
  (h2 : weight_after = 66) : 
  weight_before - weight_after = 126 := by
  sorry

end NUMINAMATH_CALUDE_jessie_weight_loss_l1983_198355


namespace NUMINAMATH_CALUDE_opposite_absolute_values_l1983_198379

theorem opposite_absolute_values (x y : ℝ) : 
  (|x - y + 9| + |2*x + y| = 0) → (x = -3 ∧ y = 6) := by
sorry

end NUMINAMATH_CALUDE_opposite_absolute_values_l1983_198379


namespace NUMINAMATH_CALUDE_inequality_proof_l1983_198307

theorem inequality_proof (x : ℝ) : (1 : ℝ) / (x^2 + 1) > (1 : ℝ) / (x^2 + 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1983_198307


namespace NUMINAMATH_CALUDE_integer_sum_problem_l1983_198394

theorem integer_sum_problem : 
  ∃ (a b : ℕ+), 
    (a.val * b.val + a.val + b.val = 143) ∧ 
    (Nat.gcd a.val b.val = 1) ∧ 
    (a.val < 30 ∧ b.val < 30) ∧ 
    (a.val + b.val = 23 ∨ a.val + b.val = 24 ∨ a.val + b.val = 28) := by
  sorry

end NUMINAMATH_CALUDE_integer_sum_problem_l1983_198394


namespace NUMINAMATH_CALUDE_medical_team_selection_l1983_198312

theorem medical_team_selection (orthopedic neurosurgeons internists : ℕ) 
  (h1 : orthopedic = 3) 
  (h2 : neurosurgeons = 4) 
  (h3 : internists = 5) 
  (team_size : ℕ) 
  (h4 : team_size = 5) : 
  (Nat.choose (orthopedic + neurosurgeons + internists) team_size) -
  ((Nat.choose (neurosurgeons + internists) team_size - 1) +
   (Nat.choose (orthopedic + internists) team_size - 1) +
   (Nat.choose (orthopedic + neurosurgeons) team_size) +
   1) = 590 := by
  sorry

end NUMINAMATH_CALUDE_medical_team_selection_l1983_198312


namespace NUMINAMATH_CALUDE_trigonometric_identity_l1983_198329

theorem trigonometric_identity (α β γ : ℝ) : 
  (Real.sin α + Real.sin β + Real.sin γ - Real.sin (α + β + γ)) / 
  (Real.cos α + Real.cos β + Real.cos γ + Real.cos (α + β + γ)) = 
  Real.tan ((α + β) / 2) * Real.tan ((β + γ) / 2) * Real.tan ((γ + α) / 2) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l1983_198329


namespace NUMINAMATH_CALUDE_percentage_relationship_l1983_198376

theorem percentage_relationship (x y : ℝ) : 
  Real.sqrt (0.3 * (x - y)) = Real.sqrt (0.2 * (x + y)) → y = 0.2 * x := by
  sorry

end NUMINAMATH_CALUDE_percentage_relationship_l1983_198376


namespace NUMINAMATH_CALUDE_maria_coffee_shop_visits_l1983_198309

/-- 
Given that Maria orders 3 cups of coffee each time she goes to the coffee shop
and orders 6 cups of coffee per day, prove that she goes to the coffee shop 2 times per day.
-/
theorem maria_coffee_shop_visits 
  (cups_per_visit : ℕ) 
  (cups_per_day : ℕ) 
  (h1 : cups_per_visit = 3)
  (h2 : cups_per_day = 6) :
  cups_per_day / cups_per_visit = 2 := by
  sorry

end NUMINAMATH_CALUDE_maria_coffee_shop_visits_l1983_198309


namespace NUMINAMATH_CALUDE_johns_friends_count_l1983_198363

def total_cost : ℕ := 12100
def cost_per_person : ℕ := 1100

theorem johns_friends_count : 
  (total_cost / cost_per_person) - 1 = 10 := by sorry

end NUMINAMATH_CALUDE_johns_friends_count_l1983_198363


namespace NUMINAMATH_CALUDE_max_value_of_expression_l1983_198336

theorem max_value_of_expression (x y z : ℝ) : 
  x ≥ 0 → y ≥ 0 → z ≥ 0 → x + y + z = 1 → 
  2*x*y + y*z + 2*z*x ≤ 4/7 ∧ 
  ∃ x' y' z' : ℝ, x' ≥ 0 ∧ y' ≥ 0 ∧ z' ≥ 0 ∧ x' + y' + z' = 1 ∧ 2*x'*y' + y'*z' + 2*z'*x' = 4/7 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l1983_198336


namespace NUMINAMATH_CALUDE_perpendicular_lines_b_value_l1983_198342

-- Define the slopes of the two lines
def slope1 : ℚ := -2/3
def slope2 (b : ℚ) : ℚ := -b/3

-- Define the perpendicularity condition
def perpendicular (b : ℚ) : Prop := slope1 * slope2 b = -1

-- Theorem statement
theorem perpendicular_lines_b_value :
  ∃ b : ℚ, perpendicular b ∧ b = -9/2 := by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_b_value_l1983_198342


namespace NUMINAMATH_CALUDE_vector_inequality_l1983_198300

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem vector_inequality (a b c : V) :
  ‖a‖ + ‖b‖ + ‖c‖ + ‖a + b + c‖ ≥ ‖a + b‖ + ‖b + c‖ + ‖c + a‖ := by
  sorry

end NUMINAMATH_CALUDE_vector_inequality_l1983_198300


namespace NUMINAMATH_CALUDE_max_profit_selling_price_daily_profit_unachievable_monthly_profit_prices_l1983_198324

/-- Represents the profit function for desk lamp sales -/
def profit_function (x : ℝ) : ℝ :=
  (x - 30) * (600 - 10 * (x - 40))

/-- Theorem stating the maximum profit and corresponding selling price -/
theorem max_profit_selling_price :
  ∃ (max_price max_profit : ℝ),
    max_price = 65 ∧
    max_profit = 12250 ∧
    ∀ (x : ℝ), profit_function x ≤ max_profit :=
by
  sorry

/-- Theorem stating that 15,000 yuan daily profit is not achievable -/
theorem daily_profit_unachievable :
  ∀ (x : ℝ), profit_function x < 15000 :=
by
  sorry

/-- Theorem stating the selling prices for 10,000 yuan monthly profit -/
theorem monthly_profit_prices :
  ∃ (price1 price2 : ℝ),
    price1 = 80 ∧
    price2 = 50 ∧
    profit_function price1 = 10000 ∧
    profit_function price2 = 10000 ∧
    ∀ (x : ℝ), profit_function x = 10000 → (x = price1 ∨ x = price2) :=
by
  sorry

end NUMINAMATH_CALUDE_max_profit_selling_price_daily_profit_unachievable_monthly_profit_prices_l1983_198324


namespace NUMINAMATH_CALUDE_four_percent_of_fifty_l1983_198340

theorem four_percent_of_fifty : ∃ x : ℝ, x = 50 * (4 / 100) ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_four_percent_of_fifty_l1983_198340


namespace NUMINAMATH_CALUDE_max_fraction_sum_l1983_198325

/-- Represents a digit from 0 to 9 -/
def Digit := Fin 10

/-- Checks if four digits are distinct -/
def distinct (a b c d : Digit) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

/-- The fraction (A+B)/(C+D) is an integer -/
def is_integer_fraction (a b c d : Digit) : Prop :=
  ∃ k : ℕ, k * (c.val + d.val) = a.val + b.val

/-- The fraction (A+B)/(C+D) is maximized -/
def is_maximized (a b c d : Digit) : Prop :=
  ∀ w x y z : Digit, distinct w x y z →
    is_integer_fraction w x y z →
    (a.val + b.val : ℚ) / (c.val + d.val) ≥ (w.val + x.val : ℚ) / (y.val + z.val)

theorem max_fraction_sum (a b c d : Digit) :
  distinct a b c d →
  is_integer_fraction a b c d →
  is_maximized a b c d →
  a.val + b.val = 17 :=
sorry

end NUMINAMATH_CALUDE_max_fraction_sum_l1983_198325


namespace NUMINAMATH_CALUDE_logans_score_l1983_198391

theorem logans_score (total_students : ℕ) (average_without_logan : ℚ) (average_with_logan : ℚ) :
  total_students = 20 →
  average_without_logan = 85 →
  average_with_logan = 86 →
  (total_students * average_with_logan - (total_students - 1) * average_without_logan : ℚ) = 105 :=
by sorry

end NUMINAMATH_CALUDE_logans_score_l1983_198391


namespace NUMINAMATH_CALUDE_function_value_at_two_l1983_198377

/-- Given a function f(x) = ax^5 + bx^3 - x + 2 where a and b are constants,
    and f(-2) = 5, prove that f(2) = -1 -/
theorem function_value_at_two
  (a b : ℝ)
  (f : ℝ → ℝ)
  (h1 : ∀ x, f x = a * x^5 + b * x^3 - x + 2)
  (h2 : f (-2) = 5) :
  f 2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_two_l1983_198377


namespace NUMINAMATH_CALUDE_gcd_of_seven_digit_set_l1983_198374

/-- A function that generates a seven-digit number from a three-digit number -/
def seven_digit_from_three (n : ℕ) : ℕ := 1001 * n

/-- The set of all seven-digit numbers formed by repeating three-digit numbers -/
def seven_digit_set : Set ℕ := {m | ∃ n, 100 ≤ n ∧ n < 1000 ∧ m = seven_digit_from_three n}

/-- The theorem stating that 1001 is the greatest common divisor of all numbers in the set -/
theorem gcd_of_seven_digit_set :
  ∃ d, d > 0 ∧ (∀ m ∈ seven_digit_set, d ∣ m) ∧
  (∀ d' > 0, (∀ m ∈ seven_digit_set, d' ∣ m) → d' ≤ d) ∧
  d = 1001 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_seven_digit_set_l1983_198374


namespace NUMINAMATH_CALUDE_quadratic_root_range_l1983_198387

/-- Represents a quadratic equation ax^2 + (a+2)x + 9a = 0 -/
def quadratic_equation (a x : ℝ) : ℝ := a * x^2 + (a + 2) * x + 9 * a

theorem quadratic_root_range :
  ∀ a : ℝ,
  (∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₁ < 1 ∧ 1 < x₂ ∧
    quadratic_equation a x₁ = 0 ∧ quadratic_equation a x₂ = 0) →
  -2/11 < a ∧ a < 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_range_l1983_198387


namespace NUMINAMATH_CALUDE_sparrow_count_l1983_198326

theorem sparrow_count (bluebird_count : ℕ) (ratio_bluebird : ℕ) (ratio_sparrow : ℕ) 
  (h1 : bluebird_count = 28)
  (h2 : ratio_bluebird = 4)
  (h3 : ratio_sparrow = 5) :
  (bluebird_count * ratio_sparrow) / ratio_bluebird = 35 :=
by sorry

end NUMINAMATH_CALUDE_sparrow_count_l1983_198326


namespace NUMINAMATH_CALUDE_alpha_values_l1983_198361

theorem alpha_values (α : Real) 
  (h1 : 0 < α ∧ α < 2 * Real.pi)
  (h2 : Real.sin α = Real.cos α)
  (h3 : (Real.sin α > 0 ∧ Real.cos α > 0) ∨ (Real.sin α < 0 ∧ Real.cos α < 0)) :
  α = Real.pi / 4 ∨ α = 5 * Real.pi / 4 := by
sorry

end NUMINAMATH_CALUDE_alpha_values_l1983_198361


namespace NUMINAMATH_CALUDE_min_distance_sum_l1983_198378

/-- A scalene triangle with sides a, b, c where a > b > c -/
structure ScaleneTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  scalene : a > b ∧ b > c
  positive : a > 0 ∧ b > 0 ∧ c > 0

/-- A point inside or on the boundary of a triangle -/
structure TrianglePoint (t : ScaleneTriangle) where
  x : ℝ
  y : ℝ
  z : ℝ
  in_triangle : x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ x + y + z ≤ t.a

/-- The sum of distances from a point to the sides of the triangle -/
def distance_sum (t : ScaleneTriangle) (p : TrianglePoint t) : ℝ :=
  p.x + p.y + p.z

/-- The vertex opposite to the largest side -/
def opposite_vertex (t : ScaleneTriangle) : TrianglePoint t where
  x := t.a
  y := 0
  z := 0
  in_triangle := by sorry

/-- Theorem: The point that minimizes the sum of distances is the vertex opposite to the largest side -/
theorem min_distance_sum (t : ScaleneTriangle) :
  ∀ p : TrianglePoint t, distance_sum t (opposite_vertex t) ≤ distance_sum t p :=
by sorry

end NUMINAMATH_CALUDE_min_distance_sum_l1983_198378


namespace NUMINAMATH_CALUDE_quadratic_equality_implies_coefficient_l1983_198332

theorem quadratic_equality_implies_coefficient (a : ℝ) :
  (∀ x : ℝ, x^2 + a*x + 9 = (x - 3)^2) → a = -6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equality_implies_coefficient_l1983_198332


namespace NUMINAMATH_CALUDE_rook_game_theorem_l1983_198334

/-- Represents the result of the game -/
inductive GameResult
  | FirstPlayerWins
  | SecondPlayerWins

/-- Represents a chessboard of size K × N with a rook in the upper right corner -/
structure ChessBoard where
  K : Nat
  N : Nat

/-- Determines the winner of the rook game based on the chessboard dimensions -/
def rook_game_winner (board : ChessBoard) : GameResult :=
  if board.K = board.N then
    GameResult.SecondPlayerWins
  else
    GameResult.FirstPlayerWins

/-- Theorem stating the winning condition for the rook game -/
theorem rook_game_theorem (board : ChessBoard) :
  rook_game_winner board =
    if board.K = board.N then
      GameResult.SecondPlayerWins
    else
      GameResult.FirstPlayerWins := by
  sorry

end NUMINAMATH_CALUDE_rook_game_theorem_l1983_198334


namespace NUMINAMATH_CALUDE_flower_cost_ratio_l1983_198357

/-- Given the conditions of Nadia's flower purchase, prove the ratio of lily cost to rose cost. -/
theorem flower_cost_ratio :
  ∀ (roses : ℕ) (lilies : ℚ) (rose_cost lily_cost total_cost : ℚ),
    roses = 20 →
    lilies = (3 / 4) * roses →
    rose_cost = 5 →
    total_cost = 250 →
    total_cost = roses * rose_cost + lilies * lily_cost →
    lily_cost / rose_cost = 2 := by
  sorry

end NUMINAMATH_CALUDE_flower_cost_ratio_l1983_198357


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1983_198381

def A : Set ℤ := {0, 3, 4}
def B : Set ℤ := {-1, 0, 2, 3}

theorem intersection_of_A_and_B : A ∩ B = {0, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1983_198381


namespace NUMINAMATH_CALUDE_elisa_current_amount_l1983_198310

def current_amount (target : ℕ) (needed : ℕ) : ℕ :=
  target - needed

theorem elisa_current_amount :
  let target : ℕ := 53
  let needed : ℕ := 16
  current_amount target needed = 37 := by
  sorry

end NUMINAMATH_CALUDE_elisa_current_amount_l1983_198310


namespace NUMINAMATH_CALUDE_sine_of_pi_thirds_minus_two_theta_l1983_198399

theorem sine_of_pi_thirds_minus_two_theta (θ : ℝ) 
  (h : Real.tan (θ + π / 12) = 2) : 
  Real.sin (π / 3 - 2 * θ) = -3 / 5 := by
sorry

end NUMINAMATH_CALUDE_sine_of_pi_thirds_minus_two_theta_l1983_198399


namespace NUMINAMATH_CALUDE_min_troupe_size_l1983_198323

theorem min_troupe_size : ∃ n : ℕ, n > 0 ∧ 8 ∣ n ∧ 10 ∣ n ∧ 12 ∣ n ∧ ∀ m : ℕ, (m > 0 ∧ 8 ∣ m ∧ 10 ∣ m ∧ 12 ∣ m) → n ≤ m :=
by
  use 120
  sorry

end NUMINAMATH_CALUDE_min_troupe_size_l1983_198323
