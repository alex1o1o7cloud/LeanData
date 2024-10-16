import Mathlib

namespace NUMINAMATH_CALUDE_triangle_properties_l4125_412569

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Theorem about properties of a specific triangle --/
theorem triangle_properties (t : Triangle) 
  (h1 : t.B = π / 3) :
  (t.a = 2 ∧ t.b = 2 * Real.sqrt 3 → t.c = 4) ∧
  (Real.tan t.A = 2 * Real.sqrt 3 → Real.tan t.C = 3 * Real.sqrt 3 / 5) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l4125_412569


namespace NUMINAMATH_CALUDE_unique_x_l4125_412515

theorem unique_x : ∃! x : ℕ, 
  (∃ k : ℕ, x = 12 * k) ∧ 
  x^2 > 200 ∧ 
  x < 30 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_unique_x_l4125_412515


namespace NUMINAMATH_CALUDE_complement_of_M_in_U_l4125_412551

def U : Set Nat := {0, 1, 2, 3, 4, 5}
def M : Set Nat := {0, 1}

theorem complement_of_M_in_U : 
  (U \ M) = {2, 3, 4, 5} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_in_U_l4125_412551


namespace NUMINAMATH_CALUDE_unique_magnitude_of_quadratic_roots_l4125_412516

theorem unique_magnitude_of_quadratic_roots (w : ℂ) :
  w^2 - 6*w + 40 = 0 → ∃! m : ℝ, ∃ w : ℂ, w^2 - 6*w + 40 = 0 ∧ Complex.abs w = m := by
  sorry

end NUMINAMATH_CALUDE_unique_magnitude_of_quadratic_roots_l4125_412516


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l4125_412501

-- Define the hyperbola E
def E (x y : ℝ) : Prop := x^2 / 4 - y^2 = 1

-- Define the asymptotes
def asymptotes (x y : ℝ) : Prop := y = x/2 ∨ y = -x/2

-- Theorem statement
theorem hyperbola_asymptotes :
  ∀ x y : ℝ, E x y → (∃ x' y' : ℝ, E x' y' ∧ asymptotes x' y') :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l4125_412501


namespace NUMINAMATH_CALUDE_quadratic_y_values_order_l4125_412582

def quadratic_function (x : ℝ) : ℝ := 3 * (x + 2)^2

theorem quadratic_y_values_order :
  ∀ (y₁ y₂ y₃ : ℝ),
  quadratic_function 1 = y₁ →
  quadratic_function 2 = y₂ →
  quadratic_function (-3) = y₃ →
  y₃ < y₁ ∧ y₁ < y₂ :=
by sorry

end NUMINAMATH_CALUDE_quadratic_y_values_order_l4125_412582


namespace NUMINAMATH_CALUDE_lives_per_player_l4125_412591

/-- Given 8 friends playing a video game with a total of 64 lives,
    prove that each friend has 8 lives. -/
theorem lives_per_player (num_friends : ℕ) (total_lives : ℕ) :
  num_friends = 8 →
  total_lives = 64 →
  total_lives / num_friends = 8 :=
by sorry

end NUMINAMATH_CALUDE_lives_per_player_l4125_412591


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l4125_412514

theorem sqrt_product_equality (x : ℝ) : 
  Real.sqrt (x * (x - 6)) = Real.sqrt x * Real.sqrt (x - 6) → x ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l4125_412514


namespace NUMINAMATH_CALUDE_train_passengers_l4125_412556

theorem train_passengers (initial_passengers : ℕ) (num_stops : ℕ) : 
  initial_passengers = 64 → num_stops = 4 → 
  (initial_passengers : ℚ) * (2/3)^num_stops = 1024/81 := by
  sorry

end NUMINAMATH_CALUDE_train_passengers_l4125_412556


namespace NUMINAMATH_CALUDE_absolute_value_equation_l4125_412573

theorem absolute_value_equation (x : ℝ) : |4*x - 3| + 2 = 2 → x = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_l4125_412573


namespace NUMINAMATH_CALUDE_locus_of_P_l4125_412577

/-- Given two variable points A and B on the x-axis and y-axis respectively,
    such that AB is in the first quadrant and has fixed length 2d,
    and a point P such that P and the origin are on opposite sides of AB,
    and PC is perpendicular to AB with length d (where C is the midpoint of AB),
    prove that P lies on the line y = x and its distance from the origin
    is between d√2 and 2d inclusive. -/
theorem locus_of_P (d : ℝ) (A B P : ℝ × ℝ) (h_d : d > 0) :
  let C := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  (A.2 = 0) →
  (B.1 = 0) →
  (A.1 ≥ 0 ∧ B.2 ≥ 0) →
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = (2*d)^2 →
  ((P.1 - C.1) * (B.1 - A.1) + (P.2 - C.2) * (B.2 - A.2) = 0) →
  ((P.1 - C.1)^2 + (P.2 - C.2)^2 = d^2) →
  (P.1 * B.2 > P.2 * A.1) →
  (P.1 = P.2 ∧ d * Real.sqrt 2 ≤ Real.sqrt (P.1^2 + P.2^2) ∧ Real.sqrt (P.1^2 + P.2^2) ≤ 2*d) :=
by sorry

end NUMINAMATH_CALUDE_locus_of_P_l4125_412577


namespace NUMINAMATH_CALUDE_money_transfer_problem_l4125_412529

/-- Represents the money transfer problem between Marco and Mary -/
theorem money_transfer_problem (marco_initial : ℕ) (mary_initial : ℕ) (mary_spends : ℕ) :
  marco_initial = 24 →
  mary_initial = 15 →
  mary_spends = 5 →
  let marco_gives := marco_initial / 2
  let mary_final := mary_initial + marco_gives - mary_spends
  let marco_final := marco_initial - marco_gives
  mary_final - marco_final = 10 := by sorry

end NUMINAMATH_CALUDE_money_transfer_problem_l4125_412529


namespace NUMINAMATH_CALUDE_min_cost_for_ten_boxes_l4125_412565

/-- Calculates the minimum cost for buying a given number of yogurt boxes under a "buy two get one free" promotion. -/
def min_cost (box_price : ℕ) (num_boxes : ℕ) : ℕ :=
  let full_price_boxes := (num_boxes + 2) / 3 * 2
  full_price_boxes * box_price

/-- Theorem stating that the minimum cost for 10 boxes of yogurt at 4 yuan each under the promotion is 28 yuan. -/
theorem min_cost_for_ten_boxes : min_cost 4 10 = 28 := by
  sorry

end NUMINAMATH_CALUDE_min_cost_for_ten_boxes_l4125_412565


namespace NUMINAMATH_CALUDE_johnny_october_savings_l4125_412594

/-- Proves that Johnny saved $49 in October given his savings and spending information. -/
theorem johnny_october_savings :
  let september_savings : ℕ := 30
  let november_savings : ℕ := 46
  let video_game_cost : ℕ := 58
  let remaining_money : ℕ := 67
  let october_savings : ℕ := 49
  september_savings + october_savings + november_savings - video_game_cost = remaining_money :=
by
  sorry

#check johnny_october_savings

end NUMINAMATH_CALUDE_johnny_october_savings_l4125_412594


namespace NUMINAMATH_CALUDE_jessica_driving_days_l4125_412566

/-- Calculates the number of days needed to meet a driving hour requirement -/
def daysToMeetRequirement (requiredHours : ℕ) (minutesPerTrip : ℕ) : ℕ :=
  let requiredMinutes := requiredHours * 60
  let minutesPerDay := minutesPerTrip * 2
  requiredMinutes / minutesPerDay

theorem jessica_driving_days :
  daysToMeetRequirement 50 20 = 75 := by
  sorry

#eval daysToMeetRequirement 50 20

end NUMINAMATH_CALUDE_jessica_driving_days_l4125_412566


namespace NUMINAMATH_CALUDE_arithmetic_evaluation_l4125_412576

theorem arithmetic_evaluation : 1523 + 180 / 60 - 223 = 1303 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_evaluation_l4125_412576


namespace NUMINAMATH_CALUDE_cage_cost_calculation_l4125_412504

/-- The cost of Keith's purchases -/
def total_cost : ℝ := 24.81

/-- The cost of the rabbit toy -/
def rabbit_toy_cost : ℝ := 6.51

/-- The cost of the pet food -/
def pet_food_cost : ℝ := 5.79

/-- The amount of money Keith found -/
def found_money : ℝ := 1.00

/-- The cost of the cage -/
def cage_cost : ℝ := total_cost - (rabbit_toy_cost + pet_food_cost) + found_money

theorem cage_cost_calculation : cage_cost = 13.51 := by
  sorry

end NUMINAMATH_CALUDE_cage_cost_calculation_l4125_412504


namespace NUMINAMATH_CALUDE_domain_of_f_l4125_412543

noncomputable def f (x : ℝ) := Real.log (x - 1) + Real.sqrt (2 - x)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | 1 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_domain_of_f_l4125_412543


namespace NUMINAMATH_CALUDE_certain_number_equation_l4125_412588

theorem certain_number_equation (x : ℝ) : ((x + 2 - 6) * 3) / 4 = 3 ↔ x = 8 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_equation_l4125_412588


namespace NUMINAMATH_CALUDE_min_value_sqrt_plus_reciprocal_l4125_412545

theorem min_value_sqrt_plus_reciprocal (x : ℝ) (h : x > 0) :
  3 * Real.sqrt x + 2 / x ≥ 5 ∧
  (3 * Real.sqrt x + 2 / x = 5 ↔ x = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sqrt_plus_reciprocal_l4125_412545


namespace NUMINAMATH_CALUDE_area_of_triangle_PFQ_l4125_412581

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 8*y

-- Define the focus F
def focus : ℝ × ℝ := (0, 2)

-- Define point P on the parabola
def point_on_parabola (P : ℝ × ℝ) : Prop :=
  parabola P.1 P.2

-- Define the distance between P and F is 6
def distance_PF (P : ℝ × ℝ) : Prop :=
  (P.1 - focus.1)^2 + (P.2 - focus.2)^2 = 6^2

-- Define point Q
def Q : ℝ × ℝ := (0, -2)

-- Theorem statement
theorem area_of_triangle_PFQ (P : ℝ × ℝ) :
  point_on_parabola P →
  distance_PF P →
  (1/2 : ℝ) * |P.1| * |Q.2 - focus.2| = 8 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_area_of_triangle_PFQ_l4125_412581


namespace NUMINAMATH_CALUDE_line_through_first_and_third_quadrants_has_positive_slope_l4125_412505

/-- A line in the xy-plane -/
structure Line where
  slope : ℝ
  passes_through_first_quadrant : Bool
  passes_through_third_quadrant : Bool

/-- Definition of a line passing through first and third quadrants -/
def passes_through_first_and_third (l : Line) : Prop :=
  l.passes_through_first_quadrant ∧ l.passes_through_third_quadrant

/-- Theorem: If a line y = kx (k ≠ 0) passes through the first and third quadrants, then k > 0 -/
theorem line_through_first_and_third_quadrants_has_positive_slope (l : Line) 
    (h1 : l.slope ≠ 0) 
    (h2 : passes_through_first_and_third l) : 
    l.slope > 0 := by
  sorry

end NUMINAMATH_CALUDE_line_through_first_and_third_quadrants_has_positive_slope_l4125_412505


namespace NUMINAMATH_CALUDE_number_exists_l4125_412540

theorem number_exists : ∃ x : ℝ, (x^2 * 9^2) / 356 = 51.193820224719104 := by
  sorry

end NUMINAMATH_CALUDE_number_exists_l4125_412540


namespace NUMINAMATH_CALUDE_campground_distance_l4125_412546

/-- The distance traveled by Sue's family to the campground -/
def distance_to_campground (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time

/-- Theorem: The distance to the campground is 300 miles -/
theorem campground_distance :
  distance_to_campground 60 5 = 300 := by
  sorry

end NUMINAMATH_CALUDE_campground_distance_l4125_412546


namespace NUMINAMATH_CALUDE_student_calculation_error_l4125_412568

theorem student_calculation_error (N : ℚ) : 
  (N / (4/5)) = ((4/5) * N + 45) → N = 100 := by
  sorry

end NUMINAMATH_CALUDE_student_calculation_error_l4125_412568


namespace NUMINAMATH_CALUDE_fibonacci_divisibility_sequence_l4125_412509

def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

theorem fibonacci_divisibility_sequence (m n : ℕ) (h : m > 0) (h' : n > 0) :
  m ∣ n → fib m ∣ fib n := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_divisibility_sequence_l4125_412509


namespace NUMINAMATH_CALUDE_total_bottles_drunk_l4125_412580

def morning_bottles : ℕ := 7
def afternoon_bottles : ℕ := 9
def evening_bottles : ℕ := 5
def night_bottles : ℕ := 3

theorem total_bottles_drunk :
  morning_bottles + afternoon_bottles + evening_bottles + night_bottles = 24 := by
  sorry

end NUMINAMATH_CALUDE_total_bottles_drunk_l4125_412580


namespace NUMINAMATH_CALUDE_z_in_first_quadrant_iff_m_gt_two_l4125_412521

-- Define the complex number z
def z (m : ℝ) : ℂ := (1 + Complex.I) * (m - 2 * Complex.I)

-- Define the condition for a complex number to be in the first quadrant
def in_first_quadrant (w : ℂ) : Prop := 0 < w.re ∧ 0 < w.im

-- Theorem statement
theorem z_in_first_quadrant_iff_m_gt_two (m : ℝ) :
  in_first_quadrant (z m) ↔ m > 2 := by sorry

end NUMINAMATH_CALUDE_z_in_first_quadrant_iff_m_gt_two_l4125_412521


namespace NUMINAMATH_CALUDE_final_alcohol_percentage_l4125_412558

/-- Calculates the final alcohol percentage after adding pure alcohol to a solution -/
theorem final_alcohol_percentage
  (initial_volume : ℝ)
  (initial_percentage : ℝ)
  (added_alcohol : ℝ)
  (h_initial_volume : initial_volume = 6)
  (h_initial_percentage : initial_percentage = 35)
  (h_added_alcohol : added_alcohol = 1.8) :
  let initial_alcohol := initial_volume * (initial_percentage / 100)
  let final_alcohol := initial_alcohol + added_alcohol
  let final_volume := initial_volume + added_alcohol
  final_alcohol / final_volume * 100 = 50 := by
sorry

end NUMINAMATH_CALUDE_final_alcohol_percentage_l4125_412558


namespace NUMINAMATH_CALUDE_series_sum_equation_l4125_412503

theorem series_sum_equation (k : ℝ) (h1 : k > 1) 
  (h2 : ∑' n, (7 * n - 3) / k^n = 5) : k = 1.2 + 0.2 * Real.sqrt 46 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_equation_l4125_412503


namespace NUMINAMATH_CALUDE_circle_area_is_one_l4125_412584

theorem circle_area_is_one (r : ℝ) (h : r > 0) :
  (8 / (2 * Real.pi * r) + 2 * r = 6 * r) → π * r^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_is_one_l4125_412584


namespace NUMINAMATH_CALUDE_min_a_value_for_quasi_periodic_function_l4125_412592

-- Define the a-level quasi-periodic function
def is_a_level_quasi_periodic (f : ℝ → ℝ) (a : ℝ) (D : Set ℝ) : Prop :=
  ∃ T : ℝ, T ≠ 0 ∧ ∀ x ∈ D, a * f x = f (x + T)

-- Define the function f on [1, 2)
def f_on_initial_interval (x : ℝ) : ℝ := 2 * x + 1

-- Main theorem
theorem min_a_value_for_quasi_periodic_function :
  ∀ f : ℝ → ℝ,
  (∀ a : ℝ, is_a_level_quasi_periodic f a (Set.Ici 1)) →
  (∀ x ∈ Set.Icc 1 2, f x = f_on_initial_interval x) →
  (∀ x y, x < y → x ≥ 1 → f x < f y) →
  (∃ a : ℝ, ∀ b : ℝ, is_a_level_quasi_periodic f b (Set.Ici 1) → a ≤ b) →
  (∀ a : ℝ, is_a_level_quasi_periodic f a (Set.Ici 1) → a ≥ 5/3) :=
by sorry

end NUMINAMATH_CALUDE_min_a_value_for_quasi_periodic_function_l4125_412592


namespace NUMINAMATH_CALUDE_center_of_specific_pyramid_l4125_412547

/-- The center of the circumscribed sphere of a triangular pyramid -/
def center_of_circumscribed_sphere (A B C D : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := sorry

/-- Theorem: The center of the circumscribed sphere of a triangular pyramid with
    vertices at (1,0,1), (1,1,0), (0,1,1), and (0,0,0) has coordinates (1/2, 1/2, 1/2) -/
theorem center_of_specific_pyramid :
  let A : ℝ × ℝ × ℝ := (1, 0, 1)
  let B : ℝ × ℝ × ℝ := (1, 1, 0)
  let C : ℝ × ℝ × ℝ := (0, 1, 1)
  let D : ℝ × ℝ × ℝ := (0, 0, 0)
  center_of_circumscribed_sphere A B C D = (1/2, 1/2, 1/2) := by sorry

end NUMINAMATH_CALUDE_center_of_specific_pyramid_l4125_412547


namespace NUMINAMATH_CALUDE_intersection_distance_l4125_412586

/-- The distance between the intersection points of y = 5 and y = 5x^2 + 2x - 2 is 2.4 -/
theorem intersection_distance : 
  let f (x : ℝ) := 5*x^2 + 2*x - 2
  let g (x : ℝ) := 5
  let roots := {x : ℝ | f x = g x}
  ∃ (x₁ x₂ : ℝ), x₁ ∈ roots ∧ x₂ ∈ roots ∧ x₁ ≠ x₂ ∧ |x₁ - x₂| = 2.4 :=
by sorry

end NUMINAMATH_CALUDE_intersection_distance_l4125_412586


namespace NUMINAMATH_CALUDE_max_product_sum_246_l4125_412542

theorem max_product_sum_246 : 
  ∀ x y : ℤ, x + y = 246 → x * y ≤ 15129 :=
by
  sorry

end NUMINAMATH_CALUDE_max_product_sum_246_l4125_412542


namespace NUMINAMATH_CALUDE_inequality_proof_l4125_412598

theorem inequality_proof (x y z : ℝ) (h : x * y * z + x + y + z = 4) :
  (y * z + 6)^2 + (z * x + 6)^2 + (x * y + 6)^2 ≥ 8 * (x * y * z + 5) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4125_412598


namespace NUMINAMATH_CALUDE_sum_of_extreme_prime_factors_of_1365_l4125_412522

theorem sum_of_extreme_prime_factors_of_1365 :
  ∃ (smallest largest : ℕ),
    smallest.Prime ∧ largest.Prime ∧
    smallest ∣ 1365 ∧ largest ∣ 1365 ∧
    (∀ p : ℕ, p.Prime → p ∣ 1365 → p ≤ largest) ∧
    (∀ p : ℕ, p.Prime → p ∣ 1365 → p ≥ smallest) ∧
    smallest + largest = 16 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_extreme_prime_factors_of_1365_l4125_412522


namespace NUMINAMATH_CALUDE_solution_count_equals_r_l4125_412597

def r (n : ℕ) : ℚ := (1/2 : ℚ) * (n + 1 : ℚ) + (1/4 : ℚ) * (1 + (-1)^n : ℚ)

def count_solutions (n : ℕ) : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ => p.1 + 2 * p.2 = n) (Finset.product (Finset.range (n + 1)) (Finset.range (n + 1)))).card

theorem solution_count_equals_r (n : ℕ) : 
  (count_solutions n : ℚ) = r n :=
sorry

end NUMINAMATH_CALUDE_solution_count_equals_r_l4125_412597


namespace NUMINAMATH_CALUDE_triangle_properties_l4125_412534

open Real

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  A + B + C = π →
  -- Given condition
  cos (2*C) - cos (2*A) = 2 * sin (π/3 + C) * sin (π/3 - C) →
  -- Part 1: Prove A = π/3
  A = π/3 ∧
  -- Part 2: Prove range of 2b-c
  (a = sqrt 3 ∧ b ≥ a → 2*b - c ≥ sqrt 3 ∧ 2*b - c < 2 * sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l4125_412534


namespace NUMINAMATH_CALUDE_field_reduction_l4125_412561

theorem field_reduction (L W : ℝ) (h : L > 0 ∧ W > 0) :
  ∃ x : ℝ, x > 0 ∧ x < 100 ∧ 
  (1 - x / 100) * (1 - x / 100) * (L * W) = (1 - 0.64) * (L * W) →
  x = 40 := by
sorry

end NUMINAMATH_CALUDE_field_reduction_l4125_412561


namespace NUMINAMATH_CALUDE_max_value_theorem_l4125_412507

theorem max_value_theorem (a b : ℝ) 
  (h1 : 4 * a + 3 * b ≤ 9) 
  (h2 : 3 * a + 5 * b ≤ 12) : 
  2 * a + b ≤ 39 / 11 := by
  sorry

end NUMINAMATH_CALUDE_max_value_theorem_l4125_412507


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l4125_412571

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  ((2*x^2 + 5*x + 2)*(2*y^2 + 5*y + 2)*(2*z^2 + 5*z + 2)) / (x*y*z*(1+x)*(1+y)*(1+z)) ≥ 729/8 :=
by sorry

theorem min_value_achievable :
  ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧
  ((2*x^2 + 5*x + 2)*(2*y^2 + 5*y + 2)*(2*z^2 + 5*z + 2)) / (x*y*z*(1+x)*(1+y)*(1+z)) = 729/8 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l4125_412571


namespace NUMINAMATH_CALUDE_triangle_inequality_check_l4125_412572

def canFormTriangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_inequality_check :
  (canFormTriangle 2 3 4) ∧
  ¬(canFormTriangle 3 4 7) ∧
  ¬(canFormTriangle 4 6 2) ∧
  ¬(canFormTriangle 7 10 2) :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_check_l4125_412572


namespace NUMINAMATH_CALUDE_chebyshev_sin_substitution_l4125_412520

-- Define Chebyshev polynomials of the first kind
noncomputable def T (n : ℕ) : ℝ → ℝ :=
  λ x => Real.cos (n * Real.arccos x)

-- Define Chebyshev polynomials of the second kind
noncomputable def U (n : ℕ) : ℝ → ℝ :=
  λ x => (Real.sin ((n + 1) * Real.arccos x)) / Real.sin (Real.arccos x)

theorem chebyshev_sin_substitution (n : ℕ) (α : ℝ) :
  (∃ k : ℕ, n = 2 * k ∧ 
    T n (Real.sin α) = (-1)^k * Real.cos (n * α) ∧
    U (n - 1) (Real.sin α) = (-1)^(k + 1) * Real.sin (n * α) / Real.cos α) ∨
  (∃ k : ℕ, n = 2 * k + 1 ∧ 
    T n (Real.sin α) = (-1)^k * Real.sin (n * α) ∧
    U (n - 1) (Real.sin α) = (-1)^k * Real.cos (n * α) / Real.cos α) :=
by sorry

end NUMINAMATH_CALUDE_chebyshev_sin_substitution_l4125_412520


namespace NUMINAMATH_CALUDE_multiplication_division_remainder_problem_l4125_412511

theorem multiplication_division_remainder_problem :
  ∃ (x : ℕ), (55 * x) % 8 = 7 ∧ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_division_remainder_problem_l4125_412511


namespace NUMINAMATH_CALUDE_both_players_is_zero_l4125_412596

/-- The number of people who play both kabadi and kho kho -/
def both_players : ℕ := sorry

/-- The number of people who play kabadi (including those who play both) -/
def kabadi_players : ℕ := 10

/-- The number of people who play kho kho only -/
def kho_kho_only_players : ℕ := 15

/-- The total number of players -/
def total_players : ℕ := 25

/-- Theorem stating that the number of people playing both games is 0 -/
theorem both_players_is_zero : both_players = 0 := by
  sorry

#check both_players_is_zero

end NUMINAMATH_CALUDE_both_players_is_zero_l4125_412596


namespace NUMINAMATH_CALUDE_max_books_borrowed_l4125_412533

theorem max_books_borrowed (total_students : ℕ) (zero_books : ℕ) (one_book : ℕ) (two_books : ℕ) 
  (avg_books : ℚ) (h1 : total_students = 60) (h2 : zero_books = 4) (h3 : one_book = 18) 
  (h4 : two_books = 20) (h5 : avg_books = 5/2) : 
  ∃ (max_books : ℕ), max_books = 41 ∧ 
  ∀ (student_books : ℕ), student_books ≤ max_books :=
by
  sorry

end NUMINAMATH_CALUDE_max_books_borrowed_l4125_412533


namespace NUMINAMATH_CALUDE_composition_of_even_is_even_l4125_412587

-- Define an even function
def EvenFunction (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = g x

-- State the theorem
theorem composition_of_even_is_even (g : ℝ → ℝ) (h : EvenFunction g) : 
  EvenFunction (g ∘ g) := by
  sorry

end NUMINAMATH_CALUDE_composition_of_even_is_even_l4125_412587


namespace NUMINAMATH_CALUDE_compute_expression_l4125_412538

theorem compute_expression (x : ℝ) (h : x = 9) : 
  (x^9 - 27*x^6 + 729) / (x^6 - 27) = 730 + 1/26 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l4125_412538


namespace NUMINAMATH_CALUDE_triangle_angle_bound_triangle_side_ratio_l4125_412539

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : Real
  B : Real
  C : Real

/-- The given conditions for the triangle -/
def TriangleConditions (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.A + t.B + t.C = Real.pi ∧
  t.a + t.c = 2 * t.b

theorem triangle_angle_bound (t : Triangle) (h : TriangleConditions t) : t.B ≤ Real.pi / 3 := by
  sorry

theorem triangle_side_ratio (t : Triangle) (h : TriangleConditions t) (h2 : t.C = 2 * t.A) :
  ∃ (k : ℝ), t.a = 4 * k ∧ t.b = 5 * k ∧ t.c = 6 * k := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_bound_triangle_side_ratio_l4125_412539


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l4125_412552

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) (d : ℝ) :
  d ≠ 0 ∧
  (∀ n : ℕ, a (n + 1) = a n + d) ∧
  ∃ r : ℝ, r ≠ 0 ∧ a 3 = r * a 1 ∧ a 6 = r * a 3
  →
  ∃ r : ℝ, r = 3/2 ∧ a 3 = r * a 1 ∧ a 6 = r * a 3 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l4125_412552


namespace NUMINAMATH_CALUDE_order_relation_l4125_412554

theorem order_relation (a b c : ℝ) : 
  a = Real.exp 0.2 → b = 0.2 ^ Real.exp 1 → c = Real.log 2 → b < c ∧ c < a := by
  sorry

end NUMINAMATH_CALUDE_order_relation_l4125_412554


namespace NUMINAMATH_CALUDE_gcd_90_405_l4125_412500

theorem gcd_90_405 : Nat.gcd 90 405 = 45 := by
  sorry

end NUMINAMATH_CALUDE_gcd_90_405_l4125_412500


namespace NUMINAMATH_CALUDE_sufficient_condition_for_inequality_l4125_412585

theorem sufficient_condition_for_inequality (a : ℝ) (h : a ≥ 5) :
  ∀ x ∈ Set.Icc 1 2, x^2 - a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_inequality_l4125_412585


namespace NUMINAMATH_CALUDE_complex_equation_solution_l4125_412506

theorem complex_equation_solution (z : ℂ) :
  (1 - Complex.I)^2 * z = 3 + 2 * Complex.I →
  z = -1 + (3/2) * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l4125_412506


namespace NUMINAMATH_CALUDE_problem_solution_l4125_412557

theorem problem_solution :
  ∀ (x a b : ℝ),
  (∃ y : ℝ, y^2 = x ∧ y = a + 3) ∧
  (∃ z : ℝ, z^2 = x ∧ z = 2*a - 15) ∧
  (3^2 = 2*b - 1) →
  (a = 4 ∧ b = 5 ∧ (a + b - 1)^(1/3) = 2) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l4125_412557


namespace NUMINAMATH_CALUDE_print_height_preservation_l4125_412593

/-- Given a painting and its print with preserved aspect ratio, calculate the height of the print -/
theorem print_height_preservation (original_width original_height print_width : ℝ) 
  (hw : original_width = 15) 
  (hh : original_height = 10) 
  (pw : print_width = 37.5) :
  let print_height := (print_width * original_height) / original_width
  print_height = 25 := by
  sorry

end NUMINAMATH_CALUDE_print_height_preservation_l4125_412593


namespace NUMINAMATH_CALUDE_unique_solution_divisibility_l4125_412531

theorem unique_solution_divisibility : ∀ a b : ℕ+,
  (∃ k l : ℕ+, (a^2 + b^2 : ℕ) * k = a^3 + 1 ∧ (a^2 + b^2 : ℕ) * l = b^3 + 1) →
  a = 1 ∧ b = 1 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_divisibility_l4125_412531


namespace NUMINAMATH_CALUDE_base8_45_equals_base10_37_l4125_412553

/-- Converts a two-digit base-eight number to base-ten -/
def base8_to_base10 (tens : Nat) (units : Nat) : Nat :=
  tens * 8 + units

/-- The base-eight number 45 is equal to the base-ten number 37 -/
theorem base8_45_equals_base10_37 : base8_to_base10 4 5 = 37 := by
  sorry

end NUMINAMATH_CALUDE_base8_45_equals_base10_37_l4125_412553


namespace NUMINAMATH_CALUDE_five_balls_three_boxes_l4125_412555

/-- The number of ways to place n distinguishable objects into k distinguishable containers -/
def ways_to_place (n k : ℕ) : ℕ := k^n

/-- Theorem: There are 3^5 ways to put 5 distinguishable balls into 3 distinguishable boxes -/
theorem five_balls_three_boxes : ways_to_place 5 3 = 243 := by sorry

end NUMINAMATH_CALUDE_five_balls_three_boxes_l4125_412555


namespace NUMINAMATH_CALUDE_initial_speed_is_60_l4125_412532

/-- Represents the initial speed of a traveler given specific journey conditions -/
def initial_speed (D T : ℝ) : ℝ :=
  let remaining_time := T - T / 3
  let remaining_distance := D / 3
  60

/-- Theorem stating the initial speed under given conditions -/
theorem initial_speed_is_60 (D T : ℝ) (h1 : D > 0) (h2 : T > 0) :
  initial_speed D T = 60 := by
  sorry

#check initial_speed_is_60

end NUMINAMATH_CALUDE_initial_speed_is_60_l4125_412532


namespace NUMINAMATH_CALUDE_perpendicular_parallel_implication_parallel_perpendicular_transitivity_l4125_412549

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_plane : Plane → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)

-- Theorem 1
theorem perpendicular_parallel_implication 
  (m n : Line) (α : Plane) : 
  perpendicular m α → parallel_line_plane n α → perpendicular_lines m n :=
sorry

-- Theorem 2
theorem parallel_perpendicular_transitivity 
  (m : Line) (α β γ : Plane) :
  parallel_plane α β → parallel_plane β γ → perpendicular m α → perpendicular m γ :=
sorry

end NUMINAMATH_CALUDE_perpendicular_parallel_implication_parallel_perpendicular_transitivity_l4125_412549


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l4125_412519

/-- A quadratic function f(x) with the property that f(x) > 0 for all x except x = 1/a -/
def QuadraticFunction (a c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 - 2 * x + c

theorem quadratic_function_properties (a c : ℝ) :
  (∀ x, QuadraticFunction a c x > 0 ↔ x ≠ 1/a) →
  (∃ f : ℝ → ℝ, (∀ x, f x = (1/2) * x^2 - 2 * x + 2) ∧
    QuadraticFunction a c 2 ≥ 0 ∧
    (QuadraticFunction a c 2 = 0 → ∀ x, QuadraticFunction a c x = f x) ∧
    (∀ m : ℝ, (∀ x > 2, f x + 4 ≥ m * (x - 2)) → m ≤ 2 * Real.sqrt 2)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l4125_412519


namespace NUMINAMATH_CALUDE_range_of_a_for_meaningful_sqrt_l4125_412564

theorem range_of_a_for_meaningful_sqrt (a : ℝ) : 
  (∃ x : ℝ, x^2 = 4 - a) ↔ a ≤ 4 := by sorry

end NUMINAMATH_CALUDE_range_of_a_for_meaningful_sqrt_l4125_412564


namespace NUMINAMATH_CALUDE_function_property_l4125_412535

theorem function_property (f : ℕ+ → ℕ) 
  (h1 : ∀ (x y : ℕ+), f (x * y) = f x + f y)
  (h2 : f 30 = 21)
  (h3 : f 90 = 27) :
  f 270 = 33 := by
sorry

end NUMINAMATH_CALUDE_function_property_l4125_412535


namespace NUMINAMATH_CALUDE_complex_on_imaginary_axis_l4125_412563

theorem complex_on_imaginary_axis (a : ℝ) :
  let z : ℂ := Complex.mk (a^2 - 2*a) (a^2 - a - 2)
  (Complex.re z = 0) → (a = 0 ∨ a = 2) :=
by sorry

end NUMINAMATH_CALUDE_complex_on_imaginary_axis_l4125_412563


namespace NUMINAMATH_CALUDE_jimmy_payment_jimmy_paid_fifty_l4125_412574

/-- The amount Jimmy paid with, given his purchases and change received. -/
theorem jimmy_payment (pen_price notebook_price folder_price : ℕ)
  (pen_count notebook_count folder_count : ℕ)
  (change : ℕ) : ℕ :=
  let total_cost := pen_price * pen_count + notebook_price * notebook_count + folder_price * folder_count
  total_cost + change

/-- Proof that Jimmy paid $50 given his purchases and change received. -/
theorem jimmy_paid_fifty :
  jimmy_payment 1 3 5 3 4 2 25 = 50 := by
  sorry

end NUMINAMATH_CALUDE_jimmy_payment_jimmy_paid_fifty_l4125_412574


namespace NUMINAMATH_CALUDE_max_value_of_a_l4125_412517

theorem max_value_of_a (a : ℝ) : 
  (∀ x : ℝ, x < a → x^2 > 1) ∧ 
  (∃ x : ℝ, x^2 > 1 ∧ x ≥ a) → 
  a ≤ -1 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_a_l4125_412517


namespace NUMINAMATH_CALUDE_unique_solution_condition_l4125_412570

theorem unique_solution_condition (a b : ℝ) :
  (∃! x : ℝ, 4 * x - 7 + a = (b - 1) * x + 2) ↔ b ≠ 5 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l4125_412570


namespace NUMINAMATH_CALUDE_two_colonies_reach_limit_same_time_l4125_412528

/-- Represents the growth of a bacteria colony -/
structure BacteriaColony where
  growthRate : ℕ → ℕ
  limitDay : ℕ

/-- The number of days it takes for two colonies to reach the habitat's limit -/
def daysToLimitTwoColonies (colony : BacteriaColony) : ℕ := sorry

theorem two_colonies_reach_limit_same_time (colony : BacteriaColony) 
  (h1 : ∀ n : ℕ, colony.growthRate n = 2 * colony.growthRate (n - 1))
  (h2 : colony.limitDay = 16) :
  daysToLimitTwoColonies colony = colony.limitDay := by sorry

end NUMINAMATH_CALUDE_two_colonies_reach_limit_same_time_l4125_412528


namespace NUMINAMATH_CALUDE_taxi_charge_calculation_l4125_412537

/-- Calculates the taxi charge per mile given the initial fee, total distance, and total payment. -/
def taxi_charge_per_mile (initial_fee : ℚ) (total_distance : ℚ) (total_payment : ℚ) : ℚ :=
  (total_payment - initial_fee) / total_distance

/-- Theorem stating that the taxi charge per mile is $2.50 given the specific conditions. -/
theorem taxi_charge_calculation (initial_fee : ℚ) (total_distance : ℚ) (total_payment : ℚ)
    (h1 : initial_fee = 2)
    (h2 : total_distance = 4)
    (h3 : total_payment = 12) :
    taxi_charge_per_mile initial_fee total_distance total_payment = 2.5 := by
  sorry

#eval taxi_charge_per_mile 2 4 12

end NUMINAMATH_CALUDE_taxi_charge_calculation_l4125_412537


namespace NUMINAMATH_CALUDE_sin_cos_sum_identity_l4125_412525

open Real

theorem sin_cos_sum_identity : 
  sin (15 * π / 180) * cos (75 * π / 180) + cos (15 * π / 180) * sin (105 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_identity_l4125_412525


namespace NUMINAMATH_CALUDE_fourth_vertex_location_l4125_412536

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle defined by its four vertices -/
structure Rectangle where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Theorem: Given a rectangle ABCD with specific constraints, the fourth vertex C is at (x, -y) -/
theorem fourth_vertex_location (a y d : ℝ) :
  let O : Point := ⟨0, 0⟩
  let A : Point := ⟨a, 0⟩
  let B : Point := ⟨a, y⟩
  let D : Point := ⟨0, d⟩
  ∀ (rect : Rectangle),
    rect.A = A →
    rect.B = B →
    rect.D = D →
    (O.x - rect.C.x) * (A.x - B.x) + (O.y - rect.C.y) * (A.y - B.y) = 0 →
    (O.x - D.x) * (A.x - rect.C.x) + (O.y - D.y) * (A.y - rect.C.y) = 0 →
    rect.C = ⟨a, -y⟩ :=
by
  sorry


end NUMINAMATH_CALUDE_fourth_vertex_location_l4125_412536


namespace NUMINAMATH_CALUDE_pot_stacking_l4125_412513

theorem pot_stacking (total_pots : ℕ) (vertical_stack : ℕ) (shelves : ℕ) 
  (h1 : total_pots = 60)
  (h2 : vertical_stack = 5)
  (h3 : shelves = 4) :
  (total_pots / vertical_stack) / shelves = 3 := by
  sorry

end NUMINAMATH_CALUDE_pot_stacking_l4125_412513


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_shifted_l4125_412518

theorem root_sum_reciprocal_shifted (p q r : ℂ) : 
  (p^3 - 2*p + 2 = 0) → 
  (q^3 - 2*q + 2 = 0) → 
  (r^3 - 2*r + 2 = 0) → 
  (1/(p+2) + 1/(q+2) + 1/(r+2) = 3/5) := by
sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_shifted_l4125_412518


namespace NUMINAMATH_CALUDE_least_three_digit_7_heavy_correct_l4125_412579

/-- A number is 7-heavy if its remainder when divided by 7 is greater than 4 -/
def is_7_heavy (n : ℕ) : Prop := n % 7 > 4

/-- The least three-digit 7-heavy number -/
def least_three_digit_7_heavy : ℕ := 103

theorem least_three_digit_7_heavy_correct :
  (least_three_digit_7_heavy ≥ 100) ∧
  (least_three_digit_7_heavy < 1000) ∧
  is_7_heavy least_three_digit_7_heavy ∧
  ∀ n : ℕ, (n ≥ 100) ∧ (n < 1000) ∧ is_7_heavy n → n ≥ least_three_digit_7_heavy :=
by sorry

end NUMINAMATH_CALUDE_least_three_digit_7_heavy_correct_l4125_412579


namespace NUMINAMATH_CALUDE_geometric_difference_ratio_l4125_412562

def geometric_difference (a : ℕ+ → ℝ) (d : ℝ) :=
  ∀ n : ℕ+, a (n + 2) / a (n + 1) - a (n + 1) / a n = d

theorem geometric_difference_ratio 
  (a : ℕ+ → ℝ) 
  (h1 : geometric_difference a 2)
  (h2 : a 1 = 1)
  (h3 : a 2 = 1)
  (h4 : a 3 = 3) :
  a 12 / a 10 = 399 := by
sorry

end NUMINAMATH_CALUDE_geometric_difference_ratio_l4125_412562


namespace NUMINAMATH_CALUDE_function_properties_l4125_412599

/-- The function f(x) -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (2*a - 8)*x

/-- The theorem stating the properties of the function and the results to be proved -/
theorem function_properties (a m : ℝ) :
  (∀ x, f a x ≤ 5 ↔ -1 ≤ x ∧ x ≤ 5) →
  (∀ x, f a x ≥ m^2 - 4*m - 9) →
  a = 2 ∧ -1 ≤ m ∧ m ≤ 5 := by sorry

end NUMINAMATH_CALUDE_function_properties_l4125_412599


namespace NUMINAMATH_CALUDE_equation_equivalence_l4125_412550

theorem equation_equivalence (x : ℝ) : 
  (x - 1) / 2 - (2 * x + 3) / 3 = 1 ↔ 3 * (x - 1) - 2 * (2 * x + 3) = 6 := by
  sorry

end NUMINAMATH_CALUDE_equation_equivalence_l4125_412550


namespace NUMINAMATH_CALUDE_prob_2012_higher_than_2011_l4125_412541

/-- The probability of guessing the correct answer to a single question -/
def p : ℝ := 0.25

/-- The probability of guessing incorrectly -/
def q : ℝ := 1 - p

/-- Calculate the probability of passing the exam given the total number of questions and the minimum required correct answers -/
def prob_pass (n : ℕ) (k : ℕ) : ℝ :=
  1 - (Finset.sum (Finset.range k) (λ i => Nat.choose n i * p^i * q^(n - i)))

/-- The probability of passing the exam in 2011 -/
def prob_2011 : ℝ := prob_pass 20 3

/-- The probability of passing the exam in 2012 -/
def prob_2012 : ℝ := prob_pass 40 6

/-- Theorem stating that the probability of passing in 2012 is higher than in 2011 -/
theorem prob_2012_higher_than_2011 : prob_2012 > prob_2011 := by
  sorry

end NUMINAMATH_CALUDE_prob_2012_higher_than_2011_l4125_412541


namespace NUMINAMATH_CALUDE_root_of_two_equations_l4125_412560

theorem root_of_two_equations (a b c d k : ℂ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (h1 : a * k^3 + b * k^2 - c * k + d = 0)
  (h2 : -b * k^3 + c * k^2 - d * k + a = 0) :
  k^4 = -1 := by
sorry

end NUMINAMATH_CALUDE_root_of_two_equations_l4125_412560


namespace NUMINAMATH_CALUDE_rectangle_formation_ways_l4125_412583

/-- The number of ways to choose 2 items from a set of 5 -/
def choose_2_from_5 : ℕ := 10

/-- The number of horizontal lines -/
def num_horizontal_lines : ℕ := 5

/-- The number of vertical lines -/
def num_vertical_lines : ℕ := 5

/-- The number of lines needed to form a rectangle -/
def lines_for_rectangle : ℕ := 4

/-- Theorem: The number of ways to choose 4 lines (2 horizontal and 2 vertical) 
    from 5 horizontal and 5 vertical lines to form a rectangle is 100 -/
theorem rectangle_formation_ways : 
  (choose_2_from_5 * choose_2_from_5 = 100) ∧ 
  (num_horizontal_lines = 5) ∧ 
  (num_vertical_lines = 5) ∧ 
  (lines_for_rectangle = 4) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_formation_ways_l4125_412583


namespace NUMINAMATH_CALUDE_distance_to_circle_center_l4125_412575

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the circle
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

-- Define the problem
theorem distance_to_circle_center 
  (ABC : Triangle)
  (circle : Circle)
  (M : ℝ × ℝ)
  (h1 : ABC.C.1 = ABC.A.1 ∧ ABC.C.2 = ABC.B.2) -- Right triangle condition
  (h2 : (ABC.A.1 - ABC.B.1)^2 + (ABC.A.2 - ABC.B.2)^2 = 
        (ABC.C.1 - ABC.B.1)^2 + (ABC.C.2 - ABC.B.2)^2) -- Equal legs condition
  (h3 : circle.radius = (ABC.C.1 - ABC.A.1) / 2) -- Circle diameter is AC
  (h4 : circle.center = ((ABC.A.1 + ABC.C.1) / 2, (ABC.A.2 + ABC.C.2) / 2)) -- Circle center is midpoint of AC
  (h5 : (M.1 - ABC.A.1)^2 + (M.2 - ABC.A.2)^2 = circle.radius^2) -- M is on the circle
  (h6 : (M.1 - ABC.B.1)^2 + (M.2 - ABC.B.2)^2 = 2) -- BM = √2
  : (ABC.B.1 - circle.center.1)^2 + (ABC.B.2 - circle.center.2)^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_circle_center_l4125_412575


namespace NUMINAMATH_CALUDE_shaded_area_calculation_l4125_412589

theorem shaded_area_calculation (large_square_area medium_square_area small_square_area : ℝ)
  (h1 : large_square_area = 49)
  (h2 : medium_square_area = 25)
  (h3 : small_square_area = 9) :
  small_square_area + (large_square_area - medium_square_area) = 33 := by
sorry

end NUMINAMATH_CALUDE_shaded_area_calculation_l4125_412589


namespace NUMINAMATH_CALUDE_square_difference_given_sum_and_product_l4125_412595

theorem square_difference_given_sum_and_product (x y : ℝ) 
  (h1 : (x + y)^2 = 36) (h2 : x * y = 8) : (x - y)^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_given_sum_and_product_l4125_412595


namespace NUMINAMATH_CALUDE_digit_equation_solution_l4125_412526

theorem digit_equation_solution (A M C : ℕ) : 
  A ≤ 9 ∧ M ≤ 9 ∧ C ≤ 9 →
  (100 * A + 10 * M + C) * (A + M + C) = 2040 →
  Even (A + M + C) →
  M = 7 :=
by sorry

end NUMINAMATH_CALUDE_digit_equation_solution_l4125_412526


namespace NUMINAMATH_CALUDE_x_value_proof_l4125_412530

theorem x_value_proof (x y : ℝ) (hx : x ≠ 0) (h1 : x / 3 = y^3) (h2 : x / 9 = 9*y) :
  x = 243 * Real.sqrt 3 ∨ x = -243 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l4125_412530


namespace NUMINAMATH_CALUDE_floor_expression_l4125_412512

theorem floor_expression (n : ℕ) (hn : n = 101) : 
  ⌊(8 * (n^2 + 1) : ℚ) / (n^2 - 1)⌋ = 8 := by
  sorry

end NUMINAMATH_CALUDE_floor_expression_l4125_412512


namespace NUMINAMATH_CALUDE_probability_1AECD_l4125_412524

-- Define the structure of the license plate
structure LicensePlate where
  digit : Fin 10
  vowel1 : Fin 5
  vowel2 : Fin 5
  consonant1 : Fin 21
  consonant2 : Fin 21
  different_consonants : consonant1 ≠ consonant2

-- Define the total number of possible license plates
def total_plates : ℕ := 10 * 5 * 5 * 21 * 20

-- Define the probability of a specific plate
def probability_specific_plate : ℚ := 1 / total_plates

-- Theorem to prove
theorem probability_1AECD :
  probability_specific_plate = 1 / 105000 :=
sorry

end NUMINAMATH_CALUDE_probability_1AECD_l4125_412524


namespace NUMINAMATH_CALUDE_leila_cake_problem_l4125_412544

theorem leila_cake_problem (monday : ℕ) (friday : ℕ) (saturday : ℕ) : 
  friday = 9 →
  saturday = 3 * monday →
  monday + friday + saturday = 33 →
  monday = 6 := by
sorry

end NUMINAMATH_CALUDE_leila_cake_problem_l4125_412544


namespace NUMINAMATH_CALUDE_largest_integer_four_digits_base_seven_l4125_412527

def has_four_digits_base_seven (n : ℕ) : Prop :=
  7^3 ≤ n^2 ∧ n^2 < 7^4

theorem largest_integer_four_digits_base_seven :
  ∃ M : ℕ, has_four_digits_base_seven M ∧
    ∀ n : ℕ, has_four_digits_base_seven n → n ≤ M ∧
    M = 48 :=
sorry

end NUMINAMATH_CALUDE_largest_integer_four_digits_base_seven_l4125_412527


namespace NUMINAMATH_CALUDE_largest_n_divisibility_l4125_412502

theorem largest_n_divisibility : ∃ (n : ℕ), n > 0 ∧ (n + 12) ∣ (n^3 + 144) ∧ ∀ (m : ℕ), m > n → ¬((m + 12) ∣ (m^3 + 144)) :=
by
  use 780
  sorry

end NUMINAMATH_CALUDE_largest_n_divisibility_l4125_412502


namespace NUMINAMATH_CALUDE_problem_statement_l4125_412559

theorem problem_statement (a b c : ℝ) : 
  a < b →
  (∀ x : ℝ, (x - a) * (x - b) / (x - c) ≤ 0 ↔ (x < -1 ∨ |x - 10| ≤ 2)) →
  a + 2 * b + 3 * c = 29 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l4125_412559


namespace NUMINAMATH_CALUDE_product_of_roots_plus_two_l4125_412578

theorem product_of_roots_plus_two (u v w : ℝ) : 
  (u^3 - 18*u^2 + 20*u - 8 = 0) →
  (v^3 - 18*v^2 + 20*v - 8 = 0) →
  (w^3 - 18*w^2 + 20*w - 8 = 0) →
  (2+u)*(2+v)*(2+w) = 128 := by
sorry

end NUMINAMATH_CALUDE_product_of_roots_plus_two_l4125_412578


namespace NUMINAMATH_CALUDE_line_equation_through_point_with_inclination_l4125_412567

/-- The equation of a line passing through (-2, 3) with an inclination angle of 45° is x - y + 5 = 0 -/
theorem line_equation_through_point_with_inclination (x y : ℝ) : 
  (∃ (m : ℝ), m = Real.tan (π / 4) ∧ 
    y - 3 = m * (x - (-2))) ↔ 
  x - y + 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_equation_through_point_with_inclination_l4125_412567


namespace NUMINAMATH_CALUDE_grape_popsicles_count_l4125_412510

/-- The number of cherry popsicles -/
def cherry_popsicles : ℕ := 13

/-- The number of banana popsicles -/
def banana_popsicles : ℕ := 2

/-- The total number of popsicles -/
def total_popsicles : ℕ := 17

/-- The number of grape popsicles -/
def grape_popsicles : ℕ := total_popsicles - cherry_popsicles - banana_popsicles

theorem grape_popsicles_count : grape_popsicles = 2 := by
  sorry

end NUMINAMATH_CALUDE_grape_popsicles_count_l4125_412510


namespace NUMINAMATH_CALUDE_complementary_sets_imply_a_eq_two_subset_implies_a_range_l4125_412548

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < a + 1}
def B : Set ℝ := {x | x ≥ 3 ∨ x ≤ 1}

-- Theorem 1: If A ∩ B = ∅ and A ∪ B = ℝ, then a = 2
theorem complementary_sets_imply_a_eq_two (a : ℝ) :
  A a ∩ B = ∅ ∧ A a ∪ B = Set.univ → a = 2 := by sorry

-- Theorem 2: If A ⊆ B, then a ∈ (-∞, 0] ∪ [4, +∞)
theorem subset_implies_a_range (a : ℝ) :
  A a ⊆ B → a ≤ 0 ∨ a ≥ 4 := by sorry

end NUMINAMATH_CALUDE_complementary_sets_imply_a_eq_two_subset_implies_a_range_l4125_412548


namespace NUMINAMATH_CALUDE_remainder_theorem_l4125_412523

def polynomial (x : ℝ) : ℝ := 5*x^5 - 12*x^4 + 3*x^3 - 7*x^2 + x - 30

def divisor (x : ℝ) : ℝ := 3*x - 9

theorem remainder_theorem :
  ∃ (q : ℝ → ℝ), ∀ (x : ℝ), 
    polynomial x = (divisor x) * (q x) + 234 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l4125_412523


namespace NUMINAMATH_CALUDE_circle_area_ratio_l4125_412590

theorem circle_area_ratio (s r : ℝ) (hs : s > 0) (hr : r > 0) 
  (h : r = 0.8 * s) : 
  (π * (r / 2)^2) / (π * (s / 2)^2) = 0.64 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l4125_412590


namespace NUMINAMATH_CALUDE_interest_rate_difference_l4125_412508

/-- Given a principal amount, time period, and difference in interest earned,
    calculate the difference between two simple interest rates. -/
theorem interest_rate_difference
  (principal : ℝ)
  (time : ℝ)
  (interest_diff : ℝ)
  (h_principal : principal = 2300)
  (h_time : time = 3)
  (h_interest_diff : interest_diff = 69) :
  let rate_diff := interest_diff / (principal * time / 100)
  rate_diff = 1 := by sorry

end NUMINAMATH_CALUDE_interest_rate_difference_l4125_412508
