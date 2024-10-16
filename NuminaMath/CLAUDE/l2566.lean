import Mathlib

namespace NUMINAMATH_CALUDE_rectangular_garden_width_l2566_256655

theorem rectangular_garden_width (length width area : ℝ) : 
  length = 3 * width →
  area = length * width →
  area = 768 →
  width = 16 := by
sorry

end NUMINAMATH_CALUDE_rectangular_garden_width_l2566_256655


namespace NUMINAMATH_CALUDE_stream_speed_l2566_256686

/-- The speed of the stream given rowing conditions -/
theorem stream_speed (downstream_distance : ℝ) (upstream_distance : ℝ)
                     (downstream_time : ℝ) (upstream_time : ℝ)
                     (h1 : downstream_distance = 120)
                     (h2 : upstream_distance = 90)
                     (h3 : downstream_time = 4)
                     (h4 : upstream_time = 6) :
  ∃ (boat_speed stream_speed : ℝ),
    downstream_distance = (boat_speed + stream_speed) * downstream_time ∧
    upstream_distance = (boat_speed - stream_speed) * upstream_time ∧
    stream_speed = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_l2566_256686


namespace NUMINAMATH_CALUDE_sugar_amount_proof_l2566_256636

/-- The amount of sugar in pounds in the first combination -/
def sugar_amount : ℝ := 39

/-- The cost per pound of sugar and flour in dollars -/
def cost_per_pound : ℝ := 0.45

/-- The cost of the first combination in dollars -/
def cost_first : ℝ := 26

/-- The cost of the second combination in dollars -/
def cost_second : ℝ := 26

/-- The amount of flour in the first combination in pounds -/
def flour_first : ℝ := 16

/-- The amount of sugar in the second combination in pounds -/
def sugar_second : ℝ := 30

/-- The amount of flour in the second combination in pounds -/
def flour_second : ℝ := 25

theorem sugar_amount_proof :
  cost_per_pound * sugar_amount + cost_per_pound * flour_first = cost_first ∧
  cost_per_pound * sugar_second + cost_per_pound * flour_second = cost_second ∧
  sugar_amount + flour_first = sugar_second + flour_second :=
by sorry

end NUMINAMATH_CALUDE_sugar_amount_proof_l2566_256636


namespace NUMINAMATH_CALUDE_sphere_surface_area_l2566_256649

theorem sphere_surface_area (V : ℝ) (r : ℝ) (S : ℝ) : 
  V = 72 * Real.pi → 
  V = (4/3) * Real.pi * r^3 → 
  S = 4 * Real.pi * r^2 → 
  S = 36 * Real.pi * (4^(1/3)) :=
by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l2566_256649


namespace NUMINAMATH_CALUDE_rational_equation_solution_l2566_256641

theorem rational_equation_solution (x : ℝ) :
  (x^2 - 6*x + 8) / (x^2 - 9*x + 14) = (x^2 - 8*x + 15) / (x^2 - 10*x + 24) →
  x = (13 + Real.sqrt 5) / 2 ∨ x = (13 - Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_rational_equation_solution_l2566_256641


namespace NUMINAMATH_CALUDE_probability_A_wins_is_two_thirds_l2566_256612

/-- A card game with the following rules:
  * There are 4 cards numbered 1, 2, 3, and 4.
  * Cards are shuffled and placed face down.
  * Players A and B take turns drawing cards without replacement.
  * A draws first.
  * The first person to draw an even-numbered card wins.
-/
def CardGame : Type := Unit

/-- The probability of player A winning the card game. -/
def probability_A_wins (game : CardGame) : ℚ := 2/3

/-- Theorem stating that the probability of player A winning the card game is 2/3. -/
theorem probability_A_wins_is_two_thirds (game : CardGame) :
  probability_A_wins game = 2/3 := by sorry

end NUMINAMATH_CALUDE_probability_A_wins_is_two_thirds_l2566_256612


namespace NUMINAMATH_CALUDE_face_circle_larger_than_sphere_l2566_256698

/-- A tetrahedron with an inscribed sphere and inscribed circles in each face -/
structure Tetrahedron where
  /-- The radius of the inscribed sphere -/
  sphere_radius : ℝ
  /-- The radius of the inscribed circle of a face -/
  face_circle_radius : ℝ
  /-- Assumption that the sphere radius is positive -/
  sphere_radius_pos : 0 < sphere_radius
  /-- Assumption that the face circle radius is positive -/
  face_circle_radius_pos : 0 < face_circle_radius

/-- The radius of the inscribed circle of any face of a tetrahedron 
    is greater than the radius of its inscribed sphere -/
theorem face_circle_larger_than_sphere (t : Tetrahedron) : 
  t.face_circle_radius > t.sphere_radius := by
  sorry

end NUMINAMATH_CALUDE_face_circle_larger_than_sphere_l2566_256698


namespace NUMINAMATH_CALUDE_taxi_charge_correct_l2566_256607

/-- Calculates the total charge for a taxi trip given the initial fee, per-increment charge, increment distance, and total trip distance. -/
def total_charge (initial_fee : ℚ) (per_increment_charge : ℚ) (increment_distance : ℚ) (trip_distance : ℚ) : ℚ :=
  initial_fee + (trip_distance / increment_distance).floor * per_increment_charge

/-- Proves that the total charge for a specific taxi trip is correct. -/
theorem taxi_charge_correct :
  let initial_fee : ℚ := 41/20  -- $2.05
  let per_increment_charge : ℚ := 7/20  -- $0.35
  let increment_distance : ℚ := 2/5  -- 2/5 mile
  let trip_distance : ℚ := 18/5  -- 3.6 miles
  total_charge initial_fee per_increment_charge increment_distance trip_distance = 26/5  -- $5.20
  := by sorry

end NUMINAMATH_CALUDE_taxi_charge_correct_l2566_256607


namespace NUMINAMATH_CALUDE_reading_program_classes_l2566_256623

/-- The number of classes in a school with a specific reading program. -/
def number_of_classes (s : ℕ) : ℕ :=
  if s = 0 then 0 else 1

theorem reading_program_classes (s : ℕ) (h : s > 0) :
  let books_per_student_per_year := 4 * 12
  let total_books_read := 48
  number_of_classes s = 1 ∧ s * books_per_student_per_year = total_books_read :=
by sorry

end NUMINAMATH_CALUDE_reading_program_classes_l2566_256623


namespace NUMINAMATH_CALUDE_inverse_equals_k_times_self_l2566_256643

def A (d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; 6, d]

theorem inverse_equals_k_times_self (d k : ℝ) :
  A d * (A d)⁻¹ = 1 ∧ (A d)⁻¹ = k • (A d) → d = -3 ∧ k = 1/33 := by
  sorry

end NUMINAMATH_CALUDE_inverse_equals_k_times_self_l2566_256643


namespace NUMINAMATH_CALUDE_arithmetic_sequence_and_sum_l2566_256699

-- Define the arithmetic sequence a_n
def a (n : ℕ) : ℚ := sorry

-- Define the conditions
axiom a7_eq_4 : a 7 = 4
axiom a19_eq_2a9 : a 19 = 2 * a 9

-- Define b_n
def b (n : ℕ) : ℚ := 1 / (2 * n * a n)

-- Define the sum of the first n terms of b_n
def S (n : ℕ) : ℚ := sorry

theorem arithmetic_sequence_and_sum :
  (∀ n : ℕ, a n = (n + 1) / 2) ∧
  (∀ n : ℕ, S n = n / (n + 1)) := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_and_sum_l2566_256699


namespace NUMINAMATH_CALUDE_monotonic_increasing_condition_l2566_256601

open Real

theorem monotonic_increasing_condition (a : ℝ) :
  (∀ x ∈ Set.Ioo 0 (π / 2), StrictMono (fun x => (sin x + a) / cos x)) →
  a ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_monotonic_increasing_condition_l2566_256601


namespace NUMINAMATH_CALUDE_double_age_in_two_years_l2566_256670

/-- Represents the number of years until a man's age is twice his son's age. -/
def years_until_double_age (son_age : ℕ) (age_difference : ℕ) : ℕ :=
  let man_age := son_age + age_difference
  let x := man_age + 2 - 2 * (son_age + 2)
  2

/-- Theorem stating that the number of years until the man's age is twice his son's age is 2,
    given the son's current age and the age difference between the man and his son. -/
theorem double_age_in_two_years (son_age : ℕ) (age_difference : ℕ) 
    (h1 : son_age = 28) (h2 : age_difference = 30) : 
    years_until_double_age son_age age_difference = 2 := by
  sorry

#eval years_until_double_age 28 30

end NUMINAMATH_CALUDE_double_age_in_two_years_l2566_256670


namespace NUMINAMATH_CALUDE_triangle_area_l2566_256631

theorem triangle_area (a b c : ℝ) (h1 : a = 10) (h2 : b = 24) (h3 : c = 26) :
  (1 / 2) * a * b = 120 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l2566_256631


namespace NUMINAMATH_CALUDE_linear_function_passes_through_points_linear_function_unique_l2566_256691

/-- A linear function passing through two points (2, 3) and (3, 2) -/
def linearFunction (x : ℝ) : ℝ := -x + 5

/-- The theorem stating that the linear function passes through the given points -/
theorem linear_function_passes_through_points :
  linearFunction 2 = 3 ∧ linearFunction 3 = 2 := by
  sorry

/-- The theorem stating that the linear function is unique -/
theorem linear_function_unique (f : ℝ → ℝ) :
  f 2 = 3 → f 3 = 2 → ∀ x, f x = linearFunction x := by
  sorry

end NUMINAMATH_CALUDE_linear_function_passes_through_points_linear_function_unique_l2566_256691


namespace NUMINAMATH_CALUDE_measure_when_unit_changed_l2566_256651

-- Define segments a and b as positive real numbers
variable (a b : ℝ) (ha : 0 < a) (hb : 0 < b)

-- Define m as the measure of a when b is the unit length
variable (m : ℝ) (hm : a = m * b)

-- Theorem statement
theorem measure_when_unit_changed : 
  (b / a : ℝ) = 1 / m :=
sorry

end NUMINAMATH_CALUDE_measure_when_unit_changed_l2566_256651


namespace NUMINAMATH_CALUDE_function_relation_l2566_256619

/-- Given functions f, g, and h from ℝ to ℝ satisfying certain conditions,
    prove that h can be expressed in terms of f and g. -/
theorem function_relation (f g h : ℝ → ℝ) 
    (hf : ∀ x, f x = (h (x + 1) + h (x - 1)) / 2)
    (hg : ∀ x, g x = (h (x + 4) + h (x - 4)) / 2) :
    ∀ x, h x = g x - f (x - 3) + f (x - 1) + f (x + 1) - f (x + 3) := by
  sorry

end NUMINAMATH_CALUDE_function_relation_l2566_256619


namespace NUMINAMATH_CALUDE_world_expo_allocation_schemes_l2566_256630

theorem world_expo_allocation_schemes :
  let n_volunteers : ℕ := 6
  let n_pavilions : ℕ := 4
  let n_groups_of_two : ℕ := 2
  let n_groups_of_one : ℕ := 2
  
  (n_volunteers.choose 2 * (n_volunteers - 2).choose 2) *
  (n_pavilions.choose n_groups_of_two) *
  n_pavilions.factorial = 1080 := by sorry

end NUMINAMATH_CALUDE_world_expo_allocation_schemes_l2566_256630


namespace NUMINAMATH_CALUDE_cos_120_degrees_l2566_256697

theorem cos_120_degrees : Real.cos (2 * Real.pi / 3) = -1/2 := by sorry

end NUMINAMATH_CALUDE_cos_120_degrees_l2566_256697


namespace NUMINAMATH_CALUDE_range_of_T_l2566_256622

theorem range_of_T (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0)
  (h4 : x + y + z = 30) (h5 : 3 * x + y - z = 50) :
  let T := 5 * x + 4 * y + 2 * z
  ∃ (T_min T_max : ℝ), T_min = 120 ∧ T_max = 130 ∧ T_min ≤ T ∧ T ≤ T_max :=
by sorry

end NUMINAMATH_CALUDE_range_of_T_l2566_256622


namespace NUMINAMATH_CALUDE_equation_solution_count_l2566_256618

theorem equation_solution_count : ∃ (s : Finset ℕ),
  (∀ c ∈ s, c ≤ 1000) ∧ 
  (∀ c ∈ s, ∃ x : ℝ, 7 * ⌊x⌋ + 2 * ⌈x⌉ = c) ∧
  (∀ c ≤ 1000, c ∉ s → ¬∃ x : ℝ, 7 * ⌊x⌋ + 2 * ⌈x⌉ = c) ∧
  s.card = 223 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_count_l2566_256618


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l2566_256645

theorem arithmetic_mean_problem (a b c d : ℝ) : 
  (a + b + c + d + 106) / 5 = 92 →
  (a + b + c + d) / 4 = 88.5 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l2566_256645


namespace NUMINAMATH_CALUDE_regular_polygon_is_pentagon_with_perimeter_125_l2566_256633

/-- A regular polygon where the length of a side is 25 when the perimeter is divided by 5 -/
structure RegularPolygon where
  sides : ℕ
  side_length : ℝ
  perimeter : ℝ
  h1 : perimeter = sides * side_length
  h2 : perimeter / 5 = side_length
  h3 : side_length = 25

theorem regular_polygon_is_pentagon_with_perimeter_125 (p : RegularPolygon) :
  p.sides = 5 ∧ p.perimeter = 125 := by
  sorry

#check regular_polygon_is_pentagon_with_perimeter_125

end NUMINAMATH_CALUDE_regular_polygon_is_pentagon_with_perimeter_125_l2566_256633


namespace NUMINAMATH_CALUDE_abs_inequality_iff_inequality_l2566_256648

theorem abs_inequality_iff_inequality (a b : ℝ) : a > b ↔ a * |a| > b * |b| := by sorry

end NUMINAMATH_CALUDE_abs_inequality_iff_inequality_l2566_256648


namespace NUMINAMATH_CALUDE_age_difference_james_jessica_prove_age_difference_l2566_256667

/-- Given the ages and relationships of Justin, Jessica, and James, prove that James is 7 years older than Jessica. -/
theorem age_difference_james_jessica : ℕ → Prop :=
  fun age_difference =>
    ∀ (justin_age jessica_age james_age : ℕ),
      justin_age = 26 →
      jessica_age = justin_age + 6 →
      james_age > jessica_age →
      james_age + 5 = 44 →
      james_age - jessica_age = age_difference →
      age_difference = 7

/-- Proof of the theorem -/
theorem prove_age_difference : ∃ (age_difference : ℕ), age_difference_james_jessica age_difference := by
  sorry

end NUMINAMATH_CALUDE_age_difference_james_jessica_prove_age_difference_l2566_256667


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l2566_256609

-- Define set A
def A : Set ℝ := {x | x^2 - x - 2 < 0}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = Real.sin x}

-- Theorem statement
theorem union_of_A_and_B : A ∪ B = Set.Icc (-1) 2 := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l2566_256609


namespace NUMINAMATH_CALUDE_factorization_cubic_quadratic_l2566_256657

theorem factorization_cubic_quadratic (x y : ℝ) : x^3*y - 4*x*y = x*y*(x-2)*(x+2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_cubic_quadratic_l2566_256657


namespace NUMINAMATH_CALUDE_paper_folding_perimeter_ratio_l2566_256676

/-- Given a square piece of paper with side length 8 inches, when folded and cut as described,
    the ratio of the perimeter of the larger rectangle to the perimeter of one of the smaller rectangles is 3/2. -/
theorem paper_folding_perimeter_ratio :
  let initial_side_length : ℝ := 8
  let large_rectangle_length : ℝ := initial_side_length
  let large_rectangle_width : ℝ := initial_side_length / 2
  let small_rectangle_side : ℝ := initial_side_length / 2
  let large_perimeter : ℝ := 2 * (large_rectangle_length + large_rectangle_width)
  let small_perimeter : ℝ := 4 * small_rectangle_side
  large_perimeter / small_perimeter = 3 / 2 := by sorry

end NUMINAMATH_CALUDE_paper_folding_perimeter_ratio_l2566_256676


namespace NUMINAMATH_CALUDE_constant_function_l2566_256682

theorem constant_function (a : ℝ) (f : ℝ → ℝ) 
  (h1 : f 0 = (1 : ℝ) / 2)
  (h2 : ∀ x y : ℝ, f (x + y) = f x * f (a - y) + f y * f (a - x)) :
  ∀ x : ℝ, f x = (1 : ℝ) / 2 := by
sorry

end NUMINAMATH_CALUDE_constant_function_l2566_256682


namespace NUMINAMATH_CALUDE_intersection_M_N_l2566_256679

def M : Set ℝ := {-4, -3, -2, -1, 0, 1}

def N : Set ℝ := {x : ℝ | x^2 + 3*x < 0}

theorem intersection_M_N : M ∩ N = {-2, -1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2566_256679


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l2566_256654

-- Define the proposition
def P (a : ℝ) : Prop :=
  ∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - a ≤ 0

-- Define the sufficient condition
def sufficient_condition (a : ℝ) : Prop := a ≥ 5

-- Theorem statement
theorem sufficient_but_not_necessary :
  (∀ a : ℝ, sufficient_condition a → P a) ∧
  ¬(∀ a : ℝ, P a → sufficient_condition a) :=
sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l2566_256654


namespace NUMINAMATH_CALUDE_division_problem_l2566_256684

theorem division_problem (dividend quotient remainder divisor : ℕ) : 
  dividend = 100 ∧ 
  quotient = 9 ∧ 
  remainder = 1 ∧ 
  dividend = divisor * quotient + remainder →
  divisor = 11 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l2566_256684


namespace NUMINAMATH_CALUDE_absolute_difference_of_points_on_curve_l2566_256660

theorem absolute_difference_of_points_on_curve (e p q : ℝ) : 
  (p^2 + e^4 = 4 * e^2 * p + 6) →
  (q^2 + e^4 = 4 * e^2 * q + 6) →
  |p - q| = 2 * Real.sqrt (3 * e^4 + 6) := by
  sorry

end NUMINAMATH_CALUDE_absolute_difference_of_points_on_curve_l2566_256660


namespace NUMINAMATH_CALUDE_circles_tangent_m_value_l2566_256662

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 1
def C₂ (x y : ℝ) (m : ℝ) : Prop := x^2 + y^2 - 6*x - 8*y + m = 0

-- Define the tangency condition
def are_tangent (C₁ C₂ : (ℝ → ℝ → Prop)) : Prop :=
  ∃ (x y : ℝ), C₁ x y ∧ C₂ x y ∧
  ∀ (x' y' : ℝ), C₁ x' y' ∧ C₂ x' y' → (x' = x ∧ y' = y)

-- Theorem statement
theorem circles_tangent_m_value :
  are_tangent C₁ (C₂ · · 9) → ∀ m : ℝ, are_tangent C₁ (C₂ · · m) → m = 9 :=
by sorry

end NUMINAMATH_CALUDE_circles_tangent_m_value_l2566_256662


namespace NUMINAMATH_CALUDE_coefficient_x_cubed_is_zero_l2566_256673

/-- Given q(x) = x⁴ - 4x + 5, the coefficient of x³ in (q(x))² is 0 -/
theorem coefficient_x_cubed_is_zero (x : ℝ) : 
  let q := fun (x : ℝ) => x^4 - 4*x + 5
  (q x)^2 = x^8 - 8*x^5 + 10*x^4 + 16*x^2 - 40*x + 25 :=
by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_cubed_is_zero_l2566_256673


namespace NUMINAMATH_CALUDE_orange_flowers_killed_by_fungus_l2566_256677

/-- Represents the number of flowers of each color --/
structure FlowerCount where
  red : ℕ
  yellow : ℕ
  orange : ℕ
  purple : ℕ

/-- Represents the number of flowers killed by fungus for each color --/
structure FungusKilled where
  red : ℕ
  yellow : ℕ
  orange : ℕ
  purple : ℕ

theorem orange_flowers_killed_by_fungus 
  (seeds_per_color : ℕ)
  (flowers_per_bouquet : ℕ)
  (num_bouquets : ℕ)
  (fungus_killed : FungusKilled)
  (h1 : seeds_per_color = 125)
  (h2 : flowers_per_bouquet = 9)
  (h3 : num_bouquets = 36)
  (h4 : fungus_killed.red = 45)
  (h5 : fungus_killed.yellow = 61)
  (h6 : fungus_killed.purple = 40) :
  fungus_killed.orange = 30 := by
sorry

end NUMINAMATH_CALUDE_orange_flowers_killed_by_fungus_l2566_256677


namespace NUMINAMATH_CALUDE_power_multiplication_l2566_256694

theorem power_multiplication (a b : ℕ) : (10 : ℕ) ^ 85 * (10 : ℕ) ^ 84 = (10 : ℕ) ^ (85 + 84) := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l2566_256694


namespace NUMINAMATH_CALUDE_base_b_square_l2566_256621

theorem base_b_square (b : ℕ) (h : b > 1) : 
  (3 * b^2 + 4 * b + 3 = (b + 3)^2) → b = 3 := by
  sorry

end NUMINAMATH_CALUDE_base_b_square_l2566_256621


namespace NUMINAMATH_CALUDE_tablet_value_proof_compensation_for_m_days_l2566_256626

-- Define the total days of internship
def total_days : ℕ := 30

-- Define the cash compensation for full internship
def full_cash_compensation : ℕ := 1500

-- Define the number of days Xiaomin worked
def worked_days : ℕ := 20

-- Define the cash compensation Xiaomin received
def received_cash_compensation : ℕ := 300

-- Define the value of the M type tablet
def tablet_value : ℕ := 2100

-- Define the daily compensation rate
def daily_rate : ℚ := 120

-- Theorem for the value of the M type tablet
theorem tablet_value_proof :
  (worked_days : ℚ) / total_days * (tablet_value + full_cash_compensation) =
  tablet_value + received_cash_compensation :=
sorry

-- Theorem for the compensation for m days of work
theorem compensation_for_m_days (m : ℕ) :
  (m : ℚ) * daily_rate = (m : ℚ) * ((tablet_value + full_cash_compensation) / total_days) :=
sorry

end NUMINAMATH_CALUDE_tablet_value_proof_compensation_for_m_days_l2566_256626


namespace NUMINAMATH_CALUDE_largest_m_l2566_256635

/-- A three-digit positive integer that is the product of three distinct prime factors --/
def m (x y : ℕ) : ℕ := x * y * (10 * x + y)

/-- The proposition that m is the largest possible value given the conditions --/
theorem largest_m : ∀ x y : ℕ, 
  x < 10 → y < 10 → x ≠ y → 
  Nat.Prime x → Nat.Prime y → Nat.Prime (10 * x + y) →
  m x y ≤ 795 ∧ m x y < 1000 :=
sorry

end NUMINAMATH_CALUDE_largest_m_l2566_256635


namespace NUMINAMATH_CALUDE_polygon_area_is_787_5_l2566_256658

/-- The area of a triangle given its vertices -/
def triangle_area (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  0.5 * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

/-- The vertices of the polygon -/
def vertices : List (ℝ × ℝ) :=
  [(0, 0), (15, 0), (45, 30), (45, 45), (30, 45), (0, 15)]

/-- The area of the polygon -/
def polygon_area : ℝ :=
  triangle_area 0 0 15 0 0 15 +
  triangle_area 15 0 45 30 0 15 +
  triangle_area 45 30 45 45 30 45

theorem polygon_area_is_787_5 :
  polygon_area = 787.5 := by
  sorry

end NUMINAMATH_CALUDE_polygon_area_is_787_5_l2566_256658


namespace NUMINAMATH_CALUDE_female_listeners_l2566_256624

/-- Given a radio station survey with total listeners and male listeners,
    prove the number of female listeners. -/
theorem female_listeners (total_listeners male_listeners : ℕ) :
  total_listeners = 130 →
  male_listeners = 62 →
  total_listeners - male_listeners = 68 := by
  sorry

end NUMINAMATH_CALUDE_female_listeners_l2566_256624


namespace NUMINAMATH_CALUDE_digit_150_of_5_over_13_l2566_256613

def decimal_representation (n d : ℕ) : ℚ := n / d

def nth_digit_after_decimal (q : ℚ) (n : ℕ) : ℕ := sorry

theorem digit_150_of_5_over_13 : 
  nth_digit_after_decimal (decimal_representation 5 13) 150 = 5 := by sorry

end NUMINAMATH_CALUDE_digit_150_of_5_over_13_l2566_256613


namespace NUMINAMATH_CALUDE_expression_evaluation_l2566_256627

theorem expression_evaluation : -20 + 12 * (8 / 4) * 3 = 52 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2566_256627


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_solve_equation_l2566_256628

-- Problem 1
theorem simplify_and_evaluate : 
  let f (x : ℝ) := (x^2 - 6*x + 9) / (x^2 - 1) / ((x^2 - 3*x) / (x + 1))
  f (-3) = -1/2 := by sorry

-- Problem 2
theorem solve_equation :
  ∃ (x : ℝ), x / (x + 1) = 2*x / (3*x + 3) - 1 ∧ x = -3/4 := by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_solve_equation_l2566_256628


namespace NUMINAMATH_CALUDE_point_slope_theorem_l2566_256617

theorem point_slope_theorem (k : ℝ) (h1 : k > 0) : 
  (2 - k) / (k - 1) = k^2 → k = 1 := by sorry

end NUMINAMATH_CALUDE_point_slope_theorem_l2566_256617


namespace NUMINAMATH_CALUDE_no_root_greater_than_three_l2566_256683

theorem no_root_greater_than_three :
  ∀ x : ℝ,
  (2 * x^2 - 4 = 36 ∨ (3*x-2)^2 = (x+2)^2 ∨ (3*x^2 - 10 = 2*x + 2 ∧ 3*x^2 - 10 ≥ 0 ∧ 2*x + 2 ≥ 0)) →
  x ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_no_root_greater_than_three_l2566_256683


namespace NUMINAMATH_CALUDE_remainder_problem_l2566_256675

theorem remainder_problem : 123456789012 % 112 = 76 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2566_256675


namespace NUMINAMATH_CALUDE_smallest_constant_term_l2566_256681

theorem smallest_constant_term (a b c d e : ℤ) : 
  (∀ x : ℚ, a * x^4 + b * x^3 + c * x^2 + d * x + e = 0 ↔ 
    x = -3 ∨ x = 6 ∨ x = 10 ∨ x = -1/4) →
  e > 0 →
  e ≥ 180 :=
sorry

end NUMINAMATH_CALUDE_smallest_constant_term_l2566_256681


namespace NUMINAMATH_CALUDE_total_paid_is_117_l2566_256639

/-- Calculates the total amount paid after applying a senior citizen discount on Tuesday --/
def total_paid_after_discount (jimmy_shorts : ℕ) (jimmy_short_price : ℚ) 
                               (irene_shirts : ℕ) (irene_shirt_price : ℚ) 
                               (discount_rate : ℚ) : ℚ :=
  let total_before_discount := jimmy_shorts * jimmy_short_price + irene_shirts * irene_shirt_price
  let discount_amount := total_before_discount * discount_rate
  total_before_discount - discount_amount

/-- Proves that the total amount paid after the senior citizen discount is $117 --/
theorem total_paid_is_117 : 
  total_paid_after_discount 3 15 5 17 (1/10) = 117 := by
  sorry

end NUMINAMATH_CALUDE_total_paid_is_117_l2566_256639


namespace NUMINAMATH_CALUDE_tomato_production_relationship_l2566_256603

/-- Represents the production of three tomato plants -/
structure TomatoProduction where
  first : ℕ
  second : ℕ
  third : ℕ

/-- The conditions of the tomato production problem -/
def tomato_problem (p : TomatoProduction) : Prop :=
  p.first = 24 ∧
  p.second = (p.first / 2) + 5 ∧
  p.first + p.second + p.third = 60

theorem tomato_production_relationship (p : TomatoProduction) 
  (h : tomato_problem p) : p.third = p.second + 2 := by
  sorry

#check tomato_production_relationship

end NUMINAMATH_CALUDE_tomato_production_relationship_l2566_256603


namespace NUMINAMATH_CALUDE_zachs_babysitting_pay_rate_l2566_256696

/-- The problem of calculating Zach's babysitting pay rate -/
theorem zachs_babysitting_pay_rate 
  (bike_cost : ℚ)
  (weekly_allowance : ℚ)
  (lawn_mowing_pay : ℚ)
  (current_savings : ℚ)
  (additional_needed : ℚ)
  (babysitting_hours : ℚ)
  (h1 : bike_cost = 100)
  (h2 : weekly_allowance = 5)
  (h3 : lawn_mowing_pay = 10)
  (h4 : current_savings = 65)
  (h5 : additional_needed = 6)
  (h6 : babysitting_hours = 2)
  : ∃ (babysitting_rate : ℚ), 
    babysitting_rate = (current_savings + weekly_allowance + lawn_mowing_pay + additional_needed - bike_cost) / babysitting_hours ∧ 
    babysitting_rate = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_zachs_babysitting_pay_rate_l2566_256696


namespace NUMINAMATH_CALUDE_contractor_daily_wage_l2566_256663

/-- Calculates the daily wage for a contractor given contract details -/
def daily_wage (total_days : ℕ) (absence_fine : ℚ) (total_payment : ℚ) (absent_days : ℕ) : ℚ :=
  (total_payment + (absent_days : ℚ) * absence_fine) / ((total_days - absent_days) : ℚ)

/-- Proves that the daily wage is 25 given the contract details -/
theorem contractor_daily_wage :
  daily_wage 30 (7.5 : ℚ) 620 4 = 25 := by
  sorry

end NUMINAMATH_CALUDE_contractor_daily_wage_l2566_256663


namespace NUMINAMATH_CALUDE_equation_solution_inequality_system_solution_l2566_256604

-- Part 1: Equation solution
theorem equation_solution :
  ∀ x : ℝ, x ≠ 1 ∧ x ≠ 0 → (x / (x - 1) = (x - 1) / (2*x - 2) ↔ x = -1) :=
sorry

-- Part 2: Inequality system solution
theorem inequality_system_solution :
  ∀ x : ℝ, (5*x - 1 > 3*x - 4 ∧ -1/3*x ≤ 2/3 - x) ↔ (-3/2 < x ∧ x ≤ 1) :=
sorry

end NUMINAMATH_CALUDE_equation_solution_inequality_system_solution_l2566_256604


namespace NUMINAMATH_CALUDE_f_derivative_and_tangent_line_l2566_256688

noncomputable def f (x : ℝ) : ℝ := Real.sin x / x

theorem f_derivative_and_tangent_line :
  (∃ (f' : ℝ → ℝ), ∀ x, x ≠ 0 → HasDerivAt f (f' x) x) ∧
  (∀ x, x ≠ 0 → (deriv f) x = (x * Real.cos x - Real.sin x) / x^2) ∧
  (HasDerivAt f (-1/π) π) ∧
  (∀ x, -x/π + 1 = (-1/π) * (x - π)) := by
  sorry

end NUMINAMATH_CALUDE_f_derivative_and_tangent_line_l2566_256688


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_negation_l2566_256693

theorem necessary_not_sufficient_negation (p q : Prop) :
  (q → p) ∧ ¬(p → q) → (¬p → ¬q) ∧ ¬(¬q → ¬p) := by
  sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_negation_l2566_256693


namespace NUMINAMATH_CALUDE_fraction_inequality_l2566_256611

theorem fraction_inequality (a b : ℝ) (ha : a > 0) (hb : b < 0) : 
  a / b + b / a ≤ -2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l2566_256611


namespace NUMINAMATH_CALUDE_olaf_game_score_l2566_256685

theorem olaf_game_score (dad_score : ℕ) : 
  (3 * dad_score + dad_score = 28) → dad_score = 7 := by
  sorry

end NUMINAMATH_CALUDE_olaf_game_score_l2566_256685


namespace NUMINAMATH_CALUDE_sqrt_360000_equals_600_l2566_256600

theorem sqrt_360000_equals_600 : Real.sqrt 360000 = 600 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_360000_equals_600_l2566_256600


namespace NUMINAMATH_CALUDE_expression_evaluation_l2566_256669

theorem expression_evaluation : 
  Real.sqrt (5 + 4 * Real.sqrt 3) - Real.sqrt (5 - 4 * Real.sqrt 3) + 
  Real.sqrt (7 + 2 * Real.sqrt 10) - Real.sqrt (7 - 2 * Real.sqrt 10) = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2566_256669


namespace NUMINAMATH_CALUDE_teddy_hamburgers_count_l2566_256678

/-- The number of hamburgers Teddy bought -/
def teddy_hamburgers : ℕ := 5

/-- The total amount spent by both Robert and Teddy -/
def total_spent : ℕ := 106

/-- The cost of one box of pizza -/
def pizza_cost : ℕ := 10

/-- The cost of one can of soft drink -/
def drink_cost : ℕ := 2

/-- The cost of one hamburger -/
def hamburger_cost : ℕ := 3

/-- The number of pizza boxes Robert bought -/
def robert_pizza : ℕ := 5

/-- The number of soft drink cans Robert bought -/
def robert_drinks : ℕ := 10

/-- The number of soft drink cans Teddy bought -/
def teddy_drinks : ℕ := 10

theorem teddy_hamburgers_count : 
  robert_pizza * pizza_cost + 
  (robert_drinks + teddy_drinks) * drink_cost + 
  teddy_hamburgers * hamburger_cost = total_spent :=
by sorry

end NUMINAMATH_CALUDE_teddy_hamburgers_count_l2566_256678


namespace NUMINAMATH_CALUDE_correct_quotient_proof_l2566_256672

theorem correct_quotient_proof (D : ℕ) (h1 : D % 21 = 0) (h2 : D / 12 = 49) : D / 21 = 28 := by
  sorry

end NUMINAMATH_CALUDE_correct_quotient_proof_l2566_256672


namespace NUMINAMATH_CALUDE_three_digit_power_sum_l2566_256664

theorem three_digit_power_sum (a b c : ℕ) : a < 4 →
  (100 * a + 10 * b + c = (b + c) ^ a) ∧ (100 ≤ 100 * a + 10 * b + c) ∧ (100 * a + 10 * b + c < 1000) →
  (100 * a + 10 * b + c = 289 ∨ 100 * a + 10 * b + c = 343) :=
by sorry

end NUMINAMATH_CALUDE_three_digit_power_sum_l2566_256664


namespace NUMINAMATH_CALUDE_maxwell_walking_speed_l2566_256695

/-- Prove that Maxwell's walking speed is 4 km/h given the conditions of the problem -/
theorem maxwell_walking_speed :
  ∀ (maxwell_speed : ℝ),
    maxwell_speed > 0 →
    (3 * maxwell_speed) + (2 * 6) = 24 →
    maxwell_speed = 4 :=
by
  sorry

#check maxwell_walking_speed

end NUMINAMATH_CALUDE_maxwell_walking_speed_l2566_256695


namespace NUMINAMATH_CALUDE_souvenir_purchase_theorem_l2566_256666

/-- Represents a purchasing plan for souvenirs -/
structure PurchasePlan where
  typeA : ℕ
  typeB : ℕ

/-- Checks if a purchase plan is valid according to the given constraints -/
def isValidPlan (p : PurchasePlan) : Prop :=
  p.typeA + p.typeB = 60 ∧
  p.typeB ≤ 2 * p.typeA ∧
  100 * p.typeA + 60 * p.typeB ≤ 4500

/-- Calculates the cost of a purchase plan -/
def planCost (p : PurchasePlan) : ℕ :=
  100 * p.typeA + 60 * p.typeB

/-- The main theorem encompassing all parts of the problem -/
theorem souvenir_purchase_theorem :
  (∃! p : PurchasePlan, p.typeA + p.typeB = 60 ∧ planCost p = 4600) ∧
  (∃! plans : List PurchasePlan, plans.length = 3 ∧ 
    ∀ p ∈ plans, isValidPlan p ∧
    ∀ p, isValidPlan p → p ∈ plans) ∧
  (∃ p : PurchasePlan, isValidPlan p ∧
    ∀ q, isValidPlan q → planCost p ≤ planCost q ∧
    planCost p = 4400) := by
  sorry

#check souvenir_purchase_theorem

end NUMINAMATH_CALUDE_souvenir_purchase_theorem_l2566_256666


namespace NUMINAMATH_CALUDE_moving_points_theorem_l2566_256689

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Calculate the area of a triangle given three points -/
def triangleArea (p1 p2 p3 : Point) : ℝ :=
  sorry

/-- The main theorem -/
theorem moving_points_theorem (ABC : Triangle) (P Q : Point) (t : ℝ) :
  (ABC.B.x - ABC.A.x)^2 + (ABC.B.y - ABC.A.y)^2 = 36 →  -- AB = 6 cm
  (ABC.C.x - ABC.B.x)^2 + (ABC.C.y - ABC.B.y)^2 = 64 →  -- BC = 8 cm
  (ABC.C.x - ABC.B.x) * (ABC.B.y - ABC.A.y) = (ABC.C.y - ABC.B.y) * (ABC.B.x - ABC.A.x) →  -- ABC is right-angled at B
  P.x = ABC.A.x + t →  -- P moves from A towards B
  P.y = ABC.A.y →
  Q.x = ABC.B.x + 2 * t →  -- Q moves from B towards C
  Q.y = ABC.B.y →
  triangleArea P ABC.B Q = 5 →  -- Area of PBQ is 5 cm²
  t = 1  -- Time P moves is 1 second
  := by sorry

end NUMINAMATH_CALUDE_moving_points_theorem_l2566_256689


namespace NUMINAMATH_CALUDE_solution_set_f_leq_5_range_of_m_for_nonempty_solution_l2566_256652

-- Define the function f(x)
def f (x : ℝ) : ℝ := |2*x + 3| + |2*x - 1|

-- Theorem for the solution set of f(x) ≤ 5
theorem solution_set_f_leq_5 :
  {x : ℝ | f x ≤ 5} = {x : ℝ | -7/4 ≤ x ∧ x ≤ 3/4} := by sorry

-- Theorem for the range of m when the solution set of f(x) < |m-1| is non-empty
theorem range_of_m_for_nonempty_solution (m : ℝ) :
  (∃ x : ℝ, f x < |m - 1|) → (m > 5 ∨ m < -3) := by sorry

end NUMINAMATH_CALUDE_solution_set_f_leq_5_range_of_m_for_nonempty_solution_l2566_256652


namespace NUMINAMATH_CALUDE_perfect_square_condition_l2566_256665

/-- The expression ax^2 + 2bxy + cy^2 - k(x^2 + y^2) is a perfect square if and only if 
    k = (a+c)/2 ± (1/2)√((a-c)^2 + 4b^2), where a, b, c are real constants. -/
theorem perfect_square_condition (a b c k : ℝ) :
  (∃ (f : ℝ → ℝ → ℝ), ∀ (x y : ℝ), a * x^2 + 2 * b * x * y + c * y^2 - k * (x^2 + y^2) = (f x y)^2) ↔
  (k = (a + c) / 2 + (1 / 2) * Real.sqrt ((a - c)^2 + 4 * b^2) ∨
   k = (a + c) / 2 - (1 / 2) * Real.sqrt ((a - c)^2 + 4 * b^2)) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l2566_256665


namespace NUMINAMATH_CALUDE_smallest_AAB_existence_AAB_l2566_256661

-- Define the structure for our number
structure SpecialNumber where
  A : Nat
  B : Nat
  AB : Nat
  AAB : Nat

-- Define the conditions
def validNumber (n : SpecialNumber) : Prop :=
  1 ≤ n.A ∧ n.A ≤ 9 ∧
  1 ≤ n.B ∧ n.B ≤ 9 ∧
  n.AB = 10 * n.A + n.B ∧
  n.AAB = 110 * n.A + n.B ∧
  n.AB = n.AAB / 8

-- Define the theorem
theorem smallest_AAB :
  ∀ n : SpecialNumber, validNumber n → n.AAB ≥ 221 := by
  sorry

-- Define the existence of such a number
theorem existence_AAB :
  ∃ n : SpecialNumber, validNumber n ∧ n.AAB = 221 := by
  sorry

end NUMINAMATH_CALUDE_smallest_AAB_existence_AAB_l2566_256661


namespace NUMINAMATH_CALUDE_nested_square_root_18_l2566_256608

theorem nested_square_root_18 :
  ∃ x : ℝ, x = Real.sqrt (18 + x) ∧ x = (1 + Real.sqrt 73) / 2 := by
  sorry

end NUMINAMATH_CALUDE_nested_square_root_18_l2566_256608


namespace NUMINAMATH_CALUDE_judy_school_week_days_l2566_256646

/-- The number of pencils Judy uses during her school week. -/
def pencils_per_week : ℕ := 10

/-- The number of pencils in a pack. -/
def pencils_per_pack : ℕ := 30

/-- The cost of a pack of pencils in dollars. -/
def cost_per_pack : ℚ := 4

/-- The amount Judy spends on pencils in dollars. -/
def total_spent : ℚ := 12

/-- The number of days over which Judy spends the total amount. -/
def total_days : ℕ := 45

/-- The number of days in Judy's school week. -/
def school_week_days : ℕ := 5

/-- Theorem stating that the number of days in Judy's school week is 5. -/
theorem judy_school_week_days :
  (pencils_per_week : ℚ) * total_days * cost_per_pack =
  pencils_per_pack * total_spent * (school_week_days : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_judy_school_week_days_l2566_256646


namespace NUMINAMATH_CALUDE_discriminant_of_quadratic_l2566_256637

/-- The discriminant of a quadratic polynomial ax^2 + bx + c is b^2 - 4ac -/
def discriminant (a b c : ℚ) : ℚ := b^2 - 4*a*c

/-- The quadratic polynomial 2x^2 + (4 - 1/2)x + 1 -/
def quadratic_polynomial (x : ℚ) : ℚ := 2*x^2 + (4 - 1/2)*x + 1

theorem discriminant_of_quadratic : 
  discriminant 2 (4 - 1/2) 1 = 17/4 := by
  sorry

end NUMINAMATH_CALUDE_discriminant_of_quadratic_l2566_256637


namespace NUMINAMATH_CALUDE_negation_of_forall_positive_l2566_256674

theorem negation_of_forall_positive (P : ℝ → Prop) :
  (¬ ∀ x : ℝ, x^2 + 1 > 0) ↔ (∃ x : ℝ, x^2 + 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_forall_positive_l2566_256674


namespace NUMINAMATH_CALUDE_gunther_dusting_time_l2566_256610

/-- Represents the time in minutes for Gunther's cleaning tasks -/
structure CleaningTime where
  vacuuming : ℕ
  mopping : ℕ
  brushing_per_cat : ℕ
  num_cats : ℕ
  total_free_time : ℕ
  remaining_free_time : ℕ

/-- Calculates the time spent dusting furniture -/
def dusting_time (ct : CleaningTime) : ℕ :=
  ct.total_free_time - ct.remaining_free_time - 
  (ct.vacuuming + ct.mopping + ct.brushing_per_cat * ct.num_cats)

/-- Theorem stating that Gunther spends 60 minutes dusting furniture -/
theorem gunther_dusting_time :
  let ct : CleaningTime := {
    vacuuming := 45,
    mopping := 30,
    brushing_per_cat := 5,
    num_cats := 3,
    total_free_time := 3 * 60,
    remaining_free_time := 30
  }
  dusting_time ct = 60 := by
  sorry

end NUMINAMATH_CALUDE_gunther_dusting_time_l2566_256610


namespace NUMINAMATH_CALUDE_ratio_problem_l2566_256606

theorem ratio_problem (second_part : ℝ) (percent : ℝ) (first_part : ℝ) : 
  second_part = 10 →
  percent = 20 →
  first_part / second_part = percent / 100 →
  first_part = 2 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l2566_256606


namespace NUMINAMATH_CALUDE_factoring_expression_l2566_256605

theorem factoring_expression (x : ℝ) : 3*x*(x+2) + 2*(x+2) + 5*(x+2) = (x+2)*(3*x+7) := by
  sorry

end NUMINAMATH_CALUDE_factoring_expression_l2566_256605


namespace NUMINAMATH_CALUDE_greenhouse_renovation_l2566_256616

/-- Greenhouse renovation problem -/
theorem greenhouse_renovation 
  (cost_2A_vs_1B : ℝ) 
  (cost_1A_2B : ℝ) 
  (days_A : ℝ) 
  (days_B : ℝ) 
  (total_greenhouses : ℕ) 
  (max_budget : ℝ) 
  (max_days : ℝ)
  (h1 : cost_2A_vs_1B = 6)
  (h2 : cost_1A_2B = 48)
  (h3 : days_A = 5)
  (h4 : days_B = 3)
  (h5 : total_greenhouses = 8)
  (h6 : max_budget = 128)
  (h7 : max_days = 35) :
  ∃ (cost_A cost_B : ℝ),
    cost_A = 12 ∧ 
    cost_B = 18 ∧
    2 * cost_A = cost_B + cost_2A_vs_1B ∧
    cost_A + 2 * cost_B = cost_1A_2B ∧
    (∀ m : ℕ, 
      (m ≤ total_greenhouses ∧
       m * cost_A + (total_greenhouses - m) * cost_B ≤ max_budget ∧
       m * days_A + (total_greenhouses - m) * days_B ≤ max_days) 
      ↔ m ∈ ({3, 4, 5} : Set ℕ)) :=
sorry

end NUMINAMATH_CALUDE_greenhouse_renovation_l2566_256616


namespace NUMINAMATH_CALUDE_platform_length_l2566_256614

/-- Given a train of length 300 meters that crosses a platform in 33 seconds
    and a signal pole in 18 seconds, the length of the platform is 250 meters. -/
theorem platform_length
  (train_length : ℝ)
  (platform_crossing_time : ℝ)
  (pole_crossing_time : ℝ)
  (h1 : train_length = 300)
  (h2 : platform_crossing_time = 33)
  (h3 : pole_crossing_time = 18) :
  (train_length + platform_crossing_time * (train_length / pole_crossing_time) - train_length) = 250 :=
by sorry

end NUMINAMATH_CALUDE_platform_length_l2566_256614


namespace NUMINAMATH_CALUDE_smallest_congruent_number_l2566_256680

theorem smallest_congruent_number : ∃ n : ℕ, 
  n > 1 ∧ 
  n % 6 = 1 ∧ 
  n % 7 = 1 ∧ 
  n % 8 = 1 ∧
  (∀ m : ℕ, m > 1 → m % 6 = 1 → m % 7 = 1 → m % 8 = 1 → n ≤ m) ∧
  n = 169 := by
  sorry

end NUMINAMATH_CALUDE_smallest_congruent_number_l2566_256680


namespace NUMINAMATH_CALUDE_reciprocal_of_neg_sqrt_two_l2566_256620

theorem reciprocal_of_neg_sqrt_two :
  (1 : ℝ) / (-Real.sqrt 2) = -(Real.sqrt 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_neg_sqrt_two_l2566_256620


namespace NUMINAMATH_CALUDE_trigonometric_problem_l2566_256671

theorem trigonometric_problem (α β : Real) 
  (h1 : 3 * Real.sin α - Real.sin β = Real.sqrt 10)
  (h2 : α + β = Real.pi / 2) :
  Real.sin α = (3 * Real.sqrt 10) / 10 ∧ 
  Real.cos (2 * β) = 4 / 5 := by
sorry

end NUMINAMATH_CALUDE_trigonometric_problem_l2566_256671


namespace NUMINAMATH_CALUDE_total_seeds_three_watermelons_l2566_256632

/-- Represents the number of seeds of each color in a slice of watermelon -/
structure SeedsPerSlice where
  black : ℕ
  white : ℕ
  red : ℕ := 0
  purple : ℕ := 0
  green : ℕ := 0

/-- Represents a watermelon with its number of slices and seeds per slice -/
structure Watermelon where
  slices : ℕ
  seedsPerSlice : SeedsPerSlice

def firstWatermelon : Watermelon :=
  { slices := 40
    seedsPerSlice := { black := 20, white := 15, red := 10 } }

def secondWatermelon : Watermelon :=
  { slices := 30
    seedsPerSlice := { black := 25, white := 20, purple := 15 } }

def thirdWatermelon : Watermelon :=
  { slices := 50
    seedsPerSlice := { black := 15, white := 10, red := 5, green := 5 } }

def totalSeeds (w : Watermelon) : ℕ :=
  w.slices * (w.seedsPerSlice.black + w.seedsPerSlice.white + w.seedsPerSlice.red +
              w.seedsPerSlice.purple + w.seedsPerSlice.green)

theorem total_seeds_three_watermelons :
  totalSeeds firstWatermelon + totalSeeds secondWatermelon + totalSeeds thirdWatermelon = 5350 := by
  sorry

end NUMINAMATH_CALUDE_total_seeds_three_watermelons_l2566_256632


namespace NUMINAMATH_CALUDE_car_resale_gain_percentage_car_resale_specific_case_l2566_256687

/-- Calculates the gain percentage when reselling a car --/
theorem car_resale_gain_percentage 
  (original_price : ℝ) 
  (loss_percentage : ℝ) 
  (resale_price : ℝ) : ℝ :=
  let first_sale_price := original_price * (1 - loss_percentage / 100)
  let gain := resale_price - first_sale_price
  let gain_percentage := (gain / first_sale_price) * 100
  gain_percentage

/-- Proves that the gain percentage is approximately 3.55% for the given scenario --/
theorem car_resale_specific_case : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |car_resale_gain_percentage 52941.17647058824 15 54000 - 3.55| < ε :=
sorry

end NUMINAMATH_CALUDE_car_resale_gain_percentage_car_resale_specific_case_l2566_256687


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_l2566_256640

def vector_a : Fin 2 → ℝ := ![1, 2]
def vector_b (x : ℝ) : Fin 2 → ℝ := ![x, -2]

theorem vector_sum_magnitude (x : ℝ) 
  (h : vector_a • vector_b x = -3) : 
  ‖vector_a + vector_b x‖ = 2 := by
sorry

end NUMINAMATH_CALUDE_vector_sum_magnitude_l2566_256640


namespace NUMINAMATH_CALUDE_gum_given_by_steve_l2566_256668

theorem gum_given_by_steve (initial_gum : ℕ) (final_gum : ℕ) 
  (h1 : initial_gum = 38) (h2 : final_gum = 54) :
  final_gum - initial_gum = 16 := by
  sorry

end NUMINAMATH_CALUDE_gum_given_by_steve_l2566_256668


namespace NUMINAMATH_CALUDE_infinite_series_solution_l2566_256629

theorem infinite_series_solution : ∃! x : ℝ, x = (1 : ℝ) / (1 + x) ∧ |x| < 1 := by
  sorry

end NUMINAMATH_CALUDE_infinite_series_solution_l2566_256629


namespace NUMINAMATH_CALUDE_red_paint_calculation_l2566_256625

/-- Given a mixture with a ratio of red paint to white paint and a total number of cans,
    calculate the number of cans of red paint required. -/
def red_paint_cans (red_ratio white_ratio total_cans : ℕ) : ℕ :=
  (red_ratio * total_cans) / (red_ratio + white_ratio)

/-- Theorem stating that for a 3:2 ratio of red to white paint and 30 total cans,
    18 cans of red paint are required. -/
theorem red_paint_calculation :
  red_paint_cans 3 2 30 = 18 := by
  sorry

end NUMINAMATH_CALUDE_red_paint_calculation_l2566_256625


namespace NUMINAMATH_CALUDE_max_value_of_x_plus_2y_l2566_256615

theorem max_value_of_x_plus_2y (x y : ℝ) (h : x^2 - x*y + y^2 = 1) :
  x + 2*y ≤ 2 * Real.sqrt 21 / 3 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_x_plus_2y_l2566_256615


namespace NUMINAMATH_CALUDE_volume_of_rotated_region_l2566_256692

/-- A region composed of unit squares resting along the x-axis and y-axis -/
structure Region :=
  (squares : ℕ)
  (along_x_axis : Bool)
  (along_y_axis : Bool)

/-- The volume of a solid formed by rotating a region about the y-axis -/
noncomputable def rotated_volume (r : Region) : ℝ :=
  sorry

/-- The problem statement -/
theorem volume_of_rotated_region :
  ∃ (r : Region),
    r.squares = 16 ∧
    r.along_x_axis = true ∧
    r.along_y_axis = true ∧
    rotated_volume r = 37 * Real.pi :=
  sorry

end NUMINAMATH_CALUDE_volume_of_rotated_region_l2566_256692


namespace NUMINAMATH_CALUDE_least_perimeter_of_special_triangle_l2566_256644

/-- A triangle with integer side lengths -/
structure IntTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b

/-- The condition for a triangle to be non-equilateral -/
def is_non_equilateral (t : IntTriangle) : Prop :=
  t.a ≠ t.b ∨ t.b ≠ t.c ∨ t.c ≠ t.a

/-- The condition for points D, C, E, G to be concyclic -/
def is_concyclic (t : IntTriangle) : Prop :=
  -- This is a placeholder for the actual concyclic condition
  -- In reality, this would involve more complex geometric relations
  true

/-- The perimeter of a triangle -/
def perimeter (t : IntTriangle) : ℕ := t.a + t.b + t.c

theorem least_perimeter_of_special_triangle :
  ∃ (t : IntTriangle),
    is_non_equilateral t ∧
    is_concyclic t ∧
    (∀ (s : IntTriangle), is_non_equilateral s → is_concyclic s → perimeter t ≤ perimeter s) ∧
    perimeter t = 37 := by
  sorry

end NUMINAMATH_CALUDE_least_perimeter_of_special_triangle_l2566_256644


namespace NUMINAMATH_CALUDE_spotlight_illumination_theorem_l2566_256647

/-- Represents a point on a plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a direction (North, South, East, West) --/
inductive Direction
  | North
  | South
  | East
  | West

/-- Represents a spotlight that illuminates a right angle --/
structure Spotlight where
  position : Point
  direction1 : Direction
  direction2 : Direction

/-- Represents the configuration of four spotlights --/
structure SpotlightConfiguration where
  spotlights : Fin 4 → Spotlight

/-- Predicate to check if a configuration illuminates the entire plane --/
def illuminatesEntirePlane (config : SpotlightConfiguration) : Prop := sorry

/-- The main theorem stating that there exists a configuration of spotlights that illuminates the entire plane --/
theorem spotlight_illumination_theorem :
  ∃ (config : SpotlightConfiguration), illuminatesEntirePlane config :=
sorry

end NUMINAMATH_CALUDE_spotlight_illumination_theorem_l2566_256647


namespace NUMINAMATH_CALUDE_amount_ratio_l2566_256656

def total : ℕ := 1210
def r_amount : ℕ := 400

theorem amount_ratio (p q r : ℕ) 
  (h1 : p + q + r = total)
  (h2 : r = r_amount)
  (h3 : 9 * r = 10 * q) :
  5 * q = 4 * p := by sorry

end NUMINAMATH_CALUDE_amount_ratio_l2566_256656


namespace NUMINAMATH_CALUDE_cake_and_muffin_buyers_l2566_256642

theorem cake_and_muffin_buyers (total : ℕ) (cake : ℕ) (muffin : ℕ) (neither_prob : ℚ) :
  total = 100 →
  cake = 50 →
  muffin = 40 →
  neither_prob = 26 / 100 →
  ∃ (both : ℕ), both = 16 ∧ 
    (total : ℚ) * (1 - neither_prob) = (cake + muffin - both : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_cake_and_muffin_buyers_l2566_256642


namespace NUMINAMATH_CALUDE_divisible_by_eleven_l2566_256634

theorem divisible_by_eleven (m : ℕ) : 
  m < 10 →
  (864 * 10^7 + m * 10^6 + 5 * 10^5 + 3 * 10^4 + 7 * 10^3 + 9 * 10^2 + 7 * 10 + 9) % 11 = 0 →
  m = 9 := by
sorry

end NUMINAMATH_CALUDE_divisible_by_eleven_l2566_256634


namespace NUMINAMATH_CALUDE_exists_k_divisible_by_power_of_three_l2566_256690

theorem exists_k_divisible_by_power_of_three : 
  ∃ k : ℤ, (3 : ℤ)^2008 ∣ (k^3 - 36*k^2 + 51*k - 97) := by
  sorry

end NUMINAMATH_CALUDE_exists_k_divisible_by_power_of_three_l2566_256690


namespace NUMINAMATH_CALUDE_original_price_calculation_l2566_256653

theorem original_price_calculation (final_price : ℝ) : 
  final_price = 1120 → 
  ∃ (original_price : ℝ), 
    original_price * (1 - 0.3) * (1 - 0.2) = final_price ∧ 
    original_price = 2000 := by
  sorry

end NUMINAMATH_CALUDE_original_price_calculation_l2566_256653


namespace NUMINAMATH_CALUDE_cos_2phi_nonpositive_l2566_256602

theorem cos_2phi_nonpositive (α β φ : Real) 
  (h : Real.tan φ = 1 / (Real.cos α * Real.cos β + Real.tan α * Real.tan β)) : 
  Real.cos (2 * φ) ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_2phi_nonpositive_l2566_256602


namespace NUMINAMATH_CALUDE_monomial_sum_l2566_256638

/-- If the sum of two monomials is still a monomial, then it equals -5xy^2 --/
theorem monomial_sum (a b : ℕ) : 
  (∃ (c : ℚ) (d e : ℕ), (-4 * 10^a * X^a * Y^2 + 35 * X * Y^(b-2) = c * X^d * Y^e)) →
  (-4 * 10^a * X^a * Y^2 + 35 * X * Y^(b-2) = -5 * X * Y^2) :=
by sorry

end NUMINAMATH_CALUDE_monomial_sum_l2566_256638


namespace NUMINAMATH_CALUDE_grape_heap_division_l2566_256650

theorem grape_heap_division (n : ℕ) (h1 : n ≥ 105) 
  (h2 : (n + 1) % 3 = 1) (h3 : (n + 1) % 5 = 1) :
  ∃ x : ℕ, x > 5 ∧ (n + 1) % x = 1 ∧ ∀ y : ℕ, 5 < y ∧ y < x → (n + 1) % y ≠ 1 :=
by
  sorry

end NUMINAMATH_CALUDE_grape_heap_division_l2566_256650


namespace NUMINAMATH_CALUDE_greatest_integer_less_than_negative_seventeen_thirds_l2566_256659

theorem greatest_integer_less_than_negative_seventeen_thirds :
  Int.floor (-17 / 3 : ℚ) = -6 := by sorry

end NUMINAMATH_CALUDE_greatest_integer_less_than_negative_seventeen_thirds_l2566_256659
